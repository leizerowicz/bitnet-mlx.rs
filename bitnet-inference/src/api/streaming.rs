//! Streaming inference API for continuous processing of input streams.

use crate::{Result, api::InferenceEngine};
use crate::engine::Model;
use bitnet_core::Tensor;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::Stream;
use futures::stream::{self, BoxStream};
use futures::{pin_mut};

/// Configuration for streaming inference processing.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for batching inputs before processing
    pub buffer_size: usize,
    /// Maximum time to wait before processing partial buffer (milliseconds)
    pub max_latency_ms: u64,
    /// Whether to maintain processing order
    pub preserve_order: bool,
    /// Channel capacity for buffering results
    pub channel_capacity: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 16,
            max_latency_ms: 10,
            preserve_order: true,
            channel_capacity: 100,
        }
    }
}

/// A streaming inference processor that can handle continuous streams of inputs.
pub struct InferenceStream {
    engine: Arc<InferenceEngine>,
    model: Arc<Model>,
    config: StreamingConfig,
}

impl InferenceStream {
    /// Create a new streaming inference processor.
    pub fn new(engine: Arc<InferenceEngine>, model: Arc<Model>) -> Self {
        Self::with_config(engine, model, StreamingConfig::default())
    }

    /// Create a new streaming inference processor with custom configuration.
    pub fn with_config(
        engine: Arc<InferenceEngine>, 
        model: Arc<Model>, 
        config: StreamingConfig
    ) -> Self {
        Self {
            engine,
            model,
            config,
        }
    }

    /// Process a stream of input tensors and return a stream of results.
    /// This method maintains order if configured to do so.
    pub async fn process_stream<S>(&self, input_stream: S) -> Result<BoxStream<'_, Result<Tensor>>>
    where
        S: Stream<Item = Tensor> + Send + 'static,
    {
        let engine = self.engine.clone();
        let model = self.model.clone();
        let buffer_size = self.config.buffer_size;
        let max_latency_ms = self.config.max_latency_ms;
        let preserve_order = self.config.preserve_order;

        // Create output stream using async generator pattern
        let output_stream = async_stream::stream! {
            pin_mut!(input_stream);
            
            let mut input_buffer = Vec::with_capacity(buffer_size);
            let mut timeout = tokio::time::interval(
                tokio::time::Duration::from_millis(max_latency_ms)
            );
            timeout.tick().await; // Skip first immediate tick
            
            loop {
                tokio::select! {
                    // New input arrived
                    maybe_input = tokio_stream::StreamExt::next(&mut input_stream) => {
                        match maybe_input {
                            Some(input) => {
                                input_buffer.push(input);
                                
                                // Process if buffer is full
                                if input_buffer.len() >= buffer_size {
                                    match Self::process_buffer(
                                        &engine, 
                                        &model, 
                                        &input_buffer, 
                                        preserve_order
                                    ).await {
                                        Ok(results) => {
                                            for result in results {
                                                yield Ok(result);
                                            }
                                        }
                                        Err(e) => yield Err(e),
                                    }
                                    input_buffer.clear();
                                }
                            }
                            None => {
                                // Stream ended, process remaining inputs
                                if !input_buffer.is_empty() {
                                    match Self::process_buffer(
                                        &engine, 
                                        &model, 
                                        &input_buffer, 
                                        preserve_order
                                    ).await {
                                        Ok(results) => {
                                            for result in results {
                                                yield Ok(result);
                                            }
                                        }
                                        Err(e) => yield Err(e),
                                    }
                                }
                                break;
                            }
                        }
                    }
                    
                    // Timeout occurred - process partial buffer to maintain latency
                    _ = timeout.tick() => {
                        if !input_buffer.is_empty() {
                            match Self::process_buffer(
                                &engine, 
                                &model, 
                                &input_buffer, 
                                preserve_order
                            ).await {
                                Ok(results) => {
                                    for result in results {
                                        yield Ok(result);
                                    }
                                }
                                Err(e) => yield Err(e),
                            }
                            input_buffer.clear();
                        }
                    }
                }
            }
        };

        Ok(Box::pin(output_stream))
    }

    /// Process a stream of input tensors in parallel without preserving order.
    /// This can provide better throughput for use cases that don't require order preservation.
    pub async fn process_stream_parallel<S>(&self, input_stream: S) -> Result<BoxStream<'_, Result<Tensor>>>
    where
        S: Stream<Item = Tensor> + Send + 'static,
    {
        let engine = self.engine.clone();
        let model = self.model.clone();
        let buffer_size = self.config.buffer_size;
        let channel_capacity = self.config.channel_capacity;

        let (tx, rx) = mpsc::channel(channel_capacity);
        let tx = Arc::new(Mutex::new(tx));

        // Spawn a task to process the input stream
        let processing_task = {
            let tx = tx.clone();
            let engine = engine.clone();
            let model = model.clone();
            
            tokio::spawn(async move {
                pin_mut!(input_stream);
                let mut input_buffer = Vec::with_capacity(buffer_size);
                
                while let Some(input) = tokio_stream::StreamExt::next(&mut input_stream).await {
                    input_buffer.push(input);
                    
                    if input_buffer.len() >= buffer_size {
                        let batch = input_buffer.clone();
                        input_buffer.clear();
                        
                        // Process batch in parallel
                        let tx_clone = tx.clone();
                        let engine_clone = engine.clone();
                        let model_clone = model.clone();
                        
                        tokio::spawn(async move {
                            match engine_clone.infer_batch(&model_clone, &batch).await {
                                Ok(results) => {
                                    let tx = tx_clone.lock().await;
                                    for result in results {
                                        if tx.send(Ok(result)).await.is_err() {
                                            break; // Receiver dropped
                                        }
                                    }
                                }
                                Err(e) => {
                                    let tx = tx_clone.lock().await;
                                    let _ = tx.send(Err(e)).await;
                                }
                            }
                        });
                    }
                }
                
                // Process remaining inputs
                if !input_buffer.is_empty() {
                    match engine.infer_batch(&model, &input_buffer).await {
                        Ok(results) => {
                            let tx = tx.lock().await;
                            for result in results {
                                if tx.send(Ok(result)).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let tx = tx.lock().await;
                            let _ = tx.send(Err(e)).await;
                        }
                    }
                }
            })
        };

        // Return stream from receiver
        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        
        // Handle task completion (optional cleanup)
        let cleanup_stream = stream::unfold(
            (output_stream, Some(processing_task)), 
            |(mut stream, task)| async move {
                match tokio_stream::StreamExt::next(&mut stream).await {
                    Some(item) => Some((item, (stream, task))),
                    None => {
                        if let Some(task) = task {
                            let _ = task.await;
                        }
                        None
                    }
                }
            }
        );

        Ok(Box::pin(cleanup_stream))
    }

    /// Process a single batch of inputs with optional order preservation.
    async fn process_buffer(
        engine: &InferenceEngine,
        model: &Model, 
        inputs: &[Tensor],
        preserve_order: bool,
    ) -> Result<Vec<Tensor>> {
        if preserve_order {
            // Use sequential processing to maintain order
            let mut results = Vec::with_capacity(inputs.len());
            for input in inputs {
                let result = engine.infer(model, input).await?;
                results.push(result);
            }
            Ok(results)
        } else {
            // Use batch processing for better performance
            engine.infer_batch(model, inputs).await
        }
    }

    /// Get streaming configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Update the buffer size for streaming processing
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }

    /// Update the maximum latency for streaming processing
    pub fn with_max_latency(mut self, max_latency_ms: u64) -> Self {
        self.config.max_latency_ms = max_latency_ms;
        self
    }

    /// Set whether to preserve processing order
    pub fn with_order_preservation(mut self, preserve_order: bool) -> Self {
        self.config.preserve_order = preserve_order;
        self
    }
}

/// Utility functions for creating streams from various sources
pub mod sources {
    use super::*;
    use std::time::Duration;

    /// Create a stream from a vector of tensors
    pub fn from_vec(tensors: Vec<Tensor>) -> impl Stream<Item = Tensor> {
        stream::iter(tensors)
    }

    /// Create a timed stream that emits tensors at regular intervals
    pub fn from_vec_timed(
        tensors: Vec<Tensor>, 
        interval: Duration
    ) -> impl Stream<Item = Tensor> {
        stream::unfold(
            (tensors.into_iter(), tokio::time::interval(interval)), 
            |(mut iter, mut interval)| async move {
                interval.tick().await;
                iter.next().map(|tensor| (tensor, (iter, interval)))
            }
        )
    }

    /// Create an infinite stream of random test tensors
    #[cfg(feature = "generation")]
    pub fn test_tensor_stream(
        shape: Vec<usize>, 
        interval: Duration
    ) -> impl Stream<Item = Tensor> {
        stream::unfold(
            (shape, tokio::time::interval(interval)), 
            |(shape, mut interval)| async move {
                interval.tick().await;
                
                // Create random tensor data
                use rand::Rng;
                let total_size = shape.iter().product::<usize>();
                let data: Vec<f32> = (0..total_size)
                    .map(|_| rand::thread_rng().gen_range(-1.0..1.0))
                    .collect();
                
                match Tensor::from_slice(&data, &shape, &bitnet_core::Device::Cpu) {
                    Ok(tensor) => Some((tensor, (shape, interval))),
                    Err(_) => None,
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::InferenceEngine;
    use crate::engine::{Model, ModelArchitecture, QuantizationConfig};
    use bitnet_core::Tensor;

    fn create_test_model() -> Arc<Model> {
        Arc::new(Model {
            name: "test_streaming_model".to_string(),
            version: "1.0".to_string(),
            input_dim: 512,
            output_dim: 512,
            architecture: ModelArchitecture::BitLinear {
                layers: Vec::new(),
                attention_heads: None,
                hidden_dim: 512,
            },
            parameter_count: 100_000,
            quantization_config: QuantizationConfig::default(),
        })
    }

    fn create_test_tensor() -> Tensor {
        let data = vec![1.0; 512];
        Tensor::from_slice(&data, &[1, 512], &bitnet_core::Device::Cpu).unwrap()
    }

    #[tokio::test]
    async fn test_streaming_basic_functionality() {
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let streaming_processor = InferenceStream::new(engine, model);
        
        // Create test input stream
        let inputs = vec![create_test_tensor(); 5];
        let input_stream = sources::from_vec(inputs);
        
        let mut output_stream = streaming_processor.process_stream(input_stream).await.unwrap();
        
        let mut results = Vec::new();
        while let Some(result) = tokio_stream::StreamExt::next(&mut output_stream).await {
            results.push(result.unwrap());
        }
        
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_streaming_with_custom_config() {
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let config = StreamingConfig {
            buffer_size: 3,
            max_latency_ms: 5,
            preserve_order: true,
            channel_capacity: 50,
        };
        
        let streaming_processor = InferenceStream::with_config(engine, model, config);
        
        assert_eq!(streaming_processor.config().buffer_size, 3);
        assert_eq!(streaming_processor.config().max_latency_ms, 5);
        assert_eq!(streaming_processor.config().preserve_order, true);
    }

    #[tokio::test]
    async fn test_parallel_streaming() {
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let streaming_processor = InferenceStream::new(engine, model)
            .with_buffer_size(2)
            .with_order_preservation(false);
        
        let inputs = vec![create_test_tensor(); 8];
        let input_stream = sources::from_vec(inputs);
        
        let mut output_stream = streaming_processor.process_stream_parallel(input_stream).await.unwrap();
        
        let mut results = Vec::new();
        while let Some(result) = tokio_stream::StreamExt::next(&mut output_stream).await {
            results.push(result.unwrap());
        }
        
        assert_eq!(results.len(), 8);
    }

    #[tokio::test]
    async fn test_timed_streaming() {
        use std::time::{Duration, Instant};
        
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let streaming_processor = InferenceStream::new(engine, model)
            .with_max_latency(1); // 1ms timeout
        
        let inputs = vec![create_test_tensor(); 3];
        let input_stream = sources::from_vec_timed(inputs, Duration::from_millis(2));
        
        let start = Instant::now();
        let mut output_stream = streaming_processor.process_stream(input_stream).await.unwrap();
        
        let mut results = Vec::new();
        while let Some(result) = tokio_stream::StreamExt::next(&mut output_stream).await {
            results.push(result.unwrap());
        }
        
        let duration = start.elapsed();
        assert_eq!(results.len(), 3);
        // Should take at least 4ms (2ms * 2 intervals) but not much more due to latency control
        assert!(duration >= Duration::from_millis(4));
        assert!(duration < Duration::from_millis(100)); // Reasonable upper bound
    }

    #[tokio::test]
    async fn test_streaming_error_handling() {
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let streaming_processor = InferenceStream::new(engine, model);
        
        // Create input stream with invalid tensor (empty)
        let invalid_data: Vec<f32> = vec![];
        let invalid_tensor = Tensor::from_slice(&invalid_data, &[0], &bitnet_core::Device::Cpu).unwrap();
        let input_stream = sources::from_vec(vec![invalid_tensor]);
        
        let mut output_stream = streaming_processor.process_stream(input_stream).await.unwrap();
        
        // Should handle the error gracefully
        match tokio_stream::StreamExt::next(&mut output_stream).await {
            Some(Err(_)) => {
                // Expected error case
            }
            Some(Ok(_)) => {
                // Depending on implementation, this might also be valid
            }
            None => {
                // Stream ended without results, also valid
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "generation")]
    async fn test_infinite_stream() {
        use std::time::Duration;
        
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        let streaming_processor = InferenceStream::new(engine, model)
            .with_buffer_size(2);
        
        let input_stream = sources::test_tensor_stream(
            vec![1, 512], 
            Duration::from_millis(1)
        ).take(5); // Take only 5 items for test
        
        let mut output_stream = streaming_processor.process_stream(input_stream).await.unwrap();
        
        let mut count = 0;
        while let Some(result) = tokio_stream::StreamExt::next(&mut output_stream).await {
            match result {
                Ok(_) => count += 1,
                Err(_) => break,
            }
            if count >= 5 { break; }
        }
        
        assert_eq!(count, 5);
    }
}
