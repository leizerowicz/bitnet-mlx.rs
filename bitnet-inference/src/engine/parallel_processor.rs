//! Parallel inference processing pipeline for high-throughput batch processing.

use crate::{InferenceError, Result};
use bitnet_core::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};

/// Task for inference processing.
#[derive(Debug)]
pub struct InferenceTask {
    /// Unique task identifier
    pub id: usize,
    /// Input tensor for processing
    pub tensor: Tensor,
    /// Timestamp when task was created
    pub timestamp: Instant,
    /// Original position in batch (for result ordering)
    pub original_index: usize,
}

/// Result of inference processing.
#[derive(Debug)]
pub struct InferenceResult {
    /// Task identifier
    pub task_id: usize,
    /// Output tensor
    pub output_tensor: Tensor,
    /// Processing duration
    pub processing_time: Duration,
    /// Original position in batch
    pub original_index: usize,
    /// Worker that processed this task
    pub worker_id: usize,
}

/// Worker statistics for monitoring.
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker identifier
    pub worker_id: usize,
    /// Number of tasks processed
    pub tasks_processed: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per task
    pub average_processing_time: Duration,
    /// Current status
    pub is_active: bool,
}

/// Statistics for the parallel processor.
#[derive(Debug, Clone)]
pub struct ParallelProcessorStats {
    /// Number of active workers
    pub active_workers: usize,
    /// Total tasks processed
    pub total_tasks_processed: usize,
    /// Tasks currently queued
    pub queued_tasks: usize,
    /// Average throughput (tasks per second)
    pub throughput: f64,
    /// Worker-specific statistics
    pub worker_stats: Vec<WorkerStats>,
}

/// Configuration for parallel processing.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub worker_count: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Task timeout duration
    pub task_timeout: Duration,
    /// Enable worker load balancing
    pub load_balancing: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get().max(1),
            max_queue_size: 1000,
            task_timeout: Duration::from_secs(30),
            load_balancing: true,
        }
    }
}

/// Parallel inference processor for high-throughput batch processing.
pub struct ParallelInferenceProcessor {
    /// Number of workers
    worker_count: usize,
    /// Task queue sender
    task_sender: mpsc::Sender<InferenceTask>,
    /// Result collector receiver
    result_receiver: mpsc::Receiver<InferenceResult>,
    /// Semaphore for controlling concurrency
    semaphore: Arc<Semaphore>,
    /// Task counter for unique IDs
    task_counter: Arc<AtomicUsize>,
    /// Worker statistics
    worker_stats: Arc<tokio::sync::RwLock<Vec<Arc<tokio::sync::RwLock<WorkerStats>>>>>,
    /// Configuration
    config: ParallelConfig,
}

impl ParallelInferenceProcessor {
    /// Create a new parallel processor with default configuration.
    pub async fn new() -> Result<Self> {
        Self::with_config(ParallelConfig::default()).await
    }

    /// Create a new parallel processor with custom configuration.
    pub async fn with_config(config: ParallelConfig) -> Result<Self> {
        let (task_sender, task_receiver) = mpsc::channel(config.max_queue_size);
        let (result_sender, result_receiver) = mpsc::channel(config.max_queue_size * 2);
        let semaphore = Arc::new(Semaphore::new(config.worker_count));
        let task_counter = Arc::new(AtomicUsize::new(0));

        // Initialize worker statistics
        let worker_stats = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        {
            let mut stats = worker_stats.write().await;
            for worker_id in 0..config.worker_count {
                stats.push(Arc::new(tokio::sync::RwLock::new(WorkerStats {
                    worker_id,
                    tasks_processed: 0,
                    total_processing_time: Duration::default(),
                    average_processing_time: Duration::default(),
                    is_active: false,
                })));
            }
        }

        // Clone necessary data for worker tasks
        let task_receiver = Arc::new(tokio::sync::Mutex::new(task_receiver));
        
        // Spawn worker tasks
        for worker_id in 0..config.worker_count {
            let task_rx = Arc::clone(&task_receiver);
            let result_tx = result_sender.clone();
            let worker_semaphore = Arc::clone(&semaphore);
            let worker_stats_ref = Arc::clone(&worker_stats);
            let task_timeout = config.task_timeout;

            tokio::spawn(async move {
                Self::worker_loop(
                    worker_id,
                    task_rx,
                    result_tx,
                    worker_semaphore,
                    worker_stats_ref,
                    task_timeout,
                ).await;
            });
        }

        Ok(Self {
            worker_count: config.worker_count,
            task_sender,
            result_receiver,
            semaphore,
            task_counter,
            worker_stats,
            config,
        })
    }

    /// Process a batch of tensors in parallel.
    pub async fn process_batch_parallel(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = inputs.len();
        let mut tasks_sent = 0;

        // Send tasks to workers
        for (index, input) in inputs.into_iter().enumerate() {
            let task_id = self.task_counter.fetch_add(1, Ordering::Relaxed);
            let task = InferenceTask {
                id: task_id,
                tensor: input,
                timestamp: Instant::now(),
                original_index: index,
            };

            // Try to send task with timeout
            match tokio::time::timeout(
                Duration::from_secs(5),
                self.task_sender.send(task)
            ).await {
                Ok(Ok(())) => tasks_sent += 1,
                Ok(Err(_)) => {
                    return Err(InferenceError::batch_processing(
                        "Task channel closed unexpectedly".to_string()
                    ));
                }
                Err(_) => {
                    return Err(InferenceError::batch_processing(
                        "Timeout sending task to workers".to_string()
                    ));
                }
            }
        }

        // Collect results
        let mut results = Vec::with_capacity(batch_size);
        let mut collected = 0;

        while collected < tasks_sent {
            match tokio::time::timeout(
                self.config.task_timeout,
                self.result_receiver.recv()
            ).await {
                Ok(Some(result)) => {
                    results.push(result);
                    collected += 1;
                }
                Ok(None) => {
                    return Err(InferenceError::batch_processing(
                        "Result channel closed unexpectedly".to_string()
                    ));
                }
                Err(_) => {
                    return Err(InferenceError::batch_processing(
                        format!("Timeout waiting for results. Collected {}/{}", collected, tasks_sent)
                    ));
                }
            }
        }

        // Sort results by original order
        results.sort_by_key(|r| r.original_index);
        Ok(results.into_iter().map(|r| r.output_tensor).collect())
    }

    /// Process a batch with streaming results.
    pub async fn process_batch_streaming(
        &mut self,
        inputs: Vec<Tensor>,
    ) -> Result<mpsc::Receiver<InferenceResult>> {
        let (streaming_sender, streaming_receiver) = mpsc::channel(inputs.len());
        
        // Process batch and forward results to streaming channel
        let batch_size = inputs.len();
        for (index, input) in inputs.into_iter().enumerate() {
            let task_id = self.task_counter.fetch_add(1, Ordering::Relaxed);
            let task = InferenceTask {
                id: task_id,
                tensor: input,
                timestamp: Instant::now(),
                original_index: index,
            };

            self.task_sender.send(task).await.map_err(|_| {
                InferenceError::batch_processing("Failed to send task".to_string())
            })?;
        }

        // Spawn task to collect and forward results
        let mut result_receiver = std::mem::replace(
            &mut self.result_receiver,
            mpsc::channel(self.config.max_queue_size * 2).1
        );

        tokio::spawn(async move {
            let mut collected = 0;
            while collected < batch_size {
                if let Some(result) = result_receiver.recv().await {
                    let _ = streaming_sender.send(result).await;
                    collected += 1;
                } else {
                    break;
                }
            }
        });

        Ok(streaming_receiver)
    }

    /// Worker loop for processing inference tasks.
    async fn worker_loop(
        worker_id: usize,
        task_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<InferenceTask>>>,
        result_sender: mpsc::Sender<InferenceResult>,
        semaphore: Arc<Semaphore>,
        worker_stats: Arc<tokio::sync::RwLock<Vec<Arc<tokio::sync::RwLock<WorkerStats>>>>>,
        task_timeout: Duration,
    ) {
        loop {
            // Acquire semaphore permit
            let _permit = match semaphore.acquire().await {
                Ok(permit) => permit,
                Err(_) => break, // Semaphore closed
            };

            // Mark worker as active
            {
                let stats_vec = worker_stats.read().await;
                if let Some(worker_stat) = stats_vec.get(worker_id) {
                    if let Ok(mut stat) = worker_stat.try_write() {
                        stat.is_active = true;
                    }
                }
            }

            // Try to receive a task
            let task = {
                let mut receiver = task_receiver.lock().await;
                receiver.recv().await
            };

            match task {
                Some(task) => {
                    let start_time = Instant::now();
                    
                    // Process the task with timeout
                    let result = match tokio::time::timeout(
                        task_timeout,
                        Self::process_single_task(task, worker_id)
                    ).await {
                        Ok(result) => result,
                        Err(_) => {
                            // Timeout occurred
                            continue;
                        }
                    };

                    let processing_time = start_time.elapsed();

                    // Update worker statistics
                    {
                        let stats_vec = worker_stats.read().await;
                        if let Some(worker_stat) = stats_vec.get(worker_id) {
                            if let Ok(mut stat) = worker_stat.try_write() {
                                stat.tasks_processed += 1;
                                stat.total_processing_time += processing_time;
                                stat.average_processing_time = stat.total_processing_time / stat.tasks_processed as u32;
                            }
                        }
                    }

                    // Send result
                    if result_sender.send(result).await.is_err() {
                        break; // Channel closed
                    }
                }
                None => break, // Channel closed
            }

            // Mark worker as inactive
            {
                let stats_vec = worker_stats.read().await;
                if let Some(worker_stat) = stats_vec.get(worker_id) {
                    if let Ok(mut stat) = worker_stat.try_write() {
                        stat.is_active = false;
                    }
                }
            }
        }
    }

    /// Process a single inference task.
    async fn process_single_task(task: InferenceTask, worker_id: usize) -> InferenceResult {
        let start_time = Instant::now();
        
        // TODO: Replace with actual inference logic
        // For now, just clone the tensor as a placeholder
        let output_tensor = task.tensor.clone();
        
        let processing_time = start_time.elapsed();

        InferenceResult {
            task_id: task.id,
            output_tensor,
            processing_time,
            original_index: task.original_index,
            worker_id,
        }
    }

    /// Get the number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Get current processor statistics.
    pub async fn get_stats(&self) -> ParallelProcessorStats {
        let worker_stats = {
            let stats_vec = self.worker_stats.read().await;
            let mut worker_stats = Vec::new();
            
            for stat_ref in stats_vec.iter() {
                let stat = stat_ref.read().await;
                worker_stats.push(stat.clone());
            }
            worker_stats
        };

        let total_tasks_processed = worker_stats.iter()
            .map(|s| s.tasks_processed)
            .sum();

        let active_workers = worker_stats.iter()
            .filter(|s| s.is_active)
            .count();

        let total_processing_time: Duration = worker_stats.iter()
            .map(|s| s.total_processing_time)
            .sum();

        let throughput = if total_processing_time.as_secs_f64() > 0.0 {
            total_tasks_processed as f64 / total_processing_time.as_secs_f64()
        } else {
            0.0
        };

        ParallelProcessorStats {
            active_workers,
            total_tasks_processed,
            queued_tasks: 0, // Would need additional tracking for accurate queue size
            throughput,
            worker_stats,
        }
    }

    /// Gracefully shutdown the processor.
    pub async fn shutdown(self) -> Result<()> {
        // Close the task sender to signal workers to stop
        drop(self.task_sender);

        // Wait a bit for workers to finish current tasks
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::{Device, DType, Tensor};

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.worker_count > 0);
        assert!(config.max_queue_size > 0);
        assert!(config.task_timeout.as_secs() > 0);
        assert!(config.load_balancing);
    }

    #[tokio::test]
    async fn test_parallel_processor_creation() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        assert!(processor.worker_count > 0);
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.worker_stats.len(), processor.worker_count);
    }

    #[tokio::test]
    async fn test_parallel_processor_with_config() {
        let config = ParallelConfig {
            worker_count: 2,
            max_queue_size: 100,
            task_timeout: Duration::from_secs(10),
            load_balancing: false,
        };
        
        let processor = ParallelInferenceProcessor::with_config(config).await.unwrap();
        assert_eq!(processor.worker_count, 2);
    }

    #[tokio::test]
    async fn test_parallel_batch_processing() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let device = Device::Cpu;
        
        // Create test tensors
        let inputs = vec![
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            Tensor::ones((2, 3), DType::F32, &device).unwrap(),
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            Tensor::ones((2, 3), DType::F32, &device).unwrap(),
        ];
        
        let results = processor.process_batch_parallel(inputs.clone()).await.unwrap();
        assert_eq!(results.len(), inputs.len());
    }

    #[tokio::test]
    async fn test_empty_batch_processing() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let results = processor.process_batch_parallel(Vec::new()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_batch_processing() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let device = Device::Cpu;
        
        // Create test tensors
        let inputs = vec![
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            Tensor::ones((2, 3), DType::F32, &device).unwrap(),
        ];
        
        let mut stream = processor.process_batch_streaming(inputs.clone()).await.unwrap();
        
        let mut results = Vec::new();
        while let Some(result) = stream.recv().await {
            results.push(result);
            if results.len() == inputs.len() {
                break;
            }
        }
        
        assert_eq!(results.len(), inputs.len());
    }

    #[tokio::test]
    async fn test_processor_stats() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        let stats = processor.get_stats().await;
        
        assert!(stats.worker_stats.len() > 0);
        assert_eq!(stats.total_tasks_processed, 0); // No tasks processed yet
    }

    #[tokio::test]
    async fn test_processor_shutdown() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        let result = processor.shutdown().await;
        assert!(result.is_ok());
    }
}
