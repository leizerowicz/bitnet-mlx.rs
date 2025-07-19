//! Conversion Engine
//!
//! This module provides the main conversion engine that orchestrates all
//! conversion operations, automatically selecting the optimal strategy
//! and managing resources efficiently.

use crate::memory::conversion::{
    ConversionResult, ConversionError, ConversionContext, Converter, ConversionStrategy,
    config::ConversionConfig, ZeroCopyConverter, StreamingConverter, InPlaceConverter, 
    BatchConverter, ConversionPipeline, ConversionStats, ConversionEvent, ConversionQuality
};
use crate::memory::tensor::{BitNetTensor, BitNetDType};
use crate::memory::HybridMemoryPool;
use candle_core::Device;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Main conversion engine that orchestrates all conversion operations
pub struct ConversionEngine {
    /// Engine configuration
    config: ConversionConfig,
    /// Memory pool for allocations
    pool: Arc<HybridMemoryPool>,
    /// Zero-copy converter
    zero_copy_converter: ZeroCopyConverter,
    /// Streaming converter
    streaming_converter: StreamingConverter,
    /// In-place converter
    in_place_converter: InPlaceConverter,
    /// Batch converter
    batch_converter: BatchConverter,
    /// Statistics collector
    stats: Arc<ConversionStats>,
}

impl ConversionEngine {
    /// Creates a new conversion engine
    pub fn new(config: ConversionConfig, pool: Arc<HybridMemoryPool>) -> ConversionResult<Self> {
        config.validate().map_err(|e| ConversionError::ConfigError { reason: e })?;

        let streaming_converter = StreamingConverter::new(config.streaming.clone())?;
        let batch_converter = BatchConverter::new(config.batch.clone())?;
        let stats = Arc::new(ConversionStats::new(1000)); // Keep last 1000 events

        Ok(Self {
            config,
            pool,
            zero_copy_converter: ZeroCopyConverter::new(),
            streaming_converter,
            in_place_converter: InPlaceConverter::new_lossy(),
            batch_converter,
            stats,
        })
    }

    /// Converts a single tensor to the target data type
    pub fn convert(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
    ) -> ConversionResult<BitNetTensor> {
        self.convert_with_quality(source, target_dtype, self.config.default_quality)
    }

    /// Converts a tensor with specified quality settings
    pub fn convert_with_quality(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
        quality: ConversionQuality,
    ) -> ConversionResult<BitNetTensor> {
        let start_time = Instant::now();
        let source_dtype = source.dtype();
        let device = source.device();
        let shape = source.shape();
        let element_count = source.element_count();
        let input_size = source.size_bytes();

        #[cfg(feature = "tracing")]
        info!("Converting tensor from {} to {} with quality {:?}", source_dtype, target_dtype, quality);

        // Create conversion event for tracking
        let mut event = ConversionEvent::new(
            source_dtype,
            target_dtype,
            self.config.default_strategy,
            quality,
            &device,
            input_size,
            0, // Will be updated after conversion
            element_count,
        );

        // Skip conversion if already the target type
        if source_dtype == target_dtype {
            let duration = start_time.elapsed();
            event.output_size_bytes = input_size;
            let completed_event = event.complete_success(duration, 0, input_size);
            self.stats.record_event(completed_event);
            return Ok(source.clone());
        }

        // Create conversion context
        let context = ConversionContext::new(
            source_dtype,
            target_dtype,
            device.clone(),
            device.clone(),
            shape,
        ).with_strategy(self.config.default_strategy)
         .with_quality(quality);

        // Select optimal converter and strategy
        let (converter, strategy) = self.select_optimal_converter(&context)?;
        event.strategy = strategy;

        // Perform conversion
        let result = match converter.convert(source, &context, &self.pool) {
            Ok(tensor) => {
                let duration = start_time.elapsed();
                event.output_size_bytes = tensor.size_bytes();
                let completed_event = event.complete_success(duration, tensor.size_bytes(), tensor.size_bytes());
                self.stats.record_event(completed_event);

                #[cfg(feature = "tracing")]
                info!("Conversion completed successfully in {:?} using strategy {:?}", duration, strategy);

                Ok(tensor)
            }
            Err(e) => {
                let duration = start_time.elapsed();
                let completed_event = event.complete_failure(duration, e.to_string());
                self.stats.record_event(completed_event);

                #[cfg(feature = "tracing")]
                warn!("Conversion failed after {:?}: {}", duration, e);

                Err(e)
            }
        };

        result
    }

    /// Converts multiple tensors to the same target type
    pub fn batch_convert(
        &self,
        sources: &[BitNetTensor],
        target_dtype: BitNetDType,
    ) -> ConversionResult<Vec<BitNetTensor>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "tracing")]
        info!("Batch converting {} tensors to {}", sources.len(), target_dtype);

        let start_time = Instant::now();
        let result = self.batch_converter.batch_convert(sources, target_dtype, &self.pool);

        match result {
            Ok(tensors) => {
                let duration = start_time.elapsed();
                #[cfg(feature = "tracing")]
                info!("Batch conversion completed successfully in {:?}", duration);

                // Record events for each tensor
                for (source, target) in sources.iter().zip(tensors.iter()) {
                    let event = ConversionEvent::new(
                        source.dtype(),
                        target_dtype,
                        ConversionStrategy::Auto, // Batch converter selects strategy
                        self.config.default_quality,
                        &source.device(),
                        source.size_bytes(),
                        target.size_bytes(),
                        source.element_count(),
                    ).complete_success(duration / sources.len() as u32, target.size_bytes(), target.size_bytes());
                    
                    self.stats.record_event(event);
                }

                Ok(tensors)
            }
            Err(e) => {
                let duration = start_time.elapsed();
                #[cfg(feature = "tracing")]
                warn!("Batch conversion failed after {:?}: {}", duration, e);

                // Record failure events
                for source in sources {
                    let event = ConversionEvent::new(
                        source.dtype(),
                        target_dtype,
                        ConversionStrategy::Auto,
                        self.config.default_quality,
                        &source.device(),
                        source.size_bytes(),
                        0,
                        source.element_count(),
                    ).complete_failure(duration / sources.len() as u32, e.to_string());
                    
                    self.stats.record_event(event);
                }

                Err(e)
            }
        }
    }

    /// Converts multiple tensors with different target types
    pub fn batch_convert_mixed(
        &self,
        conversions: &[(BitNetTensor, BitNetDType)],
    ) -> ConversionResult<Vec<BitNetTensor>> {
        if conversions.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "tracing")]
        info!("Mixed batch converting {} tensors", conversions.len());

        let start_time = Instant::now();
        let result = self.batch_converter.batch_convert_mixed(conversions, &self.pool);

        match result {
            Ok(tensors) => {
                let duration = start_time.elapsed();
                #[cfg(feature = "tracing")]
                info!("Mixed batch conversion completed successfully in {:?}", duration);

                // Record events for each conversion
                for ((source, target_dtype), target) in conversions.iter().zip(tensors.iter()) {
                    let event = ConversionEvent::new(
                        source.dtype(),
                        *target_dtype,
                        ConversionStrategy::Auto,
                        self.config.default_quality,
                        &source.device(),
                        source.size_bytes(),
                        target.size_bytes(),
                        source.element_count(),
                    ).complete_success(duration / conversions.len() as u32, target.size_bytes(), target.size_bytes());
                    
                    self.stats.record_event(event);
                }

                Ok(tensors)
            }
            Err(e) => {
                let duration = start_time.elapsed();
                #[cfg(feature = "tracing")]
                warn!("Mixed batch conversion failed after {:?}: {}", duration, e);

                // Record failure events
                for (source, target_dtype) in conversions {
                    let event = ConversionEvent::new(
                        source.dtype(),
                        *target_dtype,
                        ConversionStrategy::Auto,
                        self.config.default_quality,
                        &source.device(),
                        source.size_bytes(),
                        0,
                        source.element_count(),
                    ).complete_failure(duration / conversions.len() as u32, e.to_string());
                    
                    self.stats.record_event(event);
                }

                Err(e)
            }
        }
    }

    /// Creates a conversion pipeline for chaining multiple conversions
    pub fn create_pipeline(&self) -> ConversionResult<ConversionPipeline> {
        ConversionPipeline::new(self.config.clone(), self.pool.clone())
    }

    /// Performs zero-copy conversion (if possible)
    pub fn zero_copy_convert(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
    ) -> ConversionResult<BitNetTensor> {
        let context = ConversionContext::new(
            source.dtype(),
            target_dtype,
            source.device(),
            source.device(),
            source.shape(),
        ).with_strategy(ConversionStrategy::ZeroCopy);

        if !self.zero_copy_converter.supports(&context) {
            return Err(ConversionError::UnsupportedConversion {
                from: source.dtype(),
                to: target_dtype,
            });
        }

        self.zero_copy_converter.convert(source, &context, &self.pool)
    }

    /// Performs streaming conversion for large tensors
    pub fn streaming_convert(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
        chunk_size: usize,
    ) -> ConversionResult<BitNetTensor> {
        let mut config = self.config.streaming.clone();
        config.chunk_size = chunk_size;
        
        let streaming_converter = StreamingConverter::new(config)?;
        let context = ConversionContext::new(
            source.dtype(),
            target_dtype,
            source.device(),
            source.device(),
            source.shape(),
        ).with_strategy(ConversionStrategy::Streaming);

        streaming_converter.convert(source, &context, &self.pool)
    }

    /// Performs in-place conversion (modifies the source tensor)
    pub fn in_place_convert(
        &self,
        tensor: &mut BitNetTensor,
        target_dtype: BitNetDType,
    ) -> ConversionResult<()> {
        if !self.in_place_converter.is_in_place_compatible(tensor.dtype(), target_dtype) {
            return Err(ConversionError::UnsupportedConversion {
                from: tensor.dtype(),
                to: target_dtype,
            });
        }

        self.in_place_converter.convert_in_place(tensor, target_dtype)
    }

    /// Selects the optimal converter and strategy for the given context
    fn select_optimal_converter(
        &self,
        context: &ConversionContext,
    ) -> ConversionResult<(Box<dyn Converter + Send + Sync>, ConversionStrategy)> {
        let strategy = if context.strategy == ConversionStrategy::Auto {
            context.optimal_strategy()
        } else {
            context.strategy
        };

        let converter: Box<dyn Converter + Send + Sync> = match strategy {
            ConversionStrategy::ZeroCopy => {
                if self.zero_copy_converter.supports(context) {
                    Box::new(ZeroCopyConverter::new())
                } else {
                    // Fallback to in-place if zero-copy not supported
                    if self.in_place_converter.supports(context) {
                        return Ok((Box::new(InPlaceConverter::new_lossy()), ConversionStrategy::InPlace));
                    } else {
                        return Ok((Box::new(self.streaming_converter.clone()), ConversionStrategy::Streaming));
                    }
                }
            }
            ConversionStrategy::InPlace => {
                if self.in_place_converter.supports(context) {
                    Box::new(InPlaceConverter::new_lossy())
                } else {
                    // Fallback to streaming
                    return Ok((Box::new(self.streaming_converter.clone()), ConversionStrategy::Streaming));
                }
            }
            ConversionStrategy::Streaming => {
                Box::new(self.streaming_converter.clone())
            }
            ConversionStrategy::Standard => {
                Box::new(self.streaming_converter.clone())
            }
            ConversionStrategy::Auto => {
                unreachable!("Auto strategy should have been resolved")
            }
        };

        Ok((converter, strategy))
    }

    /// Returns conversion statistics
    pub fn get_stats(&self) -> crate::memory::conversion::ConversionMetrics {
        self.stats.generate_metrics()
    }

    /// Clears conversion statistics
    pub fn clear_stats(&self) {
        self.stats.clear();
    }

    /// Returns the engine configuration
    pub fn config(&self) -> &ConversionConfig {
        &self.config
    }

    /// Returns the memory pool
    pub fn pool(&self) -> &Arc<HybridMemoryPool> {
        &self.pool
    }

    /// Estimates the time required for a conversion
    pub fn estimate_conversion_time(
        &self,
        source_dtype: BitNetDType,
        target_dtype: BitNetDType,
        shape: &[usize],
        device: &Device,
    ) -> u64 {
        let context = ConversionContext::new(
            source_dtype,
            target_dtype,
            device.clone(),
            device.clone(),
            shape.to_vec(),
        );

        match self.select_optimal_converter(&context) {
            Ok((converter, _)) => converter.estimate_time_ms(&context),
            Err(_) => u64::MAX, // Unsupported conversion
        }
    }

    /// Checks if a conversion is supported
    pub fn is_conversion_supported(
        &self,
        source_dtype: BitNetDType,
        target_dtype: BitNetDType,
        device: &Device,
    ) -> bool {
        let context = ConversionContext::new(
            source_dtype,
            target_dtype,
            device.clone(),
            device.clone(),
            vec![1], // Dummy shape for checking support
        );

        self.zero_copy_converter.supports(&context) ||
        self.in_place_converter.supports(&context) ||
        self.streaming_converter.supports(&context)
    }

    /// Returns information about the optimal strategy for a conversion
    pub fn get_optimal_strategy_info(
        &self,
        source_dtype: BitNetDType,
        target_dtype: BitNetDType,
        shape: &[usize],
        device: &Device,
    ) -> ConversionStrategyInfo {
        let context = ConversionContext::new(
            source_dtype,
            target_dtype,
            device.clone(),
            device.clone(),
            shape.to_vec(),
        );

        let optimal_strategy = context.optimal_strategy();
        let memory_overhead = context.memory_overhead_bytes();
        let estimated_time = self.estimate_conversion_time(source_dtype, target_dtype, shape, device);
        let is_supported = self.is_conversion_supported(source_dtype, target_dtype, device);

        ConversionStrategyInfo {
            strategy: optimal_strategy,
            estimated_time_ms: estimated_time,
            memory_overhead_bytes: memory_overhead,
            is_supported,
            is_zero_copy: optimal_strategy == ConversionStrategy::ZeroCopy,
            is_in_place: optimal_strategy == ConversionStrategy::InPlace,
            compression_ratio: Self::calculate_compression_ratio(source_dtype, target_dtype),
        }
    }

    /// Calculates the compression ratio for a data type conversion
    fn calculate_compression_ratio(source_dtype: BitNetDType, target_dtype: BitNetDType) -> f64 {
        let source_bits = source_dtype.bits_per_element() as f64;
        let target_bits = target_dtype.bits_per_element() as f64;
        
        if target_bits == 0.0 {
            1.0
        } else {
            source_bits / target_bits
        }
    }
}

// We need to implement Clone for StreamingConverter to use it in the engine
impl Clone for StreamingConverter {
    fn clone(&self) -> Self {
        // Create a new instance with the same config
        StreamingConverter::new(self.config.clone()).unwrap()
    }
}

/// Information about the optimal conversion strategy
#[derive(Debug, Clone)]
pub struct ConversionStrategyInfo {
    pub strategy: ConversionStrategy,
    pub estimated_time_ms: u64,
    pub memory_overhead_bytes: usize,
    pub is_supported: bool,
    pub is_zero_copy: bool,
    pub is_in_place: bool,
    pub compression_ratio: f64,
}

impl ConversionStrategyInfo {
    /// Returns a human-readable description of the strategy
    pub fn description(&self) -> String {
        if !self.is_supported {
            return "Conversion not supported".to_string();
        }

        let strategy_desc = match self.strategy {
            ConversionStrategy::ZeroCopy => "Zero-copy (no data copying)",
            ConversionStrategy::InPlace => "In-place (modifies existing buffer)",
            ConversionStrategy::Streaming => "Streaming (processes in chunks)",
            ConversionStrategy::Standard => "Standard (allocates new buffer)",
            ConversionStrategy::Auto => "Auto-selected",
        };

        format!(
            "{} - Est. time: {}ms, Memory overhead: {} bytes, Compression: {:.2}x",
            strategy_desc,
            self.estimated_time_ms,
            self.memory_overhead_bytes,
            self.compression_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_engine_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool).unwrap();
        
        assert!(engine.is_conversion_supported(BitNetDType::F32, BitNetDType::F16, &get_cpu_device()));
    }

    #[test]
    fn test_single_tensor_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let source = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = engine.convert(&source, BitNetDType::F16).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_batch_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let sources = vec![
            BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
        ];

        let results = engine.batch_convert(&sources, BitNetDType::F16).unwrap();
        assert_eq!(results.len(), 2);
        
        for result in &results {
            assert_eq!(result.dtype(), BitNetDType::F16);
        }
    }

    #[test]
    fn test_zero_copy_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let source = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = engine.zero_copy_convert(&source, BitNetDType::F32).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_strategy_info() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool).unwrap();

        let info = engine.get_optimal_strategy_info(
            BitNetDType::F32,
            BitNetDType::F16,
            &[100, 100],
            &device,
        );

        assert!(info.is_supported);
        assert!(info.estimated_time_ms > 0);
        assert_eq!(info.compression_ratio, 2.0); // F32 to F16 is 2x compression
    }

    #[test]
    fn test_conversion_statistics() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let source = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let _result = engine.convert(&source, BitNetDType::F16).unwrap();

        let stats = engine.get_stats();
        assert_eq!(stats.total_conversions, 1);
        assert_eq!(stats.successful_conversions, 1);
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_unsupported_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let source = BitNetTensor::ones(&[2, 2], BitNetDType::F16, &device, &pool).unwrap();
        let result = engine.zero_copy_convert(&source, BitNetDType::I8);

        assert!(result.is_err());
    }
}