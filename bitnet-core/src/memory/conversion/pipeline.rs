//! Conversion Pipeline with Memory Pooling
//!
//! This module implements a conversion pipeline that chains multiple conversion
//! operations while optimizing memory usage through intelligent pooling and reuse.

use crate::memory::conversion::{
    ConversionResult, ConversionError, ConversionContext, Converter, ConversionStrategy,
    ZeroCopyConverter, StreamingConverter, InPlaceConverter,
    config::ConversionConfig
};
use crate::memory::tensor::{BitNetTensor, BitNetDType};
use crate::memory::HybridMemoryPool;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// A conversion pipeline that chains multiple operations with memory optimization
pub struct ConversionPipeline {
    /// Pipeline configuration
    config: ConversionConfig,
    /// Memory pool for intermediate results
    pool: Arc<HybridMemoryPool>,
    /// Pipeline stages
    stages: Vec<PipelineStage>,
    /// Intermediate tensor cache
    cache: Arc<RwLock<TensorCache>>,
    /// Memory usage tracker
    memory_tracker: Arc<Mutex<MemoryTracker>>,
}

impl ConversionPipeline {
    /// Creates a new conversion pipeline
    pub fn new(config: ConversionConfig, pool: Arc<HybridMemoryPool>) -> ConversionResult<Self> {
        config.validate().map_err(|e| ConversionError::ConfigError { reason: e })?;

        Ok(Self {
            config,
            pool,
            stages: Vec::new(),
            cache: Arc::new(RwLock::new(TensorCache::new())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::new())),
        })
    }

    /// Adds a conversion stage to the pipeline
    pub fn add_stage(mut self, target_dtype: BitNetDType) -> Self {
        let stage = PipelineStage {
            target_dtype,
            strategy: ConversionStrategy::Auto,
            cache_intermediate: true,
        };
        self.stages.push(stage);
        self
    }

    /// Adds a conversion stage with specific strategy
    pub fn add_stage_with_strategy(
        mut self, 
        target_dtype: BitNetDType, 
        strategy: ConversionStrategy
    ) -> Self {
        let stage = PipelineStage {
            target_dtype,
            strategy,
            cache_intermediate: true,
        };
        self.stages.push(stage);
        self
    }

    /// Adds a conversion stage without caching intermediate results
    pub fn add_stage_no_cache(mut self, target_dtype: BitNetDType) -> Self {
        let stage = PipelineStage {
            target_dtype,
            strategy: ConversionStrategy::Auto,
            cache_intermediate: false,
        };
        self.stages.push(stage);
        self
    }

    /// Executes the pipeline on a single tensor
    pub fn execute(&self, input: &BitNetTensor) -> ConversionResult<BitNetTensor> {
        if self.stages.is_empty() {
            return Ok(input.clone());
        }

        #[cfg(feature = "tracing")]
        info!("Executing conversion pipeline with {} stages", self.stages.len());

        let mut current_tensor = input.clone();
        let mut stage_results = Vec::new();

        // Track memory usage
        {
            let mut tracker = self.memory_tracker.lock()
                .map_err(|_| ConversionError::InternalError {
                    reason: "Failed to acquire memory tracker lock".to_string()
                })?;
            tracker.start_pipeline(input.size_bytes());
        }

        // Execute each stage
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            #[cfg(feature = "tracing")]
            debug!("Executing pipeline stage {} -> {}", stage_idx, stage.target_dtype);

            // Check cache first
            if stage.cache_intermediate {
                if let Some(cached_result) = self.check_cache(&current_tensor, stage.target_dtype)? {
                    #[cfg(feature = "tracing")]
                    debug!("Using cached result for stage {}", stage_idx);
                    current_tensor = cached_result;
                    continue;
                }
            }

            // Execute conversion
            let converted = self.execute_stage(&current_tensor, stage, stage_idx)?;
            
            // Cache result if requested
            if stage.cache_intermediate {
                self.cache_result(&current_tensor, &converted)?;
            }

            // Track intermediate result
            stage_results.push(StageResult {
                stage_index: stage_idx,
                input_size: current_tensor.size_bytes(),
                output_size: converted.size_bytes(),
                strategy_used: stage.strategy,
            });

            current_tensor = converted;
        }

        // Update memory tracker
        {
            let mut tracker = self.memory_tracker.lock()
                .map_err(|_| ConversionError::InternalError {
                    reason: "Failed to acquire memory tracker lock".to_string()
                })?;
            tracker.finish_pipeline(current_tensor.size_bytes(), stage_results);
        }

        // Cleanup intermediate tensors if memory pressure is high
        self.cleanup_if_needed()?;

        #[cfg(feature = "tracing")]
        info!("Pipeline execution completed successfully");

        Ok(current_tensor)
    }

    /// Executes the pipeline on multiple tensors in batch
    pub fn execute_batch(&self, inputs: &[BitNetTensor]) -> ConversionResult<Vec<BitNetTensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "tracing")]
        info!("Executing pipeline batch with {} tensors", inputs.len());

        // For batch processing, we can optimize by grouping tensors with similar characteristics
        let mut results = Vec::with_capacity(inputs.len());

        // Process each tensor through the pipeline
        for (idx, input) in inputs.iter().enumerate() {
            #[cfg(feature = "tracing")]
            debug!("Processing batch tensor {} of {}", idx + 1, inputs.len());

            let result = self.execute(input)?;
            results.push(result);

            // Periodic cleanup during batch processing
            if idx % 10 == 9 {
                self.cleanup_if_needed()?;
            }
        }

        Ok(results)
    }

    /// Executes a single pipeline stage
    fn execute_stage(
        &self,
        input: &BitNetTensor,
        stage: &PipelineStage,
        stage_idx: usize,
    ) -> ConversionResult<BitNetTensor> {
        let source_dtype = input.dtype();
        let device = input.device();

        // Skip if already the target type
        if source_dtype == stage.target_dtype {
            return Ok(input.clone());
        }

        // Create conversion context
        let context = ConversionContext::new(
            source_dtype,
            stage.target_dtype,
            device.clone(),
            device.clone(),
            input.shape(),
        ).with_strategy(stage.strategy);

        // Select appropriate converter
        let converter = self.select_converter(&context)?;
        
        // Execute conversion
        converter.convert(input, &context, &self.pool)
            .map_err(|e| ConversionError::PipelineError {
                stage: stage_idx,
                reason: format!("Stage conversion failed: {}", e),
            })
    }

    /// Selects the appropriate converter for the given context
    fn select_converter(&self, context: &ConversionContext) -> ConversionResult<Box<dyn Converter + Send + Sync>> {
        let strategy = if context.strategy == ConversionStrategy::Auto {
            context.optimal_strategy()
        } else {
            context.strategy
        };

        match strategy {
            ConversionStrategy::ZeroCopy => {
                Ok(Box::new(ZeroCopyConverter::new()))
            }
            ConversionStrategy::InPlace => {
                Ok(Box::new(InPlaceConverter::new_lossy()))
            }
            ConversionStrategy::Streaming => {
                let converter = StreamingConverter::default()
                    .map_err(|e| ConversionError::InternalError {
                        reason: format!("Failed to create streaming converter: {}", e)
                    })?;
                Ok(Box::new(converter))
            }
            ConversionStrategy::Standard => {
                let converter = StreamingConverter::default()
                    .map_err(|e| ConversionError::InternalError {
                        reason: format!("Failed to create standard converter: {}", e)
                    })?;
                Ok(Box::new(converter))
            }
            ConversionStrategy::Auto => {
                unreachable!("Auto strategy should have been resolved")
            }
        }
    }

    /// Checks the cache for a previously computed result
    fn check_cache(
        &self,
        input: &BitNetTensor,
        target_dtype: BitNetDType,
    ) -> ConversionResult<Option<BitNetTensor>> {
        let mut cache = self.cache.write()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire cache write lock".to_string()
            })?;

        let key = CacheKey {
            source_id: input.id(),
            target_dtype,
        };

        Ok(cache.get(&key).cloned())
    }

    /// Caches a conversion result
    fn cache_result(
        &self,
        input: &BitNetTensor,
        output: &BitNetTensor,
    ) -> ConversionResult<()> {
        let mut cache = self.cache.write()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire cache write lock".to_string()
            })?;

        let key = CacheKey {
            source_id: input.id(),
            target_dtype: output.dtype(),
        };

        cache.insert(key, output.clone());
        Ok(())
    }

    /// Cleans up cache and intermediate results if memory pressure is high
    fn cleanup_if_needed(&self) -> ConversionResult<()> {
        let should_cleanup = {
            let tracker = self.memory_tracker.lock()
                .map_err(|_| ConversionError::InternalError {
                    reason: "Failed to acquire memory tracker lock".to_string()
                })?;
            tracker.should_cleanup()
        };

        if should_cleanup {
            #[cfg(feature = "tracing")]
            debug!("Performing pipeline cleanup due to memory pressure");

            let mut cache = self.cache.write()
                .map_err(|_| ConversionError::InternalError {
                    reason: "Failed to acquire cache write lock".to_string()
                })?;
            cache.cleanup_lru();
        }

        Ok(())
    }

    /// Returns pipeline statistics
    pub fn get_stats(&self) -> ConversionResult<PipelineStats> {
        let tracker = self.memory_tracker.lock()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire memory tracker lock".to_string()
            })?;

        let cache = self.cache.read()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire cache read lock".to_string()
            })?;

        Ok(PipelineStats {
            total_executions: tracker.total_executions,
            total_stages_executed: tracker.total_stages_executed,
            cache_hits: cache.hits,
            cache_misses: cache.misses,
            cache_size: cache.entries.len(),
            peak_memory_usage: tracker.peak_memory_usage,
            average_execution_time_ms: tracker.average_execution_time_ms(),
        })
    }

    /// Clears the pipeline cache
    pub fn clear_cache(&self) -> ConversionResult<()> {
        let mut cache = self.cache.write()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire cache write lock".to_string()
            })?;
        cache.clear();
        Ok(())
    }

    /// Optimizes the pipeline by reordering stages for better memory efficiency
    pub fn optimize(mut self) -> Self {
        // Sort stages to minimize memory usage
        // 1. Zero-copy conversions first
        // 2. In-place conversions next
        // 3. Other conversions last
        self.stages.sort_by_key(|stage| {
            match stage.strategy {
                ConversionStrategy::ZeroCopy => 0,
                ConversionStrategy::InPlace => 1,
                ConversionStrategy::Auto => 2,
                ConversionStrategy::Standard => 3,
                ConversionStrategy::Streaming => 4,
            }
        });

        self
    }
}

/// A single stage in the conversion pipeline
#[derive(Debug, Clone)]
struct PipelineStage {
    target_dtype: BitNetDType,
    strategy: ConversionStrategy,
    cache_intermediate: bool,
}

/// Cache key for memoizing conversion results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    source_id: u64,
    target_dtype: BitNetDType,
}

/// Cache for storing intermediate conversion results
#[derive(Debug)]
struct TensorCache {
    entries: std::collections::HashMap<CacheKey, BitNetTensor>,
    access_order: VecDeque<CacheKey>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

impl TensorCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            access_order: VecDeque::new(),
            max_size: 100, // Maximum number of cached tensors
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<&BitNetTensor> {
        if let Some(tensor) = self.entries.get(key) {
            // Move to front of access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_front(key.clone());
            self.hits += 1;
            Some(tensor)
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: CacheKey, tensor: BitNetTensor) {
        // Remove if already exists
        if self.entries.contains_key(&key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }

        // Add to front
        self.entries.insert(key.clone(), tensor);
        self.access_order.push_front(key);

        // Evict if over capacity
        while self.entries.len() > self.max_size {
            if let Some(old_key) = self.access_order.pop_back() {
                self.entries.remove(&old_key);
            }
        }
    }

    fn cleanup_lru(&mut self) {
        // Remove half of the least recently used entries
        let remove_count = self.entries.len() / 2;
        for _ in 0..remove_count {
            if let Some(old_key) = self.access_order.pop_back() {
                self.entries.remove(&old_key);
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }
}

/// Tracks memory usage and performance metrics for the pipeline
#[derive(Debug)]
struct MemoryTracker {
    total_executions: u64,
    total_stages_executed: u64,
    peak_memory_usage: usize,
    current_memory_usage: usize,
    execution_times: VecDeque<u64>,
    max_execution_history: usize,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            total_executions: 0,
            total_stages_executed: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            execution_times: VecDeque::new(),
            max_execution_history: 100,
        }
    }

    fn start_pipeline(&mut self, input_size: usize) {
        self.total_executions += 1;
        self.current_memory_usage = input_size;
        self.peak_memory_usage = self.peak_memory_usage.max(input_size);
    }

    fn finish_pipeline(&mut self, output_size: usize, stage_results: Vec<StageResult>) {
        self.total_stages_executed += stage_results.len() as u64;
        self.current_memory_usage = output_size;
        
        // Calculate peak memory during pipeline execution
        let max_intermediate = stage_results.iter()
            .map(|r| r.input_size.max(r.output_size))
            .max()
            .unwrap_or(0);
        self.peak_memory_usage = self.peak_memory_usage.max(max_intermediate);
    }

    fn should_cleanup(&self) -> bool {
        // Cleanup if current memory usage is high
        self.current_memory_usage > 500 * 1024 * 1024 // 500MB threshold
    }

    fn average_execution_time_ms(&self) -> f64 {
        if self.execution_times.is_empty() {
            0.0
        } else {
            let sum: u64 = self.execution_times.iter().sum();
            sum as f64 / self.execution_times.len() as f64
        }
    }
}

/// Result of executing a pipeline stage
#[derive(Debug, Clone)]
struct StageResult {
    stage_index: usize,
    input_size: usize,
    output_size: usize,
    strategy_used: ConversionStrategy,
}

/// Statistics about pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_executions: u64,
    pub total_stages_executed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_size: usize,
    pub peak_memory_usage: usize,
    pub average_execution_time_ms: f64,
}

impl PipelineStats {
    /// Returns the cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_accesses as f64
        }
    }

    /// Returns the average stages per execution
    pub fn average_stages_per_execution(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.total_stages_executed as f64 / self.total_executions as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_pipeline_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool).unwrap();
        assert_eq!(pipeline.stages.len(), 0);
    }

    #[test]
    fn test_pipeline_stage_addition() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool).unwrap()
            .add_stage(BitNetDType::F16)
            .add_stage(BitNetDType::I8);
        
        assert_eq!(pipeline.stages.len(), 2);
        assert_eq!(pipeline.stages[0].target_dtype, BitNetDType::F16);
        assert_eq!(pipeline.stages[1].target_dtype, BitNetDType::I8);
    }

    #[test]
    fn test_empty_pipeline_execution() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool.clone()).unwrap();

        let input = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = pipeline.execute(&input).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_single_stage_pipeline() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool.clone()).unwrap()
            .add_stage(BitNetDType::F16);

        let input = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = pipeline.execute(&input).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_multi_stage_pipeline() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool.clone()).unwrap()
            .add_stage(BitNetDType::F16)
            .add_stage(BitNetDType::I8);

        let input = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = pipeline.execute(&input).unwrap();

        assert_eq!(result.dtype(), BitNetDType::I8);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_batch_pipeline_execution() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool.clone()).unwrap()
            .add_stage(BitNetDType::F16);

        let inputs = vec![
            BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
        ];

        let results = pipeline.execute_batch(&inputs).unwrap();
        assert_eq!(results.len(), 2);
        
        for result in &results {
            assert_eq!(result.dtype(), BitNetDType::F16);
        }
    }

    #[test]
    fn test_pipeline_stats() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool.clone()).unwrap()
            .add_stage(BitNetDType::F16);

        let input = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let _result = pipeline.execute(&input).unwrap();

        let stats = pipeline.get_stats().unwrap();
        assert_eq!(stats.total_executions, 1);
        assert_eq!(stats.total_stages_executed, 1);
    }

    #[test]
    fn test_cache_functionality() {
        let mut cache = TensorCache::new();
        let pool = HybridMemoryPool::new().unwrap();
        let device = get_cpu_device();

        let key = CacheKey {
            source_id: 1,
            target_dtype: BitNetDType::F16,
        };

        let tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F16, &device, &pool).unwrap();

        // Test miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.misses, 1);

        // Test insert and hit
        cache.insert(key.clone(), tensor);
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_pipeline_optimization() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = ConversionConfig::default();
        let pipeline = ConversionPipeline::new(config, pool).unwrap()
            .add_stage_with_strategy(BitNetDType::F16, ConversionStrategy::Streaming)
            .add_stage_with_strategy(BitNetDType::I8, ConversionStrategy::ZeroCopy)
            .add_stage_with_strategy(BitNetDType::I4, ConversionStrategy::InPlace)
            .optimize();

        // After optimization, zero-copy should come first
        assert_eq!(pipeline.stages[0].strategy, ConversionStrategy::ZeroCopy);
        assert_eq!(pipeline.stages[1].strategy, ConversionStrategy::InPlace);
        assert_eq!(pipeline.stages[2].strategy, ConversionStrategy::Streaming);
    }
}