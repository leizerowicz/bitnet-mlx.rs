//! Model caching implementation with LRU eviction and memory management.

use crate::{Result, InferenceError};
use crate::engine::model_loader::{LoadedModel, ModelMetadata};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use bincode;

/// Execution plan for optimized model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Optimized layer execution order
    pub layers: Vec<LayerExecution>,
    /// Memory layout optimization strategy
    pub memory_layout: MemoryLayout,
    /// Operator fusion opportunities
    pub operator_fusion: Vec<FusionGroup>,
    /// Estimated memory requirements in bytes
    pub estimated_memory: usize,
}

/// Individual layer execution details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerExecution {
    /// Layer identifier
    pub id: usize,
    /// Layer type and configuration
    pub layer_type: LayerType,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
    /// Device placement (CPU, GPU, etc.)
    pub device_placement: DevicePlacement,
}

/// Memory layout optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Sequential layout (default)
    Sequential,
    /// Cache-optimized layout for better locality
    CacheOptimized,
    /// Memory pooled layout for reduced allocations
    Pooled { pool_size: usize },
}

/// Operator fusion group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    /// Layers that can be fused together
    pub fused_layers: Vec<usize>,
    /// Type of fusion
    pub fusion_type: FusionType,
    /// Expected performance improvement
    pub performance_gain: f32,
}

/// Types of operator fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionType {
    /// Element-wise operations fusion
    ElementWise,
    /// Matrix multiplication + bias addition
    MatMulBias,
    /// Activation function fusion
    Activation,
    /// Custom fusion pattern
    Custom(String),
}

/// Tensor specification for execution planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String, // TODO: Use proper DType when available
    /// Memory layout
    pub layout: TensorLayout,
}

/// Tensor memory layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorLayout {
    /// Contiguous layout
    Contiguous,
    /// Strided layout with custom strides
    Strided(Vec<usize>),
}

/// Device placement for layer execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePlacement {
    /// CPU execution
    CPU,
    /// GPU execution (Metal/CUDA)
    GPU { device_id: usize },
    /// Mixed CPU/GPU execution
    Mixed,
}

/// Layer type for execution planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    /// BitLinear layer
    BitLinear { 
        input_dim: usize,
        output_dim: usize,
        quantization_bits: u8,
    },
    /// Regular linear layer
    Linear {
        input_dim: usize,
        output_dim: usize,
    },
    /// Activation function
    Activation(String),
    /// Normalization layer
    Normalization(String),
    /// Custom layer
    Custom(String),
}

/// A cached model with additional metadata and execution plan.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// The loaded model
    pub model: LoadedModel,
    /// Optimized weights for fast inference
    pub optimized_weights: Vec<u8>,
    /// Execution plan for optimized inference
    pub execution_plan: ExecutionPlan,
    /// Size of the model in memory (bytes)
    pub memory_size: usize,
    /// Number of times this model has been accessed
    pub access_count: u64,
    /// Timestamp of last access
    pub last_accessed: std::time::Instant,
    /// Serialization cache for optimized loading
    pub serialized_cache: Option<Vec<u8>>,
}

impl CachedModel {
    /// Create a new cached model.
    pub fn new(model: LoadedModel) -> Self {
        let memory_size = Self::calculate_memory_size(&model);
        let optimized_weights = Self::optimize_weights(&model);
        let execution_plan = Self::create_execution_plan(&model);
        
        Self {
            model,
            optimized_weights,
            execution_plan,
            memory_size,
            access_count: 0,
            last_accessed: std::time::Instant::now(),
            serialized_cache: None,
        }
    }

    /// Create cached model from loaded model with execution plan
    pub fn from_loaded(model: LoadedModel) -> Self {
        Self::new(model)
    }

    /// Update access statistics.
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::Instant::now();
    }

    /// Get memory size of this cached model
    pub fn memory_size(&self) -> usize {
        self.memory_size
    }

    /// Serialize the model to bytes for persistent caching
    pub fn serialize(&mut self) -> Result<&[u8]> {
        if self.serialized_cache.is_none() {
            let serialized = bincode::serialize(&SerializableModel {
                metadata: &self.model.metadata,
                optimized_weights: &self.optimized_weights,
                execution_plan: &self.execution_plan,
            })?;
            self.serialized_cache = Some(serialized);
        }
        
        Ok(self.serialized_cache.as_ref().unwrap())
    }

    /// Optimize weights for the specific model architecture
    fn optimize_weights(model: &LoadedModel) -> Vec<u8> {
        // TODO: Implement weight optimization based on model architecture
        // For now, just compress the weights using a simple strategy
        let mut optimized = Vec::new();
        
        for (layer_id, weights) in &model.weights.layer_weights {
            // Simple optimization: pack weights more efficiently
            optimized.extend_from_slice(&layer_id.to_le_bytes());
            optimized.extend_from_slice(&(weights.len() as u32).to_le_bytes());
            optimized.extend_from_slice(weights);
        }
        
        optimized
    }

    /// Create execution plan based on model architecture
    fn create_execution_plan(model: &LoadedModel) -> ExecutionPlan {
        let mut layers = Vec::new();
        let mut estimated_memory = 0;

        // Create layer execution plans
        for (i, layer) in model.architecture.layers.iter().enumerate() {
            let layer_execution = LayerExecution {
                id: i,
                layer_type: Self::convert_layer_type(layer),
                inputs: vec![TensorSpec {
                    shape: model.metadata.input_shape.clone(),
                    dtype: "f32".to_string(),
                    layout: TensorLayout::Contiguous,
                }],
                outputs: vec![TensorSpec {
                    shape: model.metadata.output_shape.clone(),
                    dtype: "f32".to_string(),
                    layout: TensorLayout::Contiguous,
                }],
                device_placement: DevicePlacement::CPU, // Default to CPU, can be optimized later
            };
            
            estimated_memory += Self::estimate_layer_memory(&layer_execution);
            layers.push(layer_execution);
        }

        ExecutionPlan {
            layers,
            memory_layout: MemoryLayout::Sequential,
            operator_fusion: Self::identify_fusion_opportunities(&layers),
            estimated_memory,
        }
    }

    /// Convert model layer to execution layer type
    fn convert_layer_type(layer: &crate::engine::model_loader::LayerType) -> LayerType {
        match layer {
            crate::engine::model_loader::LayerType::Dense { units, .. } => {
                LayerType::Linear {
                    input_dim: *units, // Simplified
                    output_dim: *units,
                }
            }
            crate::engine::model_loader::LayerType::BitLinear { 
                input_dim, output_dim, .. 
            } => {
                LayerType::BitLinear {
                    input_dim: *input_dim,
                    output_dim: *output_dim,
                    quantization_bits: 1, // Default for BitNet
                }
            }
            crate::engine::model_loader::LayerType::Quantization { bits, .. } => {
                LayerType::BitLinear {
                    input_dim: 512, // Default
                    output_dim: 512,
                    quantization_bits: *bits,
                }
            }
        }
    }

    /// Estimate memory usage for a layer
    fn estimate_layer_memory(layer: &LayerExecution) -> usize {
        let input_size: usize = layer.inputs.iter()
            .map(|spec| spec.shape.iter().product::<usize>())
            .sum();
        let output_size: usize = layer.outputs.iter()
            .map(|spec| spec.shape.iter().product::<usize>())
            .sum();
        
        (input_size + output_size) * 4 // Assuming f32
    }

    /// Identify operator fusion opportunities
    fn identify_fusion_opportunities(layers: &[LayerExecution]) -> Vec<FusionGroup> {
        let mut fusion_groups = Vec::new();
        
        // Simple fusion: look for MatMul + Bias patterns
        for i in 0..(layers.len().saturating_sub(1)) {
            if matches!(layers[i].layer_type, LayerType::Linear { .. }) &&
               matches!(layers[i + 1].layer_type, LayerType::Activation(_)) {
                fusion_groups.push(FusionGroup {
                    fused_layers: vec![i, i + 1],
                    fusion_type: FusionType::MatMulBias,
                    performance_gain: 0.15, // 15% estimated gain
                });
            }
        }
        
        fusion_groups
    }

    /// Calculate the approximate memory size of a loaded model.
    fn calculate_memory_size(model: &LoadedModel) -> usize {
        // Base size for metadata and architecture
        let mut size = std::mem::size_of::<LoadedModel>();
        
        // Add weight data size
        size += model.weights.total_size;
        
        // Add some overhead for internal structures
        size += 1024; // 1KB overhead
        
        // Add execution plan overhead (estimated)
        size += 4096; // 4KB for execution plan
        
        size
    }
}

/// Serializable version of cached model for persistent storage
#[derive(Serialize, Deserialize)]
struct SerializableModel<'a> {
    metadata: &'a ModelMetadata,
    optimized_weights: &'a [u8],
    execution_plan: &'a ExecutionPlan,
}

/// High-performance model cache with LRU eviction.
pub struct ModelCache {
    /// LRU cache for models
    cache: Arc<Mutex<LruCache<String, CachedModel>>>,
    /// Current memory usage
    current_memory: Arc<Mutex<usize>>,
    /// Maximum memory allowed
    max_memory: usize,
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Statistics for cache performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions due to memory pressure
    pub evictions: u64,
    /// Total memory allocated
    pub total_memory_allocated: u64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
}

impl ModelCache {
    /// Create a new model cache with specified capacity and memory limit.
    pub fn new(capacity: usize, max_memory: usize) -> Self {
        let cache_size = NonZeroUsize::new(capacity)
            .expect("Cache capacity must be greater than 0");
            
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            current_memory: Arc::new(Mutex::new(0)),
            max_memory,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get a model from cache or load it using the provided loader function.
    pub fn get_or_load<F>(&self, key: &str, loader: F) -> Result<CachedModel>
    where
        F: FnOnce() -> Result<LoadedModel>,
    {
        // Try to get from cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(mut cached_model) = cache.get_mut(key) {
                cached_model.mark_accessed();
                self.record_hit();
                return Ok(cached_model.clone());
            }
        }

        // Cache miss - load the model
        self.record_miss();
        let loaded_model = loader()?;
        let mut cached_model = CachedModel::new(loaded_model);
        
        // Ensure we have enough memory
        self.ensure_memory_capacity(cached_model.memory_size)?;
        
        // Add to cache
        cached_model.mark_accessed();
        let result = cached_model.clone();
        
        {
            let mut cache = self.cache.lock().unwrap();
            let mut current_memory = self.current_memory.lock().unwrap();
            
            cache.put(key.to_string(), cached_model.clone());
            *current_memory += cached_model.memory_size;
            
            // Update peak memory usage
            let mut stats = self.stats.lock().unwrap();
            stats.peak_memory_usage = stats.peak_memory_usage.max(*current_memory);
        }
        
        Ok(result)
    }

    /// Remove a model from the cache.
    pub fn remove(&self, key: &str) -> Option<CachedModel> {
        let mut cache = self.cache.lock().unwrap();
        let mut current_memory = self.current_memory.lock().unwrap();
        
        if let Some(cached_model) = cache.pop(key) {
            *current_memory = current_memory.saturating_sub(cached_model.memory_size);
            Some(cached_model)
        } else {
            None
        }
    }

    /// Clear all models from the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let mut current_memory = self.current_memory.lock().unwrap();
        
        cache.clear();
        *current_memory = 0;
    }

    /// Get current memory usage in bytes.
    pub fn current_memory_usage(&self) -> usize {
        *self.current_memory.lock().unwrap()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.hits + stats.misses;
        
        if total == 0 {
            0.0
        } else {
            (stats.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get list of currently cached model keys.
    pub fn cached_keys(&self) -> Vec<String> {
        let cache = self.cache.lock().unwrap();
        cache.iter().map(|(key, _)| key.clone()).collect()
    }

    /// Ensure we have enough memory capacity for a new model.
    fn ensure_memory_capacity(&self, required: usize) -> Result<()> {
        let mut current_memory = self.current_memory.lock().unwrap();
        
        while *current_memory + required > self.max_memory {
            // Need to evict something
            let evicted = {
                let mut cache = self.cache.lock().unwrap();
                cache.pop_lru()
            };
            
            match evicted {
                Some((_, cached_model)) => {
                    *current_memory = current_memory.saturating_sub(cached_model.memory_size);
                    self.record_eviction();
                    
                    tracing::debug!(
                        "Evicted model from cache, freed {} bytes", 
                        cached_model.memory_size
                    );
                },
                None => {
                    // Cache is empty but we still don't have enough memory
                    return Err(InferenceError::memory(
                        format!(
                            "Cannot allocate {} bytes: exceeds maximum memory limit of {} bytes",
                            required,
                            self.max_memory
                        )
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// Record a cache hit.
    fn record_hit(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.hits += 1;
    }

    /// Record a cache miss.
    fn record_miss(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
    }

    /// Record a cache eviction.
    fn record_eviction(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.evictions += 1;
    }
}

impl CacheStats {
    /// Get total number of cache accesses.
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }

    /// Get cache hit rate as a percentage.
    pub fn hit_rate_percent(&self) -> f64 {
        let total = self.total_accesses();
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Check if cache performance is good.
    pub fn is_performing_well(&self) -> bool {
        let total = self.total_accesses();
        if total < 10 {
            return true; // Not enough data to judge
        }
        
        // Consider performance good if hit rate > 80%
        self.hit_rate_percent() > 80.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model_loader::*;
    use std::collections::HashMap;

    fn create_test_model(name: &str, size_mb: usize) -> LoadedModel {
        let weights_size = size_mb * 1024 * 1024; // Convert MB to bytes
        let mut layer_weights = HashMap::new();
        layer_weights.insert(0, vec![0u8; weights_size]);

        LoadedModel {
            metadata: ModelMetadata {
                name: name.to_string(),
                version: "1.0".to_string(),
                architecture: "test".to_string(),
                parameter_count: 1000,
                quantization_bits: 1,
                input_shape: vec![1, 512],
                output_shape: vec![1, 1000],
                extra: HashMap::new(),
            },
            architecture: ModelArchitecture {
                layers: vec![],
                execution_order: vec![],
            },
            weights: ModelWeights {
                layer_weights,
                total_size: weights_size,
            },
        }
    }

    #[test]
    fn test_cache_basic_operations() {
        let cache = ModelCache::new(2, 100 * 1024 * 1024); // 100MB limit
        
        // Test cache miss and load
        let model1 = cache.get_or_load("model1", || Ok(create_test_model("model1", 10))).unwrap();
        assert_eq!(model1.model.metadata.name, "model1");
        
        // Test cache hit
        let model1_again = cache.get_or_load("model1", || panic!("Should not be called")).unwrap();
        assert_eq!(model1_again.model.metadata.name, "model1");
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ModelCache::new(10, 25 * 1024 * 1024); // 25MB limit
        
        // Load a 20MB model
        let _model1 = cache.get_or_load("model1", || Ok(create_test_model("model1", 20))).unwrap();
        
        // Try to load a 10MB model - should cause eviction
        let _model2 = cache.get_or_load("model2", || Ok(create_test_model("model2", 10))).unwrap();
        
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        
        // Model1 should have been evicted
        let keys = cache.cached_keys();
        assert!(!keys.contains(&"model1".to_string()));
        assert!(keys.contains(&"model2".to_string()));
    }

    #[test]
    fn test_cache_memory_tracking() {
        let cache = ModelCache::new(5, 100 * 1024 * 1024);
        
        assert_eq!(cache.current_memory_usage(), 0);
        
        let _model = cache.get_or_load("test", || Ok(create_test_model("test", 10))).unwrap();
        
        // Should have some memory usage now (at least 10MB + overhead)
        assert!(cache.current_memory_usage() > 10 * 1024 * 1024);
    }
}
