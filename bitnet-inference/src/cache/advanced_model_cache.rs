//! Advanced model caching implementation with serialization and execution planning.

use crate::{Result, InferenceError};
use crate::engine::model_loader::{LoadedModel, ModelMetadata, LayerType, LayerDefinition, LayerParameters};
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
    pub layer_type: ExecutionLayerType,
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
pub enum ExecutionLayerType {
    /// BitLinear layer
    BitLinear { 
        input_dim: usize,
        output_dim: usize,
        weight_bits: u8,
        activation_bits: u8,
    },
    /// RMS Normalization layer
    RMSNorm {
        eps: f32,
        normalized_shape: Vec<usize>,
    },
    /// SwiGLU activation
    SwiGLU {
        hidden_dim: usize,
    },
    /// Embedding layer
    Embedding {
        vocab_size: usize,
        embedding_dim: usize,
    },
    /// Output projection layer
    OutputProjection {
        vocab_size: usize,
    },
}

/// A cached model with execution plan and optimized weights.
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
    /// Create a new cached model with execution plan.
    pub fn new(model: LoadedModel) -> Self {
        let optimized_weights = Self::optimize_weights(&model);
        let execution_plan = Self::create_execution_plan(&model);
        let memory_size = Self::calculate_memory_size(&model, &optimized_weights, &execution_plan);
        
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

    /// Create cached model from loaded model (alias for new)
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
                metadata: self.model.metadata.clone(),
                optimized_weights: self.optimized_weights.clone(),
                execution_plan: self.execution_plan.clone(),
            }).map_err(|e| InferenceError::serialization(&format!("Failed to serialize model: {}", e)))?;
            self.serialized_cache = Some(serialized);
        }
        
        Ok(self.serialized_cache.as_ref().unwrap())
    }

    /// Deserialize cached model from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let serializable: SerializableModel = bincode::deserialize(bytes)
            .map_err(|e| InferenceError::serialization(&format!("Failed to deserialize model: {}", e)))?;
        
        // Reconstruct the full model (simplified - in practice would need full weight reconstruction)
        let model = LoadedModel {
            metadata: serializable.metadata.clone(),
            architecture: crate::engine::model_loader::ModelArchitecture {
                layers: Vec::new(), // Would need to reconstruct from execution plan
                execution_order: Vec::new(),
            },
            weights: crate::engine::model_loader::ModelWeights {
                layer_weights: HashMap::new(), // Would need to reconstruct from optimized weights
                total_size: serializable.optimized_weights.len(),
            },
        };

        Ok(Self {
            model,
            optimized_weights: serializable.optimized_weights.clone(),
            execution_plan: serializable.execution_plan.clone(),
            memory_size: Self::calculate_memory_size_from_parts(
                &serializable.metadata,
                &serializable.optimized_weights,
                &serializable.execution_plan
            ),
            access_count: 0,
            last_accessed: std::time::Instant::now(),
            serialized_cache: Some(bytes.to_vec()),
        })
    }

    /// Optimize weights for the specific model architecture
    fn optimize_weights(model: &LoadedModel) -> Vec<u8> {
        let mut optimized = Vec::new();
        
        // Pack weights efficiently with compression opportunities
        for (layer_id, weights) in &model.weights.layer_weights {
            // Write layer header
            optimized.extend_from_slice(&layer_id.to_le_bytes());
            optimized.extend_from_slice(&(weights.len() as u32).to_le_bytes());
            
            // TODO: Apply layer-specific optimizations based on LayerType
            // For now, just store weights directly but could apply:
            // - Quantization-aware packing for BitLinear layers
            // - Sparse storage for layers with many zeros
            // - Huffman encoding for repeated patterns
            optimized.extend_from_slice(weights);
        }
        
        optimized
    }

    /// Create execution plan based on model architecture
    fn create_execution_plan(model: &LoadedModel) -> ExecutionPlan {
        let mut layers = Vec::new();
        let mut estimated_memory = 0;

        // Create layer execution plans
        for layer_def in &model.architecture.layers {
            let layer_execution = LayerExecution {
                id: layer_def.id,
                layer_type: Self::convert_layer_type(&layer_def.layer_type, &layer_def.parameters),
                inputs: vec![TensorSpec {
                    shape: layer_def.input_dims.clone(),
                    dtype: "f32".to_string(),
                    layout: TensorLayout::Contiguous,
                }],
                outputs: vec![TensorSpec {
                    shape: layer_def.output_dims.clone(),
                    dtype: "f32".to_string(),
                    layout: TensorLayout::Contiguous,
                }],
                device_placement: DevicePlacement::CPU, // Default to CPU, can be optimized later
            };
            
            estimated_memory += Self::estimate_layer_memory(&layer_execution);
            layers.push(layer_execution);
        }

        ExecutionPlan {
            layers: layers.clone(),
            memory_layout: MemoryLayout::Sequential,
            operator_fusion: Self::identify_fusion_opportunities(&layers),
            estimated_memory,
        }
    }

    /// Convert model layer to execution layer type
    fn convert_layer_type(layer_type: &LayerType, parameters: &LayerParameters) -> ExecutionLayerType {
        match layer_type {
            LayerType::BitLinear => {
                if let LayerParameters::BitLinear { weight_bits, activation_bits } = parameters {
                    ExecutionLayerType::BitLinear {
                        input_dim: 512, // Default - would need to get from layer definition
                        output_dim: 512,
                        weight_bits: *weight_bits,
                        activation_bits: *activation_bits,
                    }
                } else {
                    ExecutionLayerType::BitLinear {
                        input_dim: 512,
                        output_dim: 512,
                        weight_bits: 1,
                        activation_bits: 8,
                    }
                }
            }
            LayerType::RMSNorm => {
                if let LayerParameters::RMSNorm { eps } = parameters {
                    ExecutionLayerType::RMSNorm {
                        eps: *eps,
                        normalized_shape: vec![512], // Default
                    }
                } else {
                    ExecutionLayerType::RMSNorm {
                        eps: 1e-6,
                        normalized_shape: vec![512],
                    }
                }
            }
            LayerType::SwiGLU => {
                if let LayerParameters::SwiGLU { hidden_dim } = parameters {
                    ExecutionLayerType::SwiGLU {
                        hidden_dim: *hidden_dim,
                    }
                } else {
                    ExecutionLayerType::SwiGLU {
                        hidden_dim: 2048,
                    }
                }
            }
            LayerType::Embedding => {
                if let LayerParameters::Embedding { vocab_size, embedding_dim } = parameters {
                    ExecutionLayerType::Embedding {
                        vocab_size: *vocab_size,
                        embedding_dim: *embedding_dim,
                    }
                } else {
                    ExecutionLayerType::Embedding {
                        vocab_size: 32000,
                        embedding_dim: 512,
                    }
                }
            }
            LayerType::OutputProjection => {
                if let LayerParameters::OutputProjection { vocab_size } = parameters {
                    ExecutionLayerType::OutputProjection {
                        vocab_size: *vocab_size,
                    }
                } else {
                    ExecutionLayerType::OutputProjection {
                        vocab_size: 32000,
                    }
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
        
        // Base memory for input/output tensors (f32)
        let tensor_memory = (input_size + output_size) * 4;
        
        // Add layer-specific weight memory
        let weight_memory = match &layer.layer_type {
            ExecutionLayerType::BitLinear { input_dim, output_dim, .. } => {
                input_dim * output_dim / 8 // Bit-packed weights
            }
            ExecutionLayerType::Embedding { vocab_size, embedding_dim } => {
                vocab_size * embedding_dim * 4 // f32 embeddings
            }
            _ => 1024, // Default 1KB for other layers
        };
        
        tensor_memory + weight_memory
    }

    /// Identify operator fusion opportunities
    fn identify_fusion_opportunities(layers: &[LayerExecution]) -> Vec<FusionGroup> {
        let mut fusion_groups = Vec::new();
        
        // Look for BitLinear + RMSNorm patterns (common in BitNet)
        for i in 0..(layers.len().saturating_sub(1)) {
            if matches!(layers[i].layer_type, ExecutionLayerType::BitLinear { .. }) &&
               matches!(layers[i + 1].layer_type, ExecutionLayerType::RMSNorm { .. }) {
                fusion_groups.push(FusionGroup {
                    fused_layers: vec![i, i + 1],
                    fusion_type: FusionType::Custom("BitLinear+RMSNorm".to_string()),
                    performance_gain: 0.20, // 20% estimated gain for BitNet-specific fusion
                });
            }
        }
        
        // Look for SwiGLU activation patterns
        for i in 0..(layers.len().saturating_sub(2)) {
            if matches!(layers[i].layer_type, ExecutionLayerType::BitLinear { .. }) &&
               matches!(layers[i + 1].layer_type, ExecutionLayerType::SwiGLU { .. }) &&
               matches!(layers[i + 2].layer_type, ExecutionLayerType::BitLinear { .. }) {
                fusion_groups.push(FusionGroup {
                    fused_layers: vec![i, i + 1, i + 2],
                    fusion_type: FusionType::Custom("FFN-SwiGLU".to_string()),
                    performance_gain: 0.25, // 25% estimated gain for fused FFN
                });
            }
        }
        
        fusion_groups
    }

    /// Calculate the total memory size including optimized weights and execution plan
    fn calculate_memory_size(model: &LoadedModel, optimized_weights: &[u8], execution_plan: &ExecutionPlan) -> usize {
        let base_size = std::mem::size_of::<LoadedModel>();
        let weights_size = optimized_weights.len();
        let plan_size = execution_plan.estimated_memory;
        let overhead = 4096; // 4KB overhead for structures
        
        base_size + weights_size + plan_size + overhead
    }

    /// Calculate memory size from serialized parts
    fn calculate_memory_size_from_parts(metadata: &ModelMetadata, weights: &[u8], plan: &ExecutionPlan) -> usize {
        let base_size = std::mem::size_of::<ModelMetadata>();
        let weights_size = weights.len();
        let plan_size = plan.estimated_memory;
        let overhead = 4096;
        
        base_size + weights_size + plan_size + overhead
    }
}

/// Serializable version of cached model for persistent storage
#[derive(Serialize, Deserialize)]
struct SerializableModel {
    metadata: ModelMetadata,
    optimized_weights: Vec<u8>,
    execution_plan: ExecutionPlan,
}

/// Advanced high-performance model cache with serialization support
pub struct AdvancedModelCache {
    /// LRU cache for models
    cache: Arc<Mutex<LruCache<String, CachedModel>>>,
    /// Current memory usage
    current_memory: Arc<Mutex<usize>>,
    /// Maximum memory usage
    max_memory: usize,
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of cache evictions
    pub evictions: u64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of serialization operations
    pub serializations: u64,
    /// Number of deserialization operations
    pub deserializations: u64,
}

impl AdvancedModelCache {
    /// Create a new model cache with specified capacity and memory limit.
    pub fn new(capacity: usize, max_memory: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(
                LruCache::new(NonZeroUsize::new(capacity).unwrap())
            )),
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
            if let Some(cached_model) = cache.get_mut(key) {
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
            
            cache.put(key.to_string(), cached_model);
            *current_memory += result.memory_size;
            
            // Update peak memory usage
            let mut stats = self.stats.lock().unwrap();
            stats.peak_memory_usage = stats.peak_memory_usage.max(*current_memory);
        }
        
        Ok(result)
    }

    /// Serialize a cached model to bytes for persistent storage
    pub fn serialize_model(&self, key: &str) -> Result<Vec<u8>> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(cached_model) = cache.get_mut(key) {
            let serialized = cached_model.serialize()?.to_vec();
            let mut stats = self.stats.lock().unwrap();
            stats.serializations += 1;
            Ok(serialized)
        } else {
            Err(InferenceError::model_load(&format!("Model '{}' not found in cache", key)))
        }
    }

    /// Deserialize and cache a model from bytes
    pub fn deserialize_and_cache(&self, key: &str, bytes: &[u8]) -> Result<CachedModel> {
        let cached_model = CachedModel::deserialize(bytes)?;
        
        // Ensure we have enough memory
        self.ensure_memory_capacity(cached_model.memory_size)?;
        
        // Add to cache
        {
            let mut cache = self.cache.lock().unwrap();
            let mut current_memory = self.current_memory.lock().unwrap();
            
            cache.put(key.to_string(), cached_model.clone());
            *current_memory += cached_model.memory_size;
            
            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.deserializations += 1;
            stats.peak_memory_usage = stats.peak_memory_usage.max(*current_memory);
        }
        
        Ok(cached_model)
    }

    /// Remove a model from the cache
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

    /// Clear all models from the cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        let mut current_memory = self.current_memory.lock().unwrap();
        
        cache.clear();
        *current_memory = 0;
    }

    /// Get current memory usage in bytes
    pub fn current_memory_usage(&self) -> usize {
        *self.current_memory.lock().unwrap()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.hits + stats.misses;
        
        if total == 0 {
            0.0
        } else {
            (stats.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get list of currently cached model keys
    pub fn cached_keys(&self) -> Vec<String> {
        let cache = self.cache.lock().unwrap();
        cache.iter().map(|(key, _)| key.clone()).collect()
    }

    /// Ensure we have enough memory capacity for a new model
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
                        &format!(
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

    /// Record a cache hit
    fn record_hit(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.hits += 1;
    }

    /// Record a cache miss
    fn record_miss(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
    }

    /// Record a cache eviction
    fn record_eviction(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.evictions += 1;
    }
}

impl CacheStats {
    /// Get total number of cache accesses
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }

    /// Get cache hit rate as a percentage
    pub fn hit_rate_percent(&self) -> f64 {
        let total = self.total_accesses();
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Check if cache performance is good
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
                layers: vec![LayerDefinition {
                    id: 0,
                    layer_type: LayerType::BitLinear,
                    input_dims: vec![512],
                    output_dims: vec![1000],
                    parameters: LayerParameters::BitLinear {
                        weight_bits: 1,
                        activation_bits: 8,
                    },
                }],
                execution_order: vec![0],
            },
            weights: ModelWeights {
                layer_weights,
                total_size: weights_size,
            },
        }
    }

    #[test]
    fn test_advanced_cache_basic_operations() {
        let cache = AdvancedModelCache::new(2, 100 * 1024 * 1024); // 100MB limit
        
        // Test cache miss and load
        let model1 = cache.get_or_load("model1", || Ok(create_test_model("model1", 10))).unwrap();
        assert_eq!(model1.model.metadata.name, "model1");
        assert!(!model1.optimized_weights.is_empty());
        assert!(!model1.execution_plan.layers.is_empty());
        
        // Test cache hit
        let model1_again = cache.get_or_load("model1", || panic!("Should not be called")).unwrap();
        assert_eq!(model1_again.model.metadata.name, "model1");
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_advanced_cache_serialization() {
        let cache = AdvancedModelCache::new(2, 100 * 1024 * 1024);
        
        // Load a model
        let _model = cache.get_or_load("test_model", || Ok(create_test_model("test_model", 5))).unwrap();
        
        // Serialize the model
        let serialized = cache.serialize_model("test_model").unwrap();
        assert!(!serialized.is_empty());
        
        // Test deserialization
        let deserialized = CachedModel::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.model.metadata.name, "test_model");
        
        let stats = cache.stats();
        assert_eq!(stats.serializations, 1);
    }

    #[test]
    fn test_execution_plan_creation() {
        let model = create_test_model("plan_test", 1);
        let cached = CachedModel::new(model);
        
        assert_eq!(cached.execution_plan.layers.len(), 1);
        assert!(matches!(
            cached.execution_plan.layers[0].layer_type,
            ExecutionLayerType::BitLinear { weight_bits: 1, activation_bits: 8, .. }
        ));
        assert!(cached.execution_plan.estimated_memory > 0);
    }

    #[test]
    fn test_fusion_opportunities() {
        let mut model = create_test_model("fusion_test", 1);
        
        // Add RMSNorm layer after BitLinear
        model.architecture.layers.push(LayerDefinition {
            id: 1,
            layer_type: LayerType::RMSNorm,
            input_dims: vec![1000],
            output_dims: vec![1000],
            parameters: LayerParameters::RMSNorm { eps: 1e-6 },
        });
        
        let cached = CachedModel::new(model);
        
        // Should identify BitLinear+RMSNorm fusion opportunity
        assert!(!cached.execution_plan.operator_fusion.is_empty());
        assert_eq!(cached.execution_plan.operator_fusion[0].fused_layers, vec![0, 1]);
        assert!(matches!(
            cached.execution_plan.operator_fusion[0].fusion_type,
            FusionType::Custom(ref name) if name == "BitLinear+RMSNorm"
        ));
    }
}
