//! Model caching and memory management for efficient inference.

pub mod enhanced_memory_pool;
pub mod advanced_model_cache;
pub mod kv_cache;
pub mod enhanced_kv_cache;

pub use advanced_model_cache::{
    AdvancedModelCache, CachedModel, ExecutionPlan, LayerExecution, 
    MemoryLayout, FusionGroup, FusionType, TensorSpec, DevicePlacement,
    ExecutionLayerType, TensorLayout, CacheStats
};
// Re-export the advanced cache as the primary ModelCache for compatibility
pub use advanced_model_cache::AdvancedModelCache as ModelCache;

pub use kv_cache::{
    KVCacheConfig, KVCacheStats, LayerKVCache, MultiLayerKVCache,
    GenerationState, GenerationConfig
};

pub use enhanced_kv_cache::{
    EnhancedKVCache, EnhancedKVCacheConfig, MemoryPoolConfig, CacheEvictionStrategy,
    PreallocationStrategy, SlidingWindowState
};

use std::path::PathBuf;

/// Configuration for caching behavior.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of models to keep in cache
    pub max_models: usize,
    /// Maximum memory usage for cache (in bytes)
    pub max_memory: usize,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Whether to persist cache to disk
    pub persistent: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            max_memory: 2 * 1024 * 1024 * 1024, // 2GB
            cache_dir: std::env::temp_dir().join("bitnet-inference-cache"),
            persistent: true,
        }
    }
}
