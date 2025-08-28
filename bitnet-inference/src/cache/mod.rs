//! Model caching and memory management for efficient inference.

pub mod model_cache;

use crate::Result;
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

// Re-export commonly used types from submodules
pub use model_cache::{ModelCache, CachedModel};
