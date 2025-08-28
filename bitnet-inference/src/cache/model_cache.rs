//! Model caching implementation with LRU eviction and memory management.

use crate::{Result, InferenceError};
use crate::engine::model_loader::{LoadedModel, ModelMetadata};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// A cached model with additional metadata.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// The loaded model
    pub model: LoadedModel,
    /// Size of the model in memory (bytes)
    pub memory_size: usize,
    /// Number of times this model has been accessed
    pub access_count: u64,
    /// Timestamp of last access
    pub last_accessed: std::time::Instant,
}

impl CachedModel {
    /// Create a new cached model.
    pub fn new(model: LoadedModel) -> Self {
        let memory_size = Self::calculate_memory_size(&model);
        
        Self {
            model,
            memory_size,
            access_count: 0,
            last_accessed: std::time::Instant::now(),
        }
    }

    /// Update access statistics.
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::Instant::now();
    }

    /// Calculate the approximate memory size of a loaded model.
    fn calculate_memory_size(model: &LoadedModel) -> usize {
        // Base size for metadata and architecture
        let mut size = std::mem::size_of::<LoadedModel>();
        
        // Add weight data size
        size += model.weights.total_size;
        
        // Add some overhead for internal structures
        size += size / 10; // 10% overhead estimate
        
        size
    }
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
