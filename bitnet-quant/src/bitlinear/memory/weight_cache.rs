//! Weight Cache Manager for Efficient Weight Reuse
//!
//! This module provides sophisticated caching for quantized weights that can be
//! reused across multiple forward passes, significantly reducing quantization overhead.

use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use bitnet_core::memory::{HybridMemoryPool, MemoryHandle};
use candle_core::{Device, Tensor, DType, Shape};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for weight cache management
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached weight entries
    pub max_entries: usize,
    /// Maximum total memory usage for cache (bytes)
    pub max_memory_bytes: usize,
    /// Enable LRU eviction when cache is full
    pub enable_lru_eviction: bool,
    /// Enable memory size tracking
    pub enable_size_tracking: bool,
    /// Enable automatic cleanup of expired entries
    pub enable_auto_cleanup: bool,
    /// Cache entries expire after this duration (seconds)
    pub entry_ttl_seconds: u64,
    /// Cleanup interval (seconds)
    pub cleanup_interval_seconds: u64,
    /// Enable memory pressure-aware cache sizing
    pub enable_pressure_aware_sizing: bool,
    /// Cache growth factor when memory is available
    pub growth_factor: f32,
    /// Cache shrink factor under memory pressure
    pub shrink_factor: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            enable_lru_eviction: true,
            enable_size_tracking: true,
            enable_auto_cleanup: true,
            entry_ttl_seconds: 3600, // 1 hour
            cleanup_interval_seconds: 300, // 5 minutes
            enable_pressure_aware_sizing: true,
            growth_factor: 1.5,
            shrink_factor: 0.7,
        }
    }
}

/// Cache entry containing quantized weights and metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Quantized weights tensor
    quantized_weights: Tensor,
    /// Scaling factors
    scales: Tensor,
    /// Hash of original weights for validation
    original_hash: u64,
    /// Original weight shape for validation
    original_shape: Shape,
    /// Original weight dtype for validation
    original_dtype: DType,
    /// Layer name
    layer_name: String,
    /// Creation timestamp
    created_at: u64,
    /// Last access timestamp
    last_accessed: u64,
    /// Access count
    access_count: u64,
    /// Estimated memory usage in bytes
    memory_usage: usize,
    /// Memory handle (if using dedicated allocation)
    memory_handle: Option<MemoryHandle>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(
        quantized_weights: Tensor,
        scales: Tensor,
        original_weights: Tensor,
        layer_name: String,
    ) -> BitLinearResult<Self> {
        let original_hash = compute_tensor_hash(&original_weights)?;
        let original_shape = original_weights.shape().clone();
        let original_dtype = original_weights.dtype();
        let current_time = current_timestamp();
        
        let memory_usage = estimate_memory_usage(&quantized_weights) + estimate_memory_usage(&scales);
        
        Ok(Self {
            quantized_weights,
            scales,
            original_hash,
            original_shape,
            original_dtype,
            layer_name,
            created_at: current_time,
            last_accessed: current_time,
            access_count: 0,
            memory_usage,
            memory_handle: None,
        })
    }
    
    /// Access the cached entry (updates access statistics)
    pub fn access(&mut self) -> (&Tensor, &Tensor) {
        self.last_accessed = current_timestamp();
        self.access_count += 1;
        (&self.quantized_weights, &self.scales)
    }
    
    /// Get quantized weights
    pub fn quantized_weights(&self) -> &Tensor {
        &self.quantized_weights
    }
    
    /// Get scales
    pub fn scales(&self) -> &Tensor {
        &self.scales
    }
    
    /// Check if this entry is valid for the given tensor
    pub fn is_valid_for_tensor(&self, tensor: &Tensor) -> BitLinearResult<bool> {
        // Check basic properties
        if tensor.shape() != &self.original_shape || tensor.dtype() != self.original_dtype {
            return Ok(false);
        }
        
        // Compute hash and compare
        let tensor_hash = compute_tensor_hash(tensor)?;
        Ok(tensor_hash == self.original_hash)
    }
    
    /// Check if entry has expired
    pub fn is_expired(&self, ttl_seconds: u64) -> bool {
        let current_time = current_timestamp();
        current_time.saturating_sub(self.created_at) > ttl_seconds
    }
    
    /// Get age in seconds
    pub fn age_seconds(&self) -> u64 {
        let current_time = current_timestamp();
        current_time.saturating_sub(self.created_at)
    }
    
    /// Get time since last access in seconds
    pub fn idle_seconds(&self) -> u64 {
        let current_time = current_timestamp();
        current_time.saturating_sub(self.last_accessed)
    }
    
    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Get layer name
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }
    
    /// Get access count
    pub fn access_count(&self) -> u64 {
        self.access_count
    }
}

/// Cache statistics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses  
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total entries created
    pub entries_created: u64,
    /// Current number of entries
    pub current_entries: usize,
    /// Peak number of entries
    pub peak_entries: usize,
    /// Current memory usage (bytes)
    pub current_memory_bytes: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Total memory freed through evictions
    pub memory_freed_bytes: usize,
    /// Average access count per entry
    pub average_access_count: f64,
    /// Cache hit rate
    pub hit_rate: f32,
}

impl CacheStats {
    /// Update hit rate calculation
    pub fn update_hit_rate(&mut self) {
        let total_requests = self.hits + self.misses;
        self.hit_rate = if total_requests > 0 {
            self.hits as f32 / total_requests as f32
        } else {
            0.0
        };
    }
    
    /// Calculate memory utilization percentage
    pub fn memory_utilization(&self, max_memory: usize) -> f32 {
        if max_memory > 0 {
            (self.current_memory_bytes as f32) / (max_memory as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Weight cache manager implementation
pub struct WeightCacheManager {
    /// Configuration
    config: CacheConfig,
    /// Main cache storage
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// LRU order tracking
    lru_order: Arc<Mutex<VecDeque<String>>>,
    /// Memory pool for allocations
    memory_pool: Arc<HybridMemoryPool>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Current memory usage tracking
    current_memory_usage: Arc<Mutex<usize>>,
    /// Last cleanup timestamp
    last_cleanup: Arc<Mutex<u64>>,
}

impl WeightCacheManager {
    /// Create a new weight cache manager
    pub fn new(
        config: CacheConfig,
        memory_pool: Arc<HybridMemoryPool>,
    ) -> BitLinearResult<Self> {
        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(Mutex::new(VecDeque::new())),
            memory_pool,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            current_memory_usage: Arc::new(Mutex::new(0)),
            last_cleanup: Arc::new(Mutex::new(current_timestamp())),
        })
    }
    
    /// Get cached quantized weights
    pub fn get(&mut self, layer_name: &str) -> Option<CacheEntry> {
        // Check if auto-cleanup is needed
        if self.config.enable_auto_cleanup {
            let _ = self.maybe_auto_cleanup();
        }
        
        let mut cache = match self.cache.write() {
            Ok(cache) => cache,
            Err(_) => {
                self.update_stats(|stats| stats.misses += 1);
                return None;
            }
        };
        
        if let Some(mut entry) = cache.get_mut(layer_name) {
            // Check if entry has expired
            if entry.is_expired(self.config.entry_ttl_seconds) {
                // Remove expired entry
                let expired_entry = cache.remove(layer_name).unwrap();
                self.remove_from_lru(layer_name);
                self.update_memory_usage(|usage| {
                    *usage = usage.saturating_sub(expired_entry.memory_usage());
                });
                self.update_stats(|stats| {
                    stats.misses += 1;
                    stats.evictions += 1;
                });
                return None;
            }
            
            // Update access statistics
            entry.last_accessed = current_timestamp();
            entry.access_count += 1;
            
            // Update LRU order
            self.update_lru_order(layer_name);
            
            self.update_stats(|stats| stats.hits += 1);
            
            Some(entry.clone())
        } else {
            self.update_stats(|stats| stats.misses += 1);
            None
        }
    }
    
    /// Insert or update cache entry
    pub fn insert(&mut self, key: String, entry: CacheEntry) -> BitLinearResult<()> {
        // Check if we need to evict entries first
        self.ensure_cache_capacity(&key, &entry)?;
        
        let memory_usage = entry.memory_usage();
        let is_new_entry = {
            let cache = self.cache.read()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache read lock"))?;
            !cache.contains_key(&key)
        };
        
        // Insert the entry
        {
            let mut cache = self.cache.write()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache write lock"))?;
            
            // If replacing existing entry, subtract old memory usage
            if let Some(old_entry) = cache.get(&key) {
                self.update_memory_usage(|usage| {
                    *usage = usage.saturating_sub(old_entry.memory_usage());
                });
            }
            
            cache.insert(key.clone(), entry);
        }
        
        // Update memory tracking
        self.update_memory_usage(|usage| *usage += memory_usage);
        
        // Update LRU order
        self.update_lru_order(&key);
        
        // Update statistics
        if is_new_entry {
            self.update_stats(|stats| {
                stats.entries_created += 1;
                stats.current_entries = self.len();
                if stats.current_entries > stats.peak_entries {
                    stats.peak_entries = stats.current_entries;
                }
            });
        }
        
        self.update_memory_stats();
        
        Ok(())
    }
    
    /// Invalidate a cache entry
    pub fn invalidate(&mut self, key: &str) -> bool {
        let removed_entry = {
            let mut cache = match self.cache.write() {
                Ok(cache) => cache,
                Err(_) => return false,
            };
            cache.remove(key)
        };
        
        if let Some(entry) = removed_entry {
            self.remove_from_lru(key);
            self.update_memory_usage(|usage| {
                *usage = usage.saturating_sub(entry.memory_usage());
            });
            self.update_stats(|stats| {
                stats.current_entries = self.len();
                stats.evictions += 1;
            });
            true
        } else {
            false
        }
    }
    
    /// Clear all cache entries
    pub fn clear(&mut self) {
        {
            let mut cache = match self.cache.write() {
                Ok(cache) => cache,
                Err(_) => return,
            };
            cache.clear();
        }
        
        {
            let mut lru_order = match self.lru_order.lock() {
                Ok(order) => order,
                Err(_) => return,
            };
            lru_order.clear();
        }
        
        self.update_memory_usage(|usage| *usage = 0);
        self.update_stats(|stats| {
            stats.current_entries = 0;
            stats.current_memory_bytes = 0;
        });
    }
    
    /// Evict LRU entries by specified factor (0.0 to 1.0)
    pub fn evict_lru(&mut self, eviction_factor: f32) -> BitLinearResult<usize> {
        let eviction_factor = eviction_factor.clamp(0.0, 1.0);
        let current_size = self.len();
        let target_evictions = (current_size as f32 * eviction_factor).ceil() as usize;
        
        self.evict_lru_entries(target_evictions)
    }
    
    /// Get current number of cache entries
    pub fn len(&self) -> usize {
        self.cache.read()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Check if cache contains a specific key
    pub fn contains_key(&self, key: &str) -> bool {
        self.cache.read()
            .map(|cache| cache.contains_key(key))
            .unwrap_or(false)
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }
    
    /// Reset cache statistics
    pub fn reset_metrics(&mut self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = CacheStats::default();
        }
    }
    
    /// Get current memory usage in bytes
    pub fn current_memory_usage(&self) -> usize {
        self.current_memory_usage.lock()
            .map(|usage| *usage)
            .unwrap_or(0)
    }
    
    /// Get bytes freed through cleanup operations
    pub fn bytes_freed(&self) -> usize {
        self.stats()
            .memory_freed_bytes
    }
    
    /// Force cleanup of expired entries
    pub fn cleanup(&mut self) -> BitLinearResult<usize> {
        self.cleanup_expired_entries()
    }
    
    // Private helper methods
    
    fn ensure_cache_capacity(&mut self, key: &str, entry: &CacheEntry) -> BitLinearResult<()> {
        let current_size = self.len();
        let current_memory = self.current_memory_usage();
        let entry_memory = entry.memory_usage();
        
        // Check if we need to make room
        let needs_capacity = current_size >= self.config.max_entries && !self.contains_key(key);
        let needs_memory = current_memory + entry_memory > self.config.max_memory_bytes;
        
        if needs_capacity || needs_memory {
            if !self.config.enable_lru_eviction {
                return Err(BitLinearError::cache_lock_error("Cache is full and eviction is disabled"));
            }
            
            // Calculate how many entries to evict
            let entries_to_evict = if needs_capacity {
                std::cmp::max(1, (current_size - self.config.max_entries + 1))
            } else {
                // Estimate entries needed to free enough memory
                let memory_to_free = current_memory + entry_memory - self.config.max_memory_bytes;
                let avg_entry_size = if current_size > 0 { current_memory / current_size } else { entry_memory };
                std::cmp::max(1, (memory_to_free + avg_entry_size - 1) / avg_entry_size)
            };
            
            self.evict_lru_entries(entries_to_evict)?;
        }
        
        Ok(())
    }
    
    fn evict_lru_entries(&mut self, count: usize) -> BitLinearResult<usize> {
        let keys_to_evict = {
            let lru_order = self.lru_order.lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;
            
            lru_order.iter()
                .take(count)
                .cloned()
                .collect::<Vec<_>>()
        };
        
        let mut total_memory_freed = 0;
        let evicted_count = keys_to_evict.len();
        
        {
            let mut cache = self.cache.write()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;
            
            for key in &keys_to_evict {
                if let Some(entry) = cache.remove(key) {
                    total_memory_freed += entry.memory_usage();
                }
            }
        }
        
        {
            let mut lru_order = self.lru_order.lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;
            
            for key in &keys_to_evict {
                if let Some(pos) = lru_order.iter().position(|k| k == key) {
                    lru_order.remove(pos);
                }
            }
        }
        
        // Update memory tracking
        self.update_memory_usage(|usage| {
            *usage = usage.saturating_sub(total_memory_freed);
        });
        
        // Update statistics
        self.update_stats(|stats| {
            stats.evictions += evicted_count as u64;
            stats.current_entries = self.len();
            stats.memory_freed_bytes += total_memory_freed;
        });
        
        Ok(evicted_count)
    }
    
    fn cleanup_expired_entries(&mut self) -> BitLinearResult<usize> {
        let current_time = current_timestamp();
        let ttl = self.config.entry_ttl_seconds;
        
        let expired_keys = {
            let cache = self.cache.read()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;
            
            cache.iter()
                .filter_map(|(key, entry)| {
                    if entry.is_expired(ttl) {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        
        let expired_count = expired_keys.len();
        let mut total_memory_freed = 0;
        
        if !expired_keys.is_empty() {
            {
                let mut cache = self.cache.write()
                    .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;
                
                for key in &expired_keys {
                    if let Some(entry) = cache.remove(key) {
                        total_memory_freed += entry.memory_usage();
                    }
                }
            }
            
            {
                let mut lru_order = self.lru_order.lock()
                    .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;
                
                for key in &expired_keys {
                    if let Some(pos) = lru_order.iter().position(|k| k == key) {
                        lru_order.remove(pos);
                    }
                }
            }
            
            // Update memory tracking
            self.update_memory_usage(|usage| {
                *usage = usage.saturating_sub(total_memory_freed);
            });
            
            // Update statistics
            self.update_stats(|stats| {
                stats.evictions += expired_count as u64;
                stats.current_entries = self.len();
                stats.memory_freed_bytes += total_memory_freed;
            });
        }
        
        // Update last cleanup timestamp
        if let Ok(mut last_cleanup) = self.last_cleanup.lock() {
            *last_cleanup = current_time;
        }
        
        Ok(expired_count)
    }
    
    fn maybe_auto_cleanup(&mut self) -> BitLinearResult<()> {
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.lock()
                .map(|t| *t)
                .unwrap_or(0);
            
            let current_time = current_timestamp();
            current_time.saturating_sub(last_cleanup) >= self.config.cleanup_interval_seconds
        };
        
        if should_cleanup {
            self.cleanup_expired_entries()?;
        }
        
        Ok(())
    }
    
    fn update_lru_order(&self, key: &str) {
        if let Ok(mut lru_order) = self.lru_order.lock() {
            // Remove if already exists
            if let Some(pos) = lru_order.iter().position(|k| k == key) {
                lru_order.remove(pos);
            }
            // Add to back (most recently used)
            lru_order.push_back(key.to_string());
        }
    }
    
    fn remove_from_lru(&self, key: &str) {
        if let Ok(mut lru_order) = self.lru_order.lock() {
            if let Some(pos) = lru_order.iter().position(|k| k == key) {
                lru_order.remove(pos);
            }
        }
    }
    
    fn update_memory_usage<F>(&self, updater: F)
    where
        F: FnOnce(&mut usize),
    {
        if let Ok(mut usage) = self.current_memory_usage.lock() {
            updater(&mut usage);
        }
    }
    
    fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut CacheStats),
    {
        if let Ok(mut stats) = self.stats.write() {
            updater(&mut stats);
            stats.update_hit_rate();
        }
    }
    
    fn update_memory_stats(&self) {
        let current_memory = self.current_memory_usage();
        
        self.update_stats(|stats| {
            stats.current_memory_bytes = current_memory;
            if current_memory > stats.peak_memory_bytes {
                stats.peak_memory_bytes = current_memory;
            }
        });
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn compute_tensor_hash(tensor: &Tensor) -> BitLinearResult<u64> {
    // Simple hash based on shape, dtype, and a sample of data
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    
    // Hash shape
    for dim in tensor.shape().dims() {
        dim.hash(&mut hasher);
    }
    
    // Hash dtype
    let dtype_id = match tensor.dtype() {
        DType::F16 => 0,
        DType::F32 => 1,
        DType::F64 => 2,
        DType::U8 => 3,
        DType::I64 => 4,
        _ => 99,
    };
    dtype_id.hash(&mut hasher);
    
    // Hash a sample of the data (first and last few elements)
    let flat = tensor.flatten_all()
        .map_err(|e| BitLinearError::TensorError(format!("Tensor flattening failed: {}", e)))?;
    
    let sample_size = std::cmp::min(100, flat.elem_count());
    if sample_size > 0 {
        let data = flat.to_vec1::<f32>()
            .map_err(|e| BitLinearError::TensorError(format!("Tensor data extraction failed: {}", e)))?;
        
        // Hash first few elements
        for i in 0..std::cmp::min(sample_size / 2, data.len()) {
            data[i].to_bits().hash(&mut hasher);
        }
        
        // Hash last few elements
        let start_idx = data.len().saturating_sub(sample_size / 2);
        for i in start_idx..data.len() {
            data[i].to_bits().hash(&mut hasher);
        }
    }
    
    Ok(hasher.finish())
}

fn estimate_memory_usage(tensor: &Tensor) -> usize {
    let elem_count = tensor.elem_count();
    let dtype_size = match tensor.dtype() {
        DType::F16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        DType::U8 => 1,
        DType::I64 => 8,
        _ => 4, // Default assumption
    };
    elem_count * dtype_size
}

impl std::fmt::Debug for WeightCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightCacheManager")
            .field("config", &self.config)
            .field("cache_size", &self.len())
            .field("memory_usage", &self.current_memory_usage())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::device::get_cpu_device;
    use candle_core::{Tensor, DType};

    #[test]
    fn test_weight_cache_basic_operations() {
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = CacheConfig::default();
        let mut cache = WeightCacheManager::new(config, memory_pool).unwrap();
        
        let device = get_cpu_device();
        let weights = Tensor::ones(&[3, 4], DType::F32, &device).unwrap();
        let quantized = Tensor::zeros(&[3, 4], DType::F32, &device).unwrap();
        let scales = Tensor::ones(&[1], DType::F32, &device).unwrap();
        
        // Create cache entry
        let entry = CacheEntry::new(quantized, scales, weights, "test_layer".to_string()).unwrap();
        
        // Insert entry
        cache.insert("test_layer".to_string(), entry).unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("test_layer"));
        
        // Retrieve entry
        let retrieved = cache.get("test_layer");
        assert!(retrieved.is_some());
        
        // Check statistics
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entries_created, 1);
    }
    
    #[test]
    fn test_cache_eviction() {
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let mut config = CacheConfig::default();
        config.max_entries = 2; // Small cache for testing
        
        let mut cache = WeightCacheManager::new(config, memory_pool).unwrap();
        let device = get_cpu_device();
        
        // Fill cache to capacity
        for i in 0..3 {
            let weights = Tensor::ones(&[2, 2], DType::F32, &device).unwrap();
            let quantized = Tensor::zeros(&[2, 2], DType::F32, &device).unwrap();
            let scales = Tensor::ones(&[1], DType::F32, &device).unwrap();
            
            let entry = CacheEntry::new(quantized, scales, weights, format!("layer_{}", i)).unwrap();
            cache.insert(format!("layer_{}", i), entry).unwrap();
        }
        
        // Cache should not exceed max capacity
        assert!(cache.len() <= 2);
        
        // Check that eviction occurred
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }
    
    #[test]
    fn test_cache_expiration() {
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let mut config = CacheConfig::default();
        config.entry_ttl_seconds = 1; // 1 second TTL
        
        let mut cache = WeightCacheManager::new(config, memory_pool).unwrap();
        let device = get_cpu_device();
        
        let weights = Tensor::ones(&[2, 2], DType::F32, &device).unwrap();
        let quantized = Tensor::zeros(&[2, 2], DType::F32, &device).unwrap();
        let scales = Tensor::ones(&[1], DType::F32, &device).unwrap();
        
        let entry = CacheEntry::new(quantized, scales, weights, "test_layer".to_string()).unwrap();
        cache.insert("test_layer".to_string(), entry).unwrap();
        
        assert_eq!(cache.len(), 1);
        
        // Wait for expiration
        std::thread::sleep(std::time::Duration::from_secs(2));
        
        // Entry should be expired and removed on access
        let retrieved = cache.get("test_layer");
        assert!(retrieved.is_none());
        assert_eq!(cache.len(), 0);
    }
}
