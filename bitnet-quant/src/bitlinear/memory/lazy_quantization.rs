//! Lazy Quantization Implementation
//!
//! This module provides on-demand quantization capabilities, only quantizing weights
//! when needed for forward pass operations, reducing memory usage and computational overhead.

use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use bitnet_core::memory::{HybridMemoryPool, MemoryHandle};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for lazy quantization
#[derive(Debug, Clone)]
pub struct LazyQuantizationConfig {
    /// Maximum number of cached quantized weights
    pub max_cache_entries: usize,
    /// Enable LRU eviction when cache is full
    pub enable_lru_eviction: bool,
    /// Cache entries expire after this duration (seconds)
    pub cache_ttl_seconds: u64,
    /// Minimum memory threshold before triggering quantization
    pub memory_threshold_bytes: usize,
    /// Enable memory pressure-aware quantization
    pub enable_pressure_aware: bool,
    /// Quantization batch size for memory efficiency
    pub batch_size: usize,
}

impl Default for LazyQuantizationConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 128,
            enable_lru_eviction: true,
            cache_ttl_seconds: 3600,                   // 1 hour
            memory_threshold_bytes: 100 * 1024 * 1024, // 100MB
            enable_pressure_aware: true,
            batch_size: 1024 * 1024, // 1M elements per batch
        }
    }
}

/// State of quantization for a particular weight tensor
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationState {
    /// Not quantized yet
    NotQuantized,
    /// Currently being quantized (to prevent duplicate work)
    Quantizing,
    /// Successfully quantized and cached
    Quantized { timestamp: u64, memory_usage: usize },
    /// Quantization failed
    Failed { error: String, timestamp: u64 },
}

impl QuantizationState {
    /// Check if the state indicates quantization is available
    pub fn is_quantized(&self) -> bool {
        matches!(self, QuantizationState::Quantized { .. })
    }

    /// Check if the state indicates quantization is in progress
    pub fn is_quantizing(&self) -> bool {
        matches!(self, QuantizationState::Quantizing)
    }

    /// Get the timestamp of the state
    pub fn timestamp(&self) -> Option<u64> {
        match self {
            QuantizationState::Quantized { timestamp, .. }
            | QuantizationState::Failed { timestamp, .. } => Some(*timestamp),
            _ => None,
        }
    }
}

/// Cache entry for lazily quantized weights
#[derive(Debug)]
struct LazyQuantizationEntry {
    /// Original full-precision weights
    original_weights: Tensor,
    /// Quantized weights (if available)
    quantized_weights: Option<Tensor>,
    /// Scaling factors
    scales: Option<Tensor>,
    /// Current quantization state
    state: QuantizationState,
    /// Last access timestamp
    last_accessed: u64,
    /// Memory handle for the quantized data
    memory_handle: Option<MemoryHandle>,
    /// Layer name for debugging
    layer_name: String,
}

impl LazyQuantizationEntry {
    fn new(weights: Tensor, layer_name: String) -> Self {
        Self {
            original_weights: weights,
            quantized_weights: None,
            scales: None,
            state: QuantizationState::NotQuantized,
            last_accessed: current_timestamp(),
            memory_handle: None,
            layer_name,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = current_timestamp();
    }

    fn is_expired(&self, ttl_seconds: u64) -> bool {
        let current_time = current_timestamp();
        current_time.saturating_sub(self.last_accessed) > ttl_seconds
    }

    fn memory_usage(&self) -> usize {
        let mut usage = 0;

        // Original weights
        usage += tensor_memory_usage(&self.original_weights);

        // Quantized weights
        if let Some(ref quantized) = self.quantized_weights {
            usage += tensor_memory_usage(quantized);
        }

        // Scales
        if let Some(ref scales) = self.scales {
            usage += tensor_memory_usage(scales);
        }

        usage
    }
}

/// Lazy quantization statistics
#[derive(Debug, Clone, Default)]
pub struct LazyQuantizationStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total quantizations performed
    pub quantizations_performed: u64,
    /// Total cache evictions
    pub evictions: u64,
    /// Total memory saved by lazy quantization
    pub memory_saved_bytes: usize,
    /// Current cache size
    pub current_cache_entries: usize,
    /// Peak cache size
    pub peak_cache_entries: usize,
}

impl LazyQuantizationStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

/// Lazy quantizer implementation
pub struct LazyQuantizer {
    /// Configuration
    config: LazyQuantizationConfig,
    /// Cache storage
    cache: Arc<RwLock<HashMap<String, LazyQuantizationEntry>>>,
    /// LRU order tracking
    lru_order: Arc<Mutex<Vec<String>>>,
    /// Memory pool for allocations
    memory_pool: Arc<HybridMemoryPool>,
    /// Target device
    device: Device,
    /// Statistics
    stats: Arc<RwLock<LazyQuantizationStats>>,
}

impl LazyQuantizer {
    /// Create a new lazy quantizer
    pub fn new(
        config: LazyQuantizationConfig,
        memory_pool: Arc<HybridMemoryPool>,
        device: &Device,
    ) -> BitLinearResult<Self> {
        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(Mutex::new(Vec::new())),
            memory_pool,
            device: device.clone(),
            stats: Arc::new(RwLock::new(LazyQuantizationStats::default())),
        })
    }

    /// Get or quantize weights on-demand
    pub fn get_or_quantize(
        &mut self,
        layer_name: &str,
        weights: &Tensor,
    ) -> BitLinearResult<(Tensor, Tensor)> {
        // Try to get from cache first
        if let Some((quantized, scales)) = self.try_get_cached(layer_name, weights)? {
            self.update_stats(|stats| stats.hits += 1);
            return Ok((quantized, scales));
        }

        self.update_stats(|stats| stats.misses += 1);

        // Check if we should quantize based on memory pressure
        if self.config.enable_pressure_aware && self.should_defer_quantization()? {
            // Return identity quantization (no actual quantization)
            let scales = Tensor::ones(&[1], weights.dtype(), weights.device()).map_err(|e| {
                BitLinearError::TensorError(format!("Scale tensor creation failed: {e}"))
            })?;

            return Ok((weights.clone(), scales));
        }

        // Perform lazy quantization
        self.perform_quantization(layer_name, weights)
    }

    /// Force quantization of weights (bypass lazy loading)
    pub fn force_quantize(
        &mut self,
        layer_name: &str,
        weights: &Tensor,
    ) -> BitLinearResult<(Tensor, Tensor)> {
        self.perform_quantization(layer_name, weights)
    }

    /// Check if weights are already quantized
    pub fn is_quantized(&self, layer_name: &str) -> bool {
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(layer_name) {
                return entry.state.is_quantized();
            }
        }
        false
    }

    /// Get current cache size
    pub fn cache_size(&self) -> usize {
        if let Ok(cache) = self.cache.read() {
            cache.len()
        } else {
            0
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> LazyQuantizationStats {
        if let Ok(stats) = self.stats.read() {
            stats.clone()
        } else {
            LazyQuantizationStats::default()
        }
    }

    /// Reset statistics
    pub fn reset_metrics(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = LazyQuantizationStats::default();
        }
    }

    /// Cleanup expired entries
    pub fn cleanup(&mut self) -> BitLinearResult<()> {
        self.evict_expired_entries()?;
        self.update_cache_stats();
        Ok(())
    }

    /// Reduce cache size by the specified factor (0.0 to 1.0)
    pub fn reduce_cache_size(&mut self, reduction_factor: f32) -> BitLinearResult<()> {
        let reduction_factor = reduction_factor.clamp(0.0, 1.0);

        let target_evictions = {
            let cache = self
                .cache
                .read()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;
            (cache.len() as f32 * reduction_factor) as usize
        };

        self.evict_lru_entries(target_evictions)?;
        Ok(())
    }

    /// Clear all cache entries
    pub fn clear_cache(&mut self) -> BitLinearResult<()> {
        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;
            cache.clear();
        }

        {
            let mut lru_order = self
                .lru_order
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;
            lru_order.clear();
        }

        self.update_cache_stats();
        Ok(())
    }

    // Private helper methods

    fn try_get_cached(
        &self,
        layer_name: &str,
        weights: &Tensor,
    ) -> BitLinearResult<Option<(Tensor, Tensor)>> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

        if let Some(entry) = cache.get_mut(layer_name) {
            // Check if entry is valid and not expired
            if entry.is_expired(self.config.cache_ttl_seconds) {
                // Remove expired entry
                cache.remove(layer_name);
                self.remove_from_lru(layer_name);
                return Ok(None);
            }

            // Check if entry is quantized and still valid for the current weights
            if entry.state.is_quantized() && self.is_entry_valid_for_weights(entry, weights)? {
                entry.touch();
                self.update_lru_order(layer_name);

                if let (Some(ref quantized), Some(ref scales)) =
                    (&entry.quantized_weights, &entry.scales)
                {
                    return Ok(Some((quantized.clone(), scales.clone())));
                }
            }
        }

        Ok(None)
    }

    fn perform_quantization(
        &mut self,
        layer_name: &str,
        weights: &Tensor,
    ) -> BitLinearResult<(Tensor, Tensor)> {
        // Mark as being quantized to prevent duplicate work
        self.set_quantization_state(layer_name, QuantizationState::Quantizing)?;

        // Perform actual quantization
        let quantization_result = self.quantize_tensor(weights);

        match quantization_result {
            Ok((quantized, scales)) => {
                // Store in cache
                self.cache_quantized_result(layer_name, weights, &quantized, &scales)?;

                // Update state to quantized
                let memory_usage = tensor_memory_usage(&quantized) + tensor_memory_usage(&scales);
                self.set_quantization_state(
                    layer_name,
                    QuantizationState::Quantized {
                        timestamp: current_timestamp(),
                        memory_usage,
                    },
                )?;

                self.update_stats(|stats| {
                    stats.quantizations_performed += 1;
                    stats.memory_saved_bytes += memory_usage;
                });

                Ok((quantized, scales))
            }
            Err(e) => {
                // Update state to failed
                self.set_quantization_state(
                    layer_name,
                    QuantizationState::Failed {
                        error: e.to_string(),
                        timestamp: current_timestamp(),
                    },
                )?;

                Err(e)
            }
        }
    }

    fn quantize_tensor(&self, weights: &Tensor) -> BitLinearResult<(Tensor, Tensor)> {
        // Compute scales using absolute mean method
        let abs_weights = weights.abs().map_err(|e| {
            BitLinearError::TensorError(format!("Absolute value computation failed: {e}"))
        })?;

        let scales = abs_weights
            .mean_all()
            .map_err(|e| BitLinearError::TensorError(format!("Scale computation failed: {e}")))?;

        // Add small epsilon to prevent division by zero
        let epsilon = Tensor::full(1e-8f32, scales.shape(), scales.device()).map_err(|e| {
            BitLinearError::TensorError(format!("Epsilon tensor creation failed: {e}"))
        })?;
        let scales_safe = scales.add(&epsilon).map_err(|e| {
            BitLinearError::TensorError(format!("Safe scale computation failed: {e}"))
        })?;

        // Normalize and quantize to {-1, 0, 1}
        let normalized = weights.broadcast_div(&scales_safe).map_err(|e| {
            BitLinearError::TensorError(format!("Weight normalization failed: {e}"))
        })?;

        let quantized = normalized
            .clamp(-1.0f32, 1.0f32)
            .and_then(|t| t.round())
            .map_err(|e| BitLinearError::TensorError(format!("Quantization failed: {e}")))?;

        Ok((quantized, scales_safe))
    }

    fn cache_quantized_result(
        &mut self,
        layer_name: &str,
        original_weights: &Tensor,
        quantized_weights: &Tensor,
        scales: &Tensor,
    ) -> BitLinearResult<()> {
        // Check if cache is full
        if self.cache_size() >= self.config.max_cache_entries {
            if self.config.enable_lru_eviction {
                self.evict_lru_entries(1)?;
            } else {
                return Err(BitLinearError::cache_lock_error("Cache is full"));
            }
        }

        let mut cache = self
            .cache
            .write()
            .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

        let entry = LazyQuantizationEntry {
            original_weights: original_weights.clone(),
            quantized_weights: Some(quantized_weights.clone()),
            scales: Some(scales.clone()),
            state: QuantizationState::NotQuantized, // Will be updated by caller
            last_accessed: current_timestamp(),
            memory_handle: None, // Could allocate dedicated memory handle if needed
            layer_name: layer_name.to_string(),
        };

        cache.insert(layer_name.to_string(), entry);
        self.update_lru_order(layer_name);

        Ok(())
    }

    fn is_entry_valid_for_weights(
        &self,
        entry: &LazyQuantizationEntry,
        weights: &Tensor,
    ) -> BitLinearResult<bool> {
        // Simple validation: check if shapes and dtypes match
        let shapes_match = entry.original_weights.shape() == weights.shape();
        let dtypes_match = entry.original_weights.dtype() == weights.dtype();

        Ok(shapes_match && dtypes_match)
    }

    fn set_quantization_state(
        &self,
        layer_name: &str,
        state: QuantizationState,
    ) -> BitLinearResult<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

        if let Some(entry) = cache.get_mut(layer_name) {
            entry.state = state;
        }

        Ok(())
    }

    fn should_defer_quantization(&self) -> BitLinearResult<bool> {
        let memory_metrics = self.memory_pool.get_metrics();

        // Check if current memory usage exceeds threshold
        Ok(memory_metrics.total_allocated > self.config.memory_threshold_bytes as u64)
    }

    fn evict_lru_entries(&mut self, count: usize) -> BitLinearResult<()> {
        let keys_to_evict = {
            let lru_order = self
                .lru_order
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;

            lru_order.iter().take(count).cloned().collect::<Vec<_>>()
        };

        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

            for key in &keys_to_evict {
                cache.remove(key);
            }
        }

        {
            let mut lru_order = self
                .lru_order
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;

            lru_order.retain(|k| !keys_to_evict.contains(k));
        }

        self.update_stats(|stats| stats.evictions += keys_to_evict.len() as u64);

        Ok(())
    }

    fn evict_expired_entries(&mut self) -> BitLinearResult<()> {
        let current_time = current_timestamp();
        let ttl = self.config.cache_ttl_seconds;

        let expired_keys = {
            let cache = self
                .cache
                .read()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

            cache
                .iter()
                .filter_map(|(key, entry)| {
                    if entry.is_expired(ttl) {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };

        if !expired_keys.is_empty() {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire cache lock"))?;

            for key in &expired_keys {
                cache.remove(key);
            }

            drop(cache);

            let mut lru_order = self
                .lru_order
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("Failed to acquire LRU lock"))?;

            lru_order.retain(|k| !expired_keys.contains(k));

            self.update_stats(|stats| stats.evictions += expired_keys.len() as u64);
        }

        Ok(())
    }

    fn update_lru_order(&self, layer_name: &str) {
        if let Ok(mut lru_order) = self.lru_order.lock() {
            // Remove if already exists
            lru_order.retain(|k| k != layer_name);
            // Add to end (most recently used)
            lru_order.push(layer_name.to_string());
        }
    }

    fn remove_from_lru(&self, layer_name: &str) {
        if let Ok(mut lru_order) = self.lru_order.lock() {
            lru_order.retain(|k| k != layer_name);
        }
    }

    fn update_cache_stats(&self) {
        let current_size = self.cache_size();

        self.update_stats(|stats| {
            stats.current_cache_entries = current_size;
            if current_size > stats.peak_cache_entries {
                stats.peak_cache_entries = current_size;
            }
        });
    }

    fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut LazyQuantizationStats),
    {
        if let Ok(mut stats) = self.stats.write() {
            updater(&mut stats);
        }
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn tensor_memory_usage(tensor: &Tensor) -> usize {
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

impl std::fmt::Debug for LazyQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyQuantizer")
            .field("config", &self.config)
            .field("cache_size", &self.cache_size())
            .field("device", &self.device)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::device::get_cpu_device;
    use candle_core::{DType, Tensor};

    #[test]
    fn test_lazy_quantization_basic() {
        let device = get_cpu_device();
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = LazyQuantizationConfig::default();

        let mut quantizer = LazyQuantizer::new(config, memory_pool, &device).unwrap();

        // Create test weights
        let weights = Tensor::randn(0.0f32, 1.0f32, &[4, 4], &device).unwrap();

        // First quantization should be a miss
        let (quantized1, scales1) = quantizer.get_or_quantize("test_layer", &weights).unwrap();
        let stats = quantizer.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Second call should be a hit
        let (quantized2, scales2) = quantizer.get_or_quantize("test_layer", &weights).unwrap();
        let stats = quantizer.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);

        // Results should be identical
        assert_eq!(quantized1.shape(), quantized2.shape());
        assert_eq!(scales1.shape(), scales2.shape());
    }

    #[test]
    fn test_cache_expiration() {
        let device = get_cpu_device();
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let mut config = LazyQuantizationConfig::default();
        config.cache_ttl_seconds = 1; // 1 second TTL

        let mut quantizer = LazyQuantizer::new(config, memory_pool, &device).unwrap();

        let weights = Tensor::ones(&[2, 2], DType::F32, &device).unwrap();

        // Quantize
        let _ = quantizer.get_or_quantize("test_layer", &weights).unwrap();
        assert_eq!(quantizer.cache_size(), 1);

        // Wait for expiration and cleanup
        std::thread::sleep(std::time::Duration::from_secs(2));
        quantizer.cleanup().unwrap();

        // Cache should be empty after cleanup
        assert_eq!(quantizer.cache_size(), 0);
    }
}
