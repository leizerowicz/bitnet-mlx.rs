//! Scaling Factor Management for Efficient Scale Computation and Caching
//!
//! This module provides sophisticated management of scaling factors used in quantization,
//! with caching, reuse, and memory-efficient computation strategies.

use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use bitnet_core::memory::HybridMemoryPool;
use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Scaling computation policies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingPolicy {
    /// Use absolute mean for scaling (BitNet standard)
    #[default]
    AbsoluteMean,
    /// Use absolute maximum for scaling
    AbsoluteMaximum,
    /// Use percentile-based scaling (more robust to outliers)
    Percentile { percentile: f32 },
    /// Use running average of scales across batches
    RunningAverage { momentum: f32 },
    /// Adaptive scaling based on tensor statistics
    Adaptive {
        min_scale: f32,
        max_scale: f32,
        adaptation_rate: f32,
    },
}

/// Configuration for scaling factor management
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Default scaling policy
    pub default_policy: ScalingPolicy,
    /// Enable caching of computed scales
    pub enable_caching: bool,
    /// Maximum number of cached scales
    pub max_cached_scales: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Minimum scale value to prevent numerical issues
    pub min_scale_value: f32,
    /// Maximum scale value for numerical stability
    pub max_scale_value: f32,
    /// Enable batch processing for large tensors
    pub enable_batch_processing: bool,
    /// Batch size for processing large tensors
    pub batch_size: usize,
    /// Enable SIMD acceleration for scale computation
    pub enable_simd_acceleration: bool,
    /// Device placement preference for scale tensors
    pub scale_device_preference: Option<Device>,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            default_policy: ScalingPolicy::default(),
            enable_caching: true,
            max_cached_scales: 1024,
            cache_ttl_seconds: 1800, // 30 minutes
            min_scale_value: 1e-8,
            max_scale_value: 1e8,
            enable_batch_processing: true,
            batch_size: 1024 * 1024, // 1M elements
            enable_simd_acceleration: true,
            scale_device_preference: None,
        }
    }
}

/// Cached scaling factor entry
#[derive(Debug, Clone)]
pub struct ScaleEntry {
    /// The computed scale tensor
    scale: Tensor,
    /// Hash of the original tensor for validation
    tensor_hash: u64,
    /// Original tensor shape for validation
    tensor_shape: Shape,
    /// Original tensor dtype for validation
    tensordtype: DType,
    /// Scaling policy used for computation
    policy_used: ScalingPolicy,
    /// Creation timestamp
    created_at: u64,
    /// Last access timestamp
    last_accessed: u64,
    /// Number of times this scale has been accessed
    access_count: u64,
    /// Layer name for debugging
    layer_name: String,
}

impl ScaleEntry {
    /// Create a new scale entry
    pub fn new(
        scale: Tensor,
        original_tensor: &Tensor,
        policy: ScalingPolicy,
        layer_name: String,
    ) -> BitLinearResult<Self> {
        let tensor_hash = compute_tensor_hash(original_tensor)?;
        let current_time = current_timestamp();

        Ok(Self {
            scale,
            tensor_hash,
            tensor_shape: original_tensor.shape().clone(),
            tensordtype: original_tensor.dtype(),
            policy_used: policy,
            created_at: current_time,
            last_accessed: current_time,
            access_count: 0,
            layer_name,
        })
    }

    /// Check if this entry is valid for the given tensor and policy
    pub fn is_valid_for(&self, tensor: &Tensor, policy: &ScalingPolicy) -> BitLinearResult<bool> {
        // Check if tensor matches
        if tensor.shape() != &self.tensor_shape || tensor.dtype() != self.tensordtype {
            return Ok(false);
        }

        // Check if policy matches
        if &self.policy_used != policy {
            return Ok(false);
        }

        // Check tensor hash
        let current_hash = compute_tensor_hash(tensor)?;
        Ok(current_hash == self.tensor_hash)
    }

    /// Access the cached scale (updates access statistics)
    pub fn access(&mut self) -> &Tensor {
        self.last_accessed = current_timestamp();
        self.access_count += 1;
        &self.scale
    }

    /// Get the scale tensor without updating access statistics
    pub fn scale(&self) -> &Tensor {
        &self.scale
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
}

/// Scale cache for storing computed scaling factors
pub struct ScaleCache {
    /// Cache storage
    cache: HashMap<String, ScaleEntry>,
    /// Configuration
    config: ScalingConfig,
    /// LRU order for eviction
    lru_order: Vec<String>,
    /// Cache statistics
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl ScaleCache {
    /// Create a new scale cache
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
            lru_order: Vec::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get cached scale if available
    pub fn get(&mut self, key: &str, tensor: &Tensor, policy: &ScalingPolicy) -> Option<Tensor> {
        if let Some(entry) = self.cache.get_mut(key) {
            // Check if entry is valid and not expired
            if !entry.is_expired(self.config.cache_ttl_seconds)
                && entry.is_valid_for(tensor, policy).unwrap_or(false)
            {
                self.hits += 1;
                let result = entry.access().clone();
                self.update_lru(key);
                return Some(result);
            } else {
                // Remove invalid/expired entry
                self.cache.remove(key);
                self.remove_from_lru(key);
            }
        }

        self.misses += 1;
        None
    }

    /// Insert scale into cache
    pub fn insert(&mut self, key: String, entry: ScaleEntry) {
        // Ensure cache capacity
        if self.cache.len() >= self.config.max_cached_scales {
            self.evict_lru();
        }

        // Insert entry
        self.cache.insert(key.clone(), entry);
        self.update_lru(&key);
    }

    /// Clear all cached scales
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    // Private helper methods

    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_order.first().cloned() {
            self.cache.remove(&key);
            self.lru_order.remove(0);
            self.evictions += 1;
        }
    }

    fn update_lru(&mut self, key: &str) {
        // Remove if exists
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        // Add to end (most recently used)
        self.lru_order.push(key.to_string());
    }

    fn remove_from_lru(&mut self, key: &str) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
    }
}

/// Statistics for scaling factor management
#[derive(Debug, Clone, Default)]
pub struct ScalingStats {
    /// Total scale computations performed
    pub computations_performed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total time spent computing scales (microseconds)
    pub total_computation_time_us: u64,
    /// Average computation time per scale (microseconds)
    pub avg_computation_time_us: f64,
    /// Memory usage of cached scales (bytes)
    pub cache_memory_usage: usize,
}

impl ScalingStats {
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f32 / total as f32
        } else {
            0.0
        }
    }
}

/// Scaling factor manager
pub struct ScalingFactorManager {
    /// Configuration
    config: ScalingConfig,
    /// Scale cache
    cache: Arc<RwLock<ScaleCache>>,
    /// Memory pool for allocations
    memory_pool: Arc<HybridMemoryPool>,
    /// Target device
    device: Device,
    /// Running averages for adaptive scaling
    running_averages: Arc<Mutex<HashMap<String, f32>>>,
    /// Statistics
    stats: Arc<RwLock<ScalingStats>>,
}

impl ScalingFactorManager {
    /// Create a new scaling factor manager
    pub fn new(
        policy: ScalingPolicy,
        memory_pool: Arc<HybridMemoryPool>,
        device: &Device,
    ) -> BitLinearResult<Self> {
        let config = ScalingConfig {
            default_policy: policy,
            ..ScalingConfig::default()
        };

        let cache = Arc::new(RwLock::new(ScaleCache::new(config.clone())));

        Ok(Self {
            config,
            cache,
            memory_pool,
            device: device.clone(),
            running_averages: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ScalingStats::default())),
        })
    }

    /// Compute scaling factors for a tensor
    pub fn compute_scales(&mut self, tensor: &Tensor) -> BitLinearResult<Tensor> {
        self.compute_scales_with_policy(tensor, &self.config.default_policy.clone())
    }

    /// Compute scaling factors with specific policy
    pub fn compute_scales_with_policy(
        &mut self,
        tensor: &Tensor,
        policy: &ScalingPolicy,
    ) -> BitLinearResult<Tensor> {
        let start_time = std::time::Instant::now();

        let scale = match policy {
            ScalingPolicy::AbsoluteMean => self.compute_absolute_mean_scale(tensor)?,
            ScalingPolicy::AbsoluteMaximum => self.compute_absolute_max_scale(tensor)?,
            ScalingPolicy::Percentile { percentile } => {
                self.compute_percentile_scale(tensor, *percentile)?
            }
            ScalingPolicy::RunningAverage { momentum } => {
                self.compute_running_average_scale(tensor, *momentum, "default")?
            }
            ScalingPolicy::Adaptive {
                min_scale,
                max_scale,
                adaptation_rate,
            } => self.compute_adaptive_scale(tensor, *min_scale, *max_scale, *adaptation_rate)?,
        };

        // Apply bounds
        let bounded_scale = self.apply_scale_bounds(scale)?;

        // Update statistics
        let computation_time = start_time.elapsed().as_micros() as u64;
        self.update_stats(|stats| {
            stats.computations_performed += 1;
            stats.total_computation_time_us += computation_time;
            stats.avg_computation_time_us =
                stats.total_computation_time_us as f64 / stats.computations_performed as f64;
        });

        Ok(bounded_scale)
    }

    /// Get cached scales or compute if not available
    pub fn get_or_compute_scales(
        &mut self,
        layer_name: &str,
        tensor: &Tensor,
    ) -> BitLinearResult<Tensor> {
        self.get_or_compute_scales_with_policy(
            layer_name,
            tensor,
            &self.config.default_policy.clone(),
        )
    }

    /// Get cached scales or compute with specific policy
    pub fn get_or_compute_scales_with_policy(
        &mut self,
        layer_name: &str,
        tensor: &Tensor,
        policy: &ScalingPolicy,
    ) -> BitLinearResult<Tensor> {
        if self.config.enable_caching {
            // Try to get from cache
            if let Ok(mut cache) = self.cache.write() {
                if let Some(cached_scale) = cache.get(layer_name, tensor, policy) {
                    self.update_stats(|stats| stats.cache_hits += 1);
                    return Ok(cached_scale);
                }
            }

            self.update_stats(|stats| stats.cache_misses += 1);
        }

        // Compute new scale
        let scale = self.compute_scales_with_policy(tensor, policy)?;

        // Cache the result if enabled
        if self.config.enable_caching {
            self.cache_scales_with_policy(layer_name, tensor, &scale, policy.clone())?;
        }

        Ok(scale)
    }

    /// Cache computed scales
    pub fn cache_scales(&mut self, _layer_name: &str, _scales: &Tensor) -> BitLinearResult<()> {
        // For caching without the original tensor, we can't validate properly
        // This is mainly for compatibility with existing code
        Ok(())
    }

    /// Cache scales with full validation information
    pub fn cache_scales_with_policy(
        &mut self,
        layer_name: &str,
        original_tensor: &Tensor,
        scales: &Tensor,
        policy: ScalingPolicy,
    ) -> BitLinearResult<()> {
        if !self.config.enable_caching {
            return Ok(());
        }

        let entry = ScaleEntry::new(
            scales.clone(),
            original_tensor,
            policy,
            layer_name.to_string(),
        )?;

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(layer_name.to_string(), entry);
        }

        Ok(())
    }

    /// Clear scale cache
    pub fn clear_cache(&mut self) -> BitLinearResult<()> {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> ScalingStats {
        self.stats
            .read()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }

    /// Reset statistics
    pub fn reset_metrics(&mut self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = ScalingStats::default();
        }

        if let Ok(cache) = self.cache.write() {
            let (hits, misses, _evictions) = cache.stats();
            if let Ok(mut stats) = self.stats.write() {
                stats.cache_hits = hits;
                stats.cache_misses = misses;
            }
        }
    }

    /// Cleanup expired cache entries
    pub fn cleanup(&mut self) -> BitLinearResult<()> {
        // The cache automatically removes expired entries on access
        // This is mainly for consistency with other cache managers
        Ok(())
    }

    // Private scale computation methods

    fn compute_absolute_mean_scale(&self, tensor: &Tensor) -> BitLinearResult<Tensor> {
        let abs_tensor = tensor.abs().map_err(|e| {
            BitLinearError::TensorError(format!("Absolute value computation failed: {e}"))
        })?;

        let scale = abs_tensor
            .mean_all()
            .map_err(|e| BitLinearError::TensorError(format!("Mean computation failed: {e}")))?;

        Ok(scale)
    }

    fn compute_absolute_max_scale(&self, tensor: &Tensor) -> BitLinearResult<Tensor> {
        let abs_tensor = tensor.abs().map_err(|e| {
            BitLinearError::TensorError(format!("Absolute value computation failed: {e}"))
        })?;

        let scale = abs_tensor
            .max_keepdim(0)
            .map_err(|e| BitLinearError::TensorError(format!("Max computation failed: {e}")))?;

        // Get scalar value
        let flat_scale = scale
            .flatten_all()
            .map_err(|e| BitLinearError::TensorError(format!("Scale flattening failed: {e}")))?;

        let scale_values = flat_scale
            .to_vec1::<f32>()
            .map_err(|e| BitLinearError::TensorError(format!("Scale extraction failed: {e}")))?;

        let max_scale = scale_values.iter().fold(0.0f32, |a, &b| a.max(b));

        Tensor::from_slice(&[max_scale], &[1], tensor.device())
            .map_err(|e| BitLinearError::TensorError(format!("Scale tensor creation failed: {e}")))
    }

    fn compute_percentile_scale(
        &self,
        tensor: &Tensor,
        percentile: f32,
    ) -> BitLinearResult<Tensor> {
        let percentile = percentile.clamp(0.0, 100.0);

        let abs_tensor = tensor.abs().map_err(|e| {
            BitLinearError::TensorError(format!("Absolute value computation failed: {e}"))
        })?;

        let flat = abs_tensor
            .flatten_all()
            .map_err(|e| BitLinearError::TensorError(format!("Tensor flattening failed: {e}")))?;

        let mut values = flat
            .to_vec1::<f32>()
            .map_err(|e| BitLinearError::TensorError(format!("Value extraction failed: {e}")))?;

        // Sort values to compute percentile
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile / 100.0) * (values.len() - 1) as f32).round() as usize;
        let index = index.min(values.len() - 1);

        let scale_value = values[index];

        Tensor::from_slice(&[scale_value], &[1], tensor.device())
            .map_err(|e| BitLinearError::TensorError(format!("Scale tensor creation failed: {e}")))
    }

    fn compute_running_average_scale(
        &self,
        tensor: &Tensor,
        momentum: f32,
        key: &str,
    ) -> BitLinearResult<Tensor> {
        // Compute current scale
        let current_scale = self.compute_absolute_mean_scale(tensor)?;
        let current_value = current_scale.to_scalar::<f32>().map_err(|e| {
            BitLinearError::TensorError(format!("Scale scalar extraction failed: {e}"))
        })?;

        // Update running average
        let averaged_value = {
            let mut averages = self
                .running_averages
                .lock()
                .map_err(|__| BitLinearError::cache_lock_error("running averages"))?;

            let new_avg = if let Some(&prev_avg) = averages.get(key) {
                momentum * prev_avg + (1.0 - momentum) * current_value
            } else {
                current_value
            };

            averages.insert(key.to_string(), new_avg);
            new_avg
        };

        Tensor::from_slice(&[averaged_value], &[1], tensor.device())
            .map_err(|e| BitLinearError::TensorError(format!("Scale tensor creation failed: {e}")))
    }

    fn compute_adaptive_scale(
        &self,
        tensor: &Tensor,
        min_scale: f32,
        max_scale: f32,
        adaptation_rate: f32,
    ) -> BitLinearResult<Tensor> {
        // Start with absolute mean
        let base_scale = self.compute_absolute_mean_scale(tensor)?;
        let base_value = base_scale.to_scalar::<f32>().map_err(|e| {
            BitLinearError::TensorError(format!("Base scale extraction failed: {e}"))
        })?;

        // Compute tensor statistics for adaptation
        let variance = tensor
            .var(0)
            .map_err(|e| BitLinearError::TensorError(format!("Variance computation failed: {e}")))?
            .mean_all()
            .map_err(|e| {
                BitLinearError::TensorError(format!("Mean variance computation failed: {e}"))
            })?
            .to_scalar::<f32>()
            .map_err(|e| {
                BitLinearError::TensorError(format!("Variance scalar extraction failed: {e}"))
            })?;

        // Adapt scale based on variance
        let adaptation_factor = 1.0 + adaptation_rate * variance.sqrt();
        let adapted_scale = (base_value * adaptation_factor).clamp(min_scale, max_scale);

        Tensor::from_slice(&[adapted_scale], &[1], tensor.device()).map_err(|e| {
            BitLinearError::TensorError(format!("Adaptive scale tensor creation failed: {e}"))
        })
    }

    fn apply_scale_bounds(&self, scale: Tensor) -> BitLinearResult<Tensor> {
        let scale_value = scale.to_scalar::<f32>().map_err(|e| {
            BitLinearError::TensorError(format!("Scale value extraction failed: {e}"))
        })?;

        let bounded_value =
            scale_value.clamp(self.config.min_scale_value, self.config.max_scale_value);

        if bounded_value != scale_value {
            // Create new tensor with bounded value
            Tensor::from_slice(&[bounded_value], &[1], scale.device()).map_err(|e| {
                BitLinearError::TensorError(format!("Bounded scale creation failed: {e}"))
            })
        } else {
            Ok(scale)
        }
    }

    fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut ScalingStats),
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

fn compute_tensor_hash(tensor: &Tensor) -> BitLinearResult<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash tensor properties
    tensor.shape().dims().hash(&mut hasher);

    let dtype_id = match tensor.dtype() {
        DType::F16 => 0,
        DType::F32 => 1,
        DType::F64 => 2,
        DType::U8 => 3,
        DType::I64 => 4,
        _ => 99,
    };
    dtype_id.hash(&mut hasher);

    // Hash sample of data
    let flat = tensor
        .flatten_all()
        .map_err(|e| BitLinearError::TensorError(format!("Tensor flattening failed: {e}")))?;

    let sample_size = std::cmp::min(50, flat.elem_count());
    if sample_size > 0 {
        let data = flat.to_vec1::<f32>().map_err(|e| {
            BitLinearError::TensorError(format!("Tensor data extraction failed: {e}"))
        })?;

        for i in 0..std::cmp::min(sample_size, data.len()) {
            data[i].to_bits().hash(&mut hasher);
        }
    }

    Ok(hasher.finish())
}

impl std::fmt::Debug for ScalingFactorManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalingFactorManager")
            .field("config", &self.config)
            .field("device", &self.device)
            .field(
                "cache_size",
                &self.cache.read().map(|c| c.len()).unwrap_or(0),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::device::get_cpu_device;
    use candle_core::{DType, Tensor};

    #[test]
    fn test_absolute_mean_scaling() {
        let device = get_cpu_device();
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());

        let mut manager =
            ScalingFactorManager::new(ScalingPolicy::AbsoluteMean, memory_pool, &device).unwrap();

        let tensor =
            Tensor::from_slice(&[1.0f32, -2.0f32, 3.0f32, -4.0f32], &[4], &device).unwrap();
        let scale = manager.compute_scales(&tensor).unwrap();

        let scale_value = scale.get(0).unwrap().to_scalar::<f32>().unwrap();
        let expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0; // abs mean

        assert!((scale_value - expected).abs() < 1e-6);
    }

    #[test]
    fn test_scale_caching() {
        let device = get_cpu_device();
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());

        let mut manager =
            ScalingFactorManager::new(ScalingPolicy::AbsoluteMean, memory_pool, &device).unwrap();

        let tensor = Tensor::ones(&[3, 3], DType::F32, &device).unwrap();

        // First computation should be a cache miss
        let scale1 = manager
            .get_or_compute_scales("test_layer", &tensor)
            .unwrap();
        let stats1 = manager.stats();
        assert_eq!(stats1.cache_misses, 1);
        assert_eq!(stats1.cache_hits, 0);

        // Second computation should be a cache hit
        let scale2 = manager
            .get_or_compute_scales("test_layer", &tensor)
            .unwrap();
        let stats2 = manager.stats();
        assert_eq!(stats2.cache_misses, 1);
        assert_eq!(stats2.cache_hits, 1);

        // Scales should be identical
        let scale1_val = scale1.to_scalar::<f32>().unwrap();
        let scale2_val = scale2.to_scalar::<f32>().unwrap();
        assert_eq!(scale1_val, scale2_val);
    }

    #[test]
    fn test_scale_bounds() {
        let device = get_cpu_device();
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());

        let mut manager =
            ScalingFactorManager::new(ScalingPolicy::AbsoluteMean, memory_pool, &device).unwrap();

        // Create tensor with very small values
        let small_tensor = Tensor::from_slice(&[1e-10f32, 1e-10f32], &[2], &device).unwrap();
        let scale = manager.compute_scales(&small_tensor).unwrap();
        let scale_value = scale.get(0).unwrap().to_scalar::<f32>().unwrap();

        // Scale should be bounded by minimum value
        assert!(scale_value >= manager.config.min_scale_value);
    }
}
