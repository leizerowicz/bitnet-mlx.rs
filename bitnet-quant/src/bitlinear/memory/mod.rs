//! Memory Optimization Module for BitLinear Layers
//!
//! This module provides comprehensive memory optimization features including:
//! - Lazy quantization (quantize on-demand)
//! - Quantized weight reuse across forward passes
//! - Efficient scaling factor management
//! - Cache-friendly memory access patterns
//! - Integration with existing memory pressure detection

pub mod cache_friendly;
pub mod lazy_quantization;
pub mod pressure_detection;
pub mod scaling_factors;
pub mod weight_cache;

#[cfg(test)]
pub mod tests;

use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use bitnet_core::memory::{HybridMemoryPool, MemoryMetrics};
use candle_core::{Device, Tensor};
use std::sync::Arc;

pub use cache_friendly::{AccessPattern, CacheFriendlyTensor, MemoryLayout};
pub use lazy_quantization::{LazyQuantizationConfig, LazyQuantizer, QuantizationState};
pub use pressure_detection::{MemoryPressureIntegrator, MemoryPressureLevel, PressureConfig};
pub use scaling_factors::{ScaleCache, ScalingFactorManager, ScalingPolicy};
pub use weight_cache::{
    CacheConfig as WeightCacheConfig, CacheEntry as WeightCacheEntry, WeightCacheManager,
};

/// Comprehensive memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable lazy quantization
    pub enable_lazy_quantization: bool,
    /// Weight cache configuration
    pub weight_cache_config: WeightCacheConfig,
    /// Lazy quantization configuration
    pub lazy_quantization_config: LazyQuantizationConfig,
    /// Scaling factor management configuration
    pub scaling_policy: ScalingPolicy,
    /// Cache-friendly access pattern preferences
    pub preferred_access_pattern: AccessPattern,
    /// Memory pressure integration settings
    pub pressure_config: PressureConfig,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
    /// Enable memory layout optimization
    pub enable_layout_optimization: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_lazy_quantization: true,
            weight_cache_config: WeightCacheConfig::default(),
            lazy_quantization_config: LazyQuantizationConfig::default(),
            scaling_policy: ScalingPolicy::default(),
            preferred_access_pattern: AccessPattern::Sequential,
            pressure_config: PressureConfig::default(),
            memory_alignment: 64, // Cache line alignment
            enable_layout_optimization: true,
        }
    }
}

/// Memory optimization metrics for monitoring and debugging
#[derive(Debug, Clone, Default)]
pub struct MemoryOptimizationMetrics {
    /// Lazy quantization statistics
    pub lazy_quantization_hits: u64,
    pub lazy_quantization_misses: u64,
    pub lazy_quantization_evictions: u64,

    /// Weight cache statistics
    pub weight_cache_hits: u64,
    pub weight_cache_misses: u64,
    pub weight_cache_size_bytes: usize,

    /// Scaling factor cache statistics
    pub scale_cache_hits: u64,
    pub scale_cache_misses: u64,

    /// Memory pressure statistics
    pub pressure_events_low: u64,
    pub pressure_events_high: u64,
    pub pressure_events_critical: u64,

    /// Layout optimization statistics
    pub layout_optimizations_applied: u64,
    pub memory_copies_avoided: u64,

    /// Total memory saved through optimizations
    pub total_memory_saved_bytes: usize,
}

impl MemoryOptimizationMetrics {
    /// Calculate cache hit rate for lazy quantization
    pub fn lazy_quantization_hit_rate(&self) -> f32 {
        let total = self.lazy_quantization_hits + self.lazy_quantization_misses;
        if total == 0 {
            0.0
        } else {
            self.lazy_quantization_hits as f32 / total as f32
        }
    }

    /// Calculate cache hit rate for weight cache
    pub fn weight_cache_hit_rate(&self) -> f32 {
        let total = self.weight_cache_hits + self.weight_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.weight_cache_hits as f32 / total as f32
        }
    }

    /// Calculate cache hit rate for scaling factors
    pub fn scale_cache_hit_rate(&self) -> f32 {
        let total = self.scale_cache_hits + self.scale_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.scale_cache_hits as f32 / total as f32
        }
    }

    /// Get total memory pressure events
    pub fn total_pressure_events(&self) -> u64 {
        self.pressure_events_low + self.pressure_events_high + self.pressure_events_critical
    }
}

/// Comprehensive memory optimizer for BitLinear layers
pub struct BitLinearMemoryOptimizer {
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Lazy quantizer
    lazy_quantizer: LazyQuantizer,
    /// Weight cache manager
    weight_cache: WeightCacheManager,
    /// Scaling factor manager
    scale_manager: ScalingFactorManager,
    /// Memory pressure integrator
    pressure_integrator: MemoryPressureIntegrator,
    /// Memory pool reference
    memory_pool: Arc<HybridMemoryPool>,
    /// Optimization metrics
    metrics: MemoryOptimizationMetrics,
}

impl BitLinearMemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(
        config: MemoryOptimizationConfig,
        memory_pool: Arc<HybridMemoryPool>,
        device: &Device,
    ) -> BitLinearResult<Self> {
        let lazy_quantizer = LazyQuantizer::new(
            config.lazy_quantization_config.clone(),
            memory_pool.clone(),
            device,
        )?;

        let weight_cache =
            WeightCacheManager::new(config.weight_cache_config.clone(), memory_pool.clone())?;

        let scale_manager =
            ScalingFactorManager::new(config.scaling_policy.clone(), memory_pool.clone(), device)?;

        let pressure_integrator =
            MemoryPressureIntegrator::new(config.pressure_config.clone(), memory_pool.clone())?;

        Ok(Self {
            config,
            lazy_quantizer,
            weight_cache,
            scale_manager,
            pressure_integrator,
            memory_pool,
            metrics: MemoryOptimizationMetrics::default(),
        })
    }

    /// Optimize tensor layout for cache-friendly access
    pub fn optimize_tensor_layout(
        &mut self,
        tensor: &Tensor,
        access_pattern: AccessPattern,
    ) -> BitLinearResult<CacheFriendlyTensor> {
        if !self.config.enable_layout_optimization {
            return CacheFriendlyTensor::from_tensor(tensor.clone(), MemoryLayout::default());
        }

        let optimized = cache_friendly::optimize_for_access_pattern(
            tensor,
            access_pattern,
            self.config.memory_alignment,
            &self.memory_pool,
        )?;

        if optimized.is_optimized() {
            self.metrics.layout_optimizations_applied += 1;
            self.metrics.memory_copies_avoided += 1;
        }

        Ok(optimized)
    }

    /// Get or create quantized weights with lazy quantization
    pub fn get_quantized_weights(
        &mut self,
        layer_name: &str,
        weights: &Tensor,
        force_quantize: bool,
    ) -> BitLinearResult<(Tensor, Tensor)> {
        // (quantized_weights, scales)
        // Check weight cache first
        if let Some(cached_entry) = self.weight_cache.get(layer_name) {
            self.metrics.weight_cache_hits += 1;

            // Verify cache validity
            if cached_entry.is_valid_for_tensor(weights)? {
                return Ok((
                    cached_entry.quantized_weights().clone(),
                    cached_entry.scales().clone(),
                ));
            } else {
                // Invalidate stale cache entry
                self.weight_cache.invalidate(layer_name);
            }
        }

        self.metrics.weight_cache_misses += 1;

        // Use lazy quantization if enabled
        let (quantized_weights, scales) = if self.config.enable_lazy_quantization && !force_quantize
        {
            self.lazy_quantizer.get_or_quantize(layer_name, weights)?
        } else {
            // Force immediate quantization
            let scales = self.scale_manager.compute_scales(weights)?;
            let quantized = self.quantize_with_scales(weights, &scales)?;
            (quantized, scales)
        };

        // Update weight cache
        let cache_entry = WeightCacheEntry::new(
            quantized_weights.clone(),
            scales.clone(),
            weights.clone(),
            layer_name.to_string(),
        )?;

        let _ = self
            .weight_cache
            .insert(layer_name.to_string(), cache_entry);

        // Update scaling factor cache
        self.scale_manager.cache_scales(layer_name, &scales)?;

        Ok((quantized_weights, scales))
    }

    /// Check memory pressure and trigger optimizations if needed
    pub fn check_memory_pressure(&mut self) -> BitLinearResult<()> {
        let current_level = self.pressure_integrator.check_pressure()?;

        match current_level {
            MemoryPressureLevel::Low => {
                self.metrics.pressure_events_low += 1;
            }
            MemoryPressureLevel::High => {
                self.metrics.pressure_events_high += 1;
                self.handle_high_pressure()?;
            }
            MemoryPressureLevel::Critical => {
                self.metrics.pressure_events_critical += 1;
                self.handle_critical_pressure()?;
            }
        }

        Ok(())
    }

    /// Get current optimization metrics
    pub fn metrics(&self) -> &MemoryOptimizationMetrics {
        &self.metrics
    }

    /// Get mutable access to metrics for updating
    pub fn metrics_mut(&mut self) -> &mut MemoryOptimizationMetrics {
        &mut self.metrics
    }

    /// Get memory pool metrics
    pub fn memory_pool_metrics(&self) -> MemoryMetrics {
        self.memory_pool.get_metrics()
    }

    /// Cleanup and optimize memory usage
    pub fn cleanup(&mut self) -> BitLinearResult<()> {
        // Cleanup caches
        self.weight_cache.cleanup()?;
        self.lazy_quantizer.cleanup()?;
        self.scale_manager.cleanup()?;

        // Update metrics
        self.metrics.total_memory_saved_bytes += self.weight_cache.bytes_freed();

        Ok(())
    }

    /// Reset all optimization metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = MemoryOptimizationMetrics::default();
        self.weight_cache.reset_metrics();
        self.lazy_quantizer.reset_metrics();
        self.scale_manager.reset_metrics();
    }

    // Private helper methods

    fn quantize_with_scales(&self, weights: &Tensor, scales: &Tensor) -> BitLinearResult<Tensor> {
        // Simple ternary quantization: {-1, 0, 1}
        let normalized = weights
            .broadcast_div(scales)
            .map_err(|e| BitLinearError::TensorError(format!("Scale normalization failed: {e}")))?;

        // Clamp to [-1, 1] and round to nearest integer
        let clamped = normalized
            .clamp(-1.0, 1.0)
            .map_err(|e| BitLinearError::TensorError(format!("Clamping failed: {e}")))?;

        let quantized = clamped.round().map_err(|e| {
            BitLinearError::TensorError(format!("Quantization rounding failed: {e}"))
        })?;

        Ok(quantized)
    }

    fn handle_high_pressure(&mut self) -> BitLinearResult<()> {
        // Trigger moderate cleanup
        self.weight_cache.evict_lru(0.3)?; // Evict 30% of cache
        self.lazy_quantizer.reduce_cache_size(0.2)?; // Reduce by 20%
        Ok(())
    }

    fn handle_critical_pressure(&mut self) -> BitLinearResult<()> {
        // Aggressive cleanup
        self.weight_cache.clear();
        self.lazy_quantizer.clear_cache()?;
        self.scale_manager.clear_cache()?;

        // Force garbage collection in memory pool
        self.memory_pool.cleanup_orphaned_handles();

        Ok(())
    }
}

impl std::fmt::Debug for BitLinearMemoryOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitLinearMemoryOptimizer")
            .field("config", &self.config)
            .field("metrics", &self.metrics)
            .field("weight_cache_size", &self.weight_cache.len())
            .field("lazy_quantizer_size", &self.lazy_quantizer.cache_size())
            .finish()
    }
}
