//! BitLinear Layer Implementation
//!
//! This module provides the core BitLinear struct that implements a linear layer
//! using 1.58-bit quantized weights while maintaining full-precision weights for training.

use crate::bitlinear::cache::{CacheEntry, QuantizedWeightCache};
use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use crate::bitlinear::memory::{AccessPattern, BitLinearMemoryOptimizer, MemoryOptimizationConfig};
use crate::quantization::weights::BitNetWeightQuantizer;
use crate::quantization::{
    QuantizationConfig, QuantizationPrecision, QuantizationStrategy, QuantizedWeight,
    TernaryMethod, WeightQuantizationConfig, WeightQuantizer,
};
use bitnet_core::memory::{HybridMemoryPool, MemoryResult};
use candle_core::{Device, Shape, Tensor};
use std::sync::{Arc, Mutex, RwLock};

/// Configuration for BitLinear layer with memory optimization support
#[derive(Debug, Clone)]
pub struct BitLinearConfig {
    /// Input features dimension
    pub in_features: usize,
    /// Output features dimension
    pub out_features: usize,
    /// Whether to use bias (BitNet standard is bias-free)
    pub use_bias: bool,
    /// Weight quantization configuration
    pub weight_quantization: WeightQuantizationConfig,
    /// Enable weight caching
    pub enable_caching: bool,
    /// Cache eviction threshold
    pub cache_size_limit: Option<usize>,
    /// Memory pool configuration
    pub memory_pool_size: Option<usize>,
    /// Device preference (None = auto-select)
    pub device: Option<Device>,
    /// Memory optimization configuration
    pub memory_optimization: MemoryOptimizationConfig,
    /// Enable memory optimizations
    pub enable_memory_optimization: bool,
    /// Preferred memory access pattern
    pub preferred_access_pattern: AccessPattern,
    /// Enable cache-friendly tensor operations
    pub enable_cache_friendly_ops: bool,
}

impl Default for BitLinearConfig {
    fn default() -> Self {
        Self {
            in_features: 512,
            out_features: 512,
            use_bias: false, // BitNet standard is bias-free
            weight_quantization: WeightQuantizationConfig {
                base: QuantizationConfig {
                    precision: QuantizationPrecision::OneFiveFiveBit,
                    strategy: QuantizationStrategy::Static,
                    per_channel: false,
                    clip_threshold: None,
                    qat_enabled: false,
                    calibration_size: None,
                },
                ternary_method: TernaryMethod::MeanThreshold,
                ..Default::default()
            },
            enable_caching: true,
            cache_size_limit: Some(128), // Reasonable default
            memory_pool_size: Some(64 * 1024 * 1024), // 64MB default
            device: None,                // Auto-select device
            memory_optimization: MemoryOptimizationConfig::default(),
            enable_memory_optimization: true,
            preferred_access_pattern: AccessPattern::Sequential,
            enable_cache_friendly_ops: true,
        }
    }
}

/// BitLinear layer implementation
///
/// This struct maintains both full-precision weights for training and cached
/// quantized weights for inference. It integrates with the existing device
/// abstraction layer and memory pool system.
pub struct BitLinear {
    /// Layer configuration
    config: BitLinearConfig,

    /// Full-precision weights for training [out_features, in_features]
    weights: Arc<RwLock<Tensor>>,

    /// Optional bias term (rarely used in BitNet)
    bias: Option<Arc<RwLock<Tensor>>>,

    /// Weight quantizer
    quantizer: Arc<Mutex<Box<dyn WeightQuantizer>>>,

    /// Quantized weight cache
    cache: Option<Arc<Mutex<QuantizedWeightCache>>>,

    /// Memory pool for tensor allocations
    memory_pool: Arc<HybridMemoryPool>,

    /// Memory optimizer for advanced memory management
    memory_optimizer: Option<Arc<Mutex<BitLinearMemoryOptimizer>>>,

    /// Target device
    device: Device,

    /// Layer name for debugging and caching
    layer_name: String,
}

impl BitLinear {
    /// Create a new BitLinear layer with the given configuration
    pub fn new(config: BitLinearConfig, layer_name: String) -> Result<Self, BitLinearError> {
        // Select device (auto-select if none specified)
        let device = if let Some(dev) = &config.device {
            dev.clone()
        } else {
            Device::Cpu // Default fallback
        };

        // Initialize full-precision weights
        let weight_shape: Shape = (config.out_features, config.in_features).into();
        let weights = Tensor::randn(0.0, 0.02, weight_shape, &device)?;
        let weights = Arc::new(RwLock::new(weights));

        // Initialize bias if enabled
        let bias = if config.use_bias {
            let bias_shape: Shape = (config.out_features,).into();
            let bias_tensor = Tensor::zeros(bias_shape, candle_core::DType::F32, &device)?;
            Some(Arc::new(RwLock::new(bias_tensor)))
        } else {
            None
        };

        // Initialize quantizer
        let quantizer = Arc::new(Mutex::new(Box::new(BitNetWeightQuantizer::new(
            config.weight_quantization.clone(),
            device.clone(),
        )) as Box<dyn WeightQuantizer>));

        // Initialize cache if enabled
        let cache = if config.enable_caching {
            let cache_config = crate::bitlinear::cache::CacheConfig {
                max_entries: config.cache_size_limit.unwrap_or(128),
                enable_lru_eviction: true,
                enable_size_tracking: true,
            };
            let cache_instance = QuantizedWeightCache::new(cache_config)?;
            Some(Arc::new(Mutex::new(cache_instance)))
        } else {
            None
        };

        // Initialize memory pool
        let memory_pool = Arc::new(HybridMemoryPool::new()?);

        // Initialize memory optimizer if enabled
        let memory_optimizer = if config.enable_memory_optimization {
            Some(Arc::new(Mutex::new(BitLinearMemoryOptimizer::new(
                config.memory_optimization.clone(),
                memory_pool.clone(),
                &device,
            )?)))
        } else {
            None
        };

        Ok(Self {
            config,
            weights,
            bias,
            quantizer,
            cache,
            memory_pool,
            memory_optimizer,
            device,
            layer_name,
        })
    }

    /// Initialize weights using Xavier/Glorot initialization
    fn initialize_weights(shape: &Shape, device: &Device) -> BitLinearResult<Tensor> {
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(BitLinearError::ConfigError(
                "Weight tensor must be 2-dimensional".to_string(),
            ));
        }

        let fan_in = dims[1] as f64;
        let fan_out = dims[0] as f64;

        // Xavier/Glorot initialization: std = sqrt(2.0 / (fan_in + fan_out))
        let std_dev = (2.0 / (fan_in + fan_out)).sqrt();

        // Generate random tensor with normal distribution
        let weights = Tensor::randn(0.0, std_dev as f32, shape, device).map_err(|e| {
            BitLinearError::DeviceError(format!("Failed to initialize weights: {e}"))
        })?;

        Ok(weights)
    }

    /// Get the layer configuration
    pub fn config(&self) -> &BitLinearConfig {
        &self.config
    }

    /// Get the full-precision weights (for training)
    ///
    /// Returns a read lock to the weights tensor. This should be used
    /// for gradient computation and weight updates during training.
    pub fn weights(&self) -> Arc<RwLock<Tensor>> {
        Arc::clone(&self.weights)
    }

    /// Get the bias tensor if enabled
    pub fn bias(&self) -> Option<Arc<RwLock<Tensor>>> {
        self.bias.as_ref().map(Arc::clone)
    }

    /// Get the target device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the layer name
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Get or create quantized weights
    ///
    /// This method checks the cache first, and if the weights are not cached
    /// or have been modified, quantizes the current full-precision weights.
    pub fn get_quantized_weights(&self) -> BitLinearResult<QuantizedWeight> {
        // Try cache first if enabled
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("cache"))?;

            // Check if we have a valid cached entry
            if let Some(entry) = cache_guard.get(&self.layer_name) {
                // Verify the entry is still valid (weights haven't changed)
                let weights_guard = self.weights.read().map_err(|_| {
                    BitLinearError::MemoryError("Failed to acquire weights read lock".to_string())
                })?;

                if entry.is_valid_for_tensor(&weights_guard) {
                    return Ok(entry.quantized_weight.clone());
                }
            }
        }

        // Cache miss or disabled - quantize weights
        self.quantize_and_cache_weights()
    }

    /// Quantize the current weights and update cache
    fn quantize_and_cache_weights(&self) -> BitLinearResult<QuantizedWeight> {
        let weights_guard = self.weights.read().map_err(|_| {
            BitLinearError::MemoryError("Failed to acquire weights read lock".to_string())
        })?;

        // Perform quantization
        let quantizer_guard = self.quantizer.lock().map_err(|_| {
            BitLinearError::QuantizationError("Failed to acquire quantizer lock".to_string())
        })?;

        let quantized = quantizer_guard.quantize(&weights_guard).map_err(|e| {
            BitLinearError::QuantizationError(format!("Weight quantization failed: {e}"))
        })?;

        // Update cache if enabled
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("cache"))?;

            let cache_entry =
                CacheEntry::new(quantized.clone(), &weights_guard, self.layer_name.clone())
                    .map_err(|e| {
                        let cache_err = crate::bitlinear::cache::CacheError::TensorError(format!(
                            "Failed to create cache entry: {e}"
                        ));
                        BitLinearError::CacheError(cache_err)
                    })?;

            cache_guard.insert(self.layer_name.clone(), cache_entry);
        }

        Ok(quantized)
    }

    /// Update the full-precision weights (for training)
    ///
    /// This method updates the weights and invalidates any cached quantized weights.
    pub fn update_weights(&self, new_weights: Tensor) -> BitLinearResult<()> {
        // Validate shape
        let expected_shape = Shape::from_dims(&[self.config.out_features, self.config.in_features]);
        if new_weights.shape() != &expected_shape {
            return Err(BitLinearError::shape_mismatch(
                expected_shape.dims().to_vec(),
                new_weights.shape().dims().to_vec(),
            ));
        }

        // Update weights
        {
            let mut weights_guard = self.weights.write().map_err(|_| {
                BitLinearError::MemoryError("Failed to acquire weights write lock".to_string())
            })?;
            *weights_guard = new_weights;
        }

        // Invalidate cache if enabled
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("cache"))?;
            cache_guard.invalidate(&self.layer_name);
        }

        Ok(())
    }

    /// Update bias if enabled
    pub fn update_bias(&self, new_bias: Tensor) -> BitLinearResult<()> {
        if let Some(ref bias) = self.bias {
            let expected_shape = Shape::from_dims(&[self.config.out_features]);
            if new_bias.shape() != &expected_shape {
                return Err(BitLinearError::shape_mismatch(
                    expected_shape.dims().to_vec(),
                    new_bias.shape().dims().to_vec(),
                ));
            }

            let mut bias_guard = bias.write().map_err(|_| {
                BitLinearError::MemoryError("Failed to acquire bias write lock".to_string())
            })?;
            *bias_guard = new_bias;
        } else {
            return Err(BitLinearError::ConfigError(
                "Bias is not enabled for this layer".to_string(),
            ));
        }

        Ok(())
    }

    /// Clear the quantized weight cache
    pub fn clear_cache(&self) -> BitLinearResult<()> {
        if let Some(ref cache) = self.cache {
            let mut cache_guard = cache
                .lock()
                .map_err(|_| BitLinearError::cache_lock_error("cache"))?;
            cache_guard.clear();
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<crate::bitlinear::cache::CacheStats> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.lock().ok().map(|guard| guard.stats()))
    }

    /// Get memory pool statistics
    pub fn memory_stats(&self) -> MemoryResult<bitnet_core::memory::MemoryMetrics> {
        Ok(self.memory_pool.get_metrics())
    }

    /// Move layer to a different device
    pub fn to_device(&mut self, device: Device) -> BitLinearResult<()> {
        if std::mem::discriminant(&self.device) == std::mem::discriminant(&device) {
            // Same device type, no need to move
            return Ok(());
        }

        // Move weights to new device
        {
            let mut weights_guard = self.weights.write().map_err(|_| {
                BitLinearError::MemoryError("Failed to acquire weights write lock".to_string())
            })?;
            let new_weights = weights_guard.to_device(&device).map_err(|e| {
                BitLinearError::DeviceError(format!("Failed to move weights to device: {e}"))
            })?;
            *weights_guard = new_weights;
        }

        // Move bias to new device if enabled
        if let Some(ref bias) = self.bias {
            let mut bias_guard = bias.write().map_err(|_| {
                BitLinearError::MemoryError("Failed to acquire bias write lock".to_string())
            })?;
            let new_bias = bias_guard.to_device(&device).map_err(|e| {
                BitLinearError::DeviceError(format!("Failed to move bias to device: {e}"))
            })?;
            *bias_guard = new_bias;
        }

        // Clear cache as it may contain device-specific data
        self.clear_cache()?;

        // Update device
        self.device = device;

        Ok(())
    }
}

impl std::fmt::Debug for BitLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitLinear")
            .field("config", &self.config)
            .field("device", &self.device)
            .field("layer_name", &self.layer_name)
            .field("has_bias", &self.bias.is_some())
            .field("has_cache", &self.cache.is_some())
            .finish()
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for BitLinear {}
unsafe impl Sync for BitLinear {}
