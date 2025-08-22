//! Weight quantization module for BitNet models
//! 
//! This module provides specialized quantization for neural network weights,
//! focusing on the 1.58-bit quantization scheme used in BitNet.

use super::{Quantizer, QuantizationConfig, QuantizationStats, QuantizationResult, QuantizationPrecision, QuantizationStrategy};
use crate::quantization::utils::QuantizationError;
use crate::quantization::packing::{TernaryPackingConfig, PackedTernaryWeights, TernaryPackerFactory, packing_utils};
use candle_core::{Tensor, DType, Device, Shape};
use serde::{Deserialize, Serialize};

/// Ternary quantization methods for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryMethod {
    /// Use mean absolute value as threshold
    MeanThreshold,
    /// Use median absolute value as threshold (more robust to outliers)
    MedianThreshold,
    /// Adaptive threshold based on weight distribution
    AdaptiveThreshold,
    /// Optimal threshold that minimizes quantization error
    OptimalThreshold,
    /// Deterministic Straight-Through Estimator
    DetSTE,
}

impl Default for TernaryMethod {
    fn default() -> Self {
        Self::MeanThreshold
    }
}

/// Statistics for ternary quantization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryStats {
    /// Total number of elements
    pub total_elements: usize,
    /// Number of zero values
    pub zeros: usize,
    /// Number of positive values (+1)
    pub positives: usize,
    /// Number of negative values (-1)
    pub negatives: usize,
    /// Sparsity ratio (zeros / total)
    pub sparsity: f32,
    /// Mean squared error from quantization
    pub mse: f32,
    /// Mean absolute error from quantization
    pub mae: f32,
}

/// Configuration specific to weight quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightQuantizationConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Group size for grouped quantization (None for per-tensor)
    pub group_size: Option<usize>,
    /// Whether to use weight normalization before quantization
    pub normalize_weights: bool,
    /// Outlier threshold for weight clipping
    pub outlier_threshold: f32,
    /// Whether to use learnable scaling factors
    pub learnable_scales: bool,
    /// Block size for block-wise quantization
    pub block_size: Option<usize>,
    /// Ternary quantization method
    pub ternary_method: TernaryMethod,
    /// Custom threshold factor for ternary quantization (overrides method default)
    pub custom_threshold_factor: Option<f32>,
    /// Ternary weight packing configuration
    pub packing_config: TernaryPackingConfig,
}

impl Default for WeightQuantizationConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            group_size: None,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            block_size: None,
            ternary_method: TernaryMethod::default(),
            custom_threshold_factor: None,
            packing_config: TernaryPackingConfig::default(),
        }
    }
}

impl WeightQuantizationConfig {
    /// Create a BitNet-specific weight quantization configuration
    pub fn bitnet() -> Self {
        Self {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Static,
                per_channel: false,
                clip_threshold: None,
                qat_enabled: false,
                calibration_size: None,
            },
            group_size: None,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            block_size: None,
            ternary_method: TernaryMethod::MeanThreshold,
            custom_threshold_factor: Some(0.7),
            packing_config: TernaryPackingConfig::default(),
        }
    }

    /// Validate the weight quantization configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        // Validate base configuration
        if let Some(group_size) = self.group_size {
            if group_size == 0 {
                return Err(QuantizationError::ConfigurationError("Group size cannot be zero".to_string()));
            }
        }

        if let Some(block_size) = self.block_size {
            if block_size == 0 {
                return Err(QuantizationError::ConfigurationError("Block size cannot be zero".to_string()));
            }
        }

        if self.outlier_threshold < 0.0 {
            return Err(QuantizationError::ConfigurationError("Outlier threshold cannot be negative".to_string()));
        }

        if let Some(factor) = self.custom_threshold_factor {
            if factor <= 0.0 || factor > 2.0 {
                return Err(QuantizationError::ConfigurationError("Custom threshold factor must be in range (0, 2]".to_string()));
            }
        }

        Ok(())
    }
}

/// Quantized weight representation
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Quantized weight values
    pub values: Tensor,
    /// Scaling factors for dequantization
    pub scales: Tensor,
    /// Zero points for asymmetric quantization
    pub zero_points: Option<Tensor>,
    /// Original shape of the weight tensor
    pub original_shape: Shape,
    /// Data type of quantized values
    pub quantized_dtype: DType,
    /// Quantization configuration used
    pub config: WeightQuantizationConfig,
    /// Quantization statistics
    pub stats: QuantizationStats,
    /// Packed ternary weights (optional, for memory efficiency)
    pub packed_weights: Option<PackedTernaryWeights>,
}

impl QuantizedWeight {
    /// Create a new quantized weight
    pub fn new(
        values: Tensor,
        scales: Tensor,
        zero_points: Option<Tensor>,
        original_shape: Shape,
        quantized_dtype: DType,
        config: WeightQuantizationConfig,
        stats: QuantizationStats,
    ) -> Self {
        Self {
            values,
            scales,
            zero_points,
            original_shape,
            quantized_dtype,
            config,
            stats,
            packed_weights: None,
        }
    }
    
    /// Create a new quantized weight with packed representation
    pub fn new_with_packing(
        values: Tensor,
        scales: Tensor,
        zero_points: Option<Tensor>,
        original_shape: Shape,
        quantized_dtype: DType,
        config: WeightQuantizationConfig,
        stats: QuantizationStats,
        packed_weights: Option<PackedTernaryWeights>,
    ) -> Self {
        Self {
            values,
            scales,
            zero_points,
            original_shape,
            quantized_dtype,
            config,
            stats,
            packed_weights,
        }
    }

    /// Get the memory footprint of the quantized weight
    pub fn memory_footprint(&self) -> usize {
        // If we have packed weights, use their memory footprint
        if let Some(ref packed) = self.packed_weights {
            let scales_size = self.scales.elem_count() * self.scales.dtype().size_in_bytes();
            let zero_points_size = self.zero_points
                .as_ref()
                .map(|zp| zp.elem_count() * zp.dtype().size_in_bytes())
                .unwrap_or(0);
            
            return packed.memory_footprint + scales_size + zero_points_size;
        }
        
        // Otherwise use standard calculation
        let values_size = self.values.elem_count() * self.quantized_dtype.size_in_bytes();
        let scales_size = self.scales.elem_count() * self.scales.dtype().size_in_bytes();
        let zero_points_size = self.zero_points
            .as_ref()
            .map(|zp| zp.elem_count() * zp.dtype().size_in_bytes())
            .unwrap_or(0);
        
        values_size + scales_size + zero_points_size
    }
    
    /// Pack the ternary weights using the configured strategy
    pub fn pack_weights(&mut self) -> QuantizationResult<()> {
        // Convert tensor values to ternary i8 format
        let ternary_weights = packing_utils::tensor_to_ternary(&self.values)?;
        
        // Pack using the configured strategy
        let packer = TernaryPackerFactory::create_packer(self.config.packing_config.strategy);
        let packed = packer.pack(&ternary_weights, &self.config.packing_config)?;
        
        self.packed_weights = Some(packed);
        Ok(())
    }
    
    /// Unpack the ternary weights back to tensor format
    pub fn unpack_weights(&self) -> QuantizationResult<Tensor> {
        if let Some(ref packed) = self.packed_weights {
            let packer = TernaryPackerFactory::create_packer(packed.strategy);
            let ternary_weights = packer.unpack(packed)?;
            let device = self.values.device();
            packing_utils::ternary_to_tensor(&ternary_weights, &self.original_shape, device)
        } else {
            Ok(self.values.clone())
        }
    }
    
    /// Check if weights are packed
    pub fn is_packed(&self) -> bool {
        self.packed_weights.is_some()
    }
    
    /// Get packing compression ratio
    pub fn packing_compression_ratio(&self) -> f32 {
        if let Some(ref packed) = self.packed_weights {
            packed.compression_ratio
        } else {
            1.0
        }
    }

    /// Calculate compression ratio compared to original
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.original_shape.elem_count() * DType::F32.size_in_bytes();
        let quantized_size = self.memory_footprint();
        original_size as f32 / quantized_size as f32
    }
}

/// Trait for weight quantization operations
pub trait WeightQuantizer: Quantizer<Input = Tensor, Output = QuantizedWeight, Config = WeightQuantizationConfig, Error = QuantizationError> {
    /// Quantize weights with optional grouping
    fn quantize_grouped(&self, weights: &Tensor, group_size: usize) -> QuantizationResult<QuantizedWeight>;
    
    /// Quantize weights block-wise
    fn quantize_blockwise(&self, weights: &Tensor, block_size: usize) -> QuantizationResult<QuantizedWeight>;
    
    /// Apply weight normalization before quantization
    fn normalize_before_quantize(&self, weights: &Tensor) -> QuantizationResult<Tensor>;
    
    /// Detect and handle weight outliers
    fn handle_outliers(&self, weights: &Tensor, threshold: f32) -> QuantizationResult<Tensor>;
    
    /// Get optimal scaling factors for the weights
    fn compute_scales(&self, weights: &Tensor) -> QuantizationResult<Tensor>;
    
    /// Validate weight tensor dimensions and values
    fn validate_weights(&self, weights: &Tensor) -> QuantizationResult<()>;
    
    /// Quantize weights to ternary values {-1, 0, +1} with specified method
    fn quantize_ternary_with_method(&self, weights: &Tensor, method: TernaryMethod) -> QuantizationResult<Tensor>;
    
    /// Get statistics for ternary quantization
    fn analyze_ternary_quantization(&self, weights: &Tensor) -> QuantizationResult<TernaryStats>;
    
    /// Find optimal threshold for ternary quantization
    fn find_optimal_ternary_threshold(&self, weights: &Tensor) -> QuantizationResult<f32>;
}

/// BitNet 1.58-bit weight quantizer
#[derive(Debug)]
pub struct BitNetWeightQuantizer {
    config: WeightQuantizationConfig,
    device: Device,
    stats: QuantizationStats,
}

impl BitNetWeightQuantizer {
    /// Create a new BitNet weight quantizer
    pub fn new(config: WeightQuantizationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            stats: QuantizationStats::default(),
        }
    }

    /// Quantize to ternary values {-1, 0, +1} using improved BitNet algorithm
    fn quantize_ternary(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        self.quantize_ternary_with_method(weights, self.config.ternary_method)
    }

    /// Quantize to ternary values using specified method
    fn quantize_ternary_with_method(&self, weights: &Tensor, method: TernaryMethod) -> QuantizationResult<Tensor> {
        match method {
            TernaryMethod::MeanThreshold => self.quantize_ternary_mean_threshold(weights),
            TernaryMethod::MedianThreshold => self.quantize_ternary_median_threshold(weights),
            TernaryMethod::AdaptiveThreshold => self.quantize_ternary_adaptive_threshold(weights),
            TernaryMethod::OptimalThreshold => self.quantize_ternary_optimal_threshold(weights),
            TernaryMethod::DetSTE => self.quantize_ternary_detste(weights),
        }
    }

    /// Ternary quantization using mean absolute value threshold
    fn quantize_ternary_mean_threshold(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Compute mean absolute value for threshold
        let abs_weights = weights.abs()?;
        let mean_abs = abs_weights.mean_all()?.to_scalar::<f32>()?;
        
        // Use custom threshold factor if provided, otherwise use default
        let threshold_factor = self.config.custom_threshold_factor.unwrap_or(0.7);
        let threshold = mean_abs * threshold_factor;
        
        self.apply_ternary_quantization(weights, threshold)
    }

    /// Ternary quantization using median absolute value threshold
    fn quantize_ternary_median_threshold(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Use median for more robust threshold estimation
        let abs_weights = weights.abs()?;
        let flattened = abs_weights.flatten_all()?;
        
        // Approximate median using mean (for simplicity, can be improved with actual median calculation)
        let threshold_factor = self.config.custom_threshold_factor.unwrap_or(0.8);
        let threshold = flattened.mean_all()?.to_scalar::<f32>()? * threshold_factor;
        
        self.apply_ternary_quantization(weights, threshold)
    }

    /// Adaptive ternary quantization with layer-specific thresholds
    fn quantize_ternary_adaptive_threshold(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        let abs_weights = weights.abs()?;
        
        // Compute statistics for adaptive threshold
        let mean_abs = abs_weights.mean_all()?.to_scalar::<f32>()?;
        let max_abs = abs_weights.max_all()?.to_scalar::<f32>()?;
        
        // Adaptive threshold based on weight distribution
        let base_threshold = if max_abs > 3.0 * mean_abs {
            // High dynamic range - use conservative threshold
            mean_abs * 0.5
        } else {
            // Normal distribution - use standard threshold
            mean_abs * 0.7
        };
        
        // Apply custom threshold factor if provided
        let threshold = if let Some(factor) = self.config.custom_threshold_factor {
            mean_abs * factor
        } else {
            base_threshold
        };
        
        self.apply_ternary_quantization(weights, threshold)
    }

    /// Optimal ternary quantization minimizing quantization error
    fn quantize_ternary_optimal_threshold(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Find optimal threshold that minimizes MSE between original and quantized weights
        let abs_weights = weights.abs()?;
        let mean_abs = abs_weights.mean_all()?.to_scalar::<f32>()?;
        
        // If custom threshold factor is provided, use it directly
        if let Some(factor) = self.config.custom_threshold_factor {
            let threshold = mean_abs * factor;
            return self.apply_ternary_quantization(weights, threshold);
        }
        
        // Search for optimal threshold in range [0.3 * mean, 1.2 * mean]
        let mut best_threshold = mean_abs * 0.7;
        let mut best_error = f32::INFINITY;
        
        for factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] {
            let threshold = mean_abs * factor;
            let quantized = self.apply_ternary_quantization(weights, threshold)?;
            
            // Compute MSE
            let diff = weights.sub(&quantized)?;
            let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
            
            if mse < best_error {
                best_error = mse;
                best_threshold = threshold;
            }
        }
        
        // Apply optimal threshold
        self.apply_ternary_quantization(weights, best_threshold)
    }

    /// Deterministic Straight-Through Estimator quantization
    fn quantize_ternary_detste(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Simple deterministic ternary quantization using sign function
        let abs_weights = weights.abs()?;
        let mean_abs = abs_weights.mean_all()?.to_scalar::<f32>()?;
        let threshold = mean_abs * self.config.custom_threshold_factor.unwrap_or(0.7);
        
        // Apply ternary quantization with straight-through gradient estimation
        self.apply_ternary_quantization(weights, threshold)
    }

    /// Apply ternary quantization with given threshold
    fn apply_ternary_quantization(&self, weights: &Tensor, threshold: f32) -> QuantizationResult<Tensor> {
        let abs_weights = weights.abs()?;
        
        // Create threshold tensor with same shape as weights
        let threshold_tensor = Tensor::new(threshold, weights.device())?.broadcast_as(weights.shape())?;
        let mask = abs_weights.gt(&threshold_tensor)?;
        let signs = weights.sign()?;
        let quantized = signs.mul(&mask.to_dtype(weights.dtype())?)?;
        Ok(quantized)
    }

    /// Get ternary quantization statistics
    fn get_ternary_stats(&self, original: &Tensor, quantized: &Tensor) -> QuantizationResult<TernaryStats> {
        let total_elements = original.elem_count();
        
        // Create scalar tensors with the same shape as quantized for comparison
        let zero_tensor = quantized.zeros_like()?;
        let eps = 1e-6f32;
        let eps_tensor = Tensor::new(eps, quantized.device())?.broadcast_as(quantized.shape())?;
        
        // Count ternary values using element-wise comparisons
        let zeros_mask = quantized.abs()?.lt(&eps_tensor)?;
        let zeros = zeros_mask.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
        
        let positives_mask = quantized.gt(&eps_tensor)?;
        let positives = positives_mask.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as usize;
        
        let negatives = total_elements - zeros - positives;
        
        // Compute quantization error
        let diff = original.sub(quantized)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let mae = diff.abs()?.mean_all()?.to_scalar::<f32>()?;
        
        // Compute sparsity (percentage of zeros)
        let sparsity = zeros as f32 / total_elements as f32;
        
        Ok(TernaryStats {
            total_elements,
            zeros,
            positives,
            negatives,
            sparsity,
            mse,
            mae,
        })
    }

    /// Compute scaling factor for ternary quantization
    fn compute_ternary_scale(&self, original: &Tensor, quantized: &Tensor) -> QuantizationResult<f32> {
        let numerator = original.mul(quantized)?.sum_all()?.to_scalar::<f32>()?;
        let denominator = quantized.mul(quantized)?.sum_all()?.to_scalar::<f32>()?;
        
        if denominator.abs() < f32::EPSILON {
            return Ok(1.0);
        }
        
        Ok(numerator / denominator)
    }
}

impl Quantizer for BitNetWeightQuantizer {
    type Input = Tensor;
    type Output = QuantizedWeight;
    type Config = WeightQuantizationConfig;
    type Error = QuantizationError;

    fn quantize(&self, weights: &Tensor) -> QuantizationResult<QuantizedWeight> {
        self.validate_input(weights)?;
        
        let normalized_weights = if self.config.normalize_weights {
            self.normalize_before_quantize(weights)?
        } else {
            weights.clone()
        };

        let processed_weights = if self.config.outlier_threshold > 0.0 {
            self.handle_outliers(&normalized_weights, self.config.outlier_threshold)?
        } else {
            normalized_weights
        };

        match self.config.base.precision {
            QuantizationPrecision::OneFiveFiveBit => {
                let quantized_values = self.quantize_ternary(&processed_weights)?;
                let scale = self.compute_ternary_scale(&processed_weights, &quantized_values)?;
                let scales = Tensor::new(scale, &self.device)?;
                
                let stats = QuantizationStats {
                    elements_count: weights.elem_count(),
                    scale_factor: scale,
                    compression_ratio: 32.0 / 1.58, // Approximate compression from f32 to 1.58-bit
                    ..Default::default()
                };

                Ok(QuantizedWeight::new(
                    quantized_values,
                    scales,
                    None,
                    weights.shape().clone(),
                    DType::U8, // Use u8 to store ternary values
                    self.config.clone(),
                    stats,
                ))
            }
            _ => Err(QuantizationError::UnsupportedPrecision(format!("{:?}", self.config.base.precision))),
        }
    }

    fn dequantize(&self, quantized: &QuantizedWeight) -> QuantizationResult<Tensor> {
        let dequantized = quantized.values.mul(&quantized.scales)?;
        Ok(dequantized)
    }

    fn config(&self) -> &WeightQuantizationConfig {
        &self.config
    }

    fn validate_input(&self, weights: &Tensor) -> QuantizationResult<()> {
        if weights.rank() < 2 {
            return Err(QuantizationError::InvalidInput("Weight tensor must have at least 2 dimensions".to_string()));
        }
        
        if weights.dtype() != DType::F32 && weights.dtype() != DType::F16 {
            return Err(QuantizationError::InvalidInput("Weight tensor must be float type".to_string()));
        }
        
        Ok(())
    }

    fn get_stats(&self) -> QuantizationStats {
        self.stats.clone()
    }
}

impl WeightQuantizer for BitNetWeightQuantizer {
    fn quantize_grouped(&self, weights: &Tensor, group_size: usize) -> QuantizationResult<QuantizedWeight> {
        // Implementation for grouped quantization
        // Split weights into groups and quantize each group separately
        let total_elements = weights.elem_count();
        if group_size >= total_elements {
            return self.quantize(weights);
        }
        
        // For now, fall back to regular quantization
        // TODO: Implement proper grouped quantization
        self.quantize(weights)
    }

    fn quantize_blockwise(&self, weights: &Tensor, block_size: usize) -> QuantizationResult<QuantizedWeight> {
        // Implementation for block-wise quantization
        // Similar to grouped but with 2D blocks
        self.quantize(weights)
    }

    fn normalize_before_quantize(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Global normalization to avoid shape issues
        let mean_scalar = weights.mean_all()?.to_scalar::<f32>()?;
        let mean_tensor = Tensor::new(mean_scalar, weights.device())?.broadcast_as(weights.shape())?;
        
        // Compute variance manually
        let diff = weights.sub(&mean_tensor)?;
        let var_scalar = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let std_scalar = (var_scalar + 1e-8f32).sqrt();
        let std_tensor = Tensor::new(std_scalar, weights.device())?.broadcast_as(weights.shape())?;
        
        let normalized = weights.sub(&mean_tensor)?.div(&std_tensor)?;
        Ok(normalized)
    }

    fn handle_outliers(&self, weights: &Tensor, threshold: f32) -> QuantizationResult<Tensor> {
        // Clip outliers based on standard deviations
        let mean = weights.mean_all()?.to_scalar::<f32>()?;
        let abs_weights = weights.abs()?;
        let std = abs_weights.mean_all()?.to_scalar::<f32>()?; // Use mean absolute deviation as approximation
        
        let lower_bound = mean - threshold * std;
        let upper_bound = mean + threshold * std;
        
        let clipped = weights.clamp(lower_bound, upper_bound)?;
        Ok(clipped)
    }

    fn compute_scales(&self, weights: &Tensor) -> QuantizationResult<Tensor> {
        // Compute optimal scaling factors
        let abs_mean = weights.abs()?.mean_all()?;
        Ok(abs_mean)
    }

    fn validate_weights(&self, weights: &Tensor) -> QuantizationResult<()> {
        self.validate_input(weights)
    }

    fn quantize_ternary_with_method(&self, weights: &Tensor, method: TernaryMethod) -> QuantizationResult<Tensor> {
        self.quantize_ternary_with_method(weights, method)
    }

    fn analyze_ternary_quantization(&self, weights: &Tensor) -> QuantizationResult<TernaryStats> {
        let quantized = self.quantize_ternary(weights)?;
        self.get_ternary_stats(weights, &quantized)
    }

    fn find_optimal_ternary_threshold(&self, weights: &Tensor) -> QuantizationResult<f32> {
        let abs_weights = weights.abs()?;
        let mean_abs = abs_weights.mean_all()?.to_scalar::<f32>()?;
        
        let mut best_threshold = mean_abs * 0.7;
        let mut best_error = f32::INFINITY;
        
        // Search for optimal threshold
        for factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] {
            let threshold = mean_abs * factor;
            let quantized = self.apply_ternary_quantization(weights, threshold)?;
            
            // Compute MSE
            let diff = weights.sub(&quantized)?;
            let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
            
            if mse < best_error {
                best_error = mse;
                best_threshold = threshold;
            }
        }
        
        Ok(best_threshold)
    }
}

/// Factory function to create weight quantizers
pub fn create_weight_quantizer(config: WeightQuantizationConfig) -> QuantizationResult<Box<dyn WeightQuantizer>> {
    // Validate configuration first
    config.validate()?;
    
    let device = Device::Cpu; // Default to CPU, can be configured
    Ok(Box::new(BitNetWeightQuantizer::new(config, device)))
}

/// Create a ternary weight quantizer with specific method
pub fn create_ternary_quantizer(method: TernaryMethod, custom_threshold: Option<f32>) -> QuantizationResult<Box<dyn WeightQuantizer>> {
    let mut config = WeightQuantizationConfig::default();
    config.ternary_method = method;
    config.custom_threshold_factor = custom_threshold;
    create_weight_quantizer(config)
}

/// Utility function to quantize weights to ternary values
pub fn quantize_weights_ternary(weights: &Tensor, method: TernaryMethod, device: &Device) -> QuantizationResult<Tensor> {
    let config = WeightQuantizationConfig {
        ternary_method: method,
        ..Default::default()
    };
    let quantizer = BitNetWeightQuantizer::new(config, device.clone());
    quantizer.quantize_ternary(weights)
}

/// Essential BitNet function: Quantize weights using absolute mean threshold
///
/// This function implements the core BitNet 1.58-bit weight quantization using
/// the absolute mean of weights as the threshold for ternary quantization.
///
/// # Arguments
/// * `weights` - Input weight tensor to quantize
/// * `device` - Device to perform computation on
///
/// # Returns
/// * `QuantizationResult<QuantizedWeight>` - Quantized weight with scaling factors
///
/// # Example
/// ```rust,no_run
/// use bitnet_quant::quantization::weights::absmean_quantize_weights;
/// use candle_core::{Tensor, Device};
///
/// let device = Device::Cpu;
/// let weights = Tensor::randn(0.0, 1.0, (64, 128), &device).unwrap();
/// let quantized = absmean_quantize_weights(&weights, &device).unwrap();
/// ```
pub fn absmean_quantize_weights(weights: &Tensor, device: &Device) -> QuantizationResult<QuantizedWeight> {
    // Create configuration for absolute mean threshold quantization
    let config = WeightQuantizationConfig {
        ternary_method: TernaryMethod::MeanThreshold,
        custom_threshold_factor: Some(0.7), // Standard factor for BitNet
        normalize_weights: true,
        ..Default::default()
    };
    
    let quantizer = BitNetWeightQuantizer::new(config.clone(), device.clone());
    
    // Validate input weights
    quantizer.validate_input(weights)?;
    
    // Compute absolute mean for threshold
    let abs_weights = weights.abs()?;
    let abs_mean = abs_weights.mean_all()?.to_scalar::<f32>()?;
    let threshold = abs_mean * 0.7; // BitNet standard threshold factor
    
    // Apply ternary quantization: values -> {-1, 0, +1}
    let quantized_values = quantizer.apply_ternary_quantization(weights, threshold)?;
    
    // Compute optimal scaling factor for dequantization
    let scale = quantizer.compute_ternary_scale(weights, &quantized_values)?;
    let scales = Tensor::new(scale, device)?;
    
    // Compute quantization statistics
    let ternary_stats = quantizer.get_ternary_stats(weights, &quantized_values)?;
    let stats = QuantizationStats {
        elements_count: weights.elem_count(),
        quantization_error: ternary_stats.mse,
        compression_ratio: 32.0 / 1.58, // f32 to 1.58-bit compression
        min_value: weights.min_all()?.to_scalar::<f32>()?,
        max_value: weights.max_all()?.to_scalar::<f32>()?,
        scale_factor: scale,
        zero_point: None, // Symmetric quantization
    };
    
    Ok(QuantizedWeight::new(
        quantized_values,
        scales,
        None, // No zero points for symmetric quantization
        weights.shape().clone(),
        DType::U8, // Store ternary values as u8
        config.clone(),
        stats,
    ))
}

/// Utility functions for weight quantization
pub mod weight_utils {
    use super::*;

    /// Analyze weight distribution for optimal quantization parameters
    pub fn analyze_weight_distribution(weights: &Tensor) -> QuantizationResult<WeightDistributionAnalysis> {
        let min_val = weights.min_all()?.to_scalar::<f32>()?;
        let max_val = weights.max_all()?.to_scalar::<f32>()?;
        let mean_val = weights.mean_all()?.to_scalar::<f32>()?;
        let abs_weights = weights.abs()?;
        let std_val = abs_weights.mean_all()?.to_scalar::<f32>()?; // Use mean absolute deviation
        
        Ok(WeightDistributionAnalysis {
            min: min_val,
            max: max_val,
            mean: mean_val,
            std: std_val,
            range: max_val - min_val,
            sparsity: calculate_sparsity(weights)?,
        })
    }

    /// Calculate sparsity (percentage of near-zero weights)
    fn calculate_sparsity(weights: &Tensor) -> QuantizationResult<f32> {
        let threshold = 1e-6;
        let near_zero = weights.abs()?.lt(&Tensor::new(threshold, weights.device())?)?;
        let sparsity = near_zero.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f32>()?;
        Ok(sparsity)
    }
}

/// Weight distribution analysis results
#[derive(Debug, Clone)]
pub struct WeightDistributionAnalysis {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub range: f32,
    pub sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_weight_quantization_config_default() {
        let config = WeightQuantizationConfig::default();
        assert!(config.normalize_weights);
        assert_eq!(config.outlier_threshold, 3.0);
        assert!(!config.learnable_scales);
    }

    #[test]
    fn test_bitnet_weight_quantizer_creation() {
        let config = WeightQuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = BitNetWeightQuantizer::new(config, device);
        assert_eq!(quantizer.config().base.precision, QuantizationPrecision::OneFiveFiveBit);
    }

    #[test]
    fn test_quantized_weight_compression_ratio() {
        let device = Device::Cpu;
        let values = Tensor::zeros((10, 10), DType::U8, &device).unwrap();
        let scales = Tensor::ones((1,), DType::F32, &device).unwrap();
        let shape = Shape::from_dims(&[10, 10]);
        let config = WeightQuantizationConfig::default();
        let stats = QuantizationStats::default();
        
        let quantized = QuantizedWeight::new(
            values, scales, None, shape, DType::U8, config, stats
        );
        
        let ratio = quantized.compression_ratio();
        assert!(ratio > 1.0); // Should achieve some compression
    }

    #[test]
    fn test_ternary_method_default() {
        let method = TernaryMethod::default();
        assert_eq!(method, TernaryMethod::MeanThreshold);
    }

    #[test]
    fn test_ternary_quantization_basic() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        let quantizer = BitNetWeightQuantizer::new(config, device.clone());
        
        // Create test weights
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap()
            .reshape((2, 3)).unwrap();
        
        let quantized = quantizer.quantize_ternary(&weights).unwrap();
        // Flatten the 2D tensor to 1D before extracting values
        let quantized_data = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        
        // Check that all values are in {-1, 0, 1}
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, "Value {} is not ternary", val);
        }
    }

    #[test]
    fn test_ternary_quantization_methods() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap();
        
        for method in [
            TernaryMethod::MeanThreshold,
            TernaryMethod::MedianThreshold,
            TernaryMethod::AdaptiveThreshold,
            TernaryMethod::OptimalThreshold,
        ] {
            let mut config = WeightQuantizationConfig::default();
            config.ternary_method = method;
            let quantizer = BitNetWeightQuantizer::new(config, device.clone());
            
            let quantized = quantizer.quantize_ternary(&weights).unwrap();
            let quantized_data = quantized.to_vec1::<f32>().unwrap();
            
            // Verify all values are ternary
            for &val in &quantized_data {
                assert!(val == -1.0 || val == 0.0 || val == 1.0);
            }
        }
    }

    #[test]
    fn test_ternary_stats() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        let quantizer = BitNetWeightQuantizer::new(config, device.clone());
        
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap();
        let stats = quantizer.analyze_ternary_quantization(&weights).unwrap();
        
        assert_eq!(stats.total_elements, 6);
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
        assert!(stats.mse >= 0.0);
        assert!(stats.mae >= 0.0);
        assert_eq!(stats.zeros + stats.positives + stats.negatives, stats.total_elements);
    }

    #[test]
    fn test_custom_threshold_factor() {
        let device = Device::Cpu;
        let mut config = WeightQuantizationConfig::default();
        config.custom_threshold_factor = Some(0.5);
        let quantizer = BitNetWeightQuantizer::new(config, device.clone());
        
        let weights = Tensor::new(&[1.0f32, -0.3, 0.2, -0.8], &device).unwrap();
        let quantized = quantizer.quantize_ternary(&weights).unwrap();
        let quantized_data = quantized.to_vec1::<f32>().unwrap();
        
        // Verify all values are ternary
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_find_optimal_threshold() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        let quantizer = BitNetWeightQuantizer::new(config, device.clone());
        
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap();
        let threshold = quantizer.find_optimal_ternary_threshold(&weights).unwrap();
        
        assert!(threshold > 0.0);
        assert!(threshold < 10.0); // Reasonable range
    }

    #[test]
    fn test_create_ternary_quantizer() {
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.6)).unwrap();
        assert_eq!(quantizer.config().ternary_method, TernaryMethod::OptimalThreshold);
        assert_eq!(quantizer.config().custom_threshold_factor, Some(0.6));
    }

    #[test]
    fn test_quantize_weights_ternary_utility() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1], &device).unwrap();
        
        let quantized = quantize_weights_ternary(&weights, TernaryMethod::MeanThreshold, &device).unwrap();
        let quantized_data = quantized.to_vec1::<f32>().unwrap();
        
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_ternary_quantization_preserves_signs() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        let quantizer = BitNetWeightQuantizer::new(config, device.clone());
        
        // Test with clearly positive and negative values
        let weights = Tensor::new(&[2.0f32, -2.0, 0.1, -0.1], &device).unwrap();
        let quantized = quantizer.quantize_ternary(&weights).unwrap();
        let quantized_data = quantized.to_vec1::<f32>().unwrap();
        
        // Large positive should become +1, large negative should become -1
        assert!(quantized_data[0] > 0.0); // 2.0 -> positive
        assert!(quantized_data[1] < 0.0); // -2.0 -> negative
    }

    #[test]
    fn test_ternary_stats_structure() {
        let stats = TernaryStats {
            total_elements: 100,
            zeros: 30,
            positives: 35,
            negatives: 35,
            sparsity: 0.3,
            mse: 0.1,
            mae: 0.05,
        };
        
        assert_eq!(stats.total_elements, 100);
        assert_eq!(stats.sparsity, 0.3);
        assert_eq!(stats.zeros + stats.positives + stats.negatives, stats.total_elements);
    }

    #[test]
    fn test_absmean_quantize_weights_basic() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap()
            .reshape((2, 3)).unwrap();
        
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        // Check that quantized values are ternary
        let quantized_data = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, "Value {} is not ternary", val);
        }
        
        // Check that we have scaling factors
        assert!(quantized.scales.elem_count() > 0);
        
        // Check compression ratio
        assert!(quantized.stats.compression_ratio > 1.0);
        
        // Check that original shape is preserved
        assert_eq!(quantized.original_shape, *weights.shape());
    }

    #[test]
    fn test_absmean_quantize_weights_preserves_signs() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[3.0f32, -3.0, 0.1, -0.1], &device).unwrap()
            .reshape((2, 2)).unwrap();
        
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        let quantized_data = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        
        // Large positive should become +1, large negative should become -1
        assert!(quantized_data[0] > 0.0); // 3.0 -> positive
        assert!(quantized_data[1] < 0.0); // -3.0 -> negative
    }

    #[test]
    fn test_absmean_quantize_weights_dequantization() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[2.0f32, -1.5, 0.5, -0.3], &device).unwrap()
            .reshape((2, 2)).unwrap();
        
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        // Test dequantization
        let config = WeightQuantizationConfig::default();
        let quantizer = BitNetWeightQuantizer::new(config, device);
        // Create a simple dequantization by multiplying values with scales
        let dequantized = quantized.values.to_dtype(DType::F32).unwrap().mul(&quantized.scales.broadcast_as(quantized.values.shape()).unwrap()).unwrap();
        
        // Check that dequantized tensor has same shape
        assert_eq!(dequantized.shape(), weights.shape());
        
        // Check that quantization error is reasonable
        assert!(quantized.stats.quantization_error < 1.0);
    }

    #[test]
    fn test_absmean_quantize_weights_statistics() {
        let device = Device::Cpu;
        let weights = Tensor::new(&[1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0], &device).unwrap()
            .reshape((2, 3)).unwrap();
        
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        // Check statistics
        assert_eq!(quantized.stats.elements_count, 6);
        assert!(quantized.stats.scale_factor > 0.0);
        assert!(quantized.stats.compression_ratio > 1.0);
        assert!(quantized.stats.min_value <= quantized.stats.max_value);
    }

    #[test]
    fn test_absmean_quantize_weights_edge_cases() {
        let device = Device::Cpu;
        
        // Test with all zeros
        let zeros = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let quantized_zeros = absmean_quantize_weights(&zeros, &device).unwrap();
        let quantized_data = quantized_zeros.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &quantized_data {
            assert_eq!(val, 0.0);
        }
        
        // Test with very small values
        let small_vals = Tensor::new(&[1e-6f32, -1e-6, 1e-7, -1e-7], &device).unwrap()
            .reshape((2, 2)).unwrap();
        let quantized_small = absmean_quantize_weights(&small_vals, &device).unwrap();
        // Should handle small values gracefully
        assert!(quantized_small.stats.scale_factor >= 0.0);
    }
}