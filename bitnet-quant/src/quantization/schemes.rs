//! Configurable quantization schemes for BitNet models
//!
//! This module provides configurable quantization schemes supporting both 1-bit and 1.58-bit
//! quantization with flexible configuration options and optimized implementations.

use super::{
    QuantizationConfig, QuantizationPrecision, QuantizationResult, QuantizationStats,
    QuantizationStrategy,
};
use crate::quantization::utils::QuantizationError;
use crate::quantization::weights::TernaryMethod;
use candle_core::{DType, Device, Shape, Tensor};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Quantization scheme configuration that supports multiple precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct QuantizationSchemeConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Scheme-specific parameters
    pub scheme_params: SchemeParameters,
    /// Whether to enable adaptive thresholding
    pub adaptive_threshold: bool,
    /// Custom threshold values for different precisions
    pub custom_thresholds: Option<ThresholdConfig>,
    /// Optimization settings
    pub optimization: OptimizationConfig,
}

/// Scheme-specific parameters for different quantization types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[allow(dead_code)]
pub struct SchemeParameters {
    /// Parameters for 1-bit quantization
    pub one_bit: OneBitParams,
    /// Parameters for 1.58-bit quantization
    pub one_five_eight_bit: OneFiveEightBitParams,
    /// Parameters for higher precision quantization
    pub multi_bit: MultiBitParams,
}

/// Configuration for 1-bit quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct OneBitParams {
    /// Threshold method for binary quantization
    pub threshold_method: BinaryThresholdMethod,
    /// Custom threshold factor (overrides method default)
    pub threshold_factor: Option<f32>,
    /// Whether to use sign-based quantization
    pub sign_based: bool,
    /// Stochastic quantization probability
    pub stochastic_prob: Option<f32>,
}

/// Configuration for 1.58-bit quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct OneFiveEightBitParams {
    /// Ternary quantization method
    pub ternary_method: TernaryMethod,
    /// Threshold factor for ternary quantization
    pub threshold_factor: f32,
    /// Whether to use balanced ternary (equal +1/-1 distribution)
    pub balanced_ternary: bool,
    /// Sparsity target for zero values
    pub sparsity_target: Option<f32>,
}

/// Configuration for multi-bit quantization (2-bit, 4-bit, 8-bit)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MultiBitParams {
    /// Number of quantization levels
    pub num_levels: u32,
    /// Whether to use uniform quantization
    pub uniform_quantization: bool,
    /// Clipping range for quantization
    pub clip_range: Option<(f32, f32)>,
    /// Whether to use learnable quantization parameters
    pub learnable_params: bool,
}

/// Binary threshold methods for 1-bit quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryThresholdMethod {
    /// Use zero as threshold (sign-based)
    Zero,
    /// Use mean value as threshold
    Mean,
    /// Use median value as threshold
    Median,
    /// Use adaptive threshold based on distribution
    Adaptive,
    /// Use optimal threshold minimizing quantization error
    Optimal,
}

/// Custom threshold configuration for different precisions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ThresholdConfig {
    /// Threshold for 1-bit quantization
    pub one_bit_threshold: Option<f32>,
    /// Threshold for 1.58-bit quantization
    pub ternary_threshold: Option<f32>,
    /// Thresholds for multi-bit quantization
    pub multi_bit_thresholds: Option<Vec<f32>>,
}

/// Optimization configuration for quantization schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct OptimizationConfig {
    /// Whether to enable SIMD optimizations
    pub enable_simd: bool,
    /// Whether to use lookup tables for fast quantization
    pub use_lookup_tables: bool,
    /// Whether to enable parallel processing
    pub parallel_processing: bool,
    /// Memory optimization level (0-3)
    pub memory_optimization_level: u8,
    /// Whether to cache quantization parameters
    pub cache_parameters: bool,
}

impl Default for QuantizationSchemeConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            scheme_params: SchemeParameters::default(),
            adaptive_threshold: true,
            custom_thresholds: None,
            optimization: OptimizationConfig::default(),
        }
    }
}

impl Default for OneBitParams {
    fn default() -> Self {
        Self {
            threshold_method: BinaryThresholdMethod::Zero,
            threshold_factor: None,
            sign_based: true,
            stochastic_prob: None,
        }
    }
}

impl Default for OneFiveEightBitParams {
    fn default() -> Self {
        Self {
            ternary_method: TernaryMethod::MeanThreshold,
            threshold_factor: 0.7,
            balanced_ternary: false,
            sparsity_target: None,
        }
    }
}

impl Default for MultiBitParams {
    fn default() -> Self {
        Self {
            num_levels: 256, // 8-bit default
            uniform_quantization: true,
            clip_range: None,
            learnable_params: false,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            use_lookup_tables: false,
            parallel_processing: false,
            memory_optimization_level: 1,
            cache_parameters: true,
        }
    }
}

/// Configurable quantization scheme that supports multiple precision levels
#[derive(Debug)]
#[allow(dead_code)]
pub struct ConfigurableQuantizationScheme {
    config: QuantizationSchemeConfig,
    device: Device,
    cached_params: Option<CachedQuantizationParams>,
}

/// Cached quantization parameters for performance optimization
#[derive(Debug, Clone)]
struct CachedQuantizationParams {
    precision: QuantizationPrecision,
    threshold: f32,
    scale_factor: f32,
    zero_point: Option<i32>,
    lookup_table: Option<Vec<f32>>,
}

impl ConfigurableQuantizationScheme {
    /// Create a new configurable quantization scheme
    pub fn new(config: QuantizationSchemeConfig, device: Device) -> Self {
        Self {
            config,
            device,
            cached_params: None,
        }
    }

    /// Create a scheme for 1-bit quantization
    pub fn one_bit(device: Device) -> Self {
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneBit,
                strategy: QuantizationStrategy::Symmetric,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_bit: OneBitParams {
                    threshold_method: BinaryThresholdMethod::Zero,
                    sign_based: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        Self::new(config, device)
    }

    /// Create a scheme for 1.58-bit quantization
    pub fn one_five_eight_bit(device: Device) -> Self {
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Symmetric,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_five_eight_bit: OneFiveEightBitParams {
                    ternary_method: TernaryMethod::MeanThreshold,
                    threshold_factor: 0.7,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        Self::new(config, device)
    }

    /// Quantize tensor using the configured scheme
    pub fn quantize_tensor(&mut self, input: &Tensor) -> QuantizationResult<QuantizedTensor> {
        match self.config.base.precision {
            QuantizationPrecision::OneBit => self.quantize_one_bit(input),
            QuantizationPrecision::OneFiveFiveBit => self.quantize_one_five_eight_bit(input),
            QuantizationPrecision::TwoBit => self.quantize_multi_bit(input, 4),
            QuantizationPrecision::FourBit => self.quantize_multi_bit(input, 16),
            QuantizationPrecision::EightBit => self.quantize_multi_bit(input, 256),
        }
    }

    /// Dequantize tensor back to original precision
    pub fn dequantize_tensor(&self, quantized: &QuantizedTensor) -> QuantizationResult<Tensor> {
        match quantized.precision {
            QuantizationPrecision::OneBit => self.dequantize_one_bit(quantized),
            QuantizationPrecision::OneFiveFiveBit => self.dequantize_one_five_eight_bit(quantized),
            _ => self.dequantize_multi_bit(quantized),
        }
    }

    /// 1-bit quantization implementation
    fn quantize_one_bit(&mut self, input: &Tensor) -> QuantizationResult<QuantizedTensor> {
        let params = &self.config.scheme_params.one_bit;

        // Compute threshold based on method
        let threshold = self.compute_binary_threshold(input, params.threshold_method)?;

        // Apply binary quantization
        let quantized_values = if params.sign_based {
            // Sign-based: values -> {-1, +1}
            input.sign()?
        } else {
            // Threshold-based: values -> {0, 1} then map to {-1, +1}
            let threshold_tensor =
                Tensor::new(threshold, input.device())?.broadcast_as(input.shape())?;
            let binary_mask = input.gt(&threshold_tensor)?;
            let binary_values = binary_mask.to_dtype(DType::F32)?;
            // Map {0, 1} to {-1, +1}
            let two_tensor = Tensor::new(2.0f32, input.device())?.broadcast_as(input.shape())?;
            let one_tensor = Tensor::new(1.0f32, input.device())?.broadcast_as(input.shape())?;
            binary_values.mul(&two_tensor)?.sub(&one_tensor)?
        };

        // Compute optimal scale factor
        let scale = self.compute_optimal_scale(input, &quantized_values)?;
        let scales = Tensor::new(scale, &self.device)?;

        // Compute statistics
        let stats = self.compute_quantization_stats(input, &quantized_values, scale)?;

        Ok(QuantizedTensor {
            values: quantized_values,
            scales,
            zero_points: None,
            original_shape: input.shape().clone(),
            precision: QuantizationPrecision::OneBit,
            quantizeddtype: DType::U8,
            stats,
        })
    }

    /// 1.58-bit quantization implementation
    fn quantize_one_five_eight_bit(
        &mut self,
        input: &Tensor,
    ) -> QuantizationResult<QuantizedTensor> {
        let params = &self.config.scheme_params.one_five_eight_bit;

        // Compute ternary threshold
        let threshold = self.compute_ternary_threshold(input, params)?;

        // Apply ternary quantization: values -> {-1, 0, +1}
        let abs_input = input.abs()?;
        let threshold_tensor =
            Tensor::new(threshold, input.device())?.broadcast_as(input.shape())?;
        let mask = abs_input.gt(&threshold_tensor)?;
        let signs = input.sign()?;
        let quantized_values = signs.mul(&mask.to_dtype(input.dtype())?)?;

        // Apply sparsity target if specified
        let final_quantized = if let Some(target_sparsity) = params.sparsity_target {
            self.apply_sparsity_target(&quantized_values, target_sparsity)?
        } else {
            quantized_values
        };

        // Compute optimal scale factor
        let scale = self.compute_optimal_scale(input, &final_quantized)?;
        let scales = Tensor::new(scale, &self.device)?;

        // Compute statistics
        let stats = self.compute_quantization_stats(input, &final_quantized, scale)?;

        Ok(QuantizedTensor {
            values: final_quantized,
            scales,
            zero_points: None,
            original_shape: input.shape().clone(),
            precision: QuantizationPrecision::OneFiveFiveBit,
            quantizeddtype: DType::U8,
            stats,
        })
    }

    /// Multi-bit quantization implementation
    fn quantize_multi_bit(
        &mut self,
        input: &Tensor,
        num_levels: u32,
    ) -> QuantizationResult<QuantizedTensor> {
        let params = &self.config.scheme_params.multi_bit;

        // Determine quantization range
        let (min_val, max_val) = if let Some((clip_min, clip_max)) = params.clip_range {
            (clip_min, clip_max)
        } else {
            let min_val = input.min_all()?.to_scalar::<f32>()?;
            let max_val = input.max_all()?.to_scalar::<f32>()?;
            (min_val, max_val)
        };

        // Compute scale and zero point
        let range = max_val - min_val;
        let scale = if range > 0.0 {
            range / (num_levels - 1) as f32
        } else {
            1.0
        };
        let zero_point = if scale > 0.0 {
            (-min_val / scale).round() as i32
        } else {
            0
        };

        // Quantize values
        let scale_tensor = Tensor::new(scale, input.device())?.broadcast_as(input.shape())?;
        let zero_point_tensor =
            Tensor::new(zero_point as f32, input.device())?.broadcast_as(input.shape())?;

        let quantized_values = input
            .div(&scale_tensor)?
            .add(&zero_point_tensor)?
            .round()?
            .clamp(0.0, (num_levels - 1) as f32)?;

        // Compute statistics
        let dequantized = quantized_values
            .sub(&zero_point_tensor)?
            .mul(&scale_tensor)?;
        let stats = self.compute_quantization_stats(input, &dequantized, scale)?;

        Ok(QuantizedTensor {
            values: quantized_values,
            scales: Tensor::new(scale, &self.device)?,
            zero_points: Some(Tensor::new(zero_point as f32, &self.device)?),
            original_shape: input.shape().clone(),
            precision: self.config.base.precision,
            quantizeddtype: DType::U8, // Use U8 for all multi-bit quantization
            stats,
        })
    }

    /// Compute binary threshold for 1-bit quantization
    fn compute_binary_threshold(
        &self,
        input: &Tensor,
        method: BinaryThresholdMethod,
    ) -> QuantizationResult<f32> {
        match method {
            BinaryThresholdMethod::Zero => Ok(0.0),
            BinaryThresholdMethod::Mean => {
                let mean = input.mean_all()?.to_scalar::<f32>()?;
                Ok(mean)
            }
            BinaryThresholdMethod::Median => {
                // Approximate median with mean for simplicity
                let mean = input.mean_all()?.to_scalar::<f32>()?;
                Ok(mean)
            }
            BinaryThresholdMethod::Adaptive => {
                let mean = input.mean_all()?.to_scalar::<f32>()?;
                let abs_input = input.abs()?;
                let std = abs_input.mean_all()?.to_scalar::<f32>()?;
                Ok(mean + 0.5 * std)
            }
            BinaryThresholdMethod::Optimal => {
                // Find optimal threshold minimizing quantization error
                let mean = input.mean_all()?.to_scalar::<f32>()?;
                let std = input.abs()?.mean_all()?.to_scalar::<f32>()?;

                let mut best_threshold = 0.0;
                let mut best_error = f32::INFINITY;

                for factor in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
                    let threshold = mean + factor * std;
                    let threshold_tensor =
                        Tensor::new(threshold, input.device())?.broadcast_as(input.shape())?;
                    let binary = input.gt(&threshold_tensor)?;
                    let binary_values = binary.to_dtype(DType::F32)?;
                    let two_tensor =
                        Tensor::new(2.0f32, input.device())?.broadcast_as(input.shape())?;
                    let one_tensor =
                        Tensor::new(1.0f32, input.device())?.broadcast_as(input.shape())?;
                    let binary_quantized = binary_values.mul(&two_tensor)?.sub(&one_tensor)?;

                    let error = input
                        .sub(&binary_quantized)?
                        .sqr()?
                        .mean_all()?
                        .to_scalar::<f32>()?;
                    if error < best_error {
                        best_error = error;
                        best_threshold = threshold;
                    }
                }

                Ok(best_threshold)
            }
        }
    }

    /// Compute ternary threshold for 1.58-bit quantization
    fn compute_ternary_threshold(
        &self,
        input: &Tensor,
        params: &OneFiveEightBitParams,
    ) -> QuantizationResult<f32> {
        let abs_input = input.abs()?;
        let base_threshold = match params.ternary_method {
            TernaryMethod::MeanThreshold => abs_input.mean_all()?.to_scalar::<f32>()?,
            TernaryMethod::MedianThreshold => abs_input.mean_all()?.to_scalar::<f32>()?, // Approximate
            TernaryMethod::AdaptiveThreshold => {
                let mean = abs_input.mean_all()?.to_scalar::<f32>()?;
                let max_val = abs_input.max_all()?.to_scalar::<f32>()?;
                if max_val > 3.0 * mean {
                    mean * 0.5 // Conservative for high dynamic range
                } else {
                    mean * 0.7 // Standard threshold
                }
            }
            TernaryMethod::OptimalThreshold => {
                // Find optimal threshold minimizing MSE
                let mean = abs_input.mean_all()?.to_scalar::<f32>()?;
                let mut best_threshold = mean * 0.7;
                let mut best_error = f32::INFINITY;

                for factor in [0.3, 0.5, 0.7, 0.9, 1.1] {
                    let threshold = mean * factor;
                    let threshold_tensor = Tensor::new(threshold, input.device())?;
                    let mask = abs_input.gt(&threshold_tensor.broadcast_as(input.shape())?)?;
                    let signs = input.sign()?;
                    let ternary = signs.mul(&mask.to_dtype(input.dtype())?)?;

                    let error = input.sub(&ternary)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
                    if error < best_error {
                        best_error = error;
                        best_threshold = threshold;
                    }
                }

                best_threshold
            }
            TernaryMethod::DetSTE => {
                // Deterministic Straight-Through Estimator - use mean threshold with conservative factor
                abs_input.mean_all()?.to_scalar::<f32>()? * 0.7
            }
        };

        Ok(base_threshold * params.threshold_factor)
    }

    /// Apply sparsity target to ternary quantized values
    fn apply_sparsity_target(
        &self,
        quantized: &Tensor,
        target_sparsity: f32,
    ) -> QuantizationResult<Tensor> {
        // Count current zeros
        let eps = 1e-6f32;
        let eps_tensor = Tensor::new(eps, quantized.device())?.broadcast_as(quantized.shape())?;
        let zero_mask = quantized.abs()?.lt(&eps_tensor)?;
        let current_sparsity = zero_mask
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;

        if current_sparsity >= target_sparsity {
            return Ok(quantized.clone());
        }

        // Need to increase sparsity by setting some non-zero values to zero
        let abs_quantized = quantized.abs()?;
        let non_zero_mask = abs_quantized.gt(&eps_tensor)?;

        // For simplicity, randomly set some non-zero values to zero
        // In practice, you might want to use a more sophisticated method
        Ok(quantized.clone()) // Placeholder implementation
    }

    /// Compute optimal scale factor for quantization
    fn compute_optimal_scale(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> QuantizationResult<f32> {
        let numerator = original.mul(quantized)?.sum_all()?.to_scalar::<f32>()?;
        let denominator = quantized.mul(quantized)?.sum_all()?.to_scalar::<f32>()?;

        if denominator.abs() < f32::EPSILON {
            Ok(1.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Compute quantization statistics
    fn compute_quantization_stats(
        &self,
        original: &Tensor,
        quantized: &Tensor,
        scale: f32,
    ) -> QuantizationResult<QuantizationStats> {
        let diff = original.sub(quantized)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let min_val = original.min_all()?.to_scalar::<f32>()?;
        let max_val = original.max_all()?.to_scalar::<f32>()?;

        let compression_ratio = match self.config.base.precision {
            QuantizationPrecision::OneBit => 32.0,
            QuantizationPrecision::OneFiveFiveBit => 32.0 / 1.58,
            QuantizationPrecision::TwoBit => 16.0,
            QuantizationPrecision::FourBit => 8.0,
            QuantizationPrecision::EightBit => 4.0,
        };

        Ok(QuantizationStats {
            elements_count: original.elem_count(),
            quantization_error: mse,
            compression_ratio,
            min_value: min_val,
            max_value: max_val,
            scale_factor: scale,
            zero_point: None,
        })
    }

    /// Dequantize 1-bit quantized tensor
    fn dequantize_one_bit(&self, quantized: &QuantizedTensor) -> QuantizationResult<Tensor> {
        // Handle scalar scale factor
        let scale_tensor = if quantized.scales.dims().is_empty() {
            // Scalar scale - broadcast to match tensor shape
            quantized.scales.broadcast_as(quantized.values.shape())?
        } else {
            // Already shaped scale tensor
            quantized.scales.clone()
        };

        let dequantized = quantized.values.mul(&scale_tensor)?;
        Ok(dequantized)
    }

    /// Dequantize 1.58-bit quantized tensor
    fn dequantize_one_five_eight_bit(
        &self,
        quantized: &QuantizedTensor,
    ) -> QuantizationResult<Tensor> {
        // Handle scalar scale factor
        let scale_tensor = if quantized.scales.dims().is_empty() {
            // Scalar scale - broadcast to match tensor shape
            quantized.scales.broadcast_as(quantized.values.shape())?
        } else {
            // Already shaped scale tensor
            quantized.scales.clone()
        };

        let dequantized = quantized.values.mul(&scale_tensor)?;
        Ok(dequantized)
    }

    /// Dequantize multi-bit quantized tensor
    fn dequantize_multi_bit(&self, quantized: &QuantizedTensor) -> QuantizationResult<Tensor> {
        let mut dequantized = quantized.values.clone();

        if let Some(ref zero_points) = quantized.zero_points {
            let zero_point_tensor = if zero_points.dims().is_empty() {
                zero_points.broadcast_as(quantized.values.shape())?
            } else {
                zero_points.clone()
            };
            dequantized = dequantized.sub(&zero_point_tensor)?;
        }

        let scale_tensor = if quantized.scales.dims().is_empty() {
            quantized.scales.broadcast_as(quantized.values.shape())?
        } else {
            quantized.scales.clone()
        };

        dequantized = dequantized.mul(&scale_tensor)?;
        Ok(dequantized)
    }

    /// Get the current configuration
    pub fn config(&self) -> &QuantizationSchemeConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: QuantizationSchemeConfig) {
        self.config = config;
        self.cached_params = None; // Invalidate cache
    }

    /// Validate input tensor for quantization
    pub fn validate_input(&self, input: &Tensor) -> QuantizationResult<()> {
        if input.elem_count() == 0 {
            return Err(QuantizationError::InvalidInput("Empty tensor".to_string()));
        }

        if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(QuantizationError::InvalidInput(
                "Input must be floating point".to_string(),
            ));
        }

        Ok(())
    }
}

/// Quantized tensor representation for configurable schemes
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizedTensor {
    /// Quantized values
    pub values: Tensor,
    /// Scaling factors
    pub scales: Tensor,
    /// Zero points (for asymmetric quantization)
    pub zero_points: Option<Tensor>,
    /// Original tensor shape
    pub original_shape: Shape,
    /// Quantization precision used
    pub precision: QuantizationPrecision,
    /// Data type of quantized values
    pub quantizeddtype: DType,
    /// Quantization statistics
    pub stats: QuantizationStats,
}

impl QuantizedTensor {
    /// Get memory footprint of quantized tensor
    pub fn memory_footprint(&self) -> usize {
        let values_size = self.values.elem_count() * self.quantizeddtype.size_in_bytes();
        let scales_size = self.scales.elem_count() * self.scales.dtype().size_in_bytes();
        let zero_points_size = self
            .zero_points
            .as_ref()
            .map(|zp| zp.elem_count() * zp.dtype().size_in_bytes())
            .unwrap_or(0);

        values_size + scales_size + zero_points_size
    }

    /// Calculate compression ratio compared to original
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.original_shape.elem_count() * DType::F32.size_in_bytes();
        let quantized_size = self.memory_footprint();
        original_size as f32 / quantized_size as f32
    }
}

/// Factory for creating quantization schemes
pub struct QuantizationSchemeFactory;

impl QuantizationSchemeFactory {
    /// Create a 1-bit quantization scheme
    pub fn create_one_bit_scheme(device: Device) -> ConfigurableQuantizationScheme {
        ConfigurableQuantizationScheme::one_bit(device)
    }

    /// Create a 1.58-bit quantization scheme
    pub fn create_one_five_eight_bit_scheme(device: Device) -> ConfigurableQuantizationScheme {
        ConfigurableQuantizationScheme::one_five_eight_bit(device)
    }

    /// Create a custom quantization scheme
    pub fn create_custom_scheme(
        config: QuantizationSchemeConfig,
        device: Device,
    ) -> ConfigurableQuantizationScheme {
        ConfigurableQuantizationScheme::new(config, device)
    }

    /// Create a scheme from precision
    pub fn create_from_precision(
        precision: QuantizationPrecision,
        device: Device,
    ) -> ConfigurableQuantizationScheme {
        match precision {
            QuantizationPrecision::OneBit => Self::create_one_bit_scheme(device),
            QuantizationPrecision::OneFiveFiveBit => Self::create_one_five_eight_bit_scheme(device),
            _ => {
                let config = QuantizationSchemeConfig {
                    base: QuantizationConfig {
                        precision,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                Self::create_custom_scheme(config, device)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_one_bit_quantization_scheme() {
        let device = Device::Cpu;
        let mut scheme = ConfigurableQuantizationScheme::one_bit(device.clone());

        let input = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1], &device).unwrap();
        let quantized = scheme.quantize_tensor(&input).unwrap();

        // Check that values are binary
        let values = quantized.values.to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(val == -1.0 || val == 1.0, "Value {} is not binary", val);
        }

        // Test dequantization
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), input.shape());
    }

    #[test]
    fn test_one_five_eight_bit_quantization_scheme() {
        let device = Device::Cpu;
        let mut scheme = ConfigurableQuantizationScheme::one_five_eight_bit(device.clone());

        let input = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0], &device).unwrap();
        let quantized = scheme.quantize_tensor(&input).unwrap();

        // Check that values are ternary
        let values = quantized.values.to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(
                val == -1.0 || val == 0.0 || val == 1.0,
                "Value {} is not ternary",
                val
            );
        }

        // Test dequantization
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), input.shape());
    }

    #[test]
    fn test_binary_threshold_methods() {
        let device = Device::Cpu;
        let input = Tensor::new(&[1.0f32, -0.5, 0.3, -1.2], &device).unwrap();

        let scheme = ConfigurableQuantizationScheme::new(
            QuantizationSchemeConfig {
                scheme_params: SchemeParameters {
                    one_bit: OneBitParams {
                        threshold_method: BinaryThresholdMethod::Mean,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            device.clone(),
        );

        let threshold = scheme
            .compute_binary_threshold(&input, BinaryThresholdMethod::Mean)
            .unwrap();
        assert!(threshold != 0.0); // Should compute non-zero mean

        let zero_threshold = scheme
            .compute_binary_threshold(&input, BinaryThresholdMethod::Zero)
            .unwrap();
        assert_eq!(zero_threshold, 0.0);
    }

    #[test]
    fn test_quantization_scheme_factory() {
        let device = Device::Cpu;

        let one_bit_scheme = QuantizationSchemeFactory::create_one_bit_scheme(device.clone());
        assert_eq!(
            one_bit_scheme.config().base.precision,
            QuantizationPrecision::OneBit
        );

        let ternary_scheme =
            QuantizationSchemeFactory::create_one_five_eight_bit_scheme(device.clone());
        assert_eq!(
            ternary_scheme.config().base.precision,
            QuantizationPrecision::OneFiveFiveBit
        );

        let custom_scheme = QuantizationSchemeFactory::create_from_precision(
            QuantizationPrecision::EightBit,
            device,
        );
        assert_eq!(
            custom_scheme.config().base.precision,
            QuantizationPrecision::EightBit
        );
    }

    #[test]
    fn test_quantized_tensor_properties() {
        let device = Device::Cpu;
        let values = Tensor::zeros((10, 10), DType::U8, &device).unwrap();
        let scales = Tensor::ones((1,), DType::F32, &device).unwrap();
        let shape = Shape::from_dims(&[10, 10]);
        let stats = QuantizationStats::default();

        let quantized = QuantizedTensor {
            values,
            scales,
            zero_points: None,
            original_shape: shape,
            precision: QuantizationPrecision::OneBit,
            quantizeddtype: DType::U8,
            stats,
        };

        let memory_footprint = quantized.memory_footprint();
        assert!(memory_footprint > 0);

        let compression_ratio = quantized.compression_ratio();
        assert!(compression_ratio > 1.0);
    }

    #[test]
    fn test_scheme_config_validation() {
        let device = Device::Cpu;
        let scheme = ConfigurableQuantizationScheme::one_bit(device.clone());

        // Test valid input
        let valid_input = Tensor::new(&[1.0f32, -1.0, 0.5], &device).unwrap();
        assert!(scheme.validate_input(&valid_input).is_ok());

        // Test empty tensor
        let empty_input = Tensor::new(&[] as &[f32], &device).unwrap();
        assert!(scheme.validate_input(&empty_input).is_err());
    }
}
