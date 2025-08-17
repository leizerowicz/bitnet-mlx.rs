//! Activation quantization module for BitNet models
//! 
//! This module provides specialized quantization for neural network activations,
//! including dynamic quantization and calibration-based approaches.

use super::{Quantizer, QuantizationConfig, QuantizationStats, QuantizationResult, QuantizationPrecision, QuantizationStrategy, CalibrationQuantizer};
use crate::quantization::utils::QuantizationError;
use candle_core::{Tensor, DType, Device, Shape};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration specific to activation quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationQuantizationConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Moving average window size for dynamic scaling
    pub moving_average_window: usize,
    /// Percentile for outlier detection (e.g., 99.9)
    pub outlier_percentile: f32,
    /// Whether to use per-token quantization
    pub per_token: bool,
    /// Calibration warmup steps
    pub calibration_warmup: usize,
    /// Exponential moving average decay factor
    pub ema_decay: f32,
    /// Whether to quantize attention scores
    pub quantize_attention: bool,
}

impl Default for ActivationQuantizationConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            moving_average_window: 100,
            outlier_percentile: 99.9,
            per_token: false,
            calibration_warmup: 50,
            ema_decay: 0.99,
            quantize_attention: true,
        }
    }
}

impl ActivationQuantizationConfig {
    /// Create a BitNet-specific activation quantization configuration
    pub fn bitnet() -> Self {
        Self {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Dynamic,
                per_channel: false,
                clip_threshold: None,
                qat_enabled: false,
                calibration_size: None,
            },
            moving_average_window: 100,
            outlier_percentile: 99.9,
            per_token: false,
            calibration_warmup: 50,
            ema_decay: 0.99,
            quantize_attention: true,
        }
    }

    /// Validate the activation quantization configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.moving_average_window == 0 {
            return Err(QuantizationError::ConfigurationError("Moving average window cannot be zero".to_string()));
        }

        if self.outlier_percentile <= 0.0 || self.outlier_percentile > 100.0 {
            return Err(QuantizationError::ConfigurationError("Outlier percentile must be in range (0, 100]".to_string()));
        }

        if self.ema_decay < 0.0 || self.ema_decay > 1.0 {
            return Err(QuantizationError::ConfigurationError("EMA decay must be in range [0, 1]".to_string()));
        }

        Ok(())
    }
}

/// Quantized activation representation
#[derive(Debug, Clone)]
pub struct QuantizedActivation {
    /// Quantized activation values
    pub values: Tensor,
    /// Dynamic scaling factors
    pub scales: Tensor,
    /// Zero points for asymmetric quantization
    pub zero_points: Option<Tensor>,
    /// Original shape of the activation tensor
    pub original_shape: Shape,
    /// Data type of quantized values
    pub quantized_dtype: DType,
    /// Quantization configuration used
    pub config: ActivationQuantizationConfig,
    /// Quantization statistics
    pub stats: QuantizationStats,
    /// Sequence length (for attention mechanisms)
    pub sequence_length: Option<usize>,
}

impl QuantizedActivation {
    /// Create a new quantized activation
    pub fn new(
        values: Tensor,
        scales: Tensor,
        zero_points: Option<Tensor>,
        original_shape: Shape,
        quantized_dtype: DType,
        config: ActivationQuantizationConfig,
        stats: QuantizationStats,
        sequence_length: Option<usize>,
    ) -> Self {
        Self {
            values,
            scales,
            zero_points,
            original_shape,
            quantized_dtype,
            config,
            stats,
            sequence_length,
        }
    }

    /// Check if this is an attention activation
    pub fn is_attention_activation(&self) -> bool {
        self.sequence_length.is_some() && self.config.quantize_attention
    }

    /// Get the effective bit width used
    pub fn effective_bit_width(&self) -> f32 {
        match self.config.base.precision {
            QuantizationPrecision::OneFiveFiveBit => 1.58,
            QuantizationPrecision::OneBit => 1.0,
            QuantizationPrecision::TwoBit => 2.0,
            QuantizationPrecision::FourBit => 4.0,
            QuantizationPrecision::EightBit => 8.0,
        }
    }

    /// Get the memory footprint of the quantized activation
    pub fn memory_footprint(&self) -> usize {
        let values_size = self.values.elem_count() * self.quantized_dtype.size_in_bytes();
        let scales_size = self.scales.elem_count() * self.scales.dtype().size_in_bytes();
        let zero_points_size = self.zero_points
            .as_ref()
            .map(|zp| zp.elem_count() * zp.dtype().size_in_bytes())
            .unwrap_or(0);
        
        values_size + scales_size + zero_points_size
    }
}

/// Trait for activation quantization operations
pub trait ActivationQuantizer: Quantizer<Input = Tensor, Output = QuantizedActivation, Config = ActivationQuantizationConfig, Error = QuantizationError> {
    /// Quantize activations with dynamic scaling
    fn quantize_dynamic(&mut self, activations: &Tensor) -> QuantizationResult<QuantizedActivation>;
    
    /// Quantize attention scores specifically
    fn quantize_attention(&self, attention_scores: &Tensor, sequence_length: usize) -> QuantizationResult<QuantizedActivation>;
    
    /// Update dynamic quantization parameters
    fn update_dynamic_params(&mut self, activations: &Tensor) -> QuantizationResult<()>;
    
    /// Get current dynamic scaling factors
    fn get_dynamic_scales(&self) -> QuantizationResult<Tensor>;
    
    /// Reset dynamic quantization state
    fn reset_dynamic_state(&mut self);
    
    /// Validate activation tensor for quantization
    fn validate_activations(&self, activations: &Tensor) -> QuantizationResult<()>;
}

/// Dynamic activation quantizer with calibration support
#[derive(Debug)]
pub struct DynamicActivationQuantizer {
    config: ActivationQuantizationConfig,
    device: Device,
    stats: QuantizationStats,
    /// Moving average of activation scales
    scale_history: VecDeque<f32>,
    /// Current exponential moving average scale
    ema_scale: Option<f32>,
    /// Calibration data for static quantization
    calibration_data: Vec<Tensor>,
    /// Whether calibration is complete
    calibrated: bool,
    /// Step counter for warmup
    step_count: usize,
}

impl DynamicActivationQuantizer {
    /// Create a new dynamic activation quantizer
    pub fn new(config: ActivationQuantizationConfig, device: Device) -> Self {
        let window_size = config.moving_average_window;
        Self {
            config,
            device,
            stats: QuantizationStats::default(),
            scale_history: VecDeque::with_capacity(window_size),
            ema_scale: None,
            calibration_data: Vec::new(),
            calibrated: false,
            step_count: 0,
        }
    }

    /// Compute dynamic scale based on activation statistics
    fn compute_dynamic_scale(&mut self, activations: &Tensor) -> QuantizationResult<f32> {
        let abs_max = activations.abs()?.max_all()?.to_scalar::<f32>()?;
        
        // Update exponential moving average
        let current_scale = match self.config.base.precision {
            QuantizationPrecision::OneFiveFiveBit => abs_max / 1.0, // Scale for ternary
            QuantizationPrecision::EightBit => abs_max / 127.0,
            _ => abs_max / ((1 << (self.effective_bit_width() as u32 - 1)) - 1) as f32,
        };

        self.ema_scale = Some(match self.ema_scale {
            Some(prev) => self.config.ema_decay * prev + (1.0 - self.config.ema_decay) * current_scale,
            None => current_scale,
        });

        // Update scale history
        if self.scale_history.len() >= self.config.moving_average_window {
            self.scale_history.pop_front();
        }
        self.scale_history.push_back(current_scale);

        Ok(self.ema_scale.unwrap())
    }

    /// Quantize to 8-bit integers
    fn quantize_int8(&self, activations: &Tensor, scale: f32) -> QuantizationResult<Tensor> {
        let scaled = activations.div(&Tensor::new(scale, &self.device)?)?;
        let clamped = scaled.clamp(0.0, 255.0)?;
        let quantized = clamped.round()?.to_dtype(DType::U8)?;
        Ok(quantized)
    }

    /// Quantize to ternary values for 1.58-bit
    fn quantize_ternary_activation(&self, activations: &Tensor, scale: f32) -> QuantizationResult<Tensor> {
        let scaled = activations.div(&Tensor::new(scale, &self.device)?)?;
        
        // Use a threshold-based approach for activations
        let threshold = 0.5;
        let pos_mask = scaled.gt(&Tensor::new(threshold, &self.device)?)?;
        let neg_mask = scaled.lt(&Tensor::new(-threshold, &self.device)?)?;
        
        let pos_values = pos_mask.to_dtype(activations.dtype())?;
        let neg_values = neg_mask.to_dtype(activations.dtype())?.neg()?;
        let quantized = pos_values.add(&neg_values)?;
        
        Ok(quantized)
    }

    /// Get effective bit width for the current precision
    fn effective_bit_width(&self) -> f32 {
        match self.config.base.precision {
            QuantizationPrecision::OneFiveFiveBit => 1.58,
            QuantizationPrecision::OneBit => 1.0,
            QuantizationPrecision::TwoBit => 2.0,
            QuantizationPrecision::FourBit => 4.0,
            QuantizationPrecision::EightBit => 8.0,
        }
    }
}

impl Quantizer for DynamicActivationQuantizer {
    type Input = Tensor;
    type Output = QuantizedActivation;
    type Config = ActivationQuantizationConfig;
    type Error = QuantizationError;

    fn quantize(&self, activations: &Tensor) -> QuantizationResult<QuantizedActivation> {
        self.validate_input(activations)?;
        
        // Use current EMA scale or compute a new one
        let scale = self.ema_scale.unwrap_or_else(|| {
            activations.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap_or(1.0)
        });

        let (quantized_values, quantized_dtype) = match self.config.base.precision {
            QuantizationPrecision::OneFiveFiveBit => {
                (self.quantize_ternary_activation(activations, scale)?, DType::U8)
            }
            QuantizationPrecision::EightBit => {
                (self.quantize_int8(activations, scale)?, DType::U8)
            }
            _ => return Err(QuantizationError::UnsupportedPrecision(format!("{:?}", self.config.base.precision))),
        };

        let scales = Tensor::new(scale, &self.device)?;
        
        let stats = QuantizationStats {
            elements_count: activations.elem_count(),
            scale_factor: scale,
            compression_ratio: 32.0 / self.effective_bit_width(),
            ..Default::default()
        };

        Ok(QuantizedActivation::new(
            quantized_values,
            scales,
            None,
            activations.shape().clone(),
            quantized_dtype,
            self.config.clone(),
            stats,
            None,
        ))
    }

    fn dequantize(&self, quantized: &QuantizedActivation) -> QuantizationResult<Tensor> {
        let dequantized = quantized.values.to_dtype(DType::F32)?.mul(&quantized.scales)?;
        Ok(dequantized)
    }

    fn config(&self) -> &ActivationQuantizationConfig {
        &self.config
    }

    fn validate_input(&self, activations: &Tensor) -> QuantizationResult<()> {
        if activations.rank() < 1 {
            return Err(QuantizationError::InvalidInput("Activation tensor must have at least 1 dimension".to_string()));
        }
        
        if activations.dtype() != DType::F32 && activations.dtype() != DType::F16 {
            return Err(QuantizationError::InvalidInput("Activation tensor must be float type".to_string()));
        }
        
        Ok(())
    }

    fn get_stats(&self) -> QuantizationStats {
        self.stats.clone()
    }
}

impl ActivationQuantizer for DynamicActivationQuantizer {
    fn quantize_dynamic(&mut self, activations: &Tensor) -> QuantizationResult<QuantizedActivation> {
        self.step_count += 1;
        
        // Update dynamic parameters
        self.update_dynamic_params(activations)?;
        
        // Perform quantization
        self.quantize(activations)
    }

    fn quantize_attention(&self, attention_scores: &Tensor, sequence_length: usize) -> QuantizationResult<QuantizedActivation> {
        if !self.config.quantize_attention {
            return Err(QuantizationError::InvalidInput("Attention quantization is disabled".to_string()));
        }

        // Attention scores typically have different characteristics
        // Use a specialized approach for attention quantization
        let mut result = self.quantize(attention_scores)?;
        result.sequence_length = Some(sequence_length);
        
        Ok(result)
    }

    fn update_dynamic_params(&mut self, activations: &Tensor) -> QuantizationResult<()> {
        let _scale = self.compute_dynamic_scale(activations)?;
        
        // Store calibration data during warmup
        if self.step_count <= self.config.calibration_warmup {
            if self.calibration_data.len() < self.config.calibration_warmup {
                self.calibration_data.push(activations.clone());
            }
        } else if !self.calibrated {
            self.calibrated = true;
        }
        
        Ok(())
    }

    fn get_dynamic_scales(&self) -> QuantizationResult<Tensor> {
        let scale = self.ema_scale.unwrap_or(1.0);
        Ok(Tensor::new(scale, &self.device)?)
    }

    fn reset_dynamic_state(&mut self) {
        self.scale_history.clear();
        self.ema_scale = None;
        self.calibration_data.clear();
        self.calibrated = false;
        self.step_count = 0;
    }

    fn validate_activations(&self, activations: &Tensor) -> QuantizationResult<()> {
        self.validate_input(activations)
    }
}

impl CalibrationQuantizer for DynamicActivationQuantizer {
    type CalibrationData = Tensor;

    fn calibrate(&mut self, data: &[Tensor]) -> QuantizationResult<()> {
        if data.is_empty() {
            return Err(QuantizationError::InvalidInput("Calibration data cannot be empty".to_string()));
        }

        // Compute statistics across all calibration data
        let mut all_scales = Vec::new();
        
        for tensor in data {
            let abs_max = tensor.abs()?.max_all()?.to_scalar::<f32>()?;
            all_scales.push(abs_max);
        }

        // Use median or percentile-based scale
        all_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_idx = (all_scales.len() as f32 * self.config.outlier_percentile / 100.0) as usize;
        let calibrated_scale = all_scales[percentile_idx.min(all_scales.len() - 1)];

        self.ema_scale = Some(calibrated_scale);
        self.calibrated = true;

        Ok(())
    }

    fn needs_calibration(&self) -> bool {
        !self.calibrated || self.step_count <= self.config.calibration_warmup
    }

    fn reset_calibration(&mut self) {
        self.reset_dynamic_state();
    }
}

/// Factory function to create activation quantizers
pub fn create_activation_quantizer(config: ActivationQuantizationConfig) -> QuantizationResult<Box<dyn ActivationQuantizer>> {
    // Validate configuration first
    config.validate()?;
    
    let device = Device::Cpu; // Default to CPU, can be configured
    Ok(Box::new(DynamicActivationQuantizer::new(config, device)))
}

/// Essential BitNet function: Quantize activations using absolute maximum scaling
///
/// This function implements the core BitNet activation quantization using
/// the absolute maximum value for dynamic scaling. This is crucial for
/// maintaining activation ranges during inference.
///
/// # Arguments
/// * `activations` - Input activation tensor to quantize
/// * `device` - Device to perform computation on
/// * `precision` - Quantization precision (default: 1.58-bit)
///
/// # Returns
/// * `QuantizationResult<QuantizedActivation>` - Quantized activation with dynamic scaling
///
/// # Example
/// ```rust,no_run
/// use bitnet_quant::quantization::activations::absmax_quantize_activations;
/// use bitnet_quant::quantization::QuantizationPrecision;
/// use candle_core::{Tensor, Device};
///
/// let device = Device::Cpu;
/// let activations = Tensor::randn(0.0, 1.0, (32, 128), &device).unwrap();
/// let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
/// ```
pub fn absmax_quantize_activations(
    activations: &Tensor,
    device: &Device,
    precision: Option<QuantizationPrecision>
) -> QuantizationResult<QuantizedActivation> {
    let precision = precision.unwrap_or(QuantizationPrecision::OneFiveFiveBit);
    
    // Create configuration for absolute maximum scaling
    let config = ActivationQuantizationConfig {
        base: QuantizationConfig {
            precision,
            strategy: QuantizationStrategy::Dynamic,
            per_channel: false,
            clip_threshold: None,
            qat_enabled: false,
            calibration_size: None,
        },
        moving_average_window: 1, // Use current value only for absmax
        outlier_percentile: 100.0, // No outlier clipping for absmax
        per_token: false,
        calibration_warmup: 0, // No warmup needed for absmax
        ema_decay: 1.0, // No smoothing for absmax
        quantize_attention: true,
    };
    
    let quantizer = DynamicActivationQuantizer::new(config.clone(), device.clone());
    
    // Validate input activations
    quantizer.validate_input(activations)?;
    
    // Compute absolute maximum for scaling
    let abs_max = activations.abs()?.max_all()?.to_scalar::<f32>()?;
    
    // Prevent division by zero
    let scale = if abs_max < f32::EPSILON {
        1.0
    } else {
        match precision {
            QuantizationPrecision::OneFiveFiveBit => abs_max, // Scale for ternary range [-1, 1]
            QuantizationPrecision::EightBit => abs_max / 127.0, // Scale for int8 range [-127, 127]
            QuantizationPrecision::FourBit => abs_max / 7.0, // Scale for 4-bit range [-7, 7]
            _ => abs_max / ((1 << (get_effective_bits(precision) as u32 - 1)) - 1) as f32,
        }
    };
    
    // Quantize based on precision
    let (quantized_values, quantized_dtype) = match precision {
        QuantizationPrecision::OneFiveFiveBit => {
            // Ternary quantization for 1.58-bit
            let scale_tensor = Tensor::new(scale, device)?.broadcast_as(activations.shape())?;
            let scaled = activations.div(&scale_tensor)?;
            let threshold = 0.5f32; // Standard threshold for activations
            
            let threshold_tensor = Tensor::new(threshold, device)?.broadcast_as(activations.shape())?;
            let neg_threshold_tensor = Tensor::new(-threshold, device)?.broadcast_as(activations.shape())?;
            let pos_mask = scaled.gt(&threshold_tensor)?;
            let neg_mask = scaled.lt(&neg_threshold_tensor)?;
            
            let pos_values = pos_mask.to_dtype(activations.dtype())?;
            let neg_values = neg_mask.to_dtype(activations.dtype())?.neg()?;
            let quantized = pos_values.add(&neg_values)?;
            
            (quantized, DType::U8)
        }
        QuantizationPrecision::EightBit => {
            // 8-bit quantization
            let scale_tensor = Tensor::new(scale, device)?.broadcast_as(activations.shape())?;
            let scaled = activations.div(&scale_tensor)?;
            let clamped = scaled.clamp(-127.0, 127.0)?;
            let quantized = clamped.round()?.to_dtype(DType::U8)?;
            (quantized, DType::U8)
        }
        _ => {
            return Err(QuantizationError::UnsupportedPrecision(format!("{:?}", precision)));
        }
    };
    
    let scales = Tensor::new(scale, device)?;
    
    // Compute quantization statistics
    let scales_broadcast = scales.broadcast_as(quantized_values.shape())?;
    let dequantized = quantized_values.to_dtype(DType::F32)?.mul(&scales_broadcast)?;
    let diff = activations.sub(&dequantized)?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
    
    let stats = QuantizationStats {
        elements_count: activations.elem_count(),
        quantization_error: mse,
        compression_ratio: 32.0 / get_effective_bits(precision),
        min_value: activations.min_all()?.to_scalar::<f32>()?,
        max_value: activations.max_all()?.to_scalar::<f32>()?,
        scale_factor: scale,
        zero_point: None, // Symmetric quantization
    };
    
    Ok(QuantizedActivation::new(
        quantized_values,
        scales,
        None, // No zero points for symmetric quantization
        activations.shape().clone(),
        quantized_dtype,
        config,
        stats,
        None, // No sequence length specified
    ))
}

/// Helper function to get effective bit width for different precisions
fn get_effective_bits(precision: QuantizationPrecision) -> f32 {
    match precision {
        QuantizationPrecision::OneFiveFiveBit => 1.58,
        QuantizationPrecision::OneBit => 1.0,
        QuantizationPrecision::TwoBit => 2.0,
        QuantizationPrecision::FourBit => 4.0,
        QuantizationPrecision::EightBit => 8.0,
    }
}

/// Utility functions for activation quantization
pub mod activation_utils {
    use super::*;

    /// Analyze activation patterns for optimal quantization
    pub fn analyze_activation_patterns(activations: &[Tensor]) -> QuantizationResult<ActivationPatternAnalysis> {
        if activations.is_empty() {
            return Err(QuantizationError::InvalidInput("No activation data provided".to_string()));
        }

        let mut min_vals = Vec::new();
        let mut max_vals = Vec::new();
        let mut mean_vals = Vec::new();

        for activation in activations {
            min_vals.push(activation.min_all()?.to_scalar::<f32>()?);
            max_vals.push(activation.max_all()?.to_scalar::<f32>()?);
            mean_vals.push(activation.mean_all()?.to_scalar::<f32>()?);
        }

        let global_min = min_vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let global_max = max_vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_mean = mean_vals.iter().sum::<f32>() / mean_vals.len() as f32;

        Ok(ActivationPatternAnalysis {
            global_min,
            global_max,
            average_mean: avg_mean,
            dynamic_range: global_max - global_min,
            recommended_scale: global_max.abs().max(global_min.abs()),
        })
    }

    /// Compute optimal quantization parameters for attention
    pub fn compute_attention_quantization_params(
        attention_scores: &Tensor,
        sequence_length: usize,
    ) -> QuantizationResult<AttentionQuantParams> {
        // Attention scores are typically in [0, 1] after softmax
        let max_val = attention_scores.max_all()?.to_scalar::<f32>()?;
        let min_val = attention_scores.min_all()?.to_scalar::<f32>()?;
        
        // For attention, we often want to preserve precision around 0
        let scale = max_val.max(min_val.abs());
        
        Ok(AttentionQuantParams {
            scale,
            sequence_length,
            is_causal: sequence_length > 1, // Heuristic
            sparsity: compute_attention_sparsity(attention_scores)?,
        })
    }

    fn compute_attention_sparsity(attention: &Tensor) -> QuantizationResult<f32> {
        let threshold = 0.01; // 1% threshold for attention sparsity
        let low_attention = attention.lt(&Tensor::new(threshold, attention.device())?)?;
        let sparsity = low_attention.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f32>()?;
        Ok(sparsity)
    }
}

/// Activation pattern analysis results
#[derive(Debug, Clone)]
pub struct ActivationPatternAnalysis {
    pub global_min: f32,
    pub global_max: f32,
    pub average_mean: f32,
    pub dynamic_range: f32,
    pub recommended_scale: f32,
}

/// Attention quantization parameters
#[derive(Debug, Clone)]
pub struct AttentionQuantParams {
    pub scale: f32,
    pub sequence_length: usize,
    pub is_causal: bool,
    pub sparsity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_activation_quantization_config_default() {
        let config = ActivationQuantizationConfig::default();
        assert_eq!(config.moving_average_window, 100);
        assert_eq!(config.outlier_percentile, 99.9);
        assert!(!config.per_token);
        assert!(config.quantize_attention);
    }

    #[test]
    fn test_dynamic_activation_quantizer_creation() {
        let config = ActivationQuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = DynamicActivationQuantizer::new(config, device);
        assert!(!quantizer.calibrated);
        assert_eq!(quantizer.step_count, 0);
    }

    #[test]
    fn test_quantized_activation_bit_width() {
        let device = Device::Cpu;
        let values = Tensor::zeros((10, 10), DType::U8, &device).unwrap();
        let scales = Tensor::ones((1,), DType::F32, &device).unwrap();
        let shape = Shape::from_dims(&[10, 10]);
        let config = ActivationQuantizationConfig::default();
        let stats = QuantizationStats::default();
        
        let quantized = QuantizedActivation::new(
            values, scales, None, shape, DType::U8, config, stats, None
        );
        
        assert_eq!(quantized.effective_bit_width(), 1.58);
    }

    #[test]
    fn test_absmax_quantize_activations_basic() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap()
            .reshape((2, 3)).unwrap();
        
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // Check that quantized values are ternary for 1.58-bit precision
        let quantized_data = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &quantized_data {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, "Value {} is not ternary", val);
        }
        
        // Check that we have scaling factors
        assert!(quantized.scales.elem_count() > 0);
        
        // Check compression ratio
        assert!(quantized.stats.compression_ratio > 1.0);
        
        // Check that original shape is preserved
        assert_eq!(quantized.original_shape, *activations.shape());
    }

    #[test]
    fn test_absmax_quantize_activations_8bit() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[100.0f32, -50.0, 25.0, -75.0], &device).unwrap();
        
        let quantized = absmax_quantize_activations(
            &activations,
            &device,
            Some(QuantizationPrecision::EightBit)
        ).unwrap();
        
        // Check that quantized dtype is correct
        assert_eq!(quantized.quantized_dtype, DType::U8);
        
        // Check that values are in valid uint8 range
        // Convert to F32 first since quantized values are stored as U8
        let quantized_data = quantized.values.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        for &val in &quantized_data {
            assert!(val >= -127.0 && val <= 127.0);
        }
    }

    #[test]
    fn test_absmax_quantize_activations_preserves_signs() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[3.0f32, -3.0, 0.1, -0.1], &device).unwrap();
        
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        let quantized_data = quantized.values.to_vec1::<f32>().unwrap();
        
        // Large positive should become +1, large negative should become -1
        assert!(quantized_data[0] > 0.0); // 3.0 -> positive
        assert!(quantized_data[1] < 0.0); // -3.0 -> negative
    }

    #[test]
    fn test_absmax_quantize_activations_scaling() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[4.0f32, -2.0, 1.0, -3.0], &device).unwrap();
        
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // Check that scale factor is based on absolute maximum (4.0)
        let expected_scale = 4.0; // abs_max for 1.58-bit
        assert!((quantized.stats.scale_factor - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_absmax_quantize_activations_dequantization() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[2.0f32, -1.5, 0.5, -0.3], &device).unwrap();
        
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // Test dequantization
        let config = ActivationQuantizationConfig::default();
        let quantizer = DynamicActivationQuantizer::new(config, device);
        // Create a simple dequantization by multiplying values with scales
        let dequantized = quantized.values.to_dtype(DType::F32).unwrap().mul(&quantized.scales.broadcast_as(quantized.values.shape()).unwrap()).unwrap();
        
        // Check that dequantized tensor has same shape
        assert_eq!(dequantized.shape(), activations.shape());
        
        // Check that quantization error is reasonable
        assert!(quantized.stats.quantization_error < 2.0);
    }

    #[test]
    fn test_absmax_quantize_activations_statistics() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0], &device).unwrap();
        
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // Check statistics
        assert_eq!(quantized.stats.elements_count, 6);
        assert!(quantized.stats.scale_factor > 0.0);
        assert!(quantized.stats.compression_ratio > 1.0);
        assert!(quantized.stats.min_value <= quantized.stats.max_value);
    }

    #[test]
    fn test_absmax_quantize_activations_edge_cases() {
        let device = Device::Cpu;
        
        // Test with all zeros
        let zeros = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let quantized_zeros = absmax_quantize_activations(&zeros, &device, None).unwrap();
        // Flatten the 2D tensor to 1D before converting to vec
        let quantized_data = quantized_zeros.values.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        for &val in &quantized_data {
            assert_eq!(val, 0.0);
        }
        
        // Test with very small values
        let small_vals = Tensor::new(&[1e-6f32, -1e-6, 1e-7, -1e-7], &device).unwrap();
        let quantized_small = absmax_quantize_activations(&small_vals, &device, None).unwrap();
        // Should handle small values gracefully
        assert!(quantized_small.stats.scale_factor >= 0.0);
    }

    #[test]
    fn test_absmax_quantize_activations_different_precisions() {
        let device = Device::Cpu;
        let activations = Tensor::new(&[1.0f32, -0.5, 0.25, -0.75], &device).unwrap();
        
        // Test 1.58-bit precision
        let quantized_158 = absmax_quantize_activations(
            &activations,
            &device,
            Some(QuantizationPrecision::OneFiveFiveBit)
        ).unwrap();
        assert_eq!(quantized_158.effective_bit_width(), 1.58);
        
        // Test 8-bit precision
        let quantized_8bit = absmax_quantize_activations(
            &activations,
            &device,
            Some(QuantizationPrecision::EightBit)
        ).unwrap();
        assert_eq!(quantized_8bit.effective_bit_width(), 8.0);
    }

    #[test]
    fn test_get_effective_bits_helper() {
        assert_eq!(get_effective_bits(QuantizationPrecision::OneFiveFiveBit), 1.58);
        assert_eq!(get_effective_bits(QuantizationPrecision::OneBit), 1.0);
        assert_eq!(get_effective_bits(QuantizationPrecision::TwoBit), 2.0);
        assert_eq!(get_effective_bits(QuantizationPrecision::FourBit), 4.0);
        assert_eq!(get_effective_bits(QuantizationPrecision::EightBit), 8.0);
    }
}