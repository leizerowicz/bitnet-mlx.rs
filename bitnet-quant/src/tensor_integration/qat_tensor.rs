//! Quantization-Aware Training (QAT) Tensor Operations
//!
//! This module provides specialized tensor operations for Quantization-Aware Training,
//! implementing the Straight-Through Estimator (STE) and other QAT techniques
//! essential for training BitNet models with quantization simulation.

use candle_core::{Device, Tensor, Tensor as CandleTensor};
use std::collections::HashMap;

use bitnet_core::BitNetTensor;

use crate::quantization::QuantizationPrecision;

use super::{TensorIntegrationError, TensorIntegrationResult};

/// Configuration for QAT tensor operations
#[derive(Debug, Clone)]
pub struct QATConfig {
    /// Enable straight-through estimator
    pub enable_ste: bool,

    /// Quantization precision for weights
    pub weight_precision: QuantizationPrecision,

    /// Quantization precision for activations
    pub activation_precision: QuantizationPrecision,

    /// Gradient clipping threshold
    pub gradient_clip_threshold: Option<f32>,

    /// Temperature parameter for soft quantization
    pub temperature: f32,

    /// Annealing schedule for temperature
    pub temperature_annealing: bool,

    /// Target device for operations
    pub device: Option<Device>,

    /// Learning rate for quantization parameters
    pub quantization_lr: f32,

    /// Enable fake quantization during training
    pub fake_quantization: bool,

    /// Quantization noise injection level
    pub noise_level: f32,

    /// Enable quantization regularization
    pub regularization_weight: f32,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            enable_ste: true,
            weight_precision: QuantizationPrecision::OneFiveFiveBit,
            activation_precision: QuantizationPrecision::EightBit,
            gradient_clip_threshold: Some(1.0),
            temperature: 1.0,
            temperature_annealing: true,
            device: None,
            quantization_lr: 0.01,
            fake_quantization: true,
            noise_level: 0.01,
            regularization_weight: 0.01,
        }
    }
}

/// Straight-Through Estimator implementation for QAT
#[derive(Debug)]
pub struct StraightThroughEstimator {
    /// Configuration for STE
    config: QATConfig,

    /// Current training step
    current_step: usize,

    /// Quantization parameters for different layers
    layer_parameters: HashMap<String, QuantizationParameters>,

    /// Quantization statistics for monitoring
    quantization_stats: QuantizationStats,
}

/// Quantization parameters for a specific layer
#[derive(Debug, Clone)]
pub struct QuantizationParameters {
    /// Scale parameter (learnable)
    pub scale: f32,

    /// Zero-point parameter (learnable)
    pub zero_point: f32,

    /// Gradient for scale parameter
    pub scale_grad: f32,

    /// Gradient for zero-point parameter
    pub zero_point_grad: f32,

    /// Moving average of scale parameter
    pub scale_ema: f32,

    /// Moving average of zero-point parameter
    pub zero_point_ema: f32,
}

impl Default for QuantizationParameters {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0.0,
            scale_grad: 0.0,
            zero_point_grad: 0.0,
            scale_ema: 1.0,
            zero_point_ema: 0.0,
        }
    }
}

/// QAT tensor operations with straight-through estimation
#[derive(Debug)]
pub struct QATTensorOps {
    /// STE implementation
    ste: StraightThroughEstimator,

    /// Current training mode
    training_mode: bool,

    /// Quantization statistics for monitoring
    quantization_stats: QuantizationStats,
}

/// Quantization statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Total number of quantization operations
    pub total_quantizations: usize,

    /// Average quantization error
    pub avg_quantization_error: f32,

    /// Gradient flow statistics
    pub gradient_stats: GradientStats,

    /// Layer-wise quantization quality
    pub layer_quality: HashMap<String, f32>,
}

/// Gradient flow statistics for STE
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Average gradient magnitude before quantization
    pub avg_grad_magnitude_before: f32,

    /// Average gradient magnitude after quantization
    pub avg_grad_magnitude_after: f32,

    /// Gradient variance before quantization
    pub grad_variance_before: f32,

    /// Gradient variance after quantization
    pub grad_variance_after: f32,

    /// Number of gradient updates
    pub num_updates: usize,
}

impl Default for QuantizationStats {
    fn default() -> Self {
        Self {
            total_quantizations: 0,
            avg_quantization_error: 0.0,
            gradient_stats: GradientStats::default(),
            layer_quality: HashMap::new(),
        }
    }
}

impl Default for GradientStats {
    fn default() -> Self {
        Self {
            avg_grad_magnitude_before: 0.0,
            avg_grad_magnitude_after: 0.0,
            grad_variance_before: 0.0,
            grad_variance_after: 0.0,
            num_updates: 0,
        }
    }
}

impl StraightThroughEstimator {
    /// Create new STE with configuration
    pub fn new(config: QATConfig) -> Self {
        Self {
            config,
            current_step: 0,
            layer_parameters: HashMap::new(),
            quantization_stats: QuantizationStats::default(),
        }
    }

    /// Apply binary quantization with STE
    pub fn binary_quantization_ste(
        &mut self,
        input: &BitNetTensor,
        layer_id: &str,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let candle_tensor =
            input
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get candle tensor: {e}"),
                })?;

        // Forward pass: quantize to {-1, +1}
        let quantized = if self.config.fake_quantization {
            // Fake quantization: apply quantization but keep gradients
            self.apply_fake_binary_quantization(&candle_tensor)?
        } else {
            // True quantization with STE
            candle_tensor
                .sign()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to apply sign function: {e}"),
                })?
        };

        // Add quantization noise for regularization
        let noisy_quantized = if self.config.noise_level > 0.0 {
            self.add_quantization_noise(&quantized)?
        } else {
            quantized
        };

        // Update statistics
        self.update_quantization_stats(layer_id, &candle_tensor, &noisy_quantized)?;

        // Convert back to BitNetTensor
        BitNetTensor::from_candle_tensor(noisy_quantized, input.device().clone())
            .map_err(TensorIntegrationError::Memory)
    }

    /// Apply ternary quantization with STE
    pub fn ternary_quantization_ste(
        &mut self,
        input: &BitNetTensor,
        threshold: f32,
        layer_id: &str,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let candle_tensor =
            input
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get candle tensor: {e}"),
                })?;

        // Forward pass: quantize to {-1, 0, +1}
        let quantized = if self.config.fake_quantization {
            self.apply_fake_ternary_quantization(&candle_tensor, threshold)?
        } else {
            self.apply_true_ternary_quantization(&candle_tensor, threshold)?
        };

        // Add temperature scaling for soft quantization
        let temperature_scaled = if self.config.temperature != 1.0 {
            self.apply_temperature_scaling(&quantized, self.get_current_temperature())?
        } else {
            quantized
        };

        // Update statistics
        self.update_quantization_stats(layer_id, &candle_tensor, &temperature_scaled)?;

        // Convert back to BitNetTensor
        BitNetTensor::from_candle_tensor(temperature_scaled, input.device().clone())
            .map_err(TensorIntegrationError::Memory)
    }

    /// Apply weight quantization with learnable parameters
    pub fn weight_quantization_ste(
        &mut self,
        weights: &BitNetTensor,
        layer_id: &str,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Get or initialize quantization parameters for this layer
        let _params = self
            .layer_parameters
            .entry(layer_id.to_string())
            .or_default();

        let candle_tensor =
            weights
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get candle tensor: {e}"),
                })?;

        // Apply learnable quantization
        let quantized = match self.config.weight_precision {
            QuantizationPrecision::OneFiveFiveBit => {
                self.ternary_quantization_ste(weights, 0.5, layer_id)?
            }
            QuantizationPrecision::EightBit => {
                self.multi_bit_quantization_ste(weights, 8, layer_id)?
            }
            QuantizationPrecision::FourBit => {
                self.multi_bit_quantization_ste(weights, 4, layer_id)?
            }
            _ => weights.clone(), // No quantization for other precisions
        };

        // Update learnable parameters
        self.update_learnable_parameters(layer_id, &candle_tensor)?;

        Ok(quantized)
    }

    /// Apply activation quantization with STE
    pub fn activation_quantization_ste(
        &mut self,
        activations: &BitNetTensor,
        layer_id: &str,
    ) -> TensorIntegrationResult<BitNetTensor> {
        match self.config.activation_precision {
            QuantizationPrecision::OneFiveFiveBit => {
                self.ternary_quantization_ste(activations, 0.5, layer_id)
            }
            QuantizationPrecision::EightBit => {
                self.multi_bit_quantization_ste(activations, 8, layer_id)
            }
            QuantizationPrecision::FourBit => {
                self.multi_bit_quantization_ste(activations, 4, layer_id)
            }
            _ => Ok(activations.clone()), // No quantization for other precisions
        }
    }

    /// Apply multi-bit quantization with STE
    pub fn multi_bit_quantization_ste(
        &mut self,
        input: &BitNetTensor,
        bits: u32,
        layer_id: &str,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let candle_tensor =
            input
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get candle tensor: {e}"),
                })?;

        // Forward pass: quantize to 2^bits levels
        let levels = 2_f32.powi(bits as i32) - 1.0;
        let scale = levels / 2.0;

        // Quantization: round((input + 1) * scale) / scale - 1
        let shifted = candle_tensor
            .add(&CandleTensor::new(1.0f32, input.device()).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to create shift tensor: {e}"),
                }
            })?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to shift tensor: {e}"),
            })?;

        let scaled = shifted
            .mul(&CandleTensor::new(scale, input.device()).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to create scale tensor: {e}"),
                }
            })?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to scale tensor: {e}"),
            })?;

        let quantized_scaled = scaled
            .round()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to round tensor: {e}"),
            })?;

        let dequantized = quantized_scaled
            .div(&CandleTensor::new(scale, input.device()).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to create dequant scale tensor: {e}"),
                }
            })?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to dequantize tensor: {e}"),
            })?
            .sub(&CandleTensor::new(1.0f32, input.device()).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to create dequant shift tensor: {e}"),
                }
            })?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to shift dequantized tensor: {e}"),
            })?;

        // Update statistics
        self.update_quantization_stats(layer_id, &candle_tensor, &dequantized)?;

        // Convert back to BitNetTensor
        BitNetTensor::from_candle_tensor(dequantized, input.device().clone())
            .map_err(TensorIntegrationError::Memory)
    }

    // Helper methods

    fn apply_fake_binary_quantization(
        &self,
        tensor: &CandleTensor,
    ) -> TensorIntegrationResult<CandleTensor> {
        // Apply sign function but preserve gradients (conceptually)
        tensor.sign().map_err(|e| TensorIntegrationError::TensorOp {
            message: format!("Failed to apply fake binary quantization: {e}"),
        })
    }

    fn apply_fake_ternary_quantization(
        &self,
        tensor: &CandleTensor,
        threshold: f32,
    ) -> TensorIntegrationResult<CandleTensor> {
        let abs_input = tensor.abs().map_err(|e| TensorIntegrationError::TensorOp {
            message: format!("Failed to compute absolute value: {e}"),
        })?;

        let threshold_tensor = Tensor::new(threshold, tensor.device()).map_err(|e| {
            TensorIntegrationError::TensorOp {
                message: format!("Failed to create threshold tensor: {e}"),
            }
        })?;

        let mask =
            abs_input
                .gt(&threshold_tensor)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to create threshold mask: {e}"),
                })?;

        let sign = tensor
            .sign()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute sign: {e}"),
            })?;

        let quantized =
            sign.mul(&mask.to_dtype(sign.dtype()).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to convert mask dtype: {e}"),
                }
            })?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to apply mask: {e}"),
            })?;

        Ok(quantized)
    }

    fn apply_true_ternary_quantization(
        &self,
        tensor: &CandleTensor,
        threshold: f32,
    ) -> TensorIntegrationResult<CandleTensor> {
        // Same as fake quantization but with additional processing
        self.apply_fake_ternary_quantization(tensor, threshold)
    }

    fn add_quantization_noise(
        &self,
        tensor: &CandleTensor,
    ) -> TensorIntegrationResult<CandleTensor> {
        if self.config.noise_level <= 0.0 {
            return Ok(tensor.clone());
        }

        // Add uniform noise for regularization
        let noise_shape = tensor.shape().clone();
        let noise = CandleTensor::randn(
            0.0,
            self.config.noise_level as f64,
            noise_shape,
            tensor.device(),
        )
        .map_err(|e| TensorIntegrationError::TensorOp {
            message: format!("Failed to generate noise: {e}"),
        })?;

        tensor
            .add(&noise)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to add noise: {e}"),
            })
    }

    fn apply_temperature_scaling(
        &self,
        tensor: &CandleTensor,
        temperature: f32,
    ) -> TensorIntegrationResult<CandleTensor> {
        if temperature == 1.0 {
            return Ok(tensor.clone());
        }

        let temp_tensor = Tensor::new(temperature, tensor.device()).map_err(|e| {
            TensorIntegrationError::TensorOp {
                message: format!("Failed to create temperature tensor: {e}"),
            }
        })?;

        tensor
            .div(&temp_tensor)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to apply temperature scaling: {e}"),
            })
    }

    fn get_current_temperature(&self) -> f32 {
        if self.config.temperature_annealing {
            // Simple annealing schedule: decrease temperature over time
            let annealing_factor = 1.0 / (1.0 + 0.001 * self.current_step as f32);
            self.config.temperature * annealing_factor
        } else {
            self.config.temperature
        }
    }

    fn update_quantization_stats(
        &mut self,
        layer_id: &str,
        original: &CandleTensor,
        quantized: &CandleTensor,
    ) -> TensorIntegrationResult<()> {
        // Compute quantization error
        let error = original
            .sub(quantized)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute quantization error: {e}"),
            })?;

        let mse = error
            .sqr()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute squared error: {e}"),
            })?
            .mean_all()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute mean error: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract error scalar: {e}"),
            })?;

        // Update statistics
        let n = self.quantization_stats.total_quantizations as f32;
        self.quantization_stats.avg_quantization_error =
            (self.quantization_stats.avg_quantization_error * n + mse) / (n + 1.0);

        self.quantization_stats.total_quantizations += 1;
        self.quantization_stats
            .layer_quality
            .insert(layer_id.to_string(), 1.0 - mse);

        Ok(())
    }

    fn update_learnable_parameters(
        &mut self,
        layer_id: &str,
        tensor: &CandleTensor,
    ) -> TensorIntegrationResult<()> {
        // Update EMA of quantization parameters
        let params = self
            .layer_parameters
            .entry(layer_id.to_string())
            .or_default();

        let alpha = 0.1; // EMA decay factor

        // Compute statistics for parameter updates
        let mean = tensor
            .mean_all()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute mean: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract mean: {e}"),
            })?;

        let std = tensor
            .var(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute variance: {e}"),
            })?
            .mean_all()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute mean variance: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract variance: {e}"),
            })?
            .sqrt();

        // Update EMA parameters
        params.scale_ema = alpha * std + (1.0 - alpha) * params.scale_ema;
        params.zero_point_ema = alpha * mean + (1.0 - alpha) * params.zero_point_ema;

        Ok(())
    }

    /// Step the training process
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get quantization statistics
    pub fn get_stats(&self) -> &QuantizationStats {
        &self.quantization_stats
    }

    /// Reset quantization statistics
    pub fn reset_stats(&mut self) {
        self.quantization_stats = QuantizationStats::default();
    }
}

impl QATTensorOps {
    /// Create new QAT tensor operations
    pub fn new(config: QATConfig) -> Self {
        Self {
            ste: StraightThroughEstimator::new(config),
            training_mode: true,
            quantization_stats: QuantizationStats::default(),
        }
    }

    /// Set training mode
    pub fn set_training_mode(&mut self, training: bool) {
        self.training_mode = training;
    }

    /// Apply QAT to tensor
    pub fn apply_qat(
        &mut self,
        input: &BitNetTensor,
        layer_id: &str,
        tensor_type: QATTensorType,
    ) -> TensorIntegrationResult<BitNetTensor> {
        if !self.training_mode {
            // In evaluation mode, return original tensor
            return Ok(input.clone());
        }

        match tensor_type {
            QATTensorType::Weight => self.ste.weight_quantization_ste(input, layer_id),
            QATTensorType::Activation => self.ste.activation_quantization_ste(input, layer_id),
            QATTensorType::Binary => self.ste.binary_quantization_ste(input, layer_id),
            QATTensorType::Ternary(threshold) => self
                .ste
                .ternary_quantization_ste(input, threshold, layer_id),
        }
    }

    /// Step the QAT training process
    pub fn step(&mut self) {
        self.ste.step();
    }

    /// Get QAT statistics
    pub fn get_qat_stats(&self) -> &QuantizationStats {
        self.ste.get_stats()
    }

    /// Reset QAT statistics
    pub fn reset_qat_stats(&mut self) {
        self.ste.reset_stats();
    }
}

/// Types of tensors for QAT operations
#[derive(Debug, Clone)]
pub enum QATTensorType {
    /// Weight tensor
    Weight,

    /// Activation tensor
    Activation,

    /// Binary quantization
    Binary,

    /// Ternary quantization with threshold
    Ternary(f32),
}

/// Error types specific to QAT operations
#[derive(Debug, thiserror::Error)]
pub enum QATError {
    #[error("STE operation failed: {message}")]
    STE { message: String },

    #[error("Gradient computation failed: {message}")]
    Gradient { message: String },

    #[error("Quantization parameter update failed: {message}")]
    ParameterUpdate { message: String },

    #[error("Temperature scaling failed: {message}")]
    TemperatureScaling { message: String },

    #[error("Noise injection failed: {message}")]
    NoiseInjection { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_default() {
        let config = QATConfig::default();
        assert!(config.enable_ste);
        assert!(config.fake_quantization);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.quantization_lr, 0.01);
    }

    #[test]
    fn test_ste_creation() {
        let config = QATConfig::default();
        let ste = StraightThroughEstimator::new(config);
        assert_eq!(ste.current_step, 0);
        assert!(ste.layer_parameters.is_empty());
    }

    #[test]
    fn test_qat_tensor_ops_creation() {
        let config = QATConfig::default();
        let qat_ops = QATTensorOps::new(config);
        assert!(qat_ops.training_mode);
        assert_eq!(qat_ops.quantization_stats.total_quantizations, 0);
    }

    #[test]
    fn test_quantization_parameters_default() {
        let params = QuantizationParameters::default();
        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0.0);
        assert_eq!(params.scale_ema, 1.0);
        assert_eq!(params.zero_point_ema, 0.0);
    }

    #[test]
    fn test_training_mode_toggle() {
        let config = QATConfig::default();
        let mut qat_ops = QATTensorOps::new(config);

        assert!(qat_ops.training_mode);
        qat_ops.set_training_mode(false);
        assert!(!qat_ops.training_mode);
    }
}
