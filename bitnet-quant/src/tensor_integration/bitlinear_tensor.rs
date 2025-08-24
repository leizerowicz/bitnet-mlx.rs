//! BitLinear Layer Tensor Operations
//!
//! This module provides specialized tensor operations for BitLinear layers,
//! the core building block of BitNet neural networks, with support for
//! quantized weights, LayerNorm integration, and residual connections.

use candle_core::{Device, Tensor as CandleTensor};
use std::collections::HashMap;
use std::sync::Arc;

use bitnet_core::{BitNetDType, BitNetTensor, TensorShape};

use bitnet_core::tensor::shape::BroadcastCompatible;

use crate::quantization::{ActivationQuantizer, QuantizationPrecision, WeightQuantizer};

use super::{
    quantized_tensor::{QuantizedTensor, QuantizedTensorConfig},
    TensorIntegrationError, TensorIntegrationResult,
};

use super::bitnet_ops::BitNetQuantizationConfig;

/// Configuration for BitLinear tensor operations
#[derive(Debug, Clone)]
pub struct BitLinearConfig {
    /// Weight quantization configuration
    pub weight_quantization: BitNetQuantizationConfig,

    /// Activation quantization configuration
    pub activation_quantization: BitNetQuantizationConfig,

    /// Enable LayerNorm integration
    pub enable_layernorm: bool,

    /// LayerNorm epsilon for numerical stability
    pub layernorm_eps: f32,

    /// Enable residual connections
    pub enable_residual: bool,

    /// Target device for operations
    pub device: Option<Device>,
}

impl Default for BitLinearConfig {
    fn default() -> Self {
        Self {
            weight_quantization: BitNetQuantizationConfig::default(),
            activation_quantization: BitNetQuantizationConfig::default(),
            enable_layernorm: true,
            layernorm_eps: 1e-5,
            enable_residual: true,
            device: None,
        }
    }
}

/// Weight quantization tensor operations
#[derive(Debug)]
pub struct WeightQuantizationTensor {
    /// Original weight tensor
    pub weights: BitNetTensor,

    /// Quantized weight representation
    pub quantized_weights: QuantizedTensor,

    /// Weight shape information
    pub weight_shape: TensorShape,

    /// Quantization parameters
    pub quantization_config: BitNetQuantizationConfig,
}

/// Activation quantization tensor operations
#[derive(Debug)]
pub struct ActivationQuantizationTensor {
    /// Input activation tensor
    pub activations: BitNetTensor,

    /// Quantized activation representation
    pub quantized_activations: Option<QuantizedTensor>,

    /// Activation statistics for calibration
    pub activation_stats: ActivationStats,

    /// Quantization configuration
    pub quantization_config: BitNetQuantizationConfig,
}

/// Activation statistics for quantization calibration
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Running mean of activation values
    pub running_mean: f32,

    /// Running variance of activation values
    pub running_var: f32,

    /// Minimum observed value
    pub min_val: f32,

    /// Maximum observed value
    pub max_val: f32,

    /// Number of samples processed
    pub sample_count: usize,
}

impl Default for ActivationStats {
    fn default() -> Self {
        Self {
            running_mean: 0.0,
            running_var: 1.0,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            sample_count: 0,
        }
    }
}

/// Errors specific to BitLinear tensor operations
#[derive(Debug, thiserror::Error)]
pub enum BitLinearTensorError {
    #[error("Weight shape mismatch: expected {expected:?}, got {actual:?}")]
    WeightShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Activation quantization failed: {message}")]
    ActivationQuantization { message: String },

    #[error("LayerNorm operation failed: {message}")]
    TensorOp { message: String },

    #[error("Residual connection failed: {message}")]
    ResidualConnection { message: String },

    #[error("BitLinear forward pass failed: {message}")]
    ForwardPass { message: String },
}

/// LayerNorm integration support
pub struct LayerNormIntegration {
    /// LayerNorm weight (gamma)
    pub weight: BitNetTensor,

    /// LayerNorm bias (beta)
    pub bias: BitNetTensor,

    /// Epsilon for numerical stability
    pub eps: f32,

    /// Normalized shape
    pub normalized_shape: TensorShape,
}

/// Residual connection support
pub struct ResidualConnectionSupport {
    /// Whether to use residual connections
    pub enabled: bool,

    /// Residual scaling factor
    pub scale_factor: f32,

    /// Device placement
    pub device: Device,
}

/// Implementation of BitLinear tensor operations
#[derive(Debug, Default)]
pub struct BitLinearTensorOpsImpl {
    config: BitLinearConfig,
    weight_quantizer: Option<Arc<dyn WeightQuantizer>>,
    activation_quantizer: Option<Arc<dyn ActivationQuantizer>>,
}

impl BitLinearTensorOpsImpl {
    /// Create new BitLinear operations with configuration
    pub fn new(config: BitLinearConfig) -> Self {
        Self {
            config,
            weight_quantizer: None,
            activation_quantizer: None,
        }
    }

    /// Set weight quantizer
    pub fn with_weight_quantizer(mut self, quantizer: Arc<dyn WeightQuantizer>) -> Self {
        self.weight_quantizer = Some(quantizer);
        self
    }

    /// Set activation quantizer
    pub fn with_activation_quantizer(mut self, quantizer: Arc<dyn ActivationQuantizer>) -> Self {
        self.activation_quantizer = Some(quantizer);
        self
    }

    /// Quantize weights for BitLinear layer
    pub fn quantize_weights(
        &self,
        weights: BitNetTensor,
    ) -> TensorIntegrationResult<WeightQuantizationTensor> {
        let weight_shape = weights.shape().clone();

        let quantized_config = QuantizedTensorConfig {
            precision: self.config.weight_quantization.precision,
            strategy: self.config.weight_quantization.strategy,
            device: Some(weights.device().clone()),
            use_memory_pool: self.config.weight_quantization.use_memory_pool,
            enable_compression: true,
            compression_threshold: 0.1,
        };

        let quantized_weights =
            QuantizedTensor::from_bitnet_tensor(weights.clone(), quantized_config)?;

        Ok(WeightQuantizationTensor {
            weights,
            quantized_weights,
            weight_shape: TensorShape::new(weight_shape.dims()),
            quantization_config: self.config.weight_quantization.clone(),
        })
    }

    /// Quantize activations for BitLinear layer
    pub fn quantize_activations(
        &self,
        activations: BitNetTensor,
    ) -> TensorIntegrationResult<ActivationQuantizationTensor> {
        let mut activation_stats = ActivationStats::default();

        // Compute activation statistics
        let candle_tensor =
            activations
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get activation tensor: {e}"),
                })?;

        activation_stats.update_stats(&candle_tensor)?;

        let quantized_config = QuantizedTensorConfig {
            precision: self.config.activation_quantization.precision,
            strategy: self.config.activation_quantization.strategy,
            device: Some(activations.device().clone()),
            use_memory_pool: self.config.activation_quantization.use_memory_pool,
            enable_compression: true,
            compression_threshold: 0.1,
        };

        let quantized_activations = if self.config.activation_quantization.precision
            != QuantizationPrecision::OneFiveFiveBit
        {
            // For non-ternary activations, quantize immediately
            Some(QuantizedTensor::from_bitnet_tensor(
                activations.clone(),
                quantized_config,
            )?)
        } else {
            // For ternary activations, delay quantization
            None
        };

        Ok(ActivationQuantizationTensor {
            activations,
            quantized_activations,
            activation_stats,
            quantization_config: self.config.activation_quantization.clone(),
        })
    }

    /// Perform BitLinear forward pass
    pub fn forward(
        &self,
        input: &BitNetTensor,
        weight_tensor: &WeightQuantizationTensor,
        bias: Option<&BitNetTensor>,
        layernorm: Option<&LayerNormIntegration>,
        residual_input: Option<&BitNetTensor>,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let device = input.device().clone();

        // Step 1: Apply LayerNorm if configured
        let normalized_input = if let Some(ln) = layernorm {
            self.apply_layernorm(input, ln)?
        } else {
            input.clone()
        };

        // Step 2: Quantize activations
        let activation_tensor = self.quantize_activations(normalized_input)?;

        // Step 3: Perform quantized matrix multiplication
        let matmul_result = self.quantized_linear(&activation_tensor, weight_tensor)?;

        // Step 4: Add bias if present
        let biased_result = if let Some(b) = bias {
            matmul_result
                .add(b)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to add bias: {e}"),
                })?
        } else {
            matmul_result
        };

        // Step 5: Apply residual connection if configured
        let final_result = if let Some(residual) = residual_input {
            if self.config.enable_residual {
                self.apply_residual_connection(&biased_result, residual)?
            } else {
                biased_result
            }
        } else {
            biased_result
        };

        Ok(final_result)
    }

    /// Apply LayerNorm to input tensor
    pub fn apply_layernorm(
        &self,
        input: &BitNetTensor,
        layernorm: &LayerNormIntegration,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let input_tensor =
            input
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get input tensor: {e}"),
                })?;

        let weight_tensor =
            layernorm
                .weight
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get LayerNorm weight: {e}"),
                })?;

        let bias_tensor =
            layernorm
                .bias
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get LayerNorm bias: {e}"),
                })?;

        // Compute mean and variance along the last dimension
        let last_dim = input_tensor.dims().len() - 1;
        let mean = input_tensor.mean_keepdim(last_dim)?;

        let variance = input_tensor.var_keepdim(last_dim)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let centered =
            input_tensor
                .broadcast_sub(&mean)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to center input: {e}"),
                })?;

        let eps_tensor = variance
            .ones_like()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to create eps tensor: {e}"),
            })?
            .mul(
                &candle_core::Tensor::new(layernorm.eps, input.device()).map_err(|e| {
                    TensorIntegrationError::TensorOp {
                        message: format!("Failed to create eps scalar: {e}"),
                    }
                })?,
            )
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to scale eps: {e}"),
            })?;

        let std = variance
            .add(&eps_tensor)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to add eps to variance: {e}"),
            })?
            .sqrt()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute std: {e}"),
            })?;

        let normalized =
            centered
                .broadcast_div(&std)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to normalize: {e}"),
                })?;

        // Apply learnable parameters: normalized * weight + bias
        let scaled = normalized.broadcast_mul(&weight_tensor).map_err(|e| {
            TensorIntegrationError::TensorOp {
                message: format!("Failed to apply weight: {e}"),
            }
        })?;

        let result =
            scaled
                .broadcast_add(&bias_tensor)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to apply bias: {e}"),
                })?;

        BitNetTensor::from_candle_tensor(result, input.device().clone())
            .map_err(TensorIntegrationError::Memory)
    }

    /// Perform quantized linear transformation
    pub fn quantized_linear(
        &self,
        activation_tensor: &ActivationQuantizationTensor,
        weight_tensor: &WeightQuantizationTensor,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Get quantized activations
        let quantized_activations = if let Some(ref qa) = activation_tensor.quantized_activations {
            qa.clone()
        } else {
            // Quantize activations on-demand
            let config = QuantizedTensorConfig {
                precision: activation_tensor.quantization_config.precision,
                strategy: activation_tensor.quantization_config.strategy,
                device: Some(activation_tensor.activations.device().clone()),
                use_memory_pool: activation_tensor.quantization_config.use_memory_pool,
                enable_compression: true,
                compression_threshold: 0.1,
            };

            QuantizedTensor::from_bitnet_tensor(activation_tensor.activations.clone(), config)?
        };

        // Perform quantized matrix multiplication
        let result_quantized =
            quantized_activations.quantized_matmul(&weight_tensor.quantized_weights)?;

        // Dequantize result
        result_quantized.to_bitnet_tensor()
    }

    /// Apply residual connection
    pub fn apply_residual_connection(
        &self,
        main_path: &BitNetTensor,
        residual: &BitNetTensor,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Check shape compatibility
        if !main_path.shape().is_broadcast_compatible(residual.shape()) {
            return Err(TensorIntegrationError::ShapeMismatch {
                message: format!(
                    "Residual connection shape mismatch: {:?} vs {:?}",
                    main_path.shape(),
                    residual.shape()
                ),
            });
        }

        main_path
            .add(residual)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to apply residual connection: {e}"),
            })
    }

    /// Create LayerNorm integration
    pub fn create_layernorm_integration(
        &self,
        normalized_shape: &[usize],
        device: Device,
    ) -> TensorIntegrationResult<LayerNormIntegration> {
        // We need to get the HybridMemoryPool to create tensors
        let pool =
            bitnet_core::tensor::memory_integration::get_global_memory_pool().ok_or_else(|| {
                TensorIntegrationError::Memory(bitnet_core::MemoryError::InsufficientMemory {
                    size: 0,
                })
            })?;

        let weight = BitNetTensor::ones(normalized_shape, BitNetDType::F32, Some(device.clone()))
            .map_err(|e| TensorIntegrationError::TensorOperation {
            message: format!("Failed to create weight tensor: {e}"),
        })?;

        let bias = BitNetTensor::zeros(normalized_shape, BitNetDType::F32, Some(device.clone()))
            .map_err(|e| TensorIntegrationError::TensorOperation {
                message: format!("Failed to create bias tensor: {e}"),
            })?;

        Ok(LayerNormIntegration {
            weight,
            bias,
            eps: self.config.layernorm_eps,
            normalized_shape: TensorShape::new(normalized_shape),
        })
    }

    /// Create residual connection support
    pub fn create_residual_support(&self, device: Device) -> ResidualConnectionSupport {
        ResidualConnectionSupport {
            enabled: self.config.enable_residual,
            scale_factor: 1.0,
            device,
        }
    }

    /// Create mixed precision BitLinear operations
    pub fn create_mixed_precision_ops(
        &self,
        weight_precision: QuantizationPrecision,
        activation_precision: QuantizationPrecision,
    ) -> MixedPrecisionBitLinearOps {
        MixedPrecisionBitLinearOps {
            weight_precision,
            activation_precision,
            base_config: self.config.clone(),
            precision_stats: MixedPrecisionStats::default(),
        }
    }

    /// Perform BitLinear forward pass with mixed precision
    pub fn mixed_precision_forward(
        &self,
        input: &BitNetTensor,
        weight_tensor: &WeightQuantizationTensor,
        bias: Option<&BitNetTensor>,
        layernorm: Option<&LayerNormIntegration>,
        residual_input: Option<&BitNetTensor>,
        precision_config: &MixedPrecisionConfig,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Apply precision-specific processing based on layer importance
        let processed_input = if precision_config.enable_dynamic_precision {
            self.apply_dynamic_precision_selection(input, &precision_config.layer_importance)?
        } else {
            input.clone()
        };

        // Standard forward pass with precision-aware processing
        self.forward(
            &processed_input,
            weight_tensor,
            bias,
            layernorm,
            residual_input,
        )
    }

    /// Apply dynamic precision selection based on layer importance
    fn apply_dynamic_precision_selection(
        &self,
        input: &BitNetTensor,
        layer_importance: &HashMap<String, f32>,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // For now, return input as-is
        // In a full implementation, this would select precision based on importance
        Ok(input.clone())
    }

    /// Optimize tensor operations for specific hardware
    pub fn optimize_for_hardware(
        &self,
        tensor: &BitNetTensor,
        hardware_profile: &HardwareProfile,
    ) -> TensorIntegrationResult<BitNetTensor> {
        match hardware_profile.device_type {
            HardwareDeviceType::AppleSilicon => {
                // Use MLX optimizations
                self.optimize_for_mlx(tensor)
            }
            HardwareDeviceType::NvidiaGPU => {
                // Use CUDA optimizations
                self.optimize_for_cuda(tensor)
            }
            HardwareDeviceType::CPU => {
                // Use CPU SIMD optimizations
                self.optimize_for_cpu_simd(tensor)
            }
            HardwareDeviceType::Generic => {
                // No specific optimizations
                Ok(tensor.clone())
            }
        }
    }

    fn optimize_for_mlx(&self, tensor: &BitNetTensor) -> TensorIntegrationResult<BitNetTensor> {
        // MLX-specific optimizations would go here
        Ok(tensor.clone())
    }

    fn optimize_for_cuda(&self, tensor: &BitNetTensor) -> TensorIntegrationResult<BitNetTensor> {
        // CUDA-specific optimizations would go here
        Ok(tensor.clone())
    }

    fn optimize_for_cpu_simd(
        &self,
        tensor: &BitNetTensor,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // CPU SIMD optimizations would go here
        Ok(tensor.clone())
    }
}

/// Mixed precision BitLinear operations
#[derive(Debug)]
pub struct MixedPrecisionBitLinearOps {
    /// Weight quantization precision
    pub weight_precision: QuantizationPrecision,

    /// Activation quantization precision
    pub activation_precision: QuantizationPrecision,

    /// Base BitLinear configuration
    pub base_config: BitLinearConfig,

    /// Mixed precision statistics
    pub precision_stats: MixedPrecisionStats,
}

/// Configuration for mixed precision operations
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable dynamic precision selection
    pub enable_dynamic_precision: bool,

    /// Layer importance scores for precision selection
    pub layer_importance: HashMap<String, f32>,

    /// Performance vs accuracy trade-off (0.0 = accuracy, 1.0 = performance)
    pub performance_priority: f32,

    /// Memory budget in MB
    pub memory_budget: Option<usize>,

    /// Enable precision adaptation during training
    pub adaptive_precision: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_precision: false,
            layer_importance: HashMap::new(),
            performance_priority: 0.5,
            memory_budget: None,
            adaptive_precision: false,
        }
    }
}

/// Statistics for mixed precision operations
#[derive(Debug, Clone, Default)]
pub struct MixedPrecisionStats {
    /// Operations performed at each precision
    pub operations_by_precision: HashMap<QuantizationPrecision, usize>,

    /// Average computation time by precision
    pub avg_time_by_precision: HashMap<QuantizationPrecision, f32>,

    /// Memory usage by precision
    pub memory_usage_by_precision: HashMap<QuantizationPrecision, usize>,

    /// Accuracy metrics by precision
    pub accuracy_by_precision: HashMap<QuantizationPrecision, f32>,
}

/// Hardware profile for optimization
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Type of device
    pub device_type: HardwareDeviceType,

    /// Available memory in MB
    pub memory_mb: usize,

    /// Number of compute units
    pub compute_units: usize,

    /// Supports specific instruction sets
    pub instruction_sets: Vec<InstructionSet>,

    /// Optimal tensor tile sizes
    pub optimal_tile_sizes: Vec<usize>,
}

/// Types of hardware devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareDeviceType {
    AppleSilicon,
    NvidiaGPU,
    CPU,
    Generic,
}

/// Supported instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionSet {
    AVX2,
    AVX512,
    NEON,
    SSE4,
    CUDA,
    Metal,
}

impl MixedPrecisionBitLinearOps {
    /// Perform forward pass with mixed precision
    pub fn forward(
        &mut self,
        input: &BitNetTensor,
        weights: &WeightQuantizationTensor,
        bias: Option<&BitNetTensor>,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Record operation start time
        let start_time = std::time::Instant::now();

        // Apply weight quantization at specified precision
        let quantized_weights = self.apply_weight_precision(&weights.quantized_weights)?;

        // Apply activation quantization at specified precision
        let quantized_input = self.apply_activation_precision(input)?;

        // Perform quantized linear transformation
        let result = quantized_input
            .matmul(&quantized_weights.to_bitnet_tensor()?)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to perform matrix multiplication: {e}"),
            })?;

        // Add bias if present
        let final_result = if let Some(b) = bias {
            result
                .add(b)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to add bias: {e}"),
                })?
        } else {
            result
        };

        // Update statistics
        let duration = start_time.elapsed().as_secs_f32();
        self.update_precision_stats(duration, &final_result)?;

        Ok(final_result)
    }

    fn apply_weight_precision(
        &self,
        weights: &QuantizedTensor,
    ) -> TensorIntegrationResult<QuantizedTensor> {
        // Apply weight-specific precision transformations
        Ok(weights.clone())
    }

    fn apply_activation_precision(
        &self,
        input: &BitNetTensor,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Apply activation-specific precision transformations
        Ok(input.clone())
    }

    fn update_precision_stats(
        &mut self,
        duration: f32,
        result: &BitNetTensor,
    ) -> TensorIntegrationResult<()> {
        // Update weight precision stats
        *self
            .precision_stats
            .operations_by_precision
            .entry(self.weight_precision)
            .or_insert(0) += 1;

        *self
            .precision_stats
            .avg_time_by_precision
            .entry(self.weight_precision)
            .or_insert(0.0) = duration;

        // Update activation precision stats
        *self
            .precision_stats
            .operations_by_precision
            .entry(self.activation_precision)
            .or_insert(0) += 1;

        Ok(())
    }

    /// Get precision statistics summary
    pub fn get_precision_summary(&self) -> PrecisionSummary {
        PrecisionSummary {
            total_operations: self.precision_stats.operations_by_precision.values().sum(),
            avg_duration: self
                .precision_stats
                .avg_time_by_precision
                .values()
                .sum::<f32>()
                / self.precision_stats.avg_time_by_precision.len() as f32,
            memory_efficiency: self.calculate_memory_efficiency(),
            precision_distribution: self.precision_stats.operations_by_precision.clone(),
        }
    }

    fn calculate_memory_efficiency(&self) -> f32 {
        // Calculate memory efficiency based on precision usage
        let total_ops: usize = self.precision_stats.operations_by_precision.values().sum();
        if total_ops == 0 {
            return 1.0;
        }

        let efficiency: f32 = self
            .precision_stats
            .operations_by_precision
            .iter()
            .map(|(precision, count)| {
                let precision_efficiency = match precision {
                    QuantizationPrecision::OneFiveFiveBit => 0.9,
                    QuantizationPrecision::FourBit => 0.8,
                    QuantizationPrecision::EightBit => 0.6,
                    _ => 0.4,
                };
                precision_efficiency * (*count as f32 / total_ops as f32)
            })
            .sum();

        efficiency
    }
}

/// Summary of precision operations
#[derive(Debug, Clone)]
pub struct PrecisionSummary {
    pub total_operations: usize,
    pub avg_duration: f32,
    pub memory_efficiency: f32,
    pub precision_distribution: HashMap<QuantizationPrecision, usize>,
}

impl ActivationStats {
    /// Update statistics with new tensor data
    pub fn update_stats(&mut self, tensor: &CandleTensor) -> TensorIntegrationResult<()> {
        let mean = tensor
            .mean_all()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute tensor mean: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract mean scalar: {e}"),
            })?;

        let var = tensor
            .var(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute tensor variance: {e}"),
            })?
            .mean_all()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute mean variance: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract variance scalar: {e}"),
            })?;

        let min_val = tensor
            .min(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute tensor min: {e}"),
            })?
            .min_keepdim(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute global min: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract min scalar: {e}"),
            })?;

        let max_val = tensor
            .max(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute tensor max: {e}"),
            })?
            .max_keepdim(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute global max: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract max scalar: {e}"),
            })?;

        // Update running statistics
        let n = self.sample_count as f32;
        self.running_mean = (self.running_mean * n + mean) / (n + 1.0);
        self.running_var = (self.running_var * n + var) / (n + 1.0);
        self.min_val = self.min_val.min(min_val);
        self.max_val = self.max_val.max(max_val);
        self.sample_count += 1;

        Ok(())
    }

    /// Get current statistics summary
    pub fn get_summary(&self) -> StatsSummary {
        StatsSummary {
            mean: self.running_mean,
            variance: self.running_var,
            std_dev: self.running_var.sqrt(),
            min: self.min_val,
            max: self.max_val,
            range: self.max_val - self.min_val,
            sample_count: self.sample_count,
        }
    }
}

/// Statistics summary for activation calibration
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub mean: f32,
    pub variance: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub range: f32,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_bitlinear_config_default() {
        let config = BitLinearConfig::default();
        assert!(config.enable_layernorm);
        assert!(config.enable_residual);
        assert_eq!(config.layernorm_eps, 1e-5);
    }

    #[test]
    fn test_activation_stats_default() {
        let stats = ActivationStats::default();
        assert_eq!(stats.running_mean, 0.0);
        assert_eq!(stats.running_var, 1.0);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.min_val, f32::INFINITY);
        assert_eq!(stats.max_val, f32::NEG_INFINITY);
    }

    #[test]
    fn test_bitlinear_ops_creation() {
        let ops = BitLinearTensorOpsImpl::default();
        assert!(ops.config.enable_layernorm);

        let custom_config = BitLinearConfig {
            enable_layernorm: false,
            ..Default::default()
        };

        let custom_ops = BitLinearTensorOpsImpl::new(custom_config);
        assert!(!custom_ops.config.enable_layernorm);
    }

    #[test]
    fn test_residual_connection_support() {
        let device = Device::Cpu;
        let config = BitLinearConfig::default();
        let ops = BitLinearTensorOpsImpl::new(config);

        let residual_support = ops.create_residual_support(device.clone());
        assert!(residual_support.enabled);
        assert_eq!(residual_support.scale_factor, 1.0);
        // Note: Device comparison removed due to lack of PartialEq implementation
    }
}
