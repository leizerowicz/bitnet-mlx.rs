//! Tensor Integration Module for BitNet Quantization
//!
//! This module provides comprehensive integration between BitNet tensor operations
//! and quantization systems, enabling seamless quantization-aware tensor operations
//! with production-ready performance and memory efficiency.
//!
//! # Features
//!
//! - **BitNet-specific tensor operations**: 1.58-bit quantized arithmetic
//! - **Quantized tensor structures**: Efficient storage and manipulation
//! - **Device-aware quantization**: MLX/Metal acceleration integration
//! - **Memory-efficient operations**: Integration with HybridMemoryPool
//! - **Production-ready error handling**: Comprehensive error recovery
//!
//! # Module Organization
//!
//! - [`bitnet_ops`]: BitNet-specific quantized tensor operations
//! - [`quantized_tensor`]: Core quantized tensor implementation
//! - [`bitlinear_tensor`]: BitLinear layer tensor operations
//! - [`calibration_tensor`]: Calibration dataset tensor processing
//! - [`qat_tensor`]: Quantization-Aware Training tensor operations
//! - [`precision_tensor`]: Mixed precision tensor support

pub mod bitlinear_tensor;
pub mod bitnet_ops;
pub mod calibration_tensor;
pub mod precision_tensor;
pub mod qat_tensor;
pub mod quantized_tensor;

// Re-export core types
pub use bitnet_ops::{BitNetQuantizationConfig, BitNetTensorOps};

pub use quantized_tensor::{
    CompressionRatio, DequantizationStrategy, QuantizationParameters, QuantizedLayout,
    QuantizedStorage, QuantizedTensor, QuantizedTensorConfig, QuantizedTensorError, ScaleZeroPoint,
};

// pub use bitlinear_tensor::{
//     BitLinearTensorOps, BitLinearConfig, WeightQuantizationTensor,
//     ActivationQuantizationTensor, BitLinearTensorError, LayerNormIntegration,
//     ResidualConnectionSupport
// };

pub use bitlinear_tensor::{
    ActivationQuantizationTensor, ActivationStats, BitLinearConfig, BitLinearTensorError,
    BitLinearTensorOpsImpl as BitLinearTensorOps, HardwareDeviceType, HardwareProfile,
    InstructionSet, LayerNormIntegration, MixedPrecisionBitLinearOps, MixedPrecisionConfig,
    MixedPrecisionStats, PrecisionSummary, ResidualConnectionSupport, StatsSummary,
    WeightQuantizationTensor,
};

pub use qat_tensor::{
    GradientStats, QATConfig, QATError, QATTensorOps, QATTensorType,
    QuantizationParameters as QATQuantizationParameters, QuantizationStats,
    StraightThroughEstimator,
};

pub use calibration_tensor::{
    CalibrationConfig, CalibrationDataset, CalibrationError, CalibrationTensor, DatasetProcessor,
    DistributionAnalysis, StatisticalMoments, StatisticsCollector,
};

// pub use qat_tensor::{
//     QATensor, QATConfig, StraightThroughEstimator,
//     GradientEstimation, QATError, TrainingAwareQuantization,
//     QuantizationAwareGradients, BackwardPassQuantization
// };

pub use precision_tensor::{
    AccuracyOptimizedPrecision, MixedPrecisionError, MixedPrecisionTensor,
    PerformanceOptimizedPrecision, PrecisionConfig, PrecisionPolicy, PrecisionSelector,
    PrecisionTensorOps,
};

use candle_core::Device;
use std::sync::Arc;

use crate::quantization::{
    QuantizationConfig, QuantizationError, QuantizationPrecision, QuantizationStrategy,
};
use bitnet_core::BitNetTensor;

/// Core tensor integration error types
#[derive(Debug, thiserror::Error)]
pub enum TensorIntegrationError {
    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),

    #[error("Memory error: {0}")]
    Memory(#[from] bitnet_core::MemoryError),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tensor operation error: {message}")]
    TensorOp { message: String },

    #[error("Device compatibility error: expected {expected:?}, found {found:?}")]
    DeviceCompatibility { expected: Device, found: Device },

    #[error("Shape mismatch error: {message}")]
    ShapeMismatch { message: String },

    #[error("Quantization parameter mismatch: {message}")]
    QuantizationMismatch { message: String },

    #[error("Unsupported operation: {operation} for quantization precision {precision:?}")]
    UnsupportedOperation {
        operation: String,
        precision: QuantizationPrecision,
    },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Tensor operation failed: {message}")]
    TensorOperation { message: String },
}

/// Result type for tensor integration operations
pub type TensorIntegrationResult<T> = std::result::Result<T, TensorIntegrationError>;

/// Trait for tensor operations that support quantization
pub trait QuantizationAwareTensorOps {
    /// The underlying tensor type
    type Tensor;

    /// Quantize a tensor with the given configuration
    fn quantize(
        &self,
        tensor: &Self::Tensor,
        config: &QuantizationConfig,
    ) -> TensorIntegrationResult<QuantizedTensor>;

    /// Dequantize a tensor back to full precision
    fn dequantize(&self, tensor: &QuantizedTensor) -> TensorIntegrationResult<Self::Tensor>;

    /// Perform quantized arithmetic operation
    fn quantized_add(
        &self,
        lhs: &QuantizedTensor,
        rhs: &QuantizedTensor,
    ) -> TensorIntegrationResult<QuantizedTensor>;

    /// Perform quantized matrix multiplication
    fn quantized_matmul(
        &self,
        lhs: &QuantizedTensor,
        rhs: &QuantizedTensor,
    ) -> TensorIntegrationResult<QuantizedTensor>;

    /// Check if two quantized tensors are compatible for operations
    fn are_compatible(&self, lhs: &QuantizedTensor, rhs: &QuantizedTensor) -> bool;

    /// Convert between quantization precisions
    fn convert_precision(
        &self,
        tensor: &QuantizedTensor,
        target_precision: QuantizationPrecision,
    ) -> TensorIntegrationResult<QuantizedTensor>;
}

/// Factory for creating quantization-aware tensor operations
pub struct TensorIntegrationFactory;

impl TensorIntegrationFactory {
    /// Creates a new BitNet tensor operations implementation
    pub fn create_bitnet_ops() -> Arc<bitnet_ops::BitNetTensorOps> {
        Arc::new(bitnet_ops::BitNetTensorOps::new())
    }

    /// Create quantized tensor from BitNet tensor
    pub fn create_quantized_tensor(
        tensor: BitNetTensor,
        config: QuantizedTensorConfig,
    ) -> TensorIntegrationResult<QuantizedTensor> {
        quantized_tensor::QuantizedTensor::from_bitnet_tensor(tensor, config)
    }

    /// Create BitLinear tensor operations
    pub fn create_bitlinear_ops() -> Arc<bitlinear_tensor::BitLinearTensorOpsImpl> {
        Arc::new(bitlinear_tensor::BitLinearTensorOpsImpl::default())
    }

    /// Create calibration tensor processor
    pub fn create_calibration_processor(
        config: calibration_tensor::CalibrationConfig,
    ) -> calibration_tensor::CalibrationTensor {
        calibration_tensor::CalibrationTensor::new(config)
    }

    /// Create QAT tensor operations
    // pub fn create_qat_ops(config: qat_tensor::QATConfig)
    //     -> Arc<qat_tensor::QATensorOpsImpl> {
    //     Arc::new(qat_tensor::QATensorOpsImpl::new(config))
    // }

    /// Create mixed precision tensor operations
    pub fn create_mixed_precision_ops(
        config: precision_tensor::PrecisionConfig,
    ) -> Arc<precision_tensor::MixedPrecisionTensorOpsImpl> {
        Arc::new(precision_tensor::MixedPrecisionTensorOpsImpl::new(config))
    }
}

/// Global tensor integration configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GlobalTensorIntegrationConfig {
    /// Default quantization precision
    pub default_precision: QuantizationPrecision,

    /// Default quantization strategy
    pub default_strategy: QuantizationStrategy,

    /// Memory pool configuration for quantized tensors
    pub memory_pool_config: MemoryPoolIntegrationConfig,

    /// Device acceleration preferences
    pub acceleration_config: AccelerationIntegrationConfig,

    /// Error handling configuration
    pub error_handling_config: ErrorHandlingConfig,
}

impl Default for GlobalTensorIntegrationConfig {
    fn default() -> Self {
        Self {
            default_precision: QuantizationPrecision::OneFiveFiveBit,
            default_strategy: QuantizationStrategy::Symmetric,
            memory_pool_config: MemoryPoolIntegrationConfig::default(),
            acceleration_config: AccelerationIntegrationConfig::default(),
            error_handling_config: ErrorHandlingConfig::default(),
        }
    }
}

/// Memory pool integration configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryPoolIntegrationConfig {
    /// Use existing HybridMemoryPool for quantized tensors
    pub use_hybrid_pool: bool,

    /// Quantized tensor allocation preferences
    pub quantized_allocation_strategy: QuantizedAllocationStrategy,

    /// Memory alignment for quantized data
    pub alignment_bytes: usize,

    /// Enable memory usage tracking
    pub enable_usage_tracking: bool,
}

impl Default for MemoryPoolIntegrationConfig {
    fn default() -> Self {
        Self {
            use_hybrid_pool: true,
            quantized_allocation_strategy: QuantizedAllocationStrategy::PackedOptimal,
            alignment_bytes: 64, // Cache line alignment
            enable_usage_tracking: true,
        }
    }
}

/// Quantized tensor memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum QuantizedAllocationStrategy {
    /// Optimize for memory usage
    MemoryOptimal,

    /// Optimize for access speed
    SpeedOptimal,

    /// Balance memory and speed
    Balanced,

    /// Pack data optimally for quantization level
    PackedOptimal,
}

/// Device acceleration integration configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AccelerationIntegrationConfig {
    /// Enable MLX acceleration for Apple Silicon
    pub enable_mlx: bool,

    /// Enable Metal GPU acceleration
    pub enable_metal: bool,

    /// Enable SIMD optimization
    pub enable_simd: bool,

    /// Automatic fallback strategy
    pub auto_fallback: bool,

    /// Minimum tensor size for acceleration
    pub acceleration_threshold: usize,
}

impl Default for AccelerationIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_mlx: true,
            enable_metal: true,
            enable_simd: true,
            auto_fallback: true,
            acceleration_threshold: 1024, // Minimum elements for acceleration
        }
    }
}

/// Error handling configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ErrorHandlingConfig {
    /// Enable automatic error recovery
    pub auto_recovery: bool,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Enable detailed error logging
    pub detailed_logging: bool,

    /// Fallback to CPU on GPU errors
    pub gpu_fallback: bool,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            max_retries: 3,
            detailed_logging: true,
            gpu_fallback: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_integration_factory_creation() {
        let bitnet_ops = TensorIntegrationFactory::create_bitnet_ops();
        assert!(!std::ptr::eq(bitnet_ops.as_ref(), std::ptr::null()));

        let bitlinear_ops = TensorIntegrationFactory::create_bitlinear_ops();
        assert!(!std::ptr::eq(bitlinear_ops.as_ref(), std::ptr::null()));
    }

    #[test]
    fn test_global_config_defaults() {
        let config = GlobalTensorIntegrationConfig::default();
        assert_eq!(
            config.default_precision,
            QuantizationPrecision::OneFiveFiveBit
        );
        assert_eq!(config.default_strategy, QuantizationStrategy::Symmetric);
        assert!(config.memory_pool_config.use_hybrid_pool);
        assert!(config.acceleration_config.enable_mlx);
    }

    #[test]
    fn test_error_types() {
        let error = TensorIntegrationError::TensorOp {
            message: "Test error".to_string(),
        };
        assert!(error.to_string().contains("Test error"));

        let error = TensorIntegrationError::UnsupportedOperation {
            operation: "custom_op".to_string(),
            precision: QuantizationPrecision::OneFiveFiveBit,
        };
        assert!(error.to_string().contains("custom_op"));
    }
}
