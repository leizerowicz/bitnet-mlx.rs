//! BitNet Quantization Library
//!
//! This crate provides quantization utilities for BitNet models, implementing
//! the 1.58-bit quantization scheme and other quantization methods for neural networks.
//!
//! # Features
//!
//! - **1.58-bit quantization**: Ternary quantization with values {-1, 0, +1}
//! - **Weight quantization**: Specialized quantization for neural network weights
//! - **Activation quantization**: Dynamic quantization for activations with calibration
//! - **Quantization utilities**: Helper functions, error handling, and bit manipulation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use bitnet_quant::quantization::{
//!     QuantizerFactory, WeightQuantizationConfig, ActivationQuantizationConfig,
//!     QuantizationPrecision, QuantizationStrategy
//! };
//! use candle_core::{Tensor, Device};
//!
//! // Create a weight quantizer
//! let weight_config = WeightQuantizationConfig::default();
//! let weight_quantizer = QuantizerFactory::create_weight_quantizer(weight_config);
//!
//! // Create an activation quantizer
//! let activation_config = ActivationQuantizationConfig::default();
//! let activation_quantizer = QuantizerFactory::create_activation_quantizer(activation_config);
//! ```
//!
//! # Module Organization
//!
//! - [`quantization`]: Core quantization module with traits and implementations
//! - [`quantization::weights`]: Weight-specific quantization
//! - [`quantization::activations`]: Activation-specific quantization
//! - [`quantization::utils`]: Utilities and helper functions

pub mod bitlinear;
pub mod calibration;
pub mod metrics;
pub mod quantization;
pub mod simd;
pub mod tensor_integration;

// Re-export commonly used items for convenience
pub use quantization::{
    QuantizationConfig, QuantizationPrecision, QuantizationResult, QuantizationStats,
    QuantizationStrategy, Quantizer, QuantizerFactory,
};

pub use bitlinear::{
    BitLinear, BitLinearConfig, BitLinearError, BitLinearForward, BitLinearResult, CacheConfig,
    CacheEntry, QuantizedWeightCache,
};

pub use quantization::weights::{
    absmean_quantize_weights, create_ternary_quantizer, quantize_weights_ternary, QuantizedWeight,
    TernaryMethod, TernaryStats, WeightQuantizationConfig, WeightQuantizer,
};

pub use quantization::packing::{
    PackedTernaryWeights, PackingSavingsEstimate, TernaryPacker, TernaryPackerFactory,
    TernaryPackingConfig, TernaryPackingStrategy,
};

pub use quantization::simd_unpacking::{simd_unpack_weights, SimdCapabilities, SimdUnpacker};

pub use quantization::activations::{
    absmax_quantize_activations, ActivationQuantizationConfig, ActivationQuantizer,
    QuantizedActivation,
};

pub use quantization::utils::{
    BitUtils, CalibrationUtils, MemorySavingsEstimate, QuantizationError, QuantizationUtils,
    ScalingFactor,
};

pub use quantization::corruption_detection::{
    CorruptionDetector, CorruptionReport, CorruptionSeverity, CorruptionType, RecoveryAction,
    RecoveryPlan, StrategyValidator,
};

pub use quantization::config::{
    ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
    AttentionQuantizationConfig, ConfigValidationError, PackingConfig as EnhancedPackingConfig,
    QuantizationConfig as EnhancedQuantizationConfig, QuantizationConfigBuilder, SimdConfig,
    WeightQuantizationConfig as EnhancedWeightQuantizationConfig, WeightQuantizationConfigBuilder,
};

pub use quantization::schemes::{
    BinaryThresholdMethod, ConfigurableQuantizationScheme, MultiBitParams, OneBitParams,
    OneFiveEightBitParams, OptimizationConfig, QuantizationSchemeConfig, QuantizationSchemeFactory,
    QuantizedTensor, SchemeParameters, ThresholdConfig,
};

pub use quantization::precision_control::{
    create_aggressive_precision_controller, create_conservative_precision_controller,
    create_precision_controller, AdjustmentReason, AdjustmentStrategy, AlertThresholds,
    DynamicAdjustmentConfig, MetricsHistory, PerformanceImpact, PerformanceSummary,
    PerformanceThresholds, PrecisionAdjustment, PrecisionBounds, PrecisionControlConfig,
    PrecisionController, PrecisionMetric, PrecisionMonitoringConfig, PrecisionState,
    PrecisionValidationConfig,
};

pub use quantization::enhanced_config::{
    create_custom_enhanced_config, create_enhanced_config, ConfigurationPreset,
    EnhancedQuantizationConfigBuilder, EnhancedQuantizationConfiguration,
};

pub use metrics::QuantizationMetrics;

/// Tensor integration re-exports
pub use tensor_integration::{
    GlobalTensorIntegrationConfig, QuantizationAwareTensorOps, TensorIntegrationError,
    TensorIntegrationFactory, TensorIntegrationResult,
};

pub use tensor_integration::bitnet_ops::{BitNetQuantizationConfig, BitNetTensorOps};

pub use tensor_integration::bitlinear_tensor::{
    ActivationQuantizationTensor, BitLinearTensorError, LayerNormIntegration,
    WeightQuantizationTensor,
};

pub use tensor_integration::calibration_tensor::{
    CalibrationConfig, CalibrationDataset, CalibrationError, CalibrationResults, CalibrationTensor,
    StatisticsCollector,
};

// pub use tensor_integration::qat_tensor::{
//     QATensor, QATConfig, StraightThroughEstimator,
//     GradientEstimation, QATError, TrainingAwareQuantization,
// };

pub use tensor_integration::precision_tensor::{
    AccuracyOptimizedPrecision, LayerType, MixedPrecisionError, MixedPrecisionTensor,
    PerformanceOptimizedPrecision, PrecisionConfig, PrecisionPolicy, PrecisionTensorOps,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::quantization::{
        QuantizationConfig, QuantizationPrecision, QuantizationResult, QuantizationStrategy,
        Quantizer, QuantizerFactory,
    };

    pub use crate::quantization::weights::{
        absmean_quantize_weights, create_ternary_quantizer, QuantizedWeight, TernaryMethod,
        TernaryStats, WeightQuantizationConfig, WeightQuantizer,
    };

    pub use crate::quantization::packing::{
        PackedTernaryWeights, PackingSavingsEstimate, TernaryPacker, TernaryPackerFactory,
        TernaryPackingConfig, TernaryPackingStrategy,
    };

    pub use crate::quantization::simd_unpacking::{
        simd_unpack_weights, SimdCapabilities, SimdUnpacker,
    };

    pub use crate::quantization::activations::{
        absmax_quantize_activations, ActivationQuantizationConfig, ActivationQuantizer,
        QuantizedActivation,
    };

    pub use crate::quantization::utils::{QuantizationError, QuantizationUtils, ScalingFactor};

    pub use crate::quantization::corruption_detection::{
        CorruptionDetector, CorruptionReport, CorruptionSeverity, CorruptionType, RecoveryAction,
        RecoveryPlan,
    };

    pub use crate::quantization::config::{
        ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
        AttentionQuantizationConfig, ConfigValidationError, PackingConfig as EnhancedPackingConfig,
        QuantizationConfig as EnhancedQuantizationConfig, QuantizationConfigBuilder, SimdConfig,
        WeightQuantizationConfig as EnhancedWeightQuantizationConfig,
        WeightQuantizationConfigBuilder,
    };

    pub use crate::quantization::schemes::{
        BinaryThresholdMethod, ConfigurableQuantizationScheme, MultiBitParams, OneBitParams,
        OneFiveEightBitParams, OptimizationConfig, QuantizationSchemeConfig,
        QuantizationSchemeFactory, QuantizedTensor, SchemeParameters, ThresholdConfig,
    };

    pub use crate::quantization::precision_control::{
        create_aggressive_precision_controller, create_conservative_precision_controller,
        create_precision_controller, AdjustmentStrategy, DynamicAdjustmentConfig,
        PerformanceSummary, PrecisionBounds, PrecisionControlConfig, PrecisionController,
        PrecisionMetric, PrecisionState,
    };

    pub use crate::metrics::{
        ErrorThresholds, ExportFormat, LayerErrorAnalysis, MetricsCalculator, MetricsConfig,
        MetricsExporter, MitigationStrategy, QuantizationMetrics,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_precision_enum() {
        let precision = QuantizationPrecision::OneFiveFiveBit;
        assert_eq!(format!("{:?}", precision), "OneFiveFiveBit");
    }

    #[test]
    fn test_quantization_config_creation() {
        let config = QuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.strategy, QuantizationStrategy::Symmetric);
    }

    #[test]
    fn test_weight_quantizer_factory() {
        let config = WeightQuantizationConfig::default();
        let _quantizer = QuantizerFactory::create_weight_quantizer(config);
        // Test passes if no panic occurs
    }

    #[test]
    fn test_activation_quantizer_factory() {
        let config = ActivationQuantizationConfig::default();
        let _quantizer = QuantizerFactory::create_activation_quantizer(config);
        // Test passes if no panic occurs
    }

    #[test]
    fn test_scaling_factor_creation() {
        let scale = ScalingFactor::per_tensor(2.0);
        assert_eq!(scale.value, 2.0);
        assert!(!scale.per_channel);
    }
}
