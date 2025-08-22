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

pub mod quantization;
pub mod bitlinear;
pub mod simd;
pub mod calibration;
pub mod metrics;
pub mod tensor_integration;

// Re-export commonly used items for convenience
pub use quantization::{
    QuantizationPrecision, QuantizationStrategy, QuantizationConfig,
    QuantizationStats, QuantizationResult, Quantizer, QuantizerFactory,
};

pub use bitlinear::{
    BitLinear, BitLinearConfig, BitLinearForward, BitLinearError, BitLinearResult,
    QuantizedWeightCache, CacheEntry, CacheConfig,
};

pub use quantization::weights::{
    WeightQuantizer, WeightQuantizationConfig, QuantizedWeight,
    TernaryMethod, TernaryStats, create_ternary_quantizer, quantize_weights_ternary,
    absmean_quantize_weights,
};

pub use quantization::packing::{
    TernaryPackingStrategy, TernaryPackingConfig, PackedTernaryWeights,
    TernaryPacker, TernaryPackerFactory, PackingSavingsEstimate,
};

pub use quantization::simd_unpacking::{
    SimdUnpacker, SimdCapabilities, simd_unpack_weights,
};

pub use quantization::activations::{
    ActivationQuantizer, ActivationQuantizationConfig, QuantizedActivation,
    absmax_quantize_activations,
};

pub use quantization::utils::{
    QuantizationError, ScalingFactor, QuantizationUtils,
    MemorySavingsEstimate, BitUtils, CalibrationUtils,
};

pub use quantization::corruption_detection::{
    CorruptionDetector, CorruptionReport, CorruptionType, CorruptionSeverity,
    RecoveryAction, RecoveryPlan, StrategyValidator,
};

pub use quantization::config::{
    QuantizationConfig as EnhancedQuantizationConfig,
    WeightQuantizationConfig as EnhancedWeightQuantizationConfig,
    ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
    AttentionQuantizationConfig, PackingConfig as EnhancedPackingConfig, SimdConfig,
    ConfigValidationError, QuantizationConfigBuilder, WeightQuantizationConfigBuilder,
};

pub use quantization::schemes::{
    ConfigurableQuantizationScheme, QuantizationSchemeConfig, QuantizationSchemeFactory,
    QuantizedTensor, SchemeParameters, OneBitParams, OneFiveEightBitParams, MultiBitParams,
    BinaryThresholdMethod, ThresholdConfig, OptimizationConfig,
};

pub use quantization::precision_control::{
    PrecisionController, PrecisionControlConfig, PrecisionBounds, DynamicAdjustmentConfig,
    PrecisionMonitoringConfig, PrecisionValidationConfig, PerformanceThresholds,
    AdjustmentStrategy, PrecisionMetric, AlertThresholds, PrecisionState, MetricsHistory,
    PrecisionAdjustment, AdjustmentReason, PerformanceImpact, PerformanceSummary,
    create_precision_controller, create_conservative_precision_controller,
    create_aggressive_precision_controller,
};

pub use quantization::enhanced_config::{
    EnhancedQuantizationConfigBuilder, EnhancedQuantizationConfiguration,
    ConfigurationPreset, create_enhanced_config, create_custom_enhanced_config,
};

pub use metrics::{
    QuantizationMetrics,
};

/// Tensor integration re-exports
pub use tensor_integration::{
    TensorIntegrationError, TensorIntegrationResult, QuantizationAwareTensorOps,
    TensorIntegrationFactory, GlobalTensorIntegrationConfig,
};

pub use tensor_integration::bitnet_ops::{
    BitNetTensorOps, 
    BitNetQuantizationConfig,
};

pub use tensor_integration::bitlinear_tensor::{
    WeightQuantizationTensor, ActivationQuantizationTensor, 
    BitLinearTensorError, LayerNormIntegration,
};

pub use tensor_integration::calibration_tensor::{
    CalibrationTensor, CalibrationConfig, CalibrationDataset,
    StatisticsCollector, CalibrationError, CalibrationResults,
};

// pub use tensor_integration::qat_tensor::{
//     QATensor, QATConfig, StraightThroughEstimator,
//     GradientEstimation, QATError, TrainingAwareQuantization,
// };

pub use tensor_integration::precision_tensor::{
    MixedPrecisionTensor, PrecisionPolicy, PrecisionTensorOps,
    PrecisionConfig, MixedPrecisionError, LayerType,
    PerformanceOptimizedPrecision, AccuracyOptimizedPrecision,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::quantization::{
        QuantizationPrecision, QuantizationStrategy, QuantizationConfig,
        Quantizer, QuantizerFactory, QuantizationResult,
    };
    
    pub use crate::quantization::weights::{
        WeightQuantizer, WeightQuantizationConfig, QuantizedWeight,
        TernaryMethod, TernaryStats, create_ternary_quantizer,
        absmean_quantize_weights,
    };
    
    pub use crate::quantization::packing::{
        TernaryPackingStrategy, TernaryPackingConfig, PackedTernaryWeights,
        TernaryPacker, TernaryPackerFactory, PackingSavingsEstimate,
    };
    
    pub use crate::quantization::simd_unpacking::{
        SimdUnpacker, SimdCapabilities, simd_unpack_weights,
    };
    
    pub use crate::quantization::activations::{
        ActivationQuantizer, ActivationQuantizationConfig, QuantizedActivation,
        absmax_quantize_activations,
    };
    
    pub use crate::quantization::utils::{
        QuantizationError, ScalingFactor, QuantizationUtils,
    };
    
    pub use crate::quantization::corruption_detection::{
        CorruptionDetector, CorruptionReport, CorruptionType, CorruptionSeverity,
        RecoveryAction, RecoveryPlan,
    };
    
    pub use crate::quantization::config::{
        QuantizationConfig as EnhancedQuantizationConfig,
        WeightQuantizationConfig as EnhancedWeightQuantizationConfig,
        ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
        AttentionQuantizationConfig, PackingConfig as EnhancedPackingConfig, SimdConfig,
        ConfigValidationError, QuantizationConfigBuilder, WeightQuantizationConfigBuilder,
    };
    
    pub use crate::quantization::schemes::{
        ConfigurableQuantizationScheme, QuantizationSchemeConfig, QuantizationSchemeFactory,
        QuantizedTensor, SchemeParameters, OneBitParams, OneFiveEightBitParams, MultiBitParams,
        BinaryThresholdMethod, ThresholdConfig, OptimizationConfig,
    };
    
    pub use crate::quantization::precision_control::{
        PrecisionController, PrecisionControlConfig, PrecisionBounds, DynamicAdjustmentConfig,
        AdjustmentStrategy, PrecisionMetric, PrecisionState, PerformanceSummary,
        create_precision_controller, create_conservative_precision_controller,
        create_aggressive_precision_controller,
    };

    pub use crate::metrics::{
        QuantizationMetrics, LayerErrorAnalysis, ErrorThresholds, MitigationStrategy,
        MetricsConfig, ExportFormat, MetricsCalculator, MetricsExporter,
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