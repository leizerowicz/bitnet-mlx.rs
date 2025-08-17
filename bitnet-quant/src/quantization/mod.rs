//! Core quantization module for BitNet models
//! 
//! This module provides the fundamental quantization traits, types, and operations
//! for implementing 1.58-bit quantization in BitNet neural networks.

pub mod weights;
pub mod activations;
pub mod utils;
pub mod packing;
pub mod simd_unpacking;
pub mod corruption_detection;
pub mod config;
pub mod mixed_precision;
pub mod schemes;
pub mod precision_control;
pub mod enhanced_config;

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// Re-export commonly used items
pub use weights::{
    WeightQuantizer, WeightQuantizationConfig, QuantizedWeight,
    TernaryMethod, TernaryStats, create_ternary_quantizer, quantize_weights_ternary
};
pub use activations::{ActivationQuantizer, ActivationQuantizationConfig, QuantizedActivation};
pub use utils::{QuantizationUtils, ScalingFactor, QuantizationError};
pub use packing::{
    TernaryPackingStrategy, TernaryPackingConfig, PackedTernaryWeights,
    TernaryPacker, TernaryPackerFactory, PackingSavingsEstimate
};

pub use simd_unpacking::{
    SimdUnpacker, SimdCapabilities, simd_unpack_weights
};

pub use corruption_detection::{
    CorruptionDetector, CorruptionReport, CorruptionType, CorruptionSeverity,
    RecoveryAction, RecoveryPlan, StrategyValidator
};

pub use config::{
    QuantizationConfig as EnhancedQuantizationConfig,
    WeightQuantizationConfig as EnhancedWeightQuantizationConfig,
    ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
    AttentionQuantizationConfig, PackingConfig as EnhancedPackingConfig, SimdConfig,
    ConfigValidationError, QuantizationConfigBuilder, WeightQuantizationConfigBuilder
};

pub use schemes::{
    ConfigurableQuantizationScheme, QuantizationSchemeConfig, QuantizationSchemeFactory,
    QuantizedTensor, SchemeParameters, OneBitParams, OneFiveEightBitParams, MultiBitParams,
    BinaryThresholdMethod, ThresholdConfig, OptimizationConfig
};

pub use precision_control::{
    PrecisionController, PrecisionControlConfig, PrecisionBounds, DynamicAdjustmentConfig,
    PrecisionMonitoringConfig, PrecisionValidationConfig, PerformanceThresholds,
    AdjustmentStrategy, PrecisionMetric, AlertThresholds, PrecisionState, MetricsHistory,
    PrecisionAdjustment, AdjustmentReason, PerformanceImpact, PerformanceSummary,
    create_precision_controller, create_conservative_precision_controller,
    create_aggressive_precision_controller
};

pub use enhanced_config::{
    EnhancedQuantizationConfigBuilder, EnhancedQuantizationConfiguration,
    ConfigurationPreset, create_enhanced_config, create_custom_enhanced_config
};

/// Core quantization precision for BitNet models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 1.58-bit quantization (ternary: -1, 0, +1)
    OneFiveFiveBit,
    /// 1-bit quantization (binary: -1, +1)
    OneBit,
    /// 2-bit quantization
    TwoBit,
    /// 4-bit quantization
    FourBit,
    /// 8-bit quantization
    EightBit,
}

/// Quantization strategy for different model components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Symmetric quantization around zero
    Symmetric,
    /// Asymmetric quantization with offset
    Asymmetric,
    /// Dynamic quantization based on input statistics
    Dynamic,
    /// Static quantization with pre-computed parameters
    Static,
}

/// Core trait for all quantization operations
pub trait Quantizer: Debug + Send + Sync {
    type Input;
    type Output;
    type Config;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Quantize the input tensor
    fn quantize(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Dequantize back to original precision
    fn dequantize(&self, quantized: &Self::Output) -> Result<Self::Input, Self::Error>;
    
    /// Get the quantization configuration
    fn config(&self) -> &Self::Config;
    
    /// Validate input tensor for quantization
    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error>;
    
    /// Get quantization statistics
    fn get_stats(&self) -> QuantizationStats;
}

/// Statistics collected during quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Number of elements quantized
    pub elements_count: usize,
    /// Quantization error (MSE)
    pub quantization_error: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Min value in original tensor
    pub min_value: f32,
    /// Max value in original tensor
    pub max_value: f32,
    /// Scale factor used
    pub scale_factor: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: Option<i32>,
}

impl Default for QuantizationStats {
    fn default() -> Self {
        Self {
            elements_count: 0,
            quantization_error: 0.0,
            compression_ratio: 1.0,
            min_value: 0.0,
            max_value: 0.0,
            scale_factor: 1.0,
            zero_point: None,
        }
    }
}

/// Configuration for quantization operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization precision
    pub precision: QuantizationPrecision,
    /// Quantization strategy
    pub strategy: QuantizationStrategy,
    /// Whether to use per-channel quantization
    pub per_channel: bool,
    /// Clipping threshold for outliers
    pub clip_threshold: Option<f32>,
    /// Whether to enable quantization-aware training
    pub qat_enabled: bool,
    /// Calibration dataset size for dynamic quantization
    pub calibration_size: Option<usize>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            per_channel: false,
            clip_threshold: None,
            qat_enabled: false,
            calibration_size: None,
        }
    }
}

/// Result type for quantization operations
pub type QuantizationResult<T> = Result<T, QuantizationError>;

/// Trait for quantization-aware operations
pub trait QuantizationAware {
    /// Apply quantization noise during training
    fn apply_quantization_noise(&mut self, noise_scale: f32) -> QuantizationResult<()>;
    
    /// Get gradient scaling factor for quantized parameters
    fn get_gradient_scale(&self) -> f32;
    
    /// Update quantization parameters based on gradients
    fn update_quantization_params(&mut self, learning_rate: f32) -> QuantizationResult<()>;
}

/// Trait for calibration-based quantization
pub trait CalibrationQuantizer: Quantizer {
    type CalibrationData;
    
    /// Calibrate quantization parameters using sample data
    fn calibrate(&mut self, data: &[Self::CalibrationData]) -> QuantizationResult<()>;
    
    /// Check if calibration is required
    fn needs_calibration(&self) -> bool;
    
    /// Reset calibration state
    fn reset_calibration(&mut self);
}

/// Factory for creating quantizers
pub struct QuantizerFactory;

impl QuantizerFactory {
    /// Create a weight quantizer with the given configuration
    pub fn create_weight_quantizer(config: WeightQuantizationConfig) -> QuantizationResult<Box<dyn WeightQuantizer>> {
        weights::create_weight_quantizer(config)
    }
    
    /// Create an activation quantizer with the given configuration
    pub fn create_activation_quantizer(config: ActivationQuantizationConfig) -> QuantizationResult<Box<dyn ActivationQuantizer>> {
        activations::create_activation_quantizer(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_precision_debug() {
        let precision = QuantizationPrecision::OneFiveFiveBit;
        let debug_str = format!("{:?}", precision);
        assert!(debug_str.contains("OneFiveFiveBit"));
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.strategy, QuantizationStrategy::Symmetric);
        assert!(!config.per_channel);
    }

    #[test]
    fn test_quantization_stats_default() {
        let stats = QuantizationStats::default();
        assert_eq!(stats.elements_count, 0);
        assert_eq!(stats.compression_ratio, 1.0);
        assert_eq!(stats.scale_factor, 1.0);
    }
}