//! Enhanced Configuration Builders with Precision Control Integration
//!
//! This module provides enhanced configuration builders that integrate precision control
//! capabilities with the existing quantization configuration system.

use super::{
    QuantizationPrecision, QuantizationStrategy, QuantizationConfig,
    weights::{WeightQuantizationConfig, TernaryMethod},
    activations::ActivationQuantizationConfig,
    config::{
        QuantizationConfig as EnhancedQuantizationConfig,
        WeightQuantizationConfig as EnhancedWeightQuantizationConfig,
        ActivationQuantizationConfig as EnhancedActivationQuantizationConfig,
        PackingConfig, SimdConfig, ConfigValidationError,
    },
    precision_control::{
        PrecisionControlConfig, PrecisionBounds, DynamicAdjustmentConfig,
        PrecisionMonitoringConfig, PrecisionValidationConfig, PerformanceThresholds,
        AdjustmentStrategy, PrecisionMetric, AlertThresholds,
    },
    QuantizationResult, QuantizationError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced quantization configuration builder with precision control
#[derive(Debug, Default)]
pub struct EnhancedQuantizationConfigBuilder {
    // Base configuration
    precision: Option<QuantizationPrecision>,
    strategy: Option<QuantizationStrategy>,
    per_channel: Option<bool>,
    clip_threshold: Option<f32>,
    qat_enabled: Option<bool>,
    calibration_size: Option<usize>,
    seed: Option<u64>,
    verbose: Option<bool>,
    
    // Precision control
    precision_control: Option<PrecisionControlConfig>,
    precision_bounds: Option<PrecisionBounds>,
    dynamic_adjustment: Option<DynamicAdjustmentConfig>,
    monitoring: Option<PrecisionMonitoringConfig>,
    validation: Option<PrecisionValidationConfig>,
    performance_thresholds: Option<PerformanceThresholds>,
    
    // Advanced features
    auto_optimization: Option<bool>,
    adaptive_thresholds: Option<bool>,
    real_time_monitoring: Option<bool>,
    custom_metrics: Option<Vec<String>>,
}

impl EnhancedQuantizationConfigBuilder {
    /// Create a new enhanced configuration builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set quantization precision
    pub fn precision(mut self, precision: QuantizationPrecision) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Set quantization strategy
    pub fn strategy(mut self, strategy: QuantizationStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Enable per-channel quantization
    pub fn per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = Some(enabled);
        self
    }

    /// Set clipping threshold
    pub fn clip_threshold(mut self, threshold: f32) -> Self {
        self.clip_threshold = Some(threshold);
        self
    }

    /// Enable quantization-aware training
    pub fn qat_enabled(mut self, enabled: bool) -> Self {
        self.qat_enabled = Some(enabled);
        self
    }

    /// Set calibration size
    pub fn calibration_size(mut self, size: usize) -> Self {
        self.calibration_size = Some(size);
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.verbose = Some(enabled);
        self
    }

    /// Set precision control configuration
    pub fn precision_control(mut self, config: PrecisionControlConfig) -> Self {
        self.precision_control = Some(config);
        self
    }

    /// Set precision bounds
    pub fn precision_bounds(mut self, bounds: PrecisionBounds) -> Self {
        self.precision_bounds = Some(bounds);
        self
    }

    /// Configure dynamic adjustment
    pub fn dynamic_adjustment(mut self, config: DynamicAdjustmentConfig) -> Self {
        self.dynamic_adjustment = Some(config);
        self
    }

    /// Configure monitoring
    pub fn monitoring(mut self, config: PrecisionMonitoringConfig) -> Self {
        self.monitoring = Some(config);
        self
    }

    /// Configure validation
    pub fn validation(mut self, config: PrecisionValidationConfig) -> Self {
        self.validation = Some(config);
        self
    }

    /// Set performance thresholds
    pub fn performance_thresholds(mut self, thresholds: PerformanceThresholds) -> Self {
        self.performance_thresholds = Some(thresholds);
        self
    }

    /// Enable automatic optimization
    pub fn auto_optimization(mut self, enabled: bool) -> Self {
        self.auto_optimization = Some(enabled);
        self
    }

    /// Enable adaptive thresholds
    pub fn adaptive_thresholds(mut self, enabled: bool) -> Self {
        self.adaptive_thresholds = Some(enabled);
        self
    }

    /// Enable real-time monitoring
    pub fn real_time_monitoring(mut self, enabled: bool) -> Self {
        self.real_time_monitoring = Some(enabled);
        self
    }

    /// Add custom metrics
    pub fn custom_metrics(mut self, metrics: Vec<String>) -> Self {
        self.custom_metrics = Some(metrics);
        self
    }

    /// Build the enhanced configuration
    pub fn build(self) -> QuantizationResult<EnhancedQuantizationConfiguration> {
        // Build base configuration
        let base_config = EnhancedQuantizationConfig {
            precision: self.precision.unwrap_or(QuantizationPrecision::OneFiveFiveBit),
            strategy: self.strategy.unwrap_or(QuantizationStrategy::Symmetric),
            per_channel: self.per_channel.unwrap_or(false),
            clip_threshold: self.clip_threshold,
            qat_enabled: self.qat_enabled.unwrap_or(false),
            calibration_size: self.calibration_size,
            seed: self.seed,
            verbose: self.verbose.unwrap_or(false),
        };

        // Build precision control configuration
        let precision_control = if let Some(config) = self.precision_control {
            config
        } else {
            let mut config = PrecisionControlConfig::default();
            config.target_precision = base_config.precision;
            
            if let Some(bounds) = self.precision_bounds {
                config.precision_bounds = bounds;
            }
            if let Some(adjustment) = self.dynamic_adjustment {
                config.dynamic_adjustment = adjustment;
            }
            if let Some(monitoring) = self.monitoring {
                config.monitoring = monitoring;
            }
            if let Some(validation) = self.validation {
                config.validation = validation;
            }
            if let Some(thresholds) = self.performance_thresholds {
                config.performance_thresholds = thresholds;
            }
            
            config
        };

        // Build enhanced configuration
        let enhanced_config = EnhancedQuantizationConfiguration {
            base: base_config,
            precision_control,
            auto_optimization: self.auto_optimization.unwrap_or(false),
            adaptive_thresholds: self.adaptive_thresholds.unwrap_or(true),
            real_time_monitoring: self.real_time_monitoring.unwrap_or(false),
            custom_metrics: self.custom_metrics.unwrap_or_default(),
        };

        // Validate configuration
        enhanced_config.validate()?;

        Ok(enhanced_config)
    }

    /// Create a BitNet-optimized configuration
    pub fn bitnet_optimized() -> Self {
        Self::new()
            .precision(QuantizationPrecision::OneFiveFiveBit)
            .strategy(QuantizationStrategy::Symmetric)
            .precision_bounds(PrecisionBounds {
                min_precision: QuantizationPrecision::OneBit,
                max_precision: QuantizationPrecision::FourBit,
                min_threshold: 1e-6,
                max_threshold: 2.0,
                max_error_tolerance: 0.05,
                min_compression_ratio: 8.0,
                ..Default::default()
            })
            .dynamic_adjustment(DynamicAdjustmentConfig {
                enabled: true,
                strategy: AdjustmentStrategy::Adaptive,
                evaluation_window: 50,
                adjustment_frequency: 5,
                learning_rate: 0.1,
                stability_threshold: 0.01,
                max_adjustments: 3,
            })
            .auto_optimization(true)
            .adaptive_thresholds(true)
            .real_time_monitoring(true)
    }

    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self::new()
            .precision(QuantizationPrecision::OneBit)
            .strategy(QuantizationStrategy::Symmetric)
            .precision_bounds(PrecisionBounds {
                min_precision: QuantizationPrecision::OneBit,
                max_precision: QuantizationPrecision::TwoBit,
                max_error_tolerance: 0.1,
                min_compression_ratio: 16.0,
                ..Default::default()
            })
            .dynamic_adjustment(DynamicAdjustmentConfig {
                enabled: true,
                strategy: AdjustmentStrategy::Aggressive,
                evaluation_window: 20,
                adjustment_frequency: 2,
                learning_rate: 0.2,
                stability_threshold: 0.05,
                max_adjustments: 5,
            })
            .performance_thresholds(PerformanceThresholds {
                min_accuracy: 0.90,
                max_latency_ms: 5.0,
                max_memory_overhead_pct: 10.0,
                min_throughput: 2000.0,
                min_energy_efficiency: 0.9,
            })
            .auto_optimization(true)
            .real_time_monitoring(true)
    }

    /// Create an accuracy-optimized configuration
    pub fn accuracy_optimized() -> Self {
        Self::new()
            .precision(QuantizationPrecision::FourBit)
            .strategy(QuantizationStrategy::Asymmetric)
            .per_channel(true)
            .precision_bounds(PrecisionBounds {
                min_precision: QuantizationPrecision::TwoBit,
                max_precision: QuantizationPrecision::EightBit,
                max_error_tolerance: 0.01,
                min_compression_ratio: 2.0,
                ..Default::default()
            })
            .dynamic_adjustment(DynamicAdjustmentConfig {
                enabled: true,
                strategy: AdjustmentStrategy::Conservative,
                evaluation_window: 200,
                adjustment_frequency: 20,
                learning_rate: 0.05,
                stability_threshold: 0.005,
                max_adjustments: 2,
            })
            .performance_thresholds(PerformanceThresholds {
                min_accuracy: 0.98,
                max_latency_ms: 50.0,
                max_memory_overhead_pct: 50.0,
                min_throughput: 500.0,
                min_energy_efficiency: 0.7,
            })
            .auto_optimization(false)
            .adaptive_thresholds(true)
    }
}

/// Enhanced quantization configuration with precision control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuantizationConfiguration {
    /// Base quantization configuration
    pub base: EnhancedQuantizationConfig,
    /// Precision control configuration
    pub precision_control: PrecisionControlConfig,
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Enable adaptive thresholds
    pub adaptive_thresholds: bool,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Custom metrics to track
    pub custom_metrics: Vec<String>,
}

impl EnhancedQuantizationConfiguration {
    /// Validate the enhanced configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        // Validate base configuration
        self.base.validate()
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;
        
        // Validate precision control configuration
        self.precision_control.validate()?;
        
        // Check consistency between base and precision control
        if self.base.precision != self.precision_control.target_precision {
            return Err(QuantizationError::ConfigurationError(
                "Base precision and precision control target precision must match".to_string()
            ));
        }
        
        Ok(())
    }

    /// Convert to weight quantization configuration
    pub fn to_weight_config(&self) -> EnhancedWeightQuantizationConfig {
        EnhancedWeightQuantizationConfig {
            base: self.base.clone(),
            group_size: None,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            block_size: Some(64),
            ternary_method: TernaryMethod::MeanThreshold,
            custom_threshold_factor: Some(0.7),
            packing: PackingConfig::default(),
            freeze_weights: false,
            weight_decay: None,
            gradient_clip: None,
        }
    }

    /// Convert to activation quantization configuration
    pub fn to_activation_config(&self) -> EnhancedActivationQuantizationConfig {
        EnhancedActivationQuantizationConfig {
            base: self.base.clone(),
            moving_average_window: 100,
            outlier_percentile: 99.9,
            per_token: false,
            calibration_warmup: 50,
            ema_decay: 0.99,
            quantize_attention: true,
            attention: super::config::AttentionQuantizationConfig::default(),
            smooth_quantization: false,
            temperature: 1.0,
            enable_caching: false,
            cache_size_mb: None,
        }
    }

    /// Get recommended precision bounds based on target precision
    pub fn get_recommended_bounds(&self) -> PrecisionBounds {
        match self.base.precision {
            QuantizationPrecision::OneBit => PrecisionBounds {
                min_precision: QuantizationPrecision::OneBit,
                max_precision: QuantizationPrecision::OneFiveFiveBit,
                max_error_tolerance: 0.15,
                min_compression_ratio: 16.0,
                ..Default::default()
            },
            QuantizationPrecision::OneFiveFiveBit => PrecisionBounds {
                min_precision: QuantizationPrecision::OneBit,
                max_precision: QuantizationPrecision::TwoBit,
                max_error_tolerance: 0.08,
                min_compression_ratio: 10.0,
                ..Default::default()
            },
            QuantizationPrecision::TwoBit => PrecisionBounds {
                min_precision: QuantizationPrecision::OneFiveFiveBit,
                max_precision: QuantizationPrecision::FourBit,
                max_error_tolerance: 0.05,
                min_compression_ratio: 6.0,
                ..Default::default()
            },
            QuantizationPrecision::FourBit => PrecisionBounds {
                min_precision: QuantizationPrecision::TwoBit,
                max_precision: QuantizationPrecision::EightBit,
                max_error_tolerance: 0.02,
                min_compression_ratio: 3.0,
                ..Default::default()
            },
            QuantizationPrecision::EightBit => PrecisionBounds {
                min_precision: QuantizationPrecision::FourBit,
                max_precision: QuantizationPrecision::EightBit,
                max_error_tolerance: 0.01,
                min_compression_ratio: 2.0,
                ..Default::default()
            },
        }
    }
}

/// Configuration preset for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigurationPreset {
    /// BitNet-optimized for 1.58-bit quantization
    BitNetOptimized,
    /// Performance-optimized for maximum speed
    PerformanceOptimized,
    /// Accuracy-optimized for maximum precision
    AccuracyOptimized,
    /// Memory-optimized for minimal footprint
    MemoryOptimized,
    /// Balanced configuration
    Balanced,
    /// Custom configuration
    Custom,
}

impl ConfigurationPreset {
    /// Create configuration builder for the preset
    pub fn create_builder(self) -> EnhancedQuantizationConfigBuilder {
        match self {
            ConfigurationPreset::BitNetOptimized => {
                EnhancedQuantizationConfigBuilder::bitnet_optimized()
            },
            ConfigurationPreset::PerformanceOptimized => {
                EnhancedQuantizationConfigBuilder::performance_optimized()
            },
            ConfigurationPreset::AccuracyOptimized => {
                EnhancedQuantizationConfigBuilder::accuracy_optimized()
            },
            ConfigurationPreset::MemoryOptimized => {
                EnhancedQuantizationConfigBuilder::new()
                    .precision(QuantizationPrecision::OneBit)
                    .strategy(QuantizationStrategy::Symmetric)
                    .precision_bounds(PrecisionBounds {
                        min_precision: QuantizationPrecision::OneBit,
                        max_precision: QuantizationPrecision::OneFiveFiveBit,
                        max_error_tolerance: 0.2,
                        min_compression_ratio: 20.0,
                        ..Default::default()
                    })
                    .auto_optimization(true)
                    .real_time_monitoring(false) // Reduce overhead
            },
            ConfigurationPreset::Balanced => {
                EnhancedQuantizationConfigBuilder::new()
                    .precision(QuantizationPrecision::OneFiveFiveBit)
                    .strategy(QuantizationStrategy::Symmetric)
                    .auto_optimization(true)
                    .adaptive_thresholds(true)
            },
            ConfigurationPreset::Custom => {
                EnhancedQuantizationConfigBuilder::new()
            },
        }
    }

    /// Build configuration with the preset
    pub fn build(self) -> QuantizationResult<EnhancedQuantizationConfiguration> {
        self.create_builder().build()
    }
}

/// Factory functions for creating enhanced configurations
pub fn create_enhanced_config(
    preset: ConfigurationPreset,
) -> QuantizationResult<EnhancedQuantizationConfiguration> {
    preset.build()
}

/// Create a custom enhanced configuration
pub fn create_custom_enhanced_config<F>(
    builder_fn: F,
) -> QuantizationResult<EnhancedQuantizationConfiguration>
where
    F: FnOnce(EnhancedQuantizationConfigBuilder) -> EnhancedQuantizationConfigBuilder,
{
    let builder = EnhancedQuantizationConfigBuilder::new();
    builder_fn(builder).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_config_builder() {
        let config = EnhancedQuantizationConfigBuilder::new()
            .precision(QuantizationPrecision::OneFiveFiveBit)
            .strategy(QuantizationStrategy::Symmetric)
            .auto_optimization(true)
            .build()
            .unwrap();
        
        assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.base.strategy, QuantizationStrategy::Symmetric);
        assert!(config.auto_optimization);
    }

    #[test]
    fn test_bitnet_optimized_preset() {
        let config = ConfigurationPreset::BitNetOptimized.build().unwrap();
        assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert!(config.auto_optimization);
        assert!(config.adaptive_thresholds);
    }

    #[test]
    fn test_performance_optimized_preset() {
        let config = ConfigurationPreset::PerformanceOptimized.build().unwrap();
        assert_eq!(config.base.precision, QuantizationPrecision::OneBit);
        assert_eq!(config.precision_control.dynamic_adjustment.strategy, AdjustmentStrategy::Aggressive);
    }

    #[test]
    fn test_accuracy_optimized_preset() {
        let config = ConfigurationPreset::AccuracyOptimized.build().unwrap();
        assert_eq!(config.base.precision, QuantizationPrecision::FourBit);
        assert_eq!(config.precision_control.dynamic_adjustment.strategy, AdjustmentStrategy::Conservative);
    }

    #[test]
    fn test_config_validation() {
        let config = EnhancedQuantizationConfigBuilder::new()
            .precision(QuantizationPrecision::OneFiveFiveBit)
            .build()
            .unwrap();
        
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_weight_config_conversion() {
        let config = ConfigurationPreset::BitNetOptimized.build().unwrap();
        let weight_config = config.to_weight_config();
        
        assert_eq!(weight_config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert!(weight_config.normalize_weights);
    }

    #[test]
    fn test_activation_config_conversion() {
        let config = ConfigurationPreset::AccuracyOptimized.build().unwrap();
        let activation_config = config.to_activation_config();
        
        assert_eq!(activation_config.base.precision, QuantizationPrecision::FourBit);
        assert!(activation_config.quantize_attention);
    }

    #[test]
    fn test_recommended_bounds() {
        let config = EnhancedQuantizationConfigBuilder::new()
            .precision(QuantizationPrecision::OneBit)
            .build()
            .unwrap();
        
        let bounds = config.get_recommended_bounds();
        assert_eq!(bounds.min_precision, QuantizationPrecision::OneBit);
        assert_eq!(bounds.max_precision, QuantizationPrecision::OneFiveFiveBit);
    }

    #[test]
    fn test_custom_config_creation() {
        let config = create_custom_enhanced_config(|builder| {
            builder
                .precision(QuantizationPrecision::TwoBit)
                .auto_optimization(true)
                .adaptive_thresholds(false)
        }).unwrap();
        
        assert_eq!(config.base.precision, QuantizationPrecision::TwoBit);
        assert!(config.auto_optimization);
        assert!(!config.adaptive_thresholds);
    }
}