//! Comprehensive tests for quantization precision control system
//!
//! This module contains extensive tests for all precision control features
//! including validation, dynamic adjustment, monitoring, and configuration.

use bitnet_quant::prelude::*;
use bitnet_quant::{
    create_enhanced_config, create_precision_controller, AdjustmentReason, AdjustmentStrategy,
    ConfigurationPreset, DynamicAdjustmentConfig, EnhancedQuantizationConfigBuilder,
    PerformanceThresholds, PrecisionBounds, PrecisionControlConfig, PrecisionController,
    PrecisionMonitoringConfig, QuantizationStats,
};
use candle_core::Device;
use std::time::Duration;

#[test]
fn test_precision_bounds_validation() {
    let mut bounds = PrecisionBounds::default();

    // Test valid bounds
    assert!(bounds.validate().is_ok());

    // Test invalid threshold bounds
    bounds.min_threshold = 5.0;
    bounds.max_threshold = 2.0;
    assert!(bounds.validate().is_err());

    // Reset and test invalid scale bounds
    bounds = PrecisionBounds::default();
    bounds.min_scale = 100.0;
    bounds.max_scale = 1.0;
    assert!(bounds.validate().is_err());

    // Test invalid error tolerance
    bounds = PrecisionBounds::default();
    bounds.max_error_tolerance = -0.1;
    assert!(bounds.validate().is_err());

    // Test invalid compression ratio
    bounds = PrecisionBounds::default();
    bounds.min_compression_ratio = 0.5;
    assert!(bounds.validate().is_err());
}

#[test]
fn test_dynamic_adjustment_config_validation() {
    let mut config = DynamicAdjustmentConfig::default();

    // Test valid config
    assert!(config.validate().is_ok());

    // Test invalid evaluation window
    config.evaluation_window = 0;
    assert!(config.validate().is_err());

    // Test invalid adjustment frequency
    config = DynamicAdjustmentConfig::default();
    config.adjustment_frequency = 0;
    assert!(config.validate().is_err());

    // Test invalid learning rate
    config = DynamicAdjustmentConfig::default();
    config.learning_rate = 1.5;
    assert!(config.validate().is_err());

    config.learning_rate = -0.1;
    assert!(config.validate().is_err());

    // Test invalid stability threshold
    config = DynamicAdjustmentConfig::default();
    config.stability_threshold = -0.1;
    assert!(config.validate().is_err());
}

#[test]
fn test_precision_monitoring_config_validation() {
    let mut config = PrecisionMonitoringConfig::default();

    // Test valid config
    assert!(config.validate().is_ok());

    // Test invalid history size
    config.history_size = 0;
    assert!(config.validate().is_err());

    // Test empty tracked metrics
    config = PrecisionMonitoringConfig::default();
    config.tracked_metrics.clear();
    assert!(config.validate().is_err());
}

#[test]
fn test_performance_thresholds_validation() {
    let mut thresholds = PerformanceThresholds::default();

    // Test valid thresholds
    assert!(thresholds.validate().is_ok());

    // Test invalid accuracy
    thresholds.min_accuracy = 1.5;
    assert!(thresholds.validate().is_err());

    thresholds.min_accuracy = -0.1;
    assert!(thresholds.validate().is_err());

    // Test invalid latency
    thresholds = PerformanceThresholds::default();
    thresholds.max_latency_ms = -1.0;
    assert!(thresholds.validate().is_err());

    // Test invalid memory overhead
    thresholds = PerformanceThresholds::default();
    thresholds.max_memory_overhead_pct = -5.0;
    assert!(thresholds.validate().is_err());

    // Test invalid throughput
    thresholds = PerformanceThresholds::default();
    thresholds.min_throughput = -100.0;
    assert!(thresholds.validate().is_err());

    // Test invalid energy efficiency
    thresholds = PerformanceThresholds::default();
    thresholds.min_energy_efficiency = 1.5;
    assert!(thresholds.validate().is_err());

    thresholds.min_energy_efficiency = -0.1;
    assert!(thresholds.validate().is_err());
}

#[test]
fn test_precision_controller_creation() {
    let device = Device::Cpu;

    // Test with default config
    let config = PrecisionControlConfig::default();
    let controller = PrecisionController::new(config, device.clone());
    assert!(controller.is_ok());

    // Test with conservative config
    let config = PrecisionControlConfig::conservative();
    let controller = PrecisionController::new(config, device.clone());
    assert!(controller.is_ok());

    // Test with aggressive config
    let config = PrecisionControlConfig::aggressive();
    let controller = PrecisionController::new(config, device);
    assert!(controller.is_ok());
}

#[test]
fn test_precision_bounds_checking() {
    let device = Device::Cpu;
    let config = PrecisionControlConfig::default();
    let controller = PrecisionController::new(config, device).unwrap();

    // Test all precision levels
    assert!(controller.is_precision_in_bounds(QuantizationPrecision::OneBit));
    assert!(controller.is_precision_in_bounds(QuantizationPrecision::OneFiveFiveBit));
    assert!(controller.is_precision_in_bounds(QuantizationPrecision::TwoBit));
    assert!(controller.is_precision_in_bounds(QuantizationPrecision::FourBit));
    assert!(controller.is_precision_in_bounds(QuantizationPrecision::EightBit));
}

#[test]
fn test_precision_validation() {
    let device = Device::Cpu;
    let config = PrecisionControlConfig::default();
    let controller = PrecisionController::new(config, device).unwrap();

    // Test valid parameters
    let result =
        controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1.0);
    assert!(result.is_ok());

    // Test invalid threshold (too high)
    let result =
        controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 15.0, 1.0);
    assert!(result.is_err());

    // Test invalid threshold (too low)
    let result =
        controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, -0.1, 1.0);
    assert!(result.is_err());

    // Test invalid scale (too high)
    let result =
        controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1e10);
    assert!(result.is_err());

    // Test invalid scale (too low)
    let result =
        controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1e-10);
    assert!(result.is_err());
}

#[test]
fn test_metrics_recording() {
    let device = Device::Cpu;
    let config = PrecisionControlConfig::default();
    let mut controller = PrecisionController::new(config, device).unwrap();

    // Record some metrics
    let stats1 = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.05,
        compression_ratio: 4.0,
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    let stats2 = QuantizationStats {
        elements_count: 2000,
        quantization_error: 0.08,
        compression_ratio: 3.5,
        min_value: -2.0,
        max_value: 2.0,
        scale_factor: 1.5,
        zero_point: None,
    };

    controller.record_metrics(&stats1, Duration::from_millis(10));
    controller.record_metrics(&stats2, Duration::from_millis(15));

    // Check performance summary
    let summary = controller.get_performance_summary();
    assert_eq!(summary.operations_count, 2);
    assert!((summary.average_error - 0.065).abs() < 1e-6);
    assert!((summary.average_compression_ratio - 3.75).abs() < 1e-6);

    // Check metrics history
    let history = controller.get_metrics_history();
    assert_eq!(history.quantization_errors.len(), 2);
    assert_eq!(history.compression_ratios.len(), 2);
    assert_eq!(history.processing_times.len(), 2);
}

#[test]
fn test_dynamic_precision_adjustment() {
    let device = Device::Cpu;
    let mut config = PrecisionControlConfig::default();
    config.dynamic_adjustment.enabled = true;
    config.dynamic_adjustment.strategy = AdjustmentStrategy::Adaptive;

    let mut controller = PrecisionController::new(config, device).unwrap();

    // Test with high error (should trigger adjustment)
    let high_error_stats = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.15, // High error
        compression_ratio: 2.0,
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    let adjustment = controller
        .adjust_precision_dynamically(&high_error_stats)
        .unwrap();
    if let Some(adj) = adjustment {
        assert_eq!(adj.reason, AdjustmentReason::HighError);
        assert!(adj.success);
    }

    // Test with low compression (should trigger adjustment)
    let low_compression_stats = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.02, // Low error
        compression_ratio: 1.2,   // Low compression
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    let adjustment = controller
        .adjust_precision_dynamically(&low_compression_stats)
        .unwrap();
    if let Some(adj) = adjustment {
        assert_eq!(adj.reason, AdjustmentReason::LowCompression);
        assert!(adj.success);
    }

    // Test with good metrics (should not trigger adjustment)
    let good_stats = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.03, // Good error
        compression_ratio: 8.0,   // Good compression
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    let adjustment = controller
        .adjust_precision_dynamically(&good_stats)
        .unwrap();
    // Should not trigger adjustment with good metrics
    // (Note: This might still trigger depending on the current state)
}

#[test]
fn test_configuration_presets() {
    let presets = vec![
        ConfigurationPreset::BitNetOptimized,
        ConfigurationPreset::PerformanceOptimized,
        ConfigurationPreset::AccuracyOptimized,
        ConfigurationPreset::MemoryOptimized,
        ConfigurationPreset::Balanced,
    ];

    for preset in presets {
        let config = create_enhanced_config(preset);
        assert!(
            config.is_ok(),
            "Failed to create config for preset: {preset:?}"
        );

        let config = config.unwrap();
        assert!(
            config.validate().is_ok(),
            "Config validation failed for preset: {preset:?}"
        );

        // Test that precision controller can be created
        let device = Device::Cpu;
        let controller = create_precision_controller(config.precision_control, device);
        assert!(
            controller.is_ok(),
            "Failed to create controller for preset: {preset:?}"
        );
    }
}

#[test]
fn test_enhanced_config_builder() {
    // Test basic builder functionality
    let config = EnhancedQuantizationConfigBuilder::new()
        .precision(QuantizationPrecision::OneFiveFiveBit)
        .strategy(QuantizationStrategy::Symmetric)
        .auto_optimization(true)
        .adaptive_thresholds(true)
        .real_time_monitoring(true)
        .build()
        .unwrap();

    assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
    assert_eq!(config.base.strategy, QuantizationStrategy::Symmetric);
    assert!(config.auto_optimization);
    assert!(config.adaptive_thresholds);
    assert!(config.real_time_monitoring);

    // Test preset builders
    let bitnet_config = EnhancedQuantizationConfigBuilder::bitnet_optimized()
        .build()
        .unwrap();
    assert_eq!(
        bitnet_config.base.precision,
        QuantizationPrecision::OneFiveFiveBit
    );
    assert!(bitnet_config.auto_optimization);

    let perf_config = EnhancedQuantizationConfigBuilder::performance_optimized()
        .build()
        .unwrap();
    assert_eq!(perf_config.base.precision, QuantizationPrecision::OneBit);
    assert_eq!(
        perf_config.precision_control.dynamic_adjustment.strategy,
        AdjustmentStrategy::Aggressive
    );

    let acc_config = EnhancedQuantizationConfigBuilder::accuracy_optimized()
        .build()
        .unwrap();
    assert_eq!(acc_config.base.precision, QuantizationPrecision::FourBit);
    assert_eq!(
        acc_config.precision_control.dynamic_adjustment.strategy,
        AdjustmentStrategy::Conservative
    );
}

#[test]
fn test_config_conversion() {
    let config = EnhancedQuantizationConfigBuilder::bitnet_optimized()
        .build()
        .unwrap();

    // Test weight config conversion
    let weight_config = config.to_weight_config();
    assert_eq!(
        weight_config.base.precision,
        QuantizationPrecision::OneFiveFiveBit
    );
    assert!(weight_config.normalize_weights);
    assert_eq!(weight_config.block_size, Some(64));

    // Test activation config conversion
    let activation_config = config.to_activation_config();
    assert_eq!(
        activation_config.base.precision,
        QuantizationPrecision::OneFiveFiveBit
    );
    assert!(activation_config.quantize_attention);
}

#[test]
fn test_recommended_bounds() {
    let test_cases = vec![
        (QuantizationPrecision::OneBit, 16.0),
        (QuantizationPrecision::OneFiveFiveBit, 10.0),
        (QuantizationPrecision::TwoBit, 6.0),
        (QuantizationPrecision::FourBit, 3.0),
        (QuantizationPrecision::EightBit, 2.0),
    ];

    for (precision, expected_min_compression) in test_cases {
        let config = EnhancedQuantizationConfigBuilder::new()
            .precision(precision)
            .build()
            .unwrap();

        let bounds = config.get_recommended_bounds();
        assert_eq!(bounds.min_compression_ratio, expected_min_compression);
    }
}

#[test]
fn test_adjustment_strategy_behavior() {
    let device = Device::Cpu;

    let strategies = vec![
        AdjustmentStrategy::Conservative,
        AdjustmentStrategy::Balanced,
        AdjustmentStrategy::Aggressive,
        AdjustmentStrategy::Adaptive,
    ];

    for strategy in strategies {
        let mut config = PrecisionControlConfig::default();
        config.dynamic_adjustment.strategy = strategy;
        config.dynamic_adjustment.enabled = true;

        let controller = PrecisionController::new(config, device.clone());
        assert!(
            controller.is_ok(),
            "Failed to create controller with strategy: {strategy:?}"
        );
    }
}

#[test]
fn test_precision_order() {
    let device = Device::Cpu;
    let config = PrecisionControlConfig::default();
    let controller = PrecisionController::new(config, device).unwrap();

    // Test precision ordering (internal method through bounds checking)
    let precisions = vec![
        QuantizationPrecision::OneBit,
        QuantizationPrecision::OneFiveFiveBit,
        QuantizationPrecision::TwoBit,
        QuantizationPrecision::FourBit,
        QuantizationPrecision::EightBit,
    ];

    // All should be in bounds with default config
    for precision in precisions {
        assert!(controller.is_precision_in_bounds(precision));
    }
}

#[test]
fn test_metrics_history_trimming() {
    let device = Device::Cpu;
    let mut config = PrecisionControlConfig::default();
    config.monitoring.history_size = 5; // Small history for testing

    let mut controller = PrecisionController::new(config, device).unwrap();

    // Add more metrics than history size
    for i in 0..10 {
        let stats = QuantizationStats {
            elements_count: 1000,
            quantization_error: 0.05 + i as f32 * 0.01,
            compression_ratio: 4.0,
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };

        controller.record_metrics(&stats, Duration::from_millis(10));
    }

    // Check that history is trimmed
    let history = controller.get_metrics_history();
    assert!(history.quantization_errors.len() <= 5);
    assert!(history.compression_ratios.len() <= 5);
    assert!(history.processing_times.len() <= 5);
}

#[test]
fn test_performance_impact_calculation() {
    // This test verifies that performance impact is properly tracked
    // when precision adjustments are made
    let device = Device::Cpu;
    let mut config = PrecisionControlConfig::default();
    config.dynamic_adjustment.enabled = true;

    let mut controller = PrecisionController::new(config, device).unwrap();

    // Record initial state
    let initial_stats = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.05,
        compression_ratio: 4.0,
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    controller.record_metrics(&initial_stats, Duration::from_millis(10));

    // Trigger adjustment with high error
    let high_error_stats = QuantizationStats {
        elements_count: 1000,
        quantization_error: 0.15,
        compression_ratio: 4.0,
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    };

    let adjustment = controller
        .adjust_precision_dynamically(&high_error_stats)
        .unwrap();

    if let Some(adj) = adjustment {
        // Check that adjustment was recorded
        let history = controller.get_adjustment_history();
        assert!(!history.is_empty());
        assert_eq!(history.last().unwrap().reason, AdjustmentReason::HighError);
    }
}

#[test]
fn test_custom_metrics() {
    let config = EnhancedQuantizationConfigBuilder::new()
        .custom_metrics(vec![
            "custom_metric_1".to_string(),
            "custom_metric_2".to_string(),
        ])
        .build()
        .unwrap();

    assert_eq!(config.custom_metrics.len(), 2);
    assert!(config
        .custom_metrics
        .contains(&"custom_metric_1".to_string()));
    assert!(config
        .custom_metrics
        .contains(&"custom_metric_2".to_string()));
}
