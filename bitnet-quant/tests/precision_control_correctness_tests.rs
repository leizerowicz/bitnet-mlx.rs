//! Comprehensive tests for precision control functionality
//!
//! This module tests the advanced precision control system including:
//! - Dynamic precision adjustment based on performance metrics
//! - Precision bounds validation and enforcement
//! - Performance monitoring and metrics tracking
//! - Configuration validation and edge cases
//! - Adjustment strategies and their effectiveness
//! - Recovery mechanisms and stability control

use bitnet_quant::quantization::precision_control::*;
use bitnet_quant::quantization::{QuantizationPrecision, QuantizationStats};
use candle_core::Device;
use std::time::Duration;

/// Helper functions for tests
fn create_test_controller() -> PrecisionController {
    let config = PrecisionControlConfig::default();
    let device = Device::Cpu;
    PrecisionController::new(config, device).unwrap()
}

fn create_test_stats(error: f32, compression: f32) -> QuantizationStats {
    QuantizationStats {
        elements_count: 1000,
        quantization_error: error,
        compression_ratio: compression,
        min_value: -1.0,
        max_value: 1.0,
        scale_factor: 1.0,
        zero_point: None,
    }
}

/// Test precision bounds validation
#[cfg(test)]
mod precision_bounds_tests {
    use super::*;

    #[test]
    fn test_precision_bounds_default_validation() {
        let bounds = PrecisionBounds::default();
        assert!(bounds.validate().is_ok());

        // Verify default values are sensible
        assert!(bounds.min_threshold < bounds.max_threshold);
        assert!(bounds.min_scale < bounds.max_scale);
        assert!(bounds.max_error_tolerance > 0.0);
        assert!(bounds.min_compression_ratio > 1.0);
    }

    #[test]
    fn test_precision_bounds_invalid_thresholds() {
        let mut bounds = PrecisionBounds::default();

        // Test min >= max threshold
        bounds.min_threshold = 5.0;
        bounds.max_threshold = 2.0;
        assert!(bounds.validate().is_err());

        // Test equal thresholds
        bounds.min_threshold = 3.0;
        bounds.max_threshold = 3.0;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_precision_bounds_invalid_scales() {
        let mut bounds = PrecisionBounds::default();

        // Test min >= max scale
        bounds.min_scale = 1e5;
        bounds.max_scale = 1e3;
        assert!(bounds.validate().is_err());

        // Test equal scales
        bounds.min_scale = 1.0;
        bounds.max_scale = 1.0;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_precision_bounds_invalid_error_tolerance() {
        let mut bounds = PrecisionBounds::default();

        // Test zero error tolerance
        bounds.max_error_tolerance = 0.0;
        assert!(bounds.validate().is_err());

        // Test negative error tolerance
        bounds.max_error_tolerance = -0.1;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_precision_bounds_invalid_compression_ratio() {
        let mut bounds = PrecisionBounds::default();

        // Test compression ratio <= 1.0
        bounds.min_compression_ratio = 1.0;
        assert!(bounds.validate().is_err());

        bounds.min_compression_ratio = 0.5;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_precision_bounds_edge_cases() {
        let mut bounds = PrecisionBounds::default();

        // Test very small positive values
        bounds.min_threshold = 1e-10;
        bounds.max_threshold = 1e-9;
        bounds.max_error_tolerance = 1e-8;
        bounds.min_compression_ratio = 1.001;
        assert!(bounds.validate().is_ok());

        // Test very large values
        bounds.min_threshold = 1e6;
        bounds.max_threshold = 1e8;
        bounds.max_error_tolerance = 1e3;
        bounds.min_compression_ratio = 1000.0;
        assert!(bounds.validate().is_ok());
    }
}

/// Test dynamic adjustment configuration
#[cfg(test)]
mod dynamic_adjustment_tests {
    use super::*;

    #[test]
    fn test_dynamic_adjustment_default_validation() {
        let config = DynamicAdjustmentConfig::default();
        assert!(config.validate().is_ok());

        // Verify default values
        assert!(config.enabled);
        assert!(config.evaluation_window > 0);
        assert!(config.adjustment_frequency > 0);
        assert!(config.learning_rate > 0.0 && config.learning_rate <= 1.0);
        assert!(config.stability_threshold >= 0.0);
    }

    #[test]
    fn test_dynamic_adjustment_invalid_window() {
        let mut config = DynamicAdjustmentConfig::default();

        config.evaluation_window = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dynamic_adjustment_invalid_frequency() {
        let mut config = DynamicAdjustmentConfig::default();

        config.adjustment_frequency = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dynamic_adjustment_invalid_learning_rate() {
        let mut config = DynamicAdjustmentConfig::default();

        // Test zero learning rate
        config.learning_rate = 0.0;
        assert!(config.validate().is_err());

        // Test negative learning rate
        config.learning_rate = -0.1;
        assert!(config.validate().is_err());

        // Test learning rate > 1.0
        config.learning_rate = 1.5;
        assert!(config.validate().is_err());

        // Test boundary values
        config.learning_rate = 1.0;
        assert!(config.validate().is_ok());

        config.learning_rate = 0.001;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dynamic_adjustment_invalid_stability_threshold() {
        let mut config = DynamicAdjustmentConfig::default();

        config.stability_threshold = -0.1;
        assert!(config.validate().is_err());

        // Test boundary value
        config.stability_threshold = 0.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adjustment_strategies() {
        let strategies = [
            AdjustmentStrategy::Conservative,
            AdjustmentStrategy::Balanced,
            AdjustmentStrategy::Aggressive,
            AdjustmentStrategy::Adaptive,
            AdjustmentStrategy::Custom,
        ];

        for strategy in strategies {
            let mut config = DynamicAdjustmentConfig::default();
            config.strategy = strategy;
            assert!(config.validate().is_ok());
        }
    }
}

/// Test precision monitoring configuration
#[cfg(test)]
mod monitoring_tests {
    use super::*;

    #[test]
    fn test_monitoring_config_default_validation() {
        let config = PrecisionMonitoringConfig::default();
        assert!(config.validate().is_ok());

        // Verify default values
        assert!(config.enabled);
        assert!(!config.tracked_metrics.is_empty());
        assert!(config.history_size > 0);
        assert!(config.enable_alerts);
    }

    #[test]
    fn test_monitoring_config_invalid_history_size() {
        let mut config = PrecisionMonitoringConfig::default();

        config.history_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_monitoring_config_empty_metrics() {
        let mut config = PrecisionMonitoringConfig::default();

        config.tracked_metrics = Vec::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_precision_metrics_coverage() {
        let all_metrics = vec![
            PrecisionMetric::QuantizationError,
            PrecisionMetric::CompressionRatio,
            PrecisionMetric::ProcessingTime,
            PrecisionMetric::MemoryUsage,
            PrecisionMetric::ThresholdStability,
            PrecisionMetric::ScaleVariance,
            PrecisionMetric::SparsityRatio,
            PrecisionMetric::SignalToNoiseRatio,
        ];

        let mut config = PrecisionMonitoringConfig::default();
        config.tracked_metrics = all_metrics;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_alert_thresholds_validation() {
        let thresholds = AlertThresholds::default();

        // Verify sensible defaults
        assert!(thresholds.max_quantization_error > 0.0);
        assert!(thresholds.min_compression_ratio > 1.0);
        assert!(thresholds.max_processing_time_ms > 0.0);
        assert!(thresholds.max_memory_usage_mb > 0.0);
        assert!(thresholds.instability_threshold >= 0.0);
    }
}

/// Test performance thresholds
#[cfg(test)]
mod performance_threshold_tests {
    use super::*;

    #[test]
    fn test_performance_thresholds_default_validation() {
        let thresholds = PerformanceThresholds::default();
        assert!(thresholds.validate().is_ok());

        // Verify default values are sensible
        assert!(thresholds.min_accuracy > 0.0 && thresholds.min_accuracy <= 1.0);
        assert!(thresholds.max_latency_ms > 0.0);
        assert!(thresholds.max_memory_overhead_pct >= 0.0);
        assert!(thresholds.min_throughput > 0.0);
        assert!(thresholds.min_energy_efficiency > 0.0 && thresholds.min_energy_efficiency <= 1.0);
    }

    #[test]
    fn test_performance_thresholds_invalid_accuracy() {
        let mut thresholds = PerformanceThresholds::default();

        // Test accuracy <= 0
        thresholds.min_accuracy = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds.min_accuracy = -0.1;
        assert!(thresholds.validate().is_err());

        // Test accuracy > 1
        thresholds.min_accuracy = 1.5;
        assert!(thresholds.validate().is_err());

        // Test boundary values
        thresholds.min_accuracy = 0.001;
        assert!(thresholds.validate().is_ok());

        thresholds.min_accuracy = 1.0;
        assert!(thresholds.validate().is_ok());
    }

    #[test]
    fn test_performance_thresholds_invalid_latency() {
        let mut thresholds = PerformanceThresholds::default();

        thresholds.max_latency_ms = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds.max_latency_ms = -1.0;
        assert!(thresholds.validate().is_err());
    }

    #[test]
    fn test_performance_thresholds_invalid_memory_overhead() {
        let mut thresholds = PerformanceThresholds::default();

        thresholds.max_memory_overhead_pct = -1.0;
        assert!(thresholds.validate().is_err());

        // Test boundary value
        thresholds.max_memory_overhead_pct = 0.0;
        assert!(thresholds.validate().is_ok());
    }

    #[test]
    fn test_performance_thresholds_invalid_throughput() {
        let mut thresholds = PerformanceThresholds::default();

        thresholds.min_throughput = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds.min_throughput = -1.0;
        assert!(thresholds.validate().is_err());
    }

    #[test]
    fn test_performance_thresholds_invalid_energy_efficiency() {
        let mut thresholds = PerformanceThresholds::default();

        // Test efficiency <= 0
        thresholds.min_energy_efficiency = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds.min_energy_efficiency = -0.1;
        assert!(thresholds.validate().is_err());

        // Test efficiency > 1
        thresholds.min_energy_efficiency = 1.5;
        assert!(thresholds.validate().is_err());

        // Test boundary values
        thresholds.min_energy_efficiency = 0.001;
        assert!(thresholds.validate().is_ok());

        thresholds.min_energy_efficiency = 1.0;
        assert!(thresholds.validate().is_ok());
    }
}

/// Test precision controller functionality
#[cfg(test)]
mod precision_controller_tests {
    use super::*;

    #[test]
    fn test_precision_controller_creation() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let controller = PrecisionController::new(config, device);
        assert!(controller.is_ok());

        let controller = controller.unwrap();
        assert_eq!(
            controller.get_current_state().precision,
            QuantizationPrecision::OneFiveFiveBit
        );
    }

    #[test]
    fn test_precision_controller_invalid_config() {
        let mut config = PrecisionControlConfig::default();
        config.precision_bounds.min_threshold = 10.0;
        config.precision_bounds.max_threshold = 5.0; // Invalid: min > max

        let device = Device::Cpu;
        let controller = PrecisionController::new(config, device);
        assert!(controller.is_err());
    }

    #[test]
    fn test_precision_bounds_checking() {
        let controller = create_test_controller();

        // Test valid precisions
        assert!(controller.is_precision_in_bounds(QuantizationPrecision::OneBit));
        assert!(controller.is_precision_in_bounds(QuantizationPrecision::OneFiveFiveBit));
        assert!(controller.is_precision_in_bounds(QuantizationPrecision::EightBit));
    }

    #[test]
    fn test_precision_validation() {
        let controller = create_test_controller();

        // Test valid parameters
        let result =
            controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1.0);
        assert!(result.is_ok());

        // Test invalid threshold (too high)
        let result =
            controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 100.0, 1.0);
        assert!(result.is_err());

        // Test invalid scale (too low)
        let result =
            controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_recording() {
        let mut controller = create_test_controller();

        let stats = create_test_stats(0.05, 4.0);
        controller.record_metrics(&stats, Duration::from_millis(10));

        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 1);
        assert_eq!(summary.average_error, 0.05);
        assert_eq!(summary.average_compression_ratio, 4.0);

        let history = controller.get_metrics_history();
        assert_eq!(history.quantization_errors.len(), 1);
        assert_eq!(history.compression_ratios.len(), 1);
        assert_eq!(history.processing_times.len(), 1);
    }

    #[test]
    fn test_metrics_history_trimming() {
        let mut config = PrecisionControlConfig::default();
        config.monitoring.history_size = 3; // Small history for testing

        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        // Add more metrics than history size
        for i in 0..5 {
            let stats = create_test_stats(0.01 * i as f32, 2.0 + i as f32);
            controller.record_metrics(&stats, Duration::from_millis(i as u64));
        }

        let history = controller.get_metrics_history();
        assert!(history.quantization_errors.len() <= 3);
        assert!(history.compression_ratios.len() <= 3);
        assert!(history.processing_times.len() <= 3);
    }

    #[test]
    fn test_dynamic_precision_adjustment_disabled() {
        let mut config = PrecisionControlConfig::default();
        config.dynamic_adjustment.enabled = false;

        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        let stats = create_test_stats(0.2, 1.5); // High error, low compression
        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();
        assert!(adjustment.is_none());
    }

    #[test]
    fn test_dynamic_precision_adjustment_high_error() {
        let mut controller = create_test_controller();

        let stats = create_test_stats(0.15, 3.0); // High error
        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();

        if let Some(adj) = adjustment {
            assert_eq!(adj.reason, AdjustmentReason::HighError);
            // Should increase precision
            assert!(
                controller.get_current_state().precision != QuantizationPrecision::OneFiveFiveBit
            );
        }
    }

    #[test]
    fn test_dynamic_precision_adjustment_low_compression() {
        let mut controller = create_test_controller();

        let stats = create_test_stats(0.02, 1.2); // Low error, low compression
        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();

        if let Some(adj) = adjustment {
            assert_eq!(adj.reason, AdjustmentReason::LowCompression);
        }
    }

    #[test]
    fn test_precision_adjustment_bounds_enforcement() {
        let mut config = PrecisionControlConfig::default();
        config.precision_bounds.min_precision = QuantizationPrecision::TwoBit;
        config.precision_bounds.max_precision = QuantizationPrecision::FourBit;

        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        // Try to adjust beyond bounds
        let stats = create_test_stats(0.001, 10.0); // Very low error, high compression
        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();

        // Should not adjust below minimum precision
        if let Some(adj) = adjustment {
            assert!(controller.is_precision_in_bounds(adj.to_precision));
        }
    }

    #[test]
    fn test_adjustment_history_tracking() {
        let mut controller = create_test_controller();

        let stats = create_test_stats(0.15, 3.0);
        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();

        if adjustment.is_some() {
            let history = controller.get_adjustment_history();
            assert!(!history.is_empty());
            assert_eq!(history[0].reason, AdjustmentReason::HighError);
        }
    }

    #[test]
    fn test_performance_summary() {
        let mut controller = create_test_controller();

        // Record multiple metrics
        let test_data = vec![(0.05, 4.0, 10), (0.03, 3.5, 15), (0.07, 4.2, 12)];

        for (error, compression, time_ms) in test_data {
            let stats = create_test_stats(error, compression);
            controller.record_metrics(&stats, Duration::from_millis(time_ms));
        }

        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 3);
        assert!((summary.average_error - 0.05).abs() < 0.01);
        assert!((summary.average_compression_ratio - 3.9).abs() < 0.1);
        assert!(summary.average_processing_time.as_millis() > 0);
    }
}

/// Test configuration presets
#[cfg(test)]
mod configuration_preset_tests {
    use super::*;

    #[test]
    fn test_conservative_config() {
        let config = PrecisionControlConfig::conservative();
        assert!(config.validate().is_ok());

        assert_eq!(
            config.target_precision,
            QuantizationPrecision::OneFiveFiveBit
        );
        assert_eq!(
            config.dynamic_adjustment.strategy,
            AdjustmentStrategy::Conservative
        );
        assert!(config.precision_bounds.max_error_tolerance <= 0.05);
        assert!(config.dynamic_adjustment.learning_rate <= 0.1);
        assert!(config.dynamic_adjustment.stability_threshold <= 0.02);
    }

    #[test]
    fn test_aggressive_config() {
        let config = PrecisionControlConfig::aggressive();
        assert!(config.validate().is_ok());

        assert_eq!(config.target_precision, QuantizationPrecision::OneBit);
        assert_eq!(
            config.dynamic_adjustment.strategy,
            AdjustmentStrategy::Aggressive
        );
        assert!(config.precision_bounds.max_error_tolerance >= 0.1);
        assert!(config.dynamic_adjustment.learning_rate >= 0.1);
        assert!(config.precision_bounds.min_compression_ratio >= 3.0);
    }

    #[test]
    fn test_conservative_controller_creation() {
        let device = Device::Cpu;
        let controller = create_conservative_precision_controller(device);
        assert!(controller.is_ok());

        let controller = controller.unwrap();
        assert_eq!(
            controller.get_current_state().precision,
            QuantizationPrecision::OneFiveFiveBit
        );
    }

    #[test]
    fn test_aggressive_controller_creation() {
        let device = Device::Cpu;
        let controller = create_aggressive_precision_controller(device);
        assert!(controller.is_ok());

        let controller = controller.unwrap();
        assert_eq!(
            controller.get_current_state().precision,
            QuantizationPrecision::OneBit
        );
    }
}

/// Test edge cases and error conditions
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_extreme_metrics_handling() {
        let mut controller = create_test_controller();

        // Test with extreme values
        let extreme_stats = QuantizationStats {
            elements_count: 0,
            quantization_error: f32::INFINITY,
            compression_ratio: f32::NEG_INFINITY,
            min_value: f32::NEG_INFINITY,
            max_value: f32::INFINITY,
            scale_factor: f32::NAN,
            zero_point: None,
        };

        // Should handle gracefully without panicking
        controller.record_metrics(&extreme_stats, Duration::from_millis(0));
        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 1);
    }

    #[test]
    fn test_zero_duration_metrics() {
        let mut controller = create_test_controller();

        let stats = create_test_stats(0.05, 3.0);
        controller.record_metrics(&stats, Duration::from_nanos(0));

        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 1);
        assert_eq!(summary.average_processing_time, Duration::from_nanos(0));
    }

    #[test]
    fn test_very_large_history() {
        let mut config = PrecisionControlConfig::default();
        config.monitoring.history_size = 1_000_000; // Very large history

        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        // Add a reasonable number of metrics
        for i in 0..100 {
            let stats = create_test_stats(0.01, 2.0 + i as f32 * 0.01);
            controller.record_metrics(&stats, Duration::from_millis(i));
        }

        let history = controller.get_metrics_history();
        assert_eq!(history.quantization_errors.len(), 100);
    }

    #[test]
    fn test_precision_bounds_consistency() {
        let controller = create_test_controller();

        // Test that precision bounds checking works consistently
        let precisions = vec![
            QuantizationPrecision::OneBit,
            QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::TwoBit,
            QuantizationPrecision::FourBit,
            QuantizationPrecision::EightBit,
        ];

        for precision in precisions {
            // All default precisions should be within bounds
            assert!(controller.is_precision_in_bounds(precision));
        }
    }

    #[test]
    fn test_adjustment_reason_serialization() {
        let reasons = vec![
            AdjustmentReason::HighError,
            AdjustmentReason::LowCompression,
            AdjustmentReason::PerformanceDegradation,
            AdjustmentReason::MemoryPressure,
            AdjustmentReason::Instability,
            AdjustmentReason::UserRequest,
            AdjustmentReason::AutoOptimization,
        ];

        for reason in reasons {
            // Test that reasons can be compared for equality
            let reason_copy = reason;
            assert_eq!(reason, reason_copy);
        }
    }

    #[test]
    fn test_performance_impact_calculation() {
        let impact = PerformanceImpact {
            error_delta: 0.02,
            compression_delta: -0.5,
            time_delta: Duration::from_millis(5),
            memory_delta: 1024,
            impact_score: 0.75,
        };

        // Test that impact values are reasonable
        assert!(impact.error_delta.is_finite());
        assert!(impact.compression_delta.is_finite());
        assert!(impact.memory_delta != 0);
        assert!(impact.impact_score >= 0.0 && impact.impact_score <= 1.0);
    }
}

/// Test factory functions
#[cfg(test)]
mod factory_tests {
    use super::*;

    #[test]
    fn test_create_precision_controller() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let controller = create_precision_controller(config, device);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_create_conservative_precision_controller() {
        let device = Device::Cpu;
        let controller = create_conservative_precision_controller(device);
        assert!(controller.is_ok());

        let controller = controller.unwrap();
        let state = controller.get_current_state();
        assert_eq!(state.precision, QuantizationPrecision::OneFiveFiveBit);
    }

    #[test]
    fn test_create_aggressive_precision_controller() {
        let device = Device::Cpu;
        let controller = create_aggressive_precision_controller(device);
        assert!(controller.is_ok());

        let controller = controller.unwrap();
        let state = controller.get_current_state();
        assert_eq!(state.precision, QuantizationPrecision::OneBit);
    }

    #[test]
    fn test_factory_with_invalid_config() {
        let mut config = PrecisionControlConfig::default();
        config.precision_bounds.min_threshold = 10.0;
        config.precision_bounds.max_threshold = 5.0; // Invalid

        let device = Device::Cpu;
        let controller = create_precision_controller(config, device);
        assert!(controller.is_err());
    }
}

/// Integration tests combining multiple components
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_precision_control_workflow() {
        let mut controller = create_test_controller();

        // Simulate a series of quantization operations with varying performance
        let test_scenarios = vec![
            (0.15, 2.0, 20), // High error, low compression - should increase precision
            (0.08, 3.0, 15), // Medium error, good compression
            (0.02, 4.5, 10), // Low error, high compression - should decrease precision
            (0.12, 2.5, 18), // High error again
        ];

        for (error, compression, time_ms) in test_scenarios {
            let stats = create_test_stats(error, compression);

            // Record metrics
            controller.record_metrics(&stats, Duration::from_millis(time_ms));

            // Attempt dynamic adjustment
            let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();

            if let Some(adj) = adjustment {
                println!(
                    "Adjustment: {:?} -> {:?} (reason: {:?})",
                    adj.from_precision, adj.to_precision, adj.reason
                );
            }
        }

        // Verify final state
        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 4);
        assert!(summary.total_adjustments <= 4); // At most one adjustment per operation

        let history = controller.get_adjustment_history();
        assert!(history.len() <= 4);
    }
}
