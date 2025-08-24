//! Precision Control Demo
//!
//! This example demonstrates the advanced quantization precision control features
//! including dynamic precision adjustment, monitoring, and validation.

use bitnet_quant::prelude::*;
use bitnet_quant::{
    create_enhanced_config, create_precision_controller, ConfigurationPreset,
    EnhancedQuantizationConfigBuilder, PrecisionControlConfig, QuantizationStats,
};
use candle_core::{Device, Tensor};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 BitNet Quantization Precision Control Demo");
    println!("==============================================\n");

    let device = Device::Cpu;

    // Demo 1: Basic Precision Control
    demo_basic_precision_control(&device)?;

    // Demo 2: Dynamic Precision Adjustment
    demo_dynamic_precision_adjustment(&device)?;

    // Demo 3: Configuration Presets
    demo_configuration_presets(&device)?;

    // Demo 4: Custom Precision Monitoring
    demo_precision_monitoring(&device)?;

    // Demo 5: Precision Validation
    demo_precision_validation(&device)?;

    println!("✅ All precision control demos completed successfully!");
    Ok(())
}

fn demo_basic_precision_control(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Basic Precision Control");
    println!("----------------------------------");

    // Create a precision controller with default configuration
    let config = PrecisionControlConfig::default();
    let controller = create_precision_controller(config, device.clone())?;

    // Create test weights
    let weights = Tensor::randn(0.0, 1.0, (128, 256), device)?;
    println!("Created test weights: {:?}", weights.shape());

    // Validate precision bounds
    let validation_result = controller.validate_precision_bounds(
        QuantizationPrecision::OneFiveFiveBit,
        0.7, // threshold
        1.0, // scale
    );

    match validation_result {
        Ok(()) => println!("✅ Precision bounds validation passed"),
        Err(e) => println!("❌ Precision bounds validation failed: {e}"),
    }

    // Get current precision state
    let state = controller.get_current_state();
    println!("Current precision: {:?}", state.precision);
    println!("Stability score: {:.3}", state.stability_score);
    println!("Performance score: {:.3}", state.performance_score);

    println!();
    Ok(())
}

fn demo_dynamic_precision_adjustment(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Demo 2: Dynamic Precision Adjustment");
    println!("---------------------------------------");

    // Create configuration with dynamic adjustment enabled
    let config = EnhancedQuantizationConfigBuilder::new()
        .precision(QuantizationPrecision::OneFiveFiveBit)
        .auto_optimization(true)
        .adaptive_thresholds(true)
        .build()?;

    let mut controller = create_precision_controller(config.precision_control, device.clone())?;

    // Simulate quantization operations with varying error levels
    let test_scenarios = vec![
        ("Low error scenario", 0.02),
        ("Medium error scenario", 0.08),
        ("High error scenario", 0.15),
        ("Very high error scenario", 0.25),
    ];

    for (scenario_name, error_level) in test_scenarios {
        println!("\n🧪 Testing: {scenario_name}");

        // Create mock quantization stats
        let stats = QuantizationStats {
            elements_count: 32768,
            quantization_error: error_level,
            compression_ratio: 20.0,
            min_value: -2.0,
            max_value: 2.0,
            scale_factor: 1.0,
            zero_point: None,
        };

        // Record metrics
        controller.record_metrics(&stats, Duration::from_millis(5));

        // Attempt dynamic adjustment
        let adjustment = controller.adjust_precision_dynamically(&stats)?;

        match adjustment {
            Some(adj) => {
                println!(
                    "  📈 Precision adjusted: {:?} -> {:?}",
                    adj.from_precision, adj.to_precision
                );
                println!("  📋 Reason: {:?}", adj.reason);
                println!("  ✅ Success: {}", adj.success);
            }
            None => {
                println!("  ➡️  No adjustment needed");
            }
        }
    }

    // Show performance summary
    let summary = controller.get_performance_summary();
    println!("\n📊 Performance Summary:");
    println!("  Operations: {}", summary.operations_count);
    println!("  Average error: {:.4}", summary.average_error);
    println!(
        "  Average compression: {:.2}x",
        summary.average_compression_ratio
    );
    println!("  Total adjustments: {}", summary.total_adjustments);
    println!("  Current precision: {:?}", summary.current_precision);

    println!();
    Ok(())
}

fn demo_configuration_presets(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Demo 3: Configuration Presets");
    println!("--------------------------------");

    let presets = vec![
        ("BitNet Optimized", ConfigurationPreset::BitNetOptimized),
        (
            "Performance Optimized",
            ConfigurationPreset::PerformanceOptimized,
        ),
        ("Accuracy Optimized", ConfigurationPreset::AccuracyOptimized),
        ("Memory Optimized", ConfigurationPreset::MemoryOptimized),
        ("Balanced", ConfigurationPreset::Balanced),
    ];

    for (name, preset) in presets {
        println!("\n🎛️  Testing preset: {name}");

        let config = create_enhanced_config(preset)?;

        println!("  Target precision: {:?}", config.base.precision);
        println!("  Strategy: {:?}", config.base.strategy);
        println!("  Auto optimization: {}", config.auto_optimization);
        println!("  Adaptive thresholds: {}", config.adaptive_thresholds);
        println!("  Real-time monitoring: {}", config.real_time_monitoring);

        // Show precision bounds
        let bounds = &config.precision_control.precision_bounds;
        let controller =
            create_precision_controller(config.precision_control.clone(), device.clone())?;
        println!(
            "  Precision bounds: {:?} - {:?}",
            bounds.min_precision, bounds.max_precision
        );
        println!("  Max error tolerance: {:.3}", bounds.max_error_tolerance);
        println!(
            "  Min compression ratio: {:.1}x",
            bounds.min_compression_ratio
        );
    }

    println!();
    Ok(())
}

fn demo_precision_monitoring(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 4: Custom Precision Monitoring");
    println!("--------------------------------------");

    // Create configuration with comprehensive monitoring
    let config = EnhancedQuantizationConfigBuilder::new()
        .precision(QuantizationPrecision::OneFiveFiveBit)
        .real_time_monitoring(true)
        .custom_metrics(vec![
            "custom_accuracy".to_string(),
            "custom_latency".to_string(),
            "custom_throughput".to_string(),
        ])
        .build()?;

    let mut controller = create_precision_controller(config.precision_control, device.clone())?;

    // Simulate a series of quantization operations
    println!("🔄 Simulating quantization operations...");

    for i in 0..10 {
        let start_time = Instant::now();

        // Simulate varying performance
        let error = 0.05 + (i as f32 * 0.01);
        let compression = 15.0 + (i as f32 * 0.5);

        let stats = QuantizationStats {
            elements_count: 1024 * (i + 1),
            quantization_error: error,
            compression_ratio: compression,
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };

        let processing_time = Duration::from_millis(5 + i as u64);
        controller.record_metrics(&stats, processing_time);

        println!(
            "  Operation {}: error={:.3}, compression={:.1}x, time={:?}",
            i + 1,
            error,
            compression,
            processing_time
        );
    }

    // Show metrics history
    let history = controller.get_metrics_history();
    println!("\n📊 Metrics History:");
    println!(
        "  Quantization errors recorded: {}",
        history.quantization_errors.len()
    );
    println!(
        "  Compression ratios recorded: {}",
        history.compression_ratios.len()
    );
    println!(
        "  Processing times recorded: {}",
        history.processing_times.len()
    );

    // Show final summary
    let summary = controller.get_performance_summary();
    println!("\n📋 Final Summary:");
    println!("  Total operations: {}", summary.operations_count);
    println!("  Average error: {:.4}", summary.average_error);
    println!(
        "  Average compression: {:.2}x",
        summary.average_compression_ratio
    );
    println!(
        "  Average processing time: {:?}",
        summary.average_processing_time
    );

    println!();
    Ok(())
}

fn demo_precision_validation(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Demo 5: Precision Validation");
    println!("-------------------------------");

    let config = PrecisionControlConfig::default();
    let controller = create_precision_controller(config, device.clone())?;

    // Test various validation scenarios
    let validation_tests = vec![
        (
            "Valid precision bounds",
            QuantizationPrecision::OneFiveFiveBit,
            0.7,
            1.0,
            true,
        ),
        (
            "Invalid threshold (too high)",
            QuantizationPrecision::OneFiveFiveBit,
            15.0,
            1.0,
            false,
        ),
        (
            "Invalid threshold (too low)",
            QuantizationPrecision::OneFiveFiveBit,
            -0.1,
            1.0,
            false,
        ),
        (
            "Invalid scale (too high)",
            QuantizationPrecision::OneFiveFiveBit,
            0.7,
            1e10,
            false,
        ),
        (
            "Invalid scale (too low)",
            QuantizationPrecision::OneFiveFiveBit,
            0.7,
            1e-10,
            false,
        ),
    ];

    for (test_name, precision, threshold, scale, should_pass) in validation_tests {
        println!("\n🧪 Testing: {test_name}");

        let result = controller.validate_precision_bounds(precision, threshold, scale);

        match (result.is_ok(), should_pass) {
            (true, true) => println!("  ✅ Validation passed as expected"),
            (false, false) => println!(
                "  ✅ Validation failed as expected: {}",
                result.unwrap_err()
            ),
            (true, false) => println!("  ❌ Validation should have failed but passed"),
            (false, true) => println!(
                "  ❌ Validation should have passed but failed: {}",
                result.unwrap_err()
            ),
        }
    }

    // Test precision bounds checking
    println!("\n🔍 Testing precision bounds checking:");

    let precision_tests = vec![
        (QuantizationPrecision::OneBit, true),
        (QuantizationPrecision::OneFiveFiveBit, true),
        (QuantizationPrecision::TwoBit, true),
        (QuantizationPrecision::FourBit, true),
        (QuantizationPrecision::EightBit, true),
    ];

    for (precision, should_be_valid) in precision_tests {
        let is_valid = controller.is_precision_in_bounds(precision);
        println!(
            "  {:?}: {} (expected: {})",
            precision,
            if is_valid { "✅ Valid" } else { "❌ Invalid" },
            if should_be_valid { "Valid" } else { "Invalid" }
        );
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_control_demo() {
        let device = Device::Cpu;

        // Test that basic precision control works
        assert!(demo_basic_precision_control(&device).is_ok());

        // Test that configuration presets work
        assert!(demo_configuration_presets(&device).is_ok());

        // Test that validation works
        assert!(demo_precision_validation(&device).is_ok());
    }

    #[test]
    fn test_all_presets() {
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
                "Failed to create config for preset: {:?}",
                preset
            );

            let config = config.unwrap();
            assert!(
                config.validate().is_ok(),
                "Config validation failed for preset: {:?}",
                preset
            );
        }
    }
}
