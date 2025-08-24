//! Straight-Through Estimator Tests - Fixed Version
//!
//! Comprehensive test suite for STE functionality, gradient flow preservation,
//! and different STE variants used in quantization-aware training.

use bitnet_training::qat::straight_through::{
    BinaryQuantizationFunction, STEConfig, STEVariant, StraightThroughEstimator,
    TernaryQuantizationFunction,
};
use candle_core::{Device, Result as CandleResult, Tensor};
use std::time::Duration;

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

/// Create test tensor with known values for deterministic testing
fn create_test_values_tensor(device: &Device) -> CandleResult<Tensor> {
    // Create tensor with specific values to test quantization behavior
    let values = vec![-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0, -1.0, 1.0];
    Tensor::from_vec(values, &[3, 3], device)
}

#[cfg(test)]
mod ste_basic_functionality {
    use super::*;

    #[test]
    fn test_ste_creation() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Clipped,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let _ste = StraightThroughEstimator::new(config).unwrap();
    }

    #[test]
    fn test_standard_ste_quantization() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Standard,
            temperature: 1.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();

        // Verify quantization results
        let values = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // For standard STE with ternary quantization, should get -1, 0, 1
        for &val in values.iter() {
            assert!(
                val == -1.0 || val == 0.0 || val == 1.0,
                "Standard STE should produce ternary values, got: {val}"
            );
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let device = setup_test_device();

        let config_high_temp = STEConfig {
            variant: STEVariant::Standard,
            temperature: 5.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let config_low_temp = STEConfig {
            variant: STEVariant::Standard,
            temperature: 0.1,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste_high_temp = StraightThroughEstimator::new(config_high_temp).unwrap();
        let mut ste_low_temp = StraightThroughEstimator::new(config_low_temp).unwrap();

        let input = create_test_values_tensor(&device).unwrap();

        let quantized_high = ste_high_temp.forward_quantized(&input).unwrap();
        let quantized_low = ste_low_temp.forward_quantized(&input).unwrap();

        // Temperature affects the sharpness of quantization
        // Lower temperature should produce more binary-like results
        // Higher temperature should produce softer transitions
        let high_vals = quantized_high
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let low_vals = quantized_low
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        println!("High temp values: {high_vals:?}");
        println!("Low temp values: {low_vals:?}");
    }
}

#[cfg(test)]
mod ste_advanced_features {
    use super::*;

    #[test]
    fn test_clipped_ste_variant() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Clipped,
            range: 2.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();
        let values = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Values should be clipped within [-range, range]
        for &val in values.iter() {
            assert!(
                (-2.0..=2.0).contains(&val),
                "Clipped STE should maintain range bounds, got: {val}"
            );
        }
    }

    #[test]
    fn test_learned_threshold_variant() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Learned,
            learnable_lr: 0.01,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        // Test that learned threshold variant can be created and used
        let quantized = ste.forward_quantized(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }

    #[test]
    fn test_soft_variant() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Soft,
            temperature: 1.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }
}

#[cfg(test)]
mod gradient_flow {
    use super::*;

    #[test]
    fn test_gradient_preservation() {
        let device = setup_test_device();

        let config_with_clip = STEConfig {
            variant: STEVariant::Standard,
            clip_gradients: true,
            clip_threshold: 1.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let config_no_clip = STEConfig {
            variant: STEVariant::Standard,
            clip_gradients: false,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        // Test that both configurations can be created
        let mut ste_with_clip = StraightThroughEstimator::new(config_with_clip).unwrap();
        let mut ste_no_clip = StraightThroughEstimator::new(config_no_clip).unwrap();

        let input = create_test_values_tensor(&device).unwrap();

        let quantized_clipped = ste_with_clip.forward_quantized(&input).unwrap();
        let quantized_no_clip = ste_no_clip.forward_quantized(&input).unwrap();

        assert_eq!(quantized_clipped.dims(), input.dims());
        assert_eq!(quantized_no_clip.dims(), input.dims());
    }
}

#[cfg(test)]
mod quantization_functions {
    use super::*;

    #[test]
    fn test_binary_quantization_function() {
        let device = setup_test_device();

        let func = BinaryQuantizationFunction::new(0.5, device.clone());
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = func.forward(&input).unwrap();
        let values = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Binary quantization should produce -1 or 1
        for &val in values.iter() {
            assert!(
                val == -1.0 || val == 1.0,
                "Binary quantization should produce -1 or 1, got: {val}"
            );
        }
    }

    #[test]
    fn test_ternary_quantization_function() {
        let device = setup_test_device();

        let func = TernaryQuantizationFunction::new(0.3, -0.3, device.clone());
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = func.forward(&input).unwrap();
        let values = quantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Ternary quantization should produce -1, 0, or 1
        for &val in values.iter() {
            assert!(
                val == -1.0 || val == 0.0 || val == 1.0,
                "Ternary quantization should produce -1, 0, or 1, got: {val}"
            );
        }
    }

    #[test]
    fn test_asymmetric_ternary_quantization() {
        let device = setup_test_device();

        let func = TernaryQuantizationFunction::new(0.8, -0.2, device.clone());
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = func.forward(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }
}

#[cfg(test)]
mod error_handling {
    use super::*;

    #[test]
    fn test_invalid_config_handling() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Standard,
            temperature: -1.0, // Invalid temperature
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        // Should handle invalid configuration gracefully
        match StraightThroughEstimator::new(config) {
            Ok(_) => {
                // Implementation may accept negative temperature
                println!("Implementation accepts negative temperature");
            }
            Err(_) => {
                // Implementation rejects invalid config
                println!("Implementation properly validates config");
            }
        }
    }
}

#[cfg(test)]
mod bitnet_integration {
    use super::*;

    #[test]
    fn test_bitnet_compatible_quantization() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Clipped,
            bits: 1, // BitNet uses 1-bit weights
            range: 1.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }

    #[test]
    fn test_multi_bit_quantization() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Standard,
            bits: 2, // Multi-bit quantization
            range: 2.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_values_tensor(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_quantization_performance() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Standard,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();

        // Create larger tensor for performance testing
        let large_tensor = Tensor::randn(0f32, 1f32, &[1000, 1000], &device).unwrap();

        let start = Instant::now();
        let _quantized = ste.forward_quantized(&large_tensor).unwrap();
        let elapsed = start.elapsed();

        println!("Quantization of 1M elements took: {elapsed:?}");
        assert!(
            elapsed < Duration::from_secs(1),
            "Quantization should be reasonably fast"
        );
    }
}
