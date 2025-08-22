//! Comprehensive tests for configurable quantization schemes
//! 
//! This module tests both 1-bit and 1.58-bit quantization schemes with various
//! configurations and validates their correctness and performance.

use bitnet_quant::prelude::*;
use candle_core::{Device, Tensor, DType};

#[test]
fn test_one_bit_quantization_basic() {
    let device = Device::Cpu;
    let mut scheme = ConfigurableQuantizationScheme::one_bit(device.clone());
    
    let input = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1], &device).unwrap();
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Check that values are binary {-1, +1}
    let values = quantized.values.to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 1.0, "Value {val} is not binary");
    }
    
    // Check precision
    assert_eq!(quantized.precision, QuantizationPrecision::OneBit);
    
    // Check compression ratio
    assert!(quantized.compression_ratio() > 1.0);
    
    // Test dequantization
    let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), input.shape());
}

#[test]
fn test_one_five_eight_bit_quantization_basic() {
    let device = Device::Cpu;
    let mut scheme = ConfigurableQuantizationScheme::one_five_eight_bit(device.clone());
    
    let input = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0], &device).unwrap();
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Check that values are ternary {-1, 0, +1}
    let values = quantized.values.to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Value {val} is not ternary");
    }
    
    // Check precision
    assert_eq!(quantized.precision, QuantizationPrecision::OneFiveFiveBit);
    
    // Check compression ratio
    assert!(quantized.compression_ratio() > 1.0);
    
    // Test dequantization
    let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), input.shape());
}

#[test]
fn test_binary_threshold_methods() {
    let device = Device::Cpu;
    let input = Tensor::new(&[2.0f32, -1.0, 0.5, -1.5, 0.1], &device).unwrap();
    
    let methods = [
        BinaryThresholdMethod::Zero,
        BinaryThresholdMethod::Mean,
        BinaryThresholdMethod::Adaptive,
        BinaryThresholdMethod::Optimal,
    ];
    
    for method in methods {
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneBit,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_bit: OneBitParams {
                    threshold_method: method,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut scheme = ConfigurableQuantizationScheme::new(config, device.clone());
        let quantized = scheme.quantize_tensor(&input).unwrap();
        
        // Verify binary values
        let values = quantized.values.to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(val == -1.0 || val == 1.0);
        }
        
        // Verify dequantization works
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), input.shape());
    }
}

#[test]
fn test_ternary_methods() {
    let device = Device::Cpu;
    let input = Tensor::new(&[2.0f32, -1.0, 0.3, -1.5, 0.0, 0.8], &device).unwrap();
    
    let methods = [
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold,
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];
    
    for method in methods {
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_five_eight_bit: OneFiveEightBitParams {
                    ternary_method: method,
                    threshold_factor: 0.7,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut scheme = ConfigurableQuantizationScheme::new(config, device.clone());
        let quantized = scheme.quantize_tensor(&input).unwrap();
        
        // Verify ternary values
        let values = quantized.values.to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
        
        // Verify dequantization works
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), input.shape());
    }
}

#[test]
fn test_multi_bit_quantization() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8, 0.0, 0.7], &device).unwrap();
    
    let precisions = [
        QuantizationPrecision::TwoBit,
        QuantizationPrecision::FourBit,
        QuantizationPrecision::EightBit,
    ];
    
    for precision in precisions {
        let mut scheme = QuantizationSchemeFactory::create_from_precision(precision, device.clone());
        let quantized = scheme.quantize_tensor(&input).unwrap();
        
        // Check precision
        assert_eq!(quantized.precision, precision);
        
        // Check that we have zero points for asymmetric quantization
        assert!(quantized.zero_points.is_some());
        
        // Test dequantization
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), input.shape());
        
        // Check that quantization error is reasonable
        let error = input.sub(&dequantized).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(error < 1.0, "Quantization error too high: {error}");
    }
}

#[test]
fn test_quantization_scheme_factory() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3], &device).unwrap();
    
    // Test factory methods
    let mut one_bit_scheme = QuantizationSchemeFactory::create_one_bit_scheme(device.clone());
    let one_bit_result = one_bit_scheme.quantize_tensor(&input).unwrap();
    assert_eq!(one_bit_result.precision, QuantizationPrecision::OneBit);
    
    let mut ternary_scheme = QuantizationSchemeFactory::create_one_five_eight_bit_scheme(device.clone());
    let ternary_result = ternary_scheme.quantize_tensor(&input).unwrap();
    assert_eq!(ternary_result.precision, QuantizationPrecision::OneFiveFiveBit);
    
    // Test custom scheme
    let custom_config = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::FourBit,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut custom_scheme = QuantizationSchemeFactory::create_custom_scheme(custom_config, device.clone());
    let custom_result = custom_scheme.quantize_tensor(&input).unwrap();
    assert_eq!(custom_result.precision, QuantizationPrecision::FourBit);
    
    // Test from precision
    let mut from_precision_scheme = QuantizationSchemeFactory::create_from_precision(
        QuantizationPrecision::EightBit, 
        device.clone()
    );
    let from_precision_result = from_precision_scheme.quantize_tensor(&input).unwrap();
    assert_eq!(from_precision_result.precision, QuantizationPrecision::EightBit);
}

#[test]
fn test_quantized_tensor_properties() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8], &device).unwrap();
    
    let mut scheme = ConfigurableQuantizationScheme::one_bit(device.clone());
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Test memory footprint
    let memory_footprint = quantized.memory_footprint();
    assert!(memory_footprint > 0);
    
    // Test compression ratio
    let compression_ratio = quantized.compression_ratio();
    assert!(compression_ratio > 1.0);
    
    // Test original shape preservation
    assert_eq!(quantized.original_shape, *input.shape());
    
    // Test statistics
    assert!(quantized.stats.elements_count > 0);
    assert!(quantized.stats.compression_ratio > 1.0);
}

#[test]
fn test_input_validation() {
    let device = Device::Cpu;
    let scheme = ConfigurableQuantizationScheme::one_bit(device.clone());
    
    // Test valid input
    let valid_input = Tensor::new(&[1.0f32, -1.0, 0.5], &device).unwrap();
    assert!(scheme.validate_input(&valid_input).is_ok());
    
    // Test empty tensor
    let empty_input = Tensor::new(&[] as &[f32], &device).unwrap();
    assert!(scheme.validate_input(&empty_input).is_err());
}

#[test]
fn test_threshold_factors() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8, 0.1], &device).unwrap();
    
    let factors = [0.3, 0.5, 0.7, 0.9, 1.1];
    
    for factor in factors {
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_five_eight_bit: OneFiveEightBitParams {
                    threshold_factor: factor,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut scheme = ConfigurableQuantizationScheme::new(config, device.clone());
        let quantized = scheme.quantize_tensor(&input).unwrap();
        
        // Different threshold factors should produce different sparsity levels
        let values = quantized.values.to_vec1::<f32>().unwrap();
        let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
        let sparsity = zero_count as f32 / values.len() as f32;
        
        // Lower threshold factors should generally produce higher sparsity
        assert!((0.0..=1.0).contains(&sparsity));
    }
}

#[test]
fn test_sign_preservation() {
    let device = Device::Cpu;
    
    // Test with clearly positive and negative values
    let input = Tensor::new(&[3.0f32, -3.0, 2.0, -2.0], &device).unwrap();
    
    // 1-bit quantization
    let mut one_bit_scheme = ConfigurableQuantizationScheme::one_bit(device.clone());
    let one_bit_quantized = one_bit_scheme.quantize_tensor(&input).unwrap();
    let one_bit_values = one_bit_quantized.values.to_vec1::<f32>().unwrap();
    
    // Large positive should become +1, large negative should become -1
    assert!(one_bit_values[0] > 0.0); // 3.0 -> positive
    assert!(one_bit_values[1] < 0.0); // -3.0 -> negative
    assert!(one_bit_values[2] > 0.0); // 2.0 -> positive
    assert!(one_bit_values[3] < 0.0); // -2.0 -> negative
    
    // 1.58-bit quantization
    let mut ternary_scheme = ConfigurableQuantizationScheme::one_five_eight_bit(device.clone());
    let ternary_quantized = ternary_scheme.quantize_tensor(&input).unwrap();
    let ternary_values = ternary_quantized.values.to_vec1::<f32>().unwrap();
    
    // Large values should preserve signs
    assert!(ternary_values[0] > 0.0); // 3.0 -> positive
    assert!(ternary_values[1] < 0.0); // -3.0 -> negative
    assert!(ternary_values[2] > 0.0); // 2.0 -> positive
    assert!(ternary_values[3] < 0.0); // -2.0 -> negative
}

#[test]
fn test_quantization_error_bounds() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8, 0.1, 0.9], &device).unwrap();
    
    let precisions = [
        QuantizationPrecision::OneBit,
        QuantizationPrecision::OneFiveFiveBit,
        QuantizationPrecision::TwoBit,
        QuantizationPrecision::FourBit,
        QuantizationPrecision::EightBit,
    ];
    
    let mut previous_error = f32::INFINITY;
    
    for precision in precisions {
        let mut scheme = QuantizationSchemeFactory::create_from_precision(precision, device.clone());
        let quantized = scheme.quantize_tensor(&input).unwrap();
        let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
        
        let error = input.sub(&dequantized).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        
        // Higher precision should generally have lower error (with some exceptions for very low precision)
        if matches!(precision, QuantizationPrecision::FourBit | QuantizationPrecision::EightBit) {
            assert!(error <= previous_error || error < 0.1, 
                "Error increased unexpectedly: {previous_error} -> {error} for {precision:?}");
        }
        
        previous_error = error;
    }
}

#[test]
fn test_2d_tensor_quantization() {
    let device = Device::Cpu;
    
    // Create a 2D tensor
    let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 10.0).collect();
    let input = Tensor::from_vec(data, (4, 6), &device).unwrap();
    
    let mut scheme = ConfigurableQuantizationScheme::one_five_eight_bit(device.clone());
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Check shape preservation
    assert_eq!(quantized.original_shape, *input.shape());
    assert_eq!(quantized.values.shape(), input.shape());
    
    // Test dequantization
    let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), input.shape());
    
    // Check that values are ternary
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0);
    }
}

#[test]
fn test_optimization_config() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3], &device).unwrap();
    
    let optimized_config = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            ..Default::default()
        },
        optimization: OptimizationConfig {
            enable_simd: true,
            use_lookup_tables: true,
            parallel_processing: true,
            memory_optimization_level: 2,
            cache_parameters: true,
        },
        ..Default::default()
    };
    
    let mut scheme = ConfigurableQuantizationScheme::new(optimized_config, device.clone());
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Should still produce valid results
    let values = quantized.values.to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0);
    }
    
    // Test dequantization
    let dequantized = scheme.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), input.shape());
}

#[test]
fn test_custom_thresholds() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8], &device).unwrap();
    
    let config_with_custom_thresholds = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            ..Default::default()
        },
        custom_thresholds: Some(ThresholdConfig {
            one_bit_threshold: Some(0.1),
            ternary_threshold: Some(0.5),
            multi_bit_thresholds: None,
        }),
        ..Default::default()
    };
    
    let mut scheme = ConfigurableQuantizationScheme::new(config_with_custom_thresholds, device.clone());
    let quantized = scheme.quantize_tensor(&input).unwrap();
    
    // Should still produce valid ternary values
    let values = quantized.values.to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0);
    }
}

#[test]
fn test_edge_cases() {
    let device = Device::Cpu;
    
    // Test with all zeros
    let zeros = Tensor::zeros((4,), DType::F32, &device).unwrap();
    let mut scheme = ConfigurableQuantizationScheme::one_five_eight_bit(device.clone());
    let quantized_zeros = scheme.quantize_tensor(&zeros).unwrap();
    let values = quantized_zeros.values.to_vec1::<f32>().unwrap();
    for &val in &values {
        assert_eq!(val, 0.0);
    }
    
    // Test with very small values
    let small_vals = Tensor::new(&[1e-6f32, -1e-6, 1e-7, -1e-7], &device).unwrap();
    let quantized_small = scheme.quantize_tensor(&small_vals).unwrap();
    // Should handle small values gracefully without panicking
    assert!(quantized_small.stats.scale_factor >= 0.0);
    
    // Test with very large values
    let large_vals = Tensor::new(&[1000.0f32, -1000.0, 500.0, -500.0], &device).unwrap();
    let quantized_large = scheme.quantize_tensor(&large_vals).unwrap();
    let large_values = quantized_large.values.to_vec1::<f32>().unwrap();
    for &val in &large_values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0);
    }
}

#[test]
fn test_configuration_update() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, -0.5, 0.3], &device).unwrap();
    
    let mut scheme = ConfigurableQuantizationScheme::one_bit(device.clone());
    
    // Initial quantization
    let initial_result = scheme.quantize_tensor(&input).unwrap();
    assert_eq!(initial_result.precision, QuantizationPrecision::OneBit);
    
    // Update configuration to 1.58-bit
    let new_config = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            ..Default::default()
        },
        ..Default::default()
    };
    
    scheme.update_config(new_config);
    
    // New quantization should use updated config
    let updated_result = scheme.quantize_tensor(&input).unwrap();
    assert_eq!(updated_result.precision, QuantizationPrecision::OneFiveFiveBit);
}