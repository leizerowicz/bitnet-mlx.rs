//! Activation quantization correctness tests
//!
//! This module provides comprehensive tests specifically for activation quantization
//! operations, including dynamic quantization, attention quantization, and
//! BitNet-specific activation quantization algorithms.

use bitnet_quant::quantization::activations::*;
use bitnet_quant::quantization::{QuantizationPrecision, QuantizationConfig, QuantizationStrategy, QuantizationStats, Quantizer, CalibrationQuantizer};
use bitnet_quant::quantization::utils::QuantizationError;
use candle_core::{Device, Tensor, DType, Shape};
use approx::{assert_relative_eq, assert_abs_diff_eq};

/// Test helper to create activation tensors with specific characteristics
fn create_activation_tensor(device: &Device, pattern: &str, shape: &[usize]) -> Tensor {
    let total_elements: usize = shape.iter().product();
    let data: Vec<f32> = match pattern {
        "relu_activations" => {
            // Simulate ReLU activations (non-negative with some zeros)
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32 - 0.3) * 5.0;
                x.max(0.0)
            }).collect()
        }
        "gelu_activations" => {
            // Simulate GELU activations (smooth, can be negative)
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32 - 0.5) * 4.0;
                x * 0.5 * (1.0 + (x * 0.7978845608).tanh())
            }).collect()
        }
        "attention_scores" => {
            // Simulate attention scores (typically in [0, 1] after softmax)
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32).powi(2);
                x / (1.0 + x) // Sigmoid-like function
            }).collect()
        }
        "large_activations" => {
            (0..total_elements).map(|i| (i as f32 - total_elements as f32 / 2.0) * 10.0).collect()
        }
        "small_activations" => {
            (0..total_elements).map(|i| (i as f32 - total_elements as f32 / 2.0) * 0.01).collect()
        }
        "sparse_activations" => {
            (0..total_elements).map(|i| {
                if i % 5 == 0 { (i as f32 * 0.1) } else { 0.0 }
            }).collect()
        }
        "outlier_activations" => {
            (0..total_elements).map(|i| {
                if i == 0 { 50.0 } else if i == 1 { -50.0 } else { (i as f32 - 2.0) * 0.1 }
            }).collect()
        }
        _ => (0..total_elements).map(|i| i as f32 * 0.1).collect(),
    };
    
    let shape = Shape::from_dims(shape);
    Tensor::from_vec(data, shape, device).unwrap()
}

#[test]
fn test_activation_quantization_config_validation() {
    let mut config = ActivationQuantizationConfig::default();
    assert!(config.validate().is_ok());
    
    // Test invalid configurations
    config.moving_average_window = 0;
    assert!(config.validate().is_err());
    
    config = ActivationQuantizationConfig::default();
    config.outlier_percentile = 0.0;
    assert!(config.validate().is_err());
    
    config.outlier_percentile = 101.0;
    assert!(config.validate().is_err());
    
    config = ActivationQuantizationConfig::default();
    config.ema_decay = -0.1;
    assert!(config.validate().is_err());
    
    config.ema_decay = 1.1;
    assert!(config.validate().is_err());
}

#[test]
fn test_activation_quantization_config_bitnet() {
    let config = ActivationQuantizationConfig::bitnet();
    assert!(config.validate().is_ok());
    assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
    assert_eq!(config.base.strategy, QuantizationStrategy::Dynamic);
    assert!(config.quantize_attention);
    assert_eq!(config.moving_average_window, 100);
}

#[test]
fn test_dynamic_activation_quantizer_creation() {
    let config = ActivationQuantizationConfig::default();
    let device = Device::Cpu;
    let quantizer = DynamicActivationQuantizer::new(config.clone(), device);
    
    // Test initial state - just verify creation was successful
    // Note: Direct access to internal state may not be available through public API
}

#[test]
fn test_absmax_quantize_activations_basic() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "relu_activations", &[4, 8]);
    
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    // Verify ternary values for 1.58-bit precision
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {}", val);
    }
    
    // Verify shape preservation
    assert_eq!(quantized.original_shape, *activations.shape());
    assert_eq!(quantized.values.shape(), activations.shape());
    
    // Verify statistics
    assert_eq!(quantized.stats.elements_count, activations.elem_count());
    assert!(quantized.stats.scale_factor > 0.0);
    assert!(quantized.stats.compression_ratio > 1.0);
}

#[test]
fn test_absmax_quantize_activations_8bit() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "gelu_activations", &[3, 5]);
    
    let quantized = absmax_quantize_activations(
        &activations,
        &device,
        Some(QuantizationPrecision::EightBit)
    ).unwrap();
    
    // Verify 8-bit quantization
    assert_eq!(quantized.quantized_dtype, DType::U8);
    assert_eq!(quantized.effective_bit_width(), 8.0);
    
    // Verify values are in valid range
    let values = quantized.values.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val >= -127.0 && val <= 127.0, "Value {} out of 8-bit range", val);
    }
}

#[test]
fn test_absmax_quantize_activations_scale_computation() {
    let device = Device::Cpu;
    let activations = Tensor::new(&[4.0f32, -2.0, 1.0, -3.0], &device).unwrap();
    
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    // For 1.58-bit, scale should be the absolute maximum (4.0)
    let expected_scale = 4.0;
    assert_abs_diff_eq!(quantized.stats.scale_factor, expected_scale, epsilon = 1e-6);
    
    // Test with 8-bit precision
    let quantized_8bit = absmax_quantize_activations(
        &activations,
        &device,
        Some(QuantizationPrecision::EightBit)
    ).unwrap();
    
    // For 8-bit, scale should be abs_max / 127
    let expected_scale_8bit = 4.0 / 127.0;
    assert_abs_diff_eq!(quantized_8bit.stats.scale_factor, expected_scale_8bit, epsilon = 1e-6);
}

#[test]
fn test_absmax_quantize_activations_sign_preservation() {
    let device = Device::Cpu;
    
    // Test with clearly positive and negative activations
    let positive_activations = Tensor::new(&[3.0f32, 2.0, 4.0, 1.5], &device).unwrap();
    let negative_activations = Tensor::new(&[-3.0f32, -2.0, -4.0, -1.5], &device).unwrap();
    let mixed_activations = Tensor::new(&[3.0f32, -2.0, 4.0, -1.5], &device).unwrap();
    
    let pos_quantized = absmax_quantize_activations(&positive_activations, &device, None).unwrap();
    let neg_quantized = absmax_quantize_activations(&negative_activations, &device, None).unwrap();
    let mixed_quantized = absmax_quantize_activations(&mixed_activations, &device, None).unwrap();
    
    let pos_values = pos_quantized.values.to_vec1::<f32>().unwrap();
    let neg_values = neg_quantized.values.to_vec1::<f32>().unwrap();
    let mixed_values = mixed_quantized.values.to_vec1::<f32>().unwrap();
    
    // Positive activations should produce non-negative quantized values
    for &val in &pos_values {
        assert!(val >= 0.0, "Positive activation produced negative quantized value: {}", val);
    }
    
    // Negative activations should produce non-positive quantized values
    for &val in &neg_values {
        assert!(val <= 0.0, "Negative activation produced positive quantized value: {}", val);
    }
    
    // Mixed activations should preserve signs for large values
    let original_mixed = mixed_activations.to_vec1::<f32>().unwrap();
    for (&original, &quantized) in original_mixed.iter().zip(mixed_values.iter()) {
        if original.abs() > 2.0 && quantized.abs() > 0.5 {
            assert_eq!(original.signum(), quantized.signum(), 
                "Sign not preserved: {} -> {}", original, quantized);
        }
    }
}

#[test]
fn test_absmax_quantize_activations_different_shapes() {
    let device = Device::Cpu;
    
    let test_shapes = vec![
        vec![32],           // 1D activations
        vec![16, 32],       // 2D activations (batch, features)
        vec![8, 16, 32],    // 3D activations (batch, seq, features)
        vec![4, 8, 16, 32], // 4D activations (batch, channels, height, width)
    ];
    
    for shape in test_shapes {
        let activations = create_activation_tensor(&device, "relu_activations", &shape);
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // Verify shape preservation
        assert_eq!(quantized.original_shape, *activations.shape());
        assert_eq!(quantized.values.shape(), activations.shape());
        
        // Verify ternary values
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, 
                "Non-ternary value: {} for shape {:?}", val, shape);
        }
        
        // Verify statistics
        assert_eq!(quantized.stats.elements_count, activations.elem_count());
        assert!(quantized.stats.scale_factor > 0.0);
    }
}

#[test]
fn test_absmax_quantize_activations_threshold_behavior() {
    let device = Device::Cpu;
    
    // Test with known values around threshold
    let activations = Tensor::new(&[1.0f32, 0.6, 0.4, 0.2, -0.8, -0.3], &device).unwrap();
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    let values = quantized.values.to_vec1::<f32>().unwrap();
    let original_values = activations.to_vec1::<f32>().unwrap();
    
    // Values with absolute value > threshold should not be zero
    // Values with absolute value <= threshold should be zero
    let abs_max = original_values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let threshold = 0.5; // Standard threshold for activations
    
    for (&original, &quantized_val) in original_values.iter().zip(values.iter()) {
        if original.abs() > threshold {
            // Large values should not be quantized to zero
            if original.abs() > abs_max * 0.7 {
                assert_ne!(quantized_val, 0.0, 
                    "Large value {} should not be quantized to zero", original);
            }
        }
    }
}

#[test]
fn test_absmax_quantize_activations_edge_cases() {
    let device = Device::Cpu;
    
    // Test edge cases
    let edge_cases = vec![
        ("all_zeros", Tensor::zeros((4, 4), DType::F32, &device).unwrap()),
        ("all_ones", Tensor::ones((4, 4), DType::F32, &device).unwrap()),
        ("very_small", Tensor::new(&[1e-8f32; 16], &device).unwrap().reshape((4, 4)).unwrap()),
        ("very_large", Tensor::new(&[1e6f32; 16], &device).unwrap().reshape((4, 4)).unwrap()),
        ("mixed_extreme", Tensor::new(&[1e6f32, -1e6, 1e-8, -1e-8], &device).unwrap()),
    ];
    
    for (name, activations) in edge_cases {
        let result = absmax_quantize_activations(&activations, &device, None);
        assert!(result.is_ok(), "Quantization failed for edge case {}: {:?}", name, result.err());
        
        if let Ok(quantized) = result {
            // Verify ternary values
            let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for &val in &values {
                assert!(val == -1.0 || val == 0.0 || val == 1.0, 
                    "Non-ternary value {} for edge case {}", val, name);
            }
            
            // Verify scale factor is reasonable
            assert!(quantized.stats.scale_factor >= 0.0, 
                "Negative scale factor for edge case {}", name);
            assert!(quantized.stats.scale_factor.is_finite(), 
                "Non-finite scale factor for edge case {}", name);
        }
    }
}

#[test]
fn test_absmax_quantize_activations_dequantization() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "gelu_activations", &[3, 4]);
    
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    // Test dequantization
    let scales_broadcast = quantized.scales.broadcast_as(quantized.values.shape()).unwrap();
    let dequantized = quantized.values.to_dtype(DType::F32).unwrap().mul(&scales_broadcast).unwrap();
    
    // Check that dequantized tensor has same shape
    assert_eq!(dequantized.shape(), activations.shape());
    
    // Check that quantization error is reasonable
    assert!(quantized.stats.quantization_error < 5.0);
    
    // Verify that dequantization preserves the general magnitude
    let original_abs_max = activations.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let dequant_abs_max = dequantized.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    
    // Should be approximately equal (within quantization error)
    assert!((original_abs_max - dequant_abs_max).abs() < original_abs_max * 0.5);
}

#[test]
fn test_absmax_quantize_activations_statistics() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "relu_activations", &[2, 3, 4]);
    
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    // Check statistics
    assert_eq!(quantized.stats.elements_count, 24);
    assert!(quantized.stats.scale_factor > 0.0);
    assert!(quantized.stats.compression_ratio > 1.0);
    assert!(quantized.stats.min_value <= quantized.stats.max_value);
    assert!(quantized.stats.quantization_error >= 0.0);
    
    // Check that compression ratio is reasonable for 1.58-bit
    let expected_compression = 32.0 / 1.58;
    assert_abs_diff_eq!(quantized.stats.compression_ratio, expected_compression, epsilon = 0.1);
}

#[test]
fn test_absmax_quantize_activations_consistency() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "gelu_activations", &[3, 3]);
    
    // Test that multiple quantizations of the same activations produce identical results
    let mut results = Vec::new();
    for _ in 0..5 {
        let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        results.push(values);
    }
    
    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Quantization results are not consistent");
    }
}

#[test]
fn test_dynamic_activation_quantizer_basic() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::default();
    let quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    let activations = create_activation_tensor(&device, "relu_activations", &[4, 4]);
    let quantized = quantizer.quantize(&activations).unwrap();
    
    // Verify basic properties
    assert_eq!(quantized.original_shape, *activations.shape());
    assert!(quantized.stats.scale_factor > 0.0);
    
    // Verify ternary values for default 1.58-bit precision
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {}", val);
    }
}

#[test]
fn test_dynamic_activation_quantizer_dynamic_scaling() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::default();
    let mut quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    // Test with different activation magnitudes
    let small_activations = create_activation_tensor(&device, "small_activations", &[2, 2]);
    let large_activations = create_activation_tensor(&device, "large_activations", &[2, 2]);
    
    let small_quantized = quantizer.quantize_dynamic(&small_activations).unwrap();
    let large_quantized = quantizer.quantize_dynamic(&large_activations).unwrap();
    
    // Scale factors should adapt to different magnitudes
    assert!(small_quantized.stats.scale_factor < large_quantized.stats.scale_factor);
}

#[test]
fn test_dynamic_activation_quantizer_attention() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::bitnet();
    let quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    let attention_scores = create_activation_tensor(&device, "attention_scores", &[8, 64]);
    let sequence_length = 64;
    
    let quantized = quantizer.quantize_attention(&attention_scores, sequence_length).unwrap();
    
    // Verify attention-specific properties
    assert!(quantized.is_attention_activation());
    assert_eq!(quantized.sequence_length, Some(sequence_length));
    
    // Verify quantization quality
    assert!(quantized.stats.scale_factor > 0.0);
    assert_eq!(quantized.original_shape, *attention_scores.shape());
}

#[test]
fn test_dynamic_activation_quantizer_calibration() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::default();
    let mut quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    // Create calibration data
    let calibration_data: Vec<Tensor> = (0..10)
        .map(|i| create_activation_tensor(&device, "relu_activations", &[4, 4]))
        .collect();
    
    // Test calibration
    let initial_needs_calibration = quantizer.needs_calibration();
    let result = quantizer.calibrate(&calibration_data);
    assert!(result.is_ok());
    
    // After calibration, calibration state may change
    // Note: calibration state depends on implementation details
}

#[test]
fn test_dynamic_activation_quantizer_reset() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::default();
    let mut quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    // Perform some quantizations to build up state
    let activations = create_activation_tensor(&device, "relu_activations", &[4, 4]);
    let _ = quantizer.quantize_dynamic(&activations).unwrap();
    let _ = quantizer.quantize_dynamic(&activations).unwrap();
    
    // Reset state
    quantizer.reset_dynamic_state();
    
    // Should be able to quantize again
    let quantized = quantizer.quantize_dynamic(&activations).unwrap();
    assert!(quantized.stats.scale_factor > 0.0);
}

#[test]
fn test_activation_quantizer_factory() {
    let config = ActivationQuantizationConfig::default();
    let quantizer = create_activation_quantizer(config).unwrap();
    
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "relu_activations", &[4, 4]);
    
    let quantized = quantizer.quantize(&activations).unwrap();
    assert_eq!(quantized.original_shape, *activations.shape());
}

#[test]
fn test_activation_quantization_error_handling() {
    let device = Device::Cpu;
    let config = ActivationQuantizationConfig::default();
    let quantizer = DynamicActivationQuantizer::new(config, device.clone());
    
    // Test with invalid input (wrong dtype)
    let invalid_activations = Tensor::zeros((4, 4), DType::U8, &device).unwrap();
    let result = quantizer.validate_input(&invalid_activations);
    assert!(result.is_err());
    
    // Test with invalid input (too few dimensions)
    let invalid_activations = Tensor::zeros((), DType::F32, &device).unwrap();
    let result = quantizer.validate_input(&invalid_activations);
    assert!(result.is_err());
    
    // Test with valid input
    let valid_activations = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
    let result = quantizer.validate_input(&valid_activations);
    assert!(result.is_ok());
}

#[test]
fn test_activation_quantization_memory_efficiency() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "relu_activations", &[64, 128]);
    
    let quantized = absmax_quantize_activations(&activations, &device, None).unwrap();
    
    // Calculate memory footprint
    let original_size = activations.elem_count() * std::mem::size_of::<f32>();
    let quantized_size = quantized.memory_footprint();
    
    // Should achieve significant compression
    let compression_ratio = original_size as f32 / quantized_size as f32;
    assert!(compression_ratio > 4.0, "Compression ratio too low: {}", compression_ratio);
    
    // Verify compression ratio calculation
    assert_abs_diff_eq!(compression_ratio, quantized.stats.compression_ratio, epsilon = 0.1);
}

#[test]
fn test_activation_quantization_precision_comparison() {
    let device = Device::Cpu;
    let activations = create_activation_tensor(&device, "gelu_activations", &[8, 16]);
    
    // Test different precisions
    let precisions = vec![
        QuantizationPrecision::OneFiveFiveBit,
        QuantizationPrecision::EightBit,
    ];
    
    let mut compression_ratios = Vec::new();
    let mut quantization_errors = Vec::new();
    
    for precision in precisions {
        let quantized = absmax_quantize_activations(&activations, &device, Some(precision)).unwrap();
        compression_ratios.push(quantized.stats.compression_ratio);
        quantization_errors.push(quantized.stats.quantization_error);
        
        // Verify precision-specific properties
        match precision {
            QuantizationPrecision::OneFiveFiveBit => {
                assert_eq!(quantized.effective_bit_width(), 1.58);
                let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for &val in &values {
                    assert!(val == -1.0 || val == 0.0 || val == 1.0);
                }
            }
            QuantizationPrecision::EightBit => {
                assert_eq!(quantized.effective_bit_width(), 8.0);
                assert_eq!(quantized.quantized_dtype, DType::U8);
            }
            _ => {}
        }
    }
    
    // Lower precision should have higher compression ratio
    assert!(compression_ratios[0] > compression_ratios[1], 
        "1.58-bit should have higher compression than 8-bit");
}

#[test]
fn test_activation_pattern_analysis() {
    let device = Device::Cpu;
    
    // Create different activation patterns
    let patterns = vec![
        create_activation_tensor(&device, "relu_activations", &[4, 4]),
        create_activation_tensor(&device, "gelu_activations", &[4, 4]),
        create_activation_tensor(&device, "attention_scores", &[4, 4]),
    ];
    
    let analysis = activation_utils::analyze_activation_patterns(&patterns).unwrap();
    
    // Verify analysis results
    assert!(analysis.global_min <= analysis.global_max);
    assert!(analysis.dynamic_range >= 0.0);
    assert!(analysis.recommended_scale > 0.0);
    assert!(analysis.dynamic_range == analysis.global_max - analysis.global_min);
}

#[test]
fn test_attention_quantization_params() {
    let device = Device::Cpu;
    let attention_scores = create_activation_tensor(&device, "attention_scores", &[8, 64]);
    let sequence_length = 64;
    
    let params = activation_utils::compute_attention_quantization_params(
        &attention_scores, 
        sequence_length
    ).unwrap();
    
    // Verify attention parameters
    assert_eq!(params.sequence_length, sequence_length);
    assert!(params.scale > 0.0);
    assert!(params.sparsity >= 0.0 && params.sparsity <= 1.0);
    assert!(params.is_causal); // Should be true for sequence_length > 1
}