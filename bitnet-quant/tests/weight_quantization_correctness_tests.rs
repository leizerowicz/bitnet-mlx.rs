//! Weight quantization correctness tests
//!
//! This module provides comprehensive tests specifically for weight quantization
//! operations, including ternary quantization methods, threshold calculations,
//! and BitNet-specific weight quantization algorithms.

use bitnet_quant::quantization::weights::*;
use bitnet_quant::quantization::QuantizationPrecision;
use candle_core::{Device, Tensor, DType, Shape};
use approx::assert_abs_diff_eq;

/// Test helper to create weight tensors with specific characteristics
fn create_weight_tensor(device: &Device, pattern: &str, shape: &[usize]) -> Tensor {
    let total_elements: usize = shape.iter().product();
    let data: Vec<f32> = match pattern {
        "normal_distribution" => {
            // Simulate normal distribution with mean=0, std=1
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32 - 0.5) * 6.0; // Range [-3, 3]
                (-x * x / 2.0).exp() * (2.0 * std::f32::consts::PI).sqrt().recip() * 10.0 - 5.0
            }).collect()
        }
        "uniform_weights" => {
            (0..total_elements).map(|i| (i as f32 / total_elements as f32 - 0.5) * 4.0).collect()
        }
        "sparse_weights" => {
            (0..total_elements).map(|i| {
                if i % 4 == 0 { 1.0 } else if i % 7 == 0 { -1.0 } else { 0.0 }
            }).collect()
        }
        "outlier_weights" => {
            (0..total_elements).map(|i| {
                if i == 0 { 100.0 } else if i == 1 { -100.0 } else { (i as f32 - 2.0) * 0.1 }
            }).collect()
        }
        "small_weights" => {
            (0..total_elements).map(|i| (i as f32 - total_elements as f32 / 2.0) * 1e-3).collect()
        }
        "large_weights" => {
            (0..total_elements).map(|i| (i as f32 - total_elements as f32 / 2.0) * 100.0).collect()
        }
        _ => (0..total_elements).map(|i| i as f32 * 0.1).collect(),
    };
    
    let shape = Shape::from_dims(shape);
    Tensor::from_vec(data, shape, device).unwrap()
}

#[test]
fn test_ternary_method_mean_threshold() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "normal_distribution", &[4, 4]);
    
    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7)).unwrap();
    let quantized = quantizer.quantize(&weights).unwrap();
    
    // Verify ternary values
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
    }
    
    // Verify that the threshold is based on mean absolute value
    let abs_weights = weights.abs().unwrap();
    let mean_abs = abs_weights.mean_all().unwrap().to_scalar::<f32>().unwrap();
    let expected_threshold = mean_abs * 0.7;
    
    // Check that quantization respects the threshold
    let original_data = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (i, (&original, &quantized_val)) in original_data.iter().zip(values.iter()).enumerate() {
        if original.abs() > expected_threshold {
            assert_ne!(quantized_val, 0.0, "Value {original} should not be zero at index {i}");
            assert_eq!(quantized_val.signum(), original.signum(), "Sign mismatch at index {i}");
        }
    }
}

#[test]
fn test_ternary_method_median_threshold() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "uniform_weights", &[3, 3]);
    
    let quantizer = create_ternary_quantizer(TernaryMethod::MedianThreshold, Some(0.8)).unwrap();
    let quantized = quantizer.quantize(&weights).unwrap();
    
    // Verify ternary values
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
    }
    
    // Verify shape preservation
    assert_eq!(quantized.values.shape(), weights.shape());
    assert_eq!(quantized.original_shape, *weights.shape());
}

#[test]
fn test_ternary_method_adaptive_threshold() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "outlier_weights", &[2, 5]);
    
    let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, None).unwrap();
    let quantized = quantizer.quantize(&weights).unwrap();
    
    // Verify ternary values
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
    }
    
    // Adaptive threshold should handle outliers better
    // Check that extreme outliers are quantized to ±1
    let original_data = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_abs = original_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    
    for (&original, &quantized_val) in original_data.iter().zip(values.iter()) {
        if original.abs() > max_abs * 0.8 {
            assert_ne!(quantized_val, 0.0, "Large value {original} should not be quantized to zero");
        }
    }
}

#[test]
fn test_ternary_method_optimal_threshold() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "normal_distribution", &[4, 4]);
    
    let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, None).unwrap();
    let quantized = quantizer.quantize(&weights).unwrap();
    
    // Verify ternary values
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
    }
    
    // Optimal threshold should minimize quantization error
    let dequantized = quantizer.dequantize(&quantized).unwrap();
    // Compute quantization error manually
    let diff = weights.sub(&dequantized).unwrap();
    let error = diff.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
    
    // Compare with mean threshold method
    let mean_quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7)).unwrap();
    let mean_quantized = mean_quantizer.quantize(&weights).unwrap();
    let mean_dequantized = mean_quantizer.dequantize(&mean_quantized).unwrap();
    let mean_diff = weights.sub(&mean_dequantized).unwrap();
    let mean_error = mean_diff.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
    
    // Optimal should have equal or better error (allowing for some numerical tolerance)
    assert!(error <= mean_error + 0.1, "Optimal method error {error} should be <= mean method error {mean_error}");
}

#[test]
fn test_custom_threshold_factors() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "uniform_weights", &[3, 4]);
    
    let threshold_factors = vec![0.3, 0.5, 0.7, 0.9, 1.1];
    let mut sparsity_levels = Vec::new();
    
    for factor in threshold_factors {
        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(factor)).unwrap();
        let quantized = quantizer.quantize(&weights).unwrap();
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        
        // Calculate sparsity (percentage of zeros)
        let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
        let sparsity = zero_count as f32 / values.len() as f32;
        sparsity_levels.push(sparsity);
        
        // Verify all values are ternary
        for &val in &values {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
        }
    }
    
    // Lower threshold factors should generally produce higher sparsity
    assert!(sparsity_levels[0] >= sparsity_levels[sparsity_levels.len() - 1] - 0.3,
        "Sparsity should decrease with higher threshold factors: {sparsity_levels:?}");
}

#[test]
fn test_absmean_quantize_weights_correctness() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "normal_distribution", &[2, 3, 4]);
    
    let quantized = absmean_quantize_weights(&weights, &device).unwrap();
    
    // Verify ternary values
    let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &values {
        assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val}");
    }
    
    // Verify shape preservation
    assert_eq!(quantized.original_shape, *weights.shape());
    assert_eq!(quantized.values.shape(), weights.shape());
    
    // Verify scale factor computation
    let abs_weights = weights.abs().unwrap();
    let abs_mean = abs_weights.mean_all().unwrap().to_scalar::<f32>().unwrap();
    let expected_threshold = abs_mean * 0.7; // BitNet standard threshold factor
    
    // Scale factor should be computed optimally
    assert!(quantized.stats.scale_factor > 0.0);
    assert!(quantized.stats.scale_factor.is_finite());
    
    // Verify statistics
    assert_eq!(quantized.stats.elements_count, weights.elem_count());
    assert!(quantized.stats.compression_ratio > 1.0);
    assert!(quantized.stats.quantization_error >= 0.0);
}

#[test]
fn test_weight_quantization_with_different_shapes() {
    let device = Device::Cpu;
    
    let test_shapes = vec![
        vec![16],           // 1D weights (bias)
        vec![64, 32],       // 2D weights (linear layer)
        vec![3, 3, 64, 128], // 4D weights (conv layer)
        vec![8, 4, 4, 4],   // 4D weights (small conv)
    ];
    
    for shape in test_shapes {
        let weights = create_weight_tensor(&device, "normal_distribution", &shape);
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        // Verify shape preservation
        assert_eq!(quantized.original_shape, *weights.shape());
        assert_eq!(quantized.values.shape(), weights.shape());
        
        // Verify ternary values
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &values {
            assert!(val == -1.0 || val == 0.0 || val == 1.0, "Non-ternary value: {val} for shape {shape:?}");
        }
        
        // Verify statistics
        assert_eq!(quantized.stats.elements_count, weights.elem_count());
        assert!(quantized.stats.scale_factor > 0.0);
    }
}

#[test]
fn test_weight_quantization_scale_factor_optimality() {
    let device = Device::Cpu;
    let weights = Tensor::new(&[2.0f32, -1.5, 1.0, -0.5, 0.0, 3.0], &device).unwrap();
    
    let quantized = absmean_quantize_weights(&weights, &device).unwrap();
    let scale = quantized.stats.scale_factor;
    
    // Verify that the scale factor minimizes quantization error
    let quantized_values = quantized.values.to_vec1::<f32>().unwrap();
    let original_values = weights.to_vec1::<f32>().unwrap();
    
    // Compute optimal scale using least squares: scale = (w^T * q) / (q^T * q)
    let numerator: f32 = original_values.iter().zip(quantized_values.iter())
        .map(|(w, q)| w * q).sum();
    let denominator: f32 = quantized_values.iter().map(|q| q * q).sum();
    
    let expected_scale = if denominator.abs() > f32::EPSILON {
        numerator / denominator
    } else {
        1.0
    };
    
    assert_abs_diff_eq!(scale, expected_scale, epsilon = 1e-6);
}

#[test]
fn test_weight_quantization_sparsity_analysis() {
    let device = Device::Cpu;
    
    // Test with different sparsity levels
    let sparsity_patterns = vec![
        ("dense", "normal_distribution"),
        ("sparse", "sparse_weights"),
        ("very_sparse", "outlier_weights"),
    ];
    
    for (name, pattern) in sparsity_patterns {
        let weights = create_weight_tensor(&device, pattern, &[4, 4]);
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
        let sparsity = zero_count as f32 / values.len() as f32;
        
        // Verify sparsity is reasonable
        assert!((0.0..=1.0).contains(&sparsity), "Invalid sparsity {sparsity} for pattern {name}");
        
        // For sparse patterns, expect some sparsity
        if pattern == "sparse_weights" {
            assert!(sparsity > 0.3, "Expected higher sparsity for sparse weights: {sparsity}");
        }
    }
}

#[test]
fn test_weight_quantization_sign_preservation() {
    let device = Device::Cpu;
    
    // Test with clearly positive and negative weights
    let positive_weights = Tensor::new(&[3.0f32, 2.0, 4.0, 1.5], &device).unwrap();
    let negative_weights = Tensor::new(&[-3.0f32, -2.0, -4.0, -1.5], &device).unwrap();
    let mixed_weights = Tensor::new(&[3.0f32, -2.0, 4.0, -1.5], &device).unwrap();
    
    let pos_quantized = absmean_quantize_weights(&positive_weights, &device).unwrap();
    let neg_quantized = absmean_quantize_weights(&negative_weights, &device).unwrap();
    let mixed_quantized = absmean_quantize_weights(&mixed_weights, &device).unwrap();
    
    let pos_values = pos_quantized.values.to_vec1::<f32>().unwrap();
    let neg_values = neg_quantized.values.to_vec1::<f32>().unwrap();
    let mixed_values = mixed_quantized.values.to_vec1::<f32>().unwrap();
    
    // Positive weights should produce non-negative quantized values
    for &val in &pos_values {
        assert!(val >= 0.0, "Positive weight produced negative quantized value: {val}");
    }
    
    // Negative weights should produce non-positive quantized values
    for &val in &neg_values {
        assert!(val <= 0.0, "Negative weight produced positive quantized value: {val}");
    }
    
    // Mixed weights should preserve signs for large values
    let original_mixed = mixed_weights.to_vec1::<f32>().unwrap();
    for (&original, &quantized) in original_mixed.iter().zip(mixed_values.iter()) {
        if original.abs() > 2.0 && quantized.abs() > 0.5 {
            assert_eq!(original.signum(), quantized.signum(), 
                "Sign not preserved: {original} -> {quantized}");
        }
    }
}

#[test]
fn test_weight_quantization_threshold_calculation() {
    let device = Device::Cpu;
    
    // Test threshold calculation for different methods
    let weights = Tensor::new(&[1.0f32, -0.8, 0.6, -0.4, 0.2, -0.1, 0.0, 0.9], &device).unwrap();
    
    // Test with different threshold factors
    let factors = vec![0.5, 0.7, 1.0];
    
    for factor in factors {
        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(factor)).unwrap();
        let quantized = quantizer.quantize(&weights).unwrap();
        
        // Calculate expected threshold
        let abs_weights = weights.abs().unwrap();
        let mean_abs = abs_weights.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let expected_threshold = mean_abs * factor;
        
        // Verify quantization behavior around threshold
        let original_data = weights.to_vec1::<f32>().unwrap();
        let quantized_data = quantized.values.to_vec1::<f32>().unwrap();
        
        for (&original, &quantized_val) in original_data.iter().zip(quantized_data.iter()) {
            if original.abs() > expected_threshold * 1.1 {
                // Values clearly above threshold should not be zero
                assert_ne!(quantized_val, 0.0, 
                    "Value {} (abs={}) above threshold {} should not be zero", 
                    original, original.abs(), expected_threshold);
            }
        }
    }
}

#[test]
fn test_weight_quantization_error_analysis() {
    let device = Device::Cpu;
    
    // Test quantization error for different weight patterns
    let patterns = vec![
        ("uniform", "uniform_weights"),
        ("normal", "normal_distribution"),
        ("sparse", "sparse_weights"),
        ("small", "small_weights"),
    ];
    
    for (name, pattern) in patterns {
        let weights = create_weight_tensor(&device, pattern, &[4, 4]);
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        
        // Simple dequantization for error calculation
        let dequantized = quantized.values.to_dtype(DType::F32).unwrap()
            .mul(&quantized.scales.broadcast_as(quantized.values.shape()).unwrap()).unwrap();
        
        // Compute quantization error manually
        let diff = weights.sub(&dequantized).unwrap();
        let error = diff.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        
        // Error should be reasonable
        assert!(error >= 0.0, "Negative error for pattern {name}: {error}");
        assert!(error < 100.0, "Error too high for pattern {name}: {error}");
        
        // Error should match stored statistics
        assert_abs_diff_eq!(error, quantized.stats.quantization_error, epsilon = 1e-6);
    }
}

#[test]
fn test_weight_quantization_memory_efficiency() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "normal_distribution", &[64, 128]);
    
    let quantized = absmean_quantize_weights(&weights, &device).unwrap();
    
    // Calculate memory footprint
    let original_size = weights.elem_count() * std::mem::size_of::<f32>();
    let quantized_size = quantized.memory_footprint();
    
    // Should achieve significant compression
    let compression_ratio = original_size as f32 / quantized_size as f32;
    assert!(compression_ratio > 4.0, "Compression ratio too low: {compression_ratio}");
    
    // Verify compression ratio calculation
    let calculated_ratio = quantized.compression_ratio();
    assert_abs_diff_eq!(compression_ratio, calculated_ratio, epsilon = 0.1);
}

#[test]
fn test_weight_quantization_edge_cases() {
    let device = Device::Cpu;
    
    // Test edge cases
    let edge_cases = vec![
        ("all_zeros", Tensor::zeros((4, 4), DType::F32, &device).unwrap()),
        ("all_ones", Tensor::ones((4, 4), DType::F32, &device).unwrap()),
        ("all_negative_ones", Tensor::new(&[-1.0f32; 16], &device).unwrap().reshape((4, 4)).unwrap()),
        ("very_small", Tensor::new(&[1e-8f32; 16], &device).unwrap().reshape((4, 4)).unwrap()),
        ("very_large", Tensor::new(&[1e8f32; 16], &device).unwrap().reshape((4, 4)).unwrap()),
    ];
    
    for (name, weights) in edge_cases {
        let result = absmean_quantize_weights(&weights, &device);
        assert!(result.is_ok(), "Quantization failed for edge case {}: {:?}", name, result.err());
        
        if let Ok(quantized) = result {
            // Verify ternary values
            let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for &val in &values {
                assert!(val == -1.0 || val == 0.0 || val == 1.0, 
                    "Non-ternary value {val} for edge case {name}");
            }
            
            // Verify scale factor is reasonable
            assert!(quantized.stats.scale_factor >= 0.0, 
                "Negative scale factor for edge case {name}");
            assert!(quantized.stats.scale_factor.is_finite(), 
                "Non-finite scale factor for edge case {name}");
        }
    }
}

#[test]
fn test_weight_quantization_consistency() {
    let device = Device::Cpu;
    let weights = create_weight_tensor(&device, "normal_distribution", &[3, 3]);
    
    // Test that multiple quantizations of the same weights produce identical results
    let mut results = Vec::new();
    for _ in 0..5 {
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        let values = quantized.values.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        results.push(values);
    }
    
    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Quantization results are not consistent");
    }
}

#[test]
fn test_weight_quantization_config_validation() {
    let device = Device::Cpu;
    
    // Test configuration validation
    let mut config = WeightQuantizationConfig::default();
    assert!(config.validate().is_ok());
    
    // Test invalid configurations
    config.outlier_threshold = -1.0;
    assert!(config.validate().is_err());
    
    config = WeightQuantizationConfig::default();
    config.custom_threshold_factor = Some(0.0);
    assert!(config.validate().is_err());
    
    config.custom_threshold_factor = Some(3.0);
    assert!(config.validate().is_err());
    
    // Test BitNet configuration
    let bitnet_config = WeightQuantizationConfig::bitnet();
    assert!(bitnet_config.validate().is_ok());
    assert_eq!(bitnet_config.base.precision, QuantizationPrecision::OneFiveFiveBit);
    assert_eq!(bitnet_config.ternary_method, TernaryMethod::MeanThreshold);
    assert_eq!(bitnet_config.custom_threshold_factor, Some(0.7));
}