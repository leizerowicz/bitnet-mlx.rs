//! Comprehensive quantization correctness tests
//!
//! This module provides extensive tests to verify the correctness of all quantization
//! operations, including mathematical accuracy, edge cases, and error handling.

use approx::assert_abs_diff_eq;
use bitnet_quant::prelude::*;
use candle_core::{DType, Device, Shape, Tensor};

/// Test helper to create test tensors with known patterns
fn create_test_tensor(device: &Device, pattern: &str) -> Tensor {
    match pattern {
        "uniform" => Tensor::new(&[1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 1.5], device).unwrap(),
        "sparse" => Tensor::new(&[0.0f32, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0], device).unwrap(),
        "outliers" => {
            Tensor::new(&[0.1f32, 0.2, 100.0, 0.3, -100.0, 0.4, 0.5, 0.6], device).unwrap()
        }
        "small_values" => Tensor::new(
            &[1e-6f32, -1e-6, 1e-7, -1e-7, 1e-8, -1e-8, 0.0, 1e-5],
            device,
        )
        .unwrap(),
        "large_values" => Tensor::new(
            &[
                1000.0f32, -1000.0, 500.0, -500.0, 2000.0, -2000.0, 1500.0, -1500.0,
            ],
            device,
        )
        .unwrap(),
        "zeros" => Tensor::zeros((8,), DType::F32, device).unwrap(),
        "ones" => Tensor::ones((8,), DType::F32, device).unwrap(),
        "negative_ones" => Tensor::new(&[-1.0f32; 8], device).unwrap(),
        _ => Tensor::new(&[1.0f32, -1.0, 0.5, -0.5], device).unwrap(),
    }
}

/// Test helper to verify ternary values
fn verify_ternary_values(values: &[f32]) -> bool {
    values.iter().all(|&v| v == -1.0 || v == 0.0 || v == 1.0)
}

/// Test helper to verify binary values
fn verify_binary_values(values: &[f32]) -> bool {
    values.iter().all(|&v| v == -1.0 || v == 1.0)
}

/// Test helper to calculate sparsity
fn calculate_sparsity(values: &[f32]) -> f32 {
    let zero_count = values.iter().filter(|&&v| v.abs() < 1e-6).count();
    zero_count as f32 / values.len() as f32
}

#[test]
fn test_weight_quantization_mathematical_correctness() {
    let device = Device::Cpu;

    // Test different ternary methods for mathematical correctness
    let test_cases = vec![
        ("uniform", TernaryMethod::MeanThreshold),
        ("sparse", TernaryMethod::MedianThreshold),
        ("outliers", TernaryMethod::AdaptiveThreshold),
        ("small_values", TernaryMethod::OptimalThreshold),
    ];

    for (pattern, method) in test_cases {
        let weights = create_test_tensor(&device, pattern);
        let config = WeightQuantizationConfig {
            ternary_method: method,
            custom_threshold_factor: Some(0.7),
            ..Default::default()
        };

        let quantizer = create_ternary_quantizer(method, Some(0.7)).unwrap();
        let quantized = quantizer.quantize(&weights).unwrap();

        // Verify ternary values
        let values = quantized
            .values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            verify_ternary_values(&values),
            "Non-ternary values found for pattern {pattern} with method {method:?}"
        );

        // Verify dequantization preserves shape
        let dequantized = quantizer.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.shape(), weights.shape());

        // Verify scale factor is positive and reasonable
        assert!(quantized.stats.scale_factor > 0.0);
        assert!(quantized.stats.scale_factor < 1000.0);

        // Verify compression ratio
        assert!(quantized.compression_ratio() > 1.0);
    }
}

#[test]
fn test_activation_quantization_mathematical_correctness() {
    let device = Device::Cpu;

    let test_patterns = vec![
        "uniform",
        "sparse",
        "outliers",
        "small_values",
        "large_values",
    ];
    let precisions = vec![
        QuantizationPrecision::OneFiveFiveBit,
        QuantizationPrecision::EightBit,
    ];

    for pattern in test_patterns {
        for precision in &precisions {
            let activations = create_test_tensor(&device, pattern);
            let quantized =
                absmax_quantize_activations(&activations, &device, Some(*precision)).unwrap();

            // Verify quantized values are in correct range
            let values = quantized
                .values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            match precision {
                QuantizationPrecision::OneFiveFiveBit => {
                    assert!(
                        verify_ternary_values(&values),
                        "Non-ternary values found for pattern {pattern} with 1.58-bit precision"
                    );
                }
                QuantizationPrecision::EightBit => {
                    // 8-bit values should be in range [-127, 127]
                    assert!(
                        values.iter().all(|&v| (-127.0..=127.0).contains(&v)),
                        "8-bit values out of range for pattern {pattern}"
                    );
                }
                _ => {}
            }

            // Verify scale factor correctness
            let abs_max = activations
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            if abs_max > f32::EPSILON {
                let expected_scale = match precision {
                    QuantizationPrecision::OneFiveFiveBit => abs_max,
                    QuantizationPrecision::EightBit => abs_max / 127.0,
                    _ => abs_max,
                };
                assert_abs_diff_eq!(quantized.stats.scale_factor, expected_scale, epsilon = 1e-6);
            }

            // Verify compression ratio
            assert!(quantized.stats.compression_ratio > 1.0);
        }
    }
}

#[test]
fn test_quantization_error_bounds() {
    let device = Device::Cpu;

    // Test that quantization error is within reasonable bounds
    let test_data = [
        Tensor::new(&[1.0f32, -1.0, 0.5, -0.5, 0.0], &device).unwrap(),
        Tensor::new(&[2.0f32, -1.5, 0.3, -0.8, 0.1], &device).unwrap(),
        Tensor::new(&[0.1f32, -0.1, 0.05, -0.05, 0.0], &device).unwrap(),
    ];

    for (i, weights) in test_data.iter().enumerate() {
        // Test weight quantization error
        let weight_quantizer =
            create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7)).unwrap();
        let quantized_weights = weight_quantizer.quantize(weights).unwrap();
        let dequantized_weights = weight_quantizer.dequantize(&quantized_weights).unwrap();

        let weight_error =
            QuantizationUtils::compute_quantization_error(weights, &dequantized_weights).unwrap();
        assert!(
            weight_error < 2.0,
            "Weight quantization error too high for test case {i}: {weight_error}"
        );

        // Test activation quantization error
        let quantized_activations = absmax_quantize_activations(weights, &device, None).unwrap();
        let scales_broadcast = quantized_activations
            .scales
            .broadcast_as(quantized_activations.values.shape())
            .unwrap();
        let dequantized_activations = quantized_activations
            .values
            .to_dtype(DType::F32)
            .unwrap()
            .mul(&scales_broadcast)
            .unwrap();

        let activation_error =
            QuantizationUtils::compute_quantization_error(weights, &dequantized_activations)
                .unwrap();
        assert!(
            activation_error < 2.0,
            "Activation quantization error too high for test case {i}: {activation_error}"
        );
    }
}

#[test]
fn test_quantization_symmetry_properties() {
    let device = Device::Cpu;

    // Test that quantization preserves certain symmetry properties
    let positive_weights = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
    let negative_weights = Tensor::new(&[-1.0f32, -2.0, -3.0, -4.0], &device).unwrap();

    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7)).unwrap();

    let pos_quantized = quantizer.quantize(&positive_weights).unwrap();
    let neg_quantized = quantizer.quantize(&negative_weights).unwrap();

    let pos_values = pos_quantized.values.to_vec1::<f32>().unwrap();
    let neg_values = neg_quantized.values.to_vec1::<f32>().unwrap();

    // Check that negative values are negated versions of positive values
    for (pos, neg) in pos_values.iter().zip(neg_values.iter()) {
        if pos.abs() > 1e-6 && neg.abs() > 1e-6 {
            assert_eq!(pos, &(-neg), "Symmetry property violated: {pos} != -{neg}");
        }
    }
}

#[test]
fn test_quantization_scale_factor_correctness() {
    let device = Device::Cpu;

    // Test that scale factors are computed correctly
    let test_cases = vec![
        (vec![1.0f32, -1.0, 0.5, -0.5], 1.0),   // Max abs value is 1.0
        (vec![2.0f32, -2.0, 1.0, -1.0], 2.0),   // Max abs value is 2.0
        (vec![0.5f32, -0.5, 0.25, -0.25], 0.5), // Max abs value is 0.5
    ];

    for (data, expected_max) in test_cases {
        let weights = Tensor::from_slice(&data, (data.len(),), &device).unwrap();

        // Test weight quantization scale
        let quantized_weights = absmean_quantize_weights(&weights, &device).unwrap();

        // For ternary quantization, the scale should be related to the optimal scale
        // that minimizes quantization error
        assert!(quantized_weights.stats.scale_factor > 0.0);

        // Test activation quantization scale
        let quantized_activations = absmax_quantize_activations(&weights, &device, None).unwrap();

        // For absmax quantization, scale should be the absolute maximum
        assert_abs_diff_eq!(
            quantized_activations.stats.scale_factor,
            expected_max,
            epsilon = 1e-6
        );
    }
}

#[test]
fn test_quantization_sparsity_preservation() {
    let device = Device::Cpu;

    // Test that sparse tensors maintain reasonable sparsity after quantization
    let sparse_weights = create_test_tensor(&device, "sparse");
    let original_sparsity = calculate_sparsity(&sparse_weights.to_vec1::<f32>().unwrap());

    let quantized = absmean_quantize_weights(&sparse_weights, &device).unwrap();
    let quantized_values = quantized
        .values
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let quantized_sparsity = calculate_sparsity(&quantized_values);

    // Quantized tensor should maintain some sparsity (though it may be different)
    // For ternary quantization, we expect some sparsity to be preserved
    assert!((0.0..=1.0).contains(&quantized_sparsity));

    // If original was very sparse, quantized should have some sparsity too
    if original_sparsity > 0.5 {
        assert!(
            quantized_sparsity > 0.1,
            "Sparsity not preserved: {original_sparsity} -> {quantized_sparsity}"
        );
    }
}

#[test]
fn test_quantization_threshold_sensitivity() {
    let device = Device::Cpu;

    // Test how different threshold factors affect quantization
    let weights = Tensor::new(&[1.0f32, -0.8, 0.6, -0.4, 0.2, -0.1], &device).unwrap();
    let threshold_factors = vec![0.3, 0.5, 0.7, 0.9, 1.1];

    let mut sparsity_levels = Vec::new();

    for factor in threshold_factors {
        let quantizer =
            create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(factor)).unwrap();
        let quantized = quantizer.quantize(&weights).unwrap();
        let values = quantized.values.to_vec1::<f32>().unwrap();
        let sparsity = calculate_sparsity(&values);
        sparsity_levels.push(sparsity);
    }

    // Lower threshold factors should generally produce higher sparsity
    // (though this is not always strictly monotonic due to discrete quantization)
    assert!(
        sparsity_levels[0] >= sparsity_levels[sparsity_levels.len() - 1] - 0.2,
        "Threshold sensitivity test failed: sparsity levels {sparsity_levels:?}"
    );
}

#[test]
fn test_quantization_numerical_stability() {
    let device = Device::Cpu;

    // Test quantization with edge cases that might cause numerical issues
    let edge_cases = vec![
        ("very_small", create_test_tensor(&device, "small_values")),
        ("very_large", create_test_tensor(&device, "large_values")),
        ("all_zeros", create_test_tensor(&device, "zeros")),
        ("all_ones", create_test_tensor(&device, "ones")),
        (
            "all_negative_ones",
            create_test_tensor(&device, "negative_ones"),
        ),
    ];

    for (name, tensor) in edge_cases {
        // Test weight quantization stability
        let weight_result = absmean_quantize_weights(&tensor, &device);
        assert!(
            weight_result.is_ok(),
            "Weight quantization failed for {}: {:?}",
            name,
            weight_result.err()
        );

        if let Ok(quantized) = weight_result {
            assert!(
                quantized.stats.scale_factor.is_finite(),
                "Non-finite scale factor for {name}"
            );
            assert!(
                quantized.stats.scale_factor >= 0.0,
                "Negative scale factor for {name}"
            );

            let values = quantized
                .values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                verify_ternary_values(&values),
                "Non-ternary values for {name}"
            );
        }

        // Test activation quantization stability
        let activation_result = absmax_quantize_activations(&tensor, &device, None);
        assert!(
            activation_result.is_ok(),
            "Activation quantization failed for {}: {:?}",
            name,
            activation_result.err()
        );

        if let Ok(quantized) = activation_result {
            assert!(
                quantized.stats.scale_factor.is_finite(),
                "Non-finite scale factor for {name}"
            );
            assert!(
                quantized.stats.scale_factor >= 0.0,
                "Negative scale factor for {name}"
            );
        }
    }
}

#[test]
fn test_quantization_determinism() {
    let device = Device::Cpu;

    // Test that quantization is deterministic (same input produces same output)
    let weights = Tensor::new(&[1.5f32, -0.8, 0.2, -2.1, 0.0, 1.0], &device).unwrap();

    // Run quantization multiple times
    let mut results = Vec::new();
    for _ in 0..5 {
        let quantized = absmean_quantize_weights(&weights, &device).unwrap();
        let values = quantized
            .values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        results.push(values);
    }

    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Quantization is not deterministic");
    }

    // Test activation quantization determinism
    let mut activation_results = Vec::new();
    for _ in 0..5 {
        let quantized = absmax_quantize_activations(&weights, &device, None).unwrap();
        let values = quantized
            .values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        activation_results.push(values);
    }

    for i in 1..activation_results.len() {
        assert_eq!(
            activation_results[0], activation_results[i],
            "Activation quantization is not deterministic"
        );
    }
}

#[test]
fn test_quantization_shape_preservation() {
    let device = Device::Cpu;

    // Test various tensor shapes
    let shapes = vec![vec![10], vec![5, 4], vec![2, 3, 4], vec![2, 2, 2, 2]];

    for shape_dims in shapes {
        let total_elements: usize = shape_dims.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 - total_elements as f32 / 2.0) / 10.0)
            .collect();
        let shape = Shape::from_dims(&shape_dims);
        let tensor = Tensor::from_vec(data, shape.clone(), &device).unwrap();

        // Test weight quantization
        let quantized_weights = absmean_quantize_weights(&tensor, &device).unwrap();
        assert_eq!(quantized_weights.original_shape, shape);
        assert_eq!(quantized_weights.values.shape(), &shape);

        // Test activation quantization
        let quantized_activations = absmax_quantize_activations(&tensor, &device, None).unwrap();
        assert_eq!(quantized_activations.original_shape, shape);
        assert_eq!(quantized_activations.values.shape(), &shape);
    }
}

#[test]
fn test_quantization_error_propagation() {
    let device = Device::Cpu;

    // Test that quantization errors don't accumulate catastrophically
    let original = Tensor::new(&[1.0f32, -0.5, 0.3, -0.8, 0.1], &device).unwrap();

    // Quantize and dequantize multiple times
    let mut current = original.clone();
    let mut errors = Vec::new();

    for i in 0..3 {
        let quantized = absmean_quantize_weights(&current, &device).unwrap();

        // Simple dequantization for testing
        let dequantized = quantized
            .values
            .to_dtype(DType::F32)
            .unwrap()
            .mul(
                &quantized
                    .scales
                    .broadcast_as(quantized.values.shape())
                    .unwrap(),
            )
            .unwrap();

        let error = QuantizationUtils::compute_quantization_error(&original, &dequantized).unwrap();
        errors.push(error);

        current = dequantized;

        // Error should not grow exponentially
        assert!(error < 10.0, "Error too large at iteration {i}: {error}");
    }

    // Errors should not increase dramatically
    for i in 1..errors.len() {
        assert!(
            errors[i] < errors[i - 1] * 5.0,
            "Error increased too much: {} -> {}",
            errors[i - 1],
            errors[i]
        );
    }
}

#[test]
fn test_quantization_compression_ratios() {
    let device = Device::Cpu;

    // Test that compression ratios are as expected
    let test_tensor =
        Tensor::new(&[1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 1.5], &device).unwrap();

    // Weight quantization (1.58-bit)
    let quantized_weights = absmean_quantize_weights(&test_tensor, &device).unwrap();
    let weight_compression = quantized_weights.compression_ratio();

    // Should achieve significant compression (f32 to ~1.58 bits)
    assert!(
        weight_compression > 10.0,
        "Weight compression ratio too low: {weight_compression}"
    );
    assert!(
        weight_compression < 50.0,
        "Weight compression ratio suspiciously high: {weight_compression}"
    );

    // Activation quantization (1.58-bit)
    let quantized_activations = absmax_quantize_activations(&test_tensor, &device, None).unwrap();
    let activation_compression = quantized_activations.stats.compression_ratio;

    assert!(
        activation_compression > 10.0,
        "Activation compression ratio too low: {activation_compression}"
    );
    assert!(
        activation_compression < 50.0,
        "Activation compression ratio suspiciously high: {activation_compression}"
    );

    // 8-bit activation quantization
    let quantized_8bit =
        absmax_quantize_activations(&test_tensor, &device, Some(QuantizationPrecision::EightBit))
            .unwrap();
    let compression_8bit = quantized_8bit.stats.compression_ratio;

    // 8-bit should have lower compression than 1.58-bit
    assert!(
        compression_8bit > 3.0,
        "8-bit compression ratio too low: {compression_8bit}"
    );
    assert!(
        compression_8bit < activation_compression,
        "8-bit compression should be lower than 1.58-bit"
    );
}

#[test]
fn test_quantization_statistics_accuracy() {
    let device = Device::Cpu;

    // Test that quantization statistics are computed correctly
    let weights = Tensor::new(&[2.0f32, -1.5, 1.0, -0.5, 0.0, 3.0], &device).unwrap();

    let quantized = absmean_quantize_weights(&weights, &device).unwrap();

    // Verify element count
    assert_eq!(quantized.stats.elements_count, 6);

    // Verify min/max values
    assert_eq!(quantized.stats.min_value, -1.5);
    assert_eq!(quantized.stats.max_value, 3.0);

    // Verify scale factor is positive
    assert!(quantized.stats.scale_factor > 0.0);

    // Verify compression ratio is reasonable
    assert!(quantized.stats.compression_ratio > 1.0);
    assert!(quantized.stats.compression_ratio < 100.0);

    // Verify quantization error is computed
    assert!(quantized.stats.quantization_error >= 0.0);
}

#[test]
fn test_quantization_boundary_conditions() {
    let device = Device::Cpu;

    // Test quantization at various boundary conditions
    let boundary_cases = vec![
        ("single_element", vec![1.0f32]),
        ("two_elements", vec![1.0f32, -1.0]),
        ("alternating", vec![1.0f32, -1.0, 1.0, -1.0]),
        ("increasing", vec![0.1f32, 0.2, 0.3, 0.4, 0.5]),
        ("decreasing", vec![0.5f32, 0.4, 0.3, 0.2, 0.1]),
    ];

    for (name, data) in boundary_cases {
        let tensor = Tensor::from_slice(&data, (data.len(),), &device).unwrap();

        // Test weight quantization
        let weight_result = absmean_quantize_weights(&tensor, &device);
        assert!(
            weight_result.is_ok(),
            "Weight quantization failed for {name}"
        );

        if let Ok(quantized) = weight_result {
            let values = quantized
                .values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                verify_ternary_values(&values),
                "Non-ternary values for {name}"
            );
            assert_eq!(values.len(), data.len(), "Length mismatch for {name}");
        }

        // Test activation quantization
        let activation_result = absmax_quantize_activations(&tensor, &device, None);
        assert!(
            activation_result.is_ok(),
            "Activation quantization failed for {name}"
        );

        if let Ok(quantized) = activation_result {
            let values = quantized
                .values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(values.len(), data.len(), "Length mismatch for {name}");
        }
    }
}
