//! Comprehensive tests for SIMD operations

use crate::simd::{
    detect_simd_capabilities, vectorized_matrix_multiply, vectorized_pack_ternary,
    vectorized_ternary_dequantize, vectorized_ternary_quantize, vectorized_unpack_ternary,
    AlignedVec, SimdConfig, SimdMatrixOps, SimdPackingOps, SimdTernaryOps,
};
use approx::assert_abs_diff_eq;

#[test]
fn test_simd_capability_detection() {
    let caps = detect_simd_capabilities();
    println!("Detected SIMD capabilities: {}", caps.description());

    // Basic sanity checks
    assert!(caps.vector_size >= 4);
    assert!(caps.cache_line_size >= 16);

    if caps.has_simd() {
        println!(
            "SIMD support available with {}-byte vectors",
            caps.vector_size
        );
    } else {
        println!("No SIMD support detected - using scalar fallbacks");
    }
}

#[test]
fn test_ternary_quantization_correctness() {
    let input = vec![2.5, 0.3, -1.7, 0.0, -0.4, 3.2, -2.1, 0.1];
    let mut output_simd = vec![0i8; input.len()];
    let mut output_scalar = vec![0i8; input.len()];
    let threshold = 1.0;

    // SIMD quantization
    vectorized_ternary_quantize(&input, &mut output_simd, threshold).unwrap();

    // Reference scalar quantization
    for (i, &val) in input.iter().enumerate() {
        output_scalar[i] = if val > threshold {
            1
        } else if val < -threshold {
            -1
        } else {
            0
        };
    }

    assert_eq!(output_simd, output_scalar);

    // Verify specific expected values
    let expected = vec![1, 0, -1, 0, 0, 1, -1, 0];
    assert_eq!(output_simd, expected);
}

#[test]
fn test_ternary_dequantization_correctness() {
    let input = vec![1i8, 0, -1, 0, 1, -1, 0, 1];
    let mut output_simd = vec![0.0f32; input.len()];
    let mut output_scalar = vec![0.0f32; input.len()];
    let scale = 2.5;

    // SIMD dequantization
    vectorized_ternary_dequantize(&input, &mut output_simd, scale).unwrap();

    // Reference scalar dequantization
    for (i, &val) in input.iter().enumerate() {
        output_scalar[i] = val as f32 * scale;
    }

    for (simd, scalar) in output_simd.iter().zip(output_scalar.iter()) {
        assert_abs_diff_eq!(*simd, *scalar, epsilon = 1e-6);
    }
}

#[test]
fn test_matrix_vector_multiplication() {
    let m = 4;
    let k = 6;

    // Matrix A (4x6, ternary)
    let a = vec![
        1, -1, 0, 1, 0, -1, // row 0
        0, 1, -1, 0, 1, 0, // row 1
        -1, 0, 1, -1, 0, 1, // row 2
        0, 0, 0, 1, 1, -1, // row 3
    ];

    // Vector x (6x1)
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Result vector y (4x1)
    let mut y_simd = vec![0.0; m];
    let mut y_scalar = vec![0.0; m];
    let scale = 1.5;

    let ops = SimdMatrixOps::new();

    // SIMD computation
    ops.ternary_matrix_vector(&a, &x, &mut y_simd, m, k, scale)
        .unwrap();

    // Reference scalar computation
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += (a[i * k + j] as f32) * x[j];
        }
        y_scalar[i] = sum * scale;
    }

    for (simd, scalar) in y_simd.iter().zip(y_scalar.iter()) {
        assert_abs_diff_eq!(*simd, *scalar, epsilon = 1e-5);
    }
}

#[test]
fn test_matrix_matrix_multiplication() {
    let m = 3;
    let k = 4;
    let n = 3;

    // Matrix A (3x4, ternary)
    let a = vec![
        1, -1, 0, 1, // row 0
        0, 1, -1, 0, // row 1
        -1, 0, 1, 1, // row 2
    ];

    // Matrix B (4x3)
    let b = vec![
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
        7.0, 8.0, 9.0, // row 2
        10.0, 11.0, 12.0, // row 3
    ];

    let mut c_simd = vec![0.0; m * n];
    let mut c_scalar = vec![0.0; m * n];
    let scale = 2.0;

    // SIMD computation
    vectorized_matrix_multiply(&a, &b, &mut c_simd, m, k, n, scale).unwrap();

    // Reference scalar computation
    c_scalar.fill(0.0);
    for i in 0..m {
        for kk in 0..k {
            let a_val = a[i * k + kk];
            if a_val != 0 {
                let scaled_a = (a_val as f32) * scale;
                for j in 0..n {
                    c_scalar[i * n + j] += scaled_a * b[kk * n + j];
                }
            }
        }
    }

    for (simd, scalar) in c_simd.iter().zip(c_scalar.iter()) {
        assert_abs_diff_eq!(*simd, *scalar, epsilon = 1e-4);
    }
}

#[test]
fn test_ternary_packing_unpacking() {
    let original = vec![-1, 0, 1, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1, 1, 0, 0];
    let mut packed = vec![0u8; (original.len() + 3) / 4];
    let mut unpacked = vec![0i8; original.len()];

    // Pack the data
    vectorized_pack_ternary(&original, &mut packed).unwrap();

    // Unpack the data
    vectorized_unpack_ternary(&packed, &mut unpacked, original.len()).unwrap();

    // Should match original
    assert_eq!(original, unpacked);
}

#[test]
fn test_rle_compression() {
    let input = vec![-1; 50]; // Highly compressible
    let ops = SimdPackingOps::new();
    let mut rle_encoded = Vec::new();
    let mut decoded = vec![0i8; input.len()];

    ops.pack_ternary_rle(&input, &mut rle_encoded).unwrap();
    let decoded_len = ops.unpack_ternary_rle(&rle_encoded, &mut decoded).unwrap();

    assert_eq!(decoded_len, input.len());
    assert_eq!(&input, &decoded[..decoded_len]);

    // RLE should compress well for repeated values
    let regular_packed_size = (input.len() + 3) / 4;
    println!(
        "RLE size: {}, Regular size: {}",
        rle_encoded.len(),
        regular_packed_size
    );
    assert!(rle_encoded.len() < regular_packed_size);
}

#[test]
fn test_absmean_computation() {
    let values = vec![1.5, -2.3, 0.7, -1.1, 3.2, -0.8, 2.1, -1.7];
    let ops = SimdTernaryOps::new();

    let simd_result = ops.compute_absmean(&values);

    // Reference scalar computation
    let scalar_result = values.iter().map(|x| (*x).abs()).sum::<f32>() / values.len() as f32;

    assert_abs_diff_eq!(simd_result, scalar_result, epsilon = 1e-6);
}

#[test]
fn test_large_batch_processing() {
    let size = 10000;
    let input: Vec<f32> = (0..size)
        .map(|i| ((i as f32) - (size as f32 / 2.0)) / 1000.0)
        .collect();
    let mut output = vec![0i8; size];
    let threshold = 2.0;

    // Should not panic or produce errors
    vectorized_ternary_quantize(&input, &mut output, threshold).unwrap();

    // Verify some basic properties
    let positive_count = output.iter().filter(|&&x| x == 1).count();
    let negative_count = output.iter().filter(|&&x| x == -1).count();
    let zero_count = output.iter().filter(|&&x| x == 0).count();

    assert_eq!(positive_count + negative_count + zero_count, size);

    // Most values should be zero with this threshold
    assert!(zero_count > size / 2);
}

#[test]
fn test_edge_cases() {
    // Empty arrays
    let empty: Vec<f32> = vec![];
    let mut empty_out: Vec<i8> = vec![];
    assert!(vectorized_ternary_quantize(&empty, &mut empty_out, 1.0).is_ok());

    // Single element
    let single = vec![1.5];
    let mut single_out = vec![0i8; 1];
    vectorized_ternary_quantize(&single, &mut single_out, 1.0).unwrap();
    assert_eq!(single_out[0], 1);

    // Size mismatch should error
    let input = vec![1.0, 2.0, 3.0];
    let mut output = vec![0i8; 2];
    assert!(vectorized_ternary_quantize(&input, &mut output, 1.0).is_err());
}

#[test]
fn test_alignment_requirements() {
    let mut vec = AlignedVec::<f32>::optimal_with_capacity(1000);

    for i in 0..1000 {
        vec.push(i as f32);
    }

    assert!(vec.is_aligned());
    assert_eq!(vec.len(), 1000);

    // Should be able to use this for SIMD operations
    let mut output = vec![0i8; 1000];
    vectorized_ternary_quantize(&vec, &mut output, 100.0).unwrap();
}

#[test]
fn test_different_thresholds() {
    let input = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let thresholds = [0.1, 0.5, 1.0, 1.5];

    for &threshold in &thresholds {
        let mut output = vec![0i8; input.len()];
        vectorized_ternary_quantize(&input, &mut output, threshold).unwrap();

        // Verify quantization logic
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let expected = if inp > threshold {
                1
            } else if inp < -threshold {
                -1
            } else {
                0
            };
            assert_eq!(
                out, expected,
                "Mismatch at index {} with input {} and threshold {}: got {}, expected {}",
                i, inp, threshold, out, expected
            );
        }
    }
}

#[test]
fn test_performance_consistency() {
    let sizes = [100, 1000, 10000];

    for size in &sizes {
        let input: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();
        let mut output_simd = vec![0i8; *size];
        let mut output_scalar = vec![0i8; *size];
        let threshold = 5.0;

        // SIMD version
        let start_simd = std::time::Instant::now();
        vectorized_ternary_quantize(&input, &mut output_simd, threshold).unwrap();
        let duration_simd = start_simd.elapsed();

        // Scalar version
        let start_scalar = std::time::Instant::now();
        for (i, &val) in input.iter().enumerate() {
            output_scalar[i] = if val > threshold {
                1
            } else if val < -threshold {
                -1
            } else {
                0
            };
        }
        let duration_scalar = start_scalar.elapsed();

        // Results should be identical
        assert_eq!(output_simd, output_scalar);

        println!(
            "Size {}: SIMD {:?}, Scalar {:?}, Speedup: {:.2}x",
            size,
            duration_simd,
            duration_scalar,
            duration_scalar.as_nanos() as f64 / duration_simd.as_nanos() as f64
        );
    }
}

#[test]
fn test_simd_config() {
    let config = SimdConfig::default();
    println!("Default SIMD config: {:?}", config);

    assert!(config.memory_alignment >= 16);
    assert!(config.preferred_vector_size >= 4);

    let scalar_config = SimdConfig::scalar_only();
    assert!(!scalar_config.has_simd());
    assert_eq!(scalar_config.preferred_vector_size, 1);
}

#[test]
fn test_special_values() {
    let input = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, -0.0];
    let mut output = vec![0i8; input.len()];
    let threshold = 1.0;

    // Should handle special values gracefully (though results may be platform-specific)
    assert!(vectorized_ternary_quantize(&input, &mut output, threshold).is_ok());

    // Verify finite values
    assert_eq!(output[3], 0); // 0.0 should quantize to 0
    assert_eq!(output[4], 0); // -0.0 should quantize to 0
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_quantization_pipeline() {
        // Simulate a typical neural network weight quantization pipeline
        let original_weights: Vec<f32> = (0..1000).map(|i| ((i as f32) - 500.0) / 100.0).collect();

        // Step 1: Compute threshold (absmean)
        let ops = SimdTernaryOps::new();
        let threshold = ops.compute_absmean(&original_weights);

        // Step 2: Quantize weights
        let mut quantized_weights = vec![0i8; original_weights.len()];
        vectorized_ternary_quantize(&original_weights, &mut quantized_weights, threshold).unwrap();

        // Step 3: Pack for storage
        let mut packed_weights = vec![0u8; (quantized_weights.len() + 3) / 4];
        vectorized_pack_ternary(&quantized_weights, &mut packed_weights).unwrap();

        // Step 4: Unpack and dequantize for computation
        let mut unpacked_weights = vec![0i8; quantized_weights.len()];
        vectorized_unpack_ternary(
            &packed_weights,
            &mut unpacked_weights,
            quantized_weights.len(),
        )
        .unwrap();

        let mut dequantized_weights = vec![0.0f32; original_weights.len()];
        vectorized_ternary_dequantize(&unpacked_weights, &mut dequantized_weights, threshold)
            .unwrap();

        // Verify round-trip consistency
        assert_eq!(quantized_weights, unpacked_weights);

        // Check compression ratio
        let original_size = original_weights.len() * 4; // f32 = 4 bytes
        let compressed_size = packed_weights.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;

        println!(
            "Compression ratio: {:.2}x ({} -> {} bytes)",
            compression_ratio, original_size, compressed_size
        );
        assert!(compression_ratio >= 3.0); // Should get close to 4:1 ratio

        // Verify quantization quality
        let mut mse = 0.0f32;
        for (orig, dequant) in original_weights.iter().zip(dequantized_weights.iter()) {
            let diff = orig - dequant;
            mse += diff * diff;
        }
        mse /= original_weights.len() as f32;

        println!("Quantization MSE: {:.6}", mse);
        // MSE should be reasonable for ternary quantization
        assert!(mse < 1.0);
    }
}
