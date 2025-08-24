//! Enhanced packing correctness tests
//!
//! This module provides additional tests for advanced packing features including
//! corruption detection, integrity verification, validation, and edge cases.

use approx::assert_abs_diff_eq;
use bitnet_quant::quantization::packing::{
    packing_utils, Base3PackedPacker, BitPacked2BitPacker, ByteAlignedPacker,
    CompressedSparsePacker, HybridPacker, RunLengthEncodedPacker, TernaryPacker,
    TernaryPackerFactory, TernaryPackingConfig, TernaryPackingStrategy,
};
use candle_core::{Device, Shape, Tensor};

/// Test helper to create test weights with specific patterns
fn create_test_weights(pattern: &str, size: usize) -> Vec<i8> {
    match pattern {
        "alternating" => (0..size).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect(),
        "sparse" => {
            let mut weights = vec![0i8; size];
            for i in (0..size).step_by(10) {
                weights[i] = if i % 20 == 0 { 1 } else { -1 };
            }
            weights
        }
        "runs" => {
            let mut weights = Vec::new();
            let mut current = -1i8;
            for i in 0..size {
                if i % 5 == 0 {
                    current = -current;
                }
                weights.push(current);
            }
            weights
        }
        "random_ternary" => (0..size)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect(),
        _ => vec![0i8; size],
    }
}

#[test]
fn test_packer_validation_with_valid_weights() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;

    // Test pack_with_validation
    let packed = packer.pack_with_validation(&weights, &config).unwrap();
    assert_eq!(packed.strategy, TernaryPackingStrategy::BitPacked2Bit);

    // Test unpack_with_validation
    let unpacked = packer.unpack_with_validation(&packed).unwrap();
    assert_eq!(weights, unpacked);
}

#[test]
fn test_packer_validation_with_invalid_weights() {
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;

    // Test with empty weights
    let empty_weights: Vec<i8> = vec![];
    let result = packer.pack_with_validation(&empty_weights, &config);
    assert!(result.is_err());

    // Test with invalid ternary values
    let invalid_weights = vec![-2i8, 0, 1, 3, 0, 1]; // -2 and 3 are invalid
    let result = packer.pack_with_validation(&invalid_weights, &config);
    assert!(result.is_err());
}

#[test]
fn test_integrity_data_verification() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let mut config = TernaryPackingConfig::default();
    config.enable_compression = true; // Enable integrity data

    let packer = BitPacked2BitPacker;

    // Pack with integrity data
    let packed = packer.pack_with_validation(&weights, &config).unwrap();

    // Verify integrity data was added
    assert!(packed.metadata.extra_data.contains_key("crc32"));
    assert!(packed.metadata.extra_data.contains_key("timestamp"));

    // Unpack with integrity verification
    let unpacked = packer.unpack_with_validation(&packed).unwrap();
    assert_eq!(weights, unpacked);
}

#[test]
fn test_integrity_data_corruption_detection() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let mut config = TernaryPackingConfig::default();
    config.enable_compression = true;

    let packer = BitPacked2BitPacker;
    let mut packed = packer.pack_with_validation(&weights, &config).unwrap();

    // Corrupt the data
    if !packed.data.is_empty() {
        packed.data[0] = packed.data[0].wrapping_add(1);
    }

    // Should detect corruption
    let result = packer.unpack_with_validation(&packed);
    assert!(result.is_err());
}

#[test]
fn test_packed_data_validation() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;

    let mut packed = packer.pack(&weights, &config).unwrap();

    // Test valid packed data
    assert!(packer.validate_packed_data(&packed).is_ok());

    // Test invalid packed data (empty data with non-zero element count)
    packed.data.clear();
    assert!(packer.validate_packed_data(&packed).is_err());
}

#[test]
fn test_unpacked_weights_validation() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;

    let packed = packer.pack(&weights, &config).unwrap();

    // Test valid unpacked weights
    assert!(packer.validate_unpacked_weights(&weights, &packed).is_ok());

    // Test invalid unpacked weights (wrong length)
    let wrong_length_weights = vec![-1i8, 0, 1];
    assert!(packer
        .validate_unpacked_weights(&wrong_length_weights, &packed)
        .is_err());

    // Test invalid unpacked weights (invalid values)
    let invalid_weights = vec![-2i8, 0, 1, 3, 0, 1];
    assert!(packer
        .validate_unpacked_weights(&invalid_weights, &packed)
        .is_err());
}

#[test]
fn test_base3_encoding_correctness() {
    let packer = Base3PackedPacker;
    let config = TernaryPackingConfig::default();

    // Test specific base-3 encoding patterns
    let test_cases = vec![
        (vec![1i8, 0, -1, 1, 0], 140u8), // Expected encoding from existing test
        (vec![-1i8, -1, -1, -1, -1], 0u8), // All -1s -> all 0s in base-3
        (vec![1i8, 1, 1, 1, 1], 242u8),  // All 1s -> all 2s in base-3: 2+6+18+54+162=242
        (vec![0i8, 0, 0, 0, 0], 121u8),  // All 0s -> all 1s in base-3: 1+3+9+27+81=121
    ];

    for (weights, expected_byte) in test_cases {
        let packed = packer.pack(&weights, &config).unwrap();
        assert_eq!(packed.data.len(), 1);
        assert_eq!(packed.data[0], expected_byte);

        let unpacked = packer.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }
}

#[test]
fn test_run_length_encoding_edge_cases() {
    let packer = RunLengthEncodedPacker;
    let config = TernaryPackingConfig::default();

    // Test with maximum run length (255)
    let long_run = vec![1i8; 255];
    let packed = packer.pack(&long_run, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(long_run, unpacked);

    // Test with run length exceeding 255 (should split into multiple runs)
    let very_long_run = vec![1i8; 300];
    let packed = packer.pack(&very_long_run, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(very_long_run, unpacked);

    // Test alternating pattern (worst case for RLE)
    let alternating = create_test_weights("alternating", 20);
    let packed = packer.pack(&alternating, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(alternating, unpacked);

    // Compression ratio should be poor for alternating pattern
    assert!(packed.compression_ratio < 1.0);
}

#[test]
fn test_compressed_sparse_edge_cases() {
    let packer = CompressedSparsePacker;
    let config = TernaryPackingConfig::default();

    // Test with all zeros (completely sparse)
    let all_zeros = vec![0i8; 100];
    let packed = packer.pack(&all_zeros, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(all_zeros, unpacked);

    // Should be very efficient
    assert!(packed.compression_ratio > 20.0);

    // Test with single non-zero element
    let mut single_nonzero = vec![0i8; 100];
    single_nonzero[50] = 1;
    let packed = packer.pack(&single_nonzero, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(single_nonzero, unpacked);

    // Test with all non-zero elements (worst case for sparse)
    let all_nonzero = vec![1i8; 50];
    let packed = packer.pack(&all_nonzero, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(all_nonzero, unpacked);

    // Should be inefficient
    assert!(packed.compression_ratio < 1.0);
}

#[test]
fn test_byte_aligned_packer_alignment() {
    let packer = ByteAlignedPacker;

    // Test different alignment values
    let alignments = vec![4, 8, 16, 32];
    let weights = vec![-1i8, 0, 1, -1, 0];

    for alignment in alignments {
        let config = TernaryPackingConfig {
            alignment,
            simd_optimized: true,
            ..Default::default()
        };

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);

        // Data should be aligned to the specified boundary
        assert_eq!(packed.data.len() % alignment, 0);

        // Padding should be calculated correctly
        let expected_padding = if weights.len() % alignment == 0 {
            0
        } else {
            alignment - (weights.len() % alignment)
        };
        assert_eq!(packed.metadata.padding, expected_padding);
    }
}

#[test]
fn test_hybrid_packer_strategy_selection() {
    let packer = HybridPacker;

    // Test with mixed patterns that should trigger different strategies
    let mut mixed_weights = Vec::new();

    // Dense block (should use BitPacked or Base3)
    mixed_weights.extend(create_test_weights("alternating", 32));

    // Sparse block (should use sparse format)
    mixed_weights.extend(vec![0i8; 64]);
    mixed_weights[80] = 1; // Single non-zero in sparse block

    // Run-length block (should use RLE)
    mixed_weights.extend(vec![1i8; 20]);
    mixed_weights.extend(vec![-1i8; 20]);

    let config = TernaryPackingConfig {
        block_size: Some(32),
        ..Default::default()
    };

    let packed = packer.pack(&mixed_weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();

    assert_eq!(mixed_weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::Hybrid);

    // Should achieve reasonable compression
    assert!(packed.compression_ratio > 1.0);
}

#[test]
fn test_packer_suitability_analysis() {
    let config = TernaryPackingConfig {
        sparsity_threshold: 0.7,
        simd_optimized: true,
        ..Default::default()
    };

    // Dense weights
    let dense_weights = create_test_weights("alternating", 20);
    assert!(BitPacked2BitPacker.is_suitable(&dense_weights, &config));
    assert!(Base3PackedPacker.is_suitable(&dense_weights, &config));
    assert!(!RunLengthEncodedPacker.is_suitable(&dense_weights, &config));
    assert!(!CompressedSparsePacker.is_suitable(&dense_weights, &config));
    assert!(ByteAlignedPacker.is_suitable(&dense_weights, &config)); // SIMD enabled

    // Sparse weights (80% zeros)
    let sparse_weights = create_test_weights("sparse", 100);
    assert!(RunLengthEncodedPacker.is_suitable(&sparse_weights, &config));
    assert!(CompressedSparsePacker.is_suitable(&sparse_weights, &config));

    // Very sparse weights (90% zeros)
    let very_sparse = vec![0i8; 100];
    assert!(CompressedSparsePacker.is_suitable(&very_sparse, &config));

    // Test with SIMD disabled
    let config_no_simd = TernaryPackingConfig {
        simd_optimized: false,
        ..config
    };
    assert!(!ByteAlignedPacker.is_suitable(&dense_weights, &config_no_simd));
}

#[test]
fn test_compression_ratio_accuracy() {
    let weights = create_test_weights("random_ternary", 100);
    let config = TernaryPackingConfig::default();

    let strategies = vec![
        TernaryPackingStrategy::Uncompressed,
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::RunLengthEncoded,
        TernaryPackingStrategy::CompressedSparse,
    ];

    for strategy in strategies {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let estimate = packer.estimate_savings(&weights, &config);
        let packed = packer.pack(&weights, &config).unwrap();

        // Estimate should be reasonably close to actual
        let actual_ratio = weights.len() as f32 / packed.data.len() as f32;
        let estimate_error = (estimate.compression_ratio - actual_ratio).abs();

        // Allow some tolerance for estimation error
        assert!(
            estimate_error < 0.5,
            "Strategy {:?}: estimate {:.2} vs actual {:.2}, error {:.2}",
            strategy,
            estimate.compression_ratio,
            actual_ratio,
            estimate_error
        );

        // Verify other estimate fields
        assert_eq!(estimate.original_size_bytes, weights.len());
        assert!(estimate.packed_size_bytes > 0);
        assert_eq!(
            estimate.memory_saved_bytes,
            estimate
                .original_size_bytes
                .saturating_sub(estimate.packed_size_bytes)
        );

        if estimate.original_size_bytes > 0 {
            let expected_percentage =
                (estimate.memory_saved_bytes as f32 / estimate.original_size_bytes as f32) * 100.0;
            assert_abs_diff_eq!(
                estimate.savings_percentage,
                expected_percentage,
                epsilon = 0.1
            );
        }
    }
}

#[test]
fn test_packer_factory_optimal_selection() {
    let test_cases = vec![
        ("dense", create_test_weights("alternating", 20)),
        ("sparse", create_test_weights("sparse", 100)),
        ("runs", create_test_weights("runs", 50)),
        ("random", create_test_weights("random_ternary", 75)),
    ];

    let config = TernaryPackingConfig::default();

    for (name, weights) in test_cases {
        // Test auto-selection
        let selected_strategy = TernaryPackerFactory::auto_select_strategy(&weights, &config);

        // Test optimal packing
        let optimal_packed = TernaryPackerFactory::pack_optimal(&weights, &config).unwrap();
        assert_eq!(optimal_packed.strategy, selected_strategy);

        // Verify round-trip correctness
        let packer = TernaryPackerFactory::create_packer(selected_strategy);
        let unpacked = packer.unpack(&optimal_packed).unwrap();
        assert_eq!(weights, unpacked, "Failed round-trip for pattern: {name}");

        // Verify compression is reasonable
        assert!(optimal_packed.compression_ratio > 0.0);
    }
}

#[test]
fn test_memory_footprint_calculations() {
    let weights = create_test_weights("random_ternary", 64);
    let config = TernaryPackingConfig::default();

    let strategies = vec![
        TernaryPackingStrategy::Uncompressed,
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::ByteAligned,
    ];

    for strategy in strategies {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let packed = packer.pack(&weights, &config).unwrap();

        // Memory footprint should match data length
        assert_eq!(packed.memory_footprint, packed.data.len());

        // Compression ratio should be consistent with memory footprint
        let calculated_ratio = weights.len() as f32 / packed.memory_footprint as f32;
        assert_abs_diff_eq!(packed.compression_ratio, calculated_ratio, epsilon = 0.01);
    }
}

#[test]
fn test_packing_metadata_completeness() {
    let weights = create_test_weights("random_ternary", 50);
    let config = TernaryPackingConfig::default();

    let strategies = vec![
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::RunLengthEncoded,
        TernaryPackingStrategy::CompressedSparse,
    ];

    for strategy in strategies {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let packed = packer.pack(&weights, &config).unwrap();

        // Basic metadata should be present
        assert_eq!(packed.metadata.element_count, weights.len());

        // Strategy-specific metadata
        match strategy {
            TernaryPackingStrategy::BitPacked2Bit | TernaryPackingStrategy::Base3Packed => {
                // Should have padding information if needed
                if weights.len() % 4 != 0 || weights.len() % 5 != 0 {
                    assert!(packed.metadata.padding >= 0);
                }
            }
            TernaryPackingStrategy::RunLengthEncoded => {
                // Should have RLE data
                assert!(packed.metadata.rle_data.is_some());
            }
            TernaryPackingStrategy::CompressedSparse => {
                // Should have sparse indices
                assert!(packed.metadata.sparse_indices.is_some());
            }
            _ => {}
        }
    }
}

#[test]
fn test_large_weight_arrays() {
    let large_sizes = vec![1000, 5000, 10000];
    let config = TernaryPackingConfig::default();

    for size in large_sizes {
        let weights = create_test_weights("random_ternary", size);

        // Test with different strategies
        let strategies = vec![
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::Hybrid,
        ];

        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            let packed = packer.pack(&weights, &config).unwrap();
            let unpacked = packer.unpack(&packed).unwrap();

            assert_eq!(
                weights, unpacked,
                "Failed for strategy {strategy:?} with size {size}"
            );
            assert!(packed.compression_ratio > 0.0);
        }
    }
}

#[test]
fn test_concurrent_packing_safety() {
    use std::sync::Arc;
    use std::thread;

    let weights = Arc::new(create_test_weights("random_ternary", 100));
    let config = Arc::new(TernaryPackingConfig::default());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let weights = Arc::clone(&weights);
            let config = Arc::clone(&config);

            thread::spawn(move || {
                let strategy = match i % 3 {
                    0 => TernaryPackingStrategy::BitPacked2Bit,
                    1 => TernaryPackingStrategy::Base3Packed,
                    _ => TernaryPackingStrategy::RunLengthEncoded,
                };

                let packer = TernaryPackerFactory::create_packer(strategy);
                let packed = packer.pack(&weights, &config).unwrap();
                let unpacked = packer.unpack(&packed).unwrap();

                assert_eq!(*weights, unpacked);
                packed.compression_ratio
            })
        })
        .collect();

    // All threads should complete successfully
    for handle in handles {
        let ratio = handle.join().unwrap();
        assert!(ratio > 0.0);
    }
}

#[test]
fn test_error_recovery_and_graceful_handling() {
    let config = TernaryPackingConfig::default();

    // Test various error conditions
    let error_cases = vec![
        ("empty_weights", vec![]),
        ("invalid_values", vec![-2i8, 0, 1, 3]),
    ];

    for (name, weights) in error_cases {
        let packer = BitPacked2BitPacker;

        // Should handle errors gracefully
        let result = packer.pack_with_validation(&weights, &config);
        match name {
            "empty_weights" | "invalid_values" => {
                assert!(result.is_err(), "Expected error for case: {name}");
            }
            _ => {}
        }
    }
}

#[test]
fn test_tensor_conversion_utilities() {
    let device = Device::Cpu;

    // Test tensor to ternary conversion
    let tensor_data = vec![1.5f32, -0.8, 0.2, -2.1, 0.0, 0.9];
    let shape = Shape::from_dims(&[2, 3]);
    let tensor = Tensor::from_vec(tensor_data, shape.clone(), &device).unwrap();

    let ternary = packing_utils::tensor_to_ternary(&tensor).unwrap();

    // Verify conversion logic
    assert_eq!(ternary.len(), 6);
    assert_eq!(ternary[0], 1); // 1.5 > 0.5
    assert_eq!(ternary[1], -1); // -0.8 < -0.5
    assert_eq!(ternary[2], 0); // 0.2 in [-0.5, 0.5]
    assert_eq!(ternary[3], -1); // -2.1 < -0.5
    assert_eq!(ternary[4], 0); // 0.0 in [-0.5, 0.5]
    assert_eq!(ternary[5], 1); // 0.9 > 0.5

    // Test ternary to tensor conversion
    let reconstructed = packing_utils::ternary_to_tensor(&ternary, &shape, &device).unwrap();
    assert_eq!(reconstructed.shape(), &shape);

    let reconstructed_data = reconstructed.to_vec1::<f32>().unwrap();
    assert_eq!(reconstructed_data, vec![1.0f32, -1.0, 0.0, -1.0, 0.0, 1.0]);
}

#[test]
fn test_sparsity_analysis_comprehensive() {
    let test_cases = vec![
        ("all_zeros", vec![0i8; 10], 1.0, 0.0),
        ("all_ones", vec![1i8; 10], 0.0, 1.0),
        ("all_neg_ones", vec![-1i8; 10], 0.0, 1.0),
        ("balanced", vec![-1i8, 1, -1, 1, -1, 1], 0.0, 0.0),
        ("mixed", vec![0i8, 0, 1, 0, -1, 0, 0, 1], 0.625, 0.125),
    ];

    for (name, weights, expected_sparsity, expected_balance) in test_cases {
        let analysis = packing_utils::analyze_sparsity(&weights);

        assert_eq!(
            analysis.total_elements,
            weights.len(),
            "Total elements mismatch for {name}"
        );
        assert_abs_diff_eq!(analysis.sparsity_ratio, expected_sparsity, epsilon = 0.001);
        assert_abs_diff_eq!(analysis.balance_ratio, expected_balance, epsilon = 0.001);

        // Verify counts add up
        assert_eq!(
            analysis.zero_count + analysis.positive_count + analysis.negative_count,
            analysis.total_elements,
            "Count mismatch for {name}"
        );
    }
}

#[test]
fn test_strategy_recommendation_logic() {
    let test_cases = vec![
        (
            "very_sparse",
            vec![0i8; 100],
            TernaryPackingStrategy::CompressedSparse,
        ),
        (
            "moderately_sparse",
            create_test_weights("sparse", 50),
            TernaryPackingStrategy::RunLengthEncoded,
        ),
        (
            "base3_optimal",
            vec![-1i8, 0, 1, -1, 0],
            TernaryPackingStrategy::Base3Packed,
        ),
        (
            "dense_general",
            create_test_weights("alternating", 20),
            TernaryPackingStrategy::BitPacked2Bit,
        ),
    ];

    for (name, mut weights, expected_strategy) in test_cases {
        // Adjust sparsity for moderately sparse case
        if name == "moderately_sparse" {
            // Ensure it's between 60-80% sparse
            for i in (0..weights.len()).step_by(3) {
                weights[i] = 0;
            }
        }

        let recommended = packing_utils::recommend_strategy(&weights);
        assert_eq!(
            recommended, expected_strategy,
            "Strategy recommendation mismatch for {name}"
        );
    }
}
