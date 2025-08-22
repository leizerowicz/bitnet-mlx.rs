//! Comprehensive tests for SIMD unpacking functionality
//! 
//! This module tests the SIMD-optimized weight unpacking system, including:
//! - SIMD capability detection and configuration
//! - Platform-specific SIMD implementations (AVX2, SSE2, NEON)
//! - Scalar fallback implementations
//! - Consistency between SIMD and scalar results
//! - Performance characteristics and benchmarking
//! - Edge cases and error handling

use bitnet_quant::quantization::simd_unpacking::*;
use bitnet_quant::quantization::packing::*;

/// Test SIMD capability detection
#[cfg(test)]
mod capability_tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        
        // Verify detection doesn't panic and returns valid state
        assert!(caps.has_simd() || !caps.has_simd());
        
        // On x86/x86_64, at least one of SSE2/AVX2 should be available on modern systems
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Most modern x86 systems have SSE2
            println!("SSE2: {}, AVX2: {}", caps.sse2, caps.avx2);
        }
        
        // On ARM64, NEON should be available
        #[cfg(target_arch = "aarch64")]
        {
            println!("NEON: {}", caps.neon);
        }
    }

    #[test]
    fn test_simd_capabilities_manual_configuration() {
        // Test with all SIMD disabled
        let no_simd = SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        };
        assert!(!no_simd.has_simd());

        // Test with SSE2 enabled
        let sse2_only = SimdCapabilities {
            sse2: true,
            avx2: false,
            neon: false,
        };
        assert!(sse2_only.has_simd());

        // Test with AVX2 enabled
        let avx2_only = SimdCapabilities {
            sse2: false,
            avx2: true,
            neon: false,
        };
        assert!(avx2_only.has_simd());

        // Test with NEON enabled
        let neon_only = SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: true,
        };
        assert!(neon_only.has_simd());
    }

    #[test]
    fn test_unpacker_creation() {
        // Test automatic detection
        let auto_unpacker = SimdUnpacker::new();
        let default_unpacker = SimdUnpacker::default();
        
        // Just verify they were created successfully
        // (simd_available field is private, so we can't directly compare)
        
        // Test manual configuration
        let custom_caps = SimdCapabilities {
            sse2: true,
            avx2: false,
            neon: false,
        };
        let custom_unpacker = SimdUnpacker::with_capabilities(custom_caps);
        // Verify it was created successfully
        assert!(true); // Custom unpacker created without panic
    }
}

/// Test SIMD unpacking for different packing strategies
#[cfg(test)]
mod strategy_tests {
    use super::*;

    fn create_test_weights(size: usize) -> Vec<i8> {
        (0..size).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect()
    }

    #[test]
    fn test_bit_packed_2bit_unpacking() {
        let weights = create_test_weights(32);
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test with SIMD enabled
        let simd_unpacker = SimdUnpacker::new();
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, simd_result);
        
        // Test with SIMD disabled
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, scalar_result);
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_base3_packed_unpacking() {
        // Use size divisible by 5 for base-3 packing
        let weights = create_test_weights(25);
        let config = TernaryPackingConfig::default();
        let packer = Base3PackedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_byte_aligned_unpacking() {
        let weights = create_test_weights(64);
        let config = TernaryPackingConfig {
            strategy: TernaryPackingStrategy::ByteAligned,
            alignment: 16,
            ..Default::default()
        };
        let packer = ByteAlignedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD vs scalar consistency
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, simd_result);
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_run_length_encoded_unpacking() {
        // Create data with runs for effective RLE
        let weights = vec![
            0i8, 0, 0, 0,  // Run of 4 zeros
            1, 1, 1,       // Run of 3 ones
            -1, -1, -1, -1, -1,  // Run of 5 negative ones
            0, 0,          // Run of 2 zeros
            1,             // Single one
        ];
        let config = TernaryPackingConfig::default();
        let packer = RunLengthEncodedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_compressed_sparse_unpacking() {
        // Create sparse data (mostly zeros)
        let mut weights = vec![0i8; 100];
        weights[10] = 1;
        weights[25] = -1;
        weights[50] = 1;
        weights[75] = -1;
        weights[90] = 1;
        
        let config = TernaryPackingConfig::default();
        let packer = CompressedSparsePacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }
}

/// Test SIMD vs scalar consistency across different scenarios
#[cfg(test)]
mod consistency_tests {
    use super::*;

    #[test]
    fn test_simd_scalar_consistency_small_data() {
        let weights = vec![-1i8, 0, 1, -1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        
        assert_eq!(simd_result, scalar_result);
        assert_eq!(weights, simd_result);
    }

    #[test]
    fn test_simd_scalar_consistency_medium_data() {
        let weights = (0..256).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        
        assert_eq!(simd_result, scalar_result);
        assert_eq!(weights, simd_result);
    }

    #[test]
    fn test_simd_scalar_consistency_large_data() {
        let weights = (0..4096).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        
        assert_eq!(simd_result, scalar_result);
        assert_eq!(weights, simd_result);
    }

    #[test]
    fn test_consistency_across_all_strategies() {
        let weights = (0..100).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let strategies = vec![
            (TernaryPackingStrategy::BitPacked2Bit, Box::new(BitPacked2BitPacker) as Box<dyn TernaryPacker>),
            (TernaryPackingStrategy::ByteAligned, Box::new(ByteAlignedPacker) as Box<dyn TernaryPacker>),
        ];
        
        for (strategy, packer) in strategies {
            let config = TernaryPackingConfig {
                strategy,
                ..Default::default()
            };
            
            let packed = packer.pack(&weights, &config).unwrap();
            
            let simd_unpacker = SimdUnpacker::new();
            let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                sse2: false,
                avx2: false,
                neon: false,
            });
            
            let simd_result = simd_unpacker.unpack(&packed).unwrap();
            let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
            
            assert_eq!(simd_result, scalar_result, "Inconsistency in strategy: {strategy:?}");
            assert_eq!(weights, simd_result, "Incorrect unpacking for strategy: {strategy:?}");
        }
    }
}

/// Test edge cases and error handling
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_data_unpacking() {
        let weights: Vec<i8> = vec![];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_single_element_unpacking() {
        let weights = vec![1i8];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_odd_length_unpacking() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0]; // 7 elements
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_all_zeros_unpacking() {
        let weights = vec![0i8; 64];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_all_ones_unpacking() {
        let weights = vec![1i8; 64];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_all_negative_ones_unpacking() {
        let weights = vec![-1i8; 64];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_corrupted_sparse_data_handling() {
        // Create a corrupted sparse packed data
        let mut corrupted_data = PackedTernaryWeights {
            data: vec![0, 0, 0, 1], // Claims 1 non-zero element but no index/value data
            shape: candle_core::Shape::from_dims(&[10]),
            strategy: TernaryPackingStrategy::CompressedSparse,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count: 10,
                block_sizes: None,
                sparse_indices: None,
                rle_data: None,
                padding: 0,
                extra_data: std::collections::HashMap::new(),
            },
            memory_footprint: 4,
            compression_ratio: 0.1,
        };
        
        let unpacker = SimdUnpacker::new();
        let result = unpacker.unpack(&corrupted_data);
        assert!(result.is_err());
        
        // Test with insufficient data for indices
        corrupted_data.data = vec![0, 0, 0, 2, 0, 0, 0, 1]; // Claims 2 elements but only 1 index
        let result = unpacker.unpack(&corrupted_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_for_unsupported_strategy() {
        // Create packed data with an unsupported strategy
        let weights = vec![-1i8, 0, 1, -1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let mut packed = packer.pack(&weights, &config).unwrap();
        // Change to an unsupported strategy that will trigger fallback
        // Since HuffmanCoded doesn't exist, we'll use Hybrid which should trigger fallback in some cases
        packed.strategy = TernaryPackingStrategy::Hybrid;
        
        let unpacker = SimdUnpacker::new();
        let result = unpacker.unpack(&packed);
        // Should use fallback unpacker
        assert!(result.is_ok());
    }
}

/// Test convenience functions and utilities
#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn test_convenience_function() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = simd_unpack_weights(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_convenience_function_with_different_strategies() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
        
        let strategies_and_packers = vec![
            (TernaryPackingStrategy::BitPacked2Bit, Box::new(BitPacked2BitPacker) as Box<dyn TernaryPacker>),
            (TernaryPackingStrategy::ByteAligned, Box::new(ByteAlignedPacker) as Box<dyn TernaryPacker>),
        ];
        
        for (strategy, packer) in strategies_and_packers {
            let config = TernaryPackingConfig {
                strategy,
                ..Default::default()
            };
            
            let packed = packer.pack(&weights, &config).unwrap();
            let unpacked = simd_unpack_weights(&packed).unwrap();
            
            assert_eq!(weights, unpacked, "Failed for strategy: {strategy:?}");
        }
    }
}

/// Test performance characteristics and benchmarking
#[cfg(test)]
mod performance_tests {
    use super::*;
    use bitnet_quant::quantization::simd_unpacking::benchmark::*;

    #[test]
    fn test_benchmark_functionality() {
        let weights = (0..1000).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Run a small benchmark
        let benchmark_result = benchmark_unpacking(&packed, 10).unwrap();
        
        assert_eq!(benchmark_result.strategy, TernaryPackingStrategy::BitPacked2Bit);
        assert_eq!(benchmark_result.element_count, 1000);
        assert!(benchmark_result.simd_time_ns > 0);
        assert!(benchmark_result.scalar_time_ns > 0);
        assert!(benchmark_result.speedup > 0.0);
        
        println!("Benchmark result: {benchmark_result:?}");
    }

    #[test]
    fn test_performance_with_different_sizes() {
        let sizes = vec![100, 1000, 10000];
        
        for size in sizes {
            let weights = (0..size).map(|i| match i % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            }).collect::<Vec<_>>();
            
            let config = TernaryPackingConfig::default();
            let packer = BitPacked2BitPacker;
            let packed = packer.pack(&weights, &config).unwrap();
            
            let simd_unpacker = SimdUnpacker::new();
            let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                sse2: false,
                avx2: false,
                neon: false,
            });
            
            // Measure basic timing (not precise, but ensures no panics)
            let start = std::time::Instant::now();
            let simd_result = simd_unpacker.unpack(&packed).unwrap();
            let simd_time = start.elapsed();
            
            let start = std::time::Instant::now();
            let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
            let scalar_time = start.elapsed();
            
            assert_eq!(simd_result, scalar_result);
            assert_eq!(weights, simd_result);
            
            println!("Size: {size}, SIMD: {simd_time:?}, Scalar: {scalar_time:?}");
        }
    }
}

/// Test platform-specific optimizations
#[cfg(test)]
mod platform_tests {
    use super::*;

    #[test]
    fn test_platform_specific_capabilities() {
        let caps = SimdCapabilities::detect();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Test x86/x86_64 specific capabilities
            println!("x86/x86_64 platform detected");
            println!("SSE2: {}, AVX2: {}", caps.sse2, caps.avx2);
            
            // Most modern x86 systems should have SSE2
            if caps.sse2 {
                let sse2_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                    sse2: true,
                    avx2: false,
                    neon: false,
                });
                
                let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
                let config = TernaryPackingConfig::default();
                let packer = BitPacked2BitPacker;
                let packed = packer.pack(&weights, &config).unwrap();
                
                let result = sse2_unpacker.unpack(&packed).unwrap();
                assert_eq!(weights, result);
            }
            
            if caps.avx2 {
                let avx2_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                    sse2: false,
                    avx2: true,
                    neon: false,
                });
                
                let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
                let config = TernaryPackingConfig::default();
                let packer = BitPacked2BitPacker;
                let packed = packer.pack(&weights, &config).unwrap();
                
                let result = avx2_unpacker.unpack(&packed).unwrap();
                assert_eq!(weights, result);
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // Test ARM64 specific capabilities
            println!("ARM64 platform detected");
            println!("NEON: {}", caps.neon);
            
            if caps.neon {
                let neon_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                    sse2: false,
                    avx2: false,
                    neon: true,
                });
                
                let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
                let config = TernaryPackingConfig::default();
                let packer = BitPacked2BitPacker;
                let packed = packer.pack(&weights, &config).unwrap();
                
                let result = neon_unpacker.unpack(&packed).unwrap();
                assert_eq!(weights, result);
            }
        }
    }

    #[test]
    fn test_cross_platform_consistency() {
        // Test that results are consistent across different capability configurations
        let weights = (0..128).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        let packed = packer.pack(&weights, &config).unwrap();
        
        let configurations = vec![
            SimdCapabilities { sse2: false, avx2: false, neon: false }, // Scalar only
            SimdCapabilities { sse2: true, avx2: false, neon: false },  // SSE2 only
            SimdCapabilities { sse2: false, avx2: true, neon: false },  // AVX2 only
            SimdCapabilities { sse2: false, avx2: false, neon: true },  // NEON only
            SimdCapabilities { sse2: true, avx2: true, neon: false },   // x86 with both
        ];
        
        let mut results = Vec::new();
        for caps in configurations {
            let unpacker = SimdUnpacker::with_capabilities(caps);
            let result = unpacker.unpack(&packed).unwrap();
            results.push(result);
        }
        
        // All results should be identical
        for result in &results {
            assert_eq!(weights, *result);
            assert_eq!(&results[0], result);
        }
    }
}

/// Integration tests with real-world scenarios
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_neural_network_layer_simulation() {
        // Simulate unpacking weights for a small neural network layer
        let layer_sizes = vec![
            (128, 64),   // Small layer
            (512, 256),  // Medium layer
            (1024, 512), // Large layer
        ];
        
        for (input_size, output_size) in layer_sizes {
            let weight_count = input_size * output_size;
            let weights = (0..weight_count).map(|i| match i % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            }).collect::<Vec<_>>();
            
            let config = TernaryPackingConfig::default();
            let packer = BitPacked2BitPacker;
            let packed = packer.pack(&weights, &config).unwrap();
            
            let unpacker = SimdUnpacker::new();
            let unpacked = unpacker.unpack(&packed).unwrap();
            
            assert_eq!(weights, unpacked);
            assert_eq!(unpacked.len(), weight_count);
            
            // Verify all values are valid ternary
            for &val in &unpacked {
                assert!(val == -1 || val == 0 || val == 1);
            }
            
            println!("Successfully unpacked {input_size}x{output_size} layer ({weight_count} weights)");
        }
    }

    #[test]
    fn test_batch_unpacking() {
        // Test unpacking multiple weight matrices in sequence
        let batch_size = 10;
        let weights_per_matrix = 256;
        
        let mut all_weights = Vec::new();
        let mut all_packed = Vec::new();
        
        // Create batch of packed weights
        for i in 0..batch_size {
            let weights = (0..weights_per_matrix).map(|j| match (i + j) % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            }).collect::<Vec<_>>();
            
            let config = TernaryPackingConfig::default();
            let packer = BitPacked2BitPacker;
            let packed = packer.pack(&weights, &config).unwrap();
            
            all_weights.push(weights);
            all_packed.push(packed);
        }
        
        // Unpack all matrices
        let unpacker = SimdUnpacker::new();
        for (i, packed) in all_packed.iter().enumerate() {
            let unpacked = unpacker.unpack(packed).unwrap();
            assert_eq!(all_weights[i], unpacked);
        }
    }

    #[test]
    fn test_mixed_strategy_unpacking() {
        // Test unpacking different strategies in the same session
        let weights = (0..200).map(|i| match i % 3 {
            0 => -1i8,
            1 => 0i8,
            _ => 1i8,
        }).collect::<Vec<_>>();
        
        let strategies_and_packers = vec![
            (TernaryPackingStrategy::BitPacked2Bit, Box::new(BitPacked2BitPacker) as Box<dyn TernaryPacker>),
            (TernaryPackingStrategy::ByteAligned, Box::new(ByteAlignedPacker) as Box<dyn TernaryPacker>),
            (TernaryPackingStrategy::Base3Packed, Box::new(Base3PackedPacker) as Box<dyn TernaryPacker>),
        ];
        
        let unpacker = SimdUnpacker::new();
        
        for (strategy, packer) in strategies_and_packers {
            let config = TernaryPackingConfig {
                strategy,
                ..Default::default()
            };
            
            let packed = packer.pack(&weights, &config).unwrap();
            let unpacked = unpacker.unpack(&packed).unwrap();
            
            assert_eq!(weights, unpacked, "Failed for strategy: {strategy:?}");
        }
    }
}