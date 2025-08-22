//! Comprehensive edge case and error handling tests
//! 
//! This module tests various edge cases and error conditions across the entire
//! quantization system, including:
//! - Boundary value testing (min/max values, zero, infinity, NaN)
//! - Invalid input validation and error propagation
//! - Memory pressure and resource exhaustion scenarios
//! - Configuration validation and error handling
//! - Packing/unpacking edge cases
//! - Corruption detection scenarios

use bitnet_quant::quantization::*;
use bitnet_quant::quantization::{
    weights::*, activations::*, mixed_precision::*,
    packing::*,
};
use candle_core::{Tensor, Device, DType, Shape};
use std::time::Duration;

/// Test boundary values and extreme inputs
#[cfg(test)]
mod boundary_value_tests {
    use super::*;

    #[test]
    fn test_zero_sized_tensors() {
        let device = Device::Cpu;
        
        // Test zero-sized weight tensor
        let empty_weights = Tensor::zeros((0,), DType::F32, &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&empty_weights, &device);
        // Should handle gracefully - either succeed with empty result or fail with appropriate error
        match result {
            Ok(quantized) => {
                assert_eq!(quantized.values.elem_count(), 0);
            }
            Err(e) => {
                // Should be a validation error, not a panic
                assert!(matches!(e, QuantizationError::ValidationFailed(_)));
            }
        }
    }

    #[test]
    fn test_single_element_tensors() {
        let device = Device::Cpu;
        
        // Test single element weight tensor
        let single_weight = Tensor::from_slice(&[0.5f32], (1,), &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&single_weight, &device);
        assert!(result.is_ok());
        
        let quantized = result.unwrap();
        assert_eq!(quantized.values.elem_count(), 1);
    }

    #[test]
    fn test_extreme_tensor_values() {
        let device = Device::Cpu;
        
        // Test with extreme values
        let extreme_values = vec![
            f32::MAX,
            f32::MIN,
            0.0,
            -0.0,
            f32::EPSILON,
            -f32::EPSILON,
        ];
        
        let weights = Tensor::from_slice(&extreme_values, (extreme_values.len(),), &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&weights, &device);
        // Should handle extreme values gracefully
        match result {
            Ok(quantized) => {
                // Verify quantized values are valid
                let data = quantized.values.to_vec1::<f32>().unwrap();
                for &val in &data {
                    if val.is_finite() {
                        assert!((-1.0..=1.0).contains(&val));
                    }
                }
            }
            Err(_) => {
                // Acceptable to fail with extreme values
            }
        }
    }

    #[test]
    fn test_very_large_tensors() {
        let device = Device::Cpu;
        
        // Test with a reasonably large tensor (but not memory-exhausting)
        let size = 10_000;
        let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) * 2.0 - 1.0).collect();
        let weights = Tensor::from_slice(&data, (size,), &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&weights, &device);
        assert!(result.is_ok());
        
        let quantized = result.unwrap();
        assert_eq!(quantized.values.elem_count(), size);
    }

    #[test]
    fn test_unusual_tensor_shapes() {
        let device = Device::Cpu;
        
        // Test with unusual but valid shapes
        let shapes = vec![
            (1, 1000), // Very wide
            (1000, 1), // Very tall
            (100, 100), // Square
        ];
        
        for shape in shapes {
            let total_elements: usize = shape.0 * shape.1;
            
            let data: Vec<f32> = (0..total_elements).map(|i| (i % 3) as f32 - 1.0).collect();
            let weights = Tensor::from_slice(&data, shape, &device).unwrap();
            let config = WeightQuantizationConfig::default();
            
            let result = absmean_quantize_weights(&weights, &device);
            assert!(result.is_ok(), "Failed for shape {shape:?}");
        }
    }

    #[test]
    fn test_activation_boundary_values() {
        let device = Device::Cpu;
        
        // Test activation quantization with boundary values
        let boundary_values = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let activations = Tensor::from_slice(&boundary_values, (boundary_values.len(),), &device).unwrap();
        let config = ActivationQuantizationConfig::default();
        
        let result = absmax_quantize_activations(&activations, &device, None);
        assert!(result.is_ok());
        
        let quantized = result.unwrap();
        assert_eq!(quantized.values.elem_count(), boundary_values.len());
    }
}

/// Test invalid input validation
#[cfg(test)]
mod input_validation_tests {
    use super::*;

    #[test]
    fn test_invalid_tensor_dtypes() {
        let device = Device::Cpu;
        
        // Test with non-float tensor types
        let int_weights = Tensor::ones((5, 5), DType::I64, &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        // Should handle type conversion or fail gracefully
        let result = absmean_quantize_weights(&int_weights, &device);
        // Either succeeds with conversion or fails with appropriate error
        match result {
            Ok(_) => {}, // Conversion succeeded
            Err(e) => {
                // Should be a type-related error
                assert!(format!("{e:?}").contains("type") || format!("{e:?}").contains("dtype"));
            }
        }
    }

    #[test]
    fn test_configuration_validation() {
        // Test various invalid configurations
        let mut config = WeightQuantizationConfig::default();
        
        // Test invalid outlier threshold
        config.outlier_threshold = -1.0; // Negative threshold
        assert!(config.validate().is_err());
        
        config.outlier_threshold = 0.0; // Zero threshold might be invalid
        let validation_result = config.validate();
        // Depending on implementation, this might be valid or invalid
        if validation_result.is_err() {
            println!("Zero outlier threshold is invalid as expected");
        }
    }

    #[test]
    fn test_precision_bounds_validation() {
        let mut bounds = PrecisionBounds::default();
        
        // Test invalid bounds
        bounds.min_threshold = bounds.max_threshold + 1.0;
        assert!(bounds.validate().is_err());
        
        bounds = PrecisionBounds::default();
        bounds.min_scale = bounds.max_scale + 1.0;
        assert!(bounds.validate().is_err());
        
        bounds = PrecisionBounds::default();
        bounds.max_error_tolerance = -1.0;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_mixed_precision_config_validation() {
        let mut config = MixedPrecisionQuantizationConfig::default();
        
        // Test invalid precision adjustment parameters
        config.adjustment_params.accuracy_threshold = 1.5; // > 1.0
        assert!(config.validate().is_err());
        
        config.adjustment_params.accuracy_threshold = -0.1; // < 0.0
        assert!(config.validate().is_err());
    }
}

/// Test memory pressure and resource exhaustion
#[cfg(test)]
mod resource_exhaustion_tests {
    use super::*;

    #[test]
    fn test_memory_pressure_simulation() {
        let device = Device::Cpu;
        
        // Test with progressively larger tensors to simulate memory pressure
        let sizes = vec![1000, 5000, 10000]; // Reasonable sizes for testing
        
        for size in sizes {
            let data: Vec<f32> = (0..size).map(|i| (i % 3) as f32 - 1.0).collect();
            let weights = Tensor::from_slice(&data, (size,), &device).unwrap();
            let config = WeightQuantizationConfig::default();
            
            let result = absmean_quantize_weights(&weights, &device);
            assert!(result.is_ok(), "Failed at size {size}");
            
            // Verify memory is released properly
            drop(result);
        }
    }

    #[test]
    fn test_repeated_operations() {
        let device = Device::Cpu;
        let weights = Tensor::randn(0.0, 1.0, (100, 100), &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        // Perform many repeated operations to test for memory leaks
        for i in 0..50 {
            let result = absmean_quantize_weights(&weights, &device);
            assert!(result.is_ok(), "Failed at iteration {i}");
            
            // Immediately drop to test cleanup
            drop(result);
        }
    }

    #[test]
    fn test_packing_memory_efficiency() {
        // Test packing operations with various strategies
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1];
        let config = TernaryPackingConfig::default();
        
        let strategies = vec![
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::ByteAligned,
            TernaryPackingStrategy::RunLengthEncoded,
            TernaryPackingStrategy::CompressedSparse,
        ];
        
        for strategy in strategies {
            let mut test_config = config.clone();
            test_config.strategy = strategy;
            
            let packer = TernaryPackerFactory::create_packer(strategy);
            let result = packer.pack(&weights, &test_config);
            assert!(result.is_ok(), "Packing failed for strategy {strategy:?}");
            
            let packed = result.unwrap();
            let unpack_result = packer.unpack(&packed);
            assert!(unpack_result.is_ok(), "Unpacking failed for strategy {strategy:?}");
            
            let unpacked = unpack_result.unwrap();
            assert_eq!(weights, unpacked, "Data mismatch for strategy {strategy:?}");
        }
    }
}

/// Test error propagation and recovery
#[cfg(test)]
mod error_propagation_tests {
    use super::*;

    #[test]
    fn test_quantization_error_propagation() {
        let device = Device::Cpu;
        
        // Create a scenario that should fail
        let weights = Tensor::zeros((0, 5), DType::F32, &device).unwrap(); // Invalid shape
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&weights, &device);
        assert!(result.is_err());
        
        // Verify error type is appropriate
        match result.unwrap_err() {
            QuantizationError::ValidationFailed(_) => {}, // Expected
            QuantizationError::TensorError(_) => {}, // Also acceptable
            other => panic!("Unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn test_mixed_precision_error_handling() {
        let device = Device::Cpu;
        
        // Create invalid mixed precision configuration
        let mut config = MixedPrecisionQuantizationConfig::default();
        config.adjustment_params.accuracy_threshold = f32::NAN; // Invalid threshold
        
        let result = MixedPrecisionQuantizer::new(config, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_corruption_detection_error_handling() {
        let detector = CorruptionDetector::default();
        
        // Create obviously corrupted data
        let corrupted_data = PackedTernaryWeights {
            data: vec![255; 10], // All invalid values
            shape: Shape::from_dims(&[100]), // Mismatched size
            strategy: TernaryPackingStrategy::BitPacked2Bit,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count: 100,
                ..Default::default()
            },
            memory_footprint: 10,
            compression_ratio: 10.0,
        };
        
        let reports = detector.detect_corruption(&corrupted_data).unwrap();
        assert!(!reports.is_empty());
        
        // Should detect multiple types of corruption
        let corruption_types: Vec<_> = reports.iter()
            .map(|r| std::mem::discriminant(&r.corruption_type))
            .collect();
        assert!(corruption_types.len() > 1);
    }

    #[test]
    fn test_precision_controller_error_recovery() {
        let mut config = PrecisionControlConfig::default();
        config.precision_bounds.max_error_tolerance = 0.001; // Very strict
        
        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();
        
        // Simulate high error scenario
        let bad_stats = QuantizationStats {
            elements_count: 1000,
            quantization_error: 0.5, // Very high error
            compression_ratio: 1.1, // Poor compression
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };
        
        // Should attempt adjustment
        let adjustment = controller.adjust_precision_dynamically(&bad_stats).unwrap();
        if let Some(adj) = adjustment {
            assert_eq!(adj.reason, AdjustmentReason::HighError);
        }
    }
}

/// Test packing edge cases
#[cfg(test)]
mod packing_edge_cases {
    use super::*;

    #[test]
    fn test_empty_weight_packing() {
        let weights: Vec<i8> = vec![];
        let config = TernaryPackingConfig::default();
        
        let strategies = vec![
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::ByteAligned,
        ];
        
        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            let result = packer.pack(&weights, &config);
            
            match result {
                Ok(packed) => {
                    let unpack_result = packer.unpack(&packed);
                    assert!(unpack_result.is_ok());
                    assert_eq!(weights, unpack_result.unwrap());
                }
                Err(_) => {
                    // Some strategies might not handle empty data
                }
            }
        }
    }

    #[test]
    fn test_single_value_packing() {
        let weights = vec![1i8];
        let config = TernaryPackingConfig::default();
        
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let result = packer.pack(&weights, &config);
        assert!(result.is_ok());
        
        let packed = result.unwrap();
        let unpacked = packer.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_all_same_value_packing() {
        let weights = vec![0i8; 100]; // All zeros
        let config = TernaryPackingConfig::default();
        
        let strategies = vec![
            TernaryPackingStrategy::RunLengthEncoded,
            TernaryPackingStrategy::CompressedSparse,
        ];
        
        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            let result = packer.pack(&weights, &config);
            assert!(result.is_ok(), "Failed for strategy {strategy:?}");
            
            let packed = result.unwrap();
            let unpacked = packer.unpack(&packed).unwrap();
            assert_eq!(weights, unpacked);
        }
    }

    #[test]
    fn test_alternating_pattern_packing() {
        let weights: Vec<i8> = (0..1000).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let config = TernaryPackingConfig::default();
        
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let result = packer.pack(&weights, &config);
        assert!(result.is_ok());
        
        let packed = result.unwrap();
        let unpacked = packer.unpack(&packed).unwrap();
        assert_eq!(weights, unpacked);
    }
}

/// Test SIMD unpacking edge cases
#[cfg(test)]
mod simd_edge_cases {
    use super::*;

    #[test]
    fn test_simd_with_small_data() {
        let weights = vec![-1i8, 0, 1]; // Very small data
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        let packed = packer.pack(&weights, &config).unwrap();
        
        let unpacker = SimdUnpacker::new();
        let result = unpacker.unpack(&packed);
        assert!(result.is_ok());
        assert_eq!(weights, result.unwrap());
    }

    #[test]
    fn test_simd_fallback_consistency() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
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
        
        assert_eq!(simd_result, scalar_result);
        assert_eq!(weights, simd_result);
    }

    #[test]
    fn test_simd_with_odd_sizes() {
        let sizes = vec![1, 3, 7, 15, 31]; // Odd sizes that don't align well
        
        for size in sizes {
            let weights: Vec<i8> = (0..size).map(|i| (i % 3) as i8 - 1).collect();
            let config = TernaryPackingConfig::default();
            let packer = BitPacked2BitPacker;
            let packed = packer.pack(&weights, &config).unwrap();
            
            let unpacker = SimdUnpacker::new();
            let result = unpacker.unpack(&packed);
            assert!(result.is_ok(), "Failed for size {size}");
            assert_eq!(weights, result.unwrap(), "Data mismatch for size {size}");
        }
    }
}

/// Test performance under stress
#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_quantization_performance_consistency() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        
        // Test with increasing tensor sizes
        let sizes = vec![100, 1000, 5000];
        let mut times = Vec::new();
        
        for size in sizes {
            let weights = Tensor::randn(0.0, 1.0, (size,), &device).unwrap();
            
            let start = Instant::now();
            let result = absmean_quantize_weights(&weights, &device);
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Failed at size {size}");
            times.push(duration);
            
            // Performance should not degrade catastrophically
            if times.len() > 1 {
                let ratio = times.last().unwrap().as_secs_f64() / times[times.len()-2].as_secs_f64();
                assert!(ratio < 100.0, "Performance degraded too much: {ratio}x");
            }
        }
    }

    #[test]
    fn test_packing_performance_consistency() {
        let weights: Vec<i8> = (0..10000).map(|i| (i % 3) as i8 - 1).collect();
        let config = TernaryPackingConfig::default();
        
        let strategies = vec![
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::ByteAligned,
        ];
        
        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            
            let start = Instant::now();
            let result = packer.pack(&weights, &config);
            let pack_time = start.elapsed();
            
            assert!(result.is_ok(), "Packing failed for {strategy:?}");
            
            let packed = result.unwrap();
            let start = Instant::now();
            let unpack_result = packer.unpack(&packed);
            let unpack_time = start.elapsed();
            
            assert!(unpack_result.is_ok(), "Unpacking failed for {strategy:?}");
            
            // Performance should be reasonable
            assert!(pack_time.as_millis() < 1000, "Packing too slow for {:?}: {}ms", strategy, pack_time.as_millis());
            assert!(unpack_time.as_millis() < 1000, "Unpacking too slow for {:?}: {}ms", strategy, unpack_time.as_millis());
        }
    }

    #[test]
    fn test_memory_usage_stability() {
        let device = Device::Cpu;
        let config = WeightQuantizationConfig::default();
        
        // Perform many operations and check for memory leaks
        for i in 0..50 {
            let size = 1000 + (i * 10); // Gradually increasing size
            let weights = Tensor::randn(0.0, 1.0, (size,), &device).unwrap();
            
            let result = absmean_quantize_weights(&weights, &device);
            assert!(result.is_ok(), "Failed at iteration {i}");
            
            // Force cleanup
            drop(result);
            drop(weights);
            
            // In a real test, you might check actual memory usage here
            // For now, just ensure no panics occur
        }
    }
}

/// Integration tests for complete error scenarios
#[cfg(test)]
mod integration_error_tests {
    use super::*;

    #[test]
    fn test_end_to_end_error_handling() {
        let device = Device::Cpu;
        
        // Create a scenario that exercises multiple components
        let weights = Tensor::randn(0.0, 1.0, (50, 50), &device).unwrap();
        
        // 1. Quantize weights
        let weight_config = WeightQuantizationConfig::default();
        let quantized_weights = absmean_quantize_weights(&weights, &device).unwrap();
        
        // 2. Create activations
        let activations = Tensor::randn(0.0, 1.0, (50,), &device).unwrap();
        let activation_config = ActivationQuantizationConfig::default();
        let quantized_activations = absmax_quantize_activations(&activations, &device, None).unwrap();
        
        // 3. Test mixed precision
        let mixed_config = MixedPrecisionQuantizationConfig::default();
        let mixed_quantizer = MixedPrecisionQuantizer::new(mixed_config, device.clone()).unwrap();
        
        // 4. Test packing
        let ternary_weights: Vec<i8> = quantized_weights.values.to_vec1::<f32>().unwrap()
            .iter().map(|&x| x.clamp(-1.0, 1.0) as i8).collect();
        let packing_config = TernaryPackingConfig::default();
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let packed = packer.pack(&ternary_weights, &packing_config).unwrap();
        
        // 5. Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        assert_eq!(ternary_weights, unpacked);
        
        // 6. Test corruption detection
        let detector = CorruptionDetector::default();
        let reports = detector.detect_corruption(&packed).unwrap();
        // Should have no serious corruption reports for valid data
        let serious_corruptions: Vec<_> = reports.iter()
            .filter(|r| matches!(r.severity, CorruptionSeverity::Severe | CorruptionSeverity::Critical))
            .collect();
        assert!(serious_corruptions.is_empty());
        
        // 7. Test precision control
        let precision_config = PrecisionControlConfig::default();
        let mut precision_controller = PrecisionController::new(precision_config, device).unwrap();
        
        let stats = QuantizationStats {
            elements_count: ternary_weights.len(),
            quantization_error: 0.05,
            compression_ratio: packed.compression_ratio,
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };
        
        precision_controller.record_metrics(&stats, Duration::from_millis(10));
        let summary = precision_controller.get_performance_summary();
        assert_eq!(summary.operations_count, 1);
    }

    #[test]
    fn test_cascading_error_recovery() {
        let device = Device::Cpu;
        
        // Test recovery when one component fails
        let weights = Tensor::from_slice(&[1.0, -1.0, 0.0, 2.0], (4,), &device).unwrap();
        let config = WeightQuantizationConfig::default();
        
        let result = absmean_quantize_weights(&weights, &device);
        match result {
            Ok(quantized) => {
                // If it succeeds, verify the result is reasonable
                let data = quantized.values.to_vec1::<f32>().unwrap();
                for &val in &data {
                    assert!((-1.0..=1.0).contains(&val), "Quantized value {val} out of range");
                }
            }
            Err(e) => {
                // Should fail with appropriate error type
                assert!(matches!(e, 
                    QuantizationError::ValidationFailed(_) | 
                    QuantizationError::TensorError(_) |
                    QuantizationError::InvalidInput(_)
                ));
            }
        }
    }
}