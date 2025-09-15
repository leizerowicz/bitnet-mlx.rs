//! Comprehensive tests for weight conversion functionality

use super::*;
use crate::engine::model_loader::{ParameterData, ParameterDataType, ModelWeights, ParameterType};
use std::sync::Arc;

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    fn create_test_ternary_data(values: &[i8]) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Process in chunks of 4 values per byte
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let packed = match val {
                    -1 => 0u8,  // 00 in binary
                    0 => 1u8,   // 01 in binary  
                    1 => 2u8,   // 10 in binary
                    _ => 1u8,   // Default to 0 (01 in binary)
                };
                byte |= packed << (i * 2);
            }
            data.push(byte);
        }
        
        data
    }

    #[test]
    fn test_ternary_conversion_comprehensive() {
        let converter = WeightConverter::with_default_cache();
        
        // Test various ternary patterns
        let test_cases = vec![
            vec![-1, 0, 1, 0],
            vec![1, 1, -1, -1, 0, 0, 1, -1],
            vec![-1, -1, -1, -1],
            vec![1, 1, 1, 1],
            vec![0, 0, 0, 0],
        ];

        for (case_idx, test_values) in test_cases.iter().enumerate() {
            let packed_data = create_test_ternary_data(&test_values);
            let shape = vec![test_values.len()];
            
            let param_data = ParameterData {
                data: packed_data,
                shape,
                dtype: ParameterDataType::BitnetB158,
                tensor_name: format!("test_ternary_case_{}", case_idx), // Unique name per case
            };

            let converted = converter.convert_parameter(&param_data).unwrap();
            
            match converted.as_ref() {
                WeightArrays::Ternary(weights) => {
                    assert_eq!(weights.len(), test_values.len());
                    for (i, (&expected, &actual)) in test_values.iter().zip(weights.iter()).enumerate() {
                        assert_eq!(expected, actual, "Mismatch at position {} for values {:?}", i, test_values);
                    }
                }
                _ => panic!("Expected ternary weights"),
            }
        }
    }

    #[test]
    fn test_f32_conversion_edge_cases() {
        let converter = WeightConverter::with_default_cache();
        
        let test_values = vec![
            f32::MIN,
            f32::MAX,
            0.0f32,
            -0.0f32,
            1.0f32,
            -1.0f32,
            f32::EPSILON,
            -f32::EPSILON,
            std::f32::consts::PI,
            std::f32::consts::E,
        ];
        
        let mut data = Vec::new();
        for value in &test_values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        let param_data = ParameterData {
            data,
            shape: vec![test_values.len()],
            dtype: ParameterDataType::F32,
            tensor_name: "test_f32_edge".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::F32(weights) => {
                assert_eq!(weights.len(), test_values.len());
                for (i, (&expected, &actual)) in test_values.iter().zip(weights.iter()).enumerate() {
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "Expected NaN at position {}", i);
                    } else {
                        assert!((expected - actual).abs() < f32::EPSILON, 
                               "Mismatch at position {}: expected {}, got {}", i, expected, actual);
                    }
                }
            }
            _ => panic!("Expected F32 weights"),
        }
    }

    #[test]
    fn test_f16_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        let test_values = vec![1.0f32, -2.5f32, 0.0f32, 65504.0f32]; // Values within F16 range
        let mut data = Vec::new();
        
        for value in &test_values {
            let f16_val = half::f16::from_f32(*value);
            data.extend_from_slice(&f16_val.to_bits().to_le_bytes());
        }
        
        let param_data = ParameterData {
            data,
            shape: vec![test_values.len()],
            dtype: ParameterDataType::F16,
            tensor_name: "test_f16".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::F16(weights) => {
                assert_eq!(weights.len(), test_values.len());
                for (i, (&expected, &actual)) in test_values.iter().zip(weights.iter()).enumerate() {
                    // F16 has limited precision, so allow some tolerance
                    let tolerance = 0.001f32;
                    assert!((expected - actual).abs() < tolerance, 
                           "Mismatch at position {}: expected {}, got {}", i, expected, actual);
                }
            }
            _ => panic!("Expected F16 weights"),
        }
    }

    #[test]
    fn test_i8_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        let test_values = vec![i8::MIN, i8::MAX, 0i8, -1i8, 1i8, 127i8, -128i8];
        let data: Vec<u8> = test_values.iter().map(|&v| v as u8).collect();
        
        let param_data = ParameterData {
            data,
            shape: vec![test_values.len()],
            dtype: ParameterDataType::I8,
            tensor_name: "test_i8".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::I8(weights) => {
                assert_eq!(weights.len(), test_values.len());
                assert_eq!(*weights, test_values);
            }
            _ => panic!("Expected I8 weights"),
        }
    }

    #[test]
    fn test_quantized_q8_0_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        // Create Q8_0 test data: scale + 32 quantized values
        let scale = 0.1f32;
        let quantized_values: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_le_bytes());
        for &val in &quantized_values {
            data.push(val as u8);
        }
        
        let param_data = ParameterData {
            data,
            shape: vec![32],
            dtype: ParameterDataType::Quantized("Q8_0".to_string()),
            tensor_name: "test_q8_0".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::Quantized { weights, format, .. } => {
                assert_eq!(weights.len(), 32);
                assert_eq!(format, "Q8_0");
                assert_eq!(*weights, quantized_values);
            }
            _ => panic!("Expected quantized weights"),
        }
    }

    #[test]
    fn test_cache_behavior() {
        let converter = WeightConverter::new(1024); // 1KB cache
        
        let param_data = ParameterData {
            data: create_test_ternary_data(&[-1, 0, 1, 0]),
            shape: vec![4],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "cache_test".to_string(),
        };

        // First conversion
        let converted1 = converter.convert_parameter(&param_data).unwrap();
        let (entries1, size1, _) = converter.cache_stats();
        assert_eq!(entries1, 1);
        assert!(size1 > 0);

        // Second conversion should use cache
        let converted2 = converter.convert_parameter(&param_data).unwrap();
        let (entries2, size2, _) = converter.cache_stats();
        assert_eq!(entries2, 1); // Same number of entries
        assert_eq!(size1, size2); // Same cache size
        assert!(Arc::ptr_eq(&converted1, &converted2)); // Same Arc instance

        // Clear cache
        converter.clear_cache();
        let (entries3, size3, _) = converter.cache_stats();
        assert_eq!(entries3, 0);
        assert_eq!(size3, 0);
    }

    #[test]
    fn test_error_conditions() {
        let converter = WeightConverter::with_default_cache();
        
        // Test insufficient data for ternary
        let param_data = ParameterData {
            data: vec![], // Empty data
            shape: vec![4], // But expecting 4 elements
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "insufficient_data".to_string(),
        };

        let result = converter.convert_parameter(&param_data);
        assert!(result.is_err());

        // Test insufficient data for F32
        let param_data = ParameterData {
            data: vec![0x00, 0x01], // Only 2 bytes
            shape: vec![2], // But expecting 2 F32 values (8 bytes)
            dtype: ParameterDataType::F32,
            tensor_name: "insufficient_f32".to_string(),
        };

        let result = converter.convert_parameter(&param_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_weights_integration() {
        let converter = Arc::new(WeightConverter::with_default_cache());
        let mut model_weights = ModelWeights::with_converter(converter);
        
        // Add some test parameters
        let ternary_data = ParameterData {
            data: create_test_ternary_data(&[-1, 0, 1, 0]),
            shape: vec![4],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "layer_0.weight".to_string(),
        };
        
        let f32_data = ParameterData {
            data: 1.0f32.to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: ParameterDataType::F32,
            tensor_name: "layer_0.bias".to_string(),
        };

        model_weights.add_parameter(0, ParameterType::Weight, ternary_data);
        model_weights.add_parameter(0, ParameterType::Bias, f32_data);

        assert!(model_weights.has_converter());
        
        // Test single parameter conversion
        let converted_weight = model_weights.convert_parameter(0, ParameterType::Weight).unwrap();
        match converted_weight.as_ref() {
            WeightArrays::Ternary(weights) => {
                assert_eq!(weights, &[-1, 0, 1, 0]);
            }
            _ => panic!("Expected ternary weights"),
        }

        // Test layer parameter conversion
        let layer_params = model_weights.convert_layer_parameters(0).unwrap();
        assert_eq!(layer_params.len(), 2);
        assert!(layer_params.contains_key(&ParameterType::Weight));
        assert!(layer_params.contains_key(&ParameterType::Bias));

        // Test cache stats
        let stats = model_weights.converter_stats().unwrap();
        assert!(stats.0 > 0); // Should have cache entries
    }

    #[test]
    fn test_large_tensor_handling() {
        let converter = WeightConverter::with_default_cache();
        
        // Create a large ternary tensor (1024 elements = 256 bytes)
        let large_values: Vec<i8> = (0..1024).map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        }).collect();
        
        let packed_data = create_test_ternary_data(&large_values);
        
        let param_data = ParameterData {
            data: packed_data,
            shape: vec![1024],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "large_tensor".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                assert_eq!(weights.len(), 1024);
                for (i, (&expected, &actual)) in large_values.iter().zip(weights.iter()).enumerate() {
                    assert_eq!(expected, actual, "Mismatch at position {}", i);
                }
            }
            _ => panic!("Expected ternary weights"),
        }
    }

    #[test]
    fn test_weight_arrays_methods() {
        // Test ternary weights
        let ternary = WeightArrays::Ternary(vec![-1, 0, 1]);
        assert_eq!(ternary.len(), 3);
        assert!(!ternary.is_empty());
        assert_eq!(ternary.as_ternary(), Some([-1, 0, 1].as_slice()));
        assert_eq!(ternary.as_i8_slice(), Some([-1, 0, 1].as_slice()));
        assert_eq!(ternary.as_f32_slice(), None);

        // Test F32 weights
        let f32_weights = WeightArrays::F32(vec![1.0, 2.0, 3.0]);
        assert_eq!(f32_weights.len(), 3);
        assert!(!f32_weights.is_empty());
        assert_eq!(f32_weights.as_f32_slice(), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(f32_weights.as_i8_slice(), None);
        assert_eq!(f32_weights.as_ternary(), None);

        // Test empty weights
        let empty = WeightArrays::Ternary(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}