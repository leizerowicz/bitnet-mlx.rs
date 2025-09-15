//! Simple debug test for ternary conversion

#[cfg(test)]
mod debug_test {
    use crate::engine::weight_conversion::*;
    use crate::engine::model_loader::{ParameterData, ParameterDataType};

    #[test]
    fn debug_ternary_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        // Test single values step by step
        println!("Testing single value conversions:");
        
        // Test [-1]: should be packed as 00 in 2 bits
        let data_minus_one = vec![0b00000000]; // -1 at position 0
        let param_data = ParameterData {
            data: data_minus_one,
            shape: vec![1],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "debug_minus_one".to_string(),
        };
        
        let converted = converter.convert_parameter(&param_data).unwrap();
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                println!("Input: 0x00, Expected: [-1], Got: {:?}", weights);
                assert_eq!(weights[0], -1);
            }
            _ => panic!("Expected ternary weights"),
        }
        
        // Test [0]: should be packed as 01 in 2 bits  
        let data_zero = vec![0b00000001]; // 0 at position 0
        let param_data = ParameterData {
            data: data_zero,
            shape: vec![1],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "debug_zero".to_string(),
        };
        
        let converted = converter.convert_parameter(&param_data).unwrap();
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                println!("Input: 0x01, Expected: [0], Got: {:?}", weights);
                assert_eq!(weights[0], 0);
            }
            _ => panic!("Expected ternary weights"),
        }
        
        // Test [1]: should be packed as 10 in 2 bits
        let data_plus_one = vec![0b00000010]; // 1 at position 0
        let param_data = ParameterData {
            data: data_plus_one,
            shape: vec![1],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "debug_plus_one".to_string(),
        };
        
        let converted = converter.convert_parameter(&param_data).unwrap();
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                println!("Input: 0x02, Expected: [1], Got: {:?}", weights);
                assert_eq!(weights[0], 1);
            }
            _ => panic!("Expected ternary weights"),
        }
        
        // Test [-1, -1]: should be packed as 0000 in 4 bits
        let data_minus_ones = vec![0b00000000]; // Two -1s at positions 0 and 1
        let param_data = ParameterData {
            data: data_minus_ones,
            shape: vec![2],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "debug_minus_ones".to_string(),
        };
        
        let converted = converter.convert_parameter(&param_data).unwrap();
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                println!("Input: 0x00, Expected: [-1, -1], Got: {:?}", weights);
                assert_eq!(weights, &[-1, -1]);
            }
            _ => panic!("Expected ternary weights"),
        }
    }
}