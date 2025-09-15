#[cfg(test)]
mod specific_debug {
    use crate::engine::weight_conversion::*;
    use crate::engine::model_loader::{ParameterData, ParameterDataType};

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
    fn test_specific_failing_case() {
        let converter = WeightConverter::with_default_cache();
        
        // Test the specific failing case: [-1, -1, -1, -1]
        let test_values = vec![-1, -1, -1, -1];
        let packed_data = create_test_ternary_data(&test_values);
        
        println!("Input values: {:?}", test_values);
        println!("Packed data: {:02x?}", packed_data);
        println!("Expected: all values should be -1");
        
        // For [-1, -1, -1, -1]:
        // Position 0: -1 -> 00
        // Position 1: -1 -> 00 << 2 = 0000
        // Position 2: -1 -> 00 << 4 = 000000
        // Position 3: -1 -> 00 << 6 = 00000000
        // Result: 00000000 = 0x00
        assert_eq!(packed_data, vec![0x00]);
        
        let param_data = ParameterData {
            data: packed_data,
            shape: vec![test_values.len()],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "test_specific".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                println!("Converted weights: {:?}", weights);
                assert_eq!(weights.len(), test_values.len());
                for (i, (&expected, &actual)) in test_values.iter().zip(weights.iter()).enumerate() {
                    assert_eq!(expected, actual, "Mismatch at position {}: expected {}, got {}", i, expected, actual);
                }
            }
            _ => panic!("Expected ternary weights"),
        }
    }
}