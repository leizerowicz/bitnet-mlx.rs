//! Tests for MLX array utilities

#[cfg(test)]
#[cfg(feature = "mlx")]
mod tests {
    use crate::mlx::{create_mlx_array, mlx_to_candle_tensor, candle_to_mlx_array};
    use crate::tensor::create_tensor_f32;

    #[test]
    fn test_create_mlx_array() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = create_mlx_array(shape, data);
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.shape(), &[2, 3]);
    }

    #[test]
    fn test_create_mlx_array_invalid_data_length() {
        let shape = &[2, 3]; // Expects 6 elements
        let data = vec![1.0, 2.0, 3.0]; // Only 3 elements
        
        let result = create_mlx_array(shape, data);
        assert!(result.is_err());
    }

    #[test]
    fn test_mlx_to_candle_tensor() {
        let shape = &[2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        let array = create_mlx_array(shape, data.clone()).unwrap();
        let result = mlx_to_candle_tensor(&array);
        
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_candle_to_mlx_array() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        
        let result = candle_to_mlx_array(&tensor);
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.shape(), &[2, 2]);
    }

    #[test]
    fn test_round_trip_conversion() {
        // Test: Candle -> MLX -> Candle
        let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original_tensor = create_tensor_f32(&[2, 3], original_data.clone()).unwrap();
        
        // Convert to MLX array
        let mlx_array = candle_to_mlx_array(&original_tensor).unwrap();
        assert_eq!(mlx_array.shape(), &[2, 3]);
        
        // Convert back to Candle tensor
        let final_tensor = mlx_to_candle_tensor(&mlx_array).unwrap();
        assert_eq!(final_tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_different_shapes() {
        // Test 1D array
        let shape_1d = &[4];
        let data_1d = vec![1.0, 2.0, 3.0, 4.0];
        let array_1d = create_mlx_array(shape_1d, data_1d).unwrap();
        assert_eq!(array_1d.shape(), &[4]);

        // Test 3D array
        let shape_3d = &[2, 2, 2];
        let data_3d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let array_3d = create_mlx_array(shape_3d, data_3d).unwrap();
        assert_eq!(array_3d.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_empty_array() {
        let shape = &[0];
        let data = vec![];
        
        let result = create_mlx_array(shape, data);
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.shape(), &[0]);
    }

    #[test]
    fn test_mlx_matmul_wrapper() {
        use crate::mlx::mlx_matmul;
        use mlx_rs::Array;

        // Create two 2x2 matrices
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Array::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let result = mlx_matmul(&a, &b);
        assert!(result.is_ok());

        let result_array = result.unwrap();
        assert_eq!(result_array.shape(), &[2, 2]);

        // Verify the result values
        let result_data = result_array.as_slice::<f32>();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_mlx_quantize_wrapper() {
        use crate::mlx::mlx_quantize;
        use mlx_rs::Array;

        // Create a test array
        let array = Array::from_slice(&[1.0, 2.5, 3.7, 4.2], &[2, 2]);
        let scale = 0.5;

        let result = mlx_quantize(&array, Some(scale));
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.shape(), &[2, 2]);

        // Verify quantization: round(array / scale)
        let result_data = quantized.as_slice::<f32>();
        // Expected: round([1.0/0.5, 2.5/0.5, 3.7/0.5, 4.2/0.5]) = round([2, 5, 7.4, 8.4]) = [2, 5, 7, 8]
        let expected = vec![2.0, 5.0, 7.0, 8.0];
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_mlx_dequantize_wrapper() {
        use crate::mlx::mlx_dequantize;
        use mlx_rs::Array;

        // Create a quantized array (integers)
        let quantized = Array::from_slice(&[2.0, 5.0, 7.0, 8.0], &[2, 2]);
        let scale = 0.5;

        let result = mlx_dequantize(&quantized, Some(scale));
        assert!(result.is_ok());

        let dequantized = result.unwrap();
        assert_eq!(dequantized.shape(), &[2, 2]);

        // Verify dequantization: quantized * scale
        let result_data = dequantized.as_slice::<f32>();
        // Expected: [2*0.5, 5*0.5, 7*0.5, 8*0.5] = [1.0, 2.5, 3.5, 4.0]
        let expected = vec![1.0, 2.5, 3.5, 4.0];
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_quantize_dequantize_round_trip() {
        use crate::mlx::{mlx_quantize, mlx_dequantize};
        use mlx_rs::Array;

        // Test round-trip: original -> quantize -> dequantize
        let original = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let scale = 1.0; // Use scale of 1.0 for exact round-trip

        // Quantize
        let quantized = mlx_quantize(&original, Some(scale)).unwrap();
        
        // Dequantize
        let dequantized = mlx_dequantize(&quantized, Some(scale)).unwrap();

        assert_eq!(dequantized.shape(), &[2, 2]);
        
        // With scale=1.0, we should get back the original values (rounded)
        let result_data = dequantized.as_slice::<f32>();
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_mlx_matmul_different_shapes() {
        use crate::mlx::mlx_matmul;
        use mlx_rs::Array;

        // Test 2x3 * 3x2 = 2x2
        let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Array::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let result = mlx_matmul(&a, &b);
        assert!(result.is_ok());

        let result_array = result.unwrap();
        assert_eq!(result_array.shape(), &[2, 2]);
    }

    #[test]
    fn test_mlx_quantize_with_different_scales() {
        use crate::mlx::mlx_quantize;
        use mlx_rs::Array;

        let array = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // Test with scale = 2.0
        let result1 = mlx_quantize(&array, Some(2.0)).unwrap();
        let data1 = result1.as_slice::<f32>();
        // Expected: round([0.5, 1.0, 1.5, 2.0]) = [1, 1, 2, 2]
        assert_eq!(data1, vec![1.0, 1.0, 2.0, 2.0]);

        // Test with scale = 0.25
        let result2 = mlx_quantize(&array, Some(0.25)).unwrap();
        let data2 = result2.as_slice::<f32>();
        // Expected: round([4.0, 8.0, 12.0, 16.0]) = [4, 8, 12, 16]
        assert_eq!(data2, vec![4.0, 8.0, 12.0, 16.0]);
    }
}

#[cfg(test)]
#[cfg(not(feature = "mlx"))]
mod stub_tests {
    use super::*;

    #[test]
    fn test_mlx_functions_without_feature() {
        let shape = &[2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        // All functions should return errors when MLX feature is not enabled
        assert!(create_mlx_array(shape, data).is_err());
        assert!(mlx_to_candle_tensor(&()).is_err());
        assert!(candle_to_mlx_array(&()).is_err());
    }
}