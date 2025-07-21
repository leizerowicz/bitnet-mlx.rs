//! Comprehensive Array Conversion Tests
//!
//! This test suite validates Array conversion functionality including:
//! - Array to BitNetTensor conversions
//! - BitNetTensor to Array conversions  
//! - Array to Array conversions with different data types
//! - Different Array shapes and sizes
//! - Error handling and edge cases
//! - Performance and memory efficiency

use bitnet_core::memory::{
    HybridMemoryPool,
    conversion::{
        ConversionEngine, ConversionConfig, ConversionQuality
    },
    tensor::{BitNetTensor, BitNetDType}
};
use bitnet_core::device::get_cpu_device;
use std::sync::Arc;

#[cfg(feature = "mlx")]
use mlx_rs::Array;

/// Test basic Array to BitNetTensor conversion
#[test]
fn test_array_to_bitnet_tensor_conversion() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    
    // Test data for 2x3 array
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2, 3];
    
    // Create BitNetTensor from array data
    let tensor = BitNetTensor::from_data(data.clone(), &shape, &device, &pool).unwrap();
    
    assert_eq!(tensor.shape(), vec![2, 3]);
    assert_eq!(tensor.dtype(), BitNetDType::F32);
    assert_eq!(tensor.element_count(), 6);
    assert_eq!(tensor.size_bytes(), 6 * 4); // 6 elements * 4 bytes per f32
}

/// Test BitNetTensor to Array conversion via data extraction
#[test]
fn test_bitnet_tensor_to_array_data() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    
    // Create tensor with known data
    let original_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = BitNetTensor::from_data(original_data.clone(), &[2, 2], &device, &pool).unwrap();
    
    // Convert to candle tensor and extract data
    let candle_tensor = tensor.to_candle().unwrap();
    let extracted_data = candle_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    
    assert_eq!(extracted_data, original_data);
}

/// Test Array conversions with different data types
#[test]
fn test_array_dtype_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Test F32 to F16 conversion
    let f32_data = vec![1.0f32, 2.5, 3.7, 4.2];
    let f32_tensor = BitNetTensor::from_data(f32_data, &[2, 2], &device, &pool).unwrap();
    let f16_tensor = engine.convert(&f32_tensor, BitNetDType::F16).unwrap();
    
    assert_eq!(f16_tensor.dtype(), BitNetDType::F16);
    assert_eq!(f16_tensor.shape(), vec![2, 2]);
    assert!(f16_tensor.size_bytes() < f32_tensor.size_bytes()); // F16 uses less memory
    
    // Test F32 to I8 conversion
    let i8_tensor = engine.convert(&f32_tensor, BitNetDType::I8).unwrap();
    assert_eq!(i8_tensor.dtype(), BitNetDType::I8);
    assert_eq!(i8_tensor.shape(), vec![2, 2]);
    assert_eq!(i8_tensor.size_bytes(), 4); // 4 elements * 1 byte per i8
}

/// Test Array conversions with different shapes
#[test]
fn test_array_shape_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    
    // Test 1D array
    let data_1d = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor_1d = BitNetTensor::from_data(data_1d, &[4], &device, &pool).unwrap();
    assert_eq!(tensor_1d.shape(), vec![4]);
    
    // Test 2D array
    let data_2d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_2d = BitNetTensor::from_data(data_2d, &[2, 3], &device, &pool).unwrap();
    assert_eq!(tensor_2d.shape(), vec![2, 3]);
    
    // Test 3D array
    let data_3d = vec![1.0f32; 24]; // 2x3x4 = 24 elements
    let tensor_3d = BitNetTensor::from_data(data_3d, &[2, 3, 4], &device, &pool).unwrap();
    assert_eq!(tensor_3d.shape(), vec![2, 3, 4]);
    assert_eq!(tensor_3d.element_count(), 24);
    
    // Test 4D array
    let data_4d = vec![1.0f32; 120]; // 2x3x4x5 = 120 elements
    let tensor_4d = BitNetTensor::from_data(data_4d, &[2, 3, 4, 5], &device, &pool).unwrap();
    assert_eq!(tensor_4d.shape(), vec![2, 3, 4, 5]);
    assert_eq!(tensor_4d.element_count(), 120);
}

/// Test Array conversions with different sizes
#[test]
fn test_array_size_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Test small array (16 elements)
    let small_data = vec![1.0f32; 16];
    let small_tensor = BitNetTensor::from_data(small_data, &[4, 4], &device, &pool).unwrap();
    let small_converted = engine.convert(&small_tensor, BitNetDType::F16).unwrap();
    assert_eq!(small_converted.element_count(), 16);
    
    // Test medium array (1024 elements)
    let medium_data = vec![1.0f32; 1024];
    let medium_tensor = BitNetTensor::from_data(medium_data, &[32, 32], &device, &pool).unwrap();
    let medium_converted = engine.convert(&medium_tensor, BitNetDType::F16).unwrap();
    assert_eq!(medium_converted.element_count(), 1024);
    
    // Test large array (65536 elements)
    let large_data = vec![1.0f32; 65536];
    let large_tensor = BitNetTensor::from_data(large_data, &[256, 256], &device, &pool).unwrap();
    let large_converted = engine.convert(&large_tensor, BitNetDType::F16).unwrap();
    assert_eq!(large_converted.element_count(), 65536);
}

/// Test Array batch conversions
#[test]
fn test_array_batch_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create multiple arrays with different shapes
    let arrays = vec![
        BitNetTensor::from_data(vec![1.0f32; 4], &[2, 2], &device, &pool).unwrap(),
        BitNetTensor::from_data(vec![2.0f32; 9], &[3, 3], &device, &pool).unwrap(),
        BitNetTensor::from_data(vec![3.0f32; 16], &[4, 4], &device, &pool).unwrap(),
    ];
    
    // Batch convert to F16
    let converted_arrays = engine.batch_convert(&arrays, BitNetDType::F16).unwrap();
    
    assert_eq!(converted_arrays.len(), 3);
    for (i, converted) in converted_arrays.iter().enumerate() {
        assert_eq!(converted.dtype(), BitNetDType::F16);
        assert_eq!(converted.shape(), arrays[i].shape());
        assert_eq!(converted.element_count(), arrays[i].element_count());
    }
}

/// Test Array mixed batch conversions
#[test]
fn test_array_mixed_batch_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create conversions with different target types
    let conversions = vec![
        (BitNetTensor::from_data(vec![1.0f32; 4], &[2, 2], &device, &pool).unwrap(), BitNetDType::F16),
        (BitNetTensor::from_data(vec![2.0f32; 9], &[3, 3], &device, &pool).unwrap(), BitNetDType::I8),
        (BitNetTensor::from_data(vec![3.0f32; 16], &[4, 4], &device, &pool).unwrap(), BitNetDType::F16),
    ];
    
    let results = engine.batch_convert_mixed(&conversions).unwrap();
    
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].dtype(), BitNetDType::F16);
    assert_eq!(results[1].dtype(), BitNetDType::I8);
    assert_eq!(results[2].dtype(), BitNetDType::F16);
}

/// Test Array conversion error handling
#[test]
fn test_array_conversion_error_handling() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    
    // Test mismatched data and shape
    let data = vec![1.0f32, 2.0, 3.0]; // 3 elements
    let result = BitNetTensor::from_data(data, &[2, 2], &device, &pool); // expects 4 elements
    assert!(result.is_err());
    
    // Test empty data
    let empty_data = vec![];
    let result = BitNetTensor::from_data(empty_data, &[0], &device, &pool);
    assert!(result.is_ok()); // Empty arrays should be valid
    
    // Test zero-dimensional array (scalar)
    let scalar_data = vec![42.0f32];
    let scalar_tensor = BitNetTensor::from_data(scalar_data, &[1], &device, &pool);
    assert!(scalar_tensor.is_ok()); // Scalar tensors should be valid
}

/// Test Array conversion with quantized types
#[test]
fn test_array_quantized_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create F32 tensor
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f32_tensor = BitNetTensor::from_data(f32_data, &[2, 4], &device, &pool).unwrap();
    
    // Test conversion to I4 (4-bit quantized) - may not be supported yet
    let i4_result = engine.convert(&f32_tensor, BitNetDType::I4);
    if i4_result.is_ok() {
        let i4_tensor = i4_result.unwrap();
        assert_eq!(i4_tensor.dtype(), BitNetDType::I4);
        assert_eq!(i4_tensor.shape(), vec![2, 4]);
        assert_eq!(i4_tensor.size_bytes(), 4); // 8 elements * 0.5 bytes per I4 element
    }
    
    // Test conversion to I2 (2-bit quantized) - may not be supported yet
    let i2_result = engine.convert(&f32_tensor, BitNetDType::I2);
    if i2_result.is_ok() {
        let i2_tensor = i2_result.unwrap();
        assert_eq!(i2_tensor.dtype(), BitNetDType::I2);
        assert_eq!(i2_tensor.shape(), vec![2, 4]);
        assert_eq!(i2_tensor.size_bytes(), 2); // 8 elements * 0.25 bytes per I2 element
    }
    
    // Test conversion to I1 (1-bit quantized) - may not be supported yet
    let i1_result = engine.convert(&f32_tensor, BitNetDType::I1);
    if i1_result.is_ok() {
        let i1_tensor = i1_result.unwrap();
        assert_eq!(i1_tensor.dtype(), BitNetDType::I1);
        assert_eq!(i1_tensor.shape(), vec![2, 4]);
        assert_eq!(i1_tensor.size_bytes(), 1); // 8 elements * 0.125 bytes per I1 element
    }
    
    // Test conversion to BitNet158 (ternary) - may not be supported yet
    let bitnet158_result = engine.convert(&f32_tensor, BitNetDType::BitNet158);
    if bitnet158_result.is_ok() {
        let bitnet158_tensor = bitnet158_result.unwrap();
        assert_eq!(bitnet158_tensor.dtype(), BitNetDType::BitNet158);
        assert_eq!(bitnet158_tensor.shape(), vec![2, 4]);
        assert_eq!(bitnet158_tensor.size_bytes(), 2); // 8 elements * 0.25 bytes per BitNet158 element
    }
}

/// Test Array conversion memory efficiency
#[test]
fn test_array_conversion_memory_efficiency() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create a large F32 tensor
    let large_data = vec![1.0f32; 1024]; // 4KB
    let f32_tensor = BitNetTensor::from_data(large_data, &[32, 32], &device, &pool).unwrap();
    let original_size = f32_tensor.size_bytes();
    
    // Test compression ratios
    let f16_tensor = engine.convert(&f32_tensor, BitNetDType::F16).unwrap();
    assert_eq!(f16_tensor.size_bytes() * 2, original_size); // 2x compression
    
    let i8_tensor = engine.convert(&f32_tensor, BitNetDType::I8).unwrap();
    assert_eq!(i8_tensor.size_bytes() * 4, original_size); // 4x compression
    
    // Test I4 conversion if supported
    let i4_result = engine.convert(&f32_tensor, BitNetDType::I4);
    if i4_result.is_ok() {
        let i4_tensor = i4_result.unwrap();
        assert_eq!(i4_tensor.size_bytes() * 8, original_size); // 8x compression
    }
    
    // Test I1 conversion if supported
    let i1_result = engine.convert(&f32_tensor, BitNetDType::I1);
    if i1_result.is_ok() {
        let i1_tensor = i1_result.unwrap();
        assert_eq!(i1_tensor.size_bytes() * 32, original_size); // 32x compression
    }
}

/// Test Array conversion performance with different strategies
#[test]
fn test_array_conversion_strategies() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create test tensor
    let data = vec![1.0f32; 256];
    let tensor = BitNetTensor::from_data(data, &[16, 16], &device, &pool).unwrap();
    
    // Test zero-copy conversion (same type)
    let zero_copy_result = engine.zero_copy_convert(&tensor, BitNetDType::F32);
    assert!(zero_copy_result.is_ok());
    
    // Test streaming conversion for large tensor
    let large_data = vec![1.0f32; 4096];
    let large_tensor = BitNetTensor::from_data(large_data, &[64, 64], &device, &pool).unwrap();
    let streaming_result = engine.streaming_convert(&large_tensor, BitNetDType::F16, 1024);
    assert!(streaming_result.is_ok());
    
    // Test strategy info
    let strategy_info = engine.get_optimal_strategy_info(
        BitNetDType::F32,
        BitNetDType::F16,
        &[16, 16],
        &device,
    );
    assert!(strategy_info.is_supported);
    assert_eq!(strategy_info.compression_ratio, 2.0);
}

/// Test Array conversion with different quality settings
#[test]
fn test_array_conversion_quality() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    let data = vec![1.0f32, 2.5, 3.7, 4.2];
    let tensor = BitNetTensor::from_data(data, &[2, 2], &device, &pool).unwrap();
    
    // Test fast quality
    let fast_result = engine.convert_with_quality(&tensor, BitNetDType::F16, ConversionQuality::Fast);
    assert!(fast_result.is_ok());
    
    // Test balanced quality
    let balanced_result = engine.convert_with_quality(&tensor, BitNetDType::F16, ConversionQuality::Balanced);
    assert!(balanced_result.is_ok());
    
    // Test precise quality
    let precise_result = engine.convert_with_quality(&tensor, BitNetDType::F16, ConversionQuality::Precise);
    assert!(precise_result.is_ok());
}

/// Test Array conversion pipeline
#[test]
fn test_array_conversion_pipeline() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create a multi-stage pipeline: F32 -> F16 -> I8 -> I4
    let pipeline = engine.create_pipeline().unwrap()
        .add_stage(BitNetDType::F16)
        .add_stage(BitNetDType::I8)
        .add_stage(BitNetDType::I4);
    
    let input_data = vec![1.0f32; 64];
    let input_tensor = BitNetTensor::from_data(input_data, &[8, 8], &device, &pool).unwrap();
    
    let result = pipeline.execute(&input_tensor).unwrap();
    
    assert_eq!(result.dtype(), BitNetDType::I4);
    assert_eq!(result.shape(), vec![8, 8]);
    assert!(result.size_bytes() < input_tensor.size_bytes());
    
    // Test pipeline statistics
    let stats = pipeline.get_stats().unwrap();
    assert!(stats.total_executions > 0);
}

/// Test Array conversion with edge cases
#[test]
fn test_array_conversion_edge_cases() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Test single element array
    let single_data = vec![42.0f32];
    let single_tensor = BitNetTensor::from_data(single_data, &[1], &device, &pool).unwrap();
    let converted_single = engine.convert(&single_tensor, BitNetDType::F16).unwrap();
    assert_eq!(converted_single.element_count(), 1);
    
    // Test very large 1D array
    let large_1d_data = vec![1.0f32; 1000000];
    let large_1d_tensor = BitNetTensor::from_data(large_1d_data, &[1000000], &device, &pool).unwrap();
    let converted_large_1d = engine.convert(&large_1d_tensor, BitNetDType::I8).unwrap();
    assert_eq!(converted_large_1d.element_count(), 1000000);
    
    // Test array with prime number dimensions
    let prime_data = vec![1.0f32; 77]; // 7 * 11
    let prime_tensor = BitNetTensor::from_data(prime_data, &[7, 11], &device, &pool).unwrap();
    let converted_prime = engine.convert(&prime_tensor, BitNetDType::F16).unwrap();
    assert_eq!(converted_prime.shape(), vec![7, 11]);
}

/// Test Array conversion statistics and metrics
#[test]
fn test_array_conversion_metrics() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Perform several conversions
    let arrays = vec![
        BitNetTensor::from_data(vec![1.0f32; 16], &[4, 4], &device, &pool).unwrap(),
        BitNetTensor::from_data(vec![2.0f32; 25], &[5, 5], &device, &pool).unwrap(),
        BitNetTensor::from_data(vec![3.0f32; 36], &[6, 6], &device, &pool).unwrap(),
    ];
    
    for array in &arrays {
        let _ = engine.convert(array, BitNetDType::F16).unwrap();
        let _ = engine.convert(array, BitNetDType::I8).unwrap();
    }
    
    // Check metrics
    let metrics = engine.get_stats();
    assert_eq!(metrics.total_conversions, 6); // 3 arrays * 2 conversions each
    assert_eq!(metrics.successful_conversions, 6);
    assert_eq!(metrics.success_rate(), 100.0);
    assert!(metrics.total_bytes_processed > 0);
    assert!(metrics.average_time_ms() >= 0.0);
}

/// Test Array conversion with reshape operations
#[test]
fn test_array_conversion_with_reshape() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    
    // Create 2x6 array
    let data = vec![1.0f32; 12];
    let tensor = BitNetTensor::from_data(data, &[2, 6], &device, &pool).unwrap();
    
    // Reshape to 3x4
    let reshaped = tensor.reshape(&[3, 4]).unwrap();
    assert_eq!(reshaped.shape(), vec![3, 4]);
    assert_eq!(reshaped.element_count(), 12);
    
    // Reshape to 1D
    let flattened = tensor.reshape(&[12]).unwrap();
    assert_eq!(flattened.shape(), vec![12]);
    assert_eq!(flattened.element_count(), 12);
    
    // Test invalid reshape
    let invalid_reshape = tensor.reshape(&[2, 5]); // 10 elements != 12
    assert!(invalid_reshape.is_err());
}

/// Performance benchmark for Array conversions
#[test]
fn test_array_conversion_performance() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::high_performance();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();
    
    // Create large arrays for performance testing
    let sizes = vec![
        (64, 64),    // 4K elements
        (128, 128),  // 16K elements
        (256, 256),  // 64K elements
    ];
    
    for (height, width) in sizes {
        let data = vec![1.0f32; height * width];
        let tensor = BitNetTensor::from_data(data, &[height, width], &device, &pool).unwrap();
        
        let start = std::time::Instant::now();
        let _converted = engine.convert(&tensor, BitNetDType::F16).unwrap();
        let duration = start.elapsed();
        
        // Performance should be reasonable (less than 100ms for these sizes)
        assert!(duration.as_millis() < 100, 
                "Conversion took too long: {}ms for {}x{}", 
                duration.as_millis(), height, width);
    }
}

#[cfg(feature = "mlx")]
mod mlx_array_tests {
    use super::*;
    use bitnet_core::mlx::{create_mlx_array, mlx_to_candle_tensor, candle_to_mlx_array};
    
    /// Test MLX Array to BitNetTensor conversion
    #[test]
    fn test_mlx_array_to_bitnet_tensor() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        
        // Create MLX array
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let mlx_array = create_mlx_array(&shape, data.clone()).unwrap();
        
        // Convert to Candle tensor
        let candle_tensor = mlx_to_candle_tensor(&mlx_array).unwrap();
        
        // Convert to BitNetTensor
        let bitnet_tensor = BitNetTensor::from_candle(candle_tensor, &pool).unwrap();
        
        assert_eq!(bitnet_tensor.shape(), vec![2, 3]);
        assert_eq!(bitnet_tensor.dtype(), BitNetDType::F32);
    }
    
    /// Test BitNetTensor to MLX Array conversion
    #[test]
    fn test_bitnet_tensor_to_mlx_array() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        
        // Create BitNetTensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bitnet_tensor = BitNetTensor::from_data(data, &[2, 2], &device, &pool).unwrap();
        
        // Convert to Candle tensor
        let candle_tensor = bitnet_tensor.to_candle().unwrap();
        
        // Convert to MLX array
        let mlx_array = candle_to_mlx_array(&candle_tensor).unwrap();
        
        assert_eq!(mlx_array.shape(), &[2, 2]);
    }
    
    /// Test round-trip MLX Array conversion
    #[test]
    fn test_mlx_array_round_trip() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        
        // MLX Array -> Candle Tensor -> BitNetTensor -> Candle Tensor -> MLX Array
        let mlx_array1 = create_mlx_array(&shape, original_data.clone()).unwrap();
        let candle_tensor1 = mlx_to_candle_tensor(&mlx_array1).unwrap();
        let bitnet_tensor = BitNetTensor::from_candle(candle_tensor1, &pool).unwrap();
        let candle_tensor2 = bitnet_tensor.to_candle().unwrap();
        let mlx_array2 = candle_to_mlx_array(&candle_tensor2).unwrap();
        
        assert_eq!(mlx_array1.shape(), mlx_array2.shape());
        
        // Verify data integrity
        let data1 = mlx_array1.as_slice::<f32>();
        let data2 = mlx_array2.as_slice::<f32>();
        assert_eq!(data1, data2);
    }
}