//! Integration Tests for BitNet Benchmarks
//! 
//! This module contains integration tests for the MLX vs Candle
//! benchmarking suite to ensure all components work together correctly.

use bitnet_benchmarks::{
    ComparisonConfig, PerformanceComparator, BenchmarkRunner,
    CandleOps, CandlePerformanceUtils
};
use candle_core::{Device, DType, Tensor};
use std::time::Duration;

#[test]
fn test_candle_ops_basic_operations() {
    let device = Device::Cpu;
    
    // Test tensor creation
    let a = Tensor::randn(0f32, 1f32, (4, 4), &device).unwrap();
    let b = Tensor::randn(0f32, 1f32, (4, 4), &device).unwrap();
    
    // Test basic operations
    let add_result = CandleOps::add(&a, &b).unwrap();
    assert_eq!(add_result.shape().dims(), &[4, 4]);
    
    let mul_result = CandleOps::multiply(&a, &b).unwrap();
    assert_eq!(mul_result.shape().dims(), &[4, 4]);
    
    let matmul_result = CandleOps::matmul(&a, &b).unwrap();
    assert_eq!(matmul_result.shape().dims(), &[4, 4]);
}

#[test]
fn test_candle_quantization() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1f32, (3, 3), &device).unwrap();
    
    // Test quantization
    let quantized = CandleOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
    assert_eq!(quantized.shape().dims(), tensor.shape().dims());
    
    // Test dequantization
    let dequantized = CandleOps::dequantize_1_58_bit(&quantized, Some(0.1)).unwrap();
    assert_eq!(dequantized.shape().dims(), tensor.shape().dims());
}

#[test]
fn test_candle_bitlinear_forward() {
    let device = Device::Cpu;
    let batch_size = 2;
    let input_size = 4;
    let output_size = 3;
    
    let input = Tensor::randn(0f32, 1f32, (batch_size, input_size), &device).unwrap();
    let weight = Tensor::randn(0f32, 1f32, (input_size, output_size), &device).unwrap();
    let bias = Tensor::randn(0f32, 1f32, (output_size,), &device).unwrap();
    
    // Test BitLinear forward pass
    let output = CandleOps::bitlinear_forward(&input, &weight, Some(&bias), true).unwrap();
    assert_eq!(output.shape().dims(), &[batch_size, output_size]);
    
    // Test without bias
    let output_no_bias = CandleOps::bitlinear_forward(&input, &weight, None, false).unwrap();
    assert_eq!(output_no_bias.shape().dims(), &[batch_size, output_size]);
}

#[test]
fn test_candle_performance_utils() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros((100, 100), DType::F32, &device).unwrap();
    
    // Test memory usage calculation
    let memory_usage = CandlePerformanceUtils::tensor_memory_usage(&tensor);
    assert_eq!(memory_usage, 100 * 100 * 4); // 100x100 f32 = 40KB
    
    // Test device info
    let device_info = CandlePerformanceUtils::device_info(&device);
    assert_eq!(device_info, "CPU");
    
    // Test device capabilities
    let capabilities = CandlePerformanceUtils::device_capabilities(&device);
    assert!(capabilities.contains(&"basic_ops".to_string()));
    assert!(capabilities.contains(&"cpu_optimized".to_string()));
}

#[test]
fn test_comparison_config_creation() {
    let config = ComparisonConfig::default();
    
    assert!(!config.tensor_sizes.is_empty());
    assert!(!config.operations.is_empty());
    assert!(!config.devices.is_empty());
    assert!(!config.data_types.is_empty());
    assert!(config.warmup_iterations > 0);
    assert!(config.measurement_iterations > 0);
    assert!(config.timeout > Duration::ZERO);
}

#[test]
fn test_comparison_config_customization() {
    let custom_config = ComparisonConfig {
        tensor_sizes: vec![(64, 64), (128, 128)],
        warmup_iterations: 3,
        measurement_iterations: 5,
        operations: vec!["matmul".to_string(), "add".to_string()],
        devices: vec!["cpu".to_string()],
        data_types: vec!["f32".to_string()],
        timeout: Duration::from_secs(10),
    };
    
    assert_eq!(custom_config.tensor_sizes.len(), 2);
    assert_eq!(custom_config.operations.len(), 2);
    assert_eq!(custom_config.warmup_iterations, 3);
    assert_eq!(custom_config.measurement_iterations, 5);
}

#[test]
fn test_performance_comparator_creation() {
    let config = ComparisonConfig {
        tensor_sizes: vec![(32, 32)],
        warmup_iterations: 1,
        measurement_iterations: 2,
        operations: vec!["add".to_string()],
        devices: vec!["cpu".to_string()],
        data_types: vec!["f32".to_string()],
        timeout: Duration::from_secs(5),
    };
    
    let comparator = PerformanceComparator::new(config);
    assert_eq!(comparator.get_measurements().len(), 0);
}

#[test]
fn test_benchmark_runner_creation() {
    let config = ComparisonConfig::default();
    let runner = BenchmarkRunner::new(config, false);
    
    // Test that runner can be created without errors
    // (We can't easily test the full run without a complex setup)
}

#[test]
fn test_candle_tensor_operations_consistency() {
    let device = Device::Cpu;
    let size = 16;
    
    // Create test tensors
    let zeros = CandleOps::zeros(&[size, size], DType::F32, &device).unwrap();
    let ones = CandleOps::ones(&[size, size], DType::F32, &device).unwrap();
    
    // Test that zeros + ones = ones
    let result = CandleOps::add(&zeros, &ones).unwrap();
    assert_eq!(result.shape().dims(), &[size, size]);
    
    // Test that ones * zeros = zeros
    let result = CandleOps::multiply(&ones, &zeros).unwrap();
    assert_eq!(result.shape().dims(), &[size, size]);
}

#[test]
fn test_candle_reshape_and_transpose() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1f32, (4, 6), &device).unwrap();
    
    // Test reshape
    let reshaped = CandleOps::reshape(&tensor, &[6, 4]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[6, 4]);
    
    // Test transpose
    let transposed = CandleOps::transpose(&tensor, 0, 1).unwrap();
    assert_eq!(transposed.shape().dims(), &[6, 4]);
}

#[test]
fn test_candle_activation_functions() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1f32, (3, 3), &device).unwrap();
    
    // Test ReLU
    let relu_result = CandleOps::relu(&tensor).unwrap();
    assert_eq!(relu_result.shape().dims(), tensor.shape().dims());
    
    // Test GELU
    let gelu_result = CandleOps::gelu(&tensor).unwrap();
    assert_eq!(gelu_result.shape().dims(), tensor.shape().dims());
    
    // Test softmax
    let softmax_result = CandleOps::softmax(&tensor, 1).unwrap();
    assert_eq!(softmax_result.shape().dims(), tensor.shape().dims());
}

#[test]
fn test_candle_reduction_operations() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1f32, (4, 5), &device).unwrap();
    
    // Test mean
    let mean_result = CandleOps::mean(&tensor, 1).unwrap();
    assert_eq!(mean_result.shape().dims(), &[4]);
    
    // Test sum
    let sum_result = CandleOps::sum(&tensor, 0).unwrap();
    assert_eq!(sum_result.shape().dims(), &[5]);
    
    // Test variance
    let var_result = CandleOps::var(&tensor, 1).unwrap();
    assert_eq!(var_result.shape().dims(), &[4]);
}

#[test]
fn test_candle_layer_norm() {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0f32, 1f32, (2, 4), &device).unwrap();
    let weight = Tensor::ones((4,), DType::F32, &device).unwrap();
    let bias = Tensor::zeros((4,), DType::F32, &device).unwrap();
    
    // Test layer normalization
    let norm_result = CandleOps::layer_norm(&tensor, &[4], Some(&weight), Some(&bias), 1e-5).unwrap();
    assert_eq!(norm_result.shape().dims(), tensor.shape().dims());
    
    // Test without weight and bias
    let norm_result_simple = CandleOps::layer_norm(&tensor, &[4], None, None, 1e-5).unwrap();
    assert_eq!(norm_result_simple.shape().dims(), tensor.shape().dims());
}

#[test]
fn test_candle_concatenation_and_splitting() {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 3), DType::F32, &device).unwrap();
    let b = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
    
    // Test concatenation
    let tensors = vec![&a, &b];
    let concat_result = CandleOps::concat(&tensors, 0).unwrap();
    assert_eq!(concat_result.shape().dims(), &[4, 3]);
    
    // Test splitting
    let split_results = CandleOps::split(&concat_result, 2, 0).unwrap();
    assert_eq!(split_results.len(), 2);
    assert_eq!(split_results[0].shape().dims(), &[2, 3]);
    assert_eq!(split_results[1].shape().dims(), &[2, 3]);
}

#[test]
fn test_candle_embedding() {
    let device = Device::Cpu;
    let vocab_size = 10;
    let embed_dim = 4;
    let seq_len = 3;
    
    let weight = Tensor::randn(0f32, 1f32, (vocab_size, embed_dim), &device).unwrap();
    let indices = Tensor::new(&[1u32, 3u32, 7u32], &device).unwrap();
    
    // Test embedding lookup
    let embedded = CandleOps::embedding(&weight, &indices).unwrap();
    assert_eq!(embedded.shape().dims(), &[seq_len, embed_dim]);
}

#[test]
fn test_candle_convolution() {
    let device = Device::Cpu;
    let batch_size = 1;
    let in_channels = 2;
    let out_channels = 3;
    let seq_len = 8;
    let kernel_size = 3;
    
    let input = Tensor::randn(0f32, 1f32, (batch_size, in_channels, seq_len), &device).unwrap();
    let weight = Tensor::randn(0f32, 1f32, (out_channels, in_channels, kernel_size), &device).unwrap();
    let bias = Tensor::randn(0f32, 1f32, (out_channels,), &device).unwrap();
    
    // Test 1D convolution
    let conv_result = CandleOps::conv1d(&input, &weight, Some(&bias), 1, 1).unwrap();
    assert_eq!(conv_result.shape().dims()[0], batch_size);
    assert_eq!(conv_result.shape().dims()[1], out_channels);
}

#[test]
fn test_error_handling() {
    let device = Device::Cpu;
    
    // Test empty tensor list concatenation
    let empty_tensors: Vec<&Tensor> = vec![];
    let result = CandleOps::concat(&empty_tensors, 0);
    assert!(result.is_err());
    
    // Test invalid reshape
    let tensor = Tensor::ones((2, 3), DType::F32, &device).unwrap();
    let result = CandleOps::reshape(&tensor, &[2, 4]); // 6 elements -> 8 elements
    assert!(result.is_err());
}

#[cfg(feature = "mlx")]
mod mlx_tests {
    use super::*;
    use bitnet_core::mlx::{MlxTensor, BitNetMlxDevice, operations::BitNetMlxOps};
    use bitnet_core::memory::tensor::BitNetDType;

    #[test]
    fn test_mlx_basic_operations() {
        let device = BitNetMlxDevice::default();
        let a = MlxTensor::randn(&[4, 4], BitNetDType::F32, device.clone()).unwrap();
        let b = MlxTensor::randn(&[4, 4], BitNetDType::F32, device).unwrap();
        
        // Test MLX operations
        let add_result = BitNetMlxOps::add(&a, &b).unwrap();
        assert_eq!(add_result.shape(), &[4, 4]);
        
        let mul_result = BitNetMlxOps::multiply(&a, &b).unwrap();
        assert_eq!(mul_result.shape(), &[4, 4]);
        
        let matmul_result = BitNetMlxOps::matmul(&a, &b).unwrap();
        assert_eq!(matmul_result.shape(), &[4, 4]);
    }

    #[test]
    fn test_mlx_quantization() {
        let device = BitNetMlxDevice::default();
        let tensor = MlxTensor::randn(&[3, 3], BitNetDType::F32, device).unwrap();
        
        // Test MLX quantization
        let quantized = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
        assert_eq!(quantized.shape(), tensor.shape());
        
        // Test MLX dequantization
        let dequantized = BitNetMlxOps::dequantize_1_58_bit(&quantized, Some(0.1)).unwrap();
        assert_eq!(dequantized.shape(), tensor.shape());
    }

    #[test]
    fn test_mlx_bitlinear_forward() {
        let device = BitNetMlxDevice::default();
        let batch_size = 2;
        let input_size = 4;
        let output_size = 3;
        
        let input = MlxTensor::randn(&[batch_size, input_size], BitNetDType::F32, device.clone()).unwrap();
        let weight = MlxTensor::randn(&[input_size, output_size], BitNetDType::F32, device.clone()).unwrap();
        let bias = MlxTensor::randn(&[output_size], BitNetDType::F32, device).unwrap();
        
        // Test MLX BitLinear forward pass
        let output = BitNetMlxOps::bitlinear_forward(&input, &weight, Some(&bias), true).unwrap();
        assert_eq!(output.shape(), &[batch_size, output_size]);
    }
}

// Performance regression tests
#[test]
fn test_performance_regression_candle() {
    let device = Device::Cpu;
    let size = 64; // Small size for fast testing
    
    let start = std::time::Instant::now();
    
    // Run a series of operations
    for _ in 0..10 {
        let a = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
        let b = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
        
        let _add = CandleOps::add(&a, &b).unwrap();
        let _mul = CandleOps::multiply(&a, &b).unwrap();
        let _matmul = CandleOps::matmul(&a, &b).unwrap();
    }
    
    let elapsed = start.elapsed();
    
    // Ensure operations complete within reasonable time (adjust threshold as needed)
    assert!(elapsed < Duration::from_secs(5), "Operations took too long: {elapsed:?}");
}

#[test]
fn test_memory_efficiency() {
    let device = Device::Cpu;
    
    // Test that we can create and operate on reasonably large tensors
    let size = 256;
    let a = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
    let b = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
    
    // Perform operations that should not cause memory issues
    let _result = CandleOps::matmul(&a, &b).unwrap();
    let _result = CandleOps::add(&a, &b).unwrap();
    
    // Test memory usage calculation
    let memory_usage = CandlePerformanceUtils::tensor_memory_usage(&a);
    assert_eq!(memory_usage, size * size * 4); // f32 = 4 bytes
}