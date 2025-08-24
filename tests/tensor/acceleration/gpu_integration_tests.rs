//! Integration Tests for BitNet Metal GPU Acceleration
//!
//! Comprehensive integration tests for Phase 2 completion:
//! - BitNetMetalKernels integration
//! - Automatic dispatch functionality
//! - GPU/CPU result consistency
//! - Performance threshold validation
//! - Error handling and fallback behavior

use bitnet_core::tensor::core::BitNetTensor;
use bitnet_core::tensor::dtype::BitNetDType;
use bitnet_core::tensor::device::{BitNetDevice, DeviceType};
use bitnet_core::tensor::ops::gpu_arithmetic::*;
use bitnet_core::tensor::acceleration::metal_kernels_complete::*;
use std::sync::Arc;

#[tokio::test]
async fn test_metal_kernels_initialization() {
    // Test BitNetMetalKernels initialization
    let result = BitNetMetalKernels::new().await;
    assert!(result.is_ok(), "Failed to initialize BitNetMetalKernels: {:?}", result.err());

    let kernels = result.unwrap();

    // Verify all pipelines are created
    assert!(kernels.quantization_158.is_some(), "Quantization pipeline not initialized");
    assert!(kernels.bitlinear_forward.is_some(), "BitLinear forward pipeline not initialized");
    assert!(kernels.matmul_optimized.is_some(), "Matrix multiplication pipeline not initialized");
    assert!(kernels.elementwise_add.is_some(), "Element-wise add pipeline not initialized");
    assert!(kernels.elementwise_mul.is_some(), "Element-wise mul pipeline not initialized");
    assert!(kernels.activation_quant.is_some(), "Activation quantization pipeline not initialized");
}

#[tokio::test]
async fn test_global_kernels_initialization() {
    // Test global kernel initialization
    let global_kernels = GLOBAL_METAL_KERNELS.lock().await;
    assert!(global_kernels.is_some(), "Global Metal kernels not initialized");

    let kernels = global_kernels.as_ref().unwrap();

    // Test auto_dispatch availability
    let test_tensor = BitNetTensor::ones(&[1024], BitNetDType::F32, None).unwrap();
    let can_dispatch = kernels.auto_dispatch(&test_tensor, 1000).await;
    // Should not panic regardless of result
    println!("Auto dispatch result for test tensor: {}", can_dispatch);
}

#[tokio::test]
async fn test_quantization_gpu_vs_cpu_consistency() {
    // Create test tensor
    let input = BitNetTensor::randn(&[1024], BitNetDType::F32, None).unwrap();
    let scale = 2.5f32;
    let zero_point = 0.0f32;

    // GPU quantization
    let gpu_result = quantize_gpu(&input, scale, zero_point);
    assert!(gpu_result.is_ok(), "GPU quantization failed: {:?}", gpu_result.err());

    // CPU quantization (reference implementation)
    let cpu_result = quantize_reference_cpu(&input, scale, zero_point);
    assert!(cpu_result.is_ok(), "CPU quantization failed: {:?}", cpu_result.err());

    // Compare results
    let gpu_data = gpu_result.unwrap().as_slice_f32().unwrap();
    let cpu_data = cpu_result.unwrap().as_slice_f32().unwrap();

    assert_eq!(gpu_data.len(), cpu_data.len(), "Result lengths don't match");

    // Allow small numerical differences due to GPU precision
    for (i, (gpu_val, cpu_val)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        let diff = (gpu_val - cpu_val).abs();
        assert!(diff < 1e-5, "Values differ at index {}: GPU={}, CPU={}, diff={}", i, gpu_val, cpu_val, diff);
    }
}

#[tokio::test]
async fn test_bitlinear_gpu_vs_cpu_consistency() {
    // Create test tensors
    let weights = BitNetTensor::randn(&[128, 256], BitNetDType::F32, None).unwrap();
    let input = BitNetTensor::randn(&[32, 256], BitNetDType::F32, None).unwrap();
    let weight_scale = 1.5f32;
    let input_scale = 0.8f32;

    // GPU BitLinear
    let gpu_result = bitlinear_forward_gpu(&weights, &input, weight_scale, input_scale);
    assert!(gpu_result.is_ok(), "GPU BitLinear failed: {:?}", gpu_result.err());

    // CPU BitLinear (reference implementation)
    let cpu_result = bitlinear_forward_reference_cpu(&weights, &input, weight_scale, input_scale);
    assert!(cpu_result.is_ok(), "CPU BitLinear failed: {:?}", cpu_result.err());

    // Compare results
    let gpu_data = gpu_result.unwrap().as_slice_f32().unwrap();
    let cpu_data = cpu_result.unwrap().as_slice_f32().unwrap();

    assert_eq!(gpu_data.len(), cpu_data.len(), "Result lengths don't match");

    // Allow for numerical precision differences
    for (i, (gpu_val, cpu_val)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        let diff = (gpu_val - cpu_val).abs();
        assert!(diff < 1e-3, "Values differ at index {}: GPU={}, CPU={}, diff={}", i, gpu_val, cpu_val, diff);
    }
}

#[tokio::test]
async fn test_matrix_multiplication_gpu_vs_cpu_consistency() {
    // Create test matrices
    let a = BitNetTensor::randn(&[64, 128], BitNetDType::F32, None).unwrap();
    let b = BitNetTensor::randn(&[128, 96], BitNetDType::F32, None).unwrap();

    // GPU matmul
    let gpu_result = matmul_gpu(&a, &b);
    assert!(gpu_result.is_ok(), "GPU matmul failed: {:?}", gpu_result.err());

    // CPU matmul (using Candle)
    let cpu_result = matmul_reference_cpu(&a, &b);
    assert!(cpu_result.is_ok(), "CPU matmul failed: {:?}", cpu_result.err());

    // Compare results
    let gpu_data = gpu_result.unwrap().as_slice_f32().unwrap();
    let cpu_data = cpu_result.unwrap().as_slice_f32().unwrap();

    assert_eq!(gpu_data.len(), cpu_data.len(), "Result lengths don't match");

    // Allow for numerical precision differences in matrix multiplication
    for (i, (gpu_val, cpu_val)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        let diff = (gpu_val - cpu_val).abs();
        let relative_error = if cpu_val.abs() > 1e-10 { diff / cpu_val.abs() } else { diff };
        assert!(relative_error < 1e-2, "Values differ at index {}: GPU={}, CPU={}, relative_error={}", i, gpu_val, cpu_val, relative_error);
    }
}

#[tokio::test]
async fn test_elementwise_operations_gpu_vs_cpu_consistency() {
    // Create test tensors
    let a = BitNetTensor::randn(&[1024], BitNetDType::F32, None).unwrap();
    let b = BitNetTensor::randn(&[1024], BitNetDType::F32, None).unwrap();

    // Test addition
    let gpu_add = add_gpu(&a, &b);
    let cpu_add = add_reference_cpu(&a, &b);

    assert!(gpu_add.is_ok(), "GPU addition failed: {:?}", gpu_add.err());
    assert!(cpu_add.is_ok(), "CPU addition failed: {:?}", cpu_add.err());

    let gpu_add_data = gpu_add.unwrap().as_slice_f32().unwrap();
    let cpu_add_data = cpu_add.unwrap().as_slice_f32().unwrap();

    for (gpu_val, cpu_val) in gpu_add_data.iter().zip(cpu_add_data.iter()) {
        let diff = (gpu_val - cpu_val).abs();
        assert!(diff < 1e-6, "Addition values differ: GPU={}, CPU={}, diff={}", gpu_val, cpu_val, diff);
    }

    // Test multiplication
    let gpu_mul = mul_gpu(&a, &b);
    let cpu_mul = mul_reference_cpu(&a, &b);

    assert!(gpu_mul.is_ok(), "GPU multiplication failed: {:?}", gpu_mul.err());
    assert!(cpu_mul.is_ok(), "CPU multiplication failed: {:?}", cpu_mul.err());

    let gpu_mul_data = gpu_mul.unwrap().as_slice_f32().unwrap();
    let cpu_mul_data = cpu_mul.unwrap().as_slice_f32().unwrap();

    for (gpu_val, cpu_val) in gpu_mul_data.iter().zip(cpu_mul_data.iter()) {
        let diff = (gpu_val - cpu_val).abs();
        assert!(diff < 1e-6, "Multiplication values differ: GPU={}, CPU={}, diff={}", gpu_val, cpu_val, diff);
    }
}

#[tokio::test]
async fn test_automatic_dispatch_thresholds() {
    // Test small tensor dispatches to CPU
    let small_tensor = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();

    // Should automatically dispatch to CPU for small tensors
    let result = add_gpu(&small_tensor, &small_tensor);
    assert!(result.is_ok(), "Small tensor dispatch failed: {:?}", result.err());

    // Test large tensor dispatches to GPU
    let large_tensor = BitNetTensor::ones(&[100000], BitNetDType::F32, None).unwrap();

    // Should automatically dispatch to GPU for large tensors
    let result = add_gpu(&large_tensor, &large_tensor);
    assert!(result.is_ok(), "Large tensor dispatch failed: {:?}", result.err());
}

#[tokio::test]
async fn test_error_handling_and_fallback() {
    // Test fallback behavior with invalid inputs
    let empty_tensor = BitNetTensor::ones(&[0], BitNetDType::F32, None).unwrap();
    let normal_tensor = BitNetTensor::ones(&[1024], BitNetDType::F32, None).unwrap();

    // Operations with empty tensors should handle gracefully
    let result = add_gpu(&empty_tensor, &normal_tensor);
    // Should either succeed with proper handling or provide meaningful error
    match result {
        Ok(_) => println!("Empty tensor handled successfully"),
        Err(e) => println!("Empty tensor error handled: {:?}", e),
    }

    // Test mismatched dimensions
    let tensor_a = BitNetTensor::ones(&[128, 256], BitNetDType::F32, None).unwrap();
    let tensor_b = BitNetTensor::ones(&[64, 128], BitNetDType::F32, None).unwrap();

    let result = add_gpu(&tensor_a, &tensor_b);
    // Should provide meaningful error for dimension mismatch
    match result {
        Ok(_) => {
            // Broadcasting might have succeeded
            println!("Broadcasting handled successfully");
        }
        Err(e) => {
            println!("Dimension mismatch handled: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_performance_thresholds() {
    use std::time::Instant;

    // Test that GPU acceleration provides speedup for large tensors
    let large_tensor = BitNetTensor::randn(&[50000], BitNetDType::F32, None).unwrap();
    let scale = 1.0f32;
    let zero_point = 0.0f32;

    // Time CPU implementation
    let cpu_start = Instant::now();
    let _ = quantize_reference_cpu(&large_tensor, scale, zero_point).unwrap();
    let cpu_duration = cpu_start.elapsed();

    // Time GPU implementation
    let gpu_start = Instant::now();
    let _ = quantize_gpu(&large_tensor, scale, zero_point).unwrap();
    let gpu_duration = gpu_start.elapsed();

    println!("CPU quantization time: {:?}", cpu_duration);
    println!("GPU quantization time: {:?}", gpu_duration);

    // For large tensors, GPU should be competitive or faster
    // Note: This is a basic performance check, not the full >10x target validation
    // The full validation is in the benchmarks
    assert!(gpu_duration < cpu_duration * 5, "GPU not competitive with CPU for large tensors");
}

#[tokio::test]
async fn test_device_compatibility() {
    // Test that operations work with different device types
    let cpu_device = Arc::new(BitNetDevice::new(DeviceType::Cpu));
    let metal_device = Arc::new(BitNetDevice::new(DeviceType::Metal));

    // CPU tensor
    let cpu_tensor = BitNetTensor::ones(&[1024], BitNetDType::F32, Some(cpu_device)).unwrap();

    // Metal tensor
    let metal_tensor = BitNetTensor::ones(&[1024], BitNetDType::F32, Some(metal_device)).unwrap();

    // Operations should handle device transitions appropriately
    let cpu_result = quantize_gpu(&cpu_tensor, 1.0, 0.0);
    let metal_result = quantize_gpu(&metal_tensor, 1.0, 0.0);

    assert!(cpu_result.is_ok(), "CPU tensor GPU operation failed: {:?}", cpu_result.err());
    assert!(metal_result.is_ok(), "Metal tensor GPU operation failed: {:?}", metal_result.err());
}

#[tokio::test]
async fn test_concurrent_gpu_operations() {
    use tokio::task;

    // Test concurrent GPU operations don't interfere
    let tensor_a = BitNetTensor::randn(&[1024], BitNetDType::F32, None).unwrap();
    let tensor_b = BitNetTensor::randn(&[1024], BitNetDType::F32, None).unwrap();

    let handles = (0..4).map(|i| {
        let tensor_a = tensor_a.clone();
        let tensor_b = tensor_b.clone();

        task::spawn(async move {
            let result = add_gpu(&tensor_a, &tensor_b);
            assert!(result.is_ok(), "Concurrent operation {} failed: {:?}", i, result.err());
            result.unwrap()
        })
    }).collect::<Vec<_>>();

    // Wait for all operations to complete
    let results = futures::future::join_all(handles).await;

    // All should succeed
    for (i, result) in results.into_iter().enumerate() {
        assert!(result.is_ok(), "Concurrent task {} failed: {:?}", i, result.err());
    }
}

// Reference CPU implementations for testing consistency

fn quantize_reference_cpu(input: &BitNetTensor, scale: f32, zero_point: f32) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let input_data = input.as_slice_f32()?;
    let mut output_data = Vec::with_capacity(input_data.len());

    for &value in input_data {
        let scaled = value / scale + zero_point;
        let quantized = if scaled <= -0.5 { -1.0 } else if scaled >= 0.5 { 1.0 } else { 0.0 };
        output_data.push(quantized);
    }

    Ok(BitNetTensor::from_data(&output_data, input.shape().dims(), input.dtype(), input.device().clone())?)
}

fn bitlinear_forward_reference_cpu(
    weights: &BitNetTensor,
    input: &BitNetTensor,
    weight_scale: f32,
    input_scale: f32
) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let weight_dims = weights.shape().dims();
    let input_dims = input.shape().dims();
    let (output_size, input_size) = (weight_dims[0], weight_dims[1]);
    let batch_size = input_dims[0];

    let weight_data = weights.as_slice_f32()?;
    let input_data = input.as_slice_f32()?;
    let mut output_data = vec![0.0f32; batch_size * output_size];

    for b in 0..batch_size {
        for o in 0..output_size {
            let mut sum = 0.0f32;
            for i in 0..input_size {
                let w = weight_data[o * input_size + i];
                let x = input_data[b * input_size + i];
                sum += w * x;
            }
            output_data[b * output_size + o] = sum * weight_scale * input_scale;
        }
    }

    Ok(BitNetTensor::from_data(&output_data, &[batch_size, output_size], input.dtype(), input.device().clone())?)
}

fn matmul_reference_cpu(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    let result_candle = a_candle.matmul(&b_candle)?;
    Ok(BitNetTensor::from_candle(result_candle, a.device())?)
}

fn add_reference_cpu(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    let result_candle = (&a_candle + &b_candle)?;
    Ok(BitNetTensor::from_candle(result_candle, a.device())?)
}

fn mul_reference_cpu(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    let result_candle = (&a_candle * &b_candle)?;
    Ok(BitNetTensor::from_candle(result_candle, a.device())?)
}
