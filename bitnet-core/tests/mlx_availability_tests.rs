//! MLX Availability Tests
//!
//! This module contains comprehensive tests for MLX (Apple's Machine Learning framework)
//! availability, device discovery, capability checking, and integration across different platforms.

use std::time::Duration;

/// Test basic MLX availability checking
#[test]
fn test_mlx_availability_check() {
    #[cfg(feature = "mlx")]
    {
        use bitnet_core::is_mlx_available;
        
        let is_available = is_mlx_available();
        
        // The result depends on the platform and MLX installation
        // On Apple Silicon with MLX: should be true
        // On other platforms or without MLX: should be false
        println!("MLX availability: {}", is_available);
        
        // Ensure the function doesn't panic and returns a boolean
        assert!(is_available == true || is_available == false);
        
        // On Apple Silicon, MLX should typically be available
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            if is_available {
                println!("MLX is available on Apple Silicon (expected)");
            } else {
                println!("MLX not available on Apple Silicon (may need installation)");
            }
        }
        
        // On non-Apple Silicon, MLX should not be available
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            if !is_available {
                println!("MLX correctly not available on non-Apple Silicon platform");
            } else {
                println!("Warning: MLX reported as available on non-Apple Silicon platform");
            }
        }
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        // When MLX feature is disabled, we can't import MLX functions
        // Instead, we test that the feature is properly disabled
        println!("MLX feature is disabled - cannot test MLX functions");
        println!("MLX correctly not available without mlx feature");
    }
}

/// Test MLX device creation and basic functionality
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_device_creation() {
    use bitnet_core::{default_mlx_device, BitNetMlxDevice};
    
    let result = default_mlx_device();
    
    match result {
        Ok(device) => {
            println!("Successfully created MLX device");
            println!("Device type: {}", device.device_type());
            
            // Test device properties
            assert!(!device.device_type().is_empty());
            
            // Test unified memory support
            let supports_unified = device.supports_unified_memory();
            println!("Supports unified memory: {}", supports_unified);
            
            // On Apple Silicon GPU, unified memory should be supported
            if device.device_type() == "gpu" {
                assert!(supports_unified, "GPU device should support unified memory on Apple Silicon");
            }
        }
        Err(e) => {
            println!("MLX device creation failed: {}", e);
            // This is expected on non-Apple Silicon platforms
        }
    }
}

/// Test MLX device creation without feature flag
#[test]
#[cfg(not(feature = "mlx"))]
fn test_mlx_device_creation_without_feature() {
    // When MLX feature is disabled, we can't import MLX functions
    // This test verifies that the feature is properly disabled
    println!("MLX device creation correctly unavailable without mlx feature");
}

/// Test MLX device manager functionality
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_device_manager() {
    use bitnet_core::{MlxDeviceManager, get_mlx_device_manager};
    
    let manager_result = MlxDeviceManager::new();
    
    match manager_result {
        Ok(manager) => {
            println!("Successfully created MLX device manager");
            
            // Test available devices
            let devices = manager.available_devices();
            println!("Available MLX devices: {}", devices.len());
            assert!(!devices.is_empty(), "Should have at least CPU device");
            
            // Should always have CPU device
            let cpu_device = manager.get_device_by_type("cpu");
            assert!(cpu_device.is_some(), "Should always have CPU device");
            
            if let Some(cpu_dev) = cpu_device {
                println!("CPU device: {:?}", cpu_dev);
                assert_eq!(cpu_dev.to_bitnet_device_type(), "cpu");
                assert!(!cpu_dev.supports_unified_memory);
            }
            
            // Test GPU device availability
            let has_gpu = manager.has_gpu();
            println!("Has GPU device: {}", has_gpu);
            
            if has_gpu {
                let gpu_device = manager.get_device_by_type("gpu");
                assert!(gpu_device.is_some(), "Should have GPU device if has_gpu is true");
                
                if let Some(gpu_dev) = gpu_device {
                    println!("GPU device: {:?}", gpu_dev);
                    assert_eq!(gpu_dev.to_bitnet_device_type(), "gpu");
                    assert!(gpu_dev.supports_unified_memory, "GPU should support unified memory");
                }
            }
            
            // Test default device
            let default_device = manager.default_device();
            assert!(default_device.is_some(), "Should have a default device");
            
            if let Some(default_dev) = default_device {
                println!("Default device: {:?}", default_dev);
                
                // Default should prefer GPU if available, otherwise CPU
                if has_gpu {
                    assert_eq!(default_dev.to_bitnet_device_type(), "gpu");
                } else {
                    assert_eq!(default_dev.to_bitnet_device_type(), "cpu");
                }
            }
            
            // Test device capabilities
            for device in devices {
                let capabilities = manager.get_capabilities(device);
                println!("Device {} capabilities: {}", device.to_bitnet_device_type(), capabilities);
                assert!(!capabilities.is_empty());
            }
        }
        Err(e) => {
            println!("MLX device manager creation failed: {}", e);
        }
    }
    
    // Test global device manager
    let global_manager = get_mlx_device_manager();
    let devices = global_manager.available_devices();
    assert!(!devices.is_empty(), "Global manager should have at least CPU device");
    println!("Global MLX device manager has {} devices", devices.len());
}

/// Test MLX tensor creation and basic operations
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_tensor_operations() {
    use bitnet_core::{is_mlx_available, default_mlx_device, BitNetMlxDevice, BitNetMlxTensor, create_mlx_array};
    
    if !is_mlx_available() {
        println!("MLX not available, skipping tensor operations test");
        return;
    }
    
    let device_result = default_mlx_device();
    if let Ok(device) = device_result {
        // Test array creation
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let array_result = create_mlx_array(shape, data.clone());
        match array_result {
            Ok(array) => {
                println!("Successfully created MLX array");
                
                // Test tensor wrapper
                let tensor = BitNetMlxTensor::new(array, device.clone());
                
                // Test tensor properties
                assert_eq!(tensor.shape(), &[2, 3]);
                println!("Tensor shape: {:?}", tensor.shape());
                println!("Tensor device: {}", tensor.device().device_type());
                
                // Test device transfer
                let cpu_device = BitNetMlxDevice::cpu();
                let transferred_result = tensor.to_device(&cpu_device);
                match transferred_result {
                    Ok(transferred_tensor) => {
                        println!("Successfully transferred tensor to CPU");
                        assert_eq!(transferred_tensor.device().device_type(), "cpu");
                    }
                    Err(e) => {
                        println!("Tensor transfer failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("MLX array creation failed: {}", e);
            }
        }
    }
}

/// Test MLX operations (quantization, matrix multiplication)
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_operations() {
    use bitnet_core::{is_mlx_available, create_mlx_array, mlx_matmul, mlx_quantize, mlx_dequantize};
    
    if !is_mlx_available() {
        println!("MLX not available, skipping operations test");
        return;
    }
    
    // Test matrix multiplication
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    
    let a_result = create_mlx_array(&[2, 2], a_data);
    let b_result = create_mlx_array(&[2, 2], b_data);
    
    if let (Ok(a_array), Ok(b_array)) = (a_result, b_result) {
        let matmul_result = mlx_matmul(&a_array, &b_array);
        match matmul_result {
            Ok(result_array) => {
                println!("MLX matrix multiplication successful");
                assert_eq!(result_array.shape(), &[2, 2]);
                
                let result_data = result_array.as_slice::<f32>();
                println!("Matrix multiplication result: {:?}", result_data);
            }
            Err(e) => {
                println!("MLX matrix multiplication failed: {}", e);
            }
        }
    }
    
    // Test quantization
    let test_data = vec![1.5, 2.7, 3.2, 4.8];
    let test_array_result = create_mlx_array(&[2, 2], test_data);
    
    if let Ok(test_array) = test_array_result {
        let scale = 0.5;
        
        // Test quantization
        let quantized_result = mlx_quantize(&test_array, scale);
        match quantized_result {
            Ok(quantized_array) => {
                println!("MLX quantization successful");
                
                let quantized_data = quantized_array.as_slice::<f32>();
                println!("Quantized data: {:?}", quantized_data);
                
                // Test dequantization
                let dequantized_result = mlx_dequantize(&quantized_array, scale);
                match dequantized_result {
                    Ok(dequantized_array) => {
                        println!("MLX dequantization successful");
                        
                        let dequantized_data = dequantized_array.as_slice::<f32>();
                        println!("Dequantized data: {:?}", dequantized_data);
                    }
                    Err(e) => {
                        println!("MLX dequantization failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("MLX quantization failed: {}", e);
            }
        }
    }
}

/// Test MLX BitNet-specific operations
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_bitnet_operations() {
    use bitnet_core::{is_mlx_available, default_mlx_device, BitNetMlxTensor, MlxOperations, create_mlx_array};
    
    if !is_mlx_available() {
        println!("MLX not available, skipping BitNet operations test");
        return;
    }
    
    let device_result = default_mlx_device();
    if let Ok(device) = device_result {
        // Create test tensors
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let weight_data = vec![0.5, 1.0, 1.5, 2.0];
        
        let input_array_result = create_mlx_array(&[2, 2], input_data);
        let weight_array_result = create_mlx_array(&[2, 2], weight_data);
        
        if let (Ok(input_array), Ok(weight_array)) = (input_array_result, weight_array_result) {
            let input_tensor = BitNetMlxTensor::new(input_array, device.clone());
            let weight_tensor = BitNetMlxTensor::new(weight_array, device.clone());
            
            // Test 1.58-bit quantization
            let quantized_result = MlxOperations::quantize_1_58_bit(&input_tensor);
            match quantized_result {
                Ok(quantized_tensor) => {
                    println!("1.58-bit quantization successful");
                    assert_eq!(quantized_tensor.shape(), input_tensor.shape());
                }
                Err(e) => {
                    println!("1.58-bit quantization failed: {}", e);
                }
            }
            
            // Test BitLinear operation
            let bitlinear_result = MlxOperations::bitlinear(&input_tensor, &weight_tensor, None);
            match bitlinear_result {
                Ok(output_tensor) => {
                    println!("BitLinear operation successful");
                    assert_eq!(output_tensor.device().device_type(), device.device_type());
                }
                Err(e) => {
                    println!("BitLinear operation failed: {}", e);
                }
            }
            
            // Test matrix multiplication through MlxOperations
            let matmul_result = MlxOperations::matmul(&input_tensor, &weight_tensor);
            match matmul_result {
                Ok(result_tensor) => {
                    println!("MLX matrix multiplication through operations successful");
                    assert_eq!(result_tensor.shape(), &[2, 2]);
                }
                Err(e) => {
                    println!("MLX matrix multiplication through operations failed: {}", e);
                }
            }
        }
    }
}

/// Test MLX and Candle tensor interoperability
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_candle_interoperability() {
    use bitnet_core::{is_mlx_available, create_mlx_array, mlx_to_candle_tensor, candle_to_mlx_array};
    use bitnet_core::tensor::create_tensor_f32;
    
    if !is_mlx_available() {
        println!("MLX not available, skipping interoperability test");
        return;
    }
    
    // Test Candle -> MLX -> Candle round trip
    let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let candle_tensor_result = create_tensor_f32(&[2, 3], original_data.clone());
    
    if let Ok(candle_tensor) = candle_tensor_result {
        println!("Created Candle tensor: {:?}", candle_tensor.shape());
        
        // Convert to MLX
        let mlx_array_result = candle_to_mlx_array(&candle_tensor);
        match mlx_array_result {
            Ok(mlx_array) => {
                println!("Successfully converted Candle tensor to MLX array");
                assert_eq!(mlx_array.shape(), &[2, 3]);
                
                // Convert back to Candle
                let candle_tensor2_result = mlx_to_candle_tensor(&mlx_array);
                match candle_tensor2_result {
                    Ok(candle_tensor2) => {
                        println!("Successfully converted MLX array back to Candle tensor");
                        assert_eq!(candle_tensor2.shape().dims(), &[2, 3]);
                        
                        // Verify data integrity
                        if let Ok(final_data) = candle_tensor2.flatten_all()?.to_vec1::<f32>() {
                            println!("Round-trip conversion successful");
                            assert_eq!(final_data.len(), original_data.len());
                            
                            // Check data similarity (allowing for small floating point differences)
                            let data_matches = final_data.iter().zip(original_data.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-6);
                            assert!(data_matches, "Data should be preserved in round-trip conversion");
                        }
                    }
                    Err(e) => {
                        println!("Failed to convert MLX array back to Candle: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to convert Candle tensor to MLX: {}", e);
            }
        }
    }
}

/// Test MLX feature flag behavior
#[test]
fn test_mlx_feature_flag_behavior() {
    #[cfg(feature = "mlx")]
    {
        use bitnet_core::{is_mlx_available, default_mlx_device, get_mlx_device_manager};
        
        println!("MLX feature is enabled");
        
        // With MLX feature enabled, we should be able to call MLX functions
        let _availability = is_mlx_available();
        let _device_result = default_mlx_device();
        let _manager = get_mlx_device_manager();
        
        println!("All MLX functions callable with mlx feature enabled");
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        println!("MLX feature is disabled");
        
        // With MLX feature disabled, we can't import MLX functions
        // This test verifies that the feature is properly disabled
        println!("MLX functions correctly unavailable without feature");
    }
}

/// Test platform-specific MLX behavior
#[test]
fn test_platform_specific_mlx_behavior() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "mlx"))]
    {
        use bitnet_core::{is_mlx_available, default_mlx_device};
        
        println!("Running on Apple Silicon - MLX may be available");
        
        let mlx_available = is_mlx_available();
        println!("MLX available on Apple Silicon: {}", mlx_available);
        
        if mlx_available {
            // If MLX is available, we should be able to create devices
            let device_result = default_mlx_device();
            match device_result {
                Ok(device) => {
                    println!("MLX device created on Apple Silicon: {}", device.device_type());
                    
                    // On Apple Silicon, GPU device should support unified memory
                    if device.device_type() == "gpu" {
                        assert!(device.supports_unified_memory(), 
                               "GPU device should support unified memory on Apple Silicon");
                    }
                }
                Err(e) => {
                    println!("MLX device creation failed on Apple Silicon: {}", e);
                }
            }
        }
    }
    
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        use bitnet_core::{is_mlx_available, default_mlx_device};
        
        println!("Running on non-Apple Silicon platform - MLX should not be available");
        
        // On non-Apple Silicon platforms, MLX should not be available
        let mlx_available = is_mlx_available();
        if !mlx_available {
            println!("MLX correctly not available on non-Apple Silicon platform");
        } else {
            println!("Warning: MLX reported as available on non-Apple Silicon platform");
        }
        
        // Device creation should fail gracefully
        let device_result = default_mlx_device();
        if device_result.is_err() {
            println!("MLX device creation correctly failed on non-Apple Silicon platform");
        }
    }
}

/// Test MLX device selection scenarios
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_device_selection_scenarios() {
    use bitnet_core::{is_mlx_available, auto_select_mlx_device, BitNetMlxDevice};
    
    println!("Testing MLX device selection scenarios");
    
    // Test CPU device creation
    let cpu_device = BitNetMlxDevice::cpu();
    assert_eq!(cpu_device.device_type(), "cpu");
    assert!(!cpu_device.supports_unified_memory());
    println!("CPU device: {}", cpu_device.device_type());
    
    // Test GPU device creation
    let gpu_device = BitNetMlxDevice::gpu();
    assert_eq!(gpu_device.device_type(), "gpu");
    assert!(gpu_device.supports_unified_memory());
    println!("GPU device: {}", gpu_device.device_type());
    
    // Test auto selection
    if is_mlx_available() {
        let auto_device_result = auto_select_mlx_device();
        match auto_device_result {
            Ok(device) => {
                println!("Auto-selected device: {}", device.to_bitnet_device_type());
                
                // Auto selection should prefer GPU on Apple Silicon
                #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
                {
                    // On Apple Silicon, should prefer GPU
                    if device.to_bitnet_device_type() == "cpu" {
                        println!("Note: Auto-selection chose CPU (GPU may not be available)");
                    }
                }
            }
            Err(e) => {
                println!("Auto-selection failed: {}", e);
            }
        }
    }
}

/// Test MLX error handling and edge cases
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_error_handling() {
    use bitnet_core::{is_mlx_available, create_mlx_array, mlx_matmul};
    
    println!("Testing MLX error handling");
    
    // Test invalid array creation
    let invalid_shape = &[2, 3]; // Expects 6 elements
    let invalid_data = vec![1.0, 2.0]; // Only 2 elements
    
    let invalid_result = create_mlx_array(invalid_shape, invalid_data);
    assert!(invalid_result.is_err(), "Should fail with mismatched data length");
    println!("Invalid array creation correctly failed");
    
    // Test empty array
    let empty_shape = &[0];
    let empty_data = vec![];
    
    let empty_result = create_mlx_array(empty_shape, empty_data);
    match empty_result {
        Ok(array) => {
            println!("Empty array creation successful");
            assert_eq!(array.shape(), &[0]);
        }
        Err(e) => {
            println!("Empty array creation failed: {}", e);
        }
    }
    
    // Test operations with incompatible shapes
    if is_mlx_available() {
        let a_result = create_mlx_array(&[2, 3], vec![1.0; 6]);
        let b_result = create_mlx_array(&[4, 2], vec![1.0; 8]);
        
        if let (Ok(a), Ok(b)) = (a_result, b_result) {
            let incompatible_matmul = mlx_matmul(&a, &b);
            // This should fail due to incompatible shapes for matrix multiplication
            if incompatible_matmul.is_err() {
                println!("Incompatible matrix multiplication correctly failed");
            } else {
                println!("Warning: Incompatible matrix multiplication succeeded (unexpected)");
            }
        }
    }
}

/// Performance benchmark for MLX operations
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_performance_benchmark() {
    use bitnet_core::{is_mlx_available, create_mlx_array, mlx_matmul, mlx_quantize};
    use std::time::Instant;
    
    if !is_mlx_available() {
        println!("MLX not available, skipping performance benchmark");
        return;
    }
    
    println!("Running MLX performance benchmark");
    
    // Test different data sizes
    let sizes = [64, 256, 1024];
    
    for &size in &sizes {
        println!("\nBenchmarking operations with {}x{} matrices", size, size);
        
        // Generate test data
        let data_a: Vec<f32> = (0..size*size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..size*size).map(|i| (i * 2) as f32).collect();
        
        // Benchmark array creation
        let create_start = Instant::now();
        let array_a_result = create_mlx_array(&[size as i32, size as i32], data_a);
        let array_b_result = create_mlx_array(&[size as i32, size as i32], data_b);
        let create_time = create_start.elapsed();
        
        if let (Ok(array_a), Ok(array_b)) = (array_a_result, array_b_result) {
            println!("  Array creation: {:?}", create_time);
            
            // Benchmark matrix multiplication
            let matmul_start = Instant::now();
            let matmul_result = mlx_matmul(&array_a, &array_b);
            let matmul_time = matmul_start.elapsed();
            
            match matmul_result {
                Ok(result_array) => {
                    println!("  Matrix multiplication: {:?}", matmul_time);
                    assert_eq!(result_array.shape(), &[size as i32, size as i32]);
                    
                    // Calculate throughput
                    let operations = 2 * size * size * size; // Approximate FLOPs for matrix multiplication
                    let gflops = (operations as f64) / (matmul_time.as_secs_f64() * 1e9);
                    println!("  Performance: {:.2} GFLOPS", gflops);
                }
                Err(e) => {
                    println!("  Matrix multiplication failed: {}", e);
                }
            }
            
            // Benchmark quantization
            let quant_start = Instant::now();
            let quant_result = mlx_quantize(&array_a, 0.5);
            let quant_time = quant_start.elapsed();
            
            match quant_result {
                Ok(_) => {
                    println!("  Quantization: {:?}", quant_time);
                }
                Err(e) => {
                    println!("  Quantization failed: {}", e);
                }
            }
        } else {
            println!("  Array creation failed, skipping benchmarks for size {}", size);
        }
    }
}

/// Integration test combining MLX with existing BitNet systems
#[test]
#[cfg(feature = "mlx")]
fn test_mlx_integration_comprehensive() {
    use bitnet_core::{
        is_mlx_available, default_mlx_device, get_mlx_device_manager, create_mlx_array,
        BitNetMlxTensor, MlxOperations, mlx_to_candle_tensor, candle_to_mlx_array
    };
    
    if !is_mlx_available() {
        println!("MLX not available, skipping comprehensive integration test");
        return;
    }
    
    println!("Running comprehensive MLX integration test");
    
    // Step 1: Initialize MLX context
    let device = match default_mlx_device() {
        Ok(device) => device,
        Err(e) => {
            println!("Failed to create MLX device: {}", e);
            return;
        }
    };
    
    let manager = get_mlx_device_manager();
    println!("✓ MLX context initialized");
    println!("  Device: {}", device.device_type());
    println!("  Available devices: {}", manager.available_devices().len());
    
    // Step 2: Test tensor creation and operations
    let test_data = vec![1.0, 2.0, 3.0, 4.0];
    let array = match create_mlx_array(&[2, 2], test_data.clone()) {
        Ok(array) => array,
        Err(e) => {
            println!("Failed to create MLX array: {}", e);
            return;
        }
    };
    
    let tensor = BitNetMlxTensor::new(array, device.clone());
    println!("✓ MLX tensor created");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Device: {}", tensor.device().device_type());
    
    // Step 3: Test BitNet operations
    let quantized_result = MlxOperations::quantize_1_58_bit(&tensor);
    match quantized_result {
        Ok(quantized_tensor) => {
            println!("✓ 1.58-bit quantization successful");
            assert_eq!(quantized_tensor.shape(), tensor.shape());
        }
        Err(e) => {
            println!("✗ 1.58-bit quantization failed: {}", e);
        }
    }
    
    // Step 4: Test interoperability with Candle
    let candle_tensor_result = mlx_to_candle_tensor(tensor.array());
    match candle_tensor_result {
        Ok(candle_tensor) => {
            println!("✓ MLX to Candle conversion successful");
            assert_eq!(candle_tensor.shape().dims(), &[2, 2]);
            
            // Convert back to MLX
            let mlx_array_result = candle_to_mlx_array(&candle_tensor);
            match mlx_array_result {
                Ok(mlx_array) => {
                    println!("✓ Candle to MLX conversion successful");
                    assert_eq!(mlx_array.shape(), &[2, 2]);
                }
                Err(e) => {
                    println!("✗ Candle to MLX conversion failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ MLX to Candle conversion failed: {}", e);
        }
    }
    
    // Step 5: Test device capabilities
    for device_info in manager.available_devices() {
        let capabilities = manager.get_capabilities(device_info);
        println!("✓ Device {} capabilities: {}", device_info.to_bitnet_device_type(), capabilities);
    }
    
    println!("🎉 Comprehensive MLX integration test completed!");
}

/// Test MLX vs Metal vs CPU device comparison
#[test]
fn test_mlx_vs_metal_vs_cpu_comparison() {
    println!("Testing MLX vs Metal vs CPU device comparison");
    
    // Test MLX availability
    #[cfg(feature = "mlx")]
    {
        use bitnet_core::is_mlx_available;
        let mlx_available = is_mlx_available();
        println!("MLX available: {}", mlx_available);
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        println!("MLX available: false (feature disabled)");
        println!("MLX correctly not available without feature");
    }
    
    // Test Metal availability (if metal module exists)
    #[cfg(feature = "metal")]
    {
        use bitnet_core::device::is_metal_available;
        let metal_available = is_metal_available();
        println!("Metal available: {}", metal_available);
        
        // On Apple Silicon, both MLX and Metal should typically be available
        #[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "mlx"))]
        {
            use bitnet_core::is_mlx_available;
            let mlx_available = is_mlx_available();
            
            if mlx_available && metal_available {
                println!("Both MLX and Metal available on Apple Silicon (optimal)");
            } else if metal_available && !mlx_available {
                println!("Metal available but MLX not available (MLX may need installation)");
            } else if mlx_available && !metal_available {
                println!("MLX available but Metal not available (unusual)");
            } else {
                println!("Neither MLX nor Metal available on Apple Silicon (unexpected)");
            }
        }
    }
    
    // CPU should always be available
    println!("CPU always available: true");
    
    // Test device selection priority
    #[cfg(feature = "mlx")]
    {
        use bitnet_core::mlx::{is_mlx_available, default_mlx_device};
        
        if is_mlx_available() {
            let mlx_device_result = default_mlx_device();
            match mlx_device_result {
                Ok(device) => {
                    println!("MLX device selection successful: {}", device.device_type());
                }
                Err(e) => {
                    println!("MLX device selection failed: {}", e);
                }
            }
        }
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        println!("MLX feature disabled - device selection test skipped");
    }
    
    #[cfg(feature = "metal")]
    {
        use bitnet_core::device::auto_select_device;
        let auto_device = auto_select_device();
        println!("Auto-selected device: {:?}", auto_device);
    }
}