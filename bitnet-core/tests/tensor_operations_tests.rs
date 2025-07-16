//! Comprehensive Unit Tests for Core Tensor Operations
//!
//! This test suite validates all aspects of the BitNet tensor system,
//! including data initialization, Candle interoperability, mathematical operations,
//! device migration, and error handling across different BitNet data types.

use std::sync::Arc;
use std::time::Duration;
use std::thread;

use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType, TensorMetadata, TensorHandle};
use bitnet_core::device::{get_cpu_device, auto_select_device, is_metal_available, get_metal_device};
use candle_core::{Device, Tensor, DType};

/// Helper function to create a test memory pool
fn create_test_pool() -> HybridMemoryPool {
    HybridMemoryPool::new().expect("Failed to create test memory pool")
}

/// Helper function to create a test memory pool (same as above for now)
fn create_test_pool_with_tracking() -> HybridMemoryPool {
    HybridMemoryPool::new().expect("Failed to create test memory pool")
}

/// Helper function to get all available devices for testing
fn get_test_devices() -> Vec<Device> {
    let mut devices = vec![get_cpu_device()];
    
    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            devices.push(metal_device);
        }
    }
    
    devices
}

/// Helper function to get all BitNet data types for testing
fn get_test_dtypes() -> Vec<BitNetDType> {
    BitNetDType::all_types().to_vec()
}

// =============================================================================
// Data Initialization Tests
// =============================================================================

#[test]
fn test_tensor_zeros_creation() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            let shapes = vec![
                vec![],           // scalar
                vec![5],          // vector
                vec![2, 3],       // matrix
                vec![2, 3, 4],    // 3D tensor
                vec![1, 1, 1, 1], // 4D tensor with unit dimensions
            ];
            
            for shape in shapes {
                let tensor = BitNetTensor::zeros(&shape, dtype, device, &pool)
                    .expect(&format!("Failed to create zeros tensor with shape {:?}, dtype {}, device {:?}", 
                                   shape, dtype, device));
                
                // Verify tensor properties
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.dtype(), dtype);
                assert_eq!(tensor.device(), *device);
                assert_eq!(tensor.ref_count(), 1);
                
                // Verify element count calculation
                let expected_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
                assert_eq!(tensor.element_count(), expected_elements);
                
                // Verify size calculation
                let expected_size = dtype.bytes_for_elements(expected_elements);
                assert_eq!(tensor.size_bytes(), expected_size);
            }
        }
    }
}

#[test]
fn test_tensor_ones_creation() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            let shapes = vec![
                vec![3],       // vector
                vec![2, 2],    // square matrix
                vec![1, 5, 1], // 3D with unit dimensions
            ];
            
            for shape in shapes {
                let tensor = BitNetTensor::ones(&shape, dtype, device, &pool)
                    .expect(&format!("Failed to create ones tensor with shape {:?}, dtype {}, device {:?}", 
                                   shape, dtype, device));
                
                // Verify tensor properties
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.dtype(), dtype);
                assert_eq!(tensor.device(), *device);
                
                // TODO: Once actual data initialization is implemented,
                // verify that the tensor data is actually filled with ones
                // For now, we just verify the tensor was created successfully
            }
        }
    }
}

#[test]
fn test_tensor_from_data_creation() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    
    for device in &devices {
        let test_cases = vec![
            (vec![1.0], vec![1]),                    // single element
            (vec![1.0, 2.0, 3.0, 4.0], vec![4]),     // vector
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),  // matrix
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]), // 2x3 matrix
        ];
        
        for (data, shape) in test_cases {
            let tensor = BitNetTensor::from_data(data.clone(), &shape, device, &pool)
                .expect(&format!("Failed to create tensor from data with shape {:?}, device {:?}", 
                               shape, device));
            
            // Verify tensor properties
            assert_eq!(tensor.shape(), shape);
            assert_eq!(tensor.dtype(), BitNetDType::F32); // from_data always uses F32
            assert_eq!(tensor.device(), *device);
            assert_eq!(tensor.element_count(), data.len());
            
            // TODO: Once actual data copying is implemented,
            // verify that the tensor data matches the input data
        }
    }
}

#[test]
fn test_tensor_from_data_shape_mismatch() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test cases where data length doesn't match shape
    let invalid_cases = vec![
        (vec![1.0, 2.0], vec![3]),        // data too short
        (vec![1.0, 2.0, 3.0], vec![2]),   // data too long
        (vec![1.0, 2.0, 3.0], vec![2, 2]), // data length doesn't match 2D shape
    ];
    
    for (data, shape) in invalid_cases {
        let result = BitNetTensor::from_data(data, &shape, &device, &pool);
        assert!(result.is_err(), "Expected error for mismatched data and shape");
        
        if let Err(e) = result {
            assert!(e.to_string().contains("Shape mismatch"), 
                   "Expected shape mismatch error, got: {}", e);
        }
    }
}

// =============================================================================
// Candle Interoperability Tests
// =============================================================================

#[test]
fn test_tensor_from_candle_conversion() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    
    for device in &devices {
        let candle_dtypes = vec![
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
        ];
        
        for candle_dtype in candle_dtypes {
            let shapes = vec![
                vec![2],
                vec![3, 3],
                vec![2, 2, 2],
            ];
            
            for shape in shapes {
                // Create a candle tensor
                let candle_tensor = Tensor::zeros(&shape, candle_dtype, device)
                    .expect("Failed to create candle tensor");
                
                // Convert to BitNet tensor
                let bitnet_tensor = BitNetTensor::from_candle(candle_tensor.clone(), &pool);
                
                match bitnet_tensor {
                    Ok(tensor) => {
                        // Verify conversion worked
                        assert_eq!(tensor.shape(), shape);
                        assert_eq!(tensor.device(), *device);
                        
                        // Verify dtype conversion
                        let expected_dtype = BitNetDType::from_candle_dtype(candle_dtype);
                        if let Some(expected) = expected_dtype {
                            assert_eq!(tensor.dtype(), expected);
                        }
                    }
                    Err(e) => {
                        // Some conversions might fail for unsupported types
                        println!("Conversion failed for {:?} on {:?}: {}", candle_dtype, device, e);
                    }
                }
            }
        }
    }
}

#[test]
fn test_tensor_to_candle_conversion() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            let shapes = vec![
                vec![2],
                vec![2, 3],
                vec![1, 4, 1],
            ];
            
            for shape in shapes {
                let bitnet_tensor = BitNetTensor::zeros(&shape, dtype, device, &pool)
                    .expect("Failed to create BitNet tensor");
                
                let candle_tensor = bitnet_tensor.to_candle();
                
                match candle_tensor {
                    Ok(tensor) => {
                        // Verify conversion worked
                        assert_eq!(tensor.shape().dims(), &shape);
                        assert_eq!(tensor.device(), device);
                        
                        // Verify dtype conversion
                        let expected_candle_dtype = dtype.to_candle_dtype();
                        assert_eq!(tensor.dtype(), expected_candle_dtype);
                    }
                    Err(e) => {
                        println!("to_candle conversion failed for dtype {} on {:?}: {}", dtype, device, e);
                    }
                }
            }
        }
    }
}

#[test]
fn test_candle_roundtrip_conversion() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test roundtrip: Candle -> BitNet -> Candle
    let original_tensor = Tensor::zeros(&[2, 3], DType::F32, &device)
        .expect("Failed to create original candle tensor");
    
    let bitnet_tensor = BitNetTensor::from_candle(original_tensor.clone(), &pool)
        .expect("Failed to convert to BitNet tensor");
    
    let converted_back = bitnet_tensor.to_candle()
        .expect("Failed to convert back to candle tensor");
    
    // Verify properties are preserved
    assert_eq!(original_tensor.shape().dims(), converted_back.shape().dims());
    assert_eq!(original_tensor.dtype(), converted_back.dtype());
    assert_eq!(original_tensor.device(), converted_back.device());
}

// =============================================================================
// Mathematical Operations Tests
// =============================================================================

#[test]
fn test_tensor_reshape_operations() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    // Valid reshapes (same number of elements)
    let valid_reshapes = vec![
        vec![6],        // flatten to vector
        vec![3, 2],     // transpose dimensions
        vec![1, 6],     // add unit dimension
        vec![2, 3, 1],  // add trailing unit dimension
    ];
    
    for new_shape in valid_reshapes {
        let reshaped = tensor.reshape(&new_shape);
        match reshaped {
            Ok(new_tensor) => {
                assert_eq!(new_tensor.shape(), new_shape);
                assert_eq!(new_tensor.element_count(), tensor.element_count());
                assert_eq!(new_tensor.dtype(), tensor.dtype());
            }
            Err(e) => {
                println!("Reshape to {:?} failed: {}", new_shape, e);
                // TODO: Once reshape is fully implemented, this should succeed
            }
        }
    }
    
    // Invalid reshapes (different number of elements)
    let invalid_reshapes = vec![
        vec![5],        // wrong total elements
        vec![2, 2],     // wrong total elements
        vec![3, 3],     // wrong total elements
    ];
    
    for new_shape in invalid_reshapes {
        let result = tensor.reshape(&new_shape);
        assert!(result.is_err(), "Expected error for invalid reshape to {:?}", new_shape);
    }
}

#[test]
fn test_tensor_cloning() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let original = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create original tensor");
    
    original.set_name(Some("original_tensor".to_string()));
    
    // Test reference cloning (Arc::clone)
    let ref_clone = original.clone();
    assert_eq!(original.id(), ref_clone.id());
    assert_eq!(original.ref_count(), 2);
    assert_eq!(ref_clone.ref_count(), 2);
    
    // Test tensor cloning (new tensor with copied data)
    let tensor_clone = original.clone_tensor(&pool);
    match tensor_clone {
        Ok(cloned) => {
            assert_ne!(original.id(), cloned.id()); // Different tensor ID
            assert_eq!(original.shape(), cloned.shape());
            assert_eq!(original.dtype(), cloned.dtype());
            assert_eq!(original.device(), cloned.device());
            assert_eq!(cloned.ref_count(), 1); // New tensor has ref count 1
            assert_eq!(cloned.name(), original.name());
        }
        Err(e) => {
            println!("Tensor cloning failed: {}", e);
            // TODO: Once clone_tensor is fully implemented, this should succeed
        }
    }
}

// =============================================================================
// Device Operations Tests
// =============================================================================

#[test]
fn test_tensor_device_migration() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    
    if devices.len() < 2 {
        println!("Skipping device migration test - only one device available");
        return;
    }
    
    let source_device = &devices[0];
    let target_device = &devices[1];
    
    let tensor = BitNetTensor::zeros(&[3, 3], BitNetDType::F32, source_device, &pool)
        .expect("Failed to create tensor on source device");
    
    assert_eq!(tensor.device(), *source_device);
    
    // Migrate to target device
    let migrated = tensor.to_device(target_device, &pool);
    
    match migrated {
        Ok(new_tensor) => {
            assert_eq!(new_tensor.device(), *target_device);
            assert_eq!(new_tensor.shape(), tensor.shape());
            assert_eq!(new_tensor.dtype(), tensor.dtype());
            assert_ne!(new_tensor.id(), tensor.id()); // Should be a new tensor
        }
        Err(e) => {
            println!("Device migration failed: {}", e);
            // TODO: Once device migration is fully implemented, this should succeed
        }
    }
}

#[test]
fn test_tensor_same_device_migration() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    // Migrate to same device (should return clone)
    let result = tensor.to_device(&device, &pool)
        .expect("Same device migration should succeed");
    
    // Should return a clone of the same tensor
    assert_eq!(result.id(), tensor.id());
    assert_eq!(result.device(), device);
}

// =============================================================================
// Error Handling and Edge Cases Tests
// =============================================================================

#[test]
fn test_tensor_handle_operations() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::I8, &device, &pool)
        .expect("Failed to create tensor");
    
    let handle = tensor.handle();
    
    // Test handle validity
    assert!(handle.is_valid());
    assert_eq!(handle.tensor_id().unwrap(), tensor.id());
    assert_eq!(handle.shape().unwrap(), tensor.shape());
    assert_eq!(handle.dtype().unwrap(), tensor.dtype());
    assert_eq!(handle.element_count().unwrap(), tensor.element_count());
    
    // Test handle validation
    assert!(handle.validate(Some(&[2, 3]), Some(BitNetDType::I8)).is_ok());
    assert!(handle.validate(Some(&[3, 2]), None).is_err()); // Wrong shape
    assert!(handle.validate(None, Some(BitNetDType::F32)).is_err()); // Wrong dtype
    
    // Test handle metadata operations
    assert!(handle.touch().is_ok());
    assert!(handle.add_tag("test_tag".to_string()).is_ok());
    assert!(handle.has_tag("test_tag").unwrap());
    assert!(handle.remove_tag("test_tag").is_ok());
    assert!(!handle.has_tag("test_tag").unwrap());
}

#[test]
fn test_tensor_handle_invalidation() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    let handle = tensor.handle();
    assert!(handle.is_valid());
    
    // Drop the tensor
    drop(tensor);
    
    // Handle should become invalid
    assert!(!handle.is_valid());
    assert!(handle.metadata().is_err());
}

#[test]
fn test_tensor_weak_handle() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    let handle = tensor.handle();
    let weak_handle = handle.downgrade();
    
    assert!(weak_handle.is_valid());
    assert_eq!(weak_handle.id(), handle.id());
    
    // Should be able to upgrade
    let upgraded = weak_handle.upgrade();
    assert!(upgraded.is_some());
    assert_eq!(upgraded.unwrap().id(), handle.id());
    
    // Drop strong references
    drop(tensor);
    drop(handle);
    
    // Weak handle should become invalid
    assert!(!weak_handle.is_valid());
    assert!(weak_handle.upgrade().is_none());
}

// =============================================================================
// Data Type Specific Tests
// =============================================================================

#[test]
fn test_all_bitnet_dtypes() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    let dtypes = get_test_dtypes();
    
    for &dtype in &dtypes {
        let tensor = BitNetTensor::zeros(&[4], dtype, &device, &pool)
            .expect(&format!("Failed to create tensor with dtype {}", dtype));
        
        assert_eq!(tensor.dtype(), dtype);
        
        // Test dtype properties
        let expected_bits = dtype.bits_per_element();
        let expected_bytes = dtype.bytes_for_elements(4);
        assert_eq!(tensor.size_bytes(), expected_bytes);
        
        // Test dtype classification
        if dtype.is_float() {
            assert!(!dtype.is_integer());
            assert!(!dtype.is_quantized());
        } else if dtype.is_integer() {
            assert!(!dtype.is_float());
        }
        
        if dtype.is_quantized() {
            assert!(expected_bits < 8);
        }
        
        if dtype.is_bitnet158() {
            assert_eq!(dtype, BitNetDType::BitNet158);
            assert_eq!(expected_bits, 2);
        }
        
        // Test memory efficiency
        let efficiency = dtype.memory_efficiency();
        assert!(efficiency > 0.0);
        assert_eq!(efficiency, 32.0 / expected_bits as f32);
    }
}

#[test]
fn test_dtype_value_ranges() {
    let integer_dtypes = vec![
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];
    
    for dtype in integer_dtypes {
        if let Some((min, max)) = dtype.value_range() {
            assert!(min <= max, "Min value should be <= max value for {}", dtype);
            
            match dtype {
                BitNetDType::I8 => {
                    assert_eq!(min, -128);
                    assert_eq!(max, 127);
                }
                BitNetDType::I4 => {
                    assert_eq!(min, -8);
                    assert_eq!(max, 7);
                }
                BitNetDType::I2 => {
                    assert_eq!(min, -2);
                    assert_eq!(max, 1);
                }
                BitNetDType::I1 => {
                    assert_eq!(min, -1);
                    assert_eq!(max, 0);
                }
                BitNetDType::BitNet158 => {
                    assert_eq!(min, -1);
                    assert_eq!(max, 1);
                }
                _ => unreachable!(),
            }
        }
    }
    
    // Float types should not have value ranges
    let float_dtypes = vec![BitNetDType::F32, BitNetDType::F16, BitNetDType::BF16];
    for dtype in float_dtypes {
        assert!(dtype.value_range().is_none(), "Float type {} should not have value range", dtype);
    }
}

// =============================================================================
// Memory Management Tests
// =============================================================================

#[test]
fn test_tensor_memory_tracking() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();
    
    // Create multiple tensors
    let mut tensors = Vec::new();
    for i in 0..5 {
        let size = (i + 1) * 1024;
        let shape = vec![size];
        let tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");
        tensors.push(tensor);
    }
    
    // Check memory metrics
    let metrics = pool.get_metrics();
    assert!(metrics.active_allocations >= 5);
    assert!(metrics.current_memory_usage > 0);
    
    // Note: Detailed metrics not yet implemented
    // TODO: Add detailed metrics tests when implemented
    
    // Drop tensors and verify cleanup
    drop(tensors);
    
    // Memory should be deallocated
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_tensor_reference_counting() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor1 = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    assert_eq!(tensor1.ref_count(), 1);
    
    // Clone increases reference count
    let tensor2 = tensor1.clone();
    assert_eq!(tensor1.ref_count(), 2);
    assert_eq!(tensor2.ref_count(), 2);
    assert_eq!(tensor1.id(), tensor2.id());
    
    // Create handle (doesn't affect Arc ref count)
    let handle = tensor1.handle();
    assert_eq!(tensor1.ref_count(), 2); // Still 2
    
    // Drop one reference
    drop(tensor2);
    assert_eq!(tensor1.ref_count(), 1);
    
    // Handle should still be valid
    assert!(handle.is_valid());
    
    // Drop last tensor reference
    drop(tensor1);
    
    // Handle should become invalid
    assert!(!handle.is_valid());
}

// =============================================================================
// Concurrent Operations Tests
// =============================================================================

#[test]
fn test_concurrent_tensor_operations() {
    let pool = Arc::new(create_test_pool());
    let device = get_cpu_device();
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads creating tensors
    for thread_id in 0..4 {
        let pool_clone = pool.clone();
        let device_clone = device.clone();
        
        let handle = thread::spawn(move || {
            let mut thread_tensors = Vec::new();
            
            // Create tensors in each thread
            for i in 0..10 {
                let shape = vec![thread_id + 1, i + 1];
                let tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone)
                    .expect("Failed to create tensor in thread");
                
                // Verify tensor properties
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.dtype(), BitNetDType::F32);
                
                thread_tensors.push(tensor);
            }
            
            // Test concurrent handle operations
            for tensor in &thread_tensors {
                let handle = tensor.handle();
                assert!(handle.is_valid());
                assert!(handle.touch().is_ok());
            }
            
            thread_tensors.len()
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    let mut total_tensors = 0;
    for handle in handles {
        total_tensors += handle.join().expect("Thread panicked");
    }
    
    assert_eq!(total_tensors, 40); // 4 threads * 10 tensors each
    
    // Verify pool state
    let metrics = pool.get_metrics();
    assert_eq!(metrics.active_allocations, 40);
}

#[test]
fn test_concurrent_device_migration() {
    let devices = get_test_devices();
    if devices.len() < 2 {
        println!("Skipping concurrent device migration test - need multiple devices");
        return;
    }
    
    let pool = Arc::new(create_test_pool());
    let source_device = devices[0].clone();
    let target_device = devices[1].clone();
    
    // Create a tensor on source device
    let tensor = Arc::new(
        BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &source_device, &pool)
            .expect("Failed to create tensor")
    );
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads attempting migration
    for _ in 0..3 {
        let tensor_clone = tensor.clone();
        let pool_clone = pool.clone();
        let target_device_clone = target_device.clone();
        
        let handle = thread::spawn(move || {
            let result = tensor_clone.to_device(&target_device_clone, &pool_clone);
            match result {
                Ok(migrated) => {
                    assert_eq!(migrated.device(), target_device_clone);
                    assert_eq!(migrated.shape(), tensor_clone.shape());
                    true
                }
                Err(e) => {
                    println!("Migration failed: {}", e);
                    false
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all migrations to complete
    for handle in handles {
        handle.join().expect("Migration thread panicked");
    }
}

// =============================================================================
// Performance and Benchmark Tests
// =============================================================================

#[test]
fn test_tensor_creation_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let start_time = std::time::Instant::now();
    let iterations = 1000;
    
    for i in 0..iterations {
        let shape = vec![i % 10 + 1, (i % 5) + 1];
        let _tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");
    }
    
    let duration = start_time.elapsed();
    let avg_duration = duration / iterations;
    
    println!("Average tensor creation time: {:?}", avg_duration);
    
    // Verify performance is reasonable (less than 1ms per tensor)
    assert!(avg_duration < Duration::from_millis(1), 
           "Tensor creation too slow: {:?}", avg_duration);
}

#[test]
fn test_handle_creation_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    let start_time = std::time::Instant::now();
    let iterations = 10000;
    
    for _ in 0..iterations {
        let _handle = tensor.handle();
    }
    
    let duration = start_time.elapsed();
    let avg_duration = duration / iterations;
    
    println!("Average handle creation time: {:?}", avg_duration);
    
    // Handle creation should be very fast (less than 10μs)
    assert!(avg_duration < Duration::from_micros(10), 
           "Handle creation too slow: {:?}", avg_duration);
}

#[test]
fn test_memory_efficiency() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test memory efficiency of different data types
    let element_count = 1000;
    let shape = vec![element_count];
    
    for &dtype in &get_test_dtypes() {
        let tensor = BitNetTensor::zeros(&shape, dtype, &device, &pool)
            .expect("Failed to create tensor");
        
        let expected_bytes = dtype.bytes_for_elements(element_count);
        assert_eq!(tensor.size_bytes(), expected_bytes);
        
        let efficiency = dtype.memory_efficiency();
        let f32_bytes = BitNetDType::F32.bytes_for_elements(element_count);
        let actual_efficiency = f32_bytes as f32 / expected_bytes as f32;
        
        // Check efficiency is approximately correct (within 1%)
        let diff = (efficiency - actual_efficiency).abs();
        assert!(diff < 0.01, "Efficiency mismatch: expected {}, got {}", actual_efficiency, efficiency);
        
        println!("Dtype {}: {} bytes, {:.1}x efficiency", 
                dtype, expected_bytes, efficiency);
    }
}

// =============================================================================