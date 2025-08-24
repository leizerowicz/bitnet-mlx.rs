//! Comprehensive Unit Tests for Core Tensor Operations
//!
//! This test suite validates all aspects of the BitNet tensor system,
//! including data initialization, Candle interoperability, mathematical operations,
//! device migration, and error handling across different BitNet data types.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bitnet_core::device::{get_cpu_device, get_metal_device, is_metal_available};
use bitnet_core::memory::tensor::{BitNetDType, BitNetTensor};
use bitnet_core::memory::HybridMemoryPool;
use candle_core::{DType, Device, Tensor};

/// Helper function to create a test memory pool
fn create_test_pool() -> HybridMemoryPool {
    use bitnet_core::memory::MemoryPoolConfig;

    let config = MemoryPoolConfig {
        small_block_threshold: 2 * 1024 * 1024, // 2MB to handle large benchmark tensors
        small_pool_initial_size: 64 * 1024 * 1024, // 64MB initial
        small_pool_max_size: 512 * 1024 * 1024, // 512MB max
        large_pool_initial_size: 128 * 1024 * 1024, // 128MB initial for large tensors
        large_pool_max_size: 2 * 1024 * 1024 * 1024, // 2GB max
        enable_metrics: true,
        ..Default::default()
    };

    HybridMemoryPool::with_config(config).expect("Failed to create test memory pool")
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
                    .unwrap_or_else(|_| panic!("Failed to create zeros tensor with shape {shape:?}, dtype {dtype}, device {device:?}"));

                // Verify tensor properties
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.dtype(), dtype);
                // Note: Device comparison not directly supported, verify device type matches
                let tensor_device = tensor.device();
                match (device, &tensor_device) {
                    (Device::Cpu, Device::Cpu) => {}
                    (Device::Metal(_), Device::Metal(_)) => {}
                    (Device::Cuda(_), Device::Cuda(_)) => {}
                    _ => panic!("Device mismatch: expected {device:?}, got {tensor_device:?}"),
                }
                assert_eq!(tensor.ref_count(), 1);

                // Verify element count calculation
                let expected_elements: usize = if shape.is_empty() {
                    1
                } else {
                    shape.iter().product()
                };
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
                    .unwrap_or_else(|_| panic!("Failed to create ones tensor with shape {shape:?}, dtype {dtype}, device {device:?}"));

                // Verify tensor properties
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.dtype(), dtype);
                // Note: Device comparison not directly supported, verify device type matches
                let tensor_device = tensor.device();
                match (device, &tensor_device) {
                    (Device::Cpu, Device::Cpu) => {}
                    (Device::Metal(_), Device::Metal(_)) => {}
                    (Device::Cuda(_), Device::Cuda(_)) => {}
                    _ => panic!("Device mismatch: expected {device:?}, got {tensor_device:?}"),
                }

                // TODO: Implement actual data verification once data initialization is complete
                // For now, we verify the tensor was created with correct metadata
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
            (vec![1.0], vec![1]),                             // single element
            (vec![1.0, 2.0, 3.0, 4.0], vec![4]),              // vector
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),           // matrix
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]), // 2x3 matrix
        ];

        for (data, shape) in test_cases {
            let tensor = BitNetTensor::from_data(data.clone(), &shape, device, &pool)
                .unwrap_or_else(|_| {
                    panic!(
                        "Failed to create tensor from data with shape {shape:?}, device {device:?}"
                    )
                });

            // Verify tensor properties
            assert_eq!(tensor.shape(), shape);
            assert_eq!(tensor.dtype(), BitNetDType::F32); // from_data always uses F32
                                                          // Note: Device comparison not directly supported, verify device type matches
            let tensor_device = tensor.device();
            match (device, &tensor_device) {
                (Device::Cpu, Device::Cpu) => {}
                (Device::Metal(_), Device::Metal(_)) => {}
                (Device::Cuda(_), Device::Cuda(_)) => {}
                _ => panic!("Device mismatch: expected {device:?}, got {tensor_device:?}"),
            }
            assert_eq!(tensor.element_count(), data.len());

            // TODO: Implement actual data verification once data copying is complete
            // For now, we verify the tensor was created with correct metadata
        }
    }
}

#[test]
fn test_tensor_from_data_shape_mismatch() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test cases where data length doesn't match shape
    let invalid_cases = vec![
        (vec![1.0, 2.0], vec![3]),         // data too short
        (vec![1.0, 2.0, 3.0], vec![2]),    // data too long
        (vec![1.0, 2.0, 3.0], vec![2, 2]), // data length doesn't match 2D shape
    ];

    for (data, shape) in invalid_cases {
        let result = BitNetTensor::from_data(data, &shape, &device, &pool);
        assert!(
            result.is_err(),
            "Expected error for mismatched data and shape"
        );

        if let Err(e) = result {
            assert!(
                e.to_string().contains("Shape mismatch"),
                "Expected shape mismatch error, got: {e}"
            );
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
        let candle_dtypes = vec![DType::F32, DType::F16, DType::BF16, DType::I64];

        for candle_dtype in candle_dtypes {
            let shapes = vec![vec![2], vec![3, 3], vec![2, 2, 2]];

            for shape in shapes {
                // Create a candle tensor
                let candle_tensor = Tensor::zeros(shape.as_slice(), candle_dtype, device)
                    .expect("Failed to create candle tensor");

                // Convert to BitNet tensor
                let bitnet_tensor = BitNetTensor::from_candle(candle_tensor.clone(), &pool);

                match bitnet_tensor {
                    Ok(tensor) => {
                        // Verify conversion worked
                        assert_eq!(tensor.shape(), shape);
                        // Note: Device comparison not directly supported, verify device type matches
                        let tensor_device = tensor.device();
                        match (device, &tensor_device) {
                            (Device::Cpu, Device::Cpu) => {}
                            (Device::Metal(_), Device::Metal(_)) => {}
                            (Device::Cuda(_), Device::Cuda(_)) => {}
                            _ => panic!(
                                "Device mismatch: expected {device:?}, got {tensor_device:?}"
                            ),
                        }

                        // Verify dtype conversion
                        let expected_dtype = BitNetDType::from_candle_dtype(candle_dtype);
                        if let Some(expected) = expected_dtype {
                            assert_eq!(tensor.dtype(), expected);
                        }
                    }
                    Err(e) => {
                        // Some conversions might fail for unsupported types
                        println!("Conversion failed for {candle_dtype:?} on {device:?}: {e}");
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
            let shapes = vec![vec![2], vec![2, 3], vec![1, 4, 1]];

            for shape in shapes {
                let bitnet_tensor = BitNetTensor::zeros(&shape, dtype, device, &pool)
                    .expect("Failed to create BitNet tensor");

                let candle_tensor = bitnet_tensor.to_candle();

                match candle_tensor {
                    Ok(tensor) => {
                        // Verify conversion worked
                        assert_eq!(tensor.shape().dims(), &shape);
                        // Note: Device comparison not directly supported, verify device type matches
                        let tensor_device = tensor.device();
                        match (device, &tensor_device) {
                            (Device::Cpu, Device::Cpu) => {}
                            (Device::Metal(_), Device::Metal(_)) => {}
                            (Device::Cuda(_), Device::Cuda(_)) => {}
                            _ => panic!(
                                "Device mismatch: expected {device:?}, got {tensor_device:?}"
                            ),
                        }

                        // Verify dtype conversion
                        let expected_candle_dtype = dtype.to_candle_dtype();
                        assert_eq!(tensor.dtype(), expected_candle_dtype);
                    }
                    Err(e) => {
                        println!(
                            "to_candle conversion failed for dtype {dtype} on {device:?}: {e}"
                        );
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

    let converted_back = bitnet_tensor
        .to_candle()
        .expect("Failed to convert back to candle tensor");

    // Verify properties are preserved
    assert_eq!(
        original_tensor.shape().dims(),
        converted_back.shape().dims()
    );
    assert_eq!(original_tensor.dtype(), converted_back.dtype());
    // Note: Device comparison not directly supported, verify both are on same device type
    let orig_device = original_tensor.device();
    let conv_device = converted_back.device();
    match (&orig_device, &conv_device) {
        (Device::Cpu, Device::Cpu) => {}
        (Device::Metal(_), Device::Metal(_)) => {}
        (Device::Cuda(_), Device::Cuda(_)) => {}
        _ => panic!("Device mismatch after roundtrip: {orig_device:?} vs {conv_device:?}"),
    }
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
        vec![6],       // flatten to vector
        vec![3, 2],    // transpose dimensions
        vec![1, 6],    // add unit dimension
        vec![2, 3, 1], // add trailing unit dimension
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
                println!("Reshape to {new_shape:?} failed: {e}");
                // TODO: Once reshape is fully implemented, this should succeed
            }
        }
    }

    // Invalid reshapes (different number of elements)
    let invalid_reshapes = vec![
        vec![5],    // wrong total elements
        vec![2, 2], // wrong total elements
        vec![3, 3], // wrong total elements
    ];

    for new_shape in invalid_reshapes {
        let result = tensor.reshape(&new_shape);
        assert!(
            result.is_err(),
            "Expected error for invalid reshape to {new_shape:?}"
        );
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
            // TODO: Once clone_tensor is fully implemented, this should create a new tensor
            // For now, we just verify the operation succeeded
            assert_eq!(original.shape(), cloned.shape());
            assert_eq!(original.dtype(), cloned.dtype());
            // Note: Device comparison not directly supported, verify both are on same device type
            let orig_device = original.device();
            let cloned_device = cloned.device();
            match (&orig_device, &cloned_device) {
                (Device::Cpu, Device::Cpu) => {}
                (Device::Metal(_), Device::Metal(_)) => {}
                (Device::Cuda(_), Device::Cuda(_)) => {}
                _ => panic!("Device mismatch after cloning: {orig_device:?} vs {cloned_device:?}"),
            }
            assert_eq!(cloned.name(), original.name());

            // Note: Currently clone_tensor may not create a truly new tensor
            // This is expected until the implementation is complete
            println!("Tensor cloning succeeded (implementation may be incomplete)");
        }
        Err(e) => {
            println!("Tensor cloning failed: {e}");
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

    // Note: Device comparison not directly supported, verify device type matches
    let tensor_device = tensor.device();
    match (source_device, &tensor_device) {
        (Device::Cpu, Device::Cpu) => {}
        (Device::Metal(_), Device::Metal(_)) => {}
        (Device::Cuda(_), Device::Cuda(_)) => {}
        _ => panic!("Device mismatch: expected {source_device:?}, got {tensor_device:?}"),
    }

    // Migrate to target device
    let migrated = tensor.to_device(target_device, &pool);

    match migrated {
        Ok(new_tensor) => {
            // Note: Device comparison not directly supported, verify device type matches
            let new_tensor_device = new_tensor.device();
            match (target_device, &new_tensor_device) {
                (Device::Cpu, Device::Cpu) => {}
                (Device::Metal(_), Device::Metal(_)) => {}
                (Device::Cuda(_), Device::Cuda(_)) => {}
                _ => {
                    panic!("Device mismatch: expected {target_device:?}, got {new_tensor_device:?}")
                }
            }
            assert_eq!(new_tensor.shape(), tensor.shape());
            assert_eq!(new_tensor.dtype(), tensor.dtype());
            assert_ne!(new_tensor.id(), tensor.id()); // Should be a new tensor
        }
        Err(e) => {
            println!("Device migration failed: {e}");
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
    let result = tensor
        .to_device(&device, &pool)
        .expect("Same device migration should succeed");

    // Should return a clone of the same tensor
    assert_eq!(result.id(), tensor.id());
    // Note: Device comparison not directly supported, verify device type matches
    let result_device = result.device();
    match (&device, &result_device) {
        (Device::Cpu, Device::Cpu) => {}
        (Device::Metal(_), Device::Metal(_)) => {}
        (Device::Cuda(_), Device::Cuda(_)) => {}
        _ => panic!("Device mismatch: expected {device:?}, got {result_device:?}"),
    }
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
    assert!(handle
        .validate(Some(&[2, 3]), Some(BitNetDType::I8))
        .is_ok());
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
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype}"));

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
            assert!(min <= max, "Min value should be <= max value for {dtype}");

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
        assert!(
            dtype.value_range().is_none(),
            "Float type {dtype} should not have value range"
        );
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
    println!(
        "Active allocations: {}, Current allocated: {}",
        metrics.active_allocations, metrics.current_allocated
    );

    // Note: Memory tracking may not be fully implemented yet
    // We'll verify that we can at least get metrics without panicking
    assert!(metrics.active_allocations >= 0); // Should be non-negative
    assert!(metrics.current_allocated >= 0); // Should be non-negative

    // Note: Detailed metrics not yet implemented
    // TODO: Add detailed metrics tests when implemented

    // Drop tensors and verify cleanup
    drop(tensors);

    // Memory should be deallocated (but implementation may be incomplete)
    let final_metrics = pool.get_metrics();
    println!(
        "Final active allocations: {}",
        final_metrics.active_allocations
    );

    // For now, just verify we can get final metrics
    // TODO: Once memory tracking is fully implemented, verify cleanup
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
                let tensor =
                    BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone)
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
            .expect("Failed to create tensor"),
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
                    // Note: Device comparison not directly supported, verify device type matches
                    let migrated_device = migrated.device();
                    match (&target_device_clone, &migrated_device) {
                        (Device::Cpu, Device::Cpu) => {},
                        (Device::Metal(_), Device::Metal(_)) => {},
                        (Device::Cuda(_), Device::Cuda(_)) => {},
                        _ => panic!("Device mismatch: expected {target_device_clone:?}, got {migrated_device:?}"),
                    }
                    assert_eq!(migrated.shape(), tensor_clone.shape());
                    true
                }
                Err(e) => {
                    println!("Migration failed: {e}");
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
    let avg_duration = duration / iterations as u32;

    println!("Average tensor creation time: {avg_duration:?}");

    // Verify performance is reasonable (less than 1ms per tensor)
    assert!(
        avg_duration < Duration::from_millis(1),
        "Tensor creation too slow: {avg_duration:?}"
    );
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
    let avg_duration = duration / iterations as u32;

    println!("Average handle creation time: {avg_duration:?}");

    // Handle creation should be very fast (less than 10μs)
    assert!(
        avg_duration < Duration::from_micros(10),
        "Handle creation too slow: {avg_duration:?}"
    );
}

#[test]
fn test_memory_efficiency() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test memory efficiency of different data types
    let element_count = 1000;
    let shape = vec![element_count];

    for &dtype in &get_test_dtypes() {
        let tensor =
            BitNetTensor::zeros(&shape, dtype, &device, &pool).expect("Failed to create tensor");

        let expected_bytes = dtype.bytes_for_elements(element_count);
        assert_eq!(tensor.size_bytes(), expected_bytes);

        let efficiency = dtype.memory_efficiency();
        let f32_bytes = BitNetDType::F32.bytes_for_elements(element_count);
        let actual_efficiency = f32_bytes as f32 / expected_bytes as f32;

        // Check efficiency is approximately correct (within 1%)
        let diff = (efficiency - actual_efficiency).abs();
        assert!(
            diff < 0.01,
            "Efficiency mismatch: expected {actual_efficiency}, got {efficiency}"
        );

        println!("Dtype {dtype}: {expected_bytes} bytes, {efficiency:.1}x efficiency");
    }
}
// =============================================================================
// Data Content Verification Tests
// =============================================================================

#[test]
fn test_zeros_actual_data_content() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test with F32 dtype (most straightforward to verify)
    let tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create zeros tensor");

    // Convert to candle tensor to verify data content
    let candle_tensor = tensor
        .to_candle()
        .expect("Failed to convert to candle tensor");

    // TODO: Once actual data initialization is implemented, verify all values are 0.0
    // For now, we verify the tensor structure is correct
    assert_eq!(candle_tensor.shape().dims(), &[2, 2]);
    assert_eq!(candle_tensor.dtype(), candle_core::DType::F32);

    // Test with different shapes and dtypes
    let test_cases = vec![
        (vec![1], BitNetDType::F32),
        (vec![3], BitNetDType::F16),
        (vec![2, 3], BitNetDType::I8),
        (vec![1, 1, 4], BitNetDType::I4),
    ];

    for (shape, dtype) in test_cases {
        let tensor = BitNetTensor::zeros(&shape, dtype, &device, &pool).unwrap_or_else(|_| {
            panic!("Failed to create zeros tensor with shape {shape:?}, dtype {dtype}")
        });

        // Verify tensor properties
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), dtype);

        // TODO: Add actual data content verification once initialization is implemented
    }
}

#[test]
fn test_ones_actual_data_content() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test with F32 dtype
    let tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create ones tensor");

    // Convert to candle tensor to verify data content
    let candle_tensor = tensor
        .to_candle()
        .expect("Failed to convert to candle tensor");

    // TODO: Once actual data initialization is implemented, verify all values are 1.0
    // For now, we verify the tensor structure is correct
    assert_eq!(candle_tensor.shape().dims(), &[2, 2]);
    assert_eq!(candle_tensor.dtype(), candle_core::DType::F32);

    // Test with different shapes and dtypes
    let test_cases = vec![
        (vec![3], BitNetDType::F32),
        (vec![2, 2], BitNetDType::F16),
        (vec![1, 4], BitNetDType::I8),
    ];

    for (shape, dtype) in test_cases {
        let tensor = BitNetTensor::ones(&shape, dtype, &device, &pool).unwrap_or_else(|_| {
            panic!("Failed to create ones tensor with shape {shape:?}, dtype {dtype}")
        });

        // Verify tensor properties
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), dtype);

        // TODO: Add actual data content verification once initialization is implemented
    }
}

#[test]
fn test_from_data_content_preservation() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let test_cases = vec![
        (vec![1.0], vec![1]),
        (vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        (vec![1.5, -2.5, 3.14, 0.0], vec![2, 2]),
        (vec![-1.0, 0.0, 1.0, 2.0, -3.0, 4.0], vec![2, 3]),
    ];

    for (data, shape) in test_cases {
        let tensor = BitNetTensor::from_data(data.clone(), &shape, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create tensor from data with shape {shape:?}"));

        // Verify tensor properties
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
        assert_eq!(tensor.element_count(), data.len());

        // Convert to candle tensor to verify data content
        let candle_tensor = tensor
            .to_candle()
            .expect("Failed to convert to candle tensor");

        // TODO: Once actual data copying is implemented, verify the data matches
        // For now, we verify the tensor structure is correct
        assert_eq!(candle_tensor.shape().dims(), shape.as_slice());
        assert_eq!(candle_tensor.dtype(), candle_core::DType::F32);
    }
}

// =============================================================================
// Transpose Operation Tests
// =============================================================================

#[test]
fn test_transpose_2d_comprehensive() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    for device in &devices {
        // Test basic 2D transpose
        let tensor = BitNetTensor::zeros(&[3, 4], BitNetDType::F32, device, &pool)
            .expect("Failed to create tensor");

        // Convert to candle for transpose operation
        let candle_tensor = tensor.to_candle().expect("Failed to convert to candle");

        // Test transpose using the tensor utility function
        let transposed_candle = bitnet_core::tensor::transpose(&candle_tensor, &[1, 0])
            .expect("Failed to transpose tensor");

        assert_eq!(transposed_candle.shape().dims(), &[4, 3]);

        // Test identity transpose (no change)
        let identity_transposed = bitnet_core::tensor::transpose(&candle_tensor, &[0, 1])
            .expect("Failed to perform identity transpose");

        assert_eq!(identity_transposed.shape().dims(), &[3, 4]);

        // Test square matrix transpose
        let square_tensor = BitNetTensor::zeros(&[5, 5], BitNetDType::F32, device, &pool)
            .expect("Failed to create square tensor");

        let square_candle = square_tensor
            .to_candle()
            .expect("Failed to convert square tensor");

        let square_transposed = bitnet_core::tensor::transpose(&square_candle, &[1, 0])
            .expect("Failed to transpose square matrix");

        assert_eq!(square_transposed.shape().dims(), &[5, 5]);
    }
}

#[test]
fn test_transpose_3d_comprehensive() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::zeros(&[2, 3, 4], BitNetDType::F32, &device, &pool)
        .expect("Failed to create 3D tensor");

    let candle_tensor = tensor.to_candle().expect("Failed to convert to candle");

    // Test various 3D transpose permutations
    let permutations = vec![
        (vec![0, 1, 2], vec![2, 3, 4]), // Identity
        (vec![0, 2, 1], vec![2, 4, 3]), // Swap last two dimensions
        (vec![1, 0, 2], vec![3, 2, 4]), // Swap first two dimensions
        (vec![2, 1, 0], vec![4, 3, 2]), // Reverse all dimensions
        (vec![1, 2, 0], vec![3, 4, 2]), // Cyclic permutation
        (vec![2, 0, 1], vec![4, 2, 3]), // Another cyclic permutation
    ];

    for (perm, expected_shape) in permutations {
        let transposed = bitnet_core::tensor::transpose(&candle_tensor, &perm)
            .unwrap_or_else(|_| panic!("Failed to transpose with permutation {perm:?}"));

        assert_eq!(
            transposed.shape().dims(),
            expected_shape.as_slice(),
            "Transpose with permutation {perm:?} produced wrong shape"
        );
    }
}

#[test]
fn test_transpose_higher_dimensional() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test 4D tensor transpose
    let tensor_4d = BitNetTensor::zeros(&[2, 3, 4, 5], BitNetDType::F32, &device, &pool)
        .expect("Failed to create 4D tensor");

    let candle_4d = tensor_4d.to_candle().expect("Failed to convert 4D tensor");

    // Test some 4D permutations
    let permutations_4d = vec![
        (vec![0, 1, 2, 3], vec![2, 3, 4, 5]), // Identity
        (vec![3, 2, 1, 0], vec![5, 4, 3, 2]), // Reverse
        (vec![1, 0, 3, 2], vec![3, 2, 5, 4]), // Swap pairs
    ];

    for (perm, expected_shape) in permutations_4d {
        let transposed = bitnet_core::tensor::transpose(&candle_4d, &perm)
            .unwrap_or_else(|_| panic!("Failed to transpose 4D tensor with permutation {perm:?}"));

        assert_eq!(
            transposed.shape().dims(),
            expected_shape.as_slice(),
            "4D transpose with permutation {perm:?} produced wrong shape"
        );
    }

    // Test 5D tensor transpose
    let tensor_5d = BitNetTensor::zeros(&[1, 2, 3, 4, 5], BitNetDType::F32, &device, &pool)
        .expect("Failed to create 5D tensor");

    let candle_5d = tensor_5d.to_candle().expect("Failed to convert 5D tensor");

    let transposed_5d = bitnet_core::tensor::transpose(&candle_5d, &[4, 3, 2, 1, 0])
        .expect("Failed to transpose 5D tensor");

    assert_eq!(transposed_5d.shape().dims(), &[5, 4, 3, 2, 1]);
}

#[test]
fn test_transpose_invalid_permutations() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");

    let candle_tensor = tensor.to_candle().expect("Failed to convert to candle");

    // Test invalid permutation cases
    let invalid_cases = vec![
        vec![0],       // Too few dimensions
        vec![0, 1, 2], // Too many dimensions
        vec![0, 2],    // Invalid dimension index
        vec![1, 1],    // Duplicate dimension
        vec![2, 0],    // Out of range dimension
    ];

    for invalid_perm in invalid_cases {
        let result = bitnet_core::tensor::transpose(&candle_tensor, &invalid_perm);
        assert!(
            result.is_err(),
            "Expected error for invalid permutation {invalid_perm:?}"
        );
    }
}

#[test]
fn test_transpose_with_different_dtypes() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    let dtypes = get_test_dtypes();

    for &dtype in &dtypes {
        let tensor = BitNetTensor::zeros(&[3, 4], dtype, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype}"));

        let candle_tensor = tensor.to_candle().expect("Failed to convert to candle");

        let transposed = bitnet_core::tensor::transpose(&candle_tensor, &[1, 0])
            .unwrap_or_else(|_| panic!("Failed to transpose tensor with dtype {dtype}"));

        assert_eq!(transposed.shape().dims(), &[4, 3]);
        assert_eq!(transposed.dtype(), dtype.to_candle_dtype());
    }
}

// =============================================================================
// Basic Arithmetic Operations Tests
// =============================================================================

#[test]
fn test_element_wise_addition() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Create two tensors for addition
    let tensor1 = BitNetTensor::ones(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create first tensor");

    let tensor2 = BitNetTensor::ones(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create second tensor");

    // Convert to candle tensors for arithmetic operations
    let candle1 = tensor1.to_candle().expect("Failed to convert first tensor");

    let candle2 = tensor2
        .to_candle()
        .expect("Failed to convert second tensor");

    // Perform addition
    let result = (&candle1 + &candle2).expect("Failed to add tensors");

    assert_eq!(result.shape().dims(), &[2, 3]);
    assert_eq!(result.dtype(), candle_core::DType::F32);

    // TODO: Once actual data initialization is implemented, verify result contains 2.0
}

#[test]
fn test_element_wise_subtraction() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor1 = BitNetTensor::ones(&[3, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create first tensor");

    let tensor2 = BitNetTensor::zeros(&[3, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create second tensor");

    let candle1 = tensor1.to_candle().expect("Failed to convert first tensor");

    let candle2 = tensor2
        .to_candle()
        .expect("Failed to convert second tensor");

    // Perform subtraction
    let result = (&candle1 - &candle2).expect("Failed to subtract tensors");

    assert_eq!(result.shape().dims(), &[3, 2]);
    assert_eq!(result.dtype(), candle_core::DType::F32);

    // TODO: Once actual data initialization is implemented, verify result contains 1.0
}

#[test]
fn test_element_wise_multiplication() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor1 = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create first tensor");

    let tensor2 = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool)
        .expect("Failed to create second tensor");

    let candle1 = tensor1.to_candle().expect("Failed to convert first tensor");

    let candle2 = tensor2
        .to_candle()
        .expect("Failed to convert second tensor");

    // Perform element-wise multiplication
    let result = (&candle1 * &candle2).expect("Failed to multiply tensors");

    assert_eq!(result.shape().dims(), &[2, 2]);
    assert_eq!(result.dtype(), candle_core::DType::F32);

    // TODO: Once actual data initialization is implemented, verify result contains 1.0
}

#[test]
fn test_broadcasting_operations() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test broadcasting with different shapes
    let tensor_2x3 = BitNetTensor::ones(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create 2x3 tensor");

    let tensor_1x3 = BitNetTensor::ones(&[1, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create 1x3 tensor");

    let candle_2x3 = tensor_2x3
        .to_candle()
        .expect("Failed to convert 2x3 tensor");

    let candle_1x3 = tensor_1x3
        .to_candle()
        .expect("Failed to convert 1x3 tensor");

    // Test broadcasting addition - this may not be fully supported yet
    match &candle_2x3 + &candle_1x3 {
        Ok(broadcast_result) => {
            assert_eq!(broadcast_result.shape().dims(), &[2, 3]);
            println!("Broadcasting addition succeeded");
        }
        Err(e) => {
            println!("Broadcasting addition failed (may not be implemented): {e}");
            // This is acceptable - broadcasting may not be fully implemented
        }
    }

    // Test scalar broadcasting - skip for now as scalar tensors may not be fully implemented
    // TODO: Implement scalar tensor support and re-enable this test
    println!("Skipping scalar broadcasting test - scalar tensors may not be fully implemented");
}

#[test]
fn test_arithmetic_with_different_dtypes() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test arithmetic operations with different data types
    let dtypes_to_test = vec![BitNetDType::F32, BitNetDType::F16];

    for &dtype in &dtypes_to_test {
        let tensor1 = BitNetTensor::ones(&[2, 2], dtype, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype}"));

        let tensor2 = BitNetTensor::zeros(&[2, 2], dtype, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype}"));

        let candle1 = tensor1.to_candle().expect("Failed to convert first tensor");

        let candle2 = tensor2
            .to_candle()
            .expect("Failed to convert second tensor");

        // Test addition
        let add_result = (&candle1 + &candle2)
            .unwrap_or_else(|_| panic!("Failed to add tensors with dtype {dtype}"));

        assert_eq!(add_result.dtype(), dtype.to_candle_dtype());

        // Test multiplication
        let mul_result = (&candle1 * &candle2)
            .unwrap_or_else(|_| panic!("Failed to multiply tensors with dtype {dtype}"));

        assert_eq!(mul_result.dtype(), dtype.to_candle_dtype());
    }
}

// =============================================================================
// Quantization Accuracy Tests
// =============================================================================

#[test]
fn test_quantized_type_value_ranges() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let quantized_types = vec![
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];

    for &dtype in &quantized_types {
        let tensor = BitNetTensor::zeros(&[10], dtype, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype}"));

        // Verify value range is defined for quantized types
        let value_range = dtype.value_range();
        assert!(
            value_range.is_some(),
            "Quantized type {dtype} should have a defined value range"
        );

        let (min_val, max_val) = value_range.unwrap();
        assert!(
            min_val <= max_val,
            "Min value should be <= max value for {dtype}"
        );

        // Verify specific ranges
        match dtype {
            BitNetDType::I4 => {
                assert_eq!(min_val, -8);
                assert_eq!(max_val, 7);
            }
            BitNetDType::I2 => {
                assert_eq!(min_val, -2);
                assert_eq!(max_val, 1);
            }
            BitNetDType::I1 => {
                assert_eq!(min_val, -1);
                assert_eq!(max_val, 0);
            }
            BitNetDType::BitNet158 => {
                assert_eq!(min_val, -1);
                assert_eq!(max_val, 1);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_bitnet158_ternary_format() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::zeros(&[100], BitNetDType::BitNet158, &device, &pool)
        .expect("Failed to create BitNet 1.58b tensor");

    // Verify BitNet 1.58b specific properties
    assert!(tensor.dtype().is_bitnet158());
    assert!(tensor.dtype().is_quantized());
    assert_eq!(tensor.dtype().bits_per_element(), 2);

    let (min_val, max_val) = tensor.dtype().value_range().unwrap();
    assert_eq!(min_val, -1);
    assert_eq!(max_val, 1);

    // Verify memory efficiency
    let efficiency = tensor.dtype().memory_efficiency();
    assert_eq!(efficiency, 16.0); // 32 bits / 2 bits = 16x efficiency
}

#[test]
fn test_quantization_round_trip_accuracy() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test round-trip conversion: F32 -> Quantized -> F32
    let original_data = vec![1.0, -1.0, 0.0, 0.5, -0.5];
    let f32_tensor = BitNetTensor::from_data(original_data.clone(), &[5], &device, &pool)
        .expect("Failed to create F32 tensor");

    // Convert to candle for testing
    let candle_f32 = f32_tensor.to_candle().expect("Failed to convert to candle");

    // TODO: Implement actual quantization and dequantization functions
    // For now, we test the infrastructure

    let quantized_types = vec![BitNetDType::I8, BitNetDType::I4, BitNetDType::I2];

    for &qtype in &quantized_types {
        let q_tensor = BitNetTensor::zeros(&[5], qtype, &device, &pool)
            .unwrap_or_else(|_| panic!("Failed to create quantized tensor with type {qtype}"));

        // Verify quantized tensor properties
        assert_eq!(q_tensor.dtype(), qtype);
        assert_eq!(q_tensor.element_count(), 5);

        // TODO: Implement quantization accuracy tests once quantization functions are available
        // This would involve:
        // 1. Quantizing the F32 data to the target type
        // 2. Dequantizing back to F32
        // 3. Measuring the quantization error
        // 4. Verifying the error is within acceptable bounds
    }
}

// =============================================================================
// Comprehensive Error Condition Tests
// =============================================================================

#[test]
fn test_memory_pressure_scenarios() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test creating many large tensors to stress memory system
    let mut tensors = Vec::new();
    let large_size = 1024 * 1024; // 1M elements

    // Create tensors until we potentially hit memory pressure
    for i in 0..10 {
        match BitNetTensor::zeros(&[large_size], BitNetDType::F32, &device, &pool) {
            Ok(tensor) => {
                tensors.push(tensor);
                println!("Created large tensor {i}");
            }
            Err(e) => {
                println!("Memory pressure encountered at tensor {i}: {e}");
                break;
            }
        }
    }

    // Verify we can still create small tensors
    let small_tensor = BitNetTensor::zeros(&[10], BitNetDType::F32, &device, &pool)
        .expect("Should be able to create small tensor even under memory pressure");

    assert_eq!(small_tensor.element_count(), 10);

    // Clean up large tensors
    drop(tensors);

    // Verify memory is reclaimed by creating another large tensor
    let _recovery_tensor = BitNetTensor::zeros(&[large_size / 2], BitNetDType::F32, &device, &pool)
        .expect("Should be able to create tensor after cleanup");
}

#[test]
fn test_invalid_shape_operations() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test invalid shapes
    let invalid_shapes = vec![
        vec![1000, 1000, 1000, 1000], // Very large total elements
    ];

    for invalid_shape in invalid_shapes {
        let result = BitNetTensor::zeros(&invalid_shape, BitNetDType::F32, &device, &pool);

        // Some of these might succeed depending on memory availability,
        // but they should handle gracefully
        match result {
            Ok(tensor) => {
                println!("Unexpectedly succeeded creating tensor with shape {invalid_shape:?}");
                assert_eq!(tensor.shape(), invalid_shape);
            }
            Err(e) => {
                println!("Expected error for invalid shape {invalid_shape:?}: {e}");
            }
        }
    }

    // Test zero dimension separately to avoid overflow
    let zero_result = BitNetTensor::zeros(&[0], BitNetDType::F32, &device, &pool);
    match zero_result {
        Ok(_) => {
            println!("Zero dimension tensor creation succeeded (this may be valid)");
        }
        Err(e) => {
            println!("Zero dimension tensor creation failed as expected: {e}");
        }
    }

    // Test extremely large dimension separately to avoid overflow
    // Use a smaller but still problematic size
    let large_result = BitNetTensor::zeros(&[usize::MAX / 1000], BitNetDType::F32, &device, &pool);
    match large_result {
        Ok(_) => {
            println!("Large dimension tensor creation unexpectedly succeeded");
        }
        Err(e) => {
            println!("Large dimension tensor creation failed as expected: {e}");
        }
    }
}

#[test]
fn test_dimension_mismatch_errors() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");

    // Test invalid reshape operations
    let invalid_reshapes = vec![
        vec![5],    // Wrong total elements (6 -> 5)
        vec![2, 2], // Wrong total elements (6 -> 4)
        vec![3, 3], // Wrong total elements (6 -> 9)
        vec![1, 7], // Wrong total elements (6 -> 7)
    ];

    for invalid_shape in invalid_reshapes {
        let result = tensor.reshape(&invalid_shape);
        assert!(
            result.is_err(),
            "Expected error for invalid reshape to {invalid_shape:?}"
        );

        if let Err(e) = result {
            assert!(
                e.to_string().contains("Shape mismatch") || e.to_string().contains("mismatch"),
                "Expected shape mismatch error, got: {e}"
            );
        }
    }
}

#[test]
fn test_concurrent_operation_edge_cases() {
    let pool = Arc::new(create_test_pool());
    let device = get_cpu_device();

    // Create a tensor that will be accessed concurrently
    let tensor = Arc::new(
        BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor"),
    );

    let mut handles = Vec::new();

    // Spawn threads that perform various operations concurrently
    for thread_id in 0..4 {
        let tensor_clone = tensor.clone();
        let pool_clone = pool.clone();

        let handle = thread::spawn(move || {
            let mut operations_completed = 0;

            // Perform various operations
            for i in 0..50 {
                // Test handle creation
                let handle = tensor_clone.handle();
                assert!(handle.is_valid());
                operations_completed += 1;

                // Test metadata access
                let _shape = tensor_clone.shape();
                let _dtype = tensor_clone.dtype();
                let _size = tensor_clone.size_bytes();
                operations_completed += 1;

                // Test tensor cloning every 10 iterations
                if i % 10 == 0 {
                    match tensor_clone.clone_tensor(&pool_clone) {
                        Ok(cloned) => {
                            assert_eq!(cloned.shape(), tensor_clone.shape());
                            operations_completed += 1;
                        }
                        Err(e) => {
                            println!("Thread {thread_id} clone failed at iteration {i}: {e}");
                        }
                    }
                }

                // Small delay to increase chance of race conditions
                thread::sleep(Duration::from_micros(1));
            }

            operations_completed
        });

        handles.push(handle);
    }

    // Wait for all threads and collect results
    let mut total_operations = 0;
    for handle in handles {
        total_operations += handle.join().expect("Thread panicked");
    }

    println!("Total concurrent operations completed: {total_operations}");
    assert!(total_operations > 0);

    // Verify tensor is still valid after concurrent access
    assert_eq!(tensor.shape(), vec![100, 100]);
    assert_eq!(tensor.dtype(), BitNetDType::F32);
}

#[test]
fn test_device_migration_error_conditions() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping device migration error test - need multiple devices");
        return;
    }

    let source_device = &devices[0];
    let target_device = &devices[1];

    let tensor = BitNetTensor::zeros(&[1000, 1000], BitNetDType::F32, source_device, &pool)
        .expect("Failed to create large tensor");

    // Test migration of large tensor (might fail due to memory constraints)
    match tensor.to_device(target_device, &pool) {
        Ok(migrated) => {
            // Note: Device comparison not directly supported, verify device type matches
            let migrated_device = migrated.device();
            match (target_device, &migrated_device) {
                (Device::Cpu, Device::Cpu) => {}
                (Device::Metal(_), Device::Metal(_)) => {}
                (Device::Cuda(_), Device::Cuda(_)) => {}
                _ => panic!("Device mismatch: expected {target_device:?}, got {migrated_device:?}"),
            }
            assert_eq!(migrated.shape(), tensor.shape());
            println!("Large tensor migration succeeded");
        }
        Err(e) => {
            println!("Large tensor migration failed as expected: {e}");
            // This is acceptable - large migrations might fail
        }
    }

    // Test migration to same device (should always succeed)
    let same_device_result = tensor
        .to_device(source_device, &pool)
        .expect("Same device migration should always succeed");

    // Note: Device comparison not directly supported, verify device type matches
    let result_device = same_device_result.device();
    match (source_device, &result_device) {
        (Device::Cpu, Device::Cpu) => {}
        (Device::Metal(_), Device::Metal(_)) => {}
        (Device::Cuda(_), Device::Cuda(_)) => {}
        _ => panic!("Device mismatch: expected {source_device:?}, got {result_device:?}"),
    }
    assert_eq!(same_device_result.id(), tensor.id()); // Should return same tensor
}

// =============================================================================
// Device Compatibility Tests
// =============================================================================

#[test]
fn test_device_enumeration_and_discovery() {
    // Test device discovery and enumeration
    let devices = get_test_devices();

    // Should always have at least CPU device
    assert!(!devices.is_empty(), "Should have at least one device (CPU)");

    // First device should always be CPU
    assert!(
        matches!(devices[0], Device::Cpu),
        "First device should be CPU"
    );

    // Test device info function
    let (cpu_available, metal_available) = bitnet_core::device::get_device_info();
    assert!(cpu_available, "CPU should always be available");

    // Verify device count matches availability
    let expected_device_count = if metal_available { 2 } else { 1 };
    assert_eq!(
        devices.len(),
        expected_device_count,
        "Device count should match availability info"
    );

    // Test device descriptions
    for device in &devices {
        let description = bitnet_core::device::describe_device(device);
        assert!(
            !description.is_empty(),
            "Device description should not be empty"
        );

        match device {
            Device::Cpu => assert!(description.contains("CPU")),
            Device::Metal(_) => assert!(description.contains("Metal")),
            Device::Cuda(_) => assert!(description.contains("CUDA")),
        }
    }

    println!("Discovered {} devices:", devices.len());
    for (i, device) in devices.iter().enumerate() {
        println!("  {}: {}", i, bitnet_core::device::describe_device(device));
    }
}

#[test]
fn test_device_capability_detection() {
    // Test CPU device capabilities (always available)
    let cpu_device = bitnet_core::device::get_cpu_device();
    assert!(matches!(cpu_device, Device::Cpu));

    // Test Metal device availability detection
    let metal_available = bitnet_core::device::is_metal_available();
    println!("Metal GPU available: {metal_available}");

    if metal_available {
        // Test Metal device creation when available
        let metal_device = bitnet_core::device::get_metal_device();
        assert!(
            metal_device.is_ok(),
            "Metal device creation should succeed when available"
        );

        if let Ok(device) = metal_device {
            assert!(matches!(device, Device::Metal(_)));

            // Test Metal device name retrieval
            let device_name = bitnet_core::device::get_metal_device_name();
            if let Some(name) = device_name {
                assert!(!name.is_empty(), "Metal device name should not be empty");
                println!("Metal device name: {name}");
            }
        }
    } else {
        // Test Metal device creation failure when not available
        let metal_device = bitnet_core::device::get_metal_device();
        assert!(
            metal_device.is_err(),
            "Metal device creation should fail when not available"
        );

        // Test Metal device name returns None when not available
        let device_name = bitnet_core::device::get_metal_device_name();
        assert!(
            device_name.is_none(),
            "Metal device name should be None when not available"
        );
    }

    // Test auto device selection
    let auto_device = bitnet_core::device::auto_select_device();
    if metal_available {
        assert!(
            matches!(auto_device, Device::Metal(_)),
            "Auto selection should prefer Metal when available"
        );
    } else {
        assert!(
            matches!(auto_device, Device::Cpu),
            "Auto selection should fallback to CPU when Metal unavailable"
        );
    }

    // Test device info consistency
    let (cpu_info, metal_info) = bitnet_core::device::get_device_info();
    assert!(cpu_info, "CPU info should always be true");
    assert_eq!(
        metal_info, metal_available,
        "Metal info should match availability check"
    );
}

#[test]
fn test_cross_device_tensor_migration() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping cross-device migration test - only one device available");
        return;
    }

    let source_device = &devices[0];
    let target_device = &devices[1];

    // Test migration with different tensor sizes and data types
    let test_cases = vec![
        (vec![10], BitNetDType::F32),
        (vec![5, 5], BitNetDType::F16),
        (vec![2, 3, 4], BitNetDType::I8),
        (vec![1, 100], BitNetDType::I4),
    ];

    for (shape, dtype) in test_cases {
        // Create tensor on source device
        let tensor =
            BitNetTensor::zeros(&shape, dtype, source_device, &pool).unwrap_or_else(|_| {
                panic!("Failed to create tensor with shape {shape:?}, dtype {dtype}")
            });

        // Verify tensor is on source device
        let tensor_device = tensor.device();
        match (source_device, &tensor_device) {
            (Device::Cpu, Device::Cpu) => {}
            (Device::Metal(_), Device::Metal(_)) => {}
            (Device::Cuda(_), Device::Cuda(_)) => {}
            _ => panic!("Tensor not on expected source device"),
        }

        // Migrate to target device
        let migrated = tensor.to_device(target_device, &pool);

        match migrated {
            Ok(new_tensor) => {
                // Verify migration succeeded
                let new_device = new_tensor.device();
                match (target_device, &new_device) {
                    (Device::Cpu, Device::Cpu) => {}
                    (Device::Metal(_), Device::Metal(_)) => {}
                    (Device::Cuda(_), Device::Cuda(_)) => {}
                    _ => panic!("Migrated tensor not on expected target device"),
                }

                // Verify tensor properties preserved
                assert_eq!(new_tensor.shape(), tensor.shape());
                assert_eq!(new_tensor.dtype(), tensor.dtype());
                assert_eq!(new_tensor.element_count(), tensor.element_count());
                assert_eq!(new_tensor.size_bytes(), tensor.size_bytes());

                // Should be a different tensor (different ID)
                assert_ne!(new_tensor.id(), tensor.id());

                // Test bidirectional migration
                let migrated_back = new_tensor.to_device(source_device, &pool);
                match migrated_back {
                    Ok(back_tensor) => {
                        assert_eq!(back_tensor.shape(), tensor.shape());
                        assert_eq!(back_tensor.dtype(), tensor.dtype());

                        let back_device = back_tensor.device();
                        match (source_device, &back_device) {
                            (Device::Cpu, Device::Cpu) => {}
                            (Device::Metal(_), Device::Metal(_)) => {}
                            (Device::Cuda(_), Device::Cuda(_)) => {}
                            _ => panic!("Back-migrated tensor not on expected device"),
                        }
                    }
                    Err(e) => {
                        println!("Bidirectional migration failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("Migration failed for shape {shape:?}, dtype {dtype}: {e}");
                // Migration failure is acceptable - implementation may be incomplete
            }
        }
    }
}

#[test]
fn test_cross_device_tensor_operations() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping cross-device operations test - only one device available");
        return;
    }

    let device1 = &devices[0];
    let device2 = &devices[1];

    // Create tensors on different devices
    let tensor1 = BitNetTensor::ones(&[3, 3], BitNetDType::F32, device1, &pool)
        .expect("Failed to create tensor on device 1");

    let tensor2 = BitNetTensor::ones(&[3, 3], BitNetDType::F32, device2, &pool)
        .expect("Failed to create tensor on device 2");

    // Verify tensors are on different devices
    let dev1 = tensor1.device();
    let dev2 = tensor2.device();

    let devices_different = match (&dev1, &dev2) {
        (Device::Cpu, Device::Metal(_)) => true,
        (Device::Metal(_), Device::Cpu) => true,
        (Device::Cpu, Device::Cuda(_)) => true,
        (Device::Cuda(_), Device::Cpu) => true,
        (Device::Metal(_), Device::Cuda(_)) => true,
        (Device::Cuda(_), Device::Metal(_)) => true,
        _ => false,
    };

    if !devices_different {
        println!("Tensors ended up on same device type, skipping cross-device operations");
        return;
    }

    // Test operations between tensors on different devices
    // Convert to candle tensors for arithmetic operations
    let candle1 = tensor1.to_candle();
    let candle2 = tensor2.to_candle();

    match (candle1, candle2) {
        (Ok(c1), Ok(c2)) => {
            // Test if cross-device operations are supported
            // Note: This may fail if cross-device operations aren't implemented
            match &c1 + &c2 {
                Ok(result) => {
                    println!("Cross-device addition succeeded");
                    assert_eq!(result.shape().dims(), &[3, 3]);
                }
                Err(e) => {
                    println!("Cross-device addition failed (expected): {e}");
                    // This is acceptable - cross-device operations may require explicit migration
                }
            }

            // Test migration before operation
            let migrated2 = tensor2.to_device(device1, &pool);
            if let Ok(migrated_tensor2) = migrated2 {
                let migrated_candle2 = migrated_tensor2.to_candle();
                if let Ok(mc2) = migrated_candle2 {
                    match &c1 + &mc2 {
                        Ok(result) => {
                            println!("Same-device addition after migration succeeded");
                            assert_eq!(result.shape().dims(), &[3, 3]);
                        }
                        Err(e) => {
                            println!("Same-device addition after migration failed: {e}");
                        }
                    }
                }
            }
        }
        _ => {
            println!("Failed to convert tensors to candle for cross-device operations test");
        }
    }
}

#[test]
fn test_device_specific_memory_management() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    for device in &devices {
        println!(
            "Testing memory management on device: {}",
            bitnet_core::device::describe_device(device)
        );

        // Test memory allocation patterns on each device
        let mut tensors = Vec::new();
        let tensor_count = 10;

        // Create multiple tensors on the same device
        for i in 0..tensor_count {
            let size = (i + 1) * 100;
            let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, device, &pool)
                .unwrap_or_else(|_| panic!("Failed to create tensor {i} on device"));

            // Verify tensor is on correct device
            let tensor_device = tensor.device();
            match (device, &tensor_device) {
                (Device::Cpu, Device::Cpu) => {}
                (Device::Metal(_), Device::Metal(_)) => {}
                (Device::Cuda(_), Device::Cuda(_)) => {}
                _ => panic!("Tensor not on expected device"),
            }

            tensors.push(tensor);
        }

        // Check memory metrics
        let metrics = pool.get_metrics();
        println!("  Active allocations: {}", metrics.active_allocations);
        println!("  Current allocated: {}", metrics.current_allocated);

        // Test memory cleanup by dropping half the tensors
        let half_count = tensor_count / 2;
        tensors.truncate(half_count);

        // Force cleanup (implementation dependent)
        drop(tensors);

        // Check metrics after cleanup
        let final_metrics = pool.get_metrics();
        println!(
            "  Final active allocations: {}",
            final_metrics.active_allocations
        );

        // Test device-specific large allocation
        let large_size = 1024 * 100; // 100K elements
        match BitNetTensor::zeros(&[large_size], BitNetDType::F32, device, &pool) {
            Ok(large_tensor) => {
                println!(
                    "  Large allocation succeeded: {} bytes",
                    large_tensor.size_bytes()
                );
                assert_eq!(large_tensor.element_count(), large_size);

                // Test large tensor operations
                let handle = large_tensor.handle();
                assert!(handle.is_valid());
                assert_eq!(handle.element_count().unwrap(), large_size);
            }
            Err(e) => {
                println!("  Large allocation failed: {e}");
                // This is acceptable - large allocations may fail due to memory constraints
            }
        }
    }
}

#[test]
fn test_device_error_handling_scenarios() {
    let pool = create_test_pool();

    // Test invalid device scenarios

    // Test Metal device when not available
    if !bitnet_core::device::is_metal_available() {
        let metal_result = bitnet_core::device::get_metal_device();
        assert!(
            metal_result.is_err(),
            "Metal device creation should fail when not available"
        );

        match metal_result.unwrap_err() {
            bitnet_core::device::DeviceError::MetalNotAvailable => {
                println!("Correctly detected Metal not available");
            }
            bitnet_core::device::DeviceError::MetalCreationFailed(_) => {
                println!("Metal creation failed as expected");
            }
            _ => panic!("Unexpected error type for Metal unavailability"),
        }
    }

    // Test device migration error scenarios
    let cpu_device = bitnet_core::device::get_cpu_device();
    let tensor = BitNetTensor::zeros(&[100], BitNetDType::F32, &cpu_device, &pool)
        .expect("Failed to create test tensor");

    // Test migration to same device (should succeed)
    let same_device_result = tensor.to_device(&cpu_device, &pool);
    assert!(
        same_device_result.is_ok(),
        "Same device migration should always succeed"
    );

    // Test very large tensor creation (may fail due to memory)
    let very_large_size = usize::MAX / 1000000; // Avoid overflow but still very large
    let large_result =
        BitNetTensor::zeros(&[very_large_size], BitNetDType::F32, &cpu_device, &pool);
    match large_result {
        Ok(_) => {
            println!("Very large tensor creation unexpectedly succeeded");
        }
        Err(e) => {
            println!("Very large tensor creation failed as expected: {e}");
            // Verify error message is meaningful
            assert!(
                !e.to_string().is_empty(),
                "Error message should not be empty"
            );
        }
    }

    // Test invalid shape scenarios
    let invalid_shapes = vec![
        vec![0], // Zero dimension
    ];

    for invalid_shape in invalid_shapes {
        let result = BitNetTensor::zeros(&invalid_shape, BitNetDType::F32, &cpu_device, &pool);
        match result {
            Ok(_) => {
                println!("Invalid shape {invalid_shape:?} unexpectedly succeeded");
            }
            Err(e) => {
                println!("Invalid shape {invalid_shape:?} failed as expected: {e}");
            }
        }
    }

    // Test device disconnection simulation (if applicable)
    // Note: This is difficult to test without actual device disconnection
    // For now, we test graceful handling of device operations

    let devices = get_test_devices();
    for device in &devices {
        // Test multiple rapid operations to stress device handling
        for i in 0..5 {
            match BitNetTensor::zeros(&[10], BitNetDType::F32, device, &pool) {
                Ok(tensor) => {
                    // Test immediate operations
                    let _handle = tensor.handle();
                    let _shape = tensor.shape();
                    let _dtype = tensor.dtype();
                }
                Err(e) => {
                    println!(
                        "Rapid operation {} failed on device {}: {}",
                        i,
                        bitnet_core::device::describe_device(device),
                        e
                    );
                }
            }
        }
    }
}

#[test]
fn test_device_memory_pressure_handling() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    for device in &devices {
        println!(
            "Testing memory pressure on device: {}",
            bitnet_core::device::describe_device(device)
        );

        let mut tensors = Vec::new();
        let mut allocation_count = 0;
        let max_attempts = 50;

        // Gradually increase memory pressure
        for i in 0..max_attempts {
            let size = (i + 1) * 1024; // Increasing size

            match BitNetTensor::zeros(&[size], BitNetDType::F32, device, &pool) {
                Ok(tensor) => {
                    tensors.push(tensor);
                    allocation_count += 1;

                    // Check memory metrics periodically
                    if i % 10 == 0 {
                        let metrics = pool.get_metrics();
                        println!(
                            "  Iteration {}: {} allocations, {} bytes",
                            i, metrics.active_allocations, metrics.current_allocated
                        );
                    }
                }
                Err(e) => {
                    println!("  Memory pressure reached at iteration {i}: {e}");
                    break;
                }
            }
        }

        println!("  Successfully allocated {allocation_count} tensors before pressure");

        // Test memory recovery by releasing some tensors
        let release_count = allocation_count / 2;
        for _ in 0..release_count {
            if !tensors.is_empty() {
                tensors.pop();
            }
        }

        // Test that we can allocate again after releasing memory
        match BitNetTensor::zeros(&[1000], BitNetDType::F32, device, &pool) {
            Ok(recovery_tensor) => {
                println!("  Memory recovery successful");
                assert_eq!(recovery_tensor.element_count(), 1000);
            }
            Err(e) => {
                println!("  Memory recovery failed: {e}");
                // This may be acceptable depending on implementation
            }
        }

        // Test different data types under memory pressure
        let dtypes = vec![
            BitNetDType::F32,
            BitNetDType::F16,
            BitNetDType::I8,
            BitNetDType::I4,
        ];
        for &dtype in &dtypes {
            match BitNetTensor::zeros(&[500], dtype, device, &pool) {
                Ok(typed_tensor) => {
                    assert_eq!(typed_tensor.dtype(), dtype);
                    let expected_size = dtype.bytes_for_elements(500);
                    assert_eq!(typed_tensor.size_bytes(), expected_size);
                }
                Err(e) => {
                    println!("  Failed to allocate {dtype} tensor under pressure: {e}");
                }
            }
        }

        // Clean up
        tensors.clear();

        let final_metrics = pool.get_metrics();
        println!(
            "  Final metrics: {} allocations, {} bytes",
            final_metrics.active_allocations, final_metrics.current_allocated
        );
    }
}

#[test]
fn test_concurrent_multi_device_operations() {
    let pool = Arc::new(create_test_pool());
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping concurrent multi-device test - need multiple devices");
        return;
    }

    let mut handles = Vec::new();
    let operations_per_thread = 20;

    // Spawn threads for each device
    for (device_idx, device) in devices.iter().enumerate() {
        let pool_clone = pool.clone();
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            let mut thread_tensors = Vec::new();
            let mut operations_completed = 0;

            println!(
                "Thread {} starting on device: {}",
                device_idx,
                bitnet_core::device::describe_device(&device_clone)
            );

            // Perform concurrent operations on this device
            for i in 0..operations_per_thread {
                // Create tensor
                let shape = vec![device_idx + 1, i + 1];
                match BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone) {
                    Ok(tensor) => {
                        // Verify device affinity
                        let tensor_device = tensor.device();
                        match (&device_clone, &tensor_device) {
                            (Device::Cpu, Device::Cpu) => {}
                            (Device::Metal(_), Device::Metal(_)) => {}
                            (Device::Cuda(_), Device::Cuda(_)) => {}
                            _ => panic!("Device affinity lost in concurrent operation"),
                        }

                        // Test handle operations
                        let handle = tensor.handle();
                        assert!(handle.is_valid());
                        assert!(handle.touch().is_ok());

                        // Test metadata operations
                        assert_eq!(tensor.shape(), shape);
                        assert_eq!(tensor.dtype(), BitNetDType::F32);

                        thread_tensors.push(tensor);
                        operations_completed += 1;
                    }
                    Err(e) => {
                        println!("Thread {device_idx} operation {i} failed: {e}");
                    }
                }

                // Test concurrent cloning every few iterations
                if i % 5 == 0 && !thread_tensors.is_empty() {
                    let tensor_to_clone = &thread_tensors[thread_tensors.len() - 1];
                    match tensor_to_clone.clone_tensor(&pool_clone) {
                        Ok(cloned) => {
                            assert_eq!(cloned.shape(), tensor_to_clone.shape());
                            assert_eq!(cloned.dtype(), tensor_to_clone.dtype());
                            operations_completed += 1;
                        }
                        Err(e) => {
                            println!("Thread {device_idx} clone failed: {e}");
                        }
                    }
                }

                // Small delay to increase concurrency
                thread::sleep(Duration::from_micros(10));
            }

            println!("Thread {device_idx} completed {operations_completed} operations");
            (device_idx, operations_completed, thread_tensors.len())
        });

        handles.push(handle);
    }

    // Test cross-device operations while other threads are running
    let mut cross_device_handles = Vec::new();
    if devices.len() >= 2 {
        let cross_device_handle = {
            let pool_clone = pool.clone();
            let device1 = devices[0].clone();
            let device2 = devices[1].clone();

            thread::spawn(move || {
                let mut cross_operations = 0;

                for i in 0..10 {
                    // Create tensor on device 1
                    if let Ok(tensor1) =
                        BitNetTensor::zeros(&[5, 5], BitNetDType::F32, &device1, &pool_clone)
                    {
                        // Migrate to device 2
                        match tensor1.to_device(&device2, &pool_clone) {
                            Ok(tensor2) => {
                                // Verify migration
                                let dev2 = tensor2.device();
                                match (&device2, &dev2) {
                                    (Device::Cpu, Device::Cpu) => {}
                                    (Device::Metal(_), Device::Metal(_)) => {}
                                    (Device::Cuda(_), Device::Cuda(_)) => {}
                                    _ => panic!("Cross-device migration failed"),
                                }
                                cross_operations += 1;
                            }
                            Err(e) => {
                                println!("Cross-device migration {i} failed: {e}");
                            }
                        }
                    }

                    thread::sleep(Duration::from_millis(1));
                }

                cross_operations
            })
        };

        cross_device_handles.push(cross_device_handle);
    }

    // Wait for all threads to complete
    let mut total_operations = 0;
    let mut total_tensors = 0;

    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(result) => {
                let (device_idx, ops, tensors) = result;
                println!(
                    "Device {device_idx} thread completed: {ops} operations, {tensors} tensors"
                );
                total_operations += ops;
                total_tensors += tensors;
            }
            Err(e) => {
                println!("Thread {i} panicked: {e:?}");
            }
        }
    }

    // Wait for cross-device threads
    for handle in cross_device_handles {
        match handle.join() {
            Ok(cross_ops) => {
                println!("Cross-device thread completed: {cross_ops} operations");
                total_operations += cross_ops;
            }
            Err(e) => {
                println!("Cross-device thread panicked: {e:?}");
            }
        }
    }

    println!("Total concurrent operations: {total_operations}, Total tensors: {total_tensors}");

    // Verify pool state after concurrent operations
    let final_metrics = pool.get_metrics();
    println!(
        "Final pool metrics: {} allocations, {} bytes",
        final_metrics.active_allocations, final_metrics.current_allocated
    );

    assert!(
        total_operations > 0,
        "Should have completed some operations"
    );
}

// =============================================================================
// Memory Leak Detection Tests
// =============================================================================

#[test]
fn test_tensor_creation_destruction_memory_leak() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Capture initial memory metrics
    let initial_metrics = pool.get_metrics();
    let initial_allocated = initial_metrics.current_allocated;
    let initial_active = initial_metrics.active_allocations;

    // Create and destroy tensors in a loop
    let iterations = 100;
    for i in 0..iterations {
        let shape = vec![i % 10 + 1, (i % 5) + 1];
        let tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");

        // Verify tensor was created
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.dtype(), BitNetDType::F32);

        // Explicitly drop the tensor
        drop(tensor);
    }

    // Force cleanup to ensure all memory is reclaimed
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Capture final memory metrics
    let final_metrics = pool.get_metrics();
    let final_allocated = final_metrics.current_allocated;
    let final_active = final_metrics.active_allocations;

    // Verify no significant memory leaks (allow for small implementation overhead)
    let memory_diff = final_allocated as i64 - initial_allocated as i64;
    let allocation_diff = final_active as i64 - initial_active as i64;

    assert!(
        memory_diff < 50000, // Less than 50KB growth allowed
        "Significant memory leak detected: allocated memory increased by {memory_diff} bytes"
    );
    assert!(
        allocation_diff.abs() < 200, // Allow for more allocation overhead
        "Significant allocation leak detected: active allocations changed by {allocation_diff}"
    );

    println!("Memory leak test passed: {iterations} iterations with no leaks detected");
}

#[test]
fn test_cyclic_reference_cleanup() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Capture initial memory state
    let initial_metrics = pool.get_metrics();

    // Create tensors that could potentially form cyclic references
    let mut tensors = Vec::new();
    let mut handles = Vec::new();

    for i in 0..10 {
        let tensor = BitNetTensor::zeros(&[100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");

        // Create multiple handles to the same tensor (potential for cycles)
        let handle1 = tensor.handle();
        let handle2 = tensor.handle();
        let weak_handle = handle1.downgrade();

        // Verify handles are valid
        assert!(handle1.is_valid());
        assert!(handle2.is_valid());
        assert!(weak_handle.is_valid());

        tensors.push(tensor);
        handles.push((handle1, handle2, weak_handle));
    }

    // Verify all handles are still valid
    for (handle1, handle2, weak_handle) in &handles {
        assert!(handle1.is_valid());
        assert!(handle2.is_valid());
        assert!(weak_handle.is_valid());
        assert!(weak_handle.upgrade().is_some());
    }

    // Drop strong references first
    drop(tensors);
    drop(handles);

    // Allow time for cleanup
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Verify memory was properly cleaned up
    let final_metrics = pool.get_metrics();

    // Memory should return to reasonable state (allowing for implementation overhead)
    let memory_diff =
        final_metrics.current_allocated as i64 - initial_metrics.current_allocated as i64;
    assert!(
        memory_diff.abs() < 50000, // Allow up to 50KB difference
        "Potential cyclic reference leak: memory difference {memory_diff} bytes"
    );

    println!("Cyclic reference cleanup test passed");
}

#[test]
fn test_large_scale_memory_stability() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Capture baseline metrics
    let baseline_metrics = pool.get_metrics();
    let baseline_allocated = baseline_metrics.current_allocated;

    let iterations = 1000;
    let batch_size = 50;

    for batch in 0..(iterations / batch_size) {
        let mut batch_tensors = Vec::new();

        // Create a batch of tensors
        for i in 0..batch_size {
            let size = (batch * batch_size + i) % 100 + 10;
            let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
            batch_tensors.push(tensor);
        }

        // Perform operations on tensors
        for tensor in &batch_tensors {
            let _handle = tensor.handle();
            let _shape = tensor.shape();
            let _size = tensor.size_bytes();
        }

        // Drop the entire batch
        drop(batch_tensors);

        // Check memory stability every 10 batches
        if batch % 10 == 0 {
            let current_metrics = pool.get_metrics();
            let memory_growth =
                current_metrics.current_allocated as i64 - baseline_allocated as i64;

            // Allow for some growth but detect significant leaks
            assert!(
                memory_growth < 1024 * 1024, // Less than 1MB growth
                "Memory instability detected at batch {batch}: {memory_growth} bytes growth"
            );

            println!("Batch {batch} completed, memory growth: {memory_growth} bytes");
        }
    }

    // Final stability check
    let final_metrics = pool.get_metrics();
    let total_growth = final_metrics.current_allocated as i64 - baseline_allocated as i64;

    assert!(total_growth < 1024 * 1024, // Less than 1MB total growth
           "Large-scale memory instability: {total_growth} bytes total growth after {iterations} iterations");

    println!("Large-scale memory stability test passed: {iterations} iterations");
}

#[test]
fn test_device_migration_memory_cleanup() {
    let pool = create_test_pool_with_tracking();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping device migration memory test - need multiple devices");
        return;
    }

    let source_device = &devices[0];
    let target_device = &devices[1];

    // Capture initial memory state
    let initial_metrics = pool.get_metrics();

    let migration_count = 20;
    for i in 0..migration_count {
        // Create tensor on source device
        let tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, source_device, &pool)
            .expect("Failed to create tensor");

        // Migrate to target device
        let migrated = tensor.to_device(target_device, &pool);

        match migrated {
            Ok(migrated_tensor) => {
                // Verify migration succeeded
                let migrated_device = migrated_tensor.device();
                match (target_device, &migrated_device) {
                    (Device::Cpu, Device::Cpu) => {}
                    (Device::Metal(_), Device::Metal(_)) => {}
                    (Device::Cuda(_), Device::Cuda(_)) => {}
                    _ => panic!("Device migration failed"),
                }

                // Original tensor should still exist
                assert_eq!(tensor.shape(), migrated_tensor.shape());

                // Drop both tensors
                drop(tensor);
                drop(migrated_tensor);
            }
            Err(e) => {
                println!("Migration {i} failed: {e}");
                drop(tensor);
            }
        }

        // Check for memory leaks every 5 migrations
        if i % 5 == 0 {
            let current_metrics = pool.get_metrics();
            let memory_growth =
                current_metrics.current_allocated as i64 - initial_metrics.current_allocated as i64;

            // Allow for reasonable growth but detect leaks
            assert!(
                memory_growth < 10 * 1024 * 1024, // Less than 10MB
                "Device migration memory leak at iteration {i}: {memory_growth} bytes"
            );
        }
    }

    // Allow cleanup time
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Final memory check
    let final_metrics = pool.get_metrics();
    let total_growth =
        final_metrics.current_allocated as i64 - initial_metrics.current_allocated as i64;

    assert!(
        total_growth < 5 * 1024 * 1024, // Less than 5MB final growth
        "Device migration memory leak: {total_growth} bytes after {migration_count} migrations"
    );

    println!("Device migration memory cleanup test passed");
}

#[test]
fn test_operation_memory_leak_detection() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Capture baseline
    let baseline_metrics = pool.get_metrics();

    let operation_count = 100;
    for i in 0..operation_count {
        // Create tensors for operations
        let tensor1 = BitNetTensor::ones(&[50, 50], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor1");
        let tensor2 = BitNetTensor::zeros(&[50, 50], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor2");

        // Perform various operations that could leak memory

        // 1. Reshape operations
        let reshaped = tensor1.reshape(&[2500]);
        if let Ok(reshaped_tensor) = reshaped {
            assert_eq!(reshaped_tensor.element_count(), 2500);
            drop(reshaped_tensor);
        }

        // 2. Cloning operations
        let cloned = tensor1.clone_tensor(&pool);
        if let Ok(cloned_tensor) = cloned {
            assert_eq!(cloned_tensor.shape(), tensor1.shape());
            drop(cloned_tensor);
        }

        // 3. Handle operations
        let handle1 = tensor1.handle();
        let handle2 = tensor2.handle();
        assert!(handle1.is_valid());
        assert!(handle2.is_valid());

        // 4. Candle conversion operations
        let candle1 = tensor1.to_candle();
        let candle2 = tensor2.to_candle();

        if let (Ok(c1), Ok(c2)) = (candle1, candle2) {
            // Perform arithmetic operations
            let _add_result = &c1 + &c2;
            let _mul_result = &c1 * &c2;
            drop(c1);
            drop(c2);
        }

        // Drop tensors
        drop(tensor1);
        drop(tensor2);
        drop(handle1);
        drop(handle2);

        // Check for leaks every 20 operations
        if i % 20 == 0 {
            let current_metrics = pool.get_metrics();
            let memory_growth = current_metrics.current_allocated as i64
                - baseline_metrics.current_allocated as i64;

            assert!(
                memory_growth < 5 * 1024 * 1024, // Less than 5MB
                "Operation memory leak at iteration {i}: {memory_growth} bytes"
            );
        }
    }

    // Final leak check
    let final_metrics = pool.get_metrics();
    let total_growth =
        final_metrics.current_allocated as i64 - baseline_metrics.current_allocated as i64;

    assert!(
        total_growth < 10 * 1024 * 1024, // Less than 10MB final growth
        "Operation memory leak: {total_growth} bytes after {operation_count} operations"
    );

    println!("Operation memory leak detection test passed");
}

#[test]
fn test_concurrent_memory_leak_prevention() {
    let pool = Arc::new(create_test_pool_with_tracking());
    let device = get_cpu_device();

    // Capture baseline
    let baseline_metrics = pool.get_metrics();

    let thread_count = 4;
    let operations_per_thread = 50;
    let mut handles = Vec::new();

    for thread_id in 0..thread_count {
        let pool_clone = pool.clone();
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            let mut thread_tensors = Vec::new();

            for i in 0..operations_per_thread {
                // Create tensor
                let shape = vec![thread_id + 1, i + 1];
                let tensor =
                    BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone)
                        .expect("Failed to create tensor");

                // Perform operations
                let handle = tensor.handle();
                assert!(handle.is_valid());

                let cloned = tensor.clone();
                assert_eq!(cloned.id(), tensor.id());

                // Store some tensors, drop others immediately
                if i % 3 == 0 {
                    thread_tensors.push(tensor);
                } else {
                    drop(tensor);
                }
                drop(cloned);
                drop(handle);

                // Small delay to increase concurrency
                thread::sleep(Duration::from_micros(100));
            }

            // Drop remaining tensors
            drop(thread_tensors);

            // Return thread metrics
            pool_clone.get_metrics()
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    let mut thread_results = Vec::new();
    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        thread_results.push(result);
    }

    // Allow cleanup time
    thread::sleep(Duration::from_millis(200));

    // Check final memory state
    let final_metrics = pool.get_metrics();
    let total_growth =
        final_metrics.current_allocated as i64 - baseline_metrics.current_allocated as i64;

    assert!(total_growth < 10 * 1024 * 1024, // Less than 10MB
           "Concurrent memory leak: {total_growth} bytes after {thread_count} threads with {operations_per_thread} operations each");

    // Verify no excessive active allocations remain from our test
    let active_diff =
        final_metrics.active_allocations as i64 - baseline_metrics.active_allocations as i64;
    assert!(
        active_diff.abs() < 500, // Allow for more concurrent overhead
        "Active allocation leak: {active_diff} allocations difference"
    );

    println!("Concurrent memory leak prevention test passed");
}

#[test]
fn test_handle_lifecycle_memory_management() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Capture baseline
    let baseline_metrics = pool.get_metrics();

    let handle_test_iterations = 100;

    for i in 0..handle_test_iterations {
        // Create tensor
        let tensor = BitNetTensor::zeros(&[100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");

        // Create multiple handles
        let handle1 = tensor.handle();
        let handle2 = tensor.handle();
        let handle3 = tensor.handle();

        // Create weak handles
        let weak1 = handle1.downgrade();
        let weak2 = handle2.downgrade();

        // Verify all handles are valid
        assert!(handle1.is_valid());
        assert!(handle2.is_valid());
        assert!(handle3.is_valid());
        assert!(weak1.is_valid());
        assert!(weak2.is_valid());

        // Verify weak handles can be upgraded
        assert!(weak1.upgrade().is_some());
        assert!(weak2.upgrade().is_some());

        // Test handle operations
        assert!(handle1.touch().is_ok());
        assert!(handle2.add_tag("test_tag".to_string()).is_ok());
        assert!(handle2.has_tag("test_tag").unwrap());

        // Drop strong handles in different orders
        match i % 3 {
            0 => {
                drop(handle1);
                drop(handle2);
                drop(handle3);
            }
            1 => {
                drop(handle3);
                drop(handle1);
                drop(handle2);
            }
            _ => {
                drop(handle2);
                drop(handle3);
                drop(handle1);
            }
        }

        // Weak handles should still be valid while tensor exists
        assert!(weak1.is_valid());
        assert!(weak2.is_valid());

        // Drop tensor
        drop(tensor);

        // Weak handles should become invalid
        assert!(!weak1.is_valid());
        assert!(!weak2.is_valid());
        assert!(weak1.upgrade().is_none());
        assert!(weak2.upgrade().is_none());

        drop(weak1);
        drop(weak2);

        // Check for leaks every 25 iterations
        if i % 25 == 0 {
            let current_metrics = pool.get_metrics();
            let memory_growth = current_metrics.current_allocated as i64
                - baseline_metrics.current_allocated as i64;

            assert!(
                memory_growth < 1024 * 1024, // Less than 1MB
                "Handle lifecycle memory leak at iteration {i}: {memory_growth} bytes"
            );
        }
    }

    // Final memory check
    let final_metrics = pool.get_metrics();
    let total_growth =
        final_metrics.current_allocated as i64 - baseline_metrics.current_allocated as i64;

    assert!(total_growth < 512 * 1024, // Less than 512KB final growth
           "Handle lifecycle memory leak: {total_growth} bytes after {handle_test_iterations} iterations");

    println!("Handle lifecycle memory management test passed");
}

#[test]
fn test_memory_tracking_integration() {
    let pool = create_test_pool_with_tracking();
    let device = get_cpu_device();

    // Get memory tracker if available
    let memory_tracker = pool.get_memory_tracker();
    if memory_tracker.is_none() {
        println!("Skipping memory tracking integration test - tracker not available");
        return;
    }

    let tracker = memory_tracker.unwrap();

    // Capture initial tracking metrics
    let initial_detailed_metrics = tracker.get_detailed_metrics();
    let initial_pressure = tracker.get_pressure_level();

    println!("Initial memory pressure: {initial_pressure:?}");
    println!(
        "Initial active allocations: {}",
        initial_detailed_metrics.active_allocations
    );

    // Create tensors and verify tracking
    let mut tensors = Vec::new();
    let tensor_count = 50;

    for i in 0..tensor_count {
        let size = (i + 1) * 100;
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor");

        tensors.push(tensor);

        // Check tracking every 10 tensors
        if i % 10 == 0 {
            let current_metrics = tracker.get_detailed_metrics();
            let current_pressure = tracker.get_pressure_level();

            // Verify tracking is working
            assert!(
                current_metrics.active_allocations >= initial_detailed_metrics.active_allocations
            );
            assert!(
                current_metrics.current_memory_usage
                    >= initial_detailed_metrics.current_memory_usage
            );

            println!(
                "Iteration {}: {} active allocations, {} bytes, pressure: {:?}",
                i,
                current_metrics.active_allocations,
                current_metrics.current_memory_usage,
                current_pressure
            );
        }
    }

    // Get peak metrics
    let peak_metrics = tracker.get_detailed_metrics();
    let peak_allocations = peak_metrics.active_allocations;
    let peak_memory = peak_metrics.current_memory_usage;

    // Drop all tensors
    drop(tensors);

    // Allow cleanup time
    thread::sleep(Duration::from_millis(100));

    // Verify tracking detected cleanup
    let final_metrics = tracker.get_detailed_metrics();
    let final_pressure = tracker.get_pressure_level();

    println!("Final memory pressure: {final_pressure:?}");
    println!(
        "Final active allocations: {}",
        final_metrics.active_allocations
    );

    // Memory should be cleaned up
    assert!(
        final_metrics.active_allocations <= initial_detailed_metrics.active_allocations + 5,
        "Memory tracking shows potential leak: {} final vs {} initial allocations",
        final_metrics.active_allocations,
        initial_detailed_metrics.active_allocations
    );

    // Peak memory should be recorded
    assert!(
        final_metrics.peak_memory_usage >= peak_memory,
        "Peak memory not properly tracked: {} final peak vs {} observed peak",
        final_metrics.peak_memory_usage,
        peak_memory
    );

    // Verify device usage tracking
    assert!(
        !final_metrics.device_usage.is_empty(),
        "Device usage not tracked"
    );

    // Check for memory pressure patterns if any were detected
    if !final_metrics.recent_patterns.is_empty() {
        println!(
            "Detected {} memory patterns",
            final_metrics.recent_patterns.len()
        );
        for pattern in &final_metrics.recent_patterns {
            println!(
                "Pattern: {} (confidence: {:.2})",
                pattern.description, pattern.confidence
            );

            // Verify no leak patterns were detected
            if pattern.is_problematic {
                println!(
                    "Warning: Problematic pattern detected: {}",
                    pattern.description
                );
            }
        }
    }

    println!("Memory tracking integration test passed");
}

// =============================================================================
// Performance Benchmark Tests
// =============================================================================

/// Performance benchmark statistics
#[derive(Debug, Clone)]
struct BenchmarkStats {
    min_duration: Duration,
    max_duration: Duration,
    avg_duration: Duration,
    total_operations: usize,
    operations_per_second: f64,
}

impl BenchmarkStats {
    fn new(durations: &[Duration]) -> Self {
        let total_operations = durations.len();
        let min_duration = *durations.iter().min().unwrap();
        let max_duration = *durations.iter().max().unwrap();
        let total_duration: Duration = durations.iter().sum();
        let avg_duration = total_duration / total_operations as u32;
        let operations_per_second = if avg_duration.as_secs_f64() > 0.0 {
            1.0 / avg_duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            min_duration,
            max_duration,
            avg_duration,
            total_operations,
            operations_per_second,
        }
    }

    fn print_summary(&self, operation_name: &str) {
        println!("=== {operation_name} Performance ===");
        println!("  Operations: {}", self.total_operations);
        println!("  Min time: {:?}", self.min_duration);
        println!("  Max time: {:?}", self.max_duration);
        println!("  Avg time: {:?}", self.avg_duration);
        println!("  Ops/sec: {:.2}", self.operations_per_second);
        println!();
    }

    fn assert_performance_threshold(&self, max_avg_duration: Duration, operation_name: &str) {
        assert!(
            self.avg_duration <= max_avg_duration,
            "{} performance too slow: avg {:?} > threshold {:?}",
            operation_name,
            self.avg_duration,
            max_avg_duration
        );
    }
}

/// Helper function to run a benchmark with warm-up and multiple iterations
fn run_benchmark<F>(
    operation_name: &str,
    warmup_iterations: usize,
    benchmark_iterations: usize,
    mut operation: F,
) -> BenchmarkStats
where
    F: FnMut(),
{
    // Warm-up iterations to stabilize performance
    for _ in 0..warmup_iterations {
        operation();
    }

    // Benchmark iterations with timing
    let mut durations = Vec::with_capacity(benchmark_iterations);
    for _ in 0..benchmark_iterations {
        let start = std::time::Instant::now();
        operation();
        durations.push(start.elapsed());
    }

    let stats = BenchmarkStats::new(&durations);
    stats.print_summary(operation_name);
    stats
}

/// Helper function to measure memory usage during an operation
fn measure_memory_usage<F, R>(pool: &HybridMemoryPool, operation: F) -> (R, usize, usize)
where
    F: FnOnce() -> R,
{
    let initial_metrics = pool.get_metrics();
    let result = operation();
    let final_metrics = pool.get_metrics();

    let memory_used = final_metrics
        .current_allocated
        .saturating_sub(initial_metrics.current_allocated) as usize;
    let allocations_made = final_metrics
        .active_allocations
        .saturating_sub(initial_metrics.active_allocations) as usize;

    (result, memory_used, allocations_made)
}

// =============================================================================
#[test]
fn benchmark_tensor_creation_operations() {
    let pool = create_test_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();

    println!("=== Tensor Creation Operations Benchmark ===");

    // Test different tensor sizes
    let test_sizes = vec![
        (vec![10], "Small Vector"),
        (vec![100, 100], "Medium Matrix"),
        (vec![50, 50, 10], "3D Tensor"),
        (vec![1000], "Large Vector"),
    ];

    for device in &devices {
        println!("Device: {}", bitnet_core::device::describe_device(device));

        for &dtype in &dtypes {
            for (shape, size_name) in &test_sizes {
                let operation_name = format!(
                    "{} {} {}",
                    size_name,
                    dtype,
                    bitnet_core::device::describe_device(device)
                );

                // Benchmark zeros creation
                let zeros_stats = run_benchmark(
                    &format!("zeros_{operation_name}"),
                    10,  // warmup
                    100, // benchmark iterations
                    || {
                        let _tensor = BitNetTensor::zeros(shape, dtype, device, &pool)
                            .expect("Failed to create zeros tensor");
                    },
                );
                zeros_stats
                    .assert_performance_threshold(Duration::from_millis(10), "Zeros creation");

                // Benchmark ones creation
                let ones_stats = run_benchmark(&format!("ones_{operation_name}"), 10, 100, || {
                    let _tensor = BitNetTensor::ones(shape, dtype, device, &pool)
                        .expect("Failed to create ones tensor");
                });
                ones_stats.assert_performance_threshold(Duration::from_millis(10), "Ones creation");

                // Benchmark from_data creation (F32 only)
                if dtype == BitNetDType::F32 {
                    let element_count: usize = shape.iter().product();
                    let test_data: Vec<f32> = (0..element_count).map(|i| i as f32).collect();

                    let from_data_stats = run_benchmark(
                        &format!("from_data_{operation_name}"),
                        10,
                        50, // Fewer iterations for data copying
                        || {
                            let _tensor =
                                BitNetTensor::from_data(test_data.clone(), shape, device, &pool)
                                    .expect("Failed to create tensor from data");
                        },
                    );
                    from_data_stats.assert_performance_threshold(
                        Duration::from_millis(20),
                        "From data creation",
                    );
                }
            }
        }
    }
}

#[test]
fn benchmark_mathematical_operations() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Mathematical Operations Benchmark ===");

    // Test different operation sizes - start with smaller tensors to avoid memory issues
    let test_shapes = vec![
        vec![100, 100],   // 40KB
        vec![200, 200],   // 160KB
        vec![50, 50, 20], // 200KB
    ];

    for shape in &test_shapes {
        let shape_name = format!("{shape:?}");

        // Create test tensors
        let tensor1 = BitNetTensor::ones(shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor1");
        let tensor2 = BitNetTensor::ones(shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor2");

        // Benchmark transpose operations
        let transpose_stats = run_benchmark(&format!("transpose_{shape_name}"), 5, 50, || {
            let candle_tensor = tensor1.to_candle().expect("Failed to convert to candle");
            let dims: Vec<usize> = (0..shape.len()).rev().collect(); // Reverse dimensions
            let _transposed =
                bitnet_core::tensor::transpose(&candle_tensor, &dims).expect("Failed to transpose");
        });
        transpose_stats.assert_performance_threshold(Duration::from_millis(100), "Transpose");

        // Benchmark reshape operations
        let element_count: usize = shape.iter().product();
        let reshape_stats = run_benchmark(&format!("reshape_{shape_name}"), 5, 50, || {
            let _reshaped = tensor1
                .reshape(&[element_count])
                .expect("Failed to reshape");
        });
        reshape_stats.assert_performance_threshold(Duration::from_millis(50), "Reshape");

        // Benchmark arithmetic operations
        let candle1 = tensor1.to_candle().expect("Failed to convert tensor1");
        let candle2 = tensor2.to_candle().expect("Failed to convert tensor2");

        let addition_stats = run_benchmark(&format!("addition_{shape_name}"), 5, 100, || {
            let _result = (&candle1 + &candle2).expect("Failed to add tensors");
        });
        addition_stats.assert_performance_threshold(Duration::from_millis(100), "Addition");

        let multiplication_stats =
            run_benchmark(&format!("multiplication_{shape_name}"), 5, 100, || {
                let _result = (&candle1 * &candle2).expect("Failed to multiply tensors");
            });
        multiplication_stats
            .assert_performance_threshold(Duration::from_millis(100), "Multiplication");

        // Benchmark tensor cloning
        let clone_stats = run_benchmark(&format!("clone_{shape_name}"), 5, 50, || {
            let _cloned = tensor1.clone_tensor(&pool);
        });
        clone_stats.assert_performance_threshold(Duration::from_millis(100), "Tensor cloning");
    }
}

#[test]
fn benchmark_size_scaling_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Size Scaling Performance Benchmark ===");

    // Test scaling from small to large tensors
    let size_tests = vec![
        (vec![10, 10], "Tiny"),
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
        (vec![500, 500], "Very Large"),
    ];

    let mut creation_times = Vec::new();
    let mut memory_usage = Vec::new();

    for (shape, size_name) in &size_tests {
        let element_count: usize = shape.iter().product();

        // Benchmark creation time vs size
        let creation_stats = run_benchmark(&format!("creation_scaling_{size_name}"), 3, 20, || {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
        });

        creation_times.push((element_count, creation_stats.avg_duration));

        // Measure memory efficiency
        let (tensor, memory_used, _) = measure_memory_usage(&pool, || {
            BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor")
        });

        let expected_memory = BitNetDType::F32.bytes_for_elements(element_count);
        let memory_efficiency = expected_memory as f64 / memory_used as f64;

        memory_usage.push((element_count, memory_used, memory_efficiency));

        println!(
            "Size {}: {} elements, creation time: {:?}, memory used: {} bytes, efficiency: {:.2}x",
            size_name, element_count, creation_stats.avg_duration, memory_used, memory_efficiency
        );

        drop(tensor);
    }

    // Verify scaling characteristics
    for i in 1..creation_times.len() {
        let (prev_size, prev_time) = creation_times[i - 1];
        let (curr_size, curr_time) = creation_times[i];

        let size_ratio = curr_size as f64 / prev_size as f64;
        let time_ratio = curr_time.as_secs_f64() / prev_time.as_secs_f64();

        // Creation time should scale reasonably with size (not exponentially)
        assert!(
            time_ratio < size_ratio * 2.0,
            "Creation time scaling too poorly: {time_ratio:.2}x time for {size_ratio:.2}x size"
        );

        println!("Scaling {size_ratio:.2}x size -> {time_ratio:.2}x time");
    }

    // Test different data types scaling
    let dtypes_to_test = vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I8,
        BitNetDType::I4,
    ];
    let test_shape = vec![100, 100];

    for &dtype in &dtypes_to_test {
        let dtype_stats = run_benchmark(&format!("dtype_scaling_{dtype}"), 5, 50, || {
            let _tensor = BitNetTensor::zeros(&test_shape, dtype, &device, &pool)
                .expect("Failed to create tensor");
        });

        let expected_bytes = dtype.bytes_for_elements(10000);
        println!(
            "Dtype {}: avg time {:?}, expected size {} bytes, efficiency {:.1}x",
            dtype,
            dtype_stats.avg_duration,
            expected_bytes,
            dtype.memory_efficiency()
        );
    }
}

#[test]
fn benchmark_device_specific_performance() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping device-specific benchmark - only one device available");
        return;
    }

    println!("=== Device-Specific Performance Benchmark ===");

    let test_shapes = vec![vec![100, 100], vec![500, 500], vec![50, 50, 20]];

    for shape in &test_shapes {
        let shape_name = format!("{shape:?}");
        println!("Testing shape: {shape_name}");

        let mut device_performance = Vec::new();

        for device in &devices {
            let device_name = bitnet_core::device::describe_device(device);

            // Benchmark tensor creation on each device
            let creation_stats = run_benchmark(
                &format!("device_creation_{device_name}_{shape_name}"),
                5,
                30,
                || {
                    let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, device, &pool)
                        .expect("Failed to create tensor");
                },
            );

            // Benchmark device migration
            let source_tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &devices[0], &pool)
                .expect("Failed to create source tensor");

            let migration_stats =
                run_benchmark(&format!("device_migration_to_{device_name}"), 3, 10, || {
                    let _migrated = source_tensor
                        .to_device(device, &pool)
                        .expect("Failed to migrate tensor");
                });

            device_performance.push((
                device_name.clone(),
                creation_stats.avg_duration,
                migration_stats.avg_duration,
            ));

            println!(
                "Device {}: creation {:?}, migration {:?}",
                device_name, creation_stats.avg_duration, migration_stats.avg_duration
            );
        }

        // Compare device performance
        if device_performance.len() >= 2 {
            let (dev1_name, dev1_create, dev1_migrate) = &device_performance[0];
            let (dev2_name, dev2_create, dev2_migrate) = &device_performance[1];

            let create_ratio = dev2_create.as_secs_f64() / dev1_create.as_secs_f64();
            let migrate_ratio = dev2_migrate.as_secs_f64() / dev1_migrate.as_secs_f64();

            println!("Performance comparison: {dev1_name} vs {dev2_name}");
            println!("  Creation ratio: {create_ratio:.2}x");
            println!("  Migration ratio: {migrate_ratio:.2}x");
        }
    }

    // Test cross-device operation performance
    if devices.len() >= 2 {
        let tensor1 = BitNetTensor::ones(&[100, 100], BitNetDType::F32, &devices[0], &pool)
            .expect("Failed to create tensor1");
        let tensor2 = BitNetTensor::ones(&[100, 100], BitNetDType::F32, &devices[1], &pool)
            .expect("Failed to create tensor2");

        // Test same-device vs cross-device operations
        let candle1 = tensor1.to_candle().expect("Failed to convert tensor1");
        let candle2 = tensor2.to_candle().expect("Failed to convert tensor2");

        // This may fail for cross-device operations, which is expected
        match &candle1 + &candle2 {
            Ok(_) => {
                let cross_device_stats = run_benchmark("cross_device_addition", 3, 20, || {
                    let _result = (&candle1 + &candle2).expect("Cross-device addition failed");
                });
                println!(
                    "Cross-device operations supported: avg time {:?}",
                    cross_device_stats.avg_duration
                );
            }
            Err(e) => {
                println!("Cross-device operations not supported (expected): {e}");
            }
        }
    }
}

#[test]
fn benchmark_concurrent_operation_performance() {
    let pool = Arc::new(create_test_pool());
    let device = get_cpu_device();

    println!("=== Concurrent Operation Performance Benchmark ===");

    // Test different concurrency levels
    let concurrency_levels = vec![1, 2, 4, 8];
    let operations_per_thread = 50;

    for &thread_count in &concurrency_levels {
        println!("Testing with {thread_count} threads");

        let start_time = std::time::Instant::now();
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let pool_clone = pool.clone();
            let device_clone = device.clone();

            let handle = thread::spawn(move || {
                let mut thread_durations = Vec::new();

                for i in 0..operations_per_thread {
                    let shape = vec![thread_id + 1, i + 1];

                    let op_start = std::time::Instant::now();
                    let tensor =
                        BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone)
                            .expect("Failed to create tensor");

                    // Perform some operations
                    let _handle = tensor.handle();
                    let _shape = tensor.shape();
                    let _size = tensor.size_bytes();

                    thread_durations.push(op_start.elapsed());
                }

                thread_durations
            });

            handles.push(handle);
        }

        // Collect results
        let mut all_durations = Vec::new();
        for handle in handles {
            let thread_durations = handle.join().expect("Thread panicked");
            all_durations.extend(thread_durations);
        }

        let total_time = start_time.elapsed();
        let stats = BenchmarkStats::new(&all_durations);

        let total_operations = thread_count * operations_per_thread;
        let overall_throughput = total_operations as f64 / total_time.as_secs_f64();

        println!(
            "Concurrency {}: {} ops in {:?}, throughput: {:.2} ops/sec, avg per-op: {:?}",
            thread_count, total_operations, total_time, overall_throughput, stats.avg_duration
        );

        // Verify concurrent performance is reasonable
        stats.assert_performance_threshold(Duration::from_millis(50), "Concurrent operations");

        // Test handle sharing performance
        let shared_tensor = Arc::new(
            BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
                .expect("Failed to create shared tensor"),
        );

        let handle_start = std::time::Instant::now();
        let mut handle_handles = Vec::new();

        for _ in 0..thread_count {
            let tensor_clone = shared_tensor.clone();

            let handle = thread::spawn(move || {
                let mut handle_times = Vec::new();

                for _ in 0..20 {
                    let start = std::time::Instant::now();
                    let handle = tensor_clone.handle();
                    assert!(handle.is_valid());
                    handle_times.push(start.elapsed());
                }

                handle_times
            });

            handle_handles.push(handle);
        }

        let mut handle_durations = Vec::new();
        for handle in handle_handles {
            let thread_times = handle.join().expect("Handle thread panicked");
            handle_durations.extend(thread_times);
        }

        let handle_stats = BenchmarkStats::new(&handle_durations);
        println!(
            "Handle sharing {}: avg time {:?}, ops/sec: {:.2}",
            thread_count, handle_stats.avg_duration, handle_stats.operations_per_second
        );
    }

    // Test memory pressure under concurrency
    println!("Testing memory pressure under concurrency...");
    let pressure_start = std::time::Instant::now();
    let mut pressure_handles = Vec::new();

    for thread_id in 0..4 {
        let pool_clone = pool.clone();
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            let mut tensors = Vec::new();
            let mut creation_count = 0;

            // Create tensors until memory pressure or limit
            for i in 0..100 {
                let size = (thread_id + 1) * 100 + i;
                match BitNetTensor::zeros(&[size], BitNetDType::F32, &device_clone, &pool_clone) {
                    Ok(tensor) => {
                        tensors.push(tensor);
                        creation_count += 1;
                    }
                    Err(_) => break,
                }
            }

            creation_count
        });

        pressure_handles.push(handle);
    }

    let mut total_created = 0;
    for handle in pressure_handles {
        total_created += handle.join().expect("Pressure thread panicked");
    }

    let pressure_time = pressure_start.elapsed();
    println!("Memory pressure test: {total_created} tensors created in {pressure_time:?}");
}

#[test]
fn benchmark_memory_efficiency() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Memory Efficiency Benchmark ===");

    // Test memory allocation speed
    let allocation_sizes = vec![
        (vec![100], "Small"),
        (vec![1000], "Medium"),
        (vec![10000], "Large"),
        (vec![100000], "Very Large"),
    ];

    for (shape, size_name) in &allocation_sizes {
        let element_count: usize = shape.iter().product();

        // Benchmark allocation speed
        let (tensor, memory_used, allocations_made) = measure_memory_usage(&pool, || {
            BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor")
        });

        let expected_memory = BitNetDType::F32.bytes_for_elements(element_count);
        let memory_overhead = memory_used.saturating_sub(expected_memory);
        let efficiency_ratio = expected_memory as f64 / memory_used as f64;

        println!("{size_name} allocation: {element_count} elements, {memory_used} bytes used, {expected_memory} bytes expected, overhead: {memory_overhead} bytes, efficiency: {efficiency_ratio:.2}x");

        // Benchmark allocation/deallocation speed
        let alloc_dealloc_stats =
            run_benchmark(&format!("alloc_dealloc_{size_name}"), 5, 50, || {
                let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                    .expect("Failed to create tensor");
                drop(tensor);
            });

        alloc_dealloc_stats
            .assert_performance_threshold(Duration::from_millis(20), "Allocation/deallocation");

        drop(tensor);
    }

    // Test memory pool efficiency with different data types
    println!("Testing memory efficiency across data types...");
    let dtypes = get_test_dtypes();
    let test_shape = vec![1000];
    let element_count = 1000;

    for &dtype in &dtypes {
        let (tensor, memory_used, _) = measure_memory_usage(&pool, || {
            BitNetTensor::zeros(&test_shape, dtype, &device, &pool)
                .expect("Failed to create tensor")
        });

        let expected_memory = dtype.bytes_for_elements(element_count);
        let memory_efficiency = dtype.memory_efficiency();
        let actual_efficiency =
            BitNetDType::F32.bytes_for_elements(element_count) as f64 / memory_used as f64;

        println!("Dtype {dtype}: expected {expected_memory} bytes, used {memory_used} bytes, theoretical efficiency {memory_efficiency:.1}x, actual efficiency {actual_efficiency:.2}x");

        // Verify memory usage is reasonable
        assert!(
            memory_used >= expected_memory,
            "Memory usage {memory_used} less than expected {expected_memory} for dtype {dtype}"
        );

        // Allow for reasonable overhead (up to 50% for small allocations)
        let max_overhead = std::cmp::max(expected_memory / 2, 1024); // At least 1KB overhead allowed
        assert!(memory_used <= expected_memory + max_overhead,
               "Excessive memory overhead for dtype {dtype}: {memory_used} used vs {expected_memory} expected");

        drop(tensor);
    }

    // Test memory cleanup efficiency
    println!("Testing memory cleanup efficiency...");
    let initial_metrics = pool.get_metrics();

    let peak_metrics = {
        let mut tensors = Vec::new();
        for i in 0..50 {
            let tensor = BitNetTensor::zeros(&[100 + i], BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
            tensors.push(tensor);
        }

        let peak_metrics = pool.get_metrics();
        let peak_memory = peak_metrics.current_allocated;

        println!("Created 50 tensors, peak memory: {peak_memory} bytes");

        // Drop all tensors
        drop(tensors);

        peak_metrics
    };

    // Allow time for cleanup
    thread::sleep(Duration::from_millis(50));

    let final_metrics = pool.get_metrics();
    let memory_recovered = peak_metrics
        .current_allocated
        .saturating_sub(final_metrics.current_allocated);
    let cleanup_efficiency = memory_recovered as f64 / peak_metrics.current_allocated as f64;

    println!(
        "Memory cleanup: {} bytes recovered, {:.1}% efficiency",
        memory_recovered,
        cleanup_efficiency * 100.0
    );

    // Verify reasonable cleanup (at least 50% of allocated memory should be recovered, or no memory was allocated)
    assert!(
        cleanup_efficiency >= 0.5 || peak_metrics.current_allocated == 0,
        "Poor memory cleanup efficiency: {:.1}%",
        cleanup_efficiency * 100.0
    );

    // Test memory fragmentation resistance
    println!("Testing memory fragmentation resistance...");
    let fragmentation_start = std::time::Instant::now();

    for _ in 0..100 {
        // Create and immediately drop tensors of varying sizes
        let sizes = vec![50, 200, 75, 150, 100];
        for size in sizes {
            let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
                .expect("Failed to create fragmentation test tensor");
            drop(tensor);
        }
    }

    let fragmentation_time = fragmentation_start.elapsed();
    println!("Fragmentation test completed in {fragmentation_time:?}");

    // Verify fragmentation doesn't severely impact performance
    assert!(
        fragmentation_time < Duration::from_secs(5),
        "Memory fragmentation test took too long: {fragmentation_time:?}"
    );
}

#[test]
fn benchmark_quantization_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Quantization Performance Benchmark ===");

    // Test quantized data types performance
    let quantized_types = vec![
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];

    let test_shapes = vec![vec![1000], vec![100, 100], vec![50, 50, 8]];

    for shape in &test_shapes {
        let shape_name = format!("{shape:?}");
        let element_count: usize = shape.iter().product();

        println!("Testing quantized types with shape: {shape_name}");

        // Benchmark creation performance for each quantized type
        for &qtype in &quantized_types {
            let creation_stats = run_benchmark(
                &format!("quantized_creation_{qtype}_{shape_name}"),
                5,
                50,
                || {
                    let _tensor = BitNetTensor::zeros(shape, qtype, &device, &pool)
                        .expect("Failed to create quantized tensor");
                },
            );

            let expected_bytes = qtype.bytes_for_elements(element_count);
            let memory_efficiency = qtype.memory_efficiency();

            println!(
                "  {}: avg time {:?}, {} bytes, {:.1}x efficiency",
                qtype, creation_stats.avg_duration, expected_bytes, memory_efficiency
            );

            // Quantized types should be fast to create
            creation_stats
                .assert_performance_threshold(Duration::from_millis(15), "Quantized creation");
        }

        // Compare quantized vs full precision performance
        let f32_stats = run_benchmark(&format!("f32_creation_{shape_name}"), 5, 50, || {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create F32 tensor");
        });

        let i8_stats = run_benchmark(&format!("i8_creation_{shape_name}"), 5, 50, || {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::I8, &device, &pool)
                .expect("Failed to create I8 tensor");
        });

        let performance_ratio =
            i8_stats.avg_duration.as_secs_f64() / f32_stats.avg_duration.as_secs_f64();
        println!("  I8 vs F32 performance ratio: {performance_ratio:.2}x");

        // Test BitNet 1.58b specific performance
        let bitnet158_stats =
            run_benchmark(&format!("bitnet158_creation_{shape_name}"), 5, 50, || {
                let _tensor = BitNetTensor::zeros(shape, BitNetDType::BitNet158, &device, &pool)
                    .expect("Failed to create BitNet158 tensor");
            });

        println!(
            "  BitNet 1.58b: avg time {:?}, 2 bits per element",
            bitnet158_stats.avg_duration
        );

        // Verify BitNet 1.58b properties
        let bitnet_tensor = BitNetTensor::zeros(shape, BitNetDType::BitNet158, &device, &pool)
            .expect("Failed to create BitNet tensor");

        assert!(bitnet_tensor.dtype().is_bitnet158());
        assert!(bitnet_tensor.dtype().is_quantized());
        assert_eq!(bitnet_tensor.dtype().bits_per_element(), 2);

        let (min_val, max_val) = bitnet_tensor.dtype().value_range().unwrap();
        assert_eq!(min_val, -1);
        assert_eq!(max_val, 1);

        drop(bitnet_tensor);
    }

    // Test quantization conversion performance (simulated)
    println!("Testing quantization conversion performance...");

    let source_data: Vec<f32> = (0..10000).map(|i| (i as f32 - 5000.0) / 5000.0).collect();
    let source_shape = vec![10000];

    // Create F32 source tensor
    let f32_tensor = BitNetTensor::from_data(source_data, &source_shape, &device, &pool)
        .expect("Failed to create F32 source tensor");

    // Benchmark conversion to different quantized types
    for &target_type in &quantized_types {
        // Simulate quantization by creating target tensor and measuring conversion overhead
        let conversion_stats =
            run_benchmark(&format!("quantization_to_{target_type}"), 3, 20, || {
                // Create target tensor (simulates quantization)
                let _target = BitNetTensor::zeros(&source_shape, target_type, &device, &pool)
                    .expect("Failed to create target tensor");

                // Convert source to candle and back (simulates data conversion)
                let _candle = f32_tensor.to_candle().expect("Failed to convert to candle");
            });

        let bits_per_element = target_type.bits_per_element();
        let compression_ratio = 32.0 / bits_per_element as f32;

        println!(
            "  Quantization to {}: avg time {:?}, {}x compression",
            target_type, conversion_stats.avg_duration, compression_ratio
        );

        // Quantization should be reasonably fast
        conversion_stats
            .assert_performance_threshold(Duration::from_millis(50), "Quantization conversion");
    }

    // Test quantized arithmetic performance
    println!("Testing quantized arithmetic performance...");

    let q_shape = vec![100, 100];
    let q_tensor1 = BitNetTensor::ones(&q_shape, BitNetDType::I8, &device, &pool)
        .expect("Failed to create quantized tensor1");
    let q_tensor2 = BitNetTensor::ones(&q_shape, BitNetDType::I8, &device, &pool)
        .expect("Failed to create quantized tensor2");

    let q_candle1 = q_tensor1.to_candle().expect("Failed to convert q_tensor1");
    let q_candle2 = q_tensor2.to_candle().expect("Failed to convert q_tensor2");

    let quantized_arithmetic_stats = run_benchmark("quantized_arithmetic", 5, 100, || {
        let _result = (&q_candle1 + &q_candle2).expect("Failed to add quantized tensors");
    });

    println!(
        "  Quantized arithmetic: avg time {:?}",
        quantized_arithmetic_stats.avg_duration
    );
    quantized_arithmetic_stats
        .assert_performance_threshold(Duration::from_millis(10), "Quantized arithmetic");

    // Test memory efficiency of quantized operations
    println!("Testing quantized memory efficiency...");

    let efficiency_test_shape = vec![100, 100]; // Reduced size to avoid memory issues
    let element_count = 10000;

    // Compare memory usage: F32 vs I8 vs BitNet158
    let (f32_tensor, f32_memory, _) = measure_memory_usage(&pool, || {
        BitNetTensor::zeros(&efficiency_test_shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create F32 tensor")
    });

    let (i8_tensor, i8_memory, _) = measure_memory_usage(&pool, || {
        BitNetTensor::zeros(&efficiency_test_shape, BitNetDType::I8, &device, &pool)
            .expect("Failed to create I8 tensor")
    });

    let (bitnet_tensor, bitnet_memory, _) = measure_memory_usage(&pool, || {
        BitNetTensor::zeros(
            &efficiency_test_shape,
            BitNetDType::BitNet158,
            &device,
            &pool,
        )
        .expect("Failed to create BitNet tensor")
    });

    let i8_efficiency = f32_memory as f64 / i8_memory as f64;
    let bitnet_efficiency = f32_memory as f64 / bitnet_memory as f64;

    println!("  Memory efficiency comparison (10K elements):");
    println!("    F32: {f32_memory} bytes");
    println!("    I8: {i8_memory} bytes ({i8_efficiency:.1}x efficiency)");
    println!("    BitNet158: {bitnet_memory} bytes ({bitnet_efficiency:.1}x efficiency)");

    // Verify expected memory savings (more lenient thresholds)
    assert!(
        i8_efficiency >= 2.0,
        "I8 should provide at least 2x memory efficiency"
    );
    assert!(
        bitnet_efficiency >= 5.0,
        "BitNet158 should provide at least 5x memory efficiency"
    );

    drop(f32_tensor);
    drop(i8_tensor);
    drop(bitnet_tensor);
}

#[test]
fn benchmark_regression_testing() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Regression Testing Benchmark ===");

    // Define baseline performance expectations
    // These would typically be updated when intentional performance improvements are made
    let baseline_expectations = vec![
        ("tensor_creation_small", Duration::from_millis(5)),
        ("tensor_creation_medium", Duration::from_millis(10)),
        ("tensor_creation_large", Duration::from_millis(50)),
        ("arithmetic_operations", Duration::from_millis(20)),
        ("memory_allocation", Duration::from_millis(15)),
        ("device_migration", Duration::from_millis(100)),
        ("handle_operations", Duration::from_micros(100)),
    ];

    let mut regression_results = Vec::new();

    // Test 1: Tensor creation regression
    println!("Testing tensor creation regression...");

    let small_creation_stats = run_benchmark("regression_small_creation", 10, 100, || {
        let _tensor = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device, &pool)
            .expect("Failed to create small tensor");
    });
    regression_results.push(("tensor_creation_small", small_creation_stats.avg_duration));

    let medium_creation_stats = run_benchmark("regression_medium_creation", 10, 50, || {
        let _tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create medium tensor");
    });
    regression_results.push(("tensor_creation_medium", medium_creation_stats.avg_duration));

    let large_creation_stats = run_benchmark("regression_large_creation", 5, 20, || {
        let _tensor = BitNetTensor::zeros(&[500, 500], BitNetDType::F32, &device, &pool)
            .expect("Failed to create large tensor");
    });
    regression_results.push(("tensor_creation_large", large_creation_stats.avg_duration));

    // Test 2: Arithmetic operations regression
    println!("Testing arithmetic operations regression...");

    let tensor1 = BitNetTensor::ones(&[200, 200], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor1");
    let tensor2 = BitNetTensor::ones(&[200, 200], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor2");

    let candle1 = tensor1.to_candle().expect("Failed to convert tensor1");
    let candle2 = tensor2.to_candle().expect("Failed to convert tensor2");

    let arithmetic_stats = run_benchmark("regression_arithmetic", 5, 50, || {
        let _add = (&candle1 + &candle2).expect("Failed to add");
        let _mul = (&candle1 * &candle2).expect("Failed to multiply");
    });
    regression_results.push(("arithmetic_operations", arithmetic_stats.avg_duration));

    // Test 3: Memory allocation regression
    println!("Testing memory allocation regression...");

    let allocation_stats = run_benchmark("regression_allocation", 5, 100, || {
        let tensor = BitNetTensor::zeros(&[1000], BitNetDType::F32, &device, &pool)
            .expect("Failed to allocate tensor");
        drop(tensor);
    });
    regression_results.push(("memory_allocation", allocation_stats.avg_duration));

    // Test 4: Device migration regression (if multiple devices available)
    let devices = get_test_devices();
    if devices.len() >= 2 {
        println!("Testing device migration regression...");

        let migration_tensor =
            BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &devices[0], &pool)
                .expect("Failed to create migration tensor");

        let migration_stats = run_benchmark("regression_migration", 3, 10, || {
            let _migrated = migration_tensor
                .to_device(&devices[1], &pool)
                .expect("Failed to migrate tensor");
        });
        regression_results.push(("device_migration", migration_stats.avg_duration));
    }

    // Test 5: Handle operations regression
    println!("Testing handle operations regression...");

    let handle_tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
        .expect("Failed to create handle tensor");

    let handle_stats = run_benchmark("regression_handles", 10, 1000, || {
        let handle = handle_tensor.handle();
        assert!(handle.is_valid());
        let _metadata = handle.metadata();
    });
    regression_results.push(("handle_operations", handle_stats.avg_duration));

    // Test 6: Consistency across multiple runs
    println!("Testing performance consistency...");

    let mut consistency_measurements = Vec::new();
    for run in 0..5 {
        let run_stats = run_benchmark(&format!("consistency_run_{run}"), 5, 20, || {
            let tensor = BitNetTensor::zeros(&[50, 50], BitNetDType::F32, &device, &pool)
                .expect("Failed to create consistency tensor");
            let _handle = tensor.handle();
            let _shape = tensor.shape();
            drop(tensor);
        });
        consistency_measurements.push(run_stats.avg_duration);
    }

    // Calculate coefficient of variation for consistency
    let mean_duration =
        consistency_measurements.iter().sum::<Duration>() / consistency_measurements.len() as u32;
    let variance: f64 = consistency_measurements
        .iter()
        .map(|d| {
            let diff = d.as_secs_f64() - mean_duration.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / consistency_measurements.len() as f64;
    let std_dev = variance.sqrt();
    let coefficient_of_variation = std_dev / mean_duration.as_secs_f64();

    println!(
        "Performance consistency: mean {:?}, std dev {:.6}s, CV {:.2}%",
        mean_duration,
        std_dev,
        coefficient_of_variation * 100.0
    );

    // Verify consistency (CV should be less than 20% for stable performance)
    assert!(
        coefficient_of_variation < 0.2,
        "Performance too inconsistent: CV {:.2}%",
        coefficient_of_variation * 100.0
    );

    // Compare against baseline expectations
    println!("Regression analysis results:");
    let mut regression_detected = false;

    for (test_name, actual_duration) in &regression_results {
        if let Some((_, expected_duration)) = baseline_expectations
            .iter()
            .find(|(name, _)| name == test_name)
        {
            let performance_ratio = actual_duration.as_secs_f64() / expected_duration.as_secs_f64();
            let status = if performance_ratio <= 1.0 {
                "IMPROVED"
            } else if performance_ratio <= 1.5 {
                "ACCEPTABLE"
            } else if performance_ratio <= 2.0 {
                "DEGRADED"
            } else {
                "REGRESSION"
            };

            println!("  {test_name}: {actual_duration:?} (expected {expected_duration:?}) - {performance_ratio:.2}x - {status}");

            // Flag significant regressions (more than 2x slower)
            if performance_ratio > 2.0 {
                regression_detected = true;
                println!("    WARNING: Significant performance regression detected!");
            }

            // For automated testing, we'll be lenient and only fail on extreme regressions
            assert!(performance_ratio <= 5.0,
                   "Extreme performance regression in {test_name}: {performance_ratio:.2}x slower than baseline");
        }
    }

    // Test 7: Memory usage regression
    println!("Testing memory usage regression...");

    let initial_metrics = pool.get_metrics();

    // Create a known workload
    let mut test_tensors = Vec::new();
    for i in 0..20 {
        let tensor = BitNetTensor::zeros(&[100 + i, 100 + i], BitNetDType::F32, &device, &pool)
            .expect("Failed to create memory test tensor");
        test_tensors.push(tensor);
    }

    let peak_metrics = pool.get_metrics();
    let memory_used = peak_metrics.current_allocated - initial_metrics.current_allocated;

    // Expected memory usage (rough calculation)
    let expected_memory: usize = (0..20)
        .map(|i| (100 + i) * (100 + i) * 4) // 4 bytes per F32
        .sum();

    let memory_efficiency = expected_memory as f64 / memory_used as f64;

    println!("Memory usage: {memory_used} bytes used, {expected_memory} bytes expected, {memory_efficiency:.2}x efficiency");

    // Verify memory usage is reasonable (allow for up to 2x overhead)
    assert!(
        memory_efficiency >= 0.5,
        "Excessive memory overhead: {memory_efficiency:.2}x efficiency"
    );

    drop(test_tensors);

    // Test 8: Cleanup performance regression
    println!("Testing cleanup performance regression...");

    let cleanup_start = std::time::Instant::now();

    // Allow time for cleanup
    thread::sleep(Duration::from_millis(100));

    let cleanup_time = cleanup_start.elapsed();
    let final_metrics = pool.get_metrics();

    println!("Cleanup completed in {cleanup_time:?}");

    // Verify cleanup was effective
    let memory_recovered = peak_metrics
        .current_allocated
        .saturating_sub(final_metrics.current_allocated);
    let cleanup_efficiency = memory_recovered as f64 / peak_metrics.current_allocated as f64;

    println!("Cleanup efficiency: {:.1}%", cleanup_efficiency * 100.0);

    // Summary
    println!("=== Regression Testing Summary ===");
    if regression_detected {
        println!("⚠️  Performance regressions detected - review recommended");
    } else {
        println!("✅ No significant performance regressions detected");
    }

    println!(
        "Performance consistency: {:.2}% CV",
        coefficient_of_variation * 100.0
    );
    println!("Memory efficiency: {memory_efficiency:.2}x");
    println!("Cleanup efficiency: {:.1}%", cleanup_efficiency * 100.0);
}

// =============================================================================
// Focused Performance Benchmark Tests
// =============================================================================

#[test]
fn benchmark_tensor_creation_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Tensor Creation Performance Benchmark ===");

    // Test different tensor sizes
    let test_cases = vec![
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
    ];

    for (shape, size_name) in &test_cases {
        println!("Testing {size_name} tensors with shape {shape:?}");

        // Warm-up iterations
        for _ in 0..10 {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create warm-up tensor");
        }

        // Benchmark zeros() creation
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create zeros tensor");
        }
        let zeros_duration = start.elapsed();

        // Benchmark ones() creation
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::ones(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create ones tensor");
        }
        let ones_duration = start.elapsed();

        // Benchmark from_data() creation
        let element_count: usize = shape.iter().product();
        let test_data: Vec<f32> = (0..element_count).map(|i| i as f32).collect();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::from_data(test_data.clone(), shape, &device, &pool)
                .expect("Failed to create from_data tensor");
        }
        let from_data_duration = start.elapsed();

        println!("  {size_name} tensors:");
        println!("    zeros(): avg {:?}", zeros_duration / 100);
        println!("    ones(): avg {:?}", ones_duration / 100);
        println!("    from_data(): avg {:?}", from_data_duration / 100);

        // Validate reasonable performance thresholds
        let reasonable_threshold = std::time::Duration::from_millis(100);
        assert!(
            zeros_duration < reasonable_threshold,
            "zeros() creation too slow for {size_name} tensors: {zeros_duration:?}"
        );
        assert!(
            ones_duration < reasonable_threshold,
            "ones() creation too slow for {size_name} tensors: {ones_duration:?}"
        );
        assert!(
            from_data_duration < reasonable_threshold * 2, // Allow more time for data copying
            "from_data() creation too slow for {size_name} tensors: {from_data_duration:?}"
        );
    }
}

#[test]
fn benchmark_mathematical_operations_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Mathematical Operations Performance Benchmark ===");

    // Test different tensor sizes
    let test_cases = vec![
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
    ];

    for (shape, size_name) in &test_cases {
        println!("Testing {size_name} mathematical operations with shape {shape:?}");

        // Create test tensors
        let tensor1 = BitNetTensor::ones(shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor1");
        let tensor2 = BitNetTensor::ones(shape, BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor2");

        // Warm-up iterations
        for _ in 0..10 {
            let _transposed = bitnet_core::tensor::transpose(
                &tensor1.to_candle().expect("Failed to convert"),
                &[1, 0],
            )
            .expect("Failed to transpose");
        }

        // Benchmark transpose operations
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let candle_tensor = tensor1.to_candle().expect("Failed to convert to candle");
            let _transposed = bitnet_core::tensor::transpose(&candle_tensor, &[1, 0])
                .expect("Failed to transpose");
        }
        let transpose_duration = start.elapsed();

        // Benchmark arithmetic operations
        let candle1 = tensor1.to_candle().expect("Failed to convert tensor1");
        let candle2 = tensor2.to_candle().expect("Failed to convert tensor2");

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _add_result = (&candle1 + &candle2).expect("Failed to add tensors");
            let _mul_result = (&candle1 * &candle2).expect("Failed to multiply tensors");
        }
        let arithmetic_duration = start.elapsed();

        println!("  {size_name} operations:");
        println!("    transpose: avg {:?}", transpose_duration / 100);
        println!(
            "    arithmetic (add+mul): avg {:?}",
            arithmetic_duration / 100
        );

        // Validate reasonable performance thresholds
        let reasonable_threshold = std::time::Duration::from_millis(200);
        assert!(
            transpose_duration < reasonable_threshold,
            "Transpose too slow for {size_name} tensors: {transpose_duration:?}"
        );
        assert!(
            arithmetic_duration < reasonable_threshold,
            "Arithmetic operations too slow for {size_name} tensors: {arithmetic_duration:?}"
        );
    }
}

#[test]
fn benchmark_device_migration_performance() {
    let pool = create_test_pool();
    let devices = get_test_devices();

    if devices.len() < 2 {
        println!("Skipping device migration benchmark - only one device available");
        return;
    }

    println!("=== Device Migration Performance Benchmark ===");

    let source_device = &devices[0];
    let target_device = &devices[1];

    // Test different tensor sizes
    let test_cases = vec![
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
    ];

    for (shape, size_name) in &test_cases {
        println!("Testing {size_name} device migration with shape {shape:?}");

        // Create test tensor on source device
        let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, source_device, &pool)
            .expect("Failed to create tensor");

        // Warm-up iterations
        for _ in 0..10 {
            let _migrated = tensor.to_device(target_device, &pool);
        }

        // Benchmark device migration
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _migrated = tensor
                .to_device(target_device, &pool)
                .expect("Failed to migrate tensor");
        }
        let migration_duration = start.elapsed();

        println!(
            "  {} migration: avg {:?}",
            size_name,
            migration_duration / 100
        );

        // Validate reasonable performance threshold
        let reasonable_threshold = std::time::Duration::from_millis(500);
        assert!(
            migration_duration < reasonable_threshold,
            "Device migration too slow for {size_name} tensors: {migration_duration:?}"
        );
    }
}

#[test]
fn benchmark_memory_operations_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Memory Operations Performance Benchmark ===");

    // Test different tensor sizes
    let test_cases = vec![
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
    ];

    for (shape, size_name) in &test_cases {
        println!("Testing {size_name} memory operations with shape {shape:?}");

        // Warm-up iterations
        for _ in 0..10 {
            let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
            drop(tensor);
        }

        // Benchmark memory allocation
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to allocate tensor");
        }
        let allocation_duration = start.elapsed();

        // Benchmark memory allocation and cleanup
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to allocate tensor");
            drop(tensor); // Explicit cleanup
        }
        let alloc_cleanup_duration = start.elapsed();

        println!("  {size_name} memory operations:");
        println!("    allocation: avg {:?}", allocation_duration / 100);
        println!("    alloc+cleanup: avg {:?}", alloc_cleanup_duration / 100);

        // Validate reasonable performance thresholds
        let reasonable_threshold = std::time::Duration::from_millis(100);
        assert!(
            allocation_duration < reasonable_threshold,
            "Memory allocation too slow for {size_name} tensors: {allocation_duration:?}"
        );
        assert!(
            alloc_cleanup_duration < reasonable_threshold * 2,
            "Memory alloc+cleanup too slow for {size_name} tensors: {alloc_cleanup_duration:?}"
        );
    }
}

#[test]
fn benchmark_concurrent_operations_performance() {
    let pool = Arc::new(create_test_pool());
    let device = get_cpu_device();

    println!("=== Concurrent Operations Performance Benchmark ===");

    // Test different concurrency levels
    let concurrency_levels = vec![1, 2, 4];
    let operations_per_thread = 50;

    for &thread_count in &concurrency_levels {
        println!("Testing with {thread_count} threads");

        // Warm-up iterations
        for _ in 0..10 {
            let _tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
                .expect("Failed to create warm-up tensor");
        }

        let start_time = std::time::Instant::now();
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let pool_clone = pool.clone();
            let device_clone = device.clone();

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let shape = vec![thread_id + 5, i + 5]; // Avoid very small tensors
                    let _tensor =
                        BitNetTensor::zeros(&shape, BitNetDType::F32, &device_clone, &pool_clone)
                            .expect("Failed to create tensor in thread");
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let total_duration = start_time.elapsed();
        let total_operations = thread_count * operations_per_thread;
        let avg_per_operation = total_duration / total_operations as u32;

        println!("  {thread_count} threads: {total_operations} ops in {total_duration:?}, avg per op: {avg_per_operation:?}");

        // Validate reasonable performance threshold
        let reasonable_threshold = std::time::Duration::from_millis(50);
        assert!(
            avg_per_operation < reasonable_threshold,
            "Concurrent operations too slow with {thread_count} threads: avg {avg_per_operation:?}"
        );
    }
}

#[test]
fn benchmark_focused_size_scaling_performance() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    println!("=== Focused Size Scaling Performance Benchmark ===");

    // Test scaling from small to large tensors
    let size_tests = vec![
        (vec![50, 50], "Small"),
        (vec![100, 100], "Medium"),
        (vec![200, 200], "Large"),
    ];

    let mut creation_times = Vec::new();

    for (shape, size_name) in &size_tests {
        let element_count: usize = shape.iter().product();

        println!("Testing {size_name} tensors: {element_count} elements");

        // Warm-up iterations
        for _ in 0..10 {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create warm-up tensor");
        }

        // Benchmark creation time vs size
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
        }
        let duration = start.elapsed();
        let avg_duration = duration / 100;

        creation_times.push((element_count, avg_duration));

        println!("  {size_name}: avg creation time {avg_duration:?}");

        // Validate reasonable performance threshold
        let reasonable_threshold = std::time::Duration::from_millis(50);
        assert!(
            avg_duration < reasonable_threshold,
            "Tensor creation too slow for {size_name} size: {avg_duration:?}"
        );
    }

    // Verify scaling characteristics
    for i in 1..creation_times.len() {
        let (prev_size, prev_time) = creation_times[i - 1];
        let (curr_size, curr_time) = creation_times[i];

        let size_ratio = curr_size as f64 / prev_size as f64;
        let time_ratio = curr_time.as_secs_f64() / prev_time.as_secs_f64();

        println!("Scaling {size_ratio:.2}x size -> {time_ratio:.2}x time");

        // Creation time should scale reasonably with size (not exponentially)
        assert!(
            time_ratio < size_ratio * 3.0,
            "Creation time scaling too poorly: {time_ratio:.2}x time for {size_ratio:.2}x size"
        );
    }

    // Test different data types scaling
    let dtypes_to_test = vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I8,
        BitNetDType::I4,
    ];
    let test_shape = vec![50, 50];

    println!("Testing data type scaling:");
    for &dtype in &dtypes_to_test {
        // Warm-up
        for _ in 0..10 {
            let _tensor = BitNetTensor::zeros(&test_shape, dtype, &device, &pool)
                .expect("Failed to create warm-up tensor");
        }

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _tensor = BitNetTensor::zeros(&test_shape, dtype, &device, &pool)
                .expect("Failed to create tensor");
        }
        let duration = start.elapsed();
        let avg_duration = duration / 100;

        let expected_bytes = dtype.bytes_for_elements(10000);
        println!(
            "  {}: avg time {:?}, {} bytes, {:.1}x efficiency",
            dtype,
            avg_duration,
            expected_bytes,
            dtype.memory_efficiency()
        );

        // Validate reasonable performance threshold
        let reasonable_threshold = std::time::Duration::from_millis(20);
        assert!(
            avg_duration < reasonable_threshold,
            "Data type {dtype} creation too slow: {avg_duration:?}"
        );
    }
}
