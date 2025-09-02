//! Core Tensor Tests  
//! Comprehensive test suite for BitNet tensor system core functionality.
//! Following existing test patterns from memory_tracking_tests.rs and tensor_integration_tests.rs

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bitnet_core::device::{devices_equal, get_cpu_device};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
use bitnet_core::tensor::{set_global_memory_pool, BitNetDType, BitNetTensor, BroadcastCompatible};

#[cfg(feature = "apple-silicon")]
use bitnet_core::mlx::MlxDevice;

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Test configuration for tensor core tests
#[derive(Clone)]
struct TensorTestConfig {
    enable_tracking: bool,
    enable_cleanup: bool,
    memory_pressure_threshold: f64,
    max_test_duration: Duration,
}

impl Default for TensorTestConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            enable_cleanup: true,
            memory_pressure_threshold: 0.8,
            max_test_duration: Duration::from_secs(30),
        }
    }
}

/// Helper function to create a test memory pool and set as global
fn setup_global_memory_pool() -> Arc<HybridMemoryPool> {
    let mut config = MemoryPoolConfig::default();
    config.tracking_config = Some(TrackingConfig::detailed());

    let pool =
        Arc::new(HybridMemoryPool::with_config(config).expect("Failed to create test memory pool"));

    // Set as global pool
    set_global_memory_pool(Arc::downgrade(&pool));

    pool
}

/// Helper function to create a test memory pool (legacy, for compatibility)
fn create_test_pool() -> Arc<HybridMemoryPool> {
    setup_global_memory_pool()
}

// =============================================================================
// Basic Tensor Creation Tests
// =============================================================================

#[test]
fn test_tensor_creation_zeros() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    // Test various shapes and data types
    let test_cases = vec![
        (vec![2, 3], BitNetDType::F32),
        (vec![1], BitNetDType::F16),
        (vec![5, 5, 5], BitNetDType::I32),
    ];

    for (shape, dtype) in test_cases {
        let tensor =
            BitNetTensor::zeros(&shape, dtype, Some(device.clone())).unwrap_or_else(|_| {
                panic!("Failed to create zeros tensor: shape={shape:?}, dtype={dtype:?}")
            });

        assert_eq!(tensor.shape().dims(), &shape);
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.element_count(), shape.iter().product::<usize>());
    }
}

#[test]
fn test_tensor_creation_ones() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    // Test various shapes and data types
    let test_cases = vec![
        (vec![2, 3], BitNetDType::F32),
        (vec![1], BitNetDType::F16),
        (vec![5, 5, 5], BitNetDType::I32),
    ];

    for (shape, dtype) in test_cases {
        let tensor = BitNetTensor::ones(&shape, dtype, Some(device.clone())).unwrap_or_else(|_| {
            panic!("Failed to create ones tensor: shape={shape:?}, dtype={dtype:?}")
        });

        assert_eq!(tensor.shape().dims(), &shape);
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.element_count(), shape.iter().product::<usize>());
    }
}

#[test]
fn test_tensor_creation_from_data() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    // Test f32 data
    let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = BitNetTensor::from_vec(data_f32, &[2, 3], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor from f32 data");

    assert_eq!(tensor.shape().dims(), &[2, 3]);
    assert_eq!(tensor.dtype(), BitNetDType::F32);
    assert_eq!(tensor.element_count(), 6);

    // Test i32 data
    let data_i32 = vec![1i32, 2, 3, 4];
    let tensor = BitNetTensor::from_vec(data_i32, &[2, 2], BitNetDType::I32, Some(device.clone()))
        .expect("Failed to create tensor from i32 data");

    assert_eq!(tensor.shape().dims(), &[2, 2]);
    assert_eq!(tensor.dtype(), BitNetDType::I32);
    assert_eq!(tensor.element_count(), 4);
}

// =============================================================================
// BitNet Specific Tests
// =============================================================================

#[test]
fn test_tensor_bitnet_quantized_creation() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::bitnet_158(&[10, 20], Some(device.clone()))
        .expect("Failed to create BitNet 1.58 tensor");

    assert_eq!(tensor.dtype(), BitNetDType::BitNet158);
    assert_eq!(tensor.shape().dims(), &[10, 20]);
    assert_eq!(tensor.element_count(), 200);
    assert!(tensor.is_allocated());
}

// =============================================================================
// Memory Integration Tests
// =============================================================================

#[test]
fn test_tensor_memory_pool_integration() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let mut tensors = Vec::new();

    // Create multiple tensors and verify memory tracking
    for i in 0..10 {
        let tensor = BitNetTensor::zeros(&[64, 64], BitNetDType::F32, Some(device.clone()))
            .unwrap_or_else(|_| panic!("Failed to create tensor {i}"));

        tensors.push(tensor);
    }

    // All tensors should be valid
    for tensor in &tensors {
        assert!(tensor.validate().is_ok());
    }

    // Drop tensors and verify cleanup
    drop(tensors);
}

#[test]
fn test_tensor_memory_pressure_handling() {
    let _pool = setup_global_memory_pool();

    // This test should pass since memory pressure is handled by the pool
    let device = get_cpu_device();

    // Create tensors until we hit some reasonable limit
    let mut tensors = Vec::new();

    for i in 0..20 {
        if let Ok(tensor) = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, Some(device.clone()))
        {
            tensors.push(tensor);
        } else {
            // Memory pressure hit, which is expected behavior
            break;
        }

        // Prevent infinite loops in testing
        if i >= 19 {
            break;
        }
    }

    // Should have created at least some tensors
    assert!(
        !tensors.is_empty(),
        "Should have created at least one tensor"
    );
}

// =============================================================================
// Device Tests
// =============================================================================

#[test]
fn test_tensor_device_migration() {
    let _pool = setup_global_memory_pool();

    let cpu_device = get_cpu_device();
    let tensor = BitNetTensor::zeros(&[5, 5], BitNetDType::F32, Some(cpu_device.clone()))
        .expect("Failed to create CPU tensor");

    assert!(devices_equal(tensor.device(), &cpu_device));

    // Test device migration
    let migrated_tensor = tensor
        .to_device(&cpu_device)
        .expect("Failed to migrate tensor to CPU");

    assert!(devices_equal(migrated_tensor.device(), &cpu_device));
    assert_eq!(migrated_tensor.shape().dims(), tensor.shape().dims());
    assert_eq!(migrated_tensor.dtype(), tensor.dtype());
}

#[test]
fn test_automatic_device_selection() {
    let _pool = setup_global_memory_pool();

    let tensor = BitNetTensor::zeros(&[3, 3], BitNetDType::F32, None)
        .expect("Failed to create tensor with auto device selection");

    // Should have selected some device
    assert_eq!(tensor.shape().dims(), &[3, 3]);
    assert_eq!(tensor.dtype(), BitNetDType::F32);
}

// =============================================================================
// Shape Operations Tests
// =============================================================================

#[test]
fn test_tensor_shape_operations() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let tensor = BitNetTensor::zeros(&[2, 3, 4], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create 1D tensor");

    // Test reshape
    let reshaped = tensor.reshape(&[6, 4]).expect("Failed to reshape tensor");
    assert_eq!(reshaped.shape().dims(), &[6, 4]);
    assert_eq!(reshaped.element_count(), 24);

    // Test transpose
    let transposed = tensor.transpose().expect("Failed to transpose tensor");
    assert_eq!(transposed.shape().dims(), &[2, 4, 3]); // Last two dims swapped

    // Test squeeze
    let tensor_with_ones =
        BitNetTensor::zeros(&[1, 3, 1, 4], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create tensor with size-1 dimensions");

    let squeezed = tensor_with_ones
        .squeeze()
        .expect("Failed to squeeze tensor");
    assert_eq!(squeezed.shape().dims(), &[3, 4]);
}

// =============================================================================
// Broadcasting Tests
// =============================================================================

#[test]
fn test_tensor_broadcasting_compatibility() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let tensor_a = BitNetTensor::zeros(&[3, 1], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor A");
    let tensor_b = BitNetTensor::zeros(&[1, 4], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor B");

    assert!(tensor_a.is_broadcast_compatible(&tensor_b));

    let broadcast_shape = tensor_a
        .broadcast_shape(&tensor_b)
        .expect("Failed to compute broadcast shape");
    assert_eq!(broadcast_shape.dims(), &[3, 4]);

    // Test incompatible shapes - [3, 1] vs [2, 5] should be incompatible
    // because 3 != 2 and neither is 1
    let tensor_c = BitNetTensor::zeros(&[2, 5], BitNetDType::F32, Some(device))
        .expect("Failed to create tensor C");
    assert!(!tensor_a.is_broadcast_compatible(&tensor_c));
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

#[test]
fn test_tensor_thread_safety() {
    let pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let shared_tensor = Arc::new(
        BitNetTensor::zeros(&[10, 10], BitNetDType::F32, Some(device))
            .expect("Failed to create shared tensor"),
    );

    let mut handles = vec![];

    for i in 0..5 {
        let tensor_clone = Arc::clone(&shared_tensor);
        let pool_clone = pool.clone();

        let handle = thread::spawn(move || {
            let _pool = pool_clone; // Keep pool alive

            // Access tensor properties from multiple threads
            let shape = tensor_clone.shape();
            let dtype = tensor_clone.dtype();
            let element_count = tensor_clone.element_count();

            assert_eq!(shape.dims(), &[10, 10]);
            assert_eq!(dtype, BitNetDType::F32);
            assert_eq!(element_count, 100);

            format!("Thread {i} completed")
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        let result = handle.join().expect("Thread should complete successfully");
        assert!(result.starts_with("Thread"));
    }
}

#[test]
fn test_tensor_concurrent_operations() {
    let pool = setup_global_memory_pool();
    let device = get_cpu_device();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let pool_clone = pool.clone();
            let device_clone = device.clone();

            thread::spawn(move || {
                let _pool = pool_clone; // Keep pool alive

                BitNetTensor::zeros(&[32, 32], BitNetDType::F32, Some(device_clone))
                    .expect("Failed to create tensor concurrently")
            })
        })
        .collect();

    let tensors: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Concurrent tensor creation failed"))
        .collect();

    assert_eq!(tensors.len(), 8);

    for tensor in &tensors {
        assert_eq!(tensor.shape().dims(), &[32, 32]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
        assert!(tensor.validate().is_ok());
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_tensor_error_handling() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    // Test invalid reshape
    let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, Some(device))
        .expect("Failed to create tensor");

    let result = tensor.reshape(&[2, 2]); // Different number of elements
    assert!(
        result.is_err(),
        "Reshape with different element count should fail"
    );

    // Test empty tensor creation (should be valid)
    let empty_result = BitNetTensor::zeros(&[0], BitNetDType::F32, None);
    assert!(empty_result.is_ok(), "Empty tensor should be valid");
}

// =============================================================================
// Resource Cleanup Tests
// =============================================================================

#[test]
fn test_tensor_resource_cleanup() {
    let pool = setup_global_memory_pool();
    let device = get_cpu_device();

    {
        let tensors: Vec<_> = (0..10)
            .map(|_| {
                BitNetTensor::zeros(&[64, 64], BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor")
            })
            .collect();

        // Tensors are alive here
        for tensor in &tensors {
            assert!(tensor.validate().is_ok());
        }

        // Tensors will be dropped when exiting this scope
    }

    // After scope exit, tensors should be cleaned up
    // Memory pool should still be valid
    drop(pool);
}

// =============================================================================
// Data Type Validation Tests
// =============================================================================

#[test]
fn test_tensor_data_type_validation() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    for &dtype in &[
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I8,
        BitNetDType::U8,
    ] {
        let tensor = BitNetTensor::zeros(&[4, 4], dtype, Some(device.clone()))
            .unwrap_or_else(|_| panic!("Failed to create tensor with dtype {dtype:?}"));

        assert_eq!(tensor.dtype(), dtype);
        assert!(tensor.dtype().is_valid());
        assert!(tensor.dtype().is_numeric());

        let expected_size = dtype.size().unwrap_or(4) * 16; // 4x4 elements
        assert!(tensor.size_bytes() >= expected_size);
    }
}

// =============================================================================
// BitNet Specific Type Tests
// =============================================================================

#[test]
fn test_tensor_bitnet_specific_types() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();

    // Test BitNet 1.58
    let bitnet_158 = BitNetTensor::zeros(&[8, 8], BitNetDType::BitNet158, Some(device.clone()))
        .expect("Failed to create BitNet 1.58 tensor");
    assert!(bitnet_158.dtype().is_quantized());

    // Test BitNet 1.1
    let bitnet_11 = BitNetTensor::zeros(&[8, 8], BitNetDType::BitNet11, Some(device.clone()))
        .expect("Failed to create BitNet 1.1 tensor");
    assert!(bitnet_11.dtype().is_quantized());

    // Test BitNet 1
    let bitnet_1 = BitNetTensor::zeros(&[8, 8], BitNetDType::BitNet1, Some(device))
        .expect("Failed to create BitNet 1 tensor");
    assert!(bitnet_1.dtype().is_quantized());
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
fn test_tensor_creation_performance() {
    let _pool = setup_global_memory_pool();
    let device = get_cpu_device();
    let num_tensors: usize = 100;

    let start_time = Instant::now();

    let tensors: Vec<_> = (0..num_tensors)
        .map(|i| {
            BitNetTensor::zeros(&[64, 64], BitNetDType::F32, Some(device.clone()))
                .unwrap_or_else(|_| panic!("Failed to create tensor {i}"))
        })
        .collect();

    let creation_time = start_time.elapsed();

    println!("Created {num_tensors} tensors in {creation_time:?}");
    println!(
        "Average per tensor: {:?}",
        creation_time / num_tensors as u32
    );

    // Verify all tensors were created correctly
    assert_eq!(tensors.len(), num_tensors);

    // Basic performance expectation (adjust as needed)
    let max_expected_time = Duration::from_millis(1000); // 1 second for 100 tensors
    assert!(
        creation_time < max_expected_time,
        "Tensor creation took too long: {creation_time:?}"
    );
}
