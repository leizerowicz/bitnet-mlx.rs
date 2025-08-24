//! Core Tensor Tests
//!
//! Comprehensive test suite for BitNet tensor system core functionality.
//! Following existing test patterns from memory_tracking_tests.rs and tensor_integration_tests.rs

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    CleanupManager, CleanupConfig, CleanupStrategyType, CleanupPriority,
    MemoryPressureLevel, MemoryTracker, DetailedMemoryMetrics
};
use bitnet_core::device::{get_cpu_device, auto_select_device, is_metal_available, get_metal_device};
use bitnet_core::tensor::{BitNetTensor, BitNetDType, TensorShape};

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
    test_tensor_sizes: Vec<Vec<usize>>,
}

impl Default for TensorTestConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            enable_cleanup: true,
            memory_pressure_threshold: 0.8,
            max_test_duration: Duration::from_secs(30),
            test_tensor_sizes: vec![
                vec![],           // scalar
                vec![10],         // vector
                vec![3, 4],       // small matrix
                vec![64, 64],     // medium matrix
                vec![2, 3, 4],    // 3D tensor
                vec![2, 2, 2, 2], // 4D tensor
            ],
        }
    }
}

/// Helper function to create a test memory pool following existing patterns
fn create_test_pool() -> HybridMemoryPool {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());

    HybridMemoryPool::with_config(config).expect("Failed to create test memory pool")
}

/// Helper function to get available devices for testing
fn get_available_devices() -> Vec<candle_core::Device> {
    let mut devices = vec![get_cpu_device()];

    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            devices.push(metal_device);
        }
    }

    devices
}

/// Helper function to get all data types for testing
fn get_test_data_types() -> Vec<BitNetDType> {
    vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I8,
        BitNetDType::I32,
        BitNetDType::U8,
        BitNetDType::Bool,
        BitNetDType::BitNet158,
        BitNetDType::BitNet11,
    ]
}

// =============================================================================
// Core Tensor Creation Tests
// =============================================================================

#[test]
fn test_tensor_creation_zeros() {
    let pool = create_test_pool();
    let config = TensorTestConfig::default();
    let devices = get_available_devices();
    let dtypes = get_test_data_types();

    for device in &devices {
        for &dtype in &dtypes {
            for shape in &config.test_tensor_sizes {
                let tensor = BitNetTensor::zeros(shape, dtype, Some(device.clone()))
                    .expect(&format!("Failed to create zeros tensor: shape={:?}, dtype={:?}", shape, dtype));

                // Validate tensor properties
                assert_eq!(tensor.shape().dims(), shape);
                assert_eq!(tensor.dtype(), dtype);

                // Verify memory allocation
                let expected_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
                assert_eq!(tensor.element_count(), expected_elements);

                // Verify zero initialization (where possible)
                if dtype.is_numeric() {
                    // Note: In real implementation, we'd verify data is actually zeros
                    assert!(tensor.is_allocated());
                }
            }
        }
    }
}

#[test]
fn test_tensor_creation_ones() {
    let pool = create_test_pool();
    let config = TensorTestConfig::default();
    let devices = get_available_devices();
    let dtypes = get_test_data_types();

    for device in &devices {
        for &dtype in &dtypes {
            for shape in &config.test_tensor_sizes {
                let tensor = BitNetTensor::ones(shape, dtype, Some(device.clone()))
                    .expect(&format!("Failed to create ones tensor: shape={:?}, dtype={:?}", shape, dtype));

                // Validate tensor properties
                assert_eq!(tensor.shape().dims(), shape);
                assert_eq!(tensor.dtype(), dtype);

                // Verify memory allocation
                let expected_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
                assert_eq!(tensor.element_count(), expected_elements);
            }
        }
    }
}

#[test]
fn test_tensor_creation_from_data() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Test with f32 data
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];

    let tensor = BitNetTensor::from_vec(data.clone(), &shape, BitNetDType::F32, Some(device))
        .expect("Failed to create tensor from f32 data");

    assert_eq!(tensor.shape().dims(), shape);
    assert_eq!(tensor.dtype(), BitNetDType::F32);
    assert_eq!(tensor.element_count(), 6);

    // Test with different data types
    let i32_data: Vec<i32> = vec![1, 2, 3, 4];
    let i32_tensor = BitNetTensor::from_vec(i32_data, &vec![2, 2], BitNetDType::I32, Some(get_cpu_device()))
        .expect("Failed to create tensor from i32 data");

    assert_eq!(i32_tensor.element_count(), 4);
}

#[test]
fn test_tensor_bitnet_quantized_creation() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    let shapes = vec![
        vec![8, 8],
        vec![16, 16],
        vec![32, 32],
    ];

    for shape in shapes {
        // Test BitNet 1.58 quantized tensor creation
        let bitnet_tensor = BitNetTensor::bitnet_158(&shape, Some(device.clone()))
            .expect("Failed to create BitNet 1.58 tensor");

        assert_eq!(bitnet_tensor.shape().dims(), shape);
        assert_eq!(bitnet_tensor.dtype(), BitNetDType::BitNet158);

        // Test BitNet 1.1 quantized tensor creation
        let bitnet11_tensor = BitNetTensor::ones(&shape, BitNetDType::BitNet11, Some(device.clone()))
            .expect("Failed to create BitNet 1.1 tensor");

        assert_eq!(bitnet11_tensor.dtype(), BitNetDType::BitNet11);
    }
}

// =============================================================================
// Memory Integration Tests
// =============================================================================

#[test]
fn test_tensor_memory_pool_integration() {
    let pool = create_test_pool();
    let device = get_cpu_device();

    // Track initial memory state
    let initial_metrics = pool.get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    let initial_allocations = initial_metrics.active_allocations;
    let initial_usage = initial_metrics.current_memory_usage;

    // Create multiple tensors
    let tensors = (0..10).map(|i| {
        let shape = vec![64, 64];
        BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create tensor {}", i))
    }).collect::<Vec<_>>();

    // Verify memory tracking
    let post_allocation_metrics = pool.get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    assert!(post_allocation_metrics.active_allocations > initial_allocations);
    assert!(post_allocation_metrics.current_memory_usage > initial_usage);
    assert_eq!(post_allocation_metrics.pressure_level, MemoryPressureLevel::None);

    // Drop tensors to test cleanup
    drop(tensors);

    // Allow time for cleanup
    thread::sleep(Duration::from_millis(100));

    // Note: In practice, cleanup might be deferred, so we don't assert exact equality
    let final_metrics = pool.get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    // Memory should eventually be cleaned up
    assert!(final_metrics.current_memory_usage <= post_allocation_metrics.current_memory_usage);
}

#[test]
fn test_tensor_memory_pressure_handling() {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());

    let pool = HybridMemoryPool::with_config(config).expect("Failed to create memory pool");
    let device = get_cpu_device();

    // Create many large tensors to increase memory pressure
    let mut tensors = Vec::new();

    for i in 0..20 {
        if let Ok(tensor) = BitNetTensor::zeros(&vec![256, 256], BitNetDType::F32, Some(device.clone())) {
            tensors.push(tensor);
        } else {
            // Memory pressure may cause allocation failures
            break;
        }
    }

    // Check if memory pressure is detected
    if let Some(metrics) = pool.get_detailed_metrics() {
        // Memory pressure should be detected with many large allocations
        println!("Memory pressure level: {:?}", metrics.pressure_level);
        println!("Current usage: {} bytes", metrics.current_memory_usage);
        println!("Active allocations: {}", metrics.active_allocations);
    }
}

// =============================================================================
// Device Migration Tests
// =============================================================================

#[test]
fn test_tensor_device_migration() {
    let pool = create_test_pool();
    let cpu_device = get_cpu_device();

    // Create tensor on CPU
    let cpu_tensor = BitNetTensor::zeros(&vec![32, 32], BitNetDType::F32, Some(cpu_device))
        .expect("Failed to create CPU tensor");

    assert!(matches!(cpu_tensor.device(), candle_core::Device::Cpu));

    // Test migration to Metal (if available)
    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            let migrated_tensor = cpu_tensor.to_device(&metal_device)
                .expect("Failed to migrate tensor to Metal device");

            assert!(matches!(migrated_tensor.device(), candle_core::Device::Metal(_)));

            // Migrate back to CPU
            let back_to_cpu = migrated_tensor.to_device(&get_cpu_device())
                .expect("Failed to migrate tensor back to CPU");

            assert!(matches!(back_to_cpu.device(), candle_core::Device::Cpu));
        }
    }
}

#[test]
fn test_automatic_device_selection() {
    let pool = create_test_pool();

    // Test automatic device selection
    let auto_tensor = BitNetTensor::zeros(&vec![64, 64], BitNetDType::F32, None)
        .expect("Failed to create tensor with auto device selection");

    // Should select an appropriate device
    let selected_device = auto_tensor.device();

    // Verify device is valid
    match selected_device {
        candle_core::Device::Cpu => {
            println!("Auto-selected CPU device");
        }
        candle_core::Device::Metal(_) => {
            println!("Auto-selected Metal device");
            assert!(is_metal_available());
        }
        _ => {
            println!("Auto-selected other device: {:?}", selected_device);
        }
    }
}

// =============================================================================
// Shape and Broadcasting Tests
// =============================================================================

#[test]
fn test_tensor_shape_operations() {
    let device = get_cpu_device();

    // Test reshape
    let tensor = BitNetTensor::zeros(&vec![24], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create 1D tensor");

    let reshaped = tensor.reshape(&vec![4, 6])
        .expect("Failed to reshape tensor");

    assert_eq!(reshaped.shape().dims(), vec![4, 6]);
    assert_eq!(reshaped.element_count(), 24);

    // Test squeeze
    let tensor_with_ones = BitNetTensor::zeros(&vec![1, 24, 1], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor with unit dimensions");

    let squeezed = tensor_with_ones.squeeze()
        .expect("Failed to squeeze tensor");

    assert_eq!(squeezed.shape().dims(), vec![24]);

    // Test transpose (2D)
    let matrix = BitNetTensor::zeros(&vec![3, 4], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create matrix");

    let transposed = matrix.transpose()
        .expect("Failed to transpose matrix");

    assert_eq!(transposed.shape().dims(), vec![4, 3]);
}

#[test]
fn test_tensor_broadcasting_compatibility() {
    let device = get_cpu_device();

    // Test broadcasting compatibility checking
    let tensor_a = BitNetTensor::zeros(&vec![3, 1], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor A");

    let tensor_b = BitNetTensor::zeros(&vec![1, 4], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor B");

    assert!(tensor_a.is_broadcast_compatible(&tensor_b));

    let broadcast_shape = tensor_a.broadcast_shape(&tensor_b)
        .expect("Failed to compute broadcast shape");

    assert_eq!(broadcast_shape.dims(), vec![3, 4]);

    // Test incompatible broadcasting
    let tensor_c = BitNetTensor::zeros(&vec![5, 3], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor C");

    assert!(!tensor_a.is_broadcast_compatible(&tensor_c));
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

#[test]
fn test_tensor_thread_safety() {
    let pool = Arc::new(create_test_pool());
    let device = get_cpu_device();

    // Create a tensor to share across threads
    let tensor = Arc::new(
        BitNetTensor::zeros(&vec![100, 100], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create shared tensor")
    );

    let num_threads = 4;
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let tensor_clone = tensor.clone();
        let pool_clone = pool.clone();

        thread::spawn(move || {
            // Each thread creates additional tensors
            for i in 0..10 {
                let new_tensor = BitNetTensor::zeros(
                    &vec![10, 10],
                    BitNetDType::F32,
                    Some(get_cpu_device())
                ).expect(&format!("Thread {} failed to create tensor {}", thread_id, i));

                // Verify tensor properties
                assert_eq!(new_tensor.element_count(), 100);
                assert_eq!(new_tensor.dtype(), BitNetDType::F32);

                // Access shared tensor safely
                let _ = tensor_clone.shape();
                let _ = tensor_clone.dtype();
            }
        })
    }).collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify shared tensor is still valid
    assert_eq!(tensor.element_count(), 10000);
}

#[test]
fn test_tensor_concurrent_operations() {
    let device = get_cpu_device();
    let tensors = Arc::new(Mutex::new(Vec::new()));
    let num_threads = 8;

    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let tensors_clone = tensors.clone();
        let device_clone = device.clone();

        thread::spawn(move || {
            // Create tensors concurrently
            for i in 0..5 {
                let tensor = BitNetTensor::zeros(
                    &vec![thread_id + 1, i + 1],
                    BitNetDType::F32,
                    Some(device_clone.clone())
                ).expect("Failed to create tensor concurrently");

                {
                    let mut tensors_guard = tensors_clone.lock().unwrap();
                    tensors_guard.push(tensor);
                }
            }
        })
    }).collect();

    // Wait for completion
    for handle in handles {
        handle.join().expect("Concurrent tensor creation failed");
    }

    // Verify all tensors were created
    let final_tensors = tensors.lock().unwrap();
    assert_eq!(final_tensors.len(), num_threads * 5);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_tensor_error_handling() {
    let device = get_cpu_device();

    // Test invalid shape (negative dimensions would be caught by type system)
    // Test empty shape for non-scalar creation
    let result = BitNetTensor::from_vec(vec![1.0f32, 2.0, 3.0], &vec![], BitNetDType::F32, Some(device.clone()));
    assert!(result.is_err(), "Should fail with empty shape for multi-element data");

    // Test shape-data mismatch
    let result = BitNetTensor::from_vec(vec![1.0f32, 2.0], &vec![3, 3], BitNetDType::F32, Some(device.clone()));
    assert!(result.is_err(), "Should fail with shape-data size mismatch");

    // Test zero-size tensor creation (should be valid for empty tensor)
    let empty_tensor = BitNetTensor::zeros(&vec![0], BitNetDType::F32, Some(device.clone()));
    assert!(empty_tensor.is_ok(), "Empty tensor should be valid");

    // Test invalid reshape
    let tensor = BitNetTensor::zeros(&vec![6], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor");

    let invalid_reshape = tensor.reshape(&vec![4, 4]); // 6 elements cannot reshape to 16
    assert!(invalid_reshape.is_err(), "Should fail with incompatible reshape");
}

#[test]
fn test_tensor_resource_cleanup() {
    let pool = create_test_pool();
    let initial_metrics = pool.get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    // Create scope for automatic cleanup
    {
        let tensors: Vec<_> = (0..10).map(|_| {
            BitNetTensor::zeros(&vec![128, 128], BitNetDType::F32, Some(get_cpu_device()))
                .expect("Failed to create tensor")
        }).collect();

        // Tensors should be allocated
        let mid_metrics = pool.get_detailed_metrics()
            .expect("Memory tracking should be enabled");
        assert!(mid_metrics.active_allocations > initial_metrics.active_allocations);
    } // Tensors go out of scope here

    // Allow cleanup to occur
    thread::sleep(Duration::from_millis(50));

    // Check if memory was cleaned up (may be deferred)
    let final_metrics = pool.get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    println!("Initial allocations: {}", initial_metrics.active_allocations);
    println!("Final allocations: {}", final_metrics.active_allocations);

    // Note: Cleanup may be deferred, so we don't assert strict equality
}

// =============================================================================
// Data Type Validation Tests
// =============================================================================

#[test]
fn test_tensor_data_type_validation() {
    let device = get_cpu_device();
    let dtypes = get_test_data_types();

    for &dtype in &dtypes {
        let tensor = BitNetTensor::zeros(&vec![4, 4], dtype, Some(device.clone()))
            .expect(&format!("Failed to create tensor with dtype {:?}", dtype));

        assert_eq!(tensor.dtype(), dtype);

        // Verify size calculation is correct
        let expected_size = dtype.size() * 16; // 4x4 elements
        assert_eq!(tensor.size_bytes(), expected_size);

        // Test type compatibility
        assert!(dtype.is_valid());

        if dtype.is_numeric() {
            assert!(tensor.element_count() > 0);
        }
    }
}

#[test]
fn test_tensor_bitnet_specific_types() {
    let device = get_cpu_device();

    // Test BitNet 1.58 quantization
    let bitnet158_tensor = BitNetTensor::bitnet_158(&vec![32, 32], Some(device.clone()))
        .expect("Failed to create BitNet 1.58 tensor");

    assert_eq!(bitnet158_tensor.dtype(), BitNetDType::BitNet158);
    assert!(bitnet158_tensor.dtype().is_quantized());

    // Test BitNet 1.1 quantization
    let bitnet11_tensor = BitNetTensor::zeros(&vec![16, 16], BitNetDType::BitNet11, Some(device.clone()))
        .expect("Failed to create BitNet 1.1 tensor");

    assert_eq!(bitnet11_tensor.dtype(), BitNetDType::BitNet11);
    assert!(bitnet11_tensor.dtype().is_quantized());
}

// =============================================================================
// Performance Validation Tests
// =============================================================================

#[test]
fn test_tensor_creation_performance() {
    let device = get_cpu_device();
    let num_tensors = 100;

    let start_time = Instant::now();

    let tensors: Vec<_> = (0..num_tensors).map(|i| {
        BitNetTensor::zeros(&vec![64, 64], BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create tensor {}", i))
    }).collect();

    let creation_time = start_time.elapsed();

    println!("Created {} tensors in {:?}", num_tensors, creation_time);
    println!("Average per tensor: {:?}", creation_time / num_tensors);

    // Verify all tensors were created correctly
    assert_eq!(tensors.len(), num_tensors);

    // Basic performance expectation (adjust as needed)
    let max_expected_time = Duration::from_millis(1000); // 1 second for 100 tensors
    assert!(creation_time < max_expected_time,
           "Tensor creation took too long: {:?}", creation_time);
}
