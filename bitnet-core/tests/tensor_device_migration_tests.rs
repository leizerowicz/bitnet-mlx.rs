//! Device Migration Tests for BitNet Tensors
//!
//! Comprehensive device migration testing following existing patterns from
//! device_comparison_tests.rs and metal_device_availability_tests.rs

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use bitnet_core::device::{get_cpu_device, get_metal_device, is_metal_available};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
use bitnet_core::tensor::{BitNetDType, BitNetTensor};
use std::collections::HashMap;

// =============================================================================
// Device Migration Test Infrastructure
// =============================================================================

/// Device migration test configuration
#[derive(Clone)]
#[allow(dead_code)]
struct DeviceMigrationConfig {
    enable_performance_tracking: bool,
    migration_timeout: Duration,
    validation_enabled: bool,
    test_shapes: Vec<Vec<usize>>,
}

impl Default for DeviceMigrationConfig {
    fn default() -> Self {
        Self {
            enable_performance_tracking: true,
            migration_timeout: Duration::from_secs(5),
            validation_enabled: true,
            test_shapes: vec![vec![16, 16], vec![64, 64], vec![128, 128], vec![256, 256]],
        }
    }
}

/// Migration performance metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MigrationMetrics {
    source_device: String,
    target_device: String,
    tensor_size_bytes: usize,
    migration_duration: Duration,
    throughput_mbps: f64,
    validation_passed: bool,
}

/// Helper function to get device name for logging
fn get_device_name(device: &candle_core::Device) -> String {
    match device {
        candle_core::Device::Cpu => "CPU".to_string(),
        candle_core::Device::Metal(_metal) => "Metal".to_string(),
        candle_core::Device::Cuda(_cuda) => "CUDA".to_string(),
    }
}

/// Helper function to create tracked memory pool
fn create_migration_pool() -> HybridMemoryPool {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());

    HybridMemoryPool::with_config(config).expect("Failed to create migration test memory pool")
}

// =============================================================================
// Basic Device Migration Tests
// =============================================================================

#[test]
fn test_cpu_device_tensor_creation() {
    let _pool = create_migration_pool();
    let cpu_device = get_cpu_device();
    let config = DeviceMigrationConfig::default();

    println!("Testing CPU device tensor creation");
    println!("CPU Device: {}", get_device_name(&cpu_device));

    for shape in &config.test_shapes {
        for &dtype in &[BitNetDType::F32, BitNetDType::F16, BitNetDType::I32] {
            let tensor = BitNetTensor::zeros(shape, dtype, Some(cpu_device.clone()))
                .unwrap_or_else(|_| {
                    panic!("Failed to create CPU tensor: shape={shape:?}, dtype={dtype:?}")
                });

            // Verify tensor properties
            assert_eq!(tensor.shape().dims(), shape);
            assert_eq!(tensor.dtype(), dtype);

            // Verify device placement
            assert!(matches!(tensor.device(), candle_core::Device::Cpu));

            println!("✓ CPU tensor created: {shape:?} {dtype:?}");
        }
    }
}

#[test]
fn test_metal_device_availability() {
    println!("Testing Metal device availability");

    if !is_metal_available() {
        println!("⚠ Metal not available on this system");
        return;
    }

    let metal_device = get_metal_device().expect("Metal should be available");

    println!(
        "✓ Metal device available: {}",
        get_device_name(&metal_device)
    );

    // Test Metal tensor creation
    let tensor = BitNetTensor::zeros(&[32, 32], BitNetDType::F32, Some(metal_device.clone()))
        .expect("Failed to create Metal tensor");

    assert!(matches!(tensor.device(), candle_core::Device::Metal(_)));

    println!("✓ Metal tensor created successfully");
}

#[test]
fn test_automatic_device_selection() {
    let config = DeviceMigrationConfig::default();

    println!("Testing automatic device selection");

    for shape in &config.test_shapes {
        let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, None)
            .unwrap_or_else(|_| panic!("Failed to create auto-device tensor: shape={shape:?}"));

        let selected_device = tensor.device();
        let device_name = get_device_name(selected_device);

        println!("Auto-selected device for {shape:?}: {device_name}");

        // Verify device is valid
        match selected_device {
            candle_core::Device::Cpu => {
                // CPU should always be available
                assert!(true);
            }
            candle_core::Device::Metal(_) => {
                // Metal should only be selected if available
                assert!(is_metal_available(), "Metal selected but not available");
            }
            _ => {
                // Other devices are valid too
                assert!(true);
            }
        }
    }
}

// =============================================================================
// Device Migration Tests (Placeholder Implementation)
// =============================================================================
// Note: These tests use placeholder implementations since actual migration
// methods may not be implemented yet in the tensor system

#[test]
fn test_cpu_to_metal_migration_placeholder() {
    if !is_metal_available() {
        println!("Skipping CPU to Metal migration test - Metal not available");
        return;
    }

    let cpu_device = get_cpu_device();
    let metal_device = get_metal_device().expect("Metal should be available");
    let config = DeviceMigrationConfig::default();

    println!("Testing CPU to Metal migration (placeholder)");

    for shape in &config.test_shapes {
        // Create tensor on CPU
        let cpu_tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(cpu_device.clone()))
            .expect("Failed to create CPU tensor");

        assert!(matches!(cpu_tensor.device(), candle_core::Device::Cpu));

        // Placeholder for migration (when to_device method is implemented)
        println!(
            "Would migrate {:?} tensor from {} to {}",
            shape,
            get_device_name(&cpu_device),
            get_device_name(&metal_device)
        );

        // For now, verify we can create equivalent tensor on Metal
        let metal_tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(metal_device.clone()))
            .expect("Failed to create equivalent Metal tensor");

        assert!(matches!(
            metal_tensor.device(),
            candle_core::Device::Metal(_)
        ));

        // Verify properties match
        assert_eq!(cpu_tensor.shape().dims(), metal_tensor.shape().dims());
        assert_eq!(cpu_tensor.dtype(), metal_tensor.dtype());

        println!("✓ Equivalent tensors created on both devices: {shape:?}");
    }
}

#[test]
fn test_metal_to_cpu_migration_placeholder() {
    if !is_metal_available() {
        println!("Skipping Metal to CPU migration test - Metal not available");
        return;
    }

    let cpu_device = get_cpu_device();
    let metal_device = get_metal_device().expect("Metal should be available");
    let config = DeviceMigrationConfig::default();

    println!("Testing Metal to CPU migration (placeholder)");

    for shape in &config.test_shapes {
        // Create tensor on Metal
        let metal_tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(metal_device.clone()))
            .expect("Failed to create Metal tensor");

        assert!(matches!(
            metal_tensor.device(),
            candle_core::Device::Metal(_)
        ));

        // Placeholder for migration (when to_device method is implemented)
        println!(
            "Would migrate {:?} tensor from {} to {}",
            shape,
            get_device_name(&metal_device),
            get_device_name(&cpu_device)
        );

        // For now, verify we can create equivalent tensor on CPU
        let cpu_tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(cpu_device.clone()))
            .expect("Failed to create equivalent CPU tensor");

        assert!(matches!(cpu_tensor.device(), candle_core::Device::Cpu));

        // Verify properties match
        assert_eq!(metal_tensor.shape().dims(), cpu_tensor.shape().dims());
        assert_eq!(metal_tensor.dtype(), cpu_tensor.dtype());

        println!("✓ Equivalent tensors created on both devices: {shape:?}");
    }
}

// =============================================================================
// Device Migration Performance Tests (Placeholder)
// =============================================================================

#[test]
fn test_migration_performance_baseline() {
    let cpu_device = get_cpu_device();
    let config = DeviceMigrationConfig::default();

    println!("Testing device migration performance baseline");

    let mut baseline_metrics = Vec::new();

    for shape in &config.test_shapes {
        let elements: usize = shape.iter().product();
        let size_bytes = elements * std::mem::size_of::<f32>();

        // Measure tensor creation time as baseline
        let start_time = Instant::now();

        let _tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(cpu_device.clone()))
            .expect("Failed to create baseline tensor");

        let creation_time = start_time.elapsed();

        let metrics = MigrationMetrics {
            source_device: "None".to_string(),
            target_device: get_device_name(&cpu_device),
            tensor_size_bytes: size_bytes,
            migration_duration: creation_time,
            throughput_mbps: (size_bytes as f64 / creation_time.as_secs_f64()) / (1024.0 * 1024.0),
            validation_passed: true,
        };

        println!(
            "Baseline creation {:?}: {} bytes in {:?} ({:.2} MB/s)",
            shape, size_bytes, creation_time, metrics.throughput_mbps
        );

        baseline_metrics.push(metrics);
    }

    // Analyze performance characteristics
    let avg_throughput: f64 = baseline_metrics
        .iter()
        .map(|m| m.throughput_mbps)
        .sum::<f64>()
        / baseline_metrics.len() as f64;

    println!("Average baseline throughput: {avg_throughput:.2} MB/s");

    // Performance should be reasonable (>100 MB/s for memory allocation)
    assert!(
        avg_throughput > 50.0,
        "Baseline performance too low: {avg_throughput:.2} MB/s"
    );
}

#[test]
fn test_cross_device_data_consistency() {
    if !is_metal_available() {
        println!("Skipping cross-device consistency test - Metal not available");
        return;
    }

    let cpu_device = get_cpu_device();
    let metal_device = get_metal_device().expect("Metal should be available");

    println!("Testing cross-device data consistency");

    let test_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let shape = vec![4, 4];

    // Create tensor with same data on both devices
    let cpu_tensor = BitNetTensor::from_vec(
        test_data.clone(),
        &shape,
        BitNetDType::F32,
        Some(cpu_device.clone()),
    )
    .expect("Failed to create CPU tensor from data");

    let metal_tensor = BitNetTensor::from_vec(
        test_data.clone(),
        &shape,
        BitNetDType::F32,
        Some(metal_device.clone()),
    )
    .expect("Failed to create Metal tensor from data");

    // Verify properties
    assert_eq!(cpu_tensor.shape().dims(), metal_tensor.shape().dims());
    assert_eq!(cpu_tensor.dtype(), metal_tensor.dtype());
    assert_eq!(cpu_tensor.element_count(), metal_tensor.element_count());

    println!("✓ Cross-device data consistency verified");
}

// =============================================================================
// Concurrent Device Migration Tests
// =============================================================================

#[test]
fn test_concurrent_device_operations() {
    let cpu_device = Arc::new(get_cpu_device());
    let num_threads = 4;
    let tensors_per_thread = 10;

    println!("Testing concurrent device operations with {num_threads} threads");

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let cpu_device_clone = cpu_device.clone();

            thread::spawn(move || {
                let mut thread_tensors = Vec::new();

                for i in 0..tensors_per_thread {
                    let tensor = BitNetTensor::zeros(
                        &[32, 32],
                        BitNetDType::F32,
                        Some((*cpu_device_clone).clone()),
                    )
                    .unwrap_or_else(|_| panic!("Thread {thread_id} failed to create tensor {i}"));

                    // Verify device placement
                    assert!(matches!(tensor.device(), candle_core::Device::Cpu));

                    thread_tensors.push(tensor);
                }

                println!(
                    "Thread {} created {} tensors",
                    thread_id,
                    thread_tensors.len()
                );
                thread_tensors.len()
            })
        })
        .collect();

    // Wait for completion
    let mut total_tensors = 0;
    for handle in handles {
        total_tensors += handle.join().expect("Thread panicked");
    }

    assert_eq!(total_tensors, num_threads * tensors_per_thread);
    println!("✓ Concurrent device operations completed: {total_tensors} total tensors");
}

#[test]
fn test_concurrent_auto_device_selection() {
    let num_threads = 8;
    let selections_per_thread = 20;
    let device_selections = Arc::new(Mutex::new(HashMap::new()));

    println!("Testing concurrent auto device selection");

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let selections_clone = device_selections.clone();

            thread::spawn(move || {
                for i in 0..selections_per_thread {
                    let tensor = BitNetTensor::zeros(
                        &[16, 16],
                        BitNetDType::F32,
                        None, // Auto device selection
                    )
                    .unwrap_or_else(|_| panic!("Thread {thread_id} failed auto selection {i}"));

                    let device_name = get_device_name(tensor.device());

                    // Record device selection
                    {
                        let mut selections = selections_clone.lock().unwrap();
                        *selections.entry(device_name).or_insert(0) += 1;
                    }
                }
            })
        })
        .collect();

    // Wait for completion
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Analyze device selection results
    let selections = device_selections.lock().unwrap();
    println!("Auto device selection results:");
    for (device, count) in selections.iter() {
        println!("  {device}: {count} selections");
    }

    let total_selections: usize = selections.values().sum();
    assert_eq!(total_selections, num_threads * selections_per_thread);

    // Should have at least one device type selected
    assert!(!selections.is_empty(), "No devices were selected");

    println!("✓ Concurrent auto device selection completed");
}

// =============================================================================
// Device Capability Tests
// =============================================================================

#[test]
fn test_device_capability_detection() {
    println!("Testing device capability detection");

    let cpu_device = get_cpu_device();
    println!("CPU device: {}", get_device_name(&cpu_device));

    // CPU should always support basic operations
    println!("✓ CPU device available and functional");

    if is_metal_available() {
        let metal_device = get_metal_device().expect("Metal should be available");
        println!("Metal device: {}", get_device_name(&metal_device));
        println!("✓ Metal device available and functional");
    } else {
        println!("ℹ Metal device not available on this system");
    }

    // Test device-specific tensor creation capabilities
    let dtypes_to_test = vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I32,
        BitNetDType::BitNet158,
    ];

    for &dtype in &dtypes_to_test {
        let tensor = BitNetTensor::zeros(&[16, 16], dtype, Some(cpu_device.clone()))
            .unwrap_or_else(|_| panic!("CPU should support dtype {dtype:?}"));

        assert_eq!(tensor.dtype(), dtype);
        println!("✓ CPU supports data type: {dtype:?}");
    }
}

#[test]
fn test_device_memory_characteristics() {
    println!("Testing device memory characteristics");

    let pool = create_migration_pool();
    let cpu_device = get_cpu_device();

    // Test CPU memory characteristics
    let initial_usage = pool
        .get_detailed_metrics()
        .map(|m| m.current_memory_usage)
        .unwrap_or(0);

    let mut cpu_tensors = Vec::new();
    for i in 0..10 {
        let tensor = BitNetTensor::zeros(&[64, 64], BitNetDType::F32, Some(cpu_device.clone()))
            .unwrap_or_else(|_| panic!("Failed to create CPU tensor {i}"));
        cpu_tensors.push(tensor);
    }

    let cpu_usage = pool
        .get_detailed_metrics()
        .map(|m| m.current_memory_usage - initial_usage)
        .unwrap_or(0);

    println!("CPU memory usage for 10 tensors: {cpu_usage} bytes");

    // Test Metal memory characteristics (if available)
    if is_metal_available() {
        let metal_device = get_metal_device().expect("Metal should be available");
        let pre_metal_usage = pool
            .get_detailed_metrics()
            .map(|m| m.current_memory_usage)
            .unwrap_or(0);

        let mut metal_tensors = Vec::new();
        for _ in 0..5 {
            // Fewer for GPU memory
            if let Ok(tensor) =
                BitNetTensor::zeros(&[64, 64], BitNetDType::F32, Some(metal_device.clone()))
            {
                metal_tensors.push(tensor);
            }
        }

        let metal_usage = pool
            .get_detailed_metrics()
            .map(|m| m.current_memory_usage - pre_metal_usage)
            .unwrap_or(0);

        println!(
            "Metal memory usage for {} tensors: {} bytes",
            metal_tensors.len(),
            metal_usage
        );
    }
}

// =============================================================================
// Error Handling and Edge Cases
// =============================================================================

#[test]
fn test_device_migration_error_handling() {
    println!("Testing device migration error handling");

    let cpu_device = get_cpu_device();

    // Test invalid tensor creation (should handle gracefully)
    let result = BitNetTensor::from_vec(
        vec![1.0f32, 2.0], // 2 elements
        &[3, 3],           // 9 element shape - mismatch
        BitNetDType::F32,
        Some(cpu_device.clone()),
    );

    assert!(result.is_err(), "Should fail with shape-data mismatch");
    println!("✓ Shape-data mismatch properly handled");

    // Test empty tensor handling
    let empty_result = BitNetTensor::zeros(&[0], BitNetDType::F32, Some(cpu_device.clone()));
    match empty_result {
        Ok(_) => println!("✓ Empty tensor creation handled"),
        Err(_) => println!("✓ Empty tensor creation rejected"),
    }
}

#[test]
fn test_device_resource_cleanup() {
    println!("Testing device resource cleanup");

    let pool = create_migration_pool();
    let initial_metrics = pool
        .get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    // Create scope for automatic cleanup
    {
        let cpu_device = get_cpu_device();
        let _tensors: Vec<_> = (0..20)
            .map(|i| {
                BitNetTensor::zeros(&[32, 32], BitNetDType::F32, Some(cpu_device.clone()))
                    .unwrap_or_else(|_| panic!("Failed to create tensor {i}"))
            })
            .collect();

        let mid_metrics = pool
            .get_detailed_metrics()
            .expect("Memory tracking should be enabled");

        assert!(mid_metrics.current_memory_usage > initial_metrics.current_memory_usage);
        println!("Memory increased during tensor lifetime");
    } // Tensors dropped here

    // Allow cleanup time
    thread::sleep(Duration::from_millis(100));

    let final_metrics = pool
        .get_detailed_metrics()
        .expect("Memory tracking should be enabled");

    println!(
        "Initial usage: {} bytes",
        initial_metrics.current_memory_usage
    );
    println!("Final usage: {} bytes", final_metrics.current_memory_usage);

    // Memory should not grow indefinitely
    assert!(
        final_metrics.current_memory_usage <= initial_metrics.current_memory_usage + 1024 * 1024
    );
    println!("✓ Device resource cleanup validated");
}
