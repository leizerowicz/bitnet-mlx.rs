//! Comprehensive Tensor Integration Tests
//!
//! This test suite validates the complete tensor system integration including:
//! - Memory pool integration with tensor operations
//! - Device abstraction with tensor migration
//! - Cross-system interactions and error handling
//! - Performance validation and memory efficiency
//! - Thread safety and concurrent operations

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    CleanupManager, CleanupConfig, CleanupStrategyType, CleanupPriority,
    MemoryPressureLevel, MemoryTracker, DetailedMemoryMetrics
};
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType, TensorHandle, TensorMetadata};
use bitnet_core::device::{get_cpu_device, auto_select_device, is_metal_available, get_metal_device};
use candle_core::{Device, Tensor, DType};

// =============================================================================
// Test Configuration and Utilities
// =============================================================================

/// Comprehensive test configuration for integration testing
#[derive(Clone)]
struct IntegrationTestConfig {
    enable_tracking: bool,
    enable_cleanup: bool,
    enable_profiling: bool,
    memory_pressure_threshold: f64,
    cleanup_interval: Duration,
    max_test_duration: Duration,
    concurrent_threads: usize,
    tensor_size_range: (usize, usize),
    operations_per_thread: usize,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            enable_cleanup: true,
            enable_profiling: true,
            memory_pressure_threshold: 0.8,
            cleanup_interval: Duration::from_millis(100),
            max_test_duration: Duration::from_secs(30),
            concurrent_threads: 4,
            tensor_size_range: (1024, 1024 * 1024), // 1KB to 1MB
            operations_per_thread: 100,
        }
    }
}

#[derive(Debug, Clone)]
struct TestMetrics {
    total_tensors_created: usize,
    total_memory_allocated: usize,
    successful_operations: usize,
    failed_operations: usize,
    average_allocation_time: Duration,
    memory_efficiency: f64,
    device_migrations: usize,
}

impl Default for TestMetrics {
    fn default() -> Self {
        Self {
            total_tensors_created: 0,
            total_memory_allocated: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_allocation_time: Duration::ZERO,
            memory_efficiency: 0.0,
            device_migrations: 0,
        }
    }
}

// =============================================================================
// Memory Pool Integration Tests
// =============================================================================

#[test]
fn test_comprehensive_memory_pool_tensor_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing comprehensive memory pool tensor integration");

    let config = IntegrationTestConfig::default();
    let pool = create_test_memory_pool()?;
    let mut metrics = TestMetrics::default();

    // Test 1: Basic tensor allocation and deallocation
    test_basic_tensor_lifecycle(&pool, &mut metrics)?;

    // Test 2: Concurrent tensor operations
    test_concurrent_tensor_operations(&pool, &config, &mut metrics)?;

    // Test 3: Memory pressure handling
    test_memory_pressure_tensor_handling(&pool, &mut metrics)?;

    // Test 4: Memory leak detection
    test_memory_leak_detection(&pool, &mut metrics)?;

    // Validate overall metrics
    validate_integration_metrics(&metrics)?;

    println!("✅ Memory pool tensor integration: {} tensors created, {:.2}% efficiency",
             metrics.total_tensors_created, metrics.memory_efficiency * 100.0);

    Ok(())
}

fn test_basic_tensor_lifecycle(pool: &HybridMemoryPool, metrics: &mut TestMetrics) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    let start_time = Instant::now();

    // Create tensors of various sizes and types
    let tensor_configs = vec![
        (vec![128, 128], BitNetDType::Float32),
        (vec![256, 256], BitNetDType::Float16),
        (vec![512, 512], BitNetDType::Int8),
        (vec![64, 64, 64], BitNetDType::Float32),
        (vec![32, 32, 32, 32], BitNetDType::Float16),
    ];

    let mut tensors = Vec::new();

    for (shape, dtype) in tensor_configs {
        let metadata = TensorMetadata {
            shape: shape.clone(),
            dtype: dtype.clone(),
            device: device.clone(),
            requires_grad: false,
        };

        let tensor = BitNetTensor::zeros_with_pool(shape, dtype, device.clone(), pool.clone())?;
        tensors.push(tensor);
        metrics.total_tensors_created += 1;
        metrics.successful_operations += 1;
    }

    // Perform operations on tensors
    for tensor in &tensors {
        // Test tensor metadata access
        let shape = tensor.shape();
        let dtype = tensor.dtype();
        let device = tensor.device();

        assert!(!shape.is_empty());
        metrics.successful_operations += 1;
    }

    // Calculate allocation time
    let allocation_time = start_time.elapsed() / tensors.len() as u32;
    metrics.average_allocation_time = allocation_time;

    // Memory cleanup happens automatically via Drop
    drop(tensors);

    Ok(())
}

fn test_concurrent_tensor_operations(
    pool: &HybridMemoryPool,
    config: &IntegrationTestConfig,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let success_counter = Arc::new(AtomicUsize::new(0));
    let failure_counter = Arc::new(AtomicUsize::new(0));
    let pool = Arc::new(pool.clone());

    let mut handles = Vec::new();

    for thread_id in 0..config.concurrent_threads {
        let pool_clone = pool.clone();
        let success_counter_clone = success_counter.clone();
        let failure_counter_clone = failure_counter.clone();
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            let device = auto_select_device();
            let mut local_success = 0;
            let mut local_failure = 0;

            for _ in 0..config_clone.operations_per_thread {
                let size = rand::random::<usize>() %
                    (config_clone.tensor_size_range.1 - config_clone.tensor_size_range.0) +
                    config_clone.tensor_size_range.0;

                let shape = vec![size / 4, 4]; // Simple 2D tensor

                match BitNetTensor::zeros_with_pool(
                    shape,
                    BitNetDType::Float32,
                    device.clone(),
                    pool_clone.clone()
                ) {
                    Ok(tensor) => {
                        // Perform some operations
                        let _ = tensor.shape();
                        local_success += 1;
                    }
                    Err(_) => {
                        local_failure += 1;
                    }
                }
            }

            success_counter_clone.fetch_add(local_success, Ordering::Relaxed);
            failure_counter_clone.fetch_add(local_failure, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().map_err(|_| "Thread panicked")?;
    }

    let total_success = success_counter.load(Ordering::Relaxed);
    let total_failure = failure_counter.load(Ordering::Relaxed);

    metrics.successful_operations += total_success;
    metrics.failed_operations += total_failure;
    metrics.total_tensors_created += total_success;

    // Validate concurrent operations
    assert!(total_success > 0, "No successful concurrent operations");
    let success_rate = total_success as f64 / (total_success + total_failure) as f64;
    assert!(success_rate > 0.95, "Success rate too low: {:.2}%", success_rate * 100.0);

    Ok(())
}

fn test_memory_pressure_tensor_handling(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    let mut tensors = Vec::new();

    // Create increasingly large tensors until we hit memory pressure
    let mut tensor_size = 1024;
    let max_iterations = 100;

    for i in 0..max_iterations {
        let shape = vec![tensor_size, tensor_size];

        match BitNetTensor::zeros_with_pool(
            shape,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        ) {
            Ok(tensor) => {
                tensors.push(tensor);
                metrics.total_tensors_created += 1;
                metrics.successful_operations += 1;
                tensor_size = (tensor_size as f64 * 1.1) as usize; // Increase by 10%
            }
            Err(_) => {
                // Hit memory pressure - this is expected
                metrics.failed_operations += 1;
                break;
            }
        }

        // Check memory metrics
        let pool_metrics = pool.get_metrics();
        if pool_metrics.memory_pressure_level() > MemoryPressureLevel::Moderate {
            break;
        }
    }

    // Clean up tensors
    tensors.clear();

    // Validate that we can create tensors again after cleanup
    let recovery_tensor = BitNetTensor::zeros_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        device,
        pool.clone()
    )?;

    metrics.successful_operations += 1;

    Ok(())
}

fn test_memory_leak_detection(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let initial_metrics = pool.get_metrics();
    let device = auto_select_device();

    // Create and drop tensors in a scope
    {
        let mut tensors = Vec::new();
        for i in 0..10 {
            let tensor = BitNetTensor::zeros_with_pool(
                vec![128, 128],
                BitNetDType::Float32,
                device.clone(),
                pool.clone()
            )?;
            tensors.push(tensor);
            metrics.total_tensors_created += 1;
        }

        // Tensors are dropped here
    }

    // Force cleanup
    thread::sleep(Duration::from_millis(100));

    let final_metrics = pool.get_metrics();

    // Memory usage should be similar to initial state (allowing for some overhead)
    let memory_growth = final_metrics.total_allocated.saturating_sub(initial_metrics.total_allocated);
    let acceptable_growth = initial_metrics.total_allocated / 10; // 10% growth is acceptable

    assert!(
        memory_growth <= acceptable_growth,
        "Potential memory leak detected: {} bytes growth",
        memory_growth
    );

    metrics.successful_operations += 1;

    Ok(())
}

// =============================================================================
// Device Abstraction Integration Tests
// =============================================================================

#[test]
fn test_device_abstraction_tensor_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing device abstraction tensor integration");

    let pool = create_test_memory_pool()?;
    let mut metrics = TestMetrics::default();

    // Test device-aware tensor creation
    test_device_aware_tensor_creation(&pool, &mut metrics)?;

    // Test tensor device migration
    test_tensor_device_migration(&pool, &mut metrics)?;

    // Test multi-device tensor operations
    test_multi_device_tensor_operations(&pool, &mut metrics)?;

    println!("✅ Device abstraction integration: {} migrations performed",
             metrics.device_migrations);

    Ok(())
}

fn test_device_aware_tensor_creation(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    // Test CPU device tensor creation
    let cpu_device = get_cpu_device();
    let cpu_tensor = BitNetTensor::zeros_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    assert_eq!(cpu_tensor.device(), &cpu_device);
    metrics.total_tensors_created += 1;
    metrics.successful_operations += 1;

    // Test Metal device tensor creation (if available)
    if is_metal_available() {
        let metal_device = get_metal_device()?;
        let metal_tensor = BitNetTensor::zeros_with_pool(
            vec![256, 256],
            BitNetDType::Float32,
            metal_device.clone(),
            pool.clone()
        )?;

        assert_eq!(metal_tensor.device(), &metal_device);
        metrics.total_tensors_created += 1;
        metrics.successful_operations += 1;
    }

    // Test auto-selected device
    let auto_device = auto_select_device();
    let auto_tensor = BitNetTensor::zeros_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        auto_device.clone(),
        pool.clone()
    )?;

    assert_eq!(auto_tensor.device(), &auto_device);
    metrics.total_tensors_created += 1;
    metrics.successful_operations += 1;

    Ok(())
}

fn test_tensor_device_migration(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let cpu_device = get_cpu_device();

    // Create tensor on CPU
    let cpu_tensor = BitNetTensor::zeros_with_pool(
        vec![128, 128],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    metrics.total_tensors_created += 1;

    // Test migration to Metal (if available)
    if is_metal_available() {
        let metal_device = get_metal_device()?;

        // Migrate to Metal
        let metal_tensor = cpu_tensor.to_device(&metal_device)?;
        assert_eq!(metal_tensor.device(), &metal_device);
        metrics.device_migrations += 1;
        metrics.successful_operations += 1;

        // Migrate back to CPU
        let cpu_tensor_back = metal_tensor.to_device(&cpu_device)?;
        assert_eq!(cpu_tensor_back.device(), &cpu_device);
        metrics.device_migrations += 1;
        metrics.successful_operations += 1;
    }

    Ok(())
}

fn test_multi_device_tensor_operations(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let cpu_device = get_cpu_device();

    // Create multiple tensors on different devices
    let tensor1 = BitNetTensor::zeros_with_pool(
        vec![64, 64],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    let tensor2 = BitNetTensor::ones_with_pool(
        vec![64, 64],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    metrics.total_tensors_created += 2;

    // Test operations between tensors on same device
    let result = tensor1.add(&tensor2)?;
    assert_eq!(result.device(), &cpu_device);
    metrics.successful_operations += 1;

    // Test with Metal device if available
    if is_metal_available() {
        let metal_device = get_metal_device()?;

        let metal_tensor1 = tensor1.to_device(&metal_device)?;
        let metal_tensor2 = tensor2.to_device(&metal_device)?;

        let metal_result = metal_tensor1.add(&metal_tensor2)?;
        assert_eq!(metal_result.device(), &metal_device);

        metrics.device_migrations += 2;
        metrics.successful_operations += 1;
    }

    Ok(())
}

// =============================================================================
// Performance and Efficiency Tests
// =============================================================================

#[test]
fn test_performance_and_efficiency_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing performance and efficiency validation");

    let pool = create_test_memory_pool()?;
    let mut metrics = TestMetrics::default();

    // Test allocation performance
    test_allocation_performance(&pool, &mut metrics)?;

    // Test memory efficiency
    test_memory_efficiency(&pool, &mut metrics)?;

    // Test operation performance
    test_operation_performance(&pool, &mut metrics)?;

    println!("✅ Performance validation: {:.2}μs avg allocation, {:.1}% memory efficiency",
             metrics.average_allocation_time.as_micros(),
             metrics.memory_efficiency * 100.0);

    Ok(())
}

fn test_allocation_performance(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    let iterations = 1000;
    let start_time = Instant::now();

    let mut tensors = Vec::new();

    for _ in 0..iterations {
        let tensor = BitNetTensor::zeros_with_pool(
            vec![128, 128],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;
        tensors.push(tensor);
    }

    let total_time = start_time.elapsed();
    let avg_time = total_time / iterations;

    metrics.average_allocation_time = avg_time;
    metrics.total_tensors_created += iterations as usize;

    // Validate performance target: <100μs per allocation
    assert!(
        avg_time.as_micros() < 100,
        "Allocation too slow: {}μs average",
        avg_time.as_micros()
    );

    Ok(())
}

fn test_memory_efficiency(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    let initial_metrics = pool.get_metrics();

    // Create a known amount of tensor data
    let tensor_count = 100;
    let tensor_size = 256 * 256 * 4; // 256x256 f32 = 256KB each
    let expected_data_size = tensor_count * tensor_size;

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let tensor = BitNetTensor::zeros_with_pool(
            vec![256, 256],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;
        tensors.push(tensor);
    }

    let final_metrics = pool.get_metrics();
    let actual_allocated = final_metrics.total_allocated - initial_metrics.total_allocated;

    // Calculate efficiency (data size / total allocated)
    let efficiency = expected_data_size as f64 / actual_allocated as f64;
    metrics.memory_efficiency = efficiency;

    // Validate efficiency: >90% (allow 10% overhead)
    assert!(
        efficiency > 0.90,
        "Memory efficiency too low: {:.1}%",
        efficiency * 100.0
    );

    Ok(())
}

fn test_operation_performance(
    pool: &HybridMemoryPool,
    metrics: &mut TestMetrics
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();

    let tensor1 = BitNetTensor::zeros_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor2 = BitNetTensor::ones_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let iterations = 100;
    let start_time = Instant::now();

    for _ in 0..iterations {
        let _result = tensor1.add(&tensor2)?;
        metrics.successful_operations += 1;
    }

    let total_time = start_time.elapsed();
    let avg_time = total_time / iterations;

    // Validate operation performance: <1ms per operation for 512x512
    assert!(
        avg_time.as_millis() < 1,
        "Operations too slow: {}ms average",
        avg_time.as_millis()
    );

    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

fn create_test_memory_pool() -> Result<HybridMemoryPool, Box<dyn std::error::Error>> {
    let config = MemoryPoolConfig {
        small_block_size: 64 * 1024,      // 64KB
        large_block_threshold: 1024 * 1024, // 1MB
        initial_pool_size: 16 * 1024 * 1024, // 16MB
        max_pool_size: 256 * 1024 * 1024,    // 256MB
        tracking: TrackingConfig {
            level: TrackingLevel::Detailed,
            enable_stack_traces: false,
            enable_metrics: true,
        },
    };

    HybridMemoryPool::new_with_config(config).map_err(Into::into)
}

fn validate_integration_metrics(metrics: &TestMetrics) -> Result<(), Box<dyn std::error::Error>> {
    // Validate basic metrics
    assert!(metrics.total_tensors_created > 0, "No tensors created");
    assert!(metrics.successful_operations > 0, "No successful operations");

    // Validate success rate
    let total_ops = metrics.successful_operations + metrics.failed_operations;
    if total_ops > 0 {
        let success_rate = metrics.successful_operations as f64 / total_ops as f64;
        assert!(
            success_rate > 0.95,
            "Success rate too low: {:.2}%",
            success_rate * 100.0
        );
    }

    // Validate performance targets
    assert!(
        metrics.average_allocation_time.as_micros() < 100,
        "Allocation time too high: {}μs",
        metrics.average_allocation_time.as_micros()
    );

    Ok(())
}

// Add rand dependency for random number generation
extern crate rand;

#[cfg(test)]
mod integration_tests_tensor {
    use super::*;

    #[test]
    fn run_all_integration_tests() -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting comprehensive tensor integration tests...");

        test_comprehensive_memory_pool_tensor_integration()?;
        test_device_abstraction_tensor_integration()?;
        test_performance_and_efficiency_validation()?;

        println!("🎉 All integration tests passed!");
        Ok(())
    }
}
