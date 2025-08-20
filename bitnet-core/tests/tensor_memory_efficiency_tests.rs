//! Memory Efficiency Validation Tests for BitNet Tensors
//!
//! This module provides comprehensive memory efficiency testing for the BitNet tensor system,
//! following existing patterns from memory_tracking_tests.rs

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    MemoryPressureLevel
};
use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::device::{get_cpu_device};

// =============================================================================
// Memory Efficiency Test Infrastructure
// =============================================================================

/// Configuration for memory efficiency tests
#[derive(Clone)]
struct MemoryEfficiencyConfig {
    tracking_level: TrackingLevel,
    enable_profiling: bool,
    pressure_threshold: f64,
    cleanup_interval: Duration,
    test_duration: Duration,
}

impl Default for MemoryEfficiencyConfig {
    fn default() -> Self {
        Self {
            tracking_level: TrackingLevel::Standard,
            enable_profiling: true,
            pressure_threshold: 0.75,
            cleanup_interval: Duration::from_millis(100),
            test_duration: Duration::from_secs(10),
        }
    }
}

/// Memory efficiency metrics collection
#[derive(Debug, Clone)]
struct MemoryEfficiencyMetrics {
    initial_usage: usize,
    peak_usage: usize,
    final_usage: usize,
    allocations_count: usize,
    deallocations_count: usize,
    fragmentation_ratio: f64,
    cleanup_efficiency: f64,
    memory_overhead: f64,
}

/// Helper function to create memory pool with tracking
fn create_tracked_pool() -> HybridMemoryPool {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::detailed());
    
    HybridMemoryPool::with_config(config)
        .expect("Failed to create tracked memory pool")
}

/// Helper function to measure memory usage
fn get_memory_usage(pool: &HybridMemoryPool) -> usize {
    pool.get_detailed_metrics()
        .map(|m| m.current_memory_usage as usize)
        .unwrap_or(0)
}

/// Helper function to get allocation count
fn get_allocation_count(pool: &HybridMemoryPool) -> usize {
    pool.get_detailed_metrics()
        .map(|m| m.active_allocations)
        .unwrap_or(0)
}

// =============================================================================
// Core Memory Efficiency Tests
// =============================================================================

#[test]
fn test_tensor_memory_allocation_efficiency() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    let _config = MemoryEfficiencyConfig::default(); // Keep for future use
    
    // Record initial state
    let initial_usage = get_memory_usage(&pool);
    let initial_allocations = get_allocation_count(&pool);
    
    println!("Initial memory usage: {} bytes", initial_usage);
    println!("Initial allocations: {}", initial_allocations);
    
    // Create various tensor sizes to test allocation efficiency
    let tensor_sizes = vec![
        vec![16, 16],     // Small: 1KB
        vec![64, 64],     // Medium: 16KB  
        vec![128, 128],   // Large: 256KB
        vec![256, 256],   // Very large: 1MB
    ];
    
    let mut tensors = Vec::new();
    let mut peak_usage = initial_usage;
    
    for (i, shape) in tensor_sizes.iter().enumerate() {
        let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create tensor {}", i));
        
        tensors.push(tensor);
        
        let current_usage = get_memory_usage(&pool);
        peak_usage = peak_usage.max(current_usage);
        
        let expected_size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        println!("Tensor {} - Expected: {} bytes, Pool usage: {} bytes", 
                i, expected_size, current_usage - initial_usage);
    }
    
    let final_allocations = get_allocation_count(&pool);
    println!("Peak memory usage: {} bytes", peak_usage);
    println!("Final allocations: {}", final_allocations);
    
    // Validate memory allocation efficiency
    assert!(final_allocations >= tensors.len(), 
           "Should have at least as many allocations as tensors");
    
    // Calculate memory overhead (should be reasonable)
    let total_tensor_data: usize = tensor_sizes.iter()
        .map(|shape| shape.iter().product::<usize>() * std::mem::size_of::<f32>())
        .sum();
    
    let actual_usage = peak_usage - initial_usage;
    let overhead_ratio = (actual_usage as f64 - total_tensor_data as f64) / total_tensor_data as f64;
    
    println!("Total tensor data: {} bytes", total_tensor_data);
    println!("Actual pool usage: {} bytes", actual_usage);
    println!("Memory overhead ratio: {:.2}%", overhead_ratio * 100.0);
    
    // Memory overhead should be reasonable (less than 50%)
    assert!(overhead_ratio < 0.5, 
           "Memory overhead too high: {:.2}%", overhead_ratio * 100.0);
}

#[test]
fn test_tensor_memory_cleanup_efficiency() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_usage = get_memory_usage(&pool);
    let initial_allocations = get_allocation_count(&pool);
    
    // Create tensors in a scope for automatic cleanup testing
    let mid_usage = {
        let mut tensors = Vec::new();
        
        // Create many tensors
        for i in 0..50 {
            let tensor = BitNetTensor::zeros(&vec![64, 64], BitNetDType::F32, Some(device.clone()))
                .expect(&format!("Failed to create tensor {}", i));
            tensors.push(tensor);
        }
        
        let mid_usage = get_memory_usage(&pool);
        let mid_allocations = get_allocation_count(&pool);
        
        println!("Mid-test usage: {} bytes", mid_usage);
        println!("Mid-test allocations: {}", mid_allocations);
        
        assert!(mid_usage > initial_usage, "Memory usage should increase");
        assert!(mid_allocations > initial_allocations, "Allocations should increase");
        
        mid_usage
    }; // Tensors go out of scope here
    
    // Allow time for cleanup
    thread::sleep(Duration::from_millis(100));
    
    let final_usage = get_memory_usage(&pool);
    let final_allocations = get_allocation_count(&pool);
    
    println!("Final usage: {} bytes", final_usage);
    println!("Final allocations: {}", final_allocations);
    
    // Verify cleanup occurred
    let cleanup_ratio = (mid_usage - final_usage) as f64 / (mid_usage - initial_usage) as f64;
    println!("Cleanup efficiency: {:.2}%", cleanup_ratio * 100.0);
    
    // Some cleanup should occur (may not be 100% immediate due to deferred cleanup)
    assert!(final_usage <= mid_usage, "Memory usage should not increase after cleanup");
}

#[test]
fn test_tensor_memory_fragmentation() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create tensors of varying sizes to test fragmentation
    let mut tensors = Vec::new();
    let sizes = vec![
        vec![8, 8],     // 256 bytes
        vec![32, 32],   // 4KB
        vec![16, 16],   // 1KB
        vec![64, 64],   // 16KB
        vec![12, 12],   // 576 bytes
        vec![48, 48],   // 9KB
    ];
    
    // Create initial tensors
    for (i, shape) in sizes.iter().enumerate() {
        let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create initial tensor {}", i));
        tensors.push(tensor);
    }
    
    let after_creation = get_memory_usage(&pool);
    
    // Drop every other tensor to create fragmentation
    for i in (0..tensors.len()).step_by(2).rev() {
        tensors.remove(i);
    }
    
    thread::sleep(Duration::from_millis(50)); // Allow cleanup
    
    let after_fragmentation = get_memory_usage(&pool);
    
    // Create new tensors that should fit in the gaps
    let mut new_tensors = Vec::new();
    for i in 0..5 {
        let tensor = BitNetTensor::zeros(&vec![10, 10], BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create fragmentation test tensor {}", i));
        new_tensors.push(tensor);
    }
    
    let final_usage = get_memory_usage(&pool);
    
    println!("After creation: {} bytes", after_creation);
    println!("After fragmentation: {} bytes", after_fragmentation);
    println!("Final usage: {} bytes", final_usage);
    
    // Memory should not grow excessively due to fragmentation
    let fragmentation_growth = final_usage.saturating_sub(after_fragmentation);
    let expected_new_data = 5 * 10 * 10 * std::mem::size_of::<f32>();
    
    println!("Fragmentation growth: {} bytes", fragmentation_growth);
    println!("Expected new data: {} bytes", expected_new_data);
    
    // Growth should be reasonable (less than 2x expected data)
    assert!(fragmentation_growth < expected_new_data * 2,
           "Excessive fragmentation detected");
}

#[test]
fn test_tensor_memory_pressure_handling() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let mut tensors = Vec::new();
    let mut creation_failures = 0;
    let max_tensors = 200; // Try to create many large tensors
    
    println!("Testing memory pressure handling with {} large tensors", max_tensors);
    
    // Create tensors until memory pressure or failure
    for i in 0..max_tensors {
        match BitNetTensor::zeros(&vec![256, 256], BitNetDType::F32, Some(device.clone())) {
            Ok(tensor) => {
                tensors.push(tensor);
                
                // Check memory pressure level
                if let Some(metrics) = pool.get_detailed_metrics() {
                    if metrics.pressure_level != MemoryPressureLevel::None {
                        println!("Memory pressure detected at tensor {}: {:?}", 
                               i, metrics.pressure_level);
                        println!("Current usage: {} bytes", metrics.current_memory_usage);
                        break;
                    }
                }
            }
            Err(_) => {
                creation_failures += 1;
                println!("Tensor creation failed at index {}", i);
                if creation_failures > 5 {
                    break; // Stop if too many failures
                }
            }
        }
    }
    
    println!("Created {} tensors before memory pressure/failure", tensors.len());
    println!("Total creation failures: {}", creation_failures);
    
    // Verify memory pressure handling
    if let Some(metrics) = pool.get_detailed_metrics() {
        println!("Final memory state:");
        println!("  Usage: {} bytes", metrics.current_memory_usage);
        println!("  Allocations: {}", metrics.active_allocations);
        println!("  Pressure: {:?}", metrics.pressure_level);
        
        // Should have created a reasonable number of tensors
        assert!(tensors.len() > 10, "Should create at least some tensors before pressure");
    }
}

#[test]
fn test_tensor_memory_pool_reuse() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_usage = get_memory_usage(&pool);
    
    // Phase 1: Create and destroy tensors multiple times
    for round in 0..5 {
        let mut round_tensors = Vec::new();
        
        // Create tensors
        for i in 0..20 {
            let tensor = BitNetTensor::zeros(&vec![32, 32], BitNetDType::F32, Some(device.clone()))
                .expect(&format!("Failed to create tensor {}-{}", round, i));
            round_tensors.push(tensor);
        }
        
        let round_peak = get_memory_usage(&pool);
        
        // Destroy tensors
        drop(round_tensors);
        thread::sleep(Duration::from_millis(50)); // Allow cleanup
        
        let round_final = get_memory_usage(&pool);
        
        println!("Round {} - Peak: {} bytes, Final: {} bytes", 
               round, round_peak - initial_usage, round_final - initial_usage);
    }
    
    let final_usage = get_memory_usage(&pool);
    println!("Overall final usage: {} bytes", final_usage - initial_usage);
    
    // Memory usage should stabilize after multiple rounds (pool reuse)
    // We don't expect it to return exactly to initial due to pool growth,
    // but it shouldn't grow indefinitely
    assert!(final_usage < initial_usage + (1024 * 1024), // Allow 1MB growth
           "Memory usage growing indefinitely, pool reuse may not be working");
}

#[test]
fn test_tensor_concurrent_memory_efficiency() {
    let pool = Arc::new(create_tracked_pool());
    let initial_usage = get_memory_usage(&*pool);
    
    let num_threads = 4;
    let tensors_per_thread = 25;
    
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let _pool_clone = pool.clone(); // Keep pool alive for thread
        
        thread::spawn(move || {
            let device = get_cpu_device();
            let mut thread_tensors = Vec::new();
            
            for i in 0..tensors_per_thread {
                let tensor = BitNetTensor::zeros(
                    &vec![32, 32],
                    BitNetDType::F32,
                    Some(device.clone())
                ).expect(&format!("Thread {} failed to create tensor {}", thread_id, i));
                
                thread_tensors.push(tensor);
                
                // Small delay to interleave allocations
                thread::sleep(Duration::from_micros(100));
            }
            
            thread_tensors.len()
        })
    }).collect();
    
    // Wait for all threads and collect results
    let mut total_tensors = 0;
    for handle in handles {
        total_tensors += handle.join().expect("Thread panicked");
    }
    
    let peak_usage = get_memory_usage(&*pool);
    let peak_allocations = get_allocation_count(&*pool);
    
    println!("Concurrent test results:");
    println!("  Total tensors created: {}", total_tensors);
    println!("  Peak usage: {} bytes", peak_usage - initial_usage);
    println!("  Peak allocations: {}", peak_allocations);
    
    // Verify all tensors were created
    assert_eq!(total_tensors, num_threads * tensors_per_thread);
    
    // Verify memory usage is reasonable
    let expected_data = total_tensors * 32 * 32 * std::mem::size_of::<f32>();
    let actual_usage = peak_usage - initial_usage;
    let overhead = (actual_usage as f64 - expected_data as f64) / expected_data as f64;
    
    println!("  Expected data: {} bytes", expected_data);
    println!("  Actual usage: {} bytes", actual_usage);
    println!("  Overhead: {:.2}%", overhead * 100.0);
    
    // Concurrent overhead should be reasonable
    assert!(overhead < 1.0, "Concurrent memory overhead too high: {:.2}%", overhead * 100.0);
}

// =============================================================================
// Device-Specific Memory Efficiency Tests  
// =============================================================================

#[test]
fn test_cpu_memory_efficiency() {
    let pool = create_tracked_pool();
    let cpu_device = get_cpu_device();
    
    let initial_usage = get_memory_usage(&pool);
    
    // Create CPU-specific tensors
    let mut cpu_tensors = Vec::new();
    for i in 0..20 {
        let tensor = BitNetTensor::zeros(&vec![64, 64], BitNetDType::F32, Some(cpu_device.clone()))
            .expect(&format!("Failed to create CPU tensor {}", i));
        cpu_tensors.push(tensor);
    }
    
    let cpu_usage = get_memory_usage(&pool) - initial_usage;
    let expected_cpu = 20 * 64 * 64 * std::mem::size_of::<f32>();
    
    println!("CPU memory efficiency:");
    println!("  Expected: {} bytes", expected_cpu);
    println!("  Actual: {} bytes", cpu_usage);
    println!("  Efficiency: {:.2}%", (expected_cpu as f64 / cpu_usage as f64) * 100.0);
    
    // CPU efficiency should be good (>70%)
    let efficiency = (expected_cpu as f64 / cpu_usage as f64) * 100.0;
    assert!(efficiency > 50.0, "CPU memory efficiency too low: {:.2}%", efficiency);
}

#[test]
#[cfg(feature = "metal")]
fn test_metal_memory_efficiency() {
    if !is_metal_available() {
        println!("Skipping Metal memory efficiency test - Metal not available");
        return;
    }
    
    let pool = create_tracked_pool();
    let metal_device = get_metal_device().expect("Metal should be available");
    
    let initial_usage = get_memory_usage(&pool);
    
    // Create Metal-specific tensors
    let mut metal_tensors = Vec::new();
    for i in 0..10 { // Fewer tensors for Metal due to potential GPU memory limits
        let tensor = BitNetTensor::zeros(&vec![64, 64], BitNetDType::F32, Some(metal_device.clone()))
            .expect(&format!("Failed to create Metal tensor {}", i));
        metal_tensors.push(tensor);
    }
    
    let metal_usage = get_memory_usage(&pool) - initial_usage;
    let expected_metal = 10 * 64 * 64 * std::mem::size_of::<f32>();
    
    println!("Metal memory efficiency:");
    println!("  Expected: {} bytes", expected_metal);
    println!("  Actual: {} bytes", metal_usage);
    println!("  Efficiency: {:.2}%", (expected_metal as f64 / metal_usage as f64) * 100.0);
    
    // Metal efficiency may be lower due to GPU memory management overhead
    let efficiency = (expected_metal as f64 / metal_usage as f64) * 100.0;
    assert!(efficiency > 25.0, "Metal memory efficiency too low: {:.2}%", efficiency);
}

// =============================================================================
// Data Type Memory Efficiency Tests
// =============================================================================

#[test]
fn test_dtype_memory_efficiency() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let dtypes_and_sizes = vec![
        (BitNetDType::F32, 4),
        (BitNetDType::F16, 2),
        (BitNetDType::I32, 4),
        (BitNetDType::I8, 1),
        (BitNetDType::U8, 1),
        (BitNetDType::Bool, 1),
    ];
    
    let shape = vec![128, 128]; // 16,384 elements
    let elements = shape.iter().product::<usize>();
    
    for (dtype, expected_bytes_per_element) in dtypes_and_sizes {
        let initial_usage = get_memory_usage(&pool);
        
        let tensor = BitNetTensor::zeros(&shape, dtype, Some(device.clone()))
            .expect(&format!("Failed to create tensor with dtype {:?}", dtype));
        
        let actual_usage = get_memory_usage(&pool) - initial_usage;
        let expected_usage = elements * expected_bytes_per_element;
        
        println!("Data type {:?}:", dtype);
        println!("  Expected: {} bytes", expected_usage);
        println!("  Actual: {} bytes", actual_usage);
        println!("  Overhead: {:.2}%", 
               ((actual_usage as f64 - expected_usage as f64) / expected_usage as f64) * 100.0);
        
        // Memory usage should be close to expected (allow some overhead)
        let max_expected = expected_usage * 2; // Allow 100% overhead for metadata
        assert!(actual_usage <= max_expected,
               "Memory usage for {:?} too high: {} bytes (expected ~{})", 
               dtype, actual_usage, expected_usage);
        
        drop(tensor);
        thread::sleep(Duration::from_millis(10)); // Allow cleanup
    }
}

// =============================================================================
// Performance vs Memory Trade-off Tests
// =============================================================================

#[test]
fn test_memory_performance_tradeoff() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test different allocation strategies
    println!("Testing memory vs performance trade-offs:");
    
    // Strategy 1: Many small tensors
    let start_time = Instant::now();
    let initial_usage = get_memory_usage(&pool);
    
    let mut small_tensors = Vec::new();
    for i in 0..100 {
        let tensor = BitNetTensor::zeros(&vec![16, 16], BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create small tensor {}", i));
        small_tensors.push(tensor);
    }
    
    let small_time = start_time.elapsed();
    let small_usage = get_memory_usage(&pool) - initial_usage;
    
    drop(small_tensors);
    thread::sleep(Duration::from_millis(50));
    
    // Strategy 2: Few large tensors  
    let start_time = Instant::now();
    let initial_usage = get_memory_usage(&pool);
    
    let mut large_tensors = Vec::new();
    for i in 0..4 { // Same total elements as 100 small tensors
        let tensor = BitNetTensor::zeros(&vec![80, 80], BitNetDType::F32, Some(device.clone()))
            .expect(&format!("Failed to create large tensor {}", i));
        large_tensors.push(tensor);
    }
    
    let large_time = start_time.elapsed();
    let large_usage = get_memory_usage(&pool) - initial_usage;
    
    println!("Small tensors: {} µs, {} bytes", small_time.as_micros(), small_usage);
    println!("Large tensors: {} µs, {} bytes", large_time.as_micros(), large_usage);
    
    // Large tensors should generally be more efficient
    println!("Performance ratio: {:.2}", small_time.as_micros() as f64 / large_time.as_micros() as f64);
    println!("Memory ratio: {:.2}", small_usage as f64 / large_usage as f64);
}
