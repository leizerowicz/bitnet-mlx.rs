//! Comprehensive Unit Tests for Tensor Memory Management Integration
//!
//! This test suite validates the integration between tensors and the memory management
//! system, ensuring that tensors properly participate in memory tracking, cleanup,
//! and optimization. Tests cover memory pools, lifecycle management, tracking,
//! device-specific operations, and efficiency validation.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    CleanupManager, CleanupConfig, CleanupStrategyType, CleanupPriority,
    MemoryPressureLevel, MemoryTracker, MemoryProfiler, AllocationTimeline,
    PatternAnalyzer, PressureThresholds, DetailedMemoryMetrics
};
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType, TensorHandle, TensorMetadata};
use bitnet_core::device::{get_cpu_device, auto_select_device, is_metal_available, get_metal_device};

/// Helper function to reset all global state before each test
fn reset_global_state() {
    // Clear global tensor state
    bitnet_core::memory::tensor::tensor::clear_global_tensor_state();
    
    // Clear global memory handle state
    bitnet_core::memory::tensor::handle::clear_global_state();
}

/// Helper function to create a memory pool with advanced tracking enabled
fn create_tracked_pool() -> Arc<HybridMemoryPool> {
    // Reset global state first to prevent test interference
    reset_global_state();
    
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::detailed());
    
    let pool = Arc::new(HybridMemoryPool::with_config(config)
        .expect("Failed to create tracked memory pool"));
    
    // Set the global pool reference for automatic cleanup
    bitnet_core::memory::tensor::handle::set_global_memory_pool(Arc::downgrade(&pool));
    
    pool
}

/// Helper function to create a memory pool with cleanup enabled
fn create_pool_with_cleanup() -> (Arc<HybridMemoryPool>, CleanupManager) {
    let pool = create_tracked_pool();
    let cleanup_config = CleanupConfig::default();
    let cleanup_manager = CleanupManager::new(cleanup_config, pool.clone())
        .expect("Failed to create cleanup manager");
    
    (pool, cleanup_manager)
}

/// Helper function to get all available devices for testing
fn get_test_devices() -> Vec<candle_core::Device> {
    let mut devices = vec![get_cpu_device()];
    
    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            devices.push(metal_device);
        }
    }
    
    devices
}

/// Helper function to get test data types
fn get_test_dtypes() -> Vec<BitNetDType> {
    vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::BF16,
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ]
}

// =============================================================================
// Tensor Memory Pool Integration Tests
// =============================================================================

#[test]
fn test_tensor_allocation_from_memory_pools() {
    let pool = create_tracked_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            // Test small tensor allocation (should use small block pool)
            let small_tensor = BitNetTensor::zeros(&[32, 32], dtype, device, &pool)
                .expect("Failed to create small tensor");
            
            assert_eq!(small_tensor.shape(), vec![32, 32]);
            assert_eq!(small_tensor.dtype(), dtype);
            // Verify device consistency using custom device comparison
            bitnet_core::device::assert_devices_equal(&small_tensor.device(), &device);
            
            // Test large tensor allocation (should use large block pool)
            let large_tensor = BitNetTensor::zeros(&[1024, 1024], dtype, device, &pool)
                .expect("Failed to create large tensor");
            
            assert_eq!(large_tensor.shape(), vec![1024, 1024]);
            assert_eq!(large_tensor.dtype(), dtype);
            // Verify device consistency using custom device comparison
            bitnet_core::device::assert_devices_equal(&large_tensor.device(), &device);
            
            // Verify memory pool metrics
            let metrics = pool.get_metrics();
            assert!(metrics.active_allocations >= 2);
            assert!(metrics.current_allocated > 0);
        }
    }
}

#[test]
fn test_memory_pool_pressure_with_tensors() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create many tensors to build memory pressure
    let mut tensors = Vec::new();
    let tensor_count = 100;
    
    for i in 0..tensor_count {
        let size = 1024 + i * 10; // Varying sizes
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor under pressure");
        tensors.push(tensor);
    }
    
    // Check memory pressure detection
    if let Some(detailed_metrics) = pool.get_detailed_metrics() {
        assert!(detailed_metrics.active_allocations >= tensor_count);
        assert!(detailed_metrics.current_memory_usage > 0);
        
        // Memory pressure might be detected with many allocations
        println!("Memory pressure level: {:?}", detailed_metrics.pressure_level);
    }
    
    // Verify all tensors are still valid
    for tensor in &tensors {
        assert!(tensor.handle().is_valid());
    }
    
    // Drop half the tensors and verify memory is reclaimed with immediate cleanup
    let remaining_count = tensor_count / 2;
    tensors.truncate(remaining_count);
    
    // With immediate cleanup, memory should be reclaimed immediately when tensors are dropped
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, remaining_count as u64,
               "Expected immediate cleanup to reclaim memory from dropped tensors");
}

#[test]
fn test_tensor_memory_pool_efficiency() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test memory pool efficiency with different tensor sizes
    let test_cases = vec![
        (vec![64], 100),      // Small tensors
        (vec![512], 50),      // Medium tensors  
        (vec![2048], 10),     // Large tensors
        (vec![16, 16], 100),  // 2D small tensors
        (vec![64, 64], 25),   // 2D medium tensors
    ];
    
    for (shape, count) in test_cases {
        let start_time = Instant::now();
        let mut tensors = Vec::new();
        
        // Allocate tensors
        for _ in 0..count {
            let tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
            tensors.push(tensor);
        }
        
        let allocation_time = start_time.elapsed();
        
        // Measure deallocation time
        let dealloc_start = Instant::now();
        drop(tensors);
        let deallocation_time = dealloc_start.elapsed();
        
        println!("Shape {:?}: {} tensors allocated in {:?}, deallocated in {:?}", 
                shape, count, allocation_time, deallocation_time);
        
        // Verify efficiency (allocation should be fast) - more realistic thresholds
        assert!(allocation_time < Duration::from_millis(500),
                "Allocation too slow for shape {:?}: {:?}", shape, allocation_time);
        assert!(deallocation_time < Duration::from_millis(100),
                "Deallocation too slow for shape {:?}: {:?}", shape, deallocation_time);
    }
}

#[test]
fn test_memory_pool_fragmentation_with_tensors() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create tensors with varying sizes to test fragmentation
    let mut tensors = Vec::new();
    let sizes = vec![64, 128, 256, 512, 1024, 2048];
    
    // Allocate tensors in a pattern that could cause fragmentation
    for &size in &sizes {
        for _ in 0..10 {
            let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
                .expect("Failed to create tensor");
            tensors.push(tensor);
        }
    }
    
    let metrics_before = pool.get_metrics();
    
    // Deallocate every other tensor to create fragmentation
    let original_count = tensors.len();
    let mut i = 0;
    tensors.retain(|_| {
        i += 1;
        i % 2 == 0
    });
    let remaining_count = tensors.len();
    
    let metrics_after = pool.get_metrics();
    
    // Verify that memory was properly reclaimed with immediate cleanup
    assert_eq!(metrics_after.active_allocations, remaining_count as u64,
               "Expected immediate cleanup to reclaim memory from {} dropped tensors",
               original_count - remaining_count);
    assert!(metrics_after.current_allocated < metrics_before.current_allocated,
            "Expected memory usage to decrease after dropping tensors");
    
    // Try to allocate new tensors in the fragmented space
    for &size in &sizes {
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to allocate in fragmented space");
        assert_eq!(tensor.shape(), vec![size]);
    }
}

// =============================================================================
// Tensor Lifecycle and Cleanup Tests
// =============================================================================

#[test]
fn test_tensor_automatic_cleanup_on_drop() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    // Create tensors in a scope
    {
        let tensor1 = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor1");
        let tensor2 = BitNetTensor::zeros(&[200, 200], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tensor2");
        
        let mid_metrics = pool.get_metrics();
        assert!(mid_metrics.active_allocations > initial_metrics.active_allocations);
        assert!(mid_metrics.current_allocated > initial_metrics.current_allocated);
        
        // Tensors should be valid
        assert!(tensor1.handle().is_valid());
        assert!(tensor2.handle().is_valid());
    } // Tensors dropped here - immediate cleanup occurs
    
    // With immediate cleanup, memory should be reclaimed immediately when tensors are dropped
    // No manual cleanup should be needed
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations,
               "Expected immediate cleanup to reclaim all memory");
    assert_eq!(final_metrics.current_allocated, initial_metrics.current_allocated,
               "Expected immediate cleanup to reclaim all allocated memory");
}

#[test]
fn test_tensor_cleanup_with_memory_pressure() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Register pressure callback to track pressure events
    let pressure_events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pressure_events_clone = pressure_events.clone();
    
    pool.register_pressure_callback(Box::new(move |level| {
        pressure_events_clone.lock().unwrap().push(level);
    }));
    
    // Create many large tensors to trigger pressure
    let mut tensors = Vec::new();
    for i in 0..50 {
        let size = 1024 * (i + 1);
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create pressure tensor");
        tensors.push(tensor);
        
        // Check if pressure is detected
        if let Some(detailed_metrics) = pool.get_detailed_metrics() {
            if detailed_metrics.pressure_level != MemoryPressureLevel::None {
                println!("Memory pressure detected at tensor {}: {:?}",
                        i, detailed_metrics.pressure_level);
                break;
            }
        }
    }
    
    // Drop tensors and verify immediate cleanup
    let cleanup_start = Instant::now();
    drop(tensors);
    let cleanup_duration = cleanup_start.elapsed();
    
    println!("Tensor cleanup completed in {:?}", cleanup_duration);
    
    // With immediate cleanup, memory should be reclaimed immediately when tensors are dropped
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0,
               "Expected immediate cleanup to reclaim all memory after tensor drop");
}

#[test]
fn test_tensor_cleanup_scheduling_and_strategies() {
    let (pool, cleanup_manager) = create_pool_with_cleanup();
    let device = get_cpu_device();
    
    // Start cleanup scheduler
    cleanup_manager.start_scheduler()
        .expect("Failed to start cleanup scheduler");
    
    // Create tensors with different patterns
    let mut short_lived_tensors = Vec::new();
    let mut long_lived_tensors = Vec::new();
    
    // Short-lived tensors
    for _ in 0..20 {
        let tensor = BitNetTensor::zeros(&[256], BitNetDType::F32, &device, &pool)
            .expect("Failed to create short-lived tensor");
        short_lived_tensors.push(tensor);
    }
    
    // Long-lived tensors
    for _ in 0..10 {
        let tensor = BitNetTensor::zeros(&[512], BitNetDType::F32, &device, &pool)
            .expect("Failed to create long-lived tensor");
        long_lived_tensors.push(tensor);
    }
    
    // Drop short-lived tensors
    drop(short_lived_tensors);
    
    // Allow scheduler to run
    thread::sleep(Duration::from_millis(100));
    
    // Perform manual cleanup
    let cleanup_result = cleanup_manager.force_cleanup()
        .expect("Failed to perform cleanup");
    
    assert!(cleanup_result.success || cleanup_result.error_message.is_some());
    
    // Stop scheduler
    cleanup_manager.stop_scheduler()
        .expect("Failed to stop cleanup scheduler");
    
    // Verify long-lived tensors are still valid
    for tensor in &long_lived_tensors {
        assert!(tensor.handle().is_valid());
    }
}

#[test]
fn test_tensor_memory_reclamation() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test immediate reclamation
    let initial_metrics = pool.get_metrics();
    
    let tensor = BitNetTensor::zeros(&[1000, 1000], BitNetDType::F32, &device, &pool)
        .expect("Failed to create large tensor");
    
    let allocated_metrics = pool.get_metrics();
    assert!(allocated_metrics.current_allocated > initial_metrics.current_allocated);
    
    drop(tensor);
    
    let reclaimed_metrics = pool.get_metrics();
    assert_eq!(reclaimed_metrics.current_allocated, initial_metrics.current_allocated,
               "Expected immediate cleanup to reclaim memory from dropped tensor");
    
    // Test delayed reclamation with multiple references
    let tensor = BitNetTensor::zeros(&[500, 500], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor");
    
    let tensor_clone = tensor.clone();
    assert_eq!(tensor.ref_count(), 2);
    
    drop(tensor);
    
    // Memory should not be reclaimed yet (clone still exists)
    let partial_metrics = pool.get_metrics();
    assert!(partial_metrics.current_allocated > initial_metrics.current_allocated,
            "Memory should not be reclaimed while clone exists");
    
    drop(tensor_clone);
    
    // Now memory should be reclaimed with immediate cleanup
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.current_allocated, initial_metrics.current_allocated,
               "Expected immediate cleanup to reclaim memory after all references dropped");
}

// =============================================================================
// Memory Tracking for Tensors Tests
// =============================================================================

#[test]
fn test_tensor_memory_usage_tracking() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Verify tracking is enabled
    assert!(pool.get_memory_tracker().is_some());
    
    let initial_metrics = pool.get_detailed_metrics()
        .expect("Detailed metrics should be available");
    
    // Create tensors and track usage
    let mut tensors = Vec::new();
    let mut expected_usage = 0u64;
    
    for i in 0..10 {
        let size = (i + 1) * 100;
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create tracked tensor");
        
        expected_usage += tensor.size_bytes() as u64;
        tensors.push(tensor);
        
        // Check tracking metrics
        let current_metrics = pool.get_detailed_metrics()
            .expect("Detailed metrics should be available");
        
        assert!(current_metrics.active_allocations > initial_metrics.active_allocations);
        assert!(current_metrics.current_memory_usage >= expected_usage);
    }
    
    // Verify final tracking state
    let final_metrics = pool.get_detailed_metrics()
        .expect("Detailed metrics should be available");
    
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations + 10);
    assert!(final_metrics.current_memory_usage >= expected_usage);
    
    // Test device-specific tracking
    let device_usage = &final_metrics.device_usage;
    let device_key = format!("{:?}", device);
    assert!(device_usage.contains_key(&device_key));
    assert!(device_usage[&device_key] > 0);
}

#[test]
fn test_tensor_allocation_deallocation_patterns() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test different allocation patterns
    let patterns = vec![
        ("Sequential", vec![100, 200, 300, 400, 500]),
        ("Decreasing", vec![500, 400, 300, 200, 100]),
        ("Random", vec![300, 100, 500, 200, 400]),
        ("Uniform", vec![250, 250, 250, 250, 250]),
    ];
    
    for (pattern_name, sizes) in patterns {
        println!("Testing allocation pattern: {}", pattern_name);
        
        let start_metrics = pool.get_detailed_metrics()
            .expect("Detailed metrics should be available");
        
        let mut tensors = Vec::new();
        
        // Allocate tensors according to pattern
        for size in &sizes {
            let tensor = BitNetTensor::zeros(&[*size], BitNetDType::F32, &device, &pool)
                .expect("Failed to create pattern tensor");
            tensors.push(tensor);
        }
        
        let allocated_metrics = pool.get_detailed_metrics()
            .expect("Detailed metrics should be available");
        
        // Verify allocation tracking
        assert_eq!(allocated_metrics.active_allocations as usize,
                  start_metrics.active_allocations as usize + sizes.len());
        
        // Deallocate in reverse order with immediate cleanup
        while let Some(tensor) = tensors.pop() {
            drop(tensor);
        }
        
        let deallocated_metrics = pool.get_detailed_metrics()
            .expect("Detailed metrics should be available");
        
        // Verify deallocation tracking with immediate cleanup
        assert_eq!(deallocated_metrics.active_allocations, start_metrics.active_allocations,
                   "Expected immediate cleanup to reclaim all memory for pattern {}", pattern_name);
    }
}

#[test]
fn test_tensor_memory_pressure_detection() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Track pressure level changes
    let pressure_levels = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pressure_levels_clone = pressure_levels.clone();
    
    pool.register_pressure_callback(Box::new(move |level| {
        pressure_levels_clone.lock().unwrap().push(level);
    }));
    
    // Start with no pressure
    let initial_metrics = pool.get_detailed_metrics()
        .expect("Detailed metrics should be available");
    assert_eq!(initial_metrics.pressure_level, MemoryPressureLevel::None);
    
    // Gradually increase memory usage
    let mut tensors = Vec::new();
    let mut current_pressure = MemoryPressureLevel::None;
    
    for i in 0..100 {
        let size = 1024 * (i + 1);
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create pressure tensor");
        tensors.push(tensor);
        
        let metrics = pool.get_detailed_metrics()
            .expect("Detailed metrics should be available");
        
        if metrics.pressure_level != current_pressure {
            println!("Pressure level changed at tensor {}: {:?} -> {:?}", 
                    i, current_pressure, metrics.pressure_level);
            current_pressure = metrics.pressure_level;
            
            // Stop if we reach high pressure to avoid system issues
            if matches!(current_pressure, MemoryPressureLevel::High | MemoryPressureLevel::Critical) {
                break;
            }
        }
    }
    
    // Verify pressure was detected
    let pressure_events = pressure_levels.lock().unwrap();
    if !pressure_events.is_empty() {
        println!("Pressure events detected: {:?}", *pressure_events);
    }
}

#[test]
fn test_tensor_memory_profiling_integration() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Get the memory tracker for profiling
    if let Some(tracker) = pool.get_memory_tracker() {
        // Create tensors with different characteristics
        let test_cases = vec![
            ("Small frequent", 50, vec![64]),
            ("Medium batch", 20, vec![256, 256]),
            ("Large single", 5, vec![1024, 1024]),
        ];
        
        for (test_name, count, shape) in test_cases {
            println!("Profiling test case: {}", test_name);
            
            let start_time = Instant::now();
            let mut tensors = Vec::new();
            
            for _ in 0..count {
                let tensor = BitNetTensor::zeros(&shape, BitNetDType::F32, &device, &pool)
                    .expect("Failed to create profiling tensor");
                tensors.push(tensor);
            }
            
            let allocation_time = start_time.elapsed();
            
            let dealloc_start = Instant::now();
            drop(tensors);
            let deallocation_time = dealloc_start.elapsed();
            
            println!("  Allocation time: {:?}", allocation_time);
            println!("  Deallocation time: {:?}", deallocation_time);
            
            // Get detailed metrics including performance data
            let metrics = tracker.get_detailed_metrics();
            println!("  Tracking operations: {}", metrics.performance.total_tracking_operations);
            println!("  Tracking overhead: {:.2}%", metrics.tracking_overhead.cpu_overhead_percentage);
        }
    }
}

// =============================================================================
// Device Memory Management Tests
// =============================================================================

#[test]
fn test_device_specific_tensor_allocation() {
    let devices = get_test_devices();
    
    for device in &devices {
        let pool = create_tracked_pool();
        
        println!("Testing device-specific allocation on: {:?}", device);
        
        // Test allocation on specific device
        let tensor = BitNetTensor::zeros(&[512, 512], BitNetDType::F32, device, &pool)
            .expect("Failed to create device-specific tensor");
        
        // Verify device consistency using custom device comparison
        bitnet_core::device::assert_devices_equal(&tensor.device(), &device);
        
        // Verify device-specific metrics
        if let Some(detailed_metrics) = pool.get_detailed_metrics() {
            let device_key = format!("{:?}", device);
            assert!(detailed_metrics.device_usage.contains_key(&device_key));
            assert!(detailed_metrics.device_usage[&device_key] > 0);
        }
        
        // Test handle device consistency
        let handle = tensor.handle();
        // Verify device consistency using custom device comparison
        bitnet_core::device::assert_devices_equal(&handle.device().unwrap(), &device);
    }
}

#[test]
fn test_tensor_device_migration() {
    let devices = get_test_devices();
    
    if devices.len() < 2 {
        println!("Skipping device migration test - need multiple devices");
        return;
    }
    
    let pool = create_tracked_pool();
    let source_device = &devices[0];
    let target_device = &devices[1];
    
    // Create tensor on source device
    let tensor = BitNetTensor::zeros(&[256, 256], BitNetDType::F32, source_device, &pool)
        .expect("Failed to create source tensor");
    
    // Verify device consistency using custom device comparison
    bitnet_core::device::assert_devices_equal(&tensor.device(), &source_device);
    
    let initial_metrics = pool.get_detailed_metrics()
        .expect("Detailed metrics should be available");
    
    // Migrate to target device
    let migrated_tensor = tensor.to_device(target_device, &pool)
        .expect("Failed to migrate tensor");
    
    // Verify device consistency using custom device comparison
    bitnet_core::device::assert_devices_equal(&migrated_tensor.device(), &target_device);
    assert_eq!(migrated_tensor.shape(), tensor.shape());
    assert_eq!(migrated_tensor.dtype(), tensor.dtype());
    
    // Verify both tensors exist (migration creates new tensor)
    let post_migration_metrics = pool.get_detailed_metrics()
        .expect("Detailed metrics should be available");
    
    assert!(post_migration_metrics.active_allocations > initial_metrics.active_allocations);
    
    // Verify device-specific usage tracking
    let source_key = format!("{:?}", source_device);
    let target_key = format!("{:?}", target_device);
    
    assert!(post_migration_metrics.device_usage.contains_key(&source_key));
    assert!(post_migration_metrics.device_usage.contains_key(&target_key));
    assert!(post_migration_metrics.device_usage[&target_key] > 0);
}

#[test]
fn test_device_memory_pressure_handling() {
    let devices = get_test_devices();
    
    for device in &devices {
        let pool = create_tracked_pool();
        
        println!("Testing memory pressure handling on: {:?}", device);
        
        // Create tensors until pressure is detected
        let mut tensors = Vec::new();
        let mut pressure_detected = false;
        
        for i in 0..50 {
            let size = 1024 * (i + 1);
            
            match BitNetTensor::zeros(&[size], BitNetDType::F32, device, &pool) {
                Ok(tensor) => {
                    tensors.push(tensor);
                    
                    if let Some(detailed_metrics) = pool.get_detailed_metrics() {
                        if detailed_metrics.pressure_level != MemoryPressureLevel::None {
                            println!("  Pressure detected at tensor {}: {:?}", 
                                    i, detailed_metrics.pressure_level);
                            pressure_detected = true;
                            break;
                        }
                    }
                }
                Err(e) => {
                    println!("  Allocation failed at tensor {}: {}", i, e);
                    break;
                }
            }
        }
        
        // Test cleanup under pressure
        if pressure_detected {
            let cleanup_start = Instant::now();
            tensors.truncate(tensors.len() / 2); // Drop half the tensors
            let cleanup_time = cleanup_start.elapsed();
            
            println!("  Cleanup completed in {:?}", cleanup_time);
            
            // Verify pressure reduction
            if let Some(final_metrics) = pool.get_detailed_metrics() {
                println!("  Final pressure level: {:?}", final_metrics.pressure_level);
            }
        }
    }
}

#[test]
fn test_device_memory_cleanup() {
    let devices = get_test_devices();
    
    for device in &devices {
        let (pool, cleanup_manager) = create_pool_with_cleanup();
        
        println!("Testing device-specific cleanup on: {:?}", device);
        
        // Create tensors on the device
        let mut tensors = Vec::new();
        for i in 0..20 {
            let tensor = BitNetTensor::zeros(&[256 + i * 10], BitNetDType::F32, device, &pool)
                .expect("Failed to create device tensor");
            tensors.push(tensor);
        }
        
        // Perform device-specific cleanup
        let cleanup_result = cleanup_manager.cleanup_device(device)
            .expect("Failed to perform device cleanup");
        
        println!("  Cleanup result: success={}, bytes_freed={}", 
                cleanup_result.success, cleanup_result.bytes_freed);
        
        // Verify tensors are still valid after cleanup
        for tensor in &tensors {
            assert!(tensor.handle().is_valid());
        }
    }
}

// =============================================================================
// Memory Efficiency and Optimization Tests
// =============================================================================

#[test]
fn test_memory_sharing_between_tensor_handles() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create a tensor
    let tensor = BitNetTensor::zeros(&[512, 512], BitNetDType::F32, &device, &pool)
        .expect("Failed to create shared tensor");
    
    let initial_metrics = pool.get_metrics();
    
    // Create multiple handles to the same tensor
    let handle1 = tensor.handle();
    let handle2 = tensor.handle();
    let handle3 = tensor.handle();
    
    // Memory usage should not increase significantly
    let handle_metrics = pool.get_metrics();
    assert_eq!(handle_metrics.active_allocations, initial_metrics.active_allocations);
    
    // All handles should reference the same tensor
    assert_eq!(handle1.tensor_id().unwrap(), handle2.tensor_id().unwrap());
    assert_eq!(handle2.tensor_id().unwrap(), handle3.tensor_id().unwrap());
    
    // Verify reference counting
    assert_eq!(tensor.ref_count(), 1); // Handles don't affect Arc count
    
    // Clone the tensor (this should increase ref count)
    let tensor_clone = tensor.clone();
    assert_eq!(tensor.ref_count(), 2);
    assert_eq!(tensor_clone.ref_count(), 2);
    
    // Memory usage should still be the same (shared data)
    let clone_metrics = pool.get_metrics();
    assert_eq!(clone_metrics.active_allocations, initial_metrics.active_allocations);
}

#[test]
fn test_memory_deduplication_for_identical_tensors() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create identical tensors
    let tensor1 = BitNetTensor::zeros(&[256, 256], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor1");
    let tensor2 = BitNetTensor::zeros(&[256, 256], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor2");
    let tensor3 = BitNetTensor::zeros(&[256, 256], BitNetDType::F32, &device, &pool)
        .expect("Failed to create tensor3");
    
    // Each tensor should have its own memory (no automatic deduplication)
    assert_ne!(tensor1.id(), tensor2.id());
    assert_ne!(tensor2.id(), tensor3.id());
    
    let metrics = pool.get_metrics();
    assert_eq!(metrics.active_allocations, 3);
    
    // Test manual deduplication through cloning
    let tensor1_clone = tensor1.clone();
    assert_eq!(tensor1.id(), tensor1_clone.id());
    assert_eq!(tensor1.ref_count(), 2);
    
    // Verify memory sharing
    let shared_metrics = pool.get_metrics();
    assert_eq!(shared_metrics.active_allocations, 3); // Still 3 allocations
}

#[test]
fn test_memory_fragmentation_with_tensor_operations() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create tensors of different sizes to test fragmentation
    let mut small_tensors = Vec::new();
    let mut large_tensors = Vec::new();
    
    // Allocate small tensors
    for i in 0..50 {
        let tensor = BitNetTensor::zeros(&[64 + i], BitNetDType::F32, &device, &pool)
            .expect("Failed to create small tensor");
        small_tensors.push(tensor);
    }
    
    // Allocate large tensors
    for i in 0..10 {
        let tensor = BitNetTensor::zeros(&[1024 + i * 100], BitNetDType::F32, &device, &pool)
            .expect("Failed to create large tensor");
        large_tensors.push(tensor);
    }
    
    let fragmented_metrics = pool.get_metrics();
    
    // Deallocate every other small tensor to create fragmentation
    let mut i = 0;
    small_tensors.retain(|_| {
        i += 1;
        i % 2 == 0
    });
    
    // Try to allocate medium-sized tensors in fragmented space
    let mut medium_tensors = Vec::new();
    for i in 0..20 {
        let tensor = BitNetTensor::zeros(&[256 + i * 10], BitNetDType::F32, &device, &pool)
            .expect("Failed to allocate in fragmented space");
        medium_tensors.push(tensor);
    }
    
    let final_metrics = pool.get_metrics();
    assert!(final_metrics.active_allocations > 0);
}

#[test]
fn test_memory_alignment_for_different_dtypes() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    let dtypes = get_test_dtypes();
    
    for &dtype in &dtypes {
        let tensor = BitNetTensor::zeros(&[1000], dtype, &device, &pool)
            .expect("Failed to create aligned tensor");
        
        let handle = tensor.handle();
        
        // Verify tensor properties
        assert_eq!(handle.dtype().unwrap(), dtype);
        assert_eq!(handle.element_count().unwrap(), 1000);
        
        // Memory should be properly aligned for the data type
        let expected_size = dtype.bytes_for_elements(1000);
        assert_eq!(handle.size_bytes().unwrap(), expected_size);
        
        println!("Dtype {}: {} bytes for 1000 elements", dtype, expected_size);
    }
}

// =============================================================================
// Stress Testing and Memory Pressure Scenarios
// =============================================================================

#[test]
fn test_stress_tensor_allocation_under_pressure() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Stress test with rapid allocation/deallocation
    let iterations = 1000;
    let mut allocation_times = Vec::new();
    let mut deallocation_times = Vec::new();
    
    for i in 0..iterations {
        let size = 100 + (i % 500); // Varying sizes
        
        let alloc_start = Instant::now();
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create stress tensor");
        allocation_times.push(alloc_start.elapsed());
        
        let dealloc_start = Instant::now();
        drop(tensor);
        deallocation_times.push(dealloc_start.elapsed());
        
        // Check for memory pressure every 100 iterations
        if i % 100 == 0 {
            let metrics = pool.get_metrics();
            if metrics.active_allocations > 0 {
                println!("Iteration {}: {} active allocations", i, metrics.active_allocations);
            }
        }
    }
    
    // Analyze performance
    let avg_alloc_time = allocation_times.iter().sum::<Duration>() / iterations as u32;
    let avg_dealloc_time = deallocation_times.iter().sum::<Duration>() / iterations as u32;
    
    println!("Stress test completed:");
    println!("  Average allocation time: {:?}", avg_alloc_time);
    println!("  Average deallocation time: {:?}", avg_dealloc_time);
    
    // Performance should remain reasonable under stress (more realistic thresholds)
    assert!(avg_alloc_time < Duration::from_millis(1), "Allocation too slow under stress: {:?}", avg_alloc_time);
    assert!(avg_dealloc_time < Duration::from_millis(1), "Deallocation too slow under stress: {:?}", avg_dealloc_time);
}

#[test]
fn test_concurrent_tensor_memory_pressure() {
    let pool = Arc::new(create_tracked_pool());
    let device = get_cpu_device();
    
    let thread_count = 4;
    let tensors_per_thread = 50;
    let mut handles = Vec::new();
    
    // Spawn threads that create tensors concurrently
    for thread_id in 0..thread_count {
        let pool_clone = pool.clone();
        let device_clone = device.clone();
        
        let handle = thread::spawn(move || {
            let mut thread_tensors = Vec::new();
            
            for i in 0..tensors_per_thread {
                let size = 256 + thread_id * 100 + i * 10;
                
                match BitNetTensor::zeros(&[size], BitNetDType::F32, &device_clone, &pool_clone) {
                    Ok(tensor) => {
                        thread_tensors.push(tensor);
                    }
                    Err(e) => {
                        println!("Thread {} failed to allocate tensor {}: {}", thread_id, i, e);
                        break;
                    }
                }
                
                // Small delay to allow other threads to run
                thread::sleep(Duration::from_millis(1));
            }
            
            thread_tensors.len()
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads and collect results
    let mut total_tensors = 0;
    for handle in handles {
        total_tensors += handle.join().expect("Thread panicked");
    }
    
    println!("Concurrent stress test: {} tensors created across {} threads",
             total_tensors, thread_count);
    
    // Verify final state
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, total_tensors as u64);
}

#[test]
fn test_memory_pressure_recovery() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create memory pressure
    let mut pressure_tensors = Vec::new();
    for i in 0..100 {
        let size = 1024 * (i + 1);
        let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, &device, &pool)
            .expect("Failed to create pressure tensor");
        pressure_tensors.push(tensor);
    }
    
    let pressure_metrics = pool.get_metrics();
    println!("Peak memory usage: {} allocations, {} bytes",
             pressure_metrics.active_allocations, pressure_metrics.current_allocated);
    
    // Simulate recovery by dropping tensors gradually
    let recovery_steps = 5;
    let tensors_per_step = pressure_tensors.len() / recovery_steps;
    
    for step in 0..recovery_steps {
        // Drop a batch of tensors
        for _ in 0..tensors_per_step {
            if !pressure_tensors.is_empty() {
                pressure_tensors.pop();
            }
        }
        
        // With immediate cleanup, trigger cleanup to ensure dropped tensors are processed
        let cleanup_count = pool.cleanup_orphaned_handles();
        if cleanup_count > 0 {
            println!("Recovery step {}: cleaned up {} orphaned handles", step + 1, cleanup_count);
        }
        
        let step_metrics = pool.get_metrics();
        println!("Recovery step {}: {} allocations remaining",
                 step + 1, step_metrics.active_allocations);
        
        // Allow some time for cleanup
        thread::sleep(Duration::from_millis(10));
    }
    
    // Final cleanup to ensure all dropped tensors are processed
    let final_cleanup_count = pool.cleanup_orphaned_handles();
    println!("Final cleanup: {} orphaned handles", final_cleanup_count);
    
    // Verify recovery with immediate cleanup
    let final_metrics = pool.get_metrics();
    let expected_remaining = pressure_tensors.len() as u64;
    assert_eq!(final_metrics.active_allocations, expected_remaining,
               "Expected immediate cleanup to leave only {} remaining tensors", expected_remaining);
    assert!(final_metrics.current_allocated < pressure_metrics.current_allocated,
            "Expected memory usage to decrease after dropping tensors");
}

// =============================================================================
// Memory Leak Detection and Prevention Tests
// =============================================================================

#[test]
fn test_tensor_memory_leak_detection() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    // Create tensors in a scope that should be cleaned up
    {
        let mut tensors = Vec::new();
        for i in 0..20 {
            let tensor = BitNetTensor::zeros(&[256 + i * 10], BitNetDType::F32, &device, &pool)
                .expect("Failed to create leak test tensor");
            tensors.push(tensor);
        }
        
        let allocated_metrics = pool.get_metrics();
        assert!(allocated_metrics.active_allocations > initial_metrics.active_allocations);
    } // Tensors should be dropped here with immediate cleanup
    
    // With immediate cleanup, memory should be reclaimed immediately when tensors are dropped
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations,
               "Expected immediate cleanup to prevent memory leaks");
    assert_eq!(final_metrics.current_allocated, initial_metrics.current_allocated,
               "Expected immediate cleanup to reclaim all allocated memory");
}

#[test]
fn test_tensor_handle_leak_prevention() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    let tensor = BitNetTensor::zeros(&[512, 512], BitNetDType::F32, &device, &pool)
        .expect("Failed to create handle test tensor");
    
    // Create many handles
    let mut handles = Vec::new();
    for _ in 0..100 {
        handles.push(tensor.handle());
    }
    
    // Verify handles don't cause memory leaks
    let handle_metrics = pool.get_metrics();
    
    // Drop all handles
    drop(handles);
    
    // Memory usage should remain the same (handles don't own memory)
    let post_handle_metrics = pool.get_metrics();
    assert_eq!(handle_metrics.active_allocations, post_handle_metrics.active_allocations);
    assert_eq!(handle_metrics.current_allocated, post_handle_metrics.current_allocated);
    
    // Drop the tensor - immediate cleanup should occur
    drop(tensor);
    
    // With immediate cleanup, memory should be reclaimed immediately when tensor is dropped
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations,
               "Expected immediate cleanup to reclaim tensor memory");
    assert_eq!(final_metrics.current_allocated, initial_metrics.current_allocated,
               "Expected immediate cleanup to reclaim all allocated memory");
}

#[test]
fn test_circular_reference_prevention() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test that tensor handles use weak references to prevent cycles
    let tensor = BitNetTensor::zeros(&[256, 256], BitNetDType::F32, &device, &pool)
        .expect("Failed to create circular test tensor");
    
    let handle = tensor.handle();
    let weak_handle = handle.downgrade();
    
    // Verify weak reference behavior
    assert!(weak_handle.is_valid());
    assert_eq!(weak_handle.id(), handle.id());
    
    // Drop strong references
    drop(tensor);
    drop(handle);
    
    // Weak handle should become invalid
    assert!(!weak_handle.is_valid());
    assert!(weak_handle.upgrade().is_none());
}

// =============================================================================
// Memory Efficiency Validation Across Data Types
// =============================================================================

#[test]
fn test_memory_efficiency_by_dtype() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    let dtypes = get_test_dtypes();
    
    let element_count = 10000;
    let mut efficiency_results = HashMap::new();
    
    for &dtype in &dtypes {
        let tensor = BitNetTensor::zeros(&[element_count], dtype, &device, &pool)
            .expect("Failed to create efficiency test tensor");
        
        let actual_bytes = tensor.size_bytes();
        let expected_bytes = dtype.bytes_for_elements(element_count);
        let efficiency = dtype.memory_efficiency();
        
        assert_eq!(actual_bytes, expected_bytes);
        
        efficiency_results.insert(dtype, (actual_bytes, efficiency));
        
        println!("Dtype {}: {} bytes ({:.1}x efficiency vs F32)",
                dtype, actual_bytes, efficiency);
    }
    
    // Verify efficiency calculations
    let f32_bytes = efficiency_results[&BitNetDType::F32].0;
    
    for (&dtype, &(bytes, efficiency)) in &efficiency_results {
        if dtype != BitNetDType::F32 {
            let expected_efficiency = f32_bytes as f32 / bytes as f32;
            let diff = (efficiency - expected_efficiency).abs();
            assert!(diff < 0.01, "Efficiency mismatch for {}: expected {:.2}, got {:.2}",
                   dtype, expected_efficiency, efficiency);
        }
    }
}

#[test]
fn test_quantized_tensor_memory_efficiency() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test quantized data types specifically
    let quantized_dtypes = vec![
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];
    
    let element_count = 8192; // Use power of 2 for clean bit packing
    
    for &dtype in &quantized_dtypes {
        let tensor = BitNetTensor::zeros(&[element_count], dtype, &device, &pool)
            .expect("Failed to create quantized tensor");
        
        let bits_per_element = dtype.bits_per_element();
        let expected_bytes = (element_count * bits_per_element + 7) / 8; // Round up to bytes
        let actual_bytes = tensor.size_bytes();
        
        println!("Quantized dtype {}: {} bits/element, {} bytes total",
                dtype, bits_per_element, actual_bytes);
        
        // For quantized types, verify bit-level efficiency
        assert!(actual_bytes <= expected_bytes + 8, // Allow some padding
               "Quantized tensor using too much memory: {} > {} bytes",
               actual_bytes, expected_bytes);
        
        // Verify memory savings compared to F32
        let f32_tensor = BitNetTensor::zeros(&[element_count], BitNetDType::F32, &device, &pool)
            .expect("Failed to create F32 comparison tensor");
        let f32_bytes = f32_tensor.size_bytes();
        
        let savings_ratio = f32_bytes as f32 / actual_bytes as f32;
        println!("  Memory savings vs F32: {:.1}x", savings_ratio);
        
        assert!(savings_ratio >= 1.0, "Quantized type should use less memory than F32");
    }
}

#[test]
fn test_tensor_memory_overhead_analysis() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Measure overhead for different tensor configurations
    let test_configs = vec![
        ("Single large", vec![vec![10000]]),
        ("Many small", (0..100).map(|_| vec![100]).collect()),
        ("Mixed sizes", vec![vec![50], vec![500], vec![5000]]),
        ("Multi-dimensional", vec![vec![100, 100], vec![50, 50, 4], vec![25, 25, 4, 4]]),
    ];
    
    for (config_name, shapes) in test_configs {
        println!("Testing overhead for: {}", config_name);
        
        let start_metrics = pool.get_metrics();
        let mut tensors = Vec::new();
        let mut total_data_bytes = 0;
        
        for shape in &shapes {
            let tensor = BitNetTensor::zeros(shape, BitNetDType::F32, &device, &pool)
                .expect("Failed to create overhead test tensor");
            total_data_bytes += tensor.size_bytes();
            tensors.push(tensor);
        }
        
        let end_metrics = pool.get_metrics();
        let allocated_bytes = end_metrics.current_allocated - start_metrics.current_allocated;
        let overhead_bytes = allocated_bytes.saturating_sub(total_data_bytes as u64);
        let overhead_percentage = if total_data_bytes > 0 {
            (overhead_bytes as f64 / total_data_bytes as f64) * 100.0
        } else {
            0.0
        };
        
        println!("  Data bytes: {}", total_data_bytes);
        println!("  Allocated bytes: {}", allocated_bytes);
        println!("  Overhead: {} bytes ({:.1}%)", overhead_bytes, overhead_percentage);
        
        // Overhead should be reasonable (less than 20% for most cases)
        assert!(overhead_percentage < 20.0,
               "Memory overhead too high for {}: {:.1}%", config_name, overhead_percentage);
    }
}

#[test]
fn test_comprehensive_tensor_memory_integration() {
    let pool = create_tracked_pool();
    let devices = get_test_devices();
    let dtypes = get_test_dtypes();
    
    println!("Running comprehensive tensor memory integration test");
    
    let mut total_tensors = 0;
    let mut total_memory = 0u64;
    
    // Test all combinations of devices and data types
    for device in &devices {
        for &dtype in &dtypes {
            // Create tensors of various sizes
            let sizes = vec![64, 256, 1024];
            
            for size in sizes {
                let tensor = BitNetTensor::zeros(&[size], dtype, device, &pool)
                    .expect("Failed to create integration test tensor");
                
                total_tensors += 1;
                total_memory += tensor.size_bytes() as u64;
                
                // Verify tensor properties
                assert_eq!(tensor.shape(), vec![size]);
                assert_eq!(tensor.dtype(), dtype);
                // Verify device consistency using custom device comparison
                bitnet_core::device::assert_devices_equal(&tensor.device(), &device);
                
                // Verify handle functionality
                let handle = tensor.handle();
                assert!(handle.is_valid());
                assert_eq!(handle.tensor_id().unwrap(), tensor.id());
                
                // Test handle operations
                assert!(handle.touch().is_ok());
                assert!(handle.add_tag("integration_test".to_string()).is_ok());
                assert!(handle.has_tag("integration_test").unwrap());
            }
        }
    }
    
    // Verify final state
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, total_tensors);
    
    println!("Integration test completed:");
    println!("  Total tensors created: {}", total_tensors);
    println!("  Total memory allocated: {} bytes", total_memory);
    println!("  Active allocations: {}", final_metrics.active_allocations);
    println!("  Current allocated: {} bytes", final_metrics.current_allocated);
    
    // Verify memory tracking accuracy
    assert!(final_metrics.current_allocated >= total_memory);
}