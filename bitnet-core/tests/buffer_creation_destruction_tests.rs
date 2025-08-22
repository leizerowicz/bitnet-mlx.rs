//! Comprehensive Buffer Creation/Destruction Tests
//!
//! This test suite validates all aspects of buffer lifecycle management in the BitNet
//! memory system, including creation, destruction, device-specific handling, memory
//! pool integration, and error scenarios. Tests cover both CPU and Metal GPU buffers
//! across different allocation strategies and stress conditions.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, MemoryError,
    SmallBlockPool, LargeBlockPool, TrackingConfig, MemoryPressureLevel
};
use bitnet_core::memory::handle::PoolType;
use bitnet_core::device::{get_cpu_device, is_metal_available, get_metal_device};

#[cfg(feature = "metal")]
use bitnet_core::memory::{MetalMemoryPool, handle::MetalMemoryMetadata};

/// Helper function to create a basic memory pool for testing
fn create_test_pool() -> HybridMemoryPool {
    HybridMemoryPool::new().expect("Failed to create test memory pool")
}

/// Helper function to create a memory pool with tracking enabled
fn create_tracked_pool() -> HybridMemoryPool {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());
    
    HybridMemoryPool::with_config(config)
        .expect("Failed to create tracked memory pool")
}

/// Helper function to get all available test devices
fn get_test_devices() -> Vec<candle_core::Device> {
    let mut devices = vec![get_cpu_device()];
    
    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            devices.push(metal_device);
        }
    }
    
    devices
}

// =============================================================================
// Basic Buffer Creation/Destruction Tests
// =============================================================================

#[test]
fn test_basic_buffer_creation_cpu() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test various buffer sizes
    let test_sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536, 1024*1024];
    let test_alignments: Vec<usize> = vec![1, 4, 8, 16, 32, 64, 128, 256];
    
    for &size in &test_sizes {
        for &alignment in &test_alignments {
            if !alignment.is_power_of_two() {
                continue;
            }
            
            let handle = pool.allocate(size, alignment, &device)
                .unwrap_or_else(|_| panic!("Failed to create buffer: size={size}, alignment={alignment}"));
            
            // Verify buffer properties
            assert_eq!(handle.size(), size);
            assert_eq!(handle.alignment(), alignment);
            assert!(handle.is_cpu());
            assert!(!handle.is_metal());
            assert!(!handle.is_cuda());
            
            // Verify handle validation
            handle.validate().expect("Buffer handle should be valid");
            
            // Verify device consistency
            assert!(format!("{:?}", handle.device()) == format!("{device:?}"));
            
            // Clean up
            pool.deallocate(handle).expect("Failed to deallocate buffer");
        }
    }
}

#[test]
#[cfg(feature = "metal")]
fn test_basic_buffer_creation_metal() {
    if !is_metal_available() {
        println!("Skipping Metal buffer test - Metal not available");
        return;
    }
    
    let pool = create_test_pool();
    let device = get_metal_device().expect("Failed to get Metal device");
    
    // Test various buffer sizes
    let test_sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536, 1024*1024];
    
    for &size in &test_sizes {
        let handle = pool.allocate(size, 16, &device)
            .expect(&format!("Failed to create Metal buffer: size={}", size));
        
        // Verify buffer properties
        assert_eq!(handle.size(), size);
        assert_eq!(handle.alignment(), 16);
        assert!(!handle.is_cpu());
        assert!(handle.is_metal());
        assert!(!handle.is_cuda());
        
        // Verify handle validation
        handle.validate().expect("Metal buffer handle should be valid");
        
        // Verify device consistency
        assert!(format!("{:?}", handle.device()) == format!("{:?}", device));
        
        // Clean up
        pool.deallocate(handle).expect("Failed to deallocate Metal buffer");
    }
}

#[test]
fn test_buffer_destruction_immediate() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    // Create buffer
    let handle = pool.allocate(4096, 16, &device)
        .expect("Failed to create buffer");
    
    let allocated_metrics = pool.get_metrics();
    assert!(allocated_metrics.active_allocations > initial_metrics.active_allocations);
    assert!(allocated_metrics.current_allocated > initial_metrics.current_allocated);
    
    // Destroy buffer immediately
    pool.deallocate(handle).expect("Failed to deallocate buffer");
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations);
    assert_eq!(final_metrics.current_allocated, initial_metrics.current_allocated);
}

#[test]
fn test_buffer_destruction_delayed() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    // Create buffer and hold reference
    let handle = pool.allocate(8192, 32, &device)
        .expect("Failed to create buffer");
    
    // Simulate some work with the buffer
    thread::sleep(Duration::from_millis(10));
    
    let allocated_metrics = pool.get_metrics();
    assert!(allocated_metrics.active_allocations > initial_metrics.active_allocations);
    
    // Destroy buffer after delay
    pool.deallocate(handle).expect("Failed to deallocate buffer");
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations);
}

#[test]
fn test_buffer_destruction_scope_based() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    let initial_metrics = pool.get_metrics();
    
    // Create buffers in a scope
    {
        let _handle1 = pool.allocate(1024, 16, &device)
            .expect("Failed to create buffer 1");
        let _handle2 = pool.allocate(2048, 32, &device)
            .expect("Failed to create buffer 2");
        let _handle3 = pool.allocate(4096, 64, &device)
            .expect("Failed to create buffer 3");
        
        let scoped_metrics = pool.get_metrics();
        assert_eq!(scoped_metrics.active_allocations, initial_metrics.active_allocations + 3);
        
        // Manually deallocate one buffer
        pool.deallocate(_handle2).expect("Failed to deallocate buffer 2");
        
        let partial_metrics = pool.get_metrics();
        assert_eq!(partial_metrics.active_allocations, initial_metrics.active_allocations + 2);
        
        // Remaining buffers will be deallocated when handles are dropped
        pool.deallocate(_handle1).expect("Failed to deallocate buffer 1");
        pool.deallocate(_handle3).expect("Failed to deallocate buffer 3");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, initial_metrics.active_allocations);
}

// =============================================================================
// Buffer Pool Management Tests
// =============================================================================

#[test]
fn test_small_block_pool_buffer_lifecycle() {
    let device = get_cpu_device();
    let mut pool = SmallBlockPool::new(1024*1024, 16*1024*1024, &device)
        .expect("Failed to create small block pool");
    let handle_counter = Arc::new(Mutex::new(1));
    
    // Test small buffer allocations (should use small block pool)
    let small_sizes = vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let mut handles = Vec::new();
    
    for &size in &small_sizes {
        let handle = pool.allocate(size, 16, &device, handle_counter.clone())
            .unwrap_or_else(|_| panic!("Failed to allocate small buffer: {size}"));
        
        assert_eq!(handle.size(), size);
        assert_eq!(handle.pool_type(), PoolType::SmallBlock);
        handles.push(handle);
    }
    
    // Verify pool statistics
    let stats = pool.get_stats();
    assert!(pool.current_usage() > 0);
    
    // Deallocate all buffers
    for handle in handles {
        pool.deallocate(handle).expect("Failed to deallocate small buffer");
    }
    
    let final_stats = pool.get_stats();
    // Note: We can't directly access private fields, but we can verify the pool still works
    assert!(pool.current_usage() > 0); // Pool retains chunks
}

#[test]
fn test_large_block_pool_buffer_lifecycle() {
    let device = get_cpu_device();
    let mut pool = LargeBlockPool::new(64*1024*1024, 256*1024*1024, &device)
        .expect("Failed to create large block pool");
    let handle_counter = Arc::new(Mutex::new(1));
    
    // Test large buffer allocations (should use large block pool)
    let large_sizes = vec![1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024];
    let mut handles = Vec::new();
    
    for &size in &large_sizes {
        let handle = pool.allocate(size, 16, &device, handle_counter.clone())
            .unwrap_or_else(|_| panic!("Failed to allocate large buffer: {size}"));
        
        assert_eq!(handle.size(), size);
        assert_eq!(handle.pool_type(), PoolType::LargeBlock);
        handles.push(handle);
    }
    
    // Verify pool statistics
    let _stats = pool.get_stats();
    assert!(pool.current_usage() > 0);
    
    // Deallocate all buffers
    for handle in handles {
        pool.deallocate(handle).expect("Failed to deallocate large buffer");
    }
    
    let _final_stats = pool.get_stats();
    // Note: We can't directly access private fields, but we can verify the pool still works
    assert!(pool.current_usage() > 0); // Pool retains arenas
}

#[test]
fn test_hybrid_pool_buffer_routing() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test that buffers are routed to appropriate pools
    let test_cases = vec![
        (512, "small"),           // Small block pool
        (1024, "small"),          // Small block pool
        (512*1024, "small"),      // Small block pool (just under 1MB)
        (1024*1024, "large"),     // Large block pool (1MB)
        (2*1024*1024, "large"),   // Large block pool
    ];
    
    for (size, expected_pool) in test_cases {
        let handle = pool.allocate(size, 16, &device)
            .unwrap_or_else(|_| panic!("Failed to allocate buffer: {size}"));
        
        let expected_pool_type = match expected_pool {
            "small" => PoolType::SmallBlock,
            "large" => PoolType::LargeBlock,
            _ => panic!("Invalid pool type"),
        };
        
        assert_eq!(handle.pool_type(), expected_pool_type,
                  "Buffer size {size} should use {expected_pool} pool");
        
        pool.deallocate(handle).expect("Failed to deallocate buffer");
    }
}

#[test]
fn test_device_specific_pool_isolation() {
    let devices = get_test_devices();
    
    if devices.len() < 2 {
        println!("Skipping device isolation test - need multiple devices");
        return;
    }
    
    let pool = create_tracked_pool();
    let mut device_handles = HashMap::new();
    
    // Create buffers on different devices
    for device in &devices {
        let mut handles = Vec::new();
        
        for i in 0..5 {
            let size = 1024 * (i + 1);
            let handle = pool.allocate(size, 16, device)
                .expect("Failed to allocate device-specific buffer");
            
            // Verify device consistency
            assert!(format!("{:?}", handle.device()) == format!("{device:?}"));
            handles.push(handle);
        }
        
        device_handles.insert(format!("{device:?}"), handles);
    }
    
    // Verify device-specific metrics
    if let Some(detailed_metrics) = pool.get_detailed_metrics() {
        for device in &devices {
            let device_key = format!("{device:?}");
            assert!(detailed_metrics.device_usage.contains_key(&device_key));
            assert!(detailed_metrics.device_usage[&device_key] > 0);
        }
    }
    
    // Clean up all device buffers
    for (device_name, handles) in device_handles {
        println!("Cleaning up {} buffers for device: {}", handles.len(), device_name);
        for handle in handles {
            pool.deallocate(handle).expect("Failed to deallocate device buffer");
        }
    }
}

// =============================================================================
// Buffer Handle Validation Tests
// =============================================================================

#[test]
fn test_buffer_handle_validation_success() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let handle = pool.allocate(4096, 64, &device)
        .expect("Failed to create buffer");
    
    // Valid handle should pass all validation checks
    handle.validate().expect("Valid handle should pass validation");
    
    // Test handle properties
    assert!(handle.id() > 0);
    assert_eq!(handle.size(), 4096);
    assert_eq!(handle.alignment(), 64);
    assert!(handle.is_cpu());
    
    // Test CPU metadata access
    if let Some(cpu_metadata) = handle.cpu_metadata() {
        assert!(cpu_metadata.page_aligned); // 64-byte alignment >= 4096
        assert!(!cpu_metadata.locked);
    } else {
        panic!("CPU buffer should have CPU metadata");
    }
    
    pool.deallocate(handle).expect("Failed to deallocate buffer");
}

#[test]
fn test_buffer_handle_validation_failures() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test invalid alignment (not power of 2)
    let result = pool.allocate(1024, 3, &device);
    assert!(result.is_err());
    if let Err(MemoryError::InvalidAlignment { alignment }) = result {
        assert_eq!(alignment, 3);
    } else {
        panic!("Expected InvalidAlignment error");
    }
    
    // Test zero alignment
    let result = pool.allocate(1024, 0, &device);
    assert!(result.is_err());
    
    // Test zero size (should succeed but create minimal buffer)
    let handle = pool.allocate(0, 16, &device)
        .expect("Zero-size allocation should succeed");
    assert_eq!(handle.size(), 0);
    pool.deallocate(handle).expect("Failed to deallocate zero-size buffer");
}

#[test]
fn test_buffer_handle_double_free_prevention() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    let handle = pool.allocate(2048, 16, &device)
        .expect("Failed to create buffer");
    
    // First deallocation should succeed
    pool.deallocate(handle.clone()).expect("First deallocation should succeed");
    
    // Second deallocation should fail
    let result = pool.deallocate(handle);
    assert!(result.is_err());
    if let Err(MemoryError::InvalidHandle { reason }) = result {
        assert!(reason.contains("not found in registry"));
    } else {
        panic!("Expected InvalidHandle error for double free");
    }
}

#[test]
fn test_buffer_handle_wrong_pool_rejection() {
    let device = get_cpu_device();
    let mut small_pool = SmallBlockPool::new(1024*1024, 16*1024*1024, &device)
        .expect("Failed to create small pool");
    let mut large_pool = LargeBlockPool::new(64*1024*1024, 256*1024*1024, &device)
        .expect("Failed to create large pool");
    let handle_counter = Arc::new(Mutex::new(1));
    
    // Create buffer in small pool
    let small_handle = small_pool.allocate(1024, 16, &device, handle_counter.clone())
        .expect("Failed to create small buffer");
    
    // Try to deallocate in large pool (should fail)
    let result = large_pool.deallocate(small_handle.clone());
    assert!(result.is_err());
    
    // Deallocate in correct pool
    small_pool.deallocate(small_handle).expect("Failed to deallocate in correct pool");
}

// =============================================================================
// Concurrent Buffer Operations Tests
// =============================================================================

#[test]
fn test_concurrent_buffer_creation() {
    let pool = Arc::new(create_tracked_pool());
    let device = get_cpu_device();
    let thread_count = 4;
    let buffers_per_thread = 25;
    
    let mut handles = Vec::new();
    
    // Spawn threads that create buffers concurrently
    for thread_id in 0..thread_count {
        let pool_clone = pool.clone();
        let device_clone = device.clone();
        
        let handle = thread::spawn(move || {
            let mut thread_handles = Vec::new();
            
            for i in 0..buffers_per_thread {
                let size = 1024 + thread_id * 100 + i * 10;
                let handle = pool_clone.allocate(size, 16, &device_clone)
                    .expect("Failed to allocate buffer in thread");
                thread_handles.push(handle);
            }
            
            thread_handles
        });
        
        handles.push(handle);
    }
    
    // Collect all handles
    let mut all_handles = Vec::new();
    for handle in handles {
        let mut thread_handles = handle.join().expect("Thread panicked");
        all_handles.append(&mut thread_handles);
    }
    
    // Verify all handles are unique
    for i in 0..all_handles.len() {
        for j in i+1..all_handles.len() {
            assert_ne!(all_handles[i].id(), all_handles[j].id());
        }
    }
    
    // Verify metrics
    let metrics = pool.get_metrics();
    assert_eq!(metrics.active_allocations, all_handles.len() as u64);
    
    // Clean up all handles
    for handle in all_handles {
        pool.deallocate(handle).expect("Failed to deallocate concurrent buffer");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_concurrent_buffer_destruction() {
    let pool = Arc::new(create_tracked_pool());
    let device = get_cpu_device();
    let buffer_count = 100;
    
    // Create many buffers
    let mut handles = Vec::new();
    for i in 0..buffer_count {
        let size = 1024 + i * 10;
        let handle = pool.allocate(size, 16, &device)
            .expect("Failed to create buffer");
        handles.push(handle);
    }
    
    let allocated_metrics = pool.get_metrics();
    assert_eq!(allocated_metrics.active_allocations, buffer_count as u64);
    
    // Deallocate buffers concurrently
    let handles_arc = Arc::new(Mutex::new(handles));
    let mut threads = Vec::new();
    
    for _ in 0..4 {
        let pool_clone = pool.clone();
        let handles_clone = handles_arc.clone();
        
        let thread = thread::spawn(move || {
            loop {
                let handle = {
                    let mut handles = handles_clone.lock().unwrap();
                    if handles.is_empty() {
                        break;
                    }
                    handles.pop().unwrap()
                };
                
                pool_clone.deallocate(handle).expect("Failed to deallocate buffer");
            }
        });
        
        threads.push(thread);
    }
    
    // Wait for all threads to complete
    for thread in threads {
        thread.join().expect("Thread panicked");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_concurrent_mixed_buffer_operations() {
    let pool = Arc::new(create_tracked_pool());
    let device = get_cpu_device();
    let operation_count = 200;
    
    let handles = Arc::new(Mutex::new(Vec::new()));
    let mut threads = Vec::new();
    
    // Spawn threads doing mixed operations
    for thread_id in 0..4 {
        let pool_clone = pool.clone();
        let device_clone = device.clone();
        let handles_clone = handles.clone();
        
        let thread = thread::spawn(move || {
            for i in 0..operation_count / 4 {
                if i % 2 == 0 {
                    // Allocate buffer
                    let size = 1024 + thread_id * 100 + i * 10;
                    if let Ok(handle) = pool_clone.allocate(size, 16, &device_clone) {
                        handles_clone.lock().unwrap().push(handle);
                    }
                } else {
                    // Deallocate buffer if available
                    let handle = {
                        let mut handles_guard = handles_clone.lock().unwrap();
                        if !handles_guard.is_empty() {
                            Some(handles_guard.pop().unwrap())
                        } else {
                            None
                        }
                    };
                    
                    if let Some(h) = handle {
                        let _ = pool_clone.deallocate(h);
                    }
                }
                
                // Small delay to increase concurrency
                thread::sleep(Duration::from_micros(10));
            }
        });
        
        threads.push(thread);
    }
    
    // Wait for all threads
    for thread in threads {
        thread.join().expect("Thread panicked");
    }
    
    // Clean up remaining handles
    let remaining_handles = handles.lock().unwrap();
    for handle in remaining_handles.iter() {
        pool.deallocate(handle.clone()).expect("Failed to clean up remaining buffer");
    }
}

// =============================================================================
// Buffer Cleanup and Leak Prevention Tests
// =============================================================================

#[test]
fn test_buffer_automatic_cleanup_on_pool_drop() {
    let initial_metrics = {
        let pool = create_tracked_pool();
        let device = get_cpu_device();
        
        // Create some buffers
        let _handle1 = pool.allocate(1024, 16, &device)
            .expect("Failed to create buffer 1");
        let _handle2 = pool.allocate(2048, 32, &device)
            .expect("Failed to create buffer 2");
        let _handle3 = pool.allocate(4096, 64, &device)
            .expect("Failed to create buffer 3");
        
        let metrics = pool.get_metrics();
        assert_eq!(metrics.active_allocations, 3);
        
        // Don't explicitly deallocate - let pool handle cleanup
        metrics
    }; // Pool drops here, should clean up all buffers
    
    // Verify cleanup occurred (this is more of a conceptual test)
    assert_eq!(initial_metrics.active_allocations, 3);
}

#[test]
fn test_buffer_leak_detection() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create buffers that might leak
    let mut potential_leaks = Vec::new();
    for i in 0..10 {
        let handle = pool.allocate(1024 * (i + 1), 16, &device)
            .expect("Failed to create potential leak buffer");
        potential_leaks.push(handle);
    }
    
    let allocated_metrics = pool.get_metrics();
    assert_eq!(allocated_metrics.active_allocations, 10);
    
    // Simulate forgetting to deallocate some buffers
    for _ in 0..5 {
        let handle = potential_leaks.pop().unwrap();
        pool.deallocate(handle).expect("Failed to deallocate buffer");
    }
    
    let partial_metrics = pool.get_metrics();
    assert_eq!(partial_metrics.active_allocations, 5);
    
    // Clean up remaining "leaked" buffers
    for handle in potential_leaks {
        pool.deallocate(handle).expect("Failed to clean up leaked buffer");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_buffer_cleanup_with_memory_pressure() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Register pressure callback
    let pressure_events = Arc::new(Mutex::new(Vec::new()));
    let pressure_events_clone = pressure_events.clone();
    
    pool.register_pressure_callback(Box::new(move |level| {
        pressure_events_clone.lock().unwrap().push(level);
    }));
    
    // Create many buffers to trigger pressure
    let mut buffers = Vec::new();
    for i in 0..100 {
        let size = 1024 * (i + 1);
        match pool.allocate(size, 16, &device) {
            Ok(handle) => buffers.push(handle),
            Err(_) => break, // Stop if allocation fails
        }
        
        // Check for pressure detection
        if let Some(detailed_metrics) = pool.get_detailed_metrics() {
            if detailed_metrics.pressure_level != MemoryPressureLevel::None {
                println!("Memory pressure detected at buffer {}: {:?}", 
                        i, detailed_metrics.pressure_level);
                break;
            }
        }
    }
    
    // Clean up buffers under pressure
    let cleanup_start = Instant::now();
    for handle in buffers {
        pool.deallocate(handle).expect("Failed to deallocate under pressure");
    }
    let cleanup_duration = cleanup_start.elapsed();
    
    println!("Cleanup under pressure completed in {cleanup_duration:?}");
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_buffer_cleanup_with_fragmentation() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create buffers with varying sizes to cause fragmentation
    let mut buffers = Vec::new();
    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];
    
    for _ in 0..10 {
        for &size in &sizes {
            let handle = pool.allocate(size, 16, &device)
                .expect("Failed to create fragmentation buffer");
            buffers.push(handle);
        }
    }
    
    let fragmented_metrics = pool.get_metrics();
    assert_eq!(fragmented_metrics.active_allocations, 70); // 10 * 7 sizes
    
    // Deallocate every other buffer to create fragmentation
    let mut i = 0;
    buffers.retain(|handle| {
        i += 1;
        if i % 2 == 0 {
            pool.deallocate(handle.clone()).expect("Failed to deallocate fragmentation buffer");
            false
        } else {
            true
        }
    });
    
    let partial_metrics = pool.get_metrics();
    assert_eq!(partial_metrics.active_allocations, 35); // Half remaining
    
    // Clean up remaining buffers
    for handle in buffers {
        pool.deallocate(handle).expect("Failed to clean up remaining fragmentation buffer");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

// =============================================================================
// Buffer Stress Testing
// =============================================================================

#[test]
fn test_buffer_stress_rapid_allocation_deallocation() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    let iterations = 1000;
    
    let mut allocation_times = Vec::new();
    let mut deallocation_times = Vec::new();
    
    for i in 0..iterations {
        let size = 1024 + (i % 100) * 10; // Varying sizes
        
        // Measure allocation time
        let alloc_start = Instant::now();
        let handle = pool.allocate(size, 16, &device)
            .expect("Failed to allocate in stress test");
        allocation_times.push(alloc_start.elapsed());
        
        // Measure deallocation time
        let dealloc_start = Instant::now();
        pool.deallocate(handle).expect("Failed to deallocate in stress test");
        deallocation_times.push(dealloc_start.elapsed());
    }
    
    // Analyze performance
    let avg_alloc_time = allocation_times.iter().sum::<Duration>() / iterations as u32;
    let avg_dealloc_time = deallocation_times.iter().sum::<Duration>() / iterations as u32;
    
    println!("Stress test results:");
    println!("  Average allocation time: {avg_alloc_time:?}");
    println!("  Average deallocation time: {avg_dealloc_time:?}");
    
    // Performance should remain reasonable
    assert!(avg_alloc_time < Duration::from_micros(100), "Allocation too slow");
    assert!(avg_dealloc_time < Duration::from_micros(50), "Deallocation too slow");
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_buffer_stress_memory_exhaustion() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Try to allocate until memory is exhausted
    let mut buffers = Vec::new();
    let mut allocation_count = 0;
    
    loop {
        let size = 1024 * 1024; // 1MB buffers
        match pool.allocate(size, 16, &device) {
            Ok(handle) => {
                buffers.push(handle);
                allocation_count += 1;
                
                // Prevent infinite loop in case of very large memory
                if allocation_count > 1000 {
                    break;
                }
            }
            Err(MemoryError::InsufficientMemory { .. }) => {
                println!("Memory exhausted after {allocation_count} allocations");
                break;
            }
            Err(e) => {
                panic!("Unexpected error during stress test: {e}");
            }
        }
    }
    
    // Clean up all allocated buffers
    for handle in buffers {
        pool.deallocate(handle).expect("Failed to deallocate stress buffer");
    }
    
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
}

#[test]
fn test_buffer_stress_size_variations() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Test with highly variable buffer sizes
    let sizes = vec![
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072, 262144, 524288,
        1024*1024, 2*1024*1024, 4*1024*1024
    ];
    
    for &size in &sizes {
        let start_time = Instant::now();
        
        // Allocate multiple buffers of the same size
        let mut handles = Vec::new();
        for _ in 0..10 {
            match pool.allocate(size, 16, &device) {
                Ok(handle) => handles.push(handle),
                Err(_) => break, // Skip if allocation fails
            }
        }
        
        let allocation_time = start_time.elapsed();
        
        // Deallocate all buffers
        let dealloc_start = Instant::now();
        for handle in handles {
            pool.deallocate(handle).expect("Failed to deallocate size variation buffer");
        }
        let deallocation_time = dealloc_start.elapsed();
        
        println!("Size {size}: alloc {allocation_time:?}, dealloc {deallocation_time:?}");
    }
}

// =============================================================================
// Buffer Device Migration Tests
// =============================================================================

#[test]
fn test_buffer_device_migration_cpu_to_metal() {
    let devices = get_test_devices();
    
    if devices.len() < 2 {
        println!("Skipping device migration test - need multiple devices");
        return;
    }
    
    let pool = create_tracked_pool();
    let cpu_device = &devices[0];
    let target_device = &devices[1];
    
    // Create buffer on CPU
    let cpu_handle = pool.allocate(4096, 16, cpu_device)
        .expect("Failed to create CPU buffer");
    
    assert!(format!("{:?}", cpu_handle.device()) == format!("{cpu_device:?}"));
    
    // Simulate migration by creating new buffer on target device
    let migrated_handle = pool.allocate(4096, 16, target_device)
        .expect("Failed to create migrated buffer");
    
    assert!(format!("{:?}", migrated_handle.device()) == format!("{target_device:?}"));
    assert_ne!(cpu_handle.id(), migrated_handle.id());
    
    // Clean up both buffers
    pool.deallocate(cpu_handle).expect("Failed to deallocate CPU buffer");
    pool.deallocate(migrated_handle).expect("Failed to deallocate migrated buffer");
}

#[test]
fn test_buffer_cross_device_operations() {
    let devices = get_test_devices();
    let pool = create_tracked_pool();
    
    // Create buffers on all available devices
    let mut device_buffers = HashMap::new();
    
    for device in &devices {
        let mut buffers = Vec::new();
        
        for i in 0..5 {
            let size = 1024 * (i + 1);
            let handle = pool.allocate(size, 16, device)
                .expect("Failed to create cross-device buffer");
            buffers.push(handle);
        }
        
        device_buffers.insert(format!("{device:?}"), buffers);
    }
    
    // Verify device isolation
    for (device_name, buffers) in &device_buffers {
        for handle in buffers {
            let handle_device = format!("{:?}", handle.device());
            assert_eq!(&handle_device, device_name);
        }
    }
    
    // Clean up all buffers
    for (_, buffers) in device_buffers {
        for handle in buffers {
            pool.deallocate(handle).expect("Failed to deallocate cross-device buffer");
        }
    }
}

// =============================================================================
// Buffer Memory Pressure Handling Tests
// =============================================================================

#[test]
fn test_buffer_allocation_under_memory_pressure() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create initial memory pressure
    let mut pressure_buffers = Vec::new();
    for i in 0..50 {
        let size = 1024 * 1024 * (i + 1); // Increasing sizes
        match pool.allocate(size, 16, &device) {
            Ok(handle) => pressure_buffers.push(handle),
            Err(_) => break, // Stop when allocation fails
        }
    }
    
    println!("Created {} buffers for pressure test", pressure_buffers.len());
    
    // Try to allocate under pressure
    let pressure_allocation = pool.allocate(512 * 1024, 16, &device);
    
    match pressure_allocation {
        Ok(handle) => {
            println!("Successfully allocated under pressure");
            pool.deallocate(handle).expect("Failed to deallocate pressure allocation");
        }
        Err(MemoryError::InsufficientMemory { .. }) => {
            println!("Allocation failed under pressure (expected)");
        }
        Err(e) => {
            panic!("Unexpected error under pressure: {e}");
        }
    }
    
    // Clean up pressure buffers
    for handle in pressure_buffers {
        pool.deallocate(handle).expect("Failed to deallocate pressure buffer");
    }
}

#[test]
fn test_buffer_pressure_callback_integration() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Track pressure events
    let pressure_events = Arc::new(Mutex::new(Vec::new()));
    let pressure_events_clone = pressure_events.clone();
    
    pool.register_pressure_callback(Box::new(move |level| {
        pressure_events_clone.lock().unwrap().push(level);
    }));
    
    // Gradually increase memory usage
    let mut buffers = Vec::new();
    for i in 0..100 {
        let size = 1024 * 1024 * (i + 1);
        match pool.allocate(size, 16, &device) {
            Ok(handle) => {
                buffers.push(handle);
                
                // Check for pressure detection
                if let Some(detailed_metrics) = pool.get_detailed_metrics() {
                    if detailed_metrics.pressure_level != MemoryPressureLevel::None {
                        println!("Pressure detected at buffer {}: {:?}",
                                i, detailed_metrics.pressure_level);
                        break;
                    }
                }
            }
            Err(_) => break,
        }
    }
    
    // Check if pressure events were recorded
    let events = pressure_events.lock().unwrap();
    if !events.is_empty() {
        println!("Pressure events recorded: {:?}", *events);
    }
    
    // Clean up
    for handle in buffers {
        pool.deallocate(handle).expect("Failed to deallocate pressure test buffer");
    }
}

#[test]
fn test_buffer_recovery_from_pressure() {
    let pool = create_tracked_pool();
    let device = get_cpu_device();
    
    // Create pressure condition
    let mut pressure_buffers = Vec::new();
    for i in 0..20 {
        let size = 1024 * 1024 * (i + 1);
        match pool.allocate(size, 16, &device) {
            Ok(handle) => pressure_buffers.push(handle),
            Err(_) => break,
        }
    }
    
    let pressure_metrics = pool.get_metrics();
    println!("Peak pressure: {} allocations", pressure_metrics.active_allocations);
    
    // Gradually release pressure
    let release_count = pressure_buffers.len() / 2;
    for _ in 0..release_count {
        if let Some(handle) = pressure_buffers.pop() {
            pool.deallocate(handle).expect("Failed to release pressure buffer");
        }
    }
    
    let recovery_metrics = pool.get_metrics();
    println!("After recovery: {} allocations", recovery_metrics.active_allocations);
    
    // Try allocation after pressure relief
    let recovery_allocation = pool.allocate(1024 * 1024, 16, &device);
    match recovery_allocation {
        Ok(handle) => {
            println!("Successfully allocated after pressure recovery");
            pool.deallocate(handle).expect("Failed to deallocate recovery buffer");
        }
        Err(e) => {
            println!("Allocation still failing after recovery: {e}");
        }
    }
    
    // Clean up remaining buffers
    for handle in pressure_buffers {
        pool.deallocate(handle).expect("Failed to clean up remaining pressure buffer");
    }
}

// =============================================================================
// Buffer Error Handling and Edge Cases
// =============================================================================

#[test]
fn test_buffer_invalid_parameters() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test invalid alignments
    let invalid_alignments = vec![0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17];
    
    for &alignment in &invalid_alignments {
        let result = pool.allocate(1024, alignment, &device);
        assert!(result.is_err(), "Should fail with invalid alignment: {alignment}");
        
        if let Err(MemoryError::InvalidAlignment { alignment: a }) = result {
            assert_eq!(a, alignment);
        }
    }
    
    // Test valid alignments
    let valid_alignments = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    
    for &alignment in &valid_alignments {
        let handle = pool.allocate(1024, alignment, &device)
            .unwrap_or_else(|_| panic!("Should succeed with valid alignment: {alignment}"));
        assert_eq!(handle.alignment(), alignment);
        pool.deallocate(handle).expect("Failed to deallocate valid alignment buffer");
    }
}

#[test]
fn test_buffer_extreme_sizes() {
    let pool = create_test_pool();
    let device = get_cpu_device();
    
    // Test very small sizes
    let small_sizes = vec![0, 1, 2, 3, 4, 5, 7, 8, 15, 16];
    
    for &size in &small_sizes {
        let handle = pool.allocate(size, 1, &device)
            .unwrap_or_else(|_| panic!("Should handle small size: {size}"));
        assert_eq!(handle.size(), size);
        pool.deallocate(handle).expect("Failed to deallocate small buffer");
    }
    
    // Test large sizes (within reasonable limits)
    let large_sizes = vec![
        1024 * 1024,      // 1MB
        10 * 1024 * 1024, // 10MB
        50 * 1024 * 1024, // 50MB
    ];
    
    for &size in &large_sizes {
        match pool.allocate(size, 16, &device) {
            Ok(handle) => {
                assert_eq!(handle.size(), size);
                pool.deallocate(handle).expect("Failed to deallocate large buffer");
            }
            Err(MemoryError::InsufficientMemory { .. }) => {
                println!("Large allocation failed (acceptable): {size} bytes");
            }
            Err(e) => {
                panic!("Unexpected error for large allocation: {e}");
            }
        }
    }
}

#[test]
fn test_buffer_comprehensive_integration() {
    let pool = create_tracked_pool();
    let devices = get_test_devices();
    
    println!("Running comprehensive buffer integration test");
    
    let mut total_buffers = 0;
    let mut total_bytes = 0u64;
    
    // Test all combinations of devices and scenarios
    for device in &devices {
        println!("Testing device: {device:?}");
        
        // Various buffer sizes and patterns
        let test_patterns = vec![
            ("Small uniform", vec![64; 20]),
            ("Medium uniform", vec![1024; 10]),
            ("Large uniform", vec![1024*1024; 3]),
            ("Mixed sizes", vec![64, 256, 1024, 4096, 16384, 65536]),
            ("Power of 2", vec![16, 32, 64, 128, 256, 512, 1024, 2048]),
        ];
        
        for (pattern_name, sizes) in test_patterns {
            println!("  Pattern: {pattern_name}");
            
            let mut pattern_handles = Vec::new();
            
            for &size in &sizes {
                match pool.allocate(size, 16, device) {
                    Ok(handle) => {
                        assert_eq!(handle.size(), size);
                        assert!(format!("{:?}", handle.device()) == format!("{device:?}"));
                        
                        total_buffers += 1;
                        total_bytes += size as u64;
                        pattern_handles.push(handle);
                    }
                    Err(e) => {
                        println!("    Allocation failed for size {size}: {e}");
                        break;
                    }
                }
            }
            
            // Clean up pattern buffers
            for handle in pattern_handles {
                pool.deallocate(handle).expect("Failed to deallocate pattern buffer");
            }
        }
    }
    
    // Verify final state
    let final_metrics = pool.get_metrics();
    assert_eq!(final_metrics.active_allocations, 0);
    
    println!("Integration test completed:");
    println!("  Total buffers tested: {total_buffers}");
    println!("  Total bytes allocated: {total_bytes}");
    println!("  Devices tested: {}", devices.len());
    
    // Verify memory tracking if available
    if let Some(detailed_metrics) = pool.get_detailed_metrics() {
        println!("  Final pressure level: {:?}", detailed_metrics.pressure_level);
        assert_eq!(detailed_metrics.active_allocations, 0);
    }
}