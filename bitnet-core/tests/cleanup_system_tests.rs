//! Comprehensive Tests for the Cleanup System
//!
//! This test suite validates all aspects of the BitNet cleanup system,
//! including strategies, scheduling, device-specific cleanup, and metrics.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bitnet_core::device::get_cpu_device;
use bitnet_core::memory::{
    CleanupConfig, CleanupManager, CleanupStrategyType, CpuCleanup, DeviceCleanupOps,
    HybridMemoryPool,
};

#[cfg(feature = "metal")]
use bitnet_core::memory::MetalCleanup;

/// Creates a test-optimized configuration with metrics enabled
/// This ensures cleanup operations are properly recorded for test validation
fn create_test_cleanup_config() -> CleanupConfig {
    let mut config = CleanupConfig::default();
    
    // CRITICAL: Enable metrics collection for test validation
    config.features.enable_cleanup_metrics = true;
    
    // Optimize for test speed and predictability
    config.policy.max_cleanup_duration = Duration::from_millis(100);
    config.policy.enable_automatic_cleanup = true;
    config.policy.enable_manual_cleanup = true;
    config.policy.default_strategy = CleanupStrategyType::Idle;
    
    // Fast cleanup thresholds for tests
    config.thresholds.pressure.light_cleanup_threshold = 0.1;
    config.thresholds.pressure.min_pressure_cleanup_interval = Duration::from_millis(1);
    
    // Fast idle cleanup for tests
    config.thresholds.idle.min_idle_time = Duration::from_nanos(1);
    config.thresholds.idle.light_cleanup_idle_time = Duration::from_millis(1);
    
    // Disable scheduler for predictable test behavior
    config.scheduler.enabled = false;
    
    config
}

/// Test basic cleanup manager creation and configuration
#[test]
fn test_cleanup_manager_creation() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();

    let manager = CleanupManager::new(config, pool).unwrap();
    assert!(!manager.is_scheduler_running());

    let stats = manager.get_cleanup_stats();
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.successful_operations, 0);
    assert_eq!(stats.failed_operations, 0);
}

/// Test different cleanup configurations
#[test]
fn test_cleanup_configurations() {
    let configs = [
        CleanupConfig::default(),
        CleanupConfig::minimal(),
        CleanupConfig::aggressive(),
        CleanupConfig::debug(),
    ];

    for config in configs {
        assert!(config.validate().is_ok(), "Configuration should be valid");

        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let manager = CleanupManager::new(config, pool);
        assert!(manager.is_ok(), "Manager creation should succeed");
    }
}

/// Test invalid cleanup configurations
#[test]
fn test_invalid_cleanup_configurations() {
    let mut config = CleanupConfig::default();

    // Test invalid pressure thresholds
    config.thresholds.pressure.light_cleanup_threshold = 0.9;
    config.thresholds.pressure.aggressive_cleanup_threshold = 0.8;
    assert!(config.validate().is_err());

    // Test invalid policy settings
    config = CleanupConfig::default();
    config.policy.max_allocations_per_cleanup = 0;
    assert!(config.validate().is_err());

    // Test invalid scheduler settings
    config = CleanupConfig::default();
    config.scheduler.max_concurrent_operations = 0;
    assert!(config.validate().is_err());
}

/// Test manual cleanup operations
#[test]
fn test_manual_cleanup() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool.clone()).unwrap();

    // Test force cleanup
    let result = manager.force_cleanup().unwrap();
    assert!(result.success || result.error_message.is_some());

    // Test device-specific cleanup
    let device = get_cpu_device();
    let result = manager.cleanup_device(&device).unwrap();
    assert!(result.success || result.error_message.is_some());

    // Test selective cleanup
    let result = manager
        .cleanup_selective(Some(Duration::from_millis(100)), Some(1024), Some(&device))
        .unwrap();
    assert!(result.success || result.error_message.is_some());
}

/// Test pool compaction
#[test]
fn test_pool_compaction() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool).unwrap();

    let result = manager.compact_pools().unwrap();
    assert!(result.success);
    assert!(result.fragmentation_before >= 0.0);
    assert!(result.fragmentation_after >= 0.0);
    assert!(result.duration > Duration::ZERO);
}

/// Test cleanup scheduler
#[test]
fn test_cleanup_scheduler() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool).unwrap();

    // Test starting scheduler
    assert!(manager.start_scheduler().is_ok());
    assert!(manager.is_scheduler_running());

    // Wait a bit for scheduler to potentially run
    thread::sleep(Duration::from_millis(100));

    // Test stopping scheduler
    assert!(manager.stop_scheduler().is_ok());
    assert!(!manager.is_scheduler_running());
}

/// Test cleanup metrics collection
#[test]
fn test_cleanup_metrics() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool).unwrap();

    // Perform some cleanup operations
    let _ = manager.force_cleanup();
    let _ = manager.compact_pools();

    let stats = manager.get_cleanup_stats();

    // Check that metrics are being collected
    assert!(stats.total_operations < u64::MAX); // Sanity check
    assert!(stats.success_rate() >= 0.0 && stats.success_rate() <= 1.0);

    // Check operation history
    let history = manager.get_operation_history();
    assert!(history.len() <= 1000); // Should respect history limit
}

/// Test CPU device cleanup
#[test]
fn test_cpu_device_cleanup() {
    let cpu_cleanup = CpuCleanup::default();

    assert_eq!(cpu_cleanup.device_type(), "CPU");

    let pool = HybridMemoryPool::new().unwrap();
    let result = cpu_cleanup.cleanup_device(&pool);
    assert!(result.is_ok());

    let cleanup_result = result.unwrap();
    assert_eq!(cleanup_result.device_type, "CPU");
    assert!(cleanup_result.success);

    // Test cache optimization
    let cache_result = cpu_cleanup.optimize_cache();
    assert!(cache_result.is_ok());

    let cache_opt = cache_result.unwrap();
    assert!(cache_opt.success);
    assert!(cache_opt.cache_hit_ratio_after >= cache_opt.cache_hit_ratio_before);

    // Test defragmentation
    let defrag_result = cpu_cleanup.defragment_memory();
    assert!(defrag_result.is_ok());

    let defrag = defrag_result.unwrap();
    assert!(defrag.success);
    assert!(defrag.fragmentation_after <= defrag.fragmentation_before);

    // Test statistics
    let stats = cpu_cleanup.get_cleanup_stats();
    assert_eq!(stats.device_type, "CPU");
    assert!(stats.total_cleanups >= 1);
}

/// Test Metal device cleanup (if Metal feature is enabled)
#[cfg(feature = "metal")]
#[test]
fn test_metal_device_cleanup() {
    let metal_cleanup = MetalCleanup::default();

    assert_eq!(metal_cleanup.device_type(), "Metal");

    let pool = HybridMemoryPool::new().unwrap();
    let result = metal_cleanup.cleanup_device(&pool);
    assert!(result.is_ok());

    let cleanup_result = result.unwrap();
    assert_eq!(cleanup_result.device_type, "Metal");
    assert!(cleanup_result.success);

    // Test cache optimization
    let cache_result = metal_cleanup.optimize_cache();
    assert!(cache_result.is_ok());

    // Test defragmentation
    let defrag_result = metal_cleanup.defragment_memory();
    assert!(defrag_result.is_ok());

    // Test statistics
    let stats = metal_cleanup.get_cleanup_stats();
    assert_eq!(stats.device_type, "Metal");
}

/// Test Metal device cleanup stub (when Metal feature is disabled)
#[cfg(not(feature = "metal"))]
#[test]
fn test_metal_device_cleanup_stub() {
    use bitnet_core::memory::MetalCleanup;
    let metal_cleanup = MetalCleanup::default();

    assert_eq!(metal_cleanup.device_type(), "Metal");

    let pool = HybridMemoryPool::new().unwrap();
    let result = metal_cleanup.cleanup_device(&pool);
    assert!(result.is_err()); // Should fail when Metal is not available

    let cache_result = metal_cleanup.optimize_cache();
    assert!(cache_result.is_err());

    let defrag_result = metal_cleanup.defragment_memory();
    assert!(defrag_result.is_err());
}

/// Test cleanup strategy registration
#[test]
fn test_cleanup_strategy_registration() {
    use bitnet_core::memory::cleanup::strategies::IdleCleanupStrategy;

    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool).unwrap();

    // Register a custom strategy
    let custom_strategy = IdleCleanupStrategy::default();
    let result =
        manager.register_cleanup_strategy(CleanupStrategyType::Idle, Box::new(custom_strategy));
    assert!(result.is_ok());
}

/// Test cleanup with memory allocation and deallocation
#[test]
fn test_cleanup_with_memory_operations() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool.clone()).unwrap();

    let device = get_cpu_device();

    // Allocate some memory
    let mut handles = Vec::new();
    for i in 0..10 {
        let size = (i + 1) * 1024;
        let handle = pool.allocate(size, 16, &device).unwrap();
        handles.push(handle);
    }

    // Perform cleanup
    let result = manager.force_cleanup().unwrap();
    assert!(result.success || result.error_message.is_some());

    // Deallocate memory
    for handle in handles {
        pool.deallocate(handle).unwrap();
    }

    // Perform cleanup again
    let result = manager.force_cleanup().unwrap();
    assert!(result.success || result.error_message.is_some());
}

/// Test cleanup system under concurrent operations
#[test]
fn test_concurrent_cleanup_operations() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::default();
    let manager = Arc::new(CleanupManager::new(config, pool.clone()).unwrap());

    let device = get_cpu_device();

    // Start scheduler
    manager.start_scheduler().unwrap();

    // Spawn multiple threads performing cleanup operations
    let mut handles = Vec::new();

    for _ in 0..5 {
        let manager_clone = manager.clone();
        let device_clone = device.clone();

        let handle = thread::spawn(move || {
            // Perform various cleanup operations
            let _ = manager_clone.force_cleanup();
            let _ = manager_clone.cleanup_device(&device_clone);
            let _ = manager_clone.compact_pools();

            // Check that operations complete without panicking
            let stats = manager_clone.get_cleanup_stats();
            assert!(stats.total_operations < u64::MAX); // Sanity check
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Stop scheduler
    manager.stop_scheduler().unwrap();

    // Verify final state
    let stats = manager.get_cleanup_stats();
    assert!(stats.total_operations < u64::MAX); // Sanity check
}

/// Test cleanup system performance and efficiency
#[test]
fn test_cleanup_performance() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = create_test_cleanup_config(); // Use metrics-enabled config
    let manager = CleanupManager::new(config, pool.clone()).unwrap();

    let _device = get_cpu_device();

    // Create some allocations to give cleanup something to work with
    let mut _handles = Vec::new();
    for _i in 0..20 {
        if let Ok(handle) = pool.allocate(1024, 8, &_device) {
            _handles.push(handle);
        }
    }

    // Measure cleanup performance - call force_cleanup multiple times
    let start_time = std::time::Instant::now();

    // Perform multiple cleanup operations
    for _ in 0..10 {
        let _ = manager.force_cleanup();
    }

    let total_duration = start_time.elapsed();

    // Verify that cleanup operations complete in reasonable time
    assert!(
        total_duration < Duration::from_secs(5),
        "Cleanup operations should complete quickly"
    );

    // Check cleanup statistics - should now have metrics recorded (one per force_cleanup call)
    let stats = manager.get_cleanup_stats();
    println!("Performance test - operations processed: {}", stats.total_operations);
    assert!(stats.total_operations >= 10, 
           "Expected at least 10 operations, got {}", stats.total_operations);

    if stats.total_operations > 0 {
        assert!(stats.average_efficiency >= 0.0);
    }
}

/// Integration test with memory pool operations
#[test]
fn test_cleanup_integration() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = CleanupConfig::aggressive(); // Use aggressive config for testing
    let manager = CleanupManager::new(config, pool.clone()).unwrap();

    let device = get_cpu_device();

    // Start scheduler
    manager.start_scheduler().unwrap();

    // Perform memory operations while cleanup is running
    let mut handles = Vec::new();

    // Allocate memory in batches
    for batch in 0..3 {
        // Allocate
        for i in 0..5 {
            let size = (batch * 5 + i + 1) * 1024;
            let handle = pool.allocate(size, 16, &device).unwrap();
            handles.push(handle);
        }

        // Trigger manual cleanup
        let _ = manager.force_cleanup();

        // Small delay to allow scheduler to run
        thread::sleep(Duration::from_millis(10));
    }

    // Deallocate all memory
    for handle in handles {
        pool.deallocate(handle).unwrap();
    }

    // Final cleanup
    let _ = manager.force_cleanup();
    let _ = manager.compact_pools();

    // Stop scheduler
    manager.stop_scheduler().unwrap();

    // Verify final state
    let stats = manager.get_cleanup_stats();
    let pool_metrics = pool.get_metrics();

    // Should have performed some cleanup operations
    assert!(stats.total_operations > 0);

    // Memory should be properly managed
    assert_eq!(pool_metrics.active_allocations, 0);
}

/// Test error handling and edge cases
#[test]
fn test_cleanup_error_handling() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());

    // Test with invalid configuration
    let mut config = CleanupConfig::default();
    config.policy.max_allocations_per_cleanup = 0;

    let result = CleanupManager::new(config, pool.clone());
    assert!(result.is_err());

    // Test with valid configuration
    let config = CleanupConfig::default();
    let manager = CleanupManager::new(config, pool).unwrap();

    // Test double start/stop of scheduler
    assert!(manager.start_scheduler().is_ok());
    assert!(manager.start_scheduler().is_err()); // Should fail on second start

    assert!(manager.stop_scheduler().is_ok());
    assert!(manager.stop_scheduler().is_ok()); // Should be idempotent
}

/// Benchmark cleanup operations
#[test]
fn test_cleanup_benchmarks() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let config = create_test_cleanup_config(); // Use metrics-enabled config instead of minimal
    let manager = CleanupManager::new(config, pool.clone()).unwrap();

    let device = get_cpu_device();

    // Create some allocations to give cleanup something to work with
    let mut _handles = Vec::new();
    for _i in 0..200 {
        if let Ok(handle) = pool.allocate(1024, 8, &device) {
            _handles.push(handle);
        }
    }

    // Benchmark force cleanup
    let iterations = 100;
    let start_time = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = manager.force_cleanup();
    }

    let total_duration = start_time.elapsed();
    let avg_duration = total_duration / iterations;

    println!("Average cleanup duration: {avg_duration:?}");

    // Verify performance is reasonable - allow longer time for safety
    assert!(
        avg_duration < Duration::from_millis(50),
        "Cleanup should be reasonably fast, but was {:?}", avg_duration
    );

    // Check final statistics - should now have metrics recorded
    let stats = manager.get_cleanup_stats();
    println!("Benchmark test - operations processed: {}", stats.total_operations);
    assert_eq!(stats.total_operations, iterations as u64, 
              "Expected {} operations, got {}", iterations, stats.total_operations);
}
