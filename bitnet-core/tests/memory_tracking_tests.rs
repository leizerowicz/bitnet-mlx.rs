//! Comprehensive tests for the memory tracking system
//!
//! This test suite validates all aspects of the memory tracking utilities
//! including performance overhead validation.

use std::sync::Arc;
use std::time::{Duration, Instant};
use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    MemoryPressureLevel, MemoryTracker, MemoryProfiler, AllocationTimeline,
    PatternAnalyzer, PressureThresholds
};
use bitnet_core::device::get_cpu_device;

#[test]
fn test_memory_pool_with_tracking_integration() {
    // Test that memory pool integrates correctly with tracking
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());

    let pool = HybridMemoryPool::with_config(config).unwrap();
    let device = get_cpu_device();

    // Verify tracker is available
    assert!(pool.get_memory_tracker().is_some());

    // Perform allocations and verify tracking
    let handle1 = pool.allocate(1024, 16, &device).unwrap();
    let handle2 = pool.allocate(2048, 32, &device).unwrap();

    // Check detailed metrics
    if let Some(metrics) = pool.get_detailed_metrics() {
        assert!(metrics.active_allocations >= 2);
        assert!(metrics.current_memory_usage >= 3072);
        assert_eq!(metrics.pressure_level, MemoryPressureLevel::None);
    } else {
        panic!("Detailed metrics should be available");
    }

    // Clean up
    pool.deallocate(handle1).unwrap();
    pool.deallocate(handle2).unwrap();

    // Verify deallocation tracking
    if let Some(metrics) = pool.get_detailed_metrics() {
        assert_eq!(metrics.active_allocations, 0);
    }
}

#[test]
fn test_tracking_configuration_levels() {
    // Test different tracking levels
    let configs = vec![
        TrackingConfig::minimal(),
        TrackingConfig::standard(),
        TrackingConfig::detailed(),
        TrackingConfig::debug(),
        TrackingConfig::production(),
    ];

    for config in configs {
        let tracker = MemoryTracker::new(config.clone()).unwrap();
        let metrics = tracker.get_detailed_metrics();
        
        // All configurations should provide basic metrics
        assert_eq!(metrics.active_allocations, 0);
        assert_eq!(metrics.current_memory_usage, 0);
        assert_eq!(metrics.pressure_level, MemoryPressureLevel::None);
    }
}

#[test]
fn test_memory_pressure_detection() {
    let mut thresholds = PressureThresholds::default();
    thresholds.low_pressure_threshold = 0.1; // Very low threshold for testing
    thresholds.medium_pressure_threshold = 0.2;
    thresholds.high_pressure_threshold = 0.3;
    thresholds.critical_pressure_threshold = 0.4;

    let mut config = TrackingConfig::standard();
    config.pressure_thresholds = thresholds;

    let tracker = MemoryTracker::new(config).unwrap();
    
    // Initially should be no pressure
    assert_eq!(tracker.get_pressure_level(), MemoryPressureLevel::None);

    // Test pressure callback registration
    let callback_called = std::sync::Arc::new(std::sync::Mutex::new(false));
    let callback_called_clone = callback_called.clone();
    
    tracker.register_pressure_callback(Box::new(move |_level| {
        *callback_called_clone.lock().unwrap() = true;
    }));

    // Note: In a real test environment, we would need to simulate high memory usage
    // to trigger pressure detection. This would require more complex setup.
}

#[test]
fn test_memory_profiler_functionality() {
    let profiler = MemoryProfiler::new(Default::default()).unwrap();
    
    // Test profiling session
    profiler.start_profiling();
    
    // Create mock allocations for profiling
    let allocation_info = bitnet_core::memory::tracking::AllocationInfo {
        id: bitnet_core::memory::tracking::AllocationId::new(1),
        size: 1024,
        alignment: 16,
        device_type: "CPU".to_string(),
        timestamp: std::time::SystemTime::now(),
        elapsed: Duration::from_millis(100),
        stack_trace: None,
        pool_type: "SmallBlock".to_string(),
        is_active: true,
    };
    
    profiler.record_allocation(allocation_info.clone());
    profiler.record_deallocation(allocation_info);
    
    let report = profiler.stop_profiling();
    
    // Verify report contents
    assert_eq!(report.total_allocations, 1);
    assert_eq!(report.total_deallocations, 1);
    assert_eq!(report.active_allocations, 0);
    assert!(report.session_duration > Duration::ZERO);
}

#[test]
fn test_allocation_timeline_tracking() {
    let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
    
    // Add some events
    let event1 = bitnet_core::memory::tracking::timeline::AllocationEvent::Allocation {
        id: bitnet_core::memory::tracking::AllocationId::new(1),
        size: 1024,
        device_type: "CPU".to_string(),
        timestamp: std::time::SystemTime::now(),
    };
    
    let event2 = bitnet_core::memory::tracking::timeline::AllocationEvent::Deallocation {
        id: bitnet_core::memory::tracking::AllocationId::new(1),
        size: 1024,
        device_type: "CPU".to_string(),
        timestamp: std::time::SystemTime::now(),
    };
    
    timeline.add_event(event1);
    timeline.add_event(event2);
    
    let stats = timeline.get_statistics();
    assert_eq!(stats.total_events, 2);
    assert_eq!(stats.allocation_events, 1);
    assert_eq!(stats.deallocation_events, 1);
    
    // Test timeline query
    let query = bitnet_core::memory::tracking::timeline::TimelineQuery {
        event_type: Some("Allocation".to_string()),
        ..Default::default()
    };
    
    let analysis = timeline.query(query);
    assert_eq!(analysis.entries.len(), 1);
}

#[test]
fn test_pattern_analysis() {
    let analyzer = PatternAnalyzer::new(Default::default());
    
    // Create similar allocations to trigger pattern detection
    for i in 0..10 {
        let allocation = bitnet_core::memory::tracking::AllocationInfo {
            id: bitnet_core::memory::tracking::AllocationId::new(i),
            size: 1024, // Same size to create a pattern
            alignment: 16,
            device_type: "CPU".to_string(),
            timestamp: std::time::SystemTime::now(),
            elapsed: Duration::from_millis(100),
            stack_trace: None,
            pool_type: "SmallBlock".to_string(),
            is_active: true,
        };
        analyzer.record_allocation(allocation);
    }
    
    let patterns = analyzer.get_patterns();
    assert!(!patterns.is_empty(), "Should detect allocation patterns");
    
    let report = analyzer.generate_report();
    assert!(report.patterns.len() > 0);
    assert!(!report.summary.insights.is_empty());
}

#[test]
fn test_leak_detection() {
    let mut config = bitnet_core::memory::tracking::profiler::ProfilingConfig::default();
    config.leak_detection_threshold = Duration::from_millis(10); // Very short for testing
    
    let profiler = MemoryProfiler::new(config).unwrap();
    profiler.start_profiling();
    
    // Create an old allocation that looks like a leak
    let old_allocation = bitnet_core::memory::tracking::AllocationInfo {
        id: bitnet_core::memory::tracking::AllocationId::new(1),
        size: 1024 * 1024, // 1MB
        alignment: 16,
        device_type: "CPU".to_string(),
        timestamp: std::time::SystemTime::now() - Duration::from_secs(1), // Old
        elapsed: Duration::from_secs(1),
        stack_trace: None,
        pool_type: "LargeBlock".to_string(),
        is_active: true, // Still active (not deallocated)
    };
    
    profiler.record_allocation(old_allocation);
    
    // Wait for leak detection threshold
    std::thread::sleep(Duration::from_millis(20));
    
    let leak_report = profiler.detect_leaks();
    assert!(leak_report.leak_count > 0, "Should detect potential leaks");
    assert!(leak_report.total_leaked_bytes >= 1024 * 1024);
}

#[test]
fn test_performance_overhead_validation() {
    // This test validates that tracking overhead is under 5%
    let device = get_cpu_device();
    
    // Test without tracking
    let pool_no_tracking = HybridMemoryPool::new().unwrap();
    let start_time = Instant::now();
    
    let mut handles_no_tracking = Vec::new();
    for i in 0..1000 {
        let handle = pool_no_tracking.allocate(1024 + i, 16, &device).unwrap();
        handles_no_tracking.push(handle);
    }
    
    for handle in handles_no_tracking {
        pool_no_tracking.deallocate(handle).unwrap();
    }
    
    let time_no_tracking = start_time.elapsed();
    
    // Test with tracking
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());
    
    let pool_with_tracking = HybridMemoryPool::with_config(config).unwrap();
    let start_time = Instant::now();
    
    let mut handles_with_tracking = Vec::new();
    for i in 0..1000 {
        let handle = pool_with_tracking.allocate(1024 + i, 16, &device).unwrap();
        handles_with_tracking.push(handle);
    }
    
    for handle in handles_with_tracking {
        pool_with_tracking.deallocate(handle).unwrap();
    }
    
    let time_with_tracking = start_time.elapsed();
    
    // Calculate overhead percentage
    let overhead_ratio = time_with_tracking.as_nanos() as f64 / time_no_tracking.as_nanos() as f64;
    let overhead_percentage = (overhead_ratio - 1.0) * 100.0;
    
    println!("Performance overhead: {:.2}%", overhead_percentage);
    
    // Validate that overhead is under 5%
    assert!(overhead_percentage < 5.0, 
        "Tracking overhead ({:.2}%) exceeds 5% threshold", overhead_percentage);
    
    // Also check the tracking system's own overhead reporting
    if let Some(metrics) = pool_with_tracking.get_detailed_metrics() {
        assert!(metrics.tracking_overhead.cpu_overhead_percentage < 5.0,
            "Self-reported CPU overhead ({:.2}%) exceeds 5% threshold", 
            metrics.tracking_overhead.cpu_overhead_percentage);
    }
}

#[test]
fn test_tracking_memory_usage() {
    // Test that tracking system itself doesn't use excessive memory
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::detailed());
    
    let pool = HybridMemoryPool::with_config(config).unwrap();
    let device = get_cpu_device();
    
    // Perform many allocations to build up tracking data
    let mut handles = Vec::new();
    for i in 0..100 {
        let handle = pool.allocate(1024 + i * 10, 16, &device).unwrap();
        handles.push(handle);
    }
    
    // Check tracking overhead
    if let Some(metrics) = pool.get_detailed_metrics() {
        let tracking_memory = metrics.tracking_overhead.memory_overhead_bytes;
        let actual_memory = metrics.current_memory_usage;
        
        // Tracking memory should be reasonable compared to actual memory usage
        let overhead_ratio = tracking_memory as f64 / actual_memory as f64;
        assert!(overhead_ratio < 0.1, // Less than 10% memory overhead
            "Memory tracking overhead ({:.2}%) is too high", overhead_ratio * 100.0);
    }
    
    // Clean up
    for handle in handles {
        pool.deallocate(handle).unwrap();
    }
}

#[test]
fn test_concurrent_tracking() {
    // Test that tracking works correctly under concurrent access
    use std::sync::Arc;
    use std::thread;
    
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::standard());
    
    let pool = Arc::new(HybridMemoryPool::with_config(config).unwrap());
    let device = get_cpu_device();
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads doing allocations
    for thread_id in 0..4 {
        let pool_clone = pool.clone();
        let device_clone = device.clone();
        
        let handle = thread::spawn(move || {
            let mut thread_handles = Vec::new();
            
            for i in 0..25 {
                let size = 1024 + thread_id * 100 + i * 10;
                let handle = pool_clone.allocate(size, 16, &device_clone).unwrap();
                thread_handles.push(handle);
            }
            
            // Deallocate half of them
            for _ in 0..12 {
                if let Some(handle) = thread_handles.pop() {
                    pool_clone.deallocate(handle).unwrap();
                }
            }
            
            thread_handles
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads and collect remaining handles
    let mut remaining_handles = Vec::new();
    for handle in handles {
        let mut thread_handles = handle.join().unwrap();
        remaining_handles.append(&mut thread_handles);
    }
    
    // Verify tracking worked correctly
    if let Some(metrics) = pool.get_detailed_metrics() {
        assert_eq!(metrics.active_allocations, remaining_handles.len());
        assert!(metrics.current_memory_usage > 0);
    }
    
    // Clean up remaining handles
    for handle in remaining_handles {
        pool.deallocate(handle).unwrap();
    }
    
    // Verify final state
    if let Some(metrics) = pool.get_detailed_metrics() {
        assert_eq!(metrics.active_allocations, 0);
    }
}

#[test]
fn test_tracking_configuration_validation() {
    // Test that invalid configurations are rejected
    let mut config = TrackingConfig::standard();
    config.sampling_rate = 1.5; // Invalid (> 1.0)
    
    assert!(config.validate().is_err());
    
    config.sampling_rate = -0.1; // Invalid (< 0.0)
    assert!(config.validate().is_err());
    
    config.sampling_rate = 0.5; // Valid
    assert!(config.validate().is_ok());
    
    // Test pressure thresholds validation
    let mut thresholds = PressureThresholds::default();
    thresholds.high_pressure_threshold = 0.5;
    thresholds.low_pressure_threshold = 0.8; // Invalid (higher than high)
    
    assert!(thresholds.validate().is_err());
}

#[test]
fn test_tracking_system_integration() {
    // Comprehensive integration test
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::detailed());
    
    let pool = HybridMemoryPool::with_config(config).unwrap();
    let device = get_cpu_device();
    
    // Register pressure callback
    let pressure_events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pressure_events_clone = pressure_events.clone();
    
    pool.register_pressure_callback(Box::new(move |level| {
        pressure_events_clone.lock().unwrap().push(level);
    }));
    
    // Perform various allocation patterns
    let mut handles = Vec::new();
    
    // Small allocations (fragmentation pattern)
    for i in 0..20 {
        let handle = pool.allocate(64 + i * 4, 8, &device).unwrap();
        handles.push(handle);
    }
    
    // Large allocations
    for _ in 0..5 {
        let handle = pool.allocate(1024 * 1024, 4096, &device).unwrap();
        handles.push(handle);
    }
    
    // Mixed deallocations
    for _ in 0..10 {
        if let Some(handle) = handles.pop() {
            pool.deallocate(handle).unwrap();
        }
    }
    
    // Verify comprehensive tracking
    if let Some(metrics) = pool.get_detailed_metrics() {
        assert!(metrics.active_allocations > 0);
        assert!(metrics.current_memory_usage > 0);
        assert!(!metrics.recent_patterns.is_empty());
        assert!(metrics.performance.total_tracking_operations > 0);
        assert!(metrics.tracking_overhead.memory_overhead_bytes > 0);
    }
    
    // Clean up
    for handle in handles {
        pool.deallocate(handle).unwrap();
    }
    
    println!("✓ All tracking system integration tests passed");
}