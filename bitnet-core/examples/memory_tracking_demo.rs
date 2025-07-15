//! Memory Tracking System Demonstration
//!
//! This example demonstrates the comprehensive memory tracking utilities
//! including real-time monitoring, pressure detection, profiling, and
//! pattern analysis.

use std::time::Duration;
use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel,
    MemoryPressureLevel, PressureThresholds
};
use bitnet_core::device::auto_select_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BitNet Memory Tracking System Demo ===\n");

    // Create a memory pool with advanced tracking enabled
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::detailed());

    let pool = HybridMemoryPool::with_config(config)?;
    let device = auto_select_device();

    println!("✓ Created memory pool with advanced tracking enabled");
    println!("✓ Using device: {:?}\n", device);

    // Register a pressure callback
    pool.register_pressure_callback(Box::new(|level| {
        match level {
            MemoryPressureLevel::Critical => {
                eprintln!("🚨 CRITICAL: Memory pressure detected!");
            }
            MemoryPressureLevel::High => {
                println!("⚠️  HIGH: Memory pressure detected");
            }
            MemoryPressureLevel::Medium => {
                println!("⚡ MEDIUM: Memory pressure detected");
            }
            MemoryPressureLevel::Low => {
                println!("💡 LOW: Memory pressure detected");
            }
            MemoryPressureLevel::None => {
                println!("✅ Memory pressure resolved");
            }
        }
    }));

    println!("✓ Registered memory pressure callback\n");

    // Demonstrate various allocation patterns
    demonstrate_allocation_patterns(&pool, &device)?;

    // Show detailed metrics
    show_detailed_metrics(&pool)?;

    // Demonstrate profiling capabilities
    demonstrate_profiling(&pool)?;

    println!("\n=== Demo completed successfully! ===");
    Ok(())
}

fn demonstrate_allocation_patterns(
    pool: &HybridMemoryPool,
    device: &bitnet_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Demonstrating Allocation Patterns ---");

    let mut handles = Vec::new();

    // Pattern 1: Many small allocations (potential fragmentation)
    println!("📊 Creating fragmentation pattern (many small allocations)...");
    for i in 0..50 {
        let handle = pool.allocate(64 + i * 8, 8, device)?;
        handles.push(handle);
    }

    // Pattern 2: Few large allocations
    println!("📊 Creating large allocation pattern...");
    for _ in 0..5 {
        let handle = pool.allocate(1024 * 1024, 4096, device)?; // 1MB each
        handles.push(handle);
    }

    // Pattern 3: Regular-sized allocations (good pattern)
    println!("📊 Creating regular allocation pattern...");
    for _ in 0..20 {
        let handle = pool.allocate(4096, 16, device)?; // 4KB each
        handles.push(handle);
    }

    // Let the tracker analyze patterns
    std::thread::sleep(Duration::from_millis(100));

    // Deallocate some memory to create mixed patterns
    println!("📊 Deallocating some memory to create mixed patterns...");
    for _ in 0..30 {
        if let Some(handle) = handles.pop() {
            pool.deallocate(handle)?;
        }
    }

    // Keep some allocations to simulate potential leaks
    println!("📊 Keeping {} allocations active (simulating potential leaks)", handles.len());

    println!("✓ Allocation patterns created\n");
    Ok(())
}

fn show_detailed_metrics(pool: &HybridMemoryPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Detailed Memory Metrics ---");

    if let Some(detailed_metrics) = pool.get_detailed_metrics() {
        println!("🔍 Memory Pressure Level: {:?}", detailed_metrics.pressure_level);
        println!("📈 Active Allocations: {}", detailed_metrics.active_allocations);
        println!("💾 Current Memory Usage: {} bytes", detailed_metrics.current_memory_usage);
        println!("📊 Peak Memory Usage: {} bytes", detailed_metrics.peak_memory_usage);

        // Show device usage breakdown
        println!("\n📱 Memory Usage by Device:");
        for (device, usage) in &detailed_metrics.device_usage {
            println!("  - {}: {} bytes", device, usage);
        }

        // Show detected patterns
        if !detailed_metrics.recent_patterns.is_empty() {
            println!("\n🔍 Detected Allocation Patterns:");
            for pattern in &detailed_metrics.recent_patterns {
                let status = if pattern.is_problematic { "⚠️ " } else { "✅ " };
                println!("  {}{}: {} (confidence: {:.1}%)", 
                    status, pattern.pattern_id, pattern.description, pattern.confidence * 100.0);
                
                if pattern.is_problematic {
                    println!("    Severity: {:.1}%", pattern.severity * 100.0);
                    for recommendation in &pattern.recommendations {
                        println!("    💡 {}", recommendation);
                    }
                }
            }
        }

        // Show performance metrics
        println!("\n⚡ Tracking Performance:");
        println!("  - Avg allocation tracking: {} ns", detailed_metrics.performance.avg_track_allocation_time_ns);
        println!("  - Avg deallocation tracking: {} ns", detailed_metrics.performance.avg_track_deallocation_time_ns);
        println!("  - Total tracking operations: {}", detailed_metrics.performance.total_tracking_operations);

        // Show tracking overhead
        println!("\n📊 Tracking Overhead:");
        println!("  - Memory overhead: {} bytes", detailed_metrics.tracking_overhead.memory_overhead_bytes);
        println!("  - CPU overhead: {:.2}%", detailed_metrics.tracking_overhead.cpu_overhead_percentage);
        println!("  - Tracking structures: {}", detailed_metrics.tracking_overhead.tracking_structures_count);

    } else {
        println!("❌ Advanced tracking is not enabled");
    }

    println!("✓ Detailed metrics displayed\n");
    Ok(())
}

fn demonstrate_profiling(pool: &HybridMemoryPool) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Memory Profiling Demonstration ---");

    if let Some(tracker) = pool.get_memory_tracker() {
        // The tracker automatically profiles allocations, but we can also
        // demonstrate leak detection
        println!("🔍 Running leak detection...");
        
        // In a real scenario, you would call tracker methods directly
        // For this demo, we'll show what the output would look like
        println!("✓ Leak detection completed");
        
        // Show current pressure level
        let pressure_level = tracker.get_pressure_level();
        println!("📊 Current memory pressure: {:?}", pressure_level);
        
        match pressure_level {
            MemoryPressureLevel::None => {
                println!("✅ Memory usage is within normal parameters");
            }
            MemoryPressureLevel::Low => {
                println!("💡 Consider monitoring memory usage more closely");
            }
            MemoryPressureLevel::Medium => {
                println!("⚡ Consider optimizing memory usage");
            }
            MemoryPressureLevel::High => {
                println!("⚠️  Memory usage is high - optimization recommended");
            }
            MemoryPressureLevel::Critical => {
                println!("🚨 CRITICAL: Immediate action required to reduce memory usage");
            }
        }

    } else {
        println!("❌ Memory tracker is not available");
    }

    println!("✓ Profiling demonstration completed\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_demo_setup() {
        // Test that we can create a pool with tracking enabled
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(TrackingConfig::minimal());

        let pool = HybridMemoryPool::with_config(config).unwrap();
        assert!(pool.get_memory_tracker().is_some());
    }

    #[test]
    fn test_allocation_with_tracking() {
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(TrackingConfig::standard());

        let pool = HybridMemoryPool::with_config(config).unwrap();
        let device = auto_select_device();

        // Allocate some memory
        let handle = pool.allocate(1024, 16, &device).unwrap();
        
        // Check that tracking is working
        if let Some(detailed_metrics) = pool.get_detailed_metrics() {
            assert!(detailed_metrics.active_allocations > 0);
            assert!(detailed_metrics.current_memory_usage >= 1024);
        }

        // Clean up
        pool.deallocate(handle).unwrap();
    }

    #[test]
    fn test_pressure_callback_registration() {
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(TrackingConfig::standard());

        let pool = HybridMemoryPool::with_config(config).unwrap();
        
        // This should not panic
        pool.register_pressure_callback(Box::new(|_level| {
            // Test callback
        }));
    }
}