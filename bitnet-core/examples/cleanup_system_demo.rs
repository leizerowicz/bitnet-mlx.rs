//! Comprehensive Cleanup System Demo
//!
//! This example demonstrates the full capabilities of the BitNet cleanup system,
//! including automatic and manual cleanup operations, different strategies,
//! device-specific cleanup, and metrics collection.

use std::sync::Arc;
use std::time::Duration;
use std::thread;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, CleanupManager, CleanupConfig, CleanupResult,
    CleanupStrategyType, CleanupPriority, TrackingConfig, TrackingLevel
};
use bitnet_core::device::auto_select_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧹 BitNet Cleanup System Demo");
    println!("==============================\n");

    // 1. Create memory pool with advanced tracking enabled
    println!("📊 Setting up memory pool with advanced tracking...");
    let mut pool_config = MemoryPoolConfig::default();
    pool_config.enable_advanced_tracking = true;
    pool_config.tracking_config = Some(TrackingConfig {
        level: TrackingLevel::Detailed,
        enable_pressure_monitoring: true,
        enable_performance_metrics: true,
        enable_timeline_tracking: true,
        enable_pattern_analysis: true,
        ..Default::default()
    });

    let pool = Arc::new(HybridMemoryPool::with_config(pool_config)?);
    println!("✅ Memory pool created with advanced tracking\n");

    // 2. Create cleanup manager with comprehensive configuration
    println!("🔧 Setting up cleanup manager...");
    let cleanup_config = CleanupConfig::aggressive(); // Use aggressive cleanup for demo
    let cleanup_manager = CleanupManager::new(cleanup_config, pool.clone())?;
    println!("✅ Cleanup manager created with aggressive configuration\n");

    // 3. Start automatic cleanup scheduler
    println!("⏰ Starting automatic cleanup scheduler...");
    cleanup_manager.start_scheduler()?;
    println!("✅ Cleanup scheduler started\n");

    // 4. Demonstrate memory allocation and automatic cleanup
    println!("💾 Demonstrating memory allocation and automatic cleanup...");
    demonstrate_automatic_cleanup(&pool, &cleanup_manager)?;

    // 5. Demonstrate manual cleanup operations
    println!("\n🔧 Demonstrating manual cleanup operations...");
    demonstrate_manual_cleanup(&pool, &cleanup_manager)?;

    // 6. Demonstrate device-specific cleanup
    println!("\n🖥️  Demonstrating device-specific cleanup...");
    demonstrate_device_cleanup(&pool, &cleanup_manager)?;

    // 7. Demonstrate selective cleanup
    println!("\n🎯 Demonstrating selective cleanup...");
    demonstrate_selective_cleanup(&pool, &cleanup_manager)?;

    // 8. Demonstrate pool compaction
    println!("\n📦 Demonstrating pool compaction...");
    demonstrate_pool_compaction(&cleanup_manager)?;

    // 9. Show cleanup metrics and statistics
    println!("\n📈 Cleanup metrics and statistics:");
    show_cleanup_metrics(&cleanup_manager);

    // 10. Stop cleanup scheduler
    println!("\n⏹️  Stopping cleanup scheduler...");
    cleanup_manager.stop_scheduler()?;
    println!("✅ Cleanup scheduler stopped");

    println!("\n🎉 Cleanup system demo completed successfully!");
    Ok(())
}

fn demonstrate_automatic_cleanup(
    pool: &Arc<HybridMemoryPool>,
    cleanup_manager: &CleanupManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    
    println!("  📝 Allocating memory to trigger automatic cleanup...");
    
    // Allocate various sizes of memory to create pressure
    let mut handles = Vec::new();
    for i in 0..20 {
        let size = (i + 1) * 1024; // 1KB to 20KB
        let handle = pool.allocate(size, 16, &device)?;
        handles.push(handle);
        
        if i % 5 == 0 {
            println!("    Allocated {} blocks, total: {} KB", i + 1, (i + 1) * (i + 2) / 2);
        }
    }
    
    // Wait for automatic cleanup to potentially trigger
    println!("  ⏳ Waiting for automatic cleanup (5 seconds)...");
    thread::sleep(Duration::from_secs(5));
    
    // Check if cleanup occurred
    let stats = cleanup_manager.get_cleanup_stats();
    if stats.total_operations > 0 {
        println!("  ✅ Automatic cleanup triggered! {} operations performed", stats.total_operations);
    } else {
        println!("  ℹ️  No automatic cleanup triggered (normal for low memory pressure)");
    }
    
    // Clean up handles
    for handle in handles {
        pool.deallocate(handle)?;
    }
    
    Ok(())
}

fn demonstrate_manual_cleanup(
    pool: &Arc<HybridMemoryPool>,
    cleanup_manager: &CleanupManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    
    println!("  🔧 Performing manual force cleanup...");
    let result = cleanup_manager.force_cleanup()?;
    
    println!("    Strategy used: {:?}", result.strategy_used);
    println!("    Bytes freed: {} bytes", result.bytes_freed);
    println!("    Allocations cleaned: {}", result.allocations_cleaned);
    println!("    Duration: {:?}", result.duration);
    println!("    Success: {}", result.success);
    
    if !result.success {
        if let Some(error) = &result.error_message {
            println!("    Error: {}", error);
        }
    }
    
    Ok(())
}

fn demonstrate_device_cleanup(
    pool: &Arc<HybridMemoryPool>,
    cleanup_manager: &CleanupManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    
    println!("  🖥️  Performing device-specific cleanup for: {:?}", device);
    let result = cleanup_manager.cleanup_device(&device)?;
    
    println!("    Strategy used: {:?}", result.strategy_used);
    println!("    Bytes freed: {} bytes", result.bytes_freed);
    println!("    Duration: {:?}", result.duration);
    
    // Show device-specific metadata
    if !result.metadata.is_empty() {
        println!("    Device-specific metadata:");
        for (key, value) in &result.metadata {
            println!("      {}: {}", key, value);
        }
    }
    
    Ok(())
}

fn demonstrate_selective_cleanup(
    pool: &Arc<HybridMemoryPool>,
    cleanup_manager: &CleanupManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = auto_select_device();
    
    println!("  🎯 Performing selective cleanup (age-based)...");
    let result = cleanup_manager.cleanup_selective(
        Some(Duration::from_millis(100)), // Min age
        Some(1024),                       // Min size
        Some(&device),                    // Device filter
    )?;
    
    println!("    Strategy used: {:?}", result.strategy_used);
    println!("    Bytes freed: {} bytes", result.bytes_freed);
    println!("    Allocations cleaned: {}", result.allocations_cleaned);
    println!("    Duration: {:?}", result.duration);
    
    Ok(())
}

fn demonstrate_pool_compaction(
    cleanup_manager: &CleanupManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📦 Performing pool compaction...");
    let result = cleanup_manager.compact_pools()?;
    
    println!("    Bytes compacted: {} bytes", result.bytes_compacted);
    println!("    Blocks consolidated: {}", result.blocks_consolidated);
    println!("    Fragmentation before: {:.1}%", result.fragmentation_before * 100.0);
    println!("    Fragmentation after: {:.1}%", result.fragmentation_after * 100.0);
    println!("    Duration: {:?}", result.duration);
    println!("    Success: {}", result.success);
    
    if result.success {
        let improvement = (result.fragmentation_before - result.fragmentation_after) * 100.0;
        println!("    Fragmentation improvement: {:.1}%", improvement);
    }
    
    Ok(())
}

fn show_cleanup_metrics(cleanup_manager: &CleanupManager) {
    let stats = cleanup_manager.get_cleanup_stats();
    
    println!("  📊 Overall Statistics:");
    println!("    Total operations: {}", stats.total_operations);
    println!("    Successful operations: {}", stats.successful_operations);
    println!("    Failed operations: {}", stats.failed_operations);
    println!("    Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!("    Total bytes freed: {} bytes", stats.total_bytes_freed);
    println!("    Total allocations cleaned: {}", stats.total_allocations_cleaned);
    println!("    Average efficiency: {:.2} bytes/ms", stats.average_efficiency);
    
    if !stats.device_stats.is_empty() {
        println!("\n  🖥️  Device Statistics:");
        for (device_type, device_stats) in &stats.device_stats {
            println!("    {}:", device_type);
            println!("      Operations: {}", device_stats.operations);
            println!("      Bytes freed: {} bytes", device_stats.bytes_freed);
            println!("      Efficiency: {:.2} bytes/op", device_stats.average_efficiency);
        }
    }
    
    if !stats.strategy_stats.is_empty() {
        println!("\n  🎯 Strategy Statistics:");
        for (strategy_type, strategy_stats) in &stats.strategy_stats {
            println!("    {:?}:", strategy_type);
            println!("      Operations: {}", strategy_stats.operations);
            println!("      Bytes freed: {} bytes", strategy_stats.bytes_freed);
            println!("      Efficiency: {:.2} bytes/op", strategy_stats.average_efficiency);
        }
    }
    
    // Show operation history
    let history = cleanup_manager.get_operation_history();
    if !history.is_empty() {
        println!("\n  📜 Recent Operations (last {}):", history.len().min(5));
        for (i, operation) in history.iter().rev().take(5).enumerate() {
            println!("    {}. Strategy: {:?}, Bytes: {}, Success: {}", 
                     i + 1, operation.strategy_type, operation.bytes_freed, operation.success);
        }
    }
    
    // Show scheduler status
    println!("\n  ⏰ Scheduler Status:");
    println!("    Running: {}", cleanup_manager.is_scheduler_running());
    
    // Show configuration summary
    let config = cleanup_manager.get_config();
    println!("\n  ⚙️  Configuration Summary:");
    println!("    Automatic cleanup: {}", config.policy.enable_automatic_cleanup);
    println!("    Manual cleanup: {}", config.policy.enable_manual_cleanup);
    println!("    Default strategy: {:?}", config.policy.default_strategy);
    println!("    Max cleanup duration: {:?}", config.policy.max_cleanup_duration);
    println!("    Scheduler enabled: {}", config.scheduler.enabled);
    println!("    Base interval: {:?}", config.scheduler.base_interval);
    
    println!("\n  🎛️  Feature Flags:");
    println!("    Idle cleanup: {}", config.features.enable_idle_cleanup);
    println!("    Pressure cleanup: {}", config.features.enable_pressure_cleanup);
    println!("    Periodic cleanup: {}", config.features.enable_periodic_cleanup);
    println!("    Device cleanup: {}", config.features.enable_device_cleanup);
    println!("    Generational cleanup: {}", config.features.enable_generational_cleanup);
    println!("    Smart cleanup: {}", config.features.enable_smart_cleanup);
    println!("    Cleanup metrics: {}", config.features.enable_cleanup_metrics);
    println!("    Emergency cleanup: {}", config.features.enable_emergency_cleanup);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_demo_components() {
        // Test that we can create the basic components
        let pool_config = MemoryPoolConfig::default();
        let pool = Arc::new(HybridMemoryPool::with_config(pool_config).unwrap());
        
        let cleanup_config = CleanupConfig::default();
        let cleanup_manager = CleanupManager::new(cleanup_config, pool).unwrap();
        
        assert!(!cleanup_manager.is_scheduler_running());
        
        let stats = cleanup_manager.get_cleanup_stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_cleanup_configurations() {
        // Test different cleanup configurations
        let configs = [
            CleanupConfig::default(),
            CleanupConfig::minimal(),
            CleanupConfig::aggressive(),
            CleanupConfig::debug(),
        ];
        
        for config in configs {
            assert!(config.validate().is_ok());
        }
    }
}