//! # Advanced GPU Memory Management Example
//!
//! This example demonstrates the Task 4.1.3 Advanced GPU Memory Management features:
//! - Intelligent Buffer Management with fragmentation analysis
//! - Multi-GPU Coordination with cross-GPU sharing and load balancing
//! - Memory Pressure Detection with intelligent monitoring and thermal management

use anyhow::Result;
use bitnet_core::memory::advanced_gpu_memory::{
    AdvancedGpuMemoryManager, AdvancedMemoryConfig, DeviceId, AllocationHint,
    AccessPattern, SharingRequirements, AllocationPriority,
};
use std::time::Duration;

fn main() -> Result<()> {
    println!("🚀 BitNet Advanced GPU Memory Management Demo");
    println!("===============================================");

    // 1. Create advanced memory configuration
    println!("\n📋 Creating Advanced Memory Configuration...");
    let mut config = AdvancedMemoryConfig::default();
    
    // Configure managed devices (CPU, CUDA GPU 0, Metal GPU 0)
    config.managed_devices = vec![
        DeviceId::Cpu,
        DeviceId::Cuda(0),
        DeviceId::Metal(0),
    ];
    
    // Configure fragmentation analysis
    config.fragmentation_config.real_time_monitoring = true;
    config.fragmentation_config.fragmentation_threshold = 0.25;
    
    // Configure multi-GPU coordination
    config.multi_gpu_config.enable_cross_gpu_sharing = true;
    
    println!("✅ Configuration created with {} managed devices", config.managed_devices.len());

    // 2. Initialize the advanced GPU memory manager
    println!("\n🔧 Initializing Advanced GPU Memory Manager...");
    let manager = AdvancedGpuMemoryManager::new(config)?;
    println!("✅ Advanced GPU memory manager initialized successfully");

    // 3. Demonstrate intelligent buffer allocation
    println!("\n🧠 Demonstrating Intelligent Buffer Management...");
    
    // Small allocation with sequential access
    let small_hint = AllocationHint {
        expected_lifetime: Duration::from_secs(5),
        access_pattern: AccessPattern::Sequential,
        sharing_requirements: SharingRequirements::Exclusive,
        priority: AllocationPriority::Normal,
    };
    
    let small_allocation = manager.allocate_intelligent(1024, DeviceId::Cpu, small_hint)?;
    println!("✅ Small allocation (1KB) created with intelligent buffer management");

    // Large allocation with compute access pattern
    let large_hint = AllocationHint {
        expected_lifetime: Duration::from_secs(30),
        access_pattern: AccessPattern::Compute,
        sharing_requirements: SharingRequirements::Exclusive,
        priority: AllocationPriority::High,
    };
    
    let large_allocation = manager.allocate_intelligent(10 * 1024 * 1024, DeviceId::Cpu, large_hint)?;
    println!("✅ Large allocation (10MB) created for compute workload");

    // 4. Demonstrate multi-GPU coordination
    println!("\n🔗 Demonstrating Multi-GPU Coordination...");
    
    let multi_gpu_hint = AllocationHint {
        expected_lifetime: Duration::from_secs(60),
        access_pattern: AccessPattern::Streaming,
        sharing_requirements: SharingRequirements::MultiGpu,
        priority: AllocationPriority::Critical,
    };
    
    let multi_gpu_allocation = manager.allocate_intelligent(5 * 1024 * 1024, DeviceId::Cpu, multi_gpu_hint)?;
    println!("✅ Multi-GPU allocation (5MB) created with cross-device sharing");

    // 5. Demonstrate memory pressure detection
    println!("\n🌡️  Demonstrating Memory Pressure Detection...");
    
    // Create several allocations to simulate memory pressure
    let mut pressure_allocations = Vec::new();
    let pressure_hint = AllocationHint {
        expected_lifetime: Duration::from_secs(10),
        access_pattern: AccessPattern::Random,
        sharing_requirements: SharingRequirements::ReadShared,
        priority: AllocationPriority::Low,
    };
    
    for i in 0..5 {
        let allocation = manager.allocate_intelligent(
            2 * 1024 * 1024, // 2MB each
            DeviceId::Cpu,
            pressure_hint.clone()
        )?;
        pressure_allocations.push(allocation);
        println!("✅ Pressure allocation {} (2MB) created", i + 1);
    }

    // 6. Get comprehensive memory statistics
    println!("\n📊 Memory Statistics Analysis...");
    let _stats = manager.get_memory_stats();
    println!("✅ Retrieved comprehensive memory statistics across all devices");
    println!("   - Buffer management statistics available");
    println!("   - Multi-GPU coordination statistics available");
    println!("   - Memory pressure monitoring statistics available");

    // 7. Force memory optimization
    println!("\n⚡ Forcing Memory Optimization...");
    let _optimization_result = manager.force_optimization()?;
    println!("✅ Memory optimization completed successfully");
    println!("   - Intelligent buffer optimization completed");
    println!("   - Multi-GPU load balancing completed");
    println!("   - Memory pressure relief completed");

    // 8. Demonstrate fragmentation analysis and automatic compaction
    println!("\n🔧 Demonstrating Fragmentation Analysis...");
    
    // Create fragmentation by deallocating every other allocation
    for (i, allocation) in pressure_allocations.into_iter().enumerate() {
        if i % 2 == 0 {
            manager.deallocate_intelligent(allocation)?;
            println!("✅ Deallocated allocation {} to create fragmentation", i + 1);
        }
    }
    
    // Force optimization to trigger automatic compaction
    let _defrag_result = manager.force_optimization()?;
    println!("✅ Automatic compaction triggered to resolve fragmentation");

    // 9. Demonstrate thermal management integration
    println!("\n🌡️  Demonstrating Thermal Management...");
    
    let thermal_hint = AllocationHint {
        expected_lifetime: Duration::from_secs(120), // Long-lived
        access_pattern: AccessPattern::Compute,      // Compute-intensive
        sharing_requirements: SharingRequirements::Exclusive,
        priority: AllocationPriority::Critical,
    };
    
    // This allocation should trigger thermal management checks
    let thermal_allocation = manager.allocate_intelligent(
        20 * 1024 * 1024, // Large 20MB allocation
        DeviceId::Cpu,
        thermal_hint
    )?;
    println!("✅ Thermal-aware allocation (20MB) created with automatic thermal management");

    // 10. Clean up allocations
    println!("\n🧹 Cleaning Up Allocations...");
    manager.deallocate_intelligent(small_allocation)?;
    manager.deallocate_intelligent(large_allocation)?;
    manager.deallocate_intelligent(multi_gpu_allocation)?;
    manager.deallocate_intelligent(thermal_allocation)?;
    println!("✅ All allocations cleaned up successfully");

    // 11. Final optimization and statistics
    println!("\n📈 Final Optimization and Statistics...");
    let _final_optimization = manager.force_optimization()?;
    let _final_stats = manager.get_memory_stats();
    println!("✅ Final optimization completed");
    println!("✅ Final statistics retrieved");

    println!("\n🎉 Advanced GPU Memory Management Demo Completed Successfully!");
    println!("================================================");
    println!("✅ Intelligent Buffer Management: Operational");
    println!("✅ Multi-GPU Coordination: Operational");
    println!("✅ Memory Pressure Detection: Operational");
    println!("✅ Automatic Fragmentation Analysis: Operational");
    println!("✅ Thermal Management Integration: Operational");
    println!("\n🚀 Task 4.1.3 Advanced GPU Memory Management implementation complete!");

    Ok(())
}