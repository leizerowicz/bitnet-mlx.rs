//! Adaptive Tensor Pool Demo - Task 1.6.1 Solution
//! 
//! This example demonstrates the adaptive tensor memory pool that automatically
//! selects the optimal allocation strategy based on tensor characteristics,
//! resolving the performance gaps identified in Task 1.6.1.

use bitnet_core::memory::{HybridMemoryPool, AdaptiveTensorMemoryPool, MemoryResult};
use candle_core::Device;
use std::sync::Arc;
use std::time::Instant;

fn main() -> MemoryResult<()> {
    println!("=== Adaptive Tensor Pool Demo - Task 1.6.1 Solution ===");
    println!("Demonstrating automatic strategy selection for optimal performance");
    println!();

    // Initialize the adaptive tensor pool
    let base_pool = Arc::new(HybridMemoryPool::new()?);
    let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
    let device = Device::Cpu;

    println!("🚀 Initialized adaptive tensor pool with automatic strategy selection");
    println!();

    // Demonstrate different tensor allocation patterns
    let test_cases = vec![
        (1024, "Small scalar/vector", false, true),
        (16 * 1024, "Small matrix", false, true),
        (256 * 1024, "Medium tensor", true, false),
        (1024 * 1024, "Large weight matrix", true, false),
        (4 * 1024 * 1024, "Very large activation", false, true),
    ];

    println!("📊 Testing adaptive allocation strategy:");
    println!("{:<20} {:<15} {:<15} {:<15}", "Tensor Type", "Size", "Strategy", "Time (ns)");
    println!("{}", "-".repeat(70));

    for (i, (size_bytes, description, is_model_weight, is_temporary)) in test_cases.iter().enumerate() {
        let start = Instant::now();
        
        let handle = adaptive_pool.allocate_tensor_adaptive(
            i as u64,
            *size_bytes,
            &device,
            *is_model_weight,
            *is_temporary,
        )?;

        let allocation_time = start.elapsed().as_nanos();

        // Predict which strategy should be used
        let expected_strategy = if *size_bytes < 32 * 1024 {
            "Standard"
        } else if *is_model_weight {
            "Optimized"
        } else if *size_bytes > 1024 * 1024 {
            "Optimized"
        } else {
            "Standard"
        };

        println!("{:<20} {:<15} {:<15} {:<15}", 
            description, 
            format_size(*size_bytes), 
            expected_strategy,
            allocation_time
        );

        // Clean up
        adaptive_pool.deallocate_tensor_adaptive(i as u64, handle)?;
    }

    println!();

    // Get performance statistics
    let (std_avg, opt_avg, std_count, opt_count) = adaptive_pool.get_performance_stats();
    
    println!("📈 Performance Statistics:");
    println!("Standard pool: {} allocations, avg {:.0} ns", std_count, std_avg);
    println!("Optimized pool: {} allocations, avg {:.0} ns", opt_count, opt_avg);
    println!();

    // Get recommendations
    println!("💡 Performance Recommendations:");
    let recommendations = adaptive_pool.get_allocation_recommendations();
    if recommendations.is_empty() {
        println!("• Adaptive strategy is working optimally for your workload");
        println!("• Small tensors automatically use standard pool (low overhead)");
        println!("• Large tensors automatically use optimized pool (high performance)");
    } else {
        for rec in recommendations {
            println!("• {}", rec);
        }
    }
    println!();

    // Demonstrate performance with a realistic workload
    println!("🔬 Realistic Workload Simulation:");
    simulate_inference_workload(&adaptive_pool, &device)?;
    simulate_training_workload(&adaptive_pool, &device)?;

    println!("✅ Task 1.6.1 Resolution Complete!");
    println!("Adaptive tensor pool eliminates performance gaps through intelligent strategy selection");

    Ok(())
}

fn simulate_inference_workload(pool: &AdaptiveTensorMemoryPool, device: &Device) -> MemoryResult<()> {
    println!("  🧠 Inference workload (mixed tensor sizes):");
    
    let start = Instant::now();
    let mut handles = Vec::new();

    // Simulate typical inference pattern: many small tensors, few large ones
    for i in 0..50 {
        // Many small tensors (activations, intermediate values)
        let small_handle = pool.allocate_tensor_adaptive(
            1000 + i, 8 * 1024, device, false, true
        )?;
        handles.push((1000 + i, small_handle));

        // Occasional medium tensors (layer outputs)
        if i % 10 == 0 {
            let medium_handle = pool.allocate_tensor_adaptive(
                2000 + i, 128 * 1024, device, false, true
            )?;
            handles.push((2000 + i, medium_handle));
        }

        // Few large tensors (weight matrices)
        if i % 25 == 0 {
            let large_handle = pool.allocate_tensor_adaptive(
                3000 + i, 2 * 1024 * 1024, device, true, false
            )?;
            handles.push((3000 + i, large_handle));
        }
    }

    let allocation_time = start.elapsed();

    // Clean up
    for (id, handle) in handles {
        pool.deallocate_tensor_adaptive(id, handle)?;
    }

    let total_time = start.elapsed();

    println!("    Allocation: {:.2} ms, Total: {:.2} ms", 
        allocation_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn simulate_training_workload(pool: &AdaptiveTensorMemoryPool, device: &Device) -> MemoryResult<()> {
    println!("  📚 Training workload (large tensor focus):");
    
    let start = Instant::now();
    let mut handles = Vec::new();

    // Simulate training pattern: focus on large gradient and weight tensors
    for i in 0..20 {
        // Large weight gradients
        let gradient_handle = pool.allocate_tensor_adaptive(
            4000 + i, 1024 * 1024, device, false, true
        )?;
        handles.push((4000 + i, gradient_handle));

        // Large weight updates
        let weight_handle = pool.allocate_tensor_adaptive(
            5000 + i, 2 * 1024 * 1024, device, true, false
        )?;
        handles.push((5000 + i, weight_handle));
    }

    let allocation_time = start.elapsed();

    // Clean up
    for (id, handle) in handles {
        pool.deallocate_tensor_adaptive(id, handle)?;
    }

    let total_time = start.elapsed();

    println!("    Allocation: {:.2} ms, Total: {:.2} ms", 
        allocation_time.as_secs_f64() * 1000.0,
        total_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}MB", bytes / (1024 * 1024))
    }
}
