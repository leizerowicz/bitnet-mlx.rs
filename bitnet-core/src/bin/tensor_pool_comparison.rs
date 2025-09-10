//! Performance comparison tool for Task 1.6.1
//! Compares standard tensor pool vs optimized tensor pool performance

use bitnet_core::memory::{
    HybridMemoryPool, TensorMemoryPool, OptimizedTensorMemoryPool, OptimizedTensorPoolConfig,
    MemoryResult
};
use candle_core::Device;
use std::sync::Arc;
use std::time::Instant;

fn main() -> MemoryResult<()> {
    println!("=== BitNet-Rust Tensor Pool Performance Comparison ===");
    println!("Task 1.6.1: Analyzing performance gap between standard and optimized pools");
    println!();

    // Initialize pools
    let device = Device::Cpu;
    let base_pool = Arc::new(HybridMemoryPool::new()?);
    
    // Standard tensor pool
    let standard_pool = TensorMemoryPool::new(base_pool.clone())?;
    
    // Optimized tensor pool
    let mut config = OptimizedTensorPoolConfig::default();
    config.enable_prewarming = true;
    let optimized_pool = OptimizedTensorMemoryPool::with_config(base_pool, config)?;

    // Test scenarios
    let test_sizes = vec![
        (1024, "VerySmall (1KB)"),
        (16384, "Small (16KB)"),
        (262144, "Medium (256KB)"),
        (4194304, "Large (4MB)"),
    ];

    let iterations = 100;

    println!("Performance comparison with {} iterations per test:", iterations);
    println!("{:<20} {:<15} {:<15} {:<10}", "Size Category", "Standard (ns)", "Optimized (ns)", "Gap (%)");
    println!("{}", "-".repeat(70));

    for (size_bytes, category_name) in test_sizes {
        // Benchmark standard pool
        let start = Instant::now();
        for i in 0..iterations {
            let handle = standard_pool.allocate_tensor(i as u64, size_bytes, &device, false, false)?;
            standard_pool.deallocate_tensor(i as u64, handle)?;
        }
        let standard_duration = start.elapsed();
        let standard_avg_ns = standard_duration.as_nanos() / iterations as u128;

        // Benchmark optimized pool  
        let start = Instant::now();
        for i in 0..iterations {
            let handle = optimized_pool.allocate_tensor_optimized(i as u64, size_bytes, &device, false, false)?;
            optimized_pool.deallocate_tensor_optimized(i as u64, handle)?;
        }
        let optimized_duration = start.elapsed();
        let optimized_avg_ns = optimized_duration.as_nanos() / iterations as u128;

        // Calculate performance gap
        let gap_percent = if optimized_avg_ns > 0 {
            ((standard_avg_ns as f64 - optimized_avg_ns as f64) / optimized_avg_ns as f64) * 100.0
        } else {
            0.0
        };

        println!("{:<20} {:<15} {:<15} {:<10.1}%", 
            category_name, 
            standard_avg_ns, 
            optimized_avg_ns, 
            gap_percent
        );
    }

    println!();
    
    // Get detailed metrics from optimized pool
    let (avg_alloc_time, avg_dealloc_time, alloc_count, dealloc_count) = 
        optimized_pool.get_performance_stats();
    let cache_hit_rate = optimized_pool.get_cache_hit_rate();
    
    println!("=== Optimized Pool Detailed Metrics ===");
    println!("Average allocation time: {:.2} ns", avg_alloc_time);
    println!("Average deallocation time: {:.2} ns", avg_dealloc_time);
    println!("Total allocations: {}", alloc_count);
    println!("Total deallocations: {}", dealloc_count);
    println!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0);
    
    println!();
    println!("=== Task 1.6.1 Analysis Complete ===");
    println!("Recommended actions:");
    println!("1. If gap > 20%: Backport key optimizations to standard pool");
    println!("2. Update examples to use optimized pool by default");
    println!("3. Create migration guide for users");
    
    Ok(())
}
