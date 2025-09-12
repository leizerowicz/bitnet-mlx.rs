#!/usr/bin/env cargo script
//! ```cargo
//! [dependencies]
//! bitnet-core = { path = "../bitnet-core" }
//! ```

use bitnet_core::cpu::{KernelSelector, detect_cpu_features};
use std::time::Instant;

fn main() {
    println!("🎯 ARM64 NEON Optimization Performance Test");
    
    let cpu_features = detect_cpu_features();
    println!("Detected CPU: {:?}", cpu_features);
    
    let selector = KernelSelector::new();
    let kernel = selector.select_ternary_kernel();
    println!("Selected kernel: {}", kernel.name());
    
    // Test different sizes
    let sizes = [1024, 4096, 16384, 65536];
    
    for size in sizes {
        println!("\n📊 Testing size: {} elements", size);
        
        // Generate test data
        let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        }).collect();
        
        let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0f32; size];
        
        // Warm up
        for _ in 0..10 {
            kernel.compute(&weights, &inputs, &mut output).unwrap();
        }
        
        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            kernel.compute(&weights, &inputs, &mut output).unwrap();
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        let throughput = (size as f64) / (avg_time / 1_000_000_000.0); // elements per second
        
        println!("Average time: {:.2} ns", avg_time);
        println!("Throughput: {:.2} M elements/sec", throughput / 1_000_000.0);
    }
    
    println!("\n✅ Performance test completed");
}