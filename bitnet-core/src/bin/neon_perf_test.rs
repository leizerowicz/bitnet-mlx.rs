use bitnet_core::cpu::{KernelSelector, detect_cpu_features, performance_validator::PerformanceValidator};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🎯 ARM64 NEON Optimization Performance Test");
    
    let cpu_features = detect_cpu_features();
    println!("Detected CPU: {:?}", cpu_features);
    
    let selector = KernelSelector::new();
    let kernel = selector.select_ternary_kernel();
    println!("Selected kernel: {}", kernel.name());
    
    // Test different sizes
    let sizes = [1024, 4096, 16384];
    
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
            kernel.compute(&weights, &inputs, &mut output)?;
        }
        
        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            kernel.compute(&weights, &inputs, &mut output)?;
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        let throughput = (size as f64) / (avg_time / 1_000_000_000.0); // elements per second
        
        println!("Average time: {:.2} ns", avg_time);
        println!("Throughput: {:.2} M elements/sec", throughput / 1_000_000.0);
    }
    
    // Run Microsoft parity validation
    println!("\n🎯 Running Microsoft parity validation...");
    let mut validator = PerformanceValidator::new();
    
    // Establish baselines
    if let Err(e) = validator.establish_baseline(&sizes) {
        eprintln!("Failed to establish baselines: {}", e);
    } else {
        // Run validation
        match validator.validate_performance(&sizes) {
            Ok(results) => {
                let report = validator.generate_report(&results);
                println!("\n{}", report);
                
                // Count successes
                let passed = results.iter().filter(|r| r.meets_target).count();
                let total = results.len();
                println!("🎯 Microsoft Parity Results: {}/{} targets achieved ({:.1}%)", 
                    passed, total, (passed as f64 / total as f64) * 100.0);
            },
            Err(e) => {
                eprintln!("Validation failed: {}", e);
            }
        }
    }
    
    println!("\n✅ Performance test completed");
    Ok(())
}