//! CPU SIMD Kernel Benchmarks
//!
//! Benchmarks to validate Microsoft parity targets (1.37x-6.17x CPU speedups)
//! for TL1, TL2, and I2_S kernel implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_core::cpu::{KernelSelector, CpuArch, detect_cpu_features};
use bitnet_core::cpu::performance_validator::{PerformanceValidator};

/// Benchmark configuration for different data sizes
const BENCHMARK_SIZES: &[usize] = &[1024, 4096, 16384, 65536];

/// Generate test data for ternary kernels
fn generate_ternary_data(size: usize) -> (Vec<i8>, Vec<f32>) {
    let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 0,
    }).collect();
    
    let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    
    (weights, inputs)
}

/// Microsoft parity validation benchmark
fn benchmark_microsoft_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("microsoft_parity_validation");
    
    // Create performance validator
    let mut validator = PerformanceValidator::new();
    println!("🎯 Running Microsoft parity validation benchmark...");
    
    // Establish baselines first
    if let Err(e) = validator.establish_baseline(BENCHMARK_SIZES) {
        eprintln!("Failed to establish baselines: {}", e);
        return;
    }
    
    // Run validation
    match validator.validate_performance(BENCHMARK_SIZES) {
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
    
    group.finish();
}

/// Benchmark ternary kernels across different architectures
fn benchmark_ternary_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_kernels");
    
    let current_arch = detect_cpu_features();
    println!("🔧 Detected CPU architecture: {:?}", current_arch);
    
    for &size in BENCHMARK_SIZES {
        let (weights, inputs) = generate_ternary_data(size);
        
        // Benchmark automatic selection
        let selector = KernelSelector::new();
        let kernel = selector.select_ternary_kernel();
        
        group.bench_with_input(
            BenchmarkId::new("Auto_Ternary", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0f32; size];
                    kernel.compute(
                        black_box(&weights),
                        black_box(&inputs),
                        black_box(&mut output),
                    ).unwrap();
                    black_box(output)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_microsoft_parity,
    benchmark_ternary_kernels
);
criterion_main!(benches);