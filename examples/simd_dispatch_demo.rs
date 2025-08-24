//! SIMD and Dispatch System Demonstration
//!
//! This example demonstrates the cross-platform SIMD optimization
//! and automatic dispatch system for BitNet tensor operations.
//!
//! Features demonstrated:
//! - SIMD capability detection and optimization
//! - Automatic backend selection based on operation characteristics
//! - Performance comparison between backends
//! - Dispatch strategy customization
//! - Performance profiling and metrics

use bitnet_core::tensor::{
    core::BitNetTensor,
    dtype::BitNetDType,
    acceleration::{
        simd::{SimdAccelerator, SimdOptimization, create_simd_accelerator},
        dispatch::{
            OperationDispatcher, DispatchStrategy, OperationType, OperationContext,
            AccelerationBackend, PerformanceRequirements, create_operation_dispatcher
        },
        AccelerationBackendImpl,
    },
};
use std::time::Instant;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🚀 BitNet-Rust SIMD and Dispatch System Demo");
    println!("============================================\n");

    // 1. Demonstrate SIMD capability detection
    demonstrate_simd_detection()?;

    // 2. Show SIMD accelerator capabilities
    demonstrate_simd_accelerator()?;

    // 3. Demonstrate backend characteristics
    demonstrate_backend_characteristics();

    // 4. Show dispatch strategy comparison
    demonstrate_dispatch_strategies()?;

    // 5. Performance comparison demo
    demonstrate_performance_comparison()?;

    // 6. Show operation context usage
    demonstrate_operation_contexts()?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

fn demonstrate_simd_detection() -> Result<()> {
    println!("🔍 SIMD Capability Detection");
    println!("============================");

    let optimization = SimdOptimization::detect();

    println!("Detected SIMD optimization: {:?}", optimization);
    println!("Vector width (f32): {} elements", optimization.vector_width_f32());
    println!("Performance multiplier: {:.1}x", optimization.performance_multiplier());

    match optimization {
        SimdOptimization::None => {
            println!("❌ No SIMD support detected");
        },
        SimdOptimization::SSE2 => {
            println!("✅ SSE2 support (baseline x86_64)");
        },
        SimdOptimization::SSE41 => {
            println!("✅ SSE4.1 support");
        },
        SimdOptimization::AVX => {
            println!("✅ AVX support");
        },
        SimdOptimization::AVX2 => {
            println!("🚀 AVX2 support (high performance x86_64)");
        },
        SimdOptimization::AVX512 => {
            println!("🚀 AVX512 support (maximum performance x86_64)");
        },
        SimdOptimization::NEON => {
            println!("🚀 NEON support (ARM64/Apple Silicon)");
        },
    }

    println!();
    Ok(())
}

fn demonstrate_simd_accelerator() -> Result<()> {
    println!("⚡ SIMD Accelerator Capabilities");
    println!("===============================");

    match create_simd_accelerator() {
        Ok(Some(accelerator_box)) => {
            println!("✅ SIMD accelerator created successfully");

            let capabilities = accelerator_box.get_capabilities();
            println!("Backend: {:?}", capabilities.backend);
            println!("Max tensor size: {} elements", capabilities.max_tensor_size);
            println!("Supported data types: {:?}", capabilities.supported_dtypes);
            println!("Zero-copy support: {}", capabilities.zero_copy_support);
            println!("Parallel execution: {}", capabilities.parallel_execution);
            println!("Memory bandwidth: {:.1} GB/s", capabilities.memory_bandwidth_gbps);
            println!("Compute throughput: {:.1} GFLOPS", capabilities.compute_throughput_gflops);
        },
        Ok(None) => {
            println!("❌ SIMD accelerator not available on this platform");
        },
        Err(e) => {
            println!("❌ Error creating SIMD accelerator: {}", e);
        }
    }

    println!();
    Ok(())
}

fn demonstrate_backend_characteristics() {
    println!("🏗️  Backend Performance Characteristics");
    println!("======================================");

    let backends = [
        AccelerationBackend::MLX,
        AccelerationBackend::Metal,
        AccelerationBackend::SIMD,
        AccelerationBackend::CPU,
    ];

    println!("Backend    │ Priority │ Platform │ Throughput │ Latency  │ Bandwidth │ Efficiency");
    println!("───────────┼──────────┼──────────┼────────────┼──────────┼───────────┼───────────");

    for backend in &backends {
        let priority = backend.priority();
        let supported = backend.is_platform_supported();
        let perf = backend.performance_characteristics();

        println!(
            "{:<10} │ {:>8} │ {:>8} │ {:>7.0} GF │ {:>6.0} μs │ {:>6.0} GB/s │ {:>8.1}%",
            backend.to_string(),
            priority,
            if supported { "✅" } else { "❌" },
            perf.throughput_gflops,
            perf.latency_us,
            perf.memory_bandwidth_gbps,
            perf.power_efficiency * 100.0
        );
    }

    println!();
}

fn demonstrate_dispatch_strategies() -> Result<()> {
    println!("🎯 Dispatch Strategy Comparison");
    println!("==============================");

    let strategies = vec![
        DispatchStrategy::HighestPriority,
        DispatchStrategy::BestPerformance,
        DispatchStrategy::LowLatency,
        DispatchStrategy::HighThroughput,
        DispatchStrategy::LowMemory,
        DispatchStrategy::ForceBackend(AccelerationBackend::SIMD),
    ];

    let context = OperationContext::new(
        OperationType::MatMul,
        vec![vec![128, 256], vec![256, 512]],
        BitNetDType::F32
    );

    println!("Operation: Matrix multiplication (128×256) × (256×512)");
    println!("Complexity score: {:.1}", context.complexity_score());
    println!("Estimated memory: {:.1} MB", context.estimated_memory_bytes() as f64 / (1024.0 * 1024.0));
    println!();

    for strategy in strategies {
        println!("Strategy: {:?}", strategy);

        // Demonstrate strategy characteristics
        match strategy {
            DispatchStrategy::HighestPriority => {
                println!("  → Selects the backend with the highest priority");
                println!("  → Good for maximum performance without customization");
            },
            DispatchStrategy::BestPerformance => {
                println!("  → Uses historical performance data to select backend");
                println!("  → Learns from previous operations for optimization");
            },
            DispatchStrategy::LowLatency => {
                println!("  → Prioritizes backends with lowest latency");
                println!("  → Ideal for real-time applications");
            },
            DispatchStrategy::HighThroughput => {
                println!("  → Selects backends with highest throughput");
                println!("  → Best for batch processing workloads");
            },
            DispatchStrategy::LowMemory => {
                println!("  → Chooses backends that minimize memory usage");
                println!("  → Suitable for memory-constrained environments");
            },
            DispatchStrategy::ForceBackend(_) => {
                println!("  → Forces use of a specific backend");
                println!("  → Useful for testing or specific requirements");
            },
            _ => {}
        }
        println!();
    }

    Ok(())
}

fn demonstrate_performance_comparison() -> Result<()> {
    println!("📊 Performance Comparison Demo");
    println!("=============================");

    // Test different operation types and their characteristics
    let operations = vec![
        (OperationType::MatMul, "Matrix Multiplication", vec![vec![64, 128], vec![128, 256]]),
        (OperationType::Add, "Element-wise Addition", vec![vec![1000, 1000]]),
        (OperationType::Convolution, "Convolution", vec![vec![32, 3, 224, 224], vec![64, 3, 5, 5]]),
        (OperationType::Reduction, "Reduction (Sum)", vec![vec![1000, 1000]]),
        (OperationType::Transpose, "Transpose", vec![vec![1000, 2000]]),
    ];

    println!("Operation              │ Intensity │ Preferred │ Complexity │ Memory");
    println!("───────────────────────┼───────────┼───────────┼────────────┼────────");

    for (op_type, name, shapes) in operations {
        let context = OperationContext::new(op_type, shapes, BitNetDType::F32);
        let intensity = op_type.computational_intensity();
        let preferred = op_type.preferred_backend();
        let complexity = context.complexity_score();
        let memory_mb = context.estimated_memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:<22} │ {:>9.2} │ {:>9} │ {:>10.1} │ {:>5.1} MB",
            name, intensity, preferred, complexity, memory_mb
        );
    }

    println!();
    Ok(())
}

fn demonstrate_operation_contexts() -> Result<()> {
    println!("🎛️  Operation Context Usage");
    println!("==========================");

    // Create different operation contexts
    let contexts = vec![
        (
            "Small Matrix Multiply",
            OperationContext::new(
                OperationType::MatMul,
                vec![vec![32, 64], vec![64, 128]],
                BitNetDType::F32
            )
        ),
        (
            "Large Element-wise Add",
            OperationContext::new(
                OperationType::Add,
                vec![vec![4096, 4096]],
                BitNetDType::F32
            )
        ),
        (
            "Memory-bound Transpose",
            OperationContext::new(
                OperationType::Transpose,
                vec![vec![10000, 1000]],
                BitNetDType::F32
            )
        ),
    ];

    for (name, context) in contexts {
        println!("Context: {}", name);
        println!("  Operation: {:?}", context.operation_type);
        println!("  Input shapes: {:?}", context.input_shapes);
        println!("  Data type: {:?}", context.dtype);
        println!("  Complexity: {:.1}", context.complexity_score());
        println!("  Memory usage: {:.1} MB", context.estimated_memory_bytes() as f64 / (1024.0 * 1024.0));
        println!("  Preferred backend: {}", context.operation_type.preferred_backend());
        println!();
    }

    Ok(())
}

#[allow(dead_code)]
fn benchmark_tensor_operations() -> Result<()> {
    println!("🏁 Tensor Operation Benchmarks");
    println!("=============================");

    // Create test tensors
    let sizes = [64, 128, 256, 512];

    println!("Matrix size │ Creation time │ Memory usage");
    println!("────────────┼───────────────┼──────────────");

    for &size in &sizes {
        let start = Instant::now();
        let _tensor_a = BitNetTensor::ones(&[size, size], BitNetDType::F32)?;
        let _tensor_b = BitNetTensor::ones(&[size, size], BitNetDType::F32)?;
        let creation_time = start.elapsed();

        let memory_usage = size * size * 4 * 2; // 2 tensors, f32 = 4 bytes

        println!(
            "{:>10}  │ {:>10.2} ms │ {:>9.1} MB",
            format!("{}×{}", size, size),
            creation_time.as_secs_f64() * 1000.0,
            memory_usage as f64 / (1024.0 * 1024.0)
        );
    }

    println!();
    Ok(())
}

/// Demonstrate real-world usage patterns
#[allow(dead_code)]
fn demonstrate_real_world_usage() -> Result<()> {
    println!("🌍 Real-World Usage Patterns");
    println!("============================");

    // Pattern 1: High-performance computing
    println!("Pattern 1: High-Performance Computing");
    let hpc_context = OperationContext::new(
        OperationType::MatMul,
        vec![vec![2048, 2048], vec![2048, 2048]],
        BitNetDType::F32
    );
    println!("  → Large matrix operations requiring maximum throughput");
    println!("  → Complexity: {:.1}", hpc_context.complexity_score());
    println!("  → Preferred: {}", hpc_context.operation_type.preferred_backend());
    println!();

    // Pattern 2: Real-time inference
    println!("Pattern 2: Real-Time Inference");
    let rt_context = OperationContext::new(
        OperationType::Add,
        vec![vec![1, 1000]],
        BitNetDType::F16
    );
    println!("  → Small operations requiring minimal latency");
    println!("  → Complexity: {:.1}", rt_context.complexity_score());
    println!("  → Preferred: {}", rt_context.operation_type.preferred_backend());
    println!();

    // Pattern 3: Batch processing
    println!("Pattern 3: Batch Processing");
    let batch_context = OperationContext::new(
        OperationType::Convolution,
        vec![vec![128, 3, 224, 224], vec![256, 3, 7, 7]],
        BitNetDType::F32
    );
    println!("  → Large batch operations optimizing for throughput");
    println!("  → Complexity: {:.1}", batch_context.complexity_score());
    println!("  → Preferred: {}", batch_context.operation_type.preferred_backend());
    println!();

    Ok(())
}
