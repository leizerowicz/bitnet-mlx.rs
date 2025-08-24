//! Linear Algebra Operations Demo
//!
//! This example demonstrates the comprehensive linear algebra operations
//! available in BitNet-Rust, showcasing matrix multiplication, decompositions,
//! and optimization strategies.

use bitnet_core::{
    device::get_cpu_device,
    memory::HybridMemoryPool,
    tensor::ops::linear_algebra::*,
    tensor::{set_global_memory_pool, BitNetDType, BitNetTensor},
};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet Linear Algebra Operations Demo");
    println!("========================================");

    // Initialize memory pool and device
    let device = get_cpu_device();
    let memory_pool = Arc::new(HybridMemoryPool::new()?);

    // Initialize global memory pool
    set_global_memory_pool(Arc::downgrade(&memory_pool));

    println!("Using device: {device:?}");

    // Run all demonstrations
    demo_matrix_multiplication()?;
    demo_dot_products()?;
    demo_matrix_transformations()?;
    demo_advanced_decompositions()?;
    demo_performance_benchmarks()?;
    demo_optimization_strategies()?;

    println!("\n✅ All linear algebra operations completed successfully!");
    Ok(())
}

fn demo_matrix_multiplication() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Matrix Multiplication Operations");
    println!("{}", "-".repeat(40));

    // Basic matrix multiplication
    println!("1. Basic Matrix Multiplication:");
    let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None)?;
    let b = BitNetTensor::ones(&[4, 2], BitNetDType::F32, None)?;

    println!(
        "   Matrix A: {:?} × Matrix B: {:?}",
        a.shape().dims(),
        b.shape().dims()
    );

    let start = Instant::now();
    let result = matmul(&a, &b)?;
    let duration = start.elapsed();

    println!("   Result shape: {:?}", result.shape().dims());
    println!("   Computation time: {duration:?}");

    // Large matrix multiplication with automatic optimization
    println!("\n2. Large Matrix Multiplication (Auto-optimization):");
    let a_large = BitNetTensor::ones(&[512, 256], BitNetDType::F32, None)?;
    let b_large = BitNetTensor::ones(&[256, 128], BitNetDType::F32, None)?;

    println!(
        "   Matrix A: {:?} × Matrix B: {:?}",
        a_large.shape().dims(),
        b_large.shape().dims()
    );

    let start = Instant::now();
    let result_large = matmul(&a_large, &b_large)?;
    let duration = start.elapsed();

    println!("   Result shape: {:?}", result_large.shape().dims());
    println!("   Computation time: {duration:?}");

    // Batched matrix multiplication
    println!("\n3. Batched Matrix Multiplication:");
    let a_batch = BitNetTensor::ones(&[4, 64, 32], BitNetDType::F32, None)?;
    let b_batch = BitNetTensor::ones(&[4, 32, 16], BitNetDType::F32, None)?;

    println!(
        "   Batch A: {:?} × Batch B: {:?}",
        a_batch.shape().dims(),
        b_batch.shape().dims()
    );

    let start = Instant::now();
    let result_batch = batched_matmul(&a_batch, &b_batch)?;
    let duration = start.elapsed();

    println!("   Result shape: {:?}", result_batch.shape().dims());
    println!("   Computation time: {duration:?}");

    Ok(())
}

fn demo_dot_products() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Dot Product Operations");
    println!("{}", "-".repeat(40));

    // Vector dot product
    println!("1. Vector Dot Product:");
    let x = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;
    let y = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;

    println!(
        "   Vector X: {:?}, Vector Y: {:?}",
        x.shape().dims(),
        y.shape().dims()
    );

    let start = Instant::now();
    let dot_result = dot(&x, &y)?;
    let duration = start.elapsed();

    println!(
        "   Dot product result shape: {:?}",
        dot_result.shape().dims()
    );
    println!("   Computation time: {duration:?}");

    // Outer product
    println!("\n2. Outer Product:");
    let u = BitNetTensor::ones(&[5], BitNetDType::F32, None)?;
    let v = BitNetTensor::ones(&[3], BitNetDType::F32, None)?;

    println!(
        "   Vector U: {:?}, Vector V: {:?}",
        u.shape().dims(),
        v.shape().dims()
    );

    let start = Instant::now();
    let outer_result = outer(&u, &v)?;
    let duration = start.elapsed();

    println!(
        "   Outer product result shape: {:?}",
        outer_result.shape().dims()
    );
    println!("   Computation time: {duration:?}");

    // High-dimensional dot product
    println!("\n3. Multi-dimensional Dot Product:");
    let a = BitNetTensor::ones(&[2, 3, 100], BitNetDType::F32, None)?;
    let b = BitNetTensor::ones(&[2, 3, 100], BitNetDType::F32, None)?;

    println!(
        "   Tensor A: {:?}, Tensor B: {:?}",
        a.shape().dims(),
        b.shape().dims()
    );

    let start = Instant::now();
    let dot_nd = dot(&a, &b)?;
    let duration = start.elapsed();

    println!("   Dot product result shape: {:?}", dot_nd.shape().dims());
    println!("   Computation time: {duration:?}");

    Ok(())
}

fn demo_matrix_transformations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Matrix Transformation Operations");
    println!("{}", "-".repeat(40));

    // Matrix transpose
    println!("1. Matrix Transpose:");
    let matrix = BitNetTensor::ones(&[128, 64], BitNetDType::F32, None)?;
    println!("   Original shape: {:?}", matrix.shape().dims());

    let start = Instant::now();
    let transposed = transpose(&matrix)?;
    let duration = start.elapsed();

    println!("   Transposed shape: {:?}", transposed.shape().dims());
    println!("   Computation time: {duration:?}");

    // Tensor permutation
    println!("\n2. Tensor Permutation:");
    let tensor_3d = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None)?;
    println!("   Original shape: {:?}", tensor_3d.shape().dims());

    let start = Instant::now();
    let permuted = permute(&tensor_3d, &[2, 0, 1])?;
    let duration = start.elapsed();

    println!("   Permuted shape: {:?}", permuted.shape().dims());
    println!("   Permutation [2, 0, 1] applied");
    println!("   Computation time: {duration:?}");

    // Identity matrix creation
    println!("\n3. Identity Matrix Creation:");
    let sizes = vec![5, 16, 64, 128];

    for size in sizes {
        let start = Instant::now();
        let identity = eye(size, BitNetDType::F32, None)?;
        let duration = start.elapsed();

        println!(
            "   {}×{} identity matrix created in {:?} with shape {:?}",
            size,
            size,
            duration,
            identity.shape().dims()
        );
    }

    Ok(())
}

fn demo_advanced_decompositions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Advanced Linear Algebra Decompositions");
    println!("{}", "-".repeat(40));

    // SVD Decomposition
    println!("1. Singular Value Decomposition (SVD):");
    let matrix = BitNetTensor::ones(&[100, 50], BitNetDType::F32, None)?;
    println!("   Input matrix: {:?}", matrix.shape().dims());

    let start = Instant::now();
    let (u, s, vt) = svd(&matrix)?;
    let duration = start.elapsed();

    println!("   U matrix: {:?}", u.shape().dims());
    println!("   Singular values: {:?}", s.shape().dims());
    println!("   V^T matrix: {:?}", vt.shape().dims());
    println!("   SVD computation time: {duration:?}");
    println!("   Note: Current implementation is placeholder");

    // QR Decomposition
    println!("\n2. QR Decomposition:");
    let matrix = BitNetTensor::ones(&[80, 60], BitNetDType::F32, None)?;
    println!("   Input matrix: {:?}", matrix.shape().dims());

    let start = Instant::now();
    let (q, r) = qr(&matrix)?;
    let duration = start.elapsed();

    println!("   Q matrix: {:?}", q.shape().dims());
    println!("   R matrix: {:?}", r.shape().dims());
    println!("   QR computation time: {duration:?}");
    println!("   Note: Current implementation is placeholder");

    // Cholesky Decomposition
    println!("\n3. Cholesky Decomposition:");
    let square_matrix = BitNetTensor::ones(&[50, 50], BitNetDType::F32, None)?;
    println!("   Input matrix: {:?}", square_matrix.shape().dims());

    let start = Instant::now();
    let chol = cholesky(&square_matrix)?;
    let duration = start.elapsed();

    println!("   Cholesky factor: {:?}", chol.shape().dims());
    println!("   Cholesky computation time: {duration:?}");
    println!("   Note: Current implementation is placeholder");

    // Eigenvalue Decomposition
    println!("\n4. Eigenvalue Decomposition:");
    let square_matrix = BitNetTensor::ones(&[32, 32], BitNetDType::F32, None)?;
    println!("   Input matrix: {:?}", square_matrix.shape().dims());

    let start = Instant::now();
    let (eigenvals, eigenvecs) = eig(&square_matrix)?;
    let duration = start.elapsed();

    println!("   Eigenvalues: {:?}", eigenvals.shape().dims());
    println!("   Eigenvectors: {:?}", eigenvecs.shape().dims());
    println!("   Eigendecomposition time: {duration:?}");
    println!("   Note: Current implementation is placeholder");

    Ok(())
}

fn demo_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Performance Benchmarks");
    println!("{}", "-".repeat(40));

    println!("1. Matrix Multiplication Performance Scaling:");
    let sizes = vec![32, 64, 128, 256, 512];

    for size in sizes {
        let a = BitNetTensor::ones(&[size, size], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[size, size], BitNetDType::F32, None)?;

        let start = Instant::now();
        let _result = matmul(&a, &b)?;
        let duration = start.elapsed();

        let ops = (size as f64).powi(3) * 2.0; // Approximate FLOPS
        let gflops = ops / duration.as_secs_f64() / 1e9;

        println!("   {size}×{size}: {duration:?} ({gflops:.2} GFLOPS)");
    }

    println!("\n2. Dot Product Performance Scaling:");
    let sizes = vec![1000, 10000, 100000, 1000000];

    for size in sizes {
        let x = BitNetTensor::ones(&[size], BitNetDType::F32, None)?;
        let y = BitNetTensor::ones(&[size], BitNetDType::F32, None)?;

        let start = Instant::now();
        let _result = dot(&x, &y)?;
        let duration = start.elapsed();

        let ops = size as f64 * 2.0; // Multiply and add
        let gflops = ops / duration.as_secs_f64() / 1e9;

        println!("   {size} elements: {duration:?} ({gflops:.2} GFLOPS)");
    }

    Ok(())
}

fn demo_optimization_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Matrix Multiplication Optimization Strategies");
    println!("{}", "-".repeat(40));

    let matrix_sizes = vec![
        (64, 64, "Small matrices"),
        (256, 256, "Medium matrices"),
        (1024, 1024, "Large matrices"),
    ];

    for (size, _, description) in matrix_sizes {
        println!(
            "\n{}. {} ({}×{}):",
            if size == 64 {
                1
            } else if size == 256 {
                2
            } else {
                3
            },
            description,
            size,
            size
        );

        let a = BitNetTensor::ones(&[size, size], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[size, size], BitNetDType::F32, None)?;

        // Test different strategies
        let strategies = vec![
            (MatMulStrategy::Standard, "Standard"),
            (MatMulStrategy::Blocked, "Blocked"),
            (MatMulStrategy::SimdAccelerated, "SIMD Accelerated"),
            (MatMulStrategy::DeviceOptimized, "Device Optimized"),
        ];

        for (strategy, name) in strategies {
            let config = MatMulConfig {
                strategy,
                block_size: if size < 128 { 32 } else { 128 },
                use_simd: true,
                use_device_optimization: strategy == MatMulStrategy::DeviceOptimized,
                prefer_row_major: true,
            };

            let start = Instant::now();
            let _result = matmul_with_config(&a, &b, &config)?;
            let duration = start.elapsed();

            println!("   {name}: {duration:?}");
        }

        // Show automatic strategy selection
        println!("   Automatic selection: Uses heuristics based on matrix size and device");
    }

    Ok(())
}

// Helper function for pretty-printing performance results
#[allow(dead_code)]
fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}μs", duration.as_micros())
    }
}

#[cfg(test)]
mod demo_tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        bitnet_core::memory::initialize_global_memory().unwrap();

        // Test each demo function
        demo_matrix_multiplication().unwrap();
        demo_dot_products().unwrap();
        demo_matrix_transformations().unwrap();
        demo_advanced_decompositions().unwrap();
        demo_performance_benchmarks().unwrap();
        demo_optimization_strategies().unwrap();
    }
}
