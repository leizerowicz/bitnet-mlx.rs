//! Performance Comparison Demo
//!
//! This example provides comprehensive performance benchmarking of BitNet tensor operations:
//! - Memory allocation and deallocation performance
//! - Mathematical operations benchmarking
//! - Device migration performance analysis
//! - Memory efficiency comparisons
//! - Scaling analysis across different tensor sizes
//! - Acceleration backend performance validation

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::collections::HashMap;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel
};
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::device::{
    get_cpu_device, auto_select_device, is_metal_available, get_metal_device
};

// =============================================================================
// Performance Test Configuration
// =============================================================================

#[derive(Clone)]
struct PerformanceConfig {
    warmup_iterations: usize,
    benchmark_iterations: usize,
    memory_test_sizes: Vec<usize>,
    operation_test_shapes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    enable_memory_tracking: bool,
    enable_device_migration_tests: bool,
    enable_acceleration_tests: bool,
    max_test_duration: Duration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            memory_test_sizes: vec![1024, 4096, 16384, 65536, 262144], // 1KB to 256KB
            operation_test_shapes: vec![
                vec![64, 64],
                vec![128, 128],
                vec![256, 256],
                vec![512, 512],
                vec![1024, 1024],
                vec![2048, 2048],
            ],
            batch_sizes: vec![1, 4, 8, 16, 32, 64, 128],
            enable_memory_tracking: true,
            enable_device_migration_tests: true,
            enable_acceleration_tests: true,
            max_test_duration: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    test_name: String,
    tensor_size: Option<Vec<usize>>,
    iterations: usize,
    total_time: Duration,
    average_time: Duration,
    min_time: Duration,
    max_time: Duration,
    throughput: Option<f64>, // operations per second
    memory_used: Option<usize>,
    memory_efficiency: Option<f64>, // percentage
}

impl BenchmarkResult {
    fn new(test_name: String) -> Self {
        Self {
            test_name,
            tensor_size: None,
            iterations: 0,
            total_time: Duration::ZERO,
            average_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            throughput: None,
            memory_used: None,
            memory_efficiency: None,
        }
    }
}

// =============================================================================
// Main Performance Demo
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 BitNet Tensor Performance Comparison Demo");
    println!("============================================");

    let config = PerformanceConfig::default();
    let pool = create_performance_memory_pool()?;

    print_system_info()?;
    print_test_configuration(&config);

    let mut all_results = Vec::new();

    // Core performance benchmarks
    println!("\n🧪 Running core performance benchmarks...");
    let mut memory_results = benchmark_memory_operations(&pool, &config)?;
    let mut arithmetic_results = benchmark_arithmetic_operations(&pool, &config)?;
    let mut linear_algebra_results = benchmark_linear_algebra_operations(&pool, &config)?;

    all_results.append(&mut memory_results);
    all_results.append(&mut arithmetic_results);
    all_results.append(&mut linear_algebra_results);

    // Device-specific benchmarks
    if config.enable_device_migration_tests {
        println!("\n💻 Running device migration benchmarks...");
        let mut migration_results = benchmark_device_migration(&pool, &config)?;
        all_results.append(&mut migration_results);
    }

    // Acceleration benchmarks
    if config.enable_acceleration_tests {
        println!("\n⚡ Running acceleration benchmarks...");
        let mut acceleration_results = benchmark_acceleration(&pool, &config)?;
        all_results.append(&mut acceleration_results);
    }

    // Scaling analysis
    println!("\n📈 Running scaling analysis...");
    let mut scaling_results = benchmark_scaling_analysis(&pool, &config)?;
    all_results.append(&mut scaling_results);

    // Memory efficiency analysis
    if config.enable_memory_tracking {
        println!("\n💾 Running memory efficiency analysis...");
        let mut memory_efficiency_results = benchmark_memory_efficiency(&pool, &config)?;
        all_results.append(&mut memory_efficiency_results);
    }

    // Generate comprehensive report
    println!("\n📊 Generating performance report...");
    generate_performance_report(&all_results, &pool)?;

    println!("\n🎉 Performance comparison demo completed!");

    Ok(())
}

// =============================================================================
// Memory Operations Benchmarks
// =============================================================================

fn benchmark_memory_operations(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("💾 Benchmarking memory operations...");

    let device = auto_select_device();
    let mut results = Vec::new();

    // Tensor creation benchmark
    for &size in &config.memory_test_sizes {
        let _shape = vec![size];
        let mut result = BenchmarkResult::new(format!("tensor_creation_{}", size));
        result.tensor_size = Some(shape.clone());

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = BitNetTensor::zeros_with_pool(
                shape.clone(),
                BitNetDType::Float32,
                device.clone(),
                pool.clone()
            )?;
        }

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..config.benchmark_iterations {
            let start = Instant::now();
            let _tensor = BitNetTensor::zeros_with_pool(
                shape.clone(),
                BitNetDType::Float32,
                device.clone(),
                pool.clone()
            )?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        result.iterations = times.len();
        result.total_time = times.iter().sum();
        result.average_time = result.total_time / times.len() as u32;
        result.min_time = *times.iter().min().unwrap();
        result.max_time = *times.iter().max().unwrap();
        result.throughput = Some(1e9 / result.average_time.as_nanos() as f64);
        result.memory_used = Some(size * 4); // f32 = 4 bytes

        println!("  Tensor creation {}KB: {:.2}μs avg",
                 size * 4 / 1024, result.average_time.as_micros());

        results.push(result);
    }

    // Tensor cloning benchmark
    println!("🔄 Benchmarking tensor cloning...");

    let test_tensor = BitNetTensor::random_normal_with_pool(
        vec![1024, 1024],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let mut clone_result = BenchmarkResult::new("tensor_cloning_1024x1024".to_string());
    clone_result.tensor_size = Some(vec![1024, 1024]);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = test_tensor.clone();
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..config.benchmark_iterations {
        let start = Instant::now();
        let _cloned = test_tensor.clone();
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    clone_result.iterations = times.len();
    clone_result.total_time = times.iter().sum();
    clone_result.average_time = clone_result.total_time / times.len() as u32;
    clone_result.min_time = *times.iter().min().unwrap();
    clone_result.max_time = *times.iter().max().unwrap();
    clone_result.throughput = Some(1e9 / clone_result.average_time.as_nanos() as f64);

    println!("  Tensor cloning 1024x1024: {:.2}μs avg", clone_result.average_time.as_micros());

    results.push(clone_result);

    // Memory deallocation benchmark
    println!("🗑️  Benchmarking memory deallocation...");

    let mut dealloc_result = BenchmarkResult::new("memory_deallocation".to_string());

    let mut times = Vec::new();
    for _ in 0..config.benchmark_iterations {
        // Create tensor
        let tensor = BitNetTensor::zeros_with_pool(
            vec![512, 512],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        // Measure deallocation time
        let start = Instant::now();
        drop(tensor); // Explicit deallocation
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    dealloc_result.iterations = times.len();
    dealloc_result.total_time = times.iter().sum();
    dealloc_result.average_time = dealloc_result.total_time / times.len() as u32;
    dealloc_result.min_time = *times.iter().min().unwrap();
    dealloc_result.max_time = *times.iter().max().unwrap();

    println!("  Memory deallocation: {:.2}μs avg", dealloc_result.average_time.as_micros());

    results.push(dealloc_result);

    Ok(results)
}

// =============================================================================
// Arithmetic Operations Benchmarks
// =============================================================================

fn benchmark_arithmetic_operations(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("🧮 Benchmarking arithmetic operations...");

    let device = auto_select_device();
    let mut results = Vec::new();

    let operations = vec![
        ("addition", "add"),
        ("subtraction", "sub"),
        ("multiplication", "mul"),
        ("division", "div"),
    ];

    for shape in &config.operation_test_shapes {
        let tensor_a = BitNetTensor::ones_with_pool(
            shape.clone(),
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let tensor_b = BitNetTensor::from_scalar_with_pool(
            2.0_f32,
            shape.clone(),
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        for (op_name, _op_code) in &operations {
            let mut result = BenchmarkResult::new(
                format!("{}_{}", op_name, shape_to_string(shape))
            );
            result.tensor_size = Some(shape.clone());

            // Warmup
            for _ in 0..config.warmup_iterations {
                match *op_name {
                    "addition" => { let _ = tensor_a.add(&tensor_b)?; },
                    "subtraction" => { let _ = tensor_a.sub(&tensor_b)?; },
                    "multiplication" => { let _ = tensor_a.mul(&tensor_b)?; },
                    "division" => { let _ = tensor_a.div(&tensor_b)?; },
                    _ => {}
                }
            }

            // Benchmark
            let mut times = Vec::new();
            for _ in 0..config.benchmark_iterations {
                let start = Instant::now();
                match *op_name {
                    "addition" => { let _ = tensor_a.add(&tensor_b)?; },
                    "subtraction" => { let _ = tensor_a.sub(&tensor_b)?; },
                    "multiplication" => { let _ = tensor_a.mul(&tensor_b)?; },
                    "division" => { let _ = tensor_a.div(&tensor_b)?; },
                    _ => {}
                }
                let elapsed = start.elapsed();
                times.push(elapsed);
            }

            result.iterations = times.len();
            result.total_time = times.iter().sum();
            result.average_time = result.total_time / times.len() as u32;
            result.min_time = *times.iter().min().unwrap();
            result.max_time = *times.iter().max().unwrap();

            let elements = shape.iter().product::<usize>();
            result.throughput = Some(elements as f64 / result.average_time.as_secs_f64());

            println!("  {} {}: {:.2}μs ({:.1}M ops/s)",
                     op_name, shape_to_string(shape),
                     result.average_time.as_micros(),
                     result.throughput.unwrap() / 1e6);

            results.push(result);
        }
    }

    // Element-wise operations with broadcasting
    println!("📡 Benchmarking broadcasting operations...");

    let tensor_large = BitNetTensor::ones_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor_scalar = BitNetTensor::scalar_with_pool(
        5.0_f32,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let mut broadcast_result = BenchmarkResult::new("broadcasting_scalar_add".to_string());
    broadcast_result.tensor_size = Some(vec![256, 256]);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = tensor_large.add(&tensor_scalar)?;
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..config.benchmark_iterations {
        let start = Instant::now();
        let _ = tensor_large.add(&tensor_scalar)?;
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    broadcast_result.iterations = times.len();
    broadcast_result.total_time = times.iter().sum();
    broadcast_result.average_time = broadcast_result.total_time / times.len() as u32;
    broadcast_result.min_time = *times.iter().min().unwrap();
    broadcast_result.max_time = *times.iter().max().unwrap();

    println!("  Broadcasting scalar add 256x256: {:.2}μs",
             broadcast_result.average_time.as_micros());

    results.push(broadcast_result);

    Ok(results)
}

// =============================================================================
// Linear Algebra Operations Benchmarks
// =============================================================================

fn benchmark_linear_algebra_operations(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("📐 Benchmarking linear algebra operations...");

    let device = auto_select_device();
    let mut results = Vec::new();

    // Matrix multiplication benchmarks
    let matmul_sizes = vec![64, 128, 256, 512, 1024];

    for size in matmul_sizes {
        let tensor_a = BitNetTensor::random_normal_with_pool(
            vec![size, size],
            0.0, 0.01,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let tensor_b = BitNetTensor::random_normal_with_pool(
            vec![size, size],
            0.0, 0.01,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let mut result = BenchmarkResult::new(format!("matmul_{}x{}", size, size));
        result.tensor_size = Some(vec![size, size]);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = tensor_a.matmul(&tensor_b)?;
        }

        // Benchmark
        let mut times = Vec::new();
        let iterations = std::cmp::min(config.benchmark_iterations, 50); // Fewer iterations for large matrices

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = tensor_a.matmul(&tensor_b)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        result.iterations = times.len();
        result.total_time = times.iter().sum();
        result.average_time = result.total_time / times.len() as u32;
        result.min_time = *times.iter().min().unwrap();
        result.max_time = *times.iter().max().unwrap();

        // GFLOPS calculation: 2 * n^3 operations for n x n matrix multiplication
        let flops = 2.0 * (size as f64).powi(3);
        let gflops = flops / result.average_time.as_secs_f64() / 1e9;
        result.throughput = Some(gflops);

        println!("  Matrix multiplication {}x{}: {:.2}ms ({:.1} GFLOPS)",
                 size, size,
                 result.average_time.as_millis(),
                 gflops);

        results.push(result);
    }

    // Vector operations
    println!("📊 Benchmarking vector operations...");

    let vector_size = 1024 * 1024; // 1M elements
    let vec_a = BitNetTensor::random_normal_with_pool(
        vec![vector_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let vec_b = BitNetTensor::random_normal_with_pool(
        vec![vector_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    // Dot product
    let mut dot_result = BenchmarkResult::new("dot_product_1M".to_string());
    dot_result.tensor_size = Some(vec![vector_size]);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = vec_a.dot(&vec_b)?;
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..config.benchmark_iterations {
        let start = Instant::now();
        let _ = vec_a.dot(&vec_b)?;
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    dot_result.iterations = times.len();
    dot_result.total_time = times.iter().sum();
    dot_result.average_time = dot_result.total_time / times.len() as u32;
    dot_result.min_time = *times.iter().min().unwrap();
    dot_result.max_time = *times.iter().max().unwrap();
    dot_result.throughput = Some(vector_size as f64 / dot_result.average_time.as_secs_f64());

    println!("  Dot product 1M elements: {:.2}μs ({:.1}M ops/s)",
             dot_result.average_time.as_micros(),
             dot_result.throughput.unwrap() / 1e6);

    results.push(dot_result);

    // Transpose operation
    let matrix_512 = BitNetTensor::random_normal_with_pool(
        vec![512, 512],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let mut transpose_result = BenchmarkResult::new("transpose_512x512".to_string());
    transpose_result.tensor_size = Some(vec![512, 512]);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = matrix_512.transpose(0, 1)?;
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..config.benchmark_iterations {
        let start = Instant::now();
        let _ = matrix_512.transpose(0, 1)?;
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    transpose_result.iterations = times.len();
    transpose_result.total_time = times.iter().sum();
    transpose_result.average_time = transpose_result.total_time / times.len() as u32;
    transpose_result.min_time = *times.iter().min().unwrap();
    transpose_result.max_time = *times.iter().max().unwrap();

    println!("  Transpose 512x512: {:.2}μs", transpose_result.average_time.as_micros());

    results.push(transpose_result);

    Ok(results)
}

// =============================================================================
// Device Migration Benchmarks
// =============================================================================

fn benchmark_device_migration(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("💻 Benchmarking device migration...");

    let mut results = Vec::new();

    if !is_metal_available() {
        println!("  ⚠️  Metal not available, skipping migration benchmarks");
        return Ok(results);
    }

    let cpu_device = get_cpu_device();
    let metal_device = get_metal_device()?;

    let migration_sizes = vec![
        vec![256, 256],
        vec![512, 512],
        vec![1024, 1024],
    ];

    for shape in migration_sizes {
        let cpu_tensor = BitNetTensor::random_normal_with_pool(
            shape.clone(),
            0.0, 1.0,
            BitNetDType::Float32,
            cpu_device.clone(),
            pool.clone()
        )?;

        // CPU to Metal migration
        let mut cpu_to_metal_result = BenchmarkResult::new(
            format!("cpu_to_metal_{}", shape_to_string(&shape))
        );
        cpu_to_metal_result.tensor_size = Some(shape.clone());

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = cpu_tensor.to_device(&metal_device)?;
        }

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..config.benchmark_iterations {
            let start = Instant::now();
            let _ = cpu_tensor.to_device(&metal_device)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        cpu_to_metal_result.iterations = times.len();
        cpu_to_metal_result.total_time = times.iter().sum();
        cpu_to_metal_result.average_time = cpu_to_metal_result.total_time / times.len() as u32;
        cpu_to_metal_result.min_time = *times.iter().min().unwrap();
        cpu_to_metal_result.max_time = *times.iter().max().unwrap();

        let data_size = shape.iter().product::<usize>() * 4; // f32 = 4 bytes
        let bandwidth = data_size as f64 / cpu_to_metal_result.average_time.as_secs_f64() / 1e9; // GB/s
        cpu_to_metal_result.throughput = Some(bandwidth);

        println!("  CPU -> Metal {}: {:.2}ms ({:.1} GB/s)",
                 shape_to_string(&shape),
                 cpu_to_metal_result.average_time.as_millis(),
                 bandwidth);

        results.push(cpu_to_metal_result);

        // Metal to CPU migration
        let metal_tensor = cpu_tensor.to_device(&metal_device)?;

        let mut metal_to_cpu_result = BenchmarkResult::new(
            format!("metal_to_cpu_{}", shape_to_string(&shape))
        );
        metal_to_cpu_result.tensor_size = Some(shape.clone());

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = metal_tensor.to_device(&cpu_device)?;
        }

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..config.benchmark_iterations {
            let start = Instant::now();
            let _ = metal_tensor.to_device(&cpu_device)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        metal_to_cpu_result.iterations = times.len();
        metal_to_cpu_result.total_time = times.iter().sum();
        metal_to_cpu_result.average_time = metal_to_cpu_result.total_time / times.len() as u32;
        metal_to_cpu_result.min_time = *times.iter().min().unwrap();
        metal_to_cpu_result.max_time = *times.iter().max().unwrap();

        let bandwidth = data_size as f64 / metal_to_cpu_result.average_time.as_secs_f64() / 1e9;
        metal_to_cpu_result.throughput = Some(bandwidth);

        println!("  Metal -> CPU {}: {:.2}ms ({:.1} GB/s)",
                 shape_to_string(&shape),
                 metal_to_cpu_result.average_time.as_millis(),
                 bandwidth);

        results.push(metal_to_cpu_result);
    }

    Ok(results)
}

// =============================================================================
// Acceleration Benchmarks
// =============================================================================

fn benchmark_acceleration(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("⚡ Benchmarking acceleration backends...");

    let mut results = Vec::new();

    // CPU vs Metal comparison (if available)
    if is_metal_available() {
        let cpu_device = get_cpu_device();
        let metal_device = get_metal_device()?;

        let test_size = 512;

        // Create tensors on both devices
        let cpu_tensor_a = BitNetTensor::random_normal_with_pool(
            vec![test_size, test_size],
            0.0, 0.01,
            BitNetDType::Float32,
            cpu_device.clone(),
            pool.clone()
        )?;

        let cpu_tensor_b = BitNetTensor::random_normal_with_pool(
            vec![test_size, test_size],
            0.0, 0.01,
            BitNetDType::Float32,
            cpu_device.clone(),
            pool.clone()
        )?;

        let metal_tensor_a = cpu_tensor_a.to_device(&metal_device)?;
        let metal_tensor_b = cpu_tensor_b.to_device(&metal_device)?;

        // CPU matrix multiplication
        let mut cpu_matmul_result = BenchmarkResult::new("cpu_matmul_512x512".to_string());
        cpu_matmul_result.tensor_size = Some(vec![test_size, test_size]);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = cpu_tensor_a.matmul(&cpu_tensor_b)?;
        }

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..50 { // Fewer iterations for matrix multiplication
            let start = Instant::now();
            let _ = cpu_tensor_a.matmul(&cpu_tensor_b)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        cpu_matmul_result.iterations = times.len();
        cpu_matmul_result.total_time = times.iter().sum();
        cpu_matmul_result.average_time = cpu_matmul_result.total_time / times.len() as u32;
        cpu_matmul_result.min_time = *times.iter().min().unwrap();
        cpu_matmul_result.max_time = *times.iter().max().unwrap();

        let flops = 2.0 * (test_size as f64).powi(3);
        cpu_matmul_result.throughput = Some(flops / cpu_matmul_result.average_time.as_secs_f64() / 1e9);

        println!("  CPU matmul 512x512: {:.2}ms ({:.1} GFLOPS)",
                 cpu_matmul_result.average_time.as_millis(),
                 cpu_matmul_result.throughput.unwrap());

        results.push(cpu_matmul_result);

        // Metal matrix multiplication
        let mut metal_matmul_result = BenchmarkResult::new("metal_matmul_512x512".to_string());
        metal_matmul_result.tensor_size = Some(vec![test_size, test_size]);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = metal_tensor_a.matmul(&metal_tensor_b)?;
        }

        // Benchmark
        let mut times = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            let _ = metal_tensor_a.matmul(&metal_tensor_b)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        metal_matmul_result.iterations = times.len();
        metal_matmul_result.total_time = times.iter().sum();
        metal_matmul_result.average_time = metal_matmul_result.total_time / times.len() as u32;
        metal_matmul_result.min_time = *times.iter().min().unwrap();
        metal_matmul_result.max_time = *times.iter().max().unwrap();
        metal_matmul_result.throughput = Some(flops / metal_matmul_result.average_time.as_secs_f64() / 1e9);

        println!("  Metal matmul 512x512: {:.2}ms ({:.1} GFLOPS)",
                 metal_matmul_result.average_time.as_millis(),
                 metal_matmul_result.throughput.unwrap());

        results.push(metal_matmul_result);

        // Calculate speedup
        if cpu_matmul_result.average_time > Duration::ZERO && metal_matmul_result.average_time > Duration::ZERO {
            let speedup = cpu_matmul_result.average_time.as_nanos() as f64 /
                         metal_matmul_result.average_time.as_nanos() as f64;
            println!("  🚀 Metal speedup: {:.2}x", speedup);
        }
    } else {
        println!("  ⚠️  Metal not available, skipping acceleration benchmarks");
    }

    // MLX benchmarks (if available on Apple Silicon)
    #[cfg(target_os = "macos")]
    {
        if is_mlx_available() {
            println!("  🍎 MLX acceleration available - running MLX benchmarks");
            let mlx_results = benchmark_mlx_acceleration(pool, config)?;
            results.extend(mlx_results);
        }
    }

    Ok(results)
}

// =============================================================================
// Scaling Analysis
// =============================================================================

fn benchmark_scaling_analysis(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("📈 Running scaling analysis...");

    let device = auto_select_device();
    let mut results = Vec::new();

    // Batch processing scaling
    println!("🎯 Batch processing scaling:");

    for &batch_size in &config.batch_sizes {
        let batch_tensor = BitNetTensor::random_normal_with_pool(
            vec![batch_size, 256, 512],
            0.0, 1.0,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let weight_tensor = BitNetTensor::random_normal_with_pool(
            vec![512, 256],
            0.0, 0.01,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let mut batch_result = BenchmarkResult::new(format!("batch_processing_{}", batch_size));
        batch_result.tensor_size = Some(vec![batch_size, 256, 512]);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let _ = batch_tensor.matmul(&weight_tensor)?;
        }

        // Benchmark
        let mut times = Vec::new();
        let iterations = std::cmp::min(config.benchmark_iterations, 50);

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = batch_tensor.matmul(&weight_tensor)?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        batch_result.iterations = times.len();
        batch_result.total_time = times.iter().sum();
        batch_result.average_time = batch_result.total_time / times.len() as u32;
        batch_result.min_time = *times.iter().min().unwrap();
        batch_result.max_time = *times.iter().max().unwrap();

        // Throughput in samples/second
        batch_result.throughput = Some(batch_size as f64 / batch_result.average_time.as_secs_f64());

        println!("  Batch size {}: {:.2}ms ({:.1} samples/s)",
                 batch_size,
                 batch_result.average_time.as_millis(),
                 batch_result.throughput.unwrap());

        results.push(batch_result);
    }

    // Memory scaling analysis
    println!("💾 Memory usage scaling:");

    let sizes = vec![64, 128, 256, 512, 1024];
    let initial_memory = pool.get_metrics().total_allocated;

    for size in sizes {
        let _shape = vec![size, size];

        let before_memory = pool.get_metrics().total_allocated;

        let tensor = BitNetTensor::zeros_with_pool(
            shape.clone(),
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let after_memory = pool.get_metrics().total_allocated;
        let memory_used = after_memory - before_memory;
        let theoretical_memory = size * size * 4; // f32 = 4 bytes
        let memory_efficiency = theoretical_memory as f64 / memory_used as f64;

        let mut memory_result = BenchmarkResult::new(format!("memory_scaling_{}x{}", size, size));
        memory_result.tensor_size = Some(shape);
        memory_result.memory_used = Some(memory_used);
        memory_result.memory_efficiency = Some(memory_efficiency);

        println!("  {}x{}: {} bytes used, {:.1}% efficiency",
                 size, size, memory_used, memory_efficiency * 100.0);

        results.push(memory_result);

        // Clean up
        drop(tensor);
    }

    Ok(results)
}

// =============================================================================
// Memory Efficiency Analysis
// =============================================================================

fn benchmark_memory_efficiency(
    pool: &HybridMemoryPool,
    config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("💾 Analyzing memory efficiency...");

    let device = auto_select_device();
    let mut results = Vec::new();

    // Memory fragmentation test
    println!("🧩 Memory fragmentation analysis:");

    let initial_metrics = pool.get_metrics();
    println!("  Initial state: {} bytes allocated", initial_metrics.total_allocated);

    // Create many small tensors
    let mut small_tensors = Vec::new();
    for i in 0..100 {
        let tensor = BitNetTensor::zeros_with_pool(
            vec![64, 64],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;
        small_tensors.push(tensor);

        if i % 20 == 19 {
            let metrics = pool.get_metrics();
            println!("    After {} small tensors: {} bytes", i + 1, metrics.total_allocated);
        }
    }

    // Remove every other tensor (create fragmentation)
    let mut i = 0;
    small_tensors.retain(|_| {
        i += 1;
        i % 2 == 0
    });

    let fragmented_metrics = pool.get_metrics();
    println!("  After fragmentation: {} bytes allocated", fragmented_metrics.total_allocated);

    // Try to allocate large tensor
    let large_tensor_result = BitNetTensor::zeros_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    );

    match large_tensor_result {
        Ok(_) => {
            println!("  ✅ Large tensor allocation succeeded despite fragmentation");
            let post_large_metrics = pool.get_metrics();
            println!("  After large tensor: {} bytes allocated", post_large_metrics.total_allocated);
        }
        Err(_e) => {
            println!("  ❌ Large tensor allocation failed: {}", e);
        }
    }

    // Memory leak detection
    println!("🔍 Memory leak detection:");

    let leak_test_start = pool.get_metrics().total_allocated;

    // Create and destroy tensors in a loop
    for _ in 0..50 {
        let tensor = BitNetTensor::random_normal_with_pool(
            vec![128, 128],
            0.0, 1.0,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        // Do some operations
        let _ = tensor.add(&tensor)?;
        let _ = tensor.mul(&tensor)?;

        // Tensor is dropped here
    }

    // Force cleanup
    std::thread::sleep(std::time::Duration::from_millis(100));

    let leak_test_end = pool.get_metrics().total_allocated;
    let memory_growth = leak_test_end.saturating_sub(leak_test_start);

    println!("  Memory growth after 50 tensor cycles: {} bytes", memory_growth);

    if memory_growth < 1024 * 1024 { // Less than 1MB growth
        println!("  ✅ No significant memory leaks detected");
    } else {
        println!("  ⚠️  Potential memory leak: {} MB growth", memory_growth / 1024 / 1024);
    }

    // Memory pool utilization analysis
    println!("📊 Memory pool utilization:");

    let final_metrics = pool.get_metrics();
    println!("  Total allocated: {} bytes", final_metrics.total_allocated);
    println!("  Active allocations: {}", final_metrics.active_allocations);
    println!("  Failed allocations: {}", final_metrics.failed_allocations);
    println!("  Pool utilization: {:.1}%", final_metrics.pool_utilization() * 100.0);

    if let Some(detailed) = final_metrics.detailed_metrics() {
        println!("  Small block pool: {:.1}% utilized", detailed.small_pool_utilization * 100.0);
        println!("  Large block pool: {:.1}% utilized", detailed.large_pool_utilization * 100.0);
        println!("  Memory fragmentation: {:.1}%", detailed.fragmentation_ratio * 100.0);
    }

    let mut efficiency_result = BenchmarkResult::new("memory_efficiency_overall".to_string());
    efficiency_result.memory_used = Some(final_metrics.total_allocated);
    efficiency_result.memory_efficiency = Some(final_metrics.pool_utilization());

    results.push(efficiency_result);

    Ok(results)
}

// =============================================================================
// Performance Report Generation
// =============================================================================

fn generate_performance_report(
    results: &[BenchmarkResult],
    pool: &HybridMemoryPool
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Performance Report");
    println!("====================");

    // Summary statistics
    let total_tests = results.len();
    let total_iterations: usize = results.iter().map(|r| r.iterations).sum();
    let total_time: Duration = results.iter().map(|r| r.total_time).sum();

    println!("\n📈 Summary Statistics:");
    println!("  Total tests run: {}", total_tests);
    println!("  Total iterations: {}", total_iterations);
    println!("  Total test time: {:.2}s", total_time.as_secs_f64());

    // Performance highlights
    println!("\n🏆 Performance Highlights:");

    // Fastest operations
    let fastest_result = results.iter()
        .filter(|r| r.average_time > Duration::ZERO)
        .min_by_key(|r| r.average_time);

    if let Some(fastest) = fastest_result {
        println!("  Fastest operation: {} ({:.2}μs)",
                 fastest.test_name, fastest.average_time.as_micros());
    }

    // Highest throughput
    let highest_throughput = results.iter()
        .filter_map(|r| r.throughput.map(|t| (r, t)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    if let Some((result, throughput)) = highest_throughput {
        if result.test_name.contains("GFLOPS") || result.test_name.contains("matmul") {
            println!("  Highest compute throughput: {} ({:.1} GFLOPS)",
                     result.test_name, throughput);
        } else {
            println!("  Highest throughput: {} ({:.1} ops/s)",
                     result.test_name, throughput);
        }
    }

    // Memory efficiency
    let memory_results: Vec<_> = results.iter()
        .filter_map(|r| r.memory_efficiency.map(|e| (r, e)))
        .collect();

    if !memory_results.is_empty() {
        let avg_efficiency: f64 = memory_results.iter().map(|(_, e)| e).sum::<f64>() / memory_results.len() as f64;
        println!("  Average memory efficiency: {:.1}%", avg_efficiency * 100.0);
    }

    // Performance by category
    println!("\n📊 Performance by Category:");

    let categories = vec![
        ("Memory", "tensor_creation"),
        ("Arithmetic", "addition"),
        ("Linear Algebra", "matmul"),
        ("Device Migration", "cpu_to_metal"),
        ("Batch Processing", "batch_processing"),
    ];

    for (category, filter) in categories {
        let category_results: Vec<_> = results.iter()
            .filter(|r| r.test_name.contains(filter))
            .collect();

        if !category_results.is_empty() {
            let avg_time: Duration = category_results.iter()
                .map(|r| r.average_time)
                .sum::<Duration>() / category_results.len() as u32;

            println!("  {}: {:.2}μs average", category, avg_time.as_micros());
        }
    }

    // System resource usage
    println!("\n💻 System Resource Usage:");
    let final_metrics = pool.get_metrics();
    println!("  Memory pool: {} MB used", final_metrics.total_allocated / 1024 / 1024);
    println!("  Active allocations: {}", final_metrics.active_allocations);
    println!("  Pool efficiency: {:.1}%", final_metrics.pool_utilization() * 100.0);

    // Recommendations
    println!("\n💡 Performance Recommendations:");

    // Check for slow operations
    let slow_threshold = Duration::from_millis(10);
    let slow_ops: Vec<_> = results.iter()
        .filter(|r| r.average_time > slow_threshold)
        .collect();

    if !slow_ops.is_empty() {
        println!("  ⚠️  {} operations slower than 10ms - consider optimization", slow_ops.len());
        for op in slow_ops.iter().take(3) {
            println!("    - {}: {:.2}ms", op.test_name, op.average_time.as_millis());
        }
    }

    // Check memory efficiency
    if let Some(efficiency) = memory_results.first().map(|(_, e)| *e) {
        if efficiency < 0.8 {
            println!("  ⚠️  Memory efficiency below 80% - check for memory overhead");
        } else if efficiency > 0.95 {
            println!("  ✅ Excellent memory efficiency (>95%)");
        }
    }

    // Check for device utilization
    let device_tests = results.iter().any(|r| r.test_name.contains("metal") || r.test_name.contains("gpu"));
    if device_tests {
        println!("  ✅ GPU acceleration tested - verify optimal performance on target hardware");
    } else {
        println!("  ℹ️  No GPU tests run - GPU acceleration may provide additional speedup");
    }

    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

fn create_performance_memory_pool() -> Result<HybridMemoryPool, Box<dyn std::error::Error>> {
    let config = MemoryPoolConfig {
        small_block_size: 64 * 1024,         // 64KB
        large_block_threshold: 1024 * 1024,  // 1MB
        initial_pool_size: 64 * 1024 * 1024, // 64MB
        max_pool_size: 1024 * 1024 * 1024,   // 1GB
        tracking: TrackingConfig {
            level: TrackingLevel::Detailed,
            enable_stack_traces: false,
            enable_metrics: true,
        },
    };

    HybridMemoryPool::new_with_config(config).map_err(Into::into)
}

fn print_system_info() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💻 System Information:");
    println!("---------------------");

    // Device information
    let auto_device = auto_select_device();
    println!("  Auto-selected device: {:?}", auto_device);

    let cpu_device = get_cpu_device();
    println!("  CPU device: {:?}", cpu_device);

    let metal_available = is_metal_available();
    println!("  Metal GPU: {}", if metal_available { "Available" } else { "Not available" });

    if metal_available {
        match get_metal_device() {
            Ok(metal_device) => println!("  Metal device: {:?}", metal_device),
            Err(_e) => println!("  Metal device error: {}", e),
        }
    }

    #[cfg(target_os = "macos")]
    {
        let mlx_available = is_mlx_available();
        println!("  MLX acceleration: {}", if mlx_available { "Available" } else { "Not available" });
    }

    // Memory information
    println!("  Available CPU cores: {}", num_cpus::get());

    Ok(())
}

fn print_test_configuration(config: &PerformanceConfig) {
    println!("\n⚙️  Test Configuration:");
    println!("  Warmup iterations: {}", config.warmup_iterations);
    println!("  Benchmark iterations: {}", config.benchmark_iterations);
    println!("  Memory test sizes: {} sizes", config.memory_test_sizes.len());
    println!("  Operation test shapes: {} shapes", config.operation_test_shapes.len());
    println!("  Batch sizes: {:?}", config.batch_sizes);
    println!("  Max test duration: {:?}", config.max_test_duration);
}

fn shape_to_string(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("{}", shape[0])
    } else if shape.len() == 2 {
        format!("{}x{}", shape[0], shape[1])
    } else {
        format!("{:?}", shape)
    }
}

// Placeholder functions for MLX (would be implemented if available)
#[cfg(target_os = "macos")]
fn is_mlx_available() -> bool {
    // In real implementation, would check for MLX framework availability
    false
}

#[cfg(target_os = "macos")]
fn benchmark_mlx_acceleration(
    _pool: &HybridMemoryPool,
    _config: &PerformanceConfig
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    Ok(Vec::new()) // Placeholder
}

// External crate for CPU core count
extern crate num_cpus;
