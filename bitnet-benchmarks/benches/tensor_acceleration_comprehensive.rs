//! Comprehensive Tensor Acceleration Benchmarks
//!
//! Day 21: Acceleration Testing and Validation
//! Performance benchmarking for tensor acceleration systems:
//! - MLX acceleration validation (15-40x speedup targets)
//! - SIMD optimization benchmarks across platforms
//! - Cross-platform acceleration comparison

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use std::sync::Arc;

#[cfg(feature = "mlx")]
use std::time::Duration;

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{operations::BitNetMlxOps, BitNetMlxDevice, MlxTensor};

use bitnet_core::device::auto_select_device;
use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::tensor::ops::simd::{simd_add_f32, simd_mul_f32, simd_sum_f32};
use bitnet_core::tensor::{BitNetDType, BitNetTensor};

// ============================================================================
// Acceleration Benchmark Configuration
// ============================================================================

struct AccelerationBenchmarkConfig {
    matrix_sizes: Vec<(usize, usize)>,
    vector_sizes: Vec<usize>,
    data_types: Vec<BitNetDType>,
    iterations: usize,
    warmup_iterations: usize,
    memory_pool: Arc<HybridMemoryPool>,
}

impl Default for AccelerationBenchmarkConfig {
    fn default() -> Self {
        Self {
            matrix_sizes: vec![(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
            vector_sizes: vec![1_000, 10_000, 100_000, 1_000_000],
            data_types: vec![BitNetDType::F32],
            iterations: 100,
            warmup_iterations: 10,
            memory_pool: Arc::new(HybridMemoryPool::new().expect("Failed to create memory pool")),
        }
    }
}

// ============================================================================
// MLX Acceleration Benchmarks
// ============================================================================

#[cfg(feature = "mlx")]
fn bench_mlx_matrix_multiplication(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("mlx_acceleration/matrix_multiplication");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Test MLX availability first
    if !bitnet_core::mlx::is_mlx_available() {
        println!("⚠️  MLX not available, skipping MLX benchmarks");
        return;
    }

    let mlx_device = BitNetMlxDevice::default();
    println!(
        "🚀 Running MLX acceleration benchmarks on: {:?}",
        mlx_device.device_type()
    );

    for &(rows, cols) in &config.matrix_sizes {
        for &dtype in &config.data_types {
            let benchmark_id = format!("{}x{}_{:?}", rows, cols, dtype);

            group.throughput(Throughput::Elements((rows * cols * rows) as u64));

            group.bench_with_input(
                BenchmarkId::new("mlx", &benchmark_id),
                &(rows, cols, dtype),
                |bencher, &(rows, cols, dtype)| {
                    let tensor_a = MlxTensor::randn(&[rows, cols], dtype, mlx_device.clone())
                        .expect("Failed to create MLX tensor A");
                    let tensor_b = MlxTensor::randn(&[cols, rows], dtype, mlx_device.clone())
                        .expect("Failed to create MLX tensor B");

                    // Warmup
                    for _ in 0..config.warmup_iterations {
                        let _ =
                            BitNetMlxOps::matmul(&tensor_a, &tensor_b).expect("MLX matmul failed");
                    }

                    bencher.iter(|| {
                        let result =
                            BitNetMlxOps::matmul(&tensor_a, &tensor_b).expect("MLX matmul failed");
                        black_box(result)
                    });
                },
            );

            // Benchmark CPU baseline for comparison
            group.bench_with_input(
                BenchmarkId::new("cpu_baseline", &benchmark_id),
                &(rows, cols, dtype),
                |bencher, &(rows, cols, dtype)| {
                    use candle_core::{Device, Tensor};
                    let device = Device::Cpu;

                    let tensor_a = Tensor::randn(0f32, 1f32, (rows, cols), &device)
                        .expect("Failed to create CPU tensor A");
                    let tensor_b = Tensor::randn(0f32, 1f32, (cols, rows), &device)
                        .expect("Failed to create CPU tensor B");

                    // Warmup
                    for _ in 0..config.warmup_iterations {
                        let _ = tensor_a.matmul(&tensor_b).expect("CPU matmul failed");
                    }

                    bencher.iter(|| {
                        let result = tensor_a.matmul(&tensor_b).expect("CPU matmul failed");
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "mlx")]
fn bench_mlx_element_wise_operations(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("mlx_acceleration/element_wise");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    if !bitnet_core::mlx::is_mlx_available() {
        return;
    }

    let mlx_device = BitNetMlxDevice::default();
    let operations = vec!["add", "multiply"];

    for &size in &config.vector_sizes {
        let _shape = vec![size];

        for &dtype in &config.data_types {
            for operation in &operations {
                let benchmark_id = format!("{}_{:?}_{}", size, dtype, operation);

                group.throughput(Throughput::Elements(size as u64));

                group.bench_with_input(
                    BenchmarkId::new("mlx", &benchmark_id),
                    &(size, dtype, operation.clone()),
                    |bencher, &(size, dtype, ref operation)| {
                        let tensor_a = MlxTensor::randn(&shape, dtype, mlx_device.clone())
                            .expect("Failed to create MLX tensor A");
                        let tensor_b = MlxTensor::randn(&shape, dtype, mlx_device.clone())
                            .expect("Failed to create MLX tensor B");

                        // Warmup
                        for _ in 0..config.warmup_iterations {
                            let _ = match operation.as_str() {
                                "add" => BitNetMlxOps::add(&tensor_a, &tensor_b),
                                "multiply" => BitNetMlxOps::multiply(&tensor_a, &tensor_b),
                                _ => Ok(tensor_a.clone()),
                            }
                            .expect("MLX operation failed");
                        }

                        bencher.iter(|| {
                            let result = match operation.as_str() {
                                "add" => BitNetMlxOps::add(&tensor_a, &tensor_b),
                                "multiply" => BitNetMlxOps::multiply(&tensor_a, &tensor_b),
                                _ => Ok(tensor_a.clone()),
                            }
                            .expect("MLX operation failed");
                            black_box(result)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(feature = "mlx")]
fn bench_mlx_quantization_operations(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("mlx_acceleration/quantization");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    if !bitnet_core::mlx::is_mlx_available() {
        return;
    }

    let mlx_device = BitNetMlxDevice::default();

    for &(rows, cols) in &config.matrix_sizes {
        for &dtype in &config.data_types {
            let benchmark_id = format!("{}x{}_{:?}", rows, cols, dtype);

            group.throughput(Throughput::Elements((rows * cols) as u64));

            group.bench_with_input(
                BenchmarkId::new("mlx_1_58_bit_quantization", &benchmark_id),
                &(rows, cols, dtype),
                |bencher, &(rows, cols, dtype)| {
                    let tensor = MlxTensor::randn(&[rows, cols], dtype, mlx_device.clone())
                        .expect("Failed to create MLX tensor");

                    // Warmup
                    for _ in 0..config.warmup_iterations {
                        let _ = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1))
                            .expect("MLX quantization failed");
                    }

                    bencher.iter(|| {
                        let result = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1))
                            .expect("MLX quantization failed");
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// SIMD Optimization Benchmarks
// ============================================================================

fn bench_simd_tensor_operations(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("simd_acceleration/tensor_operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    println!("⚡ Running SIMD tensor operation benchmarks");

    let device = auto_select_device();

    for &size in &config.vector_sizes {
        let shape = vec![size];

        group.throughput(Throughput::Elements(size as u64));

        // Element-wise addition benchmark
        group.bench_with_input(
            BenchmarkId::new("tensor_simd_add", size),
            &size,
            |bencher, &_size| {
                let tensor_a = BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor A");
                let tensor_b = BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor B");

                bencher.iter(|| {
                    let result = simd_add_f32(&tensor_a, &tensor_b).expect("SIMD add failed");
                    black_box(result)
                });
            },
        );

        // Element-wise multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("tensor_simd_mul", size),
            &size,
            |bencher, &_size| {
                let tensor_a = BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor A");
                let tensor_b = BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor B");

                bencher.iter(|| {
                    let result = simd_mul_f32(&tensor_a, &tensor_b).expect("SIMD mul failed");
                    black_box(result)
                });
            },
        );

        // Reduction operations benchmark
        group.bench_with_input(
            BenchmarkId::new("tensor_simd_sum", size),
            &size,
            |bencher, &_size| {
                let tensor_a = BitNetTensor::zeros(&shape, BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor A");

                bencher.iter(|| {
                    let result = simd_sum_f32(&tensor_a).expect("SIMD sum failed");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Performance Validation and Speedup Analysis
// ============================================================================

#[cfg(feature = "mlx")]
fn bench_speedup_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("acceleration_validation/speedup_analysis");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(5));

    println!("🎯 Running speedup validation benchmarks - target: 15-40x MLX speedup");

    let test_sizes = vec![(512, 512), (1024, 1024)];

    for &(rows, cols) in &test_sizes {
        // CPU baseline timing
        group.bench_with_input(
            BenchmarkId::new("cpu_baseline_validation", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                use candle_core::{Device, Tensor};
                let device = Device::Cpu;

                let tensor_a = Tensor::randn(0f32, 1f32, (rows, cols), &device)
                    .expect("Failed to create CPU tensor A");
                let tensor_b = Tensor::randn(0f32, 1f32, (cols, rows), &device)
                    .expect("Failed to create CPU tensor B");

                bencher.iter(|| {
                    let result = tensor_a.matmul(&tensor_b).expect("CPU matmul failed");
                    black_box(result)
                });
            },
        );

        // MLX acceleration timing
        if bitnet_core::mlx::is_mlx_available() {
            group.bench_with_input(
                BenchmarkId::new("mlx_acceleration_validation", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    let mlx_device = BitNetMlxDevice::default();
                    let tensor_a =
                        MlxTensor::randn(&[rows, cols], BitNetDType::F32, mlx_device.clone())
                            .expect("Failed to create MLX tensor A");
                    let tensor_b =
                        MlxTensor::randn(&[cols, rows], BitNetDType::F32, mlx_device.clone())
                            .expect("Failed to create MLX tensor B");

                    bencher.iter(|| {
                        let result =
                            BitNetMlxOps::matmul(&tensor_a, &tensor_b).expect("MLX matmul failed");
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Memory Transfer Performance
// ============================================================================

#[cfg(feature = "mlx")]
fn bench_mlx_memory_transfer_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("acceleration_validation/mlx_memory_transfer");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    if !bitnet_core::mlx::is_mlx_available() {
        return;
    }

    let transfer_sizes = vec![1_024, 10_240, 102_400, 1_024_000];

    for &size in &transfer_sizes {
        let benchmark_id = format!("{}elements", size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("mlx_cpu_to_gpu", &benchmark_id),
            &size,
            |bencher, &size| {
                let cpu_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let mlx_device = BitNetMlxDevice::default();

                bencher.iter(|| {
                    let mlx_tensor = MlxTensor::from_slice(
                        &cpu_data,
                        &[size],
                        BitNetDType::F32,
                        mlx_device.clone(),
                    )
                    .expect("Failed to create MLX tensor from CPU data");
                    black_box(mlx_tensor)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mlx_gpu_to_cpu", &benchmark_id),
            &size,
            |bencher, &size| {
                let mlx_device = BitNetMlxDevice::default();
                let mlx_tensor = MlxTensor::randn(&[size], BitNetDType::F32, mlx_device.clone())
                    .expect("Failed to create MLX tensor");

                bencher.iter(|| {
                    let cpu_data = mlx_tensor
                        .to_vec::<f32>()
                        .expect("Failed to convert MLX tensor to CPU data");
                    black_box(cpu_data)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Pool Performance Benchmarks
// ============================================================================

fn bench_memory_pool_acceleration(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("acceleration_validation/memory_pool_performance");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    println!(
        "💾 Running memory pool acceleration benchmarks with {} iterations",
        config.iterations
    );

    let device = auto_select_device();

    for &(rows, cols) in &config.matrix_sizes {
        group.throughput(Throughput::Elements((rows * cols) as u64));

        // Benchmark memory pool allocation/deallocation patterns
        group.bench_with_input(
            BenchmarkId::new("memory_pool_allocation_pattern", format!("{rows}x{cols}")),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let memory_pool = Arc::clone(&config.memory_pool);
                let allocation_size = rows * cols * std::mem::size_of::<f32>();

                bencher.iter(|| {
                    // Test allocation pattern with config iterations
                    let mut allocations = Vec::with_capacity(config.iterations.min(50)); // Cap at 50

                    for _ in 0..config.iterations.min(50) {
                        if let Ok(allocation) = memory_pool.allocate(allocation_size, 32, &device) {
                            allocations.push(allocation);
                        }
                    }

                    // Allocations are dropped here, testing deallocation performance
                    black_box(allocations)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Configuration-Based Iteration Benchmarks
// ============================================================================

fn bench_configurable_iteration_performance(c: &mut Criterion) {
    let config = AccelerationBenchmarkConfig::default();

    let mut group = c.benchmark_group("acceleration_validation/configurable_iterations");

    println!(
        "🔄 Running configurable iteration benchmarks with {} warmup and {} iterations",
        config.warmup_iterations, config.iterations
    );

    let device = auto_select_device();
    let test_size = 256;
    let shape = vec![test_size, test_size];

    group.throughput(Throughput::Elements((test_size * test_size) as u64));

    // Benchmark using all config data types
    for &dtype in &config.data_types {
        group.bench_with_input(
            BenchmarkId::new("data_type_performance", format!("{dtype:?}")),
            &dtype,
            |bencher, &dtype| {
                // Create test tensors
                let tensor_a = BitNetTensor::zeros(&shape, dtype, Some(device.clone()))
                    .expect("Failed to create tensor A");
                let tensor_b = BitNetTensor::zeros(&shape, dtype, Some(device.clone()))
                    .expect("Failed to create tensor B");

                // Use config.warmup_iterations for actual warmup
                for _ in 0..config.warmup_iterations {
                    let _ = simd_add_f32(&tensor_a, &tensor_b);
                }

                bencher.iter(|| {
                    // Use config.iterations to control benchmark precision
                    let mut results = Vec::with_capacity(config.iterations.min(100));

                    for _ in 0..config.iterations.min(100) {
                        if let Ok(result) = simd_add_f32(&tensor_a, &tensor_b) {
                            results.push(result);
                        }
                    }

                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark Registration
// ============================================================================

#[cfg(feature = "mlx")]
criterion_group!(
    mlx_acceleration_benches,
    bench_mlx_matrix_multiplication,
    bench_mlx_element_wise_operations,
    bench_mlx_quantization_operations,
    bench_speedup_validation,
    bench_mlx_memory_transfer_performance
);

criterion_group!(simd_acceleration_benches, bench_simd_tensor_operations);

criterion_group!(
    config_usage_benches,
    bench_memory_pool_acceleration,
    bench_configurable_iteration_performance
);

#[cfg(feature = "mlx")]
criterion_main!(
    mlx_acceleration_benches,
    simd_acceleration_benches,
    config_usage_benches
);

#[cfg(not(feature = "mlx"))]
criterion_main!(simd_acceleration_benches, config_usage_benches);
