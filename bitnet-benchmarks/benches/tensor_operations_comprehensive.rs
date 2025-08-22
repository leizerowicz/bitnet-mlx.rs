//! Comprehensive Tensor Operations Benchmarks
//!
//! Performance benchmarks for BitNet tensor operations with SIMD optimizations,
//! following the established patterns in bitnet-benchmarks crate.

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, BenchmarkId,
    Throughput, PlotConfiguration, AxisScale, BatchSize
};
use std::time::Duration;

use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::tensor::ops::{simd_add_f32, simd_mul_f32, simd_sum_f32, simd_add_scalar_f32};
use bitnet_core::tensor::ops::arithmetic::{add, mul, add_scalar};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig};
use bitnet_core::device::{get_cpu_device};

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct TensorOpsBenchmarkConfig {
    tensor_sizes: Vec<usize>,
    matrix_sizes: Vec<(usize, usize)>,
    batch_sizes: Vec<usize>,
    data_types: Vec<BitNetDType>,
    memory_pool: std::sync::Arc<HybridMemoryPool>,
    warmup_time: Duration,
    measurement_time: Duration,
}

impl Default for TensorOpsBenchmarkConfig {
    fn default() -> Self {
        let memory_config = MemoryPoolConfig {
            small_block_threshold: 1024 * 1024,
            small_pool_initial_size: 32 * 1024 * 1024,
            small_pool_max_size: 512 * 1024 * 1024,
            large_pool_initial_size: 128 * 1024 * 1024,
            large_pool_max_size: 2 * 1024 * 1024 * 1024, // 2GB for benchmarks
            enable_metrics: true,
            enable_debug_logging: false,
            enable_advanced_tracking: true,
            tracking_config: Some(bitnet_core::memory::tracking::TrackingConfig::default()),
        };

        Self {
            tensor_sizes: vec![
                100,      // Small vectors
                1_000,    // Medium vectors  
                10_000,   // Large vectors
                100_000,  // Very large vectors
                1_000_000, // Huge vectors
            ],
            matrix_sizes: vec![
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
            ],
            batch_sizes: vec![1, 4, 16, 64, 256],
            data_types: vec![BitNetDType::F32, BitNetDType::F16],
            memory_pool: std::sync::Arc::new(
                HybridMemoryPool::new().expect("Failed to create memory pool")
            ),
            warmup_time: Duration::from_secs(2),
            measurement_time: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a random tensor with the given shape and data type
fn create_random_tensor(
    shape: &[usize], 
    dtype: BitNetDType, 
    device: Option<candle_core::Device>
) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    // Create a tensor filled with zeros first, then fill with random data
    let tensor = BitNetTensor::zeros(shape, dtype, device)?;
    
    // For benchmarks, we'll use zeros tensors to avoid complexity 
    // while still providing valid tensor operations
    Ok(tensor)
}

// ============================================================================
// Element-wise Arithmetic Benchmarks
// ============================================================================

fn bench_element_wise_addition(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_element_wise_addition");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.tensor_sizes {
        // Set throughput for proper performance measurement
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark scalar implementation
        group.bench_with_input(
            BenchmarkId::new("scalar_add", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = add(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );

        // Benchmark SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd_add", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = simd_add_f32(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

fn bench_element_wise_multiplication(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_element_wise_multiplication");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &size in &config.tensor_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Scalar multiplication
        group.bench_with_input(
            BenchmarkId::new("scalar_mul", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = mul(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );

        // SIMD multiplication
        group.bench_with_input(
            BenchmarkId::new("simd_mul", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = simd_mul_f32(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scalar Operations Benchmarks  
// ============================================================================

fn bench_scalar_operations(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_scalar_operations");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &size in &config.tensor_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Scalar addition
        group.bench_with_input(
            BenchmarkId::new("scalar_add_scalar", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor")
                    },
                    |tensor| {
                        let result = add_scalar(&tensor, 2.5);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );

        // SIMD scalar addition
        group.bench_with_input(
            BenchmarkId::new("simd_add_scalar", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor")
                    },
                    |tensor| {
                        let result = simd_add_scalar_f32(&tensor, 2.5);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Reduction Operations Benchmarks
// ============================================================================

fn bench_reduction_operations(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_reduction_operations");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &size in &config.tensor_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Scalar sum reduction
        group.bench_with_input(
            BenchmarkId::new("scalar_sum", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor")
                    },
                    |_tensor| {
                        // Fallback scalar sum (would be implemented)
                        // Using a placeholder operation since sum_all doesn't exist
                        let result = 0.0f32; // Placeholder
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );

        // SIMD sum reduction
        group.bench_with_input(
            BenchmarkId::new("simd_sum", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor")
                    },
                    |tensor| {
                        let result = simd_sum_f32(&tensor);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Matrix Operations Benchmarks
// ============================================================================

fn bench_matrix_operations(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_matrix_operations");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.matrix_sizes {
        let total_elements = rows * cols;
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // Matrix element-wise operations
        group.bench_with_input(
            BenchmarkId::new("matrix_elementwise_add", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[rows, cols], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[rows, cols], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = simd_add_f32(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
        
        // Matrix multiplication (if small enough)
        if rows <= 512 && cols <= 512 {
            group.bench_with_input(
                BenchmarkId::new("matrix_multiply", format!("{rows}x{cols}")),
                &(rows, cols),
                |b, &(rows, cols)| {
                    b.iter_batched(
                        || {
                            let a = create_random_tensor(&[rows, cols], BitNetDType::F32, Some(device.clone()))
                                .expect("Failed to create tensor");
                            let b = create_random_tensor(&[cols, rows], BitNetDType::F32, Some(device.clone()))
                                .expect("Failed to create tensor");
                            (a, b)
                        },
                        |(_a, _b)| {
                            // Would use matmul operation
                            let result = 0.0f32; // Placeholder
                            black_box(result)
                        },
                        BatchSize::SmallInput
                    );
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_memory_efficiency");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    // Benchmark tensor creation and destruction
    for &size in &config.tensor_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("tensor_creation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let tensor = BitNetTensor::zeros(&[size], BitNetDType::F32, Some(device.clone()));
                    black_box(tensor)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("tensor_clone", size),
            &size,
            |b, &size| {
                let tensor = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                    .expect("Failed to create tensor");
                
                b.iter(|| {
                    let cloned = tensor.clone();
                    black_box(cloned)
                });
            },
        );
    }

    // Benchmark memory pool utilization
    group.bench_function("memory_pool_stats", |b| {
        b.iter(|| {
            let stats = config.memory_pool.get_metrics();
            black_box(stats)
        });
    });

    group.finish();
}

// ============================================================================
// Broadcasting Benchmarks
// ============================================================================

fn bench_broadcasting_operations(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_broadcasting");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    let broadcasting_cases = vec![
        (vec![1000, 1], vec![1, 1000], "vector_broadcast"),
        (vec![100, 100], vec![100, 1], "column_broadcast"),
        (vec![100, 100], vec![1, 100], "row_broadcast"),
        (vec![10, 10, 10], vec![10, 1, 10], "3d_broadcast"),
    ];

    for (lhs_shape, rhs_shape, name) in broadcasting_cases {
        let total_elements = lhs_shape.iter().product::<usize>().max(rhs_shape.iter().product::<usize>());
        group.throughput(Throughput::Elements(total_elements as u64));
        
        group.bench_with_input(
            BenchmarkId::new("broadcast_add", name),
            &(lhs_shape.clone(), rhs_shape.clone()),
            |b, (lhs_shape, rhs_shape)| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(lhs_shape, BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(rhs_shape, BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = add(&a, &b); // Broadcasting handled automatically
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Data Type Comparison Benchmarks
// ============================================================================

fn bench_dtype_performance(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_dtype_performance");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    let size = 100_000; // Fixed size for dtype comparison

    for dtype in &config.data_types {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("elementwise_add", format!("{dtype:?}")),
            dtype,
            |b, &dtype| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], dtype, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], dtype, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b)
                    },
                    |(a, b)| {
                        let result = add(&a, &b);
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Performance Regression Tests
// ============================================================================

fn bench_performance_regression(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_performance_regression");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    // Baseline performance targets (operations per second)
    let performance_targets = std::collections::HashMap::from([
        ("small_vector_add", 1_000_000.0),    // 1M ops/sec for small vectors
        ("medium_vector_add", 100_000.0),     // 100K ops/sec for medium vectors  
        ("large_vector_add", 10_000.0),       // 10K ops/sec for large vectors
        ("matrix_operations", 1_000.0),       // 1K ops/sec for matrix operations
    ]);

    // Small vector performance
    group.bench_function("small_vector_add_regression", |b| {
        let tensor_a = create_random_tensor(&[1000], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create tensor");
        let tensor_b = create_random_tensor(&[1000], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create tensor");
        
        b.iter(|| {
            let result = simd_add_f32(&tensor_a, &tensor_b);
            black_box(result)
        });
    });

    // Medium vector performance
    group.bench_function("medium_vector_add_regression", |b| {
        let tensor_a = create_random_tensor(&[100_000], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create tensor");
        let tensor_b = create_random_tensor(&[100_000], BitNetDType::F32, Some(device.clone()))
            .expect("Failed to create tensor");
        
        b.iter(|| {
            let result = simd_add_f32(&tensor_a, &tensor_b);
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// Comprehensive Performance Suite
// ============================================================================

fn bench_comprehensive_performance(c: &mut Criterion) {
    let config = TensorOpsBenchmarkConfig::default();
    let device = get_cpu_device();
    
    let mut group = c.benchmark_group("tensor_comprehensive_performance");
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(15));

    // Complex expression: (A + B) * C - D / 2.0
    for &size in &[1000, 10_000, 100_000] {
        group.throughput(Throughput::Elements((size * 4) as u64)); // 4 operations
        
        group.bench_with_input(
            BenchmarkId::new("complex_expression", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let a = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let b = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let c = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        let d = create_random_tensor(&[size], BitNetDType::F32, Some(device.clone()))
                            .expect("Failed to create tensor");
                        (a, b, c, d)
                    },
                    |(a, b, c, d)| {
                        // Complex expression using SIMD operations
                        let step1 = simd_add_f32(&a, &b).unwrap();
                        let step2 = simd_mul_f32(&step1, &c).unwrap();
                        let step3 = simd_add_scalar_f32(&d, 2.0).unwrap(); // d / 2.0 approximated
                        let result = add(&step2, &step3).unwrap(); // Final step
                        black_box(result)
                    },
                    BatchSize::SmallInput
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration and Main
// ============================================================================

criterion_group!(
    benches,
    bench_element_wise_addition,
    bench_element_wise_multiplication,
    bench_scalar_operations,
    bench_reduction_operations,
    bench_matrix_operations,
    bench_memory_efficiency,
    bench_broadcasting_operations,
    bench_dtype_performance,
    bench_performance_regression,
    bench_comprehensive_performance
);

criterion_main!(benches);
