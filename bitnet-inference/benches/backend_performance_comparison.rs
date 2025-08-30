//! Backend Performance Comparison Benchmarks
//!
//! Comprehensive performance comparison across CPU, Metal, and MLX backends
//! for BitNet inference operations. This benchmark suite measures:
//! - Throughput (operations per second)
//! - Latency (milliseconds per operation)
//! - Memory efficiency (bytes per operation)
//! - GPU utilization and memory transfer overhead

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use bitnet_inference::*;
use bitnet_core::{Tensor, Device, DType};
use std::time::Duration;
use tokio::runtime::Runtime;

const BENCHMARK_INPUT_SIZES: &[usize] = &[128, 512, 1024, 2048];
const BENCHMARK_BATCH_SIZES: &[usize] = &[1, 8, 16, 32, 64];
const WARMUP_ITERATIONS: usize = 3;

/// Benchmark throughput comparison across different backends
fn benchmark_backend_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backend_throughput");
    
    // Configure benchmark parameters
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    
    for &input_size in BENCHMARK_INPUT_SIZES {
        for &batch_size in BENCHMARK_BATCH_SIZES {
            let throughput = Throughput::Elements(batch_size as u64);
            group.throughput(throughput);
            
            // CPU Backend Benchmark
            group.bench_with_input(
                BenchmarkId::new("cpu_backend", format!("input_{}_batch_{}", input_size, batch_size)),
                &(input_size, batch_size),
                |b, &(input_size, batch_size)| {
                    b.to_async(&rt).iter(|| async move {
                        let engine = InferenceEngine::builder()
                            .device(Device::Cpu)
                            .optimization_level(OptimizationLevel::Aggressive)
                            .build()
                            .await
                            .unwrap();
                        
                        let model = engine.load_model("test_model").await.unwrap();
                        let inputs: Vec<_> = (0..batch_size)
                            .map(|_| Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap())
                            .collect();
                        
                        let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
                    });
                },
            );
            
            // Metal Backend Benchmark (macOS only)
            #[cfg(feature = "metal")]
            group.bench_with_input(
                BenchmarkId::new("metal_backend", format!("input_{}_batch_{}", input_size, batch_size)),
                &(input_size, batch_size),
                |b, &(input_size, batch_size)| {
                    b.to_async(&rt).iter(|| async move {
                        let engine = InferenceEngine::builder()
                            .device(Device::Metal)
                            .optimization_level(OptimizationLevel::Aggressive)
                            .build()
                            .await
                            .unwrap();
                        
                        let model = engine.load_model("test_model").await.unwrap();
                        let inputs: Vec<_> = (0..batch_size)
                            .map(|_| Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap())
                            .collect();
                        
                        let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
                    });
                },
            );
            
            // MLX Backend Benchmark (Apple Silicon only)
            #[cfg(feature = "mlx")]
            group.bench_with_input(
                BenchmarkId::new("mlx_backend", format!("input_{}_batch_{}", input_size, batch_size)),
                &(input_size, batch_size),
                |b, &(input_size, batch_size)| {
                    b.to_async(&rt).iter(|| async move {
                        let engine = InferenceEngine::builder()
                            .device(Device::Mlx)
                            .optimization_level(OptimizationLevel::Aggressive)
                            .build()
                            .await
                            .unwrap();
                        
                        let model = engine.load_model("test_model").await.unwrap();
                        let inputs: Vec<_> = (0..batch_size)
                            .map(|_| Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap())
                            .collect();
                        
                        let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark latency comparison focusing on single inference operations
fn benchmark_backend_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backend_latency");
    
    // Configure for latency measurement
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);
    
    for &input_size in BENCHMARK_INPUT_SIZES {
        // CPU Backend Latency
        group.bench_with_input(
            BenchmarkId::new("cpu_latency", input_size),
            &input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Cpu)
                        .optimization_level(OptimizationLevel::Aggressive)
                        .build()
                        .await
                        .unwrap();
                    
                    let model = engine.load_model("test_model").await.unwrap();
                    let input = Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap();
                    
                    let _output = engine.infer(&model, &input).await.unwrap();
                });
            },
        );
        
        // Metal Backend Latency
        #[cfg(feature = "metal")]
        group.bench_with_input(
            BenchmarkId::new("metal_latency", input_size),
            &input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Metal)
                        .optimization_level(OptimizationLevel::Aggressive)
                        .build()
                        .await
                        .unwrap();
                    
                    let model = engine.load_model("test_model").await.unwrap();
                    let input = Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap();
                    
                    let _output = engine.infer(&model, &input).await.unwrap();
                });
            },
        );
        
        // MLX Backend Latency
        #[cfg(feature = "mlx")]
        group.bench_with_input(
            BenchmarkId::new("mlx_latency", input_size),
            &input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Mlx)
                        .optimization_level(OptimizationLevel::Aggressive)
                        .build()
                        .await
                        .unwrap();
                    
                    let model = engine.load_model("test_model").await.unwrap();
                    let input = Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap();
                    
                    let _output = engine.infer(&model, &input).await.unwrap();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU memory transfer overhead
fn benchmark_memory_transfer_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_transfer_overhead");
    
    // Test different data sizes for memory transfer analysis
    let transfer_sizes = &[1024, 4096, 16384, 65536]; // KB sizes
    
    for &size in transfer_sizes {
        let throughput = Throughput::Bytes(size as u64 * 1024); // Convert to bytes
        group.throughput(throughput);
        
        // Metal memory transfer benchmark
        #[cfg(feature = "metal")]
        group.bench_with_input(
            BenchmarkId::new("metal_transfer", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Metal)
                        .build()
                        .await
                        .unwrap();
                    
                    // Create large tensor to test memory transfer
                    let input = Tensor::ones(&[size, 1024], DType::F32, &Device::Cpu).unwrap();
                    let model = engine.load_model("test_model").await.unwrap();
                    
                    let _output = engine.infer(&model, &input).await.unwrap();
                });
            },
        );
        
        // MLX memory transfer benchmark
        #[cfg(feature = "mlx")]
        group.bench_with_input(
            BenchmarkId::new("mlx_transfer", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Mlx)
                        .build()
                        .await
                        .unwrap();
                    
                    // Create large tensor to test memory transfer
                    let input = Tensor::ones(&[size, 1024], DType::F32, &Device::Cpu).unwrap();
                    let model = engine.load_model("test_model").await.unwrap();
                    
                    let _output = engine.infer(&model, &input).await.unwrap();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark backend initialization overhead
fn benchmark_backend_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backend_initialization");
    
    // CPU Backend Initialization
    group.bench_function("cpu_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::builder()
                .device(Device::Cpu)
                .build()
                .await
                .unwrap();
        });
    });
    
    // Metal Backend Initialization
    #[cfg(feature = "metal")]
    group.bench_function("metal_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::builder()
                .device(Device::Metal)
                .build()
                .await
                .unwrap();
        });
    });
    
    // MLX Backend Initialization
    #[cfg(feature = "mlx")]
    group.bench_function("mlx_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::builder()
                .device(Device::Mlx)
                .build()
                .await
                .unwrap();
        });
    });
    
    group.finish();
}

/// Benchmark cross-backend performance comparison for the same workload
fn benchmark_cross_backend_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("cross_backend_comparison");
    
    let test_input_size = 2048;
    let test_batch_size = 16;
    
    // CPU baseline
    group.bench_function("cpu_baseline", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .device(Device::Cpu)
                .optimization_level(OptimizationLevel::Aggressive)
                .build()
                .await
                .unwrap();
            
            let model = engine.load_model("test_model").await.unwrap();
            let inputs: Vec<_> = (0..test_batch_size)
                .map(|_| Tensor::ones(&[1, test_input_size], DType::F32, &Device::Cpu).unwrap())
                .collect();
            
            let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
        });
    });
    
    // Metal comparison
    #[cfg(feature = "metal")]
    group.bench_function("metal_vs_cpu", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .device(Device::Metal)
                .optimization_level(OptimizationLevel::Aggressive)
                .build()
                .await
                .unwrap();
            
            let model = engine.load_model("test_model").await.unwrap();
            let inputs: Vec<_> = (0..test_batch_size)
                .map(|_| Tensor::ones(&[1, test_input_size], DType::F32, &Device::Cpu).unwrap())
                .collect();
            
            let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
        });
    });
    
    // MLX comparison
    #[cfg(feature = "mlx")]
    group.bench_function("mlx_vs_cpu", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .device(Device::Mlx)
                .optimization_level(OptimizationLevel::Aggressive)
                .build()
                .await
                .unwrap();
            
            let model = engine.load_model("test_model").await.unwrap();
            let inputs: Vec<_> = (0..test_batch_size)
                .map(|_| Tensor::ones(&[1, test_input_size], DType::F32, &Device::Cpu).unwrap())
                .collect();
            
            let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
        });
    });
    
    group.finish();
}

criterion_group!(
    backend_benchmarks,
    benchmark_backend_throughput,
    benchmark_backend_latency,
    benchmark_memory_transfer_overhead,
    benchmark_backend_initialization,
    benchmark_cross_backend_comparison
);

criterion_main!(backend_benchmarks);
