//! MLX vs Candle Performance Benchmarks
//! 
//! This benchmark suite compares the performance of equivalent operations
//! between MLX (Apple Silicon optimized) and Candle (cross-platform) backends.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use candle_core::{Tensor, Device, DType};
use std::time::Duration;

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{
    MlxTensor, BitNetMlxDevice, operations::BitNetMlxOps,
    mlx_matmul, mlx_quantize, mlx_dequantize
};

/// Configuration for benchmark parameters
struct BenchmarkConfig {
    tensor_sizes: Vec<(usize, usize)>,
    warmup_time: Duration,
    measurement_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            tensor_sizes: vec![
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
            ],
            warmup_time: Duration::from_secs(3),
            measurement_time: Duration::from_secs(10),
        }
    }
}

/// Benchmark matrix multiplication: MLX vs Candle
fn bench_matmul_comparison(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("matmul_comparison");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("candle_cpu", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                
                bencher.iter(|| {
                    let result = a.matmul(&b).unwrap();
                    black_box(result)
                })
            },
        );

        // Candle Metal benchmark (if available)
        #[cfg(target_os = "macos")]
        if Device::new_metal(0).is_ok() {
            group.bench_with_input(
                BenchmarkId::new("candle_metal", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    let device = Device::new_metal(0).unwrap();
                    let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                    let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                    
                    bencher.iter(|| {
                        let result = a.matmul(&b).unwrap();
                        black_box(result)
                    })
                },
            );
        }

        // MLX benchmark (if available)
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let a = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                    let b = MlxTensor::randn(&[cols, rows], BitNetDType::F32, device).unwrap();
                    
                    bencher.iter(|| {
                        let result = BitNetMlxOps::matmul(&a, &b).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark quantization operations: MLX vs Candle
fn bench_quantization_comparison(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("quantization_comparison");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle quantization benchmark (simplified)
        group.bench_with_input(
            BenchmarkId::new("candle_quantize", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let scale = 0.1f32;
                
                b.iter(|| {
                    // Simple quantization: divide by scale and round
                    let scaled = tensor.broadcast_div(&Tensor::new(scale, &device).unwrap()).unwrap();
                    let quantized = scaled.round().unwrap();
                    black_box(quantized)
                })
            },
        );

        // MLX quantization benchmark
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_quantize", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let tensor = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device).unwrap();
                    
                    b.iter(|| {
                        let result = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark element-wise operations: MLX vs Candle
fn bench_elementwise_comparison(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("elementwise_comparison");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle addition benchmark
        group.bench_with_input(
            BenchmarkId::new("candle_add", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                
                bencher.iter(|| {
                    let result = (&a + &b).unwrap();
                    black_box(result)
                })
            },
        );

        // Candle multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("candle_mul", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                
                bencher.iter(|| {
                    let result = (&a * &b).unwrap();
                    black_box(result)
                })
            },
        );

        // MLX addition benchmark
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_add", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let a = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                    let b = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device).unwrap();
                    
                    b.iter(|| {
                        let result = BitNetMlxOps::add(&a, &b).unwrap();
                        black_box(result)
                    })
                },
            );
        }

        // MLX multiplication benchmark
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_mul", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let a = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                    let b = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device).unwrap();
                    
                    b.iter(|| {
                        let result = BitNetMlxOps::multiply(&a, &b).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark tensor creation and memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("memory_operations");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle tensor creation
        group.bench_with_input(
            BenchmarkId::new("candle_zeros", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let device = Device::Cpu;
                
                b.iter(|| {
                    let result = Tensor::zeros((rows, cols), DType::F32, &device).unwrap();
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("candle_ones", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let device = Device::Cpu;
                
                b.iter(|| {
                    let result = Tensor::ones((rows, cols), DType::F32, &device).unwrap();
                    black_box(result)
                })
            },
        );

        // MLX tensor creation
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_zeros", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    
                    b.iter(|| {
                        let result = MlxTensor::zeros(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                        black_box(result)
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("mlx_ones", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    
                    b.iter(|| {
                        let result = MlxTensor::ones(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark BitLinear operations (BitNet-specific)
fn bench_bitlinear_comparison(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("bitlinear_comparison");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(input_size, output_size) in &config.tensor_sizes {
        let batch_size = 32;
        let elements = batch_size * input_size * output_size;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle BitLinear simulation
        group.bench_with_input(
            BenchmarkId::new("candle_bitlinear", format!("{}x{}", input_size, output_size)),
            &(input_size, output_size),
            |b, &(input_size, output_size)| {
                let device = Device::Cpu;
                let input = Tensor::randn(0f32, 1f32, (batch_size, input_size), &device).unwrap();
                let weight = Tensor::randn(0f32, 1f32, (input_size, output_size), &device).unwrap();
                let bias = Tensor::randn(0f32, 1f32, (output_size,), &device).unwrap();
                
                b.iter(|| {
                    // Simulate quantization: clamp weights to -1, 0, 1
                    let quantized_weight = weight.clamp(-1.0, 1.0).unwrap().round().unwrap();
                    let output = input.matmul(&quantized_weight).unwrap();
                    let result = output.broadcast_add(&bias).unwrap();
                    black_box(result)
                })
            },
        );

        // MLX BitLinear
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_bitlinear", format!("{}x{}", input_size, output_size)),
                &(input_size, output_size),
                |b, &(input_size, output_size)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let input = MlxTensor::randn(&[batch_size, input_size], BitNetDType::F32, device.clone()).unwrap();
                    let weight = MlxTensor::randn(&[input_size, output_size], BitNetDType::F32, device.clone()).unwrap();
                    let bias = MlxTensor::randn(&[output_size], BitNetDType::F32, device).unwrap();
                    
                    b.iter(|| {
                        let result = BitNetMlxOps::bitlinear_forward(&input, &weight, Some(&bias), true).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark data type conversions between MLX and Candle
#[cfg(feature = "mlx")]
fn bench_conversion_operations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("conversion_operations");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &(rows, cols) in &config.tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle to MLX conversion
        group.bench_with_input(
            BenchmarkId::new("candle_to_mlx", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                
                b.iter(|| {
                    let result = bitnet_core::mlx::candle_to_mlx_array(&tensor).unwrap();
                    black_box(result)
                })
            },
        );

        // MLX to Candle conversion
        group.bench_with_input(
            BenchmarkId::new("mlx_to_candle", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                use bitnet_core::memory::tensor::BitNetDType;
                let device = BitNetMlxDevice::default();
                let tensor = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device).unwrap();
                
                b.iter(|| {
                    let result = bitnet_core::mlx::mlx_to_candle_tensor(tensor.array()).unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "mlx"))]
fn bench_conversion_operations(_c: &mut Criterion) {
    // Skip conversion benchmarks when MLX is not available
}

criterion_group!(
    benches,
    bench_matmul_comparison,
    bench_quantization_comparison,
    bench_elementwise_comparison,
    bench_memory_operations,
    bench_bitlinear_comparison,
    bench_conversion_operations
);

criterion_main!(benches);