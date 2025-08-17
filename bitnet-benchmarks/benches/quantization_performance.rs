//! Quantization Performance Benchmarks
//! 
//! This benchmark suite provides comprehensive performance testing for different
//! quantization schemes used in BitNet implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use candle_core::{Tensor, Device};
use std::time::Duration;
use serde::{Serialize, Deserialize};

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{
    MlxTensor, BitNetMlxDevice, operations::BitNetMlxOps,
};

/// Quantization scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizationConfig {
    name: String,
    bits: u8,
    symmetric: bool,
    per_channel: bool,
    scale_factor: Option<f32>,
}

impl QuantizationConfig {
    fn bitnet_1_58() -> Self {
        Self {
            name: "BitNet-1.58".to_string(),
            bits: 2, // Represents {-1, 0, +1}
            symmetric: true,
            per_channel: false,
            scale_factor: Some(0.1),
        }
    }

    fn int8_symmetric() -> Self {
        Self {
            name: "INT8-Symmetric".to_string(),
            bits: 8,
            symmetric: true,
            per_channel: false,
            scale_factor: Some(127.0),
        }
    }

    fn int8_asymmetric() -> Self {
        Self {
            name: "INT8-Asymmetric".to_string(),
            bits: 8,
            symmetric: false,
            per_channel: false,
            scale_factor: Some(255.0),
        }
    }

    fn int4_symmetric() -> Self {
        Self {
            name: "INT4-Symmetric".to_string(),
            bits: 4,
            symmetric: true,
            per_channel: false,
            scale_factor: Some(7.0),
        }
    }

    fn fp16_quantization() -> Self {
        Self {
            name: "FP16".to_string(),
            bits: 16,
            symmetric: true,
            per_channel: false,
            scale_factor: None,
        }
    }
}

/// Quantization performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizationMeasurement {
    scheme: String,
    tensor_size: (usize, usize),
    quantization_time: Duration,
    dequantization_time: Duration,
    memory_reduction: f64,
    accuracy_loss: f64,
    throughput: f64,
}

/// BitNet 1.58-bit quantization benchmarks
fn bench_bitnet_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitnet_quantization");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_sizes = vec![
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];

    for &(rows, cols) in &tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Candle BitNet 1.58-bit quantization
        group.bench_with_input(
            BenchmarkId::new("candle_bitnet_1_58", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let scale = 0.1f32;
                
                bencher.iter(|| {
                    // Quantize to {-1, 0, +1}
                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                    let scaled = tensor.broadcast_div(&scale_tensor).unwrap();
                    let clamped = scaled.clamp(-1.0, 1.0).unwrap();
                    let quantized = clamped.round().unwrap();
                    
                    // Dequantize
                    let dequantized = quantized.broadcast_mul(&scale_tensor).unwrap();
                    black_box((quantized, dequantized))
                })
            },
        );

        // MLX BitNet 1.58-bit quantization
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_bitnet_1_58", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let tensor = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device).unwrap();
                    
                    bencher.iter(|| {
                        let quantized = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
                        let dequantized = BitNetMlxOps::dequantize_1_58_bit(&quantized, Some(0.1)).unwrap();
                        black_box((quantized, dequantized))
                    })
                },
            );
        }
    }

    group.finish();
}

/// INT8 quantization benchmarks
fn bench_int8_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_quantization");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_sizes = vec![
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    for &(rows, cols) in &tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // Symmetric INT8 quantization
        group.bench_with_input(
            BenchmarkId::new("int8_symmetric", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let scale = 127.0f32;
                
                bencher.iter(|| {
                    // Quantize to INT8 range [-128, 127]
                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                    let scaled = tensor.broadcast_mul(&scale_tensor).unwrap();
                    let clamped = scaled.clamp(-128.0, 127.0).unwrap();
                    let quantized = clamped.round().unwrap();
                    
                    // Dequantize
                    let dequantized = quantized.broadcast_div(&scale_tensor).unwrap();
                    black_box((quantized, dequantized))
                })
            },
        );

        // Asymmetric INT8 quantization
        group.bench_with_input(
            BenchmarkId::new("int8_asymmetric", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let scale = 255.0f32;
                let zero_point = 128.0f32;
                
                bencher.iter(|| {
                    // Quantize to UINT8 range [0, 255] with zero point
                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                    let zero_tensor = Tensor::new(zero_point, &device).unwrap();
                    
                    let scaled = tensor.broadcast_mul(&scale_tensor).unwrap();
                    let shifted = scaled.broadcast_add(&zero_tensor).unwrap();
                    let clamped = shifted.clamp(0.0, 255.0).unwrap();
                    let quantized = clamped.round().unwrap();
                    
                    // Dequantize
                    let unshifted = quantized.broadcast_sub(&zero_tensor).unwrap();
                    let dequantized = unshifted.broadcast_div(&scale_tensor).unwrap();
                    black_box((quantized, dequantized))
                })
            },
        );
    }

    group.finish();
}

/// INT4 quantization benchmarks
fn bench_int4_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("int4_quantization");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_sizes = vec![
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    for &(rows, cols) in &tensor_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // INT4 quantization
        group.bench_with_input(
            BenchmarkId::new("int4_symmetric", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let scale = 7.0f32; // 4-bit signed range: [-8, 7]
                
                bencher.iter(|| {
                    // Quantize to INT4 range [-8, 7]
                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                    let scaled = tensor.broadcast_mul(&scale_tensor).unwrap();
                    let clamped = scaled.clamp(-8.0, 7.0).unwrap();
                    let quantized = clamped.round().unwrap();
                    
                    // Dequantize
                    let dequantized = quantized.broadcast_div(&scale_tensor).unwrap();
                    black_box((quantized, dequantized))
                })
            },
        );
    }

    group.finish();
}

/// Per-channel vs per-tensor quantization comparison
fn bench_quantization_granularity(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_granularity");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_size = (1024, 1024);
    let elements = tensor_size.0 * tensor_size.1;
    group.throughput(Throughput::Elements(elements as u64));

    // Per-tensor quantization
    group.bench_function("per_tensor_quantization", |bencher| {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
        
        bencher.iter(|| {
            // Calculate global scale
            let abs_tensor = tensor.abs().unwrap();
            let max_val = abs_tensor.max(1).unwrap().max(0).unwrap();
            let scale = max_val.broadcast_div(&Tensor::new(127.0f32, &device).unwrap()).unwrap();
            
            // Quantize
            let scaled = tensor.broadcast_div(&scale).unwrap();
            let quantized = scaled.clamp(-127.0, 127.0).unwrap().round().unwrap();
            
            // Dequantize
            let dequantized = quantized.broadcast_mul(&scale).unwrap();
            black_box((quantized, dequantized))
        })
    });

    // Per-channel quantization (simplified)
    group.bench_function("per_channel_quantization", |bencher| {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
        
        bencher.iter(|| {
            // Calculate per-channel scales (per row)
            let abs_tensor = tensor.abs().unwrap();
            let max_vals = abs_tensor.max(1).unwrap(); // Max per row
            let scales = max_vals.broadcast_div(&Tensor::new(127.0f32, &device).unwrap()).unwrap();
            
            // Expand scales for broadcasting
            let expanded_scales = scales.unsqueeze(1).unwrap()
                .broadcast_as(tensor.shape()).unwrap();
            
            // Quantize
            let scaled = tensor.broadcast_div(&expanded_scales).unwrap();
            let quantized = scaled.clamp(-127.0, 127.0).unwrap().round().unwrap();
            
            // Dequantize
            let dequantized = quantized.broadcast_mul(&expanded_scales).unwrap();
            black_box((quantized, dequantized))
        })
    });

    group.finish();
}

/// Dynamic vs static quantization comparison
fn bench_dynamic_vs_static_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_vs_static_quantization");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_size = (1024, 1024);
    let elements = tensor_size.0 * tensor_size.1;
    group.throughput(Throughput::Elements(elements as u64));

    // Static quantization (pre-computed scale)
    group.bench_function("static_quantization", |bencher| {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
        let static_scale = Tensor::new(0.1f32, &device).unwrap(); // Pre-computed scale
        
        bencher.iter(|| {
            // Use pre-computed scale
            let scaled = tensor.broadcast_div(&static_scale).unwrap();
            let quantized = scaled.clamp(-127.0, 127.0).unwrap().round().unwrap();
            let dequantized = quantized.broadcast_mul(&static_scale).unwrap();
            black_box((quantized, dequantized))
        })
    });

    // Dynamic quantization (compute scale on-the-fly)
    group.bench_function("dynamic_quantization", |bencher| {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
        
        bencher.iter(|| {
            // Compute scale dynamically
            let abs_tensor = tensor.abs().unwrap();
            let max_val = abs_tensor.max(1).unwrap().max(0).unwrap();
            let scale = max_val.broadcast_div(&Tensor::new(127.0f32, &device).unwrap()).unwrap();
            
            // Quantize
            let scaled = tensor.broadcast_div(&scale).unwrap();
            let quantized = scaled.clamp(-127.0, 127.0).unwrap().round().unwrap();
            let dequantized = quantized.broadcast_mul(&scale).unwrap();
            black_box((quantized, dequantized))
        })
    });

    group.finish();
}

/// Quantized matrix multiplication performance
fn bench_quantized_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_matmul");
    
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let matrix_sizes = vec![
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    for &(rows, cols) in &matrix_sizes {
        let elements = rows * cols * cols; // For matrix multiplication
        group.throughput(Throughput::Elements(elements as u64));

        // Full precision matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("fp32_matmul", format!("{}x{}", rows, cols)),
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

        // BitNet 1.58-bit quantized matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("bitnet_1_58_matmul", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                
                bencher.iter(|| {
                    // Quantize matrices
                    let a_quantized = a.clamp(-1.0, 1.0).unwrap().round().unwrap();
                    let b_quantized = b.clamp(-1.0, 1.0).unwrap().round().unwrap();
                    
                    // Perform quantized matrix multiplication
                    let result = a_quantized.matmul(&b_quantized).unwrap();
                    black_box(result)
                })
            },
        );

        // INT8 quantized matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("int8_matmul", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                
                bencher.iter(|| {
                    // Quantize to INT8
                    let scale = 127.0f32;
                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                    
                    let a_scaled = a.broadcast_mul(&scale_tensor).unwrap();
                    let a_quantized = a_scaled.clamp(-128.0, 127.0).unwrap().round().unwrap();
                    
                    let b_scaled = b.broadcast_mul(&scale_tensor).unwrap();
                    let b_quantized = b_scaled.clamp(-128.0, 127.0).unwrap().round().unwrap();
                    
                    // Perform quantized matrix multiplication
                    let result_quantized = a_quantized.matmul(&b_quantized).unwrap();
                    
                    // Dequantize result
                    let scale_squared = scale * scale;
                    let dequant_scale = Tensor::new(1.0 / scale_squared, &device).unwrap();
                    let result = result_quantized.broadcast_mul(&dequant_scale).unwrap();
                    
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Quantization accuracy vs performance trade-offs
fn bench_accuracy_performance_tradeoffs(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_performance_tradeoffs");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let tensor_size = (1024, 1024);
    let elements = tensor_size.0 * tensor_size.1;
    group.throughput(Throughput::Elements(elements as u64));

    let quantization_schemes = vec![
        ("fp32_baseline", None),
        ("fp16", Some(16)),
        ("int8", Some(8)),
        ("int4", Some(4)),
        ("bitnet_1_58", Some(2)),
    ];

    for (scheme_name, bits) in quantization_schemes {
        group.bench_function(scheme_name, |bencher| {
            let device = Device::Cpu;
            let tensor = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
            
            bencher.iter(|| {
                let result = match bits {
                    None => {
                        // FP32 baseline - no quantization
                        tensor.clone()
                    },
                    Some(16) => {
                        // FP16 simulation (using FP32 operations)
                        tensor.clone()
                    },
                    Some(8) => {
                        // INT8 quantization
                        let scale = 127.0f32;
                        let scale_tensor = Tensor::new(scale, &device).unwrap();
                        let scaled = tensor.broadcast_mul(&scale_tensor).unwrap();
                        let quantized = scaled.clamp(-128.0, 127.0).unwrap().round().unwrap();
                        quantized.broadcast_div(&scale_tensor).unwrap()
                    },
                    Some(4) => {
                        // INT4 quantization
                        let scale = 7.0f32;
                        let scale_tensor = Tensor::new(scale, &device).unwrap();
                        let scaled = tensor.broadcast_mul(&scale_tensor).unwrap();
                        let quantized = scaled.clamp(-8.0, 7.0).unwrap().round().unwrap();
                        quantized.broadcast_div(&scale_tensor).unwrap()
                    },
                    Some(2) => {
                        // BitNet 1.58-bit quantization
                        let scale = 0.1f32;
                        let scale_tensor = Tensor::new(scale, &device).unwrap();
                        let scaled = tensor.broadcast_div(&scale_tensor).unwrap();
                        let quantized = scaled.clamp(-1.0, 1.0).unwrap().round().unwrap();
                        quantized.broadcast_mul(&scale_tensor).unwrap()
                    },
                    _ => tensor.clone(),
                };
                
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    quantization_benches,
    bench_bitnet_quantization,
    bench_int8_quantization,
    bench_int4_quantization,
    bench_quantization_granularity,
    bench_dynamic_vs_static_quantization,
    bench_quantized_matmul,
    bench_accuracy_performance_tradeoffs
);

criterion_main!(quantization_benches);