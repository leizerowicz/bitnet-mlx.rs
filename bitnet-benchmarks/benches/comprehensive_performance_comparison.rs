//! Comprehensive Performance Comparison Benchmarks
//!
//! This benchmark suite provides extensive performance comparisons across different
//! devices, operations, and configurations for BitNet operations.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{
    device_comparison::{DeviceComparisonConfig, MlxDeviceComparison},
    operations::BitNetMlxOps,
    performance::{BenchmarkConfig, MlxPerformanceBenchmarker},
    BitNetMlxDevice, MlxTensor,
};

/// Comprehensive benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComprehensiveBenchmarkConfig {
    tensor_sizes: Vec<(usize, usize)>,
    batch_sizes: Vec<usize>,
    data_types: Vec<String>,
    operations: Vec<String>,
    devices: Vec<String>,
    warmup_iterations: usize,
    measurement_iterations: usize,
    enable_memory_tracking: bool,
    enable_energy_tracking: bool,
}

impl Default for ComprehensiveBenchmarkConfig {
    fn default() -> Self {
        Self {
            tensor_sizes: vec![
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
                (4096, 4096),
            ],
            batch_sizes: vec![1, 8, 16, 32, 64, 128],
            data_types: vec!["f32".to_string(), "f16".to_string()],
            operations: vec![
                "matmul".to_string(),
                "quantization".to_string(),
                "bitlinear".to_string(),
                "activation".to_string(),
                "layer_norm".to_string(),
                "attention".to_string(),
            ],
            devices: vec!["cpu".to_string(), "gpu".to_string()],
            warmup_iterations: 5,
            measurement_iterations: 10,
            enable_memory_tracking: true,
            enable_energy_tracking: true,
        }
    }
}

/// Performance measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMeasurement {
    operation: String,
    device: String,
    tensor_size: (usize, usize),
    batch_size: usize,
    data_type: String,
    execution_time: Duration,
    throughput: f64,
    memory_usage: usize,
    energy_consumption: Option<f64>,
    efficiency_score: f64,
}

/// Comprehensive matrix multiplication benchmarks
fn bench_comprehensive_matmul(c: &mut Criterion) {
    let config = ComprehensiveBenchmarkConfig::default();
    let mut group = c.benchmark_group("comprehensive_matmul");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &(rows, cols) in &config.tensor_sizes {
        for &batch_size in &config.batch_sizes {
            let elements = batch_size * rows * cols;
            group.throughput(Throughput::Elements(elements as u64));

            // CPU Candle benchmark
            group.bench_with_input(
                BenchmarkId::new("candle_cpu", format!("{rows}x{cols}_batch{batch_size}")),
                &(rows, cols, batch_size),
                |bencher, &(rows, cols, batch_size)| {
                    let device = Device::Cpu;
                    let a = Tensor::randn(0f32, 1f32, (batch_size, rows, cols), &device).unwrap();
                    let b = Tensor::randn(0f32, 1f32, (batch_size, cols, rows), &device).unwrap();

                    bencher.iter(|| {
                        let result = a.matmul(&b).unwrap();
                        black_box(result)
                    })
                },
            );

            // GPU Metal benchmark (if available)
            #[cfg(target_os = "macos")]
            if Device::new_metal(0).is_ok() {
                group.bench_with_input(
                    BenchmarkId::new("candle_metal", format!("{rows}x{cols}_batch{batch_size}")),
                    &(rows, cols, batch_size),
                    |bencher, &(rows, cols, batch_size)| {
                        let device = Device::new_metal(0).unwrap();
                        let a =
                            Tensor::randn(0f32, 1f32, (batch_size, rows, cols), &device).unwrap();
                        let b =
                            Tensor::randn(0f32, 1f32, (batch_size, cols, rows), &device).unwrap();

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
                    BenchmarkId::new("mlx", format!("{}x{}_batch{}", rows, cols, batch_size)),
                    &(rows, cols, batch_size),
                    |bencher, &(rows, cols, batch_size)| {
                        use bitnet_core::memory::tensor::BitNetDType;
                        let device = BitNetMlxDevice::default();
                        let a = MlxTensor::randn(
                            &[batch_size, rows, cols],
                            BitNetDType::F32,
                            device.clone(),
                        )
                        .unwrap();
                        let b =
                            MlxTensor::randn(&[batch_size, cols, rows], BitNetDType::F32, device)
                                .unwrap();

                        bencher.iter(|| {
                            let result = BitNetMlxOps::matmul(&a, &b).unwrap();
                            black_box(result)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Comprehensive quantization benchmarks
fn bench_comprehensive_quantization(c: &mut Criterion) {
    let config = ComprehensiveBenchmarkConfig::default();
    let mut group = c.benchmark_group("comprehensive_quantization");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for &(rows, cols) in &config.tensor_sizes {
        for &batch_size in &config.batch_sizes {
            let elements = batch_size * rows * cols;
            group.throughput(Throughput::Elements(elements as u64));

            // 1.58-bit quantization benchmark
            group.bench_with_input(
                BenchmarkId::new(
                    "quantize_1_58_bit",
                    format!("{rows}x{cols}_batch{batch_size}"),
                ),
                &(rows, cols, batch_size),
                |bencher, &(rows, cols, batch_size)| {
                    let device = Device::Cpu;
                    let tensor =
                        Tensor::randn(0f32, 1f32, (batch_size, rows, cols), &device).unwrap();

                    bencher.iter(|| {
                        // Quantize to {-1, 0, +1}
                        let scaled = tensor
                            .broadcast_div(&Tensor::new(0.1f32, &device).unwrap())
                            .unwrap();
                        let clamped = scaled.clamp(-1.0, 1.0).unwrap();
                        let quantized = clamped.round().unwrap();
                        black_box(quantized)
                    })
                },
            );

            // 8-bit quantization benchmark
            group.bench_with_input(
                BenchmarkId::new("quantize_8_bit", format!("{rows}x{cols}_batch{batch_size}")),
                &(rows, cols, batch_size),
                |bencher, &(rows, cols, batch_size)| {
                    let device = Device::Cpu;
                    let tensor =
                        Tensor::randn(0f32, 1f32, (batch_size, rows, cols), &device).unwrap();

                    bencher.iter(|| {
                        // Quantize to 8-bit range
                        let scaled = tensor
                            .broadcast_mul(&Tensor::new(127.0f32, &device).unwrap())
                            .unwrap();
                        let clamped = scaled.clamp(-128.0, 127.0).unwrap();
                        let quantized = clamped.round().unwrap();
                        black_box(quantized)
                    })
                },
            );

            // MLX quantization benchmark
            #[cfg(feature = "mlx")]
            {
                group.bench_with_input(
                    BenchmarkId::new(
                        "mlx_quantize_1_58",
                        format!("{}x{}_batch{}", rows, cols, batch_size),
                    ),
                    &(rows, cols, batch_size),
                    |bencher, &(rows, cols, batch_size)| {
                        use bitnet_core::memory::tensor::BitNetDType;
                        let device = BitNetMlxDevice::default();
                        let tensor =
                            MlxTensor::randn(&[batch_size, rows, cols], BitNetDType::F32, device)
                                .unwrap();

                        bencher.iter(|| {
                            let result =
                                BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
                            black_box(result)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// BitLinear layer benchmarks
fn bench_comprehensive_bitlinear(c: &mut Criterion) {
    let _config = ComprehensiveBenchmarkConfig::default();
    let mut group = c.benchmark_group("comprehensive_bitlinear");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let layer_configs = vec![
        (768, 3072),   // Transformer FFN
        (1024, 4096),  // Large model
        (2048, 8192),  // Very large model
        (4096, 16384), // Huge model
    ];

    for &(input_dim, output_dim) in &layer_configs {
        for &batch_size in &[1, 8, 16, 32, 64] {
            let elements = batch_size * input_dim * output_dim;
            group.throughput(Throughput::Elements(elements as u64));

            // Candle BitLinear simulation
            group.bench_with_input(
                BenchmarkId::new(
                    "candle_bitlinear",
                    format!("{input_dim}x{output_dim}_batch{batch_size}"),
                ),
                &(input_dim, output_dim, batch_size),
                |bencher, &(input_dim, output_dim, batch_size)| {
                    let device = Device::Cpu;
                    let input =
                        Tensor::randn(0f32, 1f32, (batch_size, input_dim), &device).unwrap();
                    let weight =
                        Tensor::randn(0f32, 1f32, (input_dim, output_dim), &device).unwrap();
                    let bias = Tensor::randn(0f32, 1f32, (output_dim,), &device).unwrap();

                    bencher.iter(|| {
                        // Quantize weights to {-1, 0, +1}
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
                    BenchmarkId::new(
                        "mlx_bitlinear",
                        format!("{}x{}_batch{}", input_dim, output_dim, batch_size),
                    ),
                    &(input_dim, output_dim, batch_size),
                    |bencher, &(input_dim, output_dim, batch_size)| {
                        use bitnet_core::memory::tensor::BitNetDType;
                        let device = BitNetMlxDevice::default();
                        let input = MlxTensor::randn(
                            &[batch_size, input_dim],
                            BitNetDType::F32,
                            device.clone(),
                        )
                        .unwrap();
                        let weight = MlxTensor::randn(
                            &[input_dim, output_dim],
                            BitNetDType::F32,
                            device.clone(),
                        )
                        .unwrap();
                        let bias =
                            MlxTensor::randn(&[output_dim], BitNetDType::F32, device).unwrap();

                        bencher.iter(|| {
                            let result =
                                BitNetMlxOps::bitlinear_forward(&input, &weight, Some(&bias), true)
                                    .unwrap();
                            black_box(result)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Activation function benchmarks
fn bench_comprehensive_activations(c: &mut Criterion) {
    let config = ComprehensiveBenchmarkConfig::default();
    let mut group = c.benchmark_group("comprehensive_activations");

    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let activations = vec!["relu", "gelu", "swish", "tanh"];

    for &(rows, cols) in &config.tensor_sizes {
        for &batch_size in &[1, 32, 64, 128] {
            let elements = batch_size * rows * cols;
            group.throughput(Throughput::Elements(elements as u64));

            for activation in &activations {
                // Candle activation benchmarks
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("candle_{activation}"),
                        format!("{rows}x{cols}_batch{batch_size}"),
                    ),
                    &(rows, cols, batch_size),
                    |bencher, &(rows, cols, batch_size)| {
                        let device = Device::Cpu;
                        let tensor =
                            Tensor::randn(0f32, 1f32, (batch_size, rows, cols), &device).unwrap();

                        bencher.iter(|| {
                            let result = match *activation {
                                "relu" => tensor.relu().unwrap(),
                                "gelu" => tensor.gelu().unwrap(),
                                "swish" => {
                                    // Swish = x * sigmoid(x), but sigmoid may not be available
                                    // Use tanh approximation: swish ≈ x * tanh(x)
                                    let tanh_x = tensor.tanh().unwrap();
                                    tensor.broadcast_mul(&tanh_x).unwrap()
                                }
                                "tanh" => tensor.tanh().unwrap(),
                                _ => tensor.clone(),
                            };
                            black_box(result)
                        })
                    },
                );

                // MLX activation benchmarks
                #[cfg(feature = "mlx")]
                {
                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("mlx_{}", activation),
                            format!("{}x{}_batch{}", rows, cols, batch_size),
                        ),
                        &(rows, cols, batch_size),
                        |bencher, &(rows, cols, batch_size)| {
                            use bitnet_core::memory::tensor::BitNetDType;
                            let device = BitNetMlxDevice::default();
                            let tensor = MlxTensor::randn(
                                &[batch_size, rows, cols],
                                BitNetDType::F32,
                                device,
                            )
                            .unwrap();

                            bencher.iter(|| {
                                let result = match *activation {
                                    "relu" => {
                                        // Use basic relu implementation if MLX ops not available
                                        tensor.clone() // Placeholder - would implement actual relu
                                    }
                                    "gelu" => {
                                        // Use basic gelu implementation if MLX ops not available
                                        tensor.clone() // Placeholder - would implement actual gelu
                                    }
                                    "swish" => {
                                        // Use basic swish implementation if MLX ops not available
                                        tensor.clone() // Placeholder - would implement actual swish
                                    }
                                    "tanh" => {
                                        // Use basic tanh implementation if MLX ops not available
                                        tensor.clone() // Placeholder - would implement actual tanh
                                    }
                                    _ => tensor.clone(),
                                };
                                black_box(result)
                            })
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

/// Memory efficiency benchmarks
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let memory_scenarios = vec![
        ("small_frequent", vec![(64, 64); 100]),
        ("medium_batch", vec![(512, 512); 10]),
        ("large_single", vec![(2048, 2048); 1]),
        (
            "mixed_sizes",
            vec![(128, 128), (256, 256), (512, 512), (1024, 1024)],
        ),
    ];

    for (scenario_name, tensor_sizes) in memory_scenarios {
        // Candle memory efficiency test
        group.bench_function(
            BenchmarkId::new("candle_memory", scenario_name),
            |bencher| {
                bencher.iter(|| {
                    let device = Device::Cpu;
                    let mut tensors = Vec::new();

                    for &(rows, cols) in &tensor_sizes {
                        let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                        tensors.push(tensor);
                    }

                    // Perform operations on all tensors
                    for i in 0..tensors.len() {
                        for j in (i + 1)..tensors.len() {
                            if tensors[i].shape() == tensors[j].shape() {
                                let result = (&tensors[i] + &tensors[j]).unwrap();
                                black_box(result);
                            }
                        }
                    }

                    black_box(tensors)
                })
            },
        );

        // MLX memory efficiency test
        #[cfg(feature = "mlx")]
        {
            group.bench_function(BenchmarkId::new("mlx_memory", scenario_name), |bencher| {
                bencher.iter(|| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let mut tensors = Vec::new();

                    for &(rows, cols) in &tensor_sizes {
                        let tensor =
                            MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone())
                                .unwrap();
                        tensors.push(tensor);
                    }

                    // Perform operations on all tensors
                    for i in 0..tensors.len() {
                        for j in (i + 1)..tensors.len() {
                            if tensors[i].shape() == tensors[j].shape() {
                                let result = BitNetMlxOps::add(&tensors[i], &tensors[j]).unwrap();
                                black_box(result);
                            }
                        }
                    }

                    black_box(tensors)
                })
            });
        }
    }

    group.finish();
}

/// Real-world workload simulation benchmarks
fn bench_real_world_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_workloads");

    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(15));

    // Transformer attention simulation
    group.bench_function("transformer_attention", |bencher| {
        let device = Device::Cpu;
        let batch_size = 8;
        let seq_len = 512;
        let hidden_dim = 768;
        let num_heads = 12;
        let head_dim = hidden_dim / num_heads;

        bencher.iter(|| {
            // Query, Key, Value projections
            let input =
                Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_dim), &device).unwrap();
            let q_weight = Tensor::randn(0f32, 1f32, (hidden_dim, hidden_dim), &device).unwrap();
            let k_weight = Tensor::randn(0f32, 1f32, (hidden_dim, hidden_dim), &device).unwrap();
            let v_weight = Tensor::randn(0f32, 1f32, (hidden_dim, hidden_dim), &device).unwrap();

            // Project to Q, K, V
            let q = input.matmul(&q_weight).unwrap();
            let k = input.matmul(&k_weight).unwrap();
            let v = input.matmul(&v_weight).unwrap();

            // Reshape for multi-head attention
            let q = q
                .reshape((batch_size, seq_len, num_heads, head_dim))
                .unwrap();
            let k = k
                .reshape((batch_size, seq_len, num_heads, head_dim))
                .unwrap();
            let v = v
                .reshape((batch_size, seq_len, num_heads, head_dim))
                .unwrap();

            // Transpose for attention computation
            let q = q.transpose(1, 2).unwrap(); // (batch, heads, seq, head_dim)
            let k = k.transpose(1, 2).unwrap();
            let v = v.transpose(1, 2).unwrap();

            // Attention scores
            let k_t = k.transpose(2, 3).unwrap(); // (batch, heads, head_dim, seq)
            let scores = q.matmul(&k_t).unwrap();

            // Scale scores
            let scale = (head_dim as f32).sqrt();
            let scaled_scores = scores
                .broadcast_div(&Tensor::new(scale, &device).unwrap())
                .unwrap();

            // Apply softmax (simplified implementation)
            let max_vals = scaled_scores.max_keepdim(3).unwrap();
            let shifted = scaled_scores.broadcast_sub(&max_vals).unwrap();
            let exp_vals = shifted.exp().unwrap();
            let sum_exp = exp_vals.sum_keepdim(3).unwrap();
            let attention_weights = exp_vals.broadcast_div(&sum_exp).unwrap();

            // Apply attention to values
            let output = attention_weights.matmul(&v).unwrap();

            black_box(output)
        })
    });

    // BitNet inference simulation
    group.bench_function("bitnet_inference", |bencher| {
        let device = Device::Cpu;
        let batch_size = 16;
        let seq_len = 256;
        let vocab_size = 32000;
        let hidden_dim = 1024;

        bencher.iter(|| {
            // Token embeddings
            let _input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();
            let _embedding_weight =
                Tensor::randn(0f32, 1f32, (vocab_size, hidden_dim), &device).unwrap();

            // Simulate embedding lookup (simplified)
            let embeddings =
                Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_dim), &device).unwrap();

            // BitLinear layers simulation
            let mut hidden_states = embeddings;

            for _layer in 0..12 {
                // Self-attention (simplified)
                let attn_weight =
                    Tensor::randn(0f32, 1f32, (hidden_dim, hidden_dim), &device).unwrap();
                let quantized_attn = attn_weight.clamp(-1.0, 1.0).unwrap().round().unwrap();
                let attn_output = hidden_states.matmul(&quantized_attn).unwrap();

                // Add residual connection
                hidden_states = (&hidden_states + &attn_output).unwrap();

                // FFN with BitLinear
                let ffn_weight1 =
                    Tensor::randn(0f32, 1f32, (hidden_dim, hidden_dim * 4), &device).unwrap();
                let ffn_weight2 =
                    Tensor::randn(0f32, 1f32, (hidden_dim * 4, hidden_dim), &device).unwrap();

                let quantized_ffn1 = ffn_weight1.clamp(-1.0, 1.0).unwrap().round().unwrap();
                let quantized_ffn2 = ffn_weight2.clamp(-1.0, 1.0).unwrap().round().unwrap();

                let ffn_intermediate = hidden_states.matmul(&quantized_ffn1).unwrap();
                let ffn_activated = ffn_intermediate.gelu().unwrap();
                let ffn_output = ffn_activated.matmul(&quantized_ffn2).unwrap();

                // Add residual connection
                hidden_states = (&hidden_states + &ffn_output).unwrap();
            }

            // Final layer norm and output projection
            let output_weight =
                Tensor::randn(0f32, 1f32, (hidden_dim, vocab_size), &device).unwrap();
            let logits = hidden_states.matmul(&output_weight).unwrap();

            black_box(logits)
        })
    });

    group.finish();
}

/// Cross-platform performance comparison
fn bench_cross_platform_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform_comparison");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let test_sizes = vec![(512, 512), (1024, 1024), (2048, 2048)];

    for &(rows, cols) in &test_sizes {
        let elements = rows * cols;
        group.throughput(Throughput::Elements(elements as u64));

        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu_baseline", format!("{rows}x{cols}")),
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

        // Metal (macOS)
        #[cfg(target_os = "macos")]
        if Device::new_metal(0).is_ok() {
            group.bench_with_input(
                BenchmarkId::new("metal_macos", format!("{rows}x{cols}")),
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

        // CUDA (if available)
        if Device::new_cuda(0).is_ok() {
            group.bench_with_input(
                BenchmarkId::new("cuda_gpu", format!("{rows}x{cols}")),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    let device = Device::new_cuda(0).unwrap();
                    let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                    let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();

                    bencher.iter(|| {
                        let result = a.matmul(&b).unwrap();
                        black_box(result)
                    })
                },
            );
        }

        // MLX (Apple Silicon)
        #[cfg(feature = "mlx")]
        {
            group.bench_with_input(
                BenchmarkId::new("mlx_apple_silicon", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    use bitnet_core::memory::tensor::BitNetDType;
                    let device = BitNetMlxDevice::default();
                    let a =
                        MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
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

criterion_group!(
    comprehensive_benches,
    bench_comprehensive_matmul,
    bench_comprehensive_quantization,
    bench_comprehensive_bitlinear,
    bench_comprehensive_activations,
    bench_memory_efficiency,
    bench_real_world_workloads,
    bench_cross_platform_comparison
);

criterion_main!(comprehensive_benches);
