//! SIMD performance benchmarks for BitNet quantization operations

use bitnet_quant::simd::{
    detect_simd_capabilities, vectorized_matrix_multiply, vectorized_pack_ternary,
    vectorized_ternary_dequantize, vectorized_ternary_quantize, vectorized_unpack_ternary,
    SimdTernaryOps,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_ternary_quantization(c: &mut Criterion) {
    let caps = detect_simd_capabilities();
    println!("SIMD capabilities: {}", caps.description());

    let sizes = [64, 256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("ternary_quantization");

    for size in sizes.iter() {
        let input: Vec<f32> = (0..*size)
            .map(|i| (i as f32 - *size as f32 / 2.0) / 100.0)
            .collect();
        let mut output = vec![0i8; *size];
        let threshold = 0.5;

        // SIMD benchmark
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                vectorized_ternary_quantize(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(threshold),
                )
                .unwrap();
            });
        });

        // Scalar benchmark for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                for (i, &val) in input.iter().enumerate() {
                    output[i] = if val > threshold {
                        1
                    } else if val < -threshold {
                        -1
                    } else {
                        0
                    };
                }
            });
        });
    }

    group.finish();
}

fn benchmark_ternary_dequantization(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("ternary_dequantization");

    for size in sizes.iter() {
        let input: Vec<i8> = (0..*size).map(|i| ((i % 3) as i8) - 1).collect();
        let mut output = vec![0.0f32; *size];
        let scale = 2.5;

        // SIMD benchmark
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                vectorized_ternary_dequantize(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(scale),
                )
                .unwrap();
            });
        });

        // Scalar benchmark
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                for (i, &val) in input.iter().enumerate() {
                    output[i] = val as f32 * scale;
                }
            });
        });
    }

    group.finish();
}

fn benchmark_ternary_matrix_multiply(c: &mut Criterion) {
    let configs = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)];

    let mut group = c.benchmark_group("ternary_matrix_multiply");

    for &(m, k, n) in configs.iter() {
        let a_ternary: Vec<i8> = (0..(m * k)).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        let scale = 1.0;

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    vectorized_matrix_multiply(
                        black_box(&a_ternary),
                        black_box(&b),
                        black_box(&mut c),
                        black_box(m),
                        black_box(k),
                        black_box(n),
                        black_box(scale),
                    )
                    .unwrap();
                });
            },
        );

        // Scalar benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    c.fill(0.0);
                    for i in 0..m {
                        for kk in 0..k {
                            let a_val = a_ternary[i * k + kk];
                            if a_val != 0 {
                                let scaled_a = a_val as f32 * scale;
                                for j in 0..n {
                                    c[i * n + j] += scaled_a * b[kk * n + j];
                                }
                            }
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ternary_packing(c: &mut Criterion) {
    let sizes = [256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("ternary_packing");

    for size in sizes.iter() {
        let input: Vec<i8> = (0..*size)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();
        let mut packed = vec![0u8; (*size + 3) / 4];
        let mut unpacked = vec![0i8; *size];

        // Packing benchmark
        group.bench_with_input(BenchmarkId::new("pack_simd", size), size, |b, _| {
            b.iter(|| {
                vectorized_pack_ternary(black_box(&input), black_box(&mut packed)).unwrap();
            });
        });

        // Unpacking benchmark
        group.bench_with_input(BenchmarkId::new("unpack_simd", size), size, |b, _| {
            b.iter(|| {
                vectorized_unpack_ternary(
                    black_box(&packed),
                    black_box(&mut unpacked),
                    black_box(*size),
                )
                .unwrap();
            });
        });

        // Scalar packing benchmark
        group.bench_with_input(BenchmarkId::new("pack_scalar", size), size, |b, _| {
            b.iter(|| {
                for i in (0..*size).step_by(4) {
                    let byte_idx = i / 4;
                    if byte_idx < packed.len() {
                        let mut byte_val = 0u8;
                        for j in 0..4 {
                            if i + j < *size {
                                let val = match input[i + j] {
                                    -1 => 0u8,
                                    0 => 1u8,
                                    1 => 2u8,
                                    _ => 0u8,
                                };
                                byte_val |= (val & 0x3) << (j * 2);
                            }
                        }
                        packed[byte_idx] = byte_val;
                    }
                }
            });
        });
    }

    group.finish();
}

fn benchmark_absmean_computation(c: &mut Criterion) {
    let sizes = [256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("absmean_computation");

    for size in sizes.iter() {
        let values: Vec<f32> = (0..*size)
            .map(|i| ((i as f32) - (*size as f32 / 2.0)) / 1000.0)
            .collect();
        let ops = SimdTernaryOps::new();

        // SIMD benchmark
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(ops.compute_absmean(black_box(&values)));
            });
        });

        // Scalar benchmark
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                let sum: f32 = values.iter().map(|x| x.abs()).sum();
                black_box(sum / values.len() as f32);
            });
        });
    }

    group.finish();
}

fn benchmark_memory_patterns(c: &mut Criterion) {
    let size = 4096;
    let mut group = c.benchmark_group("memory_patterns");

    // Sequential access pattern
    let input: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let mut output = vec![0i8; size];

    group.bench_function("sequential_access", |b| {
        b.iter(|| {
            vectorized_ternary_quantize(black_box(&input), black_box(&mut output), black_box(1.0))
                .unwrap();
        });
    });

    // Strided access pattern (every 4th element)
    let strided_input: Vec<f32> = (0..size).step_by(4).map(|i| i as f32).collect();
    let mut strided_output = vec![0i8; strided_input.len()];

    group.bench_function("strided_access", |b| {
        b.iter(|| {
            vectorized_ternary_quantize(
                black_box(&strided_input),
                black_box(&mut strided_output),
                black_box(1.0),
            )
            .unwrap();
        });
    });

    group.finish();
}

fn benchmark_large_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_matrices");
    group.sample_size(10); // Reduce sample size for large matrices

    let configs = [(512, 512, 512), (1024, 1024, 256), (2048, 512, 512)];

    for &(m, k, n) in configs.iter() {
        let a_ternary: Vec<i8> = (0..(m * k)).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0f32; m * n];

        group.bench_with_input(
            BenchmarkId::new("simd_large", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    vectorized_matrix_multiply(
                        black_box(&a_ternary),
                        black_box(&b),
                        black_box(&mut c),
                        black_box(m),
                        black_box(k),
                        black_box(n),
                        black_box(1.0),
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    simd_benches,
    benchmark_ternary_quantization,
    benchmark_ternary_dequantization,
    benchmark_ternary_matrix_multiply,
    benchmark_ternary_packing,
    benchmark_absmean_computation,
    benchmark_memory_patterns,
    benchmark_large_matrices
);

criterion_main!(simd_benches);
