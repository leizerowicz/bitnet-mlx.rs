//! Performance benchmarks for ternary weight packing strategies
//!
//! This benchmark suite measures the performance characteristics of different
//! packing strategies including compression ratio, packing/unpacking speed,
//! and memory efficiency.

use bitnet_quant::quantization::packing::{
    packing_utils, HybridPacker, TernaryPacker, TernaryPackerFactory, TernaryPackingConfig,
    TernaryPackingStrategy,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// Test data generators
fn generate_dense_weights(size: usize) -> Vec<i8> {
    (0..size)
        .map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => unreachable!(),
        })
        .collect()
}

fn generate_sparse_weights(size: usize, sparsity: f32) -> Vec<i8> {
    let mut weights = vec![0i8; size];
    let non_zero_count = ((1.0 - sparsity) * size as f32) as usize;

    for i in 0..non_zero_count {
        let idx = (i * size / non_zero_count.max(1)).min(size - 1);
        weights[idx] = if i % 2 == 0 { 1 } else { -1 };
    }

    weights
}

fn generate_rle_friendly_weights(size: usize) -> Vec<i8> {
    let mut weights = Vec::new();
    let mut current_val = 0i8;
    let mut remaining = size;

    while remaining > 0 {
        let run_length = (remaining / 10).max(1).min(remaining);
        for _ in 0..run_length {
            weights.push(current_val);
        }
        remaining -= run_length;
        current_val = match current_val {
            0 => 1,
            1 => -1,
            -1 => 0,
            _ => 0,
        };
    }

    weights
}

// Benchmark packing performance
fn bench_packing_strategies(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 16384];
    let strategies = [
        ("Uncompressed", TernaryPackingStrategy::Uncompressed),
        ("BitPacked2Bit", TernaryPackingStrategy::BitPacked2Bit),
        ("Base3Packed", TernaryPackingStrategy::Base3Packed),
        ("ByteAligned", TernaryPackingStrategy::ByteAligned),
        ("RunLengthEncoded", TernaryPackingStrategy::RunLengthEncoded),
        ("CompressedSparse", TernaryPackingStrategy::CompressedSparse),
        ("Hybrid", TernaryPackingStrategy::Hybrid),
    ];

    let mut group = c.benchmark_group("packing_strategies");

    for &size in &sizes {
        let weights = generate_dense_weights(size);
        let config = TernaryPackingConfig::default();

        group.throughput(Throughput::Elements(size as u64));

        for (name, strategy) in &strategies {
            let packer = TernaryPackerFactory::create_packer(*strategy);

            if packer.is_suitable(&weights, &config) {
                group.bench_with_input(
                    BenchmarkId::new(format!("pack_{name}"), size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = packer.pack(black_box(&weights), black_box(&config));
                            black_box(result)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

// Benchmark unpacking performance
fn bench_unpacking_strategies(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 16384];
    let strategies = [
        ("Uncompressed", TernaryPackingStrategy::Uncompressed),
        ("BitPacked2Bit", TernaryPackingStrategy::BitPacked2Bit),
        ("Base3Packed", TernaryPackingStrategy::Base3Packed),
        ("ByteAligned", TernaryPackingStrategy::ByteAligned),
        ("RunLengthEncoded", TernaryPackingStrategy::RunLengthEncoded),
        ("CompressedSparse", TernaryPackingStrategy::CompressedSparse),
        ("Hybrid", TernaryPackingStrategy::Hybrid),
    ];

    let mut group = c.benchmark_group("unpacking_strategies");

    for &size in &sizes {
        let weights = generate_dense_weights(size);
        let config = TernaryPackingConfig::default();

        group.throughput(Throughput::Elements(size as u64));

        for (name, strategy) in &strategies {
            let packer = TernaryPackerFactory::create_packer(*strategy);

            if packer.is_suitable(&weights, &config) {
                if let Ok(packed) = packer.pack(&weights, &config) {
                    group.bench_with_input(
                        BenchmarkId::new(format!("unpack_{name}"), size),
                        &size,
                        |b, _| {
                            b.iter(|| {
                                let result = packer.unpack(black_box(&packed));
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

// Benchmark different sparsity levels
fn bench_sparsity_impact(c: &mut Criterion) {
    let size = 4096;
    let sparsity_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95];
    let strategies = [
        ("BitPacked2Bit", TernaryPackingStrategy::BitPacked2Bit),
        ("RunLengthEncoded", TernaryPackingStrategy::RunLengthEncoded),
        ("CompressedSparse", TernaryPackingStrategy::CompressedSparse),
    ];

    let mut group = c.benchmark_group("sparsity_impact");
    group.throughput(Throughput::Elements(size as u64));

    for &sparsity in &sparsity_levels {
        let weights = generate_sparse_weights(size, sparsity);
        let config = TernaryPackingConfig::default();

        for (name, strategy) in &strategies {
            let packer = TernaryPackerFactory::create_packer(*strategy);

            if packer.is_suitable(&weights, &config) {
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_{:.0}%_sparse", name, sparsity * 100.0), size),
                    &sparsity,
                    |b, _| {
                        b.iter(|| {
                            let result = packer.pack(black_box(&weights), black_box(&config));
                            black_box(result)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

// Benchmark compression ratios
fn bench_compression_ratios(c: &mut Criterion) {
    let size = 4096;
    let test_patterns = [
        ("Dense", generate_dense_weights(size)),
        ("Sparse_50%", generate_sparse_weights(size, 0.5)),
        ("Sparse_90%", generate_sparse_weights(size, 0.9)),
        ("RLE_Friendly", generate_rle_friendly_weights(size)),
    ];

    let strategies = [
        ("Uncompressed", TernaryPackingStrategy::Uncompressed),
        ("BitPacked2Bit", TernaryPackingStrategy::BitPacked2Bit),
        ("Base3Packed", TernaryPackingStrategy::Base3Packed),
        ("RunLengthEncoded", TernaryPackingStrategy::RunLengthEncoded),
        ("CompressedSparse", TernaryPackingStrategy::CompressedSparse),
        ("Hybrid", TernaryPackingStrategy::Hybrid),
    ];

    let mut group = c.benchmark_group("compression_analysis");
    group.measurement_time(Duration::from_secs(10));

    for (pattern_name, weights) in &test_patterns {
        let config = TernaryPackingConfig::default();

        for (strategy_name, strategy) in &strategies {
            let packer = TernaryPackerFactory::create_packer(*strategy);

            if packer.is_suitable(weights, &config) {
                group.bench_function(format!("{pattern_name}_{strategy_name}"), |b| {
                    b.iter(|| {
                        let packed = packer.pack(black_box(weights), black_box(&config)).unwrap();
                        // Return compression ratio for analysis
                        black_box(packed.compression_ratio)
                    })
                });
            }
        }
    }

    group.finish();
}

// Benchmark auto-selection performance
fn bench_auto_selection(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096];

    let mut group = c.benchmark_group("auto_selection");

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Dense pattern
        {
            let weights = generate_dense_weights(size);
            let config = TernaryPackingConfig::default();

            // Benchmark strategy recommendation
            group.bench_with_input(BenchmarkId::new("recommend_Dense", size), &size, |b, _| {
                b.iter(|| {
                    let strategy = packing_utils::recommend_strategy(black_box(&weights));
                    black_box(strategy)
                })
            });

            // Benchmark auto-selection
            group.bench_with_input(
                BenchmarkId::new("auto_select_Dense", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let strategy = TernaryPackerFactory::auto_select_strategy(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(strategy)
                    })
                },
            );

            // Benchmark optimal packing (selection + packing)
            group.bench_with_input(
                BenchmarkId::new("pack_optimal_Dense", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = TernaryPackerFactory::pack_optimal(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(result)
                    })
                },
            );
        }

        // Sparse 75% pattern
        {
            let weights = generate_sparse_weights(size, 0.75);
            let config = TernaryPackingConfig::default();

            // Benchmark strategy recommendation
            group.bench_with_input(
                BenchmarkId::new("recommend_Sparse_75%", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let strategy = packing_utils::recommend_strategy(black_box(&weights));
                        black_box(strategy)
                    })
                },
            );

            // Benchmark auto-selection
            group.bench_with_input(
                BenchmarkId::new("auto_select_Sparse_75%", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let strategy = TernaryPackerFactory::auto_select_strategy(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(strategy)
                    })
                },
            );

            // Benchmark optimal packing (selection + packing)
            group.bench_with_input(
                BenchmarkId::new("pack_optimal_Sparse_75%", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = TernaryPackerFactory::pack_optimal(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(result)
                    })
                },
            );
        }

        // RLE Friendly pattern
        {
            let weights = generate_rle_friendly_weights(size);
            let config = TernaryPackingConfig::default();

            // Benchmark strategy recommendation
            group.bench_with_input(
                BenchmarkId::new("recommend_RLE_Friendly", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let strategy = packing_utils::recommend_strategy(black_box(&weights));
                        black_box(strategy)
                    })
                },
            );

            // Benchmark auto-selection
            group.bench_with_input(
                BenchmarkId::new("auto_select_RLE_Friendly", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let strategy = TernaryPackerFactory::auto_select_strategy(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(strategy)
                    })
                },
            );

            // Benchmark optimal packing (selection + packing)
            group.bench_with_input(
                BenchmarkId::new("pack_optimal_RLE_Friendly", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = TernaryPackerFactory::pack_optimal(
                            black_box(&weights),
                            black_box(&config),
                        );
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark memory access patterns
fn bench_memory_access(c: &mut Criterion) {
    let size = 4096;
    let weights = generate_dense_weights(size);
    let config = TernaryPackingConfig::default();

    let strategies = [
        ("BitPacked2Bit", TernaryPackingStrategy::BitPacked2Bit),
        ("ByteAligned", TernaryPackingStrategy::ByteAligned),
        ("Base3Packed", TernaryPackingStrategy::Base3Packed),
    ];

    let mut group = c.benchmark_group("memory_access");
    group.throughput(Throughput::Elements(size as u64));

    for (name, strategy) in &strategies {
        let packer = TernaryPackerFactory::create_packer(*strategy);
        let packed = packer.pack(&weights, &config).unwrap();

        // Benchmark sequential access (full unpack)
        group.bench_function(format!("sequential_access_{name}"), |b| {
            b.iter(|| {
                let result = packer.unpack(black_box(&packed));
                black_box(result)
            })
        });

        // Benchmark memory footprint efficiency
        group.bench_function(format!("memory_footprint_{name}"), |b| {
            b.iter(|| {
                let footprint = packed.memory_footprint;
                let ratio = packed.compression_ratio;
                black_box((footprint, ratio))
            })
        });
    }

    group.finish();
}

// Benchmark hybrid strategy performance
fn bench_hybrid_strategy(c: &mut Criterion) {
    let sizes = [256, 1024, 4096];
    let block_sizes = [16, 32, 64, 128];

    let mut group = c.benchmark_group("hybrid_strategy");

    for &size in &sizes {
        let weights = generate_dense_weights(size);

        group.throughput(Throughput::Elements(size as u64));

        for &block_size in &block_sizes {
            let config = TernaryPackingConfig {
                strategy: TernaryPackingStrategy::Hybrid,
                block_size: Some(block_size),
                ..Default::default()
            };

            let packer = HybridPacker;

            group.bench_with_input(
                BenchmarkId::new(format!("hybrid_block_{block_size}"), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = packer.pack(black_box(&weights), black_box(&config));
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark bit manipulation operations
fn bench_bit_operations(c: &mut Criterion) {
    use bitnet_quant::quantization::utils::BitUtils;

    let sizes = [64, 256, 1024, 4096];
    let bit_widths = [1, 2, 4];

    let mut group = c.benchmark_group("bit_operations");

    for &size in &sizes {
        let values: Vec<u8> = (0..size).map(|i| (i % 4) as u8).collect();

        group.throughput(Throughput::Elements(size as u64));

        for &bits in &bit_widths {
            // Benchmark packing
            group.bench_with_input(
                BenchmarkId::new(format!("pack_{bits}bit"), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = BitUtils::pack_bits(black_box(&values), black_box(bits));
                        black_box(result)
                    })
                },
            );

            // Benchmark unpacking
            let packed = BitUtils::pack_bits(&values, bits);
            group.bench_with_input(
                BenchmarkId::new(format!("unpack_{bits}bit"), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = BitUtils::unpack_bits(
                            black_box(&packed),
                            black_box(bits),
                            black_box(size),
                        );
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_packing_strategies,
    bench_unpacking_strategies,
    bench_sparsity_impact,
    bench_compression_ratios,
    bench_auto_selection,
    bench_memory_access,
    bench_hybrid_strategy,
    bench_bit_operations
);

criterion_main!(benches);
