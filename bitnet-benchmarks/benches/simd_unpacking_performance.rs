//! SIMD Weight Unpacking Performance Benchmarks
//! 
//! This benchmark suite measures the performance improvements of SIMD-optimized
//! weight unpacking compared to scalar implementations across different strategies
//! and data sizes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_quant::prelude::*;

fn benchmark_simd_unpacking(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_unpacking");
    
    // Test different sizes
    let sizes = vec![1000, 10000, 100000];
    
    // Test different strategies
    let strategies = vec![
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::ByteAligned,
    ];
    
    for &size in &sizes {
        for &strategy in &strategies {
            // Generate test data
            let weights = generate_test_weights(size, strategy);
            let config = TernaryPackingConfig {
                strategy,
                ..Default::default()
            };
            let packer = TernaryPackerFactory::create_packer(strategy);
            let packed = packer.pack(&weights, &config).unwrap();
            
            // Create unpackers
            let simd_unpacker = SimdUnpacker::new();
            let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                sse2: false,
                avx2: false,
                neon: false,
            });
            
            // Benchmark SIMD implementation
            group.bench_with_input(
                BenchmarkId::new(format!("{strategy:?}_simd"), size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        let result = simd_unpacker.unpack(black_box(packed)).unwrap();
                        black_box(result);
                    });
                },
            );
            
            // Benchmark scalar implementation
            group.bench_with_input(
                BenchmarkId::new(format!("{strategy:?}_scalar"), size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        let result = scalar_unpacker.unpack(black_box(packed)).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_bit_packed_detailed(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_packed_2bit_detailed");
    
    // Focus on BitPacked2Bit with various sizes
    let sizes = vec![100, 500, 1000, 5000, 10000, 50000, 100000];
    
    for &size in &sizes {
        let weights = generate_test_weights(size, TernaryPackingStrategy::BitPacked2Bit);
        let config = TernaryPackingConfig::default();
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let packed = packer.pack(&weights, &config).unwrap();
        
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &packed,
            |b, packed| {
                b.iter(|| {
                    let result = simd_unpacker.unpack(black_box(packed)).unwrap();
                    black_box(result);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &packed,
            |b, packed| {
                b.iter(|| {
                    let result = scalar_unpacker.unpack(black_box(packed)).unwrap();
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_byte_aligned_detailed(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_aligned_detailed");
    
    let sizes = vec![1000, 10000, 100000];
    let alignments = vec![16, 32, 64];
    
    for &size in &sizes {
        for &alignment in &alignments {
            let weights = generate_test_weights(size, TernaryPackingStrategy::ByteAligned);
            let config = TernaryPackingConfig {
                strategy: TernaryPackingStrategy::ByteAligned,
                alignment,
                simd_optimized: true,
                ..Default::default()
            };
            let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::ByteAligned);
            let packed = packer.pack(&weights, &config).unwrap();
            
            let simd_unpacker = SimdUnpacker::new();
            let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
                sse2: false,
                avx2: false,
                neon: false,
            });
            
            group.bench_with_input(
                BenchmarkId::new(format!("simd_align_{alignment}"), size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        let result = simd_unpacker.unpack(black_box(packed)).unwrap();
                        black_box(result);
                    });
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("scalar_align_{alignment}"), size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        let result = scalar_unpacker.unpack(black_box(packed)).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_sparse_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_data_unpacking");
    
    let sizes = vec![1000, 10000, 100000];
    let sparsity_levels = vec![0.5, 0.7, 0.9]; // 50%, 70%, 90% zeros
    
    for &size in &sizes {
        for &sparsity in &sparsity_levels {
            let weights = generate_sparse_weights(size, sparsity);
            let config = TernaryPackingConfig {
                strategy: TernaryPackingStrategy::CompressedSparse,
                sparsity_threshold: sparsity - 0.1,
                ..Default::default()
            };
            let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::CompressedSparse);
            let packed = packer.pack(&weights, &config).unwrap();
            
            let simd_unpacker = SimdUnpacker::new();
            
            group.bench_with_input(
                BenchmarkId::new(format!("sparse_{:.0}%", sparsity * 100.0), size),
                &packed,
                |b, packed| {
                    b.iter(|| {
                        let result = simd_unpacker.unpack(black_box(packed)).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_convenience_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("convenience_function");
    
    let sizes = vec![1000, 10000, 100000];
    
    for &size in &sizes {
        let weights = generate_test_weights(size, TernaryPackingStrategy::BitPacked2Bit);
        let config = TernaryPackingConfig::default();
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let packed = packer.pack(&weights, &config).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("simd_unpack_weights", size),
            &packed,
            |b, packed| {
                b.iter(|| {
                    let result = simd_unpack_weights(black_box(packed)).unwrap();
                    black_box(result);
                });
            },
        );
        
        // Compare with original unpacker
        let original_packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        group.bench_with_input(
            BenchmarkId::new("original_unpack", size),
            &packed,
            |b, packed| {
                b.iter(|| {
                    let result = original_packer.unpack(black_box(packed)).unwrap();
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions

fn generate_test_weights(size: usize, strategy: TernaryPackingStrategy) -> Vec<i8> {
    match strategy {
        TernaryPackingStrategy::RunLengthEncoded => {
            // Generate data with runs for RLE
            let mut weights = Vec::with_capacity(size);
            let mut current_val = -1i8;
            let mut run_length = 0;
            
            for i in 0..size {
                if run_length == 0 {
                    current_val = match (i / 20) % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    };
                    run_length = 5 + (i % 15);
                }
                
                weights.push(current_val);
                run_length -= 1;
            }
            weights
        }
        _ => {
            // Generate balanced ternary data
            (0..size).map(|i| match i % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            }).collect()
        }
    }
}

fn generate_sparse_weights(size: usize, sparsity: f32) -> Vec<i8> {
    let mut weights = vec![0i8; size];
    let non_zero_count = ((1.0 - sparsity) * size as f32) as usize;
    
    for i in 0..non_zero_count {
        let idx = (i * size / non_zero_count) % size;
        weights[idx] = if i % 2 == 0 { 1 } else { -1 };
    }
    
    weights
}

criterion_group!(
    benches,
    benchmark_simd_unpacking,
    benchmark_bit_packed_detailed,
    benchmark_byte_aligned_detailed,
    benchmark_sparse_data,
    benchmark_convenience_function
);
criterion_main!(benches);