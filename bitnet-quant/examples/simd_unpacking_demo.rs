//! SIMD Weight Unpacking Performance Demo
//! 
//! This example demonstrates the performance benefits of SIMD-optimized
//! weight unpacking compared to scalar implementations.

use bitnet_quant::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitNet SIMD Weight Unpacking Performance Demo");
    println!("==============================================\n");

    // Detect SIMD capabilities
    let capabilities = SimdCapabilities::detect();
    println!("SIMD Capabilities:");
    println!("  SSE2: {}", capabilities.sse2);
    println!("  AVX2: {}", capabilities.avx2);
    println!("  NEON: {}", capabilities.neon);
    println!("  Has SIMD: {}\n", capabilities.has_simd());

    // Test different data sizes and patterns
    let test_sizes = vec![1000, 10000, 100000];
    let strategies = vec![
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::ByteAligned,
        TernaryPackingStrategy::RunLengthEncoded,
    ];

    for &size in &test_sizes {
        println!("Testing with {} elements:", size);
        println!("{}", "=".repeat(40));

        for &strategy in &strategies {
            test_strategy_performance(strategy, size)?;
        }
        println!();
    }

    // Detailed benchmark for BitPacked2Bit strategy
    println!("Detailed BitPacked2Bit Benchmark:");
    println!("{}", "=".repeat(40));
    detailed_bitpacked_benchmark()?;

    Ok(())
}

fn test_strategy_performance(
    strategy: TernaryPackingStrategy,
    size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate test data
    let weights = generate_test_weights(size, strategy);
    
    // Pack the weights
    let config = TernaryPackingConfig {
        strategy,
        ..Default::default()
    };
    let packer = TernaryPackerFactory::create_packer(strategy);
    let packed = packer.pack(&weights, &config)?;

    // Create SIMD and scalar unpackers
    let simd_unpacker = SimdUnpacker::new();
    let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
        sse2: false,
        avx2: false,
        neon: false,
    });

    // Warm up
    for _ in 0..10 {
        let _ = simd_unpacker.unpack(&packed)?;
        let _ = scalar_unpacker.unpack(&packed)?;
    }

    // Benchmark
    let iterations = 1000;
    
    // SIMD benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simd_unpacker.unpack(&packed)?;
    }
    let simd_time = start.elapsed();

    // Scalar benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scalar_unpacker.unpack(&packed)?;
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    let compression_ratio = packed.compression_ratio;

    println!("  {:?}:", strategy);
    println!("    Compression: {:.2}x", compression_ratio);
    println!("    SIMD time:   {:?}", simd_time);
    println!("    Scalar time: {:?}", scalar_time);
    println!("    Speedup:     {:.2}x", speedup);
    println!("    Memory:      {} -> {} bytes", 
             size, packed.memory_footprint);

    Ok(())
}

fn detailed_bitpacked_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![1000, 5000, 10000, 50000, 100000];
    
    println!("Size\t\tSIMD (μs)\tScalar (μs)\tSpeedup");
    println!("{}", "-".repeat(50));

    for &size in &sizes {
        let weights = generate_test_weights(size, TernaryPackingStrategy::BitPacked2Bit);
        let config = TernaryPackingConfig::default();
        let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
        let packed = packer.pack(&weights, &config)?;

        // Use the benchmark utility
        let benchmark = bitnet_quant::quantization::simd_unpacking::benchmark::benchmark_unpacking(
            &packed, 1000
        )?;

        let simd_us = benchmark.simd_time_ns as f64 / 1000.0 / 1000.0; // Convert to microseconds
        let scalar_us = benchmark.scalar_time_ns as f64 / 1000.0 / 1000.0;

        println!("{}\t\t{:.2}\t\t{:.2}\t\t{:.2}x",
                 size, simd_us, scalar_us, benchmark.speedup);
    }

    Ok(())
}

fn generate_test_weights(size: usize, strategy: TernaryPackingStrategy) -> Vec<i8> {
    match strategy {
        TernaryPackingStrategy::RunLengthEncoded => {
            // Generate data with runs for RLE
            let mut weights = Vec::with_capacity(size);
            let mut current_val = -1i8;
            let mut run_length = 0;
            
            for i in 0..size {
                if run_length == 0 {
                    // Start new run
                    current_val = match (i / 20) % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    };
                    run_length = 5 + (i % 15); // Variable run lengths
                }
                
                weights.push(current_val);
                run_length -= 1;
            }
            weights
        }
        TernaryPackingStrategy::CompressedSparse => {
            // Generate sparse data (mostly zeros)
            let mut weights = vec![0i8; size];
            for i in (0..size).step_by(10) {
                if i < size {
                    weights[i] = if (i / 10) % 2 == 0 { 1 } else { -1 };
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_weights() {
        let weights = generate_test_weights(100, TernaryPackingStrategy::BitPacked2Bit);
        assert_eq!(weights.len(), 100);
        
        // Check that all values are ternary
        for &w in &weights {
            assert!(w == -1 || w == 0 || w == 1);
        }
    }

    #[test]
    fn test_sparse_generation() {
        let weights = generate_test_weights(100, TernaryPackingStrategy::CompressedSparse);
        let zeros = weights.iter().filter(|&&w| w == 0).count();
        
        // Should be mostly zeros
        assert!(zeros > 80);
    }

    #[test]
    fn test_rle_generation() {
        let weights = generate_test_weights(100, TernaryPackingStrategy::RunLengthEncoded);
        
        // Should have runs of identical values
        let mut run_count = 0;
        let mut current_val = weights[0];
        
        for &w in &weights[1..] {
            if w != current_val {
                run_count += 1;
                current_val = w;
            }
        }
        
        // Should have fewer runs than total elements (indicating runs exist)
        assert!(run_count < weights.len() / 2);
    }
}