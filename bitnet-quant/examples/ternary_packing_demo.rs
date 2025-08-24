//! Demonstration of different ternary weight packing strategies
//!
//! This example shows how to use the various packing strategies available
//! in BitNet for efficiently storing ternary weights.

use bitnet_quant::quantization::packing::{
    packing_utils, TernaryPackerFactory, TernaryPackingConfig, TernaryPackingStrategy,
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 BitNet Ternary Weight Packing Strategies Demo");
    println!("================================================\n");

    let device = Device::Cpu;

    // Create sample weight tensors with different characteristics
    let dense_weights = create_dense_weights(&device)?;
    let sparse_weights = create_sparse_weights(&device)?;
    let mixed_weights = create_mixed_weights(&device)?;

    println!("📊 Testing Different Weight Patterns:");
    println!("-------------------------------------");

    // Test each weight pattern with all strategies
    test_weight_pattern("Dense Weights", &dense_weights)?;
    test_weight_pattern("Sparse Weights", &sparse_weights)?;
    test_weight_pattern("Mixed Weights", &mixed_weights)?;

    println!("\n🎯 Automatic Strategy Selection:");
    println!("--------------------------------");
    demonstrate_auto_selection()?;

    println!("\n⚡ Performance Comparison:");
    println!("-------------------------");
    performance_comparison()?;

    println!("\n🔍 Memory Usage Analysis:");
    println!("-------------------------");
    memory_analysis()?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

fn create_dense_weights(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create dense weights with roughly equal distribution of -1, 0, 1
    let data: Vec<f32> = (0..64)
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => unreachable!(),
        })
        .collect();

    Ok(Tensor::from_slice(&data, (8, 8), device)?)
}

fn create_sparse_weights(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create sparse weights (80% zeros)
    let mut data = vec![0.0f32; 64];
    for i in (0..64).step_by(5) {
        data[i] = if i % 10 == 0 { 1.0 } else { -1.0 };
    }

    Ok(Tensor::from_slice(&data, (8, 8), device)?)
}

fn create_mixed_weights(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create mixed pattern: dense regions and sparse regions
    let mut data = vec![0.0f32; 64];

    // Dense region (first 16 elements)
    for i in 0..16 {
        data[i] = match i % 3 {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => unreachable!(),
        };
    }

    // Sparse region (remaining elements, mostly zeros)
    for i in (20..64).step_by(8) {
        data[i] = 1.0;
    }

    Ok(Tensor::from_slice(&data, (8, 8), device)?)
}

fn test_weight_pattern(name: &str, weights: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{name}");
    println!("{}", "=".repeat(name.len()));

    // Convert to ternary format
    let ternary_weights = packing_utils::tensor_to_ternary(weights)?;

    // Analyze sparsity
    let analysis = packing_utils::analyze_sparsity(&ternary_weights);
    println!("📈 Sparsity Analysis:");
    println!("  - Total elements: {}", analysis.total_elements);
    println!(
        "  - Zeros: {} ({:.1}%)",
        analysis.zero_count,
        analysis.sparsity_ratio * 100.0
    );
    println!("  - Positives: {}", analysis.positive_count);
    println!("  - Negatives: {}", analysis.negative_count);
    println!("  - Balance ratio: {:.3}", analysis.balance_ratio);

    // Test all packing strategies
    let strategies = [
        TernaryPackingStrategy::Uncompressed,
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::ByteAligned,
        TernaryPackingStrategy::RunLengthEncoded,
        TernaryPackingStrategy::CompressedSparse,
    ];

    println!("\n📦 Packing Results:");
    let mut results = Vec::new();

    for strategy in strategies {
        let config = TernaryPackingConfig {
            strategy,
            ..Default::default()
        };

        let packer = TernaryPackerFactory::create_packer(strategy);

        if packer.is_suitable(&ternary_weights, &config) {
            match packer.pack(&ternary_weights, &config) {
                Ok(packed) => {
                    let estimate = packer.estimate_savings(&ternary_weights, &config);
                    println!(
                        "  {:20} | {:6} bytes | {:.2}x compression | {:.1}% savings",
                        format!("{:?}", strategy),
                        packed.memory_footprint,
                        packed.compression_ratio,
                        estimate.savings_percentage
                    );
                    results.push((
                        strategy,
                        packed.compression_ratio,
                        estimate.savings_percentage,
                    ));
                }
                Err(e) => {
                    println!("  {:20} | ERROR: {}", format!("{:?}", strategy), e);
                }
            }
        } else {
            println!(
                "  {:20} | Not suitable for this pattern",
                format!("{:?}", strategy)
            );
        }
    }

    // Find best strategy
    if let Some((best_strategy, best_ratio, best_savings)) =
        results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        println!("\n🏆 Best strategy: {best_strategy:?} ({best_ratio:.2}x compression, {best_savings:.1}% savings)");
    }

    Ok(())
}

fn demonstrate_auto_selection() -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = vec![
        ("Dense pattern", vec![-1i8, 1, -1, 1, -1, 1, -1, 1]),
        ("Sparse pattern", {
            let mut v = vec![0i8; 20];
            v[5] = 1;
            v[15] = -1;
            v
        }),
        ("Base-3 optimal", vec![-1i8, 0, 1, -1, 0]), // Length 5
        ("RLE optimal", vec![0i8, 0, 0, 1, 1, 1, -1, -1, -1]),
    ];

    for (name, weights) in test_cases {
        let config = TernaryPackingConfig::default();
        let recommended = packing_utils::recommend_strategy(&weights);
        let auto_selected = TernaryPackerFactory::auto_select_strategy(&weights, &config);

        println!("📋 {name}: Recommended={recommended:?}, Auto-selected={auto_selected:?}");

        // Test the auto-selected strategy
        let packed = TernaryPackerFactory::pack_optimal(&weights, &config)?;
        println!(
            "   Result: {:.2}x compression with {:?}",
            packed.compression_ratio, packed.strategy
        );
    }

    Ok(())
}

fn performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = [64, 256, 1024, 4096];

    for size in sizes {
        println!("\n📏 Array size: {size} elements");

        // Create test data
        let weights: Vec<i8> = (0..size)
            .map(|i| match i % 4 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => 0,
                _ => unreachable!(),
            })
            .collect();

        let config = TernaryPackingConfig::default();

        // Test key strategies
        let strategies = [
            TernaryPackingStrategy::Uncompressed,
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::Hybrid,
        ];

        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            let estimate = packer.estimate_savings(&weights, &config);

            println!(
                "  {:15} | {:6} -> {:6} bytes | {:.2}x | overhead: {:.1}%",
                format!("{:?}", strategy),
                estimate.original_size_bytes,
                estimate.packed_size_bytes,
                estimate.compression_ratio,
                estimate.access_overhead * 100.0
            );
        }
    }

    Ok(())
}

fn memory_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Analyze memory usage for different sparsity levels
    let sparsity_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95];
    let size = 1000;

    println!("Sparsity | Uncompressed | BitPacked | Base3 | RLE | Sparse | Best");
    println!("---------|--------------|-----------|-------|-----|--------|-----");

    for &sparsity in &sparsity_levels {
        let mut weights = vec![0i8; size];
        let non_zero_count = ((1.0 - sparsity) * size as f32) as usize;

        // Distribute non-zero values
        for i in 0..non_zero_count {
            let idx = (i * size / non_zero_count).min(size - 1);
            weights[idx] = if i % 2 == 0 { 1 } else { -1 };
        }

        let config = TernaryPackingConfig::default();
        let strategies = [
            TernaryPackingStrategy::Uncompressed,
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
            TernaryPackingStrategy::RunLengthEncoded,
            TernaryPackingStrategy::CompressedSparse,
        ];

        let mut results = HashMap::new();
        let mut best_ratio = 0.0f32;
        let mut best_strategy = TernaryPackingStrategy::Uncompressed;

        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            if let Ok(packed) = packer.pack(&weights, &config) {
                results.insert(strategy, packed.memory_footprint);
                if packed.compression_ratio > best_ratio {
                    best_ratio = packed.compression_ratio;
                    best_strategy = strategy;
                }
            }
        }

        print!("{:7.1}% |", sparsity * 100.0);
        for strategy in strategies {
            if let Some(&size) = results.get(&strategy) {
                print!(" {size:9} |");
            } else {
                print!(" {:9} |", "N/A");
            }
        }
        println!(" {best_strategy:?}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        let device = Device::Cpu;

        // Test weight creation
        assert!(create_dense_weights(&device).is_ok());
        assert!(create_sparse_weights(&device).is_ok());
        assert!(create_mixed_weights(&device).is_ok());

        // Test analysis functions
        assert!(demonstrate_auto_selection().is_ok());
        assert!(performance_comparison().is_ok());
        assert!(memory_analysis().is_ok());
    }

    #[test]
    fn test_weight_patterns() {
        let device = Device::Cpu;
        let weights = create_dense_weights(&device).unwrap();

        // Should not panic
        assert!(test_weight_pattern("Test", &weights).is_ok());
    }
}
