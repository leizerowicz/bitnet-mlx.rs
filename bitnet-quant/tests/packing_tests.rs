//! Comprehensive tests for ternary weight packing strategies

use bitnet_quant::quantization::packing::{
    TernaryPackingStrategy, TernaryPackingConfig, TernaryPackerFactory,
    UncompressedPacker, BitPacked2BitPacker, Base3PackedPacker,
    ByteAlignedPacker, RunLengthEncodedPacker, CompressedSparsePacker,
    HybridPacker, TernaryPacker, packing_utils,
};
use bitnet_quant::quantization::utils::QuantizationError;

#[test]
fn test_uncompressed_packer_basic() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let config = TernaryPackingConfig::default();
    let packer = UncompressedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::Uncompressed);
    assert_eq!(packed.compression_ratio, 1.0);
}

#[test]
fn test_bit_packed_2bit_compression() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1]; // 8 elements = 2 bytes when packed
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::BitPacked2Bit);
    assert_eq!(packed.data.len(), 2); // 8 values / 4 values per byte = 2 bytes
    assert!(packed.compression_ratio > 1.0);
    
    let estimate = packer.estimate_savings(&weights, &config);
    assert_eq!(estimate.compression_ratio, 4.0); // 8 bytes -> 2 bytes
}

#[test]
fn test_bit_packed_2bit_with_padding() {
    let weights = vec![-1i8, 0, 1]; // 3 elements, needs padding to 4
    let config = TernaryPackingConfig::default();
    let packer = BitPacked2BitPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.metadata.padding, 1); // 1 element of padding
    assert_eq!(packed.data.len(), 1); // 4 values fit in 1 byte
}

#[test]
fn test_base3_packed_optimal() {
    let weights = vec![-1i8, 0, 1, -1, 0]; // Exactly 5 elements = 1 byte when packed
    let config = TernaryPackingConfig::default();
    let packer = Base3PackedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::Base3Packed);
    assert_eq!(packed.data.len(), 1); // 5 values in 1 byte
    assert_eq!(packed.compression_ratio, 5.0);
}

#[test]
fn test_base3_packed_with_padding() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1, 0]; // 7 elements, needs padding to 10
    let config = TernaryPackingConfig::default();
    let packer = Base3PackedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.metadata.padding, 3); // 3 elements of padding
    assert_eq!(packed.data.len(), 2); // 10 values = 2 bytes
}

#[test]
fn test_base3_encoding_decoding() {
    // Test specific base-3 encoding values
    let weights = vec![1i8, 0, -1, 1, 0]; // Should encode as: 2*1 + 1*3 + 0*9 + 2*27 + 1*81 = 2 + 3 + 0 + 54 + 81 = 140
    let config = TernaryPackingConfig::default();
    let packer = Base3PackedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    // Verify the actual encoding
    assert_eq!(packed.data[0], 140u8);
}

#[test]
fn test_byte_aligned_packer() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1];
    let config = TernaryPackingConfig {
        alignment: 8,
        simd_optimized: true,
        ..Default::default()
    };
    let packer = ByteAlignedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::ByteAligned);
    assert_eq!(packed.data.len(), 8); // Padded to 8-byte alignment
    assert_eq!(packed.metadata.padding, 2);
}

#[test]
fn test_run_length_encoded_sparse() {
    let weights = vec![0i8, 0, 0, 1, 1, -1, -1, -1, 0, 0];
    let config = TernaryPackingConfig::default();
    let packer = RunLengthEncodedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::RunLengthEncoded);
    
    // Should have 4 runs: [0,0,0], [1,1], [-1,-1,-1], [0,0]
    // Each run = 2 bytes (value + count) = 8 bytes total
    assert_eq!(packed.data.len(), 8);
    assert!(packed.compression_ratio > 1.0);
}

#[test]
fn test_run_length_encoded_dense() {
    let weights = vec![-1i8, 1, -1, 1, -1, 1]; // No runs, should be inefficient
    let config = TernaryPackingConfig::default();
    let packer = RunLengthEncodedPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    // Should have 6 runs of length 1 each = 12 bytes (worse than original)
    assert_eq!(packed.data.len(), 12);
    assert!(packed.compression_ratio < 1.0);
}

#[test]
fn test_compressed_sparse_packer() {
    let weights = vec![0i8, 0, 1, 0, 0, 0, -1, 0, 0, 0];
    let config = TernaryPackingConfig::default();
    let packer = CompressedSparsePacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::CompressedSparse);
    
    // Should store: 4 bytes (nnz=2) + 8 bytes (2 indices) + 2 bytes (2 values) = 14 bytes
    assert_eq!(packed.data.len(), 14);
    assert!(packed.compression_ratio < 1.0); // Not efficient for this case
}

#[test]
fn test_compressed_sparse_very_sparse() {
    let mut weights = vec![0i8; 100]; // 100 zeros
    weights[10] = 1;
    weights[50] = -1;
    
    let config = TernaryPackingConfig::default();
    let packer = CompressedSparsePacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    // Should be very efficient: 4 + 8 + 2 = 14 bytes vs 100 bytes
    assert!(packed.compression_ratio > 7.0);
}

#[test]
fn test_hybrid_packer_adaptive() {
    // Create a mixed pattern: dense block + sparse block
    let mut weights = vec![-1i8, 1, -1, 1, -1, 1, -1, 1]; // Dense block
    weights.extend(vec![0i8; 56]); // Sparse block (56 zeros)
    weights[32] = 1; // One non-zero in sparse block
    
    let config = TernaryPackingConfig {
        block_size: Some(8),
        ..Default::default()
    };
    let packer = HybridPacker;
    
    let packed = packer.pack(&weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    
    assert_eq!(weights, unpacked);
    assert_eq!(packed.strategy, TernaryPackingStrategy::Hybrid);
    assert!(packed.compression_ratio > 1.0);
}

#[test]
fn test_packer_factory_auto_selection() {
    // Dense weights should prefer bit-packed
    let dense_weights = vec![-1i8, 1, -1, 1, -1, 1, -1, 1];
    let config = TernaryPackingConfig::default();
    
    let strategy = TernaryPackerFactory::auto_select_strategy(&dense_weights, &config);
    assert!(matches!(strategy, TernaryPackingStrategy::BitPacked2Bit | TernaryPackingStrategy::Base3Packed));
    
    // Sparse weights should prefer sparse formats
    let sparse_weights = vec![0i8; 100];
    let strategy = TernaryPackerFactory::auto_select_strategy(&sparse_weights, &config);
    assert!(matches!(strategy, TernaryPackingStrategy::CompressedSparse | TernaryPackingStrategy::RunLengthEncoded));
}

#[test]
fn test_packer_factory_optimal_packing() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
    let config = TernaryPackingConfig::default();
    
    let packed = TernaryPackerFactory::pack_optimal(&weights, &config).unwrap();
    
    // Should automatically select and apply the best strategy
    assert!(packed.compression_ratio >= 1.0);
    
    // Verify we can unpack correctly
    let packer = TernaryPackerFactory::create_packer(packed.strategy);
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(weights, unpacked);
}

#[test]
fn test_sparsity_analysis() {
    let weights = vec![0i8, 0, 1, 0, -1, 0, 0, 1];
    let analysis = packing_utils::analyze_sparsity(&weights);
    
    assert_eq!(analysis.total_elements, 8);
    assert_eq!(analysis.zero_count, 5);
    assert_eq!(analysis.positive_count, 2);
    assert_eq!(analysis.negative_count, 1);
    assert_eq!(analysis.sparsity_ratio, 5.0 / 8.0);
    assert_eq!(analysis.balance_ratio, 1.0 / 8.0); // |2-1|/8
}

#[test]
fn test_strategy_recommendation() {
    // Very sparse weights
    let very_sparse = vec![0i8; 100];
    assert_eq!(packing_utils::recommend_strategy(&very_sparse), TernaryPackingStrategy::CompressedSparse);
    
    // Moderately sparse weights
    let mod_sparse = vec![0i8, 0, 0, 1, 0, 0, 0, -1, 0, 0];
    assert_eq!(packing_utils::recommend_strategy(&mod_sparse), TernaryPackingStrategy::RunLengthEncoded);
    
    // Dense weights with length divisible by 5
    let base3_optimal = vec![-1i8, 0, 1, -1, 0];
    assert_eq!(packing_utils::recommend_strategy(&base3_optimal), TernaryPackingStrategy::Base3Packed);
    
    // Dense weights, general case
    let dense = vec![-1i8, 1, -1, 1, -1, 1];
    assert_eq!(packing_utils::recommend_strategy(&dense), TernaryPackingStrategy::BitPacked2Bit);
}

#[test]
fn test_packer_suitability() {
    let config = TernaryPackingConfig {
        sparsity_threshold: 0.7,
        ..Default::default()
    };
    
    // Dense weights
    let dense_weights = vec![-1i8, 1, -1, 1, -1, 1];
    assert!(BitPacked2BitPacker.is_suitable(&dense_weights, &config));
    assert!(Base3PackedPacker.is_suitable(&dense_weights, &config));
    assert!(!RunLengthEncodedPacker.is_suitable(&dense_weights, &config)); // Not sparse enough
    assert!(!CompressedSparsePacker.is_suitable(&dense_weights, &config)); // Not sparse enough
    
    // Sparse weights (80% zeros)
    let sparse_weights = vec![0i8, 0, 0, 0, 1, 0, 0, 0, 0, 0];
    assert!(RunLengthEncodedPacker.is_suitable(&sparse_weights, &config));
    assert!(CompressedSparsePacker.is_suitable(&sparse_weights, &config));
}

#[test]
fn test_compression_ratio_calculations() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1];
    let config = TernaryPackingConfig::default();
    
    let strategies = [
        TernaryPackingStrategy::Uncompressed,
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
    ];
    
    for strategy in strategies {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let estimate = packer.estimate_savings(&weights, &config);
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Estimate should be close to actual
        let actual_ratio = weights.len() as f32 / packed.data.len() as f32;
        assert!((estimate.compression_ratio - actual_ratio).abs() < 0.1);
        
        // Verify memory calculations
        assert_eq!(estimate.original_size_bytes, weights.len());
        assert!(estimate.packed_size_bytes > 0);
        assert_eq!(estimate.memory_saved_bytes, estimate.original_size_bytes.saturating_sub(estimate.packed_size_bytes));
    }
}

#[test]
fn test_edge_cases() {
    let config = TernaryPackingConfig::default();
    
    // Empty weights
    let empty_weights: Vec<i8> = vec![];
    let packer = BitPacked2BitPacker;
    let packed = packer.pack(&empty_weights, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(empty_weights, unpacked);
    
    // Single element
    let single_weight = vec![1i8];
    let packed = packer.pack(&single_weight, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(single_weight, unpacked);
    
    // All zeros
    let all_zeros = vec![0i8; 10];
    let packed = packer.pack(&all_zeros, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(all_zeros, unpacked);
    
    // All ones
    let all_ones = vec![1i8; 10];
    let packed = packer.pack(&all_ones, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(all_ones, unpacked);
    
    // All negative ones
    let all_neg_ones = vec![-1i8; 10];
    let packed = packer.pack(&all_neg_ones, &config).unwrap();
    let unpacked = packer.unpack(&packed).unwrap();
    assert_eq!(all_neg_ones, unpacked);
}

#[test]
fn test_large_weights() {
    // Test with larger weight arrays
    let mut weights = Vec::new();
    for i in 0..1000 {
        weights.push(match i % 3 {
            0 => -1i8,
            1 => 0i8,
            2 => 1i8,
            _ => unreachable!(),
        });
    }
    
    let config = TernaryPackingConfig::default();
    
    for strategy in [
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
        TernaryPackingStrategy::Hybrid,
    ] {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
        assert!(packed.compression_ratio > 1.0);
    }
}

#[test]
fn test_error_handling() {
    // Test corrupted data handling
    let weights = vec![-1i8, 0, 1];
    let config = TernaryPackingConfig::default();
    let packer = CompressedSparsePacker;
    
    let mut packed = packer.pack(&weights, &config).unwrap();
    
    // Corrupt the data
    packed.data.clear();
    
    // Should handle corrupted data gracefully
    let result = packer.unpack(&packed);
    assert!(result.is_ok()); // Should return empty or handle gracefully
    
    // Test with insufficient data
    packed.data = vec![0, 0, 0]; // Not enough bytes for header
    let result = packer.unpack(&packed);
    assert!(result.is_ok()); // Should handle gracefully
}

#[test]
fn test_memory_footprint_accuracy() {
    let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
    let config = TernaryPackingConfig::default();
    
    for strategy in [
        TernaryPackingStrategy::Uncompressed,
        TernaryPackingStrategy::BitPacked2Bit,
        TernaryPackingStrategy::Base3Packed,
    ] {
        let packer = TernaryPackerFactory::create_packer(strategy);
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Memory footprint should match data length
        assert_eq!(packed.memory_footprint, packed.data.len());
        
        // Compression ratio should be consistent
        let expected_ratio = weights.len() as f32 / packed.data.len() as f32;
        assert!((packed.compression_ratio - expected_ratio).abs() < 0.01);
    }
}