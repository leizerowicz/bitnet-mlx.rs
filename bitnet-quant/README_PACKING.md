# Ternary Weight Packing Strategies

This document describes the different ternary weight packing strategies available in BitNet Rust implementation for efficiently storing and accessing ternary weights {-1, 0, +1}.

## Overview

Ternary weights in BitNet models only require 2 bits per value (since there are 3 possible values), but standard storage uses 8 bits (1 byte) per value. The packing strategies in this module provide various approaches to compress ternary weights, each optimized for different use cases and weight patterns.

## Available Strategies

### 1. Uncompressed (`TernaryPackingStrategy::Uncompressed`)

**Description**: Stores each ternary value in a full byte without compression.

**Use Case**: Baseline comparison and debugging.

**Characteristics**:
- Compression ratio: 1.0x (no compression)
- Access overhead: 0% (direct access)
- Memory usage: 1 byte per value
- Best for: Testing and reference implementation

**Example**:
```rust
use bitnet_quant::quantization::packing::*;

let weights = vec![-1i8, 0, 1, -1];
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::Uncompressed,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// packed.data = [0, 1, 2, 0] (converted to unsigned)
```

### 2. Bit-Packed 2-Bit (`TernaryPackingStrategy::BitPacked2Bit`)

**Description**: Packs 4 ternary values into each byte using 2 bits per value.

**Use Case**: General-purpose compression for dense ternary weights.

**Characteristics**:
- Compression ratio: 4.0x
- Access overhead: ~10% (bit manipulation)
- Memory usage: 0.25 bytes per value
- Best for: Dense weight matrices with balanced distribution

**Example**:
```rust
let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1]; // 8 values
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// packed.data.len() == 2 (8 values packed into 2 bytes)
```

### 3. Base-3 Packed (`TernaryPackingStrategy::Base3Packed`)

**Description**: Uses base-3 encoding to pack 5 ternary values into each byte.

**Use Case**: Optimal compression for arrays with length divisible by 5.

**Characteristics**:
- Compression ratio: 5.0x
- Access overhead: ~15% (base-3 arithmetic)
- Memory usage: 0.2 bytes per value
- Best for: Weight tensors with dimensions divisible by 5

**Example**:
```rust
let weights = vec![-1i8, 0, 1, -1, 0]; // Exactly 5 values
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::Base3Packed,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// packed.data.len() == 1 (5 values packed into 1 byte)
```

### 4. Byte-Aligned (`TernaryPackingStrategy::ByteAligned`)

**Description**: Aligns data to memory boundaries for SIMD optimization.

**Use Case**: Performance-critical applications with SIMD operations.

**Characteristics**:
- Compression ratio: ~1.0x (no compression, adds padding)
- Access overhead: -10% (faster access due to alignment)
- Memory usage: 1+ bytes per value (with padding)
- Best for: SIMD-optimized inference engines

**Example**:
```rust
let weights = vec![-1i8, 0, 1, -1, 0, 1];
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::ByteAligned,
    alignment: 16, // 16-byte alignment for SIMD
    simd_optimized: true,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// Data is padded to 16-byte boundary
```

### 5. Run-Length Encoded (`TernaryPackingStrategy::RunLengthEncoded`)

**Description**: Compresses sequences of identical values using run-length encoding.

**Use Case**: Sparse weights with long runs of zeros or repeated values.

**Characteristics**:
- Compression ratio: Variable (depends on sparsity pattern)
- Access overhead: ~20% (sequential decoding required)
- Memory usage: 2 bytes per run (value + count)
- Best for: Weights with >60% sparsity and clustered patterns

**Example**:
```rust
let weights = vec![0i8, 0, 0, 1, 1, -1, -1, -1, 0, 0];
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::RunLengthEncoded,
    sparsity_threshold: 0.6,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// Encodes as: [(0,3), (1,2), (-1,3), (0,2)] = 8 bytes vs 10 original
```

### 6. Compressed Sparse (`TernaryPackingStrategy::CompressedSparse`)

**Description**: Stores only non-zero values with their indices.

**Use Case**: Very sparse weights (>80% zeros) with scattered non-zero values.

**Characteristics**:
- Compression ratio: Variable (high for very sparse data)
- Access overhead: ~30% (index lookup required)
- Memory usage: 4 bytes per index + 1 byte per value + header
- Best for: Pruned networks with >80% sparsity

**Example**:
```rust
let mut weights = vec![0i8; 100];
weights[10] = 1;
weights[50] = -1;

let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::CompressedSparse,
    sparsity_threshold: 0.8,
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// Stores: header(4) + indices(8) + values(2) = 14 bytes vs 100 original
```

### 7. Hybrid (`TernaryPackingStrategy::Hybrid`)

**Description**: Automatically selects the best strategy for each block of weights.

**Use Case**: Mixed weight patterns with varying density across the tensor.

**Characteristics**:
- Compression ratio: Variable (adapts to local patterns)
- Access overhead: ~25% (strategy dispatch overhead)
- Memory usage: Optimal for each block + metadata
- Best for: Large tensors with heterogeneous sparsity patterns

**Example**:
```rust
let weights = create_mixed_pattern_weights(); // Dense + sparse regions
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::Hybrid,
    block_size: Some(64), // Process in 64-element blocks
    ..Default::default()
};

let packer = TernaryPackerFactory::create_packer(config.strategy);
let packed = packer.pack(&weights, &config)?;
// Each block uses optimal strategy: BitPacked for dense, Sparse for sparse
```

## Configuration Options

### `TernaryPackingConfig`

```rust
pub struct TernaryPackingConfig {
    /// Primary packing strategy
    pub strategy: TernaryPackingStrategy,
    
    /// Block size for block-wise packing (used by Hybrid)
    pub block_size: Option<usize>,
    
    /// Sparsity threshold for switching to sparse formats (0.0-1.0)
    pub sparsity_threshold: f32,
    
    /// Whether to use SIMD-optimized layouts
    pub simd_optimized: bool,
    
    /// Alignment requirements for memory access (bytes)
    pub alignment: usize,
    
    /// Whether to enable compression for sparse formats
    pub enable_compression: bool,
}
```

### Default Configuration

```rust
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    block_size: Some(64),
    sparsity_threshold: 0.7,
    simd_optimized: true,
    alignment: 16,
    enable_compression: true,
};
```

## Usage Patterns

### Automatic Strategy Selection

The library can automatically select the optimal strategy based on weight characteristics:

```rust
use bitnet_quant::quantization::packing::*;

let weights = vec![-1i8, 0, 1, 0, 0, 0, -1, 0, 0, 1];
let config = TernaryPackingConfig::default();

// Automatic recommendation based on sparsity analysis
let recommended = packing_utils::recommend_strategy(&weights);

// Automatic selection with performance consideration
let selected = TernaryPackerFactory::auto_select_strategy(&weights, &config);

// One-step optimal packing
let packed = TernaryPackerFactory::pack_optimal(&weights, &config)?;
```

### Integration with Weight Quantization

```rust
use bitnet_quant::quantization::weights::*;
use bitnet_quant::quantization::packing::*;

// Configure quantization with packing
let mut config = WeightQuantizationConfig::default();
config.packing_config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    ..Default::default()
};

// Quantize weights
let quantized = absmean_quantize_weights(&weights, &device)?;

// Pack the quantized weights
let mut quantized_with_packing = quantized;
quantized_with_packing.pack_weights()?;

println!("Compression ratio: {:.2}x", 
    quantized_with_packing.packing_compression_ratio());
```

### Performance Analysis

```rust
use bitnet_quant::quantization::packing::*;

let weights = generate_test_weights();
let config = TernaryPackingConfig::default();

// Analyze different strategies
let strategies = [
    TernaryPackingStrategy::BitPacked2Bit,
    TernaryPackingStrategy::Base3Packed,
    TernaryPackingStrategy::RunLengthEncoded,
];

for strategy in strategies {
    let packer = TernaryPackerFactory::create_packer(strategy);
    let estimate = packer.estimate_savings(&weights, &config);
    
    println!("{:?}: {:.2}x compression, {:.1}% savings, {:.1}% overhead",
        strategy,
        estimate.compression_ratio,
        estimate.savings_percentage,
        estimate.access_overhead * 100.0
    );
}
```

## Performance Characteristics

### Compression Ratios by Strategy

| Strategy | Dense Weights | 50% Sparse | 90% Sparse | Best Use Case |
|----------|---------------|------------|-------------|---------------|
| Uncompressed | 1.0x | 1.0x | 1.0x | Reference/Debug |
| BitPacked2Bit | 4.0x | 4.0x | 4.0x | General dense |
| Base3Packed | 5.0x | 5.0x | 5.0x | Length divisible by 5 |
| ByteAligned | 1.0x | 1.0x | 1.0x | SIMD optimization |
| RunLengthEncoded | 0.5x | 2.0x | 8.0x | Clustered sparsity |
| CompressedSparse | 0.3x | 1.5x | 15.0x | Very sparse |
| Hybrid | 4.0x | 3.5x | 12.0x | Mixed patterns |

### Access Overhead

| Strategy | Overhead | Reason |
|----------|----------|---------|
| Uncompressed | 0% | Direct access |
| BitPacked2Bit | 10% | Bit manipulation |
| Base3Packed | 15% | Base-3 arithmetic |
| ByteAligned | -10% | SIMD-friendly alignment |
| RunLengthEncoded | 20% | Sequential decoding |
| CompressedSparse | 30% | Index lookup |
| Hybrid | 25% | Strategy dispatch |

## Best Practices

### Strategy Selection Guidelines

1. **Dense weights (< 30% zeros)**: Use `BitPacked2Bit` or `Base3Packed`
2. **Moderate sparsity (30-70% zeros)**: Use `RunLengthEncoded` if clustered, `BitPacked2Bit` if scattered
3. **High sparsity (> 70% zeros)**: Use `CompressedSparse`
4. **Mixed patterns**: Use `Hybrid` with appropriate block size
5. **SIMD-critical paths**: Use `ByteAligned` despite lower compression
6. **Unknown patterns**: Use `TernaryPackerFactory::pack_optimal()`

### Memory vs. Speed Tradeoffs

- **Maximum compression**: `Base3Packed` for dense, `CompressedSparse` for sparse
- **Balanced**: `BitPacked2Bit` for most cases
- **Maximum speed**: `ByteAligned` with SIMD operations
- **Adaptive**: `Hybrid` for varying patterns

### Configuration Tuning

```rust
// For inference-critical applications
let fast_config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::ByteAligned,
    simd_optimized: true,
    alignment: 32, // AVX2 alignment
    ..Default::default()
};

// For memory-constrained environments
let compact_config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::Hybrid,
    block_size: Some(32), // Smaller blocks for better adaptation
    sparsity_threshold: 0.5, // More aggressive sparse detection
    enable_compression: true,
    ..Default::default()
};

// For balanced performance
let balanced_config = TernaryPackingConfig::default();
```

## Examples

See [`bitnet-core/examples/ternary_packing_demo.rs`](../bitnet-core/examples/ternary_packing_demo.rs) for comprehensive examples and [`bitnet-benchmarks/benches/packing_performance.rs`](../bitnet-benchmarks/benches/packing_performance.rs) for performance benchmarks.

## Testing

Run the comprehensive test suite:

```bash
cargo test packing_tests
```

Run performance benchmarks:

```bash
cargo bench --bench packing_performance
```

## Future Enhancements

- **GPU-optimized formats**: CUDA/OpenCL compatible layouts
- **Streaming compression**: For very large weight tensors
- **Hardware-specific optimizations**: ARM NEON, x86 AVX-512
- **Dynamic recompression**: Adaptive strategy switching during runtime
- **Quantization-aware packing**: Joint optimization of quantization and packing