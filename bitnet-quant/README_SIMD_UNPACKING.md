# SIMD-Optimized Weight Unpacking

This document describes the SIMD-optimized weight unpacking implementation for BitNet models, which provides significant performance improvements for unpacking ternary weights from compressed formats.

## Overview

The SIMD unpacking module (`simd_unpacking.rs`) provides hardware-accelerated implementations for unpacking ternary weights {-1, 0, +1} from various compressed formats. It automatically detects available SIMD instruction sets and falls back to optimized scalar implementations when SIMD is not available.

## Supported SIMD Instruction Sets

### x86/x86_64 Architectures
- **SSE2**: 128-bit SIMD operations (baseline for most modern x86 processors)
- **AVX2**: 256-bit SIMD operations (Intel Haswell+ and AMD Excavator+)

### ARM Architectures
- **NEON**: 128-bit SIMD operations (ARMv7+ and AArch64)

### Automatic Fallback
- **Optimized Scalar**: Hand-optimized scalar implementations when SIMD is unavailable
- **Standard Scalar**: Basic scalar fallback for maximum compatibility

## Performance Benefits

The SIMD implementations provide significant performance improvements:

| Strategy | Typical Speedup | Best Case | Architecture |
|----------|----------------|-----------|--------------|
| BitPacked2Bit | 2-4x | 6x | AVX2 |
| ByteAligned | 3-8x | 12x | AVX2 |
| Base3Packed | 1.5-2x | 3x | SSE2 |
| Sparse formats | 1.2-2x | 2.5x | All |

*Note: Actual performance depends on data size, CPU architecture, and memory bandwidth.*

## Usage

### Basic Usage

```rust
use bitnet_quant::prelude::*;

// Create a SIMD-optimized unpacker
let unpacker = SimdUnpacker::new();

// Unpack weights (automatically uses best available SIMD)
let unpacked_weights = unpacker.unpack(&packed_weights)?;

// Convenience function
let unpacked_weights = simd_unpack_weights(&packed_weights)?;
```

### Capability Detection

```rust
use bitnet_quant::prelude::*;

// Detect available SIMD capabilities
let capabilities = SimdCapabilities::detect();
println!("SSE2: {}", capabilities.sse2);
println!("AVX2: {}", capabilities.avx2);
println!("NEON: {}", capabilities.neon);

// Create unpacker with specific capabilities (for testing)
let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
    sse2: false,
    avx2: false,
    neon: false,
});
```

### Performance Benchmarking

```rust
use bitnet_quant::quantization::simd_unpacking::benchmark::*;

// Benchmark SIMD vs scalar performance
let benchmark = benchmark_unpacking(&packed_weights, 1000)?;
println!("Speedup: {:.2}x", benchmark.speedup);
println!("SIMD time: {} ns", benchmark.simd_time_ns);
println!("Scalar time: {} ns", benchmark.scalar_time_ns);
```

## Supported Packing Strategies

### BitPacked2Bit (Recommended for SIMD)

**Description**: Packs 4 ternary values into each byte using 2 bits per value.

**SIMD Optimization**: Excellent - processes multiple bytes in parallel with bit manipulation.

**Performance**: 2-6x speedup depending on architecture.

```rust
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    simd_optimized: true,
    ..Default::default()
};
```

### ByteAligned (Best for SIMD)

**Description**: Stores each ternary value in a full byte with memory alignment.

**SIMD Optimization**: Excellent - direct vectorized subtraction operations.

**Performance**: 3-12x speedup, especially with proper alignment.

```rust
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::ByteAligned,
    alignment: 32, // AVX2-friendly alignment
    simd_optimized: true,
    ..Default::default()
};
```

### Base3Packed

**Description**: Uses base-3 encoding to pack 5 ternary values per byte.

**SIMD Optimization**: Limited - complex base-3 arithmetic reduces SIMD benefits.

**Performance**: 1.5-3x speedup with lookup table optimizations.

### Sparse Formats (RunLengthEncoded, CompressedSparse)

**Description**: Specialized formats for sparse weight matrices.

**SIMD Optimization**: Limited - inherently sequential or index-based operations.

**Performance**: 1.2-2.5x speedup mainly from optimized scalar code.

## Implementation Details

### Architecture-Specific Optimizations

#### x86/x86_64 AVX2 Implementation
```rust
// Example: 32 bytes processed in parallel
let input = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
let offset = _mm256_set1_epi8(1);
let ternary = _mm256_sub_epi8(input, offset);
_mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, ternary);
```

#### ARM NEON Implementation
```rust
// Example: 16 bytes processed in parallel
let input = vld1q_u8(data.as_ptr());
let offset = vdupq_n_s8(1);
let ternary = vsubq_s8(vreinterpretq_s8_u8(input), offset);
vst1q_s8(result.as_mut_ptr(), ternary);
```

### Chunked Processing

The implementation processes data in chunks to maximize SIMD utilization:

- **AVX2**: 32-byte chunks (256-bit registers)
- **SSE2/NEON**: 16-byte chunks (128-bit registers)
- **Scalar**: 8-byte chunks (optimized loop unrolling)

### Memory Alignment

For optimal performance, especially with ByteAligned strategy:

```rust
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::ByteAligned,
    alignment: 32, // Match SIMD register size
    simd_optimized: true,
    ..Default::default()
};
```

## Performance Tuning

### Choosing the Right Strategy

1. **For maximum SIMD performance**: Use `ByteAligned` with proper alignment
2. **For balanced compression and speed**: Use `BitPacked2Bit`
3. **For maximum compression**: Use `Base3Packed` (limited SIMD benefits)
4. **For sparse data**: Use `CompressedSparse` or `RunLengthEncoded`

### Optimal Configuration

```rust
// High-performance configuration
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::ByteAligned,
    alignment: 32, // AVX2 alignment
    simd_optimized: true,
    block_size: Some(1024), // Process in large blocks
    ..Default::default()
};

// Balanced configuration
let config = TernaryPackingConfig {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    simd_optimized: true,
    alignment: 16, // SSE2/NEON alignment
    ..Default::default()
};
```

### Benchmarking Your Workload

```rust
// Run comprehensive benchmarks
cargo run --example simd_unpacking_demo --release

// Run detailed criterion benchmarks
cargo bench --bench simd_unpacking_performance
```

## Integration with Existing Code

### Drop-in Replacement

The SIMD unpacker is designed as a drop-in replacement for existing unpacking:

```rust
// Before
let packer = TernaryPackerFactory::create_packer(strategy);
let unpacked = packer.unpack(&packed)?;

// After (with SIMD optimization)
let unpacked = simd_unpack_weights(&packed)?;
```

### Gradual Migration

```rust
// Use SIMD for performance-critical paths
if packed.metadata.element_count > 10000 {
    // Use SIMD for large tensors
    simd_unpack_weights(&packed)?
} else {
    // Use original implementation for small tensors
    let packer = TernaryPackerFactory::create_packer(packed.strategy);
    packer.unpack(&packed)?
}
```

## Testing and Validation

### Correctness Tests

The implementation includes comprehensive tests to ensure correctness:

```bash
# Run all SIMD unpacking tests
cargo test simd_unpacking

# Run specific strategy tests
cargo test test_bit_packed_2bit_unpacking
cargo test test_simd_vs_scalar_consistency
```

### Performance Tests

```bash
# Run performance benchmarks
cargo bench simd_unpacking

# Run example with performance analysis
cargo run --example simd_unpacking_demo --release
```

## Troubleshooting

### Common Issues

1. **No SIMD acceleration detected**
   - Check CPU capabilities with `SimdCapabilities::detect()`
   - Ensure you're compiling with appropriate target features

2. **Lower than expected performance**
   - Use `ByteAligned` strategy for maximum SIMD benefit
   - Ensure proper memory alignment
   - Test with larger data sizes (SIMD overhead affects small tensors)

3. **Compilation errors on specific targets**
   - The implementation automatically falls back to scalar on unsupported targets
   - Use conditional compilation for target-specific optimizations

### Debugging Performance

```rust
// Enable detailed benchmarking
let benchmark = benchmark_unpacking(&packed, 1000)?;
println!("Strategy: {:?}", benchmark.strategy);
println!("Element count: {}", benchmark.element_count);
println!("SIMD capabilities: {:?}", benchmark.simd_capabilities);
println!("Speedup: {:.2}x", benchmark.speedup);
```

## Future Enhancements

### Planned Optimizations

1. **AVX-512 Support**: 512-bit SIMD for latest Intel processors
2. **GPU Acceleration**: CUDA/OpenCL implementations for massive parallelism
3. **Adaptive Chunking**: Dynamic chunk size based on data characteristics
4. **Prefetching**: Memory prefetch hints for better cache utilization

### Contributing

To contribute SIMD optimizations:

1. Add target-specific implementations in `simd_unpacking.rs`
2. Include comprehensive tests for correctness
3. Add performance benchmarks
4. Update documentation with new capabilities

## Examples

See the following files for complete examples:

- [`examples/simd_unpacking_demo.rs`](examples/simd_unpacking_demo.rs) - Performance demonstration
- [`benches/simd_unpacking_performance.rs`](../bitnet-benchmarks/benches/simd_unpacking_performance.rs) - Detailed benchmarks
- [`tests/`](tests/) - Correctness tests

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Rust SIMD Documentation](https://doc.rust-lang.org/std/arch/index.html)