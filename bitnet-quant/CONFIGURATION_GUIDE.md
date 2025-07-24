# BitNet Quantization Configuration Guide

This guide provides comprehensive documentation for the enhanced quantization configuration system in BitNet Rust implementation.

## Overview

The BitNet quantization configuration system provides a unified, type-safe way to configure all aspects of quantization for neural networks. It includes configurations for:

- **Base Quantization**: Core settings shared across all quantization types
- **Weight Quantization**: Specialized settings for neural network weights
- **Activation Quantization**: Dynamic quantization settings for activations
- **Packing Configuration**: Memory-efficient storage strategies for quantized weights
- **SIMD Configuration**: Hardware acceleration settings

## Quick Start

```rust
use bitnet_quant::prelude::*;

// Create a standard BitNet 1.58-bit configuration
let config = EnhancedQuantizationConfig::bitnet_158();

// Create weight quantization configuration
let weight_config = EnhancedWeightQuantizationConfig::bitnet();

// Create activation quantization configuration
let activation_config = EnhancedActivationQuantizationConfig::bitnet();
```

## Configuration Types

### 1. Base Quantization Configuration

The `EnhancedQuantizationConfig` struct provides core settings shared across all quantization operations:

```rust
let config = EnhancedQuantizationConfig {
    precision: QuantizationPrecision::OneFiveFiveBit,
    strategy: QuantizationStrategy::Symmetric,
    per_channel: false,
    clip_threshold: Some(3.0),
    qat_enabled: false,
    calibration_size: Some(1000),
    seed: Some(42),
    verbose: false,
};
```

#### Predefined Configurations

- `QuantizationConfig::bitnet_158()`: Standard BitNet 1.58-bit settings
- `QuantizationConfig::int8()`: 8-bit integer quantization
- `QuantizationConfig::dynamic()`: Dynamic quantization settings

#### Builder Pattern

```rust
let config = QuantizationConfigBuilder::new()
    .precision(QuantizationPrecision::EightBit)
    .strategy(QuantizationStrategy::Dynamic)
    .per_channel(true)
    .clip_threshold(5.0)
    .qat_enabled(true)
    .build();
```

### 2. Weight Quantization Configuration

The `EnhancedWeightQuantizationConfig` struct provides specialized settings for neural network weights:

```rust
let weight_config = EnhancedWeightQuantizationConfig {
    base: QuantizationConfig::bitnet_158(),
    group_size: Some(128),
    normalize_weights: true,
    outlier_threshold: 3.0,
    learnable_scales: false,
    block_size: Some(64),
    ternary_method: TernaryMethod::MeanThreshold,
    custom_threshold_factor: Some(0.7),
    packing: PackingConfig::bitnet(),
    freeze_weights: false,
    weight_decay: Some(1e-4),
    gradient_clip: Some(1.0),
};
```

#### Ternary Quantization Methods

- `TernaryMethod::MeanThreshold`: Use mean absolute value as threshold
- `TernaryMethod::MedianThreshold`: Use median absolute value (more robust to outliers)
- `TernaryMethod::AdaptiveThreshold`: Adaptive threshold based on weight distribution
- `TernaryMethod::OptimalThreshold`: Optimal threshold that minimizes quantization error

#### Builder Pattern

```rust
let weight_config = WeightQuantizationConfigBuilder::new()
    .base(QuantizationConfig::bitnet_158())
    .group_size(128)
    .learnable_scales(true)
    .ternary_method(TernaryMethod::OptimalThreshold)
    .custom_threshold_factor(0.8)
    .build();
```

### 3. Activation Quantization Configuration

The `EnhancedActivationQuantizationConfig` struct provides settings for dynamic activation quantization:

```rust
let activation_config = EnhancedActivationQuantizationConfig {
    base: QuantizationConfig::bitnet_158(),
    moving_average_window: 100,
    outlier_percentile: 99.9,
    per_token: false,
    calibration_warmup: 50,
    ema_decay: 0.99,
    quantize_attention: true,
    attention: AttentionQuantizationConfig::bitnet(),
    smooth_quantization: true,
    temperature: 0.1,
    enable_caching: true,
    cache_size_mb: Some(256),
};
```

#### Attention Quantization

The `AttentionQuantizationConfig` provides fine-grained control over attention mechanism quantization:

```rust
let attention_config = AttentionQuantizationConfig {
    quantize_query: true,
    quantize_key: true,
    quantize_value: true,
    quantize_scores: true,
    quantize_output: true,
    score_clip_threshold: Some(10.0),
    causal_optimization: true,
    sparsity_threshold: Some(0.01),
};
```

### 4. Packing Configuration

The `EnhancedPackingConfig` struct controls how quantized weights are stored in memory:

```rust
let packing_config = EnhancedPackingConfig {
    strategy: TernaryPackingStrategy::Hybrid,
    block_size: Some(64),
    sparsity_threshold: 0.7,
    simd_optimized: true,
    alignment: 16,
    enable_compression: true,
    simd: SimdConfig::default(),
    integrity_checking: true,
    compression_level: 6,
    parallel_packing: false,
    num_threads: None,
};
```

#### Packing Strategies

- `TernaryPackingStrategy::Uncompressed`: No compression (baseline)
- `TernaryPackingStrategy::BitPacked2Bit`: Pack 4 values per byte (2 bits each)
- `TernaryPackingStrategy::Base3Packed`: Pack 5 values per byte using base-3 encoding
- `TernaryPackingStrategy::ByteAligned`: Byte-aligned packing for SIMD operations
- `TernaryPackingStrategy::RunLengthEncoded`: RLE for sparse weights
- `TernaryPackingStrategy::CompressedSparse`: Compressed sparse format
- `TernaryPackingStrategy::Hybrid`: Adaptive strategy selection

#### Predefined Configurations

- `PackingConfig::bitnet()`: Optimized for BitNet models
- `PackingConfig::max_compression()`: Maximum compression ratio
- `PackingConfig::max_speed()`: Maximum unpacking speed

### 5. SIMD Configuration

The `SimdConfig` struct controls hardware acceleration settings:

```rust
let simd_config = SimdConfig {
    enabled: true,
    force_sse2: false,
    force_avx2: false,
    force_neon: false,
    min_simd_size: 64,
    chunk_size: 16,
    enable_prefetch: false,
    prefetch_distance: 8,
    vectorized_ops: true,
    custom_params: HashMap::new(),
};
```

#### Predefined Configurations

- `SimdConfig::aggressive()`: Maximum performance settings
- `SimdConfig::conservative()`: Safe compatibility settings
- `SimdConfig::disabled()`: Disable SIMD optimizations

## Configuration Validation

All configuration structs include validation methods to ensure settings are valid and compatible:

```rust
let config = EnhancedQuantizationConfig::bitnet_158();
match config.validate() {
    Ok(()) => println!("Configuration is valid"),
    Err(e) => println!("Validation error: {}", e),
}
```

### Common Validation Rules

- **Clipping thresholds**: Must be positive and reasonable (0, 10]
- **Calibration sizes**: Must be positive and not excessive [1, 100000]
- **Group/block sizes**: Must be positive powers of 2 where applicable
- **Percentiles**: Must be in range (0, 100]
- **Decay factors**: Must be in range (0, 1)

## Usage Examples

### Basic BitNet Quantization

```rust
use bitnet_quant::prelude::*;
use candle_core::{Device, Tensor};

// Create BitNet configuration
let weight_config = EnhancedWeightQuantizationConfig::bitnet();

// Quantize weights
let device = Device::Cpu;
let weights = Tensor::randn(0.0, 1.0, (64, 128), &device)?;
let quantized = absmean_quantize_weights(&weights, &device)?;
```

### Custom Configuration

```rust
// Build custom configuration
let custom_config = WeightQuantizationConfigBuilder::new()
    .base(QuantizationConfig::int8())
    .group_size(64)
    .learnable_scales(true)
    .ternary_method(TernaryMethod::OptimalThreshold)
    .packing(PackingConfig::max_compression())
    .build();

// Validate before use
custom_config.validate()?;
```

### Dynamic Activation Quantization

```rust
// Configure dynamic activation quantization
let activation_config = EnhancedActivationQuantizationConfig::dynamic()
    .with_per_token()
    .with_window(200)
    .with_caching(512);

// Use with activation quantizer
let quantizer = create_activation_quantizer(activation_config);
```

### SIMD-Optimized Packing

```rust
// Configure for maximum performance
let packing_config = EnhancedPackingConfig::max_speed()
    .with_parallel_packing(Some(8))
    .with_compression_level(0);

// Use with ternary packer
let packer = TernaryPackerFactory::create_packer(packing_config.strategy);
```

## Best Practices

### 1. Configuration Selection

- **BitNet models**: Use `*.bitnet()` configurations for optimal results
- **Memory-constrained environments**: Use `PackingConfig::max_compression()`
- **Performance-critical applications**: Use `PackingConfig::max_speed()`
- **Research/experimentation**: Use builder patterns for custom configurations

### 2. Validation

- Always validate configurations before use
- Use validation in unit tests to catch configuration errors early
- Handle validation errors gracefully in production code

### 3. Hardware Optimization

- Use `SimdConfig::aggressive()` on modern CPUs
- Use `SimdConfig::conservative()` for compatibility
- Disable SIMD on older hardware or when debugging

### 4. Memory Management

- Enable caching for frequently accessed activations
- Use appropriate compression levels based on memory constraints
- Consider parallel packing for large models

## Migration from Legacy Configurations

The new configuration system is designed to be backward compatible while providing enhanced functionality:

```rust
// Legacy approach
let old_config = WeightQuantizationConfig::default();

// Enhanced approach
let new_config = EnhancedWeightQuantizationConfig::default();

// Migration
let migrated_config = EnhancedWeightQuantizationConfig {
    base: EnhancedQuantizationConfig::from_legacy(&old_config.base),
    // ... other fields
    ..Default::default()
};
```

## Performance Considerations

### Memory Usage

- **Uncompressed**: 1 byte per weight (baseline)
- **BitPacked2Bit**: 0.25 bytes per weight (4x compression)
- **Base3Packed**: 0.2 bytes per weight (5x compression)
- **CompressedSparse**: Variable, depends on sparsity

### Computational Overhead

- **Validation**: Minimal overhead, performed once
- **SIMD operations**: 2-8x speedup on compatible hardware
- **Parallel packing**: Linear speedup with thread count

### Cache Efficiency

- Use appropriate alignment settings for your target architecture
- Enable prefetching for large sequential operations
- Consider cache size when enabling activation caching

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check parameter ranges and compatibility
2. **SIMD Not Working**: Verify hardware support and enable appropriate flags
3. **Memory Issues**: Reduce cache sizes or compression levels
4. **Performance Issues**: Profile and adjust SIMD/packing settings

### Debug Mode

Enable verbose logging for detailed configuration information:

```rust
let config = EnhancedQuantizationConfig::bitnet_158().with_verbose();
```

## API Reference

For complete API documentation, see the generated docs:

```bash
cargo doc --open
```

Key modules:
- `bitnet_quant::quantization::config`: Configuration structs and builders
- `bitnet_quant::quantization::weights`: Weight quantization functions
- `bitnet_quant::quantization::activations`: Activation quantization functions
- `bitnet_quant::quantization::packing`: Packing strategies and utilities
- `bitnet_quant::quantization::simd_unpacking`: SIMD-optimized operations