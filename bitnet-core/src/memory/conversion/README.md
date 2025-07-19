# Memory-Efficient Data Conversion System

This module provides a comprehensive memory-efficient data conversion system for BitNet tensors, designed to minimize memory overhead while maximizing performance across different data types and conversion scenarios.

## Overview

The conversion system supports multiple optimization strategies:

- **Zero-Copy Conversions**: Reinterpret data without copying for compatible types
- **In-Place Conversions**: Modify existing memory buffers when target type is smaller
- **Streaming Conversions**: Process large tensors in chunks to minimize memory usage
- **Batch Conversions**: Group similar conversions for efficiency
- **Conversion Pipeline**: Chain multiple conversions with caching and optimization

## Quick Start

```rust
use bitnet_core::memory::{
    HybridMemoryPool,
    conversion::{ConversionEngine, ConversionConfig},
    tensor::{BitNetTensor, BitNetDType}
};
use std::sync::Arc;

// Initialize conversion engine
let pool = Arc::new(HybridMemoryPool::new()?);
let config = ConversionConfig::default();
let engine = ConversionEngine::new(config, pool.clone())?;

// Create a tensor
let tensor = BitNetTensor::ones(&[1024, 1024], BitNetDType::F32, &device, &pool)?;

// Convert to different data type
let converted = engine.convert(&tensor, BitNetDType::F16)?;
println!("Compression: {:.1}x", 
         tensor.size_bytes() as f64 / converted.size_bytes() as f64);
```

## Architecture

### Core Components

1. **ConversionEngine** (`engine.rs`)
   - Main orchestrator for all conversion operations
   - Automatic strategy selection based on tensor characteristics
   - Integration with memory pool for optimal allocation

2. **Zero-Copy Converter** (`zero_copy.rs`)
   - Handles conversions without memory allocation
   - TensorView for safe memory reinterpretation
   - Compatible type detection and validation

3. **In-Place Converter** (`in_place.rs`)
   - Modifies existing tensor memory buffers
   - Supports both lossy and strict conversion modes
   - Memory-efficient for downsampling operations

4. **Streaming Converter** (`streaming.rs`)
   - Processes large tensors in configurable chunks
   - Minimizes peak memory usage
   - Parallel processing support

5. **Batch Converter** (`batch.rs`)
   - Groups similar conversions for efficiency
   - Automatic batching and optimization
   - Size-based sorting and parallel processing

6. **Conversion Pipeline** (`pipeline.rs`)
   - Chains multiple conversion stages
   - Caching and optimization
   - Stage reordering for efficiency

7. **Metrics System** (`metrics.rs`)
   - Comprehensive performance tracking
   - Real-time statistics and monitoring
   - Strategy effectiveness analysis

8. **Configuration** (`config.rs`)
   - Flexible configuration for all conversion modes
   - Preset configurations for common use cases
   - Validation and optimization hints

## Conversion Strategies

### Zero-Copy Conversions

Best for:
- Same type conversions (F32 → F32)
- Compatible types (F16 ↔ BF16)
- Memory-constrained environments

```rust
// Zero-copy conversion
let result = engine.zero_copy_convert(&tensor, BitNetDType::F32)?;
```

### In-Place Conversions

Best for:
- Downsampling (F32 → F16, F16 → I8)
- When original tensor is no longer needed
- Memory-critical applications

```rust
// In-place conversion
let mut tensor = BitNetTensor::ones(&[1024, 1024], BitNetDType::F32, &device, &pool)?;
engine.in_place_convert(&mut tensor, BitNetDType::F16)?;
```

### Streaming Conversions

Best for:
- Large tensors (>100MB)
- Limited memory environments
- Background processing

```rust
// Streaming conversion with 1MB chunks
let result = engine.streaming_convert(&large_tensor, BitNetDType::I8, 1024 * 1024)?;
```

### Batch Conversions

Best for:
- Multiple similar tensors
- High-throughput scenarios
- Parallel processing

```rust
// Batch convert multiple tensors
let results = engine.batch_convert(&tensors, BitNetDType::F16)?;
```

## Data Type Support

| Type | Size (bits) | Memory Efficiency | Use Case |
|------|-------------|-------------------|----------|
| F32 | 32 | 1.0x | Full precision |
| F16 | 16 | 2.0x | Half precision |
| BF16 | 16 | 2.0x | Brain float |
| I8 | 8 | 4.0x | Quantized weights |
| I4 | 4 | 8.0x | Ultra-compressed |
| I2 | 2 | 16.0x | Extreme compression |
| I1 | 1 | 32.0x | Binary networks |
| BitNet158 | ~1.58 | ~20.0x | BitNet 1.58b |

## Configuration Options

### Preset Configurations

```rust
// High performance (prioritizes speed)
let config = ConversionConfig::high_performance();

// Low memory (minimizes memory usage)
let config = ConversionConfig::low_memory();

// High precision (preserves accuracy)
let config = ConversionConfig::high_precision();
```

### Custom Configuration

```rust
let mut config = ConversionConfig::default();

// Streaming settings
config.streaming.chunk_size = 2 * 1024 * 1024; // 2MB chunks
config.streaming.parallel_chunks = 4;
config.streaming.enable_prefetch = true;

// Batch settings
config.batch.max_batch_size = 64;
config.batch.enable_parallel_processing = true;
config.batch.sort_by_size = true;

// Performance settings
config.performance.use_simd = true;
config.performance.use_vectorization = true;
config.performance.memory_alignment = 64;
```

## Performance Monitoring

### Real-time Statistics

```rust
let stats = engine.get_stats();
println!("Total conversions: {}", stats.total_conversions);
println!("Success rate: {:.1}%", stats.success_rate());
println!("Average time: {:.2}ms", stats.average_time_ms());
println!("Throughput: {:.2} MB/s", stats.throughput_bytes_per_sec() / (1024.0 * 1024.0));
```

### Strategy Analysis

```rust
if let Some(most_used) = stats.most_used_strategy() {
    println!("Most used strategy: {:?}", most_used);
}

if let Some(fastest) = stats.fastest_strategy() {
    println!("Fastest strategy: {:?}", fastest);
}
```

### Memory Efficiency

```rust
println!("Peak memory usage: {} KB", stats.memory_stats.peak_memory_usage / 1024);
println!("Zero-copy percentage: {:.1}%", stats.memory_stats.zero_copy_percentage);
println!("In-place percentage: {:.1}%", stats.memory_stats.in_place_percentage);
```

## Error Handling

The conversion system provides comprehensive error handling:

```rust
use bitnet_core::memory::conversion::ConversionError;

match engine.convert(&tensor, target_dtype) {
    Ok(result) => println!("Conversion successful"),
    Err(ConversionError::IncompatibleTypes { from, to }) => {
        println!("Cannot convert from {} to {}", from, to);
    },
    Err(ConversionError::InsufficientMemory { required, available }) => {
        println!("Need {} bytes, only {} available", required, available);
    },
    Err(ConversionError::InvalidConfiguration(msg)) => {
        println!("Configuration error: {}", msg);
    },
    Err(e) => println!("Other error: {}", e),
}
```

## Thread Safety

All conversion operations are thread-safe and can be used concurrently:

```rust
use std::thread;

let engine = Arc::new(engine);
let handles: Vec<_> = (0..4).map(|i| {
    let engine = engine.clone();
    let tensor = tensor.clone();
    thread::spawn(move || {
        engine.convert(&tensor, BitNetDType::F16)
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("Converted tensor: {}", result);
}
```

## Integration with Memory Pool

The conversion system is tightly integrated with the HybridMemoryPool:

- Automatic memory allocation and deallocation
- Pool-aware optimization strategies
- Memory pressure monitoring
- Garbage collection coordination

## Best Practices

1. **Choose the Right Strategy**
   - Use zero-copy for compatible types
   - Use in-place for memory-critical scenarios
   - Use streaming for large tensors
   - Use batch for multiple conversions

2. **Configure Appropriately**
   - Use preset configurations as starting points
   - Tune chunk sizes based on available memory
   - Enable parallel processing when beneficial

3. **Monitor Performance**
   - Track conversion statistics
   - Identify bottlenecks and optimization opportunities
   - Adjust configuration based on metrics

4. **Handle Errors Gracefully**
   - Check for compatible types before conversion
   - Validate memory availability
   - Implement fallback strategies

## Examples

See the complete examples in:
- `examples/memory_efficient_conversion_demo.rs` - Comprehensive demonstration
- `docs/memory_efficient_conversion_guide.md` - Detailed usage guide
- `tests/memory_efficient_conversion_tests.rs` - Test cases and patterns

## Performance Characteristics

### Conversion Speed (approximate)

| Strategy | Small Tensors | Large Tensors | Memory Usage |
|----------|---------------|---------------|--------------|
| Zero-Copy | ~1μs | ~1μs | 0% overhead |
| In-Place | ~10μs | ~1ms | -50% reduction |
| Streaming | ~100μs | ~10ms | +10% overhead |
| Batch | ~50μs | ~5ms | +20% overhead |

### Memory Efficiency

| Data Type | Compression | Quality Loss | Use Case |
|-----------|-------------|--------------|----------|
| F32→F16 | 2.0x | Minimal | General purpose |
| F32→I8 | 4.0x | Low | Quantized inference |
| F32→I4 | 8.0x | Medium | Ultra-compressed |
| F32→BitNet158 | ~20.0x | Controlled | BitNet models |

## Future Enhancements

- GPU acceleration support
- Advanced compression algorithms
- Dynamic strategy selection
- Cross-device conversion optimization
- Automatic quality assessment