# Memory-Efficient Data Conversion Guide

This guide provides comprehensive documentation for the BitNet memory-efficient data conversion system, including usage examples, best practices, and performance optimization techniques.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Conversion Strategies](#conversion-strategies)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The BitNet memory-efficient data conversion system provides optimized tensor data type conversions with minimal memory overhead. It supports various conversion strategies including zero-copy, in-place, streaming, and batch operations.

### Key Features

- **Zero-copy conversions** for compatible data types
- **Streaming conversion** for large tensors that don't fit in memory
- **In-place conversions** to minimize memory allocation
- **Batch processing** for multiple tensors
- **Memory pooling** integration for efficient allocation
- **Comprehensive metrics** and monitoring
- **Device-aware conversions** (CPU ↔ Metal ↔ MLX)
- **Thread-safe operations**

### Supported Data Types

- **F32**: 32-bit floating point (full precision)
- **F16**: 16-bit floating point (half precision)
- **BF16**: 16-bit brain floating point
- **I8**: 8-bit signed integer
- **I4**: 4-bit signed integer (packed)
- **I2**: 2-bit signed integer (packed)
- **I1**: 1-bit signed integer (packed)
- **BitNet158**: BitNet 1.58b ternary format (-1, 0, +1)

## Quick Start

```rust
use bitnet_core::memory::{
    HybridMemoryPool, 
    conversion::{ConversionEngine, ConversionConfig},
    tensor::{BitNetTensor, BitNetDType}
};
use bitnet_core::device::auto_select_device;
use std::sync::Arc;

// Create memory pool and conversion engine
let pool = Arc::new(HybridMemoryPool::new()?);
let config = ConversionConfig::default();
let engine = ConversionEngine::new(config, pool.clone())?;

// Create a tensor
let device = auto_select_device();
let tensor = BitNetTensor::ones(&[1024, 1024], BitNetDType::F32, &device, &pool)?;

// Convert F32 to F16 (2x memory reduction)
let f16_tensor = engine.convert(&tensor, BitNetDType::F16)?;

// Convert F32 to I8 (4x memory reduction)
let i8_tensor = engine.convert(&tensor, BitNetDType::I8)?;

println!("Original size: {} bytes", tensor.size_bytes());
println!("F16 size: {} bytes", f16_tensor.size_bytes());
println!("I8 size: {} bytes", i8_tensor.size_bytes());
```

## Core Components

### ConversionEngine

The main interface for all conversion operations. It automatically selects the optimal conversion strategy based on the input and target data types.

```rust
let engine = ConversionEngine::new(config, pool)?;

// Single tensor conversion
let result = engine.convert(&tensor, BitNetDType::F16)?;

// Batch conversion
let tensors = vec![tensor1, tensor2, tensor3];
let results = engine.batch_convert(&tensors, BitNetDType::I8)?;

// Mixed batch conversion
let conversions = vec![
    (tensor1, BitNetDType::F16),
    (tensor2, BitNetDType::I8),
    (tensor3, BitNetDType::I4),
];
let results = engine.batch_convert_mixed(&conversions)?;
```

### ConversionPipeline

Chains multiple conversion operations with memory optimization and caching.

```rust
let pipeline = engine.create_pipeline()?
    .add_stage(BitNetDType::F16)    // F32 -> F16
    .add_stage(BitNetDType::I8)     // F16 -> I8
    .add_stage(BitNetDType::I4)     // I8 -> I4
    .optimize();                    // Reorder for efficiency

let result = pipeline.execute(&tensor)?;
```

### Individual Converters

For specific conversion strategies:

```rust
// Zero-copy converter
let zero_copy = ZeroCopyConverter::new();
let result = zero_copy.reinterpret_cast(&tensor, BitNetDType::F32, &pool)?;

// Streaming converter
let streaming = StreamingConverter::new(streaming_config)?;
let result = streaming.stream_convert(&large_tensor, BitNetDType::I8, &pool)?;

// In-place converter
let in_place = InPlaceConverter::new_lossy();
let mut tensor = tensor.clone();
in_place.convert_in_place(&mut tensor, BitNetDType::F16)?;

// Batch converter
let batch = BatchConverter::new(batch_config)?;
let results = batch.batch_convert(&tensors, BitNetDType::F16, &pool)?;
```

## Conversion Strategies

### Zero-Copy Conversion

Best for: Same data type or compatible types with identical memory layout.

```rust
// Same type (truly zero-copy)
let result = engine.zero_copy_convert(&f32_tensor, BitNetDType::F32)?;

// Compatible types (F16 ↔ BF16)
let bf16_tensor = engine.zero_copy_convert(&f16_tensor, BitNetDType::BF16)?;
```

**Memory overhead**: None
**Performance**: Fastest (instant for same type)

### In-Place Conversion

Best for: Conversions to smaller data types on the same device.

```rust
let mut tensor = BitNetTensor::ones(&[512, 512], BitNetDType::F32, &device, &pool)?;
engine.in_place_convert(&mut tensor, BitNetDType::F16)?;
// tensor is now F16, using the same memory buffer
```

**Memory overhead**: None
**Performance**: Very fast
**Limitations**: Target type must be same or smaller size

### Streaming Conversion

Best for: Large tensors or when memory is limited.

```rust
// Configure for low memory usage
let mut config = ConversionConfig::low_memory();
config.streaming.chunk_size = 1024 * 1024; // 1MB chunks

let engine = ConversionEngine::new(config, pool)?;
let result = engine.streaming_convert(&huge_tensor, BitNetDType::I8, 1024)?;
```

**Memory overhead**: Only chunk size
**Performance**: Good for large tensors
**Benefits**: Can process tensors larger than available memory

### Batch Conversion

Best for: Multiple tensors with similar characteristics.

```rust
// Group tensors by type and device for efficiency
let mut config = ConversionConfig::default();
config.batch.group_by_dtype = true;
config.batch.group_by_device = true;
config.batch.enable_parallel_processing = true;

let engine = ConversionEngine::new(config, pool)?;
let results = engine.batch_convert(&many_tensors, BitNetDType::F16)?;
```

**Memory overhead**: Depends on batch size
**Performance**: Excellent for many tensors
**Benefits**: Optimized grouping and parallel processing

## Usage Examples

### Example 1: Model Quantization Pipeline

```rust
use bitnet_core::memory::conversion::{ConversionEngine, ConversionConfig, ConversionQuality};

// Create high-precision configuration for model quantization
let mut config = ConversionConfig::high_precision();
config.default_quality = ConversionQuality::Precise;

let engine = ConversionEngine::new(config, pool)?;

// Create quantization pipeline: F32 -> F16 -> I8 -> I4
let quantization_pipeline = engine.create_pipeline()?
    .add_stage_with_strategy(BitNetDType::F16, ConversionStrategy::InPlace)
    .add_stage_with_strategy(BitNetDType::I8, ConversionStrategy::Streaming)
    .add_stage_with_strategy(BitNetDType::I4, ConversionStrategy::Streaming)
    .optimize();

// Process model weights
let weight_tensors = load_model_weights()?;
let quantized_weights = quantization_pipeline.execute_batch(&weight_tensors)?;

// Calculate compression ratio
let original_size: usize = weight_tensors.iter().map(|t| t.size_bytes()).sum();
let quantized_size: usize = quantized_weights.iter().map(|t| t.size_bytes()).sum();
let compression_ratio = original_size as f64 / quantized_size as f64;

println!("Compression ratio: {:.2}x", compression_ratio);
println!("Memory saved: {} MB", (original_size - quantized_size) / (1024 * 1024));
```

### Example 2: Memory-Constrained Environment

```rust
// Configure for minimal memory usage
let config = ConversionConfig::low_memory();
let engine = ConversionEngine::new(config, pool)?;

// Process large tensors with streaming
let large_tensors = vec![
    create_large_tensor(&[4096, 4096])?,
    create_large_tensor(&[8192, 2048])?,
    create_large_tensor(&[2048, 8192])?,
];

for (i, tensor) in large_tensors.iter().enumerate() {
    println!("Processing tensor {} of {}", i + 1, large_tensors.len());
    
    // Use small chunks to minimize memory usage
    let result = engine.streaming_convert(tensor, BitNetDType::I8, 512 * 1024)?;
    
    // Process result immediately to free memory
    process_quantized_tensor(&result)?;
    
    // Check memory usage
    let metrics = engine.get_stats();
    println!("Peak memory: {} MB", metrics.memory_stats.peak_memory_usage / (1024 * 1024));
}
```

### Example 3: High-Performance Batch Processing

```rust
// Configure for maximum performance
let config = ConversionConfig::high_performance();
let engine = ConversionEngine::new(config, pool)?;

// Prepare large batch of tensors
let batch_size = 1000;
let tensors: Vec<_> = (0..batch_size)
    .map(|i| BitNetTensor::random(&[64, 64], BitNetDType::F32, &device, &pool))
    .collect::<Result<Vec<_>, _>>()?;

// Time the batch conversion
let start = std::time::Instant::now();
let results = engine.batch_convert(&tensors, BitNetDType::F16)?;
let duration = start.elapsed();

println!("Converted {} tensors in {:?}", batch_size, duration);
println!("Throughput: {:.2} tensors/sec", batch_size as f64 / duration.as_secs_f64());

// Verify results
assert_eq!(results.len(), batch_size);
for result in &results {
    assert_eq!(result.dtype(), BitNetDType::F16);
}
```

### Example 4: Mixed Conversion Types

```rust
// Different tensors need different target types
let conversions = vec![
    // Weights: F32 -> I4 for maximum compression
    (weight_tensor, BitNetDType::I4),
    // Activations: F32 -> F16 for speed
    (activation_tensor, BitNetDType::F16),
    // Biases: F32 -> I8 for balance
    (bias_tensor, BitNetDType::I8),
    // Embeddings: F32 -> BitNet158 for BitNet compatibility
    (embedding_tensor, BitNetDType::BitNet158),
];

let results = engine.batch_convert_mixed(&conversions)?;

// Verify each conversion
for ((original, target_type), result) in conversions.iter().zip(results.iter()) {
    assert_eq!(result.dtype(), *target_type);
    println!("Converted {} -> {}: {:.2}x compression", 
             original.dtype(), target_type,
             original.size_bytes() as f64 / result.size_bytes() as f64);
}
```

## Performance Optimization

### Configuration Tuning

```rust
// For CPU-intensive workloads
let mut config = ConversionConfig::default();
config.performance.use_simd = true;
config.performance.use_vectorization = true;
config.performance.enable_loop_unrolling = true;
config.worker_threads = num_cpus::get() * 2;

// For memory-constrained environments
let mut config = ConversionConfig::low_memory();
config.streaming.chunk_size = 256 * 1024; // Smaller chunks
config.batch.max_batch_size = 8;          // Smaller batches
config.max_memory_usage = 128 * 1024 * 1024; // 128MB limit

// For precision-critical applications
let mut config = ConversionConfig::high_precision();
config.performance.use_simd = false;      // Disable SIMD for consistency
config.streaming.parallel_chunks = 1;    // Sequential processing
config.batch.enable_parallel_processing = false;
```

### Strategy Selection

```rust
// Get information about optimal strategy
let info = engine.get_optimal_strategy_info(
    BitNetDType::F32,
    BitNetDType::F16,
    &[1024, 1024],
    &device,
);

println!("Optimal strategy: {:?}", info.strategy);
println!("Estimated time: {}ms", info.estimated_time_ms);
println!("Memory overhead: {} bytes", info.memory_overhead_bytes);
println!("Compression ratio: {:.2}x", info.compression_ratio);
println!("Description: {}", info.description());

// Use the information to make decisions
if info.is_zero_copy {
    println!("This conversion is zero-copy!");
} else if info.is_in_place {
    println!("This conversion can be done in-place");
} else {
    println!("This conversion requires memory allocation");
}
```

### Memory Pool Optimization

```rust
// Configure memory pool for conversion workloads
let mut pool_config = MemoryPoolConfig::default();
pool_config.small_block_threshold = 2 * 1024 * 1024; // 2MB threshold
pool_config.large_pool_max_size = 2 * 1024 * 1024 * 1024; // 2GB max
pool_config.enable_advanced_tracking = true;

let pool = Arc::new(HybridMemoryPool::with_config(pool_config)?);

// Register memory pressure callback
pool.register_pressure_callback(Box::new(|level| {
    match level {
        MemoryPressureLevel::Critical => {
            eprintln!("CRITICAL: Memory pressure detected!");
            // Trigger aggressive cleanup
        }
        MemoryPressureLevel::High => {
            println!("High memory pressure, consider cleanup");
        }
        _ => {}
    }
}));
```

## Monitoring and Metrics

### Basic Metrics

```rust
// Get conversion statistics
let stats = engine.get_stats();

println!("Total conversions: {}", stats.total_conversions);
println!("Success rate: {:.2}%", stats.success_rate());
println!("Average time: {:.2}ms", stats.average_time_ms());
println!("Throughput: {:.2} bytes/sec", stats.throughput_bytes_per_sec());

// Strategy usage
if let Some(most_used) = stats.most_used_strategy() {
    println!("Most used strategy: {:?}", most_used);
}

if let Some(fastest) = stats.fastest_strategy() {
    println!("Fastest strategy: {:?}", fastest);
}
```

### Detailed Metrics

```rust
// Access detailed metrics by strategy
for (strategy, metrics) in &stats.strategy_metrics {
    println!("Strategy {:?}:", strategy);
    println!("  Usage: {} times", metrics.usage_count);
    println!("  Success rate: {:.2}%", metrics.success_rate());
    println!("  Average time: {:.2}ms", metrics.average_time_ms());
    println!("  Min time: {}ms", metrics.min_time_ms);
    println!("  Max time: {}ms", metrics.max_time_ms);
}

// Data type conversion metrics
for (conversion, metrics) in &stats.dtype_metrics {
    println!("Conversion {} -> {}:", conversion.from, conversion.to);
    println!("  Count: {}", metrics.conversion_count);
    println!("  Average time: {:.2}ms", metrics.average_time_ms());
    println!("  Compression ratio: {:.2}x", metrics.average_compression_ratio);
}
```

### Pipeline Metrics

```rust
let pipeline_stats = pipeline.get_stats()?;

println!("Pipeline Statistics:");
println!("  Executions: {}", pipeline_stats.total_executions);
println!("  Stages executed: {}", pipeline_stats.total_stages_executed);
println!("  Cache hit ratio: {:.2}%", pipeline_stats.cache_hit_ratio() * 100.0);
println!("  Cache size: {} entries", pipeline_stats.cache_size);
println!("  Peak memory: {} MB", pipeline_stats.peak_memory_usage / (1024 * 1024));
```

### Real-time Monitoring

```rust
use std::thread;
use std::time::Duration;

// Monitor conversion performance in real-time
thread::spawn(move || {
    loop {
        thread::sleep(Duration::from_secs(10));
        
        let stats = engine.get_stats();
        let recent_events = engine.get_recent_events(10);
        
        println!("=== Conversion Monitor ===");
        println!("Total conversions: {}", stats.total_conversions);
        println!("Recent average time: {:.2}ms", 
                 recent_events.iter()
                     .map(|e| e.duration_ms)
                     .sum::<u64>() as f64 / recent_events.len().max(1) as f64);
        
        // Check for errors
        if stats.error_stats.error_rate > 5.0 {
            eprintln!("WARNING: High error rate: {:.2}%", stats.error_stats.error_rate);
        }
        
        // Check memory usage
        if stats.memory_stats.peak_memory_usage > 1024 * 1024 * 1024 {
            println!("INFO: High memory usage: {} GB", 
                     stats.memory_stats.peak_memory_usage / (1024 * 1024 * 1024));
        }
    }
});
```

## Best Practices

### 1. Choose the Right Strategy

```rust
// For same-type operations, always use zero-copy
if source_dtype == target_dtype {
    let result = engine.zero_copy_convert(&tensor, target_dtype)?;
}

// For downsizing on same device, prefer in-place
else if target_dtype.bits_per_element() <= source_dtype.bits_per_element() {
    let mut tensor = tensor.clone();
    engine.in_place_convert(&mut tensor, target_dtype)?;
}

// For large tensors, use streaming
else if tensor.size_bytes() > 100 * 1024 * 1024 {
    let result = engine.streaming_convert(&tensor, target_dtype, chunk_size)?;
}

// For multiple tensors, use batch processing
else if tensors.len() > 10 {
    let results = engine.batch_convert(&tensors, target_dtype)?;
}
```

### 2. Memory Management

```rust
// Clear caches periodically
if conversion_count % 1000 == 0 {
    pipeline.clear_cache()?;
    engine.clear_stats();
}

// Use scoped processing for large batches
for chunk in tensors.chunks(100) {
    let results = engine.batch_convert(chunk, target_dtype)?;
    process_results(results)?;
    // Results go out of scope, freeing memory
}

// Monitor memory pressure
let detailed_metrics = pool.get_detailed_metrics();
if let Some(metrics) = detailed_metrics {
    if metrics.pressure_level == MemoryPressureLevel::High {
        // Reduce batch sizes or trigger cleanup
        trigger_memory_cleanup()?;
    }
}
```

### 3. Error Handling

```rust
use bitnet_core::memory::conversion::ConversionError;

match engine.convert(&tensor, target_dtype) {
    Ok(result) => {
        // Success
        process_result(result)?;
    }
    Err(ConversionError::UnsupportedConversion { from, to }) => {
        eprintln!("Conversion {} -> {} not supported", from, to);
        // Try alternative approach
        let intermediate = engine.convert(&tensor, BitNetDType::F16)?;
        let result = engine.convert(&intermediate, target_dtype)?;
    }
    Err(ConversionError::MemoryError(e)) => {
        eprintln!("Memory allocation failed: {}", e);
        // Try streaming approach
        let result = engine.streaming_convert(&tensor, target_dtype, 512 * 1024)?;
    }
    Err(ConversionError::DeviceMismatch { source, target }) => {
        eprintln!("Device mismatch: {} -> {}", source, target);
        // Migrate tensor to target device first
        let migrated = tensor.to_device(&target_device, &pool)?;
        let result = engine.convert(&migrated, target_dtype)?;
    }
    Err(e) => {
        eprintln!("Conversion failed: {}", e);
        return Err(e.into());
    }
}
```

### 4. Performance Optimization

```rust
// Pre-warm the conversion engine
let dummy_tensor = BitNetTensor::zeros(&[1, 1], BitNetDType::F32, &device, &pool)?;
let _ = engine.convert(&dummy_tensor, BitNetDType::F16)?;

// Use appropriate quality settings
let high_speed_result = engine.convert_with_quality(
    &tensor, 
    target_dtype, 
    ConversionQuality::Fast
)?;

let high_precision_result = engine.convert_with_quality(
    &tensor, 
    target_dtype, 
    ConversionQuality::Precise
)?;

// Batch similar operations
let f32_tensors: Vec<_> = tensors.iter()
    .filter(|t| t.dtype() == BitNetDType::F32)
    .collect();
let f32_results = engine.batch_convert(&f32_tensors, BitNetDType::F16)?;

let f16_tensors: Vec<_> = tensors.iter()
    .filter(|t| t.dtype() == BitNetDType::F16)
    .collect();
let f16_results = engine.batch_convert(&f16_tensors, BitNetDType::I8)?;
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```rust
// Problem: Large tensor conversion fails with memory error
// Solution: Use streaming conversion
let config = ConversionConfig::low_memory();
let engine = ConversionEngine::new(config, pool)?;
let result = engine.streaming_convert(&large_tensor, target_dtype, 1024 * 1024)?;
```

#### 2. Slow Conversion Performance

```rust
// Problem: Conversions are slower than expected
// Solution: Check strategy selection and use batch processing
let info = engine.get_optimal_strategy_info(source_dtype, target_dtype, &shape, &device);
println!("Using strategy: {:?}", info.strategy);

if tensors.len() > 1 {
    // Use batch conversion instead of individual conversions
    let results = engine.batch_convert(&tensors, target_dtype)?;
}
```

#### 3. Precision Loss

```rust
// Problem: Quantization introduces too much error
// Solution: Use high-precision configuration and intermediate steps
let config = ConversionConfig::high_precision();
let engine = ConversionEngine::new(config, pool)?;

// Use gradual quantization: F32 -> F16 -> I8 instead of F32 -> I8
let pipeline = engine.create_pipeline()?
    .add_stage(BitNetDType::F16)
    .add_stage(BitNetDType::I8);
let result = pipeline.execute(&tensor)?;
```

#### 4. Device Compatibility Issues

```rust
// Problem: Conversion fails due to device mismatch
// Solution: Check device compatibility and migrate if needed
if !engine.is_conversion_supported(source_dtype, target_dtype, &device) {
    eprintln!("Conversion not supported on device {:?}", device);
    
    // Try on CPU
    let cpu_device = get_cpu_device();
    if engine.is_conversion_supported(source_dtype, target_dtype, &cpu_device) {
        let cpu_tensor = tensor.to_device(&cpu_device, &pool)?;
        let result = engine.convert(&cpu_tensor, target_dtype)?;
        let final_result = result.to_device(&device, &pool)?;
    }
}
```

### Debugging Tips

```rust
// Enable debug logging
let mut config = ConversionConfig::default();
config.enable_debug_logging = true;

// Check conversion support
let is_supported = engine.is_conversion_supported(source_dtype, target_dtype, &device);
println!("Conversion supported: {}", is_supported);

// Get detailed strategy information
let info = engine.get_optimal_strategy_info(source_dtype, target_dtype, &shape, &device);
println!("Strategy info: {}", info.description());

// Monitor conversion events
let recent_events = engine.get_recent_events(5);
for event in recent_events {
    if !event.success {
        eprintln!("Failed conversion: {} -> {}, error: {:?}", 
                  event.source_dtype, event.target_dtype, event.error_message);
    }
}

// Check memory pool status
let pool_metrics = pool.get_metrics();
println!("Pool allocated: {} bytes", pool_metrics.total_allocated);
println!("Pool deallocated: {} bytes", pool_metrics.total_deallocated);
```

This comprehensive guide covers all aspects of the memory-efficient data conversion system. For additional examples and advanced usage patterns, refer to the test files and API documentation.