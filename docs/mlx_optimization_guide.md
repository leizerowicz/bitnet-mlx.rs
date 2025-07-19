# MLX Optimization Guide

This guide covers the comprehensive MLX optimization utilities available in BitNet-Rust, designed to maximize performance on Apple Silicon devices.

## Overview

The MLX optimization utilities provide advanced performance optimization capabilities specifically designed for Apple's MLX framework. These utilities include memory management, performance profiling, kernel fusion, tensor caching, auto-tuning, batch processing optimization, and computation graph analysis.

## Table of Contents

1. [Memory Optimization](#memory-optimization)
2. [Performance Profiling](#performance-profiling)
3. [Kernel Fusion](#kernel-fusion)
4. [Tensor Caching](#tensor-caching)
5. [Auto-Tuning](#auto-tuning)
6. [Batch Processing Optimization](#batch-processing-optimization)
7. [Computation Graph Optimization](#computation-graph-optimization)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

## Memory Optimization

### MlxMemoryOptimizer

The `MlxMemoryOptimizer` provides intelligent memory pooling and allocation strategies to minimize memory allocation overhead and improve cache locality.

#### Features

- **Memory Pooling**: Reuses allocated tensors to reduce allocation overhead
- **Statistics Tracking**: Monitors pool hits, misses, and memory usage
- **Configurable Pool Size**: Adjustable maximum pool size to balance memory usage and performance
- **Batch Layout Optimization**: Optimizes memory layout for batch operations

#### Usage

```rust
use bitnet_core::mlx::{MlxMemoryOptimizer, BitNetMlxDevice};

// Create memory optimizer with pool size of 100 tensors
let mut optimizer = MlxMemoryOptimizer::new(100);
let device = BitNetMlxDevice::cpu();

// Get tensor from pool (or create new if pool is empty)
let tensor = optimizer.get_or_create_tensor(
    &[64, 128], 
    mlx_rs::Dtype::Float32, 
    &device
)?;

// Use tensor...

// Return tensor to pool for reuse
optimizer.return_to_pool(tensor, &device);

// Check statistics
let stats = optimizer.get_stats();
println!("Pool efficiency: {:.1}%", 
    100.0 * stats.pool_hits as f64 / 
    (stats.pool_hits + stats.pool_misses) as f64);
```

#### Memory Statistics

The optimizer tracks detailed statistics:

- `total_allocations`: Total number of tensor allocations
- `total_deallocations`: Total number of tensor deallocations
- `pool_hits`: Number of successful pool retrievals
- `pool_misses`: Number of times new allocation was required
- `peak_memory_usage`: Maximum memory usage observed
- `current_memory_usage`: Current memory usage

## Performance Profiling

### MlxProfiler

The `MlxProfiler` provides detailed timing analysis for MLX operations, helping identify performance bottlenecks.

#### Features

- **Operation Timing**: Precise timing of individual operations
- **Statistical Analysis**: Average, minimum, and maximum execution times
- **Multiple Operation Support**: Track multiple different operations
- **Low Overhead**: Minimal impact on actual operation performance

#### Usage

```rust
use bitnet_core::mlx::MlxProfiler;
use std::time::Duration;

let mut profiler = MlxProfiler::new();

// Profile an operation
profiler.start_operation("matrix_multiply");
// ... perform matrix multiplication ...
let duration = profiler.end_operation().unwrap();

// Get statistics
let avg_time = profiler.get_average_time("matrix_multiply");
let all_stats = profiler.get_all_stats();

for (op_name, (avg, range, count)) in all_stats {
    println!("{}: avg {:?}, range {:?}, {} runs", 
        op_name, avg, range, count);
}
```

## Kernel Fusion

### MlxKernelFusion

The `MlxKernelFusion` optimizer automatically detects and fuses compatible operations to reduce kernel launch overhead and improve memory bandwidth utilization.

#### Supported Fusion Patterns

1. **MatMul + Add Bias**: Fuses matrix multiplication with bias addition
2. **Add + Multiply**: Fuses element-wise addition and multiplication
3. **Quantize + Dequantize**: Eliminates redundant quantization pairs
4. **Activation Chains**: Fuses sequences of activation functions

#### Usage

```rust
use bitnet_core::mlx::MlxKernelFusion;
use mlx_rs::Array;

let fusion = MlxKernelFusion::new();

// Define operation sequence
let operations = vec!["add".to_string(), "multiply".to_string()];
let arrays = vec![
    &Array::from_slice(&[1.0, 2.0], &[2]),
    &Array::from_slice(&[3.0, 4.0], &[2]),
    &Array::from_slice(&[2.0, 2.0], &[2]),
];

// Try to fuse operations
if let Some(result) = fusion.try_fuse(&operations, &arrays) {
    match result {
        Ok(fused_result) => {
            println!("Fusion successful!");
            // Use fused result...
        }
        Err(e) => println!("Fusion failed: {}", e),
    }
}
```

#### Custom Fusion Patterns

You can add custom fusion patterns:

```rust
use bitnet_core::mlx::{FusionPattern, MlxKernelFusion};

let mut fusion = MlxKernelFusion::new();

let custom_pattern = FusionPattern {
    name: "custom_relu_add".to_string(),
    operations: vec!["relu".to_string(), "add".to_string()],
    fused_implementation: |arrays| {
        // Custom fusion implementation
        // ...
        Ok(result_array)
    },
};

fusion.add_pattern(custom_pattern);
```

## Tensor Caching

### MlxTensorCache

The `MlxTensorCache` provides intelligent caching of frequently used tensors with automatic expiration and size management.

#### Features

- **LRU Eviction**: Least Recently Used eviction policy
- **TTL Support**: Time-to-live based expiration
- **Size Limits**: Configurable maximum cache size
- **Automatic Cleanup**: Periodic cleanup of expired entries

#### Usage

```rust
use bitnet_core::mlx::MlxTensorCache;
use std::time::Duration;
use mlx_rs::Array;

// Create cache with max 100 entries, 1-hour TTL
let mut cache = MlxTensorCache::new(100, Duration::from_secs(3600));

// Cache a tensor
let tensor = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
cache.put("model_weights_layer1".to_string(), tensor);

// Retrieve from cache
if let Some(cached_tensor) = cache.get("model_weights_layer1") {
    println!("Cache hit!");
    // Use cached tensor...
} else {
    println!("Cache miss - need to compute/load tensor");
}

// Check cache statistics
let (current_size, max_size) = cache.stats();
println!("Cache utilization: {}/{}", current_size, max_size);
```

## Auto-Tuning

### MlxAutoTuner

The `MlxAutoTuner` automatically finds optimal configurations for operations by benchmarking different parameter combinations.

#### Features

- **Configuration Benchmarking**: Tests multiple configurations automatically
- **Performance Tracking**: Stores benchmark results for analysis
- **Optimal Selection**: Automatically selects best-performing configuration
- **Persistent Results**: Remembers optimal configurations across runs

#### Usage

```rust
use bitnet_core::mlx::MlxAutoTuner;
use std::time::Duration;

let mut tuner = MlxAutoTuner::new();

// Define configurations to test
let configs = vec![
    "batch_size_16".to_string(),
    "batch_size_32".to_string(),
    "batch_size_64".to_string(),
];

// Benchmark function
let benchmark_fn = |config: &str| -> Result<Duration, anyhow::Error> {
    match config {
        "batch_size_16" => {
            // Benchmark with batch size 16
            Ok(Duration::from_millis(100))
        }
        "batch_size_32" => {
            // Benchmark with batch size 32
            Ok(Duration::from_millis(80))
        }
        "batch_size_64" => {
            // Benchmark with batch size 64
            Ok(Duration::from_millis(120))
        }
        _ => Err(anyhow::anyhow!("Unknown config")),
    }
};

// Find optimal configuration
let optimal = tuner.benchmark_operation("matmul", configs, benchmark_fn)?;
println!("Optimal configuration: {}", optimal);

// Use optimal configuration in future operations
if let Some(config) = tuner.get_optimal_config("matmul") {
    println!("Using optimal config: {}", config);
}
```

## Batch Processing Optimization

### MlxBatchOptimizer

The `MlxBatchOptimizer` finds optimal batch sizes for operations and provides efficient batch processing utilities.

#### Features

- **Optimal Batch Size Detection**: Automatically finds best batch size
- **Throughput Optimization**: Maximizes operations per second
- **Memory-Aware Processing**: Respects memory constraints
- **Automatic Batching**: Processes data in optimal batch sizes

#### Usage

```rust
use bitnet_core::mlx::MlxBatchOptimizer;
use std::time::Duration;

let mut optimizer = MlxBatchOptimizer::new(1024 * 1024); // 1MB threshold

// Find optimal batch size
let benchmark_fn = |batch_size: usize| -> Result<Duration, anyhow::Error> {
    // Simulate processing time based on batch size
    let time_per_item = Duration::from_nanos(1000);
    Ok(time_per_item * batch_size as u32)
};

let optimal_size = optimizer.find_optimal_batch_size(
    "vector_operations", 
    128, 
    benchmark_fn
)?;

println!("Optimal batch size: {}", optimal_size);

// Process data in optimal batches
let data: Vec<i32> = (1..=1000).collect();
let process_fn = |batch: &[i32]| -> Result<Vec<i32>, anyhow::Error> {
    Ok(batch.iter().map(|x| x * 2).collect())
};

let results = optimizer.process_in_batches(
    "vector_operations", 
    data, 
    process_fn
)?;

println!("Processed {} items", results.len());
```

## Computation Graph Optimization

### MlxComputationGraph and GraphBuilder

The computation graph optimization system provides advanced analysis and optimization of neural network computation graphs.

#### Features

- **Graph Construction**: Easy-to-use builder pattern for graph creation
- **Topological Sorting**: Optimal execution order determination
- **Fusion Detection**: Automatic detection of fusion opportunities
- **Memory Layout Optimization**: Optimal memory allocation and reuse
- **Execution Planning**: Complete execution plan generation

#### Graph Construction

```rust
use bitnet_core::mlx::{GraphBuilder, Operation};

let mut builder = GraphBuilder::new();

// Build a simple neural network
let input = builder.input("input", vec![32, 784], "f32", "cpu");
let weights = builder.input("weights", vec![784, 256], "f32", "cpu");
let bias = builder.input("bias", vec![32, 256], "f32", "cpu");

// Forward pass
let matmul = builder.matmul(input, weights, "cpu")?;
let hidden = builder.add(matmul, bias, "cpu")?;
let quantized = builder.quantize(hidden, 0.1, "cpu")?;
let output = builder.output(quantized, "predictions")?;

let graph = builder.build();
```

#### Graph Analysis

```rust
// Analyze execution order
let execution_order = graph.topological_sort()?;
println!("Execution steps: {}", execution_order.len());

// Find optimization opportunities
let fusion_opportunities = graph.find_fusion_opportunities();
for opportunity in &fusion_opportunities {
    println!("Fusion: {:?}, speedup: {:.2}x", 
        opportunity.pattern, opportunity.estimated_speedup);
}

// Optimize memory layout
let memory_plan = graph.optimize_memory_layout();
println!("Memory groups: {}", memory_plan.memory_groups.len());

// Generate complete execution plan
let plan = graph.generate_execution_plan()?;
println!("Estimated memory: {} bytes", plan.estimated_memory_usage);
println!("Estimated time: {:.6}s", plan.estimated_execution_time);
```

## Examples

### Complete Optimization Workflow

```rust
use bitnet_core::mlx::*;
use std::time::Duration;

fn optimized_inference() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Set up optimization utilities
    let mut memory_optimizer = MlxMemoryOptimizer::new(50);
    let mut profiler = MlxProfiler::new();
    let mut cache = MlxTensorCache::new(20, Duration::from_secs(300));
    let fusion = MlxKernelFusion::new();
    
    // 2. Build computation graph
    let mut builder = GraphBuilder::new();
    let input = builder.input("input", vec![1, 784], "f32", "cpu");
    let weights = builder.input("weights", vec![784, 10], "f32", "cpu");
    let matmul = builder.matmul(input, weights, "cpu")?;
    let output = builder.output(matmul, "logits")?;
    let graph = builder.build();
    
    // 3. Optimize execution plan
    let plan = graph.generate_execution_plan()?;
    println!("Optimization plan ready:");
    println!("  {} fusion opportunities", plan.fusion_opportunities.len());
    println!("  {} memory groups", plan.memory_plan.memory_groups.len());
    
    // 4. Execute with optimizations
    profiler.start_operation("inference");
    
    // Check cache first
    let weights_key = "model_weights";
    let weights_tensor = if let Some(cached) = cache.get(weights_key) {
        cached
    } else {
        // Load weights and cache them
        let weights = memory_optimizer.get_or_create_tensor(
            &[784, 10], 
            mlx_rs::Dtype::Float32, 
            &BitNetMlxDevice::cpu()
        )?;
        cache.put(weights_key.to_string(), weights.clone());
        weights
    };
    
    // Perform inference...
    
    let inference_time = profiler.end_operation().unwrap();
    println!("Inference completed in {:?}", inference_time);
    
    // Return tensor to pool
    memory_optimizer.return_to_pool(weights_tensor, &BitNetMlxDevice::cpu());
    
    Ok(())
}
```

## Best Practices

### Memory Management

1. **Use Memory Pooling**: Always use `MlxMemoryOptimizer` for frequent tensor allocations
2. **Return Tensors**: Always return tensors to the pool when done
3. **Monitor Statistics**: Regularly check pool efficiency and adjust pool size
4. **Batch Operations**: Group similar operations to improve memory locality

### Performance Optimization

1. **Profile First**: Use `MlxProfiler` to identify bottlenecks before optimizing
2. **Enable Fusion**: Use `MlxKernelFusion` for compatible operation sequences
3. **Cache Frequently Used Data**: Use `MlxTensorCache` for model weights and constants
4. **Auto-tune Parameters**: Use `MlxAutoTuner` to find optimal configurations

### Graph Optimization

1. **Build Complete Graphs**: Construct the entire computation graph before optimization
2. **Analyze Fusion Opportunities**: Review and implement suggested fusions
3. **Optimize Memory Layout**: Use memory layout optimization for large models
4. **Plan Execution**: Generate and review execution plans before deployment

### Batch Processing

1. **Find Optimal Batch Sizes**: Use `MlxBatchOptimizer` to determine best batch sizes
2. **Respect Memory Limits**: Set appropriate memory thresholds
3. **Process in Batches**: Use automatic batching for large datasets
4. **Monitor Throughput**: Track operations per second to validate optimizations

### Error Handling

1. **Handle Fusion Failures**: Not all operations can be fused - have fallbacks
2. **Check Cache Misses**: Handle cache misses gracefully
3. **Validate Configurations**: Ensure auto-tuned configurations are valid
4. **Monitor Memory Usage**: Watch for memory pressure and adjust accordingly

## Troubleshooting

### Common Issues

1. **Low Pool Efficiency**: Increase pool size or check tensor size distribution
2. **Cache Misses**: Adjust TTL or cache size based on usage patterns
3. **Fusion Failures**: Verify operation compatibility and array shapes
4. **Memory Pressure**: Reduce batch sizes or increase memory thresholds

### Performance Debugging

1. Use profiler to identify slow operations
2. Check fusion opportunities for missed optimizations
3. Monitor memory allocation patterns
4. Validate auto-tuned configurations periodically

### Monitoring

1. Track pool hit rates over time
2. Monitor cache utilization
3. Log fusion success rates
4. Record performance improvements from optimizations

## Conclusion

The MLX optimization utilities provide a comprehensive suite of tools for maximizing performance on Apple Silicon devices. By combining memory optimization, performance profiling, kernel fusion, tensor caching, auto-tuning, batch processing optimization, and computation graph analysis, you can achieve significant performance improvements in your BitNet applications.

For more examples and detailed API documentation, see the examples in the `bitnet-core/examples/` directory.