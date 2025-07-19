# MLX Optimization Utilities

This module provides comprehensive MLX-specific optimization utilities for BitNet on Apple Silicon devices. These utilities are designed to maximize performance by leveraging Apple's MLX framework capabilities.

## Features

### 🚀 Core Optimization Components

- **Memory Optimization** (`optimization.rs`): Intelligent memory pooling and allocation strategies
- **Performance Profiling** (`optimization.rs`): Detailed timing analysis and performance monitoring
- **Kernel Fusion** (`optimization.rs`): Automatic operation fusion for reduced overhead
- **Tensor Caching** (`optimization.rs`): Smart caching with TTL and LRU eviction
- **Auto-Tuning** (`optimization.rs`): Automatic parameter optimization through benchmarking
- **Batch Processing** (`optimization.rs`): Optimal batch size detection and processing
- **Computation Graph** (`graph.rs`): Advanced graph analysis and optimization

### 📊 Key Benefits

- **Reduced Memory Allocation Overhead**: Memory pooling reduces allocation/deallocation costs
- **Improved Cache Locality**: Optimized memory layouts improve cache performance
- **Lower Kernel Launch Overhead**: Operation fusion reduces GPU kernel launches
- **Automatic Performance Tuning**: Auto-tuning finds optimal configurations automatically
- **Intelligent Resource Management**: Smart caching and batch processing optimize resource usage

## Quick Start

### Basic Usage

```rust
use bitnet_core::mlx::{
    MlxMemoryOptimizer, MlxProfiler, MlxKernelFusion,
    MlxTensorCache, MlxAutoTuner, MlxBatchOptimizer,
    GraphBuilder
};

// Memory optimization
let mut memory_optimizer = MlxMemoryOptimizer::new(100);
let device = BitNetMlxDevice::cpu();
let tensor = memory_optimizer.get_or_create_tensor(
    &[64, 128], 
    mlx_rs::Dtype::Float32, 
    &device
)?;

// Performance profiling
let mut profiler = MlxProfiler::new();
profiler.start_operation("my_operation");
// ... perform operation ...
let duration = profiler.end_operation().unwrap();

// Kernel fusion
let fusion = MlxKernelFusion::new();
let operations = vec!["add".to_string(), "multiply".to_string()];
if let Some(result) = fusion.try_fuse(&operations, &arrays) {
    // Use fused result
}

// Computation graph optimization
let mut builder = GraphBuilder::new();
let input = builder.input("input", vec![32, 784], "f32", "cpu");
let weights = builder.input("weights", vec![784, 256], "f32", "cpu");
let matmul = builder.matmul(input, weights, "cpu")?;
let graph = builder.build();

let execution_plan = graph.generate_execution_plan()?;
println!("Optimization opportunities: {}", execution_plan.fusion_opportunities.len());
```

### Advanced Workflow

```rust
fn optimized_neural_network() -> Result<(), Box<dyn std::error::Error>> {
    // Set up optimization stack
    let mut memory_optimizer = MlxMemoryOptimizer::new(50);
    let mut profiler = MlxProfiler::new();
    let mut cache = MlxTensorCache::new(20, Duration::from_secs(300));
    let fusion = MlxKernelFusion::new();
    let mut auto_tuner = MlxAutoTuner::new();
    
    // Build and optimize computation graph
    let mut builder = GraphBuilder::new();
    // ... build graph ...
    let graph = builder.build();
    let plan = graph.generate_execution_plan()?;
    
    // Execute with all optimizations enabled
    profiler.start_operation("inference");
    
    // Use cached weights if available
    let weights = cache.get("model_weights").unwrap_or_else(|| {
        let w = memory_optimizer.get_or_create_tensor(/* ... */).unwrap();
        cache.put("model_weights".to_string(), w.clone());
        w
    });
    
    // Apply fusion optimizations
    for opportunity in &plan.fusion_opportunities {
        // Apply fusion based on opportunity.pattern
    }
    
    let inference_time = profiler.end_operation().unwrap();
    memory_optimizer.return_to_pool(weights, &device);
    
    Ok(())
}
```

## Module Structure

```
mlx/
├── mod.rs              # Module exports and feature gates
├── device.rs           # MLX device management
├── tensor.rs           # MLX tensor operations
├── operations.rs       # MLX-accelerated BitNet operations
├── optimization.rs     # Core optimization utilities
├── graph.rs           # Computation graph optimization
├── tests.rs           # Basic MLX functionality tests
├── optimization_tests.rs # Comprehensive optimization tests
└── README.md          # This file
```

## Optimization Components

### MlxMemoryOptimizer

Provides intelligent memory pooling to reduce allocation overhead:

- **Tensor Pooling**: Reuses allocated tensors based on shape and dtype
- **Statistics Tracking**: Monitors pool efficiency and memory usage
- **Batch Layout Optimization**: Optimizes memory layout for batch operations
- **Configurable Pool Size**: Adjustable limits to balance memory and performance

### MlxProfiler

Offers detailed performance analysis:

- **Operation Timing**: Precise timing of individual operations
- **Statistical Analysis**: Average, min, max execution times
- **Multi-operation Support**: Track multiple operation types simultaneously
- **Low Overhead**: Minimal impact on actual operation performance

### MlxKernelFusion

Automatically fuses compatible operations:

- **Pattern Detection**: Identifies fusable operation sequences
- **Built-in Patterns**: MatMul+Add, Add+Multiply, Quantize+Dequantize
- **Custom Patterns**: Support for user-defined fusion patterns
- **Performance Estimation**: Estimates speedup from fusion

### MlxTensorCache

Smart caching for frequently used tensors:

- **TTL Support**: Time-based expiration of cached entries
- **LRU Eviction**: Least Recently Used eviction policy
- **Size Limits**: Configurable maximum cache size
- **Automatic Cleanup**: Periodic cleanup of expired entries

### MlxAutoTuner

Automatic parameter optimization:

- **Configuration Benchmarking**: Tests multiple parameter combinations
- **Performance Tracking**: Stores and analyzes benchmark results
- **Optimal Selection**: Automatically chooses best-performing configuration
- **Persistent Results**: Remembers optimal settings across runs

### MlxBatchOptimizer

Optimal batch processing:

- **Batch Size Detection**: Finds optimal batch sizes automatically
- **Throughput Optimization**: Maximizes operations per second
- **Memory-Aware**: Respects memory constraints during optimization
- **Automatic Batching**: Processes data in optimal batch sizes

### MlxComputationGraph

Advanced graph analysis and optimization:

- **Graph Construction**: Easy-to-use builder pattern
- **Topological Sorting**: Optimal execution order determination
- **Fusion Detection**: Automatic detection of optimization opportunities
- **Memory Layout Optimization**: Optimal memory allocation and reuse
- **Execution Planning**: Complete execution plan generation

## Examples

The module includes comprehensive examples:

- `mlx_operations_demo.rs`: Basic MLX operation wrappers
- `mlx_optimization_demo.rs`: Complete optimization utilities showcase
- `mlx_graph_optimization_demo.rs`: Computation graph optimization examples

Run examples with:

```bash
# Basic operations
cargo run --example mlx_operations_demo --features mlx

# Full optimization suite
cargo run --example mlx_optimization_demo --features mlx

# Graph optimization
cargo run --example mlx_graph_optimization_demo --features mlx
```

## Testing

Comprehensive test suite covering all optimization utilities:

```bash
# Run all MLX tests
cargo test --features mlx mlx

# Run optimization-specific tests
cargo test --features mlx optimization

# Run graph optimization tests
cargo test --features mlx graph
```

## Performance Considerations

### Memory Optimization

- **Pool Size**: Larger pools reduce allocation overhead but use more memory
- **Tensor Reuse**: Higher reuse rates improve performance significantly
- **Memory Layout**: Optimized layouts improve cache performance

### Kernel Fusion

- **Fusion Opportunities**: More fusions generally mean better performance
- **Pattern Complexity**: Simple patterns fuse more reliably
- **Memory Bandwidth**: Fusion reduces memory traffic between operations

### Caching Strategy

- **Cache Size**: Balance between hit rate and memory usage
- **TTL Settings**: Longer TTL improves hit rates but may waste memory
- **Access Patterns**: Cache works best with repeated access patterns

### Auto-Tuning

- **Benchmark Quality**: More thorough benchmarking gives better results
- **Configuration Space**: Larger search spaces may find better optima
- **Stability**: Results should be stable across multiple runs

## Best Practices

1. **Start with Profiling**: Always profile before optimizing
2. **Use Memory Pooling**: Enable for any frequent tensor operations
3. **Enable Fusion**: Use for compatible operation sequences
4. **Cache Strategically**: Cache frequently accessed, expensive-to-compute tensors
5. **Auto-tune Periodically**: Re-run auto-tuning when workloads change
6. **Monitor Statistics**: Track optimization effectiveness over time

## Troubleshooting

### Common Issues

- **Low Pool Efficiency**: Increase pool size or check tensor size distribution
- **Cache Misses**: Adjust TTL or cache size based on usage patterns
- **Fusion Failures**: Verify operation compatibility and tensor shapes
- **Memory Pressure**: Reduce batch sizes or increase memory thresholds

### Performance Debugging

- Use `MlxProfiler` to identify bottlenecks
- Check fusion opportunities in computation graphs
- Monitor memory allocation patterns with `MlxMemoryOptimizer`
- Validate auto-tuned configurations periodically

## Feature Gates

All optimization utilities are gated behind the `mlx` feature:

```toml
[dependencies]
bitnet-core = { version = "0.2.1", features = ["mlx"] }
```

When the `mlx` feature is disabled, stub implementations are provided that return appropriate errors.

## Documentation

For detailed usage examples and API documentation, see:

- [MLX Optimization Guide](../../docs/mlx_optimization_guide.md)
- [API Documentation](https://docs.rs/bitnet-core)
- [Examples](../examples/)

## Contributing

When contributing to MLX optimization utilities:

1. Ensure all new features are properly tested
2. Add comprehensive documentation and examples
3. Consider performance implications of changes
4. Test on actual Apple Silicon hardware when possible
5. Update this README and the optimization guide

## License

This module is part of BitNet-Rust and is licensed under the same terms as the main project.