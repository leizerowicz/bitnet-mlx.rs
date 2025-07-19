# BitNet Benchmarks: Performance Comparison Suite

A comprehensive benchmarking suite for comparing the performance of different tensor operation backends in the context of BitNet neural network implementations. Currently supports Candle (CPU/Metal) with planned MLX support.

## Overview

This benchmarking suite provides:

- **Comprehensive Operation Coverage**: Matrix multiplication, element-wise operations, quantization, BitLinear layers, and more
- **Multiple Backend Support**: Candle (CPU/Metal) with MLX support planned
- **Detailed Metrics**: Execution time, throughput, memory usage, and performance recommendations
- **Flexible Configuration**: Customizable tensor sizes, iteration counts, and operation selection
- **Multiple Output Formats**: JSON, CSV, and Markdown reports
- **Command-Line Interface**: Easy-to-use CLI for running benchmarks

## Features

### Supported Operations

- **Matrix Operations**: Matrix multiplication, addition, element-wise multiplication
- **Quantization**: 1.58-bit quantization/dequantization (BitNet-specific)
- **BitLinear Layers**: Complete BitLinear forward pass with quantized weights
- **Memory Operations**: Tensor creation (zeros, ones, random)
- **Activation Functions**: ReLU, GELU, Softmax
- **Tensor Manipulation**: Reshape, transpose, concatenation, splitting
- **Neural Network Layers**: Layer normalization, convolution, embedding

### Backend Comparison

- **Candle CPU**: Cross-platform CPU tensor operations
- **Candle Metal**: GPU-accelerated operations on macOS (when available)
- **MLX**: Apple Silicon optimized operations (planned - currently disabled)

### Performance Metrics

- **Execution Time**: Average time per operation
- **Throughput**: Operations per second
- **Memory Usage**: Estimated memory consumption
- **Speedup Ratios**: Relative performance between backends
- **Recommendations**: Automated suggestions for optimal backend selection

## Installation

### Prerequisites

- Rust 1.70+ with Cargo
- macOS (for Metal support) or Linux/Windows (CPU only)

### Building

```bash
# Clone the repository
git clone <repository-url>
cd bitnet-rust/bitnet-benchmarks

# Build the benchmark suite
cargo build --release

# Build with memory profiling support
cargo build --release --features memory

# Note: MLX and Metal features are currently disabled due to dependency issues
# They will be re-enabled in future releases
```

## Usage

### Command Line Interface

The benchmark suite provides a comprehensive CLI for running performance comparisons:

```bash
# Run complete benchmark suite with default settings
cargo run --release -- compare

# Run quick benchmark (minimal configuration)
cargo run --release -- quick

# Generate default configuration file
cargo run --release -- generate-config

# Run with custom configuration
cargo run --release -- compare --config benchmark_config.json

# Run specific operations only
cargo run --release -- compare --operations "matmul,add,quantize"

# Run with specific tensor sizes
cargo run --release -- compare --sizes "128x128,512x512,1024x1024"

# Export results in specific format
cargo run --release -- compare --format json --output results/

# Analyze existing results
cargo run --release -- analyze --input results/benchmark_results.json --detailed
```

### Programmatic Usage

```rust
use bitnet_benchmarks::{
    ComparisonConfig, PerformanceComparator, BenchmarkRunner
};

// Create custom configuration
let config = ComparisonConfig {
    tensor_sizes: vec![(256, 256), (512, 512)],
    warmup_iterations: 5,
    measurement_iterations: 10,
    operations: vec!["matmul".to_string(), "add".to_string()],
    ..Default::default()
};

// Run benchmarks
let mut comparator = PerformanceComparator::new(config);
let comparisons = comparator.run_comparison()?;

// Export results
let json_results = comparator.export_json()?;
let csv_results = comparator.export_csv();
```

### Criterion Benchmarks

Run detailed Criterion-based benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench mlx_vs_candle

# Generate benchmark report
cargo bench -- --output-format html
```

## Configuration

### Default Configuration

The default configuration includes:

- **Tensor Sizes**: 64x64, 128x128, 256x256, 512x512, 1024x1024, 2048x2048
- **Operations**: matmul, add, multiply, quantize, bitlinear
- **Warmup Iterations**: 5
- **Measurement Iterations**: 10
- **Data Types**: f32, f16
- **Timeout**: 30 seconds per benchmark

### Custom Configuration

Create a JSON configuration file:

```json
{
  "tensor_sizes": [[128, 128], [512, 512]],
  "warmup_iterations": 3,
  "measurement_iterations": 5,
  "operations": ["matmul", "add", "quantize"],
  "devices": ["cpu", "metal"],
  "data_types": ["f32"],
  "timeout": {"secs": 15, "nanos": 0}
}
```

## Output Formats

### JSON Report

Detailed machine-readable results with full metrics:

```json
{
  "config": { ... },
  "measurements": [
    {
      "operation": "matmul",
      "backend": "candle",
      "device": "cpu",
      "tensor_size": [512, 512],
      "execution_time": {"secs": 0, "nanos": 1234567},
      "throughput": 810.5,
      "memory_usage": 1048576,
      "success": true
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### CSV Report

Tabular format for analysis in spreadsheet applications:

```csv
operation,backend,device,tensor_size,data_type,execution_time_ms,throughput,memory_usage,success
matmul,candle,cpu,512x512,f32,1.234,810.5,1048576,true
matmul,candle,metal,512x512,f32,0.567,1763.7,1048576,true
```

### Markdown Summary

Human-readable comparison report:

```markdown
# Performance Comparison Summary

## matmul

| Tensor Size | Baseline | Comparison | Speedup | Recommendation |
|-------------|----------|------------|---------|----------------|
| 512x512 | candle_cpu | candle_metal | 2.18x | Use candle_metal for better performance |

## Overall Recommendations

- Use candle_metal for better performance: 15 cases
- Performance is similar: 3 cases
```

## Performance Analysis

### Interpreting Results

- **Speedup > 1.5x**: Significant performance advantage
- **Speedup 0.8x - 1.5x**: Similar performance
- **Speedup < 0.8x**: Performance disadvantage

### Optimization Recommendations

The benchmark suite automatically provides recommendations based on:

- Relative execution times
- Memory efficiency
- Device capabilities
- Operation characteristics

### Common Patterns

- **Candle Metal**: Good GPU acceleration on macOS for supported operations
- **Candle CPU**: Consistent cross-platform performance, good for smaller operations
- **MLX**: Will provide Apple Silicon optimization when re-enabled

## Development

### Running Tests

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run tests with memory profiling
cargo test --features memory

# Run benchmarks
cargo bench
```

### Adding New Operations

1. Implement the operation in [`src/candle_ops.rs`](src/candle_ops.rs)
2. Add benchmark cases in [`benches/mlx_vs_candle.rs`](benches/mlx_vs_candle.rs)
3. Update the comparison framework in [`src/comparison.rs`](src/comparison.rs)
4. Add integration tests in [`tests/integration_tests.rs`](tests/integration_tests.rs)

### Code Structure

```
bitnet-benchmarks/
├── src/
│   ├── lib.rs              # Library exports
│   ├── main.rs             # CLI entry point
│   ├── candle_ops.rs       # Candle operation implementations
│   ├── comparison.rs       # Performance comparison framework
│   └── runner.rs           # Benchmark runner and CLI
├── benches/
│   ├── mlx_vs_candle.rs    # Criterion benchmarks
│   └── quantization.rs     # Quantization-specific benchmarks
├── tests/
│   └── integration_tests.rs # Integration tests
└── README.md               # This file
```

## Current Limitations

### Temporarily Disabled Features

The following features are currently disabled due to dependency issues and will be re-enabled in future releases:

- **MLX Support**: The `mlx` feature is commented out in [`Cargo.toml`](Cargo.toml)
- **Metal Support**: The `metal` feature is temporarily disabled
- **Training Benchmarks**: The `training` feature is disabled

### Available Features

- **Memory Profiling**: Enable with `--features memory`
- **Standard Benchmarks**: All core Candle operations are available

## Troubleshooting

### Common Issues

1. **Metal not available**: Check macOS version and GPU support
2. **Compilation errors**: Verify Rust version and dependencies
3. **Performance inconsistency**: Ensure system is not under load during benchmarking
4. **MLX not available**: Currently disabled - will be re-enabled in future releases

### Debug Mode

Run with verbose output for debugging:

```bash
cargo run --release -- compare --verbose
```

### Memory Issues

For large tensor benchmarks, monitor system memory:

```bash
# Reduce tensor sizes for memory-constrained systems
cargo run --release -- compare --sizes "64x64,128x128"
```

### Feature-Specific Issues

```bash
# Check available features
cargo build --release --features memory

# Note: MLX and Metal features are currently disabled
# Use CPU-only benchmarks for now
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Guidelines

- Follow Rust naming conventions
- Add comprehensive tests for new operations
- Update documentation for new features
- Ensure cross-platform compatibility where possible

## License

This project is licensed under the same terms as the main BitNet Rust implementation.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) for cross-platform tensor operations
- [MLX](https://github.com/ml-explore/mlx) for Apple Silicon optimization (planned)
- [Criterion](https://github.com/bheisler/criterion.rs) for benchmarking framework