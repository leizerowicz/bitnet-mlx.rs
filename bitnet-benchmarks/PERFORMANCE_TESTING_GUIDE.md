# BitNet Performance Testing Guide

This guide provides comprehensive instructions for using the BitNet performance testing and benchmarking infrastructure.

## Overview

The BitNet benchmarking suite provides comprehensive performance comparison tools for evaluating BitNet implementations across different backends (Candle CPU, Candle Metal, MLX) and configurations. The suite includes:

- **Comprehensive Performance Comparisons**: Matrix operations, quantization, BitLinear layers, and activation functions
- **Energy Efficiency Testing**: Power consumption and thermal efficiency analysis
- **Quantization Performance**: Detailed analysis of different quantization schemes (1.58-bit, INT8, INT4, FP16)
- **Regression Testing**: Automated performance degradation detection
- **Visualization and Reporting**: HTML reports with charts and detailed analysis

## Quick Start

### Running Basic Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suites
cargo bench comprehensive_performance_comparison
cargo bench energy_efficiency_comparison
cargo bench quantization_performance
cargo bench regression_performance_tests

# Run with specific features (when MLX support is available)
cargo bench --features mlx
```

### Running the Benchmarking CLI

```bash
# Build the CLI tool
cargo build --release

# Run comprehensive comparison
./target/release/bitnet-benchmarks compare --output results.json

# Run with custom configuration
./target/release/bitnet-benchmarks compare --config custom_config.json --output results.json

# Generate HTML report
./target/release/bitnet-benchmarks report --input results.json --output report.html
```

## Benchmark Suites

### 1. Comprehensive Performance Comparison

**File**: `benches/comprehensive_performance_comparison.rs`

Tests core operations across different backends:

- **Matrix Operations**: Matrix multiplication, addition, element-wise operations
- **Quantization**: BitNet 1.58-bit quantization performance
- **BitLinear Layers**: Forward pass performance with different configurations
- **Activation Functions**: ReLU, GELU, SiLU, Swish performance
- **Memory Efficiency**: Memory usage patterns and allocation efficiency
- **Real-world Workloads**: Transformer layer simulation, batch processing
- **Cross-platform Comparison**: CPU vs Metal vs MLX performance

**Key Metrics**:
- Execution time (milliseconds)
- Throughput (operations per second)
- Memory usage (MB)
- Speedup ratios between backends

### 2. Energy Efficiency Comparison

**File**: `benches/energy_efficiency_comparison.rs`

Analyzes power consumption and thermal efficiency:

- **Power Monitoring**: CPU and GPU power consumption during operations
- **Thermal Efficiency**: Temperature monitoring and thermal throttling detection
- **Energy per Operation**: Joules consumed per matrix multiplication, quantization, etc.
- **Battery Life Impact**: Estimated battery drain for mobile/laptop deployments
- **Efficiency Ratios**: Performance per watt comparisons

**Key Metrics**:
- Power consumption (watts)
- Energy efficiency (operations per joule)
- Thermal efficiency (operations per degree Celsius)
- Battery life impact (hours)

### 3. Quantization Performance

**File**: `benches/quantization_performance.rs`

Detailed analysis of quantization schemes:

- **BitNet 1.58-bit**: Performance and accuracy trade-offs
- **INT8 Quantization**: Symmetric and asymmetric quantization
- **INT4 Quantization**: Ultra-low precision performance
- **FP16 Quantization**: Half-precision floating point
- **Granularity Analysis**: Per-tensor vs per-channel quantization
- **Accuracy vs Speed**: Trade-off analysis between precision and performance

**Key Metrics**:
- Quantization time (milliseconds)
- Dequantization time (milliseconds)
- Memory reduction ratio
- Accuracy preservation (when applicable)
- Throughput improvement

### 4. Regression Performance Tests

**File**: `benches/regression_performance_tests.rs`

Automated performance regression detection:

- **Baseline Management**: Automatic baseline creation and updates
- **Performance Monitoring**: Continuous performance tracking
- **Regression Detection**: Statistical analysis for performance degradation
- **Alert System**: Configurable thresholds for performance alerts
- **Historical Analysis**: Performance trends over time

**Key Features**:
- Automatic baseline establishment
- Configurable regression thresholds (5%, 10%, 15%)
- Statistical significance testing
- Performance trend analysis
- CI/CD integration ready

## Configuration

### Benchmark Configuration

Create a custom configuration file:

```json
{
  "tensor_sizes": [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512],
    [1024, 1024],
    [2048, 2048]
  ],
  "warmup_iterations": 5,
  "measurement_iterations": 10,
  "operations": [
    "matmul",
    "add",
    "multiply",
    "quantize",
    "bitlinear"
  ],
  "devices": [
    "cpu",
    "metal",
    "mlx"
  ],
  "data_types": [
    "f32",
    "f16"
  ],
  "timeout": "30s"
}
```

### Energy Monitoring Configuration

```json
{
  "monitoring_interval": "100ms",
  "power_measurement_duration": "10s",
  "thermal_monitoring": true,
  "battery_monitoring": true,
  "device_specific_monitoring": {
    "apple_silicon": true,
    "intel_cpu": true,
    "nvidia_gpu": false
  }
}
```

### Regression Testing Configuration

```json
{
  "baseline_file": "performance_baselines.json",
  "regression_threshold": 0.05,
  "minimum_samples": 10,
  "confidence_level": 0.95,
  "alert_thresholds": {
    "warning": 0.05,
    "critical": 0.15
  },
  "auto_update_baseline": false
}
```

## Visualization and Reporting

### HTML Report Generation

The visualization module generates comprehensive HTML reports with:

- **Executive Summary**: Key performance metrics and highlights
- **Interactive Charts**: SVG-based performance charts
- **Detailed Tables**: Complete benchmark results
- **Comparison Analysis**: Side-by-side performance comparisons
- **Recommendations**: Automated performance optimization suggestions

### Chart Types

1. **Performance Overview**: Bar charts showing throughput across operations
2. **Speedup Comparison**: Speedup ratios between different backends
3. **Memory Usage**: Memory consumption patterns
4. **Energy Efficiency**: Power consumption and efficiency metrics
5. **Regression Trends**: Historical performance trends

### Export Formats

- **JSON**: Machine-readable results for further analysis
- **CSV**: Spreadsheet-compatible format for data analysis
- **HTML**: Rich interactive reports for presentation
- **PNG/SVG**: Individual charts for documentation

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run Benchmarks
      run: |
        cd bitnet-benchmarks
        cargo bench --no-run
        cargo run --release -- compare --output benchmark_results.json
        
    - name: Generate Report
      run: |
        cargo run --release -- report --input benchmark_results.json --output benchmark_report.html
        
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: |
          benchmark_results.json
          benchmark_report.html
```

### Performance Regression Detection

```yaml
    - name: Check for Regressions
      run: |
        cargo run --release -- regression-check --baseline baseline.json --current benchmark_results.json --threshold 0.05
```

## Best Practices

### 1. Benchmark Environment

- **Consistent Hardware**: Run benchmarks on the same hardware configuration
- **Thermal Management**: Ensure adequate cooling to prevent thermal throttling
- **Background Processes**: Minimize background processes during benchmarking
- **Power Management**: Use consistent power settings (plugged in vs battery)

### 2. Statistical Significance

- **Multiple Runs**: Run benchmarks multiple times for statistical significance
- **Warmup Iterations**: Include sufficient warmup iterations to stabilize performance
- **Outlier Detection**: Use statistical methods to detect and handle outliers
- **Confidence Intervals**: Report confidence intervals for performance measurements

### 3. Reproducibility

- **Version Control**: Track benchmark code and configuration versions
- **Environment Documentation**: Document hardware, OS, and software versions
- **Seed Management**: Use consistent random seeds for reproducible results
- **Dependency Pinning**: Pin dependency versions for consistent results

### 4. Performance Analysis

- **Baseline Establishment**: Establish clear performance baselines
- **Trend Analysis**: Monitor performance trends over time
- **Bottleneck Identification**: Use profiling to identify performance bottlenecks
- **Optimization Validation**: Validate optimizations with comprehensive benchmarks

## Troubleshooting

### Common Issues

1. **MLX Feature Not Available**
   ```bash
   # MLX support is currently disabled due to dependency issues
   # Run benchmarks without MLX feature
   cargo bench --no-default-features --features std
   ```

2. **Metal Backend Issues**
   ```bash
   # Ensure Metal is available on macOS
   # Check Metal device availability
   cargo run --example metal_check
   ```

3. **Memory Issues**
   ```bash
   # Enable memory profiling
   cargo bench --features memory
   ```

4. **Compilation Errors**
   ```bash
   # Clean and rebuild
   cargo clean
   cargo build --release
   ```

### Performance Debugging

1. **Enable Detailed Logging**
   ```bash
   RUST_LOG=debug cargo bench
   ```

2. **Profile Memory Usage**
   ```bash
   cargo bench --features memory
   ```

3. **Check System Resources**
   ```bash
   # Monitor system resources during benchmarks
   htop  # or Activity Monitor on macOS
   ```

## Advanced Usage

### Custom Benchmark Development

Create custom benchmarks by extending the existing framework:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_benchmarks::comparison::{PerformanceComparator, ComparisonConfig};

fn custom_benchmark(c: &mut Criterion) {
    let config = ComparisonConfig {
        tensor_sizes: vec![(1024, 1024)],
        warmup_iterations: 3,
        measurement_iterations: 5,
        operations: vec!["custom_op".to_string()],
        devices: vec!["cpu".to_string()],
        data_types: vec!["f32".to_string()],
        timeout: std::time::Duration::from_secs(30),
    };
    
    let mut comparator = PerformanceComparator::new(config);
    
    c.bench_function("custom_operation", |b| {
        b.iter(|| {
            // Your custom benchmark code here
        });
    });
}

criterion_group!(benches, custom_benchmark);
criterion_main!(benches);
```

### Integration with External Tools

- **Perf Integration**: Use Linux perf for detailed CPU profiling
- **Instruments Integration**: Use Xcode Instruments on macOS for system profiling
- **Memory Profilers**: Integration with Valgrind, AddressSanitizer
- **GPU Profilers**: Integration with Metal Performance Shaders Profiler

## Contributing

### Adding New Benchmarks

1. Create a new benchmark file in `benches/`
2. Follow the existing naming convention
3. Include comprehensive documentation
4. Add appropriate feature flags for optional dependencies
5. Update this guide with new benchmark information

### Improving Existing Benchmarks

1. Maintain backward compatibility with existing results
2. Add new metrics without removing existing ones
3. Update documentation for any changes
4. Validate changes with multiple test runs

## Support

For questions, issues, or contributions:

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share insights
- **Documentation**: Refer to inline code documentation
- **Examples**: Check the `examples/` directory for usage examples

---

*This guide is part of the BitNet Rust implementation project. For more information, see the main project documentation.*