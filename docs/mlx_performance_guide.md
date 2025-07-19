# MLX Performance Comparison Tools Guide

This guide provides comprehensive documentation for the MLX performance comparison tools implemented in BitNet Rust.

## Overview

The MLX performance comparison framework provides a complete suite of tools for analyzing, comparing, and optimizing MLX operations on Apple Silicon devices. The framework includes:

- **Performance Benchmarking**: Comprehensive benchmarking utilities for MLX operations
- **Memory Tracking**: Detailed memory usage analysis and optimization recommendations
- **Metrics Collection**: Real-time performance metrics collection and analysis
- **Report Generation**: Automated report generation with visualizations and insights
- **Advanced Profiling**: Call stack analysis, hotspot detection, and bottleneck identification
- **Device Comparison**: CPU vs GPU performance analysis and device selection guidance
- **Regression Testing**: Automated performance regression detection and baseline management

## Quick Start

### Basic Performance Benchmarking

```rust
use bitnet_core::mlx::performance::{MlxPerformanceBenchmarker, BenchmarkConfig};
use bitnet_core::mlx::BitNetMlxDevice;

// Create a benchmark configuration
let config = BenchmarkConfig::default();
let mut benchmarker = MlxPerformanceBenchmarker::new(config);

// Create devices to test
let cpu_device = BitNetMlxDevice::cpu();
let gpu_device = BitNetMlxDevice::gpu();

// Benchmark matrix multiplication on both devices
let cpu_metrics = benchmarker.benchmark_matmul(&cpu_device)?;
let gpu_metrics = benchmarker.benchmark_matmul(&gpu_device)?;

println!("CPU execution time: {:?}", cpu_metrics.execution_time);
println!("GPU execution time: {:?}", gpu_metrics.execution_time);
println!("GPU speedup: {:.2}x", 
    cpu_metrics.execution_time.as_secs_f64() / gpu_metrics.execution_time.as_secs_f64());
```

### Memory Tracking

```rust
use bitnet_core::mlx::memory_tracker::{MlxMemoryTracker, track_allocation, track_deallocation};
use bitnet_core::mlx::BitNetMlxDevice;

// Create a memory tracker
let mut tracker = MlxMemoryTracker::new();
tracker.set_tracking_enabled(true);

let device = BitNetMlxDevice::gpu();

// Track memory allocation
track_allocation(
    "tensor_1".to_string(),
    1024 * 1024, // 1MB
    &device,
    "matmul".to_string(),
)?;

// Perform operations...

// Track memory deallocation
track_deallocation(
    "tensor_1".to_string(),
    &device,
    "cleanup".to_string(),
)?;

// Get memory usage report
let report = tracker.generate_report();
println!("{}", report);
```

### Device Comparison

```rust
use bitnet_core::mlx::device_comparison::{MlxDeviceComparison, DeviceComparisonConfig};

// Create comparison configuration
let config = DeviceComparisonConfig::default();
let mut comparison = MlxDeviceComparison::new(config);

// Run comprehensive device comparison
let results = comparison.run_comparison()?;

// Export results
let summary = comparison.export_results(&results, "summary")?;
println!("{}", summary);

// Get device recommendations
for recommendation in &results.device_selection_guide.use_case_recommendations {
    println!("For {}: use {}", recommendation.0, recommendation.1);
}
```

## Detailed Usage

### 1. Performance Benchmarking

#### Configuration Options

```rust
use bitnet_core::mlx::performance::BenchmarkConfig;
use std::time::Duration;

let config = BenchmarkConfig {
    warmup_iterations: 5,
    measurement_iterations: 10,
    tensor_sizes: vec![
        vec![512, 512],
        vec![1024, 1024],
        vec![2048, 2048],
    ],
    data_types: vec!["f32".to_string(), "f16".to_string()],
    devices: vec!["cpu".to_string(), "gpu".to_string()],
    timeout: Duration::from_secs(60),
};
```

#### Available Benchmarks

```rust
// Matrix multiplication
let matmul_metrics = benchmarker.benchmark_matmul(&device)?;

// Quantization operations
let quant_metrics = benchmarker.benchmark_quantization(&device)?;

// Element-wise operations
let add_metrics = benchmarker.benchmark_elementwise(&device, "add")?;
let mul_metrics = benchmarker.benchmark_elementwise(&device, "multiply")?;

// Device comparisons
let comparisons = benchmarker.compare_devices("matmul")?;
```

### 2. Memory Analysis

#### Memory Optimization Recommendations

```rust
use bitnet_core::mlx::memory_tracker::MlxMemoryTracker;

let tracker = MlxMemoryTracker::new();
let device = BitNetMlxDevice::gpu();

// Get memory pressure level
let pressure = tracker.get_memory_pressure(&device);
println!("Memory pressure: {:?}", pressure);

// Get optimization recommendations
let optimizations = tracker.generate_optimizations(&device);
for opt in optimizations {
    println!("Optimization: {}", opt.description);
    println!("Potential savings: {} bytes", opt.potential_savings);
    println!("Priority: {:?}", opt.priority);
}
```

#### Memory Event Tracking

```rust
// Track memory transfer between devices
tracker.track_transfer(
    "tensor_1".to_string(),
    1024 * 1024,
    &cpu_device,
    &gpu_device,
    "device_transfer".to_string(),
)?;

// Get memory events
let events = tracker.get_events();
for event in events {
    println!("Event: {:?} - {} bytes at {:?}", 
        event.event_type, event.size_bytes, event.timestamp);
}
```

### 3. Metrics Collection

#### Real-time Metrics

```rust
use bitnet_core::mlx::metrics::{MlxMetricsCollector, MetricsConfig, OperationContext};

let config = MetricsConfig::default();
let mut collector = MlxMetricsCollector::new(config);

// Start collection
collector.start_collection();

// Collect metrics for an operation
let context = OperationContext {
    operation_name: "matmul".to_string(),
    batch_size: 32,
    sequence_length: Some(512),
    model_parameters: Some(1000000),
    precision: "f32".to_string(),
    optimization_level: "O2".to_string(),
    parallel_execution: true,
};

let metrics = collector.collect_operation_metrics(
    "matmul",
    &device,
    execution_time,
    tensor_shapes,
    context,
)?;

// Export metrics
let json_report = collector.export_metrics(Some(ExportFormat::Json))?;
let csv_report = collector.export_metrics(Some(ExportFormat::Csv))?;
```

### 4. Advanced Profiling

#### Profiling Sessions

```rust
use bitnet_core::mlx::profiler::{MlxAdvancedProfiler, ProfilerConfig};

let config = ProfilerConfig::default();
let mut profiler = MlxAdvancedProfiler::new(config);

// Start profiling session
profiler.start_session("performance_analysis".to_string())?;

// Profile specific operations
let (result, metrics) = profiler.profile_operation(
    "matmul",
    &device,
    || {
        // Your MLX operation here
        Ok(())
    },
)?;

// Stop session and get results
let session = profiler.stop_session()?;

// Generate reports
let flame_graph = profiler.generate_flame_graph(&session)?;
let call_tree = profiler.generate_call_tree(&session)?;
```

#### Hotspot Analysis

```rust
// Analyze hotspots from profiling session
for hotspot in &session.hotspots {
    println!("Function: {}", hotspot.function_name);
    println!("Total time: {:?}", hotspot.total_time);
    println!("Percentage: {:.2}%", hotspot.percentage_of_total);
    println!("Call count: {}", hotspot.call_count);
    println!("Optimization potential: {:.1}/100", hotspot.optimization_potential.score);
}
```

### 5. Report Generation

#### Comprehensive Reports

```rust
use bitnet_core::mlx::reports::PerformanceReportGenerator;

let generator = PerformanceReportGenerator::new();

// Generate comprehensive report
let report = generator.generate_comprehensive_report(
    &metrics,
    &comparisons,
    &memory_events,
    &optimizations,
)?;

// Generate HTML report
let html_report = generator.generate_html_report(&report)?;

// Save to file
std::fs::write("performance_report.html", html_report)?;
```

#### Custom Report Sections

```rust
// Generate specific report sections
let executive_summary = generator.generate_executive_summary(&metrics, &comparisons)?;
let performance_analysis = generator.generate_performance_analysis(&metrics)?;
let memory_analysis = generator.generate_memory_analysis(&memory_events, &optimizations)?;
```

### 6. Regression Testing

#### Creating Baselines

```rust
use bitnet_core::mlx::regression_testing::{MlxRegressionTester, RegressionTestConfig};

let config = RegressionTestConfig::default();
let mut tester = MlxRegressionTester::new(config);

// Create a new baseline
let baseline = tester.create_baseline(
    "v1.0.0".to_string(),
    "Initial performance baseline".to_string(),
)?;

println!("Created baseline: {}", baseline.baseline_id);
```

#### Running Regression Tests

```rust
// Run regression tests against baseline
let test_results = tester.run_regression_tests("v1.0.0")?;

// Check for regressions
match test_results.overall_status {
    TestStatus::AllPass => println!("All tests passed!"),
    TestStatus::HasRegressions => {
        println!("Regressions detected:");
        for regression in &test_results.regressions_detected {
            println!("- {} on {}: {:.1}% degradation", 
                regression.operation, 
                regression.device, 
                regression.performance_degradation);
        }
    },
    TestStatus::HasCriticalRegressions => {
        println!("CRITICAL regressions detected! Immediate attention required.");
    },
    _ => println!("Test status: {:?}", test_results.overall_status),
}

// Generate regression report
let report = tester.generate_report(&test_results)?;
println!("{}", report);
```

## Best Practices

### 1. Performance Optimization

- **Use appropriate tensor sizes**: Start with smaller tensors for development, scale up for production
- **Choose the right device**: Use GPU for large matrix operations, CPU for small tensors and development
- **Optimize batch sizes**: Larger batches generally improve GPU utilization
- **Monitor memory usage**: Use memory tracking to identify optimization opportunities

### 2. Memory Management

- **Enable memory tracking**: Always track memory usage in development and testing
- **Implement tensor reuse**: Reuse tensors when possible to reduce allocation overhead
- **Use memory pooling**: Implement memory pools for frequently allocated tensor sizes
- **Monitor fragmentation**: Watch for memory fragmentation and implement defragmentation strategies

### 3. Benchmarking

- **Use consistent environments**: Run benchmarks in consistent system conditions
- **Include warmup iterations**: Always include warmup iterations to account for cold start effects
- **Test multiple scenarios**: Test different tensor sizes, data types, and batch sizes
- **Document test conditions**: Record system state, temperature, and other relevant conditions

### 4. Regression Testing

- **Establish stable baselines**: Create baselines from stable, well-tested code
- **Run regular tests**: Integrate regression testing into CI/CD pipelines
- **Set appropriate thresholds**: Configure regression thresholds based on acceptable performance variance
- **Investigate regressions promptly**: Address performance regressions as soon as they're detected

## Integration Examples

### CI/CD Integration

```bash
#!/bin/bash
# Performance regression test script for CI/CD

# Run regression tests
cargo run --bin regression_test -- \
    --baseline "stable_v1.0" \
    --threshold 10.0 \
    --output-format json \
    --output-file regression_results.json

# Check exit code
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    exit 1
fi

echo "Performance tests passed"
```

### Automated Monitoring

```rust
use std::time::Duration;
use tokio::time::interval;

// Automated performance monitoring
async fn monitor_performance() -> Result<()> {
    let mut interval = interval(Duration::from_hours(1));
    let mut benchmarker = MlxPerformanceBenchmarker::new(BenchmarkConfig::default());
    
    loop {
        interval.tick().await;
        
        // Run quick performance check
        let device = BitNetMlxDevice::gpu();
        let metrics = benchmarker.benchmark_matmul(&device)?;
        
        // Check against thresholds
        if metrics.execution_time > Duration::from_millis(100) {
            // Send alert
            send_performance_alert(&metrics).await?;
        }
        
        // Log metrics
        log::info!("Performance check: {:?}", metrics.execution_time);
    }
}
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Enable memory tracking to identify allocation patterns
   - Check for memory leaks using the leak detection features
   - Implement tensor reuse and memory pooling

2. **Poor GPU Performance**
   - Verify GPU is available and properly configured
   - Check tensor sizes (GPU performs better with larger tensors)
   - Ensure proper batch sizes for GPU utilization

3. **Inconsistent Benchmarks**
   - Increase warmup iterations
   - Check system load and thermal conditions
   - Use longer measurement periods for stable results

4. **Regression Test Failures**
   - Verify baseline validity
   - Check for system changes (OS updates, hardware changes)
   - Review recent code changes for performance impacts

### Performance Tuning Tips

1. **Tensor Size Optimization**
   ```rust
   // Test different tensor sizes to find optimal performance
   let sizes = vec![
       vec![256, 256],   // Small
       vec![512, 512],   // Medium
       vec![1024, 1024], // Large
       vec![2048, 2048], // Very large
   ];
   
   for size in sizes {
       let metrics = benchmarker.benchmark_matmul_with_size(&device, &size)?;
       println!("Size {:?}: {:.2} GFLOPS", size, calculate_gflops(&metrics, &size));
   }
   ```

2. **Data Type Selection**
   ```rust
   // Compare f32 vs f16 performance
   let f32_metrics = benchmarker.benchmark_with_dtype(&device, "f32")?;
   let f16_metrics = benchmarker.benchmark_with_dtype(&device, "f16")?;
   
   println!("f32 performance: {:.2} ops/sec", f32_metrics.throughput);
   println!("f16 performance: {:.2} ops/sec", f16_metrics.throughput);
   println!("f16 speedup: {:.2}x", f16_metrics.throughput / f32_metrics.throughput);
   ```

3. **Memory Layout Optimization**
   ```rust
   // Test different memory layouts
   let contiguous_metrics = benchmarker.benchmark_contiguous(&device)?;
   let strided_metrics = benchmarker.benchmark_strided(&device)?;
   
   if contiguous_metrics.execution_time < strided_metrics.execution_time {
       println!("Use contiguous memory layout for better performance");
   }
   ```

## API Reference

For detailed API documentation, see the individual module documentation:

- [`performance`](./performance.rs) - Performance benchmarking utilities
- [`memory_tracker`](./memory_tracker.rs) - Memory usage tracking and analysis
- [`metrics`](./metrics.rs) - Real-time metrics collection
- [`reports`](./reports.rs) - Report generation and visualization
- [`profiler`](./profiler.rs) - Advanced profiling capabilities
- [`device_comparison`](./device_comparison.rs) - Device performance comparison
- [`regression_testing`](./regression_testing.rs) - Performance regression testing

## Contributing

When contributing to the MLX performance tools:

1. Add comprehensive tests for new features
2. Update documentation and examples
3. Run performance regression tests
4. Follow the established code style and patterns
5. Include benchmarks for performance-critical changes

## License

This project is licensed under the same terms as the BitNet Rust implementation.