# BitNet Calibration System

A comprehensive calibration infrastructure for BitNet quantization optimization, providing advanced dataset processing, statistics collection, and quantization parameter optimization.

## Overview

The calibration system enables precise quantization by analyzing activation patterns from representative datasets. It provides:

- **Dataset Loading & Preprocessing**: Efficient tensor loading with validation
- **Streaming Support**: Handle datasets larger than available memory
- **Representative Sampling**: Multiple strategies for optimal sample selection
- **Statistics Collection**: Comprehensive activation analysis per layer
- **Histogram Analysis**: Advanced quantization parameter optimization
- **Persistence Layer**: Save/load calibration results for reuse
- **Performance Monitoring**: Detailed metrics and monitoring

## Quick Start

```rust
use bitnet_quant::calibration::{
    CalibrationDataset, CalibrationConfigBuilder,
    SamplingStrategy, OptimizationObjective
};

// Configure calibration
let config = CalibrationConfigBuilder::new()
    .batch_size(32)
    .max_samples(1000)
    .sampling_strategy(SamplingStrategy::Stratified)
    .build()?;

// Create dataset and add layer data
let mut dataset = CalibrationDataset::new(config)?;
dataset.add_layer_data("layer1", vec![tensor1, tensor2])?;

// Process and get results
let results = dataset.process().await?;

// Use quantization parameters
for (layer, params) in &results.quantization_parameters {
    println!("Layer {}: scale={:.6}, zero_point={}", 
             layer, params.scale, params.zero_point);
}
```

## Architecture

### Core Components

#### 1. CalibrationDataset
Main interface for calibration operations:
- **Data Management**: Load and organize tensors by layer
- **Processing Pipeline**: Coordinate sampling, statistics, and optimization
- **State Management**: Track processing state and resources
- **Error Handling**: Comprehensive error recovery

#### 2. Configuration System
Flexible configuration with validation:
- **CalibrationConfig**: Main configuration container
- **Builder Pattern**: Type-safe configuration construction
- **Validation**: Comprehensive parameter validation
- **Presets**: Pre-configured settings for common use cases

#### 3. Sampling Strategies
Multiple representative sampling approaches:
- **Random**: Uniform random sampling
- **Stratified**: Balanced sampling across data distribution
- **Importance**: Weighted sampling based on activation importance
- **Systematic**: Deterministic interval-based sampling

#### 4. Statistics Collection
Comprehensive activation analysis:
- **Min/Max Tracking**: Global and per-channel extrema
- **Moment Statistics**: Mean, variance, skewness, kurtosis
- **Percentile Analysis**: Distribution percentiles and quantiles
- **Outlier Detection**: Multiple outlier detection algorithms
- **Shape Information**: Tensor metadata and sparsity analysis

#### 5. Histogram Analysis
Advanced quantization optimization:
- **Adaptive Binning**: Dynamic bin sizing based on distribution
- **Multiple Objectives**: MSE, KL-divergence, entropy optimization
- **Percentile Clipping**: Outlier-robust quantization bounds
- **Optimization Algorithms**: Advanced parameter optimization

#### 6. Streaming Processing
Memory-efficient large dataset handling:
- **Chunked Processing**: Process data in memory-friendly chunks
- **Parallel Processing**: Multi-threaded chunk processing
- **Memory Management**: Dynamic memory usage monitoring
- **Progress Tracking**: Detailed processing metrics

#### 7. Persistence Layer
Save and load calibration results:
- **Multiple Formats**: JSON, Bincode, MessagePack support
- **Compression**: Configurable compression for storage efficiency
- **Caching**: In-memory caching with LRU eviction
- **Integrity**: Checksum validation for data integrity

### Error Handling

Comprehensive error management with recovery strategies:

```rust
use bitnet_quant::calibration::CalibrationError;

match calibration_result {
    Ok(results) => { /* Process results */ },
    Err(CalibrationError::TensorShapeMismatch { expected, actual, .. }) => {
        // Handle shape mismatch with potential recovery
        println!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    },
    Err(CalibrationError::MemoryLimitExceeded { current, limit, .. }) => {
        // Reduce batch size or enable streaming
        println!("Memory limit exceeded: {} > {}", current, limit);
    },
    Err(e) => {
        println!("Calibration error: {}", e);
    }
}
```

## Configuration Guide

### Basic Configuration

```rust
use bitnet_quant::calibration::CalibrationConfigBuilder;

let config = CalibrationConfigBuilder::new()
    .batch_size(64)                    // Processing batch size
    .max_samples(5000)                 // Maximum samples to process
    .build()?;
```

### Advanced Configuration

```rust
use bitnet_quant::calibration::{
    CalibrationConfigBuilder, SamplingStrategy, HistogramConfig,
    BinningStrategy, OptimizationObjective, PersistenceConfig, StorageFormat
};

let config = CalibrationConfigBuilder::new()
    .batch_size(32)
    .max_samples(2000)
    .sampling_strategy(SamplingStrategy::Importance)
    .histogram_config(HistogramConfig {
        num_bins: 256,
        binning_strategy: BinningStrategy::Adaptive,
        percentile_range: (0.1, 99.9),
        optimization_objective: OptimizationObjective::MinimizeKLDivergence,
        enable_outlier_detection: true,
        outlier_threshold: 3.0,
    })
    .persistence_config(PersistenceConfig {
        auto_save: true,
        save_directory: Some(PathBuf::from("./calibration_cache")),
        storage_format: StorageFormat::Bincode,
        compression_enabled: true,
        compression_level: 6,
        cache_size: 1000,
        enable_checksums: true,
    })
    .streaming_enabled(true)
    .memory_limit(1024 * 1024 * 1024)  // 1GB limit
    .build()?;
```

### Device Configuration

```rust
use candle_core::Device;

let config = CalibrationConfigBuilder::new()
    .device_config(DeviceConfig {
        primary_device: Device::Cuda(0),
        fallback_device: Some(Device::Cpu),
        memory_fraction: 0.8,
        allow_growth: true,
    })
    .build()?;
```

## Sampling Strategies

### Random Sampling
```rust
use bitnet_quant::calibration::SamplingStrategy;

let config = CalibrationConfigBuilder::new()
    .sampling_strategy(SamplingStrategy::Random)
    .random_seed(Some(42))  // For reproducibility
    .build()?;
```

### Stratified Sampling
```rust
let config = CalibrationConfigBuilder::new()
    .sampling_strategy(SamplingStrategy::Stratified)
    .build()?;
```

### Importance Sampling
```rust
let config = CalibrationConfigBuilder::new()
    .sampling_strategy(SamplingStrategy::Importance)
    .build()?;
```

## Histogram Optimization

### MSE Minimization
```rust
use bitnet_quant::calibration::{HistogramConfig, OptimizationObjective};

let histogram_config = HistogramConfig {
    optimization_objective: OptimizationObjective::MinimizeMSE,
    num_bins: 256,
    percentile_range: (1.0, 99.0),
    ..Default::default()
};
```

### KL Divergence Minimization
```rust
let histogram_config = HistogramConfig {
    optimization_objective: OptimizationObjective::MinimizeKLDivergence,
    num_bins: 512,
    binning_strategy: BinningStrategy::Adaptive,
    ..Default::default()
};
```

## Streaming for Large Datasets

```rust
let config = CalibrationConfigBuilder::new()
    .streaming_enabled(true)
    .memory_limit(512 * 1024 * 1024)  // 512MB limit
    .batch_size(16)                   // Smaller batches for streaming
    .build()?;

let mut dataset = CalibrationDataset::new(config)?;

// Add large dataset
for batch in large_dataset.chunks(1000) {
    dataset.add_layer_data("large_layer", batch.to_vec())?;
}

let results = dataset.process().await?;
```

## Persistence and Caching

### Save Results
```rust
use bitnet_quant::calibration::CalibrationFactory;

let factory = CalibrationFactory::new(config);
let mut cache = factory.create_cache()?;

// Save calibration results
cache.save_calibration_results("model_v1.0", &results)?;

// Save individual components
cache.save_statistics("layer1_stats", &layer_statistics)?;
cache.save_histogram("layer1_hist", &histogram)?;
```

### Load Results
```rust
// Load complete results
let loaded_results = cache.load_calibration_results("model_v1.0")?;

// Load individual components
let stats = cache.load_statistics("layer1_stats")?;
let histogram = cache.load_histogram("layer1_hist")?;
```

### Cache Management
```rust
// Check cache metrics
let metrics = cache.get_metrics();
println!("Cache hit ratio: {:.2}%", cache.get_hit_ratio() * 100.0);

// Cleanup old entries
let removed_count = cache.cleanup_old_entries(30)?; // Remove entries older than 30 days
```

## Performance Optimization

### Memory Management
- Use streaming for datasets > 1GB
- Adjust batch size based on available memory
- Enable compression for storage-constrained environments

### Processing Speed
- Use multiple workers for parallel processing
- Choose appropriate sampling strategy for dataset size
- Consider GPU acceleration for large tensor operations

### Storage Efficiency
- Use Bincode for faster serialization
- Enable compression for reduced storage
- Regular cache cleanup to manage disk usage

## Examples

### Basic Calibration
```rust
// See examples/calibration_demo.rs for complete example
cargo run --example calibration_demo
```

### Integration Testing
```rust
// Run comprehensive integration tests
cargo test --test integration_tests
```

### Performance Benchmarking
```rust
// Run calibration benchmarks
cargo test --test integration_tests --features benchmark -- --ignored
```

## Error Handling Best Practices

1. **Always handle tensor shape mismatches**
2. **Monitor memory usage in streaming mode**
3. **Validate configuration before processing**
4. **Use checksum validation for critical applications**
5. **Implement retry logic for recoverable errors**

## Advanced Usage

### Custom Sampling Strategy
```rust
// Implement RepresentativeSampler trait for custom sampling
struct CustomSampler;

impl RepresentativeSampler for CustomSampler {
    fn sample(&mut self, data: &[Tensor], target_count: usize) -> CalibrationResult<Vec<usize>> {
        // Custom sampling logic
        todo!()
    }
}
```

### Custom Optimization Objective
```rust
// Implement custom quantization optimization
struct CustomOptimizer;

impl QuantizationOptimizer for CustomOptimizer {
    fn optimize_parameters(&self, histogram: &ActivationHistogram) -> CalibrationResult<QuantizationParameters> {
        // Custom optimization logic
        todo!()
    }
}
```

## API Reference

See the generated documentation for complete API reference:
```bash
cargo doc --open --package bitnet-quant
```

## License

MIT License - see LICENSE file for details.
