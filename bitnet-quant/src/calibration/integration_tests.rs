//! Comprehensive integration tests for the calibration system

use bitnet_quant::calibration::{
    CalibrationDataset,
    CalibrationFactory,
    CalibrationConfig,
    CalibrationConfigBuilder,
    SamplingStrategy,
    HistogramConfig,
    BinningStrategy,
    OptimizationObjective,
    PersistenceConfig,
    StorageFormat,
    CalibrationError,
    CalibrationResult,
};
use candle_core::{Tensor, Device, DType};
use std::collections::HashMap;
use tempfile::tempdir;

/// Test the complete calibration workflow
#[tokio::test]
async fn test_complete_calibration_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    
    // Create comprehensive calibration configuration
    let config = CalibrationConfigBuilder::new()
        .batch_size(16)
        .max_samples(500)
        .sampling_strategy(SamplingStrategy::Stratified)
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
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Json,
            compression_enabled: true,
            compression_level: 6,
            cache_size: 100,
            enable_checksums: true,
        })
        .build()?;

    // Create mock tensor data for testing
    let device = Device::Cpu;
    let mock_data = create_mock_calibration_data(&device, 100)?;

    // Initialize calibration dataset
    let mut dataset = CalibrationDataset::new(config.clone())?;
    
    // Load data into dataset
    for (layer_name, tensor_batch) in mock_data {
        dataset.add_layer_data(&layer_name, tensor_batch)?;
    }

    // Process the dataset
    let results = dataset.process().await?;

    // Verify results structure
    assert!(!results.layer_statistics.is_empty());
    assert!(!results.quantization_parameters.is_empty());
    assert!(results.metadata.processing_time_ms > 0);

    // Test persistence functionality
    let factory = CalibrationFactory::new(config.clone());
    let mut cache = factory.create_cache()?;

    // Save results to cache
    cache.save_calibration_results("test_model_v1", &results)?;

    // Load results from cache
    let loaded_results = cache.load_calibration_results("test_model_v1")?;
    assert!(loaded_results.is_some());
    
    let loaded_results = loaded_results.unwrap();
    assert_eq!(loaded_results.layer_statistics.len(), results.layer_statistics.len());

    // Test statistics functionality
    for (layer_name, stats) in &results.layer_statistics {
        println!("Layer: {}", layer_name);
        println!("  Min/Max: {:.6} / {:.6}", stats.min_max.global_min, stats.min_max.global_max);
        println!("  Mean/Std: {:.6} / {:.6}", stats.moments.mean, stats.moments.std_dev);
        println!("  Shape: {:?}", stats.shape_info.dimensions);
        println!("  Elements: {}", stats.shape_info.num_elements);
    }

    // Test quantization parameters
    for (layer_name, params) in &results.quantization_parameters {
        println!("Quantization params for {}: scale={:.6}, zero_point={}",
            layer_name, params.scale, params.zero_point);
    }

    println!("Calibration workflow completed successfully!");
    Ok(())
}

/// Test streaming calibration for large datasets
#[tokio::test]
async fn test_streaming_calibration() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    
    let config = CalibrationConfigBuilder::new()
        .batch_size(8)
        .max_samples(200)
        .streaming_enabled(true)
        .memory_limit(50 * 1024 * 1024) // 50MB limit to force streaming
        .build()?;

    // Create larger mock dataset
    let device = Device::Cpu;
    let large_mock_data = create_large_mock_calibration_data(&device, 500)?;

    let mut dataset = CalibrationDataset::new(config)?;
    
    // Add data in chunks to simulate streaming
    for (layer_name, tensor_batch) in large_mock_data {
        dataset.add_layer_data(&layer_name, tensor_batch)?;
    }

    let results = dataset.process().await?;
    
    // Verify streaming worked correctly
    assert!(!results.layer_statistics.is_empty());
    assert!(results.metadata.total_samples_processed > 0);
    assert!(results.metadata.streaming_enabled);

    println!("Streaming calibration test passed!");
    Ok(())
}

/// Test different sampling strategies
#[tokio::test]
async fn test_sampling_strategies() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mock_data = create_mock_calibration_data(&device, 200)?;
    
    let strategies = vec![
        SamplingStrategy::Random,
        SamplingStrategy::Stratified,
        SamplingStrategy::Importance,
        SamplingStrategy::Systematic,
    ];

    for strategy in strategies {
        println!("Testing sampling strategy: {:?}", strategy);
        
        let config = CalibrationConfigBuilder::new()
            .batch_size(16)
            .max_samples(100)
            .sampling_strategy(strategy)
            .build()?;

        let mut dataset = CalibrationDataset::new(config)?;
        
        // Add mock data
        for (layer_name, tensor_batch) in mock_data.clone() {
            dataset.add_layer_data(&layer_name, tensor_batch)?;
        }

        let results = dataset.process().await?;
        assert!(!results.layer_statistics.is_empty());
        
        println!("  Successfully processed {} layers", results.layer_statistics.len());
    }

    println!("Sampling strategies test passed!");
    Ok(())
}

/// Test histogram-based quantization optimization
#[tokio::test]
async fn test_histogram_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Create data with specific distribution for testing
    let test_tensor = Tensor::randn(0.0f32, 1.0f32, (1000, 128), &device)?;
    let mut layer_data = HashMap::new();
    layer_data.insert("test_layer".to_string(), vec![test_tensor]);

    let objectives = vec![
        OptimizationObjective::MinimizeMSE,
        OptimizationObjective::MinimizeKLDivergence,
        OptimizationObjective::MaximizeEntropy,
        OptimizationObjective::MinimizePercentileClipping,
    ];

    for objective in objectives {
        println!("Testing optimization objective: {:?}", objective);
        
        let config = CalibrationConfigBuilder::new()
            .batch_size(32)
            .max_samples(500)
            .histogram_config(HistogramConfig {
                num_bins: 128,
                binning_strategy: BinningStrategy::Adaptive,
                percentile_range: (1.0, 99.0),
                optimization_objective: objective,
                enable_outlier_detection: true,
                outlier_threshold: 2.5,
            })
            .build()?;

        let mut dataset = CalibrationDataset::new(config)?;
        
        for (layer_name, tensor_batch) in layer_data.clone() {
            dataset.add_layer_data(&layer_name, tensor_batch)?;
        }

        let results = dataset.process().await?;
        
        // Verify quantization parameters were generated
        assert!(results.quantization_parameters.contains_key("test_layer"));
        let params = &results.quantization_parameters["test_layer"];
        
        println!("  Scale: {:.6}, Zero point: {}", params.scale, params.zero_point);
        assert!(params.scale > 0.0);
    }

    println!("Histogram optimization test passed!");
    Ok(())
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Test with invalid configuration
    let result = CalibrationConfigBuilder::new()
        .batch_size(0) // Invalid batch size
        .build();
    assert!(result.is_err());

    // Test with mismatched tensor shapes
    let config = CalibrationConfigBuilder::new()
        .batch_size(16)
        .max_samples(100)
        .build()?;

    let mut dataset = CalibrationDataset::new(config)?;
    
    // Add tensors with different shapes to the same layer
    let tensor1 = Tensor::randn(0.0f32, 1.0f32, (64, 128), &device)?;
    let tensor2 = Tensor::randn(0.0f32, 1.0f32, (32, 256), &device)?; // Different shape
    
    dataset.add_layer_data("test_layer", vec![tensor1])?;
    
    // This should succeed but log a warning about shape mismatch
    let result = dataset.add_layer_data("test_layer", vec![tensor2]);
    match result {
        Ok(_) => println!("Shape mismatch handled gracefully"),
        Err(CalibrationError::TensorShapeMismatch { .. }) => {
            println!("Shape mismatch detected correctly");
        },
        Err(e) => return Err(Box::new(e)),
    }

    println!("Error handling test passed!");
    Ok(())
}

/// Test cache performance and metrics
#[tokio::test]
async fn test_cache_performance() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    
    let config = CalibrationConfigBuilder::new()
        .persistence_config(PersistenceConfig {
            auto_save: true,
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Bincode,
            compression_enabled: true,
            compression_level: 6,
            cache_size: 10,
            enable_checksums: true,
        })
        .build()?;

    let factory = CalibrationFactory::new(config);
    let mut cache = factory.create_cache()?;

    // Create test data
    let device = Device::Cpu;
    let mock_data = create_mock_calibration_data(&device, 50)?;
    
    let mut dataset = CalibrationDataset::new(factory.get_config().clone())?;
    for (layer_name, tensor_batch) in mock_data {
        dataset.add_layer_data(&layer_name, tensor_batch)?;
    }
    
    let results = dataset.process().await?;

    // Measure cache performance
    let start_time = std::time::Instant::now();
    
    // Save multiple entries
    for i in 0..5 {
        let key = format!("test_entry_{}", i);
        cache.save_calibration_results(&key, &results)?;
    }
    
    let save_time = start_time.elapsed();
    
    let start_time = std::time::Instant::now();
    
    // Load entries (should hit cache)
    for i in 0..5 {
        let key = format!("test_entry_{}", i);
        let loaded = cache.load_calibration_results(&key)?;
        assert!(loaded.is_some());
    }
    
    let load_time = start_time.elapsed();
    
    // Check cache metrics
    let metrics = cache.get_metrics();
    let hit_ratio = cache.get_hit_ratio();
    
    println!("Cache Performance Metrics:");
    println!("  Save time: {:?}", save_time);
    println!("  Load time: {:?}", load_time);
    println!("  Hit ratio: {:.2}%", hit_ratio * 100.0);
    println!("  Total entries: {}", metrics.total_entries);
    println!("  Average access time: {:.2}ms", metrics.avg_access_time);

    assert!(hit_ratio >= 0.0 && hit_ratio <= 1.0);
    assert!(metrics.total_entries > 0);

    println!("Cache performance test passed!");
    Ok(())
}

/// Helper function to create mock calibration data
fn create_mock_calibration_data(
    device: &Device, 
    num_samples: usize
) -> CalibrationResult<HashMap<String, Vec<Tensor>>> {
    let mut layer_data = HashMap::new();
    
    // Create data for different layer types
    let layer_configs = vec![
        ("conv1", vec![num_samples, 64, 224, 224]),
        ("conv2", vec![num_samples, 128, 112, 112]),
        ("fc1", vec![num_samples, 1024]),
        ("fc2", vec![num_samples, 512]),
        ("output", vec![num_samples, 10]),
    ];

    for (layer_name, shape) in layer_configs {
        let tensor = Tensor::randn(0.0f32, 1.0f32, shape, device)
            .map_err(|e| CalibrationError::tensor_operation(format!("Failed to create mock tensor: {}", e)))?;
        
        layer_data.insert(layer_name.to_string(), vec![tensor]);
    }

    Ok(layer_data)
}

/// Helper function to create large mock calibration data for streaming tests
fn create_large_mock_calibration_data(
    device: &Device,
    num_samples: usize
) -> CalibrationResult<HashMap<String, Vec<Tensor>>> {
    let mut layer_data = HashMap::new();
    
    // Create larger tensors to test streaming
    let layer_configs = vec![
        ("large_conv1", vec![num_samples, 256, 56, 56]),
        ("large_conv2", vec![num_samples, 512, 28, 28]),
        ("large_fc1", vec![num_samples, 4096]),
        ("large_fc2", vec![num_samples, 2048]),
    ];

    for (layer_name, shape) in layer_configs {
        // Create multiple tensor chunks
        let mut tensors = Vec::new();
        let chunk_size = num_samples.min(50); // Process in smaller chunks
        
        for i in (0..num_samples).step_by(chunk_size) {
            let actual_chunk_size = (num_samples - i).min(chunk_size);
            let mut chunk_shape = shape.clone();
            chunk_shape[0] = actual_chunk_size;
            
            let tensor = Tensor::randn(0.0f32, 1.0f32, chunk_shape, device)
                .map_err(|e| CalibrationError::tensor_operation(format!("Failed to create large mock tensor: {}", e)))?;
            
            tensors.push(tensor);
        }
        
        layer_data.insert(layer_name.to_string(), tensors);
    }

    Ok(layer_data)
}

/// Benchmark calibration performance
#[tokio::test] 
#[ignore] // Use `cargo test -- --ignored` to run benchmarks
async fn benchmark_calibration_performance() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let sizes = vec![100, 500, 1000, 2000];
    
    for size in sizes {
        println!("Benchmarking with {} samples...", size);
        
        let config = CalibrationConfigBuilder::new()
            .batch_size(32)
            .max_samples(size)
            .sampling_strategy(SamplingStrategy::Random)
            .build()?;

        let mock_data = create_mock_calibration_data(&device, size)?;
        
        let start_time = std::time::Instant::now();
        
        let mut dataset = CalibrationDataset::new(config)?;
        for (layer_name, tensor_batch) in mock_data {
            dataset.add_layer_data(&layer_name, tensor_batch)?;
        }
        
        let results = dataset.process().await?;
        
        let processing_time = start_time.elapsed();
        let throughput = size as f64 / processing_time.as_secs_f64();
        
        println!("  Processing time: {:?}", processing_time);
        println!("  Throughput: {:.2} samples/second", throughput);
        println!("  Layers processed: {}", results.layer_statistics.len());
        println!("  Memory usage: {} MB", results.metadata.peak_memory_usage_mb);
        println!();
    }

    Ok(())
}
