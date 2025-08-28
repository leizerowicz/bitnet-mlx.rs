//! Comprehensive tests for memory-efficient data conversion system
//!
//! This test suite validates the functionality, performance, and memory efficiency
//! of the BitNet data conversion system.

use bitnet_core::device::get_cpu_device;
use bitnet_core::memory::{
    conversion::{
        BatchConfig, ConversionConfig, ConversionEngine, ConversionEvent, ConversionQuality,
        ConversionStats, ConversionStrategy, StreamingConfig,
    },
    tensor::{BitNetDType, BitNetTensor},
    HybridMemoryPool,
};
use std::sync::Arc;
use std::time::Duration;

/// Test basic conversion engine functionality
#[test]
fn test_conversion_engine_basic() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test single tensor conversion
    let source = BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap();
    let result = engine.convert(&source, BitNetDType::F16).unwrap();

    assert_eq!(result.dtype(), BitNetDType::F16);
    assert_eq!(result.shape(), vec![4, 4]);
    assert!(result.size_bytes() < source.size_bytes()); // F16 uses less memory than F32
}

/// Test zero-copy conversions
#[test]
fn test_zero_copy_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test same-type conversion (should be zero-copy)
    let source = BitNetTensor::zeros(&[8, 8], BitNetDType::F32, &device, &pool).unwrap();
    let result = engine.zero_copy_convert(&source, BitNetDType::F32).unwrap();

    assert_eq!(result.dtype(), BitNetDType::F32);
    assert_eq!(result.shape(), vec![8, 8]);
    assert_eq!(result.size_bytes(), source.size_bytes());

    // Test F16 <-> BF16 conversion (should be zero-copy compatible)
    let f16_tensor = BitNetTensor::ones(&[4, 4], BitNetDType::F16, &device, &pool).unwrap();
    let bf16_result = engine
        .zero_copy_convert(&f16_tensor, BitNetDType::BF16)
        .unwrap();

    assert_eq!(bf16_result.dtype(), BitNetDType::BF16);
    assert_eq!(bf16_result.size_bytes(), f16_tensor.size_bytes());
}

/// Test streaming conversions for large tensors
#[test]
fn test_streaming_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();

    // Configure for streaming with small chunks to test the mechanism
    let mut config = ConversionConfig::default();
    config.streaming.chunk_size = 1024; // Small chunks for testing
    config.streaming.streaming_threshold = 512; // Low threshold to trigger streaming

    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create a tensor that will trigger streaming
    let large_tensor = BitNetTensor::ones(&[64, 64], BitNetDType::F32, &device, &pool).unwrap();
    let result = engine
        .streaming_convert(&large_tensor, BitNetDType::F16, 512)
        .unwrap();

    assert_eq!(result.dtype(), BitNetDType::F16);
    assert_eq!(result.shape(), vec![64, 64]);
    assert!(result.size_bytes() < large_tensor.size_bytes());
}

/// Test in-place conversions
#[test]
fn test_in_place_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test F32 to F16 in-place conversion
    let mut tensor = BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap();
    let original_size = tensor.size_bytes();

    engine
        .in_place_convert(&mut tensor, BitNetDType::F16)
        .unwrap();

    assert_eq!(tensor.dtype(), BitNetDType::F16);
    assert_eq!(tensor.shape(), vec![4, 4]);
    assert!(tensor.size_bytes() < original_size);
}

/// Test batch conversions
#[test]
fn test_batch_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create multiple tensors for batch conversion
    let tensors = vec![
        BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
        BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
        BitNetTensor::zeros(&[4, 4], BitNetDType::F32, &device, &pool).unwrap(),
    ];

    let results = engine.batch_convert(&tensors, BitNetDType::F16).unwrap();

    assert_eq!(results.len(), 3);
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), tensors[i].shape());
        assert!(result.size_bytes() < tensors[i].size_bytes());
    }
}

/// Test mixed batch conversions
#[test]
fn test_mixed_batch_conversions() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create conversions with different target types
    let conversions = vec![
        (
            BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetDType::F16,
        ),
        (
            BitNetTensor::zeros(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetDType::I8,
        ),
        (
            BitNetTensor::ones(&[4, 4], BitNetDType::F16, &device, &pool).unwrap(),
            BitNetDType::I8,
        ),
    ];

    let results = engine.batch_convert_mixed(&conversions).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].dtype(), BitNetDType::F16);
    assert_eq!(results[1].dtype(), BitNetDType::I8);
    assert_eq!(results[2].dtype(), BitNetDType::I8);
}

/// Test conversion pipeline
#[test]
fn test_conversion_pipeline() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create a multi-stage pipeline: F32 -> F16 -> I8
    let pipeline = engine
        .create_pipeline()
        .unwrap()
        .add_stage(BitNetDType::F16)
        .add_stage(BitNetDType::I8);

    let input = BitNetTensor::ones(&[8, 8], BitNetDType::F32, &device, &pool).unwrap();
    let result = pipeline.execute(&input).unwrap();

    assert_eq!(result.dtype(), BitNetDType::I8);
    assert_eq!(result.shape(), vec![8, 8]);
    assert!(result.size_bytes() < input.size_bytes());

    // Test pipeline statistics
    let stats = pipeline.get_stats().unwrap();
    assert!(stats.total_executions > 0);
}

/// Test conversion metrics and monitoring
#[test]
fn test_conversion_metrics() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Perform several conversions
    let tensor1 = BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap();
    let tensor2 = BitNetTensor::zeros(&[8, 8], BitNetDType::F32, &device, &pool).unwrap();

    let _result1 = engine.convert(&tensor1, BitNetDType::F16).unwrap();
    let _result2 = engine.convert(&tensor2, BitNetDType::I8).unwrap();

    // Check metrics
    let metrics = engine.get_stats();
    assert_eq!(metrics.total_conversions, 2);
    assert_eq!(metrics.successful_conversions, 2);
    assert_eq!(metrics.success_rate(), 100.0);
    assert!(metrics.total_bytes_processed > 0);
    assert!(metrics.average_time_ms() >= 0.0);
}

/// Test conversion strategy selection
#[test]
fn test_strategy_selection() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test strategy info for different conversions
    let info_same_type =
        engine.get_optimal_strategy_info(BitNetDType::F32, BitNetDType::F32, &[4, 4], &device);
    assert!(info_same_type.is_zero_copy);
    assert!(info_same_type.is_supported);

    let info_downsize =
        engine.get_optimal_strategy_info(BitNetDType::F32, BitNetDType::F16, &[4, 4], &device);
    assert!(info_downsize.is_in_place);
    assert!(info_downsize.is_supported);
    assert_eq!(info_downsize.compression_ratio, 2.0);

    let info_large_tensor =
        engine.get_optimal_strategy_info(BitNetDType::F32, BitNetDType::I8, &[1000, 1000], &device);
    assert!(info_large_tensor.is_supported);
    assert_eq!(info_large_tensor.compression_ratio, 4.0);
}

/// Test error handling and edge cases
#[test]
fn test_error_handling() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test unsupported zero-copy conversion
    let tensor = BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap();
    let result = engine.zero_copy_convert(&tensor, BitNetDType::I8);
    assert!(result.is_err());

    // Test empty batch conversion
    let empty_batch: Vec<BitNetTensor> = vec![];
    let result = engine
        .batch_convert(&empty_batch, BitNetDType::F16)
        .unwrap();
    assert!(result.is_empty());
}

/// Test memory efficiency
#[test]
fn test_memory_efficiency() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Test compression ratios for different conversions
    let f32_tensor = BitNetTensor::ones(&[16, 16], BitNetDType::F32, &device, &pool).unwrap();

    // F32 to F16 should be 2x compression
    let f16_result = engine.convert(&f32_tensor, BitNetDType::F16).unwrap();
    assert_eq!(f16_result.size_bytes() * 2, f32_tensor.size_bytes());

    // F32 to I8 should be 4x compression
    let i8_result = engine.convert(&f32_tensor, BitNetDType::I8).unwrap();
    assert_eq!(i8_result.size_bytes() * 4, f32_tensor.size_bytes());

    // F32 to I4 should be 8x compression (packed)
    let i4_result = engine.convert(&f32_tensor, BitNetDType::I4).unwrap();
    assert!(i4_result.size_bytes() * 8 >= f32_tensor.size_bytes());
}

/// Test configuration options
#[test]
fn test_configuration_options() {
    let _pool = Arc::new(HybridMemoryPool::new().unwrap());

    // Test low memory configuration
    let low_mem_config = ConversionConfig::low_memory();
    assert_eq!(
        low_mem_config.default_strategy,
        ConversionStrategy::Streaming
    );
    assert_eq!(low_mem_config.max_memory_usage, 256 * 1024 * 1024);

    // Test high performance configuration
    let high_perf_config = ConversionConfig::high_performance();
    assert_eq!(high_perf_config.default_quality, ConversionQuality::Fast);
    assert!(high_perf_config.performance.use_simd);

    // Test high precision configuration
    let high_prec_config = ConversionConfig::high_precision();
    assert_eq!(high_prec_config.default_quality, ConversionQuality::Precise);
    assert!(!high_prec_config.performance.use_simd);

    // Validate configurations
    assert!(low_mem_config.validate().is_ok());
    assert!(high_perf_config.validate().is_ok());
    assert!(high_prec_config.validate().is_ok());
}

/// Test streaming configuration
#[test]
fn test_streaming_configuration() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();

    // Test custom streaming configuration
    let mut config = ConversionConfig::default();
    config.streaming = StreamingConfig::low_memory();

    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    let tensor = BitNetTensor::ones(&[32, 32], BitNetDType::F32, &device, &pool).unwrap();
    let result = engine.convert(&tensor, BitNetDType::F16).unwrap();

    assert_eq!(result.dtype(), BitNetDType::F16);
    assert_eq!(result.shape(), vec![32, 32]);
}

/// Test batch configuration
#[test]
fn test_batch_configuration() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();

    // Test custom batch configuration
    let mut config = ConversionConfig::default();
    config.batch = BatchConfig::high_performance();

    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    let tensors = vec![
        BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap(),
        BitNetTensor::zeros(&[4, 4], BitNetDType::F32, &device, &pool).unwrap(),
    ];

    let results = engine.batch_convert(&tensors, BitNetDType::F16).unwrap();
    assert_eq!(results.len(), 2);
}

/// Test conversion event tracking
#[test]
fn test_conversion_event_tracking() {
    let device = get_cpu_device();

    // Create a conversion event
    let event = ConversionEvent::new(
        BitNetDType::F32,
        BitNetDType::F16,
        ConversionStrategy::InPlace,
        ConversionQuality::Balanced,
        &device,
        1024,
        512,
        256,
    );

    assert_eq!(event.sourcedtype, BitNetDType::F32);
    assert_eq!(event.targetdtype, BitNetDType::F16);
    assert_eq!(event.compression_ratio(), 2.0);
    assert!(!event.success);

    // Complete the event
    let completed = event.complete_success(Duration::from_millis(50), 512, 1024);
    assert!(completed.success);
    assert_eq!(completed.duration_ms, 50);
    assert_eq!(completed.elements_per_second(), 5120.0);
}

/// Test conversion statistics aggregation
#[test]
fn test_conversion_statistics() {
    let stats = ConversionStats::new(100);
    let device = get_cpu_device();

    // Add some events
    for i in 0..5 {
        let event = ConversionEvent::new(
            BitNetDType::F32,
            BitNetDType::F16,
            ConversionStrategy::InPlace,
            ConversionQuality::Balanced,
            &device,
            1024,
            512,
            256,
        )
        .complete_success(Duration::from_millis(50 + i * 10), 512, 1024);

        stats.record_event(event);
    }

    let metrics = stats.generate_metrics();
    assert_eq!(metrics.total_conversions, 5);
    assert_eq!(metrics.successful_conversions, 5);
    assert_eq!(metrics.success_rate(), 100.0);
    assert!(metrics.average_time_ms() > 0.0);

    // Test recent events
    let recent = stats.get_recent_events(3);
    assert_eq!(recent.len(), 3);
}

/// Test performance under load
#[test]
fn test_performance_under_load() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::high_performance();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create many tensors for batch processing
    let tensors: Vec<_> = (0..50)
        .map(|_| BitNetTensor::ones(&[8, 8], BitNetDType::F32, &device, &pool).unwrap())
        .collect();

    let start = std::time::Instant::now();
    let results = engine.batch_convert(&tensors, BitNetDType::F16).unwrap();
    let duration = start.elapsed();

    assert_eq!(results.len(), 50);
    println!("Batch conversion of 50 tensors took: {duration:?}");

    // Verify all conversions succeeded
    for result in &results {
        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), vec![8, 8]);
    }

    // Check final metrics
    let metrics = engine.get_stats();
    assert_eq!(metrics.total_conversions, 50);
    assert_eq!(metrics.success_rate(), 100.0);
}

/// Integration test combining multiple features
#[test]
fn test_integration_comprehensive() {
    let pool = Arc::new(HybridMemoryPool::new().unwrap());
    let device = get_cpu_device();
    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone()).unwrap();

    // Create a complex conversion pipeline
    let pipeline = engine
        .create_pipeline()
        .unwrap()
        .add_stage(BitNetDType::F16)
        .add_stage(BitNetDType::I8)
        .add_stage(BitNetDType::I4)
        .optimize();

    // Process multiple tensors through the pipeline
    let inputs = vec![
        BitNetTensor::ones(&[16, 16], BitNetDType::F32, &device, &pool).unwrap(),
        BitNetTensor::zeros(&[32, 32], BitNetDType::F32, &device, &pool).unwrap(),
    ];

    let results = pipeline.execute_batch(&inputs).unwrap();

    assert_eq!(results.len(), 2);
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.dtype(), BitNetDType::I4);
        assert_eq!(result.shape(), inputs[i].shape());
        // I4 should use significantly less memory than F32
        assert!(result.size_bytes() < inputs[i].size_bytes() / 4);
    }

    // Check pipeline statistics
    let pipeline_stats = pipeline.get_stats().unwrap();
    assert!(pipeline_stats.total_executions > 0);
    assert!(pipeline_stats.total_stages_executed > 0);

    // Check engine statistics
    let engine_stats = engine.get_stats();
    assert!(engine_stats.total_conversions > 0);
    assert_eq!(engine_stats.success_rate(), 100.0);
}
