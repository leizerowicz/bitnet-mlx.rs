//! Memory-Efficient Data Conversion Demo
//!
//! This example demonstrates the various features of the BitNet memory-efficient
//! data conversion system, including different conversion strategies, performance
//! optimization, and monitoring capabilities.

use bitnet_core::device::get_cpu_device;
use bitnet_core::memory::{
    conversion::{ConversionConfig, ConversionEngine},
    tensor::{BitNetDType, BitNetTensor},
    HybridMemoryPool,
};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BitNet Memory-Efficient Data Conversion Demo ===\n");

    // Initialize memory pool and device
    let pool = Arc::new(HybridMemoryPool::new()?);
    let device = get_cpu_device();

    // Demo 1: Basic Conversions
    demo_basic_conversions(&pool, &device)?;

    // Demo 2: Zero-Copy Conversions
    demo_zero_copy_conversions(&pool, &device)?;

    // Demo 3: In-Place Conversions
    demo_in_place_conversions(&pool, &device)?;

    // Demo 4: Streaming Conversions
    demo_streaming_conversions(&pool, &device)?;

    // Demo 5: Batch Conversions
    demo_batch_conversions(&pool, &device)?;

    // Demo 6: Performance Comparison
    demo_performance_comparison(&pool, &device)?;

    // Demo 7: Memory Efficiency Analysis
    demo_memory_efficiency(&pool, &device)?;

    // Demo 8: Configuration Options
    demo_configuration_options(&pool, &device)?;

    // Demo 9: Metrics and Monitoring
    demo_metrics_and_monitoring(&pool, &device)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_basic_conversions(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Conversions");
    println!("-------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Create a test tensor
    let tensor = BitNetTensor::ones(&[64, 64], BitNetDType::F32, device, pool)?;
    println!(
        "Original tensor: {} ({} bytes)",
        tensor.dtype(),
        tensor.size_bytes()
    );

    // Convert to different data types
    let conversions = vec![
        BitNetDType::F16,
        BitNetDType::BF16,
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::BitNet158,
    ];

    for target_dtype in conversions {
        let start = Instant::now();
        let result = engine.convert(&tensor, target_dtype)?;
        let duration = start.elapsed();

        let compression_ratio = tensor.size_bytes() as f64 / result.size_bytes() as f64;
        println!(
            "  {} -> {}: {:.2}x compression, {} bytes, {:?}",
            tensor.dtype(),
            target_dtype,
            compression_ratio,
            result.size_bytes(),
            duration
        );
    }

    println!();
    Ok(())
}

fn demo_zero_copy_conversions(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Zero-Copy Conversions");
    println!("-----------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Same type conversion (truly zero-copy)
    let f32_tensor = BitNetTensor::ones(&[32, 32], BitNetDType::F32, device, pool)?;
    let start = Instant::now();
    let same_type_result = engine.zero_copy_convert(&f32_tensor, BitNetDType::F32)?;
    let duration = start.elapsed();
    println!("  Same type (F32 -> F32): {duration:?} (zero allocation)");

    // Compatible type conversion (F16 <-> BF16)
    let f16_tensor = BitNetTensor::ones(&[32, 32], BitNetDType::F16, device, pool)?;
    let start = Instant::now();
    let bf16_result = engine.zero_copy_convert(&f16_tensor, BitNetDType::BF16)?;
    let duration = start.elapsed();
    println!("  Compatible types (F16 -> BF16): {duration:?} (memory reinterpretation)");

    // Verify sizes are the same
    assert_eq!(f16_tensor.size_bytes(), bf16_result.size_bytes());
    println!(
        "  ✓ Memory usage unchanged: {} bytes",
        bf16_result.size_bytes()
    );

    println!();
    Ok(())
}

fn demo_in_place_conversions(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. In-Place Conversions");
    println!("----------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Create tensor for in-place conversion
    let mut tensor = BitNetTensor::ones(&[128, 128], BitNetDType::F32, device, pool)?;
    let original_size = tensor.size_bytes();
    println!("  Original: {} ({} bytes)", tensor.dtype(), original_size);

    // Convert F32 -> F16 in-place
    let start = Instant::now();
    engine.in_place_convert(&mut tensor, BitNetDType::F16)?;
    let duration = start.elapsed();

    println!(
        "  After in-place conversion: {} ({} bytes)",
        tensor.dtype(),
        tensor.size_bytes()
    );
    println!("  Time: {duration:?}");
    println!(
        "  Memory saved: {} bytes ({:.1}x reduction)",
        original_size - tensor.size_bytes(),
        original_size as f64 / tensor.size_bytes() as f64
    );

    println!();
    Ok(())
}

fn demo_streaming_conversions(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Streaming Conversions");
    println!("-----------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Create a larger tensor that will trigger streaming
    let large_tensor = BitNetTensor::ones(&[512, 512], BitNetDType::F32, device, pool)?;
    println!(
        "  Large tensor: {} elements ({} bytes)",
        large_tensor.element_count(),
        large_tensor.size_bytes()
    );

    let start = Instant::now();
    let result = engine.streaming_convert(&large_tensor, BitNetDType::I8, 64 * 1024)?;
    let duration = start.elapsed();

    println!("  Streaming conversion F32 -> I8: {duration:?}");
    println!(
        "  Result: {} bytes ({:.1}x compression)",
        result.size_bytes(),
        large_tensor.size_bytes() as f64 / result.size_bytes() as f64
    );

    println!();
    Ok(())
}

fn demo_batch_conversions(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Batch Conversions");
    println!("-------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Create multiple tensors for batch processing
    let tensor_count = 10;
    let tensors: Result<Vec<_>, _> = (0..tensor_count)
        .map(|i| {
            let size = 32 + (i % 16); // Varying sizes
            BitNetTensor::ones(&[size, size], BitNetDType::F32, device, pool)
        })
        .collect();
    let tensors = tensors?;

    println!("  Created {tensor_count} tensors for batch processing");

    // Batch convert all tensors
    let start = Instant::now();
    let results = engine.batch_convert(&tensors, BitNetDType::F16)?;
    let duration = start.elapsed();

    println!("  Batch conversion F32 -> F16: {duration:?}");
    println!(
        "  Throughput: {:.2} tensors/sec",
        tensor_count as f64 / duration.as_secs_f64()
    );

    // Calculate total memory savings
    let original_size: usize = tensors.iter().map(|t| t.size_bytes()).sum();
    let converted_size: usize = results.iter().map(|t| t.size_bytes()).sum();
    println!(
        "  Total memory saved: {} bytes ({:.1}x reduction)",
        original_size - converted_size,
        original_size as f64 / converted_size as f64
    );

    println!();
    Ok(())
}

fn demo_performance_comparison(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Performance Comparison");
    println!("------------------------");

    // Test different configurations
    let configs = vec![
        ("Default", ConversionConfig::default()),
        ("High Performance", ConversionConfig::high_performance()),
        ("Low Memory", ConversionConfig::low_memory()),
        ("High Precision", ConversionConfig::high_precision()),
    ];

    let tensor = BitNetTensor::ones(&[128, 128], BitNetDType::F32, device, pool)?;

    for (name, config) in configs {
        let engine = ConversionEngine::new(config, pool.clone())?;

        let start = Instant::now();
        let _result = engine.convert(&tensor, BitNetDType::I8)?;
        let duration = start.elapsed();

        println!("  {name}: {duration:?}");
    }

    println!();
    Ok(())
}

fn demo_memory_efficiency(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Memory Efficiency Analysis");
    println!("-----------------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    let tensor = BitNetTensor::ones(&[256, 256], BitNetDType::F32, device, pool)?;
    println!(
        "  Base tensor: {} elements, {} KB",
        tensor.element_count(),
        tensor.size_bytes() / 1024
    );

    let data_types = vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::BF16,
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];

    println!("\n  Data Type Comparison:");
    println!("  Type        | Size (KB) | Compression | Memory Efficiency");
    println!("  ------------|-----------|-------------|------------------");

    for dtype in data_types {
        let converted = if dtype == BitNetDType::F32 {
            tensor.clone()
        } else {
            engine.convert(&tensor, dtype)?
        };

        let size_kb = converted.size_bytes() as f64 / 1024.0;
        let compression = tensor.size_bytes() as f64 / converted.size_bytes() as f64;
        let efficiency = dtype.memory_efficiency();

        println!(
            "  {:11} | {:8.2} | {:10.1}x | {:15.1}x",
            format!("{}", dtype),
            size_kb,
            compression,
            efficiency
        );
    }

    println!();
    Ok(())
}

fn demo_configuration_options(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("8. Configuration Options");
    println!("-----------------------");

    // Custom configuration example
    let mut config = ConversionConfig::default();

    // Customize streaming settings
    config.streaming.chunk_size = 2 * 1024 * 1024; // 2MB chunks
    config.streaming.parallel_chunks = 4;
    config.streaming.enable_prefetch = true;

    // Customize batch settings
    config.batch.max_batch_size = 64;
    config.batch.enable_parallel_processing = true;
    config.batch.sort_by_size = true;

    // Customize performance settings
    config.performance.use_simd = true;
    config.performance.use_vectorization = true;
    config.performance.memory_alignment = 64;

    println!("  Custom configuration created:");
    println!(
        "    Streaming chunk size: {} KB",
        config.streaming.chunk_size / 1024
    );
    println!("    Parallel chunks: {}", config.streaming.parallel_chunks);
    println!("    Max batch size: {}", config.batch.max_batch_size);
    println!("    SIMD enabled: {}", config.performance.use_simd);

    // Validate configuration
    match config.validate() {
        Ok(()) => println!("  ✓ Configuration is valid"),
        Err(e) => println!("  ✗ Configuration error: {e}"),
    }

    let engine = ConversionEngine::new(config, pool.clone())?;
    let tensor = BitNetTensor::ones(&[64, 64], BitNetDType::F32, device, pool)?;
    let _result = engine.convert(&tensor, BitNetDType::F16)?;
    println!("  ✓ Engine created and tested successfully");

    println!();
    Ok(())
}

fn demo_metrics_and_monitoring(
    pool: &Arc<HybridMemoryPool>,
    device: &candle_core::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("9. Metrics and Monitoring");
    println!("-------------------------");

    let config = ConversionConfig::default();
    let engine = ConversionEngine::new(config, pool.clone())?;

    // Perform several conversions to generate metrics
    let tensors = [
        BitNetTensor::ones(&[32, 32], BitNetDType::F32, device, pool)?,
        BitNetTensor::ones(&[64, 64], BitNetDType::F32, device, pool)?,
        BitNetTensor::ones(&[128, 128], BitNetDType::F32, device, pool)?,
    ];

    let targets = [BitNetDType::F16, BitNetDType::I8, BitNetDType::I4];

    for (tensor, target) in tensors.iter().zip(targets.iter()) {
        let _result = engine.convert(tensor, *target)?;
    }

    // Get and display metrics
    let stats = engine.get_stats();

    println!("  Conversion Statistics:");
    println!("    Total conversions: {}", stats.total_conversions);
    println!("    Success rate: {:.1}%", stats.success_rate());
    println!("    Average time: {:.2}ms", stats.average_time_ms());
    println!("    Total bytes processed: {}", stats.total_bytes_processed);
    println!(
        "    Throughput: {:.2} MB/s",
        stats.throughput_bytes_per_sec() / (1024.0 * 1024.0)
    );

    // Strategy usage
    if let Some(most_used) = stats.most_used_strategy() {
        println!("    Most used strategy: {most_used:?}");
    }

    if let Some(fastest) = stats.fastest_strategy() {
        println!("    Fastest strategy: {fastest:?}");
    }

    // Memory statistics
    println!("  Memory Statistics:");
    println!(
        "    Peak usage: {} KB",
        stats.memory_stats.peak_memory_usage / 1024
    );
    println!(
        "    Zero-copy percentage: {:.1}%",
        stats.memory_stats.zero_copy_percentage
    );
    println!(
        "    In-place percentage: {:.1}%",
        stats.memory_stats.in_place_percentage
    );

    // Error statistics
    println!("  Error Statistics:");
    println!("    Total errors: {}", stats.error_stats.total_errors);
    println!("    Error rate: {:.2}%", stats.error_stats.error_rate);

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_demo() {
        // Test that the main function runs without panicking
        assert!(main().is_ok());
    }

    #[test]
    fn test_basic_conversion_functionality() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let tensor = BitNetTensor::ones(&[16, 16], BitNetDType::F32, &device, &pool).unwrap();
        let result = engine.convert(&tensor, BitNetDType::F16).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), tensor.shape());
        assert!(result.size_bytes() < tensor.size_bytes());
    }

    #[test]
    fn test_zero_copy_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let tensor = BitNetTensor::ones(&[8, 8], BitNetDType::F32, &device, &pool).unwrap();
        let result = engine.zero_copy_convert(&tensor, BitNetDType::F32).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.size_bytes(), tensor.size_bytes());
    }

    #[test]
    fn test_batch_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let config = ConversionConfig::default();
        let engine = ConversionEngine::new(config, pool.clone()).unwrap();

        let tensors: Vec<_> = (0..3)
            .map(|i| BitNetTensor::ones(&[8 + i, 8 + i], BitNetDType::F32, &device, &pool).unwrap())
            .collect();

        let results = engine.batch_convert(&tensors, BitNetDType::F16).unwrap();

        assert_eq!(results.len(), tensors.len());
        for (original, converted) in tensors.iter().zip(results.iter()) {
            assert_eq!(converted.dtype(), BitNetDType::F16);
            assert_eq!(converted.shape(), original.shape());
            assert!(converted.size_bytes() < original.size_bytes());
        }
    }
}
