//! Integration tests for the inference engine.

use bitnet_inference::InferenceEngine;
use bitnet_inference::engine::batch_processor::{BatchProcessor, BatchConfig};
use bitnet_core::{Device, Tensor};
use candle_core::DType;
use std::time::Duration;

#[test]
fn test_batch_processor_creation() {
    let config = BatchConfig {
        max_batch_size: 32,
        memory_threshold: 1024 * 1024 * 1024, // 1GB
        parallel_workers: Some(4),
    };
    
    let processor = BatchProcessor::new(config.clone());
    let stats = processor.stats();
    
    assert_eq!(stats.max_batch_size, 32);
    assert_eq!(stats.memory_threshold, 1024 * 1024 * 1024);
    assert_eq!(stats.parallel_workers, 4);
}

#[test] 
fn test_empty_batch_processing() {
    let processor = BatchProcessor::new(BatchConfig::default());
    let result = processor.process_batch(Vec::new()).unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_inference_engine_creation() {
    let result = InferenceEngine::new().await;
    assert!(result.is_ok(), "Failed to create inference engine: {:?}", result.err());
    
    let engine = result.unwrap();
    
    // Test that we can get basic info from the engine
    let device = engine.device();
    assert!(matches!(device, Device::Cpu | Device::Metal(_)));
    
    let memory_usage = engine.memory_usage();
    assert!(memory_usage > 0, "Memory usage should be greater than 0");
}

#[tokio::test]
async fn test_inference_engine_builder() {
    let result = InferenceEngine::builder()
        .device(Device::Cpu)
        .batch_size(16)
        .build()
        .await;
        
    assert!(result.is_ok(), "Failed to build inference engine: {:?}", result.err());
    
    let engine = result.unwrap();
    assert!(matches!(engine.device(), Device::Cpu));
}

#[tokio::test]
async fn test_model_loading() {
    let engine = InferenceEngine::new().await.unwrap();
    
    // This will currently use a placeholder model since we don't have real model files
    // In a real implementation, we'd create actual test model files
    let model_result = engine.load_model("test_model.bin").await;
    
    // For now, we expect this to work with the placeholder implementation
    assert!(model_result.is_ok(), "Model loading should work with placeholder");
}

#[test]
fn test_cache_operations() {
    use bitnet_inference::cache::ModelCache;
    
    let cache = ModelCache::new(3, 100 * 1024 * 1024); // 3 models, 100MB
    
    // Test initial state
    assert_eq!(cache.current_memory_usage(), 0);
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_optimization_levels() {
    use bitnet_inference::engine::OptimizationLevel;
    
    // Test that optimization levels have expected properties
    assert_ne!(OptimizationLevel::None, OptimizationLevel::Basic);
    assert_ne!(OptimizationLevel::Basic, OptimizationLevel::Aggressive);
    
    let default_level = OptimizationLevel::default();
    assert_eq!(default_level, OptimizationLevel::Basic);
}

#[tokio::test]
async fn test_inference_engine_variants() {
    // Test different engine creation methods
    let balanced = InferenceEngine::balanced().await;
    assert!(balanced.is_ok(), "Balanced engine creation failed");
    
    let speed_optimized = InferenceEngine::optimized_for_speed().await;
    assert!(speed_optimized.is_ok(), "Speed-optimized engine creation failed");
    
    let memory_optimized = InferenceEngine::optimized_for_memory().await;
    assert!(memory_optimized.is_ok(), "Memory-optimized engine creation failed");
}

#[test]
fn test_error_types() {
    use bitnet_inference::InferenceError;
    
    let error = InferenceError::model_load("test error");
    assert!(error.to_string().contains("Model loading failed"));
    assert!(!error.is_recoverable());
    
    let memory_error = InferenceError::memory("memory issue");
    assert!(memory_error.is_recoverable());
}

// Performance regression test
#[tokio::test]
async fn test_basic_performance_regression() {
    let engine = InferenceEngine::new().await.unwrap();
    
    let start_time = std::time::Instant::now();
    
    // Create a simple test tensor
    let test_tensor = Tensor::zeros(&[1, 512], DType::F32, &Device::Cpu).unwrap();
    
    // This should complete quickly even with placeholder implementation
    let model = engine.load_model("test").await.unwrap();
    let _result = engine.infer(&model, &test_tensor).await.unwrap();
    
    let duration = start_time.elapsed();
    
    // Even with placeholder implementation, basic operations should be fast
    assert!(
        duration < Duration::from_secs(5),
        "Basic inference took too long: {:?}",
        duration
    );
}

#[test]
fn test_memory_size_utilities() {
    use bitnet_inference::api::builder::MemorySize;
    
    assert_eq!(MemorySize::MB(512).bytes(), 512 * 1024 * 1024);
    assert_eq!(MemorySize::GB(2).bytes(), 2 * 1024 * 1024 * 1024);
}

// Test that compilation works with different feature combinations
#[test]
fn test_feature_compilation() {
    // This test just ensures that the code compiles with different features
    // The actual functionality will be tested when features are implemented
    
    #[cfg(feature = "metal")]
    {
        // Test Metal-specific code compiles
        let _device = Device::Metal;
    }
    
    #[cfg(feature = "mlx")]
    {
        // Test MLX-specific code compiles
        let _device = Device::MLX;
    }
}

mod integration {
    use super::*;

    // These tests would be expanded as we implement more functionality
    
    #[tokio::test]
    async fn test_end_to_end_workflow() {
        // Test a complete workflow from engine creation to inference
        let engine = InferenceEngine::builder()
            .batch_size(8)
            .build()
            .await
            .unwrap();
            
        // Load model (placeholder for now)
        let model = engine.load_model("test_model").await.unwrap();
        
        // Create test input
        let input = Tensor::ones(&[1, 512], DType::F32, &Device::Cpu).unwrap();
        
        // Run inference
        let output = engine.infer(&model, &input).await.unwrap();
        
        // Basic sanity checks
        assert_eq!(output.shape().dims(), &[1, 768]); // Expected output shape from CPU backend
    }
    
    #[tokio::test]
    async fn test_batch_inference() {
        let engine = InferenceEngine::new().await.unwrap();
        let model = engine.load_model("test").await.unwrap();
        
        // Create batch of inputs
        let inputs: Vec<_> = (0..4)
            .map(|_| Tensor::ones(&[1, 512], DType::F32, &Device::Cpu).unwrap())
            .collect();
            
        let outputs = engine.infer_batch(&model, &inputs).await.unwrap();
        
        assert_eq!(outputs.len(), inputs.len());
    }
}

// Benchmarking tests (will be expanded in later days)
#[cfg(test)]
mod benchmarks {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_benchmark() {
        let test_tensor = Tensor::randn(0.0f32, 1.0f32, &[1, 512], &Device::Cpu).unwrap();
        
        let results = InferenceEngine::benchmark("test_model", &test_tensor, 10).await.unwrap();
        
        assert_eq!(results.total_iterations, 10);
        assert!(results.warmup_iterations <= 10);
        assert!(results.throughput_ops_per_sec > 0.0);
        
        println!("{}", results.display());
    }
}
