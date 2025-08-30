//! Day 8: GPU Optimization Implementation - Core Test Suite
//!
//! This test suite validates essential Day 8 GPU optimization features including:
//! - Advanced GPU memory management 
//! - Buffer allocation and management
//! - Asynchronous memory transfers
//! - Performance monitoring and statistics

use bitnet_inference::{
    Result,
    engine::{
        Model, ModelArchitecture, LayerConfig, LayerType, LayerParameters, QuantizationConfig,
        gpu_memory_optimizer::*
    },
};
use bitnet_core::Device;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio;

/// Test GPU Memory Manager initialization for different devices
#[test]
fn test_gpu_memory_manager_creation() -> Result<()> {
    // Test CPU device initialization
    let cpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    
    let stats = cpu_manager.get_memory_stats()?;
    assert_eq!(stats.active_allocations, 0);
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.peak_usage, 0);
    
    println!("✅ CPU GPU Memory Manager initialization test passed");
    Ok(())
}

/// Test inference buffer allocation for various batch sizes
#[test]
fn test_inference_buffer_allocation() -> Result<()> {
    let mut gpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    let test_model = create_test_model();
    
    // Test different batch sizes
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128] {
        let buffers = gpu_manager.allocate_inference_buffers(batch_size, &test_model)?;
        
        // Verify buffer allocations
        assert!(buffers.input.size > 0);
        assert!(buffers.output.size > 0);
        assert!(buffers.weights.size > 0);
        
        // Verify buffer alignment
        assert!(buffers.input.alignment >= 16); // Minimum SIMD alignment
        assert!(buffers.output.alignment >= 16);
        assert!(buffers.weights.alignment >= 16);
        
        // Verify buffer size scaling with batch size
        if batch_size >= 2 {
            let larger_buffers = gpu_manager.allocate_inference_buffers(batch_size * 2, &test_model)?;
            assert!(larger_buffers.input.size > buffers.input.size);
        }
        
        println!("✅ Buffer allocation test passed for batch size: {}", batch_size);
    }
    
    Ok(())
}

/// Test memory statistics and tracking functionality
#[test]
fn test_memory_statistics_tracking() -> Result<()> {
    let mut gpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    let test_model = create_test_model();
    
    // Initial state
    let initial_stats = gpu_manager.get_memory_stats()?;
    assert_eq!(initial_stats.active_allocations, 0);
    assert_eq!(initial_stats.total_allocated, 0);
    
    // Allocate some buffers
    let _buffers1 = gpu_manager.allocate_inference_buffers(8, &test_model)?;
    let mid_stats = gpu_manager.get_memory_stats()?;
    assert!(mid_stats.active_allocations > 0);
    assert!(mid_stats.total_allocated > 0);
    
    // Allocate more buffers
    let _buffers2 = gpu_manager.allocate_inference_buffers(16, &test_model)?;
    let final_stats = gpu_manager.get_memory_stats()?;
    
    // Verify memory tracking accuracy
    assert!(final_stats.active_allocations >= mid_stats.active_allocations);
    assert!(final_stats.total_allocated >= mid_stats.total_allocated);
    assert!(final_stats.peak_usage >= final_stats.total_allocated);
    
    // Verify statistical accuracy
    assert!(final_stats.buffer_pool_hit_rate >= 0.0);
    assert!(final_stats.buffer_pool_hit_rate <= 100.0);
    assert!(final_stats.fragmentation_percentage >= 0.0);
    
    println!("✅ Memory statistics tracking test passed");
    Ok(())
}

/// Test asynchronous memory transfer operations  
#[tokio::test]
async fn test_async_memory_transfers() -> Result<()> {
    let mut gpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    let test_model = create_test_model();
    
    let buffers = gpu_manager.allocate_inference_buffers(4, &test_model)?;
    let test_data = vec![1.0f32; 1024]; // Test data to transfer
    
    // Test async GPU memory copy
    let result = gpu_manager.copy_to_gpu_async(&test_data, &buffers.input).await;
    assert!(result.is_ok());
    
    // Test concurrent transfers
    let buffer2 = gpu_manager.allocate_inference_buffers(8, &test_model)?;
    let test_data2 = vec![2.0f32; 2048];
    
    // Start two async transfers concurrently
    let transfer1 = gpu_manager.copy_to_gpu_async(&test_data, &buffers.input);
    let transfer2 = gpu_manager.copy_to_gpu_async(&test_data2, &buffer2.input);
    
    let (result1, result2) = tokio::join!(transfer1, transfer2);
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    println!("✅ Async memory transfer test passed");
    Ok(())
}

/// Test performance optimization and analysis
#[tokio::test] 
async fn test_performance_optimization_analysis() -> Result<()> {
    let mut gpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    let test_model = create_test_model();
    
    // Measure allocation performance
    let start = Instant::now();
    let _buffers = gpu_manager.allocate_inference_buffers(32, &test_model)?;
    let allocation_time = start.elapsed();
    
    assert!(allocation_time < Duration::from_millis(100)); // Should be fast
    
    // Test memory transfer bandwidth
    let buffers = gpu_manager.allocate_inference_buffers(64, &test_model)?;
    let large_data = vec![1.0f32; 16384]; // 64KB of data
    
    let start = Instant::now();
    gpu_manager.copy_to_gpu_async(&large_data, &buffers.input).await?;
    let transfer_time = start.elapsed();
    
    let bandwidth = (large_data.len() * 4) as f64 / transfer_time.as_secs_f64(); // bytes/second
    assert!(bandwidth > 0.0);
    
    // Get final statistics with bandwidth info
    let stats = gpu_manager.get_memory_stats()?;
    assert!(stats.transfer_bandwidth >= 0.0);
    
    println!("✅ Performance optimization test passed - Bandwidth: {:.2} MB/s", 
             bandwidth / 1_000_000.0);
    Ok(())
}

/// Test buffer pool optimization and reuse
#[tokio::test]
async fn test_concurrent_buffer_allocation() -> Result<()> {
    use tokio::sync::Mutex;
    
    let gpu_manager = Arc::new(Mutex::new(GPUMemoryManager::new(Device::Cpu)?));
    let test_model = Arc::new(create_test_model());
    
    let mut handles = vec![];
    
    // Spawn multiple concurrent allocation tasks
    for i in 0..8 {
        let gpu_manager_clone = Arc::clone(&gpu_manager);
        let test_model_clone = Arc::clone(&test_model);
        
        let handle = tokio::spawn(async move {
            let mut manager = gpu_manager_clone.lock().await;
            let batch_size = 2usize.pow(i % 4); // Vary batch sizes: 1, 2, 4, 8
            let buffers = manager.allocate_inference_buffers(batch_size, &test_model_clone)?;
            
            // Verify buffer allocation
            assert!(buffers.input.size > 0);
            assert!(buffers.output.size > 0);
            assert!(buffers.weights.size > 0);
            
            Result::<()>::Ok(())
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap()?;
    }
    
    // Check final state
    let manager = gpu_manager.lock().await;
    let stats = manager.get_memory_stats()?;
    
    assert!(stats.active_allocations > 0);
    assert!(stats.total_allocated > 0);
    
    println!("✅ Concurrent buffer allocation test passed - {} active allocations", 
             stats.active_allocations);
    Ok(())
}

/// Integration test for the complete Day 8 GPU optimization pipeline
#[tokio::test]
async fn test_day8_integration() -> Result<()> {
    println!("🧪 Day 8 GPU Optimization Integration Test");
    
    // Initialize GPU memory manager
    let mut gpu_manager = GPUMemoryManager::new(Device::Cpu)?;
    let test_model = create_test_model();
    
    // Step 1: Allocate inference buffers
    let buffers = gpu_manager.allocate_inference_buffers(16, &test_model)?;
    println!("✅ Step 1: Inference buffers allocated");
    
    // Step 2: Prepare test data
    let input_data: Vec<f32> = (0..test_model.get_input_dim())
        .map(|i| (i as f32).sin() * 0.5)
        .collect();
    
    // Step 3: Async memory transfer
    gpu_manager.copy_to_gpu_async(&input_data, &buffers.input).await?;
    println!("✅ Step 3: Asynchronous memory transfer completed");
    
    // Step 4: Verify memory statistics
    let stats = gpu_manager.get_memory_stats()?;
    assert!(stats.active_allocations > 0);
    assert!(stats.staging_operations > 0);
    assert!(stats.transfer_bandwidth > 0.0);
    println!("✅ Step 4: Memory statistics validated");
    
    // Step 5: Performance analysis
    assert!(stats.fragmentation_percentage <= 50.0); // Reasonable fragmentation
    if stats.buffer_pool_hit_rate > 0.0 {
        assert!(stats.buffer_pool_hit_rate <= 100.0);
    }
    println!("✅ Step 5: Performance analysis completed");
    
    // Summary
    println!("📊 Integration Test Summary:");
    println!("   Active allocations: {}", stats.active_allocations);
    println!("   Total allocated: {:.2} MB", stats.total_allocated as f64 / (1024.0 * 1024.0));
    println!("   Staging operations: {}", stats.staging_operations);
    println!("   Transfer bandwidth: {:.2} bytes/s", stats.transfer_bandwidth);
    println!("   Fragmentation: {:.2}%", stats.fragmentation_percentage);
    
    println!("🎉 Day 8 GPU Optimization Integration Test passed!");
    Ok(())
}

/// Helper function to create a test model
fn create_test_model() -> Model {
    Model {
        name: "gpu_test_model".to_string(),
        version: "1.0.0".to_string(),
        input_dim: 1024,
        output_dim: 512, 
        architecture: ModelArchitecture::BitLinear {
            layers: vec![
                LayerConfig {
                    id: 0,
                    layer_type: LayerType::BitLinear,
                    input_shape: vec![1024],
                    output_shape: vec![512],
                    parameters: LayerParameters::BitLinear {
                        weight_bits: 1,
                        activation_bits: 8,
                    },
                }
            ],
            attention_heads: None,
            hidden_dim: 512,
        },
        parameter_count: 1024 * 512,
        quantization_config: QuantizationConfig::default(),
    }
}

/// Helper function to get system memory usage (simplified)
fn get_system_memory_usage() -> usize {
    // Simplified memory usage - in a real implementation this would 
    // query actual system memory usage
    0
}
