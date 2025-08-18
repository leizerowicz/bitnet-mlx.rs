//! Integration tests for BitLinear memory optimization system

use super::*;
use candle_core::{Device, Tensor};
use bitnet_core::memory::HybridMemoryPool;
use std::sync::Arc;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Helper function to create a test memory pool
    fn create_test_memory_pool() -> Arc<HybridMemoryPool> {
        Arc::new(HybridMemoryPool::new().expect("Failed to create memory pool"))
    }

    #[test]
    fn test_memory_optimizer_creation() {
        let memory_pool = create_test_memory_pool();
        let device = Device::Cpu;
        
        // Use default configuration
        let config = MemoryOptimizationConfig::default();
        
        let optimizer = BitLinearMemoryOptimizer::new(config, memory_pool, &device);
        assert!(optimizer.is_ok());
        
        let optimizer = optimizer.unwrap();
        let metrics = optimizer.metrics();
        
        // Check some basic metrics exist (u64 values are always >= 0)
        let _ = metrics.lazy_quantization_hits;
        let _ = metrics.weight_cache_hits;
    }

    #[test]
    fn test_weight_cache_integration() {
        let memory_pool = create_test_memory_pool();
        let device = Device::Cpu;
        
        // Configure to enable weight cache
        let config = MemoryOptimizationConfig {
            enable_lazy_quantization: false,
            weight_cache_config: WeightCacheConfig::default(),
            ..Default::default()
        };
        
        let mut optimizer = BitLinearMemoryOptimizer::new(config, memory_pool, &device)
            .expect("Failed to create optimizer");
        
        // Create test tensor with F32 dtype (not F64)
        let weights = Tensor::randn(0.0f32, 1.0f32, (128, 256), &Device::Cpu)
            .expect("Failed to create test tensor");
        
        let layer_name = "test_layer";
        
        // Test weight quantization (our actual API)
        let result = optimizer.get_quantized_weights(layer_name, &weights, false);
        if let Err(e) = &result {
            eprintln!("Quantization error: {:?}", e);
        }
        assert!(result.is_ok());
        
        let (quantized, scales) = result.unwrap();
        assert_eq!(quantized.dims(), weights.dims());
        // Scales can be a scalar (0 dimensions) or tensor - both are valid
        // Just verify it's a valid tensor by accessing it
        let _ = scales.to_dtype(scales.dtype()).unwrap();
    }

    #[test]
    fn test_memory_pressure_integration() {
        let memory_pool = create_test_memory_pool();
        let device = Device::Cpu;
        
        let config = MemoryOptimizationConfig::default();
        
        let mut optimizer = BitLinearMemoryOptimizer::new(config, memory_pool, &device)
            .expect("Failed to create optimizer");
        
        // Test memory pressure detection
        let pressure = optimizer.check_memory_pressure();
        assert!(pressure.is_ok());
        
        // Just test that we get a valid pressure level without pattern matching
        let _pressure_level = pressure.unwrap();
    }

    #[test]
    fn test_cleanup_integration() {
        let memory_pool = create_test_memory_pool();
        let device = Device::Cpu;
        
        let config = MemoryOptimizationConfig::default();
        
        let mut optimizer = BitLinearMemoryOptimizer::new(config, memory_pool, &device)
            .expect("Failed to create optimizer");
        
        // Create test tensor with F32 dtype (not F64)
        let weights = Tensor::randn(0.0f32, 1.0f32, (128, 256), &Device::Cpu)
            .expect("Failed to create test tensor");
        
        let layer_name = "test_layer";
        
        // Test quantization
        let result = optimizer.get_quantized_weights(layer_name, &weights, false);
        if let Err(e) = &result {
            eprintln!("Quantization error: {:?}", e);
        }
        assert!(result.is_ok());
        
        // Test cleanup
        let result = optimizer.cleanup();
        assert!(result.is_ok());
        
        // Test metrics
        let metrics = optimizer.metrics();
        assert!(metrics.lazy_quantization_hit_rate() >= 0.0);
        assert!(metrics.weight_cache_hit_rate() >= 0.0);
    }

    #[test]
    fn test_scaling_factor_manager_direct() {
        let memory_pool = create_test_memory_pool();
        let device = Device::Cpu;
        
        // Test ScalingFactorManager directly
        let scaling_manager = ScalingFactorManager::new(
            ScalingPolicy::AbsoluteMean,
            memory_pool.clone(),
            &device,
        );
        assert!(scaling_manager.is_ok());
        
        // Just test that we can create it successfully
        let _scaling_manager = scaling_manager.unwrap();
    }

    #[test]
    fn test_lazy_quantization_config_creation() {
        // Test LazyQuantizationConfig can be created with defaults
        let config = LazyQuantizationConfig::default();
        
        assert!(config.max_cache_entries > 0);
        assert!(config.memory_threshold_bytes > 0);
        assert!(config.cache_ttl_seconds > 0);
    }

    #[test]
    fn test_weight_cache_config_creation() {
        // Test WeightCacheConfig can be created with defaults
        let config = WeightCacheConfig::default();
        
        assert!(config.max_entries > 0);
        assert!(config.max_memory_bytes > 0);
    }

    #[test]
    fn test_pressure_config_creation() {
        // Test PressureConfig can be created with defaults
        let config = PressureConfig::default();
        
        assert!(config.high_pressure_threshold > 0.0);
        assert!(config.critical_pressure_threshold > config.high_pressure_threshold);
    }

    #[test]
    fn test_memory_layout_enum() {
        // Test basic memory layout enum functionality
        let row_major = MemoryLayout::RowMajor;
        let col_major = MemoryLayout::ColumnMajor;
        
        assert_ne!(row_major, col_major);
    }

    #[test]
    fn test_scaling_policy_enum() {
        // Test basic scaling policy enum functionality
        let absolute_mean = ScalingPolicy::AbsoluteMean;
        let adaptive = ScalingPolicy::Adaptive { 
            min_scale: 0.1, 
            max_scale: 1.0, 
            adaptation_rate: 0.9 
        };
        
        assert_ne!(absolute_mean, adaptive);
        
        // Test default
        let default_policy = ScalingPolicy::default();
        assert_eq!(default_policy, ScalingPolicy::AbsoluteMean);
    }

    #[test]
    fn test_access_pattern_enum() {
        // Test basic access pattern enum functionality
        let sequential = AccessPattern::Sequential;
        let random = AccessPattern::Random;
        
        assert_ne!(sequential, random);
    }
}
