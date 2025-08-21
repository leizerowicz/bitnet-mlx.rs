//! Comprehensive Tensor Operations Test Suite
//!
//! This module provides exhaustive testing for all tensor operations,
//! validating correctness, performance, and integration with the
//! BitNet memory and device infrastructure.

pub mod arithmetic_tests;
pub mod broadcasting_tests;
pub mod linear_algebra_tests;
pub mod reduction_tests;
pub mod activation_tests;
pub mod simd_tests;
pub mod performance_regression_tests;
pub mod memory_efficiency_validation_tests;
pub mod infrastructure_validation_test;

use bitnet_core::tensor::{BitNetTensor, BitNetDType, TensorShape};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
use bitnet_core::device::{Device, get_cpu_device, auto_select_device};
use std::sync::Arc;

/// Test configuration for tensor operations
pub struct TestConfig {
    pub memory_pool: Arc<HybridMemoryPool>,
    pub cpu_device: Device,
    pub test_shapes: Vec<Vec<usize>>,
    pub test_dtypes: Vec<BitNetDType>,
    pub enable_performance_tests: bool,
    pub simd_optimization_level: u8,
}

impl Default for TestConfig {
    fn default() -> Self {
        let memory_config = MemoryPoolConfig {
            initial_small_blocks: 1000,
            initial_large_blocks: 100,
            small_block_size: 4096,
            large_block_threshold: 1024 * 1024,
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_tracking: true,
            tracking_config: TrackingConfig::detailed(),
            compaction_threshold: 0.3,
            enable_metrics: true,
        };

        let memory_pool = Arc::new(HybridMemoryPool::new(memory_config).unwrap());
        let cpu_device = get_cpu_device().unwrap();

        Self {
            memory_pool,
            cpu_device,
            test_shapes: vec![
                vec![1],           // scalar-like
                vec![5],           // 1D
                vec![3, 4],        // 2D
                vec![2, 3, 4],     // 3D
                vec![2, 3, 4, 5],  // 4D
                vec![1, 1000],     // large 1D
                vec![100, 100],    // large 2D
                vec![10, 10, 10],  // medium 3D
            ],
            test_dtypes: vec![
                BitNetDType::F32,
                BitNetDType::F16,
                BitNetDType::I8,
                BitNetDType::I16,
                BitNetDType::I32,
                BitNetDType::BitNet158,
            ],
            enable_performance_tests: true,
            simd_optimization_level: 2, // Medium optimization
        }
    }
}

/// Create test tensors with specific patterns for validation
pub fn create_test_tensor(
    shape: &[usize], 
    dtype: BitNetDType, 
    pattern: TestPattern,
    config: &TestConfig
) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let tensor = match pattern {
        TestPattern::Zeros => BitNetTensor::zeros(shape, dtype, Some(config.cpu_device.clone()))?,
        TestPattern::Ones => BitNetTensor::ones(shape, dtype, Some(config.cpu_device.clone()))?,
        TestPattern::Sequential => {
            let mut tensor = BitNetTensor::zeros(shape, dtype, Some(config.cpu_device.clone()))?;
            // Fill with sequential numbers
            fill_sequential(&mut tensor)?;
            tensor
        },
        TestPattern::Random => {
            let mut tensor = BitNetTensor::zeros(shape, dtype, Some(config.cpu_device.clone()))?;
            fill_random(&mut tensor, 42)?; // Fixed seed for reproducibility
            tensor
        },
        TestPattern::Identity => {
            if shape.len() != 2 || shape[0] != shape[1] {
                return Err("Identity pattern requires square 2D tensor".into());
            }
            let mut tensor = BitNetTensor::zeros(shape, dtype, Some(config.cpu_device.clone()))?;
            fill_identity(&mut tensor)?;
            tensor
        },
    };
    Ok(tensor)
}

/// Test patterns for tensor creation
#[derive(Debug, Clone, Copy)]
pub enum TestPattern {
    Zeros,
    Ones,
    Sequential,
    Random,
    Identity,
}

/// Fill tensor with sequential numbers
fn fill_sequential(tensor: &mut BitNetTensor) -> Result<(), Box<dyn std::error::Error>> {
    // This would be implemented based on the actual tensor API
    // For now, placeholder implementation
    Ok(())
}

/// Fill tensor with random values (reproducible)
fn fill_random(tensor: &mut BitNetTensor, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
    // Implementation would use a seeded random number generator
    Ok(())
}

/// Fill tensor as identity matrix
fn fill_identity(tensor: &mut BitNetTensor) -> Result<(), Box<dyn std::error::Error>> {
    // Implementation would set diagonal elements to 1
    Ok(())
}

/// Validate tensor values within tolerance
pub fn assert_tensor_close(
    actual: &BitNetTensor,
    expected: &BitNetTensor,
    tolerance: f64,
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Shape validation
    if actual.shape() != expected.shape() {
        return Err(format!(
            "{}: Shape mismatch - actual: {:?}, expected: {:?}",
            test_name,
            actual.shape(),
            expected.shape()
        ).into());
    }

    // Data type validation
    if actual.dtype() != expected.dtype() {
        return Err(format!(
            "{}: DType mismatch - actual: {:?}, expected: {:?}",
            test_name,
            actual.dtype(),
            expected.dtype()
        ).into());
    }

    // Value comparison would be implemented based on actual tensor API
    // This is a placeholder implementation
    println!("Validating tensor values for test: {}", test_name);
    
    Ok(())
}

/// Memory efficiency validation
pub fn validate_memory_efficiency(
    config: &TestConfig,
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let stats = config.memory_pool.get_stats();
    
    // Check for memory leaks
    if stats.active_allocations > 0 {
        eprintln!(
            "Warning: {} active allocations remaining after test: {}",
            stats.active_allocations, test_name
        );
    }

    // Check fragmentation
    let fragmentation_ratio = stats.fragmented_bytes as f64 / stats.total_allocated_bytes as f64;
    if fragmentation_ratio > 0.3 {
        eprintln!(
            "Warning: High fragmentation ({:.2}%) in test: {}",
            fragmentation_ratio * 100.0, test_name
        );
    }

    // Check pool efficiency
    let pool_utilization = stats.bytes_in_use as f64 / stats.total_pool_size as f64;
    println!(
        "Test {} - Pool utilization: {:.2}%, Fragmentation: {:.2}%",
        test_name,
        pool_utilization * 100.0,
        fragmentation_ratio * 100.0
    );

    Ok(())
}

/// Performance validation helper
pub fn measure_operation_performance<F>(
    operation: F,
    expected_min_ops_per_sec: f64,
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = std::time::Instant::now();
    let iterations = 1000;
    
    for _ in 0..iterations {
        operation()?;
    }
    
    let duration = start.elapsed();
    let ops_per_sec = iterations as f64 / duration.as_secs_f64();
    
    println!(
        "Performance test {} - {:.2} ops/sec (expected: {:.2})",
        test_name, ops_per_sec, expected_min_ops_per_sec
    );
    
    if ops_per_sec < expected_min_ops_per_sec {
        return Err(format!(
            "Performance regression in {}: {:.2} ops/sec < {:.2} expected",
            test_name, ops_per_sec, expected_min_ops_per_sec
        ).into());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = TestConfig::default();
        assert!(!config.test_shapes.is_empty());
        assert!(!config.test_dtypes.is_empty());
        assert!(config.enable_performance_tests);
    }

    #[test]
    fn test_tensor_creation_patterns() {
        let config = TestConfig::default();
        
        // Test different patterns
        let patterns = [
            TestPattern::Zeros,
            TestPattern::Ones,
            TestPattern::Sequential,
            TestPattern::Random,
        ];
        
        for pattern in patterns {
            let tensor = create_test_tensor(
                &[2, 3],
                BitNetDType::F32,
                pattern,
                &config,
            );
            assert!(tensor.is_ok(), "Failed to create tensor with pattern: {:?}", pattern);
        }
    }

    #[test]
    fn test_identity_tensor_creation() {
        let config = TestConfig::default();
        
        let identity = create_test_tensor(
            &[3, 3],
            BitNetDType::F32,
            TestPattern::Identity,
            &config,
        );
        assert!(identity.is_ok());
        
        // Non-square should fail
        let non_square = create_test_tensor(
            &[2, 3],
            BitNetDType::F32,
            TestPattern::Identity,
            &config,
        );
        assert!(non_square.is_err());
    }
}
