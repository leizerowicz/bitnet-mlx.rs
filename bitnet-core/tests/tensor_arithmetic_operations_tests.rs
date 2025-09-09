//! Comprehensive tests for BitNet tensor arithmetic operations
//!
//! This test suite validates all arithmetic operations including:
//! - Basic element-wise operations
//! - Broadcasting operations
//! - In-place operations
//! - Scalar operations
//! - Error handling
//! - Memory efficiency

use bitnet_core::memory::{HybridMemoryPool, TrackingConfig};
use bitnet_core::tensor::ops::arithmetic::*;
use bitnet_core::tensor::ops::broadcasting::*;
use bitnet_core::tensor::{BitNetDType, BitNetTensor};
use std::sync::Mutex;

// Global mutex to prevent race conditions in memory pool tests
static MEMORY_POOL_MUTEX: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    /// RAII context for isolated test memory pool management
    /// Automatically cleans up global pool state when dropped
    struct TestMemoryContext {
        /// The isolated memory pool for this test
        /// Keeping this Arc alive ensures pool remains valid during test
        _pool: std::sync::Arc<HybridMemoryPool>,
        /// Mutex guard to prevent race conditions between tests
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for TestMemoryContext {
        fn drop(&mut self) {
            // Don't clear the global pool - just let it be
            // Each test will set its own isolated pool anyway
            // Clearing it causes "Global memory pool not available" errors
            // The mutex guard will be automatically released
        }
    }

    /// Setup isolated memory pool for a single test
    /// Returns context that automatically cleans up on drop
    fn setup_isolated_memory_pool() -> Result<TestMemoryContext, Box<dyn std::error::Error>> {
        // Acquire mutex to prevent race conditions between tests
        let guard = MEMORY_POOL_MUTEX.lock().unwrap();
        
        // Clear any existing global pool first to avoid conflicts
        bitnet_core::tensor::memory_integration::clear_global_memory_pool();
        
        // Create identical memory pool configuration as original
        let tracking_config = TrackingConfig::detailed();
        let mut config = bitnet_core::memory::MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(tracking_config);

        // Create isolated memory pool instance for this test
        let memory_pool = std::sync::Arc::new(HybridMemoryPool::with_config(config)?);
        
        // Set this isolated pool as the global pool for test duration
        bitnet_core::tensor::memory_integration::set_global_memory_pool(
            std::sync::Arc::downgrade(&memory_pool)
        );
        
        // Verify the global pool was set successfully
        if bitnet_core::tensor::memory_integration::get_global_memory_pool().is_none() {
            return Err("Failed to set global memory pool".into());
        }
        
        // Return context that keeps pool alive and manages cleanup
        Ok(TestMemoryContext {
            _pool: memory_pool,
            _guard: guard, // This keeps the mutex locked for the duration of the test
        })
    }

    #[test]
    fn test_basic_addition() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;

        let result = add(&a, &b)?;

        assert_eq!(result.shape().dims(), &[2, 3]);
        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.element_count(), 6);

        Ok(())
    }

    #[test]
    fn test_basic_subtraction() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[3, 2], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[3, 2], BitNetDType::F32, None)?;

        let result = sub(&a, &b)?;

        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_basic_multiplication() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None)?;

        let result = mul(&a, &b)?;

        assert_eq!(result.shape().dims(), &[4, 4]);
        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.element_count(), 16);

        Ok(())
    }

    #[test]
    fn test_basic_division() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

        let result = div(&a, &b)?;

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_basic_remainder() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;

        let result = rem(&a, &b)?;

        assert_eq!(result.shape().dims(), &[3, 3]);
        assert_eq!(result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_power_operation() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let base = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let exponent = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

        let result = pow(&base, &exponent)?;

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_broadcasting_addition() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 4], BitNetDType::F32, None)?;

        // First check if broadcasting is possible
        assert!(can_broadcast(&a, &b)?);

        let broadcast_shape = compute_broadcast_shape(&a, &b)?;
        assert_eq!(broadcast_shape, vec![3, 4]);

        let result = add(&a, &b)?;
        assert_eq!(result.shape().dims(), &[3, 4]);

        Ok(())
    }

    #[test]
    fn test_complex_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 3, 1], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 1, 4], BitNetDType::F32, None)?;

        assert!(can_broadcast(&a, &b)?);

        let broadcast_shape = compute_broadcast_shape(&a, &b)?;
        assert_eq!(broadcast_shape, vec![2, 3, 4]);

        let result = add(&a, &b)?;
        assert_eq!(result.shape().dims(), &[2, 3, 4]);
        assert_eq!(result.element_count(), 24);

        Ok(())
    }

    #[test]
    fn test_scalar_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let tensor = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;
        let scalar = BitNetTensor::ones(&[1, 1], BitNetDType::F32, None)?;

        let result = add(&tensor, &scalar)?;

        assert_eq!(result.shape().dims(), &[3, 3]);
        assert_eq!(result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_inplace_addition() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let mut a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;

        let original_id = a.tensor_id();

        add_(&mut a, &b)?;

        assert_eq!(a.shape().dims(), &[2, 3]);
        assert_eq!(a.dtype(), BitNetDType::F32);
        // Tensor ID should remain the same for in-place operations
        assert_eq!(a.tensor_id(), original_id);

        Ok(())
    }

    #[test]
    fn test_inplace_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let mut a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 4], BitNetDType::F32, None)?;

        add_(&mut a, &b)?;

        assert_eq!(a.shape().dims(), &[3, 4]);

        Ok(())
    }

    #[test]
    fn test_all_inplace_operations() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let mut tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let operand = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

        let original_shape = tensor.shape().dims().to_vec();

        add_(&mut tensor, &operand)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        sub_(&mut tensor, &operand)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        mul_(&mut tensor, &operand)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        div_(&mut tensor, &operand)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        rem_(&mut tensor, &operand)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        Ok(())
    }

    #[test]
    fn test_scalar_operations() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let tensor = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;

        let add_result = add_scalar(&tensor, 5.0)?;
        assert_eq!(add_result.shape().dims(), &[3, 3]);
        assert_eq!(add_result.dtype(), BitNetDType::F32);

        let sub_result = sub_scalar(&tensor, 2.0)?;
        assert_eq!(sub_result.shape().dims(), &[3, 3]);

        let mul_result = mul_scalar(&tensor, 3.0)?;
        assert_eq!(mul_result.shape().dims(), &[3, 3]);

        let div_result = div_scalar(&tensor, 2.0)?;
        assert_eq!(div_result.shape().dims(), &[3, 3]);

        Ok(())
    }

    #[test]
    fn test_inplace_scalar_operations() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let mut tensor = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let original_shape = tensor.shape().dims().to_vec();
        let original_id = tensor.tensor_id();

        add_scalar_(&mut tensor, 10.0)?;
        assert_eq!(tensor.shape().dims(), original_shape);
        assert_eq!(tensor.tensor_id(), original_id);

        sub_scalar_(&mut tensor, 5.0)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        mul_scalar_(&mut tensor, 2.0)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        div_scalar_(&mut tensor, 3.0)?;
        assert_eq!(tensor.shape().dims(), original_shape);

        Ok(())
    }

    #[test]
    fn test_operator_overloading() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

        // Test + operator
        let add_result = (&a + &b)?;
        assert_eq!(add_result.shape().dims(), &[2, 2]);

        // Test - operator
        let sub_result = (&a - &b)?;
        assert_eq!(sub_result.shape().dims(), &[2, 2]);

        // Test * operator
        let mul_result = (&a * &b)?;
        assert_eq!(mul_result.shape().dims(), &[2, 2]);

        // Test / operator
        let div_result = (&a / &b)?;
        assert_eq!(div_result.shape().dims(), &[2, 2]);

        // Test % operator
        let rem_result = (&a % &b)?;
        assert_eq!(rem_result.shape().dims(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_different_data_types() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        // Test F32
        let f32_a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let f32_b = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let f32_result = add(&f32_a, &f32_b)?;
        assert_eq!(f32_result.dtype(), BitNetDType::F32);

        // Test I32
        let i32_a = BitNetTensor::zeros(&[2, 2], BitNetDType::I32, None)?;
        let i32_b = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;
        let i32_result = add(&i32_a, &i32_b)?;
        assert_eq!(i32_result.dtype(), BitNetDType::I32);

        // Test I8
        let i8_a = BitNetTensor::zeros(&[3, 3], BitNetDType::I8, None)?;
        let i8_b = BitNetTensor::ones(&[3, 3], BitNetDType::I8, None)?;
        let i8_result = mul(&i8_a, &i8_b)?;
        assert_eq!(i8_result.dtype(), BitNetDType::I8);

        Ok(())
    }

    #[test]
    fn test_data_type_mismatch_error() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let f32_tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let i32_tensor = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;

        let result = add(&f32_tensor, &i32_tensor);
        assert!(result.is_err(), "Expected error for data type mismatch");

        Ok(())
    }

    #[test]
    fn test_broadcasting_incompatible_error() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[4, 2], BitNetDType::F32, None)?;

        // First check broadcasting compatibility
        assert!(!can_broadcast(&a, &b)?);

        let result = add(&a, &b);
        assert!(
            result.is_err(),
            "Expected error for incompatible broadcasting"
        );

        Ok(())
    }

    #[test]
    fn test_division_by_zero_error() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

        let result = div_scalar(&tensor, 0.0);
        assert!(result.is_err(), "Expected error for division by zero");

        let mut mutable_tensor = tensor.clone();
        let result2 = div_scalar_(&mut mutable_tensor, 0.0);
        assert!(
            result2.is_err(),
            "Expected error for in-place division by zero"
        );

        Ok(())
    }

    #[test]
    fn test_inplace_shape_mismatch_error() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let mut a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;

        // This should fail because in-place requires result shape to match lhs shape
        let result = add_(&mut a, &b);
        assert!(
            result.is_err(),
            "Expected error for in-place shape mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_power_unsupported_dtype_error() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let base = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;
        let exp = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;

        let result = pow(&base, &exp);
        assert!(
            result.is_err(),
            "Expected error for unsupported power operation on I32"
        );

        Ok(())
    }

    #[test]
    fn test_broadcast_analysis() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[3, 1, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 2, 4], BitNetDType::F32, None)?;

        let broadcast_info = analyze_broadcast(&a, &b)?;

        assert_eq!(broadcast_info.broadcast_shape, vec![3, 2, 4]);
        assert_eq!(broadcast_info.result_elements, 24);

        // Check strides are calculated
        assert_eq!(broadcast_info.left_strides.len(), 3);
        assert_eq!(broadcast_info.right_strides.len(), 3);

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        // Test with larger tensors to verify memory efficiency
        let large_a = BitNetTensor::ones(&[100, 100], BitNetDType::F32, None)?;
        let large_b = BitNetTensor::ones(&[100, 100], BitNetDType::F32, None)?;

        let result = add(&large_a, &large_b)?;

        assert_eq!(result.element_count(), 10000);
        assert_eq!(result.size_bytes(), 10000 * 4); // F32 = 4 bytes

        // Test broadcasting memory efficiency
        let broadcast_small = BitNetTensor::ones(&[1, 100], BitNetDType::F32, None)?;
        let broadcast_result = add(&large_a, &broadcast_small)?;

        assert_eq!(broadcast_result.element_count(), 10000);

        Ok(())
    }

    #[test]
    fn test_chained_operations() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
        let c = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;

        // Test chained operations: (a + b) * c
        let intermediate = add(&a, &b)?;
        let final_result = mul(&intermediate, &c)?;

        assert_eq!(final_result.shape().dims(), &[2, 3]);
        assert_eq!(final_result.dtype(), BitNetDType::F32);

        Ok(())
    }

    #[test]
    fn test_complex_broadcasting_chains() -> Result<(), Box<dyn std::error::Error>> {
        let _context = setup_isolated_memory_pool()?;

        let a = BitNetTensor::ones(&[2, 1, 3], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 4, 1], BitNetDType::F32, None)?;
        let c = BitNetTensor::ones(&[1, 1, 1], BitNetDType::F32, None)?;

        // Complex chain: ((a + b) * c) - a
        let step1 = add(&a, &b)?; // Should broadcast to [2, 4, 3]
        let step2 = mul(&step1, &c)?; // Should remain [2, 4, 3]
        let final_result = sub(&step2, &a)?; // Should remain [2, 4, 3]

        assert_eq!(final_result.shape().dims(), &[2, 4, 3]);
        assert_eq!(final_result.element_count(), 24);

        Ok(())
    }
}
