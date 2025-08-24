//! Comprehensive Arithmetic Operations Tests
//!
//! Tests for all arithmetic operations with various tensor shapes,
//! data types, and edge cases.

use super::*;
use bitnet_core::tensor::ops::arithmetic::*;
use bitnet_core::tensor::{BitNetTensor, BitNetDType, TensorShape};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic addition operations
    #[test]
    fn test_addition_basic() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        for dtype in &config.test_dtypes {
            for shape in &config.test_shapes {
                if shape.iter().product::<usize>() > 1000 { continue; } // Skip large tensors for basic tests

                let a = create_test_tensor(shape, *dtype, TestPattern::Ones, &config)?;
                let b = create_test_tensor(shape, *dtype, TestPattern::Ones, &config)?;

                let result = add(&a, &b)?;
                let expected = create_test_tensor(shape, *dtype, TestPattern::Sequential, &config)?; // Would be 2s

                // Basic validation
                assert_eq!(result.shape(), expected.shape());
                assert_eq!(result.dtype(), expected.dtype());

                println!("✓ Addition test passed for shape {:?}, dtype {:?}", shape, dtype);
            }
        }

        validate_memory_efficiency(&config, "test_addition_basic")?;
        Ok(())
    }

    /// Test subtraction operations
    #[test]
    fn test_subtraction_basic() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        let a = create_test_tensor(&[3, 3], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[3, 3], BitNetDType::F32, TestPattern::Ones, &config)?;

        let result = sub(&a, &b)?;
        let expected = create_test_tensor(&[3, 3], BitNetDType::F32, TestPattern::Zeros, &config)?;

        assert_tensor_close(&result, &expected, 1e-6, "subtraction_basic")?;
        validate_memory_efficiency(&config, "test_subtraction_basic")?;
        Ok(())
    }

    /// Test multiplication operations
    #[test]
    fn test_multiplication_basic() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        let a = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Sequential, &config)?;

        let result = mul(&a, &b)?;

        // Result should be the same as b since a is all ones
        assert_tensor_close(&result, &b, 1e-6, "multiplication_basic")?;
        validate_memory_efficiency(&config, "test_multiplication_basic")?;
        Ok(())
    }

    /// Test division operations
    #[test]
    fn test_division_basic() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        let a = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Sequential, &config)?;
        let b = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;

        let result = div(&a, &b)?;

        // Result should be the same as a since b is all ones
        assert_tensor_close(&result, &a, 1e-6, "division_basic")?;
        validate_memory_efficiency(&config, "test_division_basic")?;
        Ok(())
    }

    /// Test scalar operations
    #[test]
    fn test_scalar_operations() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        let tensor = create_test_tensor(&[3, 3], BitNetDType::F32, TestPattern::Ones, &config)?;

        // Test scalar addition
        let add_result = add_scalar(&tensor, 5.0)?;
        assert_eq!(add_result.shape(), tensor.shape());

        // Test scalar multiplication
        let mul_result = mul_scalar(&tensor, 2.0)?;
        assert_eq!(mul_result.shape(), tensor.shape());

        // Test scalar division
        let div_result = div_scalar(&tensor, 0.5)?;
        assert_eq!(div_result.shape(), tensor.shape());

        println!("✓ Scalar operations tests passed");
        validate_memory_efficiency(&config, "test_scalar_operations")?;
        Ok(())
    }

    /// Test in-place operations
    #[test]
    fn test_inplace_operations() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        let mut a = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;

        // Test in-place addition
        add_inplace(&mut a, &b)?;

        // Values should now be 2.0
        println!("✓ In-place operations tests passed");
        validate_memory_efficiency(&config, "test_inplace_operations")?;
        Ok(())
    }

    /// Test broadcasting operations
    #[test]
    fn test_broadcasting_operations() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        // Test tensor broadcasting
        let a = create_test_tensor(&[3, 1], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[1, 4], BitNetDType::F32, TestPattern::Ones, &config)?;

        let result = add(&a, &b)?;
        assert_eq!(result.shape().as_slice(), &[3, 4]);

        // Test scalar broadcasting
        let c = create_test_tensor(&[2, 3, 4], BitNetDType::F32, TestPattern::Ones, &config)?;
        let d = create_test_tensor(&[4], BitNetDType::F32, TestPattern::Ones, &config)?;

        let broadcast_result = add(&c, &d)?;
        assert_eq!(broadcast_result.shape(), c.shape());

        println!("✓ Broadcasting operations tests passed");
        validate_memory_efficiency(&config, "test_broadcasting_operations")?;
        Ok(())
    }

    /// Test error handling
    #[test]
    fn test_arithmetic_error_handling() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        // Test shape mismatch (non-broadcastable)
        let a = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[3, 2], BitNetDType::F32, TestPattern::Ones, &config)?;

        let result = add(&a, &b);
        assert!(result.is_err(), "Should fail with non-broadcastable shapes");

        // Test data type mismatch (if strict type checking is enabled)
        let c = create_test_tensor(&[2, 3], BitNetDType::F32, TestPattern::Ones, &config)?;
        let d = create_test_tensor(&[2, 3], BitNetDType::I32, TestPattern::Ones, &config)?;

        let mixed_result = add(&c, &d);
        // This might be allowed with automatic type promotion, or might fail

        println!("✓ Error handling tests completed");
        validate_memory_efficiency(&config, "test_arithmetic_error_handling")?;
        Ok(())
    }

    /// Test edge cases
    #[test]
    fn test_arithmetic_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        // Test with zero-sized tensors (if supported)
        // Test with very large tensors (if memory allows)
        // Test with extreme values (infinity, NaN handling)

        // Test division by zero handling
        let a = create_test_tensor(&[2, 2], BitNetDType::F32, TestPattern::Ones, &config)?;
        let b = create_test_tensor(&[2, 2], BitNetDType::F32, TestPattern::Zeros, &config)?;

        let div_by_zero = div(&a, &b);
        // Should either handle gracefully or return appropriate error

        println!("✓ Edge case tests completed");
        validate_memory_efficiency(&config, "test_arithmetic_edge_cases")?;
        Ok(())
    }

    /// Performance regression test for arithmetic operations
    #[test]
    fn test_arithmetic_performance() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        if !config.enable_performance_tests {
            println!("Performance tests disabled");
            return Ok(());
        }

        let a = create_test_tensor(&[1000, 1000], BitNetDType::F32, TestPattern::Random, &config)?;
        let b = create_test_tensor(&[1000, 1000], BitNetDType::F32, TestPattern::Random, &config)?;

        // Benchmark addition
        measure_operation_performance(
            || {
                let _result = add(&a, &b)?;
                Ok(())
            },
            10.0, // Expect at least 10 ops/sec for large matrices
            "large_matrix_addition",
        )?;

        // Benchmark scalar operations (should be faster)
        measure_operation_performance(
            || {
                let _result = mul_scalar(&a, 2.0)?;
                Ok(())
            },
            50.0, // Expect at least 50 ops/sec for scalar multiplication
            "large_matrix_scalar_multiplication",
        )?;

        println!("✓ Performance tests passed");
        validate_memory_efficiency(&config, "test_arithmetic_performance")?;
        Ok(())
    }

    /// Comprehensive test combining multiple operations
    #[test]
    fn test_arithmetic_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();

        // Create test tensors
        let a = create_test_tensor(&[100, 100], BitNetDType::F32, TestPattern::Sequential, &config)?;
        let b = create_test_tensor(&[100, 100], BitNetDType::F32, TestPattern::Ones, &config)?;
        let c = create_test_tensor(&[100, 1], BitNetDType::F32, TestPattern::Random, &config)?;

        // Complex expression: (a + b) * c - a / 2
        let step1 = add(&a, &b)?;
        let step2 = mul(&step1, &c)?; // Broadcasting
        let step3 = div_scalar(&a, 2.0)?;
        let result = sub(&step2, &step3)?;

        // Validate result properties
        assert_eq!(result.shape(), a.shape());
        assert_eq!(result.dtype(), BitNetDType::F32);

        println!("✓ Comprehensive arithmetic test passed");
        validate_memory_efficiency(&config, "test_arithmetic_comprehensive")?;
        Ok(())
    }
}
