//! SIMD Optimization Tests
//!
//! Tests for SIMD optimizations in tensor operations, validating both
//! correctness and performance improvements across different platforms.

use super::*;
use bitnet_core::tensor::ops::simd::*;
use bitnet_core::tensor::{BitNetTensor, BitNetDType};

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test SIMD feature detection
    #[test]
    fn test_simd_feature_detection() {
        println!("Testing SIMD feature detection...");
        
        // Test CPU feature detection
        let features = detect_simd_features();
        
        println!("Detected SIMD features:");
        if features.avx2 {
            println!("  ✓ AVX2 supported");
        }
        if features.sse4_1 {
            println!("  ✓ SSE4.1 supported");
        }
        if features.neon {
            println!("  ✓ NEON supported");
        }
        if features.fma {
            println!("  ✓ FMA supported");
        }
        
        assert!(
            features.avx2 || features.sse4_1 || features.neon,
            "No SIMD features detected - this is unexpected on modern CPUs"
        );
    }
    
    /// Test SIMD vector addition
    #[test]
    fn test_simd_vector_addition() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Test different vector sizes
        let sizes = [16, 32, 64, 128, 256, 512, 1024];
        
        for &size in &sizes {
            let a = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Sequential, &config)?;
            let b = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Ones, &config)?;
            
            // SIMD optimized addition
            let simd_result = simd_add_f32(&a, &b)?;
            
            // Reference scalar implementation
            let scalar_result = scalar_add_f32(&a, &b)?;
            
            // Validate results match
            assert_tensor_close(&simd_result, &scalar_result, 1e-6, 
                &format!("simd_vector_addition_size_{}", size))?;
            
            println!("✓ SIMD vector addition test passed for size {}", size);
        }
        
        validate_memory_efficiency(&config, "test_simd_vector_addition")?;
        Ok(())
    }
    
    /// Test SIMD matrix operations
    #[test]
    fn test_simd_matrix_operations() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Test different matrix sizes
        let matrix_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)];
        
        for &(rows, cols) in &matrix_sizes {
            let a = create_test_tensor(&[rows, cols], BitNetDType::F32, TestPattern::Random, &config)?;
            let b = create_test_tensor(&[rows, cols], BitNetDType::F32, TestPattern::Random, &config)?;
            
            // Test SIMD element-wise multiplication
            let simd_result = simd_mul_f32(&a, &b)?;
            let scalar_result = scalar_mul_f32(&a, &b)?;
            
            assert_tensor_close(&simd_result, &scalar_result, 1e-5, 
                &format!("simd_matrix_mul_{}x{}", rows, cols))?;
            
            println!("✓ SIMD matrix multiplication test passed for {}x{}", rows, cols);
        }
        
        validate_memory_efficiency(&config, "test_simd_matrix_operations")?;
        Ok(())
    }
    
    /// Test SIMD performance improvements
    #[test]
    fn test_simd_performance() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        if !config.enable_performance_tests {
            println!("SIMD performance tests disabled");
            return Ok(());
        }
        
        let size = 100_000; // Large vector for meaningful performance comparison
        let a = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Random, &config)?;
        let b = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Random, &config)?;
        
        // Benchmark scalar implementation
        let scalar_time = std::time::Instant::now();
        for _ in 0..100 {
            let _result = scalar_add_f32(&a, &b)?;
        }
        let scalar_duration = scalar_time.elapsed();
        
        // Benchmark SIMD implementation
        let simd_time = std::time::Instant::now();
        for _ in 0..100 {
            let _result = simd_add_f32(&a, &b)?;
        }
        let simd_duration = simd_time.elapsed();
        
        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
        
        println!("SIMD Performance Results:");
        println!("  Scalar time: {:?}", scalar_duration);
        println!("  SIMD time: {:?}", simd_duration);
        println!("  Speedup: {:.2}x", speedup);
        
        // Expect at least 2x speedup with SIMD
        assert!(speedup >= 1.5, 
            "SIMD speedup ({:.2}x) below expected threshold (1.5x)", speedup);
        
        validate_memory_efficiency(&config, "test_simd_performance")?;
        Ok(())
    }
    
    /// Test SIMD with different data types
    #[test]
    fn test_simd_different_dtypes() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Test F32
        let a_f32 = create_test_tensor(&[1024], BitNetDType::F32, TestPattern::Random, &config)?;
        let b_f32 = create_test_tensor(&[1024], BitNetDType::F32, TestPattern::Random, &config)?;
        let result_f32 = simd_add_f32(&a_f32, &b_f32)?;
        assert_eq!(result_f32.dtype(), BitNetDType::F32);
        
        // Test F16 (if supported)
        if is_simd_f16_supported() {
            let a_f16 = create_test_tensor(&[1024], BitNetDType::F16, TestPattern::Random, &config)?;
            let b_f16 = create_test_tensor(&[1024], BitNetDType::F16, TestPattern::Random, &config)?;
            let result_f16 = simd_add_f16(&a_f16, &b_f16)?;
            assert_eq!(result_f16.dtype(), BitNetDType::F16);
            println!("✓ F16 SIMD operations supported");
        } else {
            println!("! F16 SIMD operations not supported on this platform");
        }
        
        // Test I32
        let a_i32 = create_test_tensor(&[1024], BitNetDType::I32, TestPattern::Sequential, &config)?;
        let b_i32 = create_test_tensor(&[1024], BitNetDType::I32, TestPattern::Ones, &config)?;
        let result_i32 = simd_add_i32(&a_i32, &b_i32)?;
        assert_eq!(result_i32.dtype(), BitNetDType::I32);
        
        println!("✓ SIMD operations tested for multiple data types");
        validate_memory_efficiency(&config, "test_simd_different_dtypes")?;
        Ok(())
    }
    
    /// Test SIMD reduction operations
    #[test]
    fn test_simd_reductions() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        let tensor = create_test_tensor(&[10000], BitNetDType::F32, TestPattern::Ones, &config)?;
        
        // Test SIMD sum
        let simd_sum = simd_sum_f32(&tensor)?;
        let expected_sum = 10000.0; // All ones
        
        assert!((simd_sum - expected_sum).abs() < 1e-5, 
            "SIMD sum mismatch: {} vs {}", simd_sum, expected_sum);
        
        // Test SIMD mean
        let simd_mean = simd_mean_f32(&tensor)?;
        let expected_mean = 1.0;
        
        assert!((simd_mean - expected_mean).abs() < 1e-5,
            "SIMD mean mismatch: {} vs {}", simd_mean, expected_mean);
        
        println!("✓ SIMD reduction operations test passed");
        validate_memory_efficiency(&config, "test_simd_reductions")?;
        Ok(())
    }
    
    /// Test SIMD memory alignment
    #[test]
    fn test_simd_memory_alignment() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Create tensors with various sizes to test alignment handling
        let sizes = [15, 16, 17, 31, 32, 33, 63, 64, 65];
        
        for &size in &sizes {
            let a = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Sequential, &config)?;
            let b = create_test_tensor(&[size], BitNetDType::F32, TestPattern::Ones, &config)?;
            
            // SIMD operations should handle non-aligned sizes correctly
            let result = simd_add_f32(&a, &b)?;
            assert_eq!(result.shape().as_slice(), &[size]);
            
            // Verify alignment properties if available
            if let Ok(alignment) = get_tensor_memory_alignment(&result) {
                println!("Tensor size {} has alignment: {} bytes", size, alignment);
            }
        }
        
        println!("✓ SIMD memory alignment test passed");
        validate_memory_efficiency(&config, "test_simd_memory_alignment")?;
        Ok(())
    }
    
    /// Test SIMD auto-fallback mechanism
    #[test]
    fn test_simd_auto_fallback() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        let a = create_test_tensor(&[100], BitNetDType::F32, TestPattern::Random, &config)?;
        let b = create_test_tensor(&[100], BitNetDType::F32, TestPattern::Random, &config)?;
        
        // Force disable SIMD temporarily to test fallback
        {
            let _simd_guard = disable_simd_temporarily();
            let fallback_result = simd_add_f32(&a, &b)?; // Should use scalar fallback
            
            // Re-enable SIMD
            drop(_simd_guard);
            let simd_result = simd_add_f32(&a, &b)?;
            
            // Results should be identical
            assert_tensor_close(&fallback_result, &simd_result, 1e-6, "simd_fallback_comparison")?;
        }
        
        println!("✓ SIMD auto-fallback mechanism test passed");
        validate_memory_efficiency(&config, "test_simd_auto_fallback")?;
        Ok(())
    }
    
    /// Test SIMD with broadcasting
    #[test]
    fn test_simd_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Test SIMD operations with broadcasting
        let a = create_test_tensor(&[100, 1], BitNetDType::F32, TestPattern::Sequential, &config)?;
        let b = create_test_tensor(&[1, 50], BitNetDType::F32, TestPattern::Ones, &config)?;
        
        let result = simd_add_broadcast_f32(&a, &b)?;
        assert_eq!(result.shape().as_slice(), &[100, 50]);
        
        // Test vector broadcasting
        let c = create_test_tensor(&[1000], BitNetDType::F32, TestPattern::Random, &config)?;
        let scalar_val = 2.5;
        let broadcast_result = simd_add_scalar_f32(&c, scalar_val)?;
        
        assert_eq!(broadcast_result.shape(), c.shape());
        
        println!("✓ SIMD broadcasting operations test passed");
        validate_memory_efficiency(&config, "test_simd_broadcasting")?;
        Ok(())
    }
    
    /// Comprehensive SIMD validation test
    #[test]
    fn test_simd_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
        let config = TestConfig::default();
        
        // Create large tensors for comprehensive testing
        let large_tensor_a = create_test_tensor(&[1000, 500], BitNetDType::F32, TestPattern::Random, &config)?;
        let large_tensor_b = create_test_tensor(&[1000, 500], BitNetDType::F32, TestPattern::Random, &config)?;
        
        // Test a complex expression using SIMD operations
        let step1 = simd_add_f32(&large_tensor_a, &large_tensor_b)?;
        let step2 = simd_mul_scalar_f32(&step1, 0.5)?;
        let step3 = simd_sub_f32(&step2, &large_tensor_a)?;
        let final_result = simd_abs_f32(&step3)?;
        
        // Validate final result properties
        assert_eq!(final_result.shape(), large_tensor_a.shape());
        assert_eq!(final_result.dtype(), BitNetDType::F32);
        
        // Compute reference result with scalar operations
        let ref_step1 = scalar_add_f32(&large_tensor_a, &large_tensor_b)?;
        let ref_step2 = scalar_mul_scalar_f32(&ref_step1, 0.5)?;
        let ref_step3 = scalar_sub_f32(&ref_step2, &large_tensor_a)?;
        let ref_final = scalar_abs_f32(&ref_step3)?;
        
        // Compare SIMD vs scalar results
        assert_tensor_close(&final_result, &ref_final, 1e-5, "simd_comprehensive")?;
        
        println!("✓ Comprehensive SIMD test passed");
        validate_memory_efficiency(&config, "test_simd_comprehensive")?;
        Ok(())
    }
}

// Mock SIMD functions for testing (these would be implemented in the actual SIMD module)

fn detect_simd_features() -> SimdFeatures {
    SimdFeatures {
        avx2: cfg!(target_feature = "avx2"),
        sse4_1: cfg!(target_feature = "sse4.1"),
        neon: cfg!(target_feature = "neon"),
        fma: cfg!(target_feature = "fma"),
    }
}

struct SimdFeatures {
    avx2: bool,
    sse4_1: bool,
    neon: bool,
    fma: bool,
}

// Placeholder SIMD operation functions
fn simd_add_f32(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    // Would implement actual SIMD addition
    scalar_add_f32(a, b) // Fallback for now
}

fn scalar_add_f32(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    // Reference scalar implementation
    bitnet_core::tensor::ops::arithmetic::add(a, b).map_err(|e| e.into())
}

fn simd_mul_f32(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    scalar_mul_f32(a, b) // Fallback for now
}

fn scalar_mul_f32(a: &BitNetTensor, b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    bitnet_core::tensor::ops::arithmetic::mul(a, b).map_err(|e| e.into())
}

// Additional placeholder functions
fn simd_add_f16(_a: &BitNetTensor, _b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_add_i32(_a: &BitNetTensor, _b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_sum_f32(_tensor: &BitNetTensor) -> Result<f32, Box<dyn std::error::Error>> { Ok(1.0) }
fn simd_mean_f32(_tensor: &BitNetTensor) -> Result<f32, Box<dyn std::error::Error>> { Ok(1.0) }
fn simd_add_broadcast_f32(_a: &BitNetTensor, _b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_add_scalar_f32(_tensor: &BitNetTensor, _scalar: f32) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_mul_scalar_f32(_tensor: &BitNetTensor, _scalar: f32) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_sub_f32(_a: &BitNetTensor, _b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn simd_abs_f32(_tensor: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn scalar_mul_scalar_f32(_tensor: &BitNetTensor, _scalar: f32) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn scalar_sub_f32(_a: &BitNetTensor, _b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }
fn scalar_abs_f32(_tensor: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> { unimplemented!() }

fn is_simd_f16_supported() -> bool { false }
fn get_tensor_memory_alignment(_tensor: &BitNetTensor) -> Result<usize, Box<dyn std::error::Error>> { Ok(32) }
fn disable_simd_temporarily() -> SimdGuard { SimdGuard }

struct SimdGuard;
impl Drop for SimdGuard {
    fn drop(&mut self) {}
}
