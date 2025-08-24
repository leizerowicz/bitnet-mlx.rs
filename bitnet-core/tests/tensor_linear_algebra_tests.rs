//! Linear Algebra Operations Tests
//!
//! Comprehensive tests for BitNet tensor linear algebra operations including
//! matrix multiplication, dot products, decompositions, and device compatibility.

use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::tensor::ops::linear_algebra::*;
use bitnet_core::tensor::{memory_integration::set_global_memory_pool, BitNetDType, BitNetTensor};
use candle_core::Device;
use std::sync::Arc;

#[cfg(test)]
mod linear_algebra_tests {
    use super::*;

    // Helper function to set up memory pool for tests
    fn setup_memory_pool() -> Arc<HybridMemoryPool> {
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        memory_pool // Return the Arc to keep it alive
    }

    // ============================================================================
    // Matrix Multiplication Tests
    // ============================================================================

    #[test]
    fn test_matmul_basic() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4, 2], BitNetDType::F32, None).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.dtype(), BitNetDType::F32);
    }

    #[test]
    fn test_matmul_square_matrices() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5, 5], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[5, 5], BitNetDType::F32, None).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[5, 5]);
    }

    #[test]
    fn test_matmul_different_sizes() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 10], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[10, 3], BitNetDType::F32, None).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_matmul_large_matrices() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[128, 256], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[256, 64], BitNetDType::F32, None).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[128, 64]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[5, 2], BitNetDType::F32, None).unwrap();

        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_strategy_selection() {
        let _pool = setup_memory_pool(); // Keep pool alive
                                         // Small matrices should use standard strategy
        let _a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let _b = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();

        // TODO: Re-enable when select_optimal_matmul_strategy is properly exposed
        // let config = select_optimal_matmul_strategy(&a, &b);
        // Could be Standard or DeviceOptimized depending on device
        // assert!(matches!(
        //     config.strategy,
        //     MatMulStrategy::Standard | MatMulStrategy::DeviceOptimized | MatMulStrategy::SimdAccelerated
        // ));

        // Medium matrices should use blocked strategy
        let _a = BitNetTensor::ones(&[128, 128], BitNetDType::F32, None).unwrap();
        let _b = BitNetTensor::ones(&[128, 128], BitNetDType::F32, None).unwrap();

        // TODO: Re-enable when select_optimal_matmul_strategy is properly exposed
        // let config = select_optimal_matmul_strategy(&a, &b);
        // assert!(matches!(
        //     config.strategy,
        //     MatMulStrategy::Blocked | MatMulStrategy::DeviceOptimized | MatMulStrategy::Tiled
        // ));
    }

    #[test]
    fn test_batched_matmul() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[2, 4, 5], BitNetDType::F32, None).unwrap();

        let result = batched_matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn test_batched_matmul_batch_mismatch() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[3, 4, 5], BitNetDType::F32, None).unwrap();

        assert!(batched_matmul(&a, &b).is_err());
    }

    // ============================================================================
    // Dot Product Tests
    // ============================================================================

    #[test]
    fn test_dot_product_vectors() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();

        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[] as &[usize]); // Scalar result
        assert_eq!(result.dtype(), BitNetDType::F32);
    }

    #[test]
    fn test_dot_product_small_vectors() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();

        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_dot_product_dimension_mismatch() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[50], BitNetDType::F32, None).unwrap();

        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn test_dot_product_multidimensional() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();

        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_outer_product() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4], BitNetDType::F32, None).unwrap();

        let result = outer(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_outer_product_non_vector() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4], BitNetDType::F32, None).unwrap();

        assert!(outer(&a, &b).is_err());
    }

    // ============================================================================
    // Transpose Tests
    // ============================================================================

    #[test]
    fn test_transpose_2d() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None).unwrap();
        let result = transpose(&a).unwrap();
        assert_eq!(result.shape().dims(), &[4, 3]);
    }

    #[test]
    fn test_transpose_square() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5, 5], BitNetDType::F32, None).unwrap();
        let result = transpose(&a).unwrap();
        assert_eq!(result.shape().dims(), &[5, 5]);
    }

    #[test]
    fn test_transpose_3d() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        let result = transpose(&a).unwrap();
        assert_eq!(result.shape().dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_transpose_1d_error() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();
        assert!(transpose(&a).is_err());
    }

    #[test]
    fn test_permute() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        let result = permute(&a, &[2, 0, 1]).unwrap();
        assert_eq!(result.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_permute_invalid() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None).unwrap();
        // Invalid permutation (missing dimension 1)
        assert!(permute(&a, &[2, 0, 3]).is_err());
    }

    // ============================================================================
    // Utility Function Tests
    // ============================================================================

    #[test]
    fn test_eye() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let result = eye(5, BitNetDType::F32, None).unwrap();
        assert_eq!(result.shape().dims(), &[5, 5]);
        assert_eq!(result.dtype(), BitNetDType::F32);
    }

    #[test]
    fn test_eye_different_sizes() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let result = eye(10, BitNetDType::F32, None).unwrap();
        assert_eq!(result.shape().dims(), &[10, 10]);

        let result = eye(1, BitNetDType::F32, None).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1]);
    }

    // ============================================================================
    // Advanced Linear Algebra Tests (Placeholders)
    // ============================================================================

    #[test]
    fn test_svd() {
        let _pool = setup_memory_pool(); // Keep pool alive

        let a = BitNetTensor::ones(&[5, 3], BitNetDType::F32, None).unwrap();
        let (u, s, vt) = svd(&a).unwrap();

        assert_eq!(u.shape().dims(), &[5, 5]);
        assert_eq!(s.shape().dims(), &[3]);
        assert_eq!(vt.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_svd_square() {
        let _pool = setup_memory_pool(); // Keep pool alive

        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let (u, s, vt) = svd(&a).unwrap();

        assert_eq!(u.shape().dims(), &[4, 4]);
        assert_eq!(s.shape().dims(), &[4]);
        assert_eq!(vt.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_svd_non_2d() {
        let _pool = setup_memory_pool(); // Keep pool alive

        let a = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();
        assert!(svd(&a).is_err());
    }

    #[test]
    fn test_qr() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5, 3], BitNetDType::F32, None).unwrap();
        let (q, r) = qr(&a).unwrap();

        assert_eq!(q.shape().dims(), &[5, 5]);
        assert_eq!(r.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_qr_square() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let (q, r) = qr(&a).unwrap();

        assert_eq!(q.shape().dims(), &[4, 4]);
        assert_eq!(r.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_cholesky() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let result = cholesky(&a).unwrap();
        assert_eq!(result.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_cholesky_non_square() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 3], BitNetDType::F32, None).unwrap();
        assert!(cholesky(&a).is_err());
    }

    #[test]
    fn test_eig() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let (eigenvals, eigenvecs) = eig(&a).unwrap();

        assert_eq!(eigenvals.shape().dims(), &[4]);
        assert_eq!(eigenvecs.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_det() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let result = det(&a).unwrap();
        assert_eq!(result.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_inv() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let result = inv(&a).unwrap();
        assert_eq!(result.shape().dims(), &[4, 4]);
    }

    // ============================================================================
    // Configuration Tests
    // ============================================================================

    #[test]
    fn test_matmul_with_config() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4, 4], BitNetDType::F32, None).unwrap();

        let config = MatMulConfig {
            strategy: MatMulStrategy::Standard,
            block_size: 32,
            use_simd: false,
            use_device_optimization: false,
            prefer_row_major: true,
        };

        let result = matmul_with_config(&a, &b, &config).unwrap();
        assert_eq!(result.shape().dims(), &[4, 4]);
    }

    #[test]
    fn test_matmul_config_blocked() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[8, 8], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[8, 8], BitNetDType::F32, None).unwrap();

        let config = MatMulConfig {
            strategy: MatMulStrategy::Blocked,
            block_size: 4,
            use_simd: true,
            use_device_optimization: false,
            prefer_row_major: true,
        };

        let result = matmul_with_config(&a, &b, &config).unwrap();
        assert_eq!(result.shape().dims(), &[8, 8]);
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    #[test]
    fn test_validation_errors() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4, 5], BitNetDType::F32, None).unwrap();

        // Incompatible dimensions for matrix multiplication
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_transpose_validation() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();

        // Cannot transpose 1D tensor
        assert!(transpose(&a).is_err());
    }

    #[test]
    fn test_dot_validation() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let a = BitNetTensor::ones(&[5], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[3], BitNetDType::F32, None).unwrap();

        // Dimension mismatch
        assert!(dot(&a, &b).is_err());
    }

    // ============================================================================
    // Performance and Memory Tests
    // ============================================================================

    #[test]
    fn test_large_matrix_operations() {
        let _pool = setup_memory_pool(); // Keep pool alive
                                         // Test that large matrices can be created and operated on without panics
        let a = BitNetTensor::ones(&[256, 128], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[128, 64], BitNetDType::F32, None).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[256, 64]);
    }

    #[test]
    fn test_memory_efficiency() {
        let _pool = setup_memory_pool(); // Keep pool alive
                                         // Test that operations don't leak memory by creating many tensors
        for _ in 0..100 {
            let a = BitNetTensor::ones(&[10, 10], BitNetDType::F32, None).unwrap();
            let b = BitNetTensor::ones(&[10, 10], BitNetDType::F32, None).unwrap();
            let _result = matmul(&a, &b).unwrap();
        }
    }

    // ============================================================================
    // Device Compatibility Tests
    // ============================================================================

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_device_operations() {
        let _pool = setup_memory_pool(); // Keep pool alive
        if let Ok(device) = Device::new_metal(0) {
            let a = BitNetTensor::ones(&[32, 32], BitNetDType::F32, Some(device.clone())).unwrap();
            let b = BitNetTensor::ones(&[32, 32], BitNetDType::F32, Some(device.clone())).unwrap();

            let result = matmul(&a, &b).unwrap();
            assert_eq!(result.shape().dims(), &[32, 32]);
            // Device comparison removed as candle::Device doesn't implement PartialEq
        }
    }

    #[test]
    fn test_cpu_device_operations() {
        let _pool = setup_memory_pool(); // Keep pool alive
        let device = Device::Cpu;
        let a = BitNetTensor::ones(&[16, 16], BitNetDType::F32, Some(device.clone())).unwrap();
        let b = BitNetTensor::ones(&[16, 16], BitNetDType::F32, Some(device.clone())).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[16, 16]);
        // Device comparison by converting to string or using format
        assert_eq!(format!("{:?}", result.device()), format!("{:?}", &device));
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_matmul_sizes() {
        let sizes = vec![32, 64, 128, 256];

        for size in sizes {
            let a = BitNetTensor::ones(&[size, size], BitNetDType::F32, None).unwrap();
            let b = BitNetTensor::ones(&[size, size], BitNetDType::F32, None).unwrap();

            let start = Instant::now();
            let _result = matmul(&a, &b).unwrap();
            let duration = start.elapsed();

            println!("{size}x{size} matrix multiplication took: {duration:?}");
            assert!(duration.as_secs() < 10); // Reasonable upper bound
        }
    }

    #[test]
    fn benchmark_dot_product_sizes() {
        let sizes = vec![1000, 10000, 100000];

        for size in sizes {
            let a = BitNetTensor::ones(&[size], BitNetDType::F32, None).unwrap();
            let b = BitNetTensor::ones(&[size], BitNetDType::F32, None).unwrap();

            let start = Instant::now();
            let _result = dot(&a, &b).unwrap();
            let duration = start.elapsed();

            println!("{size} element dot product took: {duration:?}");
            assert!(duration.as_secs() < 5); // Reasonable upper bound
        }
    }
}
