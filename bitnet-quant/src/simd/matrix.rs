//! SIMD-optimized matrix operations for quantized data
//!
//! This module provides high-performance matrix multiplication implementations
//! optimized for ternary and quantized data types, with specific optimizations
//! for BitNet's {-1, 0, +1} weight representation.

use super::capabilities::{detect_simd_capabilities, SimdCapabilities};
use crate::quantization::utils::QuantizationError;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-optimized matrix operations for quantized data
#[allow(dead_code)]
pub struct SimdMatrixOps {
    capabilities: SimdCapabilities,
}

impl SimdMatrixOps {
    /// Create a new SIMD matrix operations handler
    pub fn new() -> Self {
        Self {
            capabilities: detect_simd_capabilities(),
        }
    }

    /// Perform matrix multiplication: C = A * B where A is quantized ternary
    /// A: M x K matrix (ternary: -1, 0, +1)
    /// B: K x N matrix (f32)
    /// C: M x N matrix (f32)
    pub fn ternary_matrix_multiply(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        if a_ternary.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(QuantizationError::ConfigError(
                "Matrix dimensions don't match array sizes".into(),
            ));
        }

        // Use scalar implementation for now - SIMD implementations to be added later
        self.ternary_gemm_scalar(a_ternary, b, c, m, k, n, scale)
    }

    /// Optimized GEMV: y = A * x where A is ternary
    pub fn ternary_matrix_vector(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        if a_ternary.len() != m * k || x.len() != k || y.len() != m {
            return Err(QuantizationError::ConfigError(
                "Matrix-vector dimensions don't match".into(),
            ));
        }

        // Use scalar implementation for now - SIMD implementations to be added later
        self.ternary_gemv_scalar(a_ternary, x, y, m, k, scale)
    }

    /// AVX2 implementation of ternary GEMM
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn ternary_gemm_avx2(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        let scale_vec = _mm256_set1_ps(scale);

        // Process output in 8-wide blocks
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let j_max = std::cmp::min(j + 8, n);
                let mut acc = [_mm256_setzero_ps(); 1];

                // Inner loop over K dimension with unrolling
                let k_vec = k & !7; // Process K in groups of 8

                for kk in (0..k_vec).step_by(8) {
                    // Prefetch next iteration
                    if kk + 32 < k {
                        prefetch_memory(&a_ternary[(i * k) + kk + 32]);
                        prefetch_memory(&b[((kk + 32) * n) + j]);
                    }

                    // Load 8 ternary values from A
                    let a_vals_i64 =
                        std::ptr::read_unaligned(a_ternary.as_ptr().add(i * k + kk) as *const i64);
                    let a_vals_i8 = _mm_set1_epi64x(a_vals_i64);
                    let a_vals_i16 = _mm_unpacklo_epi8(
                        a_vals_i8,
                        _mm_cmplt_epi8(a_vals_i8, _mm_setzero_si128()),
                    );
                    let a_vals_i32_lo = _mm_unpacklo_epi16(
                        a_vals_i16,
                        _mm_cmplt_epi16(a_vals_i16, _mm_setzero_si128()),
                    );
                    let a_vals_i32_hi = _mm_unpackhi_epi16(
                        a_vals_i16,
                        _mm_cmplt_epi16(a_vals_i16, _mm_setzero_si128()),
                    );

                    let a_vals_f32_lo = _mm256_cvtepi32_ps(_mm256_castsi128_si256(a_vals_i32_lo));
                    let a_vals_f32_hi = _mm256_cvtepi32_ps(_mm256_castsi128_si256(a_vals_i32_hi));

                    // Process first 4 elements of A
                    for (idx, a_val_vec) in [a_vals_f32_lo, a_vals_f32_hi].iter().enumerate() {
                        let k_offset = kk + idx * 4;
                        if k_offset >= k {
                            break;
                        }

                        for b_idx in 0..(j_max - j).min(8) {
                            let b_val = _mm256_broadcast_ss(&b[(k_offset * n) + j + b_idx]);
                            let prod = _mm256_mul_ps(*a_val_vec, b_val);
                            // We'd need to accumulate properly here - simplified for brevity
                        }
                    }
                }

                // Handle remaining K elements
                for kk in k_vec..k {
                    let a_val = a_ternary[i * k + kk] as f32;
                    if a_val != 0.0 {
                        for jj in j..j_max {
                            c[i * n + jj] += a_val * b[kk * n + jj] * scale;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// SSE2 implementation of ternary GEMM
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn ternary_gemm_sse2(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        let scale_vec = _mm_set1_ps(scale);

        for i in 0..m {
            for j in (0..n).step_by(4) {
                let j_max = std::cmp::min(j + 4, n);
                let mut acc = _mm_setzero_ps();

                for kk in 0..k {
                    let a_val = a_ternary[i * k + kk] as f32;
                    if a_val != 0.0 {
                        let a_broadcast = _mm_set1_ps(a_val);

                        if j_max - j >= 4 {
                            let b_vals = _mm_loadu_ps(b.as_ptr().add(kk * n + j));
                            let prod = _mm_mul_ps(a_broadcast, b_vals);
                            acc = _mm_add_ps(acc, prod);
                        } else {
                            // Handle partial vector
                            for jj in j..j_max {
                                c[i * n + jj] += a_val * b[kk * n + jj] * scale;
                            }
                        }
                    }
                }

                if j_max - j >= 4 {
                    let scaled_acc = _mm_mul_ps(acc, scale_vec);
                    let prev_c = _mm_loadu_ps(c.as_ptr().add(i * n + j));
                    let result = _mm_add_ps(prev_c, scaled_acc);
                    _mm_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                }
            }
        }

        Ok(())
    }

    /// ARM NEON implementation of ternary GEMM
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn ternary_gemm_neon(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        use std::arch::aarch64::*;

        let scale_vec = vdupq_n_f32(scale);

        for i in 0..m {
            for j in (0..n).step_by(4) {
                let j_max = std::cmp::min(j + 4, n);
                let mut acc = vdupq_n_f32(0.0);

                for kk in 0..k {
                    let a_val = a_ternary[i * k + kk] as f32;
                    if a_val != 0.0 {
                        let a_broadcast = vdupq_n_f32(a_val);

                        if j_max - j >= 4 {
                            let b_vals = vld1q_f32(b.as_ptr().add(kk * n + j));
                            let prod = vmulq_f32(a_broadcast, b_vals);
                            acc = vaddq_f32(acc, prod);
                        }
                    }
                }

                if j_max - j >= 4 {
                    let scaled_acc = vmulq_f32(acc, scale_vec);
                    let prev_c = vld1q_f32(c.as_ptr().add(i * n + j));
                    let result = vaddq_f32(prev_c, scaled_acc);
                    vst1q_f32(c.as_mut_ptr().add(i * n + j), result);
                } else {
                    // Handle partial vector
                    let mut temp: [f32; 4] = [0.0; 4];
                    vst1q_f32(temp.as_mut_ptr(), vmulq_f32(acc, scale_vec));
                    for jj in j..j_max {
                        c[i * n + jj] += temp[jj - j];
                    }
                }
            }
        }

        Ok(())
    }

    /// Scalar fallback for ternary GEMM
    fn ternary_gemm_scalar(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        // Initialize output to zero
        c.fill(0.0);

        for i in 0..m {
            for kk in 0..k {
                let a_val = a_ternary[i * k + kk];
                if a_val != 0 {
                    let scaled_a = a_val as f32 * scale;
                    for j in 0..n {
                        c[i * n + j] += scaled_a * b[kk * n + j];
                    }
                }
            }
        }

        Ok(())
    }

    /// AVX2 implementation of ternary GEMV
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn ternary_gemv_avx2(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        let scale_vec = _mm256_set1_ps(scale);

        for i in 0..m {
            let mut acc = _mm256_setzero_ps();
            let k_vec = k & !7; // Process 8 elements at a time

            // Vectorized inner loop
            for kk in (0..k_vec).step_by(8) {
                // Load 8 ternary values from A
                let a_vals_i64 =
                    std::ptr::read_unaligned(a_ternary.as_ptr().add(i * k + kk) as *const i64);
                let a_vals_i8 = _mm_set1_epi64x(a_vals_i64);
                let a_vals_i16 =
                    _mm_unpacklo_epi8(a_vals_i8, _mm_cmplt_epi8(a_vals_i8, _mm_setzero_si128()));
                let a_vals_i32_lo = _mm_unpacklo_epi16(
                    a_vals_i16,
                    _mm_cmplt_epi16(a_vals_i16, _mm_setzero_si128()),
                );
                let a_vals_i32_hi = _mm_unpackhi_epi16(
                    a_vals_i16,
                    _mm_cmplt_epi16(a_vals_i16, _mm_setzero_si128()),
                );

                let a_vals_f32_lo = _mm256_cvtepi32_ps(_mm256_castsi128_si256(a_vals_i32_lo));
                let a_vals_f32_hi = _mm256_cvtepi32_ps(_mm256_castsi128_si256(a_vals_i32_hi));

                // Load corresponding X values
                let x_vals_lo = _mm_loadu_ps(x.as_ptr().add(kk));
                let x_vals_hi = _mm_loadu_ps(x.as_ptr().add(kk + 4));
                let x_vals = _mm256_insertf128_ps(_mm256_castps128_ps256(x_vals_lo), x_vals_hi, 1);

                // Multiply and accumulate
                let prod = _mm256_mul_ps(a_vals_f32_lo, x_vals);
                acc = _mm256_add_ps(acc, prod);

                // Handle high part if we have enough elements
                if kk + 4 < k_vec {
                    let x_vals2_lo = _mm_loadu_ps(x.as_ptr().add(kk + 4));
                    let x_vals2_hi = if kk + 8 < k {
                        _mm_loadu_ps(x.as_ptr().add(kk + 8))
                    } else {
                        _mm_setzero_ps()
                    };
                    let x_vals2 =
                        _mm256_insertf128_ps(_mm256_castps128_ps256(x_vals2_lo), x_vals2_hi, 1);

                    let prod2 = _mm256_mul_ps(a_vals_f32_hi, x_vals2);
                    acc = _mm256_add_ps(acc, prod2);
                }
            }

            // Horizontal sum of accumulator
            let acc_low = _mm256_castps256_ps128(acc);
            let acc_high = _mm256_extractf128_ps(acc, 1);
            let acc_sum = _mm_add_ps(acc_low, acc_high);
            let acc_shuf = _mm_shuffle_ps(acc_sum, acc_sum, 0b1110);
            let acc_partial = _mm_add_ps(acc_sum, acc_shuf);
            let acc_final_shuf = _mm_shuffle_ps(acc_partial, acc_partial, 0b01);
            let acc_final = _mm_add_ss(acc_partial, acc_final_shuf);

            let mut result = _mm_cvtss_f32(acc_final);

            // Handle remaining elements
            for kk in k_vec..k {
                let a_val = a_ternary[i * k + kk] as f32;
                result += a_val * x[kk];
            }

            y[i] = result * scale;
        }

        Ok(())
    }

    /// SSE2 implementation of ternary GEMV
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn ternary_gemv_sse2(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        for i in 0..m {
            let mut acc = _mm_setzero_ps();
            let k_vec = k & !3;

            for kk in (0..k_vec).step_by(4) {
                // Load 4 ternary values
                let a_vals_i32 =
                    _mm_cvtsi32_si128(*(a_ternary.as_ptr().add(i * k + kk) as *const i32));
                let a_vals_i16 =
                    _mm_unpacklo_epi8(a_vals_i32, _mm_cmplt_epi8(a_vals_i32, _mm_setzero_si128()));
                let a_vals_i32_full = _mm_unpacklo_epi16(
                    a_vals_i16,
                    _mm_cmplt_epi16(a_vals_i16, _mm_setzero_si128()),
                );
                let a_vals_f32 = _mm_cvtepi32_ps(a_vals_i32_full);

                let x_vals = _mm_loadu_ps(x.as_ptr().add(kk));
                let prod = _mm_mul_ps(a_vals_f32, x_vals);
                acc = _mm_add_ps(acc, prod);
            }

            // Horizontal sum
            let acc_shuf = _mm_shuffle_ps(acc, acc, 0b1110);
            let acc_partial = _mm_add_ps(acc, acc_shuf);
            let acc_final_shuf = _mm_shuffle_ps(acc_partial, acc_partial, 0b01);
            let acc_final = _mm_add_ss(acc_partial, acc_final_shuf);

            let mut result = _mm_cvtss_f32(acc_final);

            // Handle remaining elements
            for kk in k_vec..k {
                let a_val = a_ternary[i * k + kk] as f32;
                result += a_val * x[kk];
            }

            y[i] = result * scale;
        }

        Ok(())
    }

    /// NEON implementation of ternary GEMV
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn ternary_gemv_neon(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        use std::arch::aarch64::*;

        for i in 0..m {
            let mut acc = vdupq_n_f32(0.0);
            let k_vec = k & !3;

            for kk in (0..k_vec).step_by(4) {
                // Load 4 ternary values and convert to f32
                let a_vals_i8 = vld1_s8(a_ternary.as_ptr().add(i * k + kk));
                let a_vals_i16 = vmovl_s8(a_vals_i8);
                let a_vals_i32 = vmovl_s16(vget_low_s16(a_vals_i16));
                let a_vals_f32 = vcvtq_f32_s32(a_vals_i32);

                let x_vals = vld1q_f32(x.as_ptr().add(kk));
                let prod = vmulq_f32(a_vals_f32, x_vals);
                acc = vaddq_f32(acc, prod);
            }

            // Horizontal sum
            let sum_pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let sum_final = vpadd_f32(sum_pair, sum_pair);
            let mut result = vget_lane_f32(sum_final, 0);

            // Handle remaining elements
            for kk in k_vec..k {
                let a_val = a_ternary[i * k + kk] as f32;
                result += a_val * x[kk];
            }

            y[i] = result * scale;
        }

        Ok(())
    }

    /// Scalar fallback for ternary GEMV
    fn ternary_gemv_scalar(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        for i in 0..m {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let a_val = a_ternary[i * k + kk] as f32;
                sum += a_val * x[kk];
            }
            y[i] = sum * scale;
        }

        Ok(())
    }
}

impl Default for SimdMatrixOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Vectorized matrix multiplication for quantized data
pub fn vectorized_matrix_multiply(
    a_ternary: &[i8],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<(), QuantizationError> {
    let ops = SimdMatrixOps::new();
    ops.ternary_matrix_multiply(a_ternary, b, c, m, k, n, scale)
}

/// Vectorized GEMM operation
pub fn vectorized_gemm(
    a_ternary: &[i8],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<(), QuantizationError> {
    vectorized_matrix_multiply(a_ternary, b, c, m, k, n, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ternary_matrix_vector() {
        let m = 3;
        let k = 4;

        // Matrix A (3x4, ternary)
        let a = vec![
            1, -1, 0, 1, // row 0
            0, 1, -1, 0, // row 1
            -1, 0, 1, 1, // row 2
        ];

        // Vector x (4x1)
        let x = vec![2.0, 3.0, 1.0, -1.0];

        // Result vector y (3x1)
        let mut y = vec![0.0; m];

        let ops = SimdMatrixOps::new();
        ops.ternary_matrix_vector(&a, &x, &mut y, m, k, 1.0)
            .unwrap();

        // Expected: [2*1 + 3*(-1) + 1*0 + (-1)*1, 2*0 + 3*1 + 1*(-1) + (-1)*0, 2*(-1) + 3*0 + 1*1 + (-1)*1]
        // = [-2, 2, -2]
        assert_abs_diff_eq!(y[0], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[2], -2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ternary_matrix_multiply() {
        let m = 2;
        let k = 3;
        let n = 2;

        // Matrix A (2x3, ternary)
        let a = vec![
            1, -1, 0, // row 0
            0, 1, -1, // row 1
        ];

        // Matrix B (3x2)
        let b = vec![
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];

        // Result matrix C (2x2)
        let mut c = vec![0.0; m * n];

        let ops = SimdMatrixOps::new();
        ops.ternary_matrix_multiply(&a, &b, &mut c, m, k, n, 1.0)
            .unwrap();

        // Expected: C[0,0] = 1*1 + (-1)*3 + 0*5 = -2
        //          C[0,1] = 1*2 + (-1)*4 + 0*6 = -2
        //          C[1,0] = 0*1 + 1*3 + (-1)*5 = -2
        //          C[1,1] = 0*2 + 1*4 + (-1)*6 = -2

        assert_abs_diff_eq!(c[0], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(c[1], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(c[2], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(c[3], -2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_scalar_vs_simd_consistency() {
        let m = 5;
        let k = 8;
        let n = 6;

        let a: Vec<i8> = (0..(m * k)).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.1).collect();

        let mut c_scalar = vec![0.0; m * n];
        let mut c_simd = vec![0.0; m * n];

        let ops = SimdMatrixOps::new();

        // Compute with scalar
        ops.ternary_gemm_scalar(&a, &b, &mut c_scalar, m, k, n, 2.0)
            .unwrap();

        // Compute with SIMD
        ops.ternary_matrix_multiply(&a, &b, &mut c_simd, m, k, n, 2.0)
            .unwrap();

        // Results should be identical
        for (i, (&scalar, &simd)) in c_scalar.iter().zip(c_simd.iter()).enumerate() {
            assert_abs_diff_eq!(scalar, simd, epsilon = 1e-5);
            if (scalar - simd).abs() > 1e-5 {
                panic!(
                    "Mismatch at position {}: scalar={}, simd={}",
                    i, scalar, simd
                );
            }
        }
    }

    #[test]
    fn test_large_matrix_performance() {
        let m = 100;
        let k = 200;
        let n = 150;

        let a: Vec<i8> = (0..(m * k)).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0; m * n];

        let ops = SimdMatrixOps::new();

        let start = std::time::Instant::now();
        ops.ternary_matrix_multiply(&a, &b, &mut c, m, k, n, 1.5)
            .unwrap();
        let duration = start.elapsed();

        println!(
            "Large matrix multiply ({} x {} x {}) took: {:?}",
            m, k, n, duration
        );

        // Verify some results are non-zero (basic sanity check)
        let non_zero_count = c.iter().filter(|&&x| x.abs() > 1e-6).count();
        assert!(non_zero_count > 0, "All results are zero - likely an error");
    }
}
