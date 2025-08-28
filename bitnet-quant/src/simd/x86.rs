//! x86/x86_64 specific SIMD optimizations
//!
//! This module contains Intel-specific SIMD implementations using SSE, AVX, and AVX-512
//! instruction sets for maximum performance on x86 architectures.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use super::capabilities::{detect_simd_capabilities, SimdCapabilities};
use crate::quantization::utils::QuantizationError;

/// x86-specific SIMD operations
#[allow(dead_code)]
pub struct X86SimdOps {
    capabilities: SimdCapabilities,
}

impl X86SimdOps {
    /// Create new x86 SIMD operations handler
    pub fn new() -> Self {
        Self {
            capabilities: detect_simd_capabilities(),
        }
    }

    /// High-performance ternary quantization using AVX-512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn quantize_ternary_avx512(
        &self,
        input: &[f32],
        output: &mut [i8],
        threshold: f32,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.avx512f {
            return Err(QuantizationError::ConfigError(
                "AVX-512 not available".into(),
            ));
        }

        let len = input.len();
        let vector_len = len & !15; // Process 16 elements at a time

        let threshold_vec = _mm512_set1_ps(threshold);
        let neg_threshold_vec = _mm512_set1_ps(-threshold);

        for i in (0..vector_len).step_by(16) {
            // Load 16 f32 values
            let values = _mm512_loadu_ps(input.as_ptr().add(i));

            // Create comparison masks
            let gt_mask = _mm512_cmp_ps_mask(values, threshold_vec, _CMP_GT_OQ);
            let lt_mask = _mm512_cmp_ps_mask(values, neg_threshold_vec, _CMP_LT_OQ);

            // Create result vector with ternary values
            let ones = _mm512_set1_ps(1.0);
            let neg_ones = _mm512_set1_ps(-1.0);
            let zeros = _mm512_setzero_ps();

            let positive = _mm512_mask_blend_ps(gt_mask, zeros, ones);
            let negative = _mm512_mask_blend_ps(lt_mask, zeros, neg_ones);
            let result = _mm512_or_ps(positive, negative);

            // Convert to i8 and store
            let result_i32 = _mm512_cvtps_epi32(result);
            let result_i16 = _mm512_packs_epi32(result_i32, result_i32);
            let result_i8 = _mm512_packs_epi16(result_i16, result_i16);

            // Store 16 i8 values
            _mm_storeu_si128(
                output.as_mut_ptr().add(i) as *mut __m128i,
                _mm512_extracti32x4_epi32(result_i8, 0),
            );
        }

        // Handle remaining elements
        for i in vector_len..len {
            let val = input[i];
            output[i] = if val > threshold {
                1
            } else if val < -threshold {
                -1
            } else {
                0
            };
        }

        Ok(())
    }

    /// Optimized ternary matrix multiplication using AVX-512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn ternary_gemm_avx512(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.avx512f {
            return Err(QuantizationError::ConfigError(
                "AVX-512 not available".into(),
            ));
        }

        let scale_vec = _mm512_set1_ps(scale);

        // Initialize output
        c.fill(0.0);

        for i in 0..m {
            for j in (0..n).step_by(16) {
                let j_max = std::cmp::min(j + 16, n);
                let mut acc = _mm512_setzero_ps();

                // Inner loop with vectorization
                for kk in 0..k {
                    let a_val = a_ternary[i * k + kk] as f32;
                    if a_val != 0.0 {
                        let a_broadcast = _mm512_set1_ps(a_val);

                        if j_max - j >= 16 {
                            let b_vals = _mm512_loadu_ps(b.as_ptr().add(kk * n + j));
                            let prod = _mm512_mul_ps(a_broadcast, b_vals);
                            acc = _mm512_add_ps(acc, prod);
                        } else {
                            // Handle partial vector
                            let mask = (1u16 << (j_max - j)) - 1;
                            let b_vals = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(kk * n + j));
                            let prod = _mm512_mul_ps(a_broadcast, b_vals);
                            acc = _mm512_add_ps(acc, prod);
                        }
                    }
                }

                // Scale and store results
                let scaled_acc = _mm512_mul_ps(acc, scale_vec);

                if j_max - j >= 16 {
                    let prev_c = _mm512_loadu_ps(c.as_ptr().add(i * n + j));
                    let result = _mm512_add_ps(prev_c, scaled_acc);
                    _mm512_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                } else {
                    // Handle partial vector
                    let mask = (1u16 << (j_max - j)) - 1;
                    let prev_c = _mm512_maskz_loadu_ps(mask, c.as_ptr().add(i * n + j));
                    let result = _mm512_add_ps(prev_c, scaled_acc);
                    _mm512_mask_storeu_ps(c.as_mut_ptr().add(i * n + j), mask, result);
                }
            }
        }

        Ok(())
    }

    /// Optimized FMA operations for quantized neural networks
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "fma")]
    pub unsafe fn fused_multiply_add_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.fma {
            return Err(QuantizationError::ConfigError("FMA not available".into()));
        }

        if a.len() != b.len() || a.len() != c.len() {
            return Err(QuantizationError::ConfigError(
                "Array length mismatch".into(),
            ));
        }

        let len = a.len();

        if self.capabilities.avx2 {
            let vector_len = len & !7; // Process 8 elements at a time

            for i in (0..vector_len).step_by(8) {
                let a_vals = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vals = _mm256_loadu_ps(b.as_ptr().add(i));
                let c_vals = _mm256_loadu_ps(c.as_ptr().add(i));

                // c = a * b + c
                let result = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), result);
            }

            // Handle remaining elements
            for i in vector_len..len {
                c[i] = a[i] * b[i] + c[i];
            }
        } else {
            // Scalar fallback
            for ((a_val, b_val), c_val) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
                *c_val = *a_val * *b_val + *c_val;
            }
        }

        Ok(())
    }

    /// Optimized dot product using AVX-512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_product_avx512(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<f32, QuantizationError> {
        if !self.capabilities.avx512f {
            return Err(QuantizationError::ConfigError(
                "AVX-512 not available".into(),
            ));
        }

        if a.len() != b.len() {
            return Err(QuantizationError::ConfigError(
                "Array length mismatch".into(),
            ));
        }

        let len = a.len();
        let vector_len = len & !15; // Process 16 elements at a time
        let mut acc = _mm512_setzero_ps();

        for i in (0..vector_len).step_by(16) {
            let a_vals = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vals = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(a_vals, b_vals, acc);
        }

        // Horizontal sum of accumulator
        let sum = _mm512_reduce_add_ps(acc);
        let mut result = sum;

        // Handle remaining elements
        for i in vector_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    /// Optimized ternary weight unpacking using AVX2
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_ternary_weights_avx2(
        &self,
        packed: &[u8],
        unpacked: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.avx2 {
            return Err(QuantizationError::ConfigError("AVX2 not available".into()));
        }

        let required_input_bytes = (count + 3) / 4;
        if packed.len() < required_input_bytes || unpacked.len() < count {
            return Err(QuantizationError::ConfigError(
                "Buffer size mismatch".into(),
            ));
        }

        // Shuffle mask for unpacking 2-bit values
        let shuffle_mask = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7,
            7, 7, 7,
        );

        // Bit shift amounts for extracting 2-bit values
        let shift_amounts = _mm256_setr_epi8(
            0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0,
            2, 4, 6,
        );

        // Decode lookup: 0->-1, 1->0, 2->1, 3->0 (reserved)
        let decode_lut = _mm256_setr_epi8(
            -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1,
            0, -1, 0, 1, 0,
        );

        let vector_len = (count & !31).min(packed.len() * 4); // Process 32 outputs at a time
        let mut input_pos = 0;
        let mut output_pos = 0;

        while output_pos + 32 <= vector_len && input_pos + 8 <= packed.len() {
            // Load 8 packed bytes
            let packed_vals = _mm_loadl_epi64(packed.as_ptr().add(input_pos) as *const __m128i);
            let packed_256 = _mm256_broadcastq_epi64(packed_vals);

            // Shuffle bytes to replicate each byte 4 times
            let shuffled = _mm256_shuffle_epi8(packed_256, shuffle_mask);

            // Shift right by appropriate amounts to isolate 2-bit values
            let shifted = _mm256_srlv_epi32(
                shuffled,
                _mm256_cvtepi8_epi32(_mm256_castsi256_si128(shift_amounts)),
            );

            // Mask to keep only 2 bits
            let mask = _mm256_set1_epi8(0x3);
            let masked = _mm256_and_si256(shifted, mask);

            // Decode using lookup table
            let decoded = _mm256_shuffle_epi8(decode_lut, masked);

            // Store result
            _mm256_storeu_si256(
                unpacked.as_mut_ptr().add(output_pos) as *mut __m256i,
                decoded,
            );

            input_pos += 8;
            output_pos += 32;
        }

        // Handle remaining elements with scalar code
        while output_pos < count && input_pos < packed.len() {
            let byte = packed[input_pos];

            for bit_pos in 0..4 {
                if output_pos >= count {
                    break;
                }

                let val_bits = (byte >> (bit_pos * 2)) & 0x3;
                unpacked[output_pos] = match val_bits {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0, // Reserved
                };
                output_pos += 1;
            }

            input_pos += 1;
        }

        Ok(())
    }

    /// Cache-friendly matrix transpose using AVX2
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn transpose_matrix_avx2(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.avx2 {
            return Err(QuantizationError::ConfigError("AVX2 not available".into()));
        }

        if input.len() != rows * cols || output.len() != rows * cols {
            return Err(QuantizationError::ConfigError(
                "Matrix size mismatch".into(),
            ));
        }

        const TILE_SIZE: usize = 8; // AVX2 processes 8 f32 at a time

        // Process in tiles for better cache locality
        for row_tile in (0..rows).step_by(TILE_SIZE) {
            for col_tile in (0..cols).step_by(TILE_SIZE) {
                let row_end = std::cmp::min(row_tile + TILE_SIZE, rows);
                let col_end = std::cmp::min(col_tile + TILE_SIZE, cols);

                // Transpose within tile
                for r in row_tile..row_end {
                    for c_start in (col_tile..col_end).step_by(8) {
                        let c_end_vec = std::cmp::min(c_start + 8, col_end);

                        if c_end_vec - c_start == 8 {
                            // Full vector load/store
                            let vals = _mm256_loadu_ps(input.as_ptr().add(r * cols + c_start));

                            // Store in transposed location (scattered)
                            for (i, c) in (c_start..c_end_vec).enumerate() {
                                let val_array: [f32; 8] = std::mem::transmute(vals);
                                output[c * rows + r] = val_array[i];
                            }
                        } else {
                            // Partial vector - use scalar
                            for c in c_start..c_end_vec {
                                output[c * rows + r] = input[r * cols + c];
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for X86SimdOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_x86_simd_availability() {
        let ops = X86SimdOps::new();
        println!("x86 SIMD capabilities: {:?}", ops.capabilities);
    }

    #[test]
    fn test_fused_multiply_add() {
        if !detect_simd_capabilities().fma {
            return; // Skip if FMA not available
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![1.0; 8];

        let ops = X86SimdOps::new();
        unsafe {
            ops.fused_multiply_add_f32(&a, &b, &mut c).unwrap();
        }

        // Expected: c[i] = a[i] * b[i] + original_c[i]
        let expected = vec![3.0, 7.0, 13.0, 21.0, 31.0, 43.0, 57.0, 73.0];
        for (actual, expected) in c.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matrix_transpose() {
        if !detect_simd_capabilities().avx2 {
            return; // Skip if AVX2 not available
        }

        let rows = 4;
        let cols = 4;
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; 16];

        let ops = X86SimdOps::new();
        unsafe {
            ops.transpose_matrix_avx2(&input, &mut output, rows, cols)
                .unwrap();
        }

        let expected = vec![
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
        ];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ternary_unpacking() {
        if !detect_simd_capabilities().avx2 {
            return; // Skip if AVX2 not available
        }

        // Pack some ternary values: -1, 0, 1, -1 (should be 0, 1, 2, 0 in 2-bit encoding)
        let packed = vec![0b00100100u8]; // 00|10|01|00
        let mut unpacked = vec![0i8; 4];

        let ops = X86SimdOps::new();
        unsafe {
            ops.unpack_ternary_weights_avx2(&packed, &mut unpacked, 4)
                .unwrap();
        }

        let expected = vec![-1, 0, 1, -1];
        assert_eq!(unpacked, expected);
    }
}
