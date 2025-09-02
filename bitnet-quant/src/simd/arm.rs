//! ARM/AArch64 specific SIMD optimizations
//!
//! This module contains ARM-specific SIMD implementations using NEON and SVE
//! instruction sets for optimal performance on ARM architectures including
//! Apple Silicon, ARM Cortex-A series, and other ARM processors.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::capabilities::{detect_simd_capabilities, SimdCapabilities};
use crate::quantization::utils::QuantizationError;

/// ARM-specific SIMD operations
pub struct ArmSimdOps {
    capabilities: SimdCapabilities,
}

impl ArmSimdOps {
    /// Create new ARM SIMD operations handler
    pub fn new() -> Self {
        Self {
            capabilities: detect_simd_capabilities(),
        }
    }

    /// High-performance ternary quantization using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_ternary_neon(
        &self,
        input: &[f32],
        output: &mut [i8],
        threshold: f32,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        let len = input.len();
        let vector_len = len & !3; // Process 4 elements at a time

        let threshold_vec = vdupq_n_f32(threshold);
        let neg_threshold_vec = vdupq_n_f32(-threshold);

        for i in (0..vector_len).step_by(4) {
            // Load 4 f32 values
            let values = vld1q_f32(input.as_ptr().add(i));

            // Create comparison masks
            let gt_mask = vcgtq_f32(values, threshold_vec);
            let lt_mask = vcltq_f32(values, neg_threshold_vec);

            // Convert masks to signed integers
            let gt_i32 = vreinterpretq_s32_u32(gt_mask);
            let lt_i32 = vreinterpretq_s32_u32(lt_mask);

            // Apply ternary quantization: +1, -1, or 0
            let positive = vandq_s32(gt_i32, vdupq_n_s32(1));
            let negative = vandq_s32(lt_i32, vdupq_n_s32(-1));
            let result_i32 = vorrq_s32(positive, negative);

            // Convert to i8 and store
            let result_i16 = vqmovn_s32(result_i32);
            let result_i8 = vqmovn_s16(vcombine_s16(result_i16, vdup_n_s16(0)));

            // Store 4 i8 values
            let mut temp: [i8; 8] = [0; 8];
            vst1_s8(temp.as_mut_ptr(), result_i8);
            for j in 0..4 {
                if i + j < output.len() {
                    output[i + j] = temp[j];
                }
            }
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

    /// Optimized ternary matrix-vector multiplication using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn ternary_gemv_neon(
        &self,
        a_ternary: &[i8],
        x: &[f32],
        y: &mut [f32],
        m: usize,
        k: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        if a_ternary.len() != m * k || x.len() != k || y.len() != m {
            return Err(QuantizationError::ConfigError("Dimension mismatch".into()));
        }

        let _scale_vec = vdupq_n_f32(scale);

        for i in 0..m {
            let mut acc = vdupq_n_f32(0.0);
            let k_vec = k & !3; // Process 4 elements at a time

            // Vectorized inner product
            for kk in (0..k_vec).step_by(4) {
                // Load 4 ternary values from A and convert to f32
                let a_vals_i8 = vld1_s8(a_ternary.as_ptr().add(i * k + kk));
                let a_vals_i16 = vmovl_s8(a_vals_i8);
                let a_vals_i32 = vmovl_s16(vget_low_s16(a_vals_i16));
                let a_vals_f32 = vcvtq_f32_s32(a_vals_i32);

                // Load 4 values from x
                let x_vals = vld1q_f32(x.as_ptr().add(kk));

                // Multiply and accumulate
                acc = vfmaq_f32(acc, a_vals_f32, x_vals);
            }

            // Horizontal sum using pairwise addition
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

    /// Optimized ternary matrix multiplication using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn ternary_gemm_neon(
        &self,
        a_ternary: &[i8],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        let scale_vec = vdupq_n_f32(scale);

        // Initialize output to zero
        c.fill(0.0);

        // Process output in 4-wide blocks
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let j_max = std::cmp::min(j + 4, n);
                let mut acc = vdupq_n_f32(0.0);

                // Inner loop over K dimension
                for kk in 0..k {
                    let a_val = a_ternary[i * k + kk];
                    if a_val != 0 {
                        let a_broadcast = vdupq_n_f32(a_val as f32);

                        if j_max - j >= 4 {
                            // Full vector operation
                            let b_vals = vld1q_f32(b.as_ptr().add(kk * n + j));
                            acc = vfmaq_f32(acc, a_broadcast, b_vals);
                        } else {
                            // Partial vector - handle manually
                            for jj in j..j_max {
                                c[i * n + jj] += (a_val as f32) * b[kk * n + jj] * scale;
                            }
                        }
                    }
                }

                if j_max - j >= 4 {
                    // Scale and add to existing values
                    let scaled_acc = vmulq_f32(acc, scale_vec);
                    let prev_c = vld1q_f32(c.as_ptr().add(i * n + j));
                    let result = vaddq_f32(prev_c, scaled_acc);
                    vst1q_f32(c.as_mut_ptr().add(i * n + j), result);
                }
            }
        }

        Ok(())
    }

    /// Vectorized absolute value sum using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn abs_sum_neon(&self, values: &[f32]) -> Result<f32, QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        let len = values.len();
        let vector_len = len & !3; // Process 4 elements at a time
        let mut acc = vdupq_n_f32(0.0);

        for i in (0..vector_len).step_by(4) {
            let vals = vld1q_f32(values.as_ptr().add(i));
            let abs_vals = vabsq_f32(vals);
            acc = vaddq_f32(acc, abs_vals);
        }

        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum_final = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(sum_final, 0);

        // Handle remaining elements
        for i in vector_len..len {
            result += values[i].abs();
        }

        Ok(result)
    }

    /// Optimized ternary weight packing using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn pack_ternary_neon(
        &self,
        input: &[i8],
        output: &mut [u8],
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        let len = input.len();
        let vector_len = len & !7; // Process 8 elements at a time
        let mut output_pos = 0;

        // Offset to convert -1,0,1 to 0,1,2
        let offset = vdup_n_s8(1);

        for i in (0..vector_len).step_by(8) {
            // Load 8 ternary values
            let vals = vld1_s8(input.as_ptr().add(i));

            // Add offset to convert -1,0,1 to 0,1,2
            let vals_offset = vadd_s8(vals, offset);

            // Extract individual values for bit packing
            let mut temp_vals: [i8; 8] = [0; 8];
            vst1_s8(temp_vals.as_mut_ptr(), vals_offset);

            // Pack pairs of values into bytes (4 ternary values per byte)
            for j in (0..8).step_by(4) {
                if output_pos < output.len() && j + 3 < 8 {
                    let packed = (temp_vals[j] & 0x3)
                        | ((temp_vals[j + 1] & 0x3) << 2)
                        | ((temp_vals[j + 2] & 0x3) << 4)
                        | ((temp_vals[j + 3] & 0x3) << 6);
                    output[output_pos] = packed as u8;
                    output_pos += 1;
                }
            }
        }

        // Handle remaining elements
        let mut i = vector_len;
        while i < len && output_pos < output.len() {
            let mut packed = 0u8;
            for bit_pos in 0..4 {
                if i < len {
                    let val = match input[i] {
                        -1 => 0u8,
                        0 => 1u8,
                        1 => 2u8,
                        _ => {
                            return Err(QuantizationError::ConfigError(format!(
                                "Invalid ternary value: {}",
                                input[i]
                            )))
                        }
                    };
                    packed |= (val & 0x3) << (bit_pos * 2);
                    i += 1;
                }
            }
            output[output_pos] = packed;
            output_pos += 1;
        }

        Ok(())
    }

    /// Optimized ternary weight unpacking using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_ternary_neon(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        let required_input_bytes = count.div_ceil(4);
        if input.len() < required_input_bytes || output.len() < count {
            return Err(QuantizationError::ConfigError(
                "Buffer size mismatch".into(),
            ));
        }

        let mut input_pos = 0;
        let mut output_pos = 0;

        // Process in chunks that produce 8 outputs at a time
        while output_pos + 8 <= count && input_pos + 2 <= input.len() {
            // Load 2 bytes (8 ternary values) using NEON
            let bytes = [input[input_pos], input[input_pos + 1], 0, 0, 0, 0, 0, 0];
            let _packed = vld1_u8(bytes.as_ptr());

            // Extract individual values manually
            let byte1 = input[input_pos];
            let byte2 = input[input_pos + 1];

            let mut decoded: [i8; 8] = [0; 8];

            // Unpack byte1 (4 values)
            for bit_pos in 0..4 {
                let val_bits = (byte1 >> (bit_pos * 2)) & 0x3;
                decoded[bit_pos] = match val_bits {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0,
                };
            }

            // Unpack byte2 (4 values)
            for bit_pos in 0..4 {
                let val_bits = (byte2 >> (bit_pos * 2)) & 0x3;
                decoded[bit_pos + 4] = match val_bits {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0,
                };
            }

            // Store results
            for j in 0..8 {
                if output_pos < count {
                    output[output_pos] = decoded[j];
                    output_pos += 1;
                }
            }

            input_pos += 2;
        }

        // Handle remaining elements
        while output_pos < count && input_pos < input.len() {
            let byte = input[input_pos];

            for bit_pos in 0..4 {
                if output_pos >= count {
                    break;
                }

                let val_bits = (byte >> (bit_pos * 2)) & 0x3;
                output[output_pos] = match val_bits {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0,
                };
                output_pos += 1;
            }

            input_pos += 1;
        }

        Ok(())
    }

    /// Vectorized element-wise multiplication using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn element_multiply_neon(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        if a.len() != b.len() || a.len() != c.len() {
            return Err(QuantizationError::ConfigError(
                "Array length mismatch".into(),
            ));
        }

        let len = a.len();
        let vector_len = len & !3; // Process 4 elements at a time

        for i in (0..vector_len).step_by(4) {
            let a_vals = vld1q_f32(a.as_ptr().add(i));
            let b_vals = vld1q_f32(b.as_ptr().add(i));
            let result = vmulq_f32(a_vals, b_vals);
            vst1q_f32(c.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in vector_len..len {
            c[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Cache-optimized matrix transpose using NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn transpose_matrix_neon(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<(), QuantizationError> {
        if !self.capabilities.neon {
            return Err(QuantizationError::ConfigError("NEON not available".into()));
        }

        if input.len() != rows * cols || output.len() != rows * cols {
            return Err(QuantizationError::ConfigError(
                "Matrix size mismatch".into(),
            ));
        }

        const TILE_SIZE: usize = 4; // NEON processes 4 f32 at a time

        // Process in tiles for better cache locality
        for row_tile in (0..rows).step_by(TILE_SIZE) {
            for col_tile in (0..cols).step_by(TILE_SIZE) {
                let row_end = std::cmp::min(row_tile + TILE_SIZE, rows);
                let col_end = std::cmp::min(col_tile + TILE_SIZE, cols);

                // Transpose within tile
                for r in row_tile..row_end {
                    for c_start in (col_tile..col_end).step_by(4) {
                        let c_end_vec = std::cmp::min(c_start + 4, col_end);

                        if c_end_vec - c_start == 4 {
                            // Full vector operation
                            let vals = vld1q_f32(input.as_ptr().add(r * cols + c_start));

                            // Store in transposed positions
                            let vals_array: [f32; 4] = std::mem::transmute(vals);
                            for (i, c) in (c_start..c_end_vec).enumerate() {
                                output[c * rows + r] = vals_array[i];
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

impl Default for ArmSimdOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_arm_simd_availability() {
        let ops = ArmSimdOps::new();
        println!("ARM SIMD capabilities: {:?}", ops.capabilities);
    }

    #[test]
    fn test_ternary_quantization_neon() {
        if !detect_simd_capabilities().neon {
            return; // Skip if NEON not available
        }

        let input = vec![2.0, 0.5, -1.5, 0.1, -0.1, 3.0, -2.0, 0.0];
        let mut output = vec![0i8; input.len()];
        let threshold = 1.0;

        let ops = ArmSimdOps::new();
        unsafe {
            ops.quantize_ternary_neon(&input, &mut output, threshold)
                .unwrap();
        }

        let expected = vec![1, 0, -1, 0, 0, 1, -1, 0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_abs_sum_neon() {
        if !detect_simd_capabilities().neon {
            return;
        }

        let values = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let ops = ArmSimdOps::new();

        let result = unsafe { ops.abs_sum_neon(&values).unwrap() };
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0;

        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_element_multiply_neon() {
        if !detect_simd_capabilities().neon {
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 5];

        let ops = ArmSimdOps::new();
        unsafe {
            ops.element_multiply_neon(&a, &b, &mut c).unwrap();
        }

        let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0];
        for (actual, expected) in c.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ternary_gemv_neon() {
        if !detect_simd_capabilities().neon {
            return;
        }

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

        let ops = ArmSimdOps::new();
        unsafe {
            ops.ternary_gemv_neon(&a, &x, &mut y, m, k, 1.0).unwrap();
        }

        // Expected: [2*1 + 3*(-1) + 1*0 + (-1)*1, 2*0 + 3*1 + 1*(-1) + (-1)*0, 2*(-1) + 3*0 + 1*1 + (-1)*1]
        // = [-2, 2, -2]
        assert_abs_diff_eq!(y[0], -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[2], -2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ternary_pack_unpack_neon() {
        if !detect_simd_capabilities().neon {
            return;
        }

        let input = vec![-1, 0, 1, -1, 0, 1, 1, 0];
        let mut packed = vec![0u8; (input.len() + 3) / 4];
        let mut unpacked = vec![0i8; input.len()];

        let ops = ArmSimdOps::new();

        unsafe {
            ops.pack_ternary_neon(&input, &mut packed).unwrap();
            ops.unpack_ternary_neon(&packed, &mut unpacked, input.len())
                .unwrap();
        }

        assert_eq!(input, unpacked);
    }

    #[test]
    fn test_matrix_transpose_neon() {
        if !detect_simd_capabilities().neon {
            return;
        }

        let rows = 4;
        let cols = 4;
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = vec![0.0; 16];

        let ops = ArmSimdOps::new();
        unsafe {
            ops.transpose_matrix_neon(&input, &mut output, rows, cols)
                .unwrap();
        }

        let expected = vec![
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
        ];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }
}
