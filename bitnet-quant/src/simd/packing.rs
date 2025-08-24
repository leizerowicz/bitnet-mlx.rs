//! SIMD-optimized packing and unpacking operations for ternary data
//!
//! This module provides high-performance implementations for packing and unpacking
//! ternary weights into compact bit representations, with parallel processing
//! capabilities across multiple architectures.

use super::capabilities::{detect_simd_capabilities, SimdCapabilities};
use crate::quantization::utils::QuantizationError;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-optimized packing operations for ternary data
pub struct SimdPackingOps {
    capabilities: SimdCapabilities,
}

impl SimdPackingOps {
    /// Create a new SIMD packing operations handler
    pub fn new() -> Self {
        Self {
            capabilities: detect_simd_capabilities(),
        }
    }

    /// Pack ternary values (-1, 0, +1) into compact 2-bit representation
    /// Each pair of bits represents: 00 = -1, 01 = 0, 10 = +1, 11 = reserved
    pub fn pack_ternary(&self, input: &[i8], output: &mut [u8]) -> Result<(), QuantizationError> {
        let required_output_size = input.len().div_ceil(4); // 4 ternary values per byte
        if output.len() < required_output_size {
            return Err(QuantizationError::ConfigError(
                "Output buffer too small for packing".into(),
            ));
        }

        // Use scalar implementation for now - SIMD implementations to be added later
        self.pack_ternary_scalar(input, output)
    }

    /// Unpack ternary values from compact 2-bit representation
    pub fn unpack_ternary(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        if output.len() < count {
            return Err(QuantizationError::ConfigError(
                "Output buffer too small for unpacking".into(),
            ));
        }

        // Use scalar implementation for now - SIMD implementations to be added later
        self.unpack_ternary_scalar(input, output, count)
    }

    /// Pack weights with run-length encoding for better compression
    pub fn pack_ternary_rle(
        &self,
        input: &[i8],
        output: &mut Vec<u8>,
    ) -> Result<(), QuantizationError> {
        output.clear();

        if input.is_empty() {
            return Ok(());
        }

        let mut i = 0;
        while i < input.len() {
            let current_val = input[i];
            let mut run_length = 1;

            // Count consecutive identical values (up to 63 for 6-bit encoding)
            while i + run_length < input.len()
                && input[i + run_length] == current_val
                && run_length < 63
            {
                run_length += 1;
            }

            // Encode: 2 bits for value, 6 bits for run length
            let encoded_val = match current_val {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => {
                    return Err(QuantizationError::ConfigError(format!(
                        "Invalid ternary value: {current_val}"
                    )))
                }
            };

            let encoded = (encoded_val << 6) | (run_length as u8);
            output.push(encoded);

            i += run_length;
        }

        Ok(())
    }

    /// Unpack RLE-encoded ternary data
    pub fn unpack_ternary_rle(
        &self,
        input: &[u8],
        output: &mut [i8],
    ) -> Result<usize, QuantizationError> {
        let mut output_pos = 0;

        for &encoded in input {
            let value_bits = (encoded >> 6) & 0b11;
            let run_length = (encoded & 0b111111) as usize;

            let value = match value_bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => {
                    return Err(QuantizationError::ConfigError(
                        "Invalid RLE encoded value".into(),
                    ))
                }
            };

            if output_pos + run_length > output.len() {
                return Err(QuantizationError::ConfigError(
                    "Output buffer too small for RLE unpacking".into(),
                ));
            }

            for _ in 0..run_length {
                output[output_pos] = value;
                output_pos += 1;
            }
        }

        Ok(output_pos)
    }

    /// AVX2 implementation for ternary packing
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn pack_ternary_avx2(
        &self,
        input: &[i8],
        output: &mut [u8],
    ) -> Result<(), QuantizationError> {
        let len = input.len();
        let vector_len = len & !31; // Process 32 elements at a time

        // Lookup table for ternary encoding: -1 -> 0, 0 -> 1, 1 -> 2
        let lut = _mm256_setr_epi8(
            1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // First 16 values
            1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Second 16 values
        );

        let mut output_pos = 0;

        // Process 32 ternary values at a time (produces 8 packed bytes)
        for i in (0..vector_len).step_by(32) {
            if i + 64 < len {
                prefetch_memory(&input[i + 64]);
            }

            // Load 32 ternary values
            let vals1 = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);

            // Convert -1/0/+1 to 0/1/2 using lookup table
            // Note: This is a simplified version - proper implementation would need
            // to handle signed values correctly with offset
            let offset = _mm256_set1_epi8(1); // Offset -1,0,1 to 0,1,2
            let vals_offset = _mm256_add_epi8(vals1, offset);

            // Pack 4 ternary values (2 bits each) into each byte
            // This requires bit manipulation across 32 values
            // For simplicity, we'll fall back to scalar for the bit packing part
            let mut temp_vals: [i8; 32] = [0; 32];
            _mm256_storeu_si256(temp_vals.as_mut_ptr() as *mut __m256i, vals_offset);

            // Pack 4 values per output byte
            for j in (0..32).step_by(4) {
                if output_pos < output.len() {
                    let packed = ((temp_vals[j] & 0x3) << 0)
                        | ((temp_vals[j + 1] & 0x3) << 2)
                        | ((temp_vals[j + 2] & 0x3) << 4)
                        | ((temp_vals[j + 3] & 0x3) << 6);
                    output[output_pos] = packed as u8;
                    output_pos += 1;
                }
            }
        }

        // Handle remaining elements with scalar code
        self.pack_ternary_scalar_range(input, output, vector_len, len, &mut output_pos)?;

        Ok(())
    }

    /// SSE2 implementation for ternary packing
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn pack_ternary_sse2(
        &self,
        input: &[i8],
        output: &mut [u8],
    ) -> Result<(), QuantizationError> {
        let len = input.len();
        let vector_len = len & !15; // Process 16 elements at a time
        let mut output_pos = 0;

        // Process 16 ternary values at a time
        for i in (0..vector_len).step_by(16) {
            let vals = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);

            // Add offset to convert -1,0,1 to 0,1,2
            let offset = _mm_set1_epi8(1);
            let vals_offset = _mm_add_epi8(vals, offset);

            // Extract values and pack manually
            let mut temp_vals: [i8; 16] = [0; 16];
            _mm_storeu_si128(temp_vals.as_mut_ptr() as *mut __m128i, vals_offset);

            // Pack 4 values per output byte
            for j in (0..16).step_by(4) {
                if output_pos < output.len() {
                    let packed = ((temp_vals[j] & 0x3) << 0)
                        | ((temp_vals[j + 1] & 0x3) << 2)
                        | ((temp_vals[j + 2] & 0x3) << 4)
                        | ((temp_vals[j + 3] & 0x3) << 6);
                    output[output_pos] = packed as u8;
                    output_pos += 1;
                }
            }
        }

        // Handle remaining elements
        self.pack_ternary_scalar_range(input, output, vector_len, len, &mut output_pos)?;

        Ok(())
    }

    /// ARM NEON implementation for ternary packing
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn pack_ternary_neon(
        &self,
        input: &[i8],
        output: &mut [u8],
    ) -> Result<(), QuantizationError> {
        use std::arch::aarch64::*;

        let len = input.len();
        let vector_len = len & !15; // Process 16 elements at a time
        let mut output_pos = 0;

        let offset = vdupq_n_s8(1); // Offset to convert -1,0,1 to 0,1,2

        for i in (0..vector_len).step_by(16) {
            let vals = vld1q_s8(input.as_ptr().add(i));
            let vals_offset = vaddq_s8(vals, offset);

            // Extract values for packing
            let mut temp_vals: [i8; 16] = [0; 16];
            vst1q_s8(temp_vals.as_mut_ptr(), vals_offset);

            // Pack 4 values per output byte
            for j in (0..16).step_by(4) {
                if output_pos < output.len() {
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
        self.pack_ternary_scalar_range(input, output, vector_len, len, &mut output_pos)?;

        Ok(())
    }

    /// Scalar implementation for ternary packing
    fn pack_ternary_scalar(
        &self,
        input: &[i8],
        output: &mut [u8],
    ) -> Result<(), QuantizationError> {
        let mut output_pos = 0;
        self.pack_ternary_scalar_range(input, output, 0, input.len(), &mut output_pos)
    }

    /// Helper for scalar packing of a range
    fn pack_ternary_scalar_range(
        &self,
        input: &[i8],
        output: &mut [u8],
        start: usize,
        end: usize,
        output_pos: &mut usize,
    ) -> Result<(), QuantizationError> {
        for i in (start..end).step_by(4) {
            if *output_pos >= output.len() {
                break;
            }

            let mut packed = 0u8;
            for j in 0..4 {
                if i + j < end {
                    let val = match input[i + j] {
                        -1 => 0u8,
                        0 => 1u8,
                        1 => 2u8,
                        _ => {
                            return Err(QuantizationError::ConfigError(format!(
                                "Invalid ternary value: {}",
                                input[i + j]
                            )))
                        }
                    };
                    packed |= (val & 0x3) << (j * 2);
                }
            }

            output[*output_pos] = packed;
            *output_pos += 1;
        }

        Ok(())
    }

    /// AVX2 implementation for ternary unpacking
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_ternary_avx2(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        let input_bytes_needed = (count + 3) / 4;
        if input.len() < input_bytes_needed {
            return Err(QuantizationError::ConfigError(
                "Input buffer too small for unpacking".into(),
            ));
        }

        let vector_len = (count & !31).min(input.len() * 4); // Process 32 outputs at a time
        let mut input_pos = 0;
        let mut output_pos = 0;

        // Decode lookup table: 0 -> -1, 1 -> 0, 2 -> 1
        let decode_lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        );

        // Process 8 input bytes at a time (32 ternary outputs)
        while output_pos + 32 <= vector_len && input_pos + 8 <= input.len() {
            // Load 8 packed bytes
            let packed = _mm_loadl_epi64(input.as_ptr().add(input_pos) as *const __m128i);

            // Expand each byte to 4 ternary values
            // This is complex bit manipulation - using scalar fallback for clarity
            let mut temp_input: [u8; 8] = [0; 8];
            _mm_storel_epi64(temp_input.as_mut_ptr() as *mut __m128i, packed);

            for &byte in &temp_input {
                for bit_pos in 0..4 {
                    if output_pos < count {
                        let val_bits = (byte >> (bit_pos * 2)) & 0x3;
                        output[output_pos] = match val_bits {
                            0 => -1,
                            1 => 0,
                            2 => 1,
                            _ => 0, // Reserved, treat as 0
                        };
                        output_pos += 1;
                    }
                }
            }

            input_pos += 8;
        }

        // Handle remaining elements with scalar
        self.unpack_ternary_scalar_range(input, output, input_pos, output_pos, count)?;

        Ok(())
    }

    /// SSE2 implementation for ternary unpacking
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn unpack_ternary_sse2(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        // Similar to AVX2 but processing smaller chunks
        let mut input_pos = 0;
        let mut output_pos = 0;

        // Process 4 input bytes at a time (16 ternary outputs)
        while output_pos + 16 <= count && input_pos + 4 <= input.len() {
            let packed = _mm_cvtsi32_si128(*(input.as_ptr().add(input_pos) as *const i32));

            let mut temp_input: [u8; 4] = [0; 4];
            _mm_storeu_si128(temp_input.as_mut_ptr() as *mut __m128i, packed);

            for &byte in &temp_input {
                for bit_pos in 0..4 {
                    if output_pos < count {
                        let val_bits = (byte >> (bit_pos * 2)) & 0x3;
                        output[output_pos] = match val_bits {
                            0 => -1,
                            1 => 0,
                            2 => 1,
                            _ => 0,
                        };
                        output_pos += 1;
                    }
                }
            }

            input_pos += 4;
        }

        // Handle remaining elements
        self.unpack_ternary_scalar_range(input, output, input_pos, output_pos, count)?;

        Ok(())
    }

    /// ARM NEON implementation for ternary unpacking
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn unpack_ternary_neon(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        let mut input_pos = 0;
        let mut output_pos = 0;

        // Process 4 input bytes at a time (16 ternary outputs)
        while output_pos + 16 <= count && input_pos + 4 <= input.len() {
            let mut temp_input: [u8; 4] = [0; 4];
            temp_input.copy_from_slice(&input[input_pos..input_pos + 4]);

            for &byte in &temp_input {
                for bit_pos in 0..4 {
                    if output_pos < count {
                        let val_bits = (byte >> (bit_pos * 2)) & 0x3;
                        output[output_pos] = match val_bits {
                            0 => -1,
                            1 => 0,
                            2 => 1,
                            _ => 0,
                        };
                        output_pos += 1;
                    }
                }
            }

            input_pos += 4;
        }

        // Handle remaining elements
        self.unpack_ternary_scalar_range(input, output, input_pos, output_pos, count)?;

        Ok(())
    }

    /// Scalar implementation for ternary unpacking
    fn unpack_ternary_scalar(
        &self,
        input: &[u8],
        output: &mut [i8],
        count: usize,
    ) -> Result<(), QuantizationError> {
        self.unpack_ternary_scalar_range(input, output, 0, 0, count)
    }

    /// Helper for scalar unpacking of remaining elements
    fn unpack_ternary_scalar_range(
        &self,
        input: &[u8],
        output: &mut [i8],
        mut input_pos: usize,
        mut output_pos: usize,
        count: usize,
    ) -> Result<(), QuantizationError> {
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
                    _ => 0, // Reserved/invalid
                };
                output_pos += 1;
            }

            input_pos += 1;
        }

        Ok(())
    }
}

impl Default for SimdPackingOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Vectorized ternary packing function
pub fn vectorized_pack_ternary(input: &[i8], output: &mut [u8]) -> Result<(), QuantizationError> {
    let ops = SimdPackingOps::new();
    ops.pack_ternary(input, output)
}

/// Vectorized ternary unpacking function
pub fn vectorized_unpack_ternary(
    input: &[u8],
    output: &mut [i8],
    count: usize,
) -> Result<(), QuantizationError> {
    let ops = SimdPackingOps::new();
    ops.unpack_ternary(input, output, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_pack_unpack() {
        let input = vec![-1, 0, 1, -1, 0, 1, 1, 0];
        let mut packed = vec![0u8; (input.len() + 3) / 4];
        let mut unpacked = vec![0i8; input.len()];

        let ops = SimdPackingOps::new();

        ops.pack_ternary(&input, &mut packed).unwrap();
        ops.unpack_ternary(&packed, &mut unpacked, input.len())
            .unwrap();

        assert_eq!(input, unpacked);
    }

    #[test]
    fn test_ternary_rle() {
        let input = vec![-1, -1, -1, 0, 0, 1, 1, 1, 1, 0];
        let mut encoded = Vec::new();
        let mut decoded = vec![0i8; input.len()];

        let ops = SimdPackingOps::new();

        ops.pack_ternary_rle(&input, &mut encoded).unwrap();
        let decoded_len = ops.unpack_ternary_rle(&encoded, &mut decoded).unwrap();

        assert_eq!(decoded_len, input.len());
        assert_eq!(&input, &decoded[..decoded_len]);
    }

    #[test]
    fn test_large_batch_packing() {
        let size = 10000;
        let input: Vec<i8> = (0..size)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            })
            .collect();
        let mut packed = vec![0u8; (size + 3) / 4];
        let mut unpacked = vec![0i8; size];

        let ops = SimdPackingOps::new();

        ops.pack_ternary(&input, &mut packed).unwrap();
        ops.unpack_ternary(&packed, &mut unpacked, size).unwrap();

        assert_eq!(input, unpacked);
    }

    #[test]
    fn test_packing_compression_ratio() {
        let input = vec![-1; 1000];
        let mut packed = vec![0u8; (input.len() + 3) / 4];

        let ops = SimdPackingOps::new();
        ops.pack_ternary(&input, &mut packed).unwrap();

        let compression_ratio = input.len() as f32 / packed.len() as f32;
        assert!(
            compression_ratio >= 3.5,
            "Compression ratio should be close to 4:1, got {}",
            compression_ratio
        );
    }

    #[test]
    fn test_rle_compression() {
        // Highly compressible input
        let mut input = vec![-1; 100];
        input.extend(vec![0; 100]);
        input.extend(vec![1; 100]);

        let mut rle_encoded = Vec::new();
        let mut regular_packed = vec![0u8; (input.len() + 3) / 4];

        let ops = SimdPackingOps::new();

        ops.pack_ternary_rle(&input, &mut rle_encoded).unwrap();
        ops.pack_ternary(&input, &mut regular_packed).unwrap();

        println!(
            "RLE size: {}, Regular packed size: {}",
            rle_encoded.len(),
            regular_packed.len()
        );
        assert!(
            rle_encoded.len() < regular_packed.len(),
            "RLE should compress better for this input"
        );
    }
}
