//! I2_S Optimized Kernels - Signed 2-bit Quantization Implementation
//!
//! High-performance implementation of signed 2-bit quantization kernels
//! with optimized memory access patterns for large-scale model support.
//! Supports both ARM64 NEON and x86_64 AVX vectorization.

use anyhow::Result;
use crate::cpu::{I2SLookupKernel};

/// ARM64 NEON-optimized I2_S kernel
pub struct I2SArmKernel {
    /// Cache line size for optimal memory access
    cache_line_size: usize,
    /// Vectorization width (NEON processes 4 f32 at once)
    vector_width: usize,
}

/// x86_64 AVX-optimized I2_S kernel
pub struct I2SX86Kernel {
    /// AVX feature level
    avx_level: AvxLevel,
    /// Cache line size for optimal memory access
    cache_line_size: usize,
    /// Vectorization width based on AVX level
    vector_width: usize,
}

/// AVX feature level for I2_S kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvxLevel {
    /// AVX2 support (256-bit vectors)
    Avx2,
    /// AVX-512 support (512-bit vectors)
    Avx512,
    /// Fallback to scalar
    None,
}

impl I2SArmKernel {
    /// Create a new I2_S ARM kernel
    pub fn new() -> Self {
        Self {
            cache_line_size: 64, // Typical ARM64 cache line size
            vector_width: 4,      // NEON f32x4 vectors
        }
    }
    
    /// Get the lookup table for signed 2-bit values
    /// Maps {-2, -1, 0, 1} to computation values
    #[inline(always)]
    fn get_i2s_lut() -> [f32; 8] {
        // Lookup table for signed 2-bit values: [-2, -1, 0, 1, padding...]
        // Index 0: -2 -> -2.0
        // Index 1: -1 -> -1.0
        // Index 2:  0 ->  0.0  
        // Index 3:  1 ->  1.0
        // Index 4-7: padding -> 0.0
        [-2.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    /// Convert signed 2-bit weight to lookup index
    #[inline(always)]
    fn i2s_to_index(weight: i8) -> usize {
        match weight {
            -2 => 0,
            -1 => 1,
            0 => 2,
            1 => 3,
            _ => 4, // Fallback for invalid values
        }
    }

    /// ARM64 NEON vectorized I2_S computation with optimized lookup operations
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compute_neon_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::aarch64::*;
        
        let vector_len = chunk_size & !3; // Process 4 elements at a time
        
        // Prepare NEON constants for efficient I2S lookup
        // Values for -2, -1, 0, 1 -> -2.0, -1.0, 0.0, 1.0
        let neg_two = vdupq_n_f32(-2.0);
        let neg_one = vdupq_n_f32(-1.0);
        let zero = vdupq_n_f32(0.0);
        let pos_one = vdupq_n_f32(1.0);
        
        // Main vectorized loop processing 4 elements at once
        for chunk_idx in (0..vector_len).step_by(4) {
            if chunk_idx + 4 <= weights.len() && chunk_idx + 4 <= inputs.len() && chunk_idx + 4 <= output.len() {
                // Load 4 f32 input values using NEON
                let input_vec = vld1q_f32(inputs.as_ptr().add(chunk_idx));
                
                // Load 4 i8 weight values and extend to i32 for comparison
                let weights_i8 = vld1_s8(weights.as_ptr().add(chunk_idx));
                let weights_i16 = vmovl_s8(weights_i8);
                let weights_i32 = vmovl_s16(vget_low_s16(weights_i16));
                
                // Create comparison masks for each I2S value
                let mask_neg_two = vceqq_s32(weights_i32, vdupq_n_s32(-2));
                let mask_neg_one = vceqq_s32(weights_i32, vdupq_n_s32(-1));
                let mask_zero = vceqq_s32(weights_i32, vdupq_n_s32(0));
                let mask_pos_one = vceqq_s32(weights_i32, vdupq_n_s32(1));
                
                // Vectorized lookup using masked selection (vbslq_f32 expects uint32x4_t mask)
                let weight_vec = vbslq_f32(mask_neg_two, neg_two,
                                 vbslq_f32(mask_neg_one, neg_one,
                                 vbslq_f32(mask_zero, zero,
                                 vbslq_f32(mask_pos_one, pos_one, zero)))); // Default to zero for invalid values
                
                // Vectorized multiply: input * weight_value
                let result_vec = vmulq_f32(input_vec, weight_vec);
                
                // Store result using NEON
                vst1q_f32(output.as_mut_ptr().add(chunk_idx), result_vec);
            }
        }
        
        // Handle remaining elements with scalar fallback
        for i in vector_len..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::i2s_to_index(weights[i]);
            let lut = Self::get_i2s_lut();
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
    
    /// Fallback scalar implementation
    #[cfg(not(target_arch = "aarch64"))]
    fn compute_scalar_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        let lut = Self::get_i2s_lut();
        
        for i in 0..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::i2s_to_index(weights[i]);
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
}

impl Default for I2SArmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl I2SLookupKernel for I2SArmKernel {
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()> {
        if weights.is_empty() || inputs.is_empty() || output.is_empty() {
            return Err(anyhow::anyhow!("Empty input arrays"));
        }
        
        let min_len = weights.len().min(inputs.len()).min(output.len());
        if min_len == 0 {
            return Err(anyhow::anyhow!("Mismatched array lengths"));
        }
        
        // Process in cache-friendly chunks
        let chunk_size = self.cache_line_size / 4; // 4 bytes per f32
        
        for chunk_start in (0..min_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(min_len);
            let current_chunk_size = chunk_end - chunk_start;
            
            let weights_chunk = &weights[chunk_start..chunk_end];
            let inputs_chunk = &inputs[chunk_start..chunk_end];
            let output_chunk = &mut output[chunk_start..chunk_end];
            
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    self.compute_neon_chunk(
                        weights_chunk,
                        inputs_chunk,
                        output_chunk,
                        current_chunk_size,
                    )?;
                }
            }
            
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.compute_scalar_chunk(
                    weights_chunk,
                    inputs_chunk,
                    output_chunk,
                    current_chunk_size,
                )?;
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "I2S-ARM64-NEON"
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            "I2S-ARM64-SCALAR"
        }
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Optimal batch size for ARM64 cache hierarchy
        8192 / std::mem::size_of::<f32>()
    }
}

impl I2SX86Kernel {
    /// Create a new I2_S x86_64 kernel with AVX2 support
    pub fn new_avx2() -> Self {
        Self {
            avx_level: AvxLevel::Avx2,
            cache_line_size: 64,
            vector_width: 8, // AVX2 processes 8 f32 at once
        }
    }
    
    /// Create a new I2_S x86_64 kernel with AVX-512 support
    pub fn new_avx512() -> Self {
        Self {
            avx_level: AvxLevel::Avx512,
            cache_line_size: 64,
            vector_width: 16, // AVX-512 processes 16 f32 at once
        }
    }
    
    /// Create a new I2_S x86_64 kernel with automatic feature detection
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                Self::new_avx512()
            } else if std::arch::is_x86_feature_detected!("avx2") {
                Self::new_avx2()
            } else {
                Self {
                    avx_level: AvxLevel::None,
                    cache_line_size: 64,
                    vector_width: 1,
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx_level: AvxLevel::None,
                cache_line_size: 64,
                vector_width: 1,
            }
        }
    }
    
    /// Get the lookup table for signed 2-bit values
    #[inline(always)]
    fn get_i2s_lut() -> [f32; 8] {
        // Same as ARM version but optimized for x86_64 cache alignment
        [-2.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    /// Convert signed 2-bit weight to lookup index
    #[inline(always)]
    fn i2s_to_index(weight: i8) -> usize {
        match weight {
            -2 => 0,
            -1 => 1,
            0 => 2,
            1 => 3,
            _ => 4, // Fallback for invalid values
        }
    }

    /// x86_64 AVX2 vectorized I2_S computation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn compute_avx2_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::x86_64::*;
        
        let lut = Self::get_i2s_lut();
        
        // Process in AVX2-sized chunks (8 f32 elements)
        for chunk_idx in (0..chunk_size).step_by(8) {
            let remaining = (chunk_size - chunk_idx).min(8);
            
            if remaining == 8 && chunk_idx + 8 <= weights.len() && chunk_idx + 8 <= inputs.len() {
                unsafe {
                    // Load 8 f32 inputs
                    let input_vec = _mm256_loadu_ps(inputs.as_ptr().add(chunk_idx));
                    
                    // Process weights and create result vector
                    let mut result_array = [0.0f32; 8];
                    for i in 0..8 {
                        let weight_idx = Self::i2s_to_index(weights[chunk_idx + i]);
                        result_array[i] = lut[weight_idx];
                    }
                    
                    // Load weight multipliers and compute
                    let weight_vec = _mm256_loadu_ps(result_array.as_ptr());
                    let result_vec = _mm256_mul_ps(input_vec, weight_vec);
                    
                    // Store result
                    _mm256_storeu_ps(output.as_mut_ptr().add(chunk_idx), result_vec);
                }
            } else {
                // Handle remaining elements with scalar fallback
                for i in 0..remaining {
                    if chunk_idx + i < weights.len() && chunk_idx + i < inputs.len() && chunk_idx + i < output.len() {
                        let weight_idx = Self::i2s_to_index(weights[chunk_idx + i]);
                        output[chunk_idx + i] = inputs[chunk_idx + i] * lut[weight_idx];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// x86_64 AVX-512 vectorized I2_S computation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn compute_avx512_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::x86_64::*;
        
        let lut = Self::get_i2s_lut();
        
        // Process in AVX-512-sized chunks (16 f32 elements)
        for chunk_idx in (0..chunk_size).step_by(16) {
            let remaining = (chunk_size - chunk_idx).min(16);
            
            if remaining == 16 && chunk_idx + 16 <= weights.len() && chunk_idx + 16 <= inputs.len() {
                unsafe {
                    // Load 16 f32 inputs
                    let input_vec = _mm512_loadu_ps(inputs.as_ptr().add(chunk_idx));
                    
                    // Process weights and create result vector
                    let mut result_array = [0.0f32; 16];
                    for i in 0..16 {
                        let weight_idx = Self::i2s_to_index(weights[chunk_idx + i]);
                        result_array[i] = lut[weight_idx];
                    }
                    
                    // Load weight multipliers and compute
                    let weight_vec = _mm512_loadu_ps(result_array.as_ptr());
                    let result_vec = _mm512_mul_ps(input_vec, weight_vec);
                    
                    // Store result
                    _mm512_storeu_ps(output.as_mut_ptr().add(chunk_idx), result_vec);
                }
            } else {
                // Handle remaining elements with scalar fallback
                for i in 0..remaining {
                    if chunk_idx + i < weights.len() && chunk_idx + i < inputs.len() && chunk_idx + i < output.len() {
                        let weight_idx = Self::i2s_to_index(weights[chunk_idx + i]);
                        output[chunk_idx + i] = inputs[chunk_idx + i] * lut[weight_idx];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Fallback scalar implementation
    #[inline]
    fn compute_scalar_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        let lut = Self::get_i2s_lut();
        
        for i in 0..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::i2s_to_index(weights[i]);
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
}

impl Default for I2SX86Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl I2SLookupKernel for I2SX86Kernel {
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()> {
        if weights.is_empty() || inputs.is_empty() || output.is_empty() {
            return Err(anyhow::anyhow!("Empty input arrays"));
        }
        
        let min_len = weights.len().min(inputs.len()).min(output.len());
        if min_len == 0 {
            return Err(anyhow::anyhow!("Mismatched array lengths"));
        }
        
        // Process in cache-friendly chunks
        let chunk_size = self.cache_line_size * 4; // 4 cache lines for better throughput
        
        for chunk_start in (0..min_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(min_len);
            let current_chunk_size = chunk_end - chunk_start;
            
            let weights_chunk = &weights[chunk_start..chunk_end];
            let inputs_chunk = &inputs[chunk_start..chunk_end];
            let output_chunk = &mut output[chunk_start..chunk_end];
            
            match self.avx_level {
                #[cfg(target_arch = "x86_64")]
                AvxLevel::Avx512 => {
                    self.compute_avx512_chunk(
                        weights_chunk,
                        inputs_chunk,
                        output_chunk,
                        current_chunk_size,
                    )?;
                }
                #[cfg(target_arch = "x86_64")]
                AvxLevel::Avx2 => {
                    self.compute_avx2_chunk(
                        weights_chunk,
                        inputs_chunk,
                        output_chunk,
                        current_chunk_size,
                    )?;
                }
                _ => {
                    self.compute_scalar_chunk(
                        weights_chunk,
                        inputs_chunk,
                        output_chunk,
                        current_chunk_size,
                    )?;
                }
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        match self.avx_level {
            AvxLevel::Avx512 => "I2S-x86_64-AVX512",
            AvxLevel::Avx2 => "I2S-x86_64-AVX2", 
            AvxLevel::None => "I2S-x86_64-SCALAR",
        }
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Optimal batch size for x86_64 cache hierarchy
        match self.avx_level {
            AvxLevel::Avx512 => 16384 / std::mem::size_of::<f32>(), // Larger for AVX-512
            AvxLevel::Avx2 => 8192 / std::mem::size_of::<f32>(),   // Medium for AVX2
            AvxLevel::None => 4096 / std::mem::size_of::<f32>(),   // Smaller for scalar
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_i2s_arm_kernel_creation() {
        let kernel = I2SArmKernel::new();
        assert_eq!(kernel.cache_line_size, 64);
        assert_eq!(kernel.vector_width, 4);
    }

    #[test]
    fn test_i2s_x86_kernel_creation() {
        let kernel_avx2 = I2SX86Kernel::new_avx2();
        assert_eq!(kernel_avx2.avx_level, AvxLevel::Avx2);
        assert_eq!(kernel_avx2.vector_width, 8);
        
        let kernel_avx512 = I2SX86Kernel::new_avx512();
        assert_eq!(kernel_avx512.avx_level, AvxLevel::Avx512);
        assert_eq!(kernel_avx512.vector_width, 16);
    }

    #[test]
    fn test_i2s_lut() {
        let lut_arm = I2SArmKernel::get_i2s_lut();
        let lut_x86 = I2SX86Kernel::get_i2s_lut();
        
        // Both should have the same lookup table
        assert_eq!(lut_arm[0..4], [-2.0, -1.0, 0.0, 1.0]);
        assert_eq!(lut_x86[0..4], [-2.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_i2s_to_index() {
        // ARM version
        assert_eq!(I2SArmKernel::i2s_to_index(-2), 0);
        assert_eq!(I2SArmKernel::i2s_to_index(-1), 1);
        assert_eq!(I2SArmKernel::i2s_to_index(0), 2);
        assert_eq!(I2SArmKernel::i2s_to_index(1), 3);
        assert_eq!(I2SArmKernel::i2s_to_index(99), 4); // Invalid fallback
        
        // x86 version
        assert_eq!(I2SX86Kernel::i2s_to_index(-2), 0);
        assert_eq!(I2SX86Kernel::i2s_to_index(-1), 1);
        assert_eq!(I2SX86Kernel::i2s_to_index(0), 2);
        assert_eq!(I2SX86Kernel::i2s_to_index(1), 3);
        assert_eq!(I2SX86Kernel::i2s_to_index(99), 4); // Invalid fallback
    }

    #[test]
    fn test_i2s_compute_basic_arm() {
        let kernel = I2SArmKernel::new();
        
        let weights = vec![-2i8, -1, 0, 1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        // Expected: [-2*2.0, -1*3.0, 0*4.0, 1*5.0] = [-4.0, -3.0, 0.0, 5.0]
        assert_relative_eq!(output[0], -4.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], -3.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_i2s_compute_basic_x86() {
        let kernel = I2SX86Kernel::new_avx2();
        
        let weights = vec![-2i8, -1, 0, 1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        // Expected: [-2*2.0, -1*3.0, 0*4.0, 1*5.0] = [-4.0, -3.0, 0.0, 5.0]
        assert_relative_eq!(output[0], -4.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], -3.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kernel_names() {
        let arm_kernel = I2SArmKernel::new();
        let x86_avx2_kernel = I2SX86Kernel::new_avx2();
        let x86_avx512_kernel = I2SX86Kernel::new_avx512();
        
        assert!(arm_kernel.name().contains("I2S-ARM64"));
        assert_eq!(x86_avx2_kernel.name(), "I2S-x86_64-AVX2");
        assert_eq!(x86_avx512_kernel.name(), "I2S-x86_64-AVX512");
    }

    #[test]
    fn test_large_array_processing() {
        let arm_kernel = I2SArmKernel::new();
        let x86_kernel = I2SX86Kernel::new_avx2();
        
        let size = 1000;
        let weights: Vec<i8> = (0..size).map(|i| match i % 4 {
            0 => -2,
            1 => -1,
            2 => 0,
            3 => 1,
            _ => 0,
        }).collect(); // -2, -1, 0, 1 pattern
        let inputs: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut output_arm = vec![0.0f32; size];
        let mut output_x86 = vec![0.0f32; size];
        
        arm_kernel.compute(&weights, &inputs, &mut output_arm).unwrap();
        x86_kernel.compute(&weights, &inputs, &mut output_x86).unwrap();
        
        // Both kernels should produce the same results
        for i in 0..size {
            assert_relative_eq!(output_arm[i], output_x86[i], epsilon = 1e-6);
        }
        
        // Verify a few specific values
        assert_relative_eq!(output_arm[0], 0.0, epsilon = 1e-6); // -2 * 0
        assert_relative_eq!(output_arm[1], -1.0, epsilon = 1e-6); // -1 * 1
        assert_relative_eq!(output_arm[2], 0.0, epsilon = 1e-6);  // 0 * 2
        assert_relative_eq!(output_arm[3], 3.0, epsilon = 1e-6);  // 1 * 3
    }
}