//! TL2 x86_64 AVX Kernels - Ternary Lookup Table Implementation
//!
//! High-performance x86_64 AVX2/AVX-512 implementation of ternary lookup tables
//! targeting Microsoft's 2.5x-8.0x performance improvements through
//! vectorized operations and cache-optimized memory access patterns.

use anyhow::Result;
use crate::cpu::{TernaryLookupKernel};

/// x86_64 AVX-optimized ternary lookup kernel
pub struct Tl2X86_64Kernel {
    /// AVX feature level
    avx_level: AvxLevel,
    /// Cache line size for optimal memory access
    cache_line_size: usize,
    /// Vectorization width based on AVX level
    vector_width: usize,
}

/// AVX feature level detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvxLevel {
    /// AVX2 support (256-bit vectors)
    Avx2,
    /// AVX-512 support (512-bit vectors)
    Avx512,
    /// Fallback to scalar
    None,
}

impl Tl2X86_64Kernel {
    /// Create a new TL2 x86_64 kernel with AVX2 support
    pub fn new_avx2() -> Self {
        Self {
            avx_level: AvxLevel::Avx2,
            cache_line_size: 64,
            vector_width: 8, // AVX2 processes 8 f32 at once
        }
    }
    
    /// Create a new TL2 x86_64 kernel with AVX-512 support
    pub fn new_avx512() -> Self {
        Self {
            avx_level: AvxLevel::Avx512,
            cache_line_size: 64,
            vector_width: 16, // AVX-512 processes 16 f32 at once
        }
    }
    
    /// Create a new TL2 x86_64 kernel with automatic feature detection
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
    
    /// Get the lookup table for ternary values optimized for x86_64
    #[inline(always)]
    fn get_ternary_lut() -> [f32; 4] {
        // Lookup table for ternary values: [-1, 0, 1, padding]
        // Optimized for x86_64 cache line alignment
        [-1.0, 0.0, 1.0, 0.0]
    }
    
    /// Convert ternary weight to lookup index
    #[inline(always)]
    fn ternary_to_index(weight: i8) -> usize {
        match weight {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => 3, // Fallback for invalid values
        }
    }

    /// x86_64 AVX2 vectorized ternary lookup computation
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
        
        let lut = Self::get_ternary_lut();
        
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
                        let weight_idx = Self::ternary_to_index(weights[chunk_idx + i]);
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
                        let weight_idx = Self::ternary_to_index(weights[chunk_idx + i]);
                        output[chunk_idx + i] = inputs[chunk_idx + i] * lut[weight_idx];
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// x86_64 AVX-512 vectorized ternary lookup computation
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
        
        let lut = Self::get_ternary_lut();
        
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
                        let weight_idx = Self::ternary_to_index(weights[chunk_idx + i]);
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
                        let weight_idx = Self::ternary_to_index(weights[chunk_idx + i]);
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
        let lut = Self::get_ternary_lut();
        
        for i in 0..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::ternary_to_index(weights[i]);
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
}

impl Default for Tl2X86_64Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl TernaryLookupKernel for Tl2X86_64Kernel {
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
            AvxLevel::Avx512 => "TL2-x86_64-AVX512",
            AvxLevel::Avx2 => "TL2-x86_64-AVX2", 
            AvxLevel::None => "TL2-x86_64-SCALAR",
        }
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Optimal batch size for x86_64 cache hierarchy
        // L1 cache (typical): 32KB
        // Process in 16KB chunks for good cache locality
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
    fn test_tl2_x86_64_kernel_creation() {
        let kernel_avx2 = Tl2X86_64Kernel::new_avx2();
        assert_eq!(kernel_avx2.avx_level, AvxLevel::Avx2);
        assert_eq!(kernel_avx2.vector_width, 8);
        
        let kernel_avx512 = Tl2X86_64Kernel::new_avx512();
        assert_eq!(kernel_avx512.avx_level, AvxLevel::Avx512);
        assert_eq!(kernel_avx512.vector_width, 16);
    }

    #[test]
    fn test_auto_detection() {
        let kernel = Tl2X86_64Kernel::new();
        // Should detect some valid AVX level or fallback
        match kernel.avx_level {
            AvxLevel::Avx2 | AvxLevel::Avx512 | AvxLevel::None => {
                // Valid detection
            }
        }
    }

    #[test]
    fn test_ternary_lut() {
        let lut = Tl2X86_64Kernel::get_ternary_lut();
        assert_eq!(lut, [-1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_ternary_to_index() {
        assert_eq!(Tl2X86_64Kernel::ternary_to_index(-1), 0);
        assert_eq!(Tl2X86_64Kernel::ternary_to_index(0), 1);
        assert_eq!(Tl2X86_64Kernel::ternary_to_index(1), 2);
        assert_eq!(Tl2X86_64Kernel::ternary_to_index(99), 3); // Invalid fallback
    }

    #[test]
    fn test_compute_basic_avx2() {
        let kernel = Tl2X86_64Kernel::new_avx2();
        
        let weights = vec![-1i8, 0, 1, -1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        // Expected: [-1*2.0, 0*3.0, 1*4.0, -1*5.0] = [-2.0, 0.0, 4.0, -5.0]
        assert_relative_eq!(output[0], -2.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 4.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], -5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_basic_avx512() {
        let kernel = Tl2X86_64Kernel::new_avx512();
        
        let weights = vec![-1i8, 0, 1, -1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        // Expected: [-1*2.0, 0*3.0, 1*4.0, -1*5.0] = [-2.0, 0.0, 4.0, -5.0]
        assert_relative_eq!(output[0], -2.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 4.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], -5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_large_array_avx2() {
        let kernel = Tl2X86_64Kernel::new_avx2();
        
        let size = 1024;
        let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; size];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        // Verify a few values
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6); // -1 * 0
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);  // 0 * 1
        assert_relative_eq!(output[2], 2.0, epsilon = 1e-6);  // 1 * 2
        assert_relative_eq!(output[3], -3.0, epsilon = 1e-6); // -1 * 3
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(Tl2X86_64Kernel::new_avx2().name(), "TL2-x86_64-AVX2");
        assert_eq!(Tl2X86_64Kernel::new_avx512().name(), "TL2-x86_64-AVX512");
    }

    #[test]
    fn test_optimal_batch_sizes() {
        let kernel_avx2 = Tl2X86_64Kernel::new_avx2();
        let kernel_avx512 = Tl2X86_64Kernel::new_avx512();
        
        let batch_avx2 = kernel_avx2.optimal_batch_size();
        let batch_avx512 = kernel_avx512.optimal_batch_size();
        
        assert!(batch_avx2 > 0);
        assert!(batch_avx512 > 0);
        assert!(batch_avx512 >= batch_avx2); // AVX-512 should handle larger batches
    }

    #[test]
    fn test_compute_empty_arrays() {
        let kernel = Tl2X86_64Kernel::new_avx2();
        
        let weights = vec![];
        let inputs = vec![];
        let mut output = vec![];
        
        let result = kernel.compute(&weights, &inputs, &mut output);
        assert!(result.is_err());
    }
}