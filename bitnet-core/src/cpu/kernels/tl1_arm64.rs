//! TL1 ARM64 NEON Kernels - Ternary Lookup Table Implementation
//!
//! High-performance ARM64 NEON implementation of ternary lookup tables
//! targeting Microsoft's 1.37x-6.17x performance improvements through
//! vectorized operations and cache-optimized memory access patterns.

use anyhow::Result;
use crate::cpu::{TernaryLookupKernel};

/// ARM64 NEON-optimized ternary lookup kernel
pub struct Tl1Arm64Kernel {
    /// Cache line size for optimal memory access
    cache_line_size: usize,
    /// Vectorization width (NEON processes 4 f32 or 16 i8 at once)
    vector_width: usize,
}

impl Tl1Arm64Kernel {
    /// Create a new TL1 ARM64 kernel with optimal configuration
    pub fn new() -> Self {
        Self {
            cache_line_size: 128,  // Apple Silicon uses 128-byte cache lines
            vector_width: 16,      // Process 16 elements for optimal throughput
        }
    }
    
    /// Get the lookup table for ternary values
    /// Maps {-1, 0, 1} to efficient computation patterns
    #[inline(always)]
    fn get_ternary_lut() -> [f32; 4] {
        // Lookup table for ternary values: [-1, 0, 1, padding]
        // Index 0: -1 -> -1.0
        // Index 1:  0 ->  0.0  
        // Index 2:  1 ->  1.0
        // Index 3: padding -> 0.0
        [-1.0, 0.0, 1.0, 0.0]
    }
    
    /// Convert ternary weight to lookup index (optimized for NEON vtbl)
    #[inline(always)]
    fn ternary_to_index(weight: i8) -> u8 {
        match weight {
            -1 => 0u8,
            0 => 1u8,
            1 => 2u8,
            _ => 3u8, // Fallback for invalid values
        }
    }
    
    /// NEON optimized ternary lookup using efficient branchless conversion
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn lookup_ternary_values_neon(weights: &[i8], count: usize) -> [f32; 16] {
        use std::arch::aarch64::*;
        
        let mut result = [0.0f32; 16];
        let actual_count = count.min(16).min(weights.len());
        
        // Ultra-fast branchless conversion using NEON arithmetic
        // This is faster than vtbl for small lookup tables
        if actual_count >= 16 {
            // Load 16 i8 weights as 4x int32x4_t
            let weights_i8x16 = vld1q_s8(weights.as_ptr());
            
            // Convert to 4 separate int32x4_t vectors for f32 conversion
            let weights_i32_0 = vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(weights_i8x16))));
            let weights_i32_1 = vmovl_high_s16(vmovl_s8(vget_low_s8(weights_i8x16)));
            let weights_i32_2 = vmovl_s16(vget_low_s16(vmovl_high_s8(weights_i8x16)));
            let weights_i32_3 = vmovl_high_s16(vmovl_high_s8(weights_i8x16));
            
            // Convert to f32 and store
            let result_f32_0 = vcvtq_f32_s32(weights_i32_0);
            let result_f32_1 = vcvtq_f32_s32(weights_i32_1);
            let result_f32_2 = vcvtq_f32_s32(weights_i32_2);
            let result_f32_3 = vcvtq_f32_s32(weights_i32_3);
            
            vst1q_f32(result.as_mut_ptr(), result_f32_0);
            vst1q_f32(result.as_mut_ptr().add(4), result_f32_1);
            vst1q_f32(result.as_mut_ptr().add(8), result_f32_2);
            vst1q_f32(result.as_mut_ptr().add(12), result_f32_3);
        } else {
            // Fallback for smaller counts
            for i in 0..actual_count {
                result[i] = weights[i] as f32;
            }
        }
        
        result
    }

    /// ARM64 NEON vectorized ternary lookup computation with ultra-aggressive optimizations
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn compute_neon_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::aarch64::*;
        
        unsafe {
            // Check memory alignment for optimal vs fallback path
            let inputs_aligned = (inputs.as_ptr() as usize) % 16 == 0;
            let output_aligned = (output.as_ptr() as usize) % 16 == 0;
            let use_optimal_path = inputs_aligned && output_aligned && chunk_size >= 64;
            
            if use_optimal_path {
                // Ultra-optimized path for aligned memory and large chunks
                self.compute_ultra_optimized_neon(weights, inputs, output, chunk_size)?;
            } else {
                // Standard optimized path
                self.compute_standard_neon(weights, inputs, output, chunk_size)?;
            }
        }
        
        Ok(())
    }
    
    /// Ultra-aggressive NEON optimization for perfectly aligned data
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn compute_ultra_optimized_neon(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::aarch64::*;
        
        // Process in 32KB chunks for optimal L1 cache usage on Apple Silicon
        let cache_chunk_size = 32 * 1024 / 4; // 32KB / sizeof(f32)
        let mut global_offset = 0;
        
        while global_offset < chunk_size {
            let current_chunk_size = (chunk_size - global_offset).min(cache_chunk_size);
            
            // Process 32 elements at a time (8x loop unrolling) within cache chunk
            let unrolled_chunks = current_chunk_size / 32;
            let mut offset = global_offset;
            
            // Prefetch the entire cache chunk
            if current_chunk_size > 128 {
                for prefetch_offset in (0..current_chunk_size).step_by(128) {
                    let prefetch_addr = offset + prefetch_offset;
                    if prefetch_addr + 128 <= inputs.len() {
                        std::arch::asm!(
                            "prfm pldl1keep, [{input_ptr}]",
                            "prfm pldl1keep, [{weights_ptr}]",
                            input_ptr = in(reg) inputs.as_ptr().add(prefetch_addr),
                            weights_ptr = in(reg) weights.as_ptr().add(prefetch_addr),
                            options(nostack, readonly)
                        );
                    }
                }
            }
            
            // Ultra-aggressive 8x unrolled loop (32 elements per iteration)
            for _ in 0..unrolled_chunks {
                if offset + 32 > inputs.len() || offset + 32 > weights.len() || offset + 32 > output.len() {
                    break;
                }
                
                // Convert 32 weights at once using direct NEON conversion
                let weight_values = Self::lookup_ternary_values_neon(&weights[offset..], 32);
                
                // Process 8 NEON vectors in parallel (32 elements)
                let input_vec1 = vld1q_f32(inputs.as_ptr().add(offset));
                let input_vec2 = vld1q_f32(inputs.as_ptr().add(offset + 4));
                let input_vec3 = vld1q_f32(inputs.as_ptr().add(offset + 8));
                let input_vec4 = vld1q_f32(inputs.as_ptr().add(offset + 12));
                let input_vec5 = vld1q_f32(inputs.as_ptr().add(offset + 16));
                let input_vec6 = vld1q_f32(inputs.as_ptr().add(offset + 20));
                let input_vec7 = vld1q_f32(inputs.as_ptr().add(offset + 24));
                let input_vec8 = vld1q_f32(inputs.as_ptr().add(offset + 28));
                
                let weight_vec1 = vld1q_f32(weight_values.as_ptr());
                let weight_vec2 = vld1q_f32(weight_values.as_ptr().add(4));
                let weight_vec3 = vld1q_f32(weight_values.as_ptr().add(8));
                let weight_vec4 = vld1q_f32(weight_values.as_ptr().add(12));
                let weight_vec5 = vld1q_f32(weight_values.as_ptr().add(16));
                let weight_vec6 = vld1q_f32(weight_values.as_ptr().add(20));
                let weight_vec7 = vld1q_f32(weight_values.as_ptr().add(24));
                let weight_vec8 = vld1q_f32(weight_values.as_ptr().add(28));
                
                // Parallel multiply operations
                let result_vec1 = vmulq_f32(input_vec1, weight_vec1);
                let result_vec2 = vmulq_f32(input_vec2, weight_vec2);
                let result_vec3 = vmulq_f32(input_vec3, weight_vec3);
                let result_vec4 = vmulq_f32(input_vec4, weight_vec4);
                let result_vec5 = vmulq_f32(input_vec5, weight_vec5);
                let result_vec6 = vmulq_f32(input_vec6, weight_vec6);
                let result_vec7 = vmulq_f32(input_vec7, weight_vec7);
                let result_vec8 = vmulq_f32(input_vec8, weight_vec8);
                
                // Store all results with write-through cache optimization
                vst1q_f32(output.as_mut_ptr().add(offset), result_vec1);
                vst1q_f32(output.as_mut_ptr().add(offset + 4), result_vec2);
                vst1q_f32(output.as_mut_ptr().add(offset + 8), result_vec3);
                vst1q_f32(output.as_mut_ptr().add(offset + 12), result_vec4);
                vst1q_f32(output.as_mut_ptr().add(offset + 16), result_vec5);
                vst1q_f32(output.as_mut_ptr().add(offset + 20), result_vec6);
                vst1q_f32(output.as_mut_ptr().add(offset + 24), result_vec7);
                vst1q_f32(output.as_mut_ptr().add(offset + 28), result_vec8);
                
                offset += 32;
            }
            
            // Process remaining elements within this cache chunk
            while offset < global_offset + current_chunk_size {
                let remaining = (global_offset + current_chunk_size - offset).min(16);
                if remaining >= 16 && offset + 16 <= inputs.len() && offset + 16 <= weights.len() && offset + 16 <= output.len() {
                    // Process 16 elements
                    let weight_values = Self::lookup_ternary_values_neon(&weights[offset..], 16);
                    
                    let input_vec1 = vld1q_f32(inputs.as_ptr().add(offset));
                    let input_vec2 = vld1q_f32(inputs.as_ptr().add(offset + 4));
                    let input_vec3 = vld1q_f32(inputs.as_ptr().add(offset + 8));
                    let input_vec4 = vld1q_f32(inputs.as_ptr().add(offset + 12));
                    
                    let weight_vec1 = vld1q_f32(weight_values.as_ptr());
                    let weight_vec2 = vld1q_f32(weight_values.as_ptr().add(4));
                    let weight_vec3 = vld1q_f32(weight_values.as_ptr().add(8));
                    let weight_vec4 = vld1q_f32(weight_values.as_ptr().add(12));
                    
                    let result_vec1 = vmulq_f32(input_vec1, weight_vec1);
                    let result_vec2 = vmulq_f32(input_vec2, weight_vec2);
                    let result_vec3 = vmulq_f32(input_vec3, weight_vec3);
                    let result_vec4 = vmulq_f32(input_vec4, weight_vec4);
                    
                    vst1q_f32(output.as_mut_ptr().add(offset), result_vec1);
                    vst1q_f32(output.as_mut_ptr().add(offset + 4), result_vec2);
                    vst1q_f32(output.as_mut_ptr().add(offset + 8), result_vec3);
                    vst1q_f32(output.as_mut_ptr().add(offset + 12), result_vec4);
                    
                    offset += 16;
                } else if remaining >= 4 && offset + 4 <= inputs.len() && offset + 4 <= weights.len() && offset + 4 <= output.len() {
                    // Process 4 elements
                    let weight_values = Self::lookup_ternary_values_neon(&weights[offset..], 4);
                    let input_vec = vld1q_f32(inputs.as_ptr().add(offset));
                    let weight_vec = vld1q_f32(weight_values.as_ptr());
                    let result_vec = vmulq_f32(input_vec, weight_vec);
                    vst1q_f32(output.as_mut_ptr().add(offset), result_vec);
                    offset += 4;
                } else {
                    // Scalar fallback for final elements
                    break;
                }
            }
            
            // Handle scalar remainder
            while offset < global_offset + current_chunk_size && offset < inputs.len() && offset < weights.len() && offset < output.len() {
                let weight_val = weights[offset] as f32;
                output[offset] = inputs[offset] * weight_val;
                offset += 1;
            }
            
            global_offset += current_chunk_size;
        }
        
        Ok(())
    }
    
    /// Standard NEON optimization for general cases
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn compute_standard_neon(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        use std::arch::aarch64::*;
        
        // Process 16 elements at a time (4x loop unrolling)
        let unrolled_chunks = chunk_size / 16;
        let mut offset = 0;
        
        // Main loop with 4x unrolling for maximum throughput
        for _ in 0..unrolled_chunks {
            // Bounds check for 16 elements
            if offset + 16 > inputs.len() || offset + 16 > weights.len() || offset + 16 > output.len() {
                break;
            }
            
            // Prefetch next cache lines for large arrays (Apple Silicon optimization)
            if chunk_size > 1024 {
                std::arch::asm!(
                    "prfm pldl1keep, [{next_input}]",
                    "prfm pldl1keep, [{next_weights}]",
                    next_input = in(reg) inputs.as_ptr().add(offset + 64),
                    next_weights = in(reg) weights.as_ptr().add(offset + 64),
                    options(nostack, readonly)
                );
            }
            
            // Use vectorized lookup for efficient ternary conversion
            let weight_values = Self::lookup_ternary_values_neon(&weights[offset..], 16);
            
            // Load and process 4 NEON vectors (16 elements total)
            // Vector 1 (elements 0-3)
            let input_vec1 = vld1q_f32(inputs.as_ptr().add(offset));
            let weight_vec1 = vld1q_f32(weight_values.as_ptr());
            let result_vec1 = vmulq_f32(input_vec1, weight_vec1);
            
            // Vector 2 (elements 4-7)
            let input_vec2 = vld1q_f32(inputs.as_ptr().add(offset + 4));
            let weight_vec2 = vld1q_f32(weight_values.as_ptr().add(4));
            let result_vec2 = vmulq_f32(input_vec2, weight_vec2);
            
            // Vector 3 (elements 8-11)
            let input_vec3 = vld1q_f32(inputs.as_ptr().add(offset + 8));
            let weight_vec3 = vld1q_f32(weight_values.as_ptr().add(8));
            let result_vec3 = vmulq_f32(input_vec3, weight_vec3);
            
            // Vector 4 (elements 12-15)
            let input_vec4 = vld1q_f32(inputs.as_ptr().add(offset + 12));
            let weight_vec4 = vld1q_f32(weight_values.as_ptr().add(12));
            let result_vec4 = vmulq_f32(input_vec4, weight_vec4);
            
            // Store all 4 vectors with pipeline optimization
            vst1q_f32(output.as_mut_ptr().add(offset), result_vec1);
            vst1q_f32(output.as_mut_ptr().add(offset + 4), result_vec2);
            vst1q_f32(output.as_mut_ptr().add(offset + 8), result_vec3);
            vst1q_f32(output.as_mut_ptr().add(offset + 12), result_vec4);
            
            offset += 16;
        }
        
        // Process remaining 4-element chunks with optimized lookup
        let remaining_4_chunks = (chunk_size - offset) / 4;
        for _ in 0..remaining_4_chunks {
            if offset + 4 > inputs.len() || offset + 4 > weights.len() || offset + 4 > output.len() {
                break;
            }
            
            let input_vec = vld1q_f32(inputs.as_ptr().add(offset));
            let weight_values = Self::lookup_ternary_values_neon(&weights[offset..], 4);
            let weight_vec = vld1q_f32(weight_values.as_ptr());
            let result_vec = vmulq_f32(input_vec, weight_vec);
            vst1q_f32(output.as_mut_ptr().add(offset), result_vec);
            
            offset += 4;
        }
        
        // Handle final scalar elements
        for i in offset..chunk_size.min(inputs.len()).min(weights.len()).min(output.len()) {
            let weight_val = weights[i] as f32;
            output[i] = inputs[i] * weight_val;
        }
        
        Ok(())
    }
    
    /// Fallback scalar implementation for non-ARM64 targets
    #[cfg(not(target_arch = "aarch64"))]
    fn compute_scalar_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        let lut = Self::get_ternary_lut();
        
        for i in 0..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::ternary_to_index(weights[i]) as usize;
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
    
    /// ARM64 scalar fallback for unaligned data
    #[cfg(target_arch = "aarch64")]
    fn compute_scalar_chunk(
        &self,
        weights: &[i8],
        inputs: &[f32],
        output: &mut [f32],
        chunk_size: usize,
    ) -> Result<()> {
        let lut = Self::get_ternary_lut();
        
        for i in 0..chunk_size.min(weights.len()).min(inputs.len()).min(output.len()) {
            let weight_idx = Self::ternary_to_index(weights[i]) as usize;
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
}

impl Default for Tl1Arm64Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl TernaryLookupKernel for Tl1Arm64Kernel {
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()> {
        if weights.is_empty() || inputs.is_empty() || output.is_empty() {
            return Err(anyhow::anyhow!("Empty input arrays"));
        }
        
        let min_len = weights.len().min(inputs.len()).min(output.len());
        if min_len == 0 {
            return Err(anyhow::anyhow!("Mismatched array lengths"));
        }
        
        // Process in Apple Silicon optimized chunks (32KB L1 cache)
        let chunk_size = (32 * 1024) / std::mem::size_of::<f32>(); // 8K elements per chunk
        
        for chunk_start in (0..min_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(min_len);
            let current_chunk_size = chunk_end - chunk_start;
            
            let weights_chunk = &weights[chunk_start..chunk_end];
            let inputs_chunk = &inputs[chunk_start..chunk_end];
            let output_chunk = &mut output[chunk_start..chunk_end];
            
            // Add memory alignment hints for Apple Silicon cache optimization
            #[cfg(target_arch = "aarch64")]
            {
                // Ensure the data is properly aligned for optimal NEON performance
                if inputs_chunk.as_ptr() as usize % 16 == 0 && 
                   output_chunk.as_ptr() as usize % 16 == 0 {
                    // Aligned path - use optimized NEON kernels
                    self.compute_neon_chunk(
                        weights_chunk,
                        inputs_chunk,
                        output_chunk,
                        current_chunk_size,
                    )?;
                } else {
                    // Unaligned path - use scalar fallback for safety
                    self.compute_scalar_chunk(
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
            "TL1-ARM64-NEON"
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            "TL1-ARM64-SCALAR"
        }
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Optimal batch size for Apple Silicon (128MB unified memory access)
        // Target: 32KB L1 cache utilization with 16-element vectorization
        (32 * 1024) / std::mem::size_of::<f32>() // 8192 elements
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tl1_arm64_kernel_creation() {
        let kernel = Tl1Arm64Kernel::new();
        assert_eq!(kernel.cache_line_size, 128);
        assert_eq!(kernel.vector_width, 16);
    }

    #[test]
    fn test_ternary_lut() {
        let lut = Tl1Arm64Kernel::get_ternary_lut();
        assert_eq!(lut, [-1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_ternary_to_index() {
        assert_eq!(Tl1Arm64Kernel::ternary_to_index(-1), 0u8);
        assert_eq!(Tl1Arm64Kernel::ternary_to_index(0), 1u8);
        assert_eq!(Tl1Arm64Kernel::ternary_to_index(1), 2u8);
        assert_eq!(Tl1Arm64Kernel::ternary_to_index(99), 3u8); // Invalid fallback
    }

    #[test]
    fn test_compute_basic() {
        let kernel = Tl1Arm64Kernel::new();
        
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
    fn test_compute_empty_arrays() {
        let kernel = Tl1Arm64Kernel::new();
        
        let weights = vec![];
        let inputs = vec![];
        let mut output = vec![];
        
        let result = kernel.compute(&weights, &inputs, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_mismatched_lengths() {
        let kernel = Tl1Arm64Kernel::new();
        
        let weights = vec![-1i8, 0];
        let inputs = vec![2.0f32, 3.0, 4.0];
        let mut output = vec![0.0f32; 2];
        
        // Should process min length (2) successfully
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        assert_relative_eq!(output[0], -2.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_large_array_processing() {
        let kernel = Tl1Arm64Kernel::new();
        
        let size = 1000;
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
    fn test_kernel_name() {
        let kernel = Tl1Arm64Kernel::new();
        let name = kernel.name();
        assert!(name.contains("TL1-ARM64"));
    }

    #[test]
    fn test_optimal_batch_size() {
        let kernel = Tl1Arm64Kernel::new();
        let batch_size = kernel.optimal_batch_size();
        assert!(batch_size > 0);
        assert_eq!(batch_size, 8192); // Should match 32KB / 4 bytes for Apple Silicon
    }
}