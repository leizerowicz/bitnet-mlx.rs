//! Generic fallback kernels for unsupported architectures
//!
//! Provides portable implementations that work on any architecture
//! as fallbacks when SIMD optimizations are not available.

use anyhow::Result;
use crate::cpu::{TernaryLookupKernel, I2SLookupKernel};

/// Generic ternary lookup kernel (portable fallback)
pub struct GenericTernaryKernel {
    /// Block size for cache-friendly processing
    block_size: usize,
}

/// Generic I2_S lookup kernel (portable fallback)
pub struct GenericI2SKernel {
    /// Block size for cache-friendly processing
    block_size: usize,
}

impl GenericTernaryKernel {
    /// Create a new generic ternary kernel
    pub fn new() -> Self {
        Self {
            block_size: 1024, // Process in 1KB blocks
        }
    }
    
    /// Get the lookup table for ternary values
    #[inline(always)]
    fn get_ternary_lut() -> [f32; 4] {
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
}

impl Default for GenericTernaryKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl TernaryLookupKernel for GenericTernaryKernel {
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()> {
        if weights.is_empty() || inputs.is_empty() || output.is_empty() {
            return Err(anyhow::anyhow!("Empty input arrays"));
        }
        
        let min_len = weights.len().min(inputs.len()).min(output.len());
        if min_len == 0 {
            return Err(anyhow::anyhow!("Mismatched array lengths"));
        }
        
        let lut = Self::get_ternary_lut();
        
        // Simple scalar implementation
        for i in 0..min_len {
            let weight_idx = Self::ternary_to_index(weights[i]);
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "Generic-Ternary-Fallback"
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Conservative batch size for generic implementation
        4096 / std::mem::size_of::<f32>()
    }
}

impl GenericI2SKernel {
    /// Create a new generic I2_S kernel
    pub fn new() -> Self {
        Self {
            block_size: 1024, // Process in 1KB blocks
        }
    }
    
    /// Get the lookup table for signed 2-bit values
    #[inline(always)]
    fn get_i2s_lut() -> [f32; 8] {
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
}

impl Default for GenericI2SKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl I2SLookupKernel for GenericI2SKernel {
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()> {
        if weights.is_empty() || inputs.is_empty() || output.is_empty() {
            return Err(anyhow::anyhow!("Empty input arrays"));
        }
        
        let min_len = weights.len().min(inputs.len()).min(output.len());
        if min_len == 0 {
            return Err(anyhow::anyhow!("Mismatched array lengths"));
        }
        
        let lut = Self::get_i2s_lut();
        
        // Simple scalar implementation
        for i in 0..min_len {
            let weight_idx = Self::i2s_to_index(weights[i]);
            output[i] = inputs[i] * lut[weight_idx];
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "Generic-I2S-Fallback"
    }
    
    fn optimal_batch_size(&self) -> usize {
        // Conservative batch size for generic implementation
        4096 / std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_generic_ternary_kernel_creation() {
        let kernel = GenericTernaryKernel::new();
        assert_eq!(kernel.block_size, 1024);
    }

    #[test]
    fn test_generic_i2s_kernel_creation() {
        let kernel = GenericI2SKernel::new();
        assert_eq!(kernel.block_size, 1024);
    }

    #[test]
    fn test_generic_ternary_lut() {
        let lut = GenericTernaryKernel::get_ternary_lut();
        assert_eq!(lut, [-1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_generic_i2s_lut() {
        let lut = GenericI2SKernel::get_i2s_lut();
        assert_eq!(lut[0..4], [-2.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ternary_to_index() {
        assert_eq!(GenericTernaryKernel::ternary_to_index(-1), 0);
        assert_eq!(GenericTernaryKernel::ternary_to_index(0), 1);
        assert_eq!(GenericTernaryKernel::ternary_to_index(1), 2);
        assert_eq!(GenericTernaryKernel::ternary_to_index(99), 3);
    }

    #[test]
    fn test_i2s_to_index() {
        assert_eq!(GenericI2SKernel::i2s_to_index(-2), 0);
        assert_eq!(GenericI2SKernel::i2s_to_index(-1), 1);
        assert_eq!(GenericI2SKernel::i2s_to_index(0), 2);
        assert_eq!(GenericI2SKernel::i2s_to_index(1), 3);
        assert_eq!(GenericI2SKernel::i2s_to_index(99), 4);
    }

    #[test]
    fn test_generic_ternary_compute() {
        let kernel = GenericTernaryKernel::new();
        
        let weights = vec![-1i8, 0, 1, -1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        assert_relative_eq!(output[0], -2.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 4.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], -5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_generic_i2s_compute() {
        let kernel = GenericI2SKernel::new();
        
        let weights = vec![-2i8, -1, 0, 1];
        let inputs = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        kernel.compute(&weights, &inputs, &mut output).unwrap();
        
        assert_relative_eq!(output[0], -4.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], -3.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kernel_names() {
        let ternary_kernel = GenericTernaryKernel::new();
        let i2s_kernel = GenericI2SKernel::new();
        
        assert_eq!(ternary_kernel.name(), "Generic-Ternary-Fallback");
        assert_eq!(i2s_kernel.name(), "Generic-I2S-Fallback");
    }

    #[test]
    fn test_optimal_batch_sizes() {
        let ternary_kernel = GenericTernaryKernel::new();
        let i2s_kernel = GenericI2SKernel::new();
        
        let ternary_batch = ternary_kernel.optimal_batch_size();
        let i2s_batch = i2s_kernel.optimal_batch_size();
        
        assert!(ternary_batch > 0);
        assert!(i2s_batch > 0);
        assert_eq!(ternary_batch, i2s_batch); // Should be the same
    }

    #[test]
    fn test_empty_arrays() {
        let ternary_kernel = GenericTernaryKernel::new();
        let i2s_kernel = GenericI2SKernel::new();
        
        let weights = vec![];
        let inputs = vec![];
        let mut output = vec![];
        
        assert!(ternary_kernel.compute(&weights, &inputs, &mut output).is_err());
        assert!(i2s_kernel.compute(&weights, &inputs, &mut output).is_err());
    }

    #[test]
    fn test_large_arrays() {
        let ternary_kernel = GenericTernaryKernel::new();
        let i2s_kernel = GenericI2SKernel::new();
        
        let size = 10000;
        
        // Test ternary kernel
        let ternary_weights: Vec<i8> = (0..size).map(|i| match i % 3 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut ternary_output = vec![0.0f32; size];
        
        ternary_kernel.compute(&ternary_weights, &inputs, &mut ternary_output).unwrap();
        
        // Test I2S kernel
        let i2s_weights: Vec<i8> = (0..size).map(|i| match i % 4 {
            0 => -2,
            1 => -1,
            2 => 0,
            3 => 1,
            _ => 0,
        }).collect();
        let mut i2s_output = vec![0.0f32; size];
        
        i2s_kernel.compute(&i2s_weights, &inputs, &mut i2s_output).unwrap();
        
        // Verify some values
        assert_relative_eq!(ternary_output[0], 0.0, epsilon = 1e-6); // -1 * 0
        assert_relative_eq!(i2s_output[0], 0.0, epsilon = 1e-6);     // -2 * 0
    }
}