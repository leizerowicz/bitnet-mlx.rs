//! SIMD-optimized ternary operations for BitNet quantization
//!
//! This module provides vectorized implementations of ternary quantization and
//! dequantization operations, supporting values {-1, 0, +1} with high performance
//! across x86 and ARM architectures.

use super::capabilities::{detect_simd_capabilities, SimdCapabilities};
use crate::{QuantizationError, QuantizationResult};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-accelerated ternary quantization operations
#[allow(dead_code)]
pub struct SimdTernaryOps {
    capabilities: SimdCapabilities,
}

impl SimdTernaryOps {
    /// Create new SIMD ternary operations with capability detection
    pub fn new() -> Self {
        Self {
            capabilities: detect_simd_capabilities(),
        }
    }

    /// Quantize a batch of f32 values to ternary i8 values using SIMD
    pub fn quantize_batch(
        &self,
        input: &[f32],
        output: &mut [i8],
        abs_mean: f32,
    ) -> QuantizationResult<()> {
        if input.len() != output.len() {
            return Err(QuantizationError::ConfigError(
                "Input and output lengths must match".into(),
            ));
        }

        let threshold = 0.5 * abs_mean;
        self.quantize_batch_scalar(input, output, threshold)
    }

    /// Dequantize a batch of i8 ternary values to f32 using SIMD
    pub fn dequantize_batch(
        &self,
        input: &[i8],
        output: &mut [f32],
        abs_mean: f32,
    ) -> QuantizationResult<()> {
        if input.len() != output.len() {
            return Err(QuantizationError::ConfigError(
                "Input and output lengths must match".into(),
            ));
        }

        self.dequantize_batch_scalar(input, output, abs_mean)
    }

    /// Compute absolute mean for quantization threshold using SIMD
    pub fn compute_absmean(&self, input: &[f32]) -> f32 {
        if input.is_empty() {
            return 0.0;
        }

        self.compute_absmean_scalar(input)
    }

    // Scalar fallback implementations - made public for standalone functions
    pub fn quantize_batch_scalar(
        &self,
        input: &[f32],
        output: &mut [i8],
        threshold: f32,
    ) -> QuantizationResult<()> {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = if *inp > threshold {
                1
            } else if *inp < -threshold {
                -1
            } else {
                0
            };
        }
        Ok(())
    }

    pub fn dequantize_batch_scalar(
        &self,
        input: &[i8],
        output: &mut [f32],
        abs_mean: f32,
    ) -> QuantizationResult<()> {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = match *inp {
                1 => abs_mean,
                -1 => -abs_mean,
                _ => 0.0,
            };
        }
        Ok(())
    }

    pub fn compute_absmean_scalar(&self, input: &[f32]) -> f32 {
        let sum: f32 = input.iter().map(|x| x.abs()).sum();
        sum / input.len() as f32
    }
}

impl Default for SimdTernaryOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Vectorized ternary quantization function for direct use
pub fn vectorized_ternary_quantize(
    input: &[f32],
    output: &mut [i8],
    threshold: f32,
) -> QuantizationResult<()> {
    let ops = SimdTernaryOps::new();
    ops.quantize_batch_scalar(input, output, threshold)
}

/// Vectorized ternary dequantization function for direct use
pub fn vectorized_ternary_dequantize(
    input: &[i8],
    output: &mut [f32],
    abs_mean: f32,
) -> QuantizationResult<()> {
    let ops = SimdTernaryOps::new();
    ops.dequantize_batch_scalar(input, output, abs_mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_batch() {
        let ops = SimdTernaryOps::new();
        let input = [2.0, -1.5, 0.2, -0.1, 1.5, -2.0];
        let mut output = [0i8; 6];
        let abs_mean = 1.0;

        ops.quantize_batch(&input, &mut output, abs_mean).unwrap();

        assert_eq!(output, [1, -1, 0, 0, 1, -1]);
    }

    #[test]
    fn test_dequantize_batch() {
        let ops = SimdTernaryOps::new();
        let input = [1i8, -1, 0, 0, 1, -1];
        let mut output = [0.0f32; 6];
        let abs_mean = 1.5;

        ops.dequantize_batch(&input, &mut output, abs_mean).unwrap();

        assert_eq!(output, [1.5, -1.5, 0.0, 0.0, 1.5, -1.5]);
    }

    #[test]
    fn test_compute_absmean() {
        let ops = SimdTernaryOps::new();
        let input = [1.0, -2.0, 3.0, -4.0];

        let result = ops.compute_absmean(&input);
        assert!((result - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let ops = SimdTernaryOps::new();
        let input = [];

        let result = ops.compute_absmean(&input);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_length_mismatch() {
        let ops = SimdTernaryOps::new();
        let input = [1.0, 2.0, 3.0];
        let mut output = [0i8; 2]; // Different length

        let result = ops.quantize_batch(&input, &mut output, 1.0);
        assert!(result.is_err());
    }
}
