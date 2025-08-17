//! Quantization utilities and helper functions
//! 
//! This module provides common utilities, error types, and helper functions
//! used throughout the quantization system.

use candle_core::{Tensor, DType, Device, Shape};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during quantization operations
#[derive(Error, Debug)]
pub enum QuantizationError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Unsupported precision: {0}")]
    UnsupportedPrecision(String),
    
    #[error("Calibration error: {0}")]
    CalibrationError(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("Tensor operation error: {0}")]
    TensorError(#[from] candle_core::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Conversion error: {0}")]
    ConversionError(String),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    /// Data corruption errors
    #[error("Data corruption detected: {0}")]
    DataCorruption(String),
    
    #[error("Checksum verification failed: expected {expected:08X}, got {actual:08X}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    
    #[error("Data size mismatch: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch { expected: usize, actual: usize },
    
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    
    #[error("Data truncation detected: {0}")]
    DataTruncation(String),
    
    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Scaling factor for quantization operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalingFactor {
    /// The scaling value
    pub value: f32,
    /// Whether this is a per-channel scale
    pub per_channel: bool,
    /// Channel dimension if per-channel
    pub channel_dim: Option<usize>,
    /// Scale tensor for per-channel quantization
    pub scale_tensor: Option<Vec<f32>>,
}

impl ScalingFactor {
    /// Create a new per-tensor scaling factor
    pub fn per_tensor(value: f32) -> Self {
        Self {
            value,
            per_channel: false,
            channel_dim: None,
            scale_tensor: None,
        }
    }

    /// Create a new per-channel scaling factor
    pub fn per_channel(scales: Vec<f32>, channel_dim: usize) -> Self {
        let value = scales.iter().sum::<f32>() / scales.len() as f32; // Average for compatibility
        Self {
            value,
            per_channel: true,
            channel_dim: Some(channel_dim),
            scale_tensor: Some(scales),
        }
    }

    /// Get the scale for a specific channel
    pub fn get_channel_scale(&self, channel: usize) -> Result<f32, QuantizationError> {
        if !self.per_channel {
            return Ok(self.value);
        }
        
        match &self.scale_tensor {
            Some(scales) => {
                if channel < scales.len() {
                    Ok(scales[channel])
                } else {
                    Err(QuantizationError::InvalidInput(format!("Channel {} out of bounds", channel)))
                }
            }
            None => Ok(self.value),
        }
    }

    /// Convert to tensor representation
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, QuantizationError> {
        if self.per_channel {
            if let Some(ref scales) = self.scale_tensor {
                Ok(Tensor::from_slice(scales, (scales.len(),), device)?)
            } else {
                Ok(Tensor::new(self.value, device)?)
            }
        } else {
            Ok(Tensor::new(self.value, device)?)
        }
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        Self::per_tensor(1.0)
    }
}

/// Quantization utilities
pub struct QuantizationUtils;

impl QuantizationUtils {
    /// Compute optimal scaling factor for symmetric quantization
    pub fn compute_symmetric_scale(tensor: &Tensor, num_bits: u8) -> Result<f32, QuantizationError> {
        let abs_max = tensor.abs()?.max_all()?.to_scalar::<f32>()?;
        let max_val = (1 << (num_bits - 1)) - 1;
        Ok(abs_max / max_val as f32)
    }

    /// Compute scaling factor and zero point for asymmetric quantization
    pub fn compute_asymmetric_params(tensor: &Tensor, num_bits: u8) -> Result<(f32, i32), QuantizationError> {
        let min_val = tensor.min_all()?.to_scalar::<f32>()?;
        let max_val = tensor.max_all()?.to_scalar::<f32>()?;
        
        let qmin = 0i32;
        let qmax = (1 << num_bits) - 1;
        
        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let zero_point = qmin as f32 - min_val / scale;
        let zero_point = zero_point.round().clamp(qmin as f32, qmax as f32) as i32;
        
        Ok((scale, zero_point))
    }

    /// Quantize tensor to specified bit width with symmetric quantization
    pub fn quantize_symmetric(
        tensor: &Tensor,
        scale: f32,
        num_bits: u8,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let max_val = (1 << (num_bits - 1)) - 1;
        let min_val = -(1 << (num_bits - 1));
        
        let scaled = tensor.div(&Tensor::new(scale, device)?)?;
        let quantized = scaled.round()?.clamp(min_val as f32, max_val as f32)?;
        
        Ok(quantized)
    }

    /// Quantize tensor to specified bit width with asymmetric quantization
    pub fn quantize_asymmetric(
        tensor: &Tensor,
        scale: f32,
        zero_point: i32,
        num_bits: u8,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let qmin = 0;
        let qmax = (1 << num_bits) - 1;
        
        let scaled = tensor.div(&Tensor::new(scale, device)?)?;
        let shifted = scaled.add(&Tensor::new(zero_point as f32, device)?)?;
        let quantized = shifted.round()?.clamp(qmin as f32, qmax as f32)?;
        
        Ok(quantized)
    }

    /// Dequantize symmetric quantized tensor
    pub fn dequantize_symmetric(
        quantized: &Tensor,
        scale: f32,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let dequantized = quantized.mul(&Tensor::new(scale, device)?)?;
        Ok(dequantized)
    }

    /// Dequantize asymmetric quantized tensor
    pub fn dequantize_asymmetric(
        quantized: &Tensor,
        scale: f32,
        zero_point: i32,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let shifted = quantized.sub(&Tensor::new(zero_point as f32, device)?)?;
        let dequantized = shifted.mul(&Tensor::new(scale, device)?)?;
        Ok(dequantized)
    }

    /// Compute quantization error (MSE) between original and dequantized tensors
    pub fn compute_quantization_error(
        original: &Tensor,
        dequantized: &Tensor,
    ) -> Result<f32, QuantizationError> {
        let diff = original.sub(dequantized)?;
        let squared_diff = diff.sqr()?;
        let mse = squared_diff.mean_all()?.to_scalar::<f32>()?;
        Ok(mse)
    }

    /// Compute signal-to-noise ratio (SNR) for quantization
    pub fn compute_snr(
        original: &Tensor,
        dequantized: &Tensor,
    ) -> Result<f32, QuantizationError> {
        let signal_power = original.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let noise_power = Self::compute_quantization_error(original, dequantized)?;
        
        if noise_power < f32::EPSILON {
            return Ok(f32::INFINITY);
        }
        
        let snr_db = 10.0 * (signal_power / noise_power).log10();
        Ok(snr_db)
    }

    /// Find optimal clipping threshold for outlier handling
    pub fn find_optimal_clip_threshold(
        tensor: &Tensor,
        percentile: f32,
    ) -> Result<f32, QuantizationError> {
        // Convert tensor to vector for percentile calculation
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let mut sorted_data = data;
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((percentile / 100.0) * sorted_data.len() as f32) as usize;
        let threshold = sorted_data[index.min(sorted_data.len() - 1)];
        
        Ok(threshold)
    }

    /// Apply gradient clipping for quantization-aware training
    pub fn clip_gradients(
        gradients: &Tensor,
        max_norm: f32,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let grad_norm = gradients.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        
        if grad_norm <= max_norm {
            return Ok(gradients.clone());
        }
        
        let scale_factor = max_norm / grad_norm;
        let clipped = gradients.mul(&Tensor::new(scale_factor, device)?)?;
        Ok(clipped)
    }

    /// Compute entropy of quantized values (for analysis)
    pub fn compute_entropy(quantized: &Tensor) -> Result<f32, QuantizationError> {
        let data = quantized.flatten_all()?.to_vec1::<f32>()?;
        let mut value_counts = std::collections::HashMap::new();
        
        for &value in &data {
            *value_counts.entry(value as i32).or_insert(0) += 1;
        }
        
        let total_count = data.len() as f32;
        let mut entropy = 0.0;
        
        for count in value_counts.values() {
            let probability = *count as f32 / total_count;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        Ok(entropy)
    }

    /// Check if tensor is suitable for quantization
    pub fn validate_for_quantization(tensor: &Tensor) -> Result<(), QuantizationError> {
        // Check dynamic range
        let min_val = tensor.min_all()?.to_scalar::<f32>()?;
        let max_val = tensor.max_all()?.to_scalar::<f32>()?;
        let dynamic_range = max_val - min_val;
        
        if dynamic_range < f32::EPSILON {
            return Err(QuantizationError::NumericalError("Tensor has zero dynamic range".to_string()));
        }
        
        // Check for extreme values that might indicate NaN/Inf
        if !min_val.is_finite() || !max_val.is_finite() {
            return Err(QuantizationError::NumericalError("Tensor contains non-finite values".to_string()));
        }
        
        Ok(())
    }

    /// Estimate memory savings from quantization
    pub fn estimate_memory_savings(
        original_shape: &Shape,
        original_dtype: DType,
        quantized_dtype: DType,
        has_scales: bool,
        has_zero_points: bool,
    ) -> MemorySavingsEstimate {
        let num_elements = original_shape.elem_count();
        let original_size = num_elements * original_dtype.size_in_bytes();
        
        let mut quantized_size = num_elements * quantized_dtype.size_in_bytes();
        
        // Add overhead for scales and zero points
        if has_scales {
            quantized_size += std::mem::size_of::<f32>(); // Assuming per-tensor scale
        }
        if has_zero_points {
            quantized_size += std::mem::size_of::<i32>(); // Assuming per-tensor zero point
        }
        
        let compression_ratio = original_size as f32 / quantized_size as f32;
        let memory_saved = original_size.saturating_sub(quantized_size);
        let savings_percentage = (memory_saved as f32 / original_size as f32) * 100.0;
        
        MemorySavingsEstimate {
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
            memory_saved_bytes: memory_saved,
            compression_ratio,
            savings_percentage,
        }
    }

    /// Round and clip a value to the specified range
    ///
    /// This function first rounds the input value to the nearest integer,
    /// then clips it to the range [min_val, max_val].
    ///
    /// # Arguments
    /// * `x` - The input value to round and clip
    /// * `min_val` - The minimum value for clipping
    /// * `max_val` - The maximum value for clipping
    ///
    /// # Returns
    /// The rounded and clipped value
    ///
    /// # Example
    /// ```rust
    /// use bitnet_quant::quantization::utils::QuantizationUtils;
    ///
    /// let result = QuantizationUtils::round_clip(1.7, -1.0, 1.0);
    /// assert_eq!(result, 1.0);
    ///
    /// let result = QuantizationUtils::round_clip(-2.3, -1.0, 1.0);
    /// assert_eq!(result, -1.0);
    ///
    /// let result = QuantizationUtils::round_clip(0.4, -1.0, 1.0);
    /// assert_eq!(result, 0.0);
    /// ```
    pub fn round_clip(x: f32, min_val: f32, max_val: f32) -> f32 {
        x.round().clamp(min_val, max_val)
    }

    /// Round and clip a tensor to the specified range
    ///
    /// This function applies round_clip element-wise to a tensor.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor to round and clip
    /// * `min_val` - The minimum value for clipping
    /// * `max_val` - The maximum value for clipping
    /// * `device` - The device to perform operations on
    ///
    /// # Returns
    /// A new tensor with rounded and clipped values
    pub fn round_clip_tensor(
        tensor: &Tensor,
        min_val: f32,
        max_val: f32,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let rounded = tensor.round()?;
        let clipped = rounded.clamp(min_val, max_val)?;
        Ok(clipped)
    }
}

/// Memory savings estimation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySavingsEstimate {
    pub original_size_bytes: usize,
    pub quantized_size_bytes: usize,
    pub memory_saved_bytes: usize,
    pub compression_ratio: f32,
    pub savings_percentage: f32,
}

/// Bit manipulation utilities for quantization
pub struct BitUtils;

impl BitUtils {
    /// Pack multiple low-bit values into bytes
    pub fn pack_bits(values: &[u8], bits_per_value: u8) -> Vec<u8> {
        if bits_per_value >= 8 {
            return values.to_vec();
        }
        
        let values_per_byte = 8 / bits_per_value;
        let mut packed = Vec::new();
        
        for chunk in values.chunks(values_per_byte as usize) {
            let mut byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                byte |= (value & ((1 << bits_per_value) - 1)) << (i as u8 * bits_per_value);
            }
            packed.push(byte);
        }
        
        packed
    }

    /// Unpack low-bit values from bytes
    pub fn unpack_bits(packed: &[u8], bits_per_value: u8, num_values: usize) -> Vec<u8> {
        if bits_per_value >= 8 {
            return packed.to_vec();
        }
        
        let values_per_byte = 8 / bits_per_value;
        let mask = (1 << bits_per_value) - 1;
        let mut unpacked = Vec::new();
        
        for &byte in packed {
            for i in 0..values_per_byte {
                if unpacked.len() >= num_values {
                    break;
                }
                let value = (byte >> (i * bits_per_value)) & mask;
                unpacked.push(value);
            }
        }
        
        unpacked.truncate(num_values);
        unpacked
    }

    /// Convert signed values to unsigned for packing
    pub fn signed_to_unsigned(value: i8, bits: u8) -> u8 {
        let offset = 1 << (bits - 1);
        ((value as i32) + offset) as u8
    }

    /// Convert unsigned values back to signed after unpacking
    pub fn unsigned_to_signed(value: u8, bits: u8) -> i8 {
        let offset = 1 << (bits - 1);
        (value as i32 - offset) as i8
    }
}

/// Calibration utilities for quantization
pub struct CalibrationUtils;

impl CalibrationUtils {
    /// Collect statistics from calibration data
    pub fn collect_statistics(data: &[Tensor]) -> Result<CalibrationStatistics, QuantizationError> {
        if data.is_empty() {
            return Err(QuantizationError::CalibrationError("No calibration data provided".to_string()));
        }

        let mut min_vals = Vec::new();
        let mut max_vals = Vec::new();
        let mut mean_vals = Vec::new();
        let mut std_vals = Vec::new();

        for tensor in data {
            min_vals.push(tensor.min_all()?.to_scalar::<f32>()?);
            max_vals.push(tensor.max_all()?.to_scalar::<f32>()?);
            mean_vals.push(tensor.mean_all()?.to_scalar::<f32>()?);
            // Use mean absolute deviation as approximation for std
            let abs_tensor = tensor.abs()?;
            std_vals.push(abs_tensor.mean_all()?.to_scalar::<f32>()?);
        }

        let global_min = min_vals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let global_max = max_vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_mean = mean_vals.iter().sum::<f32>() / mean_vals.len() as f32;
        let avg_std = std_vals.iter().sum::<f32>() / std_vals.len() as f32;

        Ok(CalibrationStatistics {
            global_min,
            global_max,
            average_mean: avg_mean,
            average_std: avg_std,
            sample_count: data.len(),
            dynamic_range: global_max - global_min,
        })
    }

    /// Find optimal quantization parameters using KL divergence
    pub fn find_optimal_threshold_kl(
        data: &[Tensor],
        num_bins: usize,
    ) -> Result<f32, QuantizationError> {
        // Simplified KL divergence-based threshold finding
        // In practice, this would implement the full algorithm
        let stats = Self::collect_statistics(data)?;
        Ok(stats.global_max.abs().max(stats.global_min.abs()))
    }
}

/// Calibration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStatistics {
    pub global_min: f32,
    pub global_max: f32,
    pub average_mean: f32,
    pub average_std: f32,
    pub sample_count: usize,
    pub dynamic_range: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_scaling_factor_per_tensor() {
        let scale = ScalingFactor::per_tensor(2.0);
        assert!(!scale.per_channel);
        assert_eq!(scale.value, 2.0);
        assert_eq!(scale.get_channel_scale(0).unwrap(), 2.0);
    }

    #[test]
    fn test_scaling_factor_per_channel() {
        let scales = vec![1.0, 2.0, 3.0];
        let scale = ScalingFactor::per_channel(scales.clone(), 0);
        assert!(scale.per_channel);
        assert_eq!(scale.get_channel_scale(1).unwrap(), 2.0);
        assert_eq!(scale.get_channel_scale(2).unwrap(), 3.0);
    }

    #[test]
    fn test_bit_utils_pack_unpack() {
        let values = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = BitUtils::pack_bits(&values, 2);
        let unpacked = BitUtils::unpack_bits(&packed, 2, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_signed_unsigned_conversion() {
        let signed_val = -1i8;
        let unsigned = BitUtils::signed_to_unsigned(signed_val, 2);
        let back_to_signed = BitUtils::unsigned_to_signed(unsigned, 2);
        assert_eq!(signed_val, back_to_signed);
    }

    #[test]
    fn test_memory_savings_estimate() {
        let shape = Shape::from_dims(&[100, 100]);
        let estimate = QuantizationUtils::estimate_memory_savings(
            &shape,
            DType::F32,
            DType::U8,
            true,
            false,
        );
        
        assert!(estimate.compression_ratio > 1.0);
        assert!(estimate.savings_percentage > 0.0);
        assert_eq!(estimate.original_size_bytes, 100 * 100 * 4); // f32 = 4 bytes
    }

    #[test]
    fn test_quantization_error_display() {
        let error = QuantizationError::InvalidInput("test error".to_string());
        assert!(format!("{}", error).contains("Invalid input"));
    }

    #[test]
    fn test_round_clip_basic() {
        // Test basic rounding and clipping
        assert_eq!(QuantizationUtils::round_clip(1.7, -1.0, 1.0), 1.0);
        assert_eq!(QuantizationUtils::round_clip(-2.3, -1.0, 1.0), -1.0);
        assert_eq!(QuantizationUtils::round_clip(0.4, -1.0, 1.0), 0.0);
        assert_eq!(QuantizationUtils::round_clip(0.6, -1.0, 1.0), 1.0);
        assert_eq!(QuantizationUtils::round_clip(-0.6, -1.0, 1.0), -1.0);
    }

    #[test]
    fn test_round_clip_edge_cases() {
        // Test exact boundary values
        assert_eq!(QuantizationUtils::round_clip(1.0, -1.0, 1.0), 1.0);
        assert_eq!(QuantizationUtils::round_clip(-1.0, -1.0, 1.0), -1.0);
        assert_eq!(QuantizationUtils::round_clip(0.0, -1.0, 1.0), 0.0);
        
        // Test values exactly at 0.5 (Rust rounds away from zero)
        assert_eq!(QuantizationUtils::round_clip(0.5, -1.0, 1.0), 1.0); // 0.5 rounds to 1
        assert_eq!(QuantizationUtils::round_clip(-0.5, -1.0, 1.0), -1.0); // -0.5 rounds to -1
        assert_eq!(QuantizationUtils::round_clip(1.5, -1.0, 1.0), 1.0); // 1.5 rounds to 2, then clipped to 1
        assert_eq!(QuantizationUtils::round_clip(-1.5, -1.0, 1.0), -1.0); // -1.5 rounds to -2, then clipped to -1
    }

    #[test]
    fn test_round_clip_tensor() {
        let device = Device::Cpu;
        let data = vec![1.7f32, -2.3, 0.4, 0.6, -0.6];
        let tensor = Tensor::from_slice(&data, (5,), &device).unwrap();
        
        let result = QuantizationUtils::round_clip_tensor(&tensor, -1.0, 1.0, &device).unwrap();
        let result_data = result.to_vec1::<f32>().unwrap();
        
        let expected = vec![1.0f32, -1.0, 0.0, 1.0, -1.0];
        assert_eq!(result_data, expected);
    }
}