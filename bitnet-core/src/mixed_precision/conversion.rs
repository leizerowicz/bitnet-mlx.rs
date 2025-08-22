//! Precision Conversion Utilities
//!
//! This module provides utilities for converting tensors between different precision levels,
//! with optimized conversion strategies and minimal precision loss.

use super::{MixedPrecisionError, MixedPrecisionResult};
use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::HybridMemoryPool;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for precision conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionStrategy {
    /// Direct conversion with potential precision loss
    Direct,
    /// Scaled conversion to minimize precision loss
    Scaled,
    /// Quantization-aware conversion
    QuantizationAware,
    /// Stochastic rounding for better precision preservation
    StochasticRounding,
    /// Custom conversion with user-defined parameters
    Custom,
}

impl Default for ConversionStrategy {
    fn default() -> Self {
        ConversionStrategy::Scaled
    }
}

/// Configuration for precision conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Conversion strategy
    pub strategy: ConversionStrategy,
    /// Whether to preserve tensor metadata during conversion
    pub preserve_metadata: bool,
    /// Whether to validate conversion results
    pub validate_results: bool,
    /// Tolerance for conversion validation
    pub validation_tolerance: f32,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Custom conversion parameters
    pub custom_params: HashMap<String, f32>,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            strategy: ConversionStrategy::default(),
            preserve_metadata: true,
            validate_results: false,
            validation_tolerance: 1e-6,
            use_simd: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Precision converter for BitNet tensors
#[derive(Debug)]
pub struct PrecisionConverter {
    /// Conversion configuration
    config: ConversionConfig,
    /// Memory pool for tensor allocation
    memory_pool: HybridMemoryPool,
    /// Conversion statistics
    stats: ConversionStats,
}

impl PrecisionConverter {
    /// Create a new precision converter
    pub fn new(config: ConversionConfig) -> MixedPrecisionResult<Self> {
        let memory_pool = HybridMemoryPool::new()
            .map_err(|e| MixedPrecisionError::MemoryAllocationError(e.to_string()))?;

        Ok(Self {
            config,
            memory_pool,
            stats: ConversionStats::default(),
        })
    }

    /// Convert a tensor to a different precision
    pub fn convert_tensor(
        &mut self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let start_time = std::time::Instant::now();
        
        // Check if conversion is needed
        if tensor.dtype() == target_precision {
            return Ok(tensor.clone());
        }

        // Validate conversion compatibility
        self.validate_conversion(tensor.dtype(), target_precision)?;

        // Perform conversion based on strategy
        let converted_tensor = match self.config.strategy {
            ConversionStrategy::Direct => self.convert_direct(tensor, target_precision)?,
            ConversionStrategy::Scaled => self.convert_scaled(tensor, target_precision)?,
            ConversionStrategy::QuantizationAware => self.convert_quantization_aware(tensor, target_precision)?,
            ConversionStrategy::StochasticRounding => self.convert_stochastic(tensor, target_precision)?,
            ConversionStrategy::Custom => self.convert_custom(tensor, target_precision)?,
        };

        // Validate results if enabled
        if self.config.validate_results {
            self.validate_conversion_result(tensor, &converted_tensor)?;
        }

        // Update statistics
        let conversion_time = start_time.elapsed();
        self.stats.record_conversion(
            tensor.dtype(),
            target_precision,
            conversion_time,
            tensor.size_bytes(),
            converted_tensor.size_bytes(),
        );

        Ok(converted_tensor)
    }

    /// Convert tensor using direct strategy
    fn convert_direct(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let device = tensor.device();
        let shape = tensor.shape();
        
        // Convert to candle tensor for processing
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Convert to target candle dtype
        let target_candle_dtype = target_precision.to_candle_dtype();
        let converted_candle = candle_tensor.to_dtype(target_candle_dtype)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert dtype: {}", e),
            })?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(converted_candle, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        Ok(converted_tensor)
    }

    /// Convert tensor using scaled strategy to minimize precision loss
    fn convert_scaled(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let device = tensor.device();
        let shape = tensor.shape();
        
        // Get tensor data as candle tensor
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Calculate optimal scaling factor
        let scale_factor = self.calculate_scale_factor(&candle_tensor, target_precision)?;

        // Apply scaling before conversion
        let scaled_tensor = if scale_factor != 1.0 {
            candle_tensor.mul(&Tensor::new(scale_factor, &device)?)
                .map_err(|e| MixedPrecisionError::ConversionError {
                    from: tensor.dtype(),
                    to: target_precision,
                    reason: format!("Failed to apply scaling: {}", e),
                })?
        } else {
            candle_tensor
        };

        // Convert to target precision
        let target_candle_dtype = target_precision.to_candle_dtype();
        let converted_candle = scaled_tensor.to_dtype(target_candle_dtype)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert dtype: {}", e),
            })?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(converted_candle, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        // Store scale factor in metadata if needed
        if self.config.preserve_metadata && scale_factor != 1.0 {
            converted_tensor.set_name(Some(format!("scaled_{}_{}", 
                tensor.name().unwrap_or_default(), scale_factor)));
        }

        Ok(converted_tensor)
    }

    /// Convert tensor using quantization-aware strategy
    fn convert_quantization_aware(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        // For quantization-aware conversion, we use the quantization pipeline
        if target_precision.is_quantized() {
            self.convert_to_quantized(tensor, target_precision)
        } else {
            self.convert_from_quantized(tensor, target_precision)
        }
    }

    /// Convert tensor using stochastic rounding
    fn convert_stochastic(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let device = tensor.device();
        
        // Get tensor data
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Apply stochastic rounding
        let rounded_tensor = self.apply_stochastic_rounding(&candle_tensor, target_precision)?;

        // Convert to target precision
        let target_candle_dtype = target_precision.to_candle_dtype();
        let converted_candle = rounded_tensor.to_dtype(target_candle_dtype)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert dtype: {}", e),
            })?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(converted_candle, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        Ok(converted_tensor)
    }

    /// Convert tensor using custom strategy
    fn convert_custom(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        // Custom conversion based on parameters
        let custom_scale = self.config.custom_params.get("scale").copied().unwrap_or(1.0);
        let custom_offset = self.config.custom_params.get("offset").copied().unwrap_or(0.0);

        let device = tensor.device();
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Apply custom transformation
        let transformed_tensor = if custom_scale != 1.0 || custom_offset != 0.0 {
            let scale_tensor = Tensor::new(custom_scale, &device)?;
            let offset_tensor = Tensor::new(custom_offset, &device)?;
            candle_tensor.mul(&scale_tensor)?.add(&offset_tensor)?
        } else {
            candle_tensor
        };

        // Convert to target precision
        let target_candle_dtype = target_precision.to_candle_dtype();
        let converted_candle = transformed_tensor.to_dtype(target_candle_dtype)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert dtype: {}", e),
            })?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(converted_candle, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        Ok(converted_tensor)
    }

    /// Convert tensor to quantized precision
    fn convert_to_quantized(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        // Use the quantization pipeline for proper quantization
        match target_precision {
            BitNetDType::BitNet158 => self.convert_to_ternary(tensor),
            BitNetDType::I8 | BitNetDType::I4 | BitNetDType::I2 | BitNetDType::I1 => {
                self.convert_to_integer_quantized(tensor, target_precision)
            }
            _ => self.convert_direct(tensor, target_precision),
        }
    }

    /// Convert tensor from quantized precision
    fn convert_from_quantized(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        // Dequantize first, then convert to target precision
        let dequantized = self.dequantize_tensor(tensor)?;
        self.convert_direct(&dequantized, target_precision)
    }

    /// Convert tensor to ternary (BitNet 1.58) precision
    fn convert_to_ternary(&self, tensor: &BitNetTensor) -> MixedPrecisionResult<BitNetTensor> {
        let device = tensor.device();
        let shape = tensor.shape();
        
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: BitNetDType::BitNet158,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Apply ternary quantization: values -> {-1, 0, +1}
        let abs_tensor = candle_tensor.abs()?;
        let threshold = abs_tensor.mean_all()?.to_scalar::<f32>()? * 0.7; // BitNet threshold
        
        let threshold_tensor = Tensor::new(threshold, &device)?.broadcast_as(candle_tensor.shape())?;
        let mask = abs_tensor.gt(&threshold_tensor)?;
        let signs = candle_tensor.sign()?;
        let ternary_tensor = signs.mul(&mask.to_dtype(candle_tensor.dtype())?)?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(ternary_tensor, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: BitNetDType::BitNet158,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        Ok(converted_tensor)
    }

    /// Convert tensor to integer quantized precision
    fn convert_to_integer_quantized(
        &self,
        tensor: &BitNetTensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let device = tensor.device();
        
        let candle_tensor = tensor.to_candle()
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to convert to candle tensor: {}", e),
            })?;

        // Get quantization range for target precision
        let (min_val, max_val) = target_precision.value_range()
            .ok_or_else(|| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: "Target precision does not have a defined range".to_string(),
            })?;

        // Calculate scale and zero point
        let tensor_min = candle_tensor.min_all()?.to_scalar::<f32>()?;
        let tensor_max = candle_tensor.max_all()?.to_scalar::<f32>()?;
        
        let scale = (tensor_max - tensor_min) / (max_val as f32 - min_val as f32);
        let zero_point = min_val as f32 - tensor_min / scale;

        // Apply quantization
        let scale_tensor = Tensor::new(1.0 / scale, &device)?;
        let zero_point_tensor = Tensor::new(zero_point, &device)?;
        
        let quantized = candle_tensor.mul(&scale_tensor)?
            .add(&zero_point_tensor)?
            .round()?
            .clamp(min_val as f32, max_val as f32)?;

        // Create new BitNet tensor
        let converted_tensor = BitNetTensor::from_candle(quantized, &self.memory_pool)
            .map_err(|e| MixedPrecisionError::ConversionError {
                from: tensor.dtype(),
                to: target_precision,
                reason: format!("Failed to create BitNet tensor: {}", e),
            })?;

        Ok(converted_tensor)
    }

    /// Dequantize a tensor
    fn dequantize_tensor(&self, tensor: &BitNetTensor) -> MixedPrecisionResult<BitNetTensor> {
        // For now, just convert to F32 as dequantization
        // In a full implementation, this would use stored scale factors
        self.convert_direct(tensor, BitNetDType::F32)
    }

    /// Calculate optimal scale factor for conversion
    fn calculate_scale_factor(
        &self,
        tensor: &Tensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<f32> {
        if let Some((min_val, max_val)) = target_precision.value_range() {
            let tensor_min = tensor.min_all()?.to_scalar::<f32>()?;
            let tensor_max = tensor.max_all()?.to_scalar::<f32>()?;
            
            let tensor_range = tensor_max - tensor_min;
            let target_range = max_val as f32 - min_val as f32;
            
            if tensor_range > 0.0 {
                Ok(target_range / tensor_range)
            } else {
                Ok(1.0)
            }
        } else {
            Ok(1.0)
        }
    }

    /// Apply stochastic rounding to tensor
    fn apply_stochastic_rounding(
        &self,
        tensor: &Tensor,
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<Tensor> {
        // For simplicity, just apply regular rounding
        // In a full implementation, this would use random rounding
        Ok(tensor.round()?)
    }

    /// Validate conversion compatibility
    fn validate_conversion(
        &self,
        from_precision: BitNetDType,
        to_precision: BitNetDType,
    ) -> MixedPrecisionResult<()> {
        // Check for potentially problematic conversions
        if from_precision.is_float() && to_precision.is_quantized() {
            // Float to quantized conversion - potential precision loss
            if to_precision.bits_per_element() < 8 {
                // Very low precision - warn but allow
            }
        }

        if from_precision.is_quantized() && to_precision.is_float() {
            // Quantized to float conversion - generally safe
        }

        Ok(())
    }

    /// Validate conversion result
    fn validate_conversion_result(
        &self,
        original: &BitNetTensor,
        converted: &BitNetTensor,
    ) -> MixedPrecisionResult<()> {
        // Check shape preservation
        if original.shape() != converted.shape() {
            return Err(MixedPrecisionError::ConversionError {
                from: original.dtype(),
                to: converted.dtype(),
                reason: "Shape mismatch after conversion".to_string(),
            });
        }

        // Check for reasonable value ranges (if both are float types)
        if original.dtype().is_float() && converted.dtype().is_float() {
            let original_candle = original.to_candle().map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to validate: {}", e))
            })?;
            let converted_candle = converted.to_candle().map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to validate: {}", e))
            })?;

            let diff = original_candle.sub(&converted_candle).map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to compute difference: {}", e))
            })?;
            let max_diff = diff.abs().map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to compute abs: {}", e))
            })?.max_all().map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to compute max: {}", e))
            })?.to_scalar::<f32>().map_err(|e| {
                MixedPrecisionError::ValidationError(format!("Failed to extract scalar: {}", e))
            })?;

            if max_diff > self.config.validation_tolerance {
                return Err(MixedPrecisionError::ValidationError(
                    format!("Conversion error {} exceeds tolerance {}", 
                        max_diff, self.config.validation_tolerance)
                ));
            }
        }

        Ok(())
    }

    /// Get conversion statistics
    pub fn get_stats(&self) -> &ConversionStats {
        &self.stats
    }

    /// Reset conversion statistics
    pub fn reset_stats(&mut self) {
        self.stats = ConversionStats::default();
    }
}

/// Statistics for precision conversions
#[derive(Debug, Default)]
pub struct ConversionStats {
    /// Total number of conversions performed
    pub total_conversions: usize,
    /// Total time spent on conversions
    pub total_time_ms: f32,
    /// Total memory saved through conversions
    pub total_memory_saved_bytes: i64,
    /// Conversion counts by precision pair
    pub conversion_counts: HashMap<(BitNetDType, BitNetDType), usize>,
    /// Average conversion times by precision pair
    pub average_times_ms: HashMap<(BitNetDType, BitNetDType), f32>,
}

impl ConversionStats {
    /// Record a conversion
    pub fn record_conversion(
        &mut self,
        from_precision: BitNetDType,
        to_precision: BitNetDType,
        duration: std::time::Duration,
        original_size: usize,
        converted_size: usize,
    ) {
        self.total_conversions += 1;
        let duration_ms = duration.as_secs_f32() * 1000.0;
        self.total_time_ms += duration_ms;
        self.total_memory_saved_bytes += original_size as i64 - converted_size as i64;

        let precision_pair = (from_precision, to_precision);
        *self.conversion_counts.entry(precision_pair).or_insert(0) += 1;
        
        let current_avg = self.average_times_ms.get(&precision_pair).copied().unwrap_or(0.0);
        let count = self.conversion_counts[&precision_pair] as f32;
        let new_avg = (current_avg * (count - 1.0) + duration_ms) / count;
        self.average_times_ms.insert(precision_pair, new_avg);
    }

    /// Get average conversion time
    pub fn average_conversion_time_ms(&self) -> f32 {
        if self.total_conversions > 0 {
            self.total_time_ms / self.total_conversions as f32
        } else {
            0.0
        }
    }

    /// Get memory efficiency (positive means memory saved)
    pub fn memory_efficiency(&self) -> f32 {
        self.total_memory_saved_bytes as f32 / (1024.0 * 1024.0) // Convert to MB
    }
}

/// Batch conversion utilities
pub struct BatchConverter {
    converter: PrecisionConverter,
}

impl BatchConverter {
    /// Create a new batch converter
    pub fn new(config: ConversionConfig) -> MixedPrecisionResult<Self> {
        Ok(Self {
            converter: PrecisionConverter::new(config)?,
        })
    }

    /// Convert multiple tensors to the same target precision
    pub fn convert_batch(
        &mut self,
        tensors: &[BitNetTensor],
        target_precision: BitNetDType,
    ) -> MixedPrecisionResult<Vec<BitNetTensor>> {
        let mut converted_tensors = Vec::with_capacity(tensors.len());
        
        for tensor in tensors {
            let converted = self.converter.convert_tensor(tensor, target_precision)?;
            converted_tensors.push(converted);
        }
        
        Ok(converted_tensors)
    }

    /// Convert tensors to different target precisions
    pub fn convert_batch_mixed(
        &mut self,
        tensors: &[(BitNetTensor, BitNetDType)],
    ) -> MixedPrecisionResult<Vec<BitNetTensor>> {
        let mut converted_tensors = Vec::with_capacity(tensors.len());
        
        for (tensor, target_precision) in tensors {
            let converted = self.converter.convert_tensor(tensor, *target_precision)?;
            converted_tensors.push(converted);
        }
        
        Ok(converted_tensors)
    }

    /// Get conversion statistics
    pub fn get_stats(&self) -> &ConversionStats {
        self.converter.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_conversion_config() {
        let config = ConversionConfig::default();
        assert_eq!(config.strategy, ConversionStrategy::Scaled);
        assert!(config.preserve_metadata);
    }

    #[test]
    fn test_precision_converter_creation() {
        let config = ConversionConfig::default();
        let converter = PrecisionConverter::new(config);
        assert!(converter.is_ok());
    }

    #[test]
    fn test_conversion_strategy_enum() {
        assert_eq!(ConversionStrategy::default(), ConversionStrategy::Scaled);
        assert_ne!(ConversionStrategy::Direct, ConversionStrategy::QuantizationAware);
    }

    #[test]
    fn test_conversion_stats() {
        let mut stats = ConversionStats::default();
        assert_eq!(stats.total_conversions, 0);
        assert_eq!(stats.average_conversion_time_ms(), 0.0);
        
        stats.record_conversion(
            BitNetDType::F32,
            BitNetDType::I8,
            std::time::Duration::from_millis(10),
            1000,
            250,
        );
        
        assert_eq!(stats.total_conversions, 1);
        assert!(stats.average_conversion_time_ms() > 0.0);
        assert!(stats.memory_efficiency() > 0.0); // Memory was saved
    }

    #[test]
    fn test_batch_converter_creation() {
        let config = ConversionConfig::default();
        let batch_converter = BatchConverter::new(config);
        assert!(batch_converter.is_ok());
    }
}