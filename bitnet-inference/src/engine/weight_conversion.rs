//! Weight data type conversion for BitNet inference
//!
//! This module provides conversion from raw tensor data (Vec<u8>) to typed arrays
//! suitable for inference operations. It handles ternary weights, float weights,
//! and quantized formats with memory-efficient lazy conversion.

use crate::{Result, InferenceError};
use crate::engine::model_loader::{ParameterData, ParameterDataType};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

/// Type alias for converted weight arrays
pub type ConvertedWeights = Arc<WeightArrays>;

/// Container for different weight array formats
#[derive(Debug, Clone)]
pub enum WeightArrays {
    /// Ternary weights as {-1, 0, +1} values
    Ternary(Vec<i8>),
    /// Float32 weights
    F32(Vec<f32>),
    /// Float16 weights (stored as f32 for computation)
    F16(Vec<f32>),
    /// Integer weights
    I8(Vec<i8>),
    /// Quantized weights with scale and zero point
    Quantized {
        weights: Vec<i8>,
        scale: f32,
        zero_point: i8,
        format: String,
    },
}

impl WeightArrays {
    /// Get the number of elements in the weight array
    pub fn len(&self) -> usize {
        match self {
            WeightArrays::Ternary(w) => w.len(),
            WeightArrays::F32(w) => w.len(),
            WeightArrays::F16(w) => w.len(),
            WeightArrays::I8(w) => w.len(),
            WeightArrays::Quantized { weights, .. } => weights.len(),
        }
    }

    /// Check if the weight array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the weight array as f32 slice (for ternary and float weights)
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            WeightArrays::F32(w) | WeightArrays::F16(w) => Some(w),
            _ => None,
        }
    }

    /// Get the weight array as i8 slice (for ternary and integer weights)
    pub fn as_i8_slice(&self) -> Option<&[i8]> {
        match self {
            WeightArrays::Ternary(w) | WeightArrays::I8(w) => Some(w),
            WeightArrays::Quantized { weights, .. } => Some(weights),
            _ => None,
        }
    }

    /// Get ternary weights specifically
    pub fn as_ternary(&self) -> Option<&[i8]> {
        match self {
            WeightArrays::Ternary(w) => Some(w),
            _ => None,
        }
    }
}

/// Lazy weight converter that converts weights on-demand
#[derive(Debug)]
pub struct WeightConverter {
    /// Cache of converted weights by parameter ID
    converted_cache: RwLock<HashMap<String, ConvertedWeights>>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size
    current_cache_size: RwLock<usize>,
}

impl WeightConverter {
    /// Create a new weight converter with specified cache size
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            converted_cache: RwLock::new(HashMap::new()),
            max_cache_size,
            current_cache_size: RwLock::new(0),
        }
    }

    /// Create a new weight converter with default 128MB cache
    pub fn with_default_cache() -> Self {
        Self::new(128 * 1024 * 1024) // 128MB cache
    }

    /// Convert parameter data to typed arrays
    pub fn convert_parameter(&self, param_data: &ParameterData) -> Result<ConvertedWeights> {
        // Check cache first
        let cache_key = self.generate_cache_key(param_data);
        
        {
            let cache = self.converted_cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Convert the parameter
        let converted = match param_data.dtype {
            ParameterDataType::BitnetB158 => {
                self.convert_ternary_weights(&param_data.data, &param_data.shape)?
            }
            ParameterDataType::F32 => {
                self.convert_f32_weights(&param_data.data, &param_data.shape)?
            }
            ParameterDataType::F16 => {
                self.convert_f16_weights(&param_data.data, &param_data.shape)?
            }
            ParameterDataType::I8 => {
                self.convert_i8_weights(&param_data.data, &param_data.shape)?
            }
            ParameterDataType::Quantized(ref format) => {
                self.convert_quantized_weights(&param_data.data, &param_data.shape, format)?
            }
        };

        // Add to cache if there's space
        let converted_arc = Arc::new(converted);
        self.try_cache_converted(&cache_key, converted_arc.clone());

        Ok(converted_arc)
    }

    /// Convert BitNet 1.58-bit ternary weights from packed format
    fn convert_ternary_weights(&self, data: &[u8], shape: &[usize]) -> Result<WeightArrays> {
        let total_elements = shape.iter().product::<usize>();
        
        if total_elements == 0 {
            return Ok(WeightArrays::Ternary(Vec::new()));
        }

        // BitNet 1.58 uses 2-bit packing: 4 weights per byte
        let expected_bytes = (total_elements + 3) / 4; // Round up for partial bytes
        
        if data.len() < expected_bytes {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for ternary weights: expected {} bytes, got {}",
                expected_bytes, data.len()
            )));
        }

        let mut weights = Vec::with_capacity(total_elements);
        
        // Unpack 2-bit values to ternary weights {-1, 0, +1}
        for (i, &byte) in data.iter().enumerate() {
            if weights.len() >= total_elements {
                break;
            }

            // Extract 4 values from each byte (2 bits each)
            for shift in [0, 2, 4, 6] {
                if weights.len() >= total_elements {
                    break;
                }

                let packed_value = (byte >> shift) & 0x03;
                let ternary_value = match packed_value {
                    0 => -1i8,
                    1 => 0i8,
                    2 => 1i8,
                    3 => 0i8, // Fallback for invalid values
                    _ => unreachable!(),
                };
                
                weights.push(ternary_value);
            }
        }

        // Ensure we have exactly the right number of elements
        weights.truncate(total_elements);
        
        Ok(WeightArrays::Ternary(weights))
    }

    /// Convert F32 weights from byte data
    fn convert_f32_weights(&self, data: &[u8], shape: &[usize]) -> Result<WeightArrays> {
        let total_elements = shape.iter().product::<usize>();
        let expected_bytes = total_elements * std::mem::size_of::<f32>();
        
        if data.len() < expected_bytes {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for F32 weights: expected {} bytes, got {}",
                expected_bytes, data.len()
            )));
        }

        // Convert bytes to f32 array
        let mut weights = Vec::with_capacity(total_elements);
        for chunk in data.chunks_exact(4) {
            if weights.len() >= total_elements {
                break;
            }

            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let value = f32::from_le_bytes(bytes);
            weights.push(value);
        }

        weights.truncate(total_elements);
        Ok(WeightArrays::F32(weights))
    }

    /// Convert F16 weights from byte data to F32 for computation
    fn convert_f16_weights(&self, data: &[u8], shape: &[usize]) -> Result<WeightArrays> {
        let total_elements = shape.iter().product::<usize>();
        let expected_bytes = total_elements * 2; // F16 is 2 bytes per element
        
        if data.len() < expected_bytes {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for F16 weights: expected {} bytes, got {}",
                expected_bytes, data.len()
            )));
        }

        // Convert F16 bytes to F32 array
        let mut weights = Vec::with_capacity(total_elements);
        for chunk in data.chunks_exact(2) {
            if weights.len() >= total_elements {
                break;
            }

            let bytes = [chunk[0], chunk[1]];
            let f16_bits = u16::from_le_bytes(bytes);
            let value = half::f16::from_bits(f16_bits).to_f32();
            weights.push(value);
        }

        weights.truncate(total_elements);
        Ok(WeightArrays::F16(weights))
    }

    /// Convert I8 weights from byte data
    fn convert_i8_weights(&self, data: &[u8], shape: &[usize]) -> Result<WeightArrays> {
        let total_elements = shape.iter().product::<usize>();
        
        if data.len() < total_elements {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for I8 weights: expected {} bytes, got {}",
                total_elements, data.len()
            )));
        }

        // Convert bytes to i8 array
        let weights: Vec<i8> = data[..total_elements]
            .iter()
            .map(|&b| b as i8)
            .collect();

        Ok(WeightArrays::I8(weights))
    }

    /// Convert quantized weights (basic implementation)
    fn convert_quantized_weights(&self, data: &[u8], shape: &[usize], format: &str) -> Result<WeightArrays> {
        // For now, implement basic quantized format support
        // This can be extended for specific GGUF quantization formats
        let total_elements = shape.iter().product::<usize>();
        
        match format {
            "Q8_0" => self.convert_q8_0_weights(data, total_elements),
            "Q4_0" => self.convert_q4_0_weights(data, total_elements),
            "Q5_0" => self.convert_q5_0_weights(data, total_elements),
            _ => {
                // Fallback: treat as raw bytes
                self.convert_i8_weights(data, shape)
            }
        }
    }

    /// Convert Q8_0 quantized weights
    fn convert_q8_0_weights(&self, data: &[u8], total_elements: usize) -> Result<WeightArrays> {
        // Q8_0 format: 32 weights per block, each block has scale + 32 int8 values
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 4 + BLOCK_SIZE; // 4-byte scale + 32 int8 values
        
        let num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let expected_bytes = num_blocks * BLOCK_BYTES;
        
        if data.len() < expected_bytes {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for Q8_0 weights: expected {} bytes, got {}",
                expected_bytes, data.len()
            )));
        }

        let mut weights = Vec::with_capacity(total_elements);
        
        for block_idx in 0..num_blocks {
            let block_start = block_idx * BLOCK_BYTES;
            
            // Extract scale (first 4 bytes as f32)
            let scale_bytes = [
                data[block_start],
                data[block_start + 1],
                data[block_start + 2],
                data[block_start + 3],
            ];
            let scale = f32::from_le_bytes(scale_bytes);
            
            // Extract quantized values (next 32 bytes)
            for i in 0..BLOCK_SIZE {
                if weights.len() >= total_elements {
                    break;
                }
                
                let quantized = data[block_start + 4 + i] as i8;
                weights.push(quantized);
            }
        }

        weights.truncate(total_elements);
        
        // For Q8_0, we store the dequantized values as i8 for now
        // In a full implementation, we'd need to store scale information separately
        Ok(WeightArrays::Quantized {
            weights,
            scale: 1.0, // Simplified: would need proper scale handling
            zero_point: 0,
            format: "Q8_0".to_string(),
        })
    }

    /// Convert Q4_0 quantized weights (simplified)
    fn convert_q4_0_weights(&self, data: &[u8], total_elements: usize) -> Result<WeightArrays> {
        // Q4_0 format: 32 weights per block, each block has scale + 16 bytes (2 weights per byte)
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 4 + 16; // 4-byte scale + 16 bytes (2 weights per byte)
        
        let num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let expected_bytes = num_blocks * BLOCK_BYTES;
        
        if data.len() < expected_bytes {
            return Err(InferenceError::model_load(format!(
                "Insufficient data for Q4_0 weights: expected {} bytes, got {}",
                expected_bytes, data.len()
            )));
        }

        let mut weights = Vec::with_capacity(total_elements);
        
        for block_idx in 0..num_blocks {
            let block_start = block_idx * BLOCK_BYTES;
            
            // Extract scale (first 4 bytes as f32)
            let scale_bytes = [
                data[block_start],
                data[block_start + 1],
                data[block_start + 2],
                data[block_start + 3],
            ];
            let _scale = f32::from_le_bytes(scale_bytes);
            
            // Extract quantized values (next 16 bytes, 2 weights per byte)
            for i in 0..16 {
                if weights.len() >= total_elements {
                    break;
                }
                
                let byte = data[block_start + 4 + i];
                let weight1 = ((byte & 0x0F) as i8) - 8; // Convert 0-15 to -8 to 7
                let weight2 = (((byte >> 4) & 0x0F) as i8) - 8;
                
                weights.push(weight1);
                if weights.len() < total_elements {
                    weights.push(weight2);
                }
            }
        }

        weights.truncate(total_elements);
        
        Ok(WeightArrays::Quantized {
            weights,
            scale: 1.0, // Simplified
            zero_point: 0,
            format: "Q4_0".to_string(),
        })
    }

    /// Convert Q5_0 quantized weights (simplified)
    fn convert_q5_0_weights(&self, data: &[u8], total_elements: usize) -> Result<WeightArrays> {
        // Q5_0 is more complex, for now use Q4_0 logic as fallback
        // In full implementation, this would handle the 5-bit encoding properly
        self.convert_q4_0_weights(data, total_elements)
    }

    /// Generate cache key for parameter data
    fn generate_cache_key(&self, param_data: &ParameterData) -> String {
        format!("{}_{:?}_{:?}", 
                param_data.tensor_name, 
                param_data.dtype, 
                param_data.shape)
    }

    /// Try to cache converted weights if there's space
    fn try_cache_converted(&self, key: &str, converted: ConvertedWeights) {
        let weight_size = self.estimate_weight_size(&converted);
        
        {
            let current_size = *self.current_cache_size.read().unwrap();
            if current_size + weight_size > self.max_cache_size {
                // Don't cache if it would exceed the limit
                return;
            }
        }

        {
            let mut cache = self.converted_cache.write().unwrap();
            let mut current_size = self.current_cache_size.write().unwrap();
            
            if *current_size + weight_size <= self.max_cache_size {
                cache.insert(key.to_string(), converted);
                *current_size += weight_size;
            }
        }
    }

    /// Estimate memory size of converted weights
    fn estimate_weight_size(&self, weights: &WeightArrays) -> usize {
        match weights {
            WeightArrays::Ternary(w) => w.len() * std::mem::size_of::<i8>(),
            WeightArrays::F32(w) => w.len() * std::mem::size_of::<f32>(),
            WeightArrays::F16(w) => w.len() * std::mem::size_of::<f32>(),
            WeightArrays::I8(w) => w.len() * std::mem::size_of::<i8>(),
            WeightArrays::Quantized { weights, .. } => weights.len() * std::mem::size_of::<i8>(),
        }
    }

    /// Clear the conversion cache
    pub fn clear_cache(&self) {
        let mut cache = self.converted_cache.write().unwrap();
        let mut current_size = self.current_cache_size.write().unwrap();
        
        cache.clear();
        *current_size = 0;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        let cache = self.converted_cache.read().unwrap();
        let current_size = *self.current_cache_size.read().unwrap();
        
        (cache.len(), current_size, self.max_cache_size)
    }
}

impl Default for WeightConverter {
    fn default() -> Self {
        Self::with_default_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_weight_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        // Create test data: 4 weights packed in 1 byte
        // Packed as: 00 01 10 11 (binary) -> 0 1 2 3 (decimal) -> -1 0 1 0 (ternary)
        let packed_data = vec![0b11100100]; // 3,2,1,0 in 2-bit chunks
        let shape = vec![4];
        
        let param_data = ParameterData {
            data: packed_data,
            shape,
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "test_ternary".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::Ternary(weights) => {
                assert_eq!(weights.len(), 4);
                assert_eq!(weights, &[-1, 0, 1, 0]);
            }
            _ => panic!("Expected ternary weights"),
        }
    }

    #[test]
    fn test_f32_weight_conversion() {
        let converter = WeightConverter::with_default_cache();
        
        // Create test F32 data
        let values = vec![1.0f32, -2.5f32, 0.0f32];
        let mut data = Vec::new();
        for value in &values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        
        let param_data = ParameterData {
            data,
            shape: vec![3],
            dtype: ParameterDataType::F32,
            tensor_name: "test_f32".to_string(),
        };

        let converted = converter.convert_parameter(&param_data).unwrap();
        
        match converted.as_ref() {
            WeightArrays::F32(weights) => {
                assert_eq!(weights.len(), 3);
                assert!((weights[0] - 1.0).abs() < 1e-6);
                assert!((weights[1] - (-2.5)).abs() < 1e-6);
                assert!((weights[2] - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected F32 weights"),
        }
    }

    #[test]
    fn test_cache_functionality() {
        let converter = WeightConverter::new(1024); // Small cache for testing
        
        let param_data = ParameterData {
            data: vec![0x00],
            shape: vec![4],
            dtype: ParameterDataType::BitnetB158,
            tensor_name: "cache_test".to_string(),
        };

        // First conversion
        let converted1 = converter.convert_parameter(&param_data).unwrap();
        let (cache_entries, _, _) = converter.cache_stats();
        assert_eq!(cache_entries, 1);

        // Second conversion should use cache
        let converted2 = converter.convert_parameter(&param_data).unwrap();
        assert!(Arc::ptr_eq(&converted1, &converted2));
    }
}

#[cfg(test)]
#[path = "weight_conversion_tests.rs"] 
mod comprehensive_tests;

#[cfg(test)]
#[path = "weight_conversion_debug.rs"] 
mod debug_test;

#[cfg(test)]
#[path = "weight_conversion_specific_debug.rs"] 
mod specific_debug;