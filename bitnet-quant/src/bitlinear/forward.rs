//! Forward Pass Implementation for BitLinear Layer
//!
//! This module implements the forward pass for BitLinear layers, including
//! optimizations for quantized matrix multiplication and memory-efficient operations.

use candle_core::Tensor;
use crate::bitlinear::{BitLinear, BitLinearResult, BitLinearError};
use crate::quantization::QuantizedWeight;

#[cfg(feature = "tracing")]
use tracing::{debug, info, instrument};

/// Trait for forward pass operations on BitLinear layers
pub trait BitLinearForward {
    /// Perform forward pass with the given input tensor
    /// 
    /// This method handles the full forward pass including:
    /// - Input validation and preprocessing
    /// - Weight quantization (with caching)
    /// - Quantized matrix multiplication
    /// - Bias addition (if enabled)
    /// - Output post-processing
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input tensor with shape [..., in_features]
    /// 
    /// # Returns
    /// 
    /// Output tensor with shape [..., out_features]
    fn forward(&self, input: &Tensor) -> BitLinearResult<Tensor>;
    
    /// Perform forward pass with pre-quantized weights
    /// 
    /// This method allows using pre-computed quantized weights, which can be
    /// useful for inference scenarios where the same weights are used multiple times.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input tensor with shape [..., in_features]
    /// * `quantized_weights` - Pre-quantized weights
    /// 
    /// # Returns
    /// 
    /// Output tensor with shape [..., out_features]
    fn forward_with_quantized_weights(
        &self,
        input: &Tensor,
        quantized_weights: &QuantizedWeight,
    ) -> BitLinearResult<Tensor>;
}

impl BitLinearForward for BitLinear {
    #[cfg_attr(feature = "tracing", instrument(skip(self, input), fields(layer = %self.layer_name())))]
    fn forward(&self, input: &Tensor) -> BitLinearResult<Tensor> {
        #[cfg(feature = "tracing")]
        debug!("Starting forward pass for layer: {}", self.layer_name());
        
        // Validate input dimensions
        let input_shape = input.shape();
        let input_dims = input_shape.dims();
        
        if input_dims.is_empty() {
            return Err(BitLinearError::ShapeMismatch {
                expected: vec![1, self.config().in_features],
                actual: input_shape.dims().to_vec(),
            });
        }
        
        let last_dim = input_dims[input_dims.len() - 1];
        if last_dim != self.config().in_features {
            return Err(BitLinearError::ShapeMismatch {
                expected: vec![input_dims[0], self.config().in_features],
                actual: input_shape.dims().to_vec(),
            });
        }
        
        #[cfg(feature = "tracing")]
        debug!("Input validation passed - shape: {:?}", input_shape);
        
        // Get quantized weights (from cache or quantize on-demand)
        let quantized_weights = self.get_quantized_weights()?;
        
        #[cfg(feature = "tracing")]
        debug!("Retrieved quantized weights for layer: {}", self.layer_name());
        
        // Perform forward pass with quantized weights
        self.forward_with_quantized_weights(input, &quantized_weights)
    }
    
    #[cfg_attr(feature = "tracing", instrument(skip(self, input, quantized_weights), fields(layer = %self.layer_name())))]
    fn forward_with_quantized_weights(
        &self,
        input: &Tensor,
        quantized_weights: &QuantizedWeight,
    ) -> BitLinearResult<Tensor> {
        #[cfg(feature = "tracing")]
        debug!("Forward pass with pre-quantized weights for layer: {}", self.layer_name());
        
        // Perform quantized matrix multiplication
        // Input: [..., in_features] × Weights: [out_features, in_features] -> [..., out_features]
        let output = self.quantized_matmul(input, quantized_weights)?;
        
        #[cfg(feature = "tracing")]
        debug!("Matrix multiplication completed - output shape: {:?}", output.shape());
        
        // Add bias if enabled
        let final_output = if let Some(ref bias) = self.bias() {
            let bias_guard = bias.read()
                .map_err(|_| BitLinearError::MemoryError("Failed to acquire bias read lock".to_string()))?;
            
            #[cfg(feature = "tracing")]
            debug!("Adding bias to output");
            
            output.broadcast_add(&bias_guard)
                .map_err(|e| BitLinearError::DeviceError(format!("Failed to add bias: {e}")))?
        } else {
            output
        };
        
        #[cfg(feature = "tracing")]
        debug!("Forward pass completed for layer: {}", self.layer_name());
        
        Ok(final_output)
    }
}

impl BitLinear {
    /// Perform quantized matrix multiplication
    /// 
    /// This method implements efficient matrix multiplication using quantized weights.
    /// It handles the ternary quantization values {-1, 0, +1} and applies the scaling factor.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input tensor with shape [..., in_features]
    /// * `quantized_weights` - Quantized weight structure
    /// 
    /// # Returns
    /// 
    /// Output tensor with shape [..., out_features]
    fn quantized_matmul(&self, input: &Tensor, quantized_weights: &QuantizedWeight) -> BitLinearResult<Tensor> {
        // Get the quantized values and scaling factor
        let q_weights = &quantized_weights.values;
        let scales = &quantized_weights.scales;
        
        #[cfg(feature = "tracing")]
        debug!("Performing quantized matmul");
        
        // Convert quantized weights to f32 tensor for computation
        // This is where we could optimize with custom kernels in the future
        let weight_f32 = self.convert_quantized_to_f32(q_weights, scales)?;
        
        // Standard matrix multiplication: input @ weight_f32.transpose()
        // Note: weight_f32 is [out_features, in_features], so we need transpose for correct shapes
        let weight_transposed = weight_f32.t()
            .map_err(|e| BitLinearError::DeviceError(format!("Failed to transpose weights: {e}")))?;
        
        // Handle different input shapes
        let input_shape = input.shape();
        let input_dims = input_shape.dims();
        
        let output = if input_dims.len() == 2 {
            // 2D input: [batch_size, in_features] @ [in_features, out_features] -> [batch_size, out_features]
            input.matmul(&weight_transposed)
                .map_err(|e| BitLinearError::DeviceError(format!("Matrix multiplication failed: {e}")))?
        } else if input_dims.len() == 3 {
            // 3D input: [batch_size, seq_len, in_features] -> reshape -> matmul -> reshape back
            let batch_size = input_dims[0];
            let seq_len = input_dims[1];
            let in_features = input_dims[2];
            
            // Reshape to 2D: [batch_size * seq_len, in_features]
            let input_2d = input.reshape(&[batch_size * seq_len, in_features])
                .map_err(|e| BitLinearError::DeviceError(format!("Failed to reshape input to 2D: {e}")))?;
            
            // Perform 2D matrix multiplication
            let output_2d = input_2d.matmul(&weight_transposed)
                .map_err(|e| BitLinearError::DeviceError(format!("Matrix multiplication failed: {e}")))?;
            
            // Reshape back to 3D: [batch_size, seq_len, out_features]
            let out_features = self.config().out_features;
            output_2d.reshape(&[batch_size, seq_len, out_features])
                .map_err(|e| BitLinearError::DeviceError(format!("Failed to reshape output back to 3D: {e}")))?
        } else {
            return Err(BitLinearError::ShapeMismatch {
                expected: vec![1, self.config().in_features],
                actual: input_shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "tracing")]
        debug!("Quantized matrix multiplication completed - output shape: {:?}", output.shape());
        
        Ok(output)
    }
    
    /// Convert quantized values back to f32 tensor with scaling
    /// 
    /// This method takes the ternary quantized values {-1, 0, +1} and converts
    /// them back to f32 values by applying the scaling factors.
    /// 
    /// # Arguments
    /// 
    /// * `quantized_values` - Ternary quantized values
    /// * `scales` - Scaling factors to apply
    /// 
    /// # Returns
    /// 
    /// F32 tensor with scaled values
    fn convert_quantized_to_f32(&self, quantized_values: &Tensor, scales: &Tensor) -> BitLinearResult<Tensor> {
        // Convert quantized tensor to f32 and apply scaling
        let q_weights_f32 = quantized_values.to_dtype(candle_core::DType::F32)
            .map_err(|e| BitLinearError::DeviceError(format!("Failed to convert quantized values to f32: {e}")))?;
        
        // Apply scaling factor element-wise or broadcast if per-tensor scaling
        let scaled_weights = q_weights_f32.broadcast_mul(scales)
            .map_err(|e| BitLinearError::DeviceError(format!("Failed to apply scaling: {e}")))?;
        
        Ok(scaled_weights)
    }
    
    /// This method efficiently processes multiple inputs in a single batch,
    /// which is useful for training scenarios.
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - Batch of input tensors with shape [batch_size, ..., in_features]
    /// 
    /// # Returns
    /// 
    /// Batch of output tensors with shape [batch_size, ..., out_features]
    pub fn forward_batch(&self, inputs: &Tensor) -> BitLinearResult<Tensor> {
        #[cfg(feature = "tracing")]
        debug!("Batch forward pass for layer: {}", self.layer_name());
        
        // Validate batch dimensions
        let input_shape = inputs.shape();
        let input_dims = input_shape.dims();
        
        if input_dims.len() < 2 {
            return Err(BitLinearError::ShapeMismatch {
                expected: vec![1, 1, self.config().in_features],
                actual: input_shape.dims().to_vec(),
            });
        }
        
        let last_dim = input_dims[input_dims.len() - 1];
        if last_dim != self.config().in_features {
            return Err(BitLinearError::ShapeMismatch {
                expected: vec![input_dims[0], input_dims[1], self.config().in_features],
                actual: input_shape.dims().to_vec(),
            });
        }
        
        // Use the standard forward pass - it handles batched inputs correctly
        self.forward(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitlinear::{BitLinearConfig};
    use candle_core::{Device, DType, Tensor};
    
    #[test]
    fn test_forward_basic() {
        let config = BitLinearConfig {
            in_features: 4,
            out_features: 3,
            use_bias: false,
            ..Default::default()
        };
        
        let layer = BitLinear::new(config, "test_layer".to_string()).unwrap();
        
        // Create test input
        let device = Device::Cpu;
        let input = Tensor::ones(&[2, 4], DType::F32, &device).unwrap();
        
        // Perform forward pass
        let output = layer.forward(&input).unwrap();
        
        // Verify output shape
        assert_eq!(output.shape().dims(), &[2, 3]);
    }
    
    #[test]
    fn test_forward_with_bias() {
        let config = BitLinearConfig {
            in_features: 4,
            out_features: 3,
            use_bias: true,
            ..Default::default()
        };
        
        let layer = BitLinear::new(config, "test_layer_bias".to_string()).unwrap();
        
        // Create test input
        let device = Device::Cpu;
        let input = Tensor::ones(&[2, 4], DType::F32, &device).unwrap();
        
        // Perform forward pass
        let output = layer.forward(&input).unwrap();
        
        // Verify output shape
        assert_eq!(output.shape().dims(), &[2, 3]);
    }
    
    #[test]
    fn test_forward_shape_mismatch() {
        let config = BitLinearConfig {
            in_features: 4,
            out_features: 3,
            ..Default::default()
        };
        
        let layer = BitLinear::new(config, "test_layer_shape".to_string()).unwrap();
        
        // Create test input with wrong shape
        let device = Device::Cpu;
        let input = Tensor::ones(&[2, 5], DType::F32, &device).unwrap(); // Wrong in_features
        
        // Perform forward pass - should fail
        let result = layer.forward(&input);
        assert!(result.is_err());
        
        if let Err(BitLinearError::ShapeMismatch { .. }) = result {
            // Expected error type
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }
}
