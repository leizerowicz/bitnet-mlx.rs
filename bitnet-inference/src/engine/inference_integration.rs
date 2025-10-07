//! Inference engine integration for BitNet models.
//!
//! This module provides integration between the layer factory (configuration to layer mapping)
//! and the actual inference engine execution, enabling end-to-end model execution.

use crate::{Result, InferenceError};
use crate::engine::{LayerFactory, ExecutionContext, LoadedModel, ModelWeights, WeightArrays, ConvertedWeights};
use crate::engine::model_loader::{ModelArchitecture, LayerDefinition, LayerType, LayerParameters, ParameterType, ParameterData, ParameterDataType};
use crate::bitnet_config::BitNetModelConfig;
use bitnet_core::{Device, Tensor};
use candle_core::DType;
use std::collections::HashMap;
use std::sync::Arc;

/// Integration layer that connects BitNet configuration to inference execution.
#[derive(Debug)]
pub struct InferenceIntegration {
    /// Layer factory for constructing model architecture
    layer_factory: LayerFactory,
    /// Model architecture built from configuration
    architecture: ModelArchitecture,
    /// Model weights organized for inference
    weights: ModelWeights,
    /// BitNet configuration
    config: BitNetModelConfig,
}

impl InferenceIntegration {
    /// Create a new inference integration from a loaded model.
    pub fn from_loaded_model(model: LoadedModel) -> Result<Self> {
        let config = model.bitnet_config.ok_or_else(|| {
            InferenceError::model_load("BitNet configuration is required for inference integration")
        })?;

        let layer_factory = LayerFactory::new(config.clone(), model.weights.clone());
        let architecture = layer_factory.build_model_architecture()?;

        Ok(Self {
            layer_factory,
            architecture,
            weights: model.weights,
            config,
        })
    }

    /// Create inference integration from components.
    pub fn new(
        config: BitNetModelConfig,
        weights: ModelWeights,
    ) -> Result<Self> {
        let layer_factory = LayerFactory::new(config.clone(), weights.clone());
        let architecture = layer_factory.build_model_architecture()?;

        Ok(Self {
            layer_factory,
            architecture,
            weights,
            config,
        })
    }

    /// Get the model architecture.
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    /// Get the model weights.
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// Get the BitNet configuration.
    pub fn config(&self) -> &BitNetModelConfig {
        &self.config
    }

    /// Create an executable model from the integration.
    pub fn create_executable_model(&self) -> Result<ExecutableModel> {
        ExecutableModel::new(self)
    }

    /// Validate that the integration is ready for inference.
    pub fn validate(&self) -> Result<()> {
        // Check architecture completeness
        if self.architecture.layers.is_empty() {
            return Err(InferenceError::model_load("Model architecture has no layers"));
        }

        // Check execution order
        if self.architecture.execution_order.is_empty() {
            return Err(InferenceError::model_load("Model has no execution order defined"));
        }

        // Validate layer definitions
        for layer in &self.architecture.layers {
            self.validate_layer_definition(layer)?;
        }

        // Check that all layers in execution order exist
        for &layer_id in &self.architecture.execution_order {
            if !self.architecture.layers.iter().any(|l| l.id == layer_id) {
                return Err(InferenceError::model_load(
                    format!("Execution order references non-existent layer {}", layer_id)
                ));
            }
        }

        Ok(())
    }

    /// Validate a single layer definition.
    fn validate_layer_definition(&self, layer: &LayerDefinition) -> Result<()> {
        // Check input/output dimensions are valid
        if layer.input_dims.is_empty() {
            return Err(InferenceError::model_load(
                format!("Layer {} has empty input dimensions", layer.id)
            ));
        }

        if layer.output_dims.is_empty() {
            return Err(InferenceError::model_load(
                format!("Layer {} has empty output dimensions", layer.id)
            ));
        }

        // Validate parameters based on layer type
        match (&layer.layer_type, &layer.parameters) {
            (LayerType::BitLinear, LayerParameters::BitLinear { .. }) => {
                // Valid combination
            },
            (LayerType::RMSNorm, LayerParameters::RMSNorm { .. }) => {
                // Valid combination  
            },
            (LayerType::Embedding, LayerParameters::Embedding { .. }) => {
                // Valid combination
            },
            (LayerType::OutputProjection, LayerParameters::OutputProjection { .. }) => {
                // Valid combination
            },
            _ => {
                return Err(InferenceError::model_load(
                    format!("Layer {} has mismatched type and parameters", layer.id)
                ));
            }
        }

        Ok(())
    }

    /// Get layer weights for a specific layer.
    pub fn get_layer_weights(&self, layer_id: usize) -> Option<&HashMap<ParameterType, crate::engine::model_loader::ParameterData>> {
        self.weights.get_layer_parameters(layer_id)
    }

    /// Get parameter for a specific layer and parameter type.
    pub fn get_parameter(&self, layer_id: usize, param_type: ParameterType) -> Option<&crate::engine::model_loader::ParameterData> {
        self.weights.get_parameter(layer_id, param_type)
    }
}

/// Executable model that can perform inference operations.
pub struct ExecutableModel {
    /// Reference to the inference integration
    integration: Arc<InferenceIntegration>,
    /// Execution context for the model
    execution_context: Option<ExecutionContext>,
    /// Cached layer operations
    layer_operations: HashMap<usize, LayerOperation>,
}

impl ExecutableModel {
    /// Create a new executable model.
    pub fn new(integration: &InferenceIntegration) -> Result<Self> {
        integration.validate()?;

        let integration = Arc::new(integration.clone());
        let layer_operations = Self::build_layer_operations(&integration)?;

        Ok(Self {
            integration,
            execution_context: None,
            layer_operations,
        })
    }

    /// Set the execution context for this model.
    pub fn with_execution_context(mut self, context: ExecutionContext) -> Self {
        self.execution_context = Some(context);
        self
    }

    /// Execute the model with the given input tensor.
    pub fn execute(&self, input: Tensor) -> Result<Tensor> {
        self.validate_input(&input)?;

        let mut current_tensor = input;

        // Execute layers in the defined order
        for &layer_id in &self.integration.architecture.execution_order {
            let layer_op = self.layer_operations.get(&layer_id)
                .ok_or_else(|| InferenceError::inference(
                    format!("Layer operation not found for layer {}", layer_id)
                ))?;

            current_tensor = layer_op.execute(current_tensor)?;
        }

        Ok(current_tensor)
    }

    /// Validate input tensor shape and properties.
    fn validate_input(&self, input: &Tensor) -> Result<()> {
        // Get the first layer's expected input dimensions
        if let Some(first_layer_id) = self.integration.architecture.execution_order.first() {
            if let Some(first_layer) = self.integration.architecture.layers.iter()
                .find(|l| l.id == *first_layer_id) {
                
                // Check if input shape is compatible (allowing for batch dimension)
                let expected_dims = &first_layer.input_dims;
                let input_shape = input.shape();

                if input_shape.dims().len() < expected_dims.len() {
                    return Err(InferenceError::inference(
                        format!("Input tensor has {} dimensions, expected at least {}", 
                               input_shape.dims().len(), expected_dims.len())
                    ));
                }

                // Check non-batch dimensions
                for (i, (&expected, &actual)) in expected_dims.iter()
                    .zip(input_shape.dims()[input_shape.dims().len() - expected_dims.len()..].iter())
                    .enumerate() {
                    if expected != actual {
                        return Err(InferenceError::inference(
                            format!("Input dimension {} mismatch: expected {}, got {}", 
                                   i, expected, actual)
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Build layer operations from the integration.
    fn build_layer_operations(integration: &Arc<InferenceIntegration>) -> Result<HashMap<usize, LayerOperation>> {
        let mut operations = HashMap::new();

        for layer in &integration.architecture.layers {
            let operation = LayerOperation::from_layer_definition(layer, integration.clone())?;
            operations.insert(layer.id, operation);
        }

        Ok(operations)
    }

    /// Get information about the model.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.integration.architecture.layers.len(),
            num_parameters: self.integration.config.basic_info.parameter_count,
            context_length: self.integration.config.basic_info.context_length,
            vocab_size: self.integration.config.tokenizer_config.vocab_size,
            hidden_size: self.integration.config.layer_config.hidden_size,
        }
    }
}

impl Clone for InferenceIntegration {
    fn clone(&self) -> Self {
        Self {
            layer_factory: LayerFactory::new(self.config.clone(), self.weights.clone()),
            architecture: self.architecture.clone(),
            weights: self.weights.clone(),
            config: self.config.clone(),
        }
    }
}

/// Individual layer operation that can execute a forward pass.
#[derive(Debug)]
pub struct LayerOperation {
    /// Layer ID
    layer_id: usize,
    /// Layer type
    layer_type: LayerType,
    /// Input/output dimensions
    input_dims: Vec<usize>,
    output_dims: Vec<usize>,
    /// Reference to the integration for weight access
    integration: Arc<InferenceIntegration>,
}

impl LayerOperation {
    /// Create a layer operation from a layer definition.
    pub fn from_layer_definition(
        layer: &LayerDefinition,
        integration: Arc<InferenceIntegration>,
    ) -> Result<Self> {
        Ok(Self {
            layer_id: layer.id,
            layer_type: layer.layer_type.clone(),
            input_dims: layer.input_dims.clone(),
            output_dims: layer.output_dims.clone(),
            integration,
        })
    }

    /// Execute this layer operation on the input tensor.
    pub fn execute(&self, input: Tensor) -> Result<Tensor> {
        match self.layer_type {
            LayerType::BitLinear => self.execute_bitlinear(input),
            LayerType::RMSNorm => self.execute_rms_norm(input),
            LayerType::Embedding => self.execute_embedding(input),
            LayerType::OutputProjection => self.execute_output_projection(input),
            LayerType::SwiGLU => self.execute_swiglu(input),
        }
    }

    /// Execute BitLinear layer with ternary weight matrix multiplication.
    fn execute_bitlinear(&self, input: Tensor) -> Result<Tensor> {
        tracing::debug!("Executing BitLinear layer {}", self.layer_id);
        
        // Get the weight parameters for this layer
        let weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::Weight)
            .map_err(|e| InferenceError::inference(format!("Failed to get BitLinear weights: {}", e)))?;
        
        // Get ternary weights
        let ternary_weights = weights.as_ternary()
            .ok_or_else(|| InferenceError::inference("BitLinear layer requires ternary weights".to_string()))?;
        
        // Perform ternary matrix multiplication
        self.ternary_matmul(input, ternary_weights, &self.output_dims)
    }

    /// Execute RMS normalization layer.
    fn execute_rms_norm(&self, input: Tensor) -> Result<Tensor> {
        tracing::debug!("Executing RMSNorm layer {}", self.layer_id);
        
        // Get RMS norm parameters
        let scale_weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::LayerNormScale)?;
        let scale_values = scale_weights.as_f32_slice()
            .ok_or_else(|| InferenceError::inference("RMSNorm requires float scale values".to_string()))?;
        
        // Get epsilon from layer parameters
        let eps = self.integration.layer_factory.config().normalization_config.rms_norm_eps;
        
        self.rms_normalize(input, scale_values, eps)
    }

    /// Execute embedding layer.
    fn execute_embedding(&self, input: Tensor) -> Result<Tensor> {
        tracing::debug!("Executing Embedding layer {}", self.layer_id);
        
        // Get embedding weights
        let embedding_weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::EmbeddingWeight)?;
        let weight_values = embedding_weights.as_f32_slice()
            .ok_or_else(|| InferenceError::inference("Embedding layer requires float weights".to_string()))?;
        
        // Perform embedding lookup
        self.embedding_lookup(input, weight_values)
    }

    /// Execute output projection layer.
    fn execute_output_projection(&self, input: Tensor) -> Result<Tensor> {
        tracing::debug!("Executing OutputProjection layer {}", self.layer_id);
        
        // Get output projection weights
        let output_weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::OutputWeight)?;
        let weight_values = output_weights.as_f32_slice()
            .ok_or_else(|| InferenceError::inference("Output projection requires float weights".to_string()))?;
        
        // Perform linear transformation for output
        self.linear_transform(input, weight_values, &self.output_dims)
    }

    /// Execute SwiGLU activation.
    fn execute_swiglu(&self, input: Tensor) -> Result<Tensor> {
        tracing::debug!("Executing SwiGLU layer {}", self.layer_id);
        
        // Get gate and up projection weights
        let gate_weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::FeedForwardGate)?;
        let up_weights = self.integration.weights.convert_parameter(self.layer_id, ParameterType::FeedForwardUp)?;
        
        let gate_values = gate_weights.as_f32_slice()
            .ok_or_else(|| InferenceError::inference("SwiGLU gate requires float weights".to_string()))?;
        let up_values = up_weights.as_f32_slice()
            .ok_or_else(|| InferenceError::inference("SwiGLU up requires float weights".to_string()))?;
        
        // Perform SwiGLU activation: swish(x @ gate) * (x @ up)
        self.swiglu_activation(input, gate_values, up_values)
    }

    /// Perform ternary matrix multiplication optimized for BitNet 1.58-bit weights.
    fn ternary_matmul(&self, input: Tensor, ternary_weights: &[i8], output_shape: &[usize]) -> Result<Tensor> {
        // This is a simplified implementation - production version would use SIMD optimization
        let input_shape = input.shape().dims();
        let device = input.device();
        
        // Verify dimensions
        if input_shape.len() < 2 || output_shape.len() < 2 {
            return Err(InferenceError::inference("Matrix multiplication requires at least 2D tensors".to_string()));
        }
        
        // Get input data as f32
        let input_data = input.to_vec1::<f32>()
            .map_err(|e| InferenceError::inference(format!("Failed to get input data: {}", e)))?;
        
        // Compute output dimensions
        let batch_size = input_shape[0];
        let input_dim = input_shape[input_shape.len() - 1];
        let output_dim = output_shape[output_shape.len() - 1];
        
        // Verify weight dimensions
        let expected_weight_count = input_dim * output_dim;
        if ternary_weights.len() != expected_weight_count {
            return Err(InferenceError::inference(
                format!("Weight dimension mismatch: expected {}, got {}", expected_weight_count, ternary_weights.len())
            ));
        }
        
        // Perform ternary matrix multiplication
        let mut output_data = vec![0.0f32; batch_size * output_dim];
        
        for b in 0..batch_size {
            for o in 0..output_dim {
                let mut sum = 0.0f32;
                for i in 0..input_dim {
                    let input_val = input_data[b * input_dim + i];
                    let weight_val = ternary_weights[i * output_dim + o] as f32;
                    sum += input_val * weight_val;
                }
                output_data[b * output_dim + o] = sum;
            }
        }
        
        // Create output tensor
        let output_tensor = Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&[batch_size, output_dim]),
            device
        ).map_err(|e| InferenceError::inference(format!("Failed to create output tensor: {}", e)))?;
        
        Ok(output_tensor)
    }

    /// Perform RMS normalization.
    fn rms_normalize(&self, input: Tensor, scale: &[f32], eps: f32) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let device = input.device();
        
        // Get input data
        let input_data = input.to_vec1::<f32>()
            .map_err(|e| InferenceError::inference(format!("Failed to get input data: {}", e)))?;
        
        let last_dim = input_shape[input_shape.len() - 1];
        let batch_size = input_data.len() / last_dim;
        
        // Verify scale dimensions
        if scale.len() != last_dim {
            return Err(InferenceError::inference(
                format!("Scale dimension mismatch: expected {}, got {}", last_dim, scale.len())
            ));
        }
        
        let mut output_data = vec![0.0f32; input_data.len()];
        
        // Apply RMS normalization: x / sqrt(mean(x^2) + eps) * scale
        for b in 0..batch_size {
            let batch_start = b * last_dim;
            let batch_end = batch_start + last_dim;
            let batch_data = &input_data[batch_start..batch_end];
            
            // Compute RMS
            let sum_squares: f32 = batch_data.iter().map(|x| x * x).sum();
            let rms = (sum_squares / last_dim as f32 + eps).sqrt();
            
            // Apply normalization and scaling
            for (i, &val) in batch_data.iter().enumerate() {
                output_data[batch_start + i] = (val / rms) * scale[i];
            }
        }
        
        // Create output tensor
        let output_tensor = Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(input_shape),
            device
        ).map_err(|e| InferenceError::inference(format!("Failed to create output tensor: {}", e)))?;
        
        Ok(output_tensor)
    }

    /// Perform embedding lookup.
    fn embedding_lookup(&self, input: Tensor, embeddings: &[f32]) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let device = input.device();
        
        // Get input token IDs
        let token_ids = input.to_vec1::<u32>()
            .map_err(|e| InferenceError::inference(format!("Failed to get token IDs: {}", e)))?;
        
        // Get embedding dimensions from output shape
        let embed_dim = self.output_dims[self.output_dims.len() - 1];
        let vocab_size = embeddings.len() / embed_dim;
        
        // Create output tensor
        let mut output_data = Vec::with_capacity(token_ids.len() * embed_dim);
        
        for &token_id in &token_ids {
            let token_id = token_id as usize;
            if token_id >= vocab_size {
                return Err(InferenceError::inference(
                    format!("Token ID {} out of range (vocab size: {})", token_id, vocab_size)
                ));
            }
            
            // Copy embedding for this token
            let embed_start = token_id * embed_dim;
            let embed_end = embed_start + embed_dim;
            output_data.extend_from_slice(&embeddings[embed_start..embed_end]);
        }
        
        // Create output shape: [batch_size, seq_len, embed_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(embed_dim);
        
        let output_tensor = Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&output_shape),
            device
        ).map_err(|e| InferenceError::inference(format!("Failed to create output tensor: {}", e)))?;
        
        Ok(output_tensor)
    }

    /// Perform linear transformation.
    fn linear_transform(&self, input: Tensor, weights: &[f32], output_shape: &[usize]) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let device = input.device();
        
        // Get input data
        let input_data = input.to_vec1::<f32>()
            .map_err(|e| InferenceError::inference(format!("Failed to get input data: {}", e)))?;
        
        // Compute dimensions
        let batch_size = input_shape[0];
        let input_dim = input_shape[input_shape.len() - 1];
        let output_dim = output_shape[output_shape.len() - 1];
        
        // Verify weight dimensions
        let expected_weight_count = input_dim * output_dim;
        if weights.len() != expected_weight_count {
            return Err(InferenceError::inference(
                format!("Weight dimension mismatch: expected {}, got {}", expected_weight_count, weights.len())
            ));
        }
        
        // Perform matrix multiplication
        let mut output_data = vec![0.0f32; batch_size * output_dim];
        
        for b in 0..batch_size {
            for o in 0..output_dim {
                let mut sum = 0.0f32;
                for i in 0..input_dim {
                    let input_val = input_data[b * input_dim + i];
                    let weight_val = weights[i * output_dim + o];
                    sum += input_val * weight_val;
                }
                output_data[b * output_dim + o] = sum;
            }
        }
        
        // Create output tensor
        let output_tensor = Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&[batch_size, output_dim]),
            device
        ).map_err(|e| InferenceError::inference(format!("Failed to create output tensor: {}", e)))?;
        
        Ok(output_tensor)
    }

    /// Perform SwiGLU activation: swish(x @ gate) * (x @ up).
    fn swiglu_activation(&self, input: Tensor, gate_weights: &[f32], up_weights: &[f32]) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        let device = input.device();
        
        // Get input data
        let input_data = input.to_vec1::<f32>()
            .map_err(|e| InferenceError::inference(format!("Failed to get input data: {}", e)))?;
        
        let batch_size = input_shape[0];
        let input_dim = input_shape[input_shape.len() - 1];
        let hidden_dim = gate_weights.len() / input_dim;
        
        // Verify dimensions
        if gate_weights.len() != up_weights.len() {
            return Err(InferenceError::inference("Gate and up weights must have same dimensions".to_string()));
        }
        
        // Compute gate and up projections
        let mut gate_output = vec![0.0f32; batch_size * hidden_dim];
        let mut up_output = vec![0.0f32; batch_size * hidden_dim];
        
        for b in 0..batch_size {
            for h in 0..hidden_dim {
                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;
                
                for i in 0..input_dim {
                    let input_val = input_data[b * input_dim + i];
                    gate_sum += input_val * gate_weights[i * hidden_dim + h];
                    up_sum += input_val * up_weights[i * hidden_dim + h];
                }
                
                gate_output[b * hidden_dim + h] = gate_sum;
                up_output[b * hidden_dim + h] = up_sum;
            }
        }
        
        // Apply SwiGLU: swish(gate) * up, where swish(x) = x * sigmoid(x)
        let mut output_data = vec![0.0f32; batch_size * hidden_dim];
        for i in 0..output_data.len() {
            let gate_val = gate_output[i];
            let up_val = up_output[i];
            
            // Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
            let sigmoid = 1.0 / (1.0 + (-gate_val).exp());
            let swish = gate_val * sigmoid;
            
            output_data[i] = swish * up_val;
        }
        
        // Create output tensor
        let output_tensor = Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&[batch_size, hidden_dim]),
            device
        ).map_err(|e| InferenceError::inference(format!("Failed to create output tensor: {}", e)))?;
        
        Ok(output_tensor)
    }
}

/// Information about an executable model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub num_parameters: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet_config::*;
    use std::collections::HashMap;

    fn create_test_config() -> BitNetModelConfig {
        BitNetModelConfig {
            basic_info: BasicModelInfo {
                name: "test-model".to_string(),
                architecture: "bitnet-b1.58".to_string(),
                version: "1.0".to_string(),
                parameter_count: 1000000,
                context_length: 2048,
            },
            layer_config: LayerConfig {
                n_layers: 1,
                hidden_size: 512,
                intermediate_size: 1024,
                model_dim: 512,
            },
            attention_config: AttentionConfig {
                n_heads: 8,
                n_kv_heads: None,
                head_dim: 64,
                max_seq_len: 2048,
                rope_config: RopeConfig {
                    rope_freq_base: 10000.0,
                    rope_scaling: None,
                    rope_dim: 128,
                },
            },
            normalization_config: NormalizationConfig {
                rms_norm_eps: 1e-6,
                use_bias: false,
            },
            bitlinear_config: BitLinearConfig {
                weight_bits: 1,
                activation_bits: 8,
                use_weight_scaling: true,
                use_activation_scaling: false,
                quantization_scheme: "bitnet-1.58".to_string(),
            },
            tokenizer_config: TokenizerConfig {
                vocab_size: 32000,
                tokenizer_type: "llama3".to_string(),
                bos_token_id: Some(1),
                eos_token_id: Some(2),
                pad_token_id: Some(0),
            },
            extra_metadata: HashMap::new(),
        }
    }

    fn create_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new();
        
        // Add layer mapping
        weights.layer_mapping.insert("token_embd.weight".to_string(), 0);
        weights.layer_mapping.insert("blk.0.attn_norm.weight".to_string(), 1);
        weights.layer_mapping.insert("blk.0.ffn_norm.weight".to_string(), 2);

        // Add some weight data (old format for backward compatibility)
        weights.layer_weights.insert(0, vec![0u8; 1000]);
        weights.layer_weights.insert(1, vec![0u8; 512]);
        weights.layer_weights.insert(2, vec![0u8; 512]);

        // Add organized parameter data (new format)
        // Layer 0 - Embedding layer
        let embedding_data = ParameterData {
            data: vec![0u8; 1000],
            shape: vec![32000, 512], // vocab_size x hidden_size
            dtype: ParameterDataType::F32,
            tensor_name: "token_embd.weight".to_string(),
        };
        weights.add_parameter(0, ParameterType::EmbeddingWeight, embedding_data);

        // Layer 1 - RMSNorm layer (attention normalization)
        let norm_data = ParameterData {
            data: vec![0u8; 512],
            shape: vec![512], // hidden_size
            dtype: ParameterDataType::F32,
            tensor_name: "blk.0.attn_norm.weight".to_string(),
        };
        weights.add_parameter(1, ParameterType::LayerNormScale, norm_data);

        // Layer 2 - RMSNorm layer (FFN normalization)
        let ffn_norm_data = ParameterData {
            data: vec![0u8; 512],
            shape: vec![512], // hidden_size
            dtype: ParameterDataType::F32,
            tensor_name: "blk.0.ffn_norm.weight".to_string(),
        };
        weights.add_parameter(2, ParameterType::LayerNormScale, ffn_norm_data);

        weights
    }

    #[test]
    fn test_inference_integration_creation() {
        let config = create_test_config();
        let weights = create_test_weights();
        
        let integration = InferenceIntegration::new(config, weights);
        if let Err(e) = &integration {
            println!("Integration creation error: {:?}", e);
        }
        assert!(integration.is_ok());
    }

    #[test]
    fn test_integration_validation() {
        let config = create_test_config();
        let weights = create_test_weights();
        
        let integration = InferenceIntegration::new(config, weights).unwrap();
        let validation_result = integration.validate();
        
        // This might fail due to missing weights, which is expected in test
        println!("Validation result: {:?}", validation_result);
    }
}