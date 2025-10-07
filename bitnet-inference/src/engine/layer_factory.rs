//! Layer factory for constructing BitNet layers from configuration.
//!
//! This module provides factory methods to construct BitLinear, RMSNorm, and other layers
//! from BitNet configuration and organized weights.

use crate::{Result, InferenceError};
use crate::bitnet_config::BitNetModelConfig;
use crate::engine::model_loader::{
    ModelArchitecture, LayerDefinition, LayerType, LayerParameters, ParameterType, ModelWeights
};
use std::collections::HashMap;

/// Factory for constructing BitNet model layers from configuration.
#[derive(Debug)]
pub struct LayerFactory {
    /// BitNet model configuration
    config: BitNetModelConfig,
    /// Organized model weights
    weights: ModelWeights,
}

impl LayerFactory {
    /// Create a new layer factory.
    pub fn new(config: BitNetModelConfig, weights: ModelWeights) -> Self {
        Self { config, weights }
    }

    /// Get the BitNet model configuration.
    pub fn config(&self) -> &BitNetModelConfig {
        &self.config
    }

    /// Build complete model architecture from configuration and weights.
    pub fn build_model_architecture(&self) -> Result<ModelArchitecture> {
        let mut layers = Vec::new();
        let mut execution_order = Vec::new();
        let mut layer_id = 0;

        // Create embedding layer
        if self.has_embedding_weights() {
            let embedding_layer = self.create_embedding_layer(layer_id)?;
            layers.push(embedding_layer);
            execution_order.push(layer_id);
            layer_id += 1;
        }

        // Create transformer layers
        for layer_idx in 0..self.config.layer_config.n_layers {
            // RMS Norm before attention
            let pre_attn_norm = self.create_rms_norm_layer(
                layer_id, 
                format!("blk.{}.attn_norm", layer_idx)
            )?;
            layers.push(pre_attn_norm);
            execution_order.push(layer_id);
            layer_id += 1;

            // Multi-head attention (as composite BitLinear layers)
            let attn_layers = self.create_attention_layers(layer_id, layer_idx)?;
            for layer in attn_layers {
                layers.push(layer);
                execution_order.push(layer_id);
                layer_id += 1;
            }

            // RMS Norm before FFN
            let pre_ffn_norm = self.create_rms_norm_layer(
                layer_id,
                format!("blk.{}.ffn_norm", layer_idx)
            )?;
            layers.push(pre_ffn_norm);
            execution_order.push(layer_id);
            layer_id += 1;

            // Feed-forward network layers
            let ffn_layers = self.create_ffn_layers(layer_id, layer_idx)?;
            for layer in ffn_layers {
                layers.push(layer);
                execution_order.push(layer_id);
                layer_id += 1;
            }
        }

        // Create final normalization
        if self.has_final_norm_weights() {
            let final_norm = self.create_rms_norm_layer(layer_id, "output_norm".to_string())?;
            layers.push(final_norm);
            execution_order.push(layer_id);
            layer_id += 1;
        }

        // Create output projection
        if self.has_output_weights() {
            let output_layer = self.create_output_layer(layer_id)?;
            layers.push(output_layer);
            execution_order.push(layer_id);
        }

        Ok(ModelArchitecture {
            layers,
            execution_order,
        })
    }

    /// Create embedding layer.
    fn create_embedding_layer(&self, layer_id: usize) -> Result<LayerDefinition> {
        self.validate_layer_weights(layer_id, &[ParameterType::EmbeddingWeight])?;

        Ok(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::Embedding,
            input_dims: vec![self.config.tokenizer_config.vocab_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::Embedding {
                vocab_size: self.config.tokenizer_config.vocab_size,
                embedding_dim: self.config.layer_config.hidden_size,
            },
        })
    }

    /// Create RMS normalization layer.
    fn create_rms_norm_layer(&self, layer_id: usize, norm_name: String) -> Result<LayerDefinition> {
        // Check if layer norm weights exist for this layer
        if !self.has_layer_norm_weights(&norm_name) {
            return Err(InferenceError::model_load(
                format!("Missing layer norm weights for {}", norm_name)
            ));
        }

        Ok(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::RMSNorm,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::RMSNorm {
                eps: self.config.normalization_config.rms_norm_eps,
            },
        })
    }

    /// Create attention layers (query, key, value, output).
    fn create_attention_layers(&self, mut layer_id: usize, layer_idx: usize) -> Result<Vec<LayerDefinition>> {
        let mut layers = Vec::new();

        // Attention query
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });
        layer_id += 1;

        // Attention key
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });
        layer_id += 1;

        // Attention value
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });
        layer_id += 1;

        // Attention output
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });

        Ok(layers)
    }

    /// Create feed-forward network layers (gate, up, down).
    fn create_ffn_layers(&self, mut layer_id: usize, layer_idx: usize) -> Result<Vec<LayerDefinition>> {
        let mut layers = Vec::new();

        // FFN gate
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.intermediate_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });
        layer_id += 1;

        // FFN up
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.layer_config.intermediate_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });
        layer_id += 1;

        // FFN down
        layers.push(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::BitLinear,
            input_dims: vec![self.config.layer_config.intermediate_size],
            output_dims: vec![self.config.layer_config.hidden_size],
            parameters: LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
        });

        Ok(layers)
    }

    /// Create output projection layer.
    fn create_output_layer(&self, layer_id: usize) -> Result<LayerDefinition> {
        self.validate_layer_weights(layer_id, &[ParameterType::OutputWeight])?;

        Ok(LayerDefinition {
            id: layer_id,
            layer_type: LayerType::OutputProjection,
            input_dims: vec![self.config.layer_config.hidden_size],
            output_dims: vec![self.config.tokenizer_config.vocab_size],
            parameters: LayerParameters::OutputProjection {
                vocab_size: self.config.tokenizer_config.vocab_size,
            },
        })
    }

    /// Validate that required weights exist for a layer.
    fn validate_layer_weights(&self, layer_id: usize, required_params: &[ParameterType]) -> Result<()> {
        for &param_type in required_params {
            if !self.weights.organized_weights.values()
                .any(|layer_weights| layer_weights.contains_key(&param_type)) {
                return Err(InferenceError::model_load(
                    format!("Missing required parameter {:?} for layer {}", param_type, layer_id)
                ));
            }
        }
        Ok(())
    }

    /// Check if embedding weights exist.
    fn has_embedding_weights(&self) -> bool {
        self.weights.layer_mapping.contains_key("token_embd.weight") ||
        self.weights.organized_weights.values()
            .any(|layer| layer.contains_key(&ParameterType::EmbeddingWeight))
    }

    /// Check if final normalization weights exist.
    fn has_final_norm_weights(&self) -> bool {
        self.weights.layer_mapping.contains_key("output_norm.weight") ||
        self.weights.layer_mapping.contains_key("norm.weight")
    }

    /// Check if output projection weights exist.
    fn has_output_weights(&self) -> bool {
        self.weights.layer_mapping.contains_key("output.weight") ||
        self.weights.layer_mapping.contains_key("lm_head.weight") ||
        self.weights.organized_weights.values()
            .any(|layer| layer.contains_key(&ParameterType::OutputWeight))
    }

    /// Check if layer normalization weights exist for a specific layer.
    fn has_layer_norm_weights(&self, norm_name: &str) -> bool {
        let weight_key = format!("{}.weight", norm_name);
        self.weights.layer_mapping.contains_key(&weight_key)
    }
}

/// Builder for configuring layer factory construction.
pub struct LayerFactoryBuilder {
    config: Option<BitNetModelConfig>,
    weights: Option<ModelWeights>,
}

impl LayerFactoryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            weights: None,
        }
    }

    /// Set the BitNet configuration.
    pub fn with_config(mut self, config: BitNetModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the model weights.
    pub fn with_weights(mut self, weights: ModelWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Build the layer factory.
    pub fn build(self) -> Result<LayerFactory> {
        let config = self.config.ok_or_else(|| {
            InferenceError::model_load("BitNet configuration is required")
        })?;

        let weights = self.weights.ok_or_else(|| {
            InferenceError::model_load("Model weights are required")
        })?;

        Ok(LayerFactory::new(config, weights))
    }
}

impl Default for LayerFactoryBuilder {
    fn default() -> Self {
        Self::new()
    }
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
                n_layers: 2,
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

        // Add some weight data
        weights.layer_weights.insert(0, vec![0u8; 1000]);
        weights.layer_weights.insert(1, vec![0u8; 512]);
        weights.layer_weights.insert(2, vec![0u8; 512]);

        weights
    }

    #[test]
    fn test_layer_factory_creation() {
        let config = create_test_config();
        let weights = create_test_weights();
        
        let factory = LayerFactory::new(config, weights);
        assert_eq!(factory.config.layer_config.n_layers, 2);
    }

    #[test]
    fn test_layer_factory_builder() {
        let config = create_test_config();
        let weights = create_test_weights();
        
        let factory = LayerFactoryBuilder::new()
            .with_config(config)
            .with_weights(weights)
            .build()
            .expect("Failed to build layer factory");
        
        assert_eq!(factory.config.layer_config.n_layers, 2);
    }

    #[test]
    fn test_has_embedding_weights() {
        let config = create_test_config();
        let weights = create_test_weights();
        
        let factory = LayerFactory::new(config, weights);
        assert!(factory.has_embedding_weights());
    }
}