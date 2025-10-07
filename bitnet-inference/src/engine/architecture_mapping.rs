//! Architecture mapping module for converting GGUF metadata to BitNet ModelArchitecture.
//!
//! This module provides comprehensive mapping from GGUF tensor metadata to a complete
//! BitNet model architecture with automatic layer type detection, parameter extraction,
//! and execution graph building.

use crate::{Result, InferenceError};
use crate::bitnet_config::BitNetModelConfig;
use crate::engine::model_loader::{
    ModelArchitecture, LayerDefinition, LayerType, LayerParameters
};
use crate::gguf::{GgufHeader, GgufTensorInfo};
use std::collections::HashMap;

/// Architecture mapper for converting GGUF structures to BitNet architectures.
#[derive(Debug)]
pub struct ArchitectureMapper {
    /// BitNet model configuration
    config: BitNetModelConfig,
    /// GGUF tensor information
    tensor_info: Vec<GgufTensorInfo>,
    /// Detected layer mapping
    layer_map: HashMap<String, LayerDefinition>,
}

/// Layer detection patterns for different types
#[derive(Debug, Clone)]
pub struct LayerPattern {
    /// Pattern name
    pub name: String,
    /// Tensor name patterns to match
    pub patterns: Vec<String>,
    /// Layer type to create
    pub layer_type: LayerType,
    /// Required parameters
    pub required_params: Vec<String>,
}

/// Execution graph builder for model layers
#[derive(Debug)]
pub struct ExecutionGraphBuilder {
    layers: Vec<LayerDefinition>,
    execution_order: Vec<usize>,
    layer_dependencies: HashMap<usize, Vec<usize>>,
}

impl ArchitectureMapper {
    /// Create a new architecture mapper
    pub fn new(config: BitNetModelConfig, tensor_info: Vec<GgufTensorInfo>) -> Self {
        Self {
            config,
            tensor_info,
            layer_map: HashMap::new(),
        }
    }

    /// Map GGUF metadata to complete BitNet ModelArchitecture
    pub fn map_to_architecture(&mut self) -> Result<ModelArchitecture> {
        // Step 1: Detect layer types from tensor names
        self.detect_layer_types()?;
        
        // Step 2: Extract layer parameters
        self.extract_layer_parameters()?;
        
        // Step 3: Build execution graph
        let execution_graph = self.build_execution_graph()?;
        
        // Step 4: Validate architecture
        self.validate_architecture(&execution_graph)?;
        
        Ok(execution_graph)
    }

    /// Detect layer types from tensor names using pattern matching
    fn detect_layer_types(&mut self) -> Result<()> {
        let patterns = self.get_layer_patterns();
        let mut layer_id = 0;

        for pattern in &patterns {
            let matching_tensors = self.find_matching_tensors(&pattern.patterns);
            
            if !matching_tensors.is_empty() {
                let layer_definition = self.create_layer_from_pattern(
                    layer_id,
                    pattern,
                    &matching_tensors,
                )?;
                
                self.layer_map.insert(pattern.name.clone(), layer_definition);
                layer_id += 1;
            }
        }

        tracing::info!("Detected {} layers from {} tensor patterns", 
                      self.layer_map.len(), patterns.len());
        
        Ok(())
    }

    /// Get predefined layer patterns for BitNet models
    fn get_layer_patterns(&self) -> Vec<LayerPattern> {
        vec![
            // Embedding layer
            LayerPattern {
                name: "token_embedding".to_string(),
                patterns: vec![
                    "token_embd.weight".to_string(),
                    "embed_tokens.weight".to_string(),
                    "tok_embeddings.weight".to_string(),
                ],
                layer_type: LayerType::Embedding,
                required_params: vec!["vocab_size".to_string(), "embed_dim".to_string()],
            },
            
            // BitLinear attention layers (query, key, value, output)
            LayerPattern {
                name: "attention_query".to_string(),
                patterns: vec![
                    "blk.*.attn_q.weight".to_string(),
                    "layers.*.attention.wq.weight".to_string(),
                    "model.layers.*.self_attn.q_proj.weight".to_string(),
                ],
                layer_type: LayerType::BitLinear,
                required_params: vec!["input_dim".to_string(), "output_dim".to_string()],
            },
            
            LayerPattern {
                name: "attention_key".to_string(),
                patterns: vec![
                    "blk.*.attn_k.weight".to_string(),
                    "layers.*.attention.wk.weight".to_string(),
                    "model.layers.*.self_attn.k_proj.weight".to_string(),
                ],
                layer_type: LayerType::BitLinear,
                required_params: vec!["input_dim".to_string(), "output_dim".to_string()],
            },

            LayerPattern {
                name: "attention_value".to_string(),
                patterns: vec![
                    "blk.*.attn_v.weight".to_string(),
                    "layers.*.attention.wv.weight".to_string(),
                    "model.layers.*.self_attn.v_proj.weight".to_string(),
                ],
                layer_type: LayerType::BitLinear,
                required_params: vec!["input_dim".to_string(), "output_dim".to_string()],
            },

            LayerPattern {
                name: "attention_output".to_string(),
                patterns: vec![
                    "blk.*.attn_output.weight".to_string(),
                    "layers.*.attention.wo.weight".to_string(),
                    "model.layers.*.self_attn.o_proj.weight".to_string(),
                ],
                layer_type: LayerType::BitLinear,
                required_params: vec!["input_dim".to_string(), "output_dim".to_string()],
            },

            // RMSNorm layers
            LayerPattern {
                name: "attention_norm".to_string(),
                patterns: vec![
                    "blk.*.attn_norm.weight".to_string(),
                    "layers.*.attention_norm.weight".to_string(),
                    "model.layers.*.input_layernorm.weight".to_string(),
                ],
                layer_type: LayerType::RMSNorm,
                required_params: vec!["hidden_size".to_string(), "eps".to_string()],
            },

            LayerPattern {
                name: "ffn_norm".to_string(),
                patterns: vec![
                    "blk.*.ffn_norm.weight".to_string(),
                    "layers.*.ffn_norm.weight".to_string(),
                    "model.layers.*.post_attention_layernorm.weight".to_string(),
                ],
                layer_type: LayerType::RMSNorm,
                required_params: vec!["hidden_size".to_string(), "eps".to_string()],
            },

            // SwiGLU FFN layers
            LayerPattern {
                name: "ffn_gate".to_string(),
                patterns: vec![
                    "blk.*.ffn_gate.weight".to_string(),
                    "layers.*.feed_forward.w1.weight".to_string(),
                    "model.layers.*.mlp.gate_proj.weight".to_string(),
                ],
                layer_type: LayerType::SwiGLU,
                required_params: vec!["input_dim".to_string(), "hidden_dim".to_string()],
            },

            LayerPattern {
                name: "ffn_up".to_string(),
                patterns: vec![
                    "blk.*.ffn_up.weight".to_string(),
                    "layers.*.feed_forward.w3.weight".to_string(),
                    "model.layers.*.mlp.up_proj.weight".to_string(),
                ],
                layer_type: LayerType::SwiGLU,
                required_params: vec!["input_dim".to_string(), "hidden_dim".to_string()],
            },

            LayerPattern {
                name: "ffn_down".to_string(),
                patterns: vec![
                    "blk.*.ffn_down.weight".to_string(),
                    "layers.*.feed_forward.w2.weight".to_string(),
                    "model.layers.*.mlp.down_proj.weight".to_string(),
                ],
                layer_type: LayerType::BitLinear,
                required_params: vec!["input_dim".to_string(), "output_dim".to_string()],
            },

            // Output layers
            LayerPattern {
                name: "output_norm".to_string(),
                patterns: vec![
                    "output_norm.weight".to_string(),
                    "norm.weight".to_string(),
                    "model.norm.weight".to_string(),
                ],
                layer_type: LayerType::RMSNorm,
                required_params: vec!["hidden_size".to_string(), "eps".to_string()],
            },

            LayerPattern {
                name: "output_projection".to_string(),
                patterns: vec![
                    "output.weight".to_string(),
                    "lm_head.weight".to_string(),
                    "model.embed_out.weight".to_string(),
                ],
                layer_type: LayerType::OutputProjection,
                required_params: vec!["input_dim".to_string(), "vocab_size".to_string()],
            },
        ]
    }

    /// Find tensors matching the given patterns
    fn find_matching_tensors(&self, patterns: &[String]) -> Vec<&GgufTensorInfo> {
        let mut matching = Vec::new();
        
        for tensor in &self.tensor_info {
            for pattern in patterns {
                if self.matches_pattern(&tensor.name, pattern) {
                    matching.push(tensor);
                    break; // Only match first pattern per tensor
                }
            }
        }
        
        matching
    }

    /// Check if tensor name matches pattern (supports wildcards)
    fn matches_pattern(&self, tensor_name: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                tensor_name.starts_with(prefix) && tensor_name.ends_with(suffix)
            } else {
                // Multiple wildcards - more complex matching
                self.complex_wildcard_match(tensor_name, pattern)
            }
        } else {
            tensor_name == pattern
        }
    }

    /// Handle complex wildcard patterns
    fn complex_wildcard_match(&self, text: &str, pattern: &str) -> bool {
        let pattern_parts: Vec<&str> = pattern.split('*').collect();
        let mut text_pos = 0;
        
        for (i, part) in pattern_parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            
            if i == 0 {
                // First part must match at start
                if !text[text_pos..].starts_with(part) {
                    return false;
                }
                text_pos += part.len();
            } else if i == pattern_parts.len() - 1 {
                // Last part must match at end
                return text[text_pos..].ends_with(part);
            } else {
                // Middle parts must be found in order
                if let Some(pos) = text[text_pos..].find(part) {
                    text_pos += pos + part.len();
                } else {
                    return false;
                }
            }
        }
        
        true
    }

    /// Create layer definition from pattern and matching tensors
    fn create_layer_from_pattern(
        &self,
        layer_id: usize,
        pattern: &LayerPattern,
        matching_tensors: &[&GgufTensorInfo],
    ) -> Result<LayerDefinition> {
        // Extract dimensions from tensor shapes
        let input_dim = self.extract_input_dimension(matching_tensors)?;
        let output_dim = self.extract_output_dimension(matching_tensors)?;

        // Create layer parameters based on type
        let parameters = match pattern.layer_type {
            LayerType::BitLinear => LayerParameters::BitLinear {
                weight_bits: self.config.bitlinear_config.weight_bits,
                activation_bits: self.config.bitlinear_config.activation_bits,
            },
            LayerType::RMSNorm => LayerParameters::RMSNorm {
                eps: self.config.normalization_config.rms_norm_eps,
            },
            LayerType::SwiGLU => LayerParameters::SwiGLU {
                hidden_dim: self.config.layer_config.intermediate_size,
            },
            LayerType::Embedding => LayerParameters::Embedding {
                vocab_size: self.config.tokenizer_config.vocab_size,
                embedding_dim: self.config.layer_config.hidden_size,
            },
            LayerType::OutputProjection => LayerParameters::OutputProjection {
                vocab_size: self.config.tokenizer_config.vocab_size,
            },
        };

        Ok(LayerDefinition {
            id: layer_id,
            layer_type: pattern.layer_type.clone(),
            input_dims: vec![input_dim],
            output_dims: vec![output_dim],
            parameters,
        })
    }

    /// Extract input dimension from tensor shapes
    fn extract_input_dimension(&self, tensors: &[&GgufTensorInfo]) -> Result<usize> {
        if let Some(tensor) = tensors.first() {
            // For 2D weight tensors, input dimension is typically the second dimension
            if tensor.dimensions.len() >= 2 {
                Ok(tensor.dimensions[1] as usize)
            } else if tensor.dimensions.len() == 1 {
                Ok(tensor.dimensions[0] as usize)
            } else {
                Err(InferenceError::ConfigurationError {
                    message: format!("Invalid tensor shape for input dimension: {:?}", tensor.dimensions),
                })
            }
        } else {
            Err(InferenceError::ConfigurationError {
                message: "No tensors provided for input dimension extraction".to_string(),
            })
        }
    }

    /// Extract output dimension from tensor shapes
    fn extract_output_dimension(&self, tensors: &[&GgufTensorInfo]) -> Result<usize> {
        if let Some(tensor) = tensors.first() {
            // For 2D weight tensors, output dimension is typically the first dimension
            if tensor.dimensions.len() >= 1 {
                Ok(tensor.dimensions[0] as usize)
            } else {
                Err(InferenceError::ConfigurationError {
                    message: format!("Invalid tensor shape for output dimension: {:?}", tensor.dimensions),
                })
            }
        } else {
            Err(InferenceError::ConfigurationError {
                message: "No tensors provided for output dimension extraction".to_string(),
            })
        }
    }

    /// Extract layer parameters from configuration
    fn extract_layer_parameters(&mut self) -> Result<()> {
        // This is already handled in create_layer_from_pattern
        // Additional parameter extraction logic can be added here
        Ok(())
    }

    /// Build execution graph from detected layers
    fn build_execution_graph(&self) -> Result<ModelArchitecture> {
        let mut builder = ExecutionGraphBuilder::new();
        
        // Add layers in proper execution order
        self.add_embedding_layer(&mut builder)?;
        self.add_transformer_layers(&mut builder)?;
        self.add_output_layers(&mut builder)?;
        
        builder.build()
    }

    /// Add embedding layer to execution graph
    fn add_embedding_layer(&self, builder: &mut ExecutionGraphBuilder) -> Result<()> {
        if let Some(embedding_layer) = self.layer_map.get("token_embedding") {
            builder.add_layer(embedding_layer.clone());
        }
        Ok(())
    }

    /// Add transformer layers to execution graph
    fn add_transformer_layers(&self, builder: &mut ExecutionGraphBuilder) -> Result<()> {
        for layer_idx in 0..self.config.layer_config.n_layers {
            // Add attention normalization
            if let Some(attn_norm) = self.layer_map.get("attention_norm") {
                let mut layer = attn_norm.clone();
                layer.id = builder.next_layer_id();
                builder.add_layer(layer);
            }

            // Add attention layers
            for pattern_name in &["attention_query", "attention_key", "attention_value", "attention_output"] {
                if let Some(attn_layer) = self.layer_map.get(*pattern_name) {
                    let mut layer = attn_layer.clone();
                    layer.id = builder.next_layer_id();
                    builder.add_layer(layer);
                }
            }

            // Add FFN normalization
            if let Some(ffn_norm) = self.layer_map.get("ffn_norm") {
                let mut layer = ffn_norm.clone();
                layer.id = builder.next_layer_id();
                builder.add_layer(layer);
            }

            // Add FFN layers
            for pattern_name in &["ffn_gate", "ffn_up", "ffn_down"] {
                if let Some(ffn_layer) = self.layer_map.get(*pattern_name) {
                    let mut layer = ffn_layer.clone();
                    layer.id = builder.next_layer_id();
                    builder.add_layer(layer);
                }
            }
        }
        Ok(())
    }

    /// Add output layers to execution graph
    fn add_output_layers(&self, builder: &mut ExecutionGraphBuilder) -> Result<()> {
        // Add output normalization
        if let Some(output_norm) = self.layer_map.get("output_norm") {
            let mut layer = output_norm.clone();
            layer.id = builder.next_layer_id();
            builder.add_layer(layer);
        }

        // Add output projection
        if let Some(output_proj) = self.layer_map.get("output_projection") {
            let mut layer = output_proj.clone();
            layer.id = builder.next_layer_id();
            builder.add_layer(layer);
        }

        Ok(())
    }

    /// Validate the constructed architecture
    fn validate_architecture(&self, architecture: &ModelArchitecture) -> Result<()> {
        if architecture.layers.is_empty() {
            return Err(InferenceError::ConfigurationError {
                message: "No layers detected in architecture".to_string(),
            });
        }

        if architecture.execution_order.is_empty() {
            return Err(InferenceError::ConfigurationError {
                message: "No execution order defined".to_string(),
            });
        }

        // Validate layer count matches expected configuration
        let expected_layers = self.estimate_expected_layers();
        if architecture.layers.len() < expected_layers / 2 {
            tracing::warn!("Detected {} layers, expected approximately {}", 
                         architecture.layers.len(), expected_layers);
        }

        tracing::info!("Architecture validation passed: {} layers, {} execution steps",
                      architecture.layers.len(), architecture.execution_order.len());

        Ok(())
    }

    /// Estimate expected number of layers based on configuration
    fn estimate_expected_layers(&self) -> usize {
        // Approximate calculation:
        // - 1 embedding layer
        // - n_layers * (2 norm + 4 attention + 3 ffn) = n_layers * 9
        // - 1 output norm + 1 output projection
        1 + (self.config.layer_config.n_layers * 9) + 2
    }
}

impl ExecutionGraphBuilder {
    fn new() -> Self {
        Self {
            layers: Vec::new(),
            execution_order: Vec::new(),
            layer_dependencies: HashMap::new(),
        }
    }

    fn add_layer(&mut self, layer: LayerDefinition) {
        let layer_id = layer.id;
        self.layers.push(layer);
        self.execution_order.push(layer_id);
    }

    fn next_layer_id(&self) -> usize {
        self.layers.len()
    }

    fn build(self) -> Result<ModelArchitecture> {
        Ok(ModelArchitecture {
            layers: self.layers,
            execution_order: self.execution_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet_config::*;

    fn create_test_config() -> BitNetModelConfig {
        BitNetModelConfig {
            basic_info: BasicModelInfo {
                name: "test-model".to_string(),
                architecture: "bitnet-b1.58".to_string(),
                version: "1.0".to_string(),
                parameter_count: 1_000_000,
                context_length: 4096,
            },
            layer_config: LayerConfig {
                n_layers: 2,
                hidden_size: 512,
                intermediate_size: 2048,
                model_dim: 512,
            },
            attention_config: AttentionConfig {
                n_heads: 8,
                n_kv_heads: Some(8),
                head_dim: 64,
                max_seq_len: 4096,
                rope_config: RopeConfig {
                    rope_freq_base: 10000.0,
                    rope_scaling: None,
                    rope_dim: 64,
                },
            },
            normalization_config: NormalizationConfig {
                rms_norm_eps: 1e-5,
                use_bias: false,
            },
            bitlinear_config: BitLinearConfig {
                weight_bits: 2,
                activation_bits: 8,
                use_weight_scaling: true,
                use_activation_scaling: true,
                quantization_scheme: "1.58bit".to_string(),
            },
            tokenizer_config: TokenizerConfig {
                vocab_size: 32000,
                tokenizer_type: "llama".to_string(),
                bos_token_id: Some(1),
                eos_token_id: Some(2),
                pad_token_id: None,
            },
            extra_metadata: HashMap::new(),
        }
    }

    fn create_test_tensor_info() -> Vec<GgufTensorInfo> {
        use crate::gguf::GgufTensorType;
        
        vec![
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                dimensions: vec![32000, 512],
                tensor_type: GgufTensorType::F32,
                offset: 0,
            },
            GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dimensions: vec![512, 512],
                tensor_type: GgufTensorType::F32,
                offset: 1000,
            },
            GgufTensorInfo {
                name: "blk.0.attn_norm.weight".to_string(),
                dimensions: vec![512],
                tensor_type: GgufTensorType::F32,
                offset: 2000,
            },
            GgufTensorInfo {
                name: "output.weight".to_string(),
                dimensions: vec![32000, 512],
                tensor_type: GgufTensorType::F32,
                offset: 3000,
            },
        ]
    }

    #[test]
    fn test_pattern_matching() {
        let config = create_test_config();
        let tensor_info = create_test_tensor_info();
        let mapper = ArchitectureMapper::new(config, tensor_info);

        assert!(mapper.matches_pattern("blk.0.attn_q.weight", "blk.*.attn_q.weight"));
        assert!(mapper.matches_pattern("token_embd.weight", "token_embd.weight"));
        assert!(!mapper.matches_pattern("other.weight", "blk.*.attn_q.weight"));
    }

    #[test]
    fn test_layer_detection() {
        let config = create_test_config();
        let tensor_info = create_test_tensor_info();
        let mut mapper = ArchitectureMapper::new(config, tensor_info);

        mapper.detect_layer_types().unwrap();
        assert!(!mapper.layer_map.is_empty());
        assert!(mapper.layer_map.contains_key("token_embedding"));
        assert!(mapper.layer_map.contains_key("attention_query"));
    }

    #[test]
    fn test_architecture_mapping() {
        let config = create_test_config();
        let tensor_info = create_test_tensor_info();
        let mut mapper = ArchitectureMapper::new(config, tensor_info);

        let architecture = mapper.map_to_architecture().unwrap();
        assert!(!architecture.layers.is_empty());
        assert!(!architecture.execution_order.is_empty());
    }
}