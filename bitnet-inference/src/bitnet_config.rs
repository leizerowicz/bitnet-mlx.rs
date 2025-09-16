//! BitNet-specific model configuration structures
//! 
//! This module defines configuration structures for BitNet models loaded from GGUF files,
//! including layer parameters, attention configuration, and quantization settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, InferenceError};

/// Comprehensive BitNet model configuration extracted from GGUF metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetModelConfig {
    /// Basic model information
    pub basic_info: BasicModelInfo,
    /// Layer configuration
    pub layer_config: LayerConfig,
    /// Attention mechanism configuration
    pub attention_config: AttentionConfig,
    /// Normalization layer configuration
    pub normalization_config: NormalizationConfig,
    /// BitLinear layer-specific configuration
    pub bitlinear_config: BitLinearConfig,
    /// Tokenizer configuration
    pub tokenizer_config: TokenizerConfig,
    /// Additional metadata not covered by specific configs
    pub extra_metadata: HashMap<String, String>,
}

/// Basic model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicModelInfo {
    /// Model name
    pub name: String,
    /// Model architecture type (e.g., "bitnet-b1.58")
    pub architecture: String,
    /// Model version
    pub version: String,
    /// Total parameter count
    pub parameter_count: usize,
    /// Context length
    pub context_length: usize,
}

/// Layer configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Number of transformer layers
    pub n_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Intermediate dimension size (usually hidden_size * 4)
    pub intermediate_size: usize,
    /// Model dimensionality
    pub model_dim: usize,
}

/// Multi-head attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for grouped-query attention)
    pub n_kv_heads: Option<usize>,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE (Rotary Position Embedding) configuration
    pub rope_config: RopeConfig,
}

/// RoPE (Rotary Position Embedding) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    /// RoPE base frequency
    pub rope_freq_base: f32,
    /// RoPE scaling factor
    pub rope_scaling: Option<f32>,
    /// RoPE dimension
    pub rope_dim: usize,
}

/// Layer normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// RMSNorm epsilon value
    pub rms_norm_eps: f32,
    /// Whether to use bias in normalization layers
    pub use_bias: bool,
}

/// BitLinear layer-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitLinearConfig {
    /// Weight quantization bits (typically 2 for 1.58-bit quantization)
    pub weight_bits: u8,
    /// Activation quantization bits (typically 8)
    pub activation_bits: u8,
    /// Whether to use weight scaling
    pub use_weight_scaling: bool,
    /// Whether to use activation scaling
    pub use_activation_scaling: bool,
    /// Quantization scheme identifier
    pub quantization_scheme: String,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Tokenizer type (e.g., "llama3")
    pub tokenizer_type: String,
    /// Beginning of sequence token ID
    pub bos_token_id: Option<u32>,
    /// End of sequence token ID
    pub eos_token_id: Option<u32>,
    /// Padding token ID
    pub pad_token_id: Option<u32>,
}

impl BitNetModelConfig {
    /// Create a new BitNet model configuration with default values
    pub fn new() -> Self {
        Self {
            basic_info: BasicModelInfo {
                name: "bitnet-model".to_string(),
                architecture: "bitnet-b1.58".to_string(),
                version: "1.0".to_string(),
                parameter_count: 0,
                context_length: 4096,
            },
            layer_config: LayerConfig {
                n_layers: 32,
                hidden_size: 2048,
                intermediate_size: 8192,
                model_dim: 2048,
            },
            attention_config: AttentionConfig {
                n_heads: 32,
                n_kv_heads: None,
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
                vocab_size: 128256, // Default LLaMA 3 vocab size
                tokenizer_type: "llama3".to_string(),
                bos_token_id: Some(1),
                eos_token_id: Some(2),
                pad_token_id: None,
            },
            extra_metadata: HashMap::new(),
        }
    }

    /// Validate the configuration for consistency
    pub fn validate(&self) -> Result<()> {
        // Validate basic info
        if self.basic_info.parameter_count == 0 {
            return Err(InferenceError::model_load("Parameter count cannot be zero"));
        }

        if self.basic_info.context_length == 0 {
            return Err(InferenceError::model_load("Context length cannot be zero"));
        }

        // Validate layer config
        if self.layer_config.n_layers == 0 {
            return Err(InferenceError::model_load("Number of layers cannot be zero"));
        }

        if self.layer_config.hidden_size == 0 {
            return Err(InferenceError::model_load("Hidden size cannot be zero"));
        }

        // Validate attention config
        if self.attention_config.n_heads == 0 {
            return Err(InferenceError::model_load("Number of attention heads cannot be zero"));
        }

        if self.attention_config.head_dim == 0 {
            return Err(InferenceError::model_load("Head dimension cannot be zero"));
        }

        // Check if hidden_size is divisible by n_heads
        if self.layer_config.hidden_size % self.attention_config.n_heads != 0 {
            return Err(InferenceError::model_load(
                "Hidden size must be divisible by number of attention heads"
            ));
        }

        // Validate that head_dim matches calculated value
        let expected_head_dim = self.layer_config.hidden_size / self.attention_config.n_heads;
        if self.attention_config.head_dim != expected_head_dim {
            return Err(InferenceError::model_load(format!(
                "Head dimension {} doesn't match calculated value {} (hidden_size {} / n_heads {})",
                self.attention_config.head_dim, expected_head_dim, 
                self.layer_config.hidden_size, self.attention_config.n_heads
            )));
        }

        // Validate BitLinear config
        if self.bitlinear_config.weight_bits == 0 || self.bitlinear_config.weight_bits > 8 {
            return Err(InferenceError::model_load("Weight bits must be between 1 and 8"));
        }

        if self.bitlinear_config.activation_bits == 0 || self.bitlinear_config.activation_bits > 32 {
            return Err(InferenceError::model_load("Activation bits must be between 1 and 32"));
        }

        // Validate tokenizer config
        if self.tokenizer_config.vocab_size == 0 {
            return Err(InferenceError::model_load("Vocabulary size cannot be zero"));
        }

        Ok(())
    }

    /// Get the calculated head dimension
    pub fn calculated_head_dim(&self) -> usize {
        self.layer_config.hidden_size / self.attention_config.n_heads
    }

    /// Check if this is a grouped-query attention model
    pub fn uses_grouped_query_attention(&self) -> bool {
        self.attention_config.n_kv_heads.is_some() && 
        self.attention_config.n_kv_heads.unwrap() != self.attention_config.n_heads
    }

    /// Get the effective number of key-value heads
    pub fn effective_n_kv_heads(&self) -> usize {
        self.attention_config.n_kv_heads.unwrap_or(self.attention_config.n_heads)
    }
}

impl Default for BitNetModelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Common GGUF metadata keys for BitNet models
pub struct GgufKeys;

impl GgufKeys {
    // Architecture and basic info
    pub const GENERAL_ARCHITECTURE: &'static str = "general.architecture";
    pub const GENERAL_NAME: &'static str = "general.name";
    pub const GENERAL_VERSION: &'static str = "general.version";
    
    // Layer configuration
    pub const LAYER_COUNT: &'static str = "bitnet.block_count";
    pub const HIDDEN_SIZE: &'static str = "bitnet.embedding_length";
    pub const INTERMEDIATE_SIZE: &'static str = "bitnet.feed_forward_length";
    
    // Attention configuration
    pub const ATTENTION_HEAD_COUNT: &'static str = "bitnet.attention.head_count";
    pub const ATTENTION_HEAD_COUNT_KV: &'static str = "bitnet.attention.head_count_kv";
    pub const ATTENTION_LAYER_NORM_RMS_EPS: &'static str = "bitnet.attention.layer_norm_rms_epsilon";
    
    // RoPE configuration
    pub const ROPE_DIMENSION_COUNT: &'static str = "bitnet.rope.dimension_count";
    pub const ROPE_FREQ_BASE: &'static str = "bitnet.rope.freq_base";
    pub const ROPE_SCALING_TYPE: &'static str = "bitnet.rope.scaling.type";
    pub const ROPE_SCALING_FACTOR: &'static str = "bitnet.rope.scaling.factor";
    
    // Context and sequence
    pub const CONTEXT_LENGTH: &'static str = "bitnet.context_length";
    
    // Tokenizer configuration
    pub const TOKENIZER_GGML_MODEL: &'static str = "tokenizer.ggml.model";
    pub const TOKENIZER_GGML_TOKENS: &'static str = "tokenizer.ggml.tokens";
    pub const TOKENIZER_GGML_TOKEN_TYPE: &'static str = "tokenizer.ggml.token_type";
    pub const TOKENIZER_GGML_BOS_TOKEN_ID: &'static str = "tokenizer.ggml.bos_token_id";
    pub const TOKENIZER_GGML_EOS_TOKEN_ID: &'static str = "tokenizer.ggml.eos_token_id";
    pub const TOKENIZER_GGML_PAD_TOKEN_ID: &'static str = "tokenizer.ggml.pad_token_id";
    
    // BitNet-specific quantization
    pub const BITNET_VERSION: &'static str = "bitnet.version";
    pub const BITNET_WEIGHT_BITS: &'static str = "bitnet.weight_bits";
    pub const BITNET_ACTIVATION_BITS: &'static str = "bitnet.activation_bits";
}