use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for LoRA adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// LoRA rank (typically 8-64)
    pub rank: usize,
    /// LoRA scaling factor
    pub alpha: f32,
    /// Dropout rate for LoRA layers
    pub dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
    /// Whether to apply LoRA to bias terms
    pub bias: bool,
    /// Task type for fine-tuning
    pub task_type: String,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![
                "attn_q".to_string(),
                "attn_k".to_string(),
                "attn_v".to_string(),
                "attn_output".to_string(),
                "ffn_up".to_string(),
                "ffn_down".to_string(),
            ],
            bias: false,
            task_type: "CAUSAL_LM".to_string(),
        }
    }
}

/// Configuration for quantized LoRA (QLoRA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAQuantConfig {
    /// Base LoRA configuration
    pub lora_config: LoRAConfig,
    /// Quantization bits for LoRA weights (4, 8, or 16)
    pub lora_bits: u8,
    /// Block size for quantization
    pub block_size: usize,
    /// Use double quantization
    pub double_quant: bool,
    /// Quantization type
    pub quant_type: String,
}

impl Default for LoRAQuantConfig {
    fn default() -> Self {
        Self {
            lora_config: LoRAConfig::default(),
            lora_bits: 8,
            block_size: 128,
            double_quant: true,
            quant_type: "nf4".to_string(),
        }
    }
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }
    
    /// Create a configuration optimized for BitNet models
    pub fn for_bitnet(rank: usize) -> Self {
        Self {
            rank,
            alpha: rank as f32 * 2.0, // Common scaling
            dropout: 0.05, // Lower dropout for ternary weights
            target_modules: vec![
                "attn_q".to_string(),
                "attn_k".to_string(),
                "attn_v".to_string(),
                "attn_output".to_string(),
                "ffn_up".to_string(),
                "ffn_down".to_string(),
                "ffn_gate".to_string(), // BitNet-specific
            ],
            bias: false,
            task_type: "CAUSAL_LM".to_string(),
        }
    }
    
    /// Calculate the reduction in trainable parameters
    pub fn parameter_reduction(&self, base_params: usize, vocab_size: usize) -> f32 {
        let target_layers = self.target_modules.len();
        let avg_layer_size = base_params / target_layers; // Rough estimate
        
        // LoRA parameters = rank * (input_size + output_size) per layer
        let lora_params = target_layers * self.rank * 2 * (avg_layer_size as f32).sqrt() as usize;
        
        1.0 - (lora_params as f32 / base_params as f32)
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(anyhow::anyhow!("LoRA rank must be greater than 0"));
        }
        
        if self.rank > 512 {
            return Err(anyhow::anyhow!("LoRA rank too large, may cause memory issues"));
        }
        
        if self.alpha <= 0.0 {
            return Err(anyhow::anyhow!("LoRA alpha must be positive"));
        }
        
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(anyhow::anyhow!("Dropout must be in range [0.0, 1.0)"));
        }
        
        if self.target_modules.is_empty() {
            return Err(anyhow::anyhow!("At least one target module must be specified"));
        }
        
        Ok(())
    }
}

impl LoRAQuantConfig {
    /// Create a new quantized LoRA configuration
    pub fn new(lora_config: LoRAConfig, lora_bits: u8) -> Self {
        Self {
            lora_config,
            lora_bits,
            ..Default::default()
        }
    }
    
    /// Create a QLoRA configuration optimized for memory efficiency
    pub fn memory_efficient(rank: usize) -> Self {
        Self {
            lora_config: LoRAConfig::for_bitnet(rank),
            lora_bits: 4, // 4-bit quantization for maximum memory savings
            block_size: 64,
            double_quant: true,
            quant_type: "nf4".to_string(),
        }
    }
    
    /// Validate the quantization configuration
    pub fn validate(&self) -> Result<()> {
        self.lora_config.validate()?;
        
        if ![4, 8, 16].contains(&self.lora_bits) {
            return Err(anyhow::anyhow!(
                "Unsupported quantization bits: {}. Supported: 4, 8, 16", 
                self.lora_bits
            ));
        }
        
        if self.block_size == 0 || self.block_size > 512 {
            return Err(anyhow::anyhow!(
                "Block size must be between 1 and 512, got: {}", 
                self.block_size
            ));
        }
        
        Ok(())
    }
    
    /// Estimate memory savings from quantization
    pub fn memory_savings(&self) -> f32 {
        let base_bits = 32.0; // FP32 baseline
        let savings = 1.0 - (self.lora_bits as f32 / base_bits);
        
        if self.double_quant {
            savings * 1.1 // Additional savings from double quantization
        } else {
            savings
        }
    }
}