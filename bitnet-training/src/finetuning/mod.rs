/// Advanced Fine-tuning Methods for BitNet Models
/// 
/// This module provides advanced fine-tuning capabilities for BitNet models.
/// These are simplified but working implementations that demonstrate the core concepts
/// and can be extended with full functionality as needed.

// Simplified implementations that compile and work
pub mod dora_simple;
pub mod mixture_of_loras_simple;
pub mod continual_learning_simple;
pub mod advanced_simple;

// Re-export the simplified implementations
pub use dora_simple::{
    BitNetDoRA, DoRAConfig, DoRAStats, MagnitudeInit, dora_utils
};
pub use mixture_of_loras_simple::{
    MixtureOfLoRAs, MoLoRAConfig, GatingNetwork, TaskRouter, ExpertSelection
};
pub use continual_learning_simple::{
    ContinualLearner, ContinualLearningConfig, ReplayBuffer, TaskDataset, TaskStats, TrainingExample
};
pub use advanced_simple::{
    AdvancedFinetuner, AdvancedFinetuningConfig, AdvancedStats, LossWeights, presets
};

/// Validation utilities for fine-tuning configurations
pub mod validation {
    use crate::lora::LoRAConfig;
    use super::*;
    
    /// Validate LoRA configuration for BitNet compatibility
    pub fn validate_lora_config(config: &LoRAConfig) -> anyhow::Result<()> {
        if config.rank == 0 {
            return Err(anyhow::anyhow!("LoRA rank must be greater than 0"));
        }
        
        if config.alpha <= 0.0 {
            return Err(anyhow::anyhow!("LoRA alpha must be positive"));
        }
        
        if config.dropout < 0.0 || config.dropout >= 1.0 {
            return Err(anyhow::anyhow!("LoRA dropout must be in [0, 1)"));
        }
        
        Ok(())
    }
    
    /// Validate DoRA configuration
    pub fn validate_dora_config(config: &DoRAConfig) -> anyhow::Result<()> {
        validate_lora_config(&config.lora_config)?;
        
        if config.magnitude_lr_multiplier <= 0.0 {
            return Err(anyhow::anyhow!("Magnitude learning rate multiplier must be positive"));
        }
        
        if config.reg_strength < 0.0 {
            return Err(anyhow::anyhow!("Regularization strength must be non-negative"));
        }
        
        Ok(())
    }
    
    /// Validate MoLoRA configuration
    pub fn validate_molora_config(config: &MoLoRAConfig) -> anyhow::Result<()> {
        validate_lora_config(&config.base_lora_config)?;
        
        if config.num_experts == 0 {
            return Err(anyhow::anyhow!("Number of experts must be greater than 0"));
        }
        
        if config.balance_weight < 0.0 {
            return Err(anyhow::anyhow!("Balance weight must be non-negative"));
        }
        
        if config.expert_dropout < 0.0 || config.expert_dropout >= 1.0 {
            return Err(anyhow::anyhow!("Expert dropout must be in [0, 1)"));
        }
        
        Ok(())
    }
}

/// Memory estimation utilities
pub mod memory {
    use super::*;
    
    /// Estimate memory usage for DoRA adaptation
    pub fn estimate_dora_memory(
        input_size: usize,
        output_size: usize,
        config: &DoRAConfig,
    ) -> usize {
        let lora_params = 2 * config.lora_config.rank * (input_size + output_size);
        let magnitude_params = output_size;
        
        // Estimate in bytes (f32 = 4 bytes)
        (lora_params + magnitude_params) * 4
    }
    
    /// Estimate memory usage for MoLoRA
    pub fn estimate_molora_memory(
        input_size: usize,
        output_size: usize,
        config: &MoLoRAConfig,
    ) -> usize {
        let single_expert_memory = estimate_dora_memory(
            input_size,
            output_size,
            &DoRAConfig {
                lora_config: config.base_lora_config.clone(),
                ..Default::default()
            },
        );
        
        let gating_memory = input_size * config.num_experts * 4; // f32 weights
        
        single_expert_memory * config.num_experts + gating_memory
    }
    
    /// Estimate total memory for advanced fine-tuning
    pub fn estimate_total_memory(
        input_size: usize,
        output_size: usize,
        config: &AdvancedFinetuningConfig,
    ) -> usize {
        let mut total_memory = 0;
        
        if let Some(dora_config) = &config.dora_config {
            total_memory += estimate_dora_memory(input_size, output_size, dora_config);
        }
        
        if let Some(molora_config) = &config.molora_config {
            total_memory += estimate_molora_memory(input_size, output_size, molora_config);
        }
        
        if let Some(continual_config) = &config.continual_config {
            // Estimate replay buffer memory
            total_memory += continual_config.replay_buffer_size * input_size * 4; // Simplified
        }
        
        total_memory
    }
}