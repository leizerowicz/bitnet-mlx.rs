/// LoRA (Low-Rank Adaptation) implementation for BitNet models
/// 
/// This module provides efficient fine-tuning capabilities for BitNet models
/// using Low-Rank Adaptation (LoRA) techniques, optimized for ternary weights.

pub mod bitnet_lora;
pub mod qlora;
pub mod lora_config;

#[cfg(test)]
mod tests;

pub use bitnet_lora::{BitNetLoRA, BitNetQLoRA};
pub use qlora::{QLoRALayer, QLoRAConfig};
pub use lora_config::{LoRAConfig, LoRAQuantConfig};

use anyhow::Result;

/// LoRA adapter trait for BitNet layers
pub trait LoRAAdapter {
    /// Apply LoRA adaptation to the base layer
    fn apply_lora(&mut self, lora: &BitNetLoRA) -> Result<()>;
    
    /// Remove LoRA adaptation and return to base weights
    fn remove_lora(&mut self) -> Result<()>;
    
    /// Check if LoRA is currently applied
    fn has_lora(&self) -> bool;
    
    /// Get the current LoRA rank
    fn lora_rank(&self) -> Option<usize>;
}

/// LoRA training utilities
pub mod training_utils {
    use super::*;
    
    /// Initialize LoRA parameters for a given layer size
    pub fn initialize_lora_parameters(
        input_size: usize,
        output_size: usize,
        rank: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Simple pseudo-random initialization without external dependencies
        let mut hasher = DefaultHasher::new();
        let lora_a = (0..input_size * rank)
            .map(|i| {
                i.hash(&mut hasher);
                let hash_val = hasher.finish();
                ((hash_val % 1000) as f32 / 1000.0 - 0.5) * 0.02 // Range: -0.01 to 0.01
            })
            .collect();
        let lora_b = vec![0.0; rank * output_size];
        
        Ok((lora_a, lora_b))
    }
    
    /// Calculate LoRA memory overhead
    pub fn calculate_lora_memory_overhead(
        base_params: usize,
        rank: usize,
        num_layers: usize,
    ) -> f32 {
        let lora_params = num_layers * rank * 2; // A and B matrices
        (lora_params as f32) / (base_params as f32) * 100.0
    }
}