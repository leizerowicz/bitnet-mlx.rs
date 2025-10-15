//! Advanced sampling methods for text generation
//! 
//! This module implements various sampling strategies for BitNet text generation,
//! including temperature scaling, top-k, top-p (nucleus) sampling.

use crate::{Result, InferenceError};
use crate::api::GenerationConfig;
use bitnet_core::Tensor;

#[cfg(feature = "generation")]
use rand::prelude::*;
#[cfg(feature = "generation")]
use rand::distributions::WeightedIndex;

/// Advanced sampler for token generation
#[cfg(feature = "generation")]
pub struct TokenSampler {
    rng: StdRng,
}

#[cfg(feature = "generation")]
impl TokenSampler {
    /// Create a new sampler with optional seed
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        Self { rng }
    }
    
    /// Sample next token from logits using the specified configuration
    pub fn sample_token(&mut self, logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
        // Get logits data (assuming 2D tensor with shape [batch, vocab_size])
        let logits_data = logits.to_vec2::<f32>()
            .map_err(|e| InferenceError::tensor_operation(format!("Failed to get logits data: {}", e)))?;
        
        // Take the first batch (since we're using batch_size=1)
        let batch_logits = logits_data.first()
            .ok_or_else(|| InferenceError::sampling("Empty logits tensor"))?;
        
        if !config.do_sample {
            // Greedy sampling - just return the token with highest probability
            return self.greedy_sample(batch_logits);
        }
        
        // Apply temperature scaling
        let mut scaled_logits = if config.temperature != 1.0 && config.temperature > 0.0 {
            self.apply_temperature(batch_logits, config.temperature)
        } else {
            batch_logits.to_vec()
        };
        
        // Apply top-k filtering if specified
        if let Some(k) = config.top_k {
            if k > 0 && k < scaled_logits.len() {
                self.apply_top_k(&mut scaled_logits, k);
            }
        }
        
        // Apply top-p (nucleus) filtering if specified
        if let Some(p) = config.top_p {
            if p > 0.0 && p < 1.0 {
                self.apply_top_p(&mut scaled_logits, p)?;
            }
        }
        
        // Convert logits to probabilities
        let probabilities = self.softmax(&scaled_logits);
        
        // Sample from the probability distribution
        self.multinomial_sample(&probabilities)
    }
    
    /// Greedy sampling - select token with highest probability
    fn greedy_sample(&self, logits: &[f32]) -> Result<u32> {
        let best_token_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| InferenceError::sampling("No valid tokens found in logits"))?;
        
        Ok(best_token_idx as u32)
    }
    
    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &[f32], temperature: f32) -> Vec<f32> {
        logits.iter().map(|&x| x / temperature).collect()
    }
    
    /// Apply top-k filtering - keep only the k tokens with highest probability
    fn apply_top_k(&self, logits: &mut [f32], k: usize) {
        // Get indices sorted by logit values (descending)
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal));
        
        // Set logits to negative infinity for tokens outside top-k
        for &idx in indices.iter().skip(k) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
    
    /// Apply top-p (nucleus) filtering - keep tokens with cumulative probability up to p
    fn apply_top_p(&self, logits: &mut [f32], p: f32) -> Result<()> {
        // Convert to probabilities for cumulative calculation
        let probabilities = self.softmax(logits);
        
        // Get indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&i, &j| probabilities[j].partial_cmp(&probabilities[i]).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate cumulative probabilities
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probabilities.len();
        
        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probabilities[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Set logits to negative infinity for tokens outside nucleus
        for &idx in indices.iter().skip(cutoff_idx) {
            logits[idx] = f32::NEG_INFINITY;
        }
        
        Ok(())
    }
    
    /// Convert logits to probabilities using softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        // Compute sum for normalization
        let sum: f32 = exp_logits.iter().sum();
        
        // Normalize to get probabilities
        if sum > 0.0 {
            exp_logits.iter().map(|&x| x / sum).collect()
        } else {
            // Fallback to uniform distribution if all logits are negative infinity
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }
    
    /// Sample from multinomial distribution
    fn multinomial_sample(&mut self, probabilities: &[f32]) -> Result<u32> {
        // Filter out zero probabilities and create weighted distribution
        let non_zero_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0 && p.is_finite())
            .map(|(i, &p)| (i, p))
            .collect();
        
        if non_zero_probs.is_empty() {
            return Err(InferenceError::sampling("No valid tokens with non-zero probability"));
        }
        
        // Create weighted index for sampling
        let weights: Vec<f32> = non_zero_probs.iter().map(|(_, p)| *p).collect();
        let indices: Vec<usize> = non_zero_probs.iter().map(|(i, _)| *i).collect();
        
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| InferenceError::sampling(format!("Failed to create weighted distribution: {}", e)))?;
        
        let sampled_idx = dist.sample(&mut self.rng);
        Ok(indices[sampled_idx] as u32)
    }
}

/// Validation functions for generation config
#[cfg(feature = "generation")]
impl GenerationConfig {
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(InferenceError::configuration("Temperature must be positive"));
        }
        
        if self.temperature > 10.0 {
            return Err(InferenceError::configuration("Temperature should be <= 10.0 for reasonable results"));
        }
        
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(InferenceError::configuration("top_k must be greater than 0"));
            }
        }
        
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 {
                return Err(InferenceError::configuration("top_p must be in (0, 1]"));
            }
        }
        
        if let Some(p) = self.typical_p {
            if p <= 0.0 || p > 1.0 {
                return Err(InferenceError::configuration("typical_p must be in (0, 1]"));
            }
        }
        
        if self.max_length == 0 {
            return Err(InferenceError::configuration("max_length must be greater than 0"));
        }
        
        if let Some(max_context) = self.max_context_length {
            if max_context == 0 {
                return Err(InferenceError::configuration("max_context_length must be greater than 0"));
            }
        }
        
        if let Some(penalty) = self.repetition_penalty {
            if penalty <= 0.0 {
                return Err(InferenceError::configuration("repetition_penalty must be positive"));
            }
        }
        
        if let Some(penalty) = self.length_penalty {
            if penalty <= 0.0 {
                return Err(InferenceError::configuration("length_penalty must be positive"));
            }
        }
        
        Ok(())
    }
    
    /// Create a conservative configuration for stable generation
    pub fn conservative() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.8),
            typical_p: Some(0.95),
            max_length: 256,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.05),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        }
    }
    
    /// Create a creative configuration for diverse generation
    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_k: Some(100),
            top_p: Some(0.95),
            typical_p: Some(0.98),
            max_length: 512,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.15),
            length_penalty: Some(1.2),
            use_lut_acceleration: true,
            target_latency_ms: Some(100),
        }
    }
    
    /// Create a deterministic configuration for reproducible generation
    pub fn deterministic(seed: u64) -> Self {
        Self {
            temperature: 0.1,
            top_k: Some(1),
            top_p: None,
            typical_p: None,
            max_length: 256,
            max_context_length: Some(2048),
            do_sample: false,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: Some(seed),
            early_stopping: true,
            repetition_penalty: None,
            length_penalty: None,
            use_lut_acceleration: true,
            target_latency_ms: Some(25),
        }
    }
}

#[cfg(all(test, feature = "generation"))]
mod tests {
    use super::*;
    use bitnet_core::{Tensor, Device};
    
    #[test]
    fn test_sampler_creation() {
        let sampler = TokenSampler::new(Some(42));
        // Just verify it doesn't panic
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = GenerationConfig::default();
        assert!(config.validate().is_ok());
        
        config.temperature = -1.0;
        assert!(config.validate().is_err());
        
        config.temperature = 1.0;
        config.top_k = Some(0);
        assert!(config.validate().is_err());
        
        config.top_k = Some(50);
        config.top_p = Some(1.5);
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_preset_configs() {
        let conservative = GenerationConfig::conservative();
        assert!(conservative.validate().is_ok());
        assert_eq!(conservative.temperature, 0.7);
        
        let creative = GenerationConfig::creative();
        assert!(creative.validate().is_ok());
        assert_eq!(creative.temperature, 1.2);
        
        let deterministic = GenerationConfig::deterministic(42);
        assert!(deterministic.validate().is_ok());
        assert_eq!(deterministic.seed, Some(42));
        assert!(!deterministic.do_sample);
    }
    
    #[test]
    fn test_softmax() {
        let sampler = TokenSampler::new(Some(42));
        let logits = vec![1.0, 2.0, 3.0];
        let probs = sampler.softmax(&logits);
        
        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check probabilities are in correct order
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}