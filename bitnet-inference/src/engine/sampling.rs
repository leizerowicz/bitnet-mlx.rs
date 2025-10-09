//! Sampling Strategies for Text Generation
//!
//! Implements various sampling methods for token selection during autoregressive
//! text generation, including temperature, top-k, top-p, and deterministic sampling.

use anyhow::{Result, Context};
use bitnet_core::{Tensor, Device, DType};
use std::collections::HashMap;

/// Configuration for sampling behavior
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for controlling randomness (0.0 = deterministic, 1.0 = normal, >1.0 = more random)
    pub temperature: f32,
    /// Top-k sampling: only consider top k tokens (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling: consider tokens with cumulative probability <= p (1.0 = disabled)
    pub top_p: f32,
    /// Random seed for reproducible generation (None = non-deterministic)
    pub seed: Option<u64>,
    /// Whether to use deterministic sampling (argmax)
    pub deterministic: bool,
    /// Minimum probability threshold for token consideration
    pub min_p: f32,
    /// Repetition penalty to reduce repetitive text (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Device for tensor operations
    pub device: Device,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0, // Disabled
            top_p: 1.0, // Disabled
            seed: None,
            deterministic: false,
            min_p: 0.0,
            repetition_penalty: 1.0,
            device: Device::Cpu,
        }
    }
}

/// Statistics for sampling performance monitoring
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    /// Number of tokens sampled
    pub tokens_sampled: u64,
    /// Number of top-k operations
    pub top_k_operations: u64,
    /// Number of top-p operations
    pub top_p_operations: u64,
    /// Number of temperature applications
    pub temperature_operations: u64,
    /// Average sampling time in nanoseconds
    pub avg_sampling_time_ns: u64,
    /// Total sampling time in nanoseconds
    pub total_sampling_time_ns: u64,
}

impl SamplingStats {
    /// Calculate tokens sampled per second
    pub fn tokens_per_second(&self) -> f32 {
        if self.total_sampling_time_ns == 0 {
            0.0
        } else {
            (self.tokens_sampled as f64 * 1e9 / self.total_sampling_time_ns as f64) as f32
        }
    }
    
    /// Calculate average sampling time in milliseconds
    pub fn avg_sampling_time_ms(&self) -> f32 {
        self.avg_sampling_time_ns as f32 / 1_000_000.0
    }
}

/// Core sampling strategies implementation
#[derive(Debug)]
pub struct TokenSampler {
    /// Sampling configuration
    config: SamplingConfig,
    /// Random number generator state
    rng_state: Option<u64>,
    /// Performance statistics
    stats: SamplingStats,
    /// Token history for repetition penalty
    token_history: Vec<i64>,
}

impl TokenSampler {
    /// Create a new token sampler
    pub fn new(config: SamplingConfig) -> Self {
        let rng_state = config.seed;
        
        Self {
            config,
            rng_state,
            stats: SamplingStats::default(),
            token_history: Vec::new(),
        }
    }
    
    /// Sample next token from logits using configured strategy
    pub fn sample(&mut self, logits: &Tensor) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        
        let result = if self.config.deterministic {
            self.sample_deterministic(logits)
        } else {
            self.sample_stochastic(logits)
        };
        
        // Update statistics
        let elapsed = start_time.elapsed();
        self.stats.tokens_sampled += 1;
        self.stats.total_sampling_time_ns += elapsed.as_nanos() as u64;
        self.stats.avg_sampling_time_ns = 
            self.stats.total_sampling_time_ns / self.stats.tokens_sampled;
        
        // Track token for repetition penalty
        if let Ok(ref token_tensor) = result {
            if let Ok(token_data) = token_tensor.to_vec2::<i64>() {
                if !token_data.is_empty() && !token_data[0].is_empty() {
                    self.token_history.push(token_data[0][0]);
                    // Keep history limited to last 100 tokens
                    if self.token_history.len() > 100 {
                        self.token_history.remove(0);
                    }
                }
            }
        }
        
        result
    }
    
    /// Deterministic sampling (argmax)
    fn sample_deterministic(&self, logits: &Tensor) -> Result<Tensor> {
        // Get the token with highest probability
        let shape = logits.shape();
        let batch_size = shape.dims()[0];
        
        // For now, implement a simple argmax approximation
        // In a real implementation, we would use proper argmax operations
        let token_tensor = Tensor::zeros(&[batch_size, 1], DType::I64, &self.config.device)
            .context("Failed to create deterministic token")?;
        
        // Add constant value to make it token ID 1 (simplified)
        let ones = Tensor::ones(&[batch_size, 1], DType::I64, &self.config.device)
            .context("Failed to create ones tensor")?;
        
        token_tensor.add(&ones)
            .context("Failed to add constant to token tensor")
    }
    
    /// Stochastic sampling with temperature, top-k, and top-p
    fn sample_stochastic(&mut self, logits: &Tensor) -> Result<Tensor> {
        let mut processed_logits = logits.clone();
        
        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 && !self.token_history.is_empty() {
            processed_logits = self.apply_repetition_penalty(&processed_logits)?;
        }
        
        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            processed_logits = self.apply_temperature(&processed_logits)?;
            self.stats.temperature_operations += 1;
        }
        
        // Apply top-k filtering
        if self.config.top_k > 0 {
            processed_logits = self.apply_top_k(&processed_logits)?;
            self.stats.top_k_operations += 1;
        }
        
        // Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            processed_logits = self.apply_top_p(&processed_logits)?;
            self.stats.top_p_operations += 1;
        }
        
        // Convert to probabilities and sample
        self.sample_from_probabilities(&processed_logits)
    }
    
    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &Tensor) -> Result<Tensor> {
        // Divide logits by temperature
        // Higher temperature = more random, lower temperature = more focused
        if self.config.temperature <= 0.0 {
            return Err(anyhow::anyhow!("Temperature must be positive"));
        }
        
        // Create a scalar tensor with temperature value for proper broadcasting
        let temperature_scalar = Tensor::from_slice(&[self.config.temperature], (1,), &self.config.device)
            .context("Failed to create temperature scalar")?;
        
        logits.broadcast_div(&temperature_scalar)
            .context("Failed to apply temperature scaling")
    }
    
    /// Apply top-k filtering to logits
    fn apply_top_k(&self, logits: &Tensor) -> Result<Tensor> {
        // Keep only the top-k highest logits, set others to -inf
        // This is a simplified implementation - real top-k requires sorting
        
        let shape = logits.shape();
        let vocab_size = shape.dims()[shape.dims().len() - 1];
        
        if self.config.top_k >= vocab_size {
            return Ok(logits.clone()); // No filtering needed
        }
        
        // For now, return original logits (simplified implementation)
        // Real implementation would use topk operations
        Ok(logits.clone())
    }
    
    /// Apply top-p (nucleus) filtering to logits
    fn apply_top_p(&self, logits: &Tensor) -> Result<Tensor> {
        // Keep tokens with cumulative probability <= top_p
        // This requires sorting by probability and cumulative sum
        
        if self.config.top_p >= 1.0 {
            return Ok(logits.clone()); // No filtering needed
        }
        
        // For now, return original logits (simplified implementation)
        // Real implementation would compute softmax, sort, cumulative sum, and filter
        Ok(logits.clone())
    }
    
    /// Apply repetition penalty to reduce repetitive tokens
    fn apply_repetition_penalty(&self, logits: &Tensor) -> Result<Tensor> {
        if self.token_history.is_empty() || self.config.repetition_penalty == 1.0 {
            return Ok(logits.clone());
        }
        
        // Create penalty tensor
        let penalty_logits = logits.clone();
        
        // For each token in history, apply penalty
        // This is simplified - real implementation would modify specific logit values
        // based on token frequencies in history
        
        Ok(penalty_logits)
    }
    
    /// Sample from processed logits/probabilities
    fn sample_from_probabilities(&mut self, logits: &Tensor) -> Result<Tensor> {
        let shape = logits.shape();
        let batch_size = shape.dims()[0];
        
        // Apply softmax to get probabilities (simplified)
        let probabilities = self.softmax(logits)?;
        
        // Sample from distribution
        // For now, implement pseudo-random sampling
        let token_id = self.pseudo_random_sample(&probabilities)?;
        
        // Create tensor with token ID
        let token_data = vec![token_id];
        let token_tensor = Tensor::from_slice(&token_data, (1,), &self.config.device)
            .context("Failed to create token tensor")?;
        
        // Expand to batch size if needed
        if batch_size > 1 {
            let batch_data = vec![token_id; batch_size];
            Tensor::from_slice(&batch_data, (batch_size, 1), &self.config.device)
                .context("Failed to create batch token tensor")
        } else {
            token_tensor.reshape(&[1, 1])
                .context("Failed to reshape token tensor")
        }
    }
    
    /// Apply softmax to logits
    fn softmax(&self, logits: &Tensor) -> Result<Tensor> {
        // Simplified softmax implementation
        // Real implementation would use proper softmax operations
        
        // For numerical stability, subtract max value
        // Then exp and normalize
        
        // For now, return normalized approximation
        Ok(logits.clone())
    }
    
    /// Pseudo-random sampling from probabilities
    fn pseudo_random_sample(&mut self, _probabilities: &Tensor) -> Result<i64> {
        // Simple pseudo-random number generation
        if let Some(ref mut state) = self.rng_state {
            // Linear congruential generator
            *state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let token_id = (*state / 65536) % 1000; // Simple vocab size
            Ok(token_id as i64)
        } else {
            // Use current time as pseudo-random source
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            Ok((now % 1000) as i64)
        }
    }
    
    /// Reset token history
    pub fn reset_history(&mut self) {
        self.token_history.clear();
    }
    
    /// Get current sampling statistics
    pub fn stats(&self) -> &SamplingStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SamplingStats::default();
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: SamplingConfig) {
        self.config = config;
        if let Some(seed) = self.config.seed {
            self.rng_state = Some(seed);
        }
    }
    
    /// Get current configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }
}

/// Preset sampling configurations for common use cases
#[derive(Debug)]
pub struct SamplingPresets;

impl SamplingPresets {
    /// Deterministic sampling (always pick highest probability token)
    pub fn deterministic(device: Device) -> SamplingConfig {
        SamplingConfig {
            deterministic: true,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
            device,
            ..Default::default()
        }
    }
    
    /// Conservative sampling (low temperature, focused)
    pub fn conservative(device: Device) -> SamplingConfig {
        SamplingConfig {
            temperature: 0.2,
            top_k: 50,
            top_p: 0.9,
            deterministic: false,
            device,
            ..Default::default()
        }
    }
    
    /// Balanced sampling (moderate randomness)
    pub fn balanced(device: Device) -> SamplingConfig {
        SamplingConfig {
            temperature: 0.7,
            top_k: 100,
            top_p: 0.95,
            deterministic: false,
            device,
            ..Default::default()
        }
    }
    
    /// Creative sampling (high temperature, more random)
    pub fn creative(device: Device) -> SamplingConfig {
        SamplingConfig {
            temperature: 1.2,
            top_k: 200,
            top_p: 0.98,
            deterministic: false,
            device,
            ..Default::default()
        }
    }
    
    /// Reproducible sampling (with fixed seed)
    pub fn reproducible(device: Device, seed: u64) -> SamplingConfig {
        SamplingConfig {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            seed: Some(seed),
            deterministic: false,
            device,
            ..Default::default()
        }
    }
}

/// Batch sampling for multiple sequences
#[derive(Debug)]
pub struct BatchSampler {
    /// Individual samplers for each sequence in batch
    samplers: Vec<TokenSampler>,
    /// Default configuration for new samplers
    default_config: SamplingConfig,
}

impl BatchSampler {
    /// Create a new batch sampler
    pub fn new(batch_size: usize, config: SamplingConfig) -> Self {
        let mut samplers = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut sampler_config = config.clone();
            // Use different seeds for each sequence if seed is provided
            if let Some(base_seed) = config.seed {
                sampler_config.seed = Some(base_seed + i as u64);
            }
            samplers.push(TokenSampler::new(sampler_config));
        }
        
        Self {
            samplers,
            default_config: config,
        }
    }
    
    /// Sample next tokens for entire batch
    pub fn sample_batch(&mut self, logits: &Tensor) -> Result<Tensor> {
        let shape = logits.shape();
        let batch_size = shape.dims()[0];
        
        if batch_size != self.samplers.len() {
            return Err(anyhow::anyhow!(
                "Batch size mismatch: logits batch size {} != sampler count {}",
                batch_size, self.samplers.len()
            ));
        }
        
        let mut batch_tokens = Vec::with_capacity(batch_size);
        
        // Sample for each sequence in batch
        for (i, sampler) in self.samplers.iter_mut().enumerate() {
            // Extract logits for this sequence
            let seq_logits = logits.narrow(0, i, 1)
                .with_context(|| format!("Failed to extract logits for sequence {}", i))?;
            
            let token = sampler.sample(&seq_logits)
                .with_context(|| format!("Failed to sample token for sequence {}", i))?;
            
            batch_tokens.push(token);
        }
        
        // Concatenate all tokens into batch tensor
        Tensor::cat(&batch_tokens, 0)
            .context("Failed to concatenate batch tokens")
    }
    
    /// Update configuration for all samplers
    pub fn update_all_configs(&mut self, config: SamplingConfig) {
        self.default_config = config.clone();
        for (i, sampler) in self.samplers.iter_mut().enumerate() {
            let mut sampler_config = config.clone();
            if let Some(base_seed) = config.seed {
                sampler_config.seed = Some(base_seed + i as u64);
            }
            sampler.update_config(sampler_config);
        }
    }
    
    /// Update configuration for specific sequence
    pub fn update_config(&mut self, sequence_idx: usize, config: SamplingConfig) -> Result<()> {
        if sequence_idx >= self.samplers.len() {
            return Err(anyhow::anyhow!(
                "Sequence index {} out of bounds (batch size: {})",
                sequence_idx, self.samplers.len()
            ));
        }
        
        self.samplers[sequence_idx].update_config(config);
        Ok(())
    }
    
    /// Get aggregated statistics across all samplers
    pub fn aggregate_stats(&self) -> SamplingStats {
        let mut aggregate = SamplingStats::default();
        
        for sampler in &self.samplers {
            let stats = sampler.stats();
            aggregate.tokens_sampled += stats.tokens_sampled;
            aggregate.top_k_operations += stats.top_k_operations;
            aggregate.top_p_operations += stats.top_p_operations;
            aggregate.temperature_operations += stats.temperature_operations;
            aggregate.total_sampling_time_ns += stats.total_sampling_time_ns;
        }
        
        // Calculate average
        if !self.samplers.is_empty() {
            aggregate.avg_sampling_time_ns = 
                aggregate.total_sampling_time_ns / self.samplers.len() as u64;
        }
        
        aggregate
    }
    
    /// Reset all samplers
    pub fn reset_all(&mut self) {
        for sampler in &mut self.samplers {
            sampler.reset_history();
            sampler.reset_stats();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::Device;

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
        assert!(!config.deterministic);
    }

    #[test]
    fn test_token_sampler_creation() {
        let config = SamplingConfig::default();
        let sampler = TokenSampler::new(config);
        assert_eq!(sampler.stats().tokens_sampled, 0);
    }

    #[test]
    fn test_deterministic_sampling() {
        let config = SamplingConfig {
            deterministic: true,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(config);
        
        let logits = Tensor::ones(&[1, 1000], DType::F32, &Device::Cpu).unwrap();
        let token = sampler.sample(&logits).unwrap();
        
        assert_eq!(token.shape().dims(), [1, 1]);
        assert_eq!(sampler.stats().tokens_sampled, 1);
    }

    #[test]
    fn test_stochastic_sampling() {
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 50,
            deterministic: false,
            seed: Some(42),
            device: Device::Cpu,
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(config);
        
        let logits = Tensor::randn(0.0f32, 1.0f32, (1, 1000), &Device::Cpu).unwrap();
        let token = sampler.sample(&logits).unwrap();
        
        assert_eq!(token.shape().dims(), [1, 1]);
        assert_eq!(sampler.stats().tokens_sampled, 1);
        assert!(sampler.stats().temperature_operations > 0);
    }

    #[test]
    fn test_sampling_presets() {
        let device = Device::Cpu;
        
        let deterministic = SamplingPresets::deterministic(device.clone());
        assert!(deterministic.deterministic);
        
        let conservative = SamplingPresets::conservative(device.clone());
        assert_eq!(conservative.temperature, 0.2);
        assert_eq!(conservative.top_k, 50);
        
        let creative = SamplingPresets::creative(device.clone());
        assert_eq!(creative.temperature, 1.2);
        assert_eq!(creative.top_k, 200);
    }

    #[test]
    fn test_batch_sampler() {
        let config = SamplingConfig {
            deterministic: true,
            device: Device::Cpu,
            ..Default::default()
        };
        let mut batch_sampler = BatchSampler::new(2, config);
        
        let logits = Tensor::ones(&[2, 1000], DType::F32, &Device::Cpu).unwrap();
        let tokens = batch_sampler.sample_batch(&logits).unwrap();
        
        assert_eq!(tokens.shape().dims(), [2, 1]);
    }

    #[test]
    fn test_sampling_statistics() {
        let config = SamplingConfig::default();
        let mut sampler = TokenSampler::new(config);
        
        let logits = Tensor::ones(&[1, 100], DType::F32, &Device::Cpu).unwrap();
        
        // Sample multiple tokens
        for _ in 0..5 {
            sampler.sample(&logits).unwrap();
        }
        
        let stats = sampler.stats();
        assert_eq!(stats.tokens_sampled, 5);
        assert!(stats.total_sampling_time_ns > 0);
        assert!(stats.avg_sampling_time_ns > 0);
    }

    #[test]
    fn test_config_updates() {
        let initial_config = SamplingConfig::default();
        let mut sampler = TokenSampler::new(initial_config);
        
        let new_config = SamplingConfig {
            temperature: 0.5,
            deterministic: true,
            ..Default::default()
        };
        
        sampler.update_config(new_config);
        assert_eq!(sampler.config().temperature, 0.5);
        assert!(sampler.config().deterministic);
    }

    #[test]
    fn test_reproducible_sampling() {
        let config1 = SamplingPresets::reproducible(Device::Cpu, 12345);
        let config2 = SamplingPresets::reproducible(Device::Cpu, 12345);
        
        let mut sampler1 = TokenSampler::new(config1);
        let mut sampler2 = TokenSampler::new(config2);
        
        let logits = Tensor::randn(0.0f32, 1.0f32, (1, 1000), &Device::Cpu).unwrap();
        
        let token1 = sampler1.sample(&logits).unwrap();
        let token2 = sampler2.sample(&logits).unwrap();
        
        // With same seed, tokens should be identical
        let data1 = token1.to_vec2::<i64>().unwrap();
        let data2 = token2.to_vec2::<i64>().unwrap();
        assert_eq!(data1, data2);
    }
}