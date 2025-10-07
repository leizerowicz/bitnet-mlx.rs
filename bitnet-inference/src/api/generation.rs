//! Text generation interface for BitNet models.

use crate::{Result, InferenceError};
use crate::api::InferenceEngine;
use crate::bitnet_config::{BitNetModelConfig, TokenizerConfig};
use bitnet_core::{Tensor, Device};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Configuration for text generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature for sampling (higher = more random)
    pub temperature: f32,
    /// Top-k sampling - consider only top k tokens
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling - consider tokens with cumulative probability up to p
    pub top_p: Option<f32>,
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Whether to use sampling or greedy decoding
    pub do_sample: bool,
    /// Stop generation when these tokens are encountered
    pub stop_tokens: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            max_length: 512,
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
        }
    }
}

/// Text generator for BitNet models
pub struct TextGenerator {
    engine: Arc<InferenceEngine>,
    model: Arc<crate::engine::Model>,
    config: GenerationConfig,
    tokenizer_config: TokenizerConfig,
}

/// Result of text generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub token_count: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Whether generation stopped due to max length or stop token
    pub finished_reason: FinishReason,
}

/// Reason why generation finished
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    /// Reached maximum length
    MaxLength,
    /// Encountered stop token
    StopToken(String),
    /// Reached end-of-sequence token
    EndOfSequence,
    /// Error during generation
    Error(String),
}

impl TextGenerator {
    /// Create a new text generator
    pub fn new(
        engine: Arc<InferenceEngine>,
        model: Arc<crate::engine::Model>,
        config: GenerationConfig,
        tokenizer_config: TokenizerConfig,
    ) -> Self {
        Self {
            engine,
            model,
            config,
            tokenizer_config,
        }
    }

    /// Generate text from a prompt
    pub async fn generate(&self, prompt: &str) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();
        
        // Tokenize input prompt
        let input_tokens = self.tokenize(prompt)?;
        let mut generated_tokens = input_tokens.clone();
        
        // Generate tokens one by one
        let mut generated_count = 0;
        let mut finish_reason = FinishReason::MaxLength;
        
        while generated_count < self.config.max_length {
            // Create input tensor from current tokens
            let input_tensor = self.tokens_to_tensor(&generated_tokens)?;
            
            // Run inference to get next token logits
            let logits = self.engine.infer(&self.model, &input_tensor).await?;
            
            // Sample next token
            let next_token = self.sample_next_token(&logits)?;
            
            // Check if it's a stop token
            let token_text = self.detokenize(&[next_token])?;
            if self.is_stop_token(&token_text) {
                finish_reason = FinishReason::StopToken(token_text);
                break;
            }
            
            // Check for EOS token
            if let Some(eos_id) = self.tokenizer_config.eos_token_id {
                if next_token == eos_id {
                    finish_reason = FinishReason::EndOfSequence;
                    break;
                }
            }
            
            // Add token to sequence
            generated_tokens.push(next_token);
            generated_count += 1;
        }
        
        // Convert tokens back to text
        let generated_text = self.detokenize(&generated_tokens[input_tokens.len()..])?;
        
        let generation_time = start_time.elapsed();
        
        Ok(GenerationResult {
            text: generated_text,
            token_count: generated_count,
            generation_time_ms: generation_time.as_millis() as u64,
            finished_reason: finish_reason,
        })
    }

    /// Generate text with streaming output
    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>> + '_> {
        // For now, return an error indicating streaming is not yet implemented
        // TODO: Implement actual streaming generation
        use async_stream::stream;
        
        let stream = stream! {
            yield Err(InferenceError::GenerationError {
                message: "Streaming generation not yet implemented".to_string(),
            });
        };
        
        Ok(stream)
    }

    /// Set generation configuration
    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Tokenize text input
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // TODO: Implement actual tokenization using the tokenizer_config
        // For now, return a simple mock implementation
        Ok(text.split_whitespace()
            .enumerate()
            .map(|(i, _)| i as u32)
            .collect())
    }

    /// Convert tokens back to text
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // TODO: Implement actual detokenization
        // For now, return a simple mock implementation
        Ok(format!("Generated text from {} tokens", tokens.len()))
    }

    /// Convert tokens to input tensor
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        // Create a tensor from token IDs
        let data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        Tensor::from_vec(data, &[1, tokens.len()], &Device::Cpu)
            .map_err(|e| InferenceError::TensorError {
                message: format!("Failed to create token tensor: {e}"),
            })
    }

    /// Sample next token from logits
    fn sample_next_token(&self, logits: &Tensor) -> Result<u32> {
        // TODO: Implement proper sampling based on generation config
        // For now, return a simple mock implementation
        
        // Get tensor dimensions
        let dims = logits.dims();
        if dims.is_empty() {
            return Err(InferenceError::GenerationError {
                message: "Empty logits tensor dimensions".to_string(),
            });
        }

        // Simple greedy decoding for now - just return a mock token
        // TODO: Implement actual logits processing and sampling
        Ok(1) // Return a simple mock token ID
    }

    /// Check if token text is a stop token
    fn is_stop_token(&self, token_text: &str) -> bool {
        self.config.stop_tokens.iter().any(|stop| token_text.contains(stop))
    }
}

/// Builder for creating text generators with fluent API
pub struct TextGeneratorBuilder {
    engine: Option<Arc<InferenceEngine>>,
    model: Option<Arc<crate::engine::Model>>,
    config: GenerationConfig,
}

impl TextGeneratorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            engine: None,
            model: None,
            config: GenerationConfig::default(),
        }
    }

    /// Set the inference engine
    pub fn with_engine(mut self, engine: Arc<InferenceEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Set the model
    pub fn with_model(mut self, model: Arc<crate::engine::Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set generation configuration
    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: Option<f32>) -> Self {
        self.config.top_p = top_p;
        self
    }

    /// Set maximum generation length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.config.max_length = max_length;
        self
    }

    /// Set stop tokens
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<String>) -> Self {
        self.config.stop_tokens = stop_tokens;
        self
    }

    /// Build the text generator
    pub fn build(self, tokenizer_config: TokenizerConfig) -> Result<TextGenerator> {
        let engine = self.engine.ok_or_else(|| InferenceError::ConfigurationError {
            message: "Inference engine is required".to_string(),
        })?;

        let model = self.model.ok_or_else(|| InferenceError::ConfigurationError {
            message: "Model is required".to_string(),
        })?;

        Ok(TextGenerator::new(engine, model, self.config, tokenizer_config))
    }
}

impl Default for TextGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, Some(50));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.max_length, 512);
        assert!(config.do_sample);
        assert!(config.stop_tokens.contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn test_text_generator_builder() {
        let builder = TextGeneratorBuilder::new()
            .with_temperature(0.8)
            .with_top_k(Some(40))
            .with_max_length(256);

        assert_eq!(builder.config.temperature, 0.8);
        assert_eq!(builder.config.top_k, Some(40));
        assert_eq!(builder.config.max_length, 256);
    }

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::MaxLength, FinishReason::MaxLength);
        assert_eq!(
            FinishReason::StopToken("</s>".to_string()),
            FinishReason::StopToken("</s>".to_string())
        );
        assert_ne!(FinishReason::MaxLength, FinishReason::EndOfSequence);
    }
}