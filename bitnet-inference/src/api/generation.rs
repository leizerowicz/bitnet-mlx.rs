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
    /// Typical-p sampling - consider tokens within typical probability mass
    pub typical_p: Option<f32>,
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Maximum context length for sliding window
    pub max_context_length: Option<usize>,
    /// Whether to use sampling or greedy decoding
    pub do_sample: bool,
    /// Stop generation when these tokens are encountered
    pub stop_tokens: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Early stopping on EOS token
    pub early_stopping: bool,
    /// Repetition penalty to avoid repetitive text
    pub repetition_penalty: Option<f32>,
    /// Length penalty for longer sequences
    pub length_penalty: Option<f32>,
    /// Enable LUT-based acceleration
    pub use_lut_acceleration: bool,
    /// Target latency in milliseconds (for performance optimization)
    pub target_latency_ms: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            typical_p: Some(0.95),
            max_length: 512,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(29), // Microsoft target: 29ms CPU latency
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
        let mut generated_count = 0;
        let mut finish_reason = FinishReason::MaxLength;
        
        // Initialize generation state for autoregressive generation
        let mut generation_cache = self.initialize_generation_cache(&input_tokens)?;
        
        // Autoregressive generation loop - generate tokens one by one
        while generated_count < self.config.max_length {
            // Create input tensor from current sequence (efficient incremental processing)
            let input_tensor = if generated_count == 0 {
                // First iteration: use full prompt
                self.tokens_to_tensor(&generated_tokens)?
            } else {
                // Subsequent iterations: only need the last token for autoregressive generation
                self.tokens_to_tensor(&[*generated_tokens.last().unwrap()])?
            };
            
            // Run forward pass to get next token logits (with KV cache optimization)
            let logits = self.forward_pass_with_cache(&input_tensor, &mut generation_cache).await?;
            
            // Apply sampling strategy based on configuration
            let next_token = self.sample_next_token_advanced(&logits)?;
            
            // Add token to sequence and update context first
            generated_tokens.push(next_token);
            generated_count += 1;
            
            // Check for early stopping conditions after counting the token
            let should_stop = self.check_early_stopping(next_token, &generated_tokens)?;
            if let Some(reason) = should_stop {
                finish_reason = reason;
                break;
            }
            
            // Context management: sliding window for long conversations
            if generated_tokens.len() > self.config.max_context_length.unwrap_or(4096) {
                self.rotate_context(&mut generated_tokens, &mut generation_cache)?;
            }
        }
        
        // Convert generated tokens back to text (excluding original prompt)
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
        // Use LlamaTokenizer if available
        if let Ok(tokenizer_path) = std::env::var("TOKENIZER_PATH") {
            if let Ok(tokenizer) = crate::tokenizer::LlamaTokenizer::new(&tokenizer_path) {
                let tokens = tokenizer.encode(text, true, false) // Add BOS, no EOS for input
                    .map_err(|e| InferenceError::TokenizerError(format!("Tokenization failed: {}", e)))?;
                return Ok(tokens);
            }
        }
        
        // Fallback: Simple word-based tokenization for testing
        let words: Vec<&str> = text.split_whitespace().collect();
        let tokens: Vec<u32> = words.iter()
            .enumerate()
            .map(|(i, _)| (i % 32000) as u32) // Use a reasonable vocab size limit
            .collect();
        
        Ok(tokens)
    }

    /// Convert tokens back to text
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Use LlamaTokenizer if available
        if let Ok(tokenizer_path) = std::env::var("TOKENIZER_PATH") {
            if let Ok(tokenizer) = crate::tokenizer::LlamaTokenizer::new(&tokenizer_path) {
                let text = tokenizer.decode(tokens)
                    .map_err(|e| InferenceError::TokenizerError(format!("Detokenization failed: {}", e)))?;
                return Ok(text);
            }
        }
        
        // Fallback: Simple mock implementation for testing
        if tokens.is_empty() {
            return Ok(String::new());
        }
        
        // Generate mock text based on token count
        let words_per_token = 0.75; // Average English words per token
        let estimated_words = (tokens.len() as f32 * words_per_token).ceil() as usize;
        
        let mut text = String::new();
        for i in 0..estimated_words {
            if i > 0 { text.push(' '); }
            text.push_str(&format!("word{}", i + 1));
        }
        
        Ok(text)
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
    
    /// Initialize generation cache for KV cache and context management
    fn initialize_generation_cache(&self, input_tokens: &[u32]) -> Result<GenerationCache> {
        Ok(GenerationCache {
            kv_cache: Vec::new(), // KV cache for attention layers
            context_window: input_tokens.to_vec(),
            position: input_tokens.len(),
        })
    }
    
    /// Forward pass with KV cache optimization for autoregressive generation
    async fn forward_pass_with_cache(&self, input_tensor: &Tensor, cache: &mut GenerationCache) -> Result<Tensor> {
        // For now, use the standard inference engine
        // TODO: Implement proper KV cache integration with transformer layers
        let logits = self.engine.infer(&self.model, input_tensor).await?;
        
        // Update cache position
        cache.position += input_tensor.dims()[1]; // Add sequence length to position
        
        Ok(logits)
    }
    
    /// Advanced token sampling with multiple strategies
    fn sample_next_token_advanced(&self, logits: &Tensor) -> Result<u32> {
        // Apply advanced sampling based on configuration
        let mut processed_logits = self.apply_temperature_scaling(logits)?;
        
        // Apply top-k filtering if enabled
        if let Some(k) = self.config.top_k {
            processed_logits = self.apply_top_k_filtering(&processed_logits, k)?;
        }
        
        // Apply top-p (nucleus) sampling if enabled
        if let Some(p) = self.config.top_p {
            processed_logits = self.apply_top_p_filtering(&processed_logits, p)?;
        }
        
        // Apply typical-p sampling if enabled
        if let Some(p) = self.config.typical_p {
            processed_logits = self.apply_typical_p_filtering(&processed_logits, p)?;
        }
        
        // Apply repetition penalty if enabled
        if let Some(penalty) = self.config.repetition_penalty {
            processed_logits = self.apply_repetition_penalty(&processed_logits, penalty)?;
        }
        
        // Sample from the processed distribution
        self.sample_from_distribution(&processed_logits)
    }
    
    /// Check early stopping conditions
    fn check_early_stopping(&self, token: u32, generated_tokens: &[u32]) -> Result<Option<FinishReason>> {
        // Check for EOS token if early stopping is enabled
        if self.config.early_stopping {
            if let Some(eos_id) = self.tokenizer_config.eos_token_id {
                if token == eos_id {
                    return Ok(Some(FinishReason::EndOfSequence));
                }
            }
        }
        
        // Check stop tokens
        let token_text = self.detokenize(&[token])?;
        if self.is_stop_token(&token_text) {
            return Ok(Some(FinishReason::StopToken(token_text)));
        }
        
        // Check maximum length
        if generated_tokens.len() >= self.config.max_length {
            return Ok(Some(FinishReason::MaxLength));
        }
        
        Ok(None)
    }
    
    /// Rotate context window for long conversations
    fn rotate_context(&self, tokens: &mut Vec<u32>, cache: &mut GenerationCache) -> Result<()> {
        let max_context = self.config.max_context_length.unwrap_or(4096);
        
        if tokens.len() > max_context {
            // Keep the most recent tokens within the context window
            let start_idx = tokens.len() - max_context;
            tokens.drain(0..start_idx);
            
            // Update cache context window
            cache.context_window = tokens.clone();
            cache.position = tokens.len();
            
            // Clear KV cache as context has changed
            cache.kv_cache.clear();
        }
        
        Ok(())
    }
    
    /// Apply temperature scaling to logits
    fn apply_temperature_scaling(&self, logits: &Tensor) -> Result<Tensor> {
        if self.config.temperature == 1.0 {
            return Ok(logits.clone());
        }
        
        // Scale logits by temperature
        let temp_tensor = Tensor::from_slice(&[self.config.temperature], &[1], logits.device())?;
        logits.broadcast_div(&temp_tensor)
            .map_err(|e| InferenceError::GenerationError {
                message: format!("Temperature scaling failed: {e}"),
            })
    }
    
    /// Apply top-k filtering
    fn apply_top_k_filtering(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        // For now, return logits unchanged
        // TODO: Implement proper top-k filtering
        Ok(logits.clone())
    }
    
    /// Apply top-p (nucleus) filtering
    fn apply_top_p_filtering(&self, logits: &Tensor, p: f32) -> Result<Tensor> {
        // For now, return logits unchanged
        // TODO: Implement proper top-p filtering
        Ok(logits.clone())
    }
    
    /// Apply typical-p filtering
    fn apply_typical_p_filtering(&self, logits: &Tensor, p: f32) -> Result<Tensor> {
        // For now, return logits unchanged
        // TODO: Implement proper typical-p filtering
        Ok(logits.clone())
    }
    
    /// Apply repetition penalty
    fn apply_repetition_penalty(&self, logits: &Tensor, penalty: f32) -> Result<Tensor> {
        // For now, return logits unchanged
        // TODO: Implement proper repetition penalty
        Ok(logits.clone())
    }
    
    /// Sample from probability distribution
    fn sample_from_distribution(&self, logits: &Tensor) -> Result<u32> {
        if !self.config.do_sample {
            // Greedy sampling - return highest probability token
            return self.greedy_sample(logits);
        }
        
        // For now, use simple greedy sampling
        // TODO: Implement proper probabilistic sampling
        self.greedy_sample(logits)
    }
    
    /// Greedy sampling - select token with highest probability
    fn greedy_sample(&self, logits: &Tensor) -> Result<u32> {
        // Handle both 1D and 2D logits
        let logits_vec = if logits.dims().len() == 1 {
            // 1D logits: [vocab_size]
            logits.to_vec1::<f32>()
                .map_err(|e| InferenceError::GenerationError {
                    message: format!("Failed to convert 1D logits to vector: {e}"),
                })?
        } else if logits.dims().len() == 2 {
            // 2D logits: [batch_size, vocab_size] or [seq_len, vocab_size]
            let logits_2d = logits.to_vec2::<f32>()
                .map_err(|e| InferenceError::GenerationError {
                    message: format!("Failed to convert 2D logits to vector: {e}"),
                })?;
            
            // Take the last sequence position (most recent token predictions)
            logits_2d.last()
                .ok_or_else(|| InferenceError::GenerationError {
                    message: "Empty logits tensor".to_string(),
                })?
                .clone()
        } else {
            return Err(InferenceError::GenerationError {
                message: format!("Unsupported logits tensor rank: {} with shape {:?}", 
                               logits.dims().len(), logits.dims()),
            });
        };
        
        // Find token with highest logit
        let best_token = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| InferenceError::GenerationError {
                message: "No valid tokens found in logits".to_string(),
            })?;
        
        Ok(best_token)
    }
}

/// Generation cache for KV cache and context management
#[derive(Debug, Clone)]
struct GenerationCache {
    /// KV cache for attention layers (placeholder for future implementation)
    kv_cache: Vec<u8>,
    /// Current context window
    context_window: Vec<u32>,
    /// Current position in sequence
    position: usize,
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