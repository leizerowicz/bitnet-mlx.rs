//! Model Execution Interface for Task 2.1.19
//! 
//! This module provides the user-facing API for BitNet model execution,
//! building on the transformer forward pass implemented in Task 2.1.18.

use crate::{Result, InferenceError};
use crate::api::{InferenceEngine, GenerationConfig, GenerationResult, FinishReason};
use crate::api::sampling::TokenSampler;
use crate::engine::{Model, InferenceBackend, InferenceContext};
use crate::tokenizer::{LlamaTokenizer, Dialog, Message, Role};
use bitnet_core::{Tensor, Device};
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use futures::stream::BoxStream;

/// Token emitted during streaming generation
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// Token ID
    pub token_id: u32,
    /// Text representation of the token
    pub text: String,
    /// Whether this is the final token in the sequence
    pub is_final: bool,
    /// Number of tokens generated so far
    pub generated_count: usize,
}

/// User-facing model interface for text generation and inference
pub struct BitNetModel {
    /// Core model structure
    model: Arc<Model>,
    /// Inference engine for computation
    engine: Arc<InferenceEngine>,
    /// Tokenizer for text processing
    tokenizer: Arc<LlamaTokenizer>,
    /// Model configuration
    config: ModelExecutionConfig,
    /// Performance metrics
    metrics: Arc<std::sync::Mutex<ModelMetrics>>,
    /// Token sampler for advanced generation
    sampler: Arc<std::sync::Mutex<TokenSampler>>,
}

/// Configuration for model execution
#[derive(Debug, Clone)]
pub struct ModelExecutionConfig {
    /// Device to run inference on
    pub device: Device,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Default generation parameters
    pub default_generation: GenerationConfig,
    /// Enable performance monitoring
    pub enable_metrics: bool,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
}

impl Default for ModelExecutionConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            max_sequence_length: 4096,
            default_generation: GenerationConfig::default(),
            enable_metrics: true,
            memory_optimization: MemoryOptimization::Balanced,
        }
    }
}

/// Memory optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOptimization {
    /// Minimal memory usage, slower inference
    Conservative,
    /// Balanced memory/speed tradeoff
    Balanced,
    /// High memory usage, faster inference
    Aggressive,
}

/// Performance metrics for model execution
#[derive(Debug, Clone, Default)]
pub struct ModelMetrics {
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: u64,
    /// Number of generation requests
    pub generation_requests: u64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Last generation latency
    pub last_generation_latency_ms: u64,
}

impl ModelMetrics {
    /// Update metrics after a generation
    pub fn update_after_generation(&mut self, tokens_generated: u64, time_ms: u64) {
        self.total_tokens_generated += tokens_generated;
        self.total_inference_time_ms += time_ms;
        self.generation_requests += 1;
        self.last_generation_latency_ms = time_ms;
        
        if self.total_inference_time_ms > 0 {
            self.avg_tokens_per_second = 
                (self.total_tokens_generated as f64 * 1000.0) / self.total_inference_time_ms as f64;
        }
    }
    
    /// Get current performance summary
    pub fn summary(&self) -> String {
        format!(
            "Tokens: {}, Requests: {}, Avg Speed: {:.1} tokens/sec, Last Latency: {}ms",
            self.total_tokens_generated,
            self.generation_requests,
            self.avg_tokens_per_second,
            self.last_generation_latency_ms
        )
    }
}

impl BitNetModel {
    /// Create a new BitNet model instance
    pub async fn new(
        model: Arc<Model>,
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<LlamaTokenizer>,
        config: ModelExecutionConfig,
    ) -> Result<Self> {
        let sampler = Arc::new(std::sync::Mutex::new(TokenSampler::new(config.default_generation.seed)));
        
        Ok(Self {
            model,
            engine,
            tokenizer,
            config,
            metrics: Arc::new(std::sync::Mutex::new(ModelMetrics::default())),
            sampler,
        })
    }
    
    /// Load a BitNet model from GGUF file path
    pub async fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
        config: Option<ModelExecutionConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // Create inference engine
        let engine = Arc::new(InferenceEngine::with_device(config.device.clone()).await?);
        
        // Load model
        let model = engine.load_model(&path).await?;
        
        // Create tokenizer (for now, use a default configuration)
        // TODO: Extract tokenizer config from GGUF metadata
        let tokenizer = Arc::new(LlamaTokenizer::new(&path)?);
        
        Self::new(model, engine, tokenizer, config).await
    }
    
    /// Simple text generation interface
    /// 
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    /// 
    /// # Returns
    /// Generated text as a string
    /// 
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::BitNetModel;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = BitNetModel::load_from_file("model.gguf", None).await?;
    /// let response = model.generate("Hello, world!", 50).await?;
    /// println!("Generated: {}", response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut config = self.config.default_generation.clone();
        config.max_length = max_tokens;
        
        let result = self.generate_with_config(prompt, config).await?;
        Ok(result.text)
    }
    
    /// Generate text with custom configuration
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationResult> {
        let start_time = Instant::now();
        
        // Validate input
        if prompt.is_empty() {
            return Err(InferenceError::input_validation("Prompt cannot be empty"));
        }
        
        if config.max_length == 0 {
            return Err(InferenceError::input_validation("max_length must be greater than 0"));
        }
        
        // Tokenize input
        let input_tokens = self.tokenize_text(prompt)?;
        
        // Check sequence length
        if input_tokens.len() > self.config.max_sequence_length {
            return Err(InferenceError::input_validation(
                format!("Input sequence length {} exceeds maximum {}", 
                        input_tokens.len(), self.config.max_sequence_length)
            ));
        }
        
        // Generate tokens using autoregressive generation
        let generated_tokens = self.generate_tokens(&input_tokens, &config).await?;
        
        // Convert back to text
        let generated_text = self.detokenize_tokens(&generated_tokens)?;
        
        let generation_time = start_time.elapsed();
        let generation_time_ms = generation_time.as_millis() as u64;
        
        // Update metrics
        if self.config.enable_metrics {
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.update_after_generation(generated_tokens.len() as u64, generation_time_ms);
            }
        }
        
        Ok(GenerationResult {
            text: generated_text,
            token_count: generated_tokens.len(),
            generation_time_ms,
            finished_reason: FinishReason::MaxLength, // TODO: Implement proper finish reason detection
        })
    }
    
    /// Generate text with token-by-token streaming
    /// 
    /// This method generates text incrementally, yielding each token as it's generated.
    /// Useful for real-time applications where you want to show partial results.
    /// 
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `config` - Generation configuration
    /// 
    /// # Returns
    /// An async stream of tokens and partial text
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<BoxStream<'static, Result<StreamToken>>> {
        use tokio_stream::StreamExt;
        
        // Validate input
        if prompt.is_empty() {
            return Err(InferenceError::input_validation("Prompt cannot be empty"));
        }
        
        // Tokenize input
        let input_tokens = self.tokenize_text(prompt)?;
        
        // Check sequence length
        if input_tokens.len() > self.config.max_sequence_length {
            return Err(InferenceError::input_validation(
                format!("Input sequence length {} exceeds maximum {}", 
                        input_tokens.len(), self.config.max_sequence_length)
            ));
        }
        
        // Clone references for the stream
        let engine = self.engine.clone();
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let max_length = config.max_length;
        let stop_tokens = config.stop_tokens.clone();
        let max_seq_length = self.config.max_sequence_length;
        let generation_config = config.clone(); // Clone the entire config for the stream
        
        // Create streaming generator
        let stream = async_stream::stream! {
            let mut current_sequence = input_tokens.clone();
            let mut generated_count = 0;
            
            while generated_count < max_length {
                // Create input tensor
                let input_tensor = match tokens_to_tensor_helper(&current_sequence) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };
                
                // Run inference
                let logits = match engine.infer(&model, &input_tensor).await {
                    Ok(logits) => logits,
                    Err(e) => {
                        yield Err(InferenceError::inference(format!("Forward pass failed: {}", e)));
                        break;
                    }
                };
                
                // Sample next token (use the cloned config)
                let next_token = match sample_next_token_helper(&logits, &generation_config) {
                    Ok(token) => token,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };
                
                // Convert token to text
                let token_text = match tokenizer.decode(&[next_token]) {
                    Ok(text) => text,
                    Err(e) => {
                        yield Err(InferenceError::tokenization(format!("Detokenization failed: {}", e)));
                        break;
                    }
                };
                
                // Check for stop conditions
                let should_stop = should_stop_generation_helper(next_token, &tokenizer, &stop_tokens);
                
                yield Ok(StreamToken {
                    token_id: next_token,
                    text: token_text,
                    is_final: should_stop,
                    generated_count: generated_count + 1,
                });
                
                if should_stop {
                    break;
                }
                
                // Add token to sequence
                current_sequence.push(next_token);
                generated_count += 1;
                
                // Prevent sequence from growing too long
                if current_sequence.len() > max_seq_length {
                    break;
                }
            }
        };
        
        Ok(Box::pin(stream))
    }
    
    /// Chat interface with conversation support
    pub async fn chat(&self, messages: Dialog, max_tokens: Option<usize>) -> Result<String> {
        // Format messages into a single prompt using chat template
        let prompt = self.format_chat_prompt(&messages)?;
        
        // Generate response
        let max_tokens = max_tokens.unwrap_or(self.config.default_generation.max_length);
        self.generate(&prompt, max_tokens).await
    }
    
    /// Add a user message and get assistant response
    pub async fn chat_turn(&self, user_message: &str, conversation: &mut Dialog, max_tokens: Option<usize>) -> Result<String> {
        // Add user message to conversation
        conversation.push(Message {
            role: Role::User,
            content: user_message.to_string(),
        });
        
        // Generate assistant response
        let response = self.chat(conversation.clone(), max_tokens).await?;
        
        // Add assistant response to conversation
        conversation.push(Message {
            role: Role::Assistant,
            content: response.clone(),
        });
        
        Ok(response)
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> Result<ModelMetrics> {
        self.metrics.lock()
            .map(|metrics| metrics.clone())
            .map_err(|_| InferenceError::internal("Failed to acquire metrics lock"))
    }
    
    /// Reset performance metrics
    pub fn reset_metrics(&self) -> Result<()> {
        self.metrics.lock()
            .map(|mut metrics| *metrics = ModelMetrics::default())
            .map_err(|_| InferenceError::internal("Failed to acquire metrics lock"))
    }
    
    // Private helper methods
    
    /// Tokenize text input
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false, false)
            .map_err(|e| InferenceError::tokenization(format!("Tokenization failed: {}", e)))
    }
    
    /// Convert tokens back to text
    fn detokenize_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
            .map_err(|e| InferenceError::tokenization(format!("Detokenization failed: {}", e)))
    }
    
    /// Generate tokens using autoregressive generation
    async fn generate_tokens(&self, input_tokens: &[u32], config: &GenerationConfig) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_sequence = input_tokens.to_vec();
        
        for _ in 0..config.max_length {
            // Create input tensor from current sequence
            let input_tensor = self.tokens_to_tensor(&current_sequence)?;
            
            // Run forward pass to get next token logits
            let logits = self.engine.infer(&self.model, &input_tensor).await?;
            
            // Sample next token based on configuration
            let next_token = self.sample_next_token(&logits, config)?;
            
            // Check for stop conditions
            if self.should_stop_generation(next_token, config) {
                break;
            }
            
            // Add token to sequences
            generated_tokens.push(next_token);
            current_sequence.push(next_token);
            
            // Prevent sequence from growing too long
            if current_sequence.len() > self.config.max_sequence_length {
                break;
            }
        }
        
        Ok(generated_tokens)
    }
    
    /// Convert token sequence to tensor for model input
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        // Create tensor with shape [batch_size=1, sequence_length]
        let sequence_length = tokens.len();
        let mut data = Vec::with_capacity(sequence_length);
        
        // Convert u32 tokens to f32 for tensor
        for &token in tokens {
            data.push(token as f32);
        }
        
        Tensor::from_vec(data, &[1, sequence_length], &self.config.device)
            .map_err(|e| InferenceError::tensor_creation(format!("Failed to create input tensor: {}", e)))
    }
    
    /// Sample next token from logits
    fn sample_next_token(&self, logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
        // Validate configuration
        config.validate()?;
        
        // Use the advanced sampler
        let mut sampler = self.sampler.lock()
            .map_err(|_| InferenceError::internal("Failed to acquire sampler lock"))?;
        
        sampler.sample_token(logits, config)
    }
    
    /// Check if generation should stop
    fn should_stop_generation(&self, token: u32, config: &GenerationConfig) -> bool {
        // Check for EOS token
        if let Ok(eos_id) = self.tokenizer.get_eos_id() {
            if token == eos_id {
                return true;
            }
        }
        
        // Check stop tokens (convert token to text and check)
        if let Ok(token_text) = self.detokenize_tokens(&[token]) {
            for stop_token in &config.stop_tokens {
                if token_text.contains(stop_token) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Format chat messages using a simple chat template
    fn format_chat_prompt(&self, messages: &Dialog) -> Result<String> {
        let mut prompt = String::new();
        
        for message in messages {
            match message.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n", message.content));
                }
                Role::User => {
                    prompt.push_str(&format!("User: {}\n", message.content));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n", message.content));
                }
            }
        }
        
        // Add prompt for assistant response
        prompt.push_str("Assistant: ");
        
        Ok(prompt)
    }
}

// Helper functions for streaming (needed to avoid lifetime issues with async streams)

/// Convert token sequence to tensor for model input (helper function)
fn tokens_to_tensor_helper(tokens: &[u32]) -> Result<Tensor> {
    // Create tensor with shape [batch_size=1, sequence_length]
    let sequence_length = tokens.len();
    let mut data = Vec::with_capacity(sequence_length);
    
    // Convert u32 tokens to f32 for tensor
    for &token in tokens {
        data.push(token as f32);
    }
    
    Tensor::from_vec(data, &[1, sequence_length], &Device::Cpu)
        .map_err(|e| InferenceError::tensor_creation(format!("Failed to create input tensor: {}", e)))
}

/// Sample next token from logits (helper function)
fn sample_next_token_helper(logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
    // Create a temporary sampler for streaming (since we can't share the model's sampler)
    let mut sampler = TokenSampler::new(config.seed);
    
    // Validate configuration
    config.validate()?;
    
    sampler.sample_token(logits, config)
}

/// Check if generation should stop (helper function)
fn should_stop_generation_helper(token: u32, tokenizer: &LlamaTokenizer, stop_tokens: &[String]) -> bool {
    // Check for EOS token (using common EOS ID for now)
    if token == 2 { // Common EOS token ID for LLaMA models
        return true;
    }
    
    // Check stop tokens (convert token to text and check)
    if let Ok(token_text) = tokenizer.decode(&[token]) {
        for stop_token in stop_tokens {
            if token_text.contains(stop_token) {
                return true;
            }
        }
    }
    
    false
}

// Extension trait for LlamaTokenizer to add missing methods
trait TokenizerExtensions {
    fn get_eos_id(&self) -> Result<u32>;
}

impl TokenizerExtensions for LlamaTokenizer {
    fn get_eos_id(&self) -> Result<u32> {
        // For now, return a common EOS token ID
        // TODO: Get actual EOS ID from tokenizer configuration
        Ok(2) // Common EOS token ID for LLaMA models
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ModelArchitecture, QuantizationConfig};
    
    fn create_test_model() -> Arc<Model> {
        Arc::new(Model {
            name: "test-bitnet".to_string(),
            version: "1.0".to_string(),
            input_dim: 512,
            output_dim: 32000,
            architecture: ModelArchitecture::BitLinear {
                layers: Vec::new(),
                attention_heads: Some(32),
                hidden_dim: 2048,
            },
            parameter_count: 1_000_000,
            quantization_config: QuantizationConfig::default(),
        })
    }
    
    #[tokio::test]
    async fn test_model_creation() {
        let config = ModelExecutionConfig::default();
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        // For testing, we'll skip tokenizer creation that requires file access
        // TODO: Create proper mock tokenizer for testing
        // let tokenizer = Arc::new(LlamaTokenizer::new("test").unwrap());
        // let bitnet_model = BitNetModel::new(model, engine, tokenizer, config).await;
        // assert!(bitnet_model.is_ok());
    }
    
    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = ModelExecutionConfig {
            enable_metrics: true,
            ..Default::default()
        };
        
        let engine = Arc::new(InferenceEngine::new().await.unwrap());
        let model = create_test_model();
        
        // For testing, we'll test just the metrics functionality
        let metrics = Arc::new(std::sync::Mutex::new(ModelMetrics::default()));
        
        // Check initial metrics
        let metrics_data = metrics.lock().unwrap();
        assert_eq!(metrics_data.total_tokens_generated, 0);
        assert_eq!(metrics_data.generation_requests, 0);
    }
    
    #[test]
    fn test_memory_optimization_levels() {
        let config = ModelExecutionConfig {
            memory_optimization: MemoryOptimization::Conservative,
            ..Default::default()
        };
        
        assert_eq!(config.memory_optimization, MemoryOptimization::Conservative);
    }
    
    #[test]
    fn test_model_metrics_update() {
        let mut metrics = ModelMetrics::default();
        
        // Test metrics update
        metrics.update_after_generation(10, 1000);
        
        assert_eq!(metrics.total_tokens_generated, 10);
        assert_eq!(metrics.generation_requests, 1);
        assert_eq!(metrics.last_generation_latency_ms, 1000);
        assert_eq!(metrics.avg_tokens_per_second, 10.0);
    }
}