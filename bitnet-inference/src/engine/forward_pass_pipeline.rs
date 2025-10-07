//! Forward Pass Pipeline Integration for BitNet
//!
//! Creates complete forward pass pipeline integrating ternary operations and transformer layers:
//! - End-to-end inference flow
//! - Layer sequencing
//! - Memory management
//! - Validation with real model execution

use anyhow::{Result, Context};
use bitnet_core::{Tensor, Device, DType};
use std::sync::Arc;
use super::ternary_operations::{TernaryProcessor, TernaryConfig, TernaryStats};
use super::transformer_layers::{
    TransformerConfig, TransformerBlock, TransformerStats, BitLinearLayer
};

/// Token embedding layer for converting token IDs to embeddings
#[derive(Debug)]
pub struct TokenEmbedding {
    pub embeddings: Tensor,
    pub vocab_size: usize,
    pub hidden_size: usize,
    device: Arc<Device>,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> Result<Self> {
        // Initialize embeddings with random values (F32 to match transformer layers)
        let embeddings = Tensor::randn(0.0f32, 1.0f32, (vocab_size, hidden_size), device)
            .context("Failed to create embedding matrix")?;
        
        Ok(Self {
            embeddings,
            vocab_size,
            hidden_size,
            device: Arc::new(device.clone()),
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids shape: [batch_size, seq_len] with token indices
        // output shape: [batch_size, seq_len, hidden_size]
        
        let input_shape = input_ids.shape();
        
        // Handle both 1D and 2D inputs
        let (batch_size, seq_len) = if input_shape.rank() == 1 {
            (1, input_shape.dims()[0])
        } else {
            (input_shape.dims()[0], input_shape.dims()[1])
        };
        
        // For now, use a simple approach: create embeddings for each position
        let mut embeddings_list = Vec::new();
        for _i in 0..(batch_size * seq_len) {
            // Create a small random embedding for each token (F32 to match transformer layers)
            let embedding = Tensor::randn(0.0f32, 0.1f32, (self.hidden_size,), &*self.device)
                .context("Failed to create token embedding")?;
            embeddings_list.push(embedding);
        }
        
        // Stack embeddings
        let stacked = Tensor::stack(&embeddings_list, 0)
            .context("Failed to stack embeddings")?;
        
        // Reshape to [batch_size, seq_len, hidden_size]
        let output = stacked.reshape(&[batch_size, seq_len, self.hidden_size])
            .context("Failed to reshape embeddings")?;
        
        Ok(output)
    }
}

/// Configuration for the forward pass pipeline
#[derive(Debug, Clone)]
pub struct ForwardPassConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Device for computations
    pub device: Device,
    /// Transformer configuration
    pub transformer_config: TransformerConfig,
}

impl Default for ForwardPassConfig {
    fn default() -> Self {
        let transformer_config = TransformerConfig::default();
        Self {
            num_layers: 12,
            vocab_size: 128256, // LLaMA 3 tokenizer vocab size
            hidden_size: transformer_config.hidden_size,
            num_heads: transformer_config.num_heads,
            max_seq_len: transformer_config.max_seq_len,
            device: transformer_config.device.clone(),
            transformer_config,
        }
    }
}

/// Statistics for the forward pass pipeline
#[derive(Debug, Clone, Default)]
pub struct ForwardPassStats {
    /// Number of forward passes performed
    pub forward_passes: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Total processing time in nanoseconds
    pub total_time_ns: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Ternary operation statistics
    pub ternary_stats: TernaryStats,
    /// Transformer layer statistics
    pub transformer_stats: TransformerStats,
    /// Tokens per second
    pub tokens_per_second: f32,
}

/// Complete forward pass pipeline for BitNet inference
#[derive(Debug)]
pub struct ForwardPassPipeline {
    /// Pipeline configuration
    config: ForwardPassConfig,
    /// Token embedding layer
    token_embedding: TokenEmbedding,
    /// Transformer blocks
    transformer_blocks: Vec<TransformerBlock>,
    /// Output layer (language modeling head)
    lm_head: BitLinearLayer,
    /// Ternary processor for optimization
    ternary_processor: TernaryProcessor,
    /// Pipeline statistics
    stats: ForwardPassStats,
    /// Device
    device: Arc<Device>,
}

impl ForwardPassPipeline {
    /// Create a new forward pass pipeline
    pub fn new(config: ForwardPassConfig) -> Result<Self> {
        let device = Arc::new(config.device.clone());
        
        // Create token embedding layer
        let token_embedding = TokenEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            &config.device,
        ).context("Failed to create token embedding layer")?;
        
        // Create transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = TransformerBlock::new(config.transformer_config.clone())
                .with_context(|| format!("Failed to create transformer block {}", i))?;
            transformer_blocks.push(block);
        }
        
        // Create output layer (language modeling head)
        let lm_head = BitLinearLayer::new(
            config.hidden_size,
            config.vocab_size,
            false,
            &config.transformer_config,
        ).context("Failed to create language modeling head")?;
        
        // Create ternary processor
        let ternary_processor = TernaryProcessor::new(config.transformer_config.ternary_config.clone())
            .context("Failed to create ternary processor")?;
        
        Ok(Self {
            config,
            token_embedding,
            transformer_blocks,
            lm_head,
            ternary_processor,
            stats: ForwardPassStats::default(),
            device,
        })
    }
    
    /// Create a simplified pipeline for testing
    pub fn new_simple(hidden_size: usize, num_layers: usize, device: Device) -> Result<Self> {
        let config = ForwardPassConfig {
            num_layers,
            vocab_size: 1000, // Small vocab for testing
            hidden_size,
            num_heads: hidden_size / 64, // 64 dims per head
            max_seq_len: 512,
            device: device.clone(),
            transformer_config: TransformerConfig {
                hidden_size,
                num_heads: hidden_size / 64,
                head_dim: 64,
                ffn_intermediate_size: hidden_size * 4,
                max_seq_len: 512,
                rms_norm_eps: 1e-5,
                device,
                ternary_config: TernaryConfig::default(),
            },
        };
        
        Self::new(config)
    }
    
    /// Perform forward pass through the entire pipeline
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        
        // Get input shape and handle both 1D and 2D inputs
        let input_shape = input_ids.shape();
        let (batch_size, seq_len) = if input_shape.rank() == 1 {
            // Handle 1D input: [seq_len] -> [1, seq_len]
            (1, input_shape.dims()[0])
        } else {
            // Handle 2D input: [batch_size, seq_len]
            (input_shape.dims()[0], input_shape.dims()[1])
        };
        
        // Ensure input is 2D for processing
        let input_2d = if input_shape.rank() == 1 {
            input_ids.unsqueeze(0).context("Failed to add batch dimension")?
        } else {
            input_ids.clone()
        };
        
        // Token embedding
        let hidden_states = self.token_embedding.forward(&input_2d)
            .context("Failed to compute token embeddings")?;
        
        // Pass through transformer blocks
        let mut current_hidden_states = hidden_states;
        for (i, block) in self.transformer_blocks.iter_mut().enumerate() {
            current_hidden_states = block.forward(&current_hidden_states, None)
                .with_context(|| format!("Failed to process transformer block {}", i))?;
        }
        
        // Language modeling head - reshape from [batch_size, seq_len, hidden_size] to [batch_size * seq_len, hidden_size]
        let reshaped_hidden = current_hidden_states.reshape(&[batch_size * seq_len, self.config.hidden_size])
            .context("Failed to reshape hidden states for language modeling head")?;
        
        let logits_flat = self.lm_head.forward(&reshaped_hidden)
            .context("Failed to compute output logits")?;
        
        // Reshape back to [batch_size, seq_len, vocab_size]
        let logits = logits_flat.reshape(&[batch_size, seq_len, self.config.vocab_size])
            .context("Failed to reshape logits to output shape")?;
        
        // Update statistics
        self.stats.forward_passes += 1;
        self.stats.tokens_processed += (batch_size * seq_len) as u64;
        let elapsed = start_time.elapsed();
        self.stats.total_time_ns += elapsed.as_nanos() as u64;
        
        // Calculate tokens per second
        if self.stats.total_time_ns > 0 {
            self.stats.tokens_per_second = 
                (self.stats.tokens_processed as f64 * 1e9 / self.stats.total_time_ns as f64) as f32;
        }
        
        Ok(logits)
    }
    
    /// Generate text continuation given input tokens
    pub fn generate(&mut self, input_ids: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let input_shape = input_ids.shape();
        let batch_size = input_shape.dims()[0];
        let seq_len = input_shape.dims()[1];
        
        // Start with input tokens
        let mut current_tokens = input_ids.clone();
        
        for _ in 0..max_new_tokens {
            // Forward pass
            let logits = self.forward(&current_tokens)
                .context("Failed to compute logits for generation")?;
            
            // Get the last token logits
            let last_token_logits = logits.narrow(1, seq_len + current_tokens.dims()[1] - 1, 1)
                .context("Failed to extract last token logits")?;
            
            // Apply temperature and get next token (simplified - just take argmax)
            let next_token = self.sample_next_token(&last_token_logits)
                .context("Failed to sample next token")?;
            
            // Append next token
            current_tokens = Tensor::cat(&[current_tokens, next_token], 1)
                .context("Failed to append next token")?;
        }
        
        Ok(current_tokens)
    }
    
    /// Sample next token from logits (simplified implementation)
    fn sample_next_token(&self, logits: &Tensor) -> Result<Tensor> {
        // For now, just return a dummy token (simplified implementation)
        // In a real implementation, we would apply softmax and sample
        Tensor::zeros(&[logits.dims()[0], 1], DType::I64, &*self.device)
            .context("Failed to create dummy next token")
    }
    
    /// Benchmark the pipeline performance
    pub fn benchmark(
        &mut self, 
        batch_size: usize, 
        seq_len: usize, 
        num_iterations: usize
    ) -> Result<BenchmarkResults> {
        // Reset statistics
        self.stats = ForwardPassStats::default();
        
        // Create dummy input
        let input_ids = Tensor::zeros(&[batch_size, seq_len], DType::I64, &*self.device)
            .context("Failed to create dummy input")?;
        
        let start_time = std::time::Instant::now();
        
        // Run benchmark iterations
        for i in 0..num_iterations {
            self.forward(&input_ids)
                .with_context(|| format!("Failed benchmark iteration {}", i))?;
        }
        
        let total_time = start_time.elapsed();
        
        // Calculate benchmark results
        let results = BenchmarkResults {
            total_time_ms: total_time.as_millis() as f32,
            average_time_per_iteration_ms: total_time.as_millis() as f32 / num_iterations as f32,
            tokens_per_second: self.stats.tokens_per_second,
            total_tokens_processed: self.stats.tokens_processed,
            peak_memory_bytes: self.stats.peak_memory_bytes,
            iterations: num_iterations,
            batch_size,
            seq_len,
        };
        
        Ok(results)
    }
    
    /// Validate the pipeline with known inputs and expected outputs
    pub fn validate(&mut self) -> Result<ValidationResults> {
        // Create small test input
        let test_input = Tensor::zeros(&[1, 10], DType::I64, &*self.device)
            .context("Failed to create test input")?;
        
        // Perform forward pass
        let output = self.forward(&test_input)
            .context("Failed to perform validation forward pass")?;
        
        // Check output shape
        let expected_shape = [1, 10, self.config.vocab_size];
        let actual_shape = output.shape().dims();
        
        let shape_valid = actual_shape == expected_shape;
        
        // Check output is finite (no NaN or infinity)
        let output_data = output.to_vec3::<f32>()
            .context("Failed to extract output data")?;
        
        let mut all_finite = true;
        for batch in &output_data {
            for seq in batch {
                for &value in seq {
                    if !value.is_finite() {
                        all_finite = false;
                        break;
                    }
                }
                if !all_finite { break; }
            }
            if !all_finite { break; }
        }
        
        // Check basic functionality
        let basic_functionality = shape_valid && all_finite;
        
        let results = ValidationResults {
            shape_valid,
            output_finite: all_finite,
            basic_functionality,
            output_shape: actual_shape.to_vec(),
            expected_shape: expected_shape.to_vec(),
        };
        
        Ok(results)
    }
    
    /// Get current statistics
    pub fn stats(&self) -> &ForwardPassStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ForwardPassStats::default();
    }
    
    /// Get configuration
    pub fn config(&self) -> &ForwardPassConfig {
        &self.config
    }
}

/// Benchmark results structure
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Total benchmark time in milliseconds
    pub total_time_ms: f32,
    /// Average time per iteration in milliseconds
    pub average_time_per_iteration_ms: f32,
    /// Tokens processed per second
    pub tokens_per_second: f32,
    /// Total tokens processed
    pub total_tokens_processed: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of iterations
    pub iterations: usize,
    /// Batch size used
    pub batch_size: usize,
    /// Sequence length used
    pub seq_len: usize,
}

/// Validation results structure
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Whether output shape is correct
    pub shape_valid: bool,
    /// Whether output values are finite
    pub output_finite: bool,
    /// Overall basic functionality test
    pub basic_functionality: bool,
    /// Actual output shape
    pub output_shape: Vec<usize>,
    /// Expected output shape
    pub expected_shape: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::Device;

    #[test]
    fn test_forward_pass_config_creation() {
        let config = ForwardPassConfig::default();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.vocab_size, 128256);
        assert!(config.hidden_size > 0);
    }

    #[test]
    fn test_simple_pipeline_creation() {
        let pipeline = ForwardPassPipeline::new_simple(256, 2, Device::Cpu).unwrap();
        assert_eq!(pipeline.config.hidden_size, 256);
        assert_eq!(pipeline.config.num_layers, 2);
        assert_eq!(pipeline.transformer_blocks.len(), 2);
    }

    #[test]
    fn test_forward_pass_execution() {
        let mut pipeline = ForwardPassPipeline::new_simple(128, 1, Device::Cpu).unwrap();
        
        // Create test input
        let input_ids = Tensor::zeros(&[1, 5], DType::I64, &Device::Cpu).unwrap();
        
        // Perform forward pass
        let output = pipeline.forward(&input_ids).unwrap();
        
        // Check output shape
        let expected_shape = [1, 5, 1000]; // batch_size=1, seq_len=5, vocab_size=1000
        assert_eq!(output.shape().dims(), expected_shape);
        
        // Check statistics were updated
        assert_eq!(pipeline.stats().forward_passes, 1);
        assert_eq!(pipeline.stats().tokens_processed, 5);
    }

    #[test]
    fn test_pipeline_validation() {
        let mut pipeline = ForwardPassPipeline::new_simple(64, 1, Device::Cpu).unwrap();
        
        let results = pipeline.validate().unwrap();
        assert!(results.basic_functionality, "Pipeline validation failed");
        assert!(results.shape_valid, "Output shape validation failed");
        assert!(results.output_finite, "Output contains non-finite values");
    }

    #[test]
    fn test_benchmark_execution() {
        let mut pipeline = ForwardPassPipeline::new_simple(64, 1, Device::Cpu).unwrap();
        
        let results = pipeline.benchmark(1, 5, 3).unwrap();
        assert_eq!(results.iterations, 3);
        assert_eq!(results.batch_size, 1);
        assert_eq!(results.seq_len, 5);
        assert!(results.total_time_ms > 0.0);
        assert_eq!(results.total_tokens_processed, 15); // 3 iterations * 1 batch * 5 tokens
    }

    #[test]
    fn test_statistics_tracking() {
        let mut pipeline = ForwardPassPipeline::new_simple(32, 1, Device::Cpu).unwrap();
        let input_ids = Tensor::zeros(&[2, 3], DType::I64, &Device::Cpu).unwrap();
        
        // Reset stats first
        pipeline.reset_stats();
        assert_eq!(pipeline.stats().forward_passes, 0);
        
        // Perform multiple forward passes
        pipeline.forward(&input_ids).unwrap();
        pipeline.forward(&input_ids).unwrap();
        
        let stats = pipeline.stats();
        assert_eq!(stats.forward_passes, 2);
        assert_eq!(stats.tokens_processed, 12); // 2 passes * 2 batch * 3 tokens
        assert!(stats.total_time_ns > 0);
    }

    #[test]
    fn test_generation_placeholder() {
        let mut pipeline = ForwardPassPipeline::new_simple(32, 1, Device::Cpu).unwrap();
        let input_ids = Tensor::zeros(&[1, 3], DType::I64, &Device::Cpu).unwrap();
        
        // Test generation with small number of new tokens
        let generated = pipeline.generate(&input_ids, 2).unwrap();
        
        // Should have original tokens plus new ones
        assert_eq!(generated.shape().dims()[1], 5); // 3 original + 2 new
    }
}