use crate::error::{InferenceError, Result};
use crate::tokenizer::{LlamaTokenizer, ChatFormat, Dialog, Message};
use std::collections::VecDeque;
use parking_lot::RwLock;
use std::sync::Arc;

/// Input validation and processing configuration
#[derive(Debug, Clone)]
pub struct InputProcessingConfig {
    /// Maximum context length in tokens (LLaMA 3 default: 4096)
    pub max_context_length: usize,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Buffer size for token management
    pub token_buffer_size: usize,
    /// Enable automatic truncation when input exceeds max context
    pub auto_truncate: bool,
    /// Enable sliding window for very long inputs
    pub sliding_window: bool,
    /// Sliding window overlap tokens
    pub window_overlap: usize,
}

impl Default for InputProcessingConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096, // LLaMA 3 default
            max_batch_size: 32,
            token_buffer_size: 8192, // 2x max context for buffering
            auto_truncate: true,
            sliding_window: false,
            window_overlap: 256,
        }
    }
}

/// Token buffer for efficient memory management
#[derive(Debug)]
pub struct TokenBuffer {
    /// Ring buffer for tokens
    buffer: VecDeque<u32>,
    /// Maximum buffer capacity
    capacity: usize,
    /// Current position in the buffer
    position: usize,
}

impl TokenBuffer {
    /// Create a new token buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    /// Add tokens to the buffer
    pub fn push_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        for &token in tokens {
            if self.buffer.len() >= self.capacity {
                // Remove oldest token if at capacity
                self.buffer.pop_front();
            }
            self.buffer.push_back(token);
        }
        Ok(())
    }

    /// Get tokens from buffer with specified length
    pub fn get_tokens(&self, start: usize, length: usize) -> Vec<u32> {
        self.buffer
            .iter()
            .skip(start)
            .take(length)
            .copied()
            .collect()
    }

    /// Get all tokens in buffer
    pub fn get_all_tokens(&self) -> Vec<u32> {
        self.buffer.iter().copied().collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.position = 0;
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.buffer.len())
    }
}

/// Validated input ready for processing
#[derive(Debug, Clone)]
pub struct ValidatedInput {
    /// Tokenized input
    pub tokens: Vec<u32>,
    /// Original text (if available)
    pub text: Option<String>,
    /// Input type
    pub input_type: InputType,
    /// Validation metadata
    pub metadata: InputMetadata,
}

/// Type of input being processed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputType {
    /// Single text prompt
    SinglePrompt,
    /// Multi-turn dialog
    Dialog,
    /// Completion request
    Completion,
    /// Chat completion
    ChatCompletion,
}

/// Metadata about processed input
#[derive(Debug, Clone)]
pub struct InputMetadata {
    /// Original token count before processing
    pub original_token_count: usize,
    /// Final token count after processing
    pub final_token_count: usize,
    /// Whether input was truncated
    pub was_truncated: bool,
    /// Truncation position (if applicable)
    pub truncation_position: Option<usize>,
    /// Processing timestamp
    pub processed_at: std::time::Instant,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
}

/// Batch of inputs for processing
#[derive(Debug)]
pub struct InputBatch {
    /// Individual inputs in the batch
    pub inputs: Vec<ValidatedInput>,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Metadata for input batches
#[derive(Debug)]
pub struct BatchMetadata {
    /// Batch ID for tracking
    pub batch_id: uuid::Uuid,
    /// Maximum token length in batch
    pub max_token_length: usize,
    /// Total tokens in batch
    pub total_tokens: usize,
    /// Batch creation time
    pub created_at: std::time::Instant,
    /// Estimated processing time
    pub estimated_processing_time: std::time::Duration,
}

/// Input processing and validation engine
#[derive(Debug)]
pub struct InputProcessor {
    /// Configuration
    config: InputProcessingConfig,
    /// Tokenizer for processing
    tokenizer: Arc<LlamaTokenizer>,
    /// Chat format handler
    chat_format: ChatFormat,
    /// Token buffer pool
    buffer_pool: Arc<RwLock<Vec<TokenBuffer>>>,
    /// Processing statistics
    stats: Arc<RwLock<ProcessingStats>>,
}

/// Processing statistics
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    /// Total inputs processed
    pub total_processed: u64,
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total truncations
    pub total_truncations: u64,
    /// Average processing time
    pub average_processing_time: std::time::Duration,
    /// Memory usage statistics
    pub peak_memory_usage: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
}

impl InputProcessor {
    /// Create a new input processor
    pub fn new(tokenizer: LlamaTokenizer, config: InputProcessingConfig) -> Self {
        let tokenizer_clone = tokenizer.clone();
        let tokenizer = Arc::new(tokenizer);
        let chat_format = ChatFormat::new(tokenizer_clone);
        
        Self {
            config,
            tokenizer,
            chat_format,
            buffer_pool: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        }
    }

    /// Process a single text prompt
    pub fn process_prompt(&self, text: &str) -> Result<ValidatedInput> {
        let start_time = std::time::Instant::now();
        
        // Validate input length
        self.validate_text_length(text)?;
        
        // Tokenize the input
        let tokens = self.tokenizer.encode(text, true, false)?; // BOS but no EOS for prompts
        let original_token_count = tokens.len();
        
        // Apply context length limits
        let (final_tokens, was_truncated, truncation_position) = 
            self.apply_context_limits(tokens)?;
        
        let final_token_count = final_tokens.len();
        
        // Calculate memory usage
        let memory_usage = self.calculate_memory_usage(&final_tokens);
        
        // Update statistics
        self.update_stats(1, final_token_count, was_truncated, start_time.elapsed(), memory_usage);
        
        Ok(ValidatedInput {
            tokens: final_tokens,
            text: Some(text.to_string()),
            input_type: InputType::SinglePrompt,
            metadata: InputMetadata {
                original_token_count,
                final_token_count,
                was_truncated,
                truncation_position,
                processed_at: start_time,
                memory_usage,
            },
        })
    }

    /// Process a dialog (multi-turn conversation)
    pub fn process_dialog(&self, dialog: &Dialog) -> Result<ValidatedInput> {
        let start_time = std::time::Instant::now();
        
        // Validate dialog
        self.validate_dialog(dialog)?;
        
        // Encode dialog for completion
        let tokens = self.chat_format.encode_dialog_prompt(dialog)?;
        let original_token_count = tokens.len();
        
        // Apply context length limits
        let (final_tokens, was_truncated, truncation_position) = 
            self.apply_context_limits(tokens)?;
        
        let final_token_count = final_tokens.len();
        
        // Calculate memory usage
        let memory_usage = self.calculate_memory_usage(&final_tokens);
        
        // Update statistics
        self.update_stats(1, final_token_count, was_truncated, start_time.elapsed(), memory_usage);
        
        Ok(ValidatedInput {
            tokens: final_tokens,
            text: None, // Dialog doesn't have single text representation
            input_type: InputType::Dialog,
            metadata: InputMetadata {
                original_token_count,
                final_token_count,
                was_truncated,
                truncation_position,
                processed_at: start_time,
                memory_usage,
            },
        })
    }

    /// Process multiple inputs as a batch
    pub fn process_batch(&self, inputs: Vec<&str>) -> Result<InputBatch> {
        let input_count = inputs.len();
        
        if input_count > self.config.max_batch_size {
            return Err(InferenceError::ConfigError(
                format!("Batch size {} exceeds maximum {}", input_count, self.config.max_batch_size)
            ));
        }

        let batch_id = uuid::Uuid::new_v4();
        let start_time = std::time::Instant::now();
        
        let mut validated_inputs = Vec::with_capacity(input_count);
        let mut max_token_length = 0;
        let mut total_tokens = 0;
        
        for input_text in inputs {
            let validated = self.process_prompt(input_text)?;
            max_token_length = max_token_length.max(validated.tokens.len());
            total_tokens += validated.tokens.len();
            validated_inputs.push(validated);
        }
        
        // Estimate processing time based on token count and batch size
        let estimated_processing_time = self.estimate_processing_time(total_tokens, input_count);
        
        Ok(InputBatch {
            inputs: validated_inputs,
            metadata: BatchMetadata {
                batch_id,
                max_token_length,
                total_tokens,
                created_at: start_time,
                estimated_processing_time,
            },
        })
    }

    /// Process multiple dialogs as a batch
    pub fn process_dialog_batch(&self, dialogs: Vec<&Dialog>) -> Result<InputBatch> {
        let dialog_count = dialogs.len();
        
        if dialog_count > self.config.max_batch_size {
            return Err(InferenceError::ConfigError(
                format!("Batch size {} exceeds maximum {}", dialog_count, self.config.max_batch_size)
            ));
        }

        let batch_id = uuid::Uuid::new_v4();
        let start_time = std::time::Instant::now();
        
        let mut validated_inputs = Vec::with_capacity(dialog_count);
        let mut max_token_length = 0;
        let mut total_tokens = 0;
        
        for dialog in dialogs {
            let validated = self.process_dialog(dialog)?;
            max_token_length = max_token_length.max(validated.tokens.len());
            total_tokens += validated.tokens.len();
            validated_inputs.push(validated);
        }
        
        let estimated_processing_time = self.estimate_processing_time(total_tokens, dialog_count);
        
        Ok(InputBatch {
            inputs: validated_inputs,
            metadata: BatchMetadata {
                batch_id,
                max_token_length,
                total_tokens,
                created_at: start_time,
                estimated_processing_time,
            },
        })
    }

    /// Get or create a token buffer from the pool
    pub fn get_token_buffer(&self) -> TokenBuffer {
        let mut pool = self.buffer_pool.write();
        pool.pop().unwrap_or_else(|| TokenBuffer::new(self.config.token_buffer_size))
    }

    /// Return a token buffer to the pool
    pub fn return_token_buffer(&self, mut buffer: TokenBuffer) {
        buffer.clear();
        let mut pool = self.buffer_pool.write();
        if pool.len() < 16 { // Limit pool size
            pool.push(buffer);
        }
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        self.stats.read().clone()
    }

    /// Reset processing statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = ProcessingStats::default();
    }

    /// Get the current configuration
    pub fn config(&self) -> &InputProcessingConfig {
        &self.config
    }

    /// Validate text length before processing
    fn validate_text_length(&self, text: &str) -> Result<()> {
        // Check for excessively long input
        const MAX_TEXT_LENGTH: usize = 1_000_000; // 1M characters
        
        if text.len() > MAX_TEXT_LENGTH {
            return Err(InferenceError::ConfigError(
                format!("Input text too long: {} characters (max: {})", text.len(), MAX_TEXT_LENGTH)
            ));
        }
        
        if text.is_empty() {
            return Err(InferenceError::ConfigError(
                "Input text cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Validate dialog structure
    fn validate_dialog(&self, dialog: &Dialog) -> Result<()> {
        if dialog.is_empty() {
            return Err(InferenceError::ConfigError(
                "Dialog cannot be empty".to_string()
            ));
        }
        
        // Check for excessively long dialogs
        if dialog.len() > 100 { // Reasonable limit for conversation turns
            return Err(InferenceError::ConfigError(
                format!("Dialog too long: {} turns (max: 100)", dialog.len())
            ));
        }
        
        // Validate each message
        for (i, message) in dialog.iter().enumerate() {
            self.validate_text_length(&message.content)
                .map_err(|_| InferenceError::ConfigError(
                    format!("Message {} content too long", i)
                ))?;
        }
        
        Ok(())
    }

    /// Apply context length limits to token sequence
    fn apply_context_limits(&self, tokens: Vec<u32>) -> Result<(Vec<u32>, bool, Option<usize>)> {
        if tokens.len() <= self.config.max_context_length {
            return Ok((tokens, false, None));
        }
        
        if !self.config.auto_truncate {
            return Err(InferenceError::ConfigError(
                format!("Input exceeds max context length {} (auto-truncate disabled)", 
                       self.config.max_context_length)
            ));
        }
        
        // Truncate to max context length
        let truncated_tokens = tokens.into_iter()
            .take(self.config.max_context_length)
            .collect();
        
        Ok((truncated_tokens, true, Some(self.config.max_context_length)))
    }

    /// Calculate memory usage for token sequence
    fn calculate_memory_usage(&self, tokens: &[u32]) -> usize {
        // Base memory for token vector
        let token_memory = tokens.len() * std::mem::size_of::<u32>();
        
        // Add overhead for vector structure
        let vector_overhead = std::mem::size_of::<Vec<u32>>();
        
        // Add estimated processing overhead (rough estimate)
        let processing_overhead = tokens.len() * 4; // 4 bytes per token for processing
        
        token_memory + vector_overhead + processing_overhead
    }

    /// Update processing statistics
    fn update_stats(&self, input_count: u64, token_count: usize, was_truncated: bool, 
                   processing_time: std::time::Duration, memory_usage: usize) {
        let mut stats = self.stats.write();
        
        stats.total_processed += input_count;
        stats.total_tokens += token_count as u64;
        
        if was_truncated {
            stats.total_truncations += 1;
        }
        
        // Update average processing time (exponential moving average)
        if stats.total_processed == 1 {
            stats.average_processing_time = processing_time;
        } else {
            let alpha = 0.1; // Smoothing factor
            let new_time_nanos = processing_time.as_nanos() as f64;
            let avg_nanos = stats.average_processing_time.as_nanos() as f64;
            let updated_avg = (alpha * new_time_nanos) + ((1.0 - alpha) * avg_nanos);
            stats.average_processing_time = std::time::Duration::from_nanos(updated_avg as u64);
        }
        
        // Update memory usage
        stats.current_memory_usage += memory_usage;
        stats.peak_memory_usage = stats.peak_memory_usage.max(stats.current_memory_usage);
    }

    /// Estimate processing time for a batch
    fn estimate_processing_time(&self, total_tokens: usize, batch_size: usize) -> std::time::Duration {
        // Base estimation: ~1ms per 1000 tokens, with batch overhead
        let base_time_per_token = std::time::Duration::from_nanos(1_000); // 1 microsecond per token
        let batch_overhead = std::time::Duration::from_millis(batch_size as u64 * 2); // 2ms per input
        
        (base_time_per_token * total_tokens as u32) + batch_overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{LlamaTokenizer, Role, Message};
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn create_test_tokenizer() -> LlamaTokenizer {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test tokenizer file").unwrap();
        LlamaTokenizer::new(temp_file.path()).unwrap()
    }

    fn create_test_processor() -> InputProcessor {
        let tokenizer = create_test_tokenizer();
        let config = InputProcessingConfig::default();
        InputProcessor::new(tokenizer, config)
    }

    #[test]
    fn test_input_processing_config() {
        let config = InputProcessingConfig::default();
        assert_eq!(config.max_context_length, 4096);
        assert_eq!(config.max_batch_size, 32);
        assert!(config.auto_truncate);
    }

    #[test]
    fn test_token_buffer() {
        let mut buffer = TokenBuffer::new(10);
        
        // Test basic operations
        assert!(buffer.is_empty());
        assert_eq!(buffer.remaining_capacity(), 10);
        
        // Add tokens
        buffer.push_tokens(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.remaining_capacity(), 5);
        
        // Test overflow behavior
        buffer.push_tokens(&[6, 7, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(buffer.len(), 10); // Should be at capacity
        
        let tokens = buffer.get_all_tokens();
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 3); // First two tokens should be removed
    }

    #[test]
    fn test_process_prompt() {
        let processor = create_test_processor();
        
        let result = processor.process_prompt("Hello, world!").unwrap();
        
        assert_eq!(result.input_type, InputType::SinglePrompt);
        assert!(result.text.is_some());
        assert!(!result.tokens.is_empty());
        assert_eq!(result.metadata.original_token_count, result.metadata.final_token_count);
        assert!(!result.metadata.was_truncated);
    }

    #[test]
    fn test_process_dialog() {
        let processor = create_test_processor();
        
        let dialog = vec![
            Message {
                role: Role::System,
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: Role::User,
                content: "What is the capital of France?".to_string(),
            },
        ];
        
        let result = processor.process_dialog(&dialog).unwrap();
        
        assert_eq!(result.input_type, InputType::Dialog);
        assert!(result.text.is_none()); // Dialog doesn't have single text
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn test_batch_processing() {
        let processor = create_test_processor();
        
        let inputs = vec!["Hello", "World", "Test"];
        let batch = processor.process_batch(inputs).unwrap();
        
        assert_eq!(batch.inputs.len(), 3);
        assert!(batch.metadata.total_tokens > 0);
        assert!(batch.metadata.max_token_length > 0);
    }

    #[test]
    fn test_input_validation() {
        let processor = create_test_processor();
        
        // Test empty input
        let result = processor.process_prompt("");
        assert!(result.is_err());
        
        // Test very long input (if auto_truncate is enabled, should succeed)
        let long_input = "a".repeat(100_000);
        let result = processor.process_prompt(&long_input);
        if processor.config.auto_truncate {
            assert!(result.is_ok());
            let validated = result.unwrap();
            assert!(validated.metadata.was_truncated || validated.tokens.len() <= processor.config.max_context_length);
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_context_length_limits() {
        let mut config = InputProcessingConfig::default();
        config.max_context_length = 10; // Very small for testing
        config.auto_truncate = true;
        
        let tokenizer = create_test_tokenizer();
        let processor = InputProcessor::new(tokenizer, config);
        
        // Create a long text that should generate more than 10 tokens
        let long_text = "word ".repeat(20); // Should generate many tokens
        let result = processor.process_prompt(&long_text).unwrap();
        
        // Either the tokens are within limit or truncation occurred
        assert!(result.tokens.len() <= 10 || result.metadata.was_truncated);
        
        if result.metadata.was_truncated {
            assert!(result.metadata.truncation_position.is_some());
        }
    }

    #[test]
    fn test_batch_size_limits() {
        let processor = create_test_processor();
        
        // Create batch larger than max_batch_size
        let large_batch: Vec<&str> = (0..100).map(|_| "test").collect();
        let result = processor.process_batch(large_batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_usage_calculation() {
        let processor = create_test_processor();
        
        let result = processor.process_prompt("Test input").unwrap();
        assert!(result.metadata.memory_usage > 0);
    }

    #[test]
    fn test_processing_stats() {
        let processor = create_test_processor();
        
        // Process some inputs
        let _ = processor.process_prompt("Test 1").unwrap();
        let _ = processor.process_prompt("Test 2").unwrap();
        
        let stats = processor.get_stats();
        assert_eq!(stats.total_processed, 2);
        assert!(stats.total_tokens > 0);
        assert!(stats.average_processing_time.as_nanos() > 0);
    }

    #[test]
    fn test_buffer_pool() {
        let processor = create_test_processor();
        
        // Get a buffer from pool
        let buffer1 = processor.get_token_buffer();
        assert_eq!(buffer1.len(), 0);
        
        // Return it to pool
        processor.return_token_buffer(buffer1);
        
        // Get another buffer (should reuse from pool)
        let buffer2 = processor.get_token_buffer();
        assert_eq!(buffer2.len(), 0);
    }
}