//! Tokenizer Integration for Sequence Management
//!
//! This module provides integration between the tokenizer and sequence management
//! systems, enabling seamless text processing with automatic sequence handling.

use super::{manager::SequenceManager, ProcessedSequence, SequenceConfig, SequenceResult};
use crate::tokenizer::{decode_tokens, encode_batch, encode_text, Tokenizer};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Enhanced tokenizer with sequence management capabilities
#[derive(Debug, Clone)]
pub struct SequenceAwareTokenizer {
    /// The underlying tokenizer
    tokenizer: Tokenizer,
    /// Sequence manager for handling length constraints
    sequence_manager: SequenceManager,
    /// Configuration for sequence processing
    config: TokenizerSequenceConfig,
}

/// Configuration for tokenizer sequence integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerSequenceConfig {
    /// Whether to automatically manage sequence lengths
    pub auto_manage_sequences: bool,
    /// Whether to add special tokens automatically
    pub add_special_tokens: bool,
    /// Special tokens to add to sequences
    pub special_tokens: SpecialTokens,
    /// Whether to return original text alongside processed sequences
    pub return_original_text: bool,
    /// Whether to validate token IDs against vocabulary
    pub validate_token_ids: bool,
}

/// Special tokens for sequence processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: Option<u32>,
    /// End of sequence token
    pub eos_token: Option<u32>,
    /// Padding token
    pub pad_token: Option<u32>,
    /// Unknown token
    pub unk_token: Option<u32>,
    /// Mask token (for masked language modeling)
    pub mask_token: Option<u32>,
    /// Classification token
    pub cls_token: Option<u32>,
    /// Separator token
    pub sep_token: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: None,
            eos_token: None,
            pad_token: Some(0),
            unk_token: Some(1),
            mask_token: None,
            cls_token: None,
            sep_token: None,
        }
    }
}

impl Default for TokenizerSequenceConfig {
    fn default() -> Self {
        Self {
            auto_manage_sequences: true,
            add_special_tokens: true,
            special_tokens: SpecialTokens::default(),
            return_original_text: false,
            validate_token_ids: true,
        }
    }
}

/// Result of tokenization with sequence management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationResult {
    /// Processed sequences
    pub sequences: Vec<ProcessedSequence>,
    /// Original texts (if requested)
    pub original_texts: Option<Vec<String>>,
    /// Tokenization metadata
    pub metadata: TokenizationMetadata,
}

/// Metadata about the tokenization process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationMetadata {
    /// Number of input texts
    pub input_count: usize,
    /// Total tokens before processing
    pub total_input_tokens: usize,
    /// Total tokens after processing
    pub total_output_tokens: usize,
    /// Number of sequences that were truncated
    pub truncated_count: usize,
    /// Number of sequences that were padded
    pub padded_count: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

impl SequenceAwareTokenizer {
    /// Create a new sequence-aware tokenizer
    pub fn new(
        tokenizer: Tokenizer,
        sequence_config: SequenceConfig,
        tokenizer_config: TokenizerSequenceConfig,
    ) -> Result<Self> {
        let sequence_manager = SequenceManager::with_config(sequence_config)?;

        Ok(Self {
            tokenizer,
            sequence_manager,
            config: tokenizer_config,
        })
    }

    /// Create with default configuration
    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        let sequence_config = SequenceConfig::default();
        let tokenizer_config = TokenizerSequenceConfig::default();

        Self {
            sequence_manager: SequenceManager::with_config(sequence_config).unwrap(),
            tokenizer,
            config: tokenizer_config,
        }
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the sequence manager
    pub fn sequence_manager(&self) -> &SequenceManager {
        &self.sequence_manager
    }

    /// Get mutable access to the sequence manager
    pub fn sequence_manager_mut(&mut self) -> &mut SequenceManager {
        &mut self.sequence_manager
    }

    /// Encode a single text with sequence management
    pub fn encode_with_sequence_management(
        &mut self,
        text: &str,
    ) -> SequenceResult<ProcessedSequence> {
        // Tokenize the text
        let mut tokens =
            encode_text(&self.tokenizer, text).map_err(|e| super::SequenceError::ConfigError {
                message: format!("Tokenization failed: {}", e),
            })?;

        // Add special tokens if configured
        if self.config.add_special_tokens {
            tokens = self.add_special_tokens(tokens);
        }

        // Validate token IDs if configured
        if self.config.validate_token_ids {
            self.validate_tokens(&tokens)?;
        }

        // Process with sequence manager
        let pad_token = self.config.special_tokens.pad_token;
        self.sequence_manager.process_sequence(tokens, pad_token)
    }

    /// Encode multiple texts as a batch with sequence management
    pub fn encode_batch_with_sequence_management(
        &mut self,
        texts: &[&str],
    ) -> SequenceResult<TokenizationResult> {
        let start_time = std::time::Instant::now();

        // Tokenize all texts
        let token_sequences = encode_batch(&self.tokenizer, texts).map_err(|e| {
            super::SequenceError::ConfigError {
                message: format!("Batch tokenization failed: {}", e),
            }
        })?;

        // Add special tokens if configured
        let processed_tokens: Vec<Vec<u32>> = if self.config.add_special_tokens {
            token_sequences
                .into_iter()
                .map(|tokens| self.add_special_tokens(tokens))
                .collect()
        } else {
            token_sequences
        };

        // Validate token IDs if configured
        if self.config.validate_token_ids {
            for tokens in &processed_tokens {
                self.validate_tokens(tokens)?;
            }
        }

        // Calculate input statistics
        let total_input_tokens: usize = processed_tokens.iter().map(|t| t.len()).sum();

        // Process with sequence manager
        let pad_token = self.config.special_tokens.pad_token;
        let batch = self
            .sequence_manager
            .process_batch(&processed_tokens, pad_token)?;

        // Calculate metadata
        let processing_time = start_time.elapsed().as_millis() as u64;
        let sequences = batch.sequences().to_vec();

        let total_output_tokens: usize = sequences.iter().map(|s| s.current_length).sum();
        let truncated_count = sequences.iter().filter(|s| s.was_truncated).count();
        let padded_count = sequences.iter().filter(|s| s.was_padded).count();

        let avg_compression_ratio = if total_input_tokens > 0 {
            total_output_tokens as f32 / total_input_tokens as f32
        } else {
            1.0
        };

        let metadata = TokenizationMetadata {
            input_count: texts.len(),
            total_input_tokens,
            total_output_tokens,
            truncated_count,
            padded_count,
            avg_compression_ratio,
            processing_time_ms: processing_time,
        };

        let original_texts = if self.config.return_original_text {
            Some(texts.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };

        Ok(TokenizationResult {
            sequences,
            original_texts,
            metadata,
        })
    }

    /// Decode sequences back to text
    pub fn decode_sequences(&self, sequences: &[ProcessedSequence]) -> Result<Vec<String>> {
        let mut decoded_texts = Vec::with_capacity(sequences.len());

        for sequence in sequences {
            // Remove special tokens before decoding
            let tokens_to_decode = self.remove_special_tokens(&sequence.tokens);

            let decoded = decode_tokens(&self.tokenizer, &tokens_to_decode)
                .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

            decoded_texts.push(decoded);
        }

        Ok(decoded_texts)
    }

    /// Add special tokens to a sequence
    fn add_special_tokens(&self, mut tokens: Vec<u32>) -> Vec<u32> {
        let special = &self.config.special_tokens;

        // Add BOS token at the beginning
        if let Some(bos) = special.bos_token {
            tokens.insert(0, bos);
        }

        // Add CLS token at the beginning (after BOS if present)
        if let Some(cls) = special.cls_token {
            let insert_pos = if special.bos_token.is_some() { 1 } else { 0 };
            tokens.insert(insert_pos, cls);
        }

        // Add EOS token at the end
        if let Some(eos) = special.eos_token {
            tokens.push(eos);
        }

        tokens
    }

    /// Remove special tokens from a sequence for decoding
    fn remove_special_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        let special = &self.config.special_tokens;
        let special_token_set: std::collections::HashSet<u32> = [
            special.bos_token,
            special.eos_token,
            special.pad_token,
            special.cls_token,
            special.sep_token,
            special.mask_token,
        ]
        .iter()
        .filter_map(|&token| token)
        .collect();

        tokens
            .iter()
            .filter(|&&token| !special_token_set.contains(&token))
            .copied()
            .collect()
    }

    /// Validate token IDs against vocabulary size
    fn validate_tokens(&self, tokens: &[u32]) -> SequenceResult<()> {
        let vocab_size = self.tokenizer.vocab_size();
        let special = &self.config.special_tokens;

        // Create set of allowed special tokens
        let special_tokens: std::collections::HashSet<u32> = [
            special.bos_token,
            special.eos_token,
            special.pad_token,
            special.unk_token,
            special.cls_token,
            special.sep_token,
            special.mask_token,
        ]
        .iter()
        .filter_map(|&token| token)
        .collect();

        for &token in tokens {
            if token >= vocab_size as u32 && !special_tokens.contains(&token) {
                return Err(super::SequenceError::InvalidPaddingToken { token });
            }
        }

        Ok(())
    }

    /// Create sequences for text classification tasks
    pub fn encode_text_classification(
        &mut self,
        text_a: &str,
        text_b: Option<&str>,
    ) -> SequenceResult<ProcessedSequence> {
        // Tokenize first text
        let mut tokens_a = encode_text(&self.tokenizer, text_a).map_err(|e| {
            super::SequenceError::ConfigError {
                message: format!("Tokenization failed: {}", e),
            }
        })?;

        let mut final_tokens = Vec::new();
        let special = &self.config.special_tokens;

        // Add CLS token
        if let Some(cls) = special.cls_token {
            final_tokens.push(cls);
        }

        // Add first text tokens
        final_tokens.append(&mut tokens_a);

        // Add SEP token
        if let Some(sep) = special.sep_token {
            final_tokens.push(sep);
        }

        // Add second text if provided
        if let Some(text_b_str) = text_b {
            let mut tokens_b = encode_text(&self.tokenizer, text_b_str).map_err(|e| {
                super::SequenceError::ConfigError {
                    message: format!("Tokenization failed: {}", e),
                }
            })?;

            final_tokens.append(&mut tokens_b);

            // Add final SEP token
            if let Some(sep) = special.sep_token {
                final_tokens.push(sep);
            }
        }

        // Validate and process
        if self.config.validate_token_ids {
            self.validate_tokens(&final_tokens)?;
        }

        let pad_token = special.pad_token;
        self.sequence_manager
            .process_sequence(final_tokens, pad_token)
    }

    /// Create sequences for token classification tasks (NER, POS tagging)
    pub fn encode_token_classification(
        &mut self,
        text: &str,
        labels: Option<&[String]>,
    ) -> SequenceResult<(ProcessedSequence, Option<Vec<u32>>)> {
        // For token classification, we typically don't add special tokens
        // except for padding, as we need 1:1 alignment with labels

        let tokens =
            encode_text(&self.tokenizer, text).map_err(|e| super::SequenceError::ConfigError {
                message: format!("Tokenization failed: {}", e),
            })?;

        // Process labels if provided (this is a simplified approach)
        let processed_labels = if let Some(label_strs) = labels {
            // In a real implementation, you'd need proper label-to-token alignment
            // This is a placeholder that assumes 1:1 mapping
            let label_ids: Vec<u32> = label_strs
                .iter()
                .enumerate()
                .map(|(i, _)| i as u32) // Simplified label encoding
                .collect();
            Some(label_ids)
        } else {
            None
        };

        if self.config.validate_token_ids {
            self.validate_tokens(&tokens)?;
        }

        let pad_token = self.config.special_tokens.pad_token;
        let sequence = self.sequence_manager.process_sequence(tokens, pad_token)?;

        Ok((sequence, processed_labels))
    }

    /// Update the tokenizer configuration
    pub fn update_config(&mut self, config: TokenizerSequenceConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &TokenizerSequenceConfig {
        &self.config
    }

    /// Get processing statistics from the sequence manager
    pub fn get_processing_stats(&self) -> Option<&super::statistics::SequenceStats> {
        self.sequence_manager.stats()
    }

    /// Reset processing statistics
    pub fn reset_stats(&mut self) {
        self.sequence_manager.reset_stats();
    }
}

/// Convenience functions for common use cases

/// Create a sequence-aware tokenizer for BERT-style models
pub fn create_bert_tokenizer(
    tokenizer: Tokenizer,
    max_length: usize,
) -> Result<SequenceAwareTokenizer> {
    let sequence_config = SequenceConfig::new()
        .with_max_length(max_length)
        .with_padding_strategy(super::PaddingStrategy::MaxLength);

    let mut tokenizer_config = TokenizerSequenceConfig::default();
    tokenizer_config.special_tokens.cls_token = Some(101); // [CLS]
    tokenizer_config.special_tokens.sep_token = Some(102); // [SEP]
    tokenizer_config.special_tokens.pad_token = Some(0); // [PAD]

    SequenceAwareTokenizer::new(tokenizer, sequence_config, tokenizer_config)
}

/// Create a sequence-aware tokenizer for GPT-style models
pub fn create_gpt_tokenizer(
    tokenizer: Tokenizer,
    max_length: usize,
) -> Result<SequenceAwareTokenizer> {
    let sequence_config = SequenceConfig::new()
        .with_max_length(max_length)
        .with_padding_strategy(super::PaddingStrategy::LongestInBatch);

    let mut tokenizer_config = TokenizerSequenceConfig::default();
    tokenizer_config.special_tokens.bos_token = Some(50256); // <|endoftext|>
    tokenizer_config.special_tokens.eos_token = Some(50256); // <|endoftext|>
    tokenizer_config.special_tokens.pad_token = Some(50256); // <|endoftext|>

    SequenceAwareTokenizer::new(tokenizer, sequence_config, tokenizer_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::create_simple_tokenizer;
    use std::collections::HashMap;

    fn create_test_tokenizer() -> Tokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("test".to_string(), 2);
        vocab.insert("<pad>".to_string(), 3);
        vocab.insert("<cls>".to_string(), 4);
        vocab.insert("<sep>".to_string(), 5);

        create_simple_tokenizer(vocab)
    }

    #[test]
    fn test_sequence_aware_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        let sequence_config = SequenceConfig::new().with_max_length(10);
        let tokenizer_config = TokenizerSequenceConfig::default();

        let seq_tokenizer =
            SequenceAwareTokenizer::new(tokenizer, sequence_config, tokenizer_config);

        assert!(seq_tokenizer.is_ok());
    }

    #[test]
    fn test_special_tokens_addition() {
        let tokenizer = create_test_tokenizer();
        let mut seq_tokenizer = SequenceAwareTokenizer::with_tokenizer(tokenizer);

        // Configure special tokens
        seq_tokenizer.config.special_tokens.cls_token = Some(4);
        seq_tokenizer.config.special_tokens.sep_token = Some(5);

        let tokens = vec![0, 1]; // "hello world"
        let tokens_with_special = seq_tokenizer.add_special_tokens(tokens);

        // Should have CLS at beginning
        assert_eq!(tokens_with_special[0], 4);
        assert_eq!(tokens_with_special[1], 0); // hello
        assert_eq!(tokens_with_special[2], 1); // world
    }

    #[test]
    fn test_special_tokens_removal() {
        let tokenizer = create_test_tokenizer();
        let seq_tokenizer = SequenceAwareTokenizer::with_tokenizer(tokenizer);

        let tokens_with_special = vec![4, 0, 1, 5, 3, 3]; // [CLS] hello world [SEP] [PAD] [PAD]
        let cleaned_tokens = seq_tokenizer.remove_special_tokens(&tokens_with_special);

        // Should only have content tokens
        assert_eq!(cleaned_tokens, vec![0, 1]); // hello world
    }

    #[test]
    fn test_bert_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        let bert_tokenizer = create_bert_tokenizer(tokenizer, 512);

        assert!(bert_tokenizer.is_ok());
        let tokenizer = bert_tokenizer.unwrap();
        assert_eq!(tokenizer.config().special_tokens.cls_token, Some(101));
        assert_eq!(tokenizer.config().special_tokens.sep_token, Some(102));
    }
}
