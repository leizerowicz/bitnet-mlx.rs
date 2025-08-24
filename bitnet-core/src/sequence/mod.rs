//! Sequence Processing Module
//!
//! This module provides comprehensive sequence processing capabilities for BitNet,
//! including padding, truncation, batching, masking, and validation operations.

pub mod batching;
pub mod manager;
pub mod masking;
pub mod padding;
pub mod statistics;
pub mod tokenizer_integration;
pub mod truncation;
pub mod validation;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Padding strategies for sequence processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to the longest sequence in the batch
    LongestInBatch,
    /// Pad to a fixed length
    FixedLength(usize),
    /// Pad to the maximum configured length
    MaxLength,
    /// Pad to the next multiple of a given value
    ToMultiple(usize),
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        PaddingStrategy::LongestInBatch
    }
}

impl PaddingStrategy {
    /// Calculate the target length for padding based on the strategy
    pub fn calculate_target_length(
        &self,
        lengths: &[usize],
        max_length: Option<usize>,
    ) -> Option<usize> {
        match self {
            PaddingStrategy::None => None,
            PaddingStrategy::LongestInBatch => lengths.iter().max().copied(),
            PaddingStrategy::FixedLength(length) => Some(*length),
            PaddingStrategy::MaxLength => max_length,
            PaddingStrategy::ToMultiple(multiple) => {
                let max_len = lengths.iter().max().copied().unwrap_or(0);
                if *multiple > 0 {
                    Some(((max_len + multiple - 1) / multiple) * multiple)
                } else {
                    Some(max_len)
                }
            }
        }
    }
}

/// Truncation strategies for sequence processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// No truncation
    None,
    /// Truncate from the left (beginning)
    TruncateLeft,
    /// Truncate from the right (end)
    TruncateRight,
    /// Truncate the longest sequences first
    LongestFirst,
    /// Only truncate if sequence exceeds maximum length
    OnlyIfExceeds,
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        TruncationStrategy::TruncateRight
    }
}

/// Configuration for sequence processing operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceConfig {
    /// Maximum allowed sequence length
    pub max_length: Option<usize>,
    /// Minimum allowed sequence length
    pub min_length: Option<usize>,
    /// Padding strategy to use
    pub padding_strategy: PaddingStrategy,
    /// Truncation strategy to use
    pub truncation_strategy: TruncationStrategy,
    /// Padding token ID
    pub pad_token_id: Option<u32>,
    /// Whether to return attention masks
    pub return_attention_mask: bool,
    /// Maximum batch size
    pub max_batch_size: Option<usize>,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            max_length: Some(512),
            min_length: None,
            padding_strategy: PaddingStrategy::LongestInBatch,
            truncation_strategy: TruncationStrategy::TruncateRight,
            pad_token_id: None,
            return_attention_mask: true,
            max_batch_size: None,
        }
    }
}

impl SequenceConfig {
    /// Create a new sequence configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set maximum length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Builder method to set minimum length
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = Some(min_length);
        self
    }

    /// Builder method to set padding strategy
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    /// Builder method to set truncation strategy
    pub fn with_truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self
    }

    /// Builder method to set padding token ID
    pub fn with_pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = Some(pad_token_id);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), SequenceError> {
        if let (Some(min), Some(max)) = (self.min_length, self.max_length) {
            if min > max {
                return Err(SequenceError::InvalidConfiguration {
                    message: format!(
                        "min_length ({}) cannot be greater than max_length ({})",
                        min, max
                    ),
                });
            }
        }

        if let Some(max_batch) = self.max_batch_size {
            if max_batch == 0 {
                return Err(SequenceError::InvalidConfiguration {
                    message: "max_batch_size cannot be zero".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// A processed sequence with metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessedSequence {
    /// The processed token sequence
    pub tokens: Vec<u32>,
    /// Original length before processing
    pub original_length: usize,
    /// Current length after processing
    pub current_length: usize,
    /// Attention mask for the sequence
    pub attention_mask: Vec<u8>,
    /// Whether the sequence was truncated
    pub was_truncated: bool,
    /// Whether the sequence was padded
    pub was_padded: bool,
    /// Number of tokens truncated
    pub tokens_truncated: usize,
    /// Amount of padding added
    pub padding_added: usize,
}

impl ProcessedSequence {
    /// Create a new processed sequence
    pub fn new(
        tokens: Vec<u32>,
        original_length: usize,
        attention_mask: Vec<u8>,
        was_truncated: bool,
        was_padded: bool,
        tokens_truncated: usize,
        padding_added: usize,
    ) -> Self {
        let current_length = tokens.len();
        Self {
            tokens,
            original_length,
            current_length,
            attention_mask,
            was_truncated,
            was_padded,
            tokens_truncated,
            padding_added,
        }
    }

    /// Get the effective length (non-padding tokens)
    pub fn effective_length(&self) -> usize {
        self.current_length - self.padding_added
    }

    /// Check if the sequence is valid
    pub fn is_valid(&self) -> bool {
        self.tokens.len() == self.attention_mask.len() && self.current_length == self.tokens.len()
    }
}

/// Sequence processing errors
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceError {
    /// Empty sequence provided
    EmptySequence,
    /// Sequence too long
    SequenceTooLong { length: usize, max_length: usize },
    /// Sequence too short
    SequenceTooShort { length: usize, min_length: usize },
    /// Sequence below minimum length (alias for validation)
    BelowMinLength { length: usize, min_length: usize },
    /// Sequence exceeds maximum length (alias for validation)
    ExceedsMaxLength { length: usize, max_length: usize },
    /// Batch too large
    BatchTooLarge { size: usize, max_size: usize },
    /// Invalid configuration
    InvalidConfiguration { message: String },
    /// Invalid padding configuration
    InvalidPadding { message: String },
    /// Invalid padding token
    InvalidPaddingToken { token: u32 },
    /// Invalid truncation configuration
    InvalidTruncation { message: String },
    /// Processing error
    ProcessingError { message: String },
    /// Configuration error
    ConfigError { message: String },
    /// Inconsistent batch lengths
    InconsistentBatchLengths,
}

impl fmt::Display for SequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SequenceError::EmptySequence => write!(f, "Empty sequence provided"),
            SequenceError::SequenceTooLong { length, max_length } => {
                write!(
                    f,
                    "Sequence length {} exceeds maximum {}",
                    length, max_length
                )
            }
            SequenceError::SequenceTooShort { length, min_length } => {
                write!(
                    f,
                    "Sequence length {} is below minimum {}",
                    length, min_length
                )
            }
            SequenceError::BelowMinLength { length, min_length } => {
                write!(
                    f,
                    "Sequence length {} is below minimum {}",
                    length, min_length
                )
            }
            SequenceError::ExceedsMaxLength { length, max_length } => {
                write!(
                    f,
                    "Sequence length {} exceeds maximum {}",
                    length, max_length
                )
            }
            SequenceError::BatchTooLarge { size, max_size } => {
                write!(f, "Batch size {} exceeds maximum {}", size, max_size)
            }
            SequenceError::InvalidConfiguration { message } => {
                write!(f, "Invalid configuration: {}", message)
            }
            SequenceError::InvalidPadding { message } => {
                write!(f, "Invalid padding: {}", message)
            }
            SequenceError::InvalidPaddingToken { token } => {
                write!(f, "Invalid padding token: {}", token)
            }
            SequenceError::InvalidTruncation { message } => {
                write!(f, "Invalid truncation: {}", message)
            }
            SequenceError::ProcessingError { message } => {
                write!(f, "Processing error: {}", message)
            }
            SequenceError::ConfigError { message } => {
                write!(f, "Configuration error: {}", message)
            }
            SequenceError::InconsistentBatchLengths => {
                write!(f, "Inconsistent batch lengths")
            }
        }
    }
}

impl std::error::Error for SequenceError {}

/// Result type for sequence operations
pub type SequenceResult<T> = Result<T, SequenceError>;

// Re-export commonly used types and functions
pub use batching::{BatchProcessor, SequenceBatch};
pub use manager::{ProcessingSummary, SequenceManager};
pub use masking::{create_attention_mask, AttentionMaskType};
pub use padding::{pad_sequence, PaddingOptions};
pub use statistics::{analyze_sequence_lengths, SequenceStats};
pub use truncation::{truncate_sequence, TruncationOptions};
pub use validation::{validate_sequence_length, SequenceValidator};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_config_default() {
        let config = SequenceConfig::default();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.padding_strategy, PaddingStrategy::LongestInBatch);
        assert_eq!(
            config.truncation_strategy,
            TruncationStrategy::TruncateRight
        );
        assert!(config.return_attention_mask);
    }

    #[test]
    fn test_sequence_config_validation() {
        let mut config = SequenceConfig::default();
        config.min_length = Some(100);
        config.max_length = Some(50);

        assert!(config.validate().is_err());

        config.max_length = Some(200);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_processed_sequence_creation() {
        let tokens = vec![1, 2, 3, 0, 0];
        let attention_mask = vec![1, 1, 1, 0, 0];

        let seq =
            ProcessedSequence::new(tokens.clone(), 3, attention_mask.clone(), false, true, 0, 2);

        assert_eq!(seq.tokens, tokens);
        assert_eq!(seq.attention_mask, attention_mask);
        assert_eq!(seq.original_length, 3);
        assert_eq!(seq.current_length, 5);
        assert_eq!(seq.effective_length(), 3);
        assert!(seq.is_valid());
    }

    #[test]
    fn test_padding_strategy_calculate_target_length() {
        let lengths = vec![3, 5, 2];

        assert_eq!(
            PaddingStrategy::LongestInBatch.calculate_target_length(&lengths, None),
            Some(5)
        );

        assert_eq!(
            PaddingStrategy::FixedLength(10).calculate_target_length(&lengths, None),
            Some(10)
        );

        assert_eq!(
            PaddingStrategy::ToMultiple(4).calculate_target_length(&lengths, None),
            Some(8) // 5 rounded up to next multiple of 4
        );
    }
}
