//! Sequence Manager
//!
//! This module provides the main SequenceManager class that orchestrates
//! all sequence processing operations including padding, truncation, and batching.

use super::{
    SequenceConfig, PaddingStrategy, TruncationStrategy,
    padding::{pad_sequence, PaddingOptions},
    truncation::{truncate_sequence, TruncationOptions},
    masking::create_attention_mask,
    batching::{SequenceBatch, BatchProcessor},
    validation::SequenceValidator,
    statistics::{SequenceStats, analyze_sequence_lengths},
    ProcessedSequence, SequenceError, SequenceResult,
};
use anyhow::Result;

/// Main sequence manager for processing variable-length sequences
#[derive(Debug, Clone)]
pub struct SequenceManager {
    config: SequenceConfig,
    validator: SequenceValidator,
    batch_processor: BatchProcessor,
    stats: Option<SequenceStats>,
}

impl SequenceManager {
    /// Create a new sequence manager with default configuration
    pub fn new() -> Self {
        let config = SequenceConfig::default();
        Self {
            validator: SequenceValidator::new(&config),
            batch_processor: BatchProcessor::new(&config),
            stats: None,
            config,
        }
    }

    /// Create a sequence manager with custom configuration
    pub fn with_config(config: SequenceConfig) -> Result<Self> {
        config.validate().map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;
        
        Ok(Self {
            validator: SequenceValidator::new(&config),
            batch_processor: BatchProcessor::new(&config),
            stats: None,
            config,
        })
    }

    /// Builder method to set maximum length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.config.max_length = Some(max_length);
        self.validator = SequenceValidator::new(&self.config);
        self.batch_processor = BatchProcessor::new(&self.config);
        self
    }

    /// Builder method to set minimum length
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.config.min_length = Some(min_length);
        self.validator = SequenceValidator::new(&self.config);
        self
    }

    /// Builder method to set padding strategy
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.config.padding_strategy = strategy;
        self.batch_processor = BatchProcessor::new(&self.config);
        self
    }

    /// Builder method to set truncation strategy
    pub fn with_truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.config.truncation_strategy = strategy;
        self
    }

    /// Builder method to set padding token
    pub fn with_pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.config.pad_token_id = Some(pad_token_id);
        self
    }

    /// Enable statistics collection
    pub fn with_statistics(mut self) -> Self {
        self.stats = Some(SequenceStats::new());
        self
    }

    /// Get the current configuration
    pub fn config(&self) -> &SequenceConfig {
        &self.config
    }

    /// Get collected statistics (if enabled)
    pub fn stats(&self) -> Option<&SequenceStats> {
        self.stats.as_ref()
    }

    /// Process a single sequence
    pub fn process_sequence(
        &mut self,
        tokens: Vec<u32>,
        pad_token_id: Option<u32>,
    ) -> SequenceResult<ProcessedSequence> {
        let original_length = tokens.len();
        
        // Validate sequence
        self.validator.validate_sequence(&tokens)?;
        
        // Update statistics if enabled
        if let Some(ref mut stats) = self.stats {
            stats.add_sequence_length(original_length);
        }

        // Apply truncation if needed
        let (processed_tokens, tokens_truncated) = self.apply_truncation(tokens)?;
        let was_truncated = tokens_truncated > 0;

        // Apply padding if needed
        let pad_token = pad_token_id.or(self.config.pad_token_id).unwrap_or(0);
        let (final_tokens, padding_added) = self.apply_padding(processed_tokens, pad_token)?;
        let was_padded = padding_added > 0;

        // Generate attention mask
        let attention_mask = if self.config.return_attention_mask {
            create_attention_mask(&final_tokens, pad_token)
        } else {
            vec![1; final_tokens.len()]
        };

        Ok(ProcessedSequence::new(
            final_tokens,
            original_length,
            attention_mask,
            was_truncated,
            was_padded,
            tokens_truncated,
            padding_added,
        ))
    }

    /// Process a batch of sequences
    pub fn process_batch(
        &mut self,
        sequences: &[Vec<u32>],
        pad_token_id: Option<u32>,
    ) -> SequenceResult<SequenceBatch> {
        if sequences.is_empty() {
            return Err(SequenceError::EmptySequence);
        }

        // Validate batch size
        if let Some(max_batch_size) = self.config.max_batch_size {
            if sequences.len() > max_batch_size {
                return Err(SequenceError::BatchTooLarge {
                    size: sequences.len(),
                    max_size: max_batch_size,
                });
            }
        }

        // Collect original lengths for statistics
        let original_lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        
        // Update statistics if enabled
        if let Some(ref mut stats) = self.stats {
            for &length in &original_lengths {
                stats.add_sequence_length(length);
            }
        }

        // Process each sequence individually first
        let mut processed_sequences = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            // Validate each sequence
            self.validator.validate_sequence(sequence)?;
            
            // Apply truncation
            let (truncated_tokens, tokens_truncated) = self.apply_truncation(sequence.clone())?;
            processed_sequences.push((truncated_tokens, tokens_truncated));
        }

        // Determine target length for padding
        let current_lengths: Vec<usize> = processed_sequences.iter().map(|(tokens, _)| tokens.len()).collect();
        let target_length = self.config.padding_strategy
            .calculate_target_length(&current_lengths, self.config.max_length);

        // Apply padding to achieve uniform length
        let pad_token = pad_token_id.or(self.config.pad_token_id).unwrap_or(0);
        let mut final_sequences = Vec::with_capacity(sequences.len());

        for (i, (tokens, tokens_truncated)) in processed_sequences.into_iter().enumerate() {
            let (final_tokens, padding_added) = if let Some(target_len) = target_length {
                self.apply_padding_to_length(tokens, pad_token, target_len)?
            } else {
                (tokens, 0)
            };

            let attention_mask = if self.config.return_attention_mask {
                create_attention_mask(&final_tokens, pad_token)
            } else {
                vec![1; final_tokens.len()]
            };

            let processed = ProcessedSequence::new(
                final_tokens,
                original_lengths[i],
                attention_mask,
                tokens_truncated > 0,
                padding_added > 0,
                tokens_truncated,
                padding_added,
            );

            final_sequences.push(processed);
        }

        Ok(SequenceBatch::new(final_sequences))
    }

    /// Apply truncation to a sequence based on the configured strategy
    fn apply_truncation(&self, mut tokens: Vec<u32>) -> SequenceResult<(Vec<u32>, usize)> {
        let original_length = tokens.len();
        
        if let Some(max_length) = self.config.max_length {
            if tokens.len() > max_length {
                let truncation_options = TruncationOptions {
                    strategy: self.config.truncation_strategy,
                    max_length,
                };
                
                tokens = truncate_sequence(tokens, &truncation_options)?;
                let tokens_truncated = original_length - tokens.len();
                return Ok((tokens, tokens_truncated));
            }
        }

        Ok((tokens, 0))
    }

    /// Apply padding to a sequence based on the configured strategy
    fn apply_padding(&self, tokens: Vec<u32>, pad_token: u32) -> SequenceResult<(Vec<u32>, usize)> {
        // For single sequence, we only pad if using FixedLength or MaxLength strategy
        match self.config.padding_strategy {
            PaddingStrategy::FixedLength(target_length) => {
                self.apply_padding_to_length(tokens, pad_token, target_length)
            }
            PaddingStrategy::MaxLength => {
                if let Some(max_length) = self.config.max_length {
                    self.apply_padding_to_length(tokens, pad_token, max_length)
                } else {
                    Ok((tokens, 0))
                }
            }
            _ => Ok((tokens, 0)), // No padding for other strategies on single sequences
        }
    }

    /// Apply padding to reach a specific target length
    fn apply_padding_to_length(
        &self,
        tokens: Vec<u32>,
        pad_token: u32,
        target_length: usize,
    ) -> SequenceResult<(Vec<u32>, usize)> {
        let original_length = tokens.len();
        if original_length >= target_length {
            return Ok((tokens, 0));
        }

        let padding_options = PaddingOptions {
            pad_token,
            target_length,
            pad_to_multiple: None,
        };

        let padded_tokens = pad_sequence(tokens, &padding_options)?;
        let padding_added = padded_tokens.len() - original_length;
        
        Ok((padded_tokens, padding_added))
    }

    /// Get sequence length statistics for the current batch
    pub fn analyze_batch_lengths(&self, sequences: &[Vec<u32>]) -> SequenceStats {
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        analyze_sequence_lengths(&lengths)
    }

    /// Reset collected statistics
    pub fn reset_stats(&mut self) {
        if let Some(ref mut stats) = self.stats {
            *stats = SequenceStats::new();
        }
    }

    /// Validate a batch of sequences without processing
    pub fn validate_batch(&mut self, sequences: &[Vec<u32>]) -> SequenceResult<()> {
        for sequence in sequences {
            self.validator.validate_sequence(sequence)?;
        }
        Ok(())
    }

    /// Get memory usage estimate for a batch
    pub fn estimate_memory_usage(&self, sequences: &[Vec<u32>]) -> usize {
        let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        
        // Estimate based on target length calculation
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let target_length = self.config.padding_strategy
            .calculate_target_length(&lengths, self.config.max_length)
            .unwrap_or_else(|| lengths.iter().max().copied().unwrap_or(0));

        // Estimate memory usage: sequences * target_length * (token_size + mask_size)
        let estimated_tokens = sequences.len() * target_length;
        let token_bytes = estimated_tokens * std::mem::size_of::<u32>();
        let mask_bytes = if self.config.return_attention_mask {
            estimated_tokens * std::mem::size_of::<u8>()
        } else {
            0
        };

        token_bytes + mask_bytes
    }

    /// Create a summary of processing results
    pub fn create_processing_summary(&self, batch: &SequenceBatch) -> ProcessingSummary {
        let sequences = batch.sequences();
        let total_sequences = sequences.len();
        
        let mut total_original_length = 0;
        let mut total_final_length = 0;
        let mut truncated_count = 0;
        let mut padded_count = 0;
        let mut total_truncated_tokens = 0;
        let mut total_padding_added = 0;

        for seq in sequences {
            total_original_length += seq.original_length;
            total_final_length += seq.current_length;
            
            if seq.was_truncated {
                truncated_count += 1;
                total_truncated_tokens += seq.tokens_truncated;
            }
            
            if seq.was_padded {
                padded_count += 1;
                total_padding_added += seq.padding_added;
            }
        }

        ProcessingSummary {
            total_sequences,
            total_original_length,
            total_final_length,
            truncated_count,
            padded_count,
            total_truncated_tokens,
            total_padding_added,
            compression_ratio: if total_original_length > 0 {
                total_final_length as f32 / total_original_length as f32
            } else {
                1.0
            },
            padding_efficiency: if total_final_length > 0 {
                (total_final_length - total_padding_added) as f32 / total_final_length as f32
            } else {
                1.0
            },
        }
    }
}

impl Default for SequenceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of sequence processing results
#[derive(Debug, Clone)]
pub struct ProcessingSummary {
    pub total_sequences: usize,
    pub total_original_length: usize,
    pub total_final_length: usize,
    pub truncated_count: usize,
    pub padded_count: usize,
    pub total_truncated_tokens: usize,
    pub total_padding_added: usize,
    pub compression_ratio: f32,
    pub padding_efficiency: f32,
}

impl ProcessingSummary {
    /// Get the average original sequence length
    pub fn avg_original_length(&self) -> f32 {
        if self.total_sequences > 0 {
            self.total_original_length as f32 / self.total_sequences as f32
        } else {
            0.0
        }
    }

    /// Get the average final sequence length
    pub fn avg_final_length(&self) -> f32 {
        if self.total_sequences > 0 {
            self.total_final_length as f32 / self.total_sequences as f32
        } else {
            0.0
        }
    }

    /// Get the percentage of sequences that were truncated
    pub fn truncation_rate(&self) -> f32 {
        if self.total_sequences > 0 {
            self.truncated_count as f32 / self.total_sequences as f32
        } else {
            0.0
        }
    }

    /// Get the percentage of sequences that were padded
    pub fn padding_rate(&self) -> f32 {
        if self.total_sequences > 0 {
            self.padded_count as f32 / self.total_sequences as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{PaddingStrategy, TruncationStrategy};

    #[test]
    fn test_sequence_manager_creation() {
        let manager = SequenceManager::new();
        assert_eq!(manager.config().max_length, Some(512));
        assert_eq!(manager.config().padding_strategy, PaddingStrategy::LongestInBatch);
    }

    #[test]
    fn test_sequence_manager_builder() {
        let manager = SequenceManager::new()
            .with_max_length(256)
            .with_padding_strategy(PaddingStrategy::MaxLength)
            .with_pad_token_id(1);

        assert_eq!(manager.config().max_length, Some(256));
        assert_eq!(manager.config().padding_strategy, PaddingStrategy::MaxLength);
        assert_eq!(manager.config().pad_token_id, Some(1));
    }

    #[test]
    fn test_estimate_memory_usage() {
        let manager = SequenceManager::new()
            .with_max_length(10)
            .with_padding_strategy(PaddingStrategy::MaxLength);

        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9],
        ];

        let memory_usage = manager.estimate_memory_usage(&sequences);
        
        // 3 sequences * 10 tokens * (4 bytes for u32 + 1 byte for mask)
        let expected = 3 * 10 * (4 + 1);
        assert_eq!(memory_usage, expected);
    }
}