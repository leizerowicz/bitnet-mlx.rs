//! Sequence Truncation Utilities
//!
//! This module provides functions for truncating sequences to fit within
//! specified length constraints, supporting various truncation strategies.

use super::{TruncationStrategy, SequenceError, SequenceResult};
use serde::{Deserialize, Serialize};

/// Options for sequence truncation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationOptions {
    /// Truncation strategy to use
    pub strategy: TruncationStrategy,
    /// Maximum allowed length
    pub max_length: usize,
}

impl TruncationOptions {
    /// Create new truncation options
    pub fn new(strategy: TruncationStrategy, max_length: usize) -> Self {
        Self {
            strategy,
            max_length,
        }
    }
}

/// Truncate a single sequence according to the specified options
///
/// # Arguments
/// * `sequence` - The input sequence to truncate
/// * `options` - Truncation configuration options
///
/// # Returns
/// The truncated sequence
///
/// # Example
/// ```rust
/// use bitnet_core::sequence::truncation::{truncate_sequence, TruncationOptions};
/// use bitnet_core::sequence::config::TruncationStrategy;
///
/// let sequence = vec![1, 2, 3, 4, 5, 6];
/// let options = TruncationOptions::new(TruncationStrategy::TruncateRight, 4);
/// let truncated = truncate_sequence(sequence, &options).unwrap();
/// assert_eq!(truncated, vec![1, 2, 3, 4]);
/// ```
pub fn truncate_sequence(sequence: Vec<u32>, options: &TruncationOptions) -> SequenceResult<Vec<u32>> {
    if sequence.len() <= options.max_length {
        return Ok(sequence);
    }

    match options.strategy {
        TruncationStrategy::None => Ok(sequence),
        TruncationStrategy::TruncateLeft => {
            let start_idx = sequence.len() - options.max_length;
            Ok(sequence[start_idx..].to_vec())
        }
        TruncationStrategy::TruncateRight => {
            Ok(sequence[..options.max_length].to_vec())
        }
        TruncationStrategy::LongestFirst => {
            // For single sequence, this is equivalent to TruncateRight
            Ok(sequence[..options.max_length].to_vec())
        }
        TruncationStrategy::OnlyIfExceeds => {
            if sequence.len() > options.max_length {
                Ok(sequence[..options.max_length].to_vec())
            } else {
                Ok(sequence)
            }
        }
    }
}

/// Truncate multiple sequences according to the specified strategy
///
/// # Arguments
/// * `sequences` - Vector of sequences to truncate
/// * `options` - Truncation configuration options
///
/// # Returns
/// Vector of truncated sequences
pub fn truncate_sequences(
    sequences: Vec<Vec<u32>>,
    options: &TruncationOptions,
) -> SequenceResult<Vec<Vec<u32>>> {
    match options.strategy {
        TruncationStrategy::LongestFirst => {
            truncate_longest_first(sequences, options.max_length)
        }
        _ => {
            // Apply the same truncation strategy to each sequence individually
            let mut truncated_sequences = Vec::with_capacity(sequences.len());
            for sequence in sequences {
                let truncated = truncate_sequence(sequence, options)?;
                truncated_sequences.push(truncated);
            }
            Ok(truncated_sequences)
        }
    }
}

/// Truncate sequences using the "longest first" strategy
///
/// This strategy iteratively truncates the longest sequence in the batch
/// until all sequences fit within the maximum length constraint.
///
/// # Arguments
/// * `sequences` - Vector of sequences to truncate
/// * `max_length` - Maximum allowed length for any sequence
///
/// # Returns
/// Vector of truncated sequences
pub fn truncate_longest_first(
    mut sequences: Vec<Vec<u32>>,
    max_length: usize,
) -> SequenceResult<Vec<Vec<u32>>> {
    loop {
        // Find the longest sequence
        let (longest_idx, longest_len) = sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| (i, seq.len()))
            .max_by_key(|(_, len)| *len)
            .unwrap_or((0, 0));

        // If the longest sequence is within the limit, we're done
        if longest_len <= max_length {
            break;
        }

        // Truncate the longest sequence by one token from the right
        if !sequences[longest_idx].is_empty() {
            sequences[longest_idx].pop();
        }
    }

    Ok(sequences)
}

/// Truncate a sequence from the left (beginning)
///
/// # Arguments
/// * `sequence` - The input sequence to truncate
/// * `max_length` - Maximum allowed length
///
/// # Returns
/// The left-truncated sequence
pub fn truncate_left(sequence: Vec<u32>, max_length: usize) -> Vec<u32> {
    if sequence.len() <= max_length {
        sequence
    } else {
        let start_idx = sequence.len() - max_length;
        sequence[start_idx..].to_vec()
    }
}

/// Truncate a sequence from the right (end)
///
/// # Arguments
/// * `sequence` - The input sequence to truncate
/// * `max_length` - Maximum allowed length
///
/// # Returns
/// The right-truncated sequence
pub fn truncate_right(sequence: Vec<u32>, max_length: usize) -> Vec<u32> {
    if sequence.len() <= max_length {
        sequence
    } else {
        sequence[..max_length].to_vec()
    }
}

/// Truncate a sequence from both ends to fit within the maximum length
///
/// # Arguments
/// * `sequence` - The input sequence to truncate
/// * `max_length` - Maximum allowed length
/// * `left_ratio` - Ratio of truncation from the left (0.0 = all from right, 1.0 = all from left)
///
/// # Returns
/// The truncated sequence
pub fn truncate_both_ends(sequence: Vec<u32>, max_length: usize, left_ratio: f32) -> SequenceResult<Vec<u32>> {
    if sequence.len() <= max_length {
        return Ok(sequence);
    }

    if left_ratio < 0.0 || left_ratio > 1.0 {
        return Err(SequenceError::ConfigError {
            message: "Left ratio must be between 0.0 and 1.0".to_string(),
        });
    }

    let tokens_to_remove = sequence.len() - max_length;
    let left_remove = (tokens_to_remove as f32 * left_ratio).round() as usize;
    let right_remove = tokens_to_remove - left_remove;

    let start_idx = left_remove;
    let end_idx = sequence.len() - right_remove;

    Ok(sequence[start_idx..end_idx].to_vec())
}

/// Truncate sequences to fit within a total token budget
///
/// This function distributes the available token budget across all sequences,
/// truncating them proportionally to their original lengths.
///
/// # Arguments
/// * `sequences` - Vector of sequences to truncate
/// * `total_budget` - Total number of tokens allowed across all sequences
///
/// # Returns
/// Vector of truncated sequences that fit within the budget
pub fn truncate_to_budget(
    sequences: Vec<Vec<u32>>,
    total_budget: usize,
) -> SequenceResult<Vec<Vec<u32>>> {
    if sequences.is_empty() {
        return Ok(sequences);
    }

    let total_original_tokens: usize = sequences.iter().map(|s| s.len()).sum();
    
    if total_original_tokens <= total_budget {
        return Ok(sequences);
    }

    let mut truncated_sequences = Vec::with_capacity(sequences.len());
    let mut remaining_budget = total_budget;

    for (i, sequence) in sequences.iter().enumerate() {
        let remaining_sequences = sequences.len() - i;
        
        // Calculate proportional allocation for this sequence
        let sequence_budget = if remaining_sequences == 1 {
            // Last sequence gets all remaining budget
            remaining_budget
        } else {
            // Proportional allocation based on original length
            let proportion = sequence.len() as f32 / total_original_tokens as f32;
            ((total_budget as f32 * proportion).round() as usize).min(remaining_budget)
        };

        let truncated_length = sequence_budget.min(sequence.len());
        let truncated = truncate_right(sequence.clone(), truncated_length);
        
        remaining_budget = remaining_budget.saturating_sub(truncated.len());
        truncated_sequences.push(truncated);
    }

    Ok(truncated_sequences)
}

/// Calculate truncation statistics for a batch of sequences
#[derive(Debug, Clone)]
pub struct TruncationStats {
    pub total_sequences: usize,
    pub truncated_sequences: usize,
    pub total_original_tokens: usize,
    pub total_final_tokens: usize,
    pub total_tokens_removed: usize,
    pub avg_original_length: f32,
    pub avg_final_length: f32,
    pub avg_tokens_removed: f32,
    pub truncation_rate: f32, // percentage of sequences truncated
    pub compression_ratio: f32, // final tokens / original tokens
}

impl TruncationStats {
    /// Calculate truncation statistics for sequences before and after truncation
    pub fn calculate(original_sequences: &[Vec<u32>], truncated_sequences: &[Vec<u32>]) -> Self {
        let total_sequences = original_sequences.len();
        let total_original_tokens: usize = original_sequences.iter().map(|s| s.len()).sum();
        let total_final_tokens: usize = truncated_sequences.iter().map(|s| s.len()).sum();
        let total_tokens_removed = total_original_tokens - total_final_tokens;
        
        let truncated_sequences = original_sequences
            .iter()
            .zip(truncated_sequences.iter())
            .filter(|(orig, trunc)| orig.len() != trunc.len())
            .count();
        
        let avg_original_length = if total_sequences > 0 {
            total_original_tokens as f32 / total_sequences as f32
        } else {
            0.0
        };
        
        let avg_final_length = if total_sequences > 0 {
            total_final_tokens as f32 / total_sequences as f32
        } else {
            0.0
        };
        
        let avg_tokens_removed = if total_sequences > 0 {
            total_tokens_removed as f32 / total_sequences as f32
        } else {
            0.0
        };
        
        let truncation_rate = if total_sequences > 0 {
            truncated_sequences as f32 / total_sequences as f32
        } else {
            0.0
        };
        
        let compression_ratio = if total_original_tokens > 0 {
            total_final_tokens as f32 / total_original_tokens as f32
        } else {
            1.0
        };
        
        Self {
            total_sequences,
            truncated_sequences,
            total_original_tokens,
            total_final_tokens,
            total_tokens_removed,
            avg_original_length,
            avg_final_length,
            avg_tokens_removed,
            truncation_rate,
            compression_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::TruncationStrategy;

    #[test]
    fn test_truncate_right() {
        let sequence = vec![1, 2, 3, 4, 5, 6];
        let options = TruncationOptions::new(TruncationStrategy::TruncateRight, 4);
        let truncated = truncate_sequence(sequence, &options).unwrap();
        assert_eq!(truncated, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_truncate_left() {
        let sequence = vec![1, 2, 3, 4, 5, 6];
        let options = TruncationOptions::new(TruncationStrategy::TruncateLeft, 4);
        let truncated = truncate_sequence(sequence, &options).unwrap();
        assert_eq!(truncated, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_truncate_no_truncation_needed() {
        let sequence = vec![1, 2, 3];
        let options = TruncationOptions::new(TruncationStrategy::TruncateRight, 5);
        let truncated = truncate_sequence(sequence.clone(), &options).unwrap();
        assert_eq!(truncated, sequence);
    }

    #[test]
    fn test_truncate_longest_first() {
        let sequences = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8], // longest
            vec![9, 10, 11, 12],           // medium
            vec![13, 14],                  // shortest
        ];
        
        let truncated = truncate_longest_first(sequences, 5).unwrap();
        
        // The longest sequence should be truncated to 5
        assert_eq!(truncated[0].len(), 5);
        assert_eq!(truncated[1], vec![9, 10, 11, 12]);
        assert_eq!(truncated[2], vec![13, 14]);
    }

    #[test]
    fn test_truncate_both_ends() {
        let sequence = vec![1, 2, 3, 4, 5, 6, 7, 8];
        
        // Remove 4 tokens: 2 from left, 2 from right
        let truncated = truncate_both_ends(sequence, 4, 0.5).unwrap();
        assert_eq!(truncated, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_truncate_to_budget() {
        let sequences = vec![
            vec![1, 2, 3, 4, 5, 6], // 6 tokens
            vec![7, 8, 9, 10],      // 4 tokens
            vec![11, 12],           // 2 tokens
        ];
        // Total: 12 tokens, budget: 8 tokens
        
        let truncated = truncate_to_budget(sequences, 8).unwrap();
        
        let total_final: usize = truncated.iter().map(|s| s.len()).sum();
        assert!(total_final <= 8);
    }

    #[test]
    fn test_truncation_stats() {
        let original = vec![
            vec![1, 2, 3, 4, 5, 6],
            vec![7, 8, 9, 10],
            vec![11, 12],
        ];
        
        let truncated = vec![
            vec![1, 2, 3, 4],  // truncated by 2
            vec![7, 8, 9, 10], // not truncated
            vec![11, 12],      // not truncated
        ];
        
        let stats = TruncationStats::calculate(&original, &truncated);
        
        assert_eq!(stats.total_sequences, 3);
        assert_eq!(stats.truncated_sequences, 1);
        assert_eq!(stats.total_original_tokens, 12); // 6 + 4 + 2
        assert_eq!(stats.total_final_tokens, 10);    // 4 + 4 + 2
        assert_eq!(stats.total_tokens_removed, 2);
        assert_eq!(stats.truncation_rate, 1.0 / 3.0);
        assert_eq!(stats.compression_ratio, 10.0 / 12.0);
    }

    #[test]
    fn test_truncate_both_ends_invalid_ratio() {
        let sequence = vec![1, 2, 3, 4, 5];
        let result = truncate_both_ends(sequence, 3, 1.5); // Invalid ratio > 1.0
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_sequences_multiple() {
        let sequences = vec![
            vec![1, 2, 3, 4, 5, 6],
            vec![7, 8, 9, 10, 11],
            vec![12, 13],
        ];
        
        let options = TruncationOptions::new(TruncationStrategy::TruncateRight, 4);
        let truncated = truncate_sequences(sequences, &options).unwrap();
        
        assert_eq!(truncated[0], vec![1, 2, 3, 4]);
        assert_eq!(truncated[1], vec![7, 8, 9, 10]);
        assert_eq!(truncated[2], vec![12, 13]); // No truncation needed
    }
}