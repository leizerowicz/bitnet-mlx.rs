//! Sequence Padding Utilities
//!
//! This module provides functions for padding sequences to uniform lengths,
//! supporting various padding strategies and configurations.

use super::{SequenceError, SequenceResult};
use serde::{Deserialize, Serialize};

/// Options for sequence padding
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PaddingOptions {
    /// Token ID to use for padding
    pub pad_token: u32,
    /// Target length to pad to
    pub target_length: usize,
    /// Optional: pad to multiple of this value
    pub pad_to_multiple: Option<usize>,
}

impl PaddingOptions {
    /// Create new padding options
    pub fn new(pad_token: u32, target_length: usize) -> Self {
        Self {
            pad_token,
            target_length,
            pad_to_multiple: None,
        }
    }

    /// Set padding to multiple of given value
    pub fn with_multiple(mut self, multiple: usize) -> Self {
        self.pad_to_multiple = Some(multiple);
        self
    }

    /// Calculate the actual target length considering multiple constraint
    pub fn calculate_target_length(&self, current_length: usize) -> usize {
        let base_target = self.target_length.max(current_length);

        if let Some(multiple) = self.pad_to_multiple {
            if multiple > 0 {
                ((base_target + multiple - 1) / multiple) * multiple
            } else {
                base_target
            }
        } else {
            base_target
        }
    }
}

/// Pad a single sequence to the specified length
///
/// # Arguments
/// * `sequence` - The input sequence to pad
/// * `options` - Padding configuration options
///
/// # Returns
/// The padded sequence
///
/// # Example
/// ```rust
/// use bitnet_core::sequence::padding::{pad_sequence, PaddingOptions};
///
/// let sequence = vec![1, 2, 3];
/// let options = PaddingOptions::new(0, 5);
/// let padded = pad_sequence(sequence, &options).unwrap();
/// assert_eq!(padded, vec![1, 2, 3, 0, 0]);
/// ```
pub fn pad_sequence(sequence: Vec<u32>, options: &PaddingOptions) -> SequenceResult<Vec<u32>> {
    let current_length = sequence.len();
    let target_length = options.calculate_target_length(current_length);

    if current_length >= target_length {
        return Ok(sequence);
    }

    let mut padded = sequence;
    let padding_needed = target_length - current_length;
    padded.extend(vec![options.pad_token; padding_needed]);

    Ok(padded)
}

/// Pad a sequence to a specific length (convenience function)
///
/// # Arguments
/// * `sequence` - The input sequence to pad
/// * `pad_token` - Token ID to use for padding
/// * `target_length` - Desired final length
///
/// # Returns
/// The padded sequence
pub fn pad_to_length(
    sequence: Vec<u32>,
    pad_token: u32,
    target_length: usize,
) -> SequenceResult<Vec<u32>> {
    let options = PaddingOptions::new(pad_token, target_length);
    pad_sequence(sequence, &options)
}

/// Pad multiple sequences to the same length
///
/// # Arguments
/// * `sequences` - Vector of sequences to pad
/// * `pad_token` - Token ID to use for padding
/// * `target_length` - Optional target length (uses longest sequence if None)
///
/// # Returns
/// Vector of padded sequences, all with the same length
pub fn pad_sequences_to_length(
    sequences: Vec<Vec<u32>>,
    pad_token: u32,
    target_length: Option<usize>,
) -> SequenceResult<Vec<Vec<u32>>> {
    if sequences.is_empty() {
        return Ok(sequences);
    }

    let max_length = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let final_length = target_length.unwrap_or(max_length);

    let mut padded_sequences = Vec::with_capacity(sequences.len());

    for sequence in sequences {
        let padded = pad_to_length(sequence, pad_token, final_length)?;
        padded_sequences.push(padded);
    }

    Ok(padded_sequences)
}

/// Pad sequences to the longest sequence in the batch
///
/// # Arguments
/// * `sequences` - Vector of sequences to pad
/// * `pad_token` - Token ID to use for padding
///
/// # Returns
/// Vector of padded sequences, all with the same length as the longest input
pub fn pad_to_longest(sequences: Vec<Vec<u32>>, pad_token: u32) -> SequenceResult<Vec<Vec<u32>>> {
    pad_sequences_to_length(sequences, pad_token, None)
}

/// Pad sequences to a multiple of a given value
///
/// # Arguments
/// * `sequences` - Vector of sequences to pad
/// * `pad_token` - Token ID to use for padding
/// * `multiple` - Pad to next multiple of this value
///
/// # Returns
/// Vector of padded sequences
pub fn pad_to_multiple(
    sequences: Vec<Vec<u32>>,
    pad_token: u32,
    multiple: usize,
) -> SequenceResult<Vec<Vec<u32>>> {
    if multiple == 0 {
        return Err(SequenceError::ConfigError {
            message: "Multiple cannot be zero".to_string(),
        });
    }

    let mut padded_sequences = Vec::with_capacity(sequences.len());

    for sequence in sequences {
        let current_length = sequence.len();
        let target_length = ((current_length + multiple - 1) / multiple) * multiple;

        let padded = pad_to_length(sequence, pad_token, target_length)?;
        padded_sequences.push(padded);
    }

    Ok(padded_sequences)
}

/// Left-pad a sequence (add padding at the beginning)
///
/// # Arguments
/// * `sequence` - The input sequence to pad
/// * `pad_token` - Token ID to use for padding
/// * `target_length` - Desired final length
///
/// # Returns
/// The left-padded sequence
pub fn left_pad_sequence(
    sequence: Vec<u32>,
    pad_token: u32,
    target_length: usize,
) -> SequenceResult<Vec<u32>> {
    if sequence.len() >= target_length {
        return Ok(sequence);
    }

    let padding_needed = target_length - sequence.len();
    let mut padded = vec![pad_token; padding_needed];
    padded.extend(sequence);

    Ok(padded)
}

/// Right-pad a sequence (add padding at the end) - same as regular padding
///
/// # Arguments
/// * `sequence` - The input sequence to pad
/// * `pad_token` - Token ID to use for padding
/// * `target_length` - Desired final length
///
/// # Returns
/// The right-padded sequence
pub fn right_pad_sequence(
    sequence: Vec<u32>,
    pad_token: u32,
    target_length: usize,
) -> SequenceResult<Vec<u32>> {
    pad_to_length(sequence, pad_token, target_length)
}

/// Remove padding from a sequence
///
/// # Arguments
/// * `sequence` - The padded sequence
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// The sequence with padding removed from the end
pub fn remove_padding(sequence: Vec<u32>, pad_token: u32) -> Vec<u32> {
    let mut result = sequence;

    // Remove padding tokens from the end
    while let Some(&last) = result.last() {
        if last == pad_token {
            result.pop();
        } else {
            break;
        }
    }

    result
}

/// Remove left padding from a sequence
///
/// # Arguments
/// * `sequence` - The padded sequence
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// The sequence with padding removed from the beginning
pub fn remove_left_padding(sequence: Vec<u32>, pad_token: u32) -> Vec<u32> {
    let mut start_idx = 0;

    // Find first non-padding token
    for (i, &token) in sequence.iter().enumerate() {
        if token != pad_token {
            start_idx = i;
            break;
        }
    }

    sequence[start_idx..].to_vec()
}

/// Calculate padding statistics for a batch of sequences
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PaddingStats {
    pub total_sequences: usize,
    pub total_original_tokens: usize,
    pub total_final_tokens: usize,
    pub total_padding_added: usize,
    pub avg_original_length: f32,
    pub avg_final_length: f32,
    pub avg_padding_per_sequence: f32,
    pub padding_efficiency: f32, // ratio of real tokens to total tokens
}

impl PaddingStats {
    /// Calculate padding statistics for sequences before and after padding
    pub fn calculate(original_sequences: &[Vec<u32>], padded_sequences: &[Vec<u32>]) -> Self {
        let total_sequences = original_sequences.len();
        let total_original_tokens: usize = original_sequences.iter().map(|s| s.len()).sum();
        let total_final_tokens: usize = padded_sequences.iter().map(|s| s.len()).sum();
        let total_padding_added = total_final_tokens - total_original_tokens;

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

        let avg_padding_per_sequence = if total_sequences > 0 {
            total_padding_added as f32 / total_sequences as f32
        } else {
            0.0
        };

        let padding_efficiency = if total_final_tokens > 0 {
            total_original_tokens as f32 / total_final_tokens as f32
        } else {
            1.0
        };

        Self {
            total_sequences,
            total_original_tokens,
            total_final_tokens,
            total_padding_added,
            avg_original_length,
            avg_final_length,
            avg_padding_per_sequence,
            padding_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_sequence_basic() {
        let sequence = vec![1, 2, 3];
        let options = PaddingOptions::new(0, 5);
        let padded = pad_sequence(sequence, &options).unwrap();
        assert_eq!(padded, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn test_pad_sequence_no_padding_needed() {
        let sequence = vec![1, 2, 3, 4, 5];
        let options = PaddingOptions::new(0, 3);
        let padded = pad_sequence(sequence.clone(), &options).unwrap();
        assert_eq!(padded, sequence);
    }

    #[test]
    fn test_pad_to_multiple() {
        let sequences = vec![
            vec![1, 2, 3],       // length 3 -> pad to 4
            vec![4, 5, 6, 7, 8], // length 5 -> pad to 8
            vec![9, 10],         // length 2 -> pad to 4
        ];

        let padded = pad_to_multiple(sequences, 0, 4).unwrap();

        assert_eq!(padded[0], vec![1, 2, 3, 0]);
        assert_eq!(padded[1], vec![4, 5, 6, 7, 8, 0, 0, 0]);
        assert_eq!(padded[2], vec![9, 10, 0, 0]);
    }

    #[test]
    fn test_left_pad_sequence() {
        let sequence = vec![1, 2, 3];
        let padded = left_pad_sequence(sequence, 0, 5).unwrap();
        assert_eq!(padded, vec![0, 0, 1, 2, 3]);
    }

    #[test]
    fn test_remove_padding() {
        let sequence = vec![1, 2, 3, 0, 0, 0];
        let unpadded = remove_padding(sequence, 0);
        assert_eq!(unpadded, vec![1, 2, 3]);
    }

    #[test]
    fn test_remove_left_padding() {
        let sequence = vec![0, 0, 1, 2, 3];
        let unpadded = remove_left_padding(sequence, 0);
        assert_eq!(unpadded, vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_sequences_to_longest() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8], vec![9, 10]];

        let padded = pad_to_longest(sequences, 0).unwrap();

        assert_eq!(padded[0], vec![1, 2, 3, 0, 0]);
        assert_eq!(padded[1], vec![4, 5, 6, 7, 8]);
        assert_eq!(padded[2], vec![9, 10, 0, 0, 0]);
    }

    #[test]
    fn test_padding_options_with_multiple() {
        let options = PaddingOptions::new(0, 10).with_multiple(8);

        // Current length 5, target 10, multiple 8 -> should pad to 16
        assert_eq!(options.calculate_target_length(5), 16);

        // Current length 12, target 10, multiple 8 -> should pad to 16
        assert_eq!(options.calculate_target_length(12), 16);
    }

    #[test]
    fn test_padding_stats() {
        let original = vec![vec![1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]];

        let padded = vec![vec![1, 2, 3, 0], vec![4, 5, 6, 7], vec![8, 9, 0, 0]];

        let stats = PaddingStats::calculate(&original, &padded);

        assert_eq!(stats.total_sequences, 3);
        assert_eq!(stats.total_original_tokens, 9); // 3 + 4 + 2
        assert_eq!(stats.total_final_tokens, 12); // 4 + 4 + 4
        assert_eq!(stats.total_padding_added, 3);
        assert_eq!(stats.avg_original_length, 3.0);
        assert_eq!(stats.avg_final_length, 4.0);
        assert_eq!(stats.padding_efficiency, 0.75); // 9/12
    }
}
