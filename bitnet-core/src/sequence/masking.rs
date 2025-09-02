//! Attention Masking Utilities
//!
//! This module provides functions for generating attention masks for sequences,
//! which are essential for transformer models to distinguish between real tokens
//! and padding tokens during attention computation.

use super::{SequenceError, SequenceResult};
use serde::{Deserialize, Serialize};

/// Type of attention mask to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionMaskType {
    /// Standard attention mask (1 for real tokens, 0 for padding)
    Standard,
    /// Causal mask for autoregressive models (lower triangular)
    Causal,
    /// Bidirectional mask (all real tokens can attend to each other)
    Bidirectional,
    /// Custom mask with specific pattern
    Custom,
}

impl Default for AttentionMaskType {
    fn default() -> Self {
        AttentionMaskType::Standard
    }
}

/// Create a basic attention mask for a sequence
///
/// # Arguments
/// * `sequence` - The tokenized sequence
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// Attention mask where 1 indicates real tokens and 0 indicates padding
///
/// # Example
/// ```rust
/// use bitnet_core::sequence::masking::create_attention_mask;
///
/// let sequence = vec![1, 2, 3, 0, 0]; // 0 is padding token
/// let mask = create_attention_mask(&sequence, 0);
/// assert_eq!(mask, vec![1, 1, 1, 0, 0]);
/// ```
pub fn create_attention_mask(sequence: &[u32], pad_token: u32) -> Vec<u8> {
    sequence
        .iter()
        .map(|&token| if token == pad_token { 0 } else { 1 })
        .collect()
}

/// Create attention masks for a batch of sequences
///
/// # Arguments
/// * `sequences` - Vector of tokenized sequences
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// Vector of attention masks, one for each sequence
pub fn create_batch_attention_masks(sequences: &[Vec<u32>], pad_token: u32) -> Vec<Vec<u8>> {
    sequences
        .iter()
        .map(|sequence| create_attention_mask(sequence, pad_token))
        .collect()
}

/// Create a padding mask (inverse of attention mask)
///
/// # Arguments
/// * `sequence` - The tokenized sequence
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// Padding mask where 1 indicates padding tokens and 0 indicates real tokens
pub fn create_padding_mask(sequence: &[u32], pad_token: u32) -> Vec<u8> {
    sequence
        .iter()
        .map(|&token| if token == pad_token { 1 } else { 0 })
        .collect()
}

/// Create a causal attention mask for autoregressive models
///
/// # Arguments
/// * `sequence_length` - Length of the sequence
///
/// # Returns
/// 2D causal mask where mask[i][j] = 1 if position i can attend to position j
pub fn create_causal_mask(sequence_length: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; sequence_length]; sequence_length];

    for i in 0..sequence_length {
        for j in 0..=i {
            mask[i][j] = 1;
        }
    }

    mask
}

/// Create a bidirectional attention mask
///
/// # Arguments
/// * `attention_mask` - Basic attention mask (1 for real tokens, 0 for padding)
///
/// # Returns
/// 2D bidirectional mask where real tokens can attend to all other real tokens
pub fn create_bidirectional_mask(attention_mask: &[u8]) -> Vec<Vec<u8>> {
    let seq_len = attention_mask.len();
    let mut mask = vec![vec![0; seq_len]; seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            // Can attend if both positions are real tokens
            if attention_mask[i] == 1 && attention_mask[j] == 1 {
                mask[i][j] = 1;
            }
        }
    }

    mask
}

/// Create a combined causal and padding mask
///
/// # Arguments
/// * `sequence` - The tokenized sequence
/// * `pad_token` - Token ID used for padding
///
/// # Returns
/// 2D mask combining causal attention with padding awareness
pub fn create_causal_padding_mask(sequence: &[u32], pad_token: u32) -> Vec<Vec<u8>> {
    let attention_mask = create_attention_mask(sequence, pad_token);
    let causal_mask = create_causal_mask(sequence.len());

    // Combine causal mask with attention mask
    let mut combined_mask = vec![vec![0; sequence.len()]; sequence.len()];

    for i in 0..sequence.len() {
        for j in 0..sequence.len() {
            // Can attend if causal constraint is satisfied AND both positions are real tokens
            if causal_mask[i][j] == 1 && attention_mask[i] == 1 && attention_mask[j] == 1 {
                combined_mask[i][j] = 1;
            }
        }
    }

    combined_mask
}

/// Create attention masks for a batch with different mask types
///
/// # Arguments
/// * `sequences` - Vector of tokenized sequences
/// * `pad_token` - Token ID used for padding
/// * `mask_type` - Type of attention mask to create
///
/// # Returns
/// Vector of 2D attention masks
pub fn create_batch_masks(
    sequences: &[Vec<u32>],
    pad_token: u32,
    mask_type: AttentionMaskType,
) -> SequenceResult<Vec<Vec<Vec<u8>>>> {
    let mut batch_masks = Vec::with_capacity(sequences.len());

    for sequence in sequences {
        let mask = match mask_type {
            AttentionMaskType::Standard => {
                let attention_mask = create_attention_mask(sequence, pad_token);
                vec![attention_mask] // Convert 1D to 2D for consistency
            }
            AttentionMaskType::Causal => create_causal_padding_mask(sequence, pad_token),
            AttentionMaskType::Bidirectional => {
                let attention_mask = create_attention_mask(sequence, pad_token);
                create_bidirectional_mask(&attention_mask)
            }
            AttentionMaskType::Custom => {
                return Err(SequenceError::ConfigError {
                    message: "Custom mask type requires specific implementation".to_string(),
                });
            }
        };

        batch_masks.push(mask);
    }

    Ok(batch_masks)
}

/// Create a mask for specific token positions
///
/// # Arguments
/// * `sequence_length` - Length of the sequence
/// * `masked_positions` - Positions to mask (0-indexed)
///
/// # Returns
/// Mask where 1 indicates positions that should be masked
pub fn create_position_mask(
    sequence_length: usize,
    masked_positions: &[usize],
) -> SequenceResult<Vec<u8>> {
    let mut mask = vec![0; sequence_length];

    for &pos in masked_positions {
        if pos >= sequence_length {
            return Err(SequenceError::ConfigError {
                message: format!(
                    "Position {} exceeds sequence length {}",
                    pos, sequence_length
                ),
            });
        }
        mask[pos] = 1;
    }

    Ok(mask)
}

/// Create a sliding window attention mask
///
/// # Arguments
/// * `sequence_length` - Length of the sequence
/// * `window_size` - Size of the attention window
///
/// # Returns
/// 2D mask where each position can only attend to tokens within the window
pub fn create_sliding_window_mask(sequence_length: usize, window_size: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; sequence_length]; sequence_length];

    for i in 0..sequence_length {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(sequence_length);

        for j in start..end {
            mask[i][j] = 1;
        }
    }

    mask
}

/// Create a block diagonal attention mask
///
/// # Arguments
/// * `sequence_length` - Length of the sequence
/// * `block_size` - Size of each attention block
///
/// # Returns
/// 2D mask with block diagonal structure
pub fn create_block_diagonal_mask(sequence_length: usize, block_size: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; sequence_length]; sequence_length];

    for i in 0..sequence_length {
        let block_start = (i / block_size) * block_size;
        let block_end = (block_start + block_size).min(sequence_length);

        for j in block_start..block_end {
            mask[i][j] = 1;
        }
    }

    mask
}

/// Combine multiple masks using logical operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskCombineOp {
    And,
    Or,
    Xor,
}

/// Combine two attention masks
///
/// # Arguments
/// * `mask1` - First attention mask
/// * `mask2` - Second attention mask
/// * `op` - Logical operation to apply
///
/// # Returns
/// Combined attention mask
pub fn combine_masks(
    mask1: &[Vec<u8>],
    mask2: &[Vec<u8>],
    op: MaskCombineOp,
) -> SequenceResult<Vec<Vec<u8>>> {
    if mask1.len() != mask2.len() {
        return Err(SequenceError::InconsistentBatchLengths);
    }

    let mut combined = vec![vec![0; mask1[0].len()]; mask1.len()];

    for i in 0..mask1.len() {
        if mask1[i].len() != mask2[i].len() {
            return Err(SequenceError::InconsistentBatchLengths);
        }

        for j in 0..mask1[i].len() {
            combined[i][j] = match op {
                MaskCombineOp::And => mask1[i][j] & mask2[i][j],
                MaskCombineOp::Or => mask1[i][j] | mask2[i][j],
                MaskCombineOp::Xor => mask1[i][j] ^ mask2[i][j],
            };
        }
    }

    Ok(combined)
}

/// Mask statistics for analysis
#[derive(Debug, Clone)]
pub struct MaskStats {
    pub total_positions: usize,
    pub masked_positions: usize,
    pub unmasked_positions: usize,
    pub mask_ratio: f32,
    pub sparsity: f32, // For 2D masks: ratio of 0s to total elements
}

impl MaskStats {
    /// Calculate statistics for a 1D mask
    pub fn from_1d_mask(mask: &[u8]) -> Self {
        let total_positions = mask.len();
        let masked_positions = mask.iter().filter(|&&x| x == 1).count();
        let unmasked_positions = total_positions - masked_positions;
        let mask_ratio = if total_positions > 0 {
            masked_positions as f32 / total_positions as f32
        } else {
            0.0
        };

        Self {
            total_positions,
            masked_positions,
            unmasked_positions,
            mask_ratio,
            sparsity: 1.0 - mask_ratio,
        }
    }

    /// Calculate statistics for a 2D mask
    pub fn from_2d_mask(mask: &[Vec<u8>]) -> Self {
        let total_positions: usize = mask.iter().map(|row| row.len()).sum();
        let masked_positions: usize = mask
            .iter()
            .map(|row| row.iter().filter(|&&x| x == 1).count())
            .sum();
        let unmasked_positions = total_positions - masked_positions;
        let mask_ratio = if total_positions > 0 {
            masked_positions as f32 / total_positions as f32
        } else {
            0.0
        };

        Self {
            total_positions,
            masked_positions,
            unmasked_positions,
            mask_ratio,
            sparsity: 1.0 - mask_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_attention_mask() {
        let sequence = vec![1, 2, 3, 0, 0];
        let mask = create_attention_mask(&sequence, 0);
        assert_eq!(mask, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_create_padding_mask() {
        let sequence = vec![1, 2, 3, 0, 0];
        let mask = create_padding_mask(&sequence, 0);
        assert_eq!(mask, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_create_causal_mask() {
        let mask = create_causal_mask(4);
        let expected = vec![
            vec![1, 0, 0, 0],
            vec![1, 1, 0, 0],
            vec![1, 1, 1, 0],
            vec![1, 1, 1, 1],
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_create_bidirectional_mask() {
        let attention_mask = vec![1, 1, 1, 0]; // Last token is padding
        let mask = create_bidirectional_mask(&attention_mask);

        // Real tokens (0, 1, 2) should be able to attend to each other
        assert_eq!(mask[0][0], 1);
        assert_eq!(mask[0][1], 1);
        assert_eq!(mask[0][2], 1);
        assert_eq!(mask[0][3], 0); // Cannot attend to padding

        assert_eq!(mask[3][0], 0); // Padding cannot attend to anything
        assert_eq!(mask[3][3], 0);
    }

    #[test]
    fn test_create_sliding_window_mask() {
        let mask = create_sliding_window_mask(5, 3);

        // Position 2 should attend to positions 1, 2, 3
        assert_eq!(mask[2][0], 0);
        assert_eq!(mask[2][1], 1);
        assert_eq!(mask[2][2], 1);
        assert_eq!(mask[2][3], 1);
        assert_eq!(mask[2][4], 0);
    }

    #[test]
    fn test_create_block_diagonal_mask() {
        let mask = create_block_diagonal_mask(6, 2);

        // Positions 0,1 should attend to each other
        assert_eq!(mask[0][0], 1);
        assert_eq!(mask[0][1], 1);
        assert_eq!(mask[0][2], 0);

        // Positions 2,3 should attend to each other
        assert_eq!(mask[2][2], 1);
        assert_eq!(mask[2][3], 1);
        assert_eq!(mask[2][0], 0);
    }

    #[test]
    fn test_combine_masks() {
        let mask1 = vec![vec![1, 1, 0], vec![1, 1, 1], vec![0, 1, 1]];

        let mask2 = vec![vec![1, 0, 1], vec![0, 1, 1], vec![1, 1, 0]];

        let combined = combine_masks(&mask1, &mask2, MaskCombineOp::And).unwrap();
        let expected = vec![vec![1, 0, 0], vec![0, 1, 1], vec![0, 1, 0]];

        assert_eq!(combined, expected);
    }

    #[test]
    fn test_mask_stats_1d() {
        let mask = vec![1, 1, 0, 0, 1];
        let stats = MaskStats::from_1d_mask(&mask);

        assert_eq!(stats.total_positions, 5);
        assert_eq!(stats.masked_positions, 3);
        assert_eq!(stats.unmasked_positions, 2);
        assert!((stats.mask_ratio - 0.6).abs() < f32::EPSILON);
        assert!((stats.sparsity - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_create_position_mask() {
        let mask = create_position_mask(5, &[1, 3]).unwrap();
        assert_eq!(mask, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_create_position_mask_invalid() {
        let result = create_position_mask(3, &[5]);
        assert!(result.is_err());
    }
}
