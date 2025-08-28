//! Sequence Batching Utilities
//!
//! This module provides functionality for efficiently batching variable-length
//! sequences with dynamic padding and memory optimization.

use super::{padding::pad_sequences_to_length, ProcessedSequence, SequenceConfig, SequenceResult};
use serde::{Deserialize, Serialize};

/// A batch of processed sequences with uniform length
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SequenceBatch {
    /// The processed sequences in the batch
    sequences: Vec<ProcessedSequence>,
    /// Batch-level metadata
    metadata: BatchMetadata,
}

/// Metadata for a sequence batch
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct BatchMetadata {
    /// Number of sequences in the batch
    pub batch_size: usize,
    /// Maximum sequence length in the batch
    pub max_length: usize,
    /// Minimum sequence length in the batch
    pub min_length: usize,
    /// Average sequence length before processing
    pub avg_original_length: f32,
    /// Average sequence length after processing
    pub avg_final_length: f32,
    /// Total number of tokens in the batch
    pub total_tokens: usize,
    /// Total number of padding tokens added
    pub total_padding: usize,
    /// Padding efficiency (real tokens / total tokens)
    pub padding_efficiency: f32,
    /// Whether all sequences have the same length
    pub uniform_length: bool,
    /// Timestamp when batch was created
    pub created_at: u64,
}

impl SequenceBatch {
    /// Create a new sequence batch
    pub fn new(sequences: Vec<ProcessedSequence>) -> Self {
        let metadata = BatchMetadata::from_sequences(&sequences);
        Self {
            sequences,
            metadata,
        }
    }

    /// Get the sequences in the batch
    pub fn sequences(&self) -> &[ProcessedSequence] {
        &self.sequences
    }

    /// Get mutable access to sequences
    pub fn sequences_mut(&mut self) -> &mut [ProcessedSequence] {
        &mut self.sequences
    }

    /// Get batch metadata
    pub fn metadata(&self) -> &BatchMetadata {
        &self.metadata
    }

    /// Get the batch size
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get all token sequences as a 2D vector
    pub fn token_ids(&self) -> Vec<Vec<u32>> {
        self.sequences
            .iter()
            .map(|seq| seq.tokens.clone())
            .collect()
    }

    /// Get all attention masks as a 2D vector
    pub fn attention_masks(&self) -> Vec<Vec<u8>> {
        self.sequences
            .iter()
            .map(|seq| seq.attention_mask.clone())
            .collect()
    }

    /// Get the maximum sequence length in the batch
    pub fn max_length(&self) -> usize {
        self.metadata.max_length
    }

    /// Get the minimum sequence length in the batch
    pub fn min_length(&self) -> usize {
        self.metadata.min_length
    }

    /// Check if all sequences have uniform length
    pub fn is_uniform_length(&self) -> bool {
        self.metadata.uniform_length
    }

    /// Get padding efficiency (ratio of real tokens to total tokens)
    pub fn padding_efficiency(&self) -> f32 {
        self.metadata.padding_efficiency
    }

    /// Split the batch into smaller batches
    pub fn split(&self, chunk_size: usize) -> Vec<SequenceBatch> {
        if chunk_size == 0 {
            return vec![];
        }

        self.sequences
            .chunks(chunk_size)
            .map(|chunk| SequenceBatch::new(chunk.to_vec()))
            .collect()
    }

    /// Merge multiple batches into one
    pub fn merge(batches: Vec<SequenceBatch>) -> SequenceBatch {
        let mut all_sequences = Vec::new();
        for batch in batches {
            all_sequences.extend(batch.sequences);
        }
        SequenceBatch::new(all_sequences)
    }

    /// Filter sequences based on a predicate
    pub fn filter<F>(&self, predicate: F) -> SequenceBatch
    where
        F: Fn(&ProcessedSequence) -> bool,
    {
        let filtered_sequences: Vec<ProcessedSequence> = self
            .sequences
            .iter()
            .filter(|seq| predicate(seq))
            .cloned()
            .collect();
        SequenceBatch::new(filtered_sequences)
    }

    /// Sort sequences by a key function
    pub fn sort_by<F, K>(&mut self, key_fn: F)
    where
        F: Fn(&ProcessedSequence) -> K,
        K: Ord,
    {
        self.sequences.sort_by_key(key_fn);
        self.metadata = BatchMetadata::from_sequences(&self.sequences);
    }

    /// Get memory usage estimate for the batch
    pub fn memory_usage(&self) -> usize {
        let token_bytes: usize = self
            .sequences
            .iter()
            .map(|seq| seq.tokens.len() * std::mem::size_of::<u32>())
            .sum();

        let mask_bytes: usize = self
            .sequences
            .iter()
            .map(|seq| seq.attention_mask.len() * std::mem::size_of::<u8>())
            .sum();

        token_bytes + mask_bytes
    }
}

impl BatchMetadata {
    /// Create metadata from a collection of sequences
    pub fn from_sequences(sequences: &[ProcessedSequence]) -> Self {
        if sequences.is_empty() {
            return Self::empty();
        }

        let batch_size = sequences.len();
        let lengths: Vec<usize> = sequences.iter().map(|seq| seq.current_length).collect();
        let original_lengths: Vec<usize> =
            sequences.iter().map(|seq| seq.original_length).collect();

        let max_length = lengths.iter().max().copied().unwrap_or(0);
        let min_length = lengths.iter().min().copied().unwrap_or(0);
        let uniform_length = max_length == min_length;

        let total_original_tokens: usize = original_lengths.iter().sum();
        let total_tokens: usize = lengths.iter().sum();
        let total_padding: usize = sequences.iter().map(|seq| seq.padding_added).sum();

        let avg_original_length = total_original_tokens as f32 / batch_size as f32;
        let avg_final_length = total_tokens as f32 / batch_size as f32;

        let padding_efficiency = if total_tokens > 0 {
            (total_tokens - total_padding) as f32 / total_tokens as f32
        } else {
            1.0
        };

        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            batch_size,
            max_length,
            min_length,
            avg_original_length,
            avg_final_length,
            total_tokens,
            total_padding,
            padding_efficiency,
            uniform_length,
            created_at,
        }
    }

    /// Create empty metadata
    fn empty() -> Self {
        Self {
            batch_size: 0,
            max_length: 0,
            min_length: 0,
            avg_original_length: 0.0,
            avg_final_length: 0.0,
            total_tokens: 0,
            total_padding: 0,
            padding_efficiency: 1.0,
            uniform_length: true,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Options for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct BatchingOptions {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Target batch size for optimal processing
    pub target_batch_size: usize,
    /// Whether to sort sequences by length before batching
    pub sort_by_length: bool,
    /// Whether to group similar-length sequences together
    pub group_by_length: bool,
    /// Length tolerance for grouping (sequences within this range are grouped)
    pub length_tolerance: usize,
    /// Whether to drop incomplete batches
    pub drop_incomplete: bool,
    /// Maximum memory usage per batch (in bytes)
    pub max_memory_per_batch: Option<usize>,
}

impl Default for BatchingOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            target_batch_size: 16,
            sort_by_length: true,
            group_by_length: true,
            length_tolerance: 10,
            drop_incomplete: false,
            max_memory_per_batch: None,
        }
    }
}

/// Batch processor for handling sequence batching
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BatchProcessor {
    config: SequenceConfig,
    options: BatchingOptions,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: &SequenceConfig) -> Self {
        Self {
            config: config.clone(),
            options: BatchingOptions::default(),
        }
    }

    /// Create a batch processor with custom options
    pub fn with_options(config: &SequenceConfig, options: BatchingOptions) -> Self {
        Self {
            config: config.clone(),
            options,
        }
    }

    /// Process sequences into batches
    pub fn create_batches(&self, sequences: Vec<Vec<u32>>) -> SequenceResult<Vec<SequenceBatch>> {
        if sequences.is_empty() {
            return Ok(vec![]);
        }

        // Sort by length if requested
        let mut indexed_sequences: Vec<(usize, Vec<u32>)> =
            sequences.into_iter().enumerate().collect();

        if self.options.sort_by_length {
            indexed_sequences.sort_by_key(|(_, seq)| seq.len());
        }

        // Group sequences by similar lengths if requested
        let groups = if self.options.group_by_length {
            self.group_by_length(indexed_sequences)?
        } else {
            vec![indexed_sequences]
        };

        // Create batches from groups
        let mut batches = Vec::new();
        for group in groups {
            let group_batches = self.create_batches_from_group(group)?;
            batches.extend(group_batches);
        }

        // Filter out incomplete batches if requested
        if self.options.drop_incomplete {
            batches.retain(|batch| batch.len() >= self.options.target_batch_size);
        }

        Ok(batches)
    }

    /// Group sequences by similar lengths
    fn group_by_length(
        &self,
        sequences: Vec<(usize, Vec<u32>)>,
    ) -> SequenceResult<Vec<Vec<(usize, Vec<u32>)>>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut current_length_range = None;

        for (idx, sequence) in sequences {
            let seq_len = sequence.len();

            match current_length_range {
                None => {
                    // Start first group
                    current_length_range = Some((seq_len, seq_len + self.options.length_tolerance));
                    current_group.push((idx, sequence));
                }
                Some((min_len, max_len)) => {
                    if seq_len >= min_len && seq_len <= max_len {
                        // Fits in current group
                        current_group.push((idx, sequence));
                    } else {
                        // Start new group
                        if !current_group.is_empty() {
                            groups.push(current_group);
                            current_group = Vec::new();
                        }
                        current_length_range =
                            Some((seq_len, seq_len + self.options.length_tolerance));
                        current_group.push((idx, sequence));
                    }
                }
            }
        }

        // Add the last group
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        Ok(groups)
    }

    /// Create batches from a group of similar-length sequences
    fn create_batches_from_group(
        &self,
        group: Vec<(usize, Vec<u32>)>,
    ) -> SequenceResult<Vec<SequenceBatch>> {
        let mut batches = Vec::new();
        let sequences: Vec<Vec<u32>> = group.into_iter().map(|(_, seq)| seq).collect();

        for chunk in sequences.chunks(self.options.max_batch_size) {
            let batch_sequences = chunk.to_vec();

            // Check memory constraint if specified
            if let Some(max_memory) = self.options.max_memory_per_batch {
                let estimated_memory = self.estimate_batch_memory(&batch_sequences);
                if estimated_memory > max_memory {
                    // Split into smaller batches
                    let smaller_batches =
                        self.split_for_memory_constraint(batch_sequences, max_memory)?;
                    batches.extend(smaller_batches);
                    continue;
                }
            }

            // Process the batch
            let processed_batch = self.process_batch_sequences(batch_sequences)?;
            batches.push(processed_batch);
        }

        Ok(batches)
    }

    /// Process a batch of sequences into a SequenceBatch
    fn process_batch_sequences(&self, sequences: Vec<Vec<u32>>) -> SequenceResult<SequenceBatch> {
        let pad_token = self.config.pad_token_id.unwrap_or(0);

        // Determine target length based on padding strategy
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let target_length = self
            .config
            .padding_strategy
            .calculate_target_length(&lengths, self.config.max_length);

        // Pad sequences to uniform length
        let padded_sequences = if let Some(target_len) = target_length {
            pad_sequences_to_length(sequences.clone(), pad_token, Some(target_len))?
        } else {
            sequences.clone()
        };

        // Create processed sequences
        let mut processed_sequences = Vec::with_capacity(sequences.len());
        for (_, (original, padded)) in sequences.iter().zip(padded_sequences.iter()).enumerate() {
            let attention_mask = if self.config.return_attention_mask {
                super::masking::create_attention_mask(padded, pad_token)
            } else {
                vec![1; padded.len()]
            };

            let padding_added = padded.len() - original.len();
            let processed = ProcessedSequence::new(
                padded.clone(),
                original.len(),
                attention_mask,
                false, // No truncation in batching
                padding_added > 0,
                0,
                padding_added,
            );
            processed_sequences.push(processed);
        }

        Ok(SequenceBatch::new(processed_sequences))
    }

    /// Estimate memory usage for a batch of sequences
    fn estimate_batch_memory(&self, sequences: &[Vec<u32>]) -> usize {
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let target_length = self
            .config
            .padding_strategy
            .calculate_target_length(&lengths, self.config.max_length)
            .unwrap_or_else(|| lengths.iter().max().copied().unwrap_or(0));

        let total_tokens = sequences.len() * target_length;
        let token_bytes = total_tokens * std::mem::size_of::<u32>();
        let mask_bytes = if self.config.return_attention_mask {
            total_tokens * std::mem::size_of::<u8>()
        } else {
            0
        };

        token_bytes + mask_bytes
    }

    /// Split sequences into smaller batches to fit memory constraint
    fn split_for_memory_constraint(
        &self,
        sequences: Vec<Vec<u32>>,
        max_memory: usize,
    ) -> SequenceResult<Vec<SequenceBatch>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        for sequence in sequences {
            current_batch.push(sequence);

            let estimated_memory = self.estimate_batch_memory(&current_batch);
            if estimated_memory > max_memory && current_batch.len() > 1 {
                // Remove the last sequence and process current batch
                let last_sequence = current_batch.pop().unwrap();
                let batch = self.process_batch_sequences(current_batch)?;
                batches.push(batch);

                // Start new batch with the sequence that didn't fit
                current_batch = vec![last_sequence];
            }
        }

        // Process remaining sequences
        if !current_batch.is_empty() {
            let batch = self.process_batch_sequences(current_batch)?;
            batches.push(batch);
        }

        Ok(batches)
    }
}

/// Dynamic batch sampler for efficient training
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DynamicBatchSampler {
    /// Maximum number of tokens per batch
    pub max_tokens_per_batch: usize,
    /// Maximum number of sequences per batch
    pub max_sequences_per_batch: usize,
    /// Whether to shuffle sequences
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
}

impl DynamicBatchSampler {
    /// Create a new dynamic batch sampler
    pub fn new(max_tokens_per_batch: usize, max_sequences_per_batch: usize) -> Self {
        Self {
            max_tokens_per_batch,
            max_sequences_per_batch,
            shuffle: true,
            seed: None,
        }
    }

    /// Sample batches from sequences with dynamic sizing
    pub fn sample_batches(&self, sequences: Vec<Vec<u32>>) -> SequenceResult<Vec<Vec<Vec<u32>>>> {
        let mut sequences = sequences;

        // Shuffle if requested
        if self.shuffle {
            // Simple shuffle implementation
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let seed = self.seed.unwrap_or_else(|| {
                let mut hasher = DefaultHasher::new();
                std::time::SystemTime::now().hash(&mut hasher);
                hasher.finish()
            });

            // Fisher-Yates shuffle with deterministic seed
            for i in (1..sequences.len()).rev() {
                let j = (seed.wrapping_mul(i as u64 + 1) % (i as u64 + 1)) as usize;
                sequences.swap(i, j);
            }
        }

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_tokens = 0;

        for sequence in sequences {
            let seq_len = sequence.len();

            // Check if adding this sequence would exceed limits
            let would_exceed_tokens = current_tokens + seq_len > self.max_tokens_per_batch;
            let would_exceed_sequences = current_batch.len() >= self.max_sequences_per_batch;

            if (would_exceed_tokens || would_exceed_sequences) && !current_batch.is_empty() {
                // Finalize current batch
                batches.push(current_batch);
                current_batch = Vec::new();
                current_tokens = 0;
            }

            current_batch.push(sequence);
            current_tokens += seq_len;
        }

        // Add the last batch if not empty
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::PaddingStrategy;

    #[test]
    fn test_sequence_batch_creation() {
        let sequences = vec![
            ProcessedSequence::new(vec![1, 2, 3, 0], 3, vec![1, 1, 1, 0], false, true, 0, 1),
            ProcessedSequence::new(vec![4, 5, 0, 0], 2, vec![1, 1, 0, 0], false, true, 0, 2),
        ];

        let batch = SequenceBatch::new(sequences);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.max_length(), 4);
        assert_eq!(batch.min_length(), 4);
        assert!(batch.is_uniform_length());
    }

    #[test]
    fn test_batch_split() {
        let sequences = vec![
            ProcessedSequence::new(vec![1, 2], 2, vec![1, 1], false, false, 0, 0),
            ProcessedSequence::new(vec![3, 4], 2, vec![1, 1], false, false, 0, 0),
            ProcessedSequence::new(vec![5, 6], 2, vec![1, 1], false, false, 0, 0),
            ProcessedSequence::new(vec![7, 8], 2, vec![1, 1], false, false, 0, 0),
        ];

        let batch = SequenceBatch::new(sequences);
        let split_batches = batch.split(2);

        assert_eq!(split_batches.len(), 2);
        assert_eq!(split_batches[0].len(), 2);
        assert_eq!(split_batches[1].len(), 2);
    }

    #[test]
    fn test_dynamic_batch_sampler() {
        let sequences = vec![
            vec![1, 2, 3],    // 3 tokens
            vec![4, 5, 6, 7], // 4 tokens
            vec![8, 9],       // 2 tokens
            vec![10, 11, 12], // 3 tokens
        ];

        let sampler = DynamicBatchSampler::new(8, 3); // Max 8 tokens or 3 sequences per batch
        let batches = sampler.sample_batches(sequences).unwrap();

        // Should create batches that respect token and sequence limits
        for batch in &batches {
            let total_tokens: usize = batch.iter().map(|seq| seq.len()).sum();
            assert!(total_tokens <= 8);
            assert!(batch.len() <= 3);
        }
    }

    #[test]
    fn test_batch_processor() {
        let config = SequenceConfig::new()
            .with_padding_strategy(PaddingStrategy::LongestInBatch)
            .with_pad_token_id(0);

        let processor = BatchProcessor::new(&config);
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8], vec![9, 10]];

        let batches = processor.create_batches(sequences).unwrap();
        assert!(!batches.is_empty());

        // All sequences in a batch should have the same length
        for batch in &batches {
            let lengths: Vec<usize> = batch.token_ids().iter().map(|seq| seq.len()).collect();
            let first_length = lengths[0];
            assert!(lengths.iter().all(|&len| len == first_length));
        }
    }
}
