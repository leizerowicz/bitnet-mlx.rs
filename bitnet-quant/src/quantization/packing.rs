//! Ternary weight packing strategies for BitNet models
//!
//! This module provides different strategies for efficiently packing ternary weights
//! {-1, 0, +1} to minimize memory usage and optimize access patterns.

use super::utils::{BitUtils, QuantizationError};
use candle_core::{Device, Shape, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Different packing strategies for ternary weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TernaryPackingStrategy {
    /// No packing - store as full bytes (1 byte per value)
    Uncompressed,
    /// Pack 4 ternary values into 1 byte (2 bits per value)
    BitPacked2Bit,
    /// Pack 5 ternary values into 1 byte using base-3 encoding
    Base3Packed,
    /// Byte-aligned packing with padding for SIMD operations
    ByteAligned,
    /// Run-length encoding for sparse ternary weights
    RunLengthEncoded,
    /// Compressed sparse format for very sparse weights
    CompressedSparse,
    /// Hybrid approach: different strategies for different weight blocks
    Hybrid,
}

impl Default for TernaryPackingStrategy {
    fn default() -> Self {
        Self::BitPacked2Bit
    }
}

/// Configuration for ternary weight packing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryPackingConfig {
    /// Primary packing strategy
    pub strategy: TernaryPackingStrategy,
    /// Block size for block-wise packing
    pub block_size: Option<usize>,
    /// Sparsity threshold for switching to sparse formats
    pub sparsity_threshold: f32,
    /// Whether to use SIMD-optimized layouts
    pub simd_optimized: bool,
    /// Alignment requirements for memory access
    pub alignment: usize,
    /// Whether to enable compression for sparse formats
    pub enable_compression: bool,
}

impl Default for TernaryPackingConfig {
    fn default() -> Self {
        Self {
            strategy: TernaryPackingStrategy::default(),
            block_size: Some(64),    // 64 values per block
            sparsity_threshold: 0.7, // 70% zeros
            simd_optimized: true,
            alignment: 16, // 16-byte alignment for SIMD
            enable_compression: true,
        }
    }
}

/// Packed ternary weight representation
#[derive(Debug, Clone)]
pub struct PackedTernaryWeights {
    /// Packed weight data
    pub data: Vec<u8>,
    /// Original shape of the weight tensor
    pub shape: Shape,
    /// Packing strategy used
    pub strategy: TernaryPackingStrategy,
    /// Packing configuration
    pub config: TernaryPackingConfig,
    /// Metadata for unpacking
    pub metadata: PackingMetadata,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Metadata required for unpacking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PackingMetadata {
    /// Number of elements in original tensor
    pub element_count: usize,
    /// Block sizes for hybrid packing
    pub block_sizes: Option<Vec<usize>>,
    /// Sparse indices for sparse formats
    pub sparse_indices: Option<Vec<usize>>,
    /// Run-length encoding data
    pub rle_data: Option<Vec<(i8, usize)>>, // (value, count) pairs
    /// Padding information
    pub padding: usize,
    /// Additional strategy-specific metadata
    pub extra_data: HashMap<String, Vec<u8>>,
}

/// Trait for ternary weight packing operations
pub trait TernaryPacker: Send + Sync {
    /// Pack ternary weights using the specified strategy
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError>;

    /// Unpack ternary weights back to original format
    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError>;

    /// Estimate memory savings for given weights
    fn estimate_savings(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate;

    /// Check if strategy is suitable for given weight characteristics
    fn is_suitable(&self, weights: &[i8], config: &TernaryPackingConfig) -> bool;

    /// Validate packed data integrity (default implementation)
    fn validate_packed_data(&self, packed: &PackedTernaryWeights) -> Result<(), QuantizationError> {
        // Basic validation - can be overridden by specific packers
        if packed.data.is_empty() && packed.metadata.element_count > 0 {
            return Err(QuantizationError::InvalidInput(
                "Packed data is empty but element count is non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Pack with validation and corruption detection
    fn pack_with_validation(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        // Validate input weights
        self.validate_input_weights(weights)?;

        // Perform packing
        let mut packed = self.pack(weights, config)?;

        // Add integrity checksum if enabled
        if config.enable_compression {
            self.add_integrity_data(&mut packed)?;
        }

        // Validate the packed result
        self.validate_packed_data(&packed)?;

        Ok(packed)
    }

    /// Unpack with corruption detection
    fn unpack_with_validation(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<i8>, QuantizationError> {
        // Validate packed data first
        self.validate_packed_data(packed)?;

        // Check integrity if available
        self.verify_integrity_data(packed)?;

        // Perform unpacking
        let unpacked = self.unpack(packed)?;

        // Validate unpacked result
        self.validate_unpacked_weights(&unpacked, packed)?;

        Ok(unpacked)
    }

    /// Validate input weights before packing
    fn validate_input_weights(&self, weights: &[i8]) -> Result<(), QuantizationError> {
        if weights.is_empty() {
            return Err(QuantizationError::InvalidInput(
                "Input weights are empty".to_string(),
            ));
        }

        // Check for valid ternary values
        for (i, &weight) in weights.iter().enumerate() {
            if !(-1..=1).contains(&weight) {
                return Err(QuantizationError::InvalidInput(format!(
                    "Invalid ternary weight {weight} at position {i}, expected range [-1, 1]"
                )));
            }
        }

        Ok(())
    }

    /// Validate unpacked weights
    fn validate_unpacked_weights(
        &self,
        unpacked: &[i8],
        packed: &PackedTernaryWeights,
    ) -> Result<(), QuantizationError> {
        if unpacked.len() != packed.metadata.element_count {
            return Err(QuantizationError::InvalidInput(format!(
                "Unpacked length {} doesn't match expected {}",
                unpacked.len(),
                packed.metadata.element_count
            )));
        }

        // Check for valid ternary values
        for (i, &weight) in unpacked.iter().enumerate() {
            if !(-1..=1).contains(&weight) {
                return Err(QuantizationError::InvalidInput(format!(
                    "Invalid unpacked ternary weight {weight} at position {i}"
                )));
            }
        }

        Ok(())
    }

    /// Add integrity data (checksum, etc.) to packed weights
    fn add_integrity_data(
        &self,
        packed: &mut PackedTernaryWeights,
    ) -> Result<(), QuantizationError> {
        // Calculate CRC32 checksum
        let checksum = calculate_crc32(&packed.data);
        packed
            .metadata
            .extra_data
            .insert("crc32".to_string(), checksum.to_le_bytes().to_vec());

        // Add packing timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        packed
            .metadata
            .extra_data
            .insert("timestamp".to_string(), timestamp.to_le_bytes().to_vec());

        Ok(())
    }

    /// Verify integrity data in packed weights
    fn verify_integrity_data(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<(), QuantizationError> {
        // Check CRC32 if present
        if let Some(stored_checksum_bytes) = packed.metadata.extra_data.get("crc32") {
            if stored_checksum_bytes.len() == 4 {
                let stored_checksum = u32::from_le_bytes([
                    stored_checksum_bytes[0],
                    stored_checksum_bytes[1],
                    stored_checksum_bytes[2],
                    stored_checksum_bytes[3],
                ]);

                let calculated_checksum = calculate_crc32(&packed.data);

                if stored_checksum != calculated_checksum {
                    return Err(QuantizationError::InvalidInput(
                        format!("Checksum mismatch: expected 0x{stored_checksum:08X}, got 0x{calculated_checksum:08X}")
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Memory savings estimation
#[derive(Debug, Clone)]
pub struct PackingSavingsEstimate {
    pub original_size_bytes: usize,
    pub packed_size_bytes: usize,
    pub compression_ratio: f32,
    pub memory_saved_bytes: usize,
    pub savings_percentage: f32,
    pub access_overhead: f32, // Estimated overhead for unpacking
}

/// Uncompressed packer (baseline)
#[derive(Debug)]
pub struct UncompressedPacker;

impl TernaryPacker for UncompressedPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let data: Vec<u8> = weights.iter().map(|&w| (w + 1) as u8).collect(); // Convert {-1,0,1} to {0,1,2}

        let metadata = PackingMetadata {
            element_count: weights.len(),
            ..Default::default()
        };

        let memory_footprint = data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::Uncompressed,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let weights = packed.data.iter().map(|&b| (b as i8) - 1).collect(); // Convert {0,1,2} back to {-1,0,1}
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        _config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let original_size = weights.len();
        let packed_size = weights.len();

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: packed_size,
            compression_ratio: 1.0,
            memory_saved_bytes: 0,
            savings_percentage: 0.0,
            access_overhead: 0.0,
        }
    }

    fn is_suitable(&self, _weights: &[i8], _config: &TernaryPackingConfig) -> bool {
        true // Always suitable as baseline
    }
}

/// 2-bit packed packer (4 values per byte)
#[derive(Debug)]
pub struct BitPacked2BitPacker;

impl TernaryPacker for BitPacked2BitPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        // Convert ternary values {-1, 0, 1} to 2-bit values {0, 1, 2}
        let converted: Vec<u8> = weights.iter().map(|&w| (w + 1) as u8).collect();

        // Pack 4 values per byte (2 bits each)
        let packed_data = BitUtils::pack_bits(&converted, 2);

        let padding = if weights.len() % 4 != 0 {
            4 - (weights.len() % 4)
        } else {
            0
        };

        let metadata = PackingMetadata {
            element_count: weights.len(),
            padding,
            ..Default::default()
        };

        let memory_footprint = packed_data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data: packed_data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::BitPacked2Bit,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let unpacked = BitUtils::unpack_bits(&packed.data, 2, packed.metadata.element_count);
        let weights = unpacked.iter().map(|&b| (b as i8) - 1).collect(); // Convert back to {-1,0,1}
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        _config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let original_size = weights.len();
        let packed_size = weights.len().div_ceil(4); // 4 values per byte, round up
        let memory_saved = original_size.saturating_sub(packed_size);

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: packed_size,
            compression_ratio: original_size as f32 / packed_size as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: (memory_saved as f32 / original_size as f32) * 100.0,
            access_overhead: 0.1, // Small overhead for bit manipulation
        }
    }

    fn is_suitable(&self, _weights: &[i8], _config: &TernaryPackingConfig) -> bool {
        true // Generally suitable for most cases
    }
}

/// Base-3 packed packer (5 values per byte using base-3 encoding)
#[derive(Debug)]
pub struct Base3PackedPacker;

impl TernaryPacker for Base3PackedPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let mut packed_data = Vec::new();
        let mut i = 0;

        while i < weights.len() {
            let mut byte_value = 0u8;
            let mut multiplier = 1u8;

            // Pack up to 5 ternary values into one byte using base-3
            for j in 0..5 {
                if i + j < weights.len() {
                    let ternary_val = (weights[i + j] + 1) as u8; // Convert {-1,0,1} to {0,1,2}
                    byte_value += ternary_val * multiplier;
                    multiplier *= 3;
                } else {
                    break;
                }
            }

            packed_data.push(byte_value);
            i += 5;
        }

        let padding = if weights.len() % 5 != 0 {
            5 - (weights.len() % 5)
        } else {
            0
        };

        let metadata = PackingMetadata {
            element_count: weights.len(),
            padding,
            ..Default::default()
        };

        let memory_footprint = packed_data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data: packed_data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::Base3Packed,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut weights = Vec::with_capacity(packed.metadata.element_count);

        for &byte_val in &packed.data {
            let mut remaining = byte_val;

            // Unpack up to 5 values from each byte
            for _ in 0..5 {
                if weights.len() >= packed.metadata.element_count {
                    break;
                }

                let ternary_val = remaining % 3;
                weights.push((ternary_val as i8) - 1); // Convert {0,1,2} back to {-1,0,1}
                remaining /= 3;
            }
        }

        weights.truncate(packed.metadata.element_count);
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        _config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let original_size = weights.len();
        let packed_size = weights.len().div_ceil(5); // 5 values per byte, round up
        let memory_saved = original_size.saturating_sub(packed_size);

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: packed_size,
            compression_ratio: original_size as f32 / packed_size as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: (memory_saved as f32 / original_size as f32) * 100.0,
            access_overhead: 0.15, // Slightly higher overhead for base-3 operations
        }
    }

    fn is_suitable(&self, _weights: &[i8], _config: &TernaryPackingConfig) -> bool {
        true // Suitable for dense ternary weights
    }
}

/// Byte-aligned packer with SIMD optimization
#[derive(Debug)]
pub struct ByteAlignedPacker;

impl TernaryPacker for ByteAlignedPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let alignment = config.alignment;
        let values_per_aligned_block = alignment; // 1 byte per value for simplicity

        // Pad to alignment boundary
        let padded_len = weights.len().div_ceil(alignment) * alignment;
        let mut padded_weights = weights.to_vec();
        padded_weights.resize(padded_len, 0); // Pad with zeros

        // Convert to unsigned representation
        let data: Vec<u8> = padded_weights.iter().map(|&w| (w + 1) as u8).collect();

        let padding = padded_len - weights.len();

        let metadata = PackingMetadata {
            element_count: weights.len(),
            padding,
            ..Default::default()
        };

        let memory_footprint = data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::ByteAligned,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let weights: Vec<i8> = packed
            .data
            .iter()
            .take(packed.metadata.element_count)
            .map(|&b| (b as i8) - 1)
            .collect();
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let alignment = config.alignment;
        let original_size = weights.len();
        let aligned_size = weights.len().div_ceil(alignment) * alignment;
        let memory_saved = 0; // No compression, just alignment

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: aligned_size,
            compression_ratio: original_size as f32 / aligned_size as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: 0.0,
            access_overhead: -0.1, // Negative overhead indicates performance benefit
        }
    }

    fn is_suitable(&self, _weights: &[i8], config: &TernaryPackingConfig) -> bool {
        config.simd_optimized
    }
}

/// Run-length encoded packer for sparse weights
#[derive(Debug)]
pub struct RunLengthEncodedPacker;

impl TernaryPacker for RunLengthEncodedPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let mut rle_data = Vec::new();
        let mut data = Vec::new();

        if weights.is_empty() {
            return Ok(PackedTernaryWeights {
                data,
                shape: Shape::from_dims(&[0]),
                strategy: TernaryPackingStrategy::RunLengthEncoded,
                config: config.clone(),
                metadata: PackingMetadata::default(),
                memory_footprint: 0,
                compression_ratio: 1.0,
            });
        }

        let mut current_value = weights[0];
        let mut count = 1usize;

        for &weight in &weights[1..] {
            if weight == current_value && count < 255 {
                // Limit count to fit in u8
                count += 1;
            } else {
                rle_data.push((current_value, count));
                // Encode as: value (1 byte) + count (1 byte)
                data.push((current_value + 1) as u8); // Convert {-1,0,1} to {0,1,2}
                data.push(count as u8);

                current_value = weight;
                count = 1;
            }
        }

        // Add the last run
        rle_data.push((current_value, count));
        data.push((current_value + 1) as u8);
        data.push(count as u8);

        let metadata = PackingMetadata {
            element_count: weights.len(),
            rle_data: Some(rle_data),
            ..Default::default()
        };

        let memory_footprint = data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::RunLengthEncoded,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut weights = Vec::with_capacity(packed.metadata.element_count);

        // Decode from data stream
        let mut i = 0;
        while i < packed.data.len() {
            if i + 1 >= packed.data.len() {
                break;
            }

            let value = (packed.data[i] as i8) - 1; // Convert {0,1,2} back to {-1,0,1}
            let count = packed.data[i + 1] as usize;

            for _ in 0..count {
                if weights.len() >= packed.metadata.element_count {
                    break;
                }
                weights.push(value);
            }

            i += 2;
        }

        weights.truncate(packed.metadata.element_count);
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        _config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        // Estimate RLE compression based on run analysis
        let mut runs = 0;
        let mut current_value = if weights.is_empty() { 0 } else { weights[0] };

        for &weight in weights {
            if weight != current_value {
                runs += 1;
                current_value = weight;
            }
        }
        runs += 1; // Add the last run

        let original_size = weights.len();
        let estimated_packed_size = runs * 2; // 2 bytes per run (value + count)
        let memory_saved = original_size.saturating_sub(estimated_packed_size);

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: estimated_packed_size,
            compression_ratio: original_size as f32 / estimated_packed_size as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: (memory_saved as f32 / original_size as f32) * 100.0,
            access_overhead: 0.2, // Higher overhead for sequential decoding
        }
    }

    fn is_suitable(&self, weights: &[i8], config: &TernaryPackingConfig) -> bool {
        // Calculate sparsity (percentage of zeros)
        let zeros = weights.iter().filter(|&&w| w == 0).count();
        let sparsity = zeros as f32 / weights.len() as f32;
        sparsity >= config.sparsity_threshold
    }
}

/// Compressed sparse packer for very sparse weights
#[derive(Debug)]
pub struct CompressedSparsePacker;

impl TernaryPacker for CompressedSparsePacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Store only non-zero values and their indices
        for (i, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                indices.push(i);
                values.push(weight);
            }
        }

        // Encode indices and values
        let mut data = Vec::new();

        // Store number of non-zero elements (4 bytes)
        let nnz = indices.len() as u32;
        data.extend_from_slice(&nnz.to_le_bytes());

        // Store indices (4 bytes each for simplicity, could be optimized)
        for &idx in &indices {
            data.extend_from_slice(&(idx as u32).to_le_bytes());
        }

        // Store values (1 byte each, converted to unsigned)
        for &val in &values {
            data.push((val + 1) as u8); // Convert {-1,0,1} to {0,1,2}
        }

        let metadata = PackingMetadata {
            element_count: weights.len(),
            sparse_indices: Some(indices),
            ..Default::default()
        };

        let memory_footprint = data.len();
        let compression_ratio = weights.len() as f32 / memory_footprint as f32;

        Ok(PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::CompressedSparse,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut weights = vec![0i8; packed.metadata.element_count];

        if packed.data.len() < 4 {
            return Ok(weights); // All zeros
        }

        // Read number of non-zero elements
        let nnz = u32::from_le_bytes([
            packed.data[0],
            packed.data[1],
            packed.data[2],
            packed.data[3],
        ]) as usize;

        let mut offset = 4;

        // Read indices
        let mut indices = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            if offset + 4 > packed.data.len() {
                return Err(QuantizationError::InvalidInput(
                    "Corrupted sparse data".to_string(),
                ));
            }
            let idx = u32::from_le_bytes([
                packed.data[offset],
                packed.data[offset + 1],
                packed.data[offset + 2],
                packed.data[offset + 3],
            ]) as usize;
            indices.push(idx);
            offset += 4;
        }

        // Read values
        for (i, &idx) in indices.iter().enumerate() {
            if offset + i >= packed.data.len() || idx >= weights.len() {
                return Err(QuantizationError::InvalidInput(
                    "Corrupted sparse data".to_string(),
                ));
            }
            let val = (packed.data[offset + i] as i8) - 1; // Convert {0,1,2} back to {-1,0,1}
            weights[idx] = val;
        }

        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        _config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let nnz = weights.iter().filter(|&&w| w != 0).count();
        let original_size = weights.len();
        let estimated_packed_size = 4 + (nnz * 4) + nnz; // header + indices + values
        let memory_saved = original_size.saturating_sub(estimated_packed_size);

        PackingSavingsEstimate {
            original_size_bytes: original_size,
            packed_size_bytes: estimated_packed_size,
            compression_ratio: original_size as f32 / estimated_packed_size as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: (memory_saved as f32 / original_size as f32) * 100.0,
            access_overhead: 0.3, // Higher overhead for sparse access
        }
    }

    fn is_suitable(&self, weights: &[i8], config: &TernaryPackingConfig) -> bool {
        let zeros = weights.iter().filter(|&&w| w == 0).count();
        let sparsity = zeros as f32 / weights.len() as f32;
        sparsity >= config.sparsity_threshold && sparsity > 0.8 // Very sparse
    }
}

/// Hybrid packer that chooses the best strategy per block
#[derive(Debug)]
pub struct HybridPacker;

impl TernaryPacker for HybridPacker {
    fn pack(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let block_size = config.block_size.unwrap_or(64);
        let mut data = Vec::new();
        let mut block_sizes = Vec::new();
        let mut total_compression_ratio = 0.0;
        let mut block_count = 0;

        // Pack each block with the best strategy
        for chunk in weights.chunks(block_size) {
            let (best_strategy, packed_block) = self.find_best_strategy_for_block(chunk, config)?;

            // Store strategy type (1 byte) + block size (2 bytes) + data
            data.push(best_strategy as u8);
            data.extend_from_slice(&(packed_block.data.len() as u16).to_le_bytes());
            data.extend_from_slice(&packed_block.data);

            block_sizes.push(packed_block.data.len());
            total_compression_ratio += packed_block.compression_ratio;
            block_count += 1;
        }

        let metadata = PackingMetadata {
            element_count: weights.len(),
            block_sizes: Some(block_sizes),
            ..Default::default()
        };

        let memory_footprint = data.len();
        let avg_compression_ratio = total_compression_ratio / block_count as f32;

        Ok(PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[weights.len()]),
            strategy: TernaryPackingStrategy::Hybrid,
            config: config.clone(),
            metadata,
            memory_footprint,
            compression_ratio: avg_compression_ratio,
        })
    }

    fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut weights = Vec::with_capacity(packed.metadata.element_count);
        let mut offset = 0;

        while offset < packed.data.len() && weights.len() < packed.metadata.element_count {
            if offset + 3 > packed.data.len() {
                break;
            }

            // Read strategy type and block size
            let strategy_byte = packed.data[offset];
            let block_data_size =
                u16::from_le_bytes([packed.data[offset + 1], packed.data[offset + 2]]) as usize;
            offset += 3;

            if offset + block_data_size > packed.data.len() {
                return Err(QuantizationError::InvalidInput(
                    "Corrupted hybrid data".to_string(),
                ));
            }

            // Extract block data
            let block_data = &packed.data[offset..offset + block_data_size];
            offset += block_data_size;

            // Unpack using the appropriate strategy
            let block_weights = self.unpack_block_with_strategy(strategy_byte, block_data)?;
            weights.extend(block_weights);
        }

        weights.truncate(packed.metadata.element_count);
        Ok(weights)
    }

    fn estimate_savings(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> PackingSavingsEstimate {
        let block_size = config.block_size.unwrap_or(64);
        let mut total_original = 0;
        let mut total_packed = 0;

        for chunk in weights.chunks(block_size) {
            let (_, estimate) = self.estimate_best_strategy_for_block(chunk, config);
            total_original += estimate.original_size_bytes;
            total_packed += estimate.packed_size_bytes + 3; // Add overhead for strategy + size
        }

        let memory_saved = total_original.saturating_sub(total_packed);

        PackingSavingsEstimate {
            original_size_bytes: total_original,
            packed_size_bytes: total_packed,
            compression_ratio: total_original as f32 / total_packed as f32,
            memory_saved_bytes: memory_saved,
            savings_percentage: (memory_saved as f32 / total_original as f32) * 100.0,
            access_overhead: 0.25, // Moderate overhead for hybrid approach
        }
    }

    fn is_suitable(&self, _weights: &[i8], _config: &TernaryPackingConfig) -> bool {
        true // Hybrid is always suitable as it adapts
    }
}

impl HybridPacker {
    /// Find the best packing strategy for a block of weights
    fn find_best_strategy_for_block(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<(TernaryPackingStrategy, PackedTernaryWeights), QuantizationError> {
        let strategies = [
            (
                TernaryPackingStrategy::BitPacked2Bit,
                Box::new(BitPacked2BitPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::Base3Packed,
                Box::new(Base3PackedPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::RunLengthEncoded,
                Box::new(RunLengthEncodedPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::CompressedSparse,
                Box::new(CompressedSparsePacker) as Box<dyn TernaryPacker>,
            ),
        ];

        let mut best_strategy = TernaryPackingStrategy::BitPacked2Bit;
        let mut best_packed = None;
        let mut best_ratio = 0.0f32;

        for (strategy, packer) in strategies {
            if packer.is_suitable(weights, config) {
                if let Ok(packed) = packer.pack(weights, config) {
                    if packed.compression_ratio > best_ratio {
                        best_ratio = packed.compression_ratio;
                        best_strategy = strategy;
                        best_packed = Some(packed);
                    }
                }
            }
        }

        match best_packed {
            Some(packed) => Ok((best_strategy, packed)),
            None => {
                // Fallback to bit-packed
                let packer = BitPacked2BitPacker;
                let packed = packer.pack(weights, config)?;
                Ok((TernaryPackingStrategy::BitPacked2Bit, packed))
            }
        }
    }

    /// Estimate the best strategy for a block without actually packing
    fn estimate_best_strategy_for_block(
        &self,
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> (TernaryPackingStrategy, PackingSavingsEstimate) {
        let strategies = [
            (
                TernaryPackingStrategy::BitPacked2Bit,
                Box::new(BitPacked2BitPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::Base3Packed,
                Box::new(Base3PackedPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::RunLengthEncoded,
                Box::new(RunLengthEncodedPacker) as Box<dyn TernaryPacker>,
            ),
            (
                TernaryPackingStrategy::CompressedSparse,
                Box::new(CompressedSparsePacker) as Box<dyn TernaryPacker>,
            ),
        ];

        let mut best_strategy = TernaryPackingStrategy::BitPacked2Bit;
        let mut best_estimate = BitPacked2BitPacker.estimate_savings(weights, config);

        for (strategy, packer) in strategies {
            if packer.is_suitable(weights, config) {
                let estimate = packer.estimate_savings(weights, config);
                if estimate.compression_ratio > best_estimate.compression_ratio {
                    best_strategy = strategy;
                    best_estimate = estimate;
                }
            }
        }

        (best_strategy, best_estimate)
    }

    /// Unpack a block using the specified strategy
    fn unpack_block_with_strategy(
        &self,
        strategy_byte: u8,
        data: &[u8],
    ) -> Result<Vec<i8>, QuantizationError> {
        let strategy = match strategy_byte {
            0 => TernaryPackingStrategy::Uncompressed,
            1 => TernaryPackingStrategy::BitPacked2Bit,
            2 => TernaryPackingStrategy::Base3Packed,
            3 => TernaryPackingStrategy::ByteAligned,
            4 => TernaryPackingStrategy::RunLengthEncoded,
            5 => TernaryPackingStrategy::CompressedSparse,
            _ => {
                return Err(QuantizationError::InvalidInput(
                    "Unknown packing strategy".to_string(),
                ))
            }
        };

        // We need to determine the original element count from the packed data
        // For now, we'll estimate based on the strategy and data size
        let estimated_element_count = match strategy {
            TernaryPackingStrategy::Uncompressed => data.len(),
            TernaryPackingStrategy::BitPacked2Bit => data.len() * 4, // 4 values per byte
            TernaryPackingStrategy::Base3Packed => data.len() * 5,   // 5 values per byte
            TernaryPackingStrategy::ByteAligned => data.len(),
            TernaryPackingStrategy::RunLengthEncoded => {
                // For RLE, we need to decode to find the actual count
                let mut count = 0;
                let mut i = 0;
                while i + 1 < data.len() {
                    count += data[i + 1] as usize; // count is in second byte
                    i += 2;
                }
                count
            }
            TernaryPackingStrategy::CompressedSparse => {
                if data.len() >= 4 {
                    let total_elements =
                        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    // This is actually the sparse element count, we need the original size
                    // For now, assume a reasonable default
                    total_elements.max(data.len())
                } else {
                    data.len()
                }
            }
            _ => data.len(),
        };

        // Create a temporary packed structure for unpacking
        let mut metadata = PackingMetadata::default();
        metadata.element_count = estimated_element_count;

        let packed = PackedTernaryWeights {
            data: data.to_vec(),
            shape: Shape::from_dims(&[estimated_element_count]),
            strategy,
            config: TernaryPackingConfig::default(),
            metadata,
            memory_footprint: data.len(),
            compression_ratio: 1.0,
        };

        match strategy {
            TernaryPackingStrategy::Uncompressed => UncompressedPacker.unpack(&packed),
            TernaryPackingStrategy::BitPacked2Bit => BitPacked2BitPacker.unpack(&packed),
            TernaryPackingStrategy::Base3Packed => Base3PackedPacker.unpack(&packed),
            TernaryPackingStrategy::ByteAligned => ByteAlignedPacker.unpack(&packed),
            TernaryPackingStrategy::RunLengthEncoded => RunLengthEncodedPacker.unpack(&packed),
            TernaryPackingStrategy::CompressedSparse => CompressedSparsePacker.unpack(&packed),
            TernaryPackingStrategy::Hybrid => Err(QuantizationError::InvalidInput(
                "Nested hybrid packing not supported".to_string(),
            )),
        }
    }
}

/// Factory for creating ternary packers
pub struct TernaryPackerFactory;

impl TernaryPackerFactory {
    /// Create a packer for the specified strategy
    pub fn create_packer(strategy: TernaryPackingStrategy) -> Box<dyn TernaryPacker> {
        match strategy {
            TernaryPackingStrategy::Uncompressed => Box::new(UncompressedPacker),
            TernaryPackingStrategy::BitPacked2Bit => Box::new(BitPacked2BitPacker),
            TernaryPackingStrategy::Base3Packed => Box::new(Base3PackedPacker),
            TernaryPackingStrategy::ByteAligned => Box::new(ByteAlignedPacker),
            TernaryPackingStrategy::RunLengthEncoded => Box::new(RunLengthEncodedPacker),
            TernaryPackingStrategy::CompressedSparse => Box::new(CompressedSparsePacker),
            TernaryPackingStrategy::Hybrid => Box::new(HybridPacker),
        }
    }

    /// Automatically select the best packing strategy for given weights
    pub fn auto_select_strategy(
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> TernaryPackingStrategy {
        let hybrid_packer = HybridPacker;
        let (best_strategy, _) = hybrid_packer.estimate_best_strategy_for_block(weights, config);
        best_strategy
    }

    /// Pack weights using the optimal strategy
    pub fn pack_optimal(
        weights: &[i8],
        config: &TernaryPackingConfig,
    ) -> Result<PackedTernaryWeights, QuantizationError> {
        let strategy = Self::auto_select_strategy(weights, config);
        let packer = Self::create_packer(strategy);
        packer.pack(weights, config)
    }
}

/// Utility functions for ternary weight packing
pub mod packing_utils {
    use super::*;

    /// Convert tensor to ternary i8 values
    pub fn tensor_to_ternary(tensor: &Tensor) -> Result<Vec<i8>, QuantizationError> {
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let ternary: Vec<i8> = data
            .iter()
            .map(|&f| {
                if f > 0.5 {
                    1
                } else if f < -0.5 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        Ok(ternary)
    }

    /// Convert ternary i8 values back to tensor
    pub fn ternary_to_tensor(
        ternary: &[i8],
        shape: &Shape,
        device: &Device,
    ) -> Result<Tensor, QuantizationError> {
        let data: Vec<f32> = ternary.iter().map(|&t| t as f32).collect();
        let tensor = Tensor::from_slice(&data, shape, device)?;
        Ok(tensor)
    }

    /// Analyze weight sparsity
    pub fn analyze_sparsity(weights: &[i8]) -> SparsityAnalysis {
        let total = weights.len();
        let zeros = weights.iter().filter(|&&w| w == 0).count();
        let positives = weights.iter().filter(|&&w| w > 0).count();
        let negatives = weights.iter().filter(|&&w| w < 0).count();

        SparsityAnalysis {
            total_elements: total,
            zero_count: zeros,
            positive_count: positives,
            negative_count: negatives,
            sparsity_ratio: zeros as f32 / total as f32,
            balance_ratio: (positives as f32 - negatives as f32).abs() / total as f32,
        }
    }

    /// Recommend optimal packing strategy based on weight characteristics
    pub fn recommend_strategy(weights: &[i8]) -> TernaryPackingStrategy {
        let analysis = analyze_sparsity(weights);

        if analysis.sparsity_ratio > 0.8 {
            TernaryPackingStrategy::CompressedSparse
        } else if analysis.sparsity_ratio > 0.6 {
            TernaryPackingStrategy::RunLengthEncoded
        } else if weights.len() % 5 == 0 {
            TernaryPackingStrategy::Base3Packed
        } else {
            TernaryPackingStrategy::BitPacked2Bit
        }
    }
}

/// Sparsity analysis results
#[derive(Debug, Clone)]
pub struct SparsityAnalysis {
    pub total_elements: usize,
    pub zero_count: usize,
    pub positive_count: usize,
    pub negative_count: usize,
    pub sparsity_ratio: f32,
    pub balance_ratio: f32, // How balanced +1 and -1 values are
}

/// Calculate CRC32 checksum for data integrity verification
fn calculate_crc32(data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = generate_crc32_table();

    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        let table_index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[table_index];
    }
    !crc
}

/// Generate CRC32 lookup table at compile time
const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;

    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;

        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }

        table[i] = crc;
        i += 1;
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncompressed_packer() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = UncompressedPacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::Uncompressed);
    }

    #[test]
    fn test_bit_packed_2bit_packer() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::BitPacked2Bit);
        assert!(packed.compression_ratio > 1.0);
    }

    #[test]
    fn test_base3_packed_packer() {
        let weights = vec![-1i8, 0, 1, -1, 0]; // Exactly 5 elements
        let config = TernaryPackingConfig::default();
        let packer = Base3PackedPacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::Base3Packed);
        assert!(packed.compression_ratio > 1.0);
    }

    #[test]
    fn test_run_length_encoded_packer() {
        let weights = vec![0i8, 0, 0, 1, 1, -1, -1, -1, 0, 0];
        let config = TernaryPackingConfig::default();
        let packer = RunLengthEncodedPacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::RunLengthEncoded);
    }

    #[test]
    fn test_compressed_sparse_packer() {
        let weights = vec![0i8, 0, 1, 0, 0, 0, -1, 0, 0, 0];
        let config = TernaryPackingConfig::default();
        let packer = CompressedSparsePacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::CompressedSparse);
    }

    #[test]
    fn test_hybrid_packer() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0];
        let config = TernaryPackingConfig {
            block_size: Some(4),
            ..Default::default()
        };
        let packer = HybridPacker;

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
        assert_eq!(packed.strategy, TernaryPackingStrategy::Hybrid);
    }

    #[test]
    fn test_packer_factory() {
        let weights = vec![-1i8, 0, 1, -1];
        let config = TernaryPackingConfig::default();

        let strategy = TernaryPackerFactory::auto_select_strategy(&weights, &config);
        let packer = TernaryPackerFactory::create_packer(strategy);

        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = packer.unpack(&packed).unwrap();

        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_sparsity_analysis() {
        let weights = vec![0i8, 0, 1, 0, -1, 0, 0, 1];
        let analysis = packing_utils::analyze_sparsity(&weights);

        assert_eq!(analysis.total_elements, 8);
        assert_eq!(analysis.zero_count, 5);
        assert_eq!(analysis.positive_count, 2);
        assert_eq!(analysis.negative_count, 1);
        assert_eq!(analysis.sparsity_ratio, 5.0 / 8.0);
    }

    #[test]
    fn test_strategy_recommendation() {
        // Dense weights
        let dense_weights = vec![-1i8, 1, -1, 1, -1, 1];
        assert_eq!(
            packing_utils::recommend_strategy(&dense_weights),
            TernaryPackingStrategy::BitPacked2Bit
        );

        // Sparse weights
        let sparse_weights = vec![0i8, 0, 0, 0, 0, 0, 0, 1, 0, 0];
        assert_eq!(
            packing_utils::recommend_strategy(&sparse_weights),
            TernaryPackingStrategy::CompressedSparse
        );

        // Base-3 optimal
        let base3_weights = vec![-1i8, 0, 1, -1, 0]; // Length 5
        assert_eq!(
            packing_utils::recommend_strategy(&base3_weights),
            TernaryPackingStrategy::Base3Packed
        );
    }

    #[test]
    fn test_compression_ratios() {
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1];
        let config = TernaryPackingConfig::default();

        let strategies = [
            TernaryPackingStrategy::Uncompressed,
            TernaryPackingStrategy::BitPacked2Bit,
            TernaryPackingStrategy::Base3Packed,
        ];

        for strategy in strategies {
            let packer = TernaryPackerFactory::create_packer(strategy);
            let estimate = packer.estimate_savings(&weights, &config);

            println!(
                "Strategy {:?}: compression ratio = {:.2}",
                strategy, estimate.compression_ratio
            );
            assert!(estimate.original_size_bytes > 0);
            assert!(estimate.packed_size_bytes > 0);
        }
    }
}
