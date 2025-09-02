//! Advanced corruption detection for packed ternary data
//!
//! This module provides comprehensive error checking and corruption detection
//! for packed ternary weight data, including integrity validation, checksum
//! verification, and recovery mechanisms.

use super::packing::{PackedTernaryWeights, TernaryPackingStrategy};
use super::utils::QuantizationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Types of corruption that can occur in packed data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorruptionType {
    /// Data size mismatch between expected and actual
    SizeMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// Invalid values found in packed data
    InvalidValues {
        invalid_bytes: Vec<u8>,
        positions: Vec<usize>,
        expected_range: String,
    },
    /// Checksum verification failure
    ChecksumMismatch {
        expected: u32,
        actual: u32,
        algorithm: String,
    },
    /// Metadata inconsistency
    MetadataInconsistency {
        field: String,
        expected: String,
        actual: String,
    },
    /// Strategy-specific corruption
    StrategySpecific {
        strategy: TernaryPackingStrategy,
        details: String,
    },
    /// Structural corruption (e.g., truncated data)
    StructuralCorruption {
        description: String,
        recovery_possible: bool,
    },
    /// Padding corruption
    PaddingCorruption {
        expected_padding: usize,
        actual_padding: usize,
    },
    /// Index out of bounds in sparse formats
    IndexOutOfBounds {
        index: usize,
        max_valid: usize,
        format: String,
    },
}

impl fmt::Display for CorruptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorruptionType::SizeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Size mismatch in {context}: expected {expected} bytes, got {actual} bytes"
                )
            }
            CorruptionType::InvalidValues {
                invalid_bytes,
                positions,
                expected_range,
            } => {
                write!(f, "Invalid values {invalid_bytes:?} at positions {positions:?}, expected range: {expected_range}")
            }
            CorruptionType::ChecksumMismatch {
                expected,
                actual,
                algorithm,
            } => {
                write!(
                    f,
                    "{algorithm} checksum mismatch: expected 0x{expected:08X}, got 0x{actual:08X}"
                )
            }
            CorruptionType::MetadataInconsistency {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Metadata inconsistency in {field}: expected '{expected}', got '{actual}'"
                )
            }
            CorruptionType::StrategySpecific { strategy, details } => {
                write!(f, "Strategy-specific corruption in {strategy:?}: {details}")
            }
            CorruptionType::StructuralCorruption {
                description,
                recovery_possible,
            } => {
                write!(
                    f,
                    "Structural corruption: {description} (recovery possible: {recovery_possible})"
                )
            }
            CorruptionType::PaddingCorruption {
                expected_padding,
                actual_padding,
            } => {
                write!(f, "Padding corruption: expected {expected_padding} padding bytes, got {actual_padding}")
            }
            CorruptionType::IndexOutOfBounds {
                index,
                max_valid,
                format,
            } => {
                write!(
                    f,
                    "Index {index} out of bounds in {format} format (max valid: {max_valid})"
                )
            }
        }
    }
}

/// Detailed corruption report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptionReport {
    /// Type of corruption detected
    pub corruption_type: CorruptionType,
    /// Severity level of the corruption
    pub severity: CorruptionSeverity,
    /// Confidence level of the detection (0.0 to 1.0)
    pub confidence: f32,
    /// Byte offset where corruption was detected
    pub byte_offset: Option<usize>,
    /// Length of corrupted region
    pub corrupted_length: Option<usize>,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<RecoveryAction>,
    /// Additional context information
    pub context: HashMap<String, String>,
}

/// Severity levels for corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CorruptionSeverity {
    /// Minor corruption that doesn't affect functionality
    Minor,
    /// Moderate corruption that may cause degraded performance
    Moderate,
    /// Severe corruption that prevents normal operation
    Severe,
    /// Critical corruption that makes data unusable
    Critical,
}

/// Suggested recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Attempt to repair the corruption automatically
    AutoRepair { description: String },
    /// Use fallback data or strategy
    UseFallback {
        fallback_strategy: TernaryPackingStrategy,
    },
    /// Re-pack the data with a different strategy
    RepackData {
        suggested_strategy: TernaryPackingStrategy,
    },
    /// Discard corrupted portion and continue
    DiscardCorrupted { safe_length: usize },
    /// Manual intervention required
    ManualIntervention { instructions: String },
}

/// Comprehensive corruption detector for packed ternary data
pub struct CorruptionDetector {
    /// Enable checksum validation
    pub enable_checksums: bool,
    /// Enable deep validation (slower but more thorough)
    pub enable_deep_validation: bool,
    /// Maximum allowed corruption ratio before failing
    pub max_corruption_ratio: f32,
    /// Strategy-specific validators
    strategy_validators: HashMap<TernaryPackingStrategy, Box<dyn StrategyValidator>>,
}

impl Default for CorruptionDetector {
    fn default() -> Self {
        let mut detector = Self {
            enable_checksums: true,
            enable_deep_validation: true,
            max_corruption_ratio: 0.1, // 10% max corruption
            strategy_validators: HashMap::new(),
        };

        // Register default validators
        detector.register_validator(
            TernaryPackingStrategy::BitPacked2Bit,
            Box::new(BitPacked2BitValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::Base3Packed,
            Box::new(Base3PackedValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::RunLengthEncoded,
            Box::new(RunLengthEncodedValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::CompressedSparse,
            Box::new(CompressedSparseValidator),
        );
        detector.register_validator(TernaryPackingStrategy::Hybrid, Box::new(HybridValidator));

        detector
    }
}

impl CorruptionDetector {
    /// Create a new corruption detector with custom settings
    pub fn new(
        enable_checksums: bool,
        enable_deep_validation: bool,
        max_corruption_ratio: f32,
    ) -> Self {
        let mut detector = Self {
            enable_checksums,
            enable_deep_validation,
            max_corruption_ratio,
            strategy_validators: HashMap::new(),
        };

        // Register default validators
        detector.register_validator(
            TernaryPackingStrategy::BitPacked2Bit,
            Box::new(BitPacked2BitValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::Base3Packed,
            Box::new(Base3PackedValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::RunLengthEncoded,
            Box::new(RunLengthEncodedValidator),
        );
        detector.register_validator(
            TernaryPackingStrategy::CompressedSparse,
            Box::new(CompressedSparseValidator),
        );
        detector.register_validator(TernaryPackingStrategy::Hybrid, Box::new(HybridValidator));

        detector
    }

    /// Register a custom validator for a specific strategy
    pub fn register_validator(
        &mut self,
        strategy: TernaryPackingStrategy,
        validator: Box<dyn StrategyValidator>,
    ) {
        self.strategy_validators.insert(strategy, validator);
    }

    /// Perform comprehensive corruption detection on packed data
    pub fn detect_corruption(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Basic validation
        reports.extend(self.validate_basic_structure(packed)?);

        // Metadata validation
        reports.extend(self.validate_metadata(packed)?);

        // Checksum validation
        if self.enable_checksums {
            if let Some(report) = self.validate_checksum(packed)? {
                reports.push(report);
            }
        }

        // Strategy-specific validation
        reports.extend(self.validate_strategy_specific(packed)?);

        // Deep validation
        if self.enable_deep_validation {
            reports.extend(self.validate_deep(packed)?);
        }

        // Check overall corruption ratio
        let corruption_ratio = self.calculate_corruption_ratio(&reports, packed);
        if corruption_ratio > self.max_corruption_ratio {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::StructuralCorruption {
                    description: format!("Overall corruption ratio {:.2}% exceeds threshold {:.2}%",
                                       corruption_ratio * 100.0, self.max_corruption_ratio * 100.0),
                    recovery_possible: false,
                },
                severity: CorruptionSeverity::Critical,
                confidence: 1.0,
                byte_offset: None,
                corrupted_length: Some(packed.data.len()),
                recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                    instructions: "Data is too corrupted for automatic recovery. Consider re-packing from source.".to_string(),
                }],
                context: HashMap::new(),
            });
        }

        Ok(reports)
    }

    /// Validate basic structure of packed data
    fn validate_basic_structure(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Check if data is empty when it shouldn't be
        if packed.metadata.element_count > 0 && packed.data.is_empty() {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::SizeMismatch {
                    expected: 1, // At least 1 byte expected
                    actual: 0,
                    context: "packed data".to_string(),
                },
                severity: CorruptionSeverity::Critical,
                confidence: 1.0,
                byte_offset: Some(0),
                corrupted_length: Some(0),
                recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                    instructions: "Data is completely empty. Re-pack from source data.".to_string(),
                }],
                context: HashMap::new(),
            });
        }

        // Check for reasonable data size bounds
        let expected_min_size = self.calculate_minimum_expected_size(packed);
        let expected_max_size = self.calculate_maximum_expected_size(packed);

        if packed.data.len() < expected_min_size {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::SizeMismatch {
                    expected: expected_min_size,
                    actual: packed.data.len(),
                    context: "minimum size check".to_string(),
                },
                severity: CorruptionSeverity::Severe,
                confidence: 0.9,
                byte_offset: Some(packed.data.len()),
                corrupted_length: Some(expected_min_size - packed.data.len()),
                recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                    instructions: "Data appears truncated. Check source data integrity."
                        .to_string(),
                }],
                context: HashMap::new(),
            });
        }

        if packed.data.len() > expected_max_size {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::SizeMismatch {
                    expected: expected_max_size,
                    actual: packed.data.len(),
                    context: "maximum size check".to_string(),
                },
                severity: CorruptionSeverity::Moderate,
                confidence: 0.7,
                byte_offset: Some(expected_max_size),
                corrupted_length: Some(packed.data.len() - expected_max_size),
                recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                    safe_length: expected_max_size,
                }],
                context: HashMap::new(),
            });
        }

        Ok(reports)
    }

    /// Validate metadata consistency
    fn validate_metadata(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Check element count consistency
        if packed.metadata.element_count == 0 && !packed.data.is_empty() {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::MetadataInconsistency {
                    field: "element_count".to_string(),
                    expected: "non-zero".to_string(),
                    actual: "0".to_string(),
                },
                severity: CorruptionSeverity::Severe,
                confidence: 1.0,
                byte_offset: None,
                corrupted_length: None,
                recovery_suggestions: vec![RecoveryAction::AutoRepair {
                    description: "Attempt to infer element count from data size and strategy"
                        .to_string(),
                }],
                context: HashMap::new(),
            });
        }

        // Check padding consistency
        if let Some(expected_padding) = self.calculate_expected_padding(packed) {
            if packed.metadata.padding != expected_padding {
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::PaddingCorruption {
                        expected_padding,
                        actual_padding: packed.metadata.padding,
                    },
                    severity: CorruptionSeverity::Moderate,
                    confidence: 0.8,
                    byte_offset: None,
                    corrupted_length: None,
                    recovery_suggestions: vec![RecoveryAction::AutoRepair {
                        description: "Recalculate padding based on element count and strategy"
                            .to_string(),
                    }],
                    context: HashMap::new(),
                });
            }
        }

        // Validate sparse indices if present
        if let Some(ref indices) = packed.metadata.sparse_indices {
            for (i, &index) in indices.iter().enumerate() {
                if index >= packed.metadata.element_count {
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::IndexOutOfBounds {
                            index,
                            max_valid: packed.metadata.element_count.saturating_sub(1),
                            format: "sparse indices".to_string(),
                        },
                        severity: CorruptionSeverity::Severe,
                        confidence: 1.0,
                        byte_offset: Some(i * 4), // Assuming 4-byte indices
                        corrupted_length: Some(4),
                        recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                            safe_length: i * 4,
                        }],
                        context: HashMap::new(),
                    });
                }
            }
        }

        Ok(reports)
    }

    /// Validate checksum if available
    fn validate_checksum(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Option<CorruptionReport>, QuantizationError> {
        // Check if checksum is stored in extra_data
        if let Some(stored_checksum_bytes) = packed.metadata.extra_data.get("crc32") {
            if stored_checksum_bytes.len() == 4 {
                let stored_checksum = u32::from_le_bytes([
                    stored_checksum_bytes[0],
                    stored_checksum_bytes[1],
                    stored_checksum_bytes[2],
                    stored_checksum_bytes[3],
                ]);

                let calculated_checksum = self.calculate_crc32(&packed.data);

                if stored_checksum != calculated_checksum {
                    return Ok(Some(CorruptionReport {
                        corruption_type: CorruptionType::ChecksumMismatch {
                            expected: stored_checksum,
                            actual: calculated_checksum,
                            algorithm: "CRC32".to_string(),
                        },
                        severity: CorruptionSeverity::Severe,
                        confidence: 0.95,
                        byte_offset: None,
                        corrupted_length: None,
                        recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                            instructions:
                                "Checksum mismatch indicates data corruption. Verify source data."
                                    .to_string(),
                        }],
                        context: HashMap::new(),
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Perform strategy-specific validation
    fn validate_strategy_specific(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        if let Some(validator) = self.strategy_validators.get(&packed.strategy) {
            validator.validate(packed)
        } else {
            Ok(Vec::new())
        }
    }

    /// Perform deep validation by attempting partial unpacking
    fn validate_deep(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Attempt to unpack a small portion to check for structural issues
        let test_size = (packed.metadata.element_count.min(100)).max(1);

        match self.attempt_partial_unpack(packed, test_size) {
            Ok(_) => {
                // Successful partial unpack, no structural issues detected
            }
            Err(e) => {
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::StructuralCorruption {
                        description: format!("Partial unpack failed: {e}"),
                        recovery_possible: true,
                    },
                    severity: CorruptionSeverity::Severe,
                    confidence: 0.8,
                    byte_offset: None,
                    corrupted_length: None,
                    recovery_suggestions: vec![
                        RecoveryAction::UseFallback {
                            fallback_strategy: TernaryPackingStrategy::Uncompressed,
                        },
                        RecoveryAction::RepackData {
                            suggested_strategy: TernaryPackingStrategy::BitPacked2Bit,
                        },
                    ],
                    context: HashMap::new(),
                });
            }
        }

        Ok(reports)
    }

    /// Calculate minimum expected size for the packed data
    fn calculate_minimum_expected_size(&self, packed: &PackedTernaryWeights) -> usize {
        match packed.strategy {
            TernaryPackingStrategy::Uncompressed => packed.metadata.element_count,
            TernaryPackingStrategy::BitPacked2Bit => packed.metadata.element_count.div_ceil(4),
            TernaryPackingStrategy::Base3Packed => packed.metadata.element_count.div_ceil(5),
            TernaryPackingStrategy::ByteAligned => packed.metadata.element_count,
            TernaryPackingStrategy::RunLengthEncoded => 2, // At least one run
            TernaryPackingStrategy::CompressedSparse => 4, // At least header
            TernaryPackingStrategy::Hybrid => 3,           // At least one block header
        }
    }

    /// Calculate maximum expected size for the packed data
    fn calculate_maximum_expected_size(&self, packed: &PackedTernaryWeights) -> usize {
        match packed.strategy {
            TernaryPackingStrategy::Uncompressed => packed.metadata.element_count,
            TernaryPackingStrategy::BitPacked2Bit => packed.metadata.element_count.div_ceil(4),
            TernaryPackingStrategy::Base3Packed => packed.metadata.element_count.div_ceil(5),
            TernaryPackingStrategy::ByteAligned => {
                let alignment = packed.config.alignment;
                packed.metadata.element_count.div_ceil(alignment) * alignment
            }
            TernaryPackingStrategy::RunLengthEncoded => packed.metadata.element_count * 2, // Worst case: no compression
            TernaryPackingStrategy::CompressedSparse => {
                4 + (packed.metadata.element_count * 5) // Header + worst case all non-zero
            }
            TernaryPackingStrategy::Hybrid => packed.metadata.element_count * 2, // Conservative estimate
        }
    }

    /// Calculate expected padding for the strategy
    fn calculate_expected_padding(&self, packed: &PackedTernaryWeights) -> Option<usize> {
        match packed.strategy {
            TernaryPackingStrategy::BitPacked2Bit => {
                let remainder = packed.metadata.element_count % 4;
                if remainder != 0 {
                    Some(4 - remainder)
                } else {
                    Some(0)
                }
            }
            TernaryPackingStrategy::Base3Packed => {
                let remainder = packed.metadata.element_count % 5;
                if remainder != 0 {
                    Some(5 - remainder)
                } else {
                    Some(0)
                }
            }
            TernaryPackingStrategy::ByteAligned => {
                let alignment = packed.config.alignment;
                let remainder = packed.metadata.element_count % alignment;
                if remainder != 0 {
                    Some(alignment - remainder)
                } else {
                    Some(0)
                }
            }
            _ => None,
        }
    }

    /// Calculate overall corruption ratio
    fn calculate_corruption_ratio(
        &self,
        reports: &[CorruptionReport],
        packed: &PackedTernaryWeights,
    ) -> f32 {
        if reports.is_empty() || packed.data.is_empty() {
            return 0.0;
        }

        let total_corrupted_bytes: usize = reports.iter().filter_map(|r| r.corrupted_length).sum();

        total_corrupted_bytes as f32 / packed.data.len() as f32
    }

    /// Calculate CRC32 checksum
    fn calculate_crc32(&self, data: &[u8]) -> u32 {
        // Simple CRC32 implementation
        const CRC32_TABLE: [u32; 256] = generate_crc32_table();

        let mut crc = 0xFFFFFFFF;
        for &byte in data {
            let table_index = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[table_index];
        }
        !crc
    }

    /// Attempt to partially unpack data for validation
    fn attempt_partial_unpack(
        &self,
        packed: &PackedTernaryWeights,
        max_elements: usize,
    ) -> Result<Vec<i8>, QuantizationError> {
        // Create a modified packed structure with limited element count
        let mut test_packed = packed.clone();
        test_packed.metadata.element_count = max_elements.min(packed.metadata.element_count);

        // Use the appropriate unpacker based on strategy
        use super::packing::TernaryPackerFactory;
        let packer = TernaryPackerFactory::create_packer(packed.strategy);
        packer.unpack(&test_packed)
    }
}

/// Trait for strategy-specific validators
pub trait StrategyValidator: Send + Sync {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError>;
}

/// Validator for BitPacked2Bit strategy
struct BitPacked2BitValidator;

impl StrategyValidator for BitPacked2BitValidator {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Check that each byte contains valid 2-bit values (0, 1, 2)
        for (i, &byte) in packed.data.iter().enumerate() {
            for shift in [0, 2, 4, 6] {
                let value = (byte >> shift) & 0x03;
                if value > 2 {
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::InvalidValues {
                            invalid_bytes: vec![value],
                            positions: vec![i],
                            expected_range: "0-2".to_string(),
                        },
                        severity: CorruptionSeverity::Moderate,
                        confidence: 1.0,
                        byte_offset: Some(i),
                        corrupted_length: Some(1),
                        recovery_suggestions: vec![RecoveryAction::AutoRepair {
                            description: "Clamp invalid values to valid range".to_string(),
                        }],
                        context: HashMap::new(),
                    });
                }
            }
        }

        Ok(reports)
    }
}

/// Validator for Base3Packed strategy
struct Base3PackedValidator;

impl StrategyValidator for Base3PackedValidator {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Check that each byte represents a valid base-3 number
        for (i, &byte) in packed.data.iter().enumerate() {
            if byte > 242 {
                // 3^5 - 1 = 242 is the maximum valid value
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::InvalidValues {
                        invalid_bytes: vec![byte],
                        positions: vec![i],
                        expected_range: "0-242".to_string(),
                    },
                    severity: CorruptionSeverity::Moderate,
                    confidence: 0.9,
                    byte_offset: Some(i),
                    corrupted_length: Some(1),
                    recovery_suggestions: vec![RecoveryAction::AutoRepair {
                        description: "Replace invalid byte with nearest valid value".to_string(),
                    }],
                    context: HashMap::new(),
                });
            }
        }

        Ok(reports)
    }
}

/// Validator for RunLengthEncoded strategy
struct RunLengthEncodedValidator;

impl StrategyValidator for RunLengthEncodedValidator {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        // Check that data length is even (value-count pairs)
        if packed.data.len() % 2 != 0 {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::StructuralCorruption {
                    description: "RLE data length must be even (value-count pairs)".to_string(),
                    recovery_possible: true,
                },
                severity: CorruptionSeverity::Severe,
                confidence: 1.0,
                byte_offset: Some(packed.data.len() - 1),
                corrupted_length: Some(1),
                recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                    safe_length: packed.data.len() - 1,
                }],
                context: HashMap::new(),
            });
        }

        // Validate value-count pairs
        let mut total_elements = 0;
        for chunk in packed.data.chunks(2) {
            if chunk.len() == 2 {
                let value = chunk[0];
                let count = chunk[1] as usize;

                // Check value is in valid range (0, 1, 2 for ternary)
                if value > 2 {
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::InvalidValues {
                            invalid_bytes: vec![value],
                            positions: vec![total_elements],
                            expected_range: "0-2".to_string(),
                        },
                        severity: CorruptionSeverity::Moderate,
                        confidence: 1.0,
                        byte_offset: Some(chunk.as_ptr() as usize - packed.data.as_ptr() as usize),
                        corrupted_length: Some(1),
                        recovery_suggestions: vec![RecoveryAction::AutoRepair {
                            description: "Clamp value to valid range".to_string(),
                        }],
                        context: HashMap::new(),
                    });
                }

                // Check count is reasonable (not zero, not too large)
                if count == 0 {
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::InvalidValues {
                            invalid_bytes: vec![chunk[1]],
                            positions: vec![total_elements],
                            expected_range: "1-255".to_string(),
                        },
                        severity: CorruptionSeverity::Severe,
                        confidence: 1.0,
                        byte_offset: Some(
                            chunk.as_ptr() as usize - packed.data.as_ptr() as usize + 1,
                        ),
                        corrupted_length: Some(1),
                        recovery_suggestions: vec![RecoveryAction::AutoRepair {
                            description: "Set count to 1".to_string(),
                        }],
                        context: HashMap::new(),
                    });
                }

                total_elements += count;
            }
        }

        // Check total elements matches metadata
        if total_elements != packed.metadata.element_count {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::MetadataInconsistency {
                    field: "element_count".to_string(),
                    expected: packed.metadata.element_count.to_string(),
                    actual: total_elements.to_string(),
                },
                severity: CorruptionSeverity::Severe,
                confidence: 1.0,
                byte_offset: None,
                corrupted_length: None,
                recovery_suggestions: vec![RecoveryAction::AutoRepair {
                    description: "Update metadata element count to match RLE data".to_string(),
                }],
                context: HashMap::new(),
            });
        }

        Ok(reports)
    }
}

/// Validator for CompressedSparse strategy
struct CompressedSparseValidator;

impl StrategyValidator for CompressedSparseValidator {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();

        if packed.data.len() < 4 {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::StructuralCorruption {
                    description: "Compressed sparse data too short for header".to_string(),
                    recovery_possible: false,
                },
                severity: CorruptionSeverity::Critical,
                confidence: 1.0,
                byte_offset: Some(0),
                corrupted_length: Some(packed.data.len()),
                recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                    instructions: "Data is too short to contain valid sparse header".to_string(),
                }],
                context: HashMap::new(),
            });
            return Ok(reports);
        }

        // Read and validate header
        let nnz = u32::from_le_bytes([
            packed.data[0],
            packed.data[1],
            packed.data[2],
            packed.data[3],
        ]) as usize;

        // Check if nnz is reasonable
        if nnz > packed.metadata.element_count {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::MetadataInconsistency {
                    field: "non_zero_count".to_string(),
                    expected: format!("<= {}", packed.metadata.element_count),
                    actual: nnz.to_string(),
                },
                severity: CorruptionSeverity::Severe,
                confidence: 1.0,
                byte_offset: Some(0),
                corrupted_length: Some(4),
                recovery_suggestions: vec![RecoveryAction::AutoRepair {
                    description: "Clamp non-zero count to element count".to_string(),
                }],
                context: HashMap::new(),
            });
        }

        // Check data size consistency
        let expected_size = 4 + (nnz * 4) + nnz; // header + indices + values
        if packed.data.len() != expected_size {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::SizeMismatch {
                    expected: expected_size,
                    actual: packed.data.len(),
                    context: "sparse data structure".to_string(),
                },
                severity: CorruptionSeverity::Severe,
                confidence: 0.9,
                byte_offset: Some(expected_size.min(packed.data.len())),
                corrupted_length: Some(
                    (expected_size as isize - packed.data.len() as isize).unsigned_abs(),
                ),
                recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                    safe_length: expected_size.min(packed.data.len()),
                }],
                context: HashMap::new(),
            });
        }

        // Validate indices
        let mut offset = 4;
        for _i in 0..nnz.min((packed.data.len().saturating_sub(4)) / 4) {
            if offset + 4 <= packed.data.len() {
                let idx = u32::from_le_bytes([
                    packed.data[offset],
                    packed.data[offset + 1],
                    packed.data[offset + 2],
                    packed.data[offset + 3],
                ]) as usize;

                if idx >= packed.metadata.element_count {
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::IndexOutOfBounds {
                            index: idx,
                            max_valid: packed.metadata.element_count.saturating_sub(1),
                            format: "sparse indices".to_string(),
                        },
                        severity: CorruptionSeverity::Severe,
                        confidence: 1.0,
                        byte_offset: Some(offset),
                        corrupted_length: Some(4),
                        recovery_suggestions: vec![RecoveryAction::AutoRepair {
                            description: "Clamp index to valid range".to_string(),
                        }],
                        context: HashMap::new(),
                    });
                }
                offset += 4;
            }
        }

        // Validate values
        let values_offset = 4 + (nnz * 4);
        for i in 0..nnz.min(packed.data.len().saturating_sub(values_offset)) {
            let value_offset = values_offset + i;
            if value_offset < packed.data.len() {
                let value = packed.data[value_offset];
                if value > 2 {
                    // Ternary values should be 0, 1, or 2
                    reports.push(CorruptionReport {
                        corruption_type: CorruptionType::InvalidValues {
                            invalid_bytes: vec![value],
                            positions: vec![i],
                            expected_range: "0-2".to_string(),
                        },
                        severity: CorruptionSeverity::Moderate,
                        confidence: 1.0,
                        byte_offset: Some(value_offset),
                        corrupted_length: Some(1),
                        recovery_suggestions: vec![RecoveryAction::AutoRepair {
                            description: "Clamp value to valid ternary range".to_string(),
                        }],
                        context: HashMap::new(),
                    });
                }
            }
        }

        Ok(reports)
    }
}

/// Validator for Hybrid strategy
struct HybridValidator;

impl StrategyValidator for HybridValidator {
    fn validate(
        &self,
        packed: &PackedTernaryWeights,
    ) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();
        let mut offset = 0;

        while offset < packed.data.len() {
            if offset + 3 > packed.data.len() {
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::StructuralCorruption {
                        description: "Incomplete hybrid block header".to_string(),
                        recovery_possible: true,
                    },
                    severity: CorruptionSeverity::Severe,
                    confidence: 1.0,
                    byte_offset: Some(offset),
                    corrupted_length: Some(packed.data.len() - offset),
                    recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                        safe_length: offset,
                    }],
                    context: HashMap::new(),
                });
                break;
            }

            let strategy_byte = packed.data[offset];
            let block_size =
                u16::from_le_bytes([packed.data[offset + 1], packed.data[offset + 2]]) as usize;

            // Validate strategy byte
            if strategy_byte > 6 {
                // Assuming 7 strategies (0-6)
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::InvalidValues {
                        invalid_bytes: vec![strategy_byte],
                        positions: vec![offset],
                        expected_range: "0-6".to_string(),
                    },
                    severity: CorruptionSeverity::Severe,
                    confidence: 1.0,
                    byte_offset: Some(offset),
                    corrupted_length: Some(1),
                    recovery_suggestions: vec![RecoveryAction::AutoRepair {
                        description: "Replace with default strategy (BitPacked2Bit)".to_string(),
                    }],
                    context: HashMap::new(),
                });
            }

            // Check block size is reasonable
            if block_size == 0 || block_size > packed.data.len() {
                reports.push(CorruptionReport {
                    corruption_type: CorruptionType::InvalidValues {
                        invalid_bytes: vec![packed.data[offset + 1], packed.data[offset + 2]],
                        positions: vec![offset + 1],
                        expected_range: format!("1-{}", packed.data.len()),
                    },
                    severity: CorruptionSeverity::Severe,
                    confidence: 1.0,
                    byte_offset: Some(offset + 1),
                    corrupted_length: Some(2),
                    recovery_suggestions: vec![RecoveryAction::DiscardCorrupted {
                        safe_length: offset,
                    }],
                    context: HashMap::new(),
                });
                break;
            }

            offset += 3 + block_size;
        }

        Ok(reports)
    }
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

/// Utility functions for corruption detection
impl CorruptionDetector {
    /// Add checksum to packed data metadata
    pub fn add_checksum(&self, packed: &mut PackedTernaryWeights) {
        if self.enable_checksums {
            let checksum = self.calculate_crc32(&packed.data);
            packed
                .metadata
                .extra_data
                .insert("crc32".to_string(), checksum.to_le_bytes().to_vec());
        }
    }

    /// Attempt to repair minor corruption automatically
    pub fn attempt_repair(
        &self,
        packed: &mut PackedTernaryWeights,
        reports: &[CorruptionReport],
    ) -> Result<usize, QuantizationError> {
        let mut repairs_made = 0;

        for report in reports {
            match &report.corruption_type {
                CorruptionType::InvalidValues {
                    invalid_bytes,
                    positions,
                    ..
                } => {
                    if report.severity <= CorruptionSeverity::Moderate {
                        // Attempt to clamp invalid values
                        for (&pos, &_invalid_byte) in positions.iter().zip(invalid_bytes.iter()) {
                            if pos < packed.data.len() {
                                // Clamp to valid ternary range (0-2)
                                packed.data[pos] = packed.data[pos].min(2);
                                repairs_made += 1;
                            }
                        }
                    }
                }
                CorruptionType::PaddingCorruption {
                    expected_padding, ..
                } => {
                    // Fix padding in metadata
                    packed.metadata.padding = *expected_padding;
                    repairs_made += 1;
                }
                _ => {
                    // Other corruption types require manual intervention
                }
            }
        }

        // Recalculate checksum after repairs
        if repairs_made > 0 && self.enable_checksums {
            self.add_checksum(packed);
        }

        Ok(repairs_made)
    }

    /// Create a recovery plan for corrupted data
    pub fn create_recovery_plan(&self, reports: &[CorruptionReport]) -> RecoveryPlan {
        let mut plan = RecoveryPlan {
            auto_repairable: Vec::new(),
            requires_fallback: Vec::new(),
            requires_manual_intervention: Vec::new(),
            recommended_strategy: None,
        };

        for report in reports {
            match report.severity {
                CorruptionSeverity::Minor | CorruptionSeverity::Moderate => {
                    if report
                        .recovery_suggestions
                        .iter()
                        .any(|s| matches!(s, RecoveryAction::AutoRepair { .. }))
                    {
                        plan.auto_repairable.push(report.clone());
                    }
                }
                CorruptionSeverity::Severe => {
                    if report
                        .recovery_suggestions
                        .iter()
                        .any(|s| matches!(s, RecoveryAction::UseFallback { .. }))
                    {
                        plan.requires_fallback.push(report.clone());
                    } else {
                        plan.requires_manual_intervention.push(report.clone());
                    }
                }
                CorruptionSeverity::Critical => {
                    plan.requires_manual_intervention.push(report.clone());
                }
            }
        }

        // Recommend fallback strategy if needed
        if !plan.requires_fallback.is_empty() {
            plan.recommended_strategy = Some(TernaryPackingStrategy::Uncompressed);
        }

        plan
    }
}

/// Recovery plan for corrupted data
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    pub auto_repairable: Vec<CorruptionReport>,
    pub requires_fallback: Vec<CorruptionReport>,
    pub requires_manual_intervention: Vec<CorruptionReport>,
    pub recommended_strategy: Option<TernaryPackingStrategy>,
}

impl RecoveryPlan {
    /// Check if the data can be automatically recovered
    pub fn is_auto_recoverable(&self) -> bool {
        self.requires_manual_intervention.is_empty() && self.requires_fallback.is_empty()
    }

    /// Check if fallback strategy can be used
    pub fn can_use_fallback(&self) -> bool {
        self.requires_manual_intervention.is_empty()
    }

    /// Get total number of issues
    pub fn total_issues(&self) -> usize {
        self.auto_repairable.len()
            + self.requires_fallback.len()
            + self.requires_manual_intervention.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::packing::*;

    #[test]
    fn test_corruption_detector_creation() {
        let detector = CorruptionDetector::default();
        assert!(detector.enable_checksums);
        assert!(detector.enable_deep_validation);
        assert_eq!(detector.max_corruption_ratio, 0.1);
    }

    #[test]
    fn test_basic_structure_validation() {
        let detector = CorruptionDetector::default();

        // Test empty data with non-zero element count
        let packed = PackedTernaryWeights {
            data: Vec::new(),
            shape: candle_core::Shape::from_dims(&[10]),
            strategy: TernaryPackingStrategy::BitPacked2Bit,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count: 10,
                ..Default::default()
            },
            memory_footprint: 0,
            compression_ratio: 1.0,
        };

        let reports = detector.validate_basic_structure(&packed).unwrap();
        assert!(!reports.is_empty());
        assert!(matches!(
            reports[0].corruption_type,
            CorruptionType::SizeMismatch { .. }
        ));
    }

    #[test]
    fn test_checksum_validation() {
        let detector = CorruptionDetector::default();

        let mut packed = PackedTernaryWeights {
            data: vec![1, 2, 3, 4],
            shape: candle_core::Shape::from_dims(&[4]),
            strategy: TernaryPackingStrategy::Uncompressed,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count: 4,
                ..Default::default()
            },
            memory_footprint: 4,
            compression_ratio: 1.0,
        };

        // Add correct checksum
        detector.add_checksum(&mut packed);

        // Validation should pass
        let report = detector.validate_checksum(&packed).unwrap();
        assert!(report.is_none());

        // Corrupt the data
        packed.data[0] = 255;

        // Validation should fail
        let report = detector.validate_checksum(&packed).unwrap();
        assert!(report.is_some());
        assert!(matches!(
            report.unwrap().corruption_type,
            CorruptionType::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn test_bit_packed_validator() {
        let validator = BitPacked2BitValidator;

        let packed = PackedTernaryWeights {
            data: vec![0b11100100], // Contains invalid value 3 (11 in binary)
            shape: candle_core::Shape::from_dims(&[4]),
            strategy: TernaryPackingStrategy::BitPacked2Bit,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count: 4,
                ..Default::default()
            },
            memory_footprint: 1,
            compression_ratio: 4.0,
        };

        let reports = validator.validate(&packed).unwrap();
        assert!(!reports.is_empty());
        assert!(matches!(
            reports[0].corruption_type,
            CorruptionType::InvalidValues { .. }
        ));
    }

    #[test]
    fn test_recovery_plan_creation() {
        let detector = CorruptionDetector::default();

        let reports = vec![CorruptionReport {
            corruption_type: CorruptionType::InvalidValues {
                invalid_bytes: vec![3],
                positions: vec![0],
                expected_range: "0-2".to_string(),
            },
            severity: CorruptionSeverity::Moderate,
            confidence: 1.0,
            byte_offset: Some(0),
            corrupted_length: Some(1),
            recovery_suggestions: vec![RecoveryAction::AutoRepair {
                description: "Clamp to valid range".to_string(),
            }],
            context: HashMap::new(),
        }];

        let plan = detector.create_recovery_plan(&reports);
        assert_eq!(plan.auto_repairable.len(), 1);
        assert!(plan.is_auto_recoverable());
    }

    #[test]
    fn test_crc32_calculation() {
        let detector = CorruptionDetector::default();

        let data = b"hello world";
        let crc1 = detector.calculate_crc32(data);
        let crc2 = detector.calculate_crc32(data);

        // Same data should produce same CRC
        assert_eq!(crc1, crc2);

        // Different data should produce different CRC
        let crc3 = detector.calculate_crc32(b"hello world!");
        assert_ne!(crc1, crc3);
    }
}
