//! Comprehensive tests for corruption detection functionality
//!
//! This module contains extensive tests for various corruption scenarios
//! and validation of the error checking framework.

#[cfg(test)]
mod tests {
    use super::super::corruption_detection::*;
    use super::super::packing::*;
    use super::super::utils::QuantizationError;
    use candle_core::Shape;
    use std::collections::HashMap;

    /// Helper function to create a basic packed weights structure
    fn create_test_packed_weights(
        data: Vec<u8>,
        strategy: TernaryPackingStrategy,
        element_count: usize,
    ) -> PackedTernaryWeights {
        PackedTernaryWeights {
            data,
            shape: Shape::from_dims(&[element_count]),
            strategy,
            config: TernaryPackingConfig::default(),
            metadata: PackingMetadata {
                element_count,
                ..Default::default()
            },
            memory_footprint: 0,
            compression_ratio: 1.0,
        }
    }

    #[test]
    fn test_corruption_detector_basic_creation() {
        let detector = CorruptionDetector::default();
        assert!(detector.enable_checksums);
        assert!(detector.enable_deep_validation);
        assert_eq!(detector.max_corruption_ratio, 0.1);
    }

    #[test]
    fn test_corruption_detector_custom_creation() {
        let detector = CorruptionDetector::new(false, true, 0.2);
        assert!(!detector.enable_checksums);
        assert!(detector.enable_deep_validation);
        assert_eq!(detector.max_corruption_ratio, 0.2);
    }

    #[test]
    fn test_empty_data_corruption() {
        let detector = CorruptionDetector::default();
        let packed = create_test_packed_weights(
            Vec::new(),
            TernaryPackingStrategy::BitPacked2Bit,
            10, // Non-zero element count with empty data
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        assert!(!reports.is_empty());
        
        let size_mismatch_found = reports.iter().any(|r| {
            matches!(r.corruption_type, CorruptionType::SizeMismatch { .. })
        });
        assert!(size_mismatch_found);
    }

    #[test]
    fn test_checksum_validation() {
        let mut detector = CorruptionDetector::default();
        let mut packed = create_test_packed_weights(
            vec![1, 2, 3, 4],
            TernaryPackingStrategy::Uncompressed,
            4,
        );

        // Add correct checksum
        detector.add_checksum(&mut packed);
        
        // Should pass validation
        let reports = detector.detect_corruption(&packed).unwrap();
        let checksum_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::ChecksumMismatch { .. }))
            .collect();
        assert!(checksum_errors.is_empty());

        // Corrupt the data
        packed.data[0] = 255;
        
        // Should fail validation
        let reports = detector.detect_corruption(&packed).unwrap();
        let checksum_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::ChecksumMismatch { .. }))
            .collect();
        assert!(!checksum_errors.is_empty());
    }

    #[test]
    fn test_bit_packed_2bit_validation() {
        let detector = CorruptionDetector::default();
        
        // Create data with invalid 2-bit values (value 3 is invalid for ternary)
        let packed = create_test_packed_weights(
            vec![0b11100100], // Contains invalid value 3 (11 in binary)
            TernaryPackingStrategy::BitPacked2Bit,
            4,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(!invalid_value_errors.is_empty());
    }

    #[test]
    fn test_base3_packed_validation() {
        let detector = CorruptionDetector::default();
        
        // Create data with invalid base-3 value (> 242)
        let packed = create_test_packed_weights(
            vec![250], // Invalid: 250 > 242 (3^5 - 1)
            TernaryPackingStrategy::Base3Packed,
            5,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(!invalid_value_errors.is_empty());
    }

    #[test]
    fn test_run_length_encoded_validation() {
        let detector = CorruptionDetector::default();
        
        // Test odd-length data (should be even for value-count pairs)
        let packed = create_test_packed_weights(
            vec![1, 2, 3], // Odd length
            TernaryPackingStrategy::RunLengthEncoded,
            5,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let structural_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::StructuralCorruption { .. }))
            .collect();
        assert!(!structural_errors.is_empty());

        // Test invalid value in RLE
        let packed = create_test_packed_weights(
            vec![5, 2], // Invalid value 5 (should be 0-2 for ternary)
            TernaryPackingStrategy::RunLengthEncoded,
            2,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(!invalid_value_errors.is_empty());

        // Test zero count in RLE
        let packed = create_test_packed_weights(
            vec![1, 0], // Zero count is invalid
            TernaryPackingStrategy::RunLengthEncoded,
            0,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(!invalid_value_errors.is_empty());
    }

    #[test]
    fn test_compressed_sparse_validation() {
        let detector = CorruptionDetector::default();
        
        // Test data too short for header
        let packed = create_test_packed_weights(
            vec![1, 2], // Too short for 4-byte header
            TernaryPackingStrategy::CompressedSparse,
            10,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let structural_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::StructuralCorruption { .. }))
            .collect();
        assert!(!structural_errors.is_empty());

        // Test invalid non-zero count
        let mut data = Vec::new();
        data.extend_from_slice(&(20u32).to_le_bytes()); // nnz = 20, but element_count = 10
        let packed = create_test_packed_weights(
            data,
            TernaryPackingStrategy::CompressedSparse,
            10, // element_count < nnz
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let metadata_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::MetadataInconsistency { .. }))
            .collect();
        assert!(!metadata_errors.is_empty());
    }

    #[test]
    fn test_hybrid_validation() {
        let detector = CorruptionDetector::default();
        
        // Test incomplete block header
        let packed = create_test_packed_weights(
            vec![1, 2], // Incomplete header (needs 3 bytes)
            TernaryPackingStrategy::Hybrid,
            10,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let structural_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::StructuralCorruption { .. }))
            .collect();
        assert!(!structural_errors.is_empty());

        // Test invalid strategy byte
        let packed = create_test_packed_weights(
            vec![10, 1, 0], // Invalid strategy byte (10 > 6)
            TernaryPackingStrategy::Hybrid,
            1,
        );

        let reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(!invalid_value_errors.is_empty());
    }

    #[test]
    fn test_metadata_validation() {
        let detector = CorruptionDetector::default();
        
        // Test sparse indices out of bounds
        let mut packed = create_test_packed_weights(
            vec![1, 2, 3, 4],
            TernaryPackingStrategy::CompressedSparse,
            5,
        );
        
        packed.metadata.sparse_indices = Some(vec![0, 2, 10]); // Index 10 is out of bounds
        
        let reports = detector.detect_corruption(&packed).unwrap();
        let index_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::IndexOutOfBounds { .. }))
            .collect();
        assert!(!index_errors.is_empty());
    }

    #[test]
    fn test_padding_validation() {
        let detector = CorruptionDetector::default();
        
        let mut packed = create_test_packed_weights(
            vec![1, 2, 3],
            TernaryPackingStrategy::BitPacked2Bit,
            10, // 10 elements need 3 bytes, so 2 padding elements expected
        );
        
        packed.metadata.padding = 5; // Wrong padding value
        
        let reports = detector.detect_corruption(&packed).unwrap();
        let padding_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::PaddingCorruption { .. }))
            .collect();
        assert!(!padding_errors.is_empty());
    }

    #[test]
    fn test_corruption_severity_ordering() {
        assert!(CorruptionSeverity::Minor < CorruptionSeverity::Moderate);
        assert!(CorruptionSeverity::Moderate < CorruptionSeverity::Severe);
        assert!(CorruptionSeverity::Severe < CorruptionSeverity::Critical);
    }

    #[test]
    fn test_recovery_plan_creation() {
        let detector = CorruptionDetector::default();
        
        let reports = vec![
            CorruptionReport {
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
            },
            CorruptionReport {
                corruption_type: CorruptionType::StructuralCorruption {
                    description: "Critical structural damage".to_string(),
                    recovery_possible: false,
                },
                severity: CorruptionSeverity::Critical,
                confidence: 1.0,
                byte_offset: None,
                corrupted_length: None,
                recovery_suggestions: vec![RecoveryAction::ManualIntervention {
                    instructions: "Manual recovery required".to_string(),
                }],
                context: HashMap::new(),
            },
        ];
        
        let plan = detector.create_recovery_plan(&reports);
        assert_eq!(plan.auto_repairable.len(), 1);
        assert_eq!(plan.requires_manual_intervention.len(), 1);
        assert!(!plan.is_auto_recoverable());
        assert!(!plan.can_use_fallback());
        assert_eq!(plan.total_issues(), 2);
    }

    #[test]
    fn test_auto_repair_functionality() {
        let detector = CorruptionDetector::default();
        
        let mut packed = create_test_packed_weights(
            vec![0b11100100], // Contains invalid value 3
            TernaryPackingStrategy::BitPacked2Bit,
            4,
        );
        
        let reports = detector.detect_corruption(&packed).unwrap();
        assert!(!reports.is_empty());
        
        // Attempt repair
        let repairs_made = detector.attempt_repair(&mut packed, &reports).unwrap();
        assert!(repairs_made > 0);
        
        // Verify repair worked
        let new_reports = detector.detect_corruption(&packed).unwrap();
        let invalid_value_errors: Vec<_> = new_reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::InvalidValues { .. }))
            .collect();
        assert!(invalid_value_errors.is_empty() || invalid_value_errors.len() < reports.len());
    }

    #[test]
    fn test_corruption_ratio_calculation() {
        let detector = CorruptionDetector::new(true, true, 0.05); // 5% max corruption
        
        // Create data where more than 5% is corrupted
        let mut packed = create_test_packed_weights(
            vec![255; 100], // All invalid values
            TernaryPackingStrategy::BitPacked2Bit,
            400,
        );
        
        let reports = detector.detect_corruption(&packed).unwrap();
        
        // Should have a critical corruption report due to high corruption ratio
        let critical_errors: Vec<_> = reports.iter()
            .filter(|r| r.severity == CorruptionSeverity::Critical)
            .collect();
        assert!(!critical_errors.is_empty());
    }

    #[test]
    fn test_deep_validation() {
        let detector = CorruptionDetector::new(true, true, 0.1);
        
        // Create structurally invalid data that would fail unpacking
        let packed = create_test_packed_weights(
            vec![255, 255, 255, 255], // Invalid data
            TernaryPackingStrategy::Base3Packed,
            20, // Claims 20 elements but data is invalid
        );
        
        let reports = detector.detect_corruption(&packed).unwrap();
        
        // Should detect structural corruption during deep validation
        let structural_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::StructuralCorruption { .. }))
            .collect();
        assert!(!structural_errors.is_empty());
    }

    #[test]
    fn test_crc32_consistency() {
        let detector = CorruptionDetector::default();
        
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"hello world!";
        
        let crc1 = detector.calculate_crc32(data1);
        let crc2 = detector.calculate_crc32(data2);
        let crc3 = detector.calculate_crc32(data3);
        
        assert_eq!(crc1, crc2); // Same data should produce same CRC
        assert_ne!(crc1, crc3); // Different data should produce different CRC
    }

    #[test]
    fn test_corruption_type_display() {
        let corruption = CorruptionType::SizeMismatch {
            expected: 100,
            actual: 50,
            context: "test data".to_string(),
        };
        
        let display_str = format!("{}", corruption);
        assert!(display_str.contains("Size mismatch"));
        assert!(display_str.contains("100"));
        assert!(display_str.contains("50"));
        assert!(display_str.contains("test data"));
    }

    #[test]
    fn test_recovery_action_serialization() {
        let action = RecoveryAction::AutoRepair {
            description: "Fix invalid values".to_string(),
        };
        
        // Test that it can be serialized/deserialized
        let serialized = serde_json::to_string(&action).unwrap();
        let deserialized: RecoveryAction = serde_json::from_str(&serialized).unwrap();
        
        match (action, deserialized) {
            (RecoveryAction::AutoRepair { description: d1 }, RecoveryAction::AutoRepair { description: d2 }) => {
                assert_eq!(d1, d2);
            }
            _ => panic!("Serialization/deserialization failed"),
        }
    }

    #[test]
    fn test_validator_registration() {
        let mut detector = CorruptionDetector::new(false, false, 1.0);
        
        // Create a custom validator
        struct CustomValidator;
        impl StrategyValidator for CustomValidator {
            fn validate(&self, _packed: &PackedTernaryWeights) -> Result<Vec<CorruptionReport>, QuantizationError> {
                Ok(vec![CorruptionReport {
                    corruption_type: CorruptionType::StrategySpecific {
                        strategy: TernaryPackingStrategy::Uncompressed,
                        details: "Custom validation failed".to_string(),
                    },
                    severity: CorruptionSeverity::Minor,
                    confidence: 1.0,
                    byte_offset: None,
                    corrupted_length: None,
                    recovery_suggestions: Vec::new(),
                    context: HashMap::new(),
                }])
            }
        }
        
        detector.register_validator(TernaryPackingStrategy::Uncompressed, Box::new(CustomValidator));
        
        let packed = create_test_packed_weights(
            vec![1, 2, 3],
            TernaryPackingStrategy::Uncompressed,
            3,
        );
        
        let reports = detector.detect_corruption(&packed).unwrap();
        let custom_errors: Vec<_> = reports.iter()
            .filter(|r| matches!(r.corruption_type, CorruptionType::StrategySpecific { .. }))
            .collect();
        assert!(!custom_errors.is_empty());
    }
}