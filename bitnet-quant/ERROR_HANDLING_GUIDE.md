# Advanced Error Checking and Corruption Detection Guide

This guide provides comprehensive information about the advanced error checking and corruption detection features implemented for packed ternary data in the BitNet quantization library.

## Overview

The corruption detection system provides multiple layers of validation to ensure data integrity:

1. **Basic Structure Validation** - Checks data size consistency and basic format requirements
2. **Metadata Validation** - Verifies metadata consistency with actual data
3. **Checksum Validation** - Uses CRC32 checksums to detect data corruption
4. **Strategy-Specific Validation** - Validates data according to specific packing strategy requirements
5. **Deep Validation** - Attempts partial unpacking to detect structural issues

## Corruption Types

### Size Mismatch
Occurs when the actual data size doesn't match expected size based on metadata.

```rust
CorruptionType::SizeMismatch {
    expected: 100,
    actual: 50,
    context: "packed data".to_string(),
}
```

### Invalid Values
Detected when data contains values outside the expected range for the packing strategy.

```rust
CorruptionType::InvalidValues {
    invalid_bytes: vec![3, 4],
    positions: vec![10, 15],
    expected_range: "0-2".to_string(),
}
```

### Checksum Mismatch
Occurs when the calculated checksum doesn't match the stored checksum.

```rust
CorruptionType::ChecksumMismatch {
    expected: 0x12345678,
    actual: 0x87654321,
    algorithm: "CRC32".to_string(),
}
```

### Metadata Inconsistency
Detected when metadata fields are inconsistent with the actual data.

```rust
CorruptionType::MetadataInconsistency {
    field: "element_count".to_string(),
    expected: "100".to_string(),
    actual: "50".to_string(),
}
```

### Strategy-Specific Corruption
Validation errors specific to particular packing strategies.

```rust
CorruptionType::StrategySpecific {
    strategy: TernaryPackingStrategy::BitPacked2Bit,
    details: "Invalid bit pattern detected".to_string(),
}
```

### Structural Corruption
Fundamental structural issues that prevent normal operation.

```rust
CorruptionType::StructuralCorruption {
    description: "Data truncated mid-block".to_string(),
    recovery_possible: true,
}
```

## Severity Levels

### Minor
- Corruption doesn't affect functionality
- Can usually be auto-repaired
- Example: Minor padding inconsistencies

### Moderate
- May cause degraded performance
- Often auto-repairable
- Example: Invalid values that can be clamped

### Severe
- Prevents normal operation
- May require fallback strategies
- Example: Structural format violations

### Critical
- Makes data completely unusable
- Requires manual intervention
- Example: Complete data corruption

## Usage Examples

### Basic Corruption Detection

```rust
use bitnet_quant::prelude::*;

// Create a corruption detector
let detector = CorruptionDetector::default();

// Detect corruption in packed data
let reports = detector.detect_corruption(&packed_weights)?;

// Check if any corruption was found
if !reports.is_empty() {
    println!("Found {} corruption issues", reports.len());
    for report in &reports {
        println!("  - {}: {}", report.severity, report.corruption_type);
    }
}
```

### Custom Corruption Detector

```rust
// Create detector with custom settings
let detector = CorruptionDetector::new(
    true,  // enable checksums
    true,  // enable deep validation
    0.05   // max 5% corruption ratio
);

let reports = detector.detect_corruption(&packed_weights)?;
```

### Automatic Repair

```rust
let detector = CorruptionDetector::default();
let mut packed_weights = /* ... */;

// Detect corruption
let reports = detector.detect_corruption(&packed_weights)?;

// Attempt automatic repair
let repairs_made = detector.attempt_repair(&mut packed_weights, &reports)?;
println!("Made {} automatic repairs", repairs_made);

// Verify repairs worked
let new_reports = detector.detect_corruption(&packed_weights)?;
```

### Recovery Planning

```rust
let detector = CorruptionDetector::default();
let reports = detector.detect_corruption(&packed_weights)?;

// Create a recovery plan
let plan = detector.create_recovery_plan(&reports);

if plan.is_auto_recoverable() {
    println!("All issues can be automatically repaired");
} else if plan.can_use_fallback() {
    println!("Can use fallback strategy: {:?}", plan.recommended_strategy);
} else {
    println!("Manual intervention required for {} issues", 
             plan.requires_manual_intervention.len());
}
```

### Enhanced Packing with Validation

```rust
use bitnet_quant::quantization::packing::*;

let packer = TernaryPackerFactory::create_packer(TernaryPackingStrategy::BitPacked2Bit);
let config = TernaryPackingConfig::default();

// Pack with validation and integrity checking
let packed = packer.pack_with_validation(&weights, &config)?;

// Unpack with corruption detection
let unpacked = packer.unpack_with_validation(&packed)?;
```

## Strategy-Specific Validation

### BitPacked2Bit
- Validates that each 2-bit value is in range [0, 2]
- Checks padding consistency
- Verifies data size matches element count

### Base3Packed
- Ensures each byte represents a valid base-3 number (≤ 242)
- Validates element count consistency
- Checks for proper padding

### RunLengthEncoded
- Verifies data length is even (value-count pairs)
- Validates value range [0, 2] for ternary
- Ensures count values are non-zero
- Checks total element count consistency

### CompressedSparse
- Validates header structure (4-byte non-zero count)
- Checks index bounds
- Verifies data size consistency
- Validates sparse value range

### Hybrid
- Validates block headers
- Checks strategy byte validity
- Verifies block size consistency
- Ensures proper block structure

## Custom Validators

You can implement custom validators for specific needs:

```rust
use bitnet_quant::quantization::corruption_detection::*;

struct CustomValidator;

impl StrategyValidator for CustomValidator {
    fn validate(&self, packed: &PackedTernaryWeights) -> Result<Vec<CorruptionReport>, QuantizationError> {
        let mut reports = Vec::new();
        
        // Custom validation logic here
        if some_custom_check_fails(packed) {
            reports.push(CorruptionReport {
                corruption_type: CorruptionType::StrategySpecific {
                    strategy: packed.strategy,
                    details: "Custom validation failed".to_string(),
                },
                severity: CorruptionSeverity::Moderate,
                confidence: 0.9,
                byte_offset: Some(0),
                corrupted_length: Some(1),
                recovery_suggestions: vec![RecoveryAction::AutoRepair {
                    description: "Apply custom fix".to_string(),
                }],
                context: HashMap::new(),
            });
        }
        
        Ok(reports)
    }
}

// Register the custom validator
let mut detector = CorruptionDetector::default();
detector.register_validator(TernaryPackingStrategy::BitPacked2Bit, Box::new(CustomValidator));
```

## Best Practices

### 1. Always Use Validation in Production
```rust
// Good: Use validation methods
let packed = packer.pack_with_validation(&weights, &config)?;
let unpacked = packer.unpack_with_validation(&packed)?;

// Avoid: Direct packing without validation in production
let packed = packer.pack(&weights, &config)?; // No validation
```

### 2. Handle Corruption Gracefully
```rust
match detector.detect_corruption(&packed_weights) {
    Ok(reports) if reports.is_empty() => {
        // Data is clean, proceed normally
        let unpacked = packer.unpack(&packed_weights)?;
    }
    Ok(reports) => {
        // Corruption detected, attempt recovery
        let plan = detector.create_recovery_plan(&reports);
        if plan.is_auto_recoverable() {
            detector.attempt_repair(&mut packed_weights, &reports)?;
        } else {
            // Use fallback or manual intervention
            return Err(QuantizationError::DataCorruption(
                "Data corruption requires manual intervention".to_string()
            ));
        }
    }
    Err(e) => {
        // Validation itself failed
        return Err(e);
    }
}
```

### 3. Enable Checksums for Critical Data
```rust
let mut config = TernaryPackingConfig::default();
config.enable_compression = true; // Enables checksum generation

let detector = CorruptionDetector::new(true, true, 0.1); // Enable checksum validation
```

### 4. Use Appropriate Corruption Thresholds
```rust
// For critical applications
let detector = CorruptionDetector::new(true, true, 0.01); // 1% max corruption

// For performance-critical applications
let detector = CorruptionDetector::new(true, false, 0.1); // Disable deep validation

// For development/testing
let detector = CorruptionDetector::new(false, false, 1.0); // Minimal validation
```

## Performance Considerations

### Validation Overhead
- Basic validation: ~1-5% overhead
- Checksum validation: ~5-10% overhead
- Deep validation: ~10-20% overhead
- Strategy-specific validation: ~2-8% overhead

### Optimization Tips
1. Disable deep validation for performance-critical paths
2. Use higher corruption thresholds for non-critical data
3. Cache corruption detectors to avoid repeated initialization
4. Consider async validation for large datasets

## Error Recovery Strategies

### Automatic Repair
- Clamp invalid values to valid ranges
- Fix metadata inconsistencies
- Recalculate checksums after repairs

### Fallback Strategies
- Switch to uncompressed format
- Use alternative packing strategy
- Discard corrupted portions

### Manual Intervention
- Log detailed corruption reports
- Provide recovery instructions
- Request data re-packing from source

## Testing Corruption Detection

The library includes comprehensive tests for corruption scenarios:

```bash
# Run corruption detection tests
cargo test corruption_detection

# Run with verbose output
cargo test corruption_detection -- --nocapture

# Test specific corruption types
cargo test test_checksum_validation
cargo test test_bit_packed_validation
```

## Integration with Existing Code

The corruption detection system is designed to integrate seamlessly with existing code:

```rust
// Minimal changes required
let packer = TernaryPackerFactory::create_packer(strategy);

// Before: Direct packing
let packed = packer.pack(&weights, &config)?;

// After: Packing with validation
let packed = packer.pack_with_validation(&weights, &config)?;
```

## Monitoring and Logging

Consider implementing monitoring for corruption detection:

```rust
use log::{warn, error, info};

let reports = detector.detect_corruption(&packed_weights)?;
for report in &reports {
    match report.severity {
        CorruptionSeverity::Critical => error!("Critical corruption: {}", report.corruption_type),
        CorruptionSeverity::Severe => warn!("Severe corruption: {}", report.corruption_type),
        CorruptionSeverity::Moderate => warn!("Moderate corruption: {}", report.corruption_type),
        CorruptionSeverity::Minor => info!("Minor corruption: {}", report.corruption_type),
    }
}
```

This comprehensive error checking system ensures data integrity while providing flexible recovery options for various corruption scenarios.