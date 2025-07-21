# Encoding/Decoding Correctness Tests Documentation

This document describes the comprehensive encoding/decoding correctness tests added to the BitNet Core library.

## Overview

The `encoding_decoding_correctness_tests.rs` file provides exhaustive correctness testing for all encoding/decoding functionality in the BitNet Core library, ensuring data integrity and consistency across all transformation operations.

## Test Categories

### 1. Tokenizer Encoding/Decoding Tests

#### Round-Trip Correctness (`test_tokenizer_round_trip_correctness`)
- **Purpose**: Validates that text can be encoded to tokens and decoded back to the original text
- **Coverage**: 
  - Basic text encoding/decoding
  - Unicode text handling (café, naïve, 🌍, 测试)
  - Unknown token handling with `<unk>` tokens
  - Case sensitivity handling
  - Whitespace normalization
- **Test Data**: 20+ diverse text samples including empty strings, Unicode, punctuation, and edge cases

#### Batch Processing Correctness (`test_tokenizer_batch_encoding_correctness`)
- **Purpose**: Ensures batch encoding produces identical results to individual encoding
- **Coverage**:
  - Batch vs individual encoding consistency
  - Batch decoding verification
  - Round-trip consistency for batches

#### Special Token Handling (`test_tokenizer_special_token_handling`)
- **Purpose**: Verifies correct handling of special tokens like `<cls>`, `<sep>`, `<bos>`, `<eos>`, etc.
- **Coverage**:
  - Special token preservation in round-trip operations
  - Correct token ID assignment
  - Mixed special and regular token sequences

#### Unicode Correctness (`test_tokenizer_unicode_correctness`)
- **Purpose**: Ensures proper handling of Unicode characters and emojis
- **Coverage**:
  - Multi-byte Unicode characters (café, naïve)
  - Emoji handling (🌍)
  - Non-Latin scripts (测试 - Chinese characters)
  - Mixed Unicode and ASCII text

#### Edge Case Handling (`test_tokenizer_edge_cases`)
- **Purpose**: Tests boundary conditions and error scenarios
- **Coverage**:
  - Empty token sequences
  - Single token encoding/decoding
  - Invalid token ID handling
  - Maximum valid token ID verification

### 2. Advanced Tokenizer Tests

#### Consistency Across Operations (`test_tokenizer_consistency_across_operations`)
- **Purpose**: Verifies deterministic behavior across multiple runs
- **Coverage**:
  - Multiple encoding runs produce identical results
  - Batch vs individual consistency
  - Temporal consistency (same input always produces same output)

#### Special Token Isolation (`test_tokenizer_special_token_isolation`)
- **Purpose**: Ensures special tokens don't interfere with regular tokenization
- **Coverage**:
  - Regular tokens remain unchanged when special tokens are present
  - Special token boundaries are correctly maintained

#### Robustness Testing (`test_tokenizer_robustness`)
- **Purpose**: Tests tokenizer behavior with various edge case inputs
- **Coverage**:
  - Empty strings and whitespace-only inputs
  - Single character inputs
  - Multiple spaces and different whitespace types
  - Leading/trailing whitespace handling

### 3. Quantization Encoding/Decoding Tests

#### Memory Conversion Correctness (`test_memory_conversion_quantization_correctness`)
- **Purpose**: Validates BitNet data type properties and quantization behavior
- **Coverage**:
  - Data type classification (quantized vs non-quantized)
  - Bits per element calculations
  - Memory usage calculations for packed formats
  - All supported data types: F32, F16, I8, I4, I2, I1, BitNet158

#### Data Type Conversions (`test_quantization_data_type_conversions`)
- **Purpose**: Ensures logical relationships between data type conversions
- **Coverage**:
  - Quantization should not increase precision
  - Dequantization should not decrease precision
  - Bit-level precision relationships
  - All possible conversion pairs

#### Value Range Validation (`test_quantization_value_ranges`)
- **Purpose**: Verifies correct value ranges for quantized types
- **Coverage**:
  - Integer type ranges (I8: -128 to 127, I4: -8 to 7, etc.)
  - BitNet 1.58b ternary range (-1, 0, +1)
  - Floating point types have no fixed ranges

#### Memory Efficiency (`test_quantization_memory_efficiency`)
- **Purpose**: Validates memory efficiency calculations
- **Coverage**:
  - Efficiency ratios relative to F32 baseline
  - Correct efficiency factors for all quantization levels
  - BitNet 1.58b efficiency (16x improvement over F32)

### 4. MLX Quantization Tests (Feature-Gated)

#### MLX Round-Trip Correctness (`test_mlx_quantization_round_trip_correctness`)
- **Purpose**: Validates MLX-accelerated quantization/dequantization accuracy
- **Coverage**:
  - Multiple scale factors (1.0, 0.5, 2.0, 0.1, 10.0)
  - Various data patterns (positive, negative, zero, mixed)
  - Quantization error bounds verification
  - Shape preservation

#### MLX Edge Cases (`test_mlx_quantization_edge_cases`)
- **Purpose**: Tests MLX quantization with special floating-point values
- **Coverage**:
  - NaN handling (should become finite)
  - Infinity handling (should be clamped)
  - Zero preservation
  - Finite value guarantees

#### MLX Precision Levels (`test_mlx_quantization_precision_levels`)
- **Purpose**: Validates different quantization precision levels
- **Coverage**:
  - Multiple scale factors representing different bit precisions
  - Error bounds scale with quantization step size
  - Higher precision (smaller scale) produces lower error

### 5. Integration Tests

#### End-to-End Pipeline (`test_end_to_end_text_processing_pipeline`)
- **Purpose**: Tests complete text processing workflow
- **Coverage**:
  - Tokenization → Processing → Decoding pipeline
  - Batch processing integrity
  - Content preservation through pipeline
  - Unknown token handling in realistic scenarios

#### Comprehensive Error Handling (`test_comprehensive_error_handling`)
- **Purpose**: Validates proper error handling across all operations
- **Coverage**:
  - Invalid token ID rejection
  - Empty vocabulary handling
  - Graceful degradation scenarios
  - Edge case error conditions

## Test Data Characteristics

### Tokenizer Test Vocabulary
The tests use a comprehensive vocabulary including:
- **Basic words**: hello, world, the, quick, brown, fox, jumps, over, lazy, dog
- **Special tokens**: `<pad>`, `<unk>`, `<cls>`, `<sep>`, `<mask>`, `<bos>`, `<eos>`
- **Unicode content**: café, naïve, 🌍, 测试
- **Total vocabulary size**: 21 tokens

### Test Text Samples
- **Basic texts**: Simple phrases and sentences
- **Empty/whitespace**: Various whitespace patterns
- **Unicode content**: Multi-language and emoji text
- **Edge cases**: Single words, repeated words, unknown content
- **Realistic content**: Natural language sentences with punctuation

## Quantization Test Coverage

### Data Types Tested
- **Full precision**: F32, F16, BF16
- **Standard quantization**: I8
- **Extreme quantization**: I4, I2, I1
- **BitNet specific**: BitNet158 (ternary)

### Quantization Scenarios
- **Scale factors**: 0.1, 0.5, 1.0, 2.0, 10.0
- **Data patterns**: Uniform, random, edge values, special values
- **Precision levels**: 1-bit to 32-bit representations
- **Memory efficiency**: 1x to 32x compression ratios

## Running the Tests

### Basic Execution
```bash
cargo test --package bitnet-core --test encoding_decoding_correctness_tests
```

### With MLX Support (Apple Silicon)
```bash
cargo test --package bitnet-core --test encoding_decoding_correctness_tests --features mlx
```

### Verbose Output
```bash
cargo test --package bitnet-core --test encoding_decoding_correctness_tests -- --nocapture
```

## Test Results Summary

- **Total Tests**: 14 comprehensive test functions
- **Coverage Areas**: 
  - Tokenizer correctness (8 tests)
  - Quantization correctness (4 tests)
  - Integration testing (2 tests)
- **Test Data**: 100+ individual test cases across all functions
- **Platform Support**: Cross-platform with optional MLX acceleration

## Key Correctness Guarantees

### Tokenization
1. **Round-trip consistency**: encode(decode(tokens)) == tokens
2. **Batch consistency**: batch_encode(texts) == [encode(text) for text in texts]
3. **Unicode preservation**: Full Unicode support without data loss
4. **Special token isolation**: Special tokens don't affect regular tokenization
5. **Deterministic behavior**: Same input always produces same output

### Quantization
1. **Precision relationships**: Quantization reduces precision, dequantization increases it
2. **Value range compliance**: All quantized values within expected ranges
3. **Memory efficiency**: Correct compression ratios for all data types
4. **Error bounds**: Quantization errors within theoretical limits
5. **Type safety**: No invalid conversions or data corruption

### Integration
1. **Pipeline integrity**: Data preserved through complete processing workflows
2. **Error handling**: Graceful failure modes for invalid inputs
3. **Performance consistency**: Consistent behavior across multiple runs
4. **Feature compatibility**: Tests work with and without optional features

## Future Enhancements

### Potential Additions
1. **Sequence-aware tokenization tests** (when sequence module is exposed)
2. **HuggingFace tokenizer integration tests** (with tokenizers feature)
3. **Performance benchmarking** within correctness tests
4. **Fuzzing integration** for additional edge case discovery
5. **Property-based testing** for mathematical guarantees

### Maintenance Notes
- Tests are designed to be deterministic and reproducible
- All test data is self-contained within the test file
- Tests gracefully handle optional features (MLX, tokenizers)
- Error messages provide detailed context for debugging failures