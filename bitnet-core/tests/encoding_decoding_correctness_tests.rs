//! Comprehensive Encoding/Decoding Correctness Tests for BitNet Core
//!
//! This module provides exhaustive correctness testing for all encoding/decoding
//! functionality in the BitNet Core library, including:
//! - Tokenizer encoding/decoding correctness
//! - Quantization/dequantization correctness
//! - Round-trip validation for all encoding/decoding paths
//! - Edge cases and error conditions

use bitnet_core::tokenizer::{
    create_simple_tokenizer, encode_text, decode_tokens, encode_batch
};
use bitnet_core::memory::tensor::dtype::BitNetDType;
use std::collections::HashMap;
use anyhow::Result;

/// Test helper to create a comprehensive test tokenizer
fn create_comprehensive_test_tokenizer() -> bitnet_core::tokenizer::Tokenizer {
    let mut vocab = HashMap::new();
    
    // Basic vocabulary
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("the".to_string(), 2);
    vocab.insert("quick".to_string(), 3);
    vocab.insert("brown".to_string(), 4);
    vocab.insert("fox".to_string(), 5);
    vocab.insert("jumps".to_string(), 6);
    vocab.insert("over".to_string(), 7);
    vocab.insert("lazy".to_string(), 8);
    vocab.insert("dog".to_string(), 9);
    
    // Special tokens
    vocab.insert("<pad>".to_string(), 10);
    vocab.insert("<unk>".to_string(), 11);
    vocab.insert("<cls>".to_string(), 12);
    vocab.insert("<sep>".to_string(), 13);
    vocab.insert("<mask>".to_string(), 14);
    vocab.insert("<bos>".to_string(), 15);
    vocab.insert("<eos>".to_string(), 16);
    
    // Unicode and special characters
    vocab.insert("café".to_string(), 17);
    vocab.insert("naïve".to_string(), 18);
    vocab.insert("🌍".to_string(), 19);
    vocab.insert("测试".to_string(), 20);
    
    create_simple_tokenizer(vocab)
}

/// Test helper to create test data with various characteristics
fn create_test_texts() -> Vec<&'static str> {
    vec![
        // Basic texts
        "hello world",
        "the quick brown fox jumps over lazy dog",
        
        // Empty and whitespace
        "",
        " ",
        "   ",
        "\t\n",
        
        // Single words
        "hello",
        "world",
        
        // Repeated words
        "hello hello hello",
        "the the the the",
        
        // Mixed case (will be normalized by simple tokenizer)
        "Hello World",
        "THE QUICK BROWN FOX",
        
        // Unicode content
        "café naïve",
        "hello 🌍 world",
        "测试 hello world",
        
        // Long sequences
        "the quick brown fox jumps over the lazy dog and then the fox runs back",
        
        // Unknown words (will use <unk>)
        "hello unknown_word world",
        "completely_unknown_sequence",
        
        // Punctuation and special characters (treated as unknown)
        "hello, world!",
        "test@example.com",
        "price: $19.99",
    ]
}

// =============================================================================
// TOKENIZER ENCODING/DECODING CORRECTNESS TESTS
// =============================================================================

#[test]
fn test_tokenizer_round_trip_correctness() {
    let tokenizer = create_comprehensive_test_tokenizer();
    let test_texts = create_test_texts();
    
    for text in test_texts {
        // Skip empty strings as they produce empty token sequences
        if text.trim().is_empty() {
            continue;
        }
        
        // Encode
        let tokens = encode_text(&tokenizer, text).unwrap();
        
        // Decode
        let decoded = decode_tokens(&tokenizer, &tokens).unwrap();
        
        // For simple tokenizer, we expect word-level tokenization
        // So we compare normalized versions (whitespace-normalized)
        let normalized_original = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let normalized_decoded = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
        
        // Handle unknown tokens - they should become <unk>
        let has_unknown_words = text.split_whitespace().any(|word| {
            !["hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
              "<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<bos>", "<eos>", "café", "naïve", "🌍", "测试"]
              .contains(&word.to_lowercase().as_str())
        });
        
        if text.contains("unknown") || text.contains("@") || text.contains("$") || text.contains(",") || text.contains("!") ||
           text.chars().any(|c| c.is_uppercase()) || has_unknown_words {
            // Text with unknown tokens should have <unk> in decoded version
            assert!(decoded.contains("<unk>"), "Expected <unk> in decoded text for: '{}'", text);
        } else if !normalized_original.is_empty() {
            assert_eq!(normalized_decoded, normalized_original,
                "Round-trip failed for text: '{}' -> tokens: {:?} -> decoded: '{}'",
                text, tokens, decoded);
        }
    }
}

#[test]
fn test_tokenizer_batch_encoding_correctness() {
    let tokenizer = create_comprehensive_test_tokenizer();
    let test_texts = create_test_texts();
    
    // Test batch encoding
    let batch_tokens = encode_batch(&tokenizer, &test_texts).unwrap();
    
    // Verify batch results match individual encoding
    for (i, text) in test_texts.iter().enumerate() {
        let individual_tokens = encode_text(&tokenizer, text).unwrap();
        assert_eq!(batch_tokens[i], individual_tokens,
            "Batch encoding mismatch for text '{}' at index {}", text, i);
    }
    
    // Test batch decoding
    for (i, tokens) in batch_tokens.iter().enumerate() {
        let decoded = decode_tokens(&tokenizer, tokens).unwrap();
        
        // Verify decoding is consistent
        let re_encoded = encode_text(&tokenizer, &decoded).unwrap();
        assert_eq!(*tokens, re_encoded,
            "Batch decode-encode inconsistency for original text: '{}'", test_texts[i]);
    }
}

#[test]
fn test_tokenizer_special_token_handling() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    // Test texts with special tokens
    let special_texts = vec![
        "<cls> hello world <sep>",
        "<bos> the quick brown fox <eos>",
        "hello <mask> world",
        "<pad> <pad> <pad>",
    ];
    
    for text in special_texts {
        let tokens = encode_text(&tokenizer, text).unwrap();
        let decoded = decode_tokens(&tokenizer, &tokens).unwrap();
        
        // Special tokens should be preserved in round-trip
        assert_eq!(text, decoded, "Special token handling failed for: '{}'", text);
    }
}

#[test]
fn test_tokenizer_unicode_correctness() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    let unicode_texts = vec![
        "café",
        "naïve", 
        "🌍",
        "测试",
        "café naïve 🌍",
        "hello 🌍 测试 world",
    ];
    
    for text in unicode_texts {
        let tokens = encode_text(&tokenizer, text).unwrap();
        let decoded = decode_tokens(&tokenizer, &tokens).unwrap();
        
        // Unicode should be preserved
        assert_eq!(text, decoded, "Unicode handling failed for: '{}'", text);
    }
}

#[test]
fn test_tokenizer_edge_cases() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    // Test empty token sequence
    let empty_decoded = decode_tokens(&tokenizer, &[]).unwrap();
    assert_eq!(empty_decoded, "");
    
    // Test single token
    let single_token = encode_text(&tokenizer, "hello").unwrap();
    assert_eq!(single_token, vec![0]);
    let single_decoded = decode_tokens(&tokenizer, &[0]).unwrap();
    assert_eq!(single_decoded, "hello");
    
    // Test invalid token ID (should error)
    let invalid_result = decode_tokens(&tokenizer, &[9999]);
    assert!(invalid_result.is_err(), "Should fail for invalid token ID");
    
    // Test maximum valid token ID
    let max_token = 20; // Our vocab goes up to 20
    let max_decoded = decode_tokens(&tokenizer, &[max_token]).unwrap();
    assert_eq!(max_decoded, "测试");
}

// =============================================================================
// ADVANCED TOKENIZER CORRECTNESS TESTS
// =============================================================================

#[test]
fn test_tokenizer_consistency_across_operations() {
    let tokenizer = create_comprehensive_test_tokenizer();
    let test_text = "the quick brown fox jumps over the lazy dog";
    
    // Test consistency across multiple runs
    let mut all_results = Vec::new();
    for _ in 0..10 {
        let tokens = encode_text(&tokenizer, test_text).unwrap();
        all_results.push(tokens);
    }
    
    // All results should be identical
    let first_result = &all_results[0];
    for (i, result) in all_results.iter().enumerate() {
        assert_eq!(result, first_result, "Inconsistent result at iteration {}", i);
    }
    
    // Test batch vs individual consistency
    let texts = vec![test_text; 5];
    let batch_results = encode_batch(&tokenizer, &texts).unwrap();
    
    for (i, batch_result) in batch_results.iter().enumerate() {
        assert_eq!(batch_result, first_result, "Batch result {} differs from individual", i);
    }
}

#[test]
fn test_tokenizer_special_token_isolation() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    // Test that special tokens don't interfere with regular tokenization
    let regular_text = "hello world";
    let regular_tokens = encode_text(&tokenizer, regular_text).unwrap();
    
    // Test with special tokens mixed in
    let mixed_text = "<cls> hello world <sep>";
    let mixed_tokens = encode_text(&tokenizer, mixed_text).unwrap();
    
    // The middle tokens should match the regular tokens
    assert!(mixed_tokens.len() >= regular_tokens.len() + 2);
    
    // Extract middle tokens (skip first and last special tokens)
    let middle_tokens = &mixed_tokens[1..mixed_tokens.len()-1];
    assert_eq!(middle_tokens, regular_tokens.as_slice(),
        "Special tokens should not affect regular token encoding");
}

// =============================================================================
// QUANTIZATION ENCODING/DECODING CORRECTNESS TESTS
// =============================================================================

#[cfg(feature = "mlx")]
mod mlx_quantization_tests {
    use super::*;
    use bitnet_core::mlx::{mlx_quantize, mlx_dequantize};
    use mlx_rs::Array;
    
    #[test]
    fn test_mlx_quantization_round_trip_correctness() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, 3.0, 4.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.5, 2.7, -3.2, 4.8],
            vec![100.0, -200.0, 300.0, -400.0],
        ];
        
        let scales = vec![1.0, 0.5, 2.0, 0.1, 10.0];
        
        for (data, scale) in test_data.iter().zip(scales.iter()) {
            let array = Array::from_slice(data, &[data.len() as i32]);
            
            // Quantize
            let quantized = mlx_quantize(&array, *scale).unwrap();
            
            // Dequantize
            let dequantized = mlx_dequantize(&quantized, *scale).unwrap();
            
            // Verify shapes match
            assert_eq!(array.shape(), dequantized.shape());
            
            // Verify values are approximately correct (within quantization error)
            let original_data = array.as_slice::<f32>();
            let recovered_data = dequantized.as_slice::<f32>();
            
            for (i, (&orig, &recovered)) in original_data.iter().zip(recovered_data.iter()).enumerate() {
                let error = (orig - recovered).abs();
                let max_error = scale * 0.5; // Maximum quantization error
                assert!(error <= max_error + 1e-6, 
                    "Quantization error too large at index {}: original={}, recovered={}, error={}, max_allowed={}", 
                    i, orig, recovered, error, max_error);
            }
        }
    }
    
    #[test]
    fn test_mlx_quantization_edge_cases() {
        // Test with NaN and infinity
        let edge_data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let array = Array::from_slice(&edge_data, &[4]);
        
        let quantized = mlx_quantize(&array, 1.0).unwrap();
        let dequantized = mlx_dequantize(&quantized, 1.0).unwrap();
        
        let recovered_data = dequantized.as_slice::<f32>();
        
        // NaN should become 0 after quantization
        assert!(recovered_data[0].is_finite());
        
        // Infinity should be clamped
        assert!(recovered_data[1].is_finite());
        assert!(recovered_data[2].is_finite());
        
        // Zero should remain zero
        assert_eq!(recovered_data[3], 0.0);
    }
    
    #[test]
    fn test_mlx_quantization_precision_levels() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let array = Array::from_slice(&data, &[8]);
        
        // Test different precision levels (different scales)
        let precision_scales = vec![
            1.0,    // 1-bit precision
            0.5,    // 2-bit precision  
            0.25,   // 4-bit precision
            0.125,  // 8-bit precision
        ];
        
        for scale in precision_scales {
            let quantized = mlx_quantize(&array, scale).unwrap();
            let dequantized = mlx_dequantize(&quantized, scale).unwrap();
            
            // Higher precision (smaller scale) should have lower error
            let original_data = array.as_slice::<f32>();
            let recovered_data = dequantized.as_slice::<f32>();
            
            let max_error = original_data.iter()
                .zip(recovered_data.iter())
                .map(|(&orig, &recovered)| (orig - recovered).abs())
                .fold(0.0f32, f32::max);
            
            // Error should be bounded by quantization step
            assert!(max_error <= scale * 0.5 + 1e-6, 
                "Max error {} exceeds expected bound {} for scale {}", 
                max_error, scale * 0.5, scale);
        }
    }
}

// Memory conversion quantization tests (always available)
#[test]
fn test_memory_conversion_quantization_correctness() {
    // Test data type properties
    assert!(BitNetDType::I4.is_quantized());
    assert!(BitNetDType::I2.is_quantized());
    assert!(BitNetDType::I1.is_quantized());
    assert!(BitNetDType::BitNet158.is_quantized());
    
    assert!(!BitNetDType::F32.is_quantized());
    assert!(!BitNetDType::F16.is_quantized());
    assert!(!BitNetDType::I8.is_quantized());
    
    // Test bits per element calculations
    assert_eq!(BitNetDType::F32.bits_per_element(), 32);
    assert_eq!(BitNetDType::F16.bits_per_element(), 16);
    assert_eq!(BitNetDType::I8.bits_per_element(), 8);
    assert_eq!(BitNetDType::I4.bits_per_element(), 4);
    assert_eq!(BitNetDType::I2.bits_per_element(), 2);
    assert_eq!(BitNetDType::I1.bits_per_element(), 1);
    assert_eq!(BitNetDType::BitNet158.bits_per_element(), 2);
    
    // Test bytes for elements calculations
    assert_eq!(BitNetDType::F32.bytes_for_elements(4), 16);
    assert_eq!(BitNetDType::I8.bytes_for_elements(4), 4);
    assert_eq!(BitNetDType::I4.bytes_for_elements(4), 2);
    assert_eq!(BitNetDType::I2.bytes_for_elements(4), 1);
    assert_eq!(BitNetDType::I1.bytes_for_elements(8), 1);
}

#[test]
fn test_quantization_data_type_conversions() {
    // Test all data type conversions are defined
    let types = vec![
        BitNetDType::F32,
        BitNetDType::F16,
        BitNetDType::I8,
        BitNetDType::I4,
        BitNetDType::I2,
        BitNetDType::I1,
        BitNetDType::BitNet158,
    ];
    
    for from_type in &types {
        for to_type in &types {
            // All conversions should have defined behavior
            let from_bits = from_type.bits_per_element();
            let to_bits = to_type.bits_per_element();
            
            // Verify bit relationships make sense
            if from_type.is_quantized() && !to_type.is_quantized() {
                // Dequantization: quantized -> full precision
                assert!(to_bits >= from_bits,
                    "Dequantization should not decrease precision: {:?} -> {:?}", from_type, to_type);
            }
            
            if !from_type.is_quantized() && to_type.is_quantized() {
                // Quantization: full precision -> quantized
                assert!(to_bits <= from_bits,
                    "Quantization should not increase precision: {:?} -> {:?}", from_type, to_type);
            }
        }
    }
}

#[test]
fn test_quantization_value_ranges() {
    // Test value ranges for quantized types
    assert_eq!(BitNetDType::I8.value_range(), Some((-128, 127)));
    assert_eq!(BitNetDType::I4.value_range(), Some((-8, 7)));
    assert_eq!(BitNetDType::I2.value_range(), Some((-2, 1)));
    assert_eq!(BitNetDType::I1.value_range(), Some((-1, 0)));
    assert_eq!(BitNetDType::BitNet158.value_range(), Some((-1, 1)));
    
    // Floating point types should not have fixed ranges
    assert_eq!(BitNetDType::F32.value_range(), None);
    assert_eq!(BitNetDType::F16.value_range(), None);
}

#[test]
fn test_quantization_memory_efficiency() {
    // Test memory efficiency calculations
    assert_eq!(BitNetDType::F32.memory_efficiency(), 1.0);
    assert_eq!(BitNetDType::F16.memory_efficiency(), 2.0);
    assert_eq!(BitNetDType::I8.memory_efficiency(), 4.0);
    assert_eq!(BitNetDType::I4.memory_efficiency(), 8.0);
    assert_eq!(BitNetDType::I2.memory_efficiency(), 16.0);
    assert_eq!(BitNetDType::I1.memory_efficiency(), 32.0);
    assert_eq!(BitNetDType::BitNet158.memory_efficiency(), 16.0); // 2 bits per element
}

// =============================================================================
// COMPREHENSIVE INTEGRATION TESTS
// =============================================================================

#[test]
fn test_end_to_end_text_processing_pipeline() {
    // Create a complete text processing pipeline
    let tokenizer = create_comprehensive_test_tokenizer();
    
    let input_texts = vec![
        "Hello, how are you today?",
        "The weather is beautiful!",
        "Machine learning is fascinating.",
    ];
    
    // Process through complete pipeline
    let batch_results = encode_batch(&tokenizer, &input_texts).unwrap();
    
    // Verify we can decode back
    let mut decoded_texts = Vec::new();
    for tokens in &batch_results {
        let decoded = decode_tokens(&tokenizer, tokens).unwrap();
        decoded_texts.push(decoded);
    }
    
    // Verify pipeline integrity
    assert_eq!(decoded_texts.len(), input_texts.len());
    assert_eq!(batch_results.len(), input_texts.len());
    
    // Each decoded text should contain recognizable content from original
    for (i, decoded) in decoded_texts.iter().enumerate() {
        // Due to unknown token handling, we check for partial matches
        let original_words: Vec<&str> = input_texts[i].split_whitespace().collect();
        let known_words = vec!["hello", "the", "weather", "machine", "learning"];
        
        let has_known_content = known_words.iter().any(|&word| {
            original_words.iter().any(|&orig_word| {
                orig_word.to_lowercase().contains(word)
            }) && decoded.to_lowercase().contains(word)
        });
        
        if !has_known_content {
            // If no known words, should at least have <unk> tokens
            assert!(decoded.contains("<unk>"),
                "Decoded text should contain either known words or <unk>: '{}'", decoded);
        }
    }
}


#[test]
fn test_comprehensive_error_handling() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    // Test various error conditions
    
    // Invalid token IDs
    let invalid_decode = decode_tokens(&tokenizer, &[9999, 10000]);
    assert!(invalid_decode.is_err(), "Should fail for invalid token IDs");
    
    // Empty tokenizer
    let empty_tokenizer = create_simple_tokenizer(HashMap::new());
    let empty_encode = encode_text(&empty_tokenizer, "hello");
    assert!(empty_encode.is_err(), "Should fail with empty vocabulary");
    
    // Test edge cases that should succeed
    let empty_text_tokens = encode_text(&tokenizer, "").unwrap();
    assert_eq!(empty_text_tokens, Vec::<u32>::new(), "Empty text should produce empty tokens");
    
    let empty_decode = decode_tokens(&tokenizer, &[]).unwrap();
    assert_eq!(empty_decode, "", "Empty tokens should produce empty text");
}

#[test]
fn test_tokenizer_robustness() {
    let tokenizer = create_comprehensive_test_tokenizer();
    
    // Test with various edge case inputs
    let edge_cases = vec![
        "",                    // Empty string
        " ",                   // Single space
        "   ",                 // Multiple spaces
        "\t",                  // Tab
        "\n",                  // Newline
        "\r\n",               // Windows line ending
        "a",                   // Single character
        "hello",               // Single word
        "hello world",         // Two words
        "hello  world",        // Multiple spaces between words
        " hello world ",       // Leading/trailing spaces
    ];
    
    for input in edge_cases {
        let encode_result = encode_text(&tokenizer, input);
        
        if input.trim().is_empty() {
            // Empty or whitespace-only strings should produce empty token sequences
            let tokens = encode_result.unwrap();
            assert_eq!(tokens, Vec::<u32>::new(), "Whitespace-only input should produce empty tokens: '{}'", input);
        } else {
            // Non-empty strings should encode successfully
            assert!(encode_result.is_ok(), "Should successfully encode: '{}'", input);
            
            let tokens = encode_result.unwrap();
            assert!(!tokens.is_empty(), "Non-empty input should produce non-empty tokens: '{}'", input);
            
            // Should be able to decode back
            let decode_result = decode_tokens(&tokenizer, &tokens);
            assert!(decode_result.is_ok(), "Should successfully decode tokens for: '{}'", input);
        }
    }
}