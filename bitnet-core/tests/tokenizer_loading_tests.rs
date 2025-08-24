//! Comprehensive tokenizer loading tests for BitNet Core
//!
//! This module tests all aspects of tokenizer loading, creation, and functionality
//! including different tokenizer types, file loading, error handling, and feature flags.

use anyhow::Result;
use bitnet_core::tokenizer::{
    add_special_tokens, create_bpe_tokenizer, create_simple_tokenizer, decode_tokens, encode_batch,
    encode_text, get_special_token_id, load_hf_tokenizer, load_tokenizer,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Test helper to create a temporary directory for test files
fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temporary directory")
}

/// Test helper to create a simple vocabulary file
fn create_test_vocab_file(dir: &Path, filename: &str) -> Result<String> {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0u32);
    vocab.insert("world".to_string(), 1u32);
    vocab.insert("test".to_string(), 2u32);
    vocab.insert("tokenizer".to_string(), 3u32);
    vocab.insert("<unk>".to_string(), 4u32);
    vocab.insert("<pad>".to_string(), 5u32);

    let vocab_path = dir.join(filename);
    let vocab_json = serde_json::to_string_pretty(&vocab)?;
    fs::write(&vocab_path, vocab_json)?;

    Ok(vocab_path.to_string_lossy().to_string())
}

/// Test helper to create a BPE merges file
fn create_test_merges_file(dir: &Path, filename: &str) -> Result<String> {
    let merges_content = "#version: 0.2\n\
                         h e\n\
                         l l\n\
                         o w\n\
                         t e\n\
                         s t\n";

    let merges_path = dir.join(filename);
    fs::write(&merges_path, merges_content)?;

    Ok(merges_path.to_string_lossy().to_string())
}

/// Test helper to create a BPE tokenizer JSON file
fn create_test_bpe_json(dir: &Path, filename: &str) -> Result<String> {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0u32);
    vocab.insert("world".to_string(), 1u32);
    vocab.insert("test".to_string(), 2u32);
    vocab.insert("<unk>".to_string(), 3u32);

    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
    ];

    let bpe_data = serde_json::json!({
        "vocab": vocab,
        "merges": merges,
        "vocab_size": vocab.len()
    });

    let bpe_path = dir.join(filename);
    fs::write(&bpe_path, serde_json::to_string_pretty(&bpe_data)?)?;

    Ok(bpe_path.to_string_lossy().to_string())
}

/// Test helper to create an invalid JSON file
fn create_invalid_json_file(dir: &Path, filename: &str) -> Result<String> {
    let invalid_json = "{ invalid json content }";
    let file_path = dir.join(filename);
    fs::write(&file_path, invalid_json)?;
    Ok(file_path.to_string_lossy().to_string())
}

#[test]
fn test_create_simple_tokenizer() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("test".to_string(), 2);
    vocab.insert("<unk>".to_string(), 3);

    let tokenizer = create_simple_tokenizer(vocab.clone());

    // Test basic properties
    assert_eq!(tokenizer.vocab_size(), vocab.len());

    // Test encoding
    let tokens = encode_text(&tokenizer, "hello world").unwrap();
    assert_eq!(tokens, vec![0, 1]);

    // Test decoding
    let text = decode_tokens(&tokenizer, &[0, 1]).unwrap();
    assert_eq!(text, "hello world");

    // Test unknown token handling
    let tokens = encode_text(&tokenizer, "hello unknown").unwrap();
    assert_eq!(tokens, vec![0, 3]); // "hello" + "<unk>"
}

#[test]
fn test_create_simple_tokenizer_empty_vocab() {
    let vocab = HashMap::new();
    let tokenizer = create_simple_tokenizer(vocab);

    assert_eq!(tokenizer.vocab_size(), 0);

    // Encoding should fail with empty vocab
    let result = encode_text(&tokenizer, "hello");
    assert!(result.is_err());
}

#[test]
fn test_create_simple_tokenizer_single_token() {
    let mut vocab = HashMap::new();
    vocab.insert("only".to_string(), 42);

    let tokenizer = create_simple_tokenizer(vocab);
    assert_eq!(tokenizer.vocab_size(), 1);

    let tokens = encode_text(&tokenizer, "only").unwrap();
    assert_eq!(tokens, vec![42]);

    let text = decode_tokens(&tokenizer, &[42]).unwrap();
    assert_eq!(text, "only");
}

#[test]
fn test_load_tokenizer_nonexistent_file() {
    let result = load_tokenizer("/nonexistent/path/tokenizer.json");
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("not found"));
}

#[test]
fn test_load_tokenizer_invalid_json() {
    let temp_dir = create_test_dir();
    let invalid_file = create_invalid_json_file(temp_dir.path(), "invalid.json").unwrap();

    let result = load_tokenizer(&invalid_file);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unable to load tokenizer"));
}

#[test]
fn test_load_tokenizer_bpe_json() {
    let temp_dir = create_test_dir();
    let bpe_file = create_test_bpe_json(temp_dir.path(), "bpe_tokenizer.json").unwrap();

    let tokenizer = load_tokenizer(&bpe_file).unwrap();
    assert_eq!(tokenizer.vocab_size(), 4);

    // Test that the loaded tokenizer works
    let tokens = encode_text(&tokenizer, "hello world").unwrap();
    assert_eq!(tokens, vec![0, 1]);

    let text = decode_tokens(&tokenizer, &[0, 1]).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn test_load_tokenizer_unsupported_extension() {
    let temp_dir = create_test_dir();
    let unsupported_file = temp_dir.path().join("tokenizer.txt");
    fs::write(&unsupported_file, "some content").unwrap();

    let result = load_tokenizer(&unsupported_file.to_string_lossy());
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unable to load tokenizer"));
}

#[test]
fn test_create_bpe_tokenizer_success() {
    let temp_dir = create_test_dir();
    let vocab_file = create_test_vocab_file(temp_dir.path(), "vocab.json").unwrap();
    let merges_file = create_test_merges_file(temp_dir.path(), "merges.txt").unwrap();

    let tokenizer = create_bpe_tokenizer(&vocab_file, &merges_file).unwrap();
    assert_eq!(tokenizer.vocab_size(), 6); // 5 tokens + <pad>

    // Test encoding
    let tokens = encode_text(&tokenizer, "hello world").unwrap();
    assert_eq!(tokens, vec![0, 1]);

    // Test decoding
    let text = decode_tokens(&tokenizer, &[0, 1]).unwrap();
    assert_eq!(text, "hello world");

    // Test unknown token handling
    let tokens = encode_text(&tokenizer, "hello unknown").unwrap();
    assert_eq!(tokens, vec![0, 4]); // "hello" + "<unk>"
}

#[test]
fn test_create_bpe_tokenizer_missing_vocab_file() {
    let temp_dir = create_test_dir();
    let merges_file = create_test_merges_file(temp_dir.path(), "merges.txt").unwrap();

    let result = create_bpe_tokenizer("/nonexistent/vocab.json", &merges_file);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Vocabulary file not found"));
}

#[test]
fn test_create_bpe_tokenizer_missing_merges_file() {
    let temp_dir = create_test_dir();
    let vocab_file = create_test_vocab_file(temp_dir.path(), "vocab.json").unwrap();

    let result = create_bpe_tokenizer(&vocab_file, "/nonexistent/merges.txt");
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Merges file not found"));
}

#[test]
fn test_create_bpe_tokenizer_invalid_vocab_json() {
    let temp_dir = create_test_dir();
    let invalid_vocab = create_invalid_json_file(temp_dir.path(), "invalid_vocab.json").unwrap();
    let merges_file = create_test_merges_file(temp_dir.path(), "merges.txt").unwrap();

    let result = create_bpe_tokenizer(&invalid_vocab, &merges_file);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Failed to parse vocabulary file"));
}

#[test]
fn test_tokenizer_special_tokens() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("<unk>".to_string(), 2);

    let mut tokenizer = create_simple_tokenizer(vocab);

    // Initially no special tokens
    assert_eq!(get_special_token_id(&tokenizer, "[CLS]"), None);
    assert_eq!(get_special_token_id(&tokenizer, "[SEP]"), None);

    // Add special tokens
    let special_tokens = vec![
        ("[CLS]", 100),
        ("[SEP]", 101),
        ("[PAD]", 102),
        ("[MASK]", 103),
    ];

    add_special_tokens(&mut tokenizer, &special_tokens);

    // Test retrieval
    assert_eq!(get_special_token_id(&tokenizer, "[CLS]"), Some(100));
    assert_eq!(get_special_token_id(&tokenizer, "[SEP]"), Some(101));
    assert_eq!(get_special_token_id(&tokenizer, "[PAD]"), Some(102));
    assert_eq!(get_special_token_id(&tokenizer, "[MASK]"), Some(103));

    // Test non-existent special token
    assert_eq!(get_special_token_id(&tokenizer, "[UNKNOWN]"), None);
}

#[test]
fn test_tokenizer_special_tokens_instance_methods() {
    let mut vocab = HashMap::new();
    vocab.insert("test".to_string(), 0);

    let mut tokenizer = create_simple_tokenizer(vocab);

    // Test instance methods
    tokenizer.add_special_tokens(&[("[START]", 200), ("[END]", 201)]);

    assert_eq!(tokenizer.get_special_token_id("[START]"), Some(200));
    assert_eq!(tokenizer.get_special_token_id("[END]"), Some(201));
    assert_eq!(tokenizer.get_special_token_id("[MISSING]"), None);
}

#[test]
fn test_tokenizer_special_tokens_overwrite() {
    let mut vocab = HashMap::new();
    vocab.insert("test".to_string(), 0);

    let mut tokenizer = create_simple_tokenizer(vocab);

    // Add a special token
    add_special_tokens(&mut tokenizer, &[("[TOKEN]", 100)]);
    assert_eq!(get_special_token_id(&tokenizer, "[TOKEN]"), Some(100));

    // Overwrite with different ID
    add_special_tokens(&mut tokenizer, &[("[TOKEN]", 200)]);
    assert_eq!(get_special_token_id(&tokenizer, "[TOKEN]"), Some(200));
}

#[test]
fn test_encode_batch_functionality() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("test".to_string(), 2);
    vocab.insert("batch".to_string(), 3);
    vocab.insert("<unk>".to_string(), 4);

    let tokenizer = create_simple_tokenizer(vocab);

    let texts = vec!["hello world", "test batch", "hello test", "world batch"];

    let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();

    assert_eq!(batch_tokens.len(), 4);
    assert_eq!(batch_tokens[0], vec![0, 1]); // "hello world"
    assert_eq!(batch_tokens[1], vec![2, 3]); // "test batch"
    assert_eq!(batch_tokens[2], vec![0, 2]); // "hello test"
    assert_eq!(batch_tokens[3], vec![1, 3]); // "world batch"
}

#[test]
fn test_encode_batch_with_unknown_tokens() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("<unk>".to_string(), 2);

    let tokenizer = create_simple_tokenizer(vocab);

    let texts = vec![
        "hello world",
        "hello unknown",
        "unknown world",
        "unknown unknown",
    ];

    let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();

    assert_eq!(batch_tokens.len(), 4);
    assert_eq!(batch_tokens[0], vec![0, 1]); // "hello world"
    assert_eq!(batch_tokens[1], vec![0, 2]); // "hello <unk>"
    assert_eq!(batch_tokens[2], vec![2, 1]); // "<unk> world"
    assert_eq!(batch_tokens[3], vec![2, 2]); // "<unk> <unk>"
}

#[test]
fn test_encode_batch_empty_input() {
    let mut vocab = HashMap::new();
    vocab.insert("test".to_string(), 0);

    let tokenizer = create_simple_tokenizer(vocab);

    // Test empty batch
    let texts: Vec<&str> = vec![];
    let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
    assert_eq!(batch_tokens.len(), 0);

    // Test batch with empty string
    let texts = vec![""];
    let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
    assert_eq!(batch_tokens.len(), 1);
    assert_eq!(batch_tokens[0], Vec::<u32>::new());
}

#[test]
fn test_encode_batch_error_propagation() {
    let vocab = HashMap::new(); // Empty vocab
    let tokenizer = create_simple_tokenizer(vocab);

    let texts = vec!["hello world"];
    let result = encode_batch(&tokenizer, &texts);

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Failed to encode text"));
}

#[test]
fn test_tokenizer_round_trip_encoding_decoding() {
    let mut vocab = HashMap::new();
    vocab.insert("the".to_string(), 0);
    vocab.insert("quick".to_string(), 1);
    vocab.insert("brown".to_string(), 2);
    vocab.insert("fox".to_string(), 3);
    vocab.insert("jumps".to_string(), 4);
    vocab.insert("over".to_string(), 5);
    vocab.insert("lazy".to_string(), 6);
    vocab.insert("dog".to_string(), 7);

    let tokenizer = create_simple_tokenizer(vocab);

    let original_text = "the quick brown fox jumps over lazy dog";

    // Encode
    let tokens = encode_text(&tokenizer, original_text).unwrap();
    assert_eq!(tokens, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // Decode
    let decoded_text = decode_tokens(&tokenizer, &tokens).unwrap();
    assert_eq!(decoded_text, original_text);
}

#[test]
fn test_tokenizer_decode_invalid_token_id() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);

    let tokenizer = create_simple_tokenizer(vocab);

    // Try to decode a token ID that doesn't exist
    let result = decode_tokens(&tokenizer, &[0, 1, 999]);
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unknown token ID"));
}

#[test]
fn test_tokenizer_encode_without_unk_token() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    // No <unk> token

    let tokenizer = create_simple_tokenizer(vocab);

    // Try to encode text with unknown word
    let result = encode_text(&tokenizer, "hello unknown");
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unknown token") && error_msg.contains("no <unk> token found"));
}

// Tests for HuggingFace tokenizer loading (conditional on tokenizers feature)
#[cfg(feature = "tokenizers")]
mod hf_tokenizer_tests {
    use super::*;

    #[test]
    fn test_load_hf_tokenizer_nonexistent_file() {
        let result = load_hf_tokenizer("/nonexistent/path/tokenizer.json");
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to load HuggingFace tokenizer"));
    }

    #[test]
    fn test_load_hf_tokenizer_invalid_file() {
        let temp_dir = create_test_dir();
        let invalid_file = create_invalid_json_file(temp_dir.path(), "tokenizer.json").unwrap();

        let result = load_hf_tokenizer(&invalid_file);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to load HuggingFace tokenizer"));
    }

    #[test]
    fn test_load_hf_tokenizer_path_handling() {
        // Test with .json extension
        let result = load_hf_tokenizer("/path/to/tokenizer.json");
        assert!(result.is_err()); // Will fail because file doesn't exist, but path handling should work

        // Test without .json extension (should append /tokenizer.json)
        let result = load_hf_tokenizer("/path/to/model");
        assert!(result.is_err()); // Will fail because file doesn't exist, but path handling should work
    }
}

// Tests for when tokenizers feature is disabled
#[cfg(not(feature = "tokenizers"))]
mod no_tokenizers_feature_tests {
    use super::*;

    #[test]
    fn test_load_hf_tokenizer_feature_disabled() {
        let result = load_hf_tokenizer("any/path");
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("HuggingFace tokenizer support not enabled"));
        assert!(error_msg.contains("Enable the 'tokenizers' feature"));
    }
}

#[test]
fn test_tokenizer_vocab_size_consistency() {
    let mut vocab = HashMap::new();
    vocab.insert("token1".to_string(), 0);
    vocab.insert("token2".to_string(), 1);
    vocab.insert("token3".to_string(), 2);

    let tokenizer = create_simple_tokenizer(vocab.clone());

    // Vocab size should match the input vocabulary size
    assert_eq!(tokenizer.vocab_size(), vocab.len());
    assert_eq!(tokenizer.vocab_size(), 3);
}

#[test]
fn test_tokenizer_large_vocab() {
    let mut vocab = HashMap::new();

    // Create a large vocabulary
    for i in 0..10000 {
        vocab.insert(format!("token_{i}"), i as u32);
    }

    let tokenizer = create_simple_tokenizer(vocab.clone());
    assert_eq!(tokenizer.vocab_size(), 10000);

    // Test encoding/decoding with large vocab
    let tokens = encode_text(&tokenizer, "token_0 token_9999").unwrap();
    assert_eq!(tokens, vec![0, 9999]);

    let text = decode_tokens(&tokenizer, &[0, 9999]).unwrap();
    assert_eq!(text, "token_0 token_9999");
}

#[test]
fn test_tokenizer_unicode_handling() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("世界".to_string(), 1); // "world" in Chinese
    vocab.insert("🌍".to_string(), 2); // Earth emoji
    vocab.insert("<unk>".to_string(), 3);

    let tokenizer = create_simple_tokenizer(vocab);

    // Test encoding Unicode text
    let tokens = encode_text(&tokenizer, "hello 世界 🌍").unwrap();
    assert_eq!(tokens, vec![0, 1, 2]);

    // Test decoding Unicode tokens
    let text = decode_tokens(&tokenizer, &[0, 1, 2]).unwrap();
    assert_eq!(text, "hello 世界 🌍");
}

#[test]
fn test_tokenizer_whitespace_handling() {
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);

    let tokenizer = create_simple_tokenizer(vocab);

    // Test various whitespace scenarios
    let test_cases = vec![
        ("hello world", vec![0, 1]),
        ("hello  world", vec![0, 1]),  // Multiple spaces
        (" hello world ", vec![0, 1]), // Leading/trailing spaces
        ("hello\tworld", vec![0, 1]),  // Tab
        ("hello\nworld", vec![0, 1]),  // Newline
    ];

    for (text, expected_tokens) in test_cases {
        let tokens = encode_text(&tokenizer, text).unwrap();
        assert_eq!(tokens, expected_tokens, "Failed for text: '{text}'");
    }
}
