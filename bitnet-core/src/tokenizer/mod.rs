//! Tokenizer module for BitNet
//!
//! This module provides functionality for loading and creating various types of tokenizers
//! including HuggingFace tokenizers and BPE tokenizers. It also supports special token
//! handling for control tokens like [CLS], [SEP], [PAD], [MASK], etc.
//!
//! ## Special Token Support
//!
//! The tokenizer supports adding and retrieving special tokens:
//! - [`add_special_tokens`] - Add special tokens to a tokenizer
//! - [`get_special_token_id`] - Retrieve the ID of a special token
//! - [`Tokenizer::add_special_tokens`] - Instance method to add special tokens
//! - [`Tokenizer::get_special_token_id`] - Instance method to get special token IDs
//!
//! ## Example Usage
//!
//! ```rust
//! use bitnet_core::tokenizer::{create_simple_tokenizer, add_special_tokens, get_special_token_id};
//! use std::collections::HashMap;
//!
//! // Create a simple tokenizer
//! let mut vocab = HashMap::new();
//! vocab.insert("hello".to_string(), 0);
//! vocab.insert("world".to_string(), 1);
//! let mut tokenizer = create_simple_tokenizer(vocab);
//!
//! // Add special tokens
//! let special_tokens = vec![
//!     ("[CLS]", 100),
//!     ("[SEP]", 101),
//!     ("[PAD]", 102),
//! ];
//! add_special_tokens(&mut tokenizer, &special_tokens);
//!
//! // Retrieve special token IDs
//! assert_eq!(get_special_token_id(&tokenizer, "[CLS]"), Some(100));
//! assert_eq!(get_special_token_id(&tokenizer, "[SEP]"), Some(101));
//! ```

use std::path::Path;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer as HfTokenizer;

/// A unified tokenizer interface for BitNet
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: TokenizerType,
    vocab_size: usize,
    special_tokens: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Clone)]
enum TokenizerType {
    #[cfg(feature = "tokenizers")]
    HuggingFace(HfTokenizer),
    Bpe(BpeTokenizer),
    Simple(SimpleTokenizer),
}

/// A simple BPE tokenizer implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    vocab: std::collections::HashMap<String, u32>,
    merges: Vec<(String, String)>,
    vocab_size: usize,
}

/// A simple word-based tokenizer for basic use cases
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: std::collections::HashMap<String, u32>,
    vocab_size: usize,
}

impl Tokenizer {
    /// Get the vocabulary size of the tokenizer
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match &self.inner {
            #[cfg(feature = "tokenizers")]
            TokenizerType::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("HuggingFace tokenizer encoding failed: {}", e))?;
                Ok(encoding.get_ids().to_vec())
            }
            TokenizerType::Bpe(tokenizer) => tokenizer.encode(text),
            TokenizerType::Simple(tokenizer) => tokenizer.encode(text),
        }
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match &self.inner {
            #[cfg(feature = "tokenizers")]
            TokenizerType::HuggingFace(tokenizer) => {
                tokenizer
                    .decode(ids, false)
                    .map_err(|e| anyhow::anyhow!("HuggingFace tokenizer decoding failed: {}", e))
            }
            TokenizerType::Bpe(tokenizer) => tokenizer.decode(ids),
            TokenizerType::Simple(tokenizer) => tokenizer.decode(ids),
        }
    }

    /// Get a special token ID by its string representation
    pub fn get_special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// Add special tokens to the tokenizer
    pub fn add_special_tokens(&mut self, tokens: &[(&str, u32)]) {
        for (token, id) in tokens {
            self.special_tokens.insert(token.to_string(), *id);
        }
    }
}

/// Add special tokens to the tokenizer
///
/// This function adds special tokens to the tokenizer's special token mapping.
/// Special tokens are typically used for control tokens like [CLS], [SEP], [PAD], etc.
pub fn add_special_tokens(tokenizer: &mut Tokenizer, tokens: &[(&str, u32)]) {
    tokenizer.add_special_tokens(tokens);
}

/// Get a special token ID by its string representation
///
/// This function retrieves the token ID for a special token string.
/// Returns None if the special token is not found.
pub fn get_special_token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.get_special_token_id(token)
}

/// Encode text into token IDs using the provided tokenizer
///
/// This is a convenience function that wraps the tokenizer's encode method
/// and provides a consistent interface for text encoding.
pub fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode(text)
}

/// Decode token IDs back to text using the provided tokenizer
///
/// This is a convenience function that wraps the tokenizer's decode method
/// and provides a consistent interface for token decoding.
pub fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer.decode(tokens)
}

/// Encode multiple texts into batches of token IDs
///
/// This function processes multiple text inputs in parallel and returns
/// a vector of token ID vectors, one for each input text.
pub fn encode_batch(tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
    let mut results = Vec::with_capacity(texts.len());
    
    for text in texts {
        let tokens = tokenizer.encode(text)
            .with_context(|| format!("Failed to encode text: '{}'", text))?;
        results.push(tokens);
    }
    
    Ok(results)
}

/// Create a simple tokenizer from a vocabulary map
///
/// This function creates a simple word-based tokenizer from a provided vocabulary.
/// It's useful for testing and simple use cases where you want to create a tokenizer
/// programmatically without loading from files.
pub fn create_simple_tokenizer(vocab: std::collections::HashMap<String, u32>) -> Tokenizer {
    let vocab_size = vocab.len();
    let simple_tokenizer = SimpleTokenizer::new(vocab);
    
    Tokenizer {
        inner: TokenizerType::Simple(simple_tokenizer),
        vocab_size,
        special_tokens: std::collections::HashMap::new(),
    }
}

/// Load a tokenizer from a file path
/// 
/// This function attempts to load a tokenizer from the specified path.
/// It supports various tokenizer formats including HuggingFace tokenizers.
pub fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(anyhow::anyhow!("Tokenizer file not found: {}", path.display()));
    }

    // Try to load as HuggingFace tokenizer first
    #[cfg(feature = "tokenizers")]
    {
        if let Ok(hf_tokenizer) = HfTokenizer::from_file(path) {
            let vocab_size = hf_tokenizer.get_vocab_size(false);
            return Ok(Tokenizer {
                inner: TokenizerType::HuggingFace(hf_tokenizer),
                vocab_size,
                special_tokens: std::collections::HashMap::new(),
            });
        }
    }

    // Try to load as BPE tokenizer
    if path.extension().and_then(|s| s.to_str()) == Some("json") {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read tokenizer file: {}", path.display()))?;
        
        if let Ok(bpe_tokenizer) = serde_json::from_str::<BpeTokenizer>(&content) {
            let vocab_size = bpe_tokenizer.vocab_size;
            return Ok(Tokenizer {
                inner: TokenizerType::Bpe(bpe_tokenizer),
                vocab_size,
                special_tokens: std::collections::HashMap::new(),
            });
        }
    }

    Err(anyhow::anyhow!(
        "Unable to load tokenizer from path: {}. Supported formats: HuggingFace tokenizer.json, BPE JSON",
        path.display()
    ))
}

/// Load a HuggingFace tokenizer from model ID
///
/// This function loads a tokenizer from a local path that contains HuggingFace tokenizer files.
/// For downloading from HuggingFace Hub, you would need to use external tools like `hf_hub` crate.
/// The model_id parameter should be a path to a local tokenizer.json file.
pub fn load_hf_tokenizer(model_id: &str) -> Result<Tokenizer> {
    #[cfg(feature = "tokenizers")]
    {
        // For now, treat model_id as a file path to tokenizer.json
        // In a full implementation, you might want to integrate with hf_hub crate
        // to download tokenizers from HuggingFace Hub
        let tokenizer_path = if model_id.ends_with(".json") {
            model_id.to_string()
        } else {
            format!("{}/tokenizer.json", model_id)
        };
        
        let hf_tokenizer = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer from '{}': {}", tokenizer_path, e))?;
        
        let vocab_size = hf_tokenizer.get_vocab_size(false);
        
        Ok(Tokenizer {
            inner: TokenizerType::HuggingFace(hf_tokenizer),
            vocab_size,
            special_tokens: std::collections::HashMap::new(),
        })
    }
    
    #[cfg(not(feature = "tokenizers"))]
    {
        Err(anyhow::anyhow!(
            "HuggingFace tokenizer support not enabled. Enable the 'tokenizers' feature to use this function."
        ))
    }
}

/// Create a BPE tokenizer from vocabulary and merges files
/// 
/// This function creates a BPE (Byte Pair Encoding) tokenizer from separate
/// vocabulary and merges files, commonly used in GPT-style models.
pub fn create_bpe_tokenizer(vocab_file: &str, merges_file: &str) -> Result<Tokenizer> {
    let vocab_path = Path::new(vocab_file);
    let merges_path = Path::new(merges_file);

    if !vocab_path.exists() {
        return Err(anyhow::anyhow!("Vocabulary file not found: {}", vocab_path.display()));
    }
    
    if !merges_path.exists() {
        return Err(anyhow::anyhow!("Merges file not found: {}", merges_path.display()));
    }

    // Load vocabulary
    let vocab_content = std::fs::read_to_string(vocab_path)
        .with_context(|| format!("Failed to read vocabulary file: {}", vocab_path.display()))?;
    
    let vocab: std::collections::HashMap<String, u32> = serde_json::from_str(&vocab_content)
        .with_context(|| format!("Failed to parse vocabulary file: {}", vocab_path.display()))?;

    // Load merges
    let merges_content = std::fs::read_to_string(merges_path)
        .with_context(|| format!("Failed to read merges file: {}", merges_path.display()))?;
    
    let mut merges = Vec::new();
    for line in merges_content.lines().skip(1) { // Skip header line
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }
    }

    let vocab_size = vocab.len();
    let bpe_tokenizer = BpeTokenizer {
        vocab,
        merges,
        vocab_size,
    };

    Ok(Tokenizer {
        inner: TokenizerType::Bpe(bpe_tokenizer),
        vocab_size,
        special_tokens: std::collections::HashMap::new(),
    })
}

impl BpeTokenizer {
    /// Encode text using BPE algorithm
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Simple word-level tokenization for now
        // In a full implementation, this would include proper BPE encoding
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();
        
        for word in words {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // Handle unknown tokens - could implement subword splitting here
                if let Some(&unk_id) = self.vocab.get("<unk>") {
                    tokens.push(unk_id);
                } else {
                    return Err(anyhow::anyhow!("Unknown token '{}' and no <unk> token found", word));
                }
            }
        }
        
        Ok(tokens)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut words = Vec::new();
        let reverse_vocab: std::collections::HashMap<u32, String> = 
            self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        
        for &id in ids {
            if let Some(word) = reverse_vocab.get(&id) {
                words.push(word.clone());
            } else {
                return Err(anyhow::anyhow!("Unknown token ID: {}", id));
            }
        }
        
        Ok(words.join(" "))
    }
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer from a vocabulary
    pub fn new(vocab: std::collections::HashMap<String, u32>) -> Self {
        let vocab_size = vocab.len();
        Self { vocab, vocab_size }
    }

    /// Encode text using simple word-level tokenization
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();
        
        for word in words {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                if let Some(&unk_id) = self.vocab.get("<unk>") {
                    tokens.push(unk_id);
                } else {
                    return Err(anyhow::anyhow!("Unknown token '{}' and no <unk> token found", word));
                }
            }
        }
        
        Ok(tokens)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut words = Vec::new();
        let reverse_vocab: std::collections::HashMap<u32, String> = 
            self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        
        for &id in ids {
            if let Some(word) = reverse_vocab.get(&id) {
                words.push(word.clone());
            } else {
                return Err(anyhow::anyhow!("Unknown token ID: {}", id));
            }
        }
        
        Ok(words.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_simple_tokenizer() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let tokenizer = SimpleTokenizer::new(vocab);
        
        // Test encoding
        let tokens = tokenizer.encode("hello world").unwrap();
        assert_eq!(tokens, vec![0, 1]);
        
        // Test decoding
        let text = tokenizer.decode(&[0, 1]).unwrap();
        assert_eq!(text, "hello world");
        
        // Test unknown token
        let tokens = tokenizer.encode("hello unknown").unwrap();
        assert_eq!(tokens, vec![0, 2]);
    }

    #[test]
    fn test_bpe_tokenizer() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let bpe_tokenizer = BpeTokenizer {
            vocab,
            merges: vec![],
            vocab_size: 3,
        };
        
        // Test encoding
        let tokens = bpe_tokenizer.encode("hello world").unwrap();
        assert_eq!(tokens, vec![0, 1]);
        
        // Test decoding
        let text = bpe_tokenizer.decode(&[0, 1]).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_encode_text_function() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 3,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test the encode_text function
        let tokens = encode_text(&tokenizer, "hello world").unwrap();
        assert_eq!(tokens, vec![0, 1]);
        
        // Test with unknown token
        let tokens = encode_text(&tokenizer, "hello unknown").unwrap();
        assert_eq!(tokens, vec![0, 2]);
    }

    #[test]
    fn test_decode_tokens_function() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 3,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test the decode_tokens function
        let text = decode_tokens(&tokenizer, &[0, 1]).unwrap();
        assert_eq!(text, "hello world");
        
        // Test with unknown token
        let text = decode_tokens(&tokenizer, &[0, 2]).unwrap();
        assert_eq!(text, "hello <unk>");
    }

    #[test]
    fn test_encode_batch_function() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("foo".to_string(), 3);
        vocab.insert("bar".to_string(), 4);
        vocab.insert("<unk>".to_string(), 2);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 5,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test the encode_batch function
        let texts = vec!["hello world", "foo bar", "hello foo"];
        let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
        
        assert_eq!(batch_tokens.len(), 3);
        assert_eq!(batch_tokens[0], vec![0, 1]); // "hello world"
        assert_eq!(batch_tokens[1], vec![3, 4]); // "foo bar"
        assert_eq!(batch_tokens[2], vec![0, 3]); // "hello foo"
    }

    #[test]
    fn test_encode_batch_with_unknown_tokens() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 3,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test with some unknown tokens
        let texts = vec!["hello world", "hello unknown", "unknown world"];
        let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
        
        assert_eq!(batch_tokens.len(), 3);
        assert_eq!(batch_tokens[0], vec![0, 1]); // "hello world"
        assert_eq!(batch_tokens[1], vec![0, 2]); // "hello <unk>"
        assert_eq!(batch_tokens[2], vec![2, 1]); // "<unk> world"
    }

    #[test]
    fn test_encode_batch_empty_input() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 2,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test with empty batch
        let texts: Vec<&str> = vec![];
        let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
        assert_eq!(batch_tokens.len(), 0);
        
        // Test with empty string
        let texts = vec![""];
        let batch_tokens = encode_batch(&tokenizer, &texts).unwrap();
        assert_eq!(batch_tokens.len(), 1);
        assert_eq!(batch_tokens[0], Vec::<u32>::new()); // Empty string should produce empty token vector
    }

    #[test]
    fn test_special_tokens() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let mut tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 3,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test adding special tokens
        let special_tokens = vec![
            ("[CLS]", 100),
            ("[SEP]", 101),
            ("[PAD]", 102),
            ("[MASK]", 103),
        ];
        
        tokenizer.add_special_tokens(&special_tokens);
        
        // Test retrieving special token IDs
        assert_eq!(tokenizer.get_special_token_id("[CLS]"), Some(100));
        assert_eq!(tokenizer.get_special_token_id("[SEP]"), Some(101));
        assert_eq!(tokenizer.get_special_token_id("[PAD]"), Some(102));
        assert_eq!(tokenizer.get_special_token_id("[MASK]"), Some(103));
        
        // Test non-existent special token
        assert_eq!(tokenizer.get_special_token_id("[UNK]"), None);
        
        // Test adding more special tokens
        tokenizer.add_special_tokens(&[("[BOS]", 104), ("[EOS]", 105)]);
        assert_eq!(tokenizer.get_special_token_id("[BOS]"), Some(104));
        assert_eq!(tokenizer.get_special_token_id("[EOS]"), Some(105));
    }

    #[test]
    fn test_special_tokens_standalone_functions() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let mut tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 2,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test standalone add_special_tokens function
        let special_tokens = vec![
            ("[START]", 200),
            ("[END]", 201),
        ];
        
        add_special_tokens(&mut tokenizer, &special_tokens);
        
        // Test standalone get_special_token_id function
        assert_eq!(get_special_token_id(&tokenizer, "[START]"), Some(200));
        assert_eq!(get_special_token_id(&tokenizer, "[END]"), Some(201));
        assert_eq!(get_special_token_id(&tokenizer, "[MISSING]"), None);
    }

    #[test]
    fn test_special_tokens_overwrite() {
        let mut vocab = HashMap::new();
        vocab.insert("test".to_string(), 0);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let mut tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 1,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Add a special token
        tokenizer.add_special_tokens(&[("[TOKEN]", 100)]);
        assert_eq!(tokenizer.get_special_token_id("[TOKEN]"), Some(100));
        
        // Overwrite the same special token with a different ID
        tokenizer.add_special_tokens(&[("[TOKEN]", 200)]);
        assert_eq!(tokenizer.get_special_token_id("[TOKEN]"), Some(200));
    }

    #[test]
    fn test_special_tokens_empty() {
        let mut vocab = HashMap::new();
        vocab.insert("test".to_string(), 0);
        
        let simple_tokenizer = SimpleTokenizer::new(vocab);
        let mut tokenizer = Tokenizer {
            inner: TokenizerType::Simple(simple_tokenizer),
            vocab_size: 1,
            special_tokens: std::collections::HashMap::new(),
        };
        
        // Test with empty special tokens array
        tokenizer.add_special_tokens(&[]);
        assert_eq!(tokenizer.get_special_token_id("[ANYTHING]"), None);
        
        // Test with empty string token
        tokenizer.add_special_tokens(&[("", 999)]);
        assert_eq!(tokenizer.get_special_token_id(""), Some(999));
    }
}