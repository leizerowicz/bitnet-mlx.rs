use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training tokenizer for BitNet custom models
/// 
/// This tokenizer provides a simplified tokenization approach for training
/// custom BitNet models. It uses a basic word-based tokenization strategy
/// that's suitable for proof-of-concept training and testing.
pub struct TrainingTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    vocab_size: usize,
    pad_token_id: u32,
    unk_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub pad_token: String,
    pub unk_token: String,
    pub bos_token: String,
    pub eos_token: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000, // Reduced vocab size for custom training
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        }
    }
}

impl TrainingTokenizer {
    /// Create a new training tokenizer with basic vocabulary
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Add special tokens
        let special_tokens = vec![
            &config.pad_token,
            &config.unk_token,
            &config.bos_token,
            &config.eos_token,
        ];
        
        for (id, token) in special_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
            reverse_vocab.insert(id as u32, token.to_string());
        }
        
        // Build basic vocabulary from common words
        let common_words = Self::get_common_vocabulary();
        let mut current_id = special_tokens.len() as u32;
        
        for word in common_words.iter().take(config.vocab_size - special_tokens.len()) {
            if !vocab.contains_key(word) {
                vocab.insert(word.clone(), current_id);
                reverse_vocab.insert(current_id, word.clone());
                current_id += 1;
            }
        }
        
        Ok(Self {
            vocab_size: config.vocab_size,
            pad_token_id: 0,
            unk_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 3,
            vocab,
            reverse_vocab,
        })
    }
    
    /// Create from pretrained (simplified for custom training)
    pub fn from_pretrained() -> Result<Self> {
        // For now, use the default configuration
        // In a real implementation, this would load a pretrained tokenizer
        Self::new(TokenizerConfig::default())
    }
    
    /// Encode a single text into token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_special_tokens {
            tokens.push(self.bos_token_id);
        }
        
        // Simple word-based tokenization
        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|w| !w.is_empty())
            .collect();
        
        for word in words {
            let token_id = self.vocab.get(word).copied().unwrap_or(self.unk_token_id);
            tokens.push(token_id);
        }
        
        if add_special_tokens {
            tokens.push(self.eos_token_id);
        }
        
        Ok(tokens)
    }
    
    /// Encode batch of texts
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| self.encode(text, true))
            .collect()
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut words = Vec::new();
        
        for &token_id in token_ids {
            if skip_special_tokens && self.is_special_token(token_id) {
                continue;
            }
            
            let word = self.reverse_vocab.get(&token_id)
                .ok_or_else(|| anyhow!("Unknown token ID: {}", token_id))?;
            words.push(word.clone());
        }
        
        Ok(words.join(" "))
    }
    
    /// Check if token ID is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        token_id == self.pad_token_id 
            || token_id == self.unk_token_id 
            || token_id == self.bos_token_id 
            || token_id == self.eos_token_id
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get special token IDs
    pub fn pad_token_id(&self) -> u32 { self.pad_token_id }
    pub fn unk_token_id(&self) -> u32 { self.unk_token_id }
    pub fn bos_token_id(&self) -> u32 { self.bos_token_id }
    pub fn eos_token_id(&self) -> u32 { self.eos_token_id }
    
    /// Get common vocabulary for basic training
    fn get_common_vocabulary() -> Vec<String> {
        vec![
            // Common English words
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
            "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            
            // Technical terms
            "model", "network", "neural", "machine", "learning", "training", "data", "algorithm",
            "optimization", "performance", "memory", "computation", "inference", "prediction",
            "classification", "regression", "supervised", "unsupervised", "reinforcement",
            "deep", "layer", "hidden", "input", "output", "weight", "bias", "gradient",
            "backpropagation", "forward", "pass", "epoch", "batch", "sample", "loss", "accuracy",
            
            // BitNet specific terms
            "bitnet", "quantization", "ternary", "binary", "precision", "bit", "compression",
            "efficient", "sparse", "dense", "matrix", "multiplication", "activation", "function",
            "transformer", "attention", "multi", "head", "self", "cross", "encoder", "decoder",
            "embedding", "position", "token", "vocabulary", "sequence", "context", "window",
            
            // Rust terms
            "rust", "cargo", "crate", "module", "function", "struct", "enum", "trait", "impl",
            "pub", "fn", "let", "mut", "ref", "move", "copy", "clone", "borrow", "ownership",
            "lifetime", "generic", "type", "result", "option", "error", "panic", "unsafe",
            "async", "await", "future", "stream", "iterator", "vector", "hashmap", "btreemap",
            
            // Common verbs and adjectives
            "create", "build", "make", "generate", "produce", "compute", "calculate", "process",
            "handle", "manage", "control", "optimize", "improve", "enhance", "reduce", "increase",
            "fast", "slow", "quick", "efficient", "effective", "accurate", "precise", "robust",
            "stable", "reliable", "scalable", "flexible", "simple", "complex", "advanced",
            "basic", "fundamental", "essential", "important", "critical", "key", "main", "primary",
            
            // Numbers and basic operations
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "last", "next", "previous", "add", "subtract", "multiply",
            "divide", "sum", "product", "average", "mean", "median", "mode", "standard", "deviation",
            "variance", "minimum", "maximum", "range", "value", "number", "count", "size", "length",
        ].into_iter().map(|s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 32000);
        assert_eq!(tokenizer.pad_token_id(), 0);
        assert_eq!(tokenizer.unk_token_id(), 1);
        assert_eq!(tokenizer.bos_token_id(), 2);
        assert_eq!(tokenizer.eos_token_id(), 3);
    }

    #[test]
    fn test_encoding_decoding() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let text = "The quick brown fox";
        
        let tokens = tokenizer.encode(text, true).unwrap();
        assert!(tokens.len() > 0);
        assert_eq!(tokens[0], tokenizer.bos_token_id());
        assert_eq!(tokens[tokens.len() - 1], tokenizer.eos_token_id());
        
        let decoded = tokenizer.decode(&tokens, true).unwrap();
        assert!(decoded.contains("quick"));
        assert!(decoded.contains("brown"));
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let texts = vec![
            "Machine learning is powerful".to_string(),
            "BitNet uses quantization".to_string(),
        ];
        
        let encoded_batch = tokenizer.encode_batch(&texts).unwrap();
        assert_eq!(encoded_batch.len(), 2);
        assert!(encoded_batch[0].len() > 0);
        assert!(encoded_batch[1].len() > 0);
    }

    #[test]
    fn test_unknown_tokens() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let text_with_unknown = "supercalifragilisticexpialidocious";
        
        let tokens = tokenizer.encode(text_with_unknown, false).unwrap();
        assert_eq!(tokens[0], tokenizer.unk_token_id());
    }
}