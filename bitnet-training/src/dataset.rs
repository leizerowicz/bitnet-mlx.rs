use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use crate::tokenization::TrainingTokenizer;

/// Training dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub max_seq_length: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub pad_to_max_length: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_seq_length: 512,
            batch_size: 8,
            shuffle: true,
            pad_to_max_length: true,
        }
    }
}

/// A single training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub text: String,
    pub category: Option<String>,
}

/// Training batch for model input
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub input_ids: Vec<u32>,
    pub labels: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Training dataset for BitNet custom models
pub struct TrainingDataset {
    examples: Vec<TrainingExample>,
    config: DatasetConfig,
    tokenizer: TrainingTokenizer,
    current_batch: usize,
}

impl TrainingDataset {
    /// Create a new training dataset
    pub fn new(
        examples: Vec<TrainingExample>,
        config: DatasetConfig,
        tokenizer: TrainingTokenizer,
    ) -> Self {
        Self {
            examples,
            config,
            tokenizer,
            current_batch: 0,
        }
    }
    
    /// Create a simple training dataset from strings (for demo purposes)
    pub fn new_simple(texts: Vec<String>, config: DatasetConfig) -> Result<Self> {
        use crate::tokenization::TokenizerConfig;
        
        let examples: Vec<TrainingExample> = texts.into_iter()
            .map(|text| TrainingExample { text, category: None })
            .collect();
            
        let tokenizer_config = TokenizerConfig::default();
        let tokenizer = TrainingTokenizer::new(tokenizer_config)?;
        
        Ok(Self::new(examples, config, tokenizer))
    }
    
    /// Load dataset from JSON file
    pub fn from_json_file<P: AsRef<Path>>(
        path: P,
        config: DatasetConfig,
        tokenizer: TrainingTokenizer,
    ) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let examples: Vec<TrainingExample> = serde_json::from_str(&content)?;
        
        if examples.is_empty() {
            return Err(anyhow!("Dataset is empty"));
        }
        
        Ok(Self::new(examples, config, tokenizer))
    }
    
    /// Load dataset from text file (one example per line)
    pub fn from_text_file<P: AsRef<Path>>(
        path: P,
        config: DatasetConfig,
        tokenizer: TrainingTokenizer,
    ) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let examples: Vec<TrainingExample> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| TrainingExample {
                text: line.trim().to_string(),
                category: None,
            })
            .collect();
        
        if examples.is_empty() {
            return Err(anyhow!("Dataset is empty"));
        }
        
        Ok(Self::new(examples, config, tokenizer))
    }
    
    /// Load dataset from file (automatically detects format)
    pub fn from_file<P: AsRef<Path>>(
        path: P, 
        config: DatasetConfig
    ) -> Result<Self> {
        use crate::tokenization::TokenizerConfig;
        
        let tokenizer_config = TokenizerConfig::default();
        let path = path.as_ref();
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        
        match extension {
            "json" => {
                let tokenizer = TrainingTokenizer::new(tokenizer_config)?;
                Self::from_json_file(path, config, tokenizer)
            },
            "txt" | "text" => {
                let tokenizer = TrainingTokenizer::new(tokenizer_config)?;
                Self::from_text_file(path, config, tokenizer)
            },
            _ => {
                // Try to parse as JSON first, then as text
                let tokenizer1 = TrainingTokenizer::new(tokenizer_config.clone())?;
                match Self::from_json_file(path, config.clone(), tokenizer1) {
                    Ok(dataset) => Ok(dataset),
                    Err(_) => {
                        let tokenizer2 = TrainingTokenizer::new(tokenizer_config)?;
                        Self::from_text_file(path, config, tokenizer2)
                    }
                }
            }
        }
    }
    
    /// Create synthetic dataset for testing
    pub fn create_synthetic(
        config: DatasetConfig,
        tokenizer: TrainingTokenizer,
    ) -> Result<Self> {
        let synthetic_examples = vec![
            TrainingExample {
                text: "The quick brown fox jumps over the lazy dog.".to_string(),
                category: Some("sample".to_string()),
            },
            TrainingExample {
                text: "Machine learning is transforming artificial intelligence.".to_string(),
                category: Some("ml".to_string()),
            },
            TrainingExample {
                text: "BitNet models use 1.58-bit quantization for efficiency.".to_string(),
                category: Some("bitnet".to_string()),
            },
            TrainingExample {
                text: "Rust programming language provides memory safety.".to_string(),
                category: Some("rust".to_string()),
            },
            TrainingExample {
                text: "Neural networks require extensive training data.".to_string(),
                category: Some("ml".to_string()),
            },
            TrainingExample {
                text: "Transformers are the foundation of modern language models.".to_string(),
                category: Some("transformers".to_string()),
            },
            TrainingExample {
                text: "Quantization reduces model size while maintaining performance.".to_string(),
                category: Some("quantization".to_string()),
            },
            TrainingExample {
                text: "Apple Silicon provides excellent acceleration for ML workloads.".to_string(),
                category: Some("hardware".to_string()),
            },
        ];
        
        Ok(Self::new(synthetic_examples, config, tokenizer))
    }
    
    /// Get total number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
    
    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.len() + self.config.batch_size - 1) / self.config.batch_size
    }
    
    /// Get configuration
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }
    
    /// Get an iterator over batches
    pub fn iter_batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator::new(self, batch_size)
    }
    
    /// Prepare a batch from examples
    fn prepare_batch(&self, examples: &[TrainingExample]) -> Result<TrainingBatch> {
        let mut all_input_ids = Vec::new();
        let mut all_labels = Vec::new();
        let mut all_attention_mask = Vec::new();
        
        let mut max_length = 0;
        
        // Tokenize all examples and find max length
        let mut tokenized_examples = Vec::new();
        for example in examples {
            let input_ids = self.tokenizer.encode(&example.text, true)?;
            let labels = input_ids.clone(); // For language modeling, labels = input_ids shifted
            
            max_length = max_length.max(input_ids.len());
            tokenized_examples.push((input_ids, labels));
        }
        
        // Apply max sequence length limit
        max_length = max_length.min(self.config.max_seq_length);
        
        // Pad or truncate sequences
        for (mut input_ids, mut labels) in tokenized_examples {
            // Truncate if too long
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                labels.truncate(max_length);
            }
            
            // Create attention mask (1 for real tokens, 0 for padding)
            let mut attention_mask = vec![1u32; input_ids.len()];
            
            // Pad if too short
            if self.config.pad_to_max_length && input_ids.len() < max_length {
                let pad_length = max_length - input_ids.len();
                input_ids.extend(vec![self.tokenizer.pad_token_id(); pad_length]);
                labels.extend(vec![self.tokenizer.pad_token_id(); pad_length]);
                attention_mask.extend(vec![0u32; pad_length]);
            }
            
            all_input_ids.extend(input_ids);
            all_labels.extend(labels);
            all_attention_mask.extend(attention_mask);
        }
        
        Ok(TrainingBatch {
            input_ids: all_input_ids,
            labels: all_labels,
            attention_mask: all_attention_mask,
            batch_size: examples.len(),
            seq_len: max_length,
        })
    }
}

/// Iterator over training batches
pub struct BatchIterator<'a> {
    dataset: &'a TrainingDataset,
    batch_size: usize,
    current_index: usize,
}

impl<'a> BatchIterator<'a> {
    fn new(dataset: &'a TrainingDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_index: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Result<TrainingBatch>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.dataset.len() {
            return None;
        }
        
        let end_index = (self.current_index + self.batch_size).min(self.dataset.len());
        let batch_examples = &self.dataset.examples[self.current_index..end_index];
        
        self.current_index = end_index;
        
        Some(self.dataset.prepare_batch(batch_examples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenization::{TrainingTokenizer, TokenizerConfig};

    fn create_test_dataset() -> TrainingDataset {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let config = DatasetConfig::default();
        
        TrainingDataset::create_synthetic(config, tokenizer).unwrap()
    }

    #[test]
    fn test_dataset_creation() {
        let dataset = create_test_dataset();
        assert!(!dataset.is_empty());
        assert!(dataset.len() > 0);
    }

    #[test]
    fn test_batch_iteration() {
        let dataset = create_test_dataset();
        let batch_size = 2;
        
        let mut batch_count = 0;
        for batch_result in dataset.iter_batches(batch_size) {
            let batch = batch_result.unwrap();
            assert!(batch.batch_size <= batch_size);
            assert!(batch.input_ids.len() > 0);
            assert_eq!(batch.input_ids.len(), batch.labels.len());
            assert_eq!(batch.input_ids.len(), batch.attention_mask.len());
            batch_count += 1;
        }
        
        assert!(batch_count > 0);
    }

    #[test]
    fn test_synthetic_dataset() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let config = DatasetConfig::default();
        
        let dataset = TrainingDataset::create_synthetic(config, tokenizer).unwrap();
        assert_eq!(dataset.len(), 8); // We created 8 synthetic examples
        
        // Test first batch
        let mut iter = dataset.iter_batches(4);
        let first_batch = iter.next().unwrap().unwrap();
        assert_eq!(first_batch.batch_size, 4);
    }

    #[test]
    fn test_batch_padding() {
        let tokenizer = TrainingTokenizer::new(TokenizerConfig::default()).unwrap();
        let mut config = DatasetConfig::default();
        config.max_seq_length = 20;
        config.pad_to_max_length = true;
        
        let dataset = TrainingDataset::create_synthetic(config, tokenizer).unwrap();
        
        let mut iter = dataset.iter_batches(2);
        let batch = iter.next().unwrap().unwrap();
        
        // Check that sequences are padded to the same length
        let expected_length = batch.seq_len * batch.batch_size;
        assert_eq!(batch.input_ids.len(), expected_length);
        assert_eq!(batch.labels.len(), expected_length);
        assert_eq!(batch.attention_mask.len(), expected_length);
    }
}