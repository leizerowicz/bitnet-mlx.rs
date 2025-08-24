//! Sequence Validation Utilities
//!
//! This module provides functionality for validating sequences against
//! length constraints and other requirements.

use super::{SequenceConfig, SequenceError, SequenceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Sequence exceeds maximum length
    TooLong { length: usize, max_length: usize },
    /// Sequence is below minimum length
    TooShort { length: usize, min_length: usize },
    /// Empty sequence when not allowed
    Empty,
    /// Invalid token found in sequence
    InvalidToken { token: u32, position: usize },
    /// Sequence contains only padding tokens
    OnlyPadding { pad_token: u32 },
    /// Custom validation error
    Custom { message: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::TooLong { length, max_length } => {
                write!(
                    f,
                    "Sequence length {} exceeds maximum {}",
                    length, max_length
                )
            }
            ValidationError::TooShort { length, min_length } => {
                write!(f, "Sequence length {} below minimum {}", length, min_length)
            }
            ValidationError::Empty => write!(f, "Empty sequence not allowed"),
            ValidationError::InvalidToken { token, position } => {
                write!(f, "Invalid token {} at position {}", token, position)
            }
            ValidationError::OnlyPadding { pad_token } => {
                write!(f, "Sequence contains only padding tokens ({})", pad_token)
            }
            ValidationError::Custom { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation rules for sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    /// Minimum allowed sequence length
    pub min_length: Option<usize>,
    /// Maximum allowed sequence length
    pub max_length: Option<usize>,
    /// Whether empty sequences are allowed
    pub allow_empty: bool,
    /// Set of allowed token IDs (None means all tokens allowed)
    pub allowed_tokens: Option<std::collections::HashSet<u32>>,
    /// Set of forbidden token IDs
    pub forbidden_tokens: std::collections::HashSet<u32>,
    /// Padding token ID (for validation)
    pub pad_token: Option<u32>,
    /// Whether sequences with only padding are allowed
    pub allow_only_padding: bool,
    /// Custom validation function name (for extensibility)
    pub custom_validators: Vec<String>,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            min_length: None,
            max_length: None,
            allow_empty: false,
            allowed_tokens: None,
            forbidden_tokens: std::collections::HashSet::new(),
            pad_token: Some(0),
            allow_only_padding: false,
            custom_validators: Vec::new(),
        }
    }
}

impl ValidationRules {
    /// Create new validation rules
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum length constraint
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = Some(min_length);
        self
    }

    /// Set maximum length constraint
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Allow empty sequences
    pub fn allow_empty(mut self) -> Self {
        self.allow_empty = true;
        self
    }

    /// Set allowed tokens (whitelist)
    pub fn with_allowed_tokens(mut self, tokens: std::collections::HashSet<u32>) -> Self {
        self.allowed_tokens = Some(tokens);
        self
    }

    /// Add forbidden tokens (blacklist)
    pub fn with_forbidden_tokens(mut self, tokens: std::collections::HashSet<u32>) -> Self {
        self.forbidden_tokens = tokens;
        self
    }

    /// Set padding token
    pub fn with_pad_token(mut self, pad_token: u32) -> Self {
        self.pad_token = Some(pad_token);
        self
    }

    /// Allow sequences with only padding tokens
    pub fn allow_only_padding(mut self) -> Self {
        self.allow_only_padding = true;
        self
    }
}

/// Sequence validator
#[derive(Debug, Clone)]
pub struct SequenceValidator {
    rules: ValidationRules,
    stats: ValidationStats,
}

impl SequenceValidator {
    /// Create a new validator from configuration
    pub fn new(config: &SequenceConfig) -> Self {
        let rules = ValidationRules {
            min_length: config.min_length,
            max_length: config.max_length,
            allow_empty: false,
            allowed_tokens: None,
            forbidden_tokens: std::collections::HashSet::new(),
            pad_token: config.pad_token_id,
            allow_only_padding: false,
            custom_validators: Vec::new(),
        };

        Self {
            rules,
            stats: ValidationStats::new(),
        }
    }

    /// Create a validator with custom rules
    pub fn with_rules(rules: ValidationRules) -> Self {
        Self {
            rules,
            stats: ValidationStats::new(),
        }
    }

    /// Validate a single sequence
    pub fn validate_sequence(&mut self, sequence: &[u32]) -> SequenceResult<()> {
        self.stats.total_validated += 1;

        // Check if empty
        if sequence.is_empty() {
            if !self.rules.allow_empty {
                self.stats.failed_validations += 1;
                return Err(SequenceError::EmptySequence);
            }
            return Ok(());
        }

        // Check length constraints
        let length = sequence.len();

        if let Some(min_len) = self.rules.min_length {
            if length < min_len {
                self.stats.failed_validations += 1;
                return Err(SequenceError::BelowMinLength {
                    length,
                    min_length: min_len,
                });
            }
        }

        if let Some(max_len) = self.rules.max_length {
            if length > max_len {
                self.stats.failed_validations += 1;
                return Err(SequenceError::ExceedsMaxLength {
                    length,
                    max_length: max_len,
                });
            }
        }

        // Check for only padding tokens
        if let Some(pad_token) = self.rules.pad_token {
            if !self.rules.allow_only_padding && sequence.iter().all(|&token| token == pad_token) {
                self.stats.failed_validations += 1;
                return Err(SequenceError::ConfigError {
                    message: format!("Sequence contains only padding tokens ({})", pad_token),
                });
            }
        }

        // Check token validity
        for (pos, &token) in sequence.iter().enumerate() {
            // Check forbidden tokens
            if self.rules.forbidden_tokens.contains(&token) {
                self.stats.failed_validations += 1;
                return Err(SequenceError::InvalidPaddingToken { token });
            }

            // Check allowed tokens (if whitelist is specified)
            if let Some(ref allowed) = self.rules.allowed_tokens {
                if !allowed.contains(&token) {
                    self.stats.failed_validations += 1;
                    return Err(SequenceError::InvalidPaddingToken { token });
                }
            }
        }

        self.stats.passed_validations += 1;
        Ok(())
    }

    /// Validate a batch of sequences
    pub fn validate_batch(&mut self, sequences: &[Vec<u32>]) -> SequenceResult<()> {
        for sequence in sequences {
            self.validate_sequence(sequence)?;
        }
        Ok(())
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ValidationStats {
        &self.stats
    }

    /// Reset validation statistics
    pub fn reset_stats(&mut self) {
        self.stats = ValidationStats::new();
    }

    /// Get the validation rules
    pub fn rules(&self) -> &ValidationRules {
        &self.rules
    }

    /// Update validation rules
    pub fn update_rules(&mut self, rules: ValidationRules) {
        self.rules = rules;
    }
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    /// Total number of sequences validated
    pub total_validated: usize,
    /// Number of sequences that passed validation
    pub passed_validations: usize,
    /// Number of sequences that failed validation
    pub failed_validations: usize,
    /// Validation success rate
    pub success_rate: f32,
    /// Common validation errors
    pub error_counts: HashMap<String, usize>,
}

impl ValidationStats {
    /// Create new validation statistics
    pub fn new() -> Self {
        Self {
            total_validated: 0,
            passed_validations: 0,
            failed_validations: 0,
            success_rate: 0.0,
            error_counts: HashMap::new(),
        }
    }

    /// Update success rate
    pub fn update_success_rate(&mut self) {
        self.success_rate = if self.total_validated > 0 {
            self.passed_validations as f32 / self.total_validated as f32
        } else {
            0.0
        };
    }

    /// Add an error to the count
    pub fn add_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get the most common error type
    pub fn most_common_error(&self) -> Option<(&String, &usize)> {
        self.error_counts.iter().max_by_key(|(_, &count)| count)
    }
}

/// Validate a single sequence with basic length constraints
///
/// # Arguments
/// * `sequence` - The sequence to validate
/// * `min_length` - Optional minimum length
/// * `max_length` - Optional maximum length
///
/// # Returns
/// Result indicating success or validation error
pub fn validate_sequence_length(
    sequence: &[u32],
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> SequenceResult<()> {
    let length = sequence.len();

    if let Some(min_len) = min_length {
        if length < min_len {
            return Err(SequenceError::BelowMinLength {
                length,
                min_length: min_len,
            });
        }
    }

    if let Some(max_len) = max_length {
        if length > max_len {
            return Err(SequenceError::ExceedsMaxLength {
                length,
                max_length: max_len,
            });
        }
    }

    Ok(())
}

/// Validate that a sequence contains valid tokens
///
/// # Arguments
/// * `sequence` - The sequence to validate
/// * `vocab_size` - Size of the vocabulary (tokens must be < vocab_size)
/// * `special_tokens` - Set of allowed special tokens (can be >= vocab_size)
///
/// # Returns
/// Result indicating success or validation error
pub fn validate_token_ids(
    sequence: &[u32],
    vocab_size: usize,
    special_tokens: Option<&std::collections::HashSet<u32>>,
) -> SequenceResult<()> {
    for (pos, &token) in sequence.iter().enumerate() {
        let is_valid = if token < vocab_size as u32 {
            true
        } else if let Some(special) = special_tokens {
            special.contains(&token)
        } else {
            false
        };

        if !is_valid {
            return Err(SequenceError::InvalidPaddingToken { token });
        }
    }

    Ok(())
}

/// Validate that sequences in a batch have consistent properties
///
/// # Arguments
/// * `sequences` - Batch of sequences to validate
/// * `require_same_length` - Whether all sequences must have the same length
///
/// # Returns
/// Result indicating success or validation error
pub fn validate_batch_consistency(
    sequences: &[Vec<u32>],
    require_same_length: bool,
) -> SequenceResult<()> {
    if sequences.is_empty() {
        return Ok(());
    }

    if require_same_length {
        let first_length = sequences[0].len();
        for (i, sequence) in sequences.iter().enumerate().skip(1) {
            if sequence.len() != first_length {
                return Err(SequenceError::InconsistentBatchLengths);
            }
        }
    }

    Ok(())
}

/// Advanced sequence validator with custom rules
pub struct AdvancedValidator {
    base_validator: SequenceValidator,
    custom_rules: Vec<Box<dyn Fn(&[u32]) -> Result<(), String> + Send + Sync>>,
}

impl AdvancedValidator {
    /// Create a new advanced validator
    pub fn new(rules: ValidationRules) -> Self {
        Self {
            base_validator: SequenceValidator::with_rules(rules),
            custom_rules: Vec::new(),
        }
    }

    /// Add a custom validation rule
    pub fn add_custom_rule<F>(&mut self, rule: F)
    where
        F: Fn(&[u32]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.custom_rules.push(Box::new(rule));
    }

    /// Validate a sequence with all rules
    pub fn validate(&mut self, sequence: &[u32]) -> SequenceResult<()> {
        // Run base validation
        self.base_validator.validate_sequence(sequence)?;

        // Run custom rules
        for rule in &self.custom_rules {
            if let Err(msg) = rule(sequence) {
                return Err(SequenceError::ConfigError { message: msg });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::SequenceConfig;

    #[test]
    fn test_validate_sequence_length() {
        let sequence = vec![1, 2, 3, 4, 5];

        // Valid length
        assert!(validate_sequence_length(&sequence, Some(3), Some(10)).is_ok());

        // Too short
        assert!(validate_sequence_length(&sequence, Some(10), None).is_err());

        // Too long
        assert!(validate_sequence_length(&sequence, None, Some(3)).is_err());
    }

    #[test]
    fn test_validate_token_ids() {
        let sequence = vec![1, 2, 3, 100]; // 100 is out of vocab
        let vocab_size = 50;

        // Should fail without special tokens
        assert!(validate_token_ids(&sequence, vocab_size, None).is_err());

        // Should pass with special tokens
        let mut special_tokens = std::collections::HashSet::new();
        special_tokens.insert(100);
        assert!(validate_token_ids(&sequence, vocab_size, Some(&special_tokens)).is_ok());
    }

    #[test]
    fn test_sequence_validator() {
        let config = SequenceConfig::new().with_min_length(2).with_max_length(10);

        let mut validator = SequenceValidator::new(&config);

        // Valid sequence
        assert!(validator.validate_sequence(&[1, 2, 3]).is_ok());

        // Too short
        assert!(validator.validate_sequence(&[1]).is_err());

        // Too long
        assert!(validator.validate_sequence(&[1; 15]).is_err());

        // Check stats
        let stats = validator.stats();
        assert_eq!(stats.total_validated, 3);
        assert_eq!(stats.passed_validations, 1);
        assert_eq!(stats.failed_validations, 2);
    }

    #[test]
    fn test_validation_rules_builder() {
        let rules = ValidationRules::new()
            .with_min_length(5)
            .with_max_length(100)
            .allow_empty()
            .with_pad_token(0);

        assert_eq!(rules.min_length, Some(5));
        assert_eq!(rules.max_length, Some(100));
        assert!(rules.allow_empty);
        assert_eq!(rules.pad_token, Some(0));
    }

    #[test]
    fn test_batch_consistency_validation() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        // Should pass with same length requirement
        assert!(validate_batch_consistency(&sequences, true).is_ok());

        let inconsistent_sequences = vec![vec![1, 2, 3], vec![4, 5], vec![7, 8, 9]];

        // Should fail with same length requirement
        assert!(validate_batch_consistency(&inconsistent_sequences, true).is_err());

        // Should pass without same length requirement
        assert!(validate_batch_consistency(&inconsistent_sequences, false).is_ok());
    }

    #[test]
    fn test_validation_stats() {
        let mut stats = ValidationStats::new();

        stats.total_validated = 10;
        stats.passed_validations = 8;
        stats.failed_validations = 2;
        stats.update_success_rate();

        assert_eq!(stats.success_rate, 0.8);

        stats.add_error("TooLong");
        stats.add_error("TooShort");
        stats.add_error("TooLong");

        let (most_common, count) = stats.most_common_error().unwrap();
        assert_eq!(most_common, "TooLong");
        assert_eq!(*count, 2);
    }
}
