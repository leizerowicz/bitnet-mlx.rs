//! Mixed Precision Support for BitNet
//!
//! This module provides comprehensive mixed precision support, allowing different layers
//! and operations to use different precision levels for optimal performance and memory usage.

pub mod config;
pub mod layer_precision;
pub mod precision_manager;
pub mod conversion;
pub mod validation;
pub mod policy;

use crate::memory::tensor::BitNetDType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during mixed precision operations
#[derive(Error, Debug)]
pub enum MixedPrecisionError {
    /// Invalid precision configuration
    #[error("Invalid precision configuration: {0}")]
    InvalidConfiguration(String),

    /// Precision conversion error
    #[error("Precision conversion failed: {from} -> {to}: {reason}")]
    ConversionError {
        from: BitNetDType,
        to: BitNetDType,
        reason: String,
    },

    /// Layer precision mismatch
    #[error("Layer precision mismatch: expected {expected}, got {actual}")]
    LayerPrecisionMismatch {
        expected: BitNetDType,
        actual: BitNetDType,
    },

    /// Unsupported precision combination
    #[error("Unsupported precision combination: {0}")]
    UnsupportedCombination(String),

    /// Memory allocation error for mixed precision
    #[error("Mixed precision memory allocation failed: {0}")]
    MemoryAllocationError(String),

    /// Validation error
    #[error("Mixed precision validation failed: {0}")]
    ValidationError(String),
}

impl From<candle_core::Error> for MixedPrecisionError {
    fn from(error: candle_core::Error) -> Self {
        MixedPrecisionError::ConversionError {
            from: BitNetDType::F32, // Default fallback
            to: BitNetDType::F32,   // Default fallback
            reason: error.to_string(),
        }
    }
}

/// Result type for mixed precision operations
pub type MixedPrecisionResult<T> = Result<T, MixedPrecisionError>;

/// Layer types that can have different precision configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Linear/Dense layers
    Linear,
    /// Convolutional layers
    Convolution,
    /// Attention layers
    Attention,
    /// Embedding layers
    Embedding,
    /// Normalization layers
    Normalization,
    /// Activation layers
    Activation,
    /// Output/Classification layers
    Output,
    /// Custom layer type
    Custom(u32),
}

impl LayerType {
    /// Get the default precision for this layer type
    pub fn default_precision(&self) -> BitNetDType {
        match self {
            LayerType::Linear => BitNetDType::BitNet158,
            LayerType::Convolution => BitNetDType::BitNet158,
            LayerType::Attention => BitNetDType::I8,
            LayerType::Embedding => BitNetDType::F16,
            LayerType::Normalization => BitNetDType::F32,
            LayerType::Activation => BitNetDType::I8,
            LayerType::Output => BitNetDType::F16,
            LayerType::Custom(_) => BitNetDType::F32,
        }
    }

    /// Check if this layer type supports the given precision
    pub fn supports_precision(&self, precision: BitNetDType) -> bool {
        match self {
            LayerType::Linear | LayerType::Convolution => {
                // Linear and conv layers support all quantized precisions
                matches!(precision, 
                    BitNetDType::BitNet158 | BitNetDType::I8 | BitNetDType::I4 | 
                    BitNetDType::I2 | BitNetDType::I1 | BitNetDType::F16 | BitNetDType::F32
                )
            }
            LayerType::Attention => {
                // Attention layers need higher precision for stability
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I8
                )
            }
            LayerType::Embedding => {
                // Embeddings typically use float or high-bit integer
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I8
                )
            }
            LayerType::Normalization => {
                // Normalization layers need float precision
                matches!(precision, BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16)
            }
            LayerType::Activation => {
                // Activations can use various precisions
                true // All precisions supported
            }
            LayerType::Output => {
                // Output layers typically need higher precision
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I8
                )
            }
            LayerType::Custom(_) => true, // Custom layers support all precisions
        }
    }

    /// Get recommended precision alternatives for this layer type
    pub fn precision_alternatives(&self) -> Vec<BitNetDType> {
        match self {
            LayerType::Linear | LayerType::Convolution => vec![
                BitNetDType::BitNet158,
                BitNetDType::I4,
                BitNetDType::I8,
                BitNetDType::F16,
            ],
            LayerType::Attention => vec![
                BitNetDType::I8,
                BitNetDType::F16,
                BitNetDType::BF16,
                BitNetDType::F32,
            ],
            LayerType::Embedding => vec![
                BitNetDType::I8,
                BitNetDType::F16,
                BitNetDType::BF16,
            ],
            LayerType::Normalization => vec![
                BitNetDType::F16,
                BitNetDType::BF16,
                BitNetDType::F32,
            ],
            LayerType::Activation => vec![
                BitNetDType::I8,
                BitNetDType::I4,
                BitNetDType::F16,
            ],
            LayerType::Output => vec![
                BitNetDType::F16,
                BitNetDType::BF16,
                BitNetDType::I8,
            ],
            LayerType::Custom(_) => BitNetDType::all_types().to_vec(),
        }
    }
}

/// Component types within a layer that can have different precisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// Weight parameters
    Weights,
    /// Bias parameters
    Bias,
    /// Activations/Inputs
    Activations,
    /// Gradients (for training)
    Gradients,
    /// Intermediate computations
    Intermediate,
    /// Attention scores
    AttentionScores,
    /// Key-Value cache
    KVCache,
}

impl ComponentType {
    /// Get the default precision for this component type
    pub fn default_precision(&self) -> BitNetDType {
        match self {
            ComponentType::Weights => BitNetDType::BitNet158,
            ComponentType::Bias => BitNetDType::F16,
            ComponentType::Activations => BitNetDType::I8,
            ComponentType::Gradients => BitNetDType::F16,
            ComponentType::Intermediate => BitNetDType::F16,
            ComponentType::AttentionScores => BitNetDType::F16,
            ComponentType::KVCache => BitNetDType::I8,
        }
    }

    /// Check if this component type supports the given precision
    pub fn supports_precision(&self, precision: BitNetDType) -> bool {
        match self {
            ComponentType::Weights => {
                // Weights support all quantized precisions
                true
            }
            ComponentType::Bias => {
                // Bias typically needs higher precision
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I8
                )
            }
            ComponentType::Activations => {
                // Activations can use various precisions
                true
            }
            ComponentType::Gradients => {
                // Gradients need sufficient precision for training
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16
                )
            }
            ComponentType::Intermediate => {
                // Intermediate computations need reasonable precision
                matches!(precision, 
                    BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I8
                )
            }
            ComponentType::AttentionScores => {
                // Attention scores need float precision for stability
                matches!(precision, BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16)
            }
            ComponentType::KVCache => {
                // KV cache can use lower precision for memory efficiency
                true
            }
        }
    }
}

/// Mixed precision strategy for automatic precision selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixedPrecisionStrategy {
    /// Conservative strategy prioritizing accuracy
    Conservative,
    /// Balanced strategy balancing accuracy and efficiency
    Balanced,
    /// Aggressive strategy prioritizing memory and speed
    Aggressive,
    /// Custom strategy with user-defined rules
    Custom,
}

impl MixedPrecisionStrategy {
    /// Get the precision configuration for a layer type using this strategy
    pub fn get_layer_precision(&self, layer_type: LayerType) -> BitNetDType {
        match self {
            MixedPrecisionStrategy::Conservative => match layer_type {
                LayerType::Linear | LayerType::Convolution => BitNetDType::I8,
                LayerType::Attention => BitNetDType::F16,
                LayerType::Embedding => BitNetDType::F16,
                LayerType::Normalization => BitNetDType::F32,
                LayerType::Activation => BitNetDType::F16,
                LayerType::Output => BitNetDType::F16,
                LayerType::Custom(_) => BitNetDType::F32,
            },
            MixedPrecisionStrategy::Balanced => match layer_type {
                LayerType::Linear | LayerType::Convolution => BitNetDType::BitNet158,
                LayerType::Attention => BitNetDType::I8,
                LayerType::Embedding => BitNetDType::F16,
                LayerType::Normalization => BitNetDType::F16,
                LayerType::Activation => BitNetDType::I8,
                LayerType::Output => BitNetDType::F16,
                LayerType::Custom(_) => BitNetDType::F16,
            },
            MixedPrecisionStrategy::Aggressive => match layer_type {
                LayerType::Linear | LayerType::Convolution => BitNetDType::BitNet158,
                LayerType::Attention => BitNetDType::I8,
                LayerType::Embedding => BitNetDType::I8,
                LayerType::Normalization => BitNetDType::F16,
                LayerType::Activation => BitNetDType::I4,
                LayerType::Output => BitNetDType::I8,
                LayerType::Custom(_) => BitNetDType::I8,
            },
            MixedPrecisionStrategy::Custom => layer_type.default_precision(),
        }
    }

    /// Get the precision configuration for a component type using this strategy
    pub fn get_component_precision(&self, component_type: ComponentType) -> BitNetDType {
        match self {
            MixedPrecisionStrategy::Conservative => match component_type {
                ComponentType::Weights => BitNetDType::I8,
                ComponentType::Bias => BitNetDType::F16,
                ComponentType::Activations => BitNetDType::F16,
                ComponentType::Gradients => BitNetDType::F32,
                ComponentType::Intermediate => BitNetDType::F16,
                ComponentType::AttentionScores => BitNetDType::F32,
                ComponentType::KVCache => BitNetDType::I8,
            },
            MixedPrecisionStrategy::Balanced => match component_type {
                ComponentType::Weights => BitNetDType::BitNet158,
                ComponentType::Bias => BitNetDType::F16,
                ComponentType::Activations => BitNetDType::I8,
                ComponentType::Gradients => BitNetDType::F16,
                ComponentType::Intermediate => BitNetDType::F16,
                ComponentType::AttentionScores => BitNetDType::F16,
                ComponentType::KVCache => BitNetDType::I8,
            },
            MixedPrecisionStrategy::Aggressive => match component_type {
                ComponentType::Weights => BitNetDType::BitNet158,
                ComponentType::Bias => BitNetDType::I8,
                ComponentType::Activations => BitNetDType::I4,
                ComponentType::Gradients => BitNetDType::F16,
                ComponentType::Intermediate => BitNetDType::I8,
                ComponentType::AttentionScores => BitNetDType::F16,
                ComponentType::KVCache => BitNetDType::I4,
            },
            MixedPrecisionStrategy::Custom => component_type.default_precision(),
        }
    }
}

/// Re-export commonly used types
pub use config::{MixedPrecisionConfig, LayerPrecisionConfig, ComponentPrecisionConfig};
pub use layer_precision::{LayerPrecisionManager, LayerPrecisionSpec};
pub use precision_manager::{PrecisionManager, PrecisionContext};
pub use conversion::{PrecisionConverter, ConversionStrategy};
pub use validation::{PrecisionValidator, ValidationRule};
pub use policy::{PrecisionPolicy, PolicyRule, PolicyEngine};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_default_precision() {
        assert_eq!(LayerType::Linear.default_precision(), BitNetDType::BitNet158);
        assert_eq!(LayerType::Attention.default_precision(), BitNetDType::I8);
        assert_eq!(LayerType::Normalization.default_precision(), BitNetDType::F32);
    }

    #[test]
    fn test_layer_type_supports_precision() {
        assert!(LayerType::Linear.supports_precision(BitNetDType::BitNet158));
        assert!(LayerType::Attention.supports_precision(BitNetDType::F16));
        assert!(!LayerType::Normalization.supports_precision(BitNetDType::I1));
    }

    #[test]
    fn test_component_type_default_precision() {
        assert_eq!(ComponentType::Weights.default_precision(), BitNetDType::BitNet158);
        assert_eq!(ComponentType::Bias.default_precision(), BitNetDType::F16);
        assert_eq!(ComponentType::Gradients.default_precision(), BitNetDType::F16);
    }

    #[test]
    fn test_mixed_precision_strategy() {
        let strategy = MixedPrecisionStrategy::Balanced;
        assert_eq!(strategy.get_layer_precision(LayerType::Linear), BitNetDType::BitNet158);
        assert_eq!(strategy.get_component_precision(ComponentType::Weights), BitNetDType::BitNet158);
    }

    #[test]
    fn test_precision_alternatives() {
        let alternatives = LayerType::Linear.precision_alternatives();
        assert!(alternatives.contains(&BitNetDType::BitNet158));
        assert!(alternatives.contains(&BitNetDType::I4));
        assert!(alternatives.len() > 0);
    }
}