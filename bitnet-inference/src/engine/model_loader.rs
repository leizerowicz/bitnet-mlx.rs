//! Model loading and serialization infrastructure.

use crate::{Result, InferenceError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// Metadata describing a BitNet model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name identifier
    pub name: String,
    /// Model version string
    pub version: String,
    /// Architecture type (e.g., "bitnet-b1.58")
    pub architecture: String,
    /// Total number of parameters
    pub parameter_count: usize,
    /// Quantization bits used
    pub quantization_bits: u8,
    /// Expected input tensor shape
    pub input_shape: Vec<usize>,
    /// Expected output tensor shape
    pub output_shape: Vec<usize>,
    /// Additional model-specific metadata
    pub extra: HashMap<String, String>,
}

impl ModelMetadata {
    /// Validate that this metadata is compatible with the current system.
    pub fn validate(&self) -> Result<()> {
        if self.quantization_bits != 1 && self.quantization_bits != 2 {
            return Err(InferenceError::model_load(
                format!("Unsupported quantization bits: {}", self.quantization_bits)
            ));
        }

        if self.input_shape.is_empty() {
            return Err(InferenceError::model_load("Input shape cannot be empty"));
        }

        if self.output_shape.is_empty() {
            return Err(InferenceError::model_load("Output shape cannot be empty"));
        }

        Ok(())
    }

    /// Check if this model is compatible with the given input shape.
    pub fn is_compatible_input(&self, shape: &[usize]) -> bool {
        if shape.len() != self.input_shape.len() {
            return false;
        }

        // Check if shapes match (allowing for batch dimension flexibility)
        for (i, (&expected, &actual)) in self.input_shape.iter().zip(shape.iter()).enumerate() {
            if i == 0 {
                // First dimension is batch size - can be different
                continue;
            }
            if expected != actual {
                return false;
            }
        }

        true
    }
}

/// A fully loaded model ready for inference.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model architecture definition
    pub architecture: ModelArchitecture,
    /// Loaded weights data
    pub weights: ModelWeights,
}

/// Architecture definition for a loaded model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelArchitecture {
    /// Layer definitions
    pub layers: Vec<LayerDefinition>,
    /// Execution graph
    pub execution_order: Vec<usize>,
}

/// Definition of a single model layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerDefinition {
    /// Layer unique identifier
    pub id: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Layer-specific parameters
    pub parameters: LayerParameters,
}

/// Types of layers supported.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    BitLinear,
    RMSNorm,
    SwiGLU,
    Embedding,
    OutputProjection,
}

/// Parameters specific to each layer type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LayerParameters {
    BitLinear {
        weight_bits: u8,
        activation_bits: u8,
    },
    RMSNorm {
        eps: f32,
    },
    SwiGLU {
        hidden_dim: usize,
    },
    Embedding {
        vocab_size: usize,
        embedding_dim: usize,
    },
    OutputProjection {
        vocab_size: usize,
    },
}

/// Container for model weight data.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Raw weight data organized by layer
    pub layer_weights: HashMap<usize, Vec<u8>>,
    /// Total size in bytes
    pub total_size: usize,
}

/// High-performance model loader with caching and validation.
pub struct ModelLoader {
    cache_dir: PathBuf,
    max_cache_size: usize,
    validate_checksums: bool,
}

impl ModelLoader {
    /// Create a new model loader with the specified cache directory.
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
            max_cache_size: 1024 * 1024 * 1024, // 1GB default
            validate_checksums: true,
        }
    }

    /// Set the maximum cache size in bytes.
    pub fn with_max_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Enable or disable checksum validation.
    pub fn with_checksum_validation(mut self, validate: bool) -> Self {
        self.validate_checksums = validate;
        self
    }

    /// Load a model from the specified path.
    pub fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<LoadedModel> {
        let path = path.as_ref();

        // 1. Read and validate metadata
        let metadata = self.read_metadata(path)?;
        metadata.validate()?;

        // 2. Load model architecture
        let architecture = self.load_architecture(path, &metadata)?;

        // 3. Load model weights
        let weights = self.load_weights(path)?;

        Ok(LoadedModel {
            metadata,
            architecture,
            weights,
        })
    }

    /// Read model metadata from file.
    fn read_metadata<P: AsRef<Path>>(&self, model_path: P) -> Result<ModelMetadata> {
        let metadata_path = model_path.as_ref().with_extension("json");
        
        if !metadata_path.exists() {
            return Err(InferenceError::model_load(
                format!("Metadata file not found: {}", metadata_path.display())
            ));
        }

        let metadata_content = std::fs::read_to_string(&metadata_path)?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| InferenceError::model_load(
                format!("Failed to parse metadata: {}", e)
            ))?;

        Ok(metadata)
    }

    /// Load model architecture from file.
    fn load_architecture<P: AsRef<Path>>(&self, model_path: P, metadata: &ModelMetadata) -> Result<ModelArchitecture> {
        let arch_path = model_path.as_ref().with_extension("arch");
        
        if !arch_path.exists() {
            // Create default architecture based on metadata
            return Ok(self.create_default_architecture(metadata));
        }

        let arch_data = std::fs::read(&arch_path)?;
        let architecture: ModelArchitecture = bincode::deserialize(&arch_data)?;

        Ok(architecture)
    }

    /// Load model weights from file.
    fn load_weights<P: AsRef<Path>>(&self, model_path: P) -> Result<ModelWeights> {
        let weights_path = model_path.as_ref().with_extension("bin");
        
        if !weights_path.exists() {
            return Err(InferenceError::model_load(
                format!("Weights file not found: {}", weights_path.display())
            ));
        }

        let weights_data = std::fs::read(&weights_path)?;
        
        // For now, store as a single blob - will be structured later
        let mut layer_weights = HashMap::new();
        layer_weights.insert(0, weights_data.clone());

        Ok(ModelWeights {
            layer_weights,
            total_size: weights_data.len(),
        })
    }

    /// Create a default architecture when no architecture file exists.
    fn create_default_architecture(&self, metadata: &ModelMetadata) -> ModelArchitecture {
        // Create a simple default architecture
        // This is a placeholder - real implementation would be more sophisticated
        let layers = vec![
            LayerDefinition {
                id: 0,
                layer_type: LayerType::BitLinear,
                input_dims: metadata.input_shape.clone(),
                output_dims: metadata.output_shape.clone(),
                parameters: LayerParameters::BitLinear {
                    weight_bits: metadata.quantization_bits,
                    activation_bits: metadata.quantization_bits,
                },
            }
        ];

        ModelArchitecture {
            layers,
            execution_order: vec![0],
        }
    }

    /// Get loader statistics.
    pub fn stats(&self) -> ModelLoaderStats {
        ModelLoaderStats {
            cache_dir: self.cache_dir.clone(),
            max_cache_size: self.max_cache_size,
            validate_checksums: self.validate_checksums,
        }
    }
}

/// Statistics for model loader monitoring.
#[derive(Debug, Clone)]
pub struct ModelLoaderStats {
    pub cache_dir: PathBuf,
    pub max_cache_size: usize,
    pub validate_checksums: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_validation() {
        let mut metadata = ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            parameter_count: 1000000,
            quantization_bits: 1,
            input_shape: vec![1, 512],
            output_shape: vec![1, 50000], // Typical vocab size
            extra: HashMap::new(),
        };

        // Valid metadata should pass
        assert!(metadata.validate().is_ok());

        // Invalid quantization bits should fail
        metadata.quantization_bits = 16;
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_input_compatibility() {
        let metadata = ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            parameter_count: 1000000,
            quantization_bits: 1,
            input_shape: vec![1, 512],
            output_shape: vec![1, 30000],
            extra: HashMap::new(),
        };

        // Compatible shapes (different batch size is OK)
        assert!(metadata.is_compatible_input(&[8, 512]));
        assert!(metadata.is_compatible_input(&[1, 512]));

        // Incompatible shapes
        assert!(!metadata.is_compatible_input(&[1, 256]));
        assert!(!metadata.is_compatible_input(&[1, 512, 128]));
    }
}
