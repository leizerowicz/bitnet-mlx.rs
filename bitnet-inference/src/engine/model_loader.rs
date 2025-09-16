//! Model loading and serialization infrastructure.

use crate::{Result, InferenceError};
use crate::engine::weight_conversion::{WeightConverter, ConvertedWeights};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;

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
    /// BitNet-specific configuration (when available)
    pub bitnet_config: Option<crate::bitnet_config::BitNetModelConfig>,
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

/// Parameter type within a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ParameterType {
    /// Main layer weights (e.g., linear transformation weights)
    Weight,
    /// Bias parameters
    Bias,
    /// Layer normalization scale parameters
    LayerNormScale,
    /// Attention query weights
    AttentionQuery,
    /// Attention key weights  
    AttentionKey,
    /// Attention value weights
    AttentionValue,
    /// Attention output projection weights
    AttentionOutput,
    /// Feed-forward gate weights (for gated activations)
    FeedForwardGate,
    /// Feed-forward up projection weights
    FeedForwardUp,
    /// Feed-forward down projection weights
    FeedForwardDown,
    /// Embedding weights
    EmbeddingWeight,
    /// Output projection weights
    OutputWeight,
}

/// Parameter data with metadata.
#[derive(Debug, Clone)]
pub struct ParameterData {
    /// Raw parameter data
    pub data: Vec<u8>,
    /// Parameter shape
    pub shape: Vec<usize>,
    /// Data type information
    pub dtype: ParameterDataType,
    /// Original tensor name from model file
    pub tensor_name: String,
}

/// Data type of parameter.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ParameterDataType {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// 8-bit integer
    I8,
    /// BitNet 1.58-bit ternary weights
    BitnetB158,
    /// Quantized formats
    Quantized(String),
}

/// Container for model weight data organized by layer and parameter type.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Weights organized by layer ID and parameter type
    pub organized_weights: HashMap<usize, HashMap<ParameterType, ParameterData>>,
    /// Raw weight data organized by tensor index (for backward compatibility)
    pub layer_weights: HashMap<usize, Vec<u8>>,
    /// Total size in bytes
    pub total_size: usize,
    /// Layer index mapping from tensor names to layer IDs
    pub layer_mapping: HashMap<String, usize>,
    /// Weight converter for lazy conversion (shared)
    weight_converter: Option<Arc<WeightConverter>>,
}

impl ModelWeights {
    /// Create new ModelWeights with empty organization
    pub fn new() -> Self {
        Self {
            organized_weights: HashMap::new(),
            layer_weights: HashMap::new(),
            total_size: 0,
            layer_mapping: HashMap::new(),
            weight_converter: None,
        }
    }

    /// Create new ModelWeights with weight converter
    pub fn with_converter(converter: Arc<WeightConverter>) -> Self {
        Self {
            organized_weights: HashMap::new(),
            layer_weights: HashMap::new(),
            total_size: 0,
            layer_mapping: HashMap::new(),
            weight_converter: Some(converter),
        }
    }

    /// Set the weight converter
    pub fn set_converter(&mut self, converter: Arc<WeightConverter>) {
        self.weight_converter = Some(converter);
    }

    /// Get parameter data for a specific layer and parameter type
    pub fn get_parameter(&self, layer_id: usize, param_type: ParameterType) -> Option<&ParameterData> {
        self.organized_weights
            .get(&layer_id)?
            .get(&param_type)
    }

    /// Get all parameters for a specific layer
    pub fn get_layer_parameters(&self, layer_id: usize) -> Option<&HashMap<ParameterType, ParameterData>> {
        self.organized_weights.get(&layer_id)
    }

    /// Add parameter data to the organized structure
    pub fn add_parameter(&mut self, layer_id: usize, param_type: ParameterType, data: ParameterData) {
        self.organized_weights
            .entry(layer_id)
            .or_insert_with(HashMap::new)
            .insert(param_type, data);
    }

    /// Check if a layer has a specific parameter type
    pub fn has_parameter(&self, layer_id: usize, param_type: ParameterType) -> bool {
        self.organized_weights
            .get(&layer_id)
            .map(|params| params.contains_key(&param_type))
            .unwrap_or(false)
    }

    /// Get all layer IDs that have been loaded
    pub fn get_layer_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.organized_weights.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get parameter count for a specific layer
    pub fn get_layer_parameter_count(&self, layer_id: usize) -> usize {
        self.organized_weights
            .get(&layer_id)
            .map(|params| params.len())
            .unwrap_or(0)
    }

    /// Get total parameter count across all layers
    pub fn get_total_parameter_count(&self) -> usize {
        self.organized_weights
            .values()
            .map(|params| params.len())
            .sum()
    }

    /// Map tensor name to layer ID
    pub fn map_tensor_to_layer(&mut self, tensor_name: String, layer_id: usize) {
        self.layer_mapping.insert(tensor_name, layer_id);
    }

    /// Get layer ID from tensor name
    pub fn get_layer_id_from_tensor(&self, tensor_name: &str) -> Option<usize> {
        self.layer_mapping.get(tensor_name).copied()
    }

    /// Convert parameter to typed weight arrays
    pub fn convert_parameter(&self, layer_id: usize, param_type: ParameterType) -> Result<ConvertedWeights> {
        let param_data = self.get_parameter(layer_id, param_type)
            .ok_or_else(|| InferenceError::model_load(
                format!("Parameter not found: layer {}, type {:?}", layer_id, param_type)
            ))?;

        let converter = self.weight_converter.as_ref()
            .ok_or_else(|| InferenceError::model_load(
                "Weight converter not available".to_string()
            ))?;

        converter.convert_parameter(param_data)
    }

    /// Convert all parameters for a layer
    pub fn convert_layer_parameters(&self, layer_id: usize) -> Result<HashMap<ParameterType, ConvertedWeights>> {
        let layer_params = self.get_layer_parameters(layer_id)
            .ok_or_else(|| InferenceError::model_load(
                format!("Layer {} not found", layer_id)
            ))?;

        let converter = self.weight_converter.as_ref()
            .ok_or_else(|| InferenceError::model_load(
                "Weight converter not available".to_string()
            ))?;

        let mut converted = HashMap::new();
        for (param_type, param_data) in layer_params {
            let converted_weights = converter.convert_parameter(param_data)?;
            converted.insert(*param_type, converted_weights);
        }

        Ok(converted)
    }

    /// Check if weight converter is available
    pub fn has_converter(&self) -> bool {
        self.weight_converter.is_some()
    }

    /// Get weight converter cache statistics
    pub fn converter_stats(&self) -> Option<(usize, usize, usize)> {
        self.weight_converter.as_ref().map(|c| c.cache_stats())
    }
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
            bitnet_config: None, // Default to None for non-GGUF models
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
        let mut weights = ModelWeights::new();
        weights.layer_weights.insert(0, weights_data.clone());
        weights.total_size = weights_data.len();

        Ok(weights)
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
