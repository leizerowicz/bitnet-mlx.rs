//! Zero-copy model loading with memory mapping for large models.

use crate::{Result, InferenceError};
use crate::engine::model_loader::{LoadedModel, ModelMetadata, ModelArchitecture, ModelWeights, LayerDefinition};
use crate::cache::{ExecutionPlan, LayerExecution, MemoryLayout, FusionGroup, ExecutionLayerType, DevicePlacement, TensorSpec, TensorLayout};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use serde::{Deserialize, Serialize};

/// Threshold for using memory mapping (64MB)
const MMAP_THRESHOLD: usize = 64 * 1024 * 1024;

/// Zero-copy model loader with memory mapping optimization
pub struct ZeroCopyModelLoader {
    /// Memory mapping threshold - files larger than this use mmap
    mmap_threshold: usize,
    /// Whether to validate model integrity
    validate_integrity: bool,
    /// Cache for parsed model headers
    header_cache: HashMap<String, ModelHeader>,
}

/// Memory-mapped model representation
pub enum MmapModel {
    /// Memory-mapped file (for large models)
    Mapped {
        mmap: Mmap,
        header: ModelHeader,
        weights_offset: usize,
    },
    /// In-memory buffer (for small models)
    InMemory {
        data: Vec<u8>,
        header: ModelHeader,
        weights_offset: usize,
    },
}

/// Model file header with metadata and layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHeader {
    /// Magic number for format identification
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Architecture definition
    pub architecture: ModelArchitecture,
    /// Weight layout information
    pub weight_layout: WeightLayout,
    /// Checksum for integrity validation
    pub checksum: u64,
}

/// Weight storage layout in the file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightLayout {
    /// Offset to weight data from file start
    pub offset: u64,
    /// Total size of weight data
    pub total_size: u64,
    /// Individual layer weight information
    pub layer_info: Vec<LayerWeightInfo>,
    /// Whether weights are compressed
    pub compressed: bool,
    /// Compression algorithm used (if any)
    pub compression: Option<String>,
}

/// Information about individual layer weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeightInfo {
    /// Layer identifier
    pub layer_id: usize,
    /// Offset within weight section
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
    /// Data type of weights
    pub dtype: String,
    /// Shape of weight tensor
    pub shape: Vec<usize>,
}

impl ZeroCopyModelLoader {
    /// Create a new zero-copy model loader
    pub fn new() -> Self {
        Self {
            mmap_threshold: MMAP_THRESHOLD,
            validate_integrity: true,
            header_cache: HashMap::new(),
        }
    }

    /// Create loader with custom memory mapping threshold
    pub fn with_mmap_threshold(threshold: usize) -> Self {
        Self {
            mmap_threshold: threshold,
            validate_integrity: true,
            header_cache: HashMap::new(),
        }
    }

    /// Load model using zero-copy techniques when possible
    pub fn load_model_zero_copy<P: AsRef<Path>>(&mut self, path: P) -> Result<MmapModel> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| InferenceError::model_load(&format!("Failed to open model file: {}", e)))?;
        
        let metadata = file.metadata()
            .map_err(|e| InferenceError::model_load(&format!("Failed to get file metadata: {}", e)))?;
        
        let file_size = metadata.len() as usize;
        
        if file_size > self.mmap_threshold {
            // Use memory mapping for large files
            self.load_with_mmap(file, file_size)
        } else {
            // Load small files into memory
            self.load_into_memory(file, file_size)
        }
    }

    /// Load model with memory mapping
    fn load_with_mmap(&self, mut file: File, file_size: usize) -> Result<MmapModel> {
        // First, read the header to understand the file structure
        let header = self.read_model_header(&mut file)?;
        
        // Validate header if requested
        if self.validate_integrity {
            self.validate_model_header(&header, file_size)?;
        }

        // Memory map the entire file
        let mmap = unsafe {
            MmapOptions::new().map(&file)
                .map_err(|e| InferenceError::model_load(&format!("Failed to memory map file: {}", e)))?
        };

        Ok(MmapModel::Mapped {
            mmap,
            header: header.clone(),
            weights_offset: header.weight_layout.offset as usize,
        })
    }

    /// Load small model into memory
    fn load_into_memory(&self, mut file: File, file_size: usize) -> Result<MmapModel> {
        let header = self.read_model_header(&mut file)?;
        
        if self.validate_integrity {
            self.validate_model_header(&header, file_size)?;
        }

        // Read entire file into memory
        file.seek(SeekFrom::Start(0))
            .map_err(|e| InferenceError::model_load(&format!("Failed to seek to file start: {}", e)))?;
        
        let mut data = Vec::with_capacity(file_size);
        file.read_to_end(&mut data)
            .map_err(|e| InferenceError::model_load(&format!("Failed to read file: {}", e)))?;

        Ok(MmapModel::InMemory {
            data,
            header: header.clone(),
            weights_offset: header.weight_layout.offset as usize,
        })
    }

    /// Read model header from file
    fn read_model_header(&self, file: &mut File) -> Result<ModelHeader> {
        // Read magic number and version first
        let mut header_size_buf = [0u8; 8];
        file.read_exact(&mut header_size_buf)
            .map_err(|e| InferenceError::model_load(&format!("Failed to read header size: {}", e)))?;
        
        let magic = u32::from_le_bytes([header_size_buf[0], header_size_buf[1], header_size_buf[2], header_size_buf[3]]);
        let version = u32::from_le_bytes([header_size_buf[4], header_size_buf[5], header_size_buf[6], header_size_buf[7]]);
        
        // Validate magic number
        if magic != 0x42544E45 { // "BTNE" (BitNet Engine)
            return Err(InferenceError::model_load("Invalid model file format"));
        }

        // Read header length
        let mut header_len_buf = [0u8; 4];
        file.read_exact(&mut header_len_buf)
            .map_err(|e| InferenceError::model_load(&format!("Failed to read header length: {}", e)))?;
        let header_len = u32::from_le_bytes(header_len_buf) as usize;

        // Read and deserialize header
        let mut header_data = vec![0u8; header_len];
        file.read_exact(&mut header_data)
            .map_err(|e| InferenceError::model_load(&format!("Failed to read header data: {}", e)))?;
        
        let mut header: ModelHeader = bincode::deserialize(&header_data)
            .map_err(|e| InferenceError::serialization(&format!("Failed to deserialize header: {}", e)))?;
        
        // Set the magic and version from file
        header.magic = magic;
        header.version = version;
        
        Ok(header)
    }

    /// Validate model header integrity
    fn validate_model_header(&self, header: &ModelHeader, file_size: usize) -> Result<()> {
        // Check version compatibility
        if header.version > 1 {
            return Err(InferenceError::model_load(&format!(
                "Unsupported model format version: {}", header.version
            )));
        }

        // Validate weight layout
        let expected_end = header.weight_layout.offset + header.weight_layout.total_size;
        if expected_end > file_size as u64 {
            return Err(InferenceError::model_load(
                "Invalid weight layout: exceeds file size"
            ));
        }

        // Validate layer weight info
        let mut total_layer_size = 0u64;
        for layer_info in &header.weight_layout.layer_info {
            if layer_info.offset + layer_info.size > header.weight_layout.total_size {
                return Err(InferenceError::model_load(&format!(
                    "Layer {} weight data exceeds weight section", layer_info.layer_id
                )));
            }
            total_layer_size += layer_info.size;
        }

        // Check if total layer sizes make sense
        if total_layer_size > header.weight_layout.total_size {
            return Err(InferenceError::model_load("Layer weight sizes exceed total weight size"));
        }

        Ok(())
    }

    /// Create execution plan from memory-mapped model
    pub fn create_execution_plan(&self, model: &MmapModel) -> Result<ExecutionPlan> {
        let header = model.header();
        let layers = self.analyze_model_layers(&header.architecture)?;
        let memory_layout = self.optimize_memory_layout(&layers)?;
        let operator_fusion = self.identify_fusion_opportunities(&layers)?;
        let estimated_memory = self.calculate_memory_requirements(&layers);

        Ok(ExecutionPlan {
            layers,
            memory_layout,
            operator_fusion,
            estimated_memory,
        })
    }

    /// Convert memory-mapped model to loaded model (may copy data)
    pub fn convert_to_loaded_model(&self, mmap_model: MmapModel) -> Result<LoadedModel> {
        let header = mmap_model.header().clone();
        let weights = self.extract_weights(&mmap_model)?;

        Ok(LoadedModel {
            metadata: header.metadata,
            architecture: header.architecture,
            weights,
        })
    }

    /// Extract weights from memory-mapped model
    fn extract_weights(&self, model: &MmapModel) -> Result<ModelWeights> {
        let header = model.header();
        let mut layer_weights = HashMap::new();
        let weights_data = model.weights_data();

        for layer_info in &header.weight_layout.layer_info {
            let start = layer_info.offset as usize;
            let end = start + layer_info.size as usize;
            
            if end > weights_data.len() {
                return Err(InferenceError::model_load(&format!(
                    "Layer {} weight data out of bounds", layer_info.layer_id
                )));
            }

            let layer_data = if header.weight_layout.compressed {
                // TODO: Implement decompression based on compression algorithm
                weights_data[start..end].to_vec()
            } else {
                weights_data[start..end].to_vec()
            };

            layer_weights.insert(layer_info.layer_id, layer_data);
        }

        Ok(ModelWeights {
            layer_weights,
            total_size: header.weight_layout.total_size as usize,
        })
    }

    /// Analyze model layers for execution planning
    fn analyze_model_layers(&self, architecture: &ModelArchitecture) -> Result<Vec<LayerExecution>> {
        let mut layers = Vec::new();

        for layer_def in &architecture.layers {
            let layer_execution = self.convert_to_layer_execution(layer_def)?;
            layers.push(layer_execution);
        }

        Ok(layers)
    }

    /// Convert layer definition to execution layer
    fn convert_to_layer_execution(&self, layer_def: &LayerDefinition) -> Result<LayerExecution> {
        
        let layer_type = match layer_def.layer_type {
            crate::engine::model_loader::LayerType::BitLinear => {
                if let crate::engine::model_loader::LayerParameters::BitLinear { weight_bits, activation_bits } = &layer_def.parameters {
                    ExecutionLayerType::BitLinear {
                        input_dim: layer_def.input_dims.get(0).copied().unwrap_or(512),
                        output_dim: layer_def.output_dims.get(0).copied().unwrap_or(512),
                        weight_bits: *weight_bits,
                        activation_bits: *activation_bits,
                    }
                } else {
                    return Err(InferenceError::model_load("Invalid BitLinear layer parameters"));
                }
            }
            crate::engine::model_loader::LayerType::RMSNorm => {
                if let crate::engine::model_loader::LayerParameters::RMSNorm { eps } = &layer_def.parameters {
                    ExecutionLayerType::RMSNorm {
                        eps: *eps,
                        normalized_shape: layer_def.input_dims.clone(),
                    }
                } else {
                    return Err(InferenceError::model_load("Invalid RMSNorm layer parameters"));
                }
            }
            crate::engine::model_loader::LayerType::SwiGLU => {
                if let crate::engine::model_loader::LayerParameters::SwiGLU { hidden_dim } = &layer_def.parameters {
                    ExecutionLayerType::SwiGLU {
                        hidden_dim: *hidden_dim,
                    }
                } else {
                    return Err(InferenceError::model_load("Invalid SwiGLU layer parameters"));
                }
            }
            crate::engine::model_loader::LayerType::Embedding => {
                if let crate::engine::model_loader::LayerParameters::Embedding { vocab_size, embedding_dim } = &layer_def.parameters {
                    ExecutionLayerType::Embedding {
                        vocab_size: *vocab_size,
                        embedding_dim: *embedding_dim,
                    }
                } else {
                    return Err(InferenceError::model_load("Invalid Embedding layer parameters"));
                }
            }
            crate::engine::model_loader::LayerType::OutputProjection => {
                if let crate::engine::model_loader::LayerParameters::OutputProjection { vocab_size } = &layer_def.parameters {
                    ExecutionLayerType::OutputProjection {
                        vocab_size: *vocab_size,
                    }
                } else {
                    return Err(InferenceError::model_load("Invalid OutputProjection layer parameters"));
                }
            }
        };

        Ok(LayerExecution {
            id: layer_def.id,
            layer_type,
            inputs: vec![TensorSpec {
                shape: layer_def.input_dims.clone(),
                dtype: "f32".to_string(),
                layout: TensorLayout::Contiguous,
            }],
            outputs: vec![TensorSpec {
                shape: layer_def.output_dims.clone(),
                dtype: "f32".to_string(),
                layout: TensorLayout::Contiguous,
            }],
            device_placement: DevicePlacement::CPU, // Default to CPU, can be optimized later
        })
    }

    /// Optimize memory layout based on layer analysis
    fn optimize_memory_layout(&self, layers: &[LayerExecution]) -> Result<MemoryLayout> {
        // Analyze memory access patterns
        let total_memory: usize = layers.iter()
            .map(|layer| self.estimate_layer_memory(layer))
            .sum();

        if total_memory > 1024 * 1024 * 1024 { // > 1GB
            // Use pooled memory for large models
            Ok(MemoryLayout::Pooled { 
                pool_size: total_memory + (total_memory / 4) // 25% extra for overhead
            })
        } else if layers.len() > 20 {
            // Use cache-optimized layout for models with many layers
            Ok(MemoryLayout::CacheOptimized)
        } else {
            // Use sequential layout for simple models
            Ok(MemoryLayout::Sequential)
        }
    }

    /// Identify fusion opportunities in the execution graph
    fn identify_fusion_opportunities(&self, layers: &[LayerExecution]) -> Result<Vec<FusionGroup>> {
        let mut fusion_groups = Vec::new();

        // Look for BitLinear + RMSNorm patterns
        for i in 0..(layers.len().saturating_sub(1)) {
            if matches!(layers[i].layer_type, ExecutionLayerType::BitLinear { .. }) &&
               matches!(layers[i + 1].layer_type, ExecutionLayerType::RMSNorm { .. }) {
                fusion_groups.push(FusionGroup {
                    fused_layers: vec![i, i + 1],
                    fusion_type: crate::cache::FusionType::Custom("BitLinear+RMSNorm".to_string()),
                    performance_gain: 0.25, // 25% estimated improvement
                });
            }
        }

        // Look for SwiGLU fusion opportunities
        for i in 0..(layers.len().saturating_sub(2)) {
            if matches!(layers[i].layer_type, ExecutionLayerType::BitLinear { .. }) &&
               matches!(layers[i + 1].layer_type, ExecutionLayerType::SwiGLU { .. }) &&
               matches!(layers[i + 2].layer_type, ExecutionLayerType::BitLinear { .. }) {
                fusion_groups.push(FusionGroup {
                    fused_layers: vec![i, i + 1, i + 2],
                    fusion_type: crate::cache::FusionType::Custom("FFN-SwiGLU".to_string()),
                    performance_gain: 0.30, // 30% estimated improvement
                });
            }
        }

        Ok(fusion_groups)
    }

    /// Calculate memory requirements for the execution plan
    fn calculate_memory_requirements(&self, layers: &[LayerExecution]) -> usize {
        layers.iter()
            .map(|layer| self.estimate_layer_memory(layer))
            .sum::<usize>() * 2 // Double for intermediate tensors
    }

    /// Estimate memory usage for a single layer
    fn estimate_layer_memory(&self, layer: &LayerExecution) -> usize {
        let input_size: usize = layer.inputs.iter()
            .map(|spec| spec.shape.iter().product::<usize>())
            .sum();
        let output_size: usize = layer.outputs.iter()
            .map(|spec| spec.shape.iter().product::<usize>())
            .sum();
        
        // Base memory for tensors (assuming f32)
        let tensor_memory = (input_size + output_size) * 4;
        
        // Add weight memory estimation
        let weight_memory = match &layer.layer_type {
            ExecutionLayerType::BitLinear { input_dim, output_dim, weight_bits, .. } => {
                (input_dim * output_dim * (*weight_bits as usize)) / 8
            }
            ExecutionLayerType::Embedding { vocab_size, embedding_dim } => {
                vocab_size * embedding_dim * 4 // f32 embeddings
            }
            _ => 1024, // Default 1KB for other layers
        };
        
        tensor_memory + weight_memory
    }
}

impl MmapModel {
    /// Get the model header
    pub fn header(&self) -> &ModelHeader {
        match self {
            MmapModel::Mapped { header, .. } => header,
            MmapModel::InMemory { header, .. } => header,
        }
    }

    /// Get weights data as byte slice
    pub fn weights_data(&self) -> &[u8] {
        match self {
            MmapModel::Mapped { mmap, weights_offset, .. } => {
                &mmap[*weights_offset..]
            }
            MmapModel::InMemory { data, weights_offset, .. } => {
                &data[*weights_offset..]
            }
        }
    }

    /// Get a specific layer's weight data
    pub fn layer_weights(&self, layer_id: usize) -> Option<&[u8]> {
        let header = self.header();
        let weights_data = self.weights_data();

        for layer_info in &header.weight_layout.layer_info {
            if layer_info.layer_id == layer_id {
                let start = layer_info.offset as usize;
                let end = start + layer_info.size as usize;
                
                if end <= weights_data.len() {
                    return Some(&weights_data[start..end]);
                }
                break;
            }
        }
        
        None
    }

    /// Check if model is memory-mapped
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self, MmapModel::Mapped { .. })
    }

    /// Get total file size
    pub fn total_size(&self) -> usize {
        match self {
            MmapModel::Mapped { mmap, .. } => mmap.len(),
            MmapModel::InMemory { data, .. } => data.len(),
        }
    }
}

impl Default for ZeroCopyModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_model_file() -> Result<NamedTempFile> {
        let mut file = NamedTempFile::new()
            .map_err(|e| InferenceError::model_load(&format!("Failed to create temp file: {}", e)))?;

        // Write mock model file
        let magic = 0x42544E45u32; // "BTNE"
        let version = 1u32;
        let header_len = 100u32; // Mock header length
        
        file.write_all(&magic.to_le_bytes())?;
        file.write_all(&version.to_le_bytes())?;
        file.write_all(&header_len.to_le_bytes())?;
        
        // Write mock header data
        let mock_header = vec![0u8; header_len as usize];
        file.write_all(&mock_header)?;
        
        // Write some mock weight data
        let weights = vec![1.0f32; 1000];
        let weight_bytes = weights.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>();
        file.write_all(&weight_bytes)?;
        
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_zero_copy_loader_creation() {
        let loader = ZeroCopyModelLoader::new();
        assert_eq!(loader.mmap_threshold, MMAP_THRESHOLD);
        assert!(loader.validate_integrity);
    }

    #[test]
    fn test_mmap_threshold_configuration() {
        let custom_threshold = 32 * 1024 * 1024; // 32MB
        let loader = ZeroCopyModelLoader::with_mmap_threshold(custom_threshold);
        assert_eq!(loader.mmap_threshold, custom_threshold);
    }

    #[test]
    fn test_model_header_validation() {
        let loader = ZeroCopyModelLoader::new();
        
        // Test invalid version
        let header = ModelHeader {
            magic: 0x42544E45,
            version: 999, // Unsupported version
            metadata: ModelMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                architecture: "test".to_string(),
                parameter_count: 1000,
                quantization_bits: 1,
                input_shape: vec![1, 512],
                output_shape: vec![1, 1000],
                extra: HashMap::new(),
            },
            architecture: ModelArchitecture {
                layers: vec![],
                execution_order: vec![],
            },
            weight_layout: WeightLayout {
                offset: 1000,
                total_size: 4000,
                layer_info: vec![],
                compressed: false,
                compression: None,
            },
            checksum: 0,
        };
        
        let result = loader.validate_model_header(&header, 10000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported model format version"));
    }

    #[test] 
    fn test_memory_layout_optimization() {
        let loader = ZeroCopyModelLoader::new();
        
        // Small model should use sequential layout
        let small_layers = vec![
            LayerExecution {
                id: 0,
                layer_type: ExecutionLayerType::BitLinear {
                    input_dim: 512,
                    output_dim: 512,
                    weight_bits: 1,
                    activation_bits: 8,
                },
                inputs: vec![],
                outputs: vec![],
                device_placement: DevicePlacement::CPU,
            }
        ];
        
        let layout = loader.optimize_memory_layout(&small_layers).unwrap();
        assert!(matches!(layout, MemoryLayout::Sequential));
    }

    #[test]
    fn test_fusion_opportunity_identification() {
        let loader = ZeroCopyModelLoader::new();
        
        let layers = vec![
            LayerExecution {
                id: 0,
                layer_type: ExecutionLayerType::BitLinear {
                    input_dim: 512,
                    output_dim: 512,
                    weight_bits: 1,
                    activation_bits: 8,
                },
                inputs: vec![],
                outputs: vec![],
                device_placement: DevicePlacement::CPU,
            },
            LayerExecution {
                id: 1,
                layer_type: ExecutionLayerType::RMSNorm {
                    eps: 1e-6,
                    normalized_shape: vec![512],
                },
                inputs: vec![],
                outputs: vec![],
                device_placement: DevicePlacement::CPU,
            }
        ];
        
        let fusion_groups = loader.identify_fusion_opportunities(&layers).unwrap();
        assert_eq!(fusion_groups.len(), 1);
        assert_eq!(fusion_groups[0].fused_layers, vec![0, 1]);
        assert!(matches!(
            fusion_groups[0].fusion_type,
            crate::cache::FusionType::Custom(ref name) if name == "BitLinear+RMSNorm"
        ));
    }
}
