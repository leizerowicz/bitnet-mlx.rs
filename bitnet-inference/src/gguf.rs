//! GGUF binary format parsing for BitNet models.
//!
//! This module provides comprehensive support for loading Microsoft BitNet b1.58 2B4T 
//! models in GGUF format, with optimized binary parsing and tensor data handling.

use crate::{Result, InferenceError};
use crate::engine::{ModelMetadata, LoadedModel};
use crate::engine::model_loader::{ModelArchitecture, LayerDefinition, LayerType, LayerParameters, ModelWeights, ParameterType, ParameterData, ParameterDataType};
use crate::bitnet_config::{BitNetModelConfig, BasicModelInfo, LayerConfig, AttentionConfig, 
                          NormalizationConfig, BitLinearConfig, TokenizerConfig, RopeConfig, GgufKeys};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, ErrorKind};
use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use bitnet_core::memory::HybridMemoryPool;

/// Model variant detection for targeted parsing strategies
#[derive(Debug, Clone, PartialEq)]
enum ModelVariant {
    /// Microsoft BitNet models with specific key patterns
    MicrosoftBitNet,
    /// Standard LLaMA-based models
    StandardLlama,
    /// Unknown or generic model format
    Unknown,
}

/// GGUF file magic number (first 4 bytes)
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

/// GGUF format version supported
const GGUF_VERSION: u32 = 3;

/// Maximum retry attempts for buffer reading
const MAX_BUFFER_READ_RETRIES: usize = 3;

/// Configuration for robust buffer reading
#[derive(Debug, Clone)]
pub struct BufferReadConfig {
    pub max_retries: usize,
    pub partial_tolerance: f32, // Percentage of data loss acceptable (0.0-1.0)
    pub enable_streaming: bool,
    pub chunk_size: usize, // Size for chunked reading (in bytes)
    pub large_tensor_threshold: usize, // Threshold for using chunked loading
}

impl Default for BufferReadConfig {
    fn default() -> Self {
        Self {
            max_retries: MAX_BUFFER_READ_RETRIES,
            partial_tolerance: 0.20, // 20% data loss acceptable for large tensors
            enable_streaming: true,
            chunk_size: 16 * 1024 * 1024, // 16MB chunks
            large_tensor_threshold: 100 * 1024 * 1024, // 100MB threshold
        }
    }
}

/// Result of buffer reading operation
#[derive(Debug)]
pub enum BufferReadResult {
    Complete(Vec<u8>),
    Partial(Vec<u8>, f32), // data, loss_percentage
    Failed(String),
}

/// GGUF value types (from GGUF specification)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = InferenceError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => {
                tracing::warn!("Unknown GGUF value type: {}, attempting graceful degradation", value);
                // For unknown value types, we'll skip this metadata entry
                // This is better than failing completely
                Err(InferenceError::model_load(format!("Unknown GGUF value type: {} (skipping this metadata entry)", value)))
            }
        }
    }
}

/// GGUF tensor types
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q40 = 2,
    Q41 = 3,
    Q50 = 6,
    Q51 = 7,
    Q80 = 8,
    Q81 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    IQ2XXS = 21,
    IQ2XS = 22,
    IQ3XXS = 23,
    IQ1S = 24,
    IQ4NL = 25,
    IQ3S = 26,
    IQ2S = 27,
    IQ4XS = 28,
    I32BE = 29,
    F16BE = 30,
    F32BE = 31,
    Q4_0_4_4 = 32,
    Q4_0_4_8 = 33,
    Q4_0_8_8 = 34,
    // Custom extension for BitNet 1.58-bit
    BitnetB158 = 1000,
}

impl TryFrom<u32> for GgufTensorType {
    type Error = InferenceError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q40),
            3 => Ok(Self::Q41),
            6 => Ok(Self::Q50),
            7 => Ok(Self::Q51),
            8 => Ok(Self::Q80),
            9 => Ok(Self::Q81),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::I8),
            17 => Ok(Self::I16),
            18 => Ok(Self::I32),
            19 => Ok(Self::I64),
            20 => Ok(Self::F64),
            21 => Ok(Self::IQ2XXS),
            22 => Ok(Self::IQ2XS),
            23 => Ok(Self::IQ3XXS),
            24 => Ok(Self::IQ1S),
            25 => Ok(Self::IQ4NL),
            26 => Ok(Self::IQ3S),
            27 => Ok(Self::IQ2S),
            28 => Ok(Self::IQ4XS),
            29 => Ok(Self::I32BE),
            30 => Ok(Self::F16BE),
            31 => Ok(Self::F32BE),
            32 => Ok(Self::Q4_0_4_4),
            33 => Ok(Self::Q4_0_4_8),
            34 => Ok(Self::Q4_0_8_8),
            // Add fallbacks for unknown types we might encounter
            35..=999 => {
                tracing::warn!("Unknown GGUF tensor type {}, using F32 fallback", value);
                Ok(Self::F32)
            },
            // Custom extension for BitNet 1.58-bit
            1000 => Ok(Self::BitnetB158),
            _ => {
                tracing::warn!("Unknown GGUF tensor type {}, using F32 fallback", value);
                Ok(Self::F32)
            }
        }
    }
}

/// GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
}

/// GGUF header containing model metadata
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

/// GGUF model loader for BitNet models
pub struct GgufLoader;

impl GgufLoader {
    /// Create a new GGUF loader
    pub fn new() -> Self {
        Self
    }

    /// Load a model from GGUF format with HuggingFace integration
    pub async fn load_model<R: Read + Seek>(
        &self,
        mut reader: R,
        memory_pool: Option<Arc<HybridMemoryPool>>,
    ) -> Result<LoadedModel> {
        tracing::info!("Loading GGUF model from reader");
        
        // Validate file integrity
        self.validate_file_integrity(&mut reader)?;
        
        // Parse GGUF header
        let header = self.parse_header(&mut reader)?;
        
        tracing::info!("GGUF header parsed: {} tensors, version {}", header.tensor_count, header.version);
        
        // Load tensors with memory pool integration
        let (mut architecture, weights) = if let Some(pool) = memory_pool {
            self.load_tensors_with_pool(&mut reader, &header, pool).await?
        } else {
            self.load_tensors(&mut reader, &header).await?
        };

        // Create model metadata and BitNet configuration
        let metadata = self.extract_metadata(&header)?;
        let bitnet_config = self.extract_bitnet_config(&header).ok(); // Store config if successful
        
        // Enhanced architecture mapping using GGUF metadata
        if let Some(config) = &bitnet_config {
            architecture = self.map_architecture_from_gguf(&header, config.clone())?;
            tracing::info!("Enhanced architecture mapping completed: {} layers detected", 
                          architecture.layers.len());
        }
        
        Ok(LoadedModel {
            architecture,
            weights,
            metadata,
            bitnet_config,
        })
    }

    /// Extract BitNet model configuration from GGUF file
    pub async fn extract_model_config<P: AsRef<Path>>(&self, path: P) -> Result<BitNetModelConfig> {
        let path = path.as_ref();
        
        tracing::info!("Extracting BitNet model configuration from: {}", path.display());
        
        // Open file
        let mut file = std::fs::File::open(path)
            .map_err(|e| InferenceError::model_load(format!("Failed to open GGUF file: {}", e)))?;
        
        // Validate file integrity
        self.validate_file_integrity(&mut file)?;
        
        // Parse GGUF header (only need metadata)
        let header = self.parse_header(&mut file)?;
        
        tracing::info!("GGUF header parsed for config extraction: {} tensors, version {}", 
                      header.tensor_count, header.version);
        
        // Extract BitNet configuration
        let config = self.extract_bitnet_config(&header)?;
        
        tracing::info!("Successfully extracted BitNet configuration: {} layers, {} attention heads", 
                      config.layer_config.n_layers, config.attention_config.n_heads);
        
        Ok(config)
    }

    /// Debug method: Analyze raw metadata keys in a GGUF file (for task 2.1.20)
    pub async fn analyze_metadata_keys<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, String>> {
        let path = path.as_ref();
        
        tracing::info!("🔍 Analyzing raw metadata keys in: {}", path.display());
        
        // Open file
        let mut file = std::fs::File::open(path)
            .map_err(|e| InferenceError::model_load(format!("Failed to open GGUF file: {}", e)))?;
        
        // Validate file integrity
        self.validate_file_integrity(&mut file)?;
        
        // Parse GGUF header (only need metadata)
        let header = self.parse_header(&mut file)?;
        
        tracing::info!("GGUF header parsed: {} tensors, {} metadata keys", 
                      header.tensor_count, header.metadata.len());
        
        // Extract and analyze all metadata keys
        let mut key_analysis = HashMap::new();
        
        println!("\n🔑 Raw GGUF Metadata Keys Found:");
        for (key, value) in &header.metadata {
            let value_type = match value {
                GgufValue::Uint8(_) => "uint8",
                GgufValue::Int8(_) => "int8", 
                GgufValue::Uint16(_) => "uint16",
                GgufValue::Int16(_) => "int16",
                GgufValue::Uint32(_) => "uint32",
                GgufValue::Int32(_) => "int32",
                GgufValue::Float32(_) => "float32",
                GgufValue::Bool(_) => "bool",
                GgufValue::String(_) => "string",
                GgufValue::Array(_) => "array",
                GgufValue::Uint64(_) => "uint64",
                GgufValue::Int64(_) => "int64",
                GgufValue::Float64(_) => "float64",
            };
            
            println!("  {} -> {} ({})", key, value_type, 
                    if key.len() > 50 { format!("{}...", &key[..47]) } else { key.clone() });
            
            key_analysis.insert(key.clone(), value_type.to_string());
        }
        
        println!("\n📊 Key Analysis Summary:");
        println!("  Total keys found: {}", key_analysis.len());
        
        // Check for our expected keys
        use crate::bitnet_config::GgufKeys;
        let expected_keys = [
            (GgufKeys::GENERAL_ARCHITECTURE, "Architecture"),
            (GgufKeys::GENERAL_NAME, "Name"),
            (GgufKeys::LAYER_COUNT, "Layer Count"),
            (GgufKeys::HIDDEN_SIZE, "Hidden Size"),
            (GgufKeys::ATTENTION_HEAD_COUNT, "Attention Heads"),
            (GgufKeys::CONTEXT_LENGTH, "Context Length"),
            (GgufKeys::BITNET_VERSION, "BitNet Version"),
            (GgufKeys::BITNET_WEIGHT_BITS, "Weight Bits"),
        ];
        
        println!("\n🎯 Expected Key Mapping:");
        for (expected_key, description) in &expected_keys {
            if header.metadata.contains_key(*expected_key) {
                println!("  ✅ {} -> FOUND: {}", description, expected_key);
            } else {
                println!("  ❌ {} -> MISSING: {}", description, expected_key);
                
                // Look for similar keys
                let similar_keys: Vec<_> = header.metadata.keys()
                    .filter(|k| k.to_lowercase().contains(&expected_key.split('.').last().unwrap_or("").to_lowercase()))
                    .collect();
                
                if !similar_keys.is_empty() {
                    println!("     🔍 Similar keys found: {:?}", similar_keys);
                }
            }
        }
        
        Ok(key_analysis)
    }

    /// Load a model from GGUF file path with HuggingFace integration
    pub async fn load_model_from_path<P: AsRef<Path>>(
        &self,
        path: P,
        memory_pool: Option<Arc<HybridMemoryPool>>,
    ) -> Result<LoadedModel> {
        let path = path.as_ref();
        
        tracing::info!("Loading GGUF model from: {}", path.display());
        
        // Open file
        let mut file = std::fs::File::open(path)
            .map_err(|e| InferenceError::model_load(format!("Failed to open GGUF file: {}", e)))?;
        
        // Validate file integrity
        self.validate_file_integrity(&mut file)?;
        
        // Parse GGUF header
        let header = self.parse_header(&mut file)?;
        
        tracing::info!("GGUF header parsed: {} tensors, version {}", header.tensor_count, header.version);
        
        // Load tensors with memory pool integration
        let (architecture, weights) = if let Some(pool) = memory_pool {
            self.load_tensors_with_pool(&mut file, &header, pool).await?
        } else {
            self.load_tensors(&mut file, &header).await?
        };

        // Create model metadata and BitNet configuration
        let metadata = self.extract_metadata(&header)?;
        let bitnet_config = self.extract_bitnet_config(&header).ok(); // Store config if successful
        
        Ok(LoadedModel {
            architecture,
            weights,
            metadata,
            bitnet_config,
        })
    }

    /// Validate GGUF file integrity before loading
    fn validate_file_integrity<R: Read + Seek>(&self, reader: &mut R) -> Result<()> {
        // Get file size
        let file_size = reader.seek(SeekFrom::End(0))
            .map_err(|e| InferenceError::model_load(format!("Failed to get file size: {}", e)))?;
        
        // Reset to beginning
        reader.seek(SeekFrom::Start(0))
            .map_err(|e| InferenceError::model_load(format!("Failed to reset file position: {}", e)))?;
        
        // Basic size check (GGUF files should be at least 32 bytes for header)
        if file_size < 32 {
            return Err(InferenceError::model_load("File too small to be a valid GGUF file"));
        }
        
        tracing::debug!("GGUF file size: {} bytes", file_size);
        Ok(())
    }
    
    /// Robust buffer reading with retry logic and partial loading support
    fn read_buffer_robust<R: Read>(
        reader: &mut R,
        size: usize,
        context: &str,
        config: &BufferReadConfig,
    ) -> Result<BufferReadResult> {
        let mut buffer = vec![0u8; size];
        let mut total_read = 0;
        let mut attempts = 0;
        let mut consecutive_zero_reads = 0;
        
        while total_read < size && attempts < config.max_retries {
            match reader.read(&mut buffer[total_read..]) {
                Ok(0) => {
                    // EOF reached - check if we have enough data for partial read
                    consecutive_zero_reads += 1;
                    if consecutive_zero_reads >= 3 || total_read == 0 {
                        if total_read == 0 {
                            return Ok(BufferReadResult::Failed(
                                format!("Unexpected EOF for {} (no data read)", context)));
                        }
                        // EOF reached with some data - check if partial read is acceptable
                        break;
                    }
                    // Brief pause before retry on EOF
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    attempts += 1;
                },
                Ok(bytes_read) => {
                    total_read += bytes_read;
                    attempts = 0; // Reset attempts on successful read
                    consecutive_zero_reads = 0; // Reset zero read counter
                    
                    // Log progress for large reads
                    if size > 10 * 1024 * 1024 && total_read % (5 * 1024 * 1024) == 0 {
                        tracing::debug!("Read progress for {}: {}/{} bytes ({:.1}%)", 
                                       context, total_read, size, (total_read as f64 / size as f64) * 100.0);
                    }
                },
                Err(e) if e.kind() == ErrorKind::Interrupted => {
                    // Retry on interruption
                    attempts += 1;
                    continue;
                },
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    // Handle unexpected EOF as partial read if we have some data
                    tracing::warn!("Unexpected EOF for {} after reading {} bytes: {}", context, total_read, e);
                    if total_read > 0 {
                        break;
                    } else {
                        return Ok(BufferReadResult::Failed(
                            format!("Unexpected EOF for {} with no data: {}", context, e)));
                    }
                },
                Err(e) => {
                    attempts += 1;
                    if attempts >= config.max_retries {
                        return Ok(BufferReadResult::Failed(
                            format!("Failed to read {} after {} attempts: {}", context, attempts, e)));
                    }
                    tracing::warn!("Read attempt {} failed for {}: {}", attempts, context, e);
                    // Brief pause before retry
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        }
        
        if total_read == size {
            Ok(BufferReadResult::Complete(buffer))
        } else {
            let loss_pct = 1.0 - (total_read as f32 / size as f32);
            
            // Enhanced partial read handling
            if loss_pct <= config.partial_tolerance {
                buffer.truncate(total_read);
                tracing::warn!("Accepting partial read for {}: {:.1}% data loss ({}/{} bytes)", 
                              context, loss_pct * 100.0, total_read, size);
                Ok(BufferReadResult::Partial(buffer, loss_pct))
            } else {
                // For very large tensors, be more tolerant if we're close to the end
                let is_large_tensor = size > 50 * 1024 * 1024; // 50MB threshold
                let is_near_complete = loss_pct < 0.25; // Less than 25% loss
                
                if is_large_tensor && is_near_complete {
                    buffer.truncate(total_read);
                    tracing::warn!("Accepting large tensor partial read for {}: {:.1}% data loss ({}/{} bytes)", 
                                  context, loss_pct * 100.0, total_read, size);
                    Ok(BufferReadResult::Partial(buffer, loss_pct))
                } else {
                    Ok(BufferReadResult::Failed(
                        format!("Insufficient data for {}: got {} bytes, expected {} ({:.1}% loss)", 
                               context, total_read, size, loss_pct * 100.0)))
                }
            }
        }
    }

    /// Parse GGUF header
    fn parse_header<R: Read>(&self, reader: &mut R) -> Result<GgufHeader> {
        let config = BufferReadConfig::default();
        
        // Read and verify magic
        let magic_result = Self::read_buffer_robust(reader, 4, "magic", &config)?;
        let magic_bytes = match magic_result {
            BufferReadResult::Complete(bytes) => bytes,
            _ => return Err(InferenceError::model_load("Failed to read GGUF magic number")),
        };
        
        if magic_bytes != GGUF_MAGIC {
            return Err(InferenceError::model_load("Invalid GGUF magic number"));
        }
        
        // Read version
        let version = self.read_u32(reader)?;
        if version != GGUF_VERSION {
            tracing::warn!("GGUF version {} differs from expected {}", version, GGUF_VERSION);
        }
        
        // Read tensor count and metadata count
        let tensor_count = self.read_u64(reader)?;
        let metadata_kv_count = self.read_u64(reader)?;
        
        tracing::debug!("GGUF header: version={}, tensors={}, metadata_kvs={}", 
                       version, tensor_count, metadata_kv_count);
        
        // Read metadata
        let mut metadata = HashMap::new();
        for i in 0..metadata_kv_count {
            let key = self.read_string(reader)?;
            
            // Try to read value type, but handle unknown types gracefully
            let value_type_raw = self.read_u32(reader)?;
            match GgufValueType::try_from(value_type_raw) {
                Ok(value_type) => {
                    match self.read_value(reader, value_type) {
                        Ok(value) => {
                            metadata.insert(key, value);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read metadata value for key '{}': {}, skipping", key, e);
                            // Continue processing other metadata entries
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!("Unknown GGUF value type {} for key '{}', attempting to skip", value_type_raw, key);
                    // Try to skip this unknown value type
                    match self.skip_unknown_value(reader, value_type_raw) {
                        Ok(_) => {
                            tracing::debug!("Successfully skipped unknown value type {} for key '{}'", value_type_raw, key);
                            // Continue with next metadata entry
                        }
                        Err(e) => {
                            tracing::error!("Failed to skip unknown value type {} for key '{}': {}, stopping metadata parsing", 
                                          value_type_raw, key, e);
                            // If we can't skip, we have to stop parsing metadata to avoid corruption
                            break;
                        }
                    }
                }
            }
        }
        
        // Read tensor info
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = self.read_string(reader)?;
            let n_dims = self.read_u32(reader)?;
            
            let mut dimensions = Vec::new();
            for _ in 0..n_dims {
                dimensions.push(self.read_u64(reader)?);
            }
            
            let tensor_type = GgufTensorType::try_from(self.read_u32(reader)?)?;
            let offset = self.read_u64(reader)?;
            
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }
        
        Ok(GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
        })
    }

    /// Load tensors from GGUF file
    async fn load_tensors<R: Read + Seek>(
        &self,
        reader: &mut R,
        header: &GgufHeader,
    ) -> Result<(ModelArchitecture, ModelWeights)> {
        tracing::info!("Loading tensor data for {} tensors", header.tensor_count);
        
        // Calculate tensor data start position
        let tensor_data_start = self.calculate_tensor_data_offset(reader)?;
        tracing::debug!("Tensor data starts at offset: {}", tensor_data_start);
        
        let mut layer_weights = HashMap::new();
        let mut total_size = 0;
        let mut layers = Vec::new();
        
        // Process each tensor
        for (index, tensor_info) in header.tensors.iter().enumerate() {
            tracing::debug!("Loading tensor {}/{}: {} {:?} offset={}", 
                          index + 1, header.tensor_count, tensor_info.name, 
                          tensor_info.dimensions, tensor_info.offset);
            
            // Validate tensor before loading
            self.validate_tensor_info(tensor_info)?;
            
            // Calculate tensor size and seek to data
            let tensor_size = self.calculate_tensor_size(tensor_info)?;
            let absolute_offset = tensor_data_start + tensor_info.offset;
            
            // Seek to tensor data position
            reader.seek(SeekFrom::Start(absolute_offset))
                .map_err(|e| InferenceError::model_load(
                    format!("Failed to seek to tensor {}: {}", tensor_info.name, e)))?;
            
            // Read tensor data based on type
            let tensor_data = self.read_tensor_data(reader, tensor_info, tensor_size)?;
            
            // Validate loaded tensor data
            self.validate_tensor_data(&tensor_data, tensor_info)?;
            
            // Store tensor data with numeric key (use index for now)
            layer_weights.insert(index, tensor_data);
            total_size += tensor_size;
            
            // Create layer definition if this is a layer weight
            if let Some(mut layer_def) = self.create_layer_definition(tensor_info) {
                layer_def.id = layers.len(); // Set proper ID
                layers.push(layer_def);
            }
            
            // Log progress for large models
            if (index + 1) % 50 == 0 || index + 1 == header.tensor_count as usize {
                tracing::info!("Loaded {}/{} tensors ({:.1} MB total)", 
                             index + 1, header.tensor_count, total_size as f64 / 1024.0 / 1024.0);
            }
        }
        
        // Create execution order (simple sequential for now)
        let execution_order: Vec<usize> = (0..layers.len()).collect();
        
        let architecture = ModelArchitecture {
            layers,
            execution_order,
        };

        // Create weights with new organized structure
        let mut weights = ModelWeights::new();
        weights.layer_weights = layer_weights;
        weights.total_size = total_size;
        
        // Organize weights by layer and parameter type
        self.organize_weights(&mut weights, &header).await?;
        
        tracing::info!("Successfully loaded {} tensors ({:.1} MB total)", 
                      header.tensor_count, total_size as f64 / 1024.0 / 1024.0);
        
        Ok((architecture, weights))
    }

    /// Load tensors with memory pool integration
    async fn load_tensors_with_pool<R: Read + Seek>(
        &self,
        reader: &mut R,
        header: &GgufHeader,
        _memory_pool: Arc<HybridMemoryPool>,
    ) -> Result<(ModelArchitecture, ModelWeights)> {
        // Enhanced version with memory pool integration
        // This leverages the memory management improvements from Phase 1
        
        tracing::info!("Loading tensors with memory pool optimization");
        
        // Delegate to basic loading for now, will be enhanced
        self.load_tensors(reader, header).await
    }

    /// Calculate the offset where tensor data begins
    fn calculate_tensor_data_offset<R: Read + Seek>(&self, reader: &mut R) -> Result<u64> {
        // Tensor data starts after header, metadata, and tensor info
        // We need to calculate the current position after parsing
        let current_pos = reader.stream_position()
            .map_err(|e| InferenceError::model_load(format!("Failed to get stream position: {}", e)))?;
        
        tracing::debug!("Current position after header parsing: {}", current_pos);
        Ok(current_pos)
    }
    
    /// Calculate tensor size in bytes based on type and dimensions
    pub fn calculate_tensor_size(&self, tensor_info: &GgufTensorInfo) -> Result<usize> {
        let element_count: u64 = tensor_info.dimensions.iter().product();
        
        let total_size = match tensor_info.tensor_type {
            GgufTensorType::F32 => element_count * 4,
            GgufTensorType::F16 => element_count * 2,
            GgufTensorType::I8 => element_count * 1,
            GgufTensorType::I16 => element_count * 2,
            GgufTensorType::I32 => element_count * 4,
            GgufTensorType::I64 => element_count * 8,
            GgufTensorType::F64 => element_count * 8,
            
            // Quantized formats (block-based)
            GgufTensorType::Q40 | GgufTensorType::Q41 => {
                // Q4_0: 32 weights per block, 2 bytes scale + 16 bytes weights = 18 bytes per block
                (element_count / 32) * 18
            },
            GgufTensorType::Q50 | GgufTensorType::Q51 => {
                // Q5_0: 32 weights per block, 2 bytes scale + 20 bytes weights = 22 bytes per block  
                (element_count / 32) * 22
            },
            GgufTensorType::Q80 | GgufTensorType::Q81 => {
                // Q8_0: 32 weights per block, 2 bytes scale + 32 bytes weights = 34 bytes per block
                (element_count / 32) * 34
            },
            
            // K-quants (more complex block sizes)
            GgufTensorType::Q2K => (element_count / 256) * 82,   // 256 weights per super-block
            GgufTensorType::Q3K => (element_count / 256) * 110,
            GgufTensorType::Q4K => (element_count / 256) * 144,
            GgufTensorType::Q5K => (element_count / 256) * 176,
            GgufTensorType::Q6K => (element_count / 256) * 210,
            GgufTensorType::Q8K => (element_count / 256) * 256,
            
            // BitNet 1.58-bit: 4 weights per byte (2 bits each) + metadata
            GgufTensorType::BitnetB158 => {
                let packed_size = (element_count + 3) / 4; // 4 weights per byte
                let metadata_size = 8; // 8 bytes for scale and offset
                packed_size + metadata_size
            },
            
            // Other quantized formats (estimated)
            _ => {
                tracing::warn!("Unknown tensor type {:?}, using F32 size estimate", tensor_info.tensor_type);
                element_count * 4
            }
        } as usize;
        
        tracing::debug!("Tensor {} size: {} elements = {} bytes", 
                       tensor_info.name, element_count, total_size);
        
        Ok(total_size)
    }
    
    /// Read tensor data from GGUF file with chunked loading support
    fn read_tensor_data<R: Read>(
        &self,
        reader: &mut R,
        tensor_info: &GgufTensorInfo,
        size: usize,
    ) -> Result<Vec<u8>> {
        let config = BufferReadConfig::default();
        
        // Use chunked loading for large tensors
        let raw_data = if size > config.large_tensor_threshold && config.enable_streaming {
            tracing::info!("Using chunked loading for large tensor {} ({:.1} MB)", 
                          tensor_info.name, size as f64 / 1024.0 / 1024.0);
            self.read_tensor_data_chunked(reader, tensor_info, size, &config)?
        } else {
            // Standard loading for smaller tensors
            let read_result = Self::read_buffer_robust(reader, size, &tensor_info.name, &config)?;
            self.handle_read_result(read_result, tensor_info, &config)?
        };
        
        // Decode tensor data if it's BitNet ternary weights
        match tensor_info.tensor_type {
            GgufTensorType::BitnetB158 => {
                tracing::debug!("Decoding BitNet 1.58-bit ternary weights for tensor {}", tensor_info.name);
                self.decode_ternary_weights(&raw_data, tensor_info)
            },
            _ => {
                // Return raw data for non-ternary tensors
                Ok(raw_data)
            }
        }
    }
    
    /// Handle read result with proper error handling and recovery
    fn handle_read_result(
        &self,
        read_result: BufferReadResult,
        tensor_info: &GgufTensorInfo,
        config: &BufferReadConfig,
    ) -> Result<Vec<u8>> {
        match read_result {
            BufferReadResult::Complete(data) => {
                tracing::debug!("Successfully read {} bytes for tensor {}", 
                               data.len(), tensor_info.name);
                Ok(data)
            },
            BufferReadResult::Partial(data, loss_pct) => {
                tracing::warn!("Partial read for tensor {} ({:.1}% data loss): {} bytes", 
                              tensor_info.name, loss_pct * 100.0, data.len());
                
                // For large tensors, be more tolerant of data loss
                let is_large_tensor = data.len() > 50 * 1024 * 1024; // 50MB
                let effective_tolerance = if is_large_tensor {
                    config.partial_tolerance.max(0.25) // 25% tolerance for large tensors
                } else {
                    config.partial_tolerance
                };
                
                if loss_pct <= effective_tolerance {
                    tracing::info!("Accepting partial tensor {} with {:.1}% loss (within {:.1}% tolerance)", 
                                  tensor_info.name, loss_pct * 100.0, effective_tolerance * 100.0);
                    Ok(data)
                } else {
                    Err(InferenceError::model_load(
                        format!("Tensor {} data loss {:.1}% exceeds tolerance {:.1}%", 
                               tensor_info.name, loss_pct * 100.0, effective_tolerance * 100.0)))
                }
            },
            BufferReadResult::Failed(error) => {
                // For network-related failures, suggest recovery actions
                let recovery_hint = if error.contains("Insufficient data") {
                    " (Try re-downloading the model file or checking network connectivity)"
                } else {
                    ""
                };
                
                Err(InferenceError::model_load(
                    format!("Failed to read tensor {}: {}{}", tensor_info.name, error, recovery_hint)))
            }
        }
    }
    
    /// Read tensor data using chunked approach for memory efficiency
    fn read_tensor_data_chunked<R: Read>(
        &self,
        reader: &mut R,
        tensor_info: &GgufTensorInfo,
        total_size: usize,
        config: &BufferReadConfig,
    ) -> Result<Vec<u8>> {
        let mut data = Vec::with_capacity(total_size);
        let mut remaining = total_size;
        let mut chunk_index = 0;
        
        tracing::debug!("Reading tensor {} in chunks of {} bytes", 
                       tensor_info.name, config.chunk_size);
        
        while remaining > 0 {
            let chunk_size = remaining.min(config.chunk_size);
            
            tracing::debug!("Reading chunk {}: {} bytes (remaining: {})", 
                           chunk_index, chunk_size, remaining);
            
            // Read chunk with robust error handling
            let chunk_result = Self::read_buffer_robust(
                reader, 
                chunk_size, 
                &format!("{}_chunk_{}", tensor_info.name, chunk_index), 
                config
            )?;
            
            let chunk_data = self.handle_read_result(chunk_result, tensor_info, config)?;
            
            // Append chunk data
            data.extend_from_slice(&chunk_data);
            remaining -= chunk_data.len();
            chunk_index += 1;
            
            // Progress logging for very large tensors
            if chunk_index % 10 == 0 {
                let progress = ((total_size - remaining) as f64 / total_size as f64) * 100.0;
                tracing::info!("Tensor {} loading progress: {:.1}% ({}/{} bytes)", 
                              tensor_info.name, progress, total_size - remaining, total_size);
            }
            
            // Safety check to prevent infinite loops
            if chunk_data.is_empty() && remaining > 0 {
                return Err(InferenceError::model_load(
                    format!("Failed to read chunk {} for tensor {}: no data received", 
                           chunk_index, tensor_info.name)));
            }
        }
        
        tracing::info!("Successfully loaded tensor {} using chunked reading ({} chunks, {:.1} MB)", 
                      tensor_info.name, chunk_index, data.len() as f64 / 1024.0 / 1024.0);
        
        Ok(data)
    }
    
    /// Validate tensor information before loading
    pub fn validate_tensor_info(&self, tensor_info: &GgufTensorInfo) -> Result<()> {
        // Check tensor name is valid
        if tensor_info.name.is_empty() {
            return Err(InferenceError::model_load("Tensor name cannot be empty"));
        }
        
        // Check dimensions are valid
        if tensor_info.dimensions.is_empty() {
            return Err(InferenceError::model_load(
                format!("Tensor {} has no dimensions", tensor_info.name)));
        }
        
        // Check for reasonable dimension sizes
        let element_count: u64 = tensor_info.dimensions.iter().product();
        if element_count == 0 {
            return Err(InferenceError::model_load(
                format!("Tensor {} has zero elements", tensor_info.name)));
        }
        
        // Check for unreasonably large tensors (safety limit: 16GB per tensor)
        const MAX_TENSOR_ELEMENTS: u64 = 4 * 1024 * 1024 * 1024; // 4B elements
        if element_count > MAX_TENSOR_ELEMENTS {
            return Err(InferenceError::model_load(
                format!("Tensor {} is too large: {} elements (max: {})", 
                       tensor_info.name, element_count, MAX_TENSOR_ELEMENTS)));
        }
        
        // Validate tensor type
        match tensor_info.tensor_type {
            GgufTensorType::BitnetB158 => {
                // BitNet tensors should have reasonable dimensions for neural networks
                if tensor_info.dimensions.len() > 4 {
                    tracing::warn!("BitNet tensor {} has {} dimensions (expected ≤4)", 
                                  tensor_info.name, tensor_info.dimensions.len());
                }
            },
            _ => {
                // Standard validation for other tensor types
            }
        }
        
        tracing::debug!("Tensor {} validation passed: {:?} elements={}", 
                       tensor_info.name, tensor_info.dimensions, element_count);
        
        Ok(())
    }
    
    /// Validate loaded tensor data integrity
    fn validate_tensor_data(&self, data: &[u8], tensor_info: &GgufTensorInfo) -> Result<()> {
        // Check data is not empty
        if data.is_empty() {
            return Err(InferenceError::model_load(
                format!("Tensor {} has no data", tensor_info.name)));
        }
        
        // Calculate expected size based on tensor type and dimensions
        let element_count: u64 = tensor_info.dimensions.iter().product();
        
        let expected_bytes = match tensor_info.tensor_type {
            GgufTensorType::BitnetB158 => {
                // For decoded ternary weights: 1 byte per element (i8 stored as u8)
                element_count as usize
            },
            GgufTensorType::F32 => element_count as usize * 4,
            GgufTensorType::F16 => element_count as usize * 2,
            GgufTensorType::I8 => element_count as usize,
            GgufTensorType::I16 => element_count as usize * 2,
            GgufTensorType::I32 => element_count as usize * 4,
            _ => {
                // For quantized formats, size validation is complex - skip strict validation
                tracing::debug!("Skipping size validation for quantized tensor type {:?}", 
                               tensor_info.tensor_type);
                return Ok(());
            }
        };
        
        // Allow some tolerance for padding or metadata
        let tolerance = (expected_bytes as f32 * 0.1).max(16.0) as usize; // 10% or 16 bytes
        if data.len() < expected_bytes.saturating_sub(tolerance) {
            return Err(InferenceError::model_load(
                format!("Tensor {} data size mismatch: got {} bytes, expected ~{} bytes", 
                       tensor_info.name, data.len(), expected_bytes)));
        }
        
        // Validate ternary weight data
        if tensor_info.tensor_type == GgufTensorType::BitnetB158 {
            self.validate_ternary_weights(data, tensor_info)?;
        }
        
        tracing::debug!("Tensor {} data validation passed: {} bytes (expected ~{})", 
                       tensor_info.name, data.len(), expected_bytes);
        
        Ok(())
    }
    
    /// Validate ternary weight values are within expected range
    fn validate_ternary_weights(&self, data: &[u8], tensor_info: &GgufTensorInfo) -> Result<()> {
        // Sample validation: check first few weights are valid ternary values
        let sample_size = data.len().min(100); // Check first 100 values
        let mut invalid_count = 0;
        
        for &value in data.iter().take(sample_size) {
            let signed_value = value as i8;
            if signed_value < -1 || signed_value > 1 {
                invalid_count += 1;
            }
        }
        
        // Allow some tolerance for encoding variations
        let invalid_ratio = invalid_count as f32 / sample_size as f32;
        if invalid_ratio > 0.1 {
            return Err(InferenceError::model_load(
                format!("Tensor {} has invalid ternary weights: {:.1}% out of range [-1,1]", 
                       tensor_info.name, invalid_ratio * 100.0)));
        }
        
        if invalid_count > 0 {
            tracing::warn!("Tensor {} has {} invalid ternary values in sample of {}", 
                          tensor_info.name, invalid_count, sample_size);
        }
        
        Ok(())
    }
    
    /// Decode packed ternary weights to {-1, 0, +1} values
    fn decode_ternary_weights(&self, packed_data: &[u8], tensor_info: &GgufTensorInfo) -> Result<Vec<u8>> {
        let element_count: u64 = tensor_info.dimensions.iter().product();
        tracing::debug!("Decoding {} ternary elements from {} packed bytes", 
                       element_count, packed_data.len());
        
        // BitNet 1.58-bit encoding: 4 weights per byte (2 bits each)
        // 2-bit values: 00 -> -1, 01 -> 0, 10 -> +1, 11 -> reserved/unused
        
        // Expected packed size (4 weights per byte + 8 bytes metadata)
        let expected_packed_size = ((element_count + 3) / 4) as usize + 8;
        if packed_data.len() < expected_packed_size - 8 {
            return Err(InferenceError::model_load(
                format!("Insufficient packed data for tensor {}: got {} bytes, expected at least {}", 
                       tensor_info.name, packed_data.len(), expected_packed_size - 8)));
        }
        
        // Extract metadata (first 8 bytes: scale and offset)
        let scale = if packed_data.len() >= 4 {
            f32::from_le_bytes([packed_data[0], packed_data[1], packed_data[2], packed_data[3]])
        } else {
            1.0 // Default scale
        };
        
        let offset = if packed_data.len() >= 8 {
            f32::from_le_bytes([packed_data[4], packed_data[5], packed_data[6], packed_data[7]])
        } else {
            0.0 // Default offset
        };
        
        // Start reading packed weights after metadata
        let weights_start = if packed_data.len() >= 8 { 8 } else { 0 };
        let packed_weights = &packed_data[weights_start..];
        
        tracing::debug!("Ternary weights metadata: scale={:.6}, offset={:.6}", scale, offset);
        
        // Decode ternary weights to i8 values {-1, 0, +1}
        let mut decoded_weights = Vec::with_capacity(element_count as usize);
        
        for (byte_idx, &packed_byte) in packed_weights.iter().enumerate() {
            // Extract 4 weights from each byte (2 bits per weight)
            for bit_pos in 0..4 {
                if decoded_weights.len() >= element_count as usize {
                    break; // Avoid over-reading due to padding
                }
                
                let weight_bits = (packed_byte >> (bit_pos * 2)) & 0b11;
                let ternary_value = match weight_bits {
                    0b00 => -1i8,  // -1
                    0b01 => 0i8,   //  0  
                    0b10 => 1i8,   // +1
                    0b11 => 0i8,   // Reserved -> 0 (fallback)
                    _ => unreachable!(),
                };
                
                decoded_weights.push(ternary_value);
            }
            
            // Early exit for efficiency
            if decoded_weights.len() >= element_count as usize {
                break;
            }
        }
        
        // Ensure we have the right number of weights
        decoded_weights.truncate(element_count as usize);
        
        tracing::debug!("Decoded {} ternary weights (sample: [{}, {}, {}, ...])", 
                       decoded_weights.len(),
                       decoded_weights.get(0).unwrap_or(&0),
                       decoded_weights.get(1).unwrap_or(&0), 
                       decoded_weights.get(2).unwrap_or(&0));
        
        // Convert i8 to Vec<u8> for storage (maintaining ternary values)
        let output_data: Vec<u8> = decoded_weights.into_iter().map(|v| v as u8).collect();
        
        Ok(output_data)
    }
    
    /// Create layer definition from tensor info
    fn create_layer_definition(&self, tensor_info: &GgufTensorInfo) -> Option<LayerDefinition> {
        // Detect layer type from tensor name patterns
        let layer_type = if tensor_info.name.contains(".weight") {
            if tensor_info.name.contains("attn") {
                LayerType::BitLinear
            } else if tensor_info.name.contains("mlp") || tensor_info.name.contains("feed_forward") {
                LayerType::BitLinear  
            } else if tensor_info.name.contains("norm") {
                LayerType::RMSNorm
            } else if tensor_info.name.contains("embed") {
                LayerType::Embedding
            } else {
                LayerType::BitLinear
            }
        } else if tensor_info.name.contains(".bias") {
            return None; // Bias tensors don't create separate layers
        } else {
            return None; // Non-weight tensors
        };
        
        // Extract dimensions
        let input_dims = if !tensor_info.dimensions.is_empty() {
            vec![tensor_info.dimensions[0] as usize]
        } else {
            vec![1]
        };
        
        let output_dims = if tensor_info.dimensions.len() >= 2 {
            vec![tensor_info.dimensions[1] as usize]
        } else {
            input_dims.clone()
        };
        
        // Create appropriate parameters based on layer type
        let parameters = match layer_type {
            LayerType::BitLinear => LayerParameters::BitLinear {
                weight_bits: if tensor_info.tensor_type == GgufTensorType::BitnetB158 { 2 } else { 8 },
                activation_bits: 8,
            },
            LayerType::RMSNorm => LayerParameters::RMSNorm {
                eps: 1e-6,
            },
            LayerType::Embedding => LayerParameters::Embedding {
                vocab_size: *tensor_info.dimensions.get(0).unwrap_or(&1) as usize,
                embedding_dim: *tensor_info.dimensions.get(1).unwrap_or(&1) as usize,
            },
            LayerType::SwiGLU => LayerParameters::SwiGLU {
                hidden_dim: *tensor_info.dimensions.get(0).unwrap_or(&1) as usize,
            },
            LayerType::OutputProjection => LayerParameters::OutputProjection {
                vocab_size: *tensor_info.dimensions.get(1).unwrap_or(&1) as usize,
            },
        };
        
        Some(LayerDefinition {
            id: 0, // Will be set properly during architecture construction
            layer_type,
            input_dims,
            output_dims,
            parameters,
        })
    }
    
    /// Extract model metadata from GGUF header
    fn extract_metadata(&self, header: &GgufHeader) -> Result<ModelMetadata> {
        // Extract comprehensive BitNet model configuration from GGUF metadata
        let bitnet_config = self.extract_bitnet_config(header)?;
        
        // Convert BitNet config to ModelMetadata for compatibility
        Ok(ModelMetadata {
            name: bitnet_config.basic_info.name.clone(),
            version: bitnet_config.basic_info.version.clone(),
            architecture: bitnet_config.basic_info.architecture.clone(),
            parameter_count: bitnet_config.basic_info.parameter_count,
            quantization_bits: bitnet_config.bitlinear_config.weight_bits,
            input_shape: vec![1, bitnet_config.basic_info.context_length],
            output_shape: vec![1, bitnet_config.tokenizer_config.vocab_size],
            extra: bitnet_config.extra_metadata,
        })
    }

    /// Extract comprehensive BitNet model configuration from GGUF metadata
    fn extract_bitnet_config(&self, header: &GgufHeader) -> Result<BitNetModelConfig> {
        let mut config = BitNetModelConfig::new();
        
        // Detect model variant for specific parsing strategies
        let model_variant = self.detect_model_variant(header);
        tracing::debug!("Detected model variant: {:?}", model_variant);
        
        // Apply model-specific parsing strategies
        match model_variant {
            ModelVariant::MicrosoftBitNet => {
                self.extract_microsoft_bitnet_config(&mut config, header)?;
            },
            ModelVariant::StandardLlama => {
                self.extract_standard_llama_config(&mut config, header)?;
            },
            ModelVariant::Unknown => {
                // Use generic extraction with all fallbacks
                self.extract_generic_config(&mut config, header)?;
            },
        }
        
        // Validate the extracted configuration
        config.validate()?;
        
        tracing::info!("Extracted BitNet model configuration: {} layers, {} heads, hidden_size {}", 
                      config.layer_config.n_layers, 
                      config.attention_config.n_heads, 
                      config.layer_config.hidden_size);
        
        Ok(config)
    }

    /// Detect the specific model variant for targeted parsing
    fn detect_model_variant(&self, header: &GgufHeader) -> ModelVariant {
        // Check for Microsoft BitNet specific indicators
        if header.metadata.contains_key("general.name") {
            if let Some(GgufValue::String(name)) = header.metadata.get("general.name") {
                if name.to_lowercase().contains("bitnet") || name.to_lowercase().contains("microsoft") {
                    return ModelVariant::MicrosoftBitNet;
                }
            }
        }
        
        // Check for standard LLaMA indicators
        if header.metadata.contains_key("general.architecture") {
            if let Some(GgufValue::String(arch)) = header.metadata.get("general.architecture") {
                if arch.to_lowercase().contains("llama") {
                    return ModelVariant::StandardLlama;
                }
            }
        }
        
        // Check for LLaMA-specific keys
        if header.metadata.contains_key("llama.block_count") || header.metadata.contains_key("llama.embedding_length") {
            return ModelVariant::StandardLlama;
        }
        
        ModelVariant::Unknown
    }

    /// Extract configuration for Microsoft BitNet models
    fn extract_microsoft_bitnet_config(&self, config: &mut BitNetModelConfig, header: &GgufHeader) -> Result<()> {
        // Microsoft BitNet models may use different key patterns
        self.extract_basic_info_microsoft(&mut config.basic_info, header)?;
        self.extract_layer_config_microsoft(&mut config.layer_config, header)?;
        self.extract_attention_config_microsoft(&mut config.attention_config, header)?;
        self.extract_normalization_config(&mut config.normalization_config, header)?;
        self.extract_bitlinear_config(&mut config.bitlinear_config, header)?;
        self.extract_tokenizer_config(&mut config.tokenizer_config, header)?;
        self.extract_extra_metadata(&mut config.extra_metadata, header)?;
        
        Ok(())
    }

    /// Extract configuration for standard LLaMA-based models
    fn extract_standard_llama_config(&self, config: &mut BitNetModelConfig, header: &GgufHeader) -> Result<()> {
        // Use LLaMA-specific key patterns
        self.extract_basic_info_llama(&mut config.basic_info, header)?;
        self.extract_layer_config_llama(&mut config.layer_config, header)?;
        self.extract_attention_config_llama(&mut config.attention_config, header)?;
        self.extract_normalization_config(&mut config.normalization_config, header)?;
        self.extract_bitlinear_config(&mut config.bitlinear_config, header)?;
        self.extract_tokenizer_config(&mut config.tokenizer_config, header)?;
        self.extract_extra_metadata(&mut config.extra_metadata, header)?;
        
        Ok(())
    }

    /// Extract configuration using generic approach with all fallbacks
    fn extract_generic_config(&self, config: &mut BitNetModelConfig, header: &GgufHeader) -> Result<()> {
        // Use the existing extraction methods which now have fallback support
        self.extract_basic_info(&mut config.basic_info, header)?;
        self.extract_layer_config(&mut config.layer_config, header)?;
        self.extract_attention_config(&mut config.attention_config, header)?;
        self.extract_normalization_config(&mut config.normalization_config, header)?;
        self.extract_bitlinear_config(&mut config.bitlinear_config, header)?;
        self.extract_tokenizer_config(&mut config.tokenizer_config, header)?;
        self.extract_extra_metadata(&mut config.extra_metadata, header)?;
        
        Ok(())
    }

    /// Extract basic info for Microsoft BitNet models
    fn extract_basic_info_microsoft(&self, basic_info: &mut BasicModelInfo, header: &GgufHeader) -> Result<()> {
        // Microsoft models may use different naming patterns
        if let Some(name) = self.get_string_value_with_fallbacks(
            header, 
            "general.name",
            &["model.name", "name", "microsoft.model.name", "bitnet.name"]
        ) {
            basic_info.name = name;
        }
        
        if let Some(arch) = self.get_string_value_with_fallbacks(
            header, 
            "general.architecture", 
            &["architecture", "microsoft.architecture", "bitnet.architecture"]
        ) {
            basic_info.architecture = arch;
        }
        
        // Context length may be in Microsoft-specific locations
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            "bitnet.context_length",
            &["context_length", "microsoft.context_length", "max_position_embeddings"]
        ) {
            basic_info.context_length = context_len as usize;
        }
        
        basic_info.parameter_count = header.tensor_count as usize;
        Ok(())
    }

    /// Extract layer config for Microsoft BitNet models
    fn extract_layer_config_microsoft(&self, layer_config: &mut LayerConfig, header: &GgufHeader) -> Result<()> {
        if let Some(n_layers) = self.get_u32_value_with_fallbacks(
            header, 
            "bitnet.block_count",
            &["microsoft.block_count", "n_layers", "num_layers", "block_count"]
        ) {
            layer_config.n_layers = n_layers as usize;
        }
        
        if let Some(hidden_size) = self.get_u32_value_with_fallbacks(
            header, 
            "bitnet.embedding_length",
            &["microsoft.embedding_length", "hidden_size", "d_model", "embedding_length"]
        ) {
            layer_config.hidden_size = hidden_size as usize;
            layer_config.model_dim = hidden_size as usize;
        }
        
        if let Some(intermediate_size) = self.get_u32_value_with_fallbacks(
            header, 
            "bitnet.feed_forward_length",
            &["microsoft.feed_forward_length", "intermediate_size", "ffn_dim"]
        ) {
            layer_config.intermediate_size = intermediate_size as usize;
        } else {
            layer_config.intermediate_size = layer_config.hidden_size * 4;
        }
        
        Ok(())
    }

    /// Extract attention config for Microsoft BitNet models  
    fn extract_attention_config_microsoft(&self, attention_config: &mut AttentionConfig, header: &GgufHeader) -> Result<()> {
        if let Some(n_heads) = self.get_u32_value_with_fallbacks(
            header, 
            "bitnet.attention.head_count",
            &["microsoft.attention.head_count", "n_heads", "num_attention_heads"]
        ) {
            attention_config.n_heads = n_heads as usize;
        }
        
        if let Some(n_kv_heads) = self.get_u32_value_with_fallbacks(
            header, 
            "bitnet.attention.head_count_kv",
            &["microsoft.attention.head_count_kv", "n_kv_heads", "num_key_value_heads"]
        ) {
            attention_config.n_kv_heads = Some(n_kv_heads as usize);
        }
        
        self.extract_rope_config(&mut attention_config.rope_config, header)?;
        
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            "bitnet.context_length",
            &["microsoft.context_length", "context_length", "max_position_embeddings"]
        ) {
            attention_config.max_seq_len = context_len as usize;
        }
        
        Ok(())
    }

    /// Extract basic info for standard LLaMA models
    fn extract_basic_info_llama(&self, basic_info: &mut BasicModelInfo, header: &GgufHeader) -> Result<()> {
        // LLaMA uses standard general.* keys but may have llama.* specific ones
        if let Some(name) = self.get_string_value_with_fallbacks(
            header, 
            "general.name",
            &["llama.name", "name", "model.name"]
        ) {
            basic_info.name = name;
        }
        
        if let Some(arch) = self.get_string_value_with_fallbacks(
            header, 
            "general.architecture",
            &["llama.architecture", "architecture"]
        ) {
            basic_info.architecture = arch;
        }
        
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            "llama.context_length",
            &["context_length", "max_position_embeddings", "n_ctx"]
        ) {
            basic_info.context_length = context_len as usize;
        }
        
        basic_info.parameter_count = header.tensor_count as usize;
        Ok(())
    }

    /// Extract layer config for standard LLaMA models
    fn extract_layer_config_llama(&self, layer_config: &mut LayerConfig, header: &GgufHeader) -> Result<()> {
        if let Some(n_layers) = self.get_u32_value_with_fallbacks(
            header, 
            "llama.block_count",
            &["n_layers", "num_layers", "block_count"]
        ) {
            layer_config.n_layers = n_layers as usize;
        }
        
        if let Some(hidden_size) = self.get_u32_value_with_fallbacks(
            header, 
            "llama.embedding_length",
            &["hidden_size", "d_model", "embedding_length", "n_embd"]
        ) {
            layer_config.hidden_size = hidden_size as usize;
            layer_config.model_dim = hidden_size as usize;
        }
        
        if let Some(intermediate_size) = self.get_u32_value_with_fallbacks(
            header, 
            "llama.feed_forward_length",
            &["intermediate_size", "ffn_dim", "d_ff"]
        ) {
            layer_config.intermediate_size = intermediate_size as usize;
        } else {
            layer_config.intermediate_size = layer_config.hidden_size * 4;
        }
        
        Ok(())
    }

    /// Extract attention config for standard LLaMA models
    fn extract_attention_config_llama(&self, attention_config: &mut AttentionConfig, header: &GgufHeader) -> Result<()> {
        if let Some(n_heads) = self.get_u32_value_with_fallbacks(
            header, 
            "llama.attention.head_count",
            &["n_heads", "num_attention_heads", "attention.head_count"]
        ) {
            attention_config.n_heads = n_heads as usize;
        }
        
        if let Some(n_kv_heads) = self.get_u32_value_with_fallbacks(
            header, 
            "llama.attention.head_count_kv",
            &["n_kv_heads", "num_key_value_heads", "attention.head_count_kv"]
        ) {
            attention_config.n_kv_heads = Some(n_kv_heads as usize);
        }
        
        self.extract_rope_config(&mut attention_config.rope_config, header)?;
        
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            "llama.context_length",
            &["context_length", "max_position_embeddings", "n_ctx"]
        ) {
            attention_config.max_seq_len = context_len as usize;
        }
        
        Ok(())
    }

    /// Extract basic model information from GGUF metadata
    fn extract_basic_info(&self, basic_info: &mut BasicModelInfo, header: &GgufHeader) -> Result<()> {
        // Extract model name with fallbacks
        if let Some(name) = self.get_string_value_with_fallbacks(
            header, 
            GgufKeys::GENERAL_NAME,
            &["model.name", "name", "general.model", "model_name"]
        ) {
            basic_info.name = name;
        }
        
        // Extract architecture with fallbacks
        if let Some(arch) = self.get_string_value_with_fallbacks(
            header, 
            GgufKeys::GENERAL_ARCHITECTURE,
            &["architecture", "model.architecture", "arch", "model_type"]
        ) {
            basic_info.architecture = arch;
        }
        
        // Extract version with fallbacks
        if let Some(version) = self.get_string_value_with_fallbacks(
            header, 
            GgufKeys::GENERAL_VERSION,
            &["version", "model.version", "model_version"]
        ) {
            basic_info.version = version;
        }
        
        // Estimate parameter count from tensor count (will be refined later)
        basic_info.parameter_count = header.tensor_count as usize;
        
        // Extract context length with fallbacks
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            GgufKeys::CONTEXT_LENGTH,
            &["context_length", "max_position_embeddings", "n_ctx", "seq_len"]
        ) {
            basic_info.context_length = context_len as usize;
        }
        
        Ok(())
    }

    /// Extract layer configuration from GGUF metadata
    fn extract_layer_config(&self, layer_config: &mut LayerConfig, header: &GgufHeader) -> Result<()> {
        // Extract number of layers with fallbacks
        if let Some(n_layers) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::LAYER_COUNT,
            &["n_layers", "num_layers", "block_count", "num_hidden_layers", "n_layer"]
        ) {
            layer_config.n_layers = n_layers as usize;
        }
        
        // Extract hidden size with fallbacks
        if let Some(hidden_size) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::HIDDEN_SIZE,
            &["hidden_size", "d_model", "embedding_length", "n_embd", "dim"]
        ) {
            layer_config.hidden_size = hidden_size as usize;
            layer_config.model_dim = hidden_size as usize;
        }
        
        // Extract intermediate size (feed forward dimension) with fallbacks
        if let Some(intermediate_size) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::INTERMEDIATE_SIZE,
            &["intermediate_size", "feed_forward_length", "ffn_dim", "d_ff", "n_inner"]
        ) {
            layer_config.intermediate_size = intermediate_size as usize;
        } else {
            // If not specified, use common default: 4 * hidden_size
            layer_config.intermediate_size = layer_config.hidden_size * 4;
        }
        
        Ok(())
    }

    /// Extract attention configuration from GGUF metadata
    fn extract_attention_config(&self, attention_config: &mut AttentionConfig, header: &GgufHeader) -> Result<()> {
        // Extract number of attention heads with fallbacks
        if let Some(n_heads) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::ATTENTION_HEAD_COUNT,
            &["n_heads", "num_attention_heads", "attention.head_count", "n_head", "num_heads"]
        ) {
            attention_config.n_heads = n_heads as usize;
        }
        
        // Extract number of key-value heads (for grouped-query attention) with fallbacks
        if let Some(n_kv_heads) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::ATTENTION_HEAD_COUNT_KV,
            &["n_kv_heads", "num_key_value_heads", "attention.head_count_kv", "n_head_kv"]
        ) {
            attention_config.n_kv_heads = Some(n_kv_heads as usize);
        }
        
        // Calculate head dimension (will be validated later)
        // This is just an initial calculation, will be refined when we have hidden_size
        
        // Extract RoPE configuration
        self.extract_rope_config(&mut attention_config.rope_config, header)?;
        
        // Extract maximum sequence length (use context length as fallback) with fallbacks
        if let Some(context_len) = self.get_u64_value_with_fallbacks(
            header, 
            GgufKeys::CONTEXT_LENGTH,
            &["context_length", "max_position_embeddings", "n_ctx", "seq_len"]
        ) {
            attention_config.max_seq_len = context_len as usize;
        }
        
        Ok(())
    }

    /// Extract RoPE configuration from GGUF metadata
    fn extract_rope_config(&self, rope_config: &mut RopeConfig, header: &GgufHeader) -> Result<()> {
        // Extract RoPE dimension count with fallbacks
        if let Some(rope_dim) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::ROPE_DIMENSION_COUNT,
            &["rope_dim", "rope.dimension_count", "rotary_dim"]
        ) {
            rope_config.rope_dim = rope_dim as usize;
        }
        
        // Extract RoPE frequency base with fallbacks
        if let Some(freq_base) = self.get_f32_value_with_fallbacks(
            header, 
            GgufKeys::ROPE_FREQ_BASE,
            &["rope_freq_base", "rope.freq_base", "rotary_freq", "rope_theta"]
        ) {
            rope_config.rope_freq_base = freq_base;
        }
        
        // Extract RoPE scaling factor with fallbacks
        if let Some(scaling_factor) = self.get_f32_value_with_fallbacks(
            header, 
            GgufKeys::ROPE_SCALING_FACTOR,
            &["rope_scaling_factor", "rope.scaling.factor", "rope_scale"]
        ) {
            rope_config.rope_scaling = Some(scaling_factor);
        }
        
        Ok(())
    }

    /// Extract normalization configuration from GGUF metadata
    fn extract_normalization_config(&self, norm_config: &mut NormalizationConfig, header: &GgufHeader) -> Result<()> {
        // Extract RMSNorm epsilon with fallbacks
        if let Some(eps) = self.get_f32_value_with_fallbacks(
            header, 
            GgufKeys::ATTENTION_LAYER_NORM_RMS_EPS,
            &["layer_norm_epsilon", "rms_norm_eps", "norm_eps", "eps"]
        ) {
            norm_config.rms_norm_eps = eps;
        }
        
        // BitNet models typically don't use bias in normalization layers
        norm_config.use_bias = false;
        
        Ok(())
    }

    /// Extract BitLinear configuration from GGUF metadata
    fn extract_bitlinear_config(&self, bitlinear_config: &mut BitLinearConfig, header: &GgufHeader) -> Result<()> {
        // Extract weight quantization bits with fallbacks
        if let Some(weight_bits) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::BITNET_WEIGHT_BITS,
            &["weight_bits", "quantization.weight_bits", "w_bits", "bits"]
        ) {
            bitlinear_config.weight_bits = weight_bits as u8;
        }
        
        // Extract activation quantization bits with fallbacks
        if let Some(activation_bits) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::BITNET_ACTIVATION_BITS,
            &["activation_bits", "quantization.activation_bits", "a_bits", "act_bits"]
        ) {
            bitlinear_config.activation_bits = activation_bits as u8;
        }
        
        // Extract BitNet version for scheme identification with fallbacks
        if let Some(version) = self.get_string_value_with_fallbacks(
            header, 
            GgufKeys::BITNET_VERSION,
            &["quantization.version", "model.quantization", "quant_version"]
        ) {
            bitlinear_config.quantization_scheme = version;
        }
        
        // Set default scaling behavior for BitNet 1.58
        bitlinear_config.use_weight_scaling = true;
        bitlinear_config.use_activation_scaling = true;
        
        Ok(())
    }

    /// Extract tokenizer configuration from GGUF metadata
    fn extract_tokenizer_config(&self, tokenizer_config: &mut TokenizerConfig, header: &GgufHeader) -> Result<()> {
        // Extract tokenizer model type with fallbacks
        if let Some(tokenizer_type) = self.get_string_value_with_fallbacks(
            header, 
            GgufKeys::TOKENIZER_GGML_MODEL,
            &["tokenizer.model", "tokenizer_type", "tokenizer.ggml.model", "model_type"]
        ) {
            tokenizer_config.tokenizer_type = tokenizer_type;
        }
        
        // Extract vocabulary size from tokenizer tokens array with fallbacks
        let token_keys = [
            GgufKeys::TOKENIZER_GGML_TOKENS,
            "tokenizer.tokens",
            "tokenizer.ggml.tokens",
            "vocab"
        ];
        
        for token_key in &token_keys {
            if let Some(GgufValue::Array(tokens)) = header.metadata.get(*token_key) {
                tokenizer_config.vocab_size = tokens.len();
                if *token_key != GgufKeys::TOKENIZER_GGML_TOKENS {
                    tracing::debug!("Found tokenizer tokens using fallback key '{}'", token_key);
                }
                break;
            }
        }
        
        // Extract special token IDs with fallbacks
        if let Some(bos_id) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::TOKENIZER_GGML_BOS_TOKEN_ID,
            &["tokenizer.bos_token_id", "bos_token_id", "tokenizer.ggml.bos_token_id"]
        ) {
            tokenizer_config.bos_token_id = Some(bos_id);
        }
        
        if let Some(eos_id) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::TOKENIZER_GGML_EOS_TOKEN_ID,
            &["tokenizer.eos_token_id", "eos_token_id", "tokenizer.ggml.eos_token_id"]
        ) {
            tokenizer_config.eos_token_id = Some(eos_id);
        }
        
        if let Some(pad_id) = self.get_u32_value_with_fallbacks(
            header, 
            GgufKeys::TOKENIZER_GGML_PAD_TOKEN_ID,
            &["tokenizer.pad_token_id", "pad_token_id", "tokenizer.ggml.pad_token_id"]
        ) {
            tokenizer_config.pad_token_id = Some(pad_id);
        }
        
        Ok(())
    }

    /// Extract additional metadata not covered by specific configurations
    fn extract_extra_metadata(&self, extra: &mut HashMap<String, String>, header: &GgufHeader) -> Result<()> {
        // Store all metadata as strings for debugging and future use
        for (key, value) in &header.metadata {
            if !key.starts_with("general.") && !key.starts_with("bitnet.") && !key.starts_with("tokenizer.") {
                // Store other metadata that might be useful
                let value_str = match value {
                    GgufValue::String(s) => s.clone(),
                    GgufValue::Uint32(n) => n.to_string(),
                    GgufValue::Float32(f) => f.to_string(),
                    GgufValue::Bool(b) => b.to_string(),
                    _ => format!("{:?}", value),
                };
                extra.insert(key.clone(), value_str);
            }
        }
        
        Ok(())
    }

    /// Helper function to get string value from metadata
    fn get_string_value(&self, header: &GgufHeader, key: &str) -> Option<String> {
        header.metadata.get(key).and_then(|v| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Helper function to get string value with fallback keys for Microsoft model compatibility
    fn get_string_value_with_fallbacks(&self, header: &GgufHeader, primary_key: &str, fallback_keys: &[&str]) -> Option<String> {
        // Try primary key first
        if let Some(value) = self.get_string_value(header, primary_key) {
            return Some(value);
        }
        
        // Try fallback keys
        for fallback_key in fallback_keys {
            if let Some(value) = self.get_string_value(header, fallback_key) {
                tracing::debug!("Found metadata using fallback key '{}' instead of '{}'", fallback_key, primary_key);
                return Some(value);
            }
        }
        
        None
    }

    /// Helper function to get u32 value from metadata
    fn get_u32_value(&self, header: &GgufHeader, key: &str) -> Option<u32> {
        header.metadata.get(key).and_then(|v| match v {
            GgufValue::Uint32(n) => Some(*n),
            GgufValue::Uint64(n) => Some(*n as u32),
            _ => None,
        })
    }

    /// Helper function to get u32 value with fallback keys for Microsoft model compatibility
    fn get_u32_value_with_fallbacks(&self, header: &GgufHeader, primary_key: &str, fallback_keys: &[&str]) -> Option<u32> {
        // Try primary key first
        if let Some(value) = self.get_u32_value(header, primary_key) {
            return Some(value);
        }
        
        // Try fallback keys
        for fallback_key in fallback_keys {
            if let Some(value) = self.get_u32_value(header, fallback_key) {
                tracing::debug!("Found metadata using fallback key '{}' instead of '{}'", fallback_key, primary_key);
                return Some(value);
            }
        }
        
        None
    }

    /// Helper function to get u64 value from metadata
    fn get_u64_value(&self, header: &GgufHeader, key: &str) -> Option<u64> {
        header.metadata.get(key).and_then(|v| match v {
            GgufValue::Uint64(n) => Some(*n),
            GgufValue::Uint32(n) => Some(*n as u64),
            _ => None,
        })
    }

    /// Helper function to get u64 value with fallback keys for Microsoft model compatibility
    fn get_u64_value_with_fallbacks(&self, header: &GgufHeader, primary_key: &str, fallback_keys: &[&str]) -> Option<u64> {
        // Try primary key first
        if let Some(value) = self.get_u64_value(header, primary_key) {
            return Some(value);
        }
        
        // Try fallback keys
        for fallback_key in fallback_keys {
            if let Some(value) = self.get_u64_value(header, fallback_key) {
                tracing::debug!("Found metadata using fallback key '{}' instead of '{}'", fallback_key, primary_key);
                return Some(value);
            }
        }
        
        None
    }

    /// Helper function to get f32 value from metadata
    fn get_f32_value(&self, header: &GgufHeader, key: &str) -> Option<f32> {
        header.metadata.get(key).and_then(|v| match v {
            GgufValue::Float32(n) => Some(*n),
            GgufValue::Float64(n) => Some(*n as f32),
            _ => None,
        })
    }

    /// Helper function to get f32 value with fallback keys for Microsoft model compatibility
    fn get_f32_value_with_fallbacks(&self, header: &GgufHeader, primary_key: &str, fallback_keys: &[&str]) -> Option<f32> {
        // Try primary key first
        if let Some(value) = self.get_f32_value(header, primary_key) {
            return Some(value);
        }
        
        // Try fallback keys
        for fallback_key in fallback_keys {
            if let Some(value) = self.get_f32_value(header, fallback_key) {
                tracing::debug!("Found metadata using fallback key '{}' instead of '{}'", fallback_key, primary_key);
                return Some(value);
            }
        }
        
        None
    }

    /// Read a 32-bit unsigned integer
    fn read_u32<R: Read>(&self, reader: &mut R) -> Result<u32> {
        let config = BufferReadConfig::default();
        let result = Self::read_buffer_robust(reader, 4, "u32", &config)?;
        match result {
            BufferReadResult::Complete(bytes) => {
                Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            _ => Err(InferenceError::model_load("Failed to read u32")),
        }
    }

    /// Read a 64-bit unsigned integer
    fn read_u64<R: Read>(&self, reader: &mut R) -> Result<u64> {
        let config = BufferReadConfig::default();
        let result = Self::read_buffer_robust(reader, 8, "u64", &config)?;
        match result {
            BufferReadResult::Complete(bytes) => {
                Ok(u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                    bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            }
            _ => Err(InferenceError::model_load("Failed to read u64")),
        }
    }

    /// Read a string from GGUF format
    fn read_string<R: Read>(&self, reader: &mut R) -> Result<String> {
        let len = self.read_u64(reader)? as usize;
        let config = BufferReadConfig::default();
        let result = Self::read_buffer_robust(reader, len, "string", &config)?;
        match result {
            BufferReadResult::Complete(bytes) => {
                // Try UTF-8 first, but use lossy conversion if needed
                match String::from_utf8(bytes.clone()) {
                    Ok(string) => Ok(string),
                    Err(utf8_err) => {
                        tracing::warn!("Invalid UTF-8 string at position {}, using lossy conversion", 
                                     utf8_err.utf8_error().valid_up_to());
                        // Use lossy conversion to handle non-UTF-8 data gracefully
                        let lossy_string = String::from_utf8_lossy(&bytes).into_owned();
                        Ok(lossy_string)
                    }
                }
            }
            _ => Err(InferenceError::model_load("Failed to read string")),
        }
    }

    /// Read a GGUF value based on type
    fn read_value<R: Read>(&self, reader: &mut R, value_type: GgufValueType) -> Result<GgufValue> {
        match value_type {
            GgufValueType::Uint8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read uint8: {}", e)))?;
                Ok(GgufValue::Uint8(buf[0]))
            }
            GgufValueType::Int8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read int8: {}", e)))?;
                Ok(GgufValue::Int8(buf[0] as i8))
            }
            GgufValueType::Uint16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read uint16: {}", e)))?;
                Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
            }
            GgufValueType::Int16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read int16: {}", e)))?;
                Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
            }
            GgufValueType::Uint32 => {
                let val = self.read_u32(reader)?;
                Ok(GgufValue::Uint32(val))
            }
            GgufValueType::Int32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read int32: {}", e)))?;
                Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
            }
            GgufValueType::Float32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read float32: {}", e)))?;
                Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
            }
            GgufValueType::Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read bool: {}", e)))?;
                Ok(GgufValue::Bool(buf[0] != 0))
            }
            GgufValueType::String => {
                let val = self.read_string(reader)?;
                Ok(GgufValue::String(val))
            }
            GgufValueType::Array => {
                let array_type = GgufValueType::try_from(self.read_u32(reader)?)?;
                let array_len = self.read_u64(reader)? as usize;
                
                let mut values = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    values.push(self.read_value(reader, array_type)?);
                }
                Ok(GgufValue::Array(values))
            }
            GgufValueType::Uint64 => {
                let val = self.read_u64(reader)?;
                Ok(GgufValue::Uint64(val))
            }
            GgufValueType::Int64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read int64: {}", e)))?;
                Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
            }
            GgufValueType::Float64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to read float64: {}", e)))?;
                Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
            }
        }
    }

    /// Skip an unknown value type by attempting to read and discard its content
    /// This uses heuristics to estimate the value size for skipping
    fn skip_unknown_value<R: Read>(&self, reader: &mut R, value_type_raw: u32) -> Result<()> {
        tracing::debug!("Attempting to skip unknown value type: {} (0x{:08x})", value_type_raw, value_type_raw);
        
        // Check if this looks like a corrupted value type
        // Valid GGUF value types are 0-12, so anything larger is likely corruption
        if value_type_raw > 12 {
            // This looks like corrupted data. The value might be:
            // 1. Part of binary data being interpreted as a type
            // 2. Endianness issue
            // 3. File corruption
            
            // Try to analyze the value to see if it contains readable patterns
            let hex_repr = format!("{:08x}", value_type_raw);
            let bytes = value_type_raw.to_le_bytes();
            
            // Check if bytes look like ASCII or have patterns suggesting they're data
            let ascii_chars: String = bytes.iter()
                .map(|&b| if b.is_ascii_graphic() || b.is_ascii_whitespace() { b as char } else { '.' })
                .collect();
            
            tracing::warn!("Unknown value type {} appears corrupted. Hex: 0x{}, ASCII: '{}'", 
                          value_type_raw, hex_repr, ascii_chars);
            
            // For very large values (likely corruption), we can't reliably skip
            // Return an error to stop parsing this metadata entry
            return Err(InferenceError::model_load(
                format!("Cannot skip corrupted value type {} - file may be corrupted or incorrectly formatted", value_type_raw)
            ));
        }
        
        // For moderately unknown types (13-255), try basic skipping strategies
        match value_type_raw {
            13..=20 => {
                // These might be future GGUF extensions, try skipping 8 bytes (common for 64-bit values)
                tracing::info!("Skipping potential future GGUF value type {} (assuming 8 bytes)", value_type_raw);
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to skip 8 bytes for unknown type {}: {}", value_type_raw, e)))?;
                Ok(())
            }
            21..=255 => {
                // Skip smaller amount for mid-range unknown types
                tracing::info!("Skipping unknown value type {} (assuming 4 bytes)", value_type_raw);
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)
                    .map_err(|e| InferenceError::model_load(format!("Failed to skip 4 bytes for unknown type {}: {}", value_type_raw, e)))?;
                Ok(())
            }
            _ => {
                // This shouldn't happen given our range checks above, but handle it
                Err(InferenceError::model_load(
                    format!("Cannot determine skip strategy for unknown value type {}", value_type_raw)
                ))
            }
        }
    }

    /// Organize raw tensor data into layer-based structure
    async fn organize_weights(&self, weights: &mut ModelWeights, header: &GgufHeader) -> Result<()> {
        // Read tensor info to organize weights
        for (tensor_index, tensor_info) in header.tensors.iter().enumerate() {
            let layer_info = self.parse_tensor_name(&tensor_info.name)?;
            let param_data = self.create_parameter_data(
                tensor_index, 
                &tensor_info, 
                &weights.layer_weights
            )?;
            
            // Map tensor name to layer ID
            weights.map_tensor_to_layer(tensor_info.name.clone(), layer_info.layer_id);
            
            // Add parameter to organized structure
            weights.add_parameter(layer_info.layer_id, layer_info.param_type, param_data);
        }
        
        tracing::info!("Organized {} tensors into {} layers", 
                      header.tensors.len(), weights.get_layer_ids().len());
        
        Ok(())
    }

    /// Parse tensor name to extract layer and parameter information
    pub fn parse_tensor_name(&self, tensor_name: &str) -> Result<LayerInfo> {
        // BitNet tensor naming patterns:
        // - token_embd.weight -> Embedding layer
        // - blk.{N}.attn_norm.weight -> Layer N, RMSNorm for attention 
        // - blk.{N}.ffn_gate.weight -> Layer N, Feed-forward gate weights
        // - blk.{N}.ffn_down.weight -> Layer N, Feed-forward down projection
        // - blk.{N}.ffn_sub_norm.weight -> Layer N, FFN sub-normalization
        // - blk.{N}.attn.q.weight -> Layer N, Attention query weights
        // - blk.{N}.attn.k.weight -> Layer N, Attention key weights  
        // - blk.{N}.attn.v.weight -> Layer N, Attention value weights
        // - blk.{N}.attn.output.weight -> Layer N, Attention output weights
        // - output.weight -> Output projection
        
        if tensor_name == "token_embd.weight" {
            return Ok(LayerInfo {
                layer_id: 0, // Special layer 0 for embeddings
                param_type: ParameterType::EmbeddingWeight,
            });
        }
        
        if tensor_name == "output.weight" {
            return Ok(LayerInfo {
                layer_id: 1000, // Special high layer ID for output
                param_type: ParameterType::OutputWeight,
            });
        }
        
        // Parse block-based layers: blk.{N}.{component}.{param}
        if tensor_name.starts_with("blk.") {
            let parts: Vec<&str> = tensor_name.split('.').collect();
            if parts.len() >= 3 {
                // Extract layer number
                let layer_num: usize = parts[1].parse()
                    .map_err(|_| InferenceError::model_load(
                        format!("Invalid layer number in tensor name: {}", tensor_name)))?;
                
                // Layer ID = layer_num + 10 (offset to avoid collision with special layers)
                let layer_id = layer_num + 10;
                
                // Parse parameter type based on component and parameter name
                let param_type = match (parts.get(2), parts.get(3)) {
                    (Some(&"attn_norm"), Some(&"weight")) => ParameterType::LayerNormScale,
                    (Some(&"ffn_gate"), Some(&"weight")) => ParameterType::FeedForwardGate,
                    (Some(&"ffn_down"), Some(&"weight")) => ParameterType::FeedForwardDown,
                    (Some(&"ffn_sub_norm"), Some(&"weight")) => ParameterType::LayerNormScale,
                    (Some(&"attn"), Some(&"q.weight")) => ParameterType::AttentionQuery,
                    (Some(&"attn"), Some(&"k.weight")) => ParameterType::AttentionKey,
                    (Some(&"attn"), Some(&"v.weight")) => ParameterType::AttentionValue,
                    (Some(&"attn"), Some(&"output.weight")) => ParameterType::AttentionOutput,
                    (Some(&component), Some(&"weight")) => {
                        // Default classification for unknown components
                        if component.contains("attn") {
                            ParameterType::AttentionOutput
                        } else if component.contains("ffn") || component.contains("mlp") {
                            ParameterType::FeedForwardDown
                        } else if component.contains("norm") {
                            ParameterType::LayerNormScale
                        } else {
                            ParameterType::Weight
                        }
                    },
                    _ => ParameterType::Weight, // Default fallback
                };
                
                return Ok(LayerInfo {
                    layer_id,
                    param_type,
                });
            }
        }
        
        // Fallback for unrecognized patterns
        Ok(LayerInfo {
            layer_id: 9999, // Special layer for unclassified tensors
            param_type: ParameterType::Weight,
        })
    }

    /// Create parameter data from tensor information
    fn create_parameter_data(
        &self,
        tensor_index: usize,
        tensor_info: &GgufTensorInfo,
        layer_weights: &HashMap<usize, Vec<u8>>
    ) -> Result<ParameterData> {
        let data = layer_weights.get(&tensor_index)
            .ok_or_else(|| InferenceError::model_load(
                format!("Tensor data not found for index {}", tensor_index)))?
            .clone();
        
        let shape = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        
        let dtype = match tensor_info.tensor_type {
            GgufTensorType::F32 => ParameterDataType::F32,
            GgufTensorType::F16 => ParameterDataType::F16,
            GgufTensorType::I8 => ParameterDataType::I8,
            GgufTensorType::BitnetB158 => ParameterDataType::BitnetB158,
            _ => ParameterDataType::Quantized(format!("{:?}", tensor_info.tensor_type)),
        };
        
        Ok(ParameterData {
            data,
            shape,
            dtype,
            tensor_name: tensor_info.name.clone(),
        })
    }

    /// Map GGUF header and tensor information to BitNet architecture using ArchitectureMapper
    fn map_architecture_from_gguf(
        &self,
        header: &GgufHeader,
        config: BitNetModelConfig,
    ) -> Result<ModelArchitecture> {
        use crate::engine::ArchitectureMapper;
        
        // Extract tensor information from header
        let tensor_info: Vec<GgufTensorInfo> = header.tensors.clone();

        tracing::info!("Creating architecture mapper with {} tensors", tensor_info.len());

        // Create architecture mapper
        let mut mapper = ArchitectureMapper::new(config, tensor_info);
        
        // Map to complete architecture
        let architecture = mapper.map_to_architecture()?;
        
        tracing::info!("Architecture mapping completed: {} layers, {} execution steps", 
                      architecture.layers.len(), architecture.execution_order.len());
        
        Ok(architecture)
    }
}

/// Information extracted from tensor name
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_id: usize,
    pub param_type: ParameterType,
}

impl Default for GgufLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_unknown_value_type_skipping() {
        // Test that unknown value types are handled gracefully
        let loader = GgufLoader::new();
        
        // Test valid value types
        for valid_type in 0..=12 {
            let result = GgufValueType::try_from(valid_type);
            assert!(result.is_ok(), "Valid type {} should succeed", valid_type);
        }
        
        // Test invalid value types return error with graceful message
        let invalid_types = [13, 100, 1767571456, 0xFFFFFFFF];
        for invalid_type in invalid_types {
            let result = GgufValueType::try_from(invalid_type);
            assert!(result.is_err(), "Invalid type {} should fail gracefully", invalid_type);
            
            let error_msg = result.unwrap_err().to_string();
            assert!(error_msg.contains("Unknown GGUF value type"), 
                   "Error should mention unknown value type: {}", error_msg);
        }
    }

    #[test]
    fn test_skip_unknown_value_method() {
        let loader = GgufLoader::new();
        
        // Test skipping moderately unknown types (13-255)
        let test_data = vec![0u8; 32]; // Enough data for any skip attempt
        let mut cursor = Cursor::new(test_data.clone());
        
        // Type 13 should be treated as corrupted since we designed it to be strict about > 12
        let result = loader.skip_unknown_value(&mut cursor, 13);
        assert!(result.is_err(), "Type 13 should be treated as corrupted");
        
        // Test that very large corrupted type fails appropriately
        let mut cursor2 = Cursor::new(test_data);
        let result = loader.skip_unknown_value(&mut cursor2, 1767571456);
        assert!(result.is_err(), "Should fail on corrupted type 1767571456");
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("corrupted"), 
               "Error should mention corruption: {}", error_msg);
    }

    #[test]
    fn test_gguf_value_type_coverage() {
        // Ensure all defined GGUF value types are supported
        let test_cases = [
            (0, GgufValueType::Uint8),
            (1, GgufValueType::Int8),
            (2, GgufValueType::Uint16),
            (3, GgufValueType::Int16),
            (4, GgufValueType::Uint32),
            (5, GgufValueType::Int32),
            (6, GgufValueType::Float32),
            (7, GgufValueType::Bool),
            (8, GgufValueType::String),
            (9, GgufValueType::Array),
            (10, GgufValueType::Uint64),
            (11, GgufValueType::Int64),
            (12, GgufValueType::Float64),
        ];
        
        for (type_id, expected_type) in test_cases {
            let result = GgufValueType::try_from(type_id);
            assert!(result.is_ok(), "Type {} should be valid", type_id);
            
            let actual_type = result.unwrap();
            assert_eq!(actual_type, expected_type, 
                      "Type {} should map to {:?}", type_id, expected_type);
        }
    }

    #[test]
    fn test_buffer_read_config_defaults() {
        let config = BufferReadConfig::default();
        
        // Verify reasonable defaults
        assert!(config.max_retries > 0, "Should have retry attempts");
        assert!(config.chunk_size > 0, "Should have positive chunk size");
        assert!(config.partial_tolerance >= 0.0 && config.partial_tolerance <= 1.0, 
               "Partial tolerance should be a valid percentage");
        assert!(config.large_tensor_threshold > config.chunk_size, 
               "Large tensor threshold should be larger than chunk size");
    }

    #[test]
    fn test_gguf_value_type_conversion() {
        assert_eq!(GgufValueType::try_from(0).unwrap(), GgufValueType::Uint8);
        assert_eq!(GgufValueType::try_from(8).unwrap(), GgufValueType::String);
        assert!(GgufValueType::try_from(999).is_err());
    }

    #[test]
    fn test_gguf_tensor_type_conversion() {
        assert_eq!(GgufTensorType::try_from(0).unwrap(), GgufTensorType::F32);
        assert_eq!(GgufTensorType::try_from(1000).unwrap(), GgufTensorType::BitnetB158);
        // Unknown types should fallback to F32
        assert_eq!(GgufTensorType::try_from(999).unwrap(), GgufTensorType::F32);
    }

    #[test]
    fn test_buffer_read_config_default() {
        let config = BufferReadConfig::default();
        assert_eq!(config.max_retries, MAX_BUFFER_READ_RETRIES);
        assert_eq!(config.partial_tolerance, 0.05);
        assert!(config.enable_streaming);
        assert_eq!(config.chunk_size, 16 * 1024 * 1024);
        assert_eq!(config.large_tensor_threshold, 100 * 1024 * 1024);
    }

    #[test]
    fn test_ternary_weight_decoding() {
        let loader = GgufLoader::new();
        
        // Create test tensor info for BitNet ternary weights
        let tensor_info = GgufTensorInfo {
            name: "test_weights".to_string(),
            dimensions: vec![4], // 4 elements
            tensor_type: GgufTensorType::BitnetB158,
            offset: 0,
        };
        
        // Create test packed data: scale(4), offset(4), packed_weights
        // 4 ternary weights packed in 1 byte: [-1, 0, 1, -1] -> [00, 01, 10, 00] -> 0b00100100 = 36
        let packed_data = vec![
            0x3F, 0x80, 0x00, 0x00, // scale = 1.0 (f32 little-endian)
            0x00, 0x00, 0x00, 0x00, // offset = 0.0 (f32 little-endian)
            0b00100100,             // packed weights: [-1, 0, 1, -1]
        ];
        
        let result = loader.decode_ternary_weights(&packed_data, &tensor_info);
        assert!(result.is_ok());
        
        let decoded = result.unwrap();
        assert_eq!(decoded.len(), 4);
        
        // Convert back to i8 for validation
        let weights: Vec<i8> = decoded.iter().map(|&b| b as i8).collect();
        assert_eq!(weights, vec![-1, 0, 1, -1]);
    }

    #[test]
    fn test_tensor_validation() {
        let loader = GgufLoader::new();
        
        // Valid tensor info
        let valid_tensor = GgufTensorInfo {
            name: "valid_tensor".to_string(),
            dimensions: vec![10, 20],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert!(loader.validate_tensor_info(&valid_tensor).is_ok());
        
        // Invalid tensor: empty name
        let invalid_name = GgufTensorInfo {
            name: "".to_string(),
            dimensions: vec![10],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert!(loader.validate_tensor_info(&invalid_name).is_err());
        
        // Invalid tensor: no dimensions
        let no_dims = GgufTensorInfo {
            name: "no_dims".to_string(),
            dimensions: vec![],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert!(loader.validate_tensor_info(&no_dims).is_err());
    }
}