use anyhow::{Result, anyhow, Context};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Enhanced model configuration with comprehensive validation and loading capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture configuration
    pub architecture: ArchitectureConfig,
    /// Training-specific configuration
    pub training: TrainingModelConfig,
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
    /// Model metadata and information
    pub metadata: ModelMetadata,
    /// Device and performance configuration
    pub device: DeviceConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Model architecture type (e.g., "BitNet", "BitNetTransformer")
    pub architecture_type: String,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout probability
    pub dropout_prob: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    /// BitNet-specific quantization bits (1.58, 2, 4, 8)
    pub quantization_bits: f32,
    /// Activation function type
    pub activation_function: String,
    /// Position embedding type
    pub position_embedding_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingModelConfig {
    /// Initialize model weights randomly
    pub random_init: bool,
    /// Random seed for reproducible initialization
    pub seed: Option<u64>,
    /// Weight initialization strategy
    pub init_strategy: String,
    /// Scale factor for weight initialization
    pub init_scale: f32,
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Mixed precision training mode
    pub mixed_precision: bool,
    /// Compile model for optimization
    pub compile_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Tokenizer type (e.g., "BPE", "SentencePiece", "LLaMA")
    pub tokenizer_type: String,
    /// Path to tokenizer file or HuggingFace model name
    pub tokenizer_path: Option<PathBuf>,
    /// Special tokens configuration
    pub special_tokens: SpecialTokens,
    /// Additional tokenizer parameters
    pub additional_params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: String,
    /// End of sequence token
    pub eos_token: String,
    /// Padding token
    pub pad_token: String,
    /// Unknown token
    pub unk_token: String,
    /// Mask token (for masked language modeling)
    pub mask_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Author/Organization
    pub author: String,
    /// License information
    pub license: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Additional metadata
    pub additional_info: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Preferred device type ("cpu", "gpu", "auto")
    pub device_type: String,
    /// GPU device ID (if using GPU)
    pub gpu_id: Option<usize>,
    /// Enable multi-GPU training
    pub multi_gpu: bool,
    /// Number of CPU threads
    pub num_threads: Option<usize>,
    /// Enable CPU optimizations (SIMD, etc.)
    pub cpu_optimizations: bool,
    /// Enable Apple Metal acceleration
    pub metal_acceleration: bool,
    /// Enable CUDA acceleration
    pub cuda_acceleration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Enable memory mapping for large models
    pub memory_mapping: bool,
    /// Enable model sharding for large models
    pub model_sharding: bool,
    /// Offload layers to CPU if needed
    pub cpu_offload: bool,
    /// Memory optimization strategy
    pub optimization_strategy: String,
}

/// Configuration validation results
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub parameter_count: Option<usize>,
    pub memory_estimate_mb: Option<usize>,
}

/// Model configuration manager with loading, validation, and utility functions
pub struct ModelConfigManager;

impl ModelConfigManager {
    /// Load model configuration from JSON file
    pub fn load_from_json(path: impl AsRef<Path>) -> Result<ModelConfig> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let config: ModelConfig = serde_json::from_str(&content)
            .with_context(|| "Failed to parse JSON configuration")?;
        
        info!("📄 Loaded model configuration from: {:?}", path.as_ref());
        Ok(config)
    }
    
    /// Load model configuration from YAML file
    pub fn load_from_yaml(path: impl AsRef<Path>) -> Result<ModelConfig> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let config: ModelConfig = serde_yaml::from_str(&content)
            .with_context(|| "Failed to parse YAML configuration")?;
        
        info!("📄 Loaded model configuration from: {:?}", path.as_ref());
        Ok(config)
    }
    
    /// Load configuration with automatic format detection
    pub fn load_auto_detect(path: impl AsRef<Path>) -> Result<ModelConfig> {
        let path_ref = path.as_ref();
        
        match path_ref.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Self::load_from_json(path_ref),
            Some("yaml") | Some("yml") => Self::load_from_yaml(path_ref),
            _ => {
                // Try JSON first, then YAML
                Self::load_from_json(path_ref)
                    .or_else(|_| Self::load_from_yaml(path_ref))
                    .with_context(|| "Failed to load config as JSON or YAML")
            }
        }
    }
    
    /// Save model configuration to JSON file
    pub fn save_to_json(config: &ModelConfig, path: impl AsRef<Path>) -> Result<()> {
        let content = serde_json::to_string_pretty(config)
            .with_context(|| "Failed to serialize configuration to JSON")?;
        
        std::fs::write(path.as_ref(), content)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;
        
        info!("💾 Saved model configuration to: {:?}", path.as_ref());
        Ok(())
    }
    
    /// Save model configuration to YAML file
    pub fn save_to_yaml(config: &ModelConfig, path: impl AsRef<Path>) -> Result<()> {
        let content = serde_yaml::to_string(config)
            .with_context(|| "Failed to serialize configuration to YAML")?;
        
        std::fs::write(path.as_ref(), content)
            .with_context(|| format!("Failed to write config file: {:?}", path.as_ref()))?;
        
        info!("💾 Saved model configuration to: {:?}", path.as_ref());
        Ok(())
    }
    
    /// Comprehensive configuration validation
    pub fn validate_config(config: &ModelConfig) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Validate architecture parameters
        Self::validate_architecture(&config.architecture, &mut errors, &mut warnings);
        
        // Validate training configuration
        Self::validate_training_config(&config.training, &mut errors, &mut warnings);
        
        // Validate tokenizer configuration
        Self::validate_tokenizer_config(&config.tokenizer, &mut errors, &mut warnings);
        
        // Validate device configuration
        Self::validate_device_config(&config.device, &mut errors, &mut warnings);
        
        // Validate memory configuration
        Self::validate_memory_config(&config.memory, &mut errors, &mut warnings);
        
        // Estimate parameter count and memory usage
        let parameter_count = Self::estimate_parameter_count(&config.architecture);
        let memory_estimate_mb = Self::estimate_memory_usage(&config.architecture, &config.training);
        
        // Cross-validation checks
        Self::validate_consistency(config, &mut errors, &mut warnings);
        
        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            parameter_count,
            memory_estimate_mb,
        }
    }
    
    /// Validate architecture configuration
    fn validate_architecture(arch: &ArchitectureConfig, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Validate dimensions
        if arch.num_layers == 0 {
            errors.push("num_layers must be greater than 0".to_string());
        }
        if arch.num_layers > 100 {
            warnings.push("num_layers > 100 may require significant memory".to_string());
        }
        
        if arch.hidden_size == 0 {
            errors.push("hidden_size must be greater than 0".to_string());
        }
        if arch.hidden_size % arch.num_attention_heads != 0 {
            errors.push("hidden_size must be divisible by num_attention_heads".to_string());
        }
        
        if arch.num_attention_heads == 0 {
            errors.push("num_attention_heads must be greater than 0".to_string());
        }
        
        if arch.max_sequence_length == 0 {
            errors.push("max_sequence_length must be greater than 0".to_string());
        }
        if arch.max_sequence_length > 32768 {
            warnings.push("max_sequence_length > 32768 may require significant memory".to_string());
        }
        
        if arch.vocab_size == 0 {
            errors.push("vocab_size must be greater than 0".to_string());
        }
        
        // Validate quantization
        if ![1.58, 2.0, 4.0, 8.0].contains(&arch.quantization_bits) {
            warnings.push(format!("Unusual quantization_bits value: {}", arch.quantization_bits));
        }
        
        // Validate probability values
        if arch.dropout_prob < 0.0 || arch.dropout_prob > 1.0 {
            errors.push("dropout_prob must be between 0.0 and 1.0".to_string());
        }
        
        // Validate supported activation functions
        let supported_activations = ["gelu", "relu", "swish", "silu"];
        if !supported_activations.contains(&arch.activation_function.as_str()) {
            warnings.push(format!("Activation function '{}' may not be supported", arch.activation_function));
        }
    }
    
    /// Validate training configuration
    fn validate_training_config(training: &TrainingModelConfig, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Validate initialization scale
        if training.init_scale <= 0.0 {
            errors.push("init_scale must be greater than 0.0".to_string());
        }
        if training.init_scale > 10.0 {
            warnings.push("init_scale > 10.0 may cause training instability".to_string());
        }
        
        // Validate initialization strategy
        let supported_strategies = ["normal", "xavier", "he", "uniform"];
        if !supported_strategies.contains(&training.init_strategy.as_str()) {
            warnings.push(format!("Init strategy '{}' may not be supported", training.init_strategy));
        }
    }
    
    /// Validate tokenizer configuration
    fn validate_tokenizer_config(tokenizer: &TokenizerConfig, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Check if tokenizer path exists (if specified)
        if let Some(ref path) = tokenizer.tokenizer_path {
            if !path.exists() {
                errors.push(format!("Tokenizer path does not exist: {:?}", path));
            }
        }
        
        // Validate special tokens are not empty
        if tokenizer.special_tokens.bos_token.is_empty() {
            warnings.push("bos_token is empty".to_string());
        }
        if tokenizer.special_tokens.eos_token.is_empty() {
            warnings.push("eos_token is empty".to_string());
        }
        if tokenizer.special_tokens.pad_token.is_empty() {
            warnings.push("pad_token is empty".to_string());
        }
    }
    
    /// Validate device configuration
    fn validate_device_config(device: &DeviceConfig, _errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Validate device type
        let supported_devices = ["cpu", "gpu", "auto"];
        if !supported_devices.contains(&device.device_type.as_str()) {
            warnings.push(format!("Device type '{}' may not be supported", device.device_type));
        }
        
        // Check for conflicting GPU options
        if device.cuda_acceleration && device.metal_acceleration {
            warnings.push("Both CUDA and Metal acceleration enabled; only one will be used".to_string());
        }
        
        // Validate thread count
        if let Some(threads) = device.num_threads {
            if threads == 0 {
                warnings.push("num_threads set to 0; will use default".to_string());
            }
            if threads > 64 {
                warnings.push("num_threads > 64 may not improve performance".to_string());
            }
        }
    }
    
    /// Validate memory configuration
    fn validate_memory_config(memory: &MemoryConfig, _errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Validate memory limit
        if let Some(max_mem) = memory.max_memory_mb {
            if max_mem < 1024 {
                warnings.push("max_memory_mb < 1024 may be insufficient for training".to_string());
            }
        }
        
        // Validate optimization strategy
        let supported_strategies = ["default", "memory_efficient", "speed_optimized", "balanced"];
        if !supported_strategies.contains(&memory.optimization_strategy.as_str()) {
            warnings.push(format!("Memory optimization strategy '{}' may not be supported", 
                                memory.optimization_strategy));
        }
    }
    
    /// Cross-validation for consistency between different config sections
    fn validate_consistency(config: &ModelConfig, errors: &mut Vec<String>, warnings: &mut Vec<String>) {
        // Check if mixed precision is compatible with device
        if config.training.mixed_precision && config.device.device_type == "cpu" {
            warnings.push("Mixed precision training may not be beneficial on CPU".to_string());
        }
        
        // Check memory configuration vs model size
        if let (Some(max_mem), Some(estimated_mem)) = (
            config.memory.max_memory_mb,
            Self::estimate_memory_usage(&config.architecture, &config.training)
        ) {
            if max_mem < estimated_mem {
                warnings.push(format!(
                    "max_memory_mb ({}) is less than estimated memory usage ({})", 
                    max_mem, estimated_mem
                ));
            }
        }
        
        // Check sequence length vs memory
        if config.architecture.max_sequence_length > 8192 && config.memory.max_memory_mb.unwrap_or(16384) < 8192 {
            warnings.push("Large sequence length with limited memory may cause OOM errors".to_string());
        }
    }
    
    /// Estimate total number of parameters in the model
    fn estimate_parameter_count(arch: &ArchitectureConfig) -> Option<usize> {
        // Simplified parameter estimation for transformer architecture
        let embedding_params = arch.vocab_size * arch.hidden_size;
        let position_embedding_params = arch.max_sequence_length * arch.hidden_size;
        
        // Per-layer parameters (simplified)
        let attention_params = 4 * arch.hidden_size * arch.hidden_size; // Q, K, V, O projections
        let ffn_params = 2 * arch.hidden_size * arch.intermediate_size; // Up and down projections
        let layer_norm_params = 2 * arch.hidden_size; // Pre and post attention layer norms
        
        let per_layer_params = attention_params + ffn_params + layer_norm_params;
        let total_layer_params = per_layer_params * arch.num_layers;
        
        let final_layer_norm_params = arch.hidden_size;
        let lm_head_params = arch.hidden_size * arch.vocab_size;
        
        let total_params = embedding_params 
            + position_embedding_params 
            + total_layer_params 
            + final_layer_norm_params 
            + lm_head_params;
        
        Some(total_params)
    }
    
    /// Estimate memory usage for the model
    fn estimate_memory_usage(arch: &ArchitectureConfig, training: &TrainingModelConfig) -> Option<usize> {
        let param_count = Self::estimate_parameter_count(arch)?;
        
        // Bytes per parameter based on quantization
        let bytes_per_param = match arch.quantization_bits {
            1.58 => 0.2, // Approximately 1.58 bits per parameter
            2.0 => 0.25,
            4.0 => 0.5,
            8.0 => 1.0,
            _ => 4.0, // Default to FP32
        };
        
        let model_memory_mb = (param_count as f32 * bytes_per_param / 1024.0 / 1024.0) as usize;
        
        // Training overhead
        let training_multiplier = if training.mixed_precision {
            2.5 // Model + gradients + optimizer state (reduced precision)
        } else {
            4.0 // Model + gradients + optimizer state (full precision)
        };
        
        let total_memory_mb = (model_memory_mb as f32 * training_multiplier) as usize;
        
        // Add activation memory (sequence-dependent)
        let activation_memory_mb = (arch.max_sequence_length * arch.hidden_size * arch.num_layers * 4) / (1024 * 1024);
        
        Some(total_memory_mb + activation_memory_mb)
    }
    
    /// Generate default configuration for common model sizes
    pub fn create_default_config(model_size: ModelSize) -> ModelConfig {
        let (num_layers, hidden_size, num_heads, intermediate_size) = match model_size {
            ModelSize::Tiny => (6, 384, 6, 1536),
            ModelSize::Small => (12, 768, 12, 3072),
            ModelSize::Medium => (24, 1024, 16, 4096),
            ModelSize::Large => (36, 1280, 20, 5120),
            ModelSize::XLarge => (48, 1600, 25, 6400),
        };
        
        ModelConfig {
            architecture: ArchitectureConfig {
                architecture_type: "BitNetTransformer".to_string(),
                num_layers,
                hidden_size,
                num_attention_heads: num_heads,
                intermediate_size,
                max_sequence_length: 2048,
                vocab_size: 32000,
                dropout_prob: 0.1,
                layer_norm_eps: 1e-5,
                quantization_bits: 1.58,
                activation_function: "gelu".to_string(),
                position_embedding_type: "absolute".to_string(),
            },
            training: TrainingModelConfig {
                random_init: true,
                seed: Some(42),
                init_strategy: "xavier".to_string(),
                init_scale: 1.0,
                gradient_checkpointing: false,
                mixed_precision: true,
                compile_model: false,
            },
            tokenizer: TokenizerConfig {
                tokenizer_type: "BPE".to_string(),
                tokenizer_path: None,
                special_tokens: SpecialTokens {
                    bos_token: "<bos>".to_string(),
                    eos_token: "<eos>".to_string(),
                    pad_token: "<pad>".to_string(),
                    unk_token: "<unk>".to_string(),
                    mask_token: Some("<mask>".to_string()),
                },
                additional_params: HashMap::new(),
            },
            metadata: ModelMetadata {
                name: format!("BitNet-{:?}", model_size),
                version: "1.0.0".to_string(),
                description: format!("BitNet model with 1.58-bit quantization ({:?} size)", model_size),
                author: "BitNet-Rust".to_string(),
                license: "MIT".to_string(),
                tags: vec!["bitnet".to_string(), "quantized".to_string(), "transformer".to_string()],
                created_at: Some(chrono::Utc::now()),
                additional_info: HashMap::new(),
            },
            device: DeviceConfig {
                device_type: "auto".to_string(),
                gpu_id: None,
                multi_gpu: false,
                num_threads: None,
                cpu_optimizations: true,
                metal_acceleration: true,
                cuda_acceleration: true,
            },
            memory: MemoryConfig {
                max_memory_mb: None,
                memory_mapping: false,
                model_sharding: false,
                cpu_offload: false,
                optimization_strategy: "balanced".to_string(),
            },
        }
    }
    
    /// Create configuration from command line overrides
    pub fn create_from_overrides(
        base_config: ModelConfig,
        overrides: &HashMap<String, serde_json::Value>
    ) -> Result<ModelConfig> {
        let mut config = base_config;
        
        for (key, value) in overrides {
            Self::apply_override(&mut config, key, value)
                .with_context(|| format!("Failed to apply override for key: {}", key))?;
        }
        
        Ok(config)
    }
    
    /// Apply a single configuration override
    fn apply_override(config: &mut ModelConfig, key: &str, value: &serde_json::Value) -> Result<()> {
        match key {
            "learning_rate" => {
                // This would typically be in TrainingConfig, but we can store it in metadata
                config.metadata.additional_info.insert("learning_rate".to_string(), value.clone());
            },
            "batch_size" => {
                config.metadata.additional_info.insert("batch_size".to_string(), value.clone());
            },
            "num_epochs" => {
                config.metadata.additional_info.insert("num_epochs".to_string(), value.clone());
            },
            "max_sequence_length" => {
                if let Some(seq_len) = value.as_u64() {
                    config.architecture.max_sequence_length = seq_len as usize;
                } else {
                    return Err(anyhow!("max_sequence_length must be a number"));
                }
            },
            "hidden_size" => {
                if let Some(hidden_size) = value.as_u64() {
                    config.architecture.hidden_size = hidden_size as usize;
                } else {
                    return Err(anyhow!("hidden_size must be a number"));
                }
            },
            "num_layers" => {
                if let Some(num_layers) = value.as_u64() {
                    config.architecture.num_layers = num_layers as usize;
                } else {
                    return Err(anyhow!("num_layers must be a number"));
                }
            },
            _ => {
                // Store unknown overrides in additional_info
                config.metadata.additional_info.insert(key.to_string(), value.clone());
            }
        }
        
        Ok(())
    }
}

/// Predefined model size configurations
#[derive(Debug, Clone, Copy)]
pub enum ModelSize {
    Tiny,    // ~10M parameters
    Small,   // ~100M parameters
    Medium,  // ~350M parameters
    Large,   // ~1B parameters
    XLarge,  // ~2.5B parameters
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfigManager::create_default_config(ModelSize::Small)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config_creation() {
        let config = ModelConfigManager::create_default_config(ModelSize::Small);
        assert_eq!(config.architecture.num_layers, 12);
        assert_eq!(config.architecture.hidden_size, 768);
        assert_eq!(config.architecture.quantization_bits, 1.58);
    }

    #[test]
    fn test_config_validation() {
        let config = ModelConfigManager::create_default_config(ModelSize::Tiny);
        let result = ModelConfigManager::validate_config(&config);
        assert!(result.is_valid);
        assert!(result.parameter_count.is_some());
        assert!(result.memory_estimate_mb.is_some());
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut config = ModelConfigManager::create_default_config(ModelSize::Small);
        config.architecture.num_layers = 0; // Invalid
        config.architecture.dropout_prob = 1.5; // Invalid
        
        let result = ModelConfigManager::validate_config(&config);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_parameter_estimation() {
        let config = ModelConfigManager::create_default_config(ModelSize::Small);
        let param_count = ModelConfigManager::estimate_parameter_count(&config.architecture);
        assert!(param_count.is_some());
        
        let count = param_count.unwrap();
        assert!(count > 50_000_000); // Should be around 100M for small model
        assert!(count < 200_000_000);
    }

    #[test]
    fn test_memory_estimation() {
        let config = ModelConfigManager::create_default_config(ModelSize::Small);
        let memory_mb = ModelConfigManager::estimate_memory_usage(
            &config.architecture, 
            &config.training
        );
        assert!(memory_mb.is_some());
        
        let memory = memory_mb.unwrap();
        assert!(memory > 100); // Should require at least 100MB
        assert!(memory < 10000); // Should be less than 10GB for small model
    }

    #[test]
    fn test_config_serialization() {
        let config = ModelConfigManager::create_default_config(ModelSize::Tiny);
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.architecture.num_layers, deserialized.architecture.num_layers);
        
        // Test YAML serialization
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.architecture.hidden_size, deserialized.architecture.hidden_size);
    }

    #[test]
    fn test_config_file_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = ModelConfigManager::create_default_config(ModelSize::Tiny);
        
        // Test JSON file operations
        let json_path = temp_dir.path().join("config.json");
        ModelConfigManager::save_to_json(&config, &json_path)?;
        let loaded_config = ModelConfigManager::load_from_json(&json_path)?;
        assert_eq!(config.architecture.num_layers, loaded_config.architecture.num_layers);
        
        // Test YAML file operations
        let yaml_path = temp_dir.path().join("config.yaml");
        ModelConfigManager::save_to_yaml(&config, &yaml_path)?;
        let loaded_config = ModelConfigManager::load_from_yaml(&yaml_path)?;
        assert_eq!(config.architecture.hidden_size, loaded_config.architecture.hidden_size);
        
        // Test auto-detection
        let auto_loaded = ModelConfigManager::load_auto_detect(&json_path)?;
        assert_eq!(config.architecture.vocab_size, auto_loaded.architecture.vocab_size);
        
        Ok(())
    }

    #[test]
    fn test_configuration_overrides() {
        let base_config = ModelConfigManager::create_default_config(ModelSize::Small);
        let mut overrides = HashMap::new();
        overrides.insert("num_layers".to_string(), serde_json::json!(24));
        overrides.insert("hidden_size".to_string(), serde_json::json!(1024));
        overrides.insert("learning_rate".to_string(), serde_json::json!(0.001));
        
        let modified_config = ModelConfigManager::create_from_overrides(base_config, &overrides).unwrap();
        assert_eq!(modified_config.architecture.num_layers, 24);
        assert_eq!(modified_config.architecture.hidden_size, 1024);
        assert!(modified_config.metadata.additional_info.contains_key("learning_rate"));
    }
}