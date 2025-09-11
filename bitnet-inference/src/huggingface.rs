//! HuggingFace Hub integration for model loading and management.
//!
//! This module provides functionality to:
//! - Download models from HuggingFace Hub
//! - Load SafeTensors format models
//! - Cache models locally for efficient reuse
//! - Convert models to BitNet format

use crate::{Result, InferenceError};
use crate::engine::{Model, ModelMetadata, LoadedModel};
use crate::engine::model_loader::{ModelArchitecture, LayerDefinition, LayerType, LayerParameters, ModelWeights};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;
use std::sync::Arc;

/// HuggingFace model repository identifier
#[derive(Debug, Clone)]
pub struct ModelRepo {
    /// Repository owner/organization
    pub owner: String,
    /// Repository name
    pub name: String,
    /// Optional revision (branch, tag, or commit)
    pub revision: Option<String>,
}

impl ModelRepo {
    /// Create a new model repository identifier
    pub fn new(owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            name: name.into(),
            revision: None,
        }
    }

    /// Set a specific revision (branch, tag, or commit)
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Get the full repository identifier (owner/name)
    pub fn repo_id(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

/// Configuration for HuggingFace model downloading
#[derive(Debug, Clone)]
pub struct HuggingFaceConfig {
    /// Local cache directory for downloaded models
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes (default: 5GB)
    pub max_cache_size: u64,
    /// Authentication token for private repositories
    pub auth_token: Option<String>,
    /// Whether to use offline mode (only use cached models)
    pub offline: bool,
    /// Download timeout in seconds
    pub timeout_seconds: u64,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("bitnet-inference")
                .join("huggingface"),
            max_cache_size: 5 * 1024 * 1024 * 1024, // 5GB
            auth_token: std::env::var("HF_TOKEN").ok(),
            offline: false,
            timeout_seconds: 300, // 5 minutes
        }
    }
}

/// HuggingFace model loader with caching and conversion capabilities
pub struct HuggingFaceLoader {
    config: HuggingFaceConfig,
    client: reqwest::Client,
}

impl HuggingFaceLoader {
    /// Create a new HuggingFace loader with default configuration
    pub fn new() -> Result<Self> {
        let config = HuggingFaceConfig::default();
        Self::with_config(config)
    }

    /// Create a new HuggingFace loader with custom configuration
    pub fn with_config(config: HuggingFaceConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| InferenceError::model_load(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
        })
    }

    /// Load a model from HuggingFace Hub
    pub async fn load_model(&self, repo: &ModelRepo) -> Result<LoadedModel> {
        // 1. Check local cache first
        if let Some(cached_model) = self.check_cache(repo).await? {
            return Ok(cached_model);
        }

        // 2. Download model files if not in offline mode
        if !self.config.offline {
            self.download_model(repo).await?;
        } else {
            return Err(InferenceError::model_load(
                format!("Model {} not found in cache and offline mode is enabled", repo.repo_id())
            ));
        }

        // 3. Load the downloaded model
        self.load_cached_model(repo).await
    }

    /// Download a model from HuggingFace Hub
    pub async fn download_model(&self, repo: &ModelRepo) -> Result<PathBuf> {
        let model_dir = self.get_model_cache_dir(repo);
        
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&model_dir).await
            .map_err(|e| InferenceError::model_load(format!("Failed to create cache directory: {}", e)))?;

        // Download model files
        self.download_model_files(repo, &model_dir).await?;

        Ok(model_dir)
    }

    /// Check if a model exists in the local cache
    async fn check_cache(&self, repo: &ModelRepo) -> Result<Option<LoadedModel>> {
        let model_dir = self.get_model_cache_dir(repo);
        
        if !model_dir.exists() {
            return Ok(None);
        }

        // Check if all required files exist
        let config_path = model_dir.join("config.json");
        let model_path = model_dir.join("model.safetensors");

        if !config_path.exists() || !model_path.exists() {
            return Ok(None);
        }

        // Load the cached model
        match self.load_cached_model(repo).await {
            Ok(model) => Ok(Some(model)),
            Err(_) => Ok(None), // If loading fails, treat as cache miss
        }
    }

    /// Load a model from the local cache
    async fn load_cached_model(&self, repo: &ModelRepo) -> Result<LoadedModel> {
        let model_dir = self.get_model_cache_dir(repo);
        
        // Load model configuration
        let config_path = model_dir.join("config.json");
        let config_content = fs::read_to_string(&config_path).await
            .map_err(|e| InferenceError::model_load(format!("Failed to read config: {}", e)))?;
        
        let hf_config: HuggingFaceModelConfig = serde_json::from_str(&config_content)
            .map_err(|e| InferenceError::model_load(format!("Failed to parse config: {}", e)))?;

        // Load SafeTensors model
        let model_path = model_dir.join("model.safetensors");
        let weights = self.load_safetensors(&model_path).await?;

        // Convert to BitNet format
        let metadata = self.convert_to_metadata(&hf_config, repo)?;
        let architecture = self.convert_to_architecture(&hf_config)?;

        Ok(LoadedModel {
            metadata,
            architecture,
            weights,
        })
    }

    /// Download model files from HuggingFace Hub
    async fn download_model_files(&self, repo: &ModelRepo, model_dir: &Path) -> Result<()> {
        let files_to_download = vec![
            "config.json",
            "model.safetensors",
            "tokenizer.json",  // Optional
            "tokenizer_config.json", // Optional
        ];

        for file_name in files_to_download {
            let url = self.build_download_url(repo, file_name);
            let file_path = model_dir.join(file_name);

            match self.download_file(&url, &file_path).await {
                Ok(_) => {
                    tracing::info!("Downloaded {}", file_name);
                }
                Err(e) => {
                    // Some files are optional
                    if file_name.starts_with("tokenizer") {
                        tracing::warn!("Optional file {} not available: {}", file_name, e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Download a single file from HuggingFace Hub
    async fn download_file(&self, url: &str, file_path: &Path) -> Result<()> {
        let mut request = self.client.get(url);

        // Add authentication header if token is available
        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await
            .map_err(|e| InferenceError::model_load(format!("Failed to download {}: {}", url, e)))?;

        if !response.status().is_success() {
            return Err(InferenceError::model_load(
                format!("Failed to download {}: HTTP {}", url, response.status())
            ));
        }

        let content = response.bytes().await
            .map_err(|e| InferenceError::model_load(format!("Failed to read response: {}", e)))?;

        fs::write(file_path, content).await
            .map_err(|e| InferenceError::model_load(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load weights from a SafeTensors file
    async fn load_safetensors(&self, path: &Path) -> Result<ModelWeights> {
        let data = fs::read(path).await
            .map_err(|e| InferenceError::model_load(format!("Failed to read SafeTensors file: {}", e)))?;

        // Parse SafeTensors format
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| InferenceError::model_load(format!("Failed to parse SafeTensors: {}", e)))?;

        let mut layer_weights = HashMap::new();
        let mut total_size = 0;

        // Extract tensor data
        for (i, (name, _tensor)) in tensors.tensors().into_iter().enumerate() {
            let tensor_data = tensors.tensor(&name)
                .map_err(|e| InferenceError::model_load(format!("Failed to extract tensor {}: {}", name, e)))?;
            
            layer_weights.insert(i, tensor_data.data().to_vec());
            total_size += tensor_data.data().len();
        }

        Ok(ModelWeights {
            layer_weights,
            total_size,
        })
    }

    /// Convert HuggingFace config to BitNet metadata
    fn convert_to_metadata(&self, config: &HuggingFaceModelConfig, repo: &ModelRepo) -> Result<ModelMetadata> {
        let extra = HashMap::new();

        Ok(ModelMetadata {
            name: repo.repo_id(),
            version: repo.revision.clone().unwrap_or_else(|| "main".to_string()),
            architecture: config.architectures.first()
                .map(|s| s.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            parameter_count: config.get_parameter_count(),
            quantization_bits: 1, // Assume 1-bit for BitNet
            input_shape: vec![1, config.max_position_embeddings.unwrap_or(2048)],
            output_shape: vec![1, config.vocab_size],
            extra,
        })
    }

    /// Convert HuggingFace config to BitNet architecture
    fn convert_to_architecture(&self, config: &HuggingFaceModelConfig) -> Result<ModelArchitecture> {
        let mut layers = Vec::new();
        let num_layers = config.num_hidden_layers.unwrap_or(12);

        // Create BitLinear layers based on the model configuration
        for i in 0..num_layers {
            layers.push(LayerDefinition {
                id: i,
                layer_type: LayerType::BitLinear,
                input_dims: vec![config.hidden_size],
                output_dims: vec![config.hidden_size],
                parameters: LayerParameters::BitLinear {
                    weight_bits: 1,
                    activation_bits: 8,
                },
            });
        }

        // Add output projection layer
        layers.push(LayerDefinition {
            id: num_layers,
            layer_type: LayerType::OutputProjection,
            input_dims: vec![config.hidden_size],
            output_dims: vec![config.vocab_size],
            parameters: LayerParameters::OutputProjection {
                vocab_size: config.vocab_size,
            },
        });

        Ok(ModelArchitecture {
            layers,
            execution_order: (0..=num_layers).collect(),
        })
    }

    /// Build download URL for a specific file
    fn build_download_url(&self, repo: &ModelRepo, file_name: &str) -> String {
        let revision = repo.revision.as_deref().unwrap_or("main");
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo.repo_id(),
            revision,
            file_name
        )
    }

    /// Get the local cache directory for a model
    fn get_model_cache_dir(&self, repo: &ModelRepo) -> PathBuf {
        let revision = repo.revision.as_deref().unwrap_or("main");
        self.config.cache_dir
            .join(&repo.owner)
            .join(&repo.name)
            .join(revision)
    }

    /// Clear the model cache
    pub async fn clear_cache(&self) -> Result<()> {
        if self.config.cache_dir.exists() {
            fs::remove_dir_all(&self.config.cache_dir).await
                .map_err(|e| InferenceError::model_load(format!("Failed to clear cache: {}", e)))?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> Result<CacheStats> {
        let mut total_size = 0;
        let mut model_count = 0;

        if self.config.cache_dir.exists() {
            let mut entries = fs::read_dir(&self.config.cache_dir).await
                .map_err(|e| InferenceError::model_load(format!("Failed to read cache dir: {}", e)))?;

            while let Some(entry) = entries.next_entry().await
                .map_err(|e| InferenceError::model_load(format!("Failed to read cache entry: {}", e)))? {
                
                if entry.file_type().await.map_err(|e| InferenceError::model_load(format!("Failed to get file type: {}", e)))?.is_dir() {
                    model_count += 1;
                    total_size += dir_size(&entry.path()).await?;
                }
            }
        }

        Ok(CacheStats {
            total_size,
            model_count,
            max_size: self.config.max_cache_size,
        })
    }
}

/// HuggingFace model configuration format
#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceModelConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: Option<usize>,
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    model_type: String,
}

impl HuggingFaceModelConfig {
    fn get_parameter_count(&self) -> usize {
        // Rough estimation based on typical transformer architecture
        let layers = self.num_hidden_layers.unwrap_or(12);
        let hidden = self.hidden_size;
        let vocab = self.vocab_size;
        
        // Embedding + hidden layers + output projection
        vocab * hidden + layers * (hidden * hidden * 4) + hidden * vocab
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_size: u64,
    pub model_count: usize,
    pub max_size: u64,
}

/// Calculate directory size recursively
fn dir_size_sync(path: &Path) -> Result<u64> {
    let mut total = 0;
    let entries = std::fs::read_dir(path)
        .map_err(|e| InferenceError::model_load(format!("Failed to read directory: {}", e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| InferenceError::model_load(format!("Failed to read entry: {}", e)))?;
        
        let metadata = entry.metadata()
            .map_err(|e| InferenceError::model_load(format!("Failed to get metadata: {}", e)))?;

        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += dir_size_sync(&entry.path())?;
        }
    }

    Ok(total)
}

async fn dir_size(path: &Path) -> Result<u64> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        dir_size_sync(&path)
    }).await
    .map_err(|e| InferenceError::model_load(format!("Task join error: {}", e)))?
}

impl Default for HuggingFaceLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create default HuggingFace loader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_repo() {
        let repo = ModelRepo::new("microsoft", "bitnet-b1.58-large");
        assert_eq!(repo.repo_id(), "microsoft/bitnet-b1.58-large");

        let repo_with_revision = repo.with_revision("v1.0");
        assert_eq!(repo_with_revision.revision, Some("v1.0".to_string()));
    }

    #[tokio::test]
    async fn test_cache_dir_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = HuggingFaceConfig {
            cache_dir: temp_dir.path().join("test_cache"),
            ..Default::default()
        };

        let loader = HuggingFaceLoader::with_config(config).unwrap();
        let repo = ModelRepo::new("test", "model");
        
        let cache_dir = loader.get_model_cache_dir(&repo);
        assert!(cache_dir.to_string_lossy().contains("test/model/main"));
    }
}
