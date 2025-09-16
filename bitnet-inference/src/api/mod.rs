//! High-level API for the BitNet inference engine.

pub mod simple;
pub mod builder;
pub mod streaming;

// Re-export streaming types for convenience
pub use streaming::{InferenceStream, StreamingConfig};

use crate::{Result, InferenceError};
use crate::engine::{InferenceBackend, InferenceContext, OptimizationLevel, Model, CpuInferenceBackend, DeviceSelector, SelectionStrategy};
use crate::cache::{ModelCache, CacheConfig};
use crate::huggingface::{HuggingFaceLoader, ModelRepo, HuggingFaceConfig};
use bitnet_core::{Device, Tensor};
use std::sync::Arc;
use std::path::Path;

/// Main inference engine providing high-level APIs.
pub struct InferenceEngine {
    backend: Box<dyn InferenceBackend>,
    context: InferenceContext,
    model_cache: Arc<ModelCache>,
    config: EngineConfig,
    hf_loader: HuggingFaceLoader,
}

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub device: Device,
    pub batch_size: usize,
    pub optimization_level: OptimizationLevel,
    pub enable_caching: bool,
    pub cache_config: CacheConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu, // Will be auto-detected in practice
            batch_size: 32,
            optimization_level: OptimizationLevel::Basic,
            enable_caching: true,
            cache_config: CacheConfig::default(),
        }
    }
}

impl InferenceEngine {
    /// Create a new inference engine with automatic device selection.
    pub async fn new() -> Result<Self> {
        let device = Self::select_optimal_device()?;
        Self::with_device(device).await
    }

    /// Create an inference engine for a specific device.
    pub async fn with_device(device: Device) -> Result<Self> {
        let config = EngineConfig {
            device,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create an inference engine with custom configuration.
    pub async fn with_config(config: EngineConfig) -> Result<Self> {
        let device = config.device.clone();
        let backend = Self::create_backend(device.clone())?;
        let context = InferenceContext::new(device)
            .with_batch_size(config.batch_size)
            .with_optimization_level(config.optimization_level);

        let model_cache = if config.enable_caching {
            Arc::new(ModelCache::new(
                config.cache_config.max_models,
                config.cache_config.max_memory,
            ))
        } else {
            // Create a minimal cache even when caching is "disabled"
            Arc::new(ModelCache::new(1, 64 * 1024 * 1024)) // 64MB minimal cache
        };

        let hf_loader = HuggingFaceLoader::new()
            .map_err(|e| InferenceError::model_load(format!("Failed to create HuggingFace loader: {}", e)))?;

        Ok(Self {
            backend,
            context,
            model_cache,
            config,
            hf_loader,
        })
    }

    /// Set the optimization level for this engine.
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.config.optimization_level = level;
        self.context = self.context.with_optimization_level(level);
        self
    }

    /// Load a model from the specified path.
    pub async fn load_model<P: AsRef<Path>>(&self, path: P) -> Result<Arc<Model>> {
        let path = path.as_ref();
        let model_key = path.to_string_lossy().to_string();

        // Try to get from cache or load
        let cached_model = self.model_cache.get_or_load(&model_key, || {
            // This is a placeholder - in a real implementation, we'd use ModelLoader
            // For now, create a simple test model
            Ok(crate::engine::model_loader::LoadedModel {
                metadata: crate::engine::model_loader::ModelMetadata {
                    name: "test_model".to_string(),
                    version: "1.0".to_string(),
                    architecture: "bitnet-1.58b".to_string(),
                    parameter_count: 1_000_000,
                    quantization_bits: 1,
                    input_shape: vec![1, 512],
                    output_shape: vec![1, 30000],
                    extra: std::collections::HashMap::new(),
                },
                architecture: crate::engine::model_loader::ModelArchitecture {
                    layers: vec![],
                    execution_order: vec![],
                },
                weights: {
                    let mut weights = crate::engine::model_loader::ModelWeights::new();
                    weights.layer_weights.insert(0, vec![0u8; 1024]); // Placeholder weights
                    weights.total_size = 1024;
                    weights
                },
                bitnet_config: None, // Placeholder model doesn't have BitNet config
            })
        })?;

        // Convert LoadedModel to Model (placeholder conversion)
        let model = Model {
            name: cached_model.model.metadata.name.clone(),
            version: cached_model.model.metadata.version.clone(),
            input_dim: cached_model.model.metadata.input_shape.get(1).copied().unwrap_or(512),
            output_dim: cached_model.model.metadata.output_shape.get(1).copied().unwrap_or(30000),
            architecture: crate::engine::ModelArchitecture::BitLinear {
                layers: Vec::new(),
                attention_heads: None,
                hidden_dim: 512,
            },
            parameter_count: cached_model.model.metadata.parameter_count,
            quantization_config: crate::engine::QuantizationConfig::default(),
        };

        Ok(Arc::new(model))
    }

    /// Load a model from HuggingFace Hub by repository identifier.
    /// 
    /// # Arguments
    /// * `repo_id` - HuggingFace repository identifier (e.g., "microsoft/bitnet-b1.58-large")
    /// 
    /// # Example
    /// ```no_run
    /// # use bitnet_inference::InferenceEngine;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let engine = InferenceEngine::new().await?;
    /// let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-large").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model_from_hub(&self, repo_id: &str) -> Result<Arc<Model>> {
        let repo = self.parse_repo_id(repo_id)?;
        self.load_model_from_repo(&repo).await
    }

    /// Load a model from HuggingFace Hub with a specific revision.
    /// 
    /// # Arguments
    /// * `repo_id` - HuggingFace repository identifier
    /// * `revision` - Specific revision (branch, tag, or commit)
    pub async fn load_model_from_hub_with_revision(&self, repo_id: &str, revision: &str) -> Result<Arc<Model>> {
        let repo = self.parse_repo_id(repo_id)?
            .with_revision(revision);
        self.load_model_from_repo(&repo).await
    }

    /// Load a model from a ModelRepo specification.
    pub async fn load_model_from_repo(&self, repo: &ModelRepo) -> Result<Arc<Model>> {
        let model_key = format!("hf:{}", repo.repo_id());
        
        // Load using the cache's get_or_load method
        let cached_model = self.model_cache.get_or_load(&model_key, || {
            // This closure will be called only if the model is not in cache
            futures::executor::block_on(async {
                self.hf_loader.load_model(repo).await
            })
        })?;

        // Convert to inference Model format
        let model = self.convert_loaded_model_to_model(cached_model.model)?;
        Ok(Arc::new(model))
    }

    /// Download a model from HuggingFace Hub to local cache.
    /// This is useful for pre-downloading models without loading them into memory.
    pub async fn download_model(&self, repo_id: &str) -> Result<std::path::PathBuf> {
        let repo = self.parse_repo_id(repo_id)?;
        self.hf_loader.download_model(&repo).await
    }

    /// Get HuggingFace cache statistics.
    pub async fn hf_cache_stats(&self) -> Result<crate::huggingface::CacheStats> {
        self.hf_loader.cache_stats().await
    }

    /// Clear HuggingFace model cache.
    pub async fn clear_hf_cache(&self) -> Result<()> {
        self.hf_loader.clear_cache().await
    }

    /// Parse a repository ID string into owner and name components.
    fn parse_repo_id(&self, repo_id: &str) -> Result<ModelRepo> {
        let parts: Vec<&str> = repo_id.split('/').collect();
        if parts.len() != 2 {
            return Err(InferenceError::model_load(
                format!("Invalid repository ID '{}'. Expected format: 'owner/name'", repo_id)
            ));
        }
        Ok(ModelRepo::new(parts[0], parts[1]))
    }

    /// Convert a LoadedModel to the inference Model format.
    fn convert_loaded_model_to_model(&self, loaded_model: crate::engine::model_loader::LoadedModel) -> Result<Model> {
        // Convert the loaded model metadata to the inference model format
        let model = Model {
            name: loaded_model.metadata.name.clone(),
            version: loaded_model.metadata.version.clone(),
            input_dim: loaded_model.metadata.input_shape.get(1).copied().unwrap_or(512),
            output_dim: loaded_model.metadata.output_shape.get(1).copied().unwrap_or(30000),
            architecture: crate::engine::ModelArchitecture::BitLinear {
                layers: loaded_model.architecture.layers.into_iter().map(|layer| {
                    crate::engine::LayerConfig {
                        id: layer.id,
                        layer_type: match layer.layer_type {
                            crate::engine::model_loader::LayerType::BitLinear => crate::engine::LayerType::BitLinear,
                            crate::engine::model_loader::LayerType::RMSNorm => crate::engine::LayerType::RMSNorm,
                            crate::engine::model_loader::LayerType::SwiGLU => crate::engine::LayerType::SwiGLU,
                            crate::engine::model_loader::LayerType::Embedding => crate::engine::LayerType::Embedding,
                            crate::engine::model_loader::LayerType::OutputProjection => crate::engine::LayerType::Linear, // Map to Linear
                        },
                        input_shape: layer.input_dims.clone(),
                        output_shape: layer.output_dims.clone(),
                        parameters: match layer.parameters {
                            crate::engine::model_loader::LayerParameters::BitLinear { weight_bits, activation_bits } => {
                                crate::engine::LayerParameters::BitLinear { weight_bits, activation_bits }
                            },
                            crate::engine::model_loader::LayerParameters::RMSNorm { eps } => {
                                crate::engine::LayerParameters::RMSNorm { eps }
                            },
                            crate::engine::model_loader::LayerParameters::SwiGLU { hidden_dim } => {
                                crate::engine::LayerParameters::SwiGLU { hidden_dim }
                            },
                            crate::engine::model_loader::LayerParameters::Embedding { vocab_size, embedding_dim } => {
                                crate::engine::LayerParameters::Embedding { vocab_size, embedding_dim }
                            },
                            crate::engine::model_loader::LayerParameters::OutputProjection { vocab_size: _ } => {
                                crate::engine::LayerParameters::Linear { 
                                    bias: true,
                                }
                            },
                        },
                    }
                }).collect(),
                attention_heads: Some(8), // Default value, should be inferred from config
                hidden_dim: loaded_model.metadata.input_shape.get(1).copied().unwrap_or(512),
            },
            parameter_count: loaded_model.metadata.parameter_count,
            quantization_config: crate::engine::QuantizationConfig::default(),
        };

        Ok(model)
    }

    /// Run inference on a single input tensor.
    pub async fn infer(&self, _model: &Model, input: &Tensor) -> Result<Tensor> {
        // Execute inference using the configured backend
        let batch_result = self.backend.execute_batch(&[input.clone()])?;
        Ok(batch_result.into_iter().next().unwrap())
    }

    /// Perform batch inference on multiple inputs.
    pub async fn infer_batch(&self, _model: &Model, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Use backend directly for batch processing
        self.backend.execute_batch(inputs)
    }

    /// Get the current device being used.
    pub fn device(&self) -> &Device {
        &self.config.device
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> crate::cache::advanced_model_cache::CacheStats {
        self.model_cache.stats()
    }

    /// Get current memory usage.
    pub fn memory_usage(&self) -> usize {
        let backend_memory = self.backend.get_memory_usage();
        let cache_memory = self.model_cache.current_memory_usage();
        let base_memory = std::mem::size_of::<Self>(); // Base struct size
        
        backend_memory + cache_memory + base_memory
    }

    /// Clear the model cache.
    pub fn clear_cache(&self) {
        self.model_cache.clear();
    }

    /// Select the optimal device for the current system.
    fn select_optimal_device() -> Result<Device> {
        DeviceSelector::select_device(SelectionStrategy::Auto)
    }

    /// Create an appropriate backend for the specified device.
    fn create_backend(device: Device) -> Result<Box<dyn InferenceBackend>> {
        // First try to create the most optimal backend based on system capabilities
        
        // Priority 1: MLX on Apple Silicon (highest performance)
        #[cfg(all(feature = "mlx", target_arch = "aarch64", target_os = "macos"))]
        {
            if DeviceSelector::is_mlx_available() {
                if let Ok(backend) = crate::engine::MLXInferenceBackend::new() {
                    tracing::info!("Using MLX backend for optimal Apple Silicon performance");
                    return Ok(Box::new(backend));
                }
            }
        }

        // Priority 2: Metal GPU acceleration (good performance)
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if DeviceSelector::is_metal_available() {
                if let Ok(backend) = crate::engine::MetalInferenceBackend::new() {
                    tracing::info!("Using Metal backend for GPU acceleration");
                    return Ok(Box::new(backend));
                }
            }
        }

        // Match the specific device request if backends above are not available
        match device {
            Device::Cpu => {
                tracing::info!("Using CPU backend");
                let backend = CpuInferenceBackend::new()?;
                Ok(Box::new(backend))
            }
            
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    match crate::engine::MetalInferenceBackend::new() {
                        Ok(backend) => {
                            tracing::info!("Using Metal backend as requested");
                            Ok(Box::new(backend))
                        }
                        Err(e) => {
                            tracing::warn!("Metal backend requested but failed to initialize: {}, falling back to CPU", e);
                            let backend = CpuInferenceBackend::new()?;
                            Ok(Box::new(backend))
                        }
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(InferenceError::device("Metal support not compiled in"))
                }
            },
            
            Device::Cuda(_) => {
                // TODO: Implement CUDA backend when available
                tracing::warn!("CUDA backend not yet implemented, falling back to CPU");
                let backend = CpuInferenceBackend::new()?;
                Ok(Box::new(backend))
            }
        }
    }

    /// Check if Metal is available.
    #[cfg(feature = "metal")]
    fn is_metal_available() -> bool {
        DeviceSelector::is_device_available(&Device::Cpu) // Simplified for now
            .unwrap_or(false)
    }

    #[cfg(not(feature = "metal"))]
    fn is_metal_available() -> bool {
        false
    }
}
