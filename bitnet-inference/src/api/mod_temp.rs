//! High-level API for the BitNet inference engine.

pub mod simple;
pub mod builder;

use crate::{Result, InferenceError};
use crate::engine::{InferenceBackend, InferenceContext, OptimizationLevel, Model, CpuInferenceBackend, DeviceSelector, SelectionStrategy};
use crate::cache::{ModelCache, CacheConfig};
use bitnet_core::{Device, Tensor};
use std::sync::Arc;
use std::path::Path;

/// Main inference engine providing high-level APIs.
pub struct InferenceEngine {
    backend: Box<dyn InferenceBackend>,
    context: InferenceContext,
    model_cache: Arc<ModelCache>,
    config: EngineConfig,
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

        Ok(Self {
            backend,
            context,
            model_cache,
            config,
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
                weights: crate::engine::model_loader::ModelWeights {
                    layer_weights: std::collections::HashMap::new(),
                    total_size: 0,
                },
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
    pub fn cache_stats(&self) -> crate::cache::model_cache::CacheStats {
        self.model_cache.stats()
    }

    /// Get current memory usage.
    pub fn memory_usage(&self) -> usize {
        self.backend.get_memory_usage() + self.model_cache.current_memory_usage()
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
        match device {
            Device::Cpu => {
                let backend = CpuInferenceBackend::new()?;
                Ok(Box::new(backend))
            }
            
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // TODO: Implement Metal backend when available
                    tracing::warn!("Metal backend not yet implemented, falling back to CPU");
                    let backend = CpuInferenceBackend::new()?;
                    Ok(Box::new(backend))
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
