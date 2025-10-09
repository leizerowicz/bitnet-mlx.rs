//! Core inference engine components.

pub mod batch_processor;
pub mod model_loader;
pub mod execution_context;
pub mod cpu_backend;
pub mod device_selector;
pub mod gpu_memory_optimizer;
pub mod zero_copy_loader;
pub mod dynamic_batching;
pub mod parallel_processor;
pub mod weight_conversion;
pub mod layer_factory;
pub mod inference_integration;
pub mod architecture_mapping;
pub mod ternary_operations;
pub mod transformer_layers;
pub mod forward_pass_pipeline;
pub mod sampling;

// Week 3 Advanced GPU Optimization
// pub mod advanced_gpu_backend;  // Disabled - needs Metal integration fixes

#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "mlx")]
pub mod mlx_backend;

pub use batch_processor::{BatchProcessor, BatchConfig, BatchProcessorStats};
pub use model_loader::{ModelLoader, LoadedModel, ModelMetadata, ModelArchitecture as LoaderArchitecture, LayerDefinition, LayerType as LoaderLayerType, LayerParameters as LoaderLayerParameters, ModelWeights};
pub use execution_context::{ExecutionContext, ExecutionConfig, ExecutionStats};
pub use cpu_backend::CpuInferenceBackend;
pub use device_selector::{DeviceSelector, SelectionStrategy, DeviceCapabilities};
pub use gpu_memory_optimizer::{GPUMemoryManager, GPUAllocation, MemoryStats};
pub use zero_copy_loader::{ZeroCopyModelLoader, MmapModel, ModelHeader, WeightLayout};
pub use dynamic_batching::{DynamicBatchProcessor, MemoryMonitor, PerformanceTracker, DynamicBatchStats, MemoryStats as DynamicMemoryStats, PerformanceStats};
pub use layer_factory::{LayerFactory, LayerFactoryBuilder};
pub use inference_integration::{InferenceIntegration, ExecutableModel, LayerOperation, ModelInfo};
pub use parallel_processor::{ParallelInferenceProcessor, InferenceTask, InferenceResult, ParallelConfig, ParallelProcessorStats, WorkerStats};
pub use weight_conversion::{WeightConverter, WeightArrays, ConvertedWeights};
pub use architecture_mapping::{ArchitectureMapper, LayerPattern, ExecutionGraphBuilder};
pub use ternary_operations::{TernaryProcessor, TernaryConfig, TernaryStats};
pub use transformer_layers::{
    TransformerConfig, TransformerStats, BitLinearLayer, RoPEEmbedding, 
    ReLUSquaredActivation, SubLNNormalization, MultiHeadAttention, 
    FeedForwardNetwork, TransformerBlock
};
pub use forward_pass_pipeline::{
    ForwardPassPipeline, ForwardPassConfig, ForwardPassStats,
    BenchmarkResults, ValidationResults
};
pub use sampling::{
    TokenSampler, SamplingConfig, SamplingStats, SamplingPresets,
    BatchSampler
};

// Week 3 Advanced GPU Optimization exports
// pub use advanced_gpu_backend::{AdvancedGPUBackend, AdvancedGPUConfig, PerformanceStatistics, MultiGPUState, DeviceCapability, AsyncMemoryPipeline, PerformanceMonitor};  // Disabled - needs fixes

#[cfg(feature = "metal")]
pub use metal_backend::MetalInferenceBackend;

#[cfg(feature = "mlx")]
pub use mlx_backend::MLXInferenceBackend;

use crate::Result;
use bitnet_core::{Device, Tensor};
use std::sync::Arc;

/// Configuration for inference operations.
#[derive(Clone)]
pub struct InferenceContext {
    /// Target device for computation
    pub device: Device,
    /// Memory pool for efficient allocation
    pub memory_pool: Arc<dyn MemoryPool>,
    /// Maximum batch size for processing
    pub batch_size: usize,
    /// Optimization level to apply
    pub optimization_level: OptimizationLevel,
}

impl InferenceContext {
    /// Create a new inference context with default settings.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            memory_pool: Arc::new(DefaultMemoryPool::new()),
            batch_size: 32,
            optimization_level: OptimizationLevel::Basic,
        }
    }

    /// Set the batch size for this context.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the optimization level for this context.
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

/// Optimization levels available for inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization - fastest compilation
    None,
    /// Basic optimizations - balanced performance
    Basic,
    /// Aggressive optimizations - best performance
    Aggressive,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Basic
    }
}

/// Trait for inference backend implementations.
pub trait InferenceBackend: Send + Sync {
    /// Execute a batch of tensors and return the results.
    fn execute_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
    
    /// Optimize a model for this backend.
    fn optimize_model(&mut self, model: &Model) -> Result<()>;
    
    /// Get current memory usage in bytes.
    fn get_memory_usage(&self) -> usize;
    
    /// Get backend capabilities.
    fn capabilities(&self) -> BackendCapabilities;
}

/// Capabilities of an inference backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub supports_batching: bool,
    pub supports_streaming: bool,
    pub max_batch_size: usize,
    pub memory_limit: Option<usize>,
    pub device_type: Device,
}

/// Memory pool trait for efficient allocation.
pub trait MemoryPool: Send + Sync {
    /// Allocate memory for a tensor.
    fn allocate(&self, size: usize) -> Result<Box<[u8]>>;
    
    /// Get current memory usage.
    fn usage(&self) -> usize;
    
    /// Get maximum memory capacity.
    fn capacity(&self) -> usize;
}

/// Default memory pool implementation.
struct DefaultMemoryPool {
    // Implementation details would go here
}

impl DefaultMemoryPool {
    fn new() -> Self {
        Self {}
    }
}

impl MemoryPool for DefaultMemoryPool {
    fn allocate(&self, size: usize) -> Result<Box<[u8]>> {
        Ok(vec![0u8; size].into_boxed_slice())
    }

    fn usage(&self) -> usize {
        // TODO: Implement actual usage tracking
        0
    }

    fn capacity(&self) -> usize {
        // TODO: Implement actual capacity tracking
        usize::MAX
    }
}

/// Model representation with detailed architecture information.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model {
    /// Model identifier
    pub name: String,
    /// Model version
    pub version: String,
    /// Input tensor dimensions
    pub input_dim: usize,
    /// Output tensor dimensions  
    pub output_dim: usize,
    /// Model architecture details
    pub architecture: ModelArchitecture,
    /// Model parameters
    pub parameter_count: usize,
    /// Quantization configuration
    pub quantization_config: QuantizationConfig,
}

/// Architecture definition for inference models.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelArchitecture {
    /// BitLinear architecture with layer configurations
    BitLinear {
        /// Layer definitions
        layers: Vec<LayerConfig>,
        /// Attention heads (for transformer models)
        attention_heads: Option<usize>,
        /// Hidden dimensions
        hidden_dim: usize,
    },
    /// Quantized models with specific precision
    Quantized {
        /// Precision bits (1, 2, 4, 8)
        precision: u8,
        /// Layer configurations
        layers: Vec<LayerConfig>,
    },
    /// Hybrid models combining different quantization approaches
    Hybrid {
        /// BitLinear layer count
        bitlinear_layers: usize,
        /// Quantized layer count
        quantized_layers: usize,
        /// Layer mapping
        layer_configs: Vec<LayerConfig>,
    },
}

/// Configuration for individual layers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerConfig {
    /// Layer identifier
    pub id: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Layer-specific parameters
    pub parameters: LayerParameters,
}

/// Types of layers in the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    /// BitLinear layer
    BitLinear,
    /// RMSNorm normalization
    RMSNorm,
    /// SwiGLU activation
    SwiGLU,
    /// Embedding layer
    Embedding,
    /// Linear projection
    Linear,
    /// Multi-head attention
    Attention,
}

/// Parameters for specific layer types.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LayerParameters {
    /// BitLinear parameters
    BitLinear {
        weight_bits: u8,
        activation_bits: u8,
    },
    /// RMSNorm parameters
    RMSNorm {
        eps: f32,
    },
    /// SwiGLU parameters
    SwiGLU {
        hidden_dim: usize,
    },
    /// Embedding parameters
    Embedding {
        vocab_size: usize,
        embedding_dim: usize,
    },
    /// Linear layer parameters
    Linear {
        bias: bool,
    },
    /// Attention parameters
    Attention {
        num_heads: usize,
        head_dim: usize,
        dropout: f32,
    },
}

/// Quantization configuration for the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizationConfig {
    /// Weight quantization bits
    pub weight_bits: u8,
    /// Activation quantization bits
    pub activation_bits: u8,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Per-channel quantization
    pub per_channel: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            weight_bits: 1,
            activation_bits: 8,
            symmetric: true,
            per_channel: true,
        }
    }
}

impl Model {
    /// Create a new model with basic configuration.
    pub fn new(name: String, version: String, input_dim: usize, output_dim: usize) -> Self {
        Self {
            name,
            version,
            input_dim,
            output_dim,
            architecture: ModelArchitecture::BitLinear {
                layers: Vec::new(),
                attention_heads: None,
                hidden_dim: input_dim,
            },
            parameter_count: 0,
            quantization_config: QuantizationConfig::default(),
        }
    }

    /// Add a layer to the model architecture.
    pub fn add_layer(&mut self, layer_config: LayerConfig) {
        match &mut self.architecture {
            ModelArchitecture::BitLinear { layers, .. } => {
                layers.push(layer_config);
            }
            ModelArchitecture::Quantized { layers, .. } => {
                layers.push(layer_config);
            }
            ModelArchitecture::Hybrid { layer_configs, .. } => {
                layer_configs.push(layer_config);
            }
        }
    }

    /// Get the total number of layers.
    pub fn layer_count(&self) -> usize {
        match &self.architecture {
            ModelArchitecture::BitLinear { layers, .. } => layers.len(),
            ModelArchitecture::Quantized { layers, .. } => layers.len(),
            ModelArchitecture::Hybrid { layer_configs, .. } => layer_configs.len(),
        }
    }

    /// Check if the model supports batch processing.
    pub fn supports_batching(&self) -> bool {
        // All our architectures support batching
        true
    }

    /// Estimate memory usage for this model.
    pub fn estimated_memory_usage(&self) -> usize {
        // Rough estimate based on parameter count and quantization
        let bits_per_param = self.quantization_config.weight_bits as usize;
        (self.parameter_count * bits_per_param) / 8
    }

    /// Get input dimension for GPU memory allocation
    pub fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension for GPU memory allocation
    pub fn get_output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get total weight count for GPU buffer sizing
    pub fn get_total_weight_count(&self) -> usize {
        self.parameter_count
    }

    /// Get model ID for caching purposes
    pub fn get_model_id(&self) -> String {
        format!("{}_{}", self.name, self.version)
    }
}
