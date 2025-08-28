//! Builder pattern API for advanced inference engine configuration.

use crate::{Result, InferenceError};
use crate::api::{InferenceEngine, EngineConfig};
use crate::engine::OptimizationLevel;
use crate::cache::CacheConfig;
use bitnet_core::Device;
use std::path::PathBuf;

/// Builder for creating customized inference engines.
pub struct InferenceEngineBuilder {
    device: Option<Device>,
    batch_size: Option<usize>,
    optimization_level: OptimizationLevel,
    enable_caching: bool,
    cache_config: CacheConfig,
    custom_operators: Vec<Box<dyn CustomOperator>>,
}

impl Default for InferenceEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngineBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            device: None,
            batch_size: None,
            optimization_level: OptimizationLevel::Basic,
            enable_caching: true,
            cache_config: CacheConfig::default(),
            custom_operators: Vec::new(),
        }
    }

    /// Set the target device for inference.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the batch size for inference operations.
    pub fn batch_size(mut self, size: usize) -> Self {
        if size == 0 {
            panic!("Batch size must be greater than 0");
        }
        self.batch_size = Some(size);
        self
    }

    /// Set the optimization level.
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Enable or disable model caching.
    pub fn enable_caching(mut self, enable: bool) -> Self {
        self.enable_caching = enable;
        self
    }

    /// Set the memory pool size for caching.
    pub fn cache_memory_size(mut self, size: MemorySize) -> Self {
        self.cache_config.max_memory = size.bytes();
        self
    }

    /// Set the maximum number of models to cache.
    pub fn max_cached_models(mut self, count: usize) -> Self {
        self.cache_config.max_models = count;
        self
    }

    /// Set the cache directory.
    pub fn cache_directory<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.cache_config.cache_dir = path.into();
        self
    }

    /// Enable persistent caching to disk.
    pub fn persistent_cache(mut self, enable: bool) -> Self {
        self.cache_config.persistent = enable;
        self
    }

    /// Add custom operators for specialized inference operations.
    pub fn with_custom_operators(mut self, ops: Vec<Box<dyn CustomOperator>>) -> Self {
        self.custom_operators = ops;
        self
    }

    /// Configure the engine for high-performance scenarios.
    pub fn for_high_performance(mut self) -> Self {
        self.optimization_level = OptimizationLevel::Aggressive;
        self.batch_size = Some(64);
        self.cache_config.max_memory = 4 * 1024 * 1024 * 1024; // 4GB
        self.cache_config.max_models = 20;
        self
    }

    /// Configure the engine for memory-constrained scenarios.
    pub fn for_low_memory(mut self) -> Self {
        self.optimization_level = OptimizationLevel::None;
        self.batch_size = Some(4);
        self.cache_config.max_memory = 256 * 1024 * 1024; // 256MB
        self.cache_config.max_models = 2;
        self
    }

    /// Configure the engine for edge deployment.
    pub fn for_edge_deployment(mut self) -> Self {
        self.optimization_level = OptimizationLevel::Basic;
        self.batch_size = Some(1);
        self.cache_config.max_memory = 128 * 1024 * 1024; // 128MB
        self.cache_config.max_models = 1;
        self.cache_config.persistent = false; // Avoid disk I/O on edge
        self
    }

    /// Configure the engine for server deployment.
    pub fn for_server_deployment(mut self) -> Self {
        self.optimization_level = OptimizationLevel::Aggressive;
        self.batch_size = Some(128);
        self.cache_config.max_memory = 8 * 1024 * 1024 * 1024; // 8GB
        self.cache_config.max_models = 50;
        self.cache_config.persistent = true;
        self
    }

    /// Build the inference engine with the configured settings.
    pub async fn build(self) -> Result<InferenceEngine> {
        let device = match self.device {
            Some(device) => device,
            None => InferenceEngine::select_optimal_device()?,
        };

        let batch_size = self.batch_size.unwrap_or_else(|| {
            // Auto-select batch size based on device
            match device {
                Device::Cpu => 16,
                Device::Metal(_) => 32,
                Device::Cuda(_) => 32,
                // Device::MLX => 64, // TODO: Add when MLX support is implemented
            }
        });

        let config = EngineConfig {
            device,
            batch_size,
            optimization_level: self.optimization_level,
            enable_caching: self.enable_caching,
            cache_config: self.cache_config,
        };

        let mut engine = InferenceEngine::with_config(config).await?;

        // Register custom operators if any
        for operator in self.custom_operators {
            engine.register_custom_operator(operator)?;
        }

        Ok(engine)
    }

    /// Validate the current configuration.
    pub fn validate(&self) -> Result<()> {
        if let Some(batch_size) = self.batch_size {
            if batch_size > 1024 {
                return Err(InferenceError::config(
                    "Batch size too large (max 1024)"
                ));
            }
        }

        if self.cache_config.max_memory < 64 * 1024 * 1024 {
            return Err(InferenceError::config(
                "Cache memory size too small (min 64MB)"
            ));
        }

        if self.cache_config.max_models == 0 {
            return Err(InferenceError::config(
                "Maximum cached models must be at least 1"
            ));
        }

        Ok(())
    }

    /// Get the estimated memory usage of this configuration.
    pub fn estimated_memory_usage(&self) -> usize {
        let base_engine_memory = 64 * 1024 * 1024; // 64MB base
        let cache_memory = self.cache_config.max_memory;
        let batch_memory = self.batch_size.unwrap_or(32) * 1024 * 1024; // 1MB per item estimate
        
        base_engine_memory + cache_memory + batch_memory
    }
}

/// Memory size specification for builder configuration.
#[derive(Debug, Clone, Copy)]
pub enum MemorySize {
    MB(usize),
    GB(usize),
}

impl MemorySize {
    /// Convert to bytes.
    pub fn bytes(self) -> usize {
        match self {
            MemorySize::MB(mb) => mb * 1024 * 1024,
            MemorySize::GB(gb) => gb * 1024 * 1024 * 1024,
        }
    }
}

/// Trait for custom operators that can be added to the inference engine.
pub trait CustomOperator: Send + Sync {
    /// Name of the operator.
    fn name(&self) -> &str;
    
    /// Execute the custom operator.
    fn execute(&self, inputs: &[bitnet_core::Tensor]) -> Result<Vec<bitnet_core::Tensor>>;
    
    /// Get operator metadata.
    fn metadata(&self) -> CustomOperatorMetadata;
}

/// Metadata for custom operators.
#[derive(Debug, Clone)]
pub struct CustomOperatorMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
}

impl InferenceEngine {
    /// Create a builder for configuring a new inference engine.
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Register a custom operator with this engine.
    fn register_custom_operator(&mut self, _operator: Box<dyn CustomOperator>) -> Result<()> {
        // This would register the custom operator with the backend
        // For now, this is a placeholder
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_configuration() {
        let builder = InferenceEngineBuilder::new()
            .device(Device::Cpu)
            .batch_size(32)
            .optimization_level(OptimizationLevel::Aggressive)
            .cache_memory_size(MemorySize::GB(2));

        // Can't test device equality since Device doesn't implement PartialEq
        assert_eq!(builder.batch_size, Some(32));
        assert_eq!(builder.optimization_level, OptimizationLevel::Aggressive);
        assert_eq!(builder.cache_config.max_memory, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_size_conversion() {
        assert_eq!(MemorySize::MB(512).bytes(), 512 * 1024 * 1024);
        assert_eq!(MemorySize::GB(2).bytes(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_builder_validation() {
        let builder = InferenceEngineBuilder::new()
            .batch_size(32)
            .cache_memory_size(MemorySize::MB(128));

        assert!(builder.validate().is_ok());

        let invalid_builder = InferenceEngineBuilder::new()
            .batch_size(2000); // Too large

        assert!(invalid_builder.validate().is_err());
    }

    #[test]
    fn test_preset_configurations() {
        let high_perf = InferenceEngineBuilder::new().for_high_performance();
        assert_eq!(high_perf.optimization_level, OptimizationLevel::Aggressive);
        assert_eq!(high_perf.batch_size, Some(64));

        let low_mem = InferenceEngineBuilder::new().for_low_memory();
        assert_eq!(low_mem.optimization_level, OptimizationLevel::None);
        assert_eq!(low_mem.batch_size, Some(4));

        let edge = InferenceEngineBuilder::new().for_edge_deployment();
        assert_eq!(edge.batch_size, Some(1));
        assert!(!edge.cache_config.persistent);
    }

    #[test]
    fn test_estimated_memory_usage() {
        let builder = InferenceEngineBuilder::new()
            .batch_size(32)
            .cache_memory_size(MemorySize::GB(1));

        let estimated = builder.estimated_memory_usage();
        assert!(estimated > 1024 * 1024 * 1024); // Should be more than 1GB
    }
}
