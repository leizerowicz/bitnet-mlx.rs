//! Configuration management for BitNet training
//!
//! This module provides comprehensive configuration management for BitNet model training,
//! including model architecture, training parameters, device settings, and validation.

pub mod model_config;

// Re-export main types for easier access
pub use model_config::{
    ModelConfig, 
    ModelConfigManager, 
    ArchitectureConfig, 
    TrainingModelConfig,
    TokenizerConfig,
    DeviceConfig,
    MemoryConfig,
    ModelSize,
    ValidationResult
};