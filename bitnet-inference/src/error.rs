//! Error types and handling for the BitNet inference engine.

use std::fmt;

/// Main error type for inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Batch processing error: {0}")]
    BatchProcessingError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("Core operation error: {0}")]
    CoreError(#[from] bitnet_core::BitNetError),

    #[error("Quantization error: {0}")]
    QuantError(#[from] bitnet_quant::QuantizationError),

    #[cfg(feature = "metal")]
    #[error("Metal backend error: {0}")]
    MetalError(#[from] bitnet_metal::MetalError),
}

impl InferenceError {
    /// Create a new model loading error.
    pub fn model_load<S: Into<String>>(msg: S) -> Self {
        InferenceError::ModelLoadError(msg.into())
    }

    /// Create a new device error.
    pub fn device<S: Into<String>>(msg: S) -> Self {
        InferenceError::DeviceError(msg.into())
    }

    /// Create a new batch processing error.
    pub fn batch_processing<S: Into<String>>(msg: S) -> Self {
        InferenceError::BatchProcessingError(msg.into())
    }

    /// Create a new memory error.
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        InferenceError::MemoryError(msg.into())
    }

    /// Create a new optimization error.
    pub fn optimization<S: Into<String>>(msg: S) -> Self {
        InferenceError::OptimizationError(msg.into())
    }

    /// Create a new cache error.
    pub fn cache<S: Into<String>>(msg: S) -> Self {
        InferenceError::CacheError(msg.into())
    }

    /// Create a new configuration error.
    pub fn config<S: Into<String>>(msg: S) -> Self {
        InferenceError::ConfigError(msg.into())
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            InferenceError::MemoryError(_) => true,
            InferenceError::CacheError(_) => true,
            InferenceError::BatchProcessingError(_) => true,
            _ => false,
        }
    }
}

/// Result type alias for inference operations.
pub type Result<T> = std::result::Result<T, InferenceError>;
