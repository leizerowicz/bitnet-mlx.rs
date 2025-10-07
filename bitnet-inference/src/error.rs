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
    QuantError(#[from] bitnet_quant::quantization::utils::QuantizationError),

    #[error("Candle tensor error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[cfg(feature = "metal")]
    #[error("Metal backend error: {0}")]
    MetalError(#[from] bitnet_metal::MetalError),

    #[error("Resource error: {0}")]
    ResourceError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    #[error("Inference execution error: {0}")]
    InferenceExecutionError(String),

    #[error("Text generation error: {message}")]
    GenerationError { message: String },

    #[error("Tokenization error: {message}")]
    TokenizationError { message: String },

    #[error("Tensor error: {message}")]
    TensorError { message: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
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

    /// Create a new resource error.
    pub fn resource<S: Into<String>>(msg: S) -> Self {
        InferenceError::ResourceError(msg.into())
    }

    /// Create a new serialization error.
    pub fn serialization<S: Into<String>>(msg: S) -> Self {
        InferenceError::SerializationError(bincode::Error::new(bincode::ErrorKind::Io(
            std::io::Error::new(std::io::ErrorKind::Other, msg.into())
        )))
    }

    /// Create a new unsupported operation error.
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        InferenceError::UnsupportedOperation(msg.into())
    }

    /// Create a new concurrency error.
    pub fn concurrency<S: Into<String>>(msg: S) -> Self {
        InferenceError::ConcurrencyError(msg.into())
    }

    /// Create a new inference execution error.
    pub fn inference<S: Into<String>>(msg: S) -> Self {
        InferenceError::InferenceExecutionError(msg.into())
    }

    /// Create a new generation error.
    pub fn generation<S: Into<String>>(msg: S) -> Self {
        InferenceError::GenerationError { message: msg.into() }
    }

    /// Create a new tokenization error.
    pub fn tokenization<S: Into<String>>(msg: S) -> Self {
        InferenceError::TokenizationError { message: msg.into() }
    }

    /// Create a new tensor error.
    pub fn tensor<S: Into<String>>(msg: S) -> Self {
        InferenceError::TensorError { message: msg.into() }
    }

    /// Create a new configuration error.
    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        InferenceError::ConfigurationError { message: msg.into() }
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
