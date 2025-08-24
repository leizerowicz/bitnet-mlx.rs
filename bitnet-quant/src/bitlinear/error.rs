//! Error Types for BitLinear Operations
//!
//! This module provides comprehensive error handling for BitLinear layer operations,
//! including quantization, caching, memory management, and tensor operations.

use thiserror::Error;

/// Errors that can occur in BitLinear layer operations
#[derive(Error, Debug)]
pub enum BitLinearError {
    /// Tensor operation failed
    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    /// Quantization operation failed
    #[error("Quantization failed: {0}")]
    QuantizationError(String),

    /// Cache operation failed
    #[error("Cache operation failed: {0}")]
    CacheError(#[from] crate::bitlinear::cache::CacheError),

    /// Memory management error
    #[error("Memory management error: {0}")]
    MemoryError(String),

    /// Device-related error
    #[error("Device operation error: {0}")]
    DeviceError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Shape mismatch error
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type mismatch error
    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DataTypeMismatch { expected: String, actual: String },

    /// Invalid layer configuration
    #[error("Invalid layer configuration: {0}")]
    InvalidConfig(String),

    /// Layer not initialized
    #[error("Layer not initialized: {0}")]
    NotInitialized(String),

    /// Forward pass failed
    #[error("Forward pass failed: {0}")]
    ForwardError(String),

    /// Backward pass failed
    #[error("Backward pass failed: {0}")]
    BackwardError(String),

    /// Memory pressure error
    #[error("Memory pressure error: {0}")]
    MemoryPressureError(String),

    /// SIMD operation failed
    #[error("SIMD operation failed: {0}")]
    SimdError(String),

    /// Threading error
    #[error("Threading error: {0}")]
    ThreadingError(String),

    /// IO operation failed
    #[error("IO operation failed: {0}")]
    IoError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl BitLinearError {
    /// Create a tensor error
    pub fn tensor<S: Into<String>>(message: S) -> Self {
        BitLinearError::TensorError(message.into())
    }

    /// Create a quantization error
    pub fn quantization<S: Into<String>>(message: S) -> Self {
        BitLinearError::QuantizationError(message.into())
    }

    /// Create a cache error
    pub fn cache_error(message: impl Into<String>) -> Self {
        let cache_err = crate::bitlinear::cache::CacheError::InvalidKey(message.into());
        BitLinearError::CacheError(cache_err)
    }

    /// Create a cache error from lock failure
    pub fn cache_lock_error(lock_type: &str) -> Self {
        let cache_err = crate::bitlinear::cache::CacheError::CorruptionDetected(format!(
            "Failed to acquire {lock_type} lock"
        ));
        BitLinearError::CacheError(cache_err)
    }

    /// Create a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        BitLinearError::MemoryError(message.into())
    }

    /// Create a device error
    pub fn device<S: Into<String>>(message: S) -> Self {
        BitLinearError::DeviceError(message.into())
    }

    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        BitLinearError::ConfigError(message.into())
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        BitLinearError::ShapeMismatch { expected, actual }
    }

    /// Create a data type mismatch error
    pub fn dtype_mismatch<S: Into<String>>(expected: S, actual: S) -> Self {
        BitLinearError::DataTypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an invalid config error
    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        BitLinearError::InvalidConfig(message.into())
    }

    /// Create a not initialized error
    pub fn not_initialized<S: Into<String>>(message: S) -> Self {
        BitLinearError::NotInitialized(message.into())
    }

    /// Create a forward error
    pub fn forward<S: Into<String>>(message: S) -> Self {
        BitLinearError::ForwardError(message.into())
    }

    /// Create a backward error
    pub fn backward<S: Into<String>>(message: S) -> Self {
        BitLinearError::BackwardError(message.into())
    }

    /// Create a memory pressure error
    pub fn memory_pressure<S: Into<String>>(message: S) -> Self {
        BitLinearError::MemoryPressureError(message.into())
    }

    /// Create a SIMD error
    pub fn simd<S: Into<String>>(message: S) -> Self {
        BitLinearError::SimdError(message.into())
    }

    /// Create a threading error
    pub fn threading<S: Into<String>>(message: S) -> Self {
        BitLinearError::ThreadingError(message.into())
    }

    /// Create an IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        BitLinearError::IoError(message.into())
    }

    /// Create a serialization error
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        BitLinearError::SerializationError(message.into())
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        BitLinearError::InternalError(message.into())
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            BitLinearError::MemoryPressureError(_) => true,
            BitLinearError::CacheError(_) => true,
            BitLinearError::ThreadingError(_) => true,
            BitLinearError::IoError(_) => true,
            _ => false,
        }
    }

    /// Check if this is a configuration-related error
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            BitLinearError::ConfigError(_)
                | BitLinearError::InvalidConfig(_)
                | BitLinearError::ShapeMismatch { .. }
                | BitLinearError::DataTypeMismatch { .. }
        )
    }

    /// Check if this is a runtime error
    pub fn is_runtime_error(&self) -> bool {
        matches!(
            self,
            BitLinearError::TensorError(_)
                | BitLinearError::QuantizationError(_)
                | BitLinearError::ForwardError(_)
                | BitLinearError::BackwardError(_)
                | BitLinearError::SimdError(_)
        )
    }
}

/// Result type for BitLinear operations
pub type BitLinearResult<T> = std::result::Result<T, BitLinearError>;

// Conversion implementations for common error types

impl From<candle_core::Error> for BitLinearError {
    fn from(err: candle_core::Error) -> Self {
        BitLinearError::TensorError(err.to_string())
    }
}

impl From<crate::quantization::QuantizationError> for BitLinearError {
    fn from(err: crate::quantization::QuantizationError) -> Self {
        BitLinearError::QuantizationError(err.to_string())
    }
}

impl From<bitnet_core::memory::MemoryError> for BitLinearError {
    fn from(err: bitnet_core::memory::MemoryError) -> Self {
        BitLinearError::MemoryError(err.to_string())
    }
}

impl From<bitnet_core::device::DeviceError> for BitLinearError {
    fn from(err: bitnet_core::device::DeviceError) -> Self {
        BitLinearError::DeviceError(err.to_string())
    }
}

impl From<std::io::Error> for BitLinearError {
    fn from(err: std::io::Error) -> Self {
        BitLinearError::IoError(err.to_string())
    }
}

/// Error context trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> BitLinearResult<T>
    where
        F: FnOnce() -> String;

    /// Add static context to an error
    fn context(self, msg: &'static str) -> BitLinearResult<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<BitLinearError>,
{
    fn with_context<F>(self, f: F) -> BitLinearResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base_error = e.into();
            let context = f();
            match base_error {
                BitLinearError::TensorError(msg) => {
                    BitLinearError::TensorError(format!("{context}: {msg}"))
                }
                BitLinearError::QuantizationError(msg) => {
                    BitLinearError::QuantizationError(format!("{context}: {msg}"))
                }
                BitLinearError::CacheError(cache_err) => {
                    let msg = cache_err.to_string();
                    let new_cache_err = crate::bitlinear::cache::CacheError::InvalidKey(format!(
                        "{context}: {msg}"
                    ));
                    BitLinearError::CacheError(new_cache_err)
                }
                BitLinearError::MemoryError(msg) => {
                    BitLinearError::MemoryError(format!("{context}: {msg}"))
                }
                other => other,
            }
        })
    }

    fn context(self, msg: &'static str) -> BitLinearResult<T> {
        self.with_context(|| msg.to_string())
    }
}

/// Macro for creating BitLinear errors with formatting
#[macro_export]
macro_rules! bitlinear_error {
    ($kind:ident, $($arg:tt)*) => {
        BitLinearError::$kind(format!($($arg)*))
    };
}

/// Macro for ensuring conditions in BitLinear operations
#[macro_export]
macro_rules! bitlinear_ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::bitlinear::error::BitLinearError::$kind(format!($($arg)*)));
        }
    };
}

/// Macro for early return on error with context
#[macro_export]
macro_rules! bitlinear_try {
    ($expr:expr, $context:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => return Err(err.into()).context($context),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = BitLinearError::tensor("Test tensor error");
        assert!(matches!(error, BitLinearError::TensorError(_)));
        assert_eq!(
            error.to_string(),
            "Tensor operation failed: Test tensor error"
        );
    }

    #[test]
    fn test_error_classification() {
        let config_error = BitLinearError::config("Test config");
        assert!(config_error.is_config_error());
        assert!(!config_error.is_runtime_error());

        let tensor_error = BitLinearError::tensor("Test tensor");
        assert!(tensor_error.is_runtime_error());
        assert!(!tensor_error.is_config_error());

        let memory_error = BitLinearError::memory_pressure("Test pressure");
        assert!(memory_error.is_recoverable());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let error = BitLinearError::shape_mismatch(vec![3, 4], vec![2, 5]);
        assert!(matches!(error, BitLinearError::ShapeMismatch { .. }));
        assert!(error.is_config_error());
    }

    #[test]
    fn test_error_context() {
        let result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));

        let with_context = result.context("loading model weights");
        assert!(with_context.is_err());

        let error = with_context.unwrap_err();
        assert!(matches!(error, BitLinearError::IoError(_)));
    }

    #[test]
    fn test_error_macros() -> Result<(), BitLinearError> {
        let error = bitlinear_error!(TensorError, "Value: {}", 42);
        assert_eq!(error.to_string(), "Tensor operation failed: Value: 42");

        // Test ensure macro (would fail compilation if condition is wrong)
        let value = 5;
        bitlinear_ensure!(value > 0, ConfigError, "Value must be positive");
        // This should pass without error
        Ok(())
    }
}
