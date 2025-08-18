//! Calibration error types and result handling
//!
//! This module defines error types and result handling for the calibration system.

use thiserror::Error;

/// Result type for calibration operations
pub type CalibrationResult<T> = Result<T, CalibrationError>;

/// Comprehensive error types for calibration operations
#[derive(Error, Debug)]
pub enum CalibrationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tensor operation error: {0}")]
    Tensor(#[from] candle_core::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Dataset error: {message}")]
    Dataset { message: String },

    #[error("Statistics error: {message}")]
    Statistics { message: String },

    #[error("Sampling error: {message}")]
    Sampling { message: String },

    #[error("Memory error: {message}")]
    Memory { message: String },

    #[error("Streaming error: {message}")]
    Streaming { message: String },

    #[error("Persistence error: {message}")]
    Persistence { message: String },

    #[error("Histogram error: {message}")]
    Histogram { message: String },

    #[error("Invalid path: {path}")]
    InvalidPath { path: String },

    #[error("Insufficient data: expected at least {expected} samples, got {actual}")]
    InsufficientData { expected: usize, actual: usize },

    #[error("Incompatible format: expected {expected}, got {actual}")]
    IncompatibleFormat { expected: String, actual: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Operation timeout: {operation} took longer than {timeout_seconds}s")]
    Timeout { operation: String, timeout_seconds: u64 },

    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },
}

impl CalibrationError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a dataset error
    pub fn dataset(message: impl Into<String>) -> Self {
        Self::Dataset {
            message: message.into(),
        }
    }

    /// Create a statistics error
    pub fn statistics(message: impl Into<String>) -> Self {
        Self::Statistics {
            message: message.into(),
        }
    }

    /// Create a sampling error
    pub fn sampling(message: impl Into<String>) -> Self {
        Self::Sampling {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a streaming error
    pub fn streaming(message: impl Into<String>) -> Self {
        Self::Streaming {
            message: message.into(),
        }
    }

    /// Create a persistence error
    pub fn persistence(message: impl Into<String>) -> Self {
        Self::Persistence {
            message: message.into(),
        }
    }

    /// Create a histogram error
    pub fn histogram(message: impl Into<String>) -> Self {
        Self::Histogram {
            message: message.into(),
        }
    }

    /// Create an invalid path error
    pub fn invalid_path(path: impl Into<String>) -> Self {
        Self::InvalidPath { path: path.into() }
    }

    /// Create an insufficient data error
    pub fn insufficient_data(expected: usize, actual: usize) -> Self {
        Self::InsufficientData { expected, actual }
    }

    /// Create an incompatible format error
    pub fn incompatible_format(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::IncompatibleFormat {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, timeout_seconds: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_seconds,
        }
    }

    /// Create a validation error
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Memory { .. } | Self::ResourceExhausted { .. } | Self::Timeout { .. } => false,
            Self::Io(_) | Self::Streaming { .. } | Self::Dataset { .. } => true,
            Self::Configuration { .. } | Self::Validation { .. } => false,
            _ => true,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Memory { .. } | Self::ResourceExhausted { .. } => ErrorSeverity::Critical,
            Self::Configuration { .. } | Self::Validation { .. } => ErrorSeverity::Error,
            Self::Timeout { .. } | Self::Streaming { .. } => ErrorSeverity::Warning,
            _ => ErrorSeverity::Info,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = CalibrationError::configuration("test config error");
        assert!(matches!(error, CalibrationError::Configuration { .. }));
    }

    #[test]
    fn test_error_recoverability() {
        let memory_error = CalibrationError::memory("out of memory");
        assert!(!memory_error.is_recoverable());

        let io_error = CalibrationError::dataset("failed to read file");
        assert!(io_error.is_recoverable());
    }

    #[test]
    fn test_error_severity() {
        let critical_error = CalibrationError::memory("out of memory");
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);

        let warning_error = CalibrationError::timeout("processing", 60);
        assert_eq!(warning_error.severity(), ErrorSeverity::Warning);
    }
}
