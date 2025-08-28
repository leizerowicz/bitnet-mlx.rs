//! Enhanced Error Handling and Formatting
//!
//! This module provides enhanced error handling with detailed context,
//! better error formatting, and improved debugging information.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub mod context;
pub mod formatting;

pub use context::{ContextualError, ErrorContext, ErrorContextBuilder};
pub use formatting::{ErrorFormatter, ErrorReport, ErrorSeverity};

/// Enhanced error with detailed context and formatting
#[derive(Error, Debug, Clone)]
#[allow(dead_code)]
pub struct BitNetError {
    /// The underlying error kind
    pub kind: BitNetErrorKind,
    /// Additional context information
    pub context: ErrorContext,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Timestamp when error occurred
    pub timestamp: u64,
    /// Unique error ID for tracking
    pub error_id: String,
}

/// Categories of errors in BitNet
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BitNetErrorKind {
    /// Device-related errors
    #[error("Device error: {message}")]
    Device {
        message: String,
        device_type: String,
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory {
        message: String,
        size: Option<usize>,
        operation: String,
    },

    /// Tensor operation errors
    #[error("Tensor error: {message}")]
    Tensor {
        message: String,
        shape: Option<Vec<usize>>,
        dtype: Option<String>,
    },

    /// Data conversion errors
    #[error("Conversion error: {message}")]
    Conversion {
        message: String,
        from_type: String,
        to_type: String,
    },

    /// Metal/GPU specific errors
    #[error("Metal error: {message}")]
    Metal {
        message: String,
        operation: String,
        device_name: Option<String>,
    },

    /// MLX framework errors
    #[error("MLX error: {message}")]
    Mlx { message: String, operation: String },

    /// Configuration and validation errors
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        parameter: Option<String>,
    },

    /// I/O and file system errors
    #[error("I/O error: {message}")]
    Io {
        message: String,
        path: Option<String>,
    },

    /// Concurrency and threading errors
    #[error("Concurrency error: {message}")]
    Concurrency { message: String, resource: String },

    /// Internal system errors
    #[error("Internal error: {message}")]
    Internal { message: String, component: String },
}

impl BitNetError {
    /// Creates a new BitNet error with minimal context
    pub fn new(kind: BitNetErrorKind) -> Self {
        Self {
            kind,
            context: ErrorContext::new(),
            severity: ErrorSeverity::Error,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            error_id: generate_error_id(),
        }
    }

    /// Creates a new BitNet error with context
    pub fn with_context(kind: BitNetErrorKind, context: ErrorContext) -> Self {
        Self {
            kind,
            context,
            severity: ErrorSeverity::Error,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            error_id: generate_error_id(),
        }
    }

    /// Creates a new BitNet error with severity
    pub fn with_severity(kind: BitNetErrorKind, severity: ErrorSeverity) -> Self {
        Self {
            kind,
            context: ErrorContext::new(),
            severity,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            error_id: generate_error_id(),
        }
    }

    /// Adds context to the error
    pub fn add_context(mut self, key: &str, value: &str) -> Self {
        self.context.add(key, value);
        self
    }

    /// Sets the error severity
    pub fn set_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Gets the error kind
    pub fn kind(&self) -> &BitNetErrorKind {
        &self.kind
    }

    /// Gets the error context
    pub fn context(&self) -> &ErrorContext {
        &self.context
    }

    /// Gets the error severity
    pub fn severity(&self) -> ErrorSeverity {
        self.severity
    }

    /// Gets the error timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Gets the error ID
    pub fn error_id(&self) -> &str {
        &self.error_id
    }

    /// Formats the error as a detailed report
    pub fn to_report(&self) -> ErrorReport {
        ErrorReport::from_error(self)
    }

    /// Checks if this is a device-related error
    pub fn is_device_error(&self) -> bool {
        matches!(self.kind, BitNetErrorKind::Device { .. })
    }

    /// Checks if this is a memory-related error
    pub fn is_memory_error(&self) -> bool {
        matches!(self.kind, BitNetErrorKind::Memory { .. })
    }

    /// Checks if this is a Metal-related error
    pub fn is_metal_error(&self) -> bool {
        matches!(self.kind, BitNetErrorKind::Metal { .. })
    }

    /// Checks if this is an MLX-related error
    pub fn is_mlx_error(&self) -> bool {
        matches!(self.kind, BitNetErrorKind::Mlx { .. })
    }

    /// Checks if this is a critical error
    pub fn is_critical(&self) -> bool {
        matches!(
            self.severity,
            ErrorSeverity::Critical | ErrorSeverity::Fatal
        )
    }

    /// Extracts device information if this is a device error
    pub fn device_info(&self) -> Option<&str> {
        match &self.kind {
            BitNetErrorKind::Device { device_type, .. } => Some(device_type),
            BitNetErrorKind::Metal { device_name, .. } => device_name.as_deref(),
            _ => None,
        }
    }

    /// Extracts memory information if this is a memory error
    pub fn memory_info(&self) -> Option<(Option<usize>, &str)> {
        match &self.kind {
            BitNetErrorKind::Memory {
                size, operation, ..
            } => Some((*size, operation)),
            _ => None,
        }
    }
}

impl fmt::Display for BitNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} (ID: {})",
            self.severity.as_str(),
            self.kind,
            self.error_id
        )?;

        if !self.context.is_empty() {
            write!(f, "\nContext: {}", self.context)?;
        }

        Ok(())
    }
}

/// Convenience functions for creating specific error types
impl BitNetError {
    /// Creates a device error
    pub fn device_error(message: impl Into<String>, device_type: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Device {
            message: message.into(),
            device_type: device_type.into(),
        })
    }

    /// Creates a memory error
    pub fn memory_error(message: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Memory {
            message: message.into(),
            size: None,
            operation: operation.into(),
        })
    }

    /// Creates a memory error with size information
    pub fn memory_error_with_size(
        message: impl Into<String>,
        size: usize,
        operation: impl Into<String>,
    ) -> Self {
        Self::new(BitNetErrorKind::Memory {
            message: message.into(),
            size: Some(size),
            operation: operation.into(),
        })
    }

    /// Creates a tensor error
    pub fn tensor_error(message: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Tensor {
            message: message.into(),
            shape: None,
            dtype: None,
        })
    }

    /// Creates a tensor error with shape and dtype information
    pub fn tensor_error_with_info(
        message: impl Into<String>,
        shape: Option<Vec<usize>>,
        dtype: Option<String>,
    ) -> Self {
        Self::new(BitNetErrorKind::Tensor {
            message: message.into(),
            shape,
            dtype,
        })
    }

    /// Creates a conversion error
    pub fn conversion_error(
        message: impl Into<String>,
        from_type: impl Into<String>,
        to_type: impl Into<String>,
    ) -> Self {
        Self::new(BitNetErrorKind::Conversion {
            message: message.into(),
            from_type: from_type.into(),
            to_type: to_type.into(),
        })
    }

    /// Creates a Metal error
    pub fn metal_error(message: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Metal {
            message: message.into(),
            operation: operation.into(),
            device_name: None,
        })
    }

    /// Creates a Metal error with device information
    pub fn metal_error_with_device(
        message: impl Into<String>,
        operation: impl Into<String>,
        device_name: impl Into<String>,
    ) -> Self {
        let device_name_str = device_name.into();
        let mut error = Self::new(BitNetErrorKind::Metal {
            message: message.into(),
            operation: operation.into(),
            device_name: Some(device_name_str.clone()),
        });
        error.context.add("device_name", &device_name_str);
        error
    }

    /// Creates an MLX error
    pub fn mlx_error(message: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Mlx {
            message: message.into(),
            operation: operation.into(),
        })
    }

    /// Creates a configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Configuration {
            message: message.into(),
            parameter: None,
        })
    }

    /// Creates a configuration error with parameter information
    pub fn config_error_with_param(
        message: impl Into<String>,
        parameter: impl Into<String>,
    ) -> Self {
        Self::new(BitNetErrorKind::Configuration {
            message: message.into(),
            parameter: Some(parameter.into()),
        })
    }

    /// Creates an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Io {
            message: message.into(),
            path: None,
        })
    }

    /// Creates an I/O error with path information
    pub fn io_error_with_path(message: impl Into<String>, path: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Io {
            message: message.into(),
            path: Some(path.into()),
        })
    }

    /// Creates a concurrency error
    pub fn concurrency_error(message: impl Into<String>, resource: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Concurrency {
            message: message.into(),
            resource: resource.into(),
        })
    }

    /// Creates an internal error
    pub fn internal_error(message: impl Into<String>, component: impl Into<String>) -> Self {
        Self::new(BitNetErrorKind::Internal {
            message: message.into(),
            component: component.into(),
        })
    }
}

/// Generates a unique error ID
fn generate_error_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);

    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    format!("BN{:08X}{:04X}", timestamp & 0xFFFFFFFF, id & 0xFFFF)
}

/// Result type using BitNetError
pub type BitNetResult<T> = std::result::Result<T, BitNetError>;

/// Trait for converting errors to BitNetError with context
pub trait ToBitNetError<T> {
    /// Converts to BitNetError with additional context
    fn to_bitnet_error(self, context: ErrorContext) -> BitNetResult<T>;

    /// Converts to BitNetError with a simple context message
    fn with_context(self, message: &str) -> BitNetResult<T>;
}

impl<T, E> ToBitNetError<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn to_bitnet_error(self, context: ErrorContext) -> BitNetResult<T> {
        self.map_err(|e| {
            let kind = BitNetErrorKind::Internal {
                message: e.to_string(),
                component: "unknown".to_string(),
            };
            BitNetError::with_context(kind, context)
        })
    }

    fn with_context(self, message: &str) -> BitNetResult<T> {
        self.map_err(|e| {
            let mut context = ErrorContext::new();
            context.add("context", message);
            context.add("original_error", &e.to_string());

            let kind = BitNetErrorKind::Internal {
                message: e.to_string(),
                component: "unknown".to_string(),
            };
            BitNetError::with_context(kind, context)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = BitNetError::device_error("Test device error", "CPU");

        assert!(error.is_device_error());
        assert!(!error.is_memory_error());
        assert_eq!(error.device_info(), Some("CPU"));
        assert!(!error.error_id().is_empty());
    }

    #[test]
    fn test_error_with_context() {
        let mut context = ErrorContext::new();
        context.add("operation", "tensor_creation");
        context.add("size", "1024");

        let error = BitNetError::memory_error_with_size("Out of memory", 1024, "allocation")
            .add_context("additional_info", "test context");

        assert!(error.is_memory_error());
        assert_eq!(error.memory_info(), Some((Some(1024), "allocation")));
    }

    #[test]
    fn test_error_severity() {
        let error = BitNetError::device_error("Critical device failure", "Metal")
            .set_severity(ErrorSeverity::Critical);

        assert!(error.is_critical());
        assert_eq!(error.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_display() {
        let error = BitNetError::tensor_error("Shape mismatch")
            .add_context("expected_shape", "[2, 3, 4]")
            .add_context("actual_shape", "[2, 4, 3]");

        let display_str = format!("{}", error);
        assert!(display_str.contains("Tensor error"));
        assert!(display_str.contains("Context:"));
    }

    #[test]
    fn test_error_id_uniqueness() {
        let error1 = BitNetError::device_error("Error 1", "CPU");
        let error2 = BitNetError::device_error("Error 2", "Metal");

        assert_ne!(error1.error_id(), error2.error_id());
    }

    #[test]
    fn test_conversion_error() {
        let error = BitNetError::conversion_error("Invalid conversion", "f32", "i8");

        if let BitNetErrorKind::Conversion {
            from_type, to_type, ..
        } = &error.kind
        {
            assert_eq!(from_type, "f32");
            assert_eq!(to_type, "i8");
        } else {
            panic!("Expected conversion error");
        }
    }

    #[test]
    fn test_to_bitnet_error_trait() {
        let result: Result<i32, std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found",
        ));

        let bitnet_result = result.with_context("Failed to read configuration file");
        assert!(bitnet_result.is_err());

        let error = bitnet_result.unwrap_err();
        assert!(!error.context().is_empty());
    }
}
