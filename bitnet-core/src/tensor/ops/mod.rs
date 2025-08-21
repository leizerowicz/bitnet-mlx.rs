//! BitNet Tensor Operations Module
//!
//! This module provides comprehensive tensor operations for BitNet tensors,
//! including arithmetic operations, broadcasting, linear algebra, and
//! BitNet-specific operations.
//!
//! # Architecture
//!
//! The operations module is organized into several sub-modules:
//! - **arithmetic**: Basic arithmetic operations (+, -, *, /)
//! - **broadcasting**: Broadcasting utilities and operations
//! - **linalg**: Linear algebra operations (matmul, dot product, etc.)
//! - **reduction**: Reduction operations (sum, mean, min, max, etc.)
//! - **activation**: Neural network activation functions
//! - **bitnet**: BitNet-specific quantized operations
//!
//! All operations are designed to work efficiently with the BitNet memory
//! management system and support heterogeneous device execution.

use thiserror::Error;
use crate::tensor::dtype::BitNetDType;

/// Result type for tensor operations
pub type TensorOpResult<T> = Result<T, TensorOpError>;

/// Errors that can occur during tensor operations
#[derive(Error, Debug)]
pub enum TensorOpError {
    /// Shape mismatch between tensors
    #[error("Shape mismatch: expected {expected:?}, found {actual:?} in operation {operation}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },

    /// Broadcasting error
    #[error("Broadcasting error in {operation}: {reason}. LHS shape: {lhs_shape:?}, RHS shape: {rhs_shape:?}")]
    BroadcastError {
        reason: String,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        operation: String,
    },

    /// Device mismatch error
    #[error("Device mismatch in operation {operation}: tensors must be on the same device")]
    DeviceMismatch { operation: String },

    /// Data type mismatch error
    #[error("Data type mismatch in operation {operation}: {reason}")]
    DTypeMismatch { operation: String, reason: String },
    
    /// Data type mismatch error (alternative form)
    #[error("Data type mismatch in operation {operation}: {reason}")]
    DataTypeMismatch { operation: String, reason: String },

    /// Computation error during operation
    #[error("Computation error in operation {operation}: {reason}")]
    ComputationError { operation: String, reason: String },

    /// Unsupported operation for the given data type
    #[error("Operation {operation} is not supported for data type {dtype:?}")]
    UnsupportedOperation { operation: String, dtype: BitNetDType },

    /// Memory-related error during operation
    #[error("Memory error in operation {operation}: {reason}")]
    MemoryError { operation: String, reason: String },

    /// Candle backend error
    #[error("Candle error in operation {operation}: {error}")]
    CandleError { operation: String, error: String },

    /// Invalid tensor state
    #[error("Invalid tensor state in operation {operation}: {reason}")]
    InvalidTensor { operation: String, reason: String },

    /// Numerical computation error (e.g., division by zero, overflow)
    #[error("Numerical error in operation {operation}: {reason}")]
    NumericalError { operation: String, reason: String },

    /// Internal operation error
    #[error("Internal operation error: {reason}")]
    InternalError { reason: String },
}

/// Conversion from MemoryError to TensorOpError
impl From<crate::memory::MemoryError> for TensorOpError {
    fn from(err: crate::memory::MemoryError) -> Self {
        match err {
            crate::memory::MemoryError::InsufficientMemory { .. } => {
                TensorOpError::MemoryError {
                    operation: "tensor_operation".to_string(),
                    reason: err.to_string(),
                }
            }
            _ => TensorOpError::InternalError {
                reason: err.to_string(),
            },
        }
    }
}

/// Conversion from TensorOpError to MemoryError 
impl From<TensorOpError> for crate::memory::MemoryError {
    fn from(err: TensorOpError) -> Self {
        match err {
            TensorOpError::MemoryError { reason, .. } => {
                crate::memory::MemoryError::InternalError { reason }
            }
            _ => crate::memory::MemoryError::InternalError {
                reason: err.to_string(),
            }
        }
    }
}

/// Conversion from candle_core::Error to TensorOpError
impl From<candle_core::Error> for TensorOpError {
    fn from(err: candle_core::Error) -> Self {
        TensorOpError::CandleError {
            operation: "candle_operation".to_string(),
            error: err.to_string(),
        }
    }
}

// Export modules
pub mod arithmetic;
pub mod broadcasting;
pub mod linear_algebra;
pub mod reduction;
pub mod activation;
pub mod simd;

// Re-exports for convenience
pub use arithmetic::*;
pub use broadcasting::*;
pub use linear_algebra::*;
pub use reduction::*;
pub use activation::*;
pub use simd::*;
