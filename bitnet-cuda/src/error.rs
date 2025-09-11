//! CUDA backend error types and handling

use thiserror::Error;

/// CUDA backend error types
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA feature not enabled
    #[error("CUDA feature not enabled - compile with --features cuda")]
    CudaNotEnabled,

    /// CUDA driver error
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaDriver(#[from] cudarc::driver::DriverError),

    /// CUDA device error
    #[error("CUDA device error: {0}")]
    DeviceError(String),

    /// Memory allocation error
    #[error("CUDA memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Kernel execution error
    #[error("CUDA kernel execution failed: {0}")]
    KernelExecution(String),

    /// Invalid configuration
    #[error("Invalid CUDA configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported operation
    #[error("Unsupported CUDA operation: {0}")]
    UnsupportedOperation(String),

    /// Stream synchronization error
    #[error("CUDA stream synchronization error: {0}")]
    StreamSync(String),

    /// Data type not supported
    #[error("Data type {dtype} not supported on CUDA device")]
    UnsupportedDataType { dtype: String },

    /// Tensor shape incompatibility
    #[error("Incompatible tensor shapes for CUDA operation: {reason}")]
    IncompatibleShapes { reason: String },

    /// Performance degradation detected
    #[error("CUDA performance below threshold: {actual} < {expected} (ratio: {ratio:.2})")]
    PerformanceDegradation {
        actual: f32,
        expected: f32,
        ratio: f32,
    },
}

/// Result type for CUDA operations
pub type CudaResult<T> = std::result::Result<T, CudaError>;

impl CudaError {
    /// Create a device error
    pub fn device_error(msg: impl Into<String>) -> Self {
        Self::DeviceError(msg.into())
    }

    /// Create a memory allocation error
    pub fn memory_allocation(msg: impl Into<String>) -> Self {
        Self::MemoryAllocation(msg.into())
    }

    /// Create a kernel execution error
    pub fn kernel_execution(msg: impl Into<String>) -> Self {
        Self::KernelExecution(msg.into())
    }

    /// Create an invalid configuration error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation(msg: impl Into<String>) -> Self {
        Self::UnsupportedOperation(msg.into())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            CudaError::CudaNotEnabled => false,
            CudaError::DeviceError(_) => false,
            CudaError::MemoryAllocation(_) => true,
            CudaError::KernelExecution(_) => true,
            CudaError::InvalidConfig(_) => false,
            CudaError::UnsupportedOperation(_) => false,
            CudaError::StreamSync(_) => true,
            CudaError::UnsupportedDataType { .. } => false,
            CudaError::IncompatibleShapes { .. } => false,
            CudaError::PerformanceDegradation { .. } => true,
            #[cfg(feature = "cuda")]
            CudaError::CudaDriver(_) => true,
        }
    }
}
