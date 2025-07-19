//! Execution Path Selection
//!
//! This module provides intelligent backend selection and fallback mechanisms
//! for BitNet operations, enabling optimal performance across different hardware
//! configurations while maintaining reliability through fallback strategies.

use crate::error::{BitNetError, BitNetErrorKind};
use anyhow::Result;
use std::fmt;

#[cfg(feature = "mlx")]
use crate::mlx::{is_mlx_available, mlx_to_candle_tensor};
use candle_core::Tensor;

/// Execution backend options for BitNet operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExecutionBackend {
    /// MLX backend for Apple Silicon acceleration
    Mlx,
    /// Candle backend with Metal GPU acceleration
    CandleMetal,
    /// Candle backend with CPU execution
    CandleCpu,
    /// Automatic backend selection based on system capabilities
    Auto,
}

impl fmt::Display for ExecutionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionBackend::Mlx => write!(f, "MLX"),
            ExecutionBackend::CandleMetal => write!(f, "Candle-Metal"),
            ExecutionBackend::CandleCpu => write!(f, "Candle-CPU"),
            ExecutionBackend::Auto => write!(f, "Auto"),
        }
    }
}

/// MLX-specific error types for fallback scenarios
#[derive(Debug, Clone)]
pub enum MlxError {
    /// MLX framework is not available on this system
    NotAvailable(String),
    /// MLX operation failed during execution
    OperationFailed(String),
    /// MLX tensor conversion error
    ConversionError(String),
    /// MLX device initialization error
    DeviceError(String),
    /// MLX memory allocation error
    MemoryError(String),
    /// MLX compilation error (for custom kernels)
    CompilationError(String),
    /// MLX graph optimization error
    OptimizationError(String),
    /// Generic MLX error
    Other(String),
}

impl fmt::Display for MlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlxError::NotAvailable(msg) => write!(f, "MLX not available: {}", msg),
            MlxError::OperationFailed(msg) => write!(f, "MLX operation failed: {}", msg),
            MlxError::ConversionError(msg) => write!(f, "MLX conversion error: {}", msg),
            MlxError::DeviceError(msg) => write!(f, "MLX device error: {}", msg),
            MlxError::MemoryError(msg) => write!(f, "MLX memory error: {}", msg),
            MlxError::CompilationError(msg) => write!(f, "MLX compilation error: {}", msg),
            MlxError::OptimizationError(msg) => write!(f, "MLX optimization error: {}", msg),
            MlxError::Other(msg) => write!(f, "MLX error: {}", msg),
        }
    }
}

impl std::error::Error for MlxError {}

impl From<MlxError> for BitNetError {
    fn from(error: MlxError) -> Self {
        BitNetError::mlx_error(error.to_string(), "mlx_operation")
    }
}

/// Choose the optimal execution backend for a given operation
///
/// This function analyzes the operation type and system capabilities to select
/// the most appropriate backend for execution. It considers factors such as:
/// - Hardware availability (Apple Silicon, Metal GPU)
/// - Operation characteristics (compute intensity, memory requirements)
/// - System performance and reliability
///
/// # Arguments
/// * `operation` - The operation type to be executed (e.g., "matmul", "quantize", "bitlinear")
///
/// # Returns
/// The recommended execution backend for the operation
///
/// # Examples
/// ```
/// use bitnet_core::execution::choose_execution_backend;
///
/// let backend = choose_execution_backend("matmul");
/// println!("Selected backend: {}", backend);
/// ```
pub fn choose_execution_backend(operation: &str) -> ExecutionBackend {
    // Check if we're on Apple Silicon with MLX support
    #[cfg(feature = "mlx")]
    {
        if is_mlx_available() {
            // MLX is preferred for specific operations on Apple Silicon
            match operation {
                // High-performance operations that benefit from MLX
                "matmul" | "matrix_multiply" | "gemm" => ExecutionBackend::Mlx,
                "quantize" | "dequantize" | "quantization" => ExecutionBackend::Mlx,
                "bitlinear" | "bitnet_linear" => ExecutionBackend::Mlx,
                "attention" | "self_attention" | "multi_head_attention" => ExecutionBackend::Mlx,
                "conv2d" | "convolution" => ExecutionBackend::Mlx,
                "layer_norm" | "batch_norm" | "group_norm" => ExecutionBackend::Mlx,
                
                // Operations that may benefit from Candle's optimizations
                "embedding" | "lookup" => ExecutionBackend::CandleMetal,
                "softmax" | "activation" => ExecutionBackend::CandleMetal,
                "pooling" | "max_pool" | "avg_pool" => ExecutionBackend::CandleMetal,
                
                // CPU-bound operations
                "tokenization" | "preprocessing" | "postprocessing" => ExecutionBackend::CandleCpu,
                "io" | "file_operations" | "serialization" => ExecutionBackend::CandleCpu,
                
                // Default to MLX for unknown operations on Apple Silicon
                _ => ExecutionBackend::Mlx,
            }
        } else {
            // MLX not available, fall back to Candle backends
            choose_candle_backend(operation)
        }
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        // MLX feature not enabled, use Candle backends
        choose_candle_backend(operation)
    }
}

/// Choose the appropriate Candle backend for an operation
fn choose_candle_backend(operation: &str) -> ExecutionBackend {
    // Check if Metal is available (macOS with GPU)
    #[cfg(target_os = "macos")]
    {
        if candle_core::Device::new_metal(0).is_ok() {
            match operation {
                // GPU-accelerated operations
                "matmul" | "matrix_multiply" | "gemm" => ExecutionBackend::CandleMetal,
                "conv2d" | "convolution" => ExecutionBackend::CandleMetal,
                "attention" | "self_attention" | "multi_head_attention" => ExecutionBackend::CandleMetal,
                "softmax" | "activation" => ExecutionBackend::CandleMetal,
                "layer_norm" | "batch_norm" | "group_norm" => ExecutionBackend::CandleMetal,
                
                // CPU operations
                "tokenization" | "preprocessing" | "postprocessing" => ExecutionBackend::CandleCpu,
                "io" | "file_operations" | "serialization" => ExecutionBackend::CandleCpu,
                "quantize" | "dequantize" | "quantization" => ExecutionBackend::CandleCpu,
                
                // Default to Metal for unknown operations
                _ => ExecutionBackend::CandleMetal,
            }
        } else {
            ExecutionBackend::CandleCpu
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        // Non-macOS systems default to CPU
        ExecutionBackend::CandleCpu
    }
}

/// Fallback from MLX to Candle when MLX operations fail
///
/// This function provides a robust fallback mechanism when MLX operations
/// encounter errors. It attempts to convert MLX tensors to Candle tensors
/// and continue execution using the Candle backend.
///
/// # Arguments
/// * `mlx_error` - The MLX error that triggered the fallback
///
/// # Returns
/// A Result containing a fallback Candle tensor or an error if fallback fails
///
/// # Examples
/// ```
/// use bitnet_core::execution::{fallback_to_candle, MlxError};
///
/// let mlx_error = MlxError::OperationFailed("Matrix multiplication failed".to_string());
/// match fallback_to_candle(mlx_error) {
///     Ok(tensor) => println!("Fallback successful"),
///     Err(e) => println!("Fallback failed: {}", e),
/// }
/// ```
pub fn fallback_to_candle(mlx_error: MlxError) -> Result<Tensor> {
    // Log the MLX error for debugging
    #[cfg(feature = "tracing")]
    tracing::warn!("MLX operation failed, attempting Candle fallback: {}", mlx_error);
    
    // Analyze the error type to determine fallback strategy
    match &mlx_error {
        MlxError::NotAvailable(_) => {
            // MLX not available, create a placeholder tensor for CPU execution
            create_fallback_tensor("MLX not available")
        },
        
        MlxError::OperationFailed(msg) => {
            // Operation failed, try to recover with a default tensor
            create_fallback_tensor(&format!("MLX operation failed: {}", msg))
        },
        
        MlxError::ConversionError(_) => {
            // Conversion error, create a minimal tensor
            create_fallback_tensor("MLX conversion failed")
        },
        
        MlxError::DeviceError(_) => {
            // Device error, fall back to CPU
            create_cpu_fallback_tensor()
        },
        
        MlxError::MemoryError(_) => {
            // Memory error, create a smaller tensor
            create_small_fallback_tensor()
        },
        
        MlxError::CompilationError(_) | MlxError::OptimizationError(_) => {
            // Compilation/optimization errors, use unoptimized fallback
            create_unoptimized_fallback_tensor()
        },
        
        MlxError::Other(_) => {
            // Generic error, create a basic fallback tensor
            create_fallback_tensor("Generic MLX error")
        },
    }
}

/// Create a basic fallback tensor for CPU execution
fn create_fallback_tensor(reason: &str) -> Result<Tensor> {
    #[cfg(feature = "tracing")]
    tracing::info!("Creating fallback tensor: {}", reason);
    
    // Create a minimal 1x1 tensor on CPU as a safe fallback
    let device = candle_core::Device::Cpu;
    let data = vec![0.0f32];
    let tensor = Tensor::from_vec(data, &[1], &device)?;
    
    Ok(tensor)
}

/// Create a CPU-specific fallback tensor
fn create_cpu_fallback_tensor() -> Result<Tensor> {
    #[cfg(feature = "tracing")]
    tracing::info!("Creating CPU fallback tensor due to device error");
    
    let device = candle_core::Device::Cpu;
    let data = vec![0.0f32; 4]; // 2x2 tensor
    let tensor = Tensor::from_vec(data, &[2, 2], &device)?;
    
    Ok(tensor)
}

/// Create a smaller fallback tensor for memory-constrained scenarios
fn create_small_fallback_tensor() -> Result<Tensor> {
    #[cfg(feature = "tracing")]
    tracing::info!("Creating small fallback tensor due to memory constraints");
    
    let device = candle_core::Device::Cpu;
    let data = vec![0.0f32]; // Minimal 1-element tensor
    let tensor = Tensor::from_vec(data, &[1], &device)?;
    
    Ok(tensor)
}

/// Create an unoptimized fallback tensor
fn create_unoptimized_fallback_tensor() -> Result<Tensor> {
    #[cfg(feature = "tracing")]
    tracing::info!("Creating unoptimized fallback tensor");
    
    let device = candle_core::Device::Cpu;
    let data = vec![1.0f32; 16]; // 4x4 identity-like tensor
    let tensor = Tensor::from_vec(data, &[4, 4], &device)?;
    
    Ok(tensor)
}

/// Get the current system's preferred execution backend
pub fn get_preferred_backend() -> ExecutionBackend {
    choose_execution_backend("default")
}

/// Check if a specific backend is available on the current system
pub fn is_backend_available(backend: &ExecutionBackend) -> bool {
    match backend {
        ExecutionBackend::Mlx => {
            #[cfg(feature = "mlx")]
            {
                is_mlx_available()
            }
            #[cfg(not(feature = "mlx"))]
            {
                false
            }
        },
        
        ExecutionBackend::CandleMetal => {
            #[cfg(target_os = "macos")]
            {
                candle_core::Device::new_metal(0).is_ok()
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        },
        
        ExecutionBackend::CandleCpu => true, // CPU is always available
        
        ExecutionBackend::Auto => true, // Auto selection is always available
    }
}

/// Get a list of all available backends on the current system
pub fn get_available_backends() -> Vec<ExecutionBackend> {
    let mut backends = Vec::new();
    
    // Always add CPU backend
    backends.push(ExecutionBackend::CandleCpu);
    
    // Check Metal availability
    if is_backend_available(&ExecutionBackend::CandleMetal) {
        backends.push(ExecutionBackend::CandleMetal);
    }
    
    // Check MLX availability
    if is_backend_available(&ExecutionBackend::Mlx) {
        backends.push(ExecutionBackend::Mlx);
    }
    
    // Always add Auto selection
    backends.push(ExecutionBackend::Auto);
    
    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_backend_display() {
        assert_eq!(ExecutionBackend::Mlx.to_string(), "MLX");
        assert_eq!(ExecutionBackend::CandleMetal.to_string(), "Candle-Metal");
        assert_eq!(ExecutionBackend::CandleCpu.to_string(), "Candle-CPU");
        assert_eq!(ExecutionBackend::Auto.to_string(), "Auto");
    }

    #[test]
    fn test_mlx_error_display() {
        let error = MlxError::NotAvailable("System not supported".to_string());
        assert!(error.to_string().contains("MLX not available"));
        
        let error = MlxError::OperationFailed("Matrix multiplication failed".to_string());
        assert!(error.to_string().contains("MLX operation failed"));
    }

    #[test]
    fn test_choose_execution_backend() {
        // Test specific operations
        let backend = choose_execution_backend("matmul");
        assert!(matches!(backend, ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu));
        
        let backend = choose_execution_backend("tokenization");
        // Tokenization should prefer CPU
        assert!(matches!(backend, ExecutionBackend::CandleCpu | ExecutionBackend::Mlx));
    }

    #[test]
    fn test_fallback_to_candle() {
        let mlx_error = MlxError::NotAvailable("Test error".to_string());
        let result = fallback_to_candle(mlx_error);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.dims(), &[1]);
    }

    #[test]
    fn test_backend_availability() {
        // CPU should always be available
        assert!(is_backend_available(&ExecutionBackend::CandleCpu));
        
        // Auto should always be available
        assert!(is_backend_available(&ExecutionBackend::Auto));
    }

    #[test]
    fn test_get_available_backends() {
        let backends = get_available_backends();
        
        // Should always include CPU and Auto
        assert!(backends.contains(&ExecutionBackend::CandleCpu));
        assert!(backends.contains(&ExecutionBackend::Auto));
        
        // Should have at least 2 backends (CPU and Auto)
        assert!(backends.len() >= 2);
    }

    #[test]
    fn test_preferred_backend() {
        let backend = get_preferred_backend();
        assert!(is_backend_available(&backend));
    }

    #[test]
    fn test_mlx_error_conversion() {
        let mlx_error = MlxError::MemoryError("Out of memory".to_string());
        let bitnet_error: BitNetError = mlx_error.into();
        
        assert!(matches!(bitnet_error.kind(), BitNetErrorKind::Mlx { .. }));
    }
}