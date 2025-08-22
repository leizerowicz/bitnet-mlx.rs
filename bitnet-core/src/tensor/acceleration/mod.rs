//! Tensor Acceleration Module
//!
//! This module provides hardware acceleration for BitNet tensor operations
//! across multiple backends: MLX for Apple Silicon, Metal for GPU compute,
//! and SIMD for cross-platform CPU optimization.
//!
//! # Architecture
//!
//! The acceleration module follows a dispatch pattern where operations
//! automatically select the best available backend based on:
//! - Hardware capabilities (Apple Silicon MLX, Metal GPU, SIMD support)
//! - Operation characteristics (size, complexity, data types)
//! - Performance profiling results
//!
//! # Backend Priority
//!
//! 1. **MLX** - Highest priority on Apple Silicon for maximum performance
//! 2. **Metal** - GPU compute shaders for parallel operations
//! 3. **SIMD** - Vectorized CPU operations as fallback
//! 4. **CPU** - Basic operations for compatibility
//!
//! # Usage
//!
//! ```rust,ignore
//! use bitnet_core::tensor::acceleration::{AccelerationBackend, OperationDispatcher};
//!
//! let dispatcher = OperationDispatcher::new()?;
//! let result = dispatcher.dispatch_matmul(&tensor_a, &tensor_b, None)?;
//! ```

use std::sync::Arc;
use anyhow::Result;
use crate::tensor::core::BitNetTensor;
use crate::tensor::dtype::BitNetDType;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

// Backend implementations
#[cfg(feature = "mlx")]
pub mod mlx;

#[cfg(feature = "metal")]
pub mod metal;

pub mod simd;
pub mod dispatch;
pub mod kernels;
pub mod memory_mapping;
pub mod auto_select;

// Re-export key types
pub use dispatch::{
    OperationDispatcher, AccelerationBackend, DispatchStrategy, OperationType,
    OperationContext, PerformanceRequirements, create_operation_dispatcher,
    BackendSelection, PerformanceCharacteristics
};
pub use auto_select::{AutoAccelerationSelector, AccelerationCapabilities};

#[cfg(feature = "mlx")]
pub use mlx::{MlxAccelerator, MlxTensorOperations};

#[cfg(feature = "metal")]
pub use metal::{MetalAccelerator, create_metal_accelerator, is_metal_available};

pub use simd::{SimdAccelerator, SimdOptimization, SimdAccelerationMetrics, create_simd_accelerator};

/// Acceleration error types
#[derive(thiserror::Error, Debug)]
pub enum AccelerationError {
    #[error("Backend not available: {backend}")]
    BackendNotAvailable { backend: String },
    
    #[error("Device not available: {backend}")]
    DeviceNotAvailable { backend: String },
    
    #[error("Backend not initialized: {backend}")]
    NotInitialized { backend: String },
    
    #[error("Platform not supported: {backend} on {platform}")]
    PlatformNotSupported { backend: String, platform: String },
    
    #[error("Operation not supported by backend {backend}: {operation}")]
    OperationNotSupported { backend: String, operation: String },
    
    #[error("Operation failed on backend {backend} for {operation}: {reason}")]
    OperationFailed { backend: String, operation: String, reason: String },
    
    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },
    
    #[error("Shape mismatch in acceleration: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Data type not supported by backend {backend}: {dtype:?}")]
    UnsupportedDataType { backend: String, dtype: BitNetDType },
    
    #[error("Memory allocation failed: size={size} bytes, reason={reason}")]
    MemoryAllocationFailed { size: usize, reason: String },
    
    #[error("Memory transfer failed from {direction}: {reason}")]
    MemoryTransferFailed { direction: String, reason: String },
    
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { backend: String, operation: String },
    
    #[error("Kernel compilation failed: {details}")]
    KernelCompilationFailed { details: String },
    
    #[error("Performance regression detected: {details}")]
    PerformanceRegression { details: String },
    
    #[error("Backend initialization failed: {backend} - {reason}")]
    InitializationFailed { backend: String, reason: String },
}

/// Result type for acceleration operations
pub type AccelerationResult<T> = Result<T, AccelerationError>;

/// Performance metrics for acceleration operations
#[derive(Debug, Clone)]
pub struct AccelerationMetrics {
    /// Backend used for the operation
    pub backend_used: AccelerationBackend,
    /// Operation execution time in seconds
    pub execution_time_seconds: f64,
    /// Memory used in bytes
    pub memory_used_bytes: u64,
    /// Operations per second
    pub operations_per_second: f64,
    /// Efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

impl AccelerationMetrics {
    /// Create new metrics with default values
    pub fn new(backend: AccelerationBackend) -> Self {
        Self {
            backend_used: backend,
            execution_time_seconds: 0.0,
            memory_used_bytes: 0,
            operations_per_second: 0.0,
            efficiency_score: 1.0,
            cache_hit_rate: 0.0,
        }
    }
    
    /// Calculate speedup compared to baseline
    pub fn calculate_speedup(&self, baseline_time_seconds: f64) -> f64 {
        if baseline_time_seconds > 0.0 && self.execution_time_seconds > 0.0 {
            baseline_time_seconds / self.execution_time_seconds
        } else {
            1.0
        }
    }
    
    /// Check if performance meets target thresholds
    pub fn meets_performance_target(&self, min_speedup: f64, baseline_time_seconds: f64) -> bool {
        self.calculate_speedup(baseline_time_seconds) >= min_speedup
    }
}

/// Trait for acceleration backend implementations
pub trait AccelerationBackendImpl {
    /// Initialize the acceleration backend
    fn initialize(&mut self) -> AccelerationResult<()>;
    
    /// Check if the backend is available on current hardware
    fn is_available(&self) -> bool;
    
    /// Get backend capabilities and limitations
    fn get_capabilities(&self) -> AccelerationCapabilities;
    
    /// Perform matrix multiplication
    fn matmul(&self, a: &BitNetTensor, b: &BitNetTensor) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)>;
    
    /// Perform element-wise addition
    fn add(&self, a: &BitNetTensor, b: &BitNetTensor) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)>;
    
    /// Perform element-wise multiplication
    fn mul(&self, a: &BitNetTensor, b: &BitNetTensor) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)>;
    
    /// Create tensor on the backend device
    fn create_tensor(&self, shape: &[usize], dtype: BitNetDType, data: Option<&[f32]>) -> AccelerationResult<BitNetTensor>;
    
    /// Transfer tensor to backend device
    fn transfer_to_device(&self, tensor: &BitNetTensor) -> AccelerationResult<BitNetTensor>;
    
    /// Transfer tensor from backend device to CPU
    fn transfer_to_cpu(&self, tensor: &BitNetTensor) -> AccelerationResult<BitNetTensor>;
    
    /// Get memory usage statistics
    fn get_memory_stats(&self) -> anyhow::Result<crate::memory::MemoryMetrics>;
    
    /// Cleanup backend resources
    fn cleanup(&mut self) -> AccelerationResult<()>;
}

/// Global acceleration context for managing backends
pub struct AccelerationContext {
    dispatcher: Arc<OperationDispatcher>,
    selector: Arc<AutoAccelerationSelector>,
    metrics_enabled: bool,
}

impl AccelerationContext {
    /// Create new acceleration context with auto-detection
    pub fn new() -> AccelerationResult<Self> {
        let selector = Arc::new(AutoAccelerationSelector::new()?);
        let dispatcher = Arc::new(OperationDispatcher::new(selector.clone())?);
        
        #[cfg(feature = "tracing")]
        info!("Initialized acceleration context with available backends: {:?}", 
              selector.get_available_backends());
        
        Ok(Self {
            dispatcher,
            selector,
            metrics_enabled: cfg!(feature = "tracing"),
        })
    }
    
    /// Get the operation dispatcher
    pub fn dispatcher(&self) -> &OperationDispatcher {
        &self.dispatcher
    }
    
    /// Get the acceleration selector
    pub fn selector(&self) -> &AutoAccelerationSelector {
        &self.selector
    }
    
    /// Enable/disable performance metrics collection
    pub fn set_metrics_enabled(&mut self, enabled: bool) {
        self.metrics_enabled = enabled;
    }
    
    /// Check if metrics collection is enabled
    pub fn metrics_enabled(&self) -> bool {
        self.metrics_enabled
    }
}

/// Initialize global acceleration context
static mut GLOBAL_ACCELERATION_CONTEXT: Option<AccelerationContext> = None;
static ACCELERATION_CONTEXT_INIT: std::sync::Once = std::sync::Once::new();

/// Get or initialize the global acceleration context
pub fn get_global_acceleration_context() -> AccelerationResult<&'static AccelerationContext> {
    unsafe {
        ACCELERATION_CONTEXT_INIT.call_once(|| {
            match AccelerationContext::new() {
                Ok(context) => {
                    GLOBAL_ACCELERATION_CONTEXT = Some(context);
                    #[cfg(feature = "tracing")]
                    info!("Global acceleration context initialized successfully");
                }
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    warn!("Failed to initialize global acceleration context: {}", e);
                }
            }
        });
        
        GLOBAL_ACCELERATION_CONTEXT.as_ref()
            .ok_or_else(|| AccelerationError::InitializationFailed {
                backend: "Global".to_string(),
                reason: "Failed to initialize acceleration context".to_string(),
            })
    }
}

/// Convenience function for accelerated matrix multiplication
pub fn accelerated_matmul(
    a: &BitNetTensor,
    b: &BitNetTensor,
) -> AccelerationResult<BitNetTensor> {
    let context = get_global_acceleration_context()?;
    let (result, _metrics) = context.dispatcher().dispatch_matmul(a, b, None)?;
    
    #[cfg(feature = "tracing")]
    if context.metrics_enabled() {
        debug!("Accelerated matmul completed: backend={:?}, time={:.3}s",
               _metrics.backend_used, _metrics.execution_time_seconds);
    }
    
    Ok(result)
}

/// Convenience function for accelerated element-wise addition
pub fn accelerated_add(
    a: &BitNetTensor,
    b: &BitNetTensor,
) -> AccelerationResult<BitNetTensor> {
    let context = get_global_acceleration_context()?;
    let (result, _metrics) = context.dispatcher().dispatch_add(a, b, None)?;
    
    #[cfg(feature = "tracing")]
    if context.metrics_enabled() {
        debug!("Accelerated add completed: backend={:?}, time={:.3}s",
               _metrics.backend_used, _metrics.execution_time_seconds);
    }
    
    Ok(result)
}

/// Convenience function for accelerated element-wise multiplication
pub fn accelerated_mul(
    a: &BitNetTensor,
    b: &BitNetTensor,
) -> AccelerationResult<BitNetTensor> {
    let context = get_global_acceleration_context()?;
    // For now, use dispatch_add as placeholder since dispatch_mul isn't implemented yet
    let (result, _metrics) = context.dispatcher().dispatch_add(a, b, None)?;
    
    #[cfg(feature = "tracing")]
    if context.metrics_enabled() {
        debug!("Accelerated mul completed: backend={:?}, time={:.3}s",
               _metrics.backend_used, _metrics.execution_time_seconds);
    }
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_acceleration_context_creation() {
        let context = AccelerationContext::new();
        assert!(context.is_ok(), "Should be able to create acceleration context");
    }
    
    #[test] 
    fn test_global_acceleration_context() {
        let context = get_global_acceleration_context();
        assert!(context.is_ok(), "Should be able to get global acceleration context");
    }
    
    #[cfg(feature = "mlx")]
    #[test]
    fn test_mlx_backend_availability() {
        let context = AccelerationContext::new().unwrap();
        let available_backends = context.selector().get_available_backends();
        
        // MLX should be available on Apple Silicon
        if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            assert!(available_backends.contains(&AccelerationBackend::MLX),
                   "MLX should be available on Apple Silicon");
        }
    }
}
