//! BitNet CUDA Backend
//! 
//! High-performance CUDA implementation of BitNet operations with Microsoft W2A8 GEMV
//! kernel parity and optimized GPU acceleration.

pub mod backend;
pub mod error;
pub mod kernels;
pub mod memory;
pub mod stream;

// Re-export main types
pub use backend::{CudaBackend, CudaBackendConfig};
pub use error::{CudaError, CudaResult};
pub use kernels::w2a8_gemv::{W2A8GemvKernel, W2A8GemvConfig};
pub use memory::{CudaMemoryManager, CudaAllocation};
pub use stream::{CudaStream, CudaStreamConfig};

#[cfg(feature = "cuda")]
pub use cudarc;

/// Current version of the BitNet CUDA backend
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Microsoft W2A8 GEMV kernel performance targets
pub mod performance_targets {
    /// Target speedup range over BF16 baseline on A100
    pub const W2A8_SPEEDUP_MIN: f32 = 1.27;
    pub const W2A8_SPEEDUP_MAX: f32 = 3.63;
    
    /// Memory bandwidth utilization target (%)
    pub const MEMORY_BANDWIDTH_TARGET: f32 = 85.0;
    
    /// Compute utilization target (%)
    pub const COMPUTE_UTILIZATION_TARGET: f32 = 90.0;
}

/// Initialize CUDA backend
#[cfg(feature = "cuda")]
pub fn initialize() -> CudaResult<CudaBackend> {
    CudaBackend::new(CudaBackendConfig::default())
}

/// Check if CUDA is available and supported
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    cudarc::driver::safe::CudaDevice::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}

/// Get CUDA device count
#[cfg(feature = "cuda")]
pub fn device_count() -> CudaResult<usize> {
    Ok(cudarc::driver::safe::device_count()?)
}

#[cfg(not(feature = "cuda"))]
pub fn device_count() -> CudaResult<usize> {
    Err(CudaError::CudaNotEnabled)
}
