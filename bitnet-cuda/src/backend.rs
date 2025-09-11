//! Main CUDA backend implementation

use crate::error::{CudaError, CudaResult};
use crate::memory::CudaMemoryManager;
use crate::stream::{CudaStream, CudaStreamConfig};
use crate::kernels::w2a8_gemv::{W2A8GemvKernel, W2A8GemvConfig};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaBackendConfig {
    /// CUDA device ID to use
    pub device_id: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Number of compute streams
    pub num_streams: usize,
    /// Enable performance monitoring
    pub enable_profiling: bool,
    /// W2A8 GEMV kernel configuration
    pub w2a8_config: W2A8GemvConfig,
}

impl Default for CudaBackendConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB default
            num_streams: 4,
            enable_profiling: false,
            w2a8_config: W2A8GemvConfig::default(),
        }
    }
}

/// Main CUDA backend for BitNet operations
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    memory_manager: CudaMemoryManager,
    streams: Vec<CudaStream>,
    w2a8_kernel: W2A8GemvKernel,
    config: CudaBackendConfig,
}

impl CudaBackend {
    /// Create a new CUDA backend
    #[cfg(feature = "cuda")]
    pub fn new(config: CudaBackendConfig) -> CudaResult<Self> {
        // Initialize CUDA device
        let device = Arc::new(CudaDevice::new(config.device_id)?);
        
        // Validate device capabilities
        Self::validate_device_capabilities(&device)?;
        
        // Initialize memory manager
        let memory_manager = CudaMemoryManager::new(
            device.clone(),
            config.memory_pool_size,
        )?;
        
        // Create compute streams
        let mut streams = Vec::with_capacity(config.num_streams);
        for i in 0..config.num_streams {
            let stream_config = CudaStreamConfig {
                name: format!("compute_stream_{}", i),
                priority: 0,
                flags: 0,
            };
            streams.push(CudaStream::new(device.clone(), stream_config)?);
        }
        
        // Initialize W2A8 GEMV kernel
        let w2a8_kernel = W2A8GemvKernel::new(device.clone(), config.w2a8_config.clone())?;
        
        #[cfg(feature = "profiling")]
        tracing::info!(
            "CUDA backend initialized on device {} with {} streams",
            config.device_id,
            config.num_streams
        );
        
        Ok(Self {
            device,
            memory_manager,
            streams,
            w2a8_kernel,
            config,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_config: CudaBackendConfig) -> CudaResult<Self> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Get device reference
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Get memory manager
    pub fn memory_manager(&self) -> &CudaMemoryManager {
        &self.memory_manager
    }

    /// Get compute stream by index
    pub fn stream(&self, index: usize) -> CudaResult<&CudaStream> {
        self.streams.get(index)
            .ok_or_else(|| CudaError::InvalidConfig(
                format!("Stream index {} out of bounds (max: {})", index, self.streams.len())
            ))
    }

    /// Get W2A8 GEMV kernel
    pub fn w2a8_kernel(&self) -> &W2A8GemvKernel {
        &self.w2a8_kernel
    }

    /// Execute W2A8 GEMV operation
    /// 
    /// Performs matrix-vector multiplication with 2-bit weights and 8-bit activations.
    /// This is the core operation for BitNet inference with Microsoft-level performance.
    #[cfg(feature = "cuda")]
    pub fn w2a8_gemv(
        &self,
        weights: &CudaSlice<u8>,      // Packed 2-bit weights
        activations: &CudaSlice<i8>,  // 8-bit activations
        output: &mut CudaSlice<i32>,  // 32-bit output
        m: usize,                     // Number of output features
        k: usize,                     // Number of input features
        stream_idx: usize,            // Stream to use for execution
    ) -> CudaResult<()> {
        let stream = self.stream(stream_idx)?;
        
        // Validate input dimensions
        self.validate_w2a8_dimensions(weights, activations, output, m, k)?;
        
        // Execute kernel
        self.w2a8_kernel.execute(
            weights,
            activations,
            output,
            m,
            k,
            stream,
        )?;
        
        #[cfg(feature = "profiling")]
        tracing::debug!("W2A8 GEMV executed: {}x{} on stream {}", m, k, stream_idx);
        
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn w2a8_gemv(
        &self,
        _weights: &[u8],
        _activations: &[i8],
        _output: &mut [i32],
        _m: usize,
        _k: usize,
        _stream_idx: usize,
    ) -> CudaResult<()> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Synchronize all streams
    pub fn synchronize(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            total_w2a8_operations: self.w2a8_kernel.operation_count(),
            average_execution_time_us: self.w2a8_kernel.average_execution_time_us(),
            memory_utilization_percent: self.memory_manager.utilization_percent(),
            peak_bandwidth_gbps: self.w2a8_kernel.peak_bandwidth_gbps(),
        }
    }

    /// Validate device capabilities for BitNet operations
    #[cfg(feature = "cuda")]
    fn validate_device_capabilities(device: &CudaDevice) -> CudaResult<()> {
        let major = device.attribute(cudarc::driver::sys::CudaDeviceAttribute::ComputeCapabilityMajor)?;
        let minor = device.attribute(cudarc::driver::sys::CudaDeviceAttribute::ComputeCapabilityMinor)?;
        
        // Require compute capability 7.5+ for dp4a instruction support
        if major < 7 || (major == 7 && minor < 5) {
            return Err(CudaError::UnsupportedOperation(
                format!("Compute capability {}.{} insufficient for BitNet operations (requires 7.5+)", major, minor)
            ));
        }
        
        // Check memory availability
        let (free, _total) = device.memory_info()?;
        let min_memory = 512 * 1024 * 1024; // 512MB minimum
        if free < min_memory {
            return Err(CudaError::MemoryAllocation(
                format!("Insufficient GPU memory: {} bytes available, {} bytes required", free, min_memory)
            ));
        }
        
        Ok(())
    }

    /// Validate W2A8 GEMV input dimensions
    #[cfg(feature = "cuda")]
    fn validate_w2a8_dimensions(
        &self,
        weights: &CudaSlice<u8>,
        activations: &CudaSlice<i8>,
        output: &CudaSlice<i32>,
        m: usize,
        k: usize,
    ) -> CudaResult<()> {
        // Weights: 2 bits per weight, packed 4 weights per byte
        let expected_weight_size = (m * k + 3) / 4; // Round up for partial bytes
        if weights.len() != expected_weight_size {
            return Err(CudaError::IncompatibleShapes {
                reason: format!(
                    "Weight tensor size mismatch: expected {} bytes for {}x{} matrix, got {}",
                    expected_weight_size, m, k, weights.len()
                ),
            });
        }
        
        // Activations: 8 bits per activation
        if activations.len() != k {
            return Err(CudaError::IncompatibleShapes {
                reason: format!(
                    "Activation tensor size mismatch: expected {} elements, got {}",
                    k, activations.len()
                ),
            });
        }
        
        // Output: 32 bits per output
        if output.len() != m {
            return Err(CudaError::IncompatibleShapes {
                reason: format!(
                    "Output tensor size mismatch: expected {} elements, got {}",
                    m, output.len()
                ),
            });
        }
        
        Ok(())
    }
}

/// Performance statistics for CUDA backend
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of W2A8 operations executed
    pub total_w2a8_operations: u64,
    /// Average execution time per operation (microseconds)
    pub average_execution_time_us: f32,
    /// Memory utilization percentage
    pub memory_utilization_percent: f32,
    /// Peak bandwidth achieved (GB/s)
    pub peak_bandwidth_gbps: f32,
}

impl PerformanceStats {
    /// Check if performance meets Microsoft W2A8 targets
    pub fn meets_performance_targets(&self) -> bool {
        use crate::performance_targets::*;
        
        // Check if we're achieving reasonable bandwidth utilization
        self.memory_utilization_percent >= MEMORY_BANDWIDTH_TARGET
    }
    
    /// Get performance ratio compared to baseline
    pub fn performance_ratio(&self, baseline_time_us: f32) -> f32 {
        if self.average_execution_time_us > 0.0 {
            baseline_time_us / self.average_execution_time_us
        } else {
            0.0
        }
    }
}
