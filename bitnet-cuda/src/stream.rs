//! CUDA stream management for asynchronous operations

use crate::error::{CudaError, CudaResult};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaStream as CudarcStream};
use std::sync::Arc;

/// CUDA stream configuration
#[derive(Debug, Clone)]
pub struct CudaStreamConfig {
    /// Stream name for debugging
    pub name: String,
    /// Stream priority (0 = highest priority)
    pub priority: i32,
    /// Stream flags
    pub flags: u32,
}

impl Default for CudaStreamConfig {
    fn default() -> Self {
        Self {
            name: "default_stream".to_string(),
            priority: 0,
            flags: 0,
        }
    }
}

/// CUDA stream wrapper for asynchronous operations
pub struct CudaStream {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    #[cfg(feature = "cuda")]
    stream: CudarcStream,
    config: CudaStreamConfig,
    operation_count: std::sync::atomic::AtomicU64,
}

impl CudaStream {
    /// Create new CUDA stream
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, config: CudaStreamConfig) -> CudaResult<Self> {
        let stream = device.fork_default_stream()?;
        
        #[cfg(feature = "profiling")]
        tracing::debug!("Created CUDA stream: {}", config.name);
        
        Ok(Self {
            device,
            stream,
            config,
            operation_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_device: Arc<()>, _config: CudaStreamConfig) -> CudaResult<Self> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Get stream name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get stream priority
    pub fn priority(&self) -> i32 {
        self.config.priority
    }

    /// Synchronize stream (wait for all operations to complete)
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> CudaResult<()> {
        self.stream.synchronize()?;
        
        #[cfg(feature = "profiling")]
        tracing::debug!("Stream {} synchronized", self.config.name);
        
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn synchronize(&self) -> CudaResult<()> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Check if stream has completed all operations
    #[cfg(feature = "cuda")]
    pub fn is_ready(&self) -> CudaResult<bool> {
        // cudarc doesn't expose query functionality directly
        // In a real implementation, you would use cudaStreamQuery
        // For now, we'll assume synchronization is needed to check
        match self.stream.synchronize() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn is_ready(&self) -> CudaResult<bool> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Record operation on this stream
    pub fn record_operation(&self) {
        self.operation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get total operation count
    pub fn operation_count(&self) -> u64 {
        self.operation_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get raw CUDA stream handle for kernel launches
    #[cfg(feature = "cuda")]
    pub fn raw_stream(&self) -> &CudarcStream {
        &self.stream
    }

    /// Wait for another stream to complete
    #[cfg(feature = "cuda")]
    pub fn wait_for_stream(&self, other: &CudaStream) -> CudaResult<()> {
        // In a real implementation, you would use CUDA events
        // For now, synchronize the other stream
        other.synchronize()?;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn wait_for_stream(&self, _other: &CudaStream) -> CudaResult<()> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Get device associated with this stream
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

/// CUDA stream pool for managing multiple streams
pub struct CudaStreamPool {
    streams: Vec<CudaStream>,
    next_stream_index: std::sync::atomic::AtomicUsize,
}

impl CudaStreamPool {
    /// Create new stream pool
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, num_streams: usize) -> CudaResult<Self> {
        let mut streams = Vec::with_capacity(num_streams);
        
        for i in 0..num_streams {
            let config = CudaStreamConfig {
                name: format!("pool_stream_{}", i),
                priority: 0,
                flags: 0,
            };
            streams.push(CudaStream::new(device.clone(), config)?);
        }
        
        #[cfg(feature = "profiling")]
        tracing::info!("Created CUDA stream pool with {} streams", num_streams);
        
        Ok(Self {
            streams,
            next_stream_index: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_device: Arc<()>, _num_streams: usize) -> CudaResult<Self> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Get next available stream (round-robin)
    pub fn get_stream(&self) -> &CudaStream {
        let index = self.next_stream_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.streams[index % self.streams.len()]
    }

    /// Get stream by index
    pub fn get_stream_by_index(&self, index: usize) -> Option<&CudaStream> {
        self.streams.get(index)
    }

    /// Get number of streams in pool
    pub fn size(&self) -> usize {
        self.streams.len()
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get total operation count across all streams
    pub fn total_operation_count(&self) -> u64 {
        self.streams.iter()
            .map(|stream| stream.operation_count())
            .sum()
    }
}
