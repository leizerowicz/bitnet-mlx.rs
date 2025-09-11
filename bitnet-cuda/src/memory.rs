//! CUDA memory management for BitNet operations

use crate::error::{CudaError, CudaResult};

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// CUDA memory allocation handle
pub struct CudaAllocation<T> {
    #[cfg(feature = "cuda")]
    slice: CudaSlice<T>,
    size_bytes: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> CudaAllocation<T> {
    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        std::mem::size_of::<T>().max(1) / self.size_bytes
    }

    /// Check if allocation is empty
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }

    /// Get underlying CUDA slice
    #[cfg(feature = "cuda")]
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get mutable CUDA slice
    #[cfg(feature = "cuda")]
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

/// CUDA memory manager for efficient allocation and deallocation
pub struct CudaMemoryManager {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    pool_size_bytes: usize,
    allocated_bytes: std::sync::atomic::AtomicUsize,
    peak_usage_bytes: std::sync::atomic::AtomicUsize,
    allocation_count: std::sync::atomic::AtomicU64,
}

impl CudaMemoryManager {
    /// Create new CUDA memory manager
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, pool_size_bytes: usize) -> CudaResult<Self> {
        #[cfg(feature = "profiling")]
        tracing::info!("CUDA memory manager initialized with {}MB pool", pool_size_bytes / 1024 / 1024);
        
        Ok(Self {
            device,
            pool_size_bytes,
            allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
            peak_usage_bytes: std::sync::atomic::AtomicUsize::new(0),
            allocation_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_device: Arc<()>, _pool_size_bytes: usize) -> CudaResult<Self> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Allocate GPU memory
    #[cfg(feature = "cuda")]
    pub fn allocate<T: Clone>(&self, count: usize) -> CudaResult<CudaAllocation<T>> {
        let size_bytes = count * std::mem::size_of::<T>();
        
        // Check if allocation would exceed pool size
        let current_allocated = self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed);
        if current_allocated + size_bytes > self.pool_size_bytes {
            return Err(CudaError::MemoryAllocation(
                format!("Allocation would exceed pool size: {} + {} > {}", 
                    current_allocated, size_bytes, self.pool_size_bytes)
            ));
        }
        
        // Allocate using cudarc
        let slice = self.device.alloc_zeros::<T>(count)?;
        
        // Update tracking
        self.allocated_bytes.fetch_add(size_bytes, std::sync::atomic::Ordering::Relaxed);
        self.allocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Update peak usage
        let new_allocated = self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed);
        let mut peak = self.peak_usage_bytes.load(std::sync::atomic::Ordering::Relaxed);
        while new_allocated > peak {
            match self.peak_usage_bytes.compare_exchange_weak(
                peak,
                new_allocated,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
        
        #[cfg(feature = "profiling")]
        tracing::debug!("Allocated {} bytes on GPU (total: {})", size_bytes, new_allocated);
        
        Ok(CudaAllocation {
            slice,
            size_bytes,
            _phantom: std::marker::PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn allocate<T>(&self, _count: usize) -> CudaResult<CudaAllocation<T>> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Copy data from host to device
    #[cfg(feature = "cuda")]
    pub fn copy_to_device<T: Clone>(&self, host_data: &[T]) -> CudaResult<CudaAllocation<T>> {
        let mut allocation = self.allocate::<T>(host_data.len())?;
        self.device.htod_copy_into(host_data, &mut allocation.slice)?;
        Ok(allocation)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn copy_to_device<T>(&self, _host_data: &[T]) -> CudaResult<CudaAllocation<T>> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Copy data from device to host
    #[cfg(feature = "cuda")]
    pub fn copy_to_host<T: Clone>(&self, allocation: &CudaAllocation<T>) -> CudaResult<Vec<T>> {
        let mut host_data = vec![allocation.slice[0].clone(); allocation.len()];
        self.device.dtoh_sync_copy_into(&allocation.slice, &mut host_data)?;
        Ok(host_data)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn copy_to_host<T>(&self, _allocation: &CudaAllocation<T>) -> CudaResult<Vec<T>> {
        Err(CudaError::CudaNotEnabled)
    }

    /// Get current memory utilization percentage
    pub fn utilization_percent(&self) -> f32 {
        let allocated = self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed);
        if self.pool_size_bytes > 0 {
            (allocated as f32 / self.pool_size_bytes as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Get peak memory usage in bytes
    pub fn peak_usage_bytes(&self) -> usize {
        self.peak_usage_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total allocation count
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get available memory in bytes
    pub fn available_bytes(&self) -> usize {
        let allocated = self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed);
        self.pool_size_bytes.saturating_sub(allocated)
    }

    /// Clear all tracking statistics (for testing)
    pub fn reset_stats(&self) {
        self.allocated_bytes.store(0, std::sync::atomic::Ordering::Relaxed);
        self.peak_usage_bytes.store(0, std::sync::atomic::Ordering::Relaxed);
        self.allocation_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

// Implement Drop to track memory deallocation
#[cfg(feature = "cuda")]
impl<T> Drop for CudaAllocation<T> {
    fn drop(&mut self) {
        // Memory is automatically freed by cudarc when CudaSlice is dropped
        // We don't have direct access to the memory manager here, but in a 
        // production implementation you would want to update allocation tracking
    }
}
