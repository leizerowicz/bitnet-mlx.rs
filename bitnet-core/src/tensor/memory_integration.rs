//! Tensor Memory Integration with HybridMemoryPool
//!
//! This module provides seamless integration between BitNet tensors and the
//! existing HybridMemoryPool infrastructure, enabling efficient memory management
//! with automatic cleanup and reference counting.

use super::dtype::BitNetDType;
use crate::memory::{HybridMemoryPool, MemoryError, MemoryHandle, MemoryResult};
use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

/// Global memory pool for tensor operations
static GLOBAL_MEMORY_POOL: Mutex<Option<Weak<HybridMemoryPool>>> = Mutex::new(None);

/// Registry for automatic cleanup of tensor memory handles
static MEMORY_CLEANUP_REGISTRY: LazyLock<Mutex<HashMap<u64, MemoryHandle>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Counter for generating unique tensor IDs
static TENSOR_ID_COUNTER: Mutex<u64> = Mutex::new(1);

/// Tensor memory manager that integrates with HybridMemoryPool
///
/// This manager provides high-level memory operations for tensors while
/// leveraging the sophisticated memory pool infrastructure.
#[derive(Debug)]
pub struct TensorMemoryManager {
    /// Reference to the memory pool
    pool: Arc<HybridMemoryPool>,
    /// Device for memory allocations
    device: Device,
    /// Registry of active tensor memory handles
    handle_registry: Arc<Mutex<HashMap<u64, MemoryHandle>>>,
    /// Next tensor ID for tracking
    next_tensor_id: Arc<Mutex<u64>>,
}

impl TensorMemoryManager {
    /// Creates a new tensor memory manager
    ///
    /// # Arguments
    ///
    /// * `pool` - Arc to the HybridMemoryPool to use
    /// * `device` - Device for memory allocations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorMemoryManager;
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// let device = get_cpu_device();
    /// let manager = TensorMemoryManager::new(pool, device);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(pool: Arc<HybridMemoryPool>, device: Device) -> Self {
        #[cfg(feature = "tracing")]
        info!("Creating tensor memory manager for device {:?}", device);

        Self {
            pool,
            device,
            handle_registry: Arc::new(Mutex::new(HashMap::new())),
            next_tensor_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Allocates memory for tensor data
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Number of bytes to allocate
    /// * `alignment` - Memory alignment requirement
    /// * `dtype` - Data type for the tensor (for debugging/tracking)
    ///
    /// # Returns
    ///
    /// Result containing (tensor_id, MemoryHandle) or error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{TensorMemoryManager, BitNetDType};
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// let device = get_cpu_device();
    /// let manager = TensorMemoryManager::new(pool, device);
    ///
    /// let (tensor_id, handle) = manager.allocate_tensor_memory(1024, 16, BitNetDType::F32)?;
    /// println!("Allocated tensor {} with {} bytes", tensor_id, handle.size());
    /// # Ok(())
    /// # }
    /// ```
    pub fn allocate_tensor_memory(
        &self,
        size_bytes: usize,
        alignment: usize,
        dtype: BitNetDType,
    ) -> MemoryResult<(u64, MemoryHandle)> {
        // Generate unique tensor ID
        let tensor_id = {
            let mut counter =
                self.next_tensor_id
                    .lock()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire tensor ID counter lock".to_string(),
                    })?;
            let id = *counter;
            *counter += 1;
            id
        };

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating tensor memory: id={}, size={} bytes, alignment={}, dtype={:?}, device={:?}",
            tensor_id, size_bytes, alignment, dtype, self.device
        );

        // Allocate memory using HybridMemoryPool
        let handle = self.pool.allocate(size_bytes, alignment, &self.device)?;

        // Register the handle for cleanup tracking
        {
            let mut registry =
                self.handle_registry
                    .lock()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire handle registry lock".to_string(),
                    })?;
            registry.insert(tensor_id, handle.clone());
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Successfully allocated tensor memory: id={}, handle_id={}, size={} bytes",
            tensor_id,
            handle.id(),
            handle.size()
        );

        Ok((tensor_id, handle))
    }

    /// Deallocates tensor memory
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - ID of the tensor to deallocate
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bitnet_core::tensor::{TensorMemoryManager, BitNetDType};
    /// # use bitnet_core::memory::HybridMemoryPool;
    /// # use bitnet_core::device::get_cpu_device;
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let pool = Arc::new(HybridMemoryPool::new()?);
    /// # let device = get_cpu_device();
    /// # let manager = TensorMemoryManager::new(pool, device);
    /// # let (tensor_id, handle) = manager.allocate_tensor_memory(1024, 16, BitNetDType::F32)?;
    ///
    /// manager.deallocate_tensor_memory(tensor_id)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn deallocate_tensor_memory(&self, tensor_id: u64) -> MemoryResult<()> {
        #[cfg(feature = "tracing")]
        debug!("Deallocating tensor memory: id={}", tensor_id);

        // Remove from registry and get handle
        let handle = {
            let mut registry =
                self.handle_registry
                    .lock()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire handle registry lock".to_string(),
                    })?;

            registry
                .remove(&tensor_id)
                .ok_or_else(|| MemoryError::InvalidHandle {
                    reason: format!("Tensor ID {} not found in registry", tensor_id),
                })?
        };

        #[cfg(feature = "tracing")]
        debug!(
            "Deallocating memory for tensor {}: handle_id={}, size={} bytes",
            tensor_id,
            handle.id(),
            handle.size()
        );

        // Deallocate using HybridMemoryPool
        self.pool.deallocate(handle)?;

        #[cfg(feature = "tracing")]
        debug!("Successfully deallocated tensor memory: id={}", tensor_id);

        Ok(())
    }

    /// Gets memory handle for a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - ID of the tensor
    ///
    /// # Returns
    ///
    /// Option containing MemoryHandle if tensor exists
    pub fn get_tensor_handle(&self, tensor_id: u64) -> Option<MemoryHandle> {
        if let Ok(registry) = self.handle_registry.lock() {
            registry.get(&tensor_id).cloned()
        } else {
            None
        }
    }

    /// Returns memory pool metrics
    pub fn get_memory_metrics(&self) -> crate::memory::MemoryMetrics {
        self.pool.get_metrics()
    }

    /// Returns detailed memory metrics if available
    pub fn get_detailed_memory_metrics(&self) -> Option<crate::memory::MemoryMetrics> {
        // For now, return the basic metrics since detailed metrics structure may not be available
        Some(self.pool.get_metrics())
    }

    /// Returns the number of active tensor memory allocations
    pub fn active_tensor_count(&self) -> usize {
        if let Ok(registry) = self.handle_registry.lock() {
            registry.len()
        } else {
            0
        }
    }

    /// Returns the device associated with this memory manager
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns reference to the underlying memory pool
    pub fn memory_pool(&self) -> &Arc<HybridMemoryPool> {
        &self.pool
    }

    /// Validates tensor memory integrity
    pub fn validate_tensor_memory(&self, tensor_id: u64) -> MemoryResult<()> {
        if let Some(handle) = self.get_tensor_handle(tensor_id) {
            handle.validate()?;
            Ok(())
        } else {
            Err(MemoryError::InvalidHandle {
                reason: format!("Tensor ID {} not found", tensor_id),
            })
        }
    }

    /// Cleanup all tensor memory allocations
    pub fn cleanup_all_tensors(&self) -> MemoryResult<()> {
        #[cfg(feature = "tracing")]
        info!("Cleaning up all tensor memory allocations");

        let handles = {
            let mut registry =
                self.handle_registry
                    .lock()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire handle registry lock".to_string(),
                    })?;
            let handles: Vec<_> = registry.drain().collect();
            handles
        };

        let mut errors = Vec::new();
        for (tensor_id, handle) in handles {
            if let Err(e) = self.pool.deallocate(handle) {
                #[cfg(feature = "tracing")]
                error!("Failed to deallocate tensor {}: {}", tensor_id, e);
                errors.push(e);
            }
        }

        if !errors.is_empty() {
            #[cfg(feature = "tracing")]
            error!("Failed to cleanup {} tensor allocations", errors.len());
            return Err(errors.into_iter().next().unwrap());
        }

        #[cfg(feature = "tracing")]
        info!("Successfully cleaned up all tensor memory allocations");

        Ok(())
    }
}

impl Drop for TensorMemoryManager {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup_all_tensors() {
            #[cfg(feature = "tracing")]
            error!("Error during tensor memory manager cleanup: {}", e);
        }
    }
}

/// Global memory pool functions
impl TensorMemoryManager {
    /// Sets the global memory pool for tensor operations
    ///
    /// # Arguments
    ///
    /// * `pool` - Weak reference to the HybridMemoryPool
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::set_global_memory_pool;
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// set_global_memory_pool(Arc::downgrade(&pool));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_global_pool(pool: Weak<HybridMemoryPool>) {
        if let Ok(mut global_pool) = GLOBAL_MEMORY_POOL.lock() {
            *global_pool = Some(pool);

            #[cfg(feature = "tracing")]
            info!("Global tensor memory pool set");
        }
    }

    /// Gets the global memory pool for tensor operations
    ///
    /// # Returns
    ///
    /// Option containing Arc to the global HybridMemoryPool
    pub fn get_global_pool() -> Option<Arc<HybridMemoryPool>> {
        if let Ok(global_pool) = GLOBAL_MEMORY_POOL.lock() {
            if let Some(ref weak_pool) = *global_pool {
                weak_pool.upgrade()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Clears the global memory pool reference
    pub fn clear_global_pool() {
        if let Ok(mut global_pool) = GLOBAL_MEMORY_POOL.lock() {
            *global_pool = None;

            #[cfg(feature = "tracing")]
            info!("Global tensor memory pool cleared");
        }
    }
}

/// Global convenience functions
/// Sets the global memory pool for tensor operations
pub fn set_global_memory_pool(pool: Weak<HybridMemoryPool>) {
    TensorMemoryManager::set_global_pool(pool);
}

/// Gets the global memory pool for tensor operations
pub fn get_global_memory_pool() -> Option<Arc<HybridMemoryPool>> {
    TensorMemoryManager::get_global_pool()
}

/// Clears the global memory pool reference
pub fn clear_global_memory_pool() {
    TensorMemoryManager::clear_global_pool();
}

/// Allocates memory for a tensor using the global pool
pub fn allocate_tensor_memory_global(
    size_bytes: usize,
    alignment: usize,
    device: &Device,
    dtype: BitNetDType,
) -> MemoryResult<(u64, MemoryHandle)> {
    let pool = get_global_memory_pool().ok_or_else(|| MemoryError::InternalError {
        reason: "Global memory pool not initialized".to_string(),
    })?;

    let manager = TensorMemoryManager::new(pool, device.clone());
    manager.allocate_tensor_memory(size_bytes, alignment, dtype)
}

/// Generates a unique tensor ID
pub fn generate_tensor_id() -> u64 {
    if let Ok(mut counter) = TENSOR_ID_COUNTER.lock() {
        let id = *counter;
        *counter += 1;
        id
    } else {
        // Fallback to timestamp-based ID if lock fails
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

/// Registers a memory handle for automatic cleanup
pub fn register_tensor_memory(tensor_id: u64, handle: MemoryHandle) {
    if let Ok(mut registry) = MEMORY_CLEANUP_REGISTRY.lock() {
        registry.insert(tensor_id, handle);

        #[cfg(feature = "tracing")]
        debug!("Registered tensor memory for cleanup: id={}", tensor_id);
    }
}

/// Unregisters and cleans up tensor memory
pub fn cleanup_tensor_memory(tensor_id: u64) -> MemoryResult<()> {
    let handle = {
        if let Ok(mut registry) = MEMORY_CLEANUP_REGISTRY.lock() {
            registry.remove(&tensor_id)
        } else {
            return Err(MemoryError::InternalError {
                reason: "Failed to acquire cleanup registry lock".to_string(),
            });
        }
    };

    if let Some(handle) = handle {
        if let Some(pool) = get_global_memory_pool() {
            pool.deallocate(handle)?;

            #[cfg(feature = "tracing")]
            debug!("Cleaned up tensor memory: id={}", tensor_id);
        }
    }

    Ok(())
}

/// Gets memory usage statistics for all registered tensors
pub fn get_tensor_memory_stats() -> (usize, usize) {
    if let Ok(registry) = MEMORY_CLEANUP_REGISTRY.lock() {
        let count = registry.len();
        let total_bytes = registry.values().map(|handle| handle.size()).sum();
        (count, total_bytes)
    } else {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_tensor_memory_manager_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorMemoryManager::new(pool, device);

        assert_eq!(manager.active_tensor_count(), 0);
        assert!(matches!(manager.device(), Device::Cpu));
    }

    #[test]
    fn test_tensor_memory_allocation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorMemoryManager::new(pool, device);

        let (tensor_id, handle) = manager
            .allocate_tensor_memory(1024, 16, BitNetDType::F32)
            .unwrap();
        assert_eq!(handle.size(), 1024);
        assert_eq!(handle.alignment(), 16);
        assert_eq!(manager.active_tensor_count(), 1);

        // Validate memory
        manager.validate_tensor_memory(tensor_id).unwrap();

        // Cleanup
        manager.deallocate_tensor_memory(tensor_id).unwrap();
        assert_eq!(manager.active_tensor_count(), 0);
    }

    #[test]
    fn test_global_memory_pool() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&pool));

        let retrieved_pool = get_global_memory_pool().unwrap();
        assert!(Arc::ptr_eq(&pool, &retrieved_pool));

        clear_global_memory_pool();
        assert!(get_global_memory_pool().is_none());
    }

    #[test]
    fn test_global_tensor_allocation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&pool));

        let device = get_cpu_device();
        let (tensor_id, handle) =
            allocate_tensor_memory_global(512, 8, &device, BitNetDType::I32).unwrap();

        assert_eq!(handle.size(), 512);
        assert_eq!(handle.alignment(), 8);

        // Cleanup
        cleanup_tensor_memory(tensor_id).unwrap();

        clear_global_memory_pool();
    }

    #[test]
    fn test_tensor_id_generation() {
        let id1 = generate_tensor_id();
        let id2 = generate_tensor_id();
        assert!(id2 > id1);
    }

    #[test]
    fn test_tensor_memory_stats() {
        let (count, bytes) = get_tensor_memory_stats();
        // Should start with clean state
        assert_eq!(count, 0);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn test_memory_manager_cleanup() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        {
            let manager = TensorMemoryManager::new(pool, device);
            let _allocation = manager
                .allocate_tensor_memory(256, 4, BitNetDType::U8)
                .unwrap();
            assert_eq!(manager.active_tensor_count(), 1);
            // manager drops here and should cleanup automatically
        }

        // After drop, allocations should be cleaned up
        // (This is hard to test directly, but the Drop impl should handle it)
    }
}
