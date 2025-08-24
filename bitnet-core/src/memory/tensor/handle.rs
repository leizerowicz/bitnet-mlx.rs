//! Tensor Handle
//!
//! This module provides safe handles for accessing BitNet tensor data with
//! automatic reference counting and lifecycle management.

use crate::memory::tensor::{BitNetDType, TensorMetadata};
use crate::memory::{MemoryError, MemoryHandle};
use candle_core::Device;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock, Weak};
use thiserror::Error;

/// Errors that can occur during tensor handle operations
#[derive(Error, Debug)]
pub enum TensorHandleError {
    /// Handle has been invalidated (tensor was dropped)
    #[error("Tensor handle is invalid: tensor has been dropped")]
    InvalidHandle,

    /// Memory access error
    #[error("Memory access error: {0}")]
    MemoryError(#[from] MemoryError),

    /// Tensor is currently being migrated
    #[error("Tensor is currently being migrated between devices")]
    TensorMigrating,

    /// Device mismatch error
    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: String, actual: String },

    /// Shape mismatch error
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type mismatch error
    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DTypeMismatch {
        expected: BitNetDType,
        actual: BitNetDType,
    },

    /// Concurrent access error
    #[error("Concurrent access error: {reason}")]
    ConcurrentAccess { reason: String },
}

/// Result type for tensor handle operations
pub type TensorHandleResult<T> = std::result::Result<T, TensorHandleError>;

/// Safe handle for accessing BitNet tensor data
///
/// This handle provides safe access to tensor data with automatic reference counting
/// and lifecycle management. It ensures that the underlying tensor data remains valid
/// as long as the handle exists.
#[derive(Debug)]
pub struct TensorHandle {
    /// Weak reference to the tensor data to avoid cycles
    tensor_ref: Weak<TensorData>,
    /// Handle ID for debugging and tracking
    handle_id: u64,
    /// Creation timestamp for debugging
    created_at: std::time::Instant,
}

/// Global registry for tracking memory handles that need cleanup
pub static MEMORY_CLEANUP_REGISTRY: Lazy<Mutex<HashMap<u64, MemoryHandle>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global reference to the memory pool for automatic cleanup
pub static GLOBAL_MEMORY_POOL: Lazy<
    Mutex<Option<std::sync::Weak<crate::memory::HybridMemoryPool>>>,
> = Lazy::new(|| Mutex::new(None));

/// Sets the global memory pool reference for automatic cleanup
pub fn set_global_memory_pool(pool: std::sync::Weak<crate::memory::HybridMemoryPool>) {
    if let Ok(mut global_pool) = GLOBAL_MEMORY_POOL.lock() {
        *global_pool = Some(pool);
    }
}

/// Register a memory handle for automatic cleanup
pub fn register_memory_handle(tensor_id: u64, handle: MemoryHandle) {
    if let Ok(mut registry) = MEMORY_CLEANUP_REGISTRY.lock() {
        registry.insert(tensor_id, handle);
    }
}

/// Unregister and return a memory handle, triggering automatic cleanup
pub fn unregister_memory_handle(tensor_id: u64) -> Option<MemoryHandle> {
    #[cfg(feature = "tracing")]
    tracing::debug!("Unregistering memory handle for tensor {}", tensor_id);

    let handle = if let Ok(mut registry) = MEMORY_CLEANUP_REGISTRY.lock() {
        let handle = registry.remove(&tensor_id);
        #[cfg(feature = "tracing")]
        if handle.is_some() {
            tracing::debug!(
                "Found and removed handle for tensor {} from cleanup registry",
                tensor_id
            );
        } else {
            tracing::warn!(
                "No handle found in cleanup registry for tensor {}",
                tensor_id
            );
        }
        handle
    } else {
        #[cfg(feature = "tracing")]
        tracing::error!(
            "Failed to acquire cleanup registry lock for tensor {}",
            tensor_id
        );
        None
    };

    // If we have a handle, try to deallocate it immediately
    if let Some(ref handle) = handle {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Attempting immediate deallocation of handle {} for tensor {}",
            handle.id(),
            tensor_id
        );

        if let Ok(global_pool) = GLOBAL_MEMORY_POOL.lock() {
            if let Some(ref pool_weak) = *global_pool {
                if let Some(pool) = pool_weak.upgrade() {
                    // Try to deallocate the handle immediately
                    match pool.deallocate(handle.clone()) {
                        Ok(()) => {
                            #[cfg(feature = "tracing")]
                            tracing::debug!(
                                "Successfully deallocated handle {} for tensor {} immediately",
                                handle.id(),
                                tensor_id
                            );
                        }
                        Err(e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!("Immediate deallocation failed for handle {} (tensor {}): {}. Triggering cleanup.", handle.id(), tensor_id, e);

                            // If immediate deallocation fails, trigger cleanup
                            let cleaned_up = pool.cleanup_orphaned_handles();
                            if cleaned_up > 0 {
                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    "Automatically cleaned up {} orphaned memory handles",
                                    cleaned_up
                                );
                            } else {
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    "No orphaned handles found during cleanup for tensor {}",
                                    tensor_id
                                );
                            }
                        }
                    }
                } else {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        "Global memory pool weak reference could not be upgraded for tensor {}",
                        tensor_id
                    );
                }
            } else {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "No global memory pool reference set for tensor {}",
                    tensor_id
                );
            }
        } else {
            #[cfg(feature = "tracing")]
            tracing::error!(
                "Failed to acquire global memory pool lock for tensor {}",
                tensor_id
            );
        }
    }

    handle
}

/// Clears all global state - used for testing to prevent test interference
pub fn clear_global_state() {
    // Clear the memory cleanup registry
    if let Ok(mut registry) = MEMORY_CLEANUP_REGISTRY.lock() {
        let count = registry.len();
        registry.clear();
        #[cfg(feature = "tracing")]
        tracing::debug!("Cleared {} handles from global cleanup registry", count);
    }

    // Clear the global memory pool reference
    if let Ok(mut global_pool) = GLOBAL_MEMORY_POOL.lock() {
        *global_pool = None;
        #[cfg(feature = "tracing")]
        tracing::debug!("Cleared global memory pool reference");
    }
}

/// Internal tensor data structure
#[derive(Debug)]
pub struct TensorData {
    /// Memory handle for the tensor data
    pub memory_handle: MemoryHandle,
    /// Tensor metadata
    pub metadata: RwLock<TensorMetadata>,
    /// Unique tensor ID
    pub tensor_id: u64,
    /// Weak reference to the memory pool for cleanup (unused for now)
    pub pool_ref: std::sync::Weak<crate::memory::HybridMemoryPool>,
}

impl Drop for TensorData {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            "TensorData {} being dropped - memory handle {} will be cleaned up by pool",
            self.tensor_id,
            self.memory_handle.id()
        );

        // Remove from cleanup registry and trigger automatic cleanup
        if let Some(handle) = unregister_memory_handle(self.tensor_id) {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "TensorData {} dropped and memory handle {} unregistered from cleanup registry",
                self.tensor_id,
                handle.id()
            );
        } else {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "TensorData {} dropped but no handle found in cleanup registry (handle {})",
                self.tensor_id,
                self.memory_handle.id()
            );
        }

        // Try to use the pool reference for automatic cleanup
        if let Some(pool) = self.pool_ref.upgrade() {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "TensorData {} attempting additional cleanup via pool_ref",
                self.tensor_id
            );

            let cleaned_up = pool.cleanup_orphaned_handles();
            if cleaned_up > 0 {
                #[cfg(feature = "tracing")]
                tracing::debug!("Automatically cleaned up {} orphaned memory handles via pool_ref for tensor {}",
                        cleaned_up, self.tensor_id);
            } else {
                #[cfg(feature = "tracing")]
                tracing::debug!(
                    "No additional orphaned handles found via pool_ref for tensor {}",
                    self.tensor_id
                );
            }
        } else {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                "TensorData {} could not upgrade pool_ref for additional cleanup",
                self.tensor_id
            );
        }
    }
}

impl TensorHandle {
    /// Creates a new tensor handle
    pub(crate) fn new(tensor_data: Weak<TensorData>, handle_id: u64) -> Self {
        Self {
            tensor_ref: tensor_data,
            handle_id,
            created_at: std::time::Instant::now(),
        }
    }

    /// Returns the handle ID
    pub fn id(&self) -> u64 {
        self.handle_id
    }

    /// Returns the tensor ID if the handle is valid
    pub fn tensor_id(&self) -> TensorHandleResult<u64> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;
        Ok(tensor_data.tensor_id)
    }

    /// Returns true if the handle is still valid
    pub fn is_valid(&self) -> bool {
        self.tensor_ref.strong_count() > 0
    }

    /// Gets a copy of the tensor metadata
    pub fn metadata(&self) -> TensorHandleResult<TensorMetadata> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;

        let metadata =
            tensor_data
                .metadata
                .read()
                .map_err(|_| TensorHandleError::ConcurrentAccess {
                    reason: "Failed to acquire metadata read lock".to_string(),
                })?;

        Ok(metadata.clone())
    }

    /// Gets the tensor shape
    pub fn shape(&self) -> TensorHandleResult<Vec<usize>> {
        let metadata = self.metadata()?;
        Ok(metadata.shape)
    }

    /// Gets the tensor data type
    pub fn dtype(&self) -> TensorHandleResult<BitNetDType> {
        let metadata = self.metadata()?;
        Ok(metadata.dtype)
    }

    /// Gets the device where the tensor is stored
    pub fn device(&self) -> TensorHandleResult<Device> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;
        Ok(tensor_data.memory_handle.device())
    }

    /// Gets the size in bytes of the tensor
    pub fn size_bytes(&self) -> TensorHandleResult<usize> {
        let metadata = self.metadata()?;
        Ok(metadata.size_bytes)
    }

    /// Gets the number of elements in the tensor
    pub fn element_count(&self) -> TensorHandleResult<usize> {
        let metadata = self.metadata()?;
        Ok(metadata.element_count)
    }

    /// Gets the current reference count
    pub fn ref_count(&self) -> TensorHandleResult<usize> {
        let metadata = self.metadata()?;
        Ok(metadata.ref_count)
    }

    /// Returns true if the tensor is currently being migrated
    pub fn is_migrating(&self) -> TensorHandleResult<bool> {
        let metadata = self.metadata()?;
        Ok(metadata.is_migrating)
    }

    /// Gets the tensor name if set
    pub fn name(&self) -> TensorHandleResult<Option<String>> {
        let metadata = self.metadata()?;
        Ok(metadata.name)
    }

    /// Gets the tensor tags
    pub fn tags(&self) -> TensorHandleResult<Vec<String>> {
        let metadata = self.metadata()?;
        Ok(metadata.tags)
    }

    /// Updates the last accessed timestamp
    pub fn touch(&self) -> TensorHandleResult<()> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;

        let mut metadata =
            tensor_data
                .metadata
                .write()
                .map_err(|_| TensorHandleError::ConcurrentAccess {
                    reason: "Failed to acquire metadata write lock".to_string(),
                })?;

        metadata.touch();
        Ok(())
    }

    /// Adds a tag to the tensor
    pub fn add_tag(&self, tag: String) -> TensorHandleResult<()> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;

        let mut metadata =
            tensor_data
                .metadata
                .write()
                .map_err(|_| TensorHandleError::ConcurrentAccess {
                    reason: "Failed to acquire metadata write lock".to_string(),
                })?;

        metadata.add_tag(tag);
        Ok(())
    }

    /// Removes a tag from the tensor
    pub fn remove_tag(&self, tag: &str) -> TensorHandleResult<()> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;

        let mut metadata =
            tensor_data
                .metadata
                .write()
                .map_err(|_| TensorHandleError::ConcurrentAccess {
                    reason: "Failed to acquire metadata write lock".to_string(),
                })?;

        metadata.remove_tag(tag);
        Ok(())
    }

    /// Returns true if the tensor has the specified tag
    pub fn has_tag(&self, tag: &str) -> TensorHandleResult<bool> {
        let metadata = self.metadata()?;
        Ok(metadata.has_tag(tag))
    }

    /// Gets the age of the tensor in seconds
    pub fn age_seconds(&self) -> TensorHandleResult<u64> {
        let metadata = self.metadata()?;
        Ok(metadata.age_seconds())
    }

    /// Gets the idle time of the tensor in seconds
    pub fn idle_time_seconds(&self) -> TensorHandleResult<u64> {
        let metadata = self.metadata()?;
        Ok(metadata.idle_time_seconds())
    }

    /// Gets the handle age in seconds
    pub fn handle_age_seconds(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    /// Returns a description of the tensor
    pub fn description(&self) -> TensorHandleResult<String> {
        let metadata = self.metadata()?;
        Ok(metadata.description())
    }

    /// Validates that the tensor matches expected properties
    pub fn validate(
        &self,
        expected_shape: Option<&[usize]>,
        expected_dtype: Option<BitNetDType>,
    ) -> TensorHandleResult<()> {
        let metadata = self.metadata()?;

        // Check if tensor is being migrated
        if metadata.is_migrating {
            return Err(TensorHandleError::TensorMigrating);
        }

        // Validate shape if provided
        if let Some(expected) = expected_shape {
            if metadata.shape != expected {
                return Err(TensorHandleError::ShapeMismatch {
                    expected: expected.to_vec(),
                    actual: metadata.shape,
                });
            }
        }

        // Validate data type if provided
        if let Some(expected) = expected_dtype {
            if metadata.dtype != expected {
                return Err(TensorHandleError::DTypeMismatch {
                    expected,
                    actual: metadata.dtype,
                });
            }
        }

        Ok(())
    }

    /// Gets access to the underlying memory handle
    pub(crate) fn memory_handle(&self) -> TensorHandleResult<MemoryHandle> {
        let tensor_data = self
            .tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)?;
        Ok(tensor_data.memory_handle.clone())
    }

    /// Gets access to the tensor data for internal operations
    pub(crate) fn tensor_data(&self) -> TensorHandleResult<Arc<TensorData>> {
        self.tensor_ref
            .upgrade()
            .ok_or(TensorHandleError::InvalidHandle)
    }

    /// Creates a weak handle that doesn't affect reference counting
    pub fn downgrade(&self) -> WeakTensorHandle {
        WeakTensorHandle {
            tensor_ref: self.tensor_ref.clone(),
            handle_id: self.handle_id,
        }
    }
}

impl Clone for TensorHandle {
    fn clone(&self) -> Self {
        Self {
            tensor_ref: self.tensor_ref.clone(),
            handle_id: self.handle_id,
            created_at: std::time::Instant::now(), // New handle gets new timestamp
        }
    }
}

impl fmt::Display for TensorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.description() {
            Ok(desc) => write!(f, "TensorHandle({}): {}", self.handle_id, desc),
            Err(_) => write!(f, "TensorHandle({}): <invalid>", self.handle_id),
        }
    }
}

/// Weak tensor handle that doesn't affect reference counting
#[derive(Debug, Clone)]
pub struct WeakTensorHandle {
    tensor_ref: Weak<TensorData>,
    handle_id: u64,
}

impl WeakTensorHandle {
    /// Attempts to upgrade to a strong handle
    pub fn upgrade(&self) -> Option<TensorHandle> {
        if self.tensor_ref.strong_count() > 0 {
            Some(TensorHandle {
                tensor_ref: self.tensor_ref.clone(),
                handle_id: self.handle_id,
                created_at: std::time::Instant::now(),
            })
        } else {
            None
        }
    }

    /// Returns the handle ID
    pub fn id(&self) -> u64 {
        self.handle_id
    }

    /// Returns true if the handle can be upgraded
    pub fn is_valid(&self) -> bool {
        self.tensor_ref.strong_count() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;
    use std::sync::Arc;

    fn create_test_tensor_data() -> Arc<TensorData> {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let memory_handle = pool.allocate(64, 8, &device).unwrap();
        let metadata = TensorMetadata::new(
            1,
            vec![4, 4],
            BitNetDType::F32,
            &device,
            Some("test".to_string()),
        );

        Arc::new(TensorData {
            memory_handle,
            metadata: RwLock::new(metadata),
            tensor_id: 1,
            pool_ref: Arc::downgrade(&pool),
        })
    }

    #[test]
    fn test_tensor_handle_creation() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        assert_eq!(handle.id(), 100);
        assert!(handle.is_valid());
        assert_eq!(handle.tensor_id().unwrap(), 1);
    }

    #[test]
    fn test_tensor_handle_metadata_access() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        assert_eq!(handle.shape().unwrap(), vec![4, 4]);
        assert_eq!(handle.dtype().unwrap(), BitNetDType::F32);
        assert_eq!(handle.element_count().unwrap(), 16);
        assert_eq!(handle.name().unwrap(), Some("test".to_string()));
    }

    #[test]
    fn test_tensor_handle_validation() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        // Valid validation
        assert!(handle
            .validate(Some(&[4, 4]), Some(BitNetDType::F32))
            .is_ok());

        // Invalid shape
        assert!(handle.validate(Some(&[2, 2]), None).is_err());

        // Invalid dtype
        assert!(handle.validate(None, Some(BitNetDType::I8)).is_err());
    }

    #[test]
    fn test_tensor_handle_tags() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        assert!(!handle.has_tag("new_tag").unwrap());

        handle.add_tag("new_tag".to_string()).unwrap();
        assert!(handle.has_tag("new_tag").unwrap());

        handle.remove_tag("new_tag").unwrap();
        assert!(!handle.has_tag("new_tag").unwrap());
    }

    #[test]
    fn test_tensor_handle_invalidation() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        assert!(handle.is_valid());

        // Drop the tensor data
        drop(tensor_data);

        assert!(!handle.is_valid());
        assert!(handle.metadata().is_err());
    }

    #[test]
    fn test_weak_tensor_handle() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle = TensorHandle::new(weak_ref, 100);

        let weak_handle = handle.downgrade();
        assert!(weak_handle.is_valid());
        assert_eq!(weak_handle.id(), 100);

        let upgraded = weak_handle.upgrade().unwrap();
        assert_eq!(upgraded.id(), 100);

        // Drop original data
        drop(tensor_data);
        drop(handle);

        assert!(!weak_handle.is_valid());
        assert!(weak_handle.upgrade().is_none());
    }

    #[test]
    fn test_handle_cloning() {
        let tensor_data = create_test_tensor_data();
        let weak_ref = Arc::downgrade(&tensor_data);
        let handle1 = TensorHandle::new(weak_ref, 100);
        let handle2 = handle1.clone();

        assert_eq!(handle1.id(), handle2.id());
        assert_eq!(handle1.tensor_id().unwrap(), handle2.tensor_id().unwrap());

        // Both handles should be valid
        assert!(handle1.is_valid());
        assert!(handle2.is_valid());
    }
}
