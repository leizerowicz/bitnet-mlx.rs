//! BitNet Tensor Implementation
//!
//! This module provides the core BitNetTensor implementation with automatic
//! reference counting, lifecycle management, and device-aware operations.

use crate::memory::tensor::handle::TensorHandleError;
use crate::memory::tensor::{BitNetDType, TensorData, TensorHandle, TensorMetadata};
use crate::memory::{HybridMemoryPool, MemoryError, MemoryHandle};
use candle_core::{Device, Tensor};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Weak};
use thiserror::Error;

#[cfg(feature = "tracing")]
use tracing::{debug, error, info};

/// Errors that can occur during BitNet tensor operations
#[derive(Error, Debug)]
pub enum BitNetTensorError {
    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryError(#[from] MemoryError),

    /// Tensor handle error
    #[error("Tensor handle error: {0}")]
    HandleError(#[from] TensorHandleError),

    /// Candle tensor operation error
    #[error("Candle tensor error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Shape mismatch error
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type conversion error
    #[error("Data type conversion error: cannot convert {from} to {to}")]
    DTypeConversionError { from: String, to: String },

    /// Device migration error
    #[error("Device migration failed: {reason}")]
    DeviceMigrationError { reason: String },

    /// Invalid tensor operation
    #[error("Invalid tensor operation: {reason}")]
    InvalidOperation { reason: String },

    /// Tensor not found
    #[error("Tensor not found: ID {id}")]
    TensorNotFound { id: u64 },

    /// Concurrent access error
    #[error("Concurrent access error: {reason}")]
    ConcurrentAccess { reason: String },
}

/// Result type for BitNet tensor operations
pub type BitNetTensorResult<T> = std::result::Result<T, BitNetTensorError>;

/// BitNet tensor with automatic reference counting and lifecycle management
#[derive(Debug)]
#[allow(dead_code)]
pub struct BitNetTensor {
    /// Strong reference to tensor data
    pub(crate) data: Arc<TensorData>,
    /// Tensor registry for global management
    registry: Arc<RwLock<TensorRegistry>>,
    /// Next tensor ID counter
    next_id: Arc<Mutex<u64>>,
}

/// Global tensor registry for tracking all tensors
#[derive(Debug, Default)]
struct TensorRegistry {
    /// Map of tensor ID to weak references
    tensors: HashMap<u64, Weak<TensorData>>,
    /// Next handle ID counter
    next_handle_id: u64,
}

/// Global tensor registry instance
static GLOBAL_TENSOR_REGISTRY: Lazy<Arc<RwLock<TensorRegistry>>> =
    Lazy::new(|| Arc::new(RwLock::new(TensorRegistry::default())));

/// Global tensor ID counter
static GLOBAL_TENSOR_ID_COUNTER: Lazy<Arc<Mutex<u64>>> = Lazy::new(|| Arc::new(Mutex::new(1)));

impl TensorRegistry {
    fn register_tensor(&mut self, tensor_id: u64, tensor_data: &Arc<TensorData>) {
        self.tensors.insert(tensor_id, Arc::downgrade(tensor_data));
    }

    fn unregister_tensor(&mut self, tensor_id: u64) {
        self.tensors.remove(&tensor_id);
    }

    #[allow(dead_code)]
    fn get_tensor(&self, tensor_id: u64) -> Option<Arc<TensorData>> {
        self.tensors.get(&tensor_id)?.upgrade()
    }

    fn next_handle_id(&mut self) -> u64 {
        self.next_handle_id += 1;
        self.next_handle_id
    }

    fn cleanup_dead_references(&mut self) {
        self.tensors
            .retain(|_, weak_ref| weak_ref.strong_count() > 0);
    }
}

impl BitNetTensor {
    /// Creates a new tensor filled with zeros
    pub fn zeros(
        shape: &[usize],
        dtype: BitNetDType,
        device: &Device,
        pool: &HybridMemoryPool,
    ) -> BitNetTensorResult<Self> {
        let element_count: usize = shape.iter().product();
        let size_bytes = dtype.bytes_for_elements(element_count);

        #[cfg(feature = "tracing")]
        debug!(
            "Creating zeros tensor: shape={:?}, dtype={}, size={} bytes",
            shape, dtype, size_bytes
        );

        // Allocate memory
        let memory_handle = pool.allocate(size_bytes, 16, device)?;

        // Create tensor and initialize with zeros
        let tensor = Self::from_memory_handle(memory_handle, shape.to_vec(), dtype, None, pool)?;

        // Initialize memory with zeros
        unsafe {
            let ptr = tensor.data.memory_handle.as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, size_bytes);
        }

        Ok(tensor)
    }

    /// Creates a new tensor filled with ones
    pub fn ones(
        shape: &[usize],
        dtype: BitNetDType,
        device: &Device,
        pool: &HybridMemoryPool,
    ) -> BitNetTensorResult<Self> {
        let element_count: usize = shape.iter().product();
        let size_bytes = dtype.bytes_for_elements(element_count);

        #[cfg(feature = "tracing")]
        debug!(
            "Creating ones tensor: shape={:?}, dtype={}, size={} bytes",
            shape, dtype, size_bytes
        );

        // Allocate memory
        let memory_handle = pool.allocate(size_bytes, 16, device)?;

        // Create tensor and initialize with ones
        let tensor = Self::from_memory_handle(memory_handle, shape.to_vec(), dtype, None, pool)?;

        // Initialize memory with ones based on dtype
        unsafe {
            let ptr = tensor.data.memory_handle.as_ptr() as *mut u8;
            Self::initialize_ones(ptr, element_count, dtype);
        }

        Ok(tensor)
    }

    /// Creates a tensor from existing data
    pub fn from_data(
        data: Vec<f32>,
        shape: &[usize],
        device: &Device,
        pool: &HybridMemoryPool,
    ) -> BitNetTensorResult<Self> {
        let element_count: usize = shape.iter().product();
        if data.len() != element_count {
            return Err(BitNetTensorError::ShapeMismatch {
                expected: vec![data.len()],
                actual: shape.to_vec(),
            });
        }

        let dtype = BitNetDType::F32;
        let size_bytes = dtype.bytes_for_elements(element_count);

        #[cfg(feature = "tracing")]
        debug!(
            "Creating tensor from data: shape={:?}, dtype={}, size={} bytes",
            shape, dtype, size_bytes
        );

        // Allocate memory
        let memory_handle = pool.allocate(size_bytes, 16, device)?;

        // Create tensor
        let tensor = Self::from_memory_handle(memory_handle, shape.to_vec(), dtype, None, pool)?;

        // Copy data to memory handle
        unsafe {
            let ptr = tensor.data.memory_handle.as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        Ok(tensor)
    }

    /// Creates a tensor from a memory handle
    fn from_memory_handle(
        memory_handle: MemoryHandle,
        shape: Vec<usize>,
        dtype: BitNetDType,
        name: Option<String>,
        _pool: &HybridMemoryPool,
    ) -> BitNetTensorResult<Self> {
        // Use global registry and ID counter
        let registry = Arc::clone(&GLOBAL_TENSOR_REGISTRY);
        let next_id = Arc::clone(&GLOBAL_TENSOR_ID_COUNTER);

        // Get next tensor ID
        let tensor_id = {
            let mut id_counter =
                next_id
                    .lock()
                    .map_err(|_| BitNetTensorError::ConcurrentAccess {
                        reason: "Failed to acquire ID counter lock".to_string(),
                    })?;
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        // Create metadata
        let device = crate::device::auto_select_device(); // Fixed - use auto-selected device
        let metadata = TensorMetadata::new(tensor_id, shape, dtype, &device, name);

        // For now, create a dummy weak reference that will never upgrade
        // This is a temporary solution - we'll implement proper cleanup differently
        let pool_ref = std::sync::Weak::new();

        // Register the memory handle for cleanup
        crate::memory::tensor::handle::register_memory_handle(tensor_id, memory_handle.clone());

        // Create tensor data
        let tensor_data = Arc::new(TensorData {
            memory_handle,
            metadata: RwLock::new(metadata),
            tensor_id: tensor_id,
            pool_ref,
        });

        // Register tensor
        {
            let mut reg = registry
                .write()
                .map_err(|_| BitNetTensorError::ConcurrentAccess {
                    reason: "Failed to acquire registry write lock".to_string(),
                })?;
            reg.register_tensor(tensor_id, &tensor_data);
        }

        #[cfg(feature = "tracing")]
        info!("Created BitNet tensor with ID {}", tensor_id);

        Ok(Self {
            data: tensor_data,
            registry,
            next_id,
        })
    }

    /// Creates a tensor from a candle tensor
    pub fn from_candle(tensor: Tensor, pool: &HybridMemoryPool) -> BitNetTensorResult<Self> {
        let shape = tensor.shape().dims().to_vec();
        let candle_dtype = tensor.dtype();
        let device = tensor.device().clone();

        // Convert candle dtype to BitNet dtype
        let dtype = BitNetDType::from_candle_dtype(candle_dtype).ok_or_else(|| {
            BitNetTensorError::DTypeConversionError {
                from: format!("{:?}", candle_dtype),
                to: "BitNetDType".to_string(),
            }
        })?;

        #[cfg(feature = "tracing")]
        debug!(
            "Converting candle tensor: shape={:?}, dtype={:?} -> {}",
            shape, candle_dtype, dtype
        );

        // Create a new tensor and copy data
        let bitnet_tensor = match dtype {
            BitNetDType::F32 => {
                // For F32, we can copy data directly
                let data_vec = tensor.flatten_all()?.to_vec1::<f32>()?;
                Self::from_data(data_vec, &shape, &device, pool)?
            }
            _ => {
                // For other types, create zeros for now
                // TODO: Implement conversion for other data types
                Self::zeros(&shape, dtype, &device, pool)?
            }
        };

        Ok(bitnet_tensor)
    }

    /// Converts the tensor to a candle tensor
    pub fn to_candle(&self) -> BitNetTensorResult<Tensor> {
        let metadata =
            self.data
                .metadata
                .read()
                .map_err(|_| BitNetTensorError::ConcurrentAccess {
                    reason: "Failed to acquire metadata read lock".to_string(),
                })?;

        let candle_dtype = metadata.dtype.to_candle_dtype();
        let device = self.device(); // Fixed - get device from self
        let shape = &metadata.shape;

        #[cfg(feature = "tracing")]
        debug!(
            "Converting to candle tensor: shape={:?}, dtype={}",
            shape, metadata.dtype
        );

        // Create tensor and copy data from BitNet tensor
        let tensor = match metadata.dtype {
            BitNetDType::F32 => {
                // Direct copy for F32
                let data_slice = unsafe {
                    let ptr = self.data.memory_handle.as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, metadata.element_count)
                };
                Tensor::from_slice(data_slice, shape.as_slice(), &device)?
            }
            _ => {
                // For other types, create zeros for now
                // TODO: Implement conversion for other data types
                Tensor::zeros(shape.as_slice(), candle_dtype, &device)?
            }
        };

        Ok(tensor)
    }

    /// Migrates the tensor to a different device
    pub fn to_device(
        &self,
        target_device: &Device,
        pool: &HybridMemoryPool,
    ) -> BitNetTensorResult<Self> {
        let current_device = self.device(); // Fixed - get current device from self

        // Check if already on target device
        if std::mem::discriminant(&current_device) == std::mem::discriminant(target_device) {
            #[cfg(feature = "tracing")]
            debug!("Tensor already on target device, returning clone");
            return Ok(self.clone());
        }

        #[cfg(feature = "tracing")]
        info!(
            "Migrating tensor from {:?} to {:?}",
            current_device, target_device
        );

        // Mark as migrating
        {
            let mut metadata =
                self.data
                    .metadata
                    .write()
                    .map_err(|_| BitNetTensorError::ConcurrentAccess {
                        reason: "Failed to acquire metadata write lock".to_string(),
                    })?;
            metadata.set_migrating(true);
        }

        // Get current metadata
        let (shape, dtype, _name) = {
            let metadata =
                self.data
                    .metadata
                    .read()
                    .map_err(|_| BitNetTensorError::ConcurrentAccess {
                        reason: "Failed to acquire metadata read lock".to_string(),
                    })?;
            (
                metadata.shape.clone(),
                metadata.dtype,
                metadata.name.clone(),
            )
        };

        // Create new tensor on target device
        let new_tensor = Self::zeros(&shape, dtype, target_device, pool)?;

        // Copy data from source to target device
        unsafe {
            let src_ptr = self.data.memory_handle.as_ptr();
            let dst_ptr = new_tensor.data.memory_handle.as_ptr() as *mut u8;
            let size_bytes = self.size_bytes();

            // For now, use simple memory copy
            // TODO: Implement device-specific optimized copy operations
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_bytes);
        }

        // Update migration status
        {
            let mut metadata =
                self.data
                    .metadata
                    .write()
                    .map_err(|_| BitNetTensorError::ConcurrentAccess {
                        reason: "Failed to acquire metadata write lock".to_string(),
                    })?;
            metadata.set_migrating(false);
        }

        #[cfg(feature = "tracing")]
        info!("Tensor migration completed");

        Ok(new_tensor)
    }

    /// Creates a handle for safe access to the tensor
    pub fn handle(&self) -> TensorHandle {
        let handle_id = {
            let mut registry = self.registry.write().unwrap();
            registry.next_handle_id()
        };

        TensorHandle::new(Arc::downgrade(&self.data), handle_id)
    }

    /// Gets the tensor ID
    pub fn id(&self) -> u64 {
        self.data.tensor_id
    }

    /// Gets the tensor shape
    pub fn shape(&self) -> Vec<usize> {
        let metadata = self.data.metadata.read().unwrap();
        metadata.shape.clone()
    }

    /// Gets the tensor data type
    pub fn dtype(&self) -> BitNetDType {
        let metadata = self.data.metadata.read().unwrap();
        metadata.dtype
    }

    /// Gets the device where the tensor is stored
    pub fn device(&self) -> Device {
        self.data.memory_handle.device()
    }

    /// Gets the size in bytes
    pub fn size_bytes(&self) -> usize {
        let metadata = self.data.metadata.read().unwrap();
        metadata.size_bytes
    }

    /// Gets the number of elements
    pub fn element_count(&self) -> usize {
        let metadata = self.data.metadata.read().unwrap();
        metadata.element_count
    }

    /// Gets the current reference count
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Gets the tensor name
    pub fn name(&self) -> Option<String> {
        let metadata = self.data.metadata.read().unwrap();
        metadata.name.clone()
    }

    /// Sets the tensor name
    pub fn set_name(&self, name: Option<String>) {
        let mut metadata = self.data.metadata.write().unwrap();
        metadata.name = name;
    }

    /// Reshapes the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> BitNetTensorResult<Self> {
        let current_elements = self.element_count();
        let new_elements: usize = new_shape.iter().product();

        if current_elements != new_elements {
            return Err(BitNetTensorError::ShapeMismatch {
                expected: vec![current_elements],
                actual: new_shape.to_vec(),
            });
        }

        // Create a new tensor with the same data but different shape
        let dtype = self.dtype();
        let device = self.device(); // Fixed - get device from self
        let pool = HybridMemoryPool::new()?; // TODO: This is a temporary workaround

        let new_tensor = Self::zeros(new_shape, dtype, &device, &pool)?;

        // Copy data from current tensor to new tensor (reshape doesn't change data layout)
        unsafe {
            let src_ptr = self.data.memory_handle.as_ptr();
            let dst_ptr = new_tensor.data.memory_handle.as_ptr() as *mut u8;
            let size_bytes = self.size_bytes();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_bytes);
        }

        Ok(new_tensor)
    }

    /// Creates a copy of the tensor
    pub fn clone_tensor(&self, pool: &HybridMemoryPool) -> BitNetTensorResult<Self> {
        let shape = self.shape();
        let dtype = self.dtype();
        let device = self.device(); // Fixed - get device from self
        let name = self.name();

        let new_tensor = Self::zeros(&shape, dtype, &device, pool)?;
        new_tensor.set_name(name);

        // Copy data from current tensor to new tensor
        unsafe {
            let src_ptr = self.data.memory_handle.as_ptr();
            let dst_ptr = new_tensor.data.memory_handle.as_ptr() as *mut u8;
            let size_bytes = self.size_bytes();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_bytes);
        }

        Ok(new_tensor)
    }

    /// Helper method to initialize memory with ones based on data type
    unsafe fn initialize_ones(ptr: *mut u8, element_count: usize, dtype: BitNetDType) {
        match dtype {
            BitNetDType::F32 => {
                let f32_ptr = ptr as *mut f32;
                for i in 0..element_count {
                    *f32_ptr.add(i) = 1.0;
                }
            }
            BitNetDType::F16 => {
                let f16_ptr = ptr as *mut u16; // f16 represented as u16
                let f16_one = 0x3C00u16; // IEEE 754 half precision 1.0
                for i in 0..element_count {
                    *f16_ptr.add(i) = f16_one;
                }
            }
            BitNetDType::BF16 => {
                let bf16_ptr = ptr as *mut u16; // bf16 represented as u16
                let bf16_one = 0x3F80u16; // BFloat16 1.0
                for i in 0..element_count {
                    *bf16_ptr.add(i) = bf16_one;
                }
            }
            BitNetDType::I8 => {
                let i8_ptr = ptr as *mut i8;
                for i in 0..element_count {
                    *i8_ptr.add(i) = 1i8;
                }
            }
            BitNetDType::I4 => {
                // I4 is packed, 2 elements per byte
                let byte_count = (element_count + 1) / 2;
                for i in 0..byte_count {
                    if i * 2 + 1 < element_count {
                        // Two I4 values: 0x11 (both nibbles = 1)
                        *ptr.add(i) = 0x11;
                    } else {
                        // Last byte with only one I4 value: 0x01
                        *ptr.add(i) = 0x01;
                    }
                }
            }
            BitNetDType::I2 => {
                // I2 is packed, 4 elements per byte
                let byte_count = (element_count + 3) / 4;
                for i in 0..byte_count {
                    // Fill with 0x55 (01010101 binary, each 2-bit value = 1)
                    *ptr.add(i) = 0x55;
                }
            }
            BitNetDType::I1 => {
                // I1 is packed, 8 elements per byte
                let byte_count = (element_count + 7) / 8;
                for i in 0..byte_count {
                    // Fill with 0xFF (all bits set to 1)
                    *ptr.add(i) = 0xFF;
                }
            }
            BitNetDType::BitNet158 => {
                // BitNet 1.58b uses 2 bits per element, values: -1, 0, 1
                // We'll use value 1 (represented as 0b01)
                let byte_count = (element_count + 3) / 4;
                for i in 0..byte_count {
                    // Fill with 0x55 (01010101 binary, each 2-bit value = 1)
                    *ptr.add(i) = 0x55;
                }
            }
        }
    }
}

impl Clone for BitNetTensor {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            registry: Arc::clone(&self.registry),
            next_id: Arc::clone(&self.next_id),
        }
    }
}

impl Drop for BitNetTensor {
    fn drop(&mut self) {
        // Only cleanup if this is the last reference
        if Arc::strong_count(&self.data) == 1 {
            let tensor_id = self.data.tensor_id;

            #[cfg(feature = "tracing")]
            debug!("Dropping BitNet tensor with ID {}", tensor_id);

            // Unregister from registry
            if let Ok(mut registry) = self.registry.write() {
                registry.unregister_tensor(tensor_id);
                registry.cleanup_dead_references();
            }

            // Force cleanup of the memory handle by triggering pool cleanup
            if let Some(pool) = crate::memory::tensor::handle::GLOBAL_MEMORY_POOL
                .lock()
                .ok()
                .and_then(|guard| guard.as_ref().and_then(|weak| weak.upgrade()))
            {
                let _ = pool.cleanup_orphaned_handles();
            }

            #[cfg(feature = "tracing")]
            info!("BitNet tensor {} dropped and cleaned up", tensor_id);
        }
    }
}

/// Clears all global tensor state - used for testing to prevent test interference
pub fn clear_global_tensor_state() {
    // Clear the global tensor registry
    if let Ok(mut registry) = GLOBAL_TENSOR_REGISTRY.write() {
        let count = registry.tensors.len();
        registry.tensors.clear();
        registry.next_handle_id = 0;
        #[cfg(feature = "tracing")]
        tracing::debug!("Cleared {} tensors from global tensor registry", count);
        #[cfg(not(feature = "tracing"))]
        let _ = count; // Suppress unused warning when tracing is disabled
    }

    // Reset the global tensor ID counter
    if let Ok(mut counter) = GLOBAL_TENSOR_ID_COUNTER.lock() {
        *counter = 1;
        #[cfg(feature = "tracing")]
        tracing::debug!("Reset global tensor ID counter to 1");
    }
}

impl std::fmt::Display for BitNetTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metadata = self.data.metadata.read().unwrap();
        write!(
            f,
            "BitNetTensor({}): {}",
            self.data.tensor_id,
            metadata.description()
        )
    }
}

// Thread safety
unsafe impl Send for BitNetTensor {}
unsafe impl Sync for BitNetTensor {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_tensor_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool).unwrap();

        assert_eq!(tensor.shape(), vec![2, 3]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
        assert_eq!(tensor.element_count(), 6);
        assert_eq!(tensor.ref_count(), 1);
    }

    #[test]
    fn test_tensor_from_data() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = BitNetTensor::from_data(data, &[2, 2], &device, &pool).unwrap();

        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.element_count(), 4);
    }

    #[test]
    fn test_tensor_handle() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let tensor = BitNetTensor::zeros(&[3, 3], BitNetDType::F32, &device, &pool).unwrap();
        let handle = tensor.handle();

        assert!(handle.is_valid());
        assert_eq!(handle.tensor_id().unwrap(), tensor.id());
        assert_eq!(handle.shape().unwrap(), tensor.shape());
    }

    #[test]
    fn test_tensor_cloning() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let tensor1 = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let tensor2 = tensor1.clone();

        assert_eq!(tensor1.id(), tensor2.id());
        assert_eq!(tensor1.ref_count(), 2);
        assert_eq!(tensor2.ref_count(), 2);
    }

    #[test]
    fn test_tensor_reshape() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();

        assert_eq!(reshaped.shape(), vec![3, 2]);
        assert_eq!(reshaped.element_count(), 6);
    }

    #[test]
    fn test_invalid_reshape() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool).unwrap();
        let result = tensor.reshape(&[2, 2]); // Different number of elements

        assert!(result.is_err());
    }
}
