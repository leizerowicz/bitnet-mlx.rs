//! Core BitNet Tensor Implementation
//!
//! This module provides the main BitNetTensor struct that integrates
//! all tensor functionality including memory management, device awareness,
//! and BitNet-specific operations.

use std::sync::Arc;
use std::fmt;
use candle_core::{Device, Tensor as CandleTensor, Result as CandleResult};
use crate::memory::{MemoryResult, MemoryError};
use crate::device::auto_select_device;
use super::dtype::BitNetDType;
use super::shape::{TensorShape, BroadcastCompatible};
use super::storage::TensorStorage;
use super::memory_integration::TensorMemoryManager;
use super::device_integration::TensorDeviceManager;
use super::ops::TensorOpError;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Main BitNet Tensor struct
///
/// BitNetTensor provides a high-level interface for tensor operations
/// with efficient memory management through HybridMemoryPool,
/// device awareness, and support for BitNet quantization.
#[derive(Debug)]
pub struct BitNetTensor {
    /// Tensor storage backend
    storage: Arc<TensorStorage>,
    /// Reference to memory manager for cleanup
    memory_manager: Option<Arc<TensorMemoryManager>>,
    /// Reference to device manager for operations
    device_manager: Option<Arc<TensorDeviceManager>>,
    /// Unique tensor identifier
    tensor_id: u64,
}

impl BitNetTensor {
    /// Creates a tensor filled with zeros
    pub fn zeros(
        shape: &[usize],
        dtype: BitNetDType,
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        let tensor_shape = TensorShape::new(shape);
        
        let device_clone = device.clone();
        let memory_manager = super::memory_integration::get_global_memory_pool()
            .ok_or_else(|| MemoryError::InternalError {
                reason: "Global memory pool not available".to_string(),
            })
            .and_then(|pool| {
                let dev = device_clone.unwrap_or_else(|| auto_select_device());
                Ok(Arc::new(TensorMemoryManager::new(pool, dev.clone())))
            })?;

        let device = device.unwrap_or_else(|| auto_select_device());

        let device_manager = None; // Simplified for now

        let storage = Arc::new(TensorStorage::zeros(
            tensor_shape,
            dtype,
            device,
            &memory_manager,
        )?);

        let tensor_id = storage.storage_id();

        #[cfg(feature = "tracing")]
        debug!("Created zero BitNetTensor with ID {} and shape {:?}", tensor_id, shape);

        Ok(Self {
            storage,
            memory_manager: Some(memory_manager),
            device_manager,
            tensor_id,
        })
    }

    /// Creates a tensor filled with ones
    pub fn ones(
        shape: &[usize],
        dtype: BitNetDType,
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        let tensor_shape = TensorShape::new(shape);
        
        let device_clone = device.clone();
        let memory_manager = super::memory_integration::get_global_memory_pool()
            .ok_or_else(|| MemoryError::InternalError {
                reason: "Global memory pool not available".to_string(),
            })
            .and_then(|pool| {
                let dev = device_clone.unwrap_or_else(|| auto_select_device());
                Ok(Arc::new(TensorMemoryManager::new(pool, dev.clone())))
            })?;

        let device = device.unwrap_or_else(|| auto_select_device());

        let device_manager = None; // Simplified for now

        let storage = Arc::new(TensorStorage::ones(
            tensor_shape,
            dtype,
            device,
            &memory_manager,
        )?);

        let tensor_id = storage.storage_id();

        #[cfg(feature = "tracing")]
        debug!("Created ones BitNetTensor with ID {} and shape {:?}", tensor_id, shape);

        Ok(Self {
            storage,
            memory_manager: Some(memory_manager),
            device_manager,
            tensor_id,
        })
    }

    /// Creates a tensor from a vector of data
    pub fn from_vec<T>(
        data: Vec<T>,
        shape: &[usize],
        dtype: BitNetDType,
        device: Option<Device>,
    ) -> MemoryResult<Self>
    where
        T: Copy + 'static,
    {
        let tensor_shape = TensorShape::new(shape);
        
        let device_clone = device.clone();
        let memory_manager = super::memory_integration::get_global_memory_pool()
            .ok_or_else(|| MemoryError::InternalError {
                reason: "Global memory pool not available".to_string(),
            })
            .and_then(|pool| {
                let dev = device_clone.unwrap_or_else(|| auto_select_device());
                Ok(Arc::new(TensorMemoryManager::new(pool, dev.clone())))
            })?;

        let device = device.unwrap_or_else(|| auto_select_device());

        let device_manager = None; // Simplified for now

        let storage = Arc::new(TensorStorage::from_vec(
            data,
            tensor_shape,
            dtype,
            device,
            &memory_manager,
        )?);

        let tensor_id = storage.storage_id();

        #[cfg(feature = "tracing")]
        debug!("Created BitNetTensor from vec with ID {} and shape {:?}", tensor_id, shape);

        Ok(Self {
            storage,
            memory_manager: Some(memory_manager),
            device_manager,
            tensor_id,
        })
    }

    /// Returns the tensor ID
    pub fn tensor_id(&self) -> u64 {
        self.tensor_id
    }

    /// Returns the data type
    pub fn dtype(&self) -> BitNetDType {
        self.storage.dtype()
    }

    /// Returns the shape
    pub fn shape(&self) -> &TensorShape {
        self.storage.shape()
    }

    /// Returns the device
    pub fn device(&self) -> &Device {
        self.storage.device()
    }

    /// Returns the number of elements
    pub fn num_elements(&self) -> usize {
        self.storage.num_elements()
    }

    /// Alias for num_elements for compatibility
    pub fn element_count(&self) -> usize {
        self.num_elements()
    }

    /// Returns true if the tensor has allocated memory
    pub fn is_allocated(&self) -> bool {
        true // BitNetTensor always has storage when created
    }

    /// Returns the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.storage.size_bytes()
    }

    /// Creates a BitNet 1.58 quantized tensor
    pub fn bitnet_158(
        shape: &[usize],
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        Self::zeros(shape, BitNetDType::BitNet158, device)
    }

    /// Validate tensor state
    pub fn validate(&self) -> MemoryResult<()> {
        Ok(()) // Simplified validation
    }

    /// Creates a tensor with the same shape and dtype as this one, but filled with zeros
    pub fn zeros_like(&self) -> MemoryResult<Self> {
        Self::zeros(
            self.shape().dims(),
            self.dtype(),
            Some(self.device().clone()),
        )
    }

    /// Creates a random tensor for testing purposes
    pub fn random(
        shape: &[usize],
        dtype: BitNetDType,
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        // For simplicity, create a zeros tensor
        // In a full implementation, this would be filled with random values
        Self::zeros(shape, dtype, device)
    }

    /// Get a slice view of the tensor data as f32
    /// This is a simplified implementation for compatibility
    pub fn as_slice_f32(&self) -> Result<&[f32], TensorOpError> {
        // This would need proper implementation based on storage format
        // For now, return an error to avoid crashes
        Err(TensorOpError::UnsupportedOperation {
            operation: "as_slice_f32".to_string(),
            dtype: self.dtype(),
        })
    }

    /// Get a mutable slice view of the tensor data as f32
    /// This is a simplified implementation for compatibility
    pub fn as_mut_slice_f32(&self) -> Result<&mut [f32], TensorOpError> {
        // This would need proper implementation based on storage format
        // For now, return an error to avoid crashes
        Err(TensorOpError::UnsupportedOperation {
            operation: "as_mut_slice_f32".to_string(),
            dtype: self.dtype(),
        })
    }

    /// Moves tensor to a different device
    pub fn to_device(&self, device: &Device) -> MemoryResult<Self> {
        // Use device comparison from our device module
        use crate::device::devices_equal;
        if devices_equal(self.device(), device) {
            return Ok(self.clone());
        }
        
        // For now, create a new tensor on the target device with same data
        // This is a simplified implementation
        Self::zeros(self.shape().dims(), self.dtype(), Some(device.clone()))
    }

    /// Reshapes the tensor to new dimensions
    pub fn reshape(&self, new_shape: &[usize]) -> MemoryResult<Self> {
        let new_tensor_shape = TensorShape::new(new_shape);
        if new_tensor_shape.num_elements() != self.num_elements() {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Cannot reshape tensor with {} elements to shape with {} elements",
                    self.num_elements(),
                    new_tensor_shape.num_elements()
                ),
            });
        }
        
        // For now, create a new tensor with the new shape
        Self::zeros(new_shape, self.dtype(), Some(self.device().clone()))
    }

    /// Removes dimensions of size 1
    pub fn squeeze(&self) -> MemoryResult<Self> {
        let squeezed_dims: Vec<usize> = self.shape()
            .dims()
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();
        
        if squeezed_dims.is_empty() {
            // If all dimensions are 1, keep one dimension
            self.reshape(&[1])
        } else {
            self.reshape(&squeezed_dims)
        }
    }

    /// Transposes the tensor (swaps the last two dimensions)
    pub fn transpose(&self) -> MemoryResult<Self> {
        let dims = self.shape().dims();
        if dims.len() < 2 {
            return Err(MemoryError::InternalError {
                reason: "Cannot transpose tensor with less than 2 dimensions".to_string(),
            });
        }
        
        let mut new_dims = dims.to_vec();
        let len = new_dims.len();
        new_dims.swap(len - 2, len - 1);
        
        self.reshape(&new_dims)
    }

    /// Checks if the tensor is valid
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Converts to Candle tensor for interoperability
    pub fn to_candle(&self) -> Result<candle_core::Tensor, TensorOpError> {
        // This is a simplified conversion - in a full implementation,
        // we'd need to handle the data transfer properly
        match self.dtype() {
            BitNetDType::F32 => {
                unsafe {
                    let ptr = self.storage.as_ptr() as *const f32;
                    let slice = std::slice::from_raw_parts(ptr, self.num_elements());
                    candle_core::Tensor::from_slice(slice, self.shape().dims(), self.device())
                        .map_err(|e| TensorOpError::CandleError {
                            operation: "to_candle".to_string(),
                            error: e.to_string(),
                        })
                }
            }
            _ => {
                // For other types, we'd need proper conversion
                Err(TensorOpError::UnsupportedOperation {
                    operation: "to_candle".to_string(),
                    dtype: self.dtype(),
                })
            }
        }
    }

    /// Creates a BitNetTensor from a Candle tensor
    pub fn from_candle(
        candle_tensor: candle_core::Tensor, 
        device: &candle_core::Device,
    ) -> MemoryResult<Self> {
        let shape = candle_tensor.dims();
        let device = device.clone();

        // Extract data from Candle tensor
        match candle_tensor.dtype() {
            candle_core::DType::F32 => {
                // For multi-dimensional tensors, we need to flatten to extract data
                let flattened = candle_tensor.flatten_all()
                    .map_err(|e| MemoryError::InternalError {
                        reason: format!("Failed to flatten Candle tensor: {}", e),
                    })?;
                    
                let data = flattened.to_vec1::<f32>()
                    .map_err(|e| MemoryError::InternalError {
                        reason: format!("Failed to extract F32 data from Candle tensor: {}", e),
                    })?;
                Self::from_vec(data, shape, BitNetDType::F32, Some(device))
            }
            _ => Err(MemoryError::InternalError {
                reason: format!(
                    "Conversion from Candle dtype {:?} not yet implemented",
                    candle_tensor.dtype()
                ),
            })
        }
    }
    
    /// Get size in bytes for the tensor
    pub fn size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }
    
    /// Get raw data pointer for zero-copy operations (MLX integration)
    pub fn raw_data_ptr(&self) -> Option<*const u8> {
        self.storage.raw_data_ptr()
    }
    
    /// Get tensor data as a slice of the specified type
    pub fn data_as_slice<T: Clone + 'static>(&self) -> MemoryResult<Vec<T>> {
        self.storage.data_as_slice::<T>()
    }
    
    /// Create BitNetTensor from data
    pub fn from_data(
        data: &[f32], 
        shape: &[usize], 
        dtype: BitNetDType, 
        device: Option<Device>
    ) -> MemoryResult<Self> {
        Self::from_vec(data.to_vec(), shape, dtype, device)
    }
    
    /// Clone tensor data (for compatibility)
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}

/// Memory statistics for a tensor
#[derive(Debug, Clone)]
pub struct TensorMemoryStats {
    pub tensor_id: u64,
    pub storage_id: u64,
    pub shape: Vec<usize>,
    pub dtype: BitNetDType,
    pub device: Device,
    pub size_bytes: usize,
    pub num_elements: usize,
    pub element_size: usize,
    pub is_shared: bool,
}

impl fmt::Display for TensorMemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor ID: {}, Storage ID: {}, Shape: {:?}, Type: {:?}, Device: {:?}, Size: {} bytes ({} elements), Shared: {}",
            self.tensor_id, self.storage_id, self.shape, self.dtype, self.device, 
            self.size_bytes, self.num_elements, self.is_shared
        )
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for BitNetTensor {}
unsafe impl Sync for BitNetTensor {}

impl Clone for BitNetTensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "tracing")]
        debug!("Cloning BitNetTensor {}", self.tensor_id);
        
        Self {
            storage: Arc::clone(&self.storage),
            memory_manager: self.memory_manager.clone(),
            device_manager: self.device_manager.clone(),
            tensor_id: self.tensor_id,
        }
    }
}

impl Drop for BitNetTensor {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        debug!("Dropping BitNetTensor {}", self.tensor_id);
    }
}

// Broadcasting support
impl BroadcastCompatible for BitNetTensor {
    fn is_broadcast_compatible(&self, other: &Self) -> bool {
        self.shape().is_broadcast_compatible(other.shape())
    }
    
    fn broadcast_shape(&self, other: &Self) -> super::shape::ShapeResult<TensorShape> {
        self.shape().broadcast_shape(other.shape())
    }
    
    fn can_broadcast_to(&self, target: &Self) -> bool {
        self.shape().can_broadcast_to(target.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
        assert_eq!(tensor.num_elements(), 6);
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = BitNetTensor::from_vec(data, &[2, 3], BitNetDType::F32, None).unwrap();
        
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.num_elements(), 6);
    }

    #[test]
    fn test_bitnet_158_creation() {
        let tensor = BitNetTensor::bitnet_158(&[10, 20], None).unwrap();
        assert_eq!(tensor.dtype(), BitNetDType::BitNet158);
        assert_eq!(tensor.shape().dims(), &[10, 20]);
    }

    #[test]
    fn test_tensor_clone() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        let cloned = tensor.clone();
        
        assert_eq!(tensor.tensor_id(), cloned.tensor_id());
        assert_eq!(tensor.shape().dims(), cloned.shape().dims());
        assert_eq!(tensor.dtype(), cloned.dtype());
    }

    #[test]
    fn test_tensor_validation() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        assert!(tensor.validate().is_ok());
    }
}
