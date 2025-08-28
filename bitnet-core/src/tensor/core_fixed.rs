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

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Main BitNet Tensor struct
///
/// BitNetTensor provides a high-level interface for tensor operations
/// with efficient memory management through HybridMemoryPool,
/// device awareness, and support for BitNet quantization.
#[derive(Debug)]
#[allow(dead_code)]
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

    /// Validates the tensor integrity
    pub fn validate(&self) -> MemoryResult<()> {
        self.storage.validate()
    }
}

/// Memory statistics for a tensor
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
