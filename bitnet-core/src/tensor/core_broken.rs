//! Core BitNet Tensor Implementation
//!
//! This module provides the main BitNetTensor struct that integrates
//! all tensor        Ok(Self {
            storage,
            memory_manager: Some(memory_manager),
            device_manager,
            tensor_id,
        })
    }

    /// Creates a tensor filled with onesality including memory management, device awareness,
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
///
/// # Features
///
/// - Automatic memory pool integration
/// - Device-aware operations (CPU/Metal GPU)
/// - BitNet quantization support
/// - Broadcasting and shape manipulation
/// - Thread-safe operations
/// - Automatic cleanup and lifecycle management
///
/// # Examples
///
/// ```rust
/// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a tensor filled with zeros
/// let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32)?;
///
/// // Create from data
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let tensor = BitNetTensor::from_vec(data, &[2, 2], BitNetDType::F32)?;
///
/// // Reshape
/// let reshaped = tensor.reshape(&[4, 1])?;
/// # Ok(())
/// # }
/// ```
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
    /// Creates a new tensor with uninitialized data
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape dimensions
    /// * `dtype` - Data type
    /// * `device` - Optional device (auto-selected if None)
    ///
    /// # Returns
    ///
    /// Result containing new BitNetTensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = BitNetTensor::new(&[2, 3], BitNetDType::F32, None)?;
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// assert_eq!(tensor.dtype(), BitNetDType::F32);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        shape: &[usize],
        dtype: BitNetDType,
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        let tensor_shape = TensorShape::new(shape);

        // Get or create memory manager
        let memory_manager = super::memory_integration::get_global_memory_pool()
            .ok_or_else(|| MemoryError::InternalError {
                reason: "Global memory pool not available".to_string(),
            })
            .and_then(|pool| {
                let device = device.unwrap_or_else(|| auto_select_device());
                Ok(Arc::new(TensorMemoryManager::new(pool, device.clone())))
            })?;

        // Get device
        let device = device.unwrap_or_else(|| auto_select_device());

        // Get device manager - simplified for now
        let device_manager = None;

        // Create storage
        let storage = Arc::new(TensorStorage::new(
            tensor_shape,
            dtype,
            device,
            &memory_manager,
        )?);

        let tensor_id = storage.storage_id();

        #[cfg(feature = "tracing")]
        info!("Created BitNetTensor with ID {} and shape {:?}", tensor_id, shape);

        Ok(Self {
            storage,
            memory_manager: Some(memory_manager),
            device_manager: Some(device_manager),
            tensor_id,
        })
    }

    /// Creates a tensor filled with zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape dimensions
    /// * `dtype` - Data type
    /// * `device` - Optional device (auto-selected if None)
    ///
    /// # Returns
    ///
    /// Result containing new BitNetTensor filled with zeros
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32)?;
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of typed data
    /// * `shape` - Shape dimensions
    /// * `dtype` - Data type (must match T)
    /// * `device` - Optional device (auto-selected if None)
    ///
    /// # Returns
    ///
    /// Result containing new BitNetTensor with the data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = BitNetTensor::from_vec(data, &[2, 3], BitNetDType::F32, None)?;
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// assert_eq!(tensor.num_elements(), 6);
    /// # Ok(())
    /// # }
    /// ```
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
            device_manager: Some(device_manager),
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

    /// Returns the element size in bytes
    pub fn element_size(&self) -> usize {
        self.storage.element_size()
    }

    /// Returns the storage ID
    pub fn storage_id(&self) -> u64 {
        self.storage.storage_id()
    }

    /// Reshapes the tensor (changes shape but keeps same data)
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape dimensions
    ///
    /// # Returns
    ///
    /// Result containing new BitNetTensor with the new shape
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32)?;
    /// let reshaped = tensor.reshape(&[3, 2])?;
    /// assert_eq!(reshaped.shape().dims(), &[3, 2]);
    /// assert_eq!(reshaped.num_elements(), 6); // Same number of elements
    /// # Ok(())
    /// # }
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> MemoryResult<Self> {
        let new_tensor_shape = TensorShape::new(new_shape);

        if new_tensor_shape.num_elements() != self.num_elements() {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Cannot reshape: element count mismatch ({} != {})",
                    new_tensor_shape.num_elements(),
                    self.num_elements()
                ),
            });
        }

        // Create a new tensor sharing the same storage but with different shape
        let new_storage = Arc::new(TensorStorage::from_data(
            unsafe { self.storage.as_slice() },
            new_tensor_shape,
            self.dtype(),
            self.device().clone(),
            self.memory_manager.as_ref().ok_or_else(|| MemoryError::InternalError {
                reason: "Memory manager not available".to_string(),
            })?,
        )?);

        #[cfg(feature = "tracing")]
        debug!("Reshaped tensor {} from {:?} to {:?}",
               self.tensor_id, self.shape().dims(), new_shape);

        Ok(Self {
            storage: new_storage,
            memory_manager: self.memory_manager.clone(),
            device_manager: self.device_manager.clone(),
            tensor_id: self.tensor_id, // Keep same ID for reshaped tensor
        })
    }

    /// Fills the tensor with a specific value
    ///
    /// # Arguments
    ///
    /// * `value` - Value to fill with
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetTensor, BitNetDType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32)?;
    /// tensor.fill(5.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn fill(&mut self, value: f64) -> MemoryResult<()> {
        // We need mutable access to storage, but it's Arc<TensorStorage>
        // For now, we'll create a new storage with the filled value
        // This is not the most efficient but maintains safety
        let new_storage = if Arc::strong_count(&self.storage) == 1 {
            // We have exclusive access, can modify in place
            Arc::get_mut(&mut self.storage)
                .ok_or_else(|| MemoryError::InternalError {
                    reason: "Failed to get mutable reference to storage".to_string(),
                })?
                .fill_with_value(value)?;
            return Ok(());
        } else {
            // Create new storage with filled value
            let mut new_storage = TensorStorage::new(
                self.shape().clone(),
                self.dtype(),
                self.device().clone(),
                self.memory_manager.as_ref().ok_or_else(|| MemoryError::InternalError {
                    reason: "Memory manager not available".to_string(),
                })?,
            )?;
            new_storage.fill_with_value(value)?;
            Arc::new(new_storage)
        };

        self.storage = new_storage;

        #[cfg(feature = "tracing")]
        debug!("Filled tensor {} with value {}", self.tensor_id, value);

        Ok(())
    }

    /// Validates the tensor integrity
    pub fn validate(&self) -> MemoryResult<()> {
        self.storage.validate()
    }

    /// Returns raw data pointer (unsafe)
    ///
    /// # Safety
    ///
    /// Caller must ensure the memory is accessed safely and within bounds
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr() as *const u8
    }

    /// Returns raw mutable data pointer (unsafe)
    ///
    /// # Safety
    ///
    /// Caller must ensure the memory is accessed safely and exclusively
    pub unsafe fn as_mut_ptr(&mut self) -> MemoryResult<*mut u8> {
        if Arc::strong_count(&self.storage) != 1 {
            return Err(MemoryError::InternalError {
                reason: "Cannot get mutable pointer to shared storage".to_string(),
            });
        }
        Ok(self.storage.as_ptr())
    }

    /// Converts to Candle tensor for interoperability
    ///
    /// # Returns
    ///
    /// Result containing Candle tensor
    pub fn to_candle(&self) -> CandleResult<CandleTensor> {
        // This is a simplified conversion - in a full implementation,
        // we'd need to handle the data transfer properly
        match self.dtype() {
            BitNetDType::F32 => {
                unsafe {
                    let ptr = self.storage.as_ptr() as *const f32;
                    let slice = std::slice::from_raw_parts(ptr, self.num_elements());
                    CandleTensor::from_slice(slice, self.shape().dims(), self.device())
                }
            }
            _ => {
                // For other types, we'd need proper conversion
                Err(candle_core::Error::UnsupportedDTypeForOp(
                    candle_core::DType::F32, // placeholder
                    "BitNet tensor conversion"
                ))
            }
        }
    }

    /// Creates a BitNetTensor from a Candle tensor
    ///
    /// # Arguments
    ///
    /// * `candle_tensor` - Source Candle tensor
    /// * `targetdtype` - Target BitNet data type
    ///
    /// # Returns
    ///
    /// Result containing new BitNetTensor
    pub fn from_candle(
        candle_tensor: &CandleTensor,
        targetdtype: BitNetDType,
    ) -> MemoryResult<Self> {
        let shape = candle_tensor.dims();
        let device = .clone();

        // Extract data from Candle tensor
        match candle_tensor.dtype() {
            candle_core::DType::F32 => {
                let data = candle_tensor.to_vec1::<f32>()
                    .map_err(|e| MemoryError::InternalError {
                        reason: format!("Failed to extract F32 data from Candle tensor: {}", e),
                    })?;
                Self::from_vec(data, shape, targetdtype, Some(device))
            }
            _ => Err(MemoryError::InternalError {
                reason: format!(
                    "Conversion from Candle dtype {:?} not yet implemented",
                    candle_tensor.dtype()
                ),
            })
        }
    }

    /// Creates a tensor for BitNet quantization
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape dimensions
    /// * `device` - Optional device (auto-selected if None)
    ///
    /// # Returns
    ///
    /// Result containing new BitNet 1.58 quantized tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetTensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = BitNetTensor::bitnet_158(&[10, 20], None)?;
    /// assert_eq!(tensor.dtype(), bitnet_core::tensor::BitNetDType::BitNet158);
    /// # Ok(())
    /// # }
    /// ```
    pub fn bitnet_158(
        shape: &[usize],
        device: Option<Device>,
    ) -> MemoryResult<Self> {
        Self::zeros(shape, BitNetDType::BitNet158, device)
    }

    /// Gets memory usage statistics for this tensor
    pub fn memory_stats(&self) -> MemoryResult<TensorMemoryStats> {
        Ok(TensorMemoryStats {
            tensor_id: self.tensor_id,
            storage_id: self.storage.storage_id(),
            shape: self.shape().dims().to_vec(),
            dtype: self.dtype(),
            device: self.device().clone(),
            size_bytes: self.size_bytes(),
            num_elements: self.num_elements(),
            element_size: self.element_size(),
            is_shared: Arc::strong_count(&self.storage) > 1,
        })
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

        // Arc will handle the actual storage cleanup
        // Additional cleanup can be added here if needed
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
        assert_eq!(tensor.size_bytes(), 24); // 6 * 4 bytes
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = BitNetTensor::from_vec(data, &[2, 3], BitNetDType::F32, None).unwrap();

        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.num_elements(), 6);
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();

        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.num_elements(), 6);
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None).unwrap();
        assert_eq!(tensor.num_elements(), 4);
    }

    #[test]
    fn test_bitnet_158_creation() {
        let tensor = BitNetTensor::bitnet_158(&[10, 20], None).unwrap();
        assert_eq!(tensor.dtype(), BitNetDType::BitNet158);
        assert_eq!(tensor.shape().dims(), &[10, 20]);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, None).unwrap();
        tensor.fill(5.0).unwrap();
        // Fill operation should succeed
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

    #[test]
    fn test_tensor_memory_stats() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        let stats = tensor.memory_stats().unwrap();

        assert_eq!(stats.shape, vec![2, 3]);
        assert_eq!(stats.dtype, BitNetDType::F32);
        assert_eq!(stats.num_elements, 6);
        assert_eq!(stats.size_bytes, 24);
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let tensor1 = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        let tensor2 = BitNetTensor::zeros(&[1, 3], BitNetDType::F32, None).unwrap();
        let tensor3 = BitNetTensor::zeros(&[2, 4], BitNetDType::F32, None).unwrap();

        assert!(tensor1.can_broadcast_with(&tensor2));
        assert!(!tensor1.can_broadcast_with(&tensor3));
    }

    #[test]
    fn test_reshape_error_handling() {
        let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None).unwrap();
        let result = tensor.reshape(&[2, 4]); // Different number of elements
        assert!(result.is_err());
    }
}
