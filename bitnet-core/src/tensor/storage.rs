//! Tensor Storage Backend
//!
//! This module provides the storage backend for BitNet tensors,
//! leveraging the HybridMemoryPool for efficient memory management
//! and supporting different memory layouts and access patterns.

use std::sync::{Arc, Weak};
use std::slice;
use candle_core::Device;
use crate::memory::{MemoryHandle, MemoryResult, MemoryError};
use super::dtype::BitNetDType;
use super::shape::TensorShape;
use super::memory_integration::TensorMemoryManager;

#[cfg(feature = "tracing")]
use tracing::{debug, warn, error};

/// Tensor storage backend that uses HybridMemoryPool
///
/// TensorStorage provides the underlying data storage for BitNet tensors,
/// with efficient memory management, device awareness, and support for
/// different data layouts.
#[derive(Debug)]
pub struct TensorStorage {
    /// Unique identifier for this storage
    storage_id: u64,
    /// Memory handle from HybridMemoryPool
    memory_handle: MemoryHandle,
    /// Data type of stored elements
    dtype: BitNetDType,
    /// Shape of the tensor
    shape: TensorShape,
    /// Number of bytes per element
    element_size: usize,
    /// Total size in bytes
    total_size: usize,
    /// Device where data is stored
    device: Device,
    /// Weak reference to memory manager for cleanup
    memory_manager: Option<Weak<TensorMemoryManager>>,
}

impl TensorStorage {
    /// Creates new tensor storage
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the tensor
    /// * `dtype` - Data type of elements
    /// * `device` - Device for storage
    /// * `memory_manager` - Memory manager for allocation
    ///
    /// # Returns
    ///
    /// Result containing TensorStorage or error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{TensorStorage, TensorShape, BitNetDType, TensorMemoryManager};
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// let device = get_cpu_device();
    /// let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
    /// let shape = TensorShape::new(&[2, 3]);
    /// 
    /// let storage = TensorStorage::new(shape, BitNetDType::F32, device, &manager)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        shape: TensorShape,
        dtype: BitNetDType,
        device: Device,
        memory_manager: &Arc<TensorMemoryManager>,
    ) -> MemoryResult<Self> {
        let element_size = dtype.size_bytes()
            .ok_or_else(|| MemoryError::InternalError {
                reason: format!("Cannot determine size for data type {:?}", dtype),
            })?;

        let num_elements = shape.num_elements();
        let total_size = num_elements * element_size;
        
        // Calculate appropriate alignment based on data type and SIMD requirements
        let alignment = Self::calculate_alignment(dtype);

        #[cfg(feature = "tracing")]
        debug!(
            "Creating tensor storage: shape={:?}, dtype={:?}, elements={}, total_size={} bytes, alignment={}",
            shape.dims(), dtype, num_elements, total_size, alignment
        );

        // Allocate memory through the memory manager
        let (storage_id, memory_handle) = memory_manager.allocate_tensor_memory(
            total_size,
            alignment,
            dtype,
        )?;

        // Initialize memory to zero if requested
        unsafe {
            let ptr = memory_handle.as_ptr();
            std::ptr::write_bytes(ptr, 0, total_size);
        }

        #[cfg(feature = "tracing")]
        debug!("Successfully created tensor storage with ID {}", storage_id);

        Ok(Self {
            storage_id,
            memory_handle,
            dtype,
            shape,
            element_size,
            total_size,
            device,
            memory_manager: Some(Arc::downgrade(memory_manager)),
        })
    }

    /// Creates tensor storage from existing data
    ///
    /// # Arguments
    ///
    /// * `data` - Raw data bytes
    /// * `shape` - Shape of the tensor
    /// * `dtype` - Data type of elements
    /// * `device` - Device for storage
    /// * `memory_manager` - Memory manager for allocation
    ///
    /// # Returns
    ///
    /// Result containing TensorStorage with copied data
    pub fn from_data(
        data: &[u8],
        shape: TensorShape,
        dtype: BitNetDType,
        device: Device,
        memory_manager: &Arc<TensorMemoryManager>,
    ) -> MemoryResult<Self> {
        let mut storage = Self::new(shape, dtype, device, memory_manager)?;
        
        if data.len() != storage.total_size {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Data size {} bytes doesn't match expected size {} bytes",
                    data.len(),
                    storage.total_size
                ),
            });
        }

        // Copy data into allocated memory
        unsafe {
            let ptr = storage.memory_handle.as_ptr();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        #[cfg(feature = "tracing")]
        debug!("Created tensor storage from data: {} bytes copied", data.len());

        Ok(storage)
    }

    /// Creates tensor storage filled with zeros
    pub fn zeros(
        shape: TensorShape,
        dtype: BitNetDType,
        device: Device,
        memory_manager: &Arc<TensorMemoryManager>,
    ) -> MemoryResult<Self> {
        // Memory is already zeroed in new(), so just create normally
        Self::new(shape, dtype, device, memory_manager)
    }

    /// Creates tensor storage filled with ones
    pub fn ones(
        shape: TensorShape,
        dtype: BitNetDType,
        device: Device,
        memory_manager: &Arc<TensorMemoryManager>,
    ) -> MemoryResult<Self> {
        let mut storage = Self::new(shape, dtype, device, memory_manager)?;
        storage.fill_with_value(1.0)?;
        Ok(storage)
    }

    /// Returns the storage ID
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    /// Returns the data type
    pub fn dtype(&self) -> BitNetDType {
        self.dtype
    }

    /// Returns the shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Returns the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.total_size
    }

    /// Returns the number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Returns the element size in bytes
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Returns raw data pointer
    ///
    /// # Safety
    ///
    /// Caller must ensure the memory is accessed safely and within bounds
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.memory_handle.as_ptr()
    }

    /// Returns raw data as a slice
    ///
    /// # Safety
    ///
    /// Caller must ensure the memory is accessed safely
    pub unsafe fn as_slice(&self) -> &[u8] {
        slice::from_raw_parts(self.memory_handle.as_ptr(), self.total_size)
    }

    /// Returns raw data as a mutable slice
    ///
    /// # Safety
    ///
    /// Caller must ensure the memory is accessed safely and exclusively
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.memory_handle.as_ptr(), self.total_size)
    }

    /// Fills storage with a specific value
    ///
    /// # Arguments
    ///
    /// * `value` - Value to fill with (will be converted to appropriate type)
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn fill_with_value(&mut self, value: f64) -> MemoryResult<()> {
        unsafe {
            let ptr = self.memory_handle.as_ptr();
            
            match self.dtype {
                BitNetDType::F32 => {
                    let typed_ptr = ptr as *mut f32;
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    typed_slice.fill(value as f32);
                }
                BitNetDType::F16 => {
                    // For F16, we need to convert to the appropriate representation
                    // This is a simplified implementation
                    let typed_ptr = ptr as *mut u16; // F16 as u16 bits
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    let f16_value = half::f16::from_f64(value).to_bits();
                    typed_slice.fill(f16_value);
                }
                BitNetDType::I8 => {
                    let typed_ptr = ptr as *mut i8;
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    typed_slice.fill(value as i8);
                }
                BitNetDType::I32 => {
                    let typed_ptr = ptr as *mut i32;
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    typed_slice.fill(value as i32);
                }
                BitNetDType::U8 => {
                    let typed_ptr = ptr as *mut u8;
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    typed_slice.fill(value as u8);
                }
                BitNetDType::BitNet158 => {
                    // For BitNet 1.58, convert to ternary values (-1, 0, 1)
                    let ternary_value = if value > 0.5 {
                        1i8
                    } else if value < -0.5 {
                        -1i8
                    } else {
                        0i8
                    };
                    let typed_ptr = ptr as *mut i8;
                    let typed_slice = slice::from_raw_parts_mut(typed_ptr, self.num_elements());
                    typed_slice.fill(ternary_value);
                }
                _ => {
                    // For other types, fill as bytes
                    let value_byte = (value.clamp(0.0, 255.0)) as u8;
                    std::ptr::write_bytes(ptr, value_byte, self.total_size);
                }
            }
        }

        Ok(())
    }

    /// Reshapes the storage (changes shape but keeps same data)
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape for the tensor
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn reshape(&mut self, new_shape: TensorShape) -> MemoryResult<()> {
        if new_shape.num_elements() != self.shape.num_elements() {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Cannot reshape: element count mismatch ({} != {})",
                    new_shape.num_elements(),
                    self.shape.num_elements()
                ),
            });
        }

        self.shape = new_shape;

        #[cfg(feature = "tracing")]
        debug!("Reshaped tensor storage {} to {:?}", self.storage_id, self.shape.dims());

        Ok(())
    }

    /// Returns the size in bytes (alias for size_bytes)
    pub fn size_in_bytes(&self) -> usize {
        self.total_size
    }
    
    /// Get raw data pointer for zero-copy operations
    pub fn raw_data_ptr(&self) -> Option<*const u8> {
        unsafe {
            Some(self.memory_handle.as_ptr() as *const u8)
        }
    }
    
    /// Get tensor data as a slice of the specified type
    pub fn data_as_slice<T: Clone + 'static>(&self) -> MemoryResult<Vec<T>> {
        // This is a simplified implementation - in practice this would need
        // proper type checking and conversion based on the stored dtype
        unsafe {
            let byte_slice = self.as_slice();
            // For now, just handle f32 type
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() 
                && self.dtype == BitNetDType::F32 {
                let float_ptr = byte_slice.as_ptr() as *const f32;
                let float_slice = slice::from_raw_parts(float_ptr, self.num_elements());
                let vec: Vec<f32> = float_slice.to_vec();
                // Safety: We know T is f32 from the type check above
                Ok(std::mem::transmute::<Vec<f32>, Vec<T>>(vec))
            } else {
                Err(MemoryError::InternalError {
                    reason: format!("Type conversion from {:?} to requested type not implemented", self.dtype),
                })
            }
        }
    }

    /// Validates storage integrity
    pub fn validate(&self) -> MemoryResult<()> {
        // Validate memory handle
        self.memory_handle.validate()?;

        // Validate size consistency
        let expected_size = self.shape.num_elements() * self.element_size;
        if self.total_size != expected_size {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Size mismatch: stored {} bytes, expected {} bytes",
                    self.total_size, expected_size
                ),
            });
        }

        // Validate device consistency (simplified comparison)
        let device_matches = match (&self.memory_handle.device(), &self.device) {
            (candle_core::Device::Cpu, candle_core::Device::Cpu) => true,
            (candle_core::Device::Metal(_), candle_core::Device::Metal(_)) => true, // Simplified comparison
            _ => false,
        };
        
        if !device_matches {
            return Err(MemoryError::InternalError {
                reason: "Device mismatch between storage and memory handle".to_string(),
            });
        }

        Ok(())
    }

    /// Checks if the storage is valid
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Calculates appropriate memory alignment for a data type
    fn calculate_alignment(dtype: BitNetDType) -> usize {
        match dtype {
            // SIMD alignment for common types
            BitNetDType::F32 | BitNetDType::I32 | BitNetDType::U32 => 16,
            BitNetDType::F16 | BitNetDType::BF16 | BitNetDType::I16 | BitNetDType::U16 => 8,
            BitNetDType::I64 | BitNetDType::U64 => 32, // Cache line alignment for large types
            // Smaller types
            BitNetDType::I8 | BitNetDType::U8 | BitNetDType::Bool => 4,
            // BitNet quantized types - align to byte boundaries but allow SIMD
            BitNetDType::BitNet158 | BitNetDType::BitNet11 | BitNetDType::BitNet1 | 
            BitNetDType::Int4 | BitNetDType::QInt8 | BitNetDType::QInt4 => 8,
        }
    }
}

impl Drop for TensorStorage {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        debug!("Dropping tensor storage {}", self.storage_id);

        // Try to clean up through memory manager if available
        if let Some(manager_weak) = &self.memory_manager {
            if let Some(manager) = manager_weak.upgrade() {
                if let Err(e) = manager.deallocate_tensor_memory(self.storage_id) {
                    #[cfg(feature = "tracing")]
                    warn!("Failed to deallocate tensor storage {}: {}", self.storage_id, e);
                }
            } else {
                #[cfg(feature = "tracing")]
                warn!("Memory manager no longer available for tensor storage {}", self.storage_id);
            }
        }
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for TensorStorage {}
unsafe impl Sync for TensorStorage {}

/// Storage creation utilities
impl TensorStorage {
    /// Creates storage from a vector of typed data
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of typed data
    /// * `shape` - Shape of the tensor
    /// * `dtype` - Data type (must match T)
    /// * `device` - Device for storage
    /// * `memory_manager` - Memory manager for allocation
    ///
    /// # Returns
    ///
    /// Result containing TensorStorage with the data
    pub fn from_vec<T>(
        data: Vec<T>,
        shape: TensorShape,
        dtype: BitNetDType,
        device: Device,
        memory_manager: &Arc<TensorMemoryManager>,
    ) -> MemoryResult<Self>
    where
        T: Copy + 'static,
    {
        // Validate that T matches the expected size for dtype
        let expected_size = dtype.size_bytes()
            .ok_or_else(|| MemoryError::InternalError {
                reason: format!("Cannot determine size for data type {:?}", dtype),
            })?;

        if std::mem::size_of::<T>() != expected_size {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Type size mismatch: {} bytes expected for {:?}, got {} bytes",
                    expected_size,
                    dtype,
                    std::mem::size_of::<T>()
                ),
            });
        }

        if data.len() != shape.num_elements() {
            return Err(MemoryError::InternalError {
                reason: format!(
                    "Data length {} doesn't match shape elements {}",
                    data.len(),
                    shape.num_elements()
                ),
            });
        }

        // Convert to bytes
        let data_bytes = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
        };

        Self::from_data(data_bytes, shape, dtype, device, memory_manager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;
    use crate::device::get_cpu_device;
    use std::sync::Arc;

    #[test]
    fn test_tensor_storage_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2, 3]);
        
        let storage = TensorStorage::new(shape, BitNetDType::F32, device, &manager).unwrap();
        
        assert_eq!(storage.dtype(), BitNetDType::F32);
        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.num_elements(), 6);
        assert_eq!(storage.size_bytes(), 24); // 6 * 4 bytes
        assert_eq!(storage.element_size(), 4);
    }

    #[test]
    fn test_zeros_and_ones() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2, 2]);
        
        let zeros_storage = TensorStorage::zeros(shape.clone(), BitNetDType::F32, device.clone(), &manager).unwrap();
        assert_eq!(zeros_storage.num_elements(), 4);
        
        let ones_storage = TensorStorage::ones(shape, BitNetDType::F32, device, &manager).unwrap();
        assert_eq!(ones_storage.num_elements(), 4);
    }

    #[test]
    fn test_from_data() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2]);
        
        let data: Vec<f32> = vec![1.0, 2.0];
        let data_bytes = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        
        let storage = TensorStorage::from_data(data_bytes, shape, BitNetDType::F32, device, &manager).unwrap();
        assert_eq!(storage.size_bytes(), 8); // 2 * 4 bytes
    }

    #[test]
    fn test_from_vec() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[3]);
        
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let storage = TensorStorage::from_vec(data, shape, BitNetDType::F32, device, &manager).unwrap();
        
        assert_eq!(storage.num_elements(), 3);
        assert_eq!(storage.size_bytes(), 12);
    }

    #[test]
    fn test_reshape() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2, 3]);
        
        let mut storage = TensorStorage::new(shape, BitNetDType::F32, device, &manager).unwrap();
        
        let new_shape = TensorShape::new(&[3, 2]);
        storage.reshape(new_shape).unwrap();
        
        assert_eq!(storage.shape().dims(), &[3, 2]);
        assert_eq!(storage.num_elements(), 6); // Same number of elements
    }

    #[test]
    fn test_fill_with_value() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2, 2]);
        
        let mut storage = TensorStorage::new(shape, BitNetDType::F32, device, &manager).unwrap();
        storage.fill_with_value(5.0).unwrap();
        
        // Verify the fill worked (unsafe access for testing)
        unsafe {
            let ptr = storage.as_ptr() as *const f32;
            let slice = slice::from_raw_parts(ptr, storage.num_elements());
            for &value in slice {
                assert_eq!(value, 5.0);
            }
        }
    }

    #[test]
    fn test_bitnet_fill() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[4]);
        
        let mut storage = TensorStorage::new(shape, BitNetDType::BitNet158, device, &manager).unwrap();
        storage.fill_with_value(1.0).unwrap(); // Should become 1
        
        unsafe {
            let ptr = storage.as_ptr() as *const i8;
            let slice = slice::from_raw_parts(ptr, storage.num_elements());
            for &value in slice {
                assert_eq!(value, 1i8);
            }
        }
    }

    #[test]
    fn test_storage_validation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = Arc::new(TensorMemoryManager::new(pool, device.clone()));
        let shape = TensorShape::new(&[2, 3]);
        
        let storage = TensorStorage::new(shape, BitNetDType::F32, device, &manager).unwrap();
        assert!(storage.validate().is_ok());
    }

    #[test]
    fn test_alignment_calculation() {
        assert_eq!(TensorStorage::calculate_alignment(BitNetDType::F32), 16);
        assert_eq!(TensorStorage::calculate_alignment(BitNetDType::F16), 8);
        assert_eq!(TensorStorage::calculate_alignment(BitNetDType::I64), 32);
        assert_eq!(TensorStorage::calculate_alignment(BitNetDType::BitNet158), 8);
    }
}
