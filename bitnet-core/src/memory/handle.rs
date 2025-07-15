//! Memory Handle Implementation
//!
//! This module provides the MemoryHandle type that represents allocated memory
//! in the memory pool system. Handles are used to track and manage memory
//! allocations across different devices and pool types.

use std::ptr::NonNull;
use std::sync::Arc;
use candle_core::Device;
use crate::memory::MemoryError;

/// A handle representing allocated memory in the memory pool system
///
/// MemoryHandle provides a safe abstraction over raw memory pointers,
/// tracking allocation metadata and ensuring proper cleanup. Each handle
/// has a unique ID and contains information about the allocated memory
/// including size, alignment, device, and the raw pointer.
///
/// # Safety
///
/// The MemoryHandle ensures memory safety by:
/// - Tracking allocation metadata
/// - Preventing double-free through handle registry
/// - Providing device-aware memory management
/// - Maintaining proper alignment information
#[derive(Debug, Clone)]
pub struct MemoryHandle {
    /// Unique identifier for this memory allocation
    id: u64,
    /// Raw pointer to the allocated memory
    ptr: NonNull<u8>,
    /// Size of the allocated memory in bytes
    size: usize,
    /// Alignment of the allocated memory
    alignment: usize,
    /// Device where the memory is allocated
    device: Device,
    /// Pool type that allocated this memory
    pool_type: PoolType,
    /// Additional metadata for device-specific handling
    metadata: Arc<MemoryMetadata>,
}

/// Type of memory pool that allocated the memory
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolType {
    /// Small block pool (< 1MB allocations)
    SmallBlock,
    /// Large block pool (>= 1MB allocations)
    LargeBlock,
}

/// Device-specific metadata for memory allocations
#[derive(Debug)]
pub struct MemoryMetadata {
    /// CPU-specific metadata
    pub cpu: Option<CpuMemoryMetadata>,
    /// Metal-specific metadata
    #[cfg(feature = "metal")]
    pub metal: Option<MetalMemoryMetadata>,
}

/// CPU memory allocation metadata
#[derive(Debug)]
pub struct CpuMemoryMetadata {
    /// Whether the memory is page-aligned
    pub page_aligned: bool,
    /// Whether the memory is locked in physical memory
    pub locked: bool,
    /// NUMA node where memory is allocated (if applicable)
    pub numa_node: Option<u32>,
}

/// Metal GPU memory allocation metadata
#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalMemoryMetadata {
    /// Metal buffer object
    pub buffer: metal_rs::Buffer,
    /// Storage mode of the Metal buffer
    pub storage_mode: metal_rs::MTLStorageMode,
    /// Whether the buffer uses unified memory
    pub unified_memory: bool,
}

impl MemoryHandle {
    /// Creates a new memory handle
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this allocation
    /// * `ptr` - Raw pointer to allocated memory
    /// * `size` - Size of allocated memory in bytes
    /// * `alignment` - Alignment of allocated memory
    /// * `device` - Device where memory is allocated
    /// * `pool_type` - Type of pool that allocated the memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid, allocated memory of at least `size` bytes
    /// - The memory is properly aligned to `alignment`
    /// - The memory remains valid for the lifetime of this handle
    /// - The `id` is unique within the memory pool system
    pub unsafe fn new(
        id: u64,
        ptr: NonNull<u8>,
        size: usize,
        alignment: usize,
        device: Device,
        pool_type: PoolType,
    ) -> Self {
        let metadata = Arc::new(MemoryMetadata {
            cpu: match device {
                Device::Cpu => Some(CpuMemoryMetadata {
                    page_aligned: alignment >= 4096,
                    locked: false,
                    numa_node: None,
                }),
                _ => None,
            },
            #[cfg(feature = "metal")]
            metal: None, // Will be set by Metal-specific allocation code
        });

        Self {
            id,
            ptr,
            size,
            alignment,
            device,
            pool_type,
            metadata,
        }
    }

    /// Creates a new memory handle with CPU-specific metadata
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this allocation
    /// * `ptr` - Raw pointer to allocated memory
    /// * `size` - Size of allocated memory in bytes
    /// * `alignment` - Alignment of allocated memory
    /// * `device` - CPU device
    /// * `pool_type` - Type of pool that allocated the memory
    /// * `cpu_metadata` - CPU-specific metadata
    ///
    /// # Safety
    ///
    /// Same safety requirements as `new()`
    pub unsafe fn new_cpu(
        id: u64,
        ptr: NonNull<u8>,
        size: usize,
        alignment: usize,
        device: Device,
        pool_type: PoolType,
        cpu_metadata: CpuMemoryMetadata,
    ) -> Self {
        let metadata = Arc::new(MemoryMetadata {
            cpu: Some(cpu_metadata),
            #[cfg(feature = "metal")]
            metal: None,
        });

        Self {
            id,
            ptr,
            size,
            alignment,
            device,
            pool_type,
            metadata,
        }
    }

    /// Creates a new memory handle with Metal-specific metadata
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this allocation
    /// * `ptr` - Raw pointer to allocated memory
    /// * `size` - Size of allocated memory in bytes
    /// * `alignment` - Alignment of allocated memory
    /// * `device` - Metal device
    /// * `pool_type` - Type of pool that allocated the memory
    /// * `metal_metadata` - Metal-specific metadata
    ///
    /// # Safety
    ///
    /// Same safety requirements as `new()`
    #[cfg(feature = "metal")]
    pub unsafe fn new_metal(
        id: u64,
        ptr: NonNull<u8>,
        size: usize,
        alignment: usize,
        device: Device,
        pool_type: PoolType,
        metal_metadata: MetalMemoryMetadata,
    ) -> Self {
        let metadata = Arc::new(MemoryMetadata {
            cpu: None,
            metal: Some(metal_metadata),
        });

        Self {
            id,
            ptr,
            size,
            alignment,
            device,
            pool_type,
            metadata,
        }
    }

    /// Returns the unique ID of this memory handle
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the raw pointer to the allocated memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The memory is still valid (handle hasn't been deallocated)
    /// - Access to the memory respects the original allocation size
    /// - Concurrent access is properly synchronized
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the raw pointer as a NonNull
    ///
    /// # Safety
    ///
    /// Same safety requirements as `as_ptr()`
    pub unsafe fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Returns the size of the allocated memory in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the alignment of the allocated memory
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns the device where the memory is allocated
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the type of pool that allocated this memory
    pub fn pool_type(&self) -> PoolType {
        self.pool_type.clone()
    }

    /// Returns a reference to the memory metadata
    pub fn metadata(&self) -> &MemoryMetadata {
        &self.metadata
    }

    /// Returns CPU-specific metadata if available
    pub fn cpu_metadata(&self) -> Option<&CpuMemoryMetadata> {
        self.metadata.cpu.as_ref()
    }

    /// Returns Metal-specific metadata if available
    #[cfg(feature = "metal")]
    pub fn metal_metadata(&self) -> Option<&MetalMemoryMetadata> {
        self.metadata.metal.as_ref()
    }

    /// Checks if the memory is allocated on a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }

    /// Checks if the memory is allocated on a Metal device
    pub fn is_metal(&self) -> bool {
        matches!(self.device, Device::Metal(_))
    }

    /// Checks if the memory is allocated on a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Returns a slice view of the memory if it's on CPU
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The memory is still valid (handle hasn't been deallocated)
    /// - The memory is on a CPU device
    /// - No concurrent mutable access occurs
    pub unsafe fn as_slice(&self) -> Result<&[u8], MemoryError> {
        if !self.is_cpu() {
            return Err(MemoryError::UnsupportedDevice {
                device_type: format!("{:?}", self.device),
            });
        }

        Ok(std::slice::from_raw_parts(self.ptr.as_ptr(), self.size))
    }

    /// Returns a mutable slice view of the memory if it's on CPU
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The memory is still valid (handle hasn't been deallocated)
    /// - The memory is on a CPU device
    /// - No concurrent access occurs
    /// - The memory is not read-only
    pub unsafe fn as_mut_slice(&mut self) -> Result<&mut [u8], MemoryError> {
        if !self.is_cpu() {
            return Err(MemoryError::UnsupportedDevice {
                device_type: format!("{:?}", self.device),
            });
        }

        Ok(std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size))
    }

    /// Validates that the handle is still valid
    ///
    /// This performs basic sanity checks on the handle to detect corruption
    /// or use-after-free scenarios.
    pub fn validate(&self) -> Result<(), MemoryError> {
        // Check that size is reasonable
        if self.size == 0 {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle has zero size".to_string(),
            });
        }

        // Check that alignment is valid
        if !self.alignment.is_power_of_two() || self.alignment == 0 {
            return Err(MemoryError::InvalidHandle {
                reason: format!("Invalid alignment: {}", self.alignment),
            });
        }

        // Check that pointer is aligned
        let ptr_addr = self.ptr.as_ptr() as usize;
        if ptr_addr % self.alignment != 0 {
            return Err(MemoryError::InvalidHandle {
                reason: "Pointer is not properly aligned".to_string(),
            });
        }

        // Device-specific validation
        match &self.device {
            Device::Cpu => {
                // For CPU, just check that we have CPU metadata
                if self.metadata.cpu.is_none() {
                    return Err(MemoryError::InvalidHandle {
                        reason: "CPU device missing CPU metadata".to_string(),
                    });
                }
            }
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // For Metal, check that we have Metal metadata
                    if self.metadata.metal.is_none() {
                        return Err(MemoryError::InvalidHandle {
                            reason: "Metal device missing Metal metadata".to_string(),
                        });
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(MemoryError::UnsupportedDevice {
                        device_type: "Metal (feature not enabled)".to_string(),
                    });
                }
            }
            Device::Cuda(_) => {
                // CUDA support is not implemented yet
                return Err(MemoryError::UnsupportedDevice {
                    device_type: "CUDA (not implemented)".to_string(),
                });
            }
        }

        Ok(())
    }
}

// Implement Send and Sync for thread safety
// Safety: MemoryHandle is safe to send between threads as long as the
// underlying memory allocation is valid and the device supports it
unsafe impl Send for MemoryHandle {}
unsafe impl Sync for MemoryHandle {}

impl PartialEq for MemoryHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MemoryHandle {}

impl std::hash::Hash for MemoryHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use std::alloc::{alloc, dealloc, Layout};

    #[test]
    fn test_memory_handle_creation() {
        let device = get_cpu_device();
        let layout = Layout::from_size_align(1024, 16).unwrap();
        
        unsafe {
            let ptr = alloc(layout);
            assert!(!ptr.is_null());
            let non_null_ptr = NonNull::new(ptr).unwrap();
            
            let handle = MemoryHandle::new(
                1,
                non_null_ptr,
                1024,
                16,
                device,
                PoolType::SmallBlock,
            );
            
            assert_eq!(handle.id(), 1);
            assert_eq!(handle.size(), 1024);
            assert_eq!(handle.alignment(), 16);
            assert_eq!(handle.pool_type(), PoolType::SmallBlock);
            assert!(handle.is_cpu());
            assert!(!handle.is_metal());
            
            // Validate the handle
            handle.validate().unwrap();
            
            // Clean up
            dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_memory_handle_cpu_metadata() {
        let device = get_cpu_device();
        let layout = Layout::from_size_align(4096, 4096).unwrap();
        
        unsafe {
            let ptr = alloc(layout);
            assert!(!ptr.is_null());
            let non_null_ptr = NonNull::new(ptr).unwrap();
            
            let cpu_metadata = CpuMemoryMetadata {
                page_aligned: true,
                locked: false,
                numa_node: Some(0),
            };
            
            let handle = MemoryHandle::new_cpu(
                2,
                non_null_ptr,
                4096,
                4096,
                device,
                PoolType::LargeBlock,
                cpu_metadata,
            );
            
            let metadata = handle.cpu_metadata().unwrap();
            assert!(metadata.page_aligned);
            assert!(!metadata.locked);
            assert_eq!(metadata.numa_node, Some(0));
            
            // Clean up
            dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_memory_handle_validation() {
        let device = get_cpu_device();
        let layout = Layout::from_size_align(1024, 16).unwrap();
        
        unsafe {
            let ptr = alloc(layout);
            assert!(!ptr.is_null());
            let non_null_ptr = NonNull::new(ptr).unwrap();
            
            // Valid handle
            let handle = MemoryHandle::new(
                3,
                non_null_ptr,
                1024,
                16,
                device.clone(),
                PoolType::SmallBlock,
            );
            assert!(handle.validate().is_ok());
            
            // Invalid alignment
            let invalid_handle = MemoryHandle::new(
                4,
                non_null_ptr,
                1024,
                3, // Not power of 2
                device,
                PoolType::SmallBlock,
            );
            assert!(invalid_handle.validate().is_err());
            
            // Clean up
            dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_memory_handle_equality() {
        let device = get_cpu_device();
        let layout = Layout::from_size_align(1024, 16).unwrap();
        
        unsafe {
            let ptr1 = alloc(layout);
            let ptr2 = alloc(layout);
            assert!(!ptr1.is_null());
            assert!(!ptr2.is_null());
            
            let non_null_ptr1 = NonNull::new(ptr1).unwrap();
            let non_null_ptr2 = NonNull::new(ptr2).unwrap();
            
            let handle1 = MemoryHandle::new(
                5,
                non_null_ptr1,
                1024,
                16,
                device.clone(),
                PoolType::SmallBlock,
            );
            
            let handle2 = MemoryHandle::new(
                5, // Same ID
                non_null_ptr2,
                1024,
                16,
                device.clone(),
                PoolType::SmallBlock,
            );
            
            let handle3 = MemoryHandle::new(
                6, // Different ID
                non_null_ptr1,
                1024,
                16,
                device,
                PoolType::SmallBlock,
            );
            
            assert_eq!(handle1, handle2); // Same ID
            assert_ne!(handle1, handle3); // Different ID
            
            // Clean up
            dealloc(ptr1, layout);
            dealloc(ptr2, layout);
        }
    }

    #[test]
    fn test_memory_handle_slice_access() {
        let device = get_cpu_device();
        let layout = Layout::from_size_align(1024, 16).unwrap();
        
        unsafe {
            let ptr = alloc(layout);
            assert!(!ptr.is_null());
            let non_null_ptr = NonNull::new(ptr).unwrap();
            
            // Initialize memory with test pattern
            for i in 0..1024 {
                *ptr.add(i) = (i % 256) as u8;
            }
            
            let handle = MemoryHandle::new(
                7,
                non_null_ptr,
                1024,
                16,
                device,
                PoolType::SmallBlock,
            );
            
            // Test slice access
            let slice = handle.as_slice().unwrap();
            assert_eq!(slice.len(), 1024);
            assert_eq!(slice[0], 0);
            assert_eq!(slice[255], 255);
            assert_eq!(slice[256], 0);
            
            // Clean up
            dealloc(ptr, layout);
        }
    }
}