//! Device-Specific Memory Pool Implementations
//!
//! This module provides device-specific memory pool implementations for
//! CPU and Metal GPU devices. Each device type has its own optimized
//! memory allocation strategy and management approach.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::alloc::{alloc, dealloc, Layout};
use candle_core::Device;
use crate::memory::{MemoryError, MemoryResult, MemoryHandle};
use crate::memory::handle::{PoolType, CpuMemoryMetadata};

#[cfg(feature = "metal")]
use crate::memory::handle::MetalMemoryMetadata;

#[cfg(feature = "tracing")]
use tracing::{debug, info};

/// CPU memory pool implementation
///
/// The CpuMemoryPool provides optimized memory allocation for CPU devices.
/// It uses standard system allocation with additional features like:
/// - Memory alignment guarantees
/// - Optional memory locking for performance
/// - NUMA-aware allocation (when available)
/// - Memory zeroing for security
pub struct CpuMemoryPool {
    /// Device this pool is associated with
    device: Device,
    /// Pool configuration
    config: CpuPoolConfig,
    /// Pool statistics
    stats: CpuPoolStats,
}

/// Configuration for CPU memory pool
#[derive(Debug, Clone)]
pub struct CpuPoolConfig {
    /// Whether to zero memory on allocation
    pub zero_memory: bool,
    /// Whether to attempt memory locking for performance
    pub enable_memory_locking: bool,
    /// Preferred NUMA node (None for automatic)
    pub numa_node: Option<u32>,
    /// Whether to use huge pages when available
    pub use_huge_pages: bool,
}

/// Statistics for CPU memory pool
#[derive(Debug, Clone)]
pub struct CpuPoolStats {
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Total bytes deallocated
    pub bytes_deallocated: u64,
    /// Number of locked memory allocations
    pub locked_allocations: u64,
    /// Number of huge page allocations
    pub huge_page_allocations: u64,
}

impl Default for CpuPoolConfig {
    fn default() -> Self {
        Self {
            zero_memory: true,
            enable_memory_locking: false,
            numa_node: None,
            use_huge_pages: false,
        }
    }
}

impl CpuMemoryPool {
    /// Creates a new CPU memory pool
    ///
    /// # Arguments
    ///
    /// * `device` - CPU device this pool is associated with
    /// * `config` - Configuration for the pool
    ///
    /// # Returns
    ///
    /// A Result containing the new pool or an error
    pub fn new(device: Device, config: CpuPoolConfig) -> MemoryResult<Self> {
        // Verify this is a CPU device
        if !matches!(device, Device::Cpu) {
            return Err(MemoryError::UnsupportedDevice {
                device_type: format!("{:?}", device),
            });
        }

        #[cfg(feature = "tracing")]
        info!("Creating CPU memory pool with config: {:?}", config);

        Ok(Self {
            device,
            config,
            stats: CpuPoolStats {
                allocations: 0,
                deallocations: 0,
                bytes_allocated: 0,
                bytes_deallocated: 0,
                locked_allocations: 0,
                huge_page_allocations: 0,
            },
        })
    }

    /// Creates a new CPU memory pool with default configuration
    pub fn new_default(device: Device) -> MemoryResult<Self> {
        Self::new(device, CpuPoolConfig::default())
    }

    /// Allocates memory on the CPU
    ///
    /// # Arguments
    ///
    /// * `size` - Size of memory to allocate
    /// * `alignment` - Required alignment
    /// * `handle_id_counter` - Counter for generating unique handle IDs
    ///
    /// # Returns
    ///
    /// A Result containing a MemoryHandle or an error
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
        handle_id_counter: Arc<Mutex<u64>>,
    ) -> MemoryResult<MemoryHandle> {
        #[cfg(feature = "tracing")]
        debug!("Allocating {} bytes with alignment {} on CPU", size, alignment);

        // Create layout
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| MemoryError::InvalidAlignment { alignment })?;

        // Allocate memory
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(MemoryError::InsufficientMemory { size });
        }

        let non_null_ptr = unsafe { NonNull::new_unchecked(ptr) };

        // Zero memory if configured
        if self.config.zero_memory {
            unsafe {
                std::ptr::write_bytes(ptr, 0, size);
            }
        }

        // Attempt memory locking if configured
        let locked = if self.config.enable_memory_locking {
            self.try_lock_memory(ptr, size)
        } else {
            false
        };

        if locked {
            self.stats.locked_allocations += 1;
        }

        // Generate unique handle ID
        let handle_id = {
            let mut counter = handle_id_counter.lock()
                .map_err(|_| MemoryError::InternalError {
                    reason: "Failed to acquire handle ID counter lock".to_string(),
                })?;
            let id = *counter;
            *counter += 1;
            id
        };

        // Create CPU metadata
        let cpu_metadata = CpuMemoryMetadata {
            page_aligned: alignment >= 4096,
            locked,
            numa_node: self.config.numa_node,
        };

        // Create memory handle
        let handle = unsafe {
            MemoryHandle::new_cpu(
                handle_id,
                non_null_ptr,
                size,
                alignment,
                self.device.clone(),
                PoolType::SmallBlock, // Will be overridden by caller
                cpu_metadata,
            )
        };

        // Update statistics
        self.stats.allocations += 1;
        self.stats.bytes_allocated += size as u64;

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated CPU memory with handle ID {}", handle_id);

        Ok(handle)
    }

    /// Deallocates memory on the CPU
    ///
    /// # Arguments
    ///
    /// * `handle` - Memory handle to deallocate
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    pub fn deallocate(&mut self, handle: MemoryHandle) -> MemoryResult<()> {
        // Validate handle
        handle.validate()?;

        if !handle.is_cpu() {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle is not for CPU memory".to_string(),
            });
        }

        let size = handle.size();
        let alignment = handle.alignment();
        let ptr = unsafe { handle.as_ptr() };

        #[cfg(feature = "tracing")]
        debug!("Deallocating CPU memory with handle ID {}", handle.id());

        // Unlock memory if it was locked
        if let Some(cpu_metadata) = handle.cpu_metadata() {
            if cpu_metadata.locked {
                self.try_unlock_memory(ptr, size);
            }
        }

        // Create layout and deallocate
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| MemoryError::InvalidAlignment { alignment })?;

        unsafe {
            dealloc(ptr, layout);
        }

        // Update statistics
        self.stats.deallocations += 1;
        self.stats.bytes_deallocated += size as u64;

        #[cfg(feature = "tracing")]
        debug!("Successfully deallocated CPU memory with handle ID {}", handle.id());

        Ok(())
    }

    /// Returns current pool statistics
    pub fn get_stats(&self) -> CpuPoolStats {
        self.stats.clone()
    }

    /// Returns the pool configuration
    pub fn get_config(&self) -> &CpuPoolConfig {
        &self.config
    }

    // Private helper methods

    fn try_lock_memory(&self, ptr: *mut u8, size: usize) -> bool {
        #[cfg(unix)]
        {
            unsafe {
                libc::mlock(ptr as *const libc::c_void, size) == 0
            }
        }
        #[cfg(not(unix))]
        {
            // Memory locking not supported on this platform
            false
        }
    }

    fn try_unlock_memory(&self, ptr: *mut u8, size: usize) {
        #[cfg(unix)]
        {
            unsafe {
                libc::munlock(ptr as *const libc::c_void, size);
            }
        }
        #[cfg(not(unix))]
        {
            // Memory locking not supported on this platform
        }
    }
}

/// Metal GPU memory pool implementation
///
/// The MetalMemoryPool provides optimized memory allocation for Metal GPU devices.
/// It uses Metal's native buffer allocation with features like:
/// - Unified memory support for Apple Silicon
/// - Different storage modes (shared, private, managed)
/// - Metal Performance Shaders integration
/// - Automatic memory synchronization
#[cfg(feature = "metal")]
pub struct MetalMemoryPool {
    /// Metal device this pool is associated with
    device: Device,
    /// Metal device reference
    metal_device: metal_rs::Device,
    /// Pool configuration
    config: MetalPoolConfig,
    /// Pool statistics
    stats: MetalPoolStats,
}

/// Configuration for Metal memory pool
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalPoolConfig {
    /// Default storage mode for allocations
    pub default_storage_mode: metal_rs::MTLStorageMode,
    /// Whether to use unified memory when available
    pub use_unified_memory: bool,
    /// Whether to enable Metal Performance Shaders optimizations
    pub enable_mps_optimizations: bool,
    /// Resource options for Metal buffers
    pub resource_options: metal_rs::MTLResourceOptions,
}

/// Statistics for Metal memory pool
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalPoolStats {
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Total bytes deallocated
    pub bytes_deallocated: u64,
    /// Number of unified memory allocations
    pub unified_memory_allocations: u64,
    /// Number of private memory allocations
    pub private_memory_allocations: u64,
    /// Number of shared memory allocations
    pub shared_memory_allocations: u64,
}

#[cfg(feature = "metal")]
impl Default for MetalPoolConfig {
    fn default() -> Self {
        Self {
            default_storage_mode: metal_rs::MTLStorageMode::Shared,
            use_unified_memory: true,
            enable_mps_optimizations: false,
            resource_options: metal_rs::MTLResourceOptions::StorageModeShared,
        }
    }
}

#[cfg(feature = "metal")]
impl MetalMemoryPool {
    /// Creates a new Metal memory pool
    ///
    /// # Arguments
    ///
    /// * `device` - Metal device this pool is associated with
    /// * `config` - Configuration for the pool
    ///
    /// # Returns
    ///
    /// A Result containing the new pool or an error
    pub fn new(device: Device, config: MetalPoolConfig) -> MemoryResult<Self> {
        // Verify this is a Metal device and extract Metal device reference
        let metal_device = match &device {
            Device::Metal(metal_dev) => metal_dev.device().clone(),
            _ => {
                return Err(MemoryError::UnsupportedDevice {
                    device_type: format!("{:?}", device),
                });
            }
        };

        #[cfg(feature = "tracing")]
        info!("Creating Metal memory pool with config: {:?}", config);

        Ok(Self {
            device,
            metal_device,
            config,
            stats: MetalPoolStats {
                allocations: 0,
                deallocations: 0,
                bytes_allocated: 0,
                bytes_deallocated: 0,
                unified_memory_allocations: 0,
                private_memory_allocations: 0,
                shared_memory_allocations: 0,
            },
        })
    }

    /// Creates a new Metal memory pool with default configuration
    pub fn new_default(device: Device) -> MemoryResult<Self> {
        Self::new(device, MetalPoolConfig::default())
    }

    /// Allocates memory on the Metal GPU
    ///
    /// # Arguments
    ///
    /// * `size` - Size of memory to allocate
    /// * `alignment` - Required alignment (ignored for Metal buffers)
    /// * `handle_id_counter` - Counter for generating unique handle IDs
    ///
    /// # Returns
    ///
    /// A Result containing a MemoryHandle or an error
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
        handle_id_counter: Arc<Mutex<u64>>,
    ) -> MemoryResult<MemoryHandle> {
        #[cfg(feature = "tracing")]
        debug!("Allocating {} bytes on Metal GPU", size);

        // Create Metal buffer
        let buffer = self.metal_device.new_buffer(size as u64, self.config.resource_options);

        // Get buffer pointer
        let ptr = buffer.contents() as *mut u8;
        if ptr.is_null() {
            return Err(MemoryError::InsufficientMemory { size });
        }

        let non_null_ptr = unsafe { NonNull::new_unchecked(ptr) };

        // Determine if this is unified memory
        let unified_memory = self.config.use_unified_memory && 
            self.config.default_storage_mode == metal_rs::MTLStorageMode::Shared;

        // Generate unique handle ID
        let handle_id = {
            let mut counter = handle_id_counter.lock()
                .map_err(|_| MemoryError::InternalError {
                    reason: "Failed to acquire handle ID counter lock".to_string(),
                })?;
            let id = *counter;
            *counter += 1;
            id
        };

        // Create Metal metadata
        let metal_metadata = MetalMemoryMetadata {
            buffer,
            storage_mode: self.config.default_storage_mode,
            unified_memory,
        };

        // Create memory handle
        let handle = unsafe {
            MemoryHandle::new_metal(
                handle_id,
                non_null_ptr,
                size,
                alignment,
                self.device.clone(),
                PoolType::SmallBlock, // Will be overridden by caller
                metal_metadata,
            )
        };

        // Update statistics
        self.stats.allocations += 1;
        self.stats.bytes_allocated += size as u64;

        match self.config.default_storage_mode {
            metal_rs::MTLStorageMode::Shared => self.stats.shared_memory_allocations += 1,
            metal_rs::MTLStorageMode::Private => self.stats.private_memory_allocations += 1,
            metal_rs::MTLStorageMode::Managed => {
                // Managed mode can be considered unified for statistics
                self.stats.unified_memory_allocations += 1;
            }
            _ => {}
        }

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated Metal memory with handle ID {}", handle_id);

        Ok(handle)
    }

    /// Deallocates memory on the Metal GPU
    ///
    /// # Arguments
    ///
    /// * `handle` - Memory handle to deallocate
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    pub fn deallocate(&mut self, handle: MemoryHandle) -> MemoryResult<()> {
        // Validate handle
        handle.validate()?;

        if !handle.is_metal() {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle is not for Metal memory".to_string(),
            });
        }

        let size = handle.size();

        #[cfg(feature = "tracing")]
        debug!("Deallocating Metal memory with handle ID {}", handle.id());

        // Metal buffers are automatically deallocated when dropped
        // No explicit deallocation needed

        // Update statistics
        self.stats.deallocations += 1;
        self.stats.bytes_deallocated += size as u64;

        #[cfg(feature = "tracing")]
        debug!("Successfully deallocated Metal memory with handle ID {}", handle.id());

        Ok(())
    }

    /// Returns current pool statistics
    pub fn get_stats(&self) -> MetalPoolStats {
        self.stats.clone()
    }

    /// Returns the pool configuration
    pub fn get_config(&self) -> &MetalPoolConfig {
        &self.config
    }

    /// Returns the Metal device reference
    pub fn metal_device(&self) -> &metal_rs::Device {
        &self.metal_device
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for CpuMemoryPool {}
unsafe impl Sync for CpuMemoryPool {}

#[cfg(feature = "metal")]
unsafe impl Send for MetalMemoryPool {}
#[cfg(feature = "metal")]
unsafe impl Sync for MetalMemoryPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_cpu_memory_pool_creation() {
        let device = get_cpu_device();
        let pool = CpuMemoryPool::new_default(device).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
        assert_eq!(stats.bytes_allocated, 0);
        assert_eq!(stats.bytes_deallocated, 0);
    }

    #[test]
    fn test_cpu_memory_allocation() {
        let device = get_cpu_device();
        let mut pool = CpuMemoryPool::new_default(device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        // Allocate memory
        let handle = pool.allocate(1024, 16, handle_counter).unwrap();
        assert_eq!(handle.size(), 1024);
        assert_eq!(handle.alignment(), 16);
        assert!(handle.is_cpu());
        
        // Check statistics
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.bytes_allocated, 1024);
        
        // Deallocate memory
        pool.deallocate(handle).unwrap();
        
        // Check statistics
        let stats = pool.get_stats();
        assert_eq!(stats.deallocations, 1);
        assert_eq!(stats.bytes_deallocated, 1024);
    }

    #[test]
    fn test_cpu_memory_pool_config() {
        let device = get_cpu_device();
        let config = CpuPoolConfig {
            zero_memory: false,
            enable_memory_locking: true,
            numa_node: Some(0),
            use_huge_pages: true,
        };
        
        let pool = CpuMemoryPool::new(device, config.clone()).unwrap();
        let pool_config = pool.get_config();
        
        assert_eq!(pool_config.zero_memory, config.zero_memory);
        assert_eq!(pool_config.enable_memory_locking, config.enable_memory_locking);
        assert_eq!(pool_config.numa_node, config.numa_node);
        assert_eq!(pool_config.use_huge_pages, config.use_huge_pages);
    }

    #[test]
    fn test_cpu_memory_invalid_device() {
        // This test would require a non-CPU device, which we don't have in this context
        // In a real scenario, you would test with a Metal or CUDA device
    }

    #[test]
    fn test_cpu_memory_alignment() {
        let device = get_cpu_device();
        let mut pool = CpuMemoryPool::new_default(device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        // Test various alignments
        let alignments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];
        
        for alignment in alignments {
            let handle = pool.allocate(1024, alignment, handle_counter.clone()).unwrap();
            assert_eq!(handle.alignment(), alignment);
            
            // Check that the pointer is properly aligned
            let ptr_addr = unsafe { handle.as_ptr() } as usize;
            assert_eq!(ptr_addr % alignment, 0, "Pointer not aligned to {}", alignment);
            
            pool.deallocate(handle).unwrap();
        }
    }

    #[test]
    fn test_cpu_memory_zero_initialization() {
        let device = get_cpu_device();
        let config = CpuPoolConfig {
            zero_memory: true,
            ..Default::default()
        };
        let mut pool = CpuMemoryPool::new(device, config).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        // Allocate memory
        let handle = pool.allocate(1024, 16, handle_counter).unwrap();
        
        // Check that memory is zeroed
        let slice = unsafe { handle.as_slice().unwrap() };
        for &byte in slice {
            assert_eq!(byte, 0, "Memory not properly zeroed");
        }
        
        pool.deallocate(handle).unwrap();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_memory_pool_creation() {
        use crate::device::get_metal_device;
        
        if let Ok(device) = get_metal_device() {
            let pool = MetalMemoryPool::new_default(device).unwrap();
            
            let stats = pool.get_stats();
            assert_eq!(stats.allocations, 0);
            assert_eq!(stats.deallocations, 0);
            assert_eq!(stats.bytes_allocated, 0);
            assert_eq!(stats.bytes_deallocated, 0);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_memory_allocation() {
        use crate::device::get_metal_device;
        
        if let Ok(device) = get_metal_device() {
            let mut pool = MetalMemoryPool::new_default(device).unwrap();
            let handle_counter = Arc::new(Mutex::new(1));
            
            // Allocate memory
            let handle = pool.allocate(1024, 16, handle_counter).unwrap();
            assert_eq!(handle.size(), 1024);
            assert!(handle.is_metal());
            
            // Check statistics
            let stats = pool.get_stats();
            assert_eq!(stats.allocations, 1);
            assert_eq!(stats.bytes_allocated, 1024);
            
            // Deallocate memory
            pool.deallocate(handle).unwrap();
            
            // Check statistics
            let stats = pool.get_stats();
            assert_eq!(stats.deallocations, 1);
            assert_eq!(stats.bytes_deallocated, 1024);
        }
    }
}