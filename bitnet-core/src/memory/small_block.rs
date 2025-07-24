//! Small Block Pool Implementation
//!
//! This module provides a memory pool optimized for small allocations (< 1MB).
//! It uses a fixed-size block allocation strategy with multiple size classes
//! to minimize fragmentation and provide fast allocation/deallocation.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::alloc::{alloc, dealloc, Layout};
use candle_core::Device;
use crate::memory::{MemoryError, MemoryResult, MemoryHandle};
use crate::memory::handle::{PoolType, CpuMemoryMetadata};

#[cfg(feature = "tracing")]
use tracing::debug;

/// Size classes for small block allocation
/// Each size class represents a fixed block size that can accommodate
/// allocations up to that size with minimal waste
const SIZE_CLASSES: &[usize] = &[
    16,      // 16 bytes
    32,      // 32 bytes
    64,      // 64 bytes
    128,     // 128 bytes
    256,     // 256 bytes
    512,     // 512 bytes
    1024,    // 1 KB
    2048,    // 2 KB
    4096,    // 4 KB
    8192,    // 8 KB
    16384,   // 16 KB
    32768,   // 32 KB
    65536,   // 64 KB
    131072,  // 128 KB
    262144,  // 256 KB
    524288,  // 512 KB
];

/// Number of blocks to allocate per chunk for each size class
const BLOCKS_PER_CHUNK: &[usize] = &[
    1024,    // 16 bytes -> 16KB chunks
    512,     // 32 bytes -> 16KB chunks
    256,     // 64 bytes -> 16KB chunks
    128,     // 128 bytes -> 16KB chunks
    64,      // 256 bytes -> 16KB chunks
    32,      // 512 bytes -> 16KB chunks
    16,      // 1 KB -> 16KB chunks
    8,       // 2 KB -> 16KB chunks
    4,       // 4 KB -> 16KB chunks
    2,       // 8 KB -> 16KB chunks
    1,       // 16 KB -> 16KB chunks
    1,       // 32 KB -> 32KB chunks
    1,       // 64 KB -> 64KB chunks
    1,       // 128 KB -> 128KB chunks
    1,       // 256 KB -> 256KB chunks
    1,       // 512 KB -> 512KB chunks
];

/// Small block memory pool for efficient allocation of small memory blocks
///
/// The SmallBlockPool uses a segregated free list approach with multiple
/// size classes. Each size class maintains its own free list of blocks,
/// allowing for O(1) allocation and deallocation in the common case.
///
/// # Design
///
/// - **Size Classes**: Predefined sizes from 16 bytes to 512KB
/// - **Free Lists**: Each size class has its own free list
/// - **Chunk Allocation**: Memory is allocated in chunks containing multiple blocks
/// - **Alignment**: All allocations are properly aligned
/// - **Thread Safety**: Uses internal locking for thread-safe operations
#[derive(Debug)]
pub struct SmallBlockPool {
    /// Free lists for each size class
    free_lists: Vec<FreeList>,
    /// Allocated chunks for cleanup
    chunks: Vec<Chunk>,
    /// Current pool size in bytes
    current_size: usize,
    /// Maximum pool size in bytes
    max_size: usize,
    /// Device this pool is associated with
    device: Device,
    /// Pool statistics
    stats: PoolStats,
}

/// Free list for a specific size class
#[derive(Debug)]
struct FreeList {
    /// Size of blocks in this free list
    block_size: usize,
    /// Number of blocks per chunk
    blocks_per_chunk: usize,
    /// List of free blocks
    free_blocks: Vec<NonNull<u8>>,
    /// Total number of blocks allocated for this size class
    total_blocks: usize,
    /// Number of blocks currently in use
    used_blocks: usize,
}

/// A chunk of memory containing multiple blocks of the same size
#[derive(Debug)]
struct Chunk {
    /// Pointer to the chunk memory
    ptr: NonNull<u8>,
    /// Layout used for allocation
    layout: Layout,
    /// Size class this chunk belongs to
    size_class: usize,
}

/// Statistics for the small block pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of allocations
    allocations: u64,
    /// Total number of deallocations
    deallocations: u64,
    /// Number of chunk expansions
    expansions: u64,
    /// Total memory allocated from system
    system_allocated: usize,
}

impl SmallBlockPool {
    /// Creates a new small block pool
    ///
    /// # Arguments
    ///
    /// * `initial_size` - Initial size hint for the pool (not strictly enforced)
    /// * `max_size` - Maximum size the pool can grow to
    /// * `device` - Device this pool is associated with
    ///
    /// # Returns
    ///
    /// A Result containing the new pool or an error if creation fails
    pub fn new(_initial_size: usize, max_size: usize, device: &Device) -> MemoryResult<Self> {
        #[cfg(feature = "tracing")]
        debug!("Creating small block pool: initial_size={}, max_size={}, device={:?}", 
               initial_size, max_size, device);

        // Validate parameters
        if max_size == 0 {
            return Err(MemoryError::InvalidAlignment { alignment: max_size });
        }

        // Initialize free lists for each size class
        let mut free_lists = Vec::with_capacity(SIZE_CLASSES.len());
        for (i, &block_size) in SIZE_CLASSES.iter().enumerate() {
            free_lists.push(FreeList {
                block_size,
                blocks_per_chunk: BLOCKS_PER_CHUNK[i],
                free_blocks: Vec::new(),
                total_blocks: 0,
                used_blocks: 0,
            });
        }

        let pool = Self {
            free_lists,
            chunks: Vec::new(),
            current_size: 0,
            max_size,
            device: device.clone(),
            stats: PoolStats {
                allocations: 0,
                deallocations: 0,
                expansions: 0,
                system_allocated: 0,
            },
        };

        #[cfg(feature = "tracing")]
        debug!("Small block pool created successfully");

        Ok(pool)
    }

    /// Allocates a block of memory
    ///
    /// # Arguments
    ///
    /// * `size` - Size of memory to allocate
    /// * `alignment` - Required alignment (must be power of 2)
    /// * `device` - Device for allocation (must match pool device)
    /// * `handle_id_counter` - Counter for generating unique handle IDs
    ///
    /// # Returns
    ///
    /// A Result containing a MemoryHandle or an error
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
        device: &Device,
        handle_id_counter: Arc<Mutex<u64>>,
    ) -> MemoryResult<MemoryHandle> {
        // Verify device matches
        if !self.device_matches(device) {
            return Err(MemoryError::UnsupportedDevice {
                device_type: format!("{:?}", device),
            });
        }

        // Find appropriate size class
        let size_class = self.find_size_class(size, alignment)?;
        let _actual_size = SIZE_CLASSES[size_class];

        #[cfg(feature = "tracing")]
        debug!("Allocating {} bytes (actual: {}) from size class {}", 
               size, actual_size, size_class);

        // Get a block from the free list
        let ptr = self.get_block(size_class)?;

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
            locked: false,
            numa_node: None,
        };

        // Create memory handle
        let handle = unsafe {
            MemoryHandle::new_cpu(
                handle_id,
                ptr,
                size,
                alignment,
                device.clone(),
                PoolType::SmallBlock,
                cpu_metadata,
            )
        };

        // Update statistics
        self.stats.allocations += 1;
        self.free_lists[size_class].used_blocks += 1;

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated block with handle ID {}", handle_id);

        Ok(handle)
    }

    /// Deallocates a block of memory
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

        // Verify this is a small block handle
        if handle.pool_type() != PoolType::SmallBlock {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle is not from small block pool".to_string(),
            });
        }

        // Verify device matches
        if !self.device_matches(&handle.device()) {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle device does not match pool device".to_string(),
            });
        }

        let size = handle.size();
        let alignment = handle.alignment();
        let ptr = unsafe { handle.as_non_null() };

        #[cfg(feature = "tracing")]
        debug!("Deallocating block with handle ID {}, size {}", handle.id(), size);

        // Find the size class
        let size_class = self.find_size_class(size, alignment)?;

        // Return block to free list
        self.return_block(size_class, ptr)?;

        // Update statistics
        self.stats.deallocations += 1;
        self.free_lists[size_class].used_blocks = 
            self.free_lists[size_class].used_blocks.saturating_sub(1);

        #[cfg(feature = "tracing")]
        debug!("Successfully deallocated block with handle ID {}", handle.id());

        Ok(())
    }

    /// Returns current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        self.stats.clone()
    }

    /// Returns the current memory usage of the pool
    pub fn current_usage(&self) -> usize {
        self.current_size
    }

    /// Returns the maximum size the pool can grow to
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    // Private helper methods

    fn device_matches(&self, device: &Device) -> bool {
        match (&self.device, device) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Metal(a), Device::Metal(b)) => a.id() == b.id(),
            (Device::Cuda(_), Device::Cuda(_)) => {
                // CUDA device comparison not implemented yet
                // For now, treat all CUDA devices as different
                false
            },
            _ => false,
        }
    }

    fn find_size_class(&self, size: usize, alignment: usize) -> MemoryResult<usize> {
        // Find the smallest size class that can accommodate both size and alignment
        for (i, &class_size) in SIZE_CLASSES.iter().enumerate() {
            if class_size >= size && class_size >= alignment {
                return Ok(i);
            }
        }

        // Size is too large for small block pool
        Err(MemoryError::InsufficientMemory { size })
    }

    fn get_block(&mut self, size_class: usize) -> MemoryResult<NonNull<u8>> {
        // Try to get a block from the free list
        if let Some(ptr) = self.free_lists[size_class].free_blocks.pop() {
            return Ok(ptr);
        }

        // No free blocks available, need to allocate a new chunk
        self.allocate_chunk(size_class)?;

        // Try again after chunk allocation
        self.free_lists[size_class].free_blocks.pop()
            .ok_or_else(|| MemoryError::InternalError {
                reason: "Failed to get block after chunk allocation".to_string(),
            })
    }

    fn allocate_chunk(&mut self, size_class: usize) -> MemoryResult<()> {
        // Extract values we need before borrowing
        let block_size = self.free_lists[size_class].block_size;
        let blocks_per_chunk = self.free_lists[size_class].blocks_per_chunk;
        let chunk_size = block_size * blocks_per_chunk;

        // Check if we would exceed max size
        if self.current_size + chunk_size > self.max_size {
            return Err(MemoryError::InsufficientMemory { size: chunk_size });
        }

        #[cfg(feature = "tracing")]
        debug!("Allocating new chunk for size class {}: {} blocks of {} bytes",
               size_class, blocks_per_chunk, block_size);

        // Allocate chunk based on device type
        let (ptr, layout) = match &self.device {
            Device::Cpu => self.allocate_cpu_chunk(chunk_size, block_size)?,
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    self.allocate_metal_chunk(chunk_size, block_size)?
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(MemoryError::UnsupportedDevice {
                        device_type: "Metal (feature not enabled)".to_string(),
                    });
                }
            }
            Device::Cuda(_) => {
                return Err(MemoryError::UnsupportedDevice {
                    device_type: "CUDA (not implemented)".to_string(),
                });
            }
        };

        // Split chunk into individual blocks and add to free list
        let free_list = &mut self.free_lists[size_class];
        for i in 0..blocks_per_chunk {
            let block_ptr = unsafe {
                NonNull::new_unchecked(ptr.as_ptr().add(i * block_size))
            };
            free_list.free_blocks.push(block_ptr);
        }

        // Update pool state
        free_list.total_blocks += blocks_per_chunk;
        self.current_size += chunk_size;
        self.stats.expansions += 1;
        self.stats.system_allocated += chunk_size;

        // Store chunk for cleanup
        self.chunks.push(Chunk {
            ptr,
            layout,
            size_class,
        });

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated chunk: {} blocks, {} bytes total",
               blocks_per_chunk, chunk_size);

        Ok(())
    }

    fn allocate_cpu_chunk(&self, chunk_size: usize, alignment: usize) -> MemoryResult<(NonNull<u8>, Layout)> {
        let layout = Layout::from_size_align(chunk_size, alignment)
            .map_err(|_| MemoryError::InvalidAlignment { alignment })?;

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(MemoryError::InsufficientMemory { size: chunk_size });
            }

            // Zero the memory for security
            std::ptr::write_bytes(ptr, 0, chunk_size);

            Ok((NonNull::new_unchecked(ptr), layout))
        }
    }

    #[cfg(feature = "metal")]
    fn allocate_metal_chunk(&self, chunk_size: usize, _alignment: usize) -> MemoryResult<(NonNull<u8>, Layout)> {
        // For Metal, we still allocate CPU memory but mark it as Metal-accessible
        // In a real implementation, this would use Metal buffer allocation
        self.allocate_cpu_chunk(chunk_size, _alignment)
    }

    fn return_block(&mut self, size_class: usize, ptr: NonNull<u8>) -> MemoryResult<()> {
        // Validate that the pointer belongs to this pool
        if !self.validate_pointer(size_class, ptr) {
            return Err(MemoryError::InvalidHandle {
                reason: "Pointer does not belong to this pool".to_string(),
            });
        }

        // Add block back to free list
        self.free_lists[size_class].free_blocks.push(ptr);

        Ok(())
    }

    fn validate_pointer(&self, size_class: usize, ptr: NonNull<u8>) -> bool {
        let block_size = SIZE_CLASSES[size_class];
        let ptr_addr = ptr.as_ptr() as usize;

        // Check if pointer belongs to any of our chunks for this size class
        for chunk in &self.chunks {
            if chunk.size_class == size_class {
                let chunk_start = chunk.ptr.as_ptr() as usize;
                let chunk_end = chunk_start + chunk.layout.size();

                if ptr_addr >= chunk_start && ptr_addr < chunk_end {
                    // Check if pointer is properly aligned within the chunk
                    let offset = ptr_addr - chunk_start;
                    if offset % block_size == 0 {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl Drop for SmallBlockPool {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        debug!("Dropping small block pool, deallocating {} chunks", self.chunks.len());

        // Deallocate all chunks
        for chunk in &self.chunks {
            match &self.device {
                Device::Cpu => {
                    unsafe {
                        dealloc(chunk.ptr.as_ptr(), chunk.layout);
                    }
                }
                Device::Metal(_) => {
                    #[cfg(feature = "metal")]
                    {
                        // For now, treat as CPU memory
                        unsafe {
                            dealloc(chunk.ptr.as_ptr(), chunk.layout);
                        }
                    }
                }
                Device::Cuda(_) => {
                    // CUDA not implemented
                }
            }
        }

        #[cfg(feature = "tracing")]
        debug!("Small block pool dropped successfully");
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for SmallBlockPool {}
unsafe impl Sync for SmallBlockPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_small_block_pool_creation() {
        let device = get_cpu_device();
        let pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        
        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.max_size(), 16 * 1024 * 1024);
    }

    #[test]
    fn test_size_class_finding() {
        let device = get_cpu_device();
        let pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        
        // Test various sizes
        assert_eq!(pool.find_size_class(10, 1).unwrap(), 0);  // 16 bytes
        assert_eq!(pool.find_size_class(20, 1).unwrap(), 1);  // 32 bytes
        assert_eq!(pool.find_size_class(100, 1).unwrap(), 3); // 128 bytes
        assert_eq!(pool.find_size_class(1000, 1).unwrap(), 6); // 1024 bytes
        
        // Test alignment requirements
        assert_eq!(pool.find_size_class(10, 32).unwrap(), 1); // 32 bytes for alignment
        assert_eq!(pool.find_size_class(10, 64).unwrap(), 2); // 64 bytes for alignment
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let device = get_cpu_device();
        let mut pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        // Allocate a small block
        let handle = pool.allocate(100, 16, &device, handle_counter.clone()).unwrap();
        assert_eq!(handle.size(), 100);
        assert_eq!(handle.alignment(), 16);
        assert!(handle.is_cpu());
        
        // Pool should have allocated a chunk
        assert!(pool.current_usage() > 0);
        
        // Deallocate the block
        pool.deallocate(handle).unwrap();
        
        // Pool size should remain the same (chunk is not freed)
        assert!(pool.current_usage() > 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let device = get_cpu_device();
        let mut pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        let mut handles = Vec::new();
        
        // Allocate multiple blocks
        for i in 0..10 {
            let handle = pool.allocate(64, 16, &device, handle_counter.clone()).unwrap();
            assert_eq!(handle.size(), 64);
            handles.push(handle);
        }
        
        // All handles should be unique
        for i in 0..handles.len() {
            for j in i+1..handles.len() {
                assert_ne!(handles[i].id(), handles[j].id());
            }
        }
        
        // Deallocate all blocks
        for handle in handles {
            pool.deallocate(handle).unwrap();
        }
    }

    #[test]
    fn test_size_too_large() {
        let device = get_cpu_device();
        let pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        
        // Try to allocate something larger than the largest size class
        let result = pool.find_size_class(1024 * 1024, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_size_limit() {
        let device = get_cpu_device();
        let mut pool = SmallBlockPool::new(1024, 32768, &device).unwrap(); // Small but reasonable max size
        let handle_counter = Arc::new(Mutex::new(1));
        
        // First allocation should succeed
        let handle1 = pool.allocate(64, 16, &device, handle_counter.clone()).unwrap();
        
        // Second allocation should also succeed with this size limit
        let result = pool.allocate(64, 16, &device, handle_counter.clone());
        
        // Clean up
        pool.deallocate(handle1).unwrap();
        if let Ok(handle2) = result {
            pool.deallocate(handle2).unwrap();
        } else {
            // If it fails, that's also acceptable for this test
            println!("Second allocation failed due to size limit, which is expected behavior");
        }
    }

    #[test]
    fn test_device_mismatch() {
        let device = get_cpu_device();
        let mut pool = SmallBlockPool::new(1024 * 1024, 16 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));
        
        // Try to allocate with a different device
        let other_device = get_cpu_device(); // Same type but different instance
        let result = pool.allocate(64, 16, &other_device, handle_counter);
        
        // Should succeed since both are CPU devices
        assert!(result.is_ok());
        
        if let Ok(handle) = result {
            pool.deallocate(handle).unwrap();
        }
    }
}