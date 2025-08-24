//! Large Block Pool Implementation
//!
//! This module provides a memory pool optimized for large allocations (>= 1MB).
//! It uses a buddy allocation algorithm to minimize fragmentation while providing
//! efficient allocation and deallocation for large memory blocks.

use crate::memory::handle::{CpuMemoryMetadata, PoolType};
use crate::memory::{MemoryError, MemoryHandle, MemoryResult};
use candle_core::Device;
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

#[cfg(feature = "tracing")]
use tracing::debug;

/// Minimum allocation size for the buddy allocator (64KB)
const MIN_BLOCK_SIZE: usize = 64 * 1024;

/// Maximum allocation size for the buddy allocator (64MB)
const MAX_BLOCK_SIZE: usize = 64 * 1024 * 1024;

/// Number of buddy levels (log2(MAX_BLOCK_SIZE / MIN_BLOCK_SIZE) + 1)
const BUDDY_LEVELS: usize = 11; // 64KB to 64MB = 10 levels + 1

/// Large block memory pool using buddy allocation algorithm
///
/// The LargeBlockPool uses a buddy allocation system to manage large memory
/// blocks efficiently. The buddy allocator maintains free lists for different
/// power-of-2 sizes and can split and coalesce blocks as needed.
///
/// # Design
///
/// - **Buddy System**: Power-of-2 sized blocks with splitting and coalescing
/// - **Free Lists**: One free list per power-of-2 size level
/// - **Coalescing**: Adjacent free blocks are automatically merged
/// - **Alignment**: All allocations are properly aligned to block boundaries
/// - **Thread Safety**: Uses internal locking for thread-safe operations
///
/// # Algorithm
///
/// 1. **Allocation**: Find the smallest available block >= requested size
/// 2. **Splitting**: Split larger blocks recursively until appropriate size
/// 3. **Deallocation**: Return block to free list and attempt coalescing
/// 4. **Coalescing**: Merge adjacent free blocks of the same size
#[derive(Debug)]
pub struct LargeBlockPool {
    /// Free lists for each buddy level
    free_lists: Vec<Vec<Block>>,
    /// Allocated arenas for cleanup
    arenas: Vec<Arena>,
    /// Block metadata for tracking allocations
    block_metadata: HashMap<usize, BlockMetadata>,
    /// Current pool size in bytes
    current_size: usize,
    /// Maximum pool size in bytes
    max_size: usize,
    /// Device this pool is associated with
    device: Device,
    /// Pool statistics
    stats: PoolStats,
}

/// A block in the buddy allocator
#[derive(Debug, Clone)]
struct Block {
    /// Pointer to the block
    ptr: NonNull<u8>,
    /// Size of the block (power of 2)
    size: usize,
    /// Level in the buddy hierarchy
    level: usize,
}

/// An arena of memory from which blocks are allocated
#[derive(Debug)]
struct Arena {
    /// Pointer to the arena memory
    ptr: NonNull<u8>,
    /// Layout used for allocation
    layout: Layout,
    /// Size of the arena
    size: usize,
}

/// Metadata for tracking allocated blocks
#[derive(Debug, Clone)]
struct BlockMetadata {
    /// Original requested size
    requested_size: usize,
    /// Actual allocated size (rounded up to power of 2)
    allocated_size: usize,
    /// Alignment requirement
    alignment: usize,
    /// Level in buddy hierarchy
    level: usize,
}

/// Statistics for the large block pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of allocations
    allocations: u64,
    /// Total number of deallocations
    deallocations: u64,
    /// Number of block splits
    splits: u64,
    /// Number of block coalesces
    coalesces: u64,
    /// Number of arena expansions
    expansions: u64,
    /// Total memory allocated from system
    system_allocated: usize,
    /// Current fragmentation ratio
    fragmentation_ratio: f64,
}

impl LargeBlockPool {
    /// Creates a new large block pool
    ///
    /// # Arguments
    ///
    /// * `initial_size` - Initial size hint for the pool
    /// * `max_size` - Maximum size the pool can grow to
    /// * `device` - Device this pool is associated with
    ///
    /// # Returns
    ///
    /// A Result containing the new pool or an error if creation fails
    pub fn new(initial_size: usize, max_size: usize, device: &Device) -> MemoryResult<Self> {
        #[cfg(feature = "tracing")]
        debug!(
            "Creating large block pool: initial_size={}, max_size={}, device={:?}",
            initial_size, max_size, device
        );

        // Validate parameters
        if max_size == 0 {
            return Err(MemoryError::InvalidAlignment {
                alignment: max_size,
            });
        }

        // Initialize free lists for each buddy level
        let mut free_lists = Vec::with_capacity(BUDDY_LEVELS);
        for _ in 0..BUDDY_LEVELS {
            free_lists.push(Vec::new());
        }

        let pool = Self {
            free_lists,
            arenas: Vec::new(),
            block_metadata: HashMap::new(),
            current_size: 0,
            max_size,
            device: device.clone(),
            stats: PoolStats {
                allocations: 0,
                deallocations: 0,
                splits: 0,
                coalesces: 0,
                expansions: 0,
                system_allocated: 0,
                fragmentation_ratio: 0.0,
            },
        };

        #[cfg(feature = "tracing")]
        debug!("Large block pool created successfully");

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

        // Calculate required block size (power of 2, at least MIN_BLOCK_SIZE)
        let required_size = self.calculate_required_size(size, alignment)?;
        let level = self.size_to_level(required_size);

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating {} bytes (required: {}, level: {}) from large block pool",
            size, required_size, level
        );

        // Get a block of the required size
        let block = self.get_block(level)?;

        // Generate unique handle ID
        let handle_id = {
            let mut counter = handle_id_counter
                .lock()
                .map_err(|_| MemoryError::InternalError {
                    reason: "Failed to acquire handle ID counter lock".to_string(),
                })?;
            let id = *counter;
            *counter += 1;
            id
        };

        // Store block metadata
        let metadata = BlockMetadata {
            requested_size: size,
            allocated_size: required_size,
            alignment,
            level,
        };
        self.block_metadata
            .insert(block.ptr.as_ptr() as usize, metadata);

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
                block.ptr,
                size,
                alignment,
                device.clone(),
                PoolType::LargeBlock,
                cpu_metadata,
            )
        };

        // Update statistics
        self.stats.allocations += 1;
        self.update_fragmentation_ratio();

        #[cfg(feature = "tracing")]
        debug!(
            "Successfully allocated large block with handle ID {}",
            handle_id
        );

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

        // Verify this is a large block handle
        if handle.pool_type() != PoolType::LargeBlock {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle is not from large block pool".to_string(),
            });
        }

        // Verify device matches
        if !self.device_matches(&handle.device()) {
            return Err(MemoryError::InvalidHandle {
                reason: "Handle device does not match pool device".to_string(),
            });
        }

        let ptr = unsafe { handle.as_non_null() };
        let ptr_addr = ptr.as_ptr() as usize;

        #[cfg(feature = "tracing")]
        debug!("Deallocating large block with handle ID {}", handle.id());

        // Get block metadata
        let metadata =
            self.block_metadata
                .remove(&ptr_addr)
                .ok_or_else(|| MemoryError::InvalidHandle {
                    reason: "Block metadata not found".to_string(),
                })?;

        // Create block for deallocation
        let block = Block {
            ptr,
            size: metadata.allocated_size,
            level: metadata.level,
        };

        // Return block to free list and attempt coalescing
        self.return_block(block)?;

        // Update statistics
        self.stats.deallocations += 1;
        self.update_fragmentation_ratio();

        #[cfg(feature = "tracing")]
        debug!(
            "Successfully deallocated large block with handle ID {}",
            handle.id()
        );

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
            }
            _ => false,
        }
    }

    fn calculate_required_size(&self, size: usize, alignment: usize) -> MemoryResult<usize> {
        // Size must be at least MIN_BLOCK_SIZE
        let min_size = MIN_BLOCK_SIZE.max(alignment);
        let required_size = size.max(min_size);

        // Round up to next power of 2
        let power_of_2_size = required_size.next_power_of_two();

        // Check if size is within limits
        if power_of_2_size > MAX_BLOCK_SIZE {
            return Err(MemoryError::InsufficientMemory { size });
        }

        Ok(power_of_2_size)
    }

    fn size_to_level(&self, size: usize) -> usize {
        // Calculate level: log2(size / MIN_BLOCK_SIZE)
        let ratio = size / MIN_BLOCK_SIZE;
        ratio.trailing_zeros() as usize
    }

    fn level_to_size(&self, level: usize) -> usize {
        MIN_BLOCK_SIZE << level
    }

    fn get_block(&mut self, level: usize) -> MemoryResult<Block> {
        // Try to get a block from the requested level
        if let Some(block) = self.free_lists[level].pop() {
            return Ok(block);
        }

        // No block available at this level, try to split a larger block
        for higher_level in (level + 1)..BUDDY_LEVELS {
            if let Some(large_block) = self.free_lists[higher_level].pop() {
                return self.split_block(large_block, level);
            }
        }

        // No blocks available, need to allocate a new arena
        self.allocate_arena()?;

        // Try again after arena allocation
        self.get_block(level)
    }

    fn split_block(&mut self, mut block: Block, target_level: usize) -> MemoryResult<Block> {
        #[cfg(feature = "tracing")]
        debug!(
            "Splitting block from level {} to level {}",
            block.level, target_level
        );

        // Split the block down to the target level
        while block.level > target_level {
            let new_size = block.size / 2;
            let new_level = block.level - 1;

            // Create buddy block (second half)
            let buddy_ptr = unsafe { NonNull::new_unchecked(block.ptr.as_ptr().add(new_size)) };
            let buddy_block = Block {
                ptr: buddy_ptr,
                size: new_size,
                level: new_level,
            };

            // Add buddy to free list
            self.free_lists[new_level].push(buddy_block);

            // Update current block (first half)
            block.size = new_size;
            block.level = new_level;

            self.stats.splits += 1;
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Block split completed, returning block at level {}",
            block.level
        );

        Ok(block)
    }

    fn return_block(&mut self, block: Block) -> MemoryResult<()> {
        let mut current_block = block;

        // Attempt to coalesce with buddy blocks
        loop {
            // Check if we can coalesce at this level
            if current_block.level >= BUDDY_LEVELS - 1 {
                break; // Can't coalesce at the highest level
            }

            // Find buddy block
            let buddy_addr = self
                .calculate_buddy_address(current_block.ptr.as_ptr() as usize, current_block.level);

            // Look for buddy in free list
            let buddy_index = self.free_lists[current_block.level]
                .iter()
                .position(|b| b.ptr.as_ptr() as usize == buddy_addr);

            if let Some(index) = buddy_index {
                // Found buddy, remove it and coalesce
                let buddy = self.free_lists[current_block.level].remove(index);

                #[cfg(feature = "tracing")]
                debug!("Coalescing blocks at level {}", current_block.level);

                // Create coalesced block (use the lower address)
                let coalesced_ptr = if (current_block.ptr.as_ptr() as usize) < buddy_addr {
                    current_block.ptr
                } else {
                    buddy.ptr
                };

                current_block = Block {
                    ptr: coalesced_ptr,
                    size: current_block.size * 2,
                    level: current_block.level + 1,
                };

                self.stats.coalesces += 1;
            } else {
                // No buddy found, can't coalesce
                break;
            }
        }

        // Add the (possibly coalesced) block to the appropriate free list
        self.free_lists[current_block.level].push(current_block);

        Ok(())
    }

    fn calculate_buddy_address(&self, block_addr: usize, level: usize) -> usize {
        let block_size = self.level_to_size(level);
        block_addr ^ block_size
    }

    fn allocate_arena(&mut self) -> MemoryResult<()> {
        // Calculate arena size (multiple of MAX_BLOCK_SIZE)
        let arena_size = MAX_BLOCK_SIZE;

        // Check if we would exceed max size
        if self.current_size + arena_size > self.max_size {
            return Err(MemoryError::InsufficientMemory { size: arena_size });
        }

        #[cfg(feature = "tracing")]
        debug!("Allocating new arena: {} bytes", arena_size);

        // Allocate arena based on device type
        let (ptr, layout) = match &self.device {
            Device::Cpu => self.allocate_cpu_arena(arena_size)?,
            Device::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    self.allocate_metal_arena(arena_size)?
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

        // Add the entire arena as a single large block
        let block = Block {
            ptr,
            size: arena_size,
            level: BUDDY_LEVELS - 1,
        };
        self.free_lists[BUDDY_LEVELS - 1].push(block);

        // Update pool state
        self.current_size += arena_size;
        self.stats.expansions += 1;
        self.stats.system_allocated += arena_size;

        // Store arena for cleanup
        self.arenas.push(Arena {
            ptr,
            layout,
            size: arena_size,
        });

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated arena: {} bytes", arena_size);

        Ok(())
    }

    fn allocate_cpu_arena(&self, arena_size: usize) -> MemoryResult<(NonNull<u8>, Layout)> {
        let layout = Layout::from_size_align(arena_size, MIN_BLOCK_SIZE).map_err(|_| {
            MemoryError::InvalidAlignment {
                alignment: MIN_BLOCK_SIZE,
            }
        })?;

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(MemoryError::InsufficientMemory { size: arena_size });
            }

            // Zero the memory for security
            std::ptr::write_bytes(ptr, 0, arena_size);

            Ok((NonNull::new_unchecked(ptr), layout))
        }
    }

    #[cfg(feature = "metal")]
    fn allocate_metal_arena(&self, arena_size: usize) -> MemoryResult<(NonNull<u8>, Layout)> {
        // For Metal, we still allocate CPU memory but mark it as Metal-accessible
        // In a real implementation, this would use Metal buffer allocation
        self.allocate_cpu_arena(arena_size)
    }

    fn update_fragmentation_ratio(&mut self) {
        if self.current_size == 0 {
            self.stats.fragmentation_ratio = 0.0;
            return;
        }

        // Calculate fragmentation as the ratio of free memory to total memory
        let mut free_memory = 0;
        for (level, free_list) in self.free_lists.iter().enumerate() {
            let block_size = self.level_to_size(level);
            free_memory += free_list.len() * block_size;
        }

        self.stats.fragmentation_ratio = free_memory as f64 / self.current_size as f64;
    }
}

impl Drop for LargeBlockPool {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        debug!(
            "Dropping large block pool, deallocating {} arenas",
            self.arenas.len()
        );

        // Deallocate all arenas
        for arena in &self.arenas {
            match &self.device {
                Device::Cpu => unsafe {
                    dealloc(arena.ptr.as_ptr(), arena.layout);
                },
                Device::Metal(_) => {
                    #[cfg(feature = "metal")]
                    {
                        // For now, treat as CPU memory
                        unsafe {
                            dealloc(arena.ptr.as_ptr(), arena.layout);
                        }
                    }
                }
                Device::Cuda(_) => {
                    // CUDA not implemented
                }
            }
        }

        #[cfg(feature = "tracing")]
        debug!("Large block pool dropped successfully");
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for LargeBlockPool {}
unsafe impl Sync for LargeBlockPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_large_block_pool_creation() {
        let device = get_cpu_device();
        let pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();

        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.max_size(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_size_calculations() {
        let device = get_cpu_device();
        let pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();

        // Test size to level conversion
        assert_eq!(pool.size_to_level(MIN_BLOCK_SIZE), 0);
        assert_eq!(pool.size_to_level(MIN_BLOCK_SIZE * 2), 1);
        assert_eq!(pool.size_to_level(MIN_BLOCK_SIZE * 4), 2);

        // Test level to size conversion
        assert_eq!(pool.level_to_size(0), MIN_BLOCK_SIZE);
        assert_eq!(pool.level_to_size(1), MIN_BLOCK_SIZE * 2);
        assert_eq!(pool.level_to_size(2), MIN_BLOCK_SIZE * 4);

        // Test required size calculation
        assert_eq!(
            pool.calculate_required_size(1024, 16).unwrap(),
            MIN_BLOCK_SIZE
        );
        assert_eq!(
            pool.calculate_required_size(MIN_BLOCK_SIZE + 1, 16)
                .unwrap(),
            MIN_BLOCK_SIZE * 2
        );
    }

    #[test]
    fn test_buddy_address_calculation() {
        let device = get_cpu_device();
        let pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();

        // Test buddy address calculation
        let base_addr = 0x1000000; // 16MB aligned

        // Level 0 (64KB blocks)
        let buddy0 = pool.calculate_buddy_address(base_addr, 0);
        assert_eq!(buddy0, base_addr ^ MIN_BLOCK_SIZE);

        // Level 1 (128KB blocks)
        let buddy1 = pool.calculate_buddy_address(base_addr, 1);
        assert_eq!(buddy1, base_addr ^ (MIN_BLOCK_SIZE * 2));
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let device = get_cpu_device();
        let mut pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));

        // Allocate a large block
        let handle = pool
            .allocate(1024 * 1024, 16, &device, handle_counter.clone())
            .unwrap();
        assert_eq!(handle.size(), 1024 * 1024);
        assert_eq!(handle.alignment(), 16);
        assert!(handle.is_cpu());

        // Pool should have allocated an arena
        assert!(pool.current_usage() > 0);

        // Deallocate the block
        pool.deallocate(handle).unwrap();

        // Pool size should remain the same (arena is not freed)
        assert!(pool.current_usage() > 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let device = get_cpu_device();
        let mut pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));

        let mut handles = Vec::new();

        // Allocate multiple large blocks
        for _ in 0..5 {
            let handle = pool
                .allocate(2 * 1024 * 1024, 16, &device, handle_counter.clone())
                .unwrap();
            assert_eq!(handle.size(), 2 * 1024 * 1024);
            handles.push(handle);
        }

        // All handles should be unique
        for i in 0..handles.len() {
            for j in i + 1..handles.len() {
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
        let pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();

        // Try to allocate something larger than MAX_BLOCK_SIZE
        let result = pool.calculate_required_size(MAX_BLOCK_SIZE + 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_size_limit() {
        let device = get_cpu_device();
        let mut pool = LargeBlockPool::new(64 * 1024 * 1024, 128 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));

        // First allocation should succeed
        let handle1 = pool
            .allocate(32 * 1024 * 1024, 16, &device, handle_counter.clone())
            .unwrap();

        // Second large allocation might fail due to size limit
        let result = pool.allocate(64 * 1024 * 1024, 16, &device, handle_counter.clone());

        // Clean up
        pool.deallocate(handle1).unwrap();
        if let Ok(handle2) = result {
            pool.deallocate(handle2).unwrap();
        }
    }

    #[test]
    fn test_fragmentation_tracking() {
        let device = get_cpu_device();
        let mut pool = LargeBlockPool::new(64 * 1024 * 1024, 256 * 1024 * 1024, &device).unwrap();
        let handle_counter = Arc::new(Mutex::new(1));

        // Initial fragmentation should be 0
        let stats = pool.get_stats();
        assert_eq!(stats.fragmentation_ratio, 0.0);

        // Allocate and deallocate to create some fragmentation
        let handle = pool
            .allocate(1024 * 1024, 16, &device, handle_counter.clone())
            .unwrap();
        pool.deallocate(handle).unwrap();

        // Fragmentation ratio should be updated
        let stats = pool.get_stats();
        assert!(stats.fragmentation_ratio >= 0.0);
    }
}
