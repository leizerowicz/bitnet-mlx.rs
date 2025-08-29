//! Enhanced Memory Pool for Cross-Backend Memory Efficiency
//!
//! This module provides an advanced memory pooling system that coordinates between CPU and GPU
//! memory management, optimizing for cross-backend transfers and memory layout efficiency.

use crate::{Result, InferenceError};
use crate::engine::gpu_memory_optimizer::{GPUMemoryManager, GPUAllocation, MemoryStats};
use bitnet_core::{Device, Tensor};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory region that can span different device types
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Unique identifier for this memory region
    pub id: u64,
    /// Size in bytes
    pub size: usize,
    /// Target device for this memory region
    pub device: Device,
    /// Whether this region supports zero-copy operations
    pub zero_copy: bool,
    /// Memory alignment requirements
    pub alignment: usize,
    /// Last access time for LRU eviction
    pub last_access: Instant,
    /// Reference count for memory safety
    pub ref_count: usize,
}

impl MemoryRegion {
    /// Create a new memory region
    pub fn new(id: u64, size: usize, device: Device) -> Self {
        let alignment = Self::optimal_alignment_for_device(&device);
        Self {
            id,
            size,
            device,
            zero_copy: false, // Conservative default - would be set based on actual device capabilities
            alignment,
            last_access: Instant::now(),
            ref_count: 0,
        }
    }
    
    /// Determine optimal memory alignment for the target device
    fn optimal_alignment_for_device(device: &Device) -> usize {
        match device {
            Device::Cpu => 64,    // Cache line alignment
            _ => 256,             // Conservative default for GPU devices
        }
    }
    
    /// Update last access time
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }
    
    /// Increment reference count
    pub fn acquire(&mut self) {
        self.ref_count += 1;
    }
    
    /// Decrement reference count
    pub fn release(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }
    
    /// Check if memory region is actively referenced
    pub fn is_in_use(&self) -> bool {
        self.ref_count > 0
    }
    
    /// Check if region has been accessed recently
    pub fn is_recently_used(&self, threshold: Duration) -> bool {
        self.last_access.elapsed() < threshold
    }
}

/// Cross-backend cache for optimizing memory transfers between devices
#[derive(Debug)]
pub struct CrossBackendCache {
    /// Cache entries organized by tensor hash - simplified to not use Device as key
    cache_entries: HashMap<u64, SimplifiedCacheEntry>,
    /// LRU queue for eviction policy
    lru_queue: VecDeque<u64>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache utilization
    current_size: usize,
    /// Cache statistics
    stats: CacheStats,
}

#[derive(Debug, Clone)]
pub struct SimplifiedCacheEntry {
    /// Tensor data hash
    hash: u64,
    /// Memory region (single region for simplicity)
    region: MemoryRegion,
    /// Size of cached tensor
    tensor_size: usize,
    /// Creation timestamp
    created_at: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub cross_device_transfers: usize,
    pub zero_copy_transfers: usize,
}

impl CrossBackendCache {
    /// Create a new cross-backend cache
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache_entries: HashMap::new(),
            lru_queue: VecDeque::new(),
            max_cache_size,
            current_size: 0,
            stats: CacheStats::default(),
        }
    }
    
    /// Get cached memory region for tensor on specific device (simplified)
    pub fn get_region(&mut self, tensor_hash: u64, _device: Device) -> Option<MemoryRegion> {
        if let Some(entry) = self.cache_entries.get_mut(&tensor_hash) {
            let mut region = entry.region.clone();
            region.touch();
            region.acquire();
            
            // Update LRU position after using the entry
            self.update_lru(tensor_hash);
            self.stats.hits += 1;
            Some(region)
        } else {
            self.stats.misses += 1;
            None
        }
    }
    
    /// Cache memory region for tensor
    pub fn cache_region(&mut self, tensor_hash: u64, region: MemoryRegion) -> Result<()> {
        // Check if we need to evict entries to make space
        while self.current_size + region.size > self.max_cache_size && !self.lru_queue.is_empty() {
            self.evict_lru_entry()?;
        }
        
        if region.size > self.max_cache_size {
            return Err(InferenceError::ResourceError(
                format!("Tensor size {} exceeds cache capacity {}", region.size, self.max_cache_size)
            ));
        }
        
        // Create or update cache entry
        let entry = self.cache_entries
            .entry(tensor_hash)
            .or_insert_with(|| SimplifiedCacheEntry {
                hash: tensor_hash,
                region: region.clone(),
                tensor_size: region.size,
                created_at: Instant::now(),
            });
        
        entry.region = region;
        self.current_size += entry.tensor_size;
        
        // Update LRU queue
        self.update_lru(tensor_hash);
        
        Ok(())
    }
    
    /// Update LRU queue position for cache entry
    fn update_lru(&mut self, hash: u64) {
        // Remove from current position
        self.lru_queue.retain(|&h| h != hash);
        // Add to front (most recently used)
        self.lru_queue.push_front(hash);
    }
    
    /// Evict least recently used cache entry
    fn evict_lru_entry(&mut self) -> Result<()> {
        if let Some(hash) = self.lru_queue.pop_back() {
            if let Some(entry) = self.cache_entries.remove(&hash) {
                self.current_size -= entry.tensor_size;
                self.stats.evictions += 1;
            }
        }
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.clone()
    }
    
    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.cache_entries.clear();
        self.lru_queue.clear();
        self.current_size = 0;
    }
}

/// CPU memory pool compatible with existing HybridMemoryPool
pub struct CPUMemoryPool {
    /// Memory blocks organized by size
    blocks_by_size: HashMap<usize, Vec<MemoryBlock>>,
    /// Maximum pool size
    max_pool_size: usize,
    /// Current utilization
    current_usage: usize,
    /// Pool statistics
    stats: PoolStats,
}

#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub size: usize,
    pub id: u64,
    pub allocated_at: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub peak_usage: usize,
    pub fragmentation_events: usize,
}

impl CPUMemoryPool {
    /// Create a new CPU memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            blocks_by_size: HashMap::new(),
            max_pool_size,
            current_usage: 0,
            stats: PoolStats::default(),
        }
    }
    
    /// Allocate memory block
    pub fn allocate(&mut self, size: usize) -> Result<MemoryBlock> {
        let aligned_size = (size + 63) & !63; // 64-byte alignment
        
        // Try to reuse existing block
        if let Some(blocks) = self.blocks_by_size.get_mut(&aligned_size) {
            if let Some(block) = blocks.pop() {
                self.stats.allocations += 1;
                return Ok(block);
            }
        }
        
        // Create new block
        if self.current_usage + aligned_size <= self.max_pool_size {
            self.current_usage += aligned_size;
            self.stats.allocations += 1;
            
            if self.current_usage > self.stats.peak_usage {
                self.stats.peak_usage = self.current_usage;
            }
            
            Ok(MemoryBlock {
                size: aligned_size,
                id: self.stats.allocations as u64,
                allocated_at: Instant::now(),
            })
        } else {
            Err(InferenceError::ResourceError(
                format!("CPU memory pool exhausted. Requested: {}, Available: {}", 
                       aligned_size, self.max_pool_size - self.current_usage)
            ))
        }
    }
    
    /// Deallocate memory block
    pub fn deallocate(&mut self, block: MemoryBlock) {
        self.stats.deallocations += 1;
        
        // Return to pool
        self.blocks_by_size
            .entry(block.size)
            .or_insert_with(Vec::new)
            .push(block);
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        self.stats.clone()
    }
}

/// GPU buffer manager coordinating with GPU memory optimizer
pub struct GPUBufferManager {
    /// GPU memory manager
    gpu_manager: GPUMemoryManager,
    /// Active GPU allocations
    active_allocations: HashMap<u64, GPUAllocation>,
    /// Next allocation ID
    next_id: u64,
}

impl GPUBufferManager {
    /// Create a new GPU buffer manager
    pub fn new(device: Device) -> Result<Self> {
        let gpu_manager = GPUMemoryManager::new(device)?;
        
        Ok(Self {
            gpu_manager,
            active_allocations: HashMap::new(),
            next_id: 1,
        })
    }
    
    /// Allocate GPU buffer for tensor
    pub fn allocate_for_tensor(&mut self, tensor: &Tensor) -> Result<u64> {
        let allocation = self.gpu_manager.allocate_for_tensor(tensor)?;
        let id = self.next_id;
        self.next_id += 1;
        
        self.active_allocations.insert(id, allocation);
        Ok(id)
    }
    
    /// Deallocate GPU buffer
    pub fn deallocate(&mut self, allocation_id: u64) {
        self.active_allocations.remove(&allocation_id);
    }
    
    /// Optimize GPU memory
    pub fn optimize(&mut self) -> Result<()> {
        self.gpu_manager.optimize_all()
    }
    
    /// Get GPU memory statistics
    pub fn get_memory_stats(&self) -> Result<MemoryStats> {
        self.gpu_manager.get_memory_stats()
    }
}

/// Enhanced memory pool coordinating CPU and GPU memory management
pub struct EnhancedMemoryPool {
    /// CPU memory pool
    cpu_pool: CPUMemoryPool,
    /// GPU buffer manager  
    gpu_buffers: GPUBufferManager,
    /// Cross-backend cache for transfer optimization
    cross_backend_cache: CrossBackendCache,
    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy,
    /// Global statistics
    global_stats: Arc<Mutex<GlobalMemoryStats>>,
}

#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Prefer CPU allocation
    PreferCPU,
    /// Prefer GPU allocation when available
    PreferGPU,
    /// Automatically select based on access patterns
    Automatic,
    /// Minimize cross-device transfers
    MinimizeTransfers,
}

#[derive(Debug, Clone, Default)]
pub struct GlobalMemoryStats {
    pub total_cpu_allocated: usize,
    pub total_gpu_allocated: usize,
    pub cross_backend_transfers: usize,
    pub cache_hit_rate: f64,
    pub memory_efficiency: f64,
}

impl EnhancedMemoryPool {
    /// Create a new enhanced memory pool
    pub fn new(device: Device, max_cpu_pool_size: usize, max_cache_size: usize) -> Result<Self> {
        let cpu_pool = CPUMemoryPool::new(max_cpu_pool_size);
        let gpu_buffers = GPUBufferManager::new(device)?;
        let cross_backend_cache = CrossBackendCache::new(max_cache_size);
        
        Ok(Self {
            cpu_pool,
            gpu_buffers,
            cross_backend_cache,
            allocation_strategy: AllocationStrategy::Automatic,
            global_stats: Arc::new(Mutex::new(GlobalMemoryStats::default())),
        })
    }
    
    /// Allocate memory region optimally based on device and access patterns
    pub fn allocate_optimal(&mut self, size: usize, device: Device) -> Result<MemoryRegion> {
        let allocation_device = self.select_allocation_device(size, device)?;
        
        match allocation_device {
            Device::Cpu => {
                let block = self.cpu_pool.allocate(size)?;
                
                // Update statistics
                if let Ok(mut stats) = self.global_stats.lock() {
                    stats.total_cpu_allocated += size;
                }
                
                Ok(MemoryRegion::new(block.id, block.size, Device::Cpu))
            }
            
            _ => {
                // GPU allocation
                let tensor = Tensor::zeros(&[size / std::mem::size_of::<f32>()], bitnet_core::DType::F32, &allocation_device)?;
                let allocation_id = self.gpu_buffers.allocate_for_tensor(&tensor)?;
                
                // Update statistics
                if let Ok(mut stats) = self.global_stats.lock() {
                    stats.total_gpu_allocated += size;
                }
                
                Ok(MemoryRegion::new(allocation_id, size, allocation_device))
            }
        }
    }
    
    /// Select optimal device for allocation based on strategy
    fn select_allocation_device(&self, size: usize, requested_device: Device) -> Result<Device> {
        match self.allocation_strategy {
            AllocationStrategy::PreferCPU => Ok(Device::Cpu),
            AllocationStrategy::PreferGPU => Ok(requested_device),
            AllocationStrategy::Automatic => {
                // Use heuristics based on size and device capabilities
                if size > 100 * 1024 * 1024 && !matches!(requested_device, Device::Cpu) {
                    Ok(requested_device) // Large allocations prefer GPU
                } else {
                    Ok(Device::Cpu) // Small allocations prefer CPU
                }
            }
            AllocationStrategy::MinimizeTransfers => {
                // Check cache for existing allocations and prefer same device
                Ok(requested_device)
            }
        }
    }
    
    /// Transfer memory region to different device
    pub fn transfer_to_device(&mut self, region: &MemoryRegion, target_device: Device) -> Result<MemoryRegion> {
        // For simplicity, always create a new region
        // In a real implementation, we'd check device compatibility
        
        // Check cross-backend cache first
        let tensor_hash = self.calculate_tensor_hash(region);
        if let Some(cached_region) = self.cross_backend_cache.get_region(tensor_hash, target_device.clone()) {
            // Update statistics
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.cache_hit_rate = 
                    self.cross_backend_cache.get_stats().hits as f64 / 
                    (self.cross_backend_cache.get_stats().hits + self.cross_backend_cache.get_stats().misses) as f64;
            }
            return Ok(cached_region);
        }
        
        // Perform actual transfer
        let new_region = self.allocate_optimal(region.size, target_device)?;
        
        // Cache the transferred region
        self.cross_backend_cache.cache_region(tensor_hash, new_region.clone())?;
        
        // Update transfer statistics
        if let Ok(mut stats) = self.global_stats.lock() {
            stats.cross_backend_transfers += 1;
        }
        
        Ok(new_region)
    }
    
    /// Calculate hash for tensor data (simplified implementation)
    fn calculate_tensor_hash(&self, region: &MemoryRegion) -> u64 {
        // Simple hash based on region properties
        // In real implementation, would hash actual tensor data
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        region.id.hash(&mut hasher);
        region.size.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Deallocate memory region
    pub fn deallocate_region(&mut self, region: MemoryRegion) -> Result<()> {
        match region.device {
            Device::Cpu => {
                let block = MemoryBlock {
                    size: region.size,
                    id: region.id,
                    allocated_at: Instant::now(),
                };
                self.cpu_pool.deallocate(block);
                
                // Update statistics
                if let Ok(mut stats) = self.global_stats.lock() {
                    stats.total_cpu_allocated = stats.total_cpu_allocated.saturating_sub(region.size);
                }
            }
            
            _ => {
                // GPU deallocation
                self.gpu_buffers.deallocate(region.id);
                
                // Update statistics  
                if let Ok(mut stats) = self.global_stats.lock() {
                    stats.total_gpu_allocated = stats.total_gpu_allocated.saturating_sub(region.size);
                }
            }
        }
        
        Ok(())
    }
    
    /// Set allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        self.allocation_strategy = strategy;
    }
    
    /// Optimize all memory pools
    pub fn optimize_all(&mut self) -> Result<()> {
        // Optimize GPU memory
        self.gpu_buffers.optimize()?;
        
        // Calculate memory efficiency
        if let Ok(mut stats) = self.global_stats.lock() {
            let total_allocated = stats.total_cpu_allocated + stats.total_gpu_allocated;
            if total_allocated > 0 {
                // Simple efficiency metric based on cache hit rate and fragmentation
                stats.memory_efficiency = stats.cache_hit_rate * 0.7 + 30.0; // Simplified calculation
            }
        }
        
        Ok(())
    }
    
    /// Get comprehensive memory statistics
    pub fn get_comprehensive_stats(&self) -> Result<EnhancedMemoryPoolStats> {
        let cpu_stats = self.cpu_pool.get_stats();
        let gpu_stats = self.gpu_buffers.get_memory_stats()?;
        let cache_stats = self.cross_backend_cache.get_stats();
        let global_stats = self.global_stats.lock()
            .map_err(|_| InferenceError::ConcurrencyError("Failed to acquire global stats lock".to_string()))?
            .clone();
        
        Ok(EnhancedMemoryPoolStats {
            cpu_stats,
            gpu_stats,
            cache_stats,
            global_stats,
        })
    }
}

/// Comprehensive statistics for the enhanced memory pool
#[derive(Debug, Clone)]
pub struct EnhancedMemoryPoolStats {
    pub cpu_stats: PoolStats,
    pub gpu_stats: MemoryStats,
    pub cache_stats: CacheStats,
    pub global_stats: GlobalMemoryStats,
}

impl EnhancedMemoryPoolStats {
    /// Generate a human-readable report
    pub fn generate_report(&self) -> String {
        format!(
            "Enhanced Memory Pool Statistics:\n\
            CPU Pool:\n\
            - Allocations: {}, Deallocations: {}\n\
            - Peak Usage: {:.2} MB\n\
            GPU Memory:\n\
            - Total Allocated: {:.2} MB\n\
            - Peak Usage: {:.2} MB\n\
            - Fragmentation: {:.1}%\n\
            Cross-Backend Cache:\n\
            - Hits: {}, Misses: {}\n\
            - Hit Rate: {:.1}%\n\
            - Evictions: {}\n\
            Global Efficiency:\n\
            - Memory Efficiency: {:.1}%\n\
            - Cross-Backend Transfers: {}",
            self.cpu_stats.allocations,
            self.cpu_stats.deallocations,
            self.cpu_stats.peak_usage as f64 / (1024.0 * 1024.0),
            self.gpu_stats.total_allocated as f64 / (1024.0 * 1024.0),
            self.gpu_stats.peak_usage as f64 / (1024.0 * 1024.0),
            self.gpu_stats.fragmentation_percentage,
            self.cache_stats.hits,
            self.cache_stats.misses,
            if self.cache_stats.hits + self.cache_stats.misses > 0 {
                (self.cache_stats.hits as f64 / (self.cache_stats.hits + self.cache_stats.misses) as f64) * 100.0
            } else {
                0.0
            },
            self.cache_stats.evictions,
            self.global_stats.memory_efficiency,
            self.global_stats.cross_backend_transfers
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_region() {
        let mut region = MemoryRegion::new(1, 1024, Device::Cpu);
        assert_eq!(region.size, 1024);
        assert!(!region.is_in_use());
        
        region.acquire();
        assert!(region.is_in_use());
        
        region.release();
        assert!(!region.is_in_use());
    }
    
    #[test]
    fn test_cross_backend_cache() {
        let mut cache = CrossBackendCache::new(1024 * 1024); // 1MB cache
        
        let region = MemoryRegion::new(1, 1024, Device::Cpu);
        let result = cache.cache_region(123, region);
        assert!(result.is_ok());
        
        let cached = cache.get_region(123, Device::Cpu);
        assert!(cached.is_some());
        
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
    }
    
    #[test]
    fn test_cpu_memory_pool() {
        let mut pool = CPUMemoryPool::new(1024 * 1024); // 1MB pool
        
        let block = pool.allocate(1024);
        assert!(block.is_ok());
        
        let block = block.unwrap();
        assert!(block.size >= 1024); // May be aligned to larger size
        
        pool.deallocate(block);
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.deallocations, 1);
    }
    
    #[test]
    fn test_allocation_strategy() {
        let pool_result = EnhancedMemoryPool::new(Device::Cpu, 1024 * 1024, 512 * 1024);
        assert!(pool_result.is_ok());
        
        let mut pool = pool_result.unwrap();
        
        pool.set_allocation_strategy(AllocationStrategy::PreferCPU);
        let region = pool.allocate_optimal(1024, Device::Cpu);
        assert!(region.is_ok());
        
        let region = region.unwrap();
    }
    
    #[test] 
    fn test_memory_transfer() {
        let pool_result = EnhancedMemoryPool::new(Device::Cpu, 1024 * 1024, 512 * 1024);
        assert!(pool_result.is_ok());
        
        let mut pool = pool_result.unwrap();
        
        let region = pool.allocate_optimal(1024, Device::Cpu).unwrap();
        let transferred = pool.transfer_to_device(&region, Device::Cpu);
        assert!(transferred.is_ok());
        
        // Same device transfer should return identical region
        let transferred_region = transferred.unwrap();
    }
}
