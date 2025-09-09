//! Specialized tensor memory pool for efficient tensor memory management
//!
//! This module provides a specialized memory pool designed specifically for tensor
//! allocations, with optimizations for common tensor patterns, efficient lifecycle
//! tracking, and memory pressure management.

use super::{MemoryError, MemoryHandle, MemoryResult, HybridMemoryPool};
use crate::memory::tracking::{OptimizedMemoryTracker, MemoryPressureDetector, MemoryPressureLevel, TrackingConfig};
use candle_core::Device;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Tensor size categories for specialized allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorSizeCategory {
    /// Very small tensors (< 4KB) - typically scalars, small vectors
    VerySmall,
    /// Small tensors (4KB - 64KB) - small matrices, embeddings
    Small,
    /// Medium tensors (64KB - 1MB) - standard layer weights
    Medium,
    /// Large tensors (1MB - 16MB) - large weight matrices
    Large,
    /// Very large tensors (> 16MB) - model parameters, activations
    VeryLarge,
}

impl TensorSizeCategory {
    /// Determine tensor size category from byte size
    pub fn from_size(size_bytes: usize) -> Self {
        match size_bytes {
            0..=4096 => Self::VerySmall,
            4097..=65536 => Self::Small,
            65537..=1048576 => Self::Medium,
            1048577..=16777216 => Self::Large,
            _ => Self::VeryLarge,
        }
    }

    /// Get the target pool size for this category
    pub fn target_pool_size(&self) -> usize {
        match self {
            Self::VerySmall => 1024 * 1024,      // 1MB pool for very small tensors
            Self::Small => 8 * 1024 * 1024,      // 8MB pool for small tensors
            Self::Medium => 32 * 1024 * 1024,    // 32MB pool for medium tensors
            Self::Large => 128 * 1024 * 1024,    // 128MB pool for large tensors
            Self::VeryLarge => 512 * 1024 * 1024, // 512MB pool for very large tensors
        }
    }

    /// Get the allocation alignment for this category
    pub fn alignment(&self) -> usize {
        match self {
            Self::VerySmall => 8,   // 8-byte alignment for scalars
            Self::Small => 16,      // 16-byte alignment for small tensors
            Self::Medium => 32,     // 32-byte alignment for medium tensors
            Self::Large => 64,      // 64-byte alignment for large tensors
            Self::VeryLarge => 256, // 256-byte alignment for very large tensors
        }
    }
}

/// Optimized metadata for tensor lifecycle tracking
#[derive(Debug, Clone)]
pub struct TensorLifecycleMetadata {
    /// Unique tensor ID
    pub tensor_id: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Size category
    pub size_category: TensorSizeCategory,
    /// Device placement
    pub device_type: String,
    /// Memory size in bytes
    pub size_bytes: usize,
    /// Whether this tensor is part of a model
    pub is_model_weight: bool,
    /// Whether this tensor is temporary (e.g., intermediate activations)
    pub is_temporary: bool,
    /// Reference count
    pub ref_count: u32,
}

impl TensorLifecycleMetadata {
    /// Create new tensor metadata
    pub fn new(
        tensor_id: u64,
        size_bytes: usize,
        device: &Device,
        is_model_weight: bool,
        is_temporary: bool,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            tensor_id,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_category: TensorSizeCategory::from_size(size_bytes),
            device_type: format!("{:?}", device),
            size_bytes,
            is_model_weight,
            is_temporary,
            ref_count: 1,
        }
    }

    /// Record access to this tensor
    pub fn record_access(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }

    /// Check if this tensor is stale (not accessed recently)
    pub fn is_stale(&self, threshold: Duration) -> bool {
        if let Ok(elapsed) = self.last_accessed.elapsed() {
            elapsed > threshold && !self.is_model_weight
        } else {
            false
        }
    }

    /// Get age of this tensor
    pub fn age(&self) -> Option<Duration> {
        self.created_at.elapsed().ok()
    }
}

/// Specialized tensor memory pool with lifecycle optimization
#[derive(Debug)]
pub struct TensorMemoryPool {
    /// Underlying hybrid memory pool
    pool: Arc<HybridMemoryPool>,
    /// Specialized pools for different tensor size categories
    category_pools: RwLock<HashMap<TensorSizeCategory, Arc<Mutex<CategoryPool>>>>,
    /// Tensor lifecycle metadata tracking
    tensor_metadata: Arc<RwLock<HashMap<u64, TensorLifecycleMetadata>>>,
    /// Memory pressure detector for tensor-specific pressure management
    pressure_detector: Arc<MemoryPressureDetector>,
    /// Optimized tracker for performance metrics
    tracker: Option<Arc<OptimizedMemoryTracker>>,
    /// LRU cache for efficient deallocation
    lru_cache: Arc<Mutex<LruCache>>,
    /// Configuration
    config: TensorPoolConfig,
    /// Statistics
    stats: Arc<RwLock<TensorPoolStats>>,
}

/// Category-specific memory pool
#[derive(Debug)]
struct CategoryPool {
    /// Available memory blocks for reuse
    available_blocks: VecDeque<MemoryHandle>,
    /// Total allocated memory for this category
    total_allocated: usize,
    /// Peak allocated memory for this category
    peak_allocated: usize,
    /// Number of allocations
    allocation_count: u64,
    /// Number of reused blocks
    reuse_count: u64,
}

/// LRU cache for tensor deallocation optimization
#[derive(Debug)]
struct LruCache {
    /// Tensor IDs ordered by access time (most recent first)
    access_order: VecDeque<u64>,
    /// Maximum number of tensors to track
    max_size: usize,
}

/// Configuration for tensor memory pool
#[derive(Debug, Clone)]
pub struct TensorPoolConfig {
    /// Enable category-specific pooling
    pub enable_category_pooling: bool,
    /// Enable tensor lifecycle tracking
    pub enable_lifecycle_tracking: bool,
    /// Enable memory pressure management
    pub enable_pressure_management: bool,
    /// Stale tensor threshold (duration before tensor is considered stale)
    pub stale_threshold: Duration,
    /// LRU cache size
    pub lru_cache_size: usize,
    /// Memory pressure cleanup threshold
    pub pressure_cleanup_threshold: f64,
}

impl Default for TensorPoolConfig {
    fn default() -> Self {
        Self {
            enable_category_pooling: true,
            enable_lifecycle_tracking: true,
            enable_pressure_management: true,
            stale_threshold: Duration::from_secs(300), // 5 minutes
            lru_cache_size: 1000,
            pressure_cleanup_threshold: 0.8, // 80% memory usage
        }
    }
}

/// Statistics for tensor memory pool
#[derive(Debug, Clone, Default)]
pub struct TensorPoolStats {
    /// Total tensors allocated
    pub total_tensors_allocated: u64,
    /// Total tensors deallocated
    pub total_tensors_deallocated: u64,
    /// Current active tensors
    pub active_tensors: u64,
    /// Total memory allocated for tensors
    pub total_memory_allocated: usize,
    /// Current memory in use
    pub current_memory_used: usize,
    /// Peak memory usage
    pub peak_memory_used: usize,
    /// Number of successful reuses
    pub successful_reuses: u64,
    /// Number of pressure-induced cleanups
    pub pressure_cleanups: u64,
    /// Category-specific statistics
    pub category_stats: HashMap<TensorSizeCategory, CategoryStats>,
}

/// Statistics for each tensor size category
#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    /// Number of tensors in this category
    pub tensor_count: u64,
    /// Total memory for this category
    pub total_memory: usize,
    /// Average tensor size
    pub average_size: usize,
    /// Pool reuse rate
    pub reuse_rate: f64,
}

impl LruCache {
    fn new(max_size: usize) -> Self {
        Self {
            access_order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn record_access(&mut self, tensor_id: u64) {
        // Remove if already exists
        if let Some(pos) = self.access_order.iter().position(|&id| id == tensor_id) {
            self.access_order.remove(pos);
        }

        // Add to front (most recent)
        self.access_order.push_front(tensor_id);

        // Maintain size limit
        if self.access_order.len() > self.max_size {
            self.access_order.pop_back();
        }
    }

    fn get_least_recently_used(&self, count: usize) -> Vec<u64> {
        self.access_order
            .iter()
            .rev()
            .take(count)
            .copied()
            .collect()
    }
}

impl CategoryPool {
    fn new() -> Self {
        Self {
            available_blocks: VecDeque::new(),
            total_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            reuse_count: 0,
        }
    }

    fn try_reuse(&mut self, size_bytes: usize) -> Option<MemoryHandle> {
        // Look for a suitable block to reuse
        if let Some(pos) = self.available_blocks.iter().position(|handle| {
            handle.size() >= size_bytes && handle.size() <= size_bytes * 2
        }) {
            let handle = self.available_blocks.remove(pos).unwrap();
            self.reuse_count += 1;
            Some(handle)
        } else {
            None
        }
    }

    fn return_block(&mut self, handle: MemoryHandle) {
        self.available_blocks.push_back(handle);
    }

    fn record_allocation(&mut self, size_bytes: usize) {
        self.allocation_count += 1;
        self.total_allocated += size_bytes;
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }
    }

    fn record_deallocation(&mut self, size_bytes: usize) {
        self.total_allocated = self.total_allocated.saturating_sub(size_bytes);
    }
}

impl TensorMemoryPool {
    /// Create a new tensor memory pool
    pub fn new(pool: Arc<HybridMemoryPool>) -> MemoryResult<Self> {
        Self::with_config(pool, TensorPoolConfig::default())
    }

    /// Create a new tensor memory pool with custom configuration
    pub fn with_config(
        pool: Arc<HybridMemoryPool>,
        config: TensorPoolConfig,
    ) -> MemoryResult<Self> {
        let pressure_detector = Arc::new(MemoryPressureDetector::new(
            config.pressure_cleanup_threshold,
        ));

        let tracker = if config.enable_lifecycle_tracking {
            let tracking_config = TrackingConfig::standard();
            match OptimizedMemoryTracker::new(tracking_config) {
                Ok(tracker) => Some(Arc::new(tracker)),
                Err(_) => None, // Fallback to no tracking on error
            }
        } else {
            None
        };

        Ok(Self {
            pool,
            category_pools: RwLock::new(HashMap::new()),
            tensor_metadata: Arc::new(RwLock::new(HashMap::new())),
            pressure_detector,
            tracker,
            lru_cache: Arc::new(Mutex::new(LruCache::new(config.lru_cache_size))),
            config,
            stats: Arc::new(RwLock::new(TensorPoolStats::default())),
        })
    }

    /// Allocate memory for a tensor with optimization
    pub fn allocate_tensor(
        &self,
        tensor_id: u64,
        size_bytes: usize,
        device: &Device,
        is_model_weight: bool,
        is_temporary: bool,
    ) -> MemoryResult<MemoryHandle> {
        let category = TensorSizeCategory::from_size(size_bytes);
        let alignment = category.alignment();

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating tensor {} with size {} bytes (category: {:?})",
            tensor_id, size_bytes, category
        );

        // Try to reuse from category pool if enabled
        if self.config.enable_category_pooling {
            if let Some(handle) = self.try_reuse_from_category(category, size_bytes)? {
                #[cfg(feature = "tracing")]
                debug!("Reused memory block for tensor {}", tensor_id);
                
                // Update metadata and return
                self.track_tensor_allocation(tensor_id, size_bytes, device, is_model_weight, is_temporary)?;
                return Ok(handle);
            }
        }

        // Check memory pressure before allocation
        if self.config.enable_pressure_management {
            self.handle_memory_pressure()?;
        }

        // Allocate new memory from underlying pool
        let handle = self.pool.allocate(size_bytes, alignment, device)?;

        // Track allocation
        self.track_tensor_allocation(tensor_id, size_bytes, device, is_model_weight, is_temporary)?;
        self.update_category_stats(category, size_bytes, true)?;

        #[cfg(feature = "tracing")]
        debug!("Allocated new memory block for tensor {} (handle: {})", tensor_id, handle.id());

        Ok(handle)
    }

    /// Deallocate tensor memory with optimization
    pub fn deallocate_tensor(
        &self,
        tensor_id: u64,
        handle: MemoryHandle,
    ) -> MemoryResult<()> {
        let size_bytes = handle.size();
        let category = TensorSizeCategory::from_size(size_bytes);

        #[cfg(feature = "tracing")]
        debug!("Deallocating tensor {} (handle: {}, size: {} bytes)", tensor_id, handle.id(), size_bytes);

        // Remove from tracking
        self.untrack_tensor(tensor_id)?;

        // Try to return to category pool for reuse
        if self.config.enable_category_pooling && !self.is_memory_pressure_high() {
            if self.return_to_category_pool(category, handle.clone())? {
                #[cfg(feature = "tracing")]
                debug!("Returned tensor {} memory to category pool for reuse", tensor_id);
                return Ok(());
            }
        }

        // Deallocate from underlying pool
        self.pool.deallocate(handle)?;
        self.update_category_stats(category, size_bytes, false)?;

        #[cfg(feature = "tracing")]
        debug!("Deallocated tensor {} memory from underlying pool", tensor_id);

        Ok(())
    }

    /// Record tensor access for LRU tracking
    pub fn record_tensor_access(&self, tensor_id: u64) -> MemoryResult<()> {
        if !self.config.enable_lifecycle_tracking {
            return Ok(());
        }

        // Update metadata
        if let Ok(mut metadata_map) = self.tensor_metadata.write() {
            if let Some(metadata) = metadata_map.get_mut(&tensor_id) {
                metadata.record_access();
            }
        }

        // Update LRU cache
        if let Ok(mut lru) = self.lru_cache.lock() {
            lru.record_access(tensor_id);
        }

        Ok(())
    }

    /// Get tensor memory pressure level
    pub fn get_memory_pressure(&self) -> MemoryPressureLevel {
        if self.config.enable_pressure_management {
            self.pressure_detector.detect_pressure(
                self.get_current_memory_usage(),
                self.get_total_memory_capacity(),
            )
        } else {
            MemoryPressureLevel::Low
        }
    }

    /// Force cleanup of stale tensors
    pub fn cleanup_stale_tensors(&self) -> MemoryResult<usize> {
        if !self.config.enable_lifecycle_tracking {
            return Ok(0);
        }

        let stale_tensors = self.identify_stale_tensors()?;
        let cleanup_count = stale_tensors.len();

        #[cfg(feature = "tracing")]
        if cleanup_count > 0 {
            info!("Cleaning up {} stale tensors", cleanup_count);
        }

        for tensor_id in stale_tensors {
            // Note: In practice, you'd need to coordinate with the tensor system
            // to actually deallocate these tensors. This is a marker for cleanup.
            self.mark_tensor_for_cleanup(tensor_id)?;
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.pressure_cleanups += 1;
        }

        Ok(cleanup_count)
    }

    /// Get comprehensive tensor pool statistics
    pub fn get_tensor_pool_stats(&self) -> MemoryResult<TensorPoolStats> {
        if let Ok(stats) = self.stats.read() {
            Ok(stats.clone())
        } else {
            Err(MemoryError::InternalError {
                reason: "Failed to acquire stats lock".to_string(),
            })
        }
    }

    // Private helper methods

    fn try_reuse_from_category(
        &self,
        category: TensorSizeCategory,
        size_bytes: usize,
    ) -> MemoryResult<Option<MemoryHandle>> {
        if let Ok(pools) = self.category_pools.read() {
            if let Some(pool) = pools.get(&category) {
                if let Ok(mut pool) = pool.lock() {
                    return Ok(pool.try_reuse(size_bytes));
                }
            }
        }
        Ok(None)
    }

    fn return_to_category_pool(
        &self,
        category: TensorSizeCategory,
        handle: MemoryHandle,
    ) -> MemoryResult<bool> {
        let mut pools = self.category_pools.write().map_err(|_| MemoryError::InternalError {
            reason: "Failed to acquire category pools lock".to_string(),
        })?;

        let pool = pools
            .entry(category)
            .or_insert_with(|| Arc::new(Mutex::new(CategoryPool::new())));

        if let Ok(mut pool) = pool.lock() {
            pool.return_block(handle);
            return Ok(true);
        }

        Ok(false)
    }

    fn track_tensor_allocation(
        &self,
        tensor_id: u64,
        size_bytes: usize,
        device: &Device,
        is_model_weight: bool,
        is_temporary: bool,
    ) -> MemoryResult<()> {
        if !self.config.enable_lifecycle_tracking {
            return Ok(());
        }

        let metadata = TensorLifecycleMetadata::new(
            tensor_id,
            size_bytes,
            device,
            is_model_weight,
            is_temporary,
        );

        if let Ok(mut metadata_map) = self.tensor_metadata.write() {
            metadata_map.insert(tensor_id, metadata);
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.total_tensors_allocated += 1;
            stats.active_tensors += 1;
            stats.total_memory_allocated += size_bytes;
            stats.current_memory_used += size_bytes;
            if stats.current_memory_used > stats.peak_memory_used {
                stats.peak_memory_used = stats.current_memory_used;
            }
        }

        Ok(())
    }

    fn untrack_tensor(&self, tensor_id: u64) -> MemoryResult<()> {
        if !self.config.enable_lifecycle_tracking {
            return Ok(());
        }

        let size_bytes = if let Ok(mut metadata_map) = self.tensor_metadata.write() {
            metadata_map.remove(&tensor_id).map(|m| m.size_bytes).unwrap_or(0)
        } else {
            0
        };

        if let Ok(mut stats) = self.stats.write() {
            stats.total_tensors_deallocated += 1;
            stats.active_tensors = stats.active_tensors.saturating_sub(1);
            stats.current_memory_used = stats.current_memory_used.saturating_sub(size_bytes);
        }

        Ok(())
    }

    fn update_category_stats(
        &self,
        category: TensorSizeCategory,
        size_bytes: usize,
        is_allocation: bool,
    ) -> MemoryResult<()> {
        if let Ok(mut stats) = self.stats.write() {
            let category_stats = stats.category_stats.entry(category).or_default();
            
            if is_allocation {
                category_stats.tensor_count += 1;
                category_stats.total_memory += size_bytes;
            } else {
                category_stats.tensor_count = category_stats.tensor_count.saturating_sub(1);
                category_stats.total_memory = category_stats.total_memory.saturating_sub(size_bytes);
            }

            // Update average size
            if category_stats.tensor_count > 0 {
                category_stats.average_size = category_stats.total_memory / category_stats.tensor_count as usize;
            }
        }

        Ok(())
    }

    fn handle_memory_pressure(&self) -> MemoryResult<()> {
        let pressure_level = self.get_memory_pressure();
        
        match pressure_level {
            MemoryPressureLevel::High | MemoryPressureLevel::Critical => {
                #[cfg(feature = "tracing")]
                warn!("High memory pressure detected, initiating cleanup");
                self.cleanup_stale_tensors()?;
            }
            _ => {}
        }

        Ok(())
    }

    fn identify_stale_tensors(&self) -> MemoryResult<Vec<u64>> {
        if let Ok(metadata_map) = self.tensor_metadata.read() {
            let stale_tensors: Vec<u64> = metadata_map
                .iter()
                .filter(|(_, metadata)| metadata.is_stale(self.config.stale_threshold))
                .map(|(&tensor_id, _)| tensor_id)
                .collect();
            Ok(stale_tensors)
        } else {
            Ok(Vec::new())
        }
    }

    fn mark_tensor_for_cleanup(&self, tensor_id: u64) -> MemoryResult<()> {
        // In a real implementation, this would coordinate with the tensor system
        // to mark tensors for cleanup. For now, we just track it.
        #[cfg(feature = "tracing")]
        debug!("Marked tensor {} for cleanup", tensor_id);
        Ok(())
    }

    fn is_memory_pressure_high(&self) -> bool {
        matches!(
            self.get_memory_pressure(),
            MemoryPressureLevel::High | MemoryPressureLevel::Critical
        )
    }

    fn get_current_memory_usage(&self) -> usize {
        if let Ok(stats) = self.stats.read() {
            stats.current_memory_used
        } else {
            0
        }
    }

    fn get_total_memory_capacity(&self) -> usize {
        // This would typically come from the underlying memory pool
        // For now, we'll use a reasonable estimate
        1024 * 1024 * 1024 // 1GB
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_tensor_size_category() {
        assert_eq!(TensorSizeCategory::from_size(1024), TensorSizeCategory::VerySmall);
        assert_eq!(TensorSizeCategory::from_size(32768), TensorSizeCategory::Small);
        assert_eq!(TensorSizeCategory::from_size(512 * 1024), TensorSizeCategory::Medium);
        assert_eq!(TensorSizeCategory::from_size(8 * 1024 * 1024), TensorSizeCategory::Large);
        assert_eq!(TensorSizeCategory::from_size(32 * 1024 * 1024), TensorSizeCategory::VeryLarge);
    }

    #[test]
    fn test_tensor_lifecycle_metadata() {
        let device = get_cpu_device();
        let mut metadata = TensorLifecycleMetadata::new(1, 1024, &device, false, true);
        
        assert_eq!(metadata.tensor_id, 1);
        assert_eq!(metadata.size_bytes, 1024);
        assert_eq!(metadata.access_count, 0);
        assert_eq!(metadata.ref_count, 1);
        
        metadata.record_access();
        assert_eq!(metadata.access_count, 1);
    }

    #[test]
    fn test_lru_cache() {
        let mut lru = LruCache::new(3);
        
        lru.record_access(1);
        lru.record_access(2);
        lru.record_access(3);
        
        assert_eq!(lru.access_order, vec![3, 2, 1]);
        
        // Access 1 again - should move to front
        lru.record_access(1);
        assert_eq!(lru.access_order, vec![1, 3, 2]);
        
        // Add 4 - should evict 2
        lru.record_access(4);
        assert_eq!(lru.access_order, vec![4, 1, 3]);
    }
}
