//! Memory Pool System for BitNet
//!
//! This module provides a hybrid memory pool system optimized for tensor operations
//! in BitNet. It implements efficient allocation strategies for both small and large
//! memory blocks, with device-specific optimizations for CPU and Metal GPU memory.
//!
//! # Architecture
//!
//! The memory pool system consists of several key components:
//!
//! - **HybridMemoryPool**: Main interface that routes allocations to appropriate pools
//! - **SmallBlockPool**: Optimized for allocations < 1MB using fixed-size blocks
//! - **LargeBlockPool**: Handles allocations >= 1MB using buddy allocation
//! - **Device-specific pools**: Separate pools for CPU and Metal GPU memory
//!
//! # Features
//!
//! - Thread-safe allocation and deallocation
//! - Memory usage tracking and metrics
//! - Device-aware memory management
//! - Efficient small block allocation
//! - Buddy allocation for large blocks
//! - Feature-gated Metal GPU support
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::memory::HybridMemoryPool;
//! use bitnet_core::device::auto_select_device;
//!
//! // Create a memory pool
//! let pool = HybridMemoryPool::new()?;
//! let device = auto_select_device();
//!
//! // Allocate memory
//! let handle = pool.allocate(1024, 16, &device)?;
//!
//! // Get memory metrics
//! let metrics = pool.get_metrics();
//! println!("Total allocated: {} bytes", metrics.total_allocated);
//!
//! // Deallocate memory
//! pool.deallocate(handle)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

// Sub-modules
pub mod cleanup;
pub mod conversion;
pub mod device_pool;
pub mod handle;
pub mod large_block;
pub mod metrics;
pub mod small_block;
pub mod tensor;
pub mod tracking;

// Re-exports
pub use cleanup::{
    CleanupConfig, CleanupId, CleanupManager, CleanupMetrics, CleanupOperationMetrics,
    CleanupPriority, CleanupResult, CleanupScheduler, CleanupStrategy, CleanupStrategyType,
    CompactionResult, CpuCleanup, DeviceCleanupOps, MetalCleanup,
};
pub use conversion::{
    BatchConverter, ConversionConfig, ConversionEngine, ConversionEvent, ConversionMetrics,
    ConversionPipeline, ConversionStats, InPlaceConverter, StreamingConverter, ZeroCopyConverter,
};
pub use device_pool::CpuMemoryPool;
pub use handle::MemoryHandle;
pub use large_block::LargeBlockPool;
pub use metrics::MemoryMetrics;
pub use small_block::SmallBlockPool;
pub use tensor::{BitNetDType, BitNetTensor, TensorHandle, TensorMetadata};
pub use tracking::{
    AllocationTimeline, DetailedMemoryMetrics, LeakReport, MemoryPressureDetector,
    MemoryPressureLevel, MemoryProfiler, MemoryTracker, PatternAnalyzer, PressureCallback,
    PressureThresholds, ProfilingReport, TrackingConfig, TrackingLevel,
};

#[cfg(feature = "metal")]
pub use device_pool::MetalMemoryPool;

/// Errors that can occur during memory pool operations
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Allocation failed due to insufficient memory
    #[error("Allocation failed: insufficient memory for {size} bytes")]
    InsufficientMemory { size: usize },

    /// Invalid alignment specified
    #[error("Invalid alignment: {alignment} (must be power of 2)")]
    InvalidAlignment { alignment: usize },

    /// Memory handle is invalid or already deallocated
    #[error("Invalid memory handle: {reason}")]
    InvalidHandle { reason: String },

    /// Device-specific allocation error
    #[error("Device allocation failed: {device_type} - {reason}")]
    DeviceAllocationFailed { device_type: String, reason: String },

    /// Memory pool is corrupted or in invalid state
    #[error("Memory pool corruption detected: {reason}")]
    PoolCorruption { reason: String },

    /// Operation not supported for the given device
    #[error("Operation not supported for device: {device_type}")]
    UnsupportedDevice { device_type: String },

    /// Metal-specific errors
    #[error("Metal memory error: {reason}")]
    MetalError { reason: String },

    /// Internal error in memory management
    #[error("Internal memory management error: {reason}")]
    InternalError { reason: String },
}

/// Result type for memory operations
pub type MemoryResult<T> = std::result::Result<T, MemoryError>;

/// Configuration for the hybrid memory pool
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryPoolConfig {
    /// Threshold for small vs large block allocation (default: 1MB)
    pub small_block_threshold: usize,
    /// Initial size for small block pools (default: 16MB)
    pub small_pool_initial_size: usize,
    /// Maximum size for small block pools (default: 256MB)
    pub small_pool_max_size: usize,
    /// Initial size for large block pools (default: 64MB)
    pub large_pool_initial_size: usize,
    /// Maximum size for large block pools (default: 1GB)
    pub large_pool_max_size: usize,
    /// Enable memory usage tracking (default: true)
    pub enable_metrics: bool,
    /// Enable debug logging (default: false)
    pub enable_debug_logging: bool,
    /// Enable advanced memory tracking (default: false)
    pub enable_advanced_tracking: bool,
    /// Configuration for advanced tracking (if enabled)
    pub tracking_config: Option<tracking::TrackingConfig>,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            small_block_threshold: 1024 * 1024,        // 1MB
            small_pool_initial_size: 16 * 1024 * 1024, // 16MB
            small_pool_max_size: 256 * 1024 * 1024,    // 256MB
            large_pool_initial_size: 64 * 1024 * 1024, // 64MB
            large_pool_max_size: 1024 * 1024 * 1024,   // 1GB
            enable_metrics: true,
            enable_debug_logging: false,
            enable_advanced_tracking: false,
            tracking_config: None,
        }
    }
}

/// Main hybrid memory pool that manages both small and large allocations
/// across different device types (CPU and Metal GPU)
#[derive(Debug)]
#[allow(dead_code)]
pub struct HybridMemoryPool {
    /// Configuration for the memory pool
    config: MemoryPoolConfig,
    /// Small block pools per device
    small_pools: RwLock<HashMap<DeviceKey, Arc<Mutex<SmallBlockPool>>>>,
    /// Large block pools per device
    large_pools: RwLock<HashMap<DeviceKey, Arc<Mutex<LargeBlockPool>>>>,
    /// Global memory metrics
    metrics: Arc<RwLock<MemoryMetrics>>,
    /// Handle registry for tracking active allocations
    handle_registry: Arc<RwLock<HashMap<u64, MemoryHandle>>>,
    /// Next handle ID
    next_handle_id: Arc<Mutex<u64>>,
    /// Advanced memory tracker (optional)
    memory_tracker: Option<Arc<tracking::MemoryTracker>>,
}

/// Key for identifying device-specific pools
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DeviceKey {
    Cpu,
    Metal(String), // Metal device ID as string
    Cuda(String),  // CUDA device ID as string
}

impl From<&Device> for DeviceKey {
    fn from(device: &Device) -> Self {
        match device {
            Device::Cpu => DeviceKey::Cpu,
            Device::Metal(metal_device) => DeviceKey::Metal(format!("{:?}", metal_device.id())),
            Device::Cuda(cuda_device) => DeviceKey::Cuda(format!("{:?}", cuda_device)),
        }
    }
}

impl HybridMemoryPool {
    /// Creates a new hybrid memory pool with default configuration
    ///
    /// # Returns
    ///
    /// A Result containing the new memory pool or an error if initialization fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::HybridMemoryPool;
    ///
    /// let pool = HybridMemoryPool::new()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new() -> MemoryResult<Self> {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Creates a new hybrid memory pool with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the memory pool
    ///
    /// # Returns
    ///
    /// A Result containing the new memory pool or an error if initialization fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig};
    ///
    /// let config = MemoryPoolConfig {
    ///     small_block_threshold: 512 * 1024, // 512KB
    ///     enable_debug_logging: true,
    ///     ..Default::default()
    /// };
    /// let pool = HybridMemoryPool::with_config(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_config(config: MemoryPoolConfig) -> MemoryResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Creating hybrid memory pool with config: {:?}", config);

        // Initialize memory tracker if advanced tracking is enabled
        let memory_tracker = if config.enable_advanced_tracking {
            let tracking_config = config
                .tracking_config
                .clone()
                .unwrap_or_else(|| tracking::TrackingConfig::standard());

            match tracking::MemoryTracker::new(tracking_config) {
                Ok(tracker) => Some(Arc::new(tracker)),
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    warn!("Failed to create memory tracker: {}", _e);
                    None
                }
            }
        } else {
            None
        };

        let pool = Self {
            config,
            small_pools: RwLock::new(HashMap::new()),
            large_pools: RwLock::new(HashMap::new()),
            metrics: Arc::new(RwLock::new(MemoryMetrics::new())),
            handle_registry: Arc::new(RwLock::new(HashMap::new())),
            next_handle_id: Arc::new(Mutex::new(1)),
            memory_tracker,
        };

        #[cfg(feature = "tracing")]
        info!("Hybrid memory pool created successfully");

        Ok(pool)
    }

    /// Allocates memory of the specified size and alignment on the given device
    ///
    /// # Arguments
    ///
    /// * `size` - Size of memory to allocate in bytes
    /// * `alignment` - Required alignment (must be power of 2)
    /// * `device` - Target device for allocation
    ///
    /// # Returns
    ///
    /// A Result containing a MemoryHandle or an error if allocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    ///
    /// let pool = HybridMemoryPool::new()?;
    /// let device = get_cpu_device();
    /// let handle = pool.allocate(1024, 16, &device)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn allocate(
        &self,
        size: usize,
        alignment: usize,
        device: &Device,
    ) -> MemoryResult<MemoryHandle> {
        // Handle zero-size allocations
        if size == 0 {
            return Err(MemoryError::InsufficientMemory { 
                size: 0,
            });
        }

        // Validate alignment
        if !alignment.is_power_of_two() || alignment == 0 {
            return Err(MemoryError::InvalidAlignment { alignment });
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Allocating {} bytes with alignment {} on device {:?}",
            size, alignment, device
        );

        let device_key = DeviceKey::from(device);

        // Choose allocation strategy based on size
        let handle = if size < self.config.small_block_threshold {
            self.allocate_small(size, alignment, &device_key, device)?
        } else {
            self.allocate_large(size, alignment, &device_key, device)?
        };

        // Optimize critical section - batch all operations together
        let handle_id = handle.id();

        // Single critical section for all registry and metrics updates
        {
            let mut registry =
                self.handle_registry
                    .write()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire handle registry lock".to_string(),
                    })?;
            registry.insert(handle_id, handle.clone());

            // Update metrics in same critical section if enabled
            if self.config.enable_metrics {
                if let Ok(mut metrics) = self.metrics.write() {
                    metrics.record_allocation(size);
                }
            }
        }

        // Track allocation if advanced tracking is enabled (outside critical section)
        if let Some(ref tracker) = self.memory_tracker {
            tracker.track_allocation(&handle, size, device);
        }

        #[cfg(feature = "tracing")]
        debug!("Successfully allocated memory with handle ID {}", handle_id);

        Ok(handle)
    }

    /// Deallocates memory associated with the given handle
    ///
    /// # Arguments
    ///
    /// * `handle` - Memory handle to deallocate
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    ///
    /// let pool = HybridMemoryPool::new()?;
    /// let device = get_cpu_device();
    /// let handle = pool.allocate(1024, 16, &device)?;
    /// pool.deallocate(handle)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn deallocate(&self, handle: MemoryHandle) -> MemoryResult<()> {
        let handle_id = handle.id();
        let size = handle.size();

        #[cfg(feature = "tracing")]
        debug!(
            "Deallocating memory with handle ID {} (size: {} bytes)",
            handle_id, size
        );

        // Remove from registry
        let registered_handle = {
            let mut registry =
                self.handle_registry
                    .write()
                    .map_err(|_| MemoryError::InternalError {
                        reason: "Failed to acquire handle registry lock".to_string(),
                    })?;

            #[cfg(feature = "tracing")]
            {
                let handle_found = registry.contains_key(&handle_id);
                debug!("Handle {} found in registry: {}", handle_id, handle_found);
            }

            registry.remove(&handle_id)
        };

        let registered_handle = registered_handle.ok_or_else(|| {
            #[cfg(feature = "tracing")]
            warn!(
                "Handle {} not found in registry during deallocation",
                handle_id
            );
            MemoryError::InvalidHandle {
                reason: format!("Handle {} not found in registry", handle_id),
            }
        })?;

        // Verify handle matches
        if registered_handle.id() != handle.id() {
            #[cfg(feature = "tracing")]
            error!(
                "Handle ID mismatch during deallocation: expected {}, got {}",
                registered_handle.id(),
                handle.id()
            );
            return Err(MemoryError::InvalidHandle {
                reason: "Handle ID mismatch".to_string(),
            });
        }

        let device_key = DeviceKey::from(&handle.device());

        #[cfg(feature = "tracing")]
        debug!(
            "Deallocating handle {} from device {:?} (size: {})",
            handle_id, device_key, size
        );

        // Deallocate based on size
        if size < self.config.small_block_threshold {
            #[cfg(feature = "tracing")]
            debug!(
                "Using small block pool for handle {} deallocation",
                handle_id
            );
            self.deallocate_small(handle, &device_key)?;
        } else {
            #[cfg(feature = "tracing")]
            debug!(
                "Using large block pool for handle {} deallocation",
                handle_id
            );
            self.deallocate_large(handle, &device_key)?;
        }

        // Track deallocation if advanced tracking is enabled (before moving handle)
        if let Some(ref tracker) = self.memory_tracker {
            tracker.track_deallocation(&registered_handle);
        }

        // Update metrics in single operation
        if self.config.enable_metrics {
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.record_deallocation(size);
                #[cfg(feature = "tracing")]
                debug!(
                    "Updated metrics for handle {} deallocation (size: {})",
                    handle_id, size
                );
            }
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Successfully deallocated memory with handle ID {} (size: {} bytes)",
            handle_id, size
        );

        Ok(())
    }

    /// Returns current memory usage metrics
    ///
    /// # Returns
    ///
    /// A copy of the current memory metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::HybridMemoryPool;
    ///
    /// let pool = HybridMemoryPool::new()?;
    /// let metrics = pool.get_metrics();
    /// println!("Total allocated: {} bytes", metrics.total_allocated);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_metrics(&self) -> MemoryMetrics {
        self.metrics
            .read()
            .map(|metrics| metrics.clone())
            .unwrap_or_else(|_| MemoryMetrics::new()) // Fixed closure signature
    }

    /// Returns detailed memory tracking metrics if advanced tracking is enabled
    ///
    /// # Returns
    ///
    /// Detailed memory metrics or None if tracking is disabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
    ///
    /// let mut config = MemoryPoolConfig::default();
    /// config.enable_advanced_tracking = true;
    /// config.tracking_config = Some(TrackingConfig::standard());
    ///
    /// let pool = HybridMemoryPool::with_config(config)?;
    /// if let Some(detailed_metrics) = pool.get_detailed_metrics() {
    ///     println!("Memory pressure: {:?}", detailed_metrics.pressure_level);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_detailed_metrics(&self) -> Option<tracking::DetailedMemoryMetrics> {
        self.memory_tracker
            .as_ref()
            .map(|tracker| tracker.get_detailed_metrics())
    }

    /// Returns the memory tracker if advanced tracking is enabled
    ///
    /// # Returns
    ///
    /// Reference to the memory tracker or None if tracking is disabled
    pub fn get_memory_tracker(&self) -> Option<&Arc<tracking::MemoryTracker>> {
        self.memory_tracker.as_ref()
    }

    /// Registers a memory pressure callback if advanced tracking is enabled
    ///
    /// # Arguments
    ///
    /// * `callback` - Callback function to be called on pressure events
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig, MemoryPressureLevel};
    ///
    /// let mut config = MemoryPoolConfig::default();
    /// config.enable_advanced_tracking = true;
    /// config.tracking_config = Some(TrackingConfig::standard());
    ///
    /// let pool = HybridMemoryPool::with_config(config)?;
    /// pool.register_pressure_callback(Box::new(|level| {
    ///     match level {
    ///         MemoryPressureLevel::Critical => eprintln!("CRITICAL: Memory pressure!"),
    ///         _ => println!("Memory pressure: {:?}", level),
    ///     }
    /// }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register_pressure_callback(&self, callback: tracking::PressureCallback) {
        if let Some(ref tracker) = self.memory_tracker {
            tracker.register_pressure_callback(callback);
        }
    }

    /// Cleans up orphaned memory handles from the pool's handle registry
    ///
    /// This method finds memory handles in the pool's registry that are no longer
    /// referenced by any tensors (not in the cleanup registry) and deallocates them.
    ///
    /// # Returns
    ///
    /// The number of handles that were cleaned up
    pub fn cleanup_orphaned_handles(&self) -> usize {
        use crate::memory::tensor::handle::MEMORY_CLEANUP_REGISTRY;

        let mut cleanup_count = 0;

        #[cfg(feature = "tracing")]
        debug!("Starting cleanup of orphaned handles");

        // Get the current cleanup registry state
        let active_tensor_handles = if let Ok(cleanup_registry) = MEMORY_CLEANUP_REGISTRY.lock() {
            let active_handles = cleanup_registry
                .values()
                .map(|handle| handle.id())
                .collect::<std::collections::HashSet<_>>();
            #[cfg(feature = "tracing")]
            debug!(
                "Found {} active tensor handles in cleanup registry",
                active_handles.len()
            );
            active_handles
        } else {
            #[cfg(feature = "tracing")]
            error!("Failed to acquire cleanup registry lock");
            return 0;
        };

        // Find handles in pool registry that are not in cleanup registry (orphaned)
        let orphaned_handles = if let Ok(handle_registry) = self.handle_registry.read() {
            let orphaned = handle_registry
                .iter()
                .filter(|(_, handle)| !active_tensor_handles.contains(&handle.id()))
                .map(|(_, handle)| handle.clone())
                .collect::<Vec<_>>();

            #[cfg(feature = "tracing")]
            debug!(
                "Pool registry has {} total handles, {} are orphaned",
                handle_registry.len(),
                orphaned.len()
            );

            orphaned
        } else {
            #[cfg(feature = "tracing")]
            error!("Failed to acquire handle registry lock");
            return 0;
        };

        #[cfg(feature = "tracing")]
        if orphaned_handles.is_empty() {
            debug!("No orphaned handles found to clean up");
        } else {
            debug!(
                "Found {} orphaned handles to clean up",
                orphaned_handles.len()
            );
        }

        // Deallocate orphaned handles
        for handle in orphaned_handles {
            #[cfg(feature = "tracing")]
            debug!(
                "Cleaning up orphaned memory handle {} (size: {} bytes)",
                handle.id(),
                handle.size()
            );

            // Attempt to deallocate the handle
            match self.deallocate(handle.clone()) {
                Ok(()) => {
                    cleanup_count += 1;
                    #[cfg(feature = "tracing")]
                    debug!("Successfully cleaned up orphaned handle {}", handle.id());
                }
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    warn!(
                        "Failed to deallocate orphaned handle {} (size: {}): {}",
                        handle.id(),
                        handle.size(),
                        _e
                    );
                }
            }
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Cleanup completed: {} orphaned handles cleaned up",
            cleanup_count
        );

        cleanup_count
    }

    /// Resets all memory pools and metrics
    ///
    /// This will deallocate all memory and reset the pools to their initial state.
    /// Any existing handles will become invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::HybridMemoryPool;
    ///
    /// let pool = HybridMemoryPool::new()?;
    /// // ... use pool ...
    /// pool.reset()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset(&self) -> MemoryResult<()> {
        #[cfg(feature = "tracing")]
        info!("Resetting hybrid memory pool");

        // Clear handle registry
        if let Ok(mut registry) = self.handle_registry.write() {
            registry.clear();
        }

        // Reset handle ID counter
        if let Ok(mut counter) = self.next_handle_id.lock() {
            *counter = 1;
        }

        // Clear all pools
        if let Ok(mut small_pools) = self.small_pools.write() {
            small_pools.clear();
        }

        if let Ok(mut large_pools) = self.large_pools.write() {
            large_pools.clear();
        }

        // Reset metrics
        if let Ok(mut metrics) = self.metrics.write() {
            *metrics = MemoryMetrics::new();
        }

        #[cfg(feature = "tracing")]
        info!("Hybrid memory pool reset completed");

        Ok(())
    }

    // Private helper methods

    fn allocate_small(
        &self,
        size: usize,
        alignment: usize,
        device_key: &DeviceKey,
        device: &Device,
    ) -> MemoryResult<MemoryHandle> {
        let pool = self.get_or_create_small_pool(device_key, device)?;
        let mut pool = pool.lock().map_err(|_| MemoryError::InternalError {
            reason: "Failed to acquire small pool lock".to_string(),
        })?;
        pool.allocate(size, alignment, device, self.next_handle_id.clone())
    }

    fn allocate_large(
        &self,
        size: usize,
        alignment: usize,
        device_key: &DeviceKey,
        device: &Device,
    ) -> MemoryResult<MemoryHandle> {
        let pool = self.get_or_create_large_pool(device_key, device)?;
        let mut pool = pool.lock().map_err(|_| MemoryError::InternalError {
            reason: "Failed to acquire large pool lock".to_string(),
        })?;
        pool.allocate(size, alignment, device, self.next_handle_id.clone())
    }

    fn deallocate_small(&self, handle: MemoryHandle, device_key: &DeviceKey) -> MemoryResult<()> {
        let pools = self
            .small_pools
            .read()
            .map_err(|_| MemoryError::InternalError {
                reason: "Failed to acquire small pools lock".to_string(),
            })?;

        let pool = pools
            .get(device_key)
            .ok_or_else(|| MemoryError::InvalidHandle {
                reason: "Small pool not found for device".to_string(),
            })?;

        let mut pool = pool.lock().map_err(|_| MemoryError::InternalError {
            reason: "Failed to acquire small pool lock".to_string(),
        })?;

        pool.deallocate(handle)
    }

    fn deallocate_large(&self, handle: MemoryHandle, device_key: &DeviceKey) -> MemoryResult<()> {
        let pools = self
            .large_pools
            .read()
            .map_err(|_| MemoryError::InternalError {
                reason: "Failed to acquire large pools lock".to_string(),
            })?;

        let pool = pools
            .get(device_key)
            .ok_or_else(|| MemoryError::InvalidHandle {
                reason: "Large pool not found for device".to_string(),
            })?;

        let mut pool = pool.lock().map_err(|_| MemoryError::InternalError {
            reason: "Failed to acquire large pool lock".to_string(),
        })?;

        pool.deallocate(handle)
    }

    fn get_or_create_small_pool(
        &self,
        device_key: &DeviceKey,
        device: &Device,
    ) -> MemoryResult<Arc<Mutex<SmallBlockPool>>> {
        // Try to get existing pool first
        {
            let pools = self
                .small_pools
                .read()
                .map_err(|_| MemoryError::InternalError {
                    reason: "Failed to acquire small pools read lock".to_string(),
                })?;

            if let Some(pool) = pools.get(device_key) {
                return Ok(pool.clone());
            }
        }

        // Create new pool
        let mut pools = self
            .small_pools
            .write()
            .map_err(|_| MemoryError::InternalError {
                reason: "Failed to acquire small pools write lock".to_string(),
            })?;

        // Double-check in case another thread created it
        if let Some(pool) = pools.get(device_key) {
            return Ok(pool.clone());
        }

        let pool = SmallBlockPool::new(
            self.config.small_pool_initial_size,
            self.config.small_pool_max_size,
            device,
        )?;

        let pool = Arc::new(Mutex::new(pool));
        pools.insert(device_key.clone(), pool.clone());

        Ok(pool)
    }

    fn get_or_create_large_pool(
        &self,
        device_key: &DeviceKey,
        device: &Device,
    ) -> MemoryResult<Arc<Mutex<LargeBlockPool>>> {
        // Try to get existing pool first
        {
            let pools = self
                .large_pools
                .read()
                .map_err(|_| MemoryError::InternalError {
                    reason: "Failed to acquire large pools read lock".to_string(),
                })?;

            if let Some(pool) = pools.get(device_key) {
                return Ok(pool.clone());
            }
        }

        // Create new pool
        let mut pools = self
            .large_pools
            .write()
            .map_err(|_| MemoryError::InternalError {
                reason: "Failed to acquire large pools write lock".to_string(),
            })?;

        // Double-check in case another thread created it
        if let Some(pool) = pools.get(device_key) {
            return Ok(pool.clone());
        }

        let pool = LargeBlockPool::new(
            self.config.large_pool_initial_size,
            self.config.large_pool_max_size,
            device,
        )?;

        let pool = Arc::new(Mutex::new(pool));
        pools.insert(device_key.clone(), pool.clone());

        Ok(pool)
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for HybridMemoryPool {}
unsafe impl Sync for HybridMemoryPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_memory_pool_creation() {
        let pool = HybridMemoryPool::new().unwrap();
        let metrics = pool.get_metrics();
        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.total_deallocated, 0);
    }

    #[test]
    fn test_memory_pool_with_config() {
        let config = MemoryPoolConfig {
            small_block_threshold: 512 * 1024,
            enable_debug_logging: true,
            ..Default::default()
        };
        let pool = HybridMemoryPool::with_config(config).unwrap();
        assert_eq!(pool.config.small_block_threshold, 512 * 1024);
    }

    #[test]
    fn test_device_key_conversion() {
        let cpu_device = get_cpu_device();
        let cpu_key = DeviceKey::from(&cpu_device);
        assert_eq!(cpu_key, DeviceKey::Cpu);
    }

    #[test]
    fn test_invalid_alignment() {
        let pool = HybridMemoryPool::new().unwrap();
        let device = get_cpu_device();

        // Test non-power-of-2 alignment
        let result = pool.allocate(1024, 3, &device);
        assert!(matches!(result, Err(MemoryError::InvalidAlignment { .. })));

        // Test zero alignment
        let result = pool.allocate(1024, 0, &device);
        assert!(matches!(result, Err(MemoryError::InvalidAlignment { .. })));
    }

    #[test]
    fn test_memory_pool_reset() {
        let pool = HybridMemoryPool::new().unwrap();
        let device = get_cpu_device();

        // Allocate some memory
        let _handle = pool.allocate(1024, 16, &device).unwrap();

        // Reset pool
        pool.reset().unwrap();

        // Metrics should be reset
        let metrics = pool.get_metrics();
        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.total_deallocated, 0);
    }
}
