//! Memory Metrics Implementation
//!
//! This module provides comprehensive memory usage tracking and metrics
//! for the memory pool system. It tracks allocation patterns, memory
//! usage statistics, and performance metrics across different devices
//! and pool types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive memory usage metrics for the memory pool system
///
/// MemoryMetrics tracks various aspects of memory usage including:
/// - Total allocation and deallocation statistics
/// - Peak memory usage
/// - Allocation patterns by size and device
/// - Performance metrics like allocation latency
/// - Pool-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryMetrics {
    /// Total number of bytes allocated since pool creation
    pub total_allocated: u64,
    /// Total number of bytes deallocated since pool creation
    pub total_deallocated: u64,
    /// Current number of bytes in use
    pub current_allocated: u64,
    /// Peak memory usage since pool creation
    pub peak_allocated: u64,
    /// Total number of allocation requests
    pub allocation_count: u64,
    /// Total number of deallocation requests
    pub deallocation_count: u64,
    /// Current number of active allocations
    pub active_allocations: u64,
    /// Peak number of active allocations
    pub peak_active_allocations: u64,
    /// Statistics by device type
    pub device_stats: HashMap<String, DeviceMemoryStats>,
    /// Statistics by pool type
    pub pool_stats: PoolMemoryStats,
    /// Allocation size distribution
    pub size_distribution: SizeDistribution,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Timestamp when metrics were last updated
    pub last_updated: Option<std::time::SystemTime>,
}

/// Memory statistics for a specific device type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeviceMemoryStats {
    /// Total bytes allocated on this device
    pub total_allocated: u64,
    /// Total bytes deallocated on this device
    pub total_deallocated: u64,
    /// Current bytes allocated on this device
    pub current_allocated: u64,
    /// Peak bytes allocated on this device
    pub peak_allocated: u64,
    /// Number of allocations on this device
    pub allocation_count: u64,
    /// Number of deallocations on this device
    pub deallocation_count: u64,
    /// Current active allocations on this device
    pub active_allocations: u64,
}

/// Memory statistics by pool type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PoolMemoryStats {
    /// Statistics for small block pools
    pub small_block: PoolTypeStats,
    /// Statistics for large block pools
    pub large_block: PoolTypeStats,
}

/// Statistics for a specific pool type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PoolTypeStats {
    /// Total bytes allocated by this pool type
    pub total_allocated: u64,
    /// Total bytes deallocated by this pool type
    pub total_deallocated: u64,
    /// Current bytes allocated by this pool type
    pub current_allocated: u64,
    /// Peak bytes allocated by this pool type
    pub peak_allocated: u64,
    /// Number of allocations by this pool type
    pub allocation_count: u64,
    /// Number of deallocations by this pool type
    pub deallocation_count: u64,
    /// Current active allocations by this pool type
    pub active_allocations: u64,
    /// Pool-specific efficiency metrics
    pub efficiency: PoolEfficiencyMetrics,
}

/// Efficiency metrics for pool types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PoolEfficiencyMetrics {
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Memory fragmentation ratio (0.0 = no fragmentation, 1.0 = high fragmentation)
    pub fragmentation_ratio: f64,
    /// Pool utilization ratio (0.0 = empty, 1.0 = full)
    pub utilization_ratio: f64,
    /// Number of pool expansions
    pub expansion_count: u64,
    /// Number of pool contractions
    pub contraction_count: u64,
}

/// Distribution of allocation sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SizeDistribution {
    /// Allocations <= 1KB
    pub tiny: u64,
    /// Allocations 1KB - 64KB
    pub small: u64,
    /// Allocations 64KB - 1MB
    pub medium: u64,
    /// Allocations 1MB - 16MB
    pub large: u64,
    /// Allocations > 16MB
    pub huge: u64,
}

/// Performance metrics for memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PerformanceMetrics {
    /// Average allocation latency in nanoseconds
    pub avg_allocation_latency_ns: u64,
    /// Average deallocation latency in nanoseconds
    pub avg_deallocation_latency_ns: u64,
    /// Maximum allocation latency in nanoseconds
    pub max_allocation_latency_ns: u64,
    /// Maximum deallocation latency in nanoseconds
    pub max_deallocation_latency_ns: u64,
    /// Total time spent in allocation operations
    pub total_allocation_time_ns: u64,
    /// Total time spent in deallocation operations
    pub total_deallocation_time_ns: u64,
    /// Number of allocation failures
    pub allocation_failures: u64,
    /// Number of deallocation failures
    pub deallocation_failures: u64,
}

impl MemoryMetrics {
    /// Creates a new MemoryMetrics instance with all counters initialized to zero
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            current_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            active_allocations: 0,
            peak_active_allocations: 0,
            device_stats: HashMap::new(),
            pool_stats: PoolMemoryStats {
                small_block: PoolTypeStats::new(),
                large_block: PoolTypeStats::new(),
            },
            size_distribution: SizeDistribution::new(),
            performance: PerformanceMetrics::new(),
            last_updated: None,
        }
    }

    /// Records a successful memory allocation
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    /// * `device_type` - Type of device where memory was allocated
    /// * `is_small_block` - Whether this was allocated by a small block pool
    /// * `latency` - Time taken for the allocation operation
    pub fn record_allocation(&mut self, size: usize) {
        let size = size as u64;

        // Update global counters
        self.total_allocated += size;
        self.current_allocated += size;
        self.allocation_count += 1;
        self.active_allocations += 1;

        // Update peak values
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
        if self.active_allocations > self.peak_active_allocations {
            self.peak_active_allocations = self.active_allocations;
        }

        // Update size distribution
        self.size_distribution.record_allocation(size);

        // Update timestamp
        self.last_updated = Some(std::time::SystemTime::now());
    }

    /// Records a successful memory allocation with detailed information
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    /// * `device_type` - Type of device where memory was allocated
    /// * `is_small_block` - Whether this was allocated by a small block pool
    /// * `latency` - Time taken for the allocation operation
    pub fn record_allocation_detailed(
        &mut self,
        size: usize,
        device_type: &str,
        is_small_block: bool,
        latency: Duration,
    ) {
        let size = size as u64;
        let latency_ns = latency.as_nanos() as u64;

        // Record basic allocation
        self.record_allocation(size as usize);

        // Update device-specific stats
        let device_stats = self
            .device_stats
            .entry(device_type.to_string())
            .or_insert_with(DeviceMemoryStats::new);
        device_stats.record_allocation(size);

        // Update pool-specific stats
        let pool_stats = if is_small_block {
            &mut self.pool_stats.small_block
        } else {
            &mut self.pool_stats.large_block
        };
        pool_stats.record_allocation(size);

        // Update performance metrics
        self.performance.record_allocation_latency(latency_ns);
    }

    /// Records a successful memory deallocation
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the deallocation in bytes
    pub fn record_deallocation(&mut self, size: usize) {
        let size = size as u64;

        // Update global counters
        self.total_deallocated += size;
        self.current_allocated = self.current_allocated.saturating_sub(size);
        self.deallocation_count += 1;
        self.active_allocations = self.active_allocations.saturating_sub(1);

        // Update timestamp
        self.last_updated = Some(std::time::SystemTime::now());
    }

    /// Records a successful memory deallocation with detailed information
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the deallocation in bytes
    /// * `device_type` - Type of device where memory was deallocated
    /// * `is_small_block` - Whether this was deallocated by a small block pool
    /// * `latency` - Time taken for the deallocation operation
    pub fn record_deallocation_detailed(
        &mut self,
        size: usize,
        device_type: &str,
        is_small_block: bool,
        latency: Duration,
    ) {
        let size = size as u64;
        let latency_ns = latency.as_nanos() as u64;

        // Record basic deallocation
        self.record_deallocation(size as usize);

        // Update device-specific stats
        if let Some(device_stats) = self.device_stats.get_mut(device_type) {
            device_stats.record_deallocation(size);
        }

        // Update pool-specific stats
        let pool_stats = if is_small_block {
            &mut self.pool_stats.small_block
        } else {
            &mut self.pool_stats.large_block
        };
        pool_stats.record_deallocation(size);

        // Update performance metrics
        self.performance.record_deallocation_latency(latency_ns);
    }

    /// Records an allocation failure
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the failed allocation in bytes
    /// * `device_type` - Type of device where allocation failed
    pub fn record_allocation_failure(&mut self, _size: usize, _device_type: &str) {
        self.performance.allocation_failures += 1;
        self.last_updated = Some(std::time::SystemTime::now());
    }

    /// Records a deallocation failure
    ///
    /// # Arguments
    ///
    /// * `device_type` - Type of device where deallocation failed
    pub fn record_deallocation_failure(&mut self, _device_type: &str) {
        self.performance.deallocation_failures += 1;
        self.last_updated = Some(std::time::SystemTime::now());
    }

    /// Returns the current memory efficiency ratio (0.0 to 1.0)
    ///
    /// This is calculated as the ratio of currently allocated memory
    /// to peak allocated memory.
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_allocated == 0 {
            1.0
        } else {
            self.current_allocated as f64 / self.peak_allocated as f64
        }
    }

    /// Returns the allocation success rate (0.0 to 1.0)
    pub fn allocation_success_rate(&self) -> f64 {
        let total_attempts = self.allocation_count + self.performance.allocation_failures;
        if total_attempts == 0 {
            1.0
        } else {
            self.allocation_count as f64 / total_attempts as f64
        }
    }

    /// Returns the deallocation success rate (0.0 to 1.0)
    pub fn deallocation_success_rate(&self) -> f64 {
        let total_attempts = self.deallocation_count + self.performance.deallocation_failures;
        if total_attempts == 0 {
            1.0
        } else {
            self.deallocation_count as f64 / total_attempts as f64
        }
    }

    /// Returns the average allocation size in bytes
    pub fn average_allocation_size(&self) -> f64 {
        if self.allocation_count == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.allocation_count as f64
        }
    }

    /// Resets all metrics to their initial state
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merges metrics from another MemoryMetrics instance
    ///
    /// This is useful for aggregating metrics from multiple pools or time periods.
    pub fn merge(&mut self, other: &MemoryMetrics) {
        self.total_allocated += other.total_allocated;
        self.total_deallocated += other.total_deallocated;
        self.current_allocated += other.current_allocated;
        self.peak_allocated = self.peak_allocated.max(other.peak_allocated);
        self.allocation_count += other.allocation_count;
        self.deallocation_count += other.deallocation_count;
        self.active_allocations += other.active_allocations;
        self.peak_active_allocations = self
            .peak_active_allocations
            .max(other.peak_active_allocations);

        // Merge device stats
        for (device, stats) in &other.device_stats {
            let entry = self
                .device_stats
                .entry(device.clone())
                .or_insert_with(DeviceMemoryStats::new);
            entry.merge(stats);
        }

        // Merge pool stats
        self.pool_stats
            .small_block
            .merge(&other.pool_stats.small_block);
        self.pool_stats
            .large_block
            .merge(&other.pool_stats.large_block);

        // Merge size distribution
        self.size_distribution.merge(&other.size_distribution);

        // Merge performance metrics
        self.performance.merge(&other.performance);

        // Update timestamp
        self.last_updated = Some(std::time::SystemTime::now());
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceMemoryStats {
    fn new() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            current_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            active_allocations: 0,
        }
    }

    fn record_allocation(&mut self, size: u64) {
        self.total_allocated += size;
        self.current_allocated += size;
        self.allocation_count += 1;
        self.active_allocations += 1;

        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
    }

    fn record_deallocation(&mut self, size: u64) {
        self.total_deallocated += size;
        self.current_allocated = self.current_allocated.saturating_sub(size);
        self.deallocation_count += 1;
        self.active_allocations = self.active_allocations.saturating_sub(1);
    }

    fn merge(&mut self, other: &DeviceMemoryStats) {
        self.total_allocated += other.total_allocated;
        self.total_deallocated += other.total_deallocated;
        self.current_allocated += other.current_allocated;
        self.peak_allocated = self.peak_allocated.max(other.peak_allocated);
        self.allocation_count += other.allocation_count;
        self.deallocation_count += other.deallocation_count;
        self.active_allocations += other.active_allocations;
    }
}

impl PoolTypeStats {
    fn new() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            current_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            active_allocations: 0,
            efficiency: PoolEfficiencyMetrics::new(),
        }
    }

    fn record_allocation(&mut self, size: u64) {
        self.total_allocated += size;
        self.current_allocated += size;
        self.allocation_count += 1;
        self.active_allocations += 1;

        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }

        // Update efficiency metrics
        self.efficiency
            .update_average_allocation_size(self.total_allocated, self.allocation_count);
    }

    fn record_deallocation(&mut self, size: u64) {
        self.total_deallocated += size;
        self.current_allocated = self.current_allocated.saturating_sub(size);
        self.deallocation_count += 1;
        self.active_allocations = self.active_allocations.saturating_sub(1);
    }

    fn merge(&mut self, other: &PoolTypeStats) {
        self.total_allocated += other.total_allocated;
        self.total_deallocated += other.total_deallocated;
        self.current_allocated += other.current_allocated;
        self.peak_allocated = self.peak_allocated.max(other.peak_allocated);
        self.allocation_count += other.allocation_count;
        self.deallocation_count += other.deallocation_count;
        self.active_allocations += other.active_allocations;
        self.efficiency.merge(&other.efficiency);
    }
}

impl PoolEfficiencyMetrics {
    fn new() -> Self {
        Self {
            average_allocation_size: 0.0,
            fragmentation_ratio: 0.0,
            utilization_ratio: 0.0,
            expansion_count: 0,
            contraction_count: 0,
        }
    }

    fn update_average_allocation_size(&mut self, total_allocated: u64, allocation_count: u64) {
        if allocation_count > 0 {
            self.average_allocation_size = total_allocated as f64 / allocation_count as f64;
        }
    }

    fn merge(&mut self, other: &PoolEfficiencyMetrics) {
        // For averages, we take the maximum as a simple heuristic
        self.average_allocation_size = self
            .average_allocation_size
            .max(other.average_allocation_size);
        self.fragmentation_ratio = self.fragmentation_ratio.max(other.fragmentation_ratio);
        self.utilization_ratio = self.utilization_ratio.max(other.utilization_ratio);
        self.expansion_count += other.expansion_count;
        self.contraction_count += other.contraction_count;
    }
}

impl SizeDistribution {
    fn new() -> Self {
        Self {
            tiny: 0,
            small: 0,
            medium: 0,
            large: 0,
            huge: 0,
        }
    }

    fn record_allocation(&mut self, size: u64) {
        match size {
            0..=1024 => self.tiny += 1,
            1025..=65536 => self.small += 1,
            65537..=1048576 => self.medium += 1,
            1048577..=16777216 => self.large += 1,
            _ => self.huge += 1,
        }
    }

    fn merge(&mut self, other: &SizeDistribution) {
        self.tiny += other.tiny;
        self.small += other.small;
        self.medium += other.medium;
        self.large += other.large;
        self.huge += other.huge;
    }
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            avg_allocation_latency_ns: 0,
            avg_deallocation_latency_ns: 0,
            max_allocation_latency_ns: 0,
            max_deallocation_latency_ns: 0,
            total_allocation_time_ns: 0,
            total_deallocation_time_ns: 0,
            allocation_failures: 0,
            deallocation_failures: 0,
        }
    }

    fn record_allocation_latency(&mut self, latency_ns: u64) {
        self.total_allocation_time_ns += latency_ns;
        self.max_allocation_latency_ns = self.max_allocation_latency_ns.max(latency_ns);

        // Update average (simple moving average)
        let count = (self.total_allocation_time_ns / self.avg_allocation_latency_ns.max(1)).max(1);
        self.avg_allocation_latency_ns = self.total_allocation_time_ns / count;
    }

    fn record_deallocation_latency(&mut self, latency_ns: u64) {
        self.total_deallocation_time_ns += latency_ns;
        self.max_deallocation_latency_ns = self.max_deallocation_latency_ns.max(latency_ns);

        // Update average (simple moving average)
        let count =
            (self.total_deallocation_time_ns / self.avg_deallocation_latency_ns.max(1)).max(1);
        self.avg_deallocation_latency_ns = self.total_deallocation_time_ns / count;
    }

    fn merge(&mut self, other: &PerformanceMetrics) {
        // Merge timing data
        self.total_allocation_time_ns += other.total_allocation_time_ns;
        self.total_deallocation_time_ns += other.total_deallocation_time_ns;
        self.max_allocation_latency_ns = self
            .max_allocation_latency_ns
            .max(other.max_allocation_latency_ns);
        self.max_deallocation_latency_ns = self
            .max_deallocation_latency_ns
            .max(other.max_deallocation_latency_ns);

        // Recalculate averages
        let total_ops = self.allocation_failures
            + other.allocation_failures
            + self.deallocation_failures
            + other.deallocation_failures;
        if total_ops > 0 {
            self.avg_allocation_latency_ns =
                (self.avg_allocation_latency_ns + other.avg_allocation_latency_ns) / 2;
            self.avg_deallocation_latency_ns =
                (self.avg_deallocation_latency_ns + other.avg_deallocation_latency_ns) / 2;
        }

        // Merge failure counts
        self.allocation_failures += other.allocation_failures;
        self.deallocation_failures += other.deallocation_failures;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_metrics_creation() {
        let metrics = MemoryMetrics::new();
        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.total_deallocated, 0);
        assert_eq!(metrics.current_allocated, 0);
        assert_eq!(metrics.allocation_count, 0);
        assert_eq!(metrics.deallocation_count, 0);
    }

    #[test]
    fn test_record_allocation() {
        let mut metrics = MemoryMetrics::new();
        metrics.record_allocation(1024);

        assert_eq!(metrics.total_allocated, 1024);
        assert_eq!(metrics.current_allocated, 1024);
        assert_eq!(metrics.peak_allocated, 1024);
        assert_eq!(metrics.allocation_count, 1);
        assert_eq!(metrics.active_allocations, 1);
        assert_eq!(metrics.peak_active_allocations, 1);
    }

    #[test]
    fn test_record_deallocation() {
        let mut metrics = MemoryMetrics::new();
        metrics.record_allocation(1024);
        metrics.record_deallocation(1024);

        assert_eq!(metrics.total_allocated, 1024);
        assert_eq!(metrics.total_deallocated, 1024);
        assert_eq!(metrics.current_allocated, 0);
        assert_eq!(metrics.peak_allocated, 1024);
        assert_eq!(metrics.allocation_count, 1);
        assert_eq!(metrics.deallocation_count, 1);
        assert_eq!(metrics.active_allocations, 0);
        assert_eq!(metrics.peak_active_allocations, 1);
    }

    #[test]
    fn test_peak_tracking() {
        let mut metrics = MemoryMetrics::new();

        // First allocation
        metrics.record_allocation(1024);
        assert_eq!(metrics.peak_allocated, 1024);

        // Second allocation (higher peak)
        metrics.record_allocation(2048);
        assert_eq!(metrics.peak_allocated, 3072);

        // Deallocate first allocation (peak should remain)
        metrics.record_deallocation(1024);
        assert_eq!(metrics.peak_allocated, 3072);
        assert_eq!(metrics.current_allocated, 2048);
    }

    #[test]
    fn test_size_distribution() {
        let mut distribution = SizeDistribution::new();

        distribution.record_allocation(512); // tiny
        distribution.record_allocation(2048); // small
        distribution.record_allocation(131072); // medium
        distribution.record_allocation(2097152); // large
        distribution.record_allocation(33554432); // huge

        assert_eq!(distribution.tiny, 1);
        assert_eq!(distribution.small, 1);
        assert_eq!(distribution.medium, 1);
        assert_eq!(distribution.large, 1);
        assert_eq!(distribution.huge, 1);
    }

    #[test]
    fn test_memory_efficiency() {
        let mut metrics = MemoryMetrics::new();

        // No allocations - should be 100% efficient
        assert_eq!(metrics.memory_efficiency(), 1.0);

        // Allocate and deallocate
        metrics.record_allocation(1024);
        assert_eq!(metrics.memory_efficiency(), 1.0); // 1024/1024

        metrics.record_allocation(1024);
        assert_eq!(metrics.memory_efficiency(), 1.0); // 2048/2048

        metrics.record_deallocation(1024);
        assert_eq!(metrics.memory_efficiency(), 0.5); // 1024/2048
    }

    #[test]
    fn test_average_allocation_size() {
        let mut metrics = MemoryMetrics::new();

        // No allocations
        assert_eq!(metrics.average_allocation_size(), 0.0);

        // Single allocation
        metrics.record_allocation(1024);
        assert_eq!(metrics.average_allocation_size(), 1024.0);

        // Multiple allocations
        metrics.record_allocation(2048);
        assert_eq!(metrics.average_allocation_size(), 1536.0); // (1024 + 2048) / 2
    }

    #[test]
    fn test_metrics_merge() {
        let mut metrics1 = MemoryMetrics::new();
        let mut metrics2 = MemoryMetrics::new();

        metrics1.record_allocation(1024);
        metrics2.record_allocation(2048);

        metrics1.merge(&metrics2);

        assert_eq!(metrics1.total_allocated, 3072);
        assert_eq!(metrics1.allocation_count, 2);
        assert_eq!(metrics1.peak_allocated, 2048);
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = MemoryMetrics::new();

        metrics.record_allocation(1024);
        metrics.record_deallocation(512);

        assert_ne!(metrics.total_allocated, 0);
        assert_ne!(metrics.allocation_count, 0);

        metrics.reset();

        assert_eq!(metrics.total_allocated, 0);
        assert_eq!(metrics.total_deallocated, 0);
        assert_eq!(metrics.allocation_count, 0);
        assert_eq!(metrics.deallocation_count, 0);
    }
}
