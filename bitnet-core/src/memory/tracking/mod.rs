//! Advanced Memory Tracking System
//!
//! This module provides comprehensive memory usage tracking utilities for the BitNet
//! memory pool system. It implements real-time memory monitoring, pressure detection,
//! profiling capabilities, and detailed analytics for memory allocation patterns.
//!
//! # Architecture
//!
//! The tracking system consists of several key components:
//!
//! - **MemoryTracker**: Main interface for real-time memory monitoring
//! - **MemoryPressureDetector**: Monitors memory pressure and triggers alerts
//! - **MemoryProfiler**: Provides debugging and leak detection capabilities
//! - **AllocationTimeline**: Tracks allocation/deallocation events over time
//! - **PatternAnalyzer**: Analyzes allocation patterns and identifies anti-patterns
//!
//! # Features
//!
//! - Real-time memory monitoring with detailed metrics
//! - Memory pressure detection with configurable thresholds
//! - Allocation timeline tracking for debugging
//! - Memory leak detection and reporting
//! - Pattern analysis for optimization opportunities
//! - Performance metrics with minimal overhead
//! - Device-specific tracking (CPU vs Metal GPU)
//! - Configurable tracking levels (minimal, standard, detailed, debug)
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::memory::tracking::{MemoryTracker, TrackingConfig};
//! use bitnet_core::device::auto_select_device;
//!
//! // Create a memory tracker with detailed tracking
//! let config = TrackingConfig::detailed();
//! let tracker = MemoryTracker::new(config)?;
//!
//! // Track an allocation
//! let device = auto_select_device();
//! let handle = pool.allocate(1024, 16, &device)?;
//! tracker.track_allocation(&handle, 1024, &device);
//!
//! // Get detailed metrics
//! let metrics = tracker.get_detailed_metrics();
//! println!("Memory pressure: {:?}", tracker.get_pressure_level());
//!
//! // Track deallocation
//! tracker.track_deallocation(&handle);
//! pool.deallocate(handle)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;

#[cfg(feature = "tracing")]
use tracing::error;

// Sub-modules
pub mod config;
pub mod patterns;
pub mod pressure;
pub mod profiler;
pub mod timeline;
pub mod tracker;

// Re-exports
pub use config::{PressureThresholds, TrackingConfig, TrackingLevel};
pub use patterns::{AllocationPattern, PatternAnalyzer, PatternReport};
pub use pressure::{MemoryPressureDetector, MemoryPressureLevel, PressureCallback};
pub use profiler::{LeakReport, MemoryProfiler, ProfilingReport};
pub use timeline::{AllocationEvent, AllocationTimeline, TimelineEntry};
pub use tracker::{DetailedMemoryMetrics, MemoryTracker};

/// Errors that can occur during memory tracking operations
#[derive(Error, Debug)]
pub enum TrackingError {
    /// Tracking system initialization failed
    #[error("Tracking system initialization failed: {reason}")]
    InitializationFailed { reason: String },

    /// Invalid tracking configuration
    #[error("Invalid tracking configuration: {reason}")]
    InvalidConfiguration { reason: String },

    /// Memory tracking operation failed
    #[error("Memory tracking operation failed: {reason}")]
    OperationFailed { reason: String },

    /// Pressure detection system error
    #[error("Pressure detection error: {reason}")]
    PressureDetectionError { reason: String },

    /// Profiling operation failed
    #[error("Profiling operation failed: {reason}")]
    ProfilingError { reason: String },

    /// Timeline tracking error
    #[error("Timeline tracking error: {reason}")]
    TimelineError { reason: String },

    /// Pattern analysis error
    #[error("Pattern analysis error: {reason}")]
    PatternAnalysisError { reason: String },

    /// Internal tracking system error
    #[error("Internal tracking system error: {reason}")]
    InternalError { reason: String },
}

/// Result type for tracking operations
pub type TrackingResult<T> = std::result::Result<T, TrackingError>;

/// Unique identifier for memory allocations in the tracking system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocationId(pub u64);

impl AllocationId {
    /// Creates a new allocation ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for AllocationId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<AllocationId> for u64 {
    fn from(id: AllocationId) -> u64 {
        id.0
    }
}

/// Information about a tracked memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Unique identifier for this allocation
    pub id: AllocationId,
    /// Size of the allocation in bytes
    pub size: usize,
    /// Alignment of the allocation
    pub alignment: usize,
    /// Device where the allocation was made
    pub device_type: String,
    /// Timestamp when the allocation was made
    pub timestamp: SystemTime,
    /// Duration since tracking system started
    pub elapsed: Duration,
    /// Optional stack trace (if enabled)
    pub stack_trace: Option<Vec<String>>,
    /// Pool type that handled the allocation
    pub pool_type: String,
    /// Whether this allocation is still active
    pub is_active: bool,
}

impl AllocationInfo {
    /// Creates a new allocation info
    pub fn new(
        id: AllocationId,
        size: usize,
        alignment: usize,
        device: &Device,
        pool_type: String,
        start_time: Instant,
    ) -> Self {
        let now = SystemTime::now();
        let elapsed = start_time.elapsed();

        Self {
            id,
            size,
            alignment,
            device_type: format!("{:?}", device),
            timestamp: now,
            elapsed,
            stack_trace: None,
            pool_type,
            is_active: true,
        }
    }

    /// Marks this allocation as deallocated
    pub fn mark_deallocated(&mut self) {
        self.is_active = false;
    }

    /// Returns the age of this allocation
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or(Duration::ZERO)
    }
}

/// Statistics for device-specific memory tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTrackingStats {
    /// Device type identifier
    pub device_type: String,
    /// Total allocations on this device
    pub total_allocations: u64,
    /// Total deallocations on this device
    pub total_deallocations: u64,
    /// Current active allocations
    pub active_allocations: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes deallocated
    pub total_bytes_deallocated: u64,
    /// Current bytes in use
    pub current_bytes_allocated: u64,
    /// Peak bytes allocated
    pub peak_bytes_allocated: u64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Deallocation rate (deallocations per second)
    pub deallocation_rate: f64,
}

impl DeviceTrackingStats {
    /// Creates new device tracking stats
    pub fn new(device_type: String) -> Self {
        Self {
            device_type,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            total_bytes_allocated: 0,
            total_bytes_deallocated: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
            average_allocation_size: 0.0,
            allocation_rate: 0.0,
            deallocation_rate: 0.0,
        }
    }

    /// Records an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocations += 1;
        self.active_allocations += 1;
        self.total_bytes_allocated += size as u64;
        self.current_bytes_allocated += size as u64;

        if self.current_bytes_allocated > self.peak_bytes_allocated {
            self.peak_bytes_allocated = self.current_bytes_allocated;
        }

        self.update_average_allocation_size();
    }

    /// Records a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.active_allocations = self.active_allocations.saturating_sub(1);
        self.total_bytes_deallocated += size as u64;
        self.current_bytes_allocated = self.current_bytes_allocated.saturating_sub(size as u64);
    }

    /// Updates the average allocation size
    fn update_average_allocation_size(&mut self) {
        if self.total_allocations > 0 {
            self.average_allocation_size =
                self.total_bytes_allocated as f64 / self.total_allocations as f64;
        }
    }

    /// Updates allocation and deallocation rates
    pub fn update_rates(&mut self, elapsed_seconds: f64) {
        if elapsed_seconds > 0.0 {
            self.allocation_rate = self.total_allocations as f64 / elapsed_seconds;
            self.deallocation_rate = self.total_deallocations as f64 / elapsed_seconds;
        }
    }
}

/// Global tracking statistics across all devices and pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalTrackingStats {
    /// Statistics per device type
    pub device_stats: HashMap<String, DeviceTrackingStats>,
    /// Total tracking time
    pub total_tracking_time: Duration,
    /// Number of pressure events detected
    pub pressure_events: u64,
    /// Number of potential leaks detected
    pub potential_leaks: u64,
    /// Number of pattern anomalies detected
    pub pattern_anomalies: u64,
    /// Tracking system overhead percentage
    pub overhead_percentage: f64,
}

impl GlobalTrackingStats {
    /// Creates new global tracking stats
    pub fn new() -> Self {
        Self {
            device_stats: HashMap::new(),
            total_tracking_time: Duration::ZERO,
            pressure_events: 0,
            potential_leaks: 0,
            pattern_anomalies: 0,
            overhead_percentage: 0.0,
        }
    }

    /// Gets or creates device stats
    pub fn get_or_create_device_stats(&mut self, device_type: &str) -> &mut DeviceTrackingStats {
        self.device_stats
            .entry(device_type.to_string())
            .or_insert_with(|| DeviceTrackingStats::new(device_type.to_string()))
    }

    /// Updates global statistics
    pub fn update(&mut self, elapsed: Duration) {
        self.total_tracking_time = elapsed;

        // Update rates for all devices
        let elapsed_seconds = elapsed.as_secs_f64();
        for stats in self.device_stats.values_mut() {
            stats.update_rates(elapsed_seconds);
        }
    }

    /// Returns total allocations across all devices
    pub fn total_allocations(&self) -> u64 {
        self.device_stats
            .values()
            .map(|s| s.total_allocations)
            .sum()
    }

    /// Returns total active allocations across all devices
    pub fn total_active_allocations(&self) -> u64 {
        self.device_stats
            .values()
            .map(|s| s.active_allocations)
            .sum()
    }

    /// Returns total bytes allocated across all devices
    pub fn total_bytes_allocated(&self) -> u64 {
        self.device_stats
            .values()
            .map(|s| s.current_bytes_allocated)
            .sum()
    }

    /// Returns peak bytes allocated across all devices
    pub fn peak_bytes_allocated(&self) -> u64 {
        self.device_stats
            .values()
            .map(|s| s.peak_bytes_allocated)
            .max()
            .unwrap_or(0)
    }
}

impl Default for GlobalTrackingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_id() {
        let id = AllocationId::new(42);
        assert_eq!(id.raw(), 42);

        let id_from_u64: AllocationId = 123.into();
        assert_eq!(id_from_u64.raw(), 123);

        let u64_from_id: u64 = id.into();
        assert_eq!(u64_from_id, 42);
    }

    #[test]
    fn test_device_tracking_stats() {
        let mut stats = DeviceTrackingStats::new("CPU".to_string());

        // Record some allocations
        stats.record_allocation(1024);
        stats.record_allocation(2048);

        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.total_bytes_allocated, 3072);
        assert_eq!(stats.current_bytes_allocated, 3072);
        assert_eq!(stats.peak_bytes_allocated, 3072);
        assert_eq!(stats.average_allocation_size, 1536.0);

        // Record a deallocation
        stats.record_deallocation(1024);

        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.current_bytes_allocated, 2048);
        assert_eq!(stats.peak_bytes_allocated, 3072); // Peak should remain
    }

    #[test]
    fn test_global_tracking_stats() {
        let mut global_stats = GlobalTrackingStats::new();

        // Add some device stats
        let cpu_stats = global_stats.get_or_create_device_stats("CPU");
        cpu_stats.record_allocation(1024);
        cpu_stats.record_allocation(2048);

        let metal_stats = global_stats.get_or_create_device_stats("Metal");
        metal_stats.record_allocation(4096);

        assert_eq!(global_stats.total_allocations(), 3);
        assert_eq!(global_stats.total_active_allocations(), 3);
        assert_eq!(global_stats.total_bytes_allocated(), 7168);
    }

    #[test]
    fn test_allocation_info() {
        use crate::device::get_cpu_device;

        let device = get_cpu_device();
        let start_time = Instant::now();
        let id = AllocationId::new(1);

        let mut info =
            AllocationInfo::new(id, 1024, 16, &device, "SmallBlock".to_string(), start_time);

        assert_eq!(info.id, id);
        assert_eq!(info.size, 1024);
        assert_eq!(info.alignment, 16);
        assert_eq!(info.pool_type, "SmallBlock");
        assert!(info.is_active);

        info.mark_deallocated();
        assert!(!info.is_active);
    }
}
