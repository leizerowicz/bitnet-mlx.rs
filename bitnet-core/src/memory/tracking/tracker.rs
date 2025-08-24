//! Core Memory Tracker Implementation
//!
//! This module provides the main MemoryTracker interface for real-time memory
//! monitoring with detailed metrics collection and minimal performance overhead.

use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "tracing")]
use tracing::{debug, error, info};

use super::{
    config::TrackingConfig,
    patterns::{AllocationPattern, PatternAnalyzer},
    pressure::{MemoryPressureDetector, MemoryPressureLevel, PressureCallback},
    timeline::{AllocationEvent, AllocationTimeline},
    AllocationId, AllocationInfo, GlobalTrackingStats, TrackingError, TrackingResult,
};
use crate::memory::MemoryHandle;

/// Main memory tracker that provides real-time monitoring and detailed metrics
#[derive(Debug)]
pub struct MemoryTracker {
    /// Tracking configuration
    config: TrackingConfig,
    /// Start time for tracking
    start_time: Instant,
    /// Global tracking statistics
    stats: Arc<RwLock<GlobalTrackingStats>>,
    /// Active allocations being tracked
    active_allocations: Arc<RwLock<HashMap<AllocationId, AllocationInfo>>>,
    /// Allocation ID counter
    next_allocation_id: Arc<Mutex<u64>>,
    /// Memory pressure detector
    pressure_detector: Option<Arc<MemoryPressureDetector>>,
    /// Allocation timeline tracker
    timeline: Option<Arc<Mutex<AllocationTimeline>>>,
    /// Pattern analyzer
    pattern_analyzer: Option<Arc<Mutex<PatternAnalyzer>>>,
    /// Performance tracking
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
}

/// Detailed memory metrics with comprehensive tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMemoryMetrics {
    /// Global tracking statistics
    pub global_stats: GlobalTrackingStats,
    /// Current memory pressure level
    pub pressure_level: MemoryPressureLevel,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Total memory currently allocated
    pub current_memory_usage: u64,
    /// Peak memory usage since tracking started
    pub peak_memory_usage: u64,
    /// Memory usage by device type
    pub device_usage: HashMap<String, u64>,
    /// Recent allocation patterns (if pattern analysis is enabled)
    pub recent_patterns: Vec<AllocationPattern>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Tracking system overhead
    pub tracking_overhead: TrackingOverhead,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

/// Performance metrics for tracking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average time to track an allocation (nanoseconds)
    pub avg_track_allocation_time_ns: u64,
    /// Average time to track a deallocation (nanoseconds)
    pub avg_track_deallocation_time_ns: u64,
    /// Maximum time to track an allocation (nanoseconds)
    pub max_track_allocation_time_ns: u64,
    /// Maximum time to track a deallocation (nanoseconds)
    pub max_track_deallocation_time_ns: u64,
    /// Total tracking operations performed
    pub total_tracking_operations: u64,
    /// Number of tracking operations that failed
    pub failed_tracking_operations: u64,
}

/// Tracking system overhead information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingOverhead {
    /// Estimated memory overhead in bytes
    pub memory_overhead_bytes: u64,
    /// CPU overhead as percentage of total CPU time
    pub cpu_overhead_percentage: f64,
    /// Number of tracking data structures
    pub tracking_structures_count: usize,
    /// Size of largest tracking structure
    pub largest_structure_size_bytes: u64,
}

/// Internal performance tracker
#[derive(Debug)]
struct PerformanceTracker {
    allocation_times: Vec<Duration>,
    deallocation_times: Vec<Duration>,
    total_operations: u64,
    failed_operations: u64,
    overhead_start_time: Instant,
    total_tracking_time: Duration,
}

impl MemoryTracker {
    /// Creates a new memory tracker with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the tracking system
    ///
    /// # Returns
    ///
    /// A Result containing the new tracker or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::{MemoryTracker, TrackingConfig};
    ///
    /// let config = TrackingConfig::standard();
    /// let tracker = MemoryTracker::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: TrackingConfig) -> TrackingResult<Self> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| TrackingError::InvalidConfiguration { reason: e })?;

        #[cfg(feature = "tracing")]
        info!("Creating memory tracker with config: {:?}", config.level);

        let start_time = Instant::now();

        // Initialize pressure detector if enabled
        let pressure_detector = if config.enable_pressure_monitoring {
            Some(Arc::new(MemoryPressureDetector::new(
                config.pressure_thresholds.clone(),
            )?))
        } else {
            None
        };

        // Initialize timeline tracker if enabled
        let timeline = if config.enable_timeline_tracking {
            Some(Arc::new(Mutex::new(AllocationTimeline::new(
                config.max_timeline_entries,
                config.timeline_retention,
            ))))
        } else {
            None
        };

        // Initialize pattern analyzer if enabled
        let pattern_analyzer = if config.enable_pattern_analysis {
            Some(Arc::new(Mutex::new(PatternAnalyzer::new(
                Default::default(),
            ))))
        } else {
            None
        };

        let tracker = Self {
            config,
            start_time,
            stats: Arc::new(RwLock::new(GlobalTrackingStats::new())),
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            next_allocation_id: Arc::new(Mutex::new(1)),
            pressure_detector,
            timeline,
            pattern_analyzer,
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::new())),
        };

        #[cfg(feature = "tracing")]
        info!("Memory tracker created successfully");

        Ok(tracker)
    }

    /// Tracks a memory allocation
    ///
    /// # Arguments
    ///
    /// * `handle` - Memory handle for the allocation
    /// * `size` - Size of the allocation in bytes
    /// * `device` - Device where the allocation was made
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::MemoryTracker;
    /// use bitnet_core::device::get_cpu_device;
    ///
    /// let tracker = MemoryTracker::new(Default::default())?;
    /// let device = get_cpu_device();
    /// // Assume we have a memory handle from pool allocation
    /// // tracker.track_allocation(&handle, 1024, &device);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn track_allocation(&self, handle: &MemoryHandle, size: usize, device: &Device) {
        let track_start = Instant::now();

        // Generate allocation ID
        let allocation_id = self.generate_allocation_id();

        // Check if we should sample this allocation
        if !self.config.should_sample_allocation(allocation_id.raw()) {
            return;
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Tracking allocation {} of {} bytes on {:?}",
            allocation_id.raw(),
            size,
            device
        );

        // Create allocation info
        let pool_type = match handle.pool_type() {
            crate::memory::handle::PoolType::SmallBlock => "SmallBlock".to_string(),
            crate::memory::handle::PoolType::LargeBlock => "LargeBlock".to_string(),
        };

        let mut allocation_info = AllocationInfo::new(
            allocation_id,
            size,
            handle.alignment(),
            device,
            pool_type,
            self.start_time,
        );

        // Capture stack trace if enabled
        if self.config.should_capture_stack_traces() {
            allocation_info.stack_trace = self.capture_stack_trace();
        }

        // Update global statistics
        if let Ok(mut stats) = self.stats.write() {
            let device_type = format!("{:?}", device);
            let device_stats = stats.get_or_create_device_stats(&device_type);
            device_stats.record_allocation(size);
            stats.update(self.start_time.elapsed());
        }

        // Store active allocation
        if let Ok(mut active) = self.active_allocations.write() {
            active.insert(allocation_id, allocation_info.clone());
        }

        // Update timeline if enabled
        if let Some(timeline) = &self.timeline {
            if let Ok(timeline) = timeline.lock() {
                let event = AllocationEvent::Allocation {
                    id: allocation_id,
                    size,
                    device_type: format!("{:?}", device),
                    timestamp: SystemTime::now(),
                };
                timeline.add_event(event);
            }
        }

        // Update pattern analyzer if enabled
        if let Some(analyzer) = &self.pattern_analyzer {
            if let Ok(analyzer) = analyzer.lock() {
                analyzer.record_allocation(allocation_info);
            }
        }

        // Update pressure detector if enabled
        if let Some(detector) = &self.pressure_detector {
            detector.update_memory_usage(self.get_current_memory_usage());
        }

        // Record performance metrics
        let track_duration = track_start.elapsed();
        if let Ok(mut perf) = self.performance_tracker.lock() {
            perf.record_allocation_time(track_duration);
        }
    }

    /// Tracks a memory deallocation
    ///
    /// # Arguments
    ///
    /// * `handle` - Memory handle for the deallocation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::MemoryTracker;
    ///
    /// let tracker = MemoryTracker::new(Default::default())?;
    /// // Assume we have a memory handle from pool allocation
    /// // tracker.track_deallocation(&handle);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn track_deallocation(&self, handle: &MemoryHandle) {
        let track_start = Instant::now();
        let handle_id = handle.id();

        #[cfg(feature = "tracing")]
        debug!("Tracking deallocation of handle {}", handle_id);

        // Find and remove the allocation
        let allocation_info = if let Ok(mut active) = self.active_allocations.write() {
            // Find allocation by handle ID (we need to search since we don't have AllocationId)
            let allocation_id = active
                .iter()
                .find(|(_, info)| {
                    // We need a way to match handle to allocation
                    // For now, we'll use a simple approach based on size and device
                    info.size == handle.size()
                        && info.device_type == format!("{:?}", handle.device())
                })
                .map(|(id, _)| *id);

            if let Some(id) = allocation_id {
                active.remove(&id)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(mut info) = allocation_info {
            let size = info.size;
            let device_type = info.device_type.clone();

            // Mark as deallocated
            info.mark_deallocated();

            // Update global statistics
            if let Ok(mut stats) = self.stats.write() {
                if let Some(device_stats) = stats.device_stats.get_mut(&device_type) {
                    device_stats.record_deallocation(size);
                }
                stats.update(self.start_time.elapsed());
            }

            // Update timeline if enabled
            if let Some(timeline) = &self.timeline {
                if let Ok(timeline) = timeline.lock() {
                    let event = AllocationEvent::Deallocation {
                        id: info.id,
                        size,
                        device_type: device_type.clone(),
                        timestamp: SystemTime::now(),
                    };
                    timeline.add_event(event);
                }
            }

            // Update pattern analyzer if enabled
            if let Some(analyzer) = &self.pattern_analyzer {
                if let Ok(analyzer) = analyzer.lock() {
                    analyzer.record_deallocation(info);
                }
            }

            // Update pressure detector if enabled
            if let Some(detector) = &self.pressure_detector {
                detector.update_memory_usage(self.get_current_memory_usage());
            }
        }

        // Record performance metrics
        let track_duration = track_start.elapsed();
        if let Ok(mut perf) = self.performance_tracker.lock() {
            perf.record_deallocation_time(track_duration);
        }
    }

    /// Returns detailed memory metrics
    ///
    /// # Returns
    ///
    /// Comprehensive memory tracking metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::MemoryTracker;
    ///
    /// let tracker = MemoryTracker::new(Default::default())?;
    /// let metrics = tracker.get_detailed_metrics();
    /// println!("Current memory usage: {} bytes", metrics.current_memory_usage);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_detailed_metrics(&self) -> DetailedMemoryMetrics {
        let global_stats = self
            .stats
            .read()
            .map(|stats| stats.clone())
            .unwrap_or_default();

        let pressure_level = self.get_pressure_level();

        let active_allocations = self
            .active_allocations
            .read()
            .map(|active| active.len())
            .unwrap_or(0);

        let current_memory_usage = self.get_current_memory_usage();
        let peak_memory_usage = global_stats.peak_bytes_allocated();

        let device_usage = global_stats
            .device_stats
            .iter()
            .map(|(device, stats)| (device.clone(), stats.current_bytes_allocated))
            .collect();

        let recent_patterns = if let Some(analyzer) = &self.pattern_analyzer {
            analyzer
                .lock()
                .map(|analyzer| analyzer.get_recent_patterns())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let performance = self
            .performance_tracker
            .lock()
            .map(|perf| perf.get_metrics())
            .unwrap_or_default();

        let tracking_overhead = self.calculate_tracking_overhead();

        DetailedMemoryMetrics {
            global_stats,
            pressure_level,
            active_allocations,
            current_memory_usage,
            peak_memory_usage,
            device_usage,
            recent_patterns,
            performance,
            tracking_overhead,
            timestamp: SystemTime::now(),
        }
    }

    /// Returns the current memory pressure level
    ///
    /// # Returns
    ///
    /// Current memory pressure level
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::{MemoryTracker, MemoryPressureLevel};
    ///
    /// let tracker = MemoryTracker::new(Default::default())?;
    /// match tracker.get_pressure_level() {
    ///     MemoryPressureLevel::None => println!("Memory usage is normal"),
    ///     MemoryPressureLevel::Low => println!("Low memory pressure detected"),
    ///     MemoryPressureLevel::Critical => println!("Critical memory pressure!"),
    ///     _ => println!("Memory pressure detected"),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_pressure_level(&self) -> MemoryPressureLevel {
        if let Some(detector) = &self.pressure_detector {
            detector.get_current_level()
        } else {
            MemoryPressureLevel::None
        }
    }

    /// Registers a callback for memory pressure events
    ///
    /// # Arguments
    ///
    /// * `callback` - Callback function to be called on pressure events
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::{MemoryTracker, MemoryPressureLevel};
    ///
    /// let tracker = MemoryTracker::new(Default::default())?;
    /// tracker.register_pressure_callback(Box::new(|level| {
    ///     println!("Memory pressure changed to: {:?}", level);
    /// }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register_pressure_callback(&self, callback: PressureCallback) {
        if let Some(detector) = &self.pressure_detector {
            detector.register_callback(callback);
        }
    }

    // Private helper methods

    fn generate_allocation_id(&self) -> AllocationId {
        let id = self
            .next_allocation_id
            .lock()
            .map(|mut counter| {
                let id = *counter;
                *counter += 1;
                id
            })
            .unwrap_or(0);
        AllocationId::new(id)
    }

    fn get_current_memory_usage(&self) -> u64 {
        self.stats
            .read()
            .map(|stats| stats.total_bytes_allocated())
            .unwrap_or(0)
    }

    fn capture_stack_trace(&self) -> Option<Vec<String>> {
        // Stack trace capture would be implemented here
        // This is a placeholder for the actual implementation
        #[cfg(feature = "backtrace")]
        {
            // Use backtrace crate to capture stack trace
            None // Placeholder
        }
        #[cfg(not(feature = "backtrace"))]
        {
            None
        }
    }

    fn calculate_tracking_overhead(&self) -> TrackingOverhead {
        let active_allocations_size = self
            .active_allocations
            .read()
            .map(|active| active.len() * std::mem::size_of::<AllocationInfo>())
            .unwrap_or(0);

        let timeline_size = if let Some(timeline) = &self.timeline {
            timeline
                .lock()
                .map(|timeline| timeline.estimated_memory_usage())
                .unwrap_or(0)
        } else {
            0
        };

        let pattern_analyzer_size = if let Some(analyzer) = &self.pattern_analyzer {
            analyzer
                .lock()
                .map(|analyzer| analyzer.estimated_memory_usage())
                .unwrap_or(0)
        } else {
            0
        };

        let total_overhead = active_allocations_size + timeline_size + pattern_analyzer_size;

        let cpu_overhead = self
            .performance_tracker
            .lock()
            .map(|perf| perf.calculate_cpu_overhead_percentage(self.start_time.elapsed()))
            .unwrap_or(0.0);

        TrackingOverhead {
            memory_overhead_bytes: total_overhead as u64,
            cpu_overhead_percentage: cpu_overhead,
            tracking_structures_count: 3, // active_allocations, timeline, pattern_analyzer
            largest_structure_size_bytes: [
                active_allocations_size,
                timeline_size,
                pattern_analyzer_size,
            ]
            .iter()
            .max()
            .copied()
            .unwrap_or(0) as u64,
        }
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            allocation_times: Vec::new(),
            deallocation_times: Vec::new(),
            total_operations: 0,
            failed_operations: 0,
            overhead_start_time: Instant::now(),
            total_tracking_time: Duration::ZERO,
        }
    }

    fn record_allocation_time(&mut self, duration: Duration) {
        self.allocation_times.push(duration);
        self.total_operations += 1;
        self.total_tracking_time += duration;

        // Keep only recent measurements to avoid unbounded growth
        if self.allocation_times.len() > 1000 {
            self.allocation_times.drain(0..500);
        }
    }

    fn record_deallocation_time(&mut self, duration: Duration) {
        self.deallocation_times.push(duration);
        self.total_operations += 1;
        self.total_tracking_time += duration;

        // Keep only recent measurements to avoid unbounded growth
        if self.deallocation_times.len() > 1000 {
            self.deallocation_times.drain(0..500);
        }
    }

    fn get_metrics(&self) -> PerformanceMetrics {
        let avg_allocation_time = if !self.allocation_times.is_empty() {
            self.allocation_times.iter().sum::<Duration>().as_nanos() as u64
                / self.allocation_times.len() as u64
        } else {
            0
        };

        let avg_deallocation_time = if !self.deallocation_times.is_empty() {
            self.deallocation_times.iter().sum::<Duration>().as_nanos() as u64
                / self.deallocation_times.len() as u64
        } else {
            0
        };

        let max_allocation_time = self
            .allocation_times
            .iter()
            .max()
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let max_deallocation_time = self
            .deallocation_times
            .iter()
            .max()
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        PerformanceMetrics {
            avg_track_allocation_time_ns: avg_allocation_time,
            avg_track_deallocation_time_ns: avg_deallocation_time,
            max_track_allocation_time_ns: max_allocation_time,
            max_track_deallocation_time_ns: max_deallocation_time,
            total_tracking_operations: self.total_operations,
            failed_tracking_operations: self.failed_operations,
        }
    }

    fn calculate_cpu_overhead_percentage(&self, total_elapsed: Duration) -> f64 {
        if total_elapsed.as_nanos() == 0 {
            return 0.0;
        }

        let tracking_percentage =
            (self.total_tracking_time.as_nanos() as f64 / total_elapsed.as_nanos() as f64) * 100.0;
        tracking_percentage.min(100.0) // Cap at 100%
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_track_allocation_time_ns: 0,
            avg_track_deallocation_time_ns: 0,
            max_track_allocation_time_ns: 0,
            max_track_deallocation_time_ns: 0,
            total_tracking_operations: 0,
            failed_tracking_operations: 0,
        }
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for MemoryTracker {}
unsafe impl Sync for MemoryTracker {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::handle::{CpuMemoryMetadata, MemoryHandle, PoolType};
    use std::ptr::NonNull;

    #[test]
    fn test_memory_tracker_creation() {
        let config = TrackingConfig::standard();
        let tracker = MemoryTracker::new(config).unwrap();

        let metrics = tracker.get_detailed_metrics();
        assert_eq!(metrics.active_allocations, 0);
        assert_eq!(metrics.current_memory_usage, 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let config = TrackingConfig::standard();
        let tracker = MemoryTracker::new(config).unwrap();
        let device = get_cpu_device();

        // Create a mock memory handle
        let ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        let handle = unsafe {
            MemoryHandle::new_cpu(
                1,
                ptr,
                1024,
                16,
                device.clone(),
                PoolType::SmallBlock,
                CpuMemoryMetadata {
                    page_aligned: false,
                    locked: false,
                    numa_node: None,
                },
            )
        };

        tracker.track_allocation(&handle, 1024, &device);

        let metrics = tracker.get_detailed_metrics();
        assert_eq!(metrics.active_allocations, 1);
        assert_eq!(metrics.current_memory_usage, 1024);
    }

    #[test]
    fn test_deallocation_tracking() {
        let config = TrackingConfig::standard();
        let tracker = MemoryTracker::new(config).unwrap();
        let device = get_cpu_device();

        // Create a mock memory handle
        let ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        let handle = unsafe {
            MemoryHandle::new_cpu(
                1,
                ptr,
                1024,
                16,
                device.clone(),
                PoolType::SmallBlock,
                CpuMemoryMetadata {
                    page_aligned: false,
                    locked: false,
                    numa_node: None,
                },
            )
        };

        tracker.track_allocation(&handle, 1024, &device);
        tracker.track_deallocation(&handle);

        let metrics = tracker.get_detailed_metrics();
        assert_eq!(metrics.active_allocations, 0);
    }

    #[test]
    fn test_performance_tracking() {
        let mut perf_tracker = PerformanceTracker::new();

        perf_tracker.record_allocation_time(Duration::from_nanos(1000));
        perf_tracker.record_allocation_time(Duration::from_nanos(2000));
        perf_tracker.record_deallocation_time(Duration::from_nanos(500));

        let metrics = perf_tracker.get_metrics();
        assert_eq!(metrics.avg_track_allocation_time_ns, 1500);
        assert_eq!(metrics.avg_track_deallocation_time_ns, 500);
        assert_eq!(metrics.max_track_allocation_time_ns, 2000);
        assert_eq!(metrics.total_tracking_operations, 3);
    }
}
