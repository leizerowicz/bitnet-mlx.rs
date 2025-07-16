//! Memory Profiler for Debugging and Leak Detection
//!
//! This module provides comprehensive memory profiling capabilities including
//! leak detection, allocation lifetime analysis, and debugging utilities.

use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info};

use super::{TrackingResult, AllocationId, AllocationInfo};

/// Memory profiler for debugging and leak detection
pub struct MemoryProfiler {
    /// Whether profiling is currently active
    is_profiling: Arc<RwLock<bool>>,
    /// Profiling start time
    start_time: Arc<RwLock<Option<Instant>>>,
    /// All allocations tracked during profiling
    tracked_allocations: Arc<RwLock<HashMap<AllocationId, AllocationInfo>>>,
    /// Allocation lifetime history
    lifetime_history: Arc<Mutex<Vec<AllocationLifetime>>>,
    /// Potential memory leaks
    potential_leaks: Arc<Mutex<Vec<LeakCandidate>>>,
    /// Profiling configuration
    config: ProfilingConfig,
    /// Allocation patterns for analysis
    allocation_patterns: Arc<Mutex<HashMap<String, AllocationPattern>>>,
    /// Memory snapshots for comparison
    snapshots: Arc<Mutex<Vec<MemorySnapshot>>>,
}

/// Configuration for memory profiling
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Minimum age for leak detection
    pub leak_detection_threshold: Duration,
    /// Whether to capture detailed stack traces
    pub capture_stack_traces: bool,
    /// Whether to track allocation patterns
    pub track_allocation_patterns: bool,
    /// Maximum number of snapshots to keep
    pub max_snapshots: usize,
    /// Interval for automatic leak detection
    pub leak_detection_interval: Duration,
}

/// Information about an allocation's lifetime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationLifetime {
    /// Allocation information
    pub allocation: AllocationInfo,
    /// Time when allocation was made
    pub allocation_time: SystemTime,
    /// Time when allocation was freed (if freed)
    pub deallocation_time: Option<SystemTime>,
    /// Duration the allocation was alive
    pub lifetime_duration: Option<Duration>,
    /// Whether this allocation was considered a leak
    pub was_leak: bool,
}

/// Candidate for a memory leak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakCandidate {
    /// Allocation information
    pub allocation: AllocationInfo,
    /// Age of the allocation
    pub age: Duration,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Reason why this is considered a leak candidate
    pub reason: String,
    /// When this leak was detected
    pub detected_at: SystemTime,
}

/// Pattern of memory allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Size range for this pattern
    pub size_range: (usize, usize),
    /// Device type for this pattern
    pub device_type: String,
    /// Number of allocations matching this pattern
    pub allocation_count: u64,
    /// Average lifetime for allocations in this pattern
    pub average_lifetime: Duration,
    /// Whether this pattern shows potential issues
    pub is_problematic: bool,
    /// Description of the pattern
    pub description: String,
}

/// Memory snapshot for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Snapshot identifier
    pub id: String,
    /// When the snapshot was taken
    pub timestamp: SystemTime,
    /// Total memory allocated at snapshot time
    pub total_allocated: u64,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Memory usage by device
    pub device_usage: HashMap<String, u64>,
    /// Memory usage by size category
    pub size_distribution: HashMap<String, u64>,
    /// Top allocations by size
    pub top_allocations: Vec<AllocationInfo>,
}

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Profiling session duration
    pub session_duration: Duration,
    /// Total allocations tracked
    pub total_allocations: usize,
    /// Total deallocations tracked
    pub total_deallocations: usize,
    /// Current active allocations
    pub active_allocations: usize,
    /// Detected memory leaks
    pub detected_leaks: Vec<LeakCandidate>,
    /// Allocation lifetime statistics
    pub lifetime_stats: LifetimeStatistics,
    /// Identified allocation patterns
    pub allocation_patterns: Vec<AllocationPattern>,
    /// Memory usage trends
    pub usage_trends: UsageTrends,
    /// Performance impact of profiling
    pub profiling_overhead: ProfilingOverhead,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
}

/// Memory leak detection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakReport {
    /// Number of potential leaks detected
    pub leak_count: usize,
    /// Total memory potentially leaked
    pub total_leaked_bytes: u64,
    /// Leaks by confidence level
    pub leaks_by_confidence: HashMap<String, usize>,
    /// Leaks by device type
    pub leaks_by_device: HashMap<String, usize>,
    /// Most significant leaks
    pub top_leaks: Vec<LeakCandidate>,
    /// Leak detection timestamp
    pub detection_timestamp: SystemTime,
}

/// Statistics about allocation lifetimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifetimeStatistics {
    /// Average allocation lifetime
    pub average_lifetime: Duration,
    /// Median allocation lifetime
    pub median_lifetime: Duration,
    /// Shortest allocation lifetime
    pub shortest_lifetime: Duration,
    /// Longest allocation lifetime
    pub longest_lifetime: Duration,
    /// Standard deviation of lifetimes
    pub lifetime_std_dev: Duration,
    /// Lifetime distribution by ranges
    pub lifetime_distribution: HashMap<String, usize>,
}

/// Memory usage trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTrends {
    /// Memory usage growth rate (bytes per second)
    pub growth_rate: f64,
    /// Peak memory usage during profiling
    pub peak_usage: u64,
    /// Average memory usage during profiling
    pub average_usage: u64,
    /// Memory usage volatility (standard deviation)
    pub usage_volatility: f64,
    /// Trend direction (positive = growing, negative = shrinking)
    pub trend_direction: f64,
}

/// Profiling overhead information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingOverhead {
    /// Memory overhead in bytes
    pub memory_overhead_bytes: u64,
    /// CPU overhead percentage
    pub cpu_overhead_percentage: f64,
    /// Number of profiling operations
    pub profiling_operations: u64,
    /// Average time per profiling operation
    pub avg_operation_time_ns: u64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            max_tracked_allocations: 100_000,
            leak_detection_threshold: Duration::from_secs(300), // 5 minutes
            capture_stack_traces: false,
            track_allocation_patterns: true,
            max_snapshots: 100,
            leak_detection_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

impl MemoryProfiler {
    /// Creates a new memory profiler
    ///
    /// # Arguments
    ///
    /// * `config` - Profiling configuration
    ///
    /// # Returns
    ///
    /// A Result containing the new profiler or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::profiler::{MemoryProfiler, ProfilingConfig};
    ///
    /// let config = ProfilingConfig::default();
    /// let profiler = MemoryProfiler::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: ProfilingConfig) -> TrackingResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Creating memory profiler with config: max_tracked={}", config.max_tracked_allocations);

        Ok(Self {
            is_profiling: Arc::new(RwLock::new(false)),
            start_time: Arc::new(RwLock::new(None)),
            tracked_allocations: Arc::new(RwLock::new(HashMap::new())),
            lifetime_history: Arc::new(Mutex::new(Vec::new())),
            potential_leaks: Arc::new(Mutex::new(Vec::new())),
            config,
            allocation_patterns: Arc::new(Mutex::new(HashMap::new())),
            snapshots: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Starts memory profiling
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::profiler::MemoryProfiler;
    ///
    /// let profiler = MemoryProfiler::new(Default::default())?;
    /// profiler.start_profiling();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start_profiling(&self) {
        #[cfg(feature = "tracing")]
        info!("Starting memory profiling session");

        {
            let mut is_profiling = self.is_profiling.write().unwrap();
            *is_profiling = true;
        }

        {
            let mut start_time = self.start_time.write().unwrap();
            *start_time = Some(Instant::now());
        }

        // Clear previous data
        {
            let mut tracked = self.tracked_allocations.write().unwrap();
            tracked.clear();
        }

        {
            let mut history = self.lifetime_history.lock().unwrap();
            history.clear();
        }

        {
            let mut leaks = self.potential_leaks.lock().unwrap();
            leaks.clear();
        }
    }

    /// Stops memory profiling and returns a comprehensive report
    ///
    /// # Returns
    ///
    /// Comprehensive profiling report
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::profiler::MemoryProfiler;
    ///
    /// let profiler = MemoryProfiler::new(Default::default())?;
    /// profiler.start_profiling();
    /// // ... perform memory operations ...
    /// let report = profiler.stop_profiling();
    /// println!("Detected {} potential leaks", report.detected_leaks.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stop_profiling(&self) -> ProfilingReport {
        #[cfg(feature = "tracing")]
        info!("Stopping memory profiling session");

        {
            let mut is_profiling = self.is_profiling.write().unwrap();
            *is_profiling = false;
        }

        let session_duration = {
            let start_time = self.start_time.read().unwrap();
            start_time.map(|start| start.elapsed()).unwrap_or(Duration::ZERO)
        };

        // Perform final leak detection
        self.detect_leaks();

        // Generate comprehensive report
        self.generate_profiling_report(session_duration)
    }

    /// Records an allocation for profiling
    ///
    /// # Arguments
    ///
    /// * `allocation` - Allocation information to record
    pub fn record_allocation(&self, allocation: AllocationInfo) {
        let is_profiling = {
            let profiling = self.is_profiling.read().unwrap();
            *profiling
        };

        if !is_profiling {
            return;
        }

        #[cfg(feature = "tracing")]
        debug!("Recording allocation {} for profiling", allocation.id.raw());

        // Check if we're at capacity
        let should_record = {
            let tracked = self.tracked_allocations.read().unwrap();
            tracked.len() < self.config.max_tracked_allocations
        };

        if should_record {
            // Record the allocation
            {
                let mut tracked = self.tracked_allocations.write().unwrap();
                tracked.insert(allocation.id, allocation.clone());
            }

            // Update allocation patterns if enabled
            if self.config.track_allocation_patterns {
                self.update_allocation_patterns(&allocation);
            }
        }
    }

    /// Records a deallocation for profiling
    ///
    /// # Arguments
    ///
    /// * `allocation` - Allocation information for the deallocated memory
    pub fn record_deallocation(&self, mut allocation: AllocationInfo) {
        let is_profiling = {
            let profiling = self.is_profiling.read().unwrap();
            *profiling
        };

        if !is_profiling {
            return;
        }

        #[cfg(feature = "tracing")]
        debug!("Recording deallocation {} for profiling", allocation.id.raw());

        // Remove from tracked allocations
        let was_tracked = {
            let mut tracked = self.tracked_allocations.write().unwrap();
            tracked.remove(&allocation.id).is_some()
        };

        if was_tracked {
            // Calculate lifetime
            allocation.mark_deallocated();
            let lifetime_duration = allocation.age();

            // Create lifetime record
            let lifetime = AllocationLifetime {
                allocation: allocation.clone(),
                allocation_time: allocation.timestamp,
                deallocation_time: Some(SystemTime::now()),
                lifetime_duration: Some(lifetime_duration),
                was_leak: false,
            };

            // Add to lifetime history
            {
                let mut history = self.lifetime_history.lock().unwrap();
                history.push(lifetime);

                // Keep history bounded
                if history.len() > self.config.max_tracked_allocations {
                    history.drain(0..1000); // Remove oldest 1000 entries
                }
            }
        }
    }

    /// Detects potential memory leaks
    ///
    /// # Returns
    ///
    /// Report of detected memory leaks
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::profiler::MemoryProfiler;
    ///
    /// let profiler = MemoryProfiler::new(Default::default())?;
    /// profiler.start_profiling();
    /// // ... perform memory operations ...
    /// let leak_report = profiler.detect_leaks();
    /// if leak_report.leak_count > 0 {
    ///     println!("WARNING: {} potential leaks detected!", leak_report.leak_count);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn detect_leaks(&self) -> LeakReport {
        #[cfg(feature = "tracing")]
        debug!("Running memory leak detection");

        let mut leak_candidates = Vec::new();
        let now = SystemTime::now();

        // Check tracked allocations for potential leaks
        {
            let tracked = self.tracked_allocations.read().unwrap();
            for allocation in tracked.values() {
                let age = allocation.age();
                
                if age >= self.config.leak_detection_threshold {
                    let confidence = self.calculate_leak_confidence(allocation, age);
                    let reason = self.determine_leak_reason(allocation, age);
                    
                    let candidate = LeakCandidate {
                        allocation: allocation.clone(),
                        age,
                        confidence,
                        reason,
                        detected_at: now,
                    };
                    
                    leak_candidates.push(candidate);
                }
            }
        }

        // Sort by confidence (highest first)
        leak_candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Update potential leaks
        {
            let mut leaks = self.potential_leaks.lock().unwrap();
            leaks.extend(leak_candidates.iter().cloned());
            
            // Keep only recent leaks
            leaks.retain(|leak| {
                now.duration_since(leak.detected_at).unwrap_or(Duration::MAX) < Duration::from_secs(3600)
            });
        }

        // Generate leak report
        self.generate_leak_report(leak_candidates, now)
    }

    /// Takes a memory snapshot for later comparison
    ///
    /// # Arguments
    ///
    /// * `snapshot_id` - Identifier for the snapshot
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::profiler::MemoryProfiler;
    ///
    /// let profiler = MemoryProfiler::new(Default::default())?;
    /// profiler.start_profiling();
    /// profiler.take_snapshot("before_operation".to_string());
    /// // ... perform memory operations ...
    /// profiler.take_snapshot("after_operation".to_string());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn take_snapshot(&self, snapshot_id: String) {
        #[cfg(feature = "tracing")]
        debug!("Taking memory snapshot: {}", snapshot_id);

        let tracked = self.tracked_allocations.read().unwrap();
        
        let total_allocated = tracked.values().map(|a| a.size as u64).sum();
        let active_allocations = tracked.len();
        
        let mut device_usage = HashMap::new();
        let mut size_distribution = HashMap::new();
        
        for allocation in tracked.values() {
            // Device usage
            *device_usage.entry(allocation.device_type.clone()).or_insert(0) += allocation.size as u64;
            
            // Size distribution
            let size_category = self.categorize_allocation_size(allocation.size);
            *size_distribution.entry(size_category).or_insert(0) += allocation.size as u64;
        }
        
        // Get top allocations by size
        let mut allocations: Vec<_> = tracked.values().cloned().collect();
        allocations.sort_by(|a, b| b.size.cmp(&a.size));
        let top_allocations = allocations.into_iter().take(10).collect();
        
        let snapshot = MemorySnapshot {
            id: snapshot_id,
            timestamp: SystemTime::now(),
            total_allocated,
            active_allocations,
            device_usage,
            size_distribution,
            top_allocations,
        };
        
        {
            let mut snapshots = self.snapshots.lock().unwrap();
            snapshots.push(snapshot);
            
            // Keep snapshots bounded
            if snapshots.len() > self.config.max_snapshots {
                snapshots.drain(0..10); // Remove oldest 10 snapshots
            }
        }
    }

    /// Returns all memory snapshots
    ///
    /// # Returns
    ///
    /// Vector of all memory snapshots
    pub fn get_snapshots(&self) -> Vec<MemorySnapshot> {
        let snapshots = self.snapshots.lock().unwrap();
        snapshots.clone()
    }

    /// Estimates memory usage of the profiler itself
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let tracked_size = {
            let tracked = self.tracked_allocations.read().unwrap();
            tracked.len() * std::mem::size_of::<AllocationInfo>()
        };
        
        let history_size = {
            let history = self.lifetime_history.lock().unwrap();
            history.len() * std::mem::size_of::<AllocationLifetime>()
        };
        
        let leaks_size = {
            let leaks = self.potential_leaks.lock().unwrap();
            leaks.len() * std::mem::size_of::<LeakCandidate>()
        };
        
        let patterns_size = {
            let patterns = self.allocation_patterns.lock().unwrap();
            patterns.len() * std::mem::size_of::<AllocationPattern>()
        };
        
        let snapshots_size = {
            let snapshots = self.snapshots.lock().unwrap();
            snapshots.len() * std::mem::size_of::<MemorySnapshot>()
        };
        
        tracked_size + history_size + leaks_size + patterns_size + snapshots_size
    }

    // Private helper methods

    fn update_allocation_patterns(&self, allocation: &AllocationInfo) {
        let pattern_id = format!("{}_{}", 
            allocation.device_type,
            self.categorize_allocation_size(allocation.size)
        );
        
        let mut patterns = self.allocation_patterns.lock().unwrap();
        let pattern = patterns.entry(pattern_id.clone()).or_insert_with(|| {
            AllocationPattern {
                pattern_id: pattern_id.clone(),
                size_range: self.get_size_range(allocation.size),
                device_type: allocation.device_type.clone(),
                allocation_count: 0,
                average_lifetime: Duration::ZERO,
                is_problematic: false,
                description: format!("Allocations on {} in size range {:?}", 
                    allocation.device_type, self.get_size_range(allocation.size)),
            }
        });
        
        pattern.allocation_count += 1;
    }

    fn calculate_leak_confidence(&self, allocation: &AllocationInfo, age: Duration) -> f64 {
        let mut confidence = 0.0;
        
        // Age factor (older = more likely to be a leak)
        let age_seconds = age.as_secs_f64();
        confidence += (age_seconds / 3600.0).min(0.5); // Max 0.5 for age
        
        // Size factor (larger allocations are more concerning)
        if allocation.size > 1024 * 1024 * 10 { // > 10MB
            confidence += 0.3;
        } else if allocation.size > 1024 * 1024 { // > 1MB
            confidence += 0.2;
        } else if allocation.size > 1024 * 100 { // > 100KB
            confidence += 0.1;
        }
        
        // Pattern factor (check if this allocation type typically has short lifetimes)
        let pattern_id = format!("{}_{}", 
            allocation.device_type,
            self.categorize_allocation_size(allocation.size)
        );
        
        if let Ok(patterns) = self.allocation_patterns.lock() {
            if let Some(pattern) = patterns.get(&pattern_id) {
                if pattern.average_lifetime < Duration::from_secs(60) && age > Duration::from_secs(300) {
                    confidence += 0.2; // This type usually has short lifetime but this one is old
                }
            }
        }
        
        confidence.min(1.0)
    }

    fn determine_leak_reason(&self, allocation: &AllocationInfo, age: Duration) -> String {
        let mut reasons = Vec::new();
        
        if age > Duration::from_secs(3600) {
            reasons.push("allocation is very old (>1 hour)".to_string());
        } else if age > Duration::from_secs(1800) {
            reasons.push("allocation is old (>30 minutes)".to_string());
        }
        
        if allocation.size > 1024 * 1024 * 10 {
            reasons.push("large allocation (>10MB)".to_string());
        }
        
        if reasons.is_empty() {
            "allocation exceeds leak detection threshold".to_string()
        } else {
            reasons.join(", ")
        }
    }

    fn categorize_allocation_size(&self, size: usize) -> String {
        match size {
            0..=1024 => "tiny".to_string(),
            1025..=65536 => "small".to_string(),
            65537..=1048576 => "medium".to_string(),
            1048577..=16777216 => "large".to_string(),
            _ => "huge".to_string(),
        }
    }

    fn get_size_range(&self, size: usize) -> (usize, usize) {
        match size {
            0..=1024 => (0, 1024),
            1025..=65536 => (1025, 65536),
            65537..=1048576 => (65537, 1048576),
            1048577..=16777216 => (1048577, 16777216),
            _ => (16777217, usize::MAX),
        }
    }

    fn generate_leak_report(&self, leak_candidates: Vec<LeakCandidate>, timestamp: SystemTime) -> LeakReport {
        let leak_count = leak_candidates.len();
        let total_leaked_bytes = leak_candidates.iter().map(|l| l.allocation.size as u64).sum();
        
        let mut leaks_by_confidence = HashMap::new();
        let mut leaks_by_device = HashMap::new();
        
        for leak in &leak_candidates {
            let confidence_category = if leak.confidence >= 0.8 {
                "high"
            } else if leak.confidence >= 0.6 {
                "medium"
            } else {
                "low"
            };
            *leaks_by_confidence.entry(confidence_category.to_string()).or_insert(0) += 1;
            *leaks_by_device.entry(leak.allocation.device_type.clone()).or_insert(0) += 1;
        }
        
        let top_leaks = leak_candidates.into_iter().take(10).collect();
        
        LeakReport {
            leak_count,
            total_leaked_bytes,
            leaks_by_confidence,
            leaks_by_device,
            top_leaks,
            detection_timestamp: timestamp,
        }
    }

    fn generate_profiling_report(&self, session_duration: Duration) -> ProfilingReport {
        let tracked = self.tracked_allocations.read().unwrap();
        let history = self.lifetime_history.lock().unwrap();
        let leaks = self.potential_leaks.lock().unwrap();
        let patterns = self.allocation_patterns.lock().unwrap();
        
        let total_allocations = history.len() + tracked.len();
        let total_deallocations = history.len();
        let active_allocations = tracked.len();
        
        let detected_leaks = leaks.clone();
        let lifetime_stats = self.calculate_lifetime_statistics(&history);
        let allocation_patterns: Vec<AllocationPattern> = patterns.values().cloned().collect();
        let usage_trends = self.calculate_usage_trends(session_duration);
        let profiling_overhead = self.calculate_profiling_overhead(session_duration);
        let recommendations = self.generate_recommendations(&detected_leaks, &allocation_patterns);
        
        ProfilingReport {
            session_duration,
            total_allocations,
            total_deallocations,
            active_allocations,
            detected_leaks,
            lifetime_stats,
            allocation_patterns: allocation_patterns.to_vec(),
            usage_trends,
            profiling_overhead,
            recommendations,
        }
    }

    fn calculate_lifetime_statistics(&self, history: &[AllocationLifetime]) -> LifetimeStatistics {
        if history.is_empty() {
            return LifetimeStatistics {
                average_lifetime: Duration::ZERO,
                median_lifetime: Duration::ZERO,
                shortest_lifetime: Duration::ZERO,
                longest_lifetime: Duration::ZERO,
                lifetime_std_dev: Duration::ZERO,
                lifetime_distribution: HashMap::new(),
            };
        }
        
        let mut lifetimes: Vec<Duration> = history.iter()
            .filter_map(|l| l.lifetime_duration)
            .collect();
        lifetimes.sort();
        
        let average_lifetime = lifetimes.iter().sum::<Duration>() / lifetimes.len() as u32;
        let median_lifetime = lifetimes[lifetimes.len() / 2];
        let shortest_lifetime = lifetimes.first().copied().unwrap_or(Duration::ZERO);
        let longest_lifetime = lifetimes.last().copied().unwrap_or(Duration::ZERO);
        
        // Calculate standard deviation
        let variance: f64 = lifetimes.iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - average_lifetime.as_secs_f64();
                diff * diff
            })
            .sum::<f64>() / lifetimes.len() as f64;
        let lifetime_std_dev = Duration::from_secs_f64(variance.sqrt());
        
        // Calculate distribution
        let mut lifetime_distribution = HashMap::new();
        for lifetime in &lifetimes {
            let category = match lifetime.as_secs() {
                0..=1 => "immediate",
                2..=60 => "short",
                61..=3600 => "medium",
                3601..=86400 => "long",
                _ => "very_long",
            };
            *lifetime_distribution.entry(category.to_string()).or_insert(0) += 1;
        }
        
        LifetimeStatistics {
            average_lifetime,
            median_lifetime,
            shortest_lifetime,
            longest_lifetime,
            lifetime_std_dev,
            lifetime_distribution,
        }
    }

    fn calculate_usage_trends(&self, _session_duration: Duration) -> UsageTrends {
        // Simplified implementation - in practice would analyze snapshots over time
        UsageTrends {
            growth_rate: 0.0,
            peak_usage: 0,
            average_usage: 0,
            usage_volatility: 0.0,
            trend_direction: 0.0,
        }
    }

    fn calculate_profiling_overhead(&self, session_duration: Duration) -> ProfilingOverhead {
        let memory_overhead_bytes = self.estimated_memory_usage() as u64;
        let profiling_operations = {
            let tracked = self.tracked_allocations.read().unwrap();
            let history = self.lifetime_history.lock().unwrap();
            tracked.len() as u64 + history.len() as u64
        };
        
        ProfilingOverhead {
            memory_overhead_bytes,
            cpu_overhead_percentage: 1.0, // Estimated
            profiling_operations,
            avg_operation_time_ns: if profiling_operations > 0 {
                (session_duration.as_nanos() as u64) / profiling_operations
            } else {
                0
            },
        }
    }

    fn generate_recommendations(&self, leaks: &[LeakCandidate], patterns: &[AllocationPattern]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if !leaks.is_empty() {
            recommendations.push(format!("Investigate {} potential memory leaks", leaks.len()));
        }
        
        for pattern in patterns {
            if pattern.is_problematic {
                recommendations.push(format!("Review allocation pattern: {}", pattern.description));
            }
        }
        
        recommendations
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for MemoryProfiler {}
unsafe impl Sync for MemoryProfiler {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilingConfig::default();
        let profiler = MemoryProfiler::new(config).unwrap();
        
        assert_eq!(profiler.estimated_memory_usage(), 0);
    }

    #[test]
    fn test_profiling_session() {
        let config = ProfilingConfig::default();
        let profiler = MemoryProfiler::new(config).unwrap();
        
        profiler.start_profiling();
        
        // Create a mock allocation
        let allocation = AllocationInfo {
            id: AllocationId::new(1),
            size: 1024,
            alignment: 16,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now(),
            elapsed: Duration::from_millis(100),
            stack_trace: None,
            pool_type: "SmallBlock".to_string(),
            is_active: true,
        };
        
        profiler.record_allocation(allocation.clone());
        profiler.record_deallocation(allocation);
        
        let report = profiler.stop_profiling();
        assert_eq!(report.total_allocations, 1);
        assert_eq!(report.total_deallocations, 1);
    }

    #[test]
    fn test_leak_detection() {
        let mut config = ProfilingConfig::default();
        config.leak_detection_threshold = Duration::from_millis(10);
        
        let profiler = MemoryProfiler::new(config).unwrap();
        profiler.start_profiling();
        
        // Create a long-lived allocation
        let allocation = AllocationInfo {
            id: AllocationId::new(1),
            size: 1024 * 1024, // 1MB
            alignment: 16,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now() - Duration::from_secs(1), // Old allocation
            elapsed: Duration::from_secs(1),
            stack_trace: None,
            pool_type: "SmallBlock".to_string(),
            is_active: true,
        };
        
        profiler.record_allocation(allocation);
        
        // Wait a bit to ensure it's old enough
        std::thread::sleep(Duration::from_millis(20));
        
        let leak_report = profiler.detect_leaks();
        assert!(leak_report.leak_count > 0);
    }

    #[test]
    fn test_snapshot_functionality() {
        let config = ProfilingConfig::default();
        let profiler = MemoryProfiler::new(config).unwrap();
        
        profiler.start_profiling();
        profiler.take_snapshot("test_snapshot".to_string());
        
        let snapshots = profiler.get_snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].id, "test_snapshot");
    }

    #[test]
    fn test_allocation_pattern_tracking() {
        let config = ProfilingConfig::default();
        let profiler = MemoryProfiler::new(config).unwrap();
        
        profiler.start_profiling();
        
        // Create allocations with similar patterns
        for i in 0..5 {
            let allocation = AllocationInfo {
                id: AllocationId::new(i),
                size: 1024, // Same size
                alignment: 16,
                device_type: "CPU".to_string(), // Same device
                timestamp: SystemTime::now(),
                elapsed: Duration::from_millis(100),
                stack_trace: None,
                pool_type: "SmallBlock".to_string(),
                is_active: true,
            };
            profiler.record_allocation(allocation);
        }
        
        let report = profiler.stop_profiling();
        assert!(!report.allocation_patterns.is_empty());
    }

    #[test]
    fn test_lifetime_statistics() {
        let lifetimes = vec![
            AllocationLifetime {
                allocation: AllocationInfo {
                    id: AllocationId::new(1),
                    size: 1024,
                    alignment: 16,
                    device_type: "CPU".to_string(),
                    timestamp: SystemTime::now(),
                    elapsed: Duration::from_millis(100),
                    stack_trace: None,
                    pool_type: "SmallBlock".to_string(),
                    is_active: false,
                },
                allocation_time: SystemTime::now(),
                deallocation_time: Some(SystemTime::now()),
                lifetime_duration: Some(Duration::from_millis(100)),
                was_leak: false,
            },
            AllocationLifetime {
                allocation: AllocationInfo {
                    id: AllocationId::new(2),
                    size: 2048,
                    alignment: 16,
                    device_type: "CPU".to_string(),
                    timestamp: SystemTime::now(),
                    elapsed: Duration::from_millis(200),
                    stack_trace: None,
                    pool_type: "SmallBlock".to_string(),
                    is_active: false,
                },
                allocation_time: SystemTime::now(),
                deallocation_time: Some(SystemTime::now()),
                lifetime_duration: Some(Duration::from_millis(200)),
                was_leak: false,
            },
        ];
        
        let config = ProfilingConfig::default();
        let profiler = MemoryProfiler::new(config).unwrap();
        
        let stats = profiler.calculate_lifetime_statistics(&lifetimes);
        assert_eq!(stats.average_lifetime, Duration::from_millis(150));
        assert_eq!(stats.shortest_lifetime, Duration::from_millis(100));
        assert_eq!(stats.longest_lifetime, Duration::from_millis(200));
    }
}