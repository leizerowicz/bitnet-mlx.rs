//! Allocation Timeline Tracking
//!
//! This module provides timeline tracking for memory allocations and deallocations,
//! enabling detailed analysis of memory usage patterns over time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

use super::AllocationId;

/// Timeline tracker for allocation and deallocation events
pub struct AllocationTimeline {
    /// Maximum number of events to keep in timeline
    max_entries: usize,
    /// Maximum age of events to keep
    retention_period: Duration,
    /// Timeline events in chronological order
    events: Arc<Mutex<VecDeque<TimelineEntry>>>,
    /// Timeline statistics
    stats: Arc<Mutex<TimelineStatistics>>,
}

/// Entry in the allocation timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// The allocation event
    pub event: AllocationEvent,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Time elapsed since timeline started
    pub elapsed_time: Duration,
}

/// Types of allocation events that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationEvent {
    /// Memory allocation event
    Allocation {
        /// Unique allocation identifier
        id: AllocationId,
        /// Size of allocation in bytes
        size: usize,
        /// Device type where allocation occurred
        device_type: String,
        /// Timestamp of allocation
        timestamp: SystemTime,
    },
    /// Memory deallocation event
    Deallocation {
        /// Unique allocation identifier
        id: AllocationId,
        /// Size of deallocation in bytes
        size: usize,
        /// Device type where deallocation occurred
        device_type: String,
        /// Timestamp of deallocation
        timestamp: SystemTime,
    },
    /// Memory pressure event
    PressureEvent {
        /// Pressure level
        level: String,
        /// Memory usage percentage
        usage_percentage: f64,
        /// Timestamp of pressure event
        timestamp: SystemTime,
    },
    /// Memory pool expansion event
    PoolExpansion {
        /// Pool type that expanded
        pool_type: String,
        /// Device type
        device_type: String,
        /// New pool size
        new_size: usize,
        /// Timestamp of expansion
        timestamp: SystemTime,
    },
}

/// Statistics about the allocation timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineStatistics {
    /// Total number of events recorded
    pub total_events: u64,
    /// Number of allocation events
    pub allocation_events: u64,
    /// Number of deallocation events
    pub deallocation_events: u64,
    /// Number of pressure events
    pub pressure_events: u64,
    /// Number of pool expansion events
    pub pool_expansion_events: u64,
    /// Timeline start time
    pub start_time: SystemTime,
    /// Timeline duration
    pub duration: Duration,
    /// Events per second rate
    pub events_per_second: f64,
    /// Peak events in a time window
    pub peak_events_per_window: u64,
    /// Memory usage trend over time
    pub memory_trend: MemoryTrend,
}

/// Memory usage trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrend {
    /// Overall trend direction (positive = growing, negative = shrinking)
    pub trend_direction: f64,
    /// Rate of change in bytes per second
    pub change_rate_bytes_per_sec: f64,
    /// Volatility measure (standard deviation of changes)
    pub volatility: f64,
    /// Peak memory usage observed
    pub peak_usage: u64,
    /// Minimum memory usage observed
    pub min_usage: u64,
    /// Average memory usage
    pub average_usage: f64,
}

/// Query parameters for timeline analysis
#[derive(Debug, Clone)]
pub struct TimelineQuery {
    /// Start time for query range
    pub start_time: Option<SystemTime>,
    /// End time for query range
    pub end_time: Option<SystemTime>,
    /// Filter by event type
    pub event_type: Option<String>,
    /// Filter by device type
    pub device_type: Option<String>,
    /// Minimum allocation size to include
    pub min_size: Option<usize>,
    /// Maximum allocation size to include
    pub max_size: Option<usize>,
    /// Maximum number of results to return
    pub limit: Option<usize>,
}

/// Result of timeline analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineAnalysis {
    /// Matching timeline entries
    pub entries: Vec<TimelineEntry>,
    /// Summary statistics for the query period
    pub summary: TimelineStatistics,
    /// Memory usage pattern during the period
    pub usage_pattern: UsagePattern,
    /// Detected anomalies
    pub anomalies: Vec<TimelineAnomaly>,
}

/// Memory usage pattern over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern type (e.g., "steady", "growing", "oscillating")
    pub pattern_type: String,
    /// Confidence in pattern detection (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern characteristics
    pub characteristics: Vec<String>,
    /// Recommended actions based on pattern
    pub recommendations: Vec<String>,
}

/// Detected anomaly in the timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineAnomaly {
    /// Type of anomaly
    pub anomaly_type: String,
    /// Severity level (0.0 to 1.0)
    pub severity: f64,
    /// Description of the anomaly
    pub description: String,
    /// Time when anomaly was detected
    pub timestamp: SystemTime,
    /// Related timeline entries
    pub related_entries: Vec<u64>, // Sequence numbers
}

impl AllocationTimeline {
    /// Creates a new allocation timeline
    ///
    /// # Arguments
    ///
    /// * `max_entries` - Maximum number of events to keep
    /// * `retention_period` - Maximum age of events to keep
    ///
    /// # Returns
    ///
    /// New allocation timeline instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::timeline::AllocationTimeline;
    /// use std::time::Duration;
    ///
    /// let timeline = AllocationTimeline::new(10000, Duration::from_secs(3600));
    /// ```
    pub fn new(max_entries: usize, retention_period: Duration) -> Self {
        #[cfg(feature = "tracing")]
        info!("Creating allocation timeline with max_entries={}, retention={:?}", 
              max_entries, retention_period);

        Self {
            max_entries,
            retention_period,
            events: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(TimelineStatistics::new())),
        }
    }

    /// Adds an event to the timeline
    ///
    /// # Arguments
    ///
    /// * `event` - Allocation event to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::timeline::{AllocationTimeline, AllocationEvent};
    /// use bitnet_core::memory::tracking::AllocationId;
    /// use std::time::{Duration, SystemTime};
    ///
    /// let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
    /// let event = AllocationEvent::Allocation {
    ///     id: AllocationId::new(1),
    ///     size: 1024,
    ///     device_type: "CPU".to_string(),
    ///     timestamp: SystemTime::now(),
    /// };
    /// timeline.add_event(event);
    /// ```
    pub fn add_event(&self, event: AllocationEvent) {
        let now = SystemTime::now();
        
        // Create timeline entry
        let (sequence, elapsed_time) = {
            let mut stats = self.stats.lock().unwrap();
            stats.total_events += 1;
            
            // Update event type counters
            match &event {
                AllocationEvent::Allocation { .. } => stats.allocation_events += 1,
                AllocationEvent::Deallocation { .. } => stats.deallocation_events += 1,
                AllocationEvent::PressureEvent { .. } => stats.pressure_events += 1,
                AllocationEvent::PoolExpansion { .. } => stats.pool_expansion_events += 1,
            }
            
            let elapsed = now.duration_since(stats.start_time).unwrap_or(Duration::ZERO);
            stats.duration = elapsed;
            
            // Update events per second
            if elapsed.as_secs() > 0 {
                stats.events_per_second = stats.total_events as f64 / elapsed.as_secs_f64();
            }
            
            (stats.total_events, elapsed)
        };

        let entry = TimelineEntry {
            event,
            sequence,
            elapsed_time,
        };

        // Add to timeline
        {
            let mut events = self.events.lock().unwrap();
            events.push_back(entry);

            // Enforce size limit
            while events.len() > self.max_entries {
                events.pop_front();
            }

            // Enforce retention period
            let cutoff_time = now - self.retention_period;
            while let Some(front) = events.front() {
                let event_time = match &front.event {
                    AllocationEvent::Allocation { timestamp, .. } => *timestamp,
                    AllocationEvent::Deallocation { timestamp, .. } => *timestamp,
                    AllocationEvent::PressureEvent { timestamp, .. } => *timestamp,
                    AllocationEvent::PoolExpansion { timestamp, .. } => *timestamp,
                };
                
                if event_time < cutoff_time {
                    events.pop_front();
                } else {
                    break;
                }
            }
        }

        #[cfg(feature = "tracing")]
        debug!("Added timeline event #{} at {:?}", sequence, elapsed_time);
    }

    /// Queries the timeline with specified parameters
    ///
    /// # Arguments
    ///
    /// * `query` - Query parameters
    ///
    /// # Returns
    ///
    /// Timeline analysis results
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::timeline::{AllocationTimeline, TimelineQuery};
    /// use std::time::Duration;
    ///
    /// let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
    /// let query = TimelineQuery {
    ///     start_time: None,
    ///     end_time: None,
    ///     event_type: Some("Allocation".to_string()),
    ///     device_type: Some("CPU".to_string()),
    ///     min_size: Some(1024),
    ///     max_size: None,
    ///     limit: Some(100),
    /// };
    /// let analysis = timeline.query(query);
    /// ```
    pub fn query(&self, query: TimelineQuery) -> TimelineAnalysis {
        let events = self.events.lock().unwrap();
        let _stats = self.stats.lock().unwrap();

        // Filter events based on query parameters
        let mut matching_entries = Vec::new();
        
        for entry in events.iter() {
            if self.matches_query(entry, &query) {
                matching_entries.push(entry.clone());
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            matching_entries.truncate(limit);
        }

        // Generate analysis
        let summary = self.calculate_summary_stats(&matching_entries);
        let usage_pattern = self.analyze_usage_pattern(&matching_entries);
        let anomalies = self.detect_anomalies(&matching_entries);

        TimelineAnalysis {
            entries: matching_entries,
            summary,
            usage_pattern,
            anomalies,
        }
    }

    /// Returns current timeline statistics
    ///
    /// # Returns
    ///
    /// Current timeline statistics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::timeline::AllocationTimeline;
    /// use std::time::Duration;
    ///
    /// let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
    /// let stats = timeline.get_statistics();
    /// println!("Total events: {}", stats.total_events);
    /// ```
    pub fn get_statistics(&self) -> TimelineStatistics {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Returns recent events from the timeline
    ///
    /// # Arguments
    ///
    /// * `count` - Number of recent events to return
    ///
    /// # Returns
    ///
    /// Vector of recent timeline entries
    pub fn get_recent_events(&self, count: usize) -> Vec<TimelineEntry> {
        let events = self.events.lock().unwrap();
        events.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Estimates memory usage of the timeline
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let events = self.events.lock().unwrap();
        events.len() * std::mem::size_of::<TimelineEntry>()
    }

    /// Clears all timeline data
    pub fn clear(&self) {
        {
            let mut events = self.events.lock().unwrap();
            events.clear();
        }
        
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = TimelineStatistics::new();
        }
    }

    // Private helper methods

    fn matches_query(&self, entry: &TimelineEntry, query: &TimelineQuery) -> bool {
        // Check time range
        let event_time = match &entry.event {
            AllocationEvent::Allocation { timestamp, .. } => *timestamp,
            AllocationEvent::Deallocation { timestamp, .. } => *timestamp,
            AllocationEvent::PressureEvent { timestamp, .. } => *timestamp,
            AllocationEvent::PoolExpansion { timestamp, .. } => *timestamp,
        };

        if let Some(start) = query.start_time {
            if event_time < start {
                return false;
            }
        }

        if let Some(end) = query.end_time {
            if event_time > end {
                return false;
            }
        }

        // Check event type
        if let Some(ref event_type) = query.event_type {
            let entry_type = match &entry.event {
                AllocationEvent::Allocation { .. } => "Allocation",
                AllocationEvent::Deallocation { .. } => "Deallocation",
                AllocationEvent::PressureEvent { .. } => "PressureEvent",
                AllocationEvent::PoolExpansion { .. } => "PoolExpansion",
            };
            if entry_type != event_type {
                return false;
            }
        }

        // Check device type
        if let Some(ref device_type) = query.device_type {
            let entry_device = match &entry.event {
                AllocationEvent::Allocation { device_type: dt, .. } => dt,
                AllocationEvent::Deallocation { device_type: dt, .. } => dt,
                AllocationEvent::PressureEvent { .. } => return true, // No device filter for pressure events
                AllocationEvent::PoolExpansion { device_type: dt, .. } => dt,
            };
            if entry_device != device_type {
                return false;
            }
        }

        // Check size range
        let entry_size = match &entry.event {
            AllocationEvent::Allocation { size, .. } => Some(*size),
            AllocationEvent::Deallocation { size, .. } => Some(*size),
            AllocationEvent::PressureEvent { .. } => None,
            AllocationEvent::PoolExpansion { new_size, .. } => Some(*new_size),
        };

        if let Some(size) = entry_size {
            if let Some(min_size) = query.min_size {
                if size < min_size {
                    return false;
                }
            }
            if let Some(max_size) = query.max_size {
                if size > max_size {
                    return false;
                }
            }
        }

        true
    }

    fn calculate_summary_stats(&self, entries: &[TimelineEntry]) -> TimelineStatistics {
        let mut stats = TimelineStatistics::new();
        
        if entries.is_empty() {
            return stats;
        }

        stats.total_events = entries.len() as u64;
        
        for entry in entries {
            match &entry.event {
                AllocationEvent::Allocation { .. } => stats.allocation_events += 1,
                AllocationEvent::Deallocation { .. } => stats.deallocation_events += 1,
                AllocationEvent::PressureEvent { .. } => stats.pressure_events += 1,
                AllocationEvent::PoolExpansion { .. } => stats.pool_expansion_events += 1,
            }
        }

        // Calculate duration and rate
        if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
            let first_time = match &first.event {
                AllocationEvent::Allocation { timestamp, .. } => *timestamp,
                AllocationEvent::Deallocation { timestamp, .. } => *timestamp,
                AllocationEvent::PressureEvent { timestamp, .. } => *timestamp,
                AllocationEvent::PoolExpansion { timestamp, .. } => *timestamp,
            };
            
            let last_time = match &last.event {
                AllocationEvent::Allocation { timestamp, .. } => *timestamp,
                AllocationEvent::Deallocation { timestamp, .. } => *timestamp,
                AllocationEvent::PressureEvent { timestamp, .. } => *timestamp,
                AllocationEvent::PoolExpansion { timestamp, .. } => *timestamp,
            };

            stats.start_time = first_time;
            stats.duration = last_time.duration_since(first_time).unwrap_or(Duration::ZERO);
            
            if stats.duration.as_secs() > 0 {
                stats.events_per_second = stats.total_events as f64 / stats.duration.as_secs_f64();
            }
        }

        stats
    }

    fn analyze_usage_pattern(&self, entries: &[TimelineEntry]) -> UsagePattern {
        if entries.is_empty() {
            return UsagePattern {
                pattern_type: "unknown".to_string(),
                confidence: 0.0,
                characteristics: vec!["No data available".to_string()],
                recommendations: vec!["Collect more data".to_string()],
            };
        }

        // Simple pattern analysis based on allocation/deallocation balance
        let allocations = entries.iter()
            .filter(|e| matches!(e.event, AllocationEvent::Allocation { .. }))
            .count();
        let deallocations = entries.iter()
            .filter(|e| matches!(e.event, AllocationEvent::Deallocation { .. }))
            .count();

        let balance_ratio = if deallocations > 0 {
            allocations as f64 / deallocations as f64
        } else if allocations > 0 {
            f64::INFINITY
        } else {
            1.0
        };

        let (pattern_type, confidence, characteristics, recommendations) = if balance_ratio > 1.5 {
            (
                "growing".to_string(),
                0.8,
                vec!["More allocations than deallocations".to_string()],
                vec!["Monitor for potential memory leaks".to_string()],
            )
        } else if balance_ratio < 0.7 {
            (
                "shrinking".to_string(),
                0.8,
                vec!["More deallocations than allocations".to_string()],
                vec!["Memory usage is decreasing".to_string()],
            )
        } else {
            (
                "steady".to_string(),
                0.9,
                vec!["Balanced allocation and deallocation".to_string()],
                vec!["Memory usage appears stable".to_string()],
            )
        };

        UsagePattern {
            pattern_type,
            confidence,
            characteristics,
            recommendations,
        }
    }

    fn detect_anomalies(&self, entries: &[TimelineEntry]) -> Vec<TimelineAnomaly> {
        let mut anomalies = Vec::new();

        // Detect allocation spikes
        let allocation_sizes: Vec<usize> = entries.iter()
            .filter_map(|e| match &e.event {
                AllocationEvent::Allocation { size, .. } => Some(*size),
                _ => None,
            })
            .collect();

        if !allocation_sizes.is_empty() {
            let avg_size = allocation_sizes.iter().sum::<usize>() as f64 / allocation_sizes.len() as f64;
            let threshold = avg_size * 10.0; // 10x average is considered anomalous

            for entry in entries {
                if let AllocationEvent::Allocation { size, timestamp, .. } = &entry.event {
                    if *size as f64 > threshold {
                        anomalies.push(TimelineAnomaly {
                            anomaly_type: "large_allocation".to_string(),
                            severity: (*size as f64 / threshold).min(1.0),
                            description: format!("Unusually large allocation: {} bytes ({}x average)", 
                                               size, *size as f64 / avg_size),
                            timestamp: *timestamp,
                            related_entries: vec![entry.sequence],
                        });
                    }
                }
            }
        }

        anomalies
    }
}

impl TimelineStatistics {
    fn new() -> Self {
        Self {
            total_events: 0,
            allocation_events: 0,
            deallocation_events: 0,
            pressure_events: 0,
            pool_expansion_events: 0,
            start_time: SystemTime::now(),
            duration: Duration::ZERO,
            events_per_second: 0.0,
            peak_events_per_window: 0,
            memory_trend: MemoryTrend {
                trend_direction: 0.0,
                change_rate_bytes_per_sec: 0.0,
                volatility: 0.0,
                peak_usage: 0,
                min_usage: 0,
                average_usage: 0.0,
            },
        }
    }
}

impl Default for TimelineQuery {
    fn default() -> Self {
        Self {
            start_time: None,
            end_time: None,
            event_type: None,
            device_type: None,
            min_size: None,
            max_size: None,
            limit: None,
        }
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for AllocationTimeline {}
unsafe impl Sync for AllocationTimeline {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_timeline_creation() {
        let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
        let stats = timeline.get_statistics();
        assert_eq!(stats.total_events, 0);
    }

    #[test]
    fn test_add_events() {
        let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
        
        let event = AllocationEvent::Allocation {
            id: AllocationId::new(1),
            size: 1024,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now(),
        };
        
        timeline.add_event(event);
        
        let stats = timeline.get_statistics();
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.allocation_events, 1);
    }

    #[test]
    fn test_timeline_query() {
        let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
        
        // Add some events
        for i in 0..5 {
            let event = AllocationEvent::Allocation {
                id: AllocationId::new(i),
                size: 1024 * (i as usize + 1),
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
            };
            timeline.add_event(event);
        }
        
        let query = TimelineQuery {
            min_size: Some(2048),
            ..Default::default()
        };
        
        let analysis = timeline.query(query);
        assert!(analysis.entries.len() < 5); // Should filter out smaller allocations
    }

    #[test]
    fn test_retention_period() {
        let timeline = AllocationTimeline::new(1000, Duration::from_millis(10));
        
        let event = AllocationEvent::Allocation {
            id: AllocationId::new(1),
            size: 1024,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now() - Duration::from_secs(1), // Old event
        };
        
        timeline.add_event(event);
        
        // Wait for retention period to pass
        std::thread::sleep(Duration::from_millis(20));
        
        // Add a new event to trigger cleanup
        let new_event = AllocationEvent::Allocation {
            id: AllocationId::new(2),
            size: 1024,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now(),
        };
        
        timeline.add_event(new_event);
        
        let recent_events = timeline.get_recent_events(10);
        assert_eq!(recent_events.len(), 1); // Old event should be cleaned up
    }

    #[test]
    fn test_usage_pattern_analysis() {
        let timeline = AllocationTimeline::new(1000, Duration::from_secs(3600));
        
        // Add more allocations than deallocations
        for i in 0..10 {
            let alloc_event = AllocationEvent::Allocation {
                id: AllocationId::new(i),
                size: 1024,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
            };
            timeline.add_event(alloc_event);
        }
        
        for i in 0..5 {
            let dealloc_event = AllocationEvent::Deallocation {
                id: AllocationId::new(i),
                size: 1024,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
            };
            timeline.add_event(dealloc_event);
        }
        
        let query = TimelineQuery::default();
        let analysis = timeline.query(query);
        
        assert_eq!(analysis.usage_pattern.pattern_type, "growing");
    }
}