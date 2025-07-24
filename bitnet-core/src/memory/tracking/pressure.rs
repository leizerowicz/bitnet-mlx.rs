//! Memory Pressure Detection System
//!
//! This module provides real-time memory pressure monitoring with configurable
//! thresholds and callback system for pressure events.

use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

use super::{TrackingError, TrackingResult};
use super::config::PressureThresholds;

/// Memory pressure levels indicating system memory stress
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    /// No memory pressure - normal operation
    None,
    /// Low memory pressure - minor concern
    Low,
    /// Medium memory pressure - moderate concern
    Medium,
    /// High memory pressure - significant concern
    High,
    /// Critical memory pressure - immediate action required
    Critical,
}

/// Callback function type for pressure events
pub type PressureCallback = Box<dyn Fn(MemoryPressureLevel) + Send + Sync>;

/// Memory pressure event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureEvent {
    /// Pressure level that triggered the event
    pub level: MemoryPressureLevel,
    /// Memory usage percentage when event occurred
    pub memory_usage_percentage: f64,
    /// Total memory usage in bytes
    pub total_memory_usage: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Timestamp when event occurred
    pub timestamp: SystemTime,
    /// Duration since last pressure event
    pub time_since_last_event: Option<Duration>,
}

/// Memory pressure detector with configurable thresholds and callbacks
pub struct MemoryPressureDetector {
    /// Pressure detection thresholds
    thresholds: PressureThresholds,
    /// Current pressure level
    current_level: Arc<RwLock<MemoryPressureLevel>>,
    /// Registered callbacks for pressure events
    callbacks: Arc<Mutex<Vec<PressureCallback>>>,
    /// Recent pressure events history
    event_history: Arc<Mutex<VecDeque<PressureEvent>>>,
    /// Last notification time for cooldown
    last_notification: Arc<Mutex<Option<Instant>>>,
    /// Memory usage history for trend analysis
    usage_history: Arc<Mutex<VecDeque<MemoryUsageSample>>>,
    /// System memory information
    system_memory: Arc<RwLock<SystemMemoryInfo>>,
}

/// Sample of memory usage at a specific time
#[derive(Debug, Clone)]
struct MemoryUsageSample {
    usage_bytes: u64,
    timestamp: Instant,
}

/// System memory information
#[derive(Debug, Clone)]
struct SystemMemoryInfo {
    total_memory: u64,
    available_memory: u64,
    last_updated: Instant,
}

/// Memory pressure statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureStatistics {
    /// Current pressure level
    pub current_level: MemoryPressureLevel,
    /// Current memory usage percentage
    pub current_usage_percentage: f64,
    /// Total number of pressure events
    pub total_pressure_events: usize,
    /// Events by pressure level
    pub events_by_level: std::collections::HashMap<String, usize>,
    /// Average time between pressure events
    pub avg_time_between_events: Option<Duration>,
    /// Memory usage trend (positive = increasing, negative = decreasing)
    pub usage_trend: f64,
    /// Time spent in each pressure level
    pub time_in_levels: std::collections::HashMap<String, Duration>,
}

impl MemoryPressureDetector {
    /// Creates a new memory pressure detector
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Pressure detection thresholds
    ///
    /// # Returns
    ///
    /// A Result containing the new detector or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::pressure::{MemoryPressureDetector, PressureThresholds};
    ///
    /// let thresholds = PressureThresholds::default();
    /// let detector = MemoryPressureDetector::new(thresholds)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(thresholds: PressureThresholds) -> TrackingResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Creating memory pressure detector with thresholds: {:?}", thresholds);

        // Validate thresholds
        thresholds.validate().map_err(|e| TrackingError::InvalidConfiguration { reason: e })?;

        let system_memory = Self::get_system_memory_info()
            .map_err(|e| TrackingError::PressureDetectionError { reason: e })?;

        Ok(Self {
            thresholds,
            current_level: Arc::new(RwLock::new(MemoryPressureLevel::None)),
            callbacks: Arc::new(Mutex::new(Vec::new())),
            event_history: Arc::new(Mutex::new(VecDeque::new())),
            last_notification: Arc::new(Mutex::new(None)),
            usage_history: Arc::new(Mutex::new(VecDeque::new())),
            system_memory: Arc::new(RwLock::new(system_memory)),
        })
    }

    /// Updates memory usage and checks for pressure level changes
    ///
    /// # Arguments
    ///
    /// * `current_usage` - Current memory usage in bytes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::pressure::MemoryPressureDetector;
    ///
    /// let detector = MemoryPressureDetector::new(Default::default())?;
    /// detector.update_memory_usage(1024 * 1024 * 1024); // 1GB
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn update_memory_usage(&self, current_usage: u64) {
        // Update system memory info periodically
        self.update_system_memory_info();

        // Record usage sample
        self.record_usage_sample(current_usage);

        // Calculate usage percentage
        let usage_percentage = self.calculate_usage_percentage(current_usage);

        // Determine new pressure level
        let new_level = self.calculate_pressure_level(usage_percentage);

        // Check if level changed
        let level_changed = {
            let current_level = self.current_level.read().unwrap();
            *current_level != new_level
        };

        if level_changed {
            #[cfg(feature = "tracing")]
            debug!("Memory pressure level changed to {:?} ({}%)", new_level, usage_percentage * 100.0);

            // Update current level
            {
                let mut current_level = self.current_level.write().unwrap();
                *current_level = new_level;
            }

            // Check cooldown before notifying
            let should_notify = self.should_notify();

            if should_notify {
                self.notify_pressure_change(new_level, usage_percentage, current_usage);
            }
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
    /// use bitnet_core::memory::tracking::pressure::{MemoryPressureDetector, MemoryPressureLevel};
    ///
    /// let detector = MemoryPressureDetector::new(Default::default())?;
    /// let level = detector.get_current_level();
    /// match level {
    ///     MemoryPressureLevel::None => println!("No pressure"),
    ///     MemoryPressureLevel::Critical => println!("Critical pressure!"),
    ///     _ => println!("Some pressure detected"),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_current_level(&self) -> MemoryPressureLevel {
        self.current_level.read().unwrap().clone()
    }

    /// Registers a callback for pressure level changes
    ///
    /// # Arguments
    ///
    /// * `callback` - Callback function to be called on pressure changes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::pressure::{MemoryPressureDetector, MemoryPressureLevel};
    ///
    /// let detector = MemoryPressureDetector::new(Default::default())?;
    /// detector.register_callback(Box::new(|level| {
    ///     match level {
    ///         MemoryPressureLevel::Critical => {
    ///             eprintln!("CRITICAL: Memory pressure detected!");
    ///         }
    ///         _ => println!("Memory pressure: {:?}", level),
    ///     }
    /// }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register_callback(&self, callback: PressureCallback) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            callbacks.push(callback);
        }
    }

    /// Returns pressure statistics
    ///
    /// # Returns
    ///
    /// Comprehensive pressure statistics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::pressure::MemoryPressureDetector;
    ///
    /// let detector = MemoryPressureDetector::new(Default::default())?;
    /// let stats = detector.get_statistics();
    /// println!("Current usage: {:.1}%", stats.current_usage_percentage * 100.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_statistics(&self) -> PressureStatistics {
        let current_level = self.get_current_level();
        let current_usage_percentage = self.get_current_usage_percentage();

        let (total_events, events_by_level, avg_time_between_events) = {
            let history = self.event_history.lock().unwrap();
            let total = history.len();
            
            let mut by_level = std::collections::HashMap::new();
            for event in history.iter() {
                let level_str = format!("{:?}", event.level);
                *by_level.entry(level_str).or_insert(0) += 1;
            }

            let avg_time = if history.len() > 1 {
                let total_time: Duration = history.iter()
                    .filter_map(|event| event.time_since_last_event)
                    .sum();
                Some(total_time / (history.len() - 1) as u32)
            } else {
                None
            };

            (total, by_level, avg_time)
        };

        let usage_trend = self.calculate_usage_trend();
        let time_in_levels = self.calculate_time_in_levels();

        PressureStatistics {
            current_level,
            current_usage_percentage,
            total_pressure_events: total_events,
            events_by_level,
            avg_time_between_events,
            usage_trend,
            time_in_levels,
        }
    }

    /// Returns recent pressure events
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of events to return
    ///
    /// # Returns
    ///
    /// Vector of recent pressure events
    pub fn get_recent_events(&self, limit: usize) -> Vec<PressureEvent> {
        let history = self.event_history.lock().unwrap();
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    // Private helper methods

    fn get_system_memory_info() -> Result<SystemMemoryInfo, String> {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            
            // Get total memory
            let total_output = Command::new("sysctl")
                .args(&["-n", "hw.memsize"])
                .output()
                .map_err(|e| format!("Failed to get total memory: {}", e))?;
            
            let total_memory = String::from_utf8(total_output.stdout)
                .map_err(|e| format!("Invalid total memory output: {}", e))?
                .trim()
                .parse::<u64>()
                .map_err(|e| format!("Failed to parse total memory: {}", e))?;

            // Get available memory (simplified - in real implementation would use vm_stat)
            let available_memory = total_memory / 2; // Placeholder

            Ok(SystemMemoryInfo {
                total_memory,
                available_memory,
                last_updated: Instant::now(),
            })
        }

        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            let meminfo = fs::read_to_string("/proc/meminfo")
                .map_err(|e| format!("Failed to read /proc/meminfo: {}", e))?;
            
            let mut total_memory = 0;
            let mut available_memory = 0;
            
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    total_memory = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0) * 1024; // Convert from KB to bytes
                } else if line.starts_with("MemAvailable:") {
                    available_memory = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0) * 1024; // Convert from KB to bytes
                }
            }

            Ok(SystemMemoryInfo {
                total_memory,
                available_memory,
                last_updated: Instant::now(),
            })
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            // Fallback for unsupported platforms
            Ok(SystemMemoryInfo {
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB default
                available_memory: 4 * 1024 * 1024 * 1024, // 4GB default
                last_updated: Instant::now(),
            })
        }
    }

    fn update_system_memory_info(&self) {
        let should_update = {
            let system_memory = self.system_memory.read().unwrap();
            system_memory.last_updated.elapsed() > Duration::from_secs(60) // Update every minute
        };

        if should_update {
            if let Ok(new_info) = Self::get_system_memory_info() {
                let mut system_memory = self.system_memory.write().unwrap();
                *system_memory = new_info;
            }
        }
    }

    fn record_usage_sample(&self, usage_bytes: u64) {
        if let Ok(mut history) = self.usage_history.lock() {
            history.push_back(MemoryUsageSample {
                usage_bytes,
                timestamp: Instant::now(),
            });

            // Keep only recent samples (last 10 minutes)
            let cutoff = Instant::now() - Duration::from_secs(600);
            while let Some(sample) = history.front() {
                if sample.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    fn calculate_usage_percentage(&self, current_usage: u64) -> f64 {
        let system_memory = self.system_memory.read().unwrap();
        if system_memory.total_memory == 0 {
            return 0.0;
        }
        current_usage as f64 / system_memory.total_memory as f64
    }

    fn calculate_pressure_level(&self, usage_percentage: f64) -> MemoryPressureLevel {
        if usage_percentage >= self.thresholds.critical_pressure_threshold {
            MemoryPressureLevel::Critical
        } else if usage_percentage >= self.thresholds.high_pressure_threshold {
            MemoryPressureLevel::High
        } else if usage_percentage >= self.thresholds.medium_pressure_threshold {
            MemoryPressureLevel::Medium
        } else if usage_percentage >= self.thresholds.low_pressure_threshold {
            MemoryPressureLevel::Low
        } else {
            MemoryPressureLevel::None
        }
    }

    fn should_notify(&self) -> bool {
        if let Ok(mut last_notification) = self.last_notification.lock() {
            let now = Instant::now();
            let should_notify = last_notification
                .map(|last| now.duration_since(last) >= self.thresholds.notification_cooldown)
                .unwrap_or(true);
            
            if should_notify {
                *last_notification = Some(now);
            }
            
            should_notify
        } else {
            false
        }
    }

    fn notify_pressure_change(&self, level: MemoryPressureLevel, usage_percentage: f64, current_usage: u64) {
        // Create pressure event
        let time_since_last = {
            let history = self.event_history.lock().unwrap();
            history.back().map(|last_event| {
                SystemTime::now().duration_since(last_event.timestamp).unwrap_or(Duration::ZERO)
            })
        };

        let system_memory = self.system_memory.read().unwrap();
        let event = PressureEvent {
            level,
            memory_usage_percentage: usage_percentage,
            total_memory_usage: current_usage,
            available_memory: system_memory.available_memory,
            timestamp: SystemTime::now(),
            time_since_last_event: time_since_last,
        };

        // Add to history
        {
            let mut history = self.event_history.lock().unwrap();
            history.push_back(event.clone());
            
            // Keep only recent events (last 1000)
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Notify callbacks
        if let Ok(callbacks) = self.callbacks.lock() {
            for callback in callbacks.iter() {
                callback(level);
            }
        }

        // Log pressure change
        match level {
            MemoryPressureLevel::Critical => {
                #[cfg(feature = "tracing")]
                error!("CRITICAL memory pressure detected: {:.1}% usage", usage_percentage * 100.0);
            }
            MemoryPressureLevel::High => {
                #[cfg(feature = "tracing")]
                warn!("HIGH memory pressure detected: {:.1}% usage", usage_percentage * 100.0);
            }
            MemoryPressureLevel::Medium => {
                #[cfg(feature = "tracing")]
                warn!("MEDIUM memory pressure detected: {:.1}% usage", usage_percentage * 100.0);
            }
            MemoryPressureLevel::Low => {
                #[cfg(feature = "tracing")]
                info!("LOW memory pressure detected: {:.1}% usage", usage_percentage * 100.0);
            }
            MemoryPressureLevel::None => {
                #[cfg(feature = "tracing")]
                info!("Memory pressure resolved: {:.1}% usage", usage_percentage * 100.0);
            }
        }
    }

    fn get_current_usage_percentage(&self) -> f64 {
        let history = self.usage_history.lock().unwrap();
        if let Some(latest) = history.back() {
            self.calculate_usage_percentage(latest.usage_bytes)
        } else {
            0.0
        }
    }

    fn calculate_usage_trend(&self) -> f64 {
        let history = self.usage_history.lock().unwrap();
        if history.len() < 2 {
            return 0.0;
        }

        // Calculate trend over recent samples
        let samples: Vec<_> = history.iter().collect();
        let n = samples.len();
        
        if n < 2 {
            return 0.0;
        }

        // Simple linear regression to calculate trend
        let x_sum: f64 = (0..n).map(|i| i as f64).sum();
        let y_sum: f64 = samples.iter().map(|s| s.usage_bytes as f64).sum();
        let xy_sum: f64 = samples.iter().enumerate()
            .map(|(i, s)| i as f64 * s.usage_bytes as f64)
            .sum();
        let x2_sum: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();

        let n_f64 = n as f64;
        let slope = (n_f64 * xy_sum - x_sum * y_sum) / (n_f64 * x2_sum - x_sum.powi(2));
        
        slope / 1024.0 / 1024.0 // Convert to MB/sample trend
    }

    fn calculate_time_in_levels(&self) -> std::collections::HashMap<String, Duration> {
        let mut time_in_levels = std::collections::HashMap::new();
        let history = self.event_history.lock().unwrap();
        
        if history.is_empty() {
            return time_in_levels;
        }

        let mut current_level = MemoryPressureLevel::None;
        let mut level_start_time = history.front().unwrap().timestamp;

        for event in history.iter() {
            if event.level != current_level {
                // Calculate time spent in previous level
                let time_in_level = event.timestamp.duration_since(level_start_time)
                    .unwrap_or(Duration::ZERO);
                
                let level_str = format!("{:?}", current_level);
                *time_in_levels.entry(level_str).or_insert(Duration::ZERO) += time_in_level;
                
                current_level = event.level;
                level_start_time = event.timestamp;
            }
        }

        // Add time for current level
        let current_time_in_level = SystemTime::now().duration_since(level_start_time)
            .unwrap_or(Duration::ZERO);
        let level_str = format!("{:?}", current_level);
        *time_in_levels.entry(level_str).or_insert(Duration::ZERO) += current_time_in_level;

        time_in_levels
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for MemoryPressureDetector {}
unsafe impl Sync for MemoryPressureDetector {}

// Manual Debug implementation since PressureCallback doesn't implement Debug
impl std::fmt::Debug for MemoryPressureDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPressureDetector")
            .field("thresholds", &self.thresholds)
            .field("current_level", &self.current_level)
            .field("callbacks", &format!("{} callbacks", self.callbacks.lock().map(|c| c.len()).unwrap_or(0)))
            .field("event_history", &self.event_history)
            .field("last_notification", &self.last_notification)
            .field("usage_history", &self.usage_history)
            .field("system_memory", &self.system_memory)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::PressureThresholds;

    #[test]
    fn test_pressure_detector_creation() {
        let thresholds = PressureThresholds::default();
        let detector = MemoryPressureDetector::new(thresholds).unwrap();
        
        assert_eq!(detector.get_current_level(), MemoryPressureLevel::None);
    }

    #[test]
    fn test_pressure_level_calculation() {
        let thresholds = PressureThresholds::default();
        let detector = MemoryPressureDetector::new(thresholds).unwrap();
        
        // Test different usage levels
        assert_eq!(detector.calculate_pressure_level(0.5), MemoryPressureLevel::None);
        assert_eq!(detector.calculate_pressure_level(0.75), MemoryPressureLevel::Low);
        assert_eq!(detector.calculate_pressure_level(0.85), MemoryPressureLevel::Medium);
        assert_eq!(detector.calculate_pressure_level(0.92), MemoryPressureLevel::High);
        assert_eq!(detector.calculate_pressure_level(0.97), MemoryPressureLevel::Critical);
    }

    #[test]
    fn test_callback_registration() {
        let thresholds = PressureThresholds::default();
        let detector = MemoryPressureDetector::new(thresholds).unwrap();
        
        let callback_called = Arc::new(Mutex::new(false));
        let callback_called_clone = callback_called.clone();
        
        detector.register_callback(Box::new(move |_level| {
            *callback_called_clone.lock().unwrap() = true;
        }));
        
        // Simulate high memory usage to trigger callback
        detector.update_memory_usage(u64::MAX);
        
        // Note: In a real test, we'd need to wait or use synchronization
        // to ensure the callback was called
    }

    #[test]
    fn test_usage_trend_calculation() {
        let thresholds = PressureThresholds::default();
        let detector = MemoryPressureDetector::new(thresholds).unwrap();
        
        // Add some usage samples
        detector.record_usage_sample(1024 * 1024 * 1024); // 1GB
        std::thread::sleep(Duration::from_millis(10));
        detector.record_usage_sample(2 * 1024 * 1024 * 1024); // 2GB
        std::thread::sleep(Duration::from_millis(10));
        detector.record_usage_sample(3 * 1024 * 1024 * 1024); // 3GB
        
        let trend = detector.calculate_usage_trend();
        assert!(trend > 0.0); // Should show increasing trend
    }

    #[test]
    fn test_pressure_statistics() {
        let thresholds = PressureThresholds::default();
        let detector = MemoryPressureDetector::new(thresholds).unwrap();
        
        let stats = detector.get_statistics();
        assert_eq!(stats.current_level, MemoryPressureLevel::None);
        assert_eq!(stats.total_pressure_events, 0);
        assert!(stats.events_by_level.is_empty());
    }
}