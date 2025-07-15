//! Memory Tracking Configuration
//!
//! This module provides configuration options for the memory tracking system,
//! allowing fine-tuned control over tracking behavior, performance overhead,
//! and feature enablement.

use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Configuration for the memory tracking system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Tracking level determines the amount of detail collected
    pub level: TrackingLevel,
    /// Whether to enable real-time pressure monitoring
    pub enable_pressure_monitoring: bool,
    /// Whether to enable allocation timeline tracking
    pub enable_timeline_tracking: bool,
    /// Whether to enable pattern analysis
    pub enable_pattern_analysis: bool,
    /// Whether to enable memory leak detection
    pub enable_leak_detection: bool,
    /// Whether to capture stack traces for allocations (expensive)
    pub enable_stack_traces: bool,
    /// Maximum number of allocations to track in timeline
    pub max_timeline_entries: usize,
    /// Maximum age for timeline entries before cleanup
    pub timeline_retention: Duration,
    /// Pressure monitoring thresholds
    pub pressure_thresholds: PressureThresholds,
    /// Sampling rate for detailed tracking (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Whether to enable performance metrics collection
    pub enable_performance_metrics: bool,
    /// Interval for updating tracking statistics
    pub stats_update_interval: Duration,
    /// Maximum memory overhead allowed for tracking (as percentage)
    pub max_overhead_percentage: f64,
}

/// Tracking level determines the detail and performance impact of tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingLevel {
    /// Minimal tracking - only basic counters, lowest overhead
    Minimal,
    /// Standard tracking - allocation/deallocation tracking with basic metrics
    Standard,
    /// Detailed tracking - comprehensive metrics and analysis
    Detailed,
    /// Debug tracking - maximum detail including stack traces and profiling
    Debug,
}

/// Memory pressure detection thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureThresholds {
    /// Low pressure threshold (percentage of available memory)
    pub low_pressure_threshold: f64,
    /// Medium pressure threshold (percentage of available memory)
    pub medium_pressure_threshold: f64,
    /// High pressure threshold (percentage of available memory)
    pub high_pressure_threshold: f64,
    /// Critical pressure threshold (percentage of available memory)
    pub critical_pressure_threshold: f64,
    /// Minimum time between pressure notifications
    pub notification_cooldown: Duration,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self::standard()
    }
}

impl TrackingConfig {
    /// Creates a minimal tracking configuration with lowest overhead
    pub fn minimal() -> Self {
        Self {
            level: TrackingLevel::Minimal,
            enable_pressure_monitoring: false,
            enable_timeline_tracking: false,
            enable_pattern_analysis: false,
            enable_leak_detection: false,
            enable_stack_traces: false,
            max_timeline_entries: 0,
            timeline_retention: Duration::from_secs(0),
            pressure_thresholds: PressureThresholds::default(),
            sampling_rate: 0.1, // Sample 10% of allocations
            enable_performance_metrics: false,
            stats_update_interval: Duration::from_secs(60),
            max_overhead_percentage: 1.0, // 1% max overhead
        }
    }

    /// Creates a standard tracking configuration with balanced features and performance
    pub fn standard() -> Self {
        Self {
            level: TrackingLevel::Standard,
            enable_pressure_monitoring: true,
            enable_timeline_tracking: true,
            enable_pattern_analysis: false,
            enable_leak_detection: true,
            enable_stack_traces: false,
            max_timeline_entries: 10000,
            timeline_retention: Duration::from_secs(3600), // 1 hour
            pressure_thresholds: PressureThresholds::default(),
            sampling_rate: 1.0, // Track all allocations
            enable_performance_metrics: true,
            stats_update_interval: Duration::from_secs(30),
            max_overhead_percentage: 3.0, // 3% max overhead
        }
    }

    /// Creates a detailed tracking configuration with comprehensive monitoring
    pub fn detailed() -> Self {
        Self {
            level: TrackingLevel::Detailed,
            enable_pressure_monitoring: true,
            enable_timeline_tracking: true,
            enable_pattern_analysis: true,
            enable_leak_detection: true,
            enable_stack_traces: false,
            max_timeline_entries: 50000,
            timeline_retention: Duration::from_secs(7200), // 2 hours
            pressure_thresholds: PressureThresholds::default(),
            sampling_rate: 1.0, // Track all allocations
            enable_performance_metrics: true,
            stats_update_interval: Duration::from_secs(10),
            max_overhead_percentage: 5.0, // 5% max overhead
        }
    }

    /// Creates a debug tracking configuration with maximum detail and profiling
    pub fn debug() -> Self {
        Self {
            level: TrackingLevel::Debug,
            enable_pressure_monitoring: true,
            enable_timeline_tracking: true,
            enable_pattern_analysis: true,
            enable_leak_detection: true,
            enable_stack_traces: true,
            max_timeline_entries: 100000,
            timeline_retention: Duration::from_secs(14400), // 4 hours
            pressure_thresholds: PressureThresholds::default(),
            sampling_rate: 1.0, // Track all allocations
            enable_performance_metrics: true,
            stats_update_interval: Duration::from_secs(5),
            max_overhead_percentage: 10.0, // 10% max overhead for debug
        }
    }

    /// Creates a custom configuration for production environments
    pub fn production() -> Self {
        Self {
            level: TrackingLevel::Standard,
            enable_pressure_monitoring: true,
            enable_timeline_tracking: false, // Disabled for production
            enable_pattern_analysis: false,  // Disabled for production
            enable_leak_detection: true,
            enable_stack_traces: false,
            max_timeline_entries: 1000,
            timeline_retention: Duration::from_secs(300), // 5 minutes
            pressure_thresholds: PressureThresholds::production(),
            sampling_rate: 0.01, // Sample 1% of allocations
            enable_performance_metrics: true,
            stats_update_interval: Duration::from_secs(120), // 2 minutes
            max_overhead_percentage: 1.0, // 1% max overhead
        }
    }

    /// Validates the configuration and returns any issues
    pub fn validate(&self) -> Result<(), String> {
        if self.sampling_rate < 0.0 || self.sampling_rate > 1.0 {
            return Err("Sampling rate must be between 0.0 and 1.0".to_string());
        }

        if self.max_overhead_percentage < 0.0 || self.max_overhead_percentage > 50.0 {
            return Err("Max overhead percentage must be between 0.0 and 50.0".to_string());
        }

        if self.max_timeline_entries > 1_000_000 {
            return Err("Max timeline entries cannot exceed 1,000,000".to_string());
        }

        if self.timeline_retention > Duration::from_secs(86400) {
            return Err("Timeline retention cannot exceed 24 hours".to_string());
        }

        self.pressure_thresholds.validate()?;

        Ok(())
    }

    /// Returns whether stack trace capture is enabled and should be used
    pub fn should_capture_stack_traces(&self) -> bool {
        self.enable_stack_traces && matches!(self.level, TrackingLevel::Debug)
    }

    /// Returns whether an allocation should be sampled based on the sampling rate
    pub fn should_sample_allocation(&self, allocation_id: u64) -> bool {
        if self.sampling_rate >= 1.0 {
            return true;
        }
        if self.sampling_rate <= 0.0 {
            return false;
        }

        // Use allocation ID for deterministic sampling
        let hash = allocation_id.wrapping_mul(2654435761) % 1000;
        (hash as f64 / 1000.0) < self.sampling_rate
    }

    /// Returns the effective timeline capacity based on configuration
    pub fn effective_timeline_capacity(&self) -> usize {
        if !self.enable_timeline_tracking {
            return 0;
        }
        self.max_timeline_entries
    }

    /// Returns whether detailed metrics should be collected
    pub fn should_collect_detailed_metrics(&self) -> bool {
        matches!(self.level, TrackingLevel::Detailed | TrackingLevel::Debug)
    }

    /// Returns whether pattern analysis should be performed
    pub fn should_analyze_patterns(&self) -> bool {
        self.enable_pattern_analysis && 
        matches!(self.level, TrackingLevel::Detailed | TrackingLevel::Debug)
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            low_pressure_threshold: 0.7,    // 70% memory usage
            medium_pressure_threshold: 0.8, // 80% memory usage
            high_pressure_threshold: 0.9,   // 90% memory usage
            critical_pressure_threshold: 0.95, // 95% memory usage
            notification_cooldown: Duration::from_secs(30),
        }
    }
}

impl PressureThresholds {
    /// Creates production-optimized pressure thresholds
    pub fn production() -> Self {
        Self {
            low_pressure_threshold: 0.6,    // 60% memory usage
            medium_pressure_threshold: 0.75, // 75% memory usage
            high_pressure_threshold: 0.85,   // 85% memory usage
            critical_pressure_threshold: 0.92, // 92% memory usage
            notification_cooldown: Duration::from_secs(60), // Longer cooldown
        }
    }

    /// Creates conservative pressure thresholds for memory-constrained environments
    pub fn conservative() -> Self {
        Self {
            low_pressure_threshold: 0.5,    // 50% memory usage
            medium_pressure_threshold: 0.65, // 65% memory usage
            high_pressure_threshold: 0.8,   // 80% memory usage
            critical_pressure_threshold: 0.9, // 90% memory usage
            notification_cooldown: Duration::from_secs(15), // Shorter cooldown
        }
    }

    /// Creates aggressive pressure thresholds that allow higher memory usage
    pub fn aggressive() -> Self {
        Self {
            low_pressure_threshold: 0.8,    // 80% memory usage
            medium_pressure_threshold: 0.9, // 90% memory usage
            high_pressure_threshold: 0.95,  // 95% memory usage
            critical_pressure_threshold: 0.98, // 98% memory usage
            notification_cooldown: Duration::from_secs(45),
        }
    }

    /// Validates the pressure thresholds
    pub fn validate(&self) -> Result<(), String> {
        let thresholds = [
            self.low_pressure_threshold,
            self.medium_pressure_threshold,
            self.high_pressure_threshold,
            self.critical_pressure_threshold,
        ];

        // Check that all thresholds are between 0.0 and 1.0
        for &threshold in &thresholds {
            if threshold < 0.0 || threshold > 1.0 {
                return Err("All pressure thresholds must be between 0.0 and 1.0".to_string());
            }
        }

        // Check that thresholds are in ascending order
        for i in 1..thresholds.len() {
            if thresholds[i] <= thresholds[i - 1] {
                return Err("Pressure thresholds must be in ascending order".to_string());
            }
        }

        if self.notification_cooldown > Duration::from_secs(300) {
            return Err("Notification cooldown cannot exceed 5 minutes".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_config_presets() {
        let minimal = TrackingConfig::minimal();
        assert_eq!(minimal.level, TrackingLevel::Minimal);
        assert!(!minimal.enable_pressure_monitoring);
        assert!(!minimal.enable_timeline_tracking);
        assert_eq!(minimal.sampling_rate, 0.1);

        let standard = TrackingConfig::standard();
        assert_eq!(standard.level, TrackingLevel::Standard);
        assert!(standard.enable_pressure_monitoring);
        assert!(standard.enable_timeline_tracking);
        assert_eq!(standard.sampling_rate, 1.0);

        let detailed = TrackingConfig::detailed();
        assert_eq!(detailed.level, TrackingLevel::Detailed);
        assert!(detailed.enable_pattern_analysis);
        assert_eq!(detailed.max_timeline_entries, 50000);

        let debug = TrackingConfig::debug();
        assert_eq!(debug.level, TrackingLevel::Debug);
        assert!(debug.enable_stack_traces);
        assert_eq!(debug.max_timeline_entries, 100000);

        let production = TrackingConfig::production();
        assert_eq!(production.level, TrackingLevel::Standard);
        assert!(!production.enable_timeline_tracking);
        assert_eq!(production.sampling_rate, 0.01);
    }

    #[test]
    fn test_config_validation() {
        let mut config = TrackingConfig::standard();
        assert!(config.validate().is_ok());

        // Test invalid sampling rate
        config.sampling_rate = 1.5;
        assert!(config.validate().is_err());

        config.sampling_rate = -0.1;
        assert!(config.validate().is_err());

        // Reset and test invalid overhead
        config = TrackingConfig::standard();
        config.max_overhead_percentage = 60.0;
        assert!(config.validate().is_err());

        // Reset and test invalid timeline entries
        config = TrackingConfig::standard();
        config.max_timeline_entries = 2_000_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pressure_thresholds() {
        let default_thresholds = PressureThresholds::default();
        assert!(default_thresholds.validate().is_ok());

        let production_thresholds = PressureThresholds::production();
        assert!(production_thresholds.validate().is_ok());
        assert!(production_thresholds.low_pressure_threshold < default_thresholds.low_pressure_threshold);

        let conservative_thresholds = PressureThresholds::conservative();
        assert!(conservative_thresholds.validate().is_ok());
        assert!(conservative_thresholds.critical_pressure_threshold < default_thresholds.critical_pressure_threshold);

        let aggressive_thresholds = PressureThresholds::aggressive();
        assert!(aggressive_thresholds.validate().is_ok());
        assert!(aggressive_thresholds.low_pressure_threshold > default_thresholds.low_pressure_threshold);
    }

    #[test]
    fn test_pressure_thresholds_validation() {
        let mut thresholds = PressureThresholds::default();
        assert!(thresholds.validate().is_ok());

        // Test invalid threshold values
        thresholds.low_pressure_threshold = 1.5;
        assert!(thresholds.validate().is_err());

        thresholds.low_pressure_threshold = -0.1;
        assert!(thresholds.validate().is_err());

        // Test invalid ordering
        thresholds = PressureThresholds::default();
        thresholds.medium_pressure_threshold = 0.6; // Less than low threshold
        assert!(thresholds.validate().is_err());

        // Test invalid cooldown
        thresholds = PressureThresholds::default();
        thresholds.notification_cooldown = Duration::from_secs(400);
        assert!(thresholds.validate().is_err());
    }

    #[test]
    fn test_sampling_logic() {
        let config = TrackingConfig::standard();
        
        // With sampling rate 1.0, all allocations should be sampled
        assert!(config.should_sample_allocation(1));
        assert!(config.should_sample_allocation(1000));
        assert!(config.should_sample_allocation(999999));

        let mut config = TrackingConfig::minimal();
        config.sampling_rate = 0.0;
        
        // With sampling rate 0.0, no allocations should be sampled
        assert!(!config.should_sample_allocation(1));
        assert!(!config.should_sample_allocation(1000));
        assert!(!config.should_sample_allocation(999999));
    }

    #[test]
    fn test_config_helpers() {
        let minimal = TrackingConfig::minimal();
        assert!(!minimal.should_capture_stack_traces());
        assert_eq!(minimal.effective_timeline_capacity(), 0);
        assert!(!minimal.should_collect_detailed_metrics());
        assert!(!minimal.should_analyze_patterns());

        let debug = TrackingConfig::debug();
        assert!(debug.should_capture_stack_traces());
        assert_eq!(debug.effective_timeline_capacity(), 100000);
        assert!(debug.should_collect_detailed_metrics());
        assert!(debug.should_analyze_patterns());
    }
}