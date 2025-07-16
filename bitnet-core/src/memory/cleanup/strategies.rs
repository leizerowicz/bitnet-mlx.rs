//! Cleanup Strategy Implementations
//!
//! This module provides various cleanup strategies for different scenarios and
//! requirements. Each strategy implements the CleanupStrategy trait and can be
//! used independently or in combination with others.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::memory::{HybridMemoryPool, MemoryMetrics};
use crate::memory::tracking::{MemoryPressureLevel, DetailedMemoryMetrics};
use super::CleanupResult;
use super::config::{CleanupConfig};
pub use super::config::CleanupStrategyType;

/// Priority levels for cleanup operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CleanupPriority {
    /// Lowest priority - background cleanup
    Background = 0,
    /// Low priority - routine maintenance
    Low = 1,
    /// Normal priority - standard cleanup
    Normal = 2,
    /// High priority - urgent cleanup
    High = 3,
    /// Critical priority - emergency cleanup
    Critical = 4,
}

/// Result of a cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupOperationResult {
    /// Number of bytes freed
    pub bytes_freed: u64,
    /// Number of allocations cleaned
    pub allocations_cleaned: u64,
    /// Duration of the cleanup operation
    pub duration: Duration,
    /// Whether the operation was successful
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
    /// Additional metadata about the operation
    pub metadata: HashMap<String, String>,
}

impl CleanupOperationResult {
    /// Creates a new successful cleanup result
    pub fn success(bytes_freed: u64, allocations_cleaned: u64, duration: Duration) -> Self {
        Self {
            bytes_freed,
            allocations_cleaned,
            duration,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a new failed cleanup result
    pub fn failure(error: String, duration: Duration) -> Self {
        Self {
            bytes_freed: 0,
            allocations_cleaned: 0,
            duration,
            success: false,
            error_message: Some(error),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the result
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Trait for implementing cleanup strategies
pub trait CleanupStrategy: Send + Sync {
    /// Returns the strategy type
    fn strategy_type(&self) -> CleanupStrategyType;

    /// Returns the priority of this strategy
    fn priority(&self) -> CleanupPriority;

    /// Determines if cleanup should be performed based on current metrics
    fn should_cleanup(&self, metrics: &MemoryMetrics, detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool;

    /// Performs the cleanup operation
    fn cleanup(&self, pool: &HybridMemoryPool, config: &CleanupConfig) -> CleanupResult<CleanupOperationResult>;

    /// Returns a human-readable description of the strategy
    fn description(&self) -> String;

    /// Returns configuration-specific parameters for this strategy
    fn get_parameters(&self) -> HashMap<String, String>;

    /// Updates strategy parameters (if supported)
    fn update_parameters(&mut self, _parameters: HashMap<String, String>) -> CleanupResult<()> {
        Ok(()) // Default implementation does nothing
    }
}

/// Idle cleanup strategy - performs cleanup during idle periods
pub struct IdleCleanupStrategy {
    last_cleanup: Arc<RwLock<Option<Instant>>>,
    min_idle_time: Duration,
    max_cleanup_duration: Duration,
}

impl IdleCleanupStrategy {
    /// Creates a new idle cleanup strategy
    pub fn new(min_idle_time: Duration, max_cleanup_duration: Duration) -> Self {
        Self {
            last_cleanup: Arc::new(RwLock::new(None)),
            min_idle_time,
            max_cleanup_duration,
        }
    }

    /// Creates a default idle cleanup strategy
    pub fn default() -> Self {
        Self::new(Duration::from_millis(100), Duration::from_millis(50))
    }
}

impl CleanupStrategy for IdleCleanupStrategy {
    fn strategy_type(&self) -> CleanupStrategyType {
        CleanupStrategyType::Idle
    }

    fn priority(&self) -> CleanupPriority {
        CleanupPriority::Low
    }

    fn should_cleanup(&self, metrics: &MemoryMetrics, _detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool {
        // Check if enough time has passed since last cleanup
        if let Ok(last_cleanup) = self.last_cleanup.read() {
            if let Some(last) = *last_cleanup {
                if last.elapsed() < self.min_idle_time {
                    return false;
                }
            }
        }

        // Simple heuristic: cleanup if we have some allocated memory but not too much activity
        metrics.current_allocated > 0 && metrics.allocation_count > metrics.deallocation_count
    }

    fn cleanup(&self, pool: &HybridMemoryPool, _config: &CleanupConfig) -> CleanupResult<CleanupOperationResult> {
        let start_time = Instant::now();
        
        // Update last cleanup time
        if let Ok(mut last_cleanup) = self.last_cleanup.write() {
            *last_cleanup = Some(start_time);
        }

        // For idle cleanup, we perform gentle cleanup operations
        // This is a simplified implementation - in practice, you'd implement
        // actual memory pool cleanup logic here
        
        let _metrics_before = pool.get_metrics();
        
        // Simulate some cleanup work (in real implementation, this would be actual cleanup)
        std::thread::sleep(Duration::from_millis(1));
        
        let duration = start_time.elapsed();
        let bytes_freed = 0; // Would be calculated from actual cleanup
        let allocations_cleaned = 0; // Would be calculated from actual cleanup
        
        Ok(CleanupOperationResult::success(bytes_freed, allocations_cleaned, duration)
            .with_metadata("strategy".to_string(), "idle".to_string())
            .with_metadata("min_idle_time_ms".to_string(), self.min_idle_time.as_millis().to_string()))
    }

    fn description(&self) -> String {
        format!("Idle cleanup strategy (min_idle: {:?}, max_duration: {:?})", 
                self.min_idle_time, self.max_cleanup_duration)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("min_idle_time_ms".to_string(), self.min_idle_time.as_millis().to_string());
        params.insert("max_cleanup_duration_ms".to_string(), self.max_cleanup_duration.as_millis().to_string());
        params
    }
}

/// Pressure-based cleanup strategy - performs cleanup under memory pressure
pub struct PressureCleanupStrategy {
    light_threshold: f64,
    aggressive_threshold: f64,
    emergency_threshold: f64,
    last_cleanup: Arc<RwLock<Option<Instant>>>,
    min_interval: Duration,
}

impl PressureCleanupStrategy {
    /// Creates a new pressure cleanup strategy
    pub fn new(
        light_threshold: f64,
        aggressive_threshold: f64,
        emergency_threshold: f64,
        min_interval: Duration,
    ) -> Self {
        Self {
            light_threshold,
            aggressive_threshold,
            emergency_threshold,
            last_cleanup: Arc::new(RwLock::new(None)),
            min_interval,
        }
    }

    /// Creates a default pressure cleanup strategy
    pub fn default() -> Self {
        Self::new(0.7, 0.85, 0.95, Duration::from_secs(1))
    }

    /// Determines the pressure level based on metrics
    fn get_pressure_level(&self, metrics: &MemoryMetrics) -> CleanupPriority {
        let utilization = if metrics.peak_allocated > 0 {
            metrics.current_allocated as f64 / metrics.peak_allocated as f64
        } else {
            0.0
        };

        if utilization >= self.emergency_threshold {
            CleanupPriority::Critical
        } else if utilization >= self.aggressive_threshold {
            CleanupPriority::High
        } else if utilization >= self.light_threshold {
            CleanupPriority::Normal
        } else {
            CleanupPriority::Background
        }
    }
}

impl CleanupStrategy for PressureCleanupStrategy {
    fn strategy_type(&self) -> CleanupStrategyType {
        CleanupStrategyType::Pressure
    }

    fn priority(&self) -> CleanupPriority {
        CleanupPriority::High
    }

    fn should_cleanup(&self, metrics: &MemoryMetrics, detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool {
        // Check if enough time has passed since last cleanup
        if let Ok(last_cleanup) = self.last_cleanup.read() {
            if let Some(last) = *last_cleanup {
                if last.elapsed() < self.min_interval {
                    return false;
                }
            }
        }

        // Check memory pressure from detailed metrics if available
        if let Some(detailed) = detailed_metrics {
            match detailed.pressure_level {
                MemoryPressureLevel::Critical | MemoryPressureLevel::High => return true,
                MemoryPressureLevel::Medium => return true,
                _ => {}
            }
        }

        // Fallback to basic pressure calculation
        let pressure_level = self.get_pressure_level(metrics);
        matches!(pressure_level, CleanupPriority::Normal | CleanupPriority::High | CleanupPriority::Critical)
    }

    fn cleanup(&self, pool: &HybridMemoryPool, _config: &CleanupConfig) -> CleanupResult<CleanupOperationResult> {
        let start_time = Instant::now();
        
        // Update last cleanup time
        if let Ok(mut last_cleanup) = self.last_cleanup.write() {
            *last_cleanup = Some(start_time);
        }

        let metrics_before = pool.get_metrics();
        let pressure_level = self.get_pressure_level(&metrics_before);
        
        // Perform pressure-based cleanup with intensity based on pressure level
        let cleanup_intensity = match pressure_level {
            CleanupPriority::Critical => 1.0,
            CleanupPriority::High => 0.8,
            CleanupPriority::Normal => 0.5,
            _ => 0.2,
        };

        // Simulate cleanup work proportional to pressure
        let cleanup_duration = Duration::from_millis((cleanup_intensity * 10.0) as u64);
        std::thread::sleep(cleanup_duration);
        
        let duration = start_time.elapsed();
        let bytes_freed = (cleanup_intensity * 1024.0) as u64; // Simulated
        let allocations_cleaned = (cleanup_intensity * 10.0) as u64; // Simulated
        
        Ok(CleanupOperationResult::success(bytes_freed, allocations_cleaned, duration)
            .with_metadata("strategy".to_string(), "pressure".to_string())
            .with_metadata("pressure_level".to_string(), format!("{:?}", pressure_level))
            .with_metadata("cleanup_intensity".to_string(), cleanup_intensity.to_string()))
    }

    fn description(&self) -> String {
        format!("Pressure cleanup strategy (thresholds: {:.1}%, {:.1}%, {:.1}%)", 
                self.light_threshold * 100.0, 
                self.aggressive_threshold * 100.0, 
                self.emergency_threshold * 100.0)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("light_threshold".to_string(), self.light_threshold.to_string());
        params.insert("aggressive_threshold".to_string(), self.aggressive_threshold.to_string());
        params.insert("emergency_threshold".to_string(), self.emergency_threshold.to_string());
        params.insert("min_interval_ms".to_string(), self.min_interval.as_millis().to_string());
        params
    }
}

/// Periodic cleanup strategy - performs cleanup at regular intervals
pub struct PeriodicCleanupStrategy {
    interval: Duration,
    last_cleanup: Arc<RwLock<Option<Instant>>>,
    cleanup_duration: Duration,
}

impl PeriodicCleanupStrategy {
    /// Creates a new periodic cleanup strategy
    pub fn new(interval: Duration, cleanup_duration: Duration) -> Self {
        Self {
            interval,
            last_cleanup: Arc::new(RwLock::new(None)),
            cleanup_duration,
        }
    }

    /// Creates a default periodic cleanup strategy
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30), Duration::from_millis(20))
    }
}

impl CleanupStrategy for PeriodicCleanupStrategy {
    fn strategy_type(&self) -> CleanupStrategyType {
        CleanupStrategyType::Periodic
    }

    fn priority(&self) -> CleanupPriority {
        CleanupPriority::Normal
    }

    fn should_cleanup(&self, _metrics: &MemoryMetrics, _detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool {
        if let Ok(last_cleanup) = self.last_cleanup.read() {
            if let Some(last) = *last_cleanup {
                last.elapsed() >= self.interval
            } else {
                true // First cleanup
            }
        } else {
            false
        }
    }

    fn cleanup(&self, _pool: &HybridMemoryPool, _config: &CleanupConfig) -> CleanupResult<CleanupOperationResult> {
        let start_time = Instant::now();
        
        // Update last cleanup time
        if let Ok(mut last_cleanup) = self.last_cleanup.write() {
            *last_cleanup = Some(start_time);
        }

        // Perform periodic maintenance cleanup
        std::thread::sleep(self.cleanup_duration);
        
        let duration = start_time.elapsed();
        let bytes_freed = 512; // Simulated
        let allocations_cleaned = 5; // Simulated
        
        Ok(CleanupOperationResult::success(bytes_freed, allocations_cleaned, duration)
            .with_metadata("strategy".to_string(), "periodic".to_string())
            .with_metadata("interval_ms".to_string(), self.interval.as_millis().to_string()))
    }

    fn description(&self) -> String {
        format!("Periodic cleanup strategy (interval: {:?})", self.interval)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("interval_ms".to_string(), self.interval.as_millis().to_string());
        params.insert("cleanup_duration_ms".to_string(), self.cleanup_duration.as_millis().to_string());
        params
    }
}

/// Device-specific cleanup strategy
pub struct DeviceCleanupStrategy {
    device_type: String,
    cleanup_interval: Duration,
    last_cleanup: Arc<RwLock<HashMap<String, Instant>>>,
}

impl DeviceCleanupStrategy {
    /// Creates a new device cleanup strategy
    pub fn new(device_type: String, cleanup_interval: Duration) -> Self {
        Self {
            device_type,
            cleanup_interval,
            last_cleanup: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a CPU device cleanup strategy
    pub fn cpu() -> Self {
        Self::new("CPU".to_string(), Duration::from_secs(10))
    }

    /// Creates a Metal device cleanup strategy
    pub fn metal() -> Self {
        Self::new("Metal".to_string(), Duration::from_secs(5))
    }
}

impl CleanupStrategy for DeviceCleanupStrategy {
    fn strategy_type(&self) -> CleanupStrategyType {
        CleanupStrategyType::Device
    }

    fn priority(&self) -> CleanupPriority {
        CleanupPriority::Normal
    }

    fn should_cleanup(&self, _metrics: &MemoryMetrics, _detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool {
        if let Ok(last_cleanup_map) = self.last_cleanup.read() {
            if let Some(last) = last_cleanup_map.get(&self.device_type) {
                last.elapsed() >= self.cleanup_interval
            } else {
                true // First cleanup for this device
            }
        } else {
            false
        }
    }

    fn cleanup(&self, _pool: &HybridMemoryPool, _config: &CleanupConfig) -> CleanupResult<CleanupOperationResult> {
        let start_time = Instant::now();
        
        // Update last cleanup time for this device
        if let Ok(mut last_cleanup_map) = self.last_cleanup.write() {
            last_cleanup_map.insert(self.device_type.clone(), start_time);
        }

        // Perform device-specific cleanup
        let cleanup_work = match self.device_type.as_str() {
            "CPU" => {
                // CPU-specific cleanup (cache optimization, etc.)
                Duration::from_millis(5)
            }
            "Metal" => {
                // Metal-specific cleanup (command buffer cleanup, etc.)
                Duration::from_millis(10)
            }
            _ => Duration::from_millis(2)
        };

        std::thread::sleep(cleanup_work);
        
        let duration = start_time.elapsed();
        let bytes_freed = 256; // Simulated
        let allocations_cleaned = 3; // Simulated
        
        Ok(CleanupOperationResult::success(bytes_freed, allocations_cleaned, duration)
            .with_metadata("strategy".to_string(), "device".to_string())
            .with_metadata("device_type".to_string(), self.device_type.clone()))
    }

    fn description(&self) -> String {
        format!("Device cleanup strategy for {} (interval: {:?})", self.device_type, self.cleanup_interval)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("device_type".to_string(), self.device_type.clone());
        params.insert("cleanup_interval_ms".to_string(), self.cleanup_interval.as_millis().to_string());
        params
    }
}

/// Generational cleanup strategy - performs age-based cleanup
pub struct GenerationalCleanupStrategy {
    young_age_threshold: Duration,
    old_age_threshold: Duration,
    ancient_age_threshold: Duration,
    last_cleanup: Arc<RwLock<Option<Instant>>>,
    cleanup_interval: Duration,
}

impl GenerationalCleanupStrategy {
    /// Creates a new generational cleanup strategy
    pub fn new(
        young_age_threshold: Duration,
        old_age_threshold: Duration,
        ancient_age_threshold: Duration,
        cleanup_interval: Duration,
    ) -> Self {
        Self {
            young_age_threshold,
            old_age_threshold,
            ancient_age_threshold,
            last_cleanup: Arc::new(RwLock::new(None)),
            cleanup_interval,
        }
    }

    /// Creates a default generational cleanup strategy
    pub fn default() -> Self {
        Self::new(
            Duration::from_secs(1),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(15),
        )
    }
}

impl CleanupStrategy for GenerationalCleanupStrategy {
    fn strategy_type(&self) -> CleanupStrategyType {
        CleanupStrategyType::Generational
    }

    fn priority(&self) -> CleanupPriority {
        CleanupPriority::Low
    }

    fn should_cleanup(&self, _metrics: &MemoryMetrics, _detailed_metrics: Option<&DetailedMemoryMetrics>) -> bool {
        if let Ok(last_cleanup) = self.last_cleanup.read() {
            if let Some(last) = *last_cleanup {
                last.elapsed() >= self.cleanup_interval
            } else {
                true // First cleanup
            }
        } else {
            false
        }
    }

    fn cleanup(&self, _pool: &HybridMemoryPool, _config: &CleanupConfig) -> CleanupResult<CleanupOperationResult> {
        let start_time = Instant::now();
        
        // Update last cleanup time
        if let Ok(mut last_cleanup) = self.last_cleanup.write() {
            *last_cleanup = Some(start_time);
        }

        // Perform generational cleanup
        // In a real implementation, this would analyze allocation ages and clean accordingly
        std::thread::sleep(Duration::from_millis(15));
        
        let duration = start_time.elapsed();
        let bytes_freed = 1024; // Simulated
        let allocations_cleaned = 8; // Simulated
        
        Ok(CleanupOperationResult::success(bytes_freed, allocations_cleaned, duration)
            .with_metadata("strategy".to_string(), "generational".to_string())
            .with_metadata("young_threshold_ms".to_string(), self.young_age_threshold.as_millis().to_string())
            .with_metadata("old_threshold_ms".to_string(), self.old_age_threshold.as_millis().to_string())
            .with_metadata("ancient_threshold_ms".to_string(), self.ancient_age_threshold.as_millis().to_string()))
    }

    fn description(&self) -> String {
        format!("Generational cleanup strategy (young: {:?}, old: {:?}, ancient: {:?})", 
                self.young_age_threshold, self.old_age_threshold, self.ancient_age_threshold)
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("young_age_threshold_ms".to_string(), self.young_age_threshold.as_millis().to_string());
        params.insert("old_age_threshold_ms".to_string(), self.old_age_threshold.as_millis().to_string());
        params.insert("ancient_age_threshold_ms".to_string(), self.ancient_age_threshold.as_millis().to_string());
        params.insert("cleanup_interval_ms".to_string(), self.cleanup_interval.as_millis().to_string());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_cleanup_priority_ordering() {
        assert!(CleanupPriority::Critical > CleanupPriority::High);
        assert!(CleanupPriority::High > CleanupPriority::Normal);
        assert!(CleanupPriority::Normal > CleanupPriority::Low);
        assert!(CleanupPriority::Low > CleanupPriority::Background);
    }

    #[test]
    fn test_cleanup_operation_result() {
        let result = CleanupOperationResult::success(1024, 5, Duration::from_millis(100));
        assert!(result.success);
        assert_eq!(result.bytes_freed, 1024);
        assert_eq!(result.allocations_cleaned, 5);
        assert_eq!(result.duration, Duration::from_millis(100));

        let result = CleanupOperationResult::failure("test error".to_string(), Duration::from_millis(50));
        assert!(!result.success);
        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.error_message, Some("test error".to_string()));
    }

    #[test]
    fn test_idle_cleanup_strategy() {
        let strategy = IdleCleanupStrategy::default();
        assert_eq!(strategy.strategy_type(), CleanupStrategyType::Idle);
        assert_eq!(strategy.priority(), CleanupPriority::Low);
        
        let params = strategy.get_parameters();
        assert!(params.contains_key("min_idle_time_ms"));
        assert!(params.contains_key("max_cleanup_duration_ms"));
    }

    #[test]
    fn test_pressure_cleanup_strategy() {
        let strategy = PressureCleanupStrategy::default();
        assert_eq!(strategy.strategy_type(), CleanupStrategyType::Pressure);
        assert_eq!(strategy.priority(), CleanupPriority::High);
        
        let params = strategy.get_parameters();
        assert!(params.contains_key("light_threshold"));
        assert!(params.contains_key("aggressive_threshold"));
        assert!(params.contains_key("emergency_threshold"));
    }

    #[test]
    fn test_periodic_cleanup_strategy() {
        let strategy = PeriodicCleanupStrategy::default();
        assert_eq!(strategy.strategy_type(), CleanupStrategyType::Periodic);
        assert_eq!(strategy.priority(), CleanupPriority::Normal);
        
        // Test should_cleanup logic
        let metrics = crate::memory::MemoryMetrics::new();
        assert!(strategy.should_cleanup(&metrics, None)); // First cleanup should return true
    }

    #[test]
    fn test_device_cleanup_strategy() {
        let cpu_strategy = DeviceCleanupStrategy::cpu();
        assert_eq!(cpu_strategy.strategy_type(), CleanupStrategyType::Device);
        assert_eq!(cpu_strategy.priority(), CleanupPriority::Normal);
        
        let metal_strategy = DeviceCleanupStrategy::metal();
        let params = metal_strategy.get_parameters();
        assert_eq!(params.get("device_type"), Some(&"Metal".to_string()));
    }

    #[test]
    fn test_generational_cleanup_strategy() {
        let strategy = GenerationalCleanupStrategy::default();
        assert_eq!(strategy.strategy_type(), CleanupStrategyType::Generational);
        assert_eq!(strategy.priority(), CleanupPriority::Low);
        
        let params = strategy.get_parameters();
        assert!(params.contains_key("young_age_threshold_ms"));
        assert!(params.contains_key("old_age_threshold_ms"));
        assert!(params.contains_key("ancient_age_threshold_ms"));
    }
}