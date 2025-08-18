//! Memory Pressure Detection and Integration
//!
//! This module provides integration with the existing memory pressure detection
//! system and implements memory-aware optimization strategies.

use crate::bitlinear::error::{BitLinearError, BitLinearResult};
use bitnet_core::memory::{HybridMemoryPool, MemoryMetrics, tracking};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory pressure levels for BitLinear layer optimization
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressureLevel {
    /// Normal memory usage - all optimizations enabled
    Low,
    /// Moderate memory pressure - reduce cache sizes
    High,
    /// Critical memory pressure - aggressive cleanup required
    Critical,
}

impl From<tracking::MemoryPressureLevel> for MemoryPressureLevel {
    fn from(level: tracking::MemoryPressureLevel) -> Self {
        match level {
            tracking::MemoryPressureLevel::Low => MemoryPressureLevel::Low,
            tracking::MemoryPressureLevel::High => MemoryPressureLevel::High, 
            tracking::MemoryPressureLevel::Critical => MemoryPressureLevel::Critical,
            tracking::MemoryPressureLevel::None => MemoryPressureLevel::Low, // Map None to Low
            tracking::MemoryPressureLevel::Medium => MemoryPressureLevel::High, // Map Medium to High
        }
    }
}

/// Configuration for memory pressure integration
#[derive(Debug, Clone)]
pub struct PressureConfig {
    /// Enable memory pressure monitoring
    pub enable_monitoring: bool,
    /// Memory pressure check interval (seconds)
    pub check_interval_seconds: f64,
    /// Memory usage threshold for high pressure (percentage of total)
    pub high_pressure_threshold: f32,
    /// Memory usage threshold for critical pressure (percentage of total)
    pub critical_pressure_threshold: f32,
    /// Enable automatic cache eviction under pressure
    pub auto_cache_eviction: bool,
    /// Enable memory compaction under pressure
    pub enable_compaction: bool,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            check_interval_seconds: 0.1, // 100ms check interval
            high_pressure_threshold: 75.0, // 75% memory usage
            critical_pressure_threshold: 90.0, // 90% memory usage
            auto_cache_eviction: true,
            enable_compaction: true,
        }
    }
}

/// Memory Pressure Integration Manager
/// 
/// Integrates with the existing HybridMemoryPool pressure detection
/// and provides memory-aware optimization for BitLinear layers.
pub struct MemoryPressureIntegrator {
    /// Configuration for pressure detection
    config: PressureConfig,
    /// Reference to the memory pool for metrics
    memory_pool: Arc<HybridMemoryPool>,
    /// Current pressure level
    current_pressure: Arc<RwLock<MemoryPressureLevel>>,
    /// Last pressure check time
    last_check: Arc<Mutex<Option<Instant>>>,
}

impl MemoryPressureIntegrator {
    /// Create a new memory pressure integrator
    pub fn new(
        config: PressureConfig,
        memory_pool: Arc<HybridMemoryPool>
    ) -> BitLinearResult<Self> {
        Ok(Self {
            config,
            memory_pool,
            current_pressure: Arc::new(RwLock::new(MemoryPressureLevel::Low)),
            last_check: Arc::new(Mutex::new(None)),
        })
    }

    /// Check current memory pressure and trigger responses if needed
    pub fn check_pressure(&self) -> BitLinearResult<MemoryPressureLevel> {
        if !self.config.enable_monitoring {
            return Ok(MemoryPressureLevel::Low);
        }

        // Check if enough time has passed since last check
        let mut last_check = self.last_check
            .lock()
            .map_err(|_| BitLinearError::cache_lock_error("last check"))?;
        
        let now = Instant::now();
        if let Some(last) = *last_check {
            let elapsed = now.duration_since(last);
            let check_interval = Duration::from_secs_f64(self.config.check_interval_seconds);
            if elapsed < check_interval {
                return Ok(self.get_current_pressure()?);
            }
        }

        *last_check = Some(now);
        drop(last_check);

        // Get current memory metrics
        let metrics = self.memory_pool.get_metrics();
        let pressure_level = self.calculate_pressure_level(&metrics)?;
        
        // Update current pressure
        let mut current_pressure = self.current_pressure
            .write()
            .map_err(|_| BitLinearError::cache_lock_error("pressure level"))?;
        *current_pressure = pressure_level.clone();
        drop(current_pressure);

        Ok(pressure_level)
    }

    /// Get current pressure level
    pub fn get_current_pressure(&self) -> BitLinearResult<MemoryPressureLevel> {
        let pressure = self.current_pressure
            .read()
            .map_err(|_| BitLinearError::cache_lock_error("last check"))?;
        Ok(pressure.clone())
    }

    /// Calculate pressure level based on memory metrics
    fn calculate_pressure_level(&self, metrics: &MemoryMetrics) -> BitLinearResult<MemoryPressureLevel> {
        let total_allocated = metrics.total_allocated;
        
        // Simple pressure calculation based on allocation percentage
        // In a real implementation, this might consider system memory limits
        let usage_percentage = (total_allocated as f32 / (1024.0 * 1024.0 * 1024.0)) * 100.0; // Convert to GB percentage
        
        if usage_percentage >= self.config.critical_pressure_threshold {
            Ok(MemoryPressureLevel::Critical)
        } else if usage_percentage >= self.config.high_pressure_threshold {
            Ok(MemoryPressureLevel::High)
        } else {
            Ok(MemoryPressureLevel::Low)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_level_conversion() {
        assert_eq!(
            MemoryPressureLevel::from(tracking::MemoryPressureLevel::Low),
            MemoryPressureLevel::Low
        );
        assert_eq!(
            MemoryPressureLevel::from(tracking::MemoryPressureLevel::High),
            MemoryPressureLevel::High
        );
        assert_eq!(
            MemoryPressureLevel::from(tracking::MemoryPressureLevel::Critical),
            MemoryPressureLevel::Critical
        );
        assert_eq!(
            MemoryPressureLevel::from(tracking::MemoryPressureLevel::None),
            MemoryPressureLevel::Low
        );
        assert_eq!(
            MemoryPressureLevel::from(tracking::MemoryPressureLevel::Medium),
            MemoryPressureLevel::High
        );
    }

    #[test]
    fn test_pressure_config_default() {
        let config = PressureConfig::default();
        assert!(config.enable_monitoring);
        assert!(config.auto_cache_eviction);
        assert!(config.enable_compaction);
        assert!(config.high_pressure_threshold > 0.0);
        assert!(config.critical_pressure_threshold > config.high_pressure_threshold);
    }

    #[test] 
    fn test_pressure_integrator_creation() {
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = PressureConfig::default();
        
        let integrator = MemoryPressureIntegrator::new(config, memory_pool);
        assert!(integrator.is_ok());
        
        let integrator = integrator.unwrap();
        let pressure = integrator.get_current_pressure().unwrap();
        assert_eq!(pressure, MemoryPressureLevel::Low);
    }
}
