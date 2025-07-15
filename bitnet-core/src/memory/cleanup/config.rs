//! Cleanup Configuration System
//!
//! This module provides comprehensive configuration options for the cleanup system,
//! allowing fine-tuning of cleanup behavior, thresholds, and policies to match
//! specific application requirements and hardware characteristics.

use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Comprehensive configuration for the cleanup system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupConfig {
    /// General cleanup policies
    pub policy: CleanupPolicy,
    /// Thresholds for triggering cleanup operations
    pub thresholds: CleanupThresholds,
    /// Feature flags for enabling/disabling specific cleanup features
    pub features: CleanupFeatureFlags,
    /// Scheduler configuration
    pub scheduler: SchedulerConfig,
    /// Device-specific configurations
    pub device_configs: DeviceConfigs,
    /// Safety and validation settings
    pub safety: SafetyConfig,
}

/// General cleanup policies and behavior settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupPolicy {
    /// Whether to enable automatic cleanup
    pub enable_automatic_cleanup: bool,
    /// Whether to enable manual cleanup APIs
    pub enable_manual_cleanup: bool,
    /// Default cleanup strategy to use
    pub default_strategy: CleanupStrategyType,
    /// Maximum time to spend in a single cleanup operation
    pub max_cleanup_duration: Duration,
    /// Maximum number of allocations to clean in a single operation
    pub max_allocations_per_cleanup: usize,
    /// Whether to perform incremental cleanup (spread across multiple cycles)
    pub enable_incremental_cleanup: bool,
    /// Size of incremental cleanup batches
    pub incremental_batch_size: usize,
    /// Whether to use cooperative cleanup (allow applications to participate)
    pub enable_cooperative_cleanup: bool,
}

/// Thresholds for triggering different types of cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupThresholds {
    /// Memory pressure thresholds
    pub pressure: PressureThresholds,
    /// Idle time thresholds
    pub idle: IdleThresholds,
    /// Age-based thresholds for generational cleanup
    pub age: AgeThresholds,
    /// Size-based thresholds
    pub size: SizeThresholds,
    /// Fragmentation thresholds
    pub fragmentation: FragmentationThresholds,
}

/// Memory pressure thresholds for triggering cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureThresholds {
    /// Memory usage percentage to trigger light cleanup (0.0-1.0)
    pub light_cleanup_threshold: f64,
    /// Memory usage percentage to trigger aggressive cleanup (0.0-1.0)
    pub aggressive_cleanup_threshold: f64,
    /// Memory usage percentage to trigger emergency cleanup (0.0-1.0)
    pub emergency_cleanup_threshold: f64,
    /// Minimum time between pressure-triggered cleanups
    pub min_pressure_cleanup_interval: Duration,
}

/// Idle time thresholds for triggering cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdleThresholds {
    /// Minimum idle time before triggering cleanup
    pub min_idle_time: Duration,
    /// Idle time for light cleanup
    pub light_cleanup_idle_time: Duration,
    /// Idle time for aggressive cleanup
    pub aggressive_cleanup_idle_time: Duration,
    /// Maximum idle cleanup duration
    pub max_idle_cleanup_duration: Duration,
}

/// Age-based thresholds for generational cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeThresholds {
    /// Age threshold for young generation cleanup
    pub young_generation_age: Duration,
    /// Age threshold for old generation cleanup
    pub old_generation_age: Duration,
    /// Age threshold for ancient generation cleanup
    pub ancient_generation_age: Duration,
    /// Minimum age before an allocation can be cleaned
    pub min_cleanup_age: Duration,
}

/// Size-based thresholds for cleanup decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeThresholds {
    /// Minimum allocation size to consider for cleanup
    pub min_cleanup_size: usize,
    /// Size threshold for prioritizing large allocations
    pub large_allocation_threshold: usize,
    /// Total memory threshold for triggering cleanup
    pub total_memory_threshold: usize,
    /// Pool utilization threshold (0.0-1.0)
    pub pool_utilization_threshold: f64,
}

/// Fragmentation thresholds for cleanup decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationThresholds {
    /// Fragmentation ratio to trigger defragmentation (0.0-1.0)
    pub defragmentation_threshold: f64,
    /// Minimum free space ratio for compaction (0.0-1.0)
    pub compaction_threshold: f64,
    /// Maximum fragmentation before emergency cleanup (0.0-1.0)
    pub emergency_fragmentation_threshold: f64,
}

/// Feature flags for enabling/disabling cleanup features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupFeatureFlags {
    /// Enable idle cleanup strategy
    pub enable_idle_cleanup: bool,
    /// Enable pressure-based cleanup strategy
    pub enable_pressure_cleanup: bool,
    /// Enable periodic cleanup strategy
    pub enable_periodic_cleanup: bool,
    /// Enable device-specific cleanup
    pub enable_device_cleanup: bool,
    /// Enable generational cleanup strategy
    pub enable_generational_cleanup: bool,
    /// Enable smart cleanup using allocation patterns
    pub enable_smart_cleanup: bool,
    /// Enable cleanup metrics collection
    pub enable_cleanup_metrics: bool,
    /// Enable cleanup operation logging
    pub enable_cleanup_logging: bool,
    /// Enable cleanup performance profiling
    pub enable_cleanup_profiling: bool,
    /// Enable emergency cleanup mechanisms
    pub enable_emergency_cleanup: bool,
}

/// Scheduler configuration for automatic cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Whether the scheduler is enabled
    pub enabled: bool,
    /// Base interval for periodic cleanup checks
    pub base_interval: Duration,
    /// Maximum number of concurrent cleanup operations
    pub max_concurrent_operations: usize,
    /// Priority levels for different cleanup types
    pub priority_levels: PriorityConfig,
    /// Scheduler thread pool size
    pub thread_pool_size: usize,
    /// Whether to use adaptive scheduling based on system load
    pub adaptive_scheduling: bool,
}

/// Priority configuration for cleanup operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityConfig {
    /// Priority for emergency cleanup (highest)
    pub emergency_priority: u8,
    /// Priority for pressure-based cleanup
    pub pressure_priority: u8,
    /// Priority for idle cleanup
    pub idle_priority: u8,
    /// Priority for periodic cleanup
    pub periodic_priority: u8,
    /// Priority for generational cleanup (lowest)
    pub generational_priority: u8,
}

/// Device-specific cleanup configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfigs {
    /// CPU-specific cleanup configuration
    pub cpu: CpuCleanupConfig,
    /// Metal GPU-specific cleanup configuration
    pub metal: MetalCleanupConfig,
}

/// CPU-specific cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCleanupConfig {
    /// Enable CPU cache optimization during cleanup
    pub enable_cache_optimization: bool,
    /// Enable NUMA-aware cleanup
    pub enable_numa_awareness: bool,
    /// Enable memory prefetching during cleanup
    pub enable_prefetching: bool,
    /// CPU cache line size for optimization
    pub cache_line_size: usize,
    /// Number of CPU cores to use for parallel cleanup
    pub parallel_cleanup_cores: Option<usize>,
}

/// Metal GPU-specific cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCleanupConfig {
    /// Enable Metal command buffer cleanup
    pub enable_command_buffer_cleanup: bool,
    /// Enable unified memory optimization
    pub enable_unified_memory_optimization: bool,
    /// Enable Metal Performance Shaders cleanup
    pub enable_mps_cleanup: bool,
    /// Command buffer cleanup interval
    pub command_buffer_cleanup_interval: Duration,
    /// Maximum number of command buffers to clean per operation
    pub max_command_buffers_per_cleanup: usize,
}

/// Safety and validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Enable comprehensive safety validation
    pub enable_safety_validation: bool,
    /// Enable cleanup operation rollback on failure
    pub enable_rollback: bool,
    /// Enable cleanup operation auditing
    pub enable_auditing: bool,
    /// Maximum number of validation retries
    pub max_validation_retries: usize,
    /// Timeout for safety validation operations
    pub validation_timeout: Duration,
    /// Enable memory corruption detection
    pub enable_corruption_detection: bool,
}

/// Cleanup strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CleanupStrategyType {
    /// Idle cleanup - clean during idle periods
    Idle,
    /// Pressure cleanup - clean under memory pressure
    Pressure,
    /// Periodic cleanup - clean at regular intervals
    Periodic,
    /// Device cleanup - device-specific cleanup
    Device,
    /// Generational cleanup - age-based cleanup
    Generational,
    /// Smart cleanup - pattern-based cleanup
    Smart,
    /// Emergency cleanup - last resort cleanup
    Emergency,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            policy: CleanupPolicy::default(),
            thresholds: CleanupThresholds::default(),
            features: CleanupFeatureFlags::default(),
            scheduler: SchedulerConfig::default(),
            device_configs: DeviceConfigs::default(),
            safety: SafetyConfig::default(),
        }
    }
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            enable_automatic_cleanup: true,
            enable_manual_cleanup: true,
            default_strategy: CleanupStrategyType::Idle,
            max_cleanup_duration: Duration::from_millis(100),
            max_allocations_per_cleanup: 1000,
            enable_incremental_cleanup: true,
            incremental_batch_size: 100,
            enable_cooperative_cleanup: false,
        }
    }
}

impl Default for CleanupThresholds {
    fn default() -> Self {
        Self {
            pressure: PressureThresholds::default(),
            idle: IdleThresholds::default(),
            age: AgeThresholds::default(),
            size: SizeThresholds::default(),
            fragmentation: FragmentationThresholds::default(),
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            light_cleanup_threshold: 0.7,
            aggressive_cleanup_threshold: 0.85,
            emergency_cleanup_threshold: 0.95,
            min_pressure_cleanup_interval: Duration::from_secs(1),
        }
    }
}

impl Default for IdleThresholds {
    fn default() -> Self {
        Self {
            min_idle_time: Duration::from_millis(100),
            light_cleanup_idle_time: Duration::from_secs(1),
            aggressive_cleanup_idle_time: Duration::from_secs(5),
            max_idle_cleanup_duration: Duration::from_millis(50),
        }
    }
}

impl Default for AgeThresholds {
    fn default() -> Self {
        Self {
            young_generation_age: Duration::from_secs(1),
            old_generation_age: Duration::from_secs(10),
            ancient_generation_age: Duration::from_secs(60),
            min_cleanup_age: Duration::from_millis(100),
        }
    }
}

impl Default for SizeThresholds {
    fn default() -> Self {
        Self {
            min_cleanup_size: 1024, // 1KB
            large_allocation_threshold: 1024 * 1024, // 1MB
            total_memory_threshold: 512 * 1024 * 1024, // 512MB
            pool_utilization_threshold: 0.8,
        }
    }
}

impl Default for FragmentationThresholds {
    fn default() -> Self {
        Self {
            defragmentation_threshold: 0.3,
            compaction_threshold: 0.5,
            emergency_fragmentation_threshold: 0.8,
        }
    }
}

impl Default for CleanupFeatureFlags {
    fn default() -> Self {
        Self {
            enable_idle_cleanup: true,
            enable_pressure_cleanup: true,
            enable_periodic_cleanup: true,
            enable_device_cleanup: true,
            enable_generational_cleanup: true,
            enable_smart_cleanup: false, // Disabled by default as it's more experimental
            enable_cleanup_metrics: true,
            enable_cleanup_logging: true,
            enable_cleanup_profiling: false,
            enable_emergency_cleanup: true,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_interval: Duration::from_millis(500),
            max_concurrent_operations: 2,
            priority_levels: PriorityConfig::default(),
            thread_pool_size: 2,
            adaptive_scheduling: true,
        }
    }
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            emergency_priority: 255,
            pressure_priority: 200,
            idle_priority: 100,
            periodic_priority: 150,
            generational_priority: 50,
        }
    }
}

impl Default for DeviceConfigs {
    fn default() -> Self {
        Self {
            cpu: CpuCleanupConfig::default(),
            metal: MetalCleanupConfig::default(),
        }
    }
}

impl Default for CpuCleanupConfig {
    fn default() -> Self {
        Self {
            enable_cache_optimization: true,
            enable_numa_awareness: false, // Disabled by default as it's platform-specific
            enable_prefetching: false,
            cache_line_size: 64,
            parallel_cleanup_cores: None, // Auto-detect
        }
    }
}

impl Default for MetalCleanupConfig {
    fn default() -> Self {
        Self {
            enable_command_buffer_cleanup: true,
            enable_unified_memory_optimization: true,
            enable_mps_cleanup: false,
            command_buffer_cleanup_interval: Duration::from_secs(1),
            max_command_buffers_per_cleanup: 10,
        }
    }
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            enable_safety_validation: true,
            enable_rollback: true,
            enable_auditing: false,
            max_validation_retries: 3,
            validation_timeout: Duration::from_millis(10),
            enable_corruption_detection: true,
        }
    }
}

impl CleanupConfig {
    /// Creates a minimal configuration for low-overhead cleanup
    pub fn minimal() -> Self {
        Self {
            policy: CleanupPolicy {
                enable_automatic_cleanup: true,
                enable_manual_cleanup: true,
                default_strategy: CleanupStrategyType::Idle,
                max_cleanup_duration: Duration::from_millis(10),
                max_allocations_per_cleanup: 100,
                enable_incremental_cleanup: false,
                incremental_batch_size: 50,
                enable_cooperative_cleanup: false,
            },
            features: CleanupFeatureFlags {
                enable_idle_cleanup: true,
                enable_pressure_cleanup: true,
                enable_periodic_cleanup: false,
                enable_device_cleanup: false,
                enable_generational_cleanup: false,
                enable_smart_cleanup: false,
                enable_cleanup_metrics: false,
                enable_cleanup_logging: false,
                enable_cleanup_profiling: false,
                enable_emergency_cleanup: true,
            },
            safety: SafetyConfig {
                enable_safety_validation: false,
                enable_rollback: false,
                enable_auditing: false,
                max_validation_retries: 1,
                validation_timeout: Duration::from_millis(1),
                enable_corruption_detection: false,
            },
            ..Default::default()
        }
    }

    /// Creates an aggressive configuration for maximum cleanup efficiency
    pub fn aggressive() -> Self {
        Self {
            policy: CleanupPolicy {
                enable_automatic_cleanup: true,
                enable_manual_cleanup: true,
                default_strategy: CleanupStrategyType::Pressure,
                max_cleanup_duration: Duration::from_millis(500),
                max_allocations_per_cleanup: 5000,
                enable_incremental_cleanup: true,
                incremental_batch_size: 500,
                enable_cooperative_cleanup: true,
            },
            thresholds: CleanupThresholds {
                pressure: PressureThresholds {
                    light_cleanup_threshold: 0.6,
                    aggressive_cleanup_threshold: 0.75,
                    emergency_cleanup_threshold: 0.9,
                    min_pressure_cleanup_interval: Duration::from_millis(500),
                },
                idle: IdleThresholds {
                    min_idle_time: Duration::from_millis(50),
                    light_cleanup_idle_time: Duration::from_millis(500),
                    aggressive_cleanup_idle_time: Duration::from_secs(2),
                    max_idle_cleanup_duration: Duration::from_millis(200),
                },
                ..Default::default()
            },
            features: CleanupFeatureFlags {
                enable_idle_cleanup: true,
                enable_pressure_cleanup: true,
                enable_periodic_cleanup: true,
                enable_device_cleanup: true,
                enable_generational_cleanup: true,
                enable_smart_cleanup: true,
                enable_cleanup_metrics: true,
                enable_cleanup_logging: true,
                enable_cleanup_profiling: true,
                enable_emergency_cleanup: true,
            },
            ..Default::default()
        }
    }

    /// Creates a debug configuration with extensive logging and validation
    pub fn debug() -> Self {
        Self {
            features: CleanupFeatureFlags {
                enable_cleanup_metrics: true,
                enable_cleanup_logging: true,
                enable_cleanup_profiling: true,
                ..Default::default()
            },
            safety: SafetyConfig {
                enable_safety_validation: true,
                enable_rollback: true,
                enable_auditing: true,
                max_validation_retries: 5,
                validation_timeout: Duration::from_millis(100),
                enable_corruption_detection: true,
            },
            ..Default::default()
        }
    }

    /// Validates the configuration for consistency and correctness
    pub fn validate(&self) -> Result<(), String> {
        // Validate pressure thresholds
        let p = &self.thresholds.pressure;
        if p.light_cleanup_threshold >= p.aggressive_cleanup_threshold {
            return Err("Light cleanup threshold must be less than aggressive threshold".to_string());
        }
        if p.aggressive_cleanup_threshold >= p.emergency_cleanup_threshold {
            return Err("Aggressive cleanup threshold must be less than emergency threshold".to_string());
        }
        if p.emergency_cleanup_threshold > 1.0 {
            return Err("Emergency cleanup threshold cannot exceed 1.0".to_string());
        }

        // Validate idle thresholds
        let i = &self.thresholds.idle;
        if i.min_idle_time >= i.light_cleanup_idle_time {
            return Err("Minimum idle time must be less than light cleanup idle time".to_string());
        }
        if i.light_cleanup_idle_time >= i.aggressive_cleanup_idle_time {
            return Err("Light cleanup idle time must be less than aggressive cleanup idle time".to_string());
        }

        // Validate age thresholds
        let a = &self.thresholds.age;
        if a.young_generation_age >= a.old_generation_age {
            return Err("Young generation age must be less than old generation age".to_string());
        }
        if a.old_generation_age >= a.ancient_generation_age {
            return Err("Old generation age must be less than ancient generation age".to_string());
        }

        // Validate policy settings
        if self.policy.max_allocations_per_cleanup == 0 {
            return Err("Max allocations per cleanup must be greater than 0".to_string());
        }
        if self.policy.incremental_batch_size == 0 {
            return Err("Incremental batch size must be greater than 0".to_string());
        }

        // Validate scheduler settings
        if self.scheduler.max_concurrent_operations == 0 {
            return Err("Max concurrent operations must be greater than 0".to_string());
        }
        if self.scheduler.thread_pool_size == 0 {
            return Err("Thread pool size must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = CleanupConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_minimal_config_validation() {
        let config = CleanupConfig::minimal();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_aggressive_config_validation() {
        let config = CleanupConfig::aggressive();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_debug_config_validation() {
        let config = CleanupConfig::debug();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_pressure_thresholds() {
        let mut config = CleanupConfig::default();
        config.thresholds.pressure.light_cleanup_threshold = 0.9;
        config.thresholds.pressure.aggressive_cleanup_threshold = 0.8;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_idle_thresholds() {
        let mut config = CleanupConfig::default();
        config.thresholds.idle.min_idle_time = Duration::from_secs(10);
        config.thresholds.idle.light_cleanup_idle_time = Duration::from_secs(5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_policy_settings() {
        let mut config = CleanupConfig::default();
        config.policy.max_allocations_per_cleanup = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cleanup_strategy_type_serialization() {
        let strategy = CleanupStrategyType::Pressure;
        let serialized = serde_json::to_string(&strategy).unwrap();
        let deserialized: CleanupStrategyType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(strategy, deserialized);
    }
}