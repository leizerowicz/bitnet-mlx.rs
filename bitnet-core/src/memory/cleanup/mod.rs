//! Comprehensive Tensor Cleanup System
//!
//! This module provides a robust, configurable cleanup system for the BitNet memory
//! management infrastructure. It implements automatic and manual cleanup mechanisms
//! with multiple strategies to efficiently reclaim unused memory across different
//! device types and usage patterns.
//!
//! # Architecture
//!
//! The cleanup system consists of several key components:
//!
//! - **CleanupManager**: Main interface for coordinating all cleanup operations
//! - **CleanupScheduler**: Handles scheduling and timing of cleanup operations
//! - **CleanupStrategy**: Pluggable strategies for different cleanup scenarios
//! - **CleanupMetrics**: Comprehensive tracking of cleanup effectiveness
//! - **Device-specific cleanup**: Optimized cleanup for CPU and Metal GPU memory
//!
//! # Features
//!
//! - **Automatic Cleanup**: Idle, pressure-based, and periodic cleanup strategies
//! - **Manual Cleanup**: Explicit cleanup APIs for immediate memory reclamation
//! - **Smart Cleanup**: Pattern-based cleanup using allocation analytics
//! - **Device Optimization**: Device-specific cleanup for CPU cache and Metal buffers
//! - **Safety**: Comprehensive validation to prevent corruption of active tensors
//! - **Configurability**: Extensive configuration options for tuning behavior
//! - **Monitoring**: Detailed metrics and performance tracking
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::memory::cleanup::{CleanupManager, CleanupConfig};
//! use bitnet_core::memory::HybridMemoryPool;
//!
//! // Create cleanup manager with default configuration
//! let pool = HybridMemoryPool::new()?;
//! let cleanup_manager = CleanupManager::new(CleanupConfig::default(), pool)?;
//!
//! // Start automatic cleanup scheduler
//! cleanup_manager.start_scheduler()?;
//!
//! // Perform manual cleanup when needed
//! let result = cleanup_manager.force_cleanup()?;
//! println!("Cleaned up {} bytes", result.bytes_freed);
//!
//! // Stop scheduler when done
//! cleanup_manager.stop_scheduler()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use thiserror::Error;

#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

// Sub-modules
pub mod config;
pub mod device_cleanup;
pub mod manager;
pub mod metrics;
pub mod scheduler;
pub mod strategies;

// Re-exports
pub use config::CleanupStrategyType;
pub use config::{CleanupConfig, CleanupFeatureFlags, CleanupPolicy, CleanupThresholds};
pub use device_cleanup::{CpuCleanup, DeviceCleanupOps, MetalCleanup};
pub use manager::{CleanupManager, CleanupOperationResult, CompactionResult};
pub use metrics::{CleanupMetrics, CleanupOperationMetrics, EfficiencyMetrics};
pub use scheduler::{CleanupId, CleanupScheduler, ScheduledCleanup};
pub use strategies::{
    CleanupPriority, CleanupStrategy, DeviceCleanupStrategy, GenerationalCleanupStrategy,
    IdleCleanupStrategy, PeriodicCleanupStrategy, PressureCleanupStrategy,
};

/// Errors that can occur during cleanup operations
#[derive(Error, Debug)]
pub enum CleanupError {
    /// Cleanup operation failed due to system constraints
    #[error("Cleanup operation failed: {reason}")]
    OperationFailed { reason: String },

    /// Invalid cleanup configuration
    #[error("Invalid cleanup configuration: {reason}")]
    InvalidConfiguration { reason: String },

    /// Cleanup scheduler error
    #[error("Cleanup scheduler error: {reason}")]
    SchedulerError { reason: String },

    /// Strategy execution failed
    #[error("Cleanup strategy '{strategy}' failed: {reason}")]
    StrategyFailed { strategy: String, reason: String },

    /// Device-specific cleanup error
    #[error("Device cleanup failed for {device_type}: {reason}")]
    DeviceCleanupFailed { device_type: String, reason: String },

    /// Safety validation failed
    #[error("Cleanup safety validation failed: {reason}")]
    SafetyValidationFailed { reason: String },

    /// Cleanup system not initialized
    #[error("Cleanup system not initialized: {reason}")]
    NotInitialized { reason: String },

    /// Concurrent cleanup operation in progress
    #[error("Concurrent cleanup operation in progress")]
    ConcurrentOperation,

    /// Internal cleanup system error
    #[error("Internal cleanup system error: {reason}")]
    InternalError { reason: String },
}

/// Result type for cleanup operations
pub type CleanupResult<T> = std::result::Result<T, CleanupError>;

/// Unique identifier for cleanup operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CleanupOperationId(pub u64);

impl CleanupOperationId {
    /// Creates a new cleanup operation ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for CleanupOperationId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<CleanupOperationId> for u64 {
    fn from(id: CleanupOperationId) -> u64 {
        id.0
    }
}

/// Information about a cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CleanupOperation {
    /// Unique identifier for this operation
    pub id: CleanupOperationId,
    /// Type of cleanup strategy used
    pub strategy_type: CleanupStrategyType,
    /// Timestamp when operation started
    pub start_time: SystemTime,
    /// Duration of the operation
    pub duration: Option<Duration>,
    /// Number of bytes freed
    pub bytes_freed: u64,
    /// Number of allocations cleaned
    pub allocations_cleaned: u64,
    /// Device where cleanup occurred
    pub device_type: String,
    /// Whether operation completed successfully
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
}

impl CleanupOperation {
    /// Creates a new cleanup operation record
    pub fn new(
        id: CleanupOperationId,
        strategy_type: CleanupStrategyType,
        device_type: String,
    ) -> Self {
        Self {
            id,
            strategy_type,
            start_time: SystemTime::now(),
            duration: None,
            bytes_freed: 0,
            allocations_cleaned: 0,
            device_type,
            success: false,
            error_message: None,
        }
    }

    /// Marks the operation as completed successfully
    pub fn complete_success(
        &mut self,
        bytes_freed: u64,
        allocations_cleaned: u64,
        duration: Duration,
    ) {
        self.bytes_freed = bytes_freed;
        self.allocations_cleaned = allocations_cleaned;
        self.duration = Some(duration);
        self.success = true;
        self.error_message = None;
    }

    /// Marks the operation as failed
    pub fn complete_failure(&mut self, error: String, duration: Duration) {
        self.duration = Some(duration);
        self.success = false;
        self.error_message = Some(error);
    }

    /// Returns the efficiency of this cleanup operation (bytes freed per millisecond)
    pub fn efficiency(&self) -> f64 {
        if let Some(duration) = self.duration {
            let duration_ms = duration.as_millis() as f64;
            if duration_ms > 0.0 {
                self.bytes_freed as f64 / duration_ms
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

/// Global cleanup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct GlobalCleanupStats {
    /// Total cleanup operations performed
    pub total_operations: u64,
    /// Total successful operations
    pub successful_operations: u64,
    /// Total failed operations
    pub failed_operations: u64,
    /// Total bytes freed across all operations
    pub total_bytes_freed: u64,
    /// Total allocations cleaned
    pub total_allocations_cleaned: u64,
    /// Total time spent in cleanup operations
    pub total_cleanup_time: Duration,
    /// Average cleanup efficiency (bytes per millisecond)
    pub average_efficiency: f64,
    /// Statistics per device type
    pub device_stats: HashMap<String, DeviceCleanupStats>,
    /// Statistics per strategy type
    pub strategy_stats: HashMap<CleanupStrategyType, StrategyCleanupStats>,
}

/// Cleanup statistics for a specific device type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeviceCleanupStats {
    /// Device type identifier
    pub device_type: String,
    /// Number of cleanup operations on this device
    pub operations: u64,
    /// Bytes freed on this device
    pub bytes_freed: u64,
    /// Allocations cleaned on this device
    pub allocations_cleaned: u64,
    /// Average efficiency for this device
    pub average_efficiency: f64,
}

/// Cleanup statistics for a specific strategy type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct StrategyCleanupStats {
    /// Strategy type
    pub strategy_type: CleanupStrategyType,
    /// Number of times this strategy was used
    pub operations: u64,
    /// Bytes freed by this strategy
    pub bytes_freed: u64,
    /// Allocations cleaned by this strategy
    pub allocations_cleaned: u64,
    /// Average efficiency for this strategy
    pub average_efficiency: f64,
}

impl GlobalCleanupStats {
    /// Creates new global cleanup statistics
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_bytes_freed: 0,
            total_allocations_cleaned: 0,
            total_cleanup_time: Duration::ZERO,
            average_efficiency: 0.0,
            device_stats: HashMap::new(),
            strategy_stats: HashMap::new(),
        }
    }

    /// Records a cleanup operation
    pub fn record_operation(&mut self, operation: &CleanupOperation) {
        self.total_operations += 1;

        if operation.success {
            self.successful_operations += 1;
            self.total_bytes_freed += operation.bytes_freed;
            self.total_allocations_cleaned += operation.allocations_cleaned;
        } else {
            self.failed_operations += 1;
        }

        if let Some(duration) = operation.duration {
            self.total_cleanup_time += duration;
        }

        // Update device stats
        let device_stats = self
            .device_stats
            .entry(operation.device_type.clone())
            .or_insert_with(|| DeviceCleanupStats {
                device_type: operation.device_type.clone(),
                operations: 0,
                bytes_freed: 0,
                allocations_cleaned: 0,
                average_efficiency: 0.0,
            });

        device_stats.operations += 1;
        if operation.success {
            device_stats.bytes_freed += operation.bytes_freed;
            device_stats.allocations_cleaned += operation.allocations_cleaned;
        }

        // Update strategy stats
        let strategy_stats = self
            .strategy_stats
            .entry(operation.strategy_type)
            .or_insert_with(|| StrategyCleanupStats {
                strategy_type: operation.strategy_type,
                operations: 0,
                bytes_freed: 0,
                allocations_cleaned: 0,
                average_efficiency: 0.0,
            });

        strategy_stats.operations += 1;
        if operation.success {
            strategy_stats.bytes_freed += operation.bytes_freed;
            strategy_stats.allocations_cleaned += operation.allocations_cleaned;
        }

        // Recalculate average efficiency
        self.update_efficiency();
    }

    /// Updates efficiency calculations
    fn update_efficiency(&mut self) {
        let total_time_ms = self.total_cleanup_time.as_millis() as f64;
        if total_time_ms > 0.0 {
            self.average_efficiency = self.total_bytes_freed as f64 / total_time_ms;
        }

        // Update device efficiencies
        for stats in self.device_stats.values_mut() {
            // This is a simplified calculation - in practice you'd track per-device timing
            stats.average_efficiency = if stats.operations > 0 {
                stats.bytes_freed as f64 / stats.operations as f64
            } else {
                0.0
            };
        }

        // Update strategy efficiencies
        for stats in self.strategy_stats.values_mut() {
            stats.average_efficiency = if stats.operations > 0 {
                stats.bytes_freed as f64 / stats.operations as f64
            } else {
                0.0
            };
        }
    }

    /// Returns the success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            1.0
        } else {
            self.successful_operations as f64 / self.total_operations as f64
        }
    }
}

impl Default for GlobalCleanupStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_operation_id() {
        let id = CleanupOperationId::new(42);
        assert_eq!(id.raw(), 42);

        let id_from_u64: CleanupOperationId = 123.into();
        assert_eq!(id_from_u64.raw(), 123);

        let u64_from_id: u64 = id.into();
        assert_eq!(u64_from_id, 42);
    }

    #[test]
    fn test_cleanup_operation() {
        let id = CleanupOperationId::new(1);
        let mut operation = CleanupOperation::new(id, CleanupStrategyType::Idle, "CPU".to_string());

        assert_eq!(operation.id, id);
        assert_eq!(operation.strategy_type, CleanupStrategyType::Idle);
        assert_eq!(operation.device_type, "CPU");
        assert!(!operation.success);
        assert_eq!(operation.bytes_freed, 0);

        // Test successful completion
        operation.complete_success(1024, 5, Duration::from_millis(100));
        assert!(operation.success);
        assert_eq!(operation.bytes_freed, 1024);
        assert_eq!(operation.allocations_cleaned, 5);
        assert_eq!(operation.duration, Some(Duration::from_millis(100)));

        // Test efficiency calculation
        let efficiency = operation.efficiency();
        assert_eq!(efficiency, 10.24); // 1024 bytes / 100 ms
    }

    #[test]
    fn test_global_cleanup_stats() {
        let mut stats = GlobalCleanupStats::new();

        // Create a successful operation
        let id = CleanupOperationId::new(1);
        let mut operation = CleanupOperation::new(id, CleanupStrategyType::Idle, "CPU".to_string());
        operation.complete_success(1024, 5, Duration::from_millis(100));

        // Record the operation
        stats.record_operation(&operation);

        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.failed_operations, 0);
        assert_eq!(stats.total_bytes_freed, 1024);
        assert_eq!(stats.total_allocations_cleaned, 5);
        assert_eq!(stats.success_rate(), 1.0);

        // Check device stats
        assert!(stats.device_stats.contains_key("CPU"));
        let cpu_stats = &stats.device_stats["CPU"];
        assert_eq!(cpu_stats.operations, 1);
        assert_eq!(cpu_stats.bytes_freed, 1024);

        // Check strategy stats
        assert!(stats
            .strategy_stats
            .contains_key(&CleanupStrategyType::Idle));
        let idle_stats = &stats.strategy_stats[&CleanupStrategyType::Idle];
        assert_eq!(idle_stats.operations, 1);
        assert_eq!(idle_stats.bytes_freed, 1024);
    }
}
