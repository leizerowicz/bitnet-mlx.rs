//! Cleanup Metrics and Monitoring System
//!
//! This module provides comprehensive metrics collection and monitoring
//! for cleanup operations, enabling performance analysis and optimization.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "tracing")]
use tracing::{debug, info};

use super::config::CleanupStrategyType;
use super::strategies::CleanupPriority;
use super::{CleanupOperation, CleanupOperationId};

/// Comprehensive cleanup metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupMetrics {
    /// Overall cleanup statistics
    pub overall: OverallCleanupMetrics,
    /// Per-strategy metrics
    pub strategy_metrics: HashMap<CleanupStrategyType, StrategyMetrics>,
    /// Per-device metrics
    pub device_metrics: HashMap<String, DeviceMetrics>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
    /// Recent operation history
    pub recent_operations: Vec<CleanupOperationMetrics>,
    /// Timestamp when metrics were last updated
    pub last_updated: SystemTime,
}

/// Overall cleanup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallCleanupMetrics {
    /// Total number of cleanup operations
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Total bytes freed across all operations
    pub total_bytes_freed: u64,
    /// Total allocations cleaned
    pub total_allocations_cleaned: u64,
    /// Total time spent in cleanup operations
    pub total_cleanup_time: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average bytes freed per operation
    pub average_bytes_per_operation: f64,
    /// Average cleanup duration
    pub average_cleanup_duration: Duration,
    /// Cleanup frequency (operations per hour)
    pub cleanup_frequency: f64,
}

/// Metrics for a specific cleanup strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    /// Strategy type
    pub strategy_type: CleanupStrategyType,
    /// Number of times this strategy was used
    pub usage_count: u64,
    /// Number of successful executions
    pub success_count: u64,
    /// Number of failed executions
    pub failure_count: u64,
    /// Total bytes freed by this strategy
    pub bytes_freed: u64,
    /// Total allocations cleaned by this strategy
    pub allocations_cleaned: u64,
    /// Total execution time for this strategy
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Success rate for this strategy
    pub success_rate: f64,
    /// Efficiency (bytes freed per millisecond)
    pub efficiency: f64,
    /// Last execution time
    pub last_execution: Option<SystemTime>,
}

/// Metrics for a specific device type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetrics {
    /// Device type
    pub device_type: String,
    /// Number of cleanup operations on this device
    pub operations: u64,
    /// Bytes freed on this device
    pub bytes_freed: u64,
    /// Allocations cleaned on this device
    pub allocations_cleaned: u64,
    /// Total cleanup time for this device
    pub total_cleanup_time: Duration,
    /// Average cleanup time for this device
    pub average_cleanup_time: Duration,
    /// Device-specific efficiency
    pub efficiency: f64,
    /// Last cleanup time
    pub last_cleanup: Option<SystemTime>,
}

/// Performance-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Minimum cleanup duration observed
    pub min_cleanup_duration: Duration,
    /// Maximum cleanup duration observed
    pub max_cleanup_duration: Duration,
    /// 95th percentile cleanup duration
    pub p95_cleanup_duration: Duration,
    /// 99th percentile cleanup duration
    pub p99_cleanup_duration: Duration,
    /// Standard deviation of cleanup durations
    pub cleanup_duration_stddev: f64,
    /// Cleanup operations per second (recent average)
    pub operations_per_second: f64,
    /// Bytes freed per second (recent average)
    pub bytes_freed_per_second: f64,
    /// Memory pressure correlation
    pub pressure_correlation: f64,
}

/// Efficiency-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Overall cleanup efficiency (bytes freed per millisecond)
    pub overall_efficiency: f64,
    /// Best efficiency observed
    pub best_efficiency: f64,
    /// Worst efficiency observed
    pub worst_efficiency: f64,
    /// Efficiency trend (positive = improving, negative = degrading)
    pub efficiency_trend: f64,
    /// Cleanup overhead percentage
    pub overhead_percentage: f64,
    /// Resource utilization during cleanup
    pub resource_utilization: f64,
    /// Cleanup impact on allocation performance
    pub allocation_impact: f64,
}

/// Metrics for a single cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupOperationMetrics {
    /// Operation ID
    pub operation_id: CleanupOperationId,
    /// Strategy used
    pub strategy_type: CleanupStrategyType,
    /// Priority level
    pub priority: CleanupPriority,
    /// Device type
    pub device_type: String,
    /// Start time
    pub start_time: SystemTime,
    /// Duration
    pub duration: Duration,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Allocations cleaned
    pub allocations_cleaned: u64,
    /// Whether operation was successful
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Operation efficiency
    pub efficiency: f64,
    /// Memory pressure at start
    pub memory_pressure: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CleanupOperationMetrics {
    /// Creates metrics from a cleanup operation
    pub fn from_operation(operation: &CleanupOperation) -> Self {
        let efficiency = if let Some(duration) = operation.duration {
            let duration_ms = duration.as_millis() as f64;
            if duration_ms > 0.0 {
                operation.bytes_freed as f64 / duration_ms
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            operation_id: operation.id,
            strategy_type: operation.strategy_type,
            priority: CleanupPriority::Normal, // Would be extracted from operation context
            device_type: operation.device_type.clone(),
            start_time: operation.start_time,
            duration: operation.duration.unwrap_or(Duration::ZERO),
            bytes_freed: operation.bytes_freed,
            allocations_cleaned: operation.allocations_cleaned,
            success: operation.success,
            error_message: operation.error_message.clone(),
            efficiency,
            memory_pressure: None, // Would be populated from memory tracker
            metadata: HashMap::new(),
        }
    }
}

/// Cleanup metrics collector and analyzer
pub struct CleanupMetricsCollector {
    /// Current metrics
    metrics: Arc<RwLock<CleanupMetrics>>,
    /// Operation history for analysis
    operation_history: Arc<RwLock<VecDeque<CleanupOperationMetrics>>>,
    /// Duration samples for percentile calculations
    duration_samples: Arc<RwLock<VecDeque<Duration>>>,
    /// Efficiency samples for trend analysis
    efficiency_samples: Arc<RwLock<VecDeque<f64>>>,
    /// Maximum history size
    max_history_size: usize,
    /// Metrics collection start time
    start_time: Instant,
}

impl CleanupMetricsCollector {
    /// Creates a new metrics collector
    pub fn new(max_history_size: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(CleanupMetrics::new())),
            operation_history: Arc::new(RwLock::new(VecDeque::new())),
            duration_samples: Arc::new(RwLock::new(VecDeque::new())),
            efficiency_samples: Arc::new(RwLock::new(VecDeque::new())),
            max_history_size,
            start_time: Instant::now(),
        }
    }

    /// Creates a default metrics collector
    pub fn default() -> Self {
        Self::new(1000)
    }

    /// Records a cleanup operation
    pub fn record_operation(&self, operation: &CleanupOperation) {
        let operation_metrics = CleanupOperationMetrics::from_operation(operation);

        // Add to operation history
        {
            let mut history = self.operation_history.write().unwrap();
            history.push_back(operation_metrics.clone());

            // Limit history size
            while history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        // Add duration sample
        if let Some(duration) = operation.duration {
            let mut duration_samples = self.duration_samples.write().unwrap();
            duration_samples.push_back(duration);

            // Limit sample size
            while duration_samples.len() > self.max_history_size {
                duration_samples.pop_front();
            }
        }

        // Add efficiency sample
        {
            let mut efficiency_samples = self.efficiency_samples.write().unwrap();
            efficiency_samples.push_back(operation_metrics.efficiency);

            // Limit sample size
            while efficiency_samples.len() > self.max_history_size {
                efficiency_samples.pop_front();
            }
        }

        // Update metrics
        self.update_metrics();

        #[cfg(feature = "tracing")]
        debug!(
            "Recorded cleanup operation metrics: {:?}",
            operation_metrics.operation_id
        );
    }

    /// Returns current cleanup metrics
    pub fn get_metrics(&self) -> CleanupMetrics {
        self.metrics
            .read()
            .map(|metrics| metrics.clone())
            .unwrap_or_else(|_| CleanupMetrics::new()) // Fixed closure signature
    }

    /// Returns recent operation history
    pub fn get_recent_operations(&self, count: usize) -> Vec<CleanupOperationMetrics> {
        let history = self.operation_history.read().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Returns performance percentiles
    pub fn get_performance_percentiles(&self) -> (Duration, Duration, Duration) {
        let duration_samples = self.duration_samples.read().unwrap();
        let mut sorted_durations: Vec<Duration> = duration_samples.iter().cloned().collect();
        sorted_durations.sort();

        let len = sorted_durations.len();
        if len == 0 {
            return (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        }

        let p95_index = (len as f64 * 0.95) as usize;
        let p99_index = (len as f64 * 0.99) as usize;
        let max_index = len - 1;

        let p95 = sorted_durations
            .get(p95_index)
            .cloned()
            .unwrap_or(Duration::ZERO);
        let p99 = sorted_durations
            .get(p99_index)
            .cloned()
            .unwrap_or(Duration::ZERO);
        let max = sorted_durations
            .get(max_index)
            .cloned()
            .unwrap_or(Duration::ZERO);

        (p95, p99, max)
    }

    /// Calculates efficiency trend
    pub fn calculate_efficiency_trend(&self) -> f64 {
        let efficiency_samples = self.efficiency_samples.read().unwrap();
        let samples: Vec<f64> = efficiency_samples.iter().cloned().collect();

        if samples.len() < 2 {
            return 0.0;
        }

        // Simple linear regression to calculate trend
        let n = samples.len() as f64;
        let sum_x: f64 = (0..samples.len()).map(|i| i as f64).sum();
        let sum_y: f64 = samples.iter().sum();
        let sum_xy: f64 = samples.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..samples.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        slope
    }

    /// Resets all metrics
    pub fn reset(&self) {
        {
            let mut metrics = self.metrics.write().unwrap();
            *metrics = CleanupMetrics::new();
        }

        {
            let mut history = self.operation_history.write().unwrap();
            history.clear();
        }

        {
            let mut duration_samples = self.duration_samples.write().unwrap();
            duration_samples.clear();
        }

        {
            let mut efficiency_samples = self.efficiency_samples.write().unwrap();
            efficiency_samples.clear();
        }

        #[cfg(feature = "tracing")]
        info!("Reset cleanup metrics");
    }

    // Private helper methods

    /// Updates all metrics based on current operation history
    fn update_metrics(&self) {
        let history = self.operation_history.read().unwrap();
        let operations: Vec<CleanupOperationMetrics> = history.iter().cloned().collect();
        drop(history);

        if operations.is_empty() {
            return;
        }

        let mut new_metrics = CleanupMetrics::new();

        // Calculate overall metrics
        new_metrics.overall = self.calculate_overall_metrics(&operations);

        // Calculate per-strategy metrics
        new_metrics.strategy_metrics = self.calculate_strategy_metrics(&operations);

        // Calculate per-device metrics
        new_metrics.device_metrics = self.calculate_device_metrics(&operations);

        // Calculate performance metrics
        new_metrics.performance = self.calculate_performance_metrics(&operations);

        // Calculate efficiency metrics
        new_metrics.efficiency = self.calculate_efficiency_metrics(&operations);

        // Set recent operations (last 10)
        new_metrics.recent_operations = operations.into_iter().rev().take(10).collect();

        // Update timestamp
        new_metrics.last_updated = SystemTime::now();

        // Store updated metrics
        if let Ok(mut metrics) = self.metrics.write() {
            *metrics = new_metrics;
        }
    }

    /// Calculates overall cleanup metrics
    fn calculate_overall_metrics(
        &self,
        operations: &[CleanupOperationMetrics],
    ) -> OverallCleanupMetrics {
        let total_operations = operations.len() as u64;
        let successful_operations = operations.iter().filter(|op| op.success).count() as u64;
        let failed_operations = total_operations - successful_operations;

        let total_bytes_freed: u64 = operations.iter().map(|op| op.bytes_freed).sum();
        let total_allocations_cleaned: u64 =
            operations.iter().map(|op| op.allocations_cleaned).sum();
        let total_cleanup_time: Duration = operations.iter().map(|op| op.duration).sum();

        let success_rate = if total_operations > 0 {
            successful_operations as f64 / total_operations as f64
        } else {
            1.0
        };

        let average_bytes_per_operation = if total_operations > 0 {
            total_bytes_freed as f64 / total_operations as f64
        } else {
            0.0
        };

        let average_cleanup_duration = if total_operations > 0 {
            total_cleanup_time / total_operations as u32
        } else {
            Duration::ZERO
        };

        let elapsed_hours = self.start_time.elapsed().as_secs_f64() / 3600.0;
        let cleanup_frequency = if elapsed_hours > 0.0 {
            total_operations as f64 / elapsed_hours
        } else {
            0.0
        };

        OverallCleanupMetrics {
            total_operations,
            successful_operations,
            failed_operations,
            total_bytes_freed,
            total_allocations_cleaned,
            total_cleanup_time,
            success_rate,
            average_bytes_per_operation,
            average_cleanup_duration,
            cleanup_frequency,
        }
    }

    /// Calculates per-strategy metrics
    fn calculate_strategy_metrics(
        &self,
        operations: &[CleanupOperationMetrics],
    ) -> HashMap<CleanupStrategyType, StrategyMetrics> {
        let mut strategy_metrics = HashMap::new();

        for strategy_type in [
            CleanupStrategyType::Idle,
            CleanupStrategyType::Pressure,
            CleanupStrategyType::Periodic,
            CleanupStrategyType::Device,
            CleanupStrategyType::Generational,
        ] {
            let strategy_ops: Vec<&CleanupOperationMetrics> = operations
                .iter()
                .filter(|op| op.strategy_type == strategy_type)
                .collect();

            if strategy_ops.is_empty() {
                continue;
            }

            let usage_count = strategy_ops.len() as u64;
            let success_count = strategy_ops.iter().filter(|op| op.success).count() as u64;
            let failure_count = usage_count - success_count;

            let bytes_freed: u64 = strategy_ops.iter().map(|op| op.bytes_freed).sum();
            let allocations_cleaned: u64 =
                strategy_ops.iter().map(|op| op.allocations_cleaned).sum();
            let total_execution_time: Duration = strategy_ops.iter().map(|op| op.duration).sum();

            let average_execution_time = if usage_count > 0 {
                total_execution_time / usage_count as u32
            } else {
                Duration::ZERO
            };

            let success_rate = if usage_count > 0 {
                success_count as f64 / usage_count as f64
            } else {
                1.0
            };

            let efficiency = if total_execution_time.as_millis() > 0 {
                bytes_freed as f64 / total_execution_time.as_millis() as f64
            } else {
                0.0
            };

            let last_execution = strategy_ops.iter().map(|op| op.start_time).max();

            strategy_metrics.insert(
                strategy_type,
                StrategyMetrics {
                    strategy_type,
                    usage_count,
                    success_count,
                    failure_count,
                    bytes_freed,
                    allocations_cleaned,
                    total_execution_time,
                    average_execution_time,
                    success_rate,
                    efficiency,
                    last_execution,
                },
            );
        }

        strategy_metrics
    }

    /// Calculates per-device metrics
    fn calculate_device_metrics(
        &self,
        operations: &[CleanupOperationMetrics],
    ) -> HashMap<String, DeviceMetrics> {
        let mut device_metrics = HashMap::new();

        // Group operations by device type
        let mut device_groups: HashMap<String, Vec<&CleanupOperationMetrics>> = HashMap::new();
        for op in operations {
            device_groups
                .entry(op.device_type.clone())
                .or_default()
                .push(op);
        }

        for (device_type, device_ops) in device_groups {
            let operations_count = device_ops.len() as u64;
            let bytes_freed: u64 = device_ops.iter().map(|op| op.bytes_freed).sum();
            let allocations_cleaned: u64 = device_ops.iter().map(|op| op.allocations_cleaned).sum();
            let total_cleanup_time: Duration = device_ops.iter().map(|op| op.duration).sum();

            let average_cleanup_time = if operations_count > 0 {
                total_cleanup_time / operations_count as u32
            } else {
                Duration::ZERO
            };

            let efficiency = if total_cleanup_time.as_millis() > 0 {
                bytes_freed as f64 / total_cleanup_time.as_millis() as f64
            } else {
                0.0
            };

            let last_cleanup = device_ops.iter().map(|op| op.start_time).max();

            device_metrics.insert(
                device_type.clone(),
                DeviceMetrics {
                    device_type,
                    operations: operations_count,
                    bytes_freed,
                    allocations_cleaned,
                    total_cleanup_time,
                    average_cleanup_time,
                    efficiency,
                    last_cleanup,
                },
            );
        }

        device_metrics
    }

    /// Calculates performance metrics
    fn calculate_performance_metrics(
        &self,
        operations: &[CleanupOperationMetrics],
    ) -> PerformanceMetrics {
        let durations: Vec<Duration> = operations.iter().map(|op| op.duration).collect();

        let min_cleanup_duration = durations.iter().min().cloned().unwrap_or(Duration::ZERO);
        let max_cleanup_duration = durations.iter().max().cloned().unwrap_or(Duration::ZERO);

        let (p95_cleanup_duration, p99_cleanup_duration, _) = self.get_performance_percentiles();

        // Calculate standard deviation
        let mean_duration = if !durations.is_empty() {
            durations.iter().sum::<Duration>() / durations.len() as u32
        } else {
            Duration::ZERO
        };

        let variance = if !durations.is_empty() {
            let mean_ms = mean_duration.as_millis() as f64;
            let sum_squared_diff: f64 = durations
                .iter()
                .map(|d| {
                    let diff = d.as_millis() as f64 - mean_ms;
                    diff * diff
                })
                .sum();
            sum_squared_diff / durations.len() as f64
        } else {
            0.0
        };

        let cleanup_duration_stddev = variance.sqrt();

        // Calculate recent rates (last 60 seconds)
        let recent_cutoff = SystemTime::now() - Duration::from_secs(60);
        let recent_ops: Vec<&CleanupOperationMetrics> = operations
            .iter()
            .filter(|op| op.start_time >= recent_cutoff)
            .collect();

        let operations_per_second = recent_ops.len() as f64 / 60.0;
        let bytes_freed_per_second =
            recent_ops.iter().map(|op| op.bytes_freed).sum::<u64>() as f64 / 60.0;

        PerformanceMetrics {
            min_cleanup_duration,
            max_cleanup_duration,
            p95_cleanup_duration,
            p99_cleanup_duration,
            cleanup_duration_stddev,
            operations_per_second,
            bytes_freed_per_second,
            pressure_correlation: 0.0, // Would be calculated from memory pressure data
        }
    }

    /// Calculates efficiency metrics
    fn calculate_efficiency_metrics(
        &self,
        operations: &[CleanupOperationMetrics],
    ) -> EfficiencyMetrics {
        let efficiencies: Vec<f64> = operations.iter().map(|op| op.efficiency).collect();

        let overall_efficiency = if !efficiencies.is_empty() {
            efficiencies.iter().sum::<f64>() / efficiencies.len() as f64
        } else {
            0.0
        };

        let best_efficiency = efficiencies.iter().cloned().fold(0.0, f64::max);
        let worst_efficiency = efficiencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_efficiency = if worst_efficiency == f64::INFINITY {
            0.0
        } else {
            worst_efficiency
        };

        let efficiency_trend = self.calculate_efficiency_trend();

        EfficiencyMetrics {
            overall_efficiency,
            best_efficiency,
            worst_efficiency,
            efficiency_trend,
            overhead_percentage: 0.0, // Would be calculated from system metrics
            resource_utilization: 0.0, // Would be calculated from system metrics
            allocation_impact: 0.0,   // Would be calculated from allocation performance
        }
    }
}

impl CleanupMetrics {
    /// Creates new cleanup metrics
    pub fn new() -> Self {
        Self {
            overall: OverallCleanupMetrics {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                total_bytes_freed: 0,
                total_allocations_cleaned: 0,
                total_cleanup_time: Duration::ZERO,
                success_rate: 1.0,
                average_bytes_per_operation: 0.0,
                average_cleanup_duration: Duration::ZERO,
                cleanup_frequency: 0.0,
            },
            strategy_metrics: HashMap::new(),
            device_metrics: HashMap::new(),
            performance: PerformanceMetrics {
                min_cleanup_duration: Duration::ZERO,
                max_cleanup_duration: Duration::ZERO,
                p95_cleanup_duration: Duration::ZERO,
                p99_cleanup_duration: Duration::ZERO,
                cleanup_duration_stddev: 0.0,
                operations_per_second: 0.0,
                bytes_freed_per_second: 0.0,
                pressure_correlation: 0.0,
            },
            efficiency: EfficiencyMetrics {
                overall_efficiency: 0.0,
                best_efficiency: 0.0,
                worst_efficiency: 0.0,
                efficiency_trend: 0.0,
                overhead_percentage: 0.0,
                resource_utilization: 0.0,
                allocation_impact: 0.0,
            },
            recent_operations: Vec::new(),
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for CleanupMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CleanupOperation, CleanupOperationId};
    use super::*;

    #[test]
    fn test_cleanup_metrics_creation() {
        let metrics = CleanupMetrics::new();
        assert_eq!(metrics.overall.total_operations, 0);
        assert_eq!(metrics.overall.success_rate, 1.0);
        assert!(metrics.strategy_metrics.is_empty());
        assert!(metrics.device_metrics.is_empty());
    }

    #[test]
    fn test_cleanup_operation_metrics() {
        let operation = CleanupOperation::new(
            CleanupOperationId::new(1),
            CleanupStrategyType::Idle,
            "CPU".to_string(),
        );

        let metrics = CleanupOperationMetrics::from_operation(&operation);
        assert_eq!(metrics.operation_id, operation.id);
        assert_eq!(metrics.strategy_type, CleanupStrategyType::Idle);
        assert_eq!(metrics.device_type, "CPU");
        assert!(!metrics.success); // Operation hasn't completed yet
    }

    #[test]
    fn test_metrics_collector() {
        let collector = CleanupMetricsCollector::new(100);

        let mut operation = CleanupOperation::new(
            CleanupOperationId::new(1),
            CleanupStrategyType::Idle,
            "CPU".to_string(),
        );
        operation.complete_success(1024, 5, Duration::from_millis(100));

        collector.record_operation(&operation);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.overall.total_operations, 1);
        assert_eq!(metrics.overall.successful_operations, 1);
        assert_eq!(metrics.overall.total_bytes_freed, 1024);

        let recent_ops = collector.get_recent_operations(10);
        assert_eq!(recent_ops.len(), 1);
        assert_eq!(recent_ops[0].bytes_freed, 1024);
    }

    #[test]
    fn test_efficiency_calculation() {
        let collector = CleanupMetricsCollector::new(100);

        // Record multiple operations with different efficiencies
        for i in 1..=5 {
            let mut operation = CleanupOperation::new(
                CleanupOperationId::new(i),
                CleanupStrategyType::Idle,
                "CPU".to_string(),
            );
            operation.complete_success(i * 1024, i * 2, Duration::from_millis(i * 50));
            collector.record_operation(&operation);
        }

        let metrics = collector.get_metrics();
        assert_eq!(metrics.overall.total_operations, 5);
        assert!(metrics.efficiency.overall_efficiency > 0.0);
        assert!(metrics.efficiency.best_efficiency >= metrics.efficiency.worst_efficiency);
    }

    #[test]
    fn test_strategy_metrics() {
        let collector = CleanupMetricsCollector::new(100);

        // Record operations with different strategies
        let strategies = [
            CleanupStrategyType::Idle,
            CleanupStrategyType::Pressure,
            CleanupStrategyType::Periodic,
        ];

        for (i, strategy) in strategies.iter().enumerate() {
            let mut operation = CleanupOperation::new(
                CleanupOperationId::new(i as u64 + 1),
                *strategy,
                "CPU".to_string(),
            );
            operation.complete_success(1024, 5, Duration::from_millis(100));
            collector.record_operation(&operation);
        }

        let metrics = collector.get_metrics();
        assert_eq!(metrics.strategy_metrics.len(), 3);

        for strategy in strategies {
            assert!(metrics.strategy_metrics.contains_key(&strategy));
            let strategy_metrics = &metrics.strategy_metrics[&strategy];
            assert_eq!(strategy_metrics.usage_count, 1);
            assert_eq!(strategy_metrics.success_count, 1);
            assert_eq!(strategy_metrics.bytes_freed, 1024);
        }
    }

    #[test]
    fn test_device_metrics() {
        let collector = CleanupMetricsCollector::new(100);

        // Record operations on different devices
        let devices = ["CPU", "Metal"];

        for (i, device) in devices.iter().enumerate() {
            let mut operation = CleanupOperation::new(
                CleanupOperationId::new(i as u64 + 1),
                CleanupStrategyType::Idle,
                device.to_string(),
            );
            operation.complete_success(1024, 5, Duration::from_millis(100));
            collector.record_operation(&operation);
        }

        let metrics = collector.get_metrics();
        assert_eq!(metrics.device_metrics.len(), 2);

        for device in devices {
            assert!(metrics.device_metrics.contains_key(device));
            let device_metrics = &metrics.device_metrics[device];
            assert_eq!(device_metrics.operations, 1);
            assert_eq!(device_metrics.bytes_freed, 1024);
        }
    }
}
