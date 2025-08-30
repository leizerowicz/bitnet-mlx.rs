//! Optimization utilities for inference performance.

use crate::{Result, InferenceError};
use bitnet_core::Device;
use std::time::{Duration, Instant};

/// Performance optimization strategies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// No optimization
    None,
    /// Memory-focused optimizations
    Memory,
    /// Speed-focused optimizations
    Speed,
    /// Balanced memory and speed optimizations
    Balanced,
}

/// Performance optimizer for inference operations.
pub struct PerformanceOptimizer {
    strategy: OptimizationStrategy,
    device: Device,
    measurement_history: Vec<PerformanceMeasurement>,
}

/// A single performance measurement.
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub operation_type: String,
    pub duration: Duration,
    pub memory_usage: usize,
    pub batch_size: usize,
    pub timestamp: Instant,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer.
    pub fn new(strategy: OptimizationStrategy, device: Device) -> Self {
        Self {
            strategy,
            device,
            measurement_history: Vec::new(),
        }
    }

    /// Record a performance measurement.
    pub fn record_measurement(&mut self, measurement: PerformanceMeasurement) {
        self.measurement_history.push(measurement);
        
        // Keep only the last 1000 measurements to prevent unbounded growth
        if self.measurement_history.len() > 1000 {
            self.measurement_history.remove(0);
        }
    }

    /// Get optimal batch size based on historical measurements.
    pub fn optimal_batch_size(&self, operation_type: &str) -> usize {
        let relevant_measurements: Vec<_> = self.measurement_history
            .iter()
            .filter(|m| m.operation_type == operation_type)
            .collect();

        if relevant_measurements.is_empty() {
            return self.default_batch_size();
        }

        // Find the batch size with the best throughput (ops/second)
        let mut best_batch_size = self.default_batch_size();
        let mut best_throughput = 0.0;

        for measurement in relevant_measurements {
            let throughput = measurement.batch_size as f64 / measurement.duration.as_secs_f64();
            if throughput > best_throughput {
                best_throughput = throughput;
                best_batch_size = measurement.batch_size;
            }
        }

        best_batch_size
    }

    /// Get performance statistics for an operation type.
    pub fn performance_stats(&self, operation_type: &str) -> Option<PerformanceStats> {
        let measurements: Vec<_> = self.measurement_history
            .iter()
            .filter(|m| m.operation_type == operation_type)
            .collect();

        if measurements.is_empty() {
            return None;
        }

        let total_duration: Duration = measurements.iter().map(|m| m.duration).sum();
        let avg_duration = total_duration / measurements.len() as u32;
        
        let avg_memory: f64 = measurements.iter().map(|m| m.memory_usage as f64).sum::<f64>() 
            / measurements.len() as f64;

        let total_ops: usize = measurements.iter().map(|m| m.batch_size).sum();
        let throughput = total_ops as f64 / total_duration.as_secs_f64();

        Some(PerformanceStats {
            operation_type: operation_type.to_string(),
            total_measurements: measurements.len(),
            average_duration: avg_duration,
            average_memory_usage: avg_memory as usize,
            throughput_ops_per_sec: throughput,
            total_operations: total_ops,
        })
    }

    /// Apply optimizations based on the current strategy.
    pub fn apply_optimizations(&self) -> OptimizationConfig {
        match self.strategy {
            OptimizationStrategy::None => OptimizationConfig::none(),
            OptimizationStrategy::Memory => self.memory_optimized_config(),
            OptimizationStrategy::Speed => self.speed_optimized_config(),
            OptimizationStrategy::Balanced => self.balanced_config(),
        }
    }

    /// Clear measurement history.
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
    }

    /// Get the default batch size for the current device.
    fn default_batch_size(&self) -> usize {
        match self.device {
            Device::Cpu => 16,
            Device::Metal(_) => 32,
            Device::Cuda(_) => 32, // Same as Metal for now
            // Device::MLX => 64, // TODO: Add when MLX support is implemented
        }
    }

    /// Create memory-optimized configuration.
    fn memory_optimized_config(&self) -> OptimizationConfig {
        OptimizationConfig {
            batch_size: 8, // Smaller batches for less memory usage
            memory_pool_size: 256 * 1024 * 1024, // 256MB
            enable_memory_mapping: true,
            enable_gradient_checkpointing: true,
            enable_mixed_precision: false, // Avoid mixed precision to save memory complexity
        }
    }

    /// Create speed-optimized configuration.
    fn speed_optimized_config(&self) -> OptimizationConfig {
        OptimizationConfig {
            batch_size: match self.device {
                Device::Cpu => 32,
                Device::Metal(_) => 64,
                Device::Cuda(_) => 64,
                // Device::MLX => 128, // TODO: Add when MLX support is implemented
            },
            memory_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            enable_memory_mapping: false, // Direct memory access is faster
            enable_gradient_checkpointing: false,
            enable_mixed_precision: true,
        }
    }

    /// Create balanced configuration.
    fn balanced_config(&self) -> OptimizationConfig {
        OptimizationConfig {
            batch_size: match self.device {
                Device::Cpu => 24,
                Device::Metal(_) => 48,
                Device::Cuda(_) => 48,
                // Device::MLX => 96, // TODO: Add when MLX support is implemented
            },
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_memory_mapping: true,
            enable_gradient_checkpointing: true,
            enable_mixed_precision: true,
        }
    }
}

/// Configuration for performance optimizations.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub batch_size: usize,
    pub memory_pool_size: usize,
    pub enable_memory_mapping: bool,
    pub enable_gradient_checkpointing: bool,
    pub enable_mixed_precision: bool,
}

impl OptimizationConfig {
    /// Create a configuration with no optimizations.
    pub fn none() -> Self {
        Self {
            batch_size: 1,
            memory_pool_size: 64 * 1024 * 1024, // 64MB
            enable_memory_mapping: false,
            enable_gradient_checkpointing: false,
            enable_mixed_precision: false,
        }
    }
}

/// Performance statistics for an operation type.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub operation_type: String,
    pub total_measurements: usize,
    pub average_duration: Duration,
    pub average_memory_usage: usize,
    pub throughput_ops_per_sec: f64,
    pub total_operations: usize,
}

impl PerformanceStats {
    /// Check if performance meets acceptable thresholds.
    pub fn is_acceptable(&self) -> bool {
        // Consider performance acceptable if:
        // - Throughput is reasonable (>1000 ops/sec for most operations)
        // - Average duration is not too high (<100ms)
        self.throughput_ops_per_sec > 1000.0 && 
        self.average_duration < Duration::from_millis(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let optimizer = PerformanceOptimizer::new(OptimizationStrategy::Balanced, Device::Cpu);
        assert_eq!(optimizer.strategy, OptimizationStrategy::Balanced);
        // Device doesn't implement PartialEq, so we can't compare it directly
    }

    #[test]
    fn test_optimization_configs() {
        let optimizer = PerformanceOptimizer::new(OptimizationStrategy::Speed, Device::Cpu);
        let config = optimizer.apply_optimizations();
        
        // Speed optimization should prefer larger batch sizes
        assert!(config.batch_size > 16);
        assert!(config.memory_pool_size > 1024 * 1024 * 1024); // > 1GB
    }

    #[test]
    fn test_performance_measurement() {
        let mut optimizer = PerformanceOptimizer::new(OptimizationStrategy::None, Device::Cpu);
        
        let measurement = PerformanceMeasurement {
            operation_type: "test_op".to_string(),
            duration: Duration::from_millis(50),
            memory_usage: 1024 * 1024, // 1MB
            batch_size: 32,
            timestamp: Instant::now(),
        };
        
        optimizer.record_measurement(measurement);
        
        let stats = optimizer.performance_stats("test_op").unwrap();
        assert_eq!(stats.total_measurements, 1);
        assert_eq!(stats.average_duration, Duration::from_millis(50));
    }

    #[test]
    fn test_optimal_batch_size_calculation() {
        let mut optimizer = PerformanceOptimizer::new(OptimizationStrategy::None, Device::Cpu);
        
        // Record some measurements with different batch sizes
        optimizer.record_measurement(PerformanceMeasurement {
            operation_type: "inference".to_string(),
            duration: Duration::from_millis(100), // 16 ops in 100ms = 160 ops/sec
            memory_usage: 1024,
            batch_size: 16,
            timestamp: Instant::now(),
        });
        
        optimizer.record_measurement(PerformanceMeasurement {
            operation_type: "inference".to_string(),
            duration: Duration::from_millis(150), // 32 ops in 150ms = 213 ops/sec (better)
            memory_usage: 2048,
            batch_size: 32,
            timestamp: Instant::now(),
        });
        
        let optimal = optimizer.optimal_batch_size("inference");
        assert_eq!(optimal, 32); // Should pick the more efficient batch size
    }
}
