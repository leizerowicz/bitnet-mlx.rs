//! MLX Performance Metrics Collection System
//! 
//! This module provides a comprehensive metrics collection system for MLX operations,
//! integrating performance benchmarking, memory tracking, and system monitoring.

use crate::mlx::{
    MlxTensor, BitNetMlxDevice, 
    performance::{PerformanceMetrics, MemoryUsage, BenchmarkConfig},
    memory_tracker::{MlxMemoryTracker, MemoryEvent, MemorySnapshot, MemoryPressure},
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};

/// Comprehensive metrics collection for MLX operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxMetrics {
    pub performance: PerformanceMetrics,
    pub memory: MemoryMetrics,
    pub system: SystemMetrics,
    pub operation_context: OperationContext,
}

/// Memory-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub current_usage: MemoryUsage,
    pub pressure_level: String, // Serialized MemoryPressure
    pub allocation_events: usize,
    pub deallocation_events: usize,
    pub transfer_events: usize,
    pub fragmentation_ratio: f64,
    pub efficiency_score: f64,
}

/// System-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub gpu_usage: f64,
    pub system_memory_usage: f64,
    pub gpu_memory_usage: f64,
    pub temperature: Option<f64>,
    pub power_consumption: Option<f64>,
    pub thermal_state: String,
}

/// Operation context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationContext {
    pub operation_name: String,
    pub batch_size: usize,
    pub sequence_length: Option<usize>,
    pub model_parameters: Option<usize>,
    pub precision: String,
    pub optimization_level: String,
    pub parallel_execution: bool,
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub collect_performance: bool,
    pub collect_memory: bool,
    pub collect_system: bool,
    pub collection_interval: Duration,
    pub max_history_size: usize,
    pub enable_detailed_profiling: bool,
    pub export_format: ExportFormat,
}

/// Export formats for metrics
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
    Custom(String),
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_performance: true,
            collect_memory: true,
            collect_system: true,
            collection_interval: Duration::from_millis(100),
            max_history_size: 10000,
            enable_detailed_profiling: false,
            export_format: ExportFormat::Json,
        }
    }
}

/// MLX Metrics Collector
pub struct MlxMetricsCollector {
    config: MetricsConfig,
    metrics_history: Arc<Mutex<Vec<MlxMetrics>>>,
    memory_tracker: Arc<Mutex<MlxMemoryTracker>>,
    collection_active: Arc<Mutex<bool>>,
    last_collection: Instant,
    aggregated_stats: Arc<Mutex<AggregatedStats>>,
}

/// Aggregated statistics over time
#[derive(Debug, Clone, Default)]
pub struct AggregatedStats {
    total_operations: usize,
    total_execution_time: Duration,
    average_throughput: f64,
    peak_memory_usage: usize,
    total_memory_allocated: usize,
    operation_counts: HashMap<String, usize>,
    device_usage: HashMap<String, Duration>,
}

impl AggregatedStats {
    /// Get total number of operations
    pub fn total_operations(&self) -> usize {
        self.total_operations
    }

    /// Get average throughput
    pub fn average_throughput(&self) -> f64 {
        self.average_throughput
    }

    /// Get peak memory usage in bytes
    pub fn peak_memory_usage(&self) -> usize {
        self.peak_memory_usage
    }

    /// Get total execution time
    pub fn total_execution_time(&self) -> Duration {
        self.total_execution_time
    }

    /// Get total memory allocated in bytes
    pub fn total_memory_allocated(&self) -> usize {
        self.total_memory_allocated
    }

    /// Get operation counts by name
    pub fn operation_counts(&self) -> &HashMap<String, usize> {
        &self.operation_counts
    }

    /// Get device usage by type
    pub fn device_usage(&self) -> &HashMap<String, Duration> {
        &self.device_usage
    }
}

impl MlxMetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            memory_tracker: Arc::new(Mutex::new(MlxMemoryTracker::new())),
            collection_active: Arc::new(Mutex::new(false)),
            last_collection: Instant::now(),
            aggregated_stats: Arc::new(Mutex::new(AggregatedStats::default())),
        }
    }

    /// Start metrics collection
    pub fn start_collection(&self) {
        let mut active = self.collection_active.lock().unwrap();
        *active = true;
    }

    /// Stop metrics collection
    pub fn stop_collection(&self) {
        let mut active = self.collection_active.lock().unwrap();
        *active = false;
    }

    /// Collect metrics for a specific operation
    pub fn collect_operation_metrics(
        &self,
        operation_name: &str,
        device: &BitNetMlxDevice,
        execution_time: Duration,
        tensor_shapes: Vec<Vec<usize>>,
        context: OperationContext,
    ) -> Result<MlxMetrics> {
        let active = self.collection_active.lock().unwrap();
        if !*active {
            return Err(anyhow::anyhow!("Metrics collection is not active"));
        }

        // Collect performance metrics
        let performance = if self.config.collect_performance {
            self.collect_performance_metrics(operation_name, device, execution_time, &tensor_shapes)?
        } else {
            self.create_empty_performance_metrics(operation_name, device)
        };

        // Collect memory metrics
        let memory = if self.config.collect_memory {
            self.collect_memory_metrics(device)?
        } else {
            self.create_empty_memory_metrics()
        };

        // Collect system metrics
        let system = if self.config.collect_system {
            self.collect_system_metrics(device)?
        } else {
            self.create_empty_system_metrics()
        };

        let metrics = MlxMetrics {
            performance,
            memory,
            system,
            operation_context: context,
        };

        // Store metrics in history
        self.store_metrics(&metrics)?;

        // Update aggregated statistics
        self.update_aggregated_stats(&metrics)?;

        Ok(metrics)
    }

    /// Collect performance metrics
    fn collect_performance_metrics(
        &self,
        operation_name: &str,
        device: &BitNetMlxDevice,
        execution_time: Duration,
        tensor_shapes: &[Vec<usize>],
    ) -> Result<PerformanceMetrics> {
        let throughput = 1.0 / execution_time.as_secs_f64();
        
        // Get memory usage from tracker
        let memory_usage = {
            let tracker = self.memory_tracker.lock().unwrap();
            if let Some(stats) = tracker.get_memory_stats(device) {
                MemoryUsage {
                    peak_memory_mb: stats.peak_allocated() as f64 / (1024.0 * 1024.0),
                    allocated_memory_mb: stats.total_allocated() as f64 / (1024.0 * 1024.0),
                    freed_memory_mb: 0.0, // Calculate from deallocation events
                    memory_efficiency: if stats.allocation_count() > 0 {
                        stats.deallocation_count() as f64 / stats.allocation_count() as f64
                    } else {
                        0.0
                    },
                }
            } else {
                MemoryUsage {
                    peak_memory_mb: 0.0,
                    allocated_memory_mb: 0.0,
                    freed_memory_mb: 0.0,
                    memory_efficiency: 0.0,
                }
            }
        };

        Ok(PerformanceMetrics {
            operation_name: operation_name.to_string(),
            device_type: device.device_type().to_string(),
            execution_time,
            memory_usage,
            throughput,
            tensor_shapes: tensor_shapes.to_vec(),
            data_type: "f32".to_string(), // Default, could be parameterized
            timestamp: SystemTime::now(),
        })
    }

    /// Collect memory metrics
    fn collect_memory_metrics(&self, device: &BitNetMlxDevice) -> Result<MemoryMetrics> {
        let tracker = self.memory_tracker.lock().unwrap();
        
        let pressure = tracker.get_memory_pressure(device);
        let pressure_str = match pressure {
            MemoryPressure::Low => "Low",
            MemoryPressure::Medium => "Medium",
            MemoryPressure::High => "High",
            MemoryPressure::Critical => "Critical",
        }.to_string();

        let events = tracker.get_events();
        let device_events: Vec<_> = events.iter()
            .filter(|e| e.device_type == device.device_type())
            .collect();

        let allocation_events = device_events.iter()
            .filter(|e| matches!(e.event_type, crate::mlx::memory_tracker::MemoryEventType::Allocation))
            .count();
        
        let deallocation_events = device_events.iter()
            .filter(|e| matches!(e.event_type, crate::mlx::memory_tracker::MemoryEventType::Deallocation))
            .count();
        
        let transfer_events = device_events.iter()
            .filter(|e| matches!(e.event_type, crate::mlx::memory_tracker::MemoryEventType::Transfer))
            .count();

        let efficiency_score = if allocation_events > 0 {
            deallocation_events as f64 / allocation_events as f64
        } else {
            1.0
        };

        let current_usage = if let Some(stats) = tracker.get_memory_stats(device) {
            MemoryUsage {
                peak_memory_mb: stats.peak_allocated() as f64 / (1024.0 * 1024.0),
                allocated_memory_mb: stats.total_allocated() as f64 / (1024.0 * 1024.0),
                freed_memory_mb: 0.0,
                memory_efficiency: efficiency_score,
            }
        } else {
            MemoryUsage {
                peak_memory_mb: 0.0,
                allocated_memory_mb: 0.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.0,
            }
        };

        Ok(MemoryMetrics {
            current_usage,
            pressure_level: pressure_str,
            allocation_events,
            deallocation_events,
            transfer_events,
            fragmentation_ratio: 0.1, // Placeholder
            efficiency_score,
        })
    }

    /// Collect system metrics
    fn collect_system_metrics(&self, device: &BitNetMlxDevice) -> Result<SystemMetrics> {
        // In a real implementation, this would query actual system metrics
        // For now, we'll provide placeholder values
        
        let (cpu_usage, gpu_usage) = self.get_system_usage()?;
        let (system_memory, gpu_memory) = self.get_memory_usage()?;
        
        Ok(SystemMetrics {
            cpu_usage,
            gpu_usage,
            system_memory_usage: system_memory,
            gpu_memory_usage: gpu_memory,
            temperature: self.get_temperature(device),
            power_consumption: self.get_power_consumption(device),
            thermal_state: self.get_thermal_state(device),
        })
    }

    /// Store metrics in history
    fn store_metrics(&self, metrics: &MlxMetrics) -> Result<()> {
        let mut history = self.metrics_history.lock().unwrap();
        
        // Add new metrics
        history.push(metrics.clone());
        
        // Trim history if it exceeds max size
        if history.len() > self.config.max_history_size {
            let excess = history.len() - self.config.max_history_size;
            history.drain(0..excess);
        }
        
        Ok(())
    }

    /// Update aggregated statistics
    fn update_aggregated_stats(&self, metrics: &MlxMetrics) -> Result<()> {
        let mut stats = self.aggregated_stats.lock().unwrap();
        
        stats.total_operations += 1;
        stats.total_execution_time += metrics.performance.execution_time;
        stats.average_throughput = (stats.average_throughput * (stats.total_operations - 1) as f64 + metrics.performance.throughput) / stats.total_operations as f64;
        
        let current_memory = (metrics.memory.current_usage.peak_memory_mb * 1024.0 * 1024.0) as usize;
        stats.peak_memory_usage = stats.peak_memory_usage.max(current_memory);
        stats.total_memory_allocated += (metrics.memory.current_usage.allocated_memory_mb * 1024.0 * 1024.0) as usize;
        
        *stats.operation_counts.entry(metrics.operation_context.operation_name.clone()).or_insert(0) += 1;
        *stats.device_usage.entry(metrics.performance.device_type.clone()).or_insert(Duration::ZERO) += metrics.performance.execution_time;
        
        Ok(())
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<MlxMetrics> {
        let history = self.metrics_history.lock().unwrap();
        history.clone()
    }

    /// Get aggregated statistics
    pub fn get_aggregated_stats(&self) -> AggregatedStats {
        let stats = self.aggregated_stats.lock().unwrap();
        stats.clone()
    }

    /// Export metrics to specified format
    pub fn export_metrics(&self, format: Option<ExportFormat>) -> Result<String> {
        let export_format = format.unwrap_or_else(|| self.config.export_format.clone());
        let history = self.get_metrics_history();
        
        match export_format {
            ExportFormat::Json => self.export_json(&history),
            ExportFormat::Csv => self.export_csv(&history),
            ExportFormat::Prometheus => self.export_prometheus(&history),
            ExportFormat::Custom(format_name) => {
                Err(anyhow::anyhow!("Custom format '{}' not implemented", format_name))
            }
        }
    }

    /// Export metrics as JSON
    fn export_json(&self, metrics: &[MlxMetrics]) -> Result<String> {
        serde_json::to_string_pretty(metrics)
            .map_err(|e| anyhow::anyhow!("Failed to serialize metrics to JSON: {}", e))
    }

    /// Export metrics as CSV
    fn export_csv(&self, metrics: &[MlxMetrics]) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("timestamp,operation,device,execution_time_ms,throughput,memory_mb,cpu_usage,gpu_usage\n");
        
        for metric in metrics {
            csv.push_str(&format!(
                "{:?},{},{},{:.3},{:.2},{:.2},{:.2},{:.2}\n",
                metric.performance.timestamp,
                metric.operation_context.operation_name,
                metric.performance.device_type,
                metric.performance.execution_time.as_millis(),
                metric.performance.throughput,
                metric.memory.current_usage.allocated_memory_mb,
                metric.system.cpu_usage,
                metric.system.gpu_usage
            ));
        }
        
        Ok(csv)
    }

    /// Export metrics in Prometheus format
    fn export_prometheus(&self, metrics: &[MlxMetrics]) -> Result<String> {
        let mut prometheus = String::new();
        
        // Add metric definitions
        prometheus.push_str("# HELP mlx_execution_time_seconds Execution time of MLX operations\n");
        prometheus.push_str("# TYPE mlx_execution_time_seconds gauge\n");
        
        prometheus.push_str("# HELP mlx_memory_usage_bytes Memory usage of MLX operations\n");
        prometheus.push_str("# TYPE mlx_memory_usage_bytes gauge\n");
        
        prometheus.push_str("# HELP mlx_throughput_ops_per_second Throughput of MLX operations\n");
        prometheus.push_str("# TYPE mlx_throughput_ops_per_second gauge\n");
        
        // Add metrics data
        for metric in metrics {
            let timestamp = metric.performance.timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            
            prometheus.push_str(&format!(
                "mlx_execution_time_seconds{{operation=\"{}\",device=\"{}\"}} {:.6} {}\n",
                metric.operation_context.operation_name,
                metric.performance.device_type,
                metric.performance.execution_time.as_secs_f64(),
                timestamp
            ));
            
            prometheus.push_str(&format!(
                "mlx_memory_usage_bytes{{operation=\"{}\",device=\"{}\"}} {:.0} {}\n",
                metric.operation_context.operation_name,
                metric.performance.device_type,
                metric.memory.current_usage.allocated_memory_mb * 1024.0 * 1024.0,
                timestamp
            ));
            
            prometheus.push_str(&format!(
                "mlx_throughput_ops_per_second{{operation=\"{}\",device=\"{}\"}} {:.2} {}\n",
                metric.operation_context.operation_name,
                metric.performance.device_type,
                metric.performance.throughput,
                timestamp
            ));
        }
        
        Ok(prometheus)
    }

    /// Clear all collected metrics
    pub fn clear_metrics(&self) {
        {
            let mut history = self.metrics_history.lock().unwrap();
            history.clear();
        }
        {
            let mut stats = self.aggregated_stats.lock().unwrap();
            *stats = AggregatedStats::default();
        }
        {
            let tracker = self.memory_tracker.lock().unwrap();
            tracker.clear();
        }
    }

    /// Helper methods for system metrics (placeholder implementations)
    fn get_system_usage(&self) -> Result<(f64, f64)> {
        // Placeholder implementation - in practice would query actual system metrics
        Ok((25.0, 15.0)) // 25% CPU, 15% GPU
    }

    fn get_memory_usage(&self) -> Result<(f64, f64)> {
        // Placeholder implementation
        Ok((60.0, 40.0)) // 60% system memory, 40% GPU memory
    }

    fn get_temperature(&self, device: &BitNetMlxDevice) -> Option<f64> {
        // Placeholder implementation
        match device.device_type() {
            "gpu" => Some(65.0), // 65°C
            _ => None,
        }
    }

    fn get_power_consumption(&self, device: &BitNetMlxDevice) -> Option<f64> {
        // Placeholder implementation
        match device.device_type() {
            "gpu" => Some(25.0), // 25W
            _ => None,
        }
    }

    fn get_thermal_state(&self, device: &BitNetMlxDevice) -> String {
        match device.device_type() {
            "gpu" => "Normal".to_string(),
            _ => "N/A".to_string(),
        }
    }

    /// Create empty performance metrics when collection is disabled
    fn create_empty_performance_metrics(&self, operation_name: &str, device: &BitNetMlxDevice) -> PerformanceMetrics {
        PerformanceMetrics {
            operation_name: operation_name.to_string(),
            device_type: device.device_type().to_string(),
            execution_time: Duration::ZERO,
            memory_usage: MemoryUsage {
                peak_memory_mb: 0.0,
                allocated_memory_mb: 0.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.0,
            },
            throughput: 0.0,
            tensor_shapes: Vec::new(),
            data_type: "unknown".to_string(),
            timestamp: SystemTime::now(),
        }
    }

    /// Create empty memory metrics when collection is disabled
    fn create_empty_memory_metrics(&self) -> MemoryMetrics {
        MemoryMetrics {
            current_usage: MemoryUsage {
                peak_memory_mb: 0.0,
                allocated_memory_mb: 0.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.0,
            },
            pressure_level: "Unknown".to_string(),
            allocation_events: 0,
            deallocation_events: 0,
            transfer_events: 0,
            fragmentation_ratio: 0.0,
            efficiency_score: 0.0,
        }
    }

    /// Create empty system metrics when collection is disabled
    fn create_empty_system_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            cpu_usage: 0.0,
            gpu_usage: 0.0,
            system_memory_usage: 0.0,
            gpu_memory_usage: 0.0,
            temperature: None,
            power_consumption: None,
            thermal_state: "Unknown".to_string(),
        }
    }
}

impl Default for MlxMetricsCollector {
    fn default() -> Self {
        Self::new(MetricsConfig::default())
    }
}

/// Global metrics collector instance
static GLOBAL_COLLECTOR: std::sync::OnceLock<Arc<Mutex<MlxMetricsCollector>>> = std::sync::OnceLock::new();

/// Get the global metrics collector
pub fn get_global_metrics_collector() -> Arc<Mutex<MlxMetricsCollector>> {
    GLOBAL_COLLECTOR.get_or_init(|| {
        Arc::new(Mutex::new(MlxMetricsCollector::default()))
    }).clone()
}

/// Convenience function to collect metrics globally
pub fn collect_operation_metrics(
    operation_name: &str,
    device: &BitNetMlxDevice,
    execution_time: Duration,
    tensor_shapes: Vec<Vec<usize>>,
    context: OperationContext,
) -> Result<MlxMetrics> {
    let collector = get_global_metrics_collector();
    let collector = collector.lock().unwrap();
    collector.collect_operation_metrics(operation_name, device, execution_time, tensor_shapes, context)
}