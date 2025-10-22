// bitnet-training/src/ane/performance_monitor.rs
//! ANE Performance Monitoring
//!
//! This module provides comprehensive performance monitoring for ANE operations,
//! comparing ANE vs GPU performance and generating detailed performance reports.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use super::device::ExtendedDevice;

/// ANE performance monitor for tracking operation performance
pub struct ANEPerformanceMonitor {
    /// Operation history for analysis
    operation_history: VecDeque<OperationRecord>,
    /// Device-specific performance metrics
    device_metrics: HashMap<ExtendedDevice, DeviceMetrics>,
    /// Performance configuration
    config: MonitorConfig,
    /// Start time for session tracking
    session_start: Instant,
}

/// Individual operation record
#[derive(Debug, Clone)]
struct OperationRecord {
    operation: ANEOperation,
    device: ExtendedDevice,
    execution_time: Duration,
    timestamp: Instant,
    success: bool,
    error_message: Option<String>,
}

/// Device-specific performance metrics
#[derive(Debug, Clone)]
struct DeviceMetrics {
    total_operations: usize,
    total_time: Duration,
    successful_operations: usize,
    failed_operations: usize,
    avg_latency: Duration,
    min_latency: Duration,
    max_latency: Duration,
    last_updated: Instant,
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum history size
    max_history_size: usize,
    /// Enable detailed logging
    detailed_logging: bool,
    /// Performance alert thresholds
    alert_thresholds: AlertThresholds,
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable latency (ms)
    max_latency_ms: f32,
    /// Minimum acceptable success rate (%)
    min_success_rate: f32,
    /// Maximum performance degradation vs baseline (%)
    max_degradation_percent: f32,
}

/// ANE operation types for performance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ANEOperation {
    MatMul,
    Attention,
    LayerNorm,
    Embedding,
    Activation,
    Softmax,
    FullForward,
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ANEPerformanceReport {
    /// Average ANE latency in milliseconds
    pub avg_ane_latency_ms: f32,
    /// Average GPU latency in milliseconds
    pub avg_gpu_latency_ms: f32,
    /// ANE success rate (0.0 to 1.0)
    pub ane_success_rate: f32,
    /// GPU success rate (0.0 to 1.0)
    pub gpu_success_rate: f32,
    /// ANE vs GPU speedup factor
    pub speedup_factor: f32,
    /// Total operations performed
    pub total_operations: usize,
    /// Session duration
    pub session_duration_ms: u64,
    /// Operation breakdown by type
    pub operation_breakdown: HashMap<ANEOperation, OperationStats>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
    /// Performance grade (A-F)
    pub performance_grade: char,
}

/// Statistics for specific operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub count: usize,
    pub avg_latency_ms: f32,
    pub success_rate: f32,
    pub speedup_vs_gpu: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            detailed_logging: false,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 100.0,
            min_success_rate: 95.0,
            max_degradation_percent: 20.0,
        }
    }
}

impl ANEPerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create performance monitor with custom configuration
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            operation_history: VecDeque::with_capacity(config.max_history_size),
            device_metrics: HashMap::new(),
            config,
            session_start: Instant::now(),
        }
    }

    /// Log an operation execution
    pub fn log_operation(&mut self, operation: ANEOperation, execution_time: f32, device: ExtendedDevice) {
        let record = OperationRecord {
            operation,
            device,
            execution_time: Duration::from_secs_f32(execution_time),
            timestamp: Instant::now(),
            success: true,
            error_message: None,
        };

        self.record_operation(record);
    }

    /// Log a failed operation
    pub fn log_failed_operation(&mut self, operation: ANEOperation, device: ExtendedDevice, error: &str) {
        let record = OperationRecord {
            operation,
            device,
            execution_time: Duration::ZERO,
            timestamp: Instant::now(),
            success: false,
            error_message: Some(error.to_string()),
        };

        self.record_operation(record);
    }

    /// Record comparison between MLX and ANE performance
    pub fn record_comparison(&mut self, mlx_time: Duration, ane_time: Duration, accuracy_diff: f32) {
        // Log MLX operation
        let mlx_record = OperationRecord {
            operation: ANEOperation::FullForward,
            device: ExtendedDevice::Gpu,
            execution_time: mlx_time,
            timestamp: Instant::now(),
            success: true,
            error_message: None,
        };

        // Log ANE operation
        let ane_record = OperationRecord {
            operation: ANEOperation::FullForward,
            device: ExtendedDevice::Ane,
            execution_time: ane_time,
            timestamp: Instant::now(),
            success: accuracy_diff < 5.0, // Consider successful if <5% accuracy difference
            error_message: if accuracy_diff >= 5.0 { 
                Some(format!("High accuracy difference: {:.2}%", accuracy_diff))
            } else { 
                None 
            },
        };

        self.record_operation(mlx_record);
        self.record_operation(ane_record);

        if self.config.detailed_logging {
            println!("📊 Performance comparison - MLX: {:.2}ms, ANE: {:.2}ms, Accuracy diff: {:.2}%", 
                    mlx_time.as_secs_f32() * 1000.0,
                    ane_time.as_secs_f32() * 1000.0,
                    accuracy_diff);
        }
    }

    /// Record individual operation
    fn record_operation(&mut self, record: OperationRecord) {
        // Update device metrics
        self.update_device_metrics(&record);

        // Add to history
        if self.operation_history.len() >= self.config.max_history_size {
            self.operation_history.pop_front();
        }
        self.operation_history.push_back(record);

        // Check for performance alerts
        self.check_performance_alerts();
    }

    /// Update device-specific metrics
    fn update_device_metrics(&mut self, record: &OperationRecord) {
        let metrics = self.device_metrics.entry(record.device).or_insert_with(|| DeviceMetrics {
            total_operations: 0,
            total_time: Duration::ZERO,
            successful_operations: 0,
            failed_operations: 0,
            avg_latency: Duration::ZERO,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            last_updated: Instant::now(),
        });

        metrics.total_operations += 1;
        metrics.total_time += record.execution_time;
        
        if record.success {
            metrics.successful_operations += 1;
            metrics.min_latency = metrics.min_latency.min(record.execution_time);
            metrics.max_latency = metrics.max_latency.max(record.execution_time);
        } else {
            metrics.failed_operations += 1;
        }

        // Update average latency
        if metrics.successful_operations > 0 {
            metrics.avg_latency = metrics.total_time / metrics.successful_operations as u32;
        }
        
        metrics.last_updated = Instant::now();
    }

    /// Check for performance alerts
    fn check_performance_alerts(&self) {
        if let Some(ane_metrics) = self.device_metrics.get(&ExtendedDevice::Ane) {
            let success_rate = (ane_metrics.successful_operations as f32 / ane_metrics.total_operations as f32) * 100.0;
            let avg_latency_ms = ane_metrics.avg_latency.as_secs_f32() * 1000.0;

            if success_rate < self.config.alert_thresholds.min_success_rate {
                println!("🚨 Performance Alert: ANE success rate {:.1}% below threshold {:.1}%",
                        success_rate, self.config.alert_thresholds.min_success_rate);
            }

            if avg_latency_ms > self.config.alert_thresholds.max_latency_ms {
                println!("🚨 Performance Alert: ANE latency {:.2}ms exceeds threshold {:.2}ms",
                        avg_latency_ms, self.config.alert_thresholds.max_latency_ms);
            }
        }
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> ANEPerformanceReport {
        let ane_metrics = self.device_metrics.get(&ExtendedDevice::Ane);
        let gpu_metrics = self.device_metrics.get(&ExtendedDevice::Gpu);

        let avg_ane_latency_ms = ane_metrics
            .map(|m| m.avg_latency.as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        let avg_gpu_latency_ms = gpu_metrics
            .map(|m| m.avg_latency.as_secs_f32() * 1000.0)
            .unwrap_or(0.0);

        let ane_success_rate = ane_metrics
            .map(|m| m.successful_operations as f32 / m.total_operations as f32)
            .unwrap_or(0.0);

        let gpu_success_rate = gpu_metrics
            .map(|m| m.successful_operations as f32 / m.total_operations as f32)
            .unwrap_or(0.0);

        let speedup_factor = if avg_ane_latency_ms > 0.0 && avg_gpu_latency_ms > 0.0 {
            avg_gpu_latency_ms / avg_ane_latency_ms
        } else {
            1.0
        };

        let total_operations = self.device_metrics.values()
            .map(|m| m.total_operations)
            .sum();

        let session_duration_ms = self.session_start.elapsed().as_millis() as u64;

        let operation_breakdown = self.generate_operation_breakdown();
        let recommendations = self.generate_recommendations();
        let performance_grade = self.calculate_performance_grade(ane_success_rate, speedup_factor);

        ANEPerformanceReport {
            avg_ane_latency_ms,
            avg_gpu_latency_ms,
            ane_success_rate,
            gpu_success_rate,
            speedup_factor,
            total_operations,
            session_duration_ms,
            operation_breakdown,
            recommendations,
            performance_grade,
        }
    }

    /// Generate operation breakdown statistics
    fn generate_operation_breakdown(&self) -> HashMap<ANEOperation, OperationStats> {
        let mut breakdown = HashMap::new();

        for operation in [ANEOperation::MatMul, ANEOperation::Attention, ANEOperation::LayerNorm, 
                         ANEOperation::Embedding, ANEOperation::Activation, ANEOperation::Softmax,
                         ANEOperation::FullForward] {
            
            let ane_ops: Vec<_> = self.operation_history.iter()
                .filter(|r| r.operation == operation && r.device == ExtendedDevice::Ane)
                .collect();
            
            let gpu_ops: Vec<_> = self.operation_history.iter()
                .filter(|r| r.operation == operation && r.device == ExtendedDevice::Gpu)
                .collect();

            if !ane_ops.is_empty() || !gpu_ops.is_empty() {
                let ane_avg_latency = if !ane_ops.is_empty() {
                    ane_ops.iter().map(|r| r.execution_time.as_secs_f32()).sum::<f32>() / ane_ops.len() as f32 * 1000.0
                } else {
                    0.0
                };

                let gpu_avg_latency = if !gpu_ops.is_empty() {
                    gpu_ops.iter().map(|r| r.execution_time.as_secs_f32()).sum::<f32>() / gpu_ops.len() as f32 * 1000.0
                } else {
                    0.0
                };

                let ane_success_rate = if !ane_ops.is_empty() {
                    ane_ops.iter().filter(|r| r.success).count() as f32 / ane_ops.len() as f32
                } else {
                    0.0
                };

                let speedup_vs_gpu = if ane_avg_latency > 0.0 && gpu_avg_latency > 0.0 {
                    gpu_avg_latency / ane_avg_latency
                } else {
                    1.0
                };

                breakdown.insert(operation, OperationStats {
                    count: ane_ops.len() + gpu_ops.len(),
                    avg_latency_ms: ane_avg_latency,
                    success_rate: ane_success_rate,
                    speedup_vs_gpu,
                });
            }
        }

        breakdown
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(ane_metrics) = self.device_metrics.get(&ExtendedDevice::Ane) {
            let success_rate = (ane_metrics.successful_operations as f32 / ane_metrics.total_operations as f32) * 100.0;
            let avg_latency_ms = ane_metrics.avg_latency.as_secs_f32() * 1000.0;

            if success_rate < 90.0 {
                recommendations.push("Consider enabling fallback to GPU for better reliability".to_string());
            }

            if avg_latency_ms > 50.0 {
                recommendations.push("ANE latency is high - consider model optimization or smaller batch sizes".to_string());
            }

            if ane_metrics.failed_operations > ane_metrics.successful_operations / 10 {
                recommendations.push("High ANE failure rate - verify model compatibility with ANE".to_string());
            }
        }

        if let (Some(ane_metrics), Some(gpu_metrics)) = (
            self.device_metrics.get(&ExtendedDevice::Ane),
            self.device_metrics.get(&ExtendedDevice::Gpu)
        ) {
            let ane_latency = ane_metrics.avg_latency.as_secs_f32() * 1000.0;
            let gpu_latency = gpu_metrics.avg_latency.as_secs_f32() * 1000.0;

            if gpu_latency < ane_latency {
                recommendations.push("GPU is faster than ANE - consider disabling ANE for this workload".to_string());
            } else {
                let speedup = gpu_latency / ane_latency;
                if speedup > 2.0 {
                    recommendations.push(format!("Excellent ANE performance: {:.1}x speedup vs GPU", speedup));
                }
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Performance looks good - no specific recommendations".to_string());
        }

        recommendations
    }

    /// Calculate performance grade
    fn calculate_performance_grade(&self, success_rate: f32, speedup_factor: f32) -> char {
        let success_score = (success_rate * 100.0).min(100.0);
        let speedup_score = (speedup_factor * 50.0).min(100.0);
        let overall_score = (success_score + speedup_score) / 2.0;

        match overall_score {
            90.0..=100.0 => 'A',
            80.0..=89.9 => 'B',
            70.0..=79.9 => 'C',
            60.0..=69.9 => 'D',
            _ => 'F',
        }
    }

    /// Generate human-readable report
    pub fn generate_report(&self, max_accuracy_diff: f32) -> ANEPerformanceReport {
        let mut report = self.generate_performance_report();
        
        // Add accuracy information to recommendations
        if max_accuracy_diff > 1.0 {
            report.recommendations.insert(0, 
                format!("Maximum accuracy difference: {:.2}% - consider model optimization", max_accuracy_diff)
            );
        }

        report
    }

    /// Get current session statistics
    pub fn get_session_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        stats.insert("session_duration".to_string(), 
                     format!("{:.2}s", self.session_start.elapsed().as_secs_f32()));
        
        stats.insert("total_operations".to_string(), 
                     self.operation_history.len().to_string());
        
        if let Some(ane_metrics) = self.device_metrics.get(&ExtendedDevice::Ane) {
            stats.insert("ane_operations".to_string(), 
                        ane_metrics.total_operations.to_string());
            stats.insert("ane_success_rate".to_string(), 
                        format!("{:.1}%", (ane_metrics.successful_operations as f32 / ane_metrics.total_operations as f32) * 100.0));
        }
        
        stats
    }

    /// Clear performance history
    pub fn clear_history(&mut self) {
        self.operation_history.clear();
        self.device_metrics.clear();
        self.session_start = Instant::now();
    }

    /// Enable or disable detailed logging
    pub fn set_detailed_logging(&mut self, enabled: bool) {
        self.config.detailed_logging = enabled;
    }
}