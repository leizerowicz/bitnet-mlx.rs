//! Conversion Metrics and Monitoring
//!
//! This module provides comprehensive metrics collection and monitoring
//! for the data conversion system, enabling performance analysis and optimization.

use crate::memory::conversion::{ConversionQuality, ConversionStrategy};
use crate::memory::tensor::BitNetDType;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Comprehensive metrics for conversion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMetrics {
    /// Total number of conversions performed
    pub total_conversions: u64,
    /// Total bytes processed
    pub total_bytes_processed: u64,
    /// Total time spent on conversions (milliseconds)
    pub total_time_ms: u64,
    /// Number of successful conversions
    pub successful_conversions: u64,
    /// Number of failed conversions
    pub failed_conversions: u64,
    /// Metrics by conversion strategy
    pub strategy_metrics: HashMap<ConversionStrategy, StrategyMetrics>,
    /// Metrics by data type conversion
    pub dtype_metrics: HashMap<DTypeConversion, DTypeMetrics>,
    /// Device-specific metrics
    pub device_metrics: HashMap<String, DeviceMetrics>,
    /// Memory efficiency statistics
    pub memory_stats: MemoryStats,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Error statistics
    pub error_stats: ErrorStats,
}

impl ConversionMetrics {
    /// Creates new empty metrics
    pub fn new() -> Self {
        Self {
            total_conversions: 0,
            total_bytes_processed: 0,
            total_time_ms: 0,
            successful_conversions: 0,
            failed_conversions: 0,
            strategy_metrics: HashMap::new(),
            dtype_metrics: HashMap::new(),
            device_metrics: HashMap::new(),
            memory_stats: MemoryStats::new(),
            performance_stats: PerformanceStats::new(),
            error_stats: ErrorStats::new(),
        }
    }

    /// Returns the success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_conversions == 0 {
            0.0
        } else {
            (self.successful_conversions as f64 / self.total_conversions as f64) * 100.0
        }
    }

    /// Returns the average conversion time in milliseconds
    pub fn average_time_ms(&self) -> f64 {
        if self.total_conversions == 0 {
            0.0
        } else {
            self.total_time_ms as f64 / self.total_conversions as f64
        }
    }

    /// Returns the throughput in bytes per second
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        if self.total_time_ms == 0 {
            0.0
        } else {
            (self.total_bytes_processed as f64 * 1000.0) / self.total_time_ms as f64
        }
    }

    /// Returns the most used conversion strategy
    pub fn most_used_strategy(&self) -> Option<ConversionStrategy> {
        self.strategy_metrics
            .iter()
            .max_by_key(|(_, metrics)| metrics.usage_count)
            .map(|(strategy, _)| *strategy)
    }

    /// Returns the fastest conversion strategy
    pub fn fastest_strategy(&self) -> Option<ConversionStrategy> {
        self.strategy_metrics
            .iter()
            .filter(|(_, metrics)| metrics.usage_count > 0)
            .min_by(|(_, a), (_, b)| {
                a.average_time_ms()
                    .partial_cmp(&b.average_time_ms())
                    .unwrap()
            })
            .map(|(strategy, _)| *strategy)
    }

    /// Merges another metrics instance into this one
    pub fn merge(&mut self, other: &ConversionMetrics) {
        self.total_conversions += other.total_conversions;
        self.total_bytes_processed += other.total_bytes_processed;
        self.total_time_ms += other.total_time_ms;
        self.successful_conversions += other.successful_conversions;
        self.failed_conversions += other.failed_conversions;

        // Merge strategy metrics
        for (strategy, metrics) in &other.strategy_metrics {
            self.strategy_metrics
                .entry(*strategy)
                .or_insert_with(StrategyMetrics::new)
                .merge(metrics);
        }

        // Merge dtype metrics
        for (conversion, metrics) in &other.dtype_metrics {
            self.dtype_metrics
                .entry(conversion.clone())
                .or_insert_with(DTypeMetrics::new)
                .merge(metrics);
        }

        // Merge device metrics
        for (device, metrics) in &other.device_metrics {
            self.device_metrics
                .entry(device.clone())
                .or_insert_with(DeviceMetrics::new)
                .merge(metrics);
        }

        self.memory_stats.merge(&other.memory_stats);
        self.performance_stats.merge(&other.performance_stats);
        self.error_stats.merge(&other.error_stats);
    }
}

impl Default for ConversionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for a specific conversion strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub usage_count: u64,
    pub total_time_ms: u64,
    pub total_bytes: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub min_time_ms: u64,
    pub max_time_ms: u64,
}

impl StrategyMetrics {
    pub fn new() -> Self {
        Self {
            usage_count: 0,
            total_time_ms: 0,
            total_bytes: 0,
            success_count: 0,
            failure_count: 0,
            min_time_ms: u64::MAX,
            max_time_ms: 0,
        }
    }

    pub fn average_time_ms(&self) -> f64 {
        if self.usage_count == 0 {
            0.0
        } else {
            self.total_time_ms as f64 / self.usage_count as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            0.0
        } else {
            (self.success_count as f64 / self.usage_count as f64) * 100.0
        }
    }

    pub fn merge(&mut self, other: &StrategyMetrics) {
        self.usage_count += other.usage_count;
        self.total_time_ms += other.total_time_ms;
        self.total_bytes += other.total_bytes;
        self.success_count += other.success_count;
        self.failure_count += other.failure_count;
        self.min_time_ms = self.min_time_ms.min(other.min_time_ms);
        self.max_time_ms = self.max_time_ms.max(other.max_time_ms);
    }
}

/// Data type conversion pair
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DTypeConversion {
    pub from: BitNetDType,
    pub to: BitNetDType,
}

impl DTypeConversion {
    pub fn new(from: BitNetDType, to: BitNetDType) -> Self {
        Self { from, to }
    }
}

/// Metrics for a specific data type conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTypeMetrics {
    pub conversion_count: u64,
    pub total_time_ms: u64,
    pub total_elements: u64,
    pub memory_efficiency_gain: f64,
    pub average_compression_ratio: f64,
}

impl DTypeMetrics {
    pub fn new() -> Self {
        Self {
            conversion_count: 0,
            total_time_ms: 0,
            total_elements: 0,
            memory_efficiency_gain: 0.0,
            average_compression_ratio: 1.0,
        }
    }

    pub fn average_time_ms(&self) -> f64 {
        if self.conversion_count == 0 {
            0.0
        } else {
            self.total_time_ms as f64 / self.conversion_count as f64
        }
    }

    pub fn merge(&mut self, other: &DTypeMetrics) {
        let total_conversions = self.conversion_count + other.conversion_count;
        if total_conversions > 0 {
            self.memory_efficiency_gain = (self.memory_efficiency_gain
                * self.conversion_count as f64
                + other.memory_efficiency_gain * other.conversion_count as f64)
                / total_conversions as f64;
            self.average_compression_ratio = (self.average_compression_ratio
                * self.conversion_count as f64
                + other.average_compression_ratio * other.conversion_count as f64)
                / total_conversions as f64;
        }

        self.conversion_count += other.conversion_count;
        self.total_time_ms += other.total_time_ms;
        self.total_elements += other.total_elements;
    }
}

/// Device-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetrics {
    pub conversions_count: u64,
    pub total_time_ms: u64,
    pub total_bytes: u64,
    pub device_utilization: f64,
    pub memory_bandwidth_gbps: f64,
}

impl DeviceMetrics {
    pub fn new() -> Self {
        Self {
            conversions_count: 0,
            total_time_ms: 0,
            total_bytes: 0,
            device_utilization: 0.0,
            memory_bandwidth_gbps: 0.0,
        }
    }

    pub fn merge(&mut self, other: &DeviceMetrics) {
        let total_conversions = self.conversions_count + other.conversions_count;
        if total_conversions > 0 {
            self.device_utilization = (self.device_utilization * self.conversions_count as f64
                + other.device_utilization * other.conversions_count as f64)
                / total_conversions as f64;
            self.memory_bandwidth_gbps = (self.memory_bandwidth_gbps
                * self.conversions_count as f64
                + other.memory_bandwidth_gbps * other.conversions_count as f64)
                / total_conversions as f64;
        }

        self.conversions_count += other.conversions_count;
        self.total_time_ms += other.total_time_ms;
        self.total_bytes += other.total_bytes;
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub peak_memory_usage: u64,
    pub total_memory_allocated: u64,
    pub total_memory_freed: u64,
    pub memory_efficiency_ratio: f64,
    pub zero_copy_percentage: f64,
    pub in_place_percentage: f64,
}

impl MemoryStats {
    pub fn new() -> Self {
        Self {
            peak_memory_usage: 0,
            total_memory_allocated: 0,
            total_memory_freed: 0,
            memory_efficiency_ratio: 1.0,
            zero_copy_percentage: 0.0,
            in_place_percentage: 0.0,
        }
    }

    pub fn merge(&mut self, other: &MemoryStats) {
        self.peak_memory_usage = self.peak_memory_usage.max(other.peak_memory_usage);
        self.total_memory_allocated += other.total_memory_allocated;
        self.total_memory_freed += other.total_memory_freed;

        // Average the ratios and percentages
        self.memory_efficiency_ratio =
            (self.memory_efficiency_ratio + other.memory_efficiency_ratio) / 2.0;
        self.zero_copy_percentage = (self.zero_copy_percentage + other.zero_copy_percentage) / 2.0;
        self.in_place_percentage = (self.in_place_percentage + other.in_place_percentage) / 2.0;
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub min_conversion_time_ms: u64,
    pub max_conversion_time_ms: u64,
    pub p50_conversion_time_ms: u64,
    pub p95_conversion_time_ms: u64,
    pub p99_conversion_time_ms: u64,
    pub throughput_elements_per_sec: f64,
    pub cache_hit_ratio: f64,
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            min_conversion_time_ms: u64::MAX,
            max_conversion_time_ms: 0,
            p50_conversion_time_ms: 0,
            p95_conversion_time_ms: 0,
            p99_conversion_time_ms: 0,
            throughput_elements_per_sec: 0.0,
            cache_hit_ratio: 0.0,
        }
    }

    pub fn merge(&mut self, other: &PerformanceStats) {
        self.min_conversion_time_ms = self
            .min_conversion_time_ms
            .min(other.min_conversion_time_ms);
        self.max_conversion_time_ms = self
            .max_conversion_time_ms
            .max(other.max_conversion_time_ms);

        // For percentiles and ratios, take the average (simplified approach)
        self.p50_conversion_time_ms =
            (self.p50_conversion_time_ms + other.p50_conversion_time_ms) / 2;
        self.p95_conversion_time_ms =
            (self.p95_conversion_time_ms + other.p95_conversion_time_ms) / 2;
        self.p99_conversion_time_ms =
            (self.p99_conversion_time_ms + other.p99_conversion_time_ms) / 2;
        self.throughput_elements_per_sec =
            (self.throughput_elements_per_sec + other.throughput_elements_per_sec) / 2.0;
        self.cache_hit_ratio = (self.cache_hit_ratio + other.cache_hit_ratio) / 2.0;
    }
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    pub total_errors: u64,
    pub memory_errors: u64,
    pub unsupported_conversion_errors: u64,
    pub device_errors: u64,
    pub timeout_errors: u64,
    pub data_loss_errors: u64,
    pub error_rate: f64,
}

impl ErrorStats {
    pub fn new() -> Self {
        Self {
            total_errors: 0,
            memory_errors: 0,
            unsupported_conversion_errors: 0,
            device_errors: 0,
            timeout_errors: 0,
            data_loss_errors: 0,
            error_rate: 0.0,
        }
    }

    pub fn merge(&mut self, other: &ErrorStats) {
        self.total_errors += other.total_errors;
        self.memory_errors += other.memory_errors;
        self.unsupported_conversion_errors += other.unsupported_conversion_errors;
        self.device_errors += other.device_errors;
        self.timeout_errors += other.timeout_errors;
        self.data_loss_errors += other.data_loss_errors;
        self.error_rate = (self.error_rate + other.error_rate) / 2.0;
    }
}

/// Represents a single conversion event for detailed tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionEvent {
    pub timestamp: u64,
    pub source_dtype: BitNetDType,
    pub target_dtype: BitNetDType,
    pub strategy: ConversionStrategy,
    pub quality: ConversionQuality,
    pub device: String,
    pub input_size_bytes: usize,
    pub output_size_bytes: usize,
    pub element_count: usize,
    pub duration_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
    pub memory_allocated: usize,
    pub memory_peak: usize,
}

impl ConversionEvent {
    /// Creates a new conversion event
    pub fn new(
        source_dtype: BitNetDType,
        target_dtype: BitNetDType,
        strategy: ConversionStrategy,
        quality: ConversionQuality,
        device: &Device,
        input_size_bytes: usize,
        output_size_bytes: usize,
        element_count: usize,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let device_str = match device {
            Device::Cpu => "cpu".to_string(),
            Device::Metal(metal_device) => format!("metal:{:?}", metal_device.id()),
            Device::Cuda(cuda_device) => format!("cuda:{:?}", cuda_device),
        };

        Self {
            timestamp,
            source_dtype,
            target_dtype,
            strategy,
            quality,
            device: device_str,
            input_size_bytes,
            output_size_bytes,
            element_count,
            duration_ms: 0,
            success: false,
            error_message: None,
            memory_allocated: 0,
            memory_peak: 0,
        }
    }

    /// Marks the event as completed successfully
    pub fn complete_success(
        mut self,
        duration: Duration,
        memory_allocated: usize,
        memory_peak: usize,
    ) -> Self {
        self.duration_ms = duration.as_millis() as u64;
        self.success = true;
        self.memory_allocated = memory_allocated;
        self.memory_peak = memory_peak;
        self
    }

    /// Marks the event as failed
    pub fn complete_failure(mut self, duration: Duration, error: String) -> Self {
        self.duration_ms = duration.as_millis() as u64;
        self.success = false;
        self.error_message = Some(error);
        self
    }

    /// Returns the compression ratio achieved
    pub fn compression_ratio(&self) -> f64 {
        if self.output_size_bytes == 0 {
            1.0
        } else {
            self.input_size_bytes as f64 / self.output_size_bytes as f64
        }
    }

    /// Returns the processing speed in elements per second
    pub fn elements_per_second(&self) -> f64 {
        if self.duration_ms == 0 {
            0.0
        } else {
            (self.element_count as f64 * 1000.0) / self.duration_ms as f64
        }
    }
}

/// Statistics aggregator for conversion events
#[derive(Debug)]
pub struct ConversionStats {
    events: Arc<RwLock<Vec<ConversionEvent>>>,
    max_events: usize,
}

impl ConversionStats {
    /// Creates a new stats aggregator
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events,
        }
    }

    /// Records a conversion event
    pub fn record_event(&self, event: ConversionEvent) {
        if let Ok(mut events) = self.events.write() {
            events.push(event);

            // Keep only the most recent events
            let max_events = self.max_events;
            if events.len() > max_events {
                let excess = events.len() - max_events;
                events.drain(0..excess);
            }
        }
    }

    /// Generates comprehensive metrics from recorded events
    pub fn generate_metrics(&self) -> ConversionMetrics {
        let events = match self.events.read() {
            Ok(events) => events.clone(),
            Err(_) => return ConversionMetrics::new(),
        };

        let mut metrics = ConversionMetrics::new();

        for event in &events {
            metrics.total_conversions += 1;
            metrics.total_bytes_processed += event.input_size_bytes as u64;
            metrics.total_time_ms += event.duration_ms;

            if event.success {
                metrics.successful_conversions += 1;
            } else {
                metrics.failed_conversions += 1;
            }

            // Update strategy metrics
            let strategy_metrics = metrics
                .strategy_metrics
                .entry(event.strategy)
                .or_insert_with(StrategyMetrics::new);

            strategy_metrics.usage_count += 1;
            strategy_metrics.total_time_ms += event.duration_ms;
            strategy_metrics.total_bytes += event.input_size_bytes as u64;

            if event.success {
                strategy_metrics.success_count += 1;
            } else {
                strategy_metrics.failure_count += 1;
            }

            strategy_metrics.min_time_ms = strategy_metrics.min_time_ms.min(event.duration_ms);
            strategy_metrics.max_time_ms = strategy_metrics.max_time_ms.max(event.duration_ms);

            // Update dtype metrics
            let dtype_conversion = DTypeConversion::new(event.source_dtype, event.target_dtype);
            let dtype_metrics = metrics
                .dtype_metrics
                .entry(dtype_conversion)
                .or_insert_with(DTypeMetrics::new);

            dtype_metrics.conversion_count += 1;
            dtype_metrics.total_time_ms += event.duration_ms;
            dtype_metrics.total_elements += event.element_count as u64;

            // Update device metrics
            let device_metrics = metrics
                .device_metrics
                .entry(event.device.clone())
                .or_insert_with(DeviceMetrics::new);

            device_metrics.conversions_count += 1;
            device_metrics.total_time_ms += event.duration_ms;
            device_metrics.total_bytes += event.input_size_bytes as u64;

            // Update memory stats
            metrics.memory_stats.peak_memory_usage = metrics
                .memory_stats
                .peak_memory_usage
                .max(event.memory_peak as u64);
            metrics.memory_stats.total_memory_allocated += event.memory_allocated as u64;
        }

        // Calculate derived statistics
        self.calculate_performance_stats(&events, &mut metrics);
        self.calculate_error_stats(&events, &mut metrics);

        metrics
    }

    /// Calculates performance statistics from events
    fn calculate_performance_stats(
        &self,
        events: &[ConversionEvent],
        metrics: &mut ConversionMetrics,
    ) {
        if events.is_empty() {
            return;
        }

        let mut durations: Vec<u64> = events.iter().map(|e| e.duration_ms).collect();
        durations.sort_unstable();

        metrics.performance_stats.min_conversion_time_ms = durations[0];
        metrics.performance_stats.max_conversion_time_ms = durations[durations.len() - 1];

        let len = durations.len();
        metrics.performance_stats.p50_conversion_time_ms = durations[len / 2];
        metrics.performance_stats.p95_conversion_time_ms = durations[(len * 95) / 100];
        metrics.performance_stats.p99_conversion_time_ms = durations[(len * 99) / 100];

        let total_elements: u64 = events.iter().map(|e| e.element_count as u64).sum();
        let total_time_sec = metrics.total_time_ms as f64 / 1000.0;
        if total_time_sec > 0.0 {
            metrics.performance_stats.throughput_elements_per_sec =
                total_elements as f64 / total_time_sec;
        }
    }

    /// Calculates error statistics from events
    fn calculate_error_stats(&self, events: &[ConversionEvent], metrics: &mut ConversionMetrics) {
        for event in events {
            if !event.success {
                metrics.error_stats.total_errors += 1;

                if let Some(ref error_msg) = event.error_message {
                    if error_msg.contains("memory") || error_msg.contains("Memory") {
                        metrics.error_stats.memory_errors += 1;
                    } else if error_msg.contains("unsupported") || error_msg.contains("Unsupported")
                    {
                        metrics.error_stats.unsupported_conversion_errors += 1;
                    } else if error_msg.contains("device") || error_msg.contains("Device") {
                        metrics.error_stats.device_errors += 1;
                    } else if error_msg.contains("timeout") || error_msg.contains("Timeout") {
                        metrics.error_stats.timeout_errors += 1;
                    } else if error_msg.contains("data loss") || error_msg.contains("Data loss") {
                        metrics.error_stats.data_loss_errors += 1;
                    }
                }
            }
        }

        if metrics.total_conversions > 0 {
            metrics.error_stats.error_rate = (metrics.error_stats.total_errors as f64
                / metrics.total_conversions as f64)
                * 100.0;
        }
    }

    /// Returns the most recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<ConversionEvent> {
        match self.events.read() {
            Ok(events) => {
                let start = if events.len() > count {
                    events.len() - count
                } else {
                    0
                };
                events[start..].to_vec()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Clears all recorded events
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.write() {
            events.clear();
        }
    }

    /// Returns the total number of recorded events
    pub fn event_count(&self) -> usize {
        match self.events.read() {
            Ok(events) => events.len(),
            Err(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use std::time::Duration;

    #[test]
    fn test_conversion_metrics_creation() {
        let metrics = ConversionMetrics::new();
        assert_eq!(metrics.total_conversions, 0);
        assert_eq!(metrics.success_rate(), 0.0);
        assert_eq!(metrics.average_time_ms(), 0.0);
    }

    #[test]
    fn test_strategy_metrics() {
        let mut metrics = StrategyMetrics::new();
        assert_eq!(metrics.usage_count, 0);
        assert_eq!(metrics.average_time_ms(), 0.0);
        assert_eq!(metrics.success_rate(), 0.0);

        metrics.usage_count = 10;
        metrics.total_time_ms = 1000;
        metrics.success_count = 8;

        assert_eq!(metrics.average_time_ms(), 100.0);
        assert_eq!(metrics.success_rate(), 80.0);
    }

    #[test]
    fn test_dtype_conversion() {
        let conversion = DTypeConversion::new(BitNetDType::F32, BitNetDType::F16);
        assert_eq!(conversion.from, BitNetDType::F32);
        assert_eq!(conversion.to, BitNetDType::F16);
    }

    #[test]
    fn test_conversion_event() {
        let device = get_cpu_device();
        let event = ConversionEvent::new(
            BitNetDType::F32,
            BitNetDType::F16,
            ConversionStrategy::InPlace,
            ConversionQuality::Balanced,
            &device,
            1024,
            512,
            256,
        );

        assert_eq!(event.source_dtype, BitNetDType::F32);
        assert_eq!(event.target_dtype, BitNetDType::F16);
        assert_eq!(event.compression_ratio(), 2.0);
        assert!(!event.success);

        let completed = event.complete_success(Duration::from_millis(100), 1024, 2048);
        assert!(completed.success);
        assert_eq!(completed.duration_ms, 100);
        assert_eq!(completed.elements_per_second(), 2560.0);
    }

    #[test]
    fn test_conversion_stats() {
        let stats = ConversionStats::new(100);
        assert_eq!(stats.event_count(), 0);

        let device = get_cpu_device();
        let event = ConversionEvent::new(
            BitNetDType::F32,
            BitNetDType::F16,
            ConversionStrategy::InPlace,
            ConversionQuality::Balanced,
            &device,
            1024,
            512,
            256,
        )
        .complete_success(Duration::from_millis(50), 1024, 1024);

        stats.record_event(event);
        assert_eq!(stats.event_count(), 1);

        let metrics = stats.generate_metrics();
        assert_eq!(metrics.total_conversions, 1);
        assert_eq!(metrics.successful_conversions, 1);
        assert_eq!(metrics.success_rate(), 100.0);
    }

    #[test]
    fn test_metrics_merging() {
        let mut metrics1 = ConversionMetrics::new();
        metrics1.total_conversions = 10;
        metrics1.successful_conversions = 8;
        metrics1.total_time_ms = 1000;

        let mut metrics2 = ConversionMetrics::new();
        metrics2.total_conversions = 5;
        metrics2.successful_conversions = 4;
        metrics2.total_time_ms = 500;

        metrics1.merge(&metrics2);

        assert_eq!(metrics1.total_conversions, 15);
        assert_eq!(metrics1.successful_conversions, 12);
        assert_eq!(metrics1.total_time_ms, 1500);
        assert_eq!(metrics1.success_rate(), 80.0);
    }

    #[test]
    fn test_stats_max_events_limit() {
        let stats = ConversionStats::new(2); // Limit to 2 events
        let device = get_cpu_device();

        // Add 3 events
        for i in 0..3 {
            let event = ConversionEvent::new(
                BitNetDType::F32,
                BitNetDType::F16,
                ConversionStrategy::InPlace,
                ConversionQuality::Balanced,
                &device,
                1024,
                512,
                256,
            )
            .complete_success(Duration::from_millis(50), 1024, 1024);

            stats.record_event(event);
        }

        // Should only keep the last 2 events
        assert_eq!(stats.event_count(), 2);

        let recent_events = stats.get_recent_events(5);
        assert_eq!(recent_events.len(), 2);
    }
}
