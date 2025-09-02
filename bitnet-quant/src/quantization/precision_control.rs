//! Advanced Quantization Precision Control System
//!
//! This module provides comprehensive precision control mechanisms for BitNet quantization,
//! including dynamic precision adjustment, precision bounds validation, and precision monitoring.

use super::{QuantizationError, QuantizationPrecision, QuantizationResult, QuantizationStats};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive precision control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionControlConfig {
    /// Target precision for quantization
    pub target_precision: QuantizationPrecision,
    /// Precision bounds and constraints
    pub precision_bounds: PrecisionBounds,
    /// Dynamic adjustment settings
    pub dynamic_adjustment: DynamicAdjustmentConfig,
    /// Precision monitoring configuration
    pub monitoring: PrecisionMonitoringConfig,
    /// Validation settings
    pub validation: PrecisionValidationConfig,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

impl Default for PrecisionControlConfig {
    fn default() -> Self {
        Self {
            target_precision: QuantizationPrecision::OneFiveFiveBit,
            precision_bounds: PrecisionBounds::default(),
            dynamic_adjustment: DynamicAdjustmentConfig::default(),
            monitoring: PrecisionMonitoringConfig::default(),
            validation: PrecisionValidationConfig::default(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Precision bounds and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionBounds {
    /// Minimum allowed precision
    pub min_precision: QuantizationPrecision,
    /// Maximum allowed precision
    pub max_precision: QuantizationPrecision,
    /// Minimum threshold value
    pub min_threshold: f32,
    /// Maximum threshold value
    pub max_threshold: f32,
    /// Minimum scale factor
    pub min_scale: f32,
    /// Maximum scale factor
    pub max_scale: f32,
    /// Maximum quantization error tolerance
    pub max_error_tolerance: f32,
    /// Minimum compression ratio requirement
    pub min_compression_ratio: f32,
}

impl Default for PrecisionBounds {
    fn default() -> Self {
        Self {
            min_precision: QuantizationPrecision::OneBit,
            max_precision: QuantizationPrecision::EightBit,
            min_threshold: 1e-6,
            max_threshold: 10.0,
            min_scale: 1e-8,
            max_scale: 1e8,
            max_error_tolerance: 0.1,
            min_compression_ratio: 1.5,
        }
    }
}

/// Dynamic precision adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAdjustmentConfig {
    /// Enable dynamic precision adjustment
    pub enabled: bool,
    /// Adjustment strategy
    pub strategy: AdjustmentStrategy,
    /// Evaluation window size (number of samples)
    pub evaluation_window: usize,
    /// Adjustment frequency (every N evaluations)
    pub adjustment_frequency: usize,
    /// Learning rate for gradual adjustments
    pub learning_rate: f32,
    /// Stability threshold (minimum improvement required)
    pub stability_threshold: f32,
    /// Maximum adjustments per session
    pub max_adjustments: usize,
}

impl Default for DynamicAdjustmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: AdjustmentStrategy::Adaptive,
            evaluation_window: 100,
            adjustment_frequency: 10,
            learning_rate: 0.1,
            stability_threshold: 0.01,
            max_adjustments: 5,
        }
    }
}

/// Precision adjustment strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentStrategy {
    /// Conservative adjustments prioritizing stability
    Conservative,
    /// Balanced adjustments between performance and accuracy
    Balanced,
    /// Aggressive adjustments prioritizing performance
    Aggressive,
    /// Adaptive adjustments based on current metrics
    Adaptive,
    /// Custom strategy with user-defined parameters
    Custom,
}

/// Precision monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionMonitoringConfig {
    /// Enable precision monitoring
    pub enabled: bool,
    /// Metrics to track
    pub tracked_metrics: Vec<PrecisionMetric>,
    /// History size for metrics
    pub history_size: usize,
    /// Sampling frequency for monitoring
    pub sampling_frequency: Duration,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for PrecisionMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tracked_metrics: vec![
                PrecisionMetric::QuantizationError,
                PrecisionMetric::CompressionRatio,
                PrecisionMetric::ProcessingTime,
                PrecisionMetric::MemoryUsage,
            ],
            history_size: 1000,
            sampling_frequency: Duration::from_millis(100),
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Precision metrics to track
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMetric {
    /// Quantization error (MSE)
    QuantizationError,
    /// Compression ratio achieved
    CompressionRatio,
    /// Processing time per operation
    ProcessingTime,
    /// Memory usage
    MemoryUsage,
    /// Threshold stability
    ThresholdStability,
    /// Scale factor variance
    ScaleVariance,
    /// Sparsity ratio
    SparsityRatio,
    /// Signal-to-noise ratio
    SignalToNoiseRatio,
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum acceptable quantization error
    pub max_quantization_error: f32,
    /// Minimum acceptable compression ratio
    pub min_compression_ratio: f32,
    /// Maximum acceptable processing time (ms)
    pub max_processing_time_ms: f32,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: f32,
    /// Threshold for unstable behavior
    pub instability_threshold: f32,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_quantization_error: 0.1,
            min_compression_ratio: 2.0,
            max_processing_time_ms: 100.0,
            max_memory_usage_mb: 1024.0,
            instability_threshold: 0.05,
        }
    }
}

/// Precision validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionValidationConfig {
    /// Enable strict validation
    pub strict_validation: bool,
    /// Validate precision bounds
    pub validate_bounds: bool,
    /// Validate numerical stability
    pub validate_stability: bool,
    /// Validate compression requirements
    pub validate_compression: bool,
    /// Tolerance for validation checks
    pub validation_tolerance: f32,
    /// Number of validation samples
    pub validation_samples: usize,
}

impl Default for PrecisionValidationConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            validate_bounds: true,
            validate_stability: true,
            validate_compression: true,
            validation_tolerance: 1e-6,
            validation_samples: 100,
        }
    }
}

/// Performance thresholds for precision control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum accuracy threshold
    pub min_accuracy: f32,
    /// Maximum latency threshold (ms)
    pub max_latency_ms: f32,
    /// Maximum memory overhead percentage
    pub max_memory_overhead_pct: f32,
    /// Minimum throughput (ops/sec)
    pub min_throughput: f32,
    /// Energy efficiency threshold
    pub min_energy_efficiency: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.95,
            max_latency_ms: 10.0,
            max_memory_overhead_pct: 20.0,
            min_throughput: 1000.0,
            min_energy_efficiency: 0.8,
        }
    }
}

/// Precision control manager
#[derive(Debug)]
pub struct PrecisionController {
    /// Configuration
    config: PrecisionControlConfig,
    /// Current precision state
    current_state: PrecisionState,
    /// Metrics history
    metrics_history: MetricsHistory,
    /// Adjustment history
    adjustment_history: Vec<PrecisionAdjustment>,
    /// Device for operations
    device: Device,
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
}

/// Current precision state
#[derive(Debug, Clone)]
pub struct PrecisionState {
    /// Current precision level
    pub precision: QuantizationPrecision,
    /// Current threshold values
    pub thresholds: HashMap<String, f32>,
    /// Current scale factors
    pub scale_factors: HashMap<String, f32>,
    /// Last update timestamp
    pub last_update: Instant,
    /// Stability score
    pub stability_score: f32,
    /// Performance score
    pub performance_score: f32,
}

/// Metrics history tracking
#[derive(Debug)]
pub struct MetricsHistory {
    /// Quantization error history
    pub quantization_errors: Vec<(Instant, f32)>,
    /// Compression ratio history
    pub compression_ratios: Vec<(Instant, f32)>,
    /// Processing time history
    pub processing_times: Vec<(Instant, Duration)>,
    /// Memory usage history
    pub memory_usage: Vec<(Instant, usize)>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Vec<(Instant, f32)>>,
}

/// Precision adjustment record
#[derive(Debug, Clone)]
pub struct PrecisionAdjustment {
    /// Timestamp of adjustment
    pub timestamp: Instant,
    /// Previous precision
    pub from_precision: QuantizationPrecision,
    /// New precision
    pub to_precision: QuantizationPrecision,
    /// Reason for adjustment
    pub reason: AdjustmentReason,
    /// Performance impact
    pub performance_impact: PerformanceImpact,
    /// Success status
    pub success: bool,
}

/// Reasons for precision adjustments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentReason {
    /// High quantization error detected
    HighError,
    /// Low compression ratio
    LowCompression,
    /// Performance degradation
    PerformanceDegradation,
    /// Memory pressure
    MemoryPressure,
    /// Stability issues
    Instability,
    /// User request
    UserRequest,
    /// Automatic optimization
    AutoOptimization,
}

/// Performance impact of adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Change in quantization error
    pub error_delta: f32,
    /// Change in compression ratio
    pub compression_delta: f32,
    /// Change in processing time
    pub time_delta: Duration,
    /// Change in memory usage
    pub memory_delta: i64,
    /// Overall impact score
    pub impact_score: f32,
}

/// Performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Start time for current session
    session_start: Instant,
    /// Total operations processed
    operations_count: u64,
    /// Total processing time
    total_processing_time: Duration,
    /// Peak memory usage
    peak_memory_usage: usize,
    /// Current metrics
    current_metrics: HashMap<PrecisionMetric, f32>,
}

impl PrecisionController {
    /// Create a new precision controller
    pub fn new(config: PrecisionControlConfig, device: Device) -> QuantizationResult<Self> {
        config.validate()?;

        let current_state = PrecisionState {
            precision: config.target_precision,
            thresholds: HashMap::new(),
            scale_factors: HashMap::new(),
            last_update: Instant::now(),
            stability_score: 1.0,
            performance_score: 1.0,
        };

        let metrics_history = MetricsHistory {
            quantization_errors: Vec::new(),
            compression_ratios: Vec::new(),
            processing_times: Vec::new(),
            memory_usage: Vec::new(),
            custom_metrics: HashMap::new(),
        };

        let performance_monitor = PerformanceMonitor {
            session_start: Instant::now(),
            operations_count: 0,
            total_processing_time: Duration::new(0, 0),
            peak_memory_usage: 0,
            current_metrics: HashMap::new(),
        };

        Ok(Self {
            config,
            current_state,
            metrics_history,
            adjustment_history: Vec::new(),
            device,
            performance_monitor,
        })
    }

    /// Validate precision bounds for a given configuration
    pub fn validate_precision_bounds(
        &self,
        precision: QuantizationPrecision,
        threshold: f32,
        scale: f32,
    ) -> QuantizationResult<()> {
        let bounds = &self.config.precision_bounds;

        // Check precision bounds
        if !self.is_precision_in_bounds(precision) {
            return Err(QuantizationError::ValidationFailed(format!(
                "Precision {:?} is outside allowed bounds [{:?}, {:?}]",
                precision, bounds.min_precision, bounds.max_precision
            )));
        }

        // Check threshold bounds
        if threshold < bounds.min_threshold || threshold > bounds.max_threshold {
            return Err(QuantizationError::ValidationFailed(format!(
                "Threshold {} is outside bounds [{}, {}]",
                threshold, bounds.min_threshold, bounds.max_threshold
            )));
        }

        // Check scale bounds
        if scale < bounds.min_scale || scale > bounds.max_scale {
            return Err(QuantizationError::ValidationFailed(format!(
                "Scale {} is outside bounds [{}, {}]",
                scale, bounds.min_scale, bounds.max_scale
            )));
        }

        Ok(())
    }

    /// Check if precision is within allowed bounds
    pub fn is_precision_in_bounds(&self, precision: QuantizationPrecision) -> bool {
        let bounds = &self.config.precision_bounds;
        let precision_order = self.get_precision_order(precision);
        let min_order = self.get_precision_order(bounds.min_precision);
        let max_order = self.get_precision_order(bounds.max_precision);

        precision_order >= min_order && precision_order <= max_order
    }

    /// Get numerical order for precision comparison
    fn get_precision_order(&self, precision: QuantizationPrecision) -> u8 {
        match precision {
            QuantizationPrecision::OneBit => 1,
            QuantizationPrecision::OneFiveFiveBit => 2,
            QuantizationPrecision::TwoBit => 3,
            QuantizationPrecision::FourBit => 4,
            QuantizationPrecision::EightBit => 5,
        }
    }

    /// Adjust precision dynamically based on current metrics
    pub fn adjust_precision_dynamically(
        &mut self,
        current_metrics: &QuantizationStats,
    ) -> QuantizationResult<Option<PrecisionAdjustment>> {
        if !self.config.dynamic_adjustment.enabled {
            return Ok(None);
        }

        // Analyze current performance
        let performance_analysis = self.analyze_performance(current_metrics)?;

        // Determine if adjustment is needed
        let adjustment_decision = self.make_adjustment_decision(&performance_analysis)?;

        if let Some((new_precision, reason)) = adjustment_decision {
            let adjustment = self.apply_precision_adjustment(new_precision, reason)?;
            return Ok(Some(adjustment));
        }

        Ok(None)
    }

    /// Analyze current performance metrics
    fn analyze_performance(
        &self,
        metrics: &QuantizationStats,
    ) -> QuantizationResult<PerformanceAnalysis> {
        let error_score = self.evaluate_error_performance(metrics.quantization_error);
        let compression_score = self.evaluate_compression_performance(metrics.compression_ratio);
        let stability_score = self.current_state.stability_score;

        // Calculate overall performance score
        let overall_score = (error_score + compression_score + stability_score) / 3.0;

        Ok(PerformanceAnalysis {
            error_score,
            compression_score,
            stability_score,
            overall_score,
            needs_adjustment: overall_score < 0.7, // Threshold for adjustment
        })
    }

    /// Evaluate error performance
    fn evaluate_error_performance(&self, error: f32) -> f32 {
        let max_error = self.config.precision_bounds.max_error_tolerance;
        if error <= max_error * 0.5 {
            1.0 // Excellent
        } else if error <= max_error {
            0.8 // Good
        } else if error <= max_error * 1.5 {
            0.6 // Acceptable
        } else {
            0.3 // Poor
        }
    }

    /// Evaluate compression performance
    fn evaluate_compression_performance(&self, ratio: f32) -> f32 {
        let min_ratio = self.config.precision_bounds.min_compression_ratio;
        if ratio >= min_ratio * 2.0 {
            1.0 // Excellent
        } else if ratio >= min_ratio * 1.5 {
            0.8 // Good
        } else if ratio >= min_ratio {
            0.6 // Acceptable
        } else {
            0.3 // Poor
        }
    }

    /// Make decision about precision adjustment
    fn make_adjustment_decision(
        &self,
        analysis: &PerformanceAnalysis,
    ) -> QuantizationResult<Option<(QuantizationPrecision, AdjustmentReason)>> {
        if !analysis.needs_adjustment {
            return Ok(None);
        }

        let current_precision = self.current_state.precision;

        // Determine adjustment direction and reason
        if analysis.error_score < 0.5 {
            // High error - increase precision if possible
            if let Some(higher_precision) = self.get_higher_precision(current_precision) {
                return Ok(Some((higher_precision, AdjustmentReason::HighError)));
            }
        } else if analysis.compression_score < 0.5 {
            // Low compression - decrease precision if possible
            if let Some(lower_precision) = self.get_lower_precision(current_precision) {
                return Ok(Some((lower_precision, AdjustmentReason::LowCompression)));
            }
        }

        Ok(None)
    }

    /// Get next higher precision level
    fn get_higher_precision(
        &self,
        current: QuantizationPrecision,
    ) -> Option<QuantizationPrecision> {
        let next = match current {
            QuantizationPrecision::OneBit => QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::OneFiveFiveBit => QuantizationPrecision::TwoBit,
            QuantizationPrecision::TwoBit => QuantizationPrecision::FourBit,
            QuantizationPrecision::FourBit => QuantizationPrecision::EightBit,
            QuantizationPrecision::EightBit => return None,
        };

        if self.is_precision_in_bounds(next) {
            Some(next)
        } else {
            None
        }
    }

    /// Get next lower precision level
    fn get_lower_precision(&self, current: QuantizationPrecision) -> Option<QuantizationPrecision> {
        let next = match current {
            QuantizationPrecision::OneBit => return None,
            QuantizationPrecision::OneFiveFiveBit => QuantizationPrecision::OneBit,
            QuantizationPrecision::TwoBit => QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::FourBit => QuantizationPrecision::TwoBit,
            QuantizationPrecision::EightBit => QuantizationPrecision::FourBit,
        };

        if self.is_precision_in_bounds(next) {
            Some(next)
        } else {
            None
        }
    }

    /// Apply precision adjustment
    fn apply_precision_adjustment(
        &mut self,
        new_precision: QuantizationPrecision,
        reason: AdjustmentReason,
    ) -> QuantizationResult<PrecisionAdjustment> {
        let old_precision = self.current_state.precision;
        let timestamp = Instant::now();

        // Update current state
        self.current_state.precision = new_precision;
        self.current_state.last_update = timestamp;

        // Create adjustment record
        let adjustment = PrecisionAdjustment {
            timestamp,
            from_precision: old_precision,
            to_precision: new_precision,
            reason,
            performance_impact: PerformanceImpact {
                error_delta: 0.0, // Will be updated after measurement
                compression_delta: 0.0,
                time_delta: Duration::new(0, 0),
                memory_delta: 0,
                impact_score: 0.0,
            },
            success: true,
        };

        // Record adjustment
        self.adjustment_history.push(adjustment.clone());

        Ok(adjustment)
    }

    /// Record metrics for monitoring
    pub fn record_metrics(&mut self, metrics: &QuantizationStats, processing_time: Duration) {
        let now = Instant::now();

        // Record in history
        self.metrics_history
            .quantization_errors
            .push((now, metrics.quantization_error));
        self.metrics_history
            .compression_ratios
            .push((now, metrics.compression_ratio));
        self.metrics_history
            .processing_times
            .push((now, processing_time));

        // Update performance monitor
        self.performance_monitor.operations_count += 1;
        self.performance_monitor.total_processing_time += processing_time;

        // Trim history if needed
        self.trim_history();

        // Update current metrics
        self.performance_monitor.current_metrics.insert(
            PrecisionMetric::QuantizationError,
            metrics.quantization_error,
        );
        self.performance_monitor
            .current_metrics
            .insert(PrecisionMetric::CompressionRatio, metrics.compression_ratio);
        self.performance_monitor.current_metrics.insert(
            PrecisionMetric::ProcessingTime,
            processing_time.as_secs_f32() * 1000.0, // Convert to ms
        );
    }

    /// Trim history to maintain size limits
    fn trim_history(&mut self) {
        let max_size = self.config.monitoring.history_size;

        if self.metrics_history.quantization_errors.len() > max_size {
            self.metrics_history
                .quantization_errors
                .drain(0..self.metrics_history.quantization_errors.len() - max_size);
        }
        if self.metrics_history.compression_ratios.len() > max_size {
            self.metrics_history
                .compression_ratios
                .drain(0..self.metrics_history.compression_ratios.len() - max_size);
        }
        if self.metrics_history.processing_times.len() > max_size {
            self.metrics_history
                .processing_times
                .drain(0..self.metrics_history.processing_times.len() - max_size);
        }
    }

    /// Get current precision state
    pub fn get_current_state(&self) -> &PrecisionState {
        &self.current_state
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> &MetricsHistory {
        &self.metrics_history
    }

    /// Get adjustment history
    pub fn get_adjustment_history(&self) -> &[PrecisionAdjustment] {
        &self.adjustment_history
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let avg_error = if !self.metrics_history.quantization_errors.is_empty() {
            self.metrics_history
                .quantization_errors
                .iter()
                .map(|(_, error)| *error)
                .sum::<f32>()
                / self.metrics_history.quantization_errors.len() as f32
        } else {
            0.0
        };

        let avg_compression = if !self.metrics_history.compression_ratios.is_empty() {
            self.metrics_history
                .compression_ratios
                .iter()
                .map(|(_, ratio)| *ratio)
                .sum::<f32>()
                / self.metrics_history.compression_ratios.len() as f32
        } else {
            0.0
        };

        let avg_processing_time = if !self.metrics_history.processing_times.is_empty() {
            let total_time: Duration = self
                .metrics_history
                .processing_times
                .iter()
                .map(|(_, time)| *time)
                .sum();
            total_time / self.metrics_history.processing_times.len() as u32
        } else {
            Duration::new(0, 0)
        };

        PerformanceSummary {
            operations_count: self.performance_monitor.operations_count,
            average_error: avg_error,
            average_compression_ratio: avg_compression,
            average_processing_time: avg_processing_time,
            total_adjustments: self.adjustment_history.len(),
            current_precision: self.current_state.precision,
            stability_score: self.current_state.stability_score,
            performance_score: self.current_state.performance_score,
        }
    }
}

/// Performance analysis results
#[derive(Debug, Clone)]
struct PerformanceAnalysis {
    error_score: f32,
    compression_score: f32,
    stability_score: f32,
    overall_score: f32,
    needs_adjustment: bool,
}

impl PerformanceAnalysis {
    /// Get the overall score based on all metrics
    pub fn get_overall_score(&self) -> f32 {
        self.overall_score
    }

    /// Get the stability score
    pub fn get_stability_score(&self) -> f32 {
        self.stability_score
    }
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub operations_count: u64,
    pub average_error: f32,
    pub average_compression_ratio: f32,
    pub average_processing_time: Duration,
    pub total_adjustments: usize,
    pub current_precision: QuantizationPrecision,
    pub stability_score: f32,
    pub performance_score: f32,
}

impl PrecisionControlConfig {
    /// Validate the precision control configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        // Validate precision bounds
        self.precision_bounds.validate()?;

        // Validate dynamic adjustment config
        self.dynamic_adjustment.validate()?;

        // Validate monitoring config
        self.monitoring.validate()?;

        // Validate performance thresholds
        self.performance_thresholds.validate()?;

        Ok(())
    }

    /// Create configuration for conservative precision control
    pub fn conservative() -> Self {
        Self {
            target_precision: QuantizationPrecision::OneFiveFiveBit,
            precision_bounds: PrecisionBounds {
                min_precision: QuantizationPrecision::OneFiveFiveBit,
                max_precision: QuantizationPrecision::EightBit,
                max_error_tolerance: 0.05,
                min_compression_ratio: 2.0,
                ..Default::default()
            },
            dynamic_adjustment: DynamicAdjustmentConfig {
                strategy: AdjustmentStrategy::Conservative,
                learning_rate: 0.05,
                stability_threshold: 0.02,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create configuration for aggressive precision control
    pub fn aggressive() -> Self {
        Self {
            target_precision: QuantizationPrecision::OneBit,
            precision_bounds: PrecisionBounds {
                min_precision: QuantizationPrecision::OneBit,
                max_precision: QuantizationPrecision::FourBit,
                max_error_tolerance: 0.2,
                min_compression_ratio: 4.0,
                ..Default::default()
            },
            dynamic_adjustment: DynamicAdjustmentConfig {
                strategy: AdjustmentStrategy::Aggressive,
                learning_rate: 0.2,
                stability_threshold: 0.05,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl PrecisionBounds {
    /// Validate precision bounds
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.min_threshold >= self.max_threshold {
            return Err(QuantizationError::ConfigurationError(
                "Min threshold must be less than max threshold".to_string(),
            ));
        }

        if self.min_scale >= self.max_scale {
            return Err(QuantizationError::ConfigurationError(
                "Min scale must be less than max scale".to_string(),
            ));
        }

        if self.max_error_tolerance <= 0.0 {
            return Err(QuantizationError::ConfigurationError(
                "Max error tolerance must be positive".to_string(),
            ));
        }

        if self.min_compression_ratio <= 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Min compression ratio must be greater than 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

impl DynamicAdjustmentConfig {
    /// Validate dynamic adjustment configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.evaluation_window == 0 {
            return Err(QuantizationError::ConfigurationError(
                "Evaluation window must be greater than 0".to_string(),
            ));
        }

        if self.adjustment_frequency == 0 {
            return Err(QuantizationError::ConfigurationError(
                "Adjustment frequency must be greater than 0".to_string(),
            ));
        }

        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Learning rate must be between 0 and 1".to_string(),
            ));
        }

        if self.stability_threshold < 0.0 {
            return Err(QuantizationError::ConfigurationError(
                "Stability threshold must be non-negative".to_string(),
            ));
        }

        Ok(())
    }
}

impl PrecisionMonitoringConfig {
    /// Validate monitoring configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.history_size == 0 {
            return Err(QuantizationError::ConfigurationError(
                "History size must be greater than 0".to_string(),
            ));
        }

        if self.tracked_metrics.is_empty() {
            return Err(QuantizationError::ConfigurationError(
                "At least one metric must be tracked".to_string(),
            ));
        }

        Ok(())
    }
}

impl PerformanceThresholds {
    /// Validate performance thresholds
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.min_accuracy <= 0.0 || self.min_accuracy > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Min accuracy must be between 0 and 1".to_string(),
            ));
        }

        if self.max_latency_ms <= 0.0 {
            return Err(QuantizationError::ConfigurationError(
                "Max latency must be positive".to_string(),
            ));
        }

        if self.max_memory_overhead_pct < 0.0 {
            return Err(QuantizationError::ConfigurationError(
                "Max memory overhead must be non-negative".to_string(),
            ));
        }

        if self.min_throughput <= 0.0 {
            return Err(QuantizationError::ConfigurationError(
                "Min throughput must be positive".to_string(),
            ));
        }

        if self.min_energy_efficiency <= 0.0 || self.min_energy_efficiency > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Min energy efficiency must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

/// Factory functions for creating precision controllers
pub fn create_precision_controller(
    config: PrecisionControlConfig,
    device: Device,
) -> QuantizationResult<PrecisionController> {
    PrecisionController::new(config, device)
}

/// Create a conservative precision controller
pub fn create_conservative_precision_controller(
    device: Device,
) -> QuantizationResult<PrecisionController> {
    let config = PrecisionControlConfig::conservative();
    PrecisionController::new(config, device)
}

/// Create an aggressive precision controller
pub fn create_aggressive_precision_controller(
    device: Device,
) -> QuantizationResult<PrecisionController> {
    let config = PrecisionControlConfig::aggressive();
    PrecisionController::new(config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_precision_bounds_validation() {
        let mut bounds = PrecisionBounds::default();
        assert!(bounds.validate().is_ok());

        // Test invalid threshold bounds
        bounds.min_threshold = 5.0;
        bounds.max_threshold = 2.0;
        assert!(bounds.validate().is_err());
    }

    #[test]
    fn test_precision_controller_creation() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let controller = PrecisionController::new(config, device);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_precision_bounds_checking() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let controller = PrecisionController::new(config, device).unwrap();

        assert!(controller.is_precision_in_bounds(QuantizationPrecision::OneFiveFiveBit));
        assert!(controller.is_precision_in_bounds(QuantizationPrecision::EightBit));
    }

    #[test]
    fn test_precision_validation() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let controller = PrecisionController::new(config, device).unwrap();

        let result =
            controller.validate_precision_bounds(QuantizationPrecision::OneFiveFiveBit, 0.5, 1.0);
        assert!(result.is_ok());

        // Test invalid threshold
        let result = controller.validate_precision_bounds(
            QuantizationPrecision::OneFiveFiveBit,
            100.0, // Too high
            1.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conservative_config() {
        let config = PrecisionControlConfig::conservative();
        assert!(config.validate().is_ok());
        assert_eq!(
            config.dynamic_adjustment.strategy,
            AdjustmentStrategy::Conservative
        );
    }

    #[test]
    fn test_aggressive_config() {
        let config = PrecisionControlConfig::aggressive();
        assert!(config.validate().is_ok());
        assert_eq!(
            config.dynamic_adjustment.strategy,
            AdjustmentStrategy::Aggressive
        );
    }

    #[test]
    fn test_metrics_recording() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        let stats = QuantizationStats {
            elements_count: 1000,
            quantization_error: 0.05,
            compression_ratio: 4.0,
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };

        controller.record_metrics(&stats, Duration::from_millis(10));

        let summary = controller.get_performance_summary();
        assert_eq!(summary.operations_count, 1);
        assert_eq!(summary.average_error, 0.05);
    }

    #[test]
    fn test_precision_adjustment() {
        let config = PrecisionControlConfig::default();
        let device = Device::Cpu;
        let mut controller = PrecisionController::new(config, device).unwrap();

        let stats = QuantizationStats {
            elements_count: 1000,
            quantization_error: 0.15, // High error to trigger adjustment
            compression_ratio: 2.0,
            min_value: -1.0,
            max_value: 1.0,
            scale_factor: 1.0,
            zero_point: None,
        };

        let adjustment = controller.adjust_precision_dynamically(&stats).unwrap();
        // Should suggest precision increase due to high error
        if let Some(adj) = adjustment {
            assert_eq!(adj.reason, AdjustmentReason::HighError);
        }
    }
}
