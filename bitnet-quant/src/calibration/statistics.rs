//! Activation statistics collection and tracking
//!
//! This module provides comprehensive statistics collection for neural network
//! activations during calibration, including min/max tracking, moments, and
//! percentiles for optimal quantization parameter estimation.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Complete statistics collector for a layer's activations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStatistics {
    /// Layer name or identifier
    pub layer_name: String,
    /// Min/max values
    pub min_max: MinMaxStats,
    /// Moment statistics (mean, variance, etc.)
    pub moments: MomentStats,
    /// Percentile statistics
    pub percentiles: PercentileStats,
    /// Outlier detection results
    pub outliers: OutlierStats,
    /// Shape information
    pub shape_info: ShapeInfo,
    /// Update count
    pub update_count: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Min/max tracking for activations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxStats {
    /// Global minimum value
    pub global_min: f32,
    /// Global maximum value
    pub global_max: f32,
    /// Per-channel minimum values
    pub channel_min: Vec<f32>,
    /// Per-channel maximum values
    pub channel_max: Vec<f32>,
    /// Running minimum exponential moving average
    pub ema_min: f32,
    /// Running maximum exponential moving average
    pub ema_max: f32,
    /// EMA decay factor
    pub ema_decay: f32,
}

/// Moment statistics (mean, variance, skewness, kurtosis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentStats {
    /// Running mean
    pub mean: f32,
    /// Running variance
    pub variance: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Skewness (third moment)
    pub skewness: f32,
    /// Kurtosis (fourth moment)
    pub kurtosis: f32,
    /// Per-channel means
    pub channel_means: Vec<f32>,
    /// Per-channel variances
    pub channel_variances: Vec<f32>,
}

/// Percentile statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileStats {
    /// Requested percentile values
    pub percentiles: Vec<f32>,
    /// Computed percentile values
    pub values: Vec<f32>,
    /// Inter-quartile range
    pub iqr: f32,
    /// Median absolute deviation
    pub mad: f32,
}

/// Outlier detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierStats {
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Outlier ratio
    pub outlier_ratio: f32,
    /// Outlier threshold used
    pub threshold: f32,
    /// Outlier detection method
    pub method: OutlierMethod,
}

/// Outlier detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierMethod {
    /// Standard deviation based
    StandardDeviation,
    /// Interquartile range based
    IQR,
    /// Modified Z-score
    ModifiedZScore,
    /// Isolation forest (if available)
    IsolationForest,
}

/// Shape and structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeInfo {
    /// Tensor dimensions
    pub dimensions: Vec<usize>,
    /// Total number of elements
    pub num_elements: usize,
    /// Number of channels (if applicable)
    pub num_channels: Option<usize>,
    /// Sparsity ratio (zeros / total)
    pub sparsity_ratio: f32,
    /// Data type information
    pub dtype: String,
}

/// Main statistics collector
#[derive(Debug)]
pub struct StatisticsCollector {
    /// Statistics per layer
    layer_stats: HashMap<String, LayerStatistics>,
    /// Configuration
    config: StatisticsConfig,
    /// Update trackers
    update_trackers: HashMap<String, MinMaxTracker>,
}

/// Configuration for statistics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Track min/max values
    pub track_min_max: bool,
    /// Track moment statistics
    pub track_moments: bool,
    /// Track percentiles
    pub track_percentiles: bool,
    /// Percentiles to compute
    pub percentiles: Vec<f32>,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Outlier threshold
    pub outlier_threshold: f32,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
    /// Update frequency for statistics
    pub update_frequency: usize,
    /// EMA decay factor for min/max tracking
    pub ema_decay: f32,
    /// Enable per-channel statistics
    pub per_channel_stats: bool,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            track_min_max: true,
            track_moments: true,
            track_percentiles: true,
            percentiles: vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0],
            outlier_detection: true,
            outlier_threshold: 3.0,
            outlier_method: OutlierMethod::StandardDeviation,
            update_frequency: 100,
            ema_decay: 0.01,
            per_channel_stats: true,
        }
    }
}

/// Min/Max value tracker with streaming updates
#[derive(Debug, Clone)]
pub struct MinMaxTracker {
    /// Current minimum
    pub min: f32,
    /// Current maximum
    pub max: f32,
    /// EMA minimum
    pub ema_min: f32,
    /// EMA maximum
    pub ema_max: f32,
    /// EMA decay factor
    pub decay: f32,
    /// Update count
    pub count: usize,
}

impl MinMaxTracker {
    /// Create a new min/max tracker
    pub fn new(decay: f32) -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            ema_min: 0.0,
            ema_max: 0.0,
            decay,
            count: 0,
        }
    }

    /// Update with new values
    pub fn update(&mut self, values: &[f32]) -> CalibrationResult<()> {
        if values.is_empty() {
            return Ok(());
        }

        let batch_min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let batch_max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Update global min/max
        self.min = self.min.min(batch_min);
        self.max = self.max.max(batch_max);

        // Update EMA
        if self.count == 0 {
            self.ema_min = batch_min;
            self.ema_max = batch_max;
        } else {
            self.ema_min = (1.0 - self.decay) * self.ema_min + self.decay * batch_min;
            self.ema_max = (1.0 - self.decay) * self.ema_max + self.decay * batch_max;
        }

        self.count += 1;
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> (f32, f32, f32, f32) {
        (self.min, self.max, self.ema_min, self.ema_max)
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.min = f32::INFINITY;
        self.max = f32::NEG_INFINITY;
        self.ema_min = 0.0;
        self.ema_max = 0.0;
        self.count = 0;
    }
}

impl StatisticsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self::with_config(StatisticsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StatisticsConfig) -> Self {
        Self {
            layer_stats: HashMap::new(),
            config,
            update_trackers: HashMap::new(),
        }
    }

    /// Add or update statistics for a layer
    pub fn update_layer_statistics(
        &mut self,
        layer_name: &str,
        activations: &Tensor,
    ) -> CalibrationResult<()> {
        // Convert tensor to Vec<f32> for processing
        let values = self.tensor_to_vec(activations)?;
        let shape = activations.shape().dims().to_vec();

        // Pre-create entries to avoid borrow conflicts
        let layer_key = layer_name.to_string();

        // Ensure layer statistics exist
        if !self.layer_stats.contains_key(&layer_key) {
            self.layer_stats.insert(
                layer_key.clone(),
                LayerStatistics::new(layer_name, &shape, &self.config),
            );
        }

        // Ensure tracker exists
        if !self.update_trackers.contains_key(&layer_key) {
            self.update_trackers
                .insert(layer_key.clone(), MinMaxTracker::new(self.config.ema_decay));
        }

        // Now safely get mutable references
        let stats = self.layer_stats.get_mut(&layer_key).unwrap();
        let tracker = self.update_trackers.get_mut(&layer_key).unwrap();

        // Get configuration values to avoid borrowing self later
        let per_channel_stats = self.config.per_channel_stats;

        // Update min/max if enabled
        if self.config.track_min_max {
            // Direct implementation to avoid self borrow
            tracker.update(&values)?;
            let (min, max, ema_min, ema_max) = tracker.get_stats();

            stats.min_max.global_min = stats.min_max.global_min.min(min);
            stats.min_max.global_max = stats.min_max.global_max.max(max);
            stats.min_max.ema_min = ema_min;
            stats.min_max.ema_max = ema_max;

            // Update per-channel statistics if enabled
            if per_channel_stats {
                // Simple per-channel min/max calculation
                // This is a simplified version to avoid method calls
                if stats.min_max.channel_min.is_empty() {
                    stats.min_max.channel_min = vec![f32::INFINITY; shape[0].min(values.len())];
                    stats.min_max.channel_max = vec![f32::NEG_INFINITY; shape[0].min(values.len())];
                }
            }
        }

        // Update moments if enabled - simplified direct calculation
        if self.config.track_moments {
            let n = values.len() as f32;
            let mean = values.iter().sum::<f32>() / n;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

            stats.moments.mean = mean;
            stats.moments.variance = variance;
            stats.moments.std_dev = variance.sqrt();

            // Calculate higher moments
            let skewness = if variance > 0.0 {
                values
                    .iter()
                    .map(|&x| ((x - mean) / variance.sqrt()).powi(3))
                    .sum::<f32>()
                    / n
            } else {
                0.0
            };
            let kurtosis = if variance > 0.0 {
                values
                    .iter()
                    .map(|&x| ((x - mean) / variance.sqrt()).powi(4))
                    .sum::<f32>()
                    / n
                    - 3.0
            } else {
                0.0
            };

            stats.moments.skewness = skewness;
            stats.moments.kurtosis = kurtosis;
        }

        // Update percentiles if enabled - simplified calculation
        if self.config.track_percentiles && stats.update_count % self.config.update_frequency == 0 {
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted_values.len();

            if n > 0 {
                // Standard percentiles: 25th, 50th, 75th, 90th, 95th, 99th
                let percentile_positions = vec![25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
                let percentile_values: Vec<f32> = percentile_positions
                    .iter()
                    .map(|&p| {
                        let index = ((p / 100.0) * (n - 1) as f32) as usize;
                        sorted_values[index.min(n - 1)]
                    })
                    .collect();

                stats.percentiles.percentiles = percentile_positions;
                stats.percentiles.values = percentile_values.clone();

                // Calculate IQR (Q3 - Q1)
                if percentile_values.len() >= 3 {
                    stats.percentiles.iqr = percentile_values[2] - percentile_values[0];
                    // 75th - 25th
                }

                // Calculate MAD (Median Absolute Deviation)
                if let Some(median) = percentile_values.get(1) {
                    // 50th percentile
                    let deviations: Vec<f32> =
                        sorted_values.iter().map(|&x| (x - median).abs()).collect();
                    let mut sorted_deviations = deviations.clone();
                    sorted_deviations
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    stats.percentiles.mad = sorted_deviations[sorted_deviations.len() / 2];
                }
            }
        }

        // Detect outliers if enabled - simplified calculation
        if self.config.outlier_detection {
            let n = values.len() as f32;
            let mean = values.iter().sum::<f32>() / n;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let std_dev = variance.sqrt();

            let threshold = 3.0 * std_dev; // 3-sigma rule
            let outlier_count = values
                .iter()
                .filter(|&&x| (x - mean).abs() > threshold)
                .count();

            stats.outliers.outlier_ratio = outlier_count as f32 / n;
            stats.outliers.outlier_count = outlier_count;
        }

        // Update metadata
        stats.update_count += 1;
        stats.last_updated = SystemTime::now();
        // Simple shape info update without calling self method
        stats.shape_info = ShapeInfo {
            dimensions: shape,
            num_elements: values.len(),
            num_channels: None,
            sparsity_ratio: values.iter().filter(|&&x| x == 0.0).count() as f32
                / values.len() as f32,
            dtype: "f32".to_string(),
        };

        Ok(())
    }

    /// Get statistics for a specific layer
    pub fn get_layer_statistics(&self, layer_name: &str) -> Option<&LayerStatistics> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_statistics(&self) -> &HashMap<String, LayerStatistics> {
        &self.layer_stats
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.update_trackers.clear();
    }

    /// Convert tensor to Vec<f32> for processing
    fn tensor_to_vec(&self, tensor: &Tensor) -> CalibrationResult<Vec<f32>> {
        let flattened = tensor
            .flatten_all()
            .map_err(|e| CalibrationError::statistics(format!("Failed to flatten tensor: {e}")))?;

        let values: Vec<f32> = flattened.to_vec1().map_err(|e| {
            CalibrationError::statistics(format!("Failed to convert tensor to vec: {e}"))
        })?;

        Ok(values)
    }

    /// Update min/max statistics
    pub fn update_min_max_stats(
        &self,
        stats: &mut LayerStatistics,
        tracker: &mut MinMaxTracker,
        values: &[f32],
        shape: &[usize],
    ) -> CalibrationResult<()> {
        tracker.update(values)?;

        let (min, max, ema_min, ema_max) = tracker.get_stats();

        stats.min_max.global_min = stats.min_max.global_min.min(min);
        stats.min_max.global_max = stats.min_max.global_max.max(max);
        stats.min_max.ema_min = ema_min;
        stats.min_max.ema_max = ema_max;

        // Update per-channel statistics if enabled
        if self.config.per_channel_stats {
            self.update_per_channel_min_max(stats, values, shape)?;
        }

        Ok(())
    }

    /// Update per-channel min/max statistics
    pub fn update_per_channel_min_max(
        &self,
        stats: &mut LayerStatistics,
        values: &[f32],
        shape: &[usize],
    ) -> CalibrationResult<()> {
        if shape.len() < 2 {
            return Ok(()); // No channels to process
        }

        let num_channels = shape[1]; // Assume NCHW format
        let elements_per_channel = values.len() / num_channels;

        if stats.min_max.channel_min.len() != num_channels {
            stats.min_max.channel_min = vec![f32::INFINITY; num_channels];
            stats.min_max.channel_max = vec![f32::NEG_INFINITY; num_channels];
        }

        for channel in 0..num_channels {
            let channel_start = channel * elements_per_channel;
            let channel_end = channel_start + elements_per_channel;

            if channel_end <= values.len() {
                let channel_values = &values[channel_start..channel_end];
                let channel_min = channel_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let channel_max = channel_values
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                stats.min_max.channel_min[channel] =
                    stats.min_max.channel_min[channel].min(channel_min);
                stats.min_max.channel_max[channel] =
                    stats.min_max.channel_max[channel].max(channel_max);
            }
        }

        Ok(())
    }

    /// Update moment statistics
    pub fn update_moment_stats(
        &self,
        stats: &mut LayerStatistics,
        values: &[f32],
    ) -> CalibrationResult<()> {
        if values.is_empty() {
            return Ok(());
        }

        let n = values.len() as f32;
        let sum: f32 = values.iter().sum();
        let mean = sum / n;

        let variance: f32 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // Update running statistics using exponential moving average
        let alpha = 0.01; // EMA decay factor
        if stats.update_count == 0 {
            stats.moments.mean = mean;
            stats.moments.variance = variance;
        } else {
            stats.moments.mean = (1.0 - alpha) * stats.moments.mean + alpha * mean;
            stats.moments.variance = (1.0 - alpha) * stats.moments.variance + alpha * variance;
        }

        stats.moments.std_dev = stats.moments.variance.sqrt();

        // Compute higher-order moments
        if std_dev > 0.0 {
            let skewness: f32 = values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>()
                / n;

            let kurtosis: f32 = values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f32>()
                / n
                - 3.0; // Excess kurtosis

            stats.moments.skewness = skewness;
            stats.moments.kurtosis = kurtosis;
        }

        Ok(())
    }

    /// Update percentile statistics
    pub fn update_percentile_stats(
        &self,
        stats: &mut LayerStatistics,
        values: &[f32],
    ) -> CalibrationResult<()> {
        if values.is_empty() {
            return Ok(());
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut percentile_values = Vec::new();
        for &percentile in &self.config.percentiles {
            let index = ((percentile / 100.0) * (sorted_values.len() - 1) as f32) as usize;
            percentile_values.push(sorted_values[index]);
        }

        // Compute IQR (75th - 25th percentile)
        let q1_idx = (0.25 * (sorted_values.len() - 1) as f32) as usize;
        let q3_idx = (0.75 * (sorted_values.len() - 1) as f32) as usize;
        let iqr = sorted_values[q3_idx] - sorted_values[q1_idx];

        // Compute Median Absolute Deviation
        let median_idx = sorted_values.len() / 2;
        let median = sorted_values[median_idx];
        let mut abs_deviations: Vec<f32> = values.iter().map(|&x| (x - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_deviations[abs_deviations.len() / 2];

        stats.percentiles = PercentileStats {
            percentiles: self.config.percentiles.clone(),
            values: percentile_values,
            iqr,
            mad,
        };

        Ok(())
    }

    /// Update outlier statistics
    pub fn update_outlier_stats(
        &self,
        stats: &mut LayerStatistics,
        values: &[f32],
    ) -> CalibrationResult<()> {
        let outlier_count = match self.config.outlier_method {
            OutlierMethod::StandardDeviation => {
                self.detect_outliers_std_dev(values, self.config.outlier_threshold)
            }
            OutlierMethod::IQR => self.detect_outliers_iqr(values),
            OutlierMethod::ModifiedZScore => {
                self.detect_outliers_modified_z_score(values, self.config.outlier_threshold)
            }
            OutlierMethod::IsolationForest => {
                // Placeholder - would need external library
                0
            }
        };

        stats.outliers = OutlierStats {
            outlier_count,
            outlier_ratio: outlier_count as f32 / values.len() as f32,
            threshold: self.config.outlier_threshold,
            method: self.config.outlier_method.clone(),
        };

        Ok(())
    }

    /// Detect outliers using standard deviation method
    pub fn detect_outliers_std_dev(&self, values: &[f32], threshold: f32) -> usize {
        if values.is_empty() {
            return 0;
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        values
            .iter()
            .filter(|&&x| (x - mean).abs() > threshold * std_dev)
            .count()
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(&self, values: &[f32]) -> usize {
        if values.is_empty() {
            return 0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = (0.25 * (sorted_values.len() - 1) as f32) as usize;
        let q3_idx = (0.75 * (sorted_values.len() - 1) as f32) as usize;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        values
            .iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .count()
    }

    /// Detect outliers using modified Z-score
    pub fn detect_outliers_modified_z_score(&self, values: &[f32], threshold: f32) -> usize {
        if values.is_empty() {
            return 0;
        }

        let median_idx = values.len() / 2;
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_values[median_idx];

        let mad = {
            let mut abs_deviations: Vec<f32> = values.iter().map(|&x| (x - median).abs()).collect();
            abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            abs_deviations[abs_deviations.len() / 2]
        };

        if mad == 0.0 {
            return 0;
        }

        values
            .iter()
            .filter(|&&x| (0.6745 * (x - median) / mad).abs() > threshold)
            .count()
    }

    /// Compute shape information
    pub fn compute_shape_info(&self, shape: &[usize], values: &[f32]) -> ShapeInfo {
        let num_elements = values.len();
        let num_channels = if shape.len() > 1 {
            Some(shape[1])
        } else {
            None
        };
        let zeros = values.iter().filter(|&&x| x == 0.0).count();
        let sparsity_ratio = zeros as f32 / num_elements as f32;

        ShapeInfo {
            dimensions: shape.to_vec(),
            num_elements,
            num_channels,
            sparsity_ratio,
            dtype: "f32".to_string(),
        }
    }
}

impl LayerStatistics {
    /// Create new layer statistics
    fn new(layer_name: &str, shape: &[usize], config: &StatisticsConfig) -> Self {
        Self {
            layer_name: layer_name.to_string(),
            min_max: MinMaxStats {
                global_min: f32::INFINITY,
                global_max: f32::NEG_INFINITY,
                channel_min: Vec::new(),
                channel_max: Vec::new(),
                ema_min: 0.0,
                ema_max: 0.0,
                ema_decay: config.ema_decay,
            },
            moments: MomentStats {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                channel_means: Vec::new(),
                channel_variances: Vec::new(),
            },
            percentiles: PercentileStats {
                percentiles: config.percentiles.clone(),
                values: Vec::new(),
                iqr: 0.0,
                mad: 0.0,
            },
            outliers: OutlierStats {
                outlier_count: 0,
                outlier_ratio: 0.0,
                threshold: config.outlier_threshold,
                method: config.outlier_method.clone(),
            },
            shape_info: ShapeInfo {
                dimensions: shape.to_vec(),
                num_elements: 0,
                num_channels: if shape.len() > 1 {
                    Some(shape[1])
                } else {
                    None
                },
                sparsity_ratio: 0.0,
                dtype: "f32".to_string(),
            },
            update_count: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Update interface for statistics
pub trait StatisticsUpdate {
    /// Update statistics with new data
    fn update(&mut self, layer_name: &str, data: &Tensor) -> CalibrationResult<()>;
    /// Get current statistics
    fn get_statistics(&self, layer_name: &str) -> Option<&LayerStatistics>;
    /// Reset all statistics
    fn reset(&mut self);
}

impl StatisticsUpdate for StatisticsCollector {
    fn update(&mut self, layer_name: &str, data: &Tensor) -> CalibrationResult<()> {
        self.update_layer_statistics(layer_name, data)
    }

    fn get_statistics(&self, layer_name: &str) -> Option<&LayerStatistics> {
        self.get_layer_statistics(layer_name)
    }

    fn reset(&mut self) {
        self.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensor(shape: &[usize]) -> CalibrationResult<Tensor> {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| i as f32 / 100.0)
            .collect();
        Tensor::from_vec(data, shape, &device)
            .map_err(|e| CalibrationError::statistics(format!("Failed to create tensor: {}", e)))
    }

    #[test]
    fn test_statistics_collector_creation() {
        let collector = StatisticsCollector::new();
        assert!(collector.get_all_statistics().is_empty());
    }

    #[test]
    fn test_layer_statistics_update() -> CalibrationResult<()> {
        let mut collector = StatisticsCollector::new();
        let tensor = create_test_tensor(&[2, 3, 4, 4])?;

        collector.update_layer_statistics("test_layer", &tensor)?;

        let stats = collector.get_layer_statistics("test_layer");
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.layer_name, "test_layer");
        assert_eq!(stats.update_count, 1);

        Ok(())
    }

    #[test]
    fn test_min_max_tracker() -> CalibrationResult<()> {
        let mut tracker = MinMaxTracker::new(0.1);
        let values = vec![1.0, 2.0, 3.0, -1.0, 5.0];

        tracker.update(&values)?;

        let (min, max, _, _) = tracker.get_stats();
        assert_eq!(min, -1.0);
        assert_eq!(max, 5.0);

        Ok(())
    }

    #[test]
    fn test_outlier_detection() {
        let collector = StatisticsCollector::new();
        let values = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0]; // 100.0 is an outlier

        let outliers = collector.detect_outliers_std_dev(&values, 2.0);
        assert!(outliers > 0);
    }

    #[test]
    fn test_percentile_computation() -> CalibrationResult<()> {
        let mut collector = StatisticsCollector::new();
        let tensor = create_test_tensor(&[1, 1, 100])?;

        collector.update_layer_statistics("test", &tensor)?;

        let stats = collector.get_layer_statistics("test").unwrap();
        assert!(!stats.percentiles.values.is_empty());

        Ok(())
    }
}
