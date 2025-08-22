//! Histogram collection for quantization optimization
//!
//! This module provides histogram-based analysis of activation distributions
//! to optimize quantization parameters and improve quantization quality.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use crate::calibration::config::{HistogramConfig, HistogramRangeStrategy};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Histogram collector for activation distributions
#[derive(Debug)]
pub struct HistogramCollector {
    /// Configuration
    config: HistogramConfig,
    /// Histograms per layer
    histograms: HashMap<String, ActivationHistogram>,
    /// Statistics for range determination
    range_stats: HashMap<String, RangeStatistics>,
}

/// Histogram for activation values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHistogram {
    /// Histogram bins
    pub bins: Vec<HistogramBin>,
    /// Total number of values processed
    pub total_count: usize,
    /// Value range covered
    pub value_range: (f32, f32),
    /// Bin width
    pub bin_width: f32,
    /// Overflow count (values above range)
    pub overflow_count: usize,
    /// Underflow count (values below range)
    pub underflow_count: usize,
    /// Update count
    pub update_count: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Individual histogram bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    /// Bin lower bound
    pub lower_bound: f32,
    /// Bin upper bound
    pub upper_bound: f32,
    /// Number of values in this bin
    pub count: usize,
    /// Density (count / total_count)
    pub density: f32,
    /// Bin center value
    pub center: f32,
}

/// Statistics for determining histogram ranges
#[derive(Debug, Clone)]
struct RangeStatistics {
    /// Minimum value seen
    min_value: f32,
    /// Maximum value seen
    max_value: f32,
    /// Running percentile estimator
    percentile_estimator: PercentileEstimator,
    /// Sample count
    sample_count: usize,
}

/// Percentile estimator for streaming percentile calculation
#[derive(Debug, Clone)]
struct PercentileEstimator {
    /// Sorted sample buffer for estimation
    sample_buffer: Vec<f32>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Sampling rate for large datasets
    sampling_rate: f32,
}

/// Quantization optimizer using histogram analysis
pub struct QuantizationOptimizer {
    /// Histogram data
    histogram: ActivationHistogram,
    /// Optimization strategy
    strategy: OptimizationStrategy,
}

/// Strategies for quantization parameter optimization
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Minimize mean squared error
    MinimizeMSE,
    /// Minimize KL divergence
    MinimizeKLDivergence,
    /// Entropy-based optimization
    EntropyBased,
    /// Percentile-based clipping
    PercentileClipping { lower: f32, upper: f32 },
}

/// Quantization parameters derived from histogram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedQuantizationParams {
    /// Optimal scaling factor
    pub scale: f32,
    /// Optimal zero point
    pub zero_point: i32,
    /// Optimal clipping range
    pub clip_range: (f32, f32),
    /// Quantization error estimate
    pub error_estimate: f32,
    /// Confidence in the parameters
    pub confidence: f32,
    /// Optimization strategy used
    pub strategy: String,
}

impl HistogramCollector {
    /// Create a new histogram collector
    pub fn new(config: HistogramConfig) -> Self {
        Self {
            config,
            histograms: HashMap::new(),
            range_stats: HashMap::new(),
        }
    }

    /// Update histogram with new activation data
    pub fn update_histogram(
        &mut self,
        layer_name: &str,
        activations: &Tensor,
    ) -> CalibrationResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Convert tensor to values
        let values = self.tensor_to_vec(activations)?;
        
        // Update range statistics
        self.update_range_stats(layer_name, &values);
        
        // Get or create histogram
        let range = self.determine_histogram_range(layer_name, &values)?;
        let histogram = self.histograms.entry(layer_name.to_string()).or_insert_with(|| {
            ActivationHistogram::new(self.config.num_bins, range, SystemTime::now())
        });
        
        // Update histogram with values
        histogram.update_with_values(&values);
        
        Ok(())
    }

    /// Get histogram for a layer
    pub fn get_histogram(&self, layer_name: &str) -> Option<&ActivationHistogram> {
        self.histograms.get(layer_name)
    }

    /// Get all histograms
    pub fn get_all_histograms(&self) -> &HashMap<String, ActivationHistogram> {
        &self.histograms
    }

    /// Create quantization optimizer for a layer
    pub fn create_optimizer(
        &self,
        layer_name: &str,
        strategy: OptimizationStrategy,
    ) -> CalibrationResult<QuantizationOptimizer> {
        let histogram = self.get_histogram(layer_name)
            .ok_or_else(|| CalibrationError::histogram(format!("No histogram for layer: {layer_name}")))?;
        
        Ok(QuantizationOptimizer::new(histogram.clone(), strategy))
    }

    /// Reset all histograms
    pub fn reset(&mut self) {
        self.histograms.clear();
        self.range_stats.clear();
    }

    /// Convert tensor to Vec<f32>
    fn tensor_to_vec(&self, tensor: &Tensor) -> CalibrationResult<Vec<f32>> {
        let flattened = tensor.flatten_all()
            .map_err(|e| CalibrationError::histogram(format!("Failed to flatten tensor: {e}")))?;
        
        let values: Vec<f32> = flattened.to_vec1()
            .map_err(|e| CalibrationError::histogram(format!("Failed to convert tensor: {e}")))?;
        
        Ok(values)
    }

    /// Update range statistics for a layer
    fn update_range_stats(&mut self, layer_name: &str, values: &[f32]) {
        let stats = self.range_stats.entry(layer_name.to_string()).or_insert_with(|| {
            RangeStatistics::new()
        });
        
        for &value in values {
            if value.is_finite() {
                stats.min_value = stats.min_value.min(value);
                stats.max_value = stats.max_value.max(value);
                stats.percentile_estimator.add_sample(value);
                stats.sample_count += 1;
            }
        }
    }

    /// Determine histogram range based on configuration
    fn determine_histogram_range(&self, layer_name: &str, values: &[f32]) -> CalibrationResult<(f32, f32)> {
        match &self.config.range_strategy {
            HistogramRangeStrategy::MinMax => {
                let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                Ok((min_val, max_val))
            },
            HistogramRangeStrategy::Percentile { lower, upper } => {
                let stats = self.range_stats.get(layer_name)
                    .ok_or_else(|| CalibrationError::histogram("Range statistics not available"))?;
                
                let lower_val = stats.percentile_estimator.estimate_percentile(*lower);
                let upper_val = stats.percentile_estimator.estimate_percentile(*upper);
                Ok((lower_val, upper_val))
            },
            HistogramRangeStrategy::Custom => {
                self.config.custom_range.ok_or_else(|| {
                    CalibrationError::histogram("Custom range not specified in configuration")
                })
            },
            HistogramRangeStrategy::Adaptive => {
                // Use a combination of min/max and percentiles for robustness
                let stats = self.range_stats.get(layer_name)
                    .ok_or_else(|| CalibrationError::histogram("Range statistics not available"))?;
                
                let percentile_range = (
                    stats.percentile_estimator.estimate_percentile(0.1),
                    stats.percentile_estimator.estimate_percentile(99.9),
                );
                
                let min_max_range = (stats.min_value, stats.max_value);
                
                // Use percentile range if it's reasonable, otherwise fall back to min/max
                let range_width = percentile_range.1 - percentile_range.0;
                let total_width = min_max_range.1 - min_max_range.0;
                
                if range_width > 0.0 && range_width < total_width * 2.0 {
                    Ok(percentile_range)
                } else {
                    Ok(min_max_range)
                }
            },
        }
    }
}

impl ActivationHistogram {
    /// Create a new activation histogram
    fn new(num_bins: usize, value_range: (f32, f32), created_at: SystemTime) -> Self {
        let bin_width = (value_range.1 - value_range.0) / num_bins as f32;
        let bins = (0..num_bins)
            .map(|i| {
                let lower_bound = value_range.0 + i as f32 * bin_width;
                let upper_bound = lower_bound + bin_width;
                HistogramBin {
                    lower_bound,
                    upper_bound,
                    count: 0,
                    density: 0.0,
                    center: (lower_bound + upper_bound) / 2.0,
                }
            })
            .collect();

        Self {
            bins,
            total_count: 0,
            value_range,
            bin_width,
            overflow_count: 0,
            underflow_count: 0,
            update_count: 0,
            last_updated: created_at,
        }
    }

    /// Update histogram with values
    fn update_with_values(&mut self, values: &[f32]) {
        for &value in values {
            if !value.is_finite() {
                continue;
            }

            if value < self.value_range.0 {
                self.underflow_count += 1;
            } else if value >= self.value_range.1 {
                self.overflow_count += 1;
            } else {
                // Find the appropriate bin
                let bin_index = ((value - self.value_range.0) / self.bin_width) as usize;
                let bin_index = bin_index.min(self.bins.len() - 1);
                self.bins[bin_index].count += 1;
            }
            
            self.total_count += 1;
        }

        // Update densities
        if self.total_count > 0 {
            for bin in &mut self.bins {
                bin.density = bin.count as f32 / self.total_count as f32;
            }
        }

        self.update_count += 1;
        self.last_updated = SystemTime::now();
    }

    /// Get bin containing a specific value
    pub fn get_bin_for_value(&self, value: f32) -> Option<&HistogramBin> {
        if value < self.value_range.0 || value >= self.value_range.1 {
            return None;
        }

        let bin_index = ((value - self.value_range.0) / self.bin_width) as usize;
        let bin_index = bin_index.min(self.bins.len() - 1);
        self.bins.get(bin_index)
    }

    /// Calculate cumulative distribution function
    pub fn calculate_cdf(&self) -> Vec<f32> {
        let mut cdf = Vec::with_capacity(self.bins.len());
        let mut cumulative = 0.0;
        
        for bin in &self.bins {
            cumulative += bin.density;
            cdf.push(cumulative);
        }
        
        cdf
    }

    /// Find optimal clipping range to minimize quantization error
    pub fn find_optimal_clip_range(&self, target_percentile: f32) -> (f32, f32) {
        let cdf = self.calculate_cdf();
        let lower_percentile = (1.0 - target_percentile) / 2.0;
        let upper_percentile = target_percentile + lower_percentile;

        let mut lower_bound = self.value_range.0;
        let mut upper_bound = self.value_range.1;

        // Find bounds based on CDF
        for (i, &cumulative) in cdf.iter().enumerate() {
            if cumulative >= lower_percentile && lower_bound == self.value_range.0 {
                lower_bound = self.bins[i].lower_bound;
            }
            if cumulative >= upper_percentile {
                upper_bound = self.bins[i].upper_bound;
                break;
            }
        }

        (lower_bound, upper_bound)
    }

    /// Calculate entropy of the distribution
    pub fn calculate_entropy(&self) -> f32 {
        self.bins
            .iter()
            .filter(|bin| bin.density > 0.0)
            .map(|bin| -bin.density * bin.density.log2())
            .sum()
    }
}

impl RangeStatistics {
    fn new() -> Self {
        Self {
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            percentile_estimator: PercentileEstimator::new(1000, 0.1),
            sample_count: 0,
        }
    }
}

impl PercentileEstimator {
    fn new(max_buffer_size: usize, sampling_rate: f32) -> Self {
        Self {
            sample_buffer: Vec::new(),
            max_buffer_size,
            sampling_rate,
        }
    }

    fn add_sample(&mut self, value: f32) {
        if self.sample_buffer.len() < self.max_buffer_size {
            self.sample_buffer.push(value);
        } else {
            // Random sampling to maintain buffer size
            if rand::random::<f32>() < self.sampling_rate {
                let index = rand::random::<usize>() % self.sample_buffer.len();
                self.sample_buffer[index] = value;
            }
        }
    }

    fn estimate_percentile(&self, percentile: f32) -> f32 {
        if self.sample_buffer.is_empty() {
            return 0.0;
        }

        let mut sorted_buffer = self.sample_buffer.clone();
        sorted_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile / 100.0) * (sorted_buffer.len() - 1) as f32) as usize;
        sorted_buffer[index]
    }
}

impl QuantizationOptimizer {
    /// Create a new quantization optimizer
    pub fn new(histogram: ActivationHistogram, strategy: OptimizationStrategy) -> Self {
        Self { histogram, strategy }
    }

    /// Optimize quantization parameters
    pub fn optimize(&self, bit_width: u8) -> CalibrationResult<OptimizedQuantizationParams> {
        match &self.strategy {
            OptimizationStrategy::MinimizeMSE => self.optimize_mse(bit_width),
            OptimizationStrategy::MinimizeKLDivergence => self.optimize_kl_divergence(bit_width),
            OptimizationStrategy::EntropyBased => self.optimize_entropy_based(bit_width),
            OptimizationStrategy::PercentileClipping { lower, upper } => {
                self.optimize_percentile_clipping(*lower, *upper, bit_width)
            },
        }
    }

    /// Optimize using MSE minimization
    fn optimize_mse(&self, bit_width: u8) -> CalibrationResult<OptimizedQuantizationParams> {
        let num_levels = (1 << bit_width) as f32;
        let clip_range = self.histogram.find_optimal_clip_range(0.99);
        
        let scale = (clip_range.1 - clip_range.0) / (num_levels - 1.0);
        let zero_point = (-clip_range.0 / scale).round() as i32;
        
        let error_estimate = self.estimate_quantization_error(scale, zero_point, clip_range);
        let confidence = self.calculate_confidence();
        
        Ok(OptimizedQuantizationParams {
            scale,
            zero_point,
            clip_range,
            error_estimate,
            confidence,
            strategy: "MinimizeMSE".to_string(),
        })
    }

    /// Optimize using KL divergence minimization
    fn optimize_kl_divergence(&self, bit_width: u8) -> CalibrationResult<OptimizedQuantizationParams> {
        // Simplified KL divergence optimization
        // In practice, this would involve iterative optimization
        let num_levels = (1 << bit_width) as f32;
        
        // Start with percentile-based clipping
        let mut best_clip_range = self.histogram.find_optimal_clip_range(0.99);
        let mut best_kl_div = f32::INFINITY;
        
        // Try different clipping percentiles
        for percentile in &[0.95, 0.97, 0.99, 0.999] {
            let clip_range = self.histogram.find_optimal_clip_range(*percentile);
            let kl_div = self.calculate_kl_divergence(clip_range, num_levels as usize);
            
            if kl_div < best_kl_div {
                best_kl_div = kl_div;
                best_clip_range = clip_range;
            }
        }
        
        let scale = (best_clip_range.1 - best_clip_range.0) / (num_levels - 1.0);
        let zero_point = (-best_clip_range.0 / scale).round() as i32;
        
        let error_estimate = best_kl_div;
        let confidence = self.calculate_confidence();
        
        Ok(OptimizedQuantizationParams {
            scale,
            zero_point,
            clip_range: best_clip_range,
            error_estimate,
            confidence,
            strategy: "MinimizeKLDivergence".to_string(),
        })
    }

    /// Optimize using entropy-based approach
    fn optimize_entropy_based(&self, bit_width: u8) -> CalibrationResult<OptimizedQuantizationParams> {
        let entropy = self.histogram.calculate_entropy();
        let num_levels = (1 << bit_width) as f32;
        
        // Use entropy to guide clipping
        let entropy_percentile = (entropy / 10.0).min(0.99).max(0.9); // Heuristic mapping
        let clip_range = self.histogram.find_optimal_clip_range(entropy_percentile);
        
        let scale = (clip_range.1 - clip_range.0) / (num_levels - 1.0);
        let zero_point = (-clip_range.0 / scale).round() as i32;
        
        let error_estimate = self.estimate_quantization_error(scale, zero_point, clip_range);
        let confidence = (entropy / 10.0).min(1.0); // Higher entropy = higher confidence
        
        Ok(OptimizedQuantizationParams {
            scale,
            zero_point,
            clip_range,
            error_estimate,
            confidence,
            strategy: "EntropyBased".to_string(),
        })
    }

    /// Optimize using percentile clipping
    fn optimize_percentile_clipping(
        &self,
        lower: f32,
        upper: f32,
        bit_width: u8,
    ) -> CalibrationResult<OptimizedQuantizationParams> {
        let cdf = self.histogram.calculate_cdf();
        let num_levels = (1 << bit_width) as f32;
        
        let mut lower_bound = self.histogram.value_range.0;
        let mut upper_bound = self.histogram.value_range.1;
        
        // Find bounds based on percentiles
        for (i, &cumulative) in cdf.iter().enumerate() {
            if cumulative >= lower / 100.0 && lower_bound == self.histogram.value_range.0 {
                lower_bound = self.histogram.bins[i].lower_bound;
            }
            if cumulative >= upper / 100.0 {
                upper_bound = self.histogram.bins[i].upper_bound;
                break;
            }
        }
        
        let clip_range = (lower_bound, upper_bound);
        let scale = (clip_range.1 - clip_range.0) / (num_levels - 1.0);
        let zero_point = (-clip_range.0 / scale).round() as i32;
        
        let error_estimate = self.estimate_quantization_error(scale, zero_point, clip_range);
        let confidence = self.calculate_confidence();
        
        Ok(OptimizedQuantizationParams {
            scale,
            zero_point,
            clip_range,
            error_estimate,
            confidence,
            strategy: format!("PercentileClipping({lower}, {upper})"),
        })
    }

    /// Estimate quantization error
    fn estimate_quantization_error(&self, scale: f32, zero_point: i32, clip_range: (f32, f32)) -> f32 {
        let mut mse = 0.0;
        let mut total_weight = 0.0;
        
        for bin in &self.histogram.bins {
            if bin.center >= clip_range.0 && bin.center <= clip_range.1 {
                // Quantize the bin center
                let quantized = ((bin.center / scale).round() + zero_point as f32) * scale;
                let error = (bin.center - quantized).powi(2);
                
                mse += error * bin.density;
                total_weight += bin.density;
            }
        }
        
        if total_weight > 0.0 {
            mse / total_weight
        } else {
            0.0
        }
    }

    /// Calculate KL divergence between original and quantized distributions
    fn calculate_kl_divergence(&self, clip_range: (f32, f32), num_levels: usize) -> f32 {
        let mut kl_div = 0.0;
        
        // Create quantized histogram
        let level_width = (clip_range.1 - clip_range.0) / num_levels as f32;
        let mut quantized_bins = vec![0.0; num_levels];
        
        // Map original bins to quantized levels
        for bin in &self.histogram.bins {
            if bin.center >= clip_range.0 && bin.center <= clip_range.1 {
                let level_index = ((bin.center - clip_range.0) / level_width) as usize;
                let level_index = level_index.min(num_levels - 1);
                quantized_bins[level_index] += bin.density;
            }
        }
        
        // Calculate KL divergence
        for bin in &self.histogram.bins {
            if bin.density > 0.0 && bin.center >= clip_range.0 && bin.center <= clip_range.1 {
                let level_index = ((bin.center - clip_range.0) / level_width) as usize;
                let level_index = level_index.min(num_levels - 1);
                let q_prob = quantized_bins[level_index];
                
                if q_prob > 0.0 {
                    kl_div += bin.density * (bin.density / q_prob).ln();
                }
            }
        }
        
        kl_div
    }

    /// Calculate confidence in the optimization
    fn calculate_confidence(&self) -> f32 {
        // Simple heuristic based on sample count and distribution spread
        let sample_confidence = (self.histogram.total_count as f32 / 10000.0).min(1.0);
        let distribution_confidence = 1.0 - (self.histogram.overflow_count + self.histogram.underflow_count) as f32 / self.histogram.total_count as f32;
        
        (sample_confidence + distribution_confidence) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_tensor(data: Vec<f32>, shape: &[usize]) -> CalibrationResult<Tensor> {
        Tensor::from_vec(data, shape, &Device::Cpu)
            .map_err(|e| CalibrationError::histogram(format!("Failed to create tensor: {}", e)))
    }

    #[test]
    fn test_histogram_collector_creation() {
        let config = HistogramConfig::default();
        let collector = HistogramCollector::new(config);
        assert!(collector.get_all_histograms().is_empty());
    }

    #[test]
    fn test_histogram_update() -> CalibrationResult<()> {
        let config = HistogramConfig::default();
        let mut collector = HistogramCollector::new(config);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = create_test_tensor(data, &[1, 5])?;
        
        collector.update_histogram("test_layer", &tensor)?;
        
        let histogram = collector.get_histogram("test_layer");
        assert!(histogram.is_some());
        
        let histogram = histogram.unwrap();
        assert_eq!(histogram.total_count, 5);
        assert!(histogram.bins.len() > 0);
        
        Ok(())
    }

    #[test]
    fn test_histogram_bin_creation() {
        let histogram = ActivationHistogram::new(10, (0.0, 10.0), SystemTime::now());
        
        assert_eq!(histogram.bins.len(), 10);
        assert_eq!(histogram.bin_width, 1.0);
        assert_eq!(histogram.bins[0].lower_bound, 0.0);
        assert_eq!(histogram.bins[9].upper_bound, 10.0);
    }

    #[test]
    fn test_histogram_value_update() {
        let mut histogram = ActivationHistogram::new(10, (0.0, 10.0), SystemTime::now());
        let values = vec![1.5, 2.5, 1.7, 8.9];
        
        histogram.update_with_values(&values);
        
        assert_eq!(histogram.total_count, 4);
        // Values should be distributed across appropriate bins
        let total_counts: usize = histogram.bins.iter().map(|b| b.count).sum();
        assert_eq!(total_counts, 4);
    }

    #[test]
    fn test_percentile_estimator() {
        let mut estimator = PercentileEstimator::new(100, 1.0);
        
        // Add values 1-100
        for i in 1..=100 {
            estimator.add_sample(i as f32);
        }
        
        let median = estimator.estimate_percentile(50.0);
        assert!((median - 50.0).abs() < 5.0); // Allow some tolerance
        
        let p95 = estimator.estimate_percentile(95.0);
        assert!((p95 - 95.0).abs() < 10.0);
    }

    #[test]
    fn test_quantization_optimizer() -> CalibrationResult<()> {
        let mut histogram = ActivationHistogram::new(100, (-10.0, 10.0), SystemTime::now());
        
        // Create a normal-like distribution
        let mut values = Vec::new();
        for i in 0..1000 {
            let x = (i as f32 - 500.0) / 50.0; // Center around 0
            values.push(x);
        }
        histogram.update_with_values(&values);
        
        let optimizer = QuantizationOptimizer::new(histogram, OptimizationStrategy::MinimizeMSE);
        let params = optimizer.optimize(8)?; // 8-bit quantization
        
        assert!(params.scale > 0.0);
        assert!(params.error_estimate >= 0.0);
        assert!(params.confidence >= 0.0 && params.confidence <= 1.0);
        
        Ok(())
    }

    #[test]
    fn test_optimal_clip_range() {
        let mut histogram = ActivationHistogram::new(100, (0.0, 100.0), SystemTime::now());
        
        // Add values with some outliers
        let mut values: Vec<f32> = (10..90).map(|x| x as f32).collect();
        values.extend(vec![1.0, 2.0, 98.0, 99.0]); // Some outliers
        
        histogram.update_with_values(&values);
        
        let clip_range = histogram.find_optimal_clip_range(0.95);
        
        // Should exclude some outliers
        assert!(clip_range.0 > 1.0);
        assert!(clip_range.1 < 99.0);
        assert!(clip_range.0 < clip_range.1);
    }

    #[test]
    fn test_histogram_entropy() {
        let mut histogram = ActivationHistogram::new(10, (0.0, 10.0), SystemTime::now());
        
        // Uniform distribution should have high entropy
        let uniform_values: Vec<f32> = (0..100).map(|x| x as f32 / 10.0).collect();
        histogram.update_with_values(&uniform_values);
        
        let entropy = histogram.calculate_entropy();
        assert!(entropy > 0.0);
    }
}
