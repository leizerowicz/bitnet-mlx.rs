// bitnet-quant/src/metrics/error_analysis.rs
//! Core Error Analysis Implementation
//! 
//! Comprehensive error analysis system for quantization quality assessment,
//! providing detailed statistical analysis of quantization errors.

use candle_core::{Tensor, Result, Device, Error as CandleError};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::metrics::{
    QuantizationMetrics, LayerErrorAnalysis, MetricsCalculator, ErrorThresholds, 
    MitigationStrategy, tensor_to_vec, safe_divide, calculate_percentile
};

/// Comprehensive error analysis engine
#[derive(Debug)]
pub struct ErrorAnalyzer {
    device: Device,
    memory_efficient: bool,
    collect_histograms: bool,
    histogram_bins: usize,
}

impl ErrorAnalyzer {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            memory_efficient: true,
            collect_histograms: true,
            histogram_bins: 256,
        }
    }

    pub fn with_config(device: Device, memory_efficient: bool, histogram_bins: usize) -> Self {
        Self {
            device,
            memory_efficient,
            collect_histograms: true,
            histogram_bins,
        }
    }

    /// Calculate detailed error statistics between original and quantized tensors
    pub fn calculate_error_statistics(&self, original: &Tensor, quantized: &Tensor) -> Result<ErrorStatistics> {
        // Validate tensor shapes match
        if original.shape() != quantized.shape() {
            return Err(CandleError::Msg(format!(
                "Shape mismatch in error analysis: original {:?} vs quantized {:?}",
                original.shape(),
                quantized.shape()
            )));
        }

        // Calculate absolute error
        let abs_error = original.sub(quantized)?.abs()?;
        let squared_error = abs_error.powf(2.0)?;

        // Convert to vectors for detailed analysis
        let original_vec = tensor_to_vec(original)?;
        let quantized_vec = tensor_to_vec(quantized)?;
        let abs_error_vec = tensor_to_vec(&abs_error)?;
        let squared_error_vec = tensor_to_vec(&squared_error)?;

        // Calculate basic statistics
        let mse = squared_error_vec.iter().sum::<f32>() / squared_error_vec.len() as f32;
        let mae = abs_error_vec.iter().sum::<f32>() / abs_error_vec.len() as f32;
        let max_error = abs_error_vec.iter().fold(0.0f32, |a, &b| a.max(b));
        
        // Calculate relative error statistics
        let relative_errors: Vec<f32> = original_vec.iter().zip(quantized_vec.iter())
            .map(|(&orig, &quant)| {
                if orig.abs() < f32::EPSILON {
                    0.0
                } else {
                    ((orig - quant).abs() / orig.abs()).min(10.0) // Cap at 1000%
                }
            })
            .collect();
        
        let mean_relative_error = relative_errors.iter().sum::<f32>() / relative_errors.len() as f32;
        let max_relative_error = relative_errors.iter().fold(0.0f32, |a, &b| a.max(b));

        // Calculate percentiles for error distribution
        let error_percentiles = ErrorPercentiles {
            p50: calculate_percentile(&abs_error_vec, 50.0),
            p90: calculate_percentile(&abs_error_vec, 90.0),
            p95: calculate_percentile(&abs_error_vec, 95.0),
            p99: calculate_percentile(&abs_error_vec, 99.0),
        };

        // Calculate bit flip analysis
        let bit_flip_analysis = self.calculate_bit_flip_ratio(&original_vec, &quantized_vec)?;

        // Build histogram if enabled
        let error_histogram = if self.collect_histograms {
            Some(self.build_error_histogram(&abs_error_vec)?)
        } else {
            None
        };

        Ok(ErrorStatistics {
            mse,
            mae,
            max_error,
            mean_relative_error,
            max_relative_error,
            percentiles: error_percentiles,
            bit_flip_ratio: bit_flip_analysis.ratio,
            significant_bit_flips: bit_flip_analysis.significant_flips,
            error_histogram,
            sample_count: original_vec.len(),
        })
    }

    /// Calculate bit flip ratio for quantization analysis
    fn calculate_bit_flip_ratio(&self, original: &[f32], quantized: &[f32]) -> Result<BitFlipAnalysis> {
        let mut total_flips = 0usize;
        let mut significant_flips = 0usize;
        let total_comparisons = original.len();

        for (&orig, &quant) in original.iter().zip(quantized.iter()) {
            // Convert to binary representation for bit analysis
            let orig_bits = orig.to_bits();
            let quant_bits = quant.to_bits();
            
            // Count differing bits
            let diff_bits = orig_bits ^ quant_bits;
            let flip_count = diff_bits.count_ones() as usize;
            
            total_flips += flip_count;
            
            // Consider significant if more than 25% of mantissa bits flipped
            if flip_count > 6 { // ~25% of 23 mantissa bits
                significant_flips += 1;
            }
        }

        let ratio = total_flips as f32 / (total_comparisons * 32) as f32; // 32 bits per f32
        let significant_ratio = significant_flips as f32 / total_comparisons as f32;

        Ok(BitFlipAnalysis {
            ratio,
            significant_flips: significant_ratio,
            total_flips,
            significant_count: significant_flips,
        })
    }

    /// Build error histogram for distribution analysis
    fn build_error_histogram(&self, errors: &[f32]) -> Result<ErrorHistogram> {
        if errors.is_empty() {
            return Ok(ErrorHistogram::default());
        }

        let min_error = errors.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_error = errors.iter().fold(0.0f32, |a, &b| a.max(b));
        
        // Use log scale for better distribution visualization
        let log_min = (min_error + 1e-10).ln();
        let log_max = (max_error + 1e-10).ln();
        let bin_width = (log_max - log_min) / self.histogram_bins as f32;

        let mut bins = vec![0usize; self.histogram_bins];
        let mut bin_edges = Vec::with_capacity(self.histogram_bins + 1);

        // Create bin edges
        for i in 0..=self.histogram_bins {
            let log_edge = log_min + i as f32 * bin_width;
            bin_edges.push(log_edge.exp() - 1e-10);
        }

        // Count errors in each bin
        for &error in errors {
            let log_error = (error + 1e-10).ln();
            let bin_index = ((log_error - log_min) / bin_width) as usize;
            let bin_index = bin_index.min(self.histogram_bins - 1);
            bins[bin_index] += 1;
        }

        Ok(ErrorHistogram {
            bins,
            bin_edges,
            total_samples: errors.len(),
        })
    }

    /// Analyze error patterns and correlations
    pub fn analyze_error_patterns(&self, original: &Tensor, quantized: &Tensor) -> Result<ErrorPatterns> {
        let original_vec = tensor_to_vec(original)?;
        let quantized_vec = tensor_to_vec(quantized)?;
        
        // Calculate magnitude-based error correlation
        let magnitude_correlation = self.calculate_magnitude_error_correlation(&original_vec, &quantized_vec)?;
        
        // Detect outliers in quantization errors
        let outlier_analysis = self.detect_error_outliers(&original_vec, &quantized_vec)?;
        
        // Analyze spatial error patterns if tensor has spatial structure
        let spatial_patterns = if original.dims().len() >= 2 {
            Some(self.analyze_spatial_error_patterns(original, quantized)?)
        } else {
            None
        };

        Ok(ErrorPatterns {
            magnitude_correlation,
            outlier_analysis,
            spatial_patterns,
        })
    }

    fn calculate_magnitude_error_correlation(&self, original: &[f32], quantized: &[f32]) -> Result<f32> {
        let errors: Vec<f32> = original.iter().zip(quantized.iter())
            .map(|(&orig, &quant)| (orig - quant).abs())
            .collect();

        let magnitudes: Vec<f32> = original.iter().map(|&x| x.abs()).collect();

        // Calculate Pearson correlation coefficient
        let n = errors.len() as f32;
        let error_mean = errors.iter().sum::<f32>() / n;
        let magnitude_mean = magnitudes.iter().sum::<f32>() / n;

        let numerator: f32 = errors.iter().zip(magnitudes.iter())
            .map(|(&err, &mag)| (err - error_mean) * (mag - magnitude_mean))
            .sum();

        let error_variance: f32 = errors.iter()
            .map(|&err| (err - error_mean).powi(2))
            .sum();

        let magnitude_variance: f32 = magnitudes.iter()
            .map(|&mag| (mag - magnitude_mean).powi(2))
            .sum();

        let correlation = safe_divide(numerator, (error_variance * magnitude_variance).sqrt());
        Ok(correlation)
    }

    fn detect_error_outliers(&self, original: &[f32], quantized: &[f32]) -> Result<OutlierAnalysis> {
        let errors: Vec<f32> = original.iter().zip(quantized.iter())
            .map(|(&orig, &quant)| (orig - quant).abs())
            .collect();

        // Use IQR method for outlier detection
        let q25 = calculate_percentile(&errors, 25.0);
        let q75 = calculate_percentile(&errors, 75.0);
        let iqr = q75 - q25;
        let outlier_threshold = q75 + 1.5 * iqr;

        let outlier_indices: Vec<usize> = errors.iter().enumerate()
            .filter_map(|(i, &error)| if error > outlier_threshold { Some(i) } else { None })
            .collect();

        let outlier_ratio = outlier_indices.len() as f32 / errors.len() as f32;
        let total_outliers = outlier_indices.len();

        Ok(OutlierAnalysis {
            outlier_indices,
            outlier_threshold,
            outlier_ratio,
            total_outliers,
        })
    }

    fn analyze_spatial_error_patterns(&self, original: &Tensor, quantized: &Tensor) -> Result<SpatialPatterns> {
        // Calculate error tensor
        let error_tensor = original.sub(quantized)?.abs()?;
        
        // For 2D tensors, calculate row and column error means
        if original.dims().len() == 2 {
            let row_errors = error_tensor.mean_keepdim(1)?; // Mean across columns
            let col_errors = error_tensor.mean_keepdim(0)?; // Mean across rows
            
            let row_errors_vec = tensor_to_vec(&row_errors)?;
            let col_errors_vec = tensor_to_vec(&col_errors)?;
            
            // Calculate variance in row and column errors
            let row_variance = self.calculate_variance(&row_errors_vec);
            let col_variance = self.calculate_variance(&col_errors_vec);
            
            Ok(SpatialPatterns {
                row_error_variance: row_variance,
                col_error_variance: col_variance,
                spatial_correlation: row_variance + col_variance, // Simple spatial correlation measure
            })
        } else {
            // For higher dimensions, calculate overall spatial variance
            let spatial_variance = self.calculate_spatial_variance(&error_tensor)?;
            Ok(SpatialPatterns {
                row_error_variance: spatial_variance,
                col_error_variance: spatial_variance,
                spatial_correlation: spatial_variance,
            })
        }
    }

    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance
    }

    fn calculate_spatial_variance(&self, tensor: &Tensor) -> Result<f32> {
        let values = tensor_to_vec(tensor)?;
        Ok(self.calculate_variance(&values))
    }
}

impl MetricsCalculator for ErrorAnalyzer {
    fn calculate_metrics(&self, original: &Tensor, quantized: &Tensor, layer_name: &str) -> Result<QuantizationMetrics> {
        let error_stats = self.calculate_error_statistics(original, quantized)?;
        
        // Calculate additional metrics
        let sqnr = crate::metrics::sqnr::calculate_sqnr(original, quantized)?;
        let cosine_sim = crate::metrics::cosine_similarity::calculate_cosine_similarity(original, quantized)?;
        
        Ok(QuantizationMetrics {
            mse: error_stats.mse,
            sqnr,
            cosine_similarity: cosine_sim,
            max_error: error_stats.max_error,
            mean_absolute_error: error_stats.mae,
            relative_error: error_stats.mean_relative_error,
            bit_flip_ratio: error_stats.bit_flip_ratio,
            layer_name: layer_name.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    fn analyze_layer_errors(&self, layer_outputs: HashMap<String, (Tensor, Tensor)>) -> Result<LayerErrorAnalysis> {
        let mut layer_metrics = HashMap::new();
        let mut all_errors = Vec::new();
        let mut sensitivity_scores = Vec::new();

        // Calculate metrics for each layer
        for (layer_name, (original, quantized)) in layer_outputs.iter() {
            let metrics = self.calculate_metrics(original, quantized, layer_name)?;
            
            // Collect error statistics for global analysis
            let error_tensor = original.sub(quantized)?.abs()?;
            let error_vec = tensor_to_vec(&error_tensor)?;
            all_errors.extend(error_vec);
            
            // Calculate sensitivity score (combination of MSE and max error)
            let sensitivity_score = metrics.mse + metrics.max_error * 0.1;
            sensitivity_scores.push((layer_name.clone(), sensitivity_score));
            
            layer_metrics.insert(layer_name.clone(), metrics);
        }

        // Sort layers by sensitivity
        sensitivity_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate global metrics
        let global_mse = all_errors.iter().map(|&x| x * x).sum::<f32>() / all_errors.len() as f32;
        let global_mae = all_errors.iter().sum::<f32>() / all_errors.len() as f32;
        let global_max = all_errors.iter().fold(0.0f32, |a, &b| a.max(b));
        
        let global_metrics = QuantizationMetrics {
            mse: global_mse,
            mean_absolute_error: global_mae,
            max_error: global_max,
            layer_name: "global".to_string(),
            ..QuantizationMetrics::default()
        };

        // Recommend bit widths based on sensitivity
        let mut recommended_bit_widths = HashMap::new();
        for (layer_name, sensitivity) in &sensitivity_scores {
            let recommended_bits = if sensitivity > &0.1 {
                8 // High sensitivity, use more bits
            } else if sensitivity > &0.01 {
                4 // Medium sensitivity
            } else {
                2 // Low sensitivity, can use fewer bits
            };
            recommended_bit_widths.insert(layer_name.clone(), recommended_bits);
        }

        Ok(LayerErrorAnalysis {
            layer_metrics,
            global_metrics,
            sensitivity_ranking: sensitivity_scores,
            error_distribution: all_errors,
            recommended_bit_widths,
        })
    }

    fn check_quality_thresholds(&self, metrics: &QuantizationMetrics, thresholds: &ErrorThresholds) -> bool {
        metrics.mse <= thresholds.max_mse
            && metrics.sqnr >= thresholds.min_sqnr
            && metrics.cosine_similarity >= thresholds.min_cosine_similarity
            && metrics.relative_error <= thresholds.max_relative_error
            && metrics.bit_flip_ratio <= thresholds.max_bit_flip_ratio
    }

    fn suggest_mitigation(&self, metrics: &QuantizationMetrics, thresholds: &ErrorThresholds) -> Vec<MitigationStrategy> {
        let mut strategies = Vec::new();

        if metrics.mse > thresholds.max_mse {
            strategies.push(MitigationStrategy::IncreaseBitWidth);
            strategies.push(MitigationStrategy::AdjustScaleFactor);
        }

        if metrics.sqnr < thresholds.min_sqnr {
            strategies.push(MitigationStrategy::UseAsymmetricQuantization);
            strategies.push(MitigationStrategy::AddRegularization);
        }

        if metrics.cosine_similarity < thresholds.min_cosine_similarity {
            strategies.push(MitigationStrategy::EnableMixedPrecision);
        }

        if metrics.max_error > metrics.mean_absolute_error * 10.0 {
            strategies.push(MitigationStrategy::ApplyClipping);
        }

        if metrics.bit_flip_ratio > thresholds.max_bit_flip_ratio {
            strategies.push(MitigationStrategy::IncreaseBitWidth);
        }

        // Remove duplicates
        strategies.sort();
        strategies.dedup();

        strategies
    }
}

/// Detailed error statistics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub mse: f32,
    pub mae: f32,
    pub max_error: f32,
    pub mean_relative_error: f32,
    pub max_relative_error: f32,
    pub percentiles: ErrorPercentiles,
    pub bit_flip_ratio: f32,
    pub significant_bit_flips: f32,
    pub error_histogram: Option<ErrorHistogram>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPercentiles {
    pub p50: f32,
    pub p90: f32,
    pub p95: f32,
    pub p99: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitFlipAnalysis {
    pub ratio: f32,
    pub significant_flips: f32,
    pub total_flips: usize,
    pub significant_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f32>,
    pub total_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPatterns {
    pub magnitude_correlation: f32,
    pub outlier_analysis: OutlierAnalysis,
    pub spatial_patterns: Option<SpatialPatterns>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    pub outlier_indices: Vec<usize>,
    pub outlier_threshold: f32,
    pub outlier_ratio: f32,
    pub total_outliers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPatterns {
    pub row_error_variance: f32,
    pub col_error_variance: f32,
    pub spatial_correlation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_error_analyzer_creation() {
        let device = Device::Cpu;
        let analyzer = ErrorAnalyzer::new(device);
        assert!(analyzer.memory_efficient);
        assert!(analyzer.collect_histograms);
    }

    #[test]
    fn test_bit_flip_analysis() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = ErrorAnalyzer::new(device);
        
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = vec![1.0, 2.0, 3.0, 4.0]; // Same values
        
        let analysis = analyzer.calculate_bit_flip_ratio(&original, &quantized)?;
        assert_eq!(analysis.ratio, 0.0); // No bit flips for identical values
        
        Ok(())
    }

    #[test]
    fn test_variance_calculation() {
        let device = Device::Cpu;
        let analyzer = ErrorAnalyzer::new(device);
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = analyzer.calculate_variance(&values);
        assert!((variance - 2.5).abs() < 1e-6);
    }
}
