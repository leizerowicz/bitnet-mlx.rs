//! Statistical analysis utilities for quantization testing
//!
//! This module provides tools for analyzing quantization quality,
//! computing metrics, and validating statistical properties.

use candle_core::Tensor;
use crate::quantization::{QuantizationResult, QuantizationError};
use std::collections::HashMap;

/// Comprehensive quantization metrics
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Signal-to-Quantization-Noise Ratio (dB)
    pub sqnr_db: f64,
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr_db: f64,
    /// Cosine similarity
    pub cosine_similarity: f64,
    /// Pearson correlation coefficient
    pub correlation_coefficient: f64,
    /// Structural Similarity Index (SSIM)
    pub ssim: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Entropy of quantized values
    pub quantized_entropy: f64,
    /// Signal dynamic range
    pub dynamic_range_db: f64,
}

impl QuantizationMetrics {
    /// Compute comprehensive metrics between original and quantized tensors
    pub fn compute(
        original: &Tensor,
        quantized: &Tensor,
        dequantized: &Tensor,
        bits_per_value: f64,
    ) -> QuantizationResult<Self> {
        // Flatten tensors for analysis
        let orig_values = tensor_to_vec(original)?;
        let quant_values = tensor_to_vec(quantized)?;
        let dequant_values = tensor_to_vec(dequantized)?;

        // Basic error metrics
        let mse = compute_mse(&orig_values, &dequant_values);
        let rmse = mse.sqrt();
        let mae = compute_mae(&orig_values, &dequant_values);

        // Signal-to-noise metrics
        let sqnr_db = compute_sqnr_db(&orig_values, &dequant_values);
        let psnr_db = compute_psnr_db(&orig_values, &dequant_values);

        // Similarity metrics
        let cosine_similarity = compute_cosine_similarity(&orig_values, &dequant_values);
        let correlation_coefficient = compute_correlation(&orig_values, &dequant_values);
        let ssim = compute_ssim(&orig_values, &dequant_values);

        // Compression and information metrics
        let original_bits = orig_values.len() as f64 * 32.0; // Assuming f32
        let quantized_bits = quant_values.len() as f64 * bits_per_value;
        let compression_ratio = original_bits / quantized_bits;

        let quantized_entropy = compute_entropy(&quant_values);
        let dynamic_range_db = compute_dynamic_range_db(&orig_values);

        Ok(QuantizationMetrics {
            mse,
            rmse,
            mae,
            sqnr_db,
            psnr_db,
            cosine_similarity,
            correlation_coefficient,
            ssim,
            compression_ratio,
            quantized_entropy,
            dynamic_range_db,
        })
    }

    /// Check if metrics meet quality thresholds
    pub fn meets_quality_thresholds(&self, thresholds: &QualityThresholds) -> QualityAssessment {
        let mut passed_criteria = Vec::new();
        let mut failed_criteria = Vec::new();

        // MSE check
        if self.mse <= thresholds.max_mse {
            passed_criteria.push("MSE".to_string());
        } else {
            failed_criteria.push(format!("MSE: {:.6} > {:.6}", self.mse, thresholds.max_mse));
        }

        // SQNR check
        if self.sqnr_db >= thresholds.min_sqnr_db {
            passed_criteria.push("SQNR".to_string());
        } else {
            failed_criteria.push(format!("SQNR: {:.2} dB < {:.2} dB", self.sqnr_db, thresholds.min_sqnr_db));
        }

        // Cosine similarity check
        if self.cosine_similarity >= thresholds.min_cosine_similarity {
            passed_criteria.push("Cosine Similarity".to_string());
        } else {
            failed_criteria.push(format!("Cosine: {:.4} < {:.4}", self.cosine_similarity, thresholds.min_cosine_similarity));
        }

        // Correlation check
        if self.correlation_coefficient >= thresholds.min_correlation {
            passed_criteria.push("Correlation".to_string());
        } else {
            failed_criteria.push(format!("Correlation: {:.4} < {:.4}", self.correlation_coefficient, thresholds.min_correlation));
        }

        // Compression ratio check
        if self.compression_ratio >= thresholds.min_compression_ratio {
            passed_criteria.push("Compression Ratio".to_string());
        } else {
            failed_criteria.push(format!("Compression: {:.2}x < {:.2}x", self.compression_ratio, thresholds.min_compression_ratio));
        }

        let overall_passed = failed_criteria.is_empty();

        QualityAssessment {
            overall_passed,
            passed_criteria,
            failed_criteria,
            score: calculate_quality_score(self, thresholds),
        }
    }
}

/// Quality assessment thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub max_mse: f64,
    pub min_sqnr_db: f64,
    pub min_cosine_similarity: f64,
    pub min_correlation: f64,
    pub min_compression_ratio: f64,
    pub max_mae: f64,
    pub min_psnr_db: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_mse: 0.01,           // 1% MSE
            min_sqnr_db: 20.0,       // 20 dB SQNR
            min_cosine_similarity: 0.95, // 95% similarity
            min_correlation: 0.90,   // 90% correlation
            min_compression_ratio: 15.0, // 15x compression
            max_mae: 0.1,           // 10% MAE
            min_psnr_db: 30.0,      // 30 dB PSNR
        }
    }
}

impl QualityThresholds {
    /// Strict thresholds for high-accuracy requirements
    pub fn strict() -> Self {
        Self {
            max_mse: 0.005,          // 0.5% MSE
            min_sqnr_db: 25.0,       // 25 dB SQNR
            min_cosine_similarity: 0.98, // 98% similarity
            min_correlation: 0.95,   // 95% correlation
            min_compression_ratio: 18.0, // 18x compression
            max_mae: 0.05,          // 5% MAE
            min_psnr_db: 35.0,      // 35 dB PSNR
        }
    }

    /// Relaxed thresholds for experimental validation
    pub fn relaxed() -> Self {
        Self {
            max_mse: 0.05,           // 5% MSE
            min_sqnr_db: 15.0,       // 15 dB SQNR
            min_cosine_similarity: 0.85, // 85% similarity
            min_correlation: 0.80,   // 80% correlation
            min_compression_ratio: 10.0, // 10x compression
            max_mae: 0.2,           // 20% MAE
            min_psnr_db: 20.0,      // 20 dB PSNR
        }
    }
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub overall_passed: bool,
    pub passed_criteria: Vec<String>,
    pub failed_criteria: Vec<String>,
    pub score: f64, // Overall quality score [0, 1]
}

impl QualityAssessment {
    pub fn success_rate(&self) -> f64 {
        let total = self.passed_criteria.len() + self.failed_criteria.len();
        if total > 0 {
            self.passed_criteria.len() as f64 / total as f64
        } else {
            1.0
        }
    }
}

/// Statistical analysis of quantization behavior
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Distribution of quantized values
    pub value_distribution: ValueDistribution,
    /// Error distribution analysis
    pub error_analysis: ErrorDistributionAnalysis,
    /// Outlier analysis
    pub outlier_analysis: OutlierAnalysis,
    /// Stability analysis across different inputs
    pub stability_analysis: StabilityAnalysis,
}

impl StatisticalAnalysis {
    /// Perform comprehensive statistical analysis
    pub fn analyze(
        original_tensors: &[Tensor],
        quantized_tensors: &[Tensor],
        dequantized_tensors: &[Tensor],
    ) -> QuantizationResult<Self> {
        let value_distribution = analyze_value_distribution(quantized_tensors)?;
        let error_analysis = analyze_error_distribution(original_tensors, dequantized_tensors)?;
        let outlier_analysis = analyze_outliers(original_tensors, quantized_tensors)?;
        let stability_analysis = analyze_stability(original_tensors, dequantized_tensors)?;

        Ok(StatisticalAnalysis {
            value_distribution,
            error_analysis,
            outlier_analysis,
            stability_analysis,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ValueDistribution {
    pub histogram: HashMap<i32, usize>,
    pub entropy: f64,
    pub sparsity: f64,
    pub balance_score: f64, // How balanced the distribution is
}

#[derive(Debug, Clone)]
pub struct ErrorDistributionAnalysis {
    pub mean_error: f64,
    pub std_error: f64,
    pub error_percentiles: HashMap<u8, f64>, // 50th, 90th, 95th, 99th percentiles
    pub error_histogram: HashMap<String, usize>,
    pub is_normally_distributed: bool,
}

#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    pub outlier_count: usize,
    pub outlier_threshold: f64,
    pub max_outlier_magnitude: f64,
    pub outlier_impact_on_quantization: f64,
}

#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    pub cross_tensor_mse_variance: f64,
    pub consistency_score: f64, // [0, 1] - higher is more consistent
    pub per_tensor_metrics: Vec<QuantizationMetrics>,
}

// Helper functions for metric computation

fn tensor_to_vec(tensor: &Tensor) -> QuantizationResult<Vec<f32>> {
    tensor.flatten_all()
        .map_err(|e| QuantizationError::TensorError {
            reason: format!("Failed to flatten tensor: {}", e)
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError {
            reason: format!("Failed to convert to vector: {}", e)
        })
}

fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f64 {
    let squared_errors: Vec<f64> = original.iter().zip(reconstructed.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .collect();
    squared_errors.iter().sum::<f64>() / squared_errors.len() as f64
}

fn compute_mae(original: &[f32], reconstructed: &[f32]) -> f64 {
    let absolute_errors: Vec<f64> = original.iter().zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .collect();
    absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64
}

fn compute_sqnr_db(original: &[f32], reconstructed: &[f32]) -> f64 {
    let signal_power: f64 = original.iter().map(|&x| (x * x) as f64).sum::<f64>()
                           / original.len() as f64;
    let mse = compute_mse(original, reconstructed);

    if mse > 1e-10 && signal_power > 1e-10 {
        10.0 * (signal_power / mse).log10()
    } else {
        f64::INFINITY
    }
}

fn compute_psnr_db(original: &[f32], reconstructed: &[f32]) -> f64 {
    let max_val = original.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())) as f64;
    let mse = compute_mse(original, reconstructed);

    if mse > 1e-10 && max_val > 1e-10 {
        20.0 * (max_val / mse.sqrt()).log10()
    } else {
        f64::INFINITY
    }
}

fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y) as f64).sum();
    let norm_a: f64 = a.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot_product / (norm_a * norm_b)
    } else {
        1.0 // Both vectors are zero
    }
}

fn compute_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f32>() as f64 / n;
    let mean_b = b.iter().sum::<f32>() as f64 / n;

    let covariance: f64 = a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - mean_a) * ((y as f64) - mean_b))
        .sum::<f64>() / n;

    let var_a: f64 = a.iter().map(|&x| ((x as f64) - mean_a).powi(2)).sum::<f64>() / n;
    let var_b: f64 = b.iter().map(|&x| ((x as f64) - mean_b).powi(2)).sum::<f64>() / n;

    if var_a > 1e-10 && var_b > 1e-10 {
        covariance / (var_a.sqrt() * var_b.sqrt())
    } else {
        0.0
    }
}

fn compute_ssim(a: &[f32], b: &[f32]) -> f64 {
    // Simplified SSIM calculation for 1D signals
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f32>() as f64 / n;
    let mean_b = b.iter().sum::<f32>() as f64 / n;

    let var_a: f64 = a.iter().map(|&x| ((x as f64) - mean_a).powi(2)).sum::<f64>() / (n - 1.0);
    let var_b: f64 = b.iter().map(|&x| ((x as f64) - mean_b).powi(2)).sum::<f64>() / (n - 1.0);
    let cov_ab: f64 = a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - mean_a) * ((y as f64) - mean_b))
        .sum::<f64>() / (n - 1.0);

    // SSIM constants (typically c1 = (0.01 * L)^2, c2 = (0.03 * L)^2 where L is dynamic range)
    let c1 = 1e-4;
    let c2 = 9e-4;

    let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
    let denominator = (mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2);

    if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        1.0
    }
}

fn compute_entropy(values: &[f32]) -> f64 {
    let mut histogram = HashMap::new();

    // Create histogram of quantized values
    for &value in values {
        let key = if value.abs() < 1e-6 { 0 } // Zero
        else if value > 1e-6 { 1 }            // Positive
        else { -1 };                          // Negative

        *histogram.entry(key).or_insert(0) += 1;
    }

    // Compute entropy
    let total = values.len() as f64;
    let mut entropy = 0.0;

    for &count in histogram.values() {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn compute_dynamic_range_db(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let min_abs = values.iter().filter(|&&x| x.abs() > 1e-10).map(|&x| x.abs()).fold(f32::INFINITY, f32::min);
    let max_abs = values.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    if min_abs.is_finite() && max_abs > min_abs {
        20.0 * (max_abs / min_abs).log10() as f64
    } else {
        0.0
    }
}

fn calculate_quality_score(metrics: &QuantizationMetrics, thresholds: &QualityThresholds) -> f64 {
    let mut score = 0.0;
    let mut weight_sum = 0.0;

    // MSE score (inverted, lower is better)
    let mse_score = if metrics.mse <= thresholds.max_mse {
        1.0 - (metrics.mse / thresholds.max_mse)
    } else {
        0.0
    };
    score += mse_score * 0.2;
    weight_sum += 0.2;

    // SQNR score
    let sqnr_score = (metrics.sqnr_db / thresholds.min_sqnr_db).min(1.0);
    score += sqnr_score * 0.25;
    weight_sum += 0.25;

    // Cosine similarity score
    let cosine_score = (metrics.cosine_similarity / thresholds.min_cosine_similarity).min(1.0);
    score += cosine_score * 0.2;
    weight_sum += 0.2;

    // Correlation score
    let corr_score = (metrics.correlation_coefficient / thresholds.min_correlation).min(1.0);
    score += corr_score * 0.15;
    weight_sum += 0.15;

    // Compression ratio score
    let comp_score = (metrics.compression_ratio / thresholds.min_compression_ratio).min(1.0);
    score += comp_score * 0.2;
    weight_sum += 0.2;

    if weight_sum > 0.0 {
        score / weight_sum
    } else {
        0.0
    }
}

// Analysis functions

fn analyze_value_distribution(tensors: &[Tensor]) -> QuantizationResult<ValueDistribution> {
    let mut histogram = HashMap::new();
    let mut total_values = 0;
    let mut zero_count = 0;

    for tensor in tensors {
        let values = tensor_to_vec(tensor)?;
        total_values += values.len();

        for &value in &values {
            let key = if value.abs() < 1e-6 {
                zero_count += 1;
                0
            } else if value > 1e-6 { 1 } else { -1 };
            *histogram.entry(key).or_insert(0) += 1;
        }
    }

    let entropy = compute_entropy_from_histogram(&histogram, total_values);
    let sparsity = zero_count as f64 / total_values as f64;
    let balance_score = calculate_balance_score(&histogram);

    Ok(ValueDistribution {
        histogram,
        entropy,
        sparsity,
        balance_score,
    })
}

fn analyze_error_distribution(
    original_tensors: &[Tensor],
    reconstructed_tensors: &[Tensor],
) -> QuantizationResult<ErrorDistributionAnalysis> {
    let mut all_errors = Vec::new();

    for (orig, recon) in original_tensors.iter().zip(reconstructed_tensors.iter()) {
        let orig_vals = tensor_to_vec(orig)?;
        let recon_vals = tensor_to_vec(recon)?;

        for (&o, &r) in orig_vals.iter().zip(recon_vals.iter()) {
            all_errors.push((o - r).abs() as f64);
        }
    }

    if all_errors.is_empty() {
        return Err(QuantizationError::ValidationError {
            reason: "No error data to analyze".to_string()
        });
    }

    let mean_error = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    let variance = all_errors.iter().map(|&e| (e - mean_error).powi(2)).sum::<f64>() / all_errors.len() as f64;
    let std_error = variance.sqrt();

    // Compute percentiles
    let mut sorted_errors = all_errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut error_percentiles = HashMap::new();
    for &percentile in &[50, 90, 95, 99] {
        let index = ((percentile as f64 / 100.0) * sorted_errors.len() as f64) as usize;
        let value = sorted_errors[index.min(sorted_errors.len() - 1)];
        error_percentiles.insert(percentile, value);
    }

    // Create error histogram
    let mut error_histogram = HashMap::new();
    for &error in &all_errors {
        let bucket = if error < 0.001 { "< 0.001" }
        else if error < 0.01 { "0.001-0.01" }
        else if error < 0.1 { "0.01-0.1" }
        else if error < 1.0 { "0.1-1.0" }
        else { "> 1.0" };

        *error_histogram.entry(bucket.to_string()).or_insert(0) += 1;
    }

    // Simple normality test (skewness and kurtosis)
    let is_normally_distributed = check_normality(&all_errors, mean_error, std_error);

    Ok(ErrorDistributionAnalysis {
        mean_error,
        std_error,
        error_percentiles,
        error_histogram,
        is_normally_distributed,
    })
}

fn analyze_outliers(
    original_tensors: &[Tensor],
    quantized_tensors: &[Tensor],
) -> QuantizationResult<OutlierAnalysis> {
    let mut all_original_values = Vec::new();

    for tensor in original_tensors {
        let values = tensor_to_vec(tensor)?;
        all_original_values.extend(values);
    }

    if all_original_values.is_empty() {
        return Ok(OutlierAnalysis {
            outlier_count: 0,
            outlier_threshold: 0.0,
            max_outlier_magnitude: 0.0,
            outlier_impact_on_quantization: 0.0,
        });
    }

    // Compute outlier threshold using IQR method
    let mut sorted_values = all_original_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1_idx = sorted_values.len() / 4;
    let q3_idx = 3 * sorted_values.len() / 4;
    let q1 = sorted_values[q1_idx];
    let q3 = sorted_values[q3_idx];
    let iqr = q3 - q1;

    let outlier_threshold = 1.5 * iqr;
    let median = sorted_values[sorted_values.len() / 2];

    let mut outlier_count = 0;
    let mut max_outlier_magnitude = 0.0f32;

    for &value in &all_original_values {
        if (value - median).abs() > outlier_threshold {
            outlier_count += 1;
            max_outlier_magnitude = max_outlier_magnitude.max((value - median).abs());
        }
    }

    // Simplified impact calculation
    let outlier_impact_on_quantization = outlier_count as f64 / all_original_values.len() as f64;

    Ok(OutlierAnalysis {
        outlier_count,
        outlier_threshold: outlier_threshold as f64,
        max_outlier_magnitude: max_outlier_magnitude as f64,
        outlier_impact_on_quantization,
    })
}

fn analyze_stability(
    original_tensors: &[Tensor],
    reconstructed_tensors: &[Tensor],
) -> QuantizationResult<StabilityAnalysis> {
    let mut per_tensor_mses = Vec::new();
    let mut per_tensor_metrics = Vec::new();

    for (orig, recon) in original_tensors.iter().zip(reconstructed_tensors.iter()) {
        let orig_vals = tensor_to_vec(orig)?;
        let recon_vals = tensor_to_vec(recon)?;
        let mse = compute_mse(&orig_vals, &recon_vals);
        per_tensor_mses.push(mse);

        // Create dummy quantized tensor for metrics (simplified)
        let quant_vals: Vec<f32> = recon_vals.iter().map(|&x| {
            if x > 0.5 { 1.0 } else if x < -0.5 { -1.0 } else { 0.0 }
        }).collect();

        // For this simplified analysis, we'll compute basic metrics
        let cosine_sim = compute_cosine_similarity(&orig_vals, &recon_vals);
        let mae = compute_mae(&orig_vals, &recon_vals);
        let sqnr = compute_sqnr_db(&orig_vals, &recon_vals);

        let metrics = QuantizationMetrics {
            mse,
            rmse: mse.sqrt(),
            mae,
            sqnr_db: sqnr,
            psnr_db: compute_psnr_db(&orig_vals, &recon_vals),
            cosine_similarity: cosine_sim,
            correlation_coefficient: compute_correlation(&orig_vals, &recon_vals),
            ssim: compute_ssim(&orig_vals, &recon_vals),
            compression_ratio: 20.25, // Approximate for 1.58-bit
            quantized_entropy: compute_entropy(&quant_vals),
            dynamic_range_db: compute_dynamic_range_db(&orig_vals),
        };

        per_tensor_metrics.push(metrics);
    }

    let mean_mse = per_tensor_mses.iter().sum::<f64>() / per_tensor_mses.len() as f64;
    let mse_variance = per_tensor_mses.iter()
        .map(|&mse| (mse - mean_mse).powi(2))
        .sum::<f64>() / per_tensor_mses.len() as f64;

    let consistency_score = if mse_variance > 0.0 {
        1.0 / (1.0 + mse_variance)
    } else {
        1.0
    };

    Ok(StabilityAnalysis {
        cross_tensor_mse_variance: mse_variance,
        consistency_score,
        per_tensor_metrics,
    })
}

// Utility functions

fn compute_entropy_from_histogram(histogram: &HashMap<i32, usize>, total: usize) -> f64 {
    let mut entropy = 0.0;
    let total_f = total as f64;

    for &count in histogram.values() {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn calculate_balance_score(histogram: &HashMap<i32, usize>) -> f64 {
    let pos = *histogram.get(&1).unwrap_or(&0) as f64;
    let neg = *histogram.get(&-1).unwrap_or(&0) as f64;
    let zero = *histogram.get(&0).unwrap_or(&0) as f64;

    let total = pos + neg + zero;
    if total == 0.0 {
        return 1.0;
    }

    // Balance score based on how evenly distributed the values are
    let expected = total / 3.0;
    let variance = ((pos - expected).powi(2) + (neg - expected).powi(2) + (zero - expected).powi(2)) / 3.0;

    1.0 / (1.0 + variance / (expected + 1.0))
}

fn check_normality(values: &[f64], mean: f64, std_dev: f64) -> bool {
    if std_dev < 1e-10 {
        return false; // Constant values are not normally distributed
    }

    // Simple normality check using skewness and kurtosis
    let n = values.len() as f64;

    let skewness = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    let kurtosis = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>() / n - 3.0; // Excess kurtosis

    // Rough thresholds for normality
    skewness.abs() < 2.0 && kurtosis.abs() < 7.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::helpers::test_data::{generate_test_tensor, TestPattern, create_test_device};

    #[test]
    fn test_quantization_metrics_computation() {
        let device = create_test_device();
        let original = generate_test_tensor(TestPattern::UniformDistribution, &[100], &device).unwrap();
        let quantized = generate_test_tensor(TestPattern::RandomTernary, &[100], &device).unwrap();
        let dequantized = original.clone(); // Perfect reconstruction for test

        let metrics = QuantizationMetrics::compute(&original, &quantized, &dequantized, 1.58).unwrap();

        assert!(metrics.compression_ratio > 1.0);
        assert!(metrics.mse >= 0.0);
        assert!(metrics.cosine_similarity <= 1.0);
        assert!(metrics.quantized_entropy >= 0.0);
    }

    #[test]
    fn test_quality_thresholds() {
        let strict_thresholds = QualityThresholds::strict();
        let relaxed_thresholds = QualityThresholds::relaxed();

        assert!(strict_thresholds.max_mse < relaxed_thresholds.max_mse);
        assert!(strict_thresholds.min_sqnr_db > relaxed_thresholds.min_sqnr_db);
        assert!(strict_thresholds.min_cosine_similarity > relaxed_thresholds.min_cosine_similarity);
    }

    #[test]
    fn test_statistical_analysis() {
        let device = create_test_device();
        let original_tensors = vec![
            generate_test_tensor(TestPattern::NormalDistribution, &[50], &device).unwrap(),
            generate_test_tensor(TestPattern::UniformDistribution, &[50], &device).unwrap(),
        ];
        let quantized_tensors = vec![
            generate_test_tensor(TestPattern::RandomTernary, &[50], &device).unwrap(),
            generate_test_tensor(TestPattern::RandomTernary, &[50], &device).unwrap(),
        ];
        let dequantized_tensors = original_tensors.clone(); // Simplified

        let analysis = StatisticalAnalysis::analyze(&original_tensors, &quantized_tensors, &dequantized_tensors).unwrap();

        assert!(analysis.value_distribution.entropy >= 0.0);
        assert!(analysis.value_distribution.sparsity >= 0.0 && analysis.value_distribution.sparsity <= 1.0);
        assert!(analysis.stability_analysis.consistency_score >= 0.0 && analysis.stability_analysis.consistency_score <= 1.0);
    }
}
