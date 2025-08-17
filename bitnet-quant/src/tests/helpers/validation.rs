//! Validation helper functions for quantization testing
//!
//! This module provides utilities for validating quantization results,
//! computing error metrics, and checking mathematical properties.

use candle_core::{Tensor, Device};
use crate::quantization::{QuantizationResult, QuantizationError, TernaryMethod, WeightQuantizer};
use std::collections::HashMap;

/// Result of ternary quantization validation
#[derive(Debug, Clone)]
pub struct TernaryValidationResult {
    /// Whether all values are strictly ternary {-1, 0, +1}
    pub is_strictly_ternary: bool,
    /// Distribution of quantized values
    pub value_distribution: TernaryDistribution,
    /// Threshold that was used
    pub threshold_used: f32,
    /// Number of values quantized to each level
    pub quantization_counts: QuantizationCounts,
    /// Error message if validation failed
    pub error_message: Option<String>,
}

/// Distribution of ternary values
#[derive(Debug, Clone)]
pub struct TernaryDistribution {
    pub negative_ones: usize,
    pub zeros: usize,
    pub positive_ones: usize,
    pub total: usize,
}

impl TernaryDistribution {
    pub fn sparsity(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.zeros as f64 / self.total as f64 }
    }

    pub fn is_balanced(&self) -> bool {
        let threshold = self.total / 10; // 10% tolerance
        let neg_pos_diff = (self.negative_ones as i32 - self.positive_ones as i32).abs() as usize;
        neg_pos_diff <= threshold
    }
}

/// Counts of quantization levels
#[derive(Debug, Clone)]
pub struct QuantizationCounts {
    pub minus_one: usize,
    pub zero: usize,
    pub plus_one: usize,
    pub invalid: usize, // Non-ternary values
}

/// Validate that a tensor contains only ternary values {-1, 0, +1}
pub fn validate_ternary_values(tensor: &Tensor) -> QuantizationResult<TernaryValidationResult> {
    let values = tensor.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten tensor: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert to vector: {}", e) 
        })?;

    let mut minus_one_count = 0;
    let mut zero_count = 0;
    let mut plus_one_count = 0;
    let mut invalid_count = 0;

    for &val in &values {
        match val {
            x if (x + 1.0).abs() < 1e-6 => minus_one_count += 1,
            x if x.abs() < 1e-6 => zero_count += 1,
            x if (x - 1.0).abs() < 1e-6 => plus_one_count += 1,
            _ => invalid_count += 1,
        }
    }

    let is_strictly_ternary = invalid_count == 0;
    let error_message = if !is_strictly_ternary {
        Some(format!("Found {} non-ternary values out of {}", invalid_count, values.len()))
    } else {
        None
    };

    Ok(TernaryValidationResult {
        is_strictly_ternary,
        value_distribution: TernaryDistribution {
            negative_ones: minus_one_count,
            zeros: zero_count,
            positive_ones: plus_one_count,
            total: values.len(),
        },
        threshold_used: 0.0, // To be filled by caller if available
        quantization_counts: QuantizationCounts {
            minus_one: minus_one_count,
            zero: zero_count,
            plus_one: plus_one_count,
            invalid: invalid_count,
        },
        error_message,
    })
}

/// Result of round-trip accuracy validation
#[derive(Debug, Clone)]
pub struct RoundTripValidationResult {
    /// Mean Squared Error
    pub mse: f64,
    /// Signal-to-Quantization-Noise Ratio in dB
    pub sqnr_db: f64,
    /// Cosine similarity between original and reconstructed
    pub cosine_similarity: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Peak Signal-to-Noise Ratio in dB
    pub psnr_db: f64,
    /// Whether the accuracy meets target thresholds
    pub meets_accuracy_target: bool,
    /// Detailed error analysis
    pub error_analysis: ErrorAnalysis,
}

/// Detailed error analysis
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Per-element absolute errors
    pub element_errors: Vec<f32>,
    /// Maximum error
    pub max_error: f32,
    /// 95th percentile error
    pub p95_error: f32,
    /// Distribution of errors
    pub error_histogram: HashMap<String, usize>,
}

/// Validate round-trip accuracy (quantization → dequantization)
pub fn validate_round_trip_accuracy(
    original: &Tensor,
    reconstructed: &Tensor,
    target_mse: f64,
) -> QuantizationResult<RoundTripValidationResult> {
    // Ensure tensors have the same shape
    if original.shape() != reconstructed.shape() {
        return Err(QuantizationError::ValidationError {
            reason: format!("Shape mismatch: original {:?}, reconstructed {:?}", 
                          original.shape(), reconstructed.shape())
        });
    }

    let orig_values = original.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten original: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert original to vector: {}", e) 
        })?;

    let recon_values = reconstructed.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten reconstructed: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert reconstructed to vector: {}", e) 
        })?;

    // Compute error metrics
    let mut squared_errors = Vec::new();
    let mut absolute_errors = Vec::new();
    
    for (&orig, &recon) in orig_values.iter().zip(recon_values.iter()) {
        let error = orig - recon;
        squared_errors.push((error * error) as f64);
        absolute_errors.push(error.abs());
    }

    let mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
    let mae = absolute_errors.iter().sum::<f32>() as f64 / absolute_errors.len() as f64;

    // Compute SQNR
    let signal_power: f64 = orig_values.iter().map(|&x| (x * x) as f64).sum::<f64>() 
                           / orig_values.len() as f64;
    let sqnr_db = if mse > 1e-10 {
        10.0 * (signal_power / mse).log10()
    } else {
        f64::INFINITY
    };

    // Compute PSNR
    let max_val = orig_values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())) as f64;
    let psnr_db = if mse > 1e-10 {
        20.0 * (max_val / mse.sqrt()).log10()
    } else {
        f64::INFINITY
    };

    // Compute cosine similarity
    let dot_product: f64 = orig_values.iter().zip(recon_values.iter())
        .map(|(&a, &b)| (a * b) as f64).sum();
    let orig_norm: f64 = orig_values.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    let recon_norm: f64 = recon_values.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    
    let cosine_similarity = if orig_norm > 1e-10 && recon_norm > 1e-10 {
        dot_product / (orig_norm * recon_norm)
    } else {
        1.0 // Both vectors are zero
    };

    // Error analysis
    let mut sorted_errors = absolute_errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let max_error = sorted_errors.last().copied().unwrap_or(0.0);
    let p95_index = ((sorted_errors.len() as f32) * 0.95) as usize;
    let p95_error = sorted_errors.get(p95_index.min(sorted_errors.len() - 1)).copied().unwrap_or(0.0);

    // Create error histogram
    let mut error_histogram = HashMap::new();
    for &error in &absolute_errors {
        let bucket = if error < 0.01 { "< 0.01" }
        else if error < 0.05 { "0.01-0.05" }
        else if error < 0.1 { "0.05-0.1" }
        else if error < 0.5 { "0.1-0.5" }
        else { "> 0.5" };
        
        *error_histogram.entry(bucket.to_string()).or_insert(0) += 1;
    }

    let error_analysis = ErrorAnalysis {
        element_errors: absolute_errors,
        max_error,
        p95_error,
        error_histogram,
    };

    let meets_accuracy_target = mse <= target_mse && cosine_similarity >= 0.90;

    Ok(RoundTripValidationResult {
        mse,
        sqnr_db,
        cosine_similarity,
        mae,
        psnr_db,
        meets_accuracy_target,
        error_analysis,
    })
}

/// Scaling factor validation result
#[derive(Debug, Clone)]
pub struct ScalingFactorValidationResult {
    pub computed_scale: f32,
    pub expected_scale: f32,
    pub relative_error: f64,
    pub is_positive: bool,
    pub is_finite: bool,
    pub is_reasonable_magnitude: bool,
    pub passes_validation: bool,
    pub validation_details: String,
}

/// Validate scaling factor computation
pub fn validate_scaling_factor(
    original: &Tensor,
    quantized: &Tensor,
    computed_scale: f32,
) -> QuantizationResult<ScalingFactorValidationResult> {
    // Manually compute expected scaling factor using least squares
    let orig_flat = original.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten original: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert original: {}", e) 
        })?;

    let quant_flat = quantized.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten quantized: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert quantized: {}", e) 
        })?;

    // Compute α = (W·Q) / (Q·Q)
    let numerator: f64 = orig_flat.iter().zip(quant_flat.iter())
        .map(|(&w, &q)| (w * q) as f64).sum();
    let denominator: f64 = quant_flat.iter()
        .map(|&q| (q * q) as f64).sum();

    let expected_scale = if denominator.abs() > 1e-10 {
        (numerator / denominator) as f32
    } else {
        1.0 // Default when denominator is zero
    };

    let relative_error = if expected_scale.abs() > 1e-6 {
        ((computed_scale - expected_scale) / expected_scale).abs() as f64
    } else {
        computed_scale.abs() as f64
    };

    let is_positive = computed_scale > 0.0;
    let is_finite = computed_scale.is_finite();
    let is_reasonable_magnitude = computed_scale.abs() > 1e-6 && computed_scale.abs() < 1000.0;

    let passes_validation = is_finite && 
                           is_reasonable_magnitude && 
                           relative_error < 0.1; // 10% tolerance

    let validation_details = format!(
        "Scale: {:.6}, Expected: {:.6}, RelErr: {:.6}, Positive: {}, Finite: {}, Reasonable: {}",
        computed_scale, expected_scale, relative_error, is_positive, is_finite, is_reasonable_magnitude
    );

    Ok(ScalingFactorValidationResult {
        computed_scale,
        expected_scale,
        relative_error,
        is_positive,
        is_finite,
        is_reasonable_magnitude,
        passes_validation,
        validation_details,
    })
}

/// General validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub score: f64,
    pub details: String,
    pub metrics: HashMap<String, f64>,
}

impl ValidationResult {
    pub fn success(score: f64, details: String) -> Self {
        Self {
            passed: true,
            score,
            details,
            metrics: HashMap::new(),
        }
    }

    pub fn failure(details: String) -> Self {
        Self {
            passed: false,
            score: 0.0,
            details,
            metrics: HashMap::new(),
        }
    }

    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics = metrics;
        self
    }
}

/// Validate tensor shapes match
pub fn validate_shape_preservation(original: &Tensor, processed: &Tensor) -> ValidationResult {
    if original.shape() == processed.shape() {
        ValidationResult::success(
            1.0,
            format!("Shape preserved: {:?}", original.shape())
        )
    } else {
        ValidationResult::failure(
            format!("Shape mismatch: original {:?}, processed {:?}", 
                   original.shape(), processed.shape())
        )
    }
}

/// Validate that all values are within expected bounds
pub fn validate_value_bounds(tensor: &Tensor, min_val: f32, max_val: f32) -> ValidationResult {
    match tensor.flatten_all() {
        Ok(flat) => {
            match flat.to_vec1::<f32>() {
                Ok(values) => {
                    let out_of_bounds = values.iter()
                        .filter(|&&v| v < min_val || v > max_val)
                        .count();
                    
                    if out_of_bounds == 0 {
                        ValidationResult::success(
                            1.0,
                            format!("All {} values within bounds [{}, {}]", values.len(), min_val, max_val)
                        )
                    } else {
                        ValidationResult::failure(
                            format!("{} out of {} values outside bounds [{}, {}]", 
                                   out_of_bounds, values.len(), min_val, max_val)
                        )
                    }
                }
                Err(e) => ValidationResult::failure(format!("Failed to convert tensor: {}", e))
            }
        }
        Err(e) => ValidationResult::failure(format!("Failed to flatten tensor: {}", e))
    }
}

/// Compute comprehensive tensor statistics
pub fn compute_tensor_statistics(tensor: &Tensor) -> QuantizationResult<TensorStatistics> {
    let values = tensor.flatten_all()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to flatten tensor: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| QuantizationError::TensorError { 
            reason: format!("Failed to convert tensor: {}", e) 
        })?;

    let n = values.len() as f64;
    let mean = values.iter().sum::<f32>() as f64 / n;
    
    let variance = values.iter()
        .map(|&x| ((x as f64) - mean).powi(2))
        .sum::<f64>() / n;
    
    let std_dev = variance.sqrt();
    
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let zero_count = values.iter().filter(|&&x| x.abs() < 1e-8).count();
    let sparsity = zero_count as f64 / n;
    
    let l1_norm = values.iter().map(|&x| x.abs() as f64).sum::<f64>();
    let l2_norm = values.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    
    Ok(TensorStatistics {
        mean,
        std_dev,
        min,
        max,
        sparsity,
        l1_norm,
        l2_norm,
        element_count: values.len(),
    })
}

#[derive(Debug, Clone)]
pub struct TensorStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f32,
    pub max: f32,
    pub sparsity: f64,
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub element_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::helpers::test_data::{generate_test_tensor, TestPattern, create_test_device};

    #[test]
    fn test_validate_ternary_values_all_valid() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::RandomTernary, &[100], &device).unwrap();
        
        let result = validate_ternary_values(&tensor).unwrap();
        assert!(result.is_strictly_ternary);
        assert_eq!(result.quantization_counts.invalid, 0);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_validate_ternary_values_invalid() {
        let device = create_test_device();
        // Generate non-ternary data
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &[50], &device).unwrap();
        
        let result = validate_ternary_values(&tensor).unwrap();
        assert!(!result.is_strictly_ternary);
        assert!(result.quantization_counts.invalid > 0);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_round_trip_accuracy_perfect() {
        let device = create_test_device();
        let original = generate_test_tensor(TestPattern::RandomTernary, &[50], &device).unwrap();
        let reconstructed = original.clone();
        
        let result = validate_round_trip_accuracy(&original, &reconstructed, 0.01).unwrap();
        assert!(result.mse < 1e-10);
        assert!(result.cosine_similarity > 0.999);
        assert!(result.meets_accuracy_target);
    }

    #[test]
    fn test_scaling_factor_validation() {
        let device = create_test_device();
        let original = generate_test_tensor(TestPattern::UniformDistribution, &[20], &device).unwrap();
        let quantized = generate_test_tensor(TestPattern::RandomTernary, &[20], &device).unwrap();
        
        // Use a reasonable scale factor
        let computed_scale = 1.5;
        
        let result = validate_scaling_factor(&original, &quantized, computed_scale).unwrap();
        assert!(result.is_positive);
        assert!(result.is_finite);
        assert!(result.is_reasonable_magnitude);
    }

    #[test]
    fn test_tensor_statistics() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::AllZeros, &[100], &device).unwrap();
        
        let stats = compute_tensor_statistics(&tensor).unwrap();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.sparsity, 1.0);
        assert_eq!(stats.element_count, 100);
    }
}
