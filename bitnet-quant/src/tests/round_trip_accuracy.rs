//! Round-trip accuracy testing for quantization operations
//!
//! This module tests the accuracy of quantization → dequantization operations,
//! measuring MSE, SQNR, cosine similarity, and other quality metrics.

use crate::quantization::{
    QuantizationResult, TernaryMethod, WeightQuantizer,
    create_ternary_quantizer, QuantizationError
};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, generate_test_tensor_set,
    validate_round_trip_accuracy, RoundTripValidationResult,
    QuantizationMetrics, QualityThresholds, QualityAssessment,
    create_test_device
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Results of comprehensive round-trip accuracy testing
#[derive(Debug, Clone)]
pub struct RoundTripResults {
    pub pattern_results: HashMap<TestPattern, PatternRoundTripResults>,
    pub method_results: HashMap<TernaryMethod, MethodRoundTripResults>,
    pub overall_metrics: OverallRoundTripMetrics,
    pub quality_assessment: QualityAssessment,
    pub failed_patterns: Vec<String>,
    pub overall_success_rate: f64,
}

impl Default for RoundTripResults {
    fn default() -> Self {
        Self {
            pattern_results: HashMap::new(),
            method_results: HashMap::new(),
            overall_metrics: OverallRoundTripMetrics::default(),
            quality_assessment: QualityAssessment {
                overall_passed: false,
                passed_criteria: Vec::new(),
                failed_criteria: Vec::new(),
                score: 0.0,
            },
            failed_patterns: Vec::new(),
            overall_success_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternRoundTripResults {
    pub pattern: TestPattern,
    pub method_performance: HashMap<TernaryMethod, RoundTripValidationResult>,
    pub best_method: TernaryMethod,
    pub worst_method: TernaryMethod,
    pub average_mse: f64,
    pub average_sqnr_db: f64,
    pub average_cosine_similarity: f64,
    pub meets_target_accuracy: bool,
}

#[derive(Debug, Clone)]
pub struct MethodRoundTripResults {
    pub method: TernaryMethod,
    pub pattern_performance: HashMap<TestPattern, RoundTripValidationResult>,
    pub average_metrics: QuantizationMetrics,
    pub consistency_score: f64,
    pub reliability_rating: ReliabilityRating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityRating {
    Excellent,  // > 95% success rate
    Good,       // > 85% success rate
    Fair,       // > 70% success rate
    Poor,       // <= 70% success rate
}

#[derive(Debug, Clone, Default)]
pub struct OverallRoundTripMetrics {
    pub total_tests: usize,
    pub successful_tests: usize,
    pub average_mse: f64,
    pub average_sqnr_db: f64,
    pub average_cosine_similarity: f64,
    pub average_compression_ratio: f64,
    pub best_performing_combination: Option<(TestPattern, TernaryMethod)>,
    pub worst_performing_combination: Option<(TestPattern, TernaryMethod)>,
}

/// Run comprehensive round-trip accuracy tests
pub fn run_round_trip_tests(device: &Device, target_mse: f64) -> QuantizationResult<RoundTripResults> {
    let mut results = RoundTripResults::default();

    // Test patterns
    let patterns = vec![
        TestPattern::NormalDistribution,
        TestPattern::UniformDistribution,
        TestPattern::SparseWeights,
        TestPattern::OutlierHeavy,
        TestPattern::SmallValues,
        TestPattern::LargeValues,
        TestPattern::BimodalDistribution,
        TestPattern::ExponentialDistribution,
    ];

    // Ternary methods
    let methods = vec![
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold,
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];

    let tensor_shapes = vec![
        vec![64],           // 1D
        vec![8, 8],         // 2D square
        vec![32, 16],       // 2D rectangle
        vec![4, 4, 4],      // 3D cube
        vec![128],          // 1D large
    ];

    let mut all_validations = Vec::new();
    let mut successful_tests = 0;
    let mut total_tests = 0;

    let mut best_mse = f64::INFINITY;
    let mut worst_mse = 0.0;
    let mut best_combination = None;
    let mut worst_combination = None;

    // Initialize pattern and method results
    for &pattern in &patterns {
        results.pattern_results.insert(pattern, PatternRoundTripResults {
            pattern,
            method_performance: HashMap::new(),
            best_method: TernaryMethod::MeanThreshold, // Will be updated
            worst_method: TernaryMethod::MeanThreshold, // Will be updated
            average_mse: f64::INFINITY,
            average_sqnr_db: 0.0,
            average_cosine_similarity: 0.0,
            meets_target_accuracy: false,
        });
    }

    for &method in &methods {
        results.method_results.insert(method, MethodRoundTripResults {
            method,
            pattern_performance: HashMap::new(),
            average_metrics: QuantizationMetrics {
                mse: 0.0,
                rmse: 0.0,
                mae: 0.0,
                sqnr_db: 0.0,
                psnr_db: 0.0,
                cosine_similarity: 0.0,
                correlation_coefficient: 0.0,
                ssim: 0.0,
                compression_ratio: 20.25, // Approximate for 1.58-bit
                quantized_entropy: 0.0,
                dynamic_range_db: 0.0,
            },
            consistency_score: 0.0,
            reliability_rating: ReliabilityRating::Poor,
        });
    }

    // Test each combination
    for &pattern in &patterns {
        for &method in &methods {
            for shape in &tensor_shapes {
                total_tests += 1;

                match test_round_trip_accuracy(device, pattern, method, shape, target_mse) {
                    Ok(validation_result) => {
                        all_validations.push(validation_result.clone());

                        if validation_result.meets_accuracy_target {
                            successful_tests += 1;
                        } else {
                            results.failed_patterns.push(
                                format!("Pattern: {:?}, Method: {:?}, Shape: {:?} - MSE: {:.6}, Target: {:.6}",
                                       pattern, method, shape, validation_result.mse, target_mse)
                            );
                        }

                        // Track best and worst combinations
                        if validation_result.mse < best_mse {
                            best_mse = validation_result.mse;
                            best_combination = Some((pattern, method));
                        }
                        if validation_result.mse > worst_mse {
                            worst_mse = validation_result.mse;
                            worst_combination = Some((pattern, method));
                        }

                        // Update pattern results
                        if let Some(pattern_result) = results.pattern_results.get_mut(&pattern) {
                            pattern_result.method_performance.insert(method, validation_result.clone());
                        }

                        // Update method results
                        if let Some(method_result) = results.method_results.get_mut(&method) {
                            method_result.pattern_performance.insert(pattern, validation_result.clone());
                        }
                    }
                    Err(e) => {
                        results.failed_patterns.push(
                            format!("Test failed: Pattern: {:?}, Method: {:?}, Shape: {:?} - Error: {}",
                                   pattern, method, shape, e)
                        );
                    }
                }
            }
        }
    }

    // Finalize pattern results
    for pattern_result in results.pattern_results.values_mut() {
        let validations: Vec<&RoundTripValidationResult> = pattern_result.method_performance.values().collect();

        if !validations.is_empty() {
            pattern_result.average_mse = validations.iter().map(|v| v.mse).sum::<f64>() / validations.len() as f64;
            pattern_result.average_sqnr_db = validations.iter().map(|v| v.sqnr_db).sum::<f64>() / validations.len() as f64;
            pattern_result.average_cosine_similarity = validations.iter().map(|v| v.cosine_similarity).sum::<f64>() / validations.len() as f64;
            pattern_result.meets_target_accuracy = validations.iter().all(|v| v.meets_accuracy_target);

            // Find best and worst methods for this pattern
            let mut best_mse = f64::INFINITY;
            let mut worst_mse = 0.0;

            for (&method, validation) in &pattern_result.method_performance {
                if validation.mse < best_mse {
                    best_mse = validation.mse;
                    pattern_result.best_method = method;
                }
                if validation.mse > worst_mse {
                    worst_mse = validation.mse;
                    pattern_result.worst_method = method;
                }
            }
        }
    }

    // Finalize method results
    for method_result in results.method_results.values_mut() {
        let validations: Vec<&RoundTripValidationResult> = method_result.pattern_performance.values().collect();

        if !validations.is_empty() {
            let success_count = validations.iter().filter(|v| v.meets_accuracy_target).count();
            let success_rate = success_count as f64 / validations.len() as f64;

            method_result.reliability_rating = match success_rate {
                r if r > 0.95 => ReliabilityRating::Excellent,
                r if r > 0.85 => ReliabilityRating::Good,
                r if r > 0.70 => ReliabilityRating::Fair,
                _ => ReliabilityRating::Poor,
            };

            // Calculate average metrics
            method_result.average_metrics.mse = validations.iter().map(|v| v.mse).sum::<f64>() / validations.len() as f64;
            method_result.average_metrics.sqnr_db = validations.iter().map(|v| v.sqnr_db).sum::<f64>() / validations.len() as f64;
            method_result.average_metrics.cosine_similarity = validations.iter().map(|v| v.cosine_similarity).sum::<f64>() / validations.len() as f64;
            method_result.average_metrics.mae = validations.iter().map(|v| v.mae).sum::<f64>() / validations.len() as f64;
            method_result.average_metrics.psnr_db = validations.iter().map(|v| v.psnr_db).sum::<f64>() / validations.len() as f64;

            // Calculate consistency score (inverse of MSE variance)
            let mean_mse = method_result.average_metrics.mse;
            let mse_variance = validations.iter()
                .map(|v| (v.mse - mean_mse).powi(2))
                .sum::<f64>() / validations.len() as f64;
            method_result.consistency_score = 1.0 / (1.0 + mse_variance);
        }
    }

    // Calculate overall metrics
    results.overall_metrics = OverallRoundTripMetrics {
        total_tests,
        successful_tests,
        average_mse: if !all_validations.is_empty() {
            all_validations.iter().map(|v| v.mse).sum::<f64>() / all_validations.len() as f64
        } else { 0.0 },
        average_sqnr_db: if !all_validations.is_empty() {
            all_validations.iter().map(|v| v.sqnr_db).sum::<f64>() / all_validations.len() as f64
        } else { 0.0 },
        average_cosine_similarity: if !all_validations.is_empty() {
            all_validations.iter().map(|v| v.cosine_similarity).sum::<f64>() / all_validations.len() as f64
        } else { 0.0 },
        average_compression_ratio: 20.25, // BitNet 1.58-bit compression ratio
        best_performing_combination: best_combination,
        worst_performing_combination: worst_combination,
    };

    results.overall_success_rate = if total_tests > 0 {
        successful_tests as f64 / total_tests as f64
    } else {
        0.0
    };

    // Perform quality assessment
    if !all_validations.is_empty() {
        let average_metrics = QuantizationMetrics {
            mse: results.overall_metrics.average_mse,
            rmse: results.overall_metrics.average_mse.sqrt(),
            mae: all_validations.iter().map(|v| v.mae).sum::<f64>() / all_validations.len() as f64,
            sqnr_db: results.overall_metrics.average_sqnr_db,
            psnr_db: all_validations.iter().map(|v| v.psnr_db).sum::<f64>() / all_validations.len() as f64,
            cosine_similarity: results.overall_metrics.average_cosine_similarity,
            correlation_coefficient: 0.0, // Would need to be calculated separately
            ssim: 0.0, // Would need to be calculated separately
            compression_ratio: 20.25,
            quantized_entropy: 0.0,
            dynamic_range_db: 0.0,
        };

        let thresholds = QualityThresholds {
            max_mse: target_mse,
            min_sqnr_db: 20.0,
            min_cosine_similarity: 0.90,
            min_correlation: 0.85,
            min_compression_ratio: 15.0,
            max_mae: 0.1,
            min_psnr_db: 25.0,
        };

        results.quality_assessment = average_metrics.meets_quality_thresholds(&thresholds);
    }

    Ok(results)
}

/// Test round-trip accuracy for a specific configuration
fn test_round_trip_accuracy(
    device: &Device,
    pattern: TestPattern,
    method: TernaryMethod,
    shape: &[usize],
    target_mse: f64,
) -> QuantizationResult<RoundTripValidationResult> {
    // Generate test tensor
    let original_tensor = generate_test_tensor(pattern, shape, device)?;

    // Create quantizer
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;

    // Perform quantization
    let quantized = quantizer.quantize(&original_tensor)?;

    // Perform dequantization
    let dequantized = quantizer.dequantize(&quantized)?;

    // Validate round-trip accuracy
    validate_round_trip_accuracy(&original_tensor, &dequantized, target_mse)
}

/// Test round-trip accuracy with different threshold factors
pub fn test_threshold_impact_on_accuracy(
    device: &Device,
    method: TernaryMethod,
    pattern: TestPattern,
) -> QuantizationResult<ThresholdAccuracyResults> {
    let threshold_factors = vec![0.3, 0.5, 0.7, 0.9, 1.1, 1.3];
    let shape = vec![64, 32];
    let target_mse = 0.01;

    let original_tensor = generate_test_tensor(pattern, &shape, device)?;
    let mut results = ThresholdAccuracyResults {
        method,
        pattern,
        threshold_accuracy: HashMap::new(),
        optimal_threshold_for_accuracy: 0.7,
        accuracy_sensitivity: 0.0,
    };

    let mut mse_values = Vec::new();

    for &threshold in &threshold_factors {
        let quantizer = create_ternary_quantizer(method, Some(threshold))?;
        let quantized = quantizer.quantize(&original_tensor)?;
        let dequantized = quantizer.dequantize(&quantized)?;

        let validation = validate_round_trip_accuracy(&original_tensor, &dequantized, target_mse)?;
        mse_values.push(validation.mse);

        results.threshold_accuracy.insert(threshold, validation);
    }

    // Find optimal threshold (minimum MSE)
    let mut best_threshold = 0.7;
    let mut best_mse = f64::INFINITY;

    for (threshold, validation) in &results.threshold_accuracy {
        if validation.mse < best_mse {
            best_mse = validation.mse;
            best_threshold = *threshold;
        }
    }

    results.optimal_threshold_for_accuracy = best_threshold;

    // Calculate accuracy sensitivity (MSE variance)
    if !mse_values.is_empty() {
        let mean_mse = mse_values.iter().sum::<f64>() / mse_values.len() as f64;
        let mse_variance = mse_values.iter()
            .map(|&mse| (mse - mean_mse).powi(2))
            .sum::<f64>() / mse_values.len() as f64;
        results.accuracy_sensitivity = mse_variance.sqrt();
    }

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct ThresholdAccuracyResults {
    pub method: TernaryMethod,
    pub pattern: TestPattern,
    pub threshold_accuracy: HashMap<f32, RoundTripValidationResult>,
    pub optimal_threshold_for_accuracy: f32,
    pub accuracy_sensitivity: f64, // Standard deviation of MSE across thresholds
}

/// Test round-trip accuracy across different tensor sizes
pub fn test_size_impact_on_accuracy(
    device: &Device,
    method: TernaryMethod,
    pattern: TestPattern,
) -> QuantizationResult<SizeAccuracyResults> {
    let sizes = vec![
        vec![16],           // Small 1D
        vec![4, 4],         // Small 2D
        vec![8, 8],         // Medium 2D
        vec![16, 16],       // Large 2D
        vec![32, 32],       // Very large 2D
        vec![2, 8, 8],      // 3D
        vec![4, 8, 8],      // Larger 3D
    ];

    let target_mse = 0.01;
    let mut results = SizeAccuracyResults {
        method,
        pattern,
        size_accuracy: HashMap::new(),
        size_scaling_behavior: SizeScalingBehavior::Unknown,
    };

    let mut size_mse_pairs = Vec::new();

    for shape in sizes {
        let total_elements: usize = shape.iter().product();
        let validation = test_round_trip_accuracy(device, pattern, method, &shape, target_mse)?;

        size_mse_pairs.push((total_elements, validation.mse));
        results.size_accuracy.insert(shape.clone(), validation);
    }

    // Analyze size scaling behavior
    results.size_scaling_behavior = analyze_size_scaling(&size_mse_pairs);

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct SizeAccuracyResults {
    pub method: TernaryMethod,
    pub pattern: TestPattern,
    pub size_accuracy: HashMap<Vec<usize>, RoundTripValidationResult>,
    pub size_scaling_behavior: SizeScalingBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeScalingBehavior {
    Improving,    // Accuracy improves with size
    Degrading,    // Accuracy degrades with size
    Stable,       // Accuracy remains stable
    Unknown,      // Unclear pattern
}

fn analyze_size_scaling(size_mse_pairs: &[(usize, f64)]) -> SizeScalingBehavior {
    if size_mse_pairs.len() < 3 {
        return SizeScalingBehavior::Unknown;
    }

    // Calculate correlation between size and MSE
    let n = size_mse_pairs.len() as f64;
    let size_sum: f64 = size_mse_pairs.iter().map(|(s, _)| *s as f64).sum();
    let mse_sum: f64 = size_mse_pairs.iter().map(|(_, mse)| *mse).sum();
    let size_mean = size_sum / n;
    let mse_mean = mse_sum / n;

    let numerator: f64 = size_mse_pairs.iter()
        .map(|(s, mse)| (*s as f64 - size_mean) * (*mse - mse_mean))
        .sum();

    let size_variance: f64 = size_mse_pairs.iter()
        .map(|(s, _)| (*s as f64 - size_mean).powi(2))
        .sum();

    let mse_variance: f64 = size_mse_pairs.iter()
        .map(|(_, mse)| (*mse - mse_mean).powi(2))
        .sum();

    if size_variance > 0.0 && mse_variance > 0.0 {
        let correlation = numerator / (size_variance.sqrt() * mse_variance.sqrt());

        if correlation > 0.5 {
            SizeScalingBehavior::Degrading // Positive correlation: larger size → higher MSE
        } else if correlation < -0.5 {
            SizeScalingBehavior::Improving // Negative correlation: larger size → lower MSE
        } else {
            SizeScalingBehavior::Stable // Weak correlation
        }
    } else {
        SizeScalingBehavior::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_accuracy_comprehensive() {
        let device = create_test_device();
        let target_mse = 0.05; // Relaxed for testing

        let results = run_round_trip_tests(&device, target_mse).unwrap();

        assert!(results.overall_success_rate > 0.5); // At least 50% should pass with relaxed threshold
        assert!(!results.pattern_results.is_empty());
        assert!(!results.method_results.is_empty());
        assert!(results.overall_metrics.total_tests > 0);
    }

    #[test]
    fn test_individual_round_trip() {
        let device = create_test_device();
        let result = test_round_trip_accuracy(
            &device,
            TestPattern::UniformDistribution,
            TernaryMethod::OptimalThreshold,
            &[32, 32],
            0.05,
        ).unwrap();

        assert!(result.mse >= 0.0);
        assert!(result.cosine_similarity >= -1.0 && result.cosine_similarity <= 1.0);
        assert!(result.sqnr_db.is_finite() || result.sqnr_db == f64::INFINITY);
    }

    #[test]
    fn test_threshold_impact() {
        let device = create_test_device();
        let results = test_threshold_impact_on_accuracy(
            &device,
            TernaryMethod::MeanThreshold,
            TestPattern::NormalDistribution,
        ).unwrap();

        assert!(!results.threshold_accuracy.is_empty());
        assert!(results.optimal_threshold_for_accuracy > 0.0);
        assert!(results.accuracy_sensitivity >= 0.0);
    }

    #[test]
    fn test_size_impact() {
        let device = create_test_device();
        let results = test_size_impact_on_accuracy(
            &device,
            TernaryMethod::AdaptiveThreshold,
            TestPattern::SparseWeights,
        ).unwrap();

        assert!(!results.size_accuracy.is_empty());
        assert!(results.size_scaling_behavior != SizeScalingBehavior::Unknown);
    }

    #[test]
    fn test_quality_assessment() {
        let device = create_test_device();
        let results = run_round_trip_tests(&device, 0.01).unwrap();

        // Quality assessment should be computed
        assert!(results.quality_assessment.score >= 0.0 && results.quality_assessment.score <= 1.0);
        let total_criteria = results.quality_assessment.passed_criteria.len() + results.quality_assessment.failed_criteria.len();
        assert!(total_criteria > 0);
    }
}
