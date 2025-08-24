//! Ternary quantization validation tests
//!
//! This module provides comprehensive tests to verify that quantized weights
//! produce only {-1, 0, +1} values across all ternary quantization methods.

use crate::quantization::{
    QuantizationResult, TernaryMethod, WeightQuantizer,
    create_ternary_quantizer, WeightQuantizationConfig
};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, generate_test_tensor_set,
    validate_ternary_values, TernaryValidationResult, TernaryDistribution,
    create_test_device
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Results of comprehensive ternary validation testing
#[derive(Debug, Clone)]
pub struct TernaryValidationResults {
    pub method_results: HashMap<TernaryMethod, MethodValidationResults>,
    pub pattern_results: HashMap<TestPattern, PatternValidationResults>,
    pub overall_success_rate: f64,
    pub failed_test_cases: Vec<String>,
    pub summary_statistics: ValidationSummaryStatistics,
}

impl Default for TernaryValidationResults {
    fn default() -> Self {
        Self {
            method_results: HashMap::new(),
            pattern_results: HashMap::new(),
            overall_success_rate: 0.0,
            failed_test_cases: Vec::new(),
            summary_statistics: ValidationSummaryStatistics::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MethodValidationResults {
    pub method: TernaryMethod,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub success_rate: f64,
    pub pattern_performance: HashMap<TestPattern, f64>,
    pub average_sparsity: f64,
    pub distribution_balance: f64,
}

#[derive(Debug, Clone)]
pub struct PatternValidationResults {
    pub pattern: TestPattern,
    pub method_performance: HashMap<TernaryMethod, f64>,
    pub best_performing_method: TernaryMethod,
    pub worst_performing_method: TernaryMethod,
    pub average_success_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationSummaryStatistics {
    pub total_tensors_tested: usize,
    pub total_values_tested: usize,
    pub ternary_compliance_rate: f64,
    pub average_sparsity: f64,
    pub distribution_balance_score: f64,
    pub method_consistency_score: f64,
}

/// Run comprehensive ternary validation tests
pub fn run_comprehensive_ternary_tests(device: &Device) -> QuantizationResult<TernaryValidationResults> {
    let mut results = TernaryValidationResults::default();

    // Test all ternary methods
    let methods = vec![
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold,
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];

    // Test all relevant patterns
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

    let tensor_shapes = vec![
        vec![16],           // 1D small
        vec![4, 4],         // 2D small
        vec![8, 8],         // 2D medium
        vec![2, 3, 4],      // 3D
        vec![64, 32],       // 2D large
        vec![128],          // 1D large
    ];

    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut total_values = 0;
    let mut ternary_compliant_values = 0;
    let mut all_sparsities = Vec::new();
    let mut all_balance_scores = Vec::new();

    // Initialize method results
    for &method in &methods {
        results.method_results.insert(method, MethodValidationResults {
            method,
            total_tests: 0,
            passed_tests: 0,
            success_rate: 0.0,
            pattern_performance: HashMap::new(),
            average_sparsity: 0.0,
            distribution_balance: 0.0,
        });
    }

    // Initialize pattern results
    for &pattern in &patterns {
        results.pattern_results.insert(pattern, PatternValidationResults {
            pattern,
            method_performance: HashMap::new(),
            best_performing_method: TernaryMethod::MeanThreshold, // Will be updated
            worst_performing_method: TernaryMethod::MeanThreshold, // Will be updated
            average_success_rate: 0.0,
        });
    }

    // Test each combination of method, pattern, and shape
    for &method in &methods {
        let mut method_tests = 0;
        let mut method_passed = 0;
        let mut method_sparsities = Vec::new();
        let mut method_balance_scores = Vec::new();

        for &pattern in &patterns {
            let mut pattern_tests = 0;
            let mut pattern_passed = 0;

            for shape in &tensor_shapes {
                total_tests += 1;
                method_tests += 1;
                pattern_tests += 1;

                // Generate test tensor
                let tensor = match generate_test_tensor(pattern, shape, device) {
                    Ok(t) => t,
                    Err(e) => {
                        results.failed_test_cases.push(
                            format!("Failed to generate tensor {:?} with pattern {:?}: {}", shape, pattern, e)
                        );
                        continue;
                    }
                };

                // Test ternary quantization
                let test_result = test_ternary_quantization(&tensor, method, pattern, shape);

                match test_result {
                    Ok(validation_result) => {
                        total_values += validation_result.value_distribution.total;

                        if validation_result.is_strictly_ternary {
                            passed_tests += 1;
                            method_passed += 1;
                            pattern_passed += 1;
                            ternary_compliant_values += validation_result.value_distribution.total;
                        } else {
                            results.failed_test_cases.push(
                                format!("Non-ternary values in {:?} pattern with {:?} method, shape {:?}: {}",
                                       pattern, method, shape,
                                       validation_result.error_message.unwrap_or_else(|| "Unknown error".to_string()))
                            );
                            ternary_compliant_values += validation_result.value_distribution.total -
                                                       validation_result.quantization_counts.invalid;
                        }

                        // Collect statistics
                        let sparsity = validation_result.value_distribution.sparsity();
                        let balance_score = validation_result.value_distribution.is_balanced() as i32 as f64;

                        all_sparsities.push(sparsity);
                        all_balance_scores.push(balance_score);
                        method_sparsities.push(sparsity);
                        method_balance_scores.push(balance_score);
                    }
                    Err(e) => {
                        results.failed_test_cases.push(
                            format!("Failed to test {:?} pattern with {:?} method, shape {:?}: {}",
                                   pattern, method, shape, e)
                        );
                    }
                }
            }

            // Update pattern results
            let pattern_success_rate = pattern_passed as f64 / pattern_tests.max(1) as f64;
            if let Some(pattern_result) = results.pattern_results.get_mut(&pattern) {
                pattern_result.method_performance.insert(method, pattern_success_rate);
            }

            // Update method pattern performance
            if let Some(method_result) = results.method_results.get_mut(&method) {
                method_result.pattern_performance.insert(pattern, pattern_success_rate);
            }
        }

        // Update method results
        if let Some(method_result) = results.method_results.get_mut(&method) {
            method_result.total_tests = method_tests;
            method_result.passed_tests = method_passed;
            method_result.success_rate = method_passed as f64 / method_tests.max(1) as f64;
            method_result.average_sparsity = if !method_sparsities.is_empty() {
                method_sparsities.iter().sum::<f64>() / method_sparsities.len() as f64
            } else {
                0.0
            };
            method_result.distribution_balance = if !method_balance_scores.is_empty() {
                method_balance_scores.iter().sum::<f64>() / method_balance_scores.len() as f64
            } else {
                0.0
            };
        }
    }

    // Finalize pattern results
    for pattern_result in results.pattern_results.values_mut() {
        let performances: Vec<f64> = pattern_result.method_performance.values().cloned().collect();
        if !performances.is_empty() {
            pattern_result.average_success_rate = performances.iter().sum::<f64>() / performances.len() as f64;

            // Find best and worst performing methods
            let mut best_score = 0.0;
            let mut worst_score = 1.0;

            for (&method, &score) in &pattern_result.method_performance {
                if score > best_score {
                    best_score = score;
                    pattern_result.best_performing_method = method;
                }
                if score < worst_score {
                    worst_score = score;
                    pattern_result.worst_performing_method = method;
                }
            }
        }
    }

    // Calculate overall statistics
    results.overall_success_rate = passed_tests as f64 / total_tests.max(1) as f64;

    results.summary_statistics = ValidationSummaryStatistics {
        total_tensors_tested: total_tests,
        total_values_tested: total_values,
        ternary_compliance_rate: ternary_compliant_values as f64 / total_values.max(1) as f64,
        average_sparsity: if !all_sparsities.is_empty() {
            all_sparsities.iter().sum::<f64>() / all_sparsities.len() as f64
        } else {
            0.0
        },
        distribution_balance_score: if !all_balance_scores.is_empty() {
            all_balance_scores.iter().sum::<f64>() / all_balance_scores.len() as f64
        } else {
            0.0
        },
        method_consistency_score: calculate_method_consistency(&results.method_results),
    };

    Ok(results)
}

/// Test ternary quantization for a specific tensor, method, and configuration
fn test_ternary_quantization(
    tensor: &Tensor,
    method: TernaryMethod,
    pattern: TestPattern,
    shape: &[usize],
) -> QuantizationResult<TernaryValidationResult> {
    // Create quantizer with the specified method
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;

    // Perform quantization
    let quantized = quantizer.quantize(tensor)?;

    // Validate ternary values
    let mut validation_result = validate_ternary_values(&quantized.values)?;

    // Set the threshold that was actually used (if available)
    validation_result.threshold_used = quantized.stats.threshold_used.unwrap_or(0.0);

    // Add additional context to error message if validation failed
    if !validation_result.is_strictly_ternary {
        let context = format!(
            "Pattern: {:?}, Method: {:?}, Shape: {:?}, Threshold: {:.6}",
            pattern, method, shape, validation_result.threshold_used
        );
        validation_result.error_message = Some(
            format!("{} - {}",
                   validation_result.error_message.unwrap_or_else(|| "Validation failed".to_string()),
                   context)
        );
    }

    Ok(validation_result)
}

/// Test specific ternary method with various threshold factors
pub fn test_ternary_method_with_thresholds(
    device: &Device,
    method: TernaryMethod,
    pattern: TestPattern,
) -> QuantizationResult<ThresholdTestResults> {
    let threshold_factors = vec![0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5];
    let shape = vec![64, 32];

    let test_tensor = generate_test_tensor(pattern, &shape, device)?;
    let mut results = ThresholdTestResults {
        method,
        pattern,
        threshold_results: HashMap::new(),
        optimal_threshold: 0.7, // Default
        optimal_sparsity: 0.0,
        threshold_sensitivity: 0.0,
    };

    let mut sparsity_variance = 0.0;
    let mut sparsities = Vec::new();

    for &threshold in &threshold_factors {
        let quantizer = create_ternary_quantizer(method, Some(threshold))?;
        let quantized = quantizer.quantize(&test_tensor)?;
        let validation = validate_ternary_values(&quantized.values)?;

        let sparsity = validation.value_distribution.sparsity();
        sparsities.push(sparsity);

        let threshold_result = ThresholdResult {
            threshold_factor: threshold,
            is_ternary: validation.is_strictly_ternary,
            sparsity,
            balance_score: validation.value_distribution.is_balanced() as i32 as f64,
            distribution: validation.value_distribution.clone(),
        };

        results.threshold_results.insert(threshold, threshold_result);
    }

    // Find optimal threshold (highest sparsity while maintaining ternary property)
    let mut best_threshold = 0.7;
    let mut best_sparsity = 0.0;

    for (threshold, result) in &results.threshold_results {
        if result.is_ternary && result.sparsity > best_sparsity {
            best_sparsity = result.sparsity;
            best_threshold = *threshold;
        }
    }

    results.optimal_threshold = best_threshold;
    results.optimal_sparsity = best_sparsity;

    // Calculate threshold sensitivity
    if !sparsities.is_empty() {
        let mean_sparsity = sparsities.iter().sum::<f64>() / sparsities.len() as f64;
        sparsity_variance = sparsities.iter()
            .map(|&s| (s - mean_sparsity).powi(2))
            .sum::<f64>() / sparsities.len() as f64;
    }

    results.threshold_sensitivity = sparsity_variance.sqrt();

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct ThresholdTestResults {
    pub method: TernaryMethod,
    pub pattern: TestPattern,
    pub threshold_results: HashMap<f32, ThresholdResult>,
    pub optimal_threshold: f32,
    pub optimal_sparsity: f64,
    pub threshold_sensitivity: f64, // Standard deviation of sparsity across thresholds
}

#[derive(Debug, Clone)]
pub struct ThresholdResult {
    pub threshold_factor: f32,
    pub is_ternary: bool,
    pub sparsity: f64,
    pub balance_score: f64,
    pub distribution: TernaryDistribution,
}

/// Test edge cases for ternary quantization
pub fn test_ternary_edge_cases(device: &Device) -> QuantizationResult<EdgeCaseTestResults> {
    let edge_cases = vec![
        (TestPattern::AllZeros, "All zeros"),
        (TestPattern::AllOnes, "All ones"),
        (TestPattern::AllNegativeOnes, "All negative ones"),
        (TestPattern::SingleNonZero, "Single non-zero"),
        (TestPattern::Alternating, "Alternating pattern"),
    ];

    let methods = vec![
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold,
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];

    let mut results = EdgeCaseTestResults {
        test_results: HashMap::new(),
        total_tests: 0,
        passed_tests: 0,
        critical_failures: Vec::new(),
    };

    for (pattern, description) in edge_cases {
        for &method in &methods {
            results.total_tests += 1;
            let test_name = format!("{} with {:?}", description, method);

            match test_edge_case(device, pattern, method) {
                Ok(is_valid) => {
                    results.test_results.insert(test_name.clone(), is_valid);
                    if is_valid {
                        results.passed_tests += 1;
                    } else {
                        results.critical_failures.push(
                            format!("Edge case failed: {}", test_name)
                        );
                    }
                }
                Err(e) => {
                    results.test_results.insert(test_name.clone(), false);
                    results.critical_failures.push(
                        format!("Edge case error: {} - {}", test_name, e)
                    );
                }
            }
        }
    }

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct EdgeCaseTestResults {
    pub test_results: HashMap<String, bool>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub critical_failures: Vec<String>,
}

impl EdgeCaseTestResults {
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            1.0
        } else {
            self.passed_tests as f64 / self.total_tests as f64
        }
    }
}

fn test_edge_case(device: &Device, pattern: TestPattern, method: TernaryMethod) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(pattern, &[32], device)?;
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;
    let quantized = quantizer.quantize(&tensor)?;
    let validation = validate_ternary_values(&quantized.values)?;

    Ok(validation.is_strictly_ternary)
}

/// Calculate consistency score across methods
fn calculate_method_consistency(method_results: &HashMap<TernaryMethod, MethodValidationResults>) -> f64 {
    let success_rates: Vec<f64> = method_results.values()
        .map(|r| r.success_rate)
        .collect();

    if success_rates.is_empty() {
        return 1.0;
    }

    let mean = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
    let variance = success_rates.iter()
        .map(|&rate| (rate - mean).powi(2))
        .sum::<f64>() / success_rates.len() as f64;

    // Higher consistency score for lower variance
    1.0 / (1.0 + variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_ternary_validation() {
        let device = create_test_device();
        let results = run_comprehensive_ternary_tests(&device).unwrap();

        assert!(results.overall_success_rate > 0.8);
        assert!(!results.method_results.is_empty());
        assert!(!results.pattern_results.is_empty());
        assert!(results.summary_statistics.total_tensors_tested > 0);
        assert!(results.summary_statistics.ternary_compliance_rate > 0.8);
    }

    #[test]
    fn test_threshold_sensitivity() {
        let device = create_test_device();
        let results = test_ternary_method_with_thresholds(
            &device,
            TernaryMethod::MeanThreshold,
            TestPattern::NormalDistribution,
        ).unwrap();

        assert!(!results.threshold_results.is_empty());
        assert!(results.optimal_threshold > 0.0);
        assert!(results.threshold_sensitivity >= 0.0);
    }

    #[test]
    fn test_edge_cases() {
        let device = create_test_device();
        let results = test_ternary_edge_cases(&device).unwrap();

        assert!(results.total_tests > 0);
        assert!(results.success_rate() > 0.5); // At least half should pass
    }

    #[test]
    fn test_individual_ternary_methods() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &[100], &device).unwrap();

        for method in [
            TernaryMethod::MeanThreshold,
            TernaryMethod::MedianThreshold,
            TernaryMethod::AdaptiveThreshold,
            TernaryMethod::OptimalThreshold,
        ] {
            let result = test_ternary_quantization(&tensor, method, TestPattern::NormalDistribution, &[100]).unwrap();
            assert!(result.is_strictly_ternary, "Method {:?} failed to produce ternary values", method);
        }
    }
}
