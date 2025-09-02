//! Scaling factor validation tests
//!
//! This module tests the mathematical correctness of scaling factor computations
//! used in BitNet quantization, including least-squares optimization and numerical stability.

use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, validate_scaling_factor,
    ScalingFactorValidationResult, create_test_device
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Results of comprehensive scaling factor validation
#[derive(Debug, Clone, Default)]
pub struct ScalingFactorResults {
    pub mathematical_correctness_rate: f64,
    pub numerical_stability_rate: f64,
    pub bounds_validation_rate: f64,
    pub overall_success_rate: f64,
    pub failed_validations: Vec<String>,
    pub method_performance: HashMap<TernaryMethod, MethodScalingResults>,
    pub pattern_performance: HashMap<TestPattern, PatternScalingResults>,
}

#[derive(Debug, Clone)]
pub struct MethodScalingResults {
    pub method: TernaryMethod,
    pub correctness_rate: f64,
    pub stability_rate: f64,
    pub average_relative_error: f64,
    pub worst_case_error: f64,
}

#[derive(Debug, Clone)]
pub struct PatternScalingResults {
    pub pattern: TestPattern,
    pub average_accuracy: f64,
    pub stability_across_methods: f64,
    pub challenging_cases: Vec<String>,
}

/// Run comprehensive scaling factor validation tests
pub fn run_scaling_factor_tests(device: &Device) -> QuantizationResult<ScalingFactorResults> {
    let mut results = ScalingFactorResults::default();

    let methods = vec![
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold,
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];

    let patterns = vec![
        TestPattern::NormalDistribution,
        TestPattern::UniformDistribution,
        TestPattern::SparseWeights,
        TestPattern::OutlierHeavy,
        TestPattern::SmallValues,
        TestPattern::LargeValues,
        TestPattern::AllZeros,
        TestPattern::AllOnes,
    ];

    let mut total_tests = 0;
    let mut mathematical_passed = 0;
    let mut stability_passed = 0;
    let mut bounds_passed = 0;

    // Test each method-pattern combination
    for &method in &methods {
        let mut method_tests = 0;
        let mut method_passed = 0;
        let mut method_errors = Vec::new();

        for &pattern in &patterns {
            for shape in &[vec![32], vec![8, 8], vec![4, 4, 4]] {
                total_tests += 1;
                method_tests += 1;

                match test_scaling_factor_correctness(device, pattern, method, shape) {
                    Ok(validation) => {
                        if validation.passes_validation {
                            mathematical_passed += 1;
                            method_passed += 1;
                        } else {
                            results.failed_validations.push(
                                format!("Pattern: {:?}, Method: {:?}, Shape: {:?} - {}",
                                       pattern, method, shape, validation.validation_details)
                            );
                        }

                        if validation.is_finite && validation.is_reasonable_magnitude {
                            stability_passed += 1;
                        }

                        if validation.is_positive && validation.is_finite {
                            bounds_passed += 1;
                        }

                        method_errors.push(validation.relative_error);
                    }
                    Err(e) => {
                        results.failed_validations.push(
                            format!("Test error: Pattern: {:?}, Method: {:?}, Shape: {:?} - {}",
                                   pattern, method, shape, e)
                        );
                    }
                }
            }
        }

        // Store method results
        let method_result = MethodScalingResults {
            method,
            correctness_rate: method_passed as f64 / method_tests as f64,
            stability_rate: stability_passed as f64 / total_tests as f64,
            average_relative_error: method_errors.iter().sum::<f64>() / method_errors.len().max(1) as f64,
            worst_case_error: method_errors.iter().fold(0.0f64, |a, &b| a.max(b)),
        };
        results.method_performance.insert(method, method_result);
    }

    // Calculate overall rates
    results.mathematical_correctness_rate = mathematical_passed as f64 / total_tests as f64;
    results.numerical_stability_rate = stability_passed as f64 / total_tests as f64;
    results.bounds_validation_rate = bounds_passed as f64 / total_tests as f64;
    results.overall_success_rate = (mathematical_passed + stability_passed + bounds_passed) as f64 / (total_tests * 3) as f64;

    Ok(results)
}

fn test_scaling_factor_correctness(
    device: &Device,
    pattern: TestPattern,
    method: TernaryMethod,
    shape: &[usize],
) -> QuantizationResult<ScalingFactorValidationResult> {
    let original = generate_test_tensor(pattern, shape, device)?;
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;
    let quantized_result = quantizer.quantize(&original)?;

    let computed_scale = quantized_result.stats.scale_factor;

    validate_scaling_factor(&original, &quantized_result.values, computed_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_factor_validation() {
        let device = create_test_device();
        let results = run_scaling_factor_tests(&device).unwrap();

        assert!(results.overall_success_rate > 0.7);
        assert!(!results.method_performance.is_empty());
    }
}
