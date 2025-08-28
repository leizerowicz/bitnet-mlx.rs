//! Numerical stability testing for quantization operations
//!
//! This module focuses on testing the numerical stability and robustness
//! of quantization algorithms under various challenging conditions.

use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, create_test_device,
    QuantizationMetrics, compute_quantization_metrics
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Results of numerical stability testing
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct NumericalStabilityResults {
    pub stability_tests_run: usize,
    pub stability_tests_passed: usize,
    pub numerical_issues_detected: usize,
    pub condition_results: HashMap<String, ConditionResult>,
    pub overall_stability_score: f64,
    pub critical_failures: Vec<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConditionResult {
    pub condition_name: String,
    pub test_iterations: usize,
    pub successful_iterations: usize,
    pub numerical_errors: usize,
    pub stability_metrics: StabilityMetrics,
    pub severity_level: SeverityLevel,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StabilityMetrics {
    pub max_relative_error: f64,
    pub mean_relative_error: f64,
    pub std_relative_error: f64,
    pub condition_number: f64,
    pub gradient_stability: f64,
    pub convergence_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SeverityLevel {
    Low,      // Minor numerical issues
    Medium,   // Moderate stability concerns
    High,     // Significant numerical problems
    Critical, // Severe instability that breaks functionality
}

impl NumericalStabilityResults {
    pub fn success_rate(&self) -> f64 {
        if self.stability_tests_run == 0 { 1.0 }
        else { self.stability_tests_passed as f64 / self.stability_tests_run as f64 }
    }

    pub fn has_critical_issues(&self) -> bool {
        !self.critical_failures.is_empty() ||
        self.condition_results.values()
            .any(|r| r.severity_level == SeverityLevel::Critical)
    }
}

/// Run comprehensive numerical stability tests
pub fn run_numerical_stability_tests(device: &Device) -> QuantizationResult<NumericalStabilityResults> {
    let mut results = NumericalStabilityResults::default();

    let test_conditions = vec![
        ("Extreme Values", test_extreme_values_stability),
        ("Near-Zero Values", test_near_zero_stability),
        ("High Dynamic Range", test_high_dynamic_range_stability),
        ("Precision Boundaries", test_precision_boundary_stability),
        ("Gradient Explosion", test_gradient_explosion_stability),
        ("Catastrophic Cancellation", test_catastrophic_cancellation_stability),
        ("Accumulated Rounding", test_accumulated_rounding_stability),
        ("Threshold Sensitivity", test_threshold_sensitivity_stability),
    ];

    for (condition_name, test_fn) in test_conditions {
        results.stability_tests_run += 1;

        match test_fn(device, 20) { // 20 iterations per condition
            Ok(condition_result) => {
                let is_stable = condition_result.severity_level != SeverityLevel::Critical;

                if is_stable {
                    results.stability_tests_passed += 1;
                } else {
                    results.critical_failures.push(
                        format!("Critical instability in {}", condition_name)
                    );
                }

                if condition_result.numerical_errors > 0 {
                    results.numerical_issues_detected += condition_result.numerical_errors;
                }

                results.condition_results.insert(condition_name.to_string(), condition_result);
            }
            Err(e) => {
                results.critical_failures.push(
                    format!("Failed to test {}: {}", condition_name, e)
                );
            }
        }
    }

    results.overall_stability_score = calculate_overall_stability_score(&results);

    Ok(results)
}

// Individual stability test functions

fn test_extreme_values_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Extreme Values".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();
    let extreme_values = vec![f32::MAX, f32::MIN, 1e10, -1e10, 1e-10, -1e-10];

    for i in 0..iterations {
        let extreme_value = extreme_values[i % extreme_values.len()];

        // Create tensor filled with extreme values
        let tensor = Tensor::new(vec![extreme_value; 64], device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create extreme tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    // Calculate relative error
                    let relative_error = calculate_relative_error(&tensor, &dequantized)?;

                    if relative_error.is_finite() && !relative_error.is_nan() {
                        relative_errors.push(relative_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_near_zero_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Near-Zero Values".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        let epsilon = f32::EPSILON * (10.0_f32).powi(i as i32 % 10);

        // Create tensor with values near zero
        let values: Vec<f32> = (0..64)
            .map(|_j| epsilon * (j as f32 % 2.0 * 2.0 - 1.0)) // Alternating signs
            .collect();

        let tensor = Tensor::new(values, device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create near-zero tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    // For near-zero values, use absolute error instead of relative
                    let absolute_error = calculate_absolute_error(&tensor, &dequantized)?;

                    if absolute_error.is_finite() && !absolute_error.is_nan() {
                        relative_errors.push(absolute_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_high_dynamic_range_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "High Dynamic Range".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        // Create tensor with high dynamic range (mix of large and small values)
        let mut values = Vec::new();
        for j in 0..64 {
            let magnitude = if j % 2 == 0 { 1e6 } else { 1e-6 };
            let sign = if (j / 2) % 2 == 0 { 1.0 } else { -1.0 };
            values.push(magnitude * sign * (i as f32 + 1.0));
        }

        let tensor = Tensor::new(values, device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create high dynamic range tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    let relative_error = calculate_relative_error(&tensor, &dequantized)?;

                    if relative_error.is_finite() && !relative_error.is_nan() {
                        relative_errors.push(relative_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_precision_boundary_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Precision Boundaries".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    // Test values at f32 precision boundaries
    let boundary_values = vec![
        1.0 + f32::EPSILON,
        1.0 - f32::EPSILON,
        -1.0 + f32::EPSILON,
        -1.0 - f32::EPSILON,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];

    for i in 0..iterations {
        let base_value = boundary_values[i % boundary_values.len()];

        // Create tensor with precision boundary values
        let values: Vec<f32> = (0..64)
            .map(|_j| base_value * (1.0 + f32::EPSILON * j as f32))
            .collect();

        let tensor = Tensor::new(values, device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create precision boundary tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::MedianThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    let relative_error = calculate_relative_error(&tensor, &dequantized)?;

                    if relative_error.is_finite() && !relative_error.is_nan() {
                        relative_errors.push(relative_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_gradient_explosion_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Gradient Explosion".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        // Simulate gradient explosion scenario with exponentially growing values
        let growth_factor = 1.1_f32.powi(i as i32);
        let values: Vec<f32> = (0..64)
            .map(|_j| growth_factor * (j as f32 - 32.0))
            .collect();

        let tensor = Tensor::new(values, device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create gradient explosion tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    let relative_error = calculate_relative_error(&tensor, &dequantized)?;

                    if relative_error.is_finite() && !relative_error.is_nan() {
                        relative_errors.push(relative_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_catastrophic_cancellation_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Catastrophic Cancellation".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        // Create scenarios prone to catastrophic cancellation
        let base_value = 1e7 * (i as f32 + 1.0);
        let small_diff = 1e-5 * (i as f32 + 1.0);

        let values: Vec<f32> = (0..32).flat_map(|_j| {
            vec![
                base_value + small_diff * j as f32,
                -base_value + small_diff * j as f32,
            ]
        }).collect();

        let tensor = Tensor::new(values, device)
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to create catastrophic cancellation tensor: {}", e)
            })?;

        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(quantized) => {
                if let Ok(dequantized) = quantizer.dequantize(&quantized) {
                    let relative_error = calculate_relative_error(&tensor, &dequantized)?;

                    if relative_error.is_finite() && !relative_error.is_nan() {
                        relative_errors.push(relative_error);
                        condition_result.successful_iterations += 1;
                    } else {
                        condition_result.numerical_errors += 1;
                    }
                } else {
                    condition_result.numerical_errors += 1;
                }
            }
            Err(_) => {
                condition_result.numerical_errors += 1;
            }
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_accumulated_rounding_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Accumulated Rounding".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        // Start with a base tensor
        let mut current_tensor = generate_test_tensor(
            TestPattern::NormalDistribution,
            &[64],
            device
        )?;

        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        let original_tensor = current_tensor.clone();

        // Apply multiple quantization-dequantization cycles
        for _ in 0..5 {
            match quantizer.quantize(&current_tensor) {
                Ok(quantized) => {
                    match quantizer.dequantize(&quantized) {
                        Ok(dequantized) => {
                            current_tensor = dequantized;
                        }
                        Err(_) => {
                            condition_result.numerical_errors += 1;
                            break;
                        }
                    }
                }
                Err(_) => {
                    condition_result.numerical_errors += 1;
                    break;
                }
            }
        }

        // Calculate accumulated error
        let relative_error = calculate_relative_error(&original_tensor, &current_tensor)?;

        if relative_error.is_finite() && !relative_error.is_nan() {
            relative_errors.push(relative_error);
            condition_result.successful_iterations += 1;
        } else {
            condition_result.numerical_errors += 1;
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

fn test_threshold_sensitivity_stability(device: &Device, iterations: usize) -> QuantizationResult<ConditionResult> {
    let mut condition_result = ConditionResult {
        condition_name: "Threshold Sensitivity".to_string(),
        test_iterations: iterations,
        successful_iterations: 0,
        numerical_errors: 0,
        stability_metrics: StabilityMetrics::default(),
        severity_level: SeverityLevel::Low,
    };

    let mut relative_errors = Vec::new();

    for i in 0..iterations {
        let tensor = generate_test_tensor(
            TestPattern::UniformDistribution,
            &[64],
            device
        )?;

        // Test with slightly different thresholds
        let base_threshold = 0.7;
        let threshold_1 = base_threshold + f32::EPSILON;
        let threshold_2 = base_threshold - f32::EPSILON;

        let quantizer_1 = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(threshold_1))?;
        let quantizer_2 = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(threshold_2))?;

        match (quantizer_1.quantize(&tensor), quantizer_2.quantize(&tensor)) {
            (Ok(quantized_1), Ok(quantized_2)) => {
                match (quantizer_1.dequantize(&quantized_1), quantizer_2.dequantize(&quantized_2)) {
                    (Ok(dequantized_1), Ok(dequantized_2)) => {
                        // Calculate sensitivity to threshold changes
                        let sensitivity = calculate_relative_error(&dequantized_1, &dequantized_2)?;

                        if sensitivity.is_finite() && !sensitivity.is_nan() {
                            relative_errors.push(sensitivity);
                            condition_result.successful_iterations += 1;
                        } else {
                            condition_result.numerical_errors += 1;
                        }
                    }
                    _ => condition_result.numerical_errors += 1,
                }
            }
            _ => condition_result.numerical_errors += 1,
        }
    }

    condition_result.stability_metrics = calculate_stability_metrics(&relative_errors);
    condition_result.severity_level = assess_severity_level(&condition_result.stability_metrics);

    Ok(condition_result)
}

// Helper functions

impl Default for StabilityMetrics {
    fn default() -> Self {
        Self {
            max_relative_error: 0.0,
            mean_relative_error: 0.0,
            std_relative_error: 0.0,
            condition_number: 1.0,
            gradient_stability: 1.0,
            convergence_rate: 1.0,
        }
    }
}

fn calculate_relative_error(original: &Tensor, reconstructed: &Tensor) -> QuantizationResult<f64> {
    let orig_values = original.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to flatten original: {}", e)
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to convert original: {}", e)
        })?;

    let recon_values = reconstructed.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to flatten reconstructed: {}", e)
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to convert reconstructed: {}", e)
        })?;

    if orig_values.len() != recon_values.len() {
        return Err(crate::quantization::QuantizationError::InvalidInput {
            reason: "Tensor dimension mismatch".to_string()
        });
    }

    let relative_error = orig_values.iter().zip(recon_values.iter())
        .map(|(&orig, &recon)| {
            if orig.abs() < f32::EPSILON {
                (recon - orig).abs() as f64
            } else {
                ((recon - orig) / orig).abs() as f64
            }
        })
        .fold(0.0, f64::max);

    Ok(relative_error)
}

fn calculate_absolute_error(original: &Tensor, reconstructed: &Tensor) -> QuantizationResult<f64> {
    let orig_values = original.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to flatten original: {}", e)
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to convert original: {}", e)
        })?;

    let recon_values = reconstructed.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to flatten reconstructed: {}", e)
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to convert reconstructed: {}", e)
        })?;

    let absolute_error = orig_values.iter().zip(recon_values.iter())
        .map(|(&orig, &recon)| (recon - orig).abs() as f64)
        .fold(0.0, f64::max);

    Ok(absolute_error)
}

fn calculate_stability_metrics(errors: &[f64]) -> StabilityMetrics {
    if errors.is_empty() {
        return StabilityMetrics::default();
    }

    let max_error = errors.iter().fold(0.0, |a, &b| a.max(b));
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

    let variance = errors.iter()
        .map(|&e| (e - mean_error).powi(2))
        .sum::<f64>() / errors.len() as f64;
    let std_error = variance.sqrt();

    StabilityMetrics {
        max_relative_error: max_error,
        mean_relative_error: mean_error,
        std_relative_error: std_error,
        condition_number: max_error / (mean_error + f64::EPSILON),
        gradient_stability: 1.0 / (1.0 + std_error),
        convergence_rate: 1.0 / (1.0 + max_error),
    }
}

fn assess_severity_level(metrics: &StabilityMetrics) -> SeverityLevel {
    // Define thresholds for severity assessment
    if metrics.max_relative_error > 1e2 || metrics.condition_number > 1e6 {
        SeverityLevel::Critical
    } else if metrics.max_relative_error > 1e1 || metrics.condition_number > 1e4 {
        SeverityLevel::High
    } else if metrics.max_relative_error > 1.0 || metrics.condition_number > 1e2 {
        SeverityLevel::Medium
    } else {
        SeverityLevel::Low
    }
}

fn calculate_overall_stability_score(results: &NumericalStabilityResults) -> f64 {
    if results.condition_results.is_empty() {
        return 0.0;
    }

    let total_score: f64 = results.condition_results.values()
        .map(|_condition| {
            let success_rate = condition.successful_iterations as f64 / condition.test_iterations as f64;
            let stability_penalty = match condition.severity_level {
                SeverityLevel::Low => 1.0,
                SeverityLevel::Medium => 0.7,
                SeverityLevel::High => 0.4,
                SeverityLevel::Critical => 0.1,
            };
            success_rate * stability_penalty
        })
        .sum();

    total_score / results.condition_results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_stability_testing() {
        let device = create_test_device();
        let results = run_numerical_stability_tests(&device).unwrap();

        assert!(results.stability_tests_run > 0);
        assert!(results.overall_stability_score >= 0.0 && results.overall_stability_score <= 1.0);
    }

    #[test]
    fn test_extreme_values_handling() {
        let device = create_test_device();
        let result = test_extreme_values_stability(&device, 5).unwrap();

        assert_eq!(result.test_iterations, 5);
        assert!(result.successful_iterations <= 5);
    }

    #[test]
    fn test_severity_assessment() {
        let low_severity_metrics = StabilityMetrics {
            max_relative_error: 0.1,
            condition_number: 10.0,
            ..Default::default()
        };
        assert_eq!(assess_severity_level(&low_severity_metrics), SeverityLevel::Low);

        let critical_severity_metrics = StabilityMetrics {
            max_relative_error: 1000.0,
            condition_number: 1e7,
            ..Default::default()
        };
        assert_eq!(assess_severity_level(&critical_severity_metrics), SeverityLevel::Critical);
    }
}
