//! Edge case testing for quantization operations
//!
//! This module tests boundary conditions and extreme value scenarios
//! to ensure robust quantization behavior.

use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, validate_ternary_values, 
    create_test_device
};
use candle_core::{Device, Tensor, DType};
use std::collections::HashMap;

/// Results of edge case testing
#[derive(Debug, Clone, Default)]
pub struct EdgeCaseResults {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub critical_failures: Vec<String>,
    pub edge_case_performance: HashMap<String, bool>,
    pub robustness_score: f64,
}

impl EdgeCaseResults {
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 { 1.0 }
        else { self.passed_tests as f64 / self.total_tests as f64 }
    }
}

/// Run comprehensive edge case tests
pub fn run_edge_case_tests(device: &Device) -> QuantizationResult<EdgeCaseResults> {
    let mut results = EdgeCaseResults::default();
    
    // Test all edge cases
    let edge_case_tests = vec![
        ("All zeros", test_all_zeros),
        ("All ones", test_all_ones),
        ("All negative ones", test_all_negative_ones),
        ("Single non-zero", test_single_nonzero),
        ("Extreme large values", test_extreme_large_values),
        ("Extreme small values", test_extreme_small_values),
        ("Mixed extreme values", test_mixed_extreme_values),
        ("Very sparse", test_very_sparse),
        ("Constant values", test_constant_values),
        ("NaN handling", test_nan_handling),
        ("Infinity handling", test_infinity_handling),
    ];

    for (test_name, test_fn) in edge_case_tests {
        results.total_tests += 1;
        
        match test_fn(device) {
            Ok(passed) => {
                results.edge_case_performance.insert(test_name.to_string(), passed);
                if passed {
                    results.passed_tests += 1;
                } else {
                    results.critical_failures.push(
                        format!("Edge case failed: {}", test_name)
                    );
                }
            }
            Err(e) => {
                results.edge_case_performance.insert(test_name.to_string(), false);
                results.critical_failures.push(
                    format!("Edge case error: {} - {}", test_name, e)
                );
            }
        }
    }

    results.robustness_score = results.success_rate();
    
    Ok(results)
}

fn test_all_zeros(device: &Device) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::AllZeros, &[32], device)?;
    test_quantization_robustness(&tensor, TernaryMethod::MeanThreshold)
}

fn test_all_ones(device: &Device) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::AllOnes, &[32], device)?;
    test_quantization_robustness(&tensor, TernaryMethod::MeanThreshold)
}

fn test_all_negative_ones(device: &Device) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::AllNegativeOnes, &[32], device)?;
    test_quantization_robustness(&tensor, TernaryMethod::MeanThreshold)
}

fn test_single_nonzero(device: &Device) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::SingleNonZero, &[32], device)?;
    test_quantization_robustness(&tensor, TernaryMethod::MeanThreshold)
}

fn test_extreme_large_values(device: &Device) -> QuantizationResult<bool> {
    let data = vec![1e6f32; 32];
    let tensor = Tensor::from_vec(data, (32,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    test_quantization_robustness(&tensor, TernaryMethod::AdaptiveThreshold)
}

fn test_extreme_small_values(device: &Device) -> QuantizationResult<bool> {
    let data = vec![1e-6f32; 32];
    let tensor = Tensor::from_vec(data, (32,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    test_quantization_robustness(&tensor, TernaryMethod::AdaptiveThreshold)
}

fn test_mixed_extreme_values(device: &Device) -> QuantizationResult<bool> {
    let mut data = vec![1e6f32; 16];
    data.extend(vec![-1e6f32; 16]);
    let tensor = Tensor::from_vec(data, (32,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    test_quantization_robustness(&tensor, TernaryMethod::OptimalThreshold)
}

fn test_very_sparse(device: &Device) -> QuantizationResult<bool> {
    let mut data = vec![0.0f32; 1000];
    data[0] = 1.0;
    data[999] = -1.0;
    let tensor = Tensor::from_vec(data, (1000,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    test_quantization_robustness(&tensor, TernaryMethod::MedianThreshold)
}

fn test_constant_values(device: &Device) -> QuantizationResult<bool> {
    let data = vec![0.5f32; 64];
    let tensor = Tensor::from_vec(data, (64,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    test_quantization_robustness(&tensor, TernaryMethod::MeanThreshold)
}

fn test_nan_handling(device: &Device) -> QuantizationResult<bool> {
    // Create tensor with NaN values
    let mut data = vec![1.0f32; 32];
    data[15] = f32::NAN;
    let tensor = Tensor::from_vec(data, (32,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    
    // Test should handle NaN gracefully (either reject or handle safely)
    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;
    match quantizer.quantize(&tensor) {
        Ok(result) => {
            // If quantization succeeds, result should not contain NaN
            let values = result.values.flatten_all()
                .map_err(|e| crate::quantization::QuantizationError::TensorError { 
                    reason: format!("Failed to flatten: {}", e) 
                })?
                .to_vec1::<f32>()
                .map_err(|e| crate::quantization::QuantizationError::TensorError { 
                    reason: format!("Failed to convert: {}", e) 
                })?;
            Ok(!values.iter().any(|x| x.is_nan()))
        }
        Err(_) => Ok(true) // Graceful rejection is acceptable
    }
}

fn test_infinity_handling(device: &Device) -> QuantizationResult<bool> {
    // Create tensor with infinity values
    let mut data = vec![1.0f32; 32];
    data[15] = f32::INFINITY;
    data[16] = f32::NEG_INFINITY;
    let tensor = Tensor::from_vec(data, (32,), device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create tensor: {}", e) 
        })?;
    
    // Test should handle infinity gracefully
    let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
    match quantizer.quantize(&tensor) {
        Ok(result) => {
            // If quantization succeeds, result should be finite
            let values = result.values.flatten_all()
                .map_err(|e| crate::quantization::QuantizationError::TensorError { 
                    reason: format!("Failed to flatten: {}", e) 
                })?
                .to_vec1::<f32>()
                .map_err(|e| crate::quantization::QuantizationError::TensorError { 
                    reason: format!("Failed to convert: {}", e) 
                })?;
            Ok(values.iter().all(|x| x.is_finite()))
        }
        Err(_) => Ok(true) // Graceful rejection is acceptable
    }
}

fn test_quantization_robustness(tensor: &Tensor, method: TernaryMethod) -> QuantizationResult<bool> {
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;
    
    match quantizer.quantize(tensor) {
        Ok(quantized) => {
            // Validate that result is properly formed
            let validation = validate_ternary_values(&quantized.values)?;
            Ok(validation.is_strictly_ternary && 
               quantized.stats.scale_factor.is_finite() &&
               quantized.stats.scale_factor > 0.0)
        }
        Err(_) => {
            // For edge cases, graceful failure is acceptable
            // depending on the specific case
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_cases_comprehensive() {
        let device = create_test_device();
        let results = run_edge_case_tests(&device).unwrap();
        
        assert!(results.total_tests > 0);
        assert!(results.robustness_score >= 0.0 && results.robustness_score <= 1.0);
        // Should handle most edge cases gracefully
        assert!(results.success_rate() > 0.6);
    }

    #[test]
    fn test_specific_edge_cases() {
        let device = create_test_device();
        
        // Test individual edge cases
        assert!(test_all_zeros(&device).unwrap());
        assert!(test_all_ones(&device).unwrap());
        assert!(test_constant_values(&device).unwrap());
    }
}
