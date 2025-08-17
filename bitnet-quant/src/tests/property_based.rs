//! Property-based testing for quantization operations
//!
//! This module implements property-based testing to verify invariants
//! and statistical properties of quantization operations.

use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, validate_ternary_values,
    validate_shape_preservation, validate_value_bounds, 
    create_test_device
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Results of property-based testing
#[derive(Debug, Clone, Default)]
pub struct PropertyTestResults {
    pub total_property_tests: usize,
    pub passed_property_tests: usize,
    pub invariant_violations: Vec<String>,
    pub property_results: HashMap<String, PropertyResult>,
    pub statistical_validity: f64,
}

#[derive(Debug, Clone)]
pub struct PropertyResult {
    pub property_name: String,
    pub test_count: usize,
    pub success_count: usize,
    pub success_rate: f64,
    pub violations: Vec<String>,
}

impl PropertyTestResults {
    pub fn success_rate(&self) -> f64 {
        if self.total_property_tests == 0 { 1.0 }
        else { self.passed_property_tests as f64 / self.total_property_tests as f64 }
    }
}

/// Run property-based tests
pub fn run_property_tests(device: &Device, iterations: usize) -> QuantizationResult<PropertyTestResults> {
    let mut results = PropertyTestResults::default();
    
    let properties = vec![
        ("Shape Preservation", test_shape_preservation_property),
        ("Value Bounds", test_value_bounds_property),
        ("Deterministic Quantization", test_deterministic_property),
        ("Scale Factor Positivity", test_scale_factor_positivity_property),
        ("Quantized Values Ternary", test_ternary_values_property),
        ("Dequantization Consistency", test_dequantization_consistency_property),
        ("Scale Invariance", test_scale_invariance_property),
        ("Translation Invariance", test_translation_invariance_property),
    ];

    for (property_name, test_fn) in properties {
        let mut property_result = PropertyResult {
            property_name: property_name.to_string(),
            test_count: 0,
            success_count: 0,
            success_rate: 0.0,
            violations: Vec::new(),
        };

        // Run property test multiple times with different inputs
        for i in 0..iterations {
            property_result.test_count += 1;
            results.total_property_tests += 1;

            match test_fn(device, i) {
                Ok(passed) => {
                    if passed {
                        property_result.success_count += 1;
                        results.passed_property_tests += 1;
                    } else {
                        let violation = format!("Property '{}' violated at iteration {}", property_name, i);
                        property_result.violations.push(violation.clone());
                        results.invariant_violations.push(violation);
                    }
                }
                Err(e) => {
                    let violation = format!("Property '{}' error at iteration {}: {}", property_name, i, e);
                    property_result.violations.push(violation.clone());
                    results.invariant_violations.push(violation);
                }
            }
        }

        property_result.success_rate = property_result.success_count as f64 / property_result.test_count as f64;
        results.property_results.insert(property_name.to_string(), property_result);
    }

    results.statistical_validity = calculate_statistical_validity(&results);
    
    Ok(results)
}

// Property test functions

fn test_shape_preservation_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let patterns = [TestPattern::NormalDistribution, TestPattern::UniformDistribution, TestPattern::SparseWeights];
    let pattern = patterns[seed % patterns.len()];
    
    let shapes = [vec![16], vec![4, 4], vec![8, 2], vec![2, 2, 4]];
    let shape = &shapes[seed % shapes.len()];
    
    let tensor = generate_test_tensor(pattern, shape, device)?;
    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;
    let quantized = quantizer.quantize(&tensor)?;
    
    let validation = validate_shape_preservation(&tensor, &quantized.values);
    Ok(validation.passed)
}

fn test_value_bounds_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let patterns = [TestPattern::NormalDistribution, TestPattern::OutlierHeavy, TestPattern::LargeValues];
    let pattern = patterns[seed % patterns.len()];
    
    let tensor = generate_test_tensor(pattern, &[64], device)?;
    let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
    let quantized = quantizer.quantize(&tensor)?;
    
    let validation = validate_value_bounds(&quantized.values, -1.0, 1.0);
    Ok(validation.passed)
}

fn test_deterministic_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let pattern = if seed % 2 == 0 { TestPattern::NormalDistribution } else { TestPattern::UniformDistribution };
    let tensor = generate_test_tensor(pattern, &[32], device)?;
    
    let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
    
    // Quantize the same tensor multiple times
    let result1 = quantizer.quantize(&tensor)?;
    let result2 = quantizer.quantize(&tensor)?;
    
    // Results should be identical
    let values1 = result1.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let values2 = result2.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    Ok(values1 == values2)
}

fn test_scale_factor_positivity_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let patterns = [TestPattern::NormalDistribution, TestPattern::SparseWeights, TestPattern::OutlierHeavy];
    let pattern = patterns[seed % patterns.len()];
    
    let tensor = generate_test_tensor(pattern, &[48], device)?;
    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.5 + (seed as f32 % 100) / 200.0))?;
    let result = quantizer.quantize(&tensor)?;
    
    Ok(result.stats.scale_factor > 0.0 && result.stats.scale_factor.is_finite())
}

fn test_ternary_values_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let methods = [TernaryMethod::MeanThreshold, TernaryMethod::MedianThreshold, 
                   TernaryMethod::AdaptiveThreshold, TernaryMethod::OptimalThreshold];
    let method = methods[seed % methods.len()];
    
    let tensor = generate_test_tensor(TestPattern::UniformDistribution, &[64], device)?;
    let quantizer = create_ternary_quantizer(method, Some(0.7))?;
    let result = quantizer.quantize(&tensor)?;
    
    let validation = validate_ternary_values(&result.values)?;
    Ok(validation.is_strictly_ternary)
}

fn test_dequantization_consistency_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let pattern = if seed % 3 == 0 { TestPattern::NormalDistribution } 
                 else if seed % 3 == 1 { TestPattern::SparseWeights } 
                 else { TestPattern::UniformDistribution };
    
    let tensor = generate_test_tensor(pattern, &[32], device)?;
    let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
    
    let quantized = quantizer.quantize(&tensor)?;
    let dequantized1 = quantizer.dequantize(&quantized)?;
    let dequantized2 = quantizer.dequantize(&quantized)?;
    
    // Multiple dequantizations should yield identical results
    let values1 = dequantized1.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let values2 = dequantized2.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let tolerance = 1e-6;
    Ok(values1.iter().zip(values2.iter()).all(|(&a, &b)| (a - b).abs() < tolerance))
}

fn test_scale_invariance_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::NormalDistribution, &[32], device)?;
    let scale_factor = 2.0 + (seed as f32 % 10) / 5.0; // Scale factors from 2.0 to 4.0
    
    let scaled_tensor = tensor.mul(&Tensor::new(scale_factor, device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create scale tensor: {}", e) 
        })?)?;
    
    let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
    
    let result1 = quantizer.quantize(&tensor)?;
    let result2 = quantizer.quantize(&scaled_tensor)?;
    
    // The quantized patterns should be the same (scale invariant)
    let values1 = result1.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let values2 = result2.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    // Check if the sign patterns are the same (allowing for some tolerance in threshold effects)
    let sign_agreement = values1.iter().zip(values2.iter())
        .filter(|(&a, &b)| a.signum() == b.signum())
        .count();
    
    let agreement_rate = sign_agreement as f64 / values1.len() as f64;
    Ok(agreement_rate > 0.8) // Allow some tolerance due to threshold effects
}

fn test_translation_invariance_property(device: &Device, seed: usize) -> QuantizationResult<bool> {
    let tensor = generate_test_tensor(TestPattern::NormalDistribution, &[32], device)?;
    let offset = (seed as f32 % 10) / 10.0; // Offsets from 0.0 to 0.9
    
    let offset_tensor = tensor.add(&Tensor::new(offset, device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to create offset tensor: {}", e) 
        })?)?;
    
    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;
    
    let result1 = quantizer.quantize(&tensor)?;
    let result2 = quantizer.quantize(&offset_tensor)?;
    
    // Translation should not dramatically change the quantization pattern
    // (some change is expected due to threshold effects, but should be limited)
    let values1 = result1.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let values2 = result2.values.flatten_all()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to flatten: {}", e) 
        })?
        .to_vec1::<f32>()
        .map_err(|e| crate::quantization::QuantizationError::TensorError { 
            reason: format!("Failed to convert: {}", e) 
        })?;
    
    let differences = values1.iter().zip(values2.iter())
        .filter(|(&a, &b)| a != b)
        .count();
    
    let change_rate = differences as f64 / values1.len() as f64;
    Ok(change_rate < 0.5) // Less than 50% of values should change
}

fn calculate_statistical_validity(results: &PropertyTestResults) -> f64 {
    if results.property_results.is_empty() {
        return 0.0;
    }

    let success_rates: Vec<f64> = results.property_results.values()
        .map(|r| r.success_rate)
        .collect();

    let mean_success_rate = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
    
    // Penalize high variance in success rates (inconsistency across properties)
    let variance = success_rates.iter()
        .map(|&rate| (rate - mean_success_rate).powi(2))
        .sum::<f64>() / success_rates.len() as f64;
    
    let consistency_penalty = 1.0 / (1.0 + variance);
    
    mean_success_rate * consistency_penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_based_testing() {
        let device = create_test_device();
        let results = run_property_tests(&device, 10).unwrap(); // Reduced iterations for testing
        
        assert!(results.total_property_tests > 0);
        assert!(!results.property_results.is_empty());
        assert!(results.statistical_validity >= 0.0 && results.statistical_validity <= 1.0);
    }

    #[test]
    fn test_individual_properties() {
        let device = create_test_device();
        
        // Test individual properties
        assert!(test_shape_preservation_property(&device, 0).unwrap());
        assert!(test_value_bounds_property(&device, 0).unwrap());
        assert!(test_deterministic_property(&device, 0).unwrap());
        assert!(test_ternary_values_property(&device, 0).unwrap());
    }

    #[test]
    fn test_property_consistency() {
        let device = create_test_device();
        
        // Test the same property multiple times to check consistency
        let mut results = Vec::new();
        for i in 0..5 {
            results.push(test_deterministic_property(&device, i).unwrap());
        }
        
        // All should pass (deterministic property should be consistent)
        assert!(results.iter().all(|&r| r));
    }
}
