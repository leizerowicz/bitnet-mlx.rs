//! Test data generation utilities
//!
//! This module provides utilities for generating test data with specific patterns
//! and characteristics for testing BitNet quantization operations.

use candle_core::{Device, Tensor, DType, Shape};
use crate::quantization::QuantizationResult;
use std::collections::HashMap;

/// Test patterns for generating different types of test data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPattern {
    /// Normal distribution (mean=0, std=1)
    NormalDistribution,
    /// Uniform distribution in [-2, 2]
    UniformDistribution,
    /// Sparse weights (mostly zeros with few non-zero values)
    SparseWeights,
    /// Weights with extreme outliers
    OutlierHeavy,
    /// Very small values near zero
    SmallValues,
    /// Very large values
    LargeValues,
    /// All zeros
    AllZeros,
    /// All ones
    AllOnes,
    /// All negative ones
    AllNegativeOnes,
    /// Single non-zero value
    SingleNonZero,
    /// Alternating pattern (+1, -1, +1, -1, ...)
    Alternating,
    /// Random ternary values {-1, 0, +1}
    RandomTernary,
    /// Bimodal distribution
    BimodalDistribution,
    /// Exponential distribution
    ExponentialDistribution,
}

/// Generate a test tensor with the specified pattern and shape
pub fn generate_test_tensor(pattern: TestPattern, shape: &[usize], device: &Device) -> QuantizationResult<Tensor> {
    let total_elements: usize = shape.iter().product();

    let data = match pattern {
        TestPattern::NormalDistribution => generate_normal_distribution(total_elements),
        TestPattern::UniformDistribution => generate_uniform_distribution(total_elements),
        TestPattern::SparseWeights => generate_sparse_weights(total_elements),
        TestPattern::OutlierHeavy => generate_outlier_heavy(total_elements),
        TestPattern::SmallValues => generate_small_values(total_elements),
        TestPattern::LargeValues => generate_large_values(total_elements),
        TestPattern::AllZeros => vec![0.0f32; total_elements],
        TestPattern::AllOnes => vec![1.0f32; total_elements],
        TestPattern::AllNegativeOnes => vec![-1.0f32; total_elements],
        TestPattern::SingleNonZero => generate_single_nonzero(total_elements),
        TestPattern::Alternating => generate_alternating(total_elements),
        TestPattern::RandomTernary => generate_random_ternary(total_elements),
        TestPattern::BimodalDistribution => generate_bimodal_distribution(total_elements),
        TestPattern::ExponentialDistribution => generate_exponential_distribution(total_elements),
    };

    let tensor_shape = Shape::from_dims(shape);
    let tensor = Tensor::from_vec(data, tensor_shape, device)
        .map_err(|e| crate::quantization::QuantizationError::TensorError {
            reason: format!("Failed to create test tensor: {}", e)
        })?;

    Ok(tensor)
}

/// Create a test device (CPU for deterministic testing)
pub fn create_test_device() -> Device {
    Device::Cpu
}

/// Generate test tensor sets for comprehensive testing
pub fn generate_test_tensor_set(device: &Device) -> QuantizationResult<TestTensorSet> {
    let shapes = vec![
        vec![4, 4],           // Small 2D
        vec![8, 8],           // Medium 2D
        vec![2, 3, 4],        // 3D
        vec![1, 256],         // Wide 2D
        vec![256, 1],         // Tall 2D
        vec![32, 32, 4],      // Large 3D
    ];

    let mut tensor_sets = HashMap::new();

    for pattern in [
        TestPattern::NormalDistribution,
        TestPattern::UniformDistribution,
        TestPattern::SparseWeights,
        TestPattern::OutlierHeavy,
        TestPattern::SmallValues,
        TestPattern::LargeValues,
    ] {
        let mut pattern_tensors = Vec::new();
        for shape in &shapes {
            let tensor = generate_test_tensor(pattern, shape, device)?;
            pattern_tensors.push(tensor);
        }
        tensor_sets.insert(pattern, pattern_tensors);
    }

    Ok(TestTensorSet { tensor_sets })
}

/// Collection of test tensors organized by pattern
pub struct TestTensorSet {
    pub tensor_sets: HashMap<TestPattern, Vec<Tensor>>,
}

impl TestTensorSet {
    /// Get tensors for a specific pattern
    pub fn get_pattern_tensors(&self, pattern: TestPattern) -> Option<&Vec<Tensor>> {
        self.tensor_sets.get(&pattern)
    }

    /// Get all patterns
    pub fn get_patterns(&self) -> Vec<TestPattern> {
        self.tensor_sets.keys().cloned().collect()
    }

    /// Get total number of test tensors
    pub fn total_tensors(&self) -> usize {
        self.tensor_sets.values().map(|v| v.len()).sum()
    }
}

// Private helper functions for generating specific patterns

fn generate_normal_distribution(n: usize) -> Vec<f32> {
    // Approximate normal distribution using Box-Muller transform
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        // Simple deterministic "random" for reproducible tests
        let u1 = ((i * 73 + 17) % 1000) as f32 / 1000.0 + 0.001;
        let u2 = ((i * 137 + 23) % 1000) as f32 / 1000.0 + 0.001;

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        result.push(z);
    }

    result
}

fn generate_uniform_distribution(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let normalized = (i * 47 + 13) % 1000;
        (normalized as f32 / 1000.0 - 0.5) * 4.0 // Range [-2, 2]
    }).collect()
}

fn generate_sparse_weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        if i % 7 == 0 {
            1.0
        } else if i % 11 == 0 {
            -1.0
        } else if i % 13 == 0 {
            0.5
        } else {
            0.0
        }
    }).collect()
}

fn generate_outlier_heavy(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        match i {
            0 => 100.0,
            1 => -100.0,
            2 => 50.0,
            3 => -50.0,
            _ => (i as f32 - 4.0) * 0.1, // Small background values
        }
    }).collect()
}

fn generate_small_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        ((i * 31 + 7) % 1000) as f32 / 1000.0 * 1e-3 - 5e-4
    }).collect()
}

fn generate_large_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        ((i * 67 + 19) % 1000) as f32 / 1000.0 * 2000.0 - 1000.0
    }).collect()
}

fn generate_single_nonzero(n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n];
    if n > 0 {
        result[n / 2] = 1.0; // Put non-zero value in the middle
    }
    result
}

fn generate_alternating(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        if i % 2 == 0 { 1.0 } else { -1.0 }
    }).collect()
}

fn generate_random_ternary(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        match (i * 37 + 29) % 3 {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => unreachable!(),
        }
    }).collect()
}

fn generate_bimodal_distribution(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let value = ((i * 83 + 41) % 1000) as f32 / 1000.0;
        if value < 0.5 {
            -2.0 + value * 2.0  // Mode around -1
        } else {
            value * 2.0         // Mode around +1
        }
    }).collect()
}

fn generate_exponential_distribution(n: usize) -> Vec<f32> {
    (0..n).map(|i| {
        let u = ((i * 53 + 61) % 1000) as f32 / 1000.0 + 0.001;
        let lambda = 1.0;
        -(-u.ln()) / lambda
    }).collect()
}

/// Statistical properties of generated data
#[derive(Debug)]
pub struct DataStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32, // Percentage of zeros
    pub skewness: f32,
    pub kurtosis: f32,
}

impl DataStatistics {
    /// Compute statistics for a tensor
    pub fn from_tensor(tensor: &Tensor) -> QuantizationResult<Self> {
        let values = tensor.flatten_all()
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to flatten tensor: {}", e)
            })?
            .to_vec1::<f32>()
            .map_err(|e| crate::quantization::QuantizationError::TensorError {
                reason: format!("Failed to convert to vec: {}", e)
            })?;

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;

        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let zero_count = values.iter().filter(|&&x| x.abs() < 1e-8).count();
        let sparsity = zero_count as f32 / n;

        // Compute skewness and kurtosis
        let skewness = if std_dev > 1e-8 {
            values.iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>() / n
        } else {
            0.0
        };

        let kurtosis = if std_dev > 1e-8 {
            values.iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f32>() / n - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Ok(DataStatistics {
            mean,
            std_dev,
            min,
            max,
            sparsity,
            skewness,
            kurtosis,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_normal_distribution() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &[100], &device).unwrap();

        let stats = DataStatistics::from_tensor(&tensor).unwrap();
        // Normal distribution should have mean close to 0 and reasonable std dev
        assert!(stats.mean.abs() < 0.5);
        assert!(stats.std_dev > 0.5 && stats.std_dev < 2.0);
    }

    #[test]
    fn test_generate_sparse_weights() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::SparseWeights, &[100], &device).unwrap();

        let stats = DataStatistics::from_tensor(&tensor).unwrap();
        // Sparse weights should have high sparsity
        assert!(stats.sparsity > 0.6);
    }

    #[test]
    fn test_generate_all_zeros() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::AllZeros, &[50], &device).unwrap();

        let stats = DataStatistics::from_tensor(&tensor).unwrap();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.sparsity, 1.0);
    }

    #[test]
    fn test_tensor_set_generation() {
        let device = create_test_device();
        let tensor_set = generate_test_tensor_set(&device).unwrap();

        assert!(tensor_set.total_tensors() > 0);
        assert!(tensor_set.get_patterns().len() >= 6);

        let normal_tensors = tensor_set.get_pattern_tensors(TestPattern::NormalDistribution);
        assert!(normal_tensors.is_some());
        assert!(normal_tensors.unwrap().len() >= 6); // Different shapes
    }

    #[test]
    fn test_outlier_heavy_pattern() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::OutlierHeavy, &[100], &device).unwrap();

        let stats = DataStatistics::from_tensor(&tensor).unwrap();
        // Should have large max/min values (outliers)
        assert!(stats.max > 50.0 || stats.min < -50.0);
    }

    #[test]
    fn test_ternary_pattern() {
        let device = create_test_device();
        let tensor = generate_test_tensor(TestPattern::RandomTernary, &[100], &device).unwrap();

        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // All values should be in {-1, 0, 1}
        for &val in &values {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
    }
}
