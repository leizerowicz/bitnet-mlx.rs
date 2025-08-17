//! Performance regression testing for quantization operations
//!
//! This module implements performance regression testing to ensure that
//! quantization performance doesn't degrade over time and meets benchmarks.

use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{TestPattern, generate_test_tensor, create_test_device};
use candle_core::{Device, Tensor};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance benchmark results
#[derive(Debug, Clone, Default)]
pub struct PerformanceRegressionResults {
    pub benchmarks_run: usize,
    pub benchmarks_passed: usize,
    pub performance_regressions: usize,
    pub benchmark_results: HashMap<String, BenchmarkResult>,
    pub overall_performance_score: f64,
    pub critical_regressions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub iterations: usize,
    pub measurements: Vec<Duration>,
    pub statistics: PerformanceStatistics,
    pub baseline_comparison: BaselineComparison,
    pub performance_category: PerformanceCategory,
}

#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub mean_duration: Duration,
    pub median_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub std_deviation: Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct BaselineComparison {
    pub baseline_mean: Duration,
    pub current_mean: Duration,
    pub performance_ratio: f64, // current/baseline (>1.0 means slower)
    pub is_regression: bool,
    pub regression_severity: RegressionSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceCategory {
    Fast,      // Top 25%
    Normal,    // Middle 50%
    Slow,      // Bottom 25%
    Critical,  // Unacceptably slow
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegressionSeverity {
    None,      // No regression or improvement
    Minor,     // 5-15% slower
    Moderate,  // 15-30% slower  
    Major,     // 30-50% slower
    Critical,  // >50% slower
}

impl PerformanceRegressionResults {
    pub fn success_rate(&self) -> f64 {
        if self.benchmarks_run == 0 { 1.0 }
        else { self.benchmarks_passed as f64 / self.benchmarks_run as f64 }
    }

    pub fn has_critical_regressions(&self) -> bool {
        !self.critical_regressions.is_empty() ||
        self.benchmark_results.values()
            .any(|r| r.baseline_comparison.regression_severity == RegressionSeverity::Critical)
    }
}

/// Run performance regression tests
pub fn run_performance_regression_tests(device: &Device) -> QuantizationResult<PerformanceRegressionResults> {
    let mut results = PerformanceRegressionResults::default();

    // Define performance benchmarks
    let benchmarks = vec![
        ("Small Tensor Quantization", benchmark_small_tensor_quantization),
        ("Large Tensor Quantization", benchmark_large_tensor_quantization),
        ("Batch Quantization", benchmark_batch_quantization),
        ("Method Comparison", benchmark_method_comparison),
        ("Memory Efficiency", benchmark_memory_efficiency),
        ("Concurrent Operations", benchmark_concurrent_operations),
        ("Edge Case Performance", benchmark_edge_case_performance),
    ];

    for (benchmark_name, benchmark_fn) in benchmarks {
        results.benchmarks_run += 1;
        
        match benchmark_fn(device, 50) { // 50 iterations per benchmark
            Ok(benchmark_result) => {
                let passes_performance_threshold = 
                    benchmark_result.baseline_comparison.regression_severity != RegressionSeverity::Critical &&
                    benchmark_result.performance_category != PerformanceCategory::Critical;
                
                if passes_performance_threshold {
                    results.benchmarks_passed += 1;
                } else {
                    results.critical_regressions.push(
                        format!("Critical performance issue in {}", benchmark_name)
                    );
                }

                if benchmark_result.baseline_comparison.is_regression {
                    results.performance_regressions += 1;
                }

                results.benchmark_results.insert(benchmark_name.to_string(), benchmark_result);
            }
            Err(e) => {
                results.critical_regressions.push(
                    format!("Failed to run benchmark {}: {}", benchmark_name, e)
                );
            }
        }
    }

    results.overall_performance_score = calculate_performance_score(&results);

    Ok(results)
}

// Individual benchmark functions

fn benchmark_small_tensor_quantization(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let tensor_size = vec![64]; // Small tensor
    
    // Warm up
    for _ in 0..5 {
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;
        let _ = quantizer.quantize(&tensor)?;
    }

    // Actual measurements
    for _ in 0..iterations {
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;
        
        let start = Instant::now();
        let _result = quantizer.quantize(&tensor)?;
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let statistics = calculate_performance_statistics(&measurements, tensor_size.iter().product());
    let baseline_comparison = compare_with_baseline("small_tensor", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Small Tensor Quantization".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_large_tensor_quantization(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let tensor_size = vec![1024, 1024]; // Large tensor
    
    // Warm up
    for _ in 0..3 {
        let tensor = generate_test_tensor(TestPattern::UniformDistribution, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        let _ = quantizer.quantize(&tensor)?;
    }

    // Actual measurements  
    for _ in 0..iterations {
        let tensor = generate_test_tensor(TestPattern::UniformDistribution, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        
        let start = Instant::now();
        let _result = quantizer.quantize(&tensor)?;
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let statistics = calculate_performance_statistics(&measurements, tensor_size.iter().product());
    let baseline_comparison = compare_with_baseline("large_tensor", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Large Tensor Quantization".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_batch_quantization(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let batch_size = 10;
    let tensor_size = vec![128, 128];
    
    // Warm up
    for _ in 0..3 {
        let tensors: Result<Vec<_>, _> = (0..batch_size)
            .map(|_| generate_test_tensor(TestPattern::SparseWeights, &tensor_size, device))
            .collect();
        let tensors = tensors?;
        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
        
        for tensor in &tensors {
            let _ = quantizer.quantize(tensor)?;
        }
    }

    // Actual measurements
    for _ in 0..iterations {
        let tensors: Result<Vec<_>, _> = (0..batch_size)
            .map(|_| generate_test_tensor(TestPattern::SparseWeights, &tensor_size, device))
            .collect();
        let tensors = tensors?;
        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
        
        let start = Instant::now();
        for tensor in &tensors {
            let _result = quantizer.quantize(tensor)?;
        }
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let total_elements = batch_size * tensor_size.iter().product::<usize>();
    let statistics = calculate_performance_statistics(&measurements, total_elements);
    let baseline_comparison = compare_with_baseline("batch_quantization", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Batch Quantization".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_method_comparison(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let tensor_size = vec![256, 256];
    let methods = [
        TernaryMethod::MeanThreshold,
        TernaryMethod::MedianThreshold, 
        TernaryMethod::AdaptiveThreshold,
        TernaryMethod::OptimalThreshold,
    ];
    
    // Warm up
    for method in &methods {
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(*method, Some(0.7))?;
        let _ = quantizer.quantize(&tensor)?;
    }

    // Actual measurements - test all methods per iteration
    for _ in 0..iterations {
        let tensor = generate_test_tensor(TestPattern::NormalDistribution, &tensor_size, device)?;
        
        let start = Instant::now();
        for method in &methods {
            let quantizer = create_ternary_quantizer(*method, Some(0.7))?;
            let _result = quantizer.quantize(&tensor)?;
        }
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let total_elements = methods.len() * tensor_size.iter().product::<usize>();
    let statistics = calculate_performance_statistics(&measurements, total_elements);
    let baseline_comparison = compare_with_baseline("method_comparison", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Method Comparison".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_memory_efficiency(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let tensor_size = vec![512, 512];
    
    // Warm up
    for _ in 0..3 {
        let tensor = generate_test_tensor(TestPattern::LargeValues, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        let quantized = quantizer.quantize(&tensor)?;
        let _ = quantizer.dequantize(&quantized)?;
    }

    // Actual measurements - including dequantization for full cycle
    for _ in 0..iterations {
        let tensor = generate_test_tensor(TestPattern::LargeValues, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        
        let start = Instant::now();
        let quantized = quantizer.quantize(&tensor)?;
        let _dequantized = quantizer.dequantize(&quantized)?;
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let statistics = calculate_performance_statistics(&measurements, tensor_size.iter().product());
    let baseline_comparison = compare_with_baseline("memory_efficiency", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Memory Efficiency".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_concurrent_operations(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    let tensor_size = vec![256, 256];
    
    // For this benchmark, we simulate concurrent-like operations by alternating methods
    let methods = [TernaryMethod::MeanThreshold, TernaryMethod::OptimalThreshold];
    
    // Warm up
    for _ in 0..3 {
        for method in &methods {
            let tensor = generate_test_tensor(TestPattern::OutlierHeavy, &tensor_size, device)?;
            let quantizer = create_ternary_quantizer(*method, Some(0.7))?;
            let _ = quantizer.quantize(&tensor)?;
        }
    }

    // Actual measurements
    for i in 0..iterations {
        let method = methods[i % methods.len()];
        let tensor = generate_test_tensor(TestPattern::OutlierHeavy, &tensor_size, device)?;
        let quantizer = create_ternary_quantizer(method, Some(0.7))?;
        
        let start = Instant::now();
        let _result = quantizer.quantize(&tensor)?;
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let statistics = calculate_performance_statistics(&measurements, tensor_size.iter().product());
    let baseline_comparison = compare_with_baseline("concurrent_operations", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Concurrent Operations".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

fn benchmark_edge_case_performance(device: &Device, iterations: usize) -> QuantizationResult<BenchmarkResult> {
    let mut measurements = Vec::new();
    
    // Test various edge case scenarios
    let edge_cases = [
        (TestPattern::AllZeros, vec![100]),
        (TestPattern::LargeValues, vec![100]),
        (TestPattern::OutlierHeavy, vec![100]),
    ];
    
    // Warm up
    for (pattern, shape) in &edge_cases {
        let tensor = generate_test_tensor(*pattern, shape, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
        let _ = quantizer.quantize(&tensor)?;
    }

    // Actual measurements
    for i in 0..iterations {
        let (pattern, shape) = &edge_cases[i % edge_cases.len()];
        let tensor = generate_test_tensor(*pattern, shape, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;
        
        let start = Instant::now();
        let _result = quantizer.quantize(&tensor)?;
        let duration = start.elapsed();
        
        measurements.push(duration);
    }

    let avg_elements = edge_cases.iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum::<usize>() / edge_cases.len();
        
    let statistics = calculate_performance_statistics(&measurements, avg_elements);
    let baseline_comparison = compare_with_baseline("edge_case_performance", &statistics);
    let performance_category = categorize_performance(&statistics);

    Ok(BenchmarkResult {
        benchmark_name: "Edge Case Performance".to_string(),
        iterations,
        measurements,
        statistics,
        baseline_comparison,
        performance_category,
    })
}

// Helper functions

fn calculate_performance_statistics(measurements: &[Duration], num_elements: usize) -> PerformanceStatistics {
    if measurements.is_empty() {
        return PerformanceStatistics {
            mean_duration: Duration::ZERO,
            median_duration: Duration::ZERO,
            min_duration: Duration::ZERO,
            max_duration: Duration::ZERO,
            std_deviation: Duration::ZERO,
            throughput_ops_per_sec: 0.0,
            memory_usage_bytes: 0,
        };
    }

    let mut sorted_measurements = measurements.to_vec();
    sorted_measurements.sort();

    let mean_nanos = measurements.iter().map(|d| d.as_nanos()).sum::<u128>() / measurements.len() as u128;
    let mean_duration = Duration::from_nanos(mean_nanos as u64);

    let median_duration = sorted_measurements[sorted_measurements.len() / 2];
    let min_duration = *sorted_measurements.first().unwrap();
    let max_duration = *sorted_measurements.last().unwrap();

    // Calculate standard deviation
    let variance = measurements.iter()
        .map(|d| {
            let diff = d.as_nanos() as i128 - mean_nanos as i128;
            (diff * diff) as u128
        })
        .sum::<u128>() / measurements.len() as u128;
    let std_deviation = Duration::from_nanos((variance as f64).sqrt() as u64);

    // Calculate throughput
    let throughput_ops_per_sec = if mean_duration.as_secs_f64() > 0.0 {
        num_elements as f64 / mean_duration.as_secs_f64()
    } else {
        0.0
    };

    // Estimate memory usage (rough approximation)
    let memory_usage_bytes = num_elements * std::mem::size_of::<f32>();

    PerformanceStatistics {
        mean_duration,
        median_duration,
        min_duration,
        max_duration,
        std_deviation,
        throughput_ops_per_sec,
        memory_usage_bytes,
    }
}

fn compare_with_baseline(benchmark_name: &str, statistics: &PerformanceStatistics) -> BaselineComparison {
    // These are approximate baseline values - in a real implementation, 
    // these would be loaded from historical performance data
    let baseline_mean = match benchmark_name {
        "small_tensor" => Duration::from_micros(10),      // 10μs for small tensors
        "large_tensor" => Duration::from_millis(1),       // 1ms for large tensors  
        "batch_quantization" => Duration::from_millis(5), // 5ms for batch operations
        "method_comparison" => Duration::from_millis(2),  // 2ms for method comparison
        "memory_efficiency" => Duration::from_millis(2),  // 2ms for full cycle
        "concurrent_operations" => Duration::from_micros(15), // 15μs per operation
        "edge_case_performance" => Duration::from_micros(20), // 20μs for edge cases
        _ => Duration::from_millis(1), // Default 1ms baseline
    };

    let current_mean = statistics.mean_duration;
    let performance_ratio = current_mean.as_secs_f64() / baseline_mean.as_secs_f64();
    
    let (is_regression, severity) = if performance_ratio <= 1.05 {
        (false, RegressionSeverity::None)
    } else if performance_ratio <= 1.15 {
        (true, RegressionSeverity::Minor)
    } else if performance_ratio <= 1.30 {
        (true, RegressionSeverity::Moderate)  
    } else if performance_ratio <= 1.50 {
        (true, RegressionSeverity::Major)
    } else {
        (true, RegressionSeverity::Critical)
    };

    BaselineComparison {
        baseline_mean,
        current_mean,
        performance_ratio,
        is_regression,
        regression_severity: severity,
    }
}

fn categorize_performance(statistics: &PerformanceStatistics) -> PerformanceCategory {
    // Categorize based on throughput (operations per second)
    let throughput = statistics.throughput_ops_per_sec;
    
    if throughput > 1_000_000.0 {
        PerformanceCategory::Fast
    } else if throughput > 100_000.0 {
        PerformanceCategory::Normal  
    } else if throughput > 10_000.0 {
        PerformanceCategory::Slow
    } else {
        PerformanceCategory::Critical
    }
}

fn calculate_performance_score(results: &PerformanceRegressionResults) -> f64 {
    if results.benchmark_results.is_empty() {
        return 0.0;
    }

    let total_score: f64 = results.benchmark_results.values()
        .map(|benchmark| {
            // Base score from performance category
            let category_score = match benchmark.performance_category {
                PerformanceCategory::Fast => 1.0,
                PerformanceCategory::Normal => 0.8,
                PerformanceCategory::Slow => 0.6,
                PerformanceCategory::Critical => 0.2,
            };
            
            // Penalty for regressions
            let regression_penalty = match benchmark.baseline_comparison.regression_severity {
                RegressionSeverity::None => 1.0,
                RegressionSeverity::Minor => 0.9,
                RegressionSeverity::Moderate => 0.7,
                RegressionSeverity::Major => 0.5,
                RegressionSeverity::Critical => 0.2,
            };
            
            category_score * regression_penalty
        })
        .sum();

    total_score / results.benchmark_results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_regression_testing() {
        let device = create_test_device();
        let results = run_performance_regression_tests(&device).unwrap();
        
        assert!(results.benchmarks_run > 0);
        assert!(results.overall_performance_score >= 0.0 && results.overall_performance_score <= 1.0);
    }

    #[test]
    fn test_individual_benchmarks() {
        let device = create_test_device();
        
        let result = benchmark_small_tensor_quantization(&device, 5).unwrap();
        assert_eq!(result.iterations, 5);
        assert_eq!(result.measurements.len(), 5);
        assert!(result.statistics.mean_duration > Duration::ZERO);
    }

    #[test] 
    fn test_performance_categorization() {
        let fast_stats = PerformanceStatistics {
            throughput_ops_per_sec: 2_000_000.0,
            ..PerformanceStatistics {
                mean_duration: Duration::from_micros(1),
                median_duration: Duration::from_micros(1),
                min_duration: Duration::from_micros(1),
                max_duration: Duration::from_micros(1),
                std_deviation: Duration::ZERO,
                memory_usage_bytes: 1000,
            }
        };
        assert_eq!(categorize_performance(&fast_stats), PerformanceCategory::Fast);

        let slow_stats = PerformanceStatistics {
            throughput_ops_per_sec: 5_000.0,
            ..PerformanceStatistics {
                mean_duration: Duration::from_millis(1),
                median_duration: Duration::from_millis(1),
                min_duration: Duration::from_millis(1),
                max_duration: Duration::from_millis(1),
                std_deviation: Duration::ZERO,
                memory_usage_bytes: 1000,
            }
        };
        assert_eq!(categorize_performance(&slow_stats), PerformanceCategory::Critical);
    }
}
