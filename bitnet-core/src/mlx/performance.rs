//! MLX Performance Comparison Tools
//!
//! This module provides comprehensive performance comparison utilities for MLX operations,
//! including benchmarking, profiling, device comparisons, and regression testing.

#[cfg(feature = "mlx")]
use mlx_rs::{ops, Array};

use crate::mlx::optimization::{MemoryStats, MlxProfiler};
use crate::mlx::{BitNetMlxDevice, MlxTensor};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics for MLX operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PerformanceMetrics {
    pub operation_name: String,
    pub device_type: String,
    pub execution_time: Duration,
    pub memory_usage: MemoryUsage,
    pub throughput: f64, // operations per second
    pub tensor_shapes: Vec<Vec<usize>>,
    pub data_type: String,
    pub timestamp: std::time::SystemTime,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryUsage {
    pub peak_memory_mb: f64,
    pub allocated_memory_mb: f64,
    pub freed_memory_mb: f64,
    pub memory_efficiency: f64, // ratio of useful memory to total allocated
}

/// Benchmark configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub tensor_sizes: Vec<Vec<usize>>,
    pub data_types: Vec<String>,
    pub devices: Vec<String>,
    pub timeout: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            tensor_sizes: vec![
                vec![128, 128],
                vec![512, 512],
                vec![1024, 1024],
                vec![2048, 2048],
            ],
            data_types: vec!["f32".to_string(), "f16".to_string()],
            devices: vec!["cpu".to_string(), "gpu".to_string()],
            timeout: Duration::from_secs(30),
        }
    }
}

/// Performance comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ComparisonResult {
    pub baseline_metrics: PerformanceMetrics,
    pub comparison_metrics: PerformanceMetrics,
    pub speedup: f64,
    pub memory_improvement: f64,
    pub throughput_improvement: f64,
    pub recommendation: String,
}

/// MLX Performance Benchmarker
#[cfg(feature = "mlx")]
#[allow(dead_code)]
pub struct MlxPerformanceBenchmarker {
    config: BenchmarkConfig,
    profiler: MlxProfiler,
    results: Vec<PerformanceMetrics>,
    baseline_results: HashMap<String, PerformanceMetrics>,
}

#[cfg(feature = "mlx")]
impl MlxPerformanceBenchmarker {
    /// Create a new performance benchmarker
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            profiler: MlxProfiler::new(),
            results: Vec::new(),
            baseline_results: HashMap::new(),
        }
    }

    /// Benchmark matrix multiplication operation
    pub fn benchmark_matmul(&mut self, device: &BitNetMlxDevice) -> Result<PerformanceMetrics> {
        let operation_name = "matmul";
        let mut total_time = Duration::ZERO;
        let mut memory_stats = MemoryUsage {
            peak_memory_mb: 0.0,
            allocated_memory_mb: 0.0,
            freed_memory_mb: 0.0,
            memory_efficiency: 0.0,
        };

        // Use the largest tensor size for this benchmark
        let default_shape = vec![1024, 1024];
        let shape = self.config.tensor_sizes.last().unwrap_or(&default_shape);

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            let a = self.create_test_tensor(shape, device)?;
            let b = self.create_test_tensor(shape, device)?;
            let _ = self.perform_matmul(&a, &b)?;
        }

        // Measurement iterations
        for _ in 0..self.config.measurement_iterations {
            let a = self.create_test_tensor(shape, device)?;
            let b = self.create_test_tensor(shape, device)?;

            let start = Instant::now();
            let _result = self.perform_matmul(&a, &b)?;
            let duration = start.elapsed();

            total_time += duration;
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        let metrics = PerformanceMetrics {
            operation_name: operation_name.to_string(),
            device_type: device.device_type().to_string(),
            execution_time: avg_time,
            memory_usage: memory_stats,
            throughput,
            tensor_shapes: vec![shape.clone()],
            data_type: "f32".to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        self.results.push(metrics.clone());
        Ok(metrics)
    }

    /// Benchmark quantization operation
    pub fn benchmark_quantization(
        &mut self,
        device: &BitNetMlxDevice,
    ) -> Result<PerformanceMetrics> {
        let operation_name = "quantization";
        let mut total_time = Duration::ZERO;

        let default_shape = vec![1024, 1024];
        let shape = self.config.tensor_sizes.last().unwrap_or(&default_shape);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let tensor = self.create_test_tensor(shape, device)?;
            let _ = self.perform_quantization(&tensor)?;
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let tensor = self.create_test_tensor(shape, device)?;

            let start = Instant::now();
            let _result = self.perform_quantization(&tensor)?;
            let duration = start.elapsed();

            total_time += duration;
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        let metrics = PerformanceMetrics {
            operation_name: operation_name.to_string(),
            device_type: device.device_type().to_string(),
            execution_time: avg_time,
            memory_usage: MemoryUsage {
                peak_memory_mb: 0.0,
                allocated_memory_mb: 0.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.0,
            },
            throughput,
            tensor_shapes: vec![shape.clone()],
            data_type: "f32".to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        self.results.push(metrics.clone());
        Ok(metrics)
    }

    /// Benchmark element-wise operations
    pub fn benchmark_elementwise(
        &mut self,
        device: &BitNetMlxDevice,
        operation: &str,
    ) -> Result<PerformanceMetrics> {
        let mut total_time = Duration::ZERO;

        let default_shape = vec![1024, 1024];
        let shape = self.config.tensor_sizes.last().unwrap_or(&default_shape);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let a = self.create_test_tensor(shape, device)?;
            let b = self.create_test_tensor(shape, device)?;
            let _ = self.perform_elementwise(&a, &b, operation)?;
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let a = self.create_test_tensor(shape, device)?;
            let b = self.create_test_tensor(shape, device)?;

            let start = Instant::now();
            let _result = self.perform_elementwise(&a, &b, operation)?;
            let duration = start.elapsed();

            total_time += duration;
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        let metrics = PerformanceMetrics {
            operation_name: operation.to_string(),
            device_type: device.device_type().to_string(),
            execution_time: avg_time,
            memory_usage: MemoryUsage {
                peak_memory_mb: 0.0,
                allocated_memory_mb: 0.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.0,
            },
            throughput,
            tensor_shapes: vec![shape.clone()],
            data_type: "f32".to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        self.results.push(metrics.clone());
        Ok(metrics)
    }

    /// Compare performance between devices
    pub fn compare_devices(&mut self, operation: &str) -> Result<Vec<ComparisonResult>> {
        let mut comparisons = Vec::new();

        let cpu_device = BitNetMlxDevice::cpu()?;
        let gpu_device = BitNetMlxDevice::gpu()?;

        // Benchmark on CPU
        let cpu_metrics = match operation {
            "matmul" => self.benchmark_matmul(&cpu_device)?,
            "quantization" => self.benchmark_quantization(&cpu_device)?,
            op => self.benchmark_elementwise(&cpu_device, op)?,
        };

        // Benchmark on GPU
        let gpu_metrics = match operation {
            "matmul" => self.benchmark_matmul(&gpu_device)?,
            "quantization" => self.benchmark_quantization(&gpu_device)?,
            op => self.benchmark_elementwise(&gpu_device, op)?,
        };

        // Calculate comparison
        let speedup =
            cpu_metrics.execution_time.as_secs_f64() / gpu_metrics.execution_time.as_secs_f64();
        let throughput_improvement = gpu_metrics.throughput / cpu_metrics.throughput;

        let recommendation = if speedup > 1.5 {
            "Use GPU for better performance".to_string()
        } else if speedup < 0.8 {
            "Use CPU for better performance".to_string()
        } else {
            "Performance is similar on both devices".to_string()
        };

        let comparison = ComparisonResult {
            baseline_metrics: cpu_metrics,
            comparison_metrics: gpu_metrics,
            speedup,
            memory_improvement: 0.0, // TODO: Calculate actual memory improvement
            throughput_improvement,
            recommendation,
        };

        comparisons.push(comparison);
        Ok(comparisons)
    }

    /// Set baseline performance metrics
    pub fn set_baseline(&mut self, operation: &str, metrics: PerformanceMetrics) {
        self.baseline_results.insert(operation.to_string(), metrics);
    }

    /// Compare current performance against baseline
    pub fn compare_against_baseline(
        &self,
        operation: &str,
        current_metrics: &PerformanceMetrics,
    ) -> Option<ComparisonResult> {
        if let Some(baseline) = self.baseline_results.get(operation) {
            let speedup = baseline.execution_time.as_secs_f64()
                / current_metrics.execution_time.as_secs_f64();
            let throughput_improvement = current_metrics.throughput / baseline.throughput;

            let recommendation = if speedup > 1.1 {
                "Performance improved".to_string()
            } else if speedup < 0.9 {
                "Performance regressed".to_string()
            } else {
                "Performance is stable".to_string()
            };

            Some(ComparisonResult {
                baseline_metrics: baseline.clone(),
                comparison_metrics: current_metrics.clone(),
                speedup,
                memory_improvement: 0.0,
                throughput_improvement,
                recommendation,
            })
        } else {
            None
        }
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[PerformanceMetrics] {
        &self.results
    }

    /// Clear all results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Helper function to create test tensors
    fn create_test_tensor(&self, shape: &[usize], device: &BitNetMlxDevice) -> Result<MlxTensor> {
        use crate::tensor::dtype::BitNetDType;
        MlxTensor::zeros(shape, BitNetDType::F32, device.clone())
    }

    /// Helper function to perform matrix multiplication
    fn perform_matmul(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        use crate::mlx::operations::BitNetMlxOps;
        BitNetMlxOps::matmul(a, b)
    }

    /// Helper function to perform quantization
    fn perform_quantization(&self, tensor: &MlxTensor) -> Result<MlxTensor> {
        use crate::mlx::operations::BitNetMlxOps;
        BitNetMlxOps::quantize_1_58_bit(tensor, None)
    }

    /// Helper function to perform element-wise operations
    fn perform_elementwise(
        &self,
        a: &MlxTensor,
        b: &MlxTensor,
        operation: &str,
    ) -> Result<MlxTensor> {
        use crate::mlx::operations::BitNetMlxOps;
        match operation {
            "add" => BitNetMlxOps::add(a, b),
            "multiply" => BitNetMlxOps::multiply(a, b),
            _ => Err(anyhow::anyhow!("Unsupported operation: {}", operation)),
        }
    }
}

/// Performance report generator
pub struct PerformanceReportGenerator;

impl PerformanceReportGenerator {
    /// Generate a comprehensive performance report
    pub fn generate_report(
        metrics: &[PerformanceMetrics],
        comparisons: &[ComparisonResult],
    ) -> String {
        let mut report = String::new();

        report.push_str("# MLX Performance Report\n\n");
        report.push_str(&format!(
            "Generated at: {:?}\n\n",
            std::time::SystemTime::now()
        ));

        // Summary statistics
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "Total operations benchmarked: {}\n",
            metrics.len()
        ));
        report.push_str(&format!("Device comparisons: {}\n\n", comparisons.len()));

        // Individual metrics
        report.push_str("## Individual Metrics\n\n");
        for metric in metrics {
            report.push_str(&format!(
                "### {} on {}\n",
                metric.operation_name, metric.device_type
            ));
            report.push_str(&format!("- Execution time: {:?}\n", metric.execution_time));
            report.push_str(&format!("- Throughput: {:.2} ops/sec\n", metric.throughput));
            report.push_str(&format!("- Tensor shapes: {:?}\n", metric.tensor_shapes));
            report.push_str(&format!("- Data type: {}\n\n", metric.data_type));
        }

        // Comparisons
        report.push_str("## Device Comparisons\n\n");
        for comparison in comparisons {
            report.push_str(&format!(
                "### {} vs {}\n",
                comparison.baseline_metrics.device_type, comparison.comparison_metrics.device_type
            ));
            report.push_str(&format!("- Speedup: {:.2}x\n", comparison.speedup));
            report.push_str(&format!(
                "- Throughput improvement: {:.2}x\n",
                comparison.throughput_improvement
            ));
            report.push_str(&format!(
                "- Recommendation: {}\n\n",
                comparison.recommendation
            ));
        }

        report
    }

    /// Generate JSON report
    pub fn generate_json_report(
        metrics: &[PerformanceMetrics],
        comparisons: &[ComparisonResult],
    ) -> Result<String> {
        let report = serde_json::json!({
            "timestamp": std::time::SystemTime::now(),
            "metrics": metrics,
            "comparisons": comparisons,
            "summary": {
                "total_operations": metrics.len(),
                "total_comparisons": comparisons.len(),
            }
        });

        serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("Failed to serialize report: {}", e))
    }

    /// Generate CSV report for metrics
    pub fn generate_csv_report(metrics: &[PerformanceMetrics]) -> String {
        let mut csv = String::new();
        csv.push_str("operation,device,execution_time_ms,throughput,data_type,timestamp\n");

        for metric in metrics {
            csv.push_str(&format!(
                "{},{},{:.3},{:.2},{},{:?}\n",
                metric.operation_name,
                metric.device_type,
                metric.execution_time.as_millis(),
                metric.throughput,
                metric.data_type,
                metric.timestamp
            ));
        }

        csv
    }
}

/// Performance regression detector
#[allow(dead_code)]
pub struct RegressionDetector {
    threshold: f64, // Performance degradation threshold (e.g., 0.1 for 10%)
    baseline_metrics: HashMap<String, PerformanceMetrics>,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            baseline_metrics: HashMap::new(),
        }
    }

    /// Set baseline metrics
    pub fn set_baseline(&mut self, metrics: Vec<PerformanceMetrics>) {
        for metric in metrics {
            let key = format!("{}_{}", metric.operation_name, metric.device_type);
            self.baseline_metrics.insert(key, metric);
        }
    }

    /// Check for performance regressions
    pub fn check_regression(&self, current_metrics: &[PerformanceMetrics]) -> Vec<String> {
        let mut regressions = Vec::new();

        for metric in current_metrics {
            let key = format!("{}_{}", metric.operation_name, metric.device_type);

            if let Some(baseline) = self.baseline_metrics.get(&key) {
                let performance_ratio =
                    metric.execution_time.as_secs_f64() / baseline.execution_time.as_secs_f64();

                if performance_ratio > (1.0 + self.threshold) {
                    let degradation = (performance_ratio - 1.0) * 100.0;
                    regressions.push(format!(
                        "Performance regression detected in {} on {}: {:.1}% slower",
                        metric.operation_name, metric.device_type, degradation
                    ));
                }
            }
        }

        regressions
    }
}

// Stub implementations when MLX is not available
#[cfg(not(feature = "mlx"))]
pub struct MlxPerformanceBenchmarker;

#[cfg(not(feature = "mlx"))]
impl MlxPerformanceBenchmarker {
    pub fn new(_config: BenchmarkConfig) -> Self {
        Self
    }

    pub fn benchmark_matmul(&mut self, _device: &()) -> Result<PerformanceMetrics> {
        anyhow::bail!("MLX support not compiled in")
    }
}
