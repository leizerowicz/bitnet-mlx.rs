//! Performance Comparison Framework
//!
//! This module provides utilities for comparing MLX and Candle performance
//! across different operations, tensor sizes, and device configurations.

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{operations::BitNetMlxOps, BitNetMlxDevice, MlxTensor};

use crate::candle_ops::CandleOps;

/// Configuration for performance comparison tests
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ComparisonConfig {
    /// Tensor sizes to test (rows, cols)
    pub tensor_sizes: Vec<(usize, usize)>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Operations to benchmark
    pub operations: Vec<String>,
    /// Devices to test
    pub devices: Vec<String>,
    /// Data types to test
    pub data_types: Vec<String>,
    /// Timeout for individual benchmarks
    pub timeout: Duration,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            tensor_sizes: vec![
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
            ],
            warmup_iterations: 5,
            measurement_iterations: 10,
            operations: vec![
                "matmul".to_string(),
                "add".to_string(),
                "multiply".to_string(),
                "quantize".to_string(),
                "bitlinear".to_string(),
            ],
            devices: vec!["cpu".to_string(), "metal".to_string(), "mlx".to_string()],
            data_types: vec!["f32".to_string(), "f16".to_string()],
            timeout: Duration::from_secs(30),
        }
    }
}

/// Performance measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PerformanceMeasurement {
    pub operation: String,
    pub backend: String,
    pub device: String,
    pub tensor_size: (usize, usize),
    pub data_type: String,
    pub execution_time: Duration,
    pub throughput: f64,     // operations per second
    pub memory_usage: usize, // bytes
    pub success: bool,
    pub error_message: Option<String>,
    pub timestamp: std::time::SystemTime,
}

/// Comparison result between two backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ComparisonResult {
    pub operation: String,
    pub tensor_size: (usize, usize),
    pub baseline_backend: String,
    pub comparison_backend: String,
    pub baseline_time: Duration,
    pub comparison_time: Duration,
    pub speedup: f64,
    pub throughput_ratio: f64,
    pub memory_ratio: f64,
    pub recommendation: String,
}

/// Performance comparison framework
#[allow(dead_code)]
pub struct PerformanceComparator {
    config: ComparisonConfig,
    measurements: Vec<PerformanceMeasurement>,
}

impl PerformanceComparator {
    /// Create a new performance comparator
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
        }
    }

    /// Run comprehensive performance comparison
    pub fn run_comparison(&mut self) -> anyhow::Result<Vec<ComparisonResult>> {
        println!("Starting performance comparison...");

        // Run benchmarks for each configuration
        for operation in &self.config.operations.clone() {
            for &tensor_size in &self.config.tensor_sizes {
                for data_type in &self.config.data_types.clone() {
                    // Benchmark Candle CPU
                    if let Ok(measurement) =
                        self.benchmark_candle_cpu(operation, tensor_size, data_type)
                    {
                        self.measurements.push(measurement);
                    }

                    // Benchmark Candle Metal (if available)
                    #[cfg(target_os = "macos")]
                    if Device::new_metal(0).is_ok() {
                        if let Ok(measurement) =
                            self.benchmark_candle_metal(operation, tensor_size, data_type)
                        {
                            self.measurements.push(measurement);
                        }
                    }

                    // Benchmark MLX (if available)
                    #[cfg(feature = "mlx")]
                    if let Ok(measurement) = self.benchmark_mlx(operation, tensor_size, data_type) {
                        self.measurements.push(measurement);
                    }
                }
            }
        }

        // Generate comparison results
        self.generate_comparisons()
    }

    /// Benchmark Candle CPU operations
    fn benchmark_candle_cpu(
        &self,
        operation: &str,
        tensor_size: (usize, usize),
        data_type: &str,
    ) -> anyhow::Result<PerformanceMeasurement> {
        let device = Device::Cpu;
        let dtype = match data_type {
            "f32" => DType::F32,
            "f16" => DType::F16,
            _ => DType::F32,
        };

        let start_time = Instant::now();
        let mut total_time = Duration::ZERO;
        let mut memory_usage = 0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = self.execute_candle_operation(operation, tensor_size, &device, dtype)?;
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let iter_start = Instant::now();
            let result = self.execute_candle_operation(operation, tensor_size, &device, dtype)?;
            let iter_time = iter_start.elapsed();
            total_time += iter_time;

            // Estimate memory usage from result tensor
            if let Some(tensor) = result {
                memory_usage = self.estimate_tensor_memory(&tensor);
            }
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(PerformanceMeasurement {
            operation: operation.to_string(),
            backend: "candle".to_string(),
            device: "cpu".to_string(),
            tensor_size,
            data_type: data_type.to_string(),
            execution_time: avg_time,
            throughput,
            memory_usage,
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Benchmark Candle Metal operations
    #[cfg(target_os = "macos")]
    fn benchmark_candle_metal(
        &self,
        operation: &str,
        tensor_size: (usize, usize),
        data_type: &str,
    ) -> anyhow::Result<PerformanceMeasurement> {
        let device = Device::new_metal(0)?;
        let dtype = match data_type {
            "f32" => DType::F32,
            "f16" => DType::F16,
            _ => DType::F32,
        };

        let mut total_time = Duration::ZERO;
        let mut memory_usage = 0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = self.execute_candle_operation(operation, tensor_size, &device, dtype)?;
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let iter_start = Instant::now();
            let result = self.execute_candle_operation(operation, tensor_size, &device, dtype)?;
            let iter_time = iter_start.elapsed();
            total_time += iter_time;

            if let Some(tensor) = result {
                memory_usage = self.estimate_tensor_memory(&tensor);
            }
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(PerformanceMeasurement {
            operation: operation.to_string(),
            backend: "candle".to_string(),
            device: "metal".to_string(),
            tensor_size,
            data_type: data_type.to_string(),
            execution_time: avg_time,
            throughput,
            memory_usage,
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Benchmark MLX operations
    #[cfg(feature = "mlx")]
    fn benchmark_mlx(
        &self,
        operation: &str,
        tensor_size: (usize, usize),
        data_type: &str,
    ) -> anyhow::Result<PerformanceMeasurement> {
        use bitnet_core::tensor::BitNetDType;

        let device = BitNetMlxDevice::default()?;
        let dtype = match data_type {
            "f32" => BitNetDType::F32,
            "f16" => BitNetDType::F16,
            _ => BitNetDType::F32,
        };

        let mut total_time = Duration::ZERO;
        let mut memory_usage = 0;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = self.execute_mlx_operation(operation, tensor_size, &device, dtype)?;
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let iter_start = Instant::now();
            let result = self.execute_mlx_operation(operation, tensor_size, &device, dtype)?;
            let iter_time = iter_start.elapsed();
            total_time += iter_time;

            if let Some(tensor) = result {
                memory_usage = self.estimate_mlx_tensor_memory(&tensor);
            }
        }

        let avg_time = total_time / self.config.measurement_iterations as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(PerformanceMeasurement {
            operation: operation.to_string(),
            backend: "mlx".to_string(),
            device: "mlx".to_string(),
            tensor_size,
            data_type: data_type.to_string(),
            execution_time: avg_time,
            throughput,
            memory_usage,
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Execute Candle operation
    fn execute_candle_operation(
        &self,
        operation: &str,
        tensor_size: (usize, usize),
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Option<Tensor>> {
        let (rows, cols) = tensor_size;

        match operation {
            "matmul" => {
                let a = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let b = Tensor::randn(0f32, 1f32, (cols, rows), device)?;
                let result = CandleOps::matmul(&a, &b)?;
                Ok(Some(result))
            }
            "add" => {
                let a = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let b = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let result = CandleOps::add(&a, &b)?;
                Ok(Some(result))
            }
            "multiply" => {
                let a = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let b = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let result = CandleOps::multiply(&a, &b)?;
                Ok(Some(result))
            }
            "quantize" => {
                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let result = CandleOps::quantize_1_58_bit(&tensor, Some(0.1))?;
                Ok(Some(result))
            }
            "bitlinear" => {
                let batch_size = 32;
                let input = Tensor::randn(0f32, 1f32, (batch_size, rows), device)?;
                let weight = Tensor::randn(0f32, 1f32, (rows, cols), device)?;
                let bias = Tensor::randn(0f32, 1f32, (cols,), device)?;
                let result = CandleOps::bitlinear_forward(&input, &weight, Some(&bias), true)?;
                Ok(Some(result))
            }
            _ => Ok(None),
        }
    }

    /// Execute MLX operation
    #[cfg(feature = "mlx")]
    fn execute_mlx_operation(
        &self,
        operation: &str,
        tensor_size: (usize, usize),
        device: &BitNetMlxDevice,
        dtype: bitnet_core::tensor::BitNetDType,
    ) -> anyhow::Result<Option<MlxTensor>> {
        let (rows, cols) = tensor_size;

        match operation {
            "matmul" => {
                let a = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let b = MlxTensor::randn(&[cols, rows], dtype, device.clone())?;
                let result = BitNetMlxOps::matmul(&a, &b)?;
                Ok(Some(result))
            }
            "add" => {
                let a = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let b = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let result = BitNetMlxOps::add(&a, &b)?;
                Ok(Some(result))
            }
            "multiply" => {
                let a = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let b = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let result = BitNetMlxOps::multiply(&a, &b)?;
                Ok(Some(result))
            }
            "quantize" => {
                let tensor = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let result = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1))?;
                Ok(Some(result))
            }
            "bitlinear" => {
                let batch_size = 32;
                let input = MlxTensor::randn(&[batch_size, rows], dtype, device.clone())?;
                let weight = MlxTensor::randn(&[rows, cols], dtype, device.clone())?;
                let bias = MlxTensor::randn(&[cols], dtype, device.clone())?;
                let result = BitNetMlxOps::bitlinear_forward(&input, &weight, Some(&bias), true)?;
                Ok(Some(result))
            }
            _ => Ok(None),
        }
    }

    /// Estimate memory usage of a Candle tensor
    fn estimate_tensor_memory(&self, tensor: &Tensor) -> usize {
        let element_count: usize = tensor.shape().dims().iter().product();
        let dtype_size = match tensor.dtype() {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I64 => 8,
            DType::U32 => 4,
            DType::U8 => 1,
            _ => 4,
        };
        element_count * dtype_size
    }

    /// Estimate memory usage of an MLX tensor
    #[cfg(feature = "mlx")]
    fn estimate_mlx_tensor_memory(&self, tensor: &MlxTensor) -> usize {
        let element_count: usize = tensor.shape().iter().product();
        let dtype_size = match tensor.dtype() {
            bitnet_core::tensor::BitNetDType::F32 => 4,
            bitnet_core::tensor::BitNetDType::F16 => 2,
            bitnet_core::tensor::BitNetDType::I8 => 1,
            _ => 4,
        };
        element_count * dtype_size
    }

    /// Generate comparison results
    fn generate_comparisons(&self) -> anyhow::Result<Vec<ComparisonResult>> {
        let mut comparisons = Vec::new();

        // Group measurements by operation and tensor size
        let mut grouped: HashMap<(String, (usize, usize)), Vec<&PerformanceMeasurement>> =
            HashMap::new();

        for measurement in &self.measurements {
            let key = (measurement.operation.clone(), measurement.tensor_size);
            grouped.entry(key).or_default().push(measurement);
        }

        // Generate comparisons for each group
        for ((operation, tensor_size), measurements) in grouped {
            // Find baseline (usually CPU Candle)
            let baseline = measurements
                .iter()
                .find(|m| m.backend == "candle" && m.device == "cpu")
                .or_else(|| measurements.first());

            if let Some(baseline) = baseline {
                for measurement in &measurements {
                    if measurement.backend != baseline.backend
                        || measurement.device != baseline.device
                    {
                        let speedup = baseline.execution_time.as_secs_f64()
                            / measurement.execution_time.as_secs_f64();
                        let throughput_ratio = measurement.throughput / baseline.throughput;
                        let memory_ratio =
                            measurement.memory_usage as f64 / baseline.memory_usage as f64;

                        let recommendation = if speedup > 1.5 {
                            format!(
                                "Use {} for better performance ({:.2}x speedup)",
                                measurement.backend, speedup
                            )
                        } else if speedup < 0.8 {
                            format!("Use {} for better performance", baseline.backend)
                        } else {
                            "Performance is similar".to_string()
                        };

                        comparisons.push(ComparisonResult {
                            operation: operation.clone(),
                            tensor_size,
                            baseline_backend: format!("{}_{}", baseline.backend, baseline.device),
                            comparison_backend: format!(
                                "{}_{}",
                                measurement.backend, measurement.device
                            ),
                            baseline_time: baseline.execution_time,
                            comparison_time: measurement.execution_time,
                            speedup,
                            throughput_ratio,
                            memory_ratio,
                            recommendation,
                        });
                    }
                }
            }
        }

        Ok(comparisons)
    }

    /// Get all measurements
    pub fn get_measurements(&self) -> &[PerformanceMeasurement] {
        &self.measurements
    }

    /// Export results to JSON
    pub fn export_json(&self) -> anyhow::Result<String> {
        let data = serde_json::json!({
            "config": self.config,
            "measurements": self.measurements,
            "timestamp": std::time::SystemTime::now(),
        });

        serde_json::to_string_pretty(&data)
            .map_err(|e| anyhow::anyhow!("Failed to serialize results: {}", e))
    }

    /// Export results to CSV
    pub fn export_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("operation,backend,device,tensor_size,data_type,execution_time_ms,throughput,memory_usage,success\n");

        for measurement in &self.measurements {
            csv.push_str(&format!(
                "{},{},{},{}x{},{},{:.3},{:.2},{},{}\n",
                measurement.operation,
                measurement.backend,
                measurement.device,
                measurement.tensor_size.0,
                measurement.tensor_size.1,
                measurement.data_type,
                measurement.execution_time.as_millis(),
                measurement.throughput,
                measurement.memory_usage,
                measurement.success
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_config_default() {
        let config = ComparisonConfig::default();
        assert!(!config.tensor_sizes.is_empty());
        assert!(!config.operations.is_empty());
        assert!(config.warmup_iterations > 0);
        assert!(config.measurement_iterations > 0);
    }

    #[test]
    fn test_performance_comparator_creation() {
        let config = ComparisonConfig::default();
        let comparator = PerformanceComparator::new(config);
        assert_eq!(comparator.measurements.len(), 0);
    }
}
