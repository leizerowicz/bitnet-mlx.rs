//! MLX Performance Regression Testing
//! 
//! This module provides comprehensive regression testing capabilities for MLX operations,
//! including baseline management, automated testing, and regression detection.

use crate::mlx::{
    performance::{MlxPerformanceBenchmarker, PerformanceMetrics, BenchmarkConfig},
    memory_tracker::{MlxMemoryTracker, MemoryEvent},
    metrics::{MlxMetricsCollector, MlxMetrics, OperationContext},
    device_comparison::{MlxDeviceComparison, DeviceComparisonConfig},
};
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// Regression testing configuration
#[derive(Debug, Clone)]
pub struct RegressionTestConfig {
    pub baseline_directory: PathBuf,
    pub test_operations: Vec<String>,
    pub test_devices: Vec<String>,
    pub tensor_sizes: Vec<Vec<usize>>,
    pub data_types: Vec<String>,
    pub regression_threshold: f64, // Percentage threshold for regression detection
    pub improvement_threshold: f64, // Percentage threshold for improvement detection
    pub iterations_per_test: usize,
    pub warmup_iterations: usize,
    pub enable_memory_regression_testing: bool,
    pub enable_automated_bisection: bool,
    pub max_bisection_iterations: usize,
    pub test_timeout: Duration,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        Self {
            baseline_directory: PathBuf::from("./baselines"),
            test_operations: vec![
                "matmul".to_string(),
                "quantization".to_string(),
                "add".to_string(),
                "multiply".to_string(),
            ],
            test_devices: vec!["cpu".to_string(), "gpu".to_string()],
            tensor_sizes: vec![
                vec![128, 128],
                vec![512, 512],
                vec![1024, 1024],
                vec![2048, 2048],
            ],
            data_types: vec!["f32".to_string(), "f16".to_string()],
            regression_threshold: 10.0, // 10% performance degradation
            improvement_threshold: 5.0,  // 5% performance improvement
            iterations_per_test: 5,
            warmup_iterations: 2,
            enable_memory_regression_testing: true,
            enable_automated_bisection: true,
            max_bisection_iterations: 10,
            test_timeout: Duration::from_secs(300),
        }
    }
}

/// Baseline performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub baseline_id: String,
    pub created_at: SystemTime,
    pub version_info: VersionInfo,
    pub system_info: SystemInfo,
    pub performance_metrics: HashMap<String, BaselineMetrics>, // test_key -> metrics
    pub memory_baselines: HashMap<String, MemoryBaseline>,
    pub metadata: BaselineMetadata,
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub mlx_version: String,
    pub rust_version: String,
    pub bitnet_version: String,
    pub commit_hash: Option<String>,
    pub build_timestamp: SystemTime,
}

/// System information for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub architecture: String,
    pub cpu_model: String,
    pub gpu_model: Option<String>,
    pub total_memory: usize,
    pub gpu_memory: Option<usize>,
    pub cpu_cores: u32,
    pub gpu_cores: Option<u32>,
}

/// Baseline metrics for a specific test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub test_key: String,
    pub operation: String,
    pub device: String,
    pub tensor_size: Vec<usize>,
    pub data_type: String,
    pub average_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub throughput: f64,
    pub memory_usage: f64,
    pub standard_deviation: Duration,
    pub confidence_interval: (Duration, Duration),
    pub sample_count: usize,
}

/// Memory baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBaseline {
    pub test_key: String,
    pub peak_memory_usage: usize,
    pub average_memory_usage: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub memory_efficiency: f64,
    pub fragmentation_ratio: f64,
}

/// Baseline metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetadata {
    pub description: String,
    pub tags: Vec<String>,
    pub test_environment: String,
    pub notes: Option<String>,
    pub is_stable: bool,
    pub validation_status: ValidationStatus,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Validated,
    Failed,
    Deprecated,
}

/// Regression test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult {
    pub test_id: String,
    pub timestamp: SystemTime,
    pub baseline_id: String,
    pub test_config: RegressionTestConfigSerialized,
    pub test_results: Vec<TestCaseResult>,
    pub regressions_detected: Vec<RegressionDetection>,
    pub improvements_detected: Vec<ImprovementDetection>,
    pub memory_regressions: Vec<MemoryRegressionDetection>,
    pub overall_status: TestStatus,
    pub summary: RegressionTestSummary,
}

/// Serializable regression test config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestConfigSerialized {
    pub test_operations: Vec<String>,
    pub test_devices: Vec<String>,
    pub tensor_sizes: Vec<Vec<usize>>,
    pub data_types: Vec<String>,
    pub regression_threshold: f64,
    pub improvement_threshold: f64,
    pub iterations_per_test: usize,
}

/// Individual test case result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    pub test_key: String,
    pub operation: String,
    pub device: String,
    pub tensor_size: Vec<usize>,
    pub data_type: String,
    pub current_metrics: BaselineMetrics,
    pub baseline_metrics: BaselineMetrics,
    pub performance_change: PerformanceChange,
    pub status: TestCaseStatus,
}

/// Performance change analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceChange {
    pub execution_time_change: f64, // Percentage change
    pub throughput_change: f64,
    pub memory_usage_change: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
}

/// Test case status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestCaseStatus {
    Pass,
    Regression,
    Improvement,
    Unstable,
    Failed,
}

/// Regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetection {
    pub test_key: String,
    pub operation: String,
    pub device: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub performance_degradation: f64, // Percentage
    pub baseline_value: f64,
    pub current_value: f64,
    pub confidence: f64,
    pub potential_causes: Vec<String>,
    pub bisection_result: Option<BisectionResult>,
}

/// Types of regressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionType {
    PerformanceRegression,
    MemoryRegression,
    ThroughputRegression,
    LatencyRegression,
    EfficiencyRegression,
}

/// Regression severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,      // < 15% degradation
    Moderate,   // 15-30% degradation
    Major,      // 30-50% degradation
    Critical,   // > 50% degradation
}

/// Improvement detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementDetection {
    pub test_key: String,
    pub operation: String,
    pub device: String,
    pub improvement_type: ImprovementType,
    pub performance_improvement: f64, // Percentage
    pub baseline_value: f64,
    pub current_value: f64,
    pub confidence: f64,
    pub likely_causes: Vec<String>,
}

/// Types of improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementType {
    PerformanceImprovement,
    MemoryImprovement,
    ThroughputImprovement,
    LatencyImprovement,
    EfficiencyImprovement,
}

/// Memory regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegressionDetection {
    pub test_key: String,
    pub operation: String,
    pub device: String,
    pub memory_increase: f64, // Percentage
    pub baseline_memory: usize,
    pub current_memory: usize,
    pub regression_type: MemoryRegressionType,
    pub potential_leak: bool,
}

/// Memory regression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryRegressionType {
    PeakMemoryIncrease,
    AverageMemoryIncrease,
    AllocationIncrease,
    FragmentationIncrease,
    EfficiencyDecrease,
}

/// Overall test status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    AllPass,
    HasRegressions,
    HasCriticalRegressions,
    HasFailures,
    Incomplete,
}

/// Regression test summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub regression_count: usize,
    pub improvement_count: usize,
    pub critical_regressions: usize,
    pub memory_regressions: usize,
    pub overall_performance_change: f64,
    pub recommendations: Vec<String>,
}

/// Bisection result for automated regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisectionResult {
    pub suspected_commit: Option<String>,
    pub commit_range: (String, String),
    pub iterations_performed: usize,
    pub confidence: f64,
    pub analysis_notes: Vec<String>,
}

/// MLX Regression Testing Engine
pub struct MlxRegressionTester {
    config: RegressionTestConfig,
    benchmarker: MlxPerformanceBenchmarker,
    memory_tracker: MlxMemoryTracker,
    device_comparison: MlxDeviceComparison,
    baseline_manager: BaselineManager,
}

impl MlxRegressionTester {
    /// Create a new regression tester
    pub fn new(config: RegressionTestConfig) -> Self {
        let benchmark_config = BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.iterations_per_test,
            tensor_sizes: config.tensor_sizes.clone(),
            data_types: config.data_types.clone(),
            devices: config.test_devices.clone(),
            timeout: config.test_timeout,
        };

        let device_comparison_config = DeviceComparisonConfig {
            devices_to_compare: config.test_devices.clone(),
            operations_to_test: config.test_operations.clone(),
            tensor_sizes: config.tensor_sizes.clone(),
            data_types: config.data_types.clone(),
            iterations_per_test: config.iterations_per_test,
            warmup_iterations: config.warmup_iterations,
            enable_memory_analysis: config.enable_memory_regression_testing,
            enable_profiling: false, // Disable for regression testing
            enable_power_analysis: false,
            comparison_timeout: config.test_timeout,
        };

        Self {
            config: config.clone(),
            benchmarker: MlxPerformanceBenchmarker::new(benchmark_config),
            memory_tracker: MlxMemoryTracker::new(),
            device_comparison: MlxDeviceComparison::new(device_comparison_config),
            baseline_manager: BaselineManager::new(config.baseline_directory),
        }
    }

    /// Create a new performance baseline
    pub fn create_baseline(&mut self, baseline_id: String, description: String) -> Result<PerformanceBaseline> {
        let mut performance_metrics = HashMap::new();
        let mut memory_baselines = HashMap::new();

        // Run benchmarks for all test cases
        let operations = self.config.test_operations.clone();
        let devices = self.config.test_devices.clone();
        let tensor_sizes = self.config.tensor_sizes.clone();
        let data_types = self.config.data_types.clone();
        
        for operation in &operations {
            for device_name in &devices {
                for tensor_size in &tensor_sizes {
                    for data_type in &data_types {
                        let test_key = self.generate_test_key(operation, device_name, tensor_size, data_type);
                        
                        // Run performance benchmark
                        let device = self.create_device(device_name)?;
                        let metrics = self.run_baseline_benchmark(&device, operation, tensor_size, data_type)?;
                        performance_metrics.insert(test_key.clone(), metrics);

                        // Run memory benchmark if enabled
                        if self.config.enable_memory_regression_testing {
                            let memory_baseline = self.run_memory_baseline(&device, operation, tensor_size, data_type)?;
                            memory_baselines.insert(test_key, memory_baseline);
                        }
                    }
                }
            }
        }

        let baseline = PerformanceBaseline {
            baseline_id: baseline_id.clone(),
            created_at: SystemTime::now(),
            version_info: self.collect_version_info()?,
            system_info: self.collect_system_info()?,
            performance_metrics,
            memory_baselines,
            metadata: BaselineMetadata {
                description,
                tags: vec!["auto-generated".to_string()],
                test_environment: "development".to_string(),
                notes: None,
                is_stable: false,
                validation_status: ValidationStatus::Pending,
            },
        };

        // Save baseline
        self.baseline_manager.save_baseline(&baseline)?;

        Ok(baseline)
    }

    /// Run regression tests against a baseline
    pub fn run_regression_tests(&mut self, baseline_id: &str) -> Result<RegressionTestResult> {
        let baseline = self.baseline_manager.load_baseline(baseline_id)?;
        let test_id = format!("regression_test_{}", SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs());

        let mut test_results = Vec::new();
        let mut regressions_detected = Vec::new();
        let mut improvements_detected = Vec::new();
        let mut memory_regressions = Vec::new();

        // Run tests for each baseline entry
        for (test_key, baseline_metrics) in &baseline.performance_metrics {
            let current_metrics = self.run_current_benchmark(
                &baseline_metrics.operation,
                &baseline_metrics.device,
                &baseline_metrics.tensor_size,
                &baseline_metrics.data_type,
            )?;

            let performance_change = self.calculate_performance_change(baseline_metrics, &current_metrics)?;
            let status = self.determine_test_status(&performance_change)?;

            let test_case_result = TestCaseResult {
                test_key: test_key.clone(),
                operation: baseline_metrics.operation.clone(),
                device: baseline_metrics.device.clone(),
                tensor_size: baseline_metrics.tensor_size.clone(),
                data_type: baseline_metrics.data_type.clone(),
                current_metrics,
                baseline_metrics: baseline_metrics.clone(),
                performance_change: performance_change.clone(),
                status: status.clone(),
            };

            test_results.push(test_case_result);

            // Check for regressions
            if matches!(status, TestCaseStatus::Regression) {
                let regression = self.analyze_regression(baseline_metrics, &performance_change)?;
                regressions_detected.push(regression);
            }

            // Check for improvements
            if matches!(status, TestCaseStatus::Improvement) {
                let improvement = self.analyze_improvement(baseline_metrics, &performance_change)?;
                improvements_detected.push(improvement);
            }

            // Check for memory regressions
            if self.config.enable_memory_regression_testing {
                if let Some(memory_baseline) = baseline.memory_baselines.get(test_key) {
                    if let Some(memory_regression) = self.check_memory_regression(memory_baseline, test_key)? {
                        memory_regressions.push(memory_regression);
                    }
                }
            }
        }

        let overall_status = self.determine_overall_status(&test_results, &regressions_detected)?;
        let summary = self.generate_test_summary(&test_results, &regressions_detected, &improvements_detected)?;

        Ok(RegressionTestResult {
            test_id,
            timestamp: SystemTime::now(),
            baseline_id: baseline_id.to_string(),
            test_config: RegressionTestConfigSerialized {
                test_operations: self.config.test_operations.clone(),
                test_devices: self.config.test_devices.clone(),
                tensor_sizes: self.config.tensor_sizes.clone(),
                data_types: self.config.data_types.clone(),
                regression_threshold: self.config.regression_threshold,
                improvement_threshold: self.config.improvement_threshold,
                iterations_per_test: self.config.iterations_per_test,
            },
            test_results,
            regressions_detected,
            improvements_detected,
            memory_regressions,
            overall_status,
            summary,
        })
    }

    /// Helper methods
    fn generate_test_key(&self, operation: &str, device: &str, tensor_size: &[usize], data_type: &str) -> String {
        format!("{}_{}_{}_{}", operation, device, format!("{:?}", tensor_size), data_type)
    }

    fn create_device(&self, device_name: &str) -> Result<crate::mlx::BitNetMlxDevice> {
        match device_name {
            "cpu" => crate::mlx::BitNetMlxDevice::cpu(),
            "gpu" => crate::mlx::BitNetMlxDevice::gpu(),
            _ => Err(anyhow::anyhow!("Unknown device: {}", device_name)),
        }
    }

    fn run_baseline_benchmark(&mut self, device: &crate::mlx::BitNetMlxDevice, operation: &str, tensor_size: &[usize], data_type: &str) -> Result<BaselineMetrics> {
        let mut execution_times = Vec::new();
        let mut throughputs = Vec::new();
        let mut memory_usages = Vec::new();

        // Run multiple iterations
        for _ in 0..self.config.iterations_per_test {
            let metrics = match operation {
                "matmul" => self.benchmarker.benchmark_matmul(device)?,
                "quantization" => self.benchmarker.benchmark_quantization(device)?,
                op => self.benchmarker.benchmark_elementwise(device, op)?,
            };

            execution_times.push(metrics.execution_time);
            throughputs.push(metrics.throughput);
            memory_usages.push(metrics.memory_usage.allocated_memory_mb);
        }

        // Calculate statistics
        let average_execution_time = Duration::from_nanos(
            (execution_times.iter().map(|d| d.as_nanos()).sum::<u128>() / execution_times.len() as u128).try_into().unwrap_or(0)
        );
        let min_execution_time = *execution_times.iter().min().unwrap();
        let max_execution_time = *execution_times.iter().max().unwrap();
        let average_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let average_memory_usage = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;

        // Calculate standard deviation
        let mean_nanos = average_execution_time.as_nanos() as f64;
        let variance = execution_times.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>() / execution_times.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        // Calculate confidence interval (95%)
        let margin = std_dev.as_nanos() as f64 * 1.96 / (execution_times.len() as f64).sqrt();
        let confidence_interval = (
            Duration::from_nanos((mean_nanos - margin) as u64),
            Duration::from_nanos((mean_nanos + margin) as u64),
        );

        Ok(BaselineMetrics {
            test_key: self.generate_test_key(operation, device.device_type(), tensor_size, data_type),
            operation: operation.to_string(),
            device: device.device_type().to_string(),
            tensor_size: tensor_size.to_vec(),
            data_type: data_type.to_string(),
            average_execution_time,
            min_execution_time,
            max_execution_time,
            throughput: average_throughput,
            memory_usage: average_memory_usage,
            standard_deviation: std_dev,
            confidence_interval,
            sample_count: execution_times.len(),
        })
    }

    fn run_memory_baseline(&self, device: &crate::mlx::BitNetMlxDevice, operation: &str, tensor_size: &[usize], data_type: &str) -> Result<MemoryBaseline> {
        // Placeholder implementation
        Ok(MemoryBaseline {
            test_key: self.generate_test_key(operation, device.device_type(), tensor_size, data_type),
            peak_memory_usage: 1024 * 1024, // 1MB
            average_memory_usage: 512 * 1024, // 512KB
            allocation_count: 10,
            deallocation_count: 10,
            memory_efficiency: 0.9,
            fragmentation_ratio: 0.1,
        })
    }

    fn run_current_benchmark(&mut self, operation: &str, device_name: &str, tensor_size: &[usize], data_type: &str) -> Result<BaselineMetrics> {
        let device = self.create_device(device_name)?;
        self.run_baseline_benchmark(&device, operation, tensor_size, data_type)
    }

    fn calculate_performance_change(&self, baseline: &BaselineMetrics, current: &BaselineMetrics) -> Result<PerformanceChange> {
        let execution_time_change = ((current.average_execution_time.as_secs_f64() - baseline.average_execution_time.as_secs_f64()) 
            / baseline.average_execution_time.as_secs_f64()) * 100.0;
        
        let throughput_change = ((current.throughput - baseline.throughput) / baseline.throughput) * 100.0;
        let memory_usage_change = ((current.memory_usage - baseline.memory_usage) / baseline.memory_usage) * 100.0;

        // Simple significance test based on confidence intervals
        let is_significant = !self.confidence_intervals_overlap(&baseline.confidence_interval, &current.confidence_interval);

        Ok(PerformanceChange {
            execution_time_change,
            throughput_change,
            memory_usage_change,
            is_significant,
            confidence_level: if is_significant { 0.95 } else { 0.5 },
        })
    }

    fn confidence_intervals_overlap(&self, interval1: &(Duration, Duration), interval2: &(Duration, Duration)) -> bool {
        interval1.1 >= interval2.0 && interval2.1 >= interval1.0
    }

    fn determine_test_status(&self, performance_change: &PerformanceChange) -> Result<TestCaseStatus> {
        if !performance_change.is_significant {
            return Ok(TestCaseStatus::Pass);
        }

        if performance_change.execution_time_change > self.config.regression_threshold {
            Ok(TestCaseStatus::Regression)
        } else if performance_change.execution_time_change < -self.config.improvement_threshold {
            Ok(TestCaseStatus::Improvement)
        } else {
            Ok(TestCaseStatus::Pass)
        }
    }

    fn analyze_regression(&self, baseline: &BaselineMetrics, performance_change: &PerformanceChange) -> Result<RegressionDetection> {
        let severity = if performance_change.execution_time_change > 50.0 {
            RegressionSeverity::Critical
        } else if performance_change.execution_time_change > 30.0 {
            RegressionSeverity::Major
        } else if performance_change.execution_time_change > 15.0 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        };

        let potential_causes = vec![
            "Algorithm changes".to_string(),
            "Memory allocation patterns".to_string(),
            "System load variations".to_string(),
            "Compiler optimizations".to_string(),
        ];

        Ok(RegressionDetection {
            test_key: baseline.test_key.clone(),
            operation: baseline.operation.clone(),
            device: baseline.device.clone(),
            regression_type: RegressionType::PerformanceRegression,
            severity,
            performance_degradation: performance_change.execution_time_change,
            baseline_value: baseline.average_execution_time.as_secs_f64(),
            current_value: baseline.average_execution_time.as_secs_f64() * (1.0 + performance_change.execution_time_change / 100.0),
            confidence: performance_change.confidence_level,
            potential_causes,
            bisection_result: None, // Would be filled by automated bisection
        })
    }

    fn analyze_improvement(&self, baseline: &BaselineMetrics, performance_change: &PerformanceChange) -> Result<ImprovementDetection> {
        let likely_causes = vec![
            "Algorithm optimizations".to_string(),
            "Compiler improvements".to_string(),
            "Memory layout optimizations".to_string(),
        ];

        Ok(ImprovementDetection {
            test_key: baseline.test_key.clone(),
            operation: baseline.operation.clone(),
            device: baseline.device.clone(),
            improvement_type: ImprovementType::PerformanceImprovement,
            performance_improvement: -performance_change.execution_time_change, // Negative because it's an improvement
            baseline_value: baseline.average_execution_time.as_secs_f64(),
            current_value: baseline.average_execution_time.as_secs_f64() * (1.0 + performance_change.execution_time_change / 100.0),
            confidence: performance_change.confidence_level,
            likely_causes,
        })
    }

    fn check_memory_regression(&self, memory_baseline: &MemoryBaseline, test_key: &str) -> Result<Option<MemoryRegressionDetection>> {
        // Placeholder implementation - would check current memory usage against baseline
        let current_memory = 1200 * 1024; // 1.2MB (20% increase)
        let memory_increase = ((current_memory - memory_baseline.peak_memory_usage) as f64 / memory_baseline.peak_memory_usage as f64) * 100.0;

        if memory_increase > 15.0 { // 15% threshold
            Ok(Some(MemoryRegressionDetection {
                test_key: test_key.to_string(),
                operation: "test_operation".to_string(), // Would extract from test_key
                device: "test_device".to_string(),
                memory_increase,
                baseline_memory: memory_baseline.peak_memory_usage,
                current_memory,
                regression_type: MemoryRegressionType::PeakMemoryIncrease,
                potential_leak: memory_increase > 50.0,
            }))
        } else {
            Ok(None)
        }
    }

    fn determine_overall_status(&self, test_results: &[TestCaseResult], regressions: &[RegressionDetection]) -> Result<TestStatus> {
        let has_critical = regressions.iter().any(|r| matches!(r.severity, RegressionSeverity::Critical));
        let has_regressions = !regressions.is_empty();
        let has_failures = test_results.iter().any(|r| matches!(r.status, TestCaseStatus::Failed));

        if has_failures {
            Ok(TestStatus::HasFailures)
        } else if has_critical {
            Ok(TestStatus::HasCriticalRegressions)
        } else if has_regressions {
            Ok(TestStatus::HasRegressions)
        } else {
            Ok(TestStatus::AllPass)
        }
    }

    fn generate_test_summary(&self, test_results: &[TestCaseResult], regressions: &[RegressionDetection], improvements: &[ImprovementDetection]) -> Result<RegressionTestSummary> {
        let total_tests = test_results.len();
        let passed_tests = test_results.iter().filter(|r| matches!(r.status, TestCaseStatus::Pass)).count();
        let failed_tests = test_results.iter().filter(|r| matches!(r.status, TestCaseStatus::Failed)).count();
        let regression_count = regressions.len();
        let improvement_count = improvements.len();
        let critical_regressions = regressions.iter().filter(|r| matches!(r.severity, RegressionSeverity::Critical)).count();

        let overall_performance_change = if !test_results.is_empty() {
            test_results.iter().map(|r| r.performance_change.execution_time_change).sum::<f64>() / test_results.len() as f64
        } else {
            0.0
        };

        let mut recommendations = Vec::new();
        if critical_regressions > 0 {
            recommendations.push("Investigate critical performance regressions immediately".to_string());
        }
        if regression_count > improvement_count {
            recommendations.push("Consider reverting recent changes or optimizing affected operations".to_string());
        }
        if improvement_count > 0 {
            recommendations.push("Document and preserve performance improvements".to_string());
        }

        Ok(RegressionTestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            regression_count,
            improvement_count,
            critical_regressions,
            memory_regressions: 0, // Would count from memory_regressions parameter
            overall_performance_change,
            recommendations,
        })
    }

    fn collect_version_info(&self) -> Result<VersionInfo> {
        Ok(VersionInfo {
            mlx_version: "0.1.0".to_string(),
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            bitnet_version: "0.1.0".to_string(),
            commit_hash: std::env::var("GIT_COMMIT").ok(),
            build_timestamp: SystemTime::now(),
        })
    }

    fn collect_system_info(&self) -> Result<SystemInfo> {
        Ok(SystemInfo {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_model: "Unknown".to_string(), // Would query actual CPU info
            gpu_model: Some("Apple Silicon GPU".to_string()),
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB
            gpu_memory: Some(16 * 1024 * 1024 * 1024), // Unified memory
            cpu_cores: 8,
            gpu_cores: Some(32),
        })
    }

    /// Generate regression test report
    pub fn generate_report(&self, test_result: &RegressionTestResult) -> Result<String> {
        let mut report = String::new();
        
        report.push_str("# MLX Performance Regression Test Report\n\n");
        report.push_str(&format!("Test ID: {}\n", test_result.test_id));
        report.push_str(&format!("Baseline: {}\n", test_result.baseline_id));
        report.push_str(&format!("Timestamp: {:?}\n\n", test_result.timestamp));

        // Overall status
        report.push_str("## Overall Status\n");
        report.push_str(&format!("Status: {:?}\n", test_result.overall_status));
        report.push_str(&format!("Total Tests: {}\n", test_result.summary.total_tests));
        report.push_str(&format!("Passed: {}\n", test_result.summary.passed_tests));
        report.push_str(&format!("Regressions: {}\n", test_result.summary.regression_count));
        report.push_str(&format!("Improvements: {}\n", test_result.summary.improvement_count));
        report.push_str(&format!("Critical Regressions: {}\n\n", test_result.summary.critical_regressions));

        // Regressions
        if !test_result.regressions_detected.is_empty() {
            report.push_str("## Regressions Detected\n");
            for regression in &test_result.regressions_detected {
                report.push_str(&format!(
                    "### {} on {} ({:?})\n",
                    regression.operation, regression.device, regression.severity
                ));
                report.push_str(&format!("- Performance degradation: {:.1}%\n", regression.performance_degradation));
                report.push_str(&format!("- Confidence: {:.1}%\n", regression.confidence * 100.0));
                
                if !regression.potential_causes.is_empty() {
                    report.push_str("- Potential causes:\n");
                    for cause in &regression.potential_causes {
                        report.push_str(&format!("  - {}\n", cause));
                    }
                }
                report.push_str("\n");
            }
        }

        // Improvements
        if !test_result.improvements_detected.is_empty() {
            report.push_str("## Improvements Detected\n");
            for improvement in &test_result.improvements_detected {
                report.push_str(&format!(
                    "### {} on {}\n",
                    improvement.operation, improvement.device
                ));
                report.push_str(&format!("- Performance improvement: {:.1}%\n", improvement.performance_improvement));
                report.push_str(&format!("- Confidence: {:.1}%\n\n", improvement.confidence * 100.0));
            }
        }

        // Recommendations
        if !test_result.summary.recommendations.is_empty() {
            report.push_str("## Recommendations\n");
            for recommendation in &test_result.summary.recommendations {
                report.push_str(&format!("- {}\n", recommendation));
            }
        }

        Ok(report)
    }

    /// Export test results
    pub fn export_results(&self, test_result: &RegressionTestResult, format: &str) -> Result<String> {
        match format {
            "json" => serde_json::to_string_pretty(test_result)
                .map_err(|e| anyhow::anyhow!("Failed to serialize test results: {}", e)),
            "report" => self.generate_report(test_result),
            _ => Err(anyhow::anyhow!("Unsupported export format: {}", format)),
        }
    }
}

/// Baseline Manager for storing and retrieving performance baselines
pub struct BaselineManager {
    baseline_directory: PathBuf,
}

impl BaselineManager {
    /// Create a new baseline manager
    pub fn new(baseline_directory: PathBuf) -> Self {
        Self { baseline_directory }
    }

    /// Save a baseline to disk
    pub fn save_baseline(&self, baseline: &PerformanceBaseline) -> Result<()> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&self.baseline_directory)?;
        
        let filename = format!("{}.json", baseline.baseline_id);
        let filepath = self.baseline_directory.join(filename);
        
        let json = serde_json::to_string_pretty(baseline)?;
        std::fs::write(filepath, json)?;
        
        Ok(())
    }

    /// Load a baseline from disk
    pub fn load_baseline(&self, baseline_id: &str) -> Result<PerformanceBaseline> {
        let filename = format!("{}.json", baseline_id);
        let filepath = self.baseline_directory.join(filename);
        
        if !filepath.exists() {
            return Err(anyhow::anyhow!("Baseline not found: {}", baseline_id));
        }
        
        let json = std::fs::read_to_string(filepath)?;
        let baseline: PerformanceBaseline = serde_json::from_str(&json)?;
        
        Ok(baseline)
    }

    /// List all available baselines
    pub fn list_baselines(&self) -> Result<Vec<String>> {
        let mut baselines = Vec::new();
        
        if self.baseline_directory.exists() {
            for entry in std::fs::read_dir(&self.baseline_directory)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        baselines.push(stem.to_string());
                    }
                }
            }
        }
        
        Ok(baselines)
    }

    /// Delete a baseline
    pub fn delete_baseline(&self, baseline_id: &str) -> Result<()> {
        let filename = format!("{}.json", baseline_id);
        let filepath = self.baseline_directory.join(filename);
        
        if filepath.exists() {
            std::fs::remove_file(filepath)?;
        }
        
        Ok(())
    }

    /// Get baseline metadata without loading full baseline
    pub fn get_baseline_metadata(&self, baseline_id: &str) -> Result<BaselineMetadata> {
        let baseline = self.load_baseline(baseline_id)?;
        Ok(baseline.metadata)
    }
}

impl Default for MlxRegressionTester {
    fn default() -> Self {
        Self::new(RegressionTestConfig::default())
    }
}

/// Utility functions for regression testing
pub mod utils {
    use super::*;

    /// Calculate statistical significance between two sets of measurements
    pub fn calculate_statistical_significance(
        baseline_times: &[Duration],
        current_times: &[Duration],
    ) -> f64 {
        // Simplified t-test implementation
        if baseline_times.is_empty() || current_times.is_empty() {
            return 0.0;
        }

        let baseline_mean = baseline_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / baseline_times.len() as f64;
        let current_mean = current_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / current_times.len() as f64;

        let baseline_var = baseline_times.iter()
            .map(|d| (d.as_secs_f64() - baseline_mean).powi(2))
            .sum::<f64>() / (baseline_times.len() - 1) as f64;

        let current_var = current_times.iter()
            .map(|d| (d.as_secs_f64() - current_mean).powi(2))
            .sum::<f64>() / (current_times.len() - 1) as f64;

        let pooled_std = ((baseline_var / baseline_times.len() as f64) + (current_var / current_times.len() as f64)).sqrt();
        
        if pooled_std == 0.0 {
            return 0.0;
        }

        let t_stat = (current_mean - baseline_mean) / pooled_std;
        
        // Return p-value approximation (simplified)
        1.0 - (t_stat.abs() / 3.0).min(1.0)
    }

    /// Detect performance trends over multiple test runs
    pub fn detect_performance_trend(test_results: &[RegressionTestResult]) -> PerformanceTrend {
        if test_results.len() < 3 {
            return PerformanceTrend::Insufficient;
        }

        let changes: Vec<f64> = test_results.iter()
            .map(|r| r.summary.overall_performance_change)
            .collect();

        let mut increasing = 0;
        let mut decreasing = 0;

        for window in changes.windows(2) {
            if window[1] > window[0] {
                increasing += 1;
            } else if window[1] < window[0] {
                decreasing += 1;
            }
        }

        if increasing > decreasing * 2 {
            PerformanceTrend::Improving
        } else if decreasing > increasing * 2 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Performance trend analysis
    #[derive(Debug, Clone)]
    pub enum PerformanceTrend {
        Improving,
        Degrading,
        Stable,
        Volatile,
        Insufficient,
    }

    /// Generate performance regression alert
    pub fn generate_regression_alert(regression: &RegressionDetection) -> String {
        let severity_emoji = match regression.severity {
            RegressionSeverity::Critical => "🚨",
            RegressionSeverity::Major => "⚠️",
            RegressionSeverity::Moderate => "⚡",
            RegressionSeverity::Minor => "📊",
        };

        format!(
            "{} Performance Regression Alert\n\
            Operation: {} on {}\n\
            Severity: {:?}\n\
            Performance degradation: {:.1}%\n\
            Confidence: {:.1}%\n\
            Test key: {}",
            severity_emoji,
            regression.operation,
            regression.device,
            regression.severity,
            regression.performance_degradation,
            regression.confidence * 100.0,
            regression.test_key
        )
    }
}