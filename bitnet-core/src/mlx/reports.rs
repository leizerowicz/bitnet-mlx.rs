//! MLX Performance Report Generation
//! 
//! This module provides comprehensive report generation capabilities for MLX performance
//! analysis, including detailed comparisons, visualizations, and recommendations.

use crate::mlx::{
    performance::{PerformanceMetrics, ComparisonResult, MemoryUsage},
    memory_tracker::{MemoryEvent, MemorySnapshot, MemoryOptimization, OptimizationType},
    metrics::{MlxMetrics, MemoryMetrics, SystemMetrics, OperationContext},
};
use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub metadata: ReportMetadata,
    pub executive_summary: ExecutiveSummary,
    pub performance_analysis: PerformanceAnalysis,
    pub memory_analysis: MemoryAnalysis,
    pub device_comparisons: Vec<DeviceComparison>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub regression_analysis: Option<RegressionAnalysis>,
    pub appendix: ReportAppendix,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generated_at: SystemTime,
    pub report_version: String,
    pub data_collection_period: Duration,
    pub total_operations_analyzed: usize,
    pub devices_tested: Vec<String>,
    pub mlx_version: String,
    pub system_info: SystemInfo,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub architecture: String,
    pub cpu_model: String,
    pub gpu_model: Option<String>,
    pub total_memory: usize,
    pub gpu_memory: Option<usize>,
}

/// Executive summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub key_findings: Vec<String>,
    pub performance_highlights: Vec<String>,
    pub critical_issues: Vec<String>,
    pub top_recommendations: Vec<String>,
    pub overall_score: f64, // 0-100 performance score
}

/// Performance analysis section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub operation_performance: HashMap<String, OperationPerformanceStats>,
    pub throughput_analysis: ThroughputAnalysis,
    pub latency_analysis: LatencyAnalysis,
    pub scalability_analysis: ScalabilityAnalysis,
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Operation-specific performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPerformanceStats {
    pub operation_name: String,
    pub total_executions: usize,
    pub average_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub p95_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub success_rate: f64,
    pub error_rate: f64,
}

/// Throughput analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    pub peak_throughput: f64,
    pub average_throughput: f64,
    pub throughput_variance: f64,
    pub throughput_trend: ThroughputTrend,
    pub bottleneck_analysis: Vec<String>,
}

/// Throughput trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThroughputTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Latency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysis {
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub latency_distribution: Vec<LatencyBucket>,
    pub outlier_analysis: OutlierAnalysis,
}

/// Latency bucket for distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBucket {
    pub range_start: Duration,
    pub range_end: Duration,
    pub count: usize,
    pub percentage: f64,
}

/// Outlier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    pub outlier_count: usize,
    pub outlier_threshold: Duration,
    pub potential_causes: Vec<String>,
}

/// Scalability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysis {
    pub batch_size_scaling: Vec<ScalingPoint>,
    pub tensor_size_scaling: Vec<ScalingPoint>,
    pub parallel_scaling: Vec<ScalingPoint>,
    pub scaling_efficiency: f64,
    pub optimal_configurations: HashMap<String, String>,
}

/// Scaling point for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPoint {
    pub parameter_value: f64,
    pub throughput: f64,
    pub latency: Duration,
    pub memory_usage: f64,
    pub efficiency: f64,
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub compute_efficiency: f64,
    pub memory_efficiency: f64,
    pub energy_efficiency: Option<f64>,
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub memory_utilization: f64,
    pub bandwidth_utilization: f64,
}

/// Memory analysis section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub allocation_patterns: AllocationPatterns,
    pub memory_pressure_analysis: MemoryPressureAnalysis,
    pub fragmentation_analysis: FragmentationAnalysis,
    pub leak_detection: LeakDetection,
    pub optimization_opportunities: Vec<MemoryOptimization>,
}

/// Allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPatterns {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub peak_memory_usage: usize,
    pub average_allocation_size: usize,
    pub allocation_frequency: f64,
    pub common_allocation_sizes: Vec<(usize, usize)>, // (size, count)
}

/// Memory pressure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureAnalysis {
    pub pressure_events: Vec<PressureEvent>,
    pub average_pressure_level: String,
    pub peak_pressure_duration: Duration,
    pub pressure_triggers: Vec<String>,
}

/// Pressure event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureEvent {
    pub timestamp: SystemTime,
    pub pressure_level: String,
    pub duration: Duration,
    pub trigger_operation: String,
    pub memory_usage: usize,
}

/// Fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    pub fragmentation_ratio: f64,
    pub fragmentation_trend: FragmentationTrend,
    pub largest_free_block: usize,
    pub fragmentation_hotspots: Vec<String>,
}

/// Fragmentation trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FragmentationTrend {
    Improving,
    Worsening,
    Stable,
}

/// Leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakDetection {
    pub potential_leaks: Vec<PotentialLeak>,
    pub leak_score: f64, // 0-100, higher means more likely leaks
    pub monitoring_recommendations: Vec<String>,
}

/// Potential memory leak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLeak {
    pub operation: String,
    pub allocation_growth_rate: f64,
    pub confidence: f64,
    pub first_detected: SystemTime,
    pub estimated_leak_rate: usize, // bytes per operation
}

/// Device comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceComparison {
    pub device_a: String,
    pub device_b: String,
    pub operation: String,
    pub performance_comparison: PerformanceComparison,
    pub memory_comparison: MemoryComparison,
    pub recommendation: String,
    pub confidence: f64,
}

/// Performance comparison between devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub speedup: f64,
    pub throughput_improvement: f64,
    pub latency_improvement: f64,
    pub consistency_comparison: f64,
}

/// Memory comparison between devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    pub memory_efficiency_improvement: f64,
    pub peak_memory_reduction: f64,
    pub allocation_efficiency: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub title: String,
    pub description: String,
    pub expected_improvement: ExpectedImprovement,
    pub implementation_effort: ImplementationEffort,
    pub priority: Priority,
    pub code_examples: Vec<CodeExample>,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Performance,
    Memory,
    Energy,
    Scalability,
    Reliability,
}

/// Expected improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    pub performance_gain: Option<f64>,
    pub memory_reduction: Option<f64>,
    pub energy_savings: Option<f64>,
    pub confidence: f64,
}

/// Implementation effort
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Priority level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub language: String,
    pub title: String,
    pub code: String,
    pub explanation: String,
}

/// Regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub baseline_period: String,
    pub comparison_period: String,
    pub regressions_detected: Vec<RegressionEvent>,
    pub improvements_detected: Vec<ImprovementEvent>,
    pub overall_trend: PerformanceTrend,
}

/// Regression event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionEvent {
    pub operation: String,
    pub device: String,
    pub regression_percentage: f64,
    pub first_detected: SystemTime,
    pub potential_causes: Vec<String>,
    pub severity: RegressionSeverity,
}

/// Improvement event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementEvent {
    pub operation: String,
    pub device: String,
    pub improvement_percentage: f64,
    pub first_detected: SystemTime,
    pub likely_causes: Vec<String>,
}

/// Regression severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Report appendix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportAppendix {
    pub raw_metrics: Vec<MlxMetrics>,
    pub configuration_details: HashMap<String, String>,
    pub methodology: String,
    pub limitations: Vec<String>,
    pub glossary: HashMap<String, String>,
}

/// Performance Report Generator
pub struct PerformanceReportGenerator {
    template_engine: Option<String>, // For future template support
}

impl PerformanceReportGenerator {
    /// Create a new report generator
    pub fn new() -> Self {
        Self {
            template_engine: None,
        }
    }

    /// Generate a comprehensive performance report
    pub fn generate_comprehensive_report(
        &self,
        metrics: &[MlxMetrics],
        comparisons: &[ComparisonResult],
        memory_events: &[MemoryEvent],
        optimizations: &[MemoryOptimization],
    ) -> Result<PerformanceReport> {
        let metadata = self.generate_metadata(metrics)?;
        let executive_summary = self.generate_executive_summary(metrics, comparisons)?;
        let performance_analysis = self.generate_performance_analysis(metrics)?;
        let memory_analysis = self.generate_memory_analysis(memory_events, optimizations)?;
        let device_comparisons = self.generate_device_comparisons(comparisons)?;
        let optimization_recommendations = self.generate_optimization_recommendations(metrics, optimizations)?;
        let regression_analysis = self.generate_regression_analysis(metrics)?;
        let appendix = self.generate_appendix(metrics)?;

        Ok(PerformanceReport {
            metadata,
            executive_summary,
            performance_analysis,
            memory_analysis,
            device_comparisons,
            optimization_recommendations,
            regression_analysis,
            appendix,
        })
    }

    /// Generate report metadata
    fn generate_metadata(&self, metrics: &[MlxMetrics]) -> Result<ReportMetadata> {
        let devices_tested: Vec<String> = metrics.iter()
            .map(|m| m.performance.device_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let data_collection_period = if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
            last.performance.timestamp.duration_since(first.performance.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        };

        Ok(ReportMetadata {
            generated_at: SystemTime::now(),
            report_version: "1.0.0".to_string(),
            data_collection_period,
            total_operations_analyzed: metrics.len(),
            devices_tested,
            mlx_version: "0.1.0".to_string(), // Placeholder
            system_info: SystemInfo {
                os: std::env::consts::OS.to_string(),
                architecture: std::env::consts::ARCH.to_string(),
                cpu_model: "Unknown".to_string(), // Would query actual CPU info
                gpu_model: Some("Apple Silicon GPU".to_string()),
                total_memory: 16 * 1024 * 1024 * 1024, // 16GB placeholder
                gpu_memory: Some(16 * 1024 * 1024 * 1024), // Unified memory
            },
        })
    }

    /// Generate executive summary
    fn generate_executive_summary(
        &self,
        metrics: &[MlxMetrics],
        comparisons: &[ComparisonResult],
    ) -> Result<ExecutiveSummary> {
        let mut key_findings = Vec::new();
        let mut performance_highlights = Vec::new();
        let mut critical_issues = Vec::new();
        let mut top_recommendations = Vec::new();

        // Analyze performance trends
        let avg_throughput: f64 = metrics.iter()
            .map(|m| m.performance.throughput)
            .sum::<f64>() / metrics.len() as f64;

        key_findings.push(format!("Average throughput across all operations: {:.2} ops/sec", avg_throughput));

        // Analyze device comparisons
        let gpu_speedups: Vec<f64> = comparisons.iter()
            .filter(|c| c.comparison_metrics.device_type == "gpu")
            .map(|c| c.speedup)
            .collect();

        if !gpu_speedups.is_empty() {
            let avg_speedup = gpu_speedups.iter().sum::<f64>() / gpu_speedups.len() as f64;
            if avg_speedup > 1.5 {
                performance_highlights.push(format!("GPU shows {:.1}x average speedup over CPU", avg_speedup));
                top_recommendations.push("Prioritize GPU execution for compute-intensive operations".to_string());
            } else if avg_speedup < 0.8 {
                critical_issues.push("GPU performance is below CPU performance".to_string());
                top_recommendations.push("Investigate GPU utilization and optimization opportunities".to_string());
            }
        }

        // Analyze memory efficiency
        let memory_efficiencies: Vec<f64> = metrics.iter()
            .map(|m| m.memory.efficiency_score)
            .collect();

        let avg_memory_efficiency = memory_efficiencies.iter().sum::<f64>() / memory_efficiencies.len() as f64;
        if avg_memory_efficiency < 0.7 {
            critical_issues.push("Low memory efficiency detected".to_string());
            top_recommendations.push("Implement memory pooling and tensor reuse strategies".to_string());
        }

        // Calculate overall score
        let performance_score = (avg_throughput.log10() * 20.0).min(50.0);
        let memory_score = avg_memory_efficiency * 30.0;
        let speedup_score = gpu_speedups.iter().sum::<f64>() / gpu_speedups.len() as f64 * 20.0;
        let overall_score = (performance_score + memory_score + speedup_score).min(100.0);

        Ok(ExecutiveSummary {
            key_findings,
            performance_highlights,
            critical_issues,
            top_recommendations,
            overall_score,
        })
    }

    /// Generate performance analysis
    fn generate_performance_analysis(&self, metrics: &[MlxMetrics]) -> Result<PerformanceAnalysis> {
        let mut operation_performance = HashMap::new();

        // Group metrics by operation
        let mut operation_groups: HashMap<String, Vec<&MlxMetrics>> = HashMap::new();
        for metric in metrics {
            operation_groups.entry(metric.operation_context.operation_name.clone())
                .or_insert_with(Vec::new)
                .push(metric);
        }

        // Calculate statistics for each operation
        for (operation_name, operation_metrics) in operation_groups {
            let execution_times: Vec<Duration> = operation_metrics.iter()
                .map(|m| m.performance.execution_time)
                .collect();

            let throughputs: Vec<f64> = operation_metrics.iter()
                .map(|m| m.performance.throughput)
                .collect();

            let mut sorted_times = execution_times.clone();
            sorted_times.sort();

            let stats = OperationPerformanceStats {
                operation_name: operation_name.clone(),
                total_executions: operation_metrics.len(),
                average_execution_time: Duration::from_nanos(
                    (execution_times.iter().map(|d| d.as_nanos()).sum::<u128>() / execution_times.len() as u128).try_into().unwrap_or(0)
                ),
                min_execution_time: *sorted_times.first().unwrap_or(&Duration::ZERO),
                max_execution_time: *sorted_times.last().unwrap_or(&Duration::ZERO),
                p95_execution_time: sorted_times.get(sorted_times.len() * 95 / 100).copied().unwrap_or(Duration::ZERO),
                p99_execution_time: sorted_times.get(sorted_times.len() * 99 / 100).copied().unwrap_or(Duration::ZERO),
                average_throughput: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
                peak_throughput: throughputs.iter().fold(0.0, |a, &b| a.max(b)),
                success_rate: 100.0, // Placeholder - would track actual success/failure
                error_rate: 0.0,
            };

            operation_performance.insert(operation_name, stats);
        }

        // Generate throughput analysis
        let all_throughputs: Vec<f64> = metrics.iter().map(|m| m.performance.throughput).collect();
        let avg_throughput = all_throughputs.iter().sum::<f64>() / all_throughputs.len() as f64;
        let peak_throughput = all_throughputs.iter().fold(0.0f64, |a, &b| a.max(b));
        let throughput_variance = self.calculate_variance(&all_throughputs);

        let throughput_analysis = ThroughputAnalysis {
            peak_throughput,
            average_throughput: avg_throughput,
            throughput_variance,
            throughput_trend: ThroughputTrend::Stable, // Would analyze trend over time
            bottleneck_analysis: vec!["Memory bandwidth may be limiting factor".to_string()],
        };

        // Generate latency analysis
        let all_latencies: Vec<Duration> = metrics.iter().map(|m| m.performance.execution_time).collect();
        let mut sorted_latencies = all_latencies.clone();
        sorted_latencies.sort();

        let latency_analysis = LatencyAnalysis {
            average_latency: Duration::from_nanos(
                (all_latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / all_latencies.len() as u128).try_into().unwrap_or(0)
            ),
            p50_latency: sorted_latencies.get(sorted_latencies.len() / 2).copied().unwrap_or(Duration::ZERO),
            p95_latency: sorted_latencies.get(sorted_latencies.len() * 95 / 100).copied().unwrap_or(Duration::ZERO),
            p99_latency: sorted_latencies.get(sorted_latencies.len() * 99 / 100).copied().unwrap_or(Duration::ZERO),
            latency_distribution: self.calculate_latency_distribution(&sorted_latencies),
            outlier_analysis: self.analyze_outliers(&sorted_latencies),
        };

        // Generate scalability analysis (placeholder)
        let scalability_analysis = ScalabilityAnalysis {
            batch_size_scaling: Vec::new(),
            tensor_size_scaling: Vec::new(),
            parallel_scaling: Vec::new(),
            scaling_efficiency: 0.8, // Placeholder
            optimal_configurations: HashMap::new(),
        };

        // Generate efficiency metrics
        let avg_cpu_usage = metrics.iter().map(|m| m.system.cpu_usage).sum::<f64>() / metrics.len() as f64;
        let avg_gpu_usage = metrics.iter().map(|m| m.system.gpu_usage).sum::<f64>() / metrics.len() as f64;
        let avg_memory_usage = metrics.iter().map(|m| m.system.system_memory_usage).sum::<f64>() / metrics.len() as f64;

        let efficiency_metrics = EfficiencyMetrics {
            compute_efficiency: (avg_cpu_usage + avg_gpu_usage) / 2.0,
            memory_efficiency: metrics.iter().map(|m| m.memory.efficiency_score).sum::<f64>() / metrics.len() as f64,
            energy_efficiency: None, // Would require power measurements
            resource_utilization: ResourceUtilization {
                cpu_utilization: avg_cpu_usage,
                gpu_utilization: avg_gpu_usage,
                memory_utilization: avg_memory_usage,
                bandwidth_utilization: 50.0, // Placeholder
            },
        };

        Ok(PerformanceAnalysis {
            operation_performance,
            throughput_analysis,
            latency_analysis,
            scalability_analysis,
            efficiency_metrics,
        })
    }

    /// Generate memory analysis
    fn generate_memory_analysis(
        &self,
        memory_events: &[MemoryEvent],
        optimizations: &[MemoryOptimization],
    ) -> Result<MemoryAnalysis> {
        // Analyze allocation patterns
        let allocations: Vec<&MemoryEvent> = memory_events.iter()
            .filter(|e| matches!(e.event_type, crate::mlx::memory_tracker::MemoryEventType::Allocation))
            .collect();

        let deallocations: Vec<&MemoryEvent> = memory_events.iter()
            .filter(|e| matches!(e.event_type, crate::mlx::memory_tracker::MemoryEventType::Deallocation))
            .collect();

        let allocation_sizes: Vec<usize> = allocations.iter().map(|e| e.size_bytes).collect();
        let avg_allocation_size = if !allocation_sizes.is_empty() {
            allocation_sizes.iter().sum::<usize>() / allocation_sizes.len()
        } else {
            0
        };

        let allocation_patterns = AllocationPatterns {
            total_allocations: allocations.len(),
            total_deallocations: deallocations.len(),
            peak_memory_usage: allocation_sizes.iter().max().copied().unwrap_or(0),
            average_allocation_size: avg_allocation_size,
            allocation_frequency: allocations.len() as f64 / 60.0, // per minute
            common_allocation_sizes: self.analyze_common_sizes(&allocation_sizes),
        };

        // Memory pressure analysis (placeholder)
        let memory_pressure_analysis = MemoryPressureAnalysis {
            pressure_events: Vec::new(),
            average_pressure_level: "Low".to_string(),
            peak_pressure_duration: Duration::from_secs(0),
            pressure_triggers: Vec::new(),
        };

        // Fragmentation analysis (placeholder)
        let fragmentation_analysis = FragmentationAnalysis {
            fragmentation_ratio: 0.1,
            fragmentation_trend: FragmentationTrend::Stable,
            largest_free_block: 1024 * 1024 * 1024, // 1GB
            fragmentation_hotspots: Vec::new(),
        };

        // Leak detection (placeholder)
        let leak_detection = LeakDetection {
            potential_leaks: Vec::new(),
            leak_score: 10.0, // Low risk
            monitoring_recommendations: vec![
                "Monitor allocation/deallocation ratios".to_string(),
                "Track long-lived allocations".to_string(),
            ],
        };

        Ok(MemoryAnalysis {
            allocation_patterns,
            memory_pressure_analysis,
            fragmentation_analysis,
            leak_detection,
            optimization_opportunities: optimizations.to_vec(),
        })
    }

    /// Generate device comparisons
    fn generate_device_comparisons(&self, comparisons: &[ComparisonResult]) -> Result<Vec<DeviceComparison>> {
        let mut device_comparisons = Vec::new();

        for comparison in comparisons {
            let device_comparison = DeviceComparison {
                device_a: comparison.baseline_metrics.device_type.clone(),
                device_b: comparison.comparison_metrics.device_type.clone(),
                operation: comparison.baseline_metrics.operation_name.clone(),
                performance_comparison: PerformanceComparison {
                    speedup: comparison.speedup,
                    throughput_improvement: comparison.throughput_improvement,
                    latency_improvement: 1.0 / comparison.speedup, // Inverse of speedup
                    consistency_comparison: 0.9, // Placeholder
                },
                memory_comparison: MemoryComparison {
                    memory_efficiency_improvement: comparison.memory_improvement,
                    peak_memory_reduction: 0.0, // Would calculate from actual data
                    allocation_efficiency: 0.8, // Placeholder
                },
                recommendation: comparison.recommendation.clone(),
                confidence: 0.85, // Placeholder confidence score
            };

            device_comparisons.push(device_comparison);
        }

        Ok(device_comparisons)
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(
        &self,
        metrics: &[MlxMetrics],
        memory_optimizations: &[MemoryOptimization],
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        let avg_throughput = metrics.iter().map(|m| m.performance.throughput).sum::<f64>() / metrics.len() as f64;
        if avg_throughput < 100.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Performance,
                title: "Optimize Batch Processing".to_string(),
                description: "Increase batch sizes to improve throughput and GPU utilization".to_string(),
                expected_improvement: ExpectedImprovement {
                    performance_gain: Some(2.0),
                    memory_reduction: None,
                    energy_savings: None,
                    confidence: 0.8,
                },
                implementation_effort: ImplementationEffort::Low,
                priority: Priority::High,
                code_examples: vec![
                    CodeExample {
                        language: "rust".to_string(),
                        title: "Batch Processing Example".to_string(),
                        code: r#"
// Process tensors in larger batches
let batch_size = 64; // Increase from 32
let batched_tensors = tensors.chunks(batch_size);
for batch in batched_tensors {
    let result = mlx_operation(batch)?;
    // Process results
}
"#.to_string(),
                        explanation: "Larger batch sizes improve GPU utilization and throughput".to_string(),
                    }
                ],
            });
        }

        // Memory recommendations from memory optimizations
        for opt in memory_optimizations {
            let category = match opt.suggestion_type {
                OptimizationType::TensorReuse => OptimizationCategory::Memory,
                OptimizationType::InPlaceOperations => OptimizationCategory::Performance,
                OptimizationType::BatchSizeReduction => OptimizationCategory::Memory,
                OptimizationType::DataTypeOptimization => OptimizationCategory::Performance,
                OptimizationType::MemoryPooling => OptimizationCategory::Memory,
                OptimizationType::GarbageCollection => OptimizationCategory::Memory,
                OptimizationType::DeviceTransferOptimization => OptimizationCategory::Performance,
            };

            let priority = match opt.priority {
                crate::mlx::memory_tracker::OptimizationPriority::Low => Priority::Low,
                crate::mlx::memory_tracker::OptimizationPriority::Medium => Priority::Medium,
                crate::mlx::memory_tracker::OptimizationPriority::High => Priority::High,
                crate::mlx::memory_tracker::OptimizationPriority::Critical => Priority::Critical,
            };

            let effort = match opt.implementation_effort {
                crate::mlx::memory_tracker::ImplementationEffort::Minimal => ImplementationEffort::Minimal,
                crate::mlx::memory_tracker::ImplementationEffort::Low => ImplementationEffort::Low,
                crate::mlx::memory_tracker::ImplementationEffort::Medium => ImplementationEffort::Medium,
                crate::mlx::memory_tracker::ImplementationEffort::High => ImplementationEffort::High,
            };

            recommendations.push(OptimizationRecommendation {
                category,
                title: format!("Memory Optimization: {}", opt.description),
                description: opt.description.clone(),
                expected_improvement: ExpectedImprovement {
                    performance_gain: None,
                    memory_reduction: Some(opt.potential_savings as f64 / (1024.0 * 1024.0)), // Convert to MB
                    energy_savings: None,
                    confidence: 0.7,
                },
                implementation_effort: effort,
                priority,
                code_examples: Vec::new(),
            });
        }

        Ok(recommendations)
    }

    /// Generate regression analysis
    fn generate_regression_analysis(&self, metrics: &[MlxMetrics]) -> Result<Option<RegressionAnalysis>> {
        if metrics.len() < 10 {
            return Ok(None); // Not enough data for regression analysis
        }

        // Split metrics into baseline and comparison periods
        let split_point = metrics.len() / 2;
        let baseline_metrics = &metrics[..split_point];
        let comparison_metrics = &metrics[split_point..];

        let mut regressions_detected = Vec::new();
        let mut improvements_detected = Vec::new();

        // Group by operation and device
        let mut baseline_stats: HashMap<String, f64> = HashMap::new();
        let mut comparison_stats: HashMap<String, f64> = HashMap::new();

        for metric in baseline_metrics {
            let key = format!("{}_{}", metric.operation_context.operation_name, metric.performance.device_type);
            baseline_stats.insert(key, metric.performance.throughput);
        }

        for metric in comparison_metrics {
            let key = format!("{}_{}", metric.operation_context.operation_name, metric.performance.device_type);
            comparison_stats.insert(key, metric.performance.throughput);
        }

        // Compare baseline vs comparison
        for (key, baseline_throughput) in baseline_stats {
            if let Some(comparison_throughput) = comparison_stats.get(&key) {
                let change_ratio = comparison_throughput / baseline_throughput;
                let change_percentage = (change_ratio - 1.0) * 100.0;

                let parts: Vec<&str> = key.split('_').collect();
                if parts.len() >= 2 {
                    let operation = parts[0].to_string();
                    let device = parts[1].to_string();

                    if change_percentage < -10.0 {
                        // Regression detected
                        regressions_detected.push(RegressionEvent {
                            operation,
                            device,
                            regression_percentage: -change_percentage,
                            first_detected: SystemTime::now(),
                            potential_causes: vec![
                                "System load changes".to_string(),
                                "Memory pressure".to_string(),
                                "Thermal throttling".to_string(),
                            ],
                            severity: if change_percentage < -25.0 {
                                RegressionSeverity::Severe
                            } else if change_percentage < -20.0 {
                                RegressionSeverity::Moderate
                            } else {
                                RegressionSeverity::Minor
                            },
                        });
                    } else if change_percentage > 10.0 {
                        // Improvement detected
                        improvements_detected.push(ImprovementEvent {
                            operation,
                            device,
                            improvement_percentage: change_percentage,
                            first_detected: SystemTime::now(),
                            likely_causes: vec![
                                "Optimization improvements".to_string(),
                                "Better resource utilization".to_string(),
                            ],
                        });
                    }
                }
            }
        }

        let overall_trend = if regressions_detected.len() > improvements_detected.len() {
            PerformanceTrend::Degrading
        } else if improvements_detected.len() > regressions_detected.len() {
            PerformanceTrend::Improving
        } else {
            PerformanceTrend::Stable
        };

        Ok(Some(RegressionAnalysis {
            baseline_period: "First half of data".to_string(),
            comparison_period: "Second half of data".to_string(),
            regressions_detected,
            improvements_detected,
            overall_trend,
        }))
    }

    /// Generate report appendix
    fn generate_appendix(&self, metrics: &[MlxMetrics]) -> Result<ReportAppendix> {
        let mut configuration_details = HashMap::new();
        configuration_details.insert("mlx_version".to_string(), "0.1.0".to_string());
        configuration_details.insert("rust_version".to_string(),
            std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
        configuration_details.insert("target_arch".to_string(), std::env::consts::ARCH.to_string());

        let mut glossary = HashMap::new();
        glossary.insert("Throughput".to_string(), "Number of operations completed per second".to_string());
        glossary.insert("Latency".to_string(), "Time taken to complete a single operation".to_string());
        glossary.insert("Memory Efficiency".to_string(), "Ratio of useful memory to total allocated memory".to_string());
        glossary.insert("Speedup".to_string(), "Performance improvement ratio between two configurations".to_string());

        Ok(ReportAppendix {
            raw_metrics: metrics.to_vec(),
            configuration_details,
            methodology: "Performance metrics collected using MLX benchmarking framework with statistical analysis".to_string(),
            limitations: vec![
                "Results may vary based on system load".to_string(),
                "Memory measurements are approximate".to_string(),
                "GPU metrics depend on hardware availability".to_string(),
            ],
            glossary,
        })
    }

    /// Generate HTML report
    pub fn generate_html_report(&self, report: &PerformanceReport) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>MLX Performance Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("h1 { color: #333; border-bottom: 2px solid #007acc; }\n");
        html.push_str("h2 { color: #555; border-bottom: 1px solid #ccc; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #f2f2f2; }\n");
        html.push_str(".metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }\n");
        html.push_str(".recommendation { background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        // Title and metadata
        html.push_str("<h1>MLX Performance Analysis Report</h1>\n");
        html.push_str(&format!("<p>Generated: {:?}</p>\n", report.metadata.generated_at));
        html.push_str(&format!("<p>Operations Analyzed: {}</p>\n", report.metadata.total_operations_analyzed));

        // Executive Summary
        html.push_str("<h2>Executive Summary</h2>\n");
        html.push_str(&format!("<div class='metric'><strong>Overall Performance Score: {:.1}/100</strong></div>\n", report.executive_summary.overall_score));
        
        html.push_str("<h3>Key Findings</h3>\n<ul>\n");
        for finding in &report.executive_summary.key_findings {
            html.push_str(&format!("<li>{}</li>\n", finding));
        }
        html.push_str("</ul>\n");

        // Performance Analysis
        html.push_str("<h2>Performance Analysis</h2>\n");
        html.push_str("<table>\n<tr><th>Operation</th><th>Avg Execution Time</th><th>Throughput</th><th>Success Rate</th></tr>\n");
        for (_, stats) in &report.performance_analysis.operation_performance {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.3}ms</td><td>{:.2} ops/sec</td><td>{:.1}%</td></tr>\n",
                stats.operation_name,
                stats.average_execution_time.as_millis(),
                stats.average_throughput,
                stats.success_rate
            ));
        }
        html.push_str("</table>\n");

        // Device Comparisons
        html.push_str("<h2>Device Comparisons</h2>\n");
        for comparison in &report.device_comparisons {
            html.push_str(&format!(
                "<div class='metric'><strong>{} vs {} ({})</strong><br>Speedup: {:.2}x<br>Recommendation: {}</div>\n",
                comparison.device_a,
                comparison.device_b,
                comparison.operation,
                comparison.performance_comparison.speedup,
                comparison.recommendation
            ));
        }

        // Optimization Recommendations
        html.push_str("<h2>Optimization Recommendations</h2>\n");
        for rec in &report.optimization_recommendations {
            html.push_str(&format!(
                "<div class='recommendation'><h3>{}</h3><p>{}</p><p><strong>Priority:</strong> {:?}</p></div>\n",
                rec.title,
                rec.description,
                rec.priority
            ));
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Generate JSON report
    pub fn generate_json_report(
        &self,
        metrics: &[MlxMetrics],
        comparisons: &[ComparisonResult],
    ) -> Result<String> {
        let report = self.generate_comprehensive_report(
            metrics,
            comparisons,
            &[], // Empty memory events for now
            &[], // Empty optimizations for now
        )?;
        
        serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("Failed to serialize report to JSON: {}", e))
    }

    /// Helper methods
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance
    }

    fn calculate_latency_distribution(&self, latencies: &[Duration]) -> Vec<LatencyBucket> {
        if latencies.is_empty() {
            return Vec::new();
        }

        let min_latency = latencies[0];
        let max_latency = latencies[latencies.len() - 1];
        let range = max_latency.saturating_sub(min_latency);
        let bucket_size = range / 10; // 10 buckets

        let mut buckets = Vec::new();
        for i in 0..10 {
            let range_start = min_latency + bucket_size * i as u32;
            let range_end = min_latency + bucket_size * (i + 1) as u32;
            
            let count = latencies.iter()
                .filter(|&&lat| lat >= range_start && lat < range_end)
                .count();
            
            let percentage = count as f64 / latencies.len() as f64 * 100.0;
            
            buckets.push(LatencyBucket {
                range_start,
                range_end,
                count,
                percentage,
            });
        }

        buckets
    }

    fn analyze_outliers(&self, latencies: &[Duration]) -> OutlierAnalysis {
        if latencies.len() < 4 {
            return OutlierAnalysis {
                outlier_count: 0,
                outlier_threshold: Duration::ZERO,
                potential_causes: Vec::new(),
            };
        }

        // Use IQR method for outlier detection
        let q1_idx = latencies.len() / 4;
        let q3_idx = latencies.len() * 3 / 4;
        let q1 = latencies[q1_idx];
        let q3 = latencies[q3_idx];
        let iqr = q3.saturating_sub(q1);
        let outlier_threshold = q3 + iqr + iqr / 2; // 1.5 * IQR above Q3

        let outlier_count = latencies.iter()
            .filter(|&&lat| lat > outlier_threshold)
            .count();

        OutlierAnalysis {
            outlier_count,
            outlier_threshold,
            potential_causes: vec![
                "System background processes".to_string(),
                "Memory pressure".to_string(),
                "Thermal throttling".to_string(),
                "Network or I/O interference".to_string(),
            ],
        }
    }

    fn analyze_common_sizes(&self, sizes: &[usize]) -> Vec<(usize, usize)> {
        let mut size_counts: HashMap<usize, usize> = HashMap::new();
        for &size in sizes {
            *size_counts.entry(size).or_insert(0) += 1;
        }

        let mut sorted_sizes: Vec<(usize, usize)> = size_counts.into_iter().collect();
        sorted_sizes.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending
        sorted_sizes.truncate(10); // Top 10 most common sizes
        sorted_sizes
    }
}

impl Default for PerformanceReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}