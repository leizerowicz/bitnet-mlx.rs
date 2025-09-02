//! Performance monitoring and analysis for test execution
//!
//! This module provides utilities for tracking test performance over time,
//! identifying performance regressions, and generating detailed reports.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use super::TestExecutionResult;

/// Performance trend analysis for a specific test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPerformanceTrend {
    /// Test name
    pub test_name: String,
    /// Historical execution times
    pub execution_history: Vec<(SystemTime, Duration)>,
    /// Average execution time over all runs
    pub average_duration: Duration,
    /// Median execution time
    pub median_duration: Duration,
    /// Standard deviation of execution times
    pub std_deviation: Duration,
    /// Trend direction (positive = getting slower, negative = getting faster)
    pub trend_slope: f64,
    /// Whether this test shows concerning performance trends
    pub is_concerning: bool,
    /// Performance category based on recent runs
    pub performance_category: PerformanceCategory,
}

/// Performance category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerformanceCategory {
    /// Consistently fast (< 1s)
    Fast,
    /// Moderate performance (1-10s)
    Moderate,
    /// Slow but acceptable (10-60s)
    Slow,
    /// Very slow, needs attention (60-300s)
    VerySlow,
    /// Extremely slow, critical issue (> 300s)
    Critical,
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Test name that regressed
    pub test_name: String,
    /// Previous average duration
    pub previous_average: Duration,
    /// Current average duration
    pub current_average: Duration,
    /// Percentage increase in execution time
    pub regression_percentage: f64,
    /// When the regression was detected
    pub detected_at: SystemTime,
    /// Severity of the regression
    pub severity: RegressionSeverity,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Severity levels for performance regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor regression (10-25% slower)
    Minor,
    /// Moderate regression (25-50% slower)
    Moderate,
    /// Major regression (50-100% slower)
    Major,
    /// Critical regression (>100% slower)
    Critical,
}

/// Comprehensive performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisReport {
    /// Analysis timestamp
    pub generated_at: SystemTime,
    /// Total tests analyzed
    pub total_tests_analyzed: usize,
    /// Performance trends for all tests
    pub test_trends: Vec<TestPerformanceTrend>,
    /// Detected performance regressions
    pub regressions: Vec<PerformanceRegression>,
    /// Tests that improved performance
    pub improvements: Vec<PerformanceImprovement>,
    /// Overall performance statistics
    pub overall_stats: OverallPerformanceStats,
    /// Recommendations for optimization
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Performance improvement detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    /// Test name that improved
    pub test_name: String,
    /// Previous average duration
    pub previous_average: Duration,
    /// Current average duration
    pub current_average: Duration,
    /// Percentage decrease in execution time
    pub improvement_percentage: f64,
    /// When the improvement was detected
    pub detected_at: SystemTime,
}

/// Overall performance statistics across all tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallPerformanceStats {
    /// Total execution time for all tests
    pub total_execution_time: Duration,
    /// Average test execution time
    pub average_test_time: Duration,
    /// Median test execution time
    pub median_test_time: Duration,
    /// Number of tests in each performance category
    pub category_distribution: HashMap<PerformanceCategory, usize>,
    /// Performance trend over time (positive = getting slower)
    pub overall_trend: f64,
    /// Test suite efficiency score (0-100)
    pub efficiency_score: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Priority level
    pub priority: Priority,
    /// Description of the recommendation
    pub description: String,
    /// Tests that would benefit from this optimization
    pub affected_tests: Vec<String>,
    /// Estimated impact
    pub estimated_impact: String,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Parallelize test execution
    Parallelization,
    /// Mock expensive operations
    Mocking,
    /// Reduce test scope
    ScopeReduction,
    /// Optimize test setup/teardown
    SetupOptimization,
    /// Use test fixtures
    Fixtures,
    /// Split large tests
    TestSplitting,
    /// Resource pooling
    ResourcePooling,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance monitor for tracking test execution over time
pub struct PerformanceMonitor {
    /// Historical performance data
    performance_history: BTreeMap<String, Vec<TestExecutionResult>>,
    /// Configuration for performance analysis
    config: PerformanceMonitorConfig,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    /// Maximum number of historical results to keep per test
    pub max_history_per_test: usize,
    /// Minimum number of runs required for trend analysis
    pub min_runs_for_trend: usize,
    /// Threshold for detecting performance regressions (percentage)
    pub regression_threshold: f64,
    /// Threshold for detecting performance improvements (percentage)
    pub improvement_threshold: f64,
    /// Whether to persist performance data to disk
    pub persist_data: bool,
    /// File path for persisting performance data
    pub data_file_path: Option<String>,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            max_history_per_test: 100,
            min_runs_for_trend: 5,
            regression_threshold: 25.0,  // 25% slower
            improvement_threshold: 15.0, // 15% faster
            persist_data: true,
            data_file_path: Some("target/test_performance_history.json".to_string()),
        }
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: PerformanceMonitorConfig) -> Self {
        let mut monitor = Self {
            performance_history: BTreeMap::new(),
            config,
        };

        // Load existing performance data if available
        if monitor.config.persist_data {
            monitor.load_performance_data();
        }

        monitor
    }

    /// Record a test execution result for performance tracking
    pub fn record_test_result(&mut self, result: TestExecutionResult) {
        let test_results = self
            .performance_history
            .entry(result.test_name.clone())
            .or_insert_with(Vec::new);

        test_results.push(result);

        // Keep history bounded
        if test_results.len() > self.config.max_history_per_test {
            test_results.drain(0..10); // Remove oldest 10 entries
        }

        // Persist data if configured
        if self.config.persist_data {
            self.save_performance_data();
        }
    }

    /// Analyze performance trends for all tracked tests
    pub fn analyze_performance_trends(&self) -> PerformanceAnalysisReport {
        let mut test_trends = Vec::new();
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for (test_name, results) in &self.performance_history {
            if results.len() >= self.config.min_runs_for_trend {
                let trend = self.calculate_test_trend(test_name, results);

                // Check for regressions and improvements
                if let Some(regression) = self.detect_regression(test_name, results) {
                    regressions.push(regression);
                }

                if let Some(improvement) = self.detect_improvement(test_name, results) {
                    improvements.push(improvement);
                }

                test_trends.push(trend);
            }
        }

        let overall_stats = self.calculate_overall_stats(&test_trends);
        let optimization_recommendations = self.generate_optimization_recommendations(&test_trends);

        PerformanceAnalysisReport {
            generated_at: SystemTime::now(),
            total_tests_analyzed: test_trends.len(),
            test_trends,
            regressions,
            improvements,
            overall_stats,
            optimization_recommendations,
        }
    }

    /// Get performance trend for a specific test
    pub fn get_test_trend(&self, test_name: &str) -> Option<TestPerformanceTrend> {
        if let Some(results) = self.performance_history.get(test_name) {
            if results.len() >= self.config.min_runs_for_trend {
                Some(self.calculate_test_trend(test_name, results))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get tests that are consistently slow
    pub fn get_slow_tests(&self) -> Vec<TestPerformanceTrend> {
        self.performance_history
            .iter()
            .filter_map(|(test_name, results)| {
                if results.len() >= self.config.min_runs_for_trend {
                    let trend = self.calculate_test_trend(test_name, results);
                    if matches!(
                        trend.performance_category,
                        PerformanceCategory::Slow
                            | PerformanceCategory::VerySlow
                            | PerformanceCategory::Critical
                    ) {
                        Some(trend)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Generate a report for FIXES.md documentation
    pub fn generate_fixes_report(&self) -> String {
        let analysis = self.analyze_performance_trends();
        let mut report = String::new();

        report.push_str("## Performance Analysis Report\n\n");
        report.push_str(&format!(
            "**Generated**: {}\n",
            chrono::DateTime::<chrono::Utc>::from(analysis.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!(
            "**Tests Analyzed**: {}\n\n",
            analysis.total_tests_analyzed
        ));

        // Performance regressions
        if !analysis.regressions.is_empty() {
            report.push_str("### 🔴 Performance Regressions Detected\n\n");
            for regression in &analysis.regressions {
                report.push_str(&format!("**{}**\n", regression.test_name));
                report.push_str(&format!("- **Severity**: {:?}\n", regression.severity));
                report.push_str(&format!(
                    "- **Previous Average**: {:.2}s\n",
                    regression.previous_average.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Current Average**: {:.2}s\n",
                    regression.current_average.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Regression**: +{:.1}%\n",
                    regression.regression_percentage
                ));

                if !regression.suggested_actions.is_empty() {
                    report.push_str("- **Suggested Actions**:\n");
                    for action in &regression.suggested_actions {
                        report.push_str(&format!("  - {}\n", action));
                    }
                }
                report.push_str("\n");
            }
        }

        // Performance improvements
        if !analysis.improvements.is_empty() {
            report.push_str("### 🟢 Performance Improvements\n\n");
            for improvement in &analysis.improvements {
                report.push_str(&format!("**{}**\n", improvement.test_name));
                report.push_str(&format!(
                    "- **Previous Average**: {:.2}s\n",
                    improvement.previous_average.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Current Average**: {:.2}s\n",
                    improvement.current_average.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Improvement**: -{:.1}%\n",
                    improvement.improvement_percentage
                ));
                report.push_str("\n");
            }
        }

        // Slow tests
        let slow_tests = self.get_slow_tests();
        if !slow_tests.is_empty() {
            report.push_str("### 🐌 Consistently Slow Tests\n\n");
            for test in &slow_tests {
                report.push_str(&format!("**{}**\n", test.test_name));
                report.push_str(&format!(
                    "- **Category**: {:?}\n",
                    test.performance_category
                ));
                report.push_str(&format!(
                    "- **Average Duration**: {:.2}s\n",
                    test.average_duration.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Median Duration**: {:.2}s\n",
                    test.median_duration.as_secs_f64()
                ));
                report.push_str(&format!(
                    "- **Trend**: {}\n",
                    if test.trend_slope > 0.0 {
                        "Getting slower"
                    } else if test.trend_slope < 0.0 {
                        "Getting faster"
                    } else {
                        "Stable"
                    }
                ));
                report.push_str("\n");
            }
        }

        // Optimization recommendations
        if !analysis.optimization_recommendations.is_empty() {
            report.push_str("### 💡 Optimization Recommendations\n\n");

            let mut by_priority: HashMap<Priority, Vec<&OptimizationRecommendation>> =
                HashMap::new();
            for rec in &analysis.optimization_recommendations {
                by_priority.entry(rec.priority).or_default().push(rec);
            }

            for priority in [
                Priority::Critical,
                Priority::High,
                Priority::Medium,
                Priority::Low,
            ] {
                if let Some(recommendations) = by_priority.get(&priority) {
                    report.push_str(&format!("#### {:?} Priority\n\n", priority));
                    for rec in recommendations {
                        report.push_str(&format!("**{:?}**\n", rec.optimization_type));
                        report.push_str(&format!("- **Description**: {}\n", rec.description));
                        report.push_str(&format!(
                            "- **Estimated Impact**: {}\n",
                            rec.estimated_impact
                        ));
                        if !rec.affected_tests.is_empty() {
                            report.push_str(&format!(
                                "- **Affected Tests**: {}\n",
                                rec.affected_tests.join(", ")
                            ));
                        }
                        report.push_str("\n");
                    }
                }
            }
        }

        // Overall statistics
        report.push_str("### 📊 Overall Performance Statistics\n\n");
        report.push_str(&format!(
            "- **Total Execution Time**: {:.2}s\n",
            analysis.overall_stats.total_execution_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "- **Average Test Time**: {:.2}s\n",
            analysis.overall_stats.average_test_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "- **Median Test Time**: {:.2}s\n",
            analysis.overall_stats.median_test_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "- **Efficiency Score**: {:.1}/100\n",
            analysis.overall_stats.efficiency_score
        ));

        report.push_str("\n#### Performance Category Distribution\n\n");
        for (category, count) in &analysis.overall_stats.category_distribution {
            report.push_str(&format!("- **{:?}**: {} tests\n", category, count));
        }

        report
    }

    // Private helper methods

    fn calculate_test_trend(
        &self,
        test_name: &str,
        results: &[TestExecutionResult],
    ) -> TestPerformanceTrend {
        let durations: Vec<Duration> = results.iter().map(|r| r.duration).collect();
        let execution_history: Vec<(SystemTime, Duration)> =
            results.iter().map(|r| (r.timestamp, r.duration)).collect();

        let average_duration = durations.iter().sum::<Duration>() / durations.len() as u32;

        let mut sorted_durations = durations.clone();
        sorted_durations.sort();
        let median_duration = sorted_durations[sorted_durations.len() / 2];

        let std_deviation = self.calculate_std_deviation(&durations, average_duration);
        let trend_slope = self.calculate_trend_slope(&execution_history);

        let performance_category = PerformanceCategory::from_duration(average_duration);
        let is_concerning = matches!(
            performance_category,
            PerformanceCategory::VerySlow | PerformanceCategory::Critical
        ) || trend_slope > 0.1; // Getting significantly slower

        TestPerformanceTrend {
            test_name: test_name.to_string(),
            execution_history,
            average_duration,
            median_duration,
            std_deviation,
            trend_slope,
            is_concerning,
            performance_category,
        }
    }

    fn calculate_std_deviation(&self, durations: &[Duration], average: Duration) -> Duration {
        if durations.len() <= 1 {
            return Duration::ZERO;
        }

        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - average.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / (durations.len() - 1) as f64;

        Duration::from_secs_f64(variance.sqrt())
    }

    fn calculate_trend_slope(&self, history: &[(SystemTime, Duration)]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        // Simple linear regression to calculate trend
        let n = history.len() as f64;
        let x_values: Vec<f64> = (0..history.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = history.iter().map(|(_, d)| d.as_secs_f64()).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;

        let numerator: f64 = x_values
            .iter()
            .zip(&y_values)
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn detect_regression(
        &self,
        test_name: &str,
        results: &[TestExecutionResult],
    ) -> Option<PerformanceRegression> {
        if results.len() < 10 {
            return None; // Need enough data for reliable regression detection
        }

        let split_point = results.len() / 2;
        let older_results = &results[..split_point];
        let newer_results = &results[split_point..];

        let older_avg =
            older_results.iter().map(|r| r.duration).sum::<Duration>() / older_results.len() as u32;
        let newer_avg =
            newer_results.iter().map(|r| r.duration).sum::<Duration>() / newer_results.len() as u32;

        let regression_percentage =
            ((newer_avg.as_secs_f64() - older_avg.as_secs_f64()) / older_avg.as_secs_f64()) * 100.0;

        if regression_percentage > self.config.regression_threshold {
            let severity = match regression_percentage {
                10.0..=25.0 => RegressionSeverity::Minor,
                25.0..=50.0 => RegressionSeverity::Moderate,
                50.0..=100.0 => RegressionSeverity::Major,
                _ => RegressionSeverity::Critical,
            };

            let suggested_actions =
                self.generate_regression_actions(test_name, severity, regression_percentage);

            Some(PerformanceRegression {
                test_name: test_name.to_string(),
                previous_average: older_avg,
                current_average: newer_avg,
                regression_percentage,
                detected_at: SystemTime::now(),
                severity,
                suggested_actions,
            })
        } else {
            None
        }
    }

    fn detect_improvement(
        &self,
        test_name: &str,
        results: &[TestExecutionResult],
    ) -> Option<PerformanceImprovement> {
        if results.len() < 10 {
            return None;
        }

        let split_point = results.len() / 2;
        let older_results = &results[..split_point];
        let newer_results = &results[split_point..];

        let older_avg =
            older_results.iter().map(|r| r.duration).sum::<Duration>() / older_results.len() as u32;
        let newer_avg =
            newer_results.iter().map(|r| r.duration).sum::<Duration>() / newer_results.len() as u32;

        let improvement_percentage =
            ((older_avg.as_secs_f64() - newer_avg.as_secs_f64()) / older_avg.as_secs_f64()) * 100.0;

        if improvement_percentage > self.config.improvement_threshold {
            Some(PerformanceImprovement {
                test_name: test_name.to_string(),
                previous_average: older_avg,
                current_average: newer_avg,
                improvement_percentage,
                detected_at: SystemTime::now(),
            })
        } else {
            None
        }
    }

    fn generate_regression_actions(
        &self,
        _test_name: &str,
        severity: RegressionSeverity,
        _percentage: f64,
    ) -> Vec<String> {
        let mut actions = Vec::new();

        match severity {
            RegressionSeverity::Minor => {
                actions.push("Monitor test performance in upcoming runs".to_string());
                actions.push("Review recent changes that might affect performance".to_string());
            }
            RegressionSeverity::Moderate => {
                actions.push("Investigate recent code changes affecting this test".to_string());
                actions.push("Profile the test to identify performance bottlenecks".to_string());
                actions.push("Consider optimizing test setup or data".to_string());
            }
            RegressionSeverity::Major => {
                actions.push("Urgent investigation required".to_string());
                actions.push("Profile test execution to identify bottlenecks".to_string());
                actions.push("Review and optimize test implementation".to_string());
                actions.push("Consider breaking test into smaller, faster tests".to_string());
            }
            RegressionSeverity::Critical => {
                actions.push("CRITICAL: Immediate action required".to_string());
                actions.push("Investigate if test is hanging or in infinite loop".to_string());
                actions.push("Consider temporarily disabling test until fixed".to_string());
                actions.push("Review for memory leaks or resource issues".to_string());
            }
        }

        actions
    }

    fn calculate_overall_stats(&self, trends: &[TestPerformanceTrend]) -> OverallPerformanceStats {
        if trends.is_empty() {
            return OverallPerformanceStats {
                total_execution_time: Duration::ZERO,
                average_test_time: Duration::ZERO,
                median_test_time: Duration::ZERO,
                category_distribution: HashMap::new(),
                overall_trend: 0.0,
                efficiency_score: 0.0,
            };
        }

        let total_execution_time = trends.iter().map(|t| t.average_duration).sum();
        let average_test_time = total_execution_time / trends.len() as u32;

        let mut durations: Vec<Duration> = trends.iter().map(|t| t.average_duration).collect();
        durations.sort();
        let median_test_time = durations[durations.len() / 2];

        let mut category_distribution = HashMap::new();
        for trend in trends {
            *category_distribution
                .entry(trend.performance_category)
                .or_insert(0) += 1;
        }

        let overall_trend = trends.iter().map(|t| t.trend_slope).sum::<f64>() / trends.len() as f64;

        // Calculate efficiency score (0-100)
        let fast_tests = category_distribution
            .get(&PerformanceCategory::Fast)
            .unwrap_or(&0);
        let total_tests = trends.len();
        let efficiency_score = if total_tests > 0 {
            (*fast_tests as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };

        OverallPerformanceStats {
            total_execution_time,
            average_test_time,
            median_test_time,
            category_distribution,
            overall_trend,
            efficiency_score,
        }
    }

    fn generate_optimization_recommendations(
        &self,
        trends: &[TestPerformanceTrend],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        let slow_tests: Vec<&TestPerformanceTrend> = trends
            .iter()
            .filter(|t| {
                matches!(
                    t.performance_category,
                    PerformanceCategory::Slow
                        | PerformanceCategory::VerySlow
                        | PerformanceCategory::Critical
                )
            })
            .collect();

        if !slow_tests.is_empty() {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::Parallelization,
                priority: Priority::High,
                description: "Parallelize independent slow tests to reduce overall execution time"
                    .to_string(),
                affected_tests: slow_tests.iter().map(|t| t.test_name.clone()).collect(),
                estimated_impact: format!(
                    "Could reduce execution time by up to {}%",
                    (slow_tests.len() as f64 / trends.len() as f64 * 50.0) as u32
                ),
            });
        }

        let very_slow_tests: Vec<&TestPerformanceTrend> = trends
            .iter()
            .filter(|t| {
                matches!(
                    t.performance_category,
                    PerformanceCategory::VerySlow | PerformanceCategory::Critical
                )
            })
            .collect();

        if !very_slow_tests.is_empty() {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::TestSplitting,
                priority: Priority::Critical,
                description: "Break down very slow tests into smaller, focused tests".to_string(),
                affected_tests: very_slow_tests
                    .iter()
                    .map(|t| t.test_name.clone())
                    .collect(),
                estimated_impact: "Significant reduction in individual test times".to_string(),
            });
        }

        recommendations
    }

    fn save_performance_data(&self) {
        if let Some(file_path) = &self.config.data_file_path {
            if let Ok(json) = serde_json::to_string_pretty(&self.performance_history) {
                let _ = fs::write(file_path, json);
            }
        }
    }

    fn load_performance_data(&mut self) {
        if let Some(file_path) = &self.config.data_file_path {
            if Path::new(file_path).exists() {
                if let Ok(content) = fs::read_to_string(file_path) {
                    if let Ok(data) = serde_json::from_str(&content) {
                        self.performance_history = data;
                    }
                }
            }
        }
    }
}

impl PerformanceCategory {
    /// Determine performance category from duration
    pub fn from_duration(duration: Duration) -> Self {
        match duration.as_secs() {
            0..=1 => PerformanceCategory::Fast,
            2..=10 => PerformanceCategory::Moderate,
            11..=60 => PerformanceCategory::Slow,
            61..=300 => PerformanceCategory::VerySlow,
            _ => PerformanceCategory::Critical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_category_classification() {
        assert_eq!(
            PerformanceCategory::from_duration(Duration::from_millis(500)),
            PerformanceCategory::Fast
        );
        assert_eq!(
            PerformanceCategory::from_duration(Duration::from_secs(5)),
            PerformanceCategory::Moderate
        );
        assert_eq!(
            PerformanceCategory::from_duration(Duration::from_secs(30)),
            PerformanceCategory::Slow
        );
        assert_eq!(
            PerformanceCategory::from_duration(Duration::from_secs(120)),
            PerformanceCategory::VerySlow
        );
        assert_eq!(
            PerformanceCategory::from_duration(Duration::from_secs(400)),
            PerformanceCategory::Critical
        );
    }

    #[test]
    fn test_performance_monitor_creation() {
        let config = PerformanceMonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);
        assert_eq!(monitor.performance_history.len(), 0);
    }
}
