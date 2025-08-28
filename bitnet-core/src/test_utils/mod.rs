//! Test utilities for timeout handling, performance monitoring, and error detection
//!
//! This module provides comprehensive test infrastructure for identifying and managing
//! long-running tests, including timeout handling, performance profiling, and
//! automated documentation generation.

pub mod categorization;
pub mod error;
pub mod monitoring;
pub mod performance;
pub mod timeout;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Test execution result with performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TestExecutionResult {
    /// Test name
    pub test_name: String,
    /// Execution duration
    pub duration: Duration,
    /// Whether the test completed successfully
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Test category
    pub category: TestCategory,
    /// Resource usage during test
    pub resource_usage: ResourceUsage,
    /// Whether test exceeded timeout
    pub timed_out: bool,
    /// Timestamp when test was executed
    pub timestamp: std::time::SystemTime,
}

/// Test category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestCategory {
    /// Fast unit tests (< 1s)
    Unit,
    /// Integration tests (1-10s)
    Integration,
    /// Performance tests (10-60s)
    Performance,
    /// Stress tests (60-300s)
    Stress,
    /// Long-running endurance tests (> 300s)
    Endurance,
}

/// Resource usage metrics during test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ResourceUsage {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average CPU usage percentage
    pub avg_cpu_percentage: f64,
    /// GPU usage if applicable
    pub gpu_usage: Option<f64>,
    /// Number of allocations made
    pub allocation_count: u64,
    /// Total bytes allocated
    pub total_allocated_bytes: u64,
}

/// Test timeout configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TimeoutConfig {
    /// Default timeout for unit tests
    pub unit_timeout: Duration,
    /// Default timeout for integration tests
    pub integration_timeout: Duration,
    /// Default timeout for performance tests
    pub performance_timeout: Duration,
    /// Default timeout for stress tests
    pub stress_timeout: Duration,
    /// Default timeout for endurance tests
    pub endurance_timeout: Duration,
    /// Whether to skip long-running tests in CI
    pub skip_long_running_in_ci: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            unit_timeout: Duration::from_secs(5),
            integration_timeout: Duration::from_secs(30),
            performance_timeout: Duration::from_secs(120),
            stress_timeout: Duration::from_secs(300),
            endurance_timeout: Duration::from_secs(600),
            skip_long_running_in_ci: true,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            peak_memory_bytes: 0,
            avg_cpu_percentage: 0.0,
            gpu_usage: None,
            allocation_count: 0,
            total_allocated_bytes: 0,
        }
    }
}

impl TestCategory {
    /// Get the default timeout for this test category
    pub fn default_timeout(&self) -> Duration {
        match self {
            TestCategory::Unit => Duration::from_secs(5),
            TestCategory::Integration => Duration::from_secs(30),
            TestCategory::Performance => Duration::from_secs(120),
            TestCategory::Stress => Duration::from_secs(300),
            TestCategory::Endurance => Duration::from_secs(600),
        }
    }

    /// Determine if this category should be skipped in CI
    pub fn should_skip_in_ci(&self) -> bool {
        matches!(self, TestCategory::Stress | TestCategory::Endurance)
    }

    /// Get category from execution duration
    pub fn from_duration(duration: Duration) -> Self {
        match duration.as_secs() {
            0..=1 => TestCategory::Unit,
            2..=10 => TestCategory::Integration,
            11..=60 => TestCategory::Performance,
            61..=300 => TestCategory::Stress,
            _ => TestCategory::Endurance,
        }
    }
}

/// Global test performance tracker
#[allow(dead_code)]
pub struct TestPerformanceTracker {
    /// Test execution history
    results: std::sync::Mutex<Vec<TestExecutionResult>>,
    /// Configuration
    config: TimeoutConfig,
}

impl TestPerformanceTracker {
    /// Create a new performance tracker
    pub fn new(config: TimeoutConfig) -> Self {
        Self {
            results: std::sync::Mutex::new(Vec::new()),
            config,
        }
    }

    /// Record a test execution result
    pub fn record_result(&self, result: TestExecutionResult) {
        let mut results = self.results.lock().unwrap();
        results.push(result);

        // Keep only recent results (last 1000)
        if results.len() > 1000 {
            results.drain(0..100);
        }
    }

    /// Get all test results
    pub fn get_results(&self) -> Vec<TestExecutionResult> {
        let results = self.results.lock().unwrap();
        results.clone()
    }

    /// Get long-running tests (above category threshold)
    pub fn get_long_running_tests(&self) -> Vec<TestExecutionResult> {
        let results = self.results.lock().unwrap();
        results
            .iter()
            .filter(|r| r.duration > r.category.default_timeout())
            .cloned()
            .collect()
    }

    /// Generate performance report
    pub fn generate_report(&self) -> TestPerformanceReport {
        let results = self.results.lock().unwrap();

        let total_tests = results.len();
        let successful_tests = results.iter().filter(|r| r.success).count();
        let timed_out_tests = results.iter().filter(|r| r.timed_out).count();

        let mut category_stats = HashMap::new();
        let mut long_running_tests = Vec::new();

        for result in results.iter() {
            // Category statistics
            let stats = category_stats
                .entry(result.category)
                .or_insert(CategoryStats::default());
            stats.total_count += 1;
            stats.total_duration += result.duration;
            if result.success {
                stats.success_count += 1;
            }
            if result.timed_out {
                stats.timeout_count += 1;
            }

            // Long-running test detection
            if result.duration > result.category.default_timeout() {
                long_running_tests.push(result.clone());
            }
        }

        // Sort long-running tests by duration (longest first)
        long_running_tests.sort_by(|a, b| b.duration.cmp(&a.duration));

        TestPerformanceReport {
            total_tests,
            successful_tests,
            timed_out_tests,
            category_stats,
            long_running_tests,
            generated_at: std::time::SystemTime::now(),
        }
    }
}

/// Statistics for a test category
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CategoryStats {
    /// Total number of tests in this category
    pub total_count: usize,
    /// Number of successful tests
    pub success_count: usize,
    /// Number of timed out tests
    pub timeout_count: usize,
    /// Total execution time for all tests
    pub total_duration: Duration,
}

impl CategoryStats {
    /// Calculate average duration for this category
    pub fn average_duration(&self) -> Duration {
        if self.total_count > 0 {
            self.total_duration / self.total_count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Calculate success rate for this category
    pub fn success_rate(&self) -> f64 {
        if self.total_count > 0 {
            self.success_count as f64 / self.total_count as f64
        } else {
            0.0
        }
    }
}

/// Comprehensive test performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TestPerformanceReport {
    /// Total number of tests executed
    pub total_tests: usize,
    /// Number of successful tests
    pub successful_tests: usize,
    /// Number of tests that timed out
    pub timed_out_tests: usize,
    /// Statistics by test category
    pub category_stats: HashMap<TestCategory, CategoryStats>,
    /// List of long-running tests
    pub long_running_tests: Vec<TestExecutionResult>,
    /// When this report was generated
    pub generated_at: std::time::SystemTime,
}

impl TestPerformanceReport {
    /// Calculate overall success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_tests > 0 {
            self.successful_tests as f64 / self.total_tests as f64
        } else {
            0.0
        }
    }

    /// Get tests that should be documented in FIXES.md
    pub fn get_problematic_tests(&self) -> Vec<&TestExecutionResult> {
        self.long_running_tests
            .iter()
            .filter(|test| {
                // Include tests that are significantly slower than expected
                test.duration > test.category.default_timeout() * 2
                    || test.timed_out
                    || !test.success
            })
            .collect()
    }

    /// Generate markdown documentation for FIXES.md
    pub fn generate_fixes_documentation(&self) -> String {
        let mut doc = String::new();

        doc.push_str("## Long-Running Test Analysis\n\n");
        doc.push_str(&format!(
            "**Analysis Date**: {}\n",
            chrono::DateTime::<chrono::Utc>::from(self.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC")
        ));
        doc.push_str(&format!("**Total Tests Analyzed**: {}\n", self.total_tests));
        doc.push_str(&format!(
            "**Overall Success Rate**: {:.1}%\n\n",
            self.success_rate() * 100.0
        ));

        // Long-running tests section
        if !self.long_running_tests.is_empty() {
            doc.push_str("### Long-Running Tests Identified\n\n");

            for test in &self.long_running_tests {
                doc.push_str(&format!("**{}**\n", test.test_name));
                doc.push_str(&format!(
                    "- **Duration**: {:.2}s (expected: {:.2}s)\n",
                    test.duration.as_secs_f64(),
                    test.category.default_timeout().as_secs_f64()
                ));
                doc.push_str(&format!("- **Category**: {:?}\n", test.category));
                doc.push_str(&format!(
                    "- **Status**: {}\n",
                    if test.success {
                        "✅ Passed"
                    } else {
                        "❌ Failed"
                    }
                ));

                if test.timed_out {
                    doc.push_str("- **Issue**: ⏰ Test timed out\n");
                }

                // Resource usage
                doc.push_str(&format!(
                    "- **Peak Memory**: {:.2} MB\n",
                    test.resource_usage.peak_memory_bytes as f64 / 1024.0 / 1024.0
                ));
                doc.push_str(&format!(
                    "- **Allocations**: {}\n",
                    test.resource_usage.allocation_count
                ));

                // Error message if available
                if let Some(error) = &test.error_message {
                    doc.push_str(&format!("- **Error**: {}\n", error));
                }

                doc.push_str("\n");
            }
        }

        // Category statistics
        doc.push_str("### Test Performance by Category\n\n");
        for (category, stats) in &self.category_stats {
            doc.push_str(&format!("**{:?} Tests**\n", category));
            doc.push_str(&format!("- Total: {}\n", stats.total_count));
            doc.push_str(&format!(
                "- Success Rate: {:.1}%\n",
                stats.success_rate() * 100.0
            ));
            doc.push_str(&format!(
                "- Average Duration: {:.2}s\n",
                stats.average_duration().as_secs_f64()
            ));
            doc.push_str(&format!("- Timeouts: {}\n", stats.timeout_count));
            doc.push_str("\n");
        }

        // Recommendations
        doc.push_str("### Recommendations\n\n");

        let problematic_tests = self.get_problematic_tests();
        if !problematic_tests.is_empty() {
            doc.push_str("#### Immediate Actions Required\n\n");
            for test in problematic_tests {
                if test.timed_out {
                    doc.push_str(&format!("- **{}**: Investigate timeout cause, consider increasing timeout or optimizing test\n", test.test_name));
                } else if test.duration > test.category.default_timeout() * 3 {
                    doc.push_str(&format!("- **{}**: Extremely slow test, consider breaking into smaller tests or optimizing\n", test.test_name));
                } else if !test.success {
                    doc.push_str(&format!(
                        "- **{}**: Test failure needs investigation\n",
                        test.test_name
                    ));
                }
            }
            doc.push_str("\n");
        }

        doc.push_str("#### General Optimizations\n\n");
        doc.push_str("- Consider running long-running tests only in nightly CI builds\n");
        doc.push_str("- Implement test parallelization for independent tests\n");
        doc.push_str("- Add conditional test execution based on available resources\n");
        doc.push_str("- Consider mocking expensive operations in unit tests\n\n");

        doc
    }
}

// Global instance for tracking test performance
lazy_static::lazy_static! {
    pub static ref GLOBAL_TEST_TRACKER: TestPerformanceTracker =
        TestPerformanceTracker::new(TimeoutConfig::default());
}
