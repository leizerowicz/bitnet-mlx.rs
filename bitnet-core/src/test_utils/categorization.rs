//! Test categorization and conditional execution utilities
//!
//! This module provides utilities for categorizing tests, implementing conditional
//! execution based on environment and resources, and managing test skipping logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::{TestCategory, TestExecutionResult};

/// Test categorization rules and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TestCategorizationConfig {
    /// Rules for automatically categorizing tests
    pub categorization_rules: Vec<CategorizationRule>,
    /// Environment-specific execution policies
    pub execution_policies: HashMap<String, ExecutionPolicy>,
    /// Resource-based execution thresholds
    pub resource_thresholds: ResourceThresholds,
    /// Default category for uncategorized tests
    pub default_category: TestCategory,
}

/// Rule for automatically categorizing tests based on patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CategorizationRule {
    /// Pattern to match test names against
    pub name_pattern: String,
    /// Category to assign if pattern matches
    pub category: TestCategory,
    /// Priority of this rule (higher = more important)
    pub priority: u32,
    /// Whether this rule is enabled
    pub enabled: bool,
    /// Description of what this rule categorizes
    pub description: String,
}

/// Execution policy for different environments
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ExecutionPolicy {
    /// Categories allowed to run in this environment
    pub allowed_categories: Vec<TestCategory>,
    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,
    /// Maximum total execution time
    pub max_total_execution_time: Duration,
    /// Whether to skip tests on resource constraints
    pub skip_on_resource_constraints: bool,
    /// Custom timeout overrides by category
    pub timeout_overrides: HashMap<TestCategory, Duration>,
}

/// Resource-based execution thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ResourceThresholds {
    /// Minimum available memory required (bytes)
    pub min_available_memory: u64,
    /// Maximum CPU usage threshold before skipping tests
    pub max_cpu_usage: f64,
    /// Minimum disk space required (bytes)
    pub min_disk_space: u64,
    /// Maximum system load before skipping tests
    pub max_system_load: f64,
}

/// Test execution decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionDecision {
    /// Execute the test normally
    Execute,
    /// Skip the test with reason
    Skip(String),
    /// Execute with modified parameters
    ExecuteModified {
        /// New timeout for the test
        timeout: Duration,
        /// Additional constraints
        constraints: Vec<String>,
    },
}

/// Test categorizer for automatic test classification
#[allow(dead_code)]
pub struct TestCategorizer {
    /// Configuration for categorization
    config: TestCategorizationConfig,
    /// Current environment name
    environment: String,
    /// Cached categorization results
    category_cache: std::sync::RwLock<HashMap<String, TestCategory>>,
}

impl Default for TestCategorizationConfig {
    fn default() -> Self {
        Self {
            categorization_rules: vec![
                CategorizationRule {
                    name_pattern: ".*unit.*".to_string(),
                    category: TestCategory::Unit,
                    priority: 100,
                    enabled: true,
                    description: "Unit tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*integration.*".to_string(),
                    category: TestCategory::Integration,
                    priority: 90,
                    enabled: true,
                    description: "Integration tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*performance.*|.*benchmark.*".to_string(),
                    category: TestCategory::Performance,
                    priority: 80,
                    enabled: true,
                    description: "Performance and benchmark tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*stress.*|.*load.*".to_string(),
                    category: TestCategory::Stress,
                    priority: 70,
                    enabled: true,
                    description: "Stress and load tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*endurance.*|.*soak.*".to_string(),
                    category: TestCategory::Endurance,
                    priority: 60,
                    enabled: true,
                    description: "Endurance and soak tests".to_string(),
                },
                // Specific patterns for known long-running tests
                CategorizationRule {
                    name_pattern: ".*allocation.*pattern.*tracking.*".to_string(),
                    category: TestCategory::Stress,
                    priority: 95,
                    enabled: true,
                    description: "Memory allocation pattern tracking tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*profiling.*session.*".to_string(),
                    category: TestCategory::Stress,
                    priority: 95,
                    enabled: true,
                    description: "Memory profiling session tests".to_string(),
                },
                CategorizationRule {
                    name_pattern: ".*shader.*compilation.*".to_string(),
                    category: TestCategory::Performance,
                    priority: 85,
                    enabled: true,
                    description: "Shader compilation tests".to_string(),
                },
            ],
            execution_policies: Self::default_execution_policies(),
            resource_thresholds: ResourceThresholds::default(),
            default_category: TestCategory::Unit,
        }
    }
}

impl TestCategorizationConfig {
    fn default_execution_policies() -> HashMap<String, ExecutionPolicy> {
        let mut policies = HashMap::new();

        // CI environment policy
        policies.insert(
            "ci".to_string(),
            ExecutionPolicy {
                allowed_categories: vec![
                    TestCategory::Unit,
                    TestCategory::Integration,
                    TestCategory::Performance,
                ],
                max_concurrent_tests: 4,
                max_total_execution_time: Duration::from_secs(1800), // 30 minutes
                skip_on_resource_constraints: true,
                timeout_overrides: HashMap::from([
                    (TestCategory::Unit, Duration::from_secs(30)),
                    (TestCategory::Integration, Duration::from_secs(120)),
                    (TestCategory::Performance, Duration::from_secs(300)),
                ]),
            },
        );

        // Local development policy
        policies.insert(
            "local".to_string(),
            ExecutionPolicy {
                allowed_categories: vec![
                    TestCategory::Unit,
                    TestCategory::Integration,
                    TestCategory::Performance,
                    TestCategory::Stress,
                    TestCategory::Endurance,
                ],
                max_concurrent_tests: 8,
                max_total_execution_time: Duration::from_secs(3600), // 1 hour
                skip_on_resource_constraints: false,
                timeout_overrides: HashMap::new(),
            },
        );

        // Nightly build policy
        policies.insert(
            "nightly".to_string(),
            ExecutionPolicy {
                allowed_categories: vec![
                    TestCategory::Unit,
                    TestCategory::Integration,
                    TestCategory::Performance,
                    TestCategory::Stress,
                    TestCategory::Endurance,
                ],
                max_concurrent_tests: 16,
                max_total_execution_time: Duration::from_secs(7200), // 2 hours
                skip_on_resource_constraints: false,
                timeout_overrides: HashMap::new(),
            },
        );

        policies
    }
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            min_available_memory: 1024 * 1024 * 1024, // 1GB
            max_cpu_usage: 80.0,                      // 80%
            min_disk_space: 1024 * 1024 * 1024,       // 1GB
            max_system_load: 4.0,
        }
    }
}

impl TestCategorizer {
    /// Create a new test categorizer
    pub fn new(config: TestCategorizationConfig, environment: String) -> Self {
        Self {
            config,
            environment,
            category_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Categorize a test based on its name and characteristics
    pub fn categorize_test(&self, test_name: &str) -> TestCategory {
        // Check cache first
        {
            let cache = self.category_cache.read().unwrap();
            if let Some(&category) = cache.get(test_name) {
                return category;
            }
        }

        // Apply categorization rules
        let mut matching_rules: Vec<&CategorizationRule> = self
            .config
            .categorization_rules
            .iter()
            .filter(|rule| rule.enabled && self.matches_pattern(&rule.name_pattern, test_name))
            .collect();

        // Sort by priority (highest first)
        matching_rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        let category = matching_rules
            .first()
            .map(|rule| rule.category)
            .unwrap_or(self.config.default_category);

        // Cache the result
        {
            let mut cache = self.category_cache.write().unwrap();
            cache.insert(test_name.to_string(), category);
        }

        category
    }

    /// Determine if a test should be executed in the current environment
    pub fn should_execute_test(
        &self,
        _test_name: &str,
        category: TestCategory,
    ) -> ExecutionDecision {
        let policy = self
            .config
            .execution_policies
            .get(&self.environment)
            .or_else(|| self.config.execution_policies.get("local"))
            .cloned()
            .unwrap_or_else(|| ExecutionPolicy {
                allowed_categories: vec![TestCategory::Unit, TestCategory::Integration],
                max_concurrent_tests: 4,
                max_total_execution_time: Duration::from_secs(600),
                skip_on_resource_constraints: true,
                timeout_overrides: HashMap::new(),
            });

        // Check if category is allowed
        if !policy.allowed_categories.contains(&category) {
            return ExecutionDecision::Skip(format!(
                "Category {:?} not allowed in {} environment",
                category, self.environment
            ));
        }

        // Check resource constraints
        if policy.skip_on_resource_constraints && !self.check_resource_availability() {
            return ExecutionDecision::Skip("Insufficient system resources available".to_string());
        }

        // Check for timeout overrides
        if let Some(&timeout) = policy.timeout_overrides.get(&category) {
            ExecutionDecision::ExecuteModified {
                timeout,
                constraints: vec![format!(
                    "Timeout overridden for {} environment",
                    self.environment
                )],
            }
        } else {
            ExecutionDecision::Execute
        }
    }

    /// Get execution statistics for the current configuration
    pub fn get_execution_statistics(
        &self,
        test_results: &[TestExecutionResult],
    ) -> ExecutionStatistics {
        let mut stats = ExecutionStatistics::default();

        for result in test_results {
            let category = self.categorize_test(&result.test_name);
            let decision = self.should_execute_test(&result.test_name, category);

            stats.total_tests += 1;

            match decision {
                ExecutionDecision::Execute | ExecutionDecision::ExecuteModified { .. } => {
                    stats.executed_tests += 1;
                    *stats.executed_by_category.entry(category).or_insert(0) += 1;

                    if result.success {
                        stats.successful_tests += 1;
                    }

                    if result.timed_out {
                        stats.timed_out_tests += 1;
                    }

                    stats.total_execution_time += result.duration;
                }
                ExecutionDecision::Skip(_) => {
                    stats.skipped_tests += 1;
                    *stats.skipped_by_category.entry(category).or_insert(0) += 1;
                }
            }
        }

        stats
    }

    /// Generate a categorization report
    pub fn generate_categorization_report(
        &self,
        test_results: &[TestExecutionResult],
    ) -> CategorizationReport {
        let stats = self.get_execution_statistics(test_results);

        let mut category_breakdown = HashMap::new();
        for result in test_results {
            let category = self.categorize_test(&result.test_name);
            let entry = category_breakdown
                .entry(category)
                .or_insert(CategoryBreakdown::default());

            entry.total_tests += 1;
            entry.total_duration += result.duration;

            if result.success {
                entry.successful_tests += 1;
            }

            if result.timed_out {
                entry.timed_out_tests += 1;
            }
        }

        let policy_effectiveness = self.calculate_policy_effectiveness(&stats);
        let recommendations =
            self.generate_categorization_recommendations(&stats, &category_breakdown);

        CategorizationReport {
            environment: self.environment.clone(),
            execution_statistics: stats,
            category_breakdown,
            policy_effectiveness,
            recommendations,
            generated_at: std::time::SystemTime::now(),
        }
    }

    /// Update categorization rules based on test execution patterns
    pub fn update_rules_from_patterns(&mut self, test_results: &[TestExecutionResult]) {
        let mut pattern_analysis = HashMap::new();

        for result in test_results {
            let category = TestCategory::from_duration(result.duration);
            let entry = pattern_analysis.entry(category).or_insert(Vec::new());
            entry.push(result.test_name.clone());
        }

        // Generate new rules for patterns that consistently fall into specific categories
        for (category, test_names) in pattern_analysis {
            if test_names.len() >= 3 {
                // Look for common patterns
                let common_patterns = self.extract_common_patterns(&test_names);

                for pattern in common_patterns {
                    // Check if we already have a rule for this pattern
                    let exists = self
                        .config
                        .categorization_rules
                        .iter()
                        .any(|rule| rule.name_pattern == pattern);

                    if !exists {
                        let new_rule = CategorizationRule {
                            name_pattern: pattern.clone(),
                            category,
                            priority: 50, // Medium priority for auto-generated rules
                            enabled: true,
                            description: format!("Auto-generated rule for {:?} tests", category),
                        };

                        self.config.categorization_rules.push(new_rule);
                    }
                }
            }
        }

        // Clear cache to force re-categorization
        {
            let mut cache = self.category_cache.write().unwrap();
            cache.clear();
        }
    }

    // Private helper methods

    fn matches_pattern(&self, pattern: &str, test_name: &str) -> bool {
        // Handle OR patterns (|)
        if pattern.contains('|') {
            return pattern.split('|').any(|sub_pattern| {
                self.matches_single_pattern(sub_pattern.trim(), test_name)
            });
        }
        
        self.matches_single_pattern(pattern, test_name)
    }

    fn matches_single_pattern(&self, pattern: &str, test_name: &str) -> bool {
        // Simple regex-like matching for .* patterns
        if pattern.contains(".*") {
            let parts: Vec<&str> = pattern.split(".*").filter(|s| !s.is_empty()).collect();
            
            if parts.is_empty() {
                // Pattern is just ".*", matches everything
                return true;
            }
            
            // Check if all parts appear in order in the test name
            let mut search_from = 0;
            for part in parts {
                if let Some(pos) = test_name[search_from..].find(part) {
                    search_from += pos + part.len();
                } else {
                    return false;
                }
            }
            true
        } else {
            test_name.contains(pattern)
        }
    }

    fn check_resource_availability(&self) -> bool {
        // Simplified resource check - in practice would use system APIs
        // For now, assume resources are available
        true
    }

    fn calculate_policy_effectiveness(&self, stats: &ExecutionStatistics) -> PolicyEffectiveness {
        let execution_rate = if stats.total_tests > 0 {
            stats.executed_tests as f64 / stats.total_tests as f64
        } else {
            0.0
        };

        let success_rate = if stats.executed_tests > 0 {
            stats.successful_tests as f64 / stats.executed_tests as f64
        } else {
            0.0
        };

        let timeout_rate = if stats.executed_tests > 0 {
            stats.timed_out_tests as f64 / stats.executed_tests as f64
        } else {
            0.0
        };

        let avg_execution_time = if stats.executed_tests > 0 {
            stats.total_execution_time / stats.executed_tests as u32
        } else {
            Duration::ZERO
        };

        PolicyEffectiveness {
            execution_rate,
            success_rate,
            timeout_rate,
            avg_execution_time,
            resource_utilization: 0.5, // Simplified
        }
    }

    fn generate_categorization_recommendations(
        &self,
        stats: &ExecutionStatistics,
        breakdown: &HashMap<TestCategory, CategoryBreakdown>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for categories with high timeout rates
        for (category, breakdown) in breakdown {
            if breakdown.total_tests > 0 {
                let timeout_rate = breakdown.timed_out_tests as f64 / breakdown.total_tests as f64;
                if timeout_rate > 0.2 {
                    recommendations.push(format!(
                        "Consider increasing timeout for {:?} tests ({}% timeout rate)",
                        category,
                        (timeout_rate * 100.0) as u32
                    ));
                }
            }
        }

        // Check for skipped categories
        if stats.skipped_tests > stats.executed_tests / 2 {
            recommendations
                .push("High skip rate detected - consider adjusting execution policy".to_string());
        }

        // Check for long execution times
        if stats.total_execution_time > Duration::from_secs(1800) {
            recommendations.push(
                "Total execution time is high - consider parallelization or test optimization"
                    .to_string(),
            );
        }

        recommendations
    }

    fn extract_common_patterns(&self, test_names: &[String]) -> Vec<String> {
        let mut patterns = Vec::new();

        // Simple pattern extraction - look for common prefixes/suffixes
        if test_names.len() >= 3 {
            // Find common prefix
            if let Some(prefix) = self.find_common_prefix(test_names) {
                if prefix.len() > 3 {
                    patterns.push(format!("{}.*", prefix));
                }
            }

            // Find common suffix
            if let Some(suffix) = self.find_common_suffix(test_names) {
                if suffix.len() > 3 {
                    patterns.push(format!(".*{}", suffix));
                }
            }
        }

        patterns
    }

    fn find_common_prefix(&self, strings: &[String]) -> Option<String> {
        if strings.is_empty() {
            return None;
        }

        let first = &strings[0];
        let mut prefix_len = 0;

        for i in 0..first.len() {
            let ch = first.chars().nth(i)?;
            if strings.iter().all(|s| s.chars().nth(i) == Some(ch)) {
                prefix_len = i + 1;
            } else {
                break;
            }
        }

        if prefix_len > 0 {
            Some(first[..prefix_len].to_string())
        } else {
            None
        }
    }

    fn find_common_suffix(&self, strings: &[String]) -> Option<String> {
        if strings.is_empty() {
            return None;
        }

        let first = &strings[0];
        let mut suffix_len = 0;

        for i in 1..=first.len() {
            let pos = first.len() - i;
            let ch = first.chars().nth(pos)?;
            if strings
                .iter()
                .all(|s| s.len() >= i && s.chars().nth(s.len() - i) == Some(ch))
            {
                suffix_len = i;
            } else {
                break;
            }
        }

        if suffix_len > 0 {
            Some(first[first.len() - suffix_len..].to_string())
        } else {
            None
        }
    }
}

/// Execution statistics for test categorization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ExecutionStatistics {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of tests executed
    pub executed_tests: usize,
    /// Number of tests skipped
    pub skipped_tests: usize,
    /// Number of successful tests
    pub successful_tests: usize,
    /// Number of tests that timed out
    pub timed_out_tests: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Tests executed by category
    pub executed_by_category: HashMap<TestCategory, usize>,
    /// Tests skipped by category
    pub skipped_by_category: HashMap<TestCategory, usize>,
}

/// Breakdown of tests by category
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CategoryBreakdown {
    /// Total tests in this category
    pub total_tests: usize,
    /// Successful tests in this category
    pub successful_tests: usize,
    /// Timed out tests in this category
    pub timed_out_tests: usize,
    /// Total duration for this category
    pub total_duration: Duration,
}

/// Policy effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PolicyEffectiveness {
    /// Percentage of tests executed (vs skipped)
    pub execution_rate: f64,
    /// Success rate of executed tests
    pub success_rate: f64,
    /// Timeout rate of executed tests
    pub timeout_rate: f64,
    /// Average execution time per test
    pub avg_execution_time: Duration,
    /// Resource utilization efficiency
    pub resource_utilization: f64,
}

/// Comprehensive categorization report
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CategorizationReport {
    /// Environment this report was generated for
    pub environment: String,
    /// Execution statistics
    pub execution_statistics: ExecutionStatistics,
    /// Breakdown by category
    pub category_breakdown: HashMap<TestCategory, CategoryBreakdown>,
    /// Policy effectiveness metrics
    pub policy_effectiveness: PolicyEffectiveness,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// When this report was generated
    pub generated_at: std::time::SystemTime,
}

/// Utility functions for test categorization
pub mod utils {
    use super::*;

    /// Detect environment from environment variables
    pub fn detect_environment() -> String {
        if std::env::var("CI").is_ok() {
            "ci".to_string()
        } else if std::env::var("NIGHTLY_BUILD").is_ok() {
            "nightly".to_string()
        } else {
            "local".to_string()
        }
    }

    /// Create a categorizer with default configuration for the current environment
    pub fn create_default_categorizer() -> TestCategorizer {
        let environment = detect_environment();
        let config = TestCategorizationConfig::default();
        TestCategorizer::new(config, environment)
    }

    /// Check if a test should be skipped based on name patterns
    pub fn should_skip_test(test_name: &str) -> bool {
        let skip_patterns = ["test_allocation_pattern_tracking", "test_profiling_session"];

        let environment = detect_environment();
        if environment == "ci" {
            skip_patterns
                .iter()
                .any(|pattern| test_name.contains(pattern))
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorization_rules() {
        let config = TestCategorizationConfig::default();
        let categorizer = TestCategorizer::new(config, "local".to_string());

        assert_eq!(
            categorizer.categorize_test("test_unit_example"),
            TestCategory::Unit
        );
        assert_eq!(
            categorizer.categorize_test("test_integration_example"),
            TestCategory::Integration
        );
        assert_eq!(
            categorizer.categorize_test("test_performance_benchmark"),
            TestCategory::Performance
        );
        assert_eq!(
            categorizer.categorize_test("test_allocation_pattern_tracking"),
            TestCategory::Stress
        );
    }

    #[test]
    fn test_execution_decisions() {
        let config = TestCategorizationConfig::default();
        let categorizer = TestCategorizer::new(config, "ci".to_string());

        let decision = categorizer.should_execute_test("test_unit", TestCategory::Unit);
        assert!(matches!(decision, ExecutionDecision::ExecuteModified { .. }));

        let decision = categorizer.should_execute_test("test_endurance", TestCategory::Endurance);
        assert!(matches!(decision, ExecutionDecision::Skip(_)));
    }

    #[test]
    fn test_pattern_matching() {
        let config = TestCategorizationConfig::default();
        let categorizer = TestCategorizer::new(config, "local".to_string());

        assert!(categorizer.matches_pattern(".*unit.*", "test_unit_example"));
        assert!(categorizer.matches_pattern(".*performance.*", "benchmark_performance_test"));
        assert!(!categorizer.matches_pattern(".*unit.*", "integration_test"));
    }

    #[test]
    fn test_environment_detection() {
        std::env::set_var("CI", "true");
        assert_eq!(utils::detect_environment(), "ci");
        std::env::remove_var("CI");

        std::env::set_var("NIGHTLY_BUILD", "true");
        assert_eq!(utils::detect_environment(), "nightly");
        std::env::remove_var("NIGHTLY_BUILD");

        assert_eq!(utils::detect_environment(), "local");
    }
}
