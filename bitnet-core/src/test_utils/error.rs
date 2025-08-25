//! Test Error Handling System
//!
//! This module provides comprehensive error handling for the test infrastructure,
//! including error classification, recovery mechanisms, and detailed reporting.

use std::fmt;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::TestCategory;

/// Comprehensive test error types
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum TestError {
    /// Test exceeded timeout
    #[error("Test '{test_name}' timed out after {duration:?} (category: {category:?})")]
    Timeout {
        test_name: String,
        duration: Duration,
        category: TestCategory,
        resource_usage: Option<String>,
    },

    /// Test panicked with a message
    #[error("Test '{test_name}' panicked: {panic_message}")]
    Panic {
        test_name: String,
        panic_message: String,
        backtrace: Option<String>,
    },

    /// Memory-related test failure
    #[error("Test '{test_name}' failed due to memory issue: {message}")]
    Memory {
        test_name: String,
        message: String,
        allocated_bytes: Option<usize>,
        peak_memory: Option<usize>,
    },

    /// Resource exhaustion during test
    #[error("Test '{test_name}' exhausted resources: {resource_type}")]
    ResourceExhaustion {
        test_name: String,
        resource_type: String,
        current_usage: Option<f64>,
        limit: Option<f64>,
    },

    /// Cross-crate integration failure
    #[error("Integration test '{test_name}' failed: {component_error}")]
    Integration {
        test_name: String,
        component_error: String,
        failed_crate: String,
        dependency_chain: Vec<String>,
    },

    /// Performance regression detected
    #[error("Performance regression in '{test_name}': {regression_type}")]
    PerformanceRegression {
        test_name: String,
        regression_type: String,
        current_time: Duration,
        baseline_time: Duration,
        regression_percentage: f64,
    },

    /// Device or hardware specific error
    #[error("Hardware error in '{test_name}': {device_error}")]
    Hardware {
        test_name: String,
        device_error: String,
        device_type: String,
        recovery_attempted: bool,
    },

    /// Test environment issue
    #[error("Environment error for '{test_name}': {env_error}")]
    Environment {
        test_name: String,
        env_error: String,
        missing_dependencies: Vec<String>,
        env_vars: HashMap<String, String>,
    },

    /// Test validation failure
    #[error("Validation error in '{test_name}': {validation_error}")]
    Validation {
        test_name: String,
        validation_error: String,
        expected_value: Option<String>,
        actual_value: Option<String>,
    },

    /// General test failure with context
    #[error("Test '{test_name}' failed: {message}")]
    General {
        test_name: String,
        message: String,
        error_code: Option<i32>,
        context: HashMap<String, String>,
    },
}

/// Error severity classification for tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestErrorSeverity {
    /// Low severity - test flakiness, minor performance issues
    Low,
    /// Medium severity - test failures that don't block development
    Medium,
    /// High severity - critical test failures that block development
    High,
    /// Critical severity - infrastructure failures, system-wide issues
    Critical,
}

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorRecoveryStrategy {
    /// Retry the test with exponential backoff
    Retry {
        max_attempts: usize,
        backoff_ms: u64,
    },
    /// Skip the test and mark as known issue
    Skip {
        reason: String,
        tracking_issue: Option<String>,
    },
    /// Run test in degraded mode with reduced requirements
    Degrade {
        fallback_category: TestCategory,
        reduced_timeout: Duration,
    },
    /// Fail fast and stop test execution
    FailFast {
        reason: String,
    },
    /// Continue with warning
    ContinueWithWarning {
        warning_message: String,
    },
}

/// Comprehensive error context for test failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestErrorContext {
    /// The test error itself
    pub error: TestError,
    /// Error severity
    pub severity: TestErrorSeverity,
    /// Recovery strategy to apply
    pub recovery_strategy: ErrorRecoveryStrategy,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Related errors that may have contributed
    pub related_errors: Vec<String>,
}

impl TestError {
    /// Create a timeout error
    pub fn timeout(test_name: String, duration: Duration, category: TestCategory) -> Self {
        TestError::Timeout {
            test_name,
            duration,
            category,
            resource_usage: None,
        }
    }

    /// Create a panic error
    pub fn panic(test_name: String, panic_message: String) -> Self {
        TestError::Panic {
            test_name,
            panic_message,
            backtrace: None,
        }
    }

    /// Create a memory error
    pub fn memory(test_name: String, message: String) -> Self {
        TestError::Memory {
            test_name,
            message,
            allocated_bytes: None,
            peak_memory: None,
        }
    }

    /// Create an integration error
    pub fn integration(test_name: String, component_error: String, failed_crate: String) -> Self {
        TestError::Integration {
            test_name,
            component_error,
            failed_crate,
            dependency_chain: Vec::new(),
        }
    }

    /// Create a performance regression error
    pub fn performance_regression(
        test_name: String,
        current_time: Duration,
        baseline_time: Duration,
    ) -> Self {
        let regression_percentage = 
            ((current_time.as_secs_f64() / baseline_time.as_secs_f64()) - 1.0) * 100.0;
        
        TestError::PerformanceRegression {
            test_name,
            regression_type: "Execution time regression".to_string(),
            current_time,
            baseline_time,
            regression_percentage,
        }
    }

    /// Get the test name from any error variant
    pub fn test_name(&self) -> &str {
        match self {
            TestError::Timeout { test_name, .. } => test_name,
            TestError::Panic { test_name, .. } => test_name,
            TestError::Memory { test_name, .. } => test_name,
            TestError::ResourceExhaustion { test_name, .. } => test_name,
            TestError::Integration { test_name, .. } => test_name,
            TestError::PerformanceRegression { test_name, .. } => test_name,
            TestError::Hardware { test_name, .. } => test_name,
            TestError::Environment { test_name, .. } => test_name,
            TestError::Validation { test_name, .. } => test_name,
            TestError::General { test_name, .. } => test_name,
        }
    }

    /// Determine error severity based on error type
    pub fn severity(&self) -> TestErrorSeverity {
        match self {
            TestError::Timeout { category, .. } => {
                match category {
                    TestCategory::Unit => TestErrorSeverity::Medium,
                    TestCategory::Integration => TestErrorSeverity::High,
                    TestCategory::Performance | TestCategory::Stress | TestCategory::Endurance => TestErrorSeverity::Low,
                }
            },
            TestError::Panic { .. } => TestErrorSeverity::High,
            TestError::Memory { .. } => TestErrorSeverity::High,
            TestError::ResourceExhaustion { .. } => TestErrorSeverity::Medium,
            TestError::Integration { .. } => TestErrorSeverity::High,
            TestError::PerformanceRegression { regression_percentage, .. } => {
                if *regression_percentage > 100.0 {
                    TestErrorSeverity::Critical
                } else if *regression_percentage > 50.0 {
                    TestErrorSeverity::High
                } else if *regression_percentage > 25.0 {
                    TestErrorSeverity::Medium
                } else {
                    TestErrorSeverity::Low
                }
            },
            TestError::Hardware { .. } => TestErrorSeverity::High,
            TestError::Environment { .. } => TestErrorSeverity::Medium,
            TestError::Validation { .. } => TestErrorSeverity::High,
            TestError::General { .. } => TestErrorSeverity::Medium,
        }
    }

    /// Suggest recovery strategy based on error type
    pub fn suggest_recovery_strategy(&self) -> ErrorRecoveryStrategy {
        match self {
            TestError::Timeout { category, .. } => {
                match category {
                    TestCategory::Unit | TestCategory::Integration => {
                        ErrorRecoveryStrategy::Retry {
                            max_attempts: 3,
                            backoff_ms: 1000,
                        }
                    },
                    _ => ErrorRecoveryStrategy::Skip {
                        reason: "Long-running test timed out".to_string(),
                        tracking_issue: None,
                    },
                }
            },
            TestError::Panic { .. } => {
                ErrorRecoveryStrategy::FailFast {
                    reason: "Test panicked - indicates serious issue".to_string(),
                }
            },
            TestError::Memory { .. } => {
                ErrorRecoveryStrategy::Degrade {
                    fallback_category: TestCategory::Unit,
                    reduced_timeout: Duration::from_secs(10),
                }
            },
            TestError::ResourceExhaustion { .. } => {
                ErrorRecoveryStrategy::Retry {
                    max_attempts: 2,
                    backoff_ms: 5000,
                }
            },
            TestError::Integration { .. } => {
                ErrorRecoveryStrategy::ContinueWithWarning {
                    warning_message: "Integration test failed - may indicate cross-crate issues".to_string(),
                }
            },
            TestError::PerformanceRegression { regression_percentage, .. } => {
                if *regression_percentage > 100.0 {
                    ErrorRecoveryStrategy::FailFast {
                        reason: "Critical performance regression detected".to_string(),
                    }
                } else {
                    ErrorRecoveryStrategy::ContinueWithWarning {
                        warning_message: format!("Performance regression: {:.1}%", regression_percentage),
                    }
                }
            },
            TestError::Hardware { recovery_attempted, .. } => {
                if *recovery_attempted {
                    ErrorRecoveryStrategy::Skip {
                        reason: "Hardware error with failed recovery".to_string(),
                        tracking_issue: None,
                    }
                } else {
                    ErrorRecoveryStrategy::Retry {
                        max_attempts: 2,
                        backoff_ms: 2000,
                    }
                }
            },
            TestError::Environment { .. } => {
                ErrorRecoveryStrategy::Skip {
                    reason: "Environment configuration issue".to_string(),
                    tracking_issue: None,
                }
            },
            TestError::Validation { .. } => {
                ErrorRecoveryStrategy::FailFast {
                    reason: "Validation error indicates test logic issue".to_string(),
                }
            },
            TestError::General { .. } => {
                ErrorRecoveryStrategy::Retry {
                    max_attempts: 2,
                    backoff_ms: 1000,
                }
            },
        }
    }
}

impl TestErrorContext {
    /// Create a new test error context
    pub fn new(error: TestError) -> Self {
        let severity = error.severity();
        let recovery_strategy = error.suggest_recovery_strategy();
        
        TestErrorContext {
            error,
            severity,
            recovery_strategy,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            stack_trace: None,
            related_errors: Vec::new(),
        }
    }

    /// Add metadata to the error context
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Add stack trace if available
    pub fn with_stack_trace(mut self, stack_trace: String) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }

    /// Add related error information
    pub fn with_related_error(mut self, related_error: String) -> Self {
        self.related_errors.push(related_error);
        self
    }

    /// Override the recovery strategy
    pub fn with_recovery_strategy(mut self, strategy: ErrorRecoveryStrategy) -> Self {
        self.recovery_strategy = strategy;
        self
    }

    /// Check if error should be retried
    pub fn should_retry(&self) -> bool {
        matches!(self.recovery_strategy, ErrorRecoveryStrategy::Retry { .. })
    }

    /// Check if test should be skipped
    pub fn should_skip(&self) -> bool {
        matches!(self.recovery_strategy, ErrorRecoveryStrategy::Skip { .. })
    }

    /// Check if execution should continue
    pub fn should_continue(&self) -> bool {
        matches!(self.recovery_strategy, ErrorRecoveryStrategy::ContinueWithWarning { .. })
    }

    /// Check if should fail fast
    pub fn should_fail_fast(&self) -> bool {
        matches!(self.recovery_strategy, ErrorRecoveryStrategy::FailFast { .. })
    }
}

/// Test error handler for managing error responses
pub struct TestErrorHandler {
    /// Configuration for error handling
    pub config: ErrorHandlerConfig,
    /// Collected errors during test execution
    pub errors: Vec<TestErrorContext>,
}

/// Configuration for the error handler
#[derive(Debug, Clone)]
pub struct ErrorHandlerConfig {
    /// Maximum number of retries for any test
    pub max_retries: usize,
    /// Whether to continue execution after critical errors
    pub continue_on_critical: bool,
    /// Whether to collect detailed diagnostics
    pub collect_diagnostics: bool,
    /// Whether to enable error pattern detection
    pub enable_pattern_detection: bool,
}

impl Default for ErrorHandlerConfig {
    fn default() -> Self {
        ErrorHandlerConfig {
            max_retries: 3,
            continue_on_critical: false,
            collect_diagnostics: true,
            enable_pattern_detection: true,
        }
    }
}

impl TestErrorHandler {
    /// Create a new error handler
    pub fn new(config: ErrorHandlerConfig) -> Self {
        TestErrorHandler {
            config,
            errors: Vec::new(),
        }
    }

    /// Handle a test error and determine next action
    pub fn handle_error(&mut self, error_context: TestErrorContext) -> ErrorHandlerAction {
        // Add to collection
        self.errors.push(error_context.clone());

        // Determine action based on error and configuration
        match error_context.severity {
            TestErrorSeverity::Critical if !self.config.continue_on_critical => {
                ErrorHandlerAction::StopExecution {
                    reason: "Critical error encountered".to_string(),
                }
            },
            _ => {
                if error_context.should_retry() {
                    ErrorHandlerAction::RetryTest {
                        test_name: error_context.error.test_name().to_string(),
                        max_attempts: self.config.max_retries,
                        backoff_ms: 1000,
                    }
                } else if error_context.should_skip() {
                    ErrorHandlerAction::SkipTest {
                        test_name: error_context.error.test_name().to_string(),
                        reason: "Error recovery strategy suggests skipping".to_string(),
                    }
                } else if error_context.should_fail_fast() {
                    ErrorHandlerAction::StopExecution {
                        reason: "Fail-fast error encountered".to_string(),
                    }
                } else {
                    ErrorHandlerAction::ContinueWithWarning {
                        warning: format!("Test error handled: {}", error_context.error),
                    }
                }
            }
        }
    }

    /// Generate error summary report
    pub fn generate_summary(&self) -> TestErrorSummary {
        let total_errors = self.errors.len();
        let mut severity_counts = HashMap::new();
        let mut error_patterns = HashMap::new();

        for error_ctx in &self.errors {
            *severity_counts.entry(error_ctx.severity).or_insert(0) += 1;
            
            // Simple pattern detection based on error type
            let pattern = match &error_ctx.error {
                TestError::Timeout { .. } => "timeout".to_string(),
                TestError::Panic { .. } => "panic".to_string(),
                TestError::Memory { .. } => "memory".to_string(),
                TestError::Integration { .. } => "integration".to_string(),
                _ => "other".to_string(),
            };
            *error_patterns.entry(pattern).or_insert(0) += 1;
        }

        TestErrorSummary {
            total_errors,
            severity_counts,
            error_patterns,
            most_common_error: self.find_most_common_error_type(),
            critical_errors: self.errors.iter()
                .filter(|e| e.severity == TestErrorSeverity::Critical)
                .count(),
        }
    }

    fn find_most_common_error_type(&self) -> Option<String> {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        
        for error_ctx in &self.errors {
            let error_type = match &error_ctx.error {
                TestError::Timeout { .. } => "Timeout",
                TestError::Panic { .. } => "Panic", 
                TestError::Memory { .. } => "Memory",
                TestError::ResourceExhaustion { .. } => "ResourceExhaustion",
                TestError::Integration { .. } => "Integration",
                TestError::PerformanceRegression { .. } => "PerformanceRegression",
                TestError::Hardware { .. } => "Hardware",
                TestError::Environment { .. } => "Environment",
                TestError::Validation { .. } => "Validation",
                TestError::General { .. } => "General",
            };
            *type_counts.entry(error_type.to_string()).or_insert(0) += 1;
        }

        type_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(error_type, _)| error_type)
    }
}

/// Actions that the error handler can recommend
#[derive(Debug, Clone)]
pub enum ErrorHandlerAction {
    /// Retry the test with specified parameters
    RetryTest {
        test_name: String,
        max_attempts: usize,
        backoff_ms: u64,
    },
    /// Skip the test with reason
    SkipTest {
        test_name: String,
        reason: String,
    },
    /// Continue execution with warning
    ContinueWithWarning {
        warning: String,
    },
    /// Stop all test execution
    StopExecution {
        reason: String,
    },
}

/// Summary of all errors encountered during testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestErrorSummary {
    /// Total number of errors
    pub total_errors: usize,
    /// Count of errors by severity
    pub severity_counts: HashMap<TestErrorSeverity, usize>,
    /// Common error patterns
    pub error_patterns: HashMap<String, usize>,
    /// Most frequently occurring error type
    pub most_common_error: Option<String>,
    /// Number of critical errors
    pub critical_errors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = TestError::timeout(
            "test_example".to_string(),
            Duration::from_secs(10),
            TestCategory::Unit,
        );
        
        assert_eq!(error.test_name(), "test_example");
        assert_eq!(error.severity(), TestErrorSeverity::Medium);
    }

    #[test]
    fn test_error_context() {
        let error = TestError::panic(
            "test_panic".to_string(), 
            "assertion failed".to_string()
        );
        
        let context = TestErrorContext::new(error)
            .with_metadata("test_file".to_string(), "test.rs".to_string());
        
        assert!(context.metadata.contains_key("test_file"));
        assert!(context.should_fail_fast());
    }

    #[test]
    fn test_error_handler() {
        let mut handler = TestErrorHandler::new(ErrorHandlerConfig::default());
        
        let error = TestError::timeout(
            "slow_test".to_string(),
            Duration::from_secs(30),
            TestCategory::Integration,
        );
        
        let context = TestErrorContext::new(error);
        let action = handler.handle_error(context);
        
        match action {
            ErrorHandlerAction::RetryTest { test_name, .. } => {
                assert_eq!(test_name, "slow_test");
            },
            _ => panic!("Expected retry action for timeout error"),
        }
    }
}
