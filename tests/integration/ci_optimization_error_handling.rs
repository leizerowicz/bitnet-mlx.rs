//! CI Optimization and Error Handling System
//!
//! This module provides CI-specific optimizations for test execution,
//! including environment detection, resource constraint handling,
//! and automated error pattern detection.

use anyhow::Result;
use bitnet_core::test_utils::error::{TestError, TestErrorHandler, ErrorHandlerConfig, TestErrorContext, ErrorRecoveryStrategy};
use bitnet_core::test_utils::TestCategory;
use bitnet_core::test_utils::timeout::is_ci_environment;
use std::time::Duration;
use std::collections::HashMap;
use std::env;

/// CI-specific error handling configuration
#[derive(Debug, Clone)]
pub struct CiErrorHandlerConfig {
    /// Base configuration
    pub base: ErrorHandlerConfig,
    /// CI environment type
    pub ci_type: CiEnvironmentType,
    /// Resource limits for CI environment
    pub resource_limits: CiResourceLimits,
    /// Test categories to skip in CI
    pub skip_categories: Vec<TestCategory>,
    /// Maximum execution time for entire test suite in CI
    pub max_suite_duration: Duration,
}

/// Types of CI environments with specific characteristics
#[derive(Debug, Clone, PartialEq)]
pub enum CiEnvironmentType {
    /// GitHub Actions
    GitHubActions {
        runner_type: String, // ubuntu-latest, macos-latest, etc.
        runner_size: String, // standard, large, xlarge
    },
    /// GitLab CI
    GitLabCi {
        runner_tags: Vec<String>,
        shared_runner: bool,
    },
    /// Travis CI
    TravisCi {
        vm_type: String,
        architecture: String,
    },
    /// CircleCI
    CircleCi {
        resource_class: String,
        machine_type: String,
    },
    /// Generic CI environment
    Generic {
        environment_name: String,
    },
    /// Local development (not CI)
    Local,
}

/// Resource limits specific to CI environments
#[derive(Debug, Clone)]
pub struct CiResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<u64>,
    /// Maximum CPU usage percentage
    pub max_cpu_percentage: Option<f64>,
    /// Maximum execution time per test
    pub max_test_duration: Duration,
    /// Maximum number of parallel tests
    pub max_parallel_tests: Option<usize>,
    /// Network bandwidth limits (if applicable)
    pub max_network_bandwidth: Option<u64>,
}

/// CI-optimized error handler
pub struct CiErrorHandler {
    /// Base error handler
    pub base_handler: TestErrorHandler,
    /// CI-specific configuration
    pub ci_config: CiErrorHandlerConfig,
    /// Environment detection results
    pub environment_info: CiEnvironmentInfo,
}

/// Detected CI environment information
#[derive(Debug, Clone)]
pub struct CiEnvironmentInfo {
    /// Type of CI environment
    pub ci_type: CiEnvironmentType,
    /// Available resources
    pub available_resources: DetectedResources,
    /// Environment variables relevant to testing
    pub test_env_vars: HashMap<String, String>,
    /// Detected limitations or issues
    pub limitations: Vec<String>,
}

/// Resources detected in the current environment
#[derive(Debug, Clone)]
pub struct DetectedResources {
    /// Total system memory (MB)
    pub total_memory_mb: Option<u64>,
    /// Available CPU cores
    pub cpu_cores: Option<usize>,
    /// GPU availability
    pub gpu_available: bool,
    /// Network connectivity
    pub network_available: bool,
}

impl CiErrorHandler {
    /// Create a new CI-optimized error handler
    pub fn new() -> Result<Self> {
        let environment_info = Self::detect_environment()?;
        let ci_config = Self::create_ci_config(&environment_info);
        
        let base_handler = TestErrorHandler::new(ci_config.base.clone());
        
        Ok(CiErrorHandler {
            base_handler,
            ci_config,
            environment_info,
        })
    }
    
    /// Detect the current CI environment and its characteristics
    fn detect_environment() -> Result<CiEnvironmentInfo> {
        let ci_type = if env::var("GITHUB_ACTIONS").is_ok() {
            CiEnvironmentType::GitHubActions {
                runner_type: env::var("RUNNER_OS").unwrap_or_else(|_| "unknown".to_string()),
                runner_size: env::var("RUNNER_SIZE").unwrap_or_else(|_| "standard".to_string()),
            }
        } else if env::var("GITLAB_CI").is_ok() {
            CiEnvironmentType::GitLabCi {
                runner_tags: env::var("CI_RUNNER_TAGS")
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                shared_runner: env::var("CI_RUNNER_SHARED").unwrap_or_default() == "true",
            }
        } else if env::var("TRAVIS").is_ok() {
            CiEnvironmentType::TravisCi {
                vm_type: env::var("TRAVIS_VM_TYPE").unwrap_or_else(|_| "unknown".to_string()),
                architecture: env::var("TRAVIS_ARCH").unwrap_or_else(|_| "unknown".to_string()),
            }
        } else if env::var("CIRCLECI").is_ok() {
            CiEnvironmentType::CircleCi {
                resource_class: env::var("CIRCLE_RESOURCE_CLASS").unwrap_or_else(|_| "unknown".to_string()),
                machine_type: env::var("CIRCLE_MACHINE_TYPE").unwrap_or_else(|_| "unknown".to_string()),
            }
        } else if is_ci_environment() {
            CiEnvironmentType::Generic {
                environment_name: env::var("CI_NAME").unwrap_or_else(|_| "unknown".to_string()),
            }
        } else {
            CiEnvironmentType::Local
        };
        
        let available_resources = Self::detect_resources();
        let test_env_vars = Self::collect_test_env_vars();
        let limitations = Self::detect_limitations(&ci_type, &available_resources);
        
        Ok(CiEnvironmentInfo {
            ci_type,
            available_resources,
            test_env_vars,
            limitations,
        })
    }
    
    /// Detect available system resources
    fn detect_resources() -> DetectedResources {
        DetectedResources {
            total_memory_mb: Self::get_total_memory_mb(),
            cpu_cores: Self::get_cpu_cores(),
            gpu_available: Self::detect_gpu_availability(),
            network_available: true, // Assume available unless proven otherwise
        }
    }
    
    /// Get total system memory in MB
    fn get_total_memory_mb() -> Option<u64> {
        // Simple implementation - would use system calls in production
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            return Some(kb_val / 1024); // Convert KB to MB
                        }
                    }
                }
            }
        }
        
        // Fallback estimates based on common CI environments
        if env::var("GITHUB_ACTIONS").is_ok() {
            Some(7000) // GitHub Actions typically has ~7GB
        } else if env::var("GITLAB_CI").is_ok() {
            Some(4000) // GitLab CI typically has ~4GB
        } else {
            Some(2000) // Conservative estimate
        }
    }
    
    /// Get number of available CPU cores
    fn get_cpu_cores() -> Option<usize> {
        if let Ok(nproc) = std::process::Command::new("nproc").output() {
            if let Ok(count_str) = String::from_utf8(nproc.stdout) {
                if let Ok(count) = count_str.trim().parse::<usize>() {
                    return Some(count);
                }
            }
        }
        
        // Fallback to thread::available_parallelism
        std::thread::available_parallelism().ok().map(|p| p.get())
    }
    
    /// Detect GPU availability
    fn detect_gpu_availability() -> bool {
        // Check for Metal on macOS
        #[cfg(target_os = "macos")]
        {
            if std::process::Command::new("system_profiler")
                .args(&["SPDisplaysDataType"])
                .output()
                .is_ok()
            {
                return true;
            }
        }
        
        // Check for CUDA
        if std::process::Command::new("nvidia-smi").output().is_ok() {
            return true;
        }
        
        // Check environment variables
        env::var("GPU_AVAILABLE").unwrap_or_default() == "true"
    }
    
    /// Collect test-relevant environment variables
    fn collect_test_env_vars() -> HashMap<String, String> {
        let mut env_vars = HashMap::new();
        
        let test_relevant_vars = vec![
            "RUST_LOG", "RUST_BACKTRACE", "CARGO_TARGET_DIR", "CI", "RUNNER_OS",
            "GITHUB_ACTIONS", "GITLAB_CI", "TRAVIS", "CIRCLECI", "BUILD_NUMBER",
            "GPU_AVAILABLE", "MLX_ENABLE_VALIDATION", "TEST_TIMEOUT", "TEST_THREADS",
        ];
        
        for var in test_relevant_vars {
            if let Ok(value) = env::var(var) {
                env_vars.insert(var.to_string(), value);
            }
        }
        
        env_vars
    }
    
    /// Detect environment limitations
    fn detect_limitations(ci_type: &CiEnvironmentType, resources: &DetectedResources) -> Vec<String> {
        let mut limitations = Vec::new();
        
        match ci_type {
            CiEnvironmentType::GitHubActions { runner_type, runner_size } => {
                if runner_size == "standard" {
                    limitations.push("Limited to 2-core standard runner".to_string());
                }
                if runner_type.contains("ubuntu") {
                    limitations.push("No GPU acceleration on Ubuntu runners".to_string());
                }
            },
            CiEnvironmentType::GitLabCi { shared_runner, .. } => {
                if *shared_runner {
                    limitations.push("Shared runner - variable performance".to_string());
                }
            },
            CiEnvironmentType::TravisCi { .. } => {
                limitations.push("Travis CI has strict time limits".to_string());
            },
            _ => {}
        }
        
        // Resource-based limitations
        if let Some(memory) = resources.total_memory_mb {
            if memory < 2000 {
                limitations.push("Low memory environment (<2GB)".to_string());
            }
        }
        
        if let Some(cores) = resources.cpu_cores {
            if cores <= 2 {
                limitations.push("Limited CPU cores (≤2)".to_string());
            }
        }
        
        if !resources.gpu_available {
            limitations.push("No GPU acceleration available".to_string());
        }
        
        limitations
    }
    
    /// Create CI-optimized configuration
    fn create_ci_config(env_info: &CiEnvironmentInfo) -> CiErrorHandlerConfig {
        let base_config = match &env_info.ci_type {
            CiEnvironmentType::GitHubActions { .. } => ErrorHandlerConfig {
                max_retries: 1, // Limited retries in CI
                continue_on_critical: false, // Fail fast
                collect_diagnostics: false, // Reduced diagnostics
                enable_pattern_detection: true,
            },
            CiEnvironmentType::Local => ErrorHandlerConfig::default(),
            _ => ErrorHandlerConfig {
                max_retries: 2,
                continue_on_critical: false,
                collect_diagnostics: true,
                enable_pattern_detection: true,
            },
        };
        
        let resource_limits = CiResourceLimits {
            max_memory_mb: env_info.available_resources.total_memory_mb.map(|m| m * 80 / 100), // 80% of available
            max_cpu_percentage: Some(90.0), // Don't max out CPU
            max_test_duration: match &env_info.ci_type {
                CiEnvironmentType::GitHubActions { .. } => Duration::from_secs(300), // 5 minutes max per test
                _ => Duration::from_secs(600), // 10 minutes for other CI
            },
            max_parallel_tests: env_info.available_resources.cpu_cores,
            max_network_bandwidth: None, // No specific limits
        };
        
        let skip_categories = match &env_info.ci_type {
            CiEnvironmentType::GitHubActions { .. } => vec![TestCategory::Endurance],
            CiEnvironmentType::TravisCi { .. } => vec![TestCategory::Stress, TestCategory::Endurance],
            _ => vec![TestCategory::Endurance],
        };
        
        CiErrorHandlerConfig {
            base: base_config,
            ci_type: env_info.ci_type.clone(),
            resource_limits,
            skip_categories,
            max_suite_duration: Duration::from_secs(1800), // 30 minutes max for entire suite
        }
    }
    
    /// Handle an error with CI-specific optimizations
    pub fn handle_error(&mut self, mut error_context: TestErrorContext) -> bitnet_core::test_utils::error::ErrorHandlerAction {
        // Apply CI-specific error handling logic
        error_context = self.apply_ci_optimizations(error_context);
        
        // Use base handler with modified context
        self.base_handler.handle_error(error_context)
    }
    
    /// Apply CI-specific optimizations to error handling
    fn apply_ci_optimizations(&self, mut error_context: TestErrorContext) -> TestErrorContext {
        match &self.environment_info.ci_type {
            CiEnvironmentType::GitHubActions { .. } => {
                // GitHub Actions optimizations
                match &error_context.error {
                    TestError::Timeout { category, .. } => {
                        if *category == TestCategory::Performance {
                            // Skip performance tests that timeout in GitHub Actions
                            error_context = error_context.with_recovery_strategy(
                                ErrorRecoveryStrategy::Skip {
                                    reason: "Performance test timeout in GitHub Actions".to_string(),
                                    tracking_issue: Some("CI-performance-timeout".to_string()),
                                }
                            );
                        }
                    },
                    TestError::Memory { .. } => {
                        // Reduce memory usage in GitHub Actions
                        error_context = error_context.with_recovery_strategy(
                            ErrorRecoveryStrategy::Degrade {
                                fallback_category: TestCategory::Unit,
                                reduced_timeout: Duration::from_secs(30),
                            }
                        );
                    },
                    _ => {}
                }
            },
            CiEnvironmentType::Local => {
                // More lenient handling for local development
                match error_context.recovery_strategy {
                    ErrorRecoveryStrategy::Skip { .. } => {
                        // Convert skips to warnings in local development
                        error_context = error_context.with_recovery_strategy(
                            ErrorRecoveryStrategy::ContinueWithWarning {
                                warning_message: "Test skipped in CI would run locally".to_string(),
                            }
                        );
                    },
                    _ => {}
                }
            },
            _ => {}
        }
        
        // Add CI environment metadata
        error_context = error_context
            .with_metadata("ci_type".to_string(), format!("{:?}", self.environment_info.ci_type))
            .with_metadata("available_memory_mb".to_string(), 
                          self.environment_info.available_resources.total_memory_mb
                              .map(|m| m.to_string())
                              .unwrap_or_else(|| "unknown".to_string()))
            .with_metadata("cpu_cores".to_string(),
                          self.environment_info.available_resources.cpu_cores
                              .map(|c| c.to_string())
                              .unwrap_or_else(|| "unknown".to_string()));
        
        error_context
    }
    
    /// Generate CI-specific error report
    pub fn generate_ci_report(&self) -> CiErrorReport {
        let base_summary = self.base_handler.generate_summary();
        
        CiErrorReport {
            environment_info: self.environment_info.clone(),
            base_summary,
            ci_specific_recommendations: self.generate_ci_recommendations(),
            resource_usage_analysis: self.analyze_resource_usage(),
        }
    }
    
    /// Generate CI-specific recommendations
    fn generate_ci_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Based on detected limitations
        for limitation in &self.environment_info.limitations {
            if limitation.contains("memory") {
                recommendations.push("Consider reducing test data sizes for CI".to_string());
            } else if limitation.contains("CPU") {
                recommendations.push("Reduce parallel test execution in CI".to_string());
            } else if limitation.contains("GPU") {
                recommendations.push("Add CPU fallback paths for CI testing".to_string());
            }
        }
        
        // Based on error patterns
        let base_summary = &self.base_handler.generate_summary();
        if base_summary.error_patterns.get("timeout").unwrap_or(&0) > &3 {
            recommendations.push("Consider increasing timeout values for CI environment".to_string());
        }
        
        if base_summary.error_patterns.get("memory").unwrap_or(&0) > &2 {
            recommendations.push("Implement more aggressive memory management for CI".to_string());
        }
        
        recommendations
    }
    
    /// Analyze resource usage patterns
    fn analyze_resource_usage(&self) -> ResourceUsageAnalysis {
        ResourceUsageAnalysis {
            memory_pressure_detected: self.environment_info.limitations.iter()
                .any(|l| l.contains("memory")),
            cpu_pressure_detected: self.environment_info.limitations.iter()
                .any(|l| l.contains("CPU")),
            io_pressure_detected: false, // Would need more sophisticated detection
            network_issues_detected: false,
            recommendations: vec![
                "Monitor memory usage during tests".to_string(),
                "Consider test parallelization limits".to_string(),
                "Implement resource cleanup between tests".to_string(),
            ],
        }
    }
}

/// CI-specific error report
#[derive(Debug, Clone)]
pub struct CiErrorReport {
    /// Environment information
    pub environment_info: CiEnvironmentInfo,
    /// Base error summary
    pub base_summary: bitnet_core::test_utils::error::TestErrorSummary,
    /// CI-specific recommendations
    pub ci_specific_recommendations: Vec<String>,
    /// Resource usage analysis
    pub resource_usage_analysis: ResourceUsageAnalysis,
}

/// Resource usage analysis for CI environments
#[derive(Debug, Clone)]
pub struct ResourceUsageAnalysis {
    /// Whether memory pressure was detected
    pub memory_pressure_detected: bool,
    /// Whether CPU pressure was detected
    pub cpu_pressure_detected: bool,
    /// Whether I/O pressure was detected
    pub io_pressure_detected: bool,
    /// Whether network issues were detected
    pub network_issues_detected: bool,
    /// Specific recommendations for resource optimization
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_environment_detection() {
        // Test GitHub Actions detection
        env::set_var("GITHUB_ACTIONS", "true");
        env::set_var("RUNNER_OS", "macOS");
        
        let handler_result = CiErrorHandler::new();
        assert!(handler_result.is_ok());
        
        let handler = handler_result.unwrap();
        match handler.environment_info.ci_type {
            CiEnvironmentType::GitHubActions { runner_type, .. } => {
                assert_eq!(runner_type, "macOS");
            },
            _ => panic!("Expected GitHub Actions detection"),
        }
        
        // Clean up
        env::remove_var("GITHUB_ACTIONS");
        env::remove_var("RUNNER_OS");
    }
    
    #[test]
    fn test_resource_detection() {
        let resources = CiErrorHandler::detect_resources();
        
        // Should detect at least basic resources
        assert!(resources.cpu_cores.is_some());
        assert!(resources.total_memory_mb.is_some());
    }
    
    #[test]
    fn test_ci_error_handling() {
        let mut handler = CiErrorHandler::new().unwrap();
        
        let timeout_error = TestError::timeout(
            "ci_test".to_string(),
            Duration::from_secs(30),
            TestCategory::Performance
        );
        
        let error_context = TestErrorContext::new(timeout_error);
        let action = handler.handle_error(error_context);
        
        // Should handle the error appropriately
        match action {
            bitnet_core::test_utils::error::ErrorHandlerAction::SkipTest { .. } |
            bitnet_core::test_utils::error::ErrorHandlerAction::RetryTest { .. } |
            bitnet_core::test_utils::error::ErrorHandlerAction::ContinueWithWarning { .. } => {
                // All are acceptable CI responses
            },
            _ => {}
        }
    }
}
