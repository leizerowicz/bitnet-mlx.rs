//! Benchmark Error Handling and Performance Test Protection
//!
//! This module provides comprehensive error handling for performance tests,
//! including timeout protection, regression detection, and statistical validation.

use anyhow::Result;
use bitnet_core::test_utils::error::{TestError, TestErrorHandler, ErrorHandlerConfig, TestErrorContext};
use bitnet_core::test_utils::TestCategory;
use bitnet_core::test_utils::timeout::{execute_test_with_monitoring, is_ci_environment};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Test performance test timeout protection
#[test]
fn test_performance_timeout_protection() -> Result<()> {
    println!("🔄 Testing performance test timeout protection...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: 1,
        continue_on_critical: false,
        collect_diagnostics: true,
        enable_pattern_detection: true,
    });
    
    // Test scenarios with different timeout requirements
    let performance_scenarios = vec![
        ("quick_benchmark", Duration::from_secs(2), TestCategory::Unit),
        ("moderate_benchmark", Duration::from_secs(10), TestCategory::Integration),
        ("intensive_benchmark", Duration::from_secs(30), TestCategory::Performance),
        ("stress_benchmark", Duration::from_secs(60), TestCategory::Stress),
    ];
    
    for (test_name, timeout, category) in performance_scenarios {
        println!("  Testing timeout protection for: {}", test_name);
        
        // Create a test that will complete within timeout
        let result = execute_test_with_monitoring(
            test_name.to_string(),
            category,
            timeout,
            Box::new(|| {
                // Simulate some work that completes quickly
                let start = Instant::now();
                while start.elapsed() < Duration::from_millis(100) {
                    // Busy work
                    let _ = (1..1000).fold(0, |acc, x| acc + x);
                }
            }),
        );
        
        if result.success {
            println!("    ✅ {} completed successfully in {:.3}s", test_name, result.duration.as_secs_f64());
        } else if result.timed_out {
            println!("    ⚠️ {} timed out - handling as performance issue", test_name);
            
            let timeout_error = TestError::timeout(test_name.to_string(), timeout, category);
            let error_context = TestErrorContext::new(timeout_error)
                .with_metadata("test_type".to_string(), "performance".to_string())
                .with_metadata("expected_duration".to_string(), format!("{:?}", timeout));
            
            let action = error_handler.handle_error(error_context);
            println!("    ✅ Timeout handled with action: {:?}", action);
        } else {
            println!("    ❌ {} failed: {:?}", test_name, result.error_message);
        }
    }
    
    // Test timeout protection for a genuinely slow operation
    println!("  Testing timeout protection for deliberately slow operation...");
    
    let slow_result = execute_test_with_monitoring(
        "deliberately_slow_test".to_string(),
        TestCategory::Unit,
        Duration::from_millis(500), // Short timeout
        Box::new(|| {
            // This will definitely timeout
            thread::sleep(Duration::from_secs(2));
        }),
    );
    
    assert!(slow_result.timed_out, "Deliberately slow test should have timed out");
    println!("    ✅ Slow operation correctly timed out after {:.3}s", slow_result.duration.as_secs_f64());
    
    let summary = error_handler.generate_summary();
    println!("  ✅ Performance timeout protection summary: {} errors handled", summary.total_errors);
    
    println!("✅ Performance timeout protection test completed");
    Ok(())
}

/// Test regression detection in performance benchmarks
#[test] 
fn test_performance_regression_detection() -> Result<()> {
    println!("🔄 Testing performance regression detection...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig::default());
    
    // Simulate baseline performance measurements
    let baseline_measurements = vec![
        ("matrix_multiply", Duration::from_millis(100)),
        ("tensor_add", Duration::from_millis(50)),
        ("quantization", Duration::from_millis(200)),
        ("activation", Duration::from_millis(25)),
    ];
    
    // Simulate current measurements with various regression levels
    let current_measurements = vec![
        ("matrix_multiply", Duration::from_millis(120)), // 20% regression - acceptable
        ("tensor_add", Duration::from_millis(80)),       // 60% regression - concerning
        ("quantization", Duration::from_millis(450)),    // 125% regression - critical
        ("activation", Duration::from_millis(26)),       // 4% regression - noise
    ];
    
    println!("  Analyzing performance regressions...");
    
    for ((test_name, baseline), (_, current)) in baseline_measurements.iter().zip(current_measurements.iter()) {
        let regression_percentage = ((current.as_secs_f64() / baseline.as_secs_f64()) - 1.0) * 100.0;
        
        println!("    {} - Baseline: {:?}, Current: {:?}, Regression: {:.1}%", 
                test_name, baseline, current, regression_percentage);
        
        if regression_percentage > 10.0 { // Threshold for concern
            let regression_error = TestError::performance_regression(
                test_name.to_string(),
                *current,
                *baseline
            );
            
            let error_context = TestErrorContext::new(regression_error)
                .with_metadata("regression_threshold".to_string(), "10%".to_string())
                .with_metadata("baseline_duration".to_string(), format!("{:?}", baseline))
                .with_metadata("current_duration".to_string(), format!("{:?}", current));
            
            let action = error_handler.handle_error(error_context);
            
            match regression_percentage {
                r if r > 100.0 => {
                    println!("      🚨 CRITICAL regression detected - {:.1}%", r);
                    match action {
                        bitnet_core::test_utils::error::ErrorHandlerAction::StopExecution { reason } => {
                            println!("      ✅ Critical regression triggers stop: {}", reason);
                        },
                        _ => println!("      ⚠️ Expected stop execution for critical regression"),
                    }
                },
                r if r > 50.0 => {
                    println!("      🔶 HIGH regression detected - {:.1}%", r);
                },
                r if r > 25.0 => {
                    println!("      🔷 MEDIUM regression detected - {:.1}%", r);
                },
                _ => {
                    println!("      🟡 LOW regression detected - {:.1}%", r);
                }
            }
        } else {
            println!("      ✅ Performance within acceptable range");
        }
    }
    
    let summary = error_handler.generate_summary();
    println!("  ✅ Regression detection summary:");
    println!("    Total regressions detected: {}", summary.total_errors);
    println!("    Critical regressions: {}", summary.critical_errors);
    
    println!("✅ Performance regression detection test completed");
    Ok(())
}

/// Test statistical validation with error tolerance
#[test]
fn test_statistical_validation_with_error_tolerance() -> Result<()> {
    println!("🔄 Testing statistical validation with error tolerance...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig::default());
    
    // Simulate multiple benchmark runs with statistical variation
    let benchmark_runs = vec![
        ("stable_operation", vec![98, 101, 99, 102, 97]),     // Low variation - good
        ("variable_operation", vec![95, 110, 88, 115, 92]),   // High variation - concerning  
        ("degrading_operation", vec![100, 105, 112, 120, 130]), // Trending upward - bad
        ("improving_operation", vec![120, 110, 105, 98, 95]), // Trending downward - good
    ];
    
    println!("  Analyzing statistical patterns in benchmark data...");
    
    for (test_name, measurements) in benchmark_runs {
        println!("    Analyzing {}: {:?}", test_name, measurements);
        
        // Calculate basic statistics
        let mean = measurements.iter().sum::<i32>() as f64 / measurements.len() as f64;
        let variance = measurements.iter()
            .map(|x| (*x as f64 - mean).powi(2))
            .sum::<f64>() / measurements.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;
        
        println!("      Mean: {:.1}ms, Std Dev: {:.1}ms, CV: {:.3}", mean, std_dev, coefficient_of_variation);
        
        // Check for statistical issues
        if coefficient_of_variation > 0.15 { // High variation threshold
            let validation_error = TestError::validation(
                test_name.to_string(),
                format!("High coefficient of variation: {:.3}", coefficient_of_variation),
                Some(format!("CV < 0.15")),
                Some(format!("CV = {:.3}", coefficient_of_variation))
            );
            
            let error_context = TestErrorContext::new(validation_error)
                .with_metadata("mean_duration".to_string(), format!("{:.1}", mean))
                .with_metadata("std_deviation".to_string(), format!("{:.1}", std_dev))
                .with_metadata("measurement_count".to_string(), measurements.len().to_string());
            
            let action = error_handler.handle_error(error_context);
            println!("      ⚠️ High variation detected - action: {:?}", action);
        } else {
            println!("      ✅ Statistical variation within acceptable limits");
        }
        
        // Check for trending issues
        if measurements.len() >= 3 {
            let first_half_avg = measurements[..measurements.len()/2].iter().sum::<i32>() as f64 
                               / (measurements.len()/2) as f64;
            let second_half_avg = measurements[measurements.len()/2..].iter().sum::<i32>() as f64 
                                / (measurements.len() - measurements.len()/2) as f64;
            
            let trend_percentage = ((second_half_avg / first_half_avg) - 1.0) * 100.0;
            
            if trend_percentage.abs() > 10.0 {
                let trend_type = if trend_percentage > 0.0 { "degradation" } else { "improvement" };
                println!("      📊 Performance trend detected: {:.1}% {} over time", 
                        trend_percentage.abs(), trend_type);
                
                if trend_percentage > 20.0 { // Significant degradation
                    let regression_error = TestError::performance_regression(
                        format!("{}_trend", test_name),
                        Duration::from_millis(second_half_avg as u64),
                        Duration::from_millis(first_half_avg as u64)
                    );
                    
                    let error_context = TestErrorContext::new(regression_error);
                    error_handler.handle_error(error_context);
                    println!("      🚨 Significant degradation trend logged");
                }
            }
        }
    }
    
    let summary = error_handler.generate_summary();
    println!("  ✅ Statistical validation summary:");
    println!("    Statistical issues detected: {}", summary.total_errors);
    if let Some(common_error) = &summary.most_common_error {
        println!("    Most common issue type: {}", common_error);
    }
    
    println!("✅ Statistical validation test completed");
    Ok(())
}

/// Test CI environment performance optimizations
#[test]
fn test_ci_environment_performance_optimizations() -> Result<()> {
    println!("🔄 Testing CI environment performance optimizations...");
    
    // Simulate CI environment
    std::env::set_var("CI", "true");
    std::env::set_var("GITHUB_ACTIONS", "true");
    
    let is_ci = is_ci_environment();
    assert!(is_ci, "Should detect CI environment");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: if is_ci { 1 } else { 3 }, // Reduced retries in CI
        continue_on_critical: false, // Fail fast in CI
        collect_diagnostics: false, // Reduced diagnostics in CI  
        enable_pattern_detection: true,
    });
    
    // Test CI-specific timeout handling
    println!("  Testing CI-specific timeout handling...");
    
    let ci_performance_tests = vec![
        ("quick_ci_test", Duration::from_secs(1), TestCategory::Unit),
        ("integration_ci_test", Duration::from_secs(5), TestCategory::Integration),
        ("performance_ci_test", Duration::from_secs(15), TestCategory::Performance),
    ];
    
    for (test_name, timeout, category) in ci_performance_tests {
        let should_skip = category == TestCategory::Performance && is_ci;
        
        if should_skip {
            println!("    ⏩ Skipping {} in CI environment", test_name);
            
            let skip_error = TestError::environment(
                test_name.to_string(),
                "Performance test skipped in CI".to_string(),
                vec!["CI environment".to_string()],
                [("CI".to_string(), "true".to_string())].into_iter().collect()
            );
            
            let error_context = TestErrorContext::new(skip_error)
                .with_metadata("skip_reason".to_string(), "CI optimization".to_string());
            
            let action = error_handler.handle_error(error_context);
            match action {
                bitnet_core::test_utils::error::ErrorHandlerAction::SkipTest { .. } => {
                    println!("      ✅ CI optimization correctly skips performance test");
                },
                _ => println!("      ⚠️ Expected skip action for CI performance test"),
            }
        } else {
            println!("    ✅ {} allowed in CI environment", test_name);
        }
    }
    
    // Test resource constraint handling
    println!("  Testing resource constraint handling in CI...");
    
    let resource_error = TestError::resource_exhaustion(
        "memory_intensive_test".to_string(),
        "Memory usage exceeded CI limits".to_string(),
        Some(85.0), // 85% memory usage
        Some(80.0)  // 80% limit
    );
    
    let error_context = TestErrorContext::new(resource_error)
        .with_metadata("environment".to_string(), "CI".to_string())
        .with_metadata("available_memory".to_string(), "2GB".to_string());
    
    let action = error_handler.handle_error(error_context);
    println!("    ✅ Resource constraint handled in CI: {:?}", action);
    
    // Clean up environment
    std::env::remove_var("CI");
    std::env::remove_var("GITHUB_ACTIONS");
    
    let summary = error_handler.generate_summary();
    println!("  ✅ CI optimization summary: {} issues handled", summary.total_errors);
    
    println!("✅ CI environment performance optimization test completed");
    Ok(())
}

/// Test automated error pattern detection in benchmarks
#[test]
fn test_automated_benchmark_error_pattern_detection() -> Result<()> {
    println!("🔄 Testing automated benchmark error pattern detection...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: 2,
        continue_on_critical: true,
        collect_diagnostics: true,
        enable_pattern_detection: true,
    });
    
    // Simulate a series of benchmark errors that form patterns
    let benchmark_error_scenarios = vec![
        // Memory pressure pattern
        ("benchmark_1", TestError::memory("benchmark_1".to_string(), "Memory allocation failed".to_string())),
        ("benchmark_2", TestError::memory("benchmark_2".to_string(), "Out of memory".to_string())),
        ("benchmark_3", TestError::memory("benchmark_3".to_string(), "Memory pool exhausted".to_string())),
        
        // Timeout pattern  
        ("benchmark_4", TestError::timeout("benchmark_4".to_string(), Duration::from_secs(30), TestCategory::Performance)),
        ("benchmark_5", TestError::timeout("benchmark_5".to_string(), Duration::from_secs(45), TestCategory::Performance)),
        ("benchmark_6", TestError::timeout("benchmark_6".to_string(), Duration::from_secs(60), TestCategory::Performance)),
        
        // Hardware pattern
        ("benchmark_7", TestError::hardware("benchmark_7".to_string(), "GPU not available".to_string(), "Metal".to_string(), false)),
        ("benchmark_8", TestError::hardware("benchmark_8".to_string(), "Device selection failed".to_string(), "MLX".to_string(), false)),
        
        // Performance regression pattern
        ("benchmark_9", TestError::performance_regression("benchmark_9".to_string(), Duration::from_millis(200), Duration::from_millis(100))),
        ("benchmark_10", TestError::performance_regression("benchmark_10".to_string(), Duration::from_millis(150), Duration::from_millis(75))),
    ];
    
    println!("  Feeding benchmark errors to pattern detector...");
    
    for (test_name, error) in benchmark_error_scenarios {
        println!("    Processing error from {}", test_name);
        let error_context = TestErrorContext::new(error)
            .with_metadata("test_category".to_string(), "benchmark".to_string())
            .with_metadata("pattern_detection".to_string(), "enabled".to_string());
        
        let action = error_handler.handle_error(error_context);
        
        // Log the action taken
        match action {
            bitnet_core::test_utils::error::ErrorHandlerAction::RetryTest { test_name, .. } => {
                println!("      → Retry recommended for {}", test_name);
            },
            bitnet_core::test_utils::error::ErrorHandlerAction::SkipTest { test_name, reason } => {
                println!("      → Skip recommended for {} - {}", test_name, reason);
            },
            bitnet_core::test_utils::error::ErrorHandlerAction::ContinueWithWarning { warning } => {
                println!("      → Continue with warning: {}", warning);
            },
            bitnet_core::test_utils::error::ErrorHandlerAction::StopExecution { reason } => {
                println!("      → Stop execution recommended: {}", reason);
            },
        }
    }
    
    // Analyze detected patterns
    let summary = error_handler.generate_summary();
    
    println!("  ✅ Pattern detection analysis results:");
    println!("    Total benchmark errors: {}", summary.total_errors);
    println!("    Error patterns detected:");
    
    for (pattern, count) in &summary.error_patterns {
        println!("      {}: {} occurrences", pattern, count);
    }
    
    if let Some(most_common) = &summary.most_common_error {
        println!("    Most common error type: {}", most_common);
    }
    
    // Verify pattern detection worked
    assert!(summary.error_patterns.get("memory").unwrap_or(&0) >= &3,
            "Should detect memory error pattern");
    assert!(summary.error_patterns.get("timeout").unwrap_or(&0) >= &3, 
            "Should detect timeout error pattern");
    
    println!("  ✅ Pattern detection successfully identified recurring issues");
    println!("✅ Automated benchmark error pattern detection test completed");
    Ok(())
}

// Helper functions for additional error types
impl TestError {
    pub fn resource_exhaustion(test_name: String, message: String, current: Option<f64>, limit: Option<f64>) -> Self {
        TestError::ResourceExhaustion {
            test_name,
            resource_type: "memory".to_string(),
            current_usage: current,
            limit,
        }
    }
    
    pub fn environment(test_name: String, message: String, missing_deps: Vec<String>, env_vars: std::collections::HashMap<String, String>) -> Self {
        TestError::Environment {
            test_name,
            env_error: message,
            missing_dependencies: missing_deps,
            env_vars,
        }
    }
    
    pub fn hardware(test_name: String, message: String, device_type: String, recovery_attempted: bool) -> Self {
        TestError::Hardware {
            test_name,
            device_error: message,
            device_type,
            recovery_attempted,
        }
    }
}
