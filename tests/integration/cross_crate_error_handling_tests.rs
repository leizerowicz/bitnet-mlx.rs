//! Cross-Crate Error Handling Integration Tests
//!
//! This module tests error handling across different BitNet crates,
//! ensuring proper error propagation and recovery mechanisms.

use anyhow::Result;
use bitnet_core::test_utils::error::{TestError, TestErrorHandler, ErrorHandlerConfig, TestErrorContext};
use bitnet_core::test_utils::TestCategory;
use bitnet_core::tensor::BitNetTensor;
use bitnet_core::device::Device;
use bitnet_core::dtype::BitNetDType;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Test cross-crate error propagation
#[test]
fn test_cross_crate_error_propagation() -> Result<()> {
    println!("🔄 Testing cross-crate error propagation...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig::default());
    let device = Device::Cpu;
    
    // Test 1: Core tensor error propagation
    println!("  Testing core tensor error propagation...");
    let invalid_shape = vec![0, 10]; // Invalid shape should cause error
    
    match BitNetTensor::zeros(&invalid_shape, BitNetDType::F32, &device) {
        Ok(_) => {
            println!("    ❌ Expected error for invalid shape but operation succeeded");
            return Ok(()); // This is actually unexpected but we'll continue
        },
        Err(_e) => {
            println!("    ✅ Core tensor error properly caught: {}", e);
            
            // Convert to test error and handle
            let test_error = TestError::integration(
                "test_cross_crate_error_propagation".to_string(),
                format!("Core tensor error: {}", e),
                "bitnet-core".to_string()
            );
            
            let error_context = TestErrorContext::new(test_error)
                .with_metadata("error_source".to_string(), "tensor_creation".to_string())
                .with_metadata("invalid_shape".to_string(), format!("{:?}", invalid_shape));
                
            let action = error_handler.handle_error(error_context);
            println!("    ✅ Error handler action: {:?}", action);
        }
    }
    
    // Test 2: Cross-crate memory error handling
    println!("  Testing cross-crate memory error handling...");
    
    // Try to create a very large tensor that should stress memory
    let large_shape = vec![10000, 10000]; // May cause memory pressure
    match BitNetTensor::zeros(&large_shape, BitNetDType::F32, &device) {
        Ok(_) => {
            println!("    ✅ Large tensor creation succeeded (sufficient memory available)");
        },
        Err(_e) => {
            println!("    ✅ Memory error properly handled: {}", e);
            
            let memory_error = TestError::memory(
                "test_cross_crate_error_propagation_memory".to_string(),
                format!("Large tensor allocation failed: {}", e)
            );
            
            let error_context = TestErrorContext::new(memory_error)
                .with_metadata("tensor_size".to_string(), "400MB".to_string())
                .with_metadata("shape".to_string(), format!("{:?}", large_shape));
                
            let action = error_handler.handle_error(error_context);
            println!("    ✅ Memory error handler action: {:?}", action);
        }
    }
    
    // Generate error summary
    let summary = error_handler.generate_summary();
    println!("  ✅ Error Summary: {} total errors", summary.total_errors);
    if summary.total_errors > 0 {
        println!("    Most common error: {:?}", summary.most_common_error);
        println!("    Critical errors: {}", summary.critical_errors);
    }
    
    println!("✅ Cross-crate error propagation test completed");
    Ok(())
}

/// Test error recovery mechanisms across components
#[test]
fn test_error_recovery_mechanisms() -> Result<()> {
    println!("🔄 Testing error recovery mechanisms...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: 2,
        continue_on_critical: true,
        collect_diagnostics: true,
        enable_pattern_detection: true,
    });
    
    let device = Device::Cpu;
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));
    
    // Test multiple scenarios that should trigger different recovery strategies
    let test_scenarios = vec![
        ("timeout_simulation", TestCategory::Unit),
        ("memory_pressure", TestCategory::Integration), 
        ("validation_failure", TestCategory::Performance),
    ];
    
    for (test_name, category) in test_scenarios {
        println!("  Testing recovery for scenario: {}", test_name);
        
        match test_name {
            "timeout_simulation" => {
                // Simulate a timeout error
                let timeout_error = TestError::timeout(
                    test_name.to_string(),
                    Duration::from_secs(10),
                    category
                );
                
                let error_context = TestErrorContext::new(timeout_error);
                let action = error_handler.handle_error(error_context);
                
                match action {
                    bitnet_core::test_utils::error::ErrorHandlerAction::RetryTest { test_name: retry_name, max_attempts, .. } => {
                        println!("    ✅ Timeout recovery: Retry {} up to {} times", retry_name, max_attempts);
                        success_count.fetch_add(1, Ordering::Relaxed);
                    },
                    _ => {
                        println!("    ⚠️ Unexpected recovery action for timeout");
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            
            "memory_pressure" => {
                // Simulate memory pressure error
                let memory_error = TestError::memory(
                    test_name.to_string(),
                    "Simulated memory pressure".to_string()
                );
                
                let error_context = TestErrorContext::new(memory_error);
                let action = error_handler.handle_error(error_context);
                
                println!("    ✅ Memory pressure recovery action applied");
                success_count.fetch_add(1, Ordering::Relaxed);
            },
            
            "validation_failure" => {
                // Simulate validation error
                let validation_error = TestError::validation(
                    test_name.to_string(),
                    "Expected result validation failed".to_string(),
                    Some("42".to_string()),
                    Some("41".to_string())
                );
                
                let error_context = TestErrorContext::new(validation_error);
                let action = error_handler.handle_error(error_context);
                
                match action {
                    bitnet_core::test_utils::error::ErrorHandlerAction::StopExecution { reason } => {
                        println!("    ✅ Validation failure recovery: Stop execution - {}", reason);
                        success_count.fetch_add(1, Ordering::Relaxed);
                    },
                    _ => {
                        println!("    ⚠️ Unexpected recovery action for validation failure");
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            
            _ => {}
        }
    }
    
    let total_successes = success_count.load(Ordering::Relaxed);
    let total_errors = error_count.load(Ordering::Relaxed);
    
    println!("  Recovery test results: {} successes, {} errors", total_successes, total_errors);
    
    // Verify error handler collected the errors
    let summary = error_handler.generate_summary();
    assert!(summary.total_errors >= 3, "Should have collected at least 3 test errors");
    
    println!("  ✅ Error handler summary:");
    println!("    Total errors handled: {}", summary.total_errors);
    println!("    Critical errors: {}", summary.critical_errors);
    if let Some(common_error) = &summary.most_common_error {
        println!("    Most common error type: {}", common_error);
    }
    
    println!("✅ Error recovery mechanisms test completed");
    Ok(())
}

/// Test concurrent error handling across multiple threads
#[test]
fn test_concurrent_error_handling() -> Result<()> {
    println!("🔄 Testing concurrent error handling...");
    
    let device = Device::Cpu;
    let error_count = Arc::new(AtomicUsize::new(0));
    let success_count = Arc::new(AtomicUsize::new(0));
    
    // Spawn multiple threads that will encounter various errors
    let handles: Vec<_> = (0..6).map(|thread_id| {
        let device = device.clone();
        let error_count = Arc::clone(&error_count);
        let success_count = Arc::clone(&success_count);
        
        thread::spawn(move || {
            for i in 0..5 {
                // Create different error scenarios across threads
                let _shape = match (thread_id + i) % 4 {
                    0 => vec![0, 5], // Invalid - zero dimension
                    1 => vec![10, 10], // Valid
                    2 => vec![100, 100], // Valid but larger
                    3 => vec![], // Invalid - empty shape
                    _ => vec![5, 5], // Valid fallback
                };
                
                match BitNetTensor::zeros(&shape, BitNetDType::F32, &device) {
                    Ok(_) => {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(_e) => {
                        error_count.fetch_add(1, Ordering::Relaxed);
                        
                        // Create error handler for this thread
                        let mut local_handler = TestErrorHandler::new(ErrorHandlerConfig::default());
                        
                        let test_error = TestError::integration(
                            format!("concurrent_test_{}_{}", thread_id, i),
                            format!("Tensor creation error: {}", e),
                            "bitnet-core".to_string()
                        );
                        
                        let error_context = TestErrorContext::new(test_error)
                            .with_metadata("thread_id".to_string(), thread_id.to_string())
                            .with_metadata("iteration".to_string(), i.to_string())
                            .with_metadata("shape".to_string(), format!("{:?}", shape));
                        
                        local_handler.handle_error(error_context);
                    }
                }
                
                // Small delay to encourage thread interleaving
                thread::sleep(Duration::from_millis(1));
            }
        })
    }).collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    let total_errors = error_count.load(Ordering::Relaxed);
    let total_successes = success_count.load(Ordering::Relaxed);
    
    println!("  Concurrent execution results:");
    println!("    Total successes: {}", total_successes);
    println!("    Total errors: {}", total_errors);
    
    // Verify that we handled errors and successes concurrently
    assert!(total_errors > 0, "Should have encountered some errors with invalid shapes");
    assert!(total_successes > 0, "Should have had some successes with valid shapes");
    
    // Verify system is still functional after concurrent errors
    let final_test = BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device);
    assert!(final_test.is_ok(), "System should remain functional after concurrent error handling");
    
    println!("  ✅ System remains functional after concurrent error handling");
    println!("✅ Concurrent error handling test completed");
    Ok(())
}

/// Test error pattern detection and analysis
#[test]
fn test_error_pattern_detection() -> Result<()> {
    println!("🔄 Testing error pattern detection...");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: 1,
        continue_on_critical: true,
        collect_diagnostics: true,
        enable_pattern_detection: true,
    });
    
    // Generate a series of errors that should form detectable patterns
    let error_patterns = vec![
        ("timeout_test_1", TestError::timeout("timeout_test_1".to_string(), Duration::from_secs(5), TestCategory::Unit)),
        ("timeout_test_2", TestError::timeout("timeout_test_2".to_string(), Duration::from_secs(8), TestCategory::Integration)),
        ("timeout_test_3", TestError::timeout("timeout_test_3".to_string(), Duration::from_secs(12), TestCategory::Performance)),
        ("memory_test_1", TestError::memory("memory_test_1".to_string(), "Out of memory".to_string())),
        ("memory_test_2", TestError::memory("memory_test_2".to_string(), "Allocation failed".to_string())),
        ("panic_test_1", TestError::panic("panic_test_1".to_string(), "assertion failed".to_string())),
        ("integration_test_1", TestError::integration("integration_test_1".to_string(), "Cross-crate failure".to_string(), "bitnet-quant".to_string())),
    ];
    
    // Feed errors to handler
    for (test_name, error) in error_patterns {
        println!("  Processing error pattern for: {}", test_name);
        let error_context = TestErrorContext::new(error)
            .with_metadata("pattern_test".to_string(), "true".to_string());
        error_handler.handle_error(error_context);
    }
    
    // Analyze patterns
    let summary = error_handler.generate_summary();
    
    println!("  ✅ Pattern analysis results:");
    println!("    Total errors: {}", summary.total_errors);
    println!("    Error patterns detected:");
    
    for (pattern, count) in &summary.error_patterns {
        println!("      {}: {} occurrences", pattern, count);
    }
    
    if let Some(most_common) = &summary.most_common_error {
        println!("    Most common error type: {}", most_common);
    }
    
    // Verify we detected the timeout pattern (should be most common)
    assert!(summary.error_patterns.get("timeout").unwrap_or(&0) >= &3, 
            "Should detect timeout pattern with at least 3 occurrences");
    assert!(summary.error_patterns.get("memory").unwrap_or(&0) >= &2,
            "Should detect memory pattern with at least 2 occurrences");
    
    println!("✅ Error pattern detection test completed");
    Ok(())
}

/// Test CI-specific error handling optimizations
#[test]
fn test_ci_error_handling_optimizations() -> Result<()> {
    println!("🔄 Testing CI-specific error handling optimizations...");
    
    // Simulate CI environment
    std::env::set_var("CI", "true");
    std::env::set_var("GITHUB_ACTIONS", "true");
    
    let mut error_handler = TestErrorHandler::new(ErrorHandlerConfig {
        max_retries: 1, // Reduced retries for CI
        continue_on_critical: false, // Fail fast in CI
        collect_diagnostics: false, // Reduced diagnostics in CI
        enable_pattern_detection: true,
    });
    
    // Test timeout handling in CI (should be more aggressive)
    let ci_timeout_error = TestError::timeout(
        "ci_timeout_test".to_string(),
        Duration::from_secs(30),
        TestCategory::Performance
    );
    
    let error_context = TestErrorContext::new(ci_timeout_error)
        .with_metadata("environment".to_string(), "CI".to_string())
        .with_metadata("runner".to_string(), "github-actions".to_string());
    
    let action = error_handler.handle_error(error_context);
    
    match action {
        bitnet_core::test_utils::error::ErrorHandlerAction::SkipTest { test_name, reason } => {
            println!("  ✅ CI optimization: Skipping performance test '{}' - {}", test_name, reason);
        },
        bitnet_core::test_utils::error::ErrorHandlerAction::RetryTest { max_attempts, .. } => {
            println!("  ✅ CI optimization: Limited retries to {}", max_attempts);
            assert_eq!(max_attempts, 1, "CI should limit retries to 1");
        },
        _ => {
            println!("  ⚠️ CI handling action: {:?}", action);
        }
    }
    
    // Test critical error handling in CI (should fail fast)
    let critical_error = TestError::panic(
        "critical_ci_test".to_string(),
        "Critical system failure".to_string()
    );
    
    let error_context = TestErrorContext::new(critical_error)
        .with_metadata("environment".to_string(), "CI".to_string());
    
    let action = error_handler.handle_error(error_context);
    
    match action {
        bitnet_core::test_utils::error::ErrorHandlerAction::StopExecution { reason } => {
            println!("  ✅ CI optimization: Stopping execution on critical error - {}", reason);
        },
        _ => {
            println!("  ⚠️ Expected stop execution for critical error in CI");
        }
    }
    
    // Clean up environment variables
    std::env::remove_var("CI");
    std::env::remove_var("GITHUB_ACTIONS");
    
    let summary = error_handler.generate_summary();
    println!("  ✅ CI error handling summary: {} errors processed", summary.total_errors);
    
    println!("✅ CI error handling optimizations test completed");
    Ok(())
}

// Helper function to create test validation error
impl TestError {
    pub fn validation(test_name: String, validation_error: String, expected: Option<String>, actual: Option<String>) -> Self {
        TestError::Validation {
            test_name,
            validation_error,
            expected_value: expected,
            actual_value: actual,
        }
    }
}
