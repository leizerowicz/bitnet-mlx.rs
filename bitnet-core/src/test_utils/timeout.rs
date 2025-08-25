//! Test timeout handling and execution monitoring
//!
//! This module provides macros and utilities for adding timeout handling to tests,
//! monitoring test execution, and capturing performance metrics.

use std::panic;
use std::thread;
use std::time::{Duration, Instant};

use super::{ResourceUsage, TestCategory, TestExecutionResult, GLOBAL_TEST_TRACKER};

/// Macro for running a test with timeout and performance monitoring
///
/// # Examples
///
/// ```rust
/// use bitnet_core::test_utils::timeout::test_with_timeout;
/// use bitnet_core::test_utils::TestCategory;
/// use std::time::Duration;
///
/// test_with_timeout!(
///     test_name: "my_test",
///     category: TestCategory::Unit,
///     timeout: Duration::from_secs(5),
///     test_fn: || {
///         // Your test code here
///         assert_eq!(2 + 2, 4);
///     }
/// );
/// ```
#[macro_export]
macro_rules! test_with_timeout {
    (
        test_name: $name:expr,
        category: $category:expr,
        timeout: $timeout:expr,
        test_fn: $test_fn:expr
    ) => {{
        use $crate::test_utils::timeout::execute_test_with_monitoring;
        use $crate::test_utils::TestCategory;

        execute_test_with_monitoring($name.to_string(), $category, $timeout, Box::new($test_fn))
    }};

    (
        test_name: $name:expr,
        category: $category:expr,
        test_fn: $test_fn:expr
    ) => {{
        use $crate::test_utils::timeout::execute_test_with_monitoring;
        use $crate::test_utils::TestCategory;

        let timeout = $category.default_timeout();
        execute_test_with_monitoring($name.to_string(), $category, timeout, Box::new($test_fn))
    }};
}

/// Macro for creating a test function with automatic timeout and monitoring
///
/// # Examples
///
/// ```rust
/// use bitnet_core::test_utils::timeout::monitored_test;
/// use bitnet_core::test_utils::TestCategory;
/// use std::time::Duration;
///
/// monitored_test! {
///     name: test_example,
///     category: TestCategory::Unit,
///     timeout: Duration::from_secs(10),
///     fn test_example() {
///         assert_eq!(2 + 2, 4);
///     }
/// }
/// ```
#[macro_export]
macro_rules! monitored_test {
    (
        name: $test_name:ident,
        category: $category:expr,
        timeout: $timeout:expr,
        fn $fn_name:ident() $body:block
    ) => {
        #[test]
        fn $test_name() {
            use $crate::test_utils::timeout::execute_test_with_monitoring;

            let result = execute_test_with_monitoring(
                stringify!($test_name).to_string(),
                $category,
                $timeout,
                Box::new(|| $body),
            );

            if !result.success {
                if let Some(error) = &result.error_message {
                    panic!("Test failed: {}", error);
                } else {
                    panic!("Test failed with unknown error");
                }
            }

            if result.timed_out {
                panic!("Test timed out after {:.2}s", $timeout.as_secs_f64());
            }
        }
    };

    (
        name: $test_name:ident,
        category: $category:expr,
        fn $fn_name:ident() $body:block
    ) => {
        #[test]
        fn $test_name() {
            use $crate::test_utils::timeout::execute_test_with_monitoring;

            let timeout = $category.default_timeout();
            let result = execute_test_with_monitoring(
                stringify!($test_name).to_string(),
                $category,
                timeout,
                Box::new(|| $body),
            );

            if !result.success {
                if let Some(error) = &result.error_message {
                    panic!("Test failed: {}", error);
                } else {
                    panic!("Test failed with unknown error");
                }
            }

            if result.timed_out {
                panic!("Test timed out after {:.2}s", timeout.as_secs_f64());
            }
        }
    };
}

/// Macro for conditionally skipping tests in CI environments
#[macro_export]
macro_rules! skip_in_ci {
    ($category:expr, $test_fn:expr) => {{
        if $category.should_skip_in_ci() && is_ci_environment() {
            println!("Skipping {:?} test in CI environment", $category);
            return;
        }
        $test_fn()
    }};
}

/// Execute a test function with comprehensive error handling and monitoring
///
/// # Arguments
///
/// * `test_name` - Name of the test for reporting
/// * `category` - Test category for timeout and classification
/// * `timeout` - Maximum execution time allowed
/// * `test_fn` - Test function to execute
///
/// # Returns
///
/// TestExecutionResult with performance metrics and execution status
pub fn execute_test_with_monitoring(
    test_name: String,
    category: TestCategory,
    timeout: Duration,
    test_fn: Box<dyn FnOnce() + Send + 'static>,
) -> TestExecutionResult {
    let start_time = Instant::now();
    let start_memory = get_current_memory_usage();

    // Create a channel for communication between threads
    let (tx, rx) = std::sync::mpsc::channel();

    // Spawn the test execution thread
    let test_handle = thread::spawn(move || {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            test_fn();
        }));
        
        match result {
            Ok(_) => tx.send((true, false, None)).unwrap_or(()),
            Err(panic_info) => {
                let error_msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "Test panicked with unknown error".to_string()
                };
                tx.send((false, false, Some(error_msg))).unwrap_or(())
            }
        }
    });

    // Wait for completion or timeout
    let execution_result = match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => {
            // Test timed out - try to handle gracefully
            println!("⚠️ Test '{}' timed out after {:.2}s", test_name, timeout.as_secs_f64());
            (false, true, Some(format!("Test timed out after {:.2}s", timeout.as_secs_f64())))
        }
    };

    // Try to join the thread (it may still be running if timed out)
    let _ = test_handle.join();

    let duration = start_time.elapsed();
    let end_memory = get_current_memory_usage();

    let resource_usage = ResourceUsage {
        peak_memory_bytes: end_memory,
        avg_cpu_percentage: 0.0, // Would need proper CPU monitoring
        gpu_usage: None,
        allocation_count: 0, // Would need integration with memory tracker
        total_allocated_bytes: 0,
    };

    let result = TestExecutionResult {
        test_name: test_name.clone(),
        duration,
        success: execution_result.0,
        error_message: execution_result.2,
        category,
        resource_usage,
        timed_out: execution_result.1,
        timestamp: std::time::SystemTime::now(),
    };

    // Track the result in the global tracker
    GLOBAL_TEST_TRACKER.record_result(result.clone());

    result
}
/// Check if running in a CI environment
pub fn is_ci_environment() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("GITLAB_CI").is_ok()
        || std::env::var("TRAVIS").is_ok()
        || std::env::var("CIRCLECI").is_ok()
}

/// Get current memory usage (simplified implementation)
fn get_current_memory_usage() -> u64 {
    // This is a simplified implementation
    // In a real implementation, you would use platform-specific APIs
    // or integrate with the existing memory tracking system

    #[cfg(target_os = "macos")]
    {
        // Simplified memory usage detection for macOS
        // Using a basic approximation since mach2 APIs are complex
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        // Fallback value for macOS
        0
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0 // Fallback for unsupported platforms
    }
}

/// Enhanced test wrapper that provides detailed error context
pub fn execute_test_with_context<F, R>(
    test_name: &str,
    category: TestCategory,
    test_fn: F,
) -> Result<R, String>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    let start_time = Instant::now();

    let result = panic::catch_unwind(|| test_fn());

    let duration = start_time.elapsed();

    match result {
        Ok(value) => {
            if duration > category.default_timeout() {
                println!(
                    "⚠️  Test '{}' completed but took {:.2}s (expected: {:.2}s)",
                    test_name,
                    duration.as_secs_f64(),
                    category.default_timeout().as_secs_f64()
                );
            }
            Ok(value)
        }
        Err(panic_info) => {
            let error_msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                format!("Test '{}' failed: {}", test_name, s)
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                format!("Test '{}' failed: {}", test_name, s)
            } else {
                format!("Test '{}' failed with unknown panic", test_name)
            };

            Err(error_msg)
        }
    }
}

/// Utility for running tests with automatic retry on transient failures
pub fn execute_test_with_retry<F>(
    test_name: &str,
    category: TestCategory,
    max_retries: usize,
    test_fn: F,
) -> TestExecutionResult
where
    F: Fn() + Send + Sync + 'static,
{
    let mut last_result = None;
    let test_fn = std::sync::Arc::new(test_fn);

    for attempt in 0..=max_retries {
        let test_fn_clone = test_fn.clone();
        let result = execute_test_with_monitoring(
            format!("{}_attempt_{}", test_name, attempt + 1),
            category,
            category.default_timeout(),
            Box::new(move || {
                // Call the function through the Arc
                (&*test_fn_clone)()
            }),
        );

        if result.success {
            return result;
        }

        last_result = Some(result);

        if attempt < max_retries {
            println!(
                "🔄 Test '{}' failed on attempt {}, retrying...",
                test_name,
                attempt + 1
            );
            thread::sleep(Duration::from_millis(100)); // Brief delay between retries
        }
    }

    last_result.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_timeout_detection() {
        let result = execute_test_with_monitoring(
            "test_timeout".to_string(),
            TestCategory::Unit,
            Duration::from_millis(100),
            Box::new(|| {
                thread::sleep(Duration::from_millis(200)); // Sleep longer than timeout
            }),
        );

        assert!(result.timed_out);
        assert!(!result.success);
        assert!(result.duration >= Duration::from_millis(100));
    }

    #[test]
    fn test_successful_execution() {
        let result = execute_test_with_monitoring(
            "test_success".to_string(),
            TestCategory::Unit,
            Duration::from_secs(5),
            Box::new(|| {
                assert_eq!(2 + 2, 4);
            }),
        );

        assert!(!result.timed_out);
        assert!(result.success);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_panic_handling() {
        let result = execute_test_with_monitoring(
            "test_panic".to_string(),
            TestCategory::Unit,
            Duration::from_secs(5),
            Box::new(|| {
                panic!("Test panic message");
            }),
        );

        assert!(!result.timed_out);
        assert!(!result.success);
        assert!(result.error_message.is_some());
        assert!(result.error_message.unwrap().contains("Test panic message"));
    }

    #[test]
    fn test_ci_environment_detection() {
        // Test with CI environment variable set
        std::env::set_var("CI", "true");
        assert!(is_ci_environment());

        // Clean up
        std::env::remove_var("CI");
    }

    #[test]
    fn test_category_timeout_defaults() {
        assert_eq!(TestCategory::Unit.default_timeout(), Duration::from_secs(5));
        assert_eq!(
            TestCategory::Integration.default_timeout(),
            Duration::from_secs(30)
        );
        assert_eq!(
            TestCategory::Performance.default_timeout(),
            Duration::from_secs(120)
        );
        assert_eq!(
            TestCategory::Stress.default_timeout(),
            Duration::from_secs(300)
        );
        assert_eq!(
            TestCategory::Endurance.default_timeout(),
            Duration::from_secs(600)
        );
    }

    #[test]
    fn test_category_ci_skipping() {
        assert!(!TestCategory::Unit.should_skip_in_ci());
        assert!(!TestCategory::Integration.should_skip_in_ci());
        assert!(!TestCategory::Performance.should_skip_in_ci());
        assert!(TestCategory::Stress.should_skip_in_ci());
        assert!(TestCategory::Endurance.should_skip_in_ci());
    }
}
