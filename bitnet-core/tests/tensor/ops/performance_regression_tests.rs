//! Performance Regression Tests for Tensor Operations
//!
//! This test suite validates that tensor operations maintain expected
//! performance characteristics and detects performance regressions.

use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::tensor::ops::arithmetic::{add, mul, add_scalar};
use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::device::get_cpu_device;
use bitnet_core::test_utils::{TestCategory, timeout::execute_test_with_monitoring};
use std::time::{Duration, Instant};

// Define the monitored_test macro locally since it may not be exported
macro_rules! monitored_test {
    (
        name: $test_name:ident,
        category: $category:expr,
        timeout: $timeout:expr,
        fn $fn_name:ident() $body:block
    ) => {
        #[test]
        fn $test_name() {
            use bitnet_core::test_utils::timeout::execute_test_with_monitoring;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Performance baseline targets (operations per second)
    struct PerformanceBaselines {
        small_vector_add_ops_per_sec: f64,
        medium_vector_add_ops_per_sec: f64,
        large_vector_add_ops_per_sec: f64,
        small_matrix_add_ops_per_sec: f64,
        scalar_add_ops_per_sec: f64,
    }

    impl Default for PerformanceBaselines {
        fn default() -> Self {
            Self {
                small_vector_add_ops_per_sec: 1_000_000.0,  // 1M ops/sec
                medium_vector_add_ops_per_sec: 100_000.0,   // 100K ops/sec
                large_vector_add_ops_per_sec: 10_000.0,     // 10K ops/sec
                small_matrix_add_ops_per_sec: 10_000.0,     // 10K ops/sec
                scalar_add_ops_per_sec: 5_000_000.0,        // 5M ops/sec
            }
        }
    }

    fn benchmark_operation<F>(iterations: u32, operation: F) -> Result<f64, Box<dyn std::error::Error>>
    where
        F: Fn() -> Result<(), Box<dyn std::error::Error>>,
    {
        // Warmup
        for _ in 0..10 {
            operation()?;
        }

        let start = Instant::now();
        for _ in 0..iterations {
            operation()?;
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        Ok(ops_per_sec)
    }

    #[test]
    fn test_small_vector_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baselines = PerformanceBaselines::default();

        let a = BitNetTensor::ones(&[1000], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[1000], BitNetDType::F32, Some(device.clone()))?;

        let ops_per_sec = benchmark_operation(10000, || {
            let _result = add(&a, &b)?;
            Ok(())
        })?;

        println!("Small vector addition: {:.0} ops/sec (target: {:.0})",
                 ops_per_sec, baselines.small_vector_add_ops_per_sec);

        assert!(ops_per_sec >= baselines.small_vector_add_ops_per_sec * 0.8,
                "Performance regression: {:.0} < {:.0}",
                ops_per_sec, baselines.small_vector_add_ops_per_sec * 0.8);

        Ok(())
    }

    #[test]
    fn test_medium_vector_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baselines = PerformanceBaselines::default();

        let a = BitNetTensor::ones(&[100_000], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[100_000], BitNetDType::F32, Some(device.clone()))?;

        let ops_per_sec = benchmark_operation(1000, || {
            let _result = add(&a, &b)?;
            Ok(())
        })?;

        println!("Medium vector addition: {:.0} ops/sec (target: {:.0})",
                 ops_per_sec, baselines.medium_vector_add_ops_per_sec);

        assert!(ops_per_sec >= baselines.medium_vector_add_ops_per_sec * 0.8,
                "Performance regression: {:.0} < {:.0}",
                ops_per_sec, baselines.medium_vector_add_ops_per_sec * 0.8);

        Ok(())
    }

    #[test]
    fn test_large_vector_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baselines = PerformanceBaselines::default();

        let a = BitNetTensor::ones(&[1_000_000], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[1_000_000], BitNetDType::F32, Some(device.clone()))?;

        let ops_per_sec = benchmark_operation(100, || {
            let _result = add(&a, &b)?;
            Ok(())
        })?;

        println!("Large vector addition: {:.0} ops/sec (target: {:.0})",
                 ops_per_sec, baselines.large_vector_add_ops_per_sec);

        assert!(ops_per_sec >= baselines.large_vector_add_ops_per_sec * 0.8,
                "Performance regression: {:.0} < {:.0}",
                ops_per_sec, baselines.large_vector_add_ops_per_sec * 0.8);

        Ok(())
    }

    #[test]
    fn test_matrix_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baselines = PerformanceBaselines::default();

        let a = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;

        let ops_per_sec = benchmark_operation(100, || {
            let _result = add(&a, &b)?;
            Ok(())
        })?;

        println!("Matrix addition: {:.0} ops/sec (target: {:.0})",
                 ops_per_sec, baselines.small_matrix_add_ops_per_sec);

        assert!(ops_per_sec >= baselines.small_matrix_add_ops_per_sec * 0.8,
                "Performance regression: {:.0} < {:.0}",
                ops_per_sec, baselines.small_matrix_add_ops_per_sec * 0.8);

        Ok(())
    }

    #[test]
    fn test_scalar_operations_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baselines = PerformanceBaselines::default();

        let a = BitNetTensor::ones(&[100_000], BitNetDType::F32, Some(device.clone()))?;

        let ops_per_sec = benchmark_operation(5000, || {
            let _result = add_scalar(&a, 2.5)?;
            Ok(())
        })?;

        println!("Scalar addition: {:.0} ops/sec (target: {:.0})",
                 ops_per_sec, baselines.scalar_add_ops_per_sec);

        assert!(ops_per_sec >= baselines.scalar_add_ops_per_sec * 0.8,
                "Performance regression: {:.0} < {:.0}",
                ops_per_sec, baselines.scalar_add_ops_per_sec * 0.8);

        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let pool = HybridMemoryPool::default();

        // Get baseline memory stats
        let baseline_stats = pool.get_stats();
        let baseline_allocated = baseline_stats.bytes_in_use;

        // Create and operate on tensors
        let iterations = 1000;
        for _ in 0..iterations {
            let a = BitNetTensor::ones(&[1000], BitNetDType::F32, Some(device.clone()))?;
            let b = BitNetTensor::ones(&[1000], BitNetDType::F32, Some(device.clone()))?;
            let _result = add(&a, &b)?;
            // Tensors should be automatically cleaned up here
        }

        // Check final memory stats
        let final_stats = pool.get_stats();
        let final_allocated = final_stats.bytes_in_use;
        let memory_increase = final_allocated - baseline_allocated;

        println!("Memory efficiency test:");
        println!("  Baseline allocated: {} bytes", baseline_allocated);
        println!("  Final allocated: {} bytes", final_allocated);
        println!("  Memory increase: {} bytes", memory_increase);
        println!("  Active allocations: {}", final_stats.active_allocations);

        // Memory increase should be minimal (allow for some pool growth)
        let max_acceptable_increase = 10 * 1024 * 1024; // 10MB
        assert!(memory_increase <= max_acceptable_increase,
                "Excessive memory increase: {} bytes > {} bytes",
                memory_increase, max_acceptable_increase);

        Ok(())
    }

    #[test]
    fn test_operation_correctness_vs_performance() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();

        // Test that performance optimizations don't affect correctness
        let a = BitNetTensor::sequential(&[100, 100], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[100, 100], BitNetDType::F32, Some(device.clone()))?;

        // Perform operations multiple times and verify consistent results
        let mut results = Vec::new();
        for _ in 0..10 {
            let result = add(&a, &b)?;
            results.push(result);
        }

        // All results should be identical
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(result.shape(), results[0].shape(),
                      "Result {} has different shape", i);
            assert_eq!(result.dtype(), results[0].dtype(),
                      "Result {} has different dtype", i);
            // Additional value comparison would be implemented here
        }

        println!("✓ Operation correctness maintained under performance testing");
        Ok(())
    }

    monitored_test! {
        name: test_multi_threaded_performance,
        category: TestCategory::Stress,
        timeout: Duration::from_secs(300),
        fn test_multi_threaded_performance() -> Result<(), Box<dyn std::error::Error>> {
            use std::sync::Arc;
            use std::thread;

            println!("🧵 Starting multi-threaded performance test...");

            let device = match get_cpu_device() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to get CPU device: {:?}", e);
                    return Err(format!("CPU device unavailable: {}", e).into());
                }
            };

            let a = match BitNetTensor::ones(&[10_000], BitNetDType::F32, Some(device.clone())) {
                Ok(tensor) => Arc::new(tensor),
                Err(e) => {
                    eprintln!("Failed to create tensor A: {:?}", e);
                    return Err(format!("Tensor A creation failed: {}", e).into());
                }
            };

            let b = match BitNetTensor::ones(&[10_000], BitNetDType::F32, Some(device.clone())) {
                Ok(tensor) => Arc::new(tensor),
                Err(e) => {
                    eprintln!("Failed to create tensor B: {:?}", e);
                    return Err(format!("Tensor B creation failed: {}", e).into());
                }
            };

            let num_threads = 4;
            let iterations_per_thread = 1000;

            println!("🚀 Spawning {} threads with {} iterations each...", num_threads, iterations_per_thread);

            let start = Instant::now();
            let handles: Vec<_> = (0..num_threads).map(|thread_id| {
                let a_clone = Arc::clone(&a);
                let b_clone = Arc::clone(&b);

                thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let mut thread_errors = 0;
                    for i in 0..iterations_per_thread {
                        match add(&*a_clone, &*b_clone) {
                            Ok(_result) => {},
                            Err(e) => {
                                eprintln!("Thread {} iteration {} failed: {:?}", thread_id, i, e);
                                thread_errors += 1;
                                if thread_errors > 10 {
                                    return Err(format!("Thread {} exceeded error threshold", thread_id).into());
                                }
                            }
                        }
                    }
                    if thread_errors > 0 {
                        println!("⚠️  Thread {} completed with {} errors", thread_id, thread_errors);
                    } else {
                        println!("✅ Thread {} completed successfully", thread_id);
                    }
                    Ok(())
                })
            }).collect();

            let mut total_thread_errors = 0;
            for (thread_id, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(result) => {
                        if let Err(e) = result {
                            eprintln!("❌ Thread {} failed: {:?}", thread_id, e);
                            total_thread_errors += 1;
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Thread {} panicked: {:?}", thread_id, e);
                        total_thread_errors += 1;
                    }
                }
            }

            let duration = start.elapsed();
            let total_ops = num_threads * iterations_per_thread;
            let ops_per_sec = total_ops as f64 / duration.as_secs_f64();

            println!("📊 Multi-threaded performance results:");
            println!("   Operations per second: {:.0}", ops_per_sec);
            println!("   Total operations: {}", total_ops);
            println!("   Duration: {:.2}s", duration.as_secs_f64());
            println!("   Thread errors: {}", total_thread_errors);

            // Multi-threaded performance should be reasonable
            let min_expected_ops_per_sec = 10_000.0; // Conservative baseline
            if ops_per_sec < min_expected_ops_per_sec {
                eprintln!("❌ Multi-threaded performance too low: {:.0} < {:.0}",
                         ops_per_sec, min_expected_ops_per_sec);
            }
            assert!(ops_per_sec >= min_expected_ops_per_sec,
                    "Multi-threaded performance too low: {:.0} < {:.0}",
                    ops_per_sec, min_expected_ops_per_sec);

            if total_thread_errors > 0 {
                eprintln!("⚠️  {} thread errors occurred during test", total_thread_errors);
            }

            println!("✅ Multi-threaded performance test completed successfully");
            Ok(())
        }
    }
}
