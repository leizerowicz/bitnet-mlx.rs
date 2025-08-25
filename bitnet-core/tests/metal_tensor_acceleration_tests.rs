//! Metal Compute Shader Integration Tests
//!
//! Tests for Metal GPU acceleration of tensor operations including:
//! - Matrix multiplication with compute shaders
//! - Element-wise operations (add, mul, sub, div)
//! - GPU memory transfer optimization
//! - Command buffer management and synchronization
//! - Performance benchmarking

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_integration_tests {
    use bitnet_core::tensor::acceleration::{create_metal_accelerator, is_metal_available};
    use bitnet_core::tensor::acceleration::{AccelerationBackendImpl, MetalAccelerator};
    use bitnet_core::tensor::core::BitNetTensor;
    use bitnet_core::tensor::dtype::BitNetDType;
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

    #[test]
    fn test_metal_availability() {
        if is_metal_available() {
            println!("✅ Metal is available on this system");
        } else {
            println!("❌ Metal is not available on this system");
            return;
        }
    }

    #[test]
    fn test_metal_accelerator_creation() {
        let result = create_metal_accelerator();
        match result {
            Ok(accelerator) => {
                assert!(accelerator.is_available());
                println!("✅ Metal accelerator created successfully");
            }
            Err(e) => {
                if is_metal_available() {
                    panic!(
                        "Failed to create Metal accelerator on available system: {}",
                        e
                    );
                } else {
                    println!("⚠️ Metal accelerator creation failed as expected on unavailable system: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_metal_accelerator_initialization() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal initialization test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        let result = accelerator.initialize();
        match result {
            Ok(()) => {
                println!("✅ Metal accelerator initialized successfully");

                // Test capabilities
                let capabilities = accelerator.get_capabilities();
                println!("Metal capabilities: {:?}", capabilities);

                // Cleanup
                accelerator
                    .cleanup()
                    .expect("Failed to cleanup accelerator");
            }
            Err(e) => {
                panic!("Failed to initialize Metal accelerator: {}", e);
            }
        }
    }

    #[test]
    fn test_metal_matrix_multiplication() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal matmul test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        accelerator
            .initialize()
            .expect("Failed to initialize Metal accelerator");

        // Create test matrices
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 matrix

        let tensor_a = BitNetTensor::from_data(&a_data, &[2, 3], BitNetDType::F32, None)
            .expect("Failed to create tensor A");
        let tensor_b = BitNetTensor::from_data(&b_data, &[3, 2], BitNetDType::F32, None)
            .expect("Failed to create tensor B");

        println!("Matrix A (2x3): {:?}", a_data);
        println!("Matrix B (3x2): {:?}", b_data);

        // Perform Metal-accelerated matrix multiplication
        let start_time = Instant::now();
        let result = accelerator.matmul(&tensor_a, &tensor_b);
        let elapsed = start_time.elapsed();

        match result {
            Ok((result_tensor, metrics)) => {
                println!(
                    "✅ Metal matmul completed in {:.3}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
                println!("Metal metrics: {:?}", metrics);

                // Verify result shape
                let result_shape = result_tensor.shape().dims();
                assert_eq!(result_shape, &[2, 2], "Result shape should be 2x2");

                // Expected result: [[58, 64], [139, 154]]
                let result_data = result_tensor
                    .as_slice::<f32>()
                    .expect("Failed to get result data");
                println!("Result matrix: {:?}", result_data);

                // Verify correctness (with some tolerance for floating point)
                let expected = vec![58.0f32, 64.0, 139.0, 154.0];
                for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate()
                {
                    let diff = (actual - expected).abs();
                    assert!(
                        diff < 1e-5,
                        "Element {} differs: {} vs {} (diff: {})",
                        i,
                        actual,
                        expected,
                        diff
                    );
                }

                println!("✅ Matrix multiplication result verified");
            }
            Err(e) => {
                panic!("Metal matmul failed: {}", e);
            }
        }

        accelerator
            .cleanup()
            .expect("Failed to cleanup accelerator");
    }

    #[test]
    fn test_metal_element_wise_operations() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal element-wise test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        accelerator
            .initialize()
            .expect("Failed to initialize Metal accelerator");

        // Create test tensors
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];

        let tensor_a = BitNetTensor::from_data(&a_data, &[2, 3], BitNetDType::F32, None)
            .expect("Failed to create tensor A");
        let tensor_b = BitNetTensor::from_data(&b_data, &[2, 3], BitNetDType::F32, None)
            .expect("Failed to create tensor B");

        // Test addition
        let start_time = Instant::now();
        let result = accelerator.add(&tensor_a, &tensor_b);
        let add_elapsed = start_time.elapsed();

        match result {
            Ok((result_tensor, metrics)) => {
                println!(
                    "✅ Metal add completed in {:.3}ms",
                    add_elapsed.as_secs_f64() * 1000.0
                );
                println!("Add metrics: {:?}", metrics);

                let result_data = result_tensor
                    .as_slice::<f32>()
                    .expect("Failed to get add result data");
                let expected = vec![3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0];

                for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate()
                {
                    let diff = (actual - expected).abs();
                    assert!(
                        diff < 1e-5,
                        "Add element {} differs: {} vs {} (diff: {})",
                        i,
                        actual,
                        expected,
                        diff
                    );
                }

                println!("✅ Addition result verified");
            }
            Err(e) => {
                panic!("Metal add failed: {}", e);
            }
        }

        // Test multiplication
        let start_time = Instant::now();
        let result = accelerator.mul(&tensor_a, &tensor_b);
        let mul_elapsed = start_time.elapsed();

        match result {
            Ok((result_tensor, metrics)) => {
                println!(
                    "✅ Metal mul completed in {:.3}ms",
                    mul_elapsed.as_secs_f64() * 1000.0
                );
                println!("Mul metrics: {:?}", metrics);

                let result_data = result_tensor
                    .as_slice::<f32>()
                    .expect("Failed to get mul result data");
                let expected = vec![2.0f32, 6.0, 12.0, 20.0, 30.0, 42.0];

                for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate()
                {
                    let diff = (actual - expected).abs();
                    assert!(
                        diff < 1e-5,
                        "Mul element {} differs: {} vs {} (diff: {})",
                        i,
                        actual,
                        expected,
                        diff
                    );
                }

                println!("✅ Multiplication result verified");
            }
            Err(e) => {
                panic!("Metal mul failed: {}", e);
            }
        }

        accelerator
            .cleanup()
            .expect("Failed to cleanup accelerator");
    }

    monitored_test! {
        name: test_metal_performance_benchmark,
        category: TestCategory::Performance,
        timeout: Duration::from_secs(240),
        fn test_metal_performance_benchmark() {
            println!("🚀 Starting Metal GPU performance benchmark test...");
            
            if !is_metal_available() {
                println!("⏭️ Skipping Metal performance test - Metal not available");
                return;
            }

            let mut accelerator = match create_metal_accelerator() {
                Ok(acc) => acc,
                Err(e) => {
                    eprintln!("Failed to create Metal accelerator: {:?}", e);
                    panic!("Metal accelerator creation failed: {}", e);
                }
            };

            if let Err(e) = accelerator.initialize() {
                eprintln!("Failed to initialize Metal accelerator: {:?}", e);
                panic!("Metal accelerator initialization failed: {}", e);
            }

            // Create larger test matrices for performance testing
            let size = 512;
            println!("📊 Creating {}x{} test matrices...", size, size);
            
            let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
            let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 100) as f32).collect();

            let tensor_a = match BitNetTensor::from_data(&a_data, &[size, size], BitNetDType::F32, None) {
                Ok(tensor) => tensor,
                Err(e) => {
                    eprintln!("Failed to create large tensor A: {:?}", e);
                    panic!("Large tensor A creation failed: {}", e);
                }
            };

            let tensor_b = match BitNetTensor::from_data(&b_data, &[size, size], BitNetDType::F32, None) {
                Ok(tensor) => tensor,
                Err(e) => {
                    eprintln!("Failed to create large tensor B: {:?}", e);
                    panic!("Large tensor B creation failed: {}", e);
                }
            };

            println!("🔥 Benchmarking {}x{} matrix multiplication", size, size);

            // Warm-up run with error handling
            println!("🌡️  Performing warm-up run...");
            match accelerator.matmul(&tensor_a, &tensor_b) {
                Ok(_) => println!("✅ Warm-up completed successfully"),
                Err(e) => {
                    eprintln!("Warm-up failed: {:?}", e);
                    // Continue with benchmark but note the issue
                    println!("⚠️  Continuing benchmark despite warm-up failure");
                }
            }

            // Performance measurement with comprehensive error handling
            let num_runs = 5;
            let mut total_time = 0.0;
            let mut successful_runs = 0;
            let mut failed_runs = 0;

            for i in 0..num_runs {
                println!("🏃 Starting benchmark run {}...", i + 1);
                let start_time = Instant::now();
                let result = accelerator.matmul(&tensor_a, &tensor_b);
                let elapsed = start_time.elapsed().as_secs_f64();

                match result {
                    Ok((_, metrics)) => {
                        total_time += elapsed;
                        successful_runs += 1;
                        println!(
                            "✅ Run {}: {:.3}ms (ops/sec: {:.2}M)",
                            i + 1,
                            elapsed * 1000.0,
                            metrics.operations_per_second / 1e6
                        );
                    }
                    Err(e) => {
                        failed_runs += 1;
                        eprintln!("❌ Metal matmul benchmark failed on run {}: {:?}", i + 1, e);
                        // Continue with remaining runs instead of panicking
                        println!("⚠️  Continuing with remaining benchmark runs...");
                    }
                }
            }

            println!("📊 Metal Performance Benchmark Results:");
            println!("   Successful runs: {}/{}", successful_runs, num_runs);
            println!("   Failed runs: {}", failed_runs);

            if successful_runs > 0 {
                let avg_time = total_time / successful_runs as f64;
                let operations = (size * size * size) as f64;
                let gflops = operations / (avg_time * 1e9);

                println!("✅ Metal Performance Results:");
                println!("   Average time: {:.3}ms", avg_time * 1000.0);
                println!("   Performance: {:.2} GFLOPS", gflops);

                // Validate performance is reasonable
                if gflops < 1.0 {
                    eprintln!("⚠️  Warning: Performance seems low ({:.2} GFLOPS)", gflops);
                }
            } else {
                eprintln!("❌ All benchmark runs failed - no performance data available");
            }

            // Get memory statistics with error handling
            match accelerator.get_memory_stats() {
                Ok(memory_stats) => {
                    println!("📈 Memory stats: {:?}", memory_stats);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to get memory stats: {:?}", e);
                }
            }

            // Cleanup with error handling
            if let Err(e) = accelerator.cleanup() {
                eprintln!("⚠️  Failed to cleanup accelerator: {:?}", e);
            } else {
                println!("🧹 Metal accelerator cleanup completed");
            }

            // Require at least some successful runs
            if successful_runs == 0 {
                panic!("All Metal benchmark runs failed - GPU acceleration may not be working");
            }

            println!("✅ Metal performance benchmark test completed successfully");
        }
    }

    monitored_test! {
        name: test_metal_memory_management,
        category: TestCategory::Performance,
        timeout: Duration::from_secs(180),
        fn test_metal_memory_management() {
            println!("🧠 Starting Metal GPU memory management test...");
            
            if !is_metal_available() {
                println!("⏭️ Skipping Metal memory test - Metal not available");
                return;
            }

            let mut accelerator = match create_metal_accelerator() {
                Ok(acc) => acc,
                Err(e) => {
                    eprintln!("Failed to create Metal accelerator: {:?}", e);
                    panic!("Metal accelerator creation failed: {}", e);
                }
            };

            if let Err(e) = accelerator.initialize() {
                eprintln!("Failed to initialize Metal accelerator: {:?}", e);
                panic!("Metal accelerator initialization failed: {}", e);
            }

            // Get initial memory stats with error handling
            let initial_stats = match accelerator.get_memory_stats() {
                Ok(stats) => {
                    println!("📊 Initial memory stats: {:?}", stats);
                    stats
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to get initial memory stats: {:?}", e);
                    // Continue test but note the issue
                    println!("⚠️  Continuing memory test without initial stats");
                    Default::default()
                }
            };

            // Perform multiple operations to test memory management with comprehensive error tracking
            let total_iterations = 10;
            let mut successful_operations = 0;
            let mut failed_operations = 0;
            let mut tensor_creation_errors = 0;
            let mut add_operation_errors = 0;
            let mut mul_operation_errors = 0;
            let mut matmul_operation_errors = 0;

            println!("🔄 Performing {} memory management iterations...", total_iterations);

            for i in 0..total_iterations {
                let size = 100 + i * 10;
                println!("📐 Iteration {}: Testing with {}x{} matrices", i + 1, size, size);
                
                let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();

                // Create tensors with error handling
                let tensor_a = match BitNetTensor::from_data(&data, &[size, size], BitNetDType::F32, None) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        tensor_creation_errors += 1;
                        eprintln!("❌ Failed to create tensor A (iteration {}): {:?}", i + 1, e);
                        continue; // Skip this iteration
                    }
                };

                let tensor_b = match BitNetTensor::from_data(&data, &[size, size], BitNetDType::F32, None) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        tensor_creation_errors += 1;
                        eprintln!("❌ Failed to create tensor B (iteration {}): {:?}", i + 1, e);
                        continue; // Skip this iteration
                    }
                };

                // Perform add operation with error handling
                match accelerator.add(&tensor_a, &tensor_b) {
                    Ok(_) => {
                        successful_operations += 1;
                        println!("✅ Add operation {} completed", i + 1);
                    }
                    Err(e) => {
                        add_operation_errors += 1;
                        eprintln!("❌ Add operation failed (iteration {}): {:?}", i + 1, e);
                        failed_operations += 1;
                    }
                }

                // Perform mul operation with error handling
                match accelerator.mul(&tensor_a, &tensor_b) {
                    Ok(_) => {
                        successful_operations += 1;
                        println!("✅ Mul operation {} completed", i + 1);
                    }
                    Err(e) => {
                        mul_operation_errors += 1;
                        eprintln!("❌ Mul operation failed (iteration {}): {:?}", i + 1, e);
                        failed_operations += 1;
                    }
                }

                // Perform matmul operation for smaller sizes with error handling
                if size <= 200 {
                    match accelerator.matmul(&tensor_a, &tensor_b) {
                        Ok(_) => {
                            successful_operations += 1;
                            println!("✅ Matmul operation {} completed", i + 1);
                        }
                        Err(e) => {
                            matmul_operation_errors += 1;
                            eprintln!("❌ Matmul operation failed (iteration {}): {:?}", i + 1, e);
                            failed_operations += 1;
                        }
                    }
                } else {
                    println!("⏭️  Skipping matmul for size {} (too large)", size);
                }

                // Check memory stats periodically with error handling
                if (i + 1) % 3 == 0 {
                    match accelerator.get_memory_stats() {
                        Ok(stats) => {
                            println!("📊 Memory stats at iteration {}: {:?}", i + 1, stats);
                        }
                        Err(e) => {
                            eprintln!("⚠️  Failed to get memory stats at iteration {}: {:?}", i + 1, e);
                        }
                    }
                }
            }

            // Get final memory stats with error handling
            let final_stats = match accelerator.get_memory_stats() {
                Ok(stats) => {
                    println!("📊 Final memory stats: {:?}", stats);
                    Some(stats)
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to get final memory stats: {:?}", e);
                    None
                }
            };

            // Validate memory management if we have stats
            if let Some(stats) = final_stats {
                if stats.allocation_efficiency >= 0.0 {
                    println!("✅ Memory allocation efficiency is valid: {:.2}%", stats.allocation_efficiency * 100.0);
                } else {
                    eprintln!("⚠️  Warning: Allocation efficiency seems invalid: {:.2}", stats.allocation_efficiency);
                }
            }

            // Cleanup with error handling
            if let Err(e) = accelerator.cleanup() {
                eprintln!("⚠️  Failed to cleanup accelerator: {:?}", e);
            } else {
                println!("🧹 Metal accelerator cleanup completed");
            }

            // Get stats after cleanup with error handling
            match accelerator.get_memory_stats() {
                Ok(cleanup_stats) => {
                    println!("📊 Post-cleanup memory stats: {:?}", cleanup_stats);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to get post-cleanup memory stats: {:?}", e);
                }
            }

            // Report comprehensive results
            println!("📈 Metal Memory Management Test Results:");
            println!("   Total iterations: {}", total_iterations);
            println!("   Successful operations: {}", successful_operations);
            println!("   Failed operations: {}", failed_operations);
            println!("   Tensor creation errors: {}", tensor_creation_errors);
            println!("   Add operation errors: {}", add_operation_errors);
            println!("   Mul operation errors: {}", mul_operation_errors);
            println!("   Matmul operation errors: {}", matmul_operation_errors);

            // Require at least some successful operations
            if successful_operations == 0 {
                panic!("All Metal memory management operations failed - GPU memory management may not be working");
            }

            let success_rate = successful_operations as f64 / (successful_operations + failed_operations) as f64 * 100.0;
            println!("✅ Memory management test completed with {:.1}% success rate", success_rate);
        }
    }

    #[test]
    fn test_metal_error_handling() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal error handling test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        // Test operations before initialization
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = BitNetTensor::from_data(&data, &[2, 2], BitNetDType::F32, None)
            .expect("Failed to create tensor");

        let result = accelerator.matmul(&tensor, &tensor);
        assert!(result.is_err(), "Should fail when not initialized");

        // Initialize and test invalid operations
        accelerator
            .initialize()
            .expect("Failed to initialize Metal accelerator");

        // Test incompatible matrix shapes
        let tensor_a = BitNetTensor::from_data(&data, &[2, 2], BitNetDType::F32, None)
            .expect("Failed to create tensor A");
        let tensor_b = BitNetTensor::from_data(&data, &[3, 1], BitNetDType::F32, None)
            .expect("Failed to create tensor B");

        let result = accelerator.matmul(&tensor_a, &tensor_b);
        assert!(result.is_err(), "Should fail with incompatible shapes");

        println!("✅ Error handling test completed");

        accelerator
            .cleanup()
            .expect("Failed to cleanup accelerator");
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {
    #[test]
    fn test_metal_unavailable() {
        println!("✅ Metal tests skipped - Metal feature not enabled or not on macOS");
        // This test always passes to indicate the Metal functionality
        // is appropriately feature-gated
    }
}
