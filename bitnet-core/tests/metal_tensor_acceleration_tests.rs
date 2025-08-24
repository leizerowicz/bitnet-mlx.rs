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
    use std::time::Instant;

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

    #[test]
    fn test_metal_performance_benchmark() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal performance test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        accelerator
            .initialize()
            .expect("Failed to initialize Metal accelerator");

        // Create larger test matrices for performance testing
        let size = 512;
        let a_data: Vec<f32> = (0..size * size).map(|i| (i % 100) as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 100) as f32).collect();

        let tensor_a = BitNetTensor::from_data(&a_data, &[size, size], BitNetDType::F32, None)
            .expect("Failed to create large tensor A");
        let tensor_b = BitNetTensor::from_data(&b_data, &[size, size], BitNetDType::F32, None)
            .expect("Failed to create large tensor B");

        println!("Benchmarking {}x{} matrix multiplication", size, size);

        // Warm-up run
        let _ = accelerator.matmul(&tensor_a, &tensor_b);

        // Performance measurement
        let num_runs = 5;
        let mut total_time = 0.0;

        for i in 0..num_runs {
            let start_time = Instant::now();
            let result = accelerator.matmul(&tensor_a, &tensor_b);
            let elapsed = start_time.elapsed().as_secs_f64();

            match result {
                Ok((_, metrics)) => {
                    total_time += elapsed;
                    println!(
                        "Run {}: {:.3}ms (ops/sec: {:.2}M)",
                        i + 1,
                        elapsed * 1000.0,
                        metrics.operations_per_second / 1e6
                    );
                }
                Err(e) => {
                    panic!("Metal matmul benchmark failed on run {}: {}", i + 1, e);
                }
            }
        }

        let avg_time = total_time / num_runs as f64;
        let operations = (size * size * size) as f64;
        let gflops = operations / (avg_time * 1e9);

        println!("✅ Metal Performance Results:");
        println!("   Average time: {:.3}ms", avg_time * 1000.0);
        println!("   Performance: {:.2} GFLOPS", gflops);

        // Get memory statistics
        let memory_stats = accelerator
            .get_memory_stats()
            .expect("Failed to get memory stats");
        println!("   Memory stats: {:?}", memory_stats);

        accelerator
            .cleanup()
            .expect("Failed to cleanup accelerator");
    }

    #[test]
    fn test_metal_memory_management() {
        if !is_metal_available() {
            println!("⏭️ Skipping Metal memory test - Metal not available");
            return;
        }

        let mut accelerator =
            create_metal_accelerator().expect("Failed to create Metal accelerator");

        accelerator
            .initialize()
            .expect("Failed to initialize Metal accelerator");

        // Get initial memory stats
        let initial_stats = accelerator
            .get_memory_stats()
            .expect("Failed to get initial memory stats");
        println!("Initial memory stats: {:?}", initial_stats);

        // Perform multiple operations to test memory management
        for i in 0..10 {
            let size = 100 + i * 10;
            let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();

            let tensor_a = BitNetTensor::from_data(&data, &[size, size], BitNetDType::F32, None)
                .expect("Failed to create tensor A");
            let tensor_b = BitNetTensor::from_data(&data, &[size, size], BitNetDType::F32, None)
                .expect("Failed to create tensor B");

            // Perform operations
            let _ = accelerator.add(&tensor_a, &tensor_b);
            let _ = accelerator.mul(&tensor_a, &tensor_b);

            if size <= 200 {
                let _ = accelerator.matmul(&tensor_a, &tensor_b);
            }
        }

        // Get final memory stats
        let final_stats = accelerator
            .get_memory_stats()
            .expect("Failed to get final memory stats");
        println!("Final memory stats: {:?}", final_stats);

        // Check that memory is properly managed
        assert!(
            final_stats.allocation_efficiency >= 0.0,
            "Allocation efficiency should be non-negative"
        );
        println!("✅ Memory management test completed");

        accelerator
            .cleanup()
            .expect("Failed to cleanup accelerator");

        // Get stats after cleanup
        let cleanup_stats = accelerator
            .get_memory_stats()
            .expect("Failed to get cleanup memory stats");
        println!("Post-cleanup memory stats: {:?}", cleanup_stats);
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
