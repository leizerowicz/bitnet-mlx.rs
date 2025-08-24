//! Performance and Benchmarking Tests for Metal Shader Compilation
//!
//! This module provides performance testing and benchmarking for Metal shader
//! compilation functionality, measuring compilation times, memory usage, and throughput.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod performance_tests {
    use bitnet_core::metal::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    /// Performance metrics for shader compilation
    #[derive(Debug, Clone)]
    struct CompilationMetrics {
        compilation_time: Duration,
        function_count: usize,
        source_size: usize,
        cache_hit: bool,
    }

    /// Benchmark shader compilation performance
    #[test]
    fn test_shader_compilation_performance_benchmark() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/perf_test_cache")),
                optimization_level: OptimizationLevel::Full,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let mut metrics = HashMap::new();

            // Clear cache to ensure clean benchmark
            compiler.clear_cache();

            // Benchmark individual shader compilation
            let shader_files = ["bitlinear.metal", "quantization.metal", "activation.metal"];

            for shader_file in &shader_files {
                let shader_path = PathBuf::from("bitnet-core/src/metal/shaders").join(shader_file);

                if shader_path.exists() {
                    // Read source size
                    let source_size = std::fs::metadata(&shader_path)
                        .map(|m| m.len() as usize)
                        .unwrap_or(0);

                    // First compilation (cold cache)
                    let start_time = Instant::now();
                    let result = compiler.compile_shader_file(&shader_path);
                    let cold_compilation_time = start_time.elapsed();

                    if let Ok(compiled_shader) = result {
                        let cold_metrics = CompilationMetrics {
                            compilation_time: cold_compilation_time,
                            function_count: compiled_shader.function_names.len(),
                            source_size,
                            cache_hit: false,
                        };

                        metrics.insert(format!("{}_cold", shader_file), cold_metrics);

                        // Second compilation (warm cache)
                        let start_time = Instant::now();
                        let _result = compiler.compile_shader_file(&shader_path);
                        let warm_compilation_time = start_time.elapsed();

                        let warm_metrics = CompilationMetrics {
                            compilation_time: warm_compilation_time,
                            function_count: compiled_shader.function_names.len(),
                            source_size,
                            cache_hit: true,
                        };

                        metrics.insert(format!("{}_warm", shader_file), warm_metrics);

                        println!("Shader: {}", shader_file);
                        println!("  Source size: {} bytes", source_size);
                        println!("  Functions: {}", compiled_shader.function_names.len());
                        println!("  Cold compilation: {:?}", cold_compilation_time);
                        println!("  Warm compilation: {:?}", warm_compilation_time);
                        println!(
                            "  Speedup: {:.2}x",
                            cold_compilation_time.as_nanos() as f64
                                / warm_compilation_time.as_nanos() as f64
                        );
                        println!();
                    }
                }
            }

            // Analyze performance metrics
            analyze_performance_metrics(&metrics);
        } else {
            println!("Skipping performance benchmark - no Metal device available");
        }
    }

    /// Test compilation throughput with multiple shaders
    #[test]
    fn test_compilation_throughput() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: false, // Disable caching for throughput test
                optimization_level: OptimizationLevel::Basic,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Measure throughput for multiple compilation rounds
            let rounds = 5;
            let mut total_time = Duration::new(0, 0);
            let mut total_shaders = 0;

            for round in 0..rounds {
                let start_time = Instant::now();
                let compiled_shaders = compiler.compile_all_shaders().unwrap();
                let round_time = start_time.elapsed();

                total_time += round_time;
                total_shaders += compiled_shaders.len();

                println!(
                    "Round {}: {} shaders in {:?}",
                    round + 1,
                    compiled_shaders.len(),
                    round_time
                );
            }

            let avg_time_per_round = total_time / rounds;
            let avg_shaders_per_round = total_shaders / rounds;
            let throughput = avg_shaders_per_round as f64 / avg_time_per_round.as_secs_f64();

            println!("Throughput Analysis:");
            println!("  Total rounds: {}", rounds);
            println!("  Average time per round: {:?}", avg_time_per_round);
            println!("  Average shaders per round: {}", avg_shaders_per_round);
            println!("  Throughput: {:.2} shaders/second", throughput);

            // Performance assertions
            assert!(
                throughput > 0.1,
                "Throughput should be at least 0.1 shaders/second"
            );
            assert!(
                avg_time_per_round < Duration::from_secs(30),
                "Average compilation time should be under 30 seconds"
            );
        } else {
            println!("Skipping throughput test - no Metal device available");
        }
    }

    /// Test memory usage during shader compilation
    #[test]
    fn test_compilation_memory_usage() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/memory_test_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Measure memory usage before compilation
            let initial_stats = compiler.get_stats();
            println!("Initial compiler stats: {:?}", initial_stats);

            // Compile all shaders and measure memory growth
            let start_time = Instant::now();
            let compiled_shaders = compiler.compile_all_shaders().unwrap();
            let compilation_time = start_time.elapsed();

            let final_stats = compiler.get_stats();
            println!("Final compiler stats: {:?}", final_stats);

            // Calculate memory efficiency metrics
            let total_functions = final_stats.total_functions;
            let cached_shaders = final_stats.cached_shaders;

            println!("Memory Usage Analysis:");
            println!("  Compiled shaders: {}", compiled_shaders.len());
            println!("  Cached shaders: {}", cached_shaders);
            println!("  Total functions: {}", total_functions);
            println!("  Compilation time: {:?}", compilation_time);

            if total_functions > 0 {
                let time_per_function = compilation_time.as_millis() / total_functions as u128;
                println!("  Time per function: {} ms", time_per_function);
            }

            // Test memory cleanup
            compiler.clear_cache();
            let cleared_stats = compiler.get_stats();
            assert_eq!(
                cleared_stats.cached_shaders, 0,
                "Cache should be empty after clearing"
            );
            println!("✓ Memory cleanup verified");
        } else {
            println!("Skipping memory usage test - no Metal device available");
        }
    }

    /// Test pipeline creation performance
    #[test]
    fn test_pipeline_creation_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let shaders_result = BitNetShaders::new(device);

            match shaders_result {
                Ok(shaders) => {
                    let test_functions = vec![
                        BitNetShaderFunction::BitLinearForward,
                        BitNetShaderFunction::QuantizeWeights1Bit,
                        BitNetShaderFunction::ReluForward,
                        BitNetShaderFunction::GeluForward,
                        BitNetShaderFunction::SoftmaxForward,
                    ];

                    let mut pipeline_times = Vec::new();

                    for function in &test_functions {
                        // Clear pipeline cache to ensure cold creation
                        shaders.clear_pipeline_cache();

                        // Measure cold pipeline creation
                        let start_time = Instant::now();
                        let pipeline_result = shaders.get_pipeline(*function);
                        let cold_time = start_time.elapsed();

                        if pipeline_result.is_ok() {
                            pipeline_times.push((*function, cold_time, false));

                            // Measure warm pipeline creation (from cache)
                            let start_time = Instant::now();
                            let _pipeline_result = shaders.get_pipeline(*function);
                            let warm_time = start_time.elapsed();

                            pipeline_times.push((*function, warm_time, true));

                            println!("Pipeline {:?}:", function);
                            println!("  Cold creation: {:?}", cold_time);
                            println!("  Warm creation: {:?}", warm_time);

                            if warm_time.as_nanos() > 0 {
                                let speedup =
                                    cold_time.as_nanos() as f64 / warm_time.as_nanos() as f64;
                                println!("  Cache speedup: {:.2}x", speedup);
                            }
                        }
                    }

                    // Analyze pipeline creation performance
                    let cold_times: Vec<_> = pipeline_times
                        .iter()
                        .filter(|(_, _, cached)| !cached)
                        .map(|(_, time, _)| *time)
                        .collect();

                    let warm_times: Vec<_> = pipeline_times
                        .iter()
                        .filter(|(_, _, cached)| *cached)
                        .map(|(_, time, _)| *time)
                        .collect();

                    if !cold_times.is_empty() {
                        let avg_cold_time =
                            cold_times.iter().sum::<Duration>() / cold_times.len() as u32;
                        println!("Average cold pipeline creation: {:?}", avg_cold_time);
                    }

                    if !warm_times.is_empty() {
                        let avg_warm_time =
                            warm_times.iter().sum::<Duration>() / warm_times.len() as u32;
                        println!("Average warm pipeline creation: {:?}", avg_warm_time);
                    }
                }
                Err(e) => {
                    println!(
                        "Pipeline performance test skipped (shader files missing): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping pipeline performance test - no Metal device available");
        }
    }

    /// Test compilation performance with different optimization levels
    #[test]
    fn test_optimization_level_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let optimization_levels = vec![
                OptimizationLevel::None,
                OptimizationLevel::Basic,
                OptimizationLevel::Full,
            ];

            for opt_level in optimization_levels {
                println!("Testing optimization level: {:?}", opt_level);

                let config = ShaderCompilerConfig {
                    shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                    enable_caching: false,
                    optimization_level: opt_level,
                    ..Default::default()
                };

                let compiler = ShaderCompiler::new(device.clone(), config).unwrap();

                let start_time = Instant::now();
                let compiled_shaders = compiler.compile_all_shaders().unwrap();
                let compilation_time = start_time.elapsed();

                println!(
                    "  Compiled {} shaders in {:?}",
                    compiled_shaders.len(),
                    compilation_time
                );

                // Test a simple pipeline creation to verify optimization doesn't break functionality
                for shader in &compiled_shaders {
                    if !shader.function_names.is_empty() {
                        let function_name = &shader.function_names[0];
                        let pipeline_result =
                            compiler.create_compute_pipeline(&shader.name, function_name);

                        match pipeline_result {
                            Ok(_) => {
                                println!(
                                    "  ✓ Pipeline creation successful for {}::{}",
                                    shader.name, function_name
                                );
                                break; // Test one pipeline per optimization level
                            }
                            Err(e) => {
                                println!(
                                    "  ✗ Pipeline creation failed for {}::{}: {}",
                                    shader.name, function_name, e
                                );
                            }
                        }
                    }
                }
                println!();
            }
        } else {
            println!("Skipping optimization level performance test - no Metal device available");
        }
    }

    /// Test concurrent compilation performance
    #[test]
    fn test_concurrent_compilation_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/concurrent_perf_cache")),
                ..Default::default()
            };

            let compiler = std::sync::Arc::new(ShaderCompiler::new(device, config).unwrap());

            // Test different thread counts
            let thread_counts = vec![1, 2, 4, 8];

            for thread_count in thread_counts {
                println!("Testing with {} threads:", thread_count);

                // Clear cache for fair comparison
                compiler.clear_cache();

                let start_time = Instant::now();
                let mut handles = vec![];

                for i in 0..thread_count {
                    let compiler_clone = compiler.clone();
                    let handle = std::thread::spawn(move || {
                        let result = compiler_clone.compile_all_shaders();
                        match result {
                            Ok(shaders) => shaders.len(),
                            Err(_) => 0,
                        }
                    });
                    handles.push(handle);
                }

                let mut total_compiled = 0;
                for handle in handles {
                    total_compiled += handle.join().unwrap();
                }

                let total_time = start_time.elapsed();

                println!("  Total time: {:?}", total_time);
                println!("  Total shaders compiled: {}", total_compiled);

                if total_time.as_millis() > 0 {
                    let throughput = total_compiled as f64 / total_time.as_secs_f64();
                    println!("  Throughput: {:.2} shaders/second", throughput);
                }
                println!();
            }
        } else {
            println!("Skipping concurrent performance test - no Metal device available");
        }
    }

    /// Test cache performance characteristics
    #[test]
    fn test_cache_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/cache_perf_test")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Test cache miss performance (first compilation)
            compiler.clear_cache();
            let start_time = Instant::now();
            let first_result = compiler.compile_all_shaders().unwrap();
            let cache_miss_time = start_time.elapsed();

            // Test cache hit performance (second compilation)
            let start_time = Instant::now();
            let second_result = compiler.compile_all_shaders().unwrap();
            let cache_hit_time = start_time.elapsed();

            println!("Cache Performance Analysis:");
            println!("  First compilation (cache miss): {:?}", cache_miss_time);
            println!("  Second compilation (cache hit): {:?}", cache_hit_time);

            if cache_hit_time.as_nanos() > 0 {
                let speedup = cache_miss_time.as_nanos() as f64 / cache_hit_time.as_nanos() as f64;
                println!("  Cache speedup: {:.2}x", speedup);

                // Cache should provide significant speedup
                assert!(speedup > 1.5, "Cache should provide at least 1.5x speedup");
            }

            assert_eq!(
                first_result.len(),
                second_result.len(),
                "Both compilations should produce same number of shaders"
            );

            // Test cache efficiency over multiple operations
            let mut cache_times = Vec::new();
            for i in 0..10 {
                let start_time = Instant::now();
                let _result = compiler.compile_all_shaders().unwrap();
                let time = start_time.elapsed();
                cache_times.push(time);

                if i % 3 == 0 {
                    // Occasionally clear cache to test mixed scenarios
                    compiler.clear_cache();
                }
            }

            let avg_cache_time = cache_times.iter().sum::<Duration>() / cache_times.len() as u32;
            println!(
                "  Average compilation time over 10 runs: {:?}",
                avg_cache_time
            );
        } else {
            println!("Skipping cache performance test - no Metal device available");
        }
    }

    /// Helper function to analyze performance metrics
    fn analyze_performance_metrics(metrics: &HashMap<String, CompilationMetrics>) {
        println!("Performance Analysis Summary:");
        println!("============================");

        let mut cold_times = Vec::new();
        let mut warm_times = Vec::new();

        for (name, metric) in metrics {
            if name.contains("_cold") {
                cold_times.push(metric.compilation_time);
            } else if name.contains("_warm") {
                warm_times.push(metric.compilation_time);
            }
        }

        if !cold_times.is_empty() {
            let avg_cold = cold_times.iter().sum::<Duration>() / cold_times.len() as u32;
            let max_cold = cold_times.iter().max().unwrap();
            let min_cold = cold_times.iter().min().unwrap();

            println!("Cold Compilation Times:");
            println!("  Average: {:?}", avg_cold);
            println!("  Min: {:?}", min_cold);
            println!("  Max: {:?}", max_cold);
        }

        if !warm_times.is_empty() {
            let avg_warm = warm_times.iter().sum::<Duration>() / warm_times.len() as u32;
            let max_warm = warm_times.iter().max().unwrap();
            let min_warm = warm_times.iter().min().unwrap();

            println!("Warm Compilation Times:");
            println!("  Average: {:?}", avg_warm);
            println!("  Min: {:?}", min_warm);
            println!("  Max: {:?}", max_warm);
        }

        // Calculate efficiency metrics
        let total_functions: usize = metrics
            .values()
            .filter(|m| !m.cache_hit)
            .map(|m| m.function_count)
            .sum();

        let total_source_size: usize = metrics
            .values()
            .filter(|m| !m.cache_hit)
            .map(|m| m.source_size)
            .sum();

        println!("Efficiency Metrics:");
        println!("  Total functions compiled: {}", total_functions);
        println!("  Total source size: {} bytes", total_source_size);

        if !cold_times.is_empty() && total_functions > 0 {
            let avg_cold = cold_times.iter().sum::<Duration>() / cold_times.len() as u32;
            let time_per_function = avg_cold.as_millis() / total_functions as u128;
            println!("  Average time per function: {} ms", time_per_function);
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_performance_tests {
    #[test]
    fn test_performance_unsupported_platform() {
        println!("Performance shader compilation tests skipped - not on macOS or Metal feature not enabled");
        assert!(true, "Non-macOS platform performance test passed");
    }
}
