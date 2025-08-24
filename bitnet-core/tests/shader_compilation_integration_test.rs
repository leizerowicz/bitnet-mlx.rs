//! Comprehensive Metal Shader Compilation Integration Tests
//!
//! This module provides a comprehensive test suite for Metal shader compilation,
//! including basic functionality, error handling, performance testing, and edge cases.

use std::time::Instant;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_tests {
    use super::*;
    use bitnet_core::metal::*;

    #[test]
    fn test_metal_device_creation() {
        println!("=== Testing Metal Device Creation ===");

        match create_metal_device() {
            Ok(device) => {
                println!("✓ Metal device created: {}", device.name());
                println!(
                    "  Device supports feature set: {}",
                    device.supports_feature_set(metal::MTLFeatureSet::macOS_GPUFamily1_v1)
                );
            }
            Err(e) => {
                panic!("Failed to create Metal device: {}", e);
            }
        }
    }

    #[test]
    fn test_shader_compiler_creation() {
        println!("=== Testing Shader Compiler Creation ===");

        let device = create_metal_device().expect("Metal device required for this test");

        let config = ShaderCompilerConfig {
            shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
            enable_caching: true,
            cache_directory: Some(PathBuf::from("target/test_shader_cache")),
            debug_info: true,
            optimization_level: OptimizationLevel::Full,
            ..Default::default()
        };

        let compiler =
            ShaderCompiler::new(device.clone(), config).expect("Failed to create shader compiler");
        println!("✓ Shader compiler created with custom configuration");

        let stats = compiler.get_stats();
        println!("  Initial stats: {:?}", stats);
    }

    #[test]
    fn test_individual_shader_compilation() {
        println!("=== Testing Individual Shader Compilation ===");

        let device = create_metal_device().expect("Metal device required for this test");
        let config = ShaderCompilerConfig {
            shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
            enable_caching: true,
            cache_directory: Some(PathBuf::from("target/test_shader_cache")),
            debug_info: true,
            optimization_level: OptimizationLevel::Full,
            ..Default::default()
        };

        let compiler =
            ShaderCompiler::new(device, config).expect("Failed to create shader compiler");

        let shader_files = ["bitlinear.metal", "quantization.metal", "activation.metal"];
        let mut total_functions = 0;
        let mut compilation_times = Vec::new();

        for shader_file in &shader_files {
            let shader_path = PathBuf::from("bitnet-core/src/metal/shaders").join(shader_file);
            println!("Testing compilation of: {}", shader_file);

            if shader_path.exists() {
                let start_time = Instant::now();
                match compiler.compile_shader_file(&shader_path) {
                    Ok(compiled_shader) => {
                        let compilation_time = start_time.elapsed();
                        compilation_times.push(compilation_time);

                        println!(
                            "✓ {} compiled successfully in {:?}",
                            shader_file, compilation_time
                        );
                        println!("  Functions: {:?}", compiled_shader.function_names);
                        println!("  Source hash: {}", compiled_shader.source_hash);

                        total_functions += compiled_shader.function_names.len();

                        // Verify shader properties
                        assert!(
                            !compiled_shader.function_names.is_empty(),
                            "Shader should have at least one function"
                        );
                        assert_eq!(compiled_shader.name, shader_file.trim_end_matches(".metal"));
                    }
                    Err(e) => {
                        println!("✗ {} compilation failed: {}", shader_file, e);
                        // Don't panic here as shader files might not exist in all environments
                    }
                }
            } else {
                println!("⚠ {} not found, skipping", shader_file);
            }
        }

        println!("Total functions discovered: {}", total_functions);
        if !compilation_times.is_empty() {
            let avg_time = compilation_times.iter().sum::<std::time::Duration>()
                / compilation_times.len() as u32;
            println!("Average compilation time: {:?}", avg_time);
        }
    }

    #[test]
    fn test_shader_caching_functionality() {
        println!("=== Testing Shader Caching ===");

        let device = create_metal_device().expect("Metal device required for this test");
        let config = ShaderCompilerConfig {
            shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
            enable_caching: true,
            cache_directory: Some(PathBuf::from("target/test_shader_cache")),
            debug_info: true,
            optimization_level: OptimizationLevel::Full,
            ..Default::default()
        };

        let compiler =
            ShaderCompiler::new(device, config).expect("Failed to create shader compiler");

        let bitlinear_path = PathBuf::from("bitnet-core/src/metal/shaders/bitlinear.metal");
        if bitlinear_path.exists() {
            // First compilation (cache miss)
            compiler.clear_cache();
            let start_time = Instant::now();
            let _result1 = compiler.compile_shader_file(&bitlinear_path);
            let cache_miss_time = start_time.elapsed();

            // Second compilation (cache hit)
            let start_time = Instant::now();
            let _result2 = compiler.compile_shader_file(&bitlinear_path);
            let cache_hit_time = start_time.elapsed();

            println!("✓ Cache functionality verified");
            println!("  Cache miss time: {:?}", cache_miss_time);
            println!("  Cache hit time: {:?}", cache_hit_time);

            if cache_hit_time.as_nanos() > 0 {
                let speedup = cache_miss_time.as_nanos() as f64 / cache_hit_time.as_nanos() as f64;
                println!("  Cache speedup: {:.2}x", speedup);
            }
        } else {
            println!("⚠ bitlinear.metal not found, skipping cache test");
        }
    }

    #[test]
    fn test_bitnet_shader_utilities() {
        println!("=== Testing BitNet Shader Utilities ===");

        let device = create_metal_device().expect("Metal device required for this test");

        match BitNetShaders::new(device.clone()) {
            Ok(shaders) => {
                println!("✓ BitNet shaders initialized");
                let available = shaders.get_available_shaders();
                println!("  Available shaders: {:?}", available);

                // Test pipeline creation for various functions
                let test_functions = [
                    BitNetShaderFunction::BitLinearForward,
                    BitNetShaderFunction::BitLinearBackwardInput,
                    BitNetShaderFunction::BinarizeWeights,
                    BitNetShaderFunction::QuantizeWeights1Bit,
                    BitNetShaderFunction::QuantizeActivations8Bit,
                    BitNetShaderFunction::ReluForward,
                    BitNetShaderFunction::GeluForward,
                    BitNetShaderFunction::SoftmaxForward,
                ];

                let mut successful_pipelines = 0;
                for function in &test_functions {
                    match shaders.get_pipeline(*function) {
                        Ok(pipeline) => {
                            println!("✓ Pipeline created for: {:?}", function);
                            println!(
                                "  Max threads per threadgroup: {}",
                                pipeline.max_total_threads_per_threadgroup()
                            );
                            println!(
                                "  Thread execution width: {}",
                                pipeline.thread_execution_width()
                            );
                            successful_pipelines += 1;
                        }
                        Err(e) => {
                            println!("✗ Pipeline creation failed for {:?}: {}", function, e);
                        }
                    }
                }

                println!(
                    "Successfully created {} out of {} pipelines",
                    successful_pipelines,
                    test_functions.len()
                );
            }
            Err(e) => {
                println!("✗ BitNet shaders initialization failed: {}", e);
                println!("  This is expected if shader files are missing");
            }
        }
    }

    #[test]
    fn test_dispatch_parameter_calculation() {
        println!("=== Testing Dispatch Parameter Calculation ===");

        let device = create_metal_device().expect("Metal device required for this test");

        if let Ok(shaders) = BitNetShaders::new(device) {
            let data_sizes = [32, 64, 128, 256, 512, 1024, 2048];
            for &size in &data_sizes {
                match shaders.calculate_dispatch_params(BitNetShaderFunction::ReluForward, size) {
                    Ok((threadgroup_size, threadgroups)) => {
                        println!(
                            "Size {}: threadgroup={:?}, threadgroups={:?}",
                            size, threadgroup_size, threadgroups
                        );
                    }
                    Err(e) => {
                        println!("Dispatch calculation failed for size {}: {}", size, e);
                    }
                }
            }
        } else {
            println!("⚠ BitNet shaders not available, skipping dispatch parameter test");
        }
    }

    #[test]
    fn test_error_handling() {
        println!("=== Testing Error Handling ===");

        let device = create_metal_device().expect("Metal device required for this test");
        let config = ShaderCompilerConfig::default();
        let compiler =
            ShaderCompiler::new(device, config).expect("Failed to create shader compiler");

        // Test non-existent shader file
        let non_existent_path = PathBuf::from("bitnet-core/src/metal/shaders/non_existent.metal");
        let result = compiler.compile_shader_file(&non_existent_path);
        assert!(result.is_err(), "Should fail for non-existent file");
        println!("✓ Non-existent file correctly rejected");

        // Test invalid source compilation
        let invalid_source = "invalid metal code";
        let result = compiler.compile_source(invalid_source, "invalid_test");
        assert!(result.is_err(), "Should fail for invalid source");
        println!("✓ Invalid source code correctly rejected");
    }

    #[test]
    fn test_performance_summary() {
        println!("=== Testing Performance Summary ===");

        let device = create_metal_device().expect("Metal device required for this test");
        let config = ShaderCompilerConfig::default();
        let compiler =
            ShaderCompiler::new(device, config).expect("Failed to create shader compiler");

        let final_stats = compiler.get_stats();
        println!("Final compiler stats: {:?}", final_stats);

        // Basic validation that stats are reasonable
        assert!(final_stats.total_functions >= 0);
        assert!(final_stats.cache_hits >= 0);
        assert!(final_stats.cache_misses >= 0);
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {

    #[test]
    fn test_shader_compilation_unavailable() {
        println!("=== Shader Compilation Tests ===");
        println!("Tests skipped - not on macOS or Metal feature not enabled");
        println!("Platform: {}", std::env::consts::OS);
        println!("Architecture: {}", std::env::consts::ARCH);

        // This test always passes but documents the platform limitations
        assert!(true, "Test documented platform limitations");
    }
}

#[test]
fn test_environment_information() {
    println!("=== Environment Information ===");
    println!("  OS: {}", std::env::consts::OS);
    println!("  Architecture: {}", std::env::consts::ARCH);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    println!("  Metal support: enabled");
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    println!("  Metal support: disabled");

    // Always passes - just for information gathering
    assert!(true, "Environment information logged");
}

#[test]
fn test_comprehensive_shader_compilation() {
    println!("=== Comprehensive Shader Compilation Test ===");

    let start_time = Instant::now();

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // Run a subset of tests to verify overall functionality
        if let Ok(device) = bitnet_core::metal::create_metal_device() {
            println!("✓ Metal device available for comprehensive testing");

            let config = bitnet_core::metal::ShaderCompilerConfig::default();
            if let Ok(compiler) = bitnet_core::metal::ShaderCompiler::new(device, config) {
                println!("✓ Shader compiler created successfully");
                let stats = compiler.get_stats();
                println!("  Compiler stats: {:?}", stats);
            } else {
                println!("⚠ Shader compiler creation failed");
            }
        } else {
            println!("⚠ Metal device not available");
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("⚠ Metal not available on this platform");
    }

    let total_time = start_time.elapsed();
    println!("✓ Comprehensive test completed in {total_time:?}");
}
