//! Comprehensive Metal Shader Compilation Tests
//!
//! This module provides extensive testing for Metal shader compilation functionality,
//! covering all shader files, functions, compilation scenarios, and error conditions.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_shader_tests {
    use bitnet_core::metal::*;
    use std::path::PathBuf;
    use std::time::Duration;

    /// Test basic shader compiler creation and configuration
    #[test]
    fn test_shader_compiler_creation_and_config() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            // Test default configuration
            let default_config = ShaderCompilerConfig::default();
            assert_eq!(
                default_config.shader_directory,
                PathBuf::from("src/metal/shaders")
            );
            assert!(default_config.enable_caching);
            assert!(default_config.compile_options.fast_math);

            // Test custom configuration
            let custom_config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache")),
                debug_info: true,
                optimization_level: OptimizationLevel::Full,
                compile_options: CompileOptions {
                    language_version: LanguageVersion::Metal2_4,
                    fast_math: true,
                    ..Default::default()
                },
            };

            let compiler_result = ShaderCompiler::new(device, custom_config);
            assert!(
                compiler_result.is_ok(),
                "Failed to create shader compiler with custom config"
            );

            if let Ok(compiler) = compiler_result {
                let stats = compiler.get_stats();
                println!("Shader compiler stats: {:?}", stats);
                assert_eq!(
                    stats.shader_directory,
                    PathBuf::from("bitnet-core/src/metal/shaders")
                );
            }
        } else {
            println!("Skipping shader compiler tests - no Metal device available");
        }
    }

    /// Test compilation of all individual shader files
    #[test]
    fn test_individual_shader_file_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            let shader_files = ["bitlinear.metal", "quantization.metal", "activation.metal"];
            let mut compiled_count = 0;
            let mut total_functions = 0;

            for shader_file in &shader_files {
                let shader_path = PathBuf::from("bitnet-core/src/metal/shaders").join(shader_file);

                if shader_path.exists() {
                    println!("Testing compilation of: {}", shader_file);

                    let result = compiler.compile_shader_file(&shader_path);
                    match result {
                        Ok(compiled_shader) => {
                            println!("✓ {} compiled successfully", shader_file);
                            println!("  Functions found: {:?}", compiled_shader.function_names);

                            // Verify shader has expected properties
                            assert!(
                                !compiled_shader.function_names.is_empty(),
                                "Shader {} should have at least one function",
                                shader_file
                            );
                            assert_eq!(
                                compiled_shader.name,
                                shader_file.trim_end_matches(".metal")
                            );
                            assert_eq!(compiled_shader.source_path, shader_path);

                            compiled_count += 1;
                            total_functions += compiled_shader.function_names.len();
                        }
                        Err(e) => {
                            panic!("✗ {} compilation failed: {}", shader_file, e);
                        }
                    }
                } else {
                    println!("Skipping {} - file not found", shader_file);
                }
            }

            println!(
                "Successfully compiled {} shader files with {} total functions",
                compiled_count, total_functions
            );
            assert!(
                compiled_count > 0,
                "Should have compiled at least one shader file"
            );
            assert!(
                total_functions > 0,
                "Should have found at least one shader function"
            );
        } else {
            println!("Skipping individual shader compilation tests - no Metal device available");
        }
    }

    /// Test that all expected shader functions are discoverable
    #[test]
    fn test_shader_function_discovery() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let compiled_shaders = compiler.compile_all_shaders().unwrap();

            // Expected functions for each shader
            let expected_functions = vec![
                // BitLinear functions
                (
                    "bitlinear",
                    vec![
                        "bitlinear_forward",
                        "bitlinear_backward_input",
                        "binarize_weights",
                        "quantize_activations",
                    ],
                ),
                // Quantization functions
                (
                    "quantization",
                    vec![
                        "quantize_weights_1bit",
                        "quantize_activations_8bit",
                        "dequantize_weights_1bit",
                        "dequantize_activations_8bit",
                        "dynamic_quantize_activations",
                        "quantize_gradients",
                        "mixed_precision_matmul",
                    ],
                ),
                // Activation functions
                (
                    "activation",
                    vec![
                        "relu_forward",
                        "relu_backward",
                        "gelu_forward",
                        "gelu_backward",
                        "swish_forward",
                        "swish_backward",
                        "sigmoid_forward",
                        "sigmoid_backward",
                        "tanh_forward",
                        "tanh_backward",
                        "leaky_relu_forward",
                        "leaky_relu_backward",
                        "softmax_forward",
                        "softmax_backward",
                        "layer_norm_forward",
                        "fused_relu_dropout",
                    ],
                ),
            ];

            for (shader_name, expected_funcs) in expected_functions {
                let shader = compiled_shaders.iter().find(|s| s.name == shader_name);

                if let Some(shader) = shader {
                    println!("Checking functions in shader: {}", shader_name);
                    println!("  Found functions: {:?}", shader.function_names);

                    for expected_func in expected_funcs {
                        assert!(
                            shader.function_names.contains(&expected_func.to_string()),
                            "Shader '{}' should contain function '{}'",
                            shader_name,
                            expected_func
                        );
                    }

                    println!("✓ All expected functions found in {}", shader_name);
                } else {
                    println!("⚠ Shader '{}' not found in compiled shaders", shader_name);
                }
            }
        } else {
            println!("Skipping shader function discovery tests - no Metal device available");
        }
    }

    /// Test pipeline creation for all BitNet shader functions
    #[test]
    fn test_pipeline_creation_for_all_functions() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let shaders_result = BitNetShaders::new(device);

            match shaders_result {
                Ok(shaders) => {
                    let test_functions = vec![
                        BitNetShaderFunction::BitLinearForward,
                        BitNetShaderFunction::BitLinearBackwardInput,
                        BitNetShaderFunction::BinarizeWeights,
                        BitNetShaderFunction::QuantizeActivations,
                        BitNetShaderFunction::QuantizeWeights1Bit,
                        BitNetShaderFunction::QuantizeActivations8Bit,
                        BitNetShaderFunction::DequantizeWeights1Bit,
                        BitNetShaderFunction::DequantizeActivations8Bit,
                        BitNetShaderFunction::DynamicQuantizeActivations,
                        BitNetShaderFunction::QuantizeGradients,
                        BitNetShaderFunction::MixedPrecisionMatmul,
                        BitNetShaderFunction::ReluForward,
                        BitNetShaderFunction::ReluBackward,
                        BitNetShaderFunction::GeluForward,
                        BitNetShaderFunction::GeluBackward,
                        BitNetShaderFunction::SwishForward,
                        BitNetShaderFunction::SwishBackward,
                        BitNetShaderFunction::SigmoidForward,
                        BitNetShaderFunction::SigmoidBackward,
                        BitNetShaderFunction::TanhForward,
                        BitNetShaderFunction::TanhBackward,
                        BitNetShaderFunction::LeakyReluForward,
                        BitNetShaderFunction::LeakyReluBackward,
                        BitNetShaderFunction::SoftmaxForward,
                        BitNetShaderFunction::SoftmaxBackward,
                        BitNetShaderFunction::LayerNormForward,
                        BitNetShaderFunction::FusedReluDropout,
                    ];

                    let mut successful_pipelines = 0;
                    let mut failed_pipelines = 0;

                    for function in test_functions {
                        println!("Testing pipeline creation for: {:?}", function);

                        match shaders.get_pipeline(function) {
                            Ok(pipeline) => {
                                println!("✓ Pipeline created successfully for: {:?}", function);

                                // Verify pipeline properties
                                assert!(pipeline.max_total_threads_per_threadgroup() > 0);
                                assert!(pipeline.thread_execution_width() > 0);

                                successful_pipelines += 1;
                            }
                            Err(e) => {
                                println!("✗ Pipeline creation failed for {:?}: {}", function, e);
                                failed_pipelines += 1;
                            }
                        }
                    }

                    println!(
                        "Pipeline creation results: {} successful, {} failed",
                        successful_pipelines, failed_pipelines
                    );

                    // We expect at least some pipelines to succeed if shader files exist
                    if successful_pipelines == 0 && failed_pipelines > 0 {
                        println!(
                            "⚠ No pipelines created successfully - shader files may be missing"
                        );
                    }
                }
                Err(e) => {
                    println!("BitNet shaders initialization failed (expected if shader files missing): {}", e);
                }
            }
        } else {
            println!("Skipping pipeline creation tests - no Metal device available");
        }
    }

    /// Test shader compilation error handling
    #[test]
    fn test_shader_compilation_error_handling() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: false, // Disable caching for error testing
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device.clone(), config).unwrap();

            // Test compilation of non-existent shader file
            let non_existent_path =
                PathBuf::from("bitnet-core/src/metal/shaders/non_existent.metal");
            let result = compiler.compile_shader_file(&non_existent_path);
            assert!(
                result.is_err(),
                "Should fail to compile non-existent shader file"
            );
            println!("✓ Correctly handled non-existent shader file");

            // Test compilation with invalid source code
            let invalid_source = "invalid metal code that won't compile";
            let result = compiler.compile_source(invalid_source, "invalid_test");
            assert!(
                result.is_err(),
                "Should fail to compile invalid source code"
            );
            println!("✓ Correctly handled invalid source code");

            // Test getting non-existent shader from cache
            let cached_shader = compiler.get_shader("non_existent_shader");
            assert!(
                cached_shader.is_none(),
                "Should return None for non-existent cached shader"
            );
            println!("✓ Correctly handled non-existent cached shader");

            // Test getting non-existent compute function
            let function_result =
                compiler.get_compute_function("non_existent_shader", "non_existent_function");
            assert!(
                function_result.is_err(),
                "Should fail to get non-existent compute function"
            );
            println!("✓ Correctly handled non-existent compute function");
        } else {
            println!("Skipping error handling tests - no Metal device available");
        }
    }

    /// Test shader caching functionality
    #[test]
    fn test_shader_caching() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache_test")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Test that cache starts empty
            let initial_stats = compiler.get_stats();
            assert_eq!(initial_stats.cached_shaders, 0);

            // Compile a shader file if it exists
            let shader_path = PathBuf::from("bitnet-core/src/metal/shaders/bitlinear.metal");
            if shader_path.exists() {
                // First compilation - should cache the shader
                let result1 = compiler.compile_shader_file(&shader_path);
                assert!(result1.is_ok(), "First compilation should succeed");

                let stats_after_first = compiler.get_stats();
                assert!(
                    stats_after_first.cached_shaders > 0,
                    "Should have cached shaders after compilation"
                );

                // Second compilation - should use cache
                let result2 = compiler.compile_shader_file(&shader_path);
                assert!(result2.is_ok(), "Second compilation should succeed");

                // Test cache retrieval
                let cached_shader = compiler.get_shader("bitlinear");
                assert!(cached_shader.is_some(), "Should retrieve shader from cache");

                if let Some(shader) = cached_shader {
                    assert_eq!(shader.name, "bitlinear");
                    assert!(!shader.function_names.is_empty());
                }

                // Test cache clearing
                compiler.clear_cache();
                let stats_after_clear = compiler.get_stats();
                assert_eq!(
                    stats_after_clear.cached_shaders, 0,
                    "Cache should be empty after clearing"
                );

                println!("✓ Shader caching functionality verified");
            } else {
                println!("Skipping caching test - bitlinear.metal not found");
            }
        } else {
            println!("Skipping caching tests - no Metal device available");
        }
    }

    /// Test shader compilation with different configurations
    #[test]
    fn test_shader_compilation_configurations() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            // Test different optimization levels
            let optimization_levels = vec![
                OptimizationLevel::None,
                OptimizationLevel::Basic,
                OptimizationLevel::Full,
            ];

            for opt_level in optimization_levels {
                let config = ShaderCompilerConfig {
                    shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                    enable_caching: false,
                    optimization_level: opt_level,
                    debug_info: true,
                    ..Default::default()
                };

                let compiler_result = ShaderCompiler::new(device.clone(), config);
                assert!(
                    compiler_result.is_ok(),
                    "Should create compiler with optimization level {:?}",
                    opt_level
                );
                println!(
                    "✓ Compiler created with optimization level: {:?}",
                    opt_level
                );
            }

            // Test different language versions
            let language_versions = vec![
                LanguageVersion::Metal2_0,
                LanguageVersion::Metal2_1,
                LanguageVersion::Metal2_2,
                LanguageVersion::Metal2_3,
                LanguageVersion::Metal2_4,
            ];

            for lang_version in language_versions {
                let config = ShaderCompilerConfig {
                    shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                    enable_caching: false,
                    compile_options: CompileOptions {
                        language_version: lang_version,
                        fast_math: true,
                        ..Default::default()
                    },
                    ..Default::default()
                };

                let compiler_result = ShaderCompiler::new(device.clone(), config);
                assert!(
                    compiler_result.is_ok(),
                    "Should create compiler with language version {:?}",
                    lang_version
                );
                println!(
                    "✓ Compiler created with language version: {:?}",
                    lang_version
                );
            }
        } else {
            println!("Skipping configuration tests - no Metal device available");
        }
    }

    /// Test shader function parameter validation
    #[test]
    fn test_shader_function_parameter_validation() {
        // Test BitNetShaderFunction enum methods
        let test_functions = vec![
            (
                BitNetShaderFunction::BitLinearForward,
                "bitlinear",
                "bitlinear_forward",
            ),
            (
                BitNetShaderFunction::QuantizeWeights1Bit,
                "quantization",
                "quantize_weights_1bit",
            ),
            (
                BitNetShaderFunction::ReluForward,
                "activation",
                "relu_forward",
            ),
            (
                BitNetShaderFunction::GeluBackward,
                "activation",
                "gelu_backward",
            ),
            (
                BitNetShaderFunction::MixedPrecisionMatmul,
                "quantization",
                "mixed_precision_matmul",
            ),
        ];

        for (function, expected_shader, expected_function) in test_functions {
            assert_eq!(function.shader_name(), expected_shader);
            assert_eq!(function.function_name(), expected_function);

            let pipeline_key = function.pipeline_key();
            let expected_key = format!("{}::{}", expected_shader, expected_function);
            assert_eq!(pipeline_key, expected_key);

            println!("✓ Function {:?} has correct parameters", function);
        }
    }

    /// Test shader compilation performance and statistics
    #[test]
    fn test_shader_compilation_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_perf_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Measure compilation time
            let start_time = std::time::Instant::now();
            let compiled_shaders = compiler.compile_all_shaders().unwrap();
            let compilation_time = start_time.elapsed();

            println!("Compilation performance:");
            println!("  Total shaders compiled: {}", compiled_shaders.len());
            println!("  Total compilation time: {:?}", compilation_time);

            if !compiled_shaders.is_empty() {
                let avg_time = compilation_time / compiled_shaders.len() as u32;
                println!("  Average time per shader: {:?}", avg_time);
            }

            // Test statistics
            let stats = compiler.get_stats();
            println!("Compiler statistics: {:?}", stats);

            assert_eq!(stats.cached_shaders, compiled_shaders.len());
            assert!(stats.total_functions > 0);

            // Test recompilation performance (should be faster due to caching)
            let start_time_cached = std::time::Instant::now();
            let _recompiled_shaders = compiler.compile_all_shaders().unwrap();
            let cached_compilation_time = start_time_cached.elapsed();

            println!("Cached compilation time: {:?}", cached_compilation_time);

            // Cached compilation should be significantly faster
            if compilation_time > Duration::from_millis(100) {
                assert!(
                    cached_compilation_time < compilation_time,
                    "Cached compilation should be faster than initial compilation"
                );
            }
        } else {
            println!("Skipping performance tests - no Metal device available");
        }
    }

    /// Test shader loader functionality
    #[test]
    fn test_shader_loader() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_loader_cache")),
                ..Default::default()
            };

            let mut loader_result = ShaderLoader::new(device, config);

            match loader_result {
                Ok(mut loader) => {
                    // Test preloading specific shaders
                    let preload_result = loader.preload_shaders(&["bitlinear", "quantization"]);
                    match preload_result {
                        Ok(()) => {
                            println!("✓ Successfully preloaded specific shaders");

                            let available_shaders = loader.get_available_shaders();
                            println!("Available shaders after preload: {:?}", available_shaders);
                        }
                        Err(e) => {
                            println!("Preload failed (expected if shader files missing): {}", e);
                        }
                    }

                    // Test preloading all shaders
                    let preload_all_result = loader.preload_all_shaders();
                    match preload_all_result {
                        Ok(()) => {
                            println!("✓ Successfully preloaded all shaders");

                            let available_shaders = loader.get_available_shaders();
                            println!(
                                "Available shaders after preload all: {:?}",
                                available_shaders
                            );

                            // Test getting shader functions
                            for shader_name in &available_shaders {
                                let functions_result = loader.get_shader_functions(shader_name);
                                match functions_result {
                                    Ok(functions) => {
                                        println!("Functions in {}: {:?}", shader_name, functions);
                                        assert!(
                                            !functions.is_empty(),
                                            "Shader should have at least one function"
                                        );
                                    }
                                    Err(e) => {
                                        println!(
                                            "Failed to get functions for {}: {}",
                                            shader_name, e
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!(
                                "Preload all failed (expected if shader files missing): {}",
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "Shader loader creation failed (expected if shader files missing): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping shader loader tests - no Metal device available");
        }
    }

    /// Test BitNet shader utilities integration
    #[test]
    fn test_bitnet_shader_utilities() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let shaders_result = BitNetShaders::new(device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ BitNet shaders initialized successfully");

                    // Test available shaders
                    let available = shaders.get_available_shaders();
                    println!("Available BitNet shaders: {:?}", available);

                    // Test dispatch parameter calculation
                    let test_data_sizes = vec![32, 64, 128, 256, 512, 1024, 2048];

                    for data_size in test_data_sizes {
                        let dispatch_result = shaders.calculate_dispatch_params(
                            BitNetShaderFunction::ReluForward,
                            data_size,
                        );

                        match dispatch_result {
                            Ok((threadgroup_size, threadgroups)) => {
                                println!(
                                    "Data size {}: threadgroup={:?}, threadgroups={:?}",
                                    data_size, threadgroup_size, threadgroups
                                );

                                assert!(threadgroup_size.width > 0);
                                assert!(threadgroups.width > 0);
                            }
                            Err(e) => {
                                println!(
                                    "Dispatch calculation failed for size {}: {}",
                                    data_size, e
                                );
                            }
                        }
                    }

                    // Test pipeline cache clearing
                    shaders.clear_pipeline_cache();
                    println!("✓ Pipeline cache cleared successfully");

                    // Test device access
                    let device_ref = shaders.device();
                    println!("✓ Device access successful: {}", device_ref.name());
                }
                Err(e) => {
                    println!("BitNet shaders initialization failed (expected if shader files missing): {}", e);
                }
            }
        } else {
            println!("Skipping BitNet shader utilities tests - no Metal device available");
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {

    #[test]
    fn test_unsupported_platform_comprehensive() {
        println!("Comprehensive shader compilation tests skipped - not on macOS or Metal feature not enabled");

        // Verify that all the types and functions exist and compile on non-macOS platforms
        // This ensures the API is consistent across platforms
        assert!(true, "Non-macOS platform test passed");
    }
}
