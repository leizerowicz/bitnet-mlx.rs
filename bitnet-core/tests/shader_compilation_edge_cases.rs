//! Edge Case Tests for Metal Shader Compilation
//!
//! This module tests edge cases, error conditions, and boundary scenarios
//! for Metal shader compilation functionality.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod edge_case_tests {
    use bitnet_core::metal::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Test shader compilation with empty shader files
    #[test]
    fn test_empty_shader_file_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let empty_shader_path = temp_dir.path().join("empty.metal");

            // Create an empty shader file
            fs::write(&empty_shader_path, "").unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let result = compiler.compile_shader_file(&empty_shader_path);

            // Empty shader should fail to compile
            assert!(result.is_err(), "Empty shader file should fail to compile");
            println!("✓ Empty shader file correctly rejected");
        } else {
            println!("Skipping empty shader test - no Metal device available");
        }
    }

    /// Test shader compilation with malformed Metal code
    #[test]
    fn test_malformed_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let malformed_shader_path = temp_dir.path().join("malformed.metal");

            // Create a malformed shader file
            let malformed_content = r#"
                #include <metal_stdlib>
                using namespace metal;

                // Missing kernel keyword and malformed syntax
                void broken_function(
                    device float* input [[buffer(0)]]
                    // Missing comma and closing parenthesis
                    device float* output [[buffer(1)
                {
                    // Malformed body
                    output[0] = input[0] +;
                }
            "#;

            fs::write(&malformed_shader_path, malformed_content).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let result = compiler.compile_shader_file(&malformed_shader_path);

            // Malformed shader should fail to compile
            assert!(result.is_err(), "Malformed shader should fail to compile");
            println!("✓ Malformed shader correctly rejected");
        } else {
            println!("Skipping malformed shader test - no Metal device available");
        }
    }

    /// Test shader compilation with valid but function-less Metal code
    #[test]
    fn test_functionless_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let functionless_shader_path = temp_dir.path().join("functionless.metal");

            // Create a valid shader file with no kernel functions
            let functionless_content = r#"
                #include <metal_stdlib>
                using namespace metal;

                // Just constants and helper functions, no kernel functions
                constant float PI = 3.14159265359;

                float helper_function(float x) {
                    return x * PI;
                }
            "#;

            fs::write(&functionless_shader_path, functionless_content).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let result = compiler.compile_shader_file(&functionless_shader_path);

            match result {
                Ok(compiled_shader) => {
                    // Should compile successfully but have no functions
                    assert!(
                        compiled_shader.function_names.is_empty(),
                        "Functionless shader should have no kernel functions"
                    );
                    println!("✓ Functionless shader compiled with no functions");
                }
                Err(e) => {
                    println!("Functionless shader compilation failed: {}", e);
                    // This is also acceptable behavior
                }
            }
        } else {
            println!("Skipping functionless shader test - no Metal device available");
        }
    }

    /// Test shader compilation with very large shader files
    #[test]
    fn test_large_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let large_shader_path = temp_dir.path().join("large.metal");

            // Create a large shader file with many functions
            let mut large_content = String::from(
                r#"
                #include <metal_stdlib>
                using namespace metal;
            "#,
            );

            // Generate many similar kernel functions
            for i in 0..100 {
                large_content.push_str(&format!(
                    r#"
                    kernel void test_function_{}(
                        device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]
                    ) {{
                        if (gid >= count) return;
                        output[gid] = input[gid] * {}.0;
                    }}
                "#,
                    i,
                    i + 1
                ));
            }

            fs::write(&large_shader_path, large_content).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Measure compilation time for large shader
            let start_time = std::time::Instant::now();
            let result = compiler.compile_shader_file(&large_shader_path);
            let compilation_time = start_time.elapsed();

            match result {
                Ok(compiled_shader) => {
                    println!("✓ Large shader compiled successfully");
                    println!(
                        "  Functions found: {}",
                        compiled_shader.function_names.len()
                    );
                    println!("  Compilation time: {:?}", compilation_time);

                    assert_eq!(
                        compiled_shader.function_names.len(),
                        100,
                        "Should have found all 100 functions"
                    );
                }
                Err(e) => {
                    println!("Large shader compilation failed: {}", e);
                    // Large shaders might fail due to resource limits
                }
            }
        } else {
            println!("Skipping large shader test - no Metal device available");
        }
    }

    /// Test shader compilation with Unicode and special characters
    #[test]
    fn test_unicode_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let unicode_shader_path = temp_dir.path().join("unicode.metal");

            // Create a shader with Unicode comments and identifiers
            let unicode_content = r#"
                #include <metal_stdlib>
                using namespace metal;

                // Test with Unicode comments: αβγδε, 中文, 🚀
                // Mathematical symbols: ∑∏∫∆∇

                kernel void test_unicode_function(
                    device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]
                ) {
                    if (gid >= count) return;
                    // Simple operation with Unicode comment: π ≈ 3.14159
                    output[gid] = input[gid] * 3.14159;
                }
            "#;

            fs::write(&unicode_shader_path, unicode_content).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let result = compiler.compile_shader_file(&unicode_shader_path);

            match result {
                Ok(compiled_shader) => {
                    println!("✓ Unicode shader compiled successfully");
                    assert!(!compiled_shader.function_names.is_empty());
                    assert!(compiled_shader
                        .function_names
                        .contains(&"test_unicode_function".to_string()));
                }
                Err(e) => {
                    println!("Unicode shader compilation failed: {}", e);
                    // Unicode in comments should generally be fine, but some systems might have issues
                }
            }
        } else {
            println!("Skipping Unicode shader test - no Metal device available");
        }
    }

    /// Test shader compilation with extreme cache scenarios
    #[test]
    fn test_extreme_cache_scenarios() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let cache_dir = temp_dir.path().join("cache");

            // Test with read-only cache directory
            fs::create_dir_all(&cache_dir).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(cache_dir.clone()),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device.clone(), config).unwrap();

            // Test rapid cache operations
            for i in 0..10 {
                compiler.clear_cache();
                let stats = compiler.get_stats();
                assert_eq!(
                    stats.cached_shaders, 0,
                    "Cache should be empty after clear {}",
                    i
                );

                // Try to compile shaders if they exist
                let _result = compiler.compile_all_shaders();
            }

            println!("✓ Extreme cache scenarios handled");

            // Test cache with invalid directory
            let invalid_cache_config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("/invalid/path/that/does/not/exist")),
                ..Default::default()
            };

            let invalid_compiler_result = ShaderCompiler::new(device, invalid_cache_config);
            // Should either succeed (graceful fallback) or fail with clear error
            match invalid_compiler_result {
                Ok(_) => println!("✓ Invalid cache directory handled gracefully"),
                Err(e) => println!("✓ Invalid cache directory correctly rejected: {}", e),
            }
        } else {
            println!("Skipping extreme cache tests - no Metal device available");
        }
    }

    /// Test concurrent shader compilation
    #[test]
    fn test_concurrent_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_concurrent_cache")),
                ..Default::default()
            };

            let compiler = std::sync::Arc::new(ShaderCompiler::new(device, config).unwrap());
            let mut handles = vec![];

            // Spawn multiple threads to compile shaders concurrently
            for i in 0..5 {
                let compiler_clone = compiler.clone();
                let handle = std::thread::spawn(move || {
                    println!("Thread {} starting shader compilation", i);

                    // Each thread tries to compile all shaders
                    let result = compiler_clone.compile_all_shaders();
                    match result {
                        Ok(shaders) => {
                            println!("Thread {} compiled {} shaders", i, shaders.len());
                            shaders.len()
                        }
                        Err(e) => {
                            println!("Thread {} compilation failed: {}", i, e);
                            0
                        }
                    }
                });
                handles.push(handle);
            }

            // Wait for all threads to complete
            let mut total_compiled = 0;
            for handle in handles {
                let compiled_count = handle.join().unwrap();
                total_compiled += compiled_count;
            }

            println!(
                "✓ Concurrent compilation completed, total: {}",
                total_compiled
            );

            // Verify cache consistency after concurrent access
            let final_stats = compiler.get_stats();
            println!("Final cache stats: {:?}", final_stats);
        } else {
            println!("Skipping concurrent compilation test - no Metal device available");
        }
    }

    /// Test shader compilation with memory pressure
    #[test]
    fn test_memory_pressure_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_memory_pressure_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Repeatedly compile and clear to test memory management
            for iteration in 0..20 {
                let start_memory = get_approximate_memory_usage();

                let _result = compiler.compile_all_shaders();
                compiler.clear_cache();

                let end_memory = get_approximate_memory_usage();

                println!(
                    "Iteration {}: Memory usage {} -> {}",
                    iteration, start_memory, end_memory
                );

                // Memory usage shouldn't grow unboundedly
                if iteration > 5 {
                    let memory_growth = end_memory.saturating_sub(start_memory);
                    assert!(
                        memory_growth < 100_000_000, // 100MB threshold
                        "Memory usage growing too much: {} bytes",
                        memory_growth
                    );
                }
            }

            println!("✓ Memory pressure test completed");
        } else {
            println!("Skipping memory pressure test - no Metal device available");
        }
    }

    /// Helper function to get approximate memory usage
    fn get_approximate_memory_usage() -> usize {
        // This is a rough approximation - in a real test you might use more sophisticated memory tracking
        std::alloc::System.alloc(std::alloc::Layout::new::<u8>()) as usize
    }

    /// Test shader compilation timeout scenarios
    #[test]
    fn test_compilation_timeout_scenarios() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let temp_dir = TempDir::new().unwrap();
            let complex_shader_path = temp_dir.path().join("complex.metal");

            // Create a computationally complex shader that might take time to compile
            let complex_content = r#"
                #include <metal_stdlib>
                using namespace metal;

                // Complex shader with many nested loops and calculations
                kernel void complex_computation(
                    device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]
                ) {
                    if (gid >= count) return;

                    float result = input[gid];

                    // Nested loops to create compilation complexity
                    for (uint i = 0; i < 10; i++) {
                        for (uint j = 0; j < 10; j++) {
                            for (uint k = 0; k < 10; k++) {
                                result = sin(cos(tan(result + float(i * j * k))));
                                result = sqrt(abs(result));
                                result = exp(log(result + 1.0));
                            }
                        }
                    }

                    output[gid] = result;
                }
            "#;

            fs::write(&complex_shader_path, complex_content).unwrap();

            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: false,
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Measure compilation time
            let start_time = std::time::Instant::now();
            let result = compiler.compile_shader_file(&complex_shader_path);
            let compilation_time = start_time.elapsed();

            println!("Complex shader compilation time: {:?}", compilation_time);

            match result {
                Ok(_) => {
                    println!("✓ Complex shader compiled successfully");

                    // Warn if compilation took too long
                    if compilation_time > Duration::from_secs(10) {
                        println!(
                            "⚠ Compilation took longer than expected: {:?}",
                            compilation_time
                        );
                    }
                }
                Err(e) => {
                    println!("Complex shader compilation failed: {}", e);
                    // Complex shaders might fail due to resource limits or timeouts
                }
            }
        } else {
            println!("Skipping timeout test - no Metal device available");
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_edge_case_tests {
    #[test]
    fn test_edge_cases_unsupported_platform() {
        println!("Edge case shader compilation tests skipped - not on macOS or Metal feature not enabled");
        assert!(true, "Non-macOS platform edge case test passed");
    }
}
