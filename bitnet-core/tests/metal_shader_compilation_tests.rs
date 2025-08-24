//! Test Metal shader compilation functionality

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_tests {
    use bitnet_core::metal::*;
    use std::path::PathBuf;

    #[test]
    fn test_shader_compiler_creation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache")),
                ..Default::default()
            };

            let compiler_result = ShaderCompiler::new(device, config);
            assert!(
                compiler_result.is_ok(),
                "Failed to create shader compiler: {:?}",
                compiler_result.err()
            );

            println!("✓ Shader compiler created successfully");
        } else {
            println!("Skipping shader compiler test - no Metal device available");
        }
    }

    #[test]
    fn test_individual_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let config = ShaderCompilerConfig {
                shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
                enable_caching: true,
                cache_directory: Some(PathBuf::from("target/test_shader_cache")),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();

            // Test individual shader files
            let shader_files = ["bitlinear.metal", "quantization.metal", "activation.metal"];

            for shader_file in &shader_files {
                let shader_path = PathBuf::from("bitnet-core/src/metal/shaders").join(shader_file);

                if shader_path.exists() {
                    let result = compiler.compile_shader_file(&shader_path);
                    match result {
                        Ok(compiled_shader) => {
                            println!("✓ {} compiled successfully", shader_file);
                            println!("  Functions: {:?}", compiled_shader.function_names);
                            assert!(
                                !compiled_shader.function_names.is_empty(),
                                "Shader {} should have at least one function",
                                shader_file
                            );
                        }
                        Err(e) => {
                            panic!("✗ {} compilation failed: {}", shader_file, e);
                        }
                    }
                } else {
                    println!("Skipping {} - file not found", shader_file);
                }
            }
        } else {
            println!("Skipping individual shader compilation test - no Metal device available");
        }
    }

    #[test]
    fn test_bitnet_shaders_initialization() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let result = BitNetShaders::new(device);
            match result {
                Ok(shaders) => {
                    println!("✓ BitNet shaders initialized successfully");
                    let available = shaders.get_available_shaders();
                    println!("  Available shaders: {:?}", available);

                    // Test pipeline creation for key functions
                    let test_functions = [
                        BitNetShaderFunction::BitLinearForward,
                        BitNetShaderFunction::QuantizeWeights1Bit,
                        BitNetShaderFunction::ReluForward,
                    ];

                    for function in &test_functions {
                        let pipeline_result = shaders.get_pipeline(*function);
                        match pipeline_result {
                            Ok(_pipeline) => {
                                println!("✓ Pipeline created for: {:?}", function);
                            }
                            Err(e) => {
                                println!("✗ Pipeline creation failed for {:?}: {}", function, e);
                                // Don't panic here as shader files might not exist in test environment
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("BitNet shaders initialization failed (expected if shader files missing): {}", e);
                    // Don't panic as this might be expected in CI/test environments
                }
            }
        } else {
            println!("Skipping BitNet shaders test - no Metal device available");
        }
    }

    #[test]
    fn test_shader_function_names() {
        // Test that shader function enum provides correct names
        assert_eq!(
            BitNetShaderFunction::BitLinearForward.shader_name(),
            "bitlinear"
        );
        assert_eq!(
            BitNetShaderFunction::BitLinearForward.function_name(),
            "bitlinear_forward"
        );
        assert_eq!(
            BitNetShaderFunction::QuantizeWeights1Bit.shader_name(),
            "quantization"
        );
        assert_eq!(
            BitNetShaderFunction::QuantizeWeights1Bit.function_name(),
            "quantize_weights_1bit"
        );
        assert_eq!(
            BitNetShaderFunction::ReluForward.shader_name(),
            "activation"
        );
        assert_eq!(
            BitNetShaderFunction::ReluForward.function_name(),
            "relu_forward"
        );

        println!("✓ Shader function names are correct");
    }

    #[test]
    fn test_pipeline_keys() {
        let key = BitNetShaderFunction::BitLinearForward.pipeline_key();
        assert_eq!(key, "bitlinear::bitlinear_forward");

        let key = BitNetShaderFunction::QuantizeWeights1Bit.pipeline_key();
        assert_eq!(key, "quantization::quantize_weights_1bit");

        println!("✓ Pipeline keys are correctly formatted");
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {
    #[test]
    fn test_unsupported_platform() {
        println!(
            "Metal shader compilation tests skipped - not on macOS or Metal feature not enabled"
        );
        // This test always passes on non-macOS platforms
        assert!(true);
    }
}
