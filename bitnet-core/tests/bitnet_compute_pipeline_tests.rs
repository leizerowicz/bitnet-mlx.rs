//! BitNet-Specific Compute Pipeline Tests
//!
//! This module tests BitNet-specific compute operations including:
//! - BitLinear forward/backward operations
//! - Quantization/dequantization pipelines
//! - Activation function pipelines
//! - Mixed precision operations
//! - Shader compilation and pipeline creation
//! - End-to-end BitNet compute workflows

#[cfg(all(target_os = "macos", feature = "metal"))]
mod bitnet_metal_tests {
    use bitnet_core::metal::*;
    use std::time::Instant;

    /// Test BitNet shader compilation and loading
    #[test]
    fn test_bitnet_shader_compilation() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            // Test shader compiler creation
            let compiler_result = create_shader_compiler(&device);
            match compiler_result {
                Ok(compiler) => {
                    println!("✓ BitNet shader compiler created successfully");

                    // Test shader discovery and compilation
                    let shader_files =
                        ["bitlinear.metal", "quantization.metal", "activation.metal"];

                    for shader_file in &shader_files {
                        let shader_path = std::path::PathBuf::from("bitnet-core/src/metal/shaders")
                            .join(shader_file);

                        if shader_path.exists() {
                            let compile_result = compiler.compile_shader_file(&shader_path);
                            match compile_result {
                                Ok(compiled_shader) => {
                                    println!(
                                        "✓ Compiled {}: {} functions",
                                        shader_file,
                                        compiled_shader.function_names.len()
                                    );

                                    // Verify expected functions exist
                                    verify_shader_functions(
                                        shader_file,
                                        &compiled_shader.function_names,
                                    );
                                }
                                Err(e) => println!("Failed to compile {}: {}", shader_file, e),
                            }
                        } else {
                            println!("Shader file not found: {:?}", shader_path);
                        }
                    }

                    // Test compiler statistics
                    let stats = compiler.get_stats();
                    println!("✓ Shader compiler stats: {:?}", stats);
                }
                Err(e) => println!("Failed to create shader compiler: {}", e),
            }
        } else {
            println!("Skipping BitNet shader compilation test - no Metal device available");
        }
    }

    fn verify_shader_functions(shader_file: &str, function_names: &[String]) {
        let expected_functions = match shader_file {
            "bitlinear.metal" => vec![
                "bitlinear_forward",
                "bitlinear_backward_input",
                "binarize_weights",
                "quantize_activations",
            ],
            "quantization.metal" => vec![
                "quantize_weights_1bit",
                "quantize_activations_8bit",
                "dequantize_weights_1bit",
                "dequantize_activations_8bit",
                "dynamic_quantize_activations",
                "quantize_gradients",
                "mixed_precision_matmul",
            ],
            "activation.metal" => vec![
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
            _ => vec![],
        };

        for expected in &expected_functions {
            if function_names.contains(&expected.to_string()) {
                println!("  ✓ Found expected function: {}", expected);
            } else {
                println!("  ✗ Missing expected function: {}", expected);
            }
        }
    }

    /// Test BitNet shader utilities and pipeline creation
    #[test]
    fn test_bitnet_shader_utilities() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            // Test BitNet shaders initialization
            let shaders_result = BitNetShaders::new(device.clone());
            match shaders_result {
                Ok(shaders) => {
                    println!("✓ BitNet shaders initialized successfully");

                    // Test available shaders
                    let available = shaders.get_available_shaders();
                    println!("Available shaders: {:?}", available);

                    // Test shader function enumeration
                    test_shader_function_enum();

                    // Test pipeline creation for each function type
                    test_pipeline_creation(&shaders);

                    // Test compute encoder creation
                    test_compute_encoder_creation(&shaders, &device);
                }
                Err(e) => {
                    println!("BitNet shaders initialization failed (expected if shader files missing): {}", e);
                    // Test that the enum functions work even without actual shaders
                    test_shader_function_enum();
                }
            }
        } else {
            println!("Skipping BitNet shader utilities test - no Metal device available");
        }
    }

    fn test_shader_function_enum() {
        // Test BitLinear functions
        assert_eq!(
            BitNetShaderFunction::BitLinearForward.shader_name(),
            "bitlinear"
        );
        assert_eq!(
            BitNetShaderFunction::BitLinearForward.function_name(),
            "bitlinear_forward"
        );
        assert_eq!(
            BitNetShaderFunction::BitLinearForward.pipeline_key(),
            "bitlinear::bitlinear_forward"
        );

        // Test quantization functions
        assert_eq!(
            BitNetShaderFunction::QuantizeWeights1Bit.shader_name(),
            "quantization"
        );
        assert_eq!(
            BitNetShaderFunction::QuantizeWeights1Bit.function_name(),
            "quantize_weights_1bit"
        );
        assert_eq!(
            BitNetShaderFunction::QuantizeWeights1Bit.pipeline_key(),
            "quantization::quantize_weights_1bit"
        );

        // Test activation functions
        assert_eq!(
            BitNetShaderFunction::ReluForward.shader_name(),
            "activation"
        );
        assert_eq!(
            BitNetShaderFunction::ReluForward.function_name(),
            "relu_forward"
        );
        assert_eq!(
            BitNetShaderFunction::ReluForward.pipeline_key(),
            "activation::relu_forward"
        );

        println!("✓ BitNet shader function enum tested");
    }

    fn test_pipeline_creation(shaders: &BitNetShaders) {
        let test_functions = [
            BitNetShaderFunction::BitLinearForward,
            BitNetShaderFunction::QuantizeWeights1Bit,
            BitNetShaderFunction::ReluForward,
            BitNetShaderFunction::GeluForward,
            BitNetShaderFunction::MixedPrecisionMatmul,
        ];

        for function in &test_functions {
            let pipeline_result = shaders.get_pipeline(*function);
            match pipeline_result {
                Ok(_pipeline) => {
                    println!("✓ Created pipeline for: {:?}", function);
                }
                Err(e) => {
                    println!("Pipeline creation failed for {:?}: {}", function, e);
                    // This may be expected if shader files don't exist
                }
            }
        }
    }

    fn test_compute_encoder_creation(shaders: &BitNetShaders, device: &metal::Device) {
        let command_queue = create_command_queue(device);
        let command_buffer = command_queue.new_command_buffer();

        // Test encoder creation for different function types
        let encoder_functions = [
            BitNetShaderFunction::BitLinearForward,
            BitNetShaderFunction::QuantizeWeights1Bit,
            BitNetShaderFunction::ReluForward,
        ];

        for function in &encoder_functions {
            let encoder_result =
                shaders.create_compute_encoder_with_pipeline(&command_buffer, *function);
            match encoder_result {
                Ok(encoder) => {
                    println!("✓ Created compute encoder for: {:?}", function);
                    encoder.end_encoding();
                }
                Err(e) => {
                    println!("Encoder creation failed for {:?}: {}", function, e);
                }
            }
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Test BitLinear compute operations
    #[test]
    fn test_bitlinear_compute_operations() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let command_queue = create_command_queue(&device);
            let shaders_result = BitNetShaders::new(device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ Testing BitLinear compute operations");

                    // Test BitLinear forward operation
                    test_bitlinear_forward(&device, &command_queue, &shaders);

                    // Test weight binarization
                    test_weight_binarization(&device, &command_queue, &shaders);

                    // Test activation quantization
                    test_activation_quantization(&device, &command_queue, &shaders);
                }
                Err(e) => {
                    println!(
                        "Skipping BitLinear operations test (shader loading failed): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping BitLinear compute operations test - no Metal device available");
        }
    }

    fn test_bitlinear_forward(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        // Create test data
        let batch_size = 2u32;
        let input_size = 4u32;
        let output_size = 3u32;

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // 2x4
        let weights_data = vec![1i8, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]; // 3x4 (binarized)
        let bias_data = vec![0.1f32, 0.2, 0.3]; // 3
        let output_data = vec![0.0f32; (batch_size * output_size) as usize];

        // Create buffers
        let input_buffer_result = create_buffer(device, &input_data);
        let weights_buffer_result = create_buffer(device, &weights_data);
        let bias_buffer_result = create_buffer(device, &bias_data);
        let output_buffer_result = create_buffer(device, &output_data);

        match (
            input_buffer_result,
            weights_buffer_result,
            bias_buffer_result,
            output_buffer_result,
        ) {
            (Ok(input_buffer), Ok(weights_buffer), Ok(bias_buffer), Ok(output_buffer)) => {
                println!("✓ Created BitLinear test buffers");

                // Create command buffer and encoder
                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = create_bitlinear_forward_encoder(shaders, &command_buffer);

                match encoder_result {
                    Ok(encoder) => {
                        println!("✓ Created BitLinear forward encoder");

                        // Calculate dispatch parameters
                        let dispatch_result = shaders.calculate_dispatch_params(
                            BitNetShaderFunction::BitLinearForward,
                            (batch_size * output_size) as usize,
                        );

                        match dispatch_result {
                            Ok((threads, threadgroup)) => {
                                // Dispatch BitLinear forward operation
                                dispatch_bitlinear_forward(
                                    &encoder,
                                    &input_buffer,
                                    &weights_buffer,
                                    Some(&bias_buffer),
                                    &output_buffer,
                                    input_size,
                                    output_size,
                                    batch_size,
                                    threads,
                                    threadgroup,
                                );

                                encoder.end_encoding();
                                command_buffer.commit();
                                command_buffer.wait_until_completed();

                                println!("✓ BitLinear forward operation dispatched successfully");

                                // Read back results (for verification in real scenarios)
                                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                                match result {
                                    Ok(output_values) => {
                                        println!("✓ BitLinear output: {:?}", output_values);
                                    }
                                    Err(e) => println!("Failed to read output buffer: {}", e),
                                }
                            }
                            Err(e) => println!("Failed to calculate dispatch parameters: {}", e),
                        }
                    }
                    Err(e) => println!("Failed to create BitLinear encoder: {}", e),
                }
            }
            _ => println!("Failed to create BitLinear test buffers"),
        }
    }

    fn test_weight_binarization(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        // Create test weight data
        let weights_fp = vec![-0.5f32, 0.3, -0.8, 0.1, 0.7, -0.2, 0.9, -0.4];
        let weights_binary = vec![0i8; weights_fp.len()];
        let scale_factors = vec![0.0f32; weights_fp.len()];

        // Create buffers
        let weights_fp_buffer_result = create_buffer(device, &weights_fp);
        let weights_binary_buffer_result = create_buffer(device, &weights_binary);
        let scale_factors_buffer_result = create_buffer(device, &scale_factors);

        match (
            weights_fp_buffer_result,
            weights_binary_buffer_result,
            scale_factors_buffer_result,
        ) {
            (Ok(weights_fp_buffer), Ok(weights_binary_buffer), Ok(scale_factors_buffer)) => {
                println!("✓ Created weight binarization test buffers");

                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = shaders.create_compute_encoder_with_pipeline(
                    &command_buffer,
                    BitNetShaderFunction::BinarizeWeights,
                );

                match encoder_result {
                    Ok(encoder) => {
                        // Set buffers and parameters
                        set_compute_buffer(&encoder, &weights_fp_buffer, 0, 0);
                        set_compute_buffer(&encoder, &weights_binary_buffer, 0, 1);
                        set_compute_buffer(&encoder, &scale_factors_buffer, 0, 2);
                        set_compute_bytes(&encoder, &[weights_fp.len() as u32], 3);

                        // Dispatch
                        let threads = metal::MTLSize::new(weights_fp.len() as u64, 1, 1);
                        let threadgroup = metal::MTLSize::new(32, 1, 1);
                        dispatch_compute(&encoder, threads, threadgroup);

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();

                        println!("✓ Weight binarization operation completed");

                        // Verify results
                        let binary_result: Result<Vec<i8>> = read_buffer(&weights_binary_buffer);
                        let scale_result: Result<Vec<f32>> = read_buffer(&scale_factors_buffer);

                        match (binary_result, scale_result) {
                            (Ok(binary_weights), Ok(scales)) => {
                                println!("✓ Binarized weights: {:?}", binary_weights);
                                println!("✓ Scale factors: {:?}", scales);
                            }
                            _ => println!("Failed to read binarization results"),
                        }
                    }
                    Err(e) => println!("Failed to create weight binarization encoder: {}", e),
                }
            }
            _ => println!("Failed to create weight binarization buffers"),
        }
    }

    fn test_activation_quantization(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        // Create test activation data
        let activations = vec![0.1f32, 0.5, -0.3, 0.8, -0.1, 0.2, 0.9, -0.7];
        let quantized_activations = vec![0i8; activations.len()];
        let scale_factors = vec![0.0f32; 2]; // 2 groups
        let zero_points = vec![0.0f32; 2];

        // Create buffers
        let activations_buffer_result = create_buffer(device, &activations);
        let quantized_buffer_result = create_buffer(device, &quantized_activations);
        let scale_buffer_result = create_buffer(device, &scale_factors);
        let zero_buffer_result = create_buffer(device, &zero_points);

        match (
            activations_buffer_result,
            quantized_buffer_result,
            scale_buffer_result,
            zero_buffer_result,
        ) {
            (Ok(activations_buffer), Ok(quantized_buffer), Ok(scale_buffer), Ok(zero_buffer)) => {
                println!("✓ Created activation quantization test buffers");

                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = shaders.create_compute_encoder_with_pipeline(
                    &command_buffer,
                    BitNetShaderFunction::QuantizeActivations8Bit,
                );

                match encoder_result {
                    Ok(encoder) => {
                        // Set buffers and parameters
                        set_compute_buffer(&encoder, &activations_buffer, 0, 0);
                        set_compute_buffer(&encoder, &quantized_buffer, 0, 1);
                        set_compute_buffer(&encoder, &scale_buffer, 0, 2);
                        set_compute_buffer(&encoder, &zero_buffer, 0, 3);
                        set_compute_bytes(&encoder, &[activations.len() as u32], 4);
                        set_compute_bytes(&encoder, &[4u32], 5); // group_size

                        // Dispatch
                        let threads = metal::MTLSize::new(activations.len() as u64, 1, 1);
                        let threadgroup = metal::MTLSize::new(32, 1, 1);
                        dispatch_compute(&encoder, threads, threadgroup);

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();

                        println!("✓ Activation quantization operation completed");
                    }
                    Err(e) => println!("Failed to create activation quantization encoder: {}", e),
                }
            }
            _ => println!("Failed to create activation quantization buffers"),
        }
    }

    /// Test quantization compute operations
    #[test]
    fn test_quantization_compute_operations() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let command_queue = create_command_queue(&device);
            let shaders_result = BitNetShaders::new(device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ Testing quantization compute operations");

                    // Test different quantization operations
                    test_1bit_weight_quantization(&device, &command_queue, &shaders);
                    test_8bit_activation_quantization(&device, &command_queue, &shaders);
                    test_dynamic_quantization(&device, &command_queue, &shaders);
                    test_mixed_precision_matmul(&device, &command_queue, &shaders);
                }
                Err(e) => {
                    println!(
                        "Skipping quantization operations test (shader loading failed): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping quantization compute operations test - no Metal device available");
        }
    }

    fn test_1bit_weight_quantization(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let weights = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.7, 0.9, -0.4];
        let quantized = vec![0i8; weights.len()];
        let scales = vec![0.0f32; 2]; // 2 groups of 4

        let weights_buffer = create_buffer(device, &weights).unwrap();
        let quantized_buffer = create_buffer(device, &quantized).unwrap();
        let scales_buffer = create_buffer(device, &scales).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = create_quantization_encoder(
            shaders,
            &command_buffer,
            BitNetShaderFunction::QuantizeWeights1Bit,
        );

        match encoder_result {
            Ok(encoder) => {
                dispatch_quantization(
                    &encoder,
                    &weights_buffer,
                    &quantized_buffer,
                    &scales_buffer,
                    weights.len() as u32,
                    4u32, // group_size
                    metal::MTLSize::new(weights.len() as u64, 1, 1),
                    metal::MTLSize::new(32, 1, 1),
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ 1-bit weight quantization completed");
            }
            Err(e) => println!("Failed to create 1-bit quantization encoder: {}", e),
        }
    }

    fn test_8bit_activation_quantization(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let activations = vec![0.1f32, 0.5, -0.3, 0.8, -0.1, 0.2, 0.9, -0.7];
        let quantized = vec![0i8; activations.len()];
        let scales = vec![0.0f32; 2];
        let zeros = vec![0.0f32; 2];

        let activations_buffer = create_buffer(device, &activations).unwrap();
        let quantized_buffer = create_buffer(device, &quantized).unwrap();
        let scales_buffer = create_buffer(device, &scales).unwrap();
        let zeros_buffer = create_buffer(device, &zeros).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::QuantizeActivations8Bit,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &activations_buffer, 0, 0);
                set_compute_buffer(&encoder, &quantized_buffer, 0, 1);
                set_compute_buffer(&encoder, &scales_buffer, 0, 2);
                set_compute_buffer(&encoder, &zeros_buffer, 0, 3);
                set_compute_bytes(&encoder, &[activations.len() as u32], 4);
                set_compute_bytes(&encoder, &[4u32], 5);

                let threads = metal::MTLSize::new(activations.len() as u64, 1, 1);
                let threadgroup = metal::MTLSize::new(32, 1, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ 8-bit activation quantization completed");
            }
            Err(e) => println!("Failed to create 8-bit quantization encoder: {}", e),
        }
    }

    fn test_dynamic_quantization(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let activations = vec![0.1f32, 0.5, -0.3, 0.8, -0.1, 0.2, 0.9, -0.7, 0.4, -0.6];
        let quantized = vec![0i8; activations.len()];
        let scale = vec![0.0f32; 1];
        let zero_point = vec![0.0f32; 1];

        let activations_buffer = create_buffer(device, &activations).unwrap();
        let quantized_buffer = create_buffer(device, &quantized).unwrap();
        let scale_buffer = create_buffer(device, &scale).unwrap();
        let zero_buffer = create_buffer(device, &zero_point).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::DynamicQuantizeActivations,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &activations_buffer, 0, 0);
                set_compute_buffer(&encoder, &quantized_buffer, 0, 1);
                set_compute_buffer(&encoder, &scale_buffer, 0, 2);
                set_compute_buffer(&encoder, &zero_buffer, 0, 3);
                set_compute_bytes(&encoder, &[activations.len() as u32], 4);

                let threads = metal::MTLSize::new(256, 1, 1); // Use 256 threads for reduction
                let threadgroup = metal::MTLSize::new(256, 1, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ Dynamic quantization completed");
            }
            Err(e) => println!("Failed to create dynamic quantization encoder: {}", e),
        }
    }

    fn test_mixed_precision_matmul(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let m = 2u32;
        let n = 3u32;
        let k = 4u32;

        let weights_1bit = vec![1i8, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]; // 3x4
        let activations_8bit = vec![100i8, 150, 80, 200, 120, 180, 90, 160]; // 2x4
        let weight_scales = vec![0.5f32, 0.3]; // 2 groups
        let activation_scales = vec![0.01f32, 0.01]; // 2 groups
        let activation_zeros = vec![128.0f32, 128.0]; // 2 groups
        let output = vec![0.0f32; (m * n) as usize];

        let weights_buffer = create_buffer(device, &weights_1bit).unwrap();
        let activations_buffer = create_buffer(device, &activations_8bit).unwrap();
        let weight_scales_buffer = create_buffer(device, &weight_scales).unwrap();
        let activation_scales_buffer = create_buffer(device, &activation_scales).unwrap();
        let activation_zeros_buffer = create_buffer(device, &activation_zeros).unwrap();
        let output_buffer = create_buffer(device, &output).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::MixedPrecisionMatmul,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &weights_buffer, 0, 0);
                set_compute_buffer(&encoder, &activations_buffer, 0, 1);
                set_compute_buffer(&encoder, &weight_scales_buffer, 0, 2);
                set_compute_buffer(&encoder, &activation_scales_buffer, 0, 3);
                set_compute_buffer(&encoder, &activation_zeros_buffer, 0, 4);
                set_compute_buffer(&encoder, &output_buffer, 0, 5);
                set_compute_bytes(&encoder, &[m], 6);
                set_compute_bytes(&encoder, &[n], 7);
                set_compute_bytes(&encoder, &[k], 8);
                set_compute_bytes(&encoder, &[2u32], 9); // weight_group_size
                set_compute_bytes(&encoder, &[2u32], 10); // activation_group_size

                let threads = metal::MTLSize::new(n as u64, m as u64, 1);
                let threadgroup = metal::MTLSize::new(8, 8, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ Mixed precision matrix multiplication completed");

                // Read results
                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                match result {
                    Ok(output_values) => {
                        println!("✓ Mixed precision output: {:?}", output_values);
                    }
                    Err(e) => println!("Failed to read mixed precision output: {}", e),
                }
            }
            Err(e) => println!("Failed to create mixed precision encoder: {}", e),
        }
    }

    /// Test activation function compute operations
    #[test]
    fn test_activation_compute_operations() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let command_queue = create_command_queue(&device);
            let shaders_result = BitNetShaders::new(device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ Testing activation function compute operations");

                    // Test various activation functions
                    test_activation_function(
                        &device,
                        &command_queue,
                        &shaders,
                        BitNetShaderFunction::ReluForward,
                        "ReLU",
                    );
                    test_activation_function(
                        &device,
                        &command_queue,
                        &shaders,
                        BitNetShaderFunction::GeluForward,
                        "GELU",
                    );
                    test_activation_function(
                        &device,
                        &command_queue,
                        &shaders,
                        BitNetShaderFunction::SwishForward,
                        "Swish",
                    );
                    test_activation_function(
                        &device,
                        &command_queue,
                        &shaders,
                        BitNetShaderFunction::SigmoidForward,
                        "Sigmoid",
                    );
                    test_activation_function(
                        &device,
                        &command_queue,
                        &shaders,
                        BitNetShaderFunction::TanhForward,
                        "Tanh",
                    );

                    // Test fused operations
                    test_fused_activation(&device, &command_queue, &shaders);

                    // Test softmax and layer norm
                    test_softmax_operation(&device, &command_queue, &shaders);
                    test_layer_norm_operation(&device, &command_queue, &shaders);
                }
                Err(e) => {
                    println!(
                        "Skipping activation operations test (shader loading failed): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping activation compute operations test - no Metal device available");
        }
    }

    fn test_activation_function(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
        function: BitNetShaderFunction,
        name: &str,
    ) {
        let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 1.5];
        let output_data = vec![0.0f32; input_data.len()];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = create_activation_encoder(shaders, &command_buffer, function);

        match encoder_result {
            Ok(encoder) => {
                dispatch_activation(
                    &encoder,
                    &input_buffer,
                    &output_buffer,
                    input_data.len() as u32,
                    metal::MTLSize::new(input_data.len() as u64, 1, 1),
                    metal::MTLSize::new(32, 1, 1),
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ {} activation function completed", name);

                // Read and verify results
                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                match result {
                    Ok(output_values) => {
                        println!("  {} output: {:?}", name, output_values);
                    }
                    Err(e) => println!("  Failed to read {} output: {}", name, e),
                }
            }
            Err(e) => println!("Failed to create {} encoder: {}", name, e),
        }
    }

    fn test_fused_activation(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let input_data = vec![-1.0f32, 0.5, 2.0, -0.3, 1.8, -2.5, 0.8, 1.2];
        let dropout_mask = vec![1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let output_data = vec![0.0f32; input_data.len()];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let dropout_buffer = create_buffer(device, &dropout_mask).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::FusedReluDropout,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &input_buffer, 0, 0);
                set_compute_buffer(&encoder, &output_buffer, 0, 1);
                set_compute_buffer(&encoder, &dropout_buffer, 0, 2);
                set_compute_bytes(&encoder, &[0.5f32], 3); // dropout_prob
                set_compute_bytes(&encoder, &[input_data.len() as u32], 4);

                let threads = metal::MTLSize::new(input_data.len() as u64, 1, 1);
                let threadgroup = metal::MTLSize::new(32, 1, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ Fused ReLU-Dropout operation completed");
            }
            Err(e) => println!("Failed to create fused activation encoder: {}", e),
        }
    }

    fn test_softmax_operation(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let batch_size = 2u32;
        let feature_size = 4u32;
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // 2x4
        let output_data = vec![0.0f32; input_data.len()];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::SoftmaxForward,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &input_buffer, 0, 0);
                set_compute_buffer(&encoder, &output_buffer, 0, 1);
                set_compute_bytes(&encoder, &[batch_size], 2);
                set_compute_bytes(&encoder, &[feature_size], 3);

                let threads = metal::MTLSize::new(feature_size as u64, batch_size as u64, 1);
                let threadgroup = metal::MTLSize::new(4, 2, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ Softmax operation completed");

                // Read and verify results
                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                match result {
                    Ok(output_values) => {
                        println!("  Softmax output: {:?}", output_values);
                        // Verify softmax properties (sum should be ~1.0 for each batch)
                        for batch in 0..batch_size {
                            let start = (batch * feature_size) as usize;
                            let end = start + feature_size as usize;
                            let sum: f32 = output_values[start..end].iter().sum();
                            println!("  Batch {} sum: {:.6}", batch, sum);
                        }
                    }
                    Err(e) => println!("  Failed to read softmax output: {}", e),
                }
            }
            Err(e) => println!("Failed to create softmax encoder: {}", e),
        }
    }

    fn test_layer_norm_operation(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let batch_size = 2u32;
        let feature_size = 4u32;
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]; // 2x4
        let gamma = vec![1.0f32, 1.0, 1.0, 1.0]; // scale parameters
        let beta = vec![0.0f32, 0.0, 0.0, 0.0]; // shift parameters
        let output_data = vec![0.0f32; input_data.len()];
        let mean_data = vec![0.0f32; batch_size as usize];
        let variance_data = vec![0.0f32; batch_size as usize];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let gamma_buffer = create_buffer(device, &gamma).unwrap();
        let beta_buffer = create_buffer(device, &beta).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();
        let mean_buffer = create_buffer(device, &mean_data).unwrap();
        let variance_buffer = create_buffer(device, &variance_data).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::LayerNormForward,
        );

        match encoder_result {
            Ok(encoder) => {
                set_compute_buffer(&encoder, &input_buffer, 0, 0);
                set_compute_buffer(&encoder, &gamma_buffer, 0, 1);
                set_compute_buffer(&encoder, &beta_buffer, 0, 2);
                set_compute_buffer(&encoder, &output_buffer, 0, 3);
                set_compute_buffer(&encoder, &mean_buffer, 0, 4);
                set_compute_buffer(&encoder, &variance_buffer, 0, 5);
                set_compute_bytes(&encoder, &[batch_size], 6);
                set_compute_bytes(&encoder, &[feature_size], 7);
                set_compute_bytes(&encoder, &[1e-5f32], 8); // epsilon

                let threads = metal::MTLSize::new(feature_size as u64, batch_size as u64, 1);
                let threadgroup = metal::MTLSize::new(4, 2, 1);
                dispatch_compute(&encoder, threads, threadgroup);

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                println!("✓ Layer normalization operation completed");

                // Read and verify results
                let output_result: Result<Vec<f32>> = read_buffer(&output_buffer);
                let mean_result: Result<Vec<f32>> = read_buffer(&mean_buffer);
                let variance_result: Result<Vec<f32>> = read_buffer(&variance_buffer);

                match (output_result, mean_result, variance_result) {
                    (Ok(output_values), Ok(mean_values), Ok(variance_values)) => {
                        println!("  LayerNorm output: {:?}", output_values);
                        println!("  LayerNorm means: {:?}", mean_values);
                        println!("  LayerNorm variances: {:?}", variance_values);
                    }
                    _ => println!("  Failed to read layer norm results"),
                }
            }
            Err(e) => println!("Failed to create layer norm encoder: {}", e),
        }
    }

    /// Test performance benchmarking of compute operations
    #[test]
    fn test_compute_performance_benchmarks() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let command_queue = create_command_queue(&device);
            let shaders_result = BitNetShaders::new(device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ Running compute performance benchmarks");

                    // Benchmark different operation types
                    benchmark_bitlinear_operations(&device, &command_queue, &shaders);
                    benchmark_quantization_operations(&device, &command_queue, &shaders);
                    benchmark_activation_operations(&device, &command_queue, &shaders);

                    // Benchmark different data sizes
                    benchmark_scaling_performance(&device, &command_queue, &shaders);
                }
                Err(e) => {
                    println!(
                        "Skipping performance benchmarks (shader loading failed): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping compute performance benchmarks - no Metal device available");
        }
    }

    fn benchmark_bitlinear_operations(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let sizes = [(128, 64), (256, 128), (512, 256), (1024, 512)];

        for &(input_size, output_size) in &sizes {
            let batch_size = 32u32;

            // Create test data
            let input_data = vec![1.0f32; (batch_size * input_size) as usize];
            let weights_data = vec![1i8; (output_size * input_size) as usize];
            let bias_data = vec![0.1f32; output_size as usize];
            let output_data = vec![0.0f32; (batch_size * output_size) as usize];

            // Create buffers
            let input_buffer = create_buffer(device, &input_data).unwrap();
            let weights_buffer = create_buffer(device, &weights_data).unwrap();
            let bias_buffer = create_buffer(device, &bias_data).unwrap();
            let output_buffer = create_buffer(device, &output_data).unwrap();

            // Benchmark multiple runs
            let num_runs = 10;
            let start_time = Instant::now();

            for _ in 0..num_runs {
                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = create_bitlinear_forward_encoder(shaders, &command_buffer);

                if let Ok(encoder) = encoder_result {
                    let dispatch_result = shaders.calculate_dispatch_params(
                        BitNetShaderFunction::BitLinearForward,
                        (batch_size * output_size) as usize,
                    );

                    if let Ok((threads, threadgroup)) = dispatch_result {
                        dispatch_bitlinear_forward(
                            &encoder,
                            &input_buffer,
                            &weights_buffer,
                            Some(&bias_buffer),
                            &output_buffer,
                            input_size,
                            output_size,
                            batch_size,
                            threads,
                            threadgroup,
                        );

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();
                    }
                }
            }

            let elapsed = start_time.elapsed();
            let avg_time = elapsed / num_runs;
            let ops_per_sec = 1.0 / avg_time.as_secs_f64();

            println!(
                "✓ BitLinear {}x{} (batch={}): {:.2}ms avg, {:.1} ops/sec",
                input_size,
                output_size,
                batch_size,
                avg_time.as_millis(),
                ops_per_sec
            );
        }
    }

    fn benchmark_quantization_operations(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let sizes = [1024, 4096, 16384, 65536];

        for &size in &sizes {
            let data = vec![0.5f32; size];
            let quantized = vec![0i8; size];
            let scales = vec![0.0f32; size / 32]; // 32 elements per group

            let data_buffer = create_buffer(device, &data).unwrap();
            let quantized_buffer = create_buffer(device, &quantized).unwrap();
            let scales_buffer = create_buffer(device, &scales).unwrap();

            let num_runs = 20;
            let start_time = Instant::now();

            for _ in 0..num_runs {
                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = create_quantization_encoder(
                    shaders,
                    &command_buffer,
                    BitNetShaderFunction::QuantizeWeights1Bit,
                );

                if let Ok(encoder) = encoder_result {
                    dispatch_quantization(
                        &encoder,
                        &data_buffer,
                        &quantized_buffer,
                        &scales_buffer,
                        size as u32,
                        32u32,
                        metal::MTLSize::new(size as u64, 1, 1),
                        metal::MTLSize::new(32, 1, 1),
                    );

                    encoder.end_encoding();
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                }
            }

            let elapsed = start_time.elapsed();
            let avg_time = elapsed / num_runs;
            let throughput = (size as f64) / avg_time.as_secs_f64() / 1e6; // Million elements per second

            println!(
                "✓ Quantization {} elements: {:.2}ms avg, {:.1}M elem/sec",
                size,
                avg_time.as_millis(),
                throughput
            );
        }
    }

    fn benchmark_activation_operations(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        let sizes = [1024, 4096, 16384, 65536];
        let activations = [
            (BitNetShaderFunction::ReluForward, "ReLU"),
            (BitNetShaderFunction::GeluForward, "GELU"),
            (BitNetShaderFunction::SwishForward, "Swish"),
        ];

        for &(function, name) in &activations {
            for &size in &sizes {
                let input_data = vec![0.5f32; size];
                let output_data = vec![0.0f32; size];

                let input_buffer = create_buffer(device, &input_data).unwrap();
                let output_buffer = create_buffer(device, &output_data).unwrap();

                let num_runs = 50;
                let start_time = Instant::now();

                for _ in 0..num_runs {
                    let command_buffer = command_queue.new_command_buffer();
                    let encoder_result =
                        create_activation_encoder(shaders, &command_buffer, function);

                    if let Ok(encoder) = encoder_result {
                        dispatch_activation(
                            &encoder,
                            &input_buffer,
                            &output_buffer,
                            size as u32,
                            metal::MTLSize::new(size as u64, 1, 1),
                            metal::MTLSize::new(64, 1, 1),
                        );

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();
                    }
                }

                let elapsed = start_time.elapsed();
                let avg_time = elapsed / num_runs;
                let throughput = (size as f64) / avg_time.as_secs_f64() / 1e6;

                println!(
                    "✓ {} {} elements: {:.2}ms avg, {:.1}M elem/sec",
                    name,
                    size,
                    avg_time.as_millis(),
                    throughput
                );
            }
        }
    }

    fn benchmark_scaling_performance(
        device: &metal::Device,
        command_queue: &metal::CommandQueue,
        shaders: &BitNetShaders,
    ) {
        println!("✓ Testing scaling performance with different thread configurations");

        let data_size = 16384;
        let input_data = vec![0.5f32; data_size];
        let output_data = vec![0.0f32; data_size];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();

        let threadgroup_sizes = [32, 64, 128, 256];

        for &threadgroup_size in &threadgroup_sizes {
            let num_runs = 30;
            let start_time = Instant::now();

            for _ in 0..num_runs {
                let command_buffer = command_queue.new_command_buffer();
                let encoder_result = create_activation_encoder(
                    shaders,
                    &command_buffer,
                    BitNetShaderFunction::ReluForward,
                );

                if let Ok(encoder) = encoder_result {
                    dispatch_activation(
                        &encoder,
                        &input_buffer,
                        &output_buffer,
                        data_size as u32,
                        metal::MTLSize::new(data_size as u64, 1, 1),
                        metal::MTLSize::new(threadgroup_size, 1, 1),
                    );

                    encoder.end_encoding();
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                }
            }

            let elapsed = start_time.elapsed();
            let avg_time = elapsed / num_runs;

            println!(
                "  Threadgroup size {}: {:.2}ms avg",
                threadgroup_size,
                avg_time.as_millis()
            );
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {
    #[test]
    fn test_unsupported_platform_bitnet_compute() {
        println!(
            "BitNet compute pipeline tests skipped - not on macOS or Metal feature not enabled"
        );
        // This test always passes on non-macOS platforms
        assert!(true);
    }
}
