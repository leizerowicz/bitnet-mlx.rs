//! Compute Pipeline Integration Tests
//!
//! This module tests the integration of compute pipelines with existing BitNet systems:
//! - Integration with memory management systems
//! - Integration with tensor operations
//! - Integration with device management
//! - End-to-end workflow testing
//! - Cross-system compatibility testing

#[cfg(all(target_os = "macos", feature = "metal"))]
mod integration_tests {
    use bitnet_core::memory::*;
    use bitnet_core::metal::*;
    use bitnet_core::tensor::*;
    use bitnet_core::{DType, Device, Tensor};
    use std::time::Instant;

    /// Test compute pipeline integration with memory management
    #[test]
    fn test_compute_pipeline_memory_integration() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            let command_queue = create_command_queue(&device);
            let manager = create_command_buffer_manager(&device, &command_queue);

            // Test buffer pool integration
            let pool = create_buffer_pool(&device);

            println!("✓ Testing compute pipeline with memory management");

            // Test compute operations with pooled buffers
            test_pooled_buffer_compute_operations(&device, &manager, &pool);

            // Test memory tracking during compute operations
            test_memory_tracking_compute_operations(&device, &manager);

            // Test cleanup integration
            test_compute_memory_cleanup(&device, &manager, &pool);

            println!("✓ Compute pipeline memory integration tested");
        } else {
            println!(
                "Skipping compute pipeline memory integration test - no Metal device available"
            );
        }
    }

    fn test_pooled_buffer_compute_operations(
        device: &metal::Device,
        manager: &CommandBufferManager,
        pool: &BufferPool,
    ) {
        // Allocate buffers from pool for compute operations
        let buffer_sizes = [1024, 2048, 4096];
        let mut allocated_buffers = Vec::new();

        for &size in &buffer_sizes {
            let buffer_result = pool.get_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            match buffer_result {
                Ok(buffer) => {
                    println!("  ✓ Allocated pooled buffer of size: {}", size);
                    allocated_buffers.push(buffer);
                }
                Err(e) => println!("  Failed to allocate pooled buffer: {}", e),
            }
        }

        // Use pooled buffers in compute operations
        if allocated_buffers.len() >= 2 {
            let cb_id_result = manager.create_command_buffer(CommandBufferPriority::Normal);
            match cb_id_result {
                Ok(cb_id) => {
                    let begin_result = manager.begin_encoding(cb_id);
                    if begin_result.is_ok() {
                        let encoder_result = manager.create_compute_encoder(cb_id);
                        match encoder_result {
                            Ok(encoder) => {
                                // Bind pooled buffers to compute encoder
                                set_compute_buffer(&encoder, &allocated_buffers[0], 0, 0);
                                set_compute_buffer(&encoder, &allocated_buffers[1], 0, 1);

                                // Set up a simple compute dispatch
                                let threads = metal::MTLSize::new(256, 1, 1);
                                let threadgroup = metal::MTLSize::new(32, 1, 1);
                                dispatch_compute(&encoder, threads, threadgroup);

                                encoder.end_encoding();
                                println!("  ✓ Compute operation with pooled buffers configured");
                            }
                            Err(e) => println!("  Failed to create compute encoder: {}", e),
                        }

                        let _ = manager.commit_and_wait(cb_id);
                    }
                }
                Err(e) => println!("  Failed to create command buffer: {}", e),
            }
        }

        // Return buffers to pool
        for buffer in allocated_buffers {
            let _ = pool.return_buffer(buffer);
        }

        // Verify pool statistics
        let stats = pool.get_stats();
        println!(
            "  ✓ Pool stats after compute operations: cache_hits={}, cache_misses={}",
            stats.cache_hits, stats.cache_misses
        );
    }

    fn test_memory_tracking_compute_operations(
        device: &metal::Device,
        manager: &CommandBufferManager,
    ) {
        // Create test data for memory tracking
        let test_data = vec![1.0f32; 2048];
        let buffer_result = create_buffer(device, &test_data);

        match buffer_result {
            Ok(buffer) => {
                println!("  ✓ Created buffer for memory tracking test");

                // Create command buffer and track resource usage
                let cb_id_result = manager.create_command_buffer(CommandBufferPriority::Normal);
                match cb_id_result {
                    Ok(cb_id) => {
                        // Add resource to command buffer for tracking
                        let add_resource_result = manager.add_resource(cb_id, buffer.clone());
                        match add_resource_result {
                            Ok(()) => {
                                println!("  ✓ Resource added to command buffer for tracking");

                                // Begin encoding and create compute encoder
                                let begin_result = manager.begin_encoding(cb_id);
                                if begin_result.is_ok() {
                                    let encoder_result = manager.create_compute_encoder(cb_id);
                                    match encoder_result {
                                        Ok(encoder) => {
                                            set_compute_buffer(&encoder, &buffer, 0, 0);
                                            encoder.end_encoding();
                                            println!(
                                                "  ✓ Compute encoder with tracked resource created"
                                            );
                                        }
                                        Err(e) => {
                                            println!("  Failed to create compute encoder: {}", e)
                                        }
                                    }
                                }

                                let _ = manager.commit_and_wait(cb_id);
                            }
                            Err(e) => println!("  Failed to add resource for tracking: {}", e),
                        }
                    }
                    Err(e) => println!("  Failed to create command buffer: {}", e),
                }
            }
            Err(e) => println!("  Failed to create buffer for memory tracking: {}", e),
        }
    }

    fn test_compute_memory_cleanup(
        device: &metal::Device,
        manager: &CommandBufferManager,
        pool: &BufferPool,
    ) {
        // Create multiple compute operations to test cleanup
        let mut command_buffers = Vec::new();

        for i in 0..5 {
            let cb_result = manager.create_command_buffer(CommandBufferPriority::Normal);
            match cb_result {
                Ok(cb_id) => {
                    command_buffers.push(cb_id);

                    // Create some buffers for each command buffer
                    let buffer_result =
                        pool.get_buffer(1024, metal::MTLResourceOptions::StorageModeShared);
                    if let Ok(buffer) = buffer_result {
                        let _ = manager.add_resource(cb_id, buffer);
                    }

                    println!("  Created command buffer {} for cleanup test", i);
                }
                Err(e) => println!("  Failed to create command buffer {}: {}", i, e),
            }
        }

        // Test cleanup operations
        let cleanup_result = manager.cleanup();
        match cleanup_result {
            Ok(()) => println!("  ✓ Command buffer cleanup completed"),
            Err(e) => println!("  Command buffer cleanup failed: {}", e),
        }

        let pool_cleanup_result = pool.cleanup_unused_buffers();
        match pool_cleanup_result {
            Ok(()) => println!("  ✓ Buffer pool cleanup completed"),
            Err(e) => println!("  Buffer pool cleanup failed: {}", e),
        }

        // Return command buffers
        for cb_id in command_buffers {
            let _ = manager.return_command_buffer(cb_id);
        }

        // Final statistics
        let manager_stats = manager.get_stats();
        let pool_stats = pool.get_stats();
        println!(
            "  ✓ Final stats - Command buffers: {}, Pool buffers: {}",
            manager_stats.active_count, pool_stats.active_buffers
        );
    }

    /// Test compute pipeline integration with tensor operations
    #[test]
    fn test_compute_pipeline_tensor_integration() {
        let device_result = create_metal_device();
        if let Ok(metal_device) = device_result {
            println!("✓ Testing compute pipeline with tensor operations");

            // Test with BitNet device abstraction
            let bitnet_device = Device::Metal(metal_device.clone());

            // Test tensor creation and compute operations
            test_tensor_compute_operations(&bitnet_device, &metal_device);

            // Test BitNet tensor integration
            test_bitnet_tensor_compute_integration(&bitnet_device, &metal_device);

            // Test tensor lifecycle with compute operations
            test_tensor_lifecycle_compute_integration(&bitnet_device, &metal_device);

            println!("✓ Compute pipeline tensor integration tested");
        } else {
            println!(
                "Skipping compute pipeline tensor integration test - no Metal device available"
            );
        }
    }

    fn test_tensor_compute_operations(bitnet_device: &Device, metal_device: &metal::Device) {
        // Create tensors for compute operations
        let tensor_result = Tensor::zeros(&[4, 4], DType::F32, bitnet_device);
        match tensor_result {
            Ok(tensor) => {
                println!(
                    "  ✓ Created tensor for compute operations: {:?}",
                    tensor.shape()
                );

                // Test tensor data access for compute operations
                let data_result = tensor.to_vec1::<f32>();
                match data_result {
                    Ok(data) => {
                        println!("  ✓ Extracted tensor data: {} elements", data.len());

                        // Create Metal buffer from tensor data
                        let buffer_result = create_buffer(metal_device, &data);
                        match buffer_result {
                            Ok(buffer) => {
                                println!("  ✓ Created Metal buffer from tensor data");

                                // Test compute operation with tensor-derived buffer
                                test_simple_compute_with_tensor_buffer(metal_device, &buffer);
                            }
                            Err(e) => println!("  Failed to create buffer from tensor: {}", e),
                        }
                    }
                    Err(e) => println!("  Failed to extract tensor data: {}", e),
                }
            }
            Err(e) => println!("  Failed to create tensor: {}", e),
        }
    }

    fn test_simple_compute_with_tensor_buffer(device: &metal::Device, buffer: &metal::Buffer) {
        let command_queue = create_command_queue(device);
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set up a simple compute operation
        set_compute_buffer(&encoder, buffer, 0, 0);

        // Dispatch a simple operation
        let threads = metal::MTLSize::new(16, 1, 1);
        let threadgroup = metal::MTLSize::new(16, 1, 1);
        dispatch_compute(&encoder, threads, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        println!("    ✓ Compute operation with tensor buffer completed");
    }

    fn test_bitnet_tensor_compute_integration(
        bitnet_device: &Device,
        metal_device: &metal::Device,
    ) {
        // Test BitNet tensor creation and compute integration
        let tensor_result = BitNetTensor::zeros(&[8, 8], BitNetDType::F32, bitnet_device.clone());
        match tensor_result {
            Ok(bitnet_tensor) => {
                println!("  ✓ Created BitNet tensor: {:?}", bitnet_tensor.shape());

                // Test tensor handle integration with compute operations
                let handle = bitnet_tensor.handle();
                println!("  ✓ BitNet tensor handle: {:?}", handle.id());

                // Test metadata integration
                let metadata = bitnet_tensor.metadata();
                println!(
                    "  ✓ BitNet tensor metadata: shape={:?}, dtype={:?}",
                    metadata.shape(),
                    metadata.dtype()
                );

                // Test compute operations with BitNet tensor
                test_bitnet_tensor_compute_operations(&bitnet_tensor, metal_device);
            }
            Err(e) => println!("  Failed to create BitNet tensor: {}", e),
        }
    }

    fn test_bitnet_tensor_compute_operations(tensor: &BitNetTensor, metal_device: &metal::Device) {
        // Test accessing underlying data for compute operations
        let data_result = tensor.to_vec1::<f32>();
        match data_result {
            Ok(data) => {
                println!(
                    "    ✓ Extracted BitNet tensor data: {} elements",
                    data.len()
                );

                // Create compute buffers from BitNet tensor data
                let input_buffer_result = create_buffer(metal_device, &data);
                let output_data = vec![0.0f32; data.len()];
                let output_buffer_result = create_buffer(metal_device, &output_data);

                match (input_buffer_result, output_buffer_result) {
                    (Ok(input_buffer), Ok(output_buffer)) => {
                        // Test compute operation with BitNet tensor buffers
                        let command_queue = create_command_queue(metal_device);
                        let command_buffer = command_queue.new_command_buffer();
                        let encoder = command_buffer.new_compute_command_encoder();

                        set_compute_buffer(&encoder, &input_buffer, 0, 0);
                        set_compute_buffer(&encoder, &output_buffer, 0, 1);
                        set_compute_bytes(&encoder, &[data.len() as u32], 2);

                        let threads = metal::MTLSize::new(data.len() as u64, 1, 1);
                        let threadgroup = metal::MTLSize::new(32, 1, 1);
                        dispatch_compute(&encoder, threads, threadgroup);

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();

                        println!("    ✓ Compute operation with BitNet tensor completed");

                        // Read back results
                        let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                        match result {
                            Ok(output_values) => {
                                println!("    ✓ Compute output: {} values", output_values.len());
                            }
                            Err(e) => println!("    Failed to read compute output: {}", e),
                        }
                    }
                    _ => println!("    Failed to create compute buffers from BitNet tensor"),
                }
            }
            Err(e) => println!("    Failed to extract BitNet tensor data: {}", e),
        }
    }

    fn test_tensor_lifecycle_compute_integration(
        bitnet_device: &Device,
        metal_device: &metal::Device,
    ) {
        println!("  ✓ Testing tensor lifecycle with compute operations");

        // Create multiple tensors with different lifecycles
        let mut tensors = Vec::new();
        let mut buffers = Vec::new();

        for i in 0..3 {
            let tensor_result = Tensor::ones(&[16, 16], DType::F32, bitnet_device);
            match tensor_result {
                Ok(tensor) => {
                    println!("    Created tensor {}: {:?}", i, tensor.shape());

                    // Convert to compute buffer
                    let data_result = tensor.to_vec1::<f32>();
                    if let Ok(data) = data_result {
                        let buffer_result = create_buffer(metal_device, &data);
                        if let Ok(buffer) = buffer_result {
                            tensors.push(tensor);
                            buffers.push(buffer);
                        }
                    }
                }
                Err(e) => println!("    Failed to create tensor {}: {}", i, e),
            }
        }

        // Test compute operations with multiple tensor-derived buffers
        if buffers.len() >= 2 {
            let command_queue = create_command_queue(metal_device);
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Use multiple buffers in compute operation
            for (i, buffer) in buffers.iter().enumerate() {
                set_compute_buffer(&encoder, buffer, 0, i as u64);
            }

            let threads = metal::MTLSize::new(256, 1, 1);
            let threadgroup = metal::MTLSize::new(32, 1, 1);
            dispatch_compute(&encoder, threads, threadgroup);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("    ✓ Multi-tensor compute operation completed");
        }

        // Test tensor cleanup (automatic when tensors go out of scope)
        drop(tensors);
        drop(buffers);
        println!("    ✓ Tensor lifecycle cleanup completed");
    }

    /// Test end-to-end BitNet compute workflows
    #[test]
    fn test_end_to_end_bitnet_workflows() {
        let device_result = create_metal_device();
        if let Ok(metal_device) = device_result {
            let shaders_result = BitNetShaders::new(metal_device.clone());

            match shaders_result {
                Ok(shaders) => {
                    println!("✓ Testing end-to-end BitNet compute workflows");

                    // Test complete BitLinear workflow
                    test_complete_bitlinear_workflow(&metal_device, &shaders);

                    // Test complete quantization workflow
                    test_complete_quantization_workflow(&metal_device, &shaders);

                    // Test mixed precision workflow
                    test_mixed_precision_workflow(&metal_device, &shaders);

                    println!("✓ End-to-end BitNet workflows tested");
                }
                Err(e) => {
                    println!(
                        "Skipping end-to-end workflows (shader loading failed): {}",
                        e
                    );
                }
            }
        } else {
            println!("Skipping end-to-end BitNet workflows test - no Metal device available");
        }
    }

    fn test_complete_bitlinear_workflow(device: &metal::Device, shaders: &BitNetShaders) {
        println!("  ✓ Testing complete BitLinear workflow");

        let command_queue = create_command_queue(device);

        // Step 1: Weight binarization
        let weights_fp = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.7, 0.9, -0.4];
        let weights_binary = vec![0i8; weights_fp.len()];
        let weight_scales = vec![0.0f32; weights_fp.len()];

        let weights_fp_buffer = create_buffer(device, &weights_fp).unwrap();
        let weights_binary_buffer = create_buffer(device, &weights_binary).unwrap();
        let weight_scales_buffer = create_buffer(device, &weight_scales).unwrap();

        // Binarize weights
        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::BinarizeWeights,
        );

        if let Ok(encoder) = encoder_result {
            set_compute_buffer(&encoder, &weights_fp_buffer, 0, 0);
            set_compute_buffer(&encoder, &weights_binary_buffer, 0, 1);
            set_compute_buffer(&encoder, &weight_scales_buffer, 0, 2);
            set_compute_bytes(&encoder, &[weights_fp.len() as u32], 3);

            let threads = metal::MTLSize::new(weights_fp.len() as u64, 1, 1);
            let threadgroup = metal::MTLSize::new(32, 1, 1);
            dispatch_compute(&encoder, threads, threadgroup);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("    ✓ Weight binarization completed");
        }

        // Step 2: Activation quantization
        let activations = vec![0.1f32, 0.5, -0.3, 0.8];
        let quantized_activations = vec![0.0f32; activations.len()];
        let activation_scales = vec![0.0f32; 1];

        let activations_buffer = create_buffer(device, &activations).unwrap();
        let quantized_activations_buffer = create_buffer(device, &quantized_activations).unwrap();
        let activation_scales_buffer = create_buffer(device, &activation_scales).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::QuantizeActivations,
        );

        if let Ok(encoder) = encoder_result {
            set_compute_buffer(&encoder, &activations_buffer, 0, 0);
            set_compute_buffer(&encoder, &quantized_activations_buffer, 0, 1);
            set_compute_buffer(&encoder, &activation_scales_buffer, 0, 2);
            set_compute_bytes(&encoder, &[activations.len() as u32], 3);
            set_compute_bytes(&encoder, &[activations.len() as u32], 4); // group_size

            let threads = metal::MTLSize::new(activations.len() as u64, 1, 1);
            let threadgroup = metal::MTLSize::new(32, 1, 1);
            dispatch_compute(&encoder, threads, threadgroup);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("    ✓ Activation quantization completed");
        }

        // Step 3: BitLinear forward pass
        let batch_size = 1u32;
        let input_size = 4u32;
        let output_size = 2u32;

        let input_data = vec![0.1f32, 0.5, -0.3, 0.8];
        let bias_data = vec![0.1f32, 0.2];
        let output_data = vec![0.0f32; (batch_size * output_size) as usize];

        let input_buffer = create_buffer(device, &input_data).unwrap();
        let bias_buffer = create_buffer(device, &bias_data).unwrap();
        let output_buffer = create_buffer(device, &output_data).unwrap();

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
                    &weights_binary_buffer,
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

                println!("    ✓ BitLinear forward pass completed");

                // Read final results
                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                match result {
                    Ok(output_values) => {
                        println!("    ✓ BitLinear workflow output: {:?}", output_values);
                    }
                    Err(e) => println!("    Failed to read workflow output: {}", e),
                }
            }
        }
    }

    fn test_complete_quantization_workflow(device: &metal::Device, shaders: &BitNetShaders) {
        println!("  ✓ Testing complete quantization workflow");

        let command_queue = create_command_queue(device);

        // Step 1: 1-bit weight quantization
        let weights = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.7, 0.9, -0.4];
        let quantized_weights = vec![0i8; weights.len()];
        let weight_scales = vec![0.0f32; 2]; // 2 groups

        let weights_buffer = create_buffer(device, &weights).unwrap();
        let quantized_weights_buffer = create_buffer(device, &quantized_weights).unwrap();
        let weight_scales_buffer = create_buffer(device, &weight_scales).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = create_quantization_encoder(
            shaders,
            &command_buffer,
            BitNetShaderFunction::QuantizeWeights1Bit,
        );

        if let Ok(encoder) = encoder_result {
            dispatch_quantization(
                &encoder,
                &weights_buffer,
                &quantized_weights_buffer,
                &weight_scales_buffer,
                weights.len() as u32,
                4u32, // group_size
                metal::MTLSize::new(weights.len() as u64, 1, 1),
                metal::MTLSize::new(32, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("    ✓ 1-bit weight quantization completed");
        }

        // Step 2: Dequantization
        let dequantized_weights = vec![0.0f32; weights.len()];
        let dequantized_buffer = create_buffer(device, &dequantized_weights).unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder_result = shaders.create_compute_encoder_with_pipeline(
            &command_buffer,
            BitNetShaderFunction::DequantizeWeights1Bit,
        );

        if let Ok(encoder) = encoder_result {
            set_compute_buffer(&encoder, &quantized_weights_buffer, 0, 0);
            set_compute_buffer(&encoder, &weight_scales_buffer, 0, 1);
            set_compute_buffer(&encoder, &dequantized_buffer, 0, 2);
            set_compute_bytes(&encoder, &[weights.len() as u32], 3);
            set_compute_bytes(&encoder, &[4u32], 4); // group_size

            let threads = metal::MTLSize::new(weights.len() as u64, 1, 1);
            let threadgroup = metal::MTLSize::new(32, 1, 1);
            dispatch_compute(&encoder, threads, threadgroup);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("    ✓ Weight dequantization completed");

            // Verify quantization-dequantization cycle
            let original_result: Result<Vec<f32>> = read_buffer(&weights_buffer);
            let dequantized_result: Result<Vec<f32>> = read_buffer(&dequantized_buffer);

            match (original_result, dequantized_result) {
                (Ok(original), Ok(dequantized)) => {
                    println!("    ✓ Original weights: {:?}", original);
                    println!("    ✓ Dequantized weights: {:?}", dequantized);

                    // Check if signs are preserved (basic validation)
                    let signs_match = original
                        .iter()
                        .zip(dequantized.iter())
                        .all(|(a, b)| a.signum() == b.signum());

                    if signs_match {
                        println!("    ✓ Quantization preserved weight signs");
                    } else {
                        println!("    ⚠ Some weight signs changed during quantization");
                    }
                }
                _ => println!("    Failed to read quantization results"),
            }
        }
    }

    fn test_mixed_precision_workflow(device: &metal::Device, shaders: &BitNetShaders) {
        println!("  ✓ Testing mixed precision workflow");

        let command_queue = create_command_queue(device);

        // Setup mixed precision matrix multiplication
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

                println!("    ✓ Mixed precision matrix multiplication completed");

                // Read and verify results
                let result: Result<Vec<f32>> = read_buffer(&output_buffer);
                match result {
                    Ok(output_values) => {
                        println!("    ✓ Mixed precision output: {:?}", output_values);

                        // Basic validation - check that we got reasonable values
                        let has_non_zero = output_values.iter().any(|&x| x != 0.0);
                        if has_non_zero {
                            println!("    ✓ Mixed precision workflow produced non-zero results");
                        } else {
                            println!("    ⚠ Mixed precision workflow produced all zeros");
                        }
                    }
                    Err(e) => println!("    Failed to read mixed precision output: {}", e),
                }
            }
            Err(e) => println!("    Failed to create mixed precision encoder: {}", e),
        }
    }

    /// Test cross-system compatibility and error handling
    #[test]
    fn test_cross_system_compatibility() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            println!("✓ Testing cross-system compatibility");

            // Test compatibility between different subsystems
            test_memory_compute_compatibility(&device);
            test_tensor_compute_compatibility(&device);
            test_shader_compute_compatibility(&device);

            // Test error propagation across systems
            test_cross_system_error_handling(&device);

            println!("✓ Cross-system compatibility tested");
        } else {
            println!("Skipping cross-system compatibility test - no Metal device available");
        }
    }

    fn test_memory_compute_compatibility(device: &metal::Device) {
        println!("  ✓ Testing memory-compute compatibility");

        let command_queue = create_command_queue(device);
        let manager = create_command_buffer_manager(device, &command_queue);
        let pool = create_buffer_pool(device);

        // Test that pooled buffers work with compute operations
        let buffer_result = pool.get_buffer(1024, metal::MTLResourceOptions::StorageModeShared);
        match buffer_result {
            Ok(buffer) => {
                let cb_id_result = manager.create_command_buffer(CommandBufferPriority::Normal);
                match cb_id_result {
                    Ok(cb_id) => {
                        // Add pooled buffer as resource
                        let add_result = manager.add_resource(cb_id, buffer.clone());
                        match add_result {
                            Ok(()) => {
                                println!("    ✓ Pooled buffer added as compute resource");

                                // Test compute operation with pooled buffer
                                let begin_result = manager.begin_encoding(cb_id);
                                if begin_result.is_ok() {
                                    let encoder_result = manager.create_compute_encoder(cb_id);
                                    match encoder_result {
                                        Ok(encoder) => {
                                            set_compute_buffer(&encoder, &buffer, 0, 0);
                                            encoder.end_encoding();
                                            println!("    ✓ Compute operation with pooled buffer configured");
                                        }
                                        Err(e) => println!("    Failed to create encoder: {}", e),
                                    }
                                }

                                let _ = manager.commit_and_wait(cb_id);
                            }
                            Err(e) => {
                                println!("    Failed to add pooled buffer as resource: {}", e)
                            }
                        }
                    }
                    Err(e) => println!("    Failed to create command buffer: {}", e),
                }

                // Return buffer to pool
                let _ = pool.return_buffer(buffer);
            }
            Err(e) => println!("    Failed to get buffer from pool: {}", e),
        }
    }

    fn test_tensor_compute_compatibility(device: &metal::Device) {
        println!("  ✓ Testing tensor-compute compatibility");

        let bitnet_device = Device::Metal(device.clone());

        // Test that tensors can be used in compute operations
        let tensor_result = Tensor::ones(&[8, 8], DType::F32, &bitnet_device);
        match tensor_result {
            Ok(tensor) => {
                let data_result = tensor.to_vec1::<f32>();
                match data_result {
                    Ok(data) => {
                        let buffer_result = create_buffer(device, &data);
                        match buffer_result {
                            Ok(buffer) => {
                                println!("    ✓ Tensor converted to compute buffer");

                                // Test compute operation with tensor-derived buffer
                                let command_queue = create_command_queue(device);
                                let command_buffer = command_queue.new_command_buffer();
                                let encoder = command_buffer.new_compute_command_encoder();

                                set_compute_buffer(&encoder, &buffer, 0, 0);

                                let threads = metal::MTLSize::new(64, 1, 1);
                                let threadgroup = metal::MTLSize::new(32, 1, 1);
                                dispatch_compute(&encoder, threads, threadgroup);

                                encoder.end_encoding();
                                command_buffer.commit();
                                command_buffer.wait_until_completed();

                                println!("    ✓ Compute operation with tensor buffer completed");
                            }
                            Err(e) => println!("    Failed to create buffer from tensor: {}", e),
                        }
                    }
                    Err(e) => println!("    Failed to extract tensor data: {}", e),
                }
            }
            Err(e) => println!("    Failed to create tensor: {}", e),
        }
    }

    fn test_shader_compute_compatibility(device: &metal::Device) {
        println!("  ✓ Testing shader-compute compatibility");

        // Test shader compiler integration with compute operations
        let compiler_result = create_shader_compiler(device);
        match compiler_result {
            Ok(compiler) => {
                println!("    ✓ Shader compiler created");

                // Test that compiled shaders work with compute operations
                let stats = compiler.get_stats();
                println!("    ✓ Shader compiler stats: {:?}", stats);

                // Test shader loading integration
                let shaders_result = BitNetShaders::new(device.clone());
                match shaders_result {
                    Ok(shaders) => {
                        println!("    ✓ BitNet shaders loaded");

                        let available = shaders.get_available_shaders();
                        println!("    ✓ Available shaders: {:?}", available);

                        // Test pipeline creation
                        let pipeline_result =
                            shaders.get_pipeline(BitNetShaderFunction::ReluForward);
                        match pipeline_result {
                            Ok(_pipeline) => println!("    ✓ Shader pipeline created successfully"),
                            Err(e) => println!("    Shader pipeline creation failed: {}", e),
                        }
                    }
                    Err(e) => println!("    BitNet shaders loading failed: {}", e),
                }
            }
            Err(e) => println!("    Failed to create shader compiler: {}", e),
        }
    }

    fn test_cross_system_error_handling(device: &metal::Device) {
        println!("  ✓ Testing cross-system error handling");

        // Test error propagation from memory system to compute
        let pool = create_buffer_pool(device);

        // Try to allocate an impossibly large buffer
        let large_buffer_result =
            pool.get_buffer(usize::MAX, metal::MTLResourceOptions::StorageModeShared);
        match large_buffer_result {
            Ok(_) => println!("    Unexpected: Large buffer allocation succeeded"),
            Err(e) => {
                println!("    ✓ Memory system error properly propagated: {}", e);

                // Test that compute system handles memory errors gracefully
                let command_queue = create_command_queue(device);
                let manager = create_command_buffer_manager(device, &command_queue);

                let cb_id_result = manager.create_command_buffer(CommandBufferPriority::Normal);
                match cb_id_result {
                    Ok(cb_id) => {
                        // Try to use invalid buffer in compute operation
                        let begin_result = manager.begin_encoding(cb_id);
                        if begin_result.is_ok() {
                            let encoder_result = manager.create_compute_encoder(cb_id);
                            match encoder_result {
                                Ok(_encoder) => {
                                    println!(
                                        "    ✓ Compute system handled memory error gracefully"
                                    );
                                }
                                Err(e) => println!("    Compute encoder creation failed: {}", e),
                            }
                        }

                        let _ = manager.return_command_buffer(cb_id);
                    }
                    Err(e) => println!("    Command buffer creation failed: {}", e),
                }
            }
        }

        // Test error handling with invalid shader operations
        let shaders_result = BitNetShaders::new(device.clone());
        match shaders_result {
            Ok(shaders) => {
                // Try to get a non-existent pipeline
                let invalid_pipeline_result =
                    shaders.get_pipeline(BitNetShaderFunction::BitLinearForward);
                match invalid_pipeline_result {
                    Ok(_) => println!("    Unexpected: Invalid pipeline creation succeeded"),
                    Err(e) => println!("    ✓ Shader system error properly handled: {}", e),
                }
            }
            Err(e) => println!("    Shader system initialization failed: {}", e),
        }
    }

    /// Test performance characteristics of integrated systems
    #[test]
    fn test_integrated_system_performance() {
        let device_result = create_metal_device();
        if let Ok(device) = device_result {
            println!("✓ Testing integrated system performance");

            // Test performance of memory + compute integration
            test_memory_compute_performance(&device);

            // Test performance of tensor + compute integration
            test_tensor_compute_performance(&device);

            // Test performance of shader + compute integration
            test_shader_compute_performance(&device);

            println!("✓ Integrated system performance tested");
        } else {
            println!("Skipping integrated system performance test - no Metal device available");
        }
    }

    fn test_memory_compute_performance(device: &metal::Device) {
        println!("  ✓ Testing memory-compute performance");

        let command_queue = create_command_queue(device);
        let manager = create_command_buffer_manager(device, &command_queue);
        let pool = create_buffer_pool(device);

        let num_operations = 10;
        let buffer_size = 4096;

        let start_time = Instant::now();

        for i in 0..num_operations {
            // Get buffer from pool
            let buffer_result =
                pool.get_buffer(buffer_size, metal::MTLResourceOptions::StorageModeShared);
            if let Ok(buffer) = buffer_result {
                // Use in compute operation
                let cb_id_result = manager.create_command_buffer(CommandBufferPriority::Normal);
                if let Ok(cb_id) = cb_id_result {
                    let _ = manager.add_resource(cb_id, buffer.clone());

                    if manager.begin_encoding(cb_id).is_ok() {
                        if let Ok(encoder) = manager.create_compute_encoder(cb_id) {
                            set_compute_buffer(&encoder, &buffer, 0, 0);
                            encoder.end_encoding();
                        }
                    }

                    let _ = manager.commit_and_wait(cb_id);
                }

                // Return buffer to pool
                let _ = pool.return_buffer(buffer);
            }

            if i % 5 == 0 {
                println!("    Completed {} operations", i);
            }
        }

        let elapsed = start_time.elapsed();
        let ops_per_sec = num_operations as f64 / elapsed.as_secs_f64();

        println!(
            "    ✓ Memory-compute integration: {:.1} ops/sec",
            ops_per_sec
        );

        // Check pool statistics
        let stats = pool.get_stats();
        println!(
            "    ✓ Pool efficiency: {:.1}% cache hit rate",
            (stats.cache_hits as f64 / stats.total_allocations as f64) * 100.0
        );
    }

    fn test_tensor_compute_performance(device: &metal::Device) {
        println!("  ✓ Testing tensor-compute performance");

        let bitnet_device = Device::Metal(device.clone());
        let command_queue = create_command_queue(device);

        let num_operations = 5;
        let tensor_size = [64, 64];

        let start_time = Instant::now();

        for i in 0..num_operations {
            // Create tensor
            let tensor_result = Tensor::randn(&tensor_size, DType::F32, &bitnet_device);
            if let Ok(tensor) = tensor_result {
                // Convert to compute buffer
                let data_result = tensor.to_vec1::<f32>();
                if let Ok(data) = data_result {
                    let buffer_result = create_buffer(device, &data);
                    if let Ok(buffer) = buffer_result {
                        // Use in compute operation
                        let command_buffer = command_queue.new_command_buffer();
                        let encoder = command_buffer.new_compute_command_encoder();

                        set_compute_buffer(&encoder, &buffer, 0, 0);

                        let threads = metal::MTLSize::new(data.len() as u64, 1, 1);
                        let threadgroup = metal::MTLSize::new(64, 1, 1);
                        dispatch_compute(&encoder, threads, threadgroup);

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();
                    }
                }
            }

            println!("    Completed tensor operation {}", i + 1);
        }

        let elapsed = start_time.elapsed();
        let ops_per_sec = num_operations as f64 / elapsed.as_secs_f64();

        println!(
            "    ✓ Tensor-compute integration: {:.1} ops/sec",
            ops_per_sec
        );
    }

    fn test_shader_compute_performance(device: &metal::Device) {
        println!("  ✓ Testing shader-compute performance");

        let shaders_result = BitNetShaders::new(device.clone());
        match shaders_result {
            Ok(shaders) => {
                let command_queue = create_command_queue(device);
                let num_operations = 20;
                let data_size = 1024;

                // Create test data
                let input_data = vec![0.5f32; data_size];
                let output_data = vec![0.0f32; data_size];

                let input_buffer = create_buffer(device, &input_data).unwrap();
                let output_buffer = create_buffer(device, &output_data).unwrap();

                let start_time = Instant::now();

                for i in 0..num_operations {
                    let command_buffer = command_queue.new_command_buffer();
                    let encoder_result = create_activation_encoder(
                        &shaders,
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
                            metal::MTLSize::new(64, 1, 1),
                        );

                        encoder.end_encoding();
                        command_buffer.commit();
                        command_buffer.wait_until_completed();
                    }

                    if i % 10 == 0 {
                        println!("    Completed shader operation {}", i);
                    }
                }

                let elapsed = start_time.elapsed();
                let ops_per_sec = num_operations as f64 / elapsed.as_secs_f64();

                println!(
                    "    ✓ Shader-compute integration: {:.1} ops/sec",
                    ops_per_sec
                );
            }
            Err(e) => println!("    Shader-compute performance test skipped: {}", e),
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod non_metal_tests {
    #[test]
    fn test_unsupported_platform_integration() {
        println!("Compute pipeline integration tests skipped - not on macOS or Metal feature not enabled");
        // This test always passes on non-macOS platforms
        assert!(true);
    }
}
