//! # BitNet Metal Shader Compilation Demo
//!
//! This example demonstrates how to use the BitNet Metal shader compilation pipeline
//! to load, compile, and execute BitNet operations on GPU.

use anyhow::Result;
use bitnet_core::metal::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<()> {
    println!("BitNet Metal Shader Compilation Demo");
    println!("====================================");

    // Initialize Metal context
    println!("1. Initializing Metal context...");
    let (device, command_queue, _library) = initialize_metal_context()?;
    println!("   ✓ Metal device: {}", device.name());

    // Create BitNet shader collection
    println!("\n2. Loading BitNet shaders...");
    let shaders = match BitNetShaders::new(device.clone()) {
        Ok(shaders) => {
            println!("   ✓ Successfully loaded BitNet shaders");
            shaders
        }
        Err(e) => {
            println!("   ⚠ Failed to load shaders (expected if shader files don't exist): {}", e);
            println!("   This is normal in a fresh checkout - shader files need to be in the correct location.");
            return Ok(());
        }
    };

    // List available shaders
    println!("\n3. Available shaders:");
    let available_shaders = shaders.get_available_shaders();
    for shader_name in &available_shaders {
        println!("   • {}", shader_name);
        
        // List functions in each shader
        if let Ok(functions) = shaders.get_shader_functions(shader_name) {
            for function in functions {
                println!("     - {}", function);
            }
        }
    }

    // Demonstrate pipeline creation
    println!("\n4. Creating compute pipelines...");
    let test_functions = [
        BitNetShaderFunction::BitLinearForward,
        BitNetShaderFunction::QuantizeWeights1Bit,
        BitNetShaderFunction::ReluForward,
    ];

    for function in &test_functions {
        match shaders.get_pipeline(*function) {
            Ok(_pipeline) => {
                println!("   ✓ Created pipeline for {}", function.function_name());
            }
            Err(e) => {
                println!("   ✗ Failed to create pipeline for {}: {}", function.function_name(), e);
            }
        }
    }

    // Demonstrate buffer creation and basic compute setup
    println!("\n5. Setting up compute operation...");
    
    // Create test data
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = create_buffer(&device, &input_data)?;
    println!("   ✓ Created input buffer with {} elements", input_data.len());

    // Create output buffer
    let output_buffer = create_empty_buffer(
        &device,
        input_data.len() * std::mem::size_of::<f32>(),
        metal::MTLResourceOptions::StorageModeShared,
    )?;
    println!("   ✓ Created output buffer");

    // Create command buffer
    let command_buffer = command_queue.new_command_buffer();
    println!("   ✓ Created command buffer");

    // Try to create a ReLU forward encoder (simplest activation function)
    match shaders.create_compute_encoder_with_pipeline(command_buffer, BitNetShaderFunction::ReluForward) {
        Ok(encoder) => {
            println!("   ✓ Created ReLU forward compute encoder");
            
            // Set up the compute operation
            encoder.set_buffer(0, Some(&input_buffer), 0);  // input
            encoder.set_buffer(1, Some(&output_buffer), 0); // output
            set_compute_bytes(&encoder, &[input_data.len() as u32], 2); // count
            
            // Calculate dispatch parameters
            if let Ok((threads, threadgroup)) = shaders.calculate_dispatch_params(
                BitNetShaderFunction::ReluForward,
                input_data.len(),
            ) {
                println!("   ✓ Calculated dispatch: threads={:?}, threadgroup={:?}", threads, threadgroup);
                
                // Dispatch the compute operation
                dispatch_compute(&encoder, threads, threadgroup);
                encoder.end_encoding();
                
                // Execute and wait
                command_buffer.commit();
                command_buffer.wait_until_completed();
                
                // Read results
                let output_data: Vec<f32> = read_buffer(&output_buffer)?;
                println!("   ✓ ReLU operation completed");
                println!("     Input:  {:?}", input_data);
                println!("     Output: {:?}", output_data);
                
                // Verify ReLU operation (should be max(0, x))
                let expected: Vec<f32> = input_data.iter().map(|&x| x.max(0.0)).collect();
                if output_data == expected {
                    println!("   ✓ ReLU operation result is correct!");
                } else {
                    println!("   ⚠ ReLU operation result differs from expected");
                }
            } else {
                println!("   ✗ Failed to calculate dispatch parameters");
            }
        }
        Err(e) => {
            println!("   ✗ Failed to create compute encoder: {}", e);
        }
    }

    // Demonstrate shader compiler directly
    println!("\n6. Using shader compiler directly...");
    let compiler = create_shader_compiler(&device)?;
    let stats = compiler.get_stats();
    println!("   Compiler stats:");
    println!("   • Cached shaders: {}", stats.cached_shaders);
    println!("   • Total functions: {}", stats.total_functions);
    println!("   • Shader directory: {:?}", stats.shader_directory);
    if let Some(cache_dir) = &stats.cache_directory {
        println!("   • Cache directory: {:?}", cache_dir);
    }

    println!("\n7. Demo completed successfully! 🎉");
    println!("\nNext steps:");
    println!("• Ensure shader files are in the correct location");
    println!("• Use BitNetShaders for high-level operations");
    println!("• Use ShaderCompiler for custom shader compilation");
    println!("• Check the shader cache for compiled artifacts");

    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() -> Result<()> {
    println!("BitNet Metal Shader Compilation Demo");
    println!("====================================");
    println!("This demo requires macOS with Metal support.");
    println!("Current platform does not support Metal operations.");
    Ok(())
}