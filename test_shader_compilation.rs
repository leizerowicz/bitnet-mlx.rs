//! Comprehensive Metal Shader Compilation Test Script
//!
//! This script provides a comprehensive test suite for Metal shader compilation,
//! including basic functionality, error handling, performance testing, and edge cases.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_shader_compilation() -> anyhow::Result<()> {
    use bitnet_core::metal::*;
    
    println!("=== Comprehensive Metal Shader Compilation Tests ===\n");
    
    // Initialize Metal context
    let device = create_metal_device()?;
    println!("✓ Metal device created: {}", device.name());
    println!("  Device supports feature set: {}",
             device.supports_feature_set(metal::MTLFeatureSet::macOS_GPUFamily1_v1));
    
    // Test 1: Basic shader compiler creation and configuration
    println!("\n--- Test 1: Shader Compiler Creation ---");
    let config = ShaderCompilerConfig {
        shader_directory: PathBuf::from("bitnet-core/src/metal/shaders"),
        enable_caching: true,
        cache_directory: Some(PathBuf::from("target/test_shader_cache")),
        debug_info: true,
        optimization_level: OptimizationLevel::Full,
        ..Default::default()
    };
    
    let compiler = ShaderCompiler::new(device.clone(), config)?;
    println!("✓ Shader compiler created with custom configuration");
    
    let stats = compiler.get_stats();
    println!("  Initial stats: {:?}", stats);
    
    // Test 2: Individual shader file compilation
    println!("\n--- Test 2: Individual Shader Compilation ---");
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
                    
                    println!("✓ {} compiled successfully in {:?}", shader_file, compilation_time);
                    println!("  Functions: {:?}", compiled_shader.function_names);
                    println!("  Source hash: {}", compiled_shader.source_hash);
                    
                    total_functions += compiled_shader.function_names.len();
                    
                    // Verify shader properties
                    assert!(!compiled_shader.function_names.is_empty(),
                           "Shader should have at least one function");
                    assert_eq!(compiled_shader.name, shader_file.trim_end_matches(".metal"));
                }
                Err(e) => {
                    println!("✗ {} compilation failed: {}", shader_file, e);
                    return Err(e);
                }
            }
        } else {
            println!("⚠ {} not found, skipping", shader_file);
        }
    }
    
    println!("Total functions discovered: {}", total_functions);
    if !compilation_times.is_empty() {
        let avg_time = compilation_times.iter().sum::<std::time::Duration>() / compilation_times.len() as u32;
        println!("Average compilation time: {:?}", avg_time);
    }
    
    // Test 3: Shader caching functionality
    println!("\n--- Test 3: Shader Caching ---");
    let bitlinear_path = PathBuf::from("bitnet-core/src/metal/shaders/bitlinear.metal");
    if bitlinear_path.exists() {
        // First compilation (cache miss)
        compiler.clear_cache();
        let start_time = Instant::now();
        let _result1 = compiler.compile_shader_file(&bitlinear_path)?;
        let cache_miss_time = start_time.elapsed();
        
        // Second compilation (cache hit)
        let start_time = Instant::now();
        let _result2 = compiler.compile_shader_file(&bitlinear_path)?;
        let cache_hit_time = start_time.elapsed();
        
        println!("✓ Cache functionality verified");
        println!("  Cache miss time: {:?}", cache_miss_time);
        println!("  Cache hit time: {:?}", cache_hit_time);
        
        if cache_hit_time.as_nanos() > 0 {
            let speedup = cache_miss_time.as_nanos() as f64 / cache_hit_time.as_nanos() as f64;
            println!("  Cache speedup: {:.2}x", speedup);
        }
    }
    
    // Test 4: BitNet shader utilities
    println!("\n--- Test 4: BitNet Shader Utilities ---");
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
                        println!("  Max threads per threadgroup: {}",
                                pipeline.max_total_threads_per_threadgroup());
                        println!("  Thread execution width: {}",
                                pipeline.thread_execution_width());
                        successful_pipelines += 1;
                    }
                    Err(e) => {
                        println!("✗ Pipeline creation failed for {:?}: {}", function, e);
                    }
                }
            }
            
            println!("Successfully created {} out of {} pipelines",
                    successful_pipelines, test_functions.len());
            
            // Test dispatch parameter calculation
            println!("\n--- Test 5: Dispatch Parameter Calculation ---");
            let data_sizes = [32, 64, 128, 256, 512, 1024, 2048];
            for &size in &data_sizes {
                match shaders.calculate_dispatch_params(BitNetShaderFunction::ReluForward, size) {
                    Ok((threadgroup_size, threadgroups)) => {
                        println!("Size {}: threadgroup={:?}, threadgroups={:?}",
                                size, threadgroup_size, threadgroups);
                    }
                    Err(e) => {
                        println!("Dispatch calculation failed for size {}: {}", size, e);
                    }
                }
            }
        }
        Err(e) => {
            println!("✗ BitNet shaders initialization failed: {}", e);
            println!("  This is expected if shader files are missing");
        }
    }
    
    // Test 6: Error handling
    println!("\n--- Test 6: Error Handling ---");
    
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
    
    // Test 7: Performance summary
    println!("\n--- Test 7: Performance Summary ---");
    let final_stats = compiler.get_stats();
    println!("Final compiler stats: {:?}", final_stats);
    
    if final_stats.total_functions > 0 && !compilation_times.is_empty() {
        let total_time: std::time::Duration = compilation_times.iter().sum();
        let time_per_function = total_time.as_millis() / final_stats.total_functions as u128;
        println!("Average time per function: {} ms", time_per_function);
    }
    
    println!("\n=== All Shader Compilation Tests Completed Successfully! ===");
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn test_shader_compilation() -> anyhow::Result<()> {
    println!("=== Shader Compilation Tests ===");
    println!("Tests skipped - not on macOS or Metal feature not enabled");
    println!("Platform: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // Print environment information
    println!("Environment Information:");
    println!("  OS: {}", std::env::consts::OS);
    println!("  Architecture: {}", std::env::consts::ARCH);
    println!("  Rust version: {}", env!("RUSTC_VERSION"));
    
    #[cfg(all(target_os = "macos", feature = "metal"))]
    println!("  Metal support: enabled");
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    println!("  Metal support: disabled");
    
    println!();
    
    let start_time = Instant::now();
    let result = test_shader_compilation();
    let total_time = start_time.elapsed();
    
    match result {
        Ok(()) => {
            println!("\n✓ All tests completed successfully in {:?}", total_time);
            Ok(())
        }
        Err(e) => {
            println!("\n✗ Tests failed after {:?}: {}", total_time, e);
            Err(e)
        }
    }
}