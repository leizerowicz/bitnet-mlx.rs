//! # MPS Integration Example
//! 
//! Demonstrates the Metal Performance Shaders integration for BitNet operations.

use anyhow::Result;

#[cfg(all(target_os = "macos", feature = "mps"))]
use bitnet_metal::{
    BitNetMPSManager, MemoryUsageHint, SharingMode, ActivationType,
    BitLinearConfig, ImageProcessingConfig, OptimizationTarget, PowerTarget,
};

fn main() -> Result<()> {
    println!("BitNet MPS Integration Example");
    println!("==============================");
    
    // Demonstrate MPS integration
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        example_mps_integration()?;
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("MPS integration is only available on macOS with the 'mps' feature enabled.");
        println!("Current platform: {}", std::env::consts::OS);
        println!("Available features: metal, unified-memory, ane");
    }
    
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "mps"))]
fn example_mps_integration() -> Result<()> {
    println!("Creating BitNet MPS Manager...");
    
    // Create MPS manager
    let mps_manager = BitNetMPSManager::new()?;
    
    // Display system information
    let system_info = mps_manager.system_info();
    println!("System Information:");
    println!("  Device: {}", system_info.device_name);
    println!("  ANE Support: {}", system_info.supports_ane);
    println!("  Unified Memory: {} GB", system_info.unified_memory_size / (1024 * 1024 * 1024));
    println!("  MPS Version: {}", system_info.mps_version);
    
    // Demonstrate framework capabilities
    let framework = mps_manager.framework();
    let capabilities = framework.capabilities();
    println!("\nMPS Capabilities:");
    println!("  Neural Network Support: {}", capabilities.supports_neural_network);
    println!("  Matrix Multiplication: {}", capabilities.supports_matrix_multiplication);
    println!("  Convolution Support: {}", capabilities.supports_convolution);
    println!("  Image Processing: {}", capabilities.supports_image_processing);
    println!("  Graph API: {}", capabilities.supports_graph_api);
    println!("  Max Texture Size: {}", capabilities.max_texture_size);
    println!("  Unified Memory: {}", capabilities.unified_memory);
    
    // Demonstrate unified memory management
    let unified_memory = mps_manager.unified_memory();
    let total_memory = unified_memory.total_memory();
    println!("\nUnified Memory:");
    println!("  Total Memory: {} GB", total_memory / (1024 * 1024 * 1024));
    
    let usage_stats = unified_memory.memory_usage();
    println!("  Current Usage: {} MB", usage_stats.total_allocated_bytes / (1024 * 1024));
    println!("  Active Allocations: {}", usage_stats.allocation_count);
    
    // Example unified memory allocation
    println!("\nTesting unified memory allocation...");
    let allocation = unified_memory.allocate_unified(
        1024 * 1024, // 1MB
        MemoryUsageHint::GPUOnly,
        SharingMode::Exclusive,
    )?;
    println!("  Allocated 1MB for GPU-only usage");
    
    // Bandwidth optimization
    println!("\nAnalyzing memory bandwidth...");
    let bandwidth_optimization = unified_memory.optimize_bandwidth()?;
    println!("  Current CPU->GPU bandwidth: {:.1} GB/s", bandwidth_optimization.current_utilization.cpu_to_gpu_gb_s);
    println!("  Current GPU->CPU bandwidth: {:.1} GB/s", bandwidth_optimization.current_utilization.gpu_to_cpu_gb_s);
    println!("  Unified memory efficiency: {:.1}%", bandwidth_optimization.current_utilization.unified_memory_efficiency * 100.0);
    println!("  Estimated improvement: {:.1}%", bandwidth_optimization.estimated_improvement * 100.0);
    
    // Clean up
    unified_memory.deallocate_unified(allocation)?;
    println!("  Memory deallocated successfully");
    
    // Demonstrate ANE integration if available
    if let Some(ane) = mps_manager.ane_integration() {
        println!("\nApple Neural Engine Integration:");
        let ane_capabilities = ane.capabilities();
        println!("  ANE Available: {}", ane_capabilities.is_available);
        println!("  ANE Version: {:?}", ane_capabilities.version);
        println!("  Max Operations/sec: {}", ane_capabilities.max_operations_per_second);
        println!("  Supported Data Types: {} types", ane_capabilities.supported_data_types.len());
        
        // Power optimization
        println!("  Optimizing for balanced power...");
        ane.optimize_power(PowerTarget::Balanced)?;
        println!("  Thermal status: {:?}", ane.thermal_status());
    } else {
        println!("\nApple Neural Engine: Not available on this device");
    }
    
    // Demonstrate configuration examples
    println!("\nConfiguration Examples:");
    
    // BitLinear layer configuration
    let bitlinear_config = BitLinearConfig {
        input_features: 768,
        output_features: 2048,
        batch_size: 32,
        use_bias: true,
        eps: 1e-5,
    };
    println!("  BitLinear Config: {}x{} (batch: {})", 
             bitlinear_config.input_features, 
             bitlinear_config.output_features,
             bitlinear_config.batch_size);
    
    // Image processing configuration
    let image_config = ImageProcessingConfig::default();
    println!("  Image Processing: {}x{}x{} channels", 
             image_config.width, 
             image_config.height, 
             image_config.channels);
    
    println!("\n✅ MPS Integration example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        let result = main();
        assert!(result.is_ok(), "Example should run without errors: {:?}", result.err());
    }
}
