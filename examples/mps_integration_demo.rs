//! # MPS Integration Production Demo
//! 
//! Demonstrates production-ready Metal Performance Shaders integration for BitNet operations
//! with comprehensive error handling, device capability detection, and fallback strategies.

use anyhow::Result;
use std::time::{Duration, Instant};

#[cfg(all(target_os = "macos", feature = "mps"))]
use bitnet_metal::{
    BitNetMPSManager, FallbackStrategy, CapabilityRequirements, DeviceCompatibilityChecker,
    ANEDataType, OptimizationTarget, PowerTarget, MemoryUsageHint, SharingMode, 
    ActivationType, BitLinearConfig, ImageProcessingConfig,
};

fn main() -> Result<()> {
    println!("🚀 BitNet MPS Production Integration Demo");
    println!("==========================================");
    
    // Platform compatibility check
    check_platform_compatibility();
    
    // Basic MPS manager creation with error handling
    demo_basic_mps_creation()?;
    
    // Custom fallback strategy demonstration
    demo_custom_fallback_strategy()?;
    
    // Device capability validation
    demo_capability_validation()?;
    
    // Error recovery demonstration
    demo_error_recovery()?;
    
    // ANE integration examples
    demo_ane_integration()?;
    
    // Performance monitoring
    demo_performance_monitoring()?;
    
    // Cross-platform compatibility
    demo_cross_platform_compatibility();
    
    println!("\n✅ MPS Production Integration Demo completed successfully!");
    Ok(())
}

/// Check platform compatibility and display detailed information
fn check_platform_compatibility() {
    println!("\n📋 Platform Compatibility Check");
    println!("--------------------------------");
    
    let platform_info = DeviceCompatibilityChecker::get_platform_info();
    println!("Platform: {}", platform_info);
    
    println!("MPS Available: {}", DeviceCompatibilityChecker::is_mps_available());
    println!("ANE Available: {}", DeviceCompatibilityChecker::is_ane_available());
    
    if platform_info.mps_available {
        println!("✅ MPS is available on this platform");
    } else {
        println!("⚠️  MPS not available - will demonstrate fallback mechanisms");
    }
}

/// Demonstrate basic MPS manager creation with production error handling
fn demo_basic_mps_creation() -> Result<()> {
    println!("\n🔧 Basic MPS Manager Creation");
    println!("------------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        match BitNetMPSManager::new() {
            Ok(manager) => {
                let system_info = manager.system_info();
                println!("✅ MPS Manager created successfully");
                println!("Device: {}", system_info.device_name);
                println!("MPS Version: {}", system_info.mps_version);
                println!("ANE Support: {}", system_info.supports_ane);
                println!("Unified Memory: {} MB", system_info.unified_memory_size / (1024 * 1024));
                
                // Display recovery statistics
                println!("Recovery Stats: {:?}", system_info.recovery_stats);
            }
            Err(e) => {
                println!("⚠️  MPS Manager creation failed: {}", e);
                println!("This is expected on non-Apple Silicon or when MPS is unavailable");
            }
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  MPS not available on this platform - using CPU fallback");
    }
    
    Ok(())
}

/// Demonstrate custom fallback strategy configuration
fn demo_custom_fallback_strategy() -> Result<()> {
    println!("\n⚙️  Custom Fallback Strategy");
    println!("----------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        // Create a custom fallback strategy for production use
        let production_strategy = FallbackStrategy {
            enable_cpu_fallback: true,
            enable_metal_fallback: true,
            max_retry_attempts: 5,
            retry_delay: Duration::from_millis(100),
            monitor_fallback_performance: true,
        };
        
        println!("📝 Production Fallback Strategy:");
        println!("  - CPU Fallback: {}", production_strategy.enable_cpu_fallback);
        println!("  - Metal Fallback: {}", production_strategy.enable_metal_fallback);
        println!("  - Max Retries: {}", production_strategy.max_retry_attempts);
        println!("  - Retry Delay: {:?}", production_strategy.retry_delay);
        println!("  - Performance Monitoring: {}", production_strategy.monitor_fallback_performance);
        
        match BitNetMPSManager::new_with_fallback_strategy(production_strategy) {
            Ok(manager) => {
                println!("✅ MPS Manager created with custom fallback strategy");
                
                // Demonstrate recovery statistics access
                let recovery_stats = manager.error_recovery().get_recovery_stats()?;
                println!("Initial recovery stats: {:?}", recovery_stats);
            }
            Err(e) => {
                println!("⚠️  Manager creation failed (fallback available): {}", e);
            }
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  Custom fallback demonstration requires macOS with MPS support");
    }
    
    Ok(())
}

/// Demonstrate device capability validation
fn demo_capability_validation() -> Result<()> {
    println!("\n🔍 Device Capability Validation");
    println!("--------------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        if let Ok(manager) = BitNetMPSManager::new() {
            // Define different capability requirements for various use cases
            let basic_requirements = CapabilityRequirements {
                neural_network_support: false,
                matrix_multiplication: true,
                convolution_support: false,
                graph_api: false,
                minimum_memory_mb: 256,
                ane_support: false,
            };
            
            let advanced_requirements = CapabilityRequirements {
                neural_network_support: true,
                matrix_multiplication: true,
                convolution_support: true,
                graph_api: true,
                minimum_memory_mb: 1024,
                ane_support: true,
            };
            
            println!("📋 Testing Basic Requirements:");
            match manager.validate_operation_requirements(&basic_requirements) {
                Ok(()) => println!("  ✅ Basic requirements met"),
                Err(e) => println!("  ❌ Basic requirements not met: {}", e),
            }
            
            println!("📋 Testing Advanced Requirements:");
            match manager.validate_operation_requirements(&advanced_requirements) {
                Ok(()) => println!("  ✅ Advanced requirements met"),
                Err(e) => println!("  ❌ Advanced requirements not met: {}", e),
            }
            
            // Check framework capabilities
            let capabilities = manager.framework().capabilities();
            let compatibility_score = capabilities.compatibility_score(&advanced_requirements);
            println!("📊 Compatibility Score: {:.1}%", compatibility_score * 100.0);
            
            if let Some(suggestion) = capabilities.suggest_fallback(&advanced_requirements) {
                println!("💡 Suggested Fallback: {}", suggestion);
            }
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  Capability validation requires macOS with MPS support");
    }
    
    Ok(())
}

/// Demonstrate error recovery mechanisms
fn demo_error_recovery() -> Result<()> {
    println!("\n🔄 Error Recovery Demonstration");
    println!("-------------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        if let Ok(manager) = BitNetMPSManager::new() {
            println!("📝 Demonstrating execute_with_recovery...");
            
            // Simulate a potentially failing operation
            let result = manager.execute_with_recovery("demo_operation", || {
                // Simulate some computation that might fail
                let start = Instant::now();
                std::thread::sleep(Duration::from_millis(10));
                
                // Simulate random failure for demo purposes
                if start.elapsed().as_millis() % 3 == 0 {
                    Err(anyhow::anyhow!("Simulated operation failure"))
                } else {
                    Ok("Operation completed successfully".to_string())
                }
            });
            
            match result {
                Ok(message) => println!("  ✅ {}", message),
                Err(e) => println!("  ❌ Operation failed after recovery attempts: {}", e),
            }
            
            // Display updated recovery statistics
            let recovery_stats = manager.error_recovery().get_recovery_stats()?;
            println!("📊 Recovery Statistics:");
            println!("  - Total Errors: {}", recovery_stats.total_errors);
            println!("  - Recent Errors: {}", recovery_stats.recent_errors);
            println!("  - CPU Fallbacks: {}", recovery_stats.cpu_fallback_count);
            println!("  - Metal Fallbacks: {}", recovery_stats.metal_fallback_count);
            println!("  - Total Recovery Time: {:?}", recovery_stats.total_recovery_time);
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  Error recovery demonstration requires macOS with MPS support");
    }
    
    Ok(())
}

/// Demonstrate Apple Neural Engine integration
fn demo_ane_integration() -> Result<()> {
    println!("\n🧠 Apple Neural Engine Integration");
    println!("----------------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        if let Ok(manager) = BitNetMPSManager::new() {
            if let Some(ane) = manager.ane_integration() {
                let capabilities = ane.capabilities();
                
                println!("✅ ANE Available:");
                println!("  - Version: {:?}", capabilities.version);
                println!("  - Max Operations/sec: {} TOPS", capabilities.max_operations_per_second / 1_000_000_000_000);
                println!("  - Supported Data Types: {:?}", capabilities.supported_data_types);
                println!("  - Max Model Size: {} MB", capabilities.max_model_size_mb);
                
                // Test ANE requirements
                let required_data_types = vec![ANEDataType::Float16, ANEDataType::Int8];
                let required_tops = 1_000_000_000_000; // 1 TOPS
                let required_model_size = 64; // 64MB
                
                println!("📋 Testing ANE Requirements:");
                if capabilities.meets_requirements(required_tops, &required_data_types, required_model_size) {
                    println!("  ✅ ANE meets all requirements");
                } else {
                    let mismatches = capabilities.get_requirement_mismatch(
                        required_tops, 
                        &required_data_types, 
                        required_model_size
                    );
                    println!("  ❌ ANE requirement mismatches:");
                    for mismatch in mismatches {
                        println!("    - {}", mismatch);
                    }
                }
                
                let compatibility = capabilities.compatibility_score(required_tops, &required_data_types, required_model_size);
                println!("📊 ANE Compatibility Score: {:.1}%", compatibility * 100.0);
                
                // Thermal status check
                println!("🌡️  Thermal Status: {:?}", ane.thermal_status());
            } else {
                println!("⚠️  ANE not available on this device");
                println!("     This is normal for Intel Macs or older Apple Silicon");
            }
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  ANE integration requires macOS with MPS support");
    }
    
    Ok(())
}

/// Demonstrate performance monitoring capabilities
fn demo_performance_monitoring() -> Result<()> {
    println!("\n📊 Performance Monitoring");
    println!("-------------------------");
    
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        if let Ok(manager) = BitNetMPSManager::new() {
            println!("🔍 System Performance Analysis:");
            
            let system_info = manager.system_info();
            println!("  - Device: {}", system_info.device_name);
            println!("  - Unified Memory: {} GB", system_info.unified_memory_size / (1024 * 1024 * 1024));
            
            // Demonstrate capability scoring for different workloads
            let workloads = vec![
                ("Lightweight Inference", CapabilityRequirements {
                    neural_network_support: true,
                    matrix_multiplication: true,
                    convolution_support: false,
                    graph_api: false,
                    minimum_memory_mb: 128,
                    ane_support: false,
                }),
                ("Computer Vision", CapabilityRequirements {
                    neural_network_support: true,
                    matrix_multiplication: true,
                    convolution_support: true,
                    graph_api: true,
                    minimum_memory_mb: 512,
                    ane_support: false,
                }),
                ("High-Performance Training", CapabilityRequirements {
                    neural_network_support: true,
                    matrix_multiplication: true,
                    convolution_support: true,
                    graph_api: true,
                    minimum_memory_mb: 2048,
                    ane_support: true,
                }),
            ];
            
            println!("🎯 Workload Compatibility Analysis:");
            for (workload_name, requirements) in workloads {
                let score = manager.framework().capabilities().compatibility_score(&requirements);
                let status = if score >= 0.9 {
                    "✅ Excellent"
                } else if score >= 0.7 {
                    "⚡ Good"
                } else if score >= 0.5 {
                    "⚠️  Limited"
                } else {
                    "❌ Poor"
                };
                println!("  - {}: {:.1}% {}", workload_name, score * 100.0, status);
            }
        }
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        println!("ℹ️  Performance monitoring requires macOS with MPS support");
    }
    
    Ok(())
}

/// Demonstrate cross-platform compatibility handling
fn demo_cross_platform_compatibility() {
    println!("\n🌐 Cross-Platform Compatibility");
    println!("-------------------------------");
    
    let platform_info = DeviceCompatibilityChecker::get_platform_info();
    
    println!("📋 Platform Information:");
    println!("  - OS: {}", platform_info.os);
    println!("  - Architecture: {}", platform_info.arch);
    println!("  - Metal Feature: {}", platform_info.metal_feature_enabled);
    println!("  - MPS Feature: {}", platform_info.mps_feature_enabled);
    println!("  - ANE Feature: {}", platform_info.ane_feature_enabled);
    
    // Platform-specific recommendations
    match platform_info.os.as_str() {
        "macos" => {
            if platform_info.mps_available {
                println!("✅ Optimal platform for BitNet MPS acceleration");
            } else {
                println!("⚠️  macOS detected but MPS unavailable - check Metal support");
            }
        }
        "linux" => {
            println!("ℹ️  Linux platform - CPU fallback recommended");
            println!("    Consider CUDA backend for GPU acceleration");
        }
        "windows" => {
            println!("ℹ️  Windows platform - CPU fallback recommended");
            println!("    Consider DirectML or CUDA backends for GPU acceleration");
        }
        _ => {
            println!("ℹ️  Unknown platform - CPU fallback recommended");
        }
    }
    
    println!("💡 Fallback Strategy Recommendations:");
    if platform_info.mps_available {
        println!("  1. Use MPS for optimal performance");
        println!("  2. Fall back to Metal compute if MPS fails");
        println!("  3. Use CPU as final fallback");
    } else if platform_info.metal_feature_enabled {
        println!("  1. Use basic Metal compute acceleration");
        println!("  2. Fall back to CPU for unsupported operations");
    } else {
        println!("  1. Use optimized CPU implementations");
        println!("  2. Consider SIMD acceleration where available");
    }
}

#[cfg(all(target_os = "macos", feature = "mps"))]
fn example_mps_integration() -> Result<()> {
    println!("🚀 Production MPS Integration Example");
    
    // Create manager with production-ready fallback strategy
    let fallback_strategy = FallbackStrategy {
        enable_cpu_fallback: true,
        enable_metal_fallback: true,
        max_retry_attempts: 3,
        retry_delay: Duration::from_millis(50),
        monitor_fallback_performance: true,
    };
    
    let manager = BitNetMPSManager::new_with_fallback_strategy(fallback_strategy)?;
    
    // System information
    let system_info = manager.system_info();
    println!("Running on: {}", system_info.device_name);
    println!("MPS Version: {}", system_info.mps_version);
    
    // Demonstrate matrix operations
    println!("\n🔢 Matrix Operations Demo:");
    let _matrix_result = manager.execute_with_recovery("matrix_operations", || {
        // Placeholder for actual matrix operations
        std::thread::sleep(Duration::from_millis(5));
        Ok("Matrix operations completed")
    })?;
    
    // Demonstrate neural network layers
    println!("\n🧠 Neural Network Operations Demo:");
    let _nn_result = manager.execute_with_recovery("nn_operations", || {
        // Placeholder for actual neural network operations
        std::thread::sleep(Duration::from_millis(10));
        Ok("Neural network operations completed")
    })?;
    
    // Demonstrate computer vision operations
    println!("\n📷 Computer Vision Operations Demo:");
    let _cv_result = manager.execute_with_recovery("cv_operations", || {
        // Placeholder for actual computer vision operations
        std::thread::sleep(Duration::from_millis(8));
        Ok("Computer vision operations completed")
    })?;
    
    // Performance summary
    let recovery_stats = manager.error_recovery().get_recovery_stats()?;
    println!("\n📊 Performance Summary:");
    println!("Total operations: 3");
    println!("Error count: {}", recovery_stats.total_errors);
    println!("Recovery time: {:?}", recovery_stats.total_recovery_time);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_compatibility() {
        let platform_info = DeviceCompatibilityChecker::get_platform_info();
        println!("Platform compatibility test: {}", platform_info);
        
        // Should not panic on any platform
        assert!(!platform_info.os.is_empty());
        assert!(!platform_info.arch.is_empty());
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn test_mps_manager_graceful_failure() {
        // Test that MPS manager creation handles failures gracefully
        match BitNetMPSManager::new() {
            Ok(manager) => {
                println!("MPS manager created successfully");
                let system_info = manager.system_info();
                assert!(!system_info.device_name.is_empty());
            }
            Err(e) => {
                println!("MPS manager creation failed gracefully: {}", e);
                // This is acceptable behavior on unsupported systems
            }
        }
    }

    #[test]
    fn test_cross_platform_demo() {
        // Should work on all platforms without panicking
        demo_cross_platform_compatibility();
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn test_error_recovery_system() {
        if let Ok(manager) = BitNetMPSManager::new() {
            // Test error recovery statistics
            let stats = manager.error_recovery().get_recovery_stats().unwrap();
            assert_eq!(stats.total_errors, 0); // Should start with no errors
            
            // Test capability validation
            let basic_requirements = CapabilityRequirements::default();
            let _validation_result = manager.validate_operation_requirements(&basic_requirements);
            // Should not panic regardless of result
        }
    }
}

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
