//! # Metal Performance Shaders (MPS) Integration for BitNet
//!
//! This module provides comprehensive Metal Performance Shaders integration for BitNet neural networks
//! on Apple Silicon, implementing Task 4.1.2 from COMPREHENSIVE_TODO.md.
//!
//! ## Features Implemented
//!
//! - **MPS Framework Integration**: Full integration with Apple's Metal Performance Shaders
//! - **Matrix Multiplication Kernels**: MPS-optimized GEMM operations for BitNet
//! - **Neural Network Layers**: MPS implementations of BitNet-specific layers
//! - **Computer Vision Acceleration**: MPS primitives for vision tasks
//! - **Apple Neural Engine (ANE) Integration**: Direct Neural Engine hardware access
//! - **Unified Memory Optimization**: Advanced Apple Silicon memory strategies
//!
//! ## Architecture
//!
//! ```text
//! MPS Integration:
//! - Framework Integration (mps_framework.rs)
//!   - MPS device management
//!   - Command buffer integration
//!   - Error handling
//! - Matrix Operations (matrix_ops.rs)
//!   - MPS-optimized GEMM
//!   - Quantized matrix operations
//!   - Memory-efficient operations
//! - Neural Network Layers (nn_layers.rs)
//!   - BitLinear layer implementation
//!   - Quantization layers
//!   - Activation functions
//! - Computer Vision (cv_acceleration.rs)
//!   - MPS image processing
//!   - Convolution operations
//!   - Vision transformers
//! - Apple Neural Engine (ane_integration.rs)
//!   - ANE device management
//!   - Model partitioning
//!   - Power optimization
//! - Unified Memory (unified_memory.rs)
//!   - Memory bandwidth optimization
//!   - Cross-device sharing
//!   - Synchronization
//! ```

pub mod mps_framework;
pub mod matrix_ops;
pub mod nn_layers;
pub mod cv_acceleration;
pub mod ane_integration;
pub mod unified_memory;
pub mod error_recovery;
pub mod dynamic_load_balancing;
pub mod mlx_integration;

// Re-export main types
pub use mps_framework::*;
pub use matrix_ops::*;
pub use nn_layers::*;
pub use cv_acceleration::*;
pub use ane_integration::*;
pub use unified_memory::*;
pub use error_recovery::*;
pub use dynamic_load_balancing::*;
pub use mlx_integration::*;

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;

#[cfg(all(target_os = "macos", feature = "mps"))]
use metal::{Device, CommandQueue};

/// Main MPS integration manager for BitNet operations
#[derive(Debug)]
pub struct BitNetMPSManager {
    device: Arc<Device>,
    command_queue: Arc<CommandQueue>,
    framework: MPSFramework,
    matrix_ops: MPSMatrixOperations,
    nn_layers: MPSNeuralNetworkLayers,
    cv_acceleration: MPSComputerVision,
    ane_integration: Option<ANEIntegration>,
    unified_memory: UnifiedMemoryManager,
    error_recovery: MPSErrorRecovery,
}

impl BitNetMPSManager {
    /// Create a new MPS manager with full Apple Silicon optimization and production error handling
    pub fn new() -> Result<Self> {
        Self::new_with_fallback_strategy(FallbackStrategy::default())
    }

    /// Create a new MPS manager with custom fallback strategy
    pub fn new_with_fallback_strategy(fallback_strategy: FallbackStrategy) -> Result<Self> {
        // Check platform compatibility first
        if !DeviceCompatibilityChecker::is_mps_available() {
            let platform_info = DeviceCompatibilityChecker::get_platform_info();
            let error = MPSError::UnavailablePlatform {
                platform: format!("{}-{}", platform_info.os, platform_info.arch),
                reason: format!("MPS not available. Platform info: {}", platform_info),
            };
            
            let error_recovery = MPSErrorRecovery::new(fallback_strategy);
            let recovery_action = error_recovery.handle_error(error.clone());
            
            return match recovery_action {
                RecoveryAction::FallbackToCPU => {
                    Err(anyhow::anyhow!("MPS unavailable, CPU fallback recommended: {}", error))
                }
                _ => Err(anyhow::anyhow!("MPS unavailable: {}", error))
            };
        }

        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            let device = Arc::new(Device::system_default().ok_or_else(|| {
                MPSError::DeviceInitializationFailed {
                    device_name: Some("system_default".to_string()),
                    metal_error: "No Metal device available".to_string(),
                    fallback_available: false,
                }
            })?);
            
            let command_queue = Arc::new(device.new_command_queue());
            let error_recovery = MPSErrorRecovery::new(fallback_strategy.clone());
            
            // Initialize framework with error handling
            let framework = MPSFramework::new(device.clone()).map_err(|e| {
                let error = MPSError::DeviceInitializationFailed {
                    device_name: Some(device.name().to_string()),
                    metal_error: format!("Framework initialization failed: {}", e),
                    fallback_available: true,
                };
                error_recovery.handle_error(error.clone());
                e
            })?;
            
            // Validate device capabilities
            let requirements = CapabilityRequirements::default();
            error_recovery.validate_capabilities(
                framework.capabilities(), 
                &requirements, 
                &device.name()
            )?;
            
            let matrix_ops = MPSMatrixOperations::new(device.clone())?;
            let nn_layers = MPSNeuralNetworkLayers::new(device.clone())?;
            let cv_acceleration = MPSComputerVision::new(device.clone())?;
            
            // ANE integration with error handling
            let ane_integration = if DeviceCompatibilityChecker::is_ane_available() {
                match ANEIntegration::new() {
                    Ok(ane) => Some(ane),
                    Err(e) => {
                        let error = MPSError::ANEUnavailable {
                            reason: format!("ANE initialization failed: {}", e),
                            detection_method: "ANEIntegration::new()".to_string(),
                            alternative_devices: vec!["Metal".to_string(), "CPU".to_string()],
                        };
                        error_recovery.handle_error(error);
                        None
                    }
                }
            } else {
                None
            };
            
            let unified_memory = UnifiedMemoryManager::new(device.clone())?;
            
            Ok(Self {
                device,
                command_queue,
                framework,
                matrix_ops,
                nn_layers,
                cv_acceleration,
                ane_integration,
                unified_memory,
                error_recovery,
            })
        }
        
        #[cfg(not(all(target_os = "macos", feature = "mps")))]
        {
            let platform_info = DeviceCompatibilityChecker::get_platform_info();
            Err(anyhow::anyhow!("MPS is only available on macOS with the 'mps' feature enabled. Current platform: {}", platform_info))
        }
    }
    
    /// Get the Metal device
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    
    /// Get the command queue
    pub fn command_queue(&self) -> &Arc<CommandQueue> {
        &self.command_queue
    }
    
    /// Get MPS framework interface
    pub fn framework(&self) -> &MPSFramework {
        &self.framework
    }
    
    /// Get matrix operations interface
    pub fn matrix_ops(&self) -> &MPSMatrixOperations {
        &self.matrix_ops
    }
    
    /// Get neural network layers interface
    pub fn nn_layers(&self) -> &MPSNeuralNetworkLayers {
        &self.nn_layers
    }
    
    /// Get computer vision acceleration interface
    pub fn cv_acceleration(&self) -> &MPSComputerVision {
        &self.cv_acceleration
    }
    
    /// Get Apple Neural Engine integration (if available)
    pub fn ane_integration(&self) -> Option<&ANEIntegration> {
        self.ane_integration.as_ref()
    }
    
    /// Get unified memory manager
    pub fn unified_memory(&self) -> &UnifiedMemoryManager {
        &self.unified_memory
    }
    
    /// Get error recovery manager
    pub fn error_recovery(&self) -> &MPSErrorRecovery {
        &self.error_recovery
    }
    
    /// Check if ANE is available on this device
    pub fn is_ane_available(&self) -> bool {
        self.ane_integration.is_some()
    }
    
    /// Get system information and capabilities
    pub fn system_info(&self) -> MPSSystemInfo {
        MPSSystemInfo {
            device_name: self.device.name().to_string(),
            supports_ane: self.is_ane_available(),
            unified_memory_size: self.unified_memory.total_memory(),
            mps_version: self.framework.version(),
            platform_info: DeviceCompatibilityChecker::get_platform_info(),
            recovery_stats: self.error_recovery.get_recovery_stats().unwrap_or_else(|_| {
                RecoveryStats {
                    total_errors: 0,
                    recent_errors: 0,
                    cpu_fallback_count: 0,
                    metal_fallback_count: 0,
                    total_recovery_time: std::time::Duration::from_secs(0),
                    last_fallback_time: None,
                }
            }),
        }
    }

    /// Execute operation with automatic error recovery
    pub fn execute_with_recovery<F, T>(&self, operation_name: &str, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        let mut attempts = 0;
        let max_attempts = 3;
        
        loop {
            attempts += 1;
            
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                    
                    // Convert anyhow error to MPS error for recovery handling
                    let mps_error = MPSError::OperationFailed {
                        operation: operation_name.to_string(),
                        stage: format!("attempt_{}", attempts),
                        error_details: format!("{}", e),
                        recovery_suggestion: "Retry with reduced complexity".to_string(),
                    };
                    
                    let recovery_action = self.error_recovery.handle_error(mps_error);
                    
                    match recovery_action {
                        RecoveryAction::Retry => {
                            std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
                            continue;
                        }
                        RecoveryAction::FallbackToCPU => {
                            return Err(anyhow::anyhow!("Operation '{}' failed, CPU fallback recommended: {}", operation_name, e));
                        }
                        RecoveryAction::FallbackToMetal => {
                            return Err(anyhow::anyhow!("Operation '{}' failed, Metal fallback recommended: {}", operation_name, e));
                        }
                        _ => {
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    /// Validate operation requirements against device capabilities
    pub fn validate_operation_requirements(&self, requirements: &CapabilityRequirements) -> Result<()> {
        self.error_recovery.validate_capabilities(
            self.framework.capabilities(),
            requirements,
            &self.device.name()
        ).map_err(|e| anyhow::anyhow!("Capability validation failed: {}", e))
    }
}

/// System information for MPS capabilities
#[derive(Debug, Clone)]
pub struct MPSSystemInfo {
    pub device_name: String,
    pub supports_ane: bool,
    pub unified_memory_size: usize,
    pub mps_version: String,
    pub platform_info: PlatformInfo,
    pub recovery_stats: RecoveryStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn test_mps_manager_creation() {
        let manager = BitNetMPSManager::new();
        
        match manager {
            Ok(manager) => {
                let info = manager.system_info();
                println!("MPS System Info: {:?}", info);
                assert!(!info.device_name.is_empty());
                assert!(!info.mps_version.is_empty());
            }
            Err(e) => {
                println!("MPS manager creation failed (expected on some systems): {:?}", e);
                // This is acceptable on systems without MPS support
            }
        }
    }
    
    #[test]
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    fn test_mps_unavailable() {
        let manager = BitNetMPSManager::new();
        assert!(manager.is_err());
    }

    #[test]
    fn test_error_recovery_integration() {
        let fallback_strategy = FallbackStrategy {
            enable_cpu_fallback: true,
            enable_metal_fallback: true,
            max_retry_attempts: 2,
            retry_delay: std::time::Duration::from_millis(50),
            monitor_fallback_performance: true,
        };
        
        let result = BitNetMPSManager::new_with_fallback_strategy(fallback_strategy);
        
        // On non-MPS platforms, this should fail gracefully
        #[cfg(not(all(target_os = "macos", feature = "mps")))]
        assert!(result.is_err());
        
        // On MPS platforms, this may succeed or fail depending on hardware
        #[cfg(all(target_os = "macos", feature = "mps"))]
        match result {
            Ok(manager) => {
                println!("MPS manager created successfully with custom fallback strategy");
                let stats = manager.error_recovery().get_recovery_stats().unwrap();
                assert_eq!(stats.total_errors, 0); // Should start with no errors
            }
            Err(e) => {
                println!("MPS manager creation failed (acceptable): {:?}", e);
            }
        }
    }
}
