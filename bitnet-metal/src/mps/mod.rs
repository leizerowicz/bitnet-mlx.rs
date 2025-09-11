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
//! ```
//! MPS Integration
//! ├── Framework Integration (mps_framework.rs)
//! │   ├── MPS device management
//! │   ├── Command buffer integration
//! │   └── Error handling
//! ├── Matrix Operations (matrix_ops.rs)
//! │   ├── MPS-optimized GEMM
//! │   ├── Quantized matrix operations
//! │   └── Memory-efficient operations
//! ├── Neural Network Layers (nn_layers.rs)
//! │   ├── BitLinear layer implementation
//! │   ├── Quantization layers
//! │   └── Activation functions
//! ├── Computer Vision (cv_acceleration.rs)
//! │   ├── MPS image processing
//! │   ├── Convolution operations
//! │   └── Vision transformers
//! ├── Apple Neural Engine (ane_integration.rs)
//! │   ├── ANE device management
//! │   ├── Model partitioning
//! │   └── Power optimization
//! └── Unified Memory (unified_memory.rs)
//!     ├── Memory bandwidth optimization
//!     ├── Cross-device sharing
//!     └── Synchronization
//! ```

pub mod mps_framework;
pub mod matrix_ops;
pub mod nn_layers;
pub mod cv_acceleration;
pub mod ane_integration;
pub mod unified_memory;

// Re-export main types
pub use mps_framework::*;
pub use matrix_ops::*;
pub use nn_layers::*;
pub use cv_acceleration::*;
pub use ane_integration::*;
pub use unified_memory::*;

use anyhow::Result;
use std::sync::Arc;

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
}

impl BitNetMPSManager {
    /// Create a new MPS manager with full Apple Silicon optimization
    pub fn new() -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            let device = Arc::new(Device::system_default().ok_or_else(|| {
                anyhow::anyhow!("No Metal device available")
            })?);
            
            let command_queue = Arc::new(device.new_command_queue());
            
            let framework = MPSFramework::new(device.clone())?;
            let matrix_ops = MPSMatrixOperations::new(device.clone())?;
            let nn_layers = MPSNeuralNetworkLayers::new(device.clone())?;
            let cv_acceleration = MPSComputerVision::new(device.clone())?;
            let ane_integration = ANEIntegration::new().ok();
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
            })
        }
        
        #[cfg(not(all(target_os = "macos", feature = "mps")))]
        {
            Err(anyhow::anyhow!("MPS is only available on macOS with the 'mps' feature enabled"))
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
        }
    }
}

/// System information for MPS capabilities
#[derive(Debug, Clone)]
pub struct MPSSystemInfo {
    pub device_name: String,
    pub supports_ane: bool,
    pub unified_memory_size: usize,
    pub mps_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn test_mps_manager_creation() {
        let manager = BitNetMPSManager::new();
        assert!(manager.is_ok(), "Failed to create MPS manager: {:?}", manager.err());
        
        if let Ok(manager) = manager {
            let info = manager.system_info();
            println!("MPS System Info: {:?}", info);
            assert!(!info.device_name.is_empty());
        }
    }
    
    #[test]
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    fn test_mps_unavailable() {
        let manager = BitNetMPSManager::new();
        assert!(manager.is_err());
    }
}
