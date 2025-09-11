//! # MPS Framework Integration
//!
//! Core Metal Performance Shaders framework integration for BitNet operations.
//! This module provides the foundational MPS integration layer.

use anyhow::Result;
use std::sync::Arc;

#[cfg(all(target_os = "macos", feature = "mps"))]
use metal::Device;

/// Core MPS framework integration
#[derive(Debug)]
pub struct MPSFramework {
    #[cfg(all(target_os = "macos", feature = "mps"))]
    device: Arc<Device>,
    version: String,
    capabilities: MPSCapabilities,
}

impl MPSFramework {
    /// Create new MPS framework instance
    pub fn new(#[cfg(all(target_os = "macos", feature = "mps"))] device: Arc<Device>) -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            let version = Self::detect_mps_version();
            let capabilities = MPSCapabilities::detect(&device)?;
            
            Ok(Self {
                device,
                version,
                capabilities,
            })
        }
        
        #[cfg(not(all(target_os = "macos", feature = "mps")))]
        {
            Err(anyhow::anyhow!("MPS framework requires macOS and 'mps' feature"))
        }
    }
    
    /// Get MPS version string
    pub fn version(&self) -> String {
        self.version.clone()
    }
    
    /// Get MPS capabilities
    pub fn capabilities(&self) -> &MPSCapabilities {
        &self.capabilities
    }
    
    /// Detect MPS version from system
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_mps_version() -> String {
        // Use objc to query MPS version
        use objc::runtime::{Class, Object};
        use objc::{msg_send, sel, sel_impl};
        
        unsafe {
            if let Some(mps_class) = Class::get("MPSKernel") {
                let version: *mut Object = msg_send![mps_class, performSelector: sel!(description)];
                if !version.is_null() {
                    return "MPS 3.0+".to_string(); // Simplified version detection
                }
            }
        }
        
        "MPS Unknown".to_string()
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    fn detect_mps_version() -> String {
        "MPS Unavailable".to_string()
    }
}

/// MPS capabilities detection
#[derive(Debug, Clone)]
pub struct MPSCapabilities {
    pub supports_neural_network: bool,
    pub supports_matrix_multiplication: bool,
    pub supports_convolution: bool,
    pub supports_image_processing: bool,
    pub supports_graph_api: bool,
    pub max_texture_size: usize,
    pub max_buffer_size: usize,
    pub unified_memory: bool,
}

impl MPSCapabilities {
    /// Detect MPS capabilities for the given device
    #[cfg(all(target_os = "macos", feature = "mps"))]
    pub fn detect(device: &Device) -> Result<Self> {
        // Query device capabilities using modern Metal GPU family checks
        // Use a simplified capability detection approach to avoid deprecated APIs
        let supports_neural_network = true; // Assume modern Metal devices support neural networks
        let supports_matrix_multiplication = true; // Assume modern Metal devices support matrix ops
        let supports_convolution = supports_neural_network;
        let supports_image_processing = true; // Basic support
        let supports_graph_api = Self::detect_graph_api_support();
        
        // Use conservative texture size for compatibility
        let max_texture_size = 8192;
        
        let max_buffer_size = if device.has_unified_memory() {
            // Unified memory systems can use larger buffers
            1024 * 1024 * 1024 // 1GB
        } else {
            256 * 1024 * 1024 // 256MB
        };
        
        let unified_memory = device.has_unified_memory();
        
        Ok(Self {
            supports_neural_network,
            supports_matrix_multiplication,
            supports_convolution,
            supports_image_processing,
            supports_graph_api,
            max_texture_size,
            max_buffer_size,
            unified_memory,
        })
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    pub fn detect(_device: &()) -> Result<Self> {
        Ok(Self::default())
    }
    
    /// Detect if MPS Graph API is available
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_graph_api_support() -> bool {
        use objc::runtime::Class;
        
        // Check if MPSGraph class exists (available in macOS 12.0+)
        Class::get("MPSGraph").is_some()
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    fn detect_graph_api_support() -> bool {
        false
    }
}

impl Default for MPSCapabilities {
    fn default() -> Self {
        Self {
            supports_neural_network: false,
            supports_matrix_multiplication: false,
            supports_convolution: false,
            supports_image_processing: false,
            supports_graph_api: false,
            max_texture_size: 0,
            max_buffer_size: 0,
            unified_memory: false,
        }
    }
}

/// MPS performance optimization settings
#[derive(Debug, Clone)]
pub struct MPSOptimizationSettings {
    pub enable_cache: bool,
    pub cache_size_mb: usize,
    pub enable_fusion: bool,
    pub batch_size_threshold: usize,
    pub memory_pressure_threshold: f32,
}

impl Default for MPSOptimizationSettings {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_size_mb: 128,
            enable_fusion: true,
            batch_size_threshold: 32,
            memory_pressure_threshold: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn test_mps_framework_creation() {
        use metal::Device;
        
        if let Some(device) = Device::system_default() {
            let device = Arc::new(device);
            let framework = MPSFramework::new(device);
            assert!(framework.is_ok());
            
            if let Ok(framework) = framework {
                println!("MPS Version: {}", framework.version());
                println!("MPS Capabilities: {:?}", framework.capabilities());
            }
        }
    }
    
    #[test]
    fn test_mps_capabilities_default() {
        let capabilities = MPSCapabilities::default();
        assert!(!capabilities.supports_neural_network);
        assert!(!capabilities.supports_matrix_multiplication);
    }
    
    #[test]
    fn test_optimization_settings() {
        let settings = MPSOptimizationSettings::default();
        assert!(settings.enable_cache);
        assert_eq!(settings.cache_size_mb, 128);
    }
}
