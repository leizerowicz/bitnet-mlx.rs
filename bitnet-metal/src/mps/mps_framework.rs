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
        // Enhanced capability detection with better hardware introspection
        let device_name = device.name();
        
        // Detect GPU family and capabilities more accurately
        let (supports_neural_network, supports_matrix_multiplication) = Self::detect_compute_capabilities(device);
        let supports_convolution = supports_neural_network;
        let supports_image_processing = Self::detect_image_processing_support(device);
        let supports_graph_api = Self::detect_graph_api_support();
        
        // Enhanced texture size detection based on GPU family
        let max_texture_size = Self::detect_max_texture_size(device);
        
        // Enhanced memory detection with unified memory considerations
        let (max_buffer_size, unified_memory) = Self::detect_memory_capabilities(device);
        
        println!("Detected MPS capabilities for device '{}': NN={}, MM={}, Conv={}, Img={}, Graph={}, MaxTex={}, MaxBuf={}MB, Unified={}",
            device_name,
            supports_neural_network,
            supports_matrix_multiplication, 
            supports_convolution,
            supports_image_processing,
            supports_graph_api,
            max_texture_size,
            max_buffer_size / (1024 * 1024),
            unified_memory
        );
        
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
    
    /// Enhanced compute capability detection
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_compute_capabilities(device: &Device) -> (bool, bool) {
        // Check Metal GPU family for more accurate capability detection
        let device_name = device.name().to_lowercase();
        
        // Apple Silicon devices generally support neural networks and matrix ops
        let is_apple_silicon = device_name.contains("apple") || 
                              device_name.contains("m1") || 
                              device_name.contains("m2") || 
                              device_name.contains("m3") ||
                              device_name.contains("m4");
        
        // Intel and AMD discrete GPUs
        let is_discrete_gpu = device_name.contains("radeon") || 
                             device_name.contains("rx ") ||
                             device_name.contains("vega") ||
                             device_name.contains("nvidia") ||
                             device_name.contains("geforce") ||
                             device_name.contains("rtx") ||
                             device_name.contains("gtx");
        
        // Intel integrated graphics
        let is_intel_integrated = device_name.contains("intel") && 
                                 (device_name.contains("iris") || device_name.contains("uhd") || device_name.contains("hd"));
        
        // Capability matrix based on GPU type
        match (is_apple_silicon, is_discrete_gpu, is_intel_integrated) {
            (true, _, _) => (true, true),      // Apple Silicon: full support
            (_, true, _) => (true, true),      // Discrete GPU: full support
            (_, _, true) => (false, true),     // Intel integrated: limited support
            _ => (false, false),               // Unknown/unsupported
        }
    }
    
    /// Detect image processing support
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_image_processing_support(_device: &Device) -> bool {
        // Most Metal devices support basic image processing
        // Check for specific image processing features via Metal feature sets
        use objc::runtime::Class;
        
        if let Some(_mps_class) = Class::get("MPSImageConversion") {
            // If MPSImageConversion class exists, image processing is supported
            return true;
        }
        
        // Fallback: assume support for modern devices
        true
    }
    
    /// Enhanced texture size detection
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_max_texture_size(device: &Device) -> usize {
        let device_name = device.name().to_lowercase();
        
        // Enhanced texture size limits based on GPU capabilities
        if device_name.contains("m3") || device_name.contains("m4") {
            16384  // Latest Apple Silicon
        } else if device_name.contains("m2") {
            16384  // M2 series
        } else if device_name.contains("m1") {
            16384  // M1 series
        } else if device_name.contains("radeon") || device_name.contains("vega") {
            16384  // Modern AMD GPUs
        } else if device_name.contains("intel") {
            8192   // Intel integrated graphics
        } else {
            8192   // Conservative default
        }
    }
    
    /// Enhanced memory capability detection
    #[cfg(all(target_os = "macos", feature = "mps"))]
    fn detect_memory_capabilities(device: &Device) -> (usize, bool) {
        let unified_memory = device.has_unified_memory();
        let device_name = device.name().to_lowercase();
        
        // Memory limits based on device type and unified memory
        let max_buffer_size = if unified_memory {
            // Apple Silicon unified memory - more generous limits
            if device_name.contains("m3") || device_name.contains("m4") {
                4 * 1024 * 1024 * 1024  // 4GB for latest chips
            } else if device_name.contains("m2") {
                2 * 1024 * 1024 * 1024  // 2GB for M2
            } else if device_name.contains("m1") {
                1 * 1024 * 1024 * 1024  // 1GB for M1
            } else {
                512 * 1024 * 1024       // 512MB conservative
            }
        } else {
            // Discrete GPU or integrated graphics
            if device_name.contains("radeon") && (device_name.contains("pro") || device_name.contains("xt")) {
                1 * 1024 * 1024 * 1024  // 1GB for high-end AMD
            } else if device_name.contains("radeon") {
                512 * 1024 * 1024       // 512MB for mid-range AMD
            } else {
                256 * 1024 * 1024       // 256MB conservative default
            }
        };
        
        (max_buffer_size, unified_memory)
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
    
    /// Check if device meets minimum requirements for operation
    pub fn meets_requirements(&self, requirements: &super::error_recovery::CapabilityRequirements) -> bool {
        if requirements.neural_network_support && !self.supports_neural_network {
            return false;
        }
        if requirements.matrix_multiplication && !self.supports_matrix_multiplication {
            return false;
        }
        if requirements.convolution_support && !self.supports_convolution {
            return false;
        }
        if requirements.graph_api && !self.supports_graph_api {
            return false;
        }
        
        let available_memory_mb = self.max_buffer_size / (1024 * 1024);
        if available_memory_mb < requirements.minimum_memory_mb {
            return false;
        }
        
        true
    }
    
    /// Get suggested fallback strategy based on available capabilities
    pub fn suggest_fallback(&self, requirements: &super::error_recovery::CapabilityRequirements) -> Option<String> {
        // If we have matrix multiplication, suggest Metal fallback
        if self.supports_matrix_multiplication && !requirements.ane_support {
            return Some("metal".to_string());
        }
        
        // If we have basic compute support, suggest limited Metal
        if self.max_buffer_size > 100 * 1024 * 1024 { // 100MB minimum
            return Some("metal_limited".to_string());
        }
        
        // Otherwise suggest CPU fallback
        Some("cpu".to_string())
    }
    
    /// Get compatibility score (0.0 to 1.0) for requirements
    pub fn compatibility_score(&self, requirements: &super::error_recovery::CapabilityRequirements) -> f32 {
        let mut score = 0.0;
        let mut total_checks = 0.0;
        
        // Neural network support
        total_checks += 1.0;
        if !requirements.neural_network_support || self.supports_neural_network {
            score += 1.0;
        }
        
        // Matrix multiplication
        total_checks += 1.0;
        if !requirements.matrix_multiplication || self.supports_matrix_multiplication {
            score += 1.0;
        }
        
        // Convolution support
        total_checks += 1.0;
        if !requirements.convolution_support || self.supports_convolution {
            score += 1.0;
        }
        
        // Graph API
        total_checks += 1.0;
        if !requirements.graph_api || self.supports_graph_api {
            score += 1.0;
        }
        
        // Memory requirements
        total_checks += 1.0;
        let available_memory_mb = self.max_buffer_size / (1024 * 1024);
        if available_memory_mb >= requirements.minimum_memory_mb {
            score += 1.0;
        } else {
            // Partial score based on available memory
            score += (available_memory_mb as f32) / (requirements.minimum_memory_mb as f32).max(1.0);
        }
        
        score / total_checks
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
