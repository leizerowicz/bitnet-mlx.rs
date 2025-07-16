//! Device Management Module
//!
//! This module provides utilities for managing compute devices in BitNet,
//! including CPU and Metal GPU devices. It handles device selection,
//! creation, and provides fallback mechanisms for optimal performance
//! across different hardware configurations.
//!
//! # Features
//!
//! - **CPU Device**: Always available, provides reliable fallback
//! - **Metal Device**: GPU acceleration on macOS (feature-gated)
//! - **Auto Selection**: Intelligent device selection with fallback
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::device::{auto_select_device, get_cpu_device};
//!
//! // Get the best available device
//! let device = auto_select_device();
//! println!("Using device: {:?}", device);
//!
//! // Always get CPU device
//! let cpu_device = get_cpu_device();
//! ```
//!
//! # Feature Gates
//!
//! - `metal`: Enables Metal GPU support (requires macOS)

use candle_core::Device;
use thiserror::Error;

#[cfg(feature = "tracing")]
use tracing::{debug, info};

/// Errors that can occur during device operations
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Metal device is not available on this platform
    #[error("Metal device is not available on this platform")]
    MetalNotAvailable,
    
    /// Metal device creation failed
    #[error("Failed to create Metal device: {0}")]
    MetalCreationFailed(String),
    
    /// Device operation failed
    #[error("Device operation failed: {0}")]
    OperationFailed(String),
}

/// Creates and returns a CPU device
///
/// This function always succeeds as CPU devices are universally available.
/// The CPU device provides a reliable fallback for all operations and is
/// suitable for development, testing, and production workloads where GPU
/// acceleration is not required or available.
///
/// # Returns
///
/// A `Device` configured for CPU computation
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::get_cpu_device;
///
/// let device = get_cpu_device();
/// // Use device for tensor operations
/// ```
pub fn get_cpu_device() -> Device {
    #[cfg(feature = "tracing")]
    debug!("Creating CPU device");
    
    Device::Cpu
}

/// Attempts to create a Metal GPU device
///
/// This function is only available when the `metal` feature is enabled and
/// will attempt to create a Metal device for GPU acceleration on macOS.
/// If Metal is not available or device creation fails, an error is returned.
///
/// # Returns
///
/// - `Ok(Device)` - Successfully created Metal device
/// - `Err(DeviceError)` - Metal not available or creation failed
///
/// # Platform Support
///
/// This function is only functional on macOS systems with Metal support.
/// On other platforms, it will return `DeviceError::MetalNotAvailable`.
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::get_metal_device;
///
/// match get_metal_device() {
///     Ok(device) => println!("Using Metal GPU: {:?}", device),
///     Err(e) => println!("Metal not available: {}", e),
/// }
/// ```
#[cfg(feature = "metal")]
pub fn get_metal_device() -> std::result::Result<Device, DeviceError> {
    #[cfg(feature = "tracing")]
    debug!("Attempting to create Metal device");
    
    match Device::new_metal(0) {
        Ok(device) => {
            #[cfg(feature = "tracing")]
            info!("Successfully created Metal device");
            Ok(device)
        }
        Err(e) => {
            #[cfg(feature = "tracing")]
            warn!("Failed to create Metal device: {}", e);
            Err(DeviceError::MetalCreationFailed(e.to_string()))
        }
    }
}

/// Attempts to create a Metal GPU device (no-op when metal feature is disabled)
///
/// When the `metal` feature is not enabled, this function always returns
/// an error indicating that Metal is not available. This allows code to
/// gracefully handle the absence of Metal support without compilation errors.
///
/// # Returns
///
/// Always returns `Err(DeviceError::MetalNotAvailable)` when metal feature is disabled
#[cfg(not(feature = "metal"))]
pub fn get_metal_device() -> std::result::Result<Device, DeviceError> {
    #[cfg(feature = "tracing")]
    debug!("Metal feature not enabled, Metal device unavailable");
    
    Err(DeviceError::MetalNotAvailable)
}

/// Automatically selects the best available device
///
/// This function implements an intelligent device selection strategy:
/// 1. First attempts to create a Metal GPU device (if available)
/// 2. Falls back to CPU device if Metal is unavailable or fails
///
/// The selection prioritizes performance while ensuring reliability through
/// fallback mechanisms. This is the recommended way to obtain a device for
/// most BitNet operations.
///
/// # Returns
///
/// A `Device` representing the best available compute device
///
/// # Selection Strategy
///
/// - **Metal GPU**: Preferred for performance on supported macOS systems
/// - **CPU**: Reliable fallback, always available
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::auto_select_device;
///
/// let device = auto_select_device();
/// println!("Selected device: {:?}", device);
/// 
/// // Use the device for tensor operations
/// use candle_core::Tensor;
/// let tensor = Tensor::zeros(&[2, 2], candle_core::DType::F32, &device).unwrap();
/// ```
pub fn auto_select_device() -> Device {
    #[cfg(feature = "tracing")]
    debug!("Auto-selecting best available device");
    
    // Try Metal first if available
    match get_metal_device() {
        Ok(device) => {
            #[cfg(feature = "tracing")]
            info!("Auto-selected Metal GPU device");
            device
        }
        Err(_) => {
            #[cfg(feature = "tracing")]
            info!("Metal unavailable, falling back to CPU device");
            get_cpu_device()
        }
    }
}

/// Returns information about device availability
///
/// This function provides runtime information about which devices are
/// available on the current system. Useful for diagnostics, logging,
/// and conditional behavior based on hardware capabilities.
///
/// # Returns
///
/// A tuple containing:
/// - `bool`: Whether CPU device is available (always true)
/// - `bool`: Whether Metal device is available
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::get_device_info;
///
/// let (cpu_available, metal_available) = get_device_info();
/// println!("CPU available: {}, Metal available: {}", cpu_available, metal_available);
/// ```
pub fn get_device_info() -> (bool, bool) {
    let cpu_available = true; // CPU is always available
    let metal_available = get_metal_device().is_ok();
    
    #[cfg(feature = "tracing")]
    debug!("Device availability - CPU: {}, Metal: {}", cpu_available, metal_available);
    
    (cpu_available, metal_available)
}

/// Checks if Metal GPU is available and usable on the current system
///
/// This function performs a runtime check to determine if Metal GPU acceleration
/// is actually available and functional, not just compiled in. It attempts to
/// create a Metal device and returns true only if the creation succeeds.
///
/// # Returns
///
/// - `true` - Metal GPU is available and usable
/// - `false` - Metal GPU is not available, not functional, or feature disabled
///
/// # Platform Support
///
/// This function is only functional on macOS systems with Metal support.
/// On other platforms or when the `metal` feature is disabled, it returns `false`.
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::is_metal_available;
///
/// if is_metal_available() {
///     println!("Metal GPU acceleration is available!");
/// } else {
///     println!("Metal GPU acceleration is not available, using CPU fallback");
/// }
/// ```
#[cfg(feature = "metal")]
pub fn is_metal_available() -> bool {
    #[cfg(feature = "tracing")]
    debug!("Checking Metal GPU availability");
    
    match Device::new_metal(0) {
        Ok(_) => {
            #[cfg(feature = "tracing")]
            debug!("Metal GPU is available and functional");
            true
        }
        Err(_e) => {
            #[cfg(feature = "tracing")]
            debug!("Metal GPU is not available: {}", _e);
            false
        }
    }
}

/// Checks if Metal GPU is available (always returns false when metal feature is disabled)
///
/// When the `metal` feature is not enabled, this function always returns `false`
/// to indicate that Metal GPU acceleration is not available. This allows code to
/// gracefully handle the absence of Metal support without compilation errors.
///
/// # Returns
///
/// Always returns `false` when metal feature is disabled
#[cfg(not(feature = "metal"))]
pub fn is_metal_available() -> bool {
    #[cfg(feature = "tracing")]
    debug!("Metal feature not enabled, Metal GPU unavailable");
    
    false
}

/// Gets the name/identifier of the Metal device if available
///
/// This function attempts to create a Metal device and extract its name or
/// identifier. It returns the actual device name (e.g., "Apple M1 Pro", "Apple M2 Max")
/// if Metal is available and functional, or `None` if Metal is not available.
///
/// # Returns
///
/// - `Some(String)` - Metal device name/identifier
/// - `None` - Metal device not available or feature disabled
///
/// # Platform Support
///
/// This function is only functional on macOS systems with Metal support.
/// On other platforms or when the `metal` feature is disabled, it returns `None`.
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::get_metal_device_name;
///
/// match get_metal_device_name() {
///     Some(name) => println!("Metal device: {}", name),
///     None => println!("No Metal device available"),
/// }
/// ```
#[cfg(feature = "metal")]
pub fn get_metal_device_name() -> Option<String> {
    #[cfg(feature = "tracing")]
    debug!("Attempting to get Metal device name");
    
    match Device::new_metal(0) {
        Ok(device) => {
            // Extract device information from the Metal device
            let device_name = get_metal_device_name_internal(&device);
            
            #[cfg(feature = "tracing")]
            debug!("Metal device name: {:?}", device_name);
            
            device_name
        }
        Err(_e) => {
            #[cfg(feature = "tracing")]
            debug!("Failed to get Metal device name: {}", _e);
            None
        }
    }
}

/// Gets the Metal device name (always returns None when metal feature is disabled)
///
/// When the `metal` feature is not enabled, this function always returns `None`
/// to indicate that no Metal device information is available. This allows code to
/// gracefully handle the absence of Metal support without compilation errors.
///
/// # Returns
///
/// Always returns `None` when metal feature is disabled
#[cfg(not(feature = "metal"))]
pub fn get_metal_device_name() -> Option<String> {
    #[cfg(feature = "tracing")]
    debug!("Metal feature not enabled, no Metal device name available");
    
    None
}

/// Internal helper function to extract Metal device name
///
/// This function uses system APIs to get the actual device name.
/// It attempts to get the device name through system_profiler on macOS.
#[cfg(feature = "metal")]
fn get_metal_device_name_internal(_device: &Device) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        
        // Try to get the chip name from system_profiler
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPHardwareDataType"])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                // Look for "Chip:" line in the output
                for line in output_str.lines() {
                    if line.trim().starts_with("Chip:") {
                        if let Some(chip_name) = line.split(':').nth(1) {
                            let chip_name = chip_name.trim();
                            #[cfg(feature = "tracing")]
                            debug!("Found chip name from system_profiler: {}", chip_name);
                            return Some(chip_name.to_string());
                        }
                    }
                }
            }
        }
        
        // Fallback: try to get processor name
        if let Ok(output) = Command::new("sysctl")
            .args(&["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if let Ok(cpu_name) = String::from_utf8(output.stdout) {
                let cpu_name = cpu_name.trim();
                if !cpu_name.is_empty() {
                    #[cfg(feature = "tracing")]
                    debug!("Found CPU name from sysctl: {}", cpu_name);
                    // For Apple Silicon, the CPU name often indicates the chip
                    if cpu_name.contains("Apple") {
                        return Some(cpu_name.to_string());
                    }
                }
            }
        }
        
        // Final fallback for Apple Silicon
        Some("Apple Silicon GPU".to_string())
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        // On non-macOS platforms, return a generic name
        Some("Metal GPU".to_string())
    }
}

/// Returns a string description of the given device
///
/// Provides human-readable descriptions of devices for logging,
/// debugging, and user interface purposes.
///
/// # Arguments
///
/// * `device` - Reference to the device to describe
///
/// # Returns
///
/// A string describing the device type and characteristics
///
/// # Examples
///
/// ```rust
/// use bitnet_core::device::{get_cpu_device, describe_device};
///
/// let device = get_cpu_device();
/// println!("Device: {}", describe_device(&device));
/// ```
pub fn describe_device(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU (Universal)".to_string(),
        Device::Cuda(_) => "CUDA GPU".to_string(),
        Device::Metal(_) => "Metal GPU (macOS)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cpu_device() {
        let device = get_cpu_device();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_auto_select_device() {
        let device = auto_select_device();
        // Should return either CPU or Metal, never fail
        assert!(matches!(device, Device::Cpu | Device::Metal(_)));
    }

    #[test]
    fn test_get_device_info() {
        let (cpu_available, _metal_available) = get_device_info();
        assert!(cpu_available); // CPU should always be available
    }

    #[test]
    fn test_describe_device() {
        let cpu_device = get_cpu_device();
        let description = describe_device(&cpu_device);
        assert_eq!(description, "CPU (Universal)");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_get_metal_device() {
        // This test will pass or fail depending on Metal availability
        // We just ensure it doesn't panic
        let _result = get_metal_device();
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn test_get_metal_device_disabled() {
        let result = get_metal_device();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DeviceError::MetalNotAvailable));
    }

    #[test]
    fn test_device_error_display() {
        let error = DeviceError::MetalNotAvailable;
        assert_eq!(error.to_string(), "Metal device is not available on this platform");
        
        let error = DeviceError::MetalCreationFailed("test error".to_string());
        assert_eq!(error.to_string(), "Failed to create Metal device: test error");
        
        let error = DeviceError::OperationFailed("test operation".to_string());
        assert_eq!(error.to_string(), "Device operation failed: test operation");
    }

    #[test]
    fn test_is_metal_available() {
        // This test should not panic regardless of Metal availability
        let _result = is_metal_available();
        // We can't assert the specific value since it depends on the system
        // but we can ensure the function executes without errors
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_is_metal_available_with_metal_feature() {
        let result = is_metal_available();
        // On systems with Metal support, this should return true
        // On systems without Metal support, this should return false
        // We just ensure it returns a boolean without panicking
        assert!(result == true || result == false);
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn test_is_metal_available_without_metal_feature() {
        let result = is_metal_available();
        // When metal feature is disabled, should always return false
        assert_eq!(result, false);
    }

    #[test]
    fn test_get_metal_device_name() {
        // This test should not panic regardless of Metal availability
        let _result = get_metal_device_name();
        // We can't assert the specific value since it depends on the system
        // but we can ensure the function executes without errors
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_get_metal_device_name_with_metal_feature() {
        let result = get_metal_device_name();
        // On systems with Metal support, this might return Some(name)
        // On systems without Metal support, this should return None
        // We just ensure it returns an Option without panicking
        match result {
            Some(name) => {
                assert!(!name.is_empty(), "Device name should not be empty");
                println!("Metal device name: {}", name);
            }
            None => {
                println!("No Metal device available");
            }
        }
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn test_get_metal_device_name_without_metal_feature() {
        let result = get_metal_device_name();
        // When metal feature is disabled, should always return None
        assert_eq!(result, None);
    }

    #[test]
    fn test_metal_compatibility_consistency() {
        // Test that is_metal_available and get_metal_device_name are consistent
        let is_available = is_metal_available();
        let device_name = get_metal_device_name();
        
        // If Metal is available, we might get a device name (but not guaranteed)
        // If Metal is not available, we should not get a device name
        if !is_available {
            assert_eq!(device_name, None, "If Metal is not available, device name should be None");
        }
        // Note: We don't assert the reverse (is_available => device_name.is_some())
        // because device name extraction might fail even if Metal is available
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_device_name_internal() {
        // Test the internal helper function if we can create a Metal device
        if let Ok(device) = get_metal_device() {
            let name = get_metal_device_name_internal(&device);
            if let Some(name) = name {
                assert!(!name.is_empty(), "Internal device name should not be empty");
                println!("Internal Metal device name: {}", name);
            }
        }
    }
}