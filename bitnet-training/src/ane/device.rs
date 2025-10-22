// bitnet-training/src/ane/device.rs
//! Device abstraction for ANE integration
//!
//! This module provides device abstractions that extend candle_core::Device
//! to support ANE and GPU enumeration for performance monitoring.

use std::fmt;
use serde::{Deserialize, Serialize};
use candle_core::backend::BackendDevice;

/// Extended device enum that includes ANE and explicit GPU variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtendedDevice {
    /// CPU device
    Cpu,
    /// GPU device (Metal/CUDA)
    Gpu,
    /// Apple Neural Engine
    Ane,
}

impl fmt::Display for ExtendedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtendedDevice::Cpu => write!(f, "CPU"),
            ExtendedDevice::Gpu => write!(f, "GPU"),
            ExtendedDevice::Ane => write!(f, "ANE"),
        }
    }
}

impl From<&candle_core::Device> for ExtendedDevice {
    fn from(device: &candle_core::Device) -> Self {
        match device {
            candle_core::Device::Cpu => ExtendedDevice::Cpu,
            candle_core::Device::Metal(_) => ExtendedDevice::Gpu,
            candle_core::Device::Cuda(_) => ExtendedDevice::Gpu,
        }
    }
}

impl ExtendedDevice {
    /// Check if this device supports ANE operations
    pub fn supports_ane(&self) -> bool {
        matches!(self, ExtendedDevice::Ane)
    }

    /// Check if this device supports GPU operations
    pub fn supports_gpu(&self) -> bool {
        matches!(self, ExtendedDevice::Gpu)
    }

    /// Get the canonical candle device for this extended device
    pub fn to_candle_device(&self) -> candle_core::Device {
        match self {
            ExtendedDevice::Cpu => candle_core::Device::Cpu,
            ExtendedDevice::Gpu => {
                // Default to Metal on macOS, CPU elsewhere
                #[cfg(target_os = "macos")]
                {
                    candle_core::Device::Metal(candle_core::MetalDevice::new(0).unwrap_or_else(|_| 
                        candle_core::MetalDevice::new(0).expect("Failed to create Metal device")
                    ))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    candle_core::Device::Cpu
                }
            },
            ExtendedDevice::Ane => {
                // ANE operations fallback to Metal/CPU
                #[cfg(target_os = "macos")]
                {
                    candle_core::Device::Metal(candle_core::MetalDevice::new(0).unwrap_or_else(|_| 
                        candle_core::MetalDevice::new(0).expect("Failed to create Metal device")
                    ))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    candle_core::Device::Cpu
                }
            }
        }
    }

    /// Check if ANE is available on this system
    pub fn is_ane_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            // Check if running on Apple Silicon
            std::env::consts::ARCH == "aarch64"
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Get the best available device for inference
    pub fn best_available() -> Self {
        if Self::is_ane_available() {
            ExtendedDevice::Ane
        } else if Self::is_gpu_available() {
            ExtendedDevice::Gpu
        } else {
            ExtendedDevice::Cpu
        }
    }

    /// Check if GPU is available
    pub fn is_gpu_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            candle_core::MetalDevice::new(0).is_ok()
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Check for CUDA availability
            false // Simplified for now
        }
    }
}