//! MLX device management for BitNet
//! 
//! This module provides device abstraction for MLX operations,
//! integrating with BitNet's memory management system.

#[cfg(feature = "mlx")]
use mlx_rs::{Device as MlxDevice, DeviceType as MlxDeviceType};

// Simplified device management for MLX
use anyhow::Result;

/// MLX device information
#[cfg(feature = "mlx")]
#[derive(Debug, Clone)]
pub struct MlxDeviceInfo {
    pub device_type: MlxDeviceType,
    pub device_id: i32,
    pub memory_limit: Option<usize>,
    pub supports_unified_memory: bool,
}

#[cfg(feature = "mlx")]
impl MlxDeviceInfo {
    /// Create device info for CPU
    pub fn cpu() -> Self {
        Self {
            device_type: MlxDeviceType::Cpu,
            device_id: 0,
            memory_limit: None,
            supports_unified_memory: false,
        }
    }

    /// Create device info for GPU
    pub fn gpu(device_id: i32) -> Self {
        Self {
            device_type: MlxDeviceType::Gpu,
            device_id,
            memory_limit: None,
            supports_unified_memory: true, // Apple Silicon unified memory
        }
    }

    /// Get the corresponding BitNet device type
    pub fn to_bitnet_device_type(&self) -> String {
        match self.device_type {
            MlxDeviceType::Cpu => "cpu".to_string(),
            MlxDeviceType::Gpu => "gpu".to_string(),
        }
    }

    /// Create MLX device from this info
    pub fn create_device(&self) -> Result<MlxDevice> {
        match self.device_type {
            MlxDeviceType::Cpu => Ok(MlxDevice::cpu()),
            MlxDeviceType::Gpu => Ok(MlxDevice::gpu()),
        }
    }
}

/// MLX device manager for BitNet integration
#[cfg(feature = "mlx")]
#[derive(Debug)]
pub struct MlxDeviceManager {
    available_devices: Vec<MlxDeviceInfo>,
    default_device: Option<MlxDeviceInfo>,
}

#[cfg(feature = "mlx")]
impl MlxDeviceManager {
    /// Create a new MLX device manager
    pub fn new() -> Result<Self> {
        let mut manager = Self {
            available_devices: Vec::new(),
            default_device: None,
        };
        
        manager.discover_devices()?;
        Ok(manager)
    }

    /// Discover available MLX devices
    fn discover_devices(&mut self) -> Result<()> {
        // Always add CPU device
        let cpu_device = MlxDeviceInfo::cpu();
        self.available_devices.push(cpu_device.clone());

        // Try to add GPU device on Apple Silicon
        if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
            // Try to create GPU device - on Apple Silicon this should work
            let gpu_device = MlxDeviceInfo::gpu(0);
            self.available_devices.push(gpu_device.clone());
            self.default_device = Some(gpu_device); // Prefer GPU
        } else {
            self.default_device = Some(cpu_device);
        }

        Ok(())
    }

    /// Get all available devices
    pub fn available_devices(&self) -> &[MlxDeviceInfo] {
        &self.available_devices
    }

    /// Get the default device
    pub fn default_device(&self) -> Option<&MlxDeviceInfo> {
        self.default_device.as_ref()
    }

    /// Get device by type
    pub fn get_device_by_type(&self, device_type: &str) -> Option<&MlxDeviceInfo> {
        self.available_devices.iter().find(|device| {
            device.to_bitnet_device_type() == device_type
        })
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.available_devices.iter().any(|device| {
            matches!(device.device_type, MlxDeviceType::Gpu)
        })
    }

    /// Get device capabilities for MLX device
    pub fn get_capabilities(&self, device_info: &MlxDeviceInfo) -> String {
        match device_info.device_type {
            MlxDeviceType::Cpu => "CPU with MLX support".to_string(),
            MlxDeviceType::Gpu => "GPU with MLX and Metal support".to_string(),
        }
    }
}

#[cfg(feature = "mlx")]
impl Default for MlxDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            available_devices: vec![MlxDeviceInfo::cpu()],
            default_device: Some(MlxDeviceInfo::cpu()),
        })
    }
}

/// Get the global MLX device manager
#[cfg(feature = "mlx")]
pub fn get_mlx_device_manager() -> &'static MlxDeviceManager {
    use std::sync::OnceLock;
    static MANAGER: OnceLock<MlxDeviceManager> = OnceLock::new();
    MANAGER.get_or_init(|| MlxDeviceManager::default())
}

/// Auto-select the best MLX device
#[cfg(feature = "mlx")]
pub fn auto_select_mlx_device() -> Result<MlxDeviceInfo> {
    let manager = get_mlx_device_manager();
    manager.default_device()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No MLX devices available"))
}

// Stub implementations when MLX is not available
#[cfg(not(feature = "mlx"))]
pub fn auto_select_mlx_device() -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}

#[cfg(not(feature = "mlx"))]
pub fn get_mlx_device_manager() -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}