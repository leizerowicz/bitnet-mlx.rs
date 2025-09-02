//! MLX device management for BitNet
//!
//! This module provides device abstraction for MLX operations,
//! integrating with BitNet's memory management system.

#[cfg(feature = "mlx")]
use mlx_rs::{Device as MlxDevice, DeviceType as MlxDeviceType};

// Simplified device management for MLX
use crate::memory::MemoryMetrics;
use anyhow::Result;

/// BitNet MLX Device wrapper
#[derive(Debug)]
pub struct BitNetMlxDevice {
    #[cfg(feature = "mlx")]
    inner: MlxDevice,
    device_info: Option<MlxDeviceInfo>,
    initialized: bool,
}

impl Clone for BitNetMlxDevice {
    fn clone(&self) -> Self {
        #[cfg(feature = "mlx")]
        {
            // Create a new device of the same type
            let inner = if let Some(info) = &self.device_info {
                match info.device_type {
                    MlxDeviceType::Cpu => MlxDevice::cpu(),
                    MlxDeviceType::Gpu => MlxDevice::gpu(),
                }
            } else {
                MlxDevice::cpu()
            };

            Self {
                inner,
                device_info: self.device_info.clone(),
                initialized: self.initialized,
            }
        }
        #[cfg(not(feature = "mlx"))]
        {
            Self {
                device_info: self.device_info.clone(),
                initialized: self.initialized,
            }
        }
    }
}

impl BitNetMlxDevice {
    /// Create a new BitNet MLX device
    #[cfg(feature = "mlx")]
    pub fn new(device: MlxDevice, device_info: MlxDeviceInfo) -> Self {
        Self {
            inner: device,
            device_info: Some(device_info),
            initialized: false,
        }
    }

    /// Create a new BitNet MLX device without MLX feature
    #[cfg(not(feature = "mlx"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            device_info: None,
            initialized: false,
        })
    }

    /// Create default BitNet MLX device
    pub fn default() -> Result<Self> {
        #[cfg(feature = "mlx")]
        {
            let device_info = auto_select_mlx_device()?;
            let mlx_device = match device_info.device_type {
                MlxDeviceType::Cpu => MlxDevice::cpu(),
                MlxDeviceType::Gpu => MlxDevice::gpu(),
            };
            Ok(Self::new(mlx_device, device_info))
        }
        #[cfg(not(feature = "mlx"))]
        {
            Self::new()
        }
    }

    /// Create a CPU-based MLX device
    pub fn cpu() -> Result<Self> {
        #[cfg(feature = "mlx")]
        {
            let mlx_device = MlxDevice::cpu();
            let device_info = MlxDeviceInfo {
                device_type: MlxDeviceType::Cpu,
                device_id: 0,
                memory_limit: Some(16 * 1024 * 1024 * 1024), // 16GB default CPU memory
                supports_unified_memory: false,
            };

            Ok(Self {
                #[cfg(feature = "mlx")]
                inner: mlx_device,
                device_info: Some(device_info),
                initialized: true,
            })
        }

        #[cfg(not(feature = "mlx"))]
        {
            Err(anyhow::anyhow!("MLX feature not enabled"))
        }
    }

    /// Create GPU device
    pub fn gpu() -> Result<Self> {
        #[cfg(feature = "mlx")]
        {
            let mlx_device = MlxDevice::gpu();
            let device_info = MlxDeviceInfo {
                device_type: MlxDeviceType::Gpu,
                device_id: 0,
                memory_limit: Some(64 * 1024 * 1024 * 1024), // 64GB default unified memory
                supports_unified_memory: true,
            };

            Ok(Self {
                #[cfg(feature = "mlx")]
                inner: mlx_device,
                device_info: Some(device_info),
                initialized: true,
            })
        }

        #[cfg(not(feature = "mlx"))]
        {
            Err(anyhow::anyhow!("MLX feature not enabled"))
        }
    }

    /// Initialize the device
    pub fn initialize(&mut self) -> Result<()> {
        // MLX devices don't require explicit initialization
        self.initialized = true;
        Ok(())
    }

    /// Check if the device is available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "mlx")]
        {
            self.device_info.is_some()
        }
        #[cfg(not(feature = "mlx"))]
        {
            false
        }
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> crate::memory::MemoryResult<crate::memory::MemoryMetrics> {
        // For now, return default metrics
        // In a real implementation, this would query MLX device memory
        Ok(crate::memory::MemoryMetrics::default())
    }

    /// Cleanup device resources
    pub fn cleanup(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    /// Get the device type as string
    pub fn device_type(&self) -> &str {
        #[cfg(feature = "mlx")]
        {
            if let Some(ref info) = self.device_info {
                match info.device_type {
                    MlxDeviceType::Cpu => "cpu",
                    MlxDeviceType::Gpu => "gpu",
                }
            } else {
                "unknown"
            }
        }
        #[cfg(not(feature = "mlx"))]
        {
            "mock"
        }
    }

    /// Check if this device supports unified memory
    pub fn supports_unified_memory(&self) -> bool {
        #[cfg(feature = "mlx")]
        {
            self.device_info
                .as_ref()
                .map(|info| info.supports_unified_memory)
                .unwrap_or(false)
        }
        #[cfg(not(feature = "mlx"))]
        {
            false
        }
    }

    /// Get the underlying MLX device
    #[cfg(feature = "mlx")]
    pub fn inner(&self) -> &MlxDevice {
        &self.inner
    }

    /// Get device info
    pub fn device_info(&self) -> Option<&MlxDeviceInfo> {
        self.device_info.as_ref()
    }
}

impl Default for BitNetMlxDevice {
    fn default() -> Self {
        #[cfg(feature = "mlx")]
        {
            // Try to create default device, fallback to CPU if it fails
            match Self::cpu() {
                Ok(device) => device,
                Err(_) => Self {
                    inner: MlxDevice::cpu(),
                    device_info: Some(MlxDeviceInfo::cpu()),
                    initialized: false,
                },
            }
        }
        #[cfg(not(feature = "mlx"))]
        {
            Self {
                device_info: None,
                initialized: false,
            }
        }
    }
}

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
        self.available_devices
            .iter()
            .find(|device| device.to_bitnet_device_type() == device_type)
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.available_devices
            .iter()
            .any(|device| matches!(device.device_type, MlxDeviceType::Gpu))
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
        Self::new().unwrap_or_else(|| Self {
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
    manager
        .default_device()
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
