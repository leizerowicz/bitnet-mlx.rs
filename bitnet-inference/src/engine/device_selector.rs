//! Device selection and optimization for inference operations.

use crate::{Result, InferenceError};
use bitnet_core::Device;

/// Device selector for optimal inference performance.
pub struct DeviceSelector;

/// Device selection strategy.
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Automatic selection based on availability and performance
    Auto,
    /// Force specific device type
    Force(Device),
    /// Prefer specific device but fallback to others
    Prefer(Device),
}

/// Device capabilities assessment.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device type
    pub device: Device,
    /// Estimated performance score (higher is better)
    pub performance_score: f32,
    /// Available memory in bytes
    pub available_memory: Option<usize>,
    /// Whether device supports batch processing efficiently
    pub efficient_batching: bool,
    /// Maximum recommended batch size
    pub max_batch_size: usize,
}

impl DeviceSelector {
    /// Select the optimal device based on strategy.
    pub fn select_device(strategy: SelectionStrategy) -> Result<Device> {
        match strategy {
            SelectionStrategy::Auto => Self::select_optimal_device(),
            SelectionStrategy::Force(device) => {
                if Self::is_device_available(&device)? {
                    Ok(device)
                } else {
                    Err(InferenceError::device(format!(
                        "Forced device {:?} is not available",
                        device
                    )))
                }
            }
            SelectionStrategy::Prefer(preferred) => {
                if Self::is_device_available(&preferred)? {
                    Ok(preferred)
                } else {
                    tracing::warn!("Preferred device {:?} not available, falling back to auto-selection", preferred);
                    Self::select_optimal_device()
                }
            }
        }
    }

    /// Automatically select the optimal device.
    pub fn select_optimal_device() -> Result<Device> {
        let capabilities = Self::assess_available_devices()?;
        
        if capabilities.is_empty() {
            return Err(InferenceError::device("No devices available for inference"));
        }

        // Sort by performance score (descending)
        let mut sorted_capabilities = capabilities;
        sorted_capabilities.sort_by(|a, b| {
            b.performance_score.partial_cmp(&a.performance_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_capabilities[0].device.clone())
    }

    /// Assess capabilities of all available devices.
    pub fn assess_available_devices() -> Result<Vec<DeviceCapabilities>> {
        let mut capabilities = Vec::new();

        // Always check CPU availability
        capabilities.push(Self::assess_cpu_capabilities());

        // Check Metal availability (macOS only)
        #[cfg(feature = "metal")]
        {
            if Self::is_metal_available() {
                capabilities.push(Self::assess_metal_capabilities()?);
            }
        }

        // Check MLX availability (Apple Silicon only)
        #[cfg(feature = "mlx")]
        {
            if Self::is_mlx_available() {
                capabilities.push(Self::assess_mlx_capabilities()?);
            }
        }

        // Check CUDA availability - skip for now as CUDA is not implemented
        // capabilities.extend(Self::assess_cuda_capabilities()?);

        Ok(capabilities)
    }

    /// Check if a specific device is available.
    pub fn is_device_available(device: &Device) -> Result<bool> {
        match device {
            Device::Cpu => Ok(true), // CPU is always available
            Device::Cuda(_) => Ok(Self::is_cuda_available()),
            Device::Metal(_) => Ok(Self::is_metal_available()),
            #[cfg(feature = "mlx")]
            _ if format!("{:?}", device).contains("Mlx") => Ok(Self::is_mlx_available()),
        }
    }

    /// Assess CPU capabilities.
    fn assess_cpu_capabilities() -> DeviceCapabilities {
        let thread_count = rayon::current_num_threads();
        let available_memory = Self::get_available_system_memory();
        
        DeviceCapabilities {
            device: Device::Cpu,
            performance_score: thread_count as f32 * 1.0, // Base score
            available_memory,
            efficient_batching: thread_count >= 4,
            max_batch_size: if thread_count >= 8 { 128 } else { 64 },
        }
    }

    /// Assess Metal capabilities (macOS).
    #[cfg(feature = "metal")]
    fn assess_metal_capabilities() -> Result<DeviceCapabilities> {
        // Metal typically provides good GPU acceleration
        Ok(DeviceCapabilities {
            device: Device::Cpu, // Use CPU as placeholder since Metal device creation is complex
            performance_score: 50.0, // High score for GPU acceleration
            available_memory: Self::get_available_system_memory(),
            efficient_batching: true,
            max_batch_size: 512,
        })
    }

    /// Assess MLX capabilities (Apple Silicon).
    #[cfg(feature = "mlx")]
    fn assess_mlx_capabilities() -> Result<DeviceCapabilities> {
        // MLX provides the best performance on Apple Silicon
        Ok(DeviceCapabilities {
            device: Device::Cpu, // MLX uses CPU device type in candle
            performance_score: 100.0, // Highest score for MLX
            available_memory: Self::get_available_system_memory(),
            efficient_batching: true,
            max_batch_size: 1024,
        })
    }

    /// Check if Metal is available.
    #[cfg(feature = "metal")]
    pub fn is_metal_available() -> bool {
        bitnet_metal::is_metal_supported()
    }

    #[cfg(not(feature = "metal"))]
    pub fn is_metal_available() -> bool {
        false
    }

    /// Check if MLX is available.
    #[cfg(feature = "mlx")]
    pub fn is_mlx_available() -> bool {
        // Check if we're on Apple Silicon and MLX is functional
        cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
    }

    #[cfg(not(feature = "mlx"))]
    pub fn is_mlx_available() -> bool {
        false
    }

    /// Check if CUDA is available.
    fn is_cuda_available() -> bool {
        // CUDA is not supported in current implementation
        false
    }

    /// Get available system memory.
    fn get_available_system_memory() -> Option<usize> {
        // This is a simplified implementation
        // In production, you'd want to use system APIs to get actual memory info
        #[cfg(target_os = "macos")]
        {
            // Rough estimate for typical Mac systems
            Some(16 * 1024 * 1024 * 1024) // 16GB
        }
        #[cfg(target_os = "linux")]
        {
            // Could read /proc/meminfo on Linux
            Some(16 * 1024 * 1024 * 1024) // 16GB estimate
        }
        #[cfg(target_os = "windows")]
        {
            // Could use Windows APIs
            Some(16 * 1024 * 1024 * 1024) // 16GB estimate
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            None
        }
    }

    /// Get recommended device for specific model requirements.
    pub fn recommend_device_for_model(
        _parameter_count: usize,
        max_batch_size: usize,
        memory_requirement: usize,
    ) -> Result<Device> {
        let capabilities = Self::assess_available_devices()?;
        
        // Filter devices that can handle the requirements
        let suitable_devices: Vec<_> = capabilities
            .into_iter()
            .filter(|cap| {
                // Check memory requirement
                if let Some(available) = cap.available_memory {
                    if available < memory_requirement {
                        return false;
                    }
                }
                
                // Check batch size capability
                if cap.max_batch_size < max_batch_size {
                    return false;
                }
                
                true
            })
            .collect();

        if suitable_devices.is_empty() {
            return Err(InferenceError::device(
                "No devices meet the model requirements"
            ));
        }

        // Select the best suitable device
        let best_device = suitable_devices
            .iter()
            .max_by(|a, b| {
                a.performance_score.partial_cmp(&b.performance_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        Ok(best_device.device.clone())
    }
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        SelectionStrategy::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection_auto() {
        let result = DeviceSelector::select_device(SelectionStrategy::Auto);
        assert!(result.is_ok());
        
        // Should at least return CPU
        let device = result.unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_capabilities_assessment() {
        let capabilities = DeviceSelector::assess_available_devices().unwrap();
        assert!(!capabilities.is_empty());
        
        // Should at least have CPU
        assert!(capabilities.iter().any(|cap| matches!(cap.device, Device::Cpu)));
        
        // All capabilities should have positive performance scores
        for cap in &capabilities {
            assert!(cap.performance_score > 0.0);
        }
    }

    #[test]
    fn test_cpu_always_available() {
        let available = DeviceSelector::is_device_available(&Device::Cpu).unwrap();
        assert!(available);
    }

    #[test]
    fn test_model_device_recommendation() {
        let device = DeviceSelector::recommend_device_for_model(
            1_000_000,    // 1M parameters
            32,           // batch size
            512 * 1024 * 1024, // 512MB memory
        ).unwrap();
        
        // Should successfully recommend a device
        assert!(matches!(device, Device::Cpu));
    }
}
