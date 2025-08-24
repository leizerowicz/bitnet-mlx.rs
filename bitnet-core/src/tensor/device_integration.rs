//! Tensor Device Integration
//!
//! This module provides device-aware tensor operations that integrate seamlessly
//! with the existing device abstraction, supporting CPU, Metal GPU, and MLX
//! acceleration with automatic device selection and migration.

use super::dtype::BitNetDType;
use super::memory_integration::TensorMemoryManager;
use crate::device::{auto_select_device, get_cpu_device, get_metal_device};
use crate::memory::HybridMemoryPool;
use candle_core::Device;
use std::sync::Arc;

#[cfg(feature = "tracing")]
use tracing::{debug, error, info, warn};

/// Device manager for tensor operations
///
/// This manager provides device-aware operations for tensors,
/// including device selection, migration, and optimization.
#[derive(Debug)]
pub struct TensorDeviceManager {
    /// Current device
    device: Device,
    /// Memory manager for this device
    memory_manager: TensorMemoryManager,
    /// Device capabilities
    capabilities: DeviceCapabilities,
    /// Performance characteristics
    performance_profile: DevicePerformanceProfile,
}

/// Device capabilities for tensor operations
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Supported data types on this device
    pub supported_dtypes: Vec<BitNetDType>,
    /// Maximum tensor size in bytes
    pub max_tensor_size: Option<usize>,
    /// Whether the device supports unified memory
    pub unified_memory: bool,
    /// Whether the device supports hardware acceleration
    pub hardware_acceleration: bool,
    /// Memory bandwidth in GB/s (if known)
    pub memory_bandwidth: Option<f32>,
    /// Compute capability description
    pub compute_capability: Option<String>,
    /// Supported tensor operations
    pub supported_operations: Vec<TensorOperation>,
    /// Metal GPU specific features
    pub metal_features: Option<MetalFeatures>,
}

/// Metal-specific GPU features
#[derive(Debug, Clone)]
pub struct MetalFeatures {
    /// Maximum threadgroup memory
    pub max_threadgroup_memory: usize,
    /// Supports unified memory
    pub has_unified_memory: bool,
    /// GPU family
    pub gpu_family: Option<String>,
    /// Compute units
    pub compute_units: Option<u32>,
}

/// Supported tensor operations by device
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorOperation {
    /// Basic arithmetic operations
    Arithmetic,
    /// Matrix multiplication
    MatrixMultiplication,
    /// Convolution operations
    Convolution,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
    /// Element-wise operations
    ElementWise,
    /// Memory operations (copy, clone)
    Memory,
    /// BitNet quantization operations
    BitNetQuantization,
    /// Mixed precision operations
    MixedPrecision,
}

/// Performance profile for device operations
#[derive(Debug, Clone)]
pub struct DevicePerformanceProfile {
    /// Memory transfer latency (ms)
    pub memory_latency_ms: f32,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth_gbps: f32,
    /// Compute throughput estimate (GFLOPS)
    pub compute_throughput_gflops: f32,
    /// Preferred data types for performance
    pub preferred_dtypes: Vec<BitNetDType>,
    /// Optimal tensor sizes for this device
    pub optimal_tensor_sizes: TensorSizeRecommendation,
}

/// Tensor size recommendations for optimal performance
#[derive(Debug, Clone)]
pub struct TensorSizeRecommendation {
    /// Small tensor threshold (elements)
    pub small_threshold: usize,
    /// Large tensor threshold (elements)
    pub large_threshold: usize,
    /// Recommended batch sizes
    pub preferred_batch_sizes: Vec<usize>,
}

/// Device migration result
#[derive(Debug)]
pub struct DeviceMigrationResult {
    /// Source device
    pub from_device: Device,
    /// Target device
    pub to_device: Device,
    /// Migration time in milliseconds
    pub migration_time_ms: f64,
    /// Data size migrated in bytes
    pub data_size_bytes: usize,
    /// Success status
    pub success: bool,
}

/// Device selection strategy
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceSelectionStrategy {
    /// Always use CPU
    ForceCpu,
    /// Prefer Metal GPU if available
    PreferMetal,
    /// Automatic selection based on workload
    Auto,
    /// Custom selection based on criteria
    Custom {
        min_memory_gb: Option<f32>,
        required_operations: Vec<TensorOperation>,
        preferred_dtype: Option<BitNetDType>,
    },
}

/// Memory allocation strategy recommendation
#[derive(Debug, Clone)]
pub struct AllocationStrategy {
    /// Recommended device
    pub device: Device,
    /// Recommended alignment
    pub alignment: usize,
    /// Memory optimization hint
    pub memory_hint: MemoryHint,
}

/// Memory optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryHint {
    /// Standard memory allocation
    Standard,
    /// Optimized for CPU operations
    CPUOptimized,
    /// Optimized for GPU operations
    GPUOptimized,
    /// Optimized for frequent access
    HighFrequency,
    /// Optimized for large sequential access
    Sequential,
}

/// Device operation errors
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    /// Device not supported for operation
    #[error("Device {device:?} not supported for operation: {reason}")]
    UnsupportedDevice { device: Device, reason: String },

    /// Data type not supported on device
    #[error("Data type {dtype:?} not supported on device {device:?}")]
    UnsupportedDataType { device: Device, dtype: BitNetDType },

    /// Tensor size exceeds device limits
    #[error("Tensor size {size} bytes exceeds device limit {limit} bytes")]
    TensorSizeExceeded { size: usize, limit: usize },

    /// Device migration failed
    #[error("Failed to migrate from {from:?} to {to:?}: {reason}")]
    MigrationFailed {
        from: Device,
        to: Device,
        reason: String,
    },

    /// Device capability detection failed
    #[error("Failed to detect device capabilities: {reason}")]
    CapabilityDetectionFailed { reason: String },
}

/// Result type for device operations
pub type DeviceResult<T> = std::result::Result<T, DeviceError>;

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            supported_dtypes: vec![
                BitNetDType::F32,
                BitNetDType::F16,
                BitNetDType::I32,
                BitNetDType::I8,
                BitNetDType::U8,
                BitNetDType::Bool,
            ],
            max_tensor_size: Some(1024 * 1024 * 1024), // 1GB default
            unified_memory: false,
            hardware_acceleration: false,
            memory_bandwidth: None,
            compute_capability: None,
            supported_operations: vec![
                TensorOperation::Arithmetic,
                TensorOperation::ElementWise,
                TensorOperation::Memory,
            ],
            metal_features: None,
        }
    }
}

impl TensorDeviceManager {
    /// Creates a new tensor device manager
    pub fn new(memory_pool: Arc<HybridMemoryPool>, device: Option<Device>) -> DeviceResult<Self> {
        let device = device.unwrap_or_else(auto_select_device);

        #[cfg(feature = "tracing")]
        info!("Creating tensor device manager for device {:?}", device);

        let memory_manager = TensorMemoryManager::new(memory_pool, device.clone());
        let capabilities = Self::detect_capabilities(&device)?;
        let performance_profile = Self::create_performance_profile(&device, &capabilities);

        Ok(Self {
            device,
            memory_manager,
            capabilities,
            performance_profile,
        })
    }

    /// Gets the current device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Gets the device capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Gets the performance profile
    pub fn performance_profile(&self) -> &DevicePerformanceProfile {
        &self.performance_profile
    }

    /// Gets the memory manager
    pub fn memory_manager(&self) -> &TensorMemoryManager {
        &self.memory_manager
    }

    /// Migrates to a different device
    pub fn migrate_to_device(
        &mut self,
        target_device: Device,
    ) -> DeviceResult<DeviceMigrationResult> {
        let start_time = std::time::Instant::now();
        let source_device = self.device.clone();

        #[cfg(feature = "tracing")]
        info!("Migrating from {:?} to {:?}", source_device, target_device);

        // Check if migration is necessary
        if std::mem::discriminant(&source_device) == std::mem::discriminant(&target_device) {
            #[cfg(feature = "tracing")]
            debug!("No migration needed - already on target device");

            return Ok(DeviceMigrationResult {
                from_device: source_device,
                to_device: target_device,
                migration_time_ms: 0.0,
                data_size_bytes: 0,
                success: true,
            });
        }

        // Update device and capabilities
        self.device = target_device.clone();
        self.capabilities = Self::detect_capabilities(&target_device)?;
        self.performance_profile =
            Self::create_performance_profile(&target_device, &self.capabilities);

        let migration_time = start_time.elapsed();

        #[cfg(feature = "tracing")]
        info!("Migration completed in {:?}", migration_time);

        Ok(DeviceMigrationResult {
            from_device: source_device,
            to_device: target_device,
            migration_time_ms: migration_time.as_secs_f64() * 1000.0,
            data_size_bytes: 0,
            success: true,
        })
    }

    /// Selects the best device for a given workload
    pub fn select_optimal_device(
        &self,
        strategy: DeviceSelectionStrategy,
        tensor_size: usize,
        dtype: BitNetDType,
        operations: &[TensorOperation],
    ) -> DeviceResult<Device> {
        match strategy {
            DeviceSelectionStrategy::ForceCpu => Ok(get_cpu_device()),

            DeviceSelectionStrategy::PreferMetal => match get_metal_device() {
                Ok(device) => {
                    let caps = Self::detect_capabilities(&device)?;
                    if caps.supports_dtype(dtype) {
                        Ok(device)
                    } else {
                        #[cfg(feature = "tracing")]
                        warn!(
                            "Metal device doesn't support dtype {:?}, falling back to CPU",
                            dtype
                        );
                        Ok(get_cpu_device())
                    }
                }
                Err(_) => {
                    #[cfg(feature = "tracing")]
                    debug!("Metal device not available, using CPU");
                    Ok(get_cpu_device())
                }
            },

            DeviceSelectionStrategy::Auto => {
                let prefers_gpu = tensor_size > 64 * 1024
                    || operations.contains(&TensorOperation::MatrixMultiplication)
                    || operations.contains(&TensorOperation::Convolution);

                if prefers_gpu {
                    match get_metal_device() {
                        Ok(device) => {
                            let caps = Self::detect_capabilities(&device)?;
                            if caps.supports_dtype(dtype) && caps.supports_operations(operations) {
                                return Ok(device);
                            }
                        }
                        Err(_) => {}
                    }
                }

                Ok(get_cpu_device())
            }

            DeviceSelectionStrategy::Custom {
                min_memory_gb,
                required_operations,
                preferred_dtype,
            } => {
                if let Ok(device) = get_metal_device() {
                    let caps = Self::detect_capabilities(&device)?;

                    let memory_ok = min_memory_gb.map_or(true, |min_gb| {
                        caps.max_tensor_size.map_or(false, |max_bytes| {
                            max_bytes as f32 >= min_gb * 1024.0 * 1024.0 * 1024.0
                        })
                    });

                    let ops_ok = caps.supports_operations(&required_operations);
                    let dtype_ok = preferred_dtype.map_or(true, |dt| caps.supports_dtype(dt));

                    if memory_ok && ops_ok && dtype_ok {
                        return Ok(device);
                    }
                }

                Ok(get_cpu_device())
            }
        }
    }

    /// Detects device capabilities
    fn detect_capabilities(device: &Device) -> DeviceResult<DeviceCapabilities> {
        let mut capabilities = DeviceCapabilities::default();

        match device {
            Device::Cpu => {
                capabilities.supported_dtypes = vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::BF16,
                    BitNetDType::I8,
                    BitNetDType::I16,
                    BitNetDType::I32,
                    BitNetDType::I64,
                    BitNetDType::U8,
                    BitNetDType::U16,
                    BitNetDType::U32,
                    BitNetDType::U64,
                    BitNetDType::Bool,
                    BitNetDType::BitNet158,
                    BitNetDType::BitNet1,
                    BitNetDType::Int4,
                    BitNetDType::QInt8,
                    BitNetDType::QInt4,
                ];
                capabilities.unified_memory = true;
                capabilities.hardware_acceleration =
                    cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64");
                capabilities.compute_capability = Some("CPU with SIMD support".to_string());
                capabilities.supported_operations = vec![
                    TensorOperation::Arithmetic,
                    TensorOperation::MatrixMultiplication,
                    TensorOperation::Reduction,
                    TensorOperation::ElementWise,
                    TensorOperation::Memory,
                    TensorOperation::BitNetQuantization,
                ];
            }

            Device::Metal(_) => {
                capabilities.supported_dtypes = vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I32,
                    BitNetDType::U32,
                    BitNetDType::Bool,
                    BitNetDType::BitNet158,
                    BitNetDType::BitNet1,
                ];
                capabilities.unified_memory = cfg!(target_os = "macos");
                capabilities.hardware_acceleration = true;
                capabilities.compute_capability = Some("Metal GPU".to_string());
                capabilities.memory_bandwidth = Some(800.0);
                capabilities.supported_operations = vec![
                    TensorOperation::Arithmetic,
                    TensorOperation::MatrixMultiplication,
                    TensorOperation::Convolution,
                    TensorOperation::Reduction,
                    TensorOperation::ElementWise,
                    TensorOperation::Memory,
                    TensorOperation::MixedPrecision,
                ];

                capabilities.metal_features = Some(MetalFeatures {
                    max_threadgroup_memory: 32 * 1024,
                    has_unified_memory: cfg!(target_os = "macos"),
                    gpu_family: Some("Apple GPU".to_string()),
                    compute_units: None,
                });
            }

            Device::Cuda(_) => {
                capabilities.supported_dtypes = vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I32,
                    BitNetDType::I8,
                ];
                capabilities.hardware_acceleration = true;
                capabilities.compute_capability = Some("CUDA GPU".to_string());
            }
        }

        #[cfg(feature = "tracing")]
        debug!("Device capabilities: {:?}", capabilities);

        Ok(capabilities)
    }

    /// Creates a performance profile for a device
    fn create_performance_profile(
        device: &Device,
        capabilities: &DeviceCapabilities,
    ) -> DevicePerformanceProfile {
        match device {
            Device::Cpu => DevicePerformanceProfile {
                memory_latency_ms: 0.001,
                peak_bandwidth_gbps: capabilities.memory_bandwidth.unwrap_or(100.0),
                compute_throughput_gflops: 200.0,
                preferred_dtypes: vec![BitNetDType::F32, BitNetDType::I32, BitNetDType::BitNet158],
                optimal_tensor_sizes: TensorSizeRecommendation {
                    small_threshold: 1024,
                    large_threshold: 1024 * 1024,
                    preferred_batch_sizes: vec![1, 8, 16, 32],
                },
            },

            Device::Metal(_) => DevicePerformanceProfile {
                memory_latency_ms: 0.1,
                peak_bandwidth_gbps: capabilities.memory_bandwidth.unwrap_or(800.0),
                compute_throughput_gflops: 3000.0,
                preferred_dtypes: vec![BitNetDType::F16, BitNetDType::F32],
                optimal_tensor_sizes: TensorSizeRecommendation {
                    small_threshold: 64 * 1024,
                    large_threshold: 64 * 1024 * 1024,
                    preferred_batch_sizes: vec![32, 64, 128, 256],
                },
            },

            Device::Cuda(_) => DevicePerformanceProfile {
                memory_latency_ms: 0.05,
                peak_bandwidth_gbps: 900.0,
                compute_throughput_gflops: 5000.0,
                preferred_dtypes: vec![BitNetDType::F16, BitNetDType::F32],
                optimal_tensor_sizes: TensorSizeRecommendation {
                    small_threshold: 32 * 1024,
                    large_threshold: 128 * 1024 * 1024,
                    preferred_batch_sizes: vec![64, 128, 256, 512],
                },
            },
        }
    }

    /// Checks if current device supports a data type
    pub fn supports_dtype(&self, dtype: BitNetDType) -> bool {
        self.capabilities.supports_dtype(dtype)
    }

    /// Checks if current device supports a set of operations
    pub fn supports_operations(&self, operations: &[TensorOperation]) -> bool {
        self.capabilities.supports_operations(operations)
    }

    /// Creates a device-specific tensor allocation recommendation
    pub fn recommend_allocation_strategy(
        &self,
        size_bytes: usize,
        dtype: BitNetDType,
    ) -> AllocationStrategy {
        if !self.supports_dtype(dtype) {
            return AllocationStrategy {
                device: get_cpu_device(),
                alignment: 16,
                memory_hint: MemoryHint::Standard,
            };
        }

        let alignment = match dtype {
            BitNetDType::F32 | BitNetDType::I32 | BitNetDType::U32 => 16,
            BitNetDType::F16 | BitNetDType::BF16 => 8,
            BitNetDType::I64 | BitNetDType::U64 => 32,
            _ => 8,
        };

        let memory_hint = match self.device {
            Device::Metal(_) if size_bytes > 1024 * 1024 => MemoryHint::GPUOptimized,
            Device::Cpu if dtype.is_quantized() => MemoryHint::CPUOptimized,
            _ => MemoryHint::Standard,
        };

        AllocationStrategy {
            device: self.device.clone(),
            alignment,
            memory_hint,
        }
    }
}

impl DeviceCapabilities {
    /// Checks if this device supports a specific data type
    pub fn supports_dtype(&self, dtype: BitNetDType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Checks if this device supports all required operations
    pub fn supports_operations(&self, operations: &[TensorOperation]) -> bool {
        operations
            .iter()
            .all(|op| self.supported_operations.contains(op))
    }

    /// Gets the maximum tensor size for this device
    pub fn max_tensor_size_bytes(&self) -> Option<usize> {
        self.max_tensor_size
    }

    /// Checks if this device has unified memory
    pub fn has_unified_memory(&self) -> bool {
        self.unified_memory
    }
}

/// Device management utilities
pub struct DeviceUtils;

impl DeviceUtils {
    /// Converts tensors to CPU device
    pub fn to_cpu() -> Device {
        get_cpu_device()
    }

    /// Converts tensors to Metal GPU device if available
    pub fn to_gpu() -> DeviceResult<Device> {
        get_metal_device().map_err(|e| DeviceError::UnsupportedDevice {
            device: get_cpu_device(),
            reason: format!("Metal GPU not available: {}", e),
        })
    }

    /// Automatically selects the best device
    pub fn auto_device() -> Device {
        auto_select_device()
    }

    /// Creates a device manager with automatic device selection
    pub fn create_auto_manager(
        memory_pool: Arc<HybridMemoryPool>,
    ) -> DeviceResult<TensorDeviceManager> {
        TensorDeviceManager::new(memory_pool, None)
    }

    /// Creates a device manager with explicit device
    pub fn create_manager(
        memory_pool: Arc<HybridMemoryPool>,
        device: Device,
    ) -> DeviceResult<TensorDeviceManager> {
        TensorDeviceManager::new(memory_pool, Some(device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::default();
        assert!(caps.supports_dtype(BitNetDType::F32));
        assert!(caps.supports_operations(&[TensorOperation::Arithmetic]));
    }

    #[test]
    fn test_device_manager_creation() -> DeviceResult<()> {
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| {
            DeviceError::CapabilityDetectionFailed {
                reason: format!("Memory pool creation failed: {}", e),
            }
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;
        assert!(manager.supports_dtype(BitNetDType::F32));
        Ok(())
    }

    #[test]
    fn test_device_selection_strategies() -> DeviceResult<()> {
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| {
            DeviceError::CapabilityDetectionFailed {
                reason: format!("Memory pool creation failed: {}", e),
            }
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;

        let cpu_device = manager.select_optimal_device(
            DeviceSelectionStrategy::ForceCpu,
            1024,
            BitNetDType::F32,
            &[TensorOperation::Arithmetic],
        )?;
        assert!(matches!(cpu_device, Device::Cpu));

        let auto_device = manager.select_optimal_device(
            DeviceSelectionStrategy::Auto,
            1024 * 1024,
            BitNetDType::F32,
            &[TensorOperation::MatrixMultiplication],
        )?;
        assert!(matches!(auto_device, Device::Cpu | Device::Metal(_)));

        Ok(())
    }

    #[test]
    fn test_allocation_strategy_recommendation() -> DeviceResult<()> {
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| {
            DeviceError::CapabilityDetectionFailed {
                reason: format!("Memory pool creation failed: {}", e),
            }
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;

        let strategy = manager.recommend_allocation_strategy(1024, BitNetDType::F32);
        assert_eq!(strategy.alignment, 16);
        assert!(matches!(
            strategy.memory_hint,
            MemoryHint::Standard | MemoryHint::CPUOptimized
        ));

        Ok(())
    }

    #[test]
    fn test_device_utils() {
        let cpu_device = DeviceUtils::to_cpu();
        assert!(matches!(cpu_device, Device::Cpu));

        let auto_device = DeviceUtils::auto_device();
        assert!(matches!(auto_device, Device::Cpu | Device::Metal(_)));
    }

    #[test]
    fn test_tensor_operation_comparison() {
        assert_eq!(TensorOperation::Arithmetic, TensorOperation::Arithmetic);
        assert_ne!(
            TensorOperation::Arithmetic,
            TensorOperation::MatrixMultiplication
        );
    }

    #[test]
    fn test_memory_hint_comparison() {
        assert_eq!(MemoryHint::Standard, MemoryHint::Standard);
        assert_ne!(MemoryHint::CPUOptimized, MemoryHint::GPUOptimized);
    }
}
