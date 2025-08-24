//! Tensor Device Integration
//!
//! This module provides device-aware tensor operations that integrate seamlessly
//! with the existing device abstraction, supporting CPU, Metal GPU, and MLX
//! acceleration with automatic device selection and migration.

use std::sync::Arc;
use candle_core::Device;
use crate::device::{auto_select_device, get_cpu_device, get_metal_device};
use super::dtype::BitNetDType;
use super::memory_integration::TensorMemoryManager;
use crate::memory::HybridMemoryPool;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

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
    MigrationFailed { from: Device, to: Device, reason: String },

    /// Device capability detection failed
    #[error("Failed to detect device capabilities: {reason}")]
    CapabilityDetectionFailed { reason: String },
}

/// Result type for device operations
pub type DeviceResult<T> = std::result::Result<T, DeviceError>;

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            supported_dtypes: BitNetDType::all_types().to_vec(),
            max_tensor_size: None,
            unified_memory: false,
            hardware_acceleration: false,
            memory_bandwidth: None,
            compute_capability: None,
        }
    }
}

impl TensorDeviceManager {
    /// Creates a new tensor device manager
    ///
    /// # Arguments
    ///
    /// * `device` - Target device for tensor operations
    /// * `pool` - Memory pool for allocations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorDeviceManager;
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use bitnet_core::device::get_cpu_device;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// let device = get_cpu_device();
    /// let manager = TensorDeviceManager::new(device, pool)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Device, pool: Arc<HybridMemoryPool>) -> DeviceResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Creating tensor device manager for device {:?}", device);

        let capabilities = Self::detect_device_capabilities(&device)?;
        let memory_manager = TensorMemoryManager::new(pool, device.clone());

        Ok(Self {
            device,
            memory_manager,
            capabilities,
        })
    }

    /// Creates a tensor device manager with auto-selected device
    ///
    /// # Arguments
    ///
    /// * `pool` - Memory pool for allocations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorDeviceManager;
    /// use bitnet_core::memory::HybridMemoryPool;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = Arc::new(HybridMemoryPool::new()?);
    /// let manager = TensorDeviceManager::auto(pool)?;
    /// println!("Using device: {:?}", manager.device());
    /// # Ok(())
    /// # }
    /// ```
    pub fn auto(pool: Arc<HybridMemoryPool>) -> DeviceResult<Self> {
        let device = auto_select_device();
        Self::new(device, pool)
    }

    /// Returns the current device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns device capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Returns the memory manager
    pub fn memory_manager(&self) -> &TensorMemoryManager {
        &self.memory_manager
    }

    /// Checks if a data type is supported on this device
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type to check
    ///
    /// # Returns
    ///
    /// True if the data type is supported
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bitnet_core::tensor::{TensorDeviceManager, BitNetDType};
    /// # use bitnet_core::memory::HybridMemoryPool;
    /// # use bitnet_core::device::get_cpu_device;
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let pool = Arc::new(HybridMemoryPool::new()?);
    /// # let device = get_cpu_device();
    /// # let manager = TensorDeviceManager::new(device, pool)?;
    ///
    /// assert!(manager.supports_dtype(BitNetDType::F32));
    /// # Ok(())
    /// # }
    /// ```
    pub fn supports_dtype(&self, dtype: BitNetDType) -> bool {
        self.capabilities.supported_dtypes.contains(&dtype)
    }

    /// Checks if a tensor size is supported on this device
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size in bytes to check
    ///
    /// # Returns
    ///
    /// True if the size is supported
    pub fn supports_tensor_size(&self, size_bytes: usize) -> bool {
        if let Some(max_size) = self.capabilities.max_tensor_size {
            size_bytes <= max_size
        } else {
            true // No limit known
        }
    }

    /// Migrates to a different device
    ///
    /// # Arguments
    ///
    /// * `new_device` - Target device to migrate to
    /// * `pool` - Memory pool for the new device
    ///
    /// # Returns
    ///
    /// Result containing new TensorDeviceManager for the target device
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bitnet_core::tensor::TensorDeviceManager;
    /// # use bitnet_core::memory::HybridMemoryPool;
    /// # use bitnet_core::device::{get_cpu_device, auto_select_device};
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let pool = Arc::new(HybridMemoryPool::new()?);
    /// # let device = get_cpu_device();
    /// # let manager = TensorDeviceManager::new(device, pool.clone())?;
    ///
    /// let new_device = auto_select_device();
    /// let migrated_manager = manager.migrate_to_device(new_device, pool)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn migrate_to_device(
        &self,
        new_device: Device,
        pool: Arc<HybridMemoryPool>,
    ) -> DeviceResult<TensorDeviceManager> {
        #[cfg(feature = "tracing")]
        info!("Migrating tensor device manager from {:?} to {:?}", self.device, new_device);

        // Create new manager for target device
        let new_manager = Self::new(new_device, pool)?;

        #[cfg(feature = "tracing")]
        info!("Successfully migrated tensor device manager");

        Ok(new_manager)
    }

    /// Gets the optimal device for a specific operation
    ///
    /// # Arguments
    ///
    /// * `operation` - Type of operation to optimize for
    /// * `tensor_size` - Size of tensors involved
    /// * `dtype` - Data type of tensors
    ///
    /// # Returns
    ///
    /// Recommended device for the operation
    pub fn get_optimal_device(
        operation: TensorOperation,
        tensor_size: usize,
        dtype: BitNetDType,
    ) -> Device {
        match operation {
            TensorOperation::MatrixMultiplication => {
                // Prefer GPU for large matrix operations
                if tensor_size > 1024 * 1024 { // > 1MB
                    get_metal_device().unwrap_or_else(|_| get_cpu_device())
                } else {
                    get_cpu_device()
                }
            }
            TensorOperation::ElementWise => {
                // CPU is often better for small element-wise operations
                if tensor_size < 64 * 1024 { // < 64KB
                    get_cpu_device()
                } else {
                    auto_select_device()
                }
            }
            TensorOperation::Reduction => {
                // GPU is good for reductions on large tensors
                if tensor_size > 512 * 1024 { // > 512KB
                    get_metal_device().unwrap_or_else(|_| get_cpu_device())
                } else {
                    get_cpu_device()
                }
            }
            TensorOperation::Quantization => {
                // BitNet quantization operations
                if dtype.is_quantized() {
                    // Use CPU for BitNet-specific quantization
                    get_cpu_device()
                } else {
                    auto_select_device()
                }
            }
            TensorOperation::Broadcasting => {
                // Broadcasting is typically memory-bound
                auto_select_device()
            }
            TensorOperation::Creation => {
                // Tensor creation - use auto-selected device
                auto_select_device()
            }
        }
    }

    /// Detects device capabilities
    fn detect_device_capabilities(device: &Device) -> DeviceResult<DeviceCapabilities> {
        let mut capabilities = DeviceCapabilities::default();

        match device {
            Device::Cpu => {
                #[cfg(feature = "tracing")]
                debug!("Detecting CPU device capabilities");

                capabilities.unified_memory = true; // CPU uses system RAM
                capabilities.hardware_acceleration = false;
                capabilities.compute_capability = Some("CPU".to_string());

                // CPU supports all data types
                capabilities.supported_dtypes = BitNetDType::all_types().to_vec();
            }
            Device::Metal(_) => {
                #[cfg(feature = "tracing")]
                debug!("Detecting Metal GPU device capabilities");

                capabilities.unified_memory = true; // Metal on Apple Silicon uses unified memory
                capabilities.hardware_acceleration = true;
                capabilities.compute_capability = Some("Metal GPU".to_string());

                // Metal supports most data types but may have limitations
                capabilities.supported_dtypes = vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I32,
                    BitNetDType::I8,
                    BitNetDType::U8,
                    BitNetDType::U32,
                    // BitNet-specific types may need special handling
                    BitNetDType::BitNet158,
                    BitNetDType::BitNet1,
                ];

                // Estimate memory bandwidth for Apple Silicon GPUs
                capabilities.memory_bandwidth = Some(400.0); // Rough estimate
            }
            Device::Cuda(_) => {
                #[cfg(feature = "tracing")]
                debug!("Detecting CUDA device capabilities");

                capabilities.unified_memory = false; // CUDA typically uses separate memory
                capabilities.hardware_acceleration = true;
                capabilities.compute_capability = Some("CUDA GPU".to_string());

                // CUDA has good support for standard types
                capabilities.supported_dtypes = vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I32,
                    BitNetDType::I8,
                    BitNetDType::U8,
                    BitNetDType::U32,
                ];
            }
        }

        #[cfg(feature = "tracing")]
        debug!("Device capabilities: {:?}", capabilities);

        Ok(capabilities)
    }

    /// Creates a device-specific tensor allocation recommendation
    pub fn recommend_allocation_strategy(&self, size_bytes: usize, dtype: BitNetDType) -> AllocationStrategy {
        if !self.supports_dtype(dtype) {
            return AllocationStrategy {
                device: get_cpu_device(), // Fallback to CPU
                alignment: 16,
                memory_hint: MemoryHint::Standard,
            };
        }

        let alignment = match dtype {
            BitNetDType::F32 | BitNetDType::I32 | BitNetDType::U32 => 16, // SIMD alignment
            BitNetDType::F16 | BitNetDType::BF16 => 8,
            BitNetDType::I64 | BitNetDType::U64 => 32, // Cache line alignment for large types
            _ => 8, // Standard alignment
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
        operations.iter().all(|op| self.supported_operations.contains(op))
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
    /// Converts tensors to a target device
    ///
    /// This is a utility function for tensor device migration.
    /// In practice, this would be implemented by the tensor types themselves.
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
    pub fn create_auto_manager(memory_pool: Arc<HybridMemoryPool>) -> DeviceResult<TensorDeviceManager> {
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
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| DeviceError::CapabilityDetectionFailed {
            reason: format!("Memory pool creation failed: {}", e),
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;
        assert!(manager.supports_dtype(BitNetDType::F32));
        Ok(())
    }

    #[test]
    fn test_device_selection_strategies() -> DeviceResult<()> {
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| DeviceError::CapabilityDetectionFailed {
            reason: format!("Memory pool creation failed: {}", e),
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;

        // Test force CPU
        let cpu_device = manager.select_optimal_device(
            DeviceSelectionStrategy::ForceCpu,
            1024,
            BitNetDType::F32,
            &[TensorOperation::Arithmetic],
        )?;
        assert!(matches!(cpu_device, Device::Cpu));

        // Test auto selection
        let auto_device = manager.select_optimal_device(
            DeviceSelectionStrategy::Auto,
            1024 * 1024, // Large tensor
            BitNetDType::F32,
            &[TensorOperation::MatrixMultiplication],
        )?;
        // Should be CPU or Metal depending on availability
        assert!(matches!(auto_device, Device::Cpu | Device::Metal(_)));

        Ok(())
    }

    #[test]
    fn test_allocation_strategy_recommendation() -> DeviceResult<()> {
        let pool = Arc::new(HybridMemoryPool::new().map_err(|e| DeviceError::CapabilityDetectionFailed {
            reason: format!("Memory pool creation failed: {}", e),
        })?);

        let manager = TensorDeviceManager::new(pool, None)?;

        let strategy = manager.recommend_allocation_strategy(1024, BitNetDType::F32);
        assert_eq!(strategy.alignment, 16); // SIMD alignment for F32
        assert!(matches!(strategy.memory_hint, MemoryHint::Standard | MemoryHint::CPUOptimized));

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
        assert_ne!(TensorOperation::Arithmetic, TensorOperation::MatrixMultiplication);
    }

    #[test]
    fn test_memory_hint_comparison() {
        assert_eq!(MemoryHint::Standard, MemoryHint::Standard);
        assert_ne!(MemoryHint::CPUOptimized, MemoryHint::GPUOptimized);
    }
}

/// Types of tensor operations for device optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOperation {
    /// Matrix multiplication operations
    MatrixMultiplication,
    /// Element-wise operations (add, multiply, etc.)
    ElementWise,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
    /// Quantization operations
    Quantization,
    /// Broadcasting operations
    Broadcasting,
    /// Tensor creation operations
    Creation,
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
    MigrationFailed { from: Device, to: Device, reason: String },

    /// Device capability detection failed
    #[error("Failed to detect device capabilities: {reason}")]
    CapabilityDetectionFailed { reason: String },
}

/// Result type for device operations
pub type DeviceResult<T> = std::result::Result<T, DeviceError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;
    use crate::device::get_cpu_device;
    use std::sync::Arc;

    #[test]
    fn test_tensor_device_manager_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorDeviceManager::new(device, pool).unwrap();

        assert!(matches!(manager.device(), Device::Cpu));
        assert!(manager.supports_dtype(BitNetDType::F32));
        assert!(manager.supports_tensor_size(1024));
    }

    #[test]
    fn test_auto_device_manager() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let manager = TensorDeviceManager::auto(pool).unwrap();

        // Should successfully create with auto-selected device
        assert!(manager.capabilities().supported_dtypes.len() > 0);
    }

    #[test]
    fn test_device_capabilities() {
        let capabilities = DeviceCapabilities::default();
        assert!(capabilities.supported_dtypes.len() > 0);
        assert!(!capabilities.hardware_acceleration);
        assert!(!capabilities.unified_memory);
    }

    #[test]
    fn test_optimal_device_selection() {
        // Test matrix multiplication recommendation
        let device = TensorDeviceManager::get_optimal_device(
            TensorOperation::MatrixMultiplication,
            2 * 1024 * 1024, // 2MB
            BitNetDType::F32
        );
        // Should prefer GPU for large matrices, but CPU is okay as fallback
        assert!(matches!(device, Device::Cpu | Device::Metal(_)));

        // Test small element-wise operation
        let device = TensorDeviceManager::get_optimal_device(
            TensorOperation::ElementWise,
            1024, // 1KB
            BitNetDType::F32
        );
        // Should prefer CPU for small operations
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_allocation_strategy() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorDeviceManager::new(device, pool).unwrap();

        let strategy = manager.recommend_allocation_strategy(1024, BitNetDType::F32);
        assert_eq!(strategy.alignment, 16); // SIMD alignment for F32
        assert!(matches!(strategy.device, Device::Cpu));
    }

    #[test]
    fn test_device_migration() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorDeviceManager::new(device, pool.clone()).unwrap();

        let new_device = auto_select_device();
        let migrated = manager.migrate_to_device(new_device, pool).unwrap();

        // Migration should succeed (might migrate to same device)
        assert!(migrated.capabilities().supported_dtypes.len() > 0);
    }

    #[test]
    fn test_dtype_support() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let manager = TensorDeviceManager::new(device, pool).unwrap();

        // CPU should support all standard data types
        assert!(manager.supports_dtype(BitNetDType::F32));
        assert!(manager.supports_dtype(BitNetDType::I32));
        assert!(manager.supports_dtype(BitNetDType::BitNet158));
    }
}
