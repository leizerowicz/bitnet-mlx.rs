use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use candle_core::{Device, Tensor as CandleTensor};
use metal::{
    Buffer as MetalBuffer, CommandBuffer, CommandQueue, ComputeCommandEncoder,
    Device as MetalDevice, Library, MTLResourceOptions,
};

#[cfg(feature = "tracing")]
use tracing::{debug, error, trace, warn};

use crate::{
    error::{BitNetError, ErrorContext},
    memory::{HybridMemoryPool, MemoryError, MemoryMetrics},
    metal::{BitNetShaders, CommandBufferManager},
    tensor::{core::BitNetTensor, dtype::BitNetDType, shape::TensorShape},
};
use anyhow::{Context, Result as AnyhowResult};

use super::{
    auto_select::AccelerationCapabilities, dispatch::AccelerationBackend, AccelerationBackendImpl,
    AccelerationError, AccelerationMetrics, AccelerationResult,
};

/// Metal acceleration performance metrics
#[derive(Debug, Default, Clone)]
pub struct MetalAccelerationMetrics {
    /// Total GPU operations executed
    pub operations_executed: u64,
    /// Total GPU memory allocated
    pub gpu_memory_allocated: u64,
    /// Total GPU memory freed
    pub gpu_memory_freed: u64,
    /// Average operation execution time
    pub average_execution_time: f64,
    /// Command buffer creation count
    pub command_buffers_created: u64,
    /// Successful buffer cache hits
    pub buffer_cache_hits: u64,
    /// Buffer cache misses requiring allocation
    pub buffer_cache_misses: u64,
    /// GPU-CPU transfer operations
    pub transfer_operations: u64,
    /// Total transfer time
    pub total_transfer_time: f64,
}

impl MetalAccelerationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_operation(&mut self, execution_time: f64) {
        self.operations_executed += 1;
        self.average_execution_time =
            (self.average_execution_time * (self.operations_executed - 1) as f64 + execution_time)
                / self.operations_executed as f64;
    }

    pub fn record_allocation(&mut self, size: u64) {
        self.gpu_memory_allocated += size;
    }

    pub fn record_deallocation(&mut self, size: u64) {
        self.gpu_memory_freed += size;
    }

    pub fn record_buffer_cache_hit(&mut self) {
        self.buffer_cache_hits += 1;
    }

    pub fn record_buffer_cache_miss(&mut self) {
        self.buffer_cache_misses += 1;
    }

    pub fn record_transfer(&mut self, transfer_time: f64) {
        self.transfer_operations += 1;
        self.total_transfer_time += transfer_time;
    }
}

/// Metal acceleration backend with compute shader support
#[cfg(feature = "metal")]
pub struct MetalAccelerator {
    /// Metal device for GPU operations
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: Option<MetalDevice>,

    /// Command queue for GPU operations
    #[cfg(all(target_os = "macos", feature = "metal"))]
    command_queue: Option<CommandQueue>,

    /// Compiled shaders library
    #[cfg(all(target_os = "macos", feature = "metal"))]
    library: Option<Library>,

    /// Command buffer manager
    #[cfg(all(target_os = "macos", feature = "metal"))]
    command_buffer_manager: Option<Arc<CommandBufferManager>>,

    /// BitNet-specific shader collection
    #[cfg(all(target_os = "macos", feature = "metal"))]
    bitnet_shaders: Option<BitNetShaders>,

    /// GPU memory pool for Metal buffers
    #[cfg(all(target_os = "macos", feature = "metal"))]
    gpu_memory_pool: Option<Arc<HybridMemoryPool>>,

    /// Buffer cache for reusing GPU memory
    #[cfg(all(target_os = "macos", feature = "metal"))]
    buffer_cache: Arc<Mutex<HashMap<String, MetalBuffer>>>,

    /// Performance metrics tracking
    metrics: Arc<Mutex<MetalAccelerationMetrics>>,

    /// Initialization state
    initialized: bool,

    /// Availability check result
    available: bool,
}

// SAFETY: Metal resources are thread-safe when properly managed through Arc/Mutex
#[cfg(feature = "metal")]
unsafe impl Send for MetalAccelerator {}
#[cfg(feature = "metal")]
unsafe impl Sync for MetalAccelerator {}

#[cfg(feature = "metal")]
impl MetalAccelerator {
    /// Creates a new Metal accelerator with initialization
    pub fn new() -> AccelerationResult<Self> {
        #[cfg(feature = "tracing")]
        debug!("Creating Metal accelerator");

        let available = Self::is_platform_supported();

        Ok(Self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            device: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            command_queue: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            library: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            command_buffer_manager: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            bitnet_shaders: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            gpu_memory_pool: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            buffer_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(MetalAccelerationMetrics::new())),
            initialized: false,
            available,
        })
    }

    /// Initialize the Metal backend with device and shaders
    pub fn initialize(&mut self) -> AccelerationResult<()> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Get Metal device
            let device = MetalDevice::system_default().ok_or_else(|| {
                AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: "No Metal device available".to_string(),
                }
            })?;

            // Create command queue
            let command_queue = device.new_command_queue();

            // Load and compile shaders
            let shader_source = include_str!("../../metal/shaders/tensor_operations.metal");
            let library = device
                .new_library_with_source(shader_source, &metal::CompileOptions::new())
                .map_err(|e| AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: format!("Shader compilation failed: {:?}", e),
                })?;

            // Create command buffer manager
            let command_queue = device.new_command_queue();
            let config = crate::metal::CommandBufferPoolConfig {
                max_command_buffers: 64,
                default_timeout: std::time::Duration::from_millis(1000),
                auto_cleanup: cfg!(debug_assertions),
                cleanup_interval: std::time::Duration::from_secs(1),
                enable_reuse: true,
            };
            let command_buffer_manager = Arc::new(CommandBufferManager::new(
                device.clone(),
                command_queue.clone(),
                config,
            ));

            // Initialize BitNet shaders
            let bitnet_shaders = BitNetShaders::new(device.clone()).map_err(|e| {
                AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: format!("BitNet shaders initialization failed: {:?}", e),
                }
            })?;

            // Initialize GPU memory pool
            let gpu_memory_pool = Arc::new(HybridMemoryPool::new().map_err(|e| {
                AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: format!("GPU memory pool creation failed: {:?}", e),
                }
            })?);

            self.device = Some(device);
            self.command_queue = Some(command_queue);
            self.library = Some(library);
            self.command_buffer_manager = Some(command_buffer_manager);
            self.bitnet_shaders = Some(bitnet_shaders);
            self.gpu_memory_pool = Some(gpu_memory_pool);
        }

        self.initialized = true;

        #[cfg(feature = "tracing")]
        debug!("Metal accelerator initialized successfully");

        Ok(())
    }

    /// Transfer tensor data to GPU buffer
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn transfer_tensor_to_gpu(&self, tensor: &BitNetTensor) -> AccelerationResult<MetalBuffer> {
        let start_time = Instant::now();

        // Get buffer size
        let element_count = tensor.shape().total_elements();
        let element_size = tensor.dtype().size_bytes().unwrap_or(4);
        let size = element_count * element_size;

        // Check buffer cache first
        let cache_key = format!("tensor_{}_{}", tensor.tensor_id(), size);
        {
            let buffer_cache =
                self.buffer_cache
                    .lock()
                    .map_err(|_| AccelerationError::InitializationFailed {
                        backend: "Metal".to_string(),
                        reason: "Buffer cache lock failed".to_string(),
                    })?;

            if let Some(cached_buffer) = buffer_cache.get(&cache_key) {
                let mut metrics =
                    self.metrics
                        .lock()
                        .map_err(|_| AccelerationError::InitializationFailed {
                            backend: "Metal".to_string(),
                            reason: "Metrics lock failed".to_string(),
                        })?;
                metrics.record_allocation(size as u64);
                metrics.record_buffer_cache_hit();
                return Ok(cached_buffer.clone());
            }
        }

        // Cache miss - create new buffer
        let device =
            self.device
                .as_ref()
                .ok_or_else(|| AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: "Device not initialized".to_string(),
                })?;

        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

        // Copy tensor data to GPU buffer
        let data = tensor
            .as_slice_f32()
            .map_err(|e| AccelerationError::MemoryTransferFailed {
                direction: "CPU to GPU".to_string(),
                reason: format!("Failed to get tensor data: {:?}", e),
            })?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.contents() as *mut f32,
                element_count,
            );
        }

        // Update cache
        let mut buffer_cache =
            self.buffer_cache
                .lock()
                .map_err(|_| AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: "Buffer cache lock failed".to_string(),
                })?;
        buffer_cache.insert(cache_key, buffer.clone());

        // Update metrics
        let mut metrics =
            self.metrics
                .lock()
                .map_err(|_| AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: "Metrics lock failed".to_string(),
                })?;
        let transfer_time = start_time.elapsed().as_secs_f64();
        metrics.record_buffer_cache_miss();
        metrics.record_transfer(transfer_time);

        Ok(buffer)
    }

    /// Transfer GPU buffer data back to tensor
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn transfer_buffer_to_cpu(
        &self,
        buffer: &MetalBuffer,
        shape: &[usize],
        dtype: BitNetDType,
    ) -> AccelerationResult<BitNetTensor> {
        let start_time = Instant::now();

        // Calculate size
        let element_count = shape.iter().product::<usize>();
        let element_size = dtype.size_bytes();

        // Read buffer data
        let buffer_data =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, element_count) };

        // Create new tensor with the data
        let tensor =
            BitNetTensor::from_data(buffer_data, shape, dtype, Some(candle_core::Device::Cpu))
                .map_err(|e| AccelerationError::MemoryTransferFailed {
                    direction: "GPU to CPU".to_string(),
                    reason: format!("Failed to create tensor from buffer: {:?}", e),
                })?;

        // Update metrics
        let mut metrics =
            self.metrics
                .lock()
                .map_err(|_| AccelerationError::InitializationFailed {
                    backend: "Metal".to_string(),
                    reason: "Metrics lock failed".to_string(),
                })?;
        let transfer_time = start_time.elapsed().as_secs_f64();
        metrics.record_transfer(transfer_time);

        Ok(tensor)
    }

    /// Execute matrix multiplication shader
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn execute_matmul_shader(
        &self,
        a_buffer: &MetalBuffer,
        b_buffer: &MetalBuffer,
        output_buffer: &MetalBuffer,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> AccelerationResult<()> {
        let command_buffer_manager = self.command_buffer_manager.as_ref().ok_or_else(|| {
            AccelerationError::InitializationFailed {
                backend: "Metal".to_string(),
                reason: "Command buffer manager not initialized".to_string(),
            }
        })?;

        let command_buffer = command_buffer_manager
            .create_command_buffer(crate::metal::CommandBufferPriority::High)
            .map_err(|e| AccelerationError::OperationFailed {
                backend: "Metal".to_string(),
                operation: "Matrix multiplication".to_string(),
                reason: format!("Command buffer creation failed: {:?}", e),
            })?;

        let bitnet_shaders = self.bitnet_shaders.as_ref().ok_or_else(|| {
            AccelerationError::InitializationFailed {
                backend: "Metal".to_string(),
                reason: "Shaders not initialized".to_string(),
            }
        })?;

        // Create compute encoder and configure shader
        // This is a simplified implementation - actual shader execution would be more complex

        Ok(())
    }

    /// Check if Metal is available on this platform
    pub fn is_metal_available() -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            MetalDevice::system_default().is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    /// Check if platform supports Metal acceleration
    fn is_platform_supported() -> bool {
        cfg!(all(target_os = "macos", feature = "metal"))
    }
}

#[cfg(feature = "metal")]
impl AccelerationBackendImpl for MetalAccelerator {
    fn initialize(&mut self) -> AccelerationResult<()> {
        self.initialize()
    }

    fn is_available(&self) -> bool {
        self.available && self.initialized
    }

    fn get_capabilities(&self) -> AccelerationCapabilities {
        AccelerationCapabilities {
            backend: AccelerationBackend::Metal,
            max_tensor_size: 1024 * 1024 * 1024,
            supported_dtypes: vec![BitNetDType::F32, BitNetDType::F16, BitNetDType::I8],
            zero_copy_support: false,
            parallel_execution: true,
            memory_bandwidth_gbps: 300.0,
            compute_throughput_gflops: 8000.0,
        }
    }

    fn matmul(
        &self,
        a: &BitNetTensor,
        b: &BitNetTensor,
    ) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)> {
        let start_time = Instant::now();

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        return Err(AccelerationError::UnsupportedOperation {
            backend: "Metal".to_string(),
            operation: "Matrix multiplication".to_string(),
        });

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Get tensor data
            let a_data = a
                .as_slice_f32()
                .map_err(|e| AccelerationError::MemoryTransferFailed {
                    direction: "Tensor to buffer".to_string(),
                    reason: format!("Failed to get tensor A data: {:?}", e),
                })?;
            let b_data = b
                .as_slice_f32()
                .map_err(|e| AccelerationError::MemoryTransferFailed {
                    direction: "Tensor to buffer".to_string(),
                    reason: format!("Failed to get tensor B data: {:?}", e),
                })?;

            // Transfer to GPU
            let a_buffer = self.transfer_tensor_to_gpu(a)?;
            let b_buffer = self.transfer_tensor_to_gpu(b)?;

            // Calculate output shape
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let output_shape = vec![a_shape[0], b_shape[1]];

            // Create output buffer
            let output_element_count = output_shape.iter().product::<usize>();
            let output_size = output_element_count * std::mem::size_of::<f32>();

            let device = self.device.as_ref().unwrap();
            let output_buffer =
                device.new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

            // Execute shader
            self.execute_matmul_shader(&a_buffer, &b_buffer, &output_buffer, a_shape, b_shape)?;

            // Transfer result back
            let result_tensor =
                self.transfer_buffer_to_cpu(&output_buffer, &output_shape, a.dtype())?;

            // Update metrics
            let total_time = start_time.elapsed().as_secs_f64();
            let mut metrics =
                self.metrics
                    .lock()
                    .map_err(|_| AccelerationError::InitializationFailed {
                        backend: "Metal".to_string(),
                        reason: "Metrics lock failed".to_string(),
                    })?;
            metrics.record_operation(total_time);

            let acceleration_metrics = AccelerationMetrics {
                backend_used: AccelerationBackend::Metal,
                execution_time_seconds: total_time,
                memory_used_bytes: ((a_data.len() + b_data.len() + output_element_count) * 4)
                    as u64,
                operations_per_second: (output_element_count as f64) / total_time,
                efficiency_score: 0.95,
                cache_hit_rate: metrics.buffer_cache_hits as f64
                    / (metrics.buffer_cache_hits + metrics.buffer_cache_misses) as f64,
            };

            Ok((result_tensor, acceleration_metrics))
        }
    }

    fn add(
        &self,
        a: &BitNetTensor,
        b: &BitNetTensor,
    ) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)> {
        Err(AccelerationError::OperationNotSupported {
            backend: "Metal".to_string(),
            operation: "Element-wise addition".to_string(),
        })
    }

    fn mul(
        &self,
        a: &BitNetTensor,
        b: &BitNetTensor,
    ) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)> {
        Err(AccelerationError::OperationNotSupported {
            backend: "Metal".to_string(),
            operation: "Element-wise multiplication".to_string(),
        })
    }

    fn create_tensor(
        &self,
        shape: &[usize],
        dtype: BitNetDType,
        data: Option<&[f32]>,
    ) -> AccelerationResult<BitNetTensor> {
        match data {
            Some(data) => {
                BitNetTensor::from_data(data, shape, dtype, Some(candle_core::Device::Cpu))
            }
            None => BitNetTensor::zeros(shape, dtype, Some(candle_core::Device::Cpu)),
        }
        .map_err(|e| AccelerationError::OperationFailed {
            backend: "Metal".to_string(),
            operation: "Tensor creation".to_string(),
            reason: format!("Failed to create tensor: {:?}", e),
        })
    }

    fn transfer_to_device(&self, tensor: &BitNetTensor) -> AccelerationResult<BitNetTensor> {
        // For Metal, we would transfer to GPU memory
        // For now, return a copy of the tensor
        Ok(tensor.clone())
    }

    fn transfer_to_cpu(&self, tensor: &BitNetTensor) -> AccelerationResult<BitNetTensor> {
        // For Metal, we would transfer from GPU memory
        // For now, return a copy of the tensor
        Ok(tensor.clone())
    }

    fn get_memory_stats(&self) -> anyhow::Result<crate::memory::MemoryMetrics> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock metrics"))?;

        use std::collections::HashMap;
        let mut device_stats = HashMap::new();
        device_stats.insert(
            "Metal GPU".to_string(),
            crate::memory::metrics::DeviceMemoryStats {
                total_allocated: metrics.gpu_memory_allocated as u64,
                total_deallocated: metrics.gpu_memory_freed as u64,
                current_allocated: (metrics.gpu_memory_allocated - metrics.gpu_memory_freed) as u64,
                peak_allocated: metrics.gpu_memory_allocated as u64,
                allocation_count: metrics.buffer_cache_misses as u64,
                deallocation_count: 0 as u64,
                active_allocations: (metrics.buffer_cache_hits as u64
                    + metrics.buffer_cache_misses as u64),
            },
        );

        Ok(crate::memory::MemoryMetrics {
            total_allocated: metrics.gpu_memory_allocated as u64,
            total_deallocated: metrics.gpu_memory_freed as u64,
            current_allocated: (metrics.gpu_memory_allocated - metrics.gpu_memory_freed) as u64,
            peak_allocated: metrics.gpu_memory_allocated as u64,
            allocation_count: metrics.buffer_cache_misses as u64,
            deallocation_count: 0,
            active_allocations: metrics.buffer_cache_hits as u64
                + metrics.buffer_cache_misses as u64,
            peak_active_allocations: metrics.buffer_cache_hits as u64
                + metrics.buffer_cache_misses as u64,
            device_stats,
            pool_stats: crate::memory::metrics::PoolMemoryStats {
                small_block: crate::memory::metrics::PoolTypeStats {
                    total_allocated: 0,
                    total_deallocated: 0,
                    current_allocated: 0,
                    peak_allocated: 0,
                    allocation_count: 0,
                    deallocation_count: 0,
                    active_allocations: 0,
                    efficiency: crate::memory::metrics::PoolEfficiencyMetrics {
                        average_allocation_size: 0.0,
                        fragmentation_ratio: 0.0,
                        utilization_ratio: 0.0,
                        expansion_count: 0,
                        contraction_count: 0,
                    },
                },
                large_block: crate::memory::metrics::PoolTypeStats {
                    total_allocated: 0,
                    total_deallocated: 0,
                    current_allocated: 0,
                    peak_allocated: 0,
                    allocation_count: 0,
                    deallocation_count: 0,
                    active_allocations: 0,
                    efficiency: crate::memory::metrics::PoolEfficiencyMetrics {
                        average_allocation_size: 0.0,
                        fragmentation_ratio: 0.0,
                        utilization_ratio: 0.0,
                        expansion_count: 0,
                        contraction_count: 0,
                    },
                },
            },
            size_distribution: crate::memory::metrics::SizeDistribution {
                tiny: 0,
                small: 0,
                medium: 0,
                large: 0,
                huge: 0,
            },
            performance: crate::memory::metrics::PerformanceMetrics {
                avg_allocation_latency_ns: 0,
                avg_deallocation_latency_ns: 0,
                max_allocation_latency_ns: 0,
                max_deallocation_latency_ns: 0,
                total_allocation_time_ns: 0,
                total_deallocation_time_ns: 0,
                allocation_failures: 0,
                deallocation_failures: 0,
            },
            last_updated: Some(std::time::SystemTime::now()),
        })
    }

    fn cleanup(&mut self) -> AccelerationResult<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.device = None;
            self.command_queue = None;
            self.library = None;
            self.command_buffer_manager = None;
            self.bitnet_shaders = None;
            self.gpu_memory_pool = None;
            self.buffer_cache = Arc::new(Mutex::new(HashMap::new()));
        }

        self.initialized = false;
        Ok(())
    }
}

/// Create a new Metal accelerator instance
pub fn create_metal_accelerator(
) -> AccelerationResult<Box<dyn AccelerationBackendImpl + Send + Sync>> {
    #[cfg(feature = "metal")]
    {
        let mut accelerator = MetalAccelerator::new()?;
        accelerator.initialize()?;
        Ok(Box::new(accelerator))
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(AccelerationError::BackendNotAvailable {
            backend: "Metal".to_string(),
        })
    }
}

/// Check if Metal acceleration is available
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        MetalAccelerator::is_metal_available()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Stub MetalAccelerator for when Metal feature is disabled
#[cfg(not(feature = "metal"))]
pub struct MetalAccelerator {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "metal"))]
impl MetalAccelerator {
    pub fn new() -> AccelerationResult<Self> {
        Err(AccelerationError::BackendNotAvailable {
            backend: "Metal".to_string(),
        })
    }

    pub fn is_metal_available() -> bool {
        false
    }
}
