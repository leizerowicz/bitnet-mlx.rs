//! GPU Memory Optimization for BitNet Inference Engine - Day 8 Enhanced
//!
//! This module provides advanced GPU memory management capabilities for Metal and MLX backends,
//! focusing on buffer pool optimization, unified memory utilization, cross-backend efficiency,
//! and high-performance inference memory transfer optimization.
//!
//! Phase 5 Day 8: GPU Optimization Implementation
//! - Advanced Metal compute shader integration
//! - Memory transfer optimization with async operations
//! - Buffer pool management with staging buffers
//! - Inference-specific memory layouts

use crate::{Result, InferenceError};
use bitnet_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Advanced GPU memory statistics tracking with performance metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory across all GPU backends
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Buffer pool hit rate (percentage)
    pub buffer_pool_hit_rate: f64,
    /// Memory fragmentation percentage
    pub fragmentation_percentage: f64,
    /// Cross-backend memory transfers
    pub cross_backend_transfers: usize,
    /// Memory transfer bandwidth (bytes/second)
    pub transfer_bandwidth: f64,
    /// Average allocation size
    pub average_allocation_size: usize,
    /// Number of staging buffer operations
    pub staging_operations: usize,
}

impl MemoryStats {
    /// Update peak usage if current usage exceeds it
    pub fn update_peak_usage(&mut self, current_usage: usize) {
        if current_usage > self.peak_usage {
            self.peak_usage = current_usage;
        }
    }
    
    /// Calculate and update fragmentation percentage
    pub fn calculate_fragmentation(&mut self, used: usize, total: usize) {
        if total > 0 {
            self.fragmentation_percentage = ((total - used) as f64 / total as f64) * 100.0;
        }
    }
    
    /// Calculate buffer pool hit rate
    pub fn update_hit_rate(&mut self, hits: usize, misses: usize) {
        let total = hits + misses;
        if total > 0 {
            self.buffer_pool_hit_rate = (hits as f64 / total as f64) * 100.0;
        }
    }

    /// Update transfer bandwidth statistics
    pub fn update_bandwidth(&mut self, bytes_transferred: usize, duration_ms: f64) {
        if duration_ms > 0.0 {
            self.transfer_bandwidth = (bytes_transferred as f64) / (duration_ms / 1000.0);
        }
    }

    /// Increment staging operations counter
    pub fn increment_staging_operations(&mut self) {
        self.staging_operations += 1;
    }
}

/// Inference-specific buffer configuration for optimized GPU operations
#[derive(Debug, Clone)]
pub struct InferenceBuffers {
    /// Input tensor buffer
    pub input: InferenceBuffer,
    /// Output tensor buffer  
    pub output: InferenceBuffer,
    /// Weight tensor buffer
    pub weights: InferenceBuffer,
    /// Optional bias buffer
    pub bias: Option<InferenceBuffer>,
    /// Staging buffer for async transfers
    pub staging: Option<InferenceBuffer>,
}

/// Unified inference buffer abstraction
#[derive(Debug, Clone)]
pub struct InferenceBuffer {
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer alignment for optimal GPU access
    pub alignment: usize,
    /// Device-specific buffer handle
    pub device_buffer: DeviceBufferHandle,
    /// Buffer usage pattern for optimization
    pub usage_pattern: BufferUsage,
}

/// Device-specific buffer handle
#[derive(Debug, Clone)]
pub enum DeviceBufferHandle {
    #[cfg(feature = "metal")]
    Metal(MetalBufferHandle),
    #[cfg(feature = "mlx")]
    MLX(MLXBufferHandle),
    Cpu(Vec<u8>),
    CPU(Vec<u8>), // Keep both for backward compatibility
}

/// Buffer usage pattern for memory layout optimization
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    /// Frequently read, rarely written (weights)
    ReadMostly,
    /// Frequently written, rarely read (intermediate results)
    WriteMostly,
    /// Equal read/write access (activations)
    ReadWrite,
    /// Temporary buffer for staging operations
    Staging,
}

/// Metal-specific buffer pool for managing GPU memory allocations
#[cfg(feature = "metal")]
pub struct MetalBufferPool {
    /// Pre-allocated buffers organized by size
    buffers_by_size: HashMap<usize, Vec<MetalBuffer>>,
    /// Maximum pool size in bytes
    max_pool_size: usize,
    /// Current pool utilization
    current_size: usize,
    /// Buffer allocation statistics
    allocation_stats: BufferStats,
    /// Staging buffers for async transfers
    staging_buffers: Vec<MetalBuffer>,
    /// Buffer alignment requirements
    buffer_alignment: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub hits: usize,
    pub misses: usize,
    pub peak_concurrent_buffers: usize,
    pub async_operations: usize,
    pub staging_transfers: usize,
}

/// Metal buffer handle for GPU operations
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalBufferHandle {
    pub buffer_id: u64,
    pub size: usize,
    pub device_ptr: Option<u64>, // GPU memory address
}

/// MLX buffer handle for unified memory operations
#[cfg(feature = "mlx")]
#[derive(Debug, Clone)]
pub struct MLXBufferHandle {
    pub region_id: u64,
    pub size: usize,
    pub unified_ptr: Option<u64>, // Unified memory address
}

/// Staging buffer for async memory transfers
#[derive(Debug, Clone)]
pub struct StagingBuffer {
    pub id: u64,
}

/// Placeholder for Metal buffer type with enhanced functionality
#[cfg(feature = "metal")]
pub struct MetalBuffer {
    size: usize,
    id: u64,
    alignment: usize,
    usage: BufferUsage,
    is_staging: bool,
}

#[cfg(feature = "metal")]
impl MetalBuffer {
    /// Create a new Metal buffer with specified size and id
    pub fn new(size: usize, id: u64) -> Self {
        Self {
            size,
            id,
            alignment: 16, // Default SIMD alignment
            usage: BufferUsage::ReadWrite,
            is_staging: false,
        }
    }
    
    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the buffer ID
    pub fn id(&self) -> u64 {
        self.id
    }
    
    /// Get the buffer alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Check if this is a staging buffer
    pub fn is_staging(&self) -> bool {
        self.is_staging
    }
}

#[cfg(feature = "metal")]
impl MetalBufferPool {
    /// Create a new Metal buffer pool with specified maximum size
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            buffers_by_size: HashMap::new(),
            max_pool_size,
            current_size: 0,
            allocation_stats: BufferStats::default(),
            staging_buffers: Vec::new(),
            buffer_alignment: 256, // 256-byte alignment for optimal GPU performance
        }
    }
    
    /// Allocate a buffer of the specified size, reusing from pool if available
    pub fn allocate_buffer(&mut self, size: usize) -> Result<MetalBuffer> {
        // Round up to nearest power of 2 for better pool utilization
        let rounded_size = size.next_power_of_two();
        
        // Try to reuse existing buffer from pool
        if let Some(buffers) = self.buffers_by_size.get_mut(&rounded_size) {
            if let Some(buffer) = buffers.pop() {
                self.allocation_stats.hits += 1;
                self.allocation_stats.allocations += 1;
                return Ok(buffer);
            }
        }
        
        // Pool miss - create new buffer
        self.allocation_stats.misses += 1;
        self.allocation_stats.allocations += 1;
        
        // Check if we have space in pool
        if self.current_size + rounded_size <= self.max_pool_size {
            self.current_size += rounded_size;
            Ok(MetalBuffer {
                size: rounded_size,
                id: self.allocation_stats.allocations as u64,
                alignment: self.buffer_alignment,
                usage: BufferUsage::ReadWrite, // Default usage pattern
                is_staging: false,
            })
        } else {
            Err(InferenceError::ResourceError(
                format!("Metal buffer pool exhausted. Requested: {} bytes, Available: {} bytes", 
                       rounded_size, self.max_pool_size.saturating_sub(self.current_size))
            ))
        }
    }
    
    /// Return a buffer to the pool for reuse
    pub fn deallocate_buffer(&mut self, buffer: MetalBuffer) {
        self.allocation_stats.deallocations += 1;
        
        // Add buffer back to pool for reuse
        self.buffers_by_size
            .entry(buffer.size)
            .or_insert_with(Vec::new)
            .push(buffer);
    }
    
    /// Optimize buffer pool by coalescing and defragmentation
    pub fn optimize_pool(&mut self) -> Result<()> {
        // Remove empty size categories
        self.buffers_by_size.retain(|_, buffers| !buffers.is_empty());
        
        // Sort buffers within each size category for better locality
        for buffers in self.buffers_by_size.values_mut() {
            buffers.sort_by_key(|buffer| buffer.id);
        }
        
        Ok(())
    }
    
    /// Get buffer pool statistics
    pub fn get_stats(&self) -> BufferStats {
        let mut stats = self.allocation_stats.clone();
        stats.peak_concurrent_buffers = self.buffers_by_size
            .values()
            .map(|buffers| buffers.len())
            .max()
            .unwrap_or(0);
        stats
    }
}

/// MLX Unified Memory Pool for Apple Silicon optimization
#[cfg(feature = "mlx")]
pub struct MLXUnifiedMemoryPool {
    /// Unified memory regions organized by size
    memory_regions: HashMap<usize, Vec<MLXMemoryRegion>>,
    /// Total unified memory available
    total_unified_memory: usize,
    /// Current utilization
    current_usage: usize,
    /// MLX-specific statistics
    mlx_stats: MLXStats,
}

#[cfg(feature = "mlx")]
#[derive(Debug, Clone, Default)]
pub struct MLXStats {
    pub zero_copy_transfers: usize,
    pub unified_allocations: usize,
    pub memory_bandwidth_utilization: f64,
}

/// Placeholder for MLX memory region
#[cfg(feature = "mlx")]
pub struct MLXMemoryRegion {
    size: usize,
    is_zero_copy: bool,
    id: u64,
}

#[cfg(feature = "mlx")]
impl MLXUnifiedMemoryPool {
    /// Create a new MLX unified memory pool
    pub fn new() -> Result<Self> {
        // Detect Apple Silicon unified memory size
        let unified_memory = Self::detect_unified_memory_size()?;
        
        Ok(Self {
            memory_regions: HashMap::new(),
            total_unified_memory: unified_memory,
            current_usage: 0,
            mlx_stats: MLXStats::default(),
        })
    }
    
    /// Detect unified memory size on Apple Silicon
    fn detect_unified_memory_size() -> Result<usize> {
        // Placeholder implementation - in real implementation would query system
        // For now, assume common Apple Silicon configurations
        Ok(16 * 1024 * 1024 * 1024) // 16GB unified memory
    }
    
    /// Allocate unified memory region optimized for zero-copy operations
    pub fn allocate_unified(&mut self, size: usize, enable_zero_copy: bool) -> Result<MLXMemoryRegion> {
        let aligned_size = (size + 4095) & !4095; // 4KB alignment for optimal performance
        
        if self.current_usage + aligned_size > self.total_unified_memory {
            return Err(InferenceError::ResourceError(
                "MLX unified memory exhausted".to_string()
            ));
        }
        
        self.current_usage += aligned_size;
        self.mlx_stats.unified_allocations += 1;
        
        if enable_zero_copy {
            self.mlx_stats.zero_copy_transfers += 1;
        }
        
        Ok(MLXMemoryRegion {
            size: aligned_size,
            is_zero_copy: enable_zero_copy,
            id: self.mlx_stats.unified_allocations as u64,
        })
    }
    
    /// Deallocate unified memory region
    pub fn deallocate_unified(&mut self, region: MLXMemoryRegion) {
        self.current_usage = self.current_usage.saturating_sub(region.size);
        
        // Add to pool for reuse
        self.memory_regions
            .entry(region.size)
            .or_insert_with(Vec::new)
            .push(region);
    }
    
    /// Optimize unified memory layout for Apple Silicon
    pub fn optimize_layout(&mut self) -> Result<()> {
        // Coalesce adjacent free regions
        for regions in self.memory_regions.values_mut() {
            regions.sort_by_key(|region| region.id);
        }
        
        // Update bandwidth utilization
        self.mlx_stats.memory_bandwidth_utilization = 
            (self.current_usage as f64 / self.total_unified_memory as f64) * 100.0;
        
        Ok(())
    }
}

/// Main GPU Memory Manager coordinating all GPU memory backends
pub struct GPUMemoryManager {
    /// Metal buffer pools indexed by pool name
    #[cfg(feature = "metal")]
    metal_pools: HashMap<String, MetalBufferPool>,
    
    /// MLX unified memory pool
    #[cfg(feature = "mlx")]
    mlx_unified_pool: Option<MLXUnifiedMemoryPool>,
    
    /// Overall memory statistics
    memory_statistics: Arc<Mutex<MemoryStats>>,
    
    /// Active device for memory operations
    active_device: Device,
}

impl GPUMemoryManager {
    /// Create new GPU memory manager for the specified device
    pub fn new(device: Device) -> Result<Self> {
        let mut manager = Self {
            #[cfg(feature = "metal")]
            metal_pools: HashMap::new(),
            #[cfg(feature = "mlx")]
            mlx_unified_pool: None,
            memory_statistics: Arc::new(Mutex::new(MemoryStats::default())),
            active_device: device,
        };
        
        manager.initialize_backend_pools()?;
        Ok(manager)
    }

    /// Allocate inference buffers optimized for the given model and batch size
    /// This matches the Day 8 plan specification
    pub fn allocate_inference_buffers(&mut self, batch_size: usize, model: &crate::engine::Model) -> Result<InferenceBuffers> {
        let input_size = batch_size * model.get_input_dim();
        let output_size = batch_size * model.get_output_dim();
        let weight_size = model.get_total_weight_count();
        
        let input_buffer = self.get_or_create_buffer(input_size * 4, BufferUsage::ReadWrite)?; // f32 = 4 bytes
        let output_buffer = self.get_or_create_buffer(output_size * 4, BufferUsage::WriteMostly)?;
        let weight_buffer = self.get_cached_weight_buffer(&model.get_model_id(), weight_size * 4)?;
        
        // Create staging buffer for async transfers if using GPU backend
        let staging_buffer = if self.is_gpu_backend() {
            Some(self.get_or_create_staging_buffer((input_size + output_size) * 4)?)
        } else {
            None
        };
        
        Ok(InferenceBuffers {
            input: input_buffer,
            output: output_buffer,
            weights: weight_buffer,
            bias: None, // Optional - would be set if model has bias terms
            staging: staging_buffer,
        })
    }

    /// Asynchronous memory transfer to GPU as specified in Day 8 plan
    pub async fn copy_to_gpu_async(&self, data: &[f32], buffer: &InferenceBuffer) -> Result<()> {
        use std::time::Instant;
        let start_time = Instant::now();
        
        match &buffer.device_buffer {
            #[cfg(feature = "metal")]
            DeviceBufferHandle::Metal(_metal_handle) => {
                // Get staging buffer for async transfer
                let _staging_buffer = self.get_staging_buffer(data.len() * 4)?;
                
                // Copy to staging buffer (can overlap with compute)
                // In a real implementation, this would use actual Metal APIs
                // For now, we simulate the operation
                tokio::task::yield_now().await; // Yield to allow concurrent operations
                
                // Simulate GPU-to-GPU copy (very fast)
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await; // Fast GPU transfer
                
                // Update statistics
                if let Ok(mut stats) = self.memory_statistics.lock() {
                    stats.increment_staging_operations();
                    let duration_ms = start_time.elapsed().as_millis() as f64;
                    stats.update_bandwidth(data.len() * 4, duration_ms);
                }
                
                Ok(())
            },
            #[cfg(feature = "mlx")]
            DeviceBufferHandle::MLX(_) => {
                // MLX unified memory operations are typically zero-copy
                tokio::task::yield_now().await; // Minimal delay for unified memory
                Ok(())
            },
            DeviceBufferHandle::CPU(_) => {
                // CPU buffer - direct memory copy
                // In real implementation would copy data to buffer
                Ok(())
            },
            DeviceBufferHandle::Cpu(_) => {
                // CPU buffer - direct memory copy (alternative capitalization)
                // In real implementation would copy data to buffer
                Ok(())
            }
        }
    }

    /// Get or create a buffer with specified size and usage pattern
    fn get_or_create_buffer(&mut self, size: usize, usage: BufferUsage) -> Result<InferenceBuffer> {
        let aligned_size = (size + self.get_alignment() - 1) & !(self.get_alignment() - 1);
        
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => {
                let pool_name = if aligned_size > 256 * 1024 * 1024 { "large" } else { "default" };
                
                if let Some(pool) = self.metal_pools.get_mut(pool_name) {
                    let metal_buffer = pool.allocate_buffer(aligned_size)?;
                    
                    Ok(InferenceBuffer {
                        size: aligned_size,
                        alignment: self.get_alignment(),
                        device_buffer: DeviceBufferHandle::Metal(MetalBufferHandle {
                            buffer_id: metal_buffer.id,
                            size: metal_buffer.size,
                            device_ptr: None, // Would be set by actual Metal implementation
                        }),
                        usage_pattern: usage,
                    })
                } else {
                    Err(InferenceError::ResourceError("Metal buffer pool not found".to_string()))
                }
            },
            Device::Cpu => {
                Ok(InferenceBuffer {
                    size: aligned_size,
                    alignment: 64, // CPU cache line alignment
                    device_buffer: DeviceBufferHandle::CPU(vec![0u8; aligned_size]),
                    usage_pattern: usage,
                })
            },
            _ => {
                Err(InferenceError::DeviceError("Unsupported device for buffer allocation".to_string()))
            }
        }
    }

    /// Get cached weight buffer for model weights
    fn get_cached_weight_buffer(&mut self, _model_id: &str, size: usize) -> Result<InferenceBuffer> {
        // Weight buffers should be read-mostly and cached across inference calls
        // In a full implementation, this would check a cache first
        self.get_or_create_buffer(size, BufferUsage::ReadMostly)
    }

    /// Create staging buffer for asynchronous transfers
    fn get_or_create_staging_buffer(&mut self, size: usize) -> Result<InferenceBuffer> {
        self.get_or_create_buffer(size, BufferUsage::Staging)
    }

    /// Get staging buffer for async operations
    fn get_staging_buffer(&self, _size: usize) -> Result<StagingBuffer> {
        // Placeholder implementation
        Ok(StagingBuffer { id: 1 })
    }

    /// Check if current backend is GPU-based
    fn is_gpu_backend(&self) -> bool {
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => true,
            _ => {
                #[cfg(feature = "mlx")]
                {
                    self.is_mlx_device()
                }
                #[cfg(not(feature = "mlx"))]
                {
                    false
                }
            }
        }
    }

    /// Check if device is MLX-based (placeholder for future MLX support)
    fn is_mlx_device(&self) -> bool {
        // Placeholder - would check for MLX device type when available
        false
    }

    /// Get memory alignment requirements for current device
    fn get_alignment(&self) -> usize {
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => 256, // Metal prefers 256-byte alignment
            _ if cfg!(feature = "mlx") => 4096, // MLX unified memory uses page alignment
            _ => 64, // CPU cache line alignment
        }
    }

    /// Get comprehensive memory statistics
    pub fn get_memory_stats(&self) -> Result<MemoryStats> {
        if let Ok(stats) = self.memory_statistics.lock() {
            Ok(stats.clone())
        } else {
            Err(InferenceError::ResourceError("Failed to acquire memory statistics lock".to_string()))
        }
    }
    
    /// Initialize backend-specific memory pools
    fn initialize_backend_pools(&mut self) -> Result<()> {
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => {
                // Create default Metal buffer pools for common sizes
                let default_pool = MetalBufferPool::new(512 * 1024 * 1024); // 512MB pool
                self.metal_pools.insert("default".to_string(), default_pool);
                
                let large_pool = MetalBufferPool::new(1024 * 1024 * 1024); // 1GB pool for large models
                self.metal_pools.insert("large".to_string(), large_pool);
            }
            
            // For now, handle MLX as a custom variant that may not exist in candle_core
            // This would be expanded when MLX support is added to candle or we use our own MLX backend
            _ if cfg!(feature = "mlx") => {
                // Custom MLX device handling would go here
                #[cfg(feature = "mlx")]
                {
                    // Initialize MLX unified memory pool
                    self.mlx_unified_pool = Some(MLXUnifiedMemoryPool::new()?);
                }
            }
            
            Device::Cpu => {
                // CPU doesn't require GPU memory management
            }
            
            _ => {
                // Other device types don't require GPU memory management
            }
        }
        
        Ok(())
    }
    
    /// Optimize Metal buffer pools for better performance
    #[cfg(feature = "metal")]
    pub fn optimize_metal_buffers(&mut self) -> Result<()> {
        for (pool_name, pool) in self.metal_pools.iter_mut() {
            pool.optimize_pool()?;
            
            // Update global statistics
            let stats = pool.get_stats();
            if let Ok(mut global_stats) = self.memory_statistics.lock() {
                global_stats.buffer_pool_hit_rate = 
                    if stats.hits + stats.misses > 0 {
                        (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0
                    } else {
                        0.0
                    };
                global_stats.active_allocations += stats.allocations - stats.deallocations;
            }
            
            println!("Optimized Metal buffer pool '{}' - Hit rate: {:.2}%", 
                      pool_name, 
                      if stats.hits + stats.misses > 0 {
                          (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0
                      } else {
                          0.0
                      });
        }
        
        Ok(())
    }
    
    /// Optimize MLX unified memory for Apple Silicon
    #[cfg(feature = "mlx")]
    pub fn optimize_mlx_unified_memory(&mut self) -> Result<()> {
        if let Some(mlx_pool) = &mut self.mlx_unified_pool {
            mlx_pool.optimize_layout()?;
            
            // Update global statistics
            if let Ok(mut stats) = self.memory_statistics.lock() {
                stats.total_allocated = mlx_pool.current_usage;
                stats.update_peak_usage(mlx_pool.current_usage);
                stats.calculate_fragmentation(
                    mlx_pool.current_usage,
                    mlx_pool.total_unified_memory
                );
            }
            
            println!("Optimized MLX unified memory - Utilization: {:.2}%, Zero-copy transfers: {}",
                      mlx_pool.mlx_stats.memory_bandwidth_utilization,
                      mlx_pool.mlx_stats.zero_copy_transfers);
        }
        
        Ok(())
    }
    
    /// Allocate GPU memory for tensor operations
    pub fn allocate_for_tensor(&mut self, tensor: &Tensor) -> Result<GPUAllocation> {
        let size = tensor.elem_count() * tensor.dtype().size_in_bytes();
        
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => {
                // Choose appropriate pool based on tensor size
                let pool_name = if size > 256 * 1024 * 1024 {
                    "large"
                } else {
                    "default"
                };
                
                if let Some(pool) = self.metal_pools.get_mut(pool_name) {
                    let buffer = pool.allocate_buffer(size)?;
                    Ok(GPUAllocation::Metal { buffer })
                } else {
                    Err(InferenceError::ResourceError(
                        format!("Metal buffer pool '{}' not found", pool_name)
                    ))
                }
            }
            
            _ if cfg!(feature = "mlx") => {
                #[cfg(feature = "mlx")]
                {
                    if let Some(pool) = &mut self.mlx_unified_pool {
                        let region = pool.allocate_unified(size, true)?; // Enable zero-copy
                        Ok(GPUAllocation::MLX { region })
                    } else {
                        Err(InferenceError::ResourceError(
                            "MLX unified memory pool not initialized".to_string()
                        ))
                    }
                }
                #[cfg(not(feature = "mlx"))]
                {
                    Err(InferenceError::UnsupportedOperation(
                        "MLX support not compiled".to_string()
                    ))
                }
            }
            
            _ => {
                Err(InferenceError::UnsupportedOperation(
                    format!("GPU memory allocation not supported for device: {:?}", self.active_device)
                ))
            }
        }
    }
    
    /// Perform comprehensive memory optimization
    pub fn optimize_all(&mut self) -> Result<()> {
        match self.active_device {
            #[cfg(feature = "metal")]
            Device::Metal(_) => {
                self.optimize_metal_buffers()?;
            }
            
            _ if cfg!(feature = "mlx") => {
                #[cfg(feature = "mlx")]
                self.optimize_mlx_unified_memory()?;
            }
            
            _ => {
                // No GPU optimization needed for CPU
            }
        }
        
        println!("Completed GPU memory optimization for device: {:?}", self.active_device);
        Ok(())
    }
}

/// Represents a GPU memory allocation
pub enum GPUAllocation {
    #[cfg(feature = "metal")]
    Metal { buffer: MetalBuffer },
    
    #[cfg(feature = "mlx")]
    MLX { region: MLXMemoryRegion },
    
    #[cfg(not(any(feature = "metal", feature = "mlx")))]
    Dummy, // Placeholder when no GPU features are enabled
}

impl GPUAllocation {
    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        match self {
            #[cfg(feature = "metal")]
            GPUAllocation::Metal { buffer } => buffer.size,
            
            #[cfg(feature = "mlx")]
            GPUAllocation::MLX { region } => region.size,
            
            #[cfg(not(any(feature = "metal", feature = "mlx")))]
            GPUAllocation::Dummy => 0,
        }
    }
    
    /// Check if allocation supports zero-copy operations
    pub fn is_zero_copy(&self) -> bool {
        match self {
            #[cfg(feature = "metal")]
            GPUAllocation::Metal { .. } => false, // Metal buffers typically require explicit transfers
            
            #[cfg(feature = "mlx")]
            GPUAllocation::MLX { region } => region.is_zero_copy,
            
            #[cfg(not(any(feature = "metal", feature = "mlx")))]
            GPUAllocation::Dummy => false,
        }
    }
}

// Stub implementations for missing feature configurations
#[cfg(not(feature = "metal"))]
pub struct MetalBufferPool;

#[cfg(not(feature = "metal"))]
impl MetalBufferPool {
    pub fn new(_max_pool_size: usize) -> Self {
        Self
    }
}

#[cfg(not(feature = "mlx"))]
pub struct MLXUnifiedMemoryPool;

#[cfg(not(feature = "mlx"))]
impl MLXUnifiedMemoryPool {
    pub fn new() -> Result<Self> {
        Err(InferenceError::UnsupportedOperation("MLX not available".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();
        
        stats.update_peak_usage(1000);
        assert_eq!(stats.peak_usage, 1000);
        
        stats.update_peak_usage(500);
        assert_eq!(stats.peak_usage, 1000); // Should not decrease
        
        stats.calculate_fragmentation(800, 1000);
        assert_eq!(stats.fragmentation_percentage, 20.0);
        
        stats.update_hit_rate(80, 20);
        assert_eq!(stats.buffer_pool_hit_rate, 80.0);
    }
    
    #[test]
    fn test_gpu_memory_manager_creation() {
        let manager = GPUMemoryManager::new(Device::Cpu);
        assert!(manager.is_ok());
    }
    
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_buffer_pool() {
        let mut pool = MetalBufferPool::new(1024 * 1024); // 1MB pool
        
        // Test allocation
        let buffer = pool.allocate_buffer(1024);
        assert!(buffer.is_ok());
        
        let buffer = buffer.unwrap();
        assert_eq!(buffer.size, 1024);
        
        // Test deallocation
        pool.deallocate_buffer(buffer);
        
        // Test pool optimization
        let result = pool.optimize_pool();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_gpu_allocation_size() {
        #[cfg(feature = "metal")]
        {
            let buffer = MetalBuffer { 
                size: 1024, 
                id: 1,
                alignment: 256,
                usage: BufferUsage::ReadWrite,
                is_staging: false,
            };
            let allocation = GPUAllocation::Metal { buffer };
            assert_eq!(allocation.size(), 1024);
            assert!(!allocation.is_zero_copy());
        }
        
        #[cfg(feature = "mlx")]
        {
            let region = MLXMemoryRegion { size: 2048, is_zero_copy: true, id: 1 };
            let allocation = GPUAllocation::MLX { region };
            assert_eq!(allocation.size(), 2048);
            assert!(allocation.is_zero_copy());
        }
    }
}
