//! Metal GPU Kernel Integration for BitNet Tensors
//!
//! This module provides seamless integration between BitNet tensor operations
//! and Metal GPU compute shaders, enabling high-performance GPU acceleration
//! for quantization, BitLinear operations, and matrix computations.

use std::sync::Arc;
use metal::{Device, CommandQueue, ComputePipelineState, Buffer, MTLResourceOptions};
use crate::tensor::core::BitNetTensor;
use crate::tensor::dtype::BitNetDType;
use super::{TensorOpResult, TensorOpError};

#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn, info};

/// BitNet Metal kernel manager
///
/// Manages Metal compute pipelines and provides high-level interface
/// for GPU-accelerated BitNet operations.
pub struct BitNetMetalKernels {
    device: Device,
    command_queue: CommandQueue,
    
    // Quantization kernels
    quantization_158_pipeline: ComputePipelineState,
    dequantization_158_pipeline: ComputePipelineState,
    adaptive_quantization_pipeline: ComputePipelineState,
    
    // BitLinear kernels
    bitlinear_forward_pipeline: ComputePipelineState,
    bitlinear_activation_quant_pipeline: ComputePipelineState,
    
    // Matrix operation kernels
    matmul_optimized_pipeline: ComputePipelineState,
}

impl BitNetMetalKernels {
    /// Create new BitNet Metal kernel manager
    ///
    /// # Arguments
    /// * `device` - Metal device to use for computations
    /// * `command_queue` - Metal command queue for dispatching kernels
    ///
    /// # Returns
    /// * Initialized kernel manager with all pipelines compiled
    pub fn new(device: Device, command_queue: CommandQueue) -> TensorOpResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Initializing BitNet Metal kernels");

        // Load and compile shader library
        let library = Self::load_bitnet_shader_library(&device)?;
        
        // Create compute pipelines for all kernels
        let quantization_158_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_158_quantize"
        )?;
        
        let dequantization_158_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_158_dequantize"
        )?;
        
        let adaptive_quantization_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_adaptive_quantize"
        )?;
        
        let bitlinear_forward_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitlinear_forward"
        )?;
        
        let bitlinear_activation_quant_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitlinear_activation_quant"
        )?;
        
        let matmul_optimized_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_matmul_optimized"
        )?;

        #[cfg(feature = "tracing")]
        info!("Successfully compiled {} Metal compute pipelines", 6);

        Ok(Self {
            device,
            command_queue,
            quantization_158_pipeline,
            dequantization_158_pipeline,
            adaptive_quantization_pipeline,
            bitlinear_forward_pipeline,
            bitlinear_activation_quant_pipeline,
            matmul_optimized_pipeline,
        })
    }

    /// Quantize tensor using BitNet 1.58-bit quantization on GPU
    ///
    /// # Arguments
    /// * `input` - Input tensor to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point
    ///
    /// # Returns
    /// * Quantized tensor with values in {-1, 0, 1}
    pub fn quantize_158(&self, 
        input: &BitNetTensor, 
        scale: f32, 
        zero_point: f32
    ) -> TensorOpResult<BitNetTensor> {
        #[cfg(feature = "tracing")]
        debug!("GPU quantizing tensor: {:?}", input.shape().dims());

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.quantization_158_pipeline);
        
        // Create buffers
        let input_buffer = self.tensor_to_metal_buffer(input)?;
        let output_buffer = self.create_output_buffer(input.num_elements(), std::mem::size_of::<i8>())?;
        let scale_buffer = self.create_scalar_buffer(&[scale])?;
        let zero_point_buffer = self.create_scalar_buffer(&[zero_point])?;
        let size_buffer = self.create_scalar_buffer(&[input.num_elements() as u32])?;
        
        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&scale_buffer), 0);
        encoder.set_buffer(3, Some(&zero_point_buffer), 0);
        encoder.set_buffer(4, Some(&size_buffer), 0);
        
        // Dispatch threads
        let (threadgroup_size, threadgroups) = self.calculate_dispatch_size(input.num_elements());
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Convert result back to BitNetTensor
        self.metal_buffer_to_tensor(&output_buffer, input.shape(), BitNetDType::I8)
    }

    /// Dequantize tensor using BitNet 1.58-bit dequantization on GPU
    ///
    /// # Arguments
    /// * `input` - Quantized input tensor
    /// * `scale` - Dequantization scale factor
    /// * `zero_point` - Dequantization zero point
    ///
    /// # Returns
    /// * Dequantized floating-point tensor
    pub fn dequantize_158(&self,
        input: &BitNetTensor,
        scale: f32,
        zero_point: f32
    ) -> TensorOpResult<BitNetTensor> {
        #[cfg(feature = "tracing")]
        debug!("GPU dequantizing tensor: {:?}", input.shape().dims());

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.dequantization_158_pipeline);
        
        // Create buffers
        let input_buffer = self.tensor_to_metal_buffer(input)?;
        let output_buffer = self.create_output_buffer(input.num_elements(), std::mem::size_of::<f32>())?;
        let scale_buffer = self.create_scalar_buffer(&[scale])?;
        let zero_point_buffer = self.create_scalar_buffer(&[zero_point])?;
        let size_buffer = self.create_scalar_buffer(&[input.num_elements() as u32])?;
        
        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&scale_buffer), 0);
        encoder.set_buffer(3, Some(&zero_point_buffer), 0);
        encoder.set_buffer(4, Some(&size_buffer), 0);
        
        // Dispatch threads
        let (threadgroup_size, threadgroups) = self.calculate_dispatch_size(input.num_elements());
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Convert result back to BitNetTensor
        self.metal_buffer_to_tensor(&output_buffer, input.shape(), BitNetDType::F32)
    }

    /// Perform BitLinear forward pass on GPU
    ///
    /// # Arguments
    /// * `weights` - Quantized weight tensor
    /// * `input` - Input activation tensor
    /// * `weight_scale` - Weight scaling factor
    /// * `input_scale` - Input scaling factor
    ///
    /// # Returns
    /// * Output tensor from BitLinear computation
    pub fn bitlinear_forward(&self,
        weights: &BitNetTensor,
        input: &BitNetTensor,
        weight_scale: f32,
        input_scale: f32
    ) -> TensorOpResult<BitNetTensor> {
        let weight_dims = weights.shape().dims();
        let input_dims = input.shape().dims();
        
        if weight_dims.len() != 2 || input_dims.len() != 2 {
            return Err(TensorOpError::ShapeMismatch {
                expected: vec![2, 2],
                actual: vec![weight_dims.len(), input_dims.len()],
                operation: "bitlinear_forward".to_string(),
            });
        }

        let (output_size, input_size) = (weight_dims[0], weight_dims[1]);
        let batch_size = input_dims[0];

        #[cfg(feature = "tracing")]
        debug!("GPU BitLinear forward: weights {:?}, input {:?}", weight_dims, input_dims);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.bitlinear_forward_pipeline);
        
        // Create buffers
        let weights_buffer = self.tensor_to_metal_buffer(weights)?;
        let input_buffer = self.tensor_to_metal_buffer(input)?;
        let output_buffer = self.create_output_buffer(batch_size * output_size, std::mem::size_of::<f32>())?;
        let weight_scale_buffer = self.create_scalar_buffer(&[weight_scale])?;
        let input_scale_buffer = self.create_scalar_buffer(&[input_scale])?;
        let input_size_buffer = self.create_scalar_buffer(&[input_size as u32])?;
        let output_size_buffer = self.create_scalar_buffer(&[output_size as u32])?;
        
        // Set buffers
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&weight_scale_buffer), 0);
        encoder.set_buffer(4, Some(&input_scale_buffer), 0);
        encoder.set_buffer(5, Some(&input_size_buffer), 0);
        encoder.set_buffer(6, Some(&output_size_buffer), 0);
        
        // Dispatch threads (2D grid for output elements)
        let threadgroup_size = metal::MTLSize::new(16, 16, 1);
        let threadgroups = metal::MTLSize::new(
            (output_size + 15) / 16,
            (batch_size + 15) / 16,
            1
        );
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Convert result back to BitNetTensor
        self.metal_buffer_to_tensor(&output_buffer, &[batch_size, output_size], BitNetDType::F32)
    }

    /// Perform optimized matrix multiplication on GPU
    ///
    /// # Arguments
    /// * `a` - Left matrix tensor
    /// * `b` - Right matrix tensor
    ///
    /// # Returns
    /// * Result of matrix multiplication
    pub fn matmul_optimized(&self, a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        if a_dims.len() != 2 || b_dims.len() != 2 {
            return Err(TensorOpError::ShapeMismatch {
                expected: vec![2, 2],
                actual: vec![a_dims.len(), b_dims.len()],
                operation: "matmul_optimized".to_string(),
            });
        }

        if a_dims[1] != b_dims[0] {
            return Err(TensorOpError::ShapeMismatch {
                expected: vec![a_dims[1], a_dims[1]],
                actual: vec![a_dims[1], b_dims[0]],
                operation: "matmul_optimized".to_string(),
            });
        }

        let (M, K, N) = (a_dims[0], a_dims[1], b_dims[1]);

        #[cfg(feature = "tracing")]
        debug!("GPU optimized matmul: {}×{} × {}×{}", M, K, K, N);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.matmul_optimized_pipeline);
        
        // Create buffers
        let a_buffer = self.tensor_to_metal_buffer(a)?;
        let b_buffer = self.tensor_to_metal_buffer(b)?;
        let output_buffer = self.create_output_buffer(M * N, std::mem::size_of::<f32>())?;
        let m_buffer = self.create_scalar_buffer(&[M as u32])?;
        let n_buffer = self.create_scalar_buffer(&[N as u32])?;
        let k_buffer = self.create_scalar_buffer(&[K as u32])?;
        
        // Set buffers
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&m_buffer), 0);
        encoder.set_buffer(4, Some(&n_buffer), 0);
        encoder.set_buffer(5, Some(&k_buffer), 0);
        
        // Dispatch threads with optimal tiling
        let threadgroup_size = metal::MTLSize::new(16, 16, 1);
        let threadgroups = metal::MTLSize::new(
            (N + 15) / 16,
            (M + 15) / 16,
            1
        );
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Convert result back to BitNetTensor
        self.metal_buffer_to_tensor(&output_buffer, &[M, N], BitNetDType::F32)
    }

    /// Check if GPU acceleration should be used for given tensor operation
    ///
    /// # Arguments
    /// * `tensor_size` - Total number of elements in tensor
    /// * `operation` - Type of operation being performed
    ///
    /// # Returns
    /// * True if GPU acceleration should be used
    pub fn should_use_gpu(&self, tensor_size: usize, operation: &str) -> bool {
        // GPU acceleration thresholds based on operation type
        let threshold = match operation {
            "quantization" | "dequantization" => 1024,      // Small operations benefit from GPU parallelism
            "bitlinear_forward" => 512,                     // Linear layers benefit from GPU even for small sizes
            "matmul" => 2048,                               // Matrix multiplication needs larger sizes to overcome overhead
            "element_wise" => 4096,                         // Element-wise operations need large tensors
            _ => 2048,                                      // Conservative default
        };

        tensor_size >= threshold
    }

    /// Automatically dispatch tensor operation to GPU or CPU
    ///
    /// # Arguments
    /// * `operation` - Operation name for dispatching logic
    /// * `tensor_size` - Total elements in primary tensor
    /// * `gpu_fn` - GPU operation closure
    /// * `cpu_fn` - CPU fallback operation closure
    ///
    /// # Returns
    /// * Result of the operation
    pub fn auto_dispatch<T, F, G>(&self, 
        operation: &str, 
        tensor_size: usize, 
        gpu_fn: F, 
        cpu_fn: G
    ) -> TensorOpResult<T>
    where
        F: FnOnce() -> TensorOpResult<T>,
        G: FnOnce() -> TensorOpResult<T>,
    {
        if self.should_use_gpu(tensor_size, operation) {
            #[cfg(feature = "tracing")]
            trace!("Dispatching {} (size: {}) to GPU", operation, tensor_size);
            
            match gpu_fn() {
                Ok(result) => Ok(result),
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    warn!("GPU {} failed, falling back to CPU: {:?}", operation, e);
                    cpu_fn()
                }
            }
        } else {
            #[cfg(feature = "tracing")]
            trace!("Dispatching {} (size: {}) to CPU", operation, tensor_size);
            cpu_fn()
        }
    }

    // Private helper methods
    
    /// Calculate optimal dispatch size for GPU kernels
    fn calculate_dispatch_size(&self, element_count: usize) -> (metal::MTLSize, metal::MTLSize) {
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(
            ((element_count + 255) / 256) as u64,
            1,
            1
        );
        (threadgroup_size, threadgroups)
    }

    /// Convert BitNet tensor to Metal buffer
    fn tensor_to_metal_buffer(&self, tensor: &BitNetTensor) -> TensorOpResult<metal::Buffer> {
        let element_count = tensor.num_elements();
        let element_size = tensor.dtype().size_bytes().unwrap_or(4);
        let buffer_size = element_count * element_size;

        let buffer = self.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared
        );

        // Copy tensor data to GPU buffer
        let tensor_data = tensor.as_slice_f32().map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to get tensor data: {:?}", e),
        })?;

        unsafe {
            let buffer_ptr = buffer.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(tensor_data.as_ptr(), buffer_ptr, element_count);
        }

        Ok(buffer)
    }

    /// Convert Metal buffer back to BitNet tensor
    fn metal_buffer_to_tensor(
        &self, 
        buffer: &metal::Buffer, 
        shape: &[usize], 
        dtype: BitNetDType
    ) -> TensorOpResult<BitNetTensor> {
        let element_count: usize = shape.iter().product();
        
        // Read buffer data
        let buffer_data = unsafe {
            std::slice::from_raw_parts(
                buffer.contents() as *const f32,
                element_count,
            )
        };

        // Create new tensor with the data
        BitNetTensor::from_data(
            buffer_data, 
            shape, 
            dtype, 
            Some(candle_core::Device::Cpu)
        ).map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create tensor from GPU buffer: {:?}", e),
        })
    }

    /// Create output buffer for GPU operations
    fn create_output_buffer(&self, element_count: usize, element_size: usize) -> TensorOpResult<metal::Buffer> {
        let buffer_size = element_count * element_size;
        Ok(self.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared
        ))
    }

    /// Create scalar buffer for single values
    fn create_scalar_buffer<T>(&self, data: &[T]) -> TensorOpResult<metal::Buffer> {
        let buffer_size = data.len() * std::mem::size_of::<T>();
        let buffer = self.device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared
        );

        unsafe {
            let buffer_ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer_ptr, data.len());
        }

        Ok(buffer)
    }

    /// Load BitNet shader library
    fn load_bitnet_shader_library(device: &metal::Device) -> TensorOpResult<metal::Library> {
        // Try to load from embedded shaders first
        let shader_source = include_str!("../../../../bitnet-metal/shaders/bitnet_quantization.metal");
        let bitlinear_source = include_str!("../../../../bitnet-metal/shaders/bitlinear_operations.metal");
        
        let combined_source = format!("{}\n\n{}", shader_source, bitlinear_source);
        
        let compile_options = metal::CompileOptions::new();
        device.new_library_with_source(&combined_source, &compile_options)
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to compile BitNet shader library: {:?}", e),
            })
    }

    /// Create compute pipeline from library function
    fn create_compute_pipeline(
        device: &metal::Device,
        library: &metal::Library,
        function_name: &str,
    ) -> TensorOpResult<metal::ComputePipelineState> {
        let function = library.get_function(function_name, None)
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to get function '{}': {:?}", function_name, e),
            })?;

        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to create pipeline for '{}': {:?}", function_name, e),
            })
    }
}

/// Global Metal kernel manager for automatic dispatch
pub struct GlobalMetalKernels {
    metal_kernels: Option<BitNetMetalKernels>,
}

impl GlobalMetalKernels {
    /// Initialize global Metal kernels
    pub fn new() -> Self {
        Self {
            metal_kernels: Self::try_initialize_metal(),
        }
    }

    /// Get Metal kernels if available
    pub fn get_metal_kernels(&self) -> Option<&BitNetMetalKernels> {
        self.metal_kernels.as_ref()
    }

    /// Try to initialize Metal kernels
    fn try_initialize_metal() -> Option<BitNetMetalKernels> {
        if let Some(device) = metal::Device::system_default() {
            let command_queue = device.new_command_queue();
            BitNetMetalKernels::new(device, command_queue).ok()
        } else {
            None
        }
    }

    /// Automatically dispatch operation with Metal acceleration
    pub fn auto_dispatch<T, F, G>(
        &self,
        operation: &str,
        tensor_size: usize,
        gpu_fn: F,
        cpu_fn: G,
    ) -> TensorOpResult<T>
    where
        F: FnOnce(&BitNetMetalKernels) -> TensorOpResult<T>,
        G: FnOnce() -> TensorOpResult<T>,
    {
        if let Some(metal_kernels) = &self.metal_kernels {
            if metal_kernels.should_use_gpu(tensor_size, operation) {
                #[cfg(feature = "tracing")]
                trace!("Auto-dispatching {} (size: {}) to Metal GPU", operation, tensor_size);
                
                match gpu_fn(metal_kernels) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        #[cfg(feature = "tracing")]
                        warn!("Metal GPU {} failed, falling back to CPU: {:?}", operation, e);
                    }
                }
            }
        }

        #[cfg(feature = "tracing")]
        trace!("Auto-dispatching {} (size: {}) to CPU", operation, tensor_size);
        cpu_fn()
    }
}

impl Default for GlobalMetalKernels {
    fn default() -> Self {
        Self::new()
    }
}
