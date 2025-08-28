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
#[allow(dead_code)]
pub struct BitNetMetalKernels {
    device: Device,
    command_queue: CommandQueue,

    // Quantization kernels
    quantization_158_pipeline: ComputePipelineState,
    dequantization_158_pipeline: ComputePipelineState,
    adaptive_quantization_pipeline: ComputePipelineState,
    weight_quantization_pipeline: ComputePipelineState,
    activation_quantization_pipeline: ComputePipelineState,

    // BitLinear kernels
    bitlinear_forward_pipeline: ComputePipelineState,
    bitlinear_activation_quant_pipeline: ComputePipelineState,

    // Matrix operation kernels
    matmul_optimized_pipeline: ComputePipelineState,

    // QAT kernels
    qat_forward_pipeline: ComputePipelineState,
    ste_gradient_pipeline: ComputePipelineState,
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

        let weight_quantization_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_weight_quantize"
        )?;

        let activation_quantization_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_activation_quantize"
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

        let qat_forward_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_qat_forward"
        )?;

        let ste_gradient_pipeline = Self::create_compute_pipeline(
            &device, &library, "bitnet_ste_gradient"
        )?;

        #[cfg(feature = "tracing")]
        info!("Successfully compiled {} Metal compute pipelines", 10);

        Ok(Self {
            device,
            command_queue,
            quantization_158_pipeline,
            dequantization_158_pipeline,
            adaptive_quantization_pipeline,
            weight_quantization_pipeline,
            activation_quantization_pipeline,
            bitlinear_forward_pipeline,
            bitlinear_activation_quant_pipeline,
            matmul_optimized_pipeline,
            qat_forward_pipeline,
            ste_gradient_pipeline,
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
    /// # Arguments
    /// * `input` - Input tensor
    /// * `scale` - Quantization scale
    /// * `training_mode` - Whether in training mode (fake quantization) or inference mode
    ///
    /// # Returns
    /// * Tuple of (fake_quantized_output, true_quantized_output)
    pub fn qat_forward(&self,
        input: &BitNetTensor,
        scale: f32,
        training_mode: bool
    ) -> TensorOpResult<(BitNetTensor, BitNetTensor)> {
        #[cfg(feature = "tracing")]
        debug!("GPU QAT forward: training_mode={}", training_mode);

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.qat_forward_pipeline);

        // Create buffers
        let input_buffer = self.tensor_to_metal_buffer(input)?;
        let fake_quantized_buffer = self.create_output_buffer(input.num_elements(), std::mem::size_of::<f32>())?;
        let true_quantized_buffer = self.create_output_buffer(input.num_elements(), std::mem::size_of::<i8>())?;
        let scale_buffer = self.create_scalar_buffer(&[scale])?;
        let size_buffer = self.create_scalar_buffer(&[input.num_elements() as u32])?;
        let training_mode_buffer = self.create_scalar_buffer(&[if training_mode { 1u32 } else { 0u32 }])?;

        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&fake_quantized_buffer), 0);
        encoder.set_buffer(2, Some(&true_quantized_buffer), 0);
        encoder.set_buffer(3, Some(&scale_buffer), 0);
        encoder.set_buffer(4, Some(&size_buffer), 0);
        encoder.set_buffer(5, Some(&training_mode_buffer), 0);

        // Dispatch threads
        let (threadgroup_size, threadgroups) = self.calculate_dispatch_size(input.num_elements());
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert results back to BitNetTensors
        let fake_quantized = self.metal_buffer_to_tensor(&fake_quantized_buffer, input.shape(), BitNetDType::F32)?;
        let true_quantized = self.metal_buffer_to_tensor(&true_quantized_buffer, input.shape(), BitNetDType::I8)?;

        Ok((fake_quantized, true_quantized))
    }

    // ============================================================================
    // Helper Methods
    // ============================================================================

    /// Load BitNet shader library from embedded shaders
    fn load_bitnet_shader_library(device: &Device) -> TensorOpResult<metal::Library> {
        // In a real implementation, this would load the compiled shader library
        // For now, we'll create a placeholder that compiles from source
        let shader_source = include_str!("../../../bitnet-metal/shaders/bitnet_quantization.metal");

        device.new_library_with_source(shader_source, &metal::CompileOptions::new())
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to compile Metal shaders: {}", e),
            })
    }

    /// Create compute pipeline from function name
    fn create_compute_pipeline(
        device: &Device,
        library: &metal::Library,
        function_name: &str
    ) -> TensorOpResult<ComputePipelineState> {
        let function = library.get_function(function_name, None)
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to get Metal function '{}': {}", function_name, e),
            })?;

        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to create compute pipeline for '{}': {}", function_name, e),
            })
    }

    /// Convert BitNetTensor to Metal buffer
    fn tensor_to_metal_buffer(&self, tensor: &BitNetTensor) -> TensorOpResult<Buffer> {
        // This is a simplified implementation
        // In practice, this would handle different data types and memory layouts
        let data = tensor.as_slice_f32()
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to get tensor data: {}", e),
            })?;

        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared
        );

        Ok(buffer)
    }

    /// Convert Metal buffer back to BitNetTensor
    fn metal_buffer_to_tensor(
        &self,
        buffer: &Buffer,
        shape: &[usize],
        dtype: BitNetDType
    ) -> TensorOpResult<BitNetTensor> {
        // This is a simplified implementation
        // In practice, this would handle different data types properly
        match dtype {
            BitNetDType::F32 => {
                let ptr = buffer.contents() as *const f32;
                let len = shape.iter().product::<usize>();
                let data = unsafe { std::slice::from_raw_parts(ptr, len) };

                BitNetTensor::from_vec(
                    data.to_vec(),
                    shape,
                    dtype,
                    Some(self.device.clone())
                )
            }
            BitNetDType::I8 => {
                let ptr = buffer.contents() as *const i8;
                let len = shape.iter().product::<usize>();
                let data = unsafe { std::slice::from_raw_parts(ptr, len) };

                // Convert i8 to f32 for BitNetTensor
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();

                BitNetTensor::from_vec(
                    f32_data,
                    shape,
                    BitNetDType::F32, // Store as F32 internally
                    Some(self.device.clone())
                )
            }
            _ => Err(TensorOpError::InternalError {
                reason: format!("Unsupported dtype for Metal buffer conversion: {:?}", dtype),
            })
        }
    }

    /// Create output buffer for computation results
    fn create_output_buffer(&self, num_elements: usize, element_size: usize) -> TensorOpResult<Buffer> {
        let buffer_size = (num_elements * element_size) as u64;
        Ok(self.device.new_buffer(buffer_size, MTLResourceOptions::StorageModeShared))
    }

    /// Create buffer for scalar values
    fn create_scalar_buffer<T>(&self, data: &[T]) -> TensorOpResult<Buffer> {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared
        );
        Ok(buffer)
    }

    /// Calculate optimal dispatch size for 1D kernels
    fn calculate_dispatch_size(&self, num_elements: usize) -> (metal::MTLSize, metal::MTLSize) {
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        let threadgroups = metal::MTLSize::new(
            (num_elements + 255) / 256,
            1,
            1
        );
        (threadgroup_size, threadgroups)
    }
}

/// GPU acceleration dispatcher for BitNet operations
///
/// Automatically selects between CPU and GPU implementations based on
/// tensor size, device availability, and operation type.
#[allow(dead_code)]
pub struct BitNetGPUDispatcher {
    metal_kernels: Option<BitNetMetalKernels>,
    gpu_threshold: usize,
}

impl BitNetGPUDispatcher {
    /// Create new GPU dispatcher
    ///
    /// # Arguments
    /// * `gpu_threshold` - Minimum tensor size to use GPU acceleration
    ///
    /// # Returns
    /// * GPU dispatcher with Metal kernels if available
    pub fn new(gpu_threshold: usize) -> Self {
        let metal_kernels = Self::try_initialize_metal();

        #[cfg(feature = "tracing")]
        if metal_kernels.is_some() {
            info!("GPU acceleration available via Metal");
        } else {
            info!("GPU acceleration not available, using CPU fallback");
        }

        Self {
            metal_kernels,
            gpu_threshold,
        }
    }

    /// Dispatch quantization operation to optimal device
    pub fn quantize_158(&self,
        input: &BitNetTensor,
        scale: f32,
        zero_point: f32
    ) -> TensorOpResult<BitNetTensor> {
        if self.should_use_gpu(input) {
            if let Some(ref kernels) = self.metal_kernels {
                return kernels.quantize_158(input, scale, zero_point);
            }
        }

        // CPU fallback
        self.quantize_158_cpu(input, scale, zero_point)
    }

    /// Dispatch BitLinear forward operation to optimal device
    pub fn bitlinear_forward(&self,
        weights: &BitNetTensor,
        input: &BitNetTensor,
        weight_scale: f32,
        input_scale: f32
    ) -> TensorOpResult<BitNetTensor> {
        if self.should_use_gpu(input) && self.should_use_gpu(weights) {
            if let Some(ref kernels) = self.metal_kernels {
                return kernels.bitlinear_forward(weights, input, weight_scale, input_scale);
            }
        }

        // CPU fallback
        self.bitlinear_forward_cpu(weights, input, weight_scale, input_scale)
    }

    /// Dispatch matrix multiplication to optimal device
    pub fn matmul(&self, a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
        if self.should_use_gpu(a) && self.should_use_gpu(b) {
            if let Some(ref kernels) = self.metal_kernels {
                return kernels.matmul_optimized(a, b);
            }
        }

        // CPU fallback
        super::super::ops::linear_algebra::matmul(a, b)
    }

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    /// Try to initialize Metal kernels
    fn try_initialize_metal() -> Option<BitNetMetalKernels> {
        if let Some(device) = Device::system_default() {
            let command_queue = device.new_command_queue();
            BitNetMetalKernels::new(device, command_queue).ok()
        } else {
            None
        }
    }

    /// Determine if GPU should be used for this tensor
    fn should_use_gpu(&self, tensor: &BitNetTensor) -> bool {
        tensor.num_elements() >= self.gpu_threshold &&
        self.metal_kernels.is_some() &&
        matches!(tensor.device(), metal::Device::Metal(_))
    }

    /// CPU fallback for quantization
    fn quantize_158_cpu(&self, input: &BitNetTensor, scale: f32, zero_point: f32) -> TensorOpResult<BitNetTensor> {
        // Simplified CPU implementation
        let data = input.as_slice_f32()?;
        let quantized_data: Vec<f32> = data.iter().map(|&x| {
            let scaled = x / scale + zero_point;
            if scaled <= -0.5 { -1.0 }
            else if scaled >= 0.5 { 1.0 }
            else { 0.0 }
        }).collect();

        BitNetTensor::from_vec(quantized_data, input.shape().dims(), input.dtype(), Some(input.device().clone()))
    }

    /// CPU fallback for BitLinear forward
    fn bitlinear_forward_cpu(&self,
        weights: &BitNetTensor,
        input: &BitNetTensor,
        weight_scale: f32,
        input_scale: f32
    ) -> TensorOpResult<BitNetTensor> {
        // Simplified CPU implementation using standard matrix multiplication
        let result = super::super::ops::linear_algebra::matmul(input, &super::super::ops::linear_algebra::transpose(weights)?)?;

        // Apply scaling
        let data = result.as_slice_f32()?;
        let scaled_data: Vec<f32> = data.iter().map(|&x| x * weight_scale * input_scale).collect();

        BitNetTensor::from_vec(scaled_data, result.shape().dims(), result.dtype(), Some(result.device().clone()))
    }
}

/// Global GPU dispatcher instance
static mut GPU_DISPATCHER: Option<BitNetGPUDispatcher> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Get global GPU dispatcher instance
pub fn get_gpu_dispatcher() -> &'static BitNetGPUDispatcher {
    unsafe {
        INIT.call_once(|| {
            GPU_DISPATCHER = Some(BitNetGPUDispatcher::new(1024)); // Use GPU for tensors with >1024 elements
        });
        GPU_DISPATCHER.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_dispatcher_creation() {
        let dispatcher = BitNetGPUDispatcher::new(1000);
        // Should not panic and should handle missing Metal gracefully
    }

    #[test]
    fn test_gpu_threshold_logic() {
        let dispatcher = BitNetGPUDispatcher::new(1000);

        let small_tensor = BitNetTensor::ones(&[10, 10], BitNetDType::F32, None).unwrap();
        let large_tensor = BitNetTensor::ones(&[100, 100], BitNetDType::F32, None).unwrap();

        // These tests will use CPU fallback since Metal may not be available in test environment
        assert!(dispatcher.quantize_158(&small_tensor, 1.0, 0.0).is_ok());
        assert!(dispatcher.quantize_158(&large_tensor, 1.0, 0.0).is_ok());
    }
}
