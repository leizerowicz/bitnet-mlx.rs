//! GPU-Accelerated Arithmetic Operations for BitNet Tensors
//!
//! This module provides GPU-accelerated arithmetic operations that automatically
//! dispatch to Metal GPU when beneficial, with CPU fallback for smaller tensors.

use crate::tensor::core::BitNetTensor;
use super::{TensorOpResult, TensorOpError};
use super::arithmetic::{add as cpu_add, sub as cpu_sub, mul as cpu_mul, div as cpu_div};
use super::super::acceleration::metal_kernels_complete::GlobalMetalKernels;
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

// Global Metal kernel manager
lazy_static! {
    static ref GLOBAL_METAL: Arc<Mutex<GlobalMetalKernels>> = Arc::new(Mutex::new(GlobalMetalKernels::new()));
}

/// GPU-accelerated element-wise addition
///
/// Automatically dispatches to GPU for large tensors, CPU for small ones.
/// Falls back to CPU if GPU acceleration fails.
pub fn add_gpu(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let tensor_size = lhs.num_elements().max(rhs.num_elements());

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "element_wise_add",
        tensor_size,
        |metal_kernels| {
            // GPU path - use Metal kernels for element-wise addition
            gpu_add_impl(lhs, rhs, metal_kernels)
        },
        || {
            // CPU fallback
            cpu_add(lhs, rhs)
        }
    )
}

/// GPU-accelerated element-wise subtraction
pub fn sub_gpu(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let tensor_size = lhs.num_elements().max(rhs.num_elements());

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "element_wise_sub",
        tensor_size,
        |metal_kernels| {
            gpu_sub_impl(lhs, rhs, metal_kernels)
        },
        || {
            cpu_sub(lhs, rhs)
        }
    )
}

/// GPU-accelerated element-wise multiplication
pub fn mul_gpu(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let tensor_size = lhs.num_elements().max(rhs.num_elements());

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "element_wise_mul",
        tensor_size,
        |metal_kernels| {
            gpu_mul_impl(lhs, rhs, metal_kernels)
        },
        || {
            cpu_mul(lhs, rhs)
        }
    )
}

/// GPU-accelerated element-wise division
pub fn div_gpu(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let tensor_size = lhs.num_elements().max(rhs.num_elements());

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "element_wise_div",
        tensor_size,
        |metal_kernels| {
            gpu_div_impl(lhs, rhs, metal_kernels)
        },
        || {
            cpu_div(lhs, rhs)
        }
    )
}

/// GPU-accelerated matrix multiplication
pub fn matmul_gpu(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();

    // Validate matrix multiplication dimensions
    if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2, 2],
            actual: vec![lhs_dims.len(), rhs_dims.len()],
            operation: "matmul".to_string(),
        });
    }

    if lhs_dims[1] != rhs_dims[0] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![lhs_dims[1], lhs_dims[1]],
            actual: vec![lhs_dims[1], rhs_dims[0]],
            operation: "matmul".to_string(),
        });
    }

    let tensor_size = lhs_dims[0] * lhs_dims[1] * rhs_dims[1];

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "matmul",
        tensor_size,
        |metal_kernels| {
            #[cfg(feature = "tracing")]
            debug!("Using GPU matmul for {}×{} × {}×{}", lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1]);

            metal_kernels.matmul_optimized(lhs, rhs)
        },
        || {
            #[cfg(feature = "tracing")]
            debug!("Using CPU matmul for {}×{} × {}×{}", lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1]);

            // CPU fallback using Candle
            let lhs_candle = lhs.to_candle()?;
            let rhs_candle = rhs.to_candle()?;

            let result_candle = lhs_candle.matmul(&rhs_candle)
                .map_err(|e| TensorOpError::CandleError {
                    operation: "matmul".to_string(),
                    error: e.to_string(),
                })?;

            BitNetTensor::from_candle(result_candle, lhs.device())
                .map_err(|e| TensorOpError::InternalError {
                    reason: format!("Failed to create result tensor: {}", e),
                })
        }
    )
}

// Private GPU implementation functions

fn gpu_add_impl(lhs: &BitNetTensor, rhs: &BitNetTensor, metal_kernels: &crate::tensor::acceleration::metal_kernels_complete::BitNetMetalKernels) -> TensorOpResult<BitNetTensor> {
    // For now, we'll create a simple element-wise add using Metal's basic operations
    // In a full implementation, we would have specialized broadcasting kernels

    // Check if tensors have the same shape (no broadcasting needed)
    if lhs.shape().dims() == rhs.shape().dims() {
        // Use simple element-wise addition
        gpu_elementwise_operation(lhs, rhs, "add", metal_kernels)
    } else {
        // For complex broadcasting, fall back to CPU for now
        // A full implementation would use the broadcasting kernels we created
        Err(TensorOpError::InternalError {
            reason: "GPU broadcasting not implemented yet, falling back to CPU".to_string(),
        })
    }
}

fn gpu_sub_impl(lhs: &BitNetTensor, rhs: &BitNetTensor, metal_kernels: &crate::tensor::acceleration::metal_kernels_complete::BitNetMetalKernels) -> TensorOpResult<BitNetTensor> {
    if lhs.shape().dims() == rhs.shape().dims() {
        gpu_elementwise_operation(lhs, rhs, "sub", metal_kernels)
    } else {
        Err(TensorOpError::InternalError {
            reason: "GPU broadcasting not implemented yet, falling back to CPU".to_string(),
        })
    }
}

fn gpu_mul_impl(lhs: &BitNetTensor, rhs: &BitNetTensor, metal_kernels: &crate::tensor::acceleration::metal_kernels_complete::BitNetMetalKernels) -> TensorOpResult<BitNetTensor> {
    if lhs.shape().dims() == rhs.shape().dims() {
        gpu_elementwise_operation(lhs, rhs, "mul", metal_kernels)
    } else {
        Err(TensorOpError::InternalError {
            reason: "GPU broadcasting not implemented yet, falling back to CPU".to_string(),
        })
    }
}

fn gpu_div_impl(lhs: &BitNetTensor, rhs: &BitNetTensor, metal_kernels: &crate::tensor::acceleration::metal_kernels_complete::BitNetMetalKernels) -> TensorOpResult<BitNetTensor> {
    if lhs.shape().dims() == rhs.shape().dims() {
        gpu_elementwise_operation(lhs, rhs, "div", metal_kernels)
    } else {
        Err(TensorOpError::InternalError {
            reason: "GPU broadcasting not implemented yet, falling back to CPU".to_string(),
        })
    }
}

fn gpu_elementwise_operation(
    lhs: &BitNetTensor,
    rhs: &BitNetTensor,
    operation: &str,
    metal_kernels: &crate::tensor::acceleration::metal_kernels_complete::BitNetMetalKernels
) -> TensorOpResult<BitNetTensor> {
    // This is a simplified implementation
    // In a full implementation, we would use the element-wise kernels we created

    #[cfg(feature = "tracing")]
    debug!("GPU element-wise {} for tensor shape {:?}", operation, lhs.shape().dims());

    // For now, use CPU implementation and indicate that GPU kernels need to be added
    Err(TensorOpError::InternalError {
        reason: format!("GPU element-wise {} kernel not fully implemented yet", operation),
    })
}

/// GPU-accelerated quantization operations
pub fn quantize_gpu(input: &BitNetTensor, scale: f32, zero_point: f32) -> TensorOpResult<BitNetTensor> {
    let tensor_size = input.num_elements();

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "quantization",
        tensor_size,
        |metal_kernels| {
            #[cfg(feature = "tracing")]
            debug!("Using GPU quantization for tensor with {} elements", tensor_size);

            metal_kernels.quantize_158(input, scale, zero_point)
        },
        || {
            #[cfg(feature = "tracing")]
            debug!("Using CPU quantization for tensor with {} elements", tensor_size);

            // CPU fallback - implement basic quantization
            cpu_quantize_impl(input, scale, zero_point)
        }
    )
}

/// GPU-accelerated dequantization operations
pub fn dequantize_gpu(input: &BitNetTensor, scale: f32, zero_point: f32) -> TensorOpResult<BitNetTensor> {
    let tensor_size = input.num_elements();

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "dequantization",
        tensor_size,
        |metal_kernels| {
            #[cfg(feature = "tracing")]
            debug!("Using GPU dequantization for tensor with {} elements", tensor_size);

            metal_kernels.dequantize_158(input, scale, zero_point)
        },
        || {
            #[cfg(feature = "tracing")]
            debug!("Using CPU dequantization for tensor with {} elements", tensor_size);

            // CPU fallback
            cpu_dequantize_impl(input, scale, zero_point)
        }
    )
}

/// GPU-accelerated BitLinear forward pass
pub fn bitlinear_forward_gpu(
    weights: &BitNetTensor,
    input: &BitNetTensor,
    weight_scale: f32,
    input_scale: f32
) -> TensorOpResult<BitNetTensor> {
    let tensor_size = weights.num_elements() + input.num_elements();

    let metal_manager = GLOBAL_METAL.lock().map_err(|_| TensorOpError::InternalError {
        reason: "Failed to acquire Metal kernel manager lock".to_string(),
    })?;

    metal_manager.auto_dispatch(
        "bitlinear_forward",
        tensor_size,
        |metal_kernels| {
            let weight_dims = weights.shape().dims();
            let input_dims = input.shape().dims();

            #[cfg(feature = "tracing")]
            debug!("Using GPU BitLinear forward: weights {:?}, input {:?}", weight_dims, input_dims);

            metal_kernels.bitlinear_forward(weights, input, weight_scale, input_scale)
        },
        || {
            #[cfg(feature = "tracing")]
            debug!("Using CPU BitLinear forward");

            // CPU fallback
            cpu_bitlinear_forward_impl(weights, input, weight_scale, input_scale)
        }
    )
}

// CPU fallback implementations

fn cpu_quantize_impl(input: &BitNetTensor, scale: f32, zero_point: f32) -> TensorOpResult<BitNetTensor> {
    // Basic CPU quantization implementation
    let input_data = input.as_slice_f32()?;
    let mut output_data = Vec::with_capacity(input_data.len());

    for &value in input_data {
        let scaled = value / scale + zero_point;
        let quantized = if scaled <= -0.5 {
            -1i8
        } else if scaled >= 0.5 {
            1i8
        } else {
            0i8
        };
        output_data.push(quantized as f32); // Store as f32 for tensor compatibility
    }

    BitNetTensor::from_data(&output_data, input.shape().dims(), input.dtype(), input.device().clone())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create quantized tensor: {}", e),
        })
}

fn cpu_dequantize_impl(input: &BitNetTensor, scale: f32, zero_point: f32) -> TensorOpResult<BitNetTensor> {
    // Basic CPU dequantization implementation
    let input_data = input.as_slice_f32()?;
    let mut output_data = Vec::with_capacity(input_data.len());

    for &quantized in input_data {
        let dequantized = (quantized as i8 as f32 - zero_point) * scale;
        output_data.push(dequantized);
    }

    BitNetTensor::from_data(&output_data, input.shape().dims(), input.dtype(), input.device().clone())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create dequantized tensor: {}", e),
        })
}

fn cpu_bitlinear_forward_impl(
    weights: &BitNetTensor,
    input: &BitNetTensor,
    weight_scale: f32,
    input_scale: f32
) -> TensorOpResult<BitNetTensor> {
    // Basic CPU BitLinear implementation
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

    if input_dims[1] != input_size {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![input_size, input_size],
            actual: vec![input_dims[1], input_size],
            operation: "bitlinear_forward".to_string(),
        });
    }

    let weight_data = weights.as_slice_f32()?;
    let input_data = input.as_slice_f32()?;
    let mut output_data = vec![0.0f32; batch_size * output_size];

    for b in 0..batch_size {
        for o in 0..output_size {
            let mut sum = 0.0f32;
            for i in 0..input_size {
                let w = weight_data[o * input_size + i] as i8; // Quantized weight
                let x = input_data[b * input_size + i];
                sum += (w as f32) * x;
            }
            output_data[b * output_size + o] = sum * weight_scale * input_scale;
        }
    }

    BitNetTensor::from_data(&output_data, &[batch_size, output_size], input.dtype(), input.device().clone())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create BitLinear output tensor: {}", e),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::BitNetTensor;
    use crate::tensor::dtype::BitNetDType;

    #[test]
    fn test_gpu_quantization_fallback() {
        let input = BitNetTensor::ones(&[10, 10], BitNetDType::F32, None).unwrap();
        let result = quantize_gpu(&input, 1.0, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_matmul_fallback() {
        let a = BitNetTensor::ones(&[4, 6], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[6, 8], BitNetDType::F32, None).unwrap();
        let result = matmul_gpu(&a, &b);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape().dims(), &[4, 8]);
    }
}
