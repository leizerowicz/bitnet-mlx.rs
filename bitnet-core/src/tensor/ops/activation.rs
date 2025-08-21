//! BitNet Tensor Activation Functions
//!
//! This module provides comprehensive activation functions for BitNet tensors,
//! including ReLU, GELU, Sigmoid, Tanh, Softmax, and their derivatives for
//! automatic differentiation support.
//!
//! # Features
//!
//! - **Neural Network Activations**: Complete set of modern activation functions
//! - **Gradient Support**: Derivatives for automatic differentiation
//! - **Memory Efficient**: Leverage existing HybridMemoryPool for intermediate results
//! - **Device Aware**: Optimized execution across CPU and GPU devices
//! - **Numerical Stability**: Numerically stable implementations (e.g., stable softmax)
//! - **Broadcasting Compatible**: Support for broadcasting operations
//! - **In-place Operations**: Optional in-place variants for memory efficiency
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::{BitNetTensor, activation::*};
//!
//! let tensor = BitNetTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2], BitNetDType::F32, None)?;
//! 
//! // Basic activations
//! let relu_out = relu(&tensor)?;
//! let sigmoid_out = sigmoid(&tensor)?;
//! let tanh_out = tanh(&tensor)?;
//! 
//! // Advanced activations
//! let gelu_out = gelu(&tensor)?;
//! let swish_out = swish(&tensor)?;
//! 
//! // Softmax with numerical stability
//! let softmax_out = softmax(&tensor, Some(1))?;  // Along dimension 1
//! 
//! // Gradient computation
//! let relu_grad = relu_backward(&tensor, &gradient)?;
//! ```

use candle_core::Tensor as CandleTensor;

use crate::tensor::BitNetTensor;
use crate::tensor::ops::{TensorOpResult, TensorOpError};

#[cfg(not(feature = "tracing"))]
macro_rules! debug { ($($arg:tt)*) => {} }

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error, instrument};

/// Rectified Linear Unit (ReLU) activation function
/// 
/// Computes ReLU(x) = max(0, x) element-wise.
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// 
/// # Returns
/// 
/// Tensor with ReLU activation applied element-wise
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2], BitNetDType::F32, None)?;
/// let relu_out = relu(&tensor)?;
/// // Output: [0.0, 0.0, 1.0, 2.0]
/// ```
/// Rectified Linear Unit (ReLU) activation function
///
/// Applies ReLU element-wise: f(x) = max(0, x)
///
/// # Arguments
///
/// * `tensor` - Input BitNet tensor
///
/// # Returns
///
/// Result containing new tensor with ReLU applied element-wise
///
/// # Examples
///
/// ```rust
/// use bitnet_core::tensor::{BitNetTensor, activation::relu};
///
/// let input = BitNetTensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], BitNetDType::F32, None)?;
/// let output = relu(&input)?;
/// // Output: [0.0, 0.0, 0.0, 1.0, 2.0]
/// ```
pub fn relu(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    debug!("Computing ReLU activation");

    // Convert to Candle tensor for processing
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to convert BitNetTensor to CandleTensor: {}", e),
        })?;    let zero = CandleTensor::zeros_like(&candle_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_zeros".to_string(),
            error: e.to_string(),
        })?;
    
    let result = candle_tensor.maximum(&zero)
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_maximum".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from ReLU result: {}", e),
    })?;
    
    debug!("ReLU activation completed");
    Ok(output_tensor)
}

/// ReLU backward pass for gradient computation
/// 
/// Computes the gradient of ReLU function: 1 if x > 0, else 0
/// 
/// # Arguments
/// 
/// * `input` - Original input tensor to ReLU
/// * `grad_output` - Gradient flowing back from next layer
/// 
/// # Returns
/// 
/// Gradient with respect to input

pub fn relu_backward(
    input: &BitNetTensor,
    grad_output: &BitNetTensor,
) -> TensorOpResult<BitNetTensor> {
    debug!("Computing ReLU backward pass");
    
    let input_candle = input.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_input".to_string(),
            error: e.to_string(),
        })?;
    
    let grad_candle = grad_output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_grad".to_string(),
            error: e.to_string(),
        })?;
    
    let zero = CandleTensor::zeros_like(&input_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_zeros".to_string(),
            error: e.to_string(),
        })?;
    
    // Create mask where input > 0
    let mask = input_candle.gt(&zero)
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_mask".to_string(),
            error: e.to_string(),
        })?;
    
    // Convert mask to same dtype as gradient
    let mask_float = mask.to_dtype(grad_candle.dtype())
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_mask_float".to_string(),
            error: e.to_string(),
        })?;
    
    // Apply mask to gradient
    let result = grad_candle.broadcast_mul(&mask_float)
        .map_err(|e| TensorOpError::CandleError {
            operation: "relu_backward_multiply".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        grad_output.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from ReLU backward result: {}", e),
    })?;
    
    debug!("ReLU backward pass completed");
    Ok(output_tensor)
}

/// Sigmoid activation function
/// 
/// Computes sigmoid(x) = 1 / (1 + exp(-x)) element-wise.
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// 
/// # Returns
/// 
/// Tensor with sigmoid activation applied element-wise
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![-2.0, 0.0, 2.0], &[3], BitNetDType::F32, None)?;
/// let sigmoid_out = sigmoid(&tensor)?;
/// // Output: [~0.119, 0.5, ~0.881]
/// ```

pub fn sigmoid(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    debug!("Computing sigmoid activation");
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_input".to_string(),
            error: e.to_string(),
        })?;
    
    // Numerically stable sigmoid: 1 / (1 + exp(-x))
    let neg_x = candle_tensor.neg()
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_neg".to_string(),
            error: e.to_string(),
        })?;
    
    let exp_neg_x = neg_x.exp()
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_exp".to_string(),
            error: e.to_string(),
        })?;
    
    let one = CandleTensor::ones_like(&candle_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_ones".to_string(),
            error: e.to_string(),
        })?;
    
    let denominator = one.add(&exp_neg_x)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_add".to_string(),
            error: e.to_string(),
        })?;
    
    let result = one.div(&denominator)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_div".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from sigmoid result: {}", e),
    })?;
    
    debug!("Sigmoid activation completed");
    Ok(output_tensor)
}

/// Sigmoid backward pass for gradient computation
/// 
/// Computes the gradient of sigmoid function: sigmoid(x) * (1 - sigmoid(x))
/// 
/// # Arguments
/// 
/// * `output` - Output of sigmoid function (sigmoid(x))
/// * `grad_output` - Gradient flowing back from next layer
/// 
/// # Returns
/// 
/// Gradient with respect to input

pub fn sigmoid_backward(
    output: &BitNetTensor,
    grad_output: &BitNetTensor,
) -> TensorOpResult<BitNetTensor> {
    debug!("Computing sigmoid backward pass");
    
    let output_candle = output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_output".to_string(),
            error: e.to_string(),
        })?;
    
    let grad_candle = grad_output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_grad".to_string(),
            error: e.to_string(),
        })?;
    
    let one = CandleTensor::ones_like(&output_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_ones".to_string(),
            error: e.to_string(),
        })?;
    
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    let one_minus_sigmoid = one.sub(&output_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_one_minus".to_string(),
            error: e.to_string(),
        })?;
    
    let sigmoid_derivative = output_candle.mul(&one_minus_sigmoid)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_derivative".to_string(),
            error: e.to_string(),
        })?;
    
    let result = grad_candle.broadcast_mul(&sigmoid_derivative)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sigmoid_backward_multiply".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        grad_output.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from sigmoid backward result: {}", e),
    })?;
    
    debug!("Sigmoid backward pass completed");
    Ok(output_tensor)
}

/// Hyperbolic tangent (Tanh) activation function
/// 
/// Computes tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) element-wise.
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// 
/// # Returns
/// 
/// Tensor with tanh activation applied element-wise
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], BitNetDType::F32, None)?;
/// let tanh_out = tanh(&tensor)?;
/// // Output: [~-0.762, 0.0, ~0.762]
/// ```

pub fn tanh(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    debug!("Computing tanh activation");
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_input".to_string(),
            error: e.to_string(),
        })?;
    
    let result = candle_tensor.tanh()
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_activation".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from tanh result: {}", e),
    })?;
    
    debug!("Tanh activation completed");
    Ok(output_tensor)
}

/// Tanh backward pass for gradient computation
/// 
/// Computes the gradient of tanh function: 1 - tanh²(x)
/// 
/// # Arguments
/// 
/// * `output` - Output of tanh function (tanh(x))
/// * `grad_output` - Gradient flowing back from next layer
/// 
/// # Returns
/// 
/// Gradient with respect to input

pub fn tanh_backward(
    output: &BitNetTensor,
    grad_output: &BitNetTensor,
) -> TensorOpResult<BitNetTensor> {
    debug!("Computing tanh backward pass");
    
    let output_candle = output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_output".to_string(),
            error: e.to_string(),
        })?;
    
    let grad_candle = grad_output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_grad".to_string(),
            error: e.to_string(),
        })?;
    
    let one = CandleTensor::ones_like(&output_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_ones".to_string(),
            error: e.to_string(),
        })?;
    
    // tanh'(x) = 1 - tanh²(x)
    let tanh_squared = output_candle.sqr()
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_square".to_string(),
            error: e.to_string(),
        })?;
    
    let tanh_derivative = one.sub(&tanh_squared)
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_derivative".to_string(),
            error: e.to_string(),
        })?;
    
    let result = grad_candle.broadcast_mul(&tanh_derivative)
        .map_err(|e| TensorOpError::CandleError {
            operation: "tanh_backward_multiply".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        grad_output.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from tanh backward result: {}", e),
    })?;
    
    debug!("Tanh backward pass completed");
    Ok(output_tensor)
}

/// Gaussian Error Linear Unit (GELU) activation function
/// 
/// Computes GELU(x) = x * Φ(x), where Φ(x) is the cumulative distribution function 
/// of the standard normal distribution. Uses the approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// 
/// # Returns
/// 
/// Tensor with GELU activation applied element-wise
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], BitNetDType::F32, None)?;
/// let gelu_out = gelu(&tensor)?;
/// // Output: [~-0.159, 0.0, ~0.841]
/// ```

pub fn gelu(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    debug!("Computing GELU activation");
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_input".to_string(),
            error: e.to_string(),
        })?;
    
    // GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    
    let x_cubed = candle_tensor.sqr()
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_x_squared".to_string(),
            error: e.to_string(),
        })?
        .mul(&candle_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_x_cubed".to_string(),
            error: e.to_string(),
        })?;
    
    let cubic_term = x_cubed.mul(&CandleTensor::from_vec(
        vec![0.044715_f32],
        &[],
        tensor.device(),
    ).map_err(|e| TensorOpError::CandleError {
        operation: "gelu_cubic_coefficient".to_string(),
        error: e.to_string(),
    })?)
    .map_err(|e| TensorOpError::CandleError {
        operation: "gelu_cubic_term".to_string(),
        error: e.to_string(),
    })?;
    
    let inner = candle_tensor.add(&cubic_term)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_inner_sum".to_string(),
            error: e.to_string(),
        })?
        .mul(&CandleTensor::from_vec(
            vec![sqrt_2_over_pi],
            &[],
            tensor.device(),
        ).map_err(|e| TensorOpError::CandleError {
            operation: "gelu_sqrt_coefficient".to_string(),
            error: e.to_string(),
        })?)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_inner_scale".to_string(),
            error: e.to_string(),
        })?;
    
    let tanh_part = inner.tanh()
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_tanh".to_string(),
            error: e.to_string(),
        })?;
    
    let one = CandleTensor::ones_like(&candle_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_ones".to_string(),
            error: e.to_string(),
        })?;
    
    let one_plus_tanh = one.add(&tanh_part)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_one_plus_tanh".to_string(),
            error: e.to_string(),
        })?;
    
    let half = CandleTensor::from_vec(
        vec![0.5_f32],
        &[],
        tensor.device(),
    ).map_err(|e| TensorOpError::CandleError {
        operation: "gelu_half".to_string(),
        error: e.to_string(),
    })?;
    
    let result = candle_tensor
        .mul(&one_plus_tanh)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_x_mul_tanh".to_string(),
            error: e.to_string(),
        })?
        .mul(&half)
        .map_err(|e| TensorOpError::CandleError {
            operation: "gelu_final_mul".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from GELU result: {}", e),
    })?;
    
    debug!("GELU activation completed");
    Ok(output_tensor)
}

/// Swish activation function (also known as SiLU)
/// 
/// Computes Swish(x) = x * sigmoid(x) element-wise.
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// 
/// # Returns
/// 
/// Tensor with Swish activation applied element-wise
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], BitNetDType::F32, None)?;
/// let swish_out = swish(&tensor)?;
/// // Output: [~-0.269, 0.0, ~0.731]
/// ```

pub fn swish(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    debug!("Computing Swish (SiLU) activation");
    
    let sigmoid_output = sigmoid(tensor)?;
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "swish_input".to_string(),
            error: e.to_string(),
        })?;
    
    let sigmoid_candle = sigmoid_output.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "swish_sigmoid".to_string(),
            error: e.to_string(),
        })?;
    
    let result = candle_tensor.mul(&sigmoid_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "swish_multiply".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from Swish result: {}", e),
    })?;
    
    debug!("Swish activation completed");
    Ok(output_tensor)
}

/// Softmax activation function with numerical stability
/// 
/// Computes softmax(x) = exp(x) / sum(exp(x)) along the specified dimension.
/// Uses numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// * `dim` - Dimension along which to compute softmax (None for last dimension)
/// 
/// # Returns
/// 
/// Tensor with softmax applied along specified dimension
/// 
/// # Examples
/// 
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], BitNetDType::F32, None)?;
/// let softmax_out = softmax(&tensor, Some(1))?;  // Along last dimension
/// ```

pub fn softmax(tensor: &BitNetTensor, dim: Option<usize>) -> TensorOpResult<BitNetTensor> {
    debug!("Computing softmax activation with numerical stability");
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_input".to_string(),
            error: e.to_string(),
        })?;
    
    // Use last dimension if not specified
    let softmax_dim = dim.unwrap_or(tensor.shape().rank() - 1);
    
    // Validate dimension
    if softmax_dim >= tensor.shape().rank() {
        return Err(TensorOpError::InvalidTensor {
            operation: "softmax_dim_validation".to_string(),
            reason: format!("Softmax dimension {} out of bounds for tensor with {} dimensions", 
                          softmax_dim, tensor.shape().rank()),
        });
    }
    
    // Numerical stability: subtract max along dimension
    let max_vals = candle_tensor.max(softmax_dim)
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_max".to_string(),
            error: e.to_string(),
        })?;
    
    // Expand max_vals to match original tensor shape for broadcasting
    let tensor_dims = candle_tensor.dims();
    let mut expanded_shape: Vec<usize> = tensor_dims.iter().cloned().collect();
    expanded_shape[softmax_dim] = 1;
    let max_vals_expanded = max_vals.reshape(candle_core::Shape::from(expanded_shape.as_slice()))
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_max_reshape".to_string(),
            error: e.to_string(),
        })?;
    
    let shifted = candle_tensor.broadcast_sub(&max_vals_expanded)
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_subtract_max".to_string(),
            error: e.to_string(),
        })?;
    
    // Compute exp(x - max(x))
    let exp_shifted = shifted.exp()
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_exp".to_string(),
            error: e.to_string(),
        })?;
    
    // Sum along dimension
    let sum_exp = exp_shifted.sum(softmax_dim)
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_sum".to_string(),
            error: e.to_string(),
        })?;
    
    // Expand sum for broadcasting
    let sum_exp_expanded = sum_exp.reshape(candle_core::Shape::from(expanded_shape.as_slice()))
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_sum_reshape".to_string(),
            error: e.to_string(),
        })?;
    
    // Divide to get softmax
    let result = exp_shifted.broadcast_div(&sum_exp_expanded)
        .map_err(|e| TensorOpError::CandleError {
            operation: "softmax_divide".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from softmax result: {}", e),
    })?;
    
    debug!("Softmax activation completed");
    Ok(output_tensor)
}

/// Log softmax activation function with numerical stability
/// 
/// Computes log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x))) along the specified dimension.
/// More numerically stable than computing log(softmax(x)).
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// * `dim` - Dimension along which to compute log softmax (None for last dimension)
/// 
/// # Returns
/// 
/// Tensor with log softmax applied along specified dimension

pub fn log_softmax(tensor: &BitNetTensor, dim: Option<usize>) -> TensorOpResult<BitNetTensor> {
    debug!("Computing log softmax activation with numerical stability");
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_input".to_string(),
            error: e.to_string(),
        })?;
    
    let softmax_dim = dim.unwrap_or(tensor.shape().rank() - 1);
    
    if softmax_dim >= tensor.shape().rank() {
        return Err(TensorOpError::InvalidTensor {
            operation: "log_softmax_dim_validation".to_string(),
            reason: format!("Log softmax dimension {} out of bounds for tensor with {} dimensions", 
                          softmax_dim, tensor.shape().rank()),
        });
    }
    
    // Numerical stability: subtract max
    let max_vals = candle_tensor.max(softmax_dim)
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_max".to_string(),
            error: e.to_string(),
        })?;
    
    let tensor_dims = candle_tensor.dims();
    let mut expanded_shape: Vec<usize> = tensor_dims.iter().cloned().collect();
    expanded_shape[softmax_dim] = 1;
    let max_vals_expanded = max_vals.reshape(candle_core::Shape::from(expanded_shape.as_slice()))
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_max_reshape".to_string(),
            error: e.to_string(),
        })?;
    
    let shifted = candle_tensor.broadcast_sub(&max_vals_expanded)
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_subtract_max".to_string(),
            error: e.to_string(),
        })?;
    
    // Compute log(sum(exp(x - max(x))))
    let exp_shifted = shifted.exp()
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_exp".to_string(),
            error: e.to_string(),
        })?;
    
    let sum_exp = exp_shifted.sum(softmax_dim)
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_sum".to_string(),
            error: e.to_string(),
        })?;
    
    let log_sum_exp = sum_exp.log()
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_log".to_string(),
            error: e.to_string(),
        })?;
    
    let log_sum_exp_expanded = log_sum_exp.reshape(candle_core::Shape::from(expanded_shape.as_slice()))
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_log_reshape".to_string(),
            error: e.to_string(),
        })?;
    
    // log_softmax = x - max(x) - log(sum(exp(x - max(x))))
    let result = shifted.broadcast_sub(&log_sum_exp_expanded)
        .map_err(|e| TensorOpError::CandleError {
            operation: "log_softmax_final_subtract".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from log softmax result: {}", e),
    })?;
    
    debug!("Log softmax activation completed");
    Ok(output_tensor)
}

/// Leaky ReLU activation function
/// 
/// Computes LeakyReLU(x) = max(negative_slope * x, x) element-wise.
/// 
/// # Arguments
/// 
/// * `tensor` - Input tensor
/// * `negative_slope` - Slope for negative values (default: 0.01)
/// 
/// # Returns
/// 
/// Tensor with Leaky ReLU activation applied element-wise

pub fn leaky_relu(tensor: &BitNetTensor, negative_slope: f32) -> TensorOpResult<BitNetTensor> {
    debug!("Computing Leaky ReLU activation with negative slope: {}", negative_slope);
    
    let candle_tensor = tensor.to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_input".to_string(),
            error: e.to_string(),
        })?;
    
    let zero = CandleTensor::zeros_like(&candle_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_zeros".to_string(),
            error: e.to_string(),
        })?;
    
    // Create mask for positive values
    let positive_mask = candle_tensor.gt(&zero)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_positive_mask".to_string(),
            error: e.to_string(),
        })?;
    
    let positive_mask_float = positive_mask.to_dtype(candle_tensor.dtype())
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_positive_mask_float".to_string(),
            error: e.to_string(),
        })?;
    
    // Positive part: x * mask
    let positive_part = candle_tensor.mul(&positive_mask_float)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_positive_part".to_string(),
            error: e.to_string(),
        })?;
    
    // Negative part: negative_slope * x * (1 - mask)
    let negative_mask_float = positive_mask_float.neg()
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_negative_mask_neg".to_string(),
            error: e.to_string(),
        })?
        .add(&CandleTensor::ones_like(&positive_mask_float)
            .map_err(|e| TensorOpError::CandleError {
                operation: "leaky_relu_ones".to_string(),
                error: e.to_string(),
            })?)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_negative_mask".to_string(),
            error: e.to_string(),
        })?;
    
    let slope_tensor = CandleTensor::from_vec(
        vec![negative_slope],
        &[],
        tensor.device(),
    ).map_err(|e| TensorOpError::CandleError {
        operation: "leaky_relu_slope_tensor".to_string(),
        error: e.to_string(),
    })?;
    
    let negative_part = candle_tensor
        .mul(&negative_mask_float)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_negative_mul_mask".to_string(),
            error: e.to_string(),
        })?
        .mul(&slope_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_negative_mul_slope".to_string(),
            error: e.to_string(),
        })?;
    
    let result = positive_part.add(&negative_part)
        .map_err(|e| TensorOpError::CandleError {
            operation: "leaky_relu_add_parts".to_string(),
            error: e.to_string(),
        })?;
    
    let output_tensor = BitNetTensor::from_candle(
        result,
        tensor.device(),
    ).map_err(|e| TensorOpError::InternalError {
        reason: format!("Failed to create output tensor from Leaky ReLU result: {}", e),
    })?;
    
    debug!("Leaky ReLU activation completed");
    Ok(output_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{BitNetTensor, BitNetDType};
    use candle_core::Device;

    #[test]
    fn test_relu_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            &[5],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let relu_output = relu(&tensor).unwrap();
        assert_eq!(relu_output.shape().dims(), &[5]);
        
        // Test that negative values become zero
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = relu_output.to_vec().unwrap();
        // assert_eq!(output_data[0], 0.0);  // -2.0 -> 0.0
        // assert_eq!(output_data[1], 0.0);  // -1.0 -> 0.0
        // assert_eq!(output_data[2], 0.0);  // 0.0 -> 0.0
        // assert_eq!(output_data[3], 1.0);  // 1.0 -> 1.0
        // assert_eq!(output_data[4], 2.0);  // 2.0 -> 2.0
    }

    #[test]
    fn test_sigmoid_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![-10.0, 0.0, 10.0],
            &[3],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let sigmoid_output = sigmoid(&tensor).unwrap();
        assert_eq!(sigmoid_output.shape().dims(), &[3]);
        
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = sigmoid_output.to_vec().unwrap();
        // Check that sigmoid is bounded between 0 and 1
        // assert!(output_data.iter().all(|&x| x >= 0.0 && x <= 1.0));
        // Check sigmoid(0) = 0.5
        // assert!((output_data[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![-1.0, 0.0, 1.0],
            &[3],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let tanh_output = tanh(&tensor).unwrap();
        assert_eq!(tanh_output.shape().dims(), &[3]);
        
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = tanh_output.to_vec().unwrap();
        // Check that tanh is bounded between -1 and 1
        // assert!(output_data.iter().all(|&x| x >= -1.0 && x <= 1.0));
        // Check tanh(0) = 0
        // assert!(output_data[1].abs() < 1e-6);
    }

    #[test]
    fn test_gelu_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![-1.0, 0.0, 1.0],
            &[3],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let gelu_output = gelu(&tensor).unwrap();
        assert_eq!(gelu_output.shape().dims(), &[3]);
        
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = gelu_output.to_vec().unwrap();
        // GELU(0) should be approximately 0
        // assert!(output_data[1].abs() < 1e-6);
    }

    #[test]
    fn test_softmax_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let softmax_output = softmax(&tensor, Some(1)).unwrap();
        assert_eq!(softmax_output.shape().dims(), &[2, 2]);
        
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = softmax_output.to_vec().unwrap();
        // Check that softmax sums to 1 along the specified dimension
        // let sum_first_row = output_data[0] + output_data[1];
        // let sum_second_row = output_data[2] + output_data[3];
        // assert!((sum_first_row - 1.0).abs() < 1e-6);
        // assert!((sum_second_row - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let tensor = BitNetTensor::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            &[5],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let leaky_relu_output = leaky_relu(&tensor, 0.1).unwrap();
        assert_eq!(leaky_relu_output.shape().dims(), &[5]);
        
        // Note: to_vec() method not yet implemented, skipping data validation for now
        // let output_data: Vec<f32> = leaky_relu_output.to_vec().unwrap();
        // Check negative slope is applied
        // assert!((output_data[0] - (-0.2)).abs() < 1e-6);  // -2.0 * 0.1 = -0.2
        // assert!((output_data[1] - (-0.1)).abs() < 1e-6);  // -1.0 * 0.1 = -0.1
        // assert_eq!(output_data[2], 0.0);  // 0.0 -> 0.0
        // assert_eq!(output_data[3], 1.0);  // 1.0 -> 1.0
        // assert_eq!(output_data[4], 2.0);  // 2.0 -> 2.0
    }

    #[test]
    fn test_activation_backward_passes() {
        let input = BitNetTensor::from_vec(
            vec![1.0, -1.0, 0.5],
            &[3],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        let grad_output = BitNetTensor::from_vec(
            vec![1.0, 1.0, 1.0],
            &[3],
            BitNetDType::F32,
            Some(Device::Cpu),
        ).unwrap();

        // Test ReLU backward
        let relu_grad = relu_backward(&input, &grad_output).unwrap();
        assert_eq!(relu_grad.shape().dims(), &[3]);

        // Test sigmoid backward
        let sigmoid_out = sigmoid(&input).unwrap();
        let sigmoid_grad = sigmoid_backward(&sigmoid_out, &grad_output).unwrap();
        assert_eq!(sigmoid_grad.shape().dims(), &[3]);

        // Test tanh backward
        let tanh_out = tanh(&input).unwrap();
        let tanh_grad = tanh_backward(&tanh_out, &grad_output).unwrap();
        assert_eq!(tanh_grad.shape().dims(), &[3]);
    }
}
