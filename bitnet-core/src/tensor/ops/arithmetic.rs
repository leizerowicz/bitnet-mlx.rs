//! Arithmetic Operations for BitNet Tensors
//!
//! This module provides comprehensive arithmetic operations with
//! broadcasting support, SIMD optimization, and device awareness.
//!
//! # Features
//!
//! - Complete arithmetic operations: add, sub, mul, div, rem, pow
//! - In-place and out-of-place variants
//! - Scalar operations with broadcasting
//! - NumPy/PyTorch compatible broadcasting semantics
//! - SIMD optimization for performance
//! - Device-aware operations
//! - Memory-efficient implementations using existing HybridMemoryPool
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::{BitNetTensor, BitNetDType};
//! use bitnet_core::tensor::ops::arithmetic::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
//! let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
//!
//! // Element-wise addition
//! let result = add(&a, &b)?;
//!
//! // Scalar operations
//! let scaled = mul_scalar(&a, 2.5)?;
//!
//! # Ok(())
//! # }
//! ```

use super::broadcasting::{can_broadcast, prepare_elementwise_broadcast};
use super::{TensorOpError, TensorOpResult};
use crate::tensor::core::BitNetTensor;
use std::ops::{Add, Div, Mul, Rem, Sub};

#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

// ============================================================================
// Core Arithmetic Functions
// ============================================================================

/// Element-wise addition of two tensors with broadcasting support
pub fn add(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "add")?;

    // Check if broadcasting is needed
    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "add".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise addition: {:?} + {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    // Prepare tensors for broadcasting
    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;

    // Use the original tensors if broadcasting is not needed, otherwise use broadcasted ones
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);

    // Get underlying Candle tensors
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    // Perform addition using Candle
    let result_candle = lhs_candle
        .add(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "add".to_string(),
            error: e.to_string(),
        })?;

    // Convert back to BitNetTensor
    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Element-wise subtraction of two tensors with broadcasting support
pub fn sub(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "sub")?;

    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "sub".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise subtraction: {:?} - {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    let result_candle = lhs_candle
        .sub(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "sub".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Element-wise multiplication of two tensors with broadcasting support
pub fn mul(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "mul")?;

    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "mul".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise multiplication: {:?} * {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    let result_candle = lhs_candle
        .mul(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "mul".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Element-wise division of two tensors with broadcasting support
pub fn div(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "div")?;

    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "div".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise division: {:?} / {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    let result_candle = lhs_candle
        .div(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "div".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Element-wise remainder of two tensors with broadcasting support
pub fn rem(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "rem")?;

    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "rem".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise remainder: {:?} % {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    // Candle doesn't have remainder, so we implement it as: a - (a / b).floor() * b
    let div_result = (&lhs_candle)
        .div(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "rem_div".to_string(),
            error: e.to_string(),
        })?;

    let floor_result = div_result.floor().map_err(|e| TensorOpError::CandleError {
        operation: "rem_floor".to_string(),
        error: e.to_string(),
    })?;

    let mul_result = floor_result
        .mul(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "rem_mul".to_string(),
            error: e.to_string(),
        })?;

    let result_candle = lhs_candle
        .sub(&mul_result)
        .map_err(|e| TensorOpError::CandleError {
            operation: "rem_sub".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Element-wise power of two tensors with broadcasting support
pub fn pow(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_binary_operation(lhs, rhs, "pow")?;

    if !can_broadcast(lhs, rhs)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Tensors have incompatible shapes for broadcasting".to_string(),
            lhs_shape: lhs.shape().dims().to_vec(),
            rhs_shape: rhs.shape().dims().to_vec(),
            operation: "pow".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing element-wise power: {:?} ** {:?}",
        lhs.shape().dims(),
        rhs.shape().dims()
    );

    let (_broadcast_info, broadcast_a, broadcast_b) = prepare_elementwise_broadcast(lhs, rhs)?;
    let lhs_to_use = broadcast_a.as_ref().unwrap_or(lhs);
    let rhs_to_use = broadcast_b.as_ref().unwrap_or(rhs);
    let lhs_candle = lhs_to_use.to_candle()?;
    let rhs_candle = rhs_to_use.to_candle()?;

    // Use pow for tensor-tensor power operations
    let result_candle = lhs_candle
        .pow(&rhs_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "pow".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, lhs.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

// ============================================================================
// Scalar Operations
// ============================================================================

/// Add scalar to tensor
pub fn add_scalar(tensor: &BitNetTensor, scalar: f64) -> TensorOpResult<BitNetTensor> {
    validate_scalar_operation(tensor, "add_scalar")?;

    #[cfg(feature = "tracing")]
    trace!(
        "Performing scalar addition: {:?} + {}",
        tensor.shape().dims(),
        scalar
    );

    let tensor_candle = tensor.to_candle()?;
    let result_candle = (tensor_candle + scalar).map_err(|e| TensorOpError::CandleError {
        operation: "add_scalar".to_string(),
        error: e.to_string(),
    })?;

    BitNetTensor::from_candle(result_candle, tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Subtract scalar from tensor
pub fn sub_scalar(tensor: &BitNetTensor, scalar: f64) -> TensorOpResult<BitNetTensor> {
    validate_scalar_operation(tensor, "sub_scalar")?;

    #[cfg(feature = "tracing")]
    trace!(
        "Performing scalar subtraction: {:?} - {}",
        tensor.shape().dims(),
        scalar
    );

    let tensor_candle = tensor.to_candle()?;
    let result_candle = (tensor_candle - scalar).map_err(|e| TensorOpError::CandleError {
        operation: "sub_scalar".to_string(),
        error: e.to_string(),
    })?;

    BitNetTensor::from_candle(result_candle, tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Multiply tensor by scalar
pub fn mul_scalar(tensor: &BitNetTensor, scalar: f64) -> TensorOpResult<BitNetTensor> {
    validate_scalar_operation(tensor, "mul_scalar")?;

    #[cfg(feature = "tracing")]
    trace!(
        "Performing scalar multiplication: {:?} * {}",
        tensor.shape().dims(),
        scalar
    );

    let tensor_candle = tensor.to_candle()?;
    let result_candle = (tensor_candle * scalar).map_err(|e| TensorOpError::CandleError {
        operation: "mul_scalar".to_string(),
        error: e.to_string(),
    })?;

    BitNetTensor::from_candle(result_candle, tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

/// Divide tensor by scalar
pub fn div_scalar(tensor: &BitNetTensor, scalar: f64) -> TensorOpResult<BitNetTensor> {
    validate_scalar_operation(tensor, "div_scalar")?;

    if scalar == 0.0 {
        return Err(TensorOpError::NumericalError {
            operation: "div_scalar".to_string(),
            reason: "Division by zero".to_string(),
        });
    }

    #[cfg(feature = "tracing")]
    trace!(
        "Performing scalar division: {:?} / {}",
        tensor.shape().dims(),
        scalar
    );

    let tensor_candle = tensor.to_candle()?;
    let result_candle = (tensor_candle / scalar).map_err(|e| TensorOpError::CandleError {
        operation: "div_scalar".to_string(),
        error: e.to_string(),
    })?;

    BitNetTensor::from_candle(result_candle, tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create result tensor: {}", e),
        }
    })
}

// ============================================================================
// In-Place Operations
// ============================================================================

/// In-place element-wise addition
pub fn add_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = add(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

/// In-place element-wise subtraction
pub fn sub_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = sub(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

/// In-place element-wise multiplication
pub fn mul_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = mul(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

/// In-place element-wise division
pub fn div_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = div(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

/// In-place element-wise remainder
pub fn rem_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = rem(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

/// In-place element-wise power
pub fn pow_(tensor: &mut BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    let result = pow(tensor, rhs)?;
    *tensor = result;
    Ok(())
}

// ============================================================================
// In-Place Scalar Operations
// ============================================================================

/// In-place scalar addition
pub fn add_scalar_(tensor: &mut BitNetTensor, scalar: f64) -> TensorOpResult<()> {
    let result = add_scalar(tensor, scalar)?;
    *tensor = result;
    Ok(())
}

/// In-place scalar subtraction
pub fn sub_scalar_(tensor: &mut BitNetTensor, scalar: f64) -> TensorOpResult<()> {
    let result = sub_scalar(tensor, scalar)?;
    *tensor = result;
    Ok(())
}

/// In-place scalar multiplication
pub fn mul_scalar_(tensor: &mut BitNetTensor, scalar: f64) -> TensorOpResult<()> {
    let result = mul_scalar(tensor, scalar)?;
    *tensor = result;
    Ok(())
}

/// In-place scalar division
pub fn div_scalar_(tensor: &mut BitNetTensor, scalar: f64) -> TensorOpResult<()> {
    let result = div_scalar(tensor, scalar)?;
    *tensor = result;
    Ok(())
}

// ============================================================================
// Operator Overloads
// ============================================================================

impl Add for &BitNetTensor {
    type Output = TensorOpResult<BitNetTensor>;

    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs)
    }
}

impl Sub for &BitNetTensor {
    type Output = TensorOpResult<BitNetTensor>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs)
    }
}

impl Mul for &BitNetTensor {
    type Output = TensorOpResult<BitNetTensor>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs)
    }
}

impl Div for &BitNetTensor {
    type Output = TensorOpResult<BitNetTensor>;

    fn div(self, rhs: Self) -> Self::Output {
        div(self, rhs)
    }
}

impl Rem for &BitNetTensor {
    type Output = TensorOpResult<BitNetTensor>;

    fn rem(self, rhs: Self) -> Self::Output {
        rem(self, rhs)
    }
}

// ============================================================================
// Validation and Utility Functions
// ============================================================================

/// Validate inputs for binary operations
fn validate_binary_operation(
    lhs: &BitNetTensor,
    rhs: &BitNetTensor,
    operation: &str,
) -> TensorOpResult<()> {
    // Check if both tensors are valid
    if !lhs.is_valid() {
        return Err(TensorOpError::InvalidTensor {
            operation: operation.to_string(),
            reason: "Left tensor is invalid".to_string(),
        });
    }

    if !rhs.is_valid() {
        return Err(TensorOpError::InvalidTensor {
            operation: operation.to_string(),
            reason: "Right tensor is invalid".to_string(),
        });
    }

    // For now, allow cross-device operations (Candle will handle device migration)
    // In the future, we might want to enforce same-device operations for performance

    Ok(())
}

/// Validate inputs for scalar operations
fn validate_scalar_operation(tensor: &BitNetTensor, operation: &str) -> TensorOpResult<()> {
    if !tensor.is_valid() {
        return Err(TensorOpError::InvalidTensor {
            operation: operation.to_string(),
            reason: "Tensor is invalid".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
    use crate::tensor::core::BitNetTensor;
    use crate::tensor::dtype::BitNetDType;
    use std::sync::Arc;

    fn setup_memory_pool() -> Result<(), Box<dyn std::error::Error>> {
        let tracking_config = TrackingConfig::detailed();
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(tracking_config);

        let memory_pool = Arc::new(HybridMemoryPool::with_config(config)?);
        crate::tensor::memory_integration::set_global_memory_pool(Arc::downgrade(&memory_pool));

        // Verify that the global pool is set
        let retrieved_pool = crate::tensor::memory_integration::get_global_memory_pool();
        if retrieved_pool.is_none() {
            return Err("Failed to set global memory pool".into());
        }

        Ok(())
    }

    #[test]
    fn test_basic_addition() -> Result<(), Box<dyn std::error::Error>> {
        setup_memory_pool()?;

        // Create tensors using direct memory pool instead of global one
        let tracking_config = TrackingConfig::detailed();
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(tracking_config);

        let memory_pool = Arc::new(HybridMemoryPool::with_config(config)?);
        let device = crate::device::get_cpu_device();

        let a = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, Some(device))?;

        let result = add(&a, &b)?;
        assert_eq!(result.shape().dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_scalar_multiplication() -> Result<(), Box<dyn std::error::Error>> {
        setup_memory_pool()?;

        let tracking_config = TrackingConfig::detailed();
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(tracking_config);

        let memory_pool = Arc::new(HybridMemoryPool::with_config(config)?);
        let device = crate::device::get_cpu_device();

        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, Some(device))?;

        let result = mul_scalar(&a, 2.5)?;
        assert_eq!(result.shape().dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_division_by_zero() -> Result<(), Box<dyn std::error::Error>> {
        setup_memory_pool()?;

        let tracking_config = TrackingConfig::detailed();
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = true;
        config.tracking_config = Some(tracking_config);

        let memory_pool = Arc::new(HybridMemoryPool::with_config(config)?);
        let device = crate::device::get_cpu_device();

        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, Some(device))?;

        let result = div_scalar(&a, 0.0);
        assert!(result.is_err());

        if let Err(TensorOpError::NumericalError { reason, .. }) = result {
            assert_eq!(reason, "Division by zero");
        } else {
            panic!("Expected NumericalError");
        }
        Ok(())
    }
}
