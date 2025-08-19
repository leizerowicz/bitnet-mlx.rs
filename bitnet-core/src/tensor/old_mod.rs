//! BitNet Tensor Operations Infrastructure
//!
//! This module provides a complete tensor operations system for BitNet,
//! built on top of the sophisticated HybridMemoryPool and device abstraction.
//! It integrates seamlessly with the existing production-ready infrastructure
//! to provide efficient, device-aware tensor operations.
//!
//! # Architecture
//!
//! The tensor system consists of several key components:
//!
//! - **BitNetTensor**: Core tensor struct with memory pool integration
//! - **TensorStorage**: Backend storage leveraging HybridMemoryPool
//! - **Shape Management**: Advanced broadcasting and dimension handling  
//! - **Device Integration**: Seamless CPU/Metal/MLX device support
//! - **Data Types**: Comprehensive type system for BitNet operations
//!
//! # Features
//!
//! - Memory-efficient tensor storage using existing HybridMemoryPool
//! - Thread-safe tensor operations with reference counting
//! - Device-aware tensor creation and migration
//! - MLX acceleration integration for Apple Silicon
//! - Metal compute shader support for GPU operations
//! - Advanced broadcasting compatible with NumPy/PyTorch
//! - Zero-copy operations where possible
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::BitNetTensor;
//! use bitnet_core::device::auto_select_device;
//! use bitnet_core::memory::HybridMemoryPool;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let pool = HybridMemoryPool::new()?;
//! let device = auto_select_device();
//! 
//! let tensor = BitNetTensor::zeros(&[2, 3], &device, &pool)?;
//! println!("Created tensor with shape: {:?}", tensor.shape());
//! # Ok(())
//! # }
//! ```

// Core tensor functionality
pub mod core;
pub mod storage;
pub mod shape;
pub mod dtype;
pub mod memory_integration;
pub mod device_integration;
pub mod lifecycle;
pub mod legacy;
pub mod memory_integration;
pub mod device_integration;

// Mathematical operations (will be implemented in Week 2)
pub mod ops;

// Acceleration integration (will be implemented in Week 3)
pub mod acceleration;

// Legacy tensor utilities (kept for backward compatibility)
mod legacy;

// Re-export core types
pub use core::{BitNetTensor, TensorError, TensorResult};
pub use storage::TensorStorage;
pub use shape::{TensorShape, BroadcastCompatible};
pub use dtype::BitNetDType;
pub use memory_integration::TensorMemoryManager;
pub use device_integration::TensorDeviceManager;

// Re-export legacy functions for backward compatibility
pub use legacy::{
    create_tensor_f32, create_tensor_i8, zeros, ones, 
    get_shape, reshape, transpose
};

// Import dependencies
use std::sync::Arc;
use candle_core::{Device, DType, Result, Tensor};
use crate::memory::{HybridMemoryPool, MemoryHandle};
use crate::device::auto_select_device;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Creates a tensor from f32 data with the specified shape
/// 
/// # Arguments
/// * `shape` - The desired shape of the tensor
/// * `data` - Vector of f32 values to populate the tensor
/// 
/// # Returns
/// A Result containing the created Tensor or an error
/// 
/// # Example
/// ```
/// use bitnet_core::tensor::create_tensor_f32;
/// 
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = create_tensor_f32(&[2, 2], data).unwrap();
/// ```
pub fn create_tensor_f32(shape: &[usize], data: Vec<f32>) -> Result<Tensor> {
    let device = Device::Cpu;
    Tensor::from_vec(data, shape, &device)
}

/// Creates a tensor from i8 data with the specified shape
///
/// # Arguments
/// * `shape` - The desired shape of the tensor
/// * `data` - Vector of i8 values to populate the tensor
///
/// # Returns
/// A Result containing the created Tensor or an error
///
/// # Example
/// ```
/// use bitnet_core::tensor::create_tensor_i8;
///
/// let data = vec![1i8, -1i8, 1i8, -1i8];
/// let tensor = create_tensor_i8(&[2, 2], data).unwrap();
/// ```
pub fn create_tensor_i8(shape: &[usize], data: Vec<i8>) -> Result<Tensor> {
    let device = Device::Cpu;
    // Convert i8 to i64 since candle doesn't support i8 directly
    let data_i64: Vec<i64> = data.into_iter().map(|x| x as i64).collect();
    Tensor::from_vec(data_i64, shape, &device)
}

/// Creates a tensor filled with zeros
/// 
/// # Arguments
/// * `shape` - The desired shape of the tensor
/// 
/// # Returns
/// A Result containing the created Tensor filled with zeros or an error
/// 
/// # Example
/// ```
/// use bitnet_core::tensor::zeros;
/// 
/// let tensor = zeros(&[3, 3]).unwrap();
/// ```
pub fn zeros(shape: &[usize]) -> Result<Tensor> {
    let device = Device::Cpu;
    Tensor::zeros(shape, DType::F32, &device)
}

/// Creates a tensor filled with ones
/// 
/// # Arguments
/// * `shape` - The desired shape of the tensor
/// 
/// # Returns
/// A Result containing the created Tensor filled with ones or an error
/// 
/// # Example
/// ```
/// use bitnet_core::tensor::ones;
/// 
/// let tensor = ones(&[2, 4]).unwrap();
/// ```
pub fn ones(shape: &[usize]) -> Result<Tensor> {
    let device = Device::Cpu;
    Tensor::ones(shape, DType::F32, &device)
}

/// Gets the shape of a tensor as a vector of dimensions
///
/// # Arguments
/// * `tensor` - Reference to the tensor to get the shape from
///
/// # Returns
/// A vector containing the dimensions of the tensor
///
/// # Example
/// ```
/// use bitnet_core::tensor::{create_tensor_f32, get_shape};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let tensor = create_tensor_f32(&[2, 3], data).unwrap();
/// let shape = get_shape(&tensor);
/// assert_eq!(shape, vec![2, 3]);
/// ```
pub fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.shape().dims().to_vec()
}

/// Reshapes a tensor to a new shape while preserving the total number of elements
///
/// # Arguments
/// * `tensor` - Reference to the tensor to reshape
/// * `new_shape` - Slice containing the new dimensions
///
/// # Returns
/// A Result containing the reshaped Tensor or an error if the reshape is invalid
///
/// # Example
/// ```
/// use bitnet_core::tensor::{create_tensor_f32, reshape};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let tensor = create_tensor_f32(&[2, 3], data).unwrap();
/// let reshaped = reshape(&tensor, &[3, 2]).unwrap();
/// ```
pub fn reshape(tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
    // Validate that the total number of elements remains the same
    let current_elements: usize = tensor.shape().dims().iter().product();
    let new_elements: usize = new_shape.iter().product();
    
    if current_elements != new_elements {
        return Err(candle_core::Error::ShapeMismatchBinaryOp {
            lhs: tensor.shape().clone(),
            rhs: candle_core::Shape::from_dims(new_shape),
            op: "reshape",
        });
    }
    
    tensor.reshape(new_shape)
}

/// Transposes a tensor by permuting its dimensions according to the specified order
///
/// # Arguments
/// * `tensor` - Reference to the tensor to transpose
/// * `dims` - Slice specifying the new order of dimensions (permutation)
///
/// # Returns
/// A Result containing the transposed Tensor or an error if the permutation is invalid
///
/// # Example
/// ```
/// use bitnet_core::tensor::{create_tensor_f32, transpose};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let tensor = create_tensor_f32(&[2, 3], data).unwrap();
/// let transposed = transpose(&tensor, &[1, 0]).unwrap(); // Swap dimensions
/// ```
pub fn transpose(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    // Validate that dims contains a valid permutation
    let tensor_rank = tensor.shape().rank();
    
    if dims.len() != tensor_rank {
        return Err(candle_core::Error::UnexpectedNumberOfDims {
            expected: tensor_rank,
            got: dims.len(),
            shape: tensor.shape().clone(),
        });
    }
    
    // Check that dims contains each dimension index exactly once
    let mut sorted_dims = dims.to_vec();
    sorted_dims.sort_unstable();
    let expected: Vec<usize> = (0..tensor_rank).collect();
    
    if sorted_dims != expected {
        return Err(candle_core::Error::InvalidIndex {
            op: "transpose",
            index: dims.iter().max().copied().unwrap_or(0),
            size: tensor_rank,
        });
    }
    
    tensor.permute(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_create_tensor_i8() {
        let data = vec![1i8, -1i8, 1i8, -1i8];
        let tensor = create_tensor_i8(&[2, 2], data).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.dtype(), DType::I64); // candle converts i8 to i64
    }

    #[test]
    fn test_zeros() {
        let tensor = zeros(&[3, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 3]);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_ones() {
        let tensor = ones(&[2, 4]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 4]);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_empty_shape() {
        let tensor = zeros(&[]).unwrap();
        assert_eq!(tensor.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_single_dimension() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = create_tensor_f32(&[3], data).unwrap();
        assert_eq!(tensor.shape().dims(), &[3]);
    }

    #[test]
    fn test_multi_dimension() {
        let data = vec![1.0; 24]; // 2 * 3 * 4 = 24
        let tensor = create_tensor_f32(&[2, 3, 4], data).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_get_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = create_tensor_f32(&[2, 3], data).unwrap();
        let shape = get_shape(&tensor);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_get_shape_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = create_tensor_f32(&[3], data).unwrap();
        let shape = get_shape(&tensor);
        assert_eq!(shape, vec![3]);
    }

    #[test]
    fn test_get_shape_scalar() {
        let tensor = zeros(&[]).unwrap();
        let shape = get_shape(&tensor);
        assert_eq!(shape, vec![] as Vec<usize>);
    }

    #[test]
    fn test_reshape_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = create_tensor_f32(&[2, 3], data).unwrap();
        let reshaped = reshape(&tensor, &[3, 2]).unwrap();
        assert_eq!(get_shape(&reshaped), vec![3, 2]);
    }

    #[test]
    fn test_reshape_to_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let reshaped = reshape(&tensor, &[4]).unwrap();
        assert_eq!(get_shape(&reshaped), vec![4]);
    }

    #[test]
    fn test_reshape_from_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = create_tensor_f32(&[6], data).unwrap();
        let reshaped = reshape(&tensor, &[2, 3]).unwrap();
        assert_eq!(get_shape(&reshaped), vec![2, 3]);
    }

    #[test]
    fn test_reshape_invalid_elements() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let result = reshape(&tensor, &[3, 2]); // 4 elements -> 6 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = create_tensor_f32(&[2, 3], data).unwrap();
        let transposed = transpose(&tensor, &[1, 0]).unwrap();
        assert_eq!(get_shape(&transposed), vec![3, 2]);
    }

    #[test]
    fn test_transpose_3d() {
        let data = vec![1.0; 24]; // 2 * 3 * 4 = 24
        let tensor = create_tensor_f32(&[2, 3, 4], data).unwrap();
        let transposed = transpose(&tensor, &[2, 0, 1]).unwrap();
        assert_eq!(get_shape(&transposed), vec![4, 2, 3]);
    }

    #[test]
    fn test_transpose_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let transposed = transpose(&tensor, &[0, 1]).unwrap();
        assert_eq!(get_shape(&transposed), vec![2, 2]);
    }

    #[test]
    fn test_transpose_invalid_dims_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let result = transpose(&tensor, &[0]); // 2D tensor with 1D permutation
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_invalid_dims_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let result = transpose(&tensor, &[0, 2]); // Invalid dimension index 2
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_duplicate_dims() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = create_tensor_f32(&[2, 2], data).unwrap();
        let result = transpose(&tensor, &[0, 0]); // Duplicate dimension index
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_utilities_integration() {
        // Test a complete workflow using all three functions
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = create_tensor_f32(&[2, 4], data).unwrap();
        
        // Get original shape
        let original_shape = get_shape(&tensor);
        assert_eq!(original_shape, vec![2, 4]);
        
        // Reshape to 4x2
        let reshaped = reshape(&tensor, &[4, 2]).unwrap();
        assert_eq!(get_shape(&reshaped), vec![4, 2]);
        
        // Transpose the reshaped tensor
        let transposed = transpose(&reshaped, &[1, 0]).unwrap();
        assert_eq!(get_shape(&transposed), vec![2, 4]);
        
        // Final shape should match original
        assert_eq!(get_shape(&transposed), original_shape);
    }
}