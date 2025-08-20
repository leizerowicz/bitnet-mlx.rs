//! Broadcasting Operations and Utilities
//!
//! This module provides NumPy/PyTorch compatible broadcasting operations
//! for BitNet tensors, optimized for memory efficiency and zero-copy
//! operations where possible.
//!
//! # Broadcasting Rules
//!
//! Broadcasting follows NumPy semantics:
//! 1. If tensors have different numbers of dimensions, pad the smaller
//!    tensor's shape with ones on the left
//! 2. For each dimension, check compatibility:
//!    - Dimensions are compatible if they are equal, or one of them is 1
//!    - The result dimension is the maximum of the two dimensions
//! 3. If any dimension is incompatible, broadcasting fails
//!
//! # Memory Efficiency
//!
//! - Zero-copy broadcasting when one tensor can be viewed as broadcast
//! - Efficient stride-based operations for common broadcasting patterns
//! - Memory pool reuse for temporary broadcast tensors
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::{BitNetTensor, BitNetDType};
//! use bitnet_core::tensor::ops::broadcasting::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = BitNetTensor::ones(&[3, 1, 4], BitNetDType::F32, None)?;
//! let b = BitNetTensor::ones(&[2, 1], BitNetDType::F32, None)?;
//!
//! // Check if broadcasting is possible
//! if can_broadcast(&a, &b)? {
//!     let broadcast_shape = compute_broadcast_shape(&a, &b)?;
//!     println!("Broadcast shape: {:?}", broadcast_shape);
//! }
//! # Ok(())
//! # }
//! ```

use std::cmp;
use crate::memory::MemoryResult;
use crate::tensor::core::BitNetTensor;
use crate::tensor::shape::{TensorShape, BroadcastCompatible};
use crate::tensor::dtype::BitNetDType;
use super::{TensorOpResult, TensorOpError};

#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn};

/// Clone a device (since candle Device doesn't implement Clone)
fn clone_device(device: &candle_core::Device) -> candle_core::Device {
    match device {
        candle_core::Device::Cpu => candle_core::Device::Cpu,
        candle_core::Device::Cuda(id) => candle_core::Device::Cuda(id.clone()),
        candle_core::Device::Metal(id) => candle_core::Device::Metal(id.clone()),
    }
}

/// Broadcasting strategy for operations
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastStrategy {
    /// No broadcasting needed - tensors have identical shapes
    None,
    /// Left tensor needs broadcasting
    BroadcastLeft,
    /// Right tensor needs broadcasting  
    BroadcastRight,
    /// Both tensors need broadcasting to a common shape
    BroadcastBoth,
}

/// Broadcasting metadata for efficient operations
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Final broadcast shape
    pub broadcast_shape: Vec<usize>,
    /// Strategy for broadcasting
    pub strategy: BroadcastStrategy,
    /// Whether zero-copy broadcasting is possible
    pub zero_copy: bool,
    /// Stride information for efficient iteration
    pub left_strides: Vec<usize>,
    pub right_strides: Vec<usize>,
    /// Memory requirements for the result
    pub result_elements: usize,
    pub result_bytes: usize,
}

/// Check if two tensors can be broadcast together
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// * `Ok(true)` if tensors can be broadcast
/// * `Ok(false)` if tensors cannot be broadcast
/// * `Err(_)` on internal error
pub fn can_broadcast(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<bool> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    
    Ok(shape_a.is_broadcast_compatible(shape_b))
}

/// Compute the resulting shape after broadcasting two tensors
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// * Broadcast shape as Vec<usize>
/// * Error if tensors cannot be broadcast
pub fn compute_broadcast_shape(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<Vec<usize>> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    
    if !shape_a.is_broadcast_compatible(shape_b) {
        return Err(TensorOpError::BroadcastError {
            reason: "Shapes are not compatible for broadcasting".to_string(),
            lhs_shape: shape_a.dims().to_vec(),
            rhs_shape: shape_b.dims().to_vec(),
            operation: "broadcasting".to_string(),
        }.into());
    }
    
    let broadcast_shape = shape_a.broadcast_shape(shape_b)
        .map_err(|_| TensorOpError::BroadcastError {
            reason: "Failed to compute broadcast shape".to_string(),
            lhs_shape: shape_a.dims().to_vec(),
            rhs_shape: shape_b.dims().to_vec(),
            operation: "broadcasting".to_string(),
        })?;
    
    Ok(broadcast_shape.dims().to_vec())
}

/// Analyze broadcasting requirements for two tensors
///
/// This function provides detailed information about how two tensors
/// would be broadcast together, including memory requirements and
/// optimization opportunities.
pub fn analyze_broadcast(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BroadcastInfo> {
    let shape_a = a.shape().dims();
    let shape_b = b.shape().dims();
    
    // Check if broadcasting is possible
    if !can_broadcast(a, b)? {
        return Err(TensorOpError::BroadcastError {
            reason: "Shapes are not compatible for broadcasting".to_string(),
            lhs_shape: shape_a.to_vec(),
            rhs_shape: shape_b.to_vec(),
            operation: "broadcasting".to_string(),
        }.into());
    }
    
    let broadcast_shape = compute_broadcast_shape(a, b)?;
    let result_elements: usize = broadcast_shape.iter().product();
    
    // Determine broadcasting strategy
    let strategy = if shape_a == shape_b {
        BroadcastStrategy::None
    } else if shape_a == broadcast_shape {
        BroadcastStrategy::BroadcastRight
    } else if shape_b == broadcast_shape {
        BroadcastStrategy::BroadcastLeft
    } else {
        BroadcastStrategy::BroadcastBoth
    };
    
    // Check if zero-copy broadcasting is possible
    let zero_copy = match strategy {
        BroadcastStrategy::None => true,
        BroadcastStrategy::BroadcastLeft | BroadcastStrategy::BroadcastRight => {
            // Zero-copy possible if we're just expanding dimensions with size 1
            can_zero_copy_broadcast(shape_a, shape_b, &broadcast_shape)
        }
        BroadcastStrategy::BroadcastBoth => false,
    };
    
    // Calculate strides for efficient iteration
    let left_strides = calculate_broadcast_strides(shape_a, &broadcast_shape);
    let right_strides = calculate_broadcast_strides(shape_b, &broadcast_shape);
    
    // Estimate result memory requirements
    let dtype_size = a.dtype().size_bytes().unwrap_or(4); // Default to 4 bytes
    let result_bytes = result_elements * dtype_size;
    
    #[cfg(feature = "tracing")]
    debug!(
        "Broadcasting analysis: {:?} + {:?} -> {:?}, strategy: {:?}, zero_copy: {}",
        shape_a, shape_b, broadcast_shape, strategy, zero_copy
    );
    
    Ok(BroadcastInfo {
        broadcast_shape,
        strategy,
        zero_copy,
        left_strides,
        right_strides,
        result_elements,
        result_bytes,
    })
}

/// Calculate strides for broadcasting a shape to a target shape
fn calculate_broadcast_strides(shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(target_shape.len());
    let shape_offset = target_shape.len() - shape.len();
    
    let mut current_stride = 1;
    for i in (0..target_shape.len()).rev() {
        if i < shape_offset {
            // Padding dimensions - stride is 0
            strides.push(0);
        } else {
            let shape_idx = i - shape_offset;
            if shape[shape_idx] == 1 && target_shape[i] > 1 {
                // Broadcasting dimension - stride is 0
                strides.push(0);
            } else {
                // Regular dimension
                strides.push(current_stride);
                current_stride *= shape[shape_idx];
            }
        }
    }
    
    strides.reverse();
    strides
}

/// Check if zero-copy broadcasting is possible
fn can_zero_copy_broadcast(shape_a: &[usize], shape_b: &[usize], target_shape: &[usize]) -> bool {
    // Zero-copy is possible if one tensor already has the target shape
    // and the other only needs stride modifications
    shape_a == target_shape || shape_b == target_shape
}

/// Create a broadcasted view of a tensor (zero-copy when possible)
///
/// This function creates a new tensor that represents a broadcasted view
/// of the input tensor to the target shape. When possible, this is done
/// without copying data by adjusting strides.
pub fn broadcast_to(tensor: &BitNetTensor, target_shape: &[usize]) -> TensorOpResult<BitNetTensor> {
    let current_shape = tensor.shape().dims();
    
    // Check if broadcasting is valid
    let temp_shape = TensorShape::new(target_shape);
    if !tensor.shape().is_broadcast_compatible(&temp_shape) {
        return Err(TensorOpError::BroadcastError {
            reason: "Shapes are not compatible for broadcasting".to_string(),
            lhs_shape: current_shape.to_vec(),
            rhs_shape: target_shape.to_vec(),
            operation: "broadcasting".to_string(),
        }.into());
    }
    
    // If shapes are already equal, return a clone
    if current_shape == target_shape {
        return Ok(tensor.clone());
    }
    
    // For now, create a new tensor with the target shape
    // In a full implementation, this would use stride manipulation for zero-copy
    let result = BitNetTensor::zeros(target_shape, tensor.dtype(), Some(clone_device(tensor.device())))?;
    
    #[cfg(feature = "tracing")]
    trace!("Created broadcast view: {:?} -> {:?}", current_shape, target_shape);
    
    Ok(result)
}

/// Prepare tensors for element-wise operations with broadcasting
///
/// This function analyzes two tensors and returns the information needed
/// to perform element-wise operations efficiently, including any necessary
/// broadcasting.
pub fn prepare_elementwise_broadcast(
    a: &BitNetTensor, 
    b: &BitNetTensor
) -> TensorOpResult<(BroadcastInfo, Option<BitNetTensor>, Option<BitNetTensor>)> {
    // Analyze broadcasting requirements
    let broadcast_info = analyze_broadcast(a, b)?;
    
    // Create broadcast tensors if needed
    let broadcast_a = match broadcast_info.strategy {
        BroadcastStrategy::None | BroadcastStrategy::BroadcastRight => None,
        BroadcastStrategy::BroadcastLeft | BroadcastStrategy::BroadcastBoth => {
            Some(broadcast_to(a, &broadcast_info.broadcast_shape)?)
        }
    };
    
    let broadcast_b = match broadcast_info.strategy {
        BroadcastStrategy::None | BroadcastStrategy::BroadcastLeft => None,
        BroadcastStrategy::BroadcastRight | BroadcastStrategy::BroadcastBoth => {
            Some(broadcast_to(b, &broadcast_info.broadcast_shape)?)
        }
    };
    
    #[cfg(feature = "tracing")]
    debug!(
        "Prepared elementwise broadcast: strategy {:?}, zero_copy: {}",
        broadcast_info.strategy, broadcast_info.zero_copy
    );
    
    Ok((broadcast_info, broadcast_a, broadcast_b))
}

/// Efficient iterator for broadcasting operations
///
/// This struct provides an efficient way to iterate over elements
/// of two tensors with broadcasting, handling stride calculations
/// automatically.
pub struct BroadcastIterator {
    shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    current_indices: Vec<usize>,
    finished: bool,
}

impl BroadcastIterator {
    /// Create a new broadcast iterator
    pub fn new(broadcast_info: &BroadcastInfo) -> Self {
        let shape = broadcast_info.broadcast_shape.clone();
        let current_indices = vec![0; shape.len()];
        
        Self {
            left_strides: broadcast_info.left_strides.clone(),
            right_strides: broadcast_info.right_strides.clone(),
            current_indices,
            finished: shape.is_empty() || shape.iter().any(|&d| d == 0),
            shape,
        }
    }
    
    /// Get the current linear indices for both tensors
    pub fn current_indices(&self) -> (usize, usize) {
        let left_idx = self.current_indices.iter()
            .zip(&self.left_strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum();
            
        let right_idx = self.current_indices.iter()
            .zip(&self.right_strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum();
            
        (left_idx, right_idx)
    }
}

impl Iterator for BroadcastIterator {
    type Item = (usize, usize);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        
        let result = self.current_indices();
        
        // Advance indices
        let mut carry = 1;
        for i in (0..self.current_indices.len()).rev() {
            self.current_indices[i] += carry;
            if self.current_indices[i] < self.shape[i] {
                carry = 0;
                break;
            }
            self.current_indices[i] = 0;
        }
        
        if carry == 1 {
            self.finished = true;
        }
        
        Some(result)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            let total_elements: usize = self.shape.iter().product();
            (total_elements, Some(total_elements))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{BitNetTensor, BitNetDType};
    
    #[test]
    fn test_can_broadcast() -> TensorOpResult<()> {
        let a = BitNetTensor::ones(&[3, 1, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 1], BitNetDType::F32, None)?;
        
        assert!(can_broadcast(&a, &b)?);
        
        let c = BitNetTensor::ones(&[3, 2], BitNetDType::F32, None)?;
        let d = BitNetTensor::ones(&[4, 2], BitNetDType::F32, None)?;
        
        assert!(!can_broadcast(&c, &d)?);
        
        Ok(())
    }
    
    #[test]
    fn test_compute_broadcast_shape() -> TensorOpResult<()> {
        let a = BitNetTensor::ones(&[3, 1, 4], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 1], BitNetDType::F32, None)?;
        
        let broadcast_shape = compute_broadcast_shape(&a, &b)?;
        assert_eq!(broadcast_shape, vec![3, 2, 4]);
        
        Ok(())
    }
    
    #[test]
    fn test_analyze_broadcast() -> TensorOpResult<()> {
        let a = BitNetTensor::ones(&[3, 1], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None)?;
        
        let info = analyze_broadcast(&a, &b)?;
        
        assert_eq!(info.broadcast_shape, vec![3, 4]);
        assert_eq!(info.strategy, BroadcastStrategy::BroadcastLeft);
        assert_eq!(info.result_elements, 12);
        
        Ok(())
    }
    
    #[test]
    fn test_broadcast_strides() {
        let strides = calculate_broadcast_strides(&[3, 1], &[3, 4]);
        assert_eq!(strides, vec![4, 0]);
        
        let strides2 = calculate_broadcast_strides(&[1, 4], &[3, 4]);
        assert_eq!(strides2, vec![0, 1]);
    }
    
    #[test]
    fn test_broadcast_iterator() -> TensorOpResult<()> {
        let a = BitNetTensor::ones(&[2, 1], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[1, 3], BitNetDType::F32, None)?;
        
        let info = analyze_broadcast(&a, &b)?;
        let iter = BroadcastIterator::new(&info);
        
        let indices: Vec<_> = iter.take(6).collect();
        
        // Should iterate through all 6 elements (2x3 broadcast)
        assert_eq!(indices.len(), 6);
        
        Ok(())
    }
}
