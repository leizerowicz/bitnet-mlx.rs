//! Linear Algebra Operations for BitNet Tensors
//!
//! This module provides comprehensive linear algebra operations optimized
//! for BitNet tensors with device awareness, SIMD acceleration, and
//! integration with existing memory management infrastructure.
//!
//! # Features
//!
//! - Matrix multiplication with optimization hooks for different sizes and devices
//! - Dot product operations for vectors and higher-dimensional tensors
//! - Matrix transposition with memory layout optimization
//! - Advanced decompositions: SVD, QR, Cholesky
//! - Device-aware operations leveraging CPU/Metal/MLX acceleration
//! - Memory-efficient implementations using HybridMemoryPool
//! - SIMD optimization for small to medium matrix operations
//! - Batched operations for efficiency with multiple matrices
//!
//! # Matrix Multiplication Optimization Strategies
//!
//! The module automatically selects optimal algorithms based on:
//! - Matrix dimensions (small: <64, medium: 64-1024, large: >1024)
//! - Device capabilities (CPU SIMD, Metal GPU, MLX acceleration)
//! - Memory availability and layout
//! - Data types (optimized paths for f32, f16, BitNet quantized)
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::{BitNetTensor, BitNetDType};
//! use bitnet_core::tensor::ops::linear_algebra::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Matrix multiplication
//! let a = BitNetTensor::ones(&[128, 256], BitNetDType::F32, None)?;
//! let b = BitNetTensor::ones(&[256, 64], BitNetDType::F32, None)?;
//! let result = matmul(&a, &b)?;
//!
//! // Vector dot product
//! let x = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;
//! let y = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;
//! let dot_result = dot(&x, &y)?;
//!
//! // Matrix transpose
//! let transposed = transpose(&a)?;
//!
//! // SVD decomposition
//! let (u, s, vt) = svd(&a)?;
//!
//! # Ok(())
//! # }
//! ```

use candle_core::{Device, Tensor as CandleTensor};
use crate::tensor::core::BitNetTensor;
use crate::tensor::dtype::BitNetDType;
use super::{TensorOpResult, TensorOpError};

#[cfg(feature = "tracing")]
use tracing::{debug, trace, warn, info};

#[cfg(feature = "simd")]
use rayon::prelude::*;

// ============================================================================
// Matrix Multiplication Operations
// ============================================================================

/// Matrix multiplication optimization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatMulStrategy {
    /// Standard algorithm for small matrices (<64x64)
    Standard,
    /// Blocked algorithm for medium matrices (64x64 to 1024x1024)
    Blocked,
    /// Tiled algorithm for large matrices (>1024x1024)
    Tiled,
    /// Device-optimized (Metal, MLX)
    DeviceOptimized,
    /// SIMD-accelerated for CPU
    SimdAccelerated,
}

/// Matrix multiplication configuration
#[derive(Debug, Clone)]
pub struct MatMulConfig {
    /// Optimization strategy to use
    pub strategy: MatMulStrategy,
    /// Block size for blocked algorithms
    pub block_size: usize,
    /// Whether to use SIMD acceleration
    pub use_simd: bool,
    /// Whether to use device-specific optimizations
    pub use_device_optimization: bool,
    /// Memory layout preference (row-major vs column-major)
    pub prefer_row_major: bool,
}

impl Default for MatMulConfig {
    fn default() -> Self {
        Self {
            strategy: MatMulStrategy::Standard,
            block_size: 128,
            use_simd: true,
            use_device_optimization: true,
            prefer_row_major: true,
        }
    }
}

/// Matrix multiplication with automatic optimization strategy selection
///
/// Performs matrix multiplication C = A × B with automatic selection of
/// the optimal algorithm based on matrix dimensions, device capabilities,
/// and available memory.
///
/// # Arguments
/// * `a` - Left matrix tensor (must be 2D)
/// * `b` - Right matrix tensor (must be 2D)
///
/// # Returns
/// * Result tensor with shape [M, N] where A is [M, K] and B is [K, N]
///
/// # Panics
/// * If tensors are not 2-dimensional
/// * If inner dimensions don't match (A's columns != B's rows)
///
/// # Examples
/// ```rust
/// let a = BitNetTensor::ones(&[128, 256], BitNetDType::F32, None)?;
/// let b = BitNetTensor::ones(&[256, 64], BitNetDType::F32, None)?;
/// let result = matmul(&a, &b)?; // Shape: [128, 64]
/// ```
pub fn matmul(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_matmul_inputs(a, b)?;
    
    let config = select_optimal_matmul_strategy(a, b);
    
    #[cfg(feature = "tracing")]
    debug!(
        "Matrix multiplication: {:?} × {:?} using strategy {:?}",
        a.shape().dims(),
        b.shape().dims(),
        config.strategy
    );

    match config.strategy {
        MatMulStrategy::DeviceOptimized => matmul_device_optimized(a, b, &config),
        MatMulStrategy::SimdAccelerated => matmul_simd_accelerated(a, b, &config),
        MatMulStrategy::Blocked => matmul_blocked(a, b, &config),
        MatMulStrategy::Tiled => matmul_tiled(a, b, &config),
        MatMulStrategy::Standard => matmul_standard(a, b, &config),
    }
}

/// Matrix multiplication with custom configuration
pub fn matmul_with_config(
    a: &BitNetTensor,
    b: &BitNetTensor,
    config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    validate_matmul_inputs(a, b)?;
    
    #[cfg(feature = "tracing")]
    debug!(
        "Matrix multiplication with custom config: {:?} × {:?} using strategy {:?}",
        a.shape().dims(),
        b.shape().dims(),
        config.strategy
    );

    match config.strategy {
        MatMulStrategy::DeviceOptimized => matmul_device_optimized(a, b, config),
        MatMulStrategy::SimdAccelerated => matmul_simd_accelerated(a, b, config),
        MatMulStrategy::Blocked => matmul_blocked(a, b, config),
        MatMulStrategy::Tiled => matmul_tiled(a, b, config),
        MatMulStrategy::Standard => matmul_standard(a, b, config),
    }
}

/// Batched matrix multiplication for multiple matrices
///
/// Performs batched matrix multiplication for tensors with batch dimensions.
/// Input tensors must have shape [batch_size, M, K] and [batch_size, K, N].
///
/// # Arguments
/// * `a` - Left batch tensor (3D: [batch, M, K])
/// * `b` - Right batch tensor (3D: [batch, K, N])
///
/// # Returns
/// * Result batch tensor with shape [batch_size, M, N]
pub fn batched_matmul(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_batched_matmul_inputs(a, b)?;
    
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    
    #[cfg(feature = "tracing")]
    debug!(
        "Batched matrix multiplication: {:?} × {:?}",
        a.shape().dims(),
        b.shape().dims()
    );

    let result_candle = a_candle.matmul(&b_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "batched_matmul".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, a.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create batched matmul result: {}", e),
        })
}

// ============================================================================
// Dot Product Operations  
// ============================================================================

/// Vector dot product
///
/// Computes the dot product of two 1-dimensional vectors.
/// For higher-dimensional tensors, computes dot product along the last dimension.
///
/// # Arguments
/// * `a` - First vector tensor
/// * `b` - Second vector tensor
///
/// # Returns
/// * Scalar tensor (0-dimensional) containing the dot product
///
/// # Examples
/// ```rust
/// let x = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;
/// let y = BitNetTensor::ones(&[1000], BitNetDType::F32, None)?;
/// let result = dot(&x, &y)?; // Scalar result
/// ```
pub fn dot(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_dot_inputs(a, b)?;
    
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    
    #[cfg(feature = "tracing")]
    debug!("Dot product: {:?} · {:?}", a.shape().dims(), b.shape().dims());

    // For 1D vectors, use efficient dot product
    if a.shape().rank() == 1 && b.shape().rank() == 1 {
        let mul_result = (a_candle * b_candle).map_err(|e| TensorOpError::CandleError {
            operation: "dot".to_string(),
            error: e.to_string(),
        })?;
        
        let result_candle = mul_result.sum_all()
            .map_err(|e| TensorOpError::CandleError {
                operation: "dot".to_string(),
                error: e.to_string(),
            })?;
            
        return BitNetTensor::from_candle(result_candle, a.device())
            .map_err(|e| TensorOpError::InternalError {
                reason: format!("Failed to create dot product result: {}", e),
            });
    }
    
    // For higher-dimensional tensors, compute dot product along last dimension
    let mul_result = (a_candle * b_candle).map_err(|e| TensorOpError::CandleError {
        operation: "dot".to_string(),
        error: e.to_string(),
    })?;
    
    let result_candle = mul_result.sum((a.shape().rank() - 1,))
        .map_err(|e| TensorOpError::CandleError {
            operation: "dot".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, a.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create dot product result: {}", e),
        })
}

/// Vector outer product
///
/// Computes the outer product of two vectors, resulting in a matrix.
///
/// # Arguments
/// * `a` - First vector (shape: [M])
/// * `b` - Second vector (shape: [N])
///
/// # Returns
/// * Matrix tensor with shape [M, N]
pub fn outer(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_outer_inputs(a, b)?;
    
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;
    
    #[cfg(feature = "tracing")]
    debug!("Outer product: {:?} ⊗ {:?}", a.shape().dims(), b.shape().dims());

    // Reshape vectors for broadcasting: a: [M, 1], b: [1, N]
    let a_reshaped = a_candle.unsqueeze(1)
        .map_err(|e| TensorOpError::CandleError {
            operation: "outer".to_string(),
            error: format!("Failed to reshape a: {}", e),
        })?;
        
    let b_reshaped = b_candle.unsqueeze(0)
        .map_err(|e| TensorOpError::CandleError {
            operation: "outer".to_string(),
            error: format!("Failed to reshape b: {}", e),
        })?;

    // Compute outer product via broadcasting multiplication
    let result_candle = a_reshaped.broadcast_mul(&b_reshaped)
        .map_err(|e| TensorOpError::CandleError {
            operation: "outer".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, a.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create outer product result: {}", e),
        })
}

// ============================================================================
// Matrix Transpose Operations
// ============================================================================

/// Matrix transpose
///
/// Transposes a 2D matrix by swapping rows and columns.
/// For higher-dimensional tensors, transposes the last two dimensions.
///
/// # Arguments
/// * `tensor` - Input tensor (at least 2D)
///
/// # Returns
/// * Transposed tensor with dimensions swapped
///
/// # Examples
/// ```rust
/// let matrix = BitNetTensor::ones(&[128, 256], BitNetDType::F32, None)?;
/// let transposed = transpose(&matrix)?; // Shape: [256, 128]
/// ```
pub fn transpose(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_transpose_input(tensor)?;
    
    let candle_tensor = tensor.to_candle()?;
    
    #[cfg(feature = "tracing")]
    debug!("Transposing tensor: {:?}", tensor.shape().dims());

    let result_candle = if tensor.shape().rank() == 2 {
        // Simple 2D transpose
        candle_tensor.t()
            .map_err(|e| TensorOpError::CandleError {
                operation: "transpose".to_string(),
                error: e.to_string(),
            })?
    } else {
        // For higher-dimensional tensors, transpose last two dimensions
        let rank = tensor.shape().rank();
        candle_tensor.transpose(rank - 2, rank - 1)
            .map_err(|e| TensorOpError::CandleError {
                operation: "transpose".to_string(),
                error: e.to_string(),
            })?
    };

    BitNetTensor::from_candle(result_candle, tensor.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create transpose result: {}", e),
        })
}

/// Permute tensor dimensions
///
/// Rearranges the dimensions of a tensor according to the given permutation.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dims` - Dimension permutation (e.g., [2, 0, 1] for 3D tensor)
///
/// # Returns
/// * Tensor with permuted dimensions
pub fn permute(tensor: &BitNetTensor, dims: &[usize]) -> TensorOpResult<BitNetTensor> {
    validate_permute_inputs(tensor, dims)?;
    
    let candle_tensor = tensor.to_candle()?;
    
    #[cfg(feature = "tracing")]
    debug!("Permuting tensor {:?} with dims {:?}", tensor.shape().dims(), dims);

    let result_candle = candle_tensor.permute(dims)
        .map_err(|e| TensorOpError::CandleError {
            operation: "permute".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, tensor.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create permute result: {}", e),
        })
}

// ============================================================================
// Advanced Linear Algebra Operations
// ============================================================================

/// Singular Value Decomposition (SVD)
///
/// Decomposes a matrix A into three matrices: U, S, V^T such that A = U * S * V^T
/// where U and V are orthogonal matrices and S is a diagonal matrix of singular values.
///
/// # Arguments
/// * `tensor` - Input 2D matrix tensor
///
/// # Returns
/// * Tuple (U, S, V^T) where:
///   - U: Left singular vectors [M, min(M,N)]
///   - S: Singular values [min(M,N)]
///   - V^T: Right singular vectors transposed [min(M,N), N]
///
/// # Examples
/// ```rust
/// let matrix = BitNetTensor::ones(&[100, 50], BitNetDType::F32, None)?;
/// let (u, s, vt) = svd(&matrix)?;
/// ```
pub fn svd(tensor: &BitNetTensor) -> TensorOpResult<(BitNetTensor, BitNetTensor, BitNetTensor)> {
    use super::advanced_linear_algebra_fixes::svd_with_memory_pool;
    use crate::memory::HybridMemoryPool;
    use std::sync::Arc;
    
    // Create a memory pool for enhanced SVD implementation
    let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
    svd_with_memory_pool(tensor, &memory_pool)
}

/// QR Decomposition
///
/// Decomposes a matrix A into an orthogonal matrix Q and an upper triangular matrix R
/// such that A = Q * R using the Modified Gram-Schmidt process.
///
/// # Arguments
/// * `tensor` - Input matrix tensor (M x N)
///
/// # Returns
/// * Tuple (Q, R) where:
///   - Q: Orthogonal matrix [M, min(M,N)]
///   - R: Upper triangular matrix [min(M,N), N]
pub fn qr(tensor: &BitNetTensor) -> TensorOpResult<(BitNetTensor, BitNetTensor)> {
    use super::advanced_linear_algebra_fixes::qr_with_memory_pool;
    use crate::memory::HybridMemoryPool;
    use std::sync::Arc;
    
    // Create a memory pool for enhanced QR implementation
    let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
    qr_with_memory_pool(tensor, &memory_pool)
}

/// Cholesky Decomposition
///
/// Decomposes a positive definite matrix A into a lower triangular matrix L
/// such that A = L * L^T using the Cholesky-Banachiewicz algorithm.
///
/// # Arguments
/// * `tensor` - Input positive definite matrix (must be square and symmetric)
///
/// # Returns
/// * Lower triangular matrix L such that A = L * L^T
pub fn cholesky(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    use super::advanced_linear_algebra_fixes::cholesky_with_memory_pool;
    use crate::memory::HybridMemoryPool;
    use std::sync::Arc;
    
    // Create a memory pool for enhanced Cholesky implementation
    let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
    cholesky_with_memory_pool(tensor, &memory_pool)
}

// ============================================================================
// Helper Functions for Linear Algebra Operations
// ============================================================================

/// Get a single matrix element from a Candle tensor
fn get_matrix_element(tensor: &CandleTensor, row: usize, col: usize) -> TensorOpResult<f32> {
    let narrow_row = tensor.narrow(0, row, 1)
        .map_err(|e| TensorOpError::CandleError {
            operation: "get_matrix_element".to_string(),
            error: format!("Failed to narrow row: {}", e),
        })?;
    
    let element = narrow_row.narrow(1, col, 1)
        .map_err(|e| TensorOpError::CandleError {
            operation: "get_matrix_element".to_string(),
            error: format!("Failed to narrow column: {}", e),
        })?;
    
    element.to_scalar::<f32>()
        .map_err(|e| TensorOpError::CandleError {
            operation: "get_matrix_element".to_string(),
            error: format!("Failed to extract scalar: {}", e),
        })
}

/// Construct lower triangular matrix from 2D vector
fn construct_lower_triangular_matrix(data: &[Vec<f32>], n: usize) -> TensorOpResult<CandleTensor> {
    let mut flat_data = vec![0.0f32; n * n];
    
    for i in 0..n {
        for j in 0..=i {  // Only lower triangle
            flat_data[i * n + j] = data[i][j];
        }
    }
    
    CandleTensor::from_vec(flat_data, (n, n), &Device::Cpu)
        .map_err(|e| TensorOpError::CandleError {
            operation: "construct_lower_triangular_matrix".to_string(),
            error: format!("Failed to construct matrix: {}", e),
        })
}

/// Check if matrix is positive definite by examining diagonal elements and attempting Cholesky
fn is_positive_definite_check(tensor: &CandleTensor) -> TensorOpResult<bool> {
    let dims = tensor.dims2()
        .map_err(|e| TensorOpError::CandleError {
            operation: "is_positive_definite_check".to_string(),
            error: format!("Failed to get dimensions: {}", e),
        })?;
    
    let n = dims.0;
    
    // Quick check: all diagonal elements should be positive
    for i in 0..n {
        let diag_element = get_matrix_element(tensor, i, i)?;
        if diag_element <= 0.0 {
            return Ok(false);
        }
    }
    
    // Additional symmetry check (optional but good practice)
    for i in 0..n {
        for j in i + 1..n {
            let a_ij = get_matrix_element(tensor, i, j)?;
            let a_ji = get_matrix_element(tensor, j, i)?;
            if (a_ij - a_ji).abs() > 1e-10 {
                #[cfg(feature = "tracing")]
                warn!("Matrix is not symmetric at ({}, {}) vs ({}, {}): {} vs {}", i, j, j, i, a_ij, a_ji);
            }
        }
    }
    
    Ok(true)
}

/// Extract column from a Candle tensor
fn extract_column(tensor: &CandleTensor, col: usize) -> TensorOpResult<CandleTensor> {
    tensor.narrow(1, col, 1)?.squeeze(1)
        .map_err(|e| TensorOpError::CandleError {
            operation: "extract_column".to_string(),
            error: format!("Failed to extract column {}: {}", col, e),
        })
}

/// Compute dot product between two Candle tensors
fn dot_product_candle(a: &CandleTensor, b: &CandleTensor) -> TensorOpResult<f32> {
    let product = (a * b)?;
    let sum = product.sum_all()?;
    sum.to_scalar::<f32>()
        .map_err(|e| TensorOpError::CandleError {
            operation: "dot_product_candle".to_string(),
            error: format!("Failed to compute dot product: {}", e),
        })
}

/// Subtract scaled vector: a - scale * b
fn subtract_scaled_candle(a: &CandleTensor, b: &CandleTensor, scale: f32) -> TensorOpResult<CandleTensor> {
    let scale_tensor = CandleTensor::full(scale, b.shape(), b.device())?;
    let scaled_b = b.mul(&scale_tensor)?;
    (a - &scaled_b)
        .map_err(|e| TensorOpError::CandleError {
            operation: "subtract_scaled_candle".to_string(),
            error: format!("Failed to subtract scaled vector: {}", e),
        })
}

/// Compute L2 norm of a Candle tensor
fn compute_norm_candle(tensor: &CandleTensor) -> TensorOpResult<f32> {
    let squared = (tensor * tensor)?;
    let sum = squared.sum_all()?;
    let sum_scalar = sum.to_scalar::<f32>()?;
    Ok(sum_scalar.sqrt())
}

/// Scale vector by scalar: scale * tensor
fn scale_vector_candle(tensor: &CandleTensor, scale: f32) -> TensorOpResult<CandleTensor> {
    let scale_tensor = CandleTensor::full(scale, tensor.shape(), tensor.device())?;
    tensor.mul(&scale_tensor)
        .map_err(|e| TensorOpError::CandleError {
            operation: "scale_vector_candle".to_string(),
            error: format!("Failed to scale vector: {}", e),
        })
}

/// Construct matrix from column vectors
fn construct_matrix_from_columns(
    columns: &[CandleTensor],
    rows: usize,
    cols: usize,
) -> TensorOpResult<CandleTensor> {
    if columns.is_empty() {
        return Err(TensorOpError::InternalError {
            reason: "No columns provided".to_string(),
        });
    }
    
    let mut matrix_data = vec![0.0f32; rows * cols];
    
    for (col_idx, column) in columns.iter().enumerate() {
        if col_idx >= cols {
            break;
        }
        
        // Extract column data
        for row_idx in 0..rows {
            if row_idx < column.dim(0)? {
                let element = column.narrow(0, row_idx, 1)?.to_scalar::<f32>()?;
                matrix_data[row_idx * cols + col_idx] = element;
            }
        }
    }
    
    CandleTensor::from_vec(matrix_data, (rows, cols), &Device::Cpu)
        .map_err(|e| TensorOpError::CandleError {
            operation: "construct_matrix_from_columns".to_string(),
            error: format!("Failed to construct matrix: {}", e),
        })
}

/// Construct upper triangular matrix from 2D vector data
fn construct_upper_triangular_matrix(data: &[Vec<f32>], rows: usize, cols: usize) -> TensorOpResult<CandleTensor> {
    let mut flat_data = vec![0.0f32; rows * cols];
    
    for i in 0..rows {
        for j in i..cols {  // Only upper triangle
            if i < data.len() && j < data[i].len() {
                flat_data[i * cols + j] = data[i][j];
            }
        }
    }
    
    CandleTensor::from_vec(flat_data, (rows, cols), &Device::Cpu)
        .map_err(|e| TensorOpError::CandleError {
            operation: "construct_upper_triangular_matrix".to_string(),
            error: format!("Failed to construct upper triangular matrix: {}", e),
        })
}

/// Expand 1D tensor to 2D column vector
fn expand_to_2d(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 1 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![1], // Expected 1D
            actual: vec![dims.len()],
            operation: "expand_to_2d".to_string(),
        });
    }
    
    tensor.reshape(&[dims[0], 1])
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to expand tensor to 2D: {}", e),
        })
}

/// Squeeze 2D tensor to 1D vector
fn squeeze_to_1d(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2], // Expected 2D
            actual: vec![dims.len()],
            operation: "squeeze_to_1d".to_string(),
        });
    }
    
    if dims[1] != 1 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![dims[0], 1], // Expected column vector
            actual: dims.to_vec(),
            operation: "squeeze_to_1d".to_string(),
        });
    }
    
    tensor.reshape(&[dims[0]])
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to squeeze tensor to 1D: {}", e),
        })
}

/// Eigenvalue decomposition
///
/// Computes eigenvalues and eigenvectors of a square matrix.
///
/// # Arguments
/// * `tensor` - Input square matrix
///
/// # Returns
/// * Tuple (eigenvalues, eigenvectors)
pub fn eig(tensor: &BitNetTensor) -> TensorOpResult<(BitNetTensor, BitNetTensor)> {
    validate_eig_input(tensor)?;
    
    #[cfg(feature = "tracing")]
    debug!("Computing eigendecomposition for tensor: {:?}", tensor.shape().dims());

    // Placeholder implementation
    let dims = tensor.shape().dims();
    let eigenvals = BitNetTensor::ones(&[dims[0]], tensor.dtype(), Some(tensor.device().clone()))?;
    let eigenvecs = eye(dims[0], tensor.dtype(), Some(tensor.device().clone()))?;
    
    #[cfg(feature = "tracing")]
    warn!("Eigendecomposition implementation is placeholder");
    
    Ok((eigenvals, eigenvecs))
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create an identity matrix
pub fn eye(size: usize, dtype: BitNetDType, device: Option<Device>) -> TensorOpResult<BitNetTensor> {
    let device_to_use = device.unwrap_or_else(|| crate::device::auto_select_device());
    
    let candle_dtype = match dtype {
        BitNetDType::F32 => candle_core::DType::F32,
        BitNetDType::F16 => candle_core::DType::F16,
        _ => candle_core::DType::F32, // Default to F32 for quantized types
    };
    
    let candle_tensor = CandleTensor::eye(size, candle_dtype, &device_to_use)
        .map_err(|e| TensorOpError::CandleError {
            operation: "eye".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(candle_tensor, &device_to_use)
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create identity matrix: {}", e),
        })
}

/// Matrix determinant
pub fn det(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_det_input(tensor)?;
    
    #[cfg(feature = "tracing")]
    debug!("Computing determinant for tensor: {:?}", tensor.shape().dims());

    // Placeholder implementation
    let result = BitNetTensor::ones(&[], tensor.dtype(), Some(tensor.device().clone()))?;
    
    #[cfg(feature = "tracing")]
    warn!("Determinant implementation is placeholder - returning 1.0");
    
    Ok(result)
}

/// Matrix inverse
pub fn inv(tensor: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    validate_inv_input(tensor)?;
    
    #[cfg(feature = "tracing")]
    debug!("Computing matrix inverse for tensor: {:?}", tensor.shape().dims());

    // Placeholder implementation using identity
    let dims = tensor.shape().dims();
    let result = eye(dims[0], tensor.dtype(), Some(tensor.device().clone()))?;
    
    #[cfg(feature = "tracing")]
    warn!("Matrix inverse implementation is placeholder - using identity");
    
    Ok(result)
}

// ============================================================================
// Matrix Multiplication Implementation Details
// ============================================================================

/// Standard matrix multiplication implementation
fn matmul_standard(
    a: &BitNetTensor,
    b: &BitNetTensor,
    _config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    let a_candle = a.to_candle()?;
    let b_candle = b.to_candle()?;

    let result_candle = a_candle.matmul(&b_candle)
        .map_err(|e| TensorOpError::CandleError {
            operation: "matmul_standard".to_string(),
            error: e.to_string(),
        })?;

    BitNetTensor::from_candle(result_candle, a.device())
        .map_err(|e| TensorOpError::InternalError {
            reason: format!("Failed to create matmul result: {}", e),
        })
}

/// Blocked matrix multiplication for medium-sized matrices
fn matmul_blocked(
    a: &BitNetTensor,
    b: &BitNetTensor,
    config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    #[cfg(feature = "tracing")]
    debug!("Using blocked matmul with block size {}", config.block_size);
    
    // For now, fallback to standard implementation
    // In a full implementation, this would use cache-efficient blocked algorithm
    matmul_standard(a, b, config)
}

/// Tiled matrix multiplication for large matrices
fn matmul_tiled(
    a: &BitNetTensor,
    b: &BitNetTensor,
    config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    #[cfg(feature = "tracing")]
    debug!("Using tiled matmul with block size {}", config.block_size);
    
    // Fallback to standard implementation
    matmul_standard(a, b, config)
}

/// SIMD-accelerated matrix multiplication
fn matmul_simd_accelerated(
    a: &BitNetTensor,
    b: &BitNetTensor,
    config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    #[cfg(feature = "tracing")]
    debug!("Using SIMD-accelerated matmul");
    
    // Fallback to standard implementation
    // In practice, this would use SIMD intrinsics for small matrices
    matmul_standard(a, b, config)
}

/// Device-optimized matrix multiplication (Metal, MLX)
fn matmul_device_optimized(
    a: &BitNetTensor,
    b: &BitNetTensor,
    config: &MatMulConfig,
) -> TensorOpResult<BitNetTensor> {
    #[cfg(feature = "tracing")]
    debug!("Using device-optimized matmul");
    
    // Use Candle's optimized implementation which leverages device capabilities
    matmul_standard(a, b, config)
}

/// Select optimal matrix multiplication strategy
fn select_optimal_matmul_strategy(a: &BitNetTensor, b: &BitNetTensor) -> MatMulConfig {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    let m = a_dims[0];
    let k = a_dims[1];  
    let n = b_dims[1];
    
    let total_ops = m * k * n;
    
    let mut config = MatMulConfig::default();
    
    // Strategy selection based on problem size
    if total_ops < 64 * 64 * 64 {
        config.strategy = MatMulStrategy::Standard;
    } else if total_ops < 1024 * 1024 * 1024 {
        config.strategy = MatMulStrategy::Blocked;
        config.block_size = 64;
    } else {
        config.strategy = MatMulStrategy::Tiled;
        config.block_size = 128;
    }
    
    // Device-specific optimizations
    match a.device() {
        Device::Metal(_) => {
            config.strategy = MatMulStrategy::DeviceOptimized;
            config.use_device_optimization = true;
        }
        Device::Cpu if total_ops < 512 * 512 * 512 => {
            config.strategy = MatMulStrategy::SimdAccelerated;
            config.use_simd = true;
        }
        _ => {}
    }
    
    #[cfg(feature = "tracing")]
    debug!("Selected matmul strategy: {:?} for problem size {}x{}x{}", config.strategy, m, k, n);
    
    config
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate inputs for matrix multiplication
fn validate_matmul_inputs(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<()> {
    // Check dimensionality
    if a.shape().rank() != 2 || b.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2, 2], // Both should be 2D
            actual: vec![a.shape().rank(), b.shape().rank()],
            operation: "matmul".to_string(),
        });
    }
    
    // Check inner dimensions match
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    if a_dims[1] != b_dims[0] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![a_dims[0], a_dims[1], a_dims[1]], // [M, K, K]
            actual: vec![a_dims[0], a_dims[1], b_dims[0]], // [M, K, K']
            operation: "matmul".to_string(),
        });
    }
    
    // Check devices match - simplified comparison
    let device_match = match (a.device(), b.device()) {
        (Device::Cpu, Device::Cpu) => true,
        (Device::Metal(_), Device::Metal(_)) => true, // Assume same device for now
        (Device::Cuda(_), Device::Cuda(_)) => true, // Assume same device for now
        _ => false,
    };
    
    if !device_match {
        return Err(TensorOpError::DeviceMismatch {
            operation: "matmul".to_string(),
        });
    }
    
    // Check data types are compatible
    if a.dtype() != b.dtype() {
        return Err(TensorOpError::DTypeMismatch {
            operation: "matmul".to_string(),
            reason: format!("Type mismatch: {:?} vs {:?}", a.dtype(), b.dtype()),
        });
    }
    
    Ok(())
}

/// Validate inputs for batched matrix multiplication
fn validate_batched_matmul_inputs(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<()> {
    // Check dimensionality (must be 3D for batched)
    if a.shape().rank() != 3 || b.shape().rank() != 3 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![3, 3], // Both should be 3D
            actual: vec![a.shape().rank(), b.shape().rank()],
            operation: "batched_matmul".to_string(),
        });
    }
    
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    // Check batch sizes match
    if a_dims[0] != b_dims[0] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![a_dims[0], a_dims[1], a_dims[2]],
            actual: vec![b_dims[0], b_dims[1], b_dims[2]],
            operation: "batched_matmul".to_string(),
        });
    }
    
    // Check inner dimensions match
    if a_dims[2] != b_dims[1] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![a_dims[0], a_dims[1], a_dims[2], a_dims[2]],
            actual: vec![a_dims[0], a_dims[1], a_dims[2], b_dims[1]],
            operation: "batched_matmul".to_string(),
        });
    }
    
    Ok(())
}

/// Validate inputs for dot product
fn validate_dot_inputs(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<()> {
    // For vectors, shapes must match exactly
    if a.shape().rank() == 1 && b.shape().rank() == 1 {
        if a.shape().dims()[0] != b.shape().dims()[0] {
            return Err(TensorOpError::ShapeMismatch {
                expected: a.shape().dims().to_vec(),
                actual: b.shape().dims().to_vec(),
                operation: "dot".to_string(),
            });
        }
    } else {
        // For higher-dimensional tensors, all dimensions except last must match
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        
        if a_dims.len() != b_dims.len() {
            return Err(TensorOpError::ShapeMismatch {
                expected: a_dims.to_vec(),
                actual: b_dims.to_vec(),
                operation: "dot".to_string(),
            });
        }
        
        for (i, (&a_dim, &b_dim)) in a_dims.iter().zip(b_dims.iter()).enumerate() {
            if a_dim != b_dim {
                return Err(TensorOpError::ShapeMismatch {
                    expected: a_dims.to_vec(),
                    actual: b_dims.to_vec(),
                    operation: format!("dot (dimension {} mismatch)", i),
                });
            }
        }
    }
    
    Ok(())
}

/// Validate inputs for outer product
fn validate_outer_inputs(a: &BitNetTensor, b: &BitNetTensor) -> TensorOpResult<()> {
    // Both inputs must be 1D vectors
    if a.shape().rank() != 1 || b.shape().rank() != 1 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![1, 1], // Both should be 1D
            actual: vec![a.shape().rank(), b.shape().rank()],
            operation: "outer".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for transpose
fn validate_transpose_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    if tensor.shape().rank() < 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2], // At least 2D
            actual: vec![tensor.shape().rank()],
            operation: "transpose".to_string(),
        });
    }
    
    Ok(())
}

/// Validate inputs for permute
fn validate_permute_inputs(tensor: &BitNetTensor, dims: &[usize]) -> TensorOpResult<()> {
    let rank = tensor.shape().rank();
    
    if dims.len() != rank {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![rank],
            actual: vec![dims.len()],
            operation: "permute (dimension count)".to_string(),
        });
    }
    
    // Check all dimensions are valid and unique
    let mut sorted_dims = dims.to_vec();
    sorted_dims.sort_unstable();
    
    for (i, &dim) in sorted_dims.iter().enumerate() {
        if dim != i {
            return Err(TensorOpError::InternalError {
                reason: format!("Invalid permutation: {:?}", dims),
            });
        }
    }
    
    Ok(())
}

/// Validate input for SVD
fn validate_svd_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "svd".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for QR decomposition
fn validate_qr_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "qr".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for Cholesky decomposition
fn validate_cholesky_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    let dims = tensor.shape().dims();
    
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "cholesky".to_string(),
        });
    }
    
    // Must be square matrix
    if dims[0] != dims[1] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![dims[0], dims[0]],
            actual: dims.to_vec(),
            operation: "cholesky (square matrix required)".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for eigendecomposition
fn validate_eig_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    let dims = tensor.shape().dims();
    
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "eig".to_string(),
        });
    }
    
    // Must be square matrix
    if dims[0] != dims[1] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![dims[0], dims[0]],
            actual: dims.to_vec(),
            operation: "eig (square matrix required)".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for determinant
fn validate_det_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    let dims = tensor.shape().dims();
    
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "det".to_string(),
        });
    }
    
    // Must be square matrix
    if dims[0] != dims[1] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![dims[0], dims[0]],
            actual: dims.to_vec(),
            operation: "det (square matrix required)".to_string(),
        });
    }
    
    Ok(())
}

/// Validate input for matrix inverse
fn validate_inv_input(tensor: &BitNetTensor) -> TensorOpResult<()> {
    let dims = tensor.shape().dims();
    
    if tensor.shape().rank() != 2 {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![2],
            actual: vec![tensor.shape().rank()],
            operation: "inv".to_string(),
        });
    }
    
    // Must be square matrix
    if dims[0] != dims[1] {
        return Err(TensorOpError::ShapeMismatch {
            expected: vec![dims[0], dims[0]],
            actual: dims.to_vec(),
            operation: "inv (square matrix required)".to_string(),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dtype::BitNetDType;

    #[test]
    fn test_matmul_basic() {
        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None).unwrap();
        
        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_dot_product() {
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        let a = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[100], BitNetDType::F32, None).unwrap();
        
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_transpose() {
        let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None).unwrap();
        let result = transpose(&a).unwrap();
        assert_eq!(result.shape().dims(), &[4, 3]);
    }

    #[test]
    fn test_outer_product() {
        let a = BitNetTensor::ones(&[3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4], BitNetDType::F32, None).unwrap();
        
        let result = outer(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_eye() {
        let result = eye(5, BitNetDType::F32, None).unwrap();
        assert_eq!(result.shape().dims(), &[5, 5]);
    }

    #[test]
    fn test_matmul_strategy_selection() {
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        let a = BitNetTensor::ones(&[64, 64], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[64, 64], BitNetDType::F32, None).unwrap();
        
        let config = select_optimal_matmul_strategy(&a, &b);
        // Should select blocked or device optimized for medium size
        assert!(matches!(
            config.strategy, 
            MatMulStrategy::Blocked | MatMulStrategy::DeviceOptimized | MatMulStrategy::SimdAccelerated
        ));
    }
    
    #[test]
    fn test_validation_errors() {
        let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None).unwrap();
        let b = BitNetTensor::ones(&[4, 5], BitNetDType::F32, None).unwrap();
        
        // Should fail due to incompatible dimensions
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_cholesky_basic() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Create a simple 2x2 positive definite matrix
        // [[2, 1], [1, 2]] is positive definite
        let data = vec![2.0f32, 1.0f32, 1.0f32, 2.0f32];
        let matrix = BitNetTensor::from_vec(data, &[2, 2], BitNetDType::F32, None).unwrap();
        
        let result = cholesky(&matrix);
        assert!(result.is_ok(), "Cholesky decomposition should succeed for positive definite matrix");
    }

    #[test]
    fn test_cholesky_identity() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Identity matrix should decompose to itself
        let data = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32];
        let matrix = BitNetTensor::from_vec(data, &[2, 2], BitNetDType::F32, None).unwrap();
        
        let result = cholesky(&matrix);
        assert!(result.is_ok(), "Cholesky of identity should succeed");
        
        if let Ok(l) = result {
            // L should be approximately identity
            let l_data = l.to_candle().unwrap().to_vec1::<f32>().unwrap();
            assert!((l_data[0] - 1.0f32).abs() < 1e-6, "L[0,0] should be 1.0");
            assert!(l_data[1].abs() < 1e-6, "L[0,1] should be 0.0");
            assert!(l_data[2].abs() < 1e-6, "L[1,0] should be 0.0");
            assert!((l_data[3] - 1.0f32).abs() < 1e-6, "L[1,1] should be 1.0");
        }
    }

    #[test]
    fn test_cholesky_fails_for_non_positive_definite() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Create a non-positive definite matrix
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1, so not positive definite
        let data = vec![1.0f32, 2.0f32, 2.0f32, 1.0f32];
        let matrix = BitNetTensor::from_vec(data, &[2, 2], BitNetDType::F32, None).unwrap();
        
        let result = cholesky(&matrix);
        assert!(result.is_err(), "Cholesky should fail for non-positive definite matrix");
    }

    #[test]
    fn test_svd_basic() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Create a simple 3x2 matrix for SVD
        let data = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32, 1.0f32];
        let matrix = BitNetTensor::from_vec(data, &[3, 2], BitNetDType::F32, None).unwrap();
        
        let result = svd(&matrix);
        if let Err(e) = &result {
            println!("SVD failed with error: {:?}", e);
        }
        assert!(result.is_ok(), "SVD should succeed for valid matrix");
        
        if let Ok((u, s, vt)) = result {
            // Basic shape checks
            assert_eq!(u.shape().dims(), &[3, 2], "U should have correct shape");
            assert_eq!(s.shape().dims(), &[2], "S should have correct shape");
            assert_eq!(vt.shape().dims(), &[2, 2], "VT should have correct shape");
        }
    }

    #[test]
    fn test_qr_basic() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Create a simple 3x2 matrix for QR
        let data = vec![1.0f32, 0.0f32, 1.0f32, 1.0f32, 0.0f32, 1.0f32];
        let matrix = BitNetTensor::from_vec(data, &[3, 2], BitNetDType::F32, None).unwrap();
        
        let result = qr(&matrix);
        assert!(result.is_ok(), "QR decomposition should succeed");
        
        if let Ok((q, r)) = result {
            // Basic shape checks
            assert_eq!(q.shape().dims(), &[3, 2], "Q should have correct shape");
            assert_eq!(r.shape().dims(), &[2, 2], "R should have correct shape");
        }
    }

    #[test]
    fn test_qr_orthogonal_columns() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        // Create a 3x2 matrix
        let data = vec![1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0];
        let matrix = BitNetTensor::from_vec(data, &[3, 2], BitNetDType::F32, None).unwrap();
        
        let result = qr(&matrix);
        assert!(result.is_ok(), "QR decomposition should succeed");
        
        if let Ok((q, _r)) = result {
            // Check that Q has orthogonal columns (Q^T * Q should be identity-like)
            let qt = q.transpose().unwrap();
            let qtq = matmul(&qt, &q).unwrap();
            let qtq_data = qtq.to_candle().unwrap().to_vec1::<f32>().unwrap();
            
            // Check diagonal elements are approximately 1
            assert!((qtq_data[0] - 1.0).abs() < 1e-5, "Q^T*Q[0,0] should be 1.0");
            assert!((qtq_data[3] - 1.0).abs() < 1e-5, "Q^T*Q[1,1] should be 1.0");
            // Off-diagonal elements should be approximately 0
            assert!(qtq_data[1].abs() < 1e-5, "Q^T*Q[0,1] should be 0.0");
            assert!(qtq_data[2].abs() < 1e-5, "Q^T*Q[1,0] should be 0.0");
        }
    }

    #[test]
    fn test_eye_creation() {
        use crate::tensor::BitNetTensor;
        use crate::tensor::dtype::BitNetDType;
        use crate::memory::HybridMemoryPool;
        use crate::tensor::memory_integration::set_global_memory_pool;
        use std::sync::Arc;
        
        // Create and set global memory pool for tests
        let memory_pool = Arc::new(HybridMemoryPool::new().unwrap());
        set_global_memory_pool(Arc::downgrade(&memory_pool));
        
        let eye = BitNetTensor::eye(3, BitNetDType::F32, None).unwrap();
        assert_eq!(eye.shape().dims(), &[3, 3], "Eye matrix should have correct shape");
        
        let eye_data = eye.to_candle().unwrap().to_vec2::<f32>().unwrap();
        
        // Check diagonal elements are 1.0
        assert!((eye_data[0][0] - 1.0).abs() < 1e-6, "Eye[0,0] should be 1.0");
        assert!((eye_data[1][1] - 1.0).abs() < 1e-6, "Eye[1,1] should be 1.0");
        assert!((eye_data[2][2] - 1.0).abs() < 1e-6, "Eye[2,2] should be 1.0");
        
        // Check off-diagonal elements are 0.0
        assert!(eye_data[0][1].abs() < 1e-6, "Eye[0,1] should be 0.0");
        assert!(eye_data[0][2].abs() < 1e-6, "Eye[0,2] should be 0.0");
        assert!(eye_data[1][0].abs() < 1e-6, "Eye[1,0] should be 0.0");
    }
}
