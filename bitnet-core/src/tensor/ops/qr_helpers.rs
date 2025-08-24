//! Helper functions for QR decomposition implementation
//!
//! This module provides utility functions needed for the Modified Gram-Schmidt
//! QR decomposition algorithm in the linear algebra operations.

use candle_core::Tensor as CandleTensor;
use super::{TensorOpResult, TensorOpError};

/// Extract a column from a Candle tensor
pub fn extract_column(tensor: &CandleTensor, col: usize) -> TensorOpResult<CandleTensor> {
    tensor.narrow(1, col, 1)
        .and_then(|t| t.squeeze(1))
        .map_err(|e| TensorOpError::CandleError {
            operation: "extract_column".to_string(),
            error: e.to_string(),
        })
}

/// Compute dot product of two Candle tensors
pub fn dot_product_candle(a: &CandleTensor, b: &CandleTensor) -> TensorOpResult<f32> {
    let result = (a * b)?
        .sum_all()?
        .to_scalar::<f32>()
        .map_err(|e| TensorOpError::CandleError {
            operation: "dot_product".to_string(),
            error: e.to_string(),
        })?;
    Ok(result)
}

/// Subtract scaled vector from another vector
pub fn subtract_scaled_candle(a: &CandleTensor, b: &CandleTensor, scale: f32) -> TensorOpResult<CandleTensor> {
    let scaled_b = b.affine(scale as f64, 0.0)?;
    (a - scaled_b).map_err(|e| TensorOpError::CandleError {
        operation: "subtract_scaled".to_string(),
        error: e.to_string(),
    })
}

/// Compute the norm of a Candle tensor
pub fn compute_norm_candle(tensor: &CandleTensor) -> TensorOpResult<f32> {
    let squared = (tensor * tensor)?;
    let sum = squared.sum_all()?;
    let norm = sum.sqrt()?.to_scalar::<f32>()
        .map_err(|e| TensorOpError::CandleError {
            operation: "compute_norm".to_string(),
            error: e.to_string(),
        })?;
    Ok(norm)
}

/// Scale a vector by a scalar value
pub fn scale_vector_candle(tensor: &CandleTensor, scale: f32) -> TensorOpResult<CandleTensor> {
    tensor.affine(scale as f64, 0.0).map_err(|e| TensorOpError::CandleError {
        operation: "scale_vector".to_string(),
        error: e.to_string(),
    })
}

/// Construct a matrix from a vector of column tensors
pub fn construct_matrix_from_columns(
    columns: &[CandleTensor],
    m: usize,
    n: usize,
) -> TensorOpResult<CandleTensor> {
    if columns.is_empty() {
        return Err(TensorOpError::InternalError {
            reason: "No columns provided".to_string(),
        });
    }

    // Stack columns to form matrix
    let stacked = CandleTensor::stack(columns, 1)
        .map_err(|e| TensorOpError::CandleError {
            operation: "construct_matrix".to_string(),
            error: e.to_string(),
        })?;

    Ok(stacked)
}

/// Construct upper triangular matrix from R matrix data
pub fn construct_upper_triangular_matrix(
    r_matrix: &[Vec<f32>],
    rows: usize,
    cols: usize,
) -> TensorOpResult<CandleTensor> {
    let mut data = vec![0.0f32; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            if j < r_matrix[i].len() {
                data[i * cols + j] = r_matrix[i][j];
            }
        }
    }

    let device = candle_core::Device::Cpu;
    CandleTensor::from_slice(&data, (rows, cols), &device)
        .map_err(|e| TensorOpError::CandleError {
            operation: "construct_upper_triangular_matrix".to_string(),
            error: e.to_string(),
        })
}

/// Get a specific element from a matrix tensor
pub fn get_matrix_element(tensor: &CandleTensor, row: usize, col: usize) -> TensorOpResult<f32> {
    tensor.get(row)
        .and_then(|r| r.get(col))
        .and_then(|e| e.to_scalar::<f32>())
        .map_err(|e| TensorOpError::CandleError {
            operation: "get_matrix_element".to_string(),
            error: e.to_string(),
        })
}
