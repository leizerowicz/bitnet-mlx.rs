//! BitNet Tensor Reduction Operations
//!
//! This module provides comprehensive statistical reduction operations for BitNet tensors,
//! including sum, mean, variance, standard deviation, min, max, and other statistical functions.
//! All operations support axis-specific reductions with keepdims functionality for maintaining
//! dimensional structure.
//!
//! # Features
//!
//! - **Statistical Operations**: Complete set of statistical reduction functions
//! - **Axis-Specific Reductions**: Reduce along specific dimensions or all dimensions
//! - **KeepDims Support**: Maintain tensor dimensionality after reduction
//! - **Memory Efficient**: Leverage existing HybridMemoryPool for intermediate results
//! - **Device Aware**: Optimized execution across CPU and GPU devices
//! - **Numerical Stability**: Numerically stable implementations for variance/std calculations
//! - **Broadcasting Compatible**: Results compatible with broadcasting operations
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::tensor::{BitNetTensor, reduction::*};
//!
//! let tensor = BitNetTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], BitNetDType::F32, None)?;
//!
//! // Global reductions
//! let sum = sum(&tensor, None, false)?;           // Scalar result
//! let mean = mean(&tensor, None, false)?;         // Scalar result
//!
//! // Axis-specific reductions
//! let sum_axis0 = sum(&tensor, Some(&[0]), false)?;  // Reduce along axis 0
//! let mean_keepdims = mean(&tensor, Some(&[1]), true)?; // Keep dimensions
//!
//! // Statistical operations
//! let variance = var(&tensor, Some(&[0]), false, 1)?;  // Sample variance (ddof=1)
//! let std_dev = std(&tensor, None, false, 0)?;         // Population std (ddof=0)
//! ```

use candle_core::Tensor as CandleTensor;

use crate::tensor::ops::{TensorOpError, TensorOpResult};
use crate::tensor::{BitNetDType, BitNetTensor};

// Tracing macros - no-op if tracing is not enabled
macro_rules! debug {
    ($($t:tt)*) => {};
}
macro_rules! warn {
    ($($t:tt)*) => {};
}

/// Reduction axis specification
#[derive(Debug, Clone)]
pub enum ReductionAxis {
    /// Reduce over all dimensions (scalar result)
    All,
    /// Reduce over specific axes
    Axes(Vec<usize>),
}

impl From<Option<&[usize]>> for ReductionAxis {
    fn from(axes: Option<&[usize]>) -> Self {
        match axes {
            None => ReductionAxis::All,
            Some(ax) => ReductionAxis::Axes(ax.to_vec()),
        }
    }
}

/// Configuration for reduction operations
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    /// Axes to reduce over
    pub axis: ReductionAxis,
    /// Whether to keep reduced dimensions as size 1
    pub keepdims: bool,
    /// Degrees of freedom for variance/std calculations
    pub ddof: i32,
}

impl ReductionConfig {
    /// Create a new reduction configuration
    pub fn new(axis: Option<&[usize]>, keepdims: bool, ddof: i32) -> Self {
        Self {
            axis: axis.into(),
            keepdims,
            ddof,
        }
    }

    /// Validate axis indices against tensor shape
    pub fn validate_axes(&self, shape: &[usize]) -> TensorOpResult<()> {
        match &self.axis {
            ReductionAxis::All => Ok(()),
            ReductionAxis::Axes(axes) => {
                for &axis in axes {
                    if axis >= shape.len() {
                        return Err(TensorOpError::InvalidTensor {
                            operation: "reduction_axis_validation".to_string(),
                            reason: format!(
                                "Axis {} out of bounds for tensor with {} dimensions",
                                axis,
                                shape.len()
                            ),
                        });
                    }
                }
                Ok(())
            }
        }
    }

    /// Calculate the output shape after reduction
    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        match &self.axis {
            ReductionAxis::All => {
                if self.keepdims {
                    vec![1; input_shape.len()]
                } else {
                    vec![]
                }
            }
            ReductionAxis::Axes(axes) => {
                let mut output_shape = input_shape.to_vec();
                for &axis in axes.iter().rev() {
                    if self.keepdims {
                        output_shape[axis] = 1;
                    } else {
                        output_shape.remove(axis);
                    }
                }
                output_shape
            }
        }
    }
}

/// Sum reduction operation
///
/// Computes the sum of tensor elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the sum along specified axes
///
/// # Examples
///
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], BitNetDType::F32, None)?;
/// let total_sum = sum(&tensor, None, false)?;       // Scalar sum
/// let axis_sum = sum(&tensor, Some(&[0]), true)?;   // Sum along axis 0 with keepdims
/// ```

pub fn sum(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing sum reduction with config: {:?}", config);

    // Get the underlying candle tensor
    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "sum_reduction".to_string(),
        error: e.to_string(),
    })?;

    // Perform the reduction
    let result_candle = match &config.axis {
        ReductionAxis::All => {
            let sum = candle_tensor
                .sum_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "sum_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                // Reshape to maintain original rank
                let shape = vec![1; tensor.shape().rank()];
                sum.reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "sum_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                sum
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = candle_tensor.clone();
            // Sort axes in descending order to avoid index shifting
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.sum(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("sum_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "sum_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    // Create output tensor with the result
    let _output_shape = config.output_shape(tensor.shape().dims());
    let output_tensor =
        BitNetTensor::from_candle(result_candle, &tensor.device()).map_err(|e| {
            TensorOpError::InternalError {
                reason: format!("Failed to create output tensor from sum result: {}", e),
            }
        })?;

    debug!("Sum reduction completed. Output shape: {:?}", output_shape);
    Ok(output_tensor)
}

/// Mean (average) reduction operation
///
/// Computes the arithmetic mean of tensor elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the mean along specified axes

pub fn mean(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing mean reduction with config: {:?}", config);

    // Get the underlying candle tensor
    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "mean_reduction".to_string(),
        error: e.to_string(),
    })?;

    // Perform the reduction
    let result_candle = match &config.axis {
        ReductionAxis::All => {
            let mean = candle_tensor
                .mean_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "mean_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                mean.reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "mean_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                mean
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = candle_tensor.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.mean(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("mean_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "mean_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    let output_tensor =
        BitNetTensor::from_candle(result_candle, &tensor.device()).map_err(|e| {
            TensorOpError::InternalError {
                reason: format!("Failed to create output tensor from mean result: {}", e),
            }
        })?;

    debug!("Mean reduction completed");
    Ok(output_tensor)
}

/// Minimum value reduction operation
///
/// Finds the minimum value of tensor elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the minimum values along specified axes

pub fn min(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing min reduction with config: {:?}", config);

    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "min_reduction".to_string(),
        error: e.to_string(),
    })?;

    let result_candle = match &config.axis {
        ReductionAxis::All => {
            let min_val = candle_tensor
                .min_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "min_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                min_val
                    .reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "min_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                min_val
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = candle_tensor.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.min(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("min_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "min_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    let output_tensor =
        BitNetTensor::from_candle(result_candle, &tensor.device()).map_err(|e| {
            TensorOpError::InternalError {
                reason: format!("Failed to create output tensor from min result: {}", e),
            }
        })?;

    debug!("Min reduction completed");
    Ok(output_tensor)
}

/// Maximum value reduction operation
///
/// Finds the maximum value of tensor elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the maximum values along specified axes

pub fn max(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing max reduction with config: {:?}", config);

    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "max_reduction".to_string(),
        error: e.to_string(),
    })?;

    let result_candle = match &config.axis {
        ReductionAxis::All => {
            let max_val = candle_tensor
                .max_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "max_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                max_val
                    .reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "max_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                max_val
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = candle_tensor.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.max(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("max_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "max_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    let output_tensor =
        BitNetTensor::from_candle(result_candle, &tensor.device()).map_err(|e| {
            TensorOpError::InternalError {
                reason: format!("Failed to create output tensor from max result: {}", e),
            }
        })?;

    debug!("Max reduction completed");
    Ok(output_tensor)
}

/// Variance reduction operation
///
/// Computes the variance of tensor elements along specified axes using
/// numerically stable algorithm.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
///
/// # Returns
///
/// Tensor containing the variance along specified axes
///
/// # Examples
///
/// ```rust
/// let tensor = BitNetTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], BitNetDType::F32, None)?;
/// let pop_var = var(&tensor, None, false, 0)?;     // Population variance
/// let sample_var = var(&tensor, None, false, 1)?;  // Sample variance
/// ```

pub fn var(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
    ddof: i32,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, ddof);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing variance reduction with config: {:?}", config);

    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "var_reduction".to_string(),
        error: e.to_string(),
    })?;

    // Compute mean first for numerically stable variance calculation
    let mean_tensor = mean(tensor, axis, keepdims)?;
    let mean_candle = mean_tensor
        .to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "var_mean_extraction".to_string(),
            error: e.to_string(),
        })?;

    // Compute (x - mean)^2
    let diff =
        candle_tensor
            .broadcast_sub(&mean_candle)
            .map_err(|e| TensorOpError::CandleError {
                operation: "var_mean_subtraction".to_string(),
                error: e.to_string(),
            })?;

    let squared_diff = diff.sqr().map_err(|e| TensorOpError::CandleError {
        operation: "var_squared_differences".to_string(),
        error: e.to_string(),
    })?;

    // Sum the squared differences
    let sum_squared = match &config.axis {
        ReductionAxis::All => {
            let sum = squared_diff
                .sum_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "var_sum_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                sum.reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "var_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                sum
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = squared_diff;
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.sum(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("var_sum_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "var_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    // Calculate the number of elements used in the calculation
    let n_elements = match &config.axis {
        ReductionAxis::All => tensor.shape().dims().iter().product::<usize>() as f64,
        ReductionAxis::Axes(axes) => axes
            .iter()
            .map(|&axis| tensor.shape().dims()[axis])
            .product::<usize>() as f64,
    };

    // Apply degrees of freedom correction
    let corrected_n = (n_elements - ddof as f64).max(1.0);
    let correction_scalar = CandleTensor::from_vec(vec![1.0f32 / corrected_n as f32], &[], tensor.device())
        .map_err(|e| TensorOpError::CandleError {
        operation: "var_correction_scalar".to_string(),
        error: e.to_string(),
    })?;

    let variance =
        sum_squared
            .broadcast_mul(&correction_scalar)
            .map_err(|e| TensorOpError::CandleError {
                operation: "var_final_division".to_string(),
                error: e.to_string(),
            })?;

    let output_tensor = BitNetTensor::from_candle(variance, &tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create output tensor from variance result: {}", e),
        }
    })?;

    debug!("Variance reduction completed with ddof={}", ddof);
    Ok(output_tensor)
}

/// Standard deviation reduction operation
///
/// Computes the standard deviation of tensor elements along specified axes.
/// This is the square root of the variance.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
///
/// # Returns
///
/// Tensor containing the standard deviation along specified axes

pub fn std(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
    ddof: i32,
) -> TensorOpResult<BitNetTensor> {
    debug!(
        "Computing standard deviation (sqrt of variance) with ddof={}",
        ddof
    );

    // Compute variance first
    let variance_tensor = var(tensor, axis, keepdims, ddof)?;

    // Take square root of variance
    let variance_candle = variance_tensor
        .to_candle()
        .map_err(|e| TensorOpError::CandleError {
            operation: "std_variance_extraction".to_string(),
            error: e.to_string(),
        })?;

    let std_candle = variance_candle
        .sqrt()
        .map_err(|e| TensorOpError::CandleError {
            operation: "std_sqrt".to_string(),
            error: e.to_string(),
        })?;

    let output_tensor = BitNetTensor::from_candle(std_candle, &tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create output tensor from std result: {}", e),
        }
    })?;

    debug!("Standard deviation reduction completed");
    Ok(output_tensor)
}

/// Median reduction operation (placeholder for future implementation)
///
/// # Note
/// This is a placeholder implementation as candle doesn't provide direct median support.
/// Future implementation would require custom median calculation algorithms.
pub fn median(
    _tensor: &BitNetTensor,
    _axis: Option<&[usize]>,
    _keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    Err(TensorOpError::UnsupportedOperation {
        operation: "median".to_string(),
        dtype: BitNetDType::F32,
    })
}

/// Quantile reduction operation (placeholder for future implementation)
///
/// # Note
/// This is a placeholder implementation. Future versions will support quantile calculations.
pub fn quantile(
    _tensor: &BitNetTensor,
    _q: f32,
    _axis: Option<&[usize]>,
    _keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    Err(TensorOpError::UnsupportedOperation {
        operation: "quantile".to_string(),
        dtype: BitNetDType::F32,
    })
}

/// Product reduction operation
///
/// Computes the product of tensor elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the product along specified axes
///
/// # Warning
///
/// Product operations can easily overflow for large tensors or large values.
/// Consider using log-sum-exp for numerical stability if dealing with large products.

pub fn prod(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!("Computing product reduction with config: {:?}", config);

    // Note: Candle doesn't have direct product operations, so we implement using log-exp
    // This provides better numerical stability
    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "prod_reduction".to_string(),
        error: e.to_string(),
    })?;

    // Check for negative values - if found, we need to handle sign separately
    let _has_negative = candle_tensor
        .lt(&CandleTensor::zeros_like(&candle_tensor)?)
        .map_err(|e| TensorOpError::CandleError {
            operation: "prod_negative_check".to_string(),
            error: e.to_string(),
        })?;

    // For now, implement a simple version that works for positive values
    // TODO: Implement full product with sign handling for negative values
    warn!("Product operation currently optimized for positive values. Negative value handling is basic.");

    // Use log-sum-exp approach for numerical stability
    let log_tensor = candle_tensor
        .abs()
        .map_err(|e| TensorOpError::CandleError {
            operation: "prod_abs".to_string(),
            error: e.to_string(),
        })?
        .log()
        .map_err(|e| TensorOpError::CandleError {
            operation: "prod_log".to_string(),
            error: e.to_string(),
        })?;

    let log_sum = match &config.axis {
        ReductionAxis::All => {
            let sum = log_tensor
                .sum_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "prod_sum_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                sum.reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "prod_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                sum
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = log_tensor;
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.sum(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("prod_sum_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "prod_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    // Convert back from log space
    let product = log_sum.exp().map_err(|e| TensorOpError::CandleError {
        operation: "prod_exp".to_string(),
        error: e.to_string(),
    })?;

    let output_tensor = BitNetTensor::from_candle(product, &tensor.device()).map_err(|e| {
        TensorOpError::InternalError {
            reason: format!("Failed to create output tensor from product result: {}", e),
        }
    })?;

    debug!("Product reduction completed (positive values optimized)");
    Ok(output_tensor)
}

/// Count non-zero elements reduction operation
///
/// Counts the number of non-zero elements along specified axes.
///
/// # Arguments
///
/// * `tensor` - Input tensor to reduce
/// * `axis` - Axes to reduce along (None for all axes)
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
///
/// Tensor containing the count of non-zero elements along specified axes

pub fn count_nonzero(
    tensor: &BitNetTensor,
    axis: Option<&[usize]>,
    keepdims: bool,
) -> TensorOpResult<BitNetTensor> {
    let config = ReductionConfig::new(axis, keepdims, 0);
    config.validate_axes(tensor.shape().dims())?;

    debug!(
        "Computing count_nonzero reduction with config: {:?}",
        config
    );

    let candle_tensor = tensor.to_candle().map_err(|e| TensorOpError::CandleError {
        operation: "count_nonzero_reduction".to_string(),
        error: e.to_string(),
    })?;

    // Create a mask of non-zero elements
    let zero =
        CandleTensor::zeros_like(&candle_tensor).map_err(|e| TensorOpError::CandleError {
            operation: "count_nonzero_zeros".to_string(),
            error: e.to_string(),
        })?;

    let nonzero_mask = candle_tensor
        .ne(&zero)
        .map_err(|e| TensorOpError::CandleError {
            operation: "count_nonzero_mask".to_string(),
            error: e.to_string(),
        })?;

    // Convert boolean mask to numbers (true = 1, false = 0) and sum
    let count_tensor =
        nonzero_mask
            .to_dtype(candle_tensor.dtype())
            .map_err(|e| TensorOpError::CandleError {
                operation: "count_nonzero_to_dtype".to_string(),
                error: e.to_string(),
            })?;

    let result_candle = match &config.axis {
        ReductionAxis::All => {
            let count = count_tensor
                .sum_all()
                .map_err(|e| TensorOpError::CandleError {
                    operation: "count_nonzero_sum_all".to_string(),
                    error: e.to_string(),
                })?;

            if keepdims {
                let shape = vec![1; tensor.shape().rank()];
                count
                    .reshape(shape.as_slice())
                    .map_err(|e| TensorOpError::CandleError {
                        operation: "count_nonzero_reshape_keepdims".to_string(),
                        error: e.to_string(),
                    })?
            } else {
                count
            }
        }
        ReductionAxis::Axes(axes) => {
            let mut result = count_tensor;
            let mut sorted_axes = axes.clone();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

            for &axis in &sorted_axes {
                result = result.sum(axis).map_err(|e| TensorOpError::CandleError {
                    operation: format!("count_nonzero_sum_axis_{}", axis),
                    error: e.to_string(),
                })?;

                if keepdims {
                    let mut shape = result.dims().to_vec();
                    shape.insert(axis, 1);
                    result = result.reshape(shape.as_slice()).map_err(|e| {
                        TensorOpError::CandleError {
                            operation: "count_nonzero_reshape_axis_keepdims".to_string(),
                            error: e.to_string(),
                        }
                    })?;
                }
            }
            result
        }
    };

    let output_tensor =
        BitNetTensor::from_candle(result_candle, &tensor.device()).map_err(|e| {
            TensorOpError::InternalError {
                reason: format!(
                    "Failed to create output tensor from count_nonzero result: {}",
                    e
                ),
            }
        })?;

    debug!("Count non-zero reduction completed");
    Ok(output_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
    use crate::tensor::memory_integration::set_global_memory_pool;
    use crate::tensor::{BitNetDType, BitNetTensor};
    use candle_core::Device;
    use std::sync::{Arc, Once};

    /// Ensures the global memory pool is initialized once for all tests
    fn setup_global_memory_pool() {
        use std::sync::OnceLock;
        static INIT: Once = Once::new();
        static MEMORY_POOL_HOLDER: OnceLock<Arc<HybridMemoryPool>> = OnceLock::new();
        
        INIT.call_once(|| {
            let mut config = MemoryPoolConfig::default();
            config.tracking_config = Some(TrackingConfig::detailed());

            let pool = Arc::new(
                HybridMemoryPool::with_config(config).expect("Failed to create test memory pool"),
            );

            // Store the Arc to keep it alive
            let _ = MEMORY_POOL_HOLDER.set(pool.clone());

            // Set as global pool
            set_global_memory_pool(Arc::downgrade(&pool));
        });
    }

    #[test]
    fn test_sum_reduction() {
        setup_global_memory_pool();
        let tensor = BitNetTensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
            &[2, 3],
            BitNetDType::F32,
            Some(Device::Cpu),
        )
        .unwrap();

        // Test global sum
        let global_sum = sum(&tensor, None, false).unwrap();
        assert_eq!(global_sum.shape().dims(), &[] as &[usize]);

        // Test axis sum
        let axis_sum = sum(&tensor, Some(&[0]), false).unwrap();
        assert_eq!(axis_sum.shape().dims(), &[3usize]);

        // Test keepdims
        let keepdims_sum = sum(&tensor, Some(&[0]), true).unwrap();
        assert_eq!(keepdims_sum.shape().dims(), &[1usize, 3usize]);
    }

    #[test]
    fn test_mean_reduction() {
        setup_global_memory_pool();
        let tensor = BitNetTensor::from_vec(
            vec![2.0f32, 4.0f32, 6.0f32, 8.0f32],
            &[2, 2],
            BitNetDType::F32,
            Some(Device::Cpu),
        )
        .unwrap();

        let mean_result = mean(&tensor, None, false).unwrap();
        assert_eq!(mean_result.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_var_std_reduction() {
        setup_global_memory_pool();
        let tensor = BitNetTensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            BitNetDType::F32,
            Some(Device::Cpu),
        )
        .unwrap();

        let variance = var(&tensor, None, false, 0).unwrap();
        let std_dev = std(&tensor, None, false, 0).unwrap();

        assert_eq!(variance.shape().dims(), &[] as &[usize]);
        assert_eq!(std_dev.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_min_max_reduction() {
        setup_global_memory_pool();
        let tensor = BitNetTensor::from_vec(
            vec![3.0f32, 1.0f32, 4.0f32, 2.0f32],
            &[2, 2],
            BitNetDType::F32,
            Some(Device::Cpu),
        )
        .unwrap();

        let min_val = min(&tensor, None, false).unwrap();
        let max_val = max(&tensor, None, false).unwrap();

        assert_eq!(min_val.shape().dims(), &[] as &[usize]);
        assert_eq!(max_val.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_reduction_axis_validation() {
        setup_global_memory_pool();
        let tensor = BitNetTensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            BitNetDType::F32,
            Some(Device::Cpu),
        )
        .unwrap();

        // Test invalid axis
        let result = sum(&tensor, Some(&[5]), false);
        assert!(result.is_err());
    }
}
