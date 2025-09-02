//! Quantized Tensor Implementation
//!
//! This module provides the core QuantizedTensor struct that integrates with BitNet's
//! tensor system while providing efficient storage and manipulation of quantized data
//! with full integration with the existing memory management and device systems.

use candle_core::{DType, Device, Tensor, Tensor as CandleTensor};
use std::sync::Arc;

use bitnet_core::{
    BitNetDType, BitNetTensor, HybridMemoryPool, MemoryError, MemoryResult, TensorShape,
};

use bitnet_core::tensor::shape::BroadcastCompatible;

use crate::quantization::{QuantizationPrecision, QuantizationStrategy};

use super::{TensorIntegrationError, TensorIntegrationResult};

use super::bitnet_ops::{QuantizedArithmetic, TernaryArithmetic, TernaryTensorRepresentation};

/// Configuration for quantized tensor creation and operations
#[derive(Debug, Clone)]
pub struct QuantizedTensorConfig {
    /// Quantization precision
    pub precision: QuantizationPrecision,

    /// Quantization strategy
    pub strategy: QuantizationStrategy,

    /// Target device for tensor operations
    pub device: Option<Device>,

    /// Use memory pool for allocations
    pub use_memory_pool: bool,

    /// Enable data compression for storage
    pub enable_compression: bool,

    /// Compression threshold (minimum compression ratio to enable)
    pub compression_threshold: f32,
}

impl Default for QuantizedTensorConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            device: None,
            use_memory_pool: true,
            enable_compression: true,
            compression_threshold: 0.1,
        }
    }
}

/// Scale and zero-point parameters for quantization
#[derive(Debug, Clone)]
pub struct ScaleZeroPoint {
    /// Scaling factor for dequantization
    pub scale: f32,

    /// Zero-point offset
    pub zero_point: f32,

    /// Quantization range (min, max)
    pub quantization_range: (f32, f32),

    /// Whether the quantization is symmetric (zero_point = 0)
    pub is_symmetric: bool,
}

impl ScaleZeroPoint {
    /// Create symmetric quantization parameters
    pub fn symmetric(scale: f32, quantization_range: (f32, f32)) -> Self {
        Self {
            scale,
            zero_point: 0.0,
            quantization_range,
            is_symmetric: true,
        }
    }

    /// Create asymmetric quantization parameters
    pub fn asymmetric(scale: f32, zero_point: f32, quantization_range: (f32, f32)) -> Self {
        Self {
            scale,
            zero_point,
            quantization_range,
            is_symmetric: false,
        }
    }

    /// Quantize a value using these parameters
    pub fn quantize(&self, value: f32) -> f32 {
        let quantized = value / self.scale + self.zero_point;
        quantized.clamp(self.quantization_range.0, self.quantization_range.1)
    }

    /// Dequantize a value using these parameters
    pub fn dequantize(&self, quantized_value: f32) -> f32 {
        (quantized_value - self.zero_point) * self.scale
    }
}

/// Complete quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParameters {
    /// Scale and zero-point information
    pub scale_zero_point: ScaleZeroPoint,

    /// Quantization precision used
    pub precision: QuantizationPrecision,

    /// Quantization strategy used
    pub strategy: QuantizationStrategy,

    /// Original tensor data type
    pub originaldtype: BitNetDType,

    /// Quantized tensor data type
    pub quantizeddtype: DType,

    /// Compression ratio achieved
    pub compression_ratio: f32,

    /// Number of quantization levels
    pub num_levels: usize,
}

impl QuantizationParameters {
    /// Create parameters for 1.58-bit (ternary) quantization
    pub fn ternary(alpha: f32, originaldtype: BitNetDType) -> Self {
        Self {
            scale_zero_point: ScaleZeroPoint::symmetric(alpha, (-1.0, 1.0)),
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            originaldtype,
            quantizeddtype: DType::I64,
            compression_ratio: 32.0 / 1.58, // Approximate compression from f32
            num_levels: 3,                  // {-1, 0, 1}
        }
    }

    /// Create parameters for general quantization
    pub fn general(
        precision: QuantizationPrecision,
        strategy: QuantizationStrategy,
        scale: f32,
        zero_point: f32,
        originaldtype: BitNetDType,
    ) -> Self {
        let (quantizeddtype, num_levels, compression_ratio) = match precision {
            QuantizationPrecision::OneFiveFiveBit => (DType::I64, 3, 32.0 / 1.58),
            QuantizationPrecision::OneBit => (DType::I64, 2, 32.0),
            QuantizationPrecision::TwoBit => (DType::I64, 4, 16.0),
            QuantizationPrecision::FourBit => (DType::I64, 16, 8.0),
            QuantizationPrecision::EightBit => (DType::I64, 256, 4.0),
        };

        let quantization_range = match precision {
            QuantizationPrecision::OneFiveFiveBit => (-1.0, 1.0),
            QuantizationPrecision::OneBit => (-1.0, 1.0),
            QuantizationPrecision::TwoBit => (-2.0, 1.0),
            QuantizationPrecision::FourBit => (-8.0, 7.0),
            QuantizationPrecision::EightBit => (-128.0, 127.0),
        };

        let scale_zero_point = match strategy {
            QuantizationStrategy::Symmetric => ScaleZeroPoint::symmetric(scale, quantization_range),
            _ => ScaleZeroPoint::asymmetric(scale, zero_point, quantization_range),
        };

        Self {
            scale_zero_point,
            precision,
            strategy,
            originaldtype,
            quantizeddtype,
            compression_ratio,
            num_levels,
        }
    }
}

/// Dequantization strategy for different use cases
#[derive(Debug, Clone, Copy)]
pub enum DequantizationStrategy {
    /// Always dequantize to full precision
    AlwaysFullPrecision,

    /// Dequantize only when necessary
    OnDemand,

    /// Keep quantized for as long as possible
    LazyDequantization,

    /// Mixed strategy based on operation type
    Adaptive,
}

/// Errors specific to quantized tensor operations
#[derive(Debug, thiserror::Error)]
pub enum QuantizedTensorError {
    #[error("Quantization parameter mismatch: {message}")]
    ParameterMismatch { message: String },

    #[error("Unsupported dequantization strategy: {strategy:?}")]
    UnsupportedDequantization { strategy: DequantizationStrategy },

    #[error("Storage format error: {message}")]
    StorageFormat { message: String },

    #[error("Compression error: {message}")]
    Compression { message: String },

    #[error("Layout incompatibility: {message}")]
    LayoutIncompatibility { message: String },
}

/// Storage backend for quantized tensor data
#[derive(Debug, Clone)]
pub enum QuantizedStorage {
    /// Dense storage for quantized values
    Dense {
        data: BitNetTensor,
        parameters: QuantizationParameters,
    },

    /// Sparse storage for highly quantized data
    Sparse {
        indices: BitNetTensor,
        values: BitNetTensor,
        shape: TensorShape,
        parameters: QuantizationParameters,
    },

    /// Packed storage for maximum compression
    Packed {
        packed_data: Vec<u8>,
        shape: TensorShape,
        parameters: QuantizationParameters,
        unpacking_info: PackingInfo,
    },

    /// Ternary-specific storage
    Ternary {
        ternary_repr: TernaryTensorRepresentation,
        parameters: QuantizationParameters,
    },
}

/// Information for unpacking packed quantized data
#[derive(Debug, Clone)]
pub struct PackingInfo {
    /// Number of values packed per byte
    pub values_per_byte: usize,

    /// Bit width of each quantized value
    pub bits_per_value: usize,

    /// Padding bits at the end
    pub padding_bits: usize,

    /// Endianness for multi-byte values
    pub is_big_endian: bool,
}

/// Memory layout information for quantized tensors
#[derive(Debug, Clone)]
pub struct QuantizedLayout {
    /// Storage backend type
    pub storage_type: QuantizedStorageType,

    /// Memory alignment requirements
    pub alignment: usize,

    /// Stride information
    pub strides: Vec<usize>,

    /// Whether the layout is contiguous
    pub is_contiguous: bool,

    /// Total memory footprint in bytes
    pub memory_footprint: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizedStorageType {
    Dense,
    Sparse,
    Packed,
    Ternary,
}

/// Compression ratio information
#[derive(Debug, Clone)]
pub struct CompressionRatio {
    /// Original size in bytes
    pub original_size: usize,

    /// Compressed size in bytes
    pub compressed_size: usize,

    /// Compression ratio (original / compressed)
    pub ratio: f32,

    /// Space savings as percentage
    pub space_savings_percent: f32,
}

impl CompressionRatio {
    pub fn new(original_size: usize, compressed_size: usize) -> Self {
        let ratio = original_size as f32 / compressed_size.max(1) as f32;
        let space_savings_percent = (1.0 - (compressed_size as f32 / original_size as f32)) * 100.0;

        Self {
            original_size,
            compressed_size,
            ratio,
            space_savings_percent,
        }
    }
}

/// Main quantized tensor implementation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Storage backend for quantized data
    storage: QuantizedStorage,

    /// Configuration used for quantization
    config: QuantizedTensorConfig,

    /// Memory layout information
    layout: QuantizedLayout,

    /// Dequantization strategy
    dequant_strategy: DequantizationStrategy,

    /// Device placement
    device: Device,

    /// Unique tensor identifier
    tensor_id: u64,

    /// Optional memory pool reference
    memory_pool: Option<Arc<HybridMemoryPool>>,
}

impl QuantizedTensor {
    /// Create quantized tensor from BitNet tensor
    pub fn from_bitnet_tensor(
        tensor: BitNetTensor,
        config: QuantizedTensorConfig,
    ) -> TensorIntegrationResult<Self> {
        let device = config
            .device
            .clone()
            .unwrap_or_else(|| tensor.device().clone());
        let tensor_id = rand::random::<u64>();

        // Determine quantization parameters
        let parameters = Self::compute_quantization_parameters(&tensor, &config)?;

        // Create storage based on precision
        let storage = match config.precision {
            QuantizationPrecision::OneFiveFiveBit => {
                Self::create_ternary_storage(tensor, parameters, &config)?
            }
            _ => Self::create_general_storage(tensor, parameters, &config)?,
        };

        // Compute layout information
        let layout = Self::compute_layout(&storage)?;

        Ok(Self {
            storage,
            config,
            layout,
            dequant_strategy: DequantizationStrategy::OnDemand,
            device,
            tensor_id,
            memory_pool: None,
        })
    }

    /// Create quantized tensor from ternary representation
    pub fn from_ternary_representation(
        ternary: TernaryTensorRepresentation,
        config: QuantizedTensorConfig,
    ) -> TensorIntegrationResult<Self> {
        let device = config
            .device
            .clone()
            .unwrap_or_else(|| ternary.device.clone());
        let tensor_id = rand::random::<u64>();

        let parameters = QuantizationParameters::ternary(
            ternary.quantization_params.alpha,
            ternary.originaldtype,
        );

        let storage = QuantizedStorage::Ternary {
            ternary_repr: ternary,
            parameters,
        };

        let layout = Self::compute_layout(&storage)?;

        Ok(Self {
            storage,
            config,
            layout,
            dequant_strategy: DequantizationStrategy::OnDemand,
            device,
            tensor_id,
            memory_pool: None,
        })
    }

    /// Convert back to BitNet tensor (dequantize)
    pub fn to_bitnet_tensor(&self) -> TensorIntegrationResult<BitNetTensor> {
        match &self.storage {
            QuantizedStorage::Dense { data, parameters } => self.dequantize_dense(data, parameters),
            QuantizedStorage::Sparse {
                indices,
                values,
                shape,
                parameters,
            } => self.dequantize_sparse(indices, values, shape, parameters),
            QuantizedStorage::Packed {
                packed_data,
                shape,
                parameters,
                unpacking_info,
            } => self.dequantize_packed(packed_data, shape, parameters, unpacking_info),
            QuantizedStorage::Ternary { ternary_repr, .. } => self.dequantize_ternary(ternary_repr),
        }
    }

    /// Get quantization precision
    pub fn precision(&self) -> QuantizationPrecision {
        self.config.precision
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get tensor shape
    pub fn shape(&self) -> &TensorShape {
        match &self.storage {
            QuantizedStorage::Dense { data, .. } => data.shape(),
            QuantizedStorage::Sparse { shape, .. } => shape,
            QuantizedStorage::Packed { shape, .. } => shape,
            QuantizedStorage::Ternary { ternary_repr, .. } => &ternary_repr.original_shape,
        }
    }

    /// Check if shapes are compatible for operations
    pub fn is_shapes_compatible(&self, other: &Self) -> bool {
        self.shape().is_broadcast_compatible(other.shape())
    }

    /// Get ternary representation if available
    pub fn get_ternary_representation(&self) -> Option<&TernaryTensorRepresentation> {
        match &self.storage {
            QuantizedStorage::Ternary { ternary_repr, .. } => Some(ternary_repr),
            _ => None,
        }
    }

    /// Get compression ratio information
    pub fn compression_ratio(&self) -> CompressionRatio {
        let original_size = match &self.storage {
            QuantizedStorage::Dense { parameters, .. } => {
                self.shape().total_elements() * Self::dtype_size(parameters.originaldtype)
            }
            QuantizedStorage::Sparse {
                shape, parameters, ..
            } => shape.total_elements() * Self::dtype_size(parameters.originaldtype),
            QuantizedStorage::Packed {
                shape, parameters, ..
            } => shape.total_elements() * Self::dtype_size(parameters.originaldtype),
            QuantizedStorage::Ternary { ternary_repr, .. } => {
                ternary_repr.original_shape.total_elements()
                    * Self::dtype_size(ternary_repr.originaldtype)
            }
        };

        CompressionRatio::new(original_size, self.layout.memory_footprint)
    }

    fn compute_quantization_parameters(
        tensor: &BitNetTensor,
        config: &QuantizedTensorConfig,
    ) -> TensorIntegrationResult<QuantizationParameters> {
        let candle_tensor =
            tensor
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get tensor data: {e}"),
                })?;

        match config.precision {
            QuantizationPrecision::OneFiveFiveBit => {
                let alpha = candle_tensor
                    .abs()
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to compute abs: {e}"),
                    })?
                    .mean_all()
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to compute mean: {e}"),
                    })?
                    .to_scalar::<f32>()
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to extract scalar: {e}"),
                    })?;

                Ok(QuantizationParameters::ternary(alpha, tensor.dtype()))
            }
            _ => {
                // For other precisions, compute scale and zero-point
                let (min_val, max_val) = Self::compute_tensor_range(&candle_tensor)?;
                let (scale, zero_point) =
                    Self::compute_scale_zero_point(min_val, max_val, config.precision);

                Ok(QuantizationParameters::general(
                    config.precision,
                    config.strategy,
                    scale,
                    zero_point,
                    tensor.dtype(),
                ))
            }
        }
    }

    fn compute_tensor_range(tensor: &CandleTensor) -> TensorIntegrationResult<(f32, f32)> {
        let min_val = tensor
            .min(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute min: {e}"),
            })?
            .min_keepdim(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute global min: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract min scalar: {e}"),
            })?;

        let max_val = tensor
            .max(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute max: {e}"),
            })?
            .max_keepdim(0)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute global max: {e}"),
            })?
            .to_scalar::<f32>()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to extract max scalar: {e}"),
            })?;

        Ok((min_val, max_val))
    }

    fn compute_scale_zero_point(
        min_val: f32,
        max_val: f32,
        precision: QuantizationPrecision,
    ) -> (f32, f32) {
        let (qmin, qmax) = match precision {
            QuantizationPrecision::EightBit => (-128.0, 127.0),
            QuantizationPrecision::FourBit => (-8.0, 7.0),
            QuantizationPrecision::TwoBit => (-2.0, 1.0),
            QuantizationPrecision::OneBit => (-1.0, 1.0),
            QuantizationPrecision::OneFiveFiveBit => (-1.0, 1.0),
        };

        let scale = (max_val - min_val) / (qmax - qmin);
        let zero_point = qmin - min_val / scale;

        (scale, zero_point)
    }

    fn create_ternary_storage(
        tensor: BitNetTensor,
        parameters: QuantizationParameters,
        _config: &QuantizedTensorConfig,
    ) -> TensorIntegrationResult<QuantizedStorage> {
        // This would typically use the BitNetTensorOps to create ternary representation
        // For now, create a placeholder that would integrate with bitnet_ops
        let candle_tensor =
            tensor
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get tensor data: {e}"),
                })?;

        // Apply ternary quantization (simplified)
        let alpha = parameters.scale_zero_point.scale;
        let sign_tensor = candle_tensor
            .sign()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to compute sign: {e}"),
            })?;

        let values = sign_tensor.clone();

        let scales = Tensor::new(vec![alpha], candle_tensor.device())
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to create scale tensor: {e}"),
            })?;

        let ternary_repr = TernaryTensorRepresentation {
            values: BitNetTensor::from_candle(values, &candle_tensor.device())?,
            scales: BitNetTensor::from_candle(scales, &candle_tensor.device())?,
            original_shape: tensor.shape().clone(),
            quantization_params: super::bitnet_ops::TernaryQuantizationParams {
                method: crate::quantization::TernaryMethod::DetSTE,
                threshold: alpha * 0.1,
                scale: alpha,
                clipping_range: (-1.0, 1.0),
                nnz_count: 0,
                alpha,
            },
            device: tensor.device().clone(),
            originaldtype: tensor.dtype(),
        };

        Ok(QuantizedStorage::Ternary {
            ternary_repr,
            parameters,
        })
    }

    fn create_general_storage(
        tensor: BitNetTensor,
        parameters: QuantizationParameters,
        _config: &QuantizedTensorConfig,
    ) -> TensorIntegrationResult<QuantizedStorage> {
        // Apply general quantization
        let candle_tensor =
            tensor
                .to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get tensor data: {e}"),
                })?;

        // Quantize using scale and zero-point
        let scale_tensor = candle_tensor
            .ones_like()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to create scale tensor: {e}"),
            })?
            .mul(
                &candle_core::Tensor::new(parameters.scale_zero_point.scale, tensor.device())
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to create scale scalar: {e}"),
                    })?,
            )
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to scale tensor: {e}"),
            })?;

        let quantized = candle_tensor
            .div(&scale_tensor)
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to apply quantization: {e}"),
            })?
            .round()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to round quantized values: {e}"),
            })?;

        let _quantized_tensor = quantized.clone();

        Ok(QuantizedStorage::Dense {
            data: BitNetTensor::from_candle(quantized, &candle_tensor.device())?,
            parameters,
        })
    }

    fn compute_layout(storage: &QuantizedStorage) -> TensorIntegrationResult<QuantizedLayout> {
        match storage {
            QuantizedStorage::Dense { data, parameters } => {
                let memory_footprint =
                    data.shape().total_elements() * Self::dtype_size(parameters.originaldtype);
                Ok(QuantizedLayout {
                    storage_type: QuantizedStorageType::Dense,
                    alignment: 64,
                    strides: data
                        .shape()
                        .compute_strides()
                        .iter()
                        .map(|&s| s as usize)
                        .collect(),
                    is_contiguous: true,
                    memory_footprint,
                })
            }
            QuantizedStorage::Ternary { ternary_repr, .. } => {
                let memory_footprint = ternary_repr.values.shape().total_elements()
                    + ternary_repr.scales.shape().total_elements();
                Ok(QuantizedLayout {
                    storage_type: QuantizedStorageType::Ternary,
                    alignment: 64,
                    strides: ternary_repr
                        .original_shape
                        .compute_strides()
                        .iter()
                        .map(|&s| s as usize)
                        .collect(),
                    is_contiguous: true,
                    memory_footprint,
                })
            }
            _ => {
                // Simplified for other storage types
                Ok(QuantizedLayout {
                    storage_type: QuantizedStorageType::Dense,
                    alignment: 64,
                    strides: vec![],
                    is_contiguous: true,
                    memory_footprint: 0,
                })
            }
        }
    }

    fn dequantize_dense(
        &self,
        data: &BitNetTensor,
        parameters: &QuantizationParameters,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let candle_tensor =
            data.to_candle_tensor()
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to get quantized data: {e}"),
                })?;

        let scale_tensor = candle_tensor
            .ones_like()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to create scale tensor: {e}"),
            })?
            .mul(
                &candle_core::Tensor::new(parameters.scale_zero_point.scale, &self.device)
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to create scale scalar: {e}"),
                    })?,
            )
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to scale tensor: {e}"),
            })?;

        let dequantized =
            candle_tensor
                .mul(&scale_tensor)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to dequantize: {e}"),
                })?;

        BitNetTensor::from_candle_tensor(dequantized, self.device.clone())
            .map_err(TensorIntegrationError::Memory)
    }

    fn dequantize_ternary(
        &self,
        ternary_repr: &TernaryTensorRepresentation,
    ) -> TensorIntegrationResult<BitNetTensor> {
        let values = ternary_repr.values.to_candle_tensor().map_err(|e| {
            TensorIntegrationError::TensorOp {
                message: format!("Failed to get ternary values: {e}"),
            }
        })?;

        let scales = ternary_repr.scales.to_candle_tensor().map_err(|e| {
            TensorIntegrationError::TensorOp {
                message: format!("Failed to get scale values: {e}"),
            }
        })?;

        let dequantized =
            values
                .broadcast_mul(&scales)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to dequantize ternary: {e}"),
                })?;

        BitNetTensor::from_candle_tensor(dequantized, self.device.clone())
            .map_err(TensorIntegrationError::Memory)
    }

    fn dequantize_sparse(
        &self,
        _indices: &BitNetTensor,
        _values: &BitNetTensor,
        _shape: &TensorShape,
        _parameters: &QuantizationParameters,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Placeholder for sparse dequantization
        Err(TensorIntegrationError::UnsupportedOperation {
            operation: "Sparse dequantization".to_string(),
            precision: self.config.precision,
        })
    }

    fn dequantize_packed(
        &self,
        _packed_data: &Vec<u8>,
        _shape: &TensorShape,
        _parameters: &QuantizationParameters,
        _unpacking_info: &PackingInfo,
    ) -> TensorIntegrationResult<BitNetTensor> {
        // Placeholder for packed dequantization
        Err(TensorIntegrationError::UnsupportedOperation {
            operation: "Packed dequantization".to_string(),
            precision: self.config.precision,
        })
    }

    fn dtype_size(dtype: BitNetDType) -> usize {
        match dtype {
            BitNetDType::F32 => 4,
            BitNetDType::F16 => 2,
            BitNetDType::I32 => 4,
            BitNetDType::I16 => 2,
            BitNetDType::I8 => 1,
            BitNetDType::U32 => 4,
            BitNetDType::U16 => 2,
            BitNetDType::U8 => 1,
            BitNetDType::Bool => 1,
            _ => 1, // Default for BitNet-specific types and future additions
        }
    }

    /// Perform quantized matrix multiplication
    pub fn quantized_matmul(&self, other: &Self) -> TensorIntegrationResult<Self> {
        // Use the TernaryArithmetic trait implementation
        self.ternary_matmul(other)
    }

    /// Get the data type of the quantized tensor
    pub fn dtype(&self) -> BitNetDType {
        match &self.storage {
            QuantizedStorage::Dense { parameters, .. } => parameters.originaldtype,
            QuantizedStorage::Ternary { ternary_repr, .. } => ternary_repr.originaldtype,
            _ => BitNetDType::F32, // Default fallback
        }
    }
}

impl QuantizedArithmetic for QuantizedTensor {
    type Error = TensorIntegrationError;
    fn quantized_add(&self, other: &Self) -> TensorIntegrationResult<Self> {
        if !self.is_shapes_compatible(other) {
            return Err(TensorIntegrationError::ShapeMismatch {
                message: format!(
                    "Cannot add tensors with shapes {:?} and {:?}",
                    self.shape(),
                    other.shape()
                ),
            });
        }

        match (&self.storage, &other.storage) {
            (
                QuantizedStorage::Dense {
                    data: lhs,
                    parameters: lhs_params,
                },
                QuantizedStorage::Dense {
                    data: rhs,
                    parameters: rhs_params,
                },
            ) => self.add_dense_tensors(lhs, rhs, lhs_params, rhs_params),
            _ => {
                // Fallback: dequantize, add, requantize
                let lhs_full = self.to_bitnet_tensor()?;
                let rhs_full = other.to_bitnet_tensor()?;
                let result_full =
                    lhs_full
                        .add(&rhs_full)
                        .map_err(|e| TensorIntegrationError::TensorOp {
                            message: format!("Failed to add dequantized tensors: {e}"),
                        })?;

                Self::from_bitnet_tensor(result_full, self.config.clone())
            }
        }
    }

    fn quantized_mul(&self, other: &Self) -> TensorIntegrationResult<Self> {
        // Similar to add but for multiplication
        let lhs_full = self.to_bitnet_tensor()?;
        let rhs_full = other.to_bitnet_tensor()?;
        let result_full =
            lhs_full
                .mul(&rhs_full)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to multiply dequantized tensors: {e}"),
                })?;

        Self::from_bitnet_tensor(result_full, self.config.clone())
    }

    fn quantized_matmul(&self, other: &Self) -> TensorIntegrationResult<Self> {
        // Matrix multiplication - typically requires dequantization for general case
        let lhs_full = self.to_bitnet_tensor()?;
        let rhs_full = other.to_bitnet_tensor()?;
        let result_full =
            lhs_full
                .matmul(&rhs_full)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to perform matrix multiplication: {e}"),
                })?;

        Self::from_bitnet_tensor(result_full, self.config.clone())
    }

    fn quantized_scale(&self, scalar: f32) -> TensorIntegrationResult<Self> {
        let full_tensor = self.to_bitnet_tensor()?;
        let scaled =
            full_tensor
                .mul_scalar(scalar)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to scale tensor: {e}"),
                })?;

        Self::from_bitnet_tensor(scaled, self.config.clone())
    }
}

impl TernaryArithmetic for QuantizedTensor {
    type Error = TensorIntegrationError;
    fn ternary_add(&self, other: &Self) -> TensorIntegrationResult<Self> {
        if let (Some(lhs_ternary), Some(rhs_ternary)) = (
            self.get_ternary_representation(),
            other.get_ternary_representation(),
        ) {
            // Ternary addition using lookup table for {-1, 0, 1} + {-1, 0, 1}
            let lhs_values = lhs_ternary.values.to_candle_tensor().map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to get LHS ternary values: {e}"),
                }
            })?;

            let rhs_values = rhs_ternary.values.to_candle_tensor().map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to get RHS ternary values: {e}"),
                }
            })?;

            // Add ternary values (result may be outside {-1, 0, 1})
            let result_values =
                lhs_values
                    .add(&rhs_values)
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to add ternary values: {e}"),
                    })?;

            // Clamp to ternary range and rescale
            let clamped_values =
                result_values
                    .clamp(-1.0, 1.0)
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to clamp ternary result: {e}"),
                    })?;

            let result_tensor =
                BitNetTensor::from_candle_tensor(clamped_values, self.device.clone())
                    .map_err(TensorIntegrationError::Memory)?;

            // Combine scales
            let combined_scale = (lhs_ternary.quantization_params.scale
                + rhs_ternary.quantization_params.scale)
                / 2.0;
            let scales = BitNetTensor::from_scalar(combined_scale, self.device.clone())
                .map_err(TensorIntegrationError::Memory)?;

            let result_ternary = TernaryTensorRepresentation {
                values: result_tensor,
                scales,
                original_shape: self.shape().clone(),
                quantization_params: super::bitnet_ops::TernaryQuantizationParams {
                    method: lhs_ternary.quantization_params.method,
                    threshold: (lhs_ternary.quantization_params.threshold
                        + rhs_ternary.quantization_params.threshold)
                        / 2.0,
                    scale: combined_scale,
                    clipping_range: lhs_ternary.quantization_params.clipping_range,
                    nnz_count: 0,
                    alpha: combined_scale,
                },
                device: self.device.clone(),
                originaldtype: self.dtype(),
            };

            Self::from_ternary_representation(result_ternary, self.config.clone())
        } else {
            // Fallback to regular quantized addition
            self.quantized_add(other)
        }
    }

    fn ternary_mul(&self, other: &Self) -> TensorIntegrationResult<Self> {
        if let (Some(lhs_ternary), Some(rhs_ternary)) = (
            self.get_ternary_representation(),
            other.get_ternary_representation(),
        ) {
            // Ternary multiplication is very efficient for {-1, 0, 1}
            let lhs_values = lhs_ternary.values.to_candle_tensor().map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to get LHS ternary values: {e}"),
                }
            })?;

            let rhs_values = rhs_ternary.values.to_candle_tensor().map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to get RHS ternary values: {e}"),
                }
            })?;

            // Multiply ternary values (result is still in {-1, 0, 1})
            let result_values =
                lhs_values
                    .mul(&rhs_values)
                    .map_err(|e| TensorIntegrationError::TensorOp {
                        message: format!("Failed to multiply ternary values: {e}"),
                    })?;

            let result_tensor =
                BitNetTensor::from_candle_tensor(result_values, self.device.clone())
                    .map_err(TensorIntegrationError::Memory)?;

            // Multiply scales
            let combined_scale =
                lhs_ternary.quantization_params.scale * rhs_ternary.quantization_params.scale;
            let scales = BitNetTensor::from_scalar(combined_scale, self.device.clone())
                .map_err(TensorIntegrationError::Memory)?;

            let result_ternary = TernaryTensorRepresentation {
                values: result_tensor,
                scales,
                original_shape: self.shape().clone(),
                quantization_params: super::bitnet_ops::TernaryQuantizationParams {
                    method: lhs_ternary.quantization_params.method,
                    threshold: (lhs_ternary.quantization_params.threshold
                        + rhs_ternary.quantization_params.threshold)
                        / 2.0,
                    scale: combined_scale,
                    clipping_range: lhs_ternary.quantization_params.clipping_range,
                    nnz_count: 0,
                    alpha: combined_scale,
                },
                device: self.device.clone(),
                originaldtype: self.dtype(),
            };

            Self::from_ternary_representation(result_ternary, self.config.clone())
        } else {
            // Fallback to regular quantized multiplication
            self.quantized_mul(other)
        }
    }

    fn ternary_matmul(&self, other: &Self) -> TensorIntegrationResult<Self> {
        // For matrix multiplication with ternary tensors
        if let (Some(lhs_ternary), Some(rhs_ternary)) = (
            self.get_ternary_representation(),
            other.get_ternary_representation(),
        ) {
            // Perform matmul on the ternary values
            let result_values = lhs_ternary
                .values
                .matmul(&rhs_ternary.values)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to perform matmul: {e}"),
                })?;

            // Combine scales for matrix multiplication
            let combined_scale = lhs_ternary.scales.mul(&rhs_ternary.scales).map_err(|e| {
                TensorIntegrationError::TensorOp {
                    message: format!("Failed to multiply scales: {e}"),
                }
            })?;

            // Determine output shape for matmul
            let output_shape = TensorShape::new(&[
                lhs_ternary.original_shape.dims()[0],
                rhs_ternary.original_shape.dims()[rhs_ternary.original_shape.dims().len() - 1],
            ]);

            let result_ternary = TernaryTensorRepresentation {
                values: result_values,
                scales: combined_scale,
                original_shape: output_shape,
                quantization_params: super::bitnet_ops::TernaryQuantizationParams {
                    method: lhs_ternary.quantization_params.method,
                    threshold: (lhs_ternary.quantization_params.threshold
                        + rhs_ternary.quantization_params.threshold)
                        / 2.0,
                    scale: 1.0,
                    clipping_range: lhs_ternary.quantization_params.clipping_range,
                    nnz_count: 0,
                    alpha: 1.0,
                },
                device: self.device.clone(),
                originaldtype: self.dtype(),
            };

            Self::from_ternary_representation(result_ternary, self.config.clone())
        } else {
            // Fallback to dequantization for non-ternary
            self.dequantize_compute_requantize(|lhs| {
                let rhs_full =
                    other
                        .to_bitnet_tensor()
                        .map_err(|e| MemoryError::InternalError {
                            reason: e.to_string(),
                        })?;
                lhs.matmul(&rhs_full)
                    .map_err(|e| MemoryError::InternalError {
                        reason: e.to_string(),
                    })
            })
        }
    }

    fn dequantize_compute_requantize<F>(&self, operation: F) -> TensorIntegrationResult<Self>
    where
        F: FnOnce(&BitNetTensor) -> MemoryResult<BitNetTensor>,
        Self: Sized,
    {
        let full_tensor = self.to_bitnet_tensor()?;
        let result = operation(&full_tensor).map_err(TensorIntegrationError::Memory)?;

        Self::from_bitnet_tensor(result, self.config.clone())
    }

    fn sparsity_ratio(&self) -> f32 {
        if let Some(ternary) = self.get_ternary_representation() {
            let total_elements = ternary.original_shape.total_elements();
            let nnz = ternary.quantization_params.nnz_count;
            1.0 - (nnz as f32 / total_elements as f32)
        } else {
            0.0
        }
    }

    fn count_nonzero(&self) -> usize {
        if let Some(ternary) = self.get_ternary_representation() {
            ternary.quantization_params.nnz_count
        } else {
            // For non-ternary, estimate based on storage
            self.shape().total_elements()
        }
    }
}

impl QuantizedTensor {
    fn add_dense_tensors(
        &self,
        lhs: &BitNetTensor,
        rhs: &BitNetTensor,
        _lhs_params: &QuantizationParameters,
        _rhs_params: &QuantizationParameters,
    ) -> TensorIntegrationResult<Self> {
        // Simplified dense addition - in practice would need proper scale handling
        let lhs_tensor = lhs
            .to_candle_tensor()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to get LHS tensor: {e}"),
            })?;

        let rhs_tensor = rhs
            .to_candle_tensor()
            .map_err(|e| TensorIntegrationError::TensorOp {
                message: format!("Failed to get RHS tensor: {e}"),
            })?;

        let result_tensor =
            lhs_tensor
                .add(&rhs_tensor)
                .map_err(|e| TensorIntegrationError::TensorOp {
                    message: format!("Failed to add tensors: {e}"),
                })?;

        let result_bitnet = BitNetTensor::from_candle_tensor(result_tensor, self.device.clone())
            .map_err(TensorIntegrationError::Memory)?;

        Self::from_bitnet_tensor(result_bitnet, self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_tensor_config_default() {
        let config = QuantizedTensorConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.strategy, QuantizationStrategy::Symmetric);
        assert!(config.use_memory_pool);
        assert!(config.enable_compression);
    }

    #[test]
    fn test_scale_zero_point_symmetric() {
        let szp = ScaleZeroPoint::symmetric(0.5, (-1.0, 1.0));
        assert_eq!(szp.scale, 0.5);
        assert_eq!(szp.zero_point, 0.0);
        assert!(szp.is_symmetric);
    }

    #[test]
    fn test_scale_zero_point_quantization() {
        let szp = ScaleZeroPoint::symmetric(0.1, (-1.0, 1.0));

        let quantized = szp.quantize(0.05);
        assert!((quantized - 0.5).abs() < 1e-6);

        let dequantized = szp.dequantize(0.5);
        assert!((dequantized - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = CompressionRatio::new(1000, 250);
        assert_eq!(ratio.ratio, 4.0);
        assert_eq!(ratio.space_savings_percent, 75.0);
    }

    #[test]
    fn test_quantization_parameters_ternary() {
        let params = QuantizationParameters::ternary(0.5, BitNetDType::F32);
        assert_eq!(params.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(params.num_levels, 3);
        assert!(params.scale_zero_point.is_symmetric);
        assert_eq!(params.scale_zero_point.scale, 0.5);
    }
}
