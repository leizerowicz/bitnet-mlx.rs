//! BitNet Core Library
//!
//! This crate provides the core functionality for BitNet implementation,
//! including tensor operations, quantization utilities, mixed precision support,
//! and fundamental data structures.

pub mod device;
pub mod error;
pub mod execution;
pub mod memory;
pub mod mixed_precision;
pub mod sequence;
pub mod tensor;
pub mod metal;
pub mod tokenizer;

// MLX support (Apple Silicon only)
#[cfg(feature = "mlx")]
pub mod mlx;

pub use device::*;
pub use error::*;
pub use execution::*;
pub use memory::*;
pub use mixed_precision::*;
pub use sequence::*;
pub use tensor::*;
pub use metal::*;
pub use tokenizer::*;

// MLX re-exports when feature is enabled
#[cfg(feature = "mlx")]
pub use mlx::*;

// Re-export commonly used types from candle
pub use candle_core::{Device, DType, Result, Tensor};

// Re-export BitNet tensor types for convenience
pub use memory::tensor::{BitNetTensor, BitNetDType, TensorHandle, TensorMetadata};

// Re-export mixed precision types for convenience
pub use mixed_precision::{
    MixedPrecisionConfig, LayerPrecisionConfig, ComponentPrecisionConfig,
    LayerType, ComponentType, MixedPrecisionStrategy, MixedPrecisionError,
    PrecisionManager, PrecisionConverter, LayerPrecisionManager,
    PrecisionValidator, PolicyEngine, PrecisionPolicy,
};

// MLX types when available
#[cfg(feature = "mlx")]
pub use mlx_rs::{Array as MlxArray, Device as MlxDevice};