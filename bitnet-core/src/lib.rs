//! BitNet Core Library
//! 
//! This crate provides the core functionality for BitNet implementation,
//! including tensor operations, quantization utilities, and fundamental
//! data structures.

pub mod device;
pub mod memory;
pub mod tensor;
pub mod metal;

// MLX support (Apple Silicon only)
#[cfg(feature = "mlx")]
pub mod mlx;

pub use device::*;
pub use memory::*;
pub use tensor::*;
pub use metal::*;

// MLX re-exports when feature is enabled
#[cfg(feature = "mlx")]
pub use mlx::*;

// Re-export commonly used types from candle
pub use candle_core::{Device, DType, Result, Tensor};

// Re-export BitNet tensor types for convenience
pub use memory::tensor::{BitNetTensor, BitNetDType, TensorHandle, TensorMetadata};

// MLX types when available
#[cfg(feature = "mlx")]
pub use mlx_rs::{Array as MlxArray, Device as MlxDevice};