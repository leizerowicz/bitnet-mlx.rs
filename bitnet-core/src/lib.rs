//! BitNet Core Library
//! 
//! This crate provides the core functionality for BitNet implementation,
//! including tensor operations, quantization utilities, and fundamental
//! data structures.

pub mod device;
pub mod memory;
pub mod tensor;

pub use device::*;
pub use memory::*;
pub use tensor::*;

// Re-export commonly used types from candle
pub use candle_core::{Device, DType, Result, Tensor};

// Re-export BitNet tensor types for convenience
pub use memory::tensor::{BitNetTensor, BitNetDType, TensorHandle, TensorMetadata};