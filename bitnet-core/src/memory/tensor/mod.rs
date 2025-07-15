//! BitNet Tensor Lifecycle Management
//!
//! This module provides a comprehensive tensor lifecycle management system for BitNet,
//! featuring automatic reference counting, device-aware memory management, and seamless
//! integration with the candle-core tensor library.
//!
//! # Architecture
//!
//! The tensor system consists of several key components:
//!
//! - **BitNetTensor**: Core tensor type with reference counting and lifecycle management
//! - **TensorHandle**: Safe handle for accessing tensor data with automatic cleanup
//! - **BitNetDType**: Specialized data types for BitNet quantization
//! - **TensorMetadata**: Tracking information for tensor lifecycle and device placement
//!
//! # Features
//!
//! - Automatic reference counting with `Arc<BitNetTensor>`
//! - Weak references for cycle prevention
//! - Device-aware tensor migration (CPU ↔ Metal)
//! - Zero-copy conversions with candle tensors where possible
//! - Integration with the hybrid memory pool system
//! - Thread-safe tensor operations
//! - Automatic memory reclamation
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
//! use bitnet_core::device::{auto_select_device, get_cpu_device};
//! use bitnet_core::memory::HybridMemoryPool;
//!
//! // Create a memory pool and device
//! let pool = HybridMemoryPool::new()?;
//! let device = auto_select_device();
//!
//! // Create a BitNet tensor
//! let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool)?;
//!
//! // Convert to candle tensor for operations
//! let candle_tensor = tensor.to_candle()?;
//!
//! // Create from existing data
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = BitNetTensor::from_data(data, &[2, 2], &device, &pool)?;
//!
//! // Migrate to different device
//! let cpu_device = get_cpu_device();
//! let cpu_tensor = tensor.to_device(&cpu_device, &pool)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod dtype;
pub mod handle;
pub mod metadata;
pub mod tensor;

// Re-exports
pub use dtype::BitNetDType;
pub use handle::{TensorHandle, TensorData};
pub use metadata::TensorMetadata;
pub use tensor::BitNetTensor;