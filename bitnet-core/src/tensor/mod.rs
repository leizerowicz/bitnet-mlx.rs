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
//! use bitnet_core::tensor::{BitNetTensor, BitNetDType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let tensor = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, None)?;
//! println!("Created tensor with shape: {:?}", tensor.shape().dims());
//! # Ok(())
//! # }
//! ```

// Core tensor modules
pub mod core;
pub mod device_integration;
pub mod dtype;
pub mod memory_integration;
pub mod shape;
pub mod storage;

// Operations module
pub mod ops;

// Acceleration module
pub mod acceleration;

// Legacy compatibility for backward compatibility
pub mod legacy;

// Main exports
pub use core::{BitNetTensor, TensorMemoryStats};
pub use device_integration::TensorDeviceManager;
pub use dtype::BitNetDType;
pub use memory_integration::{
    clear_global_memory_pool, get_global_memory_pool, set_global_memory_pool, TensorMemoryManager,
};
pub use shape::{BroadcastCompatible, TensorShape};
pub use storage::TensorStorage;

// Operations exports
pub use ops::arithmetic::*;
pub use ops::broadcasting::*;

// Legacy function re-exports for backward compatibility
pub use legacy::{
    create_tensor_f32 as legacy_create_tensor_f32, create_tensor_i8 as legacy_create_tensor_i8,
    get_shape as legacy_get_shape, ones as legacy_ones, reshape as legacy_reshape,
    transpose as legacy_transpose, zeros as legacy_zeros,
};

// Direct exports of legacy functions for backward compatibility (these are the main exports)
pub use legacy::{
    create_tensor_f32,
    create_tensor_i8,
    get_shape,
    ones, // Legacy ones returns candle Tensor
    reshape,
    transpose,
    zeros, // Legacy zeros returns candle Tensor
};

// Convenience functions that wrap the new BitNetTensor API
use crate::memory::MemoryResult;

/// Creates a tensor filled with zeros using the new BitNetTensor API
///
/// # Arguments
///
/// * `shape` - Shape dimensions
/// * `dtype` - Data type
///
/// # Returns
///
/// Result containing new BitNetTensor filled with zeros
///
/// # Examples
///
/// ```rust
/// use bitnet_core::tensor::{zeros_bitnet, BitNetDType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tensor = zeros_bitnet(&[2, 3], BitNetDType::F32)?;
/// assert_eq!(tensor.shape().dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
pub fn zeros_bitnet(shape: &[usize], dtype: BitNetDType) -> MemoryResult<BitNetTensor> {
    BitNetTensor::zeros(shape, dtype, None)
}

/// Creates a tensor filled with ones using the new BitNetTensor API
pub fn ones_bitnet(shape: &[usize], dtype: BitNetDType) -> MemoryResult<BitNetTensor> {
    BitNetTensor::ones(shape, dtype, None)
}

/// Creates a tensor from f32 data using the new BitNetTensor API
pub fn from_f32_data(data: Vec<f32>, shape: &[usize]) -> MemoryResult<BitNetTensor> {
    BitNetTensor::from_vec(data, shape, BitNetDType::F32, None)
}

/// Creates a BitNet 1.58 quantized tensor
pub fn bitnet_158(shape: &[usize]) -> MemoryResult<BitNetTensor> {
    BitNetTensor::bitnet_158(shape, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
    use crate::tensor::memory_integration::set_global_memory_pool;
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
    fn test_tensor_module_integration() {
        setup_global_memory_pool();
        // Test that we can create tensors using the convenience functions
        let tensor = zeros_bitnet(&[2, 3], BitNetDType::F32).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
    }

    #[test]
    fn test_tensor_from_data() {
        setup_global_memory_pool();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = from_f32_data(data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.num_elements(), 4);
    }

    #[test]
    fn test_bitnet_158_tensor() {
        setup_global_memory_pool();
        let tensor = bitnet_158(&[10, 10]).unwrap();
        assert_eq!(tensor.dtype(), BitNetDType::BitNet158);
        assert_eq!(tensor.shape().dims(), &[10, 10]);
    }
}
