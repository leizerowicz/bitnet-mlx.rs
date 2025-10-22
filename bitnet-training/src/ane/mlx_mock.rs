// bitnet-training/src/ane/mlx_mock.rs
//! Mock MLX types for ANE integration
//!
//! This module provides mock implementations of MLX types until
//! the actual bitnet_metal dependency is available.

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Mock MLX Array for compilation compatibility
#[derive(Debug, Clone)]
pub struct MLXArray {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl MLXArray {
    /// Create new MLX array
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get array data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Convert to tensor
    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(self.data.clone(), self.shape.as_slice(), &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("MLX to tensor conversion error: {}", e))
    }

    /// Create from tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape().dims().to_vec();
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self::new(data, shape))
    }
}

/// Mock MLX Device for compilation compatibility
#[derive(Debug, Clone)]
pub struct MLXDevice {
    device_id: usize,
}

impl MLXDevice {
    /// Create default MLX device
    pub fn default() -> Self {
        Self { device_id: 0 }
    }

    /// Create new MLX device with ID
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Check if MLX is available
    pub fn is_available() -> bool {
        // Mock implementation - would check actual MLX availability
        #[cfg(target_os = "macos")]
        {
            std::env::consts::ARCH == "aarch64"
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

/// Convert MLX array to tensor utility function
pub fn mlx_array_to_tensor(array: &MLXArray) -> Result<Tensor> {
    array.to_tensor()
}

/// Convert tensor to MLX array utility function
pub fn tensor_to_mlx_array(tensor: &Tensor) -> Result<MLXArray> {
    MLXArray::from_tensor(tensor)
}