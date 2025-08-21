//! MLX tensor operations for BitNet
//! 
//! This module provides MLX-accelerated tensor operations specifically
//! optimized for BitNet quantization and neural network operations.

#[cfg(feature = "mlx")]
use mlx_rs::{Array, Dtype as MlxDtype};

use crate::memory::MemoryHandle;
use crate::tensor::dtype::BitNetDType;
use crate::mlx::device::BitNetMlxDevice;
use anyhow::Result;

/// MLX tensor wrapper with BitNet integration
#[cfg(feature = "mlx")]
#[derive(Debug)]
pub struct MlxTensor {
    array: Array,
    device: BitNetMlxDevice,
    dtype: BitNetDType,
    memory_handle: Option<MemoryHandle>,
}

#[cfg(feature = "mlx")]
impl MlxTensor {
    /// Create a new MLX tensor
    pub fn new(
        array: Array,
        device: BitNetMlxDevice,
        dtype: BitNetDType,
    ) -> Self {
        Self {
            array,
            device,
            dtype,
            memory_handle: None,
        }
    }

    /// Create a new MLX tensor with memory tracking
    pub fn with_memory_handle(
        array: Array,
        device: BitNetMlxDevice,
        dtype: BitNetDType,
        handle: MemoryHandle,
    ) -> Self {
        let mut tensor = Self::new(array, device, dtype);
        tensor.memory_handle = Some(handle);
        tensor
    }

    /// Create a tensor from raw data
    pub fn from_data(
        data: &[f32],
        shape: &[usize],
        device: BitNetMlxDevice,
    ) -> Result<Self> {
        let mlx_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let array = Array::from_slice(data, &mlx_shape);
        
        Ok(Self::new(array, device, BitNetDType::F32))
    }

    /// Create a zeros tensor
    pub fn zeros(
        shape: &[usize],
        dtype: BitNetDType,
        device: BitNetMlxDevice,
    ) -> Result<Self> {
        let mlx_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let array = mlx_rs::ops::zeros::<f32>(&mlx_shape)?;
        
        Ok(Self::new(array, device, dtype))
    }

    /// Create a ones tensor
    pub fn ones(
        shape: &[usize],
        dtype: BitNetDType,
        device: BitNetMlxDevice,
    ) -> Result<Self> {
        let mlx_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let array = mlx_rs::ops::ones::<f32>(&mlx_shape)?;
        
        Ok(Self::new(array, device, dtype))
    }

    /// Create a random normal tensor
    pub fn randn(
        shape: &[usize],
        dtype: BitNetDType,
        device: BitNetMlxDevice,
    ) -> Result<Self> {
        let mlx_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        // For now, use ones and then we can improve this later with actual random generation
        let array = mlx_rs::ops::ones::<f32>(&mlx_shape)?;
        
        Ok(Self::new(array, device, dtype))
    }

    /// Get the underlying MLX array
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Get the device this tensor is on
    pub fn device(&self) -> &BitNetMlxDevice {
        &self.device
    }

    /// Get memory handle if available
    pub fn memory_handle(&self) -> Option<&MemoryHandle> {
        self.memory_handle.as_ref()
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.array.shape().iter().map(|&x| x as usize).collect()
    }

    /// Get tensor dtype
    pub fn dtype(&self) -> BitNetDType {
        self.dtype
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Move tensor to another device
    pub fn to_device(&self, device: &BitNetMlxDevice) -> Result<Self> {
        let new_array = self.array.clone(); // MLX arrays don't need explicit device transfer
        Ok(Self::new(new_array, device.clone(), self.dtype()))
    }

    /// Convert to CPU tensor
    pub fn to_cpu(&self) -> Result<Self> {
        let cpu_device = BitNetMlxDevice::cpu()?;
        self.to_device(&cpu_device)
    }

    /// Convert to GPU tensor (if available)
    pub fn to_gpu(&self) -> Result<Self> {
        let gpu_device = BitNetMlxDevice::gpu()?;
        self.to_device(&gpu_device)
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let mlx_shape: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        let reshaped = self.array.reshape(&mlx_shape)?;
        
        Ok(Self::new(reshaped, self.device.clone(), self.dtype()))
    }

    /// Get tensor data as Vec<f32> (copies to CPU if needed)
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let cpu_tensor = self.to_cpu()?;
        let data = cpu_tensor.array.as_slice::<f32>();
        Ok(data.to_vec())
    }

    /// Clone the tensor
    pub fn clone(&self) -> Self {
        Self::new(self.array.clone(), self.device.clone(), self.dtype)
    }
}

/// Extension trait for BitNetDType to convert to MLX dtype
#[cfg(feature = "mlx")]
impl BitNetDType {
    /// Convert BitNet dtype to MLX dtype
    pub fn to_mlx_dtype(&self) -> MlxDtype {
        match self {
            BitNetDType::F32 => MlxDtype::Float32,
            BitNetDType::F16 => MlxDtype::Float16,
            BitNetDType::BF16 => MlxDtype::Bfloat16,
            BitNetDType::I8 => MlxDtype::Int8,
            BitNetDType::I16 => MlxDtype::Int16,
            BitNetDType::I32 => MlxDtype::Int32,
            BitNetDType::I64 => MlxDtype::Int64,
            BitNetDType::U8 => MlxDtype::Uint8,
            BitNetDType::U16 => MlxDtype::Uint16,
            BitNetDType::U32 => MlxDtype::Uint32,
            BitNetDType::U64 => MlxDtype::Uint64,
            BitNetDType::Bool => MlxDtype::Bool,
            BitNetDType::QInt8 => MlxDtype::Int8, // Map to closest available
            BitNetDType::QInt4 => MlxDtype::Int8, // Map to closest available
            BitNetDType::Int4 => MlxDtype::Int8, // Map to closest available
            BitNetDType::BitNet158 => MlxDtype::Int8, // Map to closest available
            BitNetDType::BitNet11 => MlxDtype::Int8, // Map to closest available
            BitNetDType::BitNet1 => MlxDtype::Int8, // Map to closest available
        }
    }

    /// Convert from MLX dtype to BitNet dtype
    pub fn from_mlx_dtype(dtype: MlxDtype) -> Self {
        match dtype {
            MlxDtype::Float32 => BitNetDType::F32,
            MlxDtype::Float16 => BitNetDType::F16,
            MlxDtype::Bfloat16 => BitNetDType::BF16,
            MlxDtype::Int8 => BitNetDType::I8,
            _ => BitNetDType::F32, // Default fallback
        }
    }
}

/// MLX tensor operations
#[cfg(feature = "mlx")]
pub struct MlxTensorOps;

#[cfg(feature = "mlx")]
impl MlxTensorOps {
    /// Matrix multiplication
    pub fn matmul(a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        let result = mlx_rs::ops::matmul(a.array(), b.array())?;
        Ok(MlxTensor::new(result, a.device().clone(), a.dtype()))
    }

    /// Element-wise addition
    pub fn add(a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        let result = mlx_rs::ops::add(a.array(), b.array())?;
        Ok(MlxTensor::new(result, a.device().clone(), a.dtype()))
    }

    /// Element-wise multiplication
    pub fn mul(a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor> {
        let result = mlx_rs::ops::multiply(a.array(), b.array())?;
        Ok(MlxTensor::new(result, a.device().clone(), a.dtype()))
    }
}

// Stub implementations when MLX is not available
#[cfg(not(feature = "mlx"))]
pub struct MlxTensor;

#[cfg(not(feature = "mlx"))]
pub struct MlxTensorOps;

#[cfg(not(feature = "mlx"))]
impl MlxTensorOps {
    pub fn matmul(_a: &MlxTensor, _b: &MlxTensor) -> Result<MlxTensor> {
        anyhow::bail!("MLX support not compiled in")
    }
}