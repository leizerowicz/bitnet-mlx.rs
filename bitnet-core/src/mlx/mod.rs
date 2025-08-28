//! MLX integration for BitNet on Apple Silicon
//!
//! This module provides MLX-specific implementations for BitNet operations,
//! leveraging Apple's MLX framework for high-performance machine learning
//! on Apple Silicon devices.

#[cfg(feature = "mlx")]
use mlx_rs::{Array, Dtype as MlxDtype};

use crate::memory::MemoryHandle;
use anyhow::Result;

#[cfg(feature = "mlx")]
use candle_core::Tensor;

pub mod device;
pub mod graph;
pub mod operations;
pub mod optimization;
pub mod tensor;

// Performance comparison tools
pub mod device_comparison;
pub mod memory_tracker;
pub mod metrics;
pub mod performance;
pub mod profiler;
pub mod regression_testing;
pub mod reports;

// Re-export performance comparison types
pub use device_comparison::{DeviceComparisonConfig, MlxDeviceComparison};
pub use memory_tracker::{
    track_allocation, track_deallocation, ImplementationEffort, MemoryEvent, MemoryEventType,
    MemoryOptimization, MlxMemoryTracker, OptimizationPriority, OptimizationType,
};
pub use metrics::{
    ExportFormat, MemoryMetrics, MetricsConfig, MlxMetrics, MlxMetricsCollector, OperationContext,
    SystemMetrics,
};
pub use performance::{
    BenchmarkConfig, ComparisonResult, MemoryUsage, MlxPerformanceBenchmarker, PerformanceMetrics,
};
pub use profiler::{MlxAdvancedProfiler, ProfileOutputFormat, ProfilerConfig};
pub use regression_testing::{MlxRegressionTester, RegressionTestConfig};
pub use reports::PerformanceReportGenerator;

// Re-export key MLX types
pub use device::BitNetMlxDevice;
pub use tensor::MlxTensor;

/// Convenience wrapper for MLX quantization - matches test expectations
#[cfg(feature = "mlx")]
pub fn mlx_quantize(array: &Array, scale: Option<f32>) -> Result<Array> {
    use crate::mlx::operations::BitNetMlxOps;

    // Convert Array to MlxTensor for processing
    let device = BitNetMlxDevice::cpu()?;
    let tensor = MlxTensor::new(
        array.clone(),
        device,
        crate::tensor::dtype::BitNetDType::F32,
    );

    // Perform quantization
    let quantized = BitNetMlxOps::quantize_1_58_bit(&tensor, scale)?;

    Ok(quantized.array().clone())
}

/// Convenience wrapper for MLX dequantization - matches test expectations
#[cfg(feature = "mlx")]
pub fn mlx_dequantize(array: &Array, scale: Option<f32>) -> Result<Array> {
    use crate::mlx::operations::BitNetMlxOps;

    // Convert Array to MlxTensor for processing
    let device = BitNetMlxDevice::cpu()?;
    let tensor = MlxTensor::new(
        array.clone(),
        device,
        crate::tensor::dtype::BitNetDType::F32,
    );

    // Perform dequantization
    let dequantized = BitNetMlxOps::dequantize_1_58_bit(&tensor, scale)?;

    Ok(dequantized.array().clone())
}
pub use operations::BitNetMlxOps;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod optimization_tests;

// Re-export the main device type from device module
// (removed duplicate import)

/// MLX tensor wrapper for BitNet operations
#[cfg(feature = "mlx")]
#[derive(Debug)]
#[allow(dead_code)]
pub struct BitNetMlxTensor {
    array: Array,
    device: device::BitNetMlxDevice,
    memory_handle: Option<MemoryHandle>,
}

#[cfg(feature = "mlx")]
impl BitNetMlxTensor {
    /// Create a new MLX tensor from an array
    pub fn new(array: Array, device: device::BitNetMlxDevice) -> Self {
        Self {
            array,
            device,
            memory_handle: None,
        }
    }

    /// Create a new MLX tensor with memory tracking
    pub fn with_memory_handle(array: Array, device: BitNetMlxDevice, handle: MemoryHandle) -> Self {
        Self {
            array,
            device,
            memory_handle: Some(handle),
        }
    }

    /// Get the underlying MLX array
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Get the device this tensor is on
    pub fn device(&self) -> &BitNetMlxDevice {
        &self.device
    }

    /// Get the memory handle if available
    pub fn memory_handle(&self) -> Option<&MemoryHandle> {
        self.memory_handle.as_ref()
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[i32] {
        self.array.shape()
    }

    /// Get tensor dtype
    pub fn dtype(&self) -> MlxDtype {
        self.array.dtype()
    }

    /// Move tensor to another device
    pub fn to_device(&self, device: &BitNetMlxDevice) -> Result<Self> {
        let new_array = self.array.clone(); // MLX arrays don't need explicit device transfer
        Ok(Self::new(new_array, device.clone()))
    }
}

/// MLX-specific operations for BitNet
#[cfg(feature = "mlx")]
pub struct MlxOperations;

#[cfg(feature = "mlx")]
impl MlxOperations {
    /// Perform 1.58-bit quantization using MLX
    pub fn quantize_1_58_bit(tensor: &BitNetMlxTensor) -> Result<BitNetMlxTensor> {
        // Placeholder for MLX-accelerated quantization
        // This would implement the actual 1.58-bit quantization algorithm
        let quantized = tensor.array().clone(); // Placeholder
        Ok(BitNetMlxTensor::new(quantized, tensor.device().clone()))
    }

    /// Perform BitLinear operation using MLX
    pub fn bitlinear(
        input: &BitNetMlxTensor,
        _weight: &BitNetMlxTensor,
        _bias: Option<&BitNetMlxTensor>,
    ) -> Result<BitNetMlxTensor> {
        // Placeholder for MLX-accelerated BitLinear
        // This would implement the actual BitLinear layer computation
        let output = input.array().clone(); // Placeholder
        Ok(BitNetMlxTensor::new(output, input.device().clone()))
    }

    /// Perform matrix multiplication using MLX
    pub fn matmul(a: &BitNetMlxTensor, b: &BitNetMlxTensor) -> Result<BitNetMlxTensor> {
        let result = mlx_rs::ops::matmul(a.array(), b.array())?;
        Ok(BitNetMlxTensor::new(result, a.device().clone()))
    }
}

/// Check if MLX is available on the current system
#[cfg(feature = "mlx")]
pub fn is_mlx_available() -> bool {
    // Check if we're on Apple Silicon and MLX is properly initialized
    cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
}

/// Get the default MLX device for the system
#[cfg(feature = "mlx")]
pub fn default_mlx_device() -> Result<BitNetMlxDevice> {
    if is_mlx_available() {
        // Try GPU first, fall back to CPU
        BitNetMlxDevice::gpu()
    } else {
        BitNetMlxDevice::cpu()
    }
}

/// MLX Array Utilities
///
/// These functions provide conversion utilities between MLX arrays and Candle tensors,
/// enabling seamless interoperability between the two tensor frameworks.

/// Create an MLX array from shape and data
///
/// # Arguments
/// * `shape` - The shape of the array as a slice of i32 values
/// * `data` - Vector of f32 values to populate the array
///
/// # Returns
/// A Result containing the created MLX Array or an error
///
/// # Example
/// ```
/// use bitnet_core::mlx::create_mlx_array;
///
/// let shape = &[2, 3];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let array = create_mlx_array(shape, data).unwrap();
/// ```
#[cfg(feature = "mlx")]
pub fn create_mlx_array(shape: &[i32], data: Vec<f32>) -> Result<Array> {
    // Validate that the data length matches the expected number of elements
    let expected_elements: usize = shape.iter().map(|&x| x as usize).product();
    if data.len() != expected_elements {
        return Err(anyhow::anyhow!(
            "Data length {} does not match expected elements {} for shape {:?}",
            data.len(),
            expected_elements,
            shape
        ));
    }

    // Create MLX array from the data and shape
    let array = Array::from_slice(&data, shape);
    Ok(array)
}

/// Convert an MLX array to a Candle tensor
///
/// # Arguments
/// * `array` - Reference to the MLX array to convert
///
/// # Returns
/// A Result containing the converted Candle Tensor or an error
///
/// # Example
/// ```
/// use bitnet_core::mlx::{create_mlx_array, mlx_to_candle_tensor};
///
/// let shape = &[2, 2];
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let array = create_mlx_array(shape, data).unwrap();
/// let tensor = mlx_to_candle_tensor(&array).unwrap();
/// ```
#[cfg(feature = "mlx")]
pub fn mlx_to_candle_tensor(array: &Array) -> Result<Tensor> {
    // Get the array data as a slice
    let data = array.as_slice::<f32>();

    // Convert MLX shape (i32) to Candle shape (usize)
    let shape: Vec<usize> = array.shape().iter().map(|&x| x as usize).collect();

    // Create Candle tensor from the data
    let device = candle_core::Device::Cpu;
    let tensor = Tensor::from_vec(data.to_vec(), shape, &device)?;

    Ok(tensor)
}

/// Convert a Candle tensor to an MLX array
///
/// # Arguments
/// * `tensor` - Reference to the Candle tensor to convert
///
/// # Returns
/// A Result containing the converted MLX Array or an error
///
/// # Example
/// ```
/// use bitnet_core::mlx::candle_to_mlx_array;
/// use bitnet_core::tensor::create_tensor_f32;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = create_tensor_f32(&[2, 2], data).unwrap();
/// let array = candle_to_mlx_array(&tensor).unwrap();
/// ```
#[cfg(feature = "mlx")]
pub fn candle_to_mlx_array(tensor: &Tensor) -> Result<Array> {
    // First, ensure the tensor is on CPU and get its data
    let cpu_tensor = tensor.to_device(&candle_core::Device::Cpu)?;

    // Convert tensor to f32 if it's not already
    let f32_tensor = match cpu_tensor.dtype() {
        candle_core::DType::F32 => cpu_tensor,
        _ => cpu_tensor.to_dtype(candle_core::DType::F32)?,
    };

    // Get the tensor data as Vec<f32>
    let data = f32_tensor.flatten_all()?.to_vec1::<f32>()?;

    // Convert Candle shape (usize) to MLX shape (i32)
    let shape: Vec<i32> = tensor.shape().dims().iter().map(|&x| x as i32).collect();

    // Create and return MLX array
    create_mlx_array(&shape, data)
}

#[cfg(not(feature = "mlx"))]
pub fn create_mlx_array(_shape: &[i32], _data: Vec<f32>) -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}

#[cfg(not(feature = "mlx"))]
pub fn mlx_to_candle_tensor(_array: &()) -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}

#[cfg(not(feature = "mlx"))]
pub fn candle_to_mlx_array(_tensor: &()) -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}

#[cfg(not(feature = "mlx"))]
pub fn is_mlx_available() -> bool {
    false
}

#[cfg(not(feature = "mlx"))]
pub fn default_mlx_device() -> Result<()> {
    anyhow::bail!("MLX support not compiled in")
}
