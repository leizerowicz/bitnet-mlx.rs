// bitnet-quant/src/metrics/mse.rs
//! Mean Squared Error (MSE) Calculation
//!
//! Implements MSE calculation for quantization error analysis with optimized
//! memory-efficient computation and various MSE-related metrics.

use candle_core::{Device, Error as CandleError, Result, Tensor};

/// Calculate Mean Squared Error between original and quantized tensors
pub fn calculate_mse(original: &Tensor, quantized: &Tensor) -> Result<f32> {
    // Validate tensor shapes match
    if original.shape() != quantized.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in MSE calculation: original {:?} vs quantized {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    // Calculate squared difference
    let diff = original.sub(quantized)?;
    let squared_diff = diff.powf(2.0)?;

    // Calculate mean
    let mse_tensor = squared_diff.mean_all()?;
    let mse = mse_tensor.to_scalar::<f32>()?;

    Ok(mse)
}

/// Calculate MSE with memory-efficient streaming for large tensors
pub fn calculate_mse_streaming(
    original: &Tensor,
    quantized: &Tensor,
    chunk_size: usize,
) -> Result<f32> {
    if original.shape() != quantized.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in streaming MSE: original {:?} vs quantized {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    let total_elements = original.elem_count();
    if total_elements <= chunk_size {
        return calculate_mse(original, quantized);
    }

    // Flatten tensors for streaming processing
    let orig_flat = original.flatten_all()?;
    let quant_flat = quantized.flatten_all()?;

    let mut total_squared_error = 0.0f32;
    let mut processed_elements = 0;

    // Process in chunks
    for start in (0..total_elements).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_elements);

        // Extract chunks
        let orig_chunk = orig_flat.narrow(0, start, end - start)?;
        let quant_chunk = quant_flat.narrow(0, start, end - start)?;

        // Calculate squared error for chunk
        let diff = orig_chunk.sub(&quant_chunk)?;
        let squared_diff = diff.powf(2.0)?;
        let chunk_sum = squared_diff.sum_all()?.to_scalar::<f32>()?;

        total_squared_error += chunk_sum;
        processed_elements += end - start;
    }

    Ok(total_squared_error / processed_elements as f32)
}

/// Calculate Root Mean Squared Error (RMSE)
pub fn calculate_rmse(original: &Tensor, quantized: &Tensor) -> Result<f32> {
    let mse = calculate_mse(original, quantized)?;
    Ok(mse.sqrt())
}

/// Calculate Normalized Mean Squared Error (NMSE)
pub fn calculate_nmse(original: &Tensor, quantized: &Tensor) -> Result<f32> {
    let mse = calculate_mse(original, quantized)?;

    // Calculate power of original signal
    let original_power = original.powf(2.0)?.mean_all()?.to_scalar::<f32>()?;

    // Avoid division by zero
    if original_power < f32::EPSILON {
        return Ok(0.0);
    }

    Ok(mse / original_power)
}

/// Calculate Peak Signal-to-Noise Ratio (PSNR) based on MSE
pub fn calculate_psnr(original: &Tensor, quantized: &Tensor, max_value: f32) -> Result<f32> {
    let mse = calculate_mse(original, quantized)?;

    if mse < f32::EPSILON {
        return Ok(f32::INFINITY); // Perfect reconstruction
    }

    let psnr = 10.0 * (max_value * max_value / mse).log10();
    Ok(psnr)
}

/// MSE Calculator with advanced features
#[derive(Debug)]
#[allow(dead_code)]
pub struct MSECalculator {
    device: Device,
    streaming_threshold: usize,
    chunk_size: usize,
    enable_caching: bool,
}

impl MSECalculator {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            streaming_threshold: 10_000_000, // 10M elements
            chunk_size: 1_000_000,           // 1M elements per chunk
            enable_caching: false,
        }
    }

    pub fn with_streaming_config(device: Device, threshold: usize, chunk_size: usize) -> Self {
        Self {
            device,
            streaming_threshold: threshold,
            chunk_size,
            enable_caching: false,
        }
    }

    /// Calculate MSE with automatic streaming decision
    pub fn calculate(&self, original: &Tensor, quantized: &Tensor) -> Result<f32> {
        if original.elem_count() > self.streaming_threshold {
            calculate_mse_streaming(original, quantized, self.chunk_size)
        } else {
            calculate_mse(original, quantized)
        }
    }

    /// Calculate comprehensive MSE metrics
    pub fn calculate_comprehensive(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> Result<MSEMetrics> {
        let mse = self.calculate(original, quantized)?;
        let rmse = mse.sqrt();
        let nmse = self.calculate_nmse_internal(original, quantized, mse)?;

        // Calculate max value for PSNR
        let max_value = original.abs()?.max_all()?.to_scalar::<f32>()?;
        let psnr = if mse < f32::EPSILON {
            f32::INFINITY
        } else {
            10.0 * (max_value * max_value / mse).log10()
        };

        Ok(MSEMetrics {
            mse,
            rmse,
            nmse,
            psnr,
            max_value,
        })
    }

    fn calculate_nmse_internal(
        &self,
        original: &Tensor,
        _quantized: &Tensor,
        mse: f32,
    ) -> Result<f32> {
        let original_power = original.powf(2.0)?.mean_all()?.to_scalar::<f32>()?;
        // Safe divide: return 0 if denominator is too small, else divide
        Ok(if original_power < f32::EPSILON {
            0.0
        } else {
            mse / original_power
        })
    }

    /// Calculate layer-wise MSE for multi-layer analysis
    pub fn calculate_layerwise(
        &self,
        layer_outputs: &[(String, Tensor, Tensor)],
    ) -> Result<Vec<(String, MSEMetrics)>> {
        let mut results = Vec::new();

        for (layer_name, original, quantized) in layer_outputs {
            let metrics = self.calculate_comprehensive(original, quantized)?;
            results.push((layer_name.clone(), metrics));
        }

        Ok(results)
    }

    /// Calculate MSE evolution over time/iterations
    pub fn calculate_evolution(
        &self,
        original: &Tensor,
        quantized_series: &[Tensor],
    ) -> Result<Vec<f32>> {
        let mut mse_evolution = Vec::with_capacity(quantized_series.len());

        for quantized in quantized_series {
            let mse = self.calculate(original, quantized)?;
            mse_evolution.push(mse);
        }

        Ok(mse_evolution)
    }

    /// Calculate weighted MSE where different regions have different importance
    pub fn calculate_weighted(
        &self,
        original: &Tensor,
        quantized: &Tensor,
        weights: &Tensor,
    ) -> Result<f32> {
        if original.shape() != quantized.shape() || original.shape() != weights.shape() {
            return Err(CandleError::Msg(format!(
                "Shape mismatch in weighted MSE: original {:?} vs quantized {:?} vs weights {:?}",
                original.shape(),
                quantized.shape(),
                weights.shape()
            )));
        }

        // Calculate squared error
        let diff = original.sub(quantized)?;
        let squared_error = diff.powf(2.0)?;

        // Apply weights
        let weighted_error = squared_error.mul(weights)?;

        // Calculate weighted mean
        let total_weighted_error = weighted_error.sum_all()?.to_scalar::<f32>()?;
        let total_weights = weights.sum_all()?.to_scalar::<f32>()?;

        // Safe divide: return 0 if total_weights is too small, else divide
        Ok(if total_weights < f32::EPSILON {
            0.0
        } else {
            total_weighted_error / total_weights
        })
    }
}

/// Comprehensive MSE metrics structure
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MSEMetrics {
    pub mse: f32,
    pub rmse: f32,
    pub nmse: f32,
    pub psnr: f32,
    pub max_value: f32,
}

impl MSEMetrics {
    /// Check if MSE meets quality threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.mse <= threshold
    }

    /// Get quality assessment based on MSE value
    pub fn quality_assessment(&self) -> MSEQuality {
        if self.mse < 1e-6 {
            MSEQuality::Excellent
        } else if self.mse < 1e-4 {
            MSEQuality::Good
        } else if self.mse < 1e-2 {
            MSEQuality::Fair
        } else {
            MSEQuality::Poor
        }
    }

    /// Calculate improvement ratio compared to another MSE
    pub fn improvement_ratio(&self, other: &MSEMetrics) -> f32 {
        if other.mse < f32::EPSILON {
            0.0
        } else {
            (other.mse - self.mse) / other.mse
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MSEQuality {
    Excellent, // MSE < 1e-6
    Good,      // MSE < 1e-4
    Fair,      // MSE < 1e-2
    Poor,      // MSE >= 1e-2
}

/// Batch MSE calculator for efficient multiple tensor processing
#[allow(dead_code)]
pub struct BatchMSECalculator {
    calculator: MSECalculator,
    batch_size: usize,
}

impl BatchMSECalculator {
    pub fn new(device: Device, batch_size: usize) -> Self {
        Self {
            calculator: MSECalculator::new(device),
            batch_size,
        }
    }

    /// Calculate MSE for batches of tensor pairs
    pub fn calculate_batch(
        &self,
        original_batch: &[Tensor],
        quantized_batch: &[Tensor],
    ) -> Result<Vec<f32>> {
        if original_batch.len() != quantized_batch.len() {
            return Err(CandleError::Msg("Batch sizes don't match".to_string()));
        }

        let mut results = Vec::with_capacity(original_batch.len());

        for (orig, quant) in original_batch.iter().zip(quantized_batch.iter()) {
            let mse = self.calculator.calculate(orig, quant)?;
            results.push(mse);
        }

        Ok(results)
    }

    /// Calculate mean MSE across batch
    pub fn calculate_batch_mean(
        &self,
        original_batch: &[Tensor],
        quantized_batch: &[Tensor],
    ) -> Result<f32> {
        let mse_values = self.calculate_batch(original_batch, quantized_batch)?;
        let mean_mse = mse_values.iter().sum::<f32>() / mse_values.len() as f32;
        Ok(mean_mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_tensors() -> Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let original = Tensor::ones((4, 4), DType::F32, &device)?;
        let quantized = Tensor::zeros((4, 4), DType::F32, &device)?;
        Ok((original, quantized))
    }

    #[test]
    fn test_basic_mse_calculation() -> Result<()> {
        let (original, quantized) = create_test_tensors()?;
        let mse = calculate_mse(&original, &quantized)?;
        assert_eq!(mse, 1.0); // All 1s vs all 0s = MSE of 1.0
        Ok(())
    }

    #[test]
    fn test_identical_tensors_mse() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::ones((4, 4), DType::F32, &device)?;
        let mse = calculate_mse(&tensor, &tensor)?;
        assert_eq!(mse, 0.0); // Identical tensors should have MSE of 0
        Ok(())
    }

    #[test]
    fn test_rmse_calculation() -> Result<()> {
        let (original, quantized) = create_test_tensors()?;
        let rmse = calculate_rmse(&original, &quantized)?;
        assert_eq!(rmse, 1.0); // sqrt(1.0) = 1.0
        Ok(())
    }

    #[test]
    fn test_mse_calculator() -> Result<()> {
        let device = Device::Cpu;
        let calculator = MSECalculator::new(device);

        let (original, quantized) = create_test_tensors()?;
        let mse = calculator.calculate(&original, &quantized)?;
        assert_eq!(mse, 1.0);
        Ok(())
    }

    #[test]
    fn test_comprehensive_metrics() -> Result<()> {
        let device = Device::Cpu;
        let calculator = MSECalculator::new(device);

        let (original, quantized) = create_test_tensors()?;
        let metrics = calculator.calculate_comprehensive(&original, &quantized)?;

        assert_eq!(metrics.mse, 1.0);
        assert_eq!(metrics.rmse, 1.0);
        assert_eq!(metrics.quality_assessment(), MSEQuality::Poor);
        Ok(())
    }

    #[test]
    fn test_streaming_mse() -> Result<()> {
        let device = Device::Cpu;
        let large_original = Tensor::ones((1000, 1000), DType::F32, &device)?;
        let large_quantized = Tensor::zeros((1000, 1000), DType::F32, &device)?;

        let mse = calculate_mse_streaming(&large_original, &large_quantized, 10000)?;
        assert_eq!(mse, 1.0);
        Ok(())
    }

    #[test]
    fn test_weighted_mse() -> Result<()> {
        let device = Device::Cpu;
        let calculator = MSECalculator::new(device.clone());

        let original = Tensor::ones((2, 2), DType::F32, &device)?;
        let quantized = Tensor::zeros((2, 2), DType::F32, &device)?;
        let weights_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = Tensor::new(weights_data.as_slice(), &device)?.reshape((2, 2))?;

        let weighted_mse = calculator.calculate_weighted(&original, &quantized, &weights)?;
        assert_eq!(weighted_mse, 1.0); // All errors are 1, so weighted average is still 1
        Ok(())
    }

    #[test]
    fn test_batch_mse_calculator() -> Result<()> {
        let device = Device::Cpu;
        let batch_calculator = BatchMSECalculator::new(device, 4);

        let originals = vec![
            Tensor::ones((2, 2), DType::F32, &Device::Cpu)?,
            Tensor::ones((2, 2), DType::F32, &Device::Cpu)?,
        ];
        let quantized = vec![
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
        ];

        let mse_values = batch_calculator.calculate_batch(&originals, &quantized)?;
        assert_eq!(mse_values, vec![1.0, 1.0]);

        let mean_mse = batch_calculator.calculate_batch_mean(&originals, &quantized)?;
        assert_eq!(mean_mse, 1.0);
        Ok(())
    }

    #[test]
    fn test_mse_quality_assessment() {
        let excellent = MSEMetrics {
            mse: 1e-7,
            rmse: 1e-3,
            nmse: 0.1,
            psnr: 60.0,
            max_value: 1.0,
        };
        assert_eq!(excellent.quality_assessment(), MSEQuality::Excellent);

        let poor = MSEMetrics {
            mse: 1e-1,
            rmse: 0.3,
            nmse: 0.5,
            psnr: 10.0,
            max_value: 1.0,
        };
        assert_eq!(poor.quality_assessment(), MSEQuality::Poor);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let device = Device::Cpu;
        let tensor1 = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let tensor2 = Tensor::ones((3, 3), DType::F32, &device).unwrap();

        let result = calculate_mse(&tensor1, &tensor2);
        assert!(result.is_err());
    }
}
