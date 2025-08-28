// bitnet-quant/src/metrics/sqnr.rs
//! Signal-to-Quantization-Noise Ratio (SQNR) Calculation
//!
//! Implements SQNR calculation for quantization quality assessment, providing
//! comprehensive noise analysis and signal quality metrics in dB scale.

use candle_core::{Device, Error as CandleError, Result, Tensor};

/// Calculate Signal-to-Quantization-Noise Ratio in dB
pub fn calculate_sqnr(original: &Tensor, quantized: &Tensor) -> Result<f32> {
    // Validate tensor shapes match
    if original.shape() != quantized.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in SQNR calculation: original {:?} vs quantized {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    // Calculate signal power (original signal energy)
    let signal_power = original.powf(2.0)?.mean_all()?.to_scalar::<f32>()?;

    // Calculate quantization noise power
    let noise = original.sub(quantized)?;
    let noise_power = noise.powf(2.0)?.mean_all()?.to_scalar::<f32>()?;

    // Avoid division by zero and log of zero
    if noise_power < f32::EPSILON {
        return Ok(f32::INFINITY); // Perfect quantization
    }

    if signal_power < f32::EPSILON {
        return Ok(f32::NEG_INFINITY); // No signal
    }

    // Calculate SQNR in dB: 10 * log10(signal_power / noise_power)
    let sqnr_db = 10.0 * (signal_power / noise_power).log10();
    Ok(sqnr_db)
}

/// Calculate SQNR with memory-efficient streaming for large tensors
pub fn calculate_sqnr_streaming(
    original: &Tensor,
    quantized: &Tensor,
    chunk_size: usize,
) -> Result<f32> {
    if original.shape() != quantized.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in streaming SQNR: original {:?} vs quantized {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    let total_elements = original.elem_count();
    if total_elements <= chunk_size {
        return calculate_sqnr(original, quantized);
    }

    // Flatten tensors for streaming processing
    let orig_flat = original.flatten_all()?;
    let quant_flat = quantized.flatten_all()?;

    let mut total_signal_power = 0.0f32;
    let mut total_noise_power = 0.0f32;
    let mut processed_elements = 0;

    // Process in chunks
    for start in (0..total_elements).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_elements);

        // Extract chunks
        let orig_chunk = orig_flat.narrow(0, start, end - start)?;
        let quant_chunk = quant_flat.narrow(0, start, end - start)?;

        // Calculate signal power for chunk
        let signal_power_chunk = orig_chunk.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        total_signal_power += signal_power_chunk;

        // Calculate noise power for chunk
        let noise_chunk = orig_chunk.sub(&quant_chunk)?;
        let noise_power_chunk = noise_chunk.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        total_noise_power += noise_power_chunk;

        processed_elements += end - start;
    }

    // Calculate average powers
    let avg_signal_power = total_signal_power / processed_elements as f32;
    let avg_noise_power = total_noise_power / processed_elements as f32;

    // Calculate SQNR
    if avg_noise_power < f32::EPSILON {
        Ok(f32::INFINITY)
    } else if avg_signal_power < f32::EPSILON {
        Ok(f32::NEG_INFINITY)
    } else {
        Ok(10.0 * (avg_signal_power / avg_noise_power).log10())
    }
}

/// Calculate Segmental SQNR for time-series or sequential data
pub fn calculate_segmental_sqnr(
    original: &Tensor,
    quantized: &Tensor,
    segment_size: usize,
) -> Result<Vec<f32>> {
    if original.shape() != quantized.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in segmental SQNR: original {:?} vs quantized {:?}",
            original.shape(),
            quantized.shape()
        )));
    }

    let total_elements = original.elem_count();
    let orig_flat = original.flatten_all()?;
    let quant_flat = quantized.flatten_all()?;

    let mut segmental_sqnr = Vec::new();

    for start in (0..total_elements).step_by(segment_size) {
        let end = (start + segment_size).min(total_elements);

        // Extract segments
        let orig_segment = orig_flat.narrow(0, start, end - start)?;
        let quant_segment = quant_flat.narrow(0, start, end - start)?;

        // Calculate SQNR for this segment
        let segment_sqnr = calculate_sqnr(&orig_segment, &quant_segment)?;
        segmental_sqnr.push(segment_sqnr);
    }

    Ok(segmental_sqnr)
}

/// SQNR Calculator with advanced analysis features
#[derive(Debug)]
#[allow(dead_code)]
pub struct SQNRCalculator {
    device: Device,
    streaming_threshold: usize,
    chunk_size: usize,
    segment_size: usize,
}

impl SQNRCalculator {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            streaming_threshold: 10_000_000, // 10M elements
            chunk_size: 1_000_000,           // 1M elements per chunk
            segment_size: 1000,              // 1K elements per segment for segmental analysis
        }
    }

    pub fn with_config(
        device: Device,
        streaming_threshold: usize,
        chunk_size: usize,
        segment_size: usize,
    ) -> Self {
        Self {
            device,
            streaming_threshold,
            chunk_size,
            segment_size,
        }
    }

    /// Calculate SQNR with automatic streaming decision
    pub fn calculate(&self, original: &Tensor, quantized: &Tensor) -> Result<f32> {
        if original.elem_count() > self.streaming_threshold {
            calculate_sqnr_streaming(original, quantized, self.chunk_size)
        } else {
            calculate_sqnr(original, quantized)
        }
    }

    /// Calculate comprehensive SQNR analysis
    pub fn calculate_comprehensive(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> Result<SQNRAnalysis> {
        let global_sqnr = self.calculate(original, quantized)?;

        // Calculate segmental SQNR for variance analysis
        let segmental_sqnr = if original.elem_count() > self.segment_size {
            calculate_segmental_sqnr(original, quantized, self.segment_size)?
        } else {
            vec![global_sqnr]
        };

        // Calculate statistics of segmental SQNR
        let sqnr_stats = self.calculate_sqnr_statistics(&segmental_sqnr)?;

        // Calculate frequency-domain SQNR if applicable
        let frequency_sqnr = self.calculate_frequency_sqnr(original, quantized)?;

        Ok(SQNRAnalysis {
            global_sqnr,
            segmental_sqnr,
            statistics: sqnr_stats,
            frequency_sqnr,
            quality_grade: self.assess_sqnr_quality(global_sqnr),
        })
    }

    fn calculate_sqnr_statistics(&self, segmental_sqnr: &[f32]) -> Result<SQNRStatistics> {
        if segmental_sqnr.is_empty() {
            return Ok(SQNRStatistics::default());
        }

        // Filter out infinite values for statistics
        let finite_values: Vec<f32> = segmental_sqnr
            .iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();

        if finite_values.is_empty() {
            return Ok(SQNRStatistics::default());
        }

        let mean = finite_values.iter().sum::<f32>() / finite_values.len() as f32;
        let min = finite_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = finite_values
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate variance
        let variance = finite_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / finite_values.len() as f32;

        let std_dev = variance.sqrt();

        // Calculate percentiles
        let num_finite = finite_values.len();
        let mut sorted_values = finite_values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p10_idx = (0.1 * sorted_values.len() as f32) as usize;
        let p90_idx = (0.9 * sorted_values.len() as f32) as usize;

        let p10 = sorted_values[p10_idx.min(sorted_values.len() - 1)];
        let p90 = sorted_values[p90_idx.min(sorted_values.len() - 1)];

        Ok(SQNRStatistics {
            mean,
            std_dev,
            min,
            max,
            p10,
            p90,
            variance,
            num_segments: segmental_sqnr.len(),
            num_finite,
        })
    }

    fn calculate_frequency_sqnr(
        &self,
        _original: &Tensor,
        _quantized: &Tensor,
    ) -> Result<Option<f32>> {
        // For now, return None - frequency domain analysis would require FFT implementation
        // This is a placeholder for future enhancement
        Ok(None)
    }

    fn assess_sqnr_quality(&self, sqnr_db: f32) -> SQNRQuality {
        if sqnr_db.is_infinite() && sqnr_db > 0.0 {
            SQNRQuality::Perfect
        } else if sqnr_db >= 60.0 {
            SQNRQuality::Excellent
        } else if sqnr_db >= 40.0 {
            SQNRQuality::Good
        } else if sqnr_db >= 20.0 {
            SQNRQuality::Fair
        } else if sqnr_db >= 10.0 {
            SQNRQuality::Poor
        } else {
            SQNRQuality::Unacceptable
        }
    }

    /// Calculate SQNR improvement ratio between two quantization methods
    pub fn calculate_improvement(
        &self,
        original: &Tensor,
        quantized1: &Tensor,
        quantized2: &Tensor,
    ) -> Result<f32> {
        let sqnr1 = self.calculate(original, quantized1)?;
        let sqnr2 = self.calculate(original, quantized2)?;

        // Return the difference in dB (improvement of quantized2 over quantized1)
        Ok(sqnr2 - sqnr1)
    }

    /// Calculate layer-wise SQNR analysis
    pub fn analyze_layers(
        &self,
        layer_outputs: &[(String, Tensor, Tensor)],
    ) -> Result<Vec<(String, SQNRAnalysis)>> {
        let mut results = Vec::new();

        for (layer_name, original, quantized) in layer_outputs {
            let analysis = self.calculate_comprehensive(original, quantized)?;
            results.push((layer_name.clone(), analysis));
        }

        Ok(results)
    }

    /// Calculate SQNR evolution over training/iterations
    pub fn track_evolution(
        &self,
        original: &Tensor,
        quantized_series: &[Tensor],
    ) -> Result<SQNREvolution> {
        let mut sqnr_values = Vec::with_capacity(quantized_series.len());

        for quantized in quantized_series {
            let sqnr = self.calculate(original, quantized)?;
            sqnr_values.push(sqnr);
        }

        // Calculate trend analysis
        let trend = self.analyze_trend(&sqnr_values);
        let best_iteration = sqnr_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        let initial_sqnr = sqnr_values.first().copied().unwrap_or(0.0);
        let final_sqnr = sqnr_values.last().copied().unwrap_or(0.0);

        Ok(SQNREvolution {
            sqnr_values,
            trend,
            best_iteration,
            initial_sqnr,
            final_sqnr,
        })
    }

    fn analyze_trend(&self, values: &[f32]) -> Trend {
        if values.len() < 2 {
            return Trend::Stable;
        }

        let first_half = &values[0..values.len() / 2];
        let second_half = &values[values.len() / 2..];

        let first_avg = first_half.iter().sum::<f32>() / first_half.len() as f32;
        let second_avg = second_half.iter().sum::<f32>() / second_half.len() as f32;

        let improvement = second_avg - first_avg;

        if improvement > 1.0 {
            // More than 1dB improvement
            Trend::Improving
        } else if improvement < -1.0 {
            // More than 1dB degradation
            Trend::Degrading
        } else {
            Trend::Stable
        }
    }
}

/// Comprehensive SQNR analysis results
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SQNRAnalysis {
    pub global_sqnr: f32,
    pub segmental_sqnr: Vec<f32>,
    pub statistics: SQNRStatistics,
    pub frequency_sqnr: Option<f32>,
    pub quality_grade: SQNRQuality,
}

/// Statistical analysis of SQNR values
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SQNRStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub p10: f32, // 10th percentile
    pub p90: f32, // 90th percentile
    pub variance: f32,
    pub num_segments: usize,
    pub num_finite: usize,
}

impl Default for SQNRStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            p10: 0.0,
            p90: 0.0,
            variance: 0.0,
            num_segments: 0,
            num_finite: 0,
        }
    }
}

/// SQNR quality assessment levels
#[derive(Debug, Clone, PartialEq)]
pub enum SQNRQuality {
    Perfect,      // Infinite SQNR
    Excellent,    // >= 60 dB
    Good,         // >= 40 dB
    Fair,         // >= 20 dB
    Poor,         // >= 10 dB
    Unacceptable, // < 10 dB
}

impl SQNRQuality {
    pub fn to_string(&self) -> &'static str {
        match self {
            SQNRQuality::Perfect => "Perfect",
            SQNRQuality::Excellent => "Excellent",
            SQNRQuality::Good => "Good",
            SQNRQuality::Fair => "Fair",
            SQNRQuality::Poor => "Poor",
            SQNRQuality::Unacceptable => "Unacceptable",
        }
    }

    pub fn meets_threshold(&self, minimum_quality: &SQNRQuality) -> bool {
        match (self, minimum_quality) {
            (SQNRQuality::Perfect, _) => true,
            (SQNRQuality::Excellent, SQNRQuality::Perfect) => false,
            (SQNRQuality::Excellent, _) => true,
            (SQNRQuality::Good, SQNRQuality::Perfect | SQNRQuality::Excellent) => false,
            (SQNRQuality::Good, _) => true,
            (
                SQNRQuality::Fair,
                SQNRQuality::Perfect | SQNRQuality::Excellent | SQNRQuality::Good,
            ) => false,
            (SQNRQuality::Fair, _) => true,
            (SQNRQuality::Poor, SQNRQuality::Unacceptable) => true,
            (SQNRQuality::Poor, _) => false,
            (SQNRQuality::Unacceptable, _) => false,
        }
    }
}

/// SQNR evolution tracking over time
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SQNREvolution {
    pub sqnr_values: Vec<f32>,
    pub trend: Trend,
    pub best_iteration: Option<usize>,
    pub initial_sqnr: f32,
    pub final_sqnr: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Trend {
    Improving,
    Degrading,
    Stable,
}

/// Utility functions for SQNR-related calculations
pub fn db_to_linear(db_value: f32) -> f32 {
    10.0_f32.powf(db_value / 10.0)
}

pub fn linear_to_db(linear_value: f32) -> f32 {
    10.0 * linear_value.log10()
}

pub fn sqnr_to_effective_bits(sqnr_db: f32) -> f32 {
    (sqnr_db - 1.76) / 6.02 // Theoretical relationship for uniform quantization
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_tensors() -> Result<(Tensor, Tensor, Tensor)> {
        let device = Device::Cpu;
        let original = Tensor::ones((4, 4), DType::F32, &device)?;
        let identical = original.clone();
        let quantized = Tensor::new(&[0.9f32; 16], &device)?.reshape((4, 4))?;
        Ok((original, identical, quantized))
    }

    #[test]
    fn test_perfect_sqnr() -> Result<()> {
        let (original, identical, _) = create_test_tensors()?;
        let sqnr = calculate_sqnr(&original, &identical)?;
        assert!(sqnr.is_infinite()); // Perfect reconstruction
        Ok(())
    }

    #[test]
    fn test_finite_sqnr() -> Result<()> {
        let (original, _, quantized) = create_test_tensors()?;
        let sqnr = calculate_sqnr(&original, &quantized)?;
        assert!(sqnr.is_finite() && sqnr > 0.0);
        Ok(())
    }

    #[test]
    fn test_sqnr_calculator() -> Result<()> {
        let device = Device::Cpu;
        let calculator = SQNRCalculator::new(device);

        let (original, _, quantized) = create_test_tensors()?;
        let sqnr = calculator.calculate(&original, &quantized)?;
        assert!(sqnr.is_finite());
        Ok(())
    }

    #[test]
    fn test_comprehensive_sqnr_analysis() -> Result<()> {
        let device = Device::Cpu;
        let calculator = SQNRCalculator::new(device);

        let (original, _, quantized) = create_test_tensors()?;
        let analysis = calculator.calculate_comprehensive(&original, &quantized)?;

        assert!(analysis.global_sqnr.is_finite());
        assert!(!analysis.segmental_sqnr.is_empty());
        assert!(analysis.statistics.num_segments > 0);
        Ok(())
    }

    #[test]
    fn test_sqnr_quality_assessment() {
        let device = Device::Cpu;
        let calculator = SQNRCalculator::new(device);

        assert_eq!(calculator.assess_sqnr_quality(70.0), SQNRQuality::Excellent);
        assert_eq!(calculator.assess_sqnr_quality(45.0), SQNRQuality::Good);
        assert_eq!(calculator.assess_sqnr_quality(25.0), SQNRQuality::Fair);
        assert_eq!(calculator.assess_sqnr_quality(15.0), SQNRQuality::Poor);
        assert_eq!(
            calculator.assess_sqnr_quality(5.0),
            SQNRQuality::Unacceptable
        );
    }

    #[test]
    fn test_streaming_sqnr() -> Result<()> {
        let device = Device::Cpu;
        let large_original = Tensor::ones((1000, 1000), DType::F32, &device)?;
        let data = vec![0.95f32; 1_000_000];
        let large_quantized = Tensor::new(data.as_slice(), &device)?.reshape((1000, 1000))?;

        let sqnr = calculate_sqnr_streaming(&large_original, &large_quantized, 10000)?;
        assert!(sqnr.is_finite() && sqnr > 0.0);
        Ok(())
    }

    #[test]
    fn test_segmental_sqnr() -> Result<()> {
        let device = Device::Cpu;
        let original = Tensor::ones((100,), DType::F32, &device)?;
        let data = vec![0.9f32; 100];
        let quantized = Tensor::new(data.as_slice(), &device)?;

        let segmental = calculate_segmental_sqnr(&original, &quantized, 10)?;
        assert_eq!(segmental.len(), 10); // 100 elements / 10 per segment = 10 segments
        assert!(segmental.iter().all(|&x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_sqnr_improvement() -> Result<()> {
        let device = Device::Cpu;
        let calculator = SQNRCalculator::new(device.clone());

        let original = Tensor::ones((4, 4), DType::F32, &device)?;
        let data1 = vec![0.8f32; 16];
        let quantized1 = Tensor::new(data1.as_slice(), &device)?.reshape((4, 4))?;
        let data2 = vec![0.9f32; 16];
        let quantized2 = Tensor::new(data2.as_slice(), &device)?.reshape((4, 4))?;

        let improvement = calculator.calculate_improvement(&original, &quantized1, &quantized2)?;
        assert!(improvement > 0.0); // quantized2 should be better (closer to original)
        Ok(())
    }

    #[test]
    fn test_utility_functions() {
        let db_value = 20.0;
        let linear = db_to_linear(db_value);
        let back_to_db = linear_to_db(linear);
        assert!((back_to_db - db_value).abs() < 1e-6);

        let effective_bits = sqnr_to_effective_bits(50.0);
        assert!(effective_bits > 0.0 && effective_bits < 32.0);
    }

    #[test]
    fn test_sqnr_quality_comparison() {
        assert!(SQNRQuality::Excellent.meets_threshold(&SQNRQuality::Good));
        assert!(!SQNRQuality::Poor.meets_threshold(&SQNRQuality::Good));
        assert!(SQNRQuality::Perfect.meets_threshold(&SQNRQuality::Perfect));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let device = Device::Cpu;
        let tensor1 = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let tensor2 = Tensor::ones((3, 3), DType::F32, &device).unwrap();

        let result = calculate_sqnr(&tensor1, &tensor2);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_signal_sqnr() -> Result<()> {
        let device = Device::Cpu;
        let zero_signal = Tensor::zeros((4, 4), DType::F32, &device)?;
        let quantized = Tensor::ones((4, 4), DType::F32, &device)?;

        let sqnr = calculate_sqnr(&zero_signal, &quantized)?;
        assert_eq!(sqnr, f32::NEG_INFINITY); // No signal case
        Ok(())
    }
}
