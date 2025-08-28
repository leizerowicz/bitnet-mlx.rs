// bitnet-quant/src/metrics/cosine_similarity.rs
//! Cosine Similarity Metrics for Quantization Quality Assessment
//!
//! Implements cosine similarity calculation for measuring angular similarity
//! between original and quantized tensors, providing complementary quality metrics.

use candle_core::{Device, Error as CandleError, Result, Tensor};

/// Calculate cosine similarity between two tensors
pub fn calculate_cosine_similarity(tensor_a: &Tensor, tensor_b: &Tensor) -> Result<f32> {
    // Validate tensor shapes match
    if tensor_a.shape() != tensor_b.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in cosine similarity: tensor_a {:?} vs tensor_b {:?}",
            tensor_a.shape(),
            tensor_b.shape()
        )));
    }

    // Flatten tensors for dot product calculation
    let flat_a = tensor_a.flatten_all()?;
    let flat_b = tensor_b.flatten_all()?;

    // Calculate dot product
    let dot_product = flat_a.mul(&flat_b)?.sum_all()?.to_scalar::<f32>()?;

    // Calculate norms
    let norm_a = flat_a.powf(2.0)?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b = flat_b.powf(2.0)?.sum_all()?.to_scalar::<f32>()?.sqrt();

    // Calculate cosine similarity
    let norm_product = norm_a * norm_b;
    let cosine_sim = if norm_product < f32::EPSILON {
        0.0
    } else {
        dot_product / norm_product
    };
    Ok(cosine_sim)
}

/// Calculate cosine similarity with memory-efficient streaming
pub fn calculate_cosine_similarity_streaming(
    tensor_a: &Tensor,
    tensor_b: &Tensor,
    chunk_size: usize,
) -> Result<f32> {
    if tensor_a.shape() != tensor_b.shape() {
        return Err(CandleError::Msg(format!(
            "Shape mismatch in streaming cosine similarity: tensor_a {:?} vs tensor_b {:?}",
            tensor_a.shape(),
            tensor_b.shape()
        )));
    }

    let total_elements = tensor_a.elem_count();
    if total_elements <= chunk_size {
        return calculate_cosine_similarity(tensor_a, tensor_b);
    }

    // Flatten tensors for streaming processing
    let flat_a = tensor_a.flatten_all()?;
    let flat_b = tensor_b.flatten_all()?;

    let mut total_dot_product = 0.0f32;
    let mut total_norm_a_squared = 0.0f32;
    let mut total_norm_b_squared = 0.0f32;

    // Process in chunks
    for start in (0..total_elements).step_by(chunk_size) {
        let end = (start + chunk_size).min(total_elements);

        // Extract chunks
        let chunk_a = flat_a.narrow(0, start, end - start)?;
        let chunk_b = flat_b.narrow(0, start, end - start)?;

        // Calculate dot product for chunk
        let chunk_dot = chunk_a.mul(&chunk_b)?.sum_all()?.to_scalar::<f32>()?;
        total_dot_product += chunk_dot;

        // Calculate norm squared for chunks
        let norm_a_sq_chunk = chunk_a.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        let norm_b_sq_chunk = chunk_b.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;

        total_norm_a_squared += norm_a_sq_chunk;
        total_norm_b_squared += norm_b_sq_chunk;
    }

    // Calculate final cosine similarity
    let norm_product = total_norm_a_squared.sqrt() * total_norm_b_squared.sqrt();
    let cosine_sim = if norm_product < f32::EPSILON {
        0.0
    } else {
        total_dot_product / norm_product
    };
    Ok(cosine_sim)
}

/// Calculate angular distance from cosine similarity
pub fn cosine_to_angular_distance(cosine_similarity: f32) -> f32 {
    // Clamp to valid range for acos
    let clamped = cosine_similarity.clamp(-1.0, 1.0);
    clamped.acos()
}

/// Calculate angular distance in degrees
pub fn cosine_to_angular_distance_degrees(cosine_similarity: f32) -> f32 {
    cosine_to_angular_distance(cosine_similarity) * 180.0 / std::f32::consts::PI
}

/// Cosine Similarity Calculator with advanced features
#[derive(Debug)]
#[allow(dead_code)]
pub struct CosineSimilarityCalculator {
    device: Device,
    streaming_threshold: usize,
    chunk_size: usize,
    normalize_inputs: bool,
    handle_zero_vectors: ZeroVectorHandling,
}

impl CosineSimilarityCalculator {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            streaming_threshold: 10_000_000, // 10M elements
            chunk_size: 1_000_000,           // 1M elements per chunk
            normalize_inputs: false,
            handle_zero_vectors: ZeroVectorHandling::ReturnZero,
        }
    }

    pub fn with_config(
        device: Device,
        streaming_threshold: usize,
        chunk_size: usize,
        normalize_inputs: bool,
        handle_zero_vectors: ZeroVectorHandling,
    ) -> Self {
        Self {
            device,
            streaming_threshold,
            chunk_size,
            normalize_inputs,
            handle_zero_vectors,
        }
    }

    /// Calculate cosine similarity with automatic streaming decision
    pub fn calculate(&self, tensor_a: &Tensor, tensor_b: &Tensor) -> Result<f32> {
        let (processed_a, processed_b) = if self.normalize_inputs {
            (
                self.normalize_tensor(tensor_a)?,
                self.normalize_tensor(tensor_b)?,
            )
        } else {
            (tensor_a.clone(), tensor_b.clone())
        };

        if processed_a.elem_count() > self.streaming_threshold {
            calculate_cosine_similarity_streaming(&processed_a, &processed_b, self.chunk_size)
        } else {
            calculate_cosine_similarity(&processed_a, &processed_b)
        }
    }

    fn normalize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor.powf(2.0)?.sum_all()?.to_scalar::<f32>()?.sqrt();
        if norm < f32::EPSILON {
            match self.handle_zero_vectors {
                ZeroVectorHandling::ReturnZero => Ok(tensor.clone()),
                ZeroVectorHandling::ReturnOnes => {
                    let ones = Tensor::ones(tensor.shape(), tensor.dtype(), tensor.device())?;
                    Ok(ones)
                }
                ZeroVectorHandling::Error => {
                    Err(CandleError::Msg("Cannot normalize zero vector".to_string()))
                }
            }
        } else {
            tensor.div(&Tensor::new(&[norm], tensor.device())?.broadcast_as(tensor.shape())?)
        }
    }

    /// Calculate comprehensive cosine similarity analysis
    pub fn calculate_comprehensive(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> Result<CosineSimilarityAnalysis> {
        let cosine_similarity = self.calculate(original, quantized)?;
        let angular_distance_rad = cosine_to_angular_distance(cosine_similarity);
        let angular_distance_deg = cosine_to_angular_distance_degrees(cosine_similarity);

        // Calculate similarity grade
        let similarity_grade = self.assess_similarity_quality(cosine_similarity);

        // Calculate normalized cosine similarity (handling potential negative values)
        let normalized_similarity = (cosine_similarity + 1.0) / 2.0; // Map [-1, 1] to [0, 1]

        // Calculate additional metrics
        let orthogonality = (angular_distance_rad - std::f32::consts::FRAC_PI_2).abs(); // Distance from 90 degrees
        let alignment_strength = cosine_similarity.abs(); // How well aligned (ignoring direction)

        Ok(CosineSimilarityAnalysis {
            cosine_similarity,
            angular_distance_rad,
            angular_distance_deg,
            normalized_similarity,
            orthogonality,
            alignment_strength,
            similarity_grade,
        })
    }

    fn assess_similarity_quality(&self, cosine_similarity: f32) -> SimilarityQuality {
        let abs_sim = cosine_similarity.abs();

        if abs_sim >= 0.99 {
            SimilarityQuality::Excellent
        } else if abs_sim >= 0.95 {
            SimilarityQuality::Good
        } else if abs_sim >= 0.90 {
            SimilarityQuality::Fair
        } else if abs_sim >= 0.70 {
            SimilarityQuality::Poor
        } else {
            SimilarityQuality::Unacceptable
        }
    }

    /// Calculate layer-wise cosine similarity analysis
    pub fn analyze_layers(
        &self,
        layer_outputs: &[(String, Tensor, Tensor)],
    ) -> Result<Vec<(String, CosineSimilarityAnalysis)>> {
        let mut results = Vec::new();

        for (layer_name, original, quantized) in layer_outputs {
            let analysis = self.calculate_comprehensive(original, quantized)?;
            results.push((layer_name.clone(), analysis));
        }

        Ok(results)
    }

    /// Calculate cosine similarity evolution over training iterations
    pub fn track_evolution(
        &self,
        original: &Tensor,
        quantized_series: &[Tensor],
    ) -> Result<SimilarityEvolution> {
        let mut similarity_values = Vec::with_capacity(quantized_series.len());
        let mut angular_distances = Vec::with_capacity(quantized_series.len());

        for quantized in quantized_series {
            let similarity = self.calculate(original, quantized)?;
            let angular_distance = cosine_to_angular_distance(similarity);

            similarity_values.push(similarity);
            angular_distances.push(angular_distance);
        }

        // Find best iteration (highest similarity)
        let best_iteration = similarity_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        // Calculate trend
        let trend = self.analyze_similarity_trend(&similarity_values);

        let initial_similarity = similarity_values.first().copied().unwrap_or(0.0);
        let final_similarity = similarity_values.last().copied().unwrap_or(0.0);

        Ok(SimilarityEvolution {
            similarity_values,
            angular_distances,
            trend,
            best_iteration,
            initial_similarity,
            final_similarity,
        })
    }

    fn analyze_similarity_trend(&self, values: &[f32]) -> SimilarityTrend {
        if values.len() < 2 {
            return SimilarityTrend::Stable;
        }

        let first_half = &values[0..values.len() / 2];
        let second_half = &values[values.len() / 2..];

        let first_avg = first_half.iter().sum::<f32>() / first_half.len() as f32;
        let second_avg = second_half.iter().sum::<f32>() / second_half.len() as f32;

        let improvement = second_avg - first_avg;

        if improvement > 0.01 {
            // 1% improvement in similarity
            SimilarityTrend::Improving
        } else if improvement < -0.01 {
            // 1% degradation
            SimilarityTrend::Degrading
        } else {
            SimilarityTrend::Stable
        }
    }

    /// Calculate pairwise cosine similarities for batch analysis
    pub fn calculate_pairwise(&self, tensors: &[Tensor]) -> Result<Vec<Vec<f32>>> {
        let n = tensors.len();
        let mut similarity_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            similarity_matrix[i][i] = 1.0; // Self-similarity is 1.0

            for j in (i + 1)..n {
                let similarity = self.calculate(&tensors[i], &tensors[j])?;
                similarity_matrix[i][j] = similarity;
                similarity_matrix[j][i] = similarity; // Symmetric
            }
        }

        Ok(similarity_matrix)
    }

    /// Calculate similarity statistics for a set of comparisons
    pub fn calculate_statistics(&self, similarities: &[f32]) -> SimilarityStatistics {
        if similarities.is_empty() {
            return SimilarityStatistics::default();
        }

        let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;
        let min = similarities.iter().fold(1.0f32, |a, &b| a.min(b));
        let max = similarities.iter().fold(-1.0f32, |a, &b| a.max(b));

        let variance = similarities
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / similarities.len() as f32;

        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted = similarities.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
        } else {
            sorted[median_idx]
        };

        SimilarityStatistics {
            mean,
            median,
            std_dev,
            min,
            max,
            variance,
            count: similarities.len(),
        }
    }
}

/// Comprehensive cosine similarity analysis results
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CosineSimilarityAnalysis {
    pub cosine_similarity: f32,
    pub angular_distance_rad: f32,
    pub angular_distance_deg: f32,
    pub normalized_similarity: f32,
    pub orthogonality: f32,
    pub alignment_strength: f32,
    pub similarity_grade: SimilarityQuality,
}

impl CosineSimilarityAnalysis {
    pub fn is_well_aligned(&self, threshold: f32) -> bool {
        self.cosine_similarity >= threshold
    }

    pub fn is_nearly_orthogonal(&self, tolerance_deg: f32) -> bool {
        (self.angular_distance_deg - 90.0).abs() <= tolerance_deg
    }

    pub fn similarity_percentage(&self) -> f32 {
        self.normalized_similarity * 100.0
    }
}

/// Similarity quality assessment levels
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityQuality {
    Excellent,    // |_cos| >= 0.99
    Good,         // |_cos| >= 0.95
    Fair,         // |_cos| >= 0.90
    Poor,         // |_cos| >= 0.70
    Unacceptable, // |_cos| < 0.70
}

impl SimilarityQuality {
    pub fn to_string(&self) -> &'static str {
        match self {
            SimilarityQuality::Excellent => "Excellent",
            SimilarityQuality::Good => "Good",
            SimilarityQuality::Fair => "Fair",
            SimilarityQuality::Poor => "Poor",
            SimilarityQuality::Unacceptable => "Unacceptable",
        }
    }
}

/// Zero vector handling strategies
#[derive(Debug, Clone)]
pub enum ZeroVectorHandling {
    ReturnZero, // Return zero similarity
    ReturnOnes, // Replace with ones vector
    Error,      // Return error
}

/// Similarity evolution tracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SimilarityEvolution {
    pub similarity_values: Vec<f32>,
    pub angular_distances: Vec<f32>,
    pub trend: SimilarityTrend,
    pub best_iteration: Option<usize>,
    pub initial_similarity: f32,
    pub final_similarity: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityTrend {
    Improving,
    Degrading,
    Stable,
}

/// Statistical analysis of similarity values
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SimilarityStatistics {
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub variance: f32,
    pub count: usize,
}

impl Default for SimilarityStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 1.0,
            max: -1.0,
            variance: 0.0,
            count: 0,
        }
    }
}

/// Batch cosine similarity calculator for efficient processing
#[allow(dead_code)]
pub struct BatchCosineSimilarityCalculator {
    calculator: CosineSimilarityCalculator,
}

impl BatchCosineSimilarityCalculator {
    pub fn new(device: Device) -> Self {
        Self {
            calculator: CosineSimilarityCalculator::new(device),
        }
    }

    /// Calculate similarities for batches of tensor pairs
    pub fn calculate_batch(
        &self,
        original_batch: &[Tensor],
        quantized_batch: &[Tensor],
    ) -> Result<Vec<f32>> {
        if original_batch.len() != quantized_batch.len() {
            return Err(CandleError::Msg("Batch sizes don't match".to_string()));
        }

        let mut similarities = Vec::with_capacity(original_batch.len());

        for (orig, quant) in original_batch.iter().zip(quantized_batch.iter()) {
            let similarity = self.calculator.calculate(orig, quant)?;
            similarities.push(similarity);
        }

        Ok(similarities)
    }

    /// Calculate mean similarity across batch
    pub fn calculate_batch_mean(
        &self,
        original_batch: &[Tensor],
        quantized_batch: &[Tensor],
    ) -> Result<f32> {
        let similarities = self.calculate_batch(original_batch, quantized_batch)?;
        let mean_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
        Ok(mean_similarity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_tensors() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let device = Device::Cpu;
        let original = Tensor::ones((4, 4), DType::F32, &device)?;
        let identical = original.clone();
        let scaled_value = Tensor::full(2.0f32, (4, 4), &device)?;
        let scaled = original.mul(&scaled_value)?; // Same direction, different magnitude
        let orthogonal = Tensor::new(
            &[
                -1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
                1.0, -1.0,
            ],
            &device,
        )?
        .reshape((4, 4))?;
        Ok((original, identical, scaled, orthogonal))
    }

    #[test]
    fn test_identical_tensors() -> Result<()> {
        let (original, identical, _, _) = create_test_tensors()?;
        let similarity = calculate_cosine_similarity(&original, &identical)?;
        assert!((similarity - 1.0).abs() < 1e-6); // Should be exactly 1.0
        Ok(())
    }

    #[test]
    fn test_scaled_tensors() -> Result<()> {
        let (original, _, scaled, _) = create_test_tensors()?;
        let similarity = calculate_cosine_similarity(&original, &scaled)?;
        assert!((similarity - 1.0).abs() < 1e-6); // Same direction, should be 1.0
        Ok(())
    }

    #[test]
    fn test_orthogonal_tensors() -> Result<()> {
        let (original, _, _, orthogonal) = create_test_tensors()?;
        let similarity = calculate_cosine_similarity(&original, &orthogonal)?;
        assert!(similarity.abs() < 1e-4); // Should be close to 0
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_calculator() -> Result<()> {
        let device = Device::Cpu;
        let calculator = CosineSimilarityCalculator::new(device);

        let (original, identical, _, _) = create_test_tensors()?;
        let similarity = calculator.calculate(&original, &identical)?;
        assert!((similarity - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_comprehensive_analysis() -> Result<()> {
        let device = Device::Cpu;
        let calculator = CosineSimilarityCalculator::new(device);

        let (original, _, scaled, _) = create_test_tensors()?;
        let analysis = calculator.calculate_comprehensive(&original, &scaled)?;

        assert!((analysis.cosine_similarity - 1.0).abs() < 1e-6);
        assert!(analysis.angular_distance_deg < 1.0); // Very small angle
        assert_eq!(analysis.similarity_grade, SimilarityQuality::Excellent);
        Ok(())
    }

    #[test]
    fn test_angular_distance_conversion() {
        let similarity = 1.0;
        let angle_rad = cosine_to_angular_distance(similarity);
        assert!((angle_rad - 0.0).abs() < 1e-6);

        let similarity = 0.0;
        let angle_deg = cosine_to_angular_distance_degrees(similarity);
        assert!((angle_deg - 90.0).abs() < 1e-6);

        let similarity = -1.0;
        let angle_deg = cosine_to_angular_distance_degrees(similarity);
        assert!((angle_deg - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_streaming_similarity() -> Result<()> {
        let device = Device::Cpu;
        let large_tensor_a = Tensor::ones((1000, 1000), DType::F32, &device)?;
        let scalar_tensor = Tensor::full(0.5f32, (1000, 1000), &device)?; // Create tensor with same shape
        let large_tensor_b = large_tensor_a.mul(&scalar_tensor)?;

        let similarity =
            calculate_cosine_similarity_streaming(&large_tensor_a, &large_tensor_b, 10000)?;
        assert!((similarity - 1.0).abs() < 1e-6); // Same direction
        Ok(())
    }

    #[test]
    fn test_quality_assessment() {
        let device = Device::Cpu;
        let calculator = CosineSimilarityCalculator::new(device);

        assert_eq!(
            calculator.assess_similarity_quality(0.995),
            SimilarityQuality::Excellent
        );
        assert_eq!(
            calculator.assess_similarity_quality(0.96),
            SimilarityQuality::Good
        );
        assert_eq!(
            calculator.assess_similarity_quality(0.92),
            SimilarityQuality::Fair
        );
        assert_eq!(
            calculator.assess_similarity_quality(0.80),
            SimilarityQuality::Poor
        );
        assert_eq!(
            calculator.assess_similarity_quality(0.50),
            SimilarityQuality::Unacceptable
        );
    }

    #[test]
    fn test_pairwise_similarities() -> Result<()> {
        let device = Device::Cpu;
        let calculator = CosineSimilarityCalculator::new(device.clone());

        let data2 = vec![2.0f32, 2.0, 2.0, 2.0];
        let data3 = vec![-1.0f32, 1.0, -1.0, 1.0];
        let tensors = vec![
            Tensor::ones((2, 2), DType::F32, &device)?,
            Tensor::new(data2.as_slice(), &device)?.reshape((2, 2))?,
            Tensor::new(data3.as_slice(), &device)?.reshape((2, 2))?,
        ];

        let similarities = calculator.calculate_pairwise(&tensors)?;

        // Check diagonal elements
        assert!((similarities[0][0] - 1.0).abs() < 1e-6);
        assert!((similarities[1][1] - 1.0).abs() < 1e-6);
        assert!((similarities[2][2] - 1.0).abs() < 1e-6);

        // Check symmetry
        assert!((similarities[0][1] - similarities[1][0]).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_similarity_statistics() {
        let device = Device::Cpu;
        let calculator = CosineSimilarityCalculator::new(device);

        let similarities = vec![0.9, 0.95, 0.8, 0.99, 0.85];
        let stats = calculator.calculate_statistics(&similarities);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 0.898).abs() < 1e-3);
        assert!(stats.min >= 0.8 && stats.max <= 0.99);
    }

    #[test]
    fn test_batch_calculator() -> Result<()> {
        let device = Device::Cpu;
        let batch_calc = BatchCosineSimilarityCalculator::new(device.clone());

        let data2 = vec![0.5f32, 0.5, 0.5, 0.5];
        let originals = vec![
            Tensor::ones((2, 2), DType::F32, &device)?,
            Tensor::ones((2, 2), DType::F32, &device)?,
        ];
        let quantized = vec![
            Tensor::ones((2, 2), DType::F32, &device)?,
            Tensor::new(data2.as_slice(), &device)?.reshape((2, 2))?,
        ];

        let similarities = batch_calc.calculate_batch(&originals, &quantized)?;
        assert_eq!(similarities.len(), 2);
        assert!((similarities[0] - 1.0).abs() < 1e-6); // Identical
        assert!((similarities[1] - 1.0).abs() < 1e-6); // Same direction
        Ok(())
    }

    #[test]
    fn test_shape_mismatch_error() {
        let device = Device::Cpu;
        let tensor1 = Tensor::ones((2, 2), DType::F32, &device).unwrap();
        let tensor2 = Tensor::ones((3, 3), DType::F32, &device).unwrap();

        let result = calculate_cosine_similarity(&tensor1, &tensor2);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_vector_handling() -> Result<()> {
        let device = Device::Cpu;
        let zero_tensor = Tensor::zeros((2, 2), DType::F32, &device)?;
        let normal_tensor = Tensor::ones((2, 2), DType::F32, &device)?;

        let result = calculate_cosine_similarity(&zero_tensor, &normal_tensor)?;
        assert_eq!(result, 0.0); // Zero vector should give zero similarity
        Ok(())
    }
}
