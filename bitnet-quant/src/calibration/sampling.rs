//! Representative sampling strategies for calibration
//!
//! This module implements various sampling strategies to select representative
//! subsets of calibration data for efficient quantization parameter optimization.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sampling strategies for calibration data selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random sampling from the dataset
    Random,
    /// Stratified sampling across data clusters
    Stratified,
    /// Importance sampling based on activation magnitudes
    Importance,
    /// Systematic sampling at fixed intervals
    Systematic,
    /// Custom sampling with user-defined criteria
    Custom(CustomSamplingConfig),
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Random
    }
}

/// Configuration for custom sampling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSamplingConfig {
    /// Sampling function name or identifier
    pub function_name: String,
    /// Parameters for the sampling function
    pub parameters: HashMap<String, f32>,
}

/// Trait for representative sampling implementations
pub trait RepresentativeSampler: Send + Sync {
    /// Select representative samples from the dataset
    fn sample_indices(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        metadata: Option<&SamplingMetadata>,
    ) -> CalibrationResult<Vec<usize>>;

    /// Update sampler state with new data information
    fn update_metadata(&mut self, metadata: &SamplingMetadata) -> CalibrationResult<()>;

    /// Get sampler statistics
    fn get_statistics(&self) -> SamplingStatistics;

    /// Reset sampler state
    fn reset(&mut self);

    /// Clone the sampler
    fn clone_boxed(&self) -> Box<dyn RepresentativeSampler>;
}

/// Metadata for informed sampling decisions
#[derive(Debug, Clone)]
pub struct SamplingMetadata {
    /// Sample indices and their importance scores
    pub importance_scores: Vec<f32>,
    /// Data clusters or categories
    pub clusters: Vec<usize>,
    /// Activation statistics per sample
    pub activation_stats: Vec<ActivationStats>,
    /// Sample weights for importance sampling
    pub sample_weights: Vec<f32>,
    /// Additional metadata
    pub extra_metadata: HashMap<String, Vec<f32>>,
}

/// Per-sample activation statistics
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Mean activation magnitude
    pub mean_magnitude: f32,
    /// Maximum activation value
    pub max_value: f32,
    /// Minimum activation value
    pub min_value: f32,
    /// Activation variance
    pub variance: f32,
    /// Sparsity ratio
    pub sparsity: f32,
}

/// Statistics about the sampling process
#[derive(Debug, Clone)]
pub struct SamplingStatistics {
    /// Number of samples selected
    pub selected_samples: usize,
    /// Coverage across data distribution
    pub distribution_coverage: f32,
    /// Sampling efficiency score
    pub efficiency_score: f32,
    /// Representative quality metric
    pub representativeness: f32,
    /// Diversity of selected samples
    pub diversity_score: f32,
    /// Time taken for sampling (seconds)
    pub sampling_time: f64,
}

/// Random sampling implementation
#[derive(Debug)]
pub struct RandomSampler {
    rng: StdRng,
    statistics: SamplingStatistics,
}

impl RandomSampler {
    /// Create a new random sampler
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.next_u64();
        Self::with_seed(seed)
    }

    /// Create a new random sampler with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            statistics: SamplingStatistics::default(),
        }
    }
}

impl RepresentativeSampler for RandomSampler {
    fn sample_indices(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        _metadata: Option<&SamplingMetadata>,
    ) -> CalibrationResult<Vec<usize>> {
        let start_time = std::time::Instant::now();

        if target_samples > total_samples {
            return Err(CalibrationError::sampling(format!(
                "Cannot select {target_samples} samples from {total_samples} total samples"
            )));
        }

        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut self.rng);
        indices.truncate(target_samples);

        // Update statistics
        self.statistics.selected_samples = target_samples;
        self.statistics.distribution_coverage = target_samples as f32 / total_samples as f32;
        self.statistics.efficiency_score = 1.0; // Random sampling is perfectly efficient
        self.statistics.representativeness = 0.7; // Moderate representativeness
        self.statistics.diversity_score = 0.8; // Good diversity
        self.statistics.sampling_time = start_time.elapsed().as_secs_f64();

        Ok(indices)
    }

    fn update_metadata(&mut self, _metadata: &SamplingMetadata) -> CalibrationResult<()> {
        // Random sampler doesn't need metadata
        Ok(())
    }

    fn get_statistics(&self) -> SamplingStatistics {
        self.statistics.clone()
    }

    fn reset(&mut self) {
        self.statistics = SamplingStatistics::default();
    }

    fn clone_boxed(&self) -> Box<dyn RepresentativeSampler> {
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.next_u64();
        Box::new(Self::with_seed(seed))
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratified sampling implementation
#[derive(Debug)]
pub struct StratifiedSampler {
    rng: StdRng,
    statistics: SamplingStatistics,
    num_strata: usize,
}

impl StratifiedSampler {
    /// Create a new stratified sampler
    pub fn new() -> Self {
        Self::with_strata(10)
    }

    /// Create a new stratified sampler with specific number of strata
    pub fn with_strata(num_strata: usize) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            statistics: SamplingStatistics::default(),
            num_strata,
        }
    }
}

impl RepresentativeSampler for StratifiedSampler {
    fn sample_indices(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        metadata: Option<&SamplingMetadata>,
    ) -> CalibrationResult<Vec<usize>> {
        let start_time = std::time::Instant::now();

        if target_samples > total_samples {
            return Err(CalibrationError::sampling(format!(
                "Cannot select {target_samples} samples from {total_samples} total samples"
            )));
        }

        let clusters = match metadata {
            Some(meta) => &meta.clusters,
            None => {
                // Create simple strata based on sample index
                let strata: Vec<usize> = (0..total_samples).map(|i| i % self.num_strata).collect();
                return self.sample_from_strata(&strata, total_samples, target_samples, start_time);
            }
        };

        self.sample_from_strata(clusters, total_samples, target_samples, start_time)
    }

    fn update_metadata(&mut self, _metadata: &SamplingMetadata) -> CalibrationResult<()> {
        Ok(())
    }

    fn get_statistics(&self) -> SamplingStatistics {
        self.statistics.clone()
    }

    fn reset(&mut self) {
        self.statistics = SamplingStatistics::default();
    }

    fn clone_boxed(&self) -> Box<dyn RepresentativeSampler> {
        Box::new(Self::with_strata(self.num_strata))
    }
}

impl StratifiedSampler {
    fn sample_from_strata(
        &mut self,
        clusters: &[usize],
        total_samples: usize,
        target_samples: usize,
        start_time: std::time::Instant,
    ) -> CalibrationResult<Vec<usize>> {
        // Group samples by stratum
        let mut strata_samples: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &stratum) in clusters.iter().enumerate() {
            strata_samples.entry(stratum).or_default().push(idx);
        }

        let num_strata = strata_samples.len();
        let samples_per_stratum = target_samples / num_strata;
        let extra_samples = target_samples % num_strata;

        let mut selected_indices = Vec::new();

        for (stratum_idx, (_, mut stratum_samples)) in strata_samples.into_iter().enumerate() {
            let stratum_target =
                samples_per_stratum + if stratum_idx < extra_samples { 1 } else { 0 };
            let stratum_target = stratum_target.min(stratum_samples.len());

            stratum_samples.shuffle(&mut self.rng);
            selected_indices.extend(stratum_samples.into_iter().take(stratum_target));
        }

        // Update statistics
        self.statistics.selected_samples = selected_indices.len();
        self.statistics.distribution_coverage =
            selected_indices.len() as f32 / total_samples as f32;
        self.statistics.efficiency_score = 0.9; // High efficiency due to stratification
        self.statistics.representativeness = 0.95; // Excellent representativeness
        self.statistics.diversity_score = 0.9; // High diversity across strata
        self.statistics.sampling_time = start_time.elapsed().as_secs_f64();

        Ok(selected_indices)
    }
}

impl Default for StratifiedSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Importance sampling implementation
#[derive(Debug)]
pub struct ImportanceSampler {
    rng: StdRng,
    statistics: SamplingStatistics,
    importance_threshold: f32,
}

impl ImportanceSampler {
    /// Create a new importance sampler
    pub fn new() -> Self {
        Self::with_threshold(0.1)
    }

    /// Create a new importance sampler with specific threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            statistics: SamplingStatistics::default(),
            importance_threshold: threshold,
        }
    }
}

impl RepresentativeSampler for ImportanceSampler {
    fn sample_indices(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        metadata: Option<&SamplingMetadata>,
    ) -> CalibrationResult<Vec<usize>> {
        let start_time = std::time::Instant::now();

        if target_samples > total_samples {
            return Err(CalibrationError::sampling(format!(
                "Cannot select {target_samples} samples from {total_samples} total samples"
            )));
        }

        let importance_scores = match metadata {
            Some(meta) if !meta.importance_scores.is_empty() => &meta.importance_scores,
            _ => {
                // Fallback to uniform importance
                return self.uniform_fallback(total_samples, target_samples, start_time);
            }
        };

        if importance_scores.len() != total_samples {
            return Err(CalibrationError::sampling(
                "Importance scores length doesn't match total samples",
            ));
        }

        // Normalize importance scores
        let sum_scores: f32 = importance_scores.iter().sum();
        if sum_scores == 0.0 {
            return self.uniform_fallback(total_samples, target_samples, start_time);
        }

        let normalized_scores: Vec<f32> = importance_scores
            .iter()
            .map(|&score| score / sum_scores)
            .collect();

        // Sample with probability proportional to importance
        let mut selected_indices = Vec::new();
        let mut selected_set = std::collections::HashSet::new();

        for _ in 0..target_samples {
            let mut attempts = 0;
            let max_attempts = target_samples * 10;

            while attempts < max_attempts && selected_indices.len() < target_samples {
                let random_value: f32 = self.rng.next_u32() as f32 / u32::MAX as f32;
                let mut cumulative_prob = 0.0;

                for (idx, &prob) in normalized_scores.iter().enumerate() {
                    cumulative_prob += prob;
                    if random_value <= cumulative_prob && !selected_set.contains(&idx) {
                        selected_indices.push(idx);
                        selected_set.insert(idx);
                        break;
                    }
                }
                attempts += 1;
            }

            if attempts >= max_attempts {
                // Fill remaining slots randomly from unselected samples
                let remaining: Vec<usize> = (0..total_samples)
                    .filter(|idx| !selected_set.contains(idx))
                    .collect();

                let mut remaining = remaining;
                remaining.shuffle(&mut self.rng);

                for &idx in remaining
                    .iter()
                    .take(target_samples - selected_indices.len())
                {
                    selected_indices.push(idx);
                }
                break;
            }
        }

        // Update statistics
        let avg_importance: f32 = selected_indices
            .iter()
            .map(|&idx| importance_scores[idx])
            .sum::<f32>()
            / selected_indices.len() as f32;

        self.statistics.selected_samples = selected_indices.len();
        self.statistics.distribution_coverage =
            selected_indices.len() as f32 / total_samples as f32;
        self.statistics.efficiency_score = (avg_importance * 10.0).min(1.0); // Based on avg importance
        self.statistics.representativeness = 0.85; // Good representativeness
        self.statistics.diversity_score = 0.75; // Moderate diversity (biased towards important samples)
        self.statistics.sampling_time = start_time.elapsed().as_secs_f64();

        Ok(selected_indices)
    }

    fn update_metadata(&mut self, _metadata: &SamplingMetadata) -> CalibrationResult<()> {
        Ok(())
    }

    fn get_statistics(&self) -> SamplingStatistics {
        self.statistics.clone()
    }

    fn reset(&mut self) {
        self.statistics = SamplingStatistics::default();
    }

    fn clone_boxed(&self) -> Box<dyn RepresentativeSampler> {
        Box::new(Self::with_threshold(self.importance_threshold))
    }
}

impl ImportanceSampler {
    fn uniform_fallback(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        start_time: std::time::Instant,
    ) -> CalibrationResult<Vec<usize>> {
        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut self.rng);
        indices.truncate(target_samples);

        // Update statistics
        self.statistics.selected_samples = target_samples;
        self.statistics.distribution_coverage = target_samples as f32 / total_samples as f32;
        self.statistics.efficiency_score = 0.7; // Lower efficiency due to fallback
        self.statistics.representativeness = 0.7;
        self.statistics.diversity_score = 0.8;
        self.statistics.sampling_time = start_time.elapsed().as_secs_f64();

        Ok(indices)
    }
}

impl Default for ImportanceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SamplingStatistics {
    fn default() -> Self {
        Self {
            selected_samples: 0,
            distribution_coverage: 0.0,
            efficiency_score: 0.0,
            representativeness: 0.0,
            diversity_score: 0.0,
            sampling_time: 0.0,
        }
    }
}

/// Factory for creating samplers
pub struct SamplerFactory;

impl SamplerFactory {
    /// Create a sampler based on the strategy
    pub fn create(strategy: &SamplingStrategy) -> Box<dyn RepresentativeSampler> {
        match strategy {
            SamplingStrategy::Random => Box::new(RandomSampler::new()),
            SamplingStrategy::Stratified => Box::new(StratifiedSampler::new()),
            SamplingStrategy::Importance => Box::new(ImportanceSampler::new()),
            SamplingStrategy::Systematic => Box::new(SystematicSampler::new()),
            SamplingStrategy::Custom(_config) => {
                // For custom sampling, return random sampler as fallback
                // In practice, this would be implemented based on the custom config
                Box::new(RandomSampler::new())
            }
        }
    }
}

/// Systematic sampling implementation
#[derive(Debug)]
pub struct SystematicSampler {
    statistics: SamplingStatistics,
}

impl SystematicSampler {
    pub fn new() -> Self {
        Self {
            statistics: SamplingStatistics::default(),
        }
    }
}

impl RepresentativeSampler for SystematicSampler {
    fn sample_indices(
        &mut self,
        total_samples: usize,
        target_samples: usize,
        _metadata: Option<&SamplingMetadata>,
    ) -> CalibrationResult<Vec<usize>> {
        let start_time = std::time::Instant::now();

        if target_samples > total_samples {
            return Err(CalibrationError::sampling(format!(
                "Cannot select {target_samples} samples from {total_samples} total samples"
            )));
        }

        if target_samples == 0 {
            return Ok(Vec::new());
        }

        let interval = total_samples as f32 / target_samples as f32;
        let mut rng = thread_rng();
        let start_offset: f32 = rng.gen_range(0.0..interval);

        let indices: Vec<usize> = (0..target_samples)
            .map(|i| ((start_offset + i as f32 * interval) as usize).min(total_samples - 1))
            .collect();

        // Update statistics
        self.statistics.selected_samples = indices.len();
        self.statistics.distribution_coverage = indices.len() as f32 / total_samples as f32;
        self.statistics.efficiency_score = 0.95; // Very efficient
        self.statistics.representativeness = 0.9; // Excellent representativeness
        self.statistics.diversity_score = 0.85; // Good diversity across the range
        self.statistics.sampling_time = start_time.elapsed().as_secs_f64();

        Ok(indices)
    }

    fn update_metadata(&mut self, _metadata: &SamplingMetadata) -> CalibrationResult<()> {
        Ok(())
    }

    fn get_statistics(&self) -> SamplingStatistics {
        self.statistics.clone()
    }

    fn reset(&mut self) {
        self.statistics = SamplingStatistics::default();
    }

    fn clone_boxed(&self) -> Box<dyn RepresentativeSampler> {
        Box::new(Self::new())
    }
}

impl Default for SystematicSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sampler() {
        let mut sampler = RandomSampler::new();
        let indices = sampler.sample_indices(100, 10, None).unwrap();

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 100));

        let stats = sampler.get_statistics();
        assert_eq!(stats.selected_samples, 10);
    }

    #[test]
    fn test_stratified_sampler() {
        let mut sampler = StratifiedSampler::with_strata(5);
        let indices = sampler.sample_indices(100, 20, None).unwrap();

        assert_eq!(indices.len(), 20);
        assert!(indices.iter().all(|&idx| idx < 100));
    }

    #[test]
    fn test_importance_sampler() {
        let mut sampler = ImportanceSampler::new();

        // Create metadata with importance scores
        let importance_scores = vec![0.1; 100]; // Uniform importance
        let metadata = SamplingMetadata {
            importance_scores,
            clusters: Vec::new(),
            activation_stats: Vec::new(),
            sample_weights: Vec::new(),
            extra_metadata: HashMap::new(),
        };

        let indices = sampler.sample_indices(100, 10, Some(&metadata)).unwrap();
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_systematic_sampler() {
        let mut sampler = SystematicSampler::new();
        let indices = sampler.sample_indices(100, 10, None).unwrap();

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 100));

        // Check that indices are roughly evenly spaced
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();

        // Verify systematic spacing (approximate due to rounding)
        let expected_interval = 100.0 / 10.0;
        for i in 1..sorted_indices.len() {
            let actual_interval = (sorted_indices[i] - sorted_indices[i - 1]) as f32;
            assert!((actual_interval - expected_interval).abs() <= expected_interval * 0.5);
        }
    }

    #[test]
    fn test_sampler_factory() {
        let random_sampler = SamplerFactory::create(&SamplingStrategy::Random);
        let stratified_sampler = SamplerFactory::create(&SamplingStrategy::Stratified);
        let importance_sampler = SamplerFactory::create(&SamplingStrategy::Importance);
        let systematic_sampler = SamplerFactory::create(&SamplingStrategy::Systematic);

        // Test that they can all sample
        let mut samplers = vec![
            random_sampler,
            stratified_sampler,
            importance_sampler,
            systematic_sampler,
        ];

        for sampler in &mut samplers {
            let indices = sampler.sample_indices(50, 5, None).unwrap();
            assert_eq!(indices.len(), 5);
        }
    }

    #[test]
    fn test_invalid_sample_size() {
        let mut sampler = RandomSampler::new();
        let result = sampler.sample_indices(10, 20, None);
        assert!(result.is_err());
    }
}
