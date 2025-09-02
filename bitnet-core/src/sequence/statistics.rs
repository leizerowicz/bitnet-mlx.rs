//! Sequence Statistics and Analysis
//!
//! This module provides tools for analyzing sequence length distributions,
//! computing statistics, and generating insights about sequence data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive sequence statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStats {
    /// Total number of sequences analyzed
    pub count: usize,
    /// Total number of tokens across all sequences
    pub total_tokens: usize,
    /// Minimum sequence length
    pub min_length: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Mean sequence length
    pub mean_length: f64,
    /// Median sequence length
    pub median_length: f64,
    /// Standard deviation of sequence lengths
    pub std_dev: f64,
    /// Variance of sequence lengths
    pub variance: f64,
    /// 25th percentile (Q1)
    pub q1: f64,
    /// 75th percentile (Q3)
    pub q3: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
    /// Skewness of the distribution
    pub skewness: f64,
    /// Kurtosis of the distribution
    pub kurtosis: f64,
    /// Length distribution histogram
    pub length_distribution: HashMap<usize, usize>,
    /// Percentile values (0-100)
    pub percentiles: HashMap<u8, f64>,
}

impl SequenceStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            count: 0,
            total_tokens: 0,
            min_length: 0,
            max_length: 0,
            mean_length: 0.0,
            median_length: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            length_distribution: HashMap::new(),
            percentiles: HashMap::new(),
        }
    }

    /// Add a sequence length to the statistics
    pub fn add_sequence_length(&mut self, length: usize) {
        self.count += 1;
        self.total_tokens += length;

        // Update distribution
        *self.length_distribution.entry(length).or_insert(0) += 1;

        // Update min/max
        if self.count == 1 {
            self.min_length = length;
            self.max_length = length;
        } else {
            self.min_length = self.min_length.min(length);
            self.max_length = self.max_length.max(length);
        }

        // Recalculate statistics
        self.recalculate_stats();
    }

    /// Recalculate all derived statistics
    fn recalculate_stats(&mut self) {
        if self.count == 0 {
            return;
        }

        // Get all lengths for calculations
        let mut lengths = Vec::new();
        for (&length, &count) in &self.length_distribution {
            for _ in 0..count {
                lengths.push(length as f64);
            }
        }
        lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Mean
        self.mean_length = self.total_tokens as f64 / self.count as f64;

        // Median
        self.median_length = if lengths.len() % 2 == 0 {
            let mid = lengths.len() / 2;
            (lengths[mid - 1] + lengths[mid]) / 2.0
        } else {
            lengths[lengths.len() / 2]
        };

        // Variance and standard deviation
        let variance_sum: f64 = lengths
            .iter()
            .map(|&x| (x - self.mean_length).powi(2))
            .sum();
        self.variance = variance_sum / self.count as f64;
        self.std_dev = self.variance.sqrt();

        // Quartiles
        self.q1 = percentile(&lengths, 25.0);
        self.q3 = percentile(&lengths, 75.0);
        self.iqr = self.q3 - self.q1;

        // Skewness
        if self.std_dev > 0.0 {
            let skew_sum: f64 = lengths
                .iter()
                .map(|&x| ((x - self.mean_length) / self.std_dev).powi(3))
                .sum();
            self.skewness = skew_sum / self.count as f64;
        }

        // Kurtosis
        if self.std_dev > 0.0 {
            let kurt_sum: f64 = lengths
                .iter()
                .map(|&x| ((x - self.mean_length) / self.std_dev).powi(4))
                .sum();
            self.kurtosis = (kurt_sum / self.count as f64) - 3.0; // Excess kurtosis
        }

        // Calculate percentiles
        self.percentiles.clear();
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99] {
            self.percentiles.insert(p, percentile(&lengths, p as f64));
        }
    }

    /// Get the most common sequence length
    pub fn mode(&self) -> Option<usize> {
        self.length_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&length, _)| length)
    }

    /// Get the coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean_length > 0.0 {
            self.std_dev / self.mean_length
        } else {
            0.0
        }
    }

    /// Check if the distribution is approximately normal
    pub fn is_approximately_normal(&self) -> bool {
        // Simple heuristic: skewness close to 0 and kurtosis close to 0
        self.skewness.abs() < 1.0 && self.kurtosis.abs() < 1.0
    }

    /// Get outlier bounds using IQR method
    pub fn outlier_bounds(&self) -> (f64, f64) {
        let lower_bound = self.q1 - 1.5 * self.iqr;
        let upper_bound = self.q3 + 1.5 * self.iqr;
        (lower_bound, upper_bound)
    }

    /// Count outliers in the dataset
    pub fn count_outliers(&self) -> usize {
        let (lower_bound, upper_bound) = self.outlier_bounds();

        self.length_distribution
            .iter()
            .filter(|(&length, _)| {
                let len_f64 = length as f64;
                len_f64 < lower_bound || len_f64 > upper_bound
            })
            .map(|(_, &count)| count)
            .sum()
    }

    /// Get a summary description of the statistics
    pub fn summary(&self) -> String {
        format!(
            "Sequence Statistics Summary:\n\
             - Count: {} sequences\n\
             - Total tokens: {}\n\
             - Length range: {} - {}\n\
             - Mean: {:.2}\n\
             - Median: {:.2}\n\
             - Std Dev: {:.2}\n\
             - Q1: {:.2}, Q3: {:.2}\n\
             - Skewness: {:.3}\n\
             - Kurtosis: {:.3}\n\
             - Outliers: {}",
            self.count,
            self.total_tokens,
            self.min_length,
            self.max_length,
            self.mean_length,
            self.median_length,
            self.std_dev,
            self.q1,
            self.q3,
            self.skewness,
            self.kurtosis,
            self.count_outliers()
        )
    }
}

impl Default for SequenceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Length distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthDistribution {
    /// Histogram of length frequencies
    pub histogram: HashMap<usize, usize>,
    /// Cumulative distribution
    pub cumulative: HashMap<usize, f64>,
    /// Probability density
    pub density: HashMap<usize, f64>,
    /// Total number of sequences
    pub total_count: usize,
}

impl LengthDistribution {
    /// Create a new length distribution from sequence lengths
    pub fn from_lengths(lengths: &[usize]) -> Self {
        let mut histogram = HashMap::new();
        for &length in lengths {
            *histogram.entry(length).or_insert(0) += 1;
        }

        let total_count = lengths.len();
        let mut cumulative = HashMap::new();
        let mut density = HashMap::new();

        // Calculate density
        for (&length, &count) in &histogram {
            density.insert(length, count as f64 / total_count as f64);
        }

        // Calculate cumulative distribution
        let mut sorted_lengths: Vec<usize> = histogram.keys().copied().collect();
        sorted_lengths.sort();

        let mut cumulative_count = 0;
        for length in sorted_lengths {
            cumulative_count += histogram[&length];
            cumulative.insert(length, cumulative_count as f64 / total_count as f64);
        }

        Self {
            histogram,
            cumulative,
            density,
            total_count,
        }
    }

    /// Get the probability of a specific length
    pub fn probability(&self, length: usize) -> f64 {
        self.density.get(&length).copied().unwrap_or(0.0)
    }

    /// Get the cumulative probability up to a specific length
    pub fn cumulative_probability(&self, length: usize) -> f64 {
        self.cumulative
            .iter()
            .filter(|(&len, _)| len <= length)
            .map(|(_, &prob)| prob)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Find the length at a given percentile
    pub fn length_at_percentile(&self, percentile: f64) -> Option<usize> {
        let target = percentile / 100.0;

        self.cumulative
            .iter()
            .find(|(_, &prob)| prob >= target)
            .map(|(&length, _)| length)
    }
}

/// Analyze sequence lengths and return comprehensive statistics
///
/// # Arguments
/// * `lengths` - Vector of sequence lengths to analyze
///
/// # Returns
/// Comprehensive statistics about the sequence lengths
pub fn analyze_sequence_lengths(lengths: &[usize]) -> SequenceStats {
    let mut stats = SequenceStats::new();

    for &length in lengths {
        stats.add_sequence_length(length);
    }

    stats
}

/// Calculate a specific percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    if p <= 0.0 {
        return sorted_data[0];
    }
    if p >= 100.0 {
        return sorted_data[sorted_data.len() - 1];
    }

    let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Batch statistics for comparing multiple datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Individual statistics for each batch
    pub batch_stats: Vec<SequenceStats>,
    /// Combined statistics across all batches
    pub combined_stats: SequenceStats,
    /// Batch comparison metrics
    pub comparison: BatchComparison,
}

/// Comparison metrics between batches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchComparison {
    /// Variance in mean lengths across batches
    pub mean_variance: f64,
    /// Range of mean lengths
    pub mean_range: (f64, f64),
    /// Consistency score (0-1, higher is more consistent)
    pub consistency_score: f64,
    /// Most similar batch pair
    pub most_similar_batches: Option<(usize, usize, f64)>,
    /// Most different batch pair
    pub most_different_batches: Option<(usize, usize, f64)>,
}

impl BatchStats {
    /// Analyze multiple batches of sequences
    pub fn analyze_batches(batches: &[Vec<usize>]) -> Self {
        let mut batch_stats = Vec::new();
        let mut all_lengths = Vec::new();

        // Analyze each batch individually
        for batch in batches {
            let stats = analyze_sequence_lengths(batch);
            batch_stats.push(stats);
            all_lengths.extend(batch.iter().copied());
        }

        // Combined statistics
        let combined_stats = analyze_sequence_lengths(&all_lengths);

        // Comparison metrics
        let comparison = Self::calculate_comparison(&batch_stats);

        Self {
            batch_stats,
            combined_stats,
            comparison,
        }
    }

    /// Calculate comparison metrics between batches
    fn calculate_comparison(batch_stats: &[SequenceStats]) -> BatchComparison {
        if batch_stats.is_empty() {
            return BatchComparison {
                mean_variance: 0.0,
                mean_range: (0.0, 0.0),
                consistency_score: 1.0,
                most_similar_batches: None,
                most_different_batches: None,
            };
        }

        let means: Vec<f64> = batch_stats.iter().map(|s| s.mean_length).collect();
        let mean_of_means = means.iter().sum::<f64>() / means.len() as f64;

        // Variance in means
        let mean_variance = means
            .iter()
            .map(|&m| (m - mean_of_means).powi(2))
            .sum::<f64>()
            / means.len() as f64;

        // Range of means
        let min_mean = means.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_mean = means.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_range = (min_mean, max_mean);

        // Consistency score (inverse of coefficient of variation)
        let consistency_score = if mean_of_means > 0.0 {
            1.0 / (1.0 + mean_variance.sqrt() / mean_of_means)
        } else {
            1.0
        };

        // Find most similar and different batch pairs
        let mut most_similar = None;
        let mut most_different = None;
        let mut min_distance = f64::INFINITY;
        let mut max_distance = 0.0;

        for i in 0..batch_stats.len() {
            for j in (i + 1)..batch_stats.len() {
                let distance = (batch_stats[i].mean_length - batch_stats[j].mean_length).abs();

                if distance < min_distance {
                    min_distance = distance;
                    most_similar = Some((i, j, distance));
                }

                if distance > max_distance {
                    max_distance = distance;
                    most_different = Some((i, j, distance));
                }
            }
        }

        BatchComparison {
            mean_variance,
            mean_range,
            consistency_score,
            most_similar_batches: most_similar,
            most_different_batches: most_different,
        }
    }
}

/// Sequence length recommendations based on statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthRecommendations {
    /// Recommended maximum length (covers 95% of sequences)
    pub recommended_max_length: usize,
    /// Recommended minimum length (5th percentile)
    pub recommended_min_length: usize,
    /// Optimal batch size based on memory efficiency
    pub optimal_batch_size: usize,
    /// Suggested padding strategy
    pub suggested_padding_strategy: String,
    /// Memory efficiency score
    pub memory_efficiency: f64,
}

impl LengthRecommendations {
    /// Generate recommendations from sequence statistics
    pub fn from_stats(stats: &SequenceStats, target_memory_mb: Option<f64>) -> Self {
        // Use 95th percentile as max length recommendation
        let recommended_max_length = stats
            .percentiles
            .get(&95)
            .map(|&p| p.ceil() as usize)
            .unwrap_or(stats.max_length);

        // Use 5th percentile as min length recommendation
        let recommended_min_length = stats
            .percentiles
            .get(&5)
            .map(|&p| p.floor() as usize)
            .unwrap_or(stats.min_length);

        // Suggest padding strategy based on distribution
        let suggested_padding_strategy = if stats.coefficient_of_variation() < 0.3 {
            "FixedLength".to_string()
        } else if stats.coefficient_of_variation() < 0.7 {
            "LongestInBatch".to_string()
        } else {
            "MultipleOf".to_string()
        };

        // Calculate optimal batch size based on memory target
        let optimal_batch_size = if let Some(target_mb) = target_memory_mb {
            let bytes_per_token = 4; // u32
            let target_bytes = (target_mb * 1024.0 * 1024.0) as usize;
            let avg_tokens_per_sequence = stats.mean_length as usize;
            (target_bytes / (avg_tokens_per_sequence * bytes_per_token)).max(1)
        } else {
            32 // Default batch size
        };

        // Memory efficiency based on padding waste
        let memory_efficiency = if stats.mean_length > 0.0 {
            stats.mean_length / recommended_max_length as f64
        } else {
            1.0
        };

        Self {
            recommended_max_length,
            recommended_min_length,
            optimal_batch_size,
            suggested_padding_strategy,
            memory_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_stats_basic() {
        let lengths = vec![10, 20, 30, 40, 50];
        let stats = analyze_sequence_lengths(&lengths);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.min_length, 10);
        assert_eq!(stats.max_length, 50);
        assert_eq!(stats.mean_length, 30.0);
        assert_eq!(stats.median_length, 30.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_length_distribution() {
        let lengths = vec![5, 5, 10, 10, 10, 15];
        let dist = LengthDistribution::from_lengths(&lengths);

        assert_eq!(dist.total_count, 6);
        assert_eq!(dist.histogram[&5], 2);
        assert_eq!(dist.histogram[&10], 3);
        assert_eq!(dist.histogram[&15], 1);

        // Check probabilities
        assert!((dist.probability(5) - 2.0 / 6.0).abs() < 1e-10);
        assert!((dist.probability(10) - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_detection() {
        let mut stats = SequenceStats::new();

        // Add normal values
        for _ in 0..10 {
            stats.add_sequence_length(100);
        }

        // Add outliers
        stats.add_sequence_length(1); // Low outlier
        stats.add_sequence_length(200); // High outlier

        let outlier_count = stats.count_outliers();
        assert!(outlier_count > 0);
    }

    #[test]
    fn test_batch_stats() {
        let batch1 = vec![10, 20, 30];
        let batch2 = vec![15, 25, 35];
        let batch3 = vec![5, 15, 25];

        let batch_stats = BatchStats::analyze_batches(&[batch1, batch2, batch3]);

        assert_eq!(batch_stats.batch_stats.len(), 3);
        assert_eq!(batch_stats.combined_stats.count, 9);

        // Check that comparison metrics are calculated
        assert!(batch_stats.comparison.consistency_score >= 0.0);
        assert!(batch_stats.comparison.consistency_score <= 1.0);
    }

    #[test]
    fn test_length_recommendations() {
        let lengths = vec![10; 95]; // 95 sequences of length 10
        let mut lengths_with_outliers = lengths;
        lengths_with_outliers.extend(vec![100; 5]); // 5 outliers of length 100 (5% outliers)

        let stats = analyze_sequence_lengths(&lengths_with_outliers);
        let recommendations = LengthRecommendations::from_stats(&stats, Some(1.0)); // 1MB target

        // Should recommend a length that covers most sequences but not outliers
        // With 95% at length 10 and 5% at length 100, the 95th percentile should be close to 10
        assert!(recommendations.recommended_max_length > 10);
        assert!(recommendations.recommended_max_length < 100);
        assert!(recommendations.memory_efficiency > 0.0);
        assert!(recommendations.memory_efficiency <= 1.0);
    }
}
