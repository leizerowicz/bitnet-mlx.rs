//! Allocation Pattern Learning for Task 1.7.1 - Dynamic Strategy Refinement
//!
//! This module provides learning capabilities for tensor allocation patterns
//! to dynamically refine strategy selection and improve performance over time.
//!
//! Key features:
//! - Track allocation patterns and performance characteristics
//! - Learn optimal allocation strategies based on historical data
//! - Adaptive threshold adjustment for strategy selection
//! - Performance-based learning with feedback loops

use super::{TensorSizeCategory, AllocationStrategy};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant};

#[cfg(feature = "tracing")]
use tracing::{debug, info, trace};

/// Number of samples to keep for learning algorithms
const LEARNING_WINDOW_SIZE: usize = 100;

/// Minimum samples required before learning kicks in
const MIN_SAMPLES_FOR_LEARNING: usize = 10;

/// Learning rate for adaptive threshold adjustment (0.0 - 1.0)
const LEARNING_RATE: f64 = 0.1;

/// Performance difference threshold to trigger strategy adaptation (5%)
const ADAPTATION_THRESHOLD: f64 = 0.05;

/// Allocation pattern sample for learning
#[derive(Debug, Clone)]
struct AllocationSample {
    size_bytes: usize,
    allocation_time: u64,
    strategy_used: AllocationStrategy,
    is_model_weight: bool,
    timestamp: Instant,
}

/// Performance statistics for a specific allocation pattern
#[derive(Debug, Clone, Default)]
struct PatternPerformance {
    samples: VecDeque<AllocationSample>,
    average_time: f64,
    min_time: u64,
    max_time: u64,
    variance: f64,
    sample_count: usize,
}

impl PatternPerformance {
    fn add_sample(&mut self, sample: AllocationSample) {
        // Maintain sliding window
        if self.samples.len() >= LEARNING_WINDOW_SIZE {
            self.samples.pop_front();
        }
        
        self.samples.push_back(sample.clone());
        
        // Update statistics
        self.recalculate_stats();
    }
    
    fn recalculate_stats(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        
        let times: Vec<u64> = self.samples.iter().map(|s| s.allocation_time).collect();
        
        self.sample_count = times.len();
        
        // Calculate basic statistics
        self.min_time = *times.iter().min().unwrap_or(&0);
        self.max_time = *times.iter().max().unwrap_or(&0);
        
        let sum: u64 = times.iter().sum();
        self.average_time = sum as f64 / times.len() as f64;
        
        // Calculate variance
        let variance_sum: f64 = times.iter()
            .map(|&time| {
                let diff = time as f64 - self.average_time;
                diff * diff
            })
            .sum();
        
        self.variance = variance_sum / times.len() as f64;
    }
    
    fn get_consistency_score(&self) -> f64 {
        if self.variance == 0.0 || self.average_time == 0.0 {
            return 100.0;
        }
        
        // Higher consistency means lower relative variance
        let relative_variance = self.variance.sqrt() / self.average_time;
        (1.0 - relative_variance.min(1.0)) * 100.0
    }
}

/// Adaptive threshold for strategy selection
#[derive(Debug, Clone)]
struct AdaptiveThreshold {
    current_value: usize,
    base_value: usize,
    adjustment_factor: f64,
    last_adjustment: Instant,
}

impl AdaptiveThreshold {
    fn new(base_value: usize) -> Self {
        Self {
            current_value: base_value,
            base_value,
            adjustment_factor: 1.0,
            last_adjustment: Instant::now(),
        }
    }
    
    fn adjust(&mut self, performance_difference: f64) {
        // Only adjust if performance difference is significant
        if performance_difference.abs() < ADAPTATION_THRESHOLD {
            return;
        }
        
        // Increase threshold if optimized pool is performing worse than expected
        // Decrease threshold if optimized pool is performing better than expected
        let adjustment = if performance_difference > 0.0 {
            // Standard pool is faster, increase threshold (use standard pool more)
            1.0 + LEARNING_RATE
        } else {
            // Optimized pool is faster, decrease threshold (use optimized pool more)
            1.0 - LEARNING_RATE
        };
        
        self.adjustment_factor = (self.adjustment_factor * adjustment).clamp(0.5, 2.0);
        self.current_value = (self.base_value as f64 * self.adjustment_factor) as usize;
        self.last_adjustment = Instant::now();
        
        #[cfg(feature = "tracing")]
        debug!(
            "Adaptive threshold adjusted: base={}, current={}, factor={:.3}, perf_diff={:.3}",
            self.base_value, self.current_value, self.adjustment_factor, performance_difference
        );
    }
    
    fn get_current_threshold(&self) -> usize {
        self.current_value
    }
}

/// Allocation pattern learning system
pub struct AllocationPatternLearner {
    /// Performance patterns by size category and strategy
    patterns: RwLock<HashMap<(TensorSizeCategory, AllocationStrategy), PatternPerformance>>,
    
    /// Adaptive threshold for strategy selection
    adaptive_threshold: RwLock<AdaptiveThreshold>,
    
    /// Global learning statistics
    total_samples: AtomicUsize,
    learning_cycles: AtomicUsize,
    last_optimization: RwLock<Instant>,
}

impl AllocationPatternLearner {
    /// Create a new allocation pattern learner
    pub fn new(initial_threshold: usize) -> Self {
        Self {
            patterns: RwLock::new(HashMap::new()),
            adaptive_threshold: RwLock::new(AdaptiveThreshold::new(initial_threshold)),
            total_samples: AtomicUsize::new(0),
            learning_cycles: AtomicUsize::new(0),
            last_optimization: RwLock::new(Instant::now()),
        }
    }
    
    /// Record an allocation sample for learning
    pub fn record_allocation(
        &self,
        size_bytes: usize,
        allocation_time: u64,
        strategy_used: AllocationStrategy,
        is_model_weight: bool,
    ) {
        let sample = AllocationSample {
            size_bytes,
            allocation_time,
            strategy_used,
            is_model_weight,
            timestamp: Instant::now(),
        };
        
        let size_category = TensorSizeCategory::from_size(size_bytes);
        let pattern_key = (size_category, strategy_used);
        
        {
            let mut patterns = self.patterns.write().unwrap();
            let pattern = patterns.entry(pattern_key).or_default();
            pattern.add_sample(sample);
        }
        
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        
        // Trigger learning if we have enough samples
        if self.total_samples.load(Ordering::Relaxed) % MIN_SAMPLES_FOR_LEARNING == 0 {
            self.perform_learning_cycle();
        }
    }
    
    /// Get recommended strategy based on learned patterns
    pub fn get_recommended_strategy(
        &self,
        size_bytes: usize,
        is_model_weight: bool,
    ) -> AllocationStrategy {
        let size_category = TensorSizeCategory::from_size(size_bytes);
        let threshold = self.adaptive_threshold.read().unwrap().get_current_threshold();
        
        // Use learned threshold for initial decision
        if size_bytes <= threshold {
            return AllocationStrategy::Standard;
        }
        
        // Check if we have learned patterns for this category
        let patterns = self.patterns.read().unwrap();
        
        let standard_key = (size_category, AllocationStrategy::Standard);
        let optimized_key = (size_category, AllocationStrategy::Optimized);
        
        let standard_perf = patterns.get(&standard_key);
        let optimized_perf = patterns.get(&optimized_key);
        
        match (standard_perf, optimized_perf) {
            (Some(std_perf), Some(opt_perf)) if std_perf.sample_count >= MIN_SAMPLES_FOR_LEARNING && opt_perf.sample_count >= MIN_SAMPLES_FOR_LEARNING => {
                // Use learned performance data for decision
                if std_perf.average_time < opt_perf.average_time {
                    AllocationStrategy::Standard
                } else {
                    AllocationStrategy::Optimized
                }
            }
            _ => {
                // Fallback to size-based heuristic
                if is_model_weight {
                    AllocationStrategy::Optimized
                } else {
                    AllocationStrategy::Standard
                }
            }
        }
    }
    
    /// Perform a learning cycle to update adaptive thresholds
    fn perform_learning_cycle(&self) {
        let patterns = self.patterns.read().unwrap();
        
        // Find performance differences between strategies for each category
        for category in [
            TensorSizeCategory::VerySmall,
            TensorSizeCategory::Small,
            TensorSizeCategory::Medium,
            TensorSizeCategory::Large,
            TensorSizeCategory::VeryLarge,
        ] {
            let standard_key = (category, AllocationStrategy::Standard);
            let optimized_key = (category, AllocationStrategy::Optimized);
            
            if let (Some(std_perf), Some(opt_perf)) = (patterns.get(&standard_key), patterns.get(&optimized_key)) {
                if std_perf.sample_count >= MIN_SAMPLES_FOR_LEARNING && opt_perf.sample_count >= MIN_SAMPLES_FOR_LEARNING {
                    // Calculate performance difference (positive means standard is faster)
                    let perf_diff = (std_perf.average_time - opt_perf.average_time) / std_perf.average_time;
                    
                    // Adjust threshold based on performance difference
                    self.adaptive_threshold.write().unwrap().adjust(perf_diff);
                    
                    #[cfg(feature = "tracing")]
                    trace!(
                        "Learning cycle: category={:?}, std_time={:.2}ns, opt_time={:.2}ns, diff={:.3}",
                        category, std_perf.average_time, opt_perf.average_time, perf_diff
                    );
                }
            }
        }
        
        self.learning_cycles.fetch_add(1, Ordering::Relaxed);
        *self.last_optimization.write().unwrap() = Instant::now();
        
        #[cfg(feature = "tracing")]
        info!(
            "Learning cycle completed: samples={}, cycles={}, current_threshold={}",
            self.total_samples.load(Ordering::Relaxed),
            self.learning_cycles.load(Ordering::Relaxed),
            self.adaptive_threshold.read().unwrap().get_current_threshold()
        );
    }
    
    /// Get learning statistics
    pub fn get_learning_stats(&self) -> LearningStats {
        let patterns = self.patterns.read().unwrap();
        let adaptive_threshold = self.adaptive_threshold.read().unwrap();
        
        let total_patterns = patterns.len();
        let learned_patterns = patterns.values()
            .filter(|p| p.sample_count >= MIN_SAMPLES_FOR_LEARNING)
            .count();
        
        let average_consistency = if learned_patterns > 0 {
            patterns.values()
                .filter(|p| p.sample_count >= MIN_SAMPLES_FOR_LEARNING)
                .map(|p| p.get_consistency_score())
                .sum::<f64>() / learned_patterns as f64
        } else {
            0.0
        };
        
        LearningStats {
            total_samples: self.total_samples.load(Ordering::Relaxed),
            learning_cycles: self.learning_cycles.load(Ordering::Relaxed),
            total_patterns,
            learned_patterns,
            current_threshold: adaptive_threshold.get_current_threshold(),
            threshold_adjustment_factor: adaptive_threshold.adjustment_factor,
            average_consistency_score: average_consistency,
        }
    }
    
    /// Get detailed pattern performance for a specific category and strategy
    pub fn get_pattern_performance(&self, category: TensorSizeCategory, strategy: AllocationStrategy) -> Option<PatternPerformance> {
        let patterns = self.patterns.read().unwrap();
        patterns.get(&(category, strategy)).cloned()
    }
    
    /// Reset learning state (for testing)
    pub fn reset_learning(&self) {
        let mut patterns = self.patterns.write().unwrap();
        patterns.clear();
        
        self.total_samples.store(0, Ordering::Relaxed);
        self.learning_cycles.store(0, Ordering::Relaxed);
        
        // Reset adaptive threshold
        let mut threshold = self.adaptive_threshold.write().unwrap();
        threshold.current_value = threshold.base_value;
        threshold.adjustment_factor = 1.0;
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStats {
    pub total_samples: usize,
    pub learning_cycles: usize,
    pub total_patterns: usize,
    pub learned_patterns: usize,
    pub current_threshold: usize,
    pub threshold_adjustment_factor: f64,
    pub average_consistency_score: f64,
}

impl LearningStats {
    /// Check if learning system is making good progress
    pub fn is_learning_effective(&self) -> bool {
        self.learned_patterns > 0 &&
        self.average_consistency_score > 70.0 && // Good consistency
        self.total_samples >= MIN_SAMPLES_FOR_LEARNING
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_learner_creation() {
        let learner = AllocationPatternLearner::new(32 * 1024);
        let stats = learner.get_learning_stats();
        
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.current_threshold, 32 * 1024);
        assert_eq!(stats.threshold_adjustment_factor, 1.0);
    }
    
    #[test]
    fn test_allocation_recording_and_learning() {
        let learner = AllocationPatternLearner::new(32 * 1024);
        
        // Record several allocations with different patterns
        for i in 0..15 {
            // Small tensors - should prefer standard pool
            learner.record_allocation(1024, 100 + i, AllocationStrategy::Standard, false);
            learner.record_allocation(1024, 300 + i, AllocationStrategy::Optimized, false);
            
            // Large tensors - should prefer optimized pool
            learner.record_allocation(64 * 1024, 500 + i, AllocationStrategy::Standard, true);
            learner.record_allocation(64 * 1024, 200 + i, AllocationStrategy::Optimized, true);
        }
        
        let stats = learner.get_learning_stats();
        
        assert_eq!(stats.total_samples, 60);
        assert!(stats.learning_cycles > 0);
        
        // Test recommendations
        let small_recommendation = learner.get_recommended_strategy(1024, false);
        let large_recommendation = learner.get_recommended_strategy(64 * 1024, true);
        
        // Based on our recorded data, small tensors should use standard, large should use optimized
        assert_eq!(small_recommendation, AllocationStrategy::Standard);
        assert_eq!(large_recommendation, AllocationStrategy::Optimized);
    }
    
    #[test]
    fn test_adaptive_threshold_adjustment() {
        let learner = AllocationPatternLearner::new(32 * 1024);
        
        // Record patterns showing standard pool is consistently faster for medium tensors
        for i in 0..20 {
            learner.record_allocation(16 * 1024, 100, AllocationStrategy::Standard, false);
            learner.record_allocation(16 * 1024, 300, AllocationStrategy::Optimized, false);
        }
        
        let initial_stats = learner.get_learning_stats();
        let initial_threshold = initial_stats.current_threshold;
        
        // After learning, threshold might have adjusted
        let final_stats = learner.get_learning_stats();
        
        assert!(final_stats.learning_cycles > 0);
        println!("Initial threshold: {}, Final threshold: {}", initial_threshold, final_stats.current_threshold);
    }
}
