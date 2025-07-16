//! Allocation Pattern Analysis
//!
//! This module provides analysis of memory allocation patterns to identify
//! optimization opportunities and potential issues.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

use super::AllocationInfo;

/// Pattern analyzer for memory allocations
pub struct PatternAnalyzer {
    /// Detected allocation patterns
    patterns: Arc<Mutex<HashMap<String, AllocationPattern>>>,
    /// Recent allocations for pattern detection
    recent_allocations: Arc<Mutex<VecDeque<AllocationInfo>>>,
    /// Pattern detection configuration
    config: PatternAnalysisConfig,
    /// Analysis statistics
    stats: Arc<Mutex<AnalysisStatistics>>,
}

/// Configuration for pattern analysis
#[derive(Debug, Clone)]
pub struct PatternAnalysisConfig {
    /// Maximum number of recent allocations to analyze
    pub max_recent_allocations: usize,
    /// Minimum occurrences to consider a pattern
    pub min_pattern_occurrences: usize,
    /// Time window for pattern detection
    pub pattern_detection_window: Duration,
    /// Size tolerance for grouping allocations (percentage)
    pub size_tolerance_percent: f64,
    /// Whether to detect temporal patterns
    pub detect_temporal_patterns: bool,
    /// Whether to detect size patterns
    pub detect_size_patterns: bool,
    /// Whether to detect device patterns
    pub detect_device_patterns: bool,
}

/// Detected allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Unique pattern identifier
    pub pattern_id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern characteristics
    pub characteristics: PatternCharacteristics,
    /// Number of allocations matching this pattern
    pub occurrence_count: usize,
    /// Confidence level in pattern detection (0.0 to 1.0)
    pub confidence: f64,
    /// Whether this pattern indicates potential issues
    pub is_problematic: bool,
    /// Severity of issues (if problematic)
    pub severity: f64,
    /// Description of the pattern
    pub description: String,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
    /// When pattern was first detected
    pub first_detected: SystemTime,
    /// When pattern was last updated
    pub last_updated: SystemTime,
}

/// Types of allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Repeated allocations of similar size
    SizePattern {
        /// Average size of allocations
        average_size: usize,
        /// Size variance
        size_variance: f64,
    },
    /// Temporal allocation patterns
    TemporalPattern {
        /// Average interval between allocations
        average_interval: Duration,
        /// Interval variance
        interval_variance: f64,
    },
    /// Device-specific allocation patterns
    DevicePattern {
        /// Primary device type
        device_type: String,
        /// Percentage of allocations on this device
        device_percentage: f64,
    },
    /// Lifecycle patterns (allocation-deallocation pairs)
    LifecyclePattern {
        /// Average lifetime of allocations
        average_lifetime: Duration,
        /// Lifetime variance
        lifetime_variance: f64,
    },
    /// Memory leak patterns
    LeakPattern {
        /// Number of potentially leaked allocations
        leaked_count: usize,
        /// Total potentially leaked bytes
        leaked_bytes: u64,
    },
    /// Fragmentation patterns
    FragmentationPattern {
        /// Fragmentation level (0.0 to 1.0)
        fragmentation_level: f64,
        /// Number of small allocations
        small_allocation_count: usize,
    },
}

/// Characteristics of an allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCharacteristics {
    /// Size range of allocations in this pattern
    pub size_range: (usize, usize),
    /// Device types involved
    pub device_types: Vec<String>,
    /// Pool types involved
    pub pool_types: Vec<String>,
    /// Time range when pattern occurs
    pub time_range: (SystemTime, SystemTime),
    /// Frequency of pattern occurrence
    pub frequency: f64,
    /// Pattern stability (how consistent it is)
    pub stability: f64,
}

/// Statistics about pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Total patterns detected
    pub total_patterns: usize,
    /// Patterns by type
    pub patterns_by_type: HashMap<String, usize>,
    /// Problematic patterns count
    pub problematic_patterns: usize,
    /// Total allocations analyzed
    pub total_allocations_analyzed: usize,
    /// Analysis start time
    pub analysis_start_time: SystemTime,
    /// Last analysis update time
    pub last_analysis_time: SystemTime,
    /// Analysis performance metrics
    pub performance_metrics: AnalysisPerformanceMetrics,
}

/// Performance metrics for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPerformanceMetrics {
    /// Average time to analyze an allocation
    pub avg_analysis_time_ns: u64,
    /// Maximum analysis time
    pub max_analysis_time_ns: u64,
    /// Total analysis time
    pub total_analysis_time_ns: u64,
    /// Number of analysis operations
    pub analysis_operations: u64,
}

/// Report of detected patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternReport {
    /// All detected patterns
    pub patterns: Vec<AllocationPattern>,
    /// Summary statistics
    pub summary: PatternSummary,
    /// Recommendations for optimization
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Report generation timestamp
    pub generated_at: SystemTime,
}

/// Summary of pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSummary {
    /// Total patterns detected
    pub total_patterns: usize,
    /// Most common pattern type
    pub most_common_pattern_type: String,
    /// Most problematic pattern
    pub most_problematic_pattern: Option<String>,
    /// Overall memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
    /// Key insights
    pub insights: Vec<String>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: String,
    /// Priority level (0.0 to 1.0)
    pub priority: f64,
    /// Description of the recommendation
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation difficulty (0.0 to 1.0)
    pub implementation_difficulty: f64,
    /// Related patterns
    pub related_patterns: Vec<String>,
}

impl Default for PatternAnalysisConfig {
    fn default() -> Self {
        Self {
            max_recent_allocations: 10000,
            min_pattern_occurrences: 5,
            pattern_detection_window: Duration::from_secs(300), // 5 minutes
            size_tolerance_percent: 10.0, // 10% tolerance
            detect_temporal_patterns: true,
            detect_size_patterns: true,
            detect_device_patterns: true,
        }
    }
}

impl PatternAnalyzer {
    /// Creates a new pattern analyzer
    ///
    /// # Arguments
    ///
    /// * `config` - Pattern analysis configuration
    ///
    /// # Returns
    ///
    /// New pattern analyzer instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::patterns::{PatternAnalyzer, PatternAnalysisConfig};
    ///
    /// let config = PatternAnalysisConfig::default();
    /// let analyzer = PatternAnalyzer::new(config);
    /// ```
    pub fn new(config: PatternAnalysisConfig) -> Self {
        #[cfg(feature = "tracing")]
        info!("Creating pattern analyzer with config: max_recent={}", config.max_recent_allocations);

        Self {
            patterns: Arc::new(Mutex::new(HashMap::new())),
            recent_allocations: Arc::new(Mutex::new(VecDeque::new())),
            config,
            stats: Arc::new(Mutex::new(AnalysisStatistics::new())),
        }
    }

    /// Records an allocation for pattern analysis
    ///
    /// # Arguments
    ///
    /// * `allocation` - Allocation information to analyze
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::patterns::PatternAnalyzer;
    /// use bitnet_core::memory::tracking::AllocationInfo;
    ///
    /// let analyzer = PatternAnalyzer::new(Default::default());
    /// // analyzer.record_allocation(allocation_info);
    /// ```
    pub fn record_allocation(&self, allocation: AllocationInfo) {
        let analysis_start = std::time::Instant::now();

        // Add to recent allocations
        {
            let mut recent = self.recent_allocations.lock().unwrap();
            recent.push_back(allocation.clone());

            // Maintain size limit
            while recent.len() > self.config.max_recent_allocations {
                recent.pop_front();
            }
        }

        // Analyze patterns
        self.analyze_patterns();

        // Update performance metrics
        let analysis_time = analysis_start.elapsed();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations_analyzed += 1;
            stats.last_analysis_time = SystemTime::now();
            stats.performance_metrics.record_analysis_time(analysis_time);
        }

        #[cfg(feature = "tracing")]
        debug!("Recorded allocation {} for pattern analysis", allocation.id.raw());
    }

    /// Records a deallocation for pattern analysis
    ///
    /// # Arguments
    ///
    /// * `allocation` - Allocation information for the deallocated memory
    pub fn record_deallocation(&self, allocation: AllocationInfo) {
        // Update lifecycle patterns
        self.update_lifecycle_patterns(&allocation);

        #[cfg(feature = "tracing")]
        debug!("Recorded deallocation {} for pattern analysis", allocation.id.raw());
    }

    /// Returns all detected patterns
    ///
    /// # Returns
    ///
    /// Vector of detected allocation patterns
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::patterns::PatternAnalyzer;
    ///
    /// let analyzer = PatternAnalyzer::new(Default::default());
    /// let patterns = analyzer.get_patterns();
    /// println!("Detected {} patterns", patterns.len());
    /// ```
    pub fn get_patterns(&self) -> Vec<AllocationPattern> {
        let patterns = self.patterns.lock().unwrap();
        patterns.values().cloned().collect()
    }

    /// Returns recent patterns (most recently updated)
    ///
    /// # Returns
    ///
    /// Vector of recent allocation patterns
    pub fn get_recent_patterns(&self) -> Vec<AllocationPattern> {
        let patterns = self.patterns.lock().unwrap();
        let mut pattern_list: Vec<_> = patterns.values().cloned().collect();
        
        // Sort by last updated time (most recent first)
        pattern_list.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
        
        pattern_list.into_iter().take(10).collect()
    }

    /// Generates a comprehensive pattern report
    ///
    /// # Returns
    ///
    /// Comprehensive pattern analysis report
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::memory::tracking::patterns::PatternAnalyzer;
    ///
    /// let analyzer = PatternAnalyzer::new(Default::default());
    /// let report = analyzer.generate_report();
    /// println!("Found {} patterns", report.patterns.len());
    /// ```
    pub fn generate_report(&self) -> PatternReport {
        let patterns = self.get_patterns();
        let summary = self.generate_summary(&patterns);
        let recommendations = self.generate_recommendations(&patterns);

        PatternReport {
            patterns,
            summary,
            recommendations,
            generated_at: SystemTime::now(),
        }
    }

    /// Returns analysis statistics
    ///
    /// # Returns
    ///
    /// Current analysis statistics
    pub fn get_statistics(&self) -> AnalysisStatistics {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Estimates memory usage of the pattern analyzer
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let patterns_size = {
            let patterns = self.patterns.lock().unwrap();
            patterns.len() * std::mem::size_of::<AllocationPattern>()
        };

        let recent_size = {
            let recent = self.recent_allocations.lock().unwrap();
            recent.len() * std::mem::size_of::<AllocationInfo>()
        };

        patterns_size + recent_size
    }

    /// Clears all pattern data
    pub fn clear(&self) {
        {
            let mut patterns = self.patterns.lock().unwrap();
            patterns.clear();
        }

        {
            let mut recent = self.recent_allocations.lock().unwrap();
            recent.clear();
        }

        {
            let mut stats = self.stats.lock().unwrap();
            *stats = AnalysisStatistics::new();
        }
    }

    // Private helper methods

    fn analyze_patterns(&self) {
        if self.config.detect_size_patterns {
            self.detect_size_patterns();
        }

        if self.config.detect_temporal_patterns {
            self.detect_temporal_patterns();
        }

        if self.config.detect_device_patterns {
            self.detect_device_patterns();
        }

        self.detect_fragmentation_patterns();
        self.detect_leak_patterns();
    }

    fn detect_size_patterns(&self) {
        let recent = self.recent_allocations.lock().unwrap();
        let mut size_groups: HashMap<usize, Vec<&AllocationInfo>> = HashMap::new();

        // Group allocations by similar sizes
        for allocation in recent.iter() {
            let size_bucket = self.get_size_bucket(allocation.size);
            size_groups.entry(size_bucket).or_default().push(allocation);
        }

        // Analyze each size group
        for (size_bucket, allocations) in size_groups {
            if allocations.len() >= self.config.min_pattern_occurrences {
                let pattern_id = format!("size_pattern_{}", size_bucket);
                let pattern = self.create_size_pattern(&pattern_id, &allocations);
                
                let mut patterns = self.patterns.lock().unwrap();
                patterns.insert(pattern_id, pattern);
            }
        }
    }

    fn detect_temporal_patterns(&self) {
        let recent = self.recent_allocations.lock().unwrap();
        if recent.len() < 2 {
            return;
        }

        // Analyze allocation intervals
        let mut intervals = Vec::new();
        for window in recent.iter().collect::<Vec<_>>().windows(2) {
            if let [prev, curr] = window {
                if let Ok(interval) = curr.timestamp.duration_since(prev.timestamp) {
                    intervals.push(interval);
                }
            }
        }

        if intervals.len() >= self.config.min_pattern_occurrences {
            let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
            let variance = self.calculate_duration_variance(&intervals, avg_interval);

            // If intervals are relatively consistent, it's a temporal pattern
            if variance < 0.5 { // Low variance indicates consistent timing
                let pattern_id = "temporal_pattern".to_string();
                let pattern = AllocationPattern {
                    pattern_id: pattern_id.clone(),
                    pattern_type: PatternType::TemporalPattern {
                        average_interval: avg_interval,
                        interval_variance: variance,
                    },
                    characteristics: self.create_temporal_characteristics(&recent),
                    occurrence_count: intervals.len(),
                    confidence: 1.0 - variance, // Higher confidence for lower variance
                    is_problematic: false,
                    severity: 0.0,
                    description: format!("Regular allocation pattern with {}ms intervals", 
                                       avg_interval.as_millis()),
                    recommendations: vec!["Consider pre-allocating memory if pattern is predictable".to_string()],
                    first_detected: SystemTime::now(),
                    last_updated: SystemTime::now(),
                };

                let mut patterns = self.patterns.lock().unwrap();
                patterns.insert(pattern_id, pattern);
            }
        }
    }

    fn detect_device_patterns(&self) {
        let recent = self.recent_allocations.lock().unwrap();
        let mut device_counts: HashMap<String, usize> = HashMap::new();

        for allocation in recent.iter() {
            *device_counts.entry(allocation.device_type.clone()).or_insert(0) += 1;
        }

        for (device_type, count) in device_counts {
            if count >= self.config.min_pattern_occurrences {
                let percentage = count as f64 / recent.len() as f64;
                let pattern_id = format!("device_pattern_{}", device_type);
                
                let pattern = AllocationPattern {
                    pattern_id: pattern_id.clone(),
                    pattern_type: PatternType::DevicePattern {
                        device_type: device_type.clone(),
                        device_percentage: percentage,
                    },
                    characteristics: self.create_device_characteristics(&device_type, &recent),
                    occurrence_count: count,
                    confidence: percentage,
                    is_problematic: false,
                    severity: 0.0,
                    description: format!("{}% of allocations on {}", 
                                       (percentage * 100.0) as u32, device_type),
                    recommendations: if percentage > 0.8 {
                        vec!["Consider load balancing across devices".to_string()]
                    } else {
                        vec![]
                    },
                    first_detected: SystemTime::now(),
                    last_updated: SystemTime::now(),
                };

                let mut patterns = self.patterns.lock().unwrap();
                patterns.insert(pattern_id, pattern);
            }
        }
    }

    fn detect_fragmentation_patterns(&self) {
        let recent = self.recent_allocations.lock().unwrap();
        let small_threshold = 1024; // 1KB
        let small_allocations = recent.iter()
            .filter(|a| a.size <= small_threshold)
            .count();

        if small_allocations >= self.config.min_pattern_occurrences {
            let fragmentation_level = small_allocations as f64 / recent.len() as f64;
            
            if fragmentation_level > 0.3 { // More than 30% small allocations
                let pattern_id = "fragmentation_pattern".to_string();
                let pattern = AllocationPattern {
                    pattern_id: pattern_id.clone(),
                    pattern_type: PatternType::FragmentationPattern {
                        fragmentation_level,
                        small_allocation_count: small_allocations,
                    },
                    characteristics: self.create_fragmentation_characteristics(&recent),
                    occurrence_count: small_allocations,
                    confidence: fragmentation_level,
                    is_problematic: fragmentation_level > 0.5,
                    severity: fragmentation_level,
                    description: format!("High fragmentation: {}% small allocations", 
                                       (fragmentation_level * 100.0) as u32),
                    recommendations: vec![
                        "Consider using memory pools for small allocations".to_string(),
                        "Batch small allocations together".to_string(),
                    ],
                    first_detected: SystemTime::now(),
                    last_updated: SystemTime::now(),
                };

                let mut patterns = self.patterns.lock().unwrap();
                patterns.insert(pattern_id, pattern);
            }
        }
    }

    fn detect_leak_patterns(&self) {
        let recent = self.recent_allocations.lock().unwrap();
        let old_threshold = Duration::from_secs(300); // 5 minutes
        let now = SystemTime::now();
        
        let old_allocations: Vec<_> = recent.iter()
            .filter(|a| a.is_active && now.duration_since(a.timestamp).unwrap_or(Duration::ZERO) > old_threshold)
            .collect();

        if !old_allocations.is_empty() {
            let leaked_bytes: u64 = old_allocations.iter().map(|a| a.size as u64).sum();
            let pattern_id = "leak_pattern".to_string();
            
            let pattern = AllocationPattern {
                pattern_id: pattern_id.clone(),
                pattern_type: PatternType::LeakPattern {
                    leaked_count: old_allocations.len(),
                    leaked_bytes,
                },
                characteristics: self.create_leak_characteristics(&old_allocations),
                occurrence_count: old_allocations.len(),
                confidence: 0.7, // Medium confidence for leak detection
                is_problematic: true,
                severity: (old_allocations.len() as f64 / recent.len() as f64).min(1.0),
                description: format!("Potential memory leak: {} old allocations ({} bytes)", 
                                   old_allocations.len(), leaked_bytes),
                recommendations: vec![
                    "Investigate long-lived allocations".to_string(),
                    "Check for missing deallocations".to_string(),
                ],
                first_detected: SystemTime::now(),
                last_updated: SystemTime::now(),
            };

            let mut patterns = self.patterns.lock().unwrap();
            patterns.insert(pattern_id, pattern);
        }
    }

    fn update_lifecycle_patterns(&self, _allocation: &AllocationInfo) {
        // Update lifecycle patterns based on deallocation
        // This would analyze allocation-deallocation pairs
        // Implementation would track lifetimes and update patterns
    }

    fn get_size_bucket(&self, size: usize) -> usize {
        // Group sizes into buckets with tolerance
        let tolerance = (size as f64 * self.config.size_tolerance_percent / 100.0) as usize;
        (size / tolerance.max(1)) * tolerance.max(1)
    }

    fn create_size_pattern(&self, pattern_id: &str, allocations: &[&AllocationInfo]) -> AllocationPattern {
        let sizes: Vec<usize> = allocations.iter().map(|a| a.size).collect();
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        let variance = self.calculate_size_variance(&sizes, avg_size);

        AllocationPattern {
            pattern_id: pattern_id.to_string(),
            pattern_type: PatternType::SizePattern {
                average_size: avg_size,
                size_variance: variance,
            },
            characteristics: self.create_size_characteristics(allocations),
            occurrence_count: allocations.len(),
            confidence: 1.0 - variance.min(1.0),
            is_problematic: false,
            severity: 0.0,
            description: format!("Repeated allocations of ~{} bytes", avg_size),
            recommendations: vec!["Consider using a dedicated pool for this size".to_string()],
            first_detected: SystemTime::now(),
            last_updated: SystemTime::now(),
        }
    }

    fn create_size_characteristics(&self, allocations: &[&AllocationInfo]) -> PatternCharacteristics {
        let sizes: Vec<usize> = allocations.iter().map(|a| a.size).collect();
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let max_size = *sizes.iter().max().unwrap_or(&0);
        
        let device_types: std::collections::HashSet<_> = allocations.iter()
            .map(|a| a.device_type.clone())
            .collect();
        
        let pool_types: std::collections::HashSet<_> = allocations.iter()
            .map(|a| a.pool_type.clone())
            .collect();

        let timestamps: Vec<SystemTime> = allocations.iter().map(|a| a.timestamp).collect();
        let min_time = *timestamps.iter().min().unwrap_or(&SystemTime::now());
        let max_time = *timestamps.iter().max().unwrap_or(&SystemTime::now());

        PatternCharacteristics {
            size_range: (min_size, max_size),
            device_types: device_types.into_iter().collect(),
            pool_types: pool_types.into_iter().collect(),
            time_range: (min_time, max_time),
            frequency: allocations.len() as f64,
            stability: 0.8, // Placeholder
        }
    }

    fn create_temporal_characteristics(&self, allocations: &VecDeque<AllocationInfo>) -> PatternCharacteristics {
        let sizes: Vec<usize> = allocations.iter().map(|a| a.size).collect();
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let max_size = *sizes.iter().max().unwrap_or(&0);

        PatternCharacteristics {
            size_range: (min_size, max_size),
            device_types: vec!["Mixed".to_string()],
            pool_types: vec!["Mixed".to_string()],
            time_range: (SystemTime::now(), SystemTime::now()),
            frequency: allocations.len() as f64,
            stability: 0.9, // High stability for temporal patterns
        }
    }

    fn create_device_characteristics(&self, device_type: &str, allocations: &VecDeque<AllocationInfo>) -> PatternCharacteristics {
        let device_allocations: Vec<_> = allocations.iter()
            .filter(|a| a.device_type == device_type)
            .collect();

        let sizes: Vec<usize> = device_allocations.iter().map(|a| a.size).collect();
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let max_size = *sizes.iter().max().unwrap_or(&0);

        PatternCharacteristics {
            size_range: (min_size, max_size),
            device_types: vec![device_type.to_string()],
            pool_types: vec!["Mixed".to_string()],
            time_range: (SystemTime::now(), SystemTime::now()),
            frequency: device_allocations.len() as f64,
            stability: 0.7,
        }
    }

    fn create_fragmentation_characteristics(&self, allocations: &VecDeque<AllocationInfo>) -> PatternCharacteristics {
        PatternCharacteristics {
            size_range: (0, 1024), // Small allocations
            device_types: vec!["Mixed".to_string()],
            pool_types: vec!["Mixed".to_string()],
            time_range: (SystemTime::now(), SystemTime::now()),
            frequency: allocations.len() as f64,
            stability: 0.6,
        }
    }

    fn create_leak_characteristics(&self, allocations: &[&AllocationInfo]) -> PatternCharacteristics {
        let sizes: Vec<usize> = allocations.iter().map(|a| a.size).collect();
        let min_size = *sizes.iter().min().unwrap_or(&0);
        let max_size = *sizes.iter().max().unwrap_or(&0);

        PatternCharacteristics {
            size_range: (min_size, max_size),
            device_types: vec!["Mixed".to_string()],
            pool_types: vec!["Mixed".to_string()],
            time_range: (SystemTime::now(), SystemTime::now()),
            frequency: allocations.len() as f64,
            stability: 0.5, // Low stability for leaks
        }
    }

    fn calculate_size_variance(&self, sizes: &[usize], average: usize) -> f64 {
        if sizes.is_empty() {
            return 0.0;
        }

        let variance: f64 = sizes.iter()
            .map(|&size| {
                let diff = size as f64 - average as f64;
                diff * diff
            })
            .sum::<f64>() / sizes.len() as f64;

        (variance.sqrt() / average as f64).min(1.0)
    }

    fn calculate_duration_variance(&self, durations: &[Duration], average: Duration) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }

        let avg_secs = average.as_secs_f64();
        let variance: f64 = durations.iter()
            .map(|duration| {
                let diff = duration.as_secs_f64() - avg_secs;
                diff * diff
            })
            .sum::<f64>() / durations.len() as f64;

        if avg_secs > 0.0 {
            (variance.sqrt() / avg_secs).min(1.0)
        } else {
            0.0
        }
    }

    fn generate_summary(&self, patterns: &[AllocationPattern]) -> PatternSummary {
        let total_patterns = patterns.len();
        
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for pattern in patterns {
            let type_name = match &pattern.pattern_type {
                PatternType::SizePattern { .. } => "Size",
                PatternType::TemporalPattern { .. } => "Temporal",
                PatternType::DevicePattern { .. } => "Device",
                PatternType::LifecyclePattern { .. } => "Lifecycle",
                PatternType::LeakPattern { .. } => "Leak",
                PatternType::FragmentationPattern { .. } => "Fragmentation",
            };
            *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
        }

        let most_common_pattern_type = type_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "None".to_string());

        let most_problematic_pattern = patterns.iter()
            .filter(|p| p.is_problematic)
            .max_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap())
            .map(|p| p.pattern_id.clone());

        let problematic_count = patterns.iter().filter(|p| p.is_problematic).count();
        let efficiency_score = if total_patterns > 0 {
            1.0 - (problematic_count as f64 / total_patterns as f64)
        } else {
            1.0
        };

        let insights = self.generate_insights(patterns);

        PatternSummary {
            total_patterns,
            most_common_pattern_type,
            most_problematic_pattern,
            efficiency_score,
            insights,
        }
    }

    fn generate_insights(&self, patterns: &[AllocationPattern]) -> Vec<String> {
        let mut insights = Vec::new();

        let problematic_count = patterns.iter().filter(|p| p.is_problematic).count();
        if problematic_count > 0 {
            insights.push(format!("{} patterns indicate potential issues", problematic_count));
        }

        let leak_patterns = patterns.iter().filter(|p| matches!(p.pattern_type, PatternType::LeakPattern { .. })).count();
        if leak_patterns > 0 {
            insights.push("Memory leak patterns detected".to_string());
        }

        let fragmentation_patterns = patterns.iter().filter(|p| matches!(p.pattern_type, PatternType::FragmentationPattern { .. })).count();
        if fragmentation_patterns > 0 {
            insights.push("Memory fragmentation detected".to_string());
        }

        if insights.is_empty() {
            insights.push("No significant issues detected".to_string());
        }

        insights
    }

    fn generate_recommendations(&self, patterns: &[AllocationPattern]) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check for fragmentation patterns
        for pattern in patterns {
            if let PatternType::FragmentationPattern { fragmentation_level, .. } = &pattern.pattern_type {
                if *fragmentation_level > 0.5 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "Reduce Fragmentation".to_string(),
                        priority: *fragmentation_level,
                        description: "High memory fragmentation detected. Consider using memory pools for small allocations.".to_string(),
                        expected_impact: "Reduced memory overhead and improved allocation performance".to_string(),
                        implementation_difficulty: 0.3,
                        related_patterns: vec![pattern.pattern_id.clone()],
                    });
                }
            }
        }

        // Check for leak patterns
        for pattern in patterns {
            if let PatternType::LeakPattern { leaked_bytes, .. } = &pattern.pattern_type {
                if *leaked_bytes > 1024 * 1024 { // > 1MB
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "Fix Memory Leaks".to_string(),
                        priority: 0.9,
                        description: "Potential memory leaks detected. Review allocation lifetimes.".to_string(),
                        expected_impact: "Reduced memory usage and improved stability".to_string(),
                        implementation_difficulty: 0.7,
                        related_patterns: vec![pattern.pattern_id.clone()],
                    });
                }
            }
        }

        // Check for size patterns that could benefit from pooling
        for pattern in patterns {
            if let PatternType::SizePattern { average_size, .. } = &pattern.pattern_type {
                if pattern.occurrence_count > 100 && *average_size < 1024 * 1024 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: "Use Memory Pool".to_string(),
                        priority: 0.6,
                        description: format!("Frequent allocations of ~{} bytes detected. Consider using a dedicated memory pool.", average_size),
                        expected_impact: "Improved allocation performance and reduced fragmentation".to_string(),
                        implementation_difficulty: 0.4,
                        related_patterns: vec![pattern.pattern_id.clone()],
                    });
                }
            }
        }

        recommendations
    }
}

impl AnalysisStatistics {
    fn new() -> Self {
        Self {
            total_patterns: 0,
            patterns_by_type: HashMap::new(),
            problematic_patterns: 0,
            total_allocations_analyzed: 0,
            analysis_start_time: SystemTime::now(),
            last_analysis_time: SystemTime::now(),
            performance_metrics: AnalysisPerformanceMetrics::new(),
        }
    }
}

impl AnalysisPerformanceMetrics {
    fn new() -> Self {
        Self {
            avg_analysis_time_ns: 0,
            max_analysis_time_ns: 0,
            total_analysis_time_ns: 0,
            analysis_operations: 0,
        }
    }

    fn record_analysis_time(&mut self, duration: std::time::Duration) {
        let duration_ns = duration.as_nanos() as u64;
        self.total_analysis_time_ns += duration_ns;
        self.analysis_operations += 1;
        self.max_analysis_time_ns = self.max_analysis_time_ns.max(duration_ns);
        self.avg_analysis_time_ns = self.total_analysis_time_ns / self.analysis_operations;
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for PatternAnalyzer {}
unsafe impl Sync for PatternAnalyzer {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_pattern_analyzer_creation() {
        let config = PatternAnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config);
        
        let patterns = analyzer.get_patterns();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_size_pattern_detection() {
        let config = PatternAnalysisConfig {
            min_pattern_occurrences: 3,
            ..Default::default()
        };
        let analyzer = PatternAnalyzer::new(config);
        
        // Add similar-sized allocations
        for i in 0..5 {
            let allocation = AllocationInfo {
                id: AllocationId::new(i),
                size: 1024, // Same size
                alignment: 16,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
                elapsed: Duration::from_millis(100),
                stack_trace: None,
                pool_type: "SmallBlock".to_string(),
                is_active: true,
            };
            analyzer.record_allocation(allocation);
        }
        
        let patterns = analyzer.get_patterns();
        assert!(!patterns.is_empty());
        
        // Should detect a size pattern
        let size_patterns: Vec<_> = patterns.iter()
            .filter(|p| matches!(p.pattern_type, PatternType::SizePattern { .. }))
            .collect();
        assert!(!size_patterns.is_empty());
    }

    #[test]
    fn test_fragmentation_pattern_detection() {
        let config = PatternAnalysisConfig {
            min_pattern_occurrences: 3,
            ..Default::default()
        };
        let analyzer = PatternAnalyzer::new(config);
        
        // Add many small allocations
        for i in 0..10 {
            let allocation = AllocationInfo {
                id: AllocationId::new(i),
                size: 64, // Small size
                alignment: 8,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
                elapsed: Duration::from_millis(100),
                stack_trace: None,
                pool_type: "SmallBlock".to_string(),
                is_active: true,
            };
            analyzer.record_allocation(allocation);
        }
        
        let patterns = analyzer.get_patterns();
        let fragmentation_patterns: Vec<_> = patterns.iter()
            .filter(|p| matches!(p.pattern_type, PatternType::FragmentationPattern { .. }))
            .collect();
        
        assert!(!fragmentation_patterns.is_empty());
    }

    #[test]
    fn test_leak_pattern_detection() {
        let config = PatternAnalysisConfig {
            min_pattern_occurrences: 1,
            ..Default::default()
        };
        let analyzer = PatternAnalyzer::new(config);
        
        // Add old allocation (potential leak)
        let old_allocation = AllocationInfo {
            id: AllocationId::new(1),
            size: 1024 * 1024, // 1MB
            alignment: 16,
            device_type: "CPU".to_string(),
            timestamp: SystemTime::now() - Duration::from_secs(600), // 10 minutes ago
            elapsed: Duration::from_secs(600),
            stack_trace: None,
            pool_type: "LargeBlock".to_string(),
            is_active: true, // Still active (not deallocated)
        };
        analyzer.record_allocation(old_allocation);
        
        let patterns = analyzer.get_patterns();
        let leak_patterns: Vec<_> = patterns.iter()
            .filter(|p| matches!(p.pattern_type, PatternType::LeakPattern { .. }))
            .collect();
        
        assert!(!leak_patterns.is_empty());
    }

    #[test]
    fn test_pattern_report_generation() {
        let config = PatternAnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config);
        
        // Add some allocations
        for i in 0..5 {
            let allocation = AllocationInfo {
                id: AllocationId::new(i),
                size: 1024,
                alignment: 16,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
                elapsed: Duration::from_millis(100),
                stack_trace: None,
                pool_type: "SmallBlock".to_string(),
                is_active: true,
            };
            analyzer.record_allocation(allocation);
        }
        
        let report = analyzer.generate_report();
        assert!(report.generated_at <= SystemTime::now());
        assert!(!report.summary.insights.is_empty());
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = PatternAnalysisConfig::default();
        let analyzer = PatternAnalyzer::new(config);
        
        let initial_usage = analyzer.estimated_memory_usage();
        
        // Add some allocations
        for i in 0..10 {
            let allocation = AllocationInfo {
                id: AllocationId::new(i),
                size: 1024,
                alignment: 16,
                device_type: "CPU".to_string(),
                timestamp: SystemTime::now(),
                elapsed: Duration::from_millis(100),
                stack_trace: None,
                pool_type: "SmallBlock".to_string(),
                is_active: true,
            };
            analyzer.record_allocation(allocation);
        }
        
        let usage_after = analyzer.estimated_memory_usage();
        assert!(usage_after > initial_usage);
    }
}