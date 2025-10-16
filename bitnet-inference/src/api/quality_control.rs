//! Quality Control Features for BitNet Text Generation
//!
//! This module implements comprehensive quality control mechanisms:
//! - Repetition penalty to prevent repetitive text
//! - Length penalty to control sequence length bias  
//! - Frequency penalty to encourage vocabulary diversity
//! - Content filtering and safety measures
//! - Production-ready quality assurance

use crate::{Result, InferenceError};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for quality control mechanisms
#[derive(Debug, Clone)]
pub struct QualityControlConfig {
    /// Repetition penalty settings
    pub repetition_config: RepetitionPenaltyConfig,
    /// Length penalty settings
    pub length_config: LengthPenaltyConfig,
    /// Frequency penalty settings
    pub frequency_config: FrequencyPenaltyConfig,
    /// Content filtering settings
    pub content_filter_config: ContentFilterConfig,
    /// Quality assurance settings
    pub qa_config: QualityAssuranceConfig,
}

/// Configuration for repetition penalty
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyConfig {
    /// Base repetition penalty factor (1.0 = no penalty, >1.0 = penalty)
    pub penalty_factor: f32,
    /// Window size for repetition detection (tokens to look back)
    pub window_size: usize,
    /// Minimum repetition length to penalize (number of tokens)
    pub min_repetition_length: usize,
    /// Decay factor for older repetitions (0.0-1.0)
    pub decay_factor: f32,
    /// Enable adaptive penalty (adjust based on content)
    pub adaptive_penalty: bool,
    /// Penalty scaling strategy
    pub scaling_strategy: PenaltyScalingStrategy,
}

/// Configuration for length penalty
#[derive(Debug, Clone)]
pub struct LengthPenaltyConfig {
    /// Length penalty factor (1.0 = no penalty, >1.0 = longer bias, <1.0 = shorter bias)
    pub penalty_factor: f32,
    /// Target length for optimal scoring
    pub target_length: Option<usize>,
    /// Length normalization strategy
    pub normalization_strategy: LengthNormalizationStrategy,
    /// Enable dynamic length adjustment
    pub dynamic_adjustment: bool,
    /// Minimum length threshold
    pub min_length_threshold: usize,
    /// Maximum length threshold
    pub max_length_threshold: usize,
}

/// Configuration for frequency penalty
#[derive(Debug, Clone)]
pub struct FrequencyPenaltyConfig {
    /// Base frequency penalty factor
    pub penalty_factor: f32,
    /// Presence penalty (penalize any repeated token)
    pub presence_penalty: f32,
    /// Token counting strategy
    pub counting_strategy: TokenCountingStrategy,
    /// Frequency calculation method
    pub frequency_method: FrequencyCalculationMethod,
    /// Enable vocabulary diversity encouragement
    pub encourage_diversity: bool,
    /// Rare token bonus factor
    pub rare_token_bonus: f32,
}

/// Configuration for content filtering
#[derive(Debug, Clone)]
pub struct ContentFilterConfig {
    /// Enable content safety filtering
    pub enable_safety_filter: bool,
    /// Profanity filtering settings
    pub profanity_filter: ProfanityFilterConfig,
    /// Toxicity detection settings
    pub toxicity_filter: ToxicityFilterConfig,
    /// Custom content filters
    pub custom_filters: Vec<CustomFilterConfig>,
    /// Filter enforcement level
    pub enforcement_level: FilterEnforcementLevel,
}

/// Configuration for quality assurance
#[derive(Debug, Clone)]
pub struct QualityAssuranceConfig {
    /// Enable coherence checking
    pub enable_coherence_check: bool,
    /// Enable grammatical validation
    pub enable_grammar_check: bool,
    /// Enable factual consistency checking
    pub enable_factual_check: bool,
    /// Quality scoring thresholds
    pub quality_thresholds: QualityThresholds,
    /// Enable real-time quality monitoring
    pub enable_realtime_monitoring: bool,
}

/// Strategies for penalty scaling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyScalingStrategy {
    /// Linear scaling with repetition count
    Linear,
    /// Exponential scaling for stronger penalty
    Exponential,
    /// Logarithmic scaling for gradual penalty
    Logarithmic,
    /// Adaptive scaling based on context
    Adaptive,
}

/// Strategies for length normalization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LengthNormalizationStrategy {
    /// No normalization
    None,
    /// Simple length normalization
    Simple,
    /// Wu et al. (2016) length penalty
    Wu2016,
    /// Custom normalization function
    Custom,
}

/// Strategies for token counting in frequency penalty
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenCountingStrategy {
    /// Count all tokens equally
    Simple,
    /// Weight by recency (recent tokens count more)
    RecencyWeighted,
    /// Weight by importance (important tokens count more)
    ImportanceWeighted,
    /// Sliding window counting
    SlidingWindow,
}

/// Methods for calculating frequency penalty
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrequencyCalculationMethod {
    /// Simple frequency count
    Simple,
    /// TF-IDF based frequency
    TFIDF,
    /// Normalized frequency
    Normalized,
    /// Inverse frequency
    Inverse,
}

/// Configuration for profanity filtering
#[derive(Debug, Clone)]
pub struct ProfanityFilterConfig {
    /// Enable profanity detection
    pub enabled: bool,
    /// Strictness level (1-10)
    pub strictness_level: u8,
    /// Custom profanity word list
    pub custom_wordlist: Vec<String>,
    /// Action when profanity detected
    pub action: FilterActionType,
}

/// Configuration for toxicity filtering
#[derive(Debug, Clone)]
pub struct ToxicityFilterConfig {
    /// Enable toxicity detection
    pub enabled: bool,
    /// Toxicity threshold (0.0-1.0)
    pub threshold: f32,
    /// Detection model to use
    pub detection_model: ToxicityDetectionModel,
    /// Action when toxicity detected
    pub action: FilterActionType,
}

/// Configuration for custom content filters
#[derive(Debug, Clone)]
pub struct CustomFilterConfig {
    /// Filter name/identifier
    pub name: String,
    /// Filter pattern or rule
    pub pattern: FilterPattern,
    /// Filter severity level
    pub severity: FilterSeverity,
    /// Action when filter triggered
    pub action: FilterActionType,
}

/// Filter enforcement levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterEnforcementLevel {
    /// Log warnings but allow content
    Warn,
    /// Filter content but continue generation
    Filter,
    /// Stop generation when triggered
    Stop,
    /// Strict enforcement - reject any violations
    Strict,
}

/// Actions to take when filters are triggered
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterActionType {
    /// Log the violation
    Log,
    /// Replace with safe alternative
    Replace,
    /// Skip the problematic token
    Skip,
    /// Stop generation entirely
    Stop,
}

/// Toxicity detection model types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityDetectionModel {
    /// Built-in simple keyword matching
    Simple,
    /// ML-based toxicity detection
    MLBased,
    /// External API-based detection
    External,
}

/// Filter pattern types
#[derive(Debug, Clone)]
pub enum FilterPattern {
    /// Regex pattern
    Regex(String),
    /// Exact word list
    WordList(Vec<String>),
    /// Semantic pattern
    Semantic(String),
    /// Custom function
    Custom(String), // Function identifier
}

/// Filter severity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum FilterSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Quality thresholds for various metrics
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum coherence score (0.0-1.0)
    pub min_coherence_score: f32,
    /// Minimum grammar score (0.0-1.0)
    pub min_grammar_score: f32,
    /// Maximum repetition ratio (0.0-1.0)
    pub max_repetition_ratio: f32,
    /// Minimum diversity score (0.0-1.0)
    pub min_diversity_score: f32,
}

impl Default for QualityControlConfig {
    fn default() -> Self {
        Self {
            repetition_config: RepetitionPenaltyConfig::default(),
            length_config: LengthPenaltyConfig::default(),
            frequency_config: FrequencyPenaltyConfig::default(),
            content_filter_config: ContentFilterConfig::default(),
            qa_config: QualityAssuranceConfig::default(),
        }
    }
}

impl Default for RepetitionPenaltyConfig {
    fn default() -> Self {
        Self {
            penalty_factor: 1.1,
            window_size: 1024,
            min_repetition_length: 3,
            decay_factor: 0.95,
            adaptive_penalty: true,
            scaling_strategy: PenaltyScalingStrategy::Adaptive,
        }
    }
}

impl Default for LengthPenaltyConfig {
    fn default() -> Self {
        Self {
            penalty_factor: 1.0,
            target_length: None,
            normalization_strategy: LengthNormalizationStrategy::Wu2016,
            dynamic_adjustment: true,
            min_length_threshold: 10,
            max_length_threshold: 2048,
        }
    }
}

impl Default for FrequencyPenaltyConfig {
    fn default() -> Self {
        Self {
            penalty_factor: 0.1,
            presence_penalty: 0.1,
            counting_strategy: TokenCountingStrategy::RecencyWeighted,
            frequency_method: FrequencyCalculationMethod::Normalized,
            encourage_diversity: true,
            rare_token_bonus: 0.05,
        }
    }
}

impl Default for ContentFilterConfig {
    fn default() -> Self {
        Self {
            enable_safety_filter: true,
            profanity_filter: ProfanityFilterConfig::default(),
            toxicity_filter: ToxicityFilterConfig::default(),
            custom_filters: Vec::new(),
            enforcement_level: FilterEnforcementLevel::Filter,
        }
    }
}

impl Default for QualityAssuranceConfig {
    fn default() -> Self {
        Self {
            enable_coherence_check: true,
            enable_grammar_check: false, // Disabled by default for performance
            enable_factual_check: false, // Disabled by default for performance
            quality_thresholds: QualityThresholds::default(),
            enable_realtime_monitoring: true,
        }
    }
}

impl Default for ProfanityFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strictness_level: 5,
            custom_wordlist: Vec::new(),
            action: FilterActionType::Replace,
        }
    }
}

impl Default for ToxicityFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.7,
            detection_model: ToxicityDetectionModel::Simple,
            action: FilterActionType::Replace,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_coherence_score: 0.6,
            min_grammar_score: 0.7,
            max_repetition_ratio: 0.3,
            min_diversity_score: 0.4,
        }
    }
}

/// Quality control processor for text generation
pub struct QualityController {
    config: QualityControlConfig,
    repetition_tracker: RepetitionTracker,
    frequency_tracker: FrequencyTracker,
    content_filters: Vec<ContentFilter>,
    quality_monitor: QualityMonitor,
}

/// Tracker for repetition detection and penalty calculation
pub struct RepetitionTracker {
    config: RepetitionPenaltyConfig,
    token_history: Vec<u32>,
    repetition_patterns: HashMap<Vec<u32>, RepetitionPattern>,
    penalty_cache: HashMap<u32, f32>,
}

/// Pattern of repetition for tracking
#[derive(Debug, Clone)]
pub struct RepetitionPattern {
    /// The repeated token sequence
    pub pattern: Vec<u32>,
    /// Number of times pattern has been seen
    pub count: usize,
    /// Positions where pattern was found
    pub positions: Vec<usize>,
    /// Last occurrence timestamp
    pub last_seen: Instant,
    /// Calculated penalty factor
    pub penalty_factor: f32,
}

/// Tracker for token frequency and diversity
pub struct FrequencyTracker {
    config: FrequencyPenaltyConfig,
    token_counts: HashMap<u32, TokenFrequency>,
    total_tokens: usize,
    vocabulary_size: usize,
}

/// Frequency information for individual tokens
#[derive(Debug, Clone)]
pub struct TokenFrequency {
    /// Absolute count in current sequence
    pub count: usize,
    /// Relative frequency (0.0-1.0)
    pub frequency: f32,
    /// Recency-weighted count
    pub weighted_count: f32,
    /// Positions where token appeared
    pub positions: Vec<usize>,
    /// Calculated penalty factor
    pub penalty_factor: f32,
}

/// Content filter for safety and quality
pub struct ContentFilter {
    config: CustomFilterConfig,
    filter_function: Box<dyn Fn(&str) -> FilterResult + Send + Sync>,
}

/// Result of content filtering
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether content passed the filter
    pub passed: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Detected issues
    pub issues: Vec<FilterIssue>,
    /// Suggested replacements
    pub suggestions: Vec<String>,
}

/// Individual filter issue
#[derive(Debug, Clone)]
pub struct FilterIssue {
    /// Issue type/category
    pub issue_type: String,
    /// Severity level
    pub severity: FilterSeverity,
    /// Description of the issue
    pub description: String,
    /// Position in text where issue was found
    pub position: Option<usize>,
}

/// Quality monitor for real-time quality assessment
pub struct QualityMonitor {
    config: QualityAssuranceConfig,
    quality_history: Vec<QualityMetrics>,
    current_metrics: QualityMetrics,
}

/// Quality metrics for generated text
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Coherence score (0.0-1.0)
    pub coherence_score: f32,
    /// Grammar score (0.0-1.0)
    pub grammar_score: f32,
    /// Repetition ratio (0.0-1.0)
    pub repetition_ratio: f32,
    /// Diversity score (0.0-1.0)
    pub diversity_score: f32,
    /// Content safety score (0.0-1.0)
    pub safety_score: f32,
    /// Overall quality score (0.0-1.0)
    pub overall_score: f32,
}

/// Result of quality control processing
#[derive(Debug, Clone)]
pub struct QualityControlResult {
    /// Original logits
    pub original_logits: Vec<f32>,
    /// Modified logits after quality control
    pub modified_logits: Vec<f32>,
    /// Applied penalties
    pub applied_penalties: Vec<AppliedPenalty>,
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Information about an applied penalty
#[derive(Debug, Clone)]
pub struct AppliedPenalty {
    /// Token ID that was penalized
    pub token_id: u32,
    /// Type of penalty applied
    pub penalty_type: PenaltyType,
    /// Penalty factor applied
    pub penalty_factor: f32,
    /// Reason for penalty
    pub reason: String,
}

/// Types of penalties that can be applied
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyType {
    /// Repetition penalty
    Repetition,
    /// Frequency penalty
    Frequency,
    /// Length penalty
    Length,
    /// Content filter penalty
    ContentFilter,
    /// Quality assurance penalty
    QualityAssurance,
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Quality metrics
    pub metrics: QualityMetrics,
    /// Whether quality thresholds are met
    pub meets_thresholds: bool,
    /// Quality issues detected
    pub issues: Vec<QualityIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual quality issue
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Issue category
    pub category: QualityIssueCategory,
    /// Severity level
    pub severity: FilterSeverity,
    /// Description
    pub description: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Categories of quality issues
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityIssueCategory {
    /// Repetitive content
    Repetition,
    /// Low diversity
    LowDiversity,
    /// Poor coherence
    Coherence,
    /// Grammar issues
    Grammar,
    /// Content safety issues
    Safety,
    /// Length issues
    Length,
}

impl QualityController {
    /// Create a new quality controller
    pub fn new(config: QualityControlConfig) -> Self {
        let repetition_tracker = RepetitionTracker::new(config.repetition_config.clone());
        let frequency_tracker = FrequencyTracker::new(config.frequency_config.clone());
        let content_filters = Self::create_content_filters(&config.content_filter_config);
        let quality_monitor = QualityMonitor::new(config.qa_config.clone());
        
        Self {
            config,
            repetition_tracker,
            frequency_tracker,
            content_filters,
            quality_monitor,
        }
    }
    
    /// Apply quality control to logits
    pub fn apply_quality_control(
        &mut self,
        logits: Vec<f32>,
        token_history: &[u32],
        current_length: usize,
    ) -> Result<QualityControlResult> {
        let start_time = Instant::now();
        let mut modified_logits = logits.clone();
        let mut applied_penalties = Vec::new();
        
        // Update trackers with current state
        self.repetition_tracker.update_history(token_history);
        self.frequency_tracker.update_counts(token_history);
        
        // Apply repetition penalty
        if self.config.repetition_config.penalty_factor != 1.0 {
            let repetition_penalties = self.repetition_tracker.calculate_penalties(&modified_logits)?;
            for (token_id, penalty) in repetition_penalties {
                if penalty != 1.0 {
                    if let Some(logit) = modified_logits.get_mut(token_id as usize) {
                        *logit /= penalty;
                        applied_penalties.push(AppliedPenalty {
                            token_id,
                            penalty_type: PenaltyType::Repetition,
                            penalty_factor: penalty,
                            reason: format!("Repetition penalty for token {}", token_id),
                        });
                    }
                }
            }
        }
        
        // Apply frequency penalty
        if self.config.frequency_config.penalty_factor != 0.0 {
            let frequency_penalties = self.frequency_tracker.calculate_penalties(&modified_logits)?;
            for (token_id, penalty) in frequency_penalties {
                if penalty != 1.0 {
                    if let Some(logit) = modified_logits.get_mut(token_id as usize) {
                        *logit -= penalty; // Frequency penalty is subtractive
                        applied_penalties.push(AppliedPenalty {
                            token_id,
                            penalty_type: PenaltyType::Frequency,
                            penalty_factor: penalty,
                            reason: format!("Frequency penalty for token {}", token_id),
                        });
                    }
                }
            }
        }
        
        // Apply length penalty
        if self.config.length_config.penalty_factor != 1.0 {
            let length_penalty = self.calculate_length_penalty(current_length)?;
            if length_penalty != 1.0 {
                for (token_id, logit) in modified_logits.iter_mut().enumerate() {
                    *logit *= length_penalty;
                    applied_penalties.push(AppliedPenalty {
                        token_id: token_id as u32,
                        penalty_type: PenaltyType::Length,
                        penalty_factor: length_penalty,
                        reason: format!("Length penalty at position {}", current_length),
                    });
                }
            }
        }
        
        // Assess quality
        let quality_assessment = self.assess_quality(token_history, &modified_logits)?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(QualityControlResult {
            original_logits: logits,
            modified_logits,
            applied_penalties,
            quality_assessment,
            processing_time_ms: processing_time,
        })
    }
    
    /// Calculate length penalty based on current sequence length
    fn calculate_length_penalty(&self, current_length: usize) -> Result<f32> {
        let config = &self.config.length_config;
        
        match config.normalization_strategy {
            LengthNormalizationStrategy::None => Ok(1.0),
            
            LengthNormalizationStrategy::Simple => {
                if let Some(target) = config.target_length {
                    let ratio = current_length as f32 / target as f32;
                    Ok(config.penalty_factor.powf(ratio - 1.0))
                } else {
                    Ok(1.0)
                }
            }
            
            LengthNormalizationStrategy::Wu2016 => {
                // Wu et al. (2016) length penalty: (5 + len)^α / (5 + 1)^α
                let alpha = config.penalty_factor;
                let numerator = (5.0f32 + current_length as f32).powf(alpha);
                let denominator = (5.0f32 + 1.0f32).powf(alpha);
                Ok(numerator / denominator)
            }
            
            LengthNormalizationStrategy::Custom => {
                // Placeholder for custom normalization
                Ok(config.penalty_factor)
            }
        }
    }
    
    /// Assess overall quality of generated sequence
    fn assess_quality(
        &mut self,
        token_history: &[u32],
        _logits: &[f32],
    ) -> Result<QualityAssessment> {
        let mut metrics = QualityMetrics::default();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // Calculate repetition ratio
        metrics.repetition_ratio = self.repetition_tracker.calculate_repetition_ratio();
        if metrics.repetition_ratio > self.config.qa_config.quality_thresholds.max_repetition_ratio {
            issues.push(QualityIssue {
                category: QualityIssueCategory::Repetition,
                severity: FilterSeverity::Medium,
                description: "High repetition detected in generated text".to_string(),
                suggested_fix: Some("Increase repetition penalty".to_string()),
            });
        }
        
        // Calculate diversity score
        metrics.diversity_score = self.frequency_tracker.calculate_diversity_score();
        if metrics.diversity_score < self.config.qa_config.quality_thresholds.min_diversity_score {
            issues.push(QualityIssue {
                category: QualityIssueCategory::LowDiversity,
                severity: FilterSeverity::Low,
                description: "Low vocabulary diversity in generated text".to_string(),
                suggested_fix: Some("Adjust frequency penalty settings".to_string()),
            });
        }
        
        // Placeholder coherence calculation
        metrics.coherence_score = 0.8; // Would be calculated by actual coherence model
        
        // Calculate overall score
        metrics.overall_score = (
            metrics.coherence_score * 0.4 +
            metrics.diversity_score * 0.3 +
            (1.0 - metrics.repetition_ratio) * 0.3
        );
        
        let meets_thresholds = issues.is_empty();
        
        Ok(QualityAssessment {
            metrics,
            meets_thresholds,
            issues,
            recommendations,
        })
    }
    
    /// Create content filters from configuration
    fn create_content_filters(config: &ContentFilterConfig) -> Vec<ContentFilter> {
        let mut filters = Vec::new();
        
        // Add profanity filter if enabled
        if config.profanity_filter.enabled {
            filters.push(ContentFilter {
                config: CustomFilterConfig {
                    name: "profanity".to_string(),
                    pattern: FilterPattern::WordList(config.profanity_filter.custom_wordlist.clone()),
                    severity: FilterSeverity::High,
                    action: config.profanity_filter.action,
                },
                filter_function: Box::new(|text| {
                    // Placeholder profanity detection
                    FilterResult {
                        passed: true, // Would implement actual detection
                        confidence: 0.9,
                        issues: Vec::new(),
                        suggestions: Vec::new(),
                    }
                }),
            });
        }
        
        filters
    }
}

impl RepetitionTracker {
    fn new(config: RepetitionPenaltyConfig) -> Self {
        Self {
            config,
            token_history: Vec::new(),
            repetition_patterns: HashMap::new(),
            penalty_cache: HashMap::new(),
        }
    }
    
    fn update_history(&mut self, new_tokens: &[u32]) {
        self.token_history.extend_from_slice(new_tokens);
        
        // Limit history size to window size
        if self.token_history.len() > self.config.window_size {
            let overflow = self.token_history.len() - self.config.window_size;
            self.token_history.drain(0..overflow);
        }
        
        // Detect new repetition patterns
        self.detect_patterns();
    }
    
    fn detect_patterns(&mut self) {
        // Simple pattern detection - look for repeated n-grams
        for n in self.config.min_repetition_length..=5 {
            if self.token_history.len() < n * 2 {
                continue;
            }
            
            for i in 0..=(self.token_history.len() - n * 2) {
                let pattern = self.token_history[i..i + n].to_vec();
                let next_pattern = self.token_history[i + n..i + n * 2].to_vec();
                
                if pattern == next_pattern {
                    let entry = self.repetition_patterns.entry(pattern.clone()).or_insert_with(|| {
                        RepetitionPattern {
                            pattern: pattern.clone(),
                            count: 0,
                            positions: Vec::new(),
                            last_seen: Instant::now(),
                            penalty_factor: 1.0,
                        }
                    });
                    
                    entry.count += 1;
                    entry.positions.push(i);
                    entry.last_seen = Instant::now();
                    let count = entry.count;
                    entry.penalty_factor = match self.config.scaling_strategy {
                        PenaltyScalingStrategy::Linear => {
                            self.config.penalty_factor * count as f32
                        }
                        PenaltyScalingStrategy::Exponential => {
                            self.config.penalty_factor * (count as f32).powf(1.5)
                        }
                        PenaltyScalingStrategy::Logarithmic => {
                            self.config.penalty_factor * (count as f32 + 1.0).ln()
                        }
                        PenaltyScalingStrategy::Adaptive => {
                            self.config.penalty_factor * (count as f32).sqrt()
                        }
                    };
                }
            }
        }
    }
    
    fn calculate_pattern_penalty(&self, pattern: &RepetitionPattern) -> f32 {
        match self.config.scaling_strategy {
            PenaltyScalingStrategy::Linear => {
                self.config.penalty_factor * pattern.count as f32
            }
            PenaltyScalingStrategy::Exponential => {
                self.config.penalty_factor.powf(pattern.count as f32)
            }
            PenaltyScalingStrategy::Logarithmic => {
                self.config.penalty_factor * (1.0 + (pattern.count as f32).ln())
            }
            PenaltyScalingStrategy::Adaptive => {
                // Adaptive penalty based on pattern characteristics
                let base_penalty = self.config.penalty_factor;
                let count_factor = (pattern.count as f32).sqrt();
                let recency_factor = if pattern.last_seen.elapsed().as_secs() < 60 {
                    1.2 // Recent patterns get higher penalty
                } else {
                    self.config.decay_factor
                };
                
                base_penalty * count_factor * recency_factor
            }
        }
    }
    
    fn calculate_penalties(&self, _logits: &[f32]) -> Result<HashMap<u32, f32>> {
        let mut penalties = HashMap::new();
        
        // Apply penalties based on detected patterns
        for pattern in self.repetition_patterns.values() {
            for &token in &pattern.pattern {
                penalties.insert(token, pattern.penalty_factor);
            }
        }
        
        Ok(penalties)
    }
    
    fn calculate_repetition_ratio(&self) -> f32 {
        if self.token_history.is_empty() {
            return 0.0;
        }
        
        let unique_tokens: std::collections::HashSet<_> = self.token_history.iter().collect();
        1.0 - (unique_tokens.len() as f32 / self.token_history.len() as f32)
    }
}

impl FrequencyTracker {
    fn new(config: FrequencyPenaltyConfig) -> Self {
        Self {
            config,
            token_counts: HashMap::new(),
            total_tokens: 0,
            vocabulary_size: 50000, // Default vocabulary size
        }
    }
    
    fn update_counts(&mut self, tokens: &[u32]) {
        for (position, &token) in tokens.iter().enumerate() {
            let entry = self.token_counts.entry(token).or_insert_with(|| {
                TokenFrequency {
                    count: 0,
                    frequency: 0.0,
                    weighted_count: 0.0,
                    positions: Vec::new(),
                    penalty_factor: 0.0,
                }
            });
            
            entry.count += 1;
            entry.positions.push(self.total_tokens + position);
            
            // Calculate recency weight
            let recency_weight = match self.config.counting_strategy {
                TokenCountingStrategy::RecencyWeighted => {
                    1.0 + (position as f32 / tokens.len() as f32) * 0.5
                }
                _ => 1.0,
            };
            
            entry.weighted_count += recency_weight;
        }
        
        self.total_tokens += tokens.len();
        
        // Update frequencies and penalties
        self.update_frequencies();
    }
    
    fn update_frequencies(&mut self) {
        for frequency_info in self.token_counts.values_mut() {
            frequency_info.frequency = frequency_info.count as f32 / self.total_tokens as f32;
            
            frequency_info.penalty_factor = match self.config.frequency_method {
                FrequencyCalculationMethod::Simple => {
                    self.config.penalty_factor * frequency_info.frequency
                }
                FrequencyCalculationMethod::Normalized => {
                    self.config.penalty_factor * (frequency_info.frequency * self.vocabulary_size as f32).sqrt()
                }
                FrequencyCalculationMethod::Inverse => {
                    self.config.penalty_factor / (1.0 + frequency_info.frequency)
                }
                FrequencyCalculationMethod::TFIDF => {
                    // Simplified TF-IDF calculation
                    let tf = frequency_info.frequency;
                    let idf = (self.vocabulary_size as f32 / (1.0 + frequency_info.count as f32)).ln();
                    self.config.penalty_factor * tf * idf
                }
            };
            
            // Apply presence penalty
            frequency_info.penalty_factor += self.config.presence_penalty;
        }
    }
    
    fn calculate_penalties(&self, _logits: &[f32]) -> Result<HashMap<u32, f32>> {
        let mut penalties = HashMap::new();
        
        for (&token_id, frequency_info) in &self.token_counts {
            penalties.insert(token_id, frequency_info.penalty_factor);
        }
        
        Ok(penalties)
    }
    
    fn calculate_diversity_score(&self) -> f32 {
        if self.total_tokens == 0 {
            return 1.0;
        }
        
        let unique_tokens = self.token_counts.len();
        unique_tokens as f32 / self.total_tokens as f32
    }
}

impl QualityMonitor {
    fn new(config: QualityAssuranceConfig) -> Self {
        Self {
            config,
            quality_history: Vec::new(),
            current_metrics: QualityMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_control_config_default() {
        let config = QualityControlConfig::default();
        assert_eq!(config.repetition_config.penalty_factor, 1.1);
        assert_eq!(config.length_config.penalty_factor, 1.0);
        assert_eq!(config.frequency_config.penalty_factor, 0.1);
    }
    
    #[test]
    fn test_repetition_tracker() {
        let config = RepetitionPenaltyConfig::default();
        let mut tracker = RepetitionTracker::new(config);
        
        // Add some tokens with repetition
        let tokens = vec![1, 2, 3, 1, 2, 3, 4, 5];
        tracker.update_history(&tokens);
        
        assert!(tracker.repetition_patterns.len() > 0);
        let ratio = tracker.calculate_repetition_ratio();
        assert!(ratio > 0.0);
        assert!(ratio <= 1.0);
    }
    
    #[test]
    fn test_frequency_tracker() {
        let config = FrequencyPenaltyConfig::default();
        let mut tracker = FrequencyTracker::new(config);
        
        let tokens = vec![1, 1, 2, 3, 1, 4, 5];
        tracker.update_counts(&tokens);
        
        assert!(tracker.token_counts.contains_key(&1));
        assert_eq!(tracker.token_counts[&1].count, 3);
        assert!(tracker.token_counts[&1].frequency > 0.0);
        
        let diversity = tracker.calculate_diversity_score();
        assert!(diversity > 0.0);
        assert!(diversity <= 1.0);
    }
    
    #[test]
    fn test_quality_controller_creation() {
        let config = QualityControlConfig::default();
        let controller = QualityController::new(config);
        
        // Test basic functionality
        assert_eq!(controller.config.repetition_config.penalty_factor, 1.1);
    }
    
    #[tokio::test]
    async fn test_quality_control_application() -> Result<()> {
        let config = QualityControlConfig::default();
        let mut controller = QualityController::new(config);
        
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let token_history = vec![1, 2, 3, 2, 1];
        let current_length = 5;
        
        let result = controller.apply_quality_control(logits.clone(), &token_history, current_length)?;
        
        assert_eq!(result.original_logits.len(), logits.len());
        assert_eq!(result.modified_logits.len(), logits.len());
        assert!(result.processing_time_ms > 0);
        
        Ok(())
    }
}