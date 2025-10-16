//! Long Context Extension Features for BitNet Models
//!
//! This module implements advanced context management for sequences longer than 4096 tokens:
//! - Sliding window attention for efficient long context processing
//! - Context segmentation and reconstruction
//! - Memory-efficient context rotation and caching
//! - Extended conversation management with context preservation

use crate::{Result, InferenceError};
use crate::cache::enhanced_kv_cache::{EnhancedKVCache, EnhancedKVCacheConfig, SlidingWindowState};
use bitnet_core::{Tensor, Device, DType};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Configuration for long context extension
#[derive(Debug, Clone)]
pub struct ContextExtensionConfig {
    /// Maximum context length supported
    pub max_context_length: usize,
    /// Base model context window size  
    pub base_context_window: usize,
    /// Sliding window strategy
    pub sliding_strategy: SlidingStrategy,
    /// Context preservation settings
    pub preservation_config: ContextPreservationConfig,
    /// Attention optimization settings
    pub attention_optimization: AttentionOptimizationConfig,
    /// Memory management for long contexts
    pub memory_config: LongContextMemoryConfig,
}

/// Strategies for sliding window attention
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlidingStrategy {
    /// Fixed sliding window - keep most recent N tokens
    FixedWindow,
    /// Adaptive sliding - intelligently select important tokens
    Adaptive,
    /// Hierarchical attention - different attention patterns for different ranges
    Hierarchical,
    /// Segment-based - divide context into segments with different handling
    SegmentBased,
}

/// Configuration for context preservation
#[derive(Debug, Clone)]
pub struct ContextPreservationConfig {
    /// Always preserve system/instruction tokens
    pub preserve_system_tokens: bool,
    /// Preserve conversation structure markers
    pub preserve_conversation_markers: bool,
    /// Number of tokens to always keep from beginning
    pub preserve_prefix_tokens: usize,
    /// Important token detection strategy
    pub importance_strategy: ImportanceStrategy,
    /// Compression ratio for older context (0.0-1.0)
    pub compression_ratio: f64,
}

/// Strategies for determining token importance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceStrategy {
    /// Recent tokens are most important
    Recency,
    /// Attention weights determine importance
    AttentionBased,
    /// Semantic similarity to current context
    Semantic,
    /// Combined multiple strategies
    Hybrid,
}

/// Configuration for attention optimization in long contexts
#[derive(Debug, Clone)]
pub struct AttentionOptimizationConfig {
    /// Enable sparse attention patterns
    pub enable_sparse_attention: bool,
    /// Local attention window size
    pub local_attention_window: usize,
    /// Global attention stride
    pub global_attention_stride: usize,
    /// Enable flash attention optimization
    pub enable_flash_attention: bool,
    /// Memory-efficient attention computation
    pub memory_efficient_attention: bool,
}

/// Configuration for long context memory management
#[derive(Debug, Clone)]
pub struct LongContextMemoryConfig {
    /// Maximum memory usage for context storage (bytes)
    pub max_context_memory_bytes: usize,
    /// Enable context compression
    pub enable_compression: bool,
    /// Context cache eviction policy
    pub eviction_policy: ContextEvictionPolicy,
    /// Background context processing
    pub background_processing: bool,
}

/// Eviction policies for context cache management
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextEvictionPolicy {
    /// Least recently used context segments
    LRU,
    /// Least important segments based on attention
    LeastImportant,
    /// Oldest segments first
    FIFO,
    /// Adaptive based on usage patterns
    Adaptive,
}

impl Default for ContextExtensionConfig {
    fn default() -> Self {
        Self {
            max_context_length: 32768, // Support up to 32K tokens
            base_context_window: 4096,
            sliding_strategy: SlidingStrategy::Adaptive,
            preservation_config: ContextPreservationConfig::default(),
            attention_optimization: AttentionOptimizationConfig::default(),
            memory_config: LongContextMemoryConfig::default(),
        }
    }
}

impl Default for ContextPreservationConfig {
    fn default() -> Self {
        Self {
            preserve_system_tokens: true,
            preserve_conversation_markers: true,
            preserve_prefix_tokens: 128,
            importance_strategy: ImportanceStrategy::Hybrid,
            compression_ratio: 0.7, // Compress to 70% of original
        }
    }
}

impl Default for AttentionOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_sparse_attention: true,
            local_attention_window: 512,
            global_attention_stride: 64,
            enable_flash_attention: true,
            memory_efficient_attention: true,
        }
    }
}

impl Default for LongContextMemoryConfig {
    fn default() -> Self {
        Self {
            max_context_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            enable_compression: true,
            eviction_policy: ContextEvictionPolicy::Adaptive,
            background_processing: true,
        }
    }
}

/// Extended context manager for long sequence processing
pub struct LongContextManager {
    config: ContextExtensionConfig,
    context_segments: Arc<Mutex<ContextSegmentManager>>,
    attention_optimizer: Arc<AttentionOptimizer>,
    memory_manager: Arc<ContextMemoryManager>,
    kv_cache: Arc<EnhancedKVCache>,
}

/// Manager for context segments in long sequences
pub struct ContextSegmentManager {
    segments: VecDeque<ContextSegment>,
    active_window: Vec<usize>, // Token indices in current attention window
    segment_index: HashMap<String, usize>, // Segment ID to index mapping
    total_tokens: usize,
}

/// Individual context segment
#[derive(Debug, Clone)]
pub struct ContextSegment {
    /// Unique segment identifier
    pub segment_id: String,
    /// Tokens in this segment
    pub tokens: Vec<u32>,
    /// Segment metadata
    pub metadata: SegmentMetadata,
    /// Importance score (0.0-1.0)
    pub importance_score: f64,
    /// Compression state
    pub compression_state: CompressionState,
}

/// Metadata for context segments
#[derive(Debug, Clone)]
pub struct SegmentMetadata {
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access frequency
    pub access_count: u64,
    /// Segment type (system, user, assistant, etc.)
    pub segment_type: SegmentType,
    /// Position in original sequence
    pub original_position: usize,
    /// Attention statistics
    pub attention_stats: AttentionStats,
}

/// Types of context segments
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentType {
    /// System instructions
    System,
    /// User input
    User,
    /// Assistant response
    Assistant,
    /// Conversation marker
    Marker,
    /// General content
    Content,
}

/// Attention statistics for segments
#[derive(Debug, Clone, Default)]
pub struct AttentionStats {
    /// Average attention weight received
    pub avg_attention_weight: f64,
    /// Maximum attention weight received
    pub max_attention_weight: f64,
    /// Number of times attended to
    pub attention_count: u64,
    /// Attention distribution across heads
    pub head_attention_distribution: Vec<f64>,
}

/// Compression state for context segments
#[derive(Debug, Clone)]
pub struct CompressionState {
    /// Whether segment is compressed
    pub is_compressed: bool,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Original size in tokens
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression method used
    pub compression_method: CompressionMethod,
}

/// Methods for compressing context segments
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Token-level compression (remove less important tokens)
    TokenLevel,
    /// Semantic compression (summarization)
    Semantic,
    /// Attention-based compression (keep high-attention tokens)
    AttentionBased,
    /// Hybrid compression method
    Hybrid,
}

/// Attention optimization for long contexts
pub struct AttentionOptimizer {
    config: AttentionOptimizationConfig,
    sparse_patterns: HashMap<String, SparseAttentionPattern>,
    attention_cache: HashMap<String, AttentionResult>,
}

/// Sparse attention patterns for efficient computation
#[derive(Debug, Clone)]
pub struct SparseAttentionPattern {
    /// Local attention indices
    pub local_indices: Vec<usize>,
    /// Global attention indices
    pub global_indices: Vec<usize>,
    /// Pattern type
    pub pattern_type: AttentionPatternType,
    /// Efficiency metrics
    pub efficiency_metrics: AttentionEfficiencyMetrics,
}

/// Types of attention patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionPatternType {
    /// Full attention (baseline)
    Full,
    /// Local window + global sparse
    LocalGlobal,
    /// Sliding window with stride
    SlidingWindow,
    /// Hierarchical attention
    Hierarchical,
    /// Block-sparse attention
    BlockSparse,
}

/// Efficiency metrics for attention patterns
#[derive(Debug, Clone, Default)]
pub struct AttentionEfficiencyMetrics {
    /// Computation reduction factor
    pub computation_reduction: f64,
    /// Memory reduction factor
    pub memory_reduction: f64,
    /// Quality preservation score (0.0-1.0)
    pub quality_preservation: f64,
    /// Latency improvement factor
    pub latency_improvement: f64,
}

/// Result of attention computation
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Attention weights
    pub attention_weights: Vec<f64>,
    /// Attended values
    pub attended_values: Tensor,
    /// Computation metrics
    pub metrics: AttentionComputationMetrics,
}

/// Metrics for attention computation
#[derive(Debug, Clone, Default)]
pub struct AttentionComputationMetrics {
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of tokens attended
    pub tokens_attended: usize,
    /// Sparsity ratio achieved
    pub sparsity_ratio: f64,
}

/// Memory manager for long context processing
pub struct ContextMemoryManager {
    config: LongContextMemoryConfig,
    memory_usage: std::sync::atomic::AtomicUsize,
    cached_segments: Mutex<HashMap<String, CachedSegment>>,
    eviction_tracker: Mutex<EvictionTracker>,
}

/// Cached context segment
#[derive(Debug, Clone)]
pub struct CachedSegment {
    /// Segment data
    pub segment: ContextSegment,
    /// Cache timestamp
    pub cached_at: Instant,
    /// Access pattern
    pub access_pattern: Vec<Instant>,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
}

/// Tracker for cache eviction decisions
#[derive(Debug, Default)]
pub struct EvictionTracker {
    /// Eviction events
    pub eviction_events: Vec<EvictionEvent>,
    /// Eviction statistics
    pub eviction_stats: EvictionStats,
}

/// Individual eviction event
#[derive(Debug, Clone)]
pub struct EvictionEvent {
    pub timestamp: Instant,
    pub segment_id: String,
    pub reason: EvictionReason,
    pub memory_freed: usize,
}

/// Reasons for cache eviction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionReason {
    /// Memory pressure
    MemoryPressure,
    /// Least recently used
    LRU,
    /// Low importance score
    LowImportance,
    /// Manual eviction
    Manual,
    /// Segment expiration
    Expired,
}

/// Statistics for cache eviction
#[derive(Debug, Clone, Default)]
pub struct EvictionStats {
    /// Total evictions performed
    pub total_evictions: u64,
    /// Memory freed by evictions (bytes)
    pub memory_freed_bytes: usize,
    /// Average eviction frequency
    pub avg_eviction_frequency: f64,
    /// Eviction effectiveness score
    pub eviction_effectiveness: f64,
}

impl LongContextManager {
    /// Create a new long context manager
    pub fn new(
        config: ContextExtensionConfig, 
        device: Device
    ) -> Result<Self> {
        let kv_cache_config = EnhancedKVCacheConfig {
            base_config: crate::cache::kv_cache::KVCacheConfig {
                max_seq_len: config.max_context_length,
                device: device.clone(),
                memory_optimized: true,
                ..Default::default()
            },
            enable_sliding_window: true,
            sliding_window_size: config.base_context_window,
            max_total_memory_bytes: config.memory_config.max_context_memory_bytes,
            ..Default::default()
        };
        
        let kv_cache = Arc::new(EnhancedKVCache::new(kv_cache_config)?);
        let context_segments = Arc::new(Mutex::new(ContextSegmentManager::new()));
        let attention_optimizer = Arc::new(AttentionOptimizer::new(config.attention_optimization.clone()));
        let memory_manager = Arc::new(ContextMemoryManager::new(config.memory_config.clone()));
        
        Ok(Self {
            config,
            context_segments,
            attention_optimizer,
            memory_manager,
            kv_cache,
        })
    }
    
    /// Process a long context sequence
    pub async fn process_long_context(
        &self,
        tokens: Vec<u32>,
        sequence_id: String,
    ) -> Result<ProcessedContext> {
        let start_time = Instant::now();
        
        // Check if sequence exceeds base window
        if tokens.len() <= self.config.base_context_window {
            let tokens_len = tokens.len();
            // Use standard processing for short sequences
            return Ok(ProcessedContext {
                processed_tokens: tokens,
                attention_pattern: AttentionPatternType::Full,
                segments: Vec::new(),
                processing_metrics: ProcessingMetrics {
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    memory_usage_bytes: tokens_len * 4, // Approximate
                    compression_ratio: 1.0,
                    segments_created: 0,
                },
            });
        }
        
        // Segment the long context
        let segments = self.segment_context(&tokens, &sequence_id).await?;
        
        // Apply sliding window strategy
        let active_segments = self.apply_sliding_strategy(&segments).await?;
        
        // Optimize attention pattern
        let attention_pattern = self.optimize_attention_pattern(&active_segments).await?;
        
        // Process with optimized attention
        let processed_tokens = self.process_with_attention(&active_segments, attention_pattern).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let compression_ratio = self.calculate_compression_ratio(&segments, tokens.len());
        let segments_count = segments.len();
        
        Ok(ProcessedContext {
            processed_tokens,
            attention_pattern,
            segments,
            processing_metrics: ProcessingMetrics {
                processing_time_ms: processing_time,
                memory_usage_bytes: self.estimate_memory_usage(&active_segments),
                compression_ratio,
                segments_created: segments_count,
            },
        })
    }
    
    /// Segment long context into manageable chunks
    async fn segment_context(
        &self,
        tokens: &[u32],
        sequence_id: &str,
    ) -> Result<Vec<ContextSegment>> {
        let mut segments = Vec::new();
        let segment_size = self.config.base_context_window / 4; // 1K tokens per segment
        
        // Always preserve system tokens if configured
        let mut start_index = 0;
        if self.config.preservation_config.preserve_system_tokens {
            let prefix_tokens = tokens.iter()
                .take(self.config.preservation_config.preserve_prefix_tokens)
                .cloned()
                .collect::<Vec<_>>();
            
            if !prefix_tokens.is_empty() {
                let prefix_tokens_len = prefix_tokens.len();
                let segment = ContextSegment {
                    segment_id: format!("{}_system", sequence_id),
                    tokens: prefix_tokens,
                    metadata: SegmentMetadata {
                        created_at: Instant::now(),
                        last_accessed: Instant::now(),
                        access_count: 0,
                        segment_type: SegmentType::System,
                        original_position: 0,
                        attention_stats: AttentionStats::default(),
                    },
                    importance_score: 1.0, // Maximum importance for system tokens
                    compression_state: CompressionState {
                        is_compressed: false,
                        compression_ratio: 1.0,
                        original_size: prefix_tokens_len,
                        compressed_size: prefix_tokens_len * 4,
                        compression_method: CompressionMethod::None,
                    },
                };
                
                segments.push(segment);
                start_index = self.config.preservation_config.preserve_prefix_tokens;
            }
        }
        
        // Segment remaining tokens
        let mut current_pos = start_index;
        let mut segment_index = 0;
        
        while current_pos < tokens.len() {
            let end_pos = (current_pos + segment_size).min(tokens.len());
            let segment_tokens = tokens[current_pos..end_pos].to_vec();
            
            let segment = ContextSegment {
                segment_id: format!("{}_{}", sequence_id, segment_index),
                tokens: segment_tokens,
                metadata: SegmentMetadata {
                    created_at: Instant::now(),
                    last_accessed: Instant::now(),
                    access_count: 0,
                    segment_type: self.detect_segment_type(&tokens[current_pos..end_pos]),
                    original_position: current_pos,
                    attention_stats: AttentionStats::default(),
                },
                importance_score: self.calculate_initial_importance(current_pos, tokens.len()),
                compression_state: CompressionState {
                    is_compressed: false,
                    compression_ratio: 1.0,
                    original_size: end_pos - current_pos,
                    compressed_size: (end_pos - current_pos) * 4,
                    compression_method: CompressionMethod::None,
                },
            };
            
            segments.push(segment);
            current_pos = end_pos;
            segment_index += 1;
        }
        
        Ok(segments)
    }
    
    /// Apply sliding window strategy to select active segments
    async fn apply_sliding_strategy(
        &self,
        segments: &[ContextSegment],
    ) -> Result<Vec<usize>> {
        match self.config.sliding_strategy {
            SlidingStrategy::FixedWindow => {
                // Keep most recent segments that fit in window
                let total_tokens: usize = segments.iter().map(|s| s.tokens.len()).sum();
                let mut active_segments = Vec::new();
                let mut current_tokens = 0;
                
                for (i, segment) in segments.iter().enumerate().rev() {
                    if current_tokens + segment.tokens.len() <= self.config.base_context_window {
                        active_segments.push(i);
                        current_tokens += segment.tokens.len();
                    } else {
                        break;
                    }
                }
                
                active_segments.reverse();
                Ok(active_segments)
            }
            
            SlidingStrategy::Adaptive => {
                // Select segments based on importance scores
                let mut segment_scores: Vec<(usize, f64)> = segments
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (i, s.importance_score))
                    .collect();
                
                segment_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                let mut active_segments = Vec::new();
                let mut current_tokens = 0;
                
                for (i, _score) in segment_scores {
                    if current_tokens + segments[i].tokens.len() <= self.config.base_context_window {
                        active_segments.push(i);
                        current_tokens += segments[i].tokens.len();
                    }
                }
                
                active_segments.sort();
                Ok(active_segments)
            }
            
            SlidingStrategy::Hierarchical | SlidingStrategy::SegmentBased => {
                // More complex strategies - placeholder implementation
                self.apply_sliding_strategy_simple(segments).await
            }
        }
    }
    
    /// Simple sliding strategy implementation (fallback)
    async fn apply_sliding_strategy_simple(
        &self,
        segments: &[ContextSegment],
    ) -> Result<Vec<usize>> {
        // Keep most recent segments that fit
        let mut active = Vec::new();
        let mut tokens = 0;
        
        for (i, segment) in segments.iter().enumerate().rev() {
            if tokens + segment.tokens.len() <= self.config.base_context_window {
                active.push(i);
                tokens += segment.tokens.len();
            } else {
                break;
            }
        }
        
        active.reverse();
        Ok(active)
    }
    
    /// Optimize attention pattern for selected segments
    async fn optimize_attention_pattern(
        &self,
        _active_segments: &[usize],
    ) -> Result<AttentionPatternType> {
        // Return appropriate pattern based on configuration
        if self.config.attention_optimization.enable_sparse_attention {
            Ok(AttentionPatternType::LocalGlobal)
        } else {
            Ok(AttentionPatternType::Full)
        }
    }
    
    /// Process tokens with optimized attention
    async fn process_with_attention(
        &self,
        active_segments: &[usize],
        _attention_pattern: AttentionPatternType,
    ) -> Result<Vec<u32>> {
        // Placeholder implementation - would involve actual attention computation
        // For now, just return the tokens from active segments
        let processed_tokens = Vec::new();
        
        for &segment_idx in active_segments {
            // In real implementation, would apply attention optimization here
            // processed_tokens.extend(&segments[segment_idx].tokens);
        }
        
        Ok(processed_tokens)
    }
    
    /// Detect segment type based on token patterns
    fn detect_segment_type(&self, _tokens: &[u32]) -> SegmentType {
        // Placeholder - would analyze tokens to determine type
        SegmentType::Content
    }
    
    /// Calculate initial importance score for a segment
    fn calculate_initial_importance(&self, position: usize, total_length: usize) -> f64 {
        // Recent tokens are more important
        let recency_score = 1.0 - (position as f64 / total_length as f64);
        
        // Preserve some importance for early tokens (system instructions)
        if position < self.config.preservation_config.preserve_prefix_tokens {
            (recency_score + 0.5).min(1.0)
        } else {
            recency_score
        }
    }
    
    /// Estimate memory usage for active segments
    fn estimate_memory_usage(&self, active_segments: &[usize]) -> usize {
        active_segments.len() * 1024 // Placeholder estimation
    }
    
    /// Calculate compression ratio achieved
    fn calculate_compression_ratio(&self, segments: &[ContextSegment], original_length: usize) -> f64 {
        let processed_tokens: usize = segments.iter().map(|s| s.tokens.len()).sum();
        processed_tokens as f64 / original_length as f64
    }
}

/// Result of long context processing
#[derive(Debug, Clone)]
pub struct ProcessedContext {
    /// Processed tokens ready for inference
    pub processed_tokens: Vec<u32>,
    /// Attention pattern used
    pub attention_pattern: AttentionPatternType,
    /// Context segments created
    pub segments: Vec<ContextSegment>,
    /// Processing performance metrics
    pub processing_metrics: ProcessingMetrics,
}

/// Metrics for context processing performance
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Compression ratio achieved (0.0-1.0)
    pub compression_ratio: f64,
    /// Number of segments created
    pub segments_created: usize,
}

impl ContextSegmentManager {
    fn new() -> Self {
        Self {
            segments: VecDeque::new(),
            active_window: Vec::new(),
            segment_index: HashMap::new(),
            total_tokens: 0,
        }
    }
}

impl AttentionOptimizer {
    fn new(config: AttentionOptimizationConfig) -> Self {
        Self {
            config,
            sparse_patterns: HashMap::new(),
            attention_cache: HashMap::new(),
        }
    }
}

impl ContextMemoryManager {
    fn new(config: LongContextMemoryConfig) -> Self {
        Self {
            config,
            memory_usage: std::sync::atomic::AtomicUsize::new(0),
            cached_segments: Mutex::new(HashMap::new()),
            eviction_tracker: Mutex::new(EvictionTracker::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_extension_config_default() {
        let config = ContextExtensionConfig::default();
        assert_eq!(config.max_context_length, 32768);
        assert_eq!(config.base_context_window, 4096);
        assert_eq!(config.sliding_strategy, SlidingStrategy::Adaptive);
    }
    
    #[tokio::test]
    async fn test_long_context_manager_creation() -> Result<()> {
        let config = ContextExtensionConfig::default();
        let manager = LongContextManager::new(config, Device::Cpu)?;
        
        // Test processing short context (should use standard path)
        let short_tokens = vec![1, 2, 3, 4, 5]; // 5 tokens
        let result = manager.process_long_context(short_tokens.clone(), "test_short".to_string()).await?;
        
        assert_eq!(result.processed_tokens, short_tokens);
        assert_eq!(result.attention_pattern, AttentionPatternType::Full);
        assert_eq!(result.segments.len(), 0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_context_segmentation() -> Result<()> {
        let config = ContextExtensionConfig {
            base_context_window: 16, // Small window for testing
            ..Default::default()
        };
        let manager = LongContextManager::new(config, Device::Cpu)?;
        
        // Create long context that will be segmented
        let long_tokens: Vec<u32> = (1..=100).collect(); // 100 tokens
        let result = manager.process_long_context(long_tokens, "test_long".to_string()).await?;
        
        assert!(result.segments.len() > 0);
        assert!(result.processing_metrics.segments_created > 0);
        assert!(result.processing_metrics.compression_ratio > 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_segment_metadata() {
        let metadata = SegmentMetadata {
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 5,
            segment_type: SegmentType::System,
            original_position: 0,
            attention_stats: AttentionStats::default(),
        };
        
        assert_eq!(metadata.segment_type, SegmentType::System);
        assert_eq!(metadata.access_count, 5);
        assert_eq!(metadata.original_position, 0);
    }
    
    #[test]
    fn test_compression_state() {
        let compression = CompressionState {
            is_compressed: true,
            compression_ratio: 0.7,
            original_size: 1000,
            compressed_size: 700,
            compression_method: CompressionMethod::Semantic,
        };
        
        assert!(compression.is_compressed);
        assert_eq!(compression.compression_ratio, 0.7);
        assert_eq!(compression.compression_method, CompressionMethod::Semantic);
    }
}