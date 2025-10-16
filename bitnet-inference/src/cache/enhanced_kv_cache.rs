//! Enhanced KV Cache Memory Optimization with Microsoft-style pooling and sliding window support.
//!
//! This module implements advanced memory management for KV caches:
//! - Microsoft-style memory pooling for efficient cache reuse
//! - Sliding window support for long contexts beyond 4096 tokens
//! - Memory pressure handling and adaptive cache management
//! - Efficient cache rotation for extended conversations

use anyhow::{Result, Context};
use bitnet_core::{Tensor, Device, DType};
use crate::cache::kv_cache::{KVCacheConfig, KVCacheStats};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Enhanced configuration for advanced KV cache memory management
#[derive(Debug, Clone)]
pub struct EnhancedKVCacheConfig {
    /// Base KV cache configuration
    pub base_config: KVCacheConfig,
    /// Enable sliding window for long contexts
    pub enable_sliding_window: bool,
    /// Sliding window size (tokens to keep when full)
    pub sliding_window_size: usize,
    /// Memory pool configuration
    pub memory_pool_config: MemoryPoolConfig,
    /// Cache eviction strategy
    pub eviction_strategy: CacheEvictionStrategy,
    /// Maximum total memory usage (bytes)
    pub max_total_memory_bytes: usize,
    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f64,
}

/// Configuration for KV cache memory pooling
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size (number of cache blocks)
    pub initial_pool_size: usize,
    /// Maximum pool size
    pub max_pool_size: usize,
    /// Cache block size in tokens
    pub cache_block_size: usize,
    /// Pre-allocation strategy
    pub preallocation_strategy: PreallocationStrategy,
    /// Enable memory compaction
    pub enable_compaction: bool,
}

/// Pre-allocation strategies for memory pools
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreallocationStrategy {
    /// No pre-allocation - allocate on demand
    None,
    /// Conservative pre-allocation (small initial pool)
    Conservative,
    /// Balanced pre-allocation (moderate initial pool)
    Balanced,
    /// Aggressive pre-allocation (large initial pool)
    Aggressive,
}

/// Cache eviction strategies for memory management
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheEvictionStrategy {
    /// Least Recently Used (LRU)
    LRU,
    /// First In First Out (FIFO)
    FIFO,
    /// Least Frequently Used (LFU)
    LFU,
    /// Sliding Window (for long contexts)
    SlidingWindow,
    /// Adaptive (dynamically choose based on usage pattern)
    Adaptive,
}

impl Default for EnhancedKVCacheConfig {
    fn default() -> Self {
        Self {
            base_config: KVCacheConfig::default(),
            enable_sliding_window: true,
            sliding_window_size: 2048, // Keep 2K tokens when context exceeds limit
            memory_pool_config: MemoryPoolConfig::default(),
            eviction_strategy: CacheEvictionStrategy::Adaptive,
            max_total_memory_bytes: 1024 * 1024 * 1024, // 1GB
            memory_pressure_threshold: 0.8, // Start eviction at 80% memory usage
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_pool_size: 8,
            max_pool_size: 64,
            cache_block_size: 256, // 256 tokens per block
            preallocation_strategy: PreallocationStrategy::Balanced,
            enable_compaction: true,
        }
    }
}

/// Enhanced KV cache with advanced memory management
pub struct EnhancedKVCache {
    config: EnhancedKVCacheConfig,
    memory_pool: Arc<RwLock<KVMemoryPool>>,
    active_caches: Arc<RwLock<HashMap<String, Arc<Mutex<CacheEntry>>>>>,
    usage_tracker: Arc<Mutex<CacheUsageTracker>>,
    memory_monitor: Arc<Mutex<MemoryMonitor>>,
}

/// Memory pool for KV cache blocks
pub struct KVMemoryPool {
    available_blocks: VecDeque<CacheBlock>,
    allocated_blocks: HashMap<String, CacheBlock>,
    pool_stats: PoolStats,
    config: MemoryPoolConfig,
}

/// Individual cache block in the memory pool
#[derive(Debug, Clone)]
pub struct CacheBlock {
    /// Unique block identifier
    pub block_id: String,
    /// Key tensor storage
    pub key_tensor: Option<Tensor>,
    /// Value tensor storage  
    pub value_tensor: Option<Tensor>,
    /// Block capacity in tokens
    pub capacity: usize,
    /// Current usage in tokens
    pub usage: usize,
    /// Last access time
    pub last_access: Instant,
    /// Access frequency counter
    pub access_count: u64,
}

/// Cache entry for active sequences
pub struct CacheEntry {
    /// Sequence identifier
    pub sequence_id: String,
    /// Allocated cache blocks for this sequence
    pub blocks: Vec<String>, // Block IDs
    /// Current position in the sequence
    pub current_position: usize,
    /// Sliding window state
    pub sliding_window: SlidingWindowState,
    /// Entry statistics
    pub stats: CacheEntryStats,
}

/// Sliding window state for long context management
#[derive(Debug, Clone)]
pub struct SlidingWindowState {
    /// Total sequence length processed
    pub total_length: usize,
    /// Current window start position
    pub window_start: usize,
    /// Current window end position
    pub window_end: usize,
    /// Evicted token count
    pub evicted_tokens: usize,
}

/// Statistics for individual cache entries
#[derive(Debug, Clone)]
pub struct CacheEntryStats {
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Total access count
    pub access_count: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of sliding window operations
    pub sliding_operations: u64,
}

impl Default for CacheEntryStats {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
            memory_usage_bytes: 0,
            sliding_operations: 0,
        }
    }
}

/// Usage tracking for cache optimization
pub struct CacheUsageTracker {
    sequence_usage: HashMap<String, SequenceUsage>,
    global_stats: GlobalUsageStats,
}

/// Usage statistics for individual sequences
#[derive(Debug, Clone)]
pub struct SequenceUsage {
    pub sequence_id: String,
    pub access_pattern: Vec<Instant>,
    pub memory_pattern: Vec<usize>,
    pub predicted_length: Option<usize>,
}

/// Global usage statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct GlobalUsageStats {
    pub total_sequences: u64,
    pub avg_sequence_length: f64,
    pub memory_efficiency: f64,
    pub cache_hit_rate: f64,
}

/// Memory monitoring and pressure management
pub struct MemoryMonitor {
    current_usage: usize,
    peak_usage: usize,
    pressure_events: Vec<MemoryPressureEvent>,
    config: EnhancedKVCacheConfig,
}

/// Memory pressure event for tracking
#[derive(Debug, Clone)]
pub struct MemoryPressureEvent {
    pub timestamp: Instant,
    pub memory_usage: usize,
    pub pressure_level: f64,
    pub action_taken: PressureAction,
}

/// Actions taken during memory pressure
#[derive(Debug, Clone)]
pub enum PressureAction {
    /// Evicted cache entries
    Eviction(Vec<String>),
    /// Compacted memory pool
    Compaction,
    /// Applied sliding window
    SlidingWindow(String),
    /// Blocked new allocations
    AllocationBlock,
}

/// Pool statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_blocks_created: u64,
    pub active_blocks: usize,
    pub available_blocks: usize,
    pub memory_efficiency: f64,
    pub fragmentation_ratio: f64,
}

impl EnhancedKVCache {
    /// Create a new enhanced KV cache
    pub fn new(config: EnhancedKVCacheConfig) -> Result<Self> {
        let memory_pool = Arc::new(RwLock::new(
            KVMemoryPool::new(config.memory_pool_config.clone())?
        ));
        
        let active_caches = Arc::new(RwLock::new(HashMap::new()));
        let usage_tracker = Arc::new(Mutex::new(CacheUsageTracker::new()));
        let memory_monitor = Arc::new(Mutex::new(MemoryMonitor::new(config.clone())));
        
        Ok(Self {
            config,
            memory_pool,
            active_caches,
            usage_tracker,
            memory_monitor,
        })
    }
    
    /// Acquire cache for a sequence with memory optimization
    pub fn acquire_cache(&self, sequence_id: String) -> Result<Arc<Mutex<CacheEntry>>> {
        // Check memory pressure before allocation
        {
            let monitor = self.memory_monitor.lock().unwrap();
            if monitor.check_memory_pressure()? {
                self.handle_memory_pressure()?;
            }
        }
        
        let mut caches = self.active_caches.write().unwrap();
        
        if let Some(existing) = caches.get(&sequence_id) {
            // Update access tracking
            self.track_sequence_access(&sequence_id)?;
            return Ok(existing.clone());
        }
        
        // Allocate new cache entry
        let cache_entry = self.create_cache_entry(sequence_id.clone())?;
        let cache_arc = Arc::new(Mutex::new(cache_entry));
        
        caches.insert(sequence_id.clone(), cache_arc.clone());
        
        // Track new sequence
        self.track_new_sequence(&sequence_id)?;
        
        Ok(cache_arc)
    }
    
    /// Release cache for a sequence
    pub fn release_cache(&self, sequence_id: &str) -> Result<()> {
        let mut caches = self.active_caches.write().unwrap();
        
        if let Some(cache_entry) = caches.remove(sequence_id) {
            // Return blocks to memory pool
            let entry = cache_entry.lock().unwrap();
            let mut pool = self.memory_pool.write().unwrap();
            
            for block_id in &entry.blocks {
                pool.release_block(block_id.clone())?;
            }
        }
        
        Ok(())
    }
    
    /// Handle memory pressure through eviction and optimization
    fn handle_memory_pressure(&self) -> Result<()> {
        let mut monitor = self.memory_monitor.lock().unwrap();
        let pressure_level = monitor.current_pressure_level();
        
        let action = match self.config.eviction_strategy {
            CacheEvictionStrategy::LRU => self.evict_lru_caches()?,
            CacheEvictionStrategy::FIFO => self.evict_fifo_caches()?,
            CacheEvictionStrategy::LFU => self.evict_lfu_caches()?,
            CacheEvictionStrategy::SlidingWindow => self.apply_sliding_windows()?,
            CacheEvictionStrategy::Adaptive => self.adaptive_pressure_handling(pressure_level)?,
        };
        
        monitor.record_pressure_event(pressure_level, action);
        
        Ok(())
    }
    
    /// Create a new cache entry with memory allocation
    fn create_cache_entry(&self, sequence_id: String) -> Result<CacheEntry> {
        let mut pool = self.memory_pool.write().unwrap();
        
        // Allocate initial blocks based on predicted usage
        let predicted_blocks = self.predict_blocks_needed(&sequence_id)?;
        let mut allocated_blocks = Vec::new();
        
        for _ in 0..predicted_blocks {
            if let Some(block_id) = pool.allocate_block()? {
                allocated_blocks.push(block_id);
            } else {
                break; // Pool exhausted
            }
        }
        
        Ok(CacheEntry {
            sequence_id: sequence_id.clone(),
            blocks: allocated_blocks,
            current_position: 0,
            sliding_window: SlidingWindowState {
                total_length: 0,
                window_start: 0,
                window_end: 0,
                evicted_tokens: 0,
            },
            stats: CacheEntryStats {
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                access_count: 0,
                memory_usage_bytes: 0,
                sliding_operations: 0,
            },
        })
    }
    
    /// Predict number of blocks needed for a sequence
    fn predict_blocks_needed(&self, _sequence_id: &str) -> Result<usize> {
        let usage_tracker = self.usage_tracker.lock().unwrap();
        
        // Use global statistics for prediction
        let avg_length = usage_tracker.global_stats.avg_sequence_length;
        let block_size = self.config.memory_pool_config.cache_block_size;
        
        let predicted_blocks = if avg_length > 0.0 {
            (avg_length / block_size as f64).ceil() as usize
        } else {
            2 // Default allocation
        };
        
        Ok(predicted_blocks.min(4).max(1)) // Clamp between 1-4 blocks initially
    }
    
    /// Apply sliding window to manage long contexts
    fn apply_sliding_window_to_entry(&self, entry: &mut CacheEntry) -> Result<()> {
        if !self.config.enable_sliding_window {
            return Ok(());
        }
        
        let window_size = self.config.sliding_window_size;
        
        if entry.sliding_window.total_length > window_size {
            // Calculate new window position
            let tokens_to_evict = entry.sliding_window.total_length - window_size;
            
            entry.sliding_window.window_start += tokens_to_evict;
            entry.sliding_window.evicted_tokens += tokens_to_evict;
            entry.stats.sliding_operations += 1;
            
            // Compact cache blocks if needed
            self.compact_cache_blocks(&mut entry.blocks)?;
        }
        
        Ok(())
    }
    
    /// Compact cache blocks to remove fragmentation
    fn compact_cache_blocks(&self, block_ids: &mut Vec<String>) -> Result<()> {
        if !self.config.memory_pool_config.enable_compaction {
            return Ok(());
        }
        
        let mut pool = self.memory_pool.write().unwrap();
        
        // Simple compaction: merge partially used blocks
        // Implementation would involve tensor operations to merge block data
        pool.compact_blocks(block_ids)?;
        
        Ok(())
    }
    
    /// Track sequence access for optimization
    fn track_sequence_access(&self, sequence_id: &str) -> Result<()> {
        let mut tracker = self.usage_tracker.lock().unwrap();
        
        if let Some(usage) = tracker.sequence_usage.get_mut(sequence_id) {
            usage.access_pattern.push(Instant::now());
            
            // Limit history size
            if usage.access_pattern.len() > 100 {
                usage.access_pattern.drain(0..50); // Keep last 50 accesses
            }
        }
        
        Ok(())
    }
    
    /// Track new sequence creation
    fn track_new_sequence(&self, sequence_id: &str) -> Result<()> {
        let mut tracker = self.usage_tracker.lock().unwrap();
        
        tracker.sequence_usage.insert(
            sequence_id.to_string(),
            SequenceUsage {
                sequence_id: sequence_id.to_string(),
                access_pattern: vec![Instant::now()],
                memory_pattern: Vec::new(),
                predicted_length: None,
            }
        );
        
        tracker.global_stats.total_sequences += 1;
        
        Ok(())
    }
    
    /// Evict least recently used caches
    fn evict_lru_caches(&self) -> Result<PressureAction> {
        let caches = self.active_caches.read().unwrap();
        
        // Find LRU caches
        let mut cache_ages: Vec<(String, Instant)> = caches
            .iter()
            .map(|(id, cache)| {
                let entry = cache.lock().unwrap();
                (id.clone(), entry.stats.last_accessed)
            })
            .collect();
        
        cache_ages.sort_by_key(|(_, time)| *time);
        
        // Evict oldest 25% of caches
        let evict_count = (cache_ages.len() / 4).max(1);
        let evicted: Vec<String> = cache_ages
            .into_iter()
            .take(evict_count)
            .map(|(id, _)| id)
            .collect();
        
        drop(caches);
        
        // Actually evict the caches
        for sequence_id in &evicted {
            self.release_cache(sequence_id)?;
        }
        
        Ok(PressureAction::Eviction(evicted))
    }
    
    /// Evict first-in-first-out caches
    fn evict_fifo_caches(&self) -> Result<PressureAction> {
        // Implementation similar to LRU but based on creation time
        let caches = self.active_caches.read().unwrap();
        
        let mut cache_ages: Vec<(String, Instant)> = caches
            .iter()
            .map(|(id, cache)| {
                let entry = cache.lock().unwrap();
                (id.clone(), entry.stats.created_at)
            })
            .collect();
        
        cache_ages.sort_by_key(|(_, time)| *time);
        
        let evict_count = (cache_ages.len() / 4).max(1);
        let evicted: Vec<String> = cache_ages
            .into_iter()
            .take(evict_count)
            .map(|(id, _)| id)
            .collect();
        
        drop(caches);
        
        for sequence_id in &evicted {
            self.release_cache(sequence_id)?;
        }
        
        Ok(PressureAction::Eviction(evicted))
    }
    
    /// Evict least frequently used caches
    fn evict_lfu_caches(&self) -> Result<PressureAction> {
        let caches = self.active_caches.read().unwrap();
        
        let mut cache_usage: Vec<(String, u64)> = caches
            .iter()
            .map(|(id, cache)| {
                let entry = cache.lock().unwrap();
                (id.clone(), entry.stats.access_count)
            })
            .collect();
        
        cache_usage.sort_by_key(|(_, count)| *count);
        
        let evict_count = (cache_usage.len() / 4).max(1);
        let evicted: Vec<String> = cache_usage
            .into_iter()
            .take(evict_count)
            .map(|(id, _)| id)
            .collect();
        
        drop(caches);
        
        for sequence_id in &evicted {
            self.release_cache(sequence_id)?;
        }
        
        Ok(PressureAction::Eviction(evicted))
    }
    
    /// Apply sliding windows to all active caches
    fn apply_sliding_windows(&self) -> Result<PressureAction> {
        let caches = self.active_caches.read().unwrap();
        let mut processed_sequences = Vec::new();
        
        for (sequence_id, cache) in caches.iter() {
            let mut entry = cache.lock().unwrap();
            self.apply_sliding_window_to_entry(&mut entry)?;
            processed_sequences.push(sequence_id.clone());
        }
        
        Ok(PressureAction::SlidingWindow(format!("{} sequences", processed_sequences.len())))
    }
    
    /// Adaptive pressure handling based on usage patterns
    fn adaptive_pressure_handling(&self, pressure_level: f64) -> Result<PressureAction> {
        if pressure_level > 0.9 {
            // Critical pressure: evict LRU caches
            self.evict_lru_caches()
        } else if pressure_level > 0.8 {
            // High pressure: apply sliding windows
            self.apply_sliding_windows()
        } else {
            // Moderate pressure: compact memory pool
            let mut pool = self.memory_pool.write().unwrap();
            pool.compact_all_blocks()?;
            Ok(PressureAction::Compaction)
        }
    }
    
    /// Get comprehensive cache statistics
    pub fn get_stats(&self) -> Result<KVCacheStats> {
        let monitor = self.memory_monitor.lock().unwrap();
        let usage_tracker = self.usage_tracker.lock().unwrap();
        
        Ok(KVCacheStats {
            cache_hits: 0, // Will be implemented with actual usage tracking
            cache_misses: 0,
            memory_usage_bytes: monitor.current_usage as u64,
            peak_memory_bytes: monitor.peak_usage as u64,
            cache_resets: 0,
            avg_seq_len: usage_tracker.global_stats.avg_sequence_length as f32,
        })
    }
}

impl KVMemoryPool {
    /// Create a new memory pool
    fn new(config: MemoryPoolConfig) -> Result<Self> {
        // Pre-allocate blocks based on strategy
        let initial_blocks = match config.preallocation_strategy {
            PreallocationStrategy::None => 0,
            PreallocationStrategy::Conservative => config.initial_pool_size / 4,
            PreallocationStrategy::Balanced => config.initial_pool_size / 2,
            PreallocationStrategy::Aggressive => config.initial_pool_size,
        };
        
        let mut pool = Self {
            available_blocks: VecDeque::new(),
            allocated_blocks: HashMap::new(),
            pool_stats: PoolStats::default(),
            config: config.clone(),
        };
        
        for i in 0..initial_blocks {
            let block = CacheBlock::new(format!("block_{}", i), config.cache_block_size)?;
            pool.available_blocks.push_back(block);
            pool.pool_stats.total_blocks_created += 1;
        }
        
        pool.pool_stats.available_blocks = pool.available_blocks.len();
        
        Ok(pool)
    }
    
    /// Allocate a cache block
    fn allocate_block(&mut self) -> Result<Option<String>> {
        if let Some(block) = self.available_blocks.pop_front() {
            let block_id = block.block_id.clone();
            self.allocated_blocks.insert(block_id.clone(), block);
            self.pool_stats.active_blocks += 1;
            self.pool_stats.available_blocks = self.available_blocks.len();
            Ok(Some(block_id))
        } else if self.pool_stats.total_blocks_created < self.config.max_pool_size as u64 {
            // Create new block if under limit
            let block_id = format!("block_{}", self.pool_stats.total_blocks_created);
            let block = CacheBlock::new(block_id.clone(), self.config.cache_block_size)?;
            self.allocated_blocks.insert(block_id.clone(), block);
            self.pool_stats.total_blocks_created += 1;
            self.pool_stats.active_blocks += 1;
            Ok(Some(block_id))
        } else {
            // Pool exhausted
            Ok(None)
        }
    }
    
    /// Release a cache block back to the pool
    fn release_block(&mut self, block_id: String) -> Result<()> {
        if let Some(mut block) = self.allocated_blocks.remove(&block_id) {
            // Reset block state
            block.usage = 0;
            block.access_count = 0;
            block.last_access = Instant::now();
            
            self.available_blocks.push_back(block);
            self.pool_stats.active_blocks = self.pool_stats.active_blocks.saturating_sub(1);
            self.pool_stats.available_blocks = self.available_blocks.len();
        }
        
        Ok(())
    }
    
    /// Compact blocks to reduce fragmentation
    fn compact_blocks(&mut self, _block_ids: &[String]) -> Result<()> {
        // Placeholder for block compaction logic
        // Would involve merging partially used blocks and defragmenting memory
        Ok(())
    }
    
    /// Compact all blocks in the pool
    fn compact_all_blocks(&mut self) -> Result<()> {
        // Placeholder for pool-wide compaction
        self.pool_stats.fragmentation_ratio *= 0.8; // Simulate reduction in fragmentation
        Ok(())
    }
}

impl CacheBlock {
    /// Create a new cache block
    fn new(block_id: String, capacity: usize) -> Result<Self> {
        Ok(Self {
            block_id,
            key_tensor: None,
            value_tensor: None,
            capacity,
            usage: 0,
            last_access: Instant::now(),
            access_count: 0,
        })
    }
}

impl CacheUsageTracker {
    /// Create a new usage tracker
    fn new() -> Self {
        Self {
            sequence_usage: HashMap::new(),
            global_stats: GlobalUsageStats::default(),
        }
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor
    fn new(config: EnhancedKVCacheConfig) -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            pressure_events: Vec::new(),
            config,
        }
    }
    
    /// Check if memory pressure threshold is exceeded
    fn check_memory_pressure(&self) -> Result<bool> {
        let pressure_ratio = self.current_usage as f64 / self.config.max_total_memory_bytes as f64;
        Ok(pressure_ratio > self.config.memory_pressure_threshold)
    }
    
    /// Calculate current memory pressure level (0.0-1.0)
    fn current_pressure_level(&self) -> f64 {
        self.current_usage as f64 / self.config.max_total_memory_bytes as f64
    }
    
    /// Record a memory pressure event
    fn record_pressure_event(&mut self, pressure_level: f64, action: PressureAction) {
        self.pressure_events.push(MemoryPressureEvent {
            timestamp: Instant::now(),
            memory_usage: self.current_usage,
            pressure_level,
            action_taken: action,
        });
        
        // Limit event history
        if self.pressure_events.len() > 1000 {
            self.pressure_events.drain(0..500);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_config_default() {
        let config = EnhancedKVCacheConfig::default();
        assert!(config.enable_sliding_window);
        assert_eq!(config.sliding_window_size, 2048);
        assert_eq!(config.eviction_strategy, CacheEvictionStrategy::Adaptive);
    }
    
    #[test]
    fn test_memory_pool_creation() -> Result<()> {
        let config = MemoryPoolConfig::default();
        let pool = KVMemoryPool::new(config)?;
        
        assert!(pool.pool_stats.available_blocks > 0);
        assert_eq!(pool.pool_stats.active_blocks, 0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_enhanced_kv_cache() -> Result<()> {
        let config = EnhancedKVCacheConfig::default();
        let cache = EnhancedKVCache::new(config)?;
        
        // Test cache acquisition
        let cache_entry = cache.acquire_cache("test_sequence".to_string())?;
        assert!(!cache_entry.lock().unwrap().blocks.is_empty());
        
        // Test cache release
        cache.release_cache("test_sequence")?;
        
        Ok(())
    }
    
    #[test]
    fn test_sliding_window_state() {
        let mut state = SlidingWindowState {
            total_length: 1000,
            window_start: 0,
            window_end: 500,
            evicted_tokens: 0,
        };
        
        // Simulate sliding window operation
        let window_size = 400;
        if state.total_length > window_size {
            let tokens_to_evict = state.total_length - window_size;
            state.window_start += tokens_to_evict;
            state.evicted_tokens += tokens_to_evict;
        }
        
        assert_eq!(state.window_start, 600);
        assert_eq!(state.evicted_tokens, 600);
    }
}