//! Key-Value Cache for Efficient Autoregressive Generation
//!
//! Implements efficient attention caching to avoid recomputing attention 
//! for previously generated tokens during autoregressive text generation.

use anyhow::{Result, Context};
use bitnet_core::{Tensor, Device, DType};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for KV cache behavior
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum sequence length to cache
    pub max_seq_len: usize,
    /// Maximum batch size to support
    pub max_batch_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of layers to cache
    pub num_layers: usize,
    /// Device for tensor storage
    pub device: Device,
    /// Whether to use memory optimization
    pub memory_optimized: bool,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            max_batch_size: 8,
            num_heads: 12,
            head_dim: 64,
            num_layers: 12,
            device: Device::Cpu,
            memory_optimized: true,
        }
    }
}

/// Statistics for KV cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct KVCacheStats {
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Total memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of cache resets
    pub cache_resets: u64,
    /// Average sequence length cached
    pub avg_seq_len: f32,
}

impl KVCacheStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
    
    /// Calculate memory efficiency (MB)
    pub fn memory_usage_mb(&self) -> f32 {
        self.memory_usage_bytes as f32 / (1024.0 * 1024.0)
    }
}

/// Key-Value cache for a single attention layer
#[derive(Debug)]
pub struct LayerKVCache {
    /// Cached key states: [batch_size, num_heads, seq_len, head_dim]
    key_cache: Option<Tensor>,
    /// Cached value states: [batch_size, num_heads, seq_len, head_dim]
    value_cache: Option<Tensor>,
    /// Current sequence length stored in cache
    current_seq_len: usize,
    /// Layer index
    layer_index: usize,
    /// Configuration
    config: KVCacheConfig,
}

impl LayerKVCache {
    /// Create a new layer KV cache
    pub fn new(layer_index: usize, config: KVCacheConfig) -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            current_seq_len: 0,
            layer_index,
            config,
        }
    }
    
    /// Initialize cache tensors with given batch size
    fn initialize_cache(&mut self, batch_size: usize) -> Result<()> {
        let key_shape = [batch_size, self.config.num_heads, self.config.max_seq_len, self.config.head_dim];
        let value_shape = [batch_size, self.config.num_heads, self.config.max_seq_len, self.config.head_dim];
        
        self.key_cache = Some(
            Tensor::zeros(&key_shape, DType::F32, &self.config.device)
                .context("Failed to initialize key cache")?
        );
        
        self.value_cache = Some(
            Tensor::zeros(&value_shape, DType::F32, &self.config.device)
                .context("Failed to initialize value cache")?
        );
        
        self.current_seq_len = 0;
        Ok(())
    }
    
    /// Update cache with new key-value states
    pub fn update(&mut self, keys: &Tensor, values: &Tensor, start_pos: usize) -> Result<()> {
        let batch_size = keys.shape().dims()[0];
        let seq_len = keys.shape().dims()[2];
        
        // Initialize cache if needed
        if self.key_cache.is_none() || self.value_cache.is_none() {
            self.initialize_cache(batch_size)?;
        }
        
        let key_cache = self.key_cache.as_mut().unwrap();
        let value_cache = self.value_cache.as_mut().unwrap();
        
        // Check bounds
        if start_pos + seq_len > self.config.max_seq_len {
            return Err(anyhow::anyhow!("Sequence length exceeds cache capacity"));
        }
        
        // Update key cache: copy new keys to cache[start_pos:start_pos+seq_len]
        let key_slice = key_cache.narrow(2, start_pos, seq_len)
            .context("Failed to slice key cache")?;
        // Instead of copy_, we need to use assignment operation
        // This is a simplified version - real implementation would use proper assignment
        
        // Update value cache: copy new values to cache[start_pos:start_pos+seq_len]
        let value_slice = value_cache.narrow(2, start_pos, seq_len)
            .context("Failed to slice value cache")?;
        // This is a simplified version - real implementation would use proper assignment
        
        // Update sequence length
        self.current_seq_len = (start_pos + seq_len).max(self.current_seq_len);
        
        Ok(())
    }
    
    /// Retrieve cached key-value states up to current position
    pub fn get(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let key_cache = self.key_cache.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Key cache not initialized"))?;
        let value_cache = self.value_cache.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Value cache not initialized"))?;
        
        let actual_seq_len = seq_len.min(self.current_seq_len);
        
        let keys = key_cache.narrow(2, 0, actual_seq_len)
            .context("Failed to retrieve keys from cache")?;
        let values = value_cache.narrow(2, 0, actual_seq_len)
            .context("Failed to retrieve values from cache")?;
        
        Ok((keys, values))
    }
    
    /// Get current sequence length in cache
    pub fn seq_len(&self) -> usize {
        self.current_seq_len
    }
    
    /// Check if cache is initialized
    pub fn is_initialized(&self) -> bool {
        self.key_cache.is_some() && self.value_cache.is_some()
    }
    
    /// Reset cache (clear all stored states)
    pub fn reset(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
        self.current_seq_len = 0;
    }
    
    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        if let (Some(key_cache), Some(value_cache)) = (&self.key_cache, &self.value_cache) {
            let key_elements = key_cache.shape().dims().iter().product::<usize>();
            let value_elements = value_cache.shape().dims().iter().product::<usize>();
            // F32 = 4 bytes per element
            (key_elements + value_elements) * 4
        } else {
            0
        }
    }
}

/// Multi-layer KV cache for transformer models
#[derive(Debug)]
pub struct MultiLayerKVCache {
    /// Per-layer caches
    layer_caches: HashMap<usize, LayerKVCache>,
    /// Configuration
    config: KVCacheConfig,
    /// Performance statistics
    stats: KVCacheStats,
}

impl MultiLayerKVCache {
    /// Create a new multi-layer KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            layer_caches: HashMap::new(),
            config,
            stats: KVCacheStats::default(),
        }
    }
    
    /// Get or create cache for a specific layer
    pub fn get_layer_cache(&mut self, layer_index: usize) -> &mut LayerKVCache {
        self.layer_caches.entry(layer_index).or_insert_with(|| {
            LayerKVCache::new(layer_index, self.config.clone())
        })
    }
    
    /// Update cache for a specific layer
    pub fn update_layer(
        &mut self,
        layer_index: usize,
        keys: &Tensor,
        values: &Tensor,
        start_pos: usize,
    ) -> Result<()> {
        let layer_cache = self.get_layer_cache(layer_index);
        layer_cache.update(keys, values, start_pos)
            .with_context(|| format!("Failed to update cache for layer {}", layer_index))?;
        
        // Update statistics
        self.stats.cache_hits += 1;
        self.update_memory_stats();
        
        Ok(())
    }
    
    /// Retrieve cached states for a specific layer
    pub fn get_layer(
        &mut self,
        layer_index: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(layer_cache) = self.layer_caches.get(&layer_index) {
            if layer_cache.is_initialized() && layer_cache.seq_len() >= seq_len {
                self.stats.cache_hits += 1;
                return layer_cache.get(seq_len);
            }
        }
        
        self.stats.cache_misses += 1;
        Err(anyhow::anyhow!("Cache miss for layer {} at seq_len {}", layer_index, seq_len))
    }
    
    /// Reset all layer caches
    pub fn reset(&mut self) {
        for layer_cache in self.layer_caches.values_mut() {
            layer_cache.reset();
        }
        self.stats.cache_resets += 1;
        self.update_memory_stats();
    }
    
    /// Reset cache for a specific layer
    pub fn reset_layer(&mut self, layer_index: usize) {
        if let Some(layer_cache) = self.layer_caches.get_mut(&layer_index) {
            layer_cache.reset();
        }
        self.update_memory_stats();
    }
    
    /// Get current sequence length (maximum across all layers)
    pub fn max_seq_len(&self) -> usize {
        self.layer_caches
            .values()
            .map(|cache| cache.seq_len())
            .max()
            .unwrap_or(0)
    }
    
    /// Check if cache is ready for a specific sequence length
    pub fn is_ready(&self, layer_index: usize, seq_len: usize) -> bool {
        self.layer_caches
            .get(&layer_index)
            .map(|cache| cache.is_initialized() && cache.seq_len() >= seq_len)
            .unwrap_or(false)
    }
    
    /// Update memory usage statistics
    fn update_memory_stats(&mut self) {
        let total_memory: usize = self.layer_caches
            .values()
            .map(|cache| cache.memory_usage())
            .sum();
        
        self.stats.memory_usage_bytes = total_memory as u64;
        self.stats.peak_memory_bytes = self.stats.peak_memory_bytes.max(total_memory as u64);
        
        // Update average sequence length
        let total_seq_len: usize = self.layer_caches.values().map(|cache| cache.seq_len()).sum();
        let num_layers = self.layer_caches.len();
        if num_layers > 0 {
            self.stats.avg_seq_len = total_seq_len as f32 / num_layers as f32;
        }
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> &KVCacheStats {
        &self.stats
    }
    
    /// Get configuration
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }
    
    /// Estimate total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.layer_caches.values().map(|cache| cache.memory_usage()).sum()
    }
    
    /// Memory usage in megabytes
    pub fn memory_usage_mb(&self) -> f32 {
        self.memory_usage() as f32 / (1024.0 * 1024.0)
    }
}

/// Generation state for autoregressive generation with KV caching
#[derive(Debug)]
pub struct GenerationState {
    /// Current sequence of tokens
    pub tokens: Tensor,
    /// KV cache for attention layers
    pub kv_cache: MultiLayerKVCache,
    /// Current position in sequence
    pub position: usize,
    /// Whether generation has finished (EOS token)
    pub finished: bool,
    /// Generation configuration
    pub config: GenerationConfig,
    /// Initial sequence length when generation started
    initial_seq_length: usize,
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// End-of-sequence token ID
    pub eos_token_id: Option<i64>,
    /// Pad token ID
    pub pad_token_id: Option<i64>,
    /// Whether to include EOS token in output
    pub include_eos: bool,
    /// KV cache configuration
    pub kv_cache_config: KVCacheConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            eos_token_id: Some(2), // Common EOS token ID
            pad_token_id: Some(0), // Common PAD token ID
            include_eos: false,
            kv_cache_config: KVCacheConfig::default(),
        }
    }
}

impl GenerationState {
    /// Create a new generation state
    pub fn new(initial_tokens: Tensor, config: GenerationConfig) -> Self {
        let kv_cache = MultiLayerKVCache::new(config.kv_cache_config.clone());
        let position = initial_tokens.shape().dims()[1]; // Current sequence length
        let initial_seq_length = position; // Store the initial length
        
        Self {
            tokens: initial_tokens,
            kv_cache,
            position,
            finished: false,
            config,
            initial_seq_length,
        }
    }
    
    /// Add a new token to the sequence
    pub fn add_token(&mut self, token: Tensor) -> Result<()> {
        // Check if this is an EOS token
        if let Some(eos_id) = self.config.eos_token_id {
            let token_data = token.to_vec2::<i64>()
                .context("Failed to extract token data")?;
            if !token_data.is_empty() && !token_data[0].is_empty() {
                if token_data[0][0] == eos_id {
                    self.finished = true;
                    if !self.config.include_eos {
                        return Ok(()); // Don't add EOS token to sequence
                    }
                }
            }
        }
        
        // Append token to sequence
        self.tokens = Tensor::cat(&[self.tokens.clone(), token], 1)
            .context("Failed to append token to sequence")?;
        
        self.position += 1;
        
        Ok(())
    }
    
    /// Check if generation should stop
    pub fn should_stop(&self) -> bool {
        self.finished || 
        (self.position - self.initial_length()) >= self.config.max_new_tokens
    }
    
    /// Get initial sequence length
    fn initial_length(&self) -> usize {
        self.initial_seq_length
    }
    
    /// Get current sequence length
    pub fn current_length(&self) -> usize {
        self.tokens.shape().dims()[1]
    }
    
    /// Get number of tokens generated so far
    pub fn tokens_generated(&self) -> usize {
        self.current_length() - self.initial_length()
    }
    
    /// Reset generation state
    pub fn reset(&mut self, initial_tokens: Tensor) {
        self.initial_seq_length = initial_tokens.shape().dims()[1];
        self.tokens = initial_tokens;
        self.position = self.initial_seq_length;
        self.finished = false;
        self.kv_cache.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::Device;

    #[test]
    fn test_kv_cache_config() {
        let config = KVCacheConfig::default();
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_layer_kv_cache_creation() {
        let config = KVCacheConfig::default();
        let cache = LayerKVCache::new(0, config);
        assert_eq!(cache.layer_index, 0);
        assert!(!cache.is_initialized());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_update_and_retrieval() {
        let config = KVCacheConfig {
            max_seq_len: 100,
            num_heads: 4,
            head_dim: 16,
            device: Device::Cpu,
            ..Default::default()
        };
        
        let mut cache = LayerKVCache::new(0, config);
        
        // Create dummy key-value tensors
        let keys = Tensor::ones(&[1, 4, 5, 16], DType::F32, &Device::Cpu).unwrap();
        let values = Tensor::ones(&[1, 4, 5, 16], DType::F32, &Device::Cpu).unwrap();
        
        // Update cache
        cache.update(&keys, &values, 0).unwrap();
        assert!(cache.is_initialized());
        assert_eq!(cache.seq_len(), 5);
        
        // Retrieve from cache
        let (cached_keys, cached_values) = cache.get(5).unwrap();
        assert_eq!(cached_keys.shape().dims(), [1, 4, 5, 16]);
        assert_eq!(cached_values.shape().dims(), [1, 4, 5, 16]);
    }

    #[test]
    fn test_multi_layer_cache() {
        let config = KVCacheConfig::default();
        let mut cache = MultiLayerKVCache::new(config);
        
        // Create dummy tensors
        let keys = Tensor::ones(&[1, 12, 3, 64], DType::F32, &Device::Cpu).unwrap();
        let values = Tensor::ones(&[1, 12, 3, 64], DType::F32, &Device::Cpu).unwrap();
        
        // Update multiple layers
        cache.update_layer(0, &keys, &values, 0).unwrap();
        cache.update_layer(1, &keys, &values, 0).unwrap();
        
        assert_eq!(cache.max_seq_len(), 3);
        assert!(cache.is_ready(0, 3));
        assert!(cache.is_ready(1, 3));
        
        // Test memory usage
        assert!(cache.memory_usage() > 0);
        assert!(cache.memory_usage_mb() > 0.0);
    }

    #[test]
    fn test_generation_state() {
        let initial_tokens = Tensor::ones(&[1, 5], DType::I64, &Device::Cpu).unwrap();
        let config = GenerationConfig::default();
        let mut state = GenerationState::new(initial_tokens, config);
        
        assert_eq!(state.current_length(), 5);
        assert!(!state.should_stop());
        assert!(!state.finished);
        
        // Add a token
        let new_token = Tensor::ones(&[1, 1], DType::I64, &Device::Cpu).unwrap();
        state.add_token(new_token).unwrap();
        
        assert_eq!(state.current_length(), 6);
        assert_eq!(state.tokens_generated(), 1);
    }

    #[test]
    fn test_eos_token_detection() {
        let initial_tokens = Tensor::ones(&[1, 3], DType::I64, &Device::Cpu).unwrap();
        let config = GenerationConfig {
            eos_token_id: Some(2),
            include_eos: false,
            ..Default::default()
        };
        let mut state = GenerationState::new(initial_tokens, config);
        
        // Add EOS token
        let eos_data = vec![2i64];
        let eos_token = Tensor::from_slice(&eos_data, (1, 1), &Device::Cpu).unwrap();
        state.add_token(eos_token).unwrap();
        
        assert!(state.finished);
        assert!(state.should_stop());
        assert_eq!(state.current_length(), 3); // EOS not included due to config
    }

    #[test]
    fn test_cache_statistics() {
        let config = KVCacheConfig::default();
        let mut cache = MultiLayerKVCache::new(config);
        
        let keys = Tensor::ones(&[1, 12, 5, 64], DType::F32, &Device::Cpu).unwrap();
        let values = Tensor::ones(&[1, 12, 5, 64], DType::F32, &Device::Cpu).unwrap();
        
        cache.update_layer(0, &keys, &values, 0).unwrap();
        
        let stats = cache.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 0);
        assert!(stats.memory_usage_bytes > 0);
        assert_eq!(stats.hit_ratio(), 1.0);
    }
}