//! Configuration for Memory-Efficient Data Conversion
//!
//! This module provides configuration structures for customizing the behavior
//! of the data conversion system, including streaming parameters, batch settings,
//! and performance tuning options.

use crate::memory::conversion::{ConversionQuality, ConversionStrategy};
use serde::{Deserialize, Serialize};

/// Main configuration for the conversion engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Default conversion strategy
    pub default_strategy: ConversionStrategy,
    /// Default conversion quality
    pub default_quality: ConversionQuality,
    /// Whether to enable conversion metrics collection
    pub enable_metrics: bool,
    /// Whether to enable debug logging
    pub enable_debug_logging: bool,
    /// Maximum memory usage for conversions (in bytes)
    pub max_memory_usage: usize,
    /// Number of worker threads for parallel conversions
    pub worker_threads: usize,
    /// Streaming conversion configuration
    pub streaming: StreamingConfig,
    /// Batch conversion configuration
    pub batch: BatchConfig,
    /// Performance tuning options
    pub performance: PerformanceConfig,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            default_strategy: ConversionStrategy::Auto,
            default_quality: ConversionQuality::Balanced,
            enable_metrics: true,
            enable_debug_logging: false,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            worker_threads: num_cpus::get(),
            streaming: StreamingConfig::default(),
            batch: BatchConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl ConversionConfig {
    /// Creates a configuration optimized for low memory usage
    pub fn low_memory() -> Self {
        Self {
            default_strategy: ConversionStrategy::Streaming,
            max_memory_usage: 256 * 1024 * 1024, // 256MB
            streaming: StreamingConfig::low_memory(),
            batch: BatchConfig::low_memory(),
            performance: PerformanceConfig::low_memory(),
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for high performance
    pub fn high_performance() -> Self {
        Self {
            default_strategy: ConversionStrategy::Auto,
            default_quality: ConversionQuality::Fast,
            max_memory_usage: 4 * 1024 * 1024 * 1024, // 4GB
            worker_threads: num_cpus::get() * 2,
            streaming: StreamingConfig::high_performance(),
            batch: BatchConfig::high_performance(),
            performance: PerformanceConfig::high_performance(),
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for precision
    pub fn high_precision() -> Self {
        Self {
            default_quality: ConversionQuality::Precise,
            streaming: StreamingConfig::high_precision(),
            batch: BatchConfig::high_precision(),
            performance: PerformanceConfig::high_precision(),
            ..Default::default()
        }
    }

    /// Validates the configuration and returns any issues
    pub fn validate(&self) -> Result<(), String> {
        if self.max_memory_usage == 0 {
            return Err("max_memory_usage must be greater than 0".to_string());
        }

        if self.worker_threads == 0 {
            return Err("worker_threads must be greater than 0".to_string());
        }

        self.streaming.validate()?;
        self.batch.validate()?;
        self.performance.validate()?;

        Ok(())
    }
}

/// Configuration for streaming conversions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Number of chunks to process in parallel
    pub parallel_chunks: usize,
    /// Buffer size for intermediate results
    pub buffer_size: usize,
    /// Whether to use memory mapping for large files
    pub use_memory_mapping: bool,
    /// Threshold for switching to streaming mode (in bytes)
    pub streaming_threshold: usize,
    /// Whether to prefetch next chunk while processing current
    pub enable_prefetch: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1MB
            parallel_chunks: 2,
            buffer_size: 4 * 1024 * 1024, // 4MB
            use_memory_mapping: true,
            streaming_threshold: 100 * 1024 * 1024, // 100MB
            enable_prefetch: true,
        }
    }
}

impl StreamingConfig {
    /// Configuration optimized for low memory usage
    pub fn low_memory() -> Self {
        Self {
            chunk_size: 256 * 1024, // 256KB
            parallel_chunks: 1,
            buffer_size: 1024 * 1024,              // 1MB
            streaming_threshold: 10 * 1024 * 1024, // 10MB
            enable_prefetch: false,
            ..Default::default()
        }
    }

    /// Configuration optimized for high performance
    pub fn high_performance() -> Self {
        Self {
            chunk_size: 4 * 1024 * 1024, // 4MB
            parallel_chunks: num_cpus::get(),
            buffer_size: 16 * 1024 * 1024,          // 16MB
            streaming_threshold: 500 * 1024 * 1024, // 500MB
            enable_prefetch: true,
            ..Default::default()
        }
    }

    /// Configuration optimized for precision
    pub fn high_precision() -> Self {
        Self {
            chunk_size: 512 * 1024, // 512KB (smaller chunks for better precision)
            parallel_chunks: 1,     // Sequential processing for consistency
            enable_prefetch: false,
            ..Default::default()
        }
    }

    /// Validates the streaming configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0".to_string());
        }

        if self.parallel_chunks == 0 {
            return Err("parallel_chunks must be greater than 0".to_string());
        }

        if self.buffer_size < self.chunk_size {
            return Err("buffer_size must be at least as large as chunk_size".to_string());
        }

        if self.streaming_threshold == 0 {
            return Err("streaming_threshold must be greater than 0".to_string());
        }

        Ok(())
    }
}

/// Configuration for batch conversions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of tensors to process in a single batch
    pub max_batch_size: usize,
    /// Whether to sort tensors by size before batching
    pub sort_by_size: bool,
    /// Whether to group tensors by data type
    pub group_by_dtype: bool,
    /// Whether to group tensors by device
    pub group_by_device: bool,
    /// Maximum memory usage per batch (in bytes)
    pub max_batch_memory: usize,
    /// Whether to use parallel processing within batches
    pub enable_parallel_processing: bool,
    /// Number of worker threads for batch processing
    pub batch_worker_threads: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            sort_by_size: true,
            group_by_dtype: true,
            group_by_device: true,
            max_batch_memory: 256 * 1024 * 1024, // 256MB
            enable_parallel_processing: true,
            batch_worker_threads: num_cpus::get(),
        }
    }
}

impl BatchConfig {
    /// Configuration optimized for low memory usage
    pub fn low_memory() -> Self {
        Self {
            max_batch_size: 8,
            max_batch_memory: 64 * 1024 * 1024, // 64MB
            enable_parallel_processing: false,
            batch_worker_threads: 1,
            ..Default::default()
        }
    }

    /// Configuration optimized for high performance
    pub fn high_performance() -> Self {
        Self {
            max_batch_size: 128,
            max_batch_memory: 1024 * 1024 * 1024, // 1GB
            enable_parallel_processing: true,
            batch_worker_threads: num_cpus::get() * 2,
            ..Default::default()
        }
    }

    /// Configuration optimized for precision
    pub fn high_precision() -> Self {
        Self {
            max_batch_size: 16,
            sort_by_size: false,               // Preserve original order
            enable_parallel_processing: false, // Sequential for consistency
            batch_worker_threads: 1,
            ..Default::default()
        }
    }

    /// Validates the batch configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be greater than 0".to_string());
        }

        if self.max_batch_memory == 0 {
            return Err("max_batch_memory must be greater than 0".to_string());
        }

        if self.batch_worker_threads == 0 {
            return Err("batch_worker_threads must be greater than 0".to_string());
        }

        Ok(())
    }
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Whether to use SIMD instructions when available
    pub use_simd: bool,
    /// Whether to enable CPU cache optimization
    pub optimize_cache: bool,
    /// Whether to use vectorized operations
    pub use_vectorization: bool,
    /// Memory alignment for optimal performance (in bytes)
    pub memory_alignment: usize,
    /// Whether to enable loop unrolling
    pub enable_loop_unrolling: bool,
    /// Threshold for switching to optimized kernels (in elements)
    pub optimization_threshold: usize,
    /// Whether to use lookup tables for quantization
    pub use_lookup_tables: bool,
    /// Size of lookup tables (number of entries)
    pub lookup_table_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            optimize_cache: true,
            use_vectorization: true,
            memory_alignment: 64, // 64-byte alignment for AVX-512
            enable_loop_unrolling: true,
            optimization_threshold: 1024,
            use_lookup_tables: true,
            lookup_table_size: 256,
        }
    }
}

impl PerformanceConfig {
    /// Configuration optimized for low memory usage
    pub fn low_memory() -> Self {
        Self {
            use_lookup_tables: false, // Lookup tables use extra memory
            lookup_table_size: 64,
            memory_alignment: 16, // Smaller alignment
            ..Default::default()
        }
    }

    /// Configuration optimized for high performance
    pub fn high_performance() -> Self {
        Self {
            use_simd: true,
            optimize_cache: true,
            use_vectorization: true,
            memory_alignment: 64,
            enable_loop_unrolling: true,
            optimization_threshold: 512, // Lower threshold for more optimizations
            use_lookup_tables: true,
            lookup_table_size: 1024, // Larger lookup tables
        }
    }

    /// Configuration optimized for precision
    pub fn high_precision() -> Self {
        Self {
            use_simd: false, // SIMD might introduce precision differences
            use_vectorization: false,
            enable_loop_unrolling: false,
            optimization_threshold: usize::MAX, // Disable optimizations
            use_lookup_tables: false,           // Use exact calculations
            ..Default::default()
        }
    }

    /// Validates the performance configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.memory_alignment == 0 || !self.memory_alignment.is_power_of_two() {
            return Err("memory_alignment must be a power of 2 greater than 0".to_string());
        }

        if self.lookup_table_size == 0 {
            return Err("lookup_table_size must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ConversionConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.default_strategy, ConversionStrategy::Auto);
        assert_eq!(config.default_quality, ConversionQuality::Balanced);
        assert!(config.enable_metrics);
    }

    #[test]
    fn test_low_memory_config() {
        let config = ConversionConfig::low_memory();
        assert!(config.validate().is_ok());
        assert_eq!(config.default_strategy, ConversionStrategy::Streaming);
        assert_eq!(config.max_memory_usage, 256 * 1024 * 1024);
        assert_eq!(config.streaming.chunk_size, 256 * 1024);
    }

    #[test]
    fn test_high_performance_config() {
        let config = ConversionConfig::high_performance();
        assert!(config.validate().is_ok());
        assert_eq!(config.default_quality, ConversionQuality::Fast);
        assert!(config.performance.use_simd);
        assert!(config.performance.use_vectorization);
    }

    #[test]
    fn test_high_precision_config() {
        let config = ConversionConfig::high_precision();
        assert!(config.validate().is_ok());
        assert_eq!(config.default_quality, ConversionQuality::Precise);
        assert!(!config.performance.use_simd);
        assert!(!config.performance.use_vectorization);
    }

    #[test]
    fn test_streaming_config_validation() {
        let mut config = StreamingConfig::default();
        assert!(config.validate().is_ok());

        config.chunk_size = 0;
        assert!(config.validate().is_err());

        config.chunk_size = 1024;
        config.buffer_size = 512; // Smaller than chunk_size
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_batch_config_validation() {
        let mut config = BatchConfig::default();
        assert!(config.validate().is_ok());

        config.max_batch_size = 0;
        assert!(config.validate().is_err());

        config.max_batch_size = 10;
        config.max_batch_memory = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_performance_config_validation() {
        let mut config = PerformanceConfig::default();
        assert!(config.validate().is_ok());

        config.memory_alignment = 0;
        assert!(config.validate().is_err());

        config.memory_alignment = 3; // Not a power of 2
        assert!(config.validate().is_err());

        config.memory_alignment = 16;
        config.lookup_table_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = ConversionConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ConversionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.default_strategy, deserialized.default_strategy);
        assert_eq!(config.max_memory_usage, deserialized.max_memory_usage);
        assert_eq!(
            config.streaming.chunk_size,
            deserialized.streaming.chunk_size
        );
    }
}
