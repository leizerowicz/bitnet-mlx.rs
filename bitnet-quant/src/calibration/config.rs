//! Calibration configuration and validation
//!
//! This module provides configuration management for the calibration system,
//! including validation, builder pattern, and default configurations.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use crate::calibration::sampling::SamplingStrategy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main configuration for calibration dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum number of samples to process
    pub max_samples: usize,
    /// Enable streaming for large datasets
    pub streaming_enabled: bool,
    /// Memory limit for dataset loading (bytes)
    pub memory_limit: usize,
    /// Sampling strategy to use
    pub sampling_strategy: SamplingStrategy,
    /// Device to use for processing
    pub device: DeviceConfig,
    /// Statistics collection configuration
    pub statistics_config: StatisticsConfig,
    /// Histogram collection configuration
    pub histogram_config: HistogramConfig,
    /// Persistence configuration
    pub persistence_config: PersistenceConfig,
    /// Streaming configuration
    pub streaming_config: StreamingConfig,
    /// Validation settings
    pub validation_config: ValidationConfig,
    /// Timeout settings (seconds)
    pub timeout_seconds: Option<u64>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of worker threads
    pub num_workers: usize,
    /// Enable progress reporting
    pub progress_reporting: bool,
}

/// Device configuration for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Auto-select optimal device
    pub auto_select: bool,
    /// Preferred device type
    pub preferred_device: PreferredDevice,
    /// Memory fraction to use on GPU
    pub gpu_memory_fraction: f32,
}

/// Preferred device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreferredDevice {
    Cpu,
    Gpu,
    Metal,
    Auto,
}

/// Statistics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Track min/max values per layer
    pub track_min_max: bool,
    /// Track mean and variance
    pub track_moments: bool,
    /// Track percentiles
    pub track_percentiles: bool,
    /// Percentile values to track
    pub percentiles: Vec<f32>,
    /// Update frequency for statistics
    pub update_frequency: usize,
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: f32,
}

/// Histogram collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramConfig {
    /// Enable histogram collection
    pub enabled: bool,
    /// Number of bins for histograms
    pub num_bins: usize,
    /// Histogram range strategy
    pub range_strategy: HistogramRangeStrategy,
    /// Custom range (if using custom strategy)
    pub custom_range: Option<(f32, f32)>,
    /// Enable adaptive binning
    pub adaptive_binning: bool,
    /// Bin refinement threshold
    pub refinement_threshold: f32,
}

/// Histogram range strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HistogramRangeStrategy {
    /// Use min/max from data
    MinMax,
    /// Use percentile-based range
    Percentile { lower: f32, upper: f32 },
    /// Use custom range
    Custom,
    /// Use adaptive range
    Adaptive,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Enable automatic saving
    pub auto_save: bool,
    /// Save directory path
    pub save_directory: Option<PathBuf>,
    /// Storage format
    pub storage_format: StorageFormat,
    /// Compression enabled
    pub compression_enabled: bool,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Cache size (number of entries)
    pub cache_size: usize,
    /// Enable checksums for integrity
    pub enable_checksums: bool,
}

/// Storage formats for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    Json,
    Bincode,
    MessagePack,
    Parquet,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Chunk size for streaming processing
    pub chunk_size: usize,
    /// Number of chunks to prefetch
    pub prefetch_chunks: usize,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    /// Enable parallel chunk processing
    pub parallel_processing: bool,
    /// Maximum parallel chunks
    pub max_parallel_chunks: usize,
    /// Enable memory mapping for large files
    pub memory_mapping: bool,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum samples required
    pub min_samples: usize,
    /// Maximum samples allowed
    pub max_samples: Option<usize>,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: Option<usize>,
    /// Required tensor shapes (layer_name -> expected_shape)
    pub required_shapes: HashMap<String, Vec<usize>>,
    /// Allow missing layers
    pub allow_missing_layers: bool,
    /// Strict validation mode
    pub strict_mode: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_samples: 1000,
            streaming_enabled: true,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            sampling_strategy: SamplingStrategy::Random,
            device: DeviceConfig::default(),
            statistics_config: StatisticsConfig::default(),
            histogram_config: HistogramConfig::default(),
            persistence_config: PersistenceConfig::default(),
            streaming_config: StreamingConfig::default(),
            validation_config: ValidationConfig::default(),
            timeout_seconds: Some(3600), // 1 hour
            random_seed: None,
            num_workers: num_cpus::get(),
            progress_reporting: true,
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            auto_select: true,
            preferred_device: PreferredDevice::Auto,
            gpu_memory_fraction: 0.8,
        }
    }
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            track_min_max: true,
            track_moments: true,
            track_percentiles: true,
            percentiles: vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0],
            update_frequency: 100,
            outlier_detection: true,
            outlier_threshold: 3.0,
        }
    }
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_bins: 256,
            range_strategy: HistogramRangeStrategy::Percentile {
                lower: 0.1,
                upper: 99.9,
            },
            custom_range: None,
            adaptive_binning: true,
            refinement_threshold: 0.01,
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            auto_save: true,
            save_directory: None,
            storage_format: StorageFormat::Bincode,
            compression_enabled: true,
            compression_level: 6,
            cache_size: 100,
            enable_checksums: true,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,
            prefetch_chunks: 2,
            buffer_size: 64 * 1024, // 64KB
            parallel_processing: true,
            max_parallel_chunks: num_cpus::get(),
            memory_mapping: false,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: None,
            min_batch_size: 1,
            max_batch_size: None,
            required_shapes: HashMap::new(),
            allow_missing_layers: false,
            strict_mode: false,
        }
    }
}

/// Builder for calibration configuration
#[derive(Debug, Default)]
pub struct CalibrationConfigBuilder {
    config: CalibrationConfig,
}

impl CalibrationConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set maximum samples
    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Enable or disable streaming
    pub fn enable_streaming(mut self, enabled: bool) -> Self {
        self.config.streaming_enabled = enabled;
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.config.memory_limit = limit;
        self
    }

    /// Set sampling strategy
    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = strategy;
        self
    }

    /// Set device configuration
    pub fn device_config(mut self, device_config: DeviceConfig) -> Self {
        self.config.device = device_config;
        self
    }

    /// Set statistics configuration
    pub fn statistics_config(mut self, stats_config: StatisticsConfig) -> Self {
        self.config.statistics_config = stats_config;
        self
    }

    /// Set histogram configuration
    pub fn histogram_config(mut self, hist_config: HistogramConfig) -> Self {
        self.config.histogram_config = hist_config;
        self
    }

    /// Set persistence configuration
    pub fn persistence_config(mut self, persist_config: PersistenceConfig) -> Self {
        self.config.persistence_config = persist_config;
        self
    }

    /// Set streaming configuration
    pub fn streaming_config(mut self, stream_config: StreamingConfig) -> Self {
        self.config.streaming_config = stream_config;
        self
    }

    /// Set validation configuration
    pub fn validation_config(mut self, validation_config: ValidationConfig) -> Self {
        self.config.validation_config = validation_config;
        self
    }

    /// Set timeout
    pub fn timeout_seconds(mut self, timeout: Option<u64>) -> Self {
        self.config.timeout_seconds = timeout;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: Option<u64>) -> Self {
        self.config.random_seed = seed;
        self
    }

    /// Set number of workers
    pub fn num_workers(mut self, workers: usize) -> Self {
        self.config.num_workers = workers;
        self
    }

    /// Enable or disable progress reporting
    pub fn progress_reporting(mut self, enabled: bool) -> Self {
        self.config.progress_reporting = enabled;
        self
    }

    /// Build the configuration with validation
    pub fn build(self) -> CalibrationResult<CalibrationConfig> {
        self.validate()?;
        Ok(self.config)
    }

    /// Build without validation (use with caution)
    pub fn build_unchecked(self) -> CalibrationConfig {
        self.config
    }

    /// Validate the configuration
    fn validate(&self) -> CalibrationResult<()> {
        let config = &self.config;

        // Validate batch size
        if config.batch_size == 0 {
            return Err(CalibrationError::validation(
                "batch_size",
                "Batch size must be greater than 0",
            ));
        }

        if config.batch_size > config.max_samples {
            return Err(CalibrationError::validation(
                "batch_size",
                "Batch size cannot be larger than max_samples",
            ));
        }

        // Validate memory limit
        if config.memory_limit < 1024 * 1024 {
            return Err(CalibrationError::validation(
                "memory_limit",
                "Memory limit must be at least 1MB",
            ));
        }

        // Validate worker count
        if config.num_workers == 0 {
            return Err(CalibrationError::validation(
                "num_workers",
                "Number of workers must be greater than 0",
            ));
        }

        // Validate streaming configuration
        if config.streaming_enabled {
            if config.streaming_config.chunk_size == 0 {
                return Err(CalibrationError::validation(
                    "streaming_config.chunk_size",
                    "Chunk size must be greater than 0",
                ));
            }

            if config.streaming_config.max_parallel_chunks == 0 {
                return Err(CalibrationError::validation(
                    "streaming_config.max_parallel_chunks",
                    "Max parallel chunks must be greater than 0",
                ));
            }
        }

        // Validate histogram configuration
        if config.histogram_config.enabled {
            if config.histogram_config.num_bins == 0 {
                return Err(CalibrationError::validation(
                    "histogram_config.num_bins",
                    "Number of bins must be greater than 0",
                ));
            }

            if let HistogramRangeStrategy::Percentile { lower, upper } =
                &config.histogram_config.range_strategy
            {
                if *lower >= *upper {
                    return Err(CalibrationError::validation(
                        "histogram_config.range_strategy",
                        "Lower percentile must be less than upper percentile",
                    ));
                }
                if *lower < 0.0 || *upper > 100.0 {
                    return Err(CalibrationError::validation(
                        "histogram_config.range_strategy",
                        "Percentiles must be between 0 and 100",
                    ));
                }
            }
        }

        // Validate statistics configuration
        for percentile in &config.statistics_config.percentiles {
            if *percentile < 0.0 || *percentile > 100.0 {
                return Err(CalibrationError::validation(
                    "statistics_config.percentiles",
                    "Percentiles must be between 0 and 100",
                ));
            }
        }

        // Validate validation configuration
        if config.validation_config.min_samples > config.max_samples {
            return Err(CalibrationError::validation(
                "validation_config.min_samples",
                "Minimum samples cannot be greater than maximum samples",
            ));
        }

        Ok(())
    }
}

/// Configuration validation error
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Invalid configuration: {0}")]
    Invalid(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Value out of range: {field} = {value}, expected {expected}")]
    OutOfRange {
        field: String,
        value: String,
        expected: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CalibrationConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_samples, 1000);
        assert!(config.streaming_enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = CalibrationConfigBuilder::new()
            .batch_size(64)
            .max_samples(2000)
            .enable_streaming(false)
            .build()
            .unwrap();

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.max_samples, 2000);
        assert!(!config.streaming_enabled);
    }

    #[test]
    fn test_config_validation() {
        let result = CalibrationConfigBuilder::new()
            .batch_size(0) // Invalid: zero batch size
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_batch_size_larger_than_max_samples() {
        let result = CalibrationConfigBuilder::new()
            .batch_size(2000)
            .max_samples(1000) // batch_size > max_samples
            .build();

        assert!(result.is_err());
    }
}
