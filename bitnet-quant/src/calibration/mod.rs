//! Calibration system for BitNet quantization
//!
//! This module provides comprehensive calibration infrastructure for optimizing
//! quantization parameters using representative datasets. It supports streaming
//! large datasets, collecting activation statistics, and generating optimal
//! quantization parameters for improved model performance.
//!
//! # Features
//!
//! - **Streaming datasets**: Process datasets larger than available memory
//! - **Statistics collection**: Track activation statistics, min/max values, histograms
//! - **Representative sampling**: Efficient sampling strategies for calibration
//! - **Persistence**: Save and load calibration statistics for reuse
//! - **Memory efficient**: Integration with existing memory management system
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use bitnet_quant::calibration::{CalibrationDataset, CalibrationConfig, SamplingStrategy};
//! use candle_core::{Tensor, Device};
//!
//! // Create calibration dataset
//! let config = CalibrationConfig::default();
//! let mut dataset = CalibrationDataset::new(config)?;
//!
//! // Load data and collect statistics  
//! dataset.load_from_path("path/to/calibration/data")?;
//! let statistics = dataset.collect_statistics()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod dataset;
pub mod error;
pub mod histogram;
pub mod persistence;
pub mod sampling;
pub mod statistics;
pub mod streaming;

use std::time::SystemTime;

// Re-export main types
pub use config::{
    CalibrationConfig, CalibrationConfigBuilder, HistogramConfig, PersistenceConfig, StorageFormat,
    StreamingConfig, ValidationError,
};
pub use dataset::{BatchProcessor, CalibrationDataset, DatasetIterator};
pub use error::{CalibrationError, CalibrationResult};
pub use histogram::{ActivationHistogram, HistogramBin, HistogramCollector, QuantizationOptimizer};
pub use persistence::{CacheEntry, CalibrationCache, StatisticsPersistence};
pub use sampling::{
    ImportanceSampler, RandomSampler, RepresentativeSampler, SamplingStrategy, StratifiedSampler,
};
pub use statistics::{LayerStatistics, MinMaxTracker, StatisticsCollector, StatisticsUpdate};
pub use streaming::{ChunkProcessor, StreamingProcessor};

/// Core calibration result containing all collected statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct CalibrationSummary {
    /// Activation statistics per layer
    pub layer_statistics: std::collections::HashMap<String, LayerStatistics>,
    /// Histogram data for quantization optimization
    pub histograms: std::collections::HashMap<String, ActivationHistogram>,
    /// Optimal quantization parameters
    pub quantization_params: std::collections::HashMap<String, QuantizationParameters>,
    /// Collection metadata
    pub metadata: CalibrationMetadata,
}

/// Quantization parameters optimized from calibration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct QuantizationParameters {
    /// Scaling factor for quantization
    pub scale: f32,
    /// Zero point for symmetric/asymmetric quantization
    pub zero_point: i32,
    /// Minimum value observed
    pub min_value: f32,
    /// Maximum value observed
    pub max_value: f32,
    /// Optimal bit width (if dynamic)
    pub bit_width: u8,
    /// Quantization scheme confidence score
    pub confidence: f32,
}

/// Metadata about calibration process
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct CalibrationMetadata {
    /// Number of samples processed
    pub samples_processed: usize,
    /// Processing time in seconds
    pub processing_time: f64,
    /// Memory usage peak (bytes)
    pub peak_memory_usage: usize,
    /// Calibration configuration used
    pub config_hash: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Factory for creating calibration components
pub struct CalibrationFactory;

impl CalibrationFactory {
    /// Create a new calibration dataset with the given configuration
    pub fn create_dataset(config: CalibrationConfig) -> CalibrationResult<CalibrationDataset> {
        CalibrationDataset::new(config)
    }

    /// Create a statistics collector
    pub fn create_statistics_collector() -> StatisticsCollector {
        StatisticsCollector::new()
    }

    /// Create a histogram collector with the given configuration
    pub fn create_histogram_collector(config: HistogramConfig) -> HistogramCollector {
        HistogramCollector::new(config)
    }

    /// Create a streaming processor
    pub fn create_streaming_processor(config: StreamingConfig) -> StreamingProcessor {
        StreamingProcessor::new(config)
    }

    /// Create a representative sampler with the given strategy
    pub fn create_sampler(strategy: SamplingStrategy) -> Box<dyn RepresentativeSampler> {
        match strategy {
            SamplingStrategy::Random => Box::new(RandomSampler::new()),
            SamplingStrategy::Stratified => Box::new(StratifiedSampler::new()),
            SamplingStrategy::Importance => Box::new(ImportanceSampler::new()),
            SamplingStrategy::Systematic => Box::new(RandomSampler::new()), // Fallback for now
            SamplingStrategy::Custom(_) => Box::new(RandomSampler::new()),  // Fallback for now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_factory_creation() {
        let config = CalibrationConfig::default();
        let dataset = CalibrationFactory::create_dataset(config);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_statistics_collector_creation() {
        let collector = CalibrationFactory::create_statistics_collector();
        // Basic functionality test would go here
    }

    #[test]
    fn test_histogram_collector_creation() {
        let config = HistogramConfig::default();
        let collector = CalibrationFactory::create_histogram_collector(config);
        // Basic functionality test would go here
    }
}
