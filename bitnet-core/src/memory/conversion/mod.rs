//! Memory-Efficient Data Conversion System
//!
//! This module provides a comprehensive data conversion system optimized for memory efficiency
//! in BitNet operations. It includes zero-copy conversions, streaming operations, in-place
//! transformations, and batch processing capabilities.
//!
//! # Architecture
//!
//! The conversion system consists of several key components:
//!
//! - **ConversionEngine**: Main interface for all conversion operations
//! - **ZeroCopyConverter**: Handles conversions that don't require data copying
//! - **StreamingConverter**: Processes large tensors in chunks to minimize memory usage
//! - **InPlaceConverter**: Performs conversions directly in existing memory buffers
//! - **BatchConverter**: Efficiently processes multiple tensors together
//! - **ConversionPipeline**: Chains multiple conversion operations with memory pooling
//! - **ConversionMetrics**: Tracks performance and memory usage statistics
//!
//! # Features
//!
//! - Zero-copy conversions for compatible data types
//! - Streaming conversion for large tensors that don't fit in memory
//! - In-place conversions to minimize memory allocation
//! - Batch processing for multiple tensors
//! - Memory pooling integration for efficient allocation
//! - Comprehensive metrics and monitoring
//! - Device-aware conversions (CPU ↔ Metal ↔ MLX)
//! - Thread-safe operations
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::memory::conversion::{ConversionEngine, ConversionConfig};
//! use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
//! use bitnet_core::memory::HybridMemoryPool;
//! use bitnet_core::device::auto_select_device;
//!
//! // Create conversion engine
//! let pool = HybridMemoryPool::new()?;
//! let config = ConversionConfig::default();
//! let engine = ConversionEngine::new(config, &pool)?;
//!
//! // Zero-copy conversion (F32 to F32)
//! let tensor = BitNetTensor::zeros(&[1024, 1024], BitNetDType::F32, &device, &pool)?;
//! let converted = engine.zero_copy_convert(&tensor, BitNetDType::F32)?;
//!
//! // Streaming conversion for large tensors
//! let large_tensor = BitNetTensor::zeros(&[10000, 10000], BitNetDType::F32, &device, &pool)?;
//! let quantized = engine.streaming_convert(&large_tensor, BitNetDType::I8, 1024)?;
//!
//! // In-place conversion
//! let mut tensor = BitNetTensor::ones(&[512, 512], BitNetDType::F32, &device, &pool)?;
//! engine.in_place_convert(&mut tensor, BitNetDType::F16)?;
//!
//! // Batch conversion
//! let tensors = vec![tensor1, tensor2, tensor3];
//! let converted_tensors = engine.batch_convert(&tensors, BitNetDType::I4)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod batch;
pub mod config;
pub mod engine;
pub mod in_place;
pub mod metrics;
pub mod pipeline;
pub mod streaming;
pub mod zero_copy;

// Re-exports
pub use batch::BatchConverter;
pub use config::{BatchConfig, ConversionConfig, PerformanceConfig, StreamingConfig};
pub use engine::{ConversionEngine, ConversionStrategyInfo};
pub use in_place::InPlaceConverter;
pub use metrics::{
    ConversionEvent, ConversionMetrics, ConversionStats, DTypeConversion, DTypeMetrics,
    DeviceMetrics, ErrorStats, MemoryStats, PerformanceStats, StrategyMetrics,
};
pub use pipeline::{ConversionPipeline, PipelineStats};
pub use streaming::StreamingConverter;
pub use zero_copy::{TensorView, ZeroCopyConverter};

use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::{HybridMemoryPool, MemoryError};
use candle_core::Device;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during data conversion operations
#[derive(Error, Debug)]
pub enum ConversionError {
    /// Memory allocation error during conversion
    #[error("Memory allocation failed during conversion: {0}")]
    MemoryError(#[from] MemoryError),

    /// Unsupported conversion between data types
    #[error("Unsupported conversion from {from} to {to}")]
    UnsupportedConversion { from: BitNetDType, to: BitNetDType },

    /// Device mismatch during conversion
    #[error("Device mismatch during conversion")]
    DeviceMismatch,

    /// Shape mismatch during conversion
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Conversion would result in data loss
    #[error("Conversion from {from} to {to} would result in significant data loss")]
    DataLossError { from: BitNetDType, to: BitNetDType },

    /// Streaming conversion error
    #[error("Streaming conversion failed: {reason}")]
    StreamingError { reason: String },

    /// Batch conversion error
    #[error("Batch conversion failed: {reason}")]
    BatchError { reason: String },

    /// Pipeline execution error
    #[error("Conversion pipeline failed at stage {stage}: {reason}")]
    PipelineError { stage: usize, reason: String },

    /// Configuration error
    #[error("Invalid conversion configuration: {reason}")]
    ConfigError { reason: String },

    /// Internal conversion error
    #[error("Internal conversion error: {reason}")]
    InternalError { reason: String },
}

/// Result type for conversion operations
pub type ConversionResult<T> = std::result::Result<T, ConversionError>;

/// Conversion strategy for different scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ConversionStrategy {
    /// Zero-copy conversion (no data copying)
    ZeroCopy,
    /// In-place conversion (modify existing buffer)
    InPlace,
    /// Streaming conversion (process in chunks)
    Streaming,
    /// Standard conversion (allocate new buffer)
    Standard,
    /// Automatic strategy selection based on tensor properties
    Auto,
}

/// Conversion quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ConversionQuality {
    /// Fastest conversion, may sacrifice precision
    Fast,
    /// Balanced speed and precision
    Balanced,
    /// Highest precision, may be slower
    Precise,
}

/// Conversion context containing metadata about the operation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConversionContext {
    /// Source data type
    pub sourcedtype: BitNetDType,
    /// Target data type
    pub targetdtype: BitNetDType,
    /// Source device
    pub source_device: Device,
    /// Target device
    pub target_device: Device,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Conversion strategy
    pub strategy: ConversionStrategy,
    /// Quality setting
    pub quality: ConversionQuality,
    /// Whether to preserve metadata
    pub preserve_metadata: bool,
}

impl ConversionContext {
    /// Creates a new conversion context
    pub fn new(
        sourcedtype: BitNetDType,
        targetdtype: BitNetDType,
        source_device: Device,
        target_device: Device,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            sourcedtype,
            targetdtype,
            source_device,
            target_device,
            shape,
            strategy: ConversionStrategy::Auto,
            quality: ConversionQuality::Balanced,
            preserve_metadata: true,
        }
    }

    /// Sets the conversion strategy
    pub fn with_strategy(mut self, strategy: ConversionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the conversion quality
    pub fn with_quality(mut self, quality: ConversionQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Sets whether to preserve metadata
    pub fn with_preserve_metadata(mut self, preserve: bool) -> Self {
        self.preserve_metadata = preserve;
        self
    }

    /// Returns true if this is a zero-copy compatible conversion
    pub fn is_zero_copy_compatible(&self) -> bool {
        // Zero-copy is possible when:
        // 1. Same data type and device
        // 2. Compatible data types with same memory layout
        if self.sourcedtype == self.targetdtype
            && std::mem::discriminant(&self.source_device)
                == std::mem::discriminant(&self.target_device)
        {
            return true;
        }

        // Check for compatible data types with same bit width
        match (self.sourcedtype, self.targetdtype) {
            (BitNetDType::F16, BitNetDType::BF16) | (BitNetDType::BF16, BitNetDType::F16) => true,
            _ => false,
        }
    }

    /// Returns true if in-place conversion is possible
    pub fn is_in_place_compatible(&self) -> bool {
        // In-place conversion is possible when:
        // 1. Same device
        // 2. Target type has same or smaller memory footprint
        if std::mem::discriminant(&self.source_device)
            != std::mem::discriminant(&self.target_device)
        {
            return false;
        }

        let source_bits = self.sourcedtype.bits_per_element();
        let target_bits = self.targetdtype.bits_per_element();
        target_bits <= source_bits
    }

    /// Returns the optimal conversion strategy for this context
    pub fn optimal_strategy(&self) -> ConversionStrategy {
        if self.strategy != ConversionStrategy::Auto {
            return self.strategy;
        }

        if self.is_zero_copy_compatible() {
            ConversionStrategy::ZeroCopy
        } else if self.is_in_place_compatible() {
            let element_count: usize = self.shape.iter().product();
            let size_bytes = self.sourcedtype.bytes_for_elements(element_count);
            
            // Use streaming for large tensors (> 100MB) even if in-place compatible
            if size_bytes > 100 * 1024 * 1024 {
                ConversionStrategy::Streaming
            } else {
                ConversionStrategy::InPlace
            }
        } else {
            let element_count: usize = self.shape.iter().product();
            let size_bytes = self.sourcedtype.bytes_for_elements(element_count);

            // Use streaming for large tensors (> 100MB)
            if size_bytes > 100 * 1024 * 1024 {
                ConversionStrategy::Streaming
            } else {
                ConversionStrategy::Standard
            }
        }
    }

    /// Estimates the memory overhead for this conversion
    pub fn memory_overhead_bytes(&self) -> usize {
        let element_count: usize = self.shape.iter().product();
        let source_bytes = self.sourcedtype.bytes_for_elements(element_count);
        let target_bytes = self.targetdtype.bytes_for_elements(element_count);

        match self.optimal_strategy() {
            ConversionStrategy::ZeroCopy => 0,
            ConversionStrategy::InPlace => 0,
            ConversionStrategy::Streaming => {
                // Streaming uses chunk-sized buffers
                let chunk_size = 1024 * 1024; // 1MB chunks
                std::cmp::min(source_bytes, chunk_size) + std::cmp::min(target_bytes, chunk_size)
            }
            ConversionStrategy::Standard => target_bytes,
            ConversionStrategy::Auto => unreachable!(),
        }
    }
}

/// Trait for conversion operations
pub trait Converter {
    /// Performs the conversion operation
    fn convert(
        &self,
        source: &BitNetTensor,
        context: &ConversionContext,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor>;

    /// Returns true if this converter supports the given context
    fn supports(&self, context: &ConversionContext) -> bool;

    /// Estimates the conversion time in milliseconds
    fn estimate_time_ms(&self, context: &ConversionContext) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_conversion_context_creation() {
        let device = get_cpu_device();
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F16,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );

        assert_eq!(context.sourcedtype, BitNetDType::F32);
        assert_eq!(context.targetdtype, BitNetDType::F16);
        assert_eq!(context.shape, vec![2, 3]);
        assert_eq!(context.strategy, ConversionStrategy::Auto);
        assert_eq!(context.quality, ConversionQuality::Balanced);
        assert!(context.preserve_metadata);
    }

    #[test]
    fn test_zero_copy_compatibility() {
        let device = get_cpu_device();

        // Same type should be zero-copy compatible
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(context.is_zero_copy_compatible());

        // F16 <-> BF16 should be zero-copy compatible
        let context = ConversionContext::new(
            BitNetDType::F16,
            BitNetDType::BF16,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(context.is_zero_copy_compatible());

        // Different types should not be zero-copy compatible
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::I8,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(!context.is_zero_copy_compatible());
    }

    #[test]
    fn test_in_place_compatibility() {
        let device = get_cpu_device();

        // Same size should be in-place compatible
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(context.is_in_place_compatible());

        // Smaller target should be in-place compatible
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F16,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(context.is_in_place_compatible());

        // Larger target should not be in-place compatible
        let context = ConversionContext::new(
            BitNetDType::F16,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert!(!context.is_in_place_compatible());
    }

    #[test]
    fn test_optimal_strategy_selection() {
        let device = get_cpu_device();

        // Zero-copy case
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert_eq!(context.optimal_strategy(), ConversionStrategy::ZeroCopy);

        // In-place case
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F16,
            device.clone(),
            device.clone(),
            vec![2, 3],
        );
        assert_eq!(context.optimal_strategy(), ConversionStrategy::InPlace);

        // Large tensor streaming case
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::I8,
            device.clone(),
            device.clone(),
            vec![10000, 10000], // Large tensor
        );
        assert_eq!(context.optimal_strategy(), ConversionStrategy::Streaming);
    }

    #[test]
    fn test_memory_overhead_estimation() {
        let device = get_cpu_device();

        // Zero-copy should have no overhead
        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![100, 100],
        );
        assert_eq!(context.memory_overhead_bytes(), 0);

        // Standard conversion should have target size overhead
        let context = ConversionContext::new(
            BitNetDType::F16,
            BitNetDType::F32,
            device.clone(),
            device.clone(),
            vec![100, 100],
        );
        let expected_overhead = BitNetDType::F32.bytes_for_elements(10000);
        assert_eq!(context.memory_overhead_bytes(), expected_overhead);
    }
}
