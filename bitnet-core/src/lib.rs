//! BitNet Core Library
//!
//! This crate provides the core functionality for BitNet implementation,
//! including tensor operations, quantization utilities, mixed precision support,
//! and fundamental data structures.

// Allow dead code for work-in-progress implementations  
#![allow(dead_code, unused_variables, unused_imports)]

pub mod device;
pub mod error;
pub mod execution;
pub mod memory;
pub mod mixed_precision;
pub mod sequence;
pub mod tensor;
pub mod tokenizer;

// Test utilities for error handling and performance monitoring
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

// Re-export test utilities when available
#[cfg(any(test, feature = "test-utils"))]
pub use test_utils::*;

// MLX support (Apple Silicon only)
#[cfg(feature = "mlx")]
pub mod mlx;

// Re-export Metal functionality when available
#[cfg(feature = "metal")]
pub use bitnet_metal as metal;

pub use device::*;
pub use error::*;
pub use execution::*;
// Import memory types but not tensor types to avoid conflicts
#[cfg(feature = "metal")]
pub use memory::MetalMemoryPool;
pub use memory::{
    AllocationTimeline, BatchConverter, CleanupConfig, CleanupId, CleanupManager, CleanupMetrics,
    CleanupOperationMetrics, CleanupPriority, CleanupResult, CleanupScheduler, CleanupStrategy,
    CleanupStrategyType, CompactionResult, ConversionConfig, ConversionEngine, ConversionEvent,
    ConversionMetrics, ConversionPipeline, ConversionStats, CpuCleanup, CpuMemoryPool,
    DetailedMemoryMetrics, DeviceCleanupOps, HybridMemoryPool, InPlaceConverter, LargeBlockPool,
    LeakReport, MemoryError, MemoryHandle, MemoryMetrics, MemoryPoolConfig, MemoryPressureDetector,
    MemoryPressureLevel, MemoryProfiler, MemoryResult, MemoryTracker, MetalCleanup,
    PatternAnalyzer, PressureCallback, PressureThresholds, ProfilingReport, SmallBlockPool,
    StreamingConverter, TensorHandle, TensorMetadata, TrackingConfig, TrackingLevel,
    ZeroCopyConverter,
};

// Export mixed precision with explicit validation aliasing  
pub use mixed_precision::{
    ComponentPrecisionConfig, ComponentType, LayerPrecisionConfig, LayerPrecisionManager,
    LayerType, MixedPrecisionConfig, MixedPrecisionError, MixedPrecisionStrategy, PolicyEngine,
    PrecisionConverter, PrecisionManager, PrecisionPolicy, PrecisionValidator,
    validation as mixed_precision_validation,
};

// Export sequence types with explicit validation aliasing
pub use sequence::{
    PaddingStrategy, SequenceBatch, SequenceManager,
    validation as sequence_validation,
};

// Use tensor types as primary exports
pub use tensor::*;
pub use tokenizer::*;

// MLX re-exports when feature is enabled
#[cfg(feature = "mlx")]
pub use mlx::*;

// Re-export commonly used types from candle
pub use candle_core::{DType, Device, Result, Tensor};

// Re-export BitNet tensor types for convenience
pub use tensor::{BitNetDType, BitNetTensor, TensorShape, TensorStorage};

// MLX types when available
#[cfg(feature = "mlx")]
pub use mlx_rs::{Array as MlxArray, Device as MlxDevice};
