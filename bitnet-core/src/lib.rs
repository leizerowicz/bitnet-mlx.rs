//! BitNet Core Library
//!
//! This crate provides the core functionality for BitNet implementation,
//! including tensor operations, quantization utilities, mixed precision support,
//! and fundamental data structures.

pub mod device;
pub mod error;
pub mod execution;
pub mod memory;
pub mod mixed_precision;
pub mod sequence;
pub mod tensor;
pub mod tokenizer;

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
pub use memory::{
    HybridMemoryPool, MemoryPoolConfig, MemoryHandle, MemoryMetrics,
    SmallBlockPool, LargeBlockPool, CpuMemoryPool, 
    TensorHandle, TensorMetadata,
    ConversionEngine, ConversionConfig, ConversionMetrics, ConversionStats, ConversionEvent,
    ZeroCopyConverter, StreamingConverter, InPlaceConverter, BatchConverter, ConversionPipeline,
    MemoryTracker, DetailedMemoryMetrics, MemoryPressureDetector, MemoryPressureLevel,
    MemoryProfiler, ProfilingReport, LeakReport, AllocationTimeline, PatternAnalyzer,
    TrackingConfig, TrackingLevel, PressureThresholds, PressureCallback,
    CleanupManager, CleanupConfig, CleanupResult, CompactionResult, CleanupStrategy,
    CleanupStrategyType, CleanupPriority, CleanupScheduler, CleanupId, CleanupMetrics,
    CleanupOperationMetrics, CpuCleanup, MetalCleanup, DeviceCleanupOps,
    MemoryResult, MemoryError
};
#[cfg(feature = "metal")]
pub use memory::MetalMemoryPool;

pub use mixed_precision::*;
pub use sequence::*;
// Use tensor types as primary exports
pub use tensor::*;
pub use tokenizer::*;

// MLX re-exports when feature is enabled
#[cfg(feature = "mlx")]
pub use mlx::*;

// Re-export commonly used types from candle
pub use candle_core::{Device, DType, Result, Tensor};

// Re-export BitNet tensor types for convenience
pub use tensor::{BitNetTensor, BitNetDType, TensorShape, TensorStorage};

// Re-export mixed precision types for convenience
pub use mixed_precision::{
    MixedPrecisionConfig, LayerPrecisionConfig, ComponentPrecisionConfig,
    LayerType, ComponentType, MixedPrecisionStrategy, MixedPrecisionError,
    PrecisionManager, PrecisionConverter, LayerPrecisionManager,
    PrecisionValidator, PolicyEngine, PrecisionPolicy,
};

// MLX types when available
#[cfg(feature = "mlx")]
pub use mlx_rs::{Array as MlxArray, Device as MlxDevice};