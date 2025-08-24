//! BitLinear Layer Module
//!
//! This module implements the BitLinear layer, the core building block of BitNet models.
//! BitLinear layers perform matrix multiplication using quantized weights while maintaining
//! full-precision weights for training.

pub mod cache;
pub mod error;
pub mod forward;
pub mod layer;
pub mod memory;

pub use cache::{CacheConfig, CacheEntry, QuantizedWeightCache};
pub use error::{BitLinearError, BitLinearResult};
pub use forward::BitLinearForward;
pub use layer::{BitLinear, BitLinearConfig};
pub use memory::{
    AccessPattern, BitLinearMemoryOptimizer, CacheFriendlyTensor, LazyQuantizationConfig,
    LazyQuantizer, MemoryLayout, MemoryOptimizationConfig, MemoryOptimizationMetrics,
    MemoryPressureIntegrator, MemoryPressureLevel, PressureConfig, QuantizationState,
    ScalingFactorManager, ScalingPolicy, WeightCacheManager,
};
