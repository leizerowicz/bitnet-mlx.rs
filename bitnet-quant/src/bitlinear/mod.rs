//! BitLinear Layer Module
//!
//! This module implements the BitLinear layer, the core building block of BitNet models.
//! BitLinear layers perform matrix multiplication using quantized weights while maintaining
//! full-precision weights for training.

pub mod layer;
pub mod forward;
pub mod cache;
pub mod memory;
pub mod error;

pub use layer::{BitLinear, BitLinearConfig};
pub use forward::BitLinearForward;
pub use cache::{QuantizedWeightCache, CacheEntry, CacheConfig};
pub use error::{BitLinearError, BitLinearResult};
pub use memory::{
    BitLinearMemoryOptimizer, MemoryOptimizationConfig, MemoryOptimizationMetrics,
    LazyQuantizer, LazyQuantizationConfig, QuantizationState,
    WeightCacheManager, ScalingFactorManager, ScalingPolicy,
    CacheFriendlyTensor, MemoryLayout, AccessPattern,
    MemoryPressureIntegrator, MemoryPressureLevel, PressureConfig,
};
