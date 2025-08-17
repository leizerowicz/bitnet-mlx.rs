//! BitLinear Layer Module
//!
//! This module implements the BitLinear layer, the core building block of BitNet models.
//! BitLinear layers perform matrix multiplication using quantized weights while maintaining
//! full-precision weights for training.

pub mod layer;
pub mod forward;
pub mod cache;

pub use layer::{BitLinear, BitLinearConfig, BitLinearError, BitLinearResult};
pub use forward::BitLinearForward;
pub use cache::{QuantizedWeightCache, CacheEntry, CacheConfig};
