//! SIMD-optimized kernels for BitNet operations
//!
//! This module contains platform-specific kernel implementations that match
//! Microsoft's performance targets through lookup table optimization.

pub mod tl1_arm64;
pub mod tl2_x86_64; 
pub mod i2s_optimized;
pub mod generic;

// Re-export key types for convenience
pub use tl1_arm64::Tl1Arm64Kernel;
pub use tl2_x86_64::Tl2X86_64Kernel;
pub use i2s_optimized::{I2SArmKernel, I2SX86Kernel};
pub use generic::{GenericTernaryKernel, GenericI2SKernel};