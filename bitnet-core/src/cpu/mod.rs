//! CPU-specific optimization module for BitNet
//! 
//! This module provides high-performance CPU implementations for BitNet operations,
//! including SIMD-optimized kernels that match Microsoft's performance targets.
//!
//! # Performance Targets
//! 
//! - ARM64 NEON: 1.37x-6.17x speedup over reference implementation
//! - x86_64 AVX2/AVX-512: 2.5x-8.0x speedup with vectorized lookup tables
//! - I2_S kernels: Optimized memory bandwidth utilization for large models
//!
//! # Architecture Support
//!
//! - **ARM64**: NEON vectorization with cache hierarchy optimization
//! - **x86_64**: AVX2/AVX-512 support with automatic feature detection
//! - **Fallback**: Portable implementations for unsupported architectures

use anyhow::Result;

pub mod kernels;
pub mod performance_validator;
pub mod optimizer;
pub mod feature_detector;
pub mod production_validator;

/// CPU architecture detection and feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    /// ARM64 with NEON support
    Arm64Neon,
    /// x86_64 with AVX2 support
    X86_64Avx2,
    /// x86_64 with AVX-512 support
    X86_64Avx512,
    /// Generic fallback implementation
    Generic,
}

/// CPU feature detection
pub fn detect_cpu_features() -> CpuArch {
    let mut detector = feature_detector::CpuFeatureDetector::new();
    detector.get_optimal_kernel_arch().unwrap_or(CpuArch::Generic)
}

/// Check if NEON is available on ARM64
#[cfg(target_arch = "aarch64")]
fn is_neon_available() -> bool {
    // NEON is standard on all ARM64 targets we support
    true
}

/// Check if AVX2 is available on x86_64
#[cfg(target_arch = "x86_64")]
fn is_avx2_available() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        // Runtime detection would go here
        // For now, assume available on modern x86_64
        std::arch::is_x86_feature_detected!("avx2")
    }
}

/// Check if AVX-512 is available on x86_64
#[cfg(target_arch = "x86_64")]
fn is_avx512_available() -> bool {
    #[cfg(target_feature = "avx512f")]
    {
        true
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        // Runtime detection for AVX-512
        std::arch::is_x86_feature_detected!("avx512f")
    }
}

/// Kernel selection strategy based on detected CPU features
pub struct KernelSelector {
    arch: CpuArch,
}

impl KernelSelector {
    /// Create a new kernel selector with automatic feature detection
    pub fn new() -> Self {
        Self {
            arch: detect_cpu_features(),
        }
    }
    
    /// Create a kernel selector with specific architecture
    pub fn with_arch(arch: CpuArch) -> Self {
        Self { arch }
    }
    
    /// Get the detected or specified CPU architecture
    pub fn arch(&self) -> CpuArch {
        self.arch
    }
    
    /// Select the optimal ternary lookup kernel for this architecture
    pub fn select_ternary_kernel(&self) -> Box<dyn TernaryLookupKernel> {
        match self.arch {
            CpuArch::Arm64Neon => Box::new(kernels::tl1_arm64::Tl1Arm64Kernel::new()),
            CpuArch::X86_64Avx2 => Box::new(kernels::tl2_x86_64::Tl2X86_64Kernel::new_avx2()),
            CpuArch::X86_64Avx512 => Box::new(kernels::tl2_x86_64::Tl2X86_64Kernel::new_avx512()),
            CpuArch::Generic => Box::new(kernels::generic::GenericTernaryKernel::new()),
        }
    }
    
    /// Select the optimal I2_S kernel for this architecture
    pub fn select_i2s_kernel(&self) -> Box<dyn I2SLookupKernel> {
        match self.arch {
            CpuArch::Arm64Neon => Box::new(kernels::i2s_optimized::I2SArmKernel::new()),
            CpuArch::X86_64Avx2 => Box::new(kernels::i2s_optimized::I2SX86Kernel::new_avx2()),
            CpuArch::X86_64Avx512 => Box::new(kernels::i2s_optimized::I2SX86Kernel::new_avx512()),
            CpuArch::Generic => Box::new(kernels::generic::GenericI2SKernel::new()),
        }
    }
}

impl Default for KernelSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for ternary lookup table kernels
pub trait TernaryLookupKernel: Send + Sync {
    /// Perform ternary lookup table computation
    /// 
    /// # Arguments
    /// * `weights` - Ternary weights ({-1, 0, 1})
    /// * `inputs` - Input activations
    /// * `output` - Output buffer
    /// 
    /// # Returns
    /// Result indicating success or failure
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()>;
    
    /// Get the kernel name for debugging and benchmarking
    fn name(&self) -> &'static str;
    
    /// Get optimal batch size for this kernel
    fn optimal_batch_size(&self) -> usize;
}

/// Trait for I2_S (signed 2-bit) lookup kernels
pub trait I2SLookupKernel: Send + Sync {
    /// Perform signed 2-bit lookup table computation
    /// 
    /// # Arguments
    /// * `weights` - Signed 2-bit weights ({-2, -1, 0, 1})
    /// * `inputs` - Input activations
    /// * `output` - Output buffer
    /// 
    /// # Returns
    /// Result indicating success or failure
    fn compute(&self, weights: &[i8], inputs: &[f32], output: &mut [f32]) -> Result<()>;
    
    /// Get the kernel name for debugging and benchmarking
    fn name(&self) -> &'static str;
    
    /// Get optimal batch size for this kernel
    fn optimal_batch_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let arch = detect_cpu_features();
        // Should detect a valid architecture
        match arch {
            CpuArch::Arm64Neon | CpuArch::X86_64Avx2 | CpuArch::X86_64Avx512 | CpuArch::Generic => {
                // Valid architecture detected
            }
        }
    }

    #[test]
    fn test_kernel_selector() {
        let selector = KernelSelector::new();
        
        // Should be able to select kernels
        let _ternary_kernel = selector.select_ternary_kernel();
        let _i2s_kernel = selector.select_i2s_kernel();
    }

    #[test]
    fn test_specific_arch_selector() {
        let selector = KernelSelector::with_arch(CpuArch::Generic);
        assert_eq!(selector.arch(), CpuArch::Generic);
    }
}