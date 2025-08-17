//! High-performance SIMD operations for BitNet quantization
//!
//! This module provides vectorized implementations of core BitNet operations
//! including ternary quantization, matrix multiplication, and data packing/unpacking.
//! 
//! ## Supported Architectures
//! - x86/x86_64: SSE2, SSE4.1, AVX2, AVX512F
//! - ARM/AArch64: NEON, SVE
//! - Automatic fallback to scalar implementations
//!
//! ## Features
//! - Vectorized ternary operations (-1, 0, +1)
//! - Optimized quantized matrix multiplication
//! - Parallel weight packing/unpacking
//! - Auto-vectorization hints for compiler optimizations
//! - Memory-aligned operations for cache efficiency

pub mod capabilities;
pub mod ternary;
pub mod matrix;
pub mod packing;
pub mod utils;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use capabilities::{SimdCapabilities, detect_simd_capabilities};
pub use ternary::{SimdTernaryOps, vectorized_ternary_quantize, vectorized_ternary_dequantize};
pub use matrix::{SimdMatrixOps, vectorized_matrix_multiply, vectorized_gemm};
pub use packing::{SimdPackingOps, vectorized_pack_ternary, vectorized_unpack_ternary};
pub use utils::{SimdUtils, aligned_alloc, prefetch_memory, memory_barrier};

use crate::quantization::utils::QuantizationError;
use std::alloc::{alloc, dealloc, Layout};

/// Result type for SIMD operations
pub type SimdResult<T> = Result<T, QuantizationError>;

/// Alignment requirements for SIMD operations
pub const SIMD_ALIGNMENT: usize = 64; // AVX512/cache line alignment
pub const AVX2_ALIGNMENT: usize = 32;
pub const SSE_ALIGNMENT: usize = 16;
pub const NEON_ALIGNMENT: usize = 16;

/// Auto-vectorization hints for the compiler
#[inline(always)]
pub fn hint_vectorize() {
    unsafe {
        std::arch::asm!("", options(preserves_flags, nostack));
    }
}

/// SIMD-aligned memory allocator
pub struct SimdAllocator {
    alignment: usize,
}

impl SimdAllocator {
    /// Create a new SIMD allocator with the specified alignment
    pub fn new(alignment: usize) -> Self {
        Self { alignment }
    }
    
    /// Create an allocator with optimal alignment for the current architecture
    pub fn optimal() -> Self {
        let caps = detect_simd_capabilities();
        let alignment = if caps.avx512f {
            64
        } else if caps.avx2 {
            32
        } else {
            16
        };
        Self::new(alignment)
    }
    
    /// Allocate aligned memory for SIMD operations
    pub fn allocate<T>(&self, count: usize) -> SimdResult<*mut T> {
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment)
            .map_err(|_| QuantizationError::ConfigError("Invalid SIMD alignment".into()))?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(QuantizationError::MemoryError("SIMD memory allocation failed".into()));
        }
        
        Ok(ptr as *mut T)
    }
    
    /// Deallocate SIMD-aligned memory
    pub unsafe fn deallocate<T>(&self, ptr: *mut T, count: usize) {
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align_unchecked(size, self.alignment);
        dealloc(ptr as *mut u8, layout);
    }
}

/// SIMD-aligned vector for optimal performance
#[repr(align(64))]
pub struct AlignedVec<T> {
    data: Vec<T>,
    alignment: usize,
}

impl<T> AlignedVec<T> {
    /// Create a new aligned vector with the specified capacity
    pub fn with_capacity(capacity: usize, alignment: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        // Ensure vector is properly aligned
        while vec.as_ptr() as usize % alignment != 0 {
            vec.reserve(1);
        }
        
        Self {
            data: vec,
            alignment,
        }
    }
    
    /// Create an optimally aligned vector for the current architecture
    pub fn optimal_with_capacity(capacity: usize) -> Self {
        let caps = detect_simd_capabilities();
        let alignment = if caps.avx512f {
            64
        } else if caps.avx2 {
            32
        } else {
            16
        };
        Self::with_capacity(capacity, alignment)
    }
    
    /// Get a pointer to the aligned data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Get a mutable pointer to the aligned data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    
    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Push an element to the vector
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    
    /// Get the alignment of this vector
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Verify that the vector is properly aligned
    pub fn is_aligned(&self) -> bool {
        self.as_ptr() as usize % self.alignment == 0
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable SSE optimizations (x86/x86_64)
    pub enable_sse: bool,
    /// Enable AVX optimizations (x86/x86_64)
    pub enable_avx: bool,
    /// Enable AVX2 optimizations (x86/x86_64)
    pub enable_avx2: bool,
    /// Enable AVX-512 optimizations (x86/x86_64)
    pub enable_avx512: bool,
    /// Enable NEON optimizations (ARM/AArch64)
    pub enable_neon: bool,
    /// Enable SVE optimizations (ARM/AArch64)
    pub enable_sve: bool,
    /// Preferred vector size for operations
    pub preferred_vector_size: usize,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
    /// Enable automatic loop unrolling
    pub enable_unrolling: bool,
    /// Unroll factor for loops
    pub unroll_factor: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        let caps = detect_simd_capabilities();
        
        Self {
            enable_sse: caps.sse2,
            enable_avx: caps.avx,
            enable_avx2: caps.avx2,
            enable_avx512: caps.avx512f,
            enable_neon: caps.neon,
            enable_sve: caps.sve,
            preferred_vector_size: if caps.avx512f {
                64
            } else if caps.avx2 {
                32
            } else {
                16
            },
            memory_alignment: SIMD_ALIGNMENT,
            enable_unrolling: true,
            unroll_factor: 4,
        }
    }
}

impl SimdConfig {
    /// Create a new SIMD configuration with optimal settings
    pub fn optimal() -> Self {
        Self::default()
    }
    
    /// Create a configuration with SIMD disabled (scalar fallback)
    pub fn scalar_only() -> Self {
        Self {
            enable_sse: false,
            enable_avx: false,
            enable_avx2: false,
            enable_avx512: false,
            enable_neon: false,
            enable_sve: false,
            preferred_vector_size: 1,
            memory_alignment: std::mem::align_of::<f32>(),
            enable_unrolling: false,
            unroll_factor: 1,
        }
    }
    
    /// Check if any SIMD optimizations are enabled
    pub fn has_simd(&self) -> bool {
        self.enable_sse || self.enable_avx || self.enable_avx2 || 
        self.enable_avx512 || self.enable_neon || self.enable_sve
    }
}

#[cfg(test)]
mod tests;
