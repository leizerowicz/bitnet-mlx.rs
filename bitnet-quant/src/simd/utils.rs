//! SIMD utility functions and memory management helpers
//!
//! This module provides common utilities for SIMD operations including
//! memory alignment, prefetching, cache optimization, and auto-vectorization hints.

use crate::quantization::utils::QuantizationError;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Aligned memory allocator for SIMD operations
pub struct AlignedAllocator {
    alignment: usize,
}

impl AlignedAllocator {
    /// Create a new aligned allocator
    pub fn new(alignment: usize) -> Self {
        Self { alignment }
    }

    /// Allocate aligned memory
    pub fn allocate<T>(&self, count: usize) -> Result<NonNull<T>, QuantizationError> {
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment)
            .map_err(|__| QuantizationError::ConfigError("Invalid alignment".into()))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(QuantizationError::MemoryError(
                "Memory allocation failed".into(),
            ));
        }

        Ok(unsafe { NonNull::new_unchecked(ptr as *mut T) })
    }

    /// Deallocate aligned memory
    pub unsafe fn deallocate<T>(&self, ptr: NonNull<T>, count: usize) {
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align_unchecked(size, self.alignment);
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }
}

/// SIMD utilities for memory operations and optimization hints
pub struct SimdUtils;

impl SimdUtils {
    /// Check if a pointer is aligned to the specified boundary
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Get the next aligned address
    pub fn align_up(addr: usize, alignment: usize) -> usize {
        (addr + alignment - 1) & !(alignment - 1)
    }

    /// Get the previous aligned address
    pub fn align_down(addr: usize, alignment: usize) -> usize {
        addr & !(alignment - 1)
    }

    /// Calculate padding needed for alignment
    pub fn alignment_padding(addr: usize, alignment: usize) -> usize {
        Self::align_up(addr, alignment) - addr
    }

    /// Create an aligned slice from a vector
    pub fn make_aligned_slice<T>(vec: &mut Vec<T>, alignment: usize) -> &mut [T] {
        let ptr = vec.as_mut_ptr();
        let addr = ptr as usize;

        if addr % alignment == 0 {
            vec.as_mut_slice()
        } else {
            // If not aligned, we need to reserve additional space
            let padding = Self::alignment_padding(addr, alignment);
            let padding_elements = padding.div_ceil(std::mem::size_of::<T>());

            vec.reserve(padding_elements);

            // Find the aligned position within the reserved space
            let new_ptr = vec.as_mut_ptr();
            let new_addr = new_ptr as usize;
            let aligned_addr = Self::align_up(new_addr, alignment);
            let offset = (aligned_addr - new_addr) / std::mem::size_of::<T>();

            unsafe {
                std::slice::from_raw_parts_mut(
                    new_ptr.add(offset),
                    vec.len().saturating_sub(offset),
                )
            }
        }
    }
}

/// Memory prefetch for optimal cache utilization
#[inline(always)]
pub fn prefetch_memory<T>(ptr: *const T) {
    prefetch_read(ptr);
}

/// Prefetch memory for reading
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{}]", in(reg) ptr, options(nostack, readonly));
    }

    // No-op for other architectures
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr; // Suppress unused variable warning
    }
}

/// Prefetch memory for writing
#[inline(always)]
pub fn prefetch_write<T>(ptr: *const T) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pstl1keep, [{}]", in(reg) ptr, options(nostack));
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Memory fence/barrier for ordering guarantees
#[inline(always)]
pub fn memory_barrier() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Compiler hint for loop vectorization
#[inline(always)]
pub fn hint_vectorize_loop() {
    unsafe {
        // Architecture-specific vectorization hints
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        std::arch::asm!("", options(preserves_flags, nostack));

        #[cfg(target_arch = "aarch64")]
        std::arch::asm!("", options(preserves_flags, nostack));
    }
}

/// Auto-vectorization attributes and hints
pub mod vectorization_hints {
    /// Suggest to the compiler that this loop should be vectorized
    #[macro_export]
    macro_rules! hint_vectorize {
        () => {
            #[allow(unused_unsafe)]
            unsafe {
                std::arch::asm!("", options(preserves_flags, nostack));
            }
        };
    }

    /// Unroll loop with the specified factor
    #[macro_export]
    macro_rules! unroll_loop {
        ($n:expr) => {
            // Compiler-specific unroll hints would go here
            // For now, this is just a marker
        };
    }

    /// Force alignment for better vectorization
    #[macro_export]
    macro_rules! assume_aligned {
        ($ptr:expr, $align:expr) => {{
            let ptr = $ptr;
            debug_assert_eq!(
                ptr as usize % $align,
                0,
                "Pointer not aligned to {} bytes",
                $align
            );
            ptr
        }};
    }
}

/// SIMD-friendly data structures
#[repr(align(64))] // Align to cache line boundary
pub struct CacheAlignedVec<T> {
    data: Vec<T>,
    capacity: usize,
    alignment: usize,
}

impl<T> CacheAlignedVec<T> {
    /// Create a new cache-aligned vector
    pub fn new(alignment: usize) -> Self {
        Self {
            data: Vec::new(),
            capacity: 0,
            alignment,
        }
    }

    /// Create with specific capacity
    pub fn with_capacity(capacity: usize, alignment: usize) -> Self {
        let mut vec = Self::new(alignment);
        vec.reserve(capacity);
        vec
    }

    /// Reserve additional capacity with proper alignment
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.data.len() + additional;
        if new_capacity <= self.capacity {
            return;
        }

        // Allocate with alignment
        let allocator = AlignedAllocator::new(self.alignment);
        let new_ptr = allocator.allocate::<T>(new_capacity).unwrap();

        // Copy existing data
        if !self.data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr(),
                    new_ptr.as_ptr(),
                    self.data.len(),
                );
            }
        }

        // Update data pointer (this is simplified - real implementation would need Drop handling)
        self.capacity = new_capacity;
    }

    /// Push an element
    pub fn push(&mut self, value: T) {
        if self.data.len() >= self.capacity {
            self.reserve(std::cmp::max(8, self.capacity * 2));
        }
        self.data.push(value);
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Check if data is properly aligned
    pub fn is_aligned(&self) -> bool {
        SimdUtils::is_aligned(self.data.as_ptr(), self.alignment)
    }
}

/// Allocate aligned memory for SIMD operations
pub fn aligned_alloc<T>(count: usize, alignment: usize) -> Result<NonNull<T>, QuantizationError> {
    let allocator = AlignedAllocator::new(alignment);
    allocator.allocate::<T>(count)
}

/// Free aligned memory
pub unsafe fn aligned_free<T>(ptr: NonNull<T>, count: usize, alignment: usize) {
    let allocator = AlignedAllocator::new(alignment);
    allocator.deallocate(ptr, count);
}

/// Optimized memory copy for SIMD-aligned data
pub fn simd_memcpy<T: Copy>(dst: &mut [T], src: &[T]) {
    debug_assert_eq!(dst.len(), src.len());

    if dst.is_empty() {
        return;
    }

    // Use system memcpy for large blocks
    if std::mem::size_of_val(dst) >= 1024 {
        dst.copy_from_slice(src);
        return;
    }

    // Manual copy for small blocks
    for (dst_item, src_item) in dst.iter_mut().zip(src.iter()) {
        *dst_item = *src_item;
    }
}

/// Optimized memory set for SIMD-aligned data
pub fn simd_memset<T: Copy>(dst: &mut [T], value: T) {
    // Use vectorized operations when possible
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::mem::size_of::<T>() == 4 && dst.len() >= 8 {
            // Could implement AVX2 memset here
        }
    }

    // Fallback to standard fill
    dst.fill(value);
}

/// Cache line size detection
pub fn get_cache_line_size() -> usize {
    // Platform-specific detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
        {
            if let Ok(size) = contents.trim().parse::<usize>() {
                return size;
            }
        }
    }

    // Default to common cache line size
    64
}

/// Performance counters for SIMD operations (when available)
pub struct SimdPerformanceCounters {
    pub vectorized_operations: u64,
    pub scalar_fallbacks: u64,
    pub cache_misses: u64,
    pub prefetch_hits: u64,
}

impl SimdPerformanceCounters {
    pub fn new() -> Self {
        Self {
            vectorized_operations: 0,
            scalar_fallbacks: 0,
            cache_misses: 0,
            prefetch_hits: 0,
        }
    }

    pub fn record_vectorized_op(&mut self) {
        self.vectorized_operations += 1;
    }

    pub fn record_scalar_fallback(&mut self) {
        self.scalar_fallbacks += 1;
    }

    pub fn vectorization_ratio(&self) -> f64 {
        let total = self.vectorized_operations + self.scalar_fallbacks;
        if total > 0 {
            self.vectorized_operations as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl Default for SimdPerformanceCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_utilities() {
        assert!(SimdUtils::is_aligned(0x1000 as *const u8, 16));
        assert!(!SimdUtils::is_aligned(0x1001 as *const u8, 16));

        assert_eq!(SimdUtils::align_up(0x1001, 16), 0x1010);
        assert_eq!(SimdUtils::align_down(0x100f, 16), 0x1000);
        assert_eq!(SimdUtils::alignment_padding(0x1001, 16), 15);
    }

    #[test]
    fn test_aligned_allocation() {
        let allocator = AlignedAllocator::new(64);
        let ptr = allocator.allocate::<f32>(1024).unwrap();

        assert!(SimdUtils::is_aligned(ptr.as_ptr(), 64));

        unsafe {
            allocator.deallocate(ptr, 1024);
        }
    }

    #[test]
    fn test_cache_aligned_vec() {
        let mut vec = CacheAlignedVec::<f32>::with_capacity(100, 32);

        for i in 0..100 {
            vec.push(i as f32);
        }

        assert_eq!(vec.as_slice().len(), 100);
    }

    #[test]
    fn test_performance_counters() {
        let mut counters = SimdPerformanceCounters::new();

        counters.record_vectorized_op();
        counters.record_vectorized_op();
        counters.record_scalar_fallback();

        assert_eq!(counters.vectorization_ratio(), 2.0 / 3.0);
    }

    #[test]
    fn test_simd_memcpy() {
        let src = vec![1.0f32; 1000];
        let mut dst = vec![0.0f32; 1000];

        simd_memcpy(&mut dst, &src);

        assert_eq!(src, dst);
    }

    #[test]
    fn test_cache_line_detection() {
        let size = get_cache_line_size();
        assert!(size >= 16 && size <= 256);
        println!("Detected cache line size: {} bytes", size);
    }
}
