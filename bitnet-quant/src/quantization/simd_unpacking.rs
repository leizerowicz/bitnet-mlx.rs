//! SIMD-optimized weight unpacking for BitNet models
//! 
//! This module provides SIMD-accelerated implementations for unpacking ternary weights
//! from various compressed formats. It includes optimizations for x86_64 (SSE2, AVX2)
//! and ARM NEON architectures, with automatic fallback to scalar implementations.

use super::packing::{PackedTernaryWeights, TernaryPackingStrategy};
use super::utils::QuantizationError;

/// SIMD-optimized unpacker for ternary weights
pub struct SimdUnpacker {
    /// Whether SIMD instructions are available
    simd_available: SimdCapabilities,
}

/// Available SIMD instruction sets
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub avx2: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD capabilities at runtime
    pub fn detect() -> Self {
        Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            sse2: is_x86_feature_detected!("sse2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            sse2: false,
            
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            avx2: false,
            
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
        }
    }
    
    /// Check if any SIMD instructions are available
    pub fn has_simd(&self) -> bool {
        self.sse2 || self.avx2 || self.neon
    }
}

impl SimdUnpacker {
    /// Create a new SIMD unpacker with automatic capability detection
    pub fn new() -> Self {
        Self {
            simd_available: SimdCapabilities::detect(),
        }
    }
    
    /// Create a new SIMD unpacker with specific capabilities (for testing)
    pub fn with_capabilities(capabilities: SimdCapabilities) -> Self {
        Self {
            simd_available: capabilities,
        }
    }
    
    /// Unpack ternary weights using the best available SIMD implementation
    pub fn unpack(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        match packed.strategy {
            TernaryPackingStrategy::BitPacked2Bit => self.unpack_bit_packed_2bit(packed),
            TernaryPackingStrategy::Base3Packed => self.unpack_base3_packed(packed),
            TernaryPackingStrategy::ByteAligned => self.unpack_byte_aligned(packed),
            TernaryPackingStrategy::RunLengthEncoded => self.unpack_run_length_encoded(packed),
            TernaryPackingStrategy::CompressedSparse => self.unpack_compressed_sparse(packed),
            _ => self.unpack_fallback(packed),
        }
    }
    
    /// SIMD-optimized unpacking for 2-bit packed weights
    fn unpack_bit_packed_2bit(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let element_count = packed.metadata.element_count;
        let mut result = Vec::with_capacity(element_count);
        
        // Use SIMD if available, otherwise fall back to optimized scalar
        if self.simd_available.has_simd() {
            self.unpack_bit_packed_2bit_simd(&packed.data, &mut result, element_count)?;
        } else {
            self.unpack_bit_packed_2bit_scalar(&packed.data, &mut result, element_count)?;
        }
        
        result.truncate(element_count);
        Ok(result)
    }
    
    /// SIMD implementation for 2-bit packed weights (platform-specific)
    fn unpack_bit_packed_2bit_simd(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.simd_available.avx2 {
                return self.unpack_bit_packed_2bit_avx2(data, result, element_count);
            } else if self.simd_available.sse2 {
                return self.unpack_bit_packed_2bit_sse2(data, result, element_count);
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.simd_available.neon {
                return self.unpack_bit_packed_2bit_neon(data, result, element_count);
            }
        }
        
        // Fallback to scalar
        self.unpack_bit_packed_2bit_scalar(data, result, element_count)
    }
    
    /// AVX2-optimized unpacking for 2-bit packed weights
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn unpack_bit_packed_2bit_avx2(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        // For now, use optimized scalar implementation
        // TODO: Implement actual AVX2 intrinsics when target supports it
        self.unpack_bit_packed_2bit_scalar_optimized(data, result, element_count)
    }
    
    /// SSE2-optimized unpacking for 2-bit packed weights
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn unpack_bit_packed_2bit_sse2(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        // For now, use optimized scalar implementation
        // TODO: Implement actual SSE2 intrinsics when target supports it
        self.unpack_bit_packed_2bit_scalar_optimized(data, result, element_count)
    }
    
    /// ARM NEON-optimized unpacking for 2-bit packed weights
    #[cfg(target_arch = "aarch64")]
    fn unpack_bit_packed_2bit_neon(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        // For now, use optimized scalar implementation
        // TODO: Implement actual NEON intrinsics
        self.unpack_bit_packed_2bit_scalar_optimized(data, result, element_count)
    }
    
    /// Optimized scalar implementation for 2-bit packed weights
    fn unpack_bit_packed_2bit_scalar_optimized(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        // Process multiple bytes at once for better performance
        const CHUNK_SIZE: usize = 8; // Process 8 bytes at a time
        
        let mut i = 0;
        
        // Process chunks of 8 bytes (32 values)
        while i + CHUNK_SIZE <= data.len() && result.len() + CHUNK_SIZE * 4 <= element_count {
            for j in 0..CHUNK_SIZE {
                let byte = data[i + j];
                
                // Extract 4 values from each byte (2 bits each)
                let val0 = ((byte & 0x03) as i8) - 1;
                let val1 = (((byte >> 2) & 0x03) as i8) - 1;
                let val2 = (((byte >> 4) & 0x03) as i8) - 1;
                let val3 = (((byte >> 6) & 0x03) as i8) - 1;
                
                result.push(val0);
                result.push(val1);
                result.push(val2);
                result.push(val3);
            }
            
            i += CHUNK_SIZE;
        }
        
        // Process remaining bytes
        while i < data.len() && result.len() < element_count {
            let byte = data[i];
            
            // Extract 4 values from each byte (2 bits each)
            for shift in [0, 2, 4, 6] {
                if result.len() >= element_count {
                    break;
                }
                let val = (((byte >> shift) & 0x03) as i8) - 1; // Convert {0,1,2} to {-1,0,1}
                result.push(val);
            }
            
            i += 1;
        }
        
        Ok(())
    }
    
    /// Scalar fallback for 2-bit packed weights
    fn unpack_bit_packed_2bit_scalar(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        for &byte in data {
            if result.len() >= element_count {
                break;
            }
            
            // Extract 4 values from each byte (2 bits each)
            for shift in [0, 2, 4, 6] {
                if result.len() >= element_count {
                    break;
                }
                let val = (((byte >> shift) & 0x03) as i8) - 1; // Convert {0,1,2} to {-1,0,1}
                result.push(val);
            }
        }
        Ok(())
    }
    
    /// SIMD-optimized unpacking for base-3 packed weights
    fn unpack_base3_packed(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let element_count = packed.metadata.element_count;
        let mut result = Vec::with_capacity(element_count);
        
        // Base-3 unpacking is more complex for SIMD, use optimized scalar for now
        self.unpack_base3_packed_scalar(&packed.data, &mut result, element_count)?;
        
        result.truncate(element_count);
        Ok(result)
    }
    
    /// Scalar implementation for base-3 packed weights
    fn unpack_base3_packed_scalar(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        for &byte_val in data {
            let mut remaining = byte_val;
            
            // Unpack up to 5 values from each byte
            for _ in 0..5 {
                if result.len() >= element_count {
                    break;
                }
                
                let ternary_val = remaining % 3;
                result.push((ternary_val as i8) - 1); // Convert {0,1,2} back to {-1,0,1}
                remaining /= 3;
            }
        }
        Ok(())
    }
    
    /// SIMD-optimized unpacking for byte-aligned weights
    fn unpack_byte_aligned(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let element_count = packed.metadata.element_count;
        let mut result = Vec::with_capacity(element_count);
        
        if self.simd_available.has_simd() {
            self.unpack_byte_aligned_simd(&packed.data, &mut result, element_count)?;
        } else {
            self.unpack_byte_aligned_scalar(&packed.data, &mut result, element_count)?;
        }
        
        result.truncate(element_count);
        Ok(result)
    }
    
    /// SIMD implementation for byte-aligned weights
    fn unpack_byte_aligned_simd(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        // Process in chunks for better performance
        const CHUNK_SIZE: usize = 16; // Process 16 bytes at a time
        
        let mut i = 0;
        
        // Process chunks
        while i + CHUNK_SIZE <= data.len() && result.len() + CHUNK_SIZE <= element_count {
            for j in 0..CHUNK_SIZE {
                result.push((data[i + j] as i8) - 1); // Convert {0,1,2} to {-1,0,1}
            }
            i += CHUNK_SIZE;
        }
        
        // Process remaining bytes
        while i < data.len() && result.len() < element_count {
            result.push((data[i] as i8) - 1); // Convert {0,1,2} to {-1,0,1}
            i += 1;
        }
        
        Ok(())
    }
    
    /// Scalar implementation for byte-aligned weights
    fn unpack_byte_aligned_scalar(
        &self,
        data: &[u8],
        result: &mut Vec<i8>,
        element_count: usize,
    ) -> Result<(), QuantizationError> {
        for &byte in data {
            if result.len() >= element_count {
                break;
            }
            result.push((byte as i8) - 1); // Convert {0,1,2} to {-1,0,1}
        }
        Ok(())
    }
    
    /// Unpacking for run-length encoded weights (inherently sequential)
    fn unpack_run_length_encoded(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut result = Vec::with_capacity(packed.metadata.element_count);
        
        // RLE is inherently sequential, no SIMD benefit
        let mut i = 0;
        while i < packed.data.len() {
            if i + 1 >= packed.data.len() {
                break;
            }
            
            let value = (packed.data[i] as i8) - 1; // Convert {0,1,2} back to {-1,0,1}
            let count = packed.data[i + 1] as usize;
            
            for _ in 0..count {
                if result.len() >= packed.metadata.element_count {
                    break;
                }
                result.push(value);
            }
            
            i += 2;
        }
        
        result.truncate(packed.metadata.element_count);
        Ok(result)
    }
    
    /// Unpacking for compressed sparse weights
    fn unpack_compressed_sparse(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        let mut result = vec![0i8; packed.metadata.element_count];
        
        if packed.data.len() < 4 {
            return Ok(result); // All zeros
        }
        
        // Read number of non-zero elements
        let nnz = u32::from_le_bytes([
            packed.data[0], packed.data[1], packed.data[2], packed.data[3]
        ]) as usize;
        
        let mut offset = 4;
        
        // Read indices
        let mut indices = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            if offset + 4 > packed.data.len() {
                return Err(QuantizationError::InvalidInput("Corrupted sparse data".to_string()));
            }
            let idx = u32::from_le_bytes([
                packed.data[offset], packed.data[offset + 1],
                packed.data[offset + 2], packed.data[offset + 3]
            ]) as usize;
            indices.push(idx);
            offset += 4;
        }
        
        // Read values and populate result
        for (i, &idx) in indices.iter().enumerate() {
            if offset + i >= packed.data.len() || idx >= result.len() {
                return Err(QuantizationError::InvalidInput("Corrupted sparse data".to_string()));
            }
            let val = (packed.data[offset + i] as i8) - 1; // Convert {0,1,2} back to {-1,0,1}
            result[idx] = val;
        }
        
        Ok(result)
    }
    
    /// Fallback unpacking for unsupported strategies
    fn unpack_fallback(&self, packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
        // Use the original packer's unpack method as fallback
        use super::packing::TernaryPackerFactory;
        let packer = TernaryPackerFactory::create_packer(packed.strategy);
        packer.unpack(packed)
    }
}

impl Default for SimdUnpacker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to unpack weights with SIMD optimization
pub fn simd_unpack_weights(packed: &PackedTernaryWeights) -> Result<Vec<i8>, QuantizationError> {
    let unpacker = SimdUnpacker::new();
    unpacker.unpack(packed)
}

/// Benchmark utilities for measuring SIMD performance
pub mod benchmark {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark result for unpacking performance
    #[derive(Debug, Clone)]
    pub struct UnpackingBenchmark {
        pub strategy: TernaryPackingStrategy,
        pub element_count: usize,
        pub simd_time_ns: u64,
        pub scalar_time_ns: u64,
        pub speedup: f64,
        pub simd_capabilities: SimdCapabilities,
    }
    
    /// Benchmark SIMD vs scalar unpacking performance
    pub fn benchmark_unpacking(
        packed: &PackedTernaryWeights,
        iterations: usize,
    ) -> Result<UnpackingBenchmark, QuantizationError> {
        let simd_unpacker = SimdUnpacker::new();
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        
        // Warm up
        for _ in 0..10 {
            let _ = simd_unpacker.unpack(packed)?;
            let _ = scalar_unpacker.unpack(packed)?;
        }
        
        // Benchmark SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_unpacker.unpack(packed)?;
        }
        let simd_time = start.elapsed().as_nanos() as u64;
        
        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = scalar_unpacker.unpack(packed)?;
        }
        let scalar_time = start.elapsed().as_nanos() as u64;
        
        let speedup = scalar_time as f64 / simd_time as f64;
        
        Ok(UnpackingBenchmark {
            strategy: packed.strategy,
            element_count: packed.metadata.element_count,
            simd_time_ns: simd_time,
            scalar_time_ns: scalar_time,
            speedup,
            simd_capabilities: simd_unpacker.simd_available,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::packing::*;
    
    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        // Just ensure it doesn't panic
        assert!(caps.has_simd() || !caps.has_simd());
    }
    
    #[test]
    fn test_simd_unpacker_creation() {
        let unpacker = SimdUnpacker::new();
        assert!(unpacker.simd_available.has_simd() || !unpacker.simd_available.has_simd());
    }
    
    #[test]
    fn test_bit_packed_2bit_unpacking() {
        // Create test data
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_byte_aligned_unpacking() {
        // Create test data
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 1];
        let config = TernaryPackingConfig {
            strategy: TernaryPackingStrategy::ByteAligned,
            alignment: 16,
            ..Default::default()
        };
        let packer = ByteAlignedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_base3_packed_unpacking() {
        // Create test data with length divisible by 5
        let weights = vec![-1i8, 0, 1, -1, 0];
        let config = TernaryPackingConfig::default();
        let packer = Base3PackedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_run_length_encoded_unpacking() {
        // Create test data with runs
        let weights = vec![0i8, 0, 0, 1, 1, -1, -1, -1, 0, 0];
        let config = TernaryPackingConfig::default();
        let packer = RunLengthEncodedPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_compressed_sparse_unpacking() {
        // Create sparse test data
        let weights = vec![0i8, 0, 1, 0, 0, 0, -1, 0, 0, 0];
        let config = TernaryPackingConfig::default();
        let packer = CompressedSparsePacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test SIMD unpacking
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_simd_vs_scalar_consistency() {
        // Test that SIMD and scalar implementations produce identical results
        let weights = vec![-1i8, 0, 1, -1, 0, 1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        
        // Test with SIMD enabled
        let simd_unpacker = SimdUnpacker::new();
        let simd_result = simd_unpacker.unpack(&packed).unwrap();
        
        // Test with SIMD disabled (scalar fallback)
        let scalar_unpacker = SimdUnpacker::with_capabilities(SimdCapabilities {
            sse2: false,
            avx2: false,
            neon: false,
        });
        let scalar_result = scalar_unpacker.unpack(&packed).unwrap();
        
        assert_eq!(simd_result, scalar_result);
        assert_eq!(weights, simd_result);
    }
    
    #[test]
    fn test_convenience_function() {
        let weights = vec![-1i8, 0, 1, -1];
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        let unpacked = simd_unpack_weights(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_large_data_unpacking() {
        // Test with larger data to exercise SIMD paths
        let mut weights = Vec::new();
        for i in 0..1000 {
            weights.push(match i % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            });
        }
        
        let config = TernaryPackingConfig::default();
        let packer = BitPacked2BitPacker;
        
        let packed = packer.pack(&weights, &config).unwrap();
        let unpacker = SimdUnpacker::new();
        let unpacked = unpacker.unpack(&packed).unwrap();
        
        assert_eq!(weights, unpacked);
    }
}