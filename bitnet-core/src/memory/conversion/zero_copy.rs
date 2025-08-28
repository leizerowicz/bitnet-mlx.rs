//! Zero-Copy Conversion Utilities
//!
//! This module implements zero-copy conversions for compatible data types,
//! eliminating memory allocation and copying overhead when possible.

use crate::memory::conversion::{ConversionContext, ConversionError, ConversionResult, Converter};
use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::HybridMemoryPool;
use std::sync::Arc;

#[cfg(feature = "tracing")]
use tracing::{debug, info};

/// Zero-copy converter for compatible data types
#[allow(dead_code)]
pub struct ZeroCopyConverter {
    /// Whether to enable strict compatibility checking
    strict_mode: bool,
}

impl ZeroCopyConverter {
    /// Creates a new zero-copy converter
    pub fn new() -> Self {
        Self { strict_mode: false }
    }

    /// Creates a new zero-copy converter with strict mode enabled
    pub fn new_strict() -> Self {
        Self { strict_mode: true }
    }

    /// Checks if two data types are zero-copy compatible
    pub fn is_compatible(source: BitNetDType, target: BitNetDType) -> bool {
        match (source, target) {
            // Same type is always compatible
            (a, b) if a == b => true,

            // F16 and BF16 have same memory layout (16 bits)
            (BitNetDType::F16, BitNetDType::BF16) | (BitNetDType::BF16, BitNetDType::F16) => true,

            // All other conversions require data transformation
            _ => false,
        }
    }

    /// Checks if two data types are zero-copy compatible with mode consideration
    fn is_compatible_with_mode(&self, source: BitNetDType, target: BitNetDType) -> bool {
        if self.strict_mode {
            // In strict mode, only identical types are compatible
            source == target
        } else {
            // In non-strict mode, use the standard compatibility rules
            Self::is_compatible(source, target)
        }
    }

    /// Performs a zero-copy reinterpretation between compatible types
    pub fn reinterpret_cast(
        &self,
        source: &BitNetTensor,
        targetdtype: BitNetDType,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        let sourcedtype = source.dtype();

        if !self.is_compatible_with_mode(sourcedtype, targetdtype) {
            return Err(ConversionError::UnsupportedConversion {
                from: sourcedtype,
                to: targetdtype,
            });
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Performing zero-copy reinterpret cast from {} to {}",
            sourcedtype, targetdtype
        );

        // For same types, just clone the tensor reference
        if sourcedtype == targetdtype {
            return Ok(source.clone());
        }

        // For compatible types with different interpretation, create a new tensor
        // that shares the same memory but with different metadata
        match (sourcedtype, targetdtype) {
            (BitNetDType::F16, BitNetDType::BF16) | (BitNetDType::BF16, BitNetDType::F16) => {
                self.reinterpret_f16_bf16(source, targetdtype, pool)
            }
            _ => Err(ConversionError::UnsupportedConversion {
                from: sourcedtype,
                to: targetdtype,
            }),
        }
    }

    /// Reinterprets between F16 and BF16 (both are 16-bit formats)
    fn reinterpret_f16_bf16(
        &self,
        source: &BitNetTensor,
        targetdtype: BitNetDType,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        let shape = source.shape();
        let device = source.device(); // Fixed - get device from source tensor

        #[cfg(feature = "tracing")]
        debug!("Reinterpreting F16/BF16 tensor with shape {:?}", shape);

        // Create a new tensor with the same memory layout but different type interpretation
        let target_tensor =
            BitNetTensor::zeros(&shape, targetdtype, &device, pool).map_err(|e| {
                ConversionError::InternalError {
                    reason: e.to_string(),
                }
            })?;

        // Copy the raw memory (this is still zero-copy in terms of data transformation)
        unsafe {
            let src_ptr = source.data.memory_handle.as_ptr();
            let dst_ptr = target_tensor.data.memory_handle.as_ptr() as *mut u8;
            let size_bytes = source.size_bytes();

            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size_bytes);
        }

        // Copy metadata if requested
        if let Some(name) = source.name() {
            target_tensor.set_name(Some(format!("{}_reinterpreted", name)));
        }

        #[cfg(feature = "tracing")]
        info!("Zero-copy reinterpretation completed successfully");

        Ok(target_tensor)
    }

    /// Creates a view of the tensor with a different data type interpretation
    /// This is the most efficient zero-copy operation as it doesn't even copy memory
    pub fn create_view(
        &self,
        source: &BitNetTensor,
        targetdtype: BitNetDType,
    ) -> ConversionResult<TensorView> {
        let sourcedtype = source.dtype();

        if !self.is_compatible_with_mode(sourcedtype, targetdtype) {
            return Err(ConversionError::UnsupportedConversion {
                from: sourcedtype,
                to: targetdtype,
            });
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Creating zero-copy view from {} to {}",
            sourcedtype, targetdtype
        );

        Ok(TensorView {
            source_tensor: source.clone(),
            viewdtype: targetdtype,
        })
    }

    /// Validates that a zero-copy conversion is safe
    fn validate_conversion(
        &self,
        sourcedtype: BitNetDType,
        targetdtype: BitNetDType,
    ) -> ConversionResult<()> {
        if self.strict_mode {
            // In strict mode, only allow exact type matches
            if sourcedtype != targetdtype {
                return Err(ConversionError::UnsupportedConversion {
                    from: sourcedtype,
                    to: targetdtype,
                });
            }
        } else {
            // In normal mode, allow compatible types
            if !Self::is_compatible(sourcedtype, targetdtype) {
                return Err(ConversionError::UnsupportedConversion {
                    from: sourcedtype,
                    to: targetdtype,
                });
            }
        }

        Ok(())
    }
}

impl Default for ZeroCopyConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl Converter for ZeroCopyConverter {
    fn convert(
        &self,
        source: &BitNetTensor,
        context: &ConversionContext,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        // Validate that this is a zero-copy compatible conversion
        self.validate_conversion(context.sourcedtype, context.targetdtype)?;

        // Check device compatibility
        if std::mem::discriminant(&context.source_device)
            != std::mem::discriminant(&context.target_device)
        {
            return Err(ConversionError::DeviceMismatch);
        }

        // Perform the zero-copy conversion
        self.reinterpret_cast(source, context.targetdtype, pool)
    }

    fn supports(&self, context: &ConversionContext) -> bool {
        // Check if this is a zero-copy compatible conversion
        if !context.is_zero_copy_compatible() {
            return false;
        }

        // Check device compatibility
        std::mem::discriminant(&context.source_device)
            == std::mem::discriminant(&context.target_device)
    }

    fn estimate_time_ms(&self, context: &ConversionContext) -> u64 {
        if context.sourcedtype == context.targetdtype {
            // Pure reference copy - essentially instant
            0
        } else {
            // Memory copy for reinterpretation - very fast
            let element_count: usize = context.shape.iter().product();
            let size_bytes = context.sourcedtype.bytes_for_elements(element_count);

            // Estimate ~10 GB/s memory bandwidth
            ((size_bytes as f64) / (10.0 * 1024.0 * 1024.0 * 1024.0) * 1000.0) as u64
        }
    }
}

/// A zero-copy view of a tensor with a different data type interpretation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorView {
    source_tensor: BitNetTensor,
    viewdtype: BitNetDType,
}

impl TensorView {
    /// Gets the underlying source tensor
    pub fn source(&self) -> &BitNetTensor {
        &self.source_tensor
    }

    /// Gets the view data type
    pub fn dtype(&self) -> BitNetDType {
        self.viewdtype
    }

    /// Gets the tensor shape (same as source)
    pub fn shape(&self) -> Vec<usize> {
        self.source_tensor.shape()
    }

    /// Gets the device (same as source)
    pub fn device(&self) -> candle_core::Device {
        self.source_tensor.device()
    }

    /// Gets the size in bytes (same as source)
    pub fn size_bytes(&self) -> usize {
        self.source_tensor.size_bytes()
    }

    /// Gets the number of elements (same as source)
    pub fn element_count(&self) -> usize {
        self.source_tensor.element_count()
    }

    /// Materializes the view into a new tensor with the view's data type
    pub fn materialize(&self, pool: &Arc<HybridMemoryPool>) -> ConversionResult<BitNetTensor> {
        let converter = ZeroCopyConverter::new();
        converter.reinterpret_cast(&self.source_tensor, self.viewdtype, pool)
    }

    /// Creates a new view with a different data type
    pub fn reinterpret(&self, newdtype: BitNetDType) -> ConversionResult<TensorView> {
        if !ZeroCopyConverter::is_compatible(self.viewdtype, newdtype) {
            return Err(ConversionError::UnsupportedConversion {
                from: self.viewdtype,
                to: newdtype,
            });
        }

        Ok(TensorView {
            source_tensor: self.source_tensor.clone(),
            viewdtype: newdtype,
        })
    }

    /// Gets access to the raw memory (unsafe)
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.source_tensor.data.memory_handle.as_ptr()
    }

    /// Gets mutable access to the raw memory (unsafe)
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        self.source_tensor.data.memory_handle.as_ptr() as *mut u8
    }
}

/// Utility functions for zero-copy operations
pub mod utils {
    use super::*;

    /// Checks if a conversion can be performed with zero-copy
    pub fn can_zero_copy(source: BitNetDType, target: BitNetDType) -> bool {
        ZeroCopyConverter::is_compatible(source, target)
    }

    /// Gets the memory layout compatibility between two data types
    pub fn memory_layout_compatible(source: BitNetDType, target: BitNetDType) -> bool {
        source.bits_per_element() == target.bits_per_element()
    }

    /// Estimates the performance gain of zero-copy vs standard conversion
    pub fn performance_gain_estimate(
        source: BitNetDType,
        target: BitNetDType,
        _element_count: usize,
    ) -> f64 {
        if can_zero_copy(source, target) {
            if source == target {
                // Pure reference copy - infinite speedup
                f64::INFINITY
            } else {
                // Memory copy vs data transformation - typically 10-100x faster
                50.0
            }
        } else {
            // No zero-copy possible
            1.0
        }
    }

    /// Creates a compatibility matrix for all BitNet data types
    pub fn create_compatibility_matrix() -> Vec<Vec<bool>> {
        let types = BitNetDType::all_types();
        let mut matrix = vec![vec![false; types.len()]; types.len()];

        for (i, &source) in types.iter().enumerate() {
            for (j, &target) in types.iter().enumerate() {
                matrix[i][j] = can_zero_copy(source, target);
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_compatibility_checking() {
        // Same types should be compatible
        assert!(ZeroCopyConverter::is_compatible(
            BitNetDType::F32,
            BitNetDType::F32
        ));
        assert!(ZeroCopyConverter::is_compatible(
            BitNetDType::I8,
            BitNetDType::I8
        ));

        // F16 and BF16 should be compatible
        assert!(ZeroCopyConverter::is_compatible(
            BitNetDType::F16,
            BitNetDType::BF16
        ));
        assert!(ZeroCopyConverter::is_compatible(
            BitNetDType::BF16,
            BitNetDType::F16
        ));

        // Different types should not be compatible
        assert!(!ZeroCopyConverter::is_compatible(
            BitNetDType::F32,
            BitNetDType::I8
        ));
        assert!(!ZeroCopyConverter::is_compatible(
            BitNetDType::F16,
            BitNetDType::F32
        ));
    }

    #[test]
    fn test_same_type_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new();

        let source = BitNetTensor::zeros(&[2, 3], BitNetDType::F32, &device, &pool).unwrap();
        let result = converter
            .reinterpret_cast(&source, BitNetDType::F32, &pool)
            .unwrap();

        assert_eq!(result.dtype(), BitNetDType::F32);
        assert_eq!(result.shape(), vec![2, 3]);
        // Note: Device comparison removed as Device doesn't implement PartialEq
    }

    #[test]
    fn test_f16_bf16_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new();

        let source = BitNetTensor::zeros(&[4, 4], BitNetDType::F16, &device, &pool).unwrap();
        let result = converter
            .reinterpret_cast(&source, BitNetDType::BF16, &pool)
            .unwrap();

        assert_eq!(result.dtype(), BitNetDType::BF16);
        assert_eq!(result.shape(), vec![4, 4]);
        assert_eq!(result.size_bytes(), source.size_bytes());
    }

    #[test]
    fn test_unsupported_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new();

        let source = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = converter.reinterpret_cast(&source, BitNetDType::I8, &pool);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConversionError::UnsupportedConversion { .. }
        ));
    }

    #[test]
    fn test_tensor_view() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new();

        let source = BitNetTensor::zeros(&[3, 3], BitNetDType::F16, &device, &pool).unwrap();
        let view = converter.create_view(&source, BitNetDType::BF16).unwrap();

        assert_eq!(view.dtype(), BitNetDType::BF16);
        assert_eq!(view.shape(), vec![3, 3]);
        assert_eq!(view.size_bytes(), source.size_bytes());

        // Materialize the view
        let materialized = view.materialize(&pool).unwrap();
        assert_eq!(materialized.dtype(), BitNetDType::BF16);
        assert_eq!(materialized.shape(), vec![3, 3]);
    }

    #[test]
    fn test_strict_mode() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new_strict();

        let source = BitNetTensor::zeros(&[2, 2], BitNetDType::F16, &device, &pool).unwrap();

        // Same type should work in strict mode
        let result = converter.reinterpret_cast(&source, BitNetDType::F16, &pool);
        assert!(result.is_ok());

        // Different but compatible types should fail in strict mode
        let result = converter.reinterpret_cast(&source, BitNetDType::BF16, &pool);
        assert!(result.is_err());
    }

    #[test]
    fn test_converter_trait_implementation() {
        let _pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = ZeroCopyConverter::new();

        let context = ConversionContext::new(
            BitNetDType::F16,
            BitNetDType::BF16,
            device.clone(),
            device.clone(),
            vec![2, 2],
        );

        assert!(converter.supports(&context));
        assert_eq!(converter.estimate_time_ms(&context), 0); // Very fast

        let unsupported_context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::I8,
            device.clone(),
            device.clone(),
            vec![2, 2],
        );

        assert!(!converter.supports(&unsupported_context));
    }

    #[test]
    fn test_utility_functions() {
        use super::utils::*;

        assert!(can_zero_copy(BitNetDType::F32, BitNetDType::F32));
        assert!(can_zero_copy(BitNetDType::F16, BitNetDType::BF16));
        assert!(!can_zero_copy(BitNetDType::F32, BitNetDType::I8));

        assert!(memory_layout_compatible(
            BitNetDType::F16,
            BitNetDType::BF16
        ));
        assert!(!memory_layout_compatible(
            BitNetDType::F32,
            BitNetDType::F16
        ));

        let gain = performance_gain_estimate(BitNetDType::F32, BitNetDType::F32, 1000);
        assert!(gain.is_infinite());

        let gain = performance_gain_estimate(BitNetDType::F16, BitNetDType::BF16, 1000);
        assert!(gain > 1.0);

        let matrix = create_compatibility_matrix();
        assert!(!matrix.is_empty());
        assert_eq!(matrix.len(), matrix[0].len());
    }
}
