//! In-Place Conversion Operations
//!
//! This module implements in-place conversions that modify tensor data directly
//! in existing memory buffers, minimizing memory allocation overhead.

use crate::memory::conversion::{ConversionContext, ConversionError, ConversionResult, Converter};
use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::HybridMemoryPool;
use std::sync::Arc;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// In-place converter for memory-efficient conversions
pub struct InPlaceConverter {
    /// Whether to allow potentially lossy conversions
    allow_lossy: bool,
    /// Whether to validate data integrity after conversion
    validate_integrity: bool,
}

impl InPlaceConverter {
    /// Creates a new in-place converter
    pub fn new() -> Self {
        Self {
            allow_lossy: false,
            validate_integrity: true,
        }
    }

    /// Creates a new in-place converter that allows lossy conversions
    pub fn new_lossy() -> Self {
        Self {
            allow_lossy: true,
            validate_integrity: true,
        }
    }

    /// Creates a new in-place converter without integrity validation (faster)
    pub fn new_fast() -> Self {
        Self {
            allow_lossy: false,
            validate_integrity: false,
        }
    }

    /// Performs in-place conversion of a tensor
    pub fn convert_in_place(
        &self,
        tensor: &mut BitNetTensor,
        targetdtype: BitNetDType,
    ) -> ConversionResult<()> {
        let sourcedtype = tensor.dtype();

        #[cfg(feature = "tracing")]
        info!(
            "Starting in-place conversion from {} to {}",
            sourcedtype, targetdtype
        );

        // Check if conversion is possible
        if !self.is_in_place_compatible(sourcedtype, targetdtype) {
            return Err(ConversionError::UnsupportedConversion {
                from: sourcedtype,
                to: targetdtype,
            });
        }

        // Check for potential data loss
        if !self.allow_lossy && self.is_lossy_conversion(sourcedtype, targetdtype) {
            return Err(ConversionError::DataLossError {
                from: sourcedtype,
                to: targetdtype,
            });
        }

        let element_count = tensor.element_count();

        // Perform the in-place conversion
        unsafe {
            let ptr = tensor.data.memory_handle.as_ptr() as *mut u8;
            self.convert_memory_in_place(ptr, element_count, sourcedtype, targetdtype)?;
        }

        // Update tensor metadata
        {
            let mut metadata =
                tensor
                    .data
                    .metadata
                    .write()
                    .map_err(|_| ConversionError::InternalError {
                        reason: "Failed to acquire metadata write lock".to_string(),
                    })?;

            metadata.dtype = targetdtype;
            metadata.size_bytes = targetdtype.bytes_for_elements(element_count);
            metadata.touch();
        }

        // Validate integrity if requested
        if self.validate_integrity {
            self.validate_conversion_integrity(tensor, sourcedtype, targetdtype)?;
        }

        #[cfg(feature = "tracing")]
        info!("In-place conversion completed successfully");

        Ok(())
    }

    /// Checks if two data types are compatible for in-place conversion
    pub fn is_in_place_compatible(&self, source: BitNetDType, target: BitNetDType) -> bool {
        // Same type is always compatible
        if source == target {
            return true;
        }

        let source_bits = source.bits_per_element();
        let target_bits = target.bits_per_element();

        // Target must have same or fewer bits per element
        if target_bits > source_bits {
            return false;
        }

        // Check specific compatibility rules
        match (source, target) {
            // F32 can be converted to F16, BF16, or smaller integer types
            (BitNetDType::F32, BitNetDType::F16) => true,
            (BitNetDType::F32, BitNetDType::BF16) => true,
            (BitNetDType::F32, BitNetDType::I8) => true,
            (BitNetDType::F32, BitNetDType::I4) => true,

            // F16 can be converted to smaller types
            (BitNetDType::F16, BitNetDType::I8) => true,
            (BitNetDType::F16, BitNetDType::I4) => true,
            (BitNetDType::F16, BitNetDType::I2) => true,
            (BitNetDType::F16, BitNetDType::I1) => true,
            (BitNetDType::F16, BitNetDType::BitNet158) => true,

            // BF16 can be converted to smaller types
            (BitNetDType::BF16, BitNetDType::I8) => true,
            (BitNetDType::BF16, BitNetDType::I4) => true,
            (BitNetDType::BF16, BitNetDType::I2) => true,
            (BitNetDType::BF16, BitNetDType::I1) => true,
            (BitNetDType::BF16, BitNetDType::BitNet158) => true,

            // I8 can be converted to smaller integer types
            (BitNetDType::I8, BitNetDType::I4) => true,
            (BitNetDType::I8, BitNetDType::I2) => true,
            (BitNetDType::I8, BitNetDType::I1) => true,
            (BitNetDType::I8, BitNetDType::BitNet158) => true,

            // I4 can be converted to smaller types
            (BitNetDType::I4, BitNetDType::I2) => true,
            (BitNetDType::I4, BitNetDType::I1) => true,
            (BitNetDType::I4, BitNetDType::BitNet158) => true,

            // I2 can be converted to smaller types
            (BitNetDType::I2, BitNetDType::I1) => true,
            (BitNetDType::I2, BitNetDType::BitNet158) => true,

            // All other combinations are not supported for in-place conversion
            _ => false,
        }
    }

    /// Checks if a conversion would result in data loss
    fn is_lossy_conversion(&self, source: BitNetDType, target: BitNetDType) -> bool {
        if source == target {
            return false;
        }

        match (source, target) {
            // Float to integer conversions are always lossy
            (BitNetDType::F32, BitNetDType::I8) => true,
            (BitNetDType::F32, BitNetDType::I4) => true,
            (BitNetDType::F16, BitNetDType::I8) => true,
            (BitNetDType::F16, BitNetDType::I4) => true,
            (BitNetDType::F16, BitNetDType::I2) => true,
            (BitNetDType::F16, BitNetDType::I1) => true,
            (BitNetDType::BF16, BitNetDType::I8) => true,
            (BitNetDType::BF16, BitNetDType::I4) => true,
            (BitNetDType::BF16, BitNetDType::I2) => true,
            (BitNetDType::BF16, BitNetDType::I1) => true,

            // Precision reduction in floating point
            (BitNetDType::F32, BitNetDType::F16) => true,
            (BitNetDType::F32, BitNetDType::BF16) => true,

            // Integer quantization
            (BitNetDType::I8, BitNetDType::I4) => true,
            (BitNetDType::I8, BitNetDType::I2) => true,
            (BitNetDType::I8, BitNetDType::I1) => true,
            (BitNetDType::I4, BitNetDType::I2) => true,
            (BitNetDType::I4, BitNetDType::I1) => true,
            (BitNetDType::I2, BitNetDType::I1) => true,

            // BitNet conversions
            (_, BitNetDType::BitNet158) => true,

            _ => false,
        }
    }

    /// Performs the actual in-place memory conversion
    unsafe fn convert_memory_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
        sourcedtype: BitNetDType,
        targetdtype: BitNetDType,
    ) -> ConversionResult<()> {
        match (sourcedtype, targetdtype) {
            // Same type - no conversion needed
            (a, b) if a == b => Ok(()),

            // F32 to F16 conversion
            (BitNetDType::F32, BitNetDType::F16) => {
                self.convert_f32_to_f16_in_place(ptr, element_count)
            }

            // F32 to BF16 conversion
            (BitNetDType::F32, BitNetDType::BF16) => {
                self.convert_f32_to_bf16_in_place(ptr, element_count)
            }

            // F32 to I8 quantization
            (BitNetDType::F32, BitNetDType::I8) => {
                self.convert_f32_to_i8_in_place(ptr, element_count)
            }

            // F32 to I4 quantization
            (BitNetDType::F32, BitNetDType::I4) => {
                self.convert_f32_to_i4_in_place(ptr, element_count)
            }

            // F16 to I8 quantization
            (BitNetDType::F16, BitNetDType::I8) => {
                self.convert_f16_to_i8_in_place(ptr, element_count)
            }

            // I8 to I4 quantization
            (BitNetDType::I8, BitNetDType::I4) => {
                self.convert_i8_to_i4_in_place(ptr, element_count)
            }

            // I8 to I2 quantization
            (BitNetDType::I8, BitNetDType::I2) => {
                self.convert_i8_to_i2_in_place(ptr, element_count)
            }

            // I8 to I1 quantization
            (BitNetDType::I8, BitNetDType::I1) => {
                self.convert_i8_to_i1_in_place(ptr, element_count)
            }

            // Any type to BitNet 1.58b
            (_, BitNetDType::BitNet158) => {
                self.convert_to_bitnet158_in_place(ptr, element_count, sourcedtype)
            }

            _ => Err(ConversionError::UnsupportedConversion {
                from: sourcedtype,
                to: targetdtype,
            }),
        }
    }

    // Specific conversion implementations

    unsafe fn convert_f32_to_f16_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let f32_slice = std::slice::from_raw_parts_mut(ptr as *mut f32, element_count);
        let f16_slice = std::slice::from_raw_parts_mut(ptr as *mut u16, element_count);

        // Convert from back to front to avoid overwriting data
        for i in (0..element_count).rev() {
            let f32_val = f32_slice[i];
            let f16_bits = self.f32_to_f16_bits(f32_val);
            f16_slice[i] = f16_bits;
        }

        Ok(())
    }

    unsafe fn convert_f32_to_bf16_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let f32_slice = std::slice::from_raw_parts_mut(ptr as *mut f32, element_count);
        let bf16_slice = std::slice::from_raw_parts_mut(ptr as *mut u16, element_count);

        // Convert from back to front to avoid overwriting data
        for i in (0..element_count).rev() {
            let f32_val = f32_slice[i];
            let bf16_bits = self.f32_to_bf16_bits(f32_val);
            bf16_slice[i] = bf16_bits;
        }

        Ok(())
    }

    unsafe fn convert_f32_to_i8_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let f32_slice = std::slice::from_raw_parts_mut(ptr as *mut f32, element_count);
        let i8_slice = std::slice::from_raw_parts_mut(ptr as *mut i8, element_count);

        // Convert from front to back since i8 is smaller
        for i in 0..element_count {
            let f32_val = f32_slice[i];
            let i8_val = if f32_val.is_nan() {
                0i8
            } else {
                f32_val.clamp(-128.0, 127.0).round() as i8
            };
            i8_slice[i] = i8_val;
        }

        Ok(())
    }

    unsafe fn convert_f32_to_i4_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let f32_slice = std::slice::from_raw_parts_mut(ptr as *mut f32, element_count);
        let byte_slice = std::slice::from_raw_parts_mut(ptr, (element_count + 1) / 2);

        // Pack two I4 values into each byte
        for i in 0..element_count {
            let f32_val = f32_slice[i];
            let i4_val = if f32_val.is_nan() {
                0u8
            } else {
                ((f32_val.clamp(-8.0, 7.0).round() as i8).clamp(-8, 7) & 0x0F) as u8
            };

            let byte_idx = i / 2;
            if i % 2 == 0 {
                byte_slice[byte_idx] = i4_val;
            } else {
                byte_slice[byte_idx] |= i4_val << 4;
            }
        }

        Ok(())
    }

    unsafe fn convert_f16_to_i8_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let f16_slice = std::slice::from_raw_parts_mut(ptr as *mut u16, element_count);
        let i8_slice = std::slice::from_raw_parts_mut(ptr as *mut i8, element_count);

        // Convert from front to back since i8 is smaller
        for i in 0..element_count {
            let f16_bits = f16_slice[i];
            let f32_val = self.f16_bits_to_f32(f16_bits);
            let i8_val = f32_val.clamp(-128.0, 127.0).round() as i8;
            i8_slice[i] = i8_val;
        }

        Ok(())
    }

    unsafe fn convert_i8_to_i4_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let i8_slice = std::slice::from_raw_parts_mut(ptr as *mut i8, element_count);
        let byte_slice = std::slice::from_raw_parts_mut(ptr, (element_count + 1) / 2);

        // Pack two I4 values into each byte
        for i in 0..element_count {
            let i8_val = i8_slice[i];
            let i4_val = (i8_val.clamp(-8, 7) & 0x0F) as u8;

            let byte_idx = i / 2;
            if i % 2 == 0 {
                byte_slice[byte_idx] = i4_val;
            } else {
                byte_slice[byte_idx] |= i4_val << 4;
            }
        }

        Ok(())
    }

    unsafe fn convert_i8_to_i2_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let i8_slice = std::slice::from_raw_parts_mut(ptr as *mut i8, element_count);
        let byte_slice = std::slice::from_raw_parts_mut(ptr, (element_count + 3) / 4);

        // Pack four I2 values into each byte
        for i in 0..element_count {
            let i8_val = i8_slice[i];
            let i2_val = (i8_val.clamp(-2, 1) & 0x03) as u8;

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;

            if bit_offset == 0 {
                byte_slice[byte_idx] = i2_val;
            } else {
                byte_slice[byte_idx] |= i2_val << bit_offset;
            }
        }

        Ok(())
    }

    unsafe fn convert_i8_to_i1_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let i8_slice = std::slice::from_raw_parts_mut(ptr as *mut i8, element_count);
        let byte_slice = std::slice::from_raw_parts_mut(ptr, (element_count + 7) / 8);

        // Pack eight I1 values into each byte
        for i in 0..element_count {
            let i8_val = i8_slice[i];
            let i1_val = if i8_val >= 0 { 1u8 } else { 0u8 };

            let byte_idx = i / 8;
            let bit_offset = i % 8;

            if bit_offset == 0 {
                byte_slice[byte_idx] = i1_val;
            } else {
                byte_slice[byte_idx] |= i1_val << bit_offset;
            }
        }

        Ok(())
    }

    unsafe fn convert_to_bitnet158_in_place(
        &self,
        ptr: *mut u8,
        element_count: usize,
        sourcedtype: BitNetDType,
    ) -> ConversionResult<()> {
        let byte_slice = std::slice::from_raw_parts_mut(ptr, (element_count + 3) / 4);

        // First, convert source to f32 values, then quantize to {-1, 0, +1}
        for i in 0..element_count {
            let f32_val = match sourcedtype {
                BitNetDType::F32 => {
                    let f32_slice = std::slice::from_raw_parts(ptr as *const f32, element_count);
                    f32_slice[i]
                }
                BitNetDType::F16 => {
                    let f16_slice = std::slice::from_raw_parts(ptr as *const u16, element_count);
                    self.f16_bits_to_f32(f16_slice[i])
                }
                BitNetDType::I8 => {
                    let i8_slice = std::slice::from_raw_parts(ptr as *const i8, element_count);
                    i8_slice[i] as f32
                }
                _ => 0.0f32,
            };

            // Quantize to BitNet 1.58b values
            let quantized = if f32_val > 0.5 {
                1u8 // +1 -> 01
            } else if f32_val < -0.5 {
                3u8 // -1 -> 11
            } else {
                0u8 // 0 -> 00
            };

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;

            if bit_offset == 0 {
                byte_slice[byte_idx] = quantized;
            } else {
                byte_slice[byte_idx] |= quantized << bit_offset;
            }
        }

        Ok(())
    }

    // Helper functions for data type conversions

    fn f32_to_f16_bits(&self, f32_val: f32) -> u16 {
        if f32_val.is_nan() {
            return 0x7E00u16; // NaN
        }
        if f32_val.is_infinite() {
            return if f32_val.is_sign_positive() {
                0x7C00u16
            } else {
                0xFC00u16
            };
        }

        let bits = f32_val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mantissa = (bits >> 13) & 0x3FF;

        if exp <= 0 {
            sign as u16 // Underflow to zero
        } else if exp >= 31 {
            sign as u16 | 0x7C00 // Overflow to infinity
        } else {
            sign as u16 | ((exp as u16) << 10) | mantissa as u16
        }
    }

    fn f32_to_bf16_bits(&self, f32_val: f32) -> u16 {
        // BF16 is just the upper 16 bits of F32
        (f32_val.to_bits() >> 16) as u16
    }

    fn f16_bits_to_f32(&self, f16_bits: u16) -> f32 {
        let sign = (f16_bits & 0x8000) as u32;
        let exp = ((f16_bits >> 10) & 0x1F) as i32;
        let mantissa = (f16_bits & 0x3FF) as u32;

        let f32_bits = if exp == 0 {
            if mantissa == 0 {
                sign << 16 // Zero
            } else {
                // Denormalized number
                let exp_adj = 127 - 15 - 10;
                sign << 16 | ((exp_adj as u32) << 23) | (mantissa << 13)
            }
        } else if exp == 31 {
            if mantissa == 0 {
                sign << 16 | 0x7F800000 // Infinity
            } else {
                sign << 16 | 0x7FC00000 // NaN
            }
        } else {
            // Normalized number
            let exp_adj = exp - 15 + 127;
            sign << 16 | ((exp_adj as u32) << 23) | (mantissa << 13)
        };

        f32::from_bits(f32_bits)
    }

    /// Validates the integrity of the conversion
    fn validate_conversion_integrity(
        &self,
        tensor: &BitNetTensor,
        _source_dtype: BitNetDType,
        targetdtype: BitNetDType,
    ) -> ConversionResult<()> {
        // Basic validation - check that tensor metadata is consistent
        let metadata = tensor
            .data
            .metadata
            .read()
            .map_err(|_| ConversionError::InternalError {
                reason: "Failed to acquire metadata read lock".to_string(),
            })?;

        if metadata.dtype != targetdtype {
            return Err(ConversionError::InternalError {
                reason: "Tensor metadata not updated correctly".to_string(),
            });
        }

        let expected_size = targetdtype.bytes_for_elements(metadata.element_count);
        if metadata.size_bytes != expected_size {
            return Err(ConversionError::InternalError {
                reason: "Tensor size not updated correctly".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for InPlaceConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl Converter for InPlaceConverter {
    fn convert(
        &self,
        source: &BitNetTensor,
        context: &ConversionContext,
        _pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        // For the Converter trait, we need to create a copy since we can't modify the source
        let mut target =
            source
                .clone_tensor(_pool)
                .map_err(|e| ConversionError::InternalError {
                    reason: format!("Failed to clone tensor: {}", e),
                })?;

        self.convert_in_place(&mut target, context.targetdtype)?;
        Ok(target)
    }

    fn supports(&self, context: &ConversionContext) -> bool {
        // Check if this is an in-place compatible conversion
        context.is_in_place_compatible()
            && std::mem::discriminant(&context.source_device)
                == std::mem::discriminant(&context.target_device)
    }

    fn estimate_time_ms(&self, context: &ConversionContext) -> u64 {
        let element_count: usize = context.shape.iter().product();

        // In-place conversions are very fast since they don't allocate memory
        // Estimate based on element processing speed
        let elements_per_ms = 1_000_000; // ~1M elements per millisecond
        (element_count / elements_per_ms).max(1) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_in_place_compatibility() {
        let converter = InPlaceConverter::new();

        // Same type should be compatible
        assert!(converter.is_in_place_compatible(BitNetDType::F32, BitNetDType::F32));

        // Smaller target should be compatible
        assert!(converter.is_in_place_compatible(BitNetDType::F32, BitNetDType::F16));
        assert!(converter.is_in_place_compatible(BitNetDType::F32, BitNetDType::I8));
        assert!(converter.is_in_place_compatible(BitNetDType::I8, BitNetDType::I4));

        // Larger target should not be compatible
        assert!(!converter.is_in_place_compatible(BitNetDType::F16, BitNetDType::F32));
        assert!(!converter.is_in_place_compatible(BitNetDType::I4, BitNetDType::I8));
    }

    #[test]
    fn test_lossy_conversion_detection() {
        let converter = InPlaceConverter::new();

        // Float to integer should be lossy
        assert!(converter.is_lossy_conversion(BitNetDType::F32, BitNetDType::I8));
        assert!(converter.is_lossy_conversion(BitNetDType::F16, BitNetDType::I4));

        // Precision reduction should be lossy
        assert!(converter.is_lossy_conversion(BitNetDType::F32, BitNetDType::F16));
        assert!(converter.is_lossy_conversion(BitNetDType::I8, BitNetDType::I4));

        // Same type should not be lossy
        assert!(!converter.is_lossy_conversion(BitNetDType::F32, BitNetDType::F32));
    }

    #[test]
    fn test_f32_to_f16_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = InPlaceConverter::new_lossy();

        let mut tensor = BitNetTensor::ones(&[4, 4], BitNetDType::F32, &device, &pool).unwrap();
        let original_size = tensor.size_bytes();

        converter
            .convert_in_place(&mut tensor, BitNetDType::F16)
            .unwrap();

        assert_eq!(tensor.dtype(), BitNetDType::F16);
        assert_eq!(tensor.shape(), vec![4, 4]);
        assert!(tensor.size_bytes() < original_size); // Should use less memory
    }

    #[test]
    fn test_unsupported_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = InPlaceConverter::new();

        let mut tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F16, &device, &pool).unwrap();
        let result = converter.convert_in_place(&mut tensor, BitNetDType::F32);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConversionError::UnsupportedConversion { .. }
        ));
    }

    #[test]
    fn test_lossy_conversion_rejection() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = InPlaceConverter::new(); // Strict mode (no lossy)

        let mut tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = converter.convert_in_place(&mut tensor, BitNetDType::I8);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConversionError::DataLossError { .. }
        ));
    }

    #[test]
    fn test_converter_trait_implementation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let converter = InPlaceConverter::new_lossy();

        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::F16,
            device.clone(),
            device.clone(),
            vec![2, 2],
        );

        assert!(converter.supports(&context));

        let source = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let result = converter.convert(&source, &context, &pool).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_data_type_conversion_helpers() {
        let converter = InPlaceConverter::new();

        // Test F32 to F16 conversion
        let f16_bits = converter.f32_to_f16_bits(1.0f32);
        assert_eq!(f16_bits, 0x3C00); // 1.0 in F16

        let f16_bits = converter.f32_to_f16_bits(0.0f32);
        assert_eq!(f16_bits, 0x0000); // 0.0 in F16

        // Test F32 to BF16 conversion
        let bf16_bits = converter.f32_to_bf16_bits(1.0f32);
        assert_eq!(bf16_bits, 0x3F80); // Upper 16 bits of 1.0f32

        // Test F16 to F32 conversion
        let f32_val = converter.f16_bits_to_f32(0x3C00);
        assert!((f32_val - 1.0f32).abs() < 1e-6);
    }
}
