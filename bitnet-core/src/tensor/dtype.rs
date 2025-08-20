//! BitNet Data Type System
//!
//! This module provides a comprehensive data type system for BitNet tensors,
//! including support for standard floating point types, integer types, and
//! BitNet-specific quantized types.

use std::fmt;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, warn};

/// BitNet tensor data types
///
/// This enum represents all supported data types in the BitNet tensor system,
/// including standard types and BitNet-specific quantized types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitNetDType {
    // Standard floating point types
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain floating point 16-bit
    BF16,
    
    // Integer types
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer  
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    
    // Boolean type
    /// Boolean values
    Bool,
    
    // BitNet-specific quantized types
    /// 1.58-bit quantized type (ternary: -1, 0, +1)
    BitNet158,
    /// 1.1-bit quantized type
    BitNet11,
    /// 1-bit quantized type (binary: -1, +1)
    BitNet1,
    /// 4-bit quantized integer
    Int4,
    /// 8-bit quantized with scale and zero-point
    QInt8,
    /// 4-bit quantized with scale and zero-point
    QInt4,
}

impl BitNetDType {
    /// Returns the size in bytes of this data type
    ///
    /// # Returns
    ///
    /// Size in bytes, or None for variable-size types
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert_eq!(BitNetDType::F32.size_bytes(), Some(4));
    /// assert_eq!(BitNetDType::I8.size_bytes(), Some(1));
    /// assert_eq!(BitNetDType::BitNet158.size_bytes(), Some(1)); // Packed representation
    /// ```
    pub fn size_bytes(self) -> Option<usize> {
        match self {
            BitNetDType::F32 => Some(4),
            BitNetDType::F16 => Some(2),
            BitNetDType::BF16 => Some(2),
            BitNetDType::I8 => Some(1),
            BitNetDType::I16 => Some(2),
            BitNetDType::I32 => Some(4),
            BitNetDType::I64 => Some(8),
            BitNetDType::U8 => Some(1),
            BitNetDType::U16 => Some(2),
            BitNetDType::U32 => Some(4),
            BitNetDType::U64 => Some(8),
            BitNetDType::Bool => Some(1),
            BitNetDType::BitNet158 => Some(1), // Packed ternary representation
            BitNetDType::BitNet11 => Some(1),  // Packed 1.1-bit representation
            BitNetDType::BitNet1 => Some(1),   // Packed binary representation (8 bits per byte)
            BitNetDType::Int4 => Some(1),      // Packed 4-bit representation (2 values per byte)
            BitNetDType::QInt8 => Some(1),     // Quantized 8-bit
            BitNetDType::QInt4 => Some(1),     // Packed quantized 4-bit
        }
    }

    /// Alias for size_bytes() for compatibility
    pub fn size(self) -> Option<usize> {
        self.size_bytes()
    }

    /// Returns true if this is a valid data type
    pub fn is_valid(self) -> bool {
        true // All enum variants are valid by definition
    }

    /// Returns true if this type can be used for numeric operations
    pub fn is_numeric(self) -> bool {
        !matches!(self, BitNetDType::Bool)
    }

    /// Returns true if this is a floating point type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert!(BitNetDType::F32.is_floating_point());
    /// assert!(!BitNetDType::I32.is_floating_point());
    /// ```
    pub fn is_floating_point(self) -> bool {
        matches!(self,
            BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16
        )
    }

    /// Returns true if this is an integer type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert!(BitNetDType::I32.is_integer());
    /// assert!(!BitNetDType::F32.is_integer());
    /// ```
    pub fn is_integer(self) -> bool {
        matches!(self,
            BitNetDType::I8 | BitNetDType::I16 | BitNetDType::I32 | BitNetDType::I64 |
            BitNetDType::U8 | BitNetDType::U16 | BitNetDType::U32 | BitNetDType::U64
        )
    }

    /// Returns true if this is a signed type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert!(BitNetDType::I32.is_signed());
    /// assert!(!BitNetDType::U32.is_signed());
    /// assert!(BitNetDType::F32.is_signed()); // Floating point types are considered signed
    /// ```
    pub fn is_signed(self) -> bool {
        matches!(self,
            BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 |
            BitNetDType::I8 | BitNetDType::I16 | BitNetDType::I32 | BitNetDType::I64 |
            BitNetDType::BitNet158 | BitNetDType::BitNet11 | BitNetDType::BitNet1 | BitNetDType::Int4 |
            BitNetDType::QInt8 | BitNetDType::QInt4
        )
    }

    /// Returns true if this is a BitNet-specific quantized type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert!(BitNetDType::BitNet158.is_quantized());
    /// assert!(!BitNetDType::F32.is_quantized());
    /// ```
    pub fn is_quantized(self) -> bool {
        matches!(self,
            BitNetDType::BitNet158 | BitNetDType::BitNet11 | BitNetDType::BitNet1 |
            BitNetDType::Int4 | BitNetDType::QInt8 | BitNetDType::QInt4
        )
    }

    /// Returns true if this type requires special packing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert!(BitNetDType::BitNet158.is_packed());
    /// assert!(!BitNetDType::F32.is_packed());
    /// ```
    pub fn is_packed(self) -> bool {
        matches!(self,
            BitNetDType::BitNet158 | BitNetDType::BitNet11 | BitNetDType::BitNet1 |
            BitNetDType::Int4 | BitNetDType::QInt4
        )
    }

    /// Returns the canonical name for this data type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert_eq!(BitNetDType::F32.name(), "f32");
    /// assert_eq!(BitNetDType::BitNet158.name(), "bitnet1.58");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            BitNetDType::F32 => "f32",
            BitNetDType::F16 => "f16",
            BitNetDType::BF16 => "bf16",
            BitNetDType::I8 => "i8",
            BitNetDType::I16 => "i16",
            BitNetDType::I32 => "i32",
            BitNetDType::I64 => "i64",
            BitNetDType::U8 => "u8",
            BitNetDType::U16 => "u16",
            BitNetDType::U32 => "u32",
            BitNetDType::U64 => "u64",
            BitNetDType::Bool => "bool",
            BitNetDType::BitNet158 => "bitnet1.58",
            BitNetDType::BitNet11 => "bitnet1.1",
            BitNetDType::BitNet1 => "bitnet1",
            BitNetDType::Int4 => "int4",
            BitNetDType::QInt8 => "qint8",
            BitNetDType::QInt4 => "qint4",
        }
    }

    /// Returns all supported data types
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// let all_types = BitNetDType::all_types();
    /// assert!(all_types.contains(&BitNetDType::F32));
    /// assert!(all_types.contains(&BitNetDType::BitNet158));
    /// ```
    pub fn all_types() -> &'static [BitNetDType] {
        &[
            BitNetDType::F32,
            BitNetDType::F16,
            BitNetDType::BF16,
            BitNetDType::I8,
            BitNetDType::I16,
            BitNetDType::I32,
            BitNetDType::I64,
            BitNetDType::U8,
            BitNetDType::U16,
            BitNetDType::U32,
            BitNetDType::U64,
            BitNetDType::Bool,
            BitNetDType::BitNet158,
            BitNetDType::BitNet11,
            BitNetDType::BitNet1,
            BitNetDType::Int4,
            BitNetDType::QInt8,
            BitNetDType::QInt4,
        ]
    }

    /// Returns BitNet-specific quantized types
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// let quantized_types = BitNetDType::quantized_types();
    /// assert!(quantized_types.contains(&BitNetDType::BitNet158));
    /// assert!(!quantized_types.contains(&BitNetDType::F32));
    /// ```
    pub fn quantized_types() -> &'static [BitNetDType] {
        &[
            BitNetDType::BitNet158,
            BitNetDType::BitNet11,
            BitNetDType::BitNet1,
            BitNetDType::Int4,
            BitNetDType::QInt8,
            BitNetDType::QInt4,
        ]
    }

    /// Returns the default data type for standard operations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert_eq!(BitNetDType::default_type(), BitNetDType::F32);
    /// ```
    pub fn default_type() -> BitNetDType {
        BitNetDType::F32
    }

    /// Returns the default data type for BitNet quantized operations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert_eq!(BitNetDType::default_quantized_type(), BitNetDType::BitNet158);
    /// ```
    pub fn default_quantized_type() -> BitNetDType {
        BitNetDType::BitNet158
    }

    /// Converts to candle DType if possible
    ///
    /// # Returns
    ///
    /// Some(DType) for standard types, None for BitNet-specific types
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    /// use candle_core::DType;
    ///
    /// assert_eq!(BitNetDType::F32.to_candle_dtype(), Some(DType::F32));
    /// assert_eq!(BitNetDType::BitNet158.to_candle_dtype(), None);
    /// ```
    pub fn to_candle_dtype(self) -> Option<candle_core::DType> {
        match self {
            BitNetDType::F32 => Some(candle_core::DType::F32),
            BitNetDType::F16 => Some(candle_core::DType::F16),
            BitNetDType::BF16 => Some(candle_core::DType::BF16),
            BitNetDType::I64 => Some(candle_core::DType::I64),
            BitNetDType::U8 => Some(candle_core::DType::U8),
            BitNetDType::U32 => Some(candle_core::DType::U32),
            // BitNet-specific types don't have candle equivalents
            _ => {
                #[cfg(feature = "tracing")]
                debug!("Converting BitNet-specific type {:?} to candle DType not supported", self);
                None
            }
        }
    }

    /// Creates BitNetDType from candle DType
    ///
    /// # Arguments
    ///
    /// * `dtype` - Candle data type to convert from
    ///
    /// # Returns
    ///
    /// Corresponding BitNetDType
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    /// use candle_core::DType;
    ///
    /// assert_eq!(BitNetDType::from_candle_dtype(DType::F32), BitNetDType::F32);
    /// ```
    pub fn from_candle_dtype(dtype: candle_core::DType) -> BitNetDType {
        match dtype {
            candle_core::DType::F32 => BitNetDType::F32,
            candle_core::DType::F16 => BitNetDType::F16,
            candle_core::DType::BF16 => BitNetDType::BF16,
            candle_core::DType::I64 => BitNetDType::I64,
            candle_core::DType::U8 => BitNetDType::U8,
            candle_core::DType::U32 => BitNetDType::U32,
            // For unsupported types, default to F32
            _ => {
                #[cfg(feature = "tracing")]
                warn!("Unsupported candle DType {:?}, defaulting to F32", dtype);
                BitNetDType::F32
            }
        }
    }

    /// Returns the range of valid values for this data type
    ///
    /// # Returns
    ///
    /// (min, max) tuple for the data type, or None for unbounded types
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::BitNetDType;
    ///
    /// assert_eq!(BitNetDType::U8.value_range(), Some((0.0, 255.0)));
    /// assert_eq!(BitNetDType::BitNet158.value_range(), Some((-1.0, 1.0)));
    /// ```
    pub fn value_range(self) -> Option<(f64, f64)> {
        match self {
            BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16 => None, // Unbounded
            BitNetDType::I8 => Some((-128.0, 127.0)),
            BitNetDType::I16 => Some((-32768.0, 32767.0)),
            BitNetDType::I32 => Some((-2147483648.0, 2147483647.0)),
            BitNetDType::I64 => Some((-9223372036854775808.0, 9223372036854775807.0)),
            BitNetDType::U8 => Some((0.0, 255.0)),
            BitNetDType::U16 => Some((0.0, 65535.0)),
            BitNetDType::U32 => Some((0.0, 4294967295.0)),
            BitNetDType::U64 => Some((0.0, 18446744073709551615.0)),
            BitNetDType::Bool => Some((0.0, 1.0)),
            BitNetDType::BitNet158 => Some((-1.0, 1.0)),
            BitNetDType::BitNet11 => Some((-1.0, 1.0)),
            BitNetDType::BitNet1 => Some((-1.0, 1.0)),
            BitNetDType::Int4 => Some((-8.0, 7.0)),
            BitNetDType::QInt8 => Some((-128.0, 127.0)),
            BitNetDType::QInt4 => Some((-8.0, 7.0)),
        }
    }
}

impl fmt::Display for BitNetDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for BitNetDType {
    fn default() -> Self {
        BitNetDType::F32
    }
}

/// Data type conversion error
#[derive(Debug, thiserror::Error)]
pub enum DataTypeError {
    /// Unsupported conversion between data types
    #[error("Unsupported conversion from {from} to {to}")]
    UnsupportedConversion { from: String, to: String },
    
    /// Value out of range for target data type
    #[error("Value {value} out of range for data type {dtype}")]
    ValueOutOfRange { value: f64, dtype: BitNetDType },
    
    /// Invalid data type for operation
    #[error("Invalid data type {dtype} for operation {operation}")]
    InvalidDataType { dtype: BitNetDType, operation: String },
}

/// Result type for data type operations
pub type DataTypeResult<T> = std::result::Result<T, DataTypeError>;

/// Data type utilities for conversion and validation
pub struct DataTypeUtils;

impl DataTypeUtils {
    /// Validates that a value is within the range for a data type
    ///
    /// # Arguments
    ///
    /// * `value` - Value to validate
    /// * `dtype` - Target data type
    ///
    /// # Returns
    ///
    /// Result indicating if the value is valid for the data type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetDType, DataTypeUtils};
    ///
    /// assert!(DataTypeUtils::validate_value_range(100.0, BitNetDType::U8).is_ok());
    /// assert!(DataTypeUtils::validate_value_range(-1.0, BitNetDType::U8).is_err());
    /// ```
    pub fn validate_value_range(value: f64, dtype: BitNetDType) -> DataTypeResult<()> {
        if let Some((min, max)) = dtype.value_range() {
            if value < min || value > max {
                return Err(DataTypeError::ValueOutOfRange { value, dtype });
            }
        }
        Ok(())
    }

    /// Checks if two data types are compatible for operations
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left-hand side data type
    /// * `rhs` - Right-hand side data type
    ///
    /// # Returns
    ///
    /// True if the data types are compatible
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetDType, DataTypeUtils};
    ///
    /// assert!(DataTypeUtils::are_compatible(BitNetDType::F32, BitNetDType::F32));
    /// assert!(!DataTypeUtils::are_compatible(BitNetDType::F32, BitNetDType::BitNet158));
    /// ```
    pub fn are_compatible(lhs: BitNetDType, rhs: BitNetDType) -> bool {
        match (lhs, rhs) {
            // Same types are always compatible
            (a, b) if a == b => true,
            
            // Floating point types are compatible with each other
            (BitNetDType::F32, BitNetDType::F16) |
            (BitNetDType::F16, BitNetDType::F32) |
            (BitNetDType::F32, BitNetDType::BF16) |
            (BitNetDType::BF16, BitNetDType::F32) => true,
            
            // Integer types with same signedness are compatible
            (a, b) if a.is_integer() && b.is_integer() && a.is_signed() == b.is_signed() => true,
            
            // BitNet quantized types are compatible with each other
            (a, b) if a.is_quantized() && b.is_quantized() => true,
            
            // Other combinations are not compatible
            _ => false,
        }
    }

    /// Promotes two data types to a common type for operations
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left-hand side data type
    /// * `rhs` - Right-hand side data type
    ///
    /// # Returns
    ///
    /// Common data type that can represent both inputs
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{BitNetDType, DataTypeUtils};
    ///
    /// let promoted = DataTypeUtils::promote_types(BitNetDType::F16, BitNetDType::F32);
    /// assert_eq!(promoted, BitNetDType::F32);
    /// ```
    pub fn promote_types(lhs: BitNetDType, rhs: BitNetDType) -> BitNetDType {
        use BitNetDType::*;
        
        match (lhs, rhs) {
            // Same types promote to themselves
            (a, b) if a == b => a,
            
            // Floating point promotion hierarchy: F32 > BF16 > F16
            (F32, _) | (_, F32) => F32,
            (BF16, F16) | (F16, BF16) | (BF16, _) | (_, BF16) => BF16,
            (F16, _) | (_, F16) => F16,
            
            // Integer promotion based on size and signedness
            (I64, _) | (_, I64) => I64,
            (U64, a) if !a.is_signed() => U64,
            (a, U64) if !a.is_signed() => U64,
            (I32, _) | (_, I32) => I32,
            (U32, a) if !a.is_signed() => U32,
            (a, U32) if !a.is_signed() => U32,
            (I16, _) | (_, I16) => I16,
            (U16, a) if !a.is_signed() => U16,
            (a, U16) if !a.is_signed() => U16,
            (I8, _) | (_, I8) => I8,
            
            // BitNet quantized types promote to the most general
            (BitNet158, _) | (_, BitNet158) => BitNet158,
            (BitNet11, _) | (_, BitNet11) => BitNet11,
            (BitNet1, _) | (_, BitNet1) => BitNet1,
            
            // Default fallback
            _ => F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(BitNetDType::F32.size_bytes(), Some(4));
        assert_eq!(BitNetDType::I8.size_bytes(), Some(1));
        assert_eq!(BitNetDType::U64.size_bytes(), Some(8));
        assert_eq!(BitNetDType::BitNet158.size_bytes(), Some(1));
    }

    #[test]
    fn test_dtype_properties() {
        assert!(BitNetDType::F32.is_floating_point());
        assert!(!BitNetDType::I32.is_floating_point());
        
        assert!(BitNetDType::I32.is_integer());
        assert!(!BitNetDType::F32.is_integer());
        
        assert!(BitNetDType::I32.is_signed());
        assert!(!BitNetDType::U32.is_signed());
        assert!(BitNetDType::F32.is_signed());
        
        assert!(BitNetDType::BitNet158.is_quantized());
        assert!(!BitNetDType::F32.is_quantized());
        
        assert!(BitNetDType::BitNet158.is_packed());
        assert!(!BitNetDType::F32.is_packed());
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(BitNetDType::F32.name(), "f32");
        assert_eq!(BitNetDType::BitNet158.name(), "bitnet1.58");
        assert_eq!(BitNetDType::QInt8.name(), "qint8");
    }

    #[test]
    fn test_candle_dtype_conversion() {
        assert_eq!(BitNetDType::F32.to_candle_dtype(), Some(candle_core::DType::F32));
        assert_eq!(BitNetDType::BitNet158.to_candle_dtype(), None);
        
        assert_eq!(BitNetDType::from_candle_dtype(candle_core::DType::F32), BitNetDType::F32);
        assert_eq!(BitNetDType::from_candle_dtype(candle_core::DType::F16), BitNetDType::F16);
    }

    #[test]
    fn test_value_range() {
        assert_eq!(BitNetDType::U8.value_range(), Some((0.0, 255.0)));
        assert_eq!(BitNetDType::BitNet158.value_range(), Some((-1.0, 1.0)));
        assert_eq!(BitNetDType::F32.value_range(), None);
    }

    #[test]
    fn test_value_range_validation() {
        assert!(DataTypeUtils::validate_value_range(100.0, BitNetDType::U8).is_ok());
        assert!(DataTypeUtils::validate_value_range(-1.0, BitNetDType::U8).is_err());
        assert!(DataTypeUtils::validate_value_range(0.5, BitNetDType::BitNet158).is_ok());
        assert!(DataTypeUtils::validate_value_range(2.0, BitNetDType::BitNet158).is_err());
    }

    #[test]
    fn test_compatibility() {
        assert!(DataTypeUtils::are_compatible(BitNetDType::F32, BitNetDType::F32));
        assert!(DataTypeUtils::are_compatible(BitNetDType::F32, BitNetDType::F16));
        assert!(!DataTypeUtils::are_compatible(BitNetDType::F32, BitNetDType::BitNet158));
        assert!(DataTypeUtils::are_compatible(BitNetDType::BitNet158, BitNetDType::BitNet1));
    }

    #[test]
    fn test_type_promotion() {
        assert_eq!(DataTypeUtils::promote_types(BitNetDType::F16, BitNetDType::F32), BitNetDType::F32);
        assert_eq!(DataTypeUtils::promote_types(BitNetDType::I8, BitNetDType::I32), BitNetDType::I32);
        assert_eq!(DataTypeUtils::promote_types(BitNetDType::U8, BitNetDType::U16), BitNetDType::U16);
        assert_eq!(DataTypeUtils::promote_types(BitNetDType::BitNet1, BitNetDType::BitNet158), BitNetDType::BitNet158);
    }

    #[test]
    fn test_default_types() {
        assert_eq!(BitNetDType::default_type(), BitNetDType::F32);
        assert_eq!(BitNetDType::default_quantized_type(), BitNetDType::BitNet158);
        assert_eq!(BitNetDType::default(), BitNetDType::F32);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BitNetDType::F32), "f32");
        assert_eq!(format!("{}", BitNetDType::BitNet158), "bitnet1.58");
    }

    #[test]
    fn test_all_types() {
        let all_types = BitNetDType::all_types();
        assert!(all_types.contains(&BitNetDType::F32));
        assert!(all_types.contains(&BitNetDType::BitNet158));
        assert!(all_types.len() > 10);
        
        let quantized_types = BitNetDType::quantized_types();
        assert!(quantized_types.contains(&BitNetDType::BitNet158));
        assert!(!quantized_types.contains(&BitNetDType::F32));
    }
}
