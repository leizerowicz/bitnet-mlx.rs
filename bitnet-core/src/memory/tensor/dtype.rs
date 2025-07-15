//! BitNet Data Types
//!
//! This module defines specialized data types for BitNet quantization,
//! supporting various precision levels from full precision (F32) down
//! to extreme quantization (I1) and the specialized BitNet 1.58b format.

use serde::{Deserialize, Serialize};
use std::fmt;
use candle_core::DType;

/// BitNet-specific data types supporting various quantization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitNetDType {
    /// 32-bit floating point (full precision)
    F32,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit signed integer
    I8,
    /// 4-bit signed integer (packed)
    I4,
    /// 2-bit signed integer (packed)
    I2,
    /// 1-bit signed integer (packed)
    I1,
    /// BitNet 1.58b format (ternary: -1, 0, +1)
    BitNet158,
}

impl BitNetDType {
    /// Returns the number of bits per element for this data type
    pub fn bits_per_element(&self) -> usize {
        match self {
            BitNetDType::F32 => 32,
            BitNetDType::F16 => 16,
            BitNetDType::BF16 => 16,
            BitNetDType::I8 => 8,
            BitNetDType::I4 => 4,
            BitNetDType::I2 => 2,
            BitNetDType::I1 => 1,
            BitNetDType::BitNet158 => 2, // Ternary values need 2 bits
        }
    }

    /// Returns the number of bytes required to store `count` elements
    pub fn bytes_for_elements(&self, count: usize) -> usize {
        let bits = self.bits_per_element() * count;
        (bits + 7) / 8 // Round up to nearest byte
    }

    /// Returns true if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16)
    }

    /// Returns true if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, BitNetDType::I8 | BitNetDType::I4 | BitNetDType::I2 | BitNetDType::I1)
    }

    /// Returns true if this is a quantized type (< 8 bits)
    pub fn is_quantized(&self) -> bool {
        matches!(self, BitNetDType::I4 | BitNetDType::I2 | BitNetDType::I1 | BitNetDType::BitNet158)
    }

    /// Returns true if this is the special BitNet 1.58b format
    pub fn is_bitnet158(&self) -> bool {
        matches!(self, BitNetDType::BitNet158)
    }

    /// Returns the range of values for integer types
    pub fn value_range(&self) -> Option<(i64, i64)> {
        match self {
            BitNetDType::I8 => Some((-128, 127)),
            BitNetDType::I4 => Some((-8, 7)),
            BitNetDType::I2 => Some((-2, 1)),
            BitNetDType::I1 => Some((-1, 0)), // Binary: -1, 0
            BitNetDType::BitNet158 => Some((-1, 1)), // Ternary: -1, 0, +1
            _ => None, // Floating point types don't have fixed ranges
        }
    }

    /// Converts to the closest candle DType for operations
    pub fn to_candle_dtype(&self) -> DType {
        match self {
            BitNetDType::F32 => DType::F32,
            BitNetDType::F16 => DType::F16,
            BitNetDType::BF16 => DType::BF16,
            BitNetDType::I8 => DType::I64, // Candle uses I64 for integer operations
            BitNetDType::I4 => DType::I64,
            BitNetDType::I2 => DType::I64,
            BitNetDType::I1 => DType::I64,
            BitNetDType::BitNet158 => DType::I64,
        }
    }

    /// Creates a BitNetDType from a candle DType
    pub fn from_candle_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(BitNetDType::F32),
            DType::F16 => Some(BitNetDType::F16),
            DType::BF16 => Some(BitNetDType::BF16),
            DType::I64 => Some(BitNetDType::I8), // Default to I8 for integers
            _ => None,
        }
    }

    /// Returns all available BitNet data types
    pub fn all_types() -> &'static [BitNetDType] {
        &[
            BitNetDType::F32,
            BitNetDType::F16,
            BitNetDType::BF16,
            BitNetDType::I8,
            BitNetDType::I4,
            BitNetDType::I2,
            BitNetDType::I1,
            BitNetDType::BitNet158,
        ]
    }

    /// Returns the memory efficiency factor compared to F32
    pub fn memory_efficiency(&self) -> f32 {
        32.0 / self.bits_per_element() as f32
    }

    /// Returns a human-readable description of the data type
    pub fn description(&self) -> &'static str {
        match self {
            BitNetDType::F32 => "32-bit floating point (full precision)",
            BitNetDType::F16 => "16-bit floating point (half precision)",
            BitNetDType::BF16 => "16-bit brain floating point",
            BitNetDType::I8 => "8-bit signed integer",
            BitNetDType::I4 => "4-bit signed integer (packed)",
            BitNetDType::I2 => "2-bit signed integer (packed)",
            BitNetDType::I1 => "1-bit signed integer (packed)",
            BitNetDType::BitNet158 => "BitNet 1.58b ternary format (-1, 0, +1)",
        }
    }
}

impl fmt::Display for BitNetDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            BitNetDType::F32 => "f32",
            BitNetDType::F16 => "f16",
            BitNetDType::BF16 => "bf16",
            BitNetDType::I8 => "i8",
            BitNetDType::I4 => "i4",
            BitNetDType::I2 => "i2",
            BitNetDType::I1 => "i1",
            BitNetDType::BitNet158 => "bitnet1.58",
        };
        write!(f, "{}", name)
    }
}

impl Default for BitNetDType {
    fn default() -> Self {
        BitNetDType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_per_element() {
        assert_eq!(BitNetDType::F32.bits_per_element(), 32);
        assert_eq!(BitNetDType::F16.bits_per_element(), 16);
        assert_eq!(BitNetDType::I8.bits_per_element(), 8);
        assert_eq!(BitNetDType::I4.bits_per_element(), 4);
        assert_eq!(BitNetDType::I2.bits_per_element(), 2);
        assert_eq!(BitNetDType::I1.bits_per_element(), 1);
        assert_eq!(BitNetDType::BitNet158.bits_per_element(), 2);
    }

    #[test]
    fn test_bytes_for_elements() {
        assert_eq!(BitNetDType::F32.bytes_for_elements(4), 16);
        assert_eq!(BitNetDType::I8.bytes_for_elements(4), 4);
        assert_eq!(BitNetDType::I4.bytes_for_elements(4), 2);
        assert_eq!(BitNetDType::I2.bytes_for_elements(4), 1);
        assert_eq!(BitNetDType::I1.bytes_for_elements(8), 1);
        assert_eq!(BitNetDType::I1.bytes_for_elements(9), 2); // Round up
    }

    #[test]
    fn test_type_classification() {
        assert!(BitNetDType::F32.is_float());
        assert!(BitNetDType::F16.is_float());
        assert!(!BitNetDType::I8.is_float());

        assert!(BitNetDType::I8.is_integer());
        assert!(BitNetDType::I4.is_integer());
        assert!(!BitNetDType::F32.is_integer());

        assert!(BitNetDType::I4.is_quantized());
        assert!(BitNetDType::I1.is_quantized());
        assert!(BitNetDType::BitNet158.is_quantized());
        assert!(!BitNetDType::I8.is_quantized());
        assert!(!BitNetDType::F32.is_quantized());

        assert!(BitNetDType::BitNet158.is_bitnet158());
        assert!(!BitNetDType::I1.is_bitnet158());
    }

    #[test]
    fn test_value_range() {
        assert_eq!(BitNetDType::I8.value_range(), Some((-128, 127)));
        assert_eq!(BitNetDType::I4.value_range(), Some((-8, 7)));
        assert_eq!(BitNetDType::I2.value_range(), Some((-2, 1)));
        assert_eq!(BitNetDType::I1.value_range(), Some((-1, 0)));
        assert_eq!(BitNetDType::BitNet158.value_range(), Some((-1, 1)));
        assert_eq!(BitNetDType::F32.value_range(), None);
    }

    #[test]
    fn test_candle_dtype_conversion() {
        assert_eq!(BitNetDType::F32.to_candle_dtype(), DType::F32);
        assert_eq!(BitNetDType::F16.to_candle_dtype(), DType::F16);
        assert_eq!(BitNetDType::I8.to_candle_dtype(), DType::I64);

        assert_eq!(BitNetDType::from_candle_dtype(DType::F32), Some(BitNetDType::F32));
        assert_eq!(BitNetDType::from_candle_dtype(DType::F16), Some(BitNetDType::F16));
        assert_eq!(BitNetDType::from_candle_dtype(DType::I64), Some(BitNetDType::I8));
    }

    #[test]
    fn test_memory_efficiency() {
        assert_eq!(BitNetDType::F32.memory_efficiency(), 1.0);
        assert_eq!(BitNetDType::F16.memory_efficiency(), 2.0);
        assert_eq!(BitNetDType::I8.memory_efficiency(), 4.0);
        assert_eq!(BitNetDType::I4.memory_efficiency(), 8.0);
        assert_eq!(BitNetDType::I1.memory_efficiency(), 32.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BitNetDType::F32), "f32");
        assert_eq!(format!("{}", BitNetDType::I4), "i4");
        assert_eq!(format!("{}", BitNetDType::BitNet158), "bitnet1.58");
    }

    #[test]
    fn test_all_types() {
        let types = BitNetDType::all_types();
        assert_eq!(types.len(), 8);
        assert!(types.contains(&BitNetDType::F32));
        assert!(types.contains(&BitNetDType::BitNet158));
    }

    #[test]
    fn test_serialization() {
        let dtype = BitNetDType::BitNet158;
        let serialized = serde_json::to_string(&dtype).unwrap();
        let deserialized: BitNetDType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(dtype, deserialized);
    }
}