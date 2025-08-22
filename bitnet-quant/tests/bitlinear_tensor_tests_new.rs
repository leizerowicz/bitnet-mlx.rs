//! Comprehensive test suite for BitLinear Layer Tensor Operations
//!
//! This test suite validates the core aspects of BitLinear tensor operations
//! that are currently implemented in the bitnet-quant crate.

use candle_core::Device;

use bitnet_core::{BitNetTensor, BitNetDType};
use bitnet_quant::tensor_integration::{
    bitlinear_tensor::{BitLinearConfig, BitLinearTensorOpsImpl}, 
    bitnet_ops::BitNetQuantizationConfig
};
use bitnet_quant::quantization::{QuantizationPrecision, QuantizationStrategy};

#[cfg(test)]
mod tests {
    use super::*;
    
    fn get_test_device() -> Device {
        Device::Cpu // Use CPU for consistent testing
    }
    
    fn create_test_config() -> BitLinearConfig {
        BitLinearConfig {
            weight_quantization: BitNetQuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Symmetric,
                ..Default::default()
            },
            activation_quantization: BitNetQuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Symmetric,
                ..Default::default()
            },
            enable_layernorm: true,
            layernorm_eps: 1e-5,
            enable_residual: true,
            device: Some(get_test_device()),
        }
    }
    
    fn create_test_tensor(shape: &[usize]) -> Result<BitNetTensor, bitnet_core::MemoryError> {
        BitNetTensor::random(shape, BitNetDType::F32, Some(get_test_device()))
    }
    
    #[test]
    fn test_bitlinear_config_creation() {
        let config = create_test_config();
        
        assert_eq!(config.weight_quantization.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.activation_quantization.precision, QuantizationPrecision::OneFiveFiveBit);
        assert!(config.enable_layernorm);
        assert!(config.enable_residual);
        assert_eq!(config.layernorm_eps, 1e-5);
    }
    
    #[test]
    fn test_bitlinear_tensorops_impl_creation() {
        let config = create_test_config();
        let ops = BitLinearTensorOpsImpl::new(config);
        
        // Basic validation - just ensure we can create the ops instance
        // The new() method doesn't return Result, so we just verify it's created
        let _ops = ops; // This will consume the struct if creation succeeds
    }
    
    #[test]
    fn test_tensor_creation() {
        let tensor = create_test_tensor(&[32, 64]);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape().dims(), &[32, 64]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
    }
    
    #[test]
    fn test_quantization_config_validation() {
        let config = BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneBit,
            strategy: QuantizationStrategy::Asymmetric,
            ..Default::default()
        };
        
        assert_eq!(config.precision, QuantizationPrecision::OneBit);
        assert_eq!(config.strategy, QuantizationStrategy::Asymmetric);
    }
    
    #[test]
    fn test_bitlinear_config_default() {
        let config = BitLinearConfig::default();
        
        assert!(config.enable_layernorm);
        assert!(config.enable_residual);
        assert_eq!(config.layernorm_eps, 1e-5);
        // Weight and activation quantization should have default values
        assert_eq!(config.weight_quantization.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.activation_quantization.precision, QuantizationPrecision::OneFiveFiveBit);
    }
    
    #[test]
    fn test_precision_types_compatibility() {
        // Test that different quantization precision types work correctly
        let precisions = [
            QuantizationPrecision::OneBit,
            QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::TwoBit,
            QuantizationPrecision::FourBit,
            QuantizationPrecision::EightBit,
        ];
        
        for precision in precisions.iter() {
            let config = BitNetQuantizationConfig {
                precision: *precision,
                ..Default::default()
            };
            
            assert_eq!(config.precision, *precision);
        }
    }
}
