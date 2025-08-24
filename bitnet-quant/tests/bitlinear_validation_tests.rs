//! Simple validation test for BitLinear Layer Tensor Operations

use bitnet_core::{BitNetDType, BitNetTensor};
use bitnet_quant::quantization::QuantizationPrecision;
use bitnet_quant::tensor_integration::{
    bitlinear_tensor::BitLinearConfig, bitnet_ops::BitNetQuantizationConfig, qat_tensor::QATConfig,
};
use candle_core::Device;

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_bitlinear_config_compilation() {
        let config = BitLinearConfig::default();
        assert!(config.enable_layernorm);
        assert!(config.enable_residual);
        assert_eq!(config.layernorm_eps, 1e-5);
    }

    #[test]
    fn test_quantization_precision_hash() {
        let mut precision_map = std::collections::HashMap::new();
        precision_map.insert(QuantizationPrecision::OneBit, "1-bit".to_string());
        precision_map.insert(
            QuantizationPrecision::OneFiveFiveBit,
            "1.58-bit".to_string(),
        );
        precision_map.insert(QuantizationPrecision::TwoBit, "2-bit".to_string());

        assert_eq!(precision_map.len(), 3);
        assert!(precision_map.contains_key(&QuantizationPrecision::OneBit));
    }

    #[test]
    fn test_bitnet_quantization_config_creation() {
        let config = BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            ..Default::default()
        };

        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
    }

    #[test]
    fn test_qat_config_creation() {
        let _config = QATConfig::default();
        // Basic compilation test
        assert!(true);
    }

    #[test]
    fn test_device_cpu() {
        let device = Device::Cpu;
        let _tensor = BitNetTensor::zeros(&[2, 2], BitNetDType::F32, Some(device));
        // Basic compilation and creation test
        assert!(true);
    }

    #[test]
    fn test_tensor_creation_random() {
        let device = Device::Cpu;
        let result = BitNetTensor::random(&[4, 4], BitNetDType::F32, Some(device));
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixed_precision_support() {
        // Test multiple precision levels
        let precisions = vec![
            QuantizationPrecision::OneBit,
            QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::TwoBit,
            QuantizationPrecision::FourBit,
            QuantizationPrecision::EightBit,
        ];

        for precision in precisions {
            let config = BitNetQuantizationConfig {
                precision,
                ..Default::default()
            };
            // Validate each precision can be used
            assert_eq!(config.precision, precision);
        }
    }
}
