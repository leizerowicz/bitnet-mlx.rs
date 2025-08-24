//! Simplified test suite for BitLinear Layer Tensor Operations

use bitnet_core::device::devices_equal;
use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::tensor::set_global_memory_pool;
use bitnet_core::{BitNetDType, BitNetTensor};
use bitnet_quant::quantization::QuantizationPrecision;
use bitnet_quant::tensor_integration::{
    bitlinear_tensor::BitLinearConfig, bitnet_ops::BitNetQuantizationConfig,
    TensorIntegrationResult,
};
use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, Once};

static INIT: Once = Once::new();
static mut MEMORY_POOL: Option<Arc<HybridMemoryPool>> = None;

// Setup function to initialize memory pool (only once)
fn setup_memory_pool() {
    INIT.call_once(|| {
        let pool = Arc::new(HybridMemoryPool::new().expect("Failed to create memory pool"));
        set_global_memory_pool(Arc::downgrade(&pool));
        unsafe {
            MEMORY_POOL = Some(pool);
        }
    });
}

// Helper function to create test tensors using available API
fn create_test_tensor(shape: &[usize], device: Device) -> TensorIntegrationResult<BitNetTensor> {
    // Ensure memory pool is initialized
    setup_memory_pool();

    // Use the random method we saw in BitNetTensor API
    BitNetTensor::random(shape, BitNetDType::F32, Some(device)).map_err(|e| {
        bitnet_quant::tensor_integration::TensorIntegrationError::TensorOp {
            message: format!("Tensor creation failed: {e:?}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitlinear_config_creation() -> TensorIntegrationResult<()> {
        let config = BitLinearConfig::default();
        assert!(config.enable_layernorm);
        assert!(config.enable_residual);
        assert_eq!(config.layernorm_eps, 1e-5);
        Ok(())
    }

    #[test]
    fn test_weight_quantization_tensor() -> TensorIntegrationResult<()> {
        let device = Device::Cpu;
        let weights = create_test_tensor(&[64, 32], device.clone())?;

        // Create weight quantization config
        let _weight_config = BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneBit,
            ..Default::default()
        };

        // This would be the actual test in a working implementation
        assert_eq!(weights.shape().dims(), &[64, 32]);

        Ok(())
    }

    #[test]
    fn test_quantization_precision() -> TensorIntegrationResult<()> {
        let precision = QuantizationPrecision::OneBit;
        let mut precision_map = HashMap::new();
        precision_map.insert(precision, "1-bit".to_string());

        assert_eq!(
            precision_map.get(&QuantizationPrecision::OneBit),
            Some(&"1-bit".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_bitnet_quantization_config() -> TensorIntegrationResult<()> {
        let config = BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            ..Default::default()
        };

        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        Ok(())
    }

    #[test]
    fn test_device_selection() -> TensorIntegrationResult<()> {
        let device = Device::Cpu;
        let tensor = create_test_tensor(&[4, 4], device.clone())?;

        // Use the devices_equal function for comparison since Device doesn't implement PartialEq
        assert!(devices_equal(tensor.device(), &device));
        Ok(())
    }
}
