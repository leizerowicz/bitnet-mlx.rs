//! QAT Training Tests - Fixed Version
//!
//! Comprehensive test suite for QAT training functionality, including optimizer integration,
//! progressive quantization scheduling, and state tracking.

use bitnet_training::qat::optimizer::ParamsAdam;
use bitnet_training::qat::{
    CompletionCriteria, ProgressiveQuantization, ProgressiveStrategy, QATAdam, QATTrainingState,
    QuantizationPhase, STEConfig, STEVariant, StraightThroughEstimator,
};
use candle_core::{Device, Result as CandleResult, Tensor};

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

/// Create test weights tensor
fn create_test_weights(device: &Device) -> CandleResult<Tensor> {
    Tensor::randn(0f32, 1f32, &[10, 10], device)
}

#[cfg(test)]
mod qat_training_basic {
    use super::*;

    #[test]
    fn test_qat_adam_optimizer_creation() {
        let device = setup_test_device();
        let weights = create_test_weights(&device).unwrap();

        let _optimizer = QATAdam::new(vec![weights.clone()], ParamsAdam::default()).unwrap();

        // Test that optimizer was created successfully - no vars() method available
        // Just verify it was constructed without error
        assert!(true); // Optimizer created successfully
    }

    #[test]
    fn test_ste_integration() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Standard,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();
        let input = create_test_weights(&device).unwrap();

        let quantized = ste.forward_quantized(&input).unwrap();
        assert_eq!(quantized.dims(), input.dims());
    }

    #[test]
    fn test_training_state() {
        let mut state = QATTrainingState::new();

        // Test state updates (need at least 5 losses for convergence check)
        state.record_loss(1.5);
        state.record_loss(1.0);
        state.record_loss(0.8);
        state.record_loss(0.7);
        state.record_loss(0.6);

        // Test convergence check (method takes no arguments)
        assert!(state.is_converging());
    }
}

#[cfg(test)]
mod progressive_quantization {
    use super::*;

    #[test]
    fn test_progressive_quantization_creation() {
        let device = setup_test_device();

        // Create quantization phases using struct initialization
        let phase = QuantizationPhase {
            name: "test_phase".to_string(),
            min_steps: 100,
            max_steps: Some(1000),
            bit_width: 1,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        };

        let phases = vec![phase];

        let progressive = ProgressiveQuantization::new(
            ProgressiveStrategy::BitWidthReduction,
            phases,
            device.clone(),
        );

        // Test that progressive quantization was created successfully
        assert_eq!(progressive.current_phase().bit_width, 1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_training_setup() {
        let device = setup_test_device();
        let weights = create_test_weights(&device).unwrap();

        // STE configuration
        let config = STEConfig {
            variant: STEVariant::Standard,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        let mut ste = StraightThroughEstimator::new(config).unwrap();

        // Optimizer setup
        let _optimizer = QATAdam::new(vec![weights.clone()], ParamsAdam::default()).unwrap();

        // Progressive quantization setup with proper phase structure
        let phase = QuantizationPhase {
            name: "initial".to_string(),
            min_steps: 50,
            max_steps: Some(100),
            bit_width: 1,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        };

        let progressive = ProgressiveQuantization::new(
            ProgressiveStrategy::BitWidthReduction,
            vec![phase],
            device.clone(),
        );

        // Training state
        let mut state = QATTrainingState::new();
        // Record sufficient decreasing losses for convergence
        state.record_loss(1.0);
        state.record_loss(0.9);
        state.record_loss(0.8);
        state.record_loss(0.7);
        state.record_loss(0.6);

        // Test quantization
        let quantized = ste.forward_quantized(&weights).unwrap();
        assert_eq!(quantized.dims(), weights.dims());

        // Verify components work together
        assert_eq!(progressive.current_phase().bit_width, 1);
        assert!(state.is_converging());
    }

    #[test]
    fn test_state_tracking_convergence() {
        let mut state = QATTrainingState::new();

        // Record decreasing losses to simulate convergence (need at least 5 entries)
        state.record_loss(2.0);
        state.record_loss(1.8);
        state.record_loss(1.6);
        state.record_loss(1.5);
        state.record_loss(1.4);

        assert!(state.is_converging());

        // Record non-converging pattern (make it clearly non-converging)
        state.record_loss(2.0);
        state.record_loss(2.5);
        state.record_loss(3.0);
        assert!(!state.is_converging());
    }

    #[test]
    fn test_config_validation() {
        let device = setup_test_device();

        let config = STEConfig {
            variant: STEVariant::Clipped,
            bits: 1,
            range: 1.0,
            temperature: 1.0,
            device: Some(device.clone()),
            ..STEConfig::default()
        };

        // Should create successfully with valid config
        let ste = StraightThroughEstimator::new(config);
        assert!(ste.is_ok());
    }

    #[test]
    fn test_quantization_variants() {
        let device = setup_test_device();

        let variants = vec![
            STEVariant::Standard,
            STEVariant::Clipped,
            STEVariant::Soft,
            STEVariant::Learnable,
            STEVariant::Learned,
            STEVariant::Adaptive,
        ];

        for variant in variants {
            let config = STEConfig {
                variant,
                device: Some(device.clone()),
                ..STEConfig::default()
            };

            let ste = StraightThroughEstimator::new(config);
            assert!(
                ste.is_ok(),
                "Failed to create STE with variant: {variant:?}"
            );
        }
    }

    #[test]
    fn test_multi_precision_quantization() {
        let device = setup_test_device();

        for bits in [1, 2, 4, 8] {
            let config = STEConfig {
                variant: STEVariant::Standard,
                bits,
                device: Some(device.clone()),
                ..STEConfig::default()
            };

            let mut ste = StraightThroughEstimator::new(config).unwrap();
            let input = create_test_weights(&device).unwrap();

            let quantized = ste.forward_quantized(&input).unwrap();
            assert_eq!(quantized.dims(), input.dims());
        }
    }
}
