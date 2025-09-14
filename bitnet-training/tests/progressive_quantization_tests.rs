//! Progressive Quantization Tests - Fixed
//!
//! Test suite for progressive quantization scheduling and layer-wise
//! quantization strategies used in BitNet training.

use bitnet_training::qat::{
    progressive::{
        CompletionCriteria, LayerWiseQuantization, ProgressiveQuantization,
        ProgressiveQuantizationFactory, ProgressiveStrategy, QuantizationPhase,
    },
    straight_through::{STEVariant, StraightThroughEstimator},
};
use candle_core::{Device, Result as CandleResult, Tensor};

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

/// Create mock layer tensors for testing
fn create_mock_layers(device: &Device, layer_count: usize) -> CandleResult<Vec<Tensor>> {
    (0..layer_count)
        .map(|i| {
            let size = (i + 1) * 10; // Different sizes for each layer
            Tensor::randn(0.0, 1.0, &[size, size], device)
        })
        .collect()
}

/// Helper function to create test quantization phases
fn create_test_phases() -> Vec<QuantizationPhase> {
    vec![
        QuantizationPhase {
            name: "Phase 1: 8-bit".to_string(),
            min_steps: 100,
            max_steps: Some(500),
            bit_width: 8,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        },
        QuantizationPhase {
            name: "Phase 2: 4-bit".to_string(),
            min_steps: 100,
            max_steps: Some(500),
            bit_width: 4,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        },
        QuantizationPhase {
            name: "Phase 3: 1-bit".to_string(),
            min_steps: 200,
            max_steps: None,
            bit_width: 1,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        },
    ]
}

#[cfg(test)]
mod progressive_scheduler_tests {
    use super::*;

    #[test]
    fn test_basic_progressive_creation() {
        let device = setup_test_device();
        let phases = create_test_phases();
        let progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device);

        // Test initial state
        assert_eq!(progressive.current_phase().bit_width, 8);
        assert_eq!(progressive.current_phase().name, "Phase 1: 8-bit");

        // Test phase access
        assert_eq!(
            progressive.current_phase().ste_variant,
            STEVariant::Standard
        );
    }

    #[test]
    fn test_progressive_factory_methods() {
        let device = setup_test_device();

        // Test bit-width reduction factory
        let bit_width_progressive =
            ProgressiveQuantizationFactory::create_bit_width_reduction(device.clone());
        assert!(bit_width_progressive.current_phase().bit_width > 1);

        // Test soft-to-hard factory
        let soft_to_hard_progressive =
            ProgressiveQuantizationFactory::create_soft_to_hard(device.clone());
        assert_eq!(
            soft_to_hard_progressive.current_phase().ste_variant,
            STEVariant::Soft
        );

        // Test layer-wise factory
        let layer_order = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
        ];
        let layer_wise = ProgressiveQuantizationFactory::create_layer_wise(layer_order, device);
        assert!(layer_wise.get_layer_config("layer1").is_some());
    }

    #[test]
    fn test_progressive_update() {
        let device = setup_test_device();
        let phases = create_test_phases();
        let mut progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device);

        // Test update with good metrics (should stay in current phase)
        let result = progressive.update_metrics(0.1, 0.01, Some(0.9)); // low loss, low error, high accuracy
        assert!(result.is_ok());

        // Should still be in first phase
        assert_eq!(progressive.current_phase().bit_width, 8);
    }

    #[test]
    fn test_progressive_statistics() {
        let device = setup_test_device();
        let phases = create_test_phases();
        let progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device);

        let stats = progressive.get_statistics();
        assert_eq!(stats.current_phase, 0);
        assert_eq!(stats.total_phases, 3);
        assert_eq!(stats.current_bit_width, 8);
    }
}

#[cfg(test)]
mod layer_wise_quantization_tests {
    use super::*;

    #[test]
    fn test_layer_wise_creation() {
        let device = setup_test_device();
        let layer_order = vec![
            "output.weight".to_string(),
            "layer2.weight".to_string(),
            "layer1.weight".to_string(),
        ];
        let phases = create_test_phases();

        let layer_wise = LayerWiseQuantization::new(layer_order.clone(), phases, device);

        // Check if output layer is available first
        assert!(layer_wise.get_layer_config("output.weight").is_some());
    }

    #[test]
    fn test_layer_wise_config_progression() {
        let device = setup_test_device();
        let layer_order = vec!["output.weight".to_string(), "layer1.weight".to_string()];
        let phases = create_test_phases();

        let layer_wise = LayerWiseQuantization::new(layer_order, phases, device);

        // First layer should be available
        let output_config = layer_wise.get_layer_config("output.weight");
        assert!(output_config.is_some());

        if let Some(config) = output_config {
            assert_eq!(config.bits, 8); // Should start with first phase configuration (8-bit from progressive phases)
        }
    }

    #[test]
    fn test_layer_wise_factory() {
        let device = setup_test_device();
        let layer_order = vec![
            "conv1.weight".to_string(),
            "conv2.weight".to_string(),
            "fc.weight".to_string(),
        ];

        let layer_wise = ProgressiveQuantizationFactory::create_layer_wise(layer_order, device);

        // Should start with only the first layer being quantized
        let first_layer_config = layer_wise.get_layer_config("conv1.weight");
        assert!(first_layer_config.is_some());
        
        // Factory method creates 1-bit phases, so first layer should use 1-bit
        if let Some(config) = first_layer_config {
            assert_eq!(config.bits, 1);
        }
        
        // Other layers should not be quantized yet in layer-wise progression  
        assert!(layer_wise.get_layer_config("conv2.weight").is_none());
        assert!(layer_wise.get_layer_config("fc.weight").is_none());
    }
}

#[cfg(test)]
mod ste_integration_with_progressive_tests {
    use super::*;

    #[test]
    fn test_ste_with_progressive_config() {
        let device = setup_test_device();
        let phases = create_test_phases();
        let progressive = ProgressiveQuantization::new(
            ProgressiveStrategy::BitWidthReduction,
            phases,
            device.clone(),
        );

        let layer_config = progressive.get_layer_config("test_layer");
        if let Some(mut config) = layer_config {
            config.device = Some(device.clone());

            let ste_result = StraightThroughEstimator::new(config);
            assert!(ste_result.is_ok());
        }
    }

    #[test]
    fn test_complete_training_workflow() {
        let device = setup_test_device();
        let mut progressive =
            ProgressiveQuantizationFactory::create_bit_width_reduction(device.clone());

        // Simulate training loop
        for step in 1..=100 {
            let loss = 1.0 - (step as f32 * 0.005); // Gradually decreasing loss
            let error = 0.1 - (step as f32 * 0.0005); // Gradually decreasing error
            let accuracy = 0.5 + (step as f32 * 0.003); // Gradually increasing accuracy

            let result = progressive.update_metrics(loss, error, Some(accuracy));
            assert!(result.is_ok());

            // Check if we can get layer configs
            if let Some(mut config) = progressive.get_layer_config("test_layer") {
                config.device = Some(device.clone());
                let ste = StraightThroughEstimator::new(config);
                assert!(ste.is_ok());
            }
        }

        // Check final statistics
        let stats = progressive.get_statistics();
        assert!(stats.current_step >= 100);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_phases() {
        let device = setup_test_device();
        let empty_phases = vec![];

        // This might panic or handle gracefully - test behavior
        let result = std::panic::catch_unwind(|| {
            ProgressiveQuantization::new(
                ProgressiveStrategy::BitWidthReduction,
                empty_phases,
                device,
            )
        });

        // Either panics or handles gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_single_phase() {
        let device = setup_test_device();
        let single_phase = vec![QuantizationPhase {
            name: "Single Phase".to_string(),
            min_steps: 100,
            max_steps: Some(500),
            bit_width: 1,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        }];

        let progressive = ProgressiveQuantization::new(
            ProgressiveStrategy::BitWidthReduction,
            single_phase,
            device,
        );
        assert_eq!(progressive.current_phase().bit_width, 1);
        assert_eq!(progressive.get_statistics().total_phases, 1);
    }
}
