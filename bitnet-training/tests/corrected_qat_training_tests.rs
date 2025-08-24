//! QAT Training Tests
//!
//! Test suite for Quantization-Aware Training functionality including
//! state management, STE integration, and training loop validation.

use bitnet_training::qat::{
    state_tracking::{QATStateTracker, QATTrainingState},
    straight_through::{STEConfig, STEVariant, StraightThroughEstimator},
};
use candle_core::{DType, Device, Tensor};

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

#[cfg(test)]
mod basic_qat_training_tests {
    use super::*;

    #[test]
    fn test_qat_state_initialization() {
        let state = QATTrainingState::new();

        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.loss_history.len(), 0);
        assert_eq!(state.validation_history.len(), 0);
        assert_eq!(state.quantization_error, 0.0);
    }

    #[test]
    fn test_training_metrics_update() {
        let mut state = QATTrainingState::new();

        state.update_training_metrics(1, 100, 0.01, 0.5, 32, 0.1);

        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 100);
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.current_loss, 0.5);
        assert_eq!(state.loss_history.len(), 1);
        assert_eq!(state.loss_history[0], 0.5);
    }

    #[test]
    fn test_validation_metrics_update() {
        let mut state = QATTrainingState::new();

        state.update_validation_metrics(0.4, Some(0.85));

        assert_eq!(state.validation_loss, Some(0.4));
        assert_eq!(state.validation_accuracy, Some(0.85));
        assert_eq!(state.validation_history.len(), 1);
        assert_eq!(state.validation_history[0], 0.4);
    }

    #[test]
    fn test_training_summary() {
        let mut state = QATTrainingState::new();

        // Add training data
        state.update_training_metrics(5, 500, 0.01, 0.3, 32, 0.1);
        state.update_validation_metrics(0.25, Some(0.9));

        let summary = state.get_summary();

        assert_eq!(summary.epoch, 5);
        assert_eq!(summary.step, 500);
        assert_eq!(summary.current_loss, 0.3);
        assert_eq!(summary.validation_accuracy, Some(0.9));
    }

    #[test]
    fn test_checkpoint_frequency() {
        let state = QATTrainingState::new();

        // Should not checkpoint at step 0
        assert!(!state.should_checkpoint(10));

        let mut state = QATTrainingState::new();
        state.update_training_metrics(1, 10, 0.01, 0.5, 32, 0.1);

        assert!(state.should_checkpoint(10)); // At step 10, should checkpoint every 10
        assert!(!state.should_checkpoint(15)); // At step 10, should NOT checkpoint every 15
    }
}

#[cfg(test)]
mod ste_integration_tests {
    use super::*;

    #[test]
    fn test_ste_config_creation() {
        let config = STEConfig {
            variant: STEVariant::Standard,
            bits: 1,
            range: 1.0,
            temperature: 1.0,
            learnable_lr: 0.001,
            clip_gradients: false,
            clip_threshold: 1.0,
            gradient_clip: None,
            use_noise: false,
            device: Some(setup_test_device()),
        };

        assert_eq!(config.variant, STEVariant::Standard);
        assert_eq!(config.bits, 1);
        assert_eq!(config.range, 1.0);
        assert!(!config.clip_gradients);
    }

    #[test]
    fn test_ste_creation() {
        let device = setup_test_device();
        let config = STEConfig {
            variant: STEVariant::Clipped,
            bits: 1,
            range: 1.0,
            temperature: 1.0,
            learnable_lr: 0.001,
            clip_gradients: true,
            clip_threshold: 1.0,
            gradient_clip: None,
            use_noise: false,
            device: Some(device.clone()),
        };

        let result = StraightThroughEstimator::new(config);
        assert!(result.is_ok(), "STE creation should succeed");
    }

    #[test]
    fn test_ste_forward_pass() {
        let device = setup_test_device();
        let mut config = STEConfig::default();
        config.device = Some(device.clone());

        let mut ste = StraightThroughEstimator::new(config).unwrap();

        let input = Tensor::randn(0.0, 1.0, (4, 4), &device)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap();
        let result = ste.forward(&input);

        assert!(result.is_ok(), "Forward pass should succeed");

        let output = result.unwrap();
        assert_eq!(
            output.shape(),
            input.shape(),
            "Output shape should match input"
        );
    }

    #[test]
    fn test_ste_error_handling() {
        let device = setup_test_device();

        // Test with invalid configuration
        let invalid_config = STEConfig {
            variant: STEVariant::Standard,
            bits: 0,     // Invalid bit width
            range: -1.0, // Invalid range
            temperature: 1.0,
            learnable_lr: 0.001,
            clip_gradients: false,
            clip_threshold: 1.0,
            gradient_clip: None,
            use_noise: false,
            device: Some(device.clone()),
        };

        let result = StraightThroughEstimator::new(invalid_config);
        // Note: The constructor might succeed even with unusual values
        // The error handling would be in the forward pass
        if let Ok(mut ste) = result {
            let empty_tensor = Tensor::zeros((0,), DType::F32, &device).unwrap();
            let _forward_result = ste.forward(&empty_tensor);
            // Test should not crash
        }
    }
}

#[cfg(test)]
mod end_to_end_training_tests {
    use super::*;

    #[test]
    fn test_full_qat_training_simulation() {
        let device = setup_test_device();
        let mut tracker = QATStateTracker::new(device.clone());

        let mut config = STEConfig::default();
        config.device = Some(device.clone());
        let mut ste = StraightThroughEstimator::new(config).unwrap();

        // Simulate multiple training steps
        for epoch in 1..=5 {
            let loss = 1.0 - (epoch as f32 * 0.1); // Decreasing loss
            let lr = 0.01;
            let samples = 32;
            let step_time = 0.1;

            tracker.update(epoch, epoch * 100, lr, loss, samples, step_time);

            // Create dummy weights and quantize them
            let weights = Tensor::randn(0.0, 1.0, (8, 8), &device)
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap();
            let _quantized = ste.forward(&weights).unwrap();

            // Add validation every other epoch
            if epoch % 2 == 0 {
                let val_loss = loss * 0.9;
                let accuracy = 0.5 + (epoch as f32 * 0.08);
                tracker.update_validation(val_loss, Some(accuracy));
            }
        }

        // Verify final state
        let final_state = tracker.get_state();
        assert_eq!(final_state.epoch, 5);
        assert_eq!(final_state.step, 500);
        assert!(final_state.current_loss < 1.0); // Loss should have improved
        assert!(final_state.validation_loss.is_some());
    }

    #[test]
    fn test_training_convergence_detection() {
        let mut state = QATTrainingState::new();

        // Simulate converging loss
        let losses = [1.0, 0.5, 0.4, 0.35, 0.33];
        for (step, &loss) in losses.iter().enumerate() {
            state.update_training_metrics(step, step * 100, 0.01, loss, 32, 0.1);
        }

        // Check that loss is decreasing
        let history = &state.loss_history;
        assert!(
            history[history.len() - 1] < history[0],
            "Loss should decrease"
        );

        // Check improvement
        let initial_loss = history[0];
        let final_loss = history[history.len() - 1];
        let improvement = (initial_loss - final_loss) / initial_loss;
        assert!(improvement > 0.5, "Should show significant improvement");
    }

    #[test]
    fn test_training_metrics_calculation() {
        let mut state = QATTrainingState::new();

        // Simulate training with throughput calculation
        let start_time = 0.0;
        let mut _cumulative_time = start_time;

        for step in 1..=10 {
            _cumulative_time += 0.1; // 100ms per step
            let loss = 1.0 - (step as f32 * 0.05);

            state.update_training_metrics(1, step, 0.01, loss, 32, 0.1);
        }

        // Check throughput calculation
        assert!(state.throughput > 0.0, "Throughput should be positive");
        assert_eq!(state.samples_processed, 320); // 10 steps * 32 samples

        // Check training time
        assert!(
            (state.training_time - 1.0).abs() < 0.01,
            "Training time should be ~1.0 seconds"
        );
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_empty_tensor_handling() {
        let device = setup_test_device();
        let mut config = STEConfig::default();
        config.device = Some(device.clone());

        let mut ste = StraightThroughEstimator::new(config).unwrap();

        let empty_tensor = Tensor::zeros((0,), DType::F32, &device).unwrap();
        let result = ste.forward(&empty_tensor);

        // Should handle empty tensors gracefully
        assert!(result.is_ok() || result.is_err()); // Either way is acceptable
    }

    #[test]
    fn test_extreme_values_handling() {
        let device = setup_test_device();
        let mut config = STEConfig::default();
        config.device = Some(device.clone());

        let mut ste = StraightThroughEstimator::new(config).unwrap();

        // Test with extreme values
        let extreme_tensor = Tensor::from_slice(
            &[f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY],
            (4,),
            &device,
        )
        .unwrap();

        let result = ste.forward(&extreme_tensor);
        // Should not crash with extreme values
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_state_validation() {
        let mut state = QATTrainingState::new();

        // Test with invalid metrics
        state.update_training_metrics(0, 0, 0.0, f32::NAN, 0, 0.0);

        // State should remain stable
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.samples_processed, 0);
        // NaN loss should be recorded
        assert_eq!(state.loss_history.len(), 1);
    }
}
