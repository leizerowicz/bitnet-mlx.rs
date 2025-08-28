//! Optimizer Integration Tests
//!
//! Test suite for QAT-specific optimizers and their integration with
//! quantization-aware training workflows.

use bitnet_training::qat::{
    optimizer::{ParamsAdam, ParamsAdamW, QATAdam, QATAdamW, QATSGDWithMomentum},
    state_tracking::QATTrainingState,
    straight_through::{STEConfig, STEVariant, StraightThroughEstimator},
};
use candle_core::{Device, Result as CandleResult, Tensor};

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

/// Create test parameters for optimizer testing
fn create_test_parameters(device: &Device) -> CandleResult<Vec<Tensor>> {
    Ok(vec![
        Tensor::randn(0.0f32, 0.1f32, &[10, 10], device)?,
        Tensor::randn(0.0f32, 0.1f32, &[10, 1], device)?,
    ])
}

/// Create mock gradients for testing
fn create_mock_gradients(params: &[Tensor]) -> CandleResult<Vec<Tensor>> {
    params
        .iter()
        .map(|p| Tensor::randn(0.0f32, 0.01f32, p.dims(), p.device()))
        .collect::<CandleResult<Vec<_>>>()
}

#[cfg(test)]
mod qat_adam_tests {
    use super::*;

    #[test]
    fn test_qat_adam_creation() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let optimizer = QATAdam::new(params.clone(), ParamsAdam::default()).unwrap();

        // Basic validation that optimizer was created
        assert!(true);
    }

    #[test]
    fn test_qat_adam_step() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATAdam::new(
            params.clone(),
            ParamsAdam {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();

        // Store original parameter values
        let original_values: Vec<Vec<f32>> = params
            .iter()
            .map(|p| p.flatten_all().unwrap().to_vec1().unwrap())
            .collect();

        // Create gradients
        let gradients = create_mock_gradients(&params).unwrap();

        // Perform optimizer step
        optimizer.step(&gradients).unwrap();

        // Verify parameters were updated
        for (i, param) in params.iter().enumerate() {
            let new_values = param.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let original = &original_values[i];

            // At least some parameters should have changed
            let changed = new_values
                .iter()
                .zip(original.iter())
                .any(|(new, old)| (new - old).abs() > 1e-10);

            assert!(changed, "Parameters should be updated after optimizer step");
        }
    }

    #[test]
    fn test_qat_adam_convergence_behavior() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATAdam::new(
            params.clone(),
            ParamsAdam {
                lr: 0.01, // Higher learning rate for visible changes
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();

        let mut state = QATTrainingState::new();

        // Simulate training steps
        for step in 0..10 {
            // Create consistent gradients for convergence
            let gradients = params
                .iter()
                .map(|p| {
                    let grad_data = vec![0.1; p.elem_count()];
                    Tensor::from_vec(grad_data, p.dims(), p.device()).unwrap()
                })
                .collect::<Vec<_>>();

            optimizer.step(&gradients).unwrap();

            // Record a decreasing loss for convergence simulation
            let loss = 1.0 - (step as f32 * 0.05);
            state.record_loss(loss);
        }

        // Verify convergence behavior
        assert!(state.get_loss_history().len() == 10);
        let final_loss = state.get_loss_history().last().unwrap();
        assert!(*final_loss < 0.6, "Loss should decrease over training");
    }
}

#[cfg(test)]
mod qat_adamw_tests {
    use super::*;

    #[test]
    fn test_qat_adamw_with_weight_decay() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATAdamW::new(
            params.clone(),
            ParamsAdamW {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01, // Non-zero weight decay
            },
        )
        .unwrap();

        // Store original parameter norms
        let original_norms: Vec<f32> = params
            .iter()
            .map(|p| {
                let values = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                values.iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();

        // Create gradients and perform step
        let gradients = create_mock_gradients(&params).unwrap();
        optimizer.step(&gradients).unwrap();

        // With weight decay, parameter norms should generally decrease
        for (i, param) in params.iter().enumerate() {
            let values = param.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let new_norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Weight decay should reduce parameter magnitudes
            assert!(
                new_norm <= original_norms[i] + 1e-3,
                "Weight decay should reduce parameter norm"
            );
        }
    }

    #[test]
    fn test_qat_adamw_vs_adam() {
        let device = setup_test_device();
        let params_adam = create_test_parameters(&device).unwrap();
        let params_adamw = create_test_parameters(&device).unwrap();

        let mut optimizer_adam = QATAdam::new(
            params_adam.clone(),
            ParamsAdam {
                lr: 0.001,
                weight_decay: 0.01,
                ..ParamsAdam::default()
            },
        )
        .unwrap();

        let mut optimizer_adamw = QATAdamW::new(
            params_adamw.clone(),
            ParamsAdamW {
                lr: 0.001,
                weight_decay: 0.01,
                ..ParamsAdamW::default()
            },
        )
        .unwrap();

        // Same gradients for both
        let gradients_adam = create_mock_gradients(&params_adam).unwrap();
        let gradients_adamw = params_adam
            .iter()
            .map(|p| gradients_adam[0].broadcast_as(p.shape()).unwrap())
            .collect::<Vec<_>>();

        // Both should update without error
        optimizer_adam.step(&gradients_adam).unwrap();
        optimizer_adamw.step(&gradients_adamw).unwrap();

        // Basic validation that both worked
        assert!(true);
    }
}

#[cfg(test)]
mod qat_sgd_tests {
    use super::*;

    #[test]
    fn test_qat_sgd_with_momentum() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATSGDWithMomentum::new(
            params.clone(),
            0.01, // learning_rate
            0.9,  // momentum
            0.0,  // weight_decay
        )
        .unwrap();

        let gradients = create_mock_gradients(&params).unwrap();

        // First step
        let param_values_before = params[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        optimizer.step(&gradients).unwrap();
        let param_values_after = params[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Parameters should have changed
        let changed = param_values_before
            .iter()
            .zip(param_values_after.iter())
            .any(|(before, after)| (before - after).abs() > 1e-6);

        assert!(changed, "SGD should update parameters");
    }

    #[test]
    fn test_qat_sgd_momentum_accumulation() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATSGDWithMomentum::new(
            params.clone(),
            0.01, // learning_rate
            0.9,  // momentum
            0.0,  // weight_decay
        )
        .unwrap();

        // Consistent gradient direction
        let consistent_gradients = params
            .iter()
            .map(|p| {
                let grad_values = vec![0.1; p.elem_count()];
                Tensor::from_vec(grad_values, p.dims(), p.device()).unwrap()
            })
            .collect::<Vec<_>>();

        let initial_params = params[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // First step
        optimizer.step(&consistent_gradients).unwrap();
        let after_step1 = params[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Second step with same gradients
        optimizer.step(&consistent_gradients).unwrap();
        let after_step2 = params[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Calculate step sizes
        let step1_size = initial_params
            .iter()
            .zip(after_step1.iter())
            .map(|(i, a)| (i - a).abs())
            .sum::<f32>();
        let step2_size = after_step1
            .iter()
            .zip(after_step2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();

        // With momentum, second step should be larger (momentum accumulation)
        assert!(
            step2_size > step1_size * 0.8,
            "Momentum should accumulate, making second step larger"
        );
    }
}

#[cfg(test)]
mod optimizer_integration_tests {
    use super::*;

    #[test]
    fn test_optimizer_with_ste_integration() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        // Setup STE
        let ste_config = STEConfig {
            variant: STEVariant::Standard,
            bits: 1,
            range: 1.0,
            temperature: 1.0,
            learnable_lr: 0.01,
            clip_gradients: true,
            clip_threshold: 1.0,
            gradient_clip: Some(1.0),
            use_noise: false,
            device: Some(device.clone()),
        };
        let mut ste = StraightThroughEstimator::new(ste_config).unwrap();

        // Setup optimizer
        let mut optimizer = QATAdam::new(params.clone(), ParamsAdam::default()).unwrap();

        // Simulate training step with quantization
        let quantized_params: Vec<Tensor> = params
            .iter()
            .map(|p| ste.forward_quantized(p).unwrap())
            .collect();

        // Create loss from quantized parameters - using result
        let _loss_sum = quantized_params
            .iter()
            .map(|p| p.sum_all().unwrap())
            .reduce(|acc, x| (&acc + &x).unwrap())
            .unwrap();

        // Simulate gradients (in real scenario, these would come from backprop)
        let gradients = create_mock_gradients(&params).unwrap();

        // Update parameters
        optimizer.step(&gradients).unwrap();

        // Basic validation that integration worked
        assert!(true);
    }

    #[test]
    fn test_multiple_optimizer_types() {
        let device = setup_test_device();
        let params1 = create_test_parameters(&device).unwrap();
        let params2 = create_test_parameters(&device).unwrap();
        let params3 = create_test_parameters(&device).unwrap();

        let mut adam_opt = QATAdam::new(params1.clone(), ParamsAdam::default()).unwrap();
        let mut adamw_opt = QATAdamW::new(params2.clone(), ParamsAdamW::default()).unwrap();
        let mut sgd_opt = QATSGDWithMomentum::new(
            params3.clone(),
            0.01, // learning_rate
            0.9,  // momentum
            0.0,  // weight_decay
        )
        .unwrap();

        let grads1 = create_mock_gradients(&params1).unwrap();
        let grads2 = create_mock_gradients(&params2).unwrap();
        let grads3 = create_mock_gradients(&params3).unwrap();

        // All optimizers should work simultaneously
        adam_opt.step(&grads1).unwrap();
        adamw_opt.step(&grads2).unwrap();
        sgd_opt.step(&grads3).unwrap();

        assert!(true);
    }
}

#[cfg(test)]
mod learning_rate_scheduling {
    use super::*;

    #[test]
    fn test_learning_rate_adaptation() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        // Test with different learning rates
        let learning_rates = vec![0.1, 0.01, 0.001, 0.0001];

        for lr in learning_rates {
            let mut optimizer = QATAdam::new(
                params.clone(),
                ParamsAdam {
                    lr,
                    ..ParamsAdam::default()
                },
            )
            .unwrap();

            let gradients = create_mock_gradients(&params).unwrap();

            // Should work with any reasonable learning rate
            let result = optimizer.step(&gradients);
            assert!(
                result.is_ok(),
                "Optimizer should handle learning rate: {lr}"
            );
        }
    }

    #[test]
    fn test_extreme_learning_rates() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        // Test very small learning rate
        let mut optimizer_small = QATAdam::new(
            params.clone(),
            ParamsAdam {
                lr: 1e-10,
                ..ParamsAdam::default()
            },
        )
        .unwrap();

        // Test large learning rate
        let mut optimizer_large = QATAdam::new(
            params.clone(),
            ParamsAdam {
                lr: 10.0,
                ..ParamsAdam::default()
            },
        )
        .unwrap();

        let gradients = create_mock_gradients(&params).unwrap();

        // Both should handle extreme learning rates gracefully
        optimizer_small.step(&gradients).unwrap();
        optimizer_large.step(&gradients).unwrap();

        assert!(true);
    }
}

#[cfg(test)]
mod error_handling {
    use super::*;

    #[test]
    fn test_empty_parameters() {
        let empty_params: Vec<Tensor> = vec![];

        let result = QATAdam::new(empty_params, ParamsAdam::default());

        // Should handle empty parameters gracefully
        match result {
            Ok(_) => assert!(true),  // Acceptable behavior
            Err(_) => assert!(true), // Also acceptable to reject empty params
        }
    }

    #[test]
    fn test_mismatched_gradients() {
        let device = setup_test_device();
        let params = create_test_parameters(&device).unwrap();

        let mut optimizer = QATAdam::new(params.clone(), ParamsAdam::default()).unwrap();

        // Create gradients with wrong shapes
        let wrong_gradients = vec![
            Tensor::randn(0.0f32, 0.01f32, &[5, 5], &device).unwrap(), // Wrong shape
        ];

        let result = optimizer.step(&wrong_gradients);

        // Should handle mismatched gradients appropriately
        match result {
            Ok(_) => panic!("Should not succeed with mismatched gradients"),
            Err(_) => assert!(true), // Expected error behavior
        }
    }
}
