// Basic QAT Training Example - Demonstrates Phase 3.2 QAT Infrastructure
// Shows how to use Straight-Through Estimator with autograd, loss, and optimizer

use candle_core::{Result, Tensor, Device, DType};
use std::collections::HashMap;

use bitnet_training::qat::{
    straight_through::{STEConfig, STEVariant, StraightThroughEstimator},
    autograd::{quantize_with_ste_autograd, QATLayer, QATModel},
    loss::{QATLossFactory, BaseLossType},
    optimizer::{QATOptimizerFactory, QATOptimizer},
    state_tracking::{QATTrainingTracker, QATTrainingState},
};

fn main() -> Result<()> {
    println!("BitNet-Rust QAT Training Demo - Phase 3.2");
    println!("==========================================");

    let device = Device::Cpu;

    // 1. Configure Straight-Through Estimator
    println!("\n1. Setting up Straight-Through Estimator");
    let ste_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1, // Binary quantization for BitNet
        range: 1.0,
        temperature: 1.0,
        clip_gradients: true,
        clip_threshold: 1.0,
        ..Default::default()
    };

    // Create STE instance
    let mut ste = StraightThroughEstimator::new(ste_config.clone(), device.clone())?;

    // Test basic quantization
    let input = Tensor::from_slice(&[0.7, -0.3, 0.0, 1.2, -0.8], (5,), &device)?;
    println!("Input: {:?}", input.to_vec1::<f32>()?);

    let quantized = ste.forward(&input)?;
    println!("Quantized (STE): {:?}", quantized.to_vec1::<f32>()?);

    // 2. Test Autograd Integration
    println!("\n2. Testing Autograd Integration");
    let autograd_quantized = quantize_with_ste_autograd(&input, &ste_config)?;
    println!("Quantized (Autograd): {:?}", autograd_quantized.to_vec1::<f32>()?);

    // 3. Create QAT Layer
    println!("\n3. QAT Layer Configuration");
    let qat_layer = QATLayer::new(
        ste_config.clone(),
        device.clone(),
        "demo_layer".to_string(),
        true,  // quantize weights
        true,  // quantize activations
    );

    let weights = Tensor::from_slice(&[0.5, -0.3, 0.8, -0.9], (4,), &device)?;
    let quantized_weights = qat_layer.quantize_weights(&weights)?;
    println!("Original weights: {:?}", weights.to_vec1::<f32>()?);
    println!("Quantized weights: {:?}", quantized_weights.to_vec1::<f32>()?);

    // 4. QAT Model with Multiple Layers
    println!("\n4. Multi-Layer QAT Model");
    let mut qat_model = QATModel::new(ste_config.clone(), device.clone());

    // Configure different layers with different quantization
    let conv_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let linear_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 2, // Ternary quantization
        range: 1.0,
        ..Default::default()
    };

    qat_model.set_layer_config("conv1.weight".to_string(), conv_config);
    qat_model.set_layer_config("linear.weight".to_string(), linear_config);

    // Test model parameter quantization
    let mut model_params = HashMap::new();
    model_params.insert("conv1.weight".to_string(),
                       Tensor::from_slice(&[0.6, -0.4, 1.2, -1.8], (4,), &device)?);
    model_params.insert("linear.weight".to_string(),
                       Tensor::from_slice(&[0.3, -0.7, 0.9], (3,), &device)?);

    let quantized_params = qat_model.quantize_model_parameters(model_params)?;

    for (name, param) in &quantized_params {
        println!("{}: {:?}", name, param.to_vec1::<f32>()?);
    }

    // 5. QAT Loss Function
    println!("\n5. QAT Loss Function");
    let qat_loss = QATLossFactory::create_qat_loss(
        BaseLossType::MeanSquaredError,
        0.01, // regularization weight
        0.01, // quantization penalty weight
        device.clone(),
    );

    let predictions = Tensor::from_slice(&[1.0, 2.0, 3.0], (3,), &device)?;
    let targets = Tensor::from_slice(&[1.1, 1.9, 3.1], (3,), &device)?;
    let loss = qat_loss.compute_loss(&predictions, &targets)?;
    println!("QAT Loss: {:.6}", loss.to_scalar::<f32>()?);

    // 6. QAT Optimizer
    println!("\n6. QAT Optimizer");
    let mut qat_optimizer = QATOptimizerFactory::create_adam(0.001, 0.01, device.clone());

    // Create dummy parameters and gradients for optimizer test
    let mut parameters = HashMap::new();
    parameters.insert("weight1".to_string(),
                     Tensor::from_slice(&[1.0, 2.0], (2,), &device)?);

    let mut gradients = HashMap::new();
    gradients.insert("weight1".to_string(),
                    Tensor::from_slice(&[0.1, 0.2], (2,), &device)?);

    println!("Before optimization: {:?}", parameters["weight1"].to_vec1::<f32>()?);
    qat_optimizer.step(&mut parameters, &gradients)?;
    println!("After optimization: {:?}", parameters["weight1"].to_vec1::<f32>()?);

    // 7. Training State Tracking
    println!("\n7. Training State Tracking");
    let mut training_tracker = QATTrainingTracker::new(
        device.clone(),
        10,   // log_interval
        100,  // validation_interval
        1000, // checkpoint_interval
        50,   // early_stopping_patience
    );

    // Simulate training steps
    let mut ste_stats = HashMap::new();
    ste_stats.insert("conv1".to_string(), ste.get_statistics());

    training_tracker.update_training_step(
        1,    // epoch
        100,  // step
        0.001, // learning_rate
        0.5,  // loss
        0.1,  // quantization_error
        ste_stats,
        32,   // samples_processed
        0.1,  // step_time
    );

    let training_summary = training_tracker.get_state().get_training_summary();
    println!("Training Summary:");
    println!("  Epoch: {}, Step: {}", training_summary.epoch, training_summary.step);
    println!("  Loss: {:.4}, LR: {:.6}", training_summary.current_loss, training_summary.learning_rate);
    println!("  Quantization Error: {:.4}", training_summary.quantization_error);
    println!("  Throughput: {:.1} samples/s", training_summary.throughput);

    // 8. Demonstrate Different STE Variants
    println!("\n8. Testing Different STE Variants");

    // Test input with various values
    let test_input = Tensor::from_slice(&[0.7, -0.3, 1.5, -2.0, 0.0], (5,), &device)?;
    println!("Test input: {:?}", test_input.to_vec1::<f32>()?);

    // Standard STE
    let standard_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut standard_ste = StraightThroughEstimator::new(standard_config, device.clone())?;
    let standard_output = standard_ste.forward(&test_input)?;
    println!("Standard STE: {:?}", standard_output.to_vec1::<f32>()?);

    // Clipped STE (should handle out-of-range values)
    let clipped_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut clipped_ste = StraightThroughEstimator::new(clipped_config, device.clone())?;
    let clipped_output = clipped_ste.forward(&test_input)?;
    println!("Clipped STE: {:?}", clipped_output.to_vec1::<f32>()?);
    println!("Clipping rate: {:.2}%", clipped_ste.get_clipping_rate() * 100.0);

    // Soft STE (smooth quantization)
    let soft_config = STEConfig {
        variant: STEVariant::Soft,
        bits: 1,
        range: 1.0,
        temperature: 1.0,
        ..Default::default()
    };
    let mut soft_ste = StraightThroughEstimator::new(soft_config, device.clone())?;
    let soft_output = soft_ste.forward(&test_input)?;
    println!("Soft STE: {:?}", soft_output.to_vec1::<f32>()?);

    // Ternary quantization (2-bit)
    let ternary_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 2,
        range: 1.0,
        ..Default::default()
    };
    let mut ternary_ste = StraightThroughEstimator::new(ternary_config, device.clone())?;
    let ternary_output = ternary_ste.forward(&test_input)?;
    println!("Ternary STE: {:?}", ternary_output.to_vec1::<f32>()?);

    println!("\n✅ QAT Phase 3.2 Infrastructure Demo Complete!");
    println!("Key Features Demonstrated:");
    println!("  - Straight-Through Estimator with multiple variants");
    println!("  - Custom autograd functions for gradient flow");
    println!("  - QAT-specific loss functions and optimizers");
    println!("  - Multi-layer quantization management");
    println!("  - Training state tracking and monitoring");
    println!("  - Binary, ternary, and multi-bit quantization");

    Ok(())
}
