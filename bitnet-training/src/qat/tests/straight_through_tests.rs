// Integration tests for Straight-Through Estimator functionality
// Tests the core STE implementation with different variants

use candle_core::{Result, Tensor, Device, DType};
use bitnet_training::qat::{
    straight_through::{StraightThroughEstimator, STEConfig, STEVariant, MultiLayerSTE, quantize_with_ste},
    autograd::{quantize_with_ste_autograd, QATLayer, QATModel},
    loss::{QuantizationAwareLoss, BaseLossType, QATLossFactory},
    optimizer::{QuantizationAwareAdam, QATOptimizer, QATOptimizerFactory},
};
use std::collections::HashMap;

#[test]
fn test_ste_variants_quantization() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_slice(&[0.7, -0.3, 1.5, -2.0, 0.0], (5,), &device)?;

    // Test Standard STE
    let standard_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut standard_ste = StraightThroughEstimator::new(standard_config, device.clone())?;
    let standard_output = standard_ste.forward(&input)?;
    let standard_values: Vec<f32> = standard_output.to_vec1()?;

    // Binary quantization should give only +1 and -1
    for val in &standard_values {
        assert!(*val == 1.0 || *val == -1.0, "Standard STE should output binary values");
    }

    // Test Clipped STE with out-of-range input
    let clipped_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut clipped_ste = StraightThroughEstimator::new(clipped_config, device.clone())?;
    let clipped_output = clipped_ste.forward(&input)?;
    let clipped_values: Vec<f32> = clipped_output.to_vec1()?;

    // Should clip and then quantize
    for val in &clipped_values {
        assert!(*val == 1.0 || *val == -1.0, "Clipped STE should output binary values");
    }

    // Should have recorded some clipping
    assert!(clipped_ste.get_clipping_rate() > 0.0, "Should have clipped some values");

    // Test Soft STE
    let soft_config = STEConfig {
        variant: STEVariant::Soft,
        bits: 1,
        range: 1.0,
        temperature: 1.0,
        ..Default::default()
    };
    let mut soft_ste = StraightThroughEstimator::new(soft_config, device.clone())?;
    let soft_output = soft_ste.forward(&input)?;
    let soft_values: Vec<f32> = soft_output.to_vec1()?;

    // Soft quantization should give smooth values
    for val in &soft_values {
        assert!(val.abs() <= 1.0, "Soft STE should output values within range");
    }

    Ok(())
}

#[test]
fn test_multi_bit_quantization() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_slice(&[0.5, -0.3, 0.8, -0.9, 0.0], (5,), &device)?;

    // Test ternary quantization (2 bits)
    let ternary_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 2,
        range: 1.0,
        ..Default::default()
    };
    let mut ternary_ste = StraightThroughEstimator::new(ternary_config, device.clone())?;
    let ternary_output = ternary_ste.forward(&input)?;
    let ternary_values: Vec<f32> = ternary_output.to_vec1()?;

    // Ternary should give -1, 0, or 1
    for val in &ternary_values {
        assert!(
            *val == -1.0 || *val == 0.0 || *val == 1.0,
            "Ternary quantization should output -1, 0, or 1, got: {}",
            val
        );
    }

    // Test 4-bit quantization
    let multi_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 4,
        range: 1.0,
        ..Default::default()
    };
    let mut multi_ste = StraightThroughEstimator::new(multi_config, device)?;
    let multi_output = multi_ste.forward(&input)?;
    let multi_values: Vec<f32> = multi_output.to_vec1()?;

    // Should quantize to discrete levels within range
    for val in &multi_values {
        assert!(val.abs() <= 1.0, "Multi-bit quantization should stay within range");
    }

    Ok(())
}

#[test]
fn test_multi_layer_ste_management() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut multi_ste = MultiLayerSTE::new(config, device);

    let input1 = Tensor::from_slice(&[0.5, -0.3], (2,), &Device::Cpu)?;
    let input2 = Tensor::from_slice(&[0.8, -0.9], (2,), &Device::Cpu)?;

    // Forward passes through different layers
    let output1 = multi_ste.forward_layer("conv1", &input1)?;
    let output2 = multi_ste.forward_layer("conv2", &input2)?;
    let output1_again = multi_ste.forward_layer("conv1", &input1)?;

    // Should have created separate STEs
    assert_eq!(multi_ste.estimators.len(), 2);

    // Get statistics for all layers
    let all_stats = multi_ste.get_all_statistics();
    assert_eq!(all_stats.len(), 2);
    assert!(all_stats.contains_key("conv1"));
    assert!(all_stats.contains_key("conv2"));

    // conv1 should have 2 operations (called twice)
    assert_eq!(all_stats["conv1"].total_operations, 2);
    assert_eq!(all_stats["conv2"].total_operations, 1);

    // Test statistics reset
    multi_ste.reset_all_statistics();
    let reset_stats = multi_ste.get_all_statistics();
    assert_eq!(reset_stats["conv1"].total_operations, 0);
    assert_eq!(reset_stats["conv2"].total_operations, 0);

    Ok(())
}

#[test]
fn test_ste_autograd_integration() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };

    let input = Tensor::from_slice(&[0.7, -0.3, 0.0], (3,), &device)?;

    // Test autograd quantization
    let quantized = quantize_with_ste_autograd(&input, &config)?;
    let quantized_values: Vec<f32> = quantized.to_vec1()?;

    // Should be quantized to binary values
    for val in &quantized_values {
        assert!(*val == 1.0 || *val == -1.0, "Autograd STE should output binary values");
    }

    Ok(())
}

#[test]
fn test_qat_layer_functionality() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig::default();

    let layer = QATLayer::new(
        config,
        device,
        "test_layer".to_string(),
        true,  // quantize weights
        true,  // quantize activations
    );

    let weights = Tensor::from_slice(&[0.5, -0.3, 0.8], (3,), &Device::Cpu)?;
    let activations = Tensor::from_slice(&[1.2, -0.7, 0.1], (3,), &Device::Cpu)?;

    // Test weight quantization
    let quantized_weights = layer.quantize_weights(&weights)?;
    let weight_values: Vec<f32> = quantized_weights.to_vec1()?;
    for val in &weight_values {
        assert!(*val == 1.0 || *val == -1.0, "Weights should be quantized");
    }

    // Test activation quantization
    let quantized_activations = layer.quantize_activations(&activations)?;
    let activation_values: Vec<f32> = quantized_activations.to_vec1()?;
    for val in &activation_values {
        assert!(*val == 1.0 || *val == -1.0, "Activations should be quantized");
    }

    assert_eq!(layer.name(), "test_layer");

    Ok(())
}

#[test]
fn test_qat_model_layer_configs() -> Result<()> {
    let device = Device::Cpu;
    let global_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        ..Default::default()
    };
    let mut model = QATModel::new(global_config.clone(), device.clone());

    // Set layer-specific configuration
    let layer_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 2,
        ..Default::default()
    };
    model.set_layer_config("special_layer".to_string(), layer_config);

    // Test layer-specific config retrieval
    let retrieved_config = model.get_layer_config("special_layer");
    assert_eq!(retrieved_config.variant, STEVariant::Clipped);
    assert_eq!(retrieved_config.bits, 2);

    // Test global config fallback
    let default_config = model.get_layer_config("unknown_layer");
    assert_eq!(default_config.variant, STEVariant::Standard);
    assert_eq!(default_config.bits, 1);

    // Test model parameter quantization
    let mut parameters = HashMap::new();
    parameters.insert("layer1.weight".to_string(),
                     Tensor::from_slice(&[0.5, -0.3], (2,), &device)?);
    parameters.insert("layer2.weight".to_string(),
                     Tensor::from_slice(&[0.8, -0.9], (2,), &device)?);

    let quantized_params = model.quantize_model_parameters(parameters)?;

    assert_eq!(quantized_params.len(), 2);
    assert!(quantized_params.contains_key("layer1.weight"));
    assert!(quantized_params.contains_key("layer2.weight"));

    // Check quantization worked
    let layer1_values: Vec<f32> = quantized_params["layer1.weight"].to_vec1()?;
    for val in &layer1_values {
        assert!(*val == 1.0 || *val == -1.0, "Parameters should be quantized");
    }

    Ok(())
}

#[test]
fn test_qat_loss_computation() -> Result<()> {
    let device = Device::Cpu;

    // Create QAT loss
    let loss = QATLossFactory::create_qat_loss(
        BaseLossType::MeanSquaredError,
        0.01, // regularization_weight
        0.01, // quantization_penalty_weight
        device,
    );

    let predictions = Tensor::from_slice(&[1.0, 2.0, 3.0], (3,), &Device::Cpu)?;
    let targets = Tensor::from_slice(&[1.1, 1.9, 3.1], (3,), &Device::Cpu)?;

    let loss_value = loss.compute_loss(&predictions, &targets)?;
    let loss_scalar = loss_value.to_scalar::<f32>()?;

    // Should compute MSE
    assert!(loss_scalar > 0.0, "Loss should be positive");
    assert!(loss_scalar < 1.0, "Loss should be reasonable for small differences");

    Ok(())
}

#[test]
fn test_qat_optimizer_integration() -> Result<()> {
    let device = Device::Cpu;

    // Create parameters and gradients
    let mut parameters = HashMap::new();
    parameters.insert("weight1".to_string(),
                     Tensor::from_slice(&[1.0, 2.0], (2,), &device)?);
    parameters.insert("weight2".to_string(),
                     Tensor::from_slice(&[3.0, 4.0], (2,), &device)?);

    let gradients = HashMap::new();
    parameters.insert("weight1".to_string(),
                     Tensor::from_slice(&[0.1, 0.2], (2,), &device)?);
    parameters.insert("weight2".to_string(),
                     Tensor::from_slice(&[0.3, 0.4], (2,), &device)?);

    // Create QAT optimizer
    let mut optimizer = QATOptimizerFactory::create_adam(0.01, 0.001, device);

    // Store original values
    let original_weight1: Vec<f32> = parameters["weight1"].to_vec1()?;

    // Perform optimization step
    optimizer.step(&mut parameters, &gradients)?;

    // Parameters should be updated
    let updated_weight1: Vec<f32> = parameters["weight1"].to_vec1()?;
    assert_ne!(original_weight1, updated_weight1, "Parameters should be updated");

    Ok(())
}

#[test]
fn test_learnable_ste_initialization() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Learnable,
        bits: 1,
        range: 1.0,
        learnable_lr: 0.01,
        ..Default::default()
    };

    let ste = StraightThroughEstimator::new(config, device)?;

    // Learnable STE should initialize learnable parameters
    assert!(ste.learnable_scale.is_some(), "Learnable scale should be initialized");
    assert!(ste.learnable_zero_point.is_some(), "Learnable zero point should be initialized");

    Ok(())
}

#[test]
fn test_ste_statistics_tracking() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut ste = StraightThroughEstimator::new(config, device)?;

    // Input with values outside range to trigger clipping
    let input = Tensor::from_slice(&[2.0, -3.0, 0.5, -0.5], (4,), &Device::Cpu)?;

    let _output = ste.forward(&input)?;

    // Check statistics
    let stats = ste.get_statistics();
    assert!(stats.quantization_error > 0.0, "Should have quantization error");
    assert!(stats.total_operations > 0, "Should track operations");
    assert!(stats.clipping_rate > 0.0, "Should have clipping with out-of-range inputs");

    // Reset and check
    ste.reset_statistics();
    let reset_stats = ste.get_statistics();
    assert_eq!(reset_stats.total_operations, 0, "Statistics should be reset");
    assert_eq!(reset_stats.quantization_error, 0.0, "Error should be reset");

    Ok(())
}

#[test]
fn test_convenience_functions() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig::default();

    let input = Tensor::from_slice(&[0.5, -0.3, 0.8], (3,), &device)?;

    // Test convenience function
    let quantized = quantize_with_ste(&input, &config, &device)?;
    let values: Vec<f32> = quantized.to_vec1()?;

    for val in &values {
        assert!(*val == 1.0 || *val == -1.0, "Should be quantized to binary");
    }

    Ok(())
}
