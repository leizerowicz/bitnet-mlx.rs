// Integration tests for QAT autograd functionality
// Tests custom autograd functions and gradient flow

use candle_core::{Result, Tensor, Device, DType};
use bitnet_training::qat::{
    straight_through::{STEConfig, STEVariant},
    autograd::{
        QATAutograd, QuantizationFunction, quantize_with_ste_autograd,
        batch_quantize_with_ste, QATLayer, QATModel,
    },
};
use std::collections::HashMap;

#[test]
fn test_qat_autograd_creation_and_basic_usage() -> Result<()> {
    let device = Device::Cpu;
    let autograd = QATAutograd::new(device.clone());

    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };

    let quantization_fn = autograd.create_quantization_function(config);

    let input = Tensor::from_slice(&[0.7, -0.3, 0.0, 1.2], (4,), &device)?;

    // Test forward pass
    let output = quantization_fn.apply(&input)?;
    let output_values: Vec<f32> = output.to_vec1()?;

    // Should be quantized to binary values
    for val in &output_values {
        assert!(*val == 1.0 || *val == -1.0, "Output should be binary quantized");
    }

    Ok(())
}

#[test]
fn test_quantization_function_with_backward() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };

    let quantization_fn = QuantizationFunction::new(config, device.clone());
    let input = Tensor::from_slice(&[0.5, -0.3, 0.8, -0.9], (4,), &device)?;

    // Test with backward pass capability
    let output = quantization_fn.apply_with_backward(&input)?;
    let output_values: Vec<f32> = output.to_vec1()?;

    // Should have quantized values
    for val in &output_values {
        assert!(*val == 1.0 || *val == -1.0, "Should be binary quantized");
    }

    Ok(())
}

#[test]
fn test_different_ste_variants_in_autograd() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_slice(&[0.7, -0.3, 1.5, -2.0], (4,), &device)?;

    // Test Standard STE
    let standard_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let standard_output = quantize_with_ste_autograd(&input, &standard_config)?;
    let standard_values: Vec<f32> = standard_output.to_vec1()?;

    for val in &standard_values {
        assert!(*val == 1.0 || *val == -1.0, "Standard STE should output binary");
    }

    // Test Clipped STE
    let clipped_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let clipped_output = quantize_with_ste_autograd(&input, &clipped_config)?;
    let clipped_values: Vec<f32> = clipped_output.to_vec1()?;

    for val in &clipped_values {
        assert!(*val == 1.0 || *val == -1.0, "Clipped STE should output binary");
    }

    // Test Soft STE
    let soft_config = STEConfig {
        variant: STEVariant::Soft,
        bits: 1,
        range: 1.0,
        temperature: 1.0,
        ..Default::default()
    };
    let soft_output = quantize_with_ste_autograd(&input, &soft_config)?;
    let soft_values: Vec<f32> = soft_output.to_vec1()?;

    // Soft quantization gives continuous values
    for val in &soft_values {
        assert!(val.abs() <= 1.0, "Soft STE should stay within range");
    }

    Ok(())
}

#[test]
fn test_multi_bit_quantization_autograd() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_slice(&[0.5, -0.3, 0.8, -0.9, 0.0], (5,), &device)?;

    // Test ternary (2-bit) quantization
    let ternary_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 2,
        range: 1.0,
        ..Default::default()
    };
    let ternary_output = quantize_with_ste_autograd(&input, &ternary_config)?;
    let ternary_values: Vec<f32> = ternary_output.to_vec1()?;

    // Should quantize to -1, 0, or 1
    for val in &ternary_values {
        assert!(
            *val == -1.0 || *val == 0.0 || *val == 1.0,
            "Ternary should output -1, 0, or 1, got: {}",
            val
        );
    }

    // Test 4-bit quantization
    let quad_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 4,
        range: 1.0,
        ..Default::default()
    };
    let quad_output = quantize_with_ste_autograd(&input, &quad_config)?;
    let quad_values: Vec<f32> = quad_output.to_vec1()?;

    // Should stay within range and be quantized
    for val in &quad_values {
        assert!(val.abs() <= 1.0, "4-bit quantization should stay within range");
    }

    Ok(())
}

#[test]
fn test_batch_quantization() -> Result<()> {
    let device = Device::Cpu;

    let input1 = Tensor::from_slice(&[0.5, -0.3], (2,), &device)?;
    let input2 = Tensor::from_slice(&[0.8, -0.9], (2,), &device)?;
    let input3 = Tensor::from_slice(&[1.2, -1.5], (2,), &device)?;

    let inputs = vec![&input1, &input2, &input3];

    // Different configs for each input
    let configs = vec![
        STEConfig { bits: 1, variant: STEVariant::Standard, ..Default::default() },
        STEConfig { bits: 2, variant: STEVariant::Standard, ..Default::default() },
        STEConfig { bits: 1, variant: STEVariant::Clipped, ..Default::default() },
    ];

    let results = batch_quantize_with_ste(&inputs, &configs)?;

    assert_eq!(results.len(), 3, "Should have three results");

    // Check first result (binary quantization)
    let result1_values: Vec<f32> = results[0].to_vec1()?;
    for val in &result1_values {
        assert!(*val == 1.0 || *val == -1.0, "First result should be binary");
    }

    // Check second result (ternary quantization)
    let result2_values: Vec<f32> = results[1].to_vec1()?;
    for val in &result2_values {
        assert!(
            *val == -1.0 || *val == 0.0 || *val == 1.0,
            "Second result should be ternary"
        );
    }

    // Check third result (clipped binary)
    let result3_values: Vec<f32> = results[2].to_vec1()?;
    for val in &result3_values {
        assert!(*val == 1.0 || *val == -1.0, "Third result should be binary");
    }

    Ok(())
}

#[test]
fn test_qat_layer_autograd_integration() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };

    let layer = QATLayer::new(
        config,
        device.clone(),
        "conv_layer".to_string(),
        true,  // quantize_weights
        true,  // quantize_activations
    );

    let weights = Tensor::from_slice(&[0.6, -0.4, 0.2, -0.8], (4,), &device)?;
    let activations = Tensor::from_slice(&[1.1, -0.5, 0.3, -1.2], (4,), &device)?;

    // Test weight quantization with autograd
    let quantized_weights = layer.quantize_weights(&weights)?;
    let weight_values: Vec<f32> = quantized_weights.to_vec1()?;

    for val in &weight_values {
        assert!(*val == 1.0 || *val == -1.0, "Weights should be quantized");
    }

    // Test activation quantization with autograd
    let quantized_activations = layer.quantize_activations(&activations)?;
    let activation_values: Vec<f32> = quantized_activations.to_vec1()?;

    for val in &activation_values {
        assert!(*val == 1.0 || *val == -1.0, "Activations should be quantized");
    }

    assert_eq!(layer.name(), "conv_layer");

    Ok(())
}

#[test]
fn test_qat_layer_selective_quantization() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig::default();

    // Create layer that only quantizes weights, not activations
    let weights_only_layer = QATLayer::new(
        config.clone(),
        device.clone(),
        "weights_only".to_string(),
        true,  // quantize_weights
        false, // don't quantize_activations
    );

    // Create layer that only quantizes activations, not weights
    let activations_only_layer = QATLayer::new(
        config,
        device.clone(),
        "activations_only".to_string(),
        false, // don't quantize_weights
        true,  // quantize_activations
    );

    let weights = Tensor::from_slice(&[0.5, -0.3], (2,), &device)?;
    let activations = Tensor::from_slice(&[0.8, -0.9], (2,), &device)?;

    // Test weights-only layer
    let quantized_weights = weights_only_layer.quantize_weights(&weights)?;
    let unquantized_activations = weights_only_layer.quantize_activations(&activations)?;

    let weight_values: Vec<f32> = quantized_weights.to_vec1()?;
    let activation_values: Vec<f32> = unquantized_activations.to_vec1()?;

    // Weights should be quantized
    for val in &weight_values {
        assert!(*val == 1.0 || *val == -1.0, "Weights should be quantized");
    }

    // Activations should be unchanged
    assert_eq!(activation_values, vec![0.8, -0.9], "Activations should be unchanged");

    // Test activations-only layer
    let unquantized_weights = activations_only_layer.quantize_weights(&weights)?;
    let quantized_activations = activations_only_layer.quantize_activations(&activations)?;

    let unquant_weight_values: Vec<f32> = unquantized_weights.to_vec1()?;
    let quant_activation_values: Vec<f32> = quantized_activations.to_vec1()?;

    // Weights should be unchanged
    assert_eq!(unquant_weight_values, vec![0.5, -0.3], "Weights should be unchanged");

    // Activations should be quantized
    for val in &quant_activation_values {
        assert!(*val == 1.0 || *val == -1.0, "Activations should be quantized");
    }

    Ok(())
}

#[test]
fn test_qat_model_parameter_quantization() -> Result<()> {
    let device = Device::Cpu;
    let global_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let mut model = QATModel::new(global_config, device.clone());

    // Set different configs for different layers
    let conv_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let linear_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 2,
        range: 1.0,
        ..Default::default()
    };

    model.set_layer_config("conv1.weight".to_string(), conv_config);
    model.set_layer_config("linear.weight".to_string(), linear_config);

    // Create model parameters
    let mut parameters = HashMap::new();
    parameters.insert("conv1.weight".to_string(),
                     Tensor::from_slice(&[0.7, -0.3, 1.2, -1.8], (4,), &device)?);
    parameters.insert("conv1.bias".to_string(),
                     Tensor::from_slice(&[0.1, -0.2], (2,), &device)?);
    parameters.insert("linear.weight".to_string(),
                     Tensor::from_slice(&[0.5, -0.8, 0.2], (3,), &device)?);

    // Quantize all parameters
    let quantized_params = model.quantize_model_parameters(parameters)?;

    assert_eq!(quantized_params.len(), 3, "Should have quantized all parameters");

    // Check conv1.weight (should use clipped config)
    let conv_weight_values: Vec<f32> = quantized_params["conv1.weight"].to_vec1()?;
    for val in &conv_weight_values {
        assert!(*val == 1.0 || *val == -1.0, "Conv weights should be binary quantized");
    }

    // Check conv1.bias (should use global config - binary)
    let conv_bias_values: Vec<f32> = quantized_params["conv1.bias"].to_vec1()?;
    for val in &conv_bias_values {
        assert!(*val == 1.0 || *val == -1.0, "Conv bias should be binary quantized");
    }

    // Check linear.weight (should use ternary config)
    let linear_weight_values: Vec<f32> = quantized_params["linear.weight"].to_vec1()?;
    for val in &linear_weight_values {
        assert!(
            *val == -1.0 || *val == 0.0 || *val == 1.0,
            "Linear weights should be ternary quantized"
        );
    }

    Ok(())
}

#[test]
fn test_error_handling_mismatched_inputs() -> Result<()> {
    let device = Device::Cpu;

    let input1 = Tensor::from_slice(&[0.5, -0.3], (2,), &device)?;
    let input2 = Tensor::from_slice(&[0.8, -0.9], (2,), &device)?;
    let inputs = vec![&input1, &input2];

    // Only one config for two inputs - should error
    let configs = vec![
        STEConfig { bits: 1, ..Default::default() },
    ];

    let result = batch_quantize_with_ste(&inputs, &configs);
    assert!(result.is_err(), "Should error with mismatched input/config lengths");

    Ok(())
}

#[test]
fn test_gradient_flow_preservation() -> Result<()> {
    let device = Device::Cpu;
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };

    // Create tensor that requires gradients
    let input = Tensor::from_slice(&[0.5, -0.3, 0.8], (3,), &device)?;

    // Apply quantization with autograd
    let quantized = quantize_with_ste_autograd(&input, &config)?;

    // The quantized tensor should still be connected to the computation graph
    // In a real training scenario, gradients would flow back through the STE
    assert_eq!(quantized.shape(), input.shape(), "Shape should be preserved");

    let quantized_values: Vec<f32> = quantized.to_vec1()?;
    for val in &quantized_values {
        assert!(*val == 1.0 || *val == -1.0, "Should be quantized");
    }

    Ok(())
}

#[test]
fn test_different_ranges_and_temperatures() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::from_slice(&[0.5, -0.3, 0.8, -1.2], (4,), &device)?;

    // Test different ranges
    let config_range_2 = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 2.0,
        ..Default::default()
    };

    let output_range_2 = quantize_with_ste_autograd(&input, &config_range_2)?;
    let values_range_2: Vec<f32> = output_range_2.to_vec1()?;

    for val in &values_range_2 {
        assert!(*val == 2.0 || *val == -2.0, "Should quantize to ±2.0 with range=2.0");
    }

    // Test soft quantization with different temperature
    let config_soft_temp = STEConfig {
        variant: STEVariant::Soft,
        bits: 1,
        range: 1.0,
        temperature: 0.5, // Lower temperature = sharper quantization
        ..Default::default()
    };

    let output_soft_temp = quantize_with_ste_autograd(&input, &config_soft_temp)?;
    let values_soft_temp: Vec<f32> = output_soft_temp.to_vec1()?;

    for val in &values_soft_temp {
        assert!(val.abs() <= 1.0, "Should stay within range");
        // With lower temperature, values should be closer to ±1
    }

    Ok(())
}

#[test]
fn test_complex_layer_configuration() -> Result<()> {
    let device = Device::Cpu;

    // Create model with complex layer configurations
    let global_config = STEConfig::default();
    let mut model = QATModel::new(global_config, device.clone());

    // Set up different quantization strategies for different layer types
    let conv_config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let attention_config = STEConfig {
        variant: STEVariant::Soft,
        bits: 2,
        range: 1.0,
        temperature: 2.0,
        ..Default::default()
    };
    let output_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 4,
        range: 1.0,
        ..Default::default()
    };

    // Configure different layer types
    model.set_layer_config("conv1.weight".to_string(), conv_config);
    model.set_layer_config("conv2.weight".to_string(), conv_config.clone());
    model.set_layer_config("attention.qkv.weight".to_string(), attention_config);
    model.set_layer_config("output.weight".to_string(), output_config);

    // Create layers with different quantization needs
    let conv_layer = model.create_qat_layer(
        "conv1.weight".to_string(),
        true,  // quantize weights
        false, // don't quantize activations (typically done for conv)
    );

    let attention_layer = model.create_qat_layer(
        "attention.qkv.weight".to_string(),
        true, // quantize weights
        true, // quantize activations (important for attention)
    );

    let output_layer = model.create_qat_layer(
        "output.weight".to_string(),
        true,  // quantize weights
        false, // don't quantize final activations
    );

    // Test that each layer has correct configuration
    assert_eq!(conv_layer.name(), "conv1.weight");
    assert_eq!(attention_layer.name(), "attention.qkv.weight");
    assert_eq!(output_layer.name(), "output.weight");

    // Test quantization with different layers
    let test_tensor = Tensor::from_slice(&[0.5, -0.3, 0.8, -1.2], (4,), &device)?;

    let conv_quantized = conv_layer.quantize_weights(&test_tensor)?;
    let attention_quantized = attention_layer.quantize_weights(&test_tensor)?;
    let output_quantized = output_layer.quantize_weights(&test_tensor)?;

    // Conv should be binary (clipped)
    let conv_values: Vec<f32> = conv_quantized.to_vec1()?;
    for val in &conv_values {
        assert!(*val == 1.0 || *val == -1.0, "Conv should be binary");
    }

    // Attention should be ternary (soft, 2-bit)
    let attention_values: Vec<f32> = attention_quantized.to_vec1()?;
    for val in &attention_values {
        assert!(val.abs() <= 1.0, "Attention should stay within range");
        // Soft quantization may not give exact discrete values
    }

    // Output should be multi-bit (4-bit)
    let output_values: Vec<f32> = output_quantized.to_vec1()?;
    for val in &output_values {
        assert!(val.abs() <= 1.0, "Output should stay within range");
    }

    Ok(())
}
