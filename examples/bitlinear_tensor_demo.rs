//! BitLinear Layer Tensor Operations Demonstration
//!
//! This example demonstrates the comprehensive BitLinear layer tensor operations
//! including weight quantization, activation quantization, LayerNorm integration,
//! residual connections, mixed precision support, and QAT operations.

use std::collections::HashMap;
use candle_core::Device;

use bitnet_core::{BitNetTensor, BitNetDType, TensorShape, auto_select_device};
use bitnet_quant::tensor_integration::{
    BitLinearTensorOps, BitLinearConfig, QATTensorOps, QATConfig, QATTensorType,
    MixedPrecisionBitLinearOps, MixedPrecisionConfig, HardwareProfile, HardwareDeviceType
};
use bitnet_quant::quantization::{QuantizationPrecision, QuantizationStrategy};
use bitnet_quant::quantization::bitnet_ops::BitNetQuantizationConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitLinear Layer Tensor Operations Demonstration");
    println!("================================================\n");

    // Initialize device (prefer MLX/Metal for Apple Silicon)
    let device = auto_select_device().unwrap_or(Device::Cpu);
    println!("Selected device: {:?}\n", device);

    // Demo 1: Basic BitLinear Operations
    basic_bitlinear_operations(&device)?;

    // Demo 2: Weight and Activation Quantization
    weight_activation_quantization(&device)?;

    // Demo 3: LayerNorm Integration
    layernorm_integration(&device)?;

    // Demo 4: Residual Connection Support
    residual_connection_support(&device)?;

    // Demo 5: Mixed Precision Operations
    mixed_precision_operations(&device)?;

    // Demo 6: QAT (Quantization-Aware Training) Operations
    qat_operations(&device)?;

    // Demo 7: Hardware-Optimized Operations
    hardware_optimized_operations(&device)?;

    println!("✅ All BitLinear tensor operations completed successfully!");

    Ok(())
}

fn basic_bitlinear_operations(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Basic BitLinear Operations");
    println!("=====================================");

    // Create BitLinear configuration
    let config = BitLinearConfig {
        weight_quantization: BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            use_memory_pool: true,
            enable_validation: true,
        },
        activation_quantization: BitNetQuantizationConfig {
            precision: QuantizationPrecision::EightBit,
            strategy: QuantizationStrategy::Dynamic,
            use_memory_pool: true,
            enable_validation: true,
        },
        enable_layernorm: true,
        enable_residual: true,
        device: Some(device.clone()),
        ..Default::default()
    };

    // Create BitLinear tensor operations
    let bitlinear_ops = BitLinearTensorOps::new(config);

    // Create test tensors
    let input_shape = vec![4, 16]; // [batch_size, input_dim]
    let weight_shape = vec![16, 32]; // [input_dim, output_dim]

    let input = BitNetTensor::randn(&input_shape, BitNetDType::F32, Some(device.clone()))?;
    let weights = BitNetTensor::randn(&weight_shape, BitNetDType::F32, Some(device.clone()))?;
    let bias = BitNetTensor::randn(&[32], BitNetDType::F32, Some(device.clone()))?;

    println!("  Input shape: {:?}", input.shape());
    println!("  Weight shape: {:?}", weights.shape());
    println!("  Bias shape: {:?}", bias.shape());

    // Quantize weights
    let quantized_weights = bitlinear_ops.quantize_weights(weights)?;
    println!("  ✓ Weight quantization completed");
    println!("    Original dtype: {:?}", quantized_weights.weights.dtype());
    println!("    Quantized precision: {:?}", quantized_weights.quantization_config.precision);

    // Quantize activations
    let quantized_activations = bitlinear_ops.quantize_activations(input.clone())?;
    println!("  ✓ Activation quantization completed");
    println!("    Activation stats - samples: {}", quantized_activations.activation_stats.sample_count);
    println!("    Activation stats - mean: {:.4}", quantized_activations.activation_stats.running_mean);

    // Perform forward pass
    let output = bitlinear_ops.forward(
        &input,
        &quantized_weights,
        Some(&bias),
        None, // No LayerNorm for this demo
        None  // No residual connection for this demo
    )?;

    println!("  ✓ Forward pass completed");
    println!("    Output shape: {:?}", output.shape());
    println!("    Output dtype: {:?}\n", output.dtype());

    Ok(())
}

fn weight_activation_quantization(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ Demo 2: Weight and Activation Quantization");
    println!("==============================================");

    let config = BitLinearConfig::default();
    let bitlinear_ops = BitLinearTensorOps::new(config);

    // Test different quantization precisions
    let test_tensor = BitNetTensor::randn(&[8, 64], BitNetDType::F32, Some(device.clone()))?;

    // Test ternary quantization (1.58-bit)
    let ternary_config = BitLinearConfig {
        weight_quantization: BitNetQuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            use_memory_pool: true,
            enable_validation: true,
        },
        ..Default::default()
    };

    let ternary_ops = BitLinearTensorOps::new(ternary_config);
    let ternary_weights = ternary_ops.quantize_weights(test_tensor.clone())?;

    println!("  Ternary Quantization (1.58-bit):");
    println!("    ✓ Quantized to precision: {:?}", ternary_weights.quantization_config.precision);

    // Test 4-bit quantization
    let fourbit_config = BitLinearConfig {
        weight_quantization: BitNetQuantizationConfig {
            precision: QuantizationPrecision::FourBit,
            strategy: QuantizationStrategy::Asymmetric,
            use_memory_pool: true,
            enable_validation: true,
        },
        ..Default::default()
    };

    let fourbit_ops = BitLinearTensorOps::new(fourbit_config);
    let fourbit_weights = fourbit_ops.quantize_weights(test_tensor.clone())?;

    println!("  4-bit Quantization:");
    println!("    ✓ Quantized to precision: {:?}", fourbit_weights.quantization_config.precision);

    // Test 8-bit quantization
    let eightbit_config = BitLinearConfig {
        activation_quantization: BitNetQuantizationConfig {
            precision: QuantizationPrecision::EightBit,
            strategy: QuantizationStrategy::Dynamic,
            use_memory_pool: true,
            enable_validation: true,
        },
        ..Default::default()
    };

    let eightbit_ops = BitLinearTensorOps::new(eightbit_config);
    let eightbit_activations = eightbit_ops.quantize_activations(test_tensor)?;

    println!("  8-bit Activation Quantization:");
    println!("    ✓ Quantized to precision: {:?}", eightbit_activations.quantization_config.precision);
    println!("    Statistics - min: {:.4}, max: {:.4}\n",
             eightbit_activations.activation_stats.min_val,
             eightbit_activations.activation_stats.max_val);

    Ok(())
}

fn layernorm_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Demo 3: LayerNorm Integration");
    println!("=================================");

    let config = BitLinearConfig {
        enable_layernorm: true,
        layernorm_eps: 1e-5,
        ..Default::default()
    };

    let bitlinear_ops = BitLinearTensorOps::new(config);

    // Create LayerNorm integration
    let normalized_shape = [64];
    let layernorm = bitlinear_ops.create_layernorm_integration(&normalized_shape, device.clone())?;

    println!("  LayerNorm Configuration:");
    println!("    Normalized shape: {:?}", layernorm.normalized_shape);
    println!("    Epsilon: {}", layernorm.eps);
    println!("    Weight shape: {:?}", layernorm.weight.shape());
    println!("    Bias shape: {:?}", layernorm.bias.shape());

    // Test LayerNorm application
    let input = BitNetTensor::randn(&[4, 64], BitNetDType::F32, Some(device.clone()))?;
    let normalized = bitlinear_ops.apply_layernorm(&input, &layernorm)?;

    println!("  LayerNorm Application:");
    println!("    Input shape: {:?}", input.shape());
    println!("    Output shape: {:?}", normalized.shape());

    // Verify normalization properties (approximate)
    let candle_tensor = normalized.to_candle_tensor()?;
    let mean = candle_tensor.mean_keepdim(1)?.to_vec2::<f32>()?;
    let var = candle_tensor.var_keepdim(1)?.to_vec2::<f32>()?;

    println!("    ✓ Normalized mean (should be ~0): {:.6}", mean[0][0]);
    println!("    ✓ Normalized variance (should be ~1): {:.6}\n", var[0][0]);

    Ok(())
}

fn residual_connection_support(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔀 Demo 4: Residual Connection Support");
    println!("=======================================");

    let config = BitLinearConfig {
        enable_residual: true,
        ..Default::default()
    };

    let bitlinear_ops = BitLinearTensorOps::new(config);

    // Create residual connection support
    let residual_support = bitlinear_ops.create_residual_support(device.clone());

    println!("  Residual Connection Configuration:");
    println!("    Enabled: {}", residual_support.enabled);
    println!("    Scale factor: {}", residual_support.scale_factor);

    // Test residual connection
    let main_path = BitNetTensor::randn(&[4, 32], BitNetDType::F32, Some(device.clone()))?;
    let residual = BitNetTensor::randn(&[4, 32], BitNetDType::F32, Some(device.clone()))?;

    let result = bitlinear_ops.apply_residual_connection(&main_path, &residual)?;

    println!("  Residual Connection Application:");
    println!("    Main path shape: {:?}", main_path.shape());
    println!("    Residual shape: {:?}", residual.shape());
    println!("    Result shape: {:?}", result.shape());
    println!("    ✓ Residual connection completed successfully\n");

    Ok(())
}

fn mixed_precision_operations(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Demo 5: Mixed Precision Operations");
    println!("======================================");

    let base_config = BitLinearConfig::default();
    let bitlinear_ops = BitLinearTensorOps::new(base_config);

    // Create mixed precision operations
    let mut mixed_precision_ops = bitlinear_ops.create_mixed_precision_ops(
        QuantizationPrecision::OneFiveFiveBit, // Weight precision
        QuantizationPrecision::EightBit        // Activation precision
    );

    println!("  Mixed Precision Configuration:");
    println!("    Weight precision: {:?}", mixed_precision_ops.weight_precision);
    println!("    Activation precision: {:?}", mixed_precision_ops.activation_precision);

    // Create test tensors
    let input = BitNetTensor::randn(&[2, 16], BitNetDType::F32, Some(device.clone()))?;
    let weights = BitNetTensor::randn(&[16, 8], BitNetDType::F32, Some(device.clone()))?;
    let bias = BitNetTensor::randn(&[8], BitNetDType::F32, Some(device.clone()))?;

    // Quantize weights first
    let quantized_weights = bitlinear_ops.quantize_weights(weights)?;

    // Perform mixed precision forward pass
    let output = mixed_precision_ops.forward(&input, &quantized_weights, Some(&bias))?;

    println!("  Mixed Precision Forward Pass:");
    println!("    Input shape: {:?}", input.shape());
    println!("    Output shape: {:?}", output.shape());

    // Get precision summary
    let summary = mixed_precision_ops.get_precision_summary();
    println!("    Total operations: {}", summary.total_operations);
    println!("    Memory efficiency: {:.2}%", summary.memory_efficiency * 100.0);
    println!("    ✓ Mixed precision operations completed\n");

    Ok(())
}

fn qat_operations(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Demo 6: QAT (Quantization-Aware Training) Operations");
    println!("=======================================================");

    // Create QAT configuration
    let qat_config = QATConfig {
        enable_ste: true,
        weight_precision: QuantizationPrecision::OneFiveFiveBit,
        activation_precision: QuantizationPrecision::EightBit,
        gradient_clip_threshold: Some(1.0),
        temperature: 1.0,
        fake_quantization: true,
        device: Some(device.clone()),
        ..Default::default()
    };

    let mut qat_ops = QATTensorOps::new(qat_config);

    println!("  QAT Configuration:");
    println!("    STE enabled: {}", qat_ops.ste.config.enable_ste);
    println!("    Weight precision: {:?}", qat_ops.ste.config.weight_precision);
    println!("    Activation precision: {:?}", qat_ops.ste.config.activation_precision);
    println!("    Fake quantization: {}", qat_ops.ste.config.fake_quantization);

    // Test different QAT operations
    let test_tensor = BitNetTensor::randn(&[4, 32], BitNetDType::F32, Some(device.clone()))?;

    // Test weight quantization with STE
    let qat_weights = qat_ops.apply_qat(&test_tensor, "layer1_weights", QATTensorType::Weight)?;
    println!("    ✓ Weight QAT applied");

    // Test activation quantization with STE
    let qat_activations = qat_ops.apply_qat(&test_tensor, "layer1_activations", QATTensorType::Activation)?;
    println!("    ✓ Activation QAT applied");

    // Test binary quantization with STE
    let qat_binary = qat_ops.apply_qat(&test_tensor, "layer1_binary", QATTensorType::Binary)?;
    println!("    ✓ Binary QAT applied");

    // Test ternary quantization with STE
    let qat_ternary = qat_ops.apply_qat(&test_tensor, "layer1_ternary", QATTensorType::Ternary(0.5))?;
    println!("    ✓ Ternary QAT applied");

    // Step the training process
    qat_ops.step();

    // Get QAT statistics
    let qat_stats = qat_ops.get_qat_stats();
    println!("  QAT Statistics:");
    println!("    Total quantizations: {}", qat_stats.total_quantizations);
    println!("    Average quantization error: {:.6}", qat_stats.avg_quantization_error);
    println!("    Gradient updates: {}", qat_stats.gradient_stats.num_updates);
    println!("    ✓ QAT operations completed\n");

    Ok(())
}

fn hardware_optimized_operations(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Demo 7: Hardware-Optimized Operations");
    println!("=========================================");

    let config = BitLinearConfig::default();
    let bitlinear_ops = BitLinearTensorOps::new(config);

    // Create hardware profile
    let hardware_profile = HardwareProfile {
        device_type: match device {
            Device::Metal(_) => HardwareDeviceType::AppleSilicon,
            Device::Cuda(_) => HardwareDeviceType::NvidiaGPU,
            Device::Cpu => HardwareDeviceType::CPU,
        },
        memory_mb: 8192,
        compute_units: 32,
        instruction_sets: vec![],
        optimal_tile_sizes: vec![32, 64, 128],
    };

    println!("  Hardware Profile:");
    println!("    Device type: {:?}", hardware_profile.device_type);
    println!("    Memory: {} MB", hardware_profile.memory_mb);
    println!("    Compute units: {}", hardware_profile.compute_units);

    // Test hardware-optimized tensor operations
    let test_tensor = BitNetTensor::randn(&[64, 128], BitNetDType::F32, Some(device.clone()))?;
    let optimized_tensor = bitlinear_ops.optimize_for_hardware(&test_tensor, &hardware_profile)?;

    println!("  Hardware Optimization:");
    println!("    Input shape: {:?}", test_tensor.shape());
    println!("    Optimized shape: {:?}", optimized_tensor.shape());
    println!("    ✓ Hardware optimization applied\n");

    Ok(())
}
