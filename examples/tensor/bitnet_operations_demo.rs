//! BitNet-Specific Tensor Operations Demo
//!
//! This example demonstrates BitNet-specific tensor operations including:
//! - 1.58-bit quantization tensor operations
//! - BitLinear layer tensor operations
//! - Ternary weight tensor representations
//! - Mixed precision tensor operations
//! - Quantization-aware training (QAT) tensor support
//! - Integration with existing quantization infrastructure

use std::time::{Duration, Instant};
use std::sync::Arc;

use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel};
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::device::{auto_select_device, get_cpu_device};

// Import quantization components (these would be implemented in bitnet-quant)
use bitnet_quant::{
    QuantizedTensor, QuantizationConfig, QuantizationScheme,
    BitLinearLayer, TernaryWeight, CalibrationDataset,
    MixedPrecisionConfig, QATConfig
};

// =============================================================================
// BitNet Demo Configuration
// =============================================================================

struct BitNetDemoConfig {
    enable_quantization_demo: bool,
    enable_bitlinear_demo: bool,
    enable_mixed_precision_demo: bool,
    enable_qat_demo: bool,
    enable_performance_comparison: bool,
    batch_size: usize,
    sequence_length: usize,
    hidden_size: usize,
    num_heads: usize,
}

impl Default for BitNetDemoConfig {
    fn default() -> Self {
        Self {
            enable_quantization_demo: true,
            enable_bitlinear_demo: true,
            enable_mixed_precision_demo: true,
            enable_qat_demo: true,
            enable_performance_comparison: true,
            batch_size: 32,
            sequence_length: 512,
            hidden_size: 768,
            num_heads: 12,
        }
    }
}

// =============================================================================
// Main Demo Function
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 BitNet-Specific Tensor Operations Demo");
    println!("=========================================");

    let config = BitNetDemoConfig::default();
    let pool = create_bitnet_memory_pool()?;

    // Core BitNet demos
    if config.enable_quantization_demo {
        demo_bitnet_quantization(&pool, &config)?;
    }

    if config.enable_bitlinear_demo {
        demo_bitlinear_operations(&pool, &config)?;
    }

    if config.enable_mixed_precision_demo {
        demo_mixed_precision_operations(&pool, &config)?;
    }

    if config.enable_qat_demo {
        demo_quantization_aware_training(&pool, &config)?;
    }

    if config.enable_performance_comparison {
        demo_performance_comparison(&pool, &config)?;
    }

    // Advanced BitNet features
    demo_ternary_weight_operations(&pool, &config)?;
    demo_calibration_and_optimization(&pool, &config)?;
    demo_bitnet_inference_pipeline(&pool, &config)?;

    println!("\n🎉 BitNet operations demo completed successfully!");
    print_bitnet_statistics(&pool)?;

    Ok(())
}

// =============================================================================
// BitNet 1.58-bit Quantization Demo
// =============================================================================

fn demo_bitnet_quantization(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔢 1. BitNet 1.58-bit Quantization");
    println!("----------------------------------");

    let device = auto_select_device();

    // Create full precision tensor
    let fp_tensor = BitNetTensor::random_normal_with_pool(
        vec![config.hidden_size, config.hidden_size],
        0.0, 0.02, // mean=0, std=0.02 (typical transformer initialization)
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("📊 Original tensor:");
    println!("  Shape: {:?}", fp_tensor.shape());
    println!("  Data type: {:?}", fp_tensor.dtype());
    println!("  Memory size: {} bytes", fp_tensor.memory_size());

    // Create quantization configuration for 1.58-bit
    let quant_config = QuantizationConfig {
        scheme: QuantizationScheme::BitNet158,
        scale_calculation: ScaleCalculation::AbsMax,
        zero_point_calculation: ZeroPointCalculation::Symmetric,
        enable_bias_correction: true,
        calibration_method: CalibrationMethod::MinMax,
    };

    println!("\n🔢 Performing 1.58-bit quantization:");

    let start_time = Instant::now();
    let quantized_tensor = QuantizedTensor::quantize_tensor(
        &fp_tensor,
        &quant_config,
        pool.clone()
    )?;
    let quantization_time = start_time.elapsed();

    println!("  ✅ Quantization completed in {:?}", quantization_time);
    println!("  📊 Quantized tensor info:");
    println!("    Values: Ternary {-1, 0, +1}");
    println!("    Scale: {:.6}", quantized_tensor.scale());
    println!("    Zero point: {}", quantized_tensor.zero_point());
    println!("    Memory size: {} bytes ({:.1}% of original)",
             quantized_tensor.memory_size(),
             (quantized_tensor.memory_size() as f64 / fp_tensor.memory_size() as f64) * 100.0);

    // Demonstrate quantized arithmetic operations
    println!("\n🧮 Quantized arithmetic operations:");

    let quantized_tensor2 = QuantizedTensor::quantize_tensor(
        &fp_tensor.transpose(0, 1)?,
        &quant_config,
        pool.clone()
    )?;

    // Quantized matrix multiplication
    let start_time = Instant::now();
    let quantized_matmul = quantized_tensor.quantized_matmul(&quantized_tensor2)?;
    let quantized_time = start_time.elapsed();

    // Compare with full precision
    let start_time = Instant::now();
    let fp_matmul = fp_tensor.matmul(&fp_tensor.transpose(0, 1)?)?;
    let fp_time = start_time.elapsed();

    println!("  🔢 Quantized matmul: {:?}", quantized_time);
    println!("  💻 Full precision matmul: {:?}", fp_time);

    if fp_time > Duration::ZERO && quantized_time > Duration::ZERO {
        let speedup = fp_time.as_nanos() as f64 / quantized_time.as_nanos() as f64;
        println!("  🚀 Speedup: {:.2}x", speedup);
    }

    // Dequantization for accuracy comparison
    let dequantized = quantized_tensor.dequantize()?;
    let mse = compute_mse(&fp_tensor, &dequantized)?;

    println!("  📊 Quantization accuracy:");
    println!("    MSE: {:.6}", mse);
    println!("    PSNR: {:.2} dB", 20.0 * (1.0_f32 / mse.sqrt()).log10());

    // Show ternary weight distribution
    let weight_stats = quantized_tensor.get_value_distribution();
    println!("  📈 Ternary weight distribution:");
    println!("    -1 values: {} ({:.1}%)", weight_stats.negative_ones,
             weight_stats.negative_ones as f64 / weight_stats.total as f64 * 100.0);
    println!("     0 values: {} ({:.1}%)", weight_stats.zeros,
             weight_stats.zeros as f64 / weight_stats.total as f64 * 100.0);
    println!("    +1 values: {} ({:.1}%)", weight_stats.positive_ones,
             weight_stats.positive_ones as f64 / weight_stats.total as f64 * 100.0);

    println!("✅ BitNet quantization demo completed");
    Ok(())
}

// =============================================================================
// BitLinear Layer Operations Demo
// =============================================================================

fn demo_bitlinear_operations(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 2. BitLinear Layer Operations");
    println!("-------------------------------");

    let device = auto_select_device();

    // Create BitLinear layer
    let bitlinear_config = BitLinearConfig {
        input_dim: config.hidden_size,
        output_dim: config.hidden_size,
        bias: true,
        activation_quantization: ActivationQuantization::BitNet158,
        weight_quantization: WeightQuantization::BitNet158,
        normalization: LayerNormalization::RMSNorm,
    };

    println!("🏗️  Creating BitLinear layer:");
    println!("  Input dim: {}", bitlinear_config.input_dim);
    println!("  Output dim: {}", bitlinear_config.output_dim);
    println!("  Activation quantization: {:?}", bitlinear_config.activation_quantization);
    println!("  Weight quantization: {:?}", bitlinear_config.weight_quantization);

    let mut bitlinear_layer = BitLinearLayer::new_with_pool(
        bitlinear_config,
        device.clone(),
        pool.clone()
    )?;

    // Create input tensor (batch_size, sequence_length, hidden_size)
    let input_tensor = BitNetTensor::random_normal_with_pool(
        vec![config.batch_size, config.sequence_length, config.hidden_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("\n📊 Input tensor:");
    println!("  Shape: {:?}", input_tensor.shape());
    println!("  Memory: {} MB", input_tensor.memory_size() / 1024 / 1024);

    // Forward pass
    println!("\n⚡ BitLinear forward pass:");

    let start_time = Instant::now();
    let bitlinear_output = bitlinear_layer.forward(&input_tensor)?;
    let forward_time = start_time.elapsed();

    println!("  ✅ Forward pass completed in {:?}", forward_time);
    println!("  📊 Output shape: {:?}", bitlinear_output.shape());
    println!("  🔍 Output stats:");

    let output_mean = bitlinear_output.mean(None, false)?.get_scalar_value()?;
    let output_std = bitlinear_output.std(None, false)?.get_scalar_value()?;
    let output_min = bitlinear_output.min(None, false)?.get_scalar_value()?;
    let output_max = bitlinear_output.max(None, false)?.get_scalar_value()?;

    println!("    Mean: {:.4}", output_mean);
    println!("    Std:  {:.4}", output_std);
    println!("    Min:  {:.4}", output_min);
    println!("    Max:  {:.4}", output_max);

    // Compare with standard linear layer
    let standard_linear = StandardLinearLayer::new_with_pool(
        config.hidden_size,
        config.hidden_size,
        true, // bias
        device.clone(),
        pool.clone()
    )?;

    let start_time = Instant::now();
    let standard_output = standard_linear.forward(&input_tensor)?;
    let standard_time = start_time.elapsed();

    println!("\n🔄 Comparison with standard linear:");
    println!("  BitLinear time: {:?}", forward_time);
    println!("  Standard time:  {:?}", standard_time);

    if standard_time > Duration::ZERO && forward_time > Duration::ZERO {
        let speedup = standard_time.as_nanos() as f64 / forward_time.as_nanos() as f64;
        println!("  🚀 BitLinear speedup: {:.2}x", speedup);
    }

    // Memory efficiency comparison
    let bitlinear_memory = bitlinear_layer.memory_usage();
    let standard_memory = standard_linear.memory_usage();
    let memory_savings = (standard_memory as f64 - bitlinear_memory as f64) / standard_memory as f64;

    println!("  💾 Memory usage:");
    println!("    BitLinear: {} MB", bitlinear_memory / 1024 / 1024);
    println!("    Standard:  {} MB", standard_memory / 1024 / 1024);
    println!("    Savings:   {:.1}%", memory_savings * 100.0);

    // Demonstrate weight inspection
    println!("\n🔍 BitLinear weight analysis:");
    let weights = bitlinear_layer.get_quantized_weights();
    let weight_distribution = weights.get_value_distribution();

    println!("  📈 Weight distribution:");
    println!("    -1: {:.1}%", weight_distribution.negative_ones as f64 / weight_distribution.total as f64 * 100.0);
    println!("     0: {:.1}%", weight_distribution.zeros as f64 / weight_distribution.total as f64 * 100.0);
    println!("    +1: {:.1}%", weight_distribution.positive_ones as f64 / weight_distribution.total as f64 * 100.0);

    // Activation quantization analysis
    let activation_stats = bitlinear_layer.get_activation_stats();
    println!("  📊 Activation quantization stats:");
    println!("    Scale: {:.6}", activation_stats.scale);
    println!("    Clipping ratio: {:.2}%", activation_stats.clipping_ratio * 100.0);

    println!("✅ BitLinear operations demo completed");
    Ok(())
}

// =============================================================================
// Mixed Precision Operations Demo
// =============================================================================

fn demo_mixed_precision_operations(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎭 3. Mixed Precision Operations");
    println!("-------------------------------");

    let device = auto_select_device();

    // Create mixed precision configuration
    let mixed_config = MixedPrecisionConfig {
        attention_precision: PrecisionLevel::Float16,
        feedforward_precision: PrecisionLevel::BitNet158,
        embedding_precision: PrecisionLevel::Float32,
        layernorm_precision: PrecisionLevel::Float32,
        gradient_precision: PrecisionLevel::Float16,
        loss_scaling: LossScaling::Dynamic { initial_scale: 65536.0, growth_factor: 2.0 },
    };

    println!("🎯 Mixed precision configuration:");
    println!("  Attention: {:?}", mixed_config.attention_precision);
    println!("  Feed-forward: {:?}", mixed_config.feedforward_precision);
    println!("  Embeddings: {:?}", mixed_config.embedding_precision);
    println!("  Layer norm: {:?}", mixed_config.layernorm_precision);

    // Create tensors with different precisions
    println!("\n📊 Creating mixed precision tensors:");

    // High precision embeddings
    let embeddings = BitNetTensor::random_normal_with_pool(
        vec![config.batch_size, config.sequence_length, config.hidden_size],
        0.0, 0.02,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("  🔵 Embeddings: Float32, {} MB",
             embeddings.memory_size() / 1024 / 1024);

    // Medium precision attention weights
    let attention_weights = BitNetTensor::random_normal_with_pool(
        vec![config.hidden_size, config.hidden_size],
        0.0, 0.02,
        BitNetDType::Float16,
        device.clone(),
        pool.clone()
    )?;

    println!("  🟡 Attention: Float16, {} MB",
             attention_weights.memory_size() / 1024 / 1024);

    // Low precision feedforward weights (BitNet quantized)
    let ff_weights_fp = BitNetTensor::random_normal_with_pool(
        vec![config.hidden_size, config.hidden_size * 4],
        0.0, 0.02,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let quant_config = QuantizationConfig {
        scheme: QuantizationScheme::BitNet158,
        scale_calculation: ScaleCalculation::AbsMax,
        zero_point_calculation: ZeroPointCalculation::Symmetric,
        enable_bias_correction: true,
        calibration_method: CalibrationMethod::MinMax,
    };

    let ff_weights = QuantizedTensor::quantize_tensor(&ff_weights_fp, &quant_config, pool.clone())?;
    println!("  🔴 Feed-forward: BitNet 1.58-bit, {} MB",
             ff_weights.memory_size() / 1024 / 1024);

    // Demonstrate mixed precision computation
    println!("\n🧮 Mixed precision computations:");

    // Attention computation (Float16)
    let start_time = Instant::now();
    let attention_input = embeddings.to_dtype(BitNetDType::Float16)?;
    let attention_output = attention_input.matmul(&attention_weights)?;
    let attention_time = start_time.elapsed();

    println!("  🔍 Attention (Float16): {:?}", attention_time);

    // Feed-forward computation (BitNet quantized)
    let start_time = Instant::now();
    let ff_input = attention_output.to_dtype(BitNetDType::Float32)?; // Upcast for quantized ops
    let ff_output = ff_weights.quantized_matmul_tensor(&ff_input)?;
    let ff_time = start_time.elapsed();

    println!("  🚀 Feed-forward (BitNet): {:?}", ff_time);

    // Layer normalization (Float32)
    let start_time = Instant::now();
    let norm_input = ff_output.to_dtype(BitNetDType::Float32)?;
    let normalized = layer_norm(&norm_input, 1e-5)?;
    let norm_time = start_time.elapsed();

    println!("  📏 Layer norm (Float32): {:?}", norm_time);

    // Memory usage analysis
    let total_fp32_memory = embeddings.memory_size() + (ff_weights_fp.memory_size());
    let total_mixed_memory = embeddings.memory_size() + attention_weights.memory_size() + ff_weights.memory_size();
    let memory_savings = (total_fp32_memory as f64 - total_mixed_memory as f64) / total_fp32_memory as f64;

    println!("\n💾 Memory efficiency:");
    println!("  Full Float32: {} MB", total_fp32_memory / 1024 / 1024);
    println!("  Mixed precision: {} MB", total_mixed_memory / 1024 / 1024);
    println!("  Savings: {:.1}%", memory_savings * 100.0);

    // Accuracy analysis
    println!("\n📊 Accuracy analysis:");

    let fp32_result = compute_full_precision_forward(&embeddings, &ff_weights_fp, &attention_weights.to_dtype(BitNetDType::Float32)?)?;
    let mixed_result = normalized;

    let accuracy_mse = compute_mse(&fp32_result, &mixed_result)?;
    println!("  MSE vs Float32: {:.6}", accuracy_mse);

    let cosine_similarity = compute_cosine_similarity(&fp32_result, &mixed_result)?;
    println!("  Cosine similarity: {:.4}", cosine_similarity);

    // Gradient scaling demonstration
    if mixed_config.gradient_precision == PrecisionLevel::Float16 {
        println!("\n🎯 Gradient scaling:");

        let loss_scale = match mixed_config.loss_scaling {
            LossScaling::Dynamic { initial_scale, .. } => initial_scale,
            LossScaling::Fixed(scale) => scale,
        };

        println!("  Initial loss scale: {}", loss_scale);

        // Simulate gradient computation with scaling
        let scaled_gradients = mixed_result.mul_scalar(loss_scale)?;
        println!("  Gradients scaled by: {}", loss_scale);

        // Check for overflow/underflow
        let grad_max = scaled_gradients.max(None, false)?.get_scalar_value()?;
        let grad_min = scaled_gradients.min(None, false)?.get_scalar_value()?;

        if grad_max > 65504.0 || grad_min < -65504.0 { // Float16 limits
            println!("  ⚠️  Gradient overflow detected, should reduce scale");
        } else {
            println!("  ✅ Gradients within Float16 range");
        }
    }

    println!("✅ Mixed precision operations demo completed");
    Ok(())
}

// =============================================================================
// Quantization Aware Training (QAT) Demo
// =============================================================================

fn demo_quantization_aware_training(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎓 4. Quantization Aware Training (QAT)");
    println!("--------------------------------------");

    let device = auto_select_device();

    // Create QAT configuration
    let qat_config = QATConfig {
        quantization_scheme: QuantizationScheme::BitNet158,
        straight_through_estimator: true,
        temperature_scheduling: TemperatureScheduling::Exponential {
            initial: 1.0,
            decay: 0.95,
            min_temp: 0.1
        },
        sparsity_regularization: Some(SparsityConfig {
            target_sparsity: 0.1,
            regularization_weight: 1e-4,
        }),
        calibration_samples: 1000,
    };

    println!("🎯 QAT Configuration:");
    println!("  Quantization scheme: {:?}", qat_config.quantization_scheme);
    println!("  Straight-through estimator: {}", qat_config.straight_through_estimator);
    println!("  Temperature scheduling: {:?}", qat_config.temperature_scheduling);

    // Create QAT-aware layer
    let mut qat_layer = QATBitLinearLayer::new_with_pool(
        config.hidden_size,
        config.hidden_size,
        qat_config,
        device.clone(),
        pool.clone()
    )?;

    println!("\n🏗️  QAT-aware layer created");

    // Simulate training data
    let batch_size = 8;
    let num_batches = 10;

    println!("\n🎯 Simulating QAT training:");
    println!("  Batches: {}", num_batches);
    println!("  Batch size: {}", batch_size);

    let mut training_metrics = Vec::new();

    for epoch in 0..num_batches {
        // Create training batch
        let input_batch = BitNetTensor::random_normal_with_pool(
            vec![batch_size, config.sequence_length, config.hidden_size],
            0.0, 1.0,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        // Forward pass
        let start_time = Instant::now();
        let output = qat_layer.forward_training(&input_batch)?;
        let forward_time = start_time.elapsed();

        // Compute fake loss (MSE with random target)
        let target = BitNetTensor::random_normal_with_pool(
            output.shape().clone(),
            0.0, 1.0,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let loss_tensor = compute_mse_tensor(&output, &target)?;
        let loss_value = loss_tensor.mean(None, false)?.get_scalar_value()?;

        // Backward pass (simplified - in real implementation would use autograd)
        let start_time = Instant::now();
        let gradients = compute_gradients_ste(&qat_layer, &loss_tensor)?;
        let backward_time = start_time.elapsed();

        // Update parameters
        qat_layer.update_parameters(&gradients, 0.001)?; // learning_rate = 0.001

        // Update quantization parameters
        qat_layer.update_quantization_params(epoch as f32 / num_batches as f32)?;

        // Collect metrics
        let weight_stats = qat_layer.get_quantization_stats();
        training_metrics.push(TrainingMetrics {
            epoch,
            loss: loss_value,
            forward_time,
            backward_time,
            temperature: weight_stats.current_temperature,
            sparsity: weight_stats.sparsity_ratio,
            quantization_error: weight_stats.quantization_mse,
        });

        if epoch % 2 == 0 || epoch == num_batches - 1 {
            println!("  Epoch {}: loss={:.4}, temp={:.3}, sparsity={:.2}%",
                     epoch, loss_value, weight_stats.current_temperature, weight_stats.sparsity_ratio * 100.0);
        }
    }

    // Analyze training progression
    println!("\n📊 Training progression analysis:");

    let initial_loss = training_metrics[0].loss;
    let final_loss = training_metrics.last().unwrap().loss;
    let loss_reduction = (initial_loss - final_loss) / initial_loss;

    println!("  Loss reduction: {:.1}%", loss_reduction * 100.0);

    let final_temp = training_metrics.last().unwrap().temperature;
    let temp_decay = (1.0 - final_temp) * 100.0;
    println!("  Temperature decay: {:.1}%", temp_decay);

    let avg_sparsity = training_metrics.iter().map(|m| m.sparsity).sum::<f32>() / training_metrics.len() as f32;
    println!("  Average sparsity: {:.1}%", avg_sparsity * 100.0);

    // Final quantization
    println!("\n🔢 Final quantization:");

    let quantized_layer = qat_layer.finalize_quantization()?;
    let quantized_weights = quantized_layer.get_quantized_weights();
    let weight_distribution = quantized_weights.get_value_distribution();

    println!("  Final weight distribution:");
    println!("    -1: {} ({:.1}%)", weight_distribution.negative_ones,
             weight_distribution.negative_ones as f64 / weight_distribution.total as f64 * 100.0);
    println!("     0: {} ({:.1}%)", weight_distribution.zeros,
             weight_distribution.zeros as f64 / weight_distribution.total as f64 * 100.0);
    println!("    +1: {} ({:.1}%)", weight_distribution.positive_ones,
             weight_distribution.positive_ones as f64 / weight_distribution.total as f64 * 100.0);

    // Compare performance: QAT vs Post-Training Quantization
    println!("\n⚖️  QAT vs Post-Training Quantization:");

    // Create standard layer for comparison
    let standard_layer = StandardLinearLayer::new_with_pool(
        config.hidden_size,
        config.hidden_size,
        true,
        device.clone(),
        pool.clone()
    )?;

    let test_input = BitNetTensor::random_normal_with_pool(
        vec![1, config.sequence_length, config.hidden_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    // Standard output
    let standard_output = standard_layer.forward(&test_input)?;

    // QAT output
    let qat_output = quantized_layer.forward(&test_input)?;

    // Post-training quantized output
    let ptq_weights = QuantizedTensor::quantize_tensor(
        standard_layer.get_weights(),
        &QuantizationConfig {
            scheme: QuantizationScheme::BitNet158,
            scale_calculation: ScaleCalculation::AbsMax,
            zero_point_calculation: ZeroPointCalculation::Symmetric,
            enable_bias_correction: false,
            calibration_method: CalibrationMethod::MinMax,
        },
        pool.clone()
    )?;

    let ptq_output = ptq_weights.quantized_matmul_tensor(&test_input)?;

    // Accuracy comparison
    let qat_mse = compute_mse(&standard_output, &qat_output)?;
    let ptq_mse = compute_mse(&standard_output, &ptq_output)?;

    println!("  QAT MSE: {:.6}", qat_mse);
    println!("  PTQ MSE: {:.6}", ptq_mse);

    let accuracy_improvement = (ptq_mse - qat_mse) / ptq_mse;
    println!("  QAT accuracy improvement: {:.1}%", accuracy_improvement * 100.0);

    println!("✅ QAT demo completed");
    Ok(())
}

// =============================================================================
// Performance Comparison Demo
// =============================================================================

fn demo_performance_comparison(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏁 5. Performance Comparison");
    println!("---------------------------");

    let device = auto_select_device();

    println!("🎯 Comparing BitNet vs Standard operations");

    // Test configurations
    let test_sizes = vec![
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];

    println!("\n📊 Matrix multiplication benchmarks:");
    println!("  Size    | BitNet (ms) | Standard (ms) | Speedup | Memory Ratio");
    println!("  --------|-------------|---------------|---------|-------------");

    for (m, n) in test_sizes {
        // Create test tensors
        let fp_tensor_a = BitNetTensor::random_normal_with_pool(
            vec![m, n],
            0.0, 0.02,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        let fp_tensor_b = BitNetTensor::random_normal_with_pool(
            vec![n, m],
            0.0, 0.02,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        // Quantize for BitNet
        let quant_config = QuantizationConfig {
            scheme: QuantizationScheme::BitNet158,
            scale_calculation: ScaleCalculation::AbsMax,
            zero_point_calculation: ZeroPointCalculation::Symmetric,
            enable_bias_correction: true,
            calibration_method: CalibrationMethod::MinMax,
        };

        let bitnet_tensor_a = QuantizedTensor::quantize_tensor(&fp_tensor_a, &quant_config, pool.clone())?;
        let bitnet_tensor_b = QuantizedTensor::quantize_tensor(&fp_tensor_b, &quant_config, pool.clone())?;

        // Benchmark BitNet
        let iterations = 50;
        let start_time = Instant::now();
        for _ in 0..iterations {
            let _ = bitnet_tensor_a.quantized_matmul(&bitnet_tensor_b)?;
        }
        let bitnet_time = start_time.elapsed() / iterations as u32;

        // Benchmark standard
        let start_time = Instant::now();
        for _ in 0..iterations {
            let _ = fp_tensor_a.matmul(&fp_tensor_b)?;
        }
        let standard_time = start_time.elapsed() / iterations as u32;

        // Calculate metrics
        let speedup = standard_time.as_nanos() as f64 / bitnet_time.as_nanos() as f64;
        let memory_ratio = (fp_tensor_a.memory_size() + fp_tensor_b.memory_size()) as f64 /
                          (bitnet_tensor_a.memory_size() + bitnet_tensor_b.memory_size()) as f64;

        println!("  {:4}x{:<3} | {:9.2} | {:11.2} | {:5.2}x | {:9.1}x",
                 m, n,
                 bitnet_time.as_micros() as f64 / 1000.0,
                 standard_time.as_micros() as f64 / 1000.0,
                 speedup,
                 memory_ratio);
    }

    // Throughput benchmark
    println!("\n🚀 Throughput benchmark (ops/second):");

    let batch_sizes = vec![1, 8, 16, 32, 64];
    println!("  Batch Size | BitNet (ops/s) | Standard (ops/s) | Efficiency");
    println!("  -----------|----------------|------------------|-----------");

    for batch_size in batch_sizes {
        let input_tensor = BitNetTensor::random_normal_with_pool(
            vec![batch_size, config.sequence_length, config.hidden_size],
            0.0, 1.0,
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        // Create layers
        let bitlinear_layer = BitLinearLayer::new_with_pool(
            BitLinearConfig {
                input_dim: config.hidden_size,
                output_dim: config.hidden_size,
                bias: true,
                activation_quantization: ActivationQuantization::BitNet158,
                weight_quantization: WeightQuantization::BitNet158,
                normalization: LayerNormalization::None,
            },
            device.clone(),
            pool.clone()
        )?;

        let standard_layer = StandardLinearLayer::new_with_pool(
            config.hidden_size,
            config.hidden_size,
            true,
            device.clone(),
            pool.clone()
        )?;

        // Benchmark BitLinear
        let test_duration = Duration::from_secs(1);
        let start_time = Instant::now();
        let mut bitnet_ops = 0;

        while start_time.elapsed() < test_duration {
            let _ = bitlinear_layer.forward(&input_tensor)?;
            bitnet_ops += 1;
        }

        // Benchmark Standard
        let start_time = Instant::now();
        let mut standard_ops = 0;

        while start_time.elapsed() < test_duration {
            let _ = standard_layer.forward(&input_tensor)?;
            standard_ops += 1;
        }

        let efficiency = bitnet_ops as f64 / standard_ops as f64;

        println!("  {:9} | {:12} | {:14} | {:7.2}x",
                 batch_size, bitnet_ops, standard_ops, efficiency);
    }

    // Energy efficiency estimation
    println!("\n⚡ Energy efficiency estimation:");

    let bitnet_energy_per_op = 0.1; // Arbitrary units - much lower than standard
    let standard_energy_per_op = 1.0;

    let operations_per_inference = config.hidden_size * config.hidden_size * config.sequence_length;
    let bitnet_energy = operations_per_inference as f64 * bitnet_energy_per_op;
    let standard_energy = operations_per_inference as f64 * standard_energy_per_op;
    let energy_savings = (standard_energy - bitnet_energy) / standard_energy;

    println!("  BitNet energy: {:.1} units", bitnet_energy);
    println!("  Standard energy: {:.1} units", standard_energy);
    println!("  Energy savings: {:.1}%", energy_savings * 100.0);

    println!("✅ Performance comparison demo completed");
    Ok(())
}

// =============================================================================
// Additional Demo Functions
// =============================================================================

fn demo_ternary_weight_operations(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 6. Ternary Weight Operations");
    println!("------------------------------");

    let device = auto_select_device();

    // Create ternary weights
    let ternary_weights = TernaryWeight::from_distribution(
        vec![config.hidden_size, config.hidden_size],
        TernaryDistribution {
            prob_positive: 0.4,
            prob_negative: 0.4,
            prob_zero: 0.2,
        },
        device.clone(),
        pool.clone()
    )?;

    println!("✅ Created ternary weights: {:?}", ternary_weights.shape());

    // Demonstrate ternary arithmetic
    let input = BitNetTensor::random_normal_with_pool(
        vec![32, config.hidden_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let result = ternary_weights.ternary_matmul(&input)?;
    println!("✅ Ternary matrix multiplication completed");

    Ok(())
}

fn demo_calibration_and_optimization(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 7. Calibration and Optimization");
    println!("----------------------------------");

    let device = auto_select_device();

    // Create calibration dataset
    let calibration_data = CalibrationDataset::new(
        vec![64, config.sequence_length, config.hidden_size],
        1000, // num_samples
        device.clone(),
        pool.clone()
    )?;

    println!("✅ Created calibration dataset: {} samples", calibration_data.size());

    // Perform calibration
    let mut quantization_config = QuantizationConfig {
        scheme: QuantizationScheme::BitNet158,
        scale_calculation: ScaleCalculation::Percentile(99.9),
        zero_point_calculation: ZeroPointCalculation::Symmetric,
        enable_bias_correction: true,
        calibration_method: CalibrationMethod::KLDivergence,
    };

    let calibrated_config = calibration_data.calibrate_quantization(&quantization_config)?;
    println!("✅ Calibration completed");

    Ok(())
}

fn demo_bitnet_inference_pipeline(
    pool: &HybridMemoryPool,
    config: &BitNetDemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 8. BitNet Inference Pipeline");
    println!("-------------------------------");

    let device = auto_select_device();

    // Create a mini BitNet model
    let model = BitNetMiniModel::new_with_pool(
        config.hidden_size,
        config.num_heads,
        2, // num_layers
        device.clone(),
        pool.clone()
    )?;

    println!("✅ Created BitNet mini model");

    // Run inference
    let input_tokens = BitNetTensor::random_uniform_with_pool(
        vec![1, config.sequence_length, config.hidden_size],
        0.0, 1.0,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let start_time = Instant::now();
    let output = model.forward(&input_tokens)?;
    let inference_time = start_time.elapsed();

    println!("✅ Inference completed in {:?}", inference_time);
    println!("📊 Output shape: {:?}", output.shape());

    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

fn create_bitnet_memory_pool() -> Result<HybridMemoryPool, Box<dyn std::error::Error>> {
    let config = MemoryPoolConfig {
        small_block_size: 128 * 1024,      // 128KB for quantized tensors
        large_block_threshold: 2 * 1024 * 1024, // 2MB
        initial_pool_size: 64 * 1024 * 1024,    // 64MB
        max_pool_size: 1024 * 1024 * 1024,      // 1GB
        tracking: TrackingConfig {
            level: TrackingLevel::Detailed,
            enable_stack_traces: false,
            enable_metrics: true,
        },
    };

    HybridMemoryPool::new_with_config(config).map_err(Into::into)
}

fn compute_mse(tensor_a: &BitNetTensor, tensor_b: &BitNetTensor) -> Result<f32, Box<dyn std::error::Error>> {
    let diff = tensor_a.sub(tensor_b)?;
    let squared = diff.mul(&diff)?;
    let mse = squared.mean(None, false)?;
    Ok(mse.get_scalar_value()?)
}

fn compute_mse_tensor(tensor_a: &BitNetTensor, tensor_b: &BitNetTensor) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let diff = tensor_a.sub(tensor_b)?;
    let squared = diff.mul(&diff)?;
    Ok(squared)
}

fn compute_cosine_similarity(tensor_a: &BitNetTensor, tensor_b: &BitNetTensor) -> Result<f32, Box<dyn std::error::Error>> {
    let dot_product = tensor_a.flatten()?.dot(&tensor_b.flatten()?)?;
    let norm_a = tensor_a.norm(2, None, false)?.get_scalar_value()?;
    let norm_b = tensor_b.norm(2, None, false)?.get_scalar_value()?;
    Ok(dot_product.get_scalar_value()? / (norm_a * norm_b))
}

fn layer_norm(tensor: &BitNetTensor, eps: f32) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let mean = tensor.mean(Some(vec![-1]), true)?;
    let centered = tensor.sub(&mean)?;
    let variance = centered.mul(&centered)?.mean(Some(vec![-1]), true)?;
    let std = variance.add_scalar(eps)?.sqrt()?;
    Ok(centered.div(&std)?)
}

fn compute_full_precision_forward(
    embeddings: &BitNetTensor,
    weights: &BitNetTensor,
    attention_weights: &BitNetTensor
) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let attention_output = embeddings.matmul(attention_weights)?;
    let ff_output = attention_output.matmul(weights)?;
    layer_norm(&ff_output, 1e-5)
}

fn compute_gradients_ste(
    _layer: &QATBitLinearLayer,
    _loss: &BitNetTensor
) -> Result<ParameterGradients, Box<dyn std::error::Error>> {
    // Placeholder for straight-through estimator gradients
    // In real implementation, would compute gradients through quantization
    Ok(ParameterGradients::default())
}

fn print_bitnet_statistics(pool: &HybridMemoryPool) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = pool.get_metrics();

    println!("\n📊 BitNet Demo Statistics:");
    println!("-------------------------");
    println!("💾 Memory used: {} MB", metrics.total_allocated / 1024 / 1024);
    println!("🔢 Quantized operations: Enabled");
    println!("⚡ Acceleration: Device-optimized");
    println!("✅ Demo completed successfully");

    Ok(())
}

// Data structures for demo (would be implemented in actual bitnet-quant crate)
#[derive(Debug, Clone)]
struct TrainingMetrics {
    epoch: usize,
    loss: f32,
    forward_time: Duration,
    backward_time: Duration,
    temperature: f32,
    sparsity: f32,
    quantization_error: f32,
}

#[derive(Default)]
struct ParameterGradients {
    // Gradient information
}

// Note: The actual implementations of these types would be in the bitnet-quant crate
// This is a demonstration of the API and functionality
