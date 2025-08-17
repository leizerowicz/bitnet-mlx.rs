//! BitLinear Layer Usage Example
//!
//! This example demonstrates how to use the BitLinear layer with various configurations,
//! showcasing the integration with device abstraction and memory management.

use anyhow::Result;
use candle_core::Tensor;
use bitnet_quant::bitlinear::{BitLinear, BitLinearConfig, BitLinearForward};
use bitnet_quant::quantization::{QuantizationPrecision, QuantizationStrategy, TernaryMethod, QuantizationConfig};
use bitnet_core::device::auto_select_device;

fn main() -> Result<()> {
    println!("🔥 BitLinear Layer Demo");
    println!("========================");

    // 1. Basic BitLinear layer with default configuration
    println!("\n1. Creating basic BitLinear layer...");
    let basic_config = BitLinearConfig {
        in_features: 768,
        out_features: 3072,
        use_bias: false, // BitNet standard is bias-free
        ..Default::default()
    };
    
    let basic_layer = BitLinear::new(basic_config, "ffn_up".to_string())?;
    println!("   ✓ Created layer: {:?}", basic_layer);
    
    // 2. Create test input
    println!("\n2. Creating test input tensor...");
    let device = auto_select_device();
    println!("   ✓ Using device: {:?}", device);
    
    let batch_size = 4;
    let seq_len = 128;
    let input_shape = [batch_size, seq_len, basic_layer.config().in_features];
    let input = Tensor::randn(0.0f32, 1.0f32, &input_shape, &device)?.to_dtype(candle_core::DType::F32)?;
    println!("   ✓ Input shape: {:?}", input.shape());
    
    // 3. Perform forward pass
    println!("\n3. Performing forward pass...");
    let output = basic_layer.forward(&input)?;
    println!("   ✓ Output shape: {:?}", output.shape());
    println!("   ✓ Forward pass completed successfully");
    
    // 4. BitLinear layer with bias enabled (rare in BitNet)
    println!("\n4. Creating BitLinear layer with bias...");
    let bias_config = BitLinearConfig {
        in_features: 512,
        out_features: 512,
        use_bias: true,
        ..Default::default()
    };
    
    let bias_layer = BitLinear::new(bias_config, "with_bias".to_string())?;
    let bias_input = Tensor::randn(0.0f32, 1.0f32, &[2, 512], &device)?.to_dtype(candle_core::DType::F32)?;
    let bias_output = bias_layer.forward(&bias_input)?;
    println!("   ✓ Bias layer output shape: {:?}", bias_output.shape());
    
    // 5. Custom quantization configuration
    println!("\n5. Creating BitLinear with custom quantization...");
    let custom_quant_config = BitLinearConfig {
        in_features: 256,
        out_features: 1024,
        use_bias: false,
        weight_quantization: bitnet_quant::quantization::WeightQuantizationConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneFiveFiveBit,
                strategy: QuantizationStrategy::Static,
                per_channel: false,
                clip_threshold: Some(3.0),
                qat_enabled: false,
                calibration_size: None,
            },
            ternary_method: TernaryMethod::OptimalThreshold,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            ..Default::default()
        },
        enable_caching: true,
        cache_size_limit: Some(64),
        ..Default::default()
    };
    
    let custom_layer = BitLinear::new(custom_quant_config, "custom_quant".to_string())?;
    let custom_input = Tensor::randn(0.0f32, 1.0f32, &[8, 256], &device)?.to_dtype(candle_core::DType::F32)?;
    let custom_output = custom_layer.forward(&custom_input)?;
    println!("   ✓ Custom quantization output shape: {:?}", custom_output.shape());
    
    // 6. Demonstrate weight updates and cache invalidation
    println!("\n6. Testing weight updates and cache behavior...");
    let update_config = BitLinearConfig {
        in_features: 128,
        out_features: 256,
        enable_caching: true,
        ..Default::default()
    };
    
    let update_layer = BitLinear::new(update_config, "update_test".to_string())?;
    let test_input = Tensor::randn(0.0f32, 1.0f32, &[1, 128], &device)?.to_dtype(candle_core::DType::F32)?;
    
    // First forward pass (cache miss)
    let output1 = update_layer.forward(&test_input)?;
    println!("   ✓ First forward pass completed");
    
    // Second forward pass (cache hit)
    let output2 = update_layer.forward(&test_input)?;
    println!("   ✓ Second forward pass completed (should use cache)");
    
    // Update weights (invalidates cache)
    let new_weights = Tensor::randn(0.0f32, 0.1f32, &[256, 128], &device)?.to_dtype(candle_core::DType::F32)?;
    update_layer.update_weights(new_weights)?;
    println!("   ✓ Weights updated (cache invalidated)");
    
    // Third forward pass (cache miss after update)
    let output3 = update_layer.forward(&test_input)?;
    println!("   ✓ Third forward pass completed (cache rebuilt)");
    
    // Verify outputs have different values after weight update
    let diff = output1.sub(&output3)?;
    let abs_diff = diff.abs()?;
    let max_diff = abs_diff.flatten_all()?.max(candle_core::D::Minus1)?.to_scalar::<f32>()?;
    println!("   ✓ Max difference after weight update: {:.6}", max_diff);
    
    // 7. Cache statistics
    if let Some(cache_stats) = update_layer.cache_stats() {
        println!("\n7. Cache statistics:");
        println!("   ✓ Cache hits: {}", cache_stats.hits);
        println!("   ✓ Cache misses: {}", cache_stats.misses);
        println!("   ✓ Hit ratio: {:.2}%", cache_stats.hit_ratio() * 100.0);
        println!("   ✓ Current entries: {}", cache_stats.entries);
        println!("   ✓ Total size: {} bytes", cache_stats.total_size_bytes);
    }
    
    // 8. Memory statistics
    println!("\n8. Memory pool statistics:");
    if let Ok(memory_stats) = update_layer.memory_stats() {
        println!("   ✓ Total allocated: {} bytes", memory_stats.total_allocated);
        println!("   ✓ Total deallocated: {} bytes", memory_stats.total_deallocated);
        println!("   ✓ Current allocated: {} bytes", memory_stats.current_allocated);
        println!("   ✓ Peak allocated: {} bytes", memory_stats.peak_allocated);
    }
    
    // 9. Device movement demonstration
    println!("\n9. Testing device movement...");
    let mut movable_layer = BitLinear::new(
        BitLinearConfig {
            in_features: 64,
            out_features: 32,
            ..Default::default()
        },
        "movable".to_string()
    )?;
    
    let cpu_device = bitnet_core::device::get_cpu_device();
    println!("   ✓ Original device: {:?}", movable_layer.device());
    
    movable_layer.to_device(cpu_device)?;
    println!("   ✓ Moved to device: {:?}", movable_layer.device());
    
    // 10. Batch processing demonstration
    println!("\n10. Batch processing demonstration...");
    let batch_layer = BitLinear::new(
        BitLinearConfig {
            in_features: 32,
            out_features: 16,
            ..Default::default()
        },
        "batch_test".to_string()
    )?;
    
    let batch_input = Tensor::randn(0.0f32, 1.0f32, &[16, 10, 32], &device)?.to_dtype(candle_core::DType::F32)?;
    let batch_output = batch_layer.forward_batch(&batch_input)?;
    println!("   ✓ Batch input shape: {:?}", batch_input.shape());
    println!("   ✓ Batch output shape: {:?}", batch_output.shape());
    
    println!("\n🎉 BitLinear demo completed successfully!");
    println!("Key features demonstrated:");
    println!("  • Full-precision weights with quantized inference");
    println!("  • Device abstraction integration");
    println!("  • Memory pool management");
    println!("  • Quantized weight caching");
    println!("  • Weight updates with cache invalidation");
    println!("  • Bias-free operations (BitNet standard)");
    println!("  • Custom quantization configurations");
    println!("  • Batch processing support");
    println!("  • Thread-safe operations");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitlinear_basic_functionality() -> Result<()> {
        let config = BitLinearConfig {
            in_features: 4,
            out_features: 2,
            ..Default::default()
        };
        
        let layer = BitLinear::new(config, "test".to_string())?;
        let device = bitnet_core::device::get_cpu_device();
        let input = Tensor::ones(&[1, 4], DType::F32, &device)?;
        
        let output = layer.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 2]);
        
        Ok(())
    }
}
