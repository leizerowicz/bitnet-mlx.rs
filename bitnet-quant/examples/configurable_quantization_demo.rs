//! Configurable Quantization Schemes Demo
//! 
//! This example demonstrates how to use the configurable quantization schemes
//! for both 1-bit and 1.58-bit quantization with various configuration options.

use bitnet_quant::prelude::*;
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 BitNet Configurable Quantization Schemes Demo");
    println!("================================================\n");

    let device = Device::Cpu;
    
    // Create sample weight tensors
    let weights_2d = create_sample_weights_2d(&device)?;
    let weights_1d = create_sample_weights_1d(&device)?;
    
    println!("📊 Sample Data:");
    println!("2D Weights shape: {:?}", weights_2d.shape());
    println!("1D Weights shape: {:?}", weights_1d.shape());
    println!("2D Weights range: [{:.3}, {:.3}]", 
        weights_2d.min_all()?.to_scalar::<f32>()?,
        weights_2d.max_all()?.to_scalar::<f32>()?);
    println!();

    // Demo 1: 1-bit quantization with different threshold methods
    demo_one_bit_quantization(&weights_2d, &device)?;
    
    // Demo 2: 1.58-bit quantization with different configurations
    demo_one_five_eight_bit_quantization(&weights_2d, &device)?;
    
    // Demo 3: Performance comparison
    demo_performance_comparison(&weights_1d, &device)?;
    
    // Demo 4: Custom configuration
    demo_custom_configuration(&weights_2d, &device)?;
    
    // Demo 5: Quantization scheme factory
    demo_scheme_factory(&weights_2d, &device)?;

    println!("✅ Demo completed successfully!");
    Ok(())
}

fn create_sample_weights_2d(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create a realistic weight matrix with different value ranges
    let data: Vec<f32> = (0..64*128)
        .map(|i| {
            let x = i as f32 / 100.0;
            match i % 4 {
                0 => (x.sin() * 2.0).tanh(),           // Large values
                1 => (x.cos() * 0.5).tanh(),           // Medium values  
                2 => (x.sin() * 0.1).tanh(),           // Small values
                _ => if i % 10 == 0 { 0.0 } else { (x * 0.01).tanh() }, // Sparse values
            }
        })
        .collect();
    
    let tensor = Tensor::from_vec(data, (64, 128), device)?;
    Ok(tensor)
}

fn create_sample_weights_1d(device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let data: Vec<f32> = vec![
        2.5, -1.8, 0.3, -0.1, 0.0, 1.2, -2.1, 0.8, -0.4, 1.5,
        -0.9, 0.2, -1.3, 0.7, -0.2, 1.1, -0.6, 0.4, -1.7, 0.9
    ];
    let tensor = Tensor::from_vec(data, (20,), device)?;
    Ok(tensor)
}

fn demo_one_bit_quantization(weights: &Tensor, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔹 1-Bit Quantization Demo");
    println!("==========================");
    
    let threshold_methods = [
        (BinaryThresholdMethod::Zero, "Zero Threshold"),
        (BinaryThresholdMethod::Mean, "Mean Threshold"),
        (BinaryThresholdMethod::Adaptive, "Adaptive Threshold"),
        (BinaryThresholdMethod::Optimal, "Optimal Threshold"),
    ];
    
    for (method, name) in threshold_methods {
        println!("\n📋 Method: {name}");
        
        // Create 1-bit quantization scheme
        let config = QuantizationSchemeConfig {
            base: QuantizationConfig {
                precision: QuantizationPrecision::OneBit,
                strategy: QuantizationStrategy::Symmetric,
                ..Default::default()
            },
            scheme_params: SchemeParameters {
                one_bit: OneBitParams {
                    threshold_method: method,
                    sign_based: matches!(method, BinaryThresholdMethod::Zero),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut scheme = ConfigurableQuantizationScheme::new(config, device.clone());
        
        // Quantize
        let start = Instant::now();
        let quantized = scheme.quantize_tensor(weights)?;
        let quantize_time = start.elapsed();
        
        // Dequantize
        let start = Instant::now();
        let dequantized = scheme.dequantize_tensor(&quantized)?;
        let dequantize_time = start.elapsed();
        
        // Calculate error
        let error = weights.sub(&dequantized)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        
        println!("  ⚡ Quantization time: {quantize_time:?}");
        println!("  ⚡ Dequantization time: {dequantize_time:?}");
        println!("  📊 Quantization error (MSE): {error:.6}");
        println!("  💾 Compression ratio: {:.2}x", quantized.compression_ratio());
        println!("  📏 Memory footprint: {} bytes", quantized.memory_footprint());
        
        // Show value distribution
        let values = quantized.values.flatten_all()?.to_vec1::<f32>()?;
        let positive_count = values.iter().filter(|&&x| x > 0.0).count();
        let negative_count = values.iter().filter(|&&x| x < 0.0).count();
        println!("  📈 Value distribution: +1: {positive_count}, -1: {negative_count}");
    }
    
    println!();
    Ok(())
}

fn demo_one_five_eight_bit_quantization(weights: &Tensor, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔸 1.58-Bit Quantization Demo");
    println!("==============================");
    
    let ternary_methods = [
        (TernaryMethod::MeanThreshold, "Mean Threshold"),
        (TernaryMethod::MedianThreshold, "Median Threshold"),
        (TernaryMethod::AdaptiveThreshold, "Adaptive Threshold"),
        (TernaryMethod::OptimalThreshold, "Optimal Threshold"),
    ];
    
    let threshold_factors = [0.5, 0.7, 0.9];
    
    for (method, method_name) in ternary_methods {
        println!("\n📋 Method: {method_name}");
        
        for &factor in &threshold_factors {
            println!("  🎯 Threshold factor: {factor}");
            
            let config = QuantizationSchemeConfig {
                base: QuantizationConfig {
                    precision: QuantizationPrecision::OneFiveFiveBit,
                    strategy: QuantizationStrategy::Symmetric,
                    ..Default::default()
                },
                scheme_params: SchemeParameters {
                    one_five_eight_bit: OneFiveEightBitParams {
                        ternary_method: method,
                        threshold_factor: factor,
                        balanced_ternary: false,
                        sparsity_target: None,
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            
            let mut scheme = ConfigurableQuantizationScheme::new(config, device.clone());
            
            // Quantize and measure performance
            let start = Instant::now();
            let quantized = scheme.quantize_tensor(weights)?;
            let quantize_time = start.elapsed();
            
            let dequantized = scheme.dequantize_tensor(&quantized)?;
            let error = weights.sub(&dequantized)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            
            // Calculate sparsity
            let values = quantized.values.flatten_all()?.to_vec1::<f32>()?;
            let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
            let positive_count = values.iter().filter(|&&x| x > 1e-6).count();
            let negative_count = values.iter().filter(|&&x| x < -1e-6).count();
            let sparsity = zero_count as f32 / values.len() as f32;
            
            println!("    ⚡ Time: {quantize_time:?}");
            println!("    📊 MSE: {error:.6}");
            println!("    💾 Compression: {:.2}x", quantized.compression_ratio());
            println!("    🕳️  Sparsity: {:.1}%", sparsity * 100.0);
            println!("    📈 Distribution: +1: {positive_count}, 0: {zero_count}, -1: {negative_count}");
        }
    }
    
    println!();
    Ok(())
}

fn demo_performance_comparison(weights: &Tensor, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Performance Comparison");
    println!("=========================");
    
    let precisions = [
        (QuantizationPrecision::OneBit, "1-bit"),
        (QuantizationPrecision::OneFiveFiveBit, "1.58-bit"),
        (QuantizationPrecision::TwoBit, "2-bit"),
        (QuantizationPrecision::FourBit, "4-bit"),
        (QuantizationPrecision::EightBit, "8-bit"),
    ];
    
    println!("Running {} iterations for each precision...\n", 100);
    
    for (precision, name) in precisions {
        let mut scheme = QuantizationSchemeFactory::create_from_precision(precision, device.clone());
        
        // Warm up
        for _ in 0..10 {
            let _ = scheme.quantize_tensor(weights)?;
        }
        
        // Benchmark quantization
        let start = Instant::now();
        let mut total_error = 0.0;
        let mut quantized_result = None;
        
        for _ in 0..100 {
            let quantized = scheme.quantize_tensor(weights)?;
            let dequantized = scheme.dequantize_tensor(&quantized)?;
            let error = weights.sub(&dequantized)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
            total_error += error;
            quantized_result = Some(quantized);
        }
        
        let avg_time = start.elapsed() / 100;
        let avg_error = total_error / 100.0;
        
        if let Some(quantized) = quantized_result {
            println!("📊 {name} Quantization:");
            println!("  ⚡ Average time: {avg_time:?}");
            println!("  📊 Average MSE: {avg_error:.6}");
            println!("  💾 Compression ratio: {:.2}x", quantized.compression_ratio());
            println!("  📏 Memory footprint: {} bytes", quantized.memory_footprint());
        }
    }
    
    println!();
    Ok(())
}

fn demo_custom_configuration(weights: &Tensor, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Custom Configuration Demo");
    println!("==============================");
    
    // Create a highly optimized configuration
    let optimized_config = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            per_channel: false,
            clip_threshold: Some(3.0),
            qat_enabled: false,
            calibration_size: Some(1000),
        },
        scheme_params: SchemeParameters {
            one_five_eight_bit: OneFiveEightBitParams {
                ternary_method: TernaryMethod::OptimalThreshold,
                threshold_factor: 0.7,
                balanced_ternary: true,
                sparsity_target: Some(0.6), // Target 60% sparsity
            },
            ..Default::default()
        },
        adaptive_threshold: true,
        custom_thresholds: Some(ThresholdConfig {
            one_bit_threshold: Some(0.1),
            ternary_threshold: Some(0.5),
            multi_bit_thresholds: None,
        }),
        optimization: OptimizationConfig {
            enable_simd: true,
            use_lookup_tables: true,
            parallel_processing: true,
            memory_optimization_level: 2,
            cache_parameters: true,
        },
    };
    
    let mut scheme = ConfigurableQuantizationScheme::new(optimized_config, device.clone());
    
    println!("🎯 Testing optimized configuration...");
    
    let start = Instant::now();
    let quantized = scheme.quantize_tensor(weights)?;
    let quantize_time = start.elapsed();
    
    let dequantized = scheme.dequantize_tensor(&quantized)?;
    let error = weights.sub(&dequantized)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    
    // Analyze results
    let values = quantized.values.flatten_all()?.to_vec1::<f32>()?;
    let zero_count = values.iter().filter(|&&x| x.abs() < 1e-6).count();
    let sparsity = zero_count as f32 / values.len() as f32;
    
    println!("📊 Results:");
    println!("  ⚡ Quantization time: {quantize_time:?}");
    println!("  📊 Quantization error (MSE): {error:.6}");
    println!("  💾 Compression ratio: {:.2}x", quantized.compression_ratio());
    println!("  🕳️  Achieved sparsity: {:.1}%", sparsity * 100.0);
    println!("  📏 Memory footprint: {} bytes", quantized.memory_footprint());
    
    println!();
    Ok(())
}

fn demo_scheme_factory(weights: &Tensor, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 Quantization Scheme Factory Demo");
    println!("====================================");
    
    // Demo factory methods
    println!("🔧 Creating schemes using factory methods...\n");
    
    // 1-bit scheme
    let mut one_bit_scheme = QuantizationSchemeFactory::create_one_bit_scheme(device.clone());
    let one_bit_result = one_bit_scheme.quantize_tensor(weights)?;
    println!("✅ 1-bit scheme: {} bytes, {:.2}x compression", 
        one_bit_result.memory_footprint(), 
        one_bit_result.compression_ratio());
    
    // 1.58-bit scheme
    let mut ternary_scheme = QuantizationSchemeFactory::create_one_five_eight_bit_scheme(device.clone());
    let ternary_result = ternary_scheme.quantize_tensor(weights)?;
    println!("✅ 1.58-bit scheme: {} bytes, {:.2}x compression", 
        ternary_result.memory_footprint(), 
        ternary_result.compression_ratio());
    
    // Custom scheme
    let custom_config = QuantizationSchemeConfig {
        base: QuantizationConfig {
            precision: QuantizationPrecision::FourBit,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut custom_scheme = QuantizationSchemeFactory::create_custom_scheme(custom_config, device.clone());
    let custom_result = custom_scheme.quantize_tensor(weights)?;
    println!("✅ Custom 4-bit scheme: {} bytes, {:.2}x compression", 
        custom_result.memory_footprint(), 
        custom_result.compression_ratio());
    
    // From precision
    let mut from_precision_scheme = QuantizationSchemeFactory::create_from_precision(
        QuantizationPrecision::EightBit, 
        device.clone()
    );
    let from_precision_result = from_precision_scheme.quantize_tensor(weights)?;
    println!("✅ 8-bit from precision: {} bytes, {:.2}x compression", 
        from_precision_result.memory_footprint(), 
        from_precision_result.compression_ratio());
    
    println!();
    Ok(())
}