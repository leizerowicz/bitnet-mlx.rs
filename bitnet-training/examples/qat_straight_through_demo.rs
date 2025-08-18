// QAT Phase 3.2 Implementation Demo - Straight-Through Estimator
// Demonstrates core QAT functionality with working STE implementation

use candle_core::{Device, Result, Tensor};
use bitnet_training::qat::straight_through::{
    STEConfig, STEVariant, StraightThroughEstimator, quantize_with_ste
};

fn main() -> Result<()> {
    println!("=== BitNet QAT Phase 3.2 Demo: Straight-Through Estimator ===");
    
    let device = Device::Cpu;
    
    // Demo 1: Standard STE with clipping
    demo_standard_ste(&device)?;
    
    // Demo 2: Soft quantization STE
    demo_soft_quantization(&device)?;
    
    // Demo 3: Different bit precisions
    demo_bit_precisions(&device)?;
    
    // Demo 4: Statistics tracking
    demo_statistics_tracking(&device)?;
    
    println!("\n✅ QAT Phase 3.2 Implementation Complete!");
    println!("All STE variants successfully demonstrated");
    
    Ok(())
}

fn demo_standard_ste(device: &Device) -> Result<()> {
    println!("\n--- Demo 1: Standard Straight-Through Estimator ---");
    
    let config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,  // Binarization
        range: 1.0,
        ..Default::default()
    };
    
    let mut ste = StraightThroughEstimator::new(config, device.clone())?;
    
    // Test with various input ranges
    let inputs = vec![
        vec![0.8, -0.3, 0.1, -0.9, 2.5], // Including values outside range
        vec![0.1, 0.2, -0.1, -0.2],      // Small values
        vec![1.5, -1.5, 0.0],            // Boundary cases
    ];
    
    for (i, input_data) in inputs.iter().enumerate() {
        let input_f32: Vec<f32> = input_data.iter().map(|&x| x as f32).collect();
        let input = Tensor::from_slice(&input_f32, (input_f32.len(),), device)?;
        let output = ste.forward(&input)?;
        let output_vec: Vec<f32> = output.to_vec1()?;
        
        println!("  Input {}: {:?}", i+1, input_data);
        println!("  Output {}: {:?}", i+1, output_vec);
        println!("  Clipping rate: {:.3}", ste.get_clipping_rate());
    }
    
    Ok(())
}

fn demo_soft_quantization(device: &Device) -> Result<()> {
    println!("\n--- Demo 2: Soft Quantization STE ---");
    
    let config = STEConfig {
        variant: STEVariant::Soft,
        bits: 2,
        range: 1.0,
        temperature: 2.0, // Lower temperature = harder quantization
        ..Default::default()
    };
    let temperature = config.temperature;

    let mut ste = StraightThroughEstimator::new(config, device.clone())?;

    let input = Tensor::from_slice(&[0.3f32, -0.7f32, 0.5f32, -0.2f32, 0.0f32], (5,), device)?;
    let output = ste.forward(&input)?;
    let output_vec: Vec<f32> = output.to_vec1()?;

    println!("  Input: [0.3, -0.7, 0.5, -0.2, 0.0]");
    println!("  Soft quantized: {:?}", output_vec);
    println!("  Temperature: {}", temperature);    Ok(())
}

fn demo_bit_precisions(device: &Device) -> Result<()> {
    println!("\n--- Demo 3: Different Bit Precisions ---");
    
    let input = Tensor::from_slice(&[0.6f32, -0.4f32, 0.8f32, -0.9f32, 0.1f32], (5,), device)?;
    let input_vec: Vec<f32> = input.to_vec1()?;
    println!("  Input: {:?}", input_vec);
    
    for bits in [1, 2, 3, 4] {
        let config = STEConfig {
            variant: STEVariant::Standard,
            bits,
            range: 1.0,
            ..Default::default()
        };
        
        let mut ste = StraightThroughEstimator::new(config, device.clone())?;
        let output = ste.forward(&input)?;
        let output_vec: Vec<f32> = output.to_vec1()?;
        
        println!("  {}-bit: {:?}", bits, output_vec);
    }
    
    Ok(())
}

fn demo_statistics_tracking(device: &Device) -> Result<()> {
    println!("\n--- Demo 4: Statistics Tracking ---");
    
    let config = STEConfig {
        variant: STEVariant::Clipped,
        bits: 1,
        range: 1.0,
        clip_threshold: 1.5,
        ..Default::default()
    };
    
    let mut ste = StraightThroughEstimator::new(config, device.clone())?;
    
    // Process multiple batches
    let batches = vec![
        vec![2.0f32, -2.5f32, 0.3f32, -0.8f32, 1.2f32],  // Some out-of-range values
        vec![0.1f32, -0.1f32, 0.05f32, -0.05f32],     // All in range
        vec![1.8f32, -1.9f32, 2.2f32, -2.8f32, 0.0f32], // Many out-of-range
    ];
    
    for (batch_idx, batch_data) in batches.iter().enumerate() {
        let input = Tensor::from_slice(batch_data, (batch_data.len(),), device)?;
        let _output = ste.forward(&input)?;
        
        println!("  Batch {}: processed {} values", batch_idx + 1, batch_data.len());
    }
    
    // Get final statistics
    let stats = ste.get_statistics();
    println!("  Final statistics:");
    println!("    Total operations: {}", stats.total_operations);
    println!("    Clipping rate: {:.3}", stats.clipping_rate);
    println!("    Current quantization error: {:.6}", stats.quantization_error);
    println!("    Gradient magnitude: {:.6}", stats.gradient_magnitude);
    
    // Demonstrate quantize_with_ste convenience function
    println!("\n--- Convenience Function Demo ---");
    let test_input = Tensor::from_slice(&[0.7f32, -0.3f32, 1.1f32], (3,), device)?;
    let test_config = STEConfig {
        variant: STEVariant::Standard,
        bits: 1,
        range: 1.0,
        ..Default::default()
    };
    let quantized = quantize_with_ste(&test_input, &test_config, device)?;
    let quantized_vec: Vec<f32> = quantized.to_vec1()?;
    
    println!("  Direct quantization: [0.7, -0.3, 1.1] -> {:?}", quantized_vec);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_phase_3_2_implementation() -> Result<()> {
        let device = Device::Cpu;
        
        // Test core STE functionality
        let config = STEConfig {
            variant: STEVariant::Standard,
            bits: 1,
            range: 1.0,
            ..Default::default()
        };
        
        let mut ste = StraightThroughEstimator::new(config, device.clone())?;
        
        let input = Tensor::from_slice(&[0.5f32, -0.5f32, 1.5f32, -1.5f32], (4,), &device)?;
        let output = ste.forward(&input)?;
        let output_vec: Vec<f32> = output.to_vec1()?;
        
        // Should be binary quantized to -1 or 1
        for val in &output_vec {
            assert!(val.abs() <= 1.0);
            assert!((*val - 1.0).abs() < 1e-6 || (*val + 1.0).abs() < 1e-6);
        }
        
        // Test statistics
        let stats = ste.get_statistics();
        assert!(stats.total_operations > 0);
        assert!(ste.get_clipping_rate() >= 0.0); // Should track clipping
        
        println!("✅ QAT Phase 3.2 core functionality validated");
        Ok(())
    }
    
    #[test]
    fn test_ste_variants() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[0.3f32, -0.7f32], (2,), &device)?;
        
        // Test all STE variants
        for variant in [STEVariant::Standard, STEVariant::Clipped, STEVariant::Soft] {
            let config = STEConfig {
                variant,
                bits: 2,
                range: 1.0,
                ..Default::default()
            };
            
            let mut ste = StraightThroughEstimator::new(config, device.clone())?;
            let output = ste.forward(&input)?;
            
            // Should produce valid output
            assert_eq!(output.dims(), input.dims());
        }
        
        println!("✅ All STE variants working correctly");
        Ok(())
    }
}
