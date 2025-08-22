//! Demonstration of the RoundClip function
//! 
//! This example shows how to use the RoundClip function for quantization operations.

use bitnet_quant::quantization::utils::QuantizationUtils;
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RoundClip Function Demo");
    println!("=======================");
    
    // Demonstrate basic RoundClip functionality
    println!("\n1. Basic RoundClip examples:");
    let test_values = vec![1.7, -2.3, 0.4, 0.6, -0.6, 0.5, -0.5, 1.5, -1.5];
    
    for value in test_values {
        let result = QuantizationUtils::round_clip(value, -1.0, 1.0);
        println!("  RoundClip({value:.1}, -1, 1) = {result:.1}");
    }
    
    // Demonstrate tensor-based RoundClip
    println!("\n2. Tensor RoundClip example:");
    let device = Device::Cpu;
    let data = vec![1.7f32, -2.3, 0.4, 0.6, -0.6, 2.1, -1.8];
    let tensor = Tensor::from_slice(&data, (7,), &device)?;
    
    println!("  Original tensor: {:?}", tensor.to_vec1::<f32>()?);
    
    let result = QuantizationUtils::round_clip_tensor(&tensor, -1.0, 1.0, &device)?;
    println!("  After RoundClip: {:?}", result.to_vec1::<f32>()?);
    
    // Demonstrate different clipping ranges
    println!("\n3. Different clipping ranges:");
    let value = 2.7;
    
    let ranges = vec![
        (-1.0, 1.0),
        (-2.0, 2.0),
        (0.0, 3.0),
        (-3.0, 0.0),
    ];
    
    for (min_val, max_val) in ranges {
        let result = QuantizationUtils::round_clip(value, min_val, max_val);
        println!("  RoundClip({value:.1}, {min_val:.1}, {max_val:.1}) = {result:.1}");
    }
    
    // Demonstrate typical quantization use case
    println!("\n4. Typical quantization use case (BitNet 1.58-bit):");
    let weights = vec![0.8f32, -1.2, 0.3, -0.7, 1.5, -2.1, 0.1];
    println!("  Original weights: {weights:?}");
    
    let quantized: Vec<f32> = weights.iter()
        .map(|&w| QuantizationUtils::round_clip(w, -1.0, 1.0))
        .collect();
    
    println!("  Quantized to {{-1, 0, 1}}: {quantized:?}");
    
    Ok(())
}