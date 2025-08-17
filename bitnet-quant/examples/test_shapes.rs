use anyhow::Result;
use candle_core::{Tensor, Device};

fn main() -> Result<()> {
    println!("Testing tensor shapes for BitLinear");
    
    // Create weights with shape [3072, 768] (out_features, in_features)
    let weights = Tensor::randn(0.0, 1.0, &[3072, 768], &Device::Cpu)?;
    println!("Original weights shape: {:?}", weights.shape());
    
    // Create input with shape [4, 128, 768] (batch, seq_len, in_features)
    let input = Tensor::randn(0.0, 1.0, &[4, 128, 768], &Device::Cpu)?;
    println!("Input shape: {:?}", input.shape());
    
    // For neural networks, we typically want:
    // input: [batch, seq_len, in_features] @ weights.T: [in_features, out_features]
    // Result: [batch, seq_len, out_features]
    
    // Weights should be [out_features, in_features] = [3072, 768]
    // Transpose to [in_features, out_features] = [768, 3072]
    let weights_transposed = weights.t()?;
    println!("Transposed weights shape: {:?}", weights_transposed.shape());
    
    // Now let's try different approaches:
    
    // 1. Reshape input to 2D, multiply, reshape back
    let input_2d = input.reshape(&[4 * 128, 768])?;
    println!("Reshaped input to 2D: {:?}", input_2d.shape());
    
    let result_2d = input_2d.matmul(&weights_transposed)?;
    println!("2D result shape: {:?}", result_2d.shape());
    
    let result_3d = result_2d.reshape(&[4, 128, 3072])?;
    println!("Reshaped back to 3D: {:?}", result_3d.shape());
    
    // 2. Try using broadcast_left for batch matrix multiplication
    println!("\n=== Testing batch matrix multiplication ===");
    // Input: [4, 128, 768], Weights: [768, 3072]
    // We want: [4, 128, 3072]
    
    // This might work if candle supports broadcasting
    match input.matmul(&weights_transposed) {
        Ok(result) => println!("Direct 3D @ 2D worked! Result shape: {:?}", result.shape()),
        Err(e) => println!("Direct 3D @ 2D failed: {}", e),
    }
    
    Ok(())
}
