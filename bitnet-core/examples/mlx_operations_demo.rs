//! Demo of MLX operation wrappers
//!
//! This example demonstrates the usage of the new MLX operation wrappers:
//! - mlx_matmul: Matrix multiplication
//! - mlx_quantize: Quantization with scale factor
//! - mlx_dequantize: Dequantization with scale factor

#[cfg(feature = "mlx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bitnet_core::mlx::{mlx_dequantize, mlx_matmul, mlx_quantize};
    use mlx_rs::Array;

    println!("MLX Operations Demo");
    println!("==================");

    // Create test matrices for matrix multiplication
    println!("\n1. Matrix Multiplication Demo:");
    let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    println!("Matrix A: {:?}", a.as_slice::<f32>());
    println!("Matrix B: {:?}", b.as_slice::<f32>());

    let result = mlx_matmul(&a, &b)?;
    println!("A @ B = {:?}", result.as_slice::<f32>());

    // Demonstrate quantization and dequantization
    println!("\n2. Quantization Demo:");
    let original = Array::from_slice(&[1.5, 2.7, 3.2, 4.8], &[2, 2]);
    let scale = 0.5;

    println!("Original: {:?}", original.as_slice::<f32>());
    println!("Scale: {}", scale);

    let quantized = mlx_quantize(&original, scale)?;
    println!("Quantized: {:?}", quantized.as_slice::<f32>());

    let dequantized = mlx_dequantize(&quantized, scale)?;
    println!("Dequantized: {:?}", dequantized.as_slice::<f32>());

    // Round-trip test
    println!("\n3. Round-trip Test:");
    let test_data = Array::from_slice(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let scale = 1.0;

    println!("Original: {:?}", test_data.as_slice::<f32>());

    let quantized = mlx_quantize(&test_data, scale)?;
    let recovered = mlx_dequantize(&quantized, scale)?;

    println!("After round-trip: {:?}", recovered.as_slice::<f32>());

    // Check if values are approximately equal
    let original_data = test_data.as_slice::<f32>();
    let recovered_data = recovered.as_slice::<f32>();
    let mut all_close = true;

    for (orig, rec) in original_data.iter().zip(recovered_data.iter()) {
        if (orig - rec).abs() > 1e-6 {
            all_close = false;
            break;
        }
    }

    println!("Round-trip successful: {}", all_close);

    println!("\nMLX operations demo completed successfully!");
    Ok(())
}

#[cfg(not(feature = "mlx"))]
fn main() {
    println!("MLX feature not enabled. Please run with --features mlx");
}
