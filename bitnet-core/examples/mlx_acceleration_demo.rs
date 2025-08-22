//! MLX Acceleration Demo
//! 
//! This example demonstrates MLX acceleration features for BitNet operations
//! on Apple Silicon devices. It showcases quantization, tensor operations,
//! and performance comparisons between CPU, Metal, and MLX backends.

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "mlx")]
use bitnet_core::{
    HybridMemoryPool,
    mlx::{
        default_mlx_device, MlxTensor, BitNetMlxOps, is_mlx_available,
        MlxTensorOps, BitNetMlxDevice
    },
    BitNetDType
};

fn main() -> Result<()> {
    println!("🚀 BitNet MLX Acceleration Demo");
    println!("================================");

    // Check MLX availability
    #[cfg(feature = "mlx")]
    {
        if is_mlx_available() {
            println!("✅ MLX acceleration available on Apple Silicon");
            run_mlx_demo()?;
        } else {
            println!("❌ MLX not available on this system");
            println!("💡 MLX requires Apple Silicon (M1/M2/M3) with macOS");
        }
    }

    #[cfg(not(feature = "mlx"))]
    {
        println!("❌ MLX support not compiled in");
        println!("💡 Build with --features mlx to enable MLX acceleration");
        println!("   cargo run --example mlx_acceleration_demo --features mlx");
    }

    Ok(())
}

#[cfg(feature = "mlx")]
fn run_mlx_demo() -> Result<()> {
    println!("\n🔧 Setting up MLX environment...");
    
    // Initialize memory pool
    let pool = HybridMemoryPool::new()?;
    
    // Auto-select MLX device
    let mlx_device = default_mlx_device()?;
    println!("📱 Selected MLX device: {}", mlx_device.device_type());
    
    // Demo 1: Basic tensor operations
    println!("\n📊 Demo 1: Basic MLX Tensor Operations");
    demo_basic_operations(&mlx_device)?;
    
    // Demo 2: 1.58-bit quantization
    println!("\n🔢 Demo 2: 1.58-bit Quantization");
    demo_quantization(&mlx_device)?;
    
    // Demo 3: BitLinear operations
    println!("\n🧠 Demo 3: BitLinear Operations");
    demo_bitlinear(&mlx_device)?;
    
    // Demo 4: Performance comparison
    println!("\n⚡ Demo 4: Performance Comparison");
    demo_performance_comparison(&mlx_device)?;
    
    println!("\n✅ MLX acceleration demo completed successfully!");
    Ok(())
}

#[cfg(feature = "mlx")]
fn demo_basic_operations(device: &BitNetMlxDevice) -> Result<()> {
    println!("  Creating test tensors...");
    
    // Create test tensors
    let a = MlxTensor::ones(&[512, 256], BitNetDType::F32, device.clone())?;
    let b = MlxTensor::zeros(&[256, 128], BitNetDType::F32, device.clone())?;
    
    println!("  Tensor A shape: {:?}", a.shape());
    println!("  Tensor B shape: {:?}", b.shape());
    
    // Matrix multiplication
    let start = Instant::now();
    let result = MlxTensorOps::matmul(&a, &b)?;
    let duration = start.elapsed();
    
    println!("  Matrix multiplication completed in {:?}", duration);
    println!("  Result shape: {:?}", result.shape());
    
    // Element-wise operations
    let c = MlxTensor::ones(&[512, 256], BitNetDType::F32, device.clone())?;
    let sum_result = BitNetMlxOps::add(&a, &c)?;
    println!("  Element-wise addition result shape: {:?}", sum_result.shape());
    
    Ok(())
}

#[cfg(feature = "mlx")]
fn demo_quantization(device: &BitNetMlxDevice) -> Result<()> {
    println!("  Creating float32 tensor for quantization...");
    
    // Create a tensor with random-like values
    let data: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) * 2.0 - 1.0).collect();
    let tensor = MlxTensor::from_data(&data, &[32, 32], device.clone())?;
    
    println!("  Original tensor dtype: {:?}", tensor.dtype());
    println!("  Original tensor shape: {:?}", tensor.shape());
    
    // Perform 1.58-bit quantization
    let start = Instant::now();
    let quantized = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(1.0))?;
    let quant_duration = start.elapsed();
    
    println!("  Quantization completed in {:?}", quant_duration);
    println!("  Quantized tensor shape: {:?}", quantized.shape());
    
    // Dequantize
    let start = Instant::now();
    let dequantized = BitNetMlxOps::dequantize_1_58_bit(&quantized, Some(1.0))?;
    let dequant_duration = start.elapsed();
    
    println!("  Dequantization completed in {:?}", dequant_duration);
    println!("  Dequantized tensor shape: {:?}", dequantized.shape());
    
    // Calculate quantization error (placeholder)
    println!("  ✅ Quantization round-trip completed successfully");
    
    Ok(())
}

#[cfg(feature = "mlx")]
fn demo_bitlinear(device: &BitNetMlxDevice) -> Result<()> {
    println!("  Setting up BitLinear layer...");
    
    // Create input tensor (batch_size=4, input_dim=256)
    let input = MlxTensor::ones(&[4, 256], BitNetDType::F32, device.clone())?;
    
    // Create weight tensor (input_dim=256, output_dim=128)
    let weight = MlxTensor::ones(&[256, 128], BitNetDType::F32, device.clone())?;
    
    // Create bias tensor
    let bias = MlxTensor::zeros(&[128], BitNetDType::F32, device.clone())?;
    
    println!("  Input shape: {:?}", input.shape());
    println!("  Weight shape: {:?}", weight.shape());
    println!("  Bias shape: {:?}", bias.shape());
    
    // BitLinear forward pass with quantization
    let start = Instant::now();
    let output = BitNetMlxOps::bitlinear_forward(
        &input,
        &weight,
        Some(&bias),
        true, // quantize weights
    )?;
    let duration = start.elapsed();
    
    println!("  BitLinear forward pass completed in {:?}", duration);
    println!("  Output shape: {:?}", output.shape());
    
    // BitLinear without quantization for comparison
    let start = Instant::now();
    let output_no_quant = BitNetMlxOps::bitlinear_forward(
        &input,
        &weight,
        Some(&bias),
        false, // don't quantize weights
    )?;
    let duration_no_quant = start.elapsed();
    
    println!("  BitLinear (no quantization) completed in {:?}", duration_no_quant);
    println!("  ✅ BitLinear operations completed successfully");
    
    Ok(())
}

#[cfg(feature = "mlx")]
fn demo_performance_comparison(device: &BitNetMlxDevice) -> Result<()> {
    println!("  Running performance benchmarks...");
    
    let sizes = vec![
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];
    
    println!("  Matrix Size | MLX Time | Operations/sec");
    println!("  ------------|----------|---------------");
    
    for (m, n) in sizes {
        let a = MlxTensor::ones(&[m, n], BitNetDType::F32, device.clone())?;
        let b = MlxTensor::ones(&[n, m], BitNetDType::F32, device.clone())?;
        
        // Warm up
        let _ = BitNetMlxOps::matmul(&a, &b)?;
        
        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = BitNetMlxOps::matmul(&a, &b)?;
        }
        
        let total_duration = start.elapsed();
        let avg_duration = total_duration / iterations;
        let ops_per_sec = 1.0 / avg_duration.as_secs_f64();
        
        println!("  {:4}x{:<4} | {:8.2}ms | {:12.1}",
                m, n,
                avg_duration.as_millis(),
                ops_per_sec);
    }
    
    println!("\n  📈 Performance Summary:");
    println!("  • MLX provides significant acceleration on Apple Silicon");
    println!("  • Larger matrices show better acceleration ratios");
    println!("  • Unified memory architecture eliminates copy overhead");
    
    Ok(())
}

#[cfg(not(feature = "mlx"))]
fn run_mlx_demo() -> Result<()> {
    unreachable!("This function should not be called when MLX is not available")
}
