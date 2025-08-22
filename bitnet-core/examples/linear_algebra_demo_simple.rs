//! Linear Algebra Operations Demo
//! 
//! This demo showcases the linear algebra capabilities of BitNet tensors,
//! including matrix multiplication, dot products, transpositions, and
//! advanced decompositions.

use bitnet_core::{
    tensor::{BitNetTensor, BitNetDType, set_global_memory_pool},
    tensor::ops::linear_algebra::*,
    device::get_cpu_device,
    memory::HybridMemoryPool,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet Linear Algebra Operations Demo");
    println!("========================================\n");
    
    // Initialize devices and memory pool
    let device = get_cpu_device();
    let memory_pool = Arc::new(HybridMemoryPool::new()?);
    
    // Initialize global memory pool
    set_global_memory_pool(Arc::downgrade(&memory_pool));
    
    println!("Using device: {device:?}\n");
    
    demo_matrix_multiplication(&device)?;
    demo_dot_products(&device)?;
    demo_matrix_transformations(&device)?;
    
    println!("\n✅ All linear algebra operations completed successfully!");
    
    Ok(())
}

fn demo_matrix_multiplication(device: &candle_core::Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Matrix Multiplication Operations");
    println!("{}", "-".repeat(40));
    
    // Basic matrix multiplication
    println!("1. Basic Matrix Multiplication:");
    let a = BitNetTensor::ones(&[3, 4], BitNetDType::F32, Some(device.clone()))?;
    let b = BitNetTensor::ones(&[4, 2], BitNetDType::F32, Some(device.clone()))?;
    
    println!("   Matrix A: {:?} × Matrix B: {:?}", a.shape().dims(), b.shape().dims());
    
    let result = matmul(&a, &b)?;
    println!("   Result shape: {:?}", result.shape().dims());
    println!("   ✓ Matrix multiplication successful\n");
    
    // Square matrix multiplication  
    println!("2. Square Matrix Multiplication:");
    let c = BitNetTensor::ones(&[3, 3], BitNetDType::F32, Some(device.clone()))?;
    let d = BitNetTensor::ones(&[3, 3], BitNetDType::F32, Some(device.clone()))?;
    
    let result2 = matmul(&c, &d)?;
    println!("   {:?} × {:?} = {:?}", c.shape().dims(), d.shape().dims(), result2.shape().dims());
    println!("   ✓ Square matrix multiplication successful\n");
    
    Ok(())
}

fn demo_dot_products(device: &candle_core::Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Dot Product Operations");
    println!("{}", "-".repeat(30));
    
    // Vector dot product
    println!("1. Vector Dot Product:");
    let vec_a = BitNetTensor::ones(&[5], BitNetDType::F32, Some(device.clone()))?;
    let vec_b = BitNetTensor::ones(&[5], BitNetDType::F32, Some(device.clone()))?;
    
    let dot_result = dot(&vec_a, &vec_b)?;
    println!("   Vector A: {:?} · Vector B: {:?}", vec_a.shape().dims(), vec_b.shape().dims());
    println!("   Dot product result shape: {:?}", dot_result.shape().dims());
    println!("   ✓ Dot product successful\n");
    
    Ok(())
}

fn demo_matrix_transformations(device: &candle_core::Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Matrix Transformation Operations");
    println!("{}", "-".repeat(40));
    
    // Matrix transpose
    println!("1. Matrix Transpose:");
    let matrix = BitNetTensor::ones(&[4, 3], BitNetDType::F32, Some(device.clone()))?;
    let transposed = transpose(&matrix)?;
    
    println!("   Original: {:?} → Transposed: {:?}", matrix.shape().dims(), transposed.shape().dims());
    println!("   ✓ Transpose successful\n");
    
    // Identity matrix creation
    println!("2. Identity Matrix:");
    match eye(3, BitNetDType::F32, Some(device.clone())) {
        Ok(identity) => {
            println!("   Created 3×3 identity matrix: {:?}", identity.shape().dims());
            println!("   ✓ Identity matrix creation successful\n");
        }
        Err(e) => {
            println!("   ⚠️  Identity matrix creation failed: {e}\n");
        }
    }
    
    Ok(())
}
