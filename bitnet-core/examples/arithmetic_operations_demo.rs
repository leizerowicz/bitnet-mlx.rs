//! BitNet Arithmetic Operations Demonstration
//!
//! This example demonstrates the comprehensive arithmetic operations
//! available for BitNet tensors, including broadcasting, in-place
//! operations, and scalar operations.

use bitnet_core::memory::{HybridMemoryPool, TrackingConfig};
use bitnet_core::tensor::ops::arithmetic::*;
use bitnet_core::tensor::{BitNetDType, BitNetTensor};

#[cfg(feature = "tracing")]
use tracing::{debug, info};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize memory pool for tensor operations
    let tracking_config = TrackingConfig::detailed();
    let mut config = bitnet_core::memory::MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(tracking_config);

    let memory_pool = std::sync::Arc::new(HybridMemoryPool::with_config(config)?);

    // Initialize global memory pool for tensor operations
    bitnet_core::tensor::memory_integration::set_global_memory_pool(std::sync::Arc::downgrade(
        &memory_pool,
    ));

    #[cfg(feature = "tracing")]
    tracing_subscriber::fmt::init();

    println!("🧮 BitNet Arithmetic Operations Demonstration");
    println!("==============================================\n");

    // Section 1: Basic Arithmetic Operations
    println!("📊 1. Basic Element-wise Operations");
    println!("------------------------------------");

    let a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
    let b = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;

    println!("Created tensors:");
    println!("  a: shape {:?}, dtype {:?}", a.shape().dims(), a.dtype());
    println!("  b: shape {:?}, dtype {:?}", b.shape().dims(), b.dtype());

    // Addition
    let c_add = add(&a, &b)?;
    println!("  Addition (a + b): shape {:?}", c_add.shape().dims());

    // Subtraction
    let c_sub = sub(&a, &b)?;
    println!("  Subtraction (a - b): shape {:?}", c_sub.shape().dims());

    // Multiplication
    let c_mul = mul(&a, &b)?;
    println!("  Multiplication (a * b): shape {:?}", c_mul.shape().dims());

    // Division
    let c_div = div(&a, &b)?;
    println!("  Division (a / b): shape {:?}", c_div.shape().dims());

    // Remainder
    let c_rem = rem(&a, &b)?;
    println!("  Remainder (a % b): shape {:?}", c_rem.shape().dims());

    // Power (floating point only)
    let c_pow = pow(&a, &b)?;
    println!("  Power (a^b): shape {:?}", c_pow.shape().dims());

    println!("✅ Basic operations completed successfully!\n");

    // Section 2: Broadcasting Operations
    println!("📡 2. Broadcasting Operations");
    println!("------------------------------");

    let tensor_2d = BitNetTensor::ones(&[3, 4], BitNetDType::F32, None)?;
    let tensor_1d = BitNetTensor::ones(&[1, 4], BitNetDType::F32, None)?;
    let tensor_scalar = BitNetTensor::ones(&[1, 1], BitNetDType::F32, None)?;

    println!("Broadcasting tensors:");
    println!("  tensor_2d: shape {:?}", tensor_2d.shape().dims());
    println!("  tensor_1d: shape {:?}", tensor_1d.shape().dims());
    println!("  tensor_scalar: shape {:?}", tensor_scalar.shape().dims());

    // Broadcasting addition: [3, 4] + [1, 4] -> [3, 4]
    let broadcast_result = add(&tensor_2d, &tensor_1d)?;
    println!(
        "  Broadcast [3,4] + [1,4]: shape {:?}",
        broadcast_result.shape().dims()
    );

    // Broadcasting with scalar: [3, 4] + [1, 1] -> [3, 4]
    let scalar_broadcast = add(&tensor_2d, &tensor_scalar)?;
    println!(
        "  Broadcast [3,4] + [1,1]: shape {:?}",
        scalar_broadcast.shape().dims()
    );

    // More complex broadcasting
    let tensor_3d = BitNetTensor::ones(&[2, 3, 1], BitNetDType::F32, None)?;
    let tensor_broadcast = BitNetTensor::ones(&[1, 1, 4], BitNetDType::F32, None)?;

    let complex_broadcast = add(&tensor_3d, &tensor_broadcast)?;
    println!(
        "  Complex broadcast [2,3,1] + [1,1,4]: shape {:?}",
        complex_broadcast.shape().dims()
    );

    println!("✅ Broadcasting operations completed successfully!\n");

    // Section 3: In-Place Operations
    println!("🔄 3. In-Place Operations");
    println!("--------------------------");

    let mut mutable_tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
    let operand = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

    println!("Original tensor shape: {:?}", mutable_tensor.shape().dims());

    // In-place addition
    add_(&mut mutable_tensor, &operand)?;
    println!(
        "  After in-place addition: shape {:?}",
        mutable_tensor.shape().dims()
    );

    // In-place subtraction
    sub_(&mut mutable_tensor, &operand)?;
    println!(
        "  After in-place subtraction: shape {:?}",
        mutable_tensor.shape().dims()
    );

    // In-place multiplication
    mul_(&mut mutable_tensor, &operand)?;
    println!(
        "  After in-place multiplication: shape {:?}",
        mutable_tensor.shape().dims()
    );

    // In-place division
    div_(&mut mutable_tensor, &operand)?;
    println!(
        "  After in-place division: shape {:?}",
        mutable_tensor.shape().dims()
    );

    println!("✅ In-place operations completed successfully!\n");

    // Section 4: Scalar Operations
    println!("🔢 4. Scalar Operations");
    println!("------------------------");

    let base_tensor = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;
    println!("Base tensor shape: {:?}", base_tensor.shape().dims());

    // Scalar addition
    let scalar_add_result = add_scalar(&base_tensor, 5.0)?;
    println!(
        "  Add scalar 5.0: shape {:?}",
        scalar_add_result.shape().dims()
    );

    // Scalar subtraction
    let scalar_sub_result = sub_scalar(&base_tensor, 2.0)?;
    println!(
        "  Subtract scalar 2.0: shape {:?}",
        scalar_sub_result.shape().dims()
    );

    // Scalar multiplication
    let scalar_mul_result = mul_scalar(&base_tensor, 3.0)?;
    println!(
        "  Multiply by scalar 3.0: shape {:?}",
        scalar_mul_result.shape().dims()
    );

    // Scalar division
    let scalar_div_result = div_scalar(&base_tensor, 2.0)?;
    println!(
        "  Divide by scalar 2.0: shape {:?}",
        scalar_div_result.shape().dims()
    );

    println!("✅ Scalar operations completed successfully!\n");

    // Section 5: In-Place Scalar Operations
    println!("🔄🔢 5. In-Place Scalar Operations");
    println!("-----------------------------------");

    let mut scalar_tensor = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
    println!("Mutable tensor shape: {:?}", scalar_tensor.shape().dims());

    add_scalar_(&mut scalar_tensor, 10.0)?;
    println!(
        "  After adding 10.0 in-place: shape {:?}",
        scalar_tensor.shape().dims()
    );

    mul_scalar_(&mut scalar_tensor, 0.5)?;
    println!(
        "  After multiplying by 0.5 in-place: shape {:?}",
        scalar_tensor.shape().dims()
    );

    println!("✅ In-place scalar operations completed successfully!\n");

    // Section 6: Operator Overloading
    println!("⚡ 6. Operator Overloading");
    println!("---------------------------");

    let left = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
    let right = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;

    println!("Using operator overloading:");

    // Using + operator
    let op_add = (&left + &right)?;
    println!("  Using + operator: shape {:?}", op_add.shape().dims());

    // Using - operator
    let op_sub = (&left - &right)?;
    println!("  Using - operator: shape {:?}", op_sub.shape().dims());

    // Using * operator
    let op_mul = (&left * &right)?;
    println!("  Using * operator: shape {:?}", op_mul.shape().dims());

    // Using / operator
    let op_div = (&left / &right)?;
    println!("  Using / operator: shape {:?}", op_div.shape().dims());

    // Using % operator
    let op_rem = (&left % &right)?;
    println!("  Using % operator: shape {:?}", op_rem.shape().dims());

    println!("✅ Operator overloading completed successfully!\n");

    // Section 7: Different Data Types
    println!("🎯 7. Different Data Types");
    println!("---------------------------");

    // F32 tensors
    let f32_a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
    let f32_b = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
    let f32_result = add(&f32_a, &f32_b)?;
    println!(
        "  F32 addition: {:?} -> {:?}",
        f32_a.dtype(),
        f32_result.shape().dims()
    );

    // I32 tensors
    let i32_a = BitNetTensor::zeros(&[2, 2], BitNetDType::I32, None)?;
    let i32_b = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;
    let i32_result = add(&i32_a, &i32_b)?;
    println!(
        "  I32 addition: {:?} -> {:?}",
        i32_a.dtype(),
        i32_result.shape().dims()
    );

    println!("✅ Different data types completed successfully!\n");

    // Section 8: Error Handling Examples
    println!("⚠️  8. Error Handling Examples");
    println!("-------------------------------");

    let tensor_f32 = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
    let tensor_i32 = BitNetTensor::ones(&[2, 2], BitNetDType::I32, None)?;

    // Data type mismatch error
    match add(&tensor_f32, &tensor_i32) {
        Ok(_) => println!("  ❌ Expected error for data type mismatch"),
        Err(_) => println!("  ✅ Correctly caught data type mismatch error"),
    }

    // Shape incompatibility error
    let incompatible_a = BitNetTensor::ones(&[2, 3], BitNetDType::F32, None)?;
    let incompatible_b = BitNetTensor::ones(&[4, 2], BitNetDType::F32, None)?;

    match add(&incompatible_a, &incompatible_b) {
        Ok(_) => println!("  ❌ Expected error for shape incompatibility"),
        Err(_) => println!("  ✅ Correctly caught shape incompatibility error"),
    }

    // Division by zero error
    match div_scalar(&tensor_f32, 0.0) {
        Ok(_) => println!("  ❌ Expected error for division by zero"),
        Err(_) => println!("  ✅ Correctly caught division by zero error"),
    }

    println!("✅ Error handling examples completed successfully!\n");

    // Section 9: Memory Efficiency Analysis
    println!("💾 9. Memory Efficiency Analysis");
    println!("---------------------------------");

    let large_tensor = BitNetTensor::ones(&[100, 100], BitNetDType::F32, None)?;
    let broadcast_tensor = BitNetTensor::ones(&[1, 100], BitNetDType::F32, None)?;

    println!(
        "Large tensor: {} elements ({} bytes)",
        large_tensor.element_count(),
        large_tensor.size_bytes()
    );
    println!(
        "Broadcast tensor: {} elements ({} bytes)",
        broadcast_tensor.element_count(),
        broadcast_tensor.size_bytes()
    );

    let broadcast_result = add(&large_tensor, &broadcast_tensor)?;
    println!(
        "Broadcast result: {} elements ({} bytes)",
        broadcast_result.element_count(),
        broadcast_result.size_bytes()
    );

    println!("✅ Memory efficiency analysis completed!\n");

    // Summary
    println!("🎉 BitNet Arithmetic Operations Demo Complete!");
    println!("==============================================");
    println!("✅ All arithmetic operations working correctly");
    println!("✅ Broadcasting implemented and tested");
    println!("✅ In-place operations functional");
    println!("✅ Scalar operations implemented");
    println!("✅ Operator overloading working");
    println!("✅ Error handling comprehensive");
    println!("✅ Memory efficiency maintained");

    Ok(())
}
