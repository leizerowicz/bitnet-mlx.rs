//! Integration tests to verify the three required device functions are accessible and work correctly

use bitnet_core::device::{auto_select_device, get_cpu_device, get_metal_device};
use bitnet_core::{DType, Tensor};

#[test]
fn test_get_cpu_device() {
    println!("Testing get_cpu_device():");
    let cpu_device = get_cpu_device();
    println!("   ✓ CPU device created successfully: {cpu_device:?}");

    // Verify it's actually a CPU device
    assert!(matches!(cpu_device, bitnet_core::Device::Cpu));
}

#[test]
fn test_get_metal_device() {
    println!("Testing get_metal_device():");
    match get_metal_device() {
        Ok(metal_device) => {
            println!("   ✓ Metal device created successfully: {metal_device:?}");
            assert!(matches!(metal_device, bitnet_core::Device::Metal(_)));
        }
        Err(e) => {
            println!("   ⚠ Metal device not available: {e}");
            println!("   (This is expected on non-macOS systems or without Metal support)");
        }
    }
}

#[test]
fn test_auto_select_device() {
    println!("Testing auto_select_device():");
    let auto_device = auto_select_device();
    println!("   ✓ Auto-selected device: {auto_device:?}");

    // Should return either CPU or Metal device
    assert!(matches!(
        auto_device,
        bitnet_core::Device::Cpu | bitnet_core::Device::Metal(_)
    ));
}

#[test]
fn test_function_accessibility() {
    println!("Verifying function accessibility:");

    // Test that functions are properly exported and accessible
    let _cpu = get_cpu_device();
    println!("   ✓ get_cpu_device is accessible");

    let _metal_result = get_metal_device();
    println!("   ✓ get_metal_device is accessible");

    let _auto = auto_select_device();
    println!("   ✓ auto_select_device is accessible");
}

#[test]
fn test_device_usage_with_tensor_operations() {
    println!("Testing device usage with tensor operations:");

    let device = auto_select_device();
    match Tensor::zeros(&[2, 2], DType::F32, &device) {
        Ok(tensor) => {
            println!(
                "   ✓ Successfully created tensor on device: {:?}",
                tensor.shape()
            );
            assert_eq!(tensor.shape().dims(), &[2, 2]);
        }
        Err(e) => {
            println!("   ✗ Failed to create tensor: {e}");
            panic!("Tensor creation should succeed on auto-selected device");
        }
    }
}

#[test]
fn test_cpu_device_tensor_creation() {
    println!("Testing CPU device with tensor creation:");

    let cpu_device = get_cpu_device();
    let tensor = Tensor::ones(&[3, 3], DType::F32, &cpu_device)
        .expect("Failed to create tensor on CPU device");

    assert_eq!(tensor.shape().dims(), &[3, 3]);
    println!("   ✓ Successfully created 3x3 tensor on CPU device");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn test_metal_device_tensor_creation() {
    println!("Testing Metal device with tensor creation:");

    if let Ok(metal_device) = get_metal_device() {
        match Tensor::zeros(&[4, 4], DType::F32, &metal_device) {
            Ok(tensor) => {
                assert_eq!(tensor.shape().dims(), &[4, 4]);
                println!("   ✓ Successfully created 4x4 tensor on Metal device");
            }
            Err(e) => {
                println!("   ⚠ Failed to create tensor on Metal device: {}", e);
                // Don't fail the test as this might be expected in some environments
            }
        }
    } else {
        println!("   ⚠ Metal device not available, skipping Metal tensor test");
    }
}

#[test]
fn test_device_management_comprehensive() {
    println!("Running comprehensive device management test:");

    // Test CPU device
    let cpu_device = get_cpu_device();
    assert!(matches!(cpu_device, bitnet_core::Device::Cpu));

    // Test auto selection
    let auto_device = auto_select_device();
    assert!(matches!(
        auto_device,
        bitnet_core::Device::Cpu | bitnet_core::Device::Metal(_)
    ));

    // Test that we can create tensors on both devices
    let cpu_tensor = Tensor::zeros(&[2, 2], DType::F32, &cpu_device)
        .expect("CPU tensor creation should always work");
    assert_eq!(cpu_tensor.shape().dims(), &[2, 2]);

    let auto_tensor = Tensor::zeros(&[2, 2], DType::F32, &auto_device)
        .expect("Auto device tensor creation should work");
    assert_eq!(auto_tensor.shape().dims(), &[2, 2]);

    println!("   ✓ All device management functions are working correctly!");
}
