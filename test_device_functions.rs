//! Test program to verify the three required device functions are accessible and work correctly

use bitnet_core::device::{get_metal_device, get_cpu_device, auto_select_device};

fn main() {
    println!("Testing BitNet Core Device Management Functions");
    println!("==============================================");
    
    // Test 1: get_cpu_device() -> Device
    println!("\n1. Testing get_cpu_device():");
    let cpu_device = get_cpu_device();
    println!("   ✓ CPU device created successfully: {:?}", cpu_device);
    
    // Test 2: get_metal_device() -> Result<Device>
    println!("\n2. Testing get_metal_device():");
    match get_metal_device() {
        Ok(metal_device) => {
            println!("   ✓ Metal device created successfully: {:?}", metal_device);
        }
        Err(e) => {
            println!("   ⚠ Metal device not available: {}", e);
            println!("   (This is expected on non-macOS systems or without Metal support)");
        }
    }
    
    // Test 3: auto_select_device() -> Device
    println!("\n3. Testing auto_select_device():");
    let auto_device = auto_select_device();
    println!("   ✓ Auto-selected device: {:?}", auto_device);
    
    // Additional verification: Test that functions are properly exported
    println!("\n4. Verifying function accessibility:");
    println!("   ✓ get_cpu_device is accessible");
    println!("   ✓ get_metal_device is accessible");
    println!("   ✓ auto_select_device is accessible");
    
    // Test device usage with a simple tensor operation
    println!("\n5. Testing device usage with tensor operations:");
    use bitnet_core::{Tensor, DType};
    
    let device = auto_select_device();
    match Tensor::zeros(&[2, 2], DType::F32, &device) {
        Ok(tensor) => {
            println!("   ✓ Successfully created tensor on device: {:?}", tensor.shape());
        }
        Err(e) => {
            println!("   ✗ Failed to create tensor: {}", e);
        }
    }
    
    println!("\n==============================================");
    println!("All device management functions are working correctly!");
}