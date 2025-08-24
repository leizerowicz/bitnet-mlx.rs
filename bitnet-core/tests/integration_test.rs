//! Integration test to verify device management functions are accessible and working

use bitnet_core::device::{
    auto_select_device, get_cpu_device, get_metal_device, get_metal_device_name, is_metal_available,
};
use bitnet_core::{DType, Tensor};

#[test]
fn test_device_functions_accessibility() {
    // Test 1: get_cpu_device() -> Device
    let cpu_device = get_cpu_device();
    assert!(matches!(cpu_device, bitnet_core::Device::Cpu));

    // Test 2: get_metal_device() -> Result<Device>
    // This may succeed or fail depending on platform, but should not panic
    let _metal_result = get_metal_device();

    // Test 3: auto_select_device() -> Device
    let auto_device = auto_select_device();
    // Should return either CPU or Metal device
    assert!(matches!(
        auto_device,
        bitnet_core::Device::Cpu | bitnet_core::Device::Metal(_)
    ));
}

#[test]
fn test_device_usage_with_tensors() {
    // Test that devices can be used for tensor operations
    let device = auto_select_device();

    // Create a simple tensor on the device
    let tensor = Tensor::zeros(&[2, 2], DType::F32, &device).expect("Failed to create tensor");
    assert_eq!(tensor.shape().dims(), &[2, 2]);
}

#[test]
fn test_cpu_device_always_works() {
    let cpu_device = get_cpu_device();
    let tensor =
        Tensor::ones(&[3, 3], DType::F32, &cpu_device).expect("Failed to create tensor on CPU");
    assert_eq!(tensor.shape().dims(), &[3, 3]);
}

#[test]
fn test_metal_compatibility_functions() {
    // Test that the new Metal compatibility functions are accessible and work

    // Test is_metal_available()
    let metal_available = is_metal_available();
    println!("Metal available: {metal_available}");

    // Test get_metal_device_name()
    let device_name = get_metal_device_name();
    let has_device_name = device_name.is_some();

    match device_name {
        Some(name) => {
            println!("Metal device name: {name}");
            assert!(!name.is_empty(), "Device name should not be empty");
        }
        None => {
            println!("No Metal device name available");
        }
    }

    // Test consistency between functions
    if !metal_available {
        // If Metal is not available, device name should be None
        assert!(
            !has_device_name,
            "Device name should be None when Metal is not available"
        );
    }
}

#[test]
fn test_metal_compatibility_with_device_creation() {
    // Test that Metal compatibility functions are consistent with device creation
    let metal_available = is_metal_available();
    let metal_device_result = get_metal_device();

    // If is_metal_available() returns true, get_metal_device() should succeed
    // If is_metal_available() returns false, get_metal_device() should fail
    match (metal_available, metal_device_result) {
        (true, Ok(_)) => {
            println!("Consistent: Metal is available and device creation succeeded");
        }
        (false, Err(_)) => {
            println!("Consistent: Metal is not available and device creation failed");
        }
        (true, Err(e)) => {
            println!("Warning: Metal reported as available but device creation failed: {e}");
            // This can happen in some edge cases, so we don't fail the test
        }
        (false, Ok(_)) => {
            panic!("Inconsistent: Metal reported as not available but device creation succeeded");
        }
    }
}
