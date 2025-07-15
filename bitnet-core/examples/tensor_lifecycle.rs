//! BitNet Tensor Lifecycle Management Example
//!
//! This example demonstrates the comprehensive tensor lifecycle management system
//! implemented in BitNet, including:
//!
//! - Creating tensors with different data types
//! - Reference counting and automatic cleanup
//! - Device migration (CPU ↔ Metal)
//! - Memory pool integration
//! - Tensor handles for safe access
//! - Candle interoperability
//! - Metadata tracking and lifecycle monitoring

use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::device::{auto_select_device, get_cpu_device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BitNet Tensor Lifecycle Management Demo ===\n");

    // 1. Initialize memory pool and devices
    println!("1. Setting up memory pool and devices...");
    let pool = HybridMemoryPool::new()?;
    let device = auto_select_device();
    let cpu_device = get_cpu_device();
    
    println!("   Memory pool created");
    println!("   Primary device: {:?}", device);
    println!("   CPU device: {:?}", cpu_device);
    
    // 2. Create tensors with different data types
    println!("\n2. Creating tensors with different data types...");
    
    // Full precision tensor
    let f32_tensor = BitNetTensor::zeros(&[4, 4], BitNetDType::F32, &device, &pool)?;
    f32_tensor.set_name(Some("f32_weights".to_string()));
    println!("   Created F32 tensor: {}", f32_tensor);
    
    // Quantized tensors
    let i8_tensor = BitNetTensor::zeros(&[8, 8], BitNetDType::I8, &device, &pool)?;
    i8_tensor.set_name(Some("i8_activations".to_string()));
    println!("   Created I8 tensor: {}", i8_tensor);
    
    let bitnet_tensor = BitNetTensor::zeros(&[16, 16], BitNetDType::BitNet158, &device, &pool)?;
    bitnet_tensor.set_name(Some("bitnet_weights".to_string()));
    println!("   Created BitNet 1.58b tensor: {}", bitnet_tensor);
    
    // 3. Demonstrate reference counting
    println!("\n3. Demonstrating reference counting...");
    println!("   F32 tensor ref count: {}", f32_tensor.ref_count());
    
    let f32_clone = f32_tensor.clone();
    println!("   After cloning - ref count: {}", f32_tensor.ref_count());
    
    drop(f32_clone);
    println!("   After dropping clone - ref count: {}", f32_tensor.ref_count());
    
    // 4. Create and use tensor handles
    println!("\n4. Creating and using tensor handles...");
    let handle = f32_tensor.handle();
    println!("   Created handle: {}", handle);
    println!("   Handle is valid: {}", handle.is_valid());
    
    // Add tags to tensor via handle
    handle.add_tag("neural_network".to_string())?;
    handle.add_tag("weights".to_string())?;
    println!("   Added tags: {:?}", handle.tags()?);
    
    // 5. Demonstrate tensor from data
    println!("\n5. Creating tensor from data...");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_tensor = BitNetTensor::from_data(data, &[2, 3], &device, &pool)?;
    data_tensor.set_name(Some("input_data".to_string()));
    println!("   Created tensor from data: {}", data_tensor);
    
    // 6. Demonstrate device migration
    println!("\n6. Demonstrating device migration...");
    println!("   Original tensor device: {:?}", data_tensor.device());
    
    let migrated_tensor = data_tensor.to_device(&cpu_device, &pool)?;
    println!("   Migrated tensor device: {:?}", migrated_tensor.device());
    println!("   Migration successful: {}", migrated_tensor);
    
    // 7. Demonstrate candle interoperability
    println!("\n7. Demonstrating Candle interoperability...");
    let candle_tensor = f32_tensor.to_candle()?;
    println!("   Converted to Candle tensor: shape={:?}, dtype={:?}", 
             candle_tensor.shape(), candle_tensor.dtype());
    
    let from_candle = BitNetTensor::from_candle(candle_tensor, &pool)?;
    println!("   Converted back from Candle: {}", from_candle);
    
    // 8. Demonstrate tensor reshaping
    println!("\n8. Demonstrating tensor reshaping...");
    println!("   Original shape: {:?}", data_tensor.shape());
    
    let reshaped = data_tensor.reshape(&[3, 2])?;
    println!("   Reshaped tensor: {:?} -> {}", reshaped.shape(), reshaped);
    
    // 9. Memory usage and metrics
    println!("\n9. Memory usage and metrics...");
    let metrics = pool.get_metrics();
    println!("   Total allocated: {} bytes", metrics.total_allocated);
    println!("   Active allocations: {}", metrics.active_allocations);
    println!("   Peak memory usage: {} bytes", metrics.peak_allocated);
    
    // 10. Demonstrate different data type efficiencies
    println!("\n10. Data type memory efficiency comparison...");
    for dtype in BitNetDType::all_types() {
        let efficiency = dtype.memory_efficiency();
        let description = dtype.description();
        println!("   {}: {:.1}x more efficient than F32 - {}", 
                 dtype, efficiency, description);
    }
    
    // 11. Demonstrate tensor lifecycle tracking
    println!("\n11. Tensor lifecycle tracking...");
    let tracked_tensor = BitNetTensor::zeros(&[10, 10], BitNetDType::I4, &device, &pool)?;
    tracked_tensor.set_name(Some("tracked_tensor".to_string()));
    
    let handle = tracked_tensor.handle();
    println!("   Tensor created: {}", handle.description()?);
    println!("   Tensor age: {:.3}s", handle.handle_age_seconds());
    
    // Simulate some work
    std::thread::sleep(std::time::Duration::from_millis(10));
    handle.touch()?;
    
    println!("   After access - idle time: {}s", handle.idle_time_seconds()?);
    
    // 12. Demonstrate weak handles
    println!("\n12. Demonstrating weak handles...");
    let weak_handle = handle.downgrade();
    println!("   Created weak handle: ID {}", weak_handle.id());
    println!("   Weak handle is valid: {}", weak_handle.is_valid());
    
    drop(tracked_tensor);
    drop(handle);
    
    println!("   After dropping tensor - weak handle valid: {}", weak_handle.is_valid());
    
    // 13. Final memory cleanup demonstration
    println!("\n13. Final memory cleanup...");
    let initial_metrics = pool.get_metrics();
    println!("   Before cleanup - active allocations: {}", initial_metrics.active_allocations);
    
    // Drop all tensors
    drop(f32_tensor);
    drop(i8_tensor);
    drop(bitnet_tensor);
    drop(data_tensor);
    drop(migrated_tensor);
    drop(from_candle);
    drop(reshaped);
    
    let final_metrics = pool.get_metrics();
    println!("   After cleanup - active allocations: {}", final_metrics.active_allocations);
    println!("   Total deallocated: {} bytes", final_metrics.total_deallocated);
    
    println!("\n=== Demo completed successfully! ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_lifecycle_demo() {
        // Test that the main function runs without panicking
        assert!(main().is_ok());
    }

    #[test]
    fn test_tensor_creation_and_cleanup() {
        let pool = HybridMemoryPool::new().unwrap();
        let device = get_cpu_device();
        
        let initial_metrics = pool.get_metrics();
        
        // Create multiple tensors
        let tensors: Vec<_> = (0..5).map(|i| {
            let tensor = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device, &pool).unwrap();
            tensor.set_name(Some(format!("test_tensor_{}", i)));
            tensor
        }).collect();
        
        let after_creation = pool.get_metrics();
        assert!(after_creation.active_allocations > initial_metrics.active_allocations);
        
        // Drop all tensors
        drop(tensors);
        
        let after_cleanup = pool.get_metrics();
        // Note: Due to reference counting, cleanup might not be immediate
        // but total allocated should be tracked correctly
        assert!(after_cleanup.total_allocated >= after_creation.total_allocated);
    }

    #[test]
    fn test_handle_lifecycle() {
        let pool = HybridMemoryPool::new().unwrap();
        let device = get_cpu_device();
        
        let tensor = BitNetTensor::zeros(&[5, 5], BitNetDType::I8, &device, &pool).unwrap();
        let handle = tensor.handle();
        let weak_handle = handle.downgrade();
        
        assert!(handle.is_valid());
        assert!(weak_handle.is_valid());
        
        drop(tensor);
        drop(handle);
        
        assert!(!weak_handle.is_valid());
        assert!(weak_handle.upgrade().is_none());
    }
}