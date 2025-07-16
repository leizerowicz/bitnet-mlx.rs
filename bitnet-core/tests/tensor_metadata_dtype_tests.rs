//! Comprehensive Unit Tests for Tensor Metadata and DType Operations
//!
//! This test suite provides comprehensive coverage for tensor metadata and dtype operations,
//! focusing on advanced features, edge cases, and integration scenarios that extend beyond
//! the basic tests already present in the individual modules.
//!
//! # Test Coverage Areas:
//! - Enhanced TensorMetadata lifecycle scenarios
//! - Comprehensive BitNetDType validation and precision tests
//! - Metadata-DType integration and compatibility
//! - Advanced metadata features (tagging, classification, search)
//! - Performance and efficiency validation
//! - Edge cases and boundary conditions
//! - Concurrent access patterns and thread safety

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::collections::{HashMap, HashSet};

use bitnet_core::memory::tensor::{BitNetDType, TensorMetadata, DeviceInfo};
use bitnet_core::device::{get_cpu_device, auto_select_device, is_metal_available, get_metal_device};
use candle_core::{Device, DType};
use serde_json;
use approx::assert_relative_eq;
use rand::{Rng, thread_rng};

// =============================================================================
// Test Utilities and Helpers
// =============================================================================

/// Helper function to create test metadata with various configurations
fn create_test_metadata(
    id: u64,
    shape: Vec<usize>,
    dtype: BitNetDType,
    device: &Device,
    name: Option<String>,
) -> TensorMetadata {
    TensorMetadata::new(id, shape, dtype, device, name)
}

/// Helper function to get all available devices for testing
fn get_test_devices() -> Vec<Device> {
    let mut devices = vec![get_cpu_device()];
    
    if is_metal_available() {
        if let Ok(metal_device) = get_metal_device() {
            devices.push(metal_device);
        }
    }
    
    devices
}

/// Helper function to get all BitNet data types
fn get_all_dtypes() -> Vec<BitNetDType> {
    BitNetDType::all_types().to_vec()
}

/// Helper function to create metadata with random properties
fn create_random_metadata(id: u64) -> TensorMetadata {
    let mut rng = thread_rng();
    let devices = get_test_devices();
    let dtypes = get_all_dtypes();
    
    let device = &devices[rng.gen_range(0..devices.len())];
    let dtype = dtypes[rng.gen_range(0..dtypes.len())];
    
    // Generate random shape (1-4 dimensions, each 1-100 elements)
    let rank = rng.gen_range(1..=4);
    let shape: Vec<usize> = (0..rank).map(|_| rng.gen_range(1..=100)).collect();
    
    let name = if rng.gen_bool(0.5) {
        Some(format!("tensor_{}", id))
    } else {
        None
    };
    
    TensorMetadata::new(id, shape, dtype, device, name)
}

/// Helper function to simulate time passage
fn simulate_time_passage(duration: Duration) {
    thread::sleep(duration);
}

// =============================================================================
// Enhanced TensorMetadata Lifecycle Tests
// =============================================================================

#[test]
fn test_metadata_advanced_lifecycle_scenarios() {
    let device = get_cpu_device();
    
    // Test complex lifecycle with multiple state changes
    let mut metadata = create_test_metadata(
        1,
        vec![100, 100],
        BitNetDType::F32,
        &device,
        Some("lifecycle_test".to_string()),
    );
    
    let initial_created_at = metadata.created_at;
    let initial_last_accessed = metadata.last_accessed;
    
    // Simulate usage pattern
    simulate_time_passage(Duration::from_millis(10));
    metadata.touch();
    assert!(metadata.last_accessed > initial_last_accessed);
    assert_eq!(metadata.created_at, initial_created_at); // Should not change
    
    // Test reference counting lifecycle
    assert_eq!(metadata.ref_count(), 1);
    
    for i in 2..=10 {
        metadata.add_ref();
        assert_eq!(metadata.ref_count(), i);
    }
    
    for i in (1..10).rev() {
        let should_be_zero = metadata.remove_ref();
        assert_eq!(metadata.ref_count(), i);
        assert_eq!(should_be_zero, i == 0);
    }
    
    // Test migration lifecycle
    assert!(!metadata.is_migrating());
    metadata.set_migrating(true);
    assert!(metadata.is_migrating());
    metadata.set_migrating(false);
    assert!(!metadata.is_migrating());
    
    // Test age and idle time calculations
    simulate_time_passage(Duration::from_millis(50));
    let age = metadata.age_seconds();
    let idle_time = metadata.idle_time_seconds();
    
    assert!(age > 0);
    assert!(idle_time > 0);
    assert!(idle_time >= age); // Idle time should be >= age since last touch
}

#[test]
fn test_metadata_serialization_edge_cases() {
    let devices = get_test_devices();
    let dtypes = get_all_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            // Test serialization with various edge case configurations
            let test_cases = vec![
                // Empty shape (scalar)
                (vec![], "scalar"),
                // Single element vector
                (vec![1], "single_element"),
                // Large dimensions
                (vec![1000000], "large_single_dim"),
                // Many small dimensions
                (vec![2, 2, 2, 2, 2, 2], "many_small_dims"),
                // Mixed large and small
                (vec![1, 1000, 1], "mixed_dimensions"),
            ];
            
            for (shape, test_name) in test_cases {
                let mut metadata = create_test_metadata(
                    42,
                    shape.clone(),
                    dtype,
                    device,
                    Some(format!("{}_{}", test_name, dtype)),
                );
                
                // Add complex state
                metadata.add_tag("serialization_test".to_string());
                metadata.add_tag(format!("dtype_{}", dtype));
                metadata.add_tag(format!("device_{:?}", device));
                metadata.set_migrating(true);
                metadata.add_ref();
                metadata.add_ref();
                
                // Serialize
                let serialized = serde_json::to_string(&metadata)
                    .expect("Failed to serialize metadata");
                
                // Deserialize
                let deserialized: TensorMetadata = serde_json::from_str(&serialized)
                    .expect("Failed to deserialize metadata");
                
                // Verify all fields are preserved
                assert_eq!(metadata.id, deserialized.id);
                assert_eq!(metadata.shape, deserialized.shape);
                assert_eq!(metadata.dtype, deserialized.dtype);
                assert_eq!(metadata.size_bytes, deserialized.size_bytes);
                assert_eq!(metadata.element_count, deserialized.element_count);
                assert_eq!(metadata.created_at, deserialized.created_at);
                assert_eq!(metadata.last_accessed, deserialized.last_accessed);
                assert_eq!(metadata.ref_count, deserialized.ref_count);
                assert_eq!(metadata.is_migrating, deserialized.is_migrating);
                assert_eq!(metadata.name, deserialized.name);
                assert_eq!(metadata.tags, deserialized.tags);
                
                // Verify device info serialization
                match (&metadata.device, &deserialized.device) {
                    (DeviceInfo::Cpu, DeviceInfo::Cpu) => {},
                    (DeviceInfo::Metal(a), DeviceInfo::Metal(b)) => assert_eq!(a, b),
                    (DeviceInfo::Cuda(a), DeviceInfo::Cuda(b)) => assert_eq!(a, b),
                    _ => panic!("Device info mismatch after serialization"),
                }
            }
        }
    }
}

#[test]
fn test_metadata_performance_under_high_frequency_updates() {
    let device = get_cpu_device();
    let mut metadata = create_test_metadata(
        1,
        vec![1000, 1000],
        BitNetDType::F32,
        &device,
        Some("performance_test".to_string()),
    );
    
    let iterations = 10000;
    
    // Test touch() performance
    let start_time = Instant::now();
    for _ in 0..iterations {
        metadata.touch();
    }
    let touch_duration = start_time.elapsed();
    let avg_touch_time = touch_duration / iterations;
    
    println!("Average touch() time: {:?}", avg_touch_time);
    assert!(avg_touch_time < Duration::from_micros(10), "touch() too slow");
    
    // Test reference counting performance
    let start_time = Instant::now();
    for _ in 0..iterations {
        metadata.add_ref();
        metadata.remove_ref();
    }
    let ref_count_duration = start_time.elapsed();
    let avg_ref_count_time = ref_count_duration / (iterations * 2);
    
    println!("Average ref count operation time: {:?}", avg_ref_count_time);
    assert!(avg_ref_count_time < Duration::from_nanos(100), "ref counting too slow");
    
    // Test tag operations performance
    let start_time = Instant::now();
    for i in 0..1000 {
        let tag = format!("tag_{}", i);
        metadata.add_tag(tag.clone());
        assert!(metadata.has_tag(&tag));
        metadata.remove_tag(&tag);
        assert!(!metadata.has_tag(&tag));
    }
    let tag_duration = start_time.elapsed();
    let avg_tag_time = tag_duration / (1000 * 3); // 3 operations per iteration
    
    println!("Average tag operation time: {:?}", avg_tag_time);
    assert!(avg_tag_time < Duration::from_micros(50), "tag operations too slow");
}

#[test]
fn test_metadata_consistency_across_device_migrations() {
    let devices = get_test_devices();
    
    if devices.len() < 2 {
        println!("Skipping device migration test - need multiple devices");
        return;
    }
    
    let source_device = &devices[0];
    let target_device = &devices[1];
    
    let mut metadata = create_test_metadata(
        1,
        vec![256, 256],
        BitNetDType::F32,
        source_device,
        Some("migration_test".to_string()),
    );
    
    // Add complex state before migration
    metadata.add_tag("pre_migration".to_string());
    metadata.add_tag("source_device".to_string());
    metadata.add_ref();
    metadata.add_ref();
    let pre_migration_age = metadata.age_seconds();
    
    // Simulate migration
    simulate_time_passage(Duration::from_millis(10));
    metadata.update_device(target_device);
    
    // Verify state consistency after migration
    assert_eq!(metadata.shape, vec![256, 256]);
    assert_eq!(metadata.dtype, BitNetDType::F32);
    assert_eq!(metadata.element_count, 256 * 256);
    assert_eq!(metadata.size_bytes, BitNetDType::F32.bytes_for_elements(256 * 256));
    assert_eq!(metadata.ref_count, 3); // Should be preserved
    assert!(metadata.has_tag("pre_migration"));
    assert!(metadata.has_tag("source_device"));
    
    // Verify device was updated
    match metadata.device {
        DeviceInfo::Cpu if matches!(target_device, Device::Cpu) => {},
        DeviceInfo::Metal(_) if matches!(target_device, Device::Metal(_)) => {},
        DeviceInfo::Cuda(_) if matches!(target_device, Device::Cuda(_)) => {},
        _ => panic!("Device not properly updated"),
    }
    
    // Verify timestamps were updated
    let post_migration_age = metadata.age_seconds();
    assert!(post_migration_age >= pre_migration_age);
    
    // Test shape update consistency
    metadata.update_shape(vec![128, 128, 4]);
    assert_eq!(metadata.shape, vec![128, 128, 4]);
    assert_eq!(metadata.element_count, 128 * 128 * 4);
    assert_eq!(metadata.size_bytes, BitNetDType::F32.bytes_for_elements(128 * 128 * 4));
}

#[test]
fn test_metadata_concurrent_access_patterns() {
    let device = get_cpu_device();
    let metadata = Arc::new(Mutex::new(create_test_metadata(
        1,
        vec![100, 100],
        BitNetDType::F32,
        &device,
        Some("concurrent_test".to_string()),
    )));
    
    let thread_count = 8;
    let operations_per_thread = 1000;
    let mut handles = Vec::new();
    
    // Spawn threads performing concurrent operations
    for thread_id in 0..thread_count {
        let metadata_clone = metadata.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..operations_per_thread {
                let mut meta = metadata_clone.lock().unwrap();
                
                // Perform various operations
                meta.touch();
                meta.add_tag(format!("thread_{}_{}", thread_id, i));
                meta.add_ref();
                
                if i % 10 == 0 {
                    meta.set_migrating(i % 20 == 0);
                }
                
                if i % 5 == 0 {
                    meta.remove_ref();
                }
                
                // Verify consistency
                assert!(meta.ref_count() > 0);
                assert!(meta.has_tag(&format!("thread_{}_{}", thread_id, i)));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify final state consistency
    let final_meta = metadata.lock().unwrap();
    assert!(final_meta.ref_count() > 1);
    assert!(final_meta.tags.len() >= thread_count * operations_per_thread);
    
    // Verify all expected tags are present
    for thread_id in 0..thread_count {
        for i in 0..operations_per_thread {
            let tag = format!("thread_{}_{}", thread_id, i);
            assert!(final_meta.has_tag(&tag), "Missing tag: {}", tag);
        }
    }
}

// =============================================================================
// Comprehensive BitNetDType Tests
// =============================================================================

#[test]
fn test_all_bitnet_dtypes_comprehensive() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        println!("Testing dtype: {}", dtype);
        
        // Test basic properties
        let bits = dtype.bits_per_element();
        let description = dtype.description();
        let efficiency = dtype.memory_efficiency();
        
        assert!(bits > 0, "Bits per element should be positive for {}", dtype);
        assert!(!description.is_empty(), "Description should not be empty for {}", dtype);
        assert!(efficiency > 0.0, "Memory efficiency should be positive for {}", dtype);
        
        // Test bytes calculation for various element counts
        let test_counts = vec![1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1000, 10000];
        
        for count in test_counts {
            let bytes = dtype.bytes_for_elements(count);
            let expected_bits = bits * count;
            let expected_bytes = (expected_bits + 7) / 8; // Round up to bytes
            
            assert_eq!(bytes, expected_bytes, 
                      "Byte calculation mismatch for {} with {} elements", dtype, count);
        }
        
        // Test type classification
        let is_float = dtype.is_float();
        let is_integer = dtype.is_integer();
        let is_quantized = dtype.is_quantized();
        let is_bitnet158 = dtype.is_bitnet158();
        
        // Verify mutual exclusivity where appropriate
        if is_float {
            assert!(!is_integer, "Type cannot be both float and integer: {}", dtype);
        }
        
        if is_bitnet158 {
            assert_eq!(dtype, BitNetDType::BitNet158, "Only BitNet158 should return true for is_bitnet158");
            assert!(is_quantized, "BitNet158 should be considered quantized");
        }
        
        // Test value ranges for integer types
        if is_integer || is_bitnet158 {
            if let Some((min, max)) = dtype.value_range() {
                assert!(min <= max, "Min should be <= max for {}", dtype);
                
                // Verify range matches expected bit width
                let range_size = (max - min + 1) as usize;
                let max_values = 1 << bits;
                
                if dtype == BitNetDType::BitNet158 {
                    // Special case: ternary values (-1, 0, 1) = 3 values
                    assert_eq!(range_size, 3, "BitNet158 should have 3 possible values");
                } else {
                    assert!(range_size <= max_values, 
                           "Range size {} exceeds max values {} for {}", 
                           range_size, max_values, dtype);
                }
            }
        } else {
            assert!(dtype.value_range().is_none(), 
                   "Float types should not have value ranges: {}", dtype);
        }
        
        // Test Candle dtype conversion
        let candle_dtype = dtype.to_candle_dtype();
        assert!(matches!(candle_dtype, DType::F32 | DType::F16 | DType::BF16 | DType::I64),
               "Unexpected candle dtype for {}: {:?}", dtype, candle_dtype);
        
        // Test memory efficiency calculation
        let expected_efficiency = 32.0 / bits as f32;
        assert_relative_eq!(efficiency, expected_efficiency, epsilon = 0.001,
                           "Memory efficiency mismatch for {}", dtype);
    }
}

#[test]
fn test_dtype_conversion_accuracy_and_precision() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        // Test roundtrip conversion with Candle
        let candle_dtype = dtype.to_candle_dtype();
        
        // Test conversion back (where possible)
        if let Some(converted_back) = BitNetDType::from_candle_dtype(candle_dtype) {
            // For types that have direct Candle equivalents
            match dtype {
                BitNetDType::F32 => assert_eq!(converted_back, BitNetDType::F32),
                BitNetDType::F16 => assert_eq!(converted_back, BitNetDType::F16),
                BitNetDType::BF16 => assert_eq!(converted_back, BitNetDType::BF16),
                _ => {
                    // Integer types default to I8 in conversion
                    if dtype.is_integer() || dtype.is_bitnet158() {
                        assert_eq!(converted_back, BitNetDType::I8);
                    }
                }
            }
        }
        
        // Test precision for quantized types
        if dtype.is_quantized() {
            if let Some((min_val, max_val)) = dtype.value_range() {
                // Verify that the range can represent the expected precision
                let range_size = max_val - min_val + 1;
                let bits = dtype.bits_per_element();
                
                match dtype {
                    BitNetDType::I4 => {
                        assert_eq!(range_size, 16, "I4 should have 16 possible values");
                        assert_eq!(min_val, -8);
                        assert_eq!(max_val, 7);
                    }
                    BitNetDType::I2 => {
                        assert_eq!(range_size, 4, "I2 should have 4 possible values");
                        assert_eq!(min_val, -2);
                        assert_eq!(max_val, 1);
                    }
                    BitNetDType::I1 => {
                        assert_eq!(range_size, 2, "I1 should have 2 possible values");
                        assert_eq!(min_val, -1);
                        assert_eq!(max_val, 0);
                    }
                    BitNetDType::BitNet158 => {
                        assert_eq!(range_size, 3, "BitNet158 should have 3 possible values");
                        assert_eq!(min_val, -1);
                        assert_eq!(max_val, 1);
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn test_dtype_memory_efficiency_calculations() {
    let dtypes = get_all_dtypes();
    let element_counts = vec![1, 8, 64, 512, 1024, 8192];
    
    for &dtype in &dtypes {
        for &count in &element_counts {
            let bytes = dtype.bytes_for_elements(count);
            let efficiency = dtype.memory_efficiency();
            
            // Compare with F32 baseline
            let f32_bytes = BitNetDType::F32.bytes_for_elements(count);
            let actual_efficiency = f32_bytes as f32 / bytes as f32;
            
            assert_relative_eq!(efficiency, actual_efficiency, epsilon = 0.01,
                               "Efficiency mismatch for {} with {} elements", dtype, count);
            
            // Verify memory savings for quantized types
            if dtype.is_quantized() {
                assert!(bytes <= f32_bytes, 
                       "Quantized type {} should use less memory than F32", dtype);
                assert!(efficiency >= 1.0, 
                       "Quantized type {} should have efficiency >= 1.0", dtype);
            }
            
            // Verify bit-level efficiency for sub-byte types
            if dtype.bits_per_element() < 8 {
                let bits_used = dtype.bits_per_element() * count;
                let bytes_needed = (bits_used + 7) / 8; // Round up
                assert_eq!(bytes, bytes_needed,
                          "Bit packing incorrect for {} with {} elements", dtype, count);
            }
        }
    }
}

#[test]
fn test_dtype_compatibility_with_operations() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        // Test compatibility with common tensor operations
        let candle_dtype = dtype.to_candle_dtype();
        
        // Verify that the Candle dtype can be used for basic operations
        match candle_dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                // Float types should support all operations
                assert!(dtype.is_float(), "Float Candle dtype should correspond to float BitNet dtype");
            }
            DType::I64 => {
                // Integer operations
                assert!(dtype.is_integer() || dtype.is_bitnet158(), 
                       "I64 Candle dtype should correspond to integer or BitNet158 dtype");
            }
            _ => {
                panic!("Unexpected Candle dtype: {:?}", candle_dtype);
            }
        }
        
        // Test serialization compatibility
        let serialized = serde_json::to_string(&dtype)
            .expect(&format!("Failed to serialize dtype {}", dtype));
        let deserialized: BitNetDType = serde_json::from_str(&serialized)
            .expect(&format!("Failed to deserialize dtype {}", dtype));
        assert_eq!(dtype, deserialized, "Serialization roundtrip failed for {}", dtype);
    }
}

#[test]
fn test_dtype_edge_cases_and_boundary_conditions() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        // Test with zero elements
        let zero_bytes = dtype.bytes_for_elements(0);
        assert_eq!(zero_bytes, 0, "Zero elements should require zero bytes for {}", dtype);
        
        // Test with maximum reasonable element count
        let max_elements = 1_000_000;
        let max_bytes = dtype.bytes_for_elements(max_elements);
        assert!(max_bytes > 0, "Large element count should require positive bytes for {}", dtype);
        
        // Test bit packing edge cases for sub-byte types
        if dtype.bits_per_element() < 8 {
            let bits_per_element = dtype.bits_per_element();
            
            // Test exact byte boundaries
            let elements_per_byte = 8 / bits_per_element;
            let test_counts = vec![
                elements_per_byte - 1,  // Just under a byte
                elements_per_byte,      // Exactly one byte
                elements_per_byte + 1,  // Just over a byte
                elements_per_byte * 2,  // Exactly two bytes
                elements_per_byte * 2 + 1, // Just over two bytes
            ];
            
            for count in test_counts {
                let bytes = dtype.bytes_for_elements(count);
                let expected_bytes = (count * bits_per_element + 7) / 8;
                assert_eq!(bytes, expected_bytes,
                          "Bit packing boundary test failed for {} with {} elements", 
                          dtype, count);
            }
        }
        
        // Test value range boundaries for integer types
        if let Some((min_val, max_val)) = dtype.value_range() {
            // Verify the range is symmetric or follows expected patterns
            match dtype {
                BitNetDType::I8 => {
                    assert_eq!(min_val, -128);
                    assert_eq!(max_val, 127);
                }
                BitNetDType::I4 => {
                    assert_eq!(min_val, -8);
                    assert_eq!(max_val, 7);
                }
                BitNetDType::I2 => {
                    assert_eq!(min_val, -2);
                    assert_eq!(max_val, 1);
                }
                BitNetDType::I1 => {
                    assert_eq!(min_val, -1);
                    assert_eq!(max_val, 0);
                }
                BitNetDType::BitNet158 => {
                    assert_eq!(min_val, -1);
                    assert_eq!(max_val, 1);
                }
                _ => {}
            }
        }
    }
}

// =============================================================================
// Metadata-DType Integration Tests
// =============================================================================

#[test]
fn test_metadata_accuracy_with_different_dtypes() {
    let device = get_cpu_device();
    let dtypes = get_all_dtypes();
    let shapes = vec![
        vec![],           // scalar
        vec![10],         // vector
        vec![5, 5],       // matrix
        vec![2, 3, 4],    // 3D tensor
        vec![2, 2, 2, 2], // 4D tensor
    ];
    
    for &dtype in &dtypes {
        for shape in &shapes {
            let metadata = create_test_metadata(
                1,
                shape.clone(),
                dtype,
                &device,
                Some(format!("test_{}_{:?}", dtype, shape)),
            );
            
            // Verify element count calculation
            let expected_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
            assert_eq!(metadata.element_count, expected_elements,
                      "Element count mismatch for {} with shape {:?}", dtype, shape);
            
            // Verify size calculation
            let expected_size = dtype.bytes_for_elements(expected_elements);
            assert_eq!(metadata.size_bytes, expected_size,
                      "Size calculation mismatch for {} with shape {:?}", dtype, shape);
            
            // Verify memory efficiency
            let efficiency = metadata.memory_efficiency();
            let expected_efficiency = dtype.memory_efficiency();
            assert_relative_eq!(efficiency, expected_efficiency, epsilon = 0.001,
                               "Memory efficiency mismatch for {} with shape {:?}", dtype, shape);
            
            // Verify tensor classification
            assert_eq!(metadata.is_scalar(), shape.is_empty());
            assert_eq!(metadata.is_vector(), shape.len() == 1);
            assert_eq!(metadata.is_matrix(), shape.len() == 2);
            assert_eq!(metadata.rank(), shape.len());
        }
    }
}

#[test]
fn test_memory_calculations_with_various_dtypes() {
    let device = get_cpu_device();
    let dtypes = get_all_dtypes();
    
    // Test with various element counts to verify memory calculations
    let element_counts = vec![1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1000];
    
    for &dtype in &dtypes {
        for &count in &element_counts {
            let shape = vec![count];
            let metadata = create_test_metadata(1, shape, dtype, &device, None);
            
            // Verify calculations match dtype methods
            assert_eq!(metadata.element_count, count);
            assert_eq!(metadata.size_bytes, dtype.bytes_for_elements(count));
            assert_relative_eq!(metadata.memory_efficiency(), dtype.memory_efficiency(), epsilon = 0.001);
            
            // For quantized types, verify bit-level efficiency
            if dtype.is_quantized() {
                let bits_used = dtype.bits_per_element() * count;
                let bytes_used = (bits_used + 7) / 8;
                assert_eq!(metadata.size_bytes, bytes_used,
                          "Bit packing calculation incorrect for {} with {} elements", dtype, count);
            }
        }
    }
}

#[test]
fn test_device_compatibility_across_dtypes() {
    let devices = get_test_devices();
    let dtypes = get_all_dtypes();
    
    for device in &devices {
        for &dtype in &dtypes {
            let metadata = create_test_metadata(
                1,
                vec![100],
                dtype,
                device,
                Some(format!("device_test_{}_{:?}", dtype, device)),
            );
            
            // Verify device info is correctly stored
            match (&metadata.device, device) {
                (DeviceInfo::Cpu, Device::Cpu) => {},
                (DeviceInfo::Metal(_), Device::Metal(_)) => {},
                (DeviceInfo::Cuda(_), Device::Cuda(_)) => {},
                _ => panic!("Device info mismatch for {} on {:?}", dtype, device),
            }
            
            // Test device migration with different dtypes
            for target_device in &devices {
                if target_device != device {
                    let mut migrated_metadata = metadata.clone();
                    migrated_metadata.update_device(target_device);
                    
                    // Verify dtype is preserved during migration
                    assert_eq!(migrated_metadata.dtype, dtype);
                    assert_eq!(migrated_metadata.element_count, metadata.element_count);
                    assert_eq!(migrated_metadata.size_bytes, metadata.size_bytes);
                }
            }
        }
    }
}

#[test]
fn test_metadata_updates_during_dtype_conversions() {
    let device = get_cpu_device();
    let dtypes = get_all_dtypes();
    
    for &source_dtype in &dtypes {
        let mut metadata = create_test_metadata(
            1,
            vec![100],
            source_dtype,
            &device,
            Some(format!("conversion_test_{}", source_dtype)),
        );
        
        let original_element_count = metadata.element_count;
        let original_shape = metadata.shape.clone();
        
        // Simulate dtype conversion by updating metadata
        for &target_dtype in &dtypes {
            if source_dtype != target_dtype {
                // Update dtype and recalculate size
                metadata.dtype = target_dtype;
                metadata.size_bytes = target_dtype.bytes_for_elements(metadata.element_count);
                metadata.touch();
                
                // Verify consistency after conversion
                assert_eq!(metadata.dtype, target_dtype);
                assert_eq!(metadata.element_count, original_element_count);
                assert_eq!(metadata.shape, original_shape);
                assert_eq!(metadata.size_bytes, target_dtype.bytes_for_elements(original_element_count));
                
                // Verify memory efficiency is updated
                let expected_efficiency = target_dtype.memory_efficiency();
                assert_relative_eq!(metadata.memory_efficiency(), expected_efficiency, epsilon = 0.001);
            }
        }
    }
}

// =============================================================================
// Advanced Metadata Features Tests
// =============================================================================

#[test]
fn test_metadata_tagging_and_classification_systems() {
    let device = get_cpu_device();
    let mut metadata = create_test_metadata(
        1,
        vec![256, 256],
        BitNetDType::F32,
        &device,
        Some("classification_test".to_string()),
    );
    
    // Test hierarchical tagging system
    let tag_categories = vec![
        ("type", vec!["weight", "bias", "activation"]),
        ("layer", vec!["conv1", "conv2", "fc1", "fc2"]),
        ("optimization", vec!["quantized", "pruned", "compressed"]),
        ("device", vec!["cpu", "gpu", "tpu"]),
        ("precision", vec!["fp32", "fp16", "int8", "int4"]),
    ];
    
    for (category, tags) in tag_categories {
        for tag in tags {
            let full_tag = format!("{}:{}", category, tag);
            metadata.add_tag(full_tag.clone());
            assert!(metadata.has_tag(&full_tag));
        }
    }
    
    // Test tag search and filtering
    let type_tags: Vec<_> = metadata.tags.iter()
        .filter(|tag| tag.starts_with("type:"))
        .collect();
    assert_eq!(type_tags.len(), 3);
    
    let layer_tags: Vec<_> = metadata.tags.iter()
        .filter(|tag| tag.starts_with("layer:"))
        .collect();
    assert_eq!(layer_tags.len(), 4);
    
    // Test tag removal by category
    metadata.tags.retain(|tag| !tag.starts_with("optimization:"));
    assert!(!metadata.has_tag("optimization:quantized"));
    assert!(!metadata.has_tag("optimization:pruned"));
    assert!(!metadata.has_tag("optimization:compressed"));
    
    // Verify other tags remain
    assert!(metadata.has_tag("type:weight"));
    assert!(metadata.has_tag("layer:conv1"));
    
    // Test duplicate tag prevention
    let initial_count = metadata.tags.len();
    metadata.add_tag("type:weight".to_string()); // Already exists
    assert_eq!(metadata.tags.len(), initial_count);
}

#[test]
fn test_comprehensive_dtype_precision_validation() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        println!("Validating precision for: {}", dtype);
        
        // Test precision characteristics
        let bits = dtype.bits_per_element();
        
        match dtype {
            BitNetDType::F32 => {
                assert_eq!(bits, 32);
                assert!(dtype.is_float());
                assert!(!dtype.is_quantized());
            }
            BitNetDType::F16 => {
                assert_eq!(bits, 16);
                assert!(dtype.is_float());
                assert!(!dtype.is_quantized());
            }
            BitNetDType::BF16 => {
                assert_eq!(bits, 16);
                assert!(dtype.is_float());
                assert!(!dtype.is_quantized());
            }
            BitNetDType::I8 => {
                assert_eq!(bits, 8);
                assert!(dtype.is_integer());
                assert!(!dtype.is_quantized());
                
                if let Some((min, max)) = dtype.value_range() {
                    assert_eq!(min, -128);
                    assert_eq!(max, 127);
                    assert_eq!(max - min + 1, 256);
                }
            }
            BitNetDType::I4 => {
                assert_eq!(bits, 4);
                assert!(dtype.is_integer());
                assert!(dtype.is_quantized());
                
                if let Some((min, max)) = dtype.value_range() {
                    assert_eq!(min, -8);
                    assert_eq!(max, 7);
                    assert_eq!(max - min + 1, 16);
                }
            }
            BitNetDType::I2 => {
                assert_eq!(bits, 2);
                assert!(dtype.is_integer());
                assert!(dtype.is_quantized());
                
                if let Some((min, max)) = dtype.value_range() {
                    assert_eq!(min, -2);
                    assert_eq!(max, 1);
                    assert_eq!(max - min + 1, 4);
                }
            }
            BitNetDType::I1 => {
                assert_eq!(bits, 1);
                assert!(dtype.is_integer());
                assert!(dtype.is_quantized());
                
                if let Some((min, max)) = dtype.value_range() {
                    assert_eq!(min, -1);
                    assert_eq!(max, 0);
                    assert_eq!(max - min + 1, 2);
                }
            }
            BitNetDType::BitNet158 => {
                assert_eq!(bits, 2);
                assert!(!dtype.is_integer());
                assert!(dtype.is_quantized());
                assert!(dtype.is_bitnet158());
                
                if let Some((min, max)) = dtype.value_range() {
                    assert_eq!(min, -1);
                    assert_eq!(max, 1);
                    assert_eq!(max - min + 1, 3);
                }
            }
        }
    }
}

#[test]
fn test_performance_and_efficiency_comprehensive() {
    let device = get_cpu_device();
    let dtypes = get_all_dtypes();
    let element_counts = vec![1000, 10000, 100000];
    
    for &count in &element_counts {
        println!("Testing efficiency with {} elements", count);
        
        let mut efficiency_results = HashMap::new();
        let f32_bytes = BitNetDType::F32.bytes_for_elements(count);
        
        for &dtype in &dtypes {
            let bytes = dtype.bytes_for_elements(count);
            let efficiency = dtype.memory_efficiency();
            let actual_efficiency = f32_bytes as f32 / bytes as f32;
            
            efficiency_results.insert(dtype, (bytes, efficiency, actual_efficiency));
            
            // Verify efficiency calculation
            assert_relative_eq!(efficiency, actual_efficiency, epsilon = 0.01);
            
            // Create metadata to test integration
            let metadata = create_test_metadata(1, vec![count], dtype, &device, None);
            assert_eq!(metadata.size_bytes, bytes);
            assert_relative_eq!(metadata.memory_efficiency(), efficiency, epsilon = 0.001);
        }
        
        // Verify quantized types are more efficient
        for &dtype in &dtypes {
            if dtype.is_quantized() {
                let (bytes, efficiency, _) = efficiency_results[&dtype];
                assert!(efficiency > 1.0,
                       "Quantized type {} should be more efficient than F32", dtype);
                assert!(bytes < f32_bytes,
                       "Quantized type {} should use less memory than F32", dtype);
            }
        }
    }
}

#[test]
fn test_edge_cases_and_boundary_conditions_comprehensive() {
    let dtypes = get_all_dtypes();
    
    for &dtype in &dtypes {
        // Test with zero elements
        let zero_bytes = dtype.bytes_for_elements(0);
        assert_eq!(zero_bytes, 0, "Zero elements should require zero bytes for {}", dtype);
        
        // Test bit packing edge cases for sub-byte types
        if dtype.bits_per_element() < 8 {
            let bits_per_element = dtype.bits_per_element();
            let elements_per_byte = 8 / bits_per_element;
            
            // Test exact byte boundaries
            let test_counts = vec![
                elements_per_byte - 1,
                elements_per_byte,
                elements_per_byte + 1,
                elements_per_byte * 2,
                elements_per_byte * 2 + 1,
            ];
            
            for count in test_counts {
                let bytes = dtype.bytes_for_elements(count);
                let expected_bytes = (count * bits_per_element + 7) / 8;
                assert_eq!(bytes, expected_bytes,
                          "Bit packing boundary test failed for {} with {} elements",
                          dtype, count);
            }
        }
        
        // Test value range boundaries for integer types
        if let Some((min_val, max_val)) = dtype.value_range() {
            assert!(min_val <= max_val, "Min should be <= max for {}", dtype);
            
            // Test specific expected ranges
            match dtype {
                BitNetDType::I8 => {
                    assert_eq!(min_val, -128);
                    assert_eq!(max_val, 127);
                }
                BitNetDType::I4 => {
                    assert_eq!(min_val, -8);
                    assert_eq!(max_val, 7);
                }
                BitNetDType::I2 => {
                    assert_eq!(min_val, -2);
                    assert_eq!(max_val, 1);
                }
                BitNetDType::I1 => {
                    assert_eq!(min_val, -1);
                    assert_eq!(max_val, 0);
                }
                BitNetDType::BitNet158 => {
                    assert_eq!(min_val, -1);
                    assert_eq!(max_val, 1);
                }
                _ => {}
            }
        }
    }
}

#[test]
fn test_concurrent_access_and_thread_safety() {
    let device = get_cpu_device();
    let metadata = Arc::new(Mutex::new(create_test_metadata(
        1,
        vec![100, 100],
        BitNetDType::F32,
        &device,
        Some("thread_safety_test".to_string()),
    )));
    
    let thread_count = 4;
    let operations_per_thread = 100;
    let mut handles = Vec::new();
    
    // Test concurrent metadata operations
    for thread_id in 0..thread_count {
        let metadata_clone = metadata.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..operations_per_thread {
                let mut meta = metadata_clone.lock().unwrap();
                
                // Perform thread-safe operations
                meta.touch();
                meta.add_tag(format!("thread_{}_{}", thread_id, i));
                meta.add_ref();
                
                if i % 10 == 0 {
                    meta.set_migrating(i % 20 == 0);
                }
                
                if i % 5 == 0 {
                    meta.remove_ref();
                }
                
                // Verify consistency
                assert!(meta.ref_count() > 0);
                assert!(meta.has_tag(&format!("thread_{}_{}", thread_id, i)));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify final state consistency
    let final_meta = metadata.lock().unwrap();
    assert!(final_meta.ref_count() > 1);
    assert!(final_meta.tags.len() >= thread_count * operations_per_thread);
}

#[test]
fn test_comprehensive_integration_validation() {
    let devices = get_test_devices();
    let dtypes = get_all_dtypes();
    
    println!("Running comprehensive integration validation");
    
    let mut total_tests = 0;
    let mut successful_tests = 0;
    
    // Test all combinations of devices and data types
    for device in &devices {
        for &dtype in &dtypes {
            total_tests += 1;
            
            // Create metadata with various configurations
            let shapes = vec![vec![64], vec![32, 32], vec![8, 8, 8]];
            
            for shape in shapes {
                let metadata = create_test_metadata(
                    total_tests as u64,
                    shape.clone(),
                    dtype,
                    device,
                    Some(format!("integration_{}_{:?}", dtype, device)),
                );
                
                // Verify all properties are consistent
                assert_eq!(metadata.shape, shape);
                assert_eq!(metadata.dtype, dtype);
                
                let expected_elements: usize = shape.iter().product();
                assert_eq!(metadata.element_count, expected_elements);
                
                let expected_size = dtype.bytes_for_elements(expected_elements);
                assert_eq!(metadata.size_bytes, expected_size);
                
                let expected_efficiency = dtype.memory_efficiency();
                assert_relative_eq!(metadata.memory_efficiency(), expected_efficiency, epsilon = 0.001);
                
                // Test serialization
                let serialized = serde_json::to_string(&metadata)
                    .expect("Failed to serialize metadata");
                let deserialized: TensorMetadata = serde_json::from_str(&serialized)
                    .expect("Failed to deserialize metadata");
                
                assert_eq!(metadata.id, deserialized.id);
                assert_eq!(metadata.shape, deserialized.shape);
                assert_eq!(metadata.dtype, deserialized.dtype);
                
                successful_tests += 1;
            }
        }
    }
    
    println!("Integration validation completed: {}/{} tests passed",
             successful_tests, total_tests);
    assert_eq!(successful_tests, total_tests);
}