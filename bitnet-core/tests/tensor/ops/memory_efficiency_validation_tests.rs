//! Memory Efficiency Validation Tests
//!
//! This test suite validates that tensor operations use memory efficiently
//! and integrate properly with the BitNet memory management system.

use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::tensor::ops::arithmetic::{add, mul, add_scalar, sub};
use bitnet_core::memory::{HybridMemoryPool, MemoryStats};
use bitnet_core::device::get_cpu_device;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to get current memory stats
    fn get_memory_stats() -> MemoryStats {
        let pool = HybridMemoryPool::default();
        pool.get_stats()
    }

    /// Validate that memory usage is within acceptable bounds
    fn validate_memory_usage(
        test_name: &str,
        baseline: &MemoryStats,
        current: &MemoryStats,
        max_increase_bytes: usize,
        max_active_allocations: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let memory_increase = current.bytes_in_use - baseline.bytes_in_use;
        let active_allocations = current.active_allocations;

        println!("Memory validation for {}:", test_name);
        println!("  Memory increase: {} bytes", memory_increase);
        println!("  Active allocations: {}", active_allocations);
        println!("  Pool utilization: {:.2}%", 
                 (current.bytes_in_use as f64 / current.total_pool_size as f64) * 100.0);

        if memory_increase > max_increase_bytes {
            return Err(format!(
                "{}: Excessive memory increase {} > {} bytes",
                test_name, memory_increase, max_increase_bytes
            ).into());
        }

        if active_allocations > max_active_allocations {
            return Err(format!(
                "{}: Too many active allocations {} > {}",
                test_name, active_allocations, max_active_allocations
            ).into());
        }

        Ok(())
    }

    #[test]
    fn test_tensor_creation_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baseline = get_memory_stats();

        // Create multiple tensors and let them go out of scope
        {
            let _a = BitNetTensor::zeros(&[1000, 1000], BitNetDType::F32, Some(device.clone()))?;
            let _b = BitNetTensor::ones(&[1000, 1000], BitNetDType::F32, Some(device.clone()))?;
            let _c = BitNetTensor::zeros(&[500, 2000], BitNetDType::F32, Some(device.clone()))?;
            
            // Tensors should be using memory here
            let during_stats = get_memory_stats();
            assert!(during_stats.bytes_in_use > baseline.bytes_in_use);
            assert!(during_stats.active_allocations > baseline.active_allocations);
        } // Tensors should be deallocated here

        // Give time for cleanup to occur
        std::thread::sleep(Duration::from_millis(100));

        let final_stats = get_memory_stats();
        validate_memory_usage(
            "tensor_creation",
            &baseline,
            &final_stats,
            1024 * 1024, // Allow 1MB increase for pool growth
            5, // Allow 5 active allocations
        )?;

        Ok(())
    }

    #[test]
    fn test_arithmetic_operations_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baseline = get_memory_stats();

        let a = BitNetTensor::ones(&[1000, 1000], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[1000, 1000], BitNetDType::F32, Some(device.clone()))?;

        // Perform multiple operations
        {
            let _result1 = add(&a, &b)?;
            let _result2 = mul(&a, &b)?;
            let _result3 = sub(&a, &b)?;
            let _result4 = add_scalar(&a, 2.5)?;
            
            // Operations should create temporary results
            let during_stats = get_memory_stats();
            assert!(during_stats.active_allocations > baseline.active_allocations);
        } // Results should be deallocated here

        std::thread::sleep(Duration::from_millis(100));

        let final_stats = get_memory_stats();
        validate_memory_usage(
            "arithmetic_operations",
            &baseline,
            &final_stats,
            2 * 1024 * 1024, // Allow 2MB increase
            10, // Allow 10 active allocations (for input tensors + pool overhead)
        )?;

        Ok(())
    }

    #[test]
    fn test_large_tensor_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baseline = get_memory_stats();

        // Work with large tensors that should use the large block pool
        {
            let a = BitNetTensor::zeros(&[2000, 2000], BitNetDType::F32, Some(device.clone()))?; // ~16MB
            let b = BitNetTensor::ones(&[2000, 2000], BitNetDType::F32, Some(device.clone()))?;  // ~16MB
            
            let _result = add(&a, &b)?; // Another ~16MB
            
            // Should be using large block allocations
            let during_stats = get_memory_stats();
            println!("Large tensor operations - during execution:");
            println!("  Bytes in use: {} MB", during_stats.bytes_in_use / (1024 * 1024));
            println!("  Active allocations: {}", during_stats.active_allocations);
        } // Large tensors should be deallocated

        std::thread::sleep(Duration::from_millis(200)); // More time for large cleanup

        let final_stats = get_memory_stats();
        validate_memory_usage(
            "large_tensor_operations",
            &baseline,
            &final_stats,
            5 * 1024 * 1024, // Allow 5MB increase for pool growth
            5, // Allow some active allocations
        )?;

        Ok(())
    }

    #[test]
    fn test_memory_pool_reuse_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let pool = HybridMemoryPool::default();

        // Create many tensors of the same size to test pool reuse
        let tensor_size = [500, 500]; // ~1MB each
        let num_iterations = 100;

        let baseline_stats = pool.get_stats();
        
        for i in 0..num_iterations {
            let a = BitNetTensor::ones(&tensor_size, BitNetDType::F32, Some(device.clone()))?;
            let b = BitNetTensor::zeros(&tensor_size, BitNetDType::F32, Some(device.clone()))?;
            let _result = add(&a, &b)?;
            
            // Periodically check that memory isn't growing excessively
            if i % 20 == 19 {
                let current_stats = pool.get_stats();
                let memory_growth = current_stats.total_pool_size - baseline_stats.total_pool_size;
                
                // Pool should stabilize and reuse memory
                let max_expected_growth = 20 * 1024 * 1024; // 20MB max growth
                assert!(memory_growth <= max_expected_growth,
                        "Excessive pool growth at iteration {}: {} > {} bytes",
                        i, memory_growth, max_expected_growth);
            }
        }

        let final_stats = pool.get_stats();
        println!("Memory pool reuse efficiency:");
        println!("  Initial pool size: {} MB", baseline_stats.total_pool_size / (1024 * 1024));
        println!("  Final pool size: {} MB", final_stats.total_pool_size / (1024 * 1024));
        println!("  Pool growth: {} MB", (final_stats.total_pool_size - baseline_stats.total_pool_size) / (1024 * 1024));
        println!("  Final active allocations: {}", final_stats.active_allocations);

        // Verify efficient pool reuse
        let pool_growth_ratio = (final_stats.total_pool_size - baseline_stats.total_pool_size) as f64 
                              / baseline_stats.total_pool_size as f64;
        
        assert!(pool_growth_ratio < 2.0, // Pool shouldn't more than double
                "Pool grew too much: {:.2}x", pool_growth_ratio + 1.0);

        Ok(())
    }

    #[test]
    fn test_broadcasting_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baseline = get_memory_stats();

        // Test broadcasting with different tensor sizes
        let large = BitNetTensor::ones(&[1000, 1000], BitNetDType::F32, Some(device.clone()))?;
        let small = BitNetTensor::ones(&[1000, 1], BitNetDType::F32, Some(device.clone()))?;
        let scalar_tensor = BitNetTensor::ones(&[1], BitNetDType::F32, Some(device.clone()))?;

        // Broadcasting operations should be memory efficient
        {
            let _result1 = add(&large, &small)?; // Should broadcast small to large shape
            let _result2 = add(&large, &scalar_tensor)?; // Should broadcast scalar
            let _result3 = mul(&small, &scalar_tensor)?; // Small broadcasting
        }

        std::thread::sleep(Duration::from_millis(100));

        let final_stats = get_memory_stats();
        validate_memory_usage(
            "broadcasting_operations",
            &baseline,
            &final_stats,
            8 * 1024 * 1024, // Allow 8MB for input tensors and results
            15, // Allow more allocations for broadcasting
        )?;

        Ok(())
    }

    #[test]
    fn test_complex_expression_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let baseline = get_memory_stats();

        // Test a complex expression: ((A + B) * C - D) / 2.0
        let a = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;
        let c = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;
        let d = BitNetTensor::ones(&[500, 500], BitNetDType::F32, Some(device.clone()))?;

        {
            let step1 = add(&a, &b)?;           // A + B
            let step2 = mul(&step1, &c)?;       // (A + B) * C
            let step3 = sub(&step2, &d)?;       // (A + B) * C - D
            let _final = add_scalar(&step3, 0.5)?; // ((A + B) * C - D) + 0.5 (approximating /2)
            
            // Complex expression should create intermediate results
            let during_stats = get_memory_stats();
            println!("Complex expression memory usage:");
            println!("  Active allocations: {}", during_stats.active_allocations);
            println!("  Bytes in use: {} MB", during_stats.bytes_in_use / (1024 * 1024));
        } // All intermediate results should be cleaned up

        std::thread::sleep(Duration::from_millis(100));

        let final_stats = get_memory_stats();
        validate_memory_usage(
            "complex_expression",
            &baseline,
            &final_stats,
            4 * 1024 * 1024, // Allow 4MB for the four input tensors
            15, // Allow allocations for input tensors plus some overhead
        )?;

        Ok(())
    }

    #[test]
    fn test_memory_fragmentation() -> Result<(), Box<dyn std::error::Error>> {
        let device = get_cpu_device();
        let pool = HybridMemoryPool::default();

        // Create tensors of varying sizes to test fragmentation handling
        let sizes = vec![
            vec![100, 100],    // Small
            vec![1000, 1000],  // Large
            vec![50, 2000],    // Medium, different aspect ratio
            vec![2000, 50],    // Medium, different aspect ratio
            vec![10, 10000],   // Thin
        ];

        let baseline_stats = pool.get_stats();
        
        // Create and destroy tensors in a pattern that could cause fragmentation
        for _ in 0..20 {
            let mut tensors = Vec::new();
            
            // Create tensors of different sizes
            for (i, size) in sizes.iter().enumerate() {
                if i % 2 == 0 { // Create only even-indexed tensors first
                    let tensor = BitNetTensor::zeros(size, BitNetDType::F32, Some(device.clone()))?;
                    tensors.push(tensor);
                }
            }
            
            // Then create odd-indexed tensors
            for (i, size) in sizes.iter().enumerate() {
                if i % 2 == 1 {
                    let tensor = BitNetTensor::zeros(size, BitNetDType::F32, Some(device.clone()))?;
                    tensors.push(tensor);
                }
            }
            
            // Drop every other tensor to create holes
            let mut new_tensors = Vec::new();
            for (i, tensor) in tensors.into_iter().enumerate() {
                if i % 3 != 0 { // Keep 2/3 of tensors
                    new_tensors.push(tensor);
                }
            }
            
            // Create some operations to exercise the fragmented memory
            if new_tensors.len() >= 2 {
                let _result = add(&new_tensors[0], &new_tensors[1])?;
            }
        } // All tensors go out of scope

        std::thread::sleep(Duration::from_millis(200)); // Allow cleanup and compaction

        let final_stats = pool.get_stats();
        
        // Check fragmentation levels
        let fragmentation_bytes = final_stats.fragmented_bytes.unwrap_or(0);
        let fragmentation_ratio = fragmentation_bytes as f64 / final_stats.total_pool_size as f64;
        
        println!("Memory fragmentation test results:");
        println!("  Fragmented bytes: {} KB", fragmentation_bytes / 1024);
        println!("  Fragmentation ratio: {:.2}%", fragmentation_ratio * 100.0);
        println!("  Active allocations: {}", final_stats.active_allocations);

        // Fragmentation should be kept under control
        assert!(fragmentation_ratio < 0.3, // Less than 30% fragmentation
                "Excessive fragmentation: {:.2}%", fragmentation_ratio * 100.0);

        Ok(())
    }

    #[test]
    fn test_concurrent_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::Arc;
        use std::thread;

        let device = get_cpu_device();
        let pool = Arc::new(HybridMemoryPool::default());
        let baseline_stats = pool.get_stats();

        let num_threads = 4;
        let iterations_per_thread = 50;

        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let device_clone = device.clone();
            let pool_clone = Arc::clone(&pool);
            
            thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                for i in 0..iterations_per_thread {
                    let size = 100 + (thread_id * 100) + (i * 10); // Varying sizes per thread
                    let a = BitNetTensor::ones(&[size, size], BitNetDType::F32, Some(device_clone.clone()))?;
                    let b = BitNetTensor::zeros(&[size, size], BitNetDType::F32, Some(device_clone.clone()))?;
                    let _result = add(&a, &b)?;
                    
                    // Periodic memory check
                    if i % 10 == 9 {
                        let stats = pool_clone.get_stats();
                        // Ensure memory usage isn't growing uncontrollably
                        let memory_growth = stats.bytes_in_use;
                        assert!(memory_growth < 100 * 1024 * 1024, // Less than 100MB total usage
                                "Thread {}: Excessive memory usage: {} MB", 
                                thread_id, memory_growth / (1024 * 1024));
                    }
                }
                Ok(())
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        std::thread::sleep(Duration::from_millis(200)); // Allow cleanup

        let final_stats = pool.get_stats();
        
        println!("Concurrent memory efficiency test:");
        println!("  Final active allocations: {}", final_stats.active_allocations);
        println!("  Final bytes in use: {} MB", final_stats.bytes_in_use / (1024 * 1024));
        println!("  Pool size growth: {} MB", 
                 (final_stats.total_pool_size - baseline_stats.total_pool_size) / (1024 * 1024));

        // Concurrent operations should not cause excessive memory usage
        assert!(final_stats.active_allocations < 50,
                "Too many active allocations after concurrent test: {}", 
                final_stats.active_allocations);

        let memory_efficiency = final_stats.bytes_in_use as f64 / final_stats.total_pool_size as f64;
        assert!(memory_efficiency > 0.1, // At least 10% pool utilization
                "Very low memory efficiency: {:.2}%", memory_efficiency * 100.0);

        Ok(())
    }
}
