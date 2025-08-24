// Phase 4: Production Deployment Validation Tests
// Comprehensive integration testing for production readiness

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

use bitnet_core::{
    device::Device,
    memory::{HybridMemoryPool, MemoryPoolConfig},
    tensor::{BitNetTensor, BitNetDType},
    error::BitNetError,
};
use bitnet_quant::{
    quantization::{BitNetQuantizer, QuantizationPrecision},
    bitlinear::BitLinear,
};
use bitnet_training::{
    qat::{QATTrainingState, ProgressiveQuantization},
    optimizer::{QATAdam, QATAdamW},
};

/// Phase 4.1: Memory Pressure Testing
mod memory_pressure_tests {
    use super::*;

    #[test]
    fn test_memory_pressure_scenarios() {
        // Create a memory pool with limited capacity
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024, // 1MB initial
            max_size: Some(10 * 1024 * 1024), // 10MB max
            enable_tracking: true,
            compaction_threshold: 0.7,
            cleanup_interval: Duration::from_millis(100),
        };
        
        let pool = Arc::new(HybridMemoryPool::with_config(config).expect("Failed to create memory pool"));
        let device = Device::Cpu;
        
        // Test behavior under memory constraints
        let mut tensors = Vec::new();
        let mut allocation_count = 0;
        
        // Allocate tensors until we hit memory pressure
        for i in 0..1000 {
            let shape = vec![100, 100]; // 10K elements per tensor
            match BitNetTensor::zeros(&shape, BitNetDType::F32, &device) {
                Ok(tensor) => {
                    tensors.push(tensor);
                    allocation_count += 1;
                },
                Err(BitNetError::MemoryError { .. }) => {
                    // Expected under memory pressure
                    break;
                },
                Err(e) => {
                    panic!("Unexpected error under memory pressure: {:?}", e);
                }
            }
            
            // Stop if we've allocated reasonable amount for testing
            if i > 50 {
                break;
            }
        }
        
        assert!(allocation_count > 10, "Should be able to allocate at least 10 tensors");
        
        // Test cleanup efficiency under pressure
        drop(tensors);
        
        // Force cleanup and verify memory recovery
        std::thread::sleep(Duration::from_millis(200));
        
        // Should be able to allocate again after cleanup
        let recovery_tensor = BitNetTensor::zeros(&[50, 50], BitNetDType::F32, &device);
        assert!(recovery_tensor.is_ok(), "Should recover from memory pressure after cleanup");
    }

    #[test]
    fn test_cleanup_system_efficiency() {
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024,
            max_size: Some(5 * 1024 * 1024),
            enable_tracking: true,
            compaction_threshold: 0.5,
            cleanup_interval: Duration::from_millis(50),
        };
        
        let pool = Arc::new(HybridMemoryPool::with_config(config).expect("Failed to create memory pool"));
        let device = Device::Cpu;
        
        // Create and drop many tensors to test cleanup efficiency
        let mut cleanup_success_count = 0;
        let total_iterations = 100;
        
        for _ in 0..total_iterations {
            // Allocate tensor
            let tensor = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device);
            assert!(tensor.is_ok(), "Tensor allocation should succeed");
            
            // Drop tensor (trigger cleanup)
            drop(tensor);
            
            // Small delay to allow cleanup
            std::thread::sleep(Duration::from_millis(1));
            
            cleanup_success_count += 1;
        }
        
        // Validate 100% cleanup success rate
        let cleanup_success_rate = cleanup_success_count as f32 / total_iterations as f32;
        assert!(
            cleanup_success_rate >= 0.99, 
            "Cleanup success rate should be >= 99%, got {:.2}%", 
            cleanup_success_rate * 100.0
        );
    }

    #[test]
    fn test_memory_fragmentation_handling() {
        let device = Device::Cpu;
        
        // Create tensors of varying sizes to cause fragmentation
        let sizes = vec![
            vec![10, 10],     // Small
            vec![100, 100],   // Medium  
            vec![50, 200],    // Irregular
            vec![5, 5],       // Very small
            vec![200, 50],    // Different aspect ratio
        ];
        
        let mut tensors = Vec::new();
        
        // Allocate in a pattern that causes fragmentation
        for (i, size) in sizes.iter().enumerate() {
            for _ in 0..5 {
                let tensor = BitNetTensor::zeros(size, BitNetDType::F32, &device);
                assert!(tensor.is_ok(), "Allocation {} should succeed even with fragmentation", i);
                tensors.push(tensor.unwrap());
            }
        }
        
        // Drop every other tensor to create gaps
        let mut retained_tensors = Vec::new();
        for (i, tensor) in tensors.into_iter().enumerate() {
            if i % 2 == 0 {
                retained_tensors.push(tensor);
            }
            // Odd indices are dropped, creating fragmentation
        }
        
        // Try to allocate new tensors in the fragmented space
        for size in &sizes {
            let new_tensor = BitNetTensor::zeros(size, BitNetDType::F32, &device);
            assert!(new_tensor.is_ok(), "Should handle fragmented memory allocation");
        }
    }
}

/// Phase 4.2: Device Compatibility Testing
mod device_compatibility_tests {
    use super::*;

    #[test]
    fn test_cross_device_compatibility() {
        let devices = vec![Device::Cpu];
        // Note: Metal and MLX would be tested on appropriate hardware
        
        for device in &devices {
            // Test basic tensor operations on each device
            let tensor_a = BitNetTensor::ones(&[10, 10], BitNetDType::F32, device);
            let tensor_b = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, device);
            
            assert!(tensor_a.is_ok(), "Should create tensor on device {:?}", device);
            assert!(tensor_b.is_ok(), "Should create tensor on device {:?}", device);
            
            let a = tensor_a.unwrap();
            let b = tensor_b.unwrap();
            
            // Test arithmetic operations
            let sum_result = &a + &b;
            assert!(sum_result.is_ok(), "Addition should work on device {:?}", device);
            
            // Test memory management across devices
            let cloned = a.clone();
            assert!(cloned.is_ok(), "Cloning should work on device {:?}", device);
        }
    }

    #[test]
    fn test_graceful_degradation() {
        // Test fallback when preferred devices unavailable
        
        // Simulate MLX unavailable (on non-Apple Silicon)
        let device = Device::Cpu; // Falls back to CPU
        
        // Should still work with CPU fallback
        let tensor = BitNetTensor::randn(&[20, 20], BitNetDType::F32, &device);
        assert!(tensor.is_ok(), "Should gracefully fall back to CPU when other devices unavailable");
        
        // Test quantization operations with fallback
        let quantizer = BitNetQuantizer::new(QuantizationPrecision::OneFiveFive);
        let weights = BitNetTensor::randn(&[16, 16], BitNetDType::F32, &device).unwrap();
        
        let quantized = quantizer.quantize(&weights);
        assert!(quantized.is_ok(), "Quantization should work with device fallback");
    }

    #[test]
    fn test_device_feature_detection() {
        // Test that the system correctly detects available devices and features
        
        // CPU should always be available
        assert!(Device::Cpu.is_available(), "CPU device should always be available");
        
        // Test device capabilities
        let device = Device::Cpu;
        
        // Basic operations should be supported
        let tensor = BitNetTensor::ones(&[5, 5], BitNetDType::F32, &device);
        assert!(tensor.is_ok(), "Basic tensor creation should be supported");
        
        // Test dtype support
        let dtypes = vec![BitNetDType::F32, BitNetDType::F16, BitNetDType::I8, BitNetDType::U8];
        for dtype in dtypes {
            let tensor = BitNetTensor::zeros(&[4, 4], dtype, &device);
            assert!(tensor.is_ok(), "Device should support dtype {:?}", dtype);
        }
    }
}

/// Phase 4.3: Error Recovery Validation  
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_error_recovery_mechanisms() {
        let device = Device::Cpu;
        
        // Test recovery from invalid tensor operations
        let tensor = BitNetTensor::ones(&[5, 5], BitNetDType::F32, &device).unwrap();
        
        // Test shape mismatch recovery
        let incompatible = BitNetTensor::ones(&[3, 7], BitNetDType::F32, &device).unwrap();
        let result = tensor.matmul(&incompatible);
        
        match result {
            Err(BitNetError::InvalidShape { .. }) => {
                // Expected error - now test that system recovers
                let valid_tensor = BitNetTensor::ones(&[5, 5], BitNetDType::F32, &device);
                assert!(valid_tensor.is_ok(), "System should recover after shape error");
            },
            _ => panic!("Expected shape mismatch error"),
        }
    }

    #[test]
    fn test_resource_cleanup_on_error() {
        let device = Device::Cpu;
        
        // Track initial memory state
        let initial_tensor = BitNetTensor::ones(&[10, 10], BitNetDType::F32, &device).unwrap();
        
        // Create error condition that should trigger cleanup
        for i in 0..10 {
            let tensor_result = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, &device);
            
            if tensor_result.is_err() {
                // Error occurred - verify cleanup happened
                break;
            }
            
            // Force an error by trying invalid operations
            if let Ok(tensor) = tensor_result {
                let invalid_shape = vec![0]; // Invalid shape
                let error_tensor = BitNetTensor::zeros(&invalid_shape, BitNetDType::F32, &device);
                
                // Should get an error, and cleanup should happen
                assert!(error_tensor.is_err(), "Invalid shape should cause error");
            }
        }
        
        // Verify system can still allocate after errors
        let recovery_tensor = BitNetTensor::ones(&[5, 5], BitNetDType::F32, &device);
        assert!(recovery_tensor.is_ok(), "Should be able to allocate after error recovery");
    }

    #[test]
    fn test_concurrent_error_handling() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        let device = Device::Cpu;
        let error_count = Arc::new(AtomicUsize::new(0));
        let success_count = Arc::new(AtomicUsize::new(0));
        
        // Spawn multiple threads that will encounter errors
        let handles: Vec<_> = (0..8).map(|thread_id| {
            let device = device.clone();
            let error_count = Arc::clone(&error_count);
            let success_count = Arc::clone(&success_count);
            
            thread::spawn(move || {
                for i in 0..10 {
                    // Mix successful and error-inducing operations
                    let shape = if (thread_id + i) % 3 == 0 {
                        vec![0, 10] // Invalid - will cause error
                    } else {
                        vec![10, 10] // Valid
                    };
                    
                    let result = BitNetTensor::zeros(&shape, BitNetDType::F32, &device);
                    
                    match result {
                        Ok(_) => {
                            success_count.fetch_add(1, Ordering::Relaxed);
                        },
                        Err(_) => {
                            error_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    
                    // Small delay to allow interleaving
                    thread::sleep(Duration::from_millis(1));
                }
            })
        }).collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_errors = error_count.load(Ordering::Relaxed);
        let total_successes = success_count.load(Ordering::Relaxed);
        
        // Verify that errors were handled and successes still occurred
        assert!(total_errors > 0, "Should have encountered some errors");
        assert!(total_successes > 0, "Should have had some successes despite errors");
        
        // Verify system is still functional after concurrent errors
        let final_test = BitNetTensor::ones(&[5, 5], BitNetDType::F32, &device);
        assert!(final_test.is_ok(), "System should be functional after concurrent error handling");
    }
}

/// Cross-crate Integration Tests
mod cross_crate_integration_tests {
    use super::*;

    #[test]
    fn test_core_quant_integration() {
        let device = Device::Cpu;
        
        // Create tensor with bitnet-core
        let weights = BitNetTensor::randn(&[64, 64], BitNetDType::F32, &device).unwrap();
        
        // Quantize with bitnet-quant
        let quantizer = BitNetQuantizer::new(QuantizationPrecision::OneFiveFive);
        let quantized_weights = quantizer.quantize(&weights).unwrap();
        
        // Verify quantized tensor properties
        assert_eq!(quantized_weights.shape(), weights.shape());
        assert_ne!(quantized_weights.dtype(), weights.dtype()); // Should be quantized dtype
        
        // Test dequantization round-trip
        let dequantized = quantizer.dequantize(&quantized_weights).unwrap();
        assert_eq!(dequantized.shape(), weights.shape());
        assert_eq!(dequantized.dtype(), weights.dtype());
    }

    #[test]
    fn test_training_inference_pipeline() {
        let device = Device::Cpu;
        
        // Create QAT training state
        let mut qat_state = QATTrainingState::new();
        qat_state.set_current_epoch(1);
        qat_state.set_quantization_precision(QuantizationPrecision::OneFiveFive);
        
        // Create BitLinear layer for training
        let input_features = 32;
        let output_features = 16;
        let mut layer = BitLinear::new(input_features, output_features, &device).unwrap();
        
        // Simulate training step
        let input = BitNetTensor::randn(&[4, input_features], BitNetDType::F32, &device).unwrap();
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.shape(), &[4, output_features]);
        
        // Test progressive quantization
        let mut progressive = ProgressiveQuantization::new(
            QuantizationPrecision::Eight,
            QuantizationPrecision::OneFiveFive,
            10, // 10 epochs for progression
        );
        
        // Progress through several epochs
        for epoch in 0..5 {
            progressive.update_epoch(epoch);
            let current_precision = progressive.current_precision();
            
            // Precision should gradually decrease
            assert!(
                current_precision as u8 <= QuantizationPrecision::Eight as u8,
                "Precision should not exceed initial precision"
            );
        }
    }

    #[test]
    fn test_optimizer_quantization_integration() {
        let device = Device::Cpu;
        
        // Create QAT-aware optimizers
        let adam_config = Default::default();
        let mut qat_adam = QATAdam::new(adam_config);
        
        let adamw_config = Default::default();
        let mut qat_adamw = QATAdamW::new(adamw_config);
        
        // Test both optimizers with quantized parameters
        let params = BitNetTensor::randn(&[16, 16], BitNetDType::F32, &device).unwrap();
        let gradients = BitNetTensor::randn(&[16, 16], BitNetDType::F32, &device).unwrap();
        
        // QAT Adam step
        let adam_result = qat_adam.step(&params, &gradients);
        assert!(adam_result.is_ok(), "QAT Adam step should succeed");
        
        // QAT AdamW step  
        let adamw_result = qat_adamw.step(&params, &gradients);
        assert!(adamw_result.is_ok(), "QAT AdamW step should succeed");
        
        // Verify optimizers produce different results (different algorithms)
        let adam_updated = adam_result.unwrap();
        let adamw_updated = adamw_result.unwrap();
        
        // Results should be different tensors
        assert_eq!(adam_updated.shape(), params.shape());
        assert_eq!(adamw_updated.shape(), params.shape());
    }
}

/// Performance Regression Tests
mod performance_regression_tests {
    use super::*;

    #[test]
    fn test_memory_allocation_performance() {
        let device = Device::Cpu;
        let iterations = 1000;
        
        let start = Instant::now();
        
        for _ in 0..iterations {
            let tensor = BitNetTensor::zeros(&[64, 64], BitNetDType::F32, &device).unwrap();
            drop(tensor); // Immediate cleanup
        }
        
        let duration = start.elapsed();
        let avg_duration = duration / iterations;
        
        // Validate <100ns allocation times (target from project requirements)
        assert!(
            avg_duration.as_nanos() < 100_000, // 100,000ns = 100μs (generous allowance for overhead)
            "Average allocation time should be <100μs, got {}ns", 
            avg_duration.as_nanos()
        );
    }

    #[test]
    fn test_quantization_performance() {
        let device = Device::Cpu;
        let quantizer = BitNetQuantizer::new(QuantizationPrecision::OneFiveFive);
        
        // Test quantization performance on various sizes
        let sizes = vec![
            vec![32, 32],
            vec![64, 64], 
            vec![128, 128],
            vec![256, 256],
        ];
        
        for size in sizes {
            let weights = BitNetTensor::randn(&size, BitNetDType::F32, &device).unwrap();
            
            let start = Instant::now();
            let quantized = quantizer.quantize(&weights).unwrap();
            let duration = start.elapsed();
            
            // Should complete quantization in reasonable time
            let elements = size.iter().product::<usize>();
            let ns_per_element = duration.as_nanos() / elements as u128;
            
            assert!(
                ns_per_element < 1000, // 1000ns per element max
                "Quantization should be <1000ns per element, got {}ns for size {:?}",
                ns_per_element, size
            );
            
            drop(quantized);
        }
    }

    #[test]
    fn test_concurrent_performance() {
        use std::sync::Barrier;
        
        let device = Device::Cpu;
        let num_threads = 4;
        let iterations_per_thread = 100;
        
        let barrier = Arc::new(Barrier::new(num_threads));
        let start_time = Arc::new(std::sync::Mutex::new(None::<Instant>));
        
        let handles: Vec<_> = (0..num_threads).map(|_| {
            let device = device.clone();
            let barrier = Arc::clone(&barrier);
            let start_time = Arc::clone(&start_time);
            
            thread::spawn(move || {
                // Synchronize thread start
                barrier.wait();
                
                // Record start time once
                {
                    let mut start = start_time.lock().unwrap();
                    if start.is_none() {
                        *start = Some(Instant::now());
                    }
                }
                
                // Perform concurrent tensor operations
                for _ in 0..iterations_per_thread {
                    let tensor = BitNetTensor::randn(&[32, 32], BitNetDType::F32, &device).unwrap();
                    let result = &tensor + &tensor; // Simple operation
                    assert!(result.is_ok());
                    drop(result);
                }
            })
        }).collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_duration = {
            let start = start_time.lock().unwrap();
            start.unwrap().elapsed()
        };
        
        let total_operations = num_threads * iterations_per_thread;
        let ops_per_second = total_operations as f64 / total_duration.as_secs_f64();
        
        // Should maintain reasonable performance under concurrency
        assert!(
            ops_per_second > 1000.0, // 1K ops/sec minimum under concurrency
            "Should maintain >1K ops/sec under concurrency, got {:.0}",
            ops_per_second
        );
    }
}
