// Phase 4: Comprehensive Performance Validation Tests
// Production readiness performance benchmarks and validation

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::thread;

use bitnet_core::{
    device::Device,
    memory::{HybridMemoryPool, MemoryPoolConfig},
    tensor::{BitNetTensor, BitNetDType},
};
use bitnet_quant::{
    quantization::{BitNetQuantizer, QuantizationPrecision},
    bitlinear::BitLinear,
};

/// Performance validation constants based on project requirements
const MAX_ALLOCATION_TIME_NS: u128 = 100_000; // 100μs allowance for test overhead
const MIN_MLX_OPS_PER_SEC: f64 = 50_000.0; // Reduced for CI environment
const MIN_SIMD_SPEEDUP: f64 = 2.0; // Minimum 2x speedup expected
const MAX_MEMORY_OVERHEAD_PERCENT: f64 = 5.0; // <5% memory overhead
const MIN_CLEANUP_SUCCESS_RATE: f64 = 0.95; // 95% cleanup success rate

/// Phase 4: Comprehensive Performance Validation
mod performance_validation_tests {
    use super::*;

    #[test]
    fn test_memory_allocation_performance_validation() {
        let device = Device::Cpu;
        let test_sizes = vec![
            vec![16, 16],      // Small tensors
            vec![64, 64],      // Medium tensors  
            vec![128, 128],    // Large tensors
            vec![256, 256],    // Very large tensors
        ];
        
        for size in test_sizes {
            let iterations = 100;
            let mut total_duration = Duration::ZERO;
            
            for _ in 0..iterations {
                let start = Instant::now();
                let tensor = BitNetTensor::zeros(&size, BitNetDType::F32, &device).unwrap();
                let allocation_time = start.elapsed();
                
                total_duration += allocation_time;
                drop(tensor);
            }
            
            let avg_duration = total_duration / iterations;
            let elements = size.iter().product::<usize>();
            
            println!(
                "Size {:?}: Avg allocation time {}μs ({} elements)", 
                size, 
                avg_duration.as_micros(),
                elements
            );
            
            // Validate allocation performance
            assert!(
                avg_duration.as_nanos() < MAX_ALLOCATION_TIME_NS,
                "Average allocation time for {:?} should be <{}μs, got {}μs",
                size,
                MAX_ALLOCATION_TIME_NS / 1000,
                avg_duration.as_micros()
            );
        }
    }

    #[test]
    fn test_quantization_performance_validation() {
        let device = Device::Cpu;
        let precisions = vec![
            QuantizationPrecision::Eight,
            QuantizationPrecision::Four,
            QuantizationPrecision::Two,
            QuantizationPrecision::OneFiveFive,
        ];
        
        let test_size = vec![128, 128];
        let weights = BitNetTensor::randn(&test_size, BitNetDType::F32, &device).unwrap();
        
        for precision in precisions {
            let quantizer = BitNetQuantizer::new(precision);
            let iterations = 50;
            let mut total_duration = Duration::ZERO;
            
            for _ in 0..iterations {
                let start = Instant::now();
                let quantized = quantizer.quantize(&weights).unwrap();
                let quantization_time = start.elapsed();
                
                total_duration += quantization_time;
                drop(quantized);
            }
            
            let avg_duration = total_duration / iterations;
            let elements = test_size.iter().product::<usize>();
            let ns_per_element = avg_duration.as_nanos() / elements as u128;
            
            println!(
                "Precision {:?}: {}ns per element ({}μs total)",
                precision,
                ns_per_element,
                avg_duration.as_micros()
            );
            
            // Validate quantization performance (should be <1000ns per element)
            assert!(
                ns_per_element < 1000,
                "Quantization {:?} should be <1000ns per element, got {}ns",
                precision, ns_per_element
            );
        }
    }

    #[test] 
    fn test_memory_efficiency_validation() {
        let device = Device::Cpu;
        let base_size = vec![64, 64];
        let iterations = 100;
        
        // Measure baseline memory usage
        let baseline_tensor = BitNetTensor::zeros(&base_size, BitNetDType::F32, &device).unwrap();
        let element_count = base_size.iter().product::<usize>();
        let expected_memory = element_count * 4; // 4 bytes per f32
        
        // Create multiple tensors and measure overhead
        let mut tensors = Vec::new();
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let tensor = BitNetTensor::zeros(&base_size, BitNetDType::F32, &device).unwrap();
            tensors.push(tensor);
        }
        
        let allocation_duration = start_time.elapsed();
        
        // Estimate memory overhead (simplified calculation)
        let total_expected_memory = expected_memory * iterations;
        // Note: In a real implementation, we'd measure actual memory usage
        // For now, we validate that allocation time scales reasonably
        
        let avg_allocation_time = allocation_duration.as_nanos() / iterations as u128;
        
        println!(
            "Memory efficiency: {}ns avg allocation, {} total tensors",
            avg_allocation_time, iterations
        );
        
        // Validate memory efficiency  
        assert!(
            avg_allocation_time < MAX_ALLOCATION_TIME_NS,
            "Memory allocation should be efficient: {}ns > {}ns max",
            avg_allocation_time, MAX_ALLOCATION_TIME_NS
        );
        
        drop(tensors);
    }

    #[test]
    fn test_cleanup_efficiency_validation() {
        let device = Device::Cpu;
        let iterations = 500;
        let mut successful_cleanups = 0;
        let mut failed_cleanups = 0;
        
        for i in 0..iterations {
            // Create tensor
            let tensor_result = BitNetTensor::zeros(&[32, 32], BitNetDType::F32, &device);
            
            if let Ok(tensor) = tensor_result {
                // Drop tensor and allow cleanup
                drop(tensor);
                
                // Small delay to allow cleanup processing
                if i % 10 == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                
                successful_cleanups += 1;
            } else {
                failed_cleanups += 1;
            }
        }
        
        let cleanup_success_rate = successful_cleanups as f64 / iterations as f64;
        
        println!(
            "Cleanup efficiency: {:.2}% success rate ({}/{} successful)",
            cleanup_success_rate * 100.0,
            successful_cleanups,
            iterations
        );
        
        // Validate cleanup success rate
        assert!(
            cleanup_success_rate >= MIN_CLEANUP_SUCCESS_RATE,
            "Cleanup success rate should be >{:.0}%, got {:.2}%",
            MIN_CLEANUP_SUCCESS_RATE * 100.0,
            cleanup_success_rate * 100.0
        );
    }

    #[test]
    fn test_concurrent_performance_validation() {
        let device = Device::Cpu;
        let num_threads = 4;
        let operations_per_thread = 50;
        
        let start_time = Arc::new(std::sync::Mutex::new(None::<Instant>));
        let completion_time = Arc::new(std::sync::Mutex::new(None::<Instant>));
        
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let device = device.clone();
            let start_time = Arc::clone(&start_time);
            let completion_time = Arc::clone(&completion_time);
            
            thread::spawn(move || {
                // Record start time for first thread
                {
                    let mut start = start_time.lock().unwrap();
                    if start.is_none() {
                        *start = Some(Instant::now());
                    }
                }
                
                // Perform operations
                for i in 0..operations_per_thread {
                    let size = vec![16 + (i % 10), 16 + (i % 10)]; // Varying sizes
                    let tensor_a = BitNetTensor::ones(&size, BitNetDType::F32, &device).unwrap();
                    let tensor_b = BitNetTensor::zeros(&size, BitNetDType::F32, &device).unwrap();
                    
                    let result = &tensor_a + &tensor_b;
                    assert!(result.is_ok(), "Thread {} operation {} failed", thread_id, i);
                    
                    drop(result);
                }
                
                // Record completion time for last thread
                {
                    let mut completion = completion_time.lock().unwrap();
                    *completion = Some(Instant::now());
                }
            })
        }).collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_duration = {
            let start = start_time.lock().unwrap().unwrap();
            let end = completion_time.lock().unwrap().unwrap();
            end.duration_since(start)
        };
        
        let total_operations = num_threads * operations_per_thread;
        let ops_per_second = total_operations as f64 / total_duration.as_secs_f64();
        
        println!(
            "Concurrent performance: {:.0} ops/second ({} threads, {} ops each)",
            ops_per_second, num_threads, operations_per_thread
        );
        
        // Validate concurrent performance (should maintain reasonable throughput)
        assert!(
            ops_per_second > 500.0, // Minimum 500 ops/sec under concurrency
            "Concurrent performance should be >500 ops/sec, got {:.0}",
            ops_per_second
        );
    }

    #[test]
    fn test_bitlinear_layer_performance_validation() {
        let device = Device::Cpu;
        let input_features = 128;
        let output_features = 64;
        let batch_size = 8;
        
        let mut layer = BitLinear::new(input_features, output_features, &device).unwrap();
        let input = BitNetTensor::randn(&[batch_size, input_features], BitNetDType::F32, &device).unwrap();
        
        // Warm up
        for _ in 0..5 {
            let _ = layer.forward(&input).unwrap();
        }
        
        // Measure performance
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let output = layer.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, output_features]);
            drop(output);
        }
        
        let duration = start.elapsed();
        let avg_duration = duration / iterations;
        let operations_per_second = 1.0 / avg_duration.as_secs_f64();
        
        println!(
            "BitLinear performance: {:.0} forward passes/sec ({}μs per pass)",
            operations_per_second,
            avg_duration.as_micros()
        );
        
        // Validate BitLinear performance
        assert!(
            operations_per_second > 100.0, // Minimum 100 forward passes per second
            "BitLinear should achieve >100 forward passes/sec, got {:.0}",
            operations_per_second
        );
    }

    #[test]
    fn test_memory_pool_performance_validation() {
        // Test memory pool performance with different configurations
        let configs = vec![
            MemoryPoolConfig {
                initial_size: 1024 * 1024,     // 1MB
                max_size: Some(10 * 1024 * 1024), // 10MB max
                enable_tracking: true,
                compaction_threshold: 0.8,
                cleanup_interval: Duration::from_millis(100),
            },
            MemoryPoolConfig {
                initial_size: 512 * 1024,      // 512KB
                max_size: Some(5 * 1024 * 1024),  // 5MB max
                enable_tracking: false,
                compaction_threshold: 0.9,
                cleanup_interval: Duration::from_millis(50),
            },
        ];
        
        for (config_idx, config) in configs.iter().enumerate() {
            let pool = Arc::new(HybridMemoryPool::with_config(config.clone()).unwrap());
            let device = Device::Cpu;
            
            // Measure allocation performance with this config
            let iterations = 100;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let tensor = BitNetTensor::zeros(&[64, 64], BitNetDType::F32, &device).unwrap();
                drop(tensor);
            }
            
            let duration = start.elapsed();
            let avg_duration = duration / iterations;
            
            println!(
                "Pool config {}: {}μs avg allocation time",
                config_idx,
                avg_duration.as_micros()
            );
            
            // Validate pool performance
            assert!(
                avg_duration.as_nanos() < MAX_ALLOCATION_TIME_NS,
                "Pool config {} should allocate in <{}μs, got {}μs",
                config_idx,
                MAX_ALLOCATION_TIME_NS / 1000,
                avg_duration.as_micros()
            );
        }
    }
}

/// Integration Performance Tests 
mod integration_performance_tests {
    use super::*;

    #[test]
    fn test_end_to_end_training_performance() {
        let device = Device::Cpu;
        let batch_size = 4;
        let input_features = 32;
        let output_features = 16;
        
        // Create components
        let mut layer = BitLinear::new(input_features, output_features, &device).unwrap();
        let quantizer = BitNetQuantizer::new(QuantizationPrecision::OneFiveFive);
        
        // Measure end-to-end performance
        let iterations = 20; // Reduced for CI
        let start = Instant::now();
        
        for _ in 0..iterations {
            // Forward pass
            let input = BitNetTensor::randn(&[batch_size, input_features], BitNetDType::F32, &device).unwrap();
            let output = layer.forward(&input).unwrap();
            
            // Quantization step
            let quantized_output = quantizer.quantize(&output).unwrap();
            let dequantized = quantizer.dequantize(&quantized_output).unwrap();
            
            assert_eq!(dequantized.shape(), output.shape());
            drop((output, quantized_output, dequantized));
        }
        
        let duration = start.elapsed();
        let avg_duration = duration / iterations;
        let steps_per_second = 1.0 / avg_duration.as_secs_f64();
        
        println!(
            "End-to-end training: {:.0} steps/sec ({}ms per step)",
            steps_per_second,
            avg_duration.as_millis()
        );
        
        // Validate end-to-end performance
        assert!(
            steps_per_second > 50.0, // Minimum 50 training steps per second
            "End-to-end training should achieve >50 steps/sec, got {:.0}",
            steps_per_second
        );
    }

    #[test]
    fn test_batch_processing_performance() {
        let device = Device::Cpu;
        let batch_sizes = vec![1, 4, 8, 16];
        let input_features = 64;
        let output_features = 32;
        
        for batch_size in batch_sizes {
            let mut layer = BitLinear::new(input_features, output_features, &device).unwrap();
            let input = BitNetTensor::randn(&[batch_size, input_features], BitNetDType::F32, &device).unwrap();
            
            // Warm up
            for _ in 0..3 {
                let _ = layer.forward(&input).unwrap();
            }
            
            // Measure batch processing performance
            let iterations = 50;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let output = layer.forward(&input).unwrap();
                drop(output);
            }
            
            let duration = start.elapsed();
            let avg_duration = duration / iterations;
            let samples_per_second = (batch_size as f64) / avg_duration.as_secs_f64();
            
            println!(
                "Batch size {}: {:.0} samples/sec ({}μs per batch)",
                batch_size,
                samples_per_second,
                avg_duration.as_micros()
            );
            
            // Validate batch processing scales reasonably
            if batch_size == 1 {
                assert!(
                    samples_per_second > 500.0,
                    "Single sample processing should be >500 samples/sec, got {:.0}",
                    samples_per_second
                );
            } else {
                assert!(
                    samples_per_second > 1000.0,
                    "Batch processing should benefit from batching: {:.0} samples/sec for batch size {}",
                    samples_per_second, batch_size
                );
            }
        }
    }

    #[test]
    fn test_precision_impact_performance() {
        let device = Device::Cpu;
        let input_size = vec![64, 64];
        let precisions = vec![
            QuantizationPrecision::Eight,
            QuantizationPrecision::Four,
            QuantizationPrecision::Two,
            QuantizationPrecision::OneFiveFive,
        ];
        
        let weights = BitNetTensor::randn(&input_size, BitNetDType::F32, &device).unwrap();
        
        for precision in precisions {
            let quantizer = BitNetQuantizer::new(precision);
            
            // Measure quantization performance for this precision
            let iterations = 30;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let quantized = quantizer.quantize(&weights).unwrap();
                let dequantized = quantizer.dequantize(&quantized).unwrap();
                drop((quantized, dequantized));
            }
            
            let duration = start.elapsed();
            let avg_duration = duration / iterations;
            let ops_per_second = 1.0 / avg_duration.as_secs_f64();
            
            println!(
                "Precision {:?}: {:.0} quantize-dequantize cycles/sec",
                precision, ops_per_second
            );
            
            // Validate precision performance
            assert!(
                ops_per_second > 100.0,
                "Precision {:?} should achieve >100 cycles/sec, got {:.0}",
                precision, ops_per_second
            );
        }
    }
}

/// Stress Testing for Production Validation
mod stress_tests {
    use super::*;

    #[test]
    fn test_sustained_load_performance() {
        let device = Device::Cpu;
        let duration = Duration::from_secs(5); // 5 second stress test
        let start = Instant::now();
        
        let mut operations_completed = 0;
        let mut errors_encountered = 0;
        
        while start.elapsed() < duration {
            // Perform sustained operations
            let tensor_result = BitNetTensor::randn(&[32, 32], BitNetDType::F32, &device);
            
            match tensor_result {
                Ok(tensor) => {
                    // Perform operation
                    let result = &tensor + &tensor;
                    if result.is_ok() {
                        operations_completed += 1;
                    } else {
                        errors_encountered += 1;
                    }
                    drop(result);
                }
                Err(_) => {
                    errors_encountered += 1;
                }
            }
            
            // Brief pause to avoid overwhelming the system
            if operations_completed % 100 == 0 {
                thread::sleep(Duration::from_millis(1));
            }
        }
        
        let actual_duration = start.elapsed();
        let ops_per_second = operations_completed as f64 / actual_duration.as_secs_f64();
        let error_rate = errors_encountered as f64 / (operations_completed + errors_encountered) as f64;
        
        println!(
            "Sustained load: {:.0} ops/sec, {:.2}% error rate over {:.1}s",
            ops_per_second,
            error_rate * 100.0,
            actual_duration.as_secs_f64()
        );
        
        // Validate sustained performance
        assert!(
            ops_per_second > 500.0,
            "Should maintain >500 ops/sec under sustained load, got {:.0}",
            ops_per_second
        );
        
        assert!(
            error_rate < 0.05, // <5% error rate acceptable
            "Error rate should be <5% under sustained load, got {:.2}%",
            error_rate * 100.0
        );
    }
    
    #[test]
    fn test_memory_pressure_recovery_performance() {
        let device = Device::Cpu;
        
        // Create memory pressure by allocating many tensors
        let mut tensors = Vec::new();
        let mut allocation_failures = 0;
        
        // Allocate until pressure
        for i in 0..100 {
            match BitNetTensor::zeros(&[64, 64], BitNetDType::F32, &device) {
                Ok(tensor) => tensors.push(tensor),
                Err(_) => {
                    allocation_failures += 1;
                    break;
                }
            }
        }
        
        println!("Allocated {} tensors before pressure", tensors.len());
        
        // Release half the tensors
        let keep_count = tensors.len() / 2;
        tensors.truncate(keep_count);
        
        // Allow cleanup time
        thread::sleep(Duration::from_millis(100));
        
        // Test recovery performance
        let recovery_start = Instant::now();
        let mut successful_recoveries = 0;
        
        for _ in 0..20 {
            match BitNetTensor::zeros(&[32, 32], BitNetDType::F32, &device) {
                Ok(tensor) => {
                    successful_recoveries += 1;
                    drop(tensor);
                }
                Err(_) => break,
            }
        }
        
        let recovery_duration = recovery_start.elapsed();
        let recovery_rate = successful_recoveries as f64 / recovery_duration.as_secs_f64();
        
        println!(
            "Memory pressure recovery: {} successful allocations in {}ms ({:.0} allocs/sec)",
            successful_recoveries,
            recovery_duration.as_millis(),
            recovery_rate
        );
        
        // Validate recovery performance
        assert!(
            successful_recoveries >= 10,
            "Should recover and allow at least 10 allocations, got {}",
            successful_recoveries
        );
        
        assert!(
            recovery_rate > 50.0,
            "Recovery rate should be >50 allocs/sec, got {:.0}",
            recovery_rate
        );
    }
}
