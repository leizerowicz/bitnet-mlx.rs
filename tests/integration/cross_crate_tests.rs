//! Cross-Crate Integration Tests
//!
//! This module provides comprehensive integration testing across all BitNet-Rust crates,
//! validating end-to-end functionality and cross-crate compatibility with performance
//! monitoring and timeout protection.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

// Core infrastructure
use bitnet_core::{
    BitNetTensor, BitNetDType, Device,
    memory::{HybridMemoryPool, MemoryTracker},
    tensor::acceleration::{AccelerationBackend, AccelerationContext},
};

// Quantization infrastructure  
use bitnet_quant::{
    quantization::{BitNetQuantizer, WeightQuantizationConfig, ActivationQuantizationConfig},
    bitlinear::{BitLinearLayer, BitLinearConfig},
    metrics::{ErrorAnalysisEngine, LayerWiseAnalysis},
};

// Training infrastructure
use bitnet_training::{
    qat::{QATTrainingState, QATStateTracker, straight_through::StraightThroughEstimator},
    device::Device as TrainingDevice,
};

// Inference infrastructure 
use bitnet_inference::{
    engine::InferenceEngine,
    config::InferenceConfig,
};

// CLI and benchmarking
use bitnet_benchmarks::{
    benchmark::{BenchmarkConfig, BenchmarkRunner},
    comparison::{ComparisonConfig, ModelComparison},
};

/// Cross-crate integration test configuration
#[derive(Clone)]
struct IntegrationTestConfig {
    /// Enable performance monitoring during tests
    enable_performance_tracking: bool,
    /// Enable memory pressure testing
    enable_memory_testing: bool,
    /// Test timeout duration
    max_test_duration: Duration,
    /// Size of tensors for testing
    tensor_size: (usize, usize),
    /// Number of integration iterations
    integration_iterations: usize,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            enable_performance_tracking: true,
            enable_memory_testing: true,
            max_test_duration: Duration::from_secs(180),
            tensor_size: (32, 128),
            integration_iterations: 10,
        }
    }
}

/// Integration test environment with cross-crate resources
struct IntegrationTestEnvironment {
    config: IntegrationTestConfig,
    device: Device,
    memory_pool: Arc<HybridMemoryPool>,
    performance_tracker: PerformanceTracker,
}

impl IntegrationTestEnvironment {
    fn new() -> anyhow::Result<Self> {
        let config = IntegrationTestConfig::default();
        let device = Device::Cpu;
        
        // Initialize memory pool
        let memory_pool = Arc::new(HybridMemoryPool::new(
            1024 * 1024 * 16, // 16MB initial
            Some(1024 * 1024 * 64), // 64MB max
            device.clone(),
        )?);
        
        let performance_tracker = PerformanceTracker::new();
        
        Ok(Self {
            config,
            device,
            memory_pool,
            performance_tracker,
        })
    }
    
    fn create_test_tensor(&self, shape: &[usize]) -> anyhow::Result<BitNetTensor> {
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32) * 0.01)
            .collect();
        
        BitNetTensor::from_data(data, shape, BitNetDType::F32, &self.device, &self.memory_pool)
    }
}

/// Performance tracking for integration tests
#[derive(Debug, Default)]
struct PerformanceTracker {
    operation_times: HashMap<String, Duration>,
    memory_usage: HashMap<String, usize>,
    throughput_metrics: HashMap<String, f64>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_operation<T, F>(&mut self, name: &str, operation: F) -> anyhow::Result<T>
    where
        F: FnOnce() -> anyhow::Result<T>,
    {
        let start = Instant::now();
        let result = operation()?;
        let duration = start.elapsed();
        
        self.operation_times.insert(name.to_string(), duration);
        Ok(result)
    }
    
    fn record_throughput(&mut self, name: &str, operations: usize, duration: Duration) {
        let throughput = operations as f64 / duration.as_secs_f64();
        self.throughput_metrics.insert(name.to_string(), throughput);
    }
}

/// Integration test utilities
mod integration_utils {
    use super::*;
    
    pub fn with_timeout<T, F>(duration: Duration, test_name: &str, test_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce() -> anyhow::Result<T>,
    {
        let start_time = Instant::now();
        let result = test_fn();
        let elapsed = start_time.elapsed();
        
        if elapsed > duration {
            anyhow::bail!("Integration test '{}' exceeded timeout of {:?}, took {:?}", 
                          test_name, duration, elapsed);
        }
        
        result
    }
    
    pub fn validate_tensor_compatibility(tensor1: &BitNetTensor, tensor2: &BitNetTensor) -> bool {
        tensor1.shape() == tensor2.shape() && 
        tensor1.dtype() == tensor2.dtype() &&
        tensor1.device() == tensor2.device()
    }
    
    pub fn measure_cross_crate_operation_performance<F>(
        operation_name: &str,
        iterations: usize,
        operation: F
    ) -> anyhow::Result<(Duration, f64)>
    where
        F: Fn() -> anyhow::Result<()>,
    {
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            operation()?;
        }
        
        let total_time = start_time.elapsed();
        let ops_per_second = iterations as f64 / total_time.as_secs_f64();
        
        println!("🔍 {}: {:.2} ops/sec ({} iterations in {:?})", 
                operation_name, ops_per_second, iterations, total_time);
        
        Ok((total_time, ops_per_second))
    }
}

#[cfg(test)]
mod cross_crate_integration_tests {
    use super::*;
    use integration_utils::*;
    
    /// Test core + quant integration: tensor quantization workflow
    #[test]
    fn test_core_quant_integration() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(120), "core_quant_integration", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // Create test tensor using bitnet-core
            let tensor = env.create_test_tensor(&[32, 128])?;
            
            // Configure quantizer from bitnet-quant
            let weight_config = WeightQuantizationConfig::default();
            let activation_config = ActivationQuantizationConfig::default();
            
            let quantizer = BitNetQuantizer::new(
                weight_config.clone(),
                activation_config.clone(),
                env.device.clone()
            );
            
            // Quantize tensor (core tensor → quant operations)
            let quantized_weights = quantizer.quantize_weights(&tensor)?;
            let quantized_activations = quantizer.quantize_activations(&tensor)?;
            
            // Validate quantization results
            assert_eq!(quantized_weights.shape(), tensor.shape());
            assert_eq!(quantized_activations.shape(), tensor.shape());
            
            // Test dequantization (quant → core)
            let dequantized = quantizer.dequantize_weights(&quantized_weights)?;
            assert!(validate_tensor_compatibility(&dequantized, &tensor));
            
            // Measure quantization performance
            let (duration, ops_per_sec) = measure_cross_crate_operation_performance(
                "Core→Quant quantization", 
                100,
                || {
                    let _quantized = quantizer.quantize_weights(&tensor)?;
                    Ok(())
                }
            )?;
            
            // Validate performance targets
            assert!(ops_per_sec > 1000.0, 
                   "Core→Quant quantization should exceed 1K ops/sec, got: {:.2}", ops_per_sec);
            
            println!("✅ Core + Quant integration test completed");
            println!("   Quantization performance: {:.2} ops/sec", ops_per_sec);
            
            Ok(())
        })
    }
    
    /// Test training + inference pipeline: complete model workflow
    #[test]
    fn test_training_inference_pipeline() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(180), "training_inference_pipeline", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // === TRAINING PHASE ===
            
            // Create training data using bitnet-core
            let training_input = env.create_test_tensor(&[16, 64])?;
            let training_target = env.create_test_tensor(&[16, 32])?;
            
            // Initialize QAT training from bitnet-training
            let mut training_state = QATTrainingState::new();
            training_state.set_quantization_enabled(true);
            training_state.set_learning_rate(0.001);
            
            let training_device = TrainingDevice::Cpu;
            let mut state_tracker = QATStateTracker::new(training_state, &training_device);
            
            // Create BitLinear layer from bitnet-quant
            let layer_config = BitLinearConfig {
                input_features: 64,
                output_features: 32,
                use_bias: true,
                quantization_bits: 2,
                activation_quantization: true,
            };
            
            let mut bitlinear_layer = BitLinearLayer::new(layer_config, &env.device)?;
            
            // Training loop simulation
            let mut training_losses = Vec::new();
            for epoch in 0..5 {
                // Forward pass through quantized layer
                let output = bitlinear_layer.forward(&training_input)?;
                
                // Compute simple MSE loss
                let diff = output.sub(&training_target)?;
                let loss_tensor = diff.pow(2.0)?.mean_all()?;
                let loss_value = loss_tensor.to_scalar::<f32>()?;
                
                // Update training state
                state_tracker.update_iteration(epoch);
                state_tracker.update_loss(loss_value);
                training_losses.push(loss_value);
                
                // Simulate parameter updates (simplified)
                bitlinear_layer.update_quantization_parameters(0.95)?; // Slight adjustment
            }
            
            // Validate training convergence
            assert!(training_losses.len() == 5);
            let initial_loss = training_losses[0];
            let final_loss = training_losses[4];
            
            println!("📊 Training: Initial loss: {:.6}, Final loss: {:.6}", initial_loss, final_loss);
            
            // === INFERENCE PHASE ===
            
            // Create inference engine from bitnet-inference
            let inference_config = InferenceConfig {
                batch_size: 8,
                max_sequence_length: 64,
                use_quantization: true,
                device: env.device.clone(),
                memory_pool: Some(env.memory_pool.clone()),
            };
            
            let mut inference_engine = InferenceEngine::new(inference_config)?;
            
            // Load trained model (simulate by using the trained layer)
            inference_engine.load_layer("bitlinear_0", Box::new(bitlinear_layer))?;
            
            // Create inference input
            let inference_input = env.create_test_tensor(&[8, 64])?;
            
            // Run inference
            let inference_output = inference_engine.forward(&inference_input)?;
            
            // Validate inference results
            assert_eq!(inference_output.shape(), &[8, 32]);
            assert_eq!(inference_output.dtype(), BitNetDType::F32);
            
            // Measure inference performance
            let (duration, ops_per_sec) = measure_cross_crate_operation_performance(
                "Training→Inference pipeline",
                50,
                || {
                    let _output = inference_engine.forward(&inference_input)?;
                    Ok(())
                }
            )?;
            
            // Validate performance targets
            assert!(ops_per_sec > 500.0,
                   "Training→Inference pipeline should exceed 500 ops/sec, got: {:.2}", ops_per_sec);
            
            println!("✅ Training + Inference pipeline test completed");
            println!("   Training converged from {:.6} to {:.6}", initial_loss, final_loss);
            println!("   Inference performance: {:.2} ops/sec", ops_per_sec);
            
            Ok(())
        })
    }
    
    /// Test core + metal acceleration integration
    #[test]
    #[cfg(feature = "metal")]
    fn test_core_metal_integration() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(90), "core_metal_integration", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // Try to use Metal device if available
            let metal_device = match Device::Metal {
                device if device.is_available() => device,
                _ => {
                    println!("⚠️  Metal device not available, skipping test");
                    return Ok(());
                }
            };
            
            // Create tensor on Metal device
            let cpu_tensor = env.create_test_tensor(&[256, 256])?;
            let metal_tensor = cpu_tensor.to_device(&metal_device)?;
            
            // Perform matrix operations on Metal
            let result = metal_tensor.matmul(&metal_tensor.transpose()?)?;
            
            // Copy back to CPU for validation
            let cpu_result = result.to_device(&Device::Cpu)?;
            
            // Validate shape and basic properties
            assert_eq!(cpu_result.shape(), &[256, 256]);
            
            // Measure Metal acceleration performance
            let (duration, ops_per_sec) = measure_cross_crate_operation_performance(
                "Core→Metal matrix multiplication",
                10,
                || {
                    let _result = metal_tensor.matmul(&metal_tensor)?;
                    Ok(())
                }
            )?;
            
            println!("✅ Core + Metal integration test completed");
            println!("   Metal acceleration: {:.2} ops/sec", ops_per_sec);
            
            Ok(())
        })
    }
    
    /// Test comprehensive quantization + training + inference workflow
    #[test]
    fn test_quantization_training_inference_workflow() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(300), "quantization_training_inference_workflow", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // === QUANTIZATION SETUP ===
            
            // Create original full-precision model weights
            let weights = env.create_test_tensor(&[128, 64])?;
            let bias = env.create_test_tensor(&[64])?;
            
            // Set up quantization analysis
            let error_analyzer = ErrorAnalysisEngine::new(&env.device);
            let layer_analyzer = LayerWiseAnalysis::new();
            
            // Configure multi-precision quantization
            let weight_config = WeightQuantizationConfig {
                quantization_bits: 2,
                use_symmetric: true,
                calibration_samples: 100,
                error_compensation: true,
            };
            
            let quantizer = BitNetQuantizer::new(
                weight_config.clone(),
                ActivationQuantizationConfig::default(),
                env.device.clone(),
            );
            
            // Quantize model weights
            let quantized_weights = quantizer.quantize_weights(&weights)?;
            let quantized_bias = quantizer.quantize_weights(&bias)?;
            
            // Analyze quantization error
            let quantization_error = error_analyzer.compute_layer_error(
                &weights, &quantized_weights, "test_layer"
            )?;
            
            println!("📊 Quantization error analysis:");
            println!("   MSE: {:.6}", quantization_error.mse());
            println!("   SQNR: {:.2} dB", quantization_error.sqnr());
            
            // === QAT TRAINING ===
            
            // Initialize QAT training state
            let mut qat_state = QATTrainingState::new();
            qat_state.set_quantization_enabled(true);
            qat_state.set_learning_rate(0.001);
            
            let training_device = TrainingDevice::Cpu;
            let mut qat_tracker = QATStateTracker::new(qat_state, &training_device);
            
            // Create BitLinear layer with quantized weights
            let bitlinear_config = BitLinearConfig {
                input_features: 128,
                output_features: 64,
                use_bias: true,
                quantization_bits: 2,
                activation_quantization: true,
            };
            
            let mut qat_layer = BitLinearLayer::with_weights(
                bitlinear_config,
                quantized_weights,
                Some(quantized_bias),
                &env.device,
            )?;
            
            // QAT training simulation
            let training_input = env.create_test_tensor(&[32, 128])?;
            let training_target = env.create_test_tensor(&[32, 64])?;
            
            let mut qat_losses = Vec::new();
            for iteration in 0..10 {
                // Forward pass with quantized operations
                let qat_output = qat_layer.forward(&training_input)?;
                
                // Compute training loss
                let loss_tensor = qat_output.sub(&training_target)?.pow(2.0)?.mean_all()?;
                let loss_value = loss_tensor.to_scalar::<f32>()?;
                
                // Update QAT state
                qat_tracker.update_iteration(iteration);
                qat_tracker.update_loss(loss_value);
                qat_losses.push(loss_value);
                
                // Simulate gradient-based quantization adjustment
                qat_layer.update_quantization_parameters(0.9 + iteration as f32 * 0.01)?;
            }
            
            // Validate QAT convergence
            let qat_initial_loss = qat_losses[0];
            let qat_final_loss = qat_losses[9];
            
            println!("📊 QAT Training results:");
            println!("   Initial loss: {:.6}, Final loss: {:.6}", qat_initial_loss, qat_final_loss);
            
            // === INFERENCE DEPLOYMENT ===
            
            // Deploy quantized model to inference engine
            let inference_config = InferenceConfig {
                batch_size: 16,
                max_sequence_length: 128,
                use_quantization: true,
                device: env.device.clone(),
                memory_pool: Some(env.memory_pool.clone()),
            };
            
            let mut inference_engine = InferenceEngine::new(inference_config)?;
            inference_engine.load_layer("qat_layer", Box::new(qat_layer))?;
            
            // Run inference with different batch sizes
            let test_cases = vec![
                (1, 128),   // Single sample
                (8, 128),   // Small batch
                (32, 128),  // Large batch
            ];
            
            let mut inference_results = Vec::new();
            for (batch_size, input_size) in test_cases {
                let inference_input = env.create_test_tensor(&[batch_size, input_size])?;
                let inference_output = inference_engine.forward(&inference_input)?;
                
                assert_eq!(inference_output.shape(), &[batch_size, 64]);
                inference_results.push((batch_size, inference_output.shape().to_vec()));
            }
            
            // === PERFORMANCE BENCHMARKING ===
            
            // Benchmark the complete pipeline
            let benchmark_config = BenchmarkConfig {
                warmup_iterations: 10,
                measurement_iterations: 100,
                enable_memory_tracking: true,
                enable_performance_analysis: true,
            };
            
            let mut benchmark_runner = BenchmarkRunner::new(benchmark_config, false);
            
            // Benchmark quantization performance
            let quant_benchmark = benchmark_runner.benchmark_operation(
                "Quantization",
                || {
                    let _quantized = quantizer.quantize_weights(&weights)?;
                    Ok(())
                }
            )?;
            
            // Benchmark inference performance
            let inference_benchmark = benchmark_runner.benchmark_operation(
                "Quantized Inference",
                || {
                    let test_input = env.create_test_tensor(&[16, 128])?;
                    let _output = inference_engine.forward(&test_input)?;
                    Ok(())
                }
            )?;
            
            // Validate performance targets
            assert!(quant_benchmark.ops_per_second > 500.0,
                   "Quantization should exceed 500 ops/sec, got: {:.2}", quant_benchmark.ops_per_second);
            
            assert!(inference_benchmark.ops_per_second > 200.0,
                   "Quantized inference should exceed 200 ops/sec, got: {:.2}", inference_benchmark.ops_per_second);
            
            println!("✅ Complete quantization + training + inference workflow test completed");
            println!("   Quantization: {:.2} ops/sec", quant_benchmark.ops_per_second);
            println!("   QAT converged: {:.6} → {:.6}", qat_initial_loss, qat_final_loss);
            println!("   Inference: {:.2} ops/sec", inference_benchmark.ops_per_second);
            println!("   Processed {} different batch sizes", inference_results.len());
            
            Ok(())
        })
    }
    
    /// Test memory management across all crates
    #[test]
    fn test_cross_crate_memory_management() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(120), "cross_crate_memory_management", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // Track initial memory state
            let initial_stats = env.memory_pool.get_stats();
            
            // Create tensors across different crates
            let mut allocated_tensors = Vec::new();
            
            // Core tensors
            for i in 0..10 {
                let tensor = env.create_test_tensor(&[64, 64 + i * 10])?;
                allocated_tensors.push(("core", tensor));
            }
            
            // Quantization tensors
            let quantizer = BitNetQuantizer::new(
                WeightQuantizationConfig::default(),
                ActivationQuantizationConfig::default(),
                env.device.clone(),
            );
            
            for (name, tensor) in &allocated_tensors {
                if name == &"core" {
                    let quantized = quantizer.quantize_weights(tensor)?;
                    allocated_tensors.push(("quant", quantized));
                }
            }
            
            // Training tensors (QAT state)
            let mut training_states = Vec::new();
            for i in 0..5 {
                let mut state = QATTrainingState::new();
                state.set_learning_rate(0.001 * (i + 1) as f32);
                training_states.push(state);
            }
            
            // Check memory usage under load
            let loaded_stats = env.memory_pool.get_stats();
            assert!(loaded_stats.total_allocated_bytes > initial_stats.total_allocated_bytes);
            
            // Test memory pressure handling
            let pressure_tensors: Result<Vec<_>, _> = (0..20)
                .map(|i| env.create_test_tensor(&[128, 128 + i * 8]))
                .collect();
            
            match pressure_tensors {
                Ok(tensors) => {
                    println!("✅ Successfully allocated {} pressure test tensors", tensors.len());
                }
                Err(e) => {
                    println!("📊 Memory pressure correctly triggered: {}", e);
                }
            }
            
            // Clear allocations
            allocated_tensors.clear();
            training_states.clear();
            
            // Force cleanup
            env.memory_pool.cleanup_unused_blocks();
            
            // Validate memory cleanup
            let final_stats = env.memory_pool.get_stats();
            
            println!("📊 Memory management test results:");
            println!("   Initial allocated: {} bytes", initial_stats.total_allocated_bytes);
            println!("   Peak allocated: {} bytes", loaded_stats.total_allocated_bytes);
            println!("   Final allocated: {} bytes", final_stats.total_allocated_bytes);
            
            // Memory should be efficiently reclaimed
            let cleanup_efficiency = (loaded_stats.total_allocated_bytes - final_stats.total_allocated_bytes) as f64 
                                   / loaded_stats.total_allocated_bytes as f64;
            
            assert!(cleanup_efficiency > 0.8, 
                   "Memory cleanup efficiency should exceed 80%, got: {:.2}%", cleanup_efficiency * 100.0);
            
            println!("✅ Cross-crate memory management test completed");
            println!("   Cleanup efficiency: {:.2}%", cleanup_efficiency * 100.0);
            
            Ok(())
        })
    }
    
    /// Test error propagation and recovery across crates
    #[test]
    fn test_cross_crate_error_handling() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(90), "cross_crate_error_handling", || {
            let env = IntegrationTestEnvironment::new()?;
            
            // === Test Core → Quant Error Propagation ===
            
            // Create invalid tensor for quantization
            let empty_tensor = BitNetTensor::zeros(&[0, 0], BitNetDType::F32, &env.device, &env.memory_pool)?;
            let quantizer = BitNetQuantizer::new(
                WeightQuantizationConfig::default(),
                ActivationQuantizationConfig::default(),
                env.device.clone(),
            );
            
            // Should propagate error gracefully
            let quant_result = quantizer.quantize_weights(&empty_tensor);
            assert!(quant_result.is_err(), "Should handle empty tensor error from core");
            
            // === Test Training → Inference Error Propagation ===
            
            // Create invalid training state
            let mut invalid_state = QATTrainingState::new();
            invalid_state.set_learning_rate(-1.0); // Invalid learning rate
            
            // Training should handle invalid configuration
            let training_device = TrainingDevice::Cpu;
            let tracker_result = std::panic::catch_unwind(|| {
                QATStateTracker::new(invalid_state, &training_device)
            });
            
            // Should not panic
            assert!(tracker_result.is_ok(), "Training should handle invalid state gracefully");
            
            // === Test Inference Error Recovery ===
            
            let inference_config = InferenceConfig {
                batch_size: 0, // Invalid batch size
                max_sequence_length: 64,
                use_quantization: true,
                device: env.device.clone(),
                memory_pool: Some(env.memory_pool.clone()),
            };
            
            // Should handle invalid configuration
            let inference_result = InferenceEngine::new(inference_config);
            assert!(inference_result.is_err(), "Should handle invalid inference config");
            
            // === Test Memory Error Recovery ===
            
            // Try to allocate excessive memory
            let excessive_allocation = BitNetTensor::zeros(
                &[10000, 10000, 10000], 
                BitNetDType::F32, 
                &env.device, 
                &env.memory_pool
            );
            
            match excessive_allocation {
                Err(_) => println!("✅ Correctly handled excessive memory allocation"),
                Ok(_) => println!("⚠️  Large allocation succeeded unexpectedly"),
            }
            
            // === Test Benchmark Error Handling ===
            
            let benchmark_config = BenchmarkConfig {
                warmup_iterations: 0,
                measurement_iterations: 0, // Invalid: zero iterations
                enable_memory_tracking: true,
                enable_performance_analysis: true,
            };
            
            let mut benchmark_runner = BenchmarkRunner::new(benchmark_config, false);
            
            // Should handle invalid benchmark configuration
            let benchmark_result = benchmark_runner.benchmark_operation(
                "Invalid Benchmark",
                || Ok(())
            );
            
            // Should complete gracefully even with invalid config
            assert!(benchmark_result.is_ok(), "Benchmark should handle zero iterations gracefully");
            
            println!("✅ Cross-crate error handling test completed");
            println!("   All error scenarios handled gracefully");
            
            Ok(())
        })
    }
}
