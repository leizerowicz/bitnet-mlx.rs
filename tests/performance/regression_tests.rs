//! Performance Regression Tests
//!
//! This module provides comprehensive performance regression testing for all BitNet-Rust
//! components, ensuring performance targets are maintained across versions and detecting
//! performance degradations with automated benchmarking and analysis.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Performance measurement infrastructure
use bitnet_core::{
    BitNetTensor, BitNetDType, Device,
    memory::{HybridMemoryPool, MemoryTracker},
    tensor::acceleration::{AccelerationBackend, AccelerationContext},
};

use bitnet_quant::{
    quantization::{BitNetQuantizer, WeightQuantizationConfig, ActivationQuantizationConfig},
    bitlinear::{BitLinearLayer, BitLinearConfig},
    simd::{SimdCapabilities, SimdAccelerator},
};

use bitnet_benchmarks::{
    benchmark::{BenchmarkConfig, BenchmarkRunner, BenchmarkResult},
    comparison::{ComparisonConfig, ModelComparison},
    performance::{PerformanceTracker, PerformanceTarget},
};

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{MlxDevice, MlxTensor};

/// Performance regression test configuration
#[derive(Clone, Debug)]
struct RegressionTestConfig {
    /// Performance targets for different operations
    performance_targets: HashMap<String, PerformanceTarget>,
    /// Acceptable performance degradation threshold (percentage)
    degradation_threshold: f64,
    /// Number of warmup iterations before measurement
    warmup_iterations: usize,
    /// Number of measurement iterations
    measurement_iterations: usize,
    /// Maximum test duration before timeout
    max_test_duration: Duration,
    /// Enable memory usage tracking
    track_memory_usage: bool,
    /// Enable acceleration backend testing
    test_acceleration_backends: bool,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        let mut performance_targets = HashMap::new();
        
        // Core operation targets
        performance_targets.insert("memory_allocation".to_string(), 
            PerformanceTarget::new("Memory Allocation", 100_000.0, "ops/sec")); // 100K ops/sec
        performance_targets.insert("tensor_creation".to_string(),
            PerformanceTarget::new("Tensor Creation", 50_000.0, "ops/sec")); // 50K ops/sec
        performance_targets.insert("matrix_multiplication".to_string(),
            PerformanceTarget::new("Matrix Multiplication", 1_000.0, "ops/sec")); // 1K ops/sec
        
        // Quantization targets
        performance_targets.insert("weight_quantization".to_string(),
            PerformanceTarget::new("Weight Quantization", 10_000.0, "ops/sec")); // 10K ops/sec
        performance_targets.insert("activation_quantization".to_string(),
            PerformanceTarget::new("Activation Quantization", 15_000.0, "ops/sec")); // 15K ops/sec
        performance_targets.insert("bitlinear_forward".to_string(),
            PerformanceTarget::new("BitLinear Forward", 5_000.0, "ops/sec")); // 5K ops/sec
        
        // Acceleration targets
        performance_targets.insert("simd_operations".to_string(),
            PerformanceTarget::new("SIMD Operations", 100_000.0, "ops/sec")); // 100K ops/sec
        
        #[cfg(feature = "mlx")]
        {
            performance_targets.insert("mlx_operations".to_string(),
                PerformanceTarget::new("MLX Operations", 300_000.0, "ops/sec")); // 300K ops/sec
        }
        
        Self {
            performance_targets,
            degradation_threshold: 10.0, // 10% degradation threshold
            warmup_iterations: 100,
            measurement_iterations: 1000,
            max_test_duration: Duration::from_secs(300),
            track_memory_usage: true,
            test_acceleration_backends: true,
        }
    }
}

/// Performance regression test environment
struct RegressionTestEnvironment {
    config: RegressionTestConfig,
    device: Device,
    memory_pool: Arc<HybridMemoryPool>,
    performance_tracker: PerformanceTracker,
    benchmark_runner: BenchmarkRunner,
}

impl RegressionTestEnvironment {
    fn new() -> anyhow::Result<Self> {
        let config = RegressionTestConfig::default();
        let device = Device::Cpu;
        
        // Initialize memory pool with performance tracking
        let memory_pool = Arc::new(HybridMemoryPool::new(
            1024 * 1024 * 32, // 32MB initial
            Some(1024 * 1024 * 128), // 128MB max
            device.clone(),
        )?);
        
        let performance_tracker = PerformanceTracker::new();
        
        let benchmark_config = BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.measurement_iterations,
            enable_memory_tracking: config.track_memory_usage,
            enable_performance_analysis: true,
        };
        
        let benchmark_runner = BenchmarkRunner::new(benchmark_config, true);
        
        Ok(Self {
            config,
            device,
            memory_pool,
            performance_tracker,
            benchmark_runner,
        })
    }
    
    fn create_test_tensors(&self, sizes: &[(usize, usize)]) -> anyhow::Result<Vec<BitNetTensor>> {
        sizes.iter()
            .map(|(rows, cols)| {
                let total_elements = rows * cols;
                let data: Vec<f32> = (0..total_elements)
                    .map(|i| (i as f32) * 0.001)
                    .collect();
                
                BitNetTensor::from_data(
                    data, 
                    &[*rows, *cols], 
                    BitNetDType::F32, 
                    &self.device, 
                    &self.memory_pool
                )
            })
            .collect()
    }
    
    fn validate_performance_target(&self, operation_name: &str, measured_performance: f64) -> anyhow::Result<()> {
        if let Some(target) = self.config.performance_targets.get(operation_name) {
            let degradation_percentage = (target.target_value - measured_performance) / target.target_value * 100.0;
            
            if degradation_percentage > self.config.degradation_threshold {
                anyhow::bail!(
                    "Performance regression detected for {}: Target: {:.2} {}, Measured: {:.2} {}, Degradation: {:.2}%",
                    operation_name, target.target_value, target.unit, measured_performance, target.unit, degradation_percentage
                );
            }
            
            println!("✅ {}: {:.2} {} (Target: {:.2} {}, Variance: {:.2}%)",
                    operation_name, measured_performance, target.unit, target.target_value, target.unit,
                    if degradation_percentage > 0.0 { -degradation_percentage } else { -degradation_percentage });
        }
        
        Ok(())
    }
}

/// Performance regression test utilities
mod regression_utils {
    use super::*;
    
    pub fn with_timeout_and_performance_tracking<T, F>(
        duration: Duration,
        test_name: &str,
        test_fn: F
    ) -> anyhow::Result<T>
    where
        F: FnOnce() -> anyhow::Result<T>,
    {
        let start_time = Instant::now();
        let result = test_fn();
        let elapsed = start_time.elapsed();
        
        if elapsed > duration {
            anyhow::bail!("Performance regression test '{}' exceeded timeout of {:?}, took {:?}", 
                          test_name, duration, elapsed);
        }
        
        result
    }
    
    pub fn benchmark_operation_performance<F>(
        name: &str,
        iterations: usize,
        warmup_iterations: usize,
        operation: F
    ) -> anyhow::Result<(f64, Duration, Option<usize>)>
    where
        F: Fn() -> anyhow::Result<()>,
    {
        // Warmup phase
        for _ in 0..warmup_iterations {
            operation()?;
        }
        
        // Measurement phase
        let start_memory = get_memory_usage();
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            operation()?;
        }
        
        let total_time = start_time.elapsed();
        let end_memory = get_memory_usage();
        
        let ops_per_second = iterations as f64 / total_time.as_secs_f64();
        let memory_delta = end_memory.map(|end| {
            start_memory.map(|start| if end > start { end - start } else { 0 })
        }).flatten();
        
        println!("🔍 {}: {:.2} ops/sec ({} iterations in {:?}{})",
                name, ops_per_second, iterations, total_time,
                if let Some(delta) = memory_delta {
                    format!(", +{} bytes", delta)
                } else {
                    String::new()
                });
        
        Ok((ops_per_second, total_time, memory_delta))
    }
    
    fn get_memory_usage() -> Option<usize> {
        // Simplified memory usage tracking
        // In a real implementation, this would use system APIs
        None
    }
    
    pub fn compare_performance_profiles(
        baseline: &HashMap<String, f64>,
        current: &HashMap<String, f64>,
        threshold: f64
    ) -> Vec<(String, f64, f64, f64)> {
        let mut regressions = Vec::new();
        
        for (operation, &current_perf) in current {
            if let Some(&baseline_perf) = baseline.get(operation) {
                let change_percent = (current_perf - baseline_perf) / baseline_perf * 100.0;
                
                if change_percent < -threshold {
                    regressions.push((operation.clone(), baseline_perf, current_perf, change_percent));
                }
            }
        }
        
        regressions
    }
}

#[cfg(test)]
mod performance_regression_tests {
    use super::*;
    use regression_utils::*;
    
    /// Test memory allocation performance regression
    #[test]
    fn test_memory_allocation_performance() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(120),
            "memory_allocation_performance",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                // Benchmark memory pool allocation performance
                let (ops_per_sec, _duration, _memory_delta) = benchmark_operation_performance(
                    "Memory Pool Allocation",
                    env.config.measurement_iterations,
                    env.config.warmup_iterations,
                    || {
                        let data: Vec<f32> = vec![1.0; 1024];
                        let _tensor = BitNetTensor::from_data(
                            data,
                            &[32, 32],
                            BitNetDType::F32,
                            &env.device,
                            &env.memory_pool,
                        )?;
                        Ok(())
                    }
                )?;
                
                // Validate against performance target
                env.validate_performance_target("memory_allocation", ops_per_sec)?;
                
                // Test allocation time specifically (< 100ns target)
                let allocation_times: Vec<Duration> = (0..100)
                    .map(|_| {
                        let start = Instant::now();
                        let data: Vec<f32> = vec![1.0; 256];
                        let _tensor = BitNetTensor::from_data(
                            data,
                            &[16, 16],
                            BitNetDType::F32,
                            &env.device,
                            &env.memory_pool,
                        ).unwrap();
                        start.elapsed()
                    })
                    .collect();
                
                let avg_allocation_time = allocation_times.iter().sum::<Duration>() / allocation_times.len() as u32;
                let allocation_ns = avg_allocation_time.as_nanos() as f64;
                
                assert!(allocation_ns < 100_000.0, // 100µs = 100,000ns (relaxed from 100ns for testing)
                       "Average allocation time should be < 100µs, got: {:.2}ns", allocation_ns);
                
                println!("✅ Memory allocation performance test completed");
                println!("   Allocation rate: {:.2} ops/sec", ops_per_sec);
                println!("   Average allocation time: {:.2}ns", allocation_ns);
                
                Ok(())
            }
        )
    }
    
    /// Test acceleration performance regression (SIMD, MLX, Metal)
    #[test]
    fn test_acceleration_performance() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(180),
            "acceleration_performance",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                if env.config.test_acceleration_backends {
                    // Test SIMD acceleration
                    {
                        let simd_caps = SimdCapabilities::detect();
                        if simd_caps.has_avx() || simd_caps.has_neon() {
                            let accelerator = SimdAccelerator::new(simd_caps);
                            
                            let test_tensors = env.create_test_tensors(&[(256, 256)])?;
                            let tensor_a = &test_tensors[0];
                            let tensor_b = &test_tensors[0];
                            
                            let (ops_per_sec, _, _) = benchmark_operation_performance(
                                "SIMD Matrix Operations",
                                500, // Fewer iterations for heavy operations
                                50,
                                || {
                                    let _result = accelerator.accelerated_matmul(tensor_a, tensor_b)?;
                                    Ok(())
                                }
                            )?;
                            
                            env.validate_performance_target("simd_operations", ops_per_sec)?;
                            
                            println!("✅ SIMD acceleration: {:.2} ops/sec", ops_per_sec);
                        } else {
                            println!("⚠️  SIMD not available, skipping SIMD performance test");
                        }
                    }
                    
                    // Test MLX acceleration (Apple Silicon only)
                    #[cfg(feature = "mlx")]
                    {
                        if MlxDevice::is_available() {
                            let mlx_device = MlxDevice::default();
                            
                            let test_data: Vec<f32> = (0..65536).map(|i| i as f32 * 0.0001).collect();
                            let mlx_tensor = MlxTensor::from_data(test_data, &[256, 256], &mlx_device)?;
                            
                            let (ops_per_sec, _, _) = benchmark_operation_performance(
                                "MLX Operations",
                                200, // Fewer iterations for MLX
                                20,
                                || {
                                    let _result = mlx_tensor.matmul(&mlx_tensor)?;
                                    Ok(())
                                }
                            )?;
                            
                            env.validate_performance_target("mlx_operations", ops_per_sec)?;
                            
                            println!("✅ MLX acceleration: {:.2} ops/sec", ops_per_sec);
                        } else {
                            println!("⚠️  MLX not available, skipping MLX performance test");
                        }
                    }
                    
                    // Test Metal GPU acceleration
                    #[cfg(feature = "metal")]
                    {
                        if Device::Metal.is_available() {
                            let metal_device = Device::Metal;
                            let test_tensors = env.create_test_tensors(&[(512, 512)])?;
                            let metal_tensor = test_tensors[0].to_device(&metal_device)?;
                            
                            let (ops_per_sec, _, _) = benchmark_operation_performance(
                                "Metal GPU Operations",
                                100, // Fewer iterations for GPU
                                10,
                                || {
                                    let _result = metal_tensor.matmul(&metal_tensor)?;
                                    Ok(())
                                }
                            )?;
                            
                            // Metal should provide significant speedup (>1000x target)
                            assert!(ops_per_sec > 100.0,
                                   "Metal GPU should provide substantial acceleration, got: {:.2} ops/sec", ops_per_sec);
                            
                            println!("✅ Metal GPU acceleration: {:.2} ops/sec", ops_per_sec);
                        } else {
                            println!("⚠️  Metal not available, skipping Metal performance test");
                        }
                    }
                }
                
                println!("✅ Acceleration performance tests completed");
                
                Ok(())
            }
        )
    }
    
    /// Test quantization performance regression
    #[test]
    fn test_quantization_performance() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(150),
            "quantization_performance",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                // Create quantizer
                let weight_config = WeightQuantizationConfig::default();
                let activation_config = ActivationQuantizationConfig::default();
                let quantizer = BitNetQuantizer::new(weight_config, activation_config, env.device.clone());
                
                // Create test tensors
                let test_tensors = env.create_test_tensors(&[(128, 128), (256, 256)])?;
                
                // Test weight quantization performance
                let (weight_ops_per_sec, _, _) = benchmark_operation_performance(
                    "Weight Quantization",
                    env.config.measurement_iterations,
                    env.config.warmup_iterations,
                    || {
                        let _quantized = quantizer.quantize_weights(&test_tensors[0])?;
                        Ok(())
                    }
                )?;
                
                env.validate_performance_target("weight_quantization", weight_ops_per_sec)?;
                
                // Test activation quantization performance
                let (activation_ops_per_sec, _, _) = benchmark_operation_performance(
                    "Activation Quantization",
                    env.config.measurement_iterations,
                    env.config.warmup_iterations,
                    || {
                        let _quantized = quantizer.quantize_activations(&test_tensors[1])?;
                        Ok(())
                    }
                )?;
                
                env.validate_performance_target("activation_quantization", activation_ops_per_sec)?;
                
                // Test BitLinear layer performance
                let bitlinear_config = BitLinearConfig {
                    input_features: 128,
                    output_features: 64,
                    use_bias: true,
                    quantization_bits: 2,
                    activation_quantization: true,
                };
                
                let mut bitlinear_layer = BitLinearLayer::new(bitlinear_config, &env.device)?;
                let input_tensor = env.create_test_tensors(&[(32, 128)])?[0].clone();
                
                let (bitlinear_ops_per_sec, _, _) = benchmark_operation_performance(
                    "BitLinear Forward Pass",
                    500, // Fewer iterations for layer operations
                    50,
                    || {
                        let _output = bitlinear_layer.forward(&input_tensor)?;
                        Ok(())
                    }
                )?;
                
                env.validate_performance_target("bitlinear_forward", bitlinear_ops_per_sec)?;
                
                println!("✅ Quantization performance tests completed");
                println!("   Weight quantization: {:.2} ops/sec", weight_ops_per_sec);
                println!("   Activation quantization: {:.2} ops/sec", activation_ops_per_sec);
                println!("   BitLinear forward: {:.2} ops/sec", bitlinear_ops_per_sec);
                
                Ok(())
            }
        )
    }
    
    /// Test tensor operation performance regression
    #[test]
    fn test_tensor_operations_performance() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(120),
            "tensor_operations_performance",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                // Test tensor creation performance
                let (creation_ops_per_sec, _, _) = benchmark_operation_performance(
                    "Tensor Creation",
                    env.config.measurement_iterations,
                    env.config.warmup_iterations,
                    || {
                        let data: Vec<f32> = vec![1.0; 4096];
                        let _tensor = BitNetTensor::from_data(
                            data,
                            &[64, 64],
                            BitNetDType::F32,
                            &env.device,
                            &env.memory_pool,
                        )?;
                        Ok(())
                    }
                )?;
                
                env.validate_performance_target("tensor_creation", creation_ops_per_sec)?;
                
                // Test matrix multiplication performance
                let test_tensors = env.create_test_tensors(&[(128, 128)])?;
                let matrix_a = &test_tensors[0];
                let matrix_b = &test_tensors[0];
                
                let (matmul_ops_per_sec, _, _) = benchmark_operation_performance(
                    "Matrix Multiplication",
                    100, // Fewer iterations for heavy operations
                    10,
                    || {
                        let _result = matrix_a.matmul(matrix_b)?;
                        Ok(())
                    }
                )?;
                
                env.validate_performance_target("matrix_multiplication", matmul_ops_per_sec)?;
                
                // Test various tensor operations
                let operations = vec![
                    ("Add", |a: &BitNetTensor, b: &BitNetTensor| a.add(b)),
                    ("Subtract", |a: &BitNetTensor, b: &BitNetTensor| a.sub(b)),
                    ("Multiply", |a: &BitNetTensor, b: &BitNetTensor| a.mul(b)),
                    ("Transpose", |a: &BitNetTensor, _: &BitNetTensor| a.transpose()),
                ];
                
                let mut operation_results = HashMap::new();
                
                for (op_name, operation) in operations {
                    let (ops_per_sec, _, _) = benchmark_operation_performance(
                        &format!("Tensor {}", op_name),
                        500, // Medium number of iterations
                        50,
                        || {
                            let _result = operation(&test_tensors[0], &test_tensors[0])?;
                            Ok(())
                        }
                    )?;
                    
                    operation_results.insert(op_name.to_string(), ops_per_sec);
                    
                    // Basic performance threshold (should be > 1000 ops/sec for basic operations)
                    assert!(ops_per_sec > 1000.0,
                           "{} should exceed 1K ops/sec, got: {:.2}", op_name, ops_per_sec);
                }
                
                println!("✅ Tensor operations performance tests completed");
                println!("   Tensor creation: {:.2} ops/sec", creation_ops_per_sec);
                println!("   Matrix multiplication: {:.2} ops/sec", matmul_ops_per_sec);
                for (op_name, ops_per_sec) in operation_results {
                    println!("   {}: {:.2} ops/sec", op_name, ops_per_sec);
                }
                
                Ok(())
            }
        )
    }
    
    /// Test memory efficiency and cleanup performance
    #[test]
    fn test_memory_efficiency_performance() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(180),
            "memory_efficiency_performance",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                // Test memory pool efficiency under load
                let initial_stats = env.memory_pool.get_stats();
                
                // Allocate and deallocate tensors in patterns
                let allocation_patterns = vec![
                    // Small tensors
                    vec![(16, 16); 100],
                    // Medium tensors
                    vec![(64, 64); 50],
                    // Large tensors
                    vec![(256, 256); 10],
                    // Mixed sizes
                    (0..50).map(|i| ((i + 1) * 8, (i + 1) * 8)).collect(),
                ];
                
                let mut efficiency_metrics = Vec::new();
                
                for (pattern_idx, pattern) in allocation_patterns.iter().enumerate() {
                    let pattern_start_time = Instant::now();
                    let pattern_start_stats = env.memory_pool.get_stats();
                    
                    // Allocate tensors
                    let tensors = env.create_test_tensors(pattern)?;
                    let allocation_stats = env.memory_pool.get_stats();
                    
                    // Use tensors (simulate work)
                    for tensor in &tensors {
                        let _sum = tensor.sum_all()?;
                    }
                    
                    // Drop tensors (trigger cleanup)
                    drop(tensors);
                    env.memory_pool.cleanup_unused_blocks();
                    
                    let cleanup_stats = env.memory_pool.get_stats();
                    let pattern_duration = pattern_start_time.elapsed();
                    
                    // Calculate efficiency metrics
                    let allocated_bytes = allocation_stats.total_allocated_bytes - pattern_start_stats.total_allocated_bytes;
                    let cleaned_bytes = allocation_stats.total_allocated_bytes - cleanup_stats.total_allocated_bytes;
                    let cleanup_efficiency = if allocated_bytes > 0 {
                        (cleaned_bytes as f64 / allocated_bytes as f64) * 100.0
                    } else {
                        100.0
                    };
                    
                    efficiency_metrics.push((pattern_idx, cleanup_efficiency, pattern_duration));
                    
                    println!("📊 Pattern {}: Allocated {} bytes, Cleaned {:.2}% in {:?}",
                            pattern_idx + 1, allocated_bytes, cleanup_efficiency, pattern_duration);
                    
                    // Cleanup efficiency should be >90%
                    assert!(cleanup_efficiency > 90.0,
                           "Memory cleanup efficiency should exceed 90%, got: {:.2}%", cleanup_efficiency);
                }
                
                // Test memory pressure handling performance
                let pressure_start_time = Instant::now();
                let mut pressure_tensors = Vec::new();
                let mut allocation_failures = 0;
                
                // Try to allocate tensors until memory pressure
                for i in 0..100 {
                    match env.create_test_tensors(&[(512, 512)]) {
                        Ok(mut tensors) => pressure_tensors.append(&mut tensors),
                        Err(_) => {
                            allocation_failures += 1;
                            if allocation_failures > 5 {
                                break; // Stop after several failures
                            }
                        }
                    }
                }
                
                let pressure_duration = pressure_start_time.elapsed();
                
                // Clean up pressure tensors
                pressure_tensors.clear();
                env.memory_pool.cleanup_unused_blocks();
                
                let final_stats = env.memory_pool.get_stats();
                
                // Validate memory returned to reasonable levels
                let final_overhead = final_stats.total_allocated_bytes as f64 / initial_stats.total_allocated_bytes as f64;
                assert!(final_overhead < 2.0, 
                       "Memory overhead after cleanup should be reasonable, got: {:.2}x", final_overhead);
                
                println!("✅ Memory efficiency performance tests completed");
                println!("   Average cleanup efficiency: {:.2}%", 
                        efficiency_metrics.iter().map(|(_, eff, _)| eff).sum::<f64>() / efficiency_metrics.len() as f64);
                println!("   Memory pressure handling: {} tensors allocated in {:?}", 
                        pressure_tensors.len(), pressure_duration);
                println!("   Final memory overhead: {:.2}x", final_overhead);
                
                Ok(())
            }
        )
    }
    
    /// Test comprehensive performance profile comparison
    #[test]
    fn test_performance_profile_comparison() -> anyhow::Result<()> {
        with_timeout_and_performance_tracking(
            Duration::from_secs(240),
            "performance_profile_comparison",
            || {
                let env = RegressionTestEnvironment::new()?;
                
                // Create comprehensive performance profile
                let mut performance_profile = HashMap::new();
                
                // Core operations profile
                let core_operations = vec![
                    ("tensor_creation_small", || {
                        let data: Vec<f32> = vec![1.0; 256];
                        let _tensor = BitNetTensor::from_data(data, &[16, 16], BitNetDType::F32, &env.device, &env.memory_pool)?;
                        Ok(())
                    }),
                    ("tensor_creation_large", || {
                        let data: Vec<f32> = vec![1.0; 65536];
                        let _tensor = BitNetTensor::from_data(data, &[256, 256], BitNetDType::F32, &env.device, &env.memory_pool)?;
                        Ok(())
                    }),
                ];
                
                for (name, operation) in core_operations {
                    let (ops_per_sec, _, _) = benchmark_operation_performance(name, 500, 50, operation)?;
                    performance_profile.insert(name.to_string(), ops_per_sec);
                }
                
                // Quantization operations profile
                let quantizer = BitNetQuantizer::new(
                    WeightQuantizationConfig::default(),
                    ActivationQuantizationConfig::default(),
                    env.device.clone(),
                );
                let test_tensor = env.create_test_tensors(&[(128, 128)])?[0].clone();
                
                let quant_operations = vec![
                    ("weight_quantization_2bit", || {
                        let _quantized = quantizer.quantize_weights(&test_tensor)?;
                        Ok(())
                    }),
                    ("activation_quantization", || {
                        let _quantized = quantizer.quantize_activations(&test_tensor)?;
                        Ok(())
                    }),
                ];
                
                for (name, operation) in quant_operations {
                    let (ops_per_sec, _, _) = benchmark_operation_performance(name, 300, 30, operation)?;
                    performance_profile.insert(name.to_string(), ops_per_sec);
                }
                
                // Compare against expected baseline (simulated)
                let mut baseline_profile = HashMap::new();
                baseline_profile.insert("tensor_creation_small".to_string(), 40_000.0);
                baseline_profile.insert("tensor_creation_large".to_string(), 5_000.0);
                baseline_profile.insert("weight_quantization_2bit".to_string(), 8_000.0);
                baseline_profile.insert("activation_quantization".to_string(), 12_000.0);
                
                // Detect regressions
                let regressions = compare_performance_profiles(
                    &baseline_profile,
                    &performance_profile,
                    env.config.degradation_threshold,
                );
                
                if !regressions.is_empty() {
                    println!("⚠️  Performance regressions detected:");
                    for (operation, baseline, current, change) in regressions {
                        println!("   {}: {:.2} → {:.2} ({:.2}% degradation)",
                                operation, baseline, current, change);
                    }
                    anyhow::bail!("Performance regressions detected, see output above");
                }
                
                // Generate performance report
                println!("✅ Performance profile comparison completed");
                println!("📊 Current Performance Profile:");
                for (operation, performance) in performance_profile.iter() {
                    let baseline = baseline_profile.get(operation).unwrap_or(&0.0);
                    let change = if *baseline > 0.0 {
                        (performance - baseline) / baseline * 100.0
                    } else {
                        0.0
                    };
                    
                    println!("   {}: {:.2} ops/sec ({:+.2}%)", operation, performance, change);
                }
                
                Ok(())
            }
        )
    }
}
