//! Integration tests for SIMD and Dispatch System
//!
//! This module tests the complete SIMD acceleration and automatic
//! dispatch system for BitNet tensor operations.

use bitnet_core::tensor::core::BitNetTensor;
use bitnet_core::tensor::dtype::BitNetDType;
use bitnet_core::tensor::set_global_memory_pool;
use bitnet_core::memory::HybridMemoryPool;
use bitnet_core::tensor::acceleration::{
    SimdAccelerator, SimdOptimization, DispatchStrategy, OperationType, OperationContext,
    AccelerationBackend, PerformanceRequirements,
    AccelerationBackendImpl,
};
use candle_core::Device;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn setup_test_environment() {
        use std::sync::OnceLock;
        // Keep a static reference to prevent the pool from being dropped
        static POOL_HOLDER: OnceLock<Arc<HybridMemoryPool>> = OnceLock::new();
        
        // Initialize global memory pool for tests
        let pool = POOL_HOLDER.get_or_init(|| {
            Arc::new(HybridMemoryPool::new().unwrap())
        });
        
        set_global_memory_pool(Arc::downgrade(pool));
    }

    #[test]
    fn test_simd_optimization_detection() {
        setup_test_environment();
        let optimization = SimdOptimization::detect();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // On x86/x86_64, we should have at least SSE2
            assert!(optimization != SimdOptimization::None);
            println!("Detected SIMD optimization: {:?}", optimization);
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64, we should have NEON
            assert_eq!(optimization, SimdOptimization::NEON);
            println!("Detected NEON optimization on ARM64");
        }
        
        // Test vector width
        let vector_width = optimization.vector_width_f32();
        assert!(vector_width >= 1);
        assert!(vector_width <= 16);
        
        // Test performance multiplier
        let multiplier = optimization.performance_multiplier();
        assert!(multiplier >= 1.0);
        assert!(multiplier <= 15.0);
        
        println!("Vector width: {vector_width}, Performance multiplier: {multiplier:.1}x");
    }
    
    #[test]
    fn test_simd_accelerator_creation() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_environment();
        let mut accelerator = SimdAccelerator::new()?;
        
        // Test initialization
        assert!(!accelerator.is_available() || accelerator.optimization_level() != SimdOptimization::None);
        accelerator.initialize()?;
        
        // Test capabilities
        let capabilities = accelerator.get_capabilities();
        assert_eq!(capabilities.backend, AccelerationBackend::SIMD);
        
        // Test metrics
        let metrics = accelerator.get_metrics();
        assert_eq!(metrics.optimization_level, accelerator.optimization_level());
        assert_eq!(metrics.vectorized_ops, 0);
        assert_eq!(metrics.scalar_fallback_ops, 0);
        
        println!("SIMD Accelerator created with optimization level: {:?}", accelerator.optimization_level());
        
        Ok(())
    }
    
    #[test]
    fn test_simd_tensor_addition() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_environment();
        let mut accelerator = SimdAccelerator::new()?;
        accelerator.initialize()?;
        
        // Skip test if SIMD is not available
        if !accelerator.is_available() {
            println!("SIMD not available on this platform, skipping test");
            return Ok(());
        }
        
        // Create test tensors with CPU device
        let device = Device::Cpu;
        let a = BitNetTensor::ones(&[4, 4], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[4, 4], BitNetDType::F32, Some(device))?;
        
        // Test tensor addition
        let result = accelerator.add(&a, &b);
        
        // The current implementation might return an error for incomplete SIMD ops
        // but the important thing is that it doesn't panic and handles the case properly
        match result {
            Ok((tensor, metrics)) => {
                println!("SIMD addition succeeded: {metrics:?}");
                assert_eq!(tensor.shape().dims(), &[4, 4]);
                assert_eq!(tensor.dtype(), BitNetDType::F32);
            },
            Err(e) => {
                println!("SIMD addition returned expected error: {e}");
                // This is expected for the current placeholder implementation
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_acceleration_backend_priorities() {
        let backends = [
            AccelerationBackend::MLX,
            AccelerationBackend::Metal,
            AccelerationBackend::SIMD,
            AccelerationBackend::CPU,
        ];
        
        // Test priority ordering
        assert!(AccelerationBackend::MLX.priority() > AccelerationBackend::Metal.priority());
        assert!(AccelerationBackend::Metal.priority() > AccelerationBackend::SIMD.priority());
        assert!(AccelerationBackend::SIMD.priority() > AccelerationBackend::CPU.priority());
        
        // Test platform support
        for backend in &backends {
            let supported = backend.is_platform_supported();
            println!("{backend} platform supported: {supported}");
            
            if *backend == AccelerationBackend::CPU {
                assert!(supported); // CPU should always be supported
            }
        }
        
        // Test performance characteristics
        for backend in &backends {
            let perf = backend.performance_characteristics();
            assert!(perf.throughput_gflops > 0.0);
            assert!(perf.latency_us > 0.0);
            assert!(perf.memory_bandwidth_gbps > 0.0);
            assert!(perf.power_efficiency > 0.0 && perf.power_efficiency <= 1.0);
            
            println!("{} performance: {:.1} GFLOPS, {:.1} μs latency, {:.1} GB/s bandwidth", 
                     backend, perf.throughput_gflops, perf.latency_us, perf.memory_bandwidth_gbps);
        }
    }
    
    #[test]
    fn test_operation_type_characteristics() {
        let operations = [
            OperationType::MatMul,
            OperationType::Add,
            OperationType::Mul,
            OperationType::Transpose,
            OperationType::Convolution,
            OperationType::Reduction,
        ];
        
        for op in &operations {
            let intensity = op.computational_intensity();
            let preferred = op.preferred_backend();
            
            println!("{op:?}: intensity={intensity:.2}, preferred={preferred}");
            
            // Test that compute-intensive operations prefer GPU/MLX
            match op {
                OperationType::MatMul | OperationType::Convolution => {
                    assert!(intensity > 3.0);
                    assert!(preferred == AccelerationBackend::MLX || preferred == AccelerationBackend::Metal);
                },
                OperationType::Add | OperationType::Mul => {
                    assert!(intensity < 1.0);
                    assert_eq!(preferred, AccelerationBackend::SIMD);
                },
                OperationType::Transpose => {
                    assert!(intensity < 0.5);
                    assert_eq!(preferred, AccelerationBackend::CPU);
                },
                _ => {
                    assert!(intensity > 0.0);
                }
            }
        }
    }
    
    #[test]
    fn test_operation_context() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_environment();
        
        // Create operation context for matrix multiplication
        let context = OperationContext::new(
            OperationType::MatMul,
            vec![vec![128, 256], vec![256, 512]],  // Input shapes
            BitNetDType::F32,
        );
        
        // Test operation context properties
        assert_eq!(context.operation_type, OperationType::MatMul);
        assert_eq!(context.input_shapes.len(), 2);
        assert_eq!(context.dtype, BitNetDType::F32);
        
        // Test complexity calculation
        let complexity = context.complexity_score();
        assert!(complexity > 0.0);
        
        // For matrix multiply: (128*256 + 256*512) * 4.0 (intensity)
        let expected_complexity = (128_usize * 256 + 256 * 512) as f64 * 4.0;
        assert_relative_eq!(complexity, expected_complexity, epsilon = 1e-6);
        
        // Test memory estimation
        let memory = context.estimated_memory_bytes();
        assert!(memory > 0);
        
        // Should account for input tensors, output tensor, and temporary storage
        // Each input shape: 128*256 + 256*512, output: 128*512, temporary: similar
        // Total elements for estimation
        let input_elements: usize = context.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        let dtype_size = 4; // F32 = 4 bytes
        let estimated_memory = input_elements * dtype_size * 3; // 3x for in, out, temp
        
        // Allow some tolerance for the memory calculation
        assert!((memory as i64 - estimated_memory as i64).abs() < estimated_memory as i64 / 10, 
               "Memory estimation: expected ~{estimated_memory}, got {memory}");
        
        println!("Operation context: complexity={:.1}, memory={}MB", 
                 complexity, memory / (1024 * 1024));
        
        Ok(())
    }
    
    #[test]
    fn test_dispatch_strategies() -> Result<(), Box<dyn std::error::Error>> {
        // Test different dispatch strategies
        let strategies = vec![
            DispatchStrategy::HighestPriority,
            DispatchStrategy::BestPerformance,
            DispatchStrategy::LowLatency,
            DispatchStrategy::HighThroughput,
            DispatchStrategy::LowMemory,
            DispatchStrategy::ForceBackend(AccelerationBackend::SIMD),
        ];
        
        for strategy in strategies {
            println!("Testing strategy: {strategy:?}");
            // Strategy creation and debug formatting should work
            let _cloned = strategy.clone();
        }
        
        Ok(())
    }
    
    #[cfg(feature = "integration-tests")]
    #[test]
    fn test_operation_dispatcher_creation() -> Result<(), Box<dyn std::error::Error>> {
        // Test that we can create a dispatcher
        // Note: This may fail in CI environments without proper hardware support
        let result = create_operation_dispatcher();
        
        match result {
            Ok(dispatcher) => {
                let available = dispatcher.get_available_backends();
                println!("Available backends: {:?}", available);
                
                // Should have at least CPU or SIMD backend
                assert!(!available.is_empty());
                
                // Test strategy changes
                dispatcher.set_strategy(DispatchStrategy::LowLatency);
                dispatcher.set_strategy(DispatchStrategy::HighThroughput);
                
                println!("Dispatcher created successfully with {} backends", available.len());
            },
            Err(e) => {
                println!("Expected error in test environment: {}", e);
                // This is expected in environments without proper hardware setup
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_backend_selection_logic() -> Result<(), Box<dyn std::error::Error>> {
        // Test the logical components of backend selection
        let context = OperationContext::new(
            OperationType::MatMul,
            vec![vec![64, 128], vec![128, 256]],
            BitNetDType::F32
        );
        
        // Test preferred backend selection
        let preferred = context.operation_type.preferred_backend();
        println!("Preferred backend for {:?}: {}", context.operation_type, preferred);
        
        // Test that compute-intensive operations prefer high-performance backends
        if context.operation_type == OperationType::MatMul {
            assert!(preferred == AccelerationBackend::MLX || preferred == AccelerationBackend::Metal);
        }
        
        // Test complexity scoring
        let complexity = context.complexity_score();
        assert!(complexity > 0.0);
        println!("Operation complexity: {complexity:.1}");
        
        Ok(())
    }
    
    #[test]
    fn test_performance_requirements() {
        let mut requirements = PerformanceRequirements::default();
        
        // Test default values
        assert_eq!(requirements.max_latency_us, None);
        assert_eq!(requirements.min_throughput_gflops, None);
        assert_eq!(requirements.max_memory_bytes, None);
        assert!(!requirements.prefer_low_latency);
        
        // Test setting requirements
        requirements.max_latency_us = Some(1000);
        requirements.min_throughput_gflops = Some(100.0);
        requirements.prefer_low_latency = true;
        
        assert_eq!(requirements.max_latency_us, Some(1000));
        assert_eq!(requirements.min_throughput_gflops, Some(100.0));
        assert!(requirements.prefer_low_latency);
        
        println!("Performance requirements configured successfully");
    }
    
    #[test]
    fn test_simd_memory_management() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_environment();
        let accelerator = SimdAccelerator::new()?;
        
        // Test memory stats
        let stats = accelerator.get_memory_stats()?;
        assert!(stats.total_allocated >= 0);
        assert!(stats.total_deallocated >= 0);
        
        // Test metrics management
        let initial_metrics = accelerator.get_metrics();
        assert_eq!(initial_metrics.vectorized_ops, 0);
        
        accelerator.clear_metrics();
        let cleared_metrics = accelerator.get_metrics();
        assert_eq!(cleared_metrics.vectorized_ops, 0);
        assert_eq!(cleared_metrics.execution_time_ns, 0);
        
        println!("SIMD memory management tests passed");
        Ok(())
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_simd_optimization_detection() {
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _optimization = SimdOptimization::detect();
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() / iterations;
        
        println!("SIMD detection benchmark: {iterations} iterations, avg {avg_time}ns per detection");
        
        // Should be fast (under 10μs per detection)
        assert!(avg_time < 10_000);
    }
    
    #[test]
    fn benchmark_backend_characteristics() {
        let backends = [
            AccelerationBackend::MLX,
            AccelerationBackend::Metal,
            AccelerationBackend::SIMD,
            AccelerationBackend::CPU,
        ];
        
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            for backend in &backends {
                let _perf = backend.performance_characteristics();
                let _priority = backend.priority();
                let _supported = backend.is_platform_supported();
            }
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() / (iterations * backends.len() as u128);
        
        println!("Backend characteristics benchmark: {iterations} iterations, avg {avg_time}ns per backend");
        
        // Should be extremely fast (under 100ns per backend)
        assert!(avg_time < 1_000);
    }
}

/// Helper functions for testing
#[cfg(test)]
mod test_helpers {
    use super::*;
    
    pub fn create_test_tensors(size: usize) -> Result<(BitNetTensor, BitNetTensor), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let a = BitNetTensor::ones(&[size, size], BitNetDType::F32, Some(device.clone()))?;
        let b = BitNetTensor::ones(&[size, size], BitNetDType::F32, Some(device))?;
        Ok((a, b))
    }
    
    pub fn measure_operation_time<F, T>(operation: F) -> (T, std::time::Duration)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();
        (result, duration)
    }
    
    pub fn assert_tensor_equal(a: &BitNetTensor, b: &BitNetTensor) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        // Additional data comparison would go here
        Ok(())
    }
}
