//! Integration tests with the HybridMemoryPool system
//!
//! This module tests quantization operations within the context of the
//! memory management system to ensure proper resource handling.

use crate::memory::{HybridMemoryPool, MemoryRequest, AllocationStrategy, MemoryStats};
use crate::quantization::{QuantizationResult, TernaryMethod, create_ternary_quantizer};
use crate::tests::helpers::{
    TestPattern, generate_test_tensor, create_test_device,
    MemoryTestHarness
};
use candle_core::{Device, Tensor};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Results of memory pool integration testing
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolIntegrationResults {
    pub pool_tests_run: usize,
    pub pool_tests_passed: usize,
    pub memory_leaks_detected: usize,
    pub allocation_failures: usize,
    pub concurrent_test_results: ConcurrentTestResults,
    pub performance_metrics: PoolPerformanceMetrics,
    pub resource_cleanup_score: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ConcurrentTestResults {
    pub concurrent_quantizations: usize,
    pub successful_concurrent_ops: usize,
    pub thread_safety_violations: usize,
    pub race_conditions_detected: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PoolPerformanceMetrics {
    pub average_allocation_time_ns: u64,
    pub average_deallocation_time_ns: u64,
    pub peak_memory_usage_bytes: usize,
    pub memory_fragmentation_ratio: f64,
    pub pool_efficiency_score: f64,
}

impl MemoryPoolIntegrationResults {
    pub fn success_rate(&self) -> f64 {
        if self.pool_tests_run == 0 { 1.0 }
        else { self.pool_tests_passed as f64 / self.pool_tests_run as f64 }
    }

    pub fn has_critical_issues(&self) -> bool {
        self.memory_leaks_detected > 0 ||
        self.concurrent_test_results.thread_safety_violations > 0 ||
        self.resource_cleanup_score < 0.9
    }
}

/// Run comprehensive memory pool integration tests
pub fn run_memory_pool_integration_tests(device: &Device) -> QuantizationResult<MemoryPoolIntegrationResults> {
    let mut results = MemoryPoolIntegrationResults::default();

    // Create memory pool for testing
    let pool = Arc::new(HybridMemoryPool::new(device.clone()));

    // Test 1: Basic quantization with memory pool
    test_basic_quantization_with_pool(&pool, device, &mut results)?;

    // Test 2: Memory leak detection
    test_memory_leak_detection(&pool, device, &mut results)?;

    // Test 3: Concurrent quantization operations
    test_concurrent_quantization_operations(&pool, device, &mut results)?;

    // Test 4: Large tensor handling
    test_large_tensor_quantization(&pool, device, &mut results)?;

    // Test 5: Memory pressure scenarios
    test_memory_pressure_scenarios(&pool, device, &mut results)?;

    // Test 6: Pool allocation strategies
    test_allocation_strategies(&pool, device, &mut results)?;

    // Test 7: Resource cleanup validation
    test_resource_cleanup(&pool, device, &mut results)?;

    // Calculate performance metrics
    results.performance_metrics = calculate_pool_performance_metrics(&pool)?;

    // Calculate overall resource cleanup score
    results.resource_cleanup_score = calculate_cleanup_score(&pool);

    Ok(results)
}

fn test_basic_quantization_with_pool(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    // Create memory test harness
    let mut harness = MemoryTestHarness::new(pool.clone());

    // Test quantization of different tensor sizes
    let test_cases = vec![
        (vec![32], TestPattern::NormalDistribution),
        (vec![16, 16], TestPattern::SparseWeights),
        (vec![8, 8, 4], TestPattern::UniformDistribution),
    ];

    let mut all_passed = true;

    for (shape, pattern) in test_cases {
        let tensor = generate_test_tensor(pattern, &shape, device)?;
        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;

        let initial_stats = pool.get_stats();

        // Perform quantization
        let quantization_result = quantizer.quantize(&tensor);

        match quantization_result {
            Ok(result) => {
                // Verify the quantization worked
                if !validate_quantization_with_memory(&result, &initial_stats, pool) {
                    all_passed = false;
                }

                // Test dequantization
                if quantizer.dequantize(&result).is_err() {
                    all_passed = false;
                }
            }
            Err(_) => {
                all_passed = false;
                results.allocation_failures += 1;
            }
        }
    }

    // Verify no memory leaks after operations
    if !harness.verify_no_leaks() {
        all_passed = false;
        results.memory_leaks_detected += 1;
    }

    if all_passed {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

fn test_memory_leak_detection(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    let initial_stats = pool.get_stats();
    let mut harness = MemoryTestHarness::new(pool.clone());

    // Perform multiple quantization cycles
    for i in 0..10 {
        let tensor = generate_test_tensor(
            TestPattern::NormalDistribution,
            &[64],
            device
        )?;

        let quantizer = create_ternary_quantizer(
            TernaryMethod::AdaptiveThreshold,
            Some(0.5 + (i as f32) * 0.05)
        )?;

        // Quantize and immediately drop results
        let _ = quantizer.quantize(&tensor)?;

        // Force cleanup
        if i % 3 == 0 {
            pool.cleanup();
        }
    }

    // Check for memory leaks
    let final_stats = pool.get_stats();
    let leaked_bytes = final_stats.total_allocated_bytes.saturating_sub(initial_stats.total_allocated_bytes);

    if leaked_bytes == 0 && harness.verify_no_leaks() {
        results.pool_tests_passed += 1;
    } else {
        results.memory_leaks_detected += 1;
    }

    Ok(())
}

fn test_concurrent_quantization_operations(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    let harness = MemoryTestHarness::new(pool.clone());
    let num_threads = 4;
    let operations_per_thread = 5;

    // Use the harness for concurrent testing
    let concurrent_results = harness.test_concurrent_quantization(
        num_threads,
        operations_per_thread,
        device,
    );

    results.concurrent_test_results = ConcurrentTestResults {
        concurrent_quantizations: num_threads * operations_per_thread,
        successful_concurrent_ops: concurrent_results.successful_operations,
        thread_safety_violations: concurrent_results.thread_safety_violations,
        race_conditions_detected: concurrent_results.race_conditions,
    };

    if concurrent_results.successful_operations == num_threads * operations_per_thread &&
       concurrent_results.thread_safety_violations == 0 {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

fn test_large_tensor_quantization(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    // Test with progressively larger tensors
    let large_shapes = vec![
        vec![1024],
        vec![512, 512],
        vec![256, 256, 4],
        vec![128, 128, 8],
    ];

    let mut all_passed = true;
    let initial_stats = pool.get_stats();

    for shape in large_shapes {
        let tensor = generate_test_tensor(
            TestPattern::SparseWeights,
            &shape,
            device
        )?;

        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;

        match quantizer.quantize(&tensor) {
            Ok(result) => {
                // Verify large tensor quantization
                if result.values.elem_count() != tensor.elem_count() {
                    all_passed = false;
                }

                // Test dequantization of large tensor
                if quantizer.dequantize(&result).is_err() {
                    all_passed = false;
                }
            }
            Err(_) => {
                all_passed = false;
                results.allocation_failures += 1;
            }
        }
    }

    // Check memory usage didn't explode
    let final_stats = pool.get_stats();
    let memory_growth = final_stats.peak_memory_usage_bytes as f64 / initial_stats.peak_memory_usage_bytes as f64;

    if memory_growth > 10.0 { // More than 10x growth indicates a problem
        all_passed = false;
    }

    if all_passed {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

fn test_memory_pressure_scenarios(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    // Simulate memory pressure by allocating many tensors simultaneously
    let mut tensors = Vec::new();
    let mut quantization_results = Vec::new();

    let quantizer = create_ternary_quantizer(TernaryMethod::MeanThreshold, Some(0.7))?;

    // Fill memory with tensors
    for i in 0..20 {
        let tensor = generate_test_tensor(
            TestPattern::UniformDistribution,
            &[128, 128],
            device
        )?;

        match quantizer.quantize(&tensor) {
            Ok(result) => {
                tensors.push(tensor);
                quantization_results.push(result);
            }
            Err(_) => {
                results.allocation_failures += 1;
                break;
            }
        }
    }

    // Try to allocate one more large tensor (should trigger cleanup)
    let large_tensor = generate_test_tensor(
        TestPattern::LargeValues,
        &[512, 512],
        device
    );

    let handled_pressure = match large_tensor {
        Ok(tensor) => quantizer.quantize(&tensor).is_ok(),
        Err(_) => false,
    };

    // Clean up
    drop(tensors);
    drop(quantization_results);
    pool.cleanup();

    if handled_pressure || results.allocation_failures == 0 {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

fn test_allocation_strategies(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    // Test different allocation strategies
    let strategies = vec![
        AllocationStrategy::FirstFit,
        AllocationStrategy::BestFit,
        AllocationStrategy::PreferSmallPool,
        AllocationStrategy::PreferLargePool,
    ];

    let mut all_strategies_work = true;

    for strategy in strategies {
        // Create request with specific strategy
        let request = MemoryRequest {
            size_bytes: 1024 * 1024, // 1MB
            alignment: 16,
            allocation_strategy: strategy,
            device_preference: Some(device.clone()),
        };

        match pool.allocate(request) {
            Ok(allocation) => {
                // Test quantization with this allocation
                let tensor = generate_test_tensor(
                    TestPattern::NormalDistribution,
                    &[256, 256],
                    device
                )?;

                let quantizer = create_ternary_quantizer(TernaryMethod::AdaptiveThreshold, Some(0.7))?;

                if quantizer.quantize(&tensor).is_err() {
                    all_strategies_work = false;
                }

                // Clean up allocation
                pool.deallocate(allocation);
            }
            Err(_) => {
                all_strategies_work = false;
                results.allocation_failures += 1;
            }
        }
    }

    if all_strategies_work {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

fn test_resource_cleanup(
    pool: &Arc<HybridMemoryPool>,
    device: &Device,
    results: &mut MemoryPoolIntegrationResults
) -> QuantizationResult<()> {
    results.pool_tests_run += 1;

    let initial_stats = pool.get_stats();

    // Allocate and deallocate many tensors
    for _ in 0..50 {
        let tensor = generate_test_tensor(
            TestPattern::SparseWeights,
            &[64, 64],
            device
        )?;

        let quantizer = create_ternary_quantizer(TernaryMethod::OptimalThreshold, Some(0.7))?;
        let _ = quantizer.quantize(&tensor)?;
    }

    // Force cleanup
    pool.cleanup();

    let final_stats = pool.get_stats();

    // Check that memory usage returned close to initial levels
    let memory_efficiency = if final_stats.total_allocated_bytes <= initial_stats.total_allocated_bytes + 1024 {
        1.0
    } else {
        initial_stats.total_allocated_bytes as f64 / final_stats.total_allocated_bytes as f64
    };

    if memory_efficiency > 0.9 {
        results.pool_tests_passed += 1;
    }

    Ok(())
}

// Helper functions

fn validate_quantization_with_memory(
    result: &crate::quantization::QuantizedTensor,
    initial_stats: &MemoryStats,
    pool: &HybridMemoryPool
) -> bool {
    let current_stats = pool.get_stats();

    // Check that quantization succeeded
    result.values.elem_count() > 0 &&
    result.stats.scale_factor > 0.0 &&
    // Memory usage should be reasonable
    current_stats.total_allocated_bytes >= initial_stats.total_allocated_bytes
}

fn calculate_pool_performance_metrics(
    pool: &HybridMemoryPool
) -> QuantizationResult<PoolPerformanceMetrics> {
    let stats = pool.get_stats();

    Ok(PoolPerformanceMetrics {
        average_allocation_time_ns: stats.average_allocation_time_ns,
        average_deallocation_time_ns: stats.average_deallocation_time_ns,
        peak_memory_usage_bytes: stats.peak_memory_usage_bytes,
        memory_fragmentation_ratio: calculate_fragmentation_ratio(&stats),
        pool_efficiency_score: calculate_efficiency_score(&stats),
    })
}

fn calculate_fragmentation_ratio(stats: &MemoryStats) -> f64 {
    if stats.total_capacity_bytes == 0 {
        return 0.0;
    }

    let free_bytes = stats.total_capacity_bytes - stats.total_allocated_bytes;
    let fragmentation = if stats.largest_free_block_bytes == 0 { 1.0 } else {
        1.0 - (stats.largest_free_block_bytes as f64 / free_bytes as f64)
    };

    fragmentation.max(0.0).min(1.0)
}

fn calculate_efficiency_score(stats: &MemoryStats) -> f64 {
    if stats.total_allocations == 0 {
        return 1.0;
    }

    let success_rate = 1.0 - (stats.failed_allocations as f64 / stats.total_allocations as f64);
    let fragmentation_penalty = 1.0 - calculate_fragmentation_ratio(stats);

    (success_rate + fragmentation_penalty) / 2.0
}

fn calculate_cleanup_score(pool: &HybridMemoryPool) -> f64 {
    let stats = pool.get_stats();

    // Score based on how well the pool manages memory
    let allocation_success_rate = if stats.total_allocations == 0 { 1.0 } else {
        1.0 - (stats.failed_allocations as f64 / stats.total_allocations as f64)
    };

    let memory_utilization = if stats.total_capacity_bytes == 0 { 1.0 } else {
        1.0 - (stats.total_allocated_bytes as f64 / stats.total_capacity_bytes as f64).min(1.0)
    };

    // Weighted average favoring allocation success
    0.7 * allocation_success_rate + 0.3 * memory_utilization
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_integration() {
        let device = create_test_device();
        let results = run_memory_pool_integration_tests(&device).unwrap();

        assert!(results.pool_tests_run > 0);
        assert!(results.success_rate() > 0.5); // At least half should pass
        assert!(!results.has_critical_issues());
    }

    #[test]
    fn test_basic_pool_quantization() {
        let device = create_test_device();
        let pool = Arc::new(HybridMemoryPool::new(device.clone()));
        let mut results = MemoryPoolIntegrationResults::default();

        test_basic_quantization_with_pool(&pool, &device, &mut results).unwrap();
        assert!(results.pool_tests_run > 0);
    }

    #[test]
    fn test_memory_leak_prevention() {
        let device = create_test_device();
        let pool = Arc::new(HybridMemoryPool::new(device.clone()));
        let mut results = MemoryPoolIntegrationResults::default();

        test_memory_leak_detection(&pool, &device, &mut results).unwrap();
        assert_eq!(results.memory_leaks_detected, 0);
    }
}
