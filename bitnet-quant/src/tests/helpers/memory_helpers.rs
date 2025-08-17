//! Memory testing helpers for quantization operations
//!
//! This module provides utilities for testing quantization operations
//! in conjunction with the HybridMemoryPool system.

use crate::quantization::{QuantizationResult, QuantizationError};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, MemoryHandle, MemoryResult};
use bitnet_core::device::auto_select_device;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Test harness for memory pool integration with quantization
pub struct MemoryTestHarness {
    memory_pool: Arc<HybridMemoryPool>,
    device: Device,
    allocated_handles: Vec<MemoryHandle>,
    allocation_history: Vec<AllocationRecord>,
}

#[derive(Debug, Clone)]
pub struct AllocationRecord {
    pub timestamp: Instant,
    pub size: usize,
    pub alignment: usize,
    pub handle_id: u64,
    pub operation: String,
}

impl MemoryTestHarness {
    /// Create a new memory test harness with default configuration
    pub fn new() -> QuantizationResult<Self> {
        let memory_pool = Arc::new(HybridMemoryPool::new()
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Failed to create memory pool: {:?}", e) 
            })?);
        
        let device = auto_select_device();
        
        Ok(Self {
            memory_pool,
            device,
            allocated_handles: Vec::new(),
            allocation_history: Vec::new(),
        })
    }

    /// Create a memory test harness with custom memory pool configuration
    pub fn with_config(config: MemoryPoolConfig) -> QuantizationResult<Self> {
        let memory_pool = Arc::new(HybridMemoryPool::with_config(config)
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Failed to create memory pool with config: {:?}", e) 
            })?);
        
        let device = auto_select_device();
        
        Ok(Self {
            memory_pool,
            device,
            allocated_handles: Vec::new(),
            allocation_history: Vec::new(),
        })
    }

    /// Get reference to the memory pool
    pub fn memory_pool(&self) -> &HybridMemoryPool {
        &self.memory_pool
    }

    /// Get reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Allocate memory and track the allocation
    pub fn allocate(&mut self, size: usize, alignment: usize, operation: &str) -> QuantizationResult<MemoryHandle> {
        let handle = self.memory_pool.allocate(size, alignment, &self.device)
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Memory allocation failed: {:?}", e) 
            })?;

        let record = AllocationRecord {
            timestamp: Instant::now(),
            size,
            alignment,
            handle_id: handle.id(),
            operation: operation.to_string(),
        };

        self.allocated_handles.push(handle.clone());
        self.allocation_history.push(record);

        Ok(handle)
    }

    /// Deallocate all tracked memory handles
    pub fn deallocate_all(&mut self) -> QuantizationResult<()> {
        for handle in self.allocated_handles.drain(..) {
            self.memory_pool.deallocate(handle)
                .map_err(|e| QuantizationError::InternalError { 
                    reason: format!("Memory deallocation failed: {:?}", e) 
                })?;
        }
        Ok(())
    }

    /// Check for memory leaks by comparing allocated vs deallocated
    pub fn check_for_leaks(&self) -> MemoryLeakReport {
        let metrics = self.memory_pool.get_metrics();
        
        MemoryLeakReport {
            total_allocated: metrics.total_allocated,
            peak_allocated: metrics.peak_allocated,
            current_allocated: metrics.current_allocated,
            allocation_count: metrics.allocation_count,
            deallocation_count: metrics.deallocation_count,
            has_potential_leaks: metrics.current_allocated > 0,
            tracked_handles_count: self.allocated_handles.len(),
        }
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryTestStatistics {
        let metrics = self.memory_pool.get_metrics();
        
        let total_requested: usize = self.allocation_history.iter()
            .map(|r| r.size)
            .sum();

        let avg_allocation_size = if !self.allocation_history.is_empty() {
            total_requested / self.allocation_history.len()
        } else {
            0
        };

        MemoryTestStatistics {
            total_allocations: self.allocation_history.len(),
            total_bytes_requested: total_requested,
            average_allocation_size: avg_allocation_size,
            peak_memory_usage: metrics.peak_allocated,
            current_memory_usage: metrics.current_allocated,
            fragmentation_ratio: calculate_fragmentation_ratio(&metrics),
        }
    }

    /// Reset the memory pool and clear tracking
    pub fn reset(&mut self) -> QuantizationResult<()> {
        self.allocated_handles.clear();
        self.allocation_history.clear();
        
        self.memory_pool.reset()
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Memory pool reset failed: {:?}", e) 
            })?;
        
        Ok(())
    }

    /// Run a quantization operation with memory tracking
    pub fn run_with_tracking<F, T>(&mut self, operation_name: &str, f: F) -> QuantizationResult<T>
    where
        F: FnOnce(&Device, &HybridMemoryPool) -> QuantizationResult<T>,
    {
        let initial_metrics = self.memory_pool.get_metrics();
        let start_time = Instant::now();
        
        let result = f(&self.device, &self.memory_pool)?;
        
        let final_metrics = self.memory_pool.get_metrics();
        let duration = start_time.elapsed();
        
        let memory_delta = final_metrics.current_allocated as i64 - initial_metrics.current_allocated as i64;
        
        // Record the operation
        let record = AllocationRecord {
            timestamp: start_time,
            size: memory_delta.abs() as usize,
            alignment: 0, // Not applicable for tracked operations
            handle_id: 0, // Not applicable for tracked operations
            operation: format!("{} (duration: {:?}, memory_delta: {})", 
                             operation_name, duration, memory_delta),
        };
        
        self.allocation_history.push(record);
        
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLeakReport {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_allocated: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub has_potential_leaks: bool,
    pub tracked_handles_count: usize,
}

impl MemoryLeakReport {
    pub fn is_clean(&self) -> bool {
        !self.has_potential_leaks && 
        self.tracked_handles_count == 0 &&
        self.allocation_count == self.deallocation_count
    }
}

#[derive(Debug, Clone)]
pub struct MemoryTestStatistics {
    pub total_allocations: usize,
    pub total_bytes_requested: usize,
    pub average_allocation_size: usize,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
    pub fragmentation_ratio: f64,
}

/// Test memory pool specifically configured for quantization testing
pub struct TestMemoryPool {
    pool: HybridMemoryPool,
    config: MemoryPoolConfig,
}

impl TestMemoryPool {
    /// Create a memory pool optimized for testing
    pub fn for_testing() -> QuantizationResult<Self> {
        let config = MemoryPoolConfig {
            small_block_threshold: 64 * 1024,    // 64KB threshold
            small_pool_initial_size: 1024 * 1024, // 1MB small pool
            large_pool_initial_size: 16 * 1024 * 1024, // 16MB large pool
            enable_metrics: true,
            enable_debug_logging: true,
            enable_advanced_tracking: true,
            ..Default::default()
        };

        let pool = HybridMemoryPool::with_config(config.clone())
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Failed to create test memory pool: {:?}", e) 
            })?;

        Ok(Self { pool, config })
    }

    /// Create a memory pool with stress testing configuration
    pub fn for_stress_testing() -> QuantizationResult<Self> {
        let config = MemoryPoolConfig {
            small_block_threshold: 32 * 1024,    // 32KB threshold
            small_pool_initial_size: 512 * 1024,  // 512KB small pool
            large_pool_initial_size: 8 * 1024 * 1024, // 8MB large pool
            small_pool_max_size: 2 * 1024 * 1024,     // 2MB max small
            large_pool_max_size: 64 * 1024 * 1024,    // 64MB max large
            enable_metrics: true,
            enable_advanced_tracking: true,
            ..Default::default()
        };

        let pool = HybridMemoryPool::with_config(config.clone())
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Failed to create stress test memory pool: {:?}", e) 
            })?;

        Ok(Self { pool, config })
    }

    /// Get reference to the underlying pool
    pub fn pool(&self) -> &HybridMemoryPool {
        &self.pool
    }

    /// Get the pool configuration
    pub fn config(&self) -> &MemoryPoolConfig {
        &self.config
    }
}

/// Concurrent memory test harness for thread-safety testing
pub struct ConcurrentMemoryTestHarness {
    memory_pool: Arc<HybridMemoryPool>,
    device: Device,
}

impl ConcurrentMemoryTestHarness {
    pub fn new() -> QuantizationResult<Self> {
        let memory_pool = Arc::new(HybridMemoryPool::new()
            .map_err(|e| QuantizationError::InternalError { 
                reason: format!("Failed to create concurrent memory pool: {:?}", e) 
            })?);
        
        let device = auto_select_device();
        
        Ok(Self { memory_pool, device })
    }

    pub fn memory_pool(&self) -> Arc<HybridMemoryPool> {
        self.memory_pool.clone()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run concurrent memory operations
    pub fn run_concurrent_test<F>(&self, num_threads: usize, operations_per_thread: usize, test_fn: F) -> QuantizationResult<ConcurrentTestResults>
    where
        F: Fn(Arc<HybridMemoryPool>, Device, usize) -> QuantizationResult<Vec<Duration>> + Send + Sync + 'static,
    {
        use std::thread;
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        let test_fn = Arc::new(test_fn);

        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let pool = self.memory_pool.clone();
            let device = self.device.clone();
            let tx = tx.clone();
            let test_fn = test_fn.clone();

            let handle = thread::spawn(move || {
                let result = test_fn(pool, device, thread_id);
                tx.send((thread_id, result)).unwrap();
            });

            handles.push(handle);
        }

        drop(tx); // Close the sender

        // Collect results
        let mut thread_results = HashMap::new();
        for _ in 0..num_threads {
            let (thread_id, result) = rx.recv()
                .map_err(|e| QuantizationError::InternalError { 
                    reason: format!("Failed to receive thread result: {}", e) 
                })?;
            thread_results.insert(thread_id, result);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join()
                .map_err(|e| QuantizationError::InternalError { 
                    reason: format!("Thread join failed: {:?}", e) 
                })?;
        }

        // Analyze results
        let mut all_durations = Vec::new();
        let mut successful_threads = 0;
        let mut failed_threads = 0;

        for (_, result) in thread_results {
            match result {
                Ok(durations) => {
                    all_durations.extend(durations);
                    successful_threads += 1;
                }
                Err(_) => {
                    failed_threads += 1;
                }
            }
        }

        let avg_duration = if !all_durations.is_empty() {
            all_durations.iter().sum::<Duration>() / all_durations.len() as u32
        } else {
            Duration::from_secs(0)
        };

        Ok(ConcurrentTestResults {
            num_threads,
            operations_per_thread,
            successful_threads,
            failed_threads,
            total_operations: all_durations.len(),
            average_operation_duration: avg_duration,
            memory_metrics: self.memory_pool.get_metrics(),
        })
    }
}

#[derive(Debug)]
pub struct ConcurrentTestResults {
    pub num_threads: usize,
    pub operations_per_thread: usize,
    pub successful_threads: usize,
    pub failed_threads: usize,
    pub total_operations: usize,
    pub average_operation_duration: Duration,
    pub memory_metrics: bitnet_core::memory::MemoryMetrics,
}

impl ConcurrentTestResults {
    pub fn success_rate(&self) -> f64 {
        if self.num_threads == 0 { 0.0 } 
        else { self.successful_threads as f64 / self.num_threads as f64 }
    }

    pub fn is_successful(&self) -> bool {
        self.failed_threads == 0 && self.successful_threads == self.num_threads
    }
}

// Helper function to calculate fragmentation ratio
fn calculate_fragmentation_ratio(metrics: &bitnet_core::memory::MemoryMetrics) -> f64 {
    if metrics.peak_allocated > 0 {
        1.0 - (metrics.current_allocated as f64 / metrics.peak_allocated as f64)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_test_harness_creation() {
        let harness = MemoryTestHarness::new().unwrap();
        assert_eq!(harness.allocated_handles.len(), 0);
        assert_eq!(harness.allocation_history.len(), 0);
    }

    #[test]
    fn test_allocation_and_tracking() {
        let mut harness = MemoryTestHarness::new().unwrap();
        
        let handle = harness.allocate(1024, 16, "test_allocation").unwrap();
        assert_eq!(harness.allocated_handles.len(), 1);
        assert_eq!(harness.allocation_history.len(), 1);
        
        let stats = harness.get_memory_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_requested, 1024);
    }

    #[test] 
    fn test_memory_leak_detection() {
        let mut harness = MemoryTestHarness::new().unwrap();
        
        // Allocate some memory
        let _handle = harness.allocate(2048, 32, "leak_test").unwrap();
        
        let leak_report = harness.check_for_leaks();
        assert!(leak_report.has_potential_leaks);
        assert_eq!(leak_report.tracked_handles_count, 1);
        
        // Clean up
        harness.deallocate_all().unwrap();
        
        let clean_report = harness.check_for_leaks();
        assert_eq!(clean_report.tracked_handles_count, 0);
    }

    #[test]
    fn test_concurrent_memory_harness() {
        let harness = ConcurrentMemoryTestHarness::new().unwrap();
        
        // Simple test function that allocates and deallocates memory
        let test_fn = |pool: Arc<HybridMemoryPool>, device: Device, _thread_id: usize| -> QuantizationResult<Vec<Duration>> {
            let mut durations = Vec::new();
            
            for _ in 0..10 {
                let start = Instant::now();
                let handle = pool.allocate(1024, 16, &device)
                    .map_err(|e| QuantizationError::InternalError { 
                        reason: format!("Allocation failed: {:?}", e) 
                    })?;
                pool.deallocate(handle)
                    .map_err(|e| QuantizationError::InternalError { 
                        reason: format!("Deallocation failed: {:?}", e) 
                    })?;
                durations.push(start.elapsed());
            }
            
            Ok(durations)
        };

        let results = harness.run_concurrent_test(4, 10, test_fn).unwrap();
        
        assert_eq!(results.num_threads, 4);
        assert_eq!(results.operations_per_thread, 10);
        assert!(results.is_successful());
        assert_eq!(results.total_operations, 40); // 4 threads * 10 operations
    }
}
