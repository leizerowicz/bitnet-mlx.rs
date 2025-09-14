//! Adaptive Tensor Pool for Task 1.6.1 Performance Gap Resolution
//!
//! This module provides an adaptive tensor pool that automatically selects
//! the best allocation strategy based on tensor size and performance characteristics.
//! 
//! Key features:
//! - Automatic strategy selection based on tensor size
//! - Performance monitoring and adaptation
//! - Fallback to standard allocation for small tensors where overhead dominates
//! - Optimized allocation for large tensors where complexity pays off

use super::{MemoryError, MemoryHandle, MemoryResult, HybridMemoryPool, TensorMemoryPool, OptimizedTensorMemoryPool, OptimizedTensorPoolConfig, TensorSizeCategory};
use candle_core::Device;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::Instant;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Performance thresholds for adaptive strategy selection
const SMALL_TENSOR_THRESHOLD: usize = 32 * 1024; // 32KB
const OVERHEAD_TOLERANCE_NS: u64 = 100; // Maximum acceptable overhead in nanoseconds

/// Performance statistics for strategy selection
#[derive(Debug, Default)]
struct AdaptiveStats {
    standard_allocations: AtomicU64,
    optimized_allocations: AtomicU64,
    standard_total_time: AtomicU64,
    optimized_total_time: AtomicU64,
    adaptation_count: AtomicU64,
}

impl AdaptiveStats {
    fn record_standard_allocation(&self, duration_ns: u64) {
        self.standard_allocations.fetch_add(1, Ordering::Relaxed);
        self.standard_total_time.fetch_add(duration_ns, Ordering::Relaxed);
    }

    fn record_optimized_allocation(&self, duration_ns: u64) {
        self.optimized_allocations.fetch_add(1, Ordering::Relaxed);
        self.optimized_total_time.fetch_add(duration_ns, Ordering::Relaxed);
    }

    fn get_average_times(&self) -> (f64, f64) {
        let std_count = self.standard_allocations.load(Ordering::Relaxed);
        let opt_count = self.optimized_allocations.load(Ordering::Relaxed);
        let std_total = self.standard_total_time.load(Ordering::Relaxed);
        let opt_total = self.optimized_total_time.load(Ordering::Relaxed);

        let std_avg = if std_count > 0 { std_total as f64 / std_count as f64 } else { 0.0 };
        let opt_avg = if opt_count > 0 { opt_total as f64 / opt_count as f64 } else { 0.0 };

        (std_avg, opt_avg)
    }
}

/// Adaptive strategy for tensor allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationStrategy {
    /// Use standard tensor pool (low overhead for small tensors)
    Standard,
    /// Use optimized tensor pool (high performance for large tensors)
    Optimized,
    /// Automatic selection based on tensor characteristics
    Adaptive,
}

/// Adaptive tensor memory pool that automatically selects optimal allocation strategy
pub struct AdaptiveTensorMemoryPool {
    /// Standard tensor pool for low-overhead allocations
    standard_pool: TensorMemoryPool,
    /// Optimized tensor pool for high-performance allocations
    optimized_pool: OptimizedTensorMemoryPool,
    /// Performance statistics for strategy selection
    stats: AdaptiveStats,
    /// Current allocation strategy
    strategy: RwLock<AllocationStrategy>,
    /// Performance monitoring threshold
    performance_threshold: u64,
}

impl AdaptiveTensorMemoryPool {
    /// Create a new adaptive tensor memory pool
    pub fn new(base_pool: Arc<HybridMemoryPool>) -> MemoryResult<Self> {
        let standard_pool = TensorMemoryPool::new(base_pool.clone())?;
        
        let mut config = OptimizedTensorPoolConfig::default();
        config.enable_prewarming = true;
        config.enable_prefetching = true;
        let optimized_pool = OptimizedTensorMemoryPool::with_config(base_pool, config)?;

        Ok(Self {
            standard_pool,
            optimized_pool,
            stats: AdaptiveStats::default(),
            strategy: RwLock::new(AllocationStrategy::Adaptive),
            performance_threshold: OVERHEAD_TOLERANCE_NS,
        })
    }

    /// Create with explicit strategy
    pub fn with_strategy(base_pool: Arc<HybridMemoryPool>, strategy: AllocationStrategy) -> MemoryResult<Self> {
        let pool = Self::new(base_pool)?;
        *pool.strategy.write().unwrap() = strategy;
        Ok(pool)
    }

    /// Allocate tensor with adaptive strategy selection
    pub fn allocate_tensor_adaptive(
        &self,
        tensor_id: u64,
        size_bytes: usize,
        device: &Device,
        is_model_weight: bool,
        is_temporary: bool,
    ) -> MemoryResult<MemoryHandle> {
        let strategy = self.select_allocation_strategy(size_bytes, is_model_weight);
        
        match strategy {
            AllocationStrategy::Standard => {
                let start = Instant::now();
                let result = self.standard_pool.allocate_tensor(tensor_id, size_bytes, device, is_model_weight, is_temporary);
                let duration = start.elapsed().as_nanos() as u64;
                self.stats.record_standard_allocation(duration);
                result
            }
            AllocationStrategy::Optimized => {
                let start = Instant::now();
                let result = self.optimized_pool.allocate_tensor_optimized(tensor_id, size_bytes, device, is_model_weight, is_temporary);
                let duration = start.elapsed().as_nanos() as u64;
                self.stats.record_optimized_allocation(duration);
                result
            }
            AllocationStrategy::Adaptive => {
                // This case handled by select_allocation_strategy returning specific strategy
                unreachable!("Adaptive strategy should resolve to concrete strategy")
            }
        }
    }

    /// Deallocate tensor using the same strategy as allocation
    pub fn deallocate_tensor_adaptive(&self, tensor_id: u64, handle: MemoryHandle) -> MemoryResult<()> {
        // Try optimized pool first (it will return false if not from optimized pool)
        if self.optimized_pool.deallocate_tensor_optimized(tensor_id, handle.clone()).is_ok() {
            return Ok(());
        }

        // Fallback to standard pool
        self.standard_pool.deallocate_tensor(tensor_id, handle)
    }

    /// Select the optimal allocation strategy based on tensor characteristics
    fn select_allocation_strategy(&self, size_bytes: usize, is_model_weight: bool) -> AllocationStrategy {
        let strategy = *self.strategy.read().unwrap();

        match strategy {
            AllocationStrategy::Standard => AllocationStrategy::Standard,
            AllocationStrategy::Optimized => AllocationStrategy::Optimized,
            AllocationStrategy::Adaptive => {
                // Adaptive logic based on tensor characteristics
                if is_model_weight {
                    // Model weights benefit from optimized pool regardless of size
                    AllocationStrategy::Optimized
                } else if size_bytes <= SMALL_TENSOR_THRESHOLD {
                    // For small tensors, use standard pool to avoid overhead
                    AllocationStrategy::Standard
                } else {
                    // For medium/large temporary tensors, check performance history
                    let (std_avg, opt_avg) = self.stats.get_average_times();
                    
                    if std_avg > 0.0 && opt_avg > 0.0 {
                        // If optimized pool is significantly faster, use it
                        if std_avg > opt_avg + self.performance_threshold as f64 {
                            AllocationStrategy::Optimized
                        } else {
                            AllocationStrategy::Standard
                        }
                    } else {
                        // Default to optimized for large tensors if no history
                        AllocationStrategy::Optimized
                    }
                }
            }
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (f64, f64, u64, u64) {
        let (std_avg, opt_avg) = self.stats.get_average_times();
        let std_count = self.stats.standard_allocations.load(Ordering::Relaxed);
        let opt_count = self.stats.optimized_allocations.load(Ordering::Relaxed);
        
        (std_avg, opt_avg, std_count, opt_count)
    }

    /// Force strategy update (for testing/benchmarking)
    pub fn set_strategy(&self, strategy: AllocationStrategy) {
        *self.strategy.write().unwrap() = strategy;
        self.stats.adaptation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> AllocationStrategy {
        *self.strategy.read().unwrap()
    }

    /// Get allocation pattern recommendations
    pub fn get_allocation_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let (std_avg, opt_avg, std_count, opt_count) = self.get_performance_stats();

        if std_count > 0 && opt_count > 0 {
            if std_avg < opt_avg {
                recommendations.push(format!(
                    "Standard pool is {:.1}% faster on average ({:.0}ns vs {:.0}ns)",
                    ((opt_avg - std_avg) / std_avg) * 100.0,
                    std_avg,
                    opt_avg
                ));
                recommendations.push("Consider using standard pool for current workload".to_string());
            } else {
                recommendations.push(format!(
                    "Optimized pool is {:.1}% faster on average ({:.0}ns vs {:.0}ns)",
                    ((std_avg - opt_avg) / opt_avg) * 100.0,
                    opt_avg,
                    std_avg
                ));
                recommendations.push("Consider using optimized pool for current workload".to_string());
            }
        }

        if std_count == 0 {
            recommendations.push("No standard allocations recorded - enable for comparison".to_string());
        }
        
        if opt_count == 0 {
            recommendations.push("No optimized allocations recorded - enable for comparison".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_adaptive_pool_creation() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let _adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        Ok(())
    }

    #[test]
    fn test_strategy_selection_small_tensors() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        
        // Small tensors should use standard strategy
        let strategy = adaptive_pool.select_allocation_strategy(1024, false);
        assert_eq!(strategy, AllocationStrategy::Standard);
        
        Ok(())
    }

    #[test]
    fn test_strategy_selection_large_tensors() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        
        // Large tensors should use optimized strategy
        let strategy = adaptive_pool.select_allocation_strategy(1024 * 1024, false);
        assert_eq!(strategy, AllocationStrategy::Optimized);
        
        Ok(())
    }

    #[test]
    fn test_model_weight_optimization() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        
        // Model weights should prefer optimized pool
        let strategy = adaptive_pool.select_allocation_strategy(16 * 1024, true);
        assert_eq!(strategy, AllocationStrategy::Optimized);
        
        Ok(())
    }

    #[test]
    fn test_adaptive_allocation() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        let device = Device::Cpu;
        
        // Test small tensor allocation
        let handle1 = adaptive_pool.allocate_tensor_adaptive(1, 1024, &device, false, false)?;
        adaptive_pool.deallocate_tensor_adaptive(1, handle1)?;
        
        // Test large tensor allocation
        let handle2 = adaptive_pool.allocate_tensor_adaptive(2, 1024 * 1024, &device, false, false)?;
        adaptive_pool.deallocate_tensor_adaptive(2, handle2)?;
        
        Ok(())
    }

    #[test]
    fn test_performance_stats() -> MemoryResult<()> {
        let base_pool = Arc::new(HybridMemoryPool::new()?);
        let adaptive_pool = AdaptiveTensorMemoryPool::new(base_pool)?;
        let device = Device::Cpu;
        
        // Perform some allocations
        for i in 0..10 {
            let handle = adaptive_pool.allocate_tensor_adaptive(i, 1024, &device, false, false)?;
            adaptive_pool.deallocate_tensor_adaptive(i, handle)?;
        }
        
        let (std_avg, opt_avg, std_count, opt_count) = adaptive_pool.get_performance_stats();
        
        // Should have some allocations recorded
        assert!(std_count > 0 || opt_count > 0);
        
        Ok(())
    }
}
