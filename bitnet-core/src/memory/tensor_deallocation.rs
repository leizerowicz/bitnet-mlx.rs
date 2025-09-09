//! Tensor deallocation pattern optimization
//!
//! This module provides intelligent deallocation patterns for tensors,
//! including batch deallocation, deferred cleanup, and priority-based
//! deallocation strategies.

use super::{MemoryError, MemoryHandle, MemoryResult, TensorMemoryPool};
use crate::memory::tracking::MemoryPressureLevel;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::cmp::Ordering;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Priority for tensor deallocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeallocationPriority {
    /// Immediate deallocation - critical tensors
    Immediate = 0,
    /// High priority - important but not critical
    High = 1,
    /// Normal priority - regular tensors
    Normal = 2,
    /// Low priority - can be deferred
    Low = 3,
    /// Deferred - cleanup when convenient
    Deferred = 4,
}

/// Deallocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeallocationStrategy {
    /// Deallocate immediately when tensor is dropped
    Immediate,
    /// Batch deallocations for efficiency
    Batched,
    /// Defer deallocation until pressure builds
    Deferred,
    /// Use pressure-aware strategy (default)
    PressureAware,
}

/// Tensor deallocation request
#[derive(Debug)]
struct DeallocationRequest {
    tensor_id: u64,
    handle: MemoryHandle,
    priority: DeallocationPriority,
    requested_at: Instant,
    size_bytes: usize,
    is_temporary: bool,
}

impl PartialEq for DeallocationRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for DeallocationRequest {}

impl PartialOrd for DeallocationRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeallocationRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max heap (highest priority first)
        other.priority.cmp(&self.priority)
            .then_with(|| other.requested_at.cmp(&self.requested_at))
    }
}

/// Batch deallocation configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (number of tensors)
    pub max_batch_size: usize,
    /// Maximum batch memory size
    pub max_batch_memory: usize,
    /// Batch timeout (time to wait before forcing batch)
    pub batch_timeout: Duration,
    /// Minimum batch size to trigger deallocation
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_batch_memory: 64 * 1024 * 1024, // 64MB
            batch_timeout: Duration::from_millis(100),
            min_batch_size: 5,
        }
    }
}

/// Tensor deallocation manager with intelligent patterns
#[derive(Debug)]
pub struct TensorDeallocationManager {
    /// Pending deallocation requests (priority queue)
    pending_requests: Arc<Mutex<BinaryHeap<DeallocationRequest>>>,
    /// Current batch being built
    current_batch: Arc<Mutex<Vec<DeallocationRequest>>>,
    /// Last batch processing time
    last_batch_time: Arc<Mutex<Instant>>,
    /// Deallocation strategy
    strategy: Arc<RwLock<DeallocationStrategy>>,
    /// Batch configuration
    batch_config: Arc<RwLock<BatchConfig>>,
    /// Reference to tensor memory pool
    pool: Arc<TensorMemoryPool>,
    /// Statistics
    stats: Arc<RwLock<DeallocationStats>>,
}

/// Statistics for deallocation operations
#[derive(Debug, Clone, Default)]
pub struct DeallocationStats {
    /// Total deallocation requests processed
    pub total_requests: u64,
    /// Number of immediate deallocations
    pub immediate_deallocations: u64,
    /// Number of batched deallocations
    pub batched_deallocations: u64,
    /// Number of deferred deallocations
    pub deferred_deallocations: u64,
    /// Total memory deallocated
    pub total_memory_deallocated: usize,
    /// Average batch size
    pub average_batch_size: f64,
    /// Total batch processing time
    pub total_batch_time: Duration,
    /// Number of pressure-triggered cleanups
    pub pressure_cleanups: u64,
}

impl TensorDeallocationManager {
    /// Create a new tensor deallocation manager
    pub fn new(pool: Arc<TensorMemoryPool>) -> Self {
        Self {
            pending_requests: Arc::new(Mutex::new(BinaryHeap::new())),
            current_batch: Arc::new(Mutex::new(Vec::new())),
            last_batch_time: Arc::new(Mutex::new(Instant::now())),
            strategy: Arc::new(RwLock::new(DeallocationStrategy::PressureAware)),
            batch_config: Arc::new(RwLock::new(BatchConfig::default())),
            pool,
            stats: Arc::new(RwLock::new(DeallocationStats::default())),
        }
    }

    /// Request tensor deallocation with specified priority
    pub fn request_deallocation(
        &self,
        tensor_id: u64,
        handle: MemoryHandle,
        priority: DeallocationPriority,
        is_temporary: bool,
    ) -> MemoryResult<()> {
        let request = DeallocationRequest {
            tensor_id,
            size_bytes: handle.size(),
            handle,
            priority,
            requested_at: Instant::now(),
            is_temporary,
        };

        #[cfg(feature = "tracing")]
        debug!(
            "Deallocation request for tensor {} (size: {} bytes, priority: {:?})",
            tensor_id, request.size_bytes, priority
        );

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_requests += 1;
        }

        match priority {
            DeallocationPriority::Immediate => {
                self.deallocate_immediate(request)?;
            }
            _ => {
                let strategy = if let Ok(strategy) = self.strategy.read() {
                    *strategy
                } else {
                    DeallocationStrategy::PressureAware
                };

                match strategy {
                    DeallocationStrategy::Immediate => {
                        self.deallocate_immediate(request)?;
                    }
                    DeallocationStrategy::Batched => {
                        self.add_to_batch(request)?;
                    }
                    DeallocationStrategy::Deferred => {
                        self.add_to_pending(request)?;
                    }
                    DeallocationStrategy::PressureAware => {
                        self.deallocate_pressure_aware(request)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Process all pending deallocations
    pub fn process_pending(&self) -> MemoryResult<usize> {
        let mut processed = 0;

        // Process high priority items first
        if let Ok(mut pending) = self.pending_requests.lock() {
            let mut to_process = Vec::new();
            
            // Extract high priority items
            while let Some(request) = pending.peek() {
                if request.priority <= DeallocationPriority::High {
                    to_process.push(pending.pop().unwrap());
                } else {
                    break;
                }
            }

            drop(pending);

            // Process the high priority items
            for request in to_process {
                self.deallocate_immediate(request)?;
                processed += 1;
            }
        }

        // Check if we need to process a batch
        if self.should_process_batch()? {
            processed += self.process_current_batch()?;
        }

        #[cfg(feature = "tracing")]
        if processed > 0 {
            debug!("Processed {} pending deallocations", processed);
        }

        Ok(processed)
    }

    /// Force processing of all pending deallocations
    pub fn process_all_pending(&self) -> MemoryResult<usize> {
        let mut processed = 0;

        // Process all pending requests
        if let Ok(mut pending) = self.pending_requests.lock() {
            while let Some(request) = pending.pop() {
                self.deallocate_immediate(request)?;
                processed += 1;
            }
        }

        // Process current batch
        processed += self.process_current_batch()?;

        #[cfg(feature = "tracing")]
        if processed > 0 {
            info!("Force processed {} pending deallocations", processed);
        }

        Ok(processed)
    }

    /// Handle memory pressure by processing deallocations
    pub fn handle_pressure(&self, pressure_level: MemoryPressureLevel) -> MemoryResult<usize> {
        let processed = match pressure_level {
            MemoryPressureLevel::High => {
                // Process all high and normal priority requests
                self.process_priority_requests(DeallocationPriority::Normal)?
            }
            MemoryPressureLevel::Critical => {
                // Process everything
                self.process_all_pending()?
            }
            _ => {
                // Process high priority only
                self.process_priority_requests(DeallocationPriority::High)?
            }
        };

        if processed > 0 {
            if let Ok(mut stats) = self.stats.write() {
                stats.pressure_cleanups += 1;
            }
        }

        Ok(processed)
    }

    /// Set deallocation strategy
    pub fn set_strategy(&self, strategy: DeallocationStrategy) -> MemoryResult<()> {
        if let Ok(mut s) = self.strategy.write() {
            *s = strategy;
            #[cfg(feature = "tracing")]
            debug!("Deallocation strategy changed to {:?}", strategy);
        }
        Ok(())
    }

    /// Get deallocation statistics
    pub fn get_stats(&self) -> MemoryResult<DeallocationStats> {
        if let Ok(stats) = self.stats.read() {
            Ok(stats.clone())
        } else {
            Err(MemoryError::InternalError {
                reason: "Failed to acquire stats lock".to_string(),
            })
        }
    }

    // Private helper methods

    fn deallocate_immediate(&self, request: DeallocationRequest) -> MemoryResult<()> {
        #[cfg(feature = "tracing")]
        debug!("Immediate deallocation of tensor {}", request.tensor_id);

        self.pool.deallocate_tensor(request.tensor_id, request.handle)?;

        if let Ok(mut stats) = self.stats.write() {
            stats.immediate_deallocations += 1;
            stats.total_memory_deallocated += request.size_bytes;
        }

        Ok(())
    }

    fn add_to_batch(&self, request: DeallocationRequest) -> MemoryResult<()> {
        if let Ok(mut batch) = self.current_batch.lock() {
            batch.push(request);

            // Check if batch is ready to process
            if self.should_process_batch()? {
                drop(batch);
                self.process_current_batch()?;
            }
        }
        Ok(())
    }

    fn add_to_pending(&self, request: DeallocationRequest) -> MemoryResult<()> {
        if let Ok(mut pending) = self.pending_requests.lock() {
            pending.push(request);
        }
        Ok(())
    }

    fn deallocate_pressure_aware(&self, request: DeallocationRequest) -> MemoryResult<()> {
        let pressure_level = self.pool.get_memory_pressure();

        match pressure_level {
            MemoryPressureLevel::High | MemoryPressureLevel::Critical => {
                // High pressure - deallocate immediately
                self.deallocate_immediate(request)
            }
            MemoryPressureLevel::Medium => {
                // Medium pressure - use batching for efficiency
                self.add_to_batch(request)
            }
            _ => {
                // Low pressure - defer if possible
                match request.priority {
                    DeallocationPriority::Immediate | DeallocationPriority::High => {
                        self.deallocate_immediate(request)
                    }
                    _ => {
                        self.add_to_pending(request)
                    }
                }
            }
        }
    }

    fn should_process_batch(&self) -> MemoryResult<bool> {
        let config = if let Ok(config) = self.batch_config.read() {
            config.clone()
        } else {
            return Ok(false);
        };

        if let Ok(batch) = self.current_batch.lock() {
            // Check batch size
            if batch.len() >= config.max_batch_size {
                return Ok(true);
            }

            // Check batch memory
            let total_memory: usize = batch.iter().map(|r| r.size_bytes).sum();
            if total_memory >= config.max_batch_memory {
                return Ok(true);
            }

            // Check timeout
            if let Ok(last_time) = self.last_batch_time.lock() {
                if last_time.elapsed() >= config.batch_timeout && batch.len() >= config.min_batch_size {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn process_current_batch(&self) -> MemoryResult<usize> {
        let batch = if let Ok(mut batch) = self.current_batch.lock() {
            let current_batch = std::mem::take(&mut *batch);
            current_batch
        } else {
            return Ok(0);
        };

        if batch.is_empty() {
            return Ok(0);
        }

        let batch_size = batch.len();
        let start_time = Instant::now();

        #[cfg(feature = "tracing")]
        debug!("Processing batch of {} deallocation requests", batch_size);

        let mut total_memory = 0;
        for request in batch {
            total_memory += request.size_bytes;
            self.pool.deallocate_tensor(request.tensor_id, request.handle)?;
        }

        let processing_time = start_time.elapsed();

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.batched_deallocations += batch_size as u64;
            stats.total_memory_deallocated += total_memory;
            stats.total_batch_time += processing_time;
            
            // Update average batch size
            let total_batches = stats.batched_deallocations / batch_size as u64;
            if total_batches > 0 {
                stats.average_batch_size = stats.batched_deallocations as f64 / total_batches as f64;
            }
        }

        // Update last batch time
        if let Ok(mut last_time) = self.last_batch_time.lock() {
            *last_time = Instant::now();
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Processed batch of {} requests ({} bytes) in {:?}",
            batch_size, total_memory, processing_time
        );

        Ok(batch_size)
    }

    fn process_priority_requests(&self, max_priority: DeallocationPriority) -> MemoryResult<usize> {
        let mut processed = 0;

        if let Ok(mut pending) = self.pending_requests.lock() {
            let mut to_process = Vec::new();
            
            // Extract items with priority <= max_priority
            while let Some(request) = pending.peek() {
                if request.priority <= max_priority {
                    to_process.push(pending.pop().unwrap());
                } else {
                    break;
                }
            }

            drop(pending);

            // Process the items
            for request in to_process {
                self.deallocate_immediate(request)?;
                processed += 1;
            }
        }

        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HybridMemoryPool, MemoryHandle};
    use crate::device::get_cpu_device;

    #[test]
    fn test_deallocation_priority_ordering() {
        let immediate = DeallocationPriority::Immediate;
        let high = DeallocationPriority::High;
        let normal = DeallocationPriority::Normal;
        
        assert!(immediate < high);
        assert!(high < normal);
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 100);
        assert_eq!(config.max_batch_memory, 64 * 1024 * 1024);
        assert_eq!(config.min_batch_size, 5);
    }
}
