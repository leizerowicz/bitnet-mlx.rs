//! Execution context management for inference operations.

use crate::{Result, InferenceError};
use bitnet_core::Device;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Execution context that manages the runtime environment for inference.
#[derive(Debug)]
pub struct ExecutionContext {
    /// Target device for computation
    device: Device,
    /// Execution statistics
    stats: Arc<std::sync::Mutex<ExecutionStats>>,
    /// Configuration parameters
    config: ExecutionConfig,
}

/// Configuration for execution context.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Timeout for individual operations
    pub operation_timeout: Duration,
    /// Maximum memory usage allowed
    pub memory_limit: Option<usize>,
    /// Whether to collect detailed timing statistics
    pub collect_stats: bool,
    /// Number of warm-up iterations for performance optimization
    pub warmup_iterations: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            operation_timeout: Duration::from_secs(30),
            memory_limit: None,
            collect_stats: true,
            warmup_iterations: 3,
        }
    }
}

/// Statistics collected during execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total number of operations executed
    pub operations_count: u64,
    /// Total time spent in operations
    pub total_execution_time: Duration,
    /// Average operation time
    pub average_operation_time: Duration,
    /// Peak memory usage observed
    pub peak_memory_usage: usize,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Last operation timestamp
    pub last_operation_time: Option<Instant>,
}

impl ExecutionContext {
    /// Create a new execution context for the specified device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            stats: Arc::new(std::sync::Mutex::new(ExecutionStats::default())),
            config: ExecutionConfig::default(),
        }
    }

    /// Create an execution context with custom configuration.
    pub fn with_config(device: Device, config: ExecutionConfig) -> Self {
        Self {
            device,
            stats: Arc::new(std::sync::Mutex::new(ExecutionStats::default())),
            config,
        }
    }

    /// Get the target device for this context.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Execute an operation within this context.
    pub fn execute<F, T>(&self, operation_name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = Instant::now();

        // Execute the operation
        let result = operation();
        
        let execution_time = start_time.elapsed();

        // Update statistics if enabled
        if self.config.collect_stats {
            self.update_stats(execution_time, result.is_ok());
        }

        // Check timeout
        if execution_time > self.config.operation_timeout {
            tracing::warn!(
                "Operation '{}' took {:?}, exceeding timeout of {:?}",
                operation_name,
                execution_time,
                self.config.operation_timeout
            );
        }

        result
    }

    /// Execute an async operation within this context.
    pub async fn execute_async<F, Fut, T>(&self, operation_name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let start_time = Instant::now();

        // Execute the async operation with timeout
        let result = tokio::time::timeout(
            self.config.operation_timeout,
            operation()
        ).await;

        let execution_time = start_time.elapsed();

        let final_result = match result {
            Ok(inner_result) => inner_result,
            Err(_timeout) => Err(InferenceError::device(format!(
                "Operation '{}' timed out after {:?}",
                operation_name,
                self.config.operation_timeout
            ))),
        };

        // Update statistics if enabled
        if self.config.collect_stats {
            self.update_stats(execution_time, final_result.is_ok());
        }

        final_result
    }

    /// Perform warmup operations to optimize performance.
    pub fn warmup<F>(&self, warmup_operation: F) -> Result<()>
    where
        F: Fn() -> Result<()>,
    {
        tracing::info!("Starting warmup with {} iterations", self.config.warmup_iterations);

        for i in 0..self.config.warmup_iterations {
            let start = Instant::now();
            warmup_operation()?;
            let duration = start.elapsed();
            
            tracing::debug!("Warmup iteration {}: {:?}", i + 1, duration);
        }

        tracing::info!("Warmup completed");
        Ok(())
    }

    /// Get current execution statistics.
    pub fn stats(&self) -> ExecutionStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset execution statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = ExecutionStats::default();
    }

    /// Check if memory usage is within limits.
    pub fn check_memory_usage(&self, current_usage: usize) -> Result<()> {
        if let Some(limit) = self.config.memory_limit {
            if current_usage > limit {
                return Err(InferenceError::memory(format!(
                    "Memory usage ({} bytes) exceeds limit ({} bytes)",
                    current_usage,
                    limit
                )));
            }
        }
        Ok(())
    }

    /// Update execution statistics.
    fn update_stats(&self, execution_time: Duration, success: bool) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.operations_count += 1;
            stats.total_execution_time += execution_time;
            
            if stats.operations_count > 0 {
                stats.average_operation_time = stats.total_execution_time / stats.operations_count as u32;
            }

            if !success {
                stats.failed_operations += 1;
            }

            stats.last_operation_time = Some(Instant::now());
        }
    }
}

impl ExecutionStats {
    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.operations_count == 0 {
            return 0.0;
        }
        
        let successful_ops = self.operations_count - self.failed_operations;
        (successful_ops as f64 / self.operations_count as f64) * 100.0
    }

    /// Get operations per second.
    pub fn operations_per_second(&self) -> f64 {
        if self.total_execution_time.is_zero() {
            return 0.0;
        }
        
        self.operations_count as f64 / self.total_execution_time.as_secs_f64()
    }

    /// Check if statistics indicate good performance.
    pub fn is_performing_well(&self) -> bool {
        self.success_rate() > 95.0 && 
        self.average_operation_time < Duration::from_millis(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context_creation() {
        let ctx = ExecutionContext::new(Device::Cpu);
        assert!(matches!(ctx.device(), Device::Cpu));
        
        let stats = ctx.stats();
        assert_eq!(stats.operations_count, 0);
    }

    #[test]
    fn test_execution_with_stats() {
        let ctx = ExecutionContext::new(Device::Cpu);
        
        // Execute a simple operation
        let result = ctx.execute("test_op", || -> Result<i32> {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        
        let stats = ctx.stats();
        assert_eq!(stats.operations_count, 1);
        assert_eq!(stats.failed_operations, 0);
        assert!(stats.total_execution_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_execution_stats_calculations() {
        let mut stats = ExecutionStats::default();
        stats.operations_count = 100;
        stats.failed_operations = 5;
        stats.total_execution_time = Duration::from_secs(10);
        
        assert_eq!(stats.success_rate(), 95.0);
        assert_eq!(stats.operations_per_second(), 10.0);
        assert!(!stats.is_performing_well()); // Because success rate is exactly 95%, not > 95%
    }

    #[tokio::test]
    async fn test_async_execution() {
        let ctx = ExecutionContext::new(Device::Cpu);
        
        let result = ctx.execute_async("async_test", || async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok::<i32, InferenceError>(42)
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}
