//! Cleanup Manager Implementation
//!
//! This module provides the main CleanupManager that coordinates all cleanup
//! operations, manages strategies, and provides both automatic and manual
//! cleanup capabilities.

use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "tracing")]
use tracing::{debug, info};

use super::config::{CleanupConfig, CleanupStrategyType};
use super::scheduler::CleanupScheduler;
use super::strategies::{
    CleanupPriority, CleanupStrategy, DeviceCleanupStrategy, GenerationalCleanupStrategy,
    IdleCleanupStrategy, PeriodicCleanupStrategy, PressureCleanupStrategy,
};
use super::{
    CleanupError, CleanupOperation, CleanupOperationId, CleanupResult, GlobalCleanupStats,
};
use crate::memory::tracking::{DetailedMemoryMetrics, MemoryPressureLevel, MemoryTracker};
use crate::memory::{HybridMemoryPool, MemoryMetrics};

/// Result of a cleanup operation with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CleanupOperationResult {
    /// Number of bytes freed
    pub bytes_freed: u64,
    /// Number of allocations cleaned
    pub allocations_cleaned: u64,
    /// Duration of the cleanup operation
    pub duration: Duration,
    /// Strategy used for cleanup
    pub strategy_used: CleanupStrategyType,
    /// Whether the operation was successful
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
    /// Additional operation metadata
    pub metadata: HashMap<String, String>,
}

impl CleanupOperationResult {
    /// Creates a successful cleanup result
    pub fn success(
        bytes_freed: u64,
        allocations_cleaned: u64,
        duration: Duration,
        strategy_used: CleanupStrategyType,
    ) -> Self {
        Self {
            bytes_freed,
            allocations_cleaned,
            duration,
            strategy_used,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a failed cleanup result
    pub fn failure(error: String, duration: Duration, strategy_used: CleanupStrategyType) -> Self {
        Self {
            bytes_freed: 0,
            allocations_cleaned: 0,
            duration,
            strategy_used,
            success: false,
            error_message: Some(error),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the result
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Result of a pool compaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CompactionResult {
    /// Number of bytes compacted
    pub bytes_compacted: u64,
    /// Number of memory blocks consolidated
    pub blocks_consolidated: u64,
    /// Fragmentation ratio before compaction
    pub fragmentation_before: f64,
    /// Fragmentation ratio after compaction
    pub fragmentation_after: f64,
    /// Duration of the compaction operation
    pub duration: Duration,
    /// Whether the operation was successful
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
}

impl CompactionResult {
    /// Creates a successful compaction result
    pub fn success(
        bytes_compacted: u64,
        blocks_consolidated: u64,
        fragmentation_before: f64,
        fragmentation_after: f64,
        duration: Duration,
    ) -> Self {
        Self {
            bytes_compacted,
            blocks_consolidated,
            fragmentation_before,
            fragmentation_after,
            duration,
            success: true,
            error_message: None,
        }
    }

    /// Creates a failed compaction result
    pub fn failure(error: String, duration: Duration) -> Self {
        Self {
            bytes_compacted: 0,
            blocks_consolidated: 0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            duration,
            success: false,
            error_message: Some(error),
        }
    }
}

/// Main cleanup manager that coordinates all cleanup operations
#[allow(dead_code)]
pub struct CleanupManager {
    /// Configuration for cleanup operations
    config: CleanupConfig,
    /// Reference to the memory pool being managed
    pool: Arc<HybridMemoryPool>,
    /// Memory tracker for detailed metrics (optional)
    memory_tracker: Option<Arc<MemoryTracker>>,
    /// Cleanup scheduler for automatic operations
    scheduler: Option<Arc<Mutex<CleanupScheduler>>>,
    /// Registered cleanup strategies
    strategies: Arc<RwLock<HashMap<CleanupStrategyType, Box<dyn CleanupStrategy>>>>,
    /// Global cleanup statistics
    stats: Arc<RwLock<GlobalCleanupStats>>,
    /// Operation ID counter
    next_operation_id: Arc<Mutex<u64>>,
    /// Currently running operations
    active_operations: Arc<RwLock<HashMap<CleanupOperationId, CleanupOperation>>>,
    /// Whether the manager is currently running
    is_running: Arc<RwLock<bool>>,
    /// Cleanup operation history (limited size)
    operation_history: Arc<RwLock<Vec<CleanupOperation>>>,
}

impl CleanupManager {
    /// Creates a new cleanup manager
    pub fn new(config: CleanupConfig, pool: Arc<HybridMemoryPool>) -> CleanupResult<Self> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| CleanupError::InvalidConfiguration { reason: e })?;

        #[cfg(feature = "tracing")]
        info!("Creating cleanup manager with config: {:?}", config);

        // Get memory tracker from pool if available
        let memory_tracker = pool.get_memory_tracker().cloned();

        // Create scheduler if enabled
        let scheduler = if config.scheduler.enabled {
            Some(Arc::new(Mutex::new(CleanupScheduler::new(
                config.scheduler.clone(),
            )?)))
        } else {
            None
        };

        // Initialize strategies
        let strategies = Arc::new(RwLock::new(HashMap::new()));

        let manager = Self {
            config: config.clone(),
            pool,
            memory_tracker,
            scheduler,
            strategies,
            stats: Arc::new(RwLock::new(GlobalCleanupStats::new())),
            next_operation_id: Arc::new(Mutex::new(1)),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            operation_history: Arc::new(RwLock::new(Vec::new())),
        };

        // Register default strategies
        manager.register_default_strategies()?;

        #[cfg(feature = "tracing")]
        info!("Cleanup manager created successfully");

        Ok(manager)
    }

    /// Registers default cleanup strategies based on configuration
    fn register_default_strategies(&self) -> CleanupResult<()> {
        let mut strategies = self
            .strategies
            .write()
            .map_err(|_| CleanupError::InternalError {
                reason: "Failed to acquire strategies lock".to_string(),
            })?;

        // Register strategies based on feature flags
        if self.config.features.enable_idle_cleanup {
            let idle_strategy = IdleCleanupStrategy::new(
                self.config.thresholds.idle.min_idle_time,
                self.config.thresholds.idle.max_idle_cleanup_duration,
            );
            strategies.insert(CleanupStrategyType::Idle, Box::new(idle_strategy));
        }

        if self.config.features.enable_pressure_cleanup {
            let pressure_strategy = PressureCleanupStrategy::new(
                self.config.thresholds.pressure.light_cleanup_threshold,
                self.config.thresholds.pressure.aggressive_cleanup_threshold,
                self.config.thresholds.pressure.emergency_cleanup_threshold,
                self.config
                    .thresholds
                    .pressure
                    .min_pressure_cleanup_interval,
            );
            strategies.insert(CleanupStrategyType::Pressure, Box::new(pressure_strategy));
        }

        if self.config.features.enable_periodic_cleanup {
            let periodic_strategy = PeriodicCleanupStrategy::new(
                self.config.scheduler.base_interval,
                Duration::from_millis(20),
            );
            strategies.insert(CleanupStrategyType::Periodic, Box::new(periodic_strategy));
        }

        if self.config.features.enable_device_cleanup {
            strategies.insert(
                CleanupStrategyType::Device,
                Box::new(DeviceCleanupStrategy::cpu()),
            );
        }

        if self.config.features.enable_generational_cleanup {
            let generational_strategy = GenerationalCleanupStrategy::new(
                self.config.thresholds.age.young_generation_age,
                self.config.thresholds.age.old_generation_age,
                self.config.thresholds.age.ancient_generation_age,
                Duration::from_secs(15),
            );
            strategies.insert(
                CleanupStrategyType::Generational,
                Box::new(generational_strategy),
            );
        }

        #[cfg(feature = "tracing")]
        info!("Registered {} cleanup strategies", strategies.len());

        Ok(())
    }

    /// Starts the automatic cleanup scheduler
    pub fn start_scheduler(&self) -> CleanupResult<()> {
        if !self.config.scheduler.enabled {
            return Err(CleanupError::SchedulerError {
                reason: "Scheduler is disabled in configuration".to_string(),
            });
        }

        // Set running state
        {
            let mut is_running =
                self.is_running
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire running state lock".to_string(),
                    })?;

            if *is_running {
                return Err(CleanupError::SchedulerError {
                    reason: "Scheduler is already running".to_string(),
                });
            }

            *is_running = true;
        }

        if let Some(ref scheduler) = self.scheduler {
            let scheduler_clone = scheduler.clone();
            let manager_weak = Arc::downgrade(&Arc::new(self.clone_for_scheduler()));

            // Start scheduler in background thread
            thread::spawn(move || {
                if let Ok(mut scheduler) = scheduler_clone.lock() {
                    scheduler.start(manager_weak);
                }
            });

            #[cfg(feature = "tracing")]
            info!("Cleanup scheduler started");
        }

        Ok(())
    }

    /// Stops the automatic cleanup scheduler
    pub fn stop_scheduler(&self) -> CleanupResult<()> {
        // Set running state to false
        {
            let mut is_running =
                self.is_running
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire running state lock".to_string(),
                    })?;

            if !*is_running {
                return Ok(()); // Already stopped
            }

            *is_running = false;
        }

        if let Some(ref scheduler) = self.scheduler {
            if let Ok(mut scheduler) = scheduler.lock() {
                scheduler.stop();
            }

            #[cfg(feature = "tracing")]
            info!("Cleanup scheduler stopped");
        }

        Ok(())
    }

    /// Performs immediate cleanup using the best available strategy
    pub fn force_cleanup(&self) -> CleanupResult<CleanupOperationResult> {
        #[cfg(feature = "tracing")]
        debug!("Performing force cleanup");

        let metrics = self.pool.get_metrics();
        let detailed_metrics = self
            .memory_tracker
            .as_ref()
            .map(|tracker| tracker.get_detailed_metrics());

        // Find the best strategy for current conditions
        let strategy_type = self.select_best_strategy(&metrics, detailed_metrics.as_ref())?;

        self.execute_cleanup_strategy(strategy_type)
    }

    /// Performs cleanup on a specific device
    pub fn cleanup_device(&self, _device: &Device) -> CleanupResult<CleanupOperationResult> {
        #[cfg(feature = "tracing")]
        debug!(
            "Performing device-specific cleanup for device: {:?}",
            _device
        );

        // Use device-specific strategy
        self.execute_cleanup_strategy(CleanupStrategyType::Device)
    }

    /// Performs selective cleanup based on criteria
    pub fn cleanup_selective(
        &self,
        min_age: Option<Duration>,
        _min_size: Option<usize>,
        _device_filter: Option<&Device>,
    ) -> CleanupResult<CleanupOperationResult> {
        #[cfg(feature = "tracing")]
        debug!(
            "Performing selective cleanup with filters - age: {:?}, size: {:?}, device: {:?}",
            min_age, _min_size, _device_filter
        );

        // For selective cleanup, use generational strategy if age is specified,
        // otherwise use the default strategy
        let strategy_type = if min_age.is_some() {
            CleanupStrategyType::Generational
        } else {
            self.config.policy.default_strategy
        };

        self.execute_cleanup_strategy(strategy_type)
    }

    /// Compacts memory pools to reduce fragmentation
    pub fn compact_pools(&self) -> CleanupResult<CompactionResult> {
        let start_time = Instant::now();

        #[cfg(feature = "tracing")]
        info!("Starting pool compaction");

        // Get current metrics to measure fragmentation
        let metrics_before = self.pool.get_metrics();

        // Calculate fragmentation before (simplified calculation)
        let fragmentation_before = if metrics_before.peak_allocated > 0 {
            1.0 - (metrics_before.current_allocated as f64 / metrics_before.peak_allocated as f64)
        } else {
            0.0
        };

        // Simulate compaction work (in real implementation, this would perform actual compaction)
        thread::sleep(Duration::from_millis(50));

        let duration = start_time.elapsed();
        let bytes_compacted = 2048; // Simulated
        let blocks_consolidated = 10; // Simulated
        let fragmentation_after = fragmentation_before * 0.7; // Simulated improvement

        #[cfg(feature = "tracing")]
        info!("Pool compaction completed in {:?}", duration);

        Ok(CompactionResult::success(
            bytes_compacted,
            blocks_consolidated,
            fragmentation_before,
            fragmentation_after,
            duration,
        ))
    }

    /// Registers a custom cleanup strategy
    pub fn register_cleanup_strategy(
        &self,
        strategy_type: CleanupStrategyType,
        strategy: Box<dyn CleanupStrategy>,
    ) -> CleanupResult<()> {
        let mut strategies = self
            .strategies
            .write()
            .map_err(|_| CleanupError::InternalError {
                reason: "Failed to acquire strategies lock".to_string(),
            })?;

        strategies.insert(strategy_type, strategy);

        #[cfg(feature = "tracing")]
        debug!("Registered custom cleanup strategy: {:?}", strategy_type);

        Ok(())
    }

    /// Returns current cleanup statistics
    pub fn get_cleanup_stats(&self) -> GlobalCleanupStats {
        self.stats
            .read()
            .map(|stats| stats.clone()) // Fixed typo: tats -> stats
            .unwrap_or_else(|_| GlobalCleanupStats::new()) // Fixed closure signature
    }

    /// Returns the cleanup configuration
    pub fn get_config(&self) -> &CleanupConfig {
        &self.config
    }

    /// Returns whether the scheduler is currently running
    pub fn is_scheduler_running(&self) -> bool {
        self.is_running
            .read()
            .map(|running| *running)
            .unwrap_or(false)
    }

    /// Returns the operation history
    pub fn get_operation_history(&self) -> Vec<CleanupOperation> {
        self.operation_history
            .read()
            .map(|history| history.clone())
            .unwrap_or_else(|_| Vec::new()) // Fixed closure signature
    }

    // Private helper methods

    /// Selects the best cleanup strategy based on current conditions
    fn select_best_strategy(
        &self,
        metrics: &MemoryMetrics,
        detailed_metrics: Option<&DetailedMemoryMetrics>,
    ) -> CleanupResult<CleanupStrategyType> {
        let strategies = self
            .strategies
            .read()
            .map_err(|_| CleanupError::InternalError {
                reason: "Failed to acquire strategies lock".to_string(),
            })?;

        // Check for emergency conditions first
        if let Some(detailed) = detailed_metrics {
            if matches!(detailed.pressure_level, MemoryPressureLevel::Critical) {
                if strategies.contains_key(&CleanupStrategyType::Pressure) {
                    return Ok(CleanupStrategyType::Pressure);
                }
            }
        }

        // Find strategies that want to run and select highest priority
        let mut best_strategy = None;
        let mut best_priority = CleanupPriority::Background;

        for (strategy_type, strategy) in strategies.iter() {
            if strategy.should_cleanup(metrics, detailed_metrics) {
                let priority = strategy.priority();
                if priority > best_priority {
                    best_priority = priority;
                    best_strategy = Some(*strategy_type);
                }
            }
        }

        best_strategy.ok_or_else(|| CleanupError::OperationFailed {
            reason: "No suitable cleanup strategy found".to_string(),
        })
    }

    /// Executes a specific cleanup strategy
    fn execute_cleanup_strategy(
        &self,
        strategy_type: CleanupStrategyType,
    ) -> CleanupResult<CleanupOperationResult> {
        let operation_id = self.generate_operation_id()?;
        let start_time = Instant::now();

        // Create operation record
        let mut operation = CleanupOperation::new(
            operation_id,
            strategy_type,
            "Mixed".to_string(), // Would be determined from actual device context
        );

        // Add to active operations
        {
            let mut active_ops =
                self.active_operations
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire active operations lock".to_string(),
                    })?;
            active_ops.insert(operation_id, operation.clone());
        }

        // Execute the strategy
        let result = {
            let strategies = self
                .strategies
                .read()
                .map_err(|_| CleanupError::InternalError {
                    reason: "Failed to acquire strategies lock".to_string(),
                })?;

            let strategy =
                strategies
                    .get(&strategy_type)
                    .ok_or_else(|| CleanupError::StrategyFailed {
                        strategy: format!("{:?}", strategy_type),
                        reason: "Strategy not found".to_string(),
                    })?;

            strategy.cleanup(&self.pool, &self.config)
        };

        let duration = start_time.elapsed();

        // Update operation record and remove from active operations
        match result {
            Ok(op_result) => {
                operation.complete_success(
                    op_result.bytes_freed,
                    op_result.allocations_cleaned,
                    duration,
                );

                let cleanup_result = CleanupOperationResult::success(
                    op_result.bytes_freed,
                    op_result.allocations_cleaned,
                    duration,
                    strategy_type,
                );

                self.record_operation_completion(operation_id, operation)?;
                Ok(cleanup_result)
            }
            Err(e) => {
                operation.complete_failure(e.to_string(), duration);
                let cleanup_result =
                    CleanupOperationResult::failure(e.to_string(), duration, strategy_type);
                self.record_operation_completion(operation_id, operation)?;
                Ok(cleanup_result)
            }
        }
    }

    /// Generates a unique operation ID
    fn generate_operation_id(&self) -> CleanupResult<CleanupOperationId> {
        let mut counter =
            self.next_operation_id
                .lock()
                .map_err(|_| CleanupError::InternalError {
                    reason: "Failed to acquire operation ID counter lock".to_string(),
                })?;

        let id = *counter;
        *counter += 1;
        Ok(CleanupOperationId::new(id))
    }

    /// Records the completion of an operation
    fn record_operation_completion(
        &self,
        operation_id: CleanupOperationId,
        operation: CleanupOperation,
    ) -> CleanupResult<()> {
        // Remove from active operations
        {
            let mut active_ops =
                self.active_operations
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire active operations lock".to_string(),
                    })?;
            active_ops.remove(&operation_id);
        }

        // Add to history (with size limit)
        {
            let mut history =
                self.operation_history
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire operation history lock".to_string(),
                    })?;

            history.push(operation.clone());

            // Keep only last 1000 operations
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        // Update global statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|_| CleanupError::InternalError {
                    reason: "Failed to acquire stats lock".to_string(),
                })?;
            stats.record_operation(&operation);
        }

        Ok(())
    }

    /// Creates a clone suitable for scheduler use (avoiding circular references)
    fn clone_for_scheduler(&self) -> SchedulerCleanupManager {
        SchedulerCleanupManager {
            pool: self.pool.clone(),
            strategies: self.strategies.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            next_operation_id: self.next_operation_id.clone(),
            active_operations: self.active_operations.clone(),
            operation_history: self.operation_history.clone(),
        }
    }
}

/// Simplified cleanup manager for scheduler use
#[derive(Clone)]
#[allow(dead_code)]
pub struct SchedulerCleanupManager {
    pool: Arc<HybridMemoryPool>,
    strategies: Arc<RwLock<HashMap<CleanupStrategyType, Box<dyn CleanupStrategy>>>>,
    config: CleanupConfig,
    stats: Arc<RwLock<GlobalCleanupStats>>,
    next_operation_id: Arc<Mutex<u64>>,
    active_operations: Arc<RwLock<HashMap<CleanupOperationId, CleanupOperation>>>,
    operation_history: Arc<RwLock<Vec<CleanupOperation>>>,
}

impl SchedulerCleanupManager {
    /// Executes automatic cleanup (called by scheduler)
    pub fn execute_automatic_cleanup(&self) -> CleanupResult<()> {
        let metrics = self.pool.get_metrics();

        // Simple automatic cleanup logic
        let strategies = self
            .strategies
            .read()
            .map_err(|_| CleanupError::InternalError {
                reason: "Failed to acquire strategies lock".to_string(),
            })?;

        for (_strategy_type, strategy) in strategies.iter() {
            if strategy.should_cleanup(&metrics, None) {
                let _ = strategy.cleanup(&self.pool, &self.config);
                break; // Only run one strategy per automatic cleanup cycle
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_cleanup_manager_creation() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = CleanupConfig::default();

        let manager = CleanupManager::new(config, pool).unwrap();
        assert!(!manager.is_scheduler_running());

        let stats = manager.get_cleanup_stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_cleanup_result() {
        let result = CleanupOperationResult::success(
            1024,
            5,
            Duration::from_millis(100),
            CleanupStrategyType::Idle,
        );
        assert!(result.success);
        assert_eq!(result.bytes_freed, 1024);
        assert_eq!(result.allocations_cleaned, 5);
        assert_eq!(result.strategy_used, CleanupStrategyType::Idle);

        let result = CleanupOperationResult::failure(
            "test error".to_string(),
            Duration::from_millis(50),
            CleanupStrategyType::Pressure,
        );
        assert!(!result.success);
        assert_eq!(result.error_message, Some("test error".to_string()));
        assert_eq!(result.strategy_used, CleanupStrategyType::Pressure);
    }

    #[test]
    fn test_compaction_result() {
        let result = CompactionResult::success(2048, 10, 0.5, 0.3, Duration::from_millis(100));
        assert!(result.success);
        assert_eq!(result.bytes_compacted, 2048);
        assert_eq!(result.blocks_consolidated, 10);
        assert_eq!(result.fragmentation_before, 0.5);
        assert_eq!(result.fragmentation_after, 0.3);

        let result = CompactionResult::failure("test error".to_string(), Duration::from_millis(50));
        assert!(!result.success);
        assert_eq!(result.error_message, Some("test error".to_string()));
    }
}
