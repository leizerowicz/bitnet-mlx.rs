//! Cleanup Scheduler Implementation
//!
//! This module provides the CleanupScheduler that manages automatic cleanup
//! operations, scheduling them based on various triggers and priorities.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

use super::config::{CleanupStrategyType, SchedulerConfig};
use super::manager::SchedulerCleanupManager;
use super::strategies::CleanupPriority;
use super::{CleanupError, CleanupResult};

/// Unique identifier for scheduled cleanup operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CleanupId(pub u64);

impl CleanupId {
    /// Creates a new cleanup ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for CleanupId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<CleanupId> for u64 {
    fn from(id: CleanupId) -> u64 {
        id.0
    }
}

/// A scheduled cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ScheduledCleanup {
    /// Unique identifier for this scheduled cleanup
    pub id: CleanupId,
    /// Type of cleanup strategy to use
    pub strategy_type: CleanupStrategyType,
    /// Priority of this cleanup operation
    pub priority: CleanupPriority,
    /// When this cleanup should be executed
    pub scheduled_time: SystemTime,
    /// Delay before execution
    pub delay: Duration,
    /// Whether this is a recurring cleanup
    pub recurring: bool,
    /// Interval for recurring cleanups
    pub recurrence_interval: Option<Duration>,
    /// Maximum number of retries if cleanup fails
    pub max_retries: u32,
    /// Current retry count
    pub retry_count: u32,
    /// Whether this cleanup is currently active
    pub active: bool,
    /// Optional metadata for the cleanup
    pub metadata: HashMap<String, String>,
}

impl PartialEq for ScheduledCleanup {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.priority == other.priority && self.scheduled_time == other.scheduled_time
    }
}

impl Eq for ScheduledCleanup {}

impl PartialOrd for ScheduledCleanup {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledCleanup {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority should come first 
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // If priorities are equal, earlier scheduled time comes first
                other.scheduled_time.cmp(&self.scheduled_time)
            }
            ord => ord,
        }
    }
}

impl ScheduledCleanup {
    /// Creates a new scheduled cleanup
    pub fn new(
        id: CleanupId,
        strategy_type: CleanupStrategyType,
        priority: CleanupPriority,
        delay: Duration,
    ) -> Self {
        let scheduled_time = SystemTime::now() + delay;

        Self {
            id,
            strategy_type,
            priority,
            scheduled_time,
            delay,
            recurring: false,
            recurrence_interval: None,
            max_retries: 3,
            retry_count: 0,
            active: true,
            metadata: HashMap::new(),
        }
    }

    /// Creates a recurring scheduled cleanup
    pub fn new_recurring(
        id: CleanupId,
        strategy_type: CleanupStrategyType,
        priority: CleanupPriority,
        delay: Duration,
        interval: Duration,
    ) -> Self {
        let mut cleanup = Self::new(id, strategy_type, priority, delay);
        cleanup.recurring = true;
        cleanup.recurrence_interval = Some(interval);
        cleanup
    }

    /// Checks if this cleanup is ready to execute
    pub fn is_ready(&self) -> bool {
        self.active && SystemTime::now() >= self.scheduled_time
    }

    /// Reschedules this cleanup for the next occurrence (for recurring cleanups)
    pub fn reschedule(&mut self) -> bool {
        if self.recurring {
            if let Some(interval) = self.recurrence_interval {
                self.scheduled_time = SystemTime::now() + interval;
                self.retry_count = 0;
                return true;
            }
        }
        false
    }

    /// Increments the retry count and reschedules if retries are available
    pub fn retry(&mut self) -> bool {
        if self.retry_count < self.max_retries {
            self.retry_count += 1;
            // Exponential backoff for retries
            let backoff = Duration::from_millis(1000 * (1 << self.retry_count));
            self.scheduled_time = SystemTime::now() + backoff;
            return true;
        }
        false
    }

    /// Marks this cleanup as inactive
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Adds metadata to the scheduled cleanup
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Statistics for the cleanup scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SchedulerStats {
    /// Total number of cleanups scheduled
    pub total_scheduled: u64,
    /// Total number of cleanups executed
    pub total_executed: u64,
    /// Total number of cleanups cancelled
    pub total_cancelled: u64,
    /// Total number of cleanup failures
    pub total_failures: u64,
    /// Total number of cleanup retries
    pub total_retries: u64,
    /// Current number of active scheduled cleanups
    pub active_scheduled: u64,
    /// Average execution delay (actual vs scheduled time)
    pub average_execution_delay: Duration,
    /// Scheduler uptime
    pub uptime: Duration,
    /// Last cleanup execution time
    pub last_execution: Option<SystemTime>,
}

impl SchedulerStats {
    /// Creates new scheduler statistics
    pub fn new() -> Self {
        Self {
            total_scheduled: 0,
            total_executed: 0,
            total_cancelled: 0,
            total_failures: 0,
            total_retries: 0,
            active_scheduled: 0,
            average_execution_delay: Duration::ZERO,
            uptime: Duration::ZERO,
            last_execution: None,
        }
    }
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Cleanup scheduler that manages automatic cleanup operations
#[allow(dead_code)]
pub struct CleanupScheduler {
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Scheduled cleanups (priority queue)
    scheduled_cleanups: Arc<Mutex<BinaryHeap<ScheduledCleanup>>>,
    /// Cleanup lookup by ID
    cleanup_lookup: Arc<RwLock<HashMap<CleanupId, ScheduledCleanup>>>,
    /// Next cleanup ID counter
    next_cleanup_id: Arc<Mutex<u64>>,
    /// Whether the scheduler is running
    is_running: Arc<RwLock<bool>>,
    /// Scheduler statistics
    stats: Arc<RwLock<SchedulerStats>>,
    /// Scheduler start time
    start_time: Arc<RwLock<Option<Instant>>>,
    /// Currently executing cleanups
    executing_cleanups: Arc<RwLock<HashMap<CleanupId, Instant>>>,
}

impl CleanupScheduler {
    /// Creates a new cleanup scheduler
    pub fn new(config: SchedulerConfig) -> CleanupResult<Self> {
        #[cfg(feature = "tracing")]
        info!("Creating cleanup scheduler with config: {:?}", config);

        Ok(Self {
            config,
            scheduled_cleanups: Arc::new(Mutex::new(BinaryHeap::new())),
            cleanup_lookup: Arc::new(RwLock::new(HashMap::new())),
            next_cleanup_id: Arc::new(Mutex::new(1)),
            is_running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(SchedulerStats::new())),
            start_time: Arc::new(RwLock::new(None)),
            executing_cleanups: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Starts the scheduler
    pub fn start(&mut self, manager: Weak<SchedulerCleanupManager>) {
        // Set running state
        {
            let mut is_running = self.is_running.write().unwrap();
            if *is_running {
                return; // Already running
            }
            *is_running = true;
        }

        // Set start time
        {
            let mut start_time = self.start_time.write().unwrap();
            *start_time = Some(Instant::now());
        }

        #[cfg(feature = "tracing")]
        info!("Starting cleanup scheduler");

        // Clone necessary data for the scheduler thread
        let scheduled_cleanups = self.scheduled_cleanups.clone();
        let cleanup_lookup = self.cleanup_lookup.clone();
        let is_running = self.is_running.clone();
        let stats = self.stats.clone();
        let executing_cleanups = self.executing_cleanups.clone();
        let config = self.config.clone();

        // Start scheduler thread
        thread::spawn(move || {
            Self::scheduler_loop(
                manager,
                scheduled_cleanups,
                cleanup_lookup,
                is_running,
                stats,
                executing_cleanups,
                config,
            );
        });
    }

    /// Stops the scheduler
    pub fn stop(&mut self) {
        let mut is_running = self.is_running.write().unwrap();
        *is_running = false;

        #[cfg(feature = "tracing")]
        info!("Stopping cleanup scheduler");
    }

    /// Schedules a cleanup operation
    pub fn schedule_cleanup(
        &self,
        strategy_type: CleanupStrategyType,
        priority: CleanupPriority,
        delay: Duration,
    ) -> CleanupResult<CleanupId> {
        let cleanup_id = self.generate_cleanup_id()?;
        let scheduled_cleanup = ScheduledCleanup::new(cleanup_id, strategy_type, priority, delay);

        self.add_scheduled_cleanup(scheduled_cleanup)?;

        #[cfg(feature = "tracing")]
        debug!(
            "Scheduled cleanup {:?} with strategy {:?} and delay {:?}",
            cleanup_id, strategy_type, delay
        );

        Ok(cleanup_id)
    }

    /// Schedules a recurring cleanup operation
    pub fn schedule_recurring_cleanup(
        &self,
        strategy_type: CleanupStrategyType,
        priority: CleanupPriority,
        delay: Duration,
        interval: Duration,
    ) -> CleanupResult<CleanupId> {
        let cleanup_id = self.generate_cleanup_id()?;
        let scheduled_cleanup =
            ScheduledCleanup::new_recurring(cleanup_id, strategy_type, priority, delay, interval);

        self.add_scheduled_cleanup(scheduled_cleanup)?;

        #[cfg(feature = "tracing")]
        debug!(
            "Scheduled recurring cleanup {:?} with strategy {:?}, delay {:?}, interval {:?}",
            cleanup_id, strategy_type, delay, interval
        );

        Ok(cleanup_id)
    }

    /// Cancels a scheduled cleanup
    pub fn cancel_cleanup(&self, cleanup_id: CleanupId) -> bool {
        // Remove from lookup
        let removed = {
            let mut lookup = self.cleanup_lookup.write().unwrap();
            lookup.remove(&cleanup_id).is_some()
        };

        if removed {
            // Update statistics
            if let Ok(mut stats) = self.stats.write() {
                stats.total_cancelled += 1;
                stats.active_scheduled = stats.active_scheduled.saturating_sub(1);
            }

            #[cfg(feature = "tracing")]
            debug!("Cancelled cleanup {:?}", cleanup_id);
        }

        removed
    }

    /// Returns current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        let mut stats = self
            .stats
            .read()
            .map(|s| s.clone())
            .unwrap_or_else(|_| SchedulerStats::new()); // Fixed closure signature

        // Update uptime
        if let Ok(start_time) = self.start_time.read() {
            if let Some(start) = *start_time {
                stats.uptime = start.elapsed();
            }
        }

        stats
    }

    /// Returns whether the scheduler is running
    pub fn is_running(&self) -> bool {
        self.is_running
            .read()
            .map(|running| *running)
            .unwrap_or(false)
    }

    /// Returns the number of active scheduled cleanups
    pub fn active_cleanup_count(&self) -> usize {
        self.cleanup_lookup
            .read()
            .map(|lookup| lookup.len())
            .unwrap_or(0)
    }

    // Private helper methods

    /// Generates a unique cleanup ID
    fn generate_cleanup_id(&self) -> CleanupResult<CleanupId> {
        let mut counter = self
            .next_cleanup_id
            .lock()
            .map_err(|_| CleanupError::InternalError {
                reason: "Failed to acquire cleanup ID counter lock".to_string(),
            })?;

        let id = *counter;
        *counter += 1;
        Ok(CleanupId::new(id))
    }

    /// Adds a scheduled cleanup to the scheduler
    fn add_scheduled_cleanup(&self, scheduled_cleanup: ScheduledCleanup) -> CleanupResult<()> {
        let cleanup_id = scheduled_cleanup.id;

        // Add to priority queue
        {
            let mut scheduled =
                self.scheduled_cleanups
                    .lock()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire scheduled cleanups lock".to_string(),
                    })?;
            scheduled.push(scheduled_cleanup.clone());
        }

        // Add to lookup
        {
            let mut lookup =
                self.cleanup_lookup
                    .write()
                    .map_err(|_| CleanupError::InternalError {
                        reason: "Failed to acquire cleanup lookup lock".to_string(),
                    })?;
            lookup.insert(cleanup_id, scheduled_cleanup);
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_scheduled += 1;
            stats.active_scheduled += 1;
        }

        Ok(())
    }

    /// Main scheduler loop (runs in background thread)
    fn scheduler_loop(
        manager: Weak<SchedulerCleanupManager>,
        scheduled_cleanups: Arc<Mutex<BinaryHeap<ScheduledCleanup>>>,
        cleanup_lookup: Arc<RwLock<HashMap<CleanupId, ScheduledCleanup>>>,
        is_running: Arc<RwLock<bool>>,
        stats: Arc<RwLock<SchedulerStats>>,
        executing_cleanups: Arc<RwLock<HashMap<CleanupId, Instant>>>,
        config: SchedulerConfig,
    ) {
        let mut last_check = Instant::now();

        while *is_running.read().unwrap() {
            let now = Instant::now();

            // Check if it's time to process cleanups
            if now.duration_since(last_check) >= config.base_interval {
                Self::process_ready_cleanups(
                    &manager,
                    &scheduled_cleanups,
                    &cleanup_lookup,
                    &stats,
                    &executing_cleanups,
                    &config,
                );
                last_check = now;
            }

            // Sleep for a short time to avoid busy waiting
            thread::sleep(Duration::from_millis(10));
        }

        #[cfg(feature = "tracing")]
        info!("Cleanup scheduler loop terminated");
    }

    /// Processes cleanups that are ready to execute
    fn process_ready_cleanups(
        manager: &Weak<SchedulerCleanupManager>,
        scheduled_cleanups: &Arc<Mutex<BinaryHeap<ScheduledCleanup>>>,
        cleanup_lookup: &Arc<RwLock<HashMap<CleanupId, ScheduledCleanup>>>,
        stats: &Arc<RwLock<SchedulerStats>>,
        executing_cleanups: &Arc<RwLock<HashMap<CleanupId, Instant>>>,
        config: &SchedulerConfig,
    ) {
        let mut ready_cleanups = Vec::new();

        // Find ready cleanups
        {
            let mut scheduled = scheduled_cleanups.lock().unwrap();
            let mut temp_heap = BinaryHeap::new();

            while let Some(cleanup) = scheduled.pop() {
                if cleanup.is_ready() && cleanup.active {
                    ready_cleanups.push(cleanup);
                } else {
                    temp_heap.push(cleanup);
                }
            }

            // Put non-ready cleanups back
            *scheduled = temp_heap;
        }

        // Execute ready cleanups
        for mut cleanup in ready_cleanups {
            // Check if we're under the concurrent operation limit
            let current_executing = executing_cleanups.read().unwrap().len();
            if current_executing >= config.max_concurrent_operations {
                // Put the cleanup back in the queue with a small delay
                cleanup.scheduled_time = SystemTime::now() + Duration::from_millis(100);
                scheduled_cleanups.lock().unwrap().push(cleanup);
                continue;
            }

            // Execute the cleanup
            Self::execute_scheduled_cleanup(
                manager,
                cleanup,
                cleanup_lookup,
                stats,
                executing_cleanups,
                scheduled_cleanups,
            );
        }
    }

    /// Executes a single scheduled cleanup
    fn execute_scheduled_cleanup(
        manager: &Weak<SchedulerCleanupManager>,
        mut cleanup: ScheduledCleanup,
        cleanup_lookup: &Arc<RwLock<HashMap<CleanupId, ScheduledCleanup>>>,
        stats: &Arc<RwLock<SchedulerStats>>,
        executing_cleanups: &Arc<RwLock<HashMap<CleanupId, Instant>>>,
        scheduled_cleanups: &Arc<Mutex<BinaryHeap<ScheduledCleanup>>>,
    ) {
        let cleanup_id = cleanup.id;
        let execution_start = Instant::now();

        // Mark as executing
        {
            let mut executing = executing_cleanups.write().unwrap();
            executing.insert(cleanup_id, execution_start);
        }

        #[cfg(feature = "tracing")]
        debug!(
            "Executing scheduled cleanup {:?} with strategy {:?}",
            cleanup_id, cleanup.strategy_type
        );

        // Execute the cleanup
        let success = if let Some(manager) = manager.upgrade() {
            manager.execute_automatic_cleanup().is_ok()
        } else {
            false
        };

        // Remove from executing
        {
            let mut executing = executing_cleanups.write().unwrap();
            executing.remove(&cleanup_id);
        }

        // Update statistics
        {
            if let Ok(mut stats) = stats.write() {
                stats.total_executed += 1;
                stats.last_execution = Some(SystemTime::now());

                if !success {
                    stats.total_failures += 1;
                }
            }
        }

        // Handle cleanup result
        if success {
            // Handle recurring cleanup
            if cleanup.reschedule() {
                // Put back in queue for next occurrence
                {
                    let mut lookup = cleanup_lookup.write().unwrap();
                    lookup.insert(cleanup_id, cleanup.clone());
                }
                scheduled_cleanups.lock().unwrap().push(cleanup);
            } else {
                // Remove from lookup (one-time cleanup completed)
                {
                    let mut lookup = cleanup_lookup.write().unwrap();
                    lookup.remove(&cleanup_id);
                }

                if let Ok(mut stats) = stats.write() {
                    stats.active_scheduled = stats.active_scheduled.saturating_sub(1);
                }
            }
        } else {
            // Handle failure - retry if possible
            if cleanup.retry() {
                {
                    if let Ok(mut stats) = stats.write() {
                        stats.total_retries += 1;
                    }
                }

                // Put back in queue for retry
                {
                    let mut lookup = cleanup_lookup.write().unwrap();
                    lookup.insert(cleanup_id, cleanup.clone());
                }
                scheduled_cleanups.lock().unwrap().push(cleanup);
            } else {
                // Max retries exceeded, remove cleanup
                {
                    let mut lookup = cleanup_lookup.write().unwrap();
                    lookup.remove(&cleanup_id);
                }

                if let Ok(mut stats) = stats.write() {
                    stats.active_scheduled = stats.active_scheduled.saturating_sub(1);
                }

                #[cfg(feature = "tracing")]
                warn!("Cleanup {:?} failed after maximum retries", cleanup_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::SchedulerConfig;
    use super::*;

    #[test]
    fn test_cleanup_id() {
        let id = CleanupId::new(42);
        assert_eq!(id.raw(), 42);

        let id_from_u64: CleanupId = 123.into();
        assert_eq!(id_from_u64.raw(), 123);

        let u64_from_id: u64 = id.into();
        assert_eq!(u64_from_id, 42);
    }

    #[test]
    fn test_scheduled_cleanup() {
        let id = CleanupId::new(1);
        let cleanup = ScheduledCleanup::new(
            id,
            CleanupStrategyType::Idle,
            CleanupPriority::Normal,
            Duration::from_millis(100),
        );

        assert_eq!(cleanup.id, id);
        assert_eq!(cleanup.strategy_type, CleanupStrategyType::Idle);
        assert_eq!(cleanup.priority, CleanupPriority::Normal);
        assert!(!cleanup.recurring);
        assert!(cleanup.active);
        assert_eq!(cleanup.retry_count, 0);
    }

    #[test]
    fn test_scheduled_cleanup_ordering() {
        let cleanup1 = ScheduledCleanup::new(
            CleanupId::new(1),
            CleanupStrategyType::Idle,
            CleanupPriority::Low,
            Duration::from_millis(100),
        );

        let cleanup2 = ScheduledCleanup::new(
            CleanupId::new(2),
            CleanupStrategyType::Pressure,
            CleanupPriority::High,
            Duration::from_millis(100),
        );

        // Higher priority should come first
        assert!(cleanup2 > cleanup1);
    }

    #[test]
    fn test_cleanup_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = CleanupScheduler::new(config).unwrap();

        assert!(!scheduler.is_running());
        assert_eq!(scheduler.active_cleanup_count(), 0);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_scheduled, 0);
        assert_eq!(stats.total_executed, 0);
    }

    #[test]
    fn test_schedule_cleanup() {
        let config = SchedulerConfig::default();
        let scheduler = CleanupScheduler::new(config).unwrap();

        let cleanup_id = scheduler
            .schedule_cleanup(
                CleanupStrategyType::Idle,
                CleanupPriority::Normal,
                Duration::from_millis(100),
            )
            .unwrap();

        assert_eq!(scheduler.active_cleanup_count(), 1);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_scheduled, 1);
        assert_eq!(stats.active_scheduled, 1);

        // Test cancellation
        assert!(scheduler.cancel_cleanup(cleanup_id));
        assert_eq!(scheduler.active_cleanup_count(), 0);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_cancelled, 1);
        assert_eq!(stats.active_scheduled, 0);
    }

    #[test]
    fn test_recurring_cleanup() {
        let config = SchedulerConfig::default();
        let scheduler = CleanupScheduler::new(config).unwrap();

        let _cleanup_id = scheduler
            .schedule_recurring_cleanup(
                CleanupStrategyType::Periodic,
                CleanupPriority::Normal,
                Duration::from_millis(100),
                Duration::from_secs(1),
            )
            .unwrap();

        assert_eq!(scheduler.active_cleanup_count(), 1);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_scheduled, 1);
    }
}
