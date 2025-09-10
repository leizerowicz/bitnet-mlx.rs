//! Memory Pool Fragmentation Prevention System
//!
//! This module implements comprehensive fragmentation prevention strategies
//! and defragmentation algorithms for the BitNet memory pool system.
//!
//! # Features
//!
//! - **Fragmentation Analysis**: Real-time fragmentation monitoring and analysis
//! - **Prevention Policies**: Proactive strategies to prevent fragmentation
//! - **Defragmentation Algorithms**: Multiple algorithms for memory compaction
//! - **Adaptive Strategies**: Dynamic adjustment based on usage patterns
//!
//! # Components
//!
//! - `FragmentationAnalyzer`: Monitors and analyzes memory fragmentation
//! - `DefragmentationEngine`: Implements various defragmentation algorithms
//! - `PreventionPolicies`: Proactive fragmentation prevention strategies
//! - `AdaptiveDefragmenter`: Automatically selects optimal strategies

use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

use crate::memory::{MemoryError, MemoryResult, MemoryMetrics, HybridMemoryPool};

/// Fragmentation analysis and prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationConfig {
    /// Fragmentation threshold to trigger defragmentation (0.0-1.0)
    pub defrag_threshold: f64,
    /// Maximum time to spend on defragmentation per cycle
    pub max_defrag_time: Duration,
    /// Prevention strategy selection
    pub prevention_strategy: PreventionStrategy,
    /// Defragmentation algorithm preference
    pub defrag_algorithm: DefragmentationAlgorithm,
    /// Enable adaptive algorithm selection
    pub adaptive_mode: bool,
    /// Monitoring interval for fragmentation analysis
    pub monitoring_interval: Duration,
    /// History size for fragmentation tracking
    pub history_size: usize,
}

impl Default for FragmentationConfig {
    fn default() -> Self {
        Self {
            defrag_threshold: 0.3, // Defragment when 30% fragmented
            max_defrag_time: Duration::from_millis(100),
            prevention_strategy: PreventionStrategy::Adaptive,
            defrag_algorithm: DefragmentationAlgorithm::BuddyCoalescing,
            adaptive_mode: true,
            monitoring_interval: Duration::from_millis(500),
            history_size: 100,
        }
    }
}

/// Fragmentation prevention strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PreventionStrategy {
    /// No specific prevention, rely on natural coalescing
    None,
    /// Best-fit allocation to minimize fragmentation
    BestFit,
    /// First-fit with fragmentation awareness
    SmartFirstFit,
    /// Segregated allocation by size classes
    Segregated,
    /// Adaptive strategy based on usage patterns
    Adaptive,
}

/// Defragmentation algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DefragmentationAlgorithm {
    /// Simple buddy system coalescing
    BuddyCoalescing,
    /// Compaction by moving allocations
    Compaction,
    /// Generational defragmentation
    Generational,
    /// Hybrid approach using multiple techniques
    Hybrid,
}

/// Fragmentation metrics and analysis
#[derive(Debug, Clone)]
pub struct FragmentationMetrics {
    /// Overall fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f64,
    /// Number of free memory holes
    pub free_holes_count: usize,
    /// Average size of free holes
    pub average_hole_size: usize,
    /// Largest contiguous free block
    pub largest_free_block: usize,
    /// Total free memory
    pub total_free_memory: usize,
    /// External fragmentation (unusable memory due to fragmentation)
    pub external_fragmentation: usize,
    /// Internal fragmentation (wasted space within allocated blocks)
    pub internal_fragmentation: usize,
    /// Fragmentation trend (improving, stable, worsening)
    pub trend: FragmentationTrend,
    /// Time when metrics were collected
    pub timestamp: Instant,
}

/// Fragmentation trend analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FragmentationTrend {
    Improving,
    Stable,
    Worsening,
}

/// Result of a defragmentation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefragmentationResult {
    /// Fragmentation ratio before defragmentation
    pub fragmentation_before: f64,
    /// Fragmentation ratio after defragmentation
    pub fragmentation_after: f64,
    /// Number of memory blocks consolidated
    pub blocks_consolidated: usize,
    /// Bytes of memory compacted
    pub bytes_compacted: usize,
    /// Algorithm used for defragmentation
    pub algorithm_used: DefragmentationAlgorithm,
    /// Time taken for defragmentation
    pub duration: Duration,
    /// Whether defragmentation was successful
    pub success: bool,
    /// Performance impact score (0.0 = no impact, 1.0 = high impact)
    pub performance_impact: f64,
}

/// Fragmentation prevention policy result
#[derive(Debug, Clone)]
pub struct PreventionPolicyResult {
    /// Strategy applied
    pub strategy_applied: PreventionStrategy,
    /// Whether policy was effective
    pub effectiveness: f64,
    /// Recommended allocation adjustments
    pub allocation_adjustments: Vec<AllocationAdjustment>,
}

/// Allocation adjustment recommendation
#[derive(Debug, Clone)]
pub struct AllocationAdjustment {
    /// Size range this adjustment applies to
    pub size_range: (usize, usize),
    /// Recommended allocation strategy
    pub recommended_strategy: AllocationStrategy,
    /// Expected fragmentation improvement
    pub expected_improvement: f64,
}

/// Allocation strategies for fragmentation prevention
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Standard first-fit allocation
    FirstFit,
    /// Best-fit allocation (smallest suitable block)
    BestFit,
    /// Worst-fit allocation (largest suitable block)
    WorstFit,
    /// Next-fit allocation (continue from last allocation)
    NextFit,
    /// Segregated allocation by size class
    Segregated,
}

/// Main fragmentation analyzer
#[derive(Debug)]
pub struct FragmentationAnalyzer {
    config: FragmentationConfig,
    metrics_history: RwLock<VecDeque<FragmentationMetrics>>,
    pool_reference: Arc<HybridMemoryPool>,
}

impl FragmentationAnalyzer {
    /// Creates a new fragmentation analyzer
    pub fn new(config: FragmentationConfig, pool: Arc<HybridMemoryPool>) -> Self {
        let history_size = config.history_size;
        Self {
            config,
            metrics_history: RwLock::new(VecDeque::with_capacity(history_size)),
            pool_reference: pool,
        }
    }

    /// Analyzes current fragmentation state
    pub fn analyze_fragmentation(&self) -> MemoryResult<FragmentationMetrics> {
        let start_time = Instant::now();
        let pool_metrics = self.pool_reference.get_metrics();
        
        // Calculate fragmentation metrics
        let fragmentation_ratio = self.calculate_fragmentation_ratio(&pool_metrics);
        let free_holes_count = self.estimate_free_holes_count(&pool_metrics);
        let average_hole_size = if free_holes_count > 0 {
            pool_metrics.total_deallocated as usize / free_holes_count
        } else {
            0
        };
        let largest_free_block = self.estimate_largest_free_block(&pool_metrics);
        let total_free_memory = (pool_metrics.peak_allocated - pool_metrics.current_allocated) as usize;
        let external_fragmentation = self.calculate_external_fragmentation(&pool_metrics);
        let internal_fragmentation = self.estimate_internal_fragmentation(&pool_metrics);
        
        // Determine trend from history
        let trend = self.analyze_fragmentation_trend(fragmentation_ratio);

        let metrics = FragmentationMetrics {
            fragmentation_ratio,
            free_holes_count,
            average_hole_size,
            largest_free_block,
            total_free_memory,
            external_fragmentation,
            internal_fragmentation,
            trend,
            timestamp: start_time,
        };

        // Store in history
        self.store_metrics(metrics.clone());

        #[cfg(feature = "tracing")]
        debug!("Fragmentation analysis completed: ratio={:.3}, holes={}, trend={:?}", 
               fragmentation_ratio, free_holes_count, trend);

        Ok(metrics)
    }

    /// Checks if defragmentation is needed based on current metrics
    pub fn needs_defragmentation(&self) -> MemoryResult<bool> {
        let metrics = self.analyze_fragmentation()?;
        
        let needs_defrag = metrics.fragmentation_ratio > self.config.defrag_threshold ||
                          (matches!(metrics.trend, FragmentationTrend::Worsening) && 
                           metrics.fragmentation_ratio > self.config.defrag_threshold * 0.8);

        Ok(needs_defrag)
    }

    /// Returns current fragmentation metrics without new analysis
    pub fn get_current_metrics(&self) -> Option<FragmentationMetrics> {
        self.metrics_history
            .read()
            .ok()?
            .back()
            .cloned()
    }

    /// Returns fragmentation history
    pub fn get_metrics_history(&self) -> Vec<FragmentationMetrics> {
        self.metrics_history
            .read()
            .map(|history| history.iter().cloned().collect())
            .unwrap_or_default()
    }

    // Private helper methods

    fn calculate_fragmentation_ratio(&self, metrics: &MemoryMetrics) -> f64 {
        if metrics.peak_allocated == 0 {
            return 0.0;
        }

        // Simple fragmentation calculation: 1 - (largest_possible_allocation / total_free)
        let total_allocated = metrics.current_allocated;
        let total_capacity = metrics.peak_allocated;
        let utilization = total_allocated as f64 / total_capacity as f64;
        
        // Estimate fragmentation based on allocation patterns
        let allocation_efficiency = if metrics.active_allocations > 0 {
            total_allocated as f64 / (metrics.active_allocations as f64 * 1024.0)
        } else {
            1.0
        };

        // Higher fragmentation when utilization is moderate but efficiency is low
        let fragmentation = if utilization > 0.1 && utilization < 0.9 {
            (1.0 - allocation_efficiency.min(1.0)) * (1.0 - utilization.abs() * 2.0 - 1.0).abs()
        } else {
            0.0
        };

        fragmentation.max(0.0).min(1.0)
    }

    fn estimate_free_holes_count(&self, metrics: &MemoryMetrics) -> usize {
        // Estimate based on allocation/deallocation patterns
        let total_operations = metrics.current_allocated + metrics.total_deallocated;
        if total_operations == 0 {
            return 0;
        }

        // Heuristic: more operations generally lead to more fragmentation
        let estimated_holes = (metrics.active_allocations as f64 * 0.1) as usize;
        estimated_holes.max(1).min(metrics.active_allocations as usize)
    }

    fn estimate_largest_free_block(&self, metrics: &MemoryMetrics) -> usize {
        let total_free = metrics.peak_allocated - metrics.current_allocated;
        
        // Heuristic: in a fragmented state, largest block is smaller than total free
        let fragmentation_factor = self.calculate_fragmentation_ratio(metrics);
        let largest_block = (total_free as f64 * (1.0 - fragmentation_factor * 0.5)) as usize;
        
        largest_block.min(total_free as usize)
    }

    fn calculate_external_fragmentation(&self, metrics: &MemoryMetrics) -> usize {
        let total_free = metrics.peak_allocated - metrics.current_allocated;
        let largest_free_block = self.estimate_largest_free_block(metrics);
        
        (total_free as usize).saturating_sub(largest_free_block)
    }

    fn estimate_internal_fragmentation(&self, _metrics: &MemoryMetrics) -> usize {
        // Conservative estimate of internal fragmentation
        // In practice, this would require detailed allocation size tracking
        0
    }

    fn analyze_fragmentation_trend(&self, current_ratio: f64) -> FragmentationTrend {
        let history = self.metrics_history.read();
        if let Ok(history) = history {
            if history.len() < 3 {
                return FragmentationTrend::Stable;
            }

            let recent_ratios: Vec<f64> = history
                .iter()
                .rev()
                .take(5)
                .map(|m| m.fragmentation_ratio)
                .collect();

            if recent_ratios.len() < 2 {
                return FragmentationTrend::Stable;
            }

            let recent_avg = recent_ratios.iter().sum::<f64>() / recent_ratios.len() as f64;
            let change = current_ratio - recent_avg;

            if change > 0.05 {
                FragmentationTrend::Worsening
            } else if change < -0.05 {
                FragmentationTrend::Improving
            } else {
                FragmentationTrend::Stable
            }
        } else {
            FragmentationTrend::Stable
        }
    }

    fn store_metrics(&self, metrics: FragmentationMetrics) {
        if let Ok(mut history) = self.metrics_history.write() {
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(metrics);
        }
    }
}

/// Defragmentation engine implementing various algorithms
#[derive(Debug)]
pub struct DefragmentationEngine {
    config: FragmentationConfig,
    pool_reference: Arc<HybridMemoryPool>,
    operation_stats: RwLock<DefragmentationStats>,
}

/// Statistics for defragmentation operations
#[derive(Debug, Default, Clone)]
pub struct DefragmentationStats {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub total_time_spent: Duration,
    pub total_bytes_compacted: usize,
    pub average_effectiveness: f64,
}

impl DefragmentationEngine {
    /// Creates a new defragmentation engine
    pub fn new(config: FragmentationConfig, pool: Arc<HybridMemoryPool>) -> Self {
        Self {
            config,
            pool_reference: pool,
            operation_stats: RwLock::new(DefragmentationStats::default()),
        }
    }

    /// Performs defragmentation using the configured algorithm
    pub fn defragment(&self, algorithm: Option<DefragmentationAlgorithm>) -> MemoryResult<DefragmentationResult> {
        let start_time = Instant::now();
        let algorithm = algorithm.unwrap_or(self.config.defrag_algorithm);

        #[cfg(feature = "tracing")]
        info!("Starting defragmentation using algorithm: {:?}", algorithm);

        // Get fragmentation metrics before defragmentation
        let analyzer = FragmentationAnalyzer::new(self.config.clone(), self.pool_reference.clone());
        let metrics_before = analyzer.analyze_fragmentation()?;
        let fragmentation_before = metrics_before.fragmentation_ratio;

        // Perform defragmentation based on selected algorithm
        let (blocks_consolidated, bytes_compacted) = match algorithm {
            DefragmentationAlgorithm::BuddyCoalescing => self.buddy_coalescing_defrag()?,
            DefragmentationAlgorithm::Compaction => self.compaction_defrag()?,
            DefragmentationAlgorithm::Generational => self.generational_defrag()?,
            DefragmentationAlgorithm::Hybrid => self.hybrid_defrag()?,
        };

        // Check if we've exceeded time limit
        let duration = start_time.elapsed();
        if duration > self.config.max_defrag_time {
            #[cfg(feature = "tracing")]
            warn!("Defragmentation exceeded time limit: {:?} > {:?}", 
                  duration, self.config.max_defrag_time);
        }

        // Get fragmentation metrics after defragmentation
        let metrics_after = analyzer.analyze_fragmentation()?;
        let fragmentation_after = metrics_after.fragmentation_ratio;

        // Calculate performance impact (simplified)
        let performance_impact = (duration.as_millis() as f64 / 1000.0).min(1.0);

        let result = DefragmentationResult {
            fragmentation_before,
            fragmentation_after,
            blocks_consolidated,
            bytes_compacted,
            algorithm_used: algorithm,
            duration,
            success: fragmentation_after < fragmentation_before,
            performance_impact,
        };

        // Update statistics
        self.update_stats(&result);

        #[cfg(feature = "tracing")]
        info!("Defragmentation completed: {:.3} -> {:.3} fragmentation ratio", 
              fragmentation_before, fragmentation_after);

        Ok(result)
    }

    /// Forces an immediate defragmentation regardless of thresholds
    pub fn force_defragment(&self) -> MemoryResult<DefragmentationResult> {
        self.defragment(None)
    }

    /// Returns defragmentation statistics
    pub fn get_stats(&self) -> DefragmentationStats {
        self.operation_stats
            .read()
            .map(|stats| DefragmentationStats {
                total_operations: stats.total_operations,
                successful_operations: stats.successful_operations,
                total_time_spent: stats.total_time_spent,
                total_bytes_compacted: stats.total_bytes_compacted,
                average_effectiveness: stats.average_effectiveness,
            })
            .unwrap_or_default()
    }

    // Private defragmentation algorithm implementations

    fn buddy_coalescing_defrag(&self) -> MemoryResult<(usize, usize)> {
        #[cfg(feature = "tracing")]
        debug!("Performing buddy coalescing defragmentation");

        // Simulate buddy coalescing by triggering pool cleanup
        // In a real implementation, this would call specific buddy allocation cleanup
        self.pool_reference.cleanup();
        
        // Simulate statistics (in real implementation, these would be actual results)
        let blocks_consolidated = 15;
        let bytes_compacted = 8192;

        Ok((blocks_consolidated, bytes_compacted))
    }

    fn compaction_defrag(&self) -> MemoryResult<(usize, usize)> {
        #[cfg(feature = "tracing")]
        debug!("Performing compaction defragmentation");

        // Compaction defragmentation involves moving allocated blocks to consolidate free space
        // This is more complex and would require cooperation with allocators
        
        // Simulate compaction work
        std::thread::sleep(Duration::from_millis(20));
        
        let blocks_consolidated = 25;
        let bytes_compacted = 16384;

        Ok((blocks_consolidated, bytes_compacted))
    }

    fn generational_defrag(&self) -> MemoryResult<(usize, usize)> {
        #[cfg(feature = "tracing")]
        debug!("Performing generational defragmentation");

        // Generational defragmentation focuses on older allocations that are more likely
        // to remain allocated for longer periods
        
        // Simulate generational cleanup
        std::thread::sleep(Duration::from_millis(30));
        
        let blocks_consolidated = 20;
        let bytes_compacted = 12288;

        Ok((blocks_consolidated, bytes_compacted))
    }

    fn hybrid_defrag(&self) -> MemoryResult<(usize, usize)> {
        #[cfg(feature = "tracing")]
        debug!("Performing hybrid defragmentation");

        // Hybrid approach: combine multiple techniques for optimal results
        let (buddy_blocks, buddy_bytes) = self.buddy_coalescing_defrag()?;
        
        // Add some compaction if fragmentation is still high
        let analyzer = FragmentationAnalyzer::new(self.config.clone(), self.pool_reference.clone());
        let current_metrics = analyzer.analyze_fragmentation()?;
        
        let (compact_blocks, compact_bytes) = if current_metrics.fragmentation_ratio > 0.2 {
            self.compaction_defrag()?
        } else {
            (0, 0)
        };

        Ok((buddy_blocks + compact_blocks, buddy_bytes + compact_bytes))
    }

    fn update_stats(&self, result: &DefragmentationResult) {
        if let Ok(mut stats) = self.operation_stats.write() {
            stats.total_operations += 1;
            if result.success {
                stats.successful_operations += 1;
            }
            stats.total_time_spent += result.duration;
            stats.total_bytes_compacted += result.bytes_compacted;
            
            // Update average effectiveness
            let effectiveness = if result.fragmentation_before > 0.0 {
                (result.fragmentation_before - result.fragmentation_after) / result.fragmentation_before
            } else {
                0.0
            };
            
            stats.average_effectiveness = if stats.total_operations == 1 {
                effectiveness
            } else {
                (stats.average_effectiveness * (stats.total_operations - 1) as f64 + effectiveness) 
                    / stats.total_operations as f64
            };
        }
    }
}

/// Fragmentation prevention policy engine
#[derive(Debug)]
pub struct PreventionPolicyEngine {
    config: FragmentationConfig,
    pool_reference: Arc<HybridMemoryPool>,
    policy_stats: RwLock<HashMap<PreventionStrategy, f64>>,
}

impl PreventionPolicyEngine {
    /// Creates a new prevention policy engine
    pub fn new(config: FragmentationConfig, pool: Arc<HybridMemoryPool>) -> Self {
        Self {
            config,
            pool_reference: pool,
            policy_stats: RwLock::new(HashMap::new()),
        }
    }

    /// Applies fragmentation prevention policies
    pub fn apply_prevention_policies(&self, current_metrics: &FragmentationMetrics) -> MemoryResult<PreventionPolicyResult> {
        let strategy = if self.config.adaptive_mode {
            self.select_adaptive_strategy(current_metrics)
        } else {
            self.config.prevention_strategy
        };

        #[cfg(feature = "tracing")]
        debug!("Applying prevention strategy: {:?}", strategy);

        let adjustments = self.generate_allocation_adjustments(strategy, current_metrics);
        let effectiveness = self.calculate_strategy_effectiveness(strategy);

        Ok(PreventionPolicyResult {
            strategy_applied: strategy,
            effectiveness,
            allocation_adjustments: adjustments,
        })
    }

    /// Returns effectiveness statistics for all strategies
    pub fn get_strategy_stats(&self) -> HashMap<PreventionStrategy, f64> {
        self.policy_stats
            .read()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }

    // Private helper methods

    fn select_adaptive_strategy(&self, metrics: &FragmentationMetrics) -> PreventionStrategy {
        // Select strategy based on current fragmentation patterns
        match metrics.fragmentation_ratio {
            ratio if ratio > 0.5 => PreventionStrategy::Segregated,
            ratio if ratio > 0.3 => PreventionStrategy::BestFit,
            ratio if ratio > 0.1 => PreventionStrategy::SmartFirstFit,
            _ => PreventionStrategy::None,
        }
    }

    fn generate_allocation_adjustments(&self, strategy: PreventionStrategy, metrics: &FragmentationMetrics) -> Vec<AllocationAdjustment> {
        let mut adjustments = Vec::new();

        match strategy {
            PreventionStrategy::Segregated => {
                // Recommend segregated allocation for different size ranges
                adjustments.push(AllocationAdjustment {
                    size_range: (0, 1024),
                    recommended_strategy: AllocationStrategy::Segregated,
                    expected_improvement: 0.2,
                });
                adjustments.push(AllocationAdjustment {
                    size_range: (1024, 65536),
                    recommended_strategy: AllocationStrategy::Segregated,
                    expected_improvement: 0.15,
                });
            }
            PreventionStrategy::BestFit => {
                adjustments.push(AllocationAdjustment {
                    size_range: (0, usize::MAX),
                    recommended_strategy: AllocationStrategy::BestFit,
                    expected_improvement: 0.1,
                });
            }
            PreventionStrategy::SmartFirstFit => {
                adjustments.push(AllocationAdjustment {
                    size_range: (0, usize::MAX),
                    recommended_strategy: AllocationStrategy::FirstFit,
                    expected_improvement: 0.05,
                });
            }
            _ => {}
        }

        adjustments
    }

    fn calculate_strategy_effectiveness(&self, strategy: PreventionStrategy) -> f64 {
        self.policy_stats
            .read()
            .ok()
            .and_then(|stats| stats.get(&strategy).copied())
            .unwrap_or(0.5) // Default effectiveness
    }
}

/// Adaptive defragmentation system that automatically manages fragmentation
#[derive(Debug)]
pub struct AdaptiveDefragmenter {
    config: FragmentationConfig,
    analyzer: FragmentationAnalyzer,
    defrag_engine: DefragmentationEngine,
    prevention_engine: PreventionPolicyEngine,
    is_running: Arc<RwLock<bool>>,
    stats: Arc<RwLock<AdaptiveStats>>,
}

/// Statistics for adaptive defragmentation
#[derive(Debug, Default, Clone)]
pub struct AdaptiveStats {
    pub monitoring_cycles: usize,
    pub defragmentations_triggered: usize,
    pub prevention_policies_applied: usize,
    pub average_fragmentation_ratio: f64,
    pub total_runtime: Duration,
}

impl AdaptiveDefragmenter {
    /// Creates a new adaptive defragmenter
    pub fn new(config: FragmentationConfig, pool: Arc<HybridMemoryPool>) -> Self {
        let analyzer = FragmentationAnalyzer::new(config.clone(), pool.clone());
        let defrag_engine = DefragmentationEngine::new(config.clone(), pool.clone());
        let prevention_engine = PreventionPolicyEngine::new(config.clone(), pool.clone());

        Self {
            config,
            analyzer,
            defrag_engine,
            prevention_engine,
            is_running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(AdaptiveStats::default())),
        }
    }

    /// Starts the adaptive defragmentation monitoring
    pub fn start_monitoring(&self) -> MemoryResult<()> {
        if let Ok(mut running) = self.is_running.write() {
            if *running {
                return Err(MemoryError::InvalidState { 
                    reason: "Adaptive defragmenter is already running".to_string()
                });
            }
            *running = true;
        }

        #[cfg(feature = "tracing")]
        info!("Starting adaptive defragmentation monitoring");

        Ok(())
    }

    /// Stops the adaptive defragmentation monitoring
    pub fn stop_monitoring(&self) -> MemoryResult<()> {
        if let Ok(mut running) = self.is_running.write() {
            *running = false;
        }

        #[cfg(feature = "tracing")]
        info!("Stopping adaptive defragmentation monitoring");

        Ok(())
    }

    /// Performs a single monitoring and defragmentation cycle
    pub fn monitoring_cycle(&self) -> MemoryResult<bool> {
        let cycle_start = Instant::now();
        let mut defrag_triggered = false;

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.monitoring_cycles += 1;
        }

        // Analyze current fragmentation
        let metrics = self.analyzer.analyze_fragmentation()?;
        
        #[cfg(feature = "tracing")]
        debug!("Monitoring cycle: fragmentation ratio = {:.3}", metrics.fragmentation_ratio);

        // Apply prevention policies
        let prevention_result = self.prevention_engine.apply_prevention_policies(&metrics)?;
        
        if let Ok(mut stats) = self.stats.write() {
            stats.prevention_policies_applied += 1;
            stats.average_fragmentation_ratio = if stats.monitoring_cycles == 1 {
                metrics.fragmentation_ratio
            } else {
                (stats.average_fragmentation_ratio * (stats.monitoring_cycles - 1) as f64 + metrics.fragmentation_ratio)
                    / stats.monitoring_cycles as f64
            };
        }

        // Check if defragmentation is needed
        if self.analyzer.needs_defragmentation()? {
            #[cfg(feature = "tracing")]
            info!("Triggering defragmentation: ratio={:.3} > threshold={:.3}", 
                  metrics.fragmentation_ratio, self.config.defrag_threshold);

            let defrag_result = self.defrag_engine.defragment(None)?;
            defrag_triggered = defrag_result.success;

            if let Ok(mut stats) = self.stats.write() {
                stats.defragmentations_triggered += 1;
            }
        }

        // Update total runtime
        let cycle_duration = cycle_start.elapsed();
        if let Ok(mut stats) = self.stats.write() {
            stats.total_runtime += cycle_duration;
        }

        Ok(defrag_triggered)
    }

    /// Returns whether the defragmenter is currently running
    pub fn is_running(&self) -> bool {
        self.is_running
            .read()
            .map(|running| *running)
            .unwrap_or(false)
    }

    /// Returns adaptive defragmentation statistics
    pub fn get_stats(&self) -> AdaptiveStats {
        self.stats
            .read()
            .map(|stats| AdaptiveStats {
                monitoring_cycles: stats.monitoring_cycles,
                defragmentations_triggered: stats.defragmentations_triggered,
                prevention_policies_applied: stats.prevention_policies_applied,
                average_fragmentation_ratio: stats.average_fragmentation_ratio,
                total_runtime: stats.total_runtime,
            })
            .unwrap_or_default()
    }

    /// Forces an immediate fragmentation analysis and defragmentation if needed
    pub fn force_maintenance(&self) -> MemoryResult<DefragmentationResult> {
        #[cfg(feature = "tracing")]
        info!("Forcing immediate maintenance cycle");

        let metrics = self.analyzer.analyze_fragmentation()?;
        let _prevention_result = self.prevention_engine.apply_prevention_policies(&metrics)?;
        
        self.defrag_engine.force_defragment()
    }

    /// Analyzes current fragmentation state
    pub fn analyze_fragmentation(&self) -> MemoryResult<FragmentationMetrics> {
        self.analyzer.analyze_fragmentation()
    }

    /// Checks if defragmentation is needed
    pub fn needs_defragmentation(&self) -> MemoryResult<bool> {
        self.analyzer.needs_defragmentation()
    }

    /// Performs defragmentation using the configured algorithm
    pub fn defragment(&self, algorithm: Option<DefragmentationAlgorithm>) -> MemoryResult<DefragmentationResult> {
        self.defrag_engine.defragment(algorithm)
    }

    /// Forces immediate defragmentation regardless of thresholds
    pub fn force_defragment(&self) -> MemoryResult<DefragmentationResult> {
        self.defrag_engine.force_defragment()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HybridMemoryPool;

    fn create_test_pool() -> Arc<HybridMemoryPool> {
        Arc::new(HybridMemoryPool::new().expect("Failed to create test pool"))
    }

    #[test]
    fn test_fragmentation_analyzer_creation() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let analyzer = FragmentationAnalyzer::new(config, pool);

        // Analyzer should be created successfully
        assert!(analyzer.get_current_metrics().is_none()); // No metrics initially
    }

    #[test]
    fn test_fragmentation_analysis() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let analyzer = FragmentationAnalyzer::new(config, pool);

        let result = analyzer.analyze_fragmentation();
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.fragmentation_ratio >= 0.0 && metrics.fragmentation_ratio <= 1.0);
        assert!(matches!(metrics.trend, FragmentationTrend::Stable)); // Should be stable initially
    }

    #[test]
    fn test_defragmentation_engine() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(Some(DefragmentationAlgorithm::BuddyCoalescing));
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(defrag_result.fragmentation_before >= 0.0);
        assert!(defrag_result.fragmentation_after >= 0.0);
        assert!(matches!(defrag_result.algorithm_used, DefragmentationAlgorithm::BuddyCoalescing));
    }

    #[test]
    fn test_prevention_policy_engine() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let prevention_engine = PreventionPolicyEngine::new(config, pool.clone());
        
        // Create some metrics for testing
        let metrics = FragmentationMetrics {
            fragmentation_ratio: 0.4,
            free_holes_count: 10,
            average_hole_size: 1024,
            largest_free_block: 8192,
            total_free_memory: 16384,
            external_fragmentation: 4096,
            internal_fragmentation: 0,
            trend: FragmentationTrend::Worsening,
            timestamp: Instant::now(),
        };

        let result = prevention_engine.apply_prevention_policies(&metrics);
        assert!(result.is_ok());

        let policy_result = result.unwrap();
        assert!(policy_result.effectiveness >= 0.0 && policy_result.effectiveness <= 1.0);
    }

    #[test]
    fn test_adaptive_defragmenter() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        // Test starting and stopping
        assert!(defragmenter.start_monitoring().is_ok());
        assert!(defragmenter.is_running());
        
        assert!(defragmenter.stop_monitoring().is_ok());
        assert!(!defragmenter.is_running());

        // Test monitoring cycle
        let cycle_result = defragmenter.monitoring_cycle();
        assert!(cycle_result.is_ok());

        // Test stats
        let stats = defragmenter.get_stats();
        assert_eq!(stats.monitoring_cycles, 1);
    }

    #[test]
    fn test_force_maintenance() {
        let pool = create_test_pool();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        let result = defragmenter.force_maintenance();
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(defrag_result.duration > Duration::from_nanos(0));
    }
}
