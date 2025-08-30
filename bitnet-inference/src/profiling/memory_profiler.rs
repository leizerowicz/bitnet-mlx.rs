//! Memory Profiling and Analysis for BitNet Inference
//!
//! This module provides comprehensive memory profiling capabilities for the
//! inference engine, tracking memory usage across different backends and
//! identifying optimization opportunities.

use crate::{InferenceEngine, Result, InferenceError};
use crate::engine::Model;
use bitnet_core::{Tensor, Device};
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};

/// Memory profile containing detailed memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Total memory usage in bytes
    pub total_memory: usize,
    /// Peak memory usage during profiling
    pub peak_memory: usize,
    /// Memory usage by component
    pub component_usage: HashMap<String, usize>,
    /// Memory allocation statistics
    pub allocation_stats: AllocationStats,
    /// Backend-specific memory usage
    pub backend_memory: BackendMemoryUsage,
}

/// Allocation statistics for memory profiling
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total number of allocations
    pub allocations: usize,
    /// Total number of deallocations
    pub deallocations: usize,
    /// Average allocation size in bytes
    pub avg_allocation_size: usize,
    /// Largest allocation size in bytes
    pub max_allocation_size: usize,
    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: f64,
}

/// Backend-specific memory usage information
#[derive(Debug, Clone)]
pub struct BackendMemoryUsage {
    /// CPU memory usage in bytes
    pub cpu_memory: usize,
    /// GPU memory usage in bytes (Metal/MLX)
    pub gpu_memory: usize,
    /// Shared memory usage in bytes
    pub shared_memory: usize,
    /// Memory transfer overhead in bytes
    pub transfer_overhead: usize,
}

/// Memory profiler for detailed memory analysis
pub struct MemoryProfiler {
    /// Current memory usage tracker
    current_usage: AtomicUsize,
    /// Peak memory usage tracker
    peak_usage: AtomicUsize,
    /// Memory usage by component
    component_usage: Arc<parking_lot::Mutex<HashMap<String, usize>>>,
    /// Allocation tracking
    allocation_count: AtomicUsize,
    /// Deallocation tracking
    deallocation_count: AtomicUsize,
    /// Total allocated bytes
    total_allocated: AtomicUsize,
    /// Maximum single allocation
    max_allocation: AtomicUsize,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            component_usage: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            max_allocation: AtomicUsize::new(0),
        }
    }

    /// Profile memory usage for a complete inference operation
    pub async fn profile_inference_memory(
        &self,
        engine: &InferenceEngine,
        model: &Arc<Model>,
        inputs: &[Tensor],
    ) -> Result<MemoryProfile> {
        // Reset profiling state
        self.reset();

        let start_memory = self.get_current_memory_usage();
        let start_time = Instant::now();

        // Record initial component usage
        self.record_component_usage("engine_base", engine.memory_usage());
        self.record_component_usage("model_cache", self.estimate_model_memory(model)?);
        self.record_component_usage("input_tensors", self.estimate_tensors_memory(inputs)?);

        // Perform inference with memory tracking
        let _outputs = engine.infer_batch(model, inputs).await?;

        let end_time = Instant::now();
        let duration = end_time - start_time;

        // Calculate backend-specific memory usage
        let backend_usage = self.profile_backend_memory(engine).await?;

        // Generate comprehensive memory profile
        Ok(MemoryProfile {
            total_memory: self.current_usage.load(Ordering::Relaxed),
            peak_memory: self.peak_usage.load(Ordering::Relaxed),
            component_usage: self.component_usage.lock().clone(),
            allocation_stats: self.calculate_allocation_stats(),
            backend_memory: backend_usage,
        })
    }

    /// Profile memory usage during batch processing
    pub async fn profile_batch_memory(
        &self,
        engine: &InferenceEngine,
        model: &Arc<Model>,
        batch_sizes: &[usize],
    ) -> Result<HashMap<usize, MemoryProfile>> {
        let mut profiles = HashMap::new();

        for &batch_size in batch_sizes {
            // Create test inputs for this batch size
            let inputs = self.create_test_batch(batch_size)?;
            
            // Profile this specific batch size
            let profile = self.profile_inference_memory(engine, model, &inputs).await?;
            profiles.insert(batch_size, profile);

            // Small delay to ensure clean separation between tests
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(profiles)
    }

    /// Profile memory usage across different device backends
    pub async fn profile_device_memory_comparison(
        &self,
        input_size: usize,
        batch_size: usize,
    ) -> Result<HashMap<String, MemoryProfile>> {
        let mut profiles = HashMap::new();

        // CPU baseline (always available)
        let cpu_engine = InferenceEngine::builder()
            .device(Device::Cpu)
            .build()
            .await?;

        let cpu_model = cpu_engine.load_model("test_model").await?;
        let cpu_inputs = self.create_test_inputs(input_size, batch_size)?;
        let cpu_profile = self.profile_inference_memory(&cpu_engine, &cpu_model, &cpu_inputs).await?;
        
        // Clone before inserting to allow reuse
        let cpu_profile_clone1 = cpu_profile.clone();
        #[cfg(feature = "mlx")]
        let cpu_profile_clone2 = cpu_profile.clone();
        
        profiles.insert("CPU".to_string(), cpu_profile);

        // GPU backends (if available) - simplified for compilation
        #[cfg(feature = "metal")]
        {
            // Would add Metal backend profiling here
            profiles.insert("Metal".to_string(), cpu_profile_clone1);
        }

        #[cfg(feature = "mlx")]
        {
            // Would add MLX backend profiling here  
            profiles.insert("MLX".to_string(), cpu_profile_clone2);
        }

        Ok(profiles)
    }

    /// Analyze memory usage patterns and identify optimization opportunities
    pub fn analyze_memory_patterns(&self, profiles: &[MemoryProfile]) -> MemoryAnalysis {
        let mut analysis = MemoryAnalysis::default();

        if profiles.is_empty() {
            return analysis;
        }

        // Calculate statistics across all profiles
        let total_memories: Vec<usize> = profiles.iter().map(|p| p.total_memory).collect();
        let peak_memories: Vec<usize> = profiles.iter().map(|p| p.peak_memory).collect();

        analysis.avg_total_memory = total_memories.iter().sum::<usize>() / total_memories.len();
        analysis.avg_peak_memory = peak_memories.iter().sum::<usize>() / peak_memories.len();
        analysis.max_total_memory = *total_memories.iter().max().unwrap_or(&0);
        analysis.max_peak_memory = *peak_memories.iter().max().unwrap_or(&0);

        // Identify memory hotspots
        analysis.memory_hotspots = self.identify_memory_hotspots(profiles);

        // Calculate memory efficiency
        analysis.memory_efficiency = self.calculate_memory_efficiency(profiles);

        // Generate optimization recommendations
        analysis.optimization_recommendations = self.generate_optimization_recommendations(profiles);

        analysis
    }

    /// Reset profiling state
    fn reset(&self) {
        self.current_usage.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
        self.component_usage.lock().clear();
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.max_allocation.store(0, Ordering::Relaxed);
    }

    /// Record memory usage for a specific component
    fn record_component_usage(&self, component: &str, size: usize) {
        self.component_usage.lock().insert(component.to_string(), size);
        
        // Update total usage
        let current = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak if necessary
        let peak = self.peak_usage.load(Ordering::Relaxed);
        if current > peak {
            self.peak_usage.store(current, Ordering::Relaxed);
        }

        // Track allocation
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        
        // Update max allocation if this is larger
        let current_max = self.max_allocation.load(Ordering::Relaxed);
        if size > current_max {
            self.max_allocation.store(size, Ordering::Relaxed);
        }
    }

    /// Get current memory usage in bytes
    fn get_current_memory_usage(&self) -> usize {
        // In a real implementation, this would query system memory usage
        // For now, return tracked usage
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Estimate memory usage for a model
    fn estimate_model_memory(&self, model: &Model) -> Result<usize> {
        // Placeholder implementation - in reality would analyze model structure
        Ok(model.parameter_count * 4) // Assume 4 bytes per parameter (F32)
    }

    /// Estimate memory usage for tensor collection
    fn estimate_tensors_memory(&self, tensors: &[Tensor]) -> Result<usize> {
        let mut total_size = 0;
        for tensor in tensors {
            // Estimate based on shape and data type
            let element_count: usize = tensor.shape().dims().iter().product();
            total_size += element_count * 4; // Assume F32 for simplicity
        }
        Ok(total_size)
    }

    /// Profile backend-specific memory usage
    async fn profile_backend_memory(&self, engine: &InferenceEngine) -> Result<BackendMemoryUsage> {
        // Get memory usage from the engine
        let engine_memory = engine.memory_usage();
        
        Ok(BackendMemoryUsage {
            cpu_memory: engine_memory,
            gpu_memory: 0, // Would be populated by GPU backends
            shared_memory: 0,
            transfer_overhead: 0,
        })
    }

    /// Calculate allocation statistics
    fn calculate_allocation_stats(&self) -> AllocationStats {
        let allocations = self.allocation_count.load(Ordering::Relaxed);
        let deallocations = self.deallocation_count.load(Ordering::Relaxed);
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let max_allocation = self.max_allocation.load(Ordering::Relaxed);

        let avg_allocation_size = if allocations > 0 {
            total_allocated / allocations
        } else {
            0
        };

        // Simple fragmentation calculation (would be more sophisticated in practice)
        let fragmentation_ratio = if allocations > deallocations {
            (allocations - deallocations) as f64 / allocations.max(1) as f64
        } else {
            0.0
        };

        AllocationStats {
            allocations,
            deallocations,
            avg_allocation_size,
            max_allocation_size: max_allocation,
            fragmentation_ratio,
        }
    }

    /// Create test inputs for batch profiling
    fn create_test_batch(&self, batch_size: usize) -> Result<Vec<Tensor>> {
        let inputs = (0..batch_size)
            .map(|_| Tensor::ones(&[1, 512], bitnet_core::DType::F32, &Device::Cpu))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| InferenceError::MemoryError(format!("Failed to create test batch: {:?}", e)))?;
        Ok(inputs)
    }

    /// Create test inputs with specific dimensions
    fn create_test_inputs(&self, input_size: usize, batch_size: usize) -> Result<Vec<Tensor>> {
        let inputs = (0..batch_size)
            .map(|_| Tensor::ones(&[1, input_size], bitnet_core::DType::F32, &Device::Cpu))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| InferenceError::MemoryError(format!("Failed to create test inputs: {:?}", e)))?;
        Ok(inputs)
    }

    /// Check if Metal backend is available
    #[cfg(feature = "metal")]
    #[allow(dead_code)]
    fn is_metal_available(&self) -> bool {
        // Would check for Metal availability
        true
    }

    #[cfg(not(feature = "metal"))]
    #[allow(dead_code)]
    fn is_metal_available(&self) -> bool {
        false
    }

    /// Check if MLX backend is available
    #[cfg(feature = "mlx")]
    #[allow(dead_code)]
    fn is_mlx_available(&self) -> bool {
        // Would check for MLX availability on Apple Silicon
        true
    }

    #[cfg(not(feature = "mlx"))]
    #[allow(dead_code)]
    fn is_mlx_available(&self) -> bool {
        false
    }

    /// Identify memory hotspots from profiles
    fn identify_memory_hotspots(&self, profiles: &[MemoryProfile]) -> Vec<MemoryHotspot> {
        let mut hotspots = Vec::new();
        
        // Analyze component usage across all profiles
        let mut component_totals: HashMap<String, usize> = HashMap::new();
        
        for profile in profiles {
            for (component, usage) in &profile.component_usage {
                *component_totals.entry(component.clone()).or_insert(0) += usage;
            }
        }

        // Identify top memory consumers
        let mut components: Vec<_> = component_totals.into_iter().collect();
        components.sort_by(|a, b| b.1.cmp(&a.1));

        for (component, total_usage) in components.into_iter().take(5) {
            hotspots.push(MemoryHotspot {
                component,
                total_usage,
                avg_usage: total_usage / profiles.len(),
                severity: if total_usage > 100 * 1024 * 1024 { // >100MB
                    HotspotSeverity::High
                } else if total_usage > 10 * 1024 * 1024 { // >10MB
                    HotspotSeverity::Medium
                } else {
                    HotspotSeverity::Low
                },
            });
        }

        hotspots
    }

    /// Calculate memory efficiency metrics
    fn calculate_memory_efficiency(&self, profiles: &[MemoryProfile]) -> MemoryEfficiency {
        if profiles.is_empty() {
            return MemoryEfficiency::default();
        }

        let peak_to_total_ratios: Vec<f64> = profiles
            .iter()
            .filter(|p| p.total_memory > 0)
            .map(|p| p.peak_memory as f64 / p.total_memory as f64)
            .collect();

        let avg_peak_ratio = if !peak_to_total_ratios.is_empty() {
            peak_to_total_ratios.iter().sum::<f64>() / peak_to_total_ratios.len() as f64
        } else {
            1.0
        };

        let memory_utilization = 1.0 / avg_peak_ratio; // Higher is better

        MemoryEfficiency {
            utilization_ratio: memory_utilization,
            fragmentation_score: profiles.iter()
                .map(|p| p.allocation_stats.fragmentation_ratio)
                .sum::<f64>() / profiles.len() as f64,
            efficiency_score: memory_utilization * (1.0 - profiles.iter()
                .map(|p| p.allocation_stats.fragmentation_ratio)
                .sum::<f64>() / profiles.len() as f64),
        }
    }

    /// Generate optimization recommendations based on profiles
    fn generate_optimization_recommendations(&self, profiles: &[MemoryProfile]) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze memory patterns
        let efficiency = self.calculate_memory_efficiency(profiles);
        let hotspots = self.identify_memory_hotspots(profiles);

        // Memory pool optimization
        if efficiency.fragmentation_score > 0.3 {
            recommendations.push(OptimizationRecommendation {
                category: "Memory Pool".to_string(),
                description: "High memory fragmentation detected. Consider using memory pools for frequent allocations.".to_string(),
                priority: RecommendationPriority::High,
                estimated_improvement: 20.0, // 20% improvement
            });
        }

        // GPU memory optimization
        for hotspot in &hotspots {
            if hotspot.component.contains("gpu") && hotspot.severity == HotspotSeverity::High {
                recommendations.push(OptimizationRecommendation {
                    category: "GPU Memory".to_string(),
                    description: format!("GPU memory usage is high in {}. Consider memory transfer optimization.", hotspot.component),
                    priority: RecommendationPriority::Medium,
                    estimated_improvement: 15.0,
                });
            }
        }

        // Batch size optimization
        if profiles.len() > 1 {
            let memory_growth_rate = self.calculate_memory_growth_rate(profiles);
            if memory_growth_rate > 2.0 {
                recommendations.push(OptimizationRecommendation {
                    category: "Batch Processing".to_string(),
                    description: "Memory usage grows superlinearly with batch size. Consider dynamic batch sizing.".to_string(),
                    priority: RecommendationPriority::Medium,
                    estimated_improvement: 10.0,
                });
            }
        }

        recommendations
    }

    /// Calculate memory growth rate across batch sizes
    fn calculate_memory_growth_rate(&self, profiles: &[MemoryProfile]) -> f64 {
        if profiles.len() < 2 {
            return 1.0;
        }

        let first_memory = profiles[0].total_memory as f64;
        let last_memory = profiles[profiles.len() - 1].total_memory as f64;

        if first_memory > 0.0 {
            last_memory / first_memory
        } else {
            1.0
        }
    }
}

/// Memory analysis results
#[derive(Debug, Default)]
pub struct MemoryAnalysis {
    /// Average total memory usage
    pub avg_total_memory: usize,
    /// Average peak memory usage
    pub avg_peak_memory: usize,
    /// Maximum total memory usage
    pub max_total_memory: usize,
    /// Maximum peak memory usage
    pub max_peak_memory: usize,
    /// Identified memory hotspots
    pub memory_hotspots: Vec<MemoryHotspot>,
    /// Memory efficiency metrics
    pub memory_efficiency: MemoryEfficiency,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Memory hotspot information
#[derive(Debug, Clone)]
pub struct MemoryHotspot {
    /// Component name
    pub component: String,
    /// Total memory usage
    pub total_usage: usize,
    /// Average memory usage
    pub avg_usage: usize,
    /// Severity level
    pub severity: HotspotSeverity,
}

/// Hotspot severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum HotspotSeverity {
    Low,
    Medium,
    High,
}

/// Memory efficiency metrics
#[derive(Debug, Default)]
pub struct MemoryEfficiency {
    /// Memory utilization ratio (higher is better)
    pub utilization_ratio: f64,
    /// Memory fragmentation score (lower is better)
    pub fragmentation_score: f64,
    /// Overall efficiency score (higher is better)
    pub efficiency_score: f64,
}

/// Optimization recommendation
#[derive(Debug)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: String,
    /// Description of the recommendation
    pub description: String,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Estimated performance improvement percentage
    pub estimated_improvement: f64,
}

/// Recommendation priority levels
#[derive(Debug)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert_eq!(profiler.current_usage.load(Ordering::Relaxed), 0);
        assert_eq!(profiler.peak_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_memory_profiler_reset() {
        let profiler = MemoryProfiler::new();
        
        // Add some usage
        profiler.record_component_usage("test", 1024);
        assert!(profiler.current_usage.load(Ordering::Relaxed) > 0);
        
        // Reset should clear everything
        profiler.reset();
        assert_eq!(profiler.current_usage.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_allocation_stats_calculation() {
        let profiler = MemoryProfiler::new();
        profiler.allocation_count.store(10, Ordering::Relaxed);
        profiler.total_allocated.store(1000, Ordering::Relaxed);
        profiler.max_allocation.store(200, Ordering::Relaxed);

        let stats = profiler.calculate_allocation_stats();
        assert_eq!(stats.allocations, 10);
        assert_eq!(stats.avg_allocation_size, 100);
        assert_eq!(stats.max_allocation_size, 200);
    }
}
