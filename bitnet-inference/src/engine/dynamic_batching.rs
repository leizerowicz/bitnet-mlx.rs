//! Dynamic batch processing for optimal inference performance.

use crate::{InferenceError, Result};
use bitnet_core::Tensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Memory monitoring system for dynamic batch optimization.
#[derive(Debug, Clone)]
pub struct MemoryMonitor {
    /// Available system memory in bytes
    available_memory: Arc<AtomicUsize>,
    /// Memory usage threshold (0.0 to 1.0)
    memory_threshold: f64,
    /// Last memory check timestamp
    last_check: Arc<RwLock<Instant>>,
}

impl MemoryMonitor {
    /// Create a new memory monitor with default settings.
    pub fn new() -> Self {
        Self {
            available_memory: Arc::new(AtomicUsize::new(Self::detect_system_memory())),
            memory_threshold: 0.8, // Use up to 80% of available memory
            last_check: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Create a new memory monitor with custom threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        let mut monitor = Self::new();
        monitor.memory_threshold = threshold.clamp(0.1, 0.95);
        monitor
    }

    /// Get the currently available memory in bytes.
    pub fn available_memory(&self) -> usize {
        // Check if we need to update memory info (every 5 seconds)
        if let Ok(last_check) = self.last_check.read() {
            if last_check.elapsed() > Duration::from_secs(5) {
                drop(last_check);
                self.update_memory_info();
            }
        }

        let total_memory = self.available_memory.load(Ordering::Relaxed);
        ((total_memory as f64) * self.memory_threshold) as usize
    }

    /// Update system memory information.
    fn update_memory_info(&self) {
        let new_memory = Self::detect_system_memory();
        self.available_memory.store(new_memory, Ordering::Relaxed);
        
        if let Ok(mut last_check) = self.last_check.write() {
            *last_check = Instant::now();
        }
    }

    /// Detect system memory (simplified implementation).
    fn detect_system_memory() -> usize {
        #[cfg(target_os = "macos")]
        {
            // For macOS, estimate based on typical Apple Silicon configurations
            // In production, this would use system APIs
            8 * 1024 * 1024 * 1024 // 8GB default
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Default for other platforms
            4 * 1024 * 1024 * 1024 // 4GB default
        }
    }

    /// Get memory usage statistics.
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_memory: self.available_memory.load(Ordering::Relaxed),
            available_memory: self.available_memory(),
            threshold: self.memory_threshold,
        }
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance tracking system for batch size optimization.
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Batch size performance history (batch_size -> average_latency_ms)
    performance_history: Arc<RwLock<HashMap<usize, f64>>>,
    /// Current optimal batch size
    optimal_batch_size: Arc<AtomicUsize>,
    /// Performance measurement window
    measurement_window: Duration,
    /// Minimum samples needed for optimization
    min_samples: usize,
}

impl PerformanceTracker {
    /// Create a new performance tracker.
    pub fn new() -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            optimal_batch_size: Arc::new(AtomicUsize::new(16)), // Default starting batch size
            measurement_window: Duration::from_secs(60), // 1-minute window
            min_samples: 5,
        }
    }

    /// Record performance for a specific batch size.
    pub fn record_performance(&self, batch_size: usize, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;
        
        if let Ok(mut history) = self.performance_history.write() {
            let current_avg = history.get(&batch_size).copied().unwrap_or(0.0);
            // Simple exponential moving average
            let alpha = 0.1;
            let new_avg = if current_avg == 0.0 {
                latency_ms
            } else {
                alpha * latency_ms + (1.0 - alpha) * current_avg
            };
            
            history.insert(batch_size, new_avg);
            
            // Update optimal batch size if we have enough samples
            if history.len() >= self.min_samples {
                self.update_optimal_batch_size(&history);
            }
        }
    }

    /// Get the current optimal batch size.
    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size.load(Ordering::Relaxed)
    }

    /// Update the optimal batch size based on performance history.
    fn update_optimal_batch_size(&self, history: &HashMap<usize, f64>) {
        // Find batch size with best throughput (latency per item)
        let optimal = history
            .iter()
            .map(|(&batch_size, &avg_latency)| {
                let throughput = batch_size as f64 / avg_latency;
                (batch_size, throughput)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(batch_size, _)| batch_size)
            .unwrap_or(16);

        self.optimal_batch_size.store(optimal, Ordering::Relaxed);
    }

    /// Get performance statistics.
    pub fn get_stats(&self) -> PerformanceStats {
        let history = self.performance_history.read().unwrap();
        let samples = history.len();
        let optimal_batch_size = self.get_optimal_batch_size();
        
        PerformanceStats {
            optimal_batch_size,
            total_samples: samples,
            measurement_window: self.measurement_window,
        }
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic batch processor that adapts batch sizes based on system conditions.
#[derive(Debug)]
pub struct DynamicBatchProcessor {
    /// Memory monitoring system
    memory_monitor: MemoryMonitor,
    /// Performance tracking system
    performance_tracker: PerformanceTracker,
    /// Current batch size
    current_batch_size: AtomicUsize,
    /// Minimum allowed batch size
    min_batch_size: usize,
    /// Maximum allowed batch size
    max_batch_size: usize,
    /// Adaptation rate (how quickly to adjust batch sizes)
    adaptation_rate: f64,
}

impl DynamicBatchProcessor {
    /// Create a new dynamic batch processor with default settings.
    pub fn new() -> Self {
        Self {
            memory_monitor: MemoryMonitor::new(),
            performance_tracker: PerformanceTracker::new(),
            current_batch_size: AtomicUsize::new(16),
            min_batch_size: 1,
            max_batch_size: 128,
            adaptation_rate: 0.1,
        }
    }

    /// Create a new dynamic batch processor with custom configuration.
    pub fn with_config(
        min_batch_size: usize,
        max_batch_size: usize,
        memory_threshold: f64,
    ) -> Self {
        Self {
            memory_monitor: MemoryMonitor::with_threshold(memory_threshold),
            performance_tracker: PerformanceTracker::new(),
            current_batch_size: AtomicUsize::new(min_batch_size.max(1)),
            min_batch_size: min_batch_size.max(1),
            max_batch_size: max_batch_size.max(min_batch_size),
            adaptation_rate: 0.1,
        }
    }

    /// Process a batch with adaptive sizing.
    pub fn process_adaptive_batch(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let optimal_batch_size = self.calculate_optimal_batch_size(&inputs)?;
        
        let results = if inputs.len() <= optimal_batch_size {
            self.process_single_batch(inputs)?
        } else {
            // Process in optimally-sized chunks
            let mut results = Vec::new();
            for chunk in inputs.chunks(optimal_batch_size) {
                let chunk_results = self.process_single_batch(chunk.to_vec())?;
                results.extend(chunk_results);
            }
            results
        };

        // Record performance for future optimization
        let processing_time = start_time.elapsed();
        self.performance_tracker.record_performance(optimal_batch_size, processing_time);

        Ok(results)
    }

    /// Process a batch with async adaptive sizing.
    pub async fn process_adaptive_batch_async(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let optimal_batch_size = self.calculate_optimal_batch_size(&inputs)?;
        
        let results = if inputs.len() <= optimal_batch_size {
            self.process_single_batch_async(inputs).await?
        } else {
            // Process in optimally-sized chunks asynchronously
            let mut results = Vec::new();
            for chunk in inputs.chunks(optimal_batch_size) {
                let chunk_results = self.process_single_batch_async(chunk.to_vec()).await?;
                results.extend(chunk_results);
            }
            results
        };

        // Record performance for future optimization
        let processing_time = start_time.elapsed();
        self.performance_tracker.record_performance(optimal_batch_size, processing_time);

        Ok(results)
    }

    /// Calculate the optimal batch size for the given inputs.
    fn calculate_optimal_batch_size(&self, inputs: &[Tensor]) -> Result<usize> {
        let available_memory = self.memory_monitor.available_memory();
        let estimated_memory_per_tensor = self.estimate_memory_per_tensor(&inputs[0]);
        
        // Calculate memory-constrained batch size
        let memory_constrained_size = if estimated_memory_per_tensor > 0 {
            (available_memory / estimated_memory_per_tensor).max(1)
        } else {
            self.max_batch_size
        };
        
        // Get performance-optimal batch size
        let performance_optimal_size = self.performance_tracker.get_optimal_batch_size();
        
        // Choose the most restrictive constraint
        let optimal_size = memory_constrained_size
            .min(performance_optimal_size)
            .min(self.max_batch_size)
            .max(self.min_batch_size);

        // Update current batch size with adaptation rate
        let current = self.current_batch_size.load(Ordering::Relaxed);
        let adapted_size = if optimal_size != current {
            let diff = optimal_size as f64 - current as f64;
            let adjustment = diff * self.adaptation_rate;
            (current as f64 + adjustment).round() as usize
        } else {
            current
        };

        self.current_batch_size.store(adapted_size, Ordering::Relaxed);
        Ok(adapted_size)
    }

    /// Process a single batch synchronously.
    fn process_single_batch(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // TODO: This will be replaced with actual inference logic
        // For now, just clone the tensors as a placeholder
        Ok(inputs.into_iter().map(|t| t.clone()).collect())
    }

    /// Process a single batch asynchronously.
    async fn process_single_batch_async(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        // Process in a separate task to avoid blocking
        let results = tokio::task::spawn_blocking(move || {
            // TODO: This will be replaced with actual inference logic
            // For now, just clone the tensors as a placeholder
            inputs.into_iter().map(|t| t.clone()).collect::<Vec<_>>()
        })
        .await
        .map_err(|e| InferenceError::batch_processing(format!("Async processing failed: {}", e)))?;

        Ok(results)
    }

    /// Estimate memory usage per tensor.
    fn estimate_memory_per_tensor(&self, tensor: &Tensor) -> usize {
        let element_count: usize = tensor.shape().dims().iter().product();
        // Estimate: tensor size * 3 (input + intermediate + output)
        element_count * std::mem::size_of::<f32>() * 3
    }

    /// Get current processor statistics.
    pub fn get_stats(&self) -> DynamicBatchStats {
        DynamicBatchStats {
            current_batch_size: self.current_batch_size.load(Ordering::Relaxed),
            min_batch_size: self.min_batch_size,
            max_batch_size: self.max_batch_size,
            memory_stats: self.memory_monitor.get_stats(),
            performance_stats: self.performance_tracker.get_stats(),
        }
    }
}

impl Default for DynamicBatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory: usize,
    pub available_memory: usize,
    pub threshold: f64,
}

/// Performance tracking statistics.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub optimal_batch_size: usize,
    pub total_samples: usize,
    pub measurement_window: Duration,
}

/// Complete statistics for dynamic batch processor.
#[derive(Debug, Clone)]
pub struct DynamicBatchStats {
    pub current_batch_size: usize,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub memory_stats: MemoryStats,
    pub performance_stats: PerformanceStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::{Device, DType, Tensor};

    #[test]
    fn test_memory_monitor_creation() {
        let monitor = MemoryMonitor::new();
        let available = monitor.available_memory();
        assert!(available > 0);
        
        let stats = monitor.get_stats();
        assert!(stats.total_memory > 0);
        assert!(stats.available_memory > 0);
        assert!(stats.threshold > 0.0 && stats.threshold <= 1.0);
    }

    #[test]
    fn test_memory_monitor_with_threshold() {
        let monitor = MemoryMonitor::with_threshold(0.5);
        let stats = monitor.get_stats();
        assert_eq!(stats.threshold, 0.5);
    }

    #[test]
    fn test_performance_tracker_creation() {
        let tracker = PerformanceTracker::new();
        let optimal_size = tracker.get_optimal_batch_size();
        assert!(optimal_size > 0);
        
        let stats = tracker.get_stats();
        assert_eq!(stats.optimal_batch_size, optimal_size);
    }

    #[test]
    fn test_performance_tracker_recording() {
        let tracker = PerformanceTracker::new();
        
        // Record some performance data
        tracker.record_performance(8, Duration::from_millis(100));
        tracker.record_performance(16, Duration::from_millis(150));
        tracker.record_performance(32, Duration::from_millis(300));
        
        let stats = tracker.get_stats();
        assert!(stats.total_samples > 0);
    }

    #[test]
    fn test_dynamic_batch_processor_creation() {
        let processor = DynamicBatchProcessor::new();
        let stats = processor.get_stats();
        
        assert!(stats.current_batch_size > 0);
        assert!(stats.min_batch_size > 0);
        assert!(stats.max_batch_size >= stats.min_batch_size);
    }

    #[test]
    fn test_dynamic_batch_processor_with_config() {
        let processor = DynamicBatchProcessor::with_config(2, 64, 0.7);
        let stats = processor.get_stats();
        
        assert_eq!(stats.min_batch_size, 2);
        assert_eq!(stats.max_batch_size, 64);
        assert_eq!(stats.memory_stats.threshold, 0.7);
    }

    #[test]
    fn test_adaptive_batch_processing() {
        let mut processor = DynamicBatchProcessor::new();
        let device = Device::Cpu;
        
        // Create test tensors
        let inputs = vec![
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            Tensor::ones((2, 3), DType::F32, &device).unwrap(),
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        ];
        
        let results = processor.process_adaptive_batch(inputs.clone()).unwrap();
        assert_eq!(results.len(), inputs.len());
    }

    #[tokio::test]
    async fn test_adaptive_batch_processing_async() {
        let mut processor = DynamicBatchProcessor::new();
        let device = Device::Cpu;
        
        // Create test tensors
        let inputs = vec![
            Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            Tensor::ones((2, 3), DType::F32, &device).unwrap(),
        ];
        
        let results = processor.process_adaptive_batch_async(inputs.clone()).await.unwrap();
        assert_eq!(results.len(), inputs.len());
    }

    #[test]
    fn test_empty_batch_processing() {
        let mut processor = DynamicBatchProcessor::new();
        let results = processor.process_adaptive_batch(Vec::new()).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_empty_batch_processing_async() {
        let mut processor = DynamicBatchProcessor::new();
        let results = processor.process_adaptive_batch_async(Vec::new()).await.unwrap();
        assert!(results.is_empty());
    }
}
