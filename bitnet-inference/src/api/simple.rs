//! Simple high-level API for easy inference operations.

use crate::{Result, InferenceError};
use crate::api::{InferenceEngine, EngineConfig};
use crate::engine::OptimizationLevel;
use bitnet_core::{Device, Tensor};
use std::sync::Arc;
use std::path::Path;

/// Simple inference API for basic use cases.
impl InferenceEngine {
    /// Quick inference on a single tensor with automatic model loading.
    pub async fn quick_infer<P: AsRef<Path>>(
        model_path: P,
        input: &Tensor,
    ) -> Result<Tensor> {
        let engine = Self::new().await?;
        let model = engine.load_model(model_path).await?;
        engine.infer(&model, input).await
    }

    /// Quick batch inference with automatic model loading.
    pub async fn quick_infer_batch<P: AsRef<Path>>(
        model_path: P,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let engine = Self::new().await?;
        let model = engine.load_model(model_path).await?;
        engine.infer_batch(&model, inputs).await
    }

    /// Create an inference engine optimized for speed.
    pub async fn optimized_for_speed() -> Result<Self> {
        let config = EngineConfig {
            optimization_level: OptimizationLevel::Aggressive,
            batch_size: 64,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create an inference engine optimized for memory usage.
    pub async fn optimized_for_memory() -> Result<Self> {
        let mut cache_config = crate::cache::CacheConfig::default();
        cache_config.max_memory = 512 * 1024 * 1024; // 512MB
        cache_config.max_models = 3;

        let config = EngineConfig {
            optimization_level: OptimizationLevel::None,
            batch_size: 8,
            cache_config,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Create an inference engine with balanced performance and memory usage.
    pub async fn balanced() -> Result<Self> {
        let config = EngineConfig {
            optimization_level: OptimizationLevel::Basic,
            batch_size: 32,
            ..Default::default()
        };
        Self::with_config(config).await
    }

    /// Run inference with automatic batching for optimal performance.
    pub async fn smart_infer<P: AsRef<Path>>(
        model_path: P,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let engine = Self::balanced().await?;
        let model = engine.load_model(model_path).await?;

        // Determine optimal batch size based on input count
        let optimal_batch_size = if inputs.len() <= 8 {
            inputs.len()
        } else if inputs.len() <= 32 {
            16
        } else {
            32
        };

        let mut results = Vec::with_capacity(inputs.len());
        
        for chunk in inputs.chunks(optimal_batch_size) {
            let chunk_results = engine.infer_batch(&model, chunk).await?;
            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Simple benchmark function to test inference performance.
    pub async fn benchmark<P: AsRef<Path>>(
        model_path: P,
        test_tensor: &Tensor,
        iterations: usize,
    ) -> Result<BenchmarkResults> {
        let start_time = std::time::Instant::now();
        
        let engine = Self::optimized_for_speed().await?;
        let model = engine.load_model(model_path).await?;

        let warmup_iterations = 5.min(iterations);
        
        // Warmup
        for _ in 0..warmup_iterations {
            let _ = engine.infer(&model, test_tensor).await?;
        }

        let benchmark_start = std::time::Instant::now();
        
        // Actual benchmark
        for _ in 0..iterations {
            let _ = engine.infer(&model, test_tensor).await?;
        }

        let benchmark_duration = benchmark_start.elapsed();
        let total_duration = start_time.elapsed();

        Ok(BenchmarkResults {
            total_iterations: iterations,
            warmup_iterations,
            benchmark_duration,
            total_duration_with_setup: total_duration,
            average_inference_time: benchmark_duration / iterations as u32,
            throughput_ops_per_sec: iterations as f64 / benchmark_duration.as_secs_f64(),
            memory_usage: engine.memory_usage(),
        })
    }
}

/// Results from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub total_iterations: usize,
    pub warmup_iterations: usize,
    pub benchmark_duration: std::time::Duration,
    pub total_duration_with_setup: std::time::Duration,
    pub average_inference_time: std::time::Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_usage: usize,
}

impl BenchmarkResults {
    /// Display benchmark results in a human-readable format.
    pub fn display(&self) -> String {
        format!(
            "Benchmark Results:\n\
             - Total iterations: {}\n\
             - Warmup iterations: {}\n\
             - Benchmark duration: {:?}\n\
             - Total duration (including setup): {:?}\n\
             - Average inference time: {:?}\n\
             - Throughput: {:.2} ops/sec\n\
             - Memory usage: {:.2} MB",
            self.total_iterations,
            self.warmup_iterations,
            self.benchmark_duration,
            self.total_duration_with_setup,
            self.average_inference_time,
            self.throughput_ops_per_sec,
            self.memory_usage as f64 / 1024.0 / 1024.0
        )
    }

    /// Check if the benchmark results meet performance targets.
    pub fn meets_targets(&self, target_ops_per_sec: f64, target_latency_ms: u64) -> bool {
        self.throughput_ops_per_sec >= target_ops_per_sec &&
        self.average_inference_time <= std::time::Duration::from_millis(target_latency_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation_variants() {
        // Test different creation methods
        let _balanced = InferenceEngine::balanced().await.unwrap();
        let _speed = InferenceEngine::optimized_for_speed().await.unwrap();
        let _memory = InferenceEngine::optimized_for_memory().await.unwrap();
    }

    #[test]
    fn test_benchmark_results() {
        let results = BenchmarkResults {
            total_iterations: 100,
            warmup_iterations: 5,
            benchmark_duration: std::time::Duration::from_millis(1000),
            total_duration_with_setup: std::time::Duration::from_millis(1200),
            average_inference_time: std::time::Duration::from_millis(10),
            throughput_ops_per_sec: 100.0,
            memory_usage: 64 * 1024 * 1024, // 64MB
        };

        assert!(results.meets_targets(50.0, 20)); // Should meet these targets
        assert!(!results.meets_targets(200.0, 5)); // Should not meet these targets
        
        let display = results.display();
        assert!(display.contains("100.00 ops/sec"));
        assert!(display.contains("64.00 MB"));
    }
}
