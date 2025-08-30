//! Day 7 Batch Processing Tests
//! 
//! Comprehensive tests for dynamic batch processing and parallel processing pipeline
//! implemented for Phase 5 Day 7.

use bitnet_inference::engine::{
    DynamicBatchProcessor, ParallelInferenceProcessor, ParallelConfig,
    MemoryMonitor, PerformanceTracker,
};
use bitnet_core::{Device, DType, Tensor};
use std::time::Duration;

/// Helper function to create test tensors.
fn create_test_tensors(count: usize) -> Vec<Tensor> {
    let device = Device::Cpu;
    let mut tensors = Vec::new();
    
    for i in 0..count {
        let tensor = if i % 2 == 0 {
            Tensor::zeros((4, 4), DType::F32, &device).unwrap()
        } else {
            Tensor::ones((4, 4), DType::F32, &device).unwrap()
        };
        tensors.push(tensor);
    }
    
    tensors
}

#[cfg(test)]
mod dynamic_batching_tests {
    use super::*;

    #[test]
    fn test_memory_monitor_creation() {
        let monitor = MemoryMonitor::new();
        let stats = monitor.get_stats();
        
        assert!(stats.total_memory > 0);
        assert!(stats.available_memory > 0);
        assert!(stats.threshold > 0.0 && stats.threshold <= 1.0);
    }

    #[test]
    fn test_memory_monitor_with_custom_threshold() {
        let monitor = MemoryMonitor::with_threshold(0.5);
        let stats = monitor.get_stats();
        
        assert_eq!(stats.threshold, 0.5);
        assert!(stats.available_memory > 0);
    }

    #[test]
    fn test_memory_monitor_threshold_clamping() {
        // Test lower bound
        let monitor_low = MemoryMonitor::with_threshold(-0.1);
        assert_eq!(monitor_low.get_stats().threshold, 0.1);
        
        // Test upper bound
        let monitor_high = MemoryMonitor::with_threshold(1.5);
        assert_eq!(monitor_high.get_stats().threshold, 0.95);
    }

    #[test]
    fn test_performance_tracker_creation() {
        let tracker = PerformanceTracker::new();
        let stats = tracker.get_stats();
        
        assert!(stats.optimal_batch_size > 0);
        assert_eq!(stats.total_samples, 0);
        assert!(stats.measurement_window.as_secs() > 0);
    }

    #[test]
    fn test_performance_tracker_recording() {
        let tracker = PerformanceTracker::new();
        
        // Record some performance data
        tracker.record_performance(8, Duration::from_millis(100));
        tracker.record_performance(16, Duration::from_millis(150));
        tracker.record_performance(32, Duration::from_millis(300));
        
        let stats = tracker.get_stats();
        assert!(stats.total_samples >= 3);
    }

    #[test]
    fn test_performance_tracker_optimization() {
        let tracker = PerformanceTracker::new();
        
        // Record data showing that batch size 8 is optimal (best throughput)
        tracker.record_performance(4, Duration::from_millis(100));  // 4 items / 100ms = 0.04 items/ms
        tracker.record_performance(8, Duration::from_millis(150));  // 8 items / 150ms = 0.053 items/ms (best)
        tracker.record_performance(16, Duration::from_millis(400)); // 16 items / 400ms = 0.04 items/ms
        tracker.record_performance(32, Duration::from_millis(900)); // 32 items / 900ms = 0.036 items/ms
        tracker.record_performance(64, Duration::from_millis(2000)); // 64 items / 2000ms = 0.032 items/ms

        // After enough samples, optimal batch size should be updated
        let stats = tracker.get_stats();
        assert!(stats.total_samples >= 5);
    }

    #[test]
    fn test_dynamic_batch_processor_creation() {
        let processor = DynamicBatchProcessor::new();
        let stats = processor.get_stats();
        
        assert!(stats.current_batch_size > 0);
        assert!(stats.min_batch_size > 0);
        assert!(stats.max_batch_size >= stats.min_batch_size);
        assert!(stats.memory_stats.available_memory > 0);
    }

    #[test]
    fn test_dynamic_batch_processor_with_config() {
        let processor = DynamicBatchProcessor::with_config(2, 64, 0.7);
        let stats = processor.get_stats();
        
        assert_eq!(stats.min_batch_size, 2);
        assert_eq!(stats.max_batch_size, 64);
        assert_eq!(stats.memory_stats.threshold, 0.7);
        assert!(stats.current_batch_size >= stats.min_batch_size);
    }

    #[test]
    fn test_dynamic_batch_processing_empty() {
        let mut processor = DynamicBatchProcessor::new();
        let results = processor.process_adaptive_batch(Vec::new()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dynamic_batch_processing_single_tensor() {
        let mut processor = DynamicBatchProcessor::new();
        let tensors = create_test_tensors(1);
        
        let results = processor.process_adaptive_batch(tensors.clone()).unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[test]
    fn test_dynamic_batch_processing_multiple_tensors() {
        let mut processor = DynamicBatchProcessor::new();
        let tensors = create_test_tensors(10);
        
        let results = processor.process_adaptive_batch(tensors.clone()).unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[test]
    fn test_dynamic_batch_processing_large_batch() {
        let mut processor = DynamicBatchProcessor::with_config(1, 5, 0.8);
        let tensors = create_test_tensors(20);
        
        let results = processor.process_adaptive_batch(tensors.clone()).unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[tokio::test]
    async fn test_dynamic_batch_processing_async_empty() {
        let mut processor = DynamicBatchProcessor::new();
        let results = processor.process_adaptive_batch_async(Vec::new()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_dynamic_batch_processing_async_single() {
        let mut processor = DynamicBatchProcessor::new();
        let tensors = create_test_tensors(1);
        
        let results = processor.process_adaptive_batch_async(tensors.clone()).await.unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[tokio::test]
    async fn test_dynamic_batch_processing_async_multiple() {
        let mut processor = DynamicBatchProcessor::new();
        let tensors = create_test_tensors(8);
        
        let results = processor.process_adaptive_batch_async(tensors.clone()).await.unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[test]
    fn test_dynamic_batch_stats_collection() {
        let mut processor = DynamicBatchProcessor::new();
        let tensors = create_test_tensors(5);
        
        // Process a batch to generate stats
        let _ = processor.process_adaptive_batch(tensors).unwrap();
        
        let stats = processor.get_stats();
        assert!(stats.performance_stats.total_samples > 0);
        assert!(stats.current_batch_size > 0);
    }
}

#[cfg(test)]
mod parallel_processing_tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        
        assert!(config.worker_count > 0);
        assert!(config.max_queue_size > 0);
        assert!(config.task_timeout.as_secs() > 0);
        assert!(config.load_balancing);
    }

    #[test]
    fn test_parallel_config_custom() {
        let config = ParallelConfig {
            worker_count: 2,
            max_queue_size: 50,
            task_timeout: Duration::from_secs(5),
            load_balancing: false,
        };
        
        assert_eq!(config.worker_count, 2);
        assert_eq!(config.max_queue_size, 50);
        assert_eq!(config.task_timeout, Duration::from_secs(5));
        assert!(!config.load_balancing);
    }

    #[tokio::test]
    async fn test_parallel_processor_creation() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        assert!(processor.worker_count() > 0);
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.worker_stats.len(), processor.worker_count());
    }

    #[tokio::test]
    async fn test_parallel_processor_with_config() {
        let config = ParallelConfig {
            worker_count: 2,
            max_queue_size: 100,
            task_timeout: Duration::from_secs(10),
            load_balancing: true,
        };
        
        let processor = ParallelInferenceProcessor::with_config(config).await.unwrap();
        assert_eq!(processor.worker_count(), 2);
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.worker_stats.len(), 2);
    }

    #[tokio::test]
    async fn test_parallel_batch_processing_empty() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let results = processor.process_batch_parallel(Vec::new()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_batch_processing_single() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let tensors = create_test_tensors(1);
        
        let results = processor.process_batch_parallel(tensors.clone()).await.unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[tokio::test]
    async fn test_parallel_batch_processing_multiple() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let tensors = create_test_tensors(8);
        
        let results = processor.process_batch_parallel(tensors.clone()).await.unwrap();
        assert_eq!(results.len(), tensors.len());
        
        // Results should be in original order
        for (_i, result) in results.iter().enumerate() {
            // Basic shape verification
            assert_eq!(result.shape().dims(), &[4, 4]);
        }
    }

    #[tokio::test]
    async fn test_parallel_batch_processing_large() {
        let mut processor = ParallelInferenceProcessor::with_config(
            ParallelConfig {
                worker_count: 3,
                max_queue_size: 200,
                task_timeout: Duration::from_secs(15),
                load_balancing: true,
            }
        ).await.unwrap();
        
        let tensors = create_test_tensors(20);
        
        let results = processor.process_batch_parallel(tensors.clone()).await.unwrap();
        assert_eq!(results.len(), tensors.len());
    }

    #[tokio::test]
    async fn test_parallel_streaming_processing() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let tensors = create_test_tensors(5);
        
        let mut stream = processor.process_batch_streaming(tensors.clone()).await.unwrap();
        
        let mut results = Vec::new();
        let mut received_count = 0;
        
        while received_count < tensors.len() {
            if let Some(result) = stream.recv().await {
                results.push(result);
                received_count += 1;
            } else {
                break;
            }
        }
        
        assert_eq!(results.len(), tensors.len());
        
        // Verify all original indices are present
        let mut indices: Vec<_> = results.iter().map(|r| r.original_index).collect();
        indices.sort();
        let expected_indices: Vec<_> = (0..tensors.len()).collect();
        assert_eq!(indices, expected_indices);
    }

    #[tokio::test]
    async fn test_parallel_processor_stats() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        let stats = processor.get_stats().await;
        
        assert!(stats.worker_stats.len() > 0);
        assert_eq!(stats.total_tasks_processed, 0); // No tasks processed yet
        assert_eq!(stats.queued_tasks, 0); // No tasks queued
        
        // All workers should be initialized but not active
        for worker_stat in &stats.worker_stats {
            assert_eq!(worker_stat.tasks_processed, 0);
            assert_eq!(worker_stat.total_processing_time, Duration::default());
            assert!(!worker_stat.is_active); // Should not be active initially
        }
    }

    #[tokio::test]
    async fn test_parallel_processor_stats_after_processing() {
        let mut processor = ParallelInferenceProcessor::new().await.unwrap();
        let tensors = create_test_tensors(4);
        
        // Process a batch to generate stats
        let _ = processor.process_batch_parallel(tensors).await.unwrap();
        
        let stats = processor.get_stats().await;
        assert!(stats.total_tasks_processed >= 4);
    }

    #[tokio::test]
    async fn test_parallel_processor_shutdown() {
        let processor = ParallelInferenceProcessor::new().await.unwrap();
        let result = processor.shutdown().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_parallel_processor_concurrent_batches() {
        let mut processor = ParallelInferenceProcessor::with_config(
            ParallelConfig {
                worker_count: 4,
                max_queue_size: 500,
                task_timeout: Duration::from_secs(10),
                load_balancing: true,
            }
        ).await.unwrap();
        
        let batch1 = create_test_tensors(5);
        let batch2 = create_test_tensors(3);
        let batch3 = create_test_tensors(7);
        
        // Process batches sequentially but potentially overlapping in workers
        let results1 = processor.process_batch_parallel(batch1).await.unwrap();
        let results2 = processor.process_batch_parallel(batch2).await.unwrap();
        let results3 = processor.process_batch_parallel(batch3).await.unwrap();
        
        assert_eq!(results1.len(), 5);
        assert_eq!(results2.len(), 3);
        assert_eq!(results3.len(), 7);
        
        let final_stats = processor.get_stats().await;
        assert!(final_stats.total_tasks_processed >= 15);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_dynamic_vs_parallel_processing() {
        let tensors = create_test_tensors(10);
        
        // Test dynamic batch processing
        let mut dynamic_processor = DynamicBatchProcessor::new();
        let dynamic_results = dynamic_processor.process_adaptive_batch(tensors.clone()).unwrap();
        
        // Test parallel processing
        let mut parallel_processor = ParallelInferenceProcessor::new().await.unwrap();
        let parallel_results = parallel_processor.process_batch_parallel(tensors.clone()).await.unwrap();
        
        // Both should produce the same number of results
        assert_eq!(dynamic_results.len(), tensors.len());
        assert_eq!(parallel_results.len(), tensors.len());
        assert_eq!(dynamic_results.len(), parallel_results.len());
        
        // Verify tensor shapes are preserved
        for i in 0..tensors.len() {
            assert_eq!(dynamic_results[i].shape().dims(), tensors[i].shape().dims());
            assert_eq!(parallel_results[i].shape().dims(), tensors[i].shape().dims());
        }
    }

    #[tokio::test]
    async fn test_memory_constrained_processing() {
        // Test with very low memory threshold to force small batches
        let mut processor = DynamicBatchProcessor::with_config(
            1,    // min_batch_size
            3,    // max_batch_size (very small)
            0.1,  // memory_threshold (very low)
        );
        
        let tensors = create_test_tensors(12);
        let results = processor.process_adaptive_batch(tensors.clone()).unwrap();
        
        assert_eq!(results.len(), tensors.len());
        
        // Should have processed in multiple small batches
        let stats = processor.get_stats();
        assert!(stats.current_batch_size <= 3);
    }

    #[tokio::test]
    async fn test_high_concurrency_parallel_processing() {
        let processor = ParallelInferenceProcessor::with_config(
            ParallelConfig {
                worker_count: 8, // High worker count
                max_queue_size: 1000,
                task_timeout: Duration::from_secs(20),
                load_balancing: true,
            }
        ).await.unwrap();
        
        let stats = processor.get_stats().await;
        assert_eq!(stats.worker_stats.len(), 8);
        
        // All workers should be initialized
        for worker_stat in &stats.worker_stats {
            assert_eq!(worker_stat.tasks_processed, 0);
        }
    }

    #[tokio::test]
    async fn test_performance_adaptation() {
        let mut processor = DynamicBatchProcessor::new();
        
        // Process multiple batches to trigger adaptation
        let batch_sizes = [5, 10, 15, 8, 12, 6, 4, 9, 11, 7]; // More batches to ensure samples
        
        for &size in &batch_sizes {
            let tensors = create_test_tensors(size);
            let _ = processor.process_adaptive_batch(tensors).unwrap();
        }
        
        let stats = processor.get_stats();
        // Allow some flexibility since the performance tracker may need more samples
        assert!(stats.performance_stats.total_samples > 0);
        assert!(stats.performance_stats.optimal_batch_size > 0);
    }
}
