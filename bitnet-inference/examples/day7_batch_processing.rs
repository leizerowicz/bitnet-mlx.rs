//! Day 7 Batch Processing Implementation Example
//! 
//! This example demonstrates the advanced batch processing capabilities implemented
//! for Phase 5 Day 7, including dynamic batch size optimization and parallel processing.

use bitnet_inference::engine::{
    DynamicBatchProcessor, ParallelInferenceProcessor, ParallelConfig,
    MemoryMonitor, PerformanceTracker,
};
use bitnet_core::{Device, DType, Tensor};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet-Rust Phase 5 Day 7: Advanced Batch Processing");
    println!("=========================================================");
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Create test data
    let device = Device::Cpu;
    let test_tensors = create_test_tensors(&device, 20)?;
    
    println!("\n📊 Created {} test tensors for processing", test_tensors.len());

    // Section 1: Dynamic Batch Processing
    println!("\n🔄 Section 1: Dynamic Batch Size Optimization");
    println!("----------------------------------------------");
    demonstrate_dynamic_batching(test_tensors.clone()).await?;

    // Section 2: Parallel Processing Pipeline
    println!("\n⚡ Section 2: Parallel Processing Pipeline");
    println!("------------------------------------------");
    demonstrate_parallel_processing(test_tensors.clone()).await?;

    // Section 3: Performance Comparison
    println!("\n📈 Section 3: Performance Comparison");
    println!("------------------------------------");
    demonstrate_performance_comparison(test_tensors.clone()).await?;

    // Section 4: Memory and System Monitoring
    println!("\n🔍 Section 4: Memory and System Monitoring");
    println!("------------------------------------------");
    demonstrate_system_monitoring().await?;

    println!("\n✅ Day 7 Batch Processing Implementation Complete!");
    println!("🎯 Next: Day 8 - GPU Optimization Implementation");

    Ok(())
}

/// Create test tensors for demonstration.
fn create_test_tensors(device: &Device, count: usize) -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    let mut tensors = Vec::new();
    
    for i in 0..count {
        let tensor = if i % 2 == 0 {
            Tensor::zeros((8, 16), DType::F32, device)?
        } else {
            Tensor::ones((8, 16), DType::F32, device)?
        };
        tensors.push(tensor);
    }
    
    Ok(tensors)
}

/// Demonstrate dynamic batch size optimization.
async fn demonstrate_dynamic_batching(tensors: Vec<Tensor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Creating dynamic batch processor...");
    
    // Create dynamic batch processor with custom configuration
    let mut processor = DynamicBatchProcessor::with_config(
        2,    // min_batch_size
        16,   // max_batch_size  
        0.75, // memory_threshold
    );

    println!("📊 Initial processor statistics:");
    let initial_stats = processor.get_stats();
    println!("   - Current batch size: {}", initial_stats.current_batch_size);
    println!("   - Min/Max batch size: {}/{}", initial_stats.min_batch_size, initial_stats.max_batch_size);
    println!("   - Memory threshold: {:.1}%", initial_stats.memory_stats.threshold * 100.0);
    println!("   - Available memory: {:.2} MB", initial_stats.memory_stats.available_memory as f64 / (1024.0 * 1024.0));

    // Process batches with different sizes to show adaptation
    let test_cases = [
        ("Small batch", tensors[..4].to_vec()),
        ("Medium batch", tensors[..10].to_vec()),
        ("Large batch", tensors.clone()),
    ];

    for (name, batch) in test_cases {
        println!("\n🔍 Processing {}: {} tensors", name, batch.len());
        
        let start = Instant::now();
        let results = processor.process_adaptive_batch(batch.clone())?;
        let duration = start.elapsed();
        
        println!("   ✅ Processed {} tensors in {:.2}ms", results.len(), duration.as_secs_f64() * 1000.0);
        
        let stats = processor.get_stats();
        println!("   📊 Adapted batch size: {}", stats.current_batch_size);
        println!("   📊 Performance samples: {}", stats.performance_stats.total_samples);
    }

    // Demonstrate async processing
    println!("\n🔄 Testing async adaptive processing...");
    let start = Instant::now();
    let results = processor.process_adaptive_batch_async(tensors[..8].to_vec()).await?;
    let duration = start.elapsed();
    
    println!("   ✅ Async processed {} tensors in {:.2}ms", results.len(), duration.as_secs_f64() * 1000.0);

    Ok(())
}

/// Demonstrate parallel processing pipeline.
async fn demonstrate_parallel_processing(tensors: Vec<Tensor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Creating parallel processing pipeline...");
    
    // Create parallel processor with custom configuration
    let config = ParallelConfig {
        worker_count: 4,
        max_queue_size: 100,
        task_timeout: Duration::from_secs(10),
        load_balancing: true,
    };
    
    let mut processor = ParallelInferenceProcessor::with_config(config).await?;
    
    println!("📊 Parallel processor configuration:");
    println!("   - Worker count: {}", processor.worker_count());
    
    let initial_stats = processor.get_stats().await;
    println!("   - Active workers: {}", initial_stats.active_workers);
    println!("   - Worker statistics: {} workers initialized", initial_stats.worker_stats.len());

    // Process batches in parallel
    let test_batches = [
        ("Small parallel batch", tensors[..6].to_vec()),
        ("Large parallel batch", tensors.clone()),
    ];

    for (name, batch) in test_batches {
        println!("\n🔍 Processing {}: {} tensors", name, batch.len());
        
        let start = Instant::now();
        let results = processor.process_batch_parallel(batch.clone()).await?;
        let duration = start.elapsed();
        
        println!("   ✅ Parallel processed {} tensors in {:.2}ms", results.len(), duration.as_secs_f64() * 1000.0);
        
        let stats = processor.get_stats().await;
        println!("   📊 Total tasks processed: {}", stats.total_tasks_processed);
        println!("   📊 Throughput: {:.1} tasks/sec", stats.throughput);
        
        // Show worker statistics
        for worker_stat in stats.worker_stats.iter().take(3) {
            println!("   👷 Worker {}: {} tasks, avg {:.2}ms/task", 
                worker_stat.worker_id, 
                worker_stat.tasks_processed,
                worker_stat.average_processing_time.as_secs_f64() * 1000.0
            );
        }
    }

    // Demonstrate streaming processing
    println!("\n🌊 Testing streaming processing...");
    let start = Instant::now();
    let mut stream = processor.process_batch_streaming(tensors[..5].to_vec()).await?;
    
    let mut results_count = 0;
    while let Some(result) = stream.recv().await {
        results_count += 1;
        println!("   📨 Received result {} from worker {} (processed in {:.2}ms)", 
            result.original_index + 1,
            result.worker_id,
            result.processing_time.as_secs_f64() * 1000.0
        );
        
        if results_count >= 5 {
            break;
        }
    }
    
    let duration = start.elapsed();
    println!("   ✅ Streaming processed {} results in {:.2}ms", results_count, duration.as_secs_f64() * 1000.0);

    // Shutdown the processor
    processor.shutdown().await?;
    println!("   🔧 Parallel processor shutdown complete");

    Ok(())
}

/// Demonstrate performance comparison between different approaches.
async fn demonstrate_performance_comparison(tensors: Vec<Tensor>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Running performance comparison...");
    
    let test_batch = tensors[..12].to_vec();
    
    // Test 1: Dynamic batching
    println!("\n🔄 Test 1: Dynamic Batch Processing");
    let mut dynamic_processor = DynamicBatchProcessor::new();
    
    let start = Instant::now();
    let results = dynamic_processor.process_adaptive_batch(test_batch.clone())?;
    let dynamic_time = start.elapsed();
    
    println!("   ✅ Dynamic: {} tensors in {:.2}ms", results.len(), dynamic_time.as_secs_f64() * 1000.0);

    // Test 2: Parallel processing
    println!("\n⚡ Test 2: Parallel Processing");
    let mut parallel_processor = ParallelInferenceProcessor::new().await?;
    
    let start = Instant::now();
    let results = parallel_processor.process_batch_parallel(test_batch.clone()).await?;
    let parallel_time = start.elapsed();
    
    println!("   ✅ Parallel: {} tensors in {:.2}ms", results.len(), parallel_time.as_secs_f64() * 1000.0);

    // Test 3: Sequential processing (baseline)
    println!("\n📝 Test 3: Sequential Processing (Baseline)");
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    for tensor in test_batch.iter() {
        // Simulate processing time
        tokio::time::sleep(Duration::from_micros(100)).await;
        sequential_results.push(tensor.clone());
    }
    let sequential_time = start.elapsed();
    
    println!("   ✅ Sequential: {} tensors in {:.2}ms", sequential_results.len(), sequential_time.as_secs_f64() * 1000.0);

    // Performance summary
    println!("\n📊 Performance Summary:");
    println!("   🏃 Dynamic Batch: {:.2}ms", dynamic_time.as_secs_f64() * 1000.0);
    println!("   ⚡ Parallel:      {:.2}ms", parallel_time.as_secs_f64() * 1000.0);
    println!("   📝 Sequential:    {:.2}ms", sequential_time.as_secs_f64() * 1000.0);

    let speedup_dynamic = sequential_time.as_secs_f64() / dynamic_time.as_secs_f64();
    let speedup_parallel = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    
    println!("   📈 Dynamic speedup:  {:.1}x", speedup_dynamic);
    println!("   📈 Parallel speedup: {:.1}x", speedup_parallel);

    parallel_processor.shutdown().await?;

    Ok(())
}

/// Demonstrate system monitoring capabilities.
async fn demonstrate_system_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Demonstrating system monitoring capabilities...");

    // Memory monitoring
    println!("\n💾 Memory Monitoring:");
    let memory_monitor = MemoryMonitor::new();
    let memory_stats = memory_monitor.get_stats();
    
    println!("   📊 Total system memory: {:.1} GB", memory_stats.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   📊 Available for inference: {:.1} GB", memory_stats.available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   📊 Memory threshold: {:.1}%", memory_stats.threshold * 100.0);

    // Custom threshold testing
    let custom_monitor = MemoryMonitor::with_threshold(0.6);
    let custom_stats = custom_monitor.get_stats();
    println!("   📊 Custom threshold (60%): {:.1} GB available", custom_stats.available_memory as f64 / (1024.0 * 1024.0 * 1024.0));

    // Performance tracking
    println!("\n⏱️ Performance Tracking:");
    let performance_tracker = PerformanceTracker::new();
    
    // Simulate some performance recordings
    performance_tracker.record_performance(8, Duration::from_millis(50));
    performance_tracker.record_performance(16, Duration::from_millis(85));
    performance_tracker.record_performance(32, Duration::from_millis(150));
    performance_tracker.record_performance(8, Duration::from_millis(48));
    performance_tracker.record_performance(16, Duration::from_millis(82));
    
    let perf_stats = performance_tracker.get_stats();
    println!("   📊 Optimal batch size: {}", perf_stats.optimal_batch_size);
    println!("   📊 Performance samples: {}", perf_stats.total_samples);
    println!("   📊 Measurement window: {:?}", perf_stats.measurement_window);

    // System resource information
    println!("\n🖥️ System Resources:");
    println!("   📊 CPU cores detected: {}", num_cpus::get());
    println!("   📊 Logical CPUs: {}", num_cpus::get_physical());

    Ok(())
}
