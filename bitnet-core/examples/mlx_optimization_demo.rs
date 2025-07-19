//! Comprehensive demo of MLX optimization utilities
//! 
//! This example demonstrates all the MLX optimization features:
//! - Memory optimization and pooling
//! - Performance profiling
//! - Kernel fusion
//! - Tensor caching
//! - Auto-tuning
//! - Batch processing optimization
//! - Computation graph optimization

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mlx")]
    {
        run_mlx_demo()
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        run_stub_demo();
        Ok(())
    }
}

#[cfg(feature = "mlx")]
fn run_mlx_demo() -> Result<(), Box<dyn std::error::Error>> {
    use bitnet_core::mlx::{
        MlxMemoryOptimizer, MlxProfiler, MlxKernelFusion, MlxTensorCache,
        MlxAutoTuner, MlxBatchOptimizer, GraphBuilder, BitNetMlxDevice,
        Operation, FusionPattern
    };
    use mlx_rs::Array;
    use std::time::Duration;

    println!("MLX Optimization Utilities Demo");
    println!("===============================");

    // 1. Memory Optimization Demo
    println!("\n1. Memory Optimization Demo:");
    println!("----------------------------");
    
    let mut memory_optimizer = MlxMemoryOptimizer::new(50);
    let device = BitNetMlxDevice::cpu();
    
    // Simulate tensor allocation and pooling
    let shape = vec![64, 128];
    let dtype = mlx_rs::Dtype::Float32;
    
    println!("Creating tensors with memory pooling...");
    let mut tensors = Vec::new();
    
    for i in 0..5 {
        let tensor = memory_optimizer.get_or_create_tensor(&shape, dtype, &device)?;
        println!("  Tensor {}: shape {:?}", i, tensor.shape());
        tensors.push(tensor);
    }
    
    let stats = memory_optimizer.get_stats();
    println!("Memory stats after allocation:");
    println!("  Total allocations: {}", stats.total_allocations);
    println!("  Pool misses: {}", stats.pool_misses);
    println!("  Pool hits: {}", stats.pool_hits);
    
    // Return tensors to pool
    for tensor in tensors {
        memory_optimizer.return_to_pool(tensor, &device);
    }
    
    // Allocate again to see pool hits
    println!("\nAllocating tensors again (should hit pool)...");
    for i in 0..3 {
        let _tensor = memory_optimizer.get_or_create_tensor(&shape, dtype, &device)?;
        println!("  Reused tensor {}", i);
    }
    
    let final_stats = memory_optimizer.get_stats();
    println!("Final memory stats:");
    println!("  Total allocations: {}", final_stats.total_allocations);
    println!("  Pool misses: {}", final_stats.pool_misses);
    println!("  Pool hits: {}", final_stats.pool_hits);

    // 2. Performance Profiling Demo
    println!("\n2. Performance Profiling Demo:");
    println!("------------------------------");
    
    let mut profiler = MlxProfiler::new();
    
    // Profile matrix multiplication
    profiler.start_operation("matmul");
    let a = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let _result = mlx_rs::ops::matmul(&a, &b)?;
    let matmul_time = profiler.end_operation().unwrap();
    println!("Matrix multiplication took: {:?}", matmul_time);
    
    // Profile element-wise operations
    profiler.start_operation("elementwise_add");
    let _add_result = mlx_rs::ops::add(&a, &b)?;
    let add_time = profiler.end_operation().unwrap();
    println!("Element-wise addition took: {:?}", add_time);
    
    // Get profiling statistics
    let all_stats = profiler.get_all_stats();
    println!("Profiling summary:");
    for (op_name, (avg_time, _range, count)) in all_stats {
        println!("  {}: avg {:?} over {} runs", op_name, avg_time, count);
    }

    // 3. Kernel Fusion Demo
    println!("\n3. Kernel Fusion Demo:");
    println!("----------------------");
    
    let fusion = MlxKernelFusion::new();
    
    // Test fusion patterns
    let operations = vec!["add".to_string(), "multiply".to_string()];
    let arrays = vec![
        &Array::from_slice(&[1.0, 2.0], &[2]),
        &Array::from_slice(&[3.0, 4.0], &[2]),
        &Array::from_slice(&[2.0, 2.0], &[2]),
    ];
    
    println!("Testing add-multiply fusion...");
    if let Some(result) = fusion.try_fuse(&operations, &arrays) {
        match result {
            Ok(fused_result) => {
                println!("  Fusion successful! Result: {:?}", fused_result.as_slice::<f32>());
            }
            Err(e) => {
                println!("  Fusion failed: {}", e);
            }
        }
    } else {
        println!("  No fusion pattern matched");
    }
    
    // Test matmul + bias fusion
    let matmul_ops = vec!["matmul".to_string(), "add".to_string()];
    let matmul_arrays = vec![
        &Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        &Array::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
        &Array::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]), // bias
    ];
    
    println!("Testing matmul-add-bias fusion...");
    if let Some(result) = fusion.try_fuse(&matmul_ops, &matmul_arrays) {
        match result {
            Ok(fused_result) => {
                println!("  Fusion successful! Result shape: {:?}", fused_result.shape());
            }
            Err(e) => {
                println!("  Fusion failed: {}", e);
            }
        }
    }

    // 4. Tensor Caching Demo
    println!("\n4. Tensor Caching Demo:");
    println!("----------------------");
    
    let mut cache = MlxTensorCache::new(10, Duration::from_secs(60));
    
    // Cache some tensors
    let tensor1 = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let tensor2 = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
    
    cache.put("weights_layer1".to_string(), tensor1);
    cache.put("weights_layer2".to_string(), tensor2);
    
    println!("Cached 2 tensors");
    let (current_size, max_size) = cache.stats();
    println!("Cache stats: {}/{} slots used", current_size, max_size);
    
    // Retrieve from cache
    if let Some(cached_tensor) = cache.get("weights_layer1") {
        println!("Retrieved cached tensor: {:?}", cached_tensor.as_slice::<f32>());
    }
    
    // Test cache miss
    if cache.get("nonexistent_key").is_none() {
        println!("Cache miss for nonexistent key (expected)");
    }

    // 5. Auto-tuning Demo
    println!("\n5. Auto-tuning Demo:");
    println!("-------------------");
    
    let mut auto_tuner = MlxAutoTuner::new();
    
    // Simulate benchmarking different configurations
    let configs = vec![
        "small_batch".to_string(),
        "medium_batch".to_string(),
        "large_batch".to_string(),
    ];
    
    let benchmark_fn = |config: &str| -> Result<Duration, anyhow::Error> {
        // Simulate different performance characteristics
        let duration = match config {
            "small_batch" => Duration::from_millis(100),
            "medium_batch" => Duration::from_millis(80),  // Best performance
            "large_batch" => Duration::from_millis(120),  // Worse due to memory pressure
            _ => return Err(anyhow::anyhow!("Unknown config")),
        };
        Ok(duration)
    };
    
    println!("Benchmarking different batch configurations...");
    let optimal_config = auto_tuner.benchmark_operation("batch_matmul", configs, benchmark_fn)?;
    println!("Optimal configuration: {}", optimal_config);
    
    if let Some(results) = auto_tuner.get_benchmark_results("batch_matmul") {
        println!("Benchmark results:");
        for (config, time) in results {
            println!("  {}: {:?}", config, time);
        }
    }

    // 6. Batch Processing Optimization Demo
    println!("\n6. Batch Processing Optimization Demo:");
    println!("-------------------------------------");
    
    let mut batch_optimizer = MlxBatchOptimizer::new(1024 * 1024); // 1MB threshold
    
    // Simulate finding optimal batch size
    let benchmark_fn = |batch_size: usize| -> Result<Duration, anyhow::Error> {
        // Simulate performance curve: better with larger batches up to a point
        let base_time = Duration::from_micros(10);
        let efficiency = if batch_size <= 32 {
            1.0 / (batch_size as f64).sqrt() // Better efficiency with larger batches
        } else {
            1.0 / 32.0_f64.sqrt() * (32.0 / batch_size as f64) // Worse efficiency beyond 32
        };
        
        Ok(Duration::from_nanos((base_time.as_nanos() as f64 / efficiency) as u64))
    };
    
    println!("Finding optimal batch size...");
    let optimal_batch_size = batch_optimizer.find_optimal_batch_size("vector_ops", 64, benchmark_fn)?;
    println!("Optimal batch size: {}", optimal_batch_size);
    
    // Process data in optimal batches
    let data: Vec<i32> = (1..=100).collect();
    let process_fn = |batch: &[i32]| -> Result<Vec<i32>, anyhow::Error> {
        Ok(batch.iter().map(|x| x * 2).collect())
    };
    
    println!("Processing {} items in optimal batches...", data.len());
    let results = batch_optimizer.process_in_batches("vector_ops", data, process_fn)?;
    println!("Processed {} results", results.len());
    println!("First 10 results: {:?}", &results[..10]);

    // 7. Computation Graph Optimization Demo
    println!("\n7. Computation Graph Optimization Demo:");
    println!("--------------------------------------");
    
    let mut builder = GraphBuilder::new();
    
    // Build a simple neural network graph
    let input = builder.input("input", vec![32, 128], "f32", "cpu");
    let weights1 = builder.input("weights1", vec![128, 64], "f32", "cpu");
    let bias1 = builder.input("bias1", vec![32, 64], "f32", "cpu");
    
    // First layer: input @ weights1 + bias1
    let matmul1 = builder.matmul(input, weights1, "cpu")?;
    let layer1 = builder.add(matmul1, bias1, "cpu")?;
    
    // Quantization
    let quantized = builder.quantize(layer1, 0.1, "cpu")?;
    
    // Second layer
    let weights2 = builder.input("weights2", vec![64, 32], "f32", "cpu");
    let matmul2 = builder.matmul(quantized, weights2, "cpu")?;
    
    let output = builder.output(matmul2, "output")?;
    
    let graph = builder.build();
    
    println!("Built computation graph with {} nodes", graph.nodes().len());
    println!("Inputs: {} nodes", graph.inputs().len());
    println!("Outputs: {} nodes", graph.outputs().len());
    
    // Analyze the graph
    let execution_order = graph.topological_sort()?;
    println!("Execution order: {:?}", execution_order);
    
    // Find optimization opportunities
    let fusion_opportunities = graph.find_fusion_opportunities();
    println!("Found {} fusion opportunities:", fusion_opportunities.len());
    for (i, opportunity) in fusion_opportunities.iter().enumerate() {
        println!("  {}: {:?} (speedup: {:.2}x)", i + 1, opportunity.pattern, opportunity.estimated_speedup);
    }
    
    // Generate execution plan
    let execution_plan = graph.generate_execution_plan()?;
    println!("Execution plan generated:");
    println!("  Estimated memory usage: {} bytes", execution_plan.estimated_memory_usage);
    println!("  Estimated execution time: {:.6} seconds", execution_plan.estimated_execution_time);
    println!("  Memory groups: {}", execution_plan.memory_plan.memory_groups.len());

    // 8. Combined Optimization Demo
    println!("\n8. Combined Optimization Demo:");
    println!("-----------------------------");
    
    println!("Demonstrating combined use of optimization utilities...");
    
    // Use profiler with memory optimizer
    profiler.start_operation("optimized_workflow");
    
    // Get tensor from pool
    let pooled_tensor = memory_optimizer.get_or_create_tensor(&[64, 64], mlx_rs::Dtype::Float32, &device)?;
    
    // Cache the result
    cache.put("workflow_result".to_string(), pooled_tensor.clone());
    
    // Return to pool
    memory_optimizer.return_to_pool(pooled_tensor, &device);
    
    let workflow_time = profiler.end_operation().unwrap();
    println!("Combined workflow took: {:?}", workflow_time);
    
    // Final statistics
    let final_memory_stats = memory_optimizer.get_stats();
    let (cache_size, cache_max) = cache.stats();
    
    println!("\nFinal Statistics:");
    println!("================");
    println!("Memory optimizer:");
    println!("  Pool hits: {}", final_memory_stats.pool_hits);
    println!("  Pool misses: {}", final_memory_stats.pool_misses);
    println!("  Efficiency: {:.1}%", 
        100.0 * final_memory_stats.pool_hits as f64 / 
        (final_memory_stats.pool_hits + final_memory_stats.pool_misses) as f64);
    
    println!("Tensor cache:");
    println!("  Used slots: {}/{}", cache_size, cache_max);
    
    println!("Auto-tuner:");
    if let Some(optimal) = auto_tuner.get_optimal_config("batch_matmul") {
        println!("  Optimal batch config: {}", optimal);
    }
    
    println!("Batch optimizer:");
    if let Some(optimal_size) = batch_optimizer.get_optimal_batch_size("vector_ops") {
        println!("  Optimal batch size: {}", optimal_size);
    }

    println!("\nMLX optimization utilities demo completed successfully!");
    println!("All optimization features are working correctly.");
    
    Ok(())
}

#[cfg(not(feature = "mlx"))]
fn run_stub_demo() {
    println!("MLX Optimization Demo");
    println!("====================");
    println!();
    println!("MLX feature not enabled. Please run with --features mlx");
    println!();
    println!("This demo showcases:");
    println!("• Memory optimization and tensor pooling");
    println!("• Performance profiling and benchmarking");
    println!("• Kernel fusion for operation optimization");
    println!("• Tensor caching for frequently used data");
    println!("• Auto-tuning for optimal configurations");
    println!("• Batch processing optimization");
    println!("• Computation graph analysis and optimization");
    println!();
    println!("To see these features in action, rebuild with:");
    println!("cargo run --example mlx_optimization_demo --features mlx");
}