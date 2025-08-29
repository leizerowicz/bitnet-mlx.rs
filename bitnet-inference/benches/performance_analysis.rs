//! Performance Analysis Benchmark
//!
//! This benchmark performs comprehensive performance analysis to identify
//! optimization opportunities and bottlenecks in the inference engine.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_inference::*;
use bitnet_inference::profiling::*;
use bitnet_core::{Tensor, Device, DType};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

/// Comprehensive memory usage analysis across different scenarios
fn benchmark_memory_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = MemoryProfiler::new();

    c.bench_function("memory_profile_small_model", |b| {
        b.to_async(&rt).iter(|| async {
            // Create small model scenario
            let engine = InferenceEngine::builder()
                .device(Device::Cpu)
                .build()
                .await
                .unwrap();
            
            let model = engine.load_model("test_model").await.unwrap();
            let inputs = vec![
                Tensor::ones(&[1, 128], DType::F32, &Device::Cpu).unwrap()
            ];

            // Profile memory usage
            let _profile = profiler.profile_inference_memory(&engine, &model, &inputs)
                .await
                .unwrap();
        });
    });

    c.bench_function("memory_profile_batch_analysis", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .device(Device::Cpu)
                .build()
                .await
                .unwrap();
            
            let model = engine.load_model("test_model").await.unwrap();
            let batch_sizes = vec![1, 4, 8, 16];

            // Profile batch memory scaling
            let _profiles = profiler.profile_batch_memory(&engine, &model, &batch_sizes)
                .await
                .unwrap();
        });
    });
}

/// Performance regression detection benchmark
fn benchmark_performance_regression_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let benchmark_scenarios = vec![
        ("small_input", 128),
        ("medium_input", 512),
        ("large_input", 1024),
    ];

    let mut group = c.benchmark_group("performance_regression");
    
    for (scenario_name, input_size) in benchmark_scenarios {
        group.bench_with_input(
            BenchmarkId::new("latency_baseline", scenario_name),
            &input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Cpu)
                        .optimization_level(OptimizationLevel::Aggressive)
                        .build()
                        .await
                        .unwrap();
                    
                    let model = engine.load_model("test_model").await.unwrap();
                    let input = Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap();
                    
                    let start = Instant::now();
                    let _output = engine.infer(&model, &input).await.unwrap();
                    let duration = start.elapsed();

                    // Assert performance expectations
                    assert!(duration < Duration::from_millis(100), 
                           "Inference took too long: {:?}", duration);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("throughput_baseline", scenario_name),
            &input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Cpu)
                        .optimization_level(OptimizationLevel::Aggressive)
                        .build()
                        .await
                        .unwrap();
                    
                    let model = engine.load_model("test_model").await.unwrap();
                    let batch_size = 8;
                    let inputs: Vec<_> = (0..batch_size)
                        .map(|_| Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap())
                        .collect();
                    
                    let start = Instant::now();
                    let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
                    let duration = start.elapsed();

                    // Calculate throughput
                    let throughput = batch_size as f64 / duration.as_secs_f64();
                    
                    // Assert throughput expectations (should be > 10 ops/sec for CPU)
                    assert!(throughput > 10.0, 
                           "Throughput too low: {:.2} ops/sec", throughput);
                });
            },
        );
    }

    group.finish();
}

/// Memory usage pattern analysis
fn benchmark_memory_pattern_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = MemoryProfiler::new();

    c.bench_function("memory_pattern_identification", |b| {
        b.to_async(&rt).iter(|| async {
            // Create multiple engines with different configurations
            let engines_and_configs = vec![
                ("aggressive_opt", OptimizationLevel::Aggressive),
                ("basic_opt", OptimizationLevel::Basic),
                ("no_opt", OptimizationLevel::None),
            ];

            let mut all_profiles = Vec::new();

            for (_name, opt_level) in engines_and_configs {
                let engine = InferenceEngine::builder()
                    .device(Device::Cpu)
                    .optimization_level(opt_level)
                    .build()
                    .await
                    .unwrap();

                let model = engine.load_model("test_model").await.unwrap();
                let inputs = vec![
                    Tensor::ones(&[1, 512], DType::F32, &Device::Cpu).unwrap();
                    4
                ];

                let profile = profiler.profile_inference_memory(&engine, &model, &inputs)
                    .await
                    .unwrap();
                all_profiles.push(profile);
            }

            // Analyze patterns across all profiles
            let _analysis = profiler.analyze_memory_patterns(&all_profiles);
        });
    });
}

/// Optimization opportunity identification
fn benchmark_optimization_identification(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = MemoryProfiler::new();

    c.bench_function("optimization_recommendation_generation", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate different workload patterns
            let workloads = vec![
                ("memory_intensive", vec![2048, 1024]), // Large inputs
                ("batch_intensive", vec![256; 16]),     // Many small inputs  
                ("mixed_workload", vec![128, 512, 1024, 256]), // Mixed sizes
            ];

            let mut all_profiles = Vec::new();

            for (_workload_name, input_sizes) in workloads {
                for &input_size in &input_sizes {
                    let engine = InferenceEngine::builder()
                        .device(Device::Cpu)
                        .build()
                        .await
                        .unwrap();

                    let model = engine.load_model("test_model").await.unwrap();
                    let input = vec![
                        Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap()
                    ];

                    if let Ok(profile) = profiler.profile_inference_memory(&engine, &model, &input).await {
                        all_profiles.push(profile);
                    }
                }
            }

            // Generate optimization recommendations
            let analysis = profiler.analyze_memory_patterns(&all_profiles);
            
            // Validate that recommendations are generated
            assert!(!analysis.optimization_recommendations.is_empty(),
                   "Should generate optimization recommendations");
        });
    });
}

/// Cross-device performance comparison analysis
fn benchmark_cross_device_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = MemoryProfiler::new();

    c.bench_function("cross_device_performance_comparison", |b| {
        b.to_async(&rt).iter(|| async {
            let input_size = 512;
            let batch_size = 8;

            // Profile CPU performance (baseline)
            let _device_profiles = profiler.profile_device_memory_comparison(input_size, batch_size)
                .await
                .unwrap();

            // This would compare CPU vs GPU performance in a real scenario
            // For now, we just ensure the profiling completes successfully
        });
    });
}

/// Performance bottleneck detection
fn benchmark_bottleneck_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("bottleneck_identification", |b| {
        b.to_async(&rt).iter(|| async {
            // Create scenarios that might expose bottlenecks
            let scenarios = vec![
                ("cpu_bound", Device::Cpu, 1024),
                ("memory_bound", Device::Cpu, 2048),
            ];

            for (scenario_name, device, input_size) in scenarios {
                let engine = InferenceEngine::builder()
                    .device(device)
                    .build()
                    .await
                    .unwrap();

                let model = engine.load_model("test_model").await.unwrap();
                
                // Time different components
                let load_start = Instant::now();
                let _model = engine.load_model("test_model").await.unwrap();
                let load_time = load_start.elapsed();

                let input = Tensor::ones(&[1, input_size], DType::F32, &Device::Cpu).unwrap();
                
                let inference_start = Instant::now();
                let _output = engine.infer(&model, &input).await.unwrap();
                let inference_time = inference_start.elapsed();

                // Log timing for analysis (in a real implementation)
                tracing::debug!("Scenario {}: Load time: {:?}, Inference time: {:?}",
                               scenario_name, load_time, inference_time);

                // Simple bottleneck detection heuristic
                if load_time > inference_time * 2 {
                    tracing::warn!("Potential model loading bottleneck in {}", scenario_name);
                }
            }
        });
    });
}

/// Memory efficiency analysis across different batch sizes
fn benchmark_memory_efficiency_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let profiler = MemoryProfiler::new();
    
    let mut group = c.benchmark_group("memory_efficiency_scaling");

    let batch_sizes = vec![1, 2, 4, 8, 16, 32];
    
    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_scaling", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::builder()
                        .device(Device::Cpu)
                        .build()
                        .await
                        .unwrap();

                    let model = engine.load_model("test_model").await.unwrap();
                    let inputs: Vec<_> = (0..batch_size)
                        .map(|_| Tensor::ones(&[1, 512], DType::F32, &Device::Cpu).unwrap())
                        .collect();

                    let profile = profiler.profile_inference_memory(&engine, &model, &inputs)
                        .await
                        .unwrap();

                    // Calculate memory efficiency metrics
                    let memory_per_input = profile.total_memory / batch_size;
                    let peak_ratio = profile.peak_memory as f64 / profile.total_memory as f64;

                    tracing::debug!("Batch size {}: {} bytes/input, peak ratio: {:.2}",
                                   batch_size, memory_per_input, peak_ratio);

                    // Assert reasonable memory scaling
                    assert!(peak_ratio < 2.0, "Peak memory usage too high relative to total");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    performance_analysis_benchmarks,
    benchmark_memory_analysis,
    benchmark_performance_regression_detection,
    benchmark_memory_pattern_analysis,
    benchmark_optimization_identification,
    benchmark_cross_device_analysis,
    benchmark_bottleneck_detection,
    benchmark_memory_efficiency_scaling
);

criterion_main!(performance_analysis_benchmarks);
