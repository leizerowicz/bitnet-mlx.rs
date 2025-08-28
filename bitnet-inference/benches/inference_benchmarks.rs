//! Comprehensive benchmarks for BitNet inference performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use bitnet_inference::*;
use bitnet_core::Tensor;
use std::time::Duration;

/// Benchmark single tensor inference performance.
fn benchmark_single_inference(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("single_inference_placeholder", |b| {
        b.to_async(&rt).iter(|| async {
            // Placeholder benchmark - will be expanded when full implementation is ready
            let engine = InferenceEngine::new().await.unwrap();
            let model = engine.load_model("test_model").await.unwrap();
            let input = Tensor::ones(&[1, 512]).unwrap();
            
            let _output = engine.infer(&model, &input).await.unwrap();
        })
    });
}

/// Benchmark batch inference performance.
fn benchmark_batch_inference(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("batch_inference");
    
    for batch_size in [1, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async move {
                    let engine = InferenceEngine::new().await.unwrap();
                    let model = engine.load_model("test_model").await.unwrap();
                    
                    let inputs: Vec<_> = (0..batch_size)
                        .map(|_| Tensor::ones(&[1, 512]).unwrap())
                        .collect();
                    
                    let _outputs = engine.infer_batch(&model, &inputs).await.unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark memory usage and cache performance.
fn benchmark_cache_performance(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("cache_performance", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::new().await.unwrap();
            
            // Load same model multiple times to test caching
            for i in 0..5 {
                let model_name = format!("test_model_{}", i % 2); // Alternate between 2 models
                let _model = engine.load_model(&model_name).await.unwrap();
            }
        })
    });
}

/// Benchmark different optimization levels.
fn benchmark_optimization_levels(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("optimization_levels");
    
    group.bench_function("none", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .optimization_level(OptimizationLevel::None)
                .build()
                .await
                .unwrap();
                
            let model = engine.load_model("test_model").await.unwrap();
            let input = Tensor::ones(&[1, 512]).unwrap();
            let _output = engine.infer(&model, &input).await.unwrap();
        })
    });
    
    group.bench_function("basic", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .optimization_level(OptimizationLevel::Basic)
                .build()
                .await
                .unwrap();
                
            let model = engine.load_model("test_model").await.unwrap();
            let input = Tensor::ones(&[1, 512]).unwrap();
            let _output = engine.infer(&model, &input).await.unwrap();
        })
    });
    
    group.bench_function("aggressive", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = InferenceEngine::builder()
                .optimization_level(OptimizationLevel::Aggressive)
                .build()
                .await
                .unwrap();
                
            let model = engine.load_model("test_model").await.unwrap();
            let input = Tensor::ones(&[1, 512]).unwrap();
            let _output = engine.infer(&model, &input).await.unwrap();
        })
    });
    
    group.finish();
}

/// Benchmark engine creation overhead.
fn benchmark_engine_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("engine_creation");
    
    group.bench_function("default", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::new().await.unwrap();
        })
    });
    
    group.bench_function("optimized_for_speed", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::optimized_for_speed().await.unwrap();
        })
    });
    
    group.bench_function("optimized_for_memory", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = InferenceEngine::optimized_for_memory().await.unwrap();
        })
    });
    
    group.finish();
}

criterion_group!(
    benches, 
    benchmark_single_inference,
    benchmark_batch_inference,
    benchmark_cache_performance,
    benchmark_optimization_levels,
    benchmark_engine_creation
);
criterion_main!(benches);
