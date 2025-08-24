//! Minimal Test Benchmark
//!
//! This is a minimal benchmark to diagnose criterion benchmark registration issues.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn minimal_benchmark(c: &mut Criterion) {
    c.bench_function("minimal_test", |b| {
        b.iter(|| {
            let x = black_box(42);
            x * 2
        })
    });
}

fn simple_arithmetic_benchmark(c: &mut Criterion) {
    c.bench_function("simple_arithmetic", |b| {
        b.iter(|| {
            let a = black_box(100.0);
            let b = black_box(200.0);
            a + b + a * b
        })
    });
}

fn memory_allocation_benchmark(c: &mut Criterion) {
    c.bench_function("memory_allocation", |b| {
        b.iter(|| {
            let vec: Vec<i32> = black_box((0..1000).collect());
            vec.len()
        })
    });
}

criterion_group!(
    minimal_benches,
    minimal_benchmark,
    simple_arithmetic_benchmark,
    memory_allocation_benchmark
);
criterion_main!(minimal_benches);
