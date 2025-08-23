use criterion::{criterion_group, criterion_main, Criterion};
use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::tensor::ops::arithmetic::{add, mul, add_scalar};
use bitnet_core::device::get_cpu_device;
use std::time::Instant;

fn create_test_tensor(shape: &[usize]) -> Result<BitNetTensor, Box<dyn std::error::Error>> {
    let device = get_cpu_device();
    BitNetTensor::zeros(shape, BitNetDType::F32, Some(device)).map_err(Into::into)
}

fn benchmark_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");
    
    let shapes = vec![
        vec![1024, 1024],
        vec![2048, 2048],
        vec![4096, 4096],
    ];
    
    for shape in shapes {
        let size_str = format!("{}x{}", shape[0], shape[1]);
        
        // CPU benchmark
        group.bench_function(format!("cpu_{}", size_str), |b| {
            b.iter(|| {
                let tensor = create_test_tensor(&shape).unwrap();
                // Simulate quantization operation with scalar addition
                let start = Instant::now();
                let _result = add_scalar(&tensor, 1.0);
                start.elapsed()
            })
        });
        
        // GPU benchmark placeholder (when GPU support is available)
        group.bench_function(format!("gpu_{}", size_str), |b| {
            b.iter(|| {
                // TODO: Replace with actual GPU implementation when available
                let tensor = create_test_tensor(&shape).unwrap();
                let start = Instant::now();
                let _result = add_scalar(&tensor, 1.0);
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

fn benchmark_bitlinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitlinear");
    
    let shapes = vec![
        vec![128, 512],
        vec![256, 1024], 
        vec![512, 2048],
    ];
    
    for shape in shapes {
        let size_str = format!("{}x{}", shape[0], shape[1]);
        
        // CPU benchmark
        group.bench_function(format!("cpu_{}", size_str), |b| {
            b.iter(|| {
                let input = create_test_tensor(&shape).unwrap();
                let weight = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                // Simulate BitLinear operation using available tensor operations
                let _result = add(&input, &weight);
                start.elapsed()
            })
        });
        
        // GPU benchmark placeholder
        group.bench_function(format!("gpu_{}", size_str), |b| {
            b.iter(|| {
                let input = create_test_tensor(&shape).unwrap();
                let weight = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                // TODO: Replace with actual GPU BitLinear when available
                let _result = add(&input, &weight);
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    let shapes = vec![
        vec![256, 512],
        vec![512, 1024],
        vec![1024, 2048],
    ];
    
    for shape in shapes {
        let size_str = format!("{}x{}", shape[0], shape[1]);
        
        // CPU benchmark
        group.bench_function(format!("cpu_{}", size_str), |b| {
            b.iter(|| {
                let a = create_test_tensor(&shape).unwrap();
                let b = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                // Simulate matrix multiplication using tensor operations
                let _result = mul(&a, &b);
                start.elapsed()
            })
        });
        
        // GPU benchmark placeholder
        group.bench_function(format!("gpu_{}", size_str), |b| {
            b.iter(|| {
                let a = create_test_tensor(&shape).unwrap();
                let b = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                // TODO: Replace with actual GPU matmul when available
                let _result = mul(&a, &b);
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

fn benchmark_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_operations");
    
    let shapes = vec![
        vec![1024, 1024],
        vec![2048, 2048],
        vec![4096, 4096],
    ];
    
    for shape in shapes {
        let size_str = format!("{}x{}", shape[0], shape[1]);
        
        // CPU benchmark
        group.bench_function(format!("cpu_add_{}", size_str), |b| {
            b.iter(|| {
                let a = create_test_tensor(&shape).unwrap();
                let b = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                let _result = add(&a, &b);
                start.elapsed()
            })
        });
        
        // GPU benchmark placeholder
        group.bench_function(format!("gpu_add_{}", size_str), |b| {
            b.iter(|| {
                let a = create_test_tensor(&shape).unwrap();
                let b = create_test_tensor(&shape).unwrap();
                
                let start = Instant::now();
                // TODO: Replace with actual GPU operations when available
                let _result = add(&a, &b);
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

criterion_group!(benches, 
    benchmark_quantization,
    benchmark_bitlinear, 
    benchmark_matmul,
    benchmark_elementwise
);
criterion_main!(benches);