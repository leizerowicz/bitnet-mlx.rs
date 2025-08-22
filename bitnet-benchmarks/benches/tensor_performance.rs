//! Comprehensive Tensor Benchmarks
//!
//! Performance benchmarks for BitNet tensor operations, following the established
//! benchmarking patterns in the bitnet-benchmarks crate.

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, BenchmarkId, 
    Throughput, PlotConfiguration, AxisScale
};
use std::time::Duration;
use candle_core::Device;

use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig, TrackingConfig};
use bitnet_core::device::{get_cpu_device, is_metal_available, get_metal_device};

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Benchmark configuration for tensor operations
struct TensorBenchmarkConfig {
    tensor_sizes: Vec<Vec<usize>>,
    data_types: Vec<BitNetDType>,
    batch_sizes: Vec<usize>,
    warmup_time: Duration,
    measurement_time: Duration,
}

impl Default for TensorBenchmarkConfig {
    fn default() -> Self {
        Self {
            tensor_sizes: vec![
                vec![32, 32],
                vec![64, 64],
                vec![128, 128],
                vec![256, 256],
                vec![512, 512],
                vec![1024, 1024],
                vec![2048, 2048],
            ],
            data_types: vec![
                BitNetDType::F32,
                BitNetDType::F16,
                BitNetDType::I8,
                BitNetDType::BitNet158,
            ],
            batch_sizes: vec![1, 8, 16, 32, 64, 128],
            warmup_time: Duration::from_secs(2),
            measurement_time: Duration::from_secs(10),
        }
    }
}

/// Helper function to create benchmark memory pool
fn create_benchmark_pool() -> HybridMemoryPool {
    let mut config = MemoryPoolConfig::default();
    config.enable_advanced_tracking = true;
    config.tracking_config = Some(TrackingConfig::production());
    
    HybridMemoryPool::with_config(config)
        .expect("Failed to create benchmark memory pool")
}

// =============================================================================
// Core Tensor Creation Benchmarks
// =============================================================================

/// Benchmark tensor creation performance
fn bench_tensor_creation(c: &mut Criterion) {
    let config = TensorBenchmarkConfig::default();
    let mut group = c.benchmark_group("tensor_creation");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    let devices = vec![get_cpu_device()];
    
    for device in &devices {
        for &dtype in &config.data_types {
            for shape in &config.tensor_sizes {
                let elements: usize = shape.iter().product();
                group.throughput(Throughput::Elements(elements as u64));
                
                // Benchmark zeros creation
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("zeros_{:?}_{}", dtype, device_name(device)),
                        format!("{}x{}", shape[0], shape.get(1).unwrap_or(&1))
                    ),
                    &(shape.clone(), dtype, device.clone()),
                    |bencher, (shape, dtype, device)| {
                        bencher.iter(|| {
                            let tensor = BitNetTensor::zeros(
                                black_box(shape),
                                black_box(*dtype),
                                Some(black_box(device.clone()))
                            ).expect("Failed to create zeros tensor");
                            black_box(tensor)
                        });
                    },
                );
                
                // Benchmark ones creation
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("ones_{:?}_{}", dtype, device_name(device)),
                        format!("{}x{}", shape[0], shape.get(1).unwrap_or(&1))
                    ),
                    &(shape.clone(), dtype, device.clone()),
                    |bencher, (shape, dtype, device)| {
                        bencher.iter(|| {
                            let tensor = BitNetTensor::ones(
                                black_box(shape),
                                black_box(*dtype),
                                Some(black_box(device.clone()))
                            ).expect("Failed to create ones tensor");
                            black_box(tensor)
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark tensor creation from data
fn bench_tensor_from_data(c: &mut Criterion) {
    let config = TensorBenchmarkConfig::default();
    let mut group = c.benchmark_group("tensor_from_data");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);
    
    let device = get_cpu_device();
    
    for shape in &config.tensor_sizes {
        if shape.len() != 2 { continue; } // Only test 2D for simplicity
        
        let elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(elements as u64));
        
        // Prepare test data
        let f32_data: Vec<f32> = (0..elements).map(|i| i as f32).collect();
        let i32_data: Vec<i32> = (0..elements).map(|i| i as i32).collect();
        
        // Benchmark F32 data
        group.bench_with_input(
            BenchmarkId::new("f32", format!("{}x{}", shape[0], shape[1])),
            &(f32_data.clone(), shape.clone()),
            |bencher, (data, shape)| {
                bencher.iter(|| {
                    let tensor = BitNetTensor::from_vec(
                        black_box(data.clone()),
                        black_box(shape),
                        black_box(BitNetDType::F32),
                        Some(black_box(device.clone()))
                    ).expect("Failed to create tensor from f32 data");
                    black_box(tensor)
                });
            },
        );
        
        // Benchmark I32 data
        group.bench_with_input(
            BenchmarkId::new("i32", format!("{}x{}", shape[0], shape[1])),
            &(i32_data.clone(), shape.clone()),
            |bencher, (data, shape)| {
                bencher.iter(|| {
                    let tensor = BitNetTensor::from_vec(
                        black_box(data.clone()),
                        black_box(shape),
                        black_box(BitNetDType::I32),
                        Some(black_box(device.clone()))
                    ).expect("Failed to create tensor from i32 data");
                    black_box(tensor)
                });
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// Memory Management Benchmarks
// =============================================================================

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let config = TensorBenchmarkConfig::default();
    let mut group = c.benchmark_group("memory_allocation");
    
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);
    
    let pool = create_benchmark_pool();
    let device = get_cpu_device();
    
    // Benchmark single large allocation
    group.bench_function("single_large_allocation", |bencher| {
        bencher.iter(|| {
            let tensor = BitNetTensor::zeros(
                black_box(&[1024, 1024]),
                black_box(BitNetDType::F32),
                Some(black_box(device.clone()))
            ).expect("Failed to create large tensor");
            black_box(tensor)
        });
    });
    
    // Benchmark many small allocations
    group.bench_function("many_small_allocations", |bencher| {
        bencher.iter(|| {
            let tensors: Vec<_> = (0..100).map(|_| {
                BitNetTensor::zeros(
                    black_box(&[8, 8]),
                    black_box(BitNetDType::F32),
                    Some(black_box(device.clone()))
                ).expect("Failed to create small tensor")
            }).collect();
            black_box(tensors)
        });
    });
    
    // Benchmark mixed allocation sizes
    group.bench_function("mixed_allocation_sizes", |bencher| {
        let sizes = [vec![16, 16], vec![32, 32], vec![64, 64], 
            vec![128, 128], vec![256, 256]];
        
        bencher.iter(|| {
            let tensors: Vec<_> = sizes.iter().map(|shape| {
                BitNetTensor::zeros(
                    black_box(shape),
                    black_box(BitNetDType::F32),
                    Some(black_box(device.clone()))
                ).expect("Failed to create mixed size tensor")
            }).collect();
            black_box(tensors)
        });
    });
    
    group.finish();
}

/// Benchmark memory pool efficiency
fn bench_memory_pool_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_efficiency");
    
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    
    let device = get_cpu_device();
    
    // Compare with and without memory pool tracking
    group.bench_function("with_tracking", |bencher| {
        let pool = create_benchmark_pool();
        
        bencher.iter(|| {
            let tensors: Vec<_> = (0..50).map(|_| {
                BitNetTensor::zeros(
                    black_box(&[64, 64]),
                    black_box(BitNetDType::F32),
                    Some(black_box(device.clone()))
                ).expect("Failed to create tensor with tracking")
            }).collect();
            black_box(tensors)
        });
    });
    
    group.bench_function("basic_allocation", |bencher| {
        let mut config = MemoryPoolConfig::default();
        config.enable_advanced_tracking = false;
        let pool = HybridMemoryPool::with_config(config)
            .expect("Failed to create basic pool");
        
        bencher.iter(|| {
            let tensors: Vec<_> = (0..50).map(|_| {
                BitNetTensor::zeros(
                    black_box(&[64, 64]),
                    black_box(BitNetDType::F32),
                    Some(black_box(device.clone()))
                ).expect("Failed to create tensor without tracking")
            }).collect();
            black_box(tensors)
        });
    });
    
    group.finish();
}

// =============================================================================
// Device Migration Benchmarks
// =============================================================================

/// Benchmark device migration performance
fn bench_device_migration(c: &mut Criterion) {
    if !is_metal_available() {
        println!("Skipping device migration benchmarks - Metal not available");
        return;
    }
    
    let mut group = c.benchmark_group("device_migration");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    
    let cpu_device = get_cpu_device();
    let metal_device = get_metal_device().expect("Metal device should be available");
    
    let shapes = vec![
        vec![64, 64],
        vec![256, 256],
        vec![512, 512],
        vec![1024, 1024],
    ];
    
    for shape in &shapes {
        let elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(elements as u64));
        
        // Benchmark CPU to Metal migration (placeholder - method not yet implemented)
        group.bench_with_input(
            BenchmarkId::new("cpu_to_metal", format!("{}x{}", shape[0], shape[1])),
            shape,
            |bencher, shape| {
                let cpu_tensor = BitNetTensor::zeros(
                    shape,
                    BitNetDType::F32,
                    Some(cpu_device.clone())
                ).expect("Failed to create CPU tensor");
                
                bencher.iter(|| {
                    // Placeholder: would migrate to Metal when to_device is implemented
                    let _placeholder = &cpu_tensor;
                    black_box(_placeholder)
                });
            },
        );
        
        // Benchmark Metal to CPU migration (placeholder - method not yet implemented)
        group.bench_with_input(
            BenchmarkId::new("metal_to_cpu", format!("{}x{}", shape[0], shape[1])),
            shape,
            |bencher, shape| {
                let metal_tensor = BitNetTensor::zeros(
                    shape,
                    BitNetDType::F32,
                    Some(metal_device.clone())
                ).expect("Failed to create Metal tensor");
                
                bencher.iter(|| {
                    // Placeholder: would migrate to CPU when to_device is implemented
                    let _placeholder = &metal_tensor;
                    black_box(_placeholder)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark automatic device selection
fn bench_auto_device_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_device_selection");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    
    let shapes = vec![
        vec![32, 32],
        vec![128, 128],
        vec![512, 512],
        vec![2048, 2048],
    ];
    
    for shape in &shapes {
        let elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(elements as u64));
        
        group.bench_with_input(
            BenchmarkId::new("auto_selection", format!("{}x{}", shape[0], shape[1])),
            shape,
            |bencher, shape| {
                bencher.iter(|| {
                    let tensor = BitNetTensor::zeros(
                        black_box(shape),
                        black_box(BitNetDType::F32),
                        None // Auto device selection
                    ).expect("Failed to create auto-device tensor");
                    black_box(tensor)
                });
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// Shape Operations Benchmarks
// =============================================================================

/// Benchmark shape manipulation operations
fn bench_shape_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_operations");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    
    let device = get_cpu_device();
    
    // Prepare test tensors
    let tensor_1d = BitNetTensor::zeros(&[1024], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create 1D tensor");
    let tensor_2d = BitNetTensor::zeros(&[32, 32], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create 2D tensor");
    
    // Benchmark reshape operations (placeholder - methods not yet implemented)
    group.bench_function("reshape_1d_to_2d", |bencher| {
        bencher.iter(|| {
            // Placeholder: would reshape when method is implemented
            let _placeholder = &tensor_1d;
            black_box(_placeholder)
        });
    });
    
    group.bench_function("reshape_2d_to_1d", |bencher| {
        bencher.iter(|| {
            // Placeholder: would reshape when method is implemented
            let _placeholder = &tensor_2d;
            black_box(_placeholder)
        });
    });
    
    // Benchmark transpose operations (placeholder - method not yet implemented)
    group.bench_function("transpose_2d", |bencher| {
        bencher.iter(|| {
            // Placeholder: would transpose when method is implemented
            let _placeholder = &tensor_2d;
            black_box(_placeholder)
        });
    });
    
    // Benchmark squeeze operations (placeholder - method not yet implemented)
    let tensor_with_ones = BitNetTensor::zeros(&[1, 32, 1, 32, 1], BitNetDType::F32, Some(device.clone()))
        .expect("Failed to create tensor with unit dimensions");
    
    group.bench_function("squeeze", |bencher| {
        bencher.iter(|| {
            // Placeholder: would squeeze when method is implemented
            let _placeholder = &tensor_with_ones;
            black_box(_placeholder)
        });
    });
    
    group.finish();
}

// =============================================================================
// BitNet Quantization Benchmarks
// =============================================================================

/// Benchmark BitNet-specific tensor operations
fn bench_bitnet_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitnet_operations");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    
    let device = get_cpu_device();
    let shapes = vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256],
        vec![512, 512],
    ];
    
    for shape in &shapes {
        let elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(elements as u64));
        
        // Benchmark BitNet 1.58 tensor creation
        group.bench_with_input(
            BenchmarkId::new("bitnet158_creation", format!("{}x{}", shape[0], shape[1])),
            shape,
            |bencher, shape| {
                bencher.iter(|| {
                    let tensor = BitNetTensor::bitnet_158(
                        black_box(shape),
                        Some(black_box(device.clone()))
                    ).expect("Failed to create BitNet 1.58 tensor");
                    black_box(tensor)
                });
            },
        );
        
        // Benchmark regular tensor vs BitNet quantized
        group.bench_with_input(
            BenchmarkId::new("f32_vs_bitnet158", format!("{}x{}", shape[0], shape[1])),
            shape,
            |bencher, shape| {
                bencher.iter(|| {
                    let f32_tensor = BitNetTensor::zeros(
                        black_box(shape),
                        black_box(BitNetDType::F32),
                        Some(black_box(device.clone()))
                    ).expect("Failed to create F32 tensor");
                    
                    let bitnet_tensor = BitNetTensor::zeros(
                        black_box(shape),
                        black_box(BitNetDType::BitNet158),
                        Some(black_box(device.clone()))
                    ).expect("Failed to create BitNet tensor");
                    
                    black_box((f32_tensor, bitnet_tensor))
                });
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get device name for benchmark identification
fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}

// =============================================================================
// Criterion Group Registration
// =============================================================================

criterion_group!(
    tensor_benchmarks,
    bench_tensor_creation,
    bench_tensor_from_data,
    bench_memory_allocation,
    bench_memory_pool_efficiency,
    bench_device_migration,
    bench_auto_device_selection,
    bench_shape_operations,
    bench_bitnet_operations
);

criterion_main!(tensor_benchmarks);
