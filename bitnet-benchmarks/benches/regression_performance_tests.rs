//! Performance Regression Testing Benchmarks
//! 
//! This benchmark suite provides automated performance regression detection
//! and continuous performance monitoring for BitNet operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use candle_core::{Tensor, Device};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[cfg(feature = "mlx")]
use bitnet_core::mlx::{
    MlxTensor, BitNetMlxDevice, operations::BitNetMlxOps,
};

/// Performance baseline for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceBaseline {
    operation: String,
    device: String,
    tensor_size: (usize, usize),
    baseline_time: Duration,
    baseline_throughput: f64,
    baseline_memory: usize,
    tolerance_percent: f64,
    timestamp: std::time::SystemTime,
}

/// Performance regression result
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegressionResult {
    operation: String,
    device: String,
    current_time: Duration,
    baseline_time: Duration,
    performance_change: f64, // Percentage change
    is_regression: bool,
    severity: RegressionSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RegressionSeverity {
    None,
    Minor,      // 5-15% degradation
    Moderate,   // 15-30% degradation
    Major,      // 30-50% degradation
    Critical,   // >50% degradation
}

/// Regression detector utility
struct RegressionDetector {
    baselines: HashMap<String, PerformanceBaseline>,
    tolerance_percent: f64,
}

impl RegressionDetector {
    fn new(tolerance_percent: f64) -> Self {
        Self {
            baselines: HashMap::new(),
            tolerance_percent,
        }
    }

    fn add_baseline(&mut self, baseline: PerformanceBaseline) {
        let key = format!("{}_{}_{}x{}", 
            baseline.operation, baseline.device, 
            baseline.tensor_size.0, baseline.tensor_size.1);
        self.baselines.insert(key, baseline);
    }

    fn check_regression(&self, operation: &str, device: &str, tensor_size: (usize, usize), 
                       current_time: Duration) -> Option<RegressionResult> {
        let key = format!("{}_{}_{}x{}", operation, device, tensor_size.0, tensor_size.1);
        
        if let Some(baseline) = self.baselines.get(&key) {
            let performance_change = ((current_time.as_secs_f64() - baseline.baseline_time.as_secs_f64()) 
                / baseline.baseline_time.as_secs_f64()) * 100.0;
            
            let is_regression = performance_change > self.tolerance_percent;
            let severity = if performance_change > 50.0 {
                RegressionSeverity::Critical
            } else if performance_change > 30.0 {
                RegressionSeverity::Major
            } else if performance_change > 15.0 {
                RegressionSeverity::Moderate
            } else if performance_change > 5.0 {
                RegressionSeverity::Minor
            } else {
                RegressionSeverity::None
            };

            Some(RegressionResult {
                operation: operation.to_string(),
                device: device.to_string(),
                current_time,
                baseline_time: baseline.baseline_time,
                performance_change,
                is_regression,
                severity,
            })
        } else {
            None
        }
    }
}

/// Core operation regression tests
fn bench_core_operations_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_operations_regression");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    // Define baseline expectations (these would typically be loaded from a file)
    let mut detector = RegressionDetector::new(10.0); // 10% tolerance
    
    // Add some example baselines (in practice, these would be historical data)
    detector.add_baseline(PerformanceBaseline {
        operation: "matmul".to_string(),
        device: "cpu".to_string(),
        tensor_size: (512, 512),
        baseline_time: Duration::from_millis(50), // Expected baseline
        baseline_throughput: 20.0,
        baseline_memory: 1024 * 1024,
        tolerance_percent: 10.0,
        timestamp: std::time::SystemTime::now(),
    });

    let test_operations = vec![
        ("matmul", vec![(256, 256), (512, 512), (1024, 1024)]),
        ("quantization", vec![(512, 512), (1024, 1024)]),
        ("bitlinear", vec![(768, 3072), (1024, 4096)]),
    ];

    for (operation, sizes) in test_operations {
        for &(rows, cols) in &sizes {
            let elements = rows * cols;
            group.throughput(Throughput::Elements(elements as u64));

            // CPU regression test
            group.bench_with_input(
                BenchmarkId::new(format!("cpu_{operation}_regression"), format!("{rows}x{cols}")),
                &(rows, cols),
                |bencher, &(rows, cols)| {
                    let device = Device::Cpu;
                    
                    bencher.iter_custom(|iters| {
                        let mut total_time = Duration::ZERO;
                        
                        for _ in 0..iters {
                            let start = Instant::now();
                            
                            match operation {
                                "matmul" => {
                                    let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                    let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                                    let result = a.matmul(&b).unwrap();
                                    black_box(result);
                                },
                                "quantization" => {
                                    let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                    let scale = 0.1f32;
                                    let scale_tensor = Tensor::new(scale, &device).unwrap();
                                    let scaled = tensor.broadcast_div(&scale_tensor).unwrap();
                                    let quantized = scaled.clamp(-1.0, 1.0).unwrap().round().unwrap();
                                    black_box(quantized);
                                },
                                "bitlinear" => {
                                    let batch_size = 32;
                                    let input = Tensor::randn(0f32, 1f32, (batch_size, rows), &device).unwrap();
                                    let weight = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                    let quantized_weight = weight.clamp(-1.0, 1.0).unwrap().round().unwrap();
                                    let result = input.matmul(&quantized_weight).unwrap();
                                    black_box(result);
                                },
                                _ => {}
                            }
                            
                            total_time += start.elapsed();
                        }
                        
                        // Check for regression
                        let avg_time = total_time / iters as u32;
                        if let Some(regression) = detector.check_regression(operation, "cpu", (rows, cols), avg_time) {
                            if regression.is_regression {
                                eprintln!("⚠️  Performance regression detected in {} on CPU: {:.1}% slower", 
                                    operation, regression.performance_change);
                            }
                        }
                        
                        total_time
                    })
                },
            );

            // MLX regression test
            #[cfg(feature = "mlx")]
            {
                group.bench_with_input(
                    BenchmarkId::new(format!("mlx_{}_regression", operation), format!("{}x{}", rows, cols)),
                    &(rows, cols),
                    |bencher, &(rows, cols)| {
                        use bitnet_core::memory::tensor::BitNetDType;
                        let device = BitNetMlxDevice::default();
                        
                        bencher.iter_custom(|iters| {
                            let mut total_time = Duration::ZERO;
                            
                            for _ in 0..iters {
                                let start = Instant::now();
                                
                                match operation {
                                    "matmul" => {
                                        let a = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                                        let b = MlxTensor::randn(&[cols, rows], BitNetDType::F32, device.clone()).unwrap();
                                        let result = BitNetMlxOps::matmul(&a, &b).unwrap();
                                        black_box(result);
                                    },
                                    "quantization" => {
                                        let tensor = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                                        let result = BitNetMlxOps::quantize_1_58_bit(&tensor, Some(0.1)).unwrap();
                                        black_box(result);
                                    },
                                    "bitlinear" => {
                                        let batch_size = 32;
                                        let input = MlxTensor::randn(&[batch_size, rows], BitNetDType::F32, device.clone()).unwrap();
                                        let weight = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                                        let bias = MlxTensor::randn(&[cols], BitNetDType::F32, device.clone()).unwrap();
                                        let result = BitNetMlxOps::bitlinear_forward(&input, &weight, Some(&bias), true).unwrap();
                                        black_box(result);
                                    },
                                    _ => {}
                                }
                                
                                total_time += start.elapsed();
                            }
                            
                            // Check for regression
                            let avg_time = total_time / iters as u32;
                            if let Some(regression) = detector.check_regression(operation, "mlx", (rows, cols), avg_time) {
                                if regression.is_regression {
                                    eprintln!("⚠️  Performance regression detected in {} on MLX: {:.1}% slower", 
                                        operation, regression.performance_change);
                                }
                            }
                            
                            total_time
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Memory usage regression tests
fn bench_memory_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_regression");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(6));

    let memory_scenarios = vec![
        ("small_tensors", vec![(128, 128); 50]),
        ("medium_tensors", vec![(512, 512); 10]),
        ("large_tensor", vec![(2048, 2048); 1]),
    ];

    for (scenario_name, tensor_sizes) in memory_scenarios {
        // CPU memory regression test
        group.bench_function(
            format!("cpu_memory_{scenario_name}"),
            |bencher| {
                bencher.iter(|| {
                    let device = Device::Cpu;
                    let mut tensors = Vec::new();
                    
                    // Allocate tensors
                    for &(rows, cols) in &tensor_sizes {
                        let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                        tensors.push(tensor);
                    }
                    
                    // Perform operations
                    for i in 0..tensors.len().min(5) {
                        for j in (i + 1)..tensors.len().min(5) {
                            if tensors[i].shape() == tensors[j].shape() {
                                let result = (&tensors[i] + &tensors[j]).unwrap();
                                black_box(result);
                            }
                        }
                    }
                    
                    black_box(tensors)
                })
            },
        );

        // MLX memory regression test
        #[cfg(feature = "mlx")]
        {
            group.bench_function(
                &format!("mlx_memory_{}", scenario_name),
                |bencher| {
                    bencher.iter(|| {
                        use bitnet_core::memory::tensor::BitNetDType;
                        let device = BitNetMlxDevice::default();
                        let mut tensors = Vec::new();
                        
                        // Allocate tensors
                        for &(rows, cols) in &tensor_sizes {
                            let tensor = MlxTensor::randn(&[rows, cols], BitNetDType::F32, device.clone()).unwrap();
                            tensors.push(tensor);
                        }
                        
                        // Perform operations
                        for i in 0..tensors.len().min(5) {
                            for j in (i + 1)..tensors.len().min(5) {
                                if tensors[i].shape() == tensors[j].shape() {
                                    let result = BitNetMlxOps::add(&tensors[i], &tensors[j]).unwrap();
                                    black_box(result);
                                }
                            }
                        }
                        
                        black_box(tensors)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Throughput regression tests
fn bench_throughput_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_regression");
    
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = vec![1, 8, 16, 32, 64];
    let tensor_size = (1024, 1024);

    for &batch_size in &batch_sizes {
        let elements = batch_size * tensor_size.0 * tensor_size.1;
        group.throughput(Throughput::Elements(elements as u64));

        // CPU throughput regression
        group.bench_with_input(
            BenchmarkId::new("cpu_throughput_regression", batch_size),
            &batch_size,
            |bencher, &batch_size| {
                let device = Device::Cpu;
                
                bencher.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    
                    for _ in 0..iters {
                        let start = Instant::now();
                        
                        // Simulate batch processing
                        for _ in 0..batch_size {
                            let a = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
                            let b = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
                            let result = a.matmul(&b).unwrap();
                            black_box(result);
                        }
                        
                        total_time += start.elapsed();
                    }
                    
                    // Calculate and check throughput
                    let avg_time = total_time / iters as u32;
                    let throughput = batch_size as f64 / avg_time.as_secs_f64();
                    
                    // Expected throughput baselines (these would be loaded from historical data)
                    let expected_throughput = match batch_size {
                        1 => 1.0,
                        8 => 6.0,
                        16 => 10.0,
                        32 => 15.0,
                        64 => 20.0,
                        _ => 1.0,
                    };
                    
                    let throughput_change = ((throughput - expected_throughput) / expected_throughput) * 100.0;
                    if throughput_change < -10.0 {
                        eprintln!("⚠️  Throughput regression detected for batch size {}: {:.1}% decrease", 
                            batch_size, -throughput_change);
                    }
                    
                    total_time
                })
            },
        );
    }

    group.finish();
}

/// Latency regression tests
fn bench_latency_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_regression");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    let latency_critical_ops = vec![
        ("single_matmul", (512, 512)),
        ("small_quantization", (256, 256)),
        ("inference_step", (1024, 1024)),
    ];

    for (op_name, tensor_size) in latency_critical_ops {
        let elements = tensor_size.0 * tensor_size.1;
        group.throughput(Throughput::Elements(elements as u64));

        // CPU latency regression
        group.bench_with_input(
            BenchmarkId::new("cpu_latency_regression", op_name),
            &tensor_size,
            |bencher, &(rows, cols)| {
                let device = Device::Cpu;
                
                bencher.iter_custom(|iters| {
                    let mut latencies = Vec::new();
                    
                    for _ in 0..iters {
                        let start = Instant::now();
                        
                        match op_name {
                            "single_matmul" => {
                                let a = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                let b = Tensor::randn(0f32, 1f32, (cols, rows), &device).unwrap();
                                let result = a.matmul(&b).unwrap();
                                black_box(result);
                            },
                            "small_quantization" => {
                                let tensor = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                let quantized = tensor.clamp(-1.0, 1.0).unwrap().round().unwrap();
                                black_box(quantized);
                            },
                            "inference_step" => {
                                let input = Tensor::randn(0f32, 1f32, (1, rows), &device).unwrap();
                                let weight = Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap();
                                let quantized_weight = weight.clamp(-1.0, 1.0).unwrap().round().unwrap();
                                let output = input.matmul(&quantized_weight).unwrap();
                                let activated = output.relu().unwrap();
                                black_box(activated);
                            },
                            _ => {}
                        }
                        
                        latencies.push(start.elapsed());
                    }
                    
                    // Analyze latency distribution
                    latencies.sort();
                    let p95_latency = latencies[latencies.len() * 95 / 100];
                    let p99_latency = latencies[latencies.len() * 99 / 100];
                    
                    // Expected latency baselines (would be loaded from historical data)
                    let expected_p95 = Duration::from_millis(match op_name {
                        "single_matmul" => 10,
                        "small_quantization" => 5,
                        "inference_step" => 15,
                        _ => 10,
                    });
                    
                    if p95_latency > expected_p95 * 110 / 100 { // 10% tolerance
                        eprintln!("⚠️  Latency regression detected in {}: P95 latency {:.2}ms vs expected {:.2}ms", 
                            op_name, p95_latency.as_millis(), expected_p95.as_millis());
                    }
                    
                    latencies.iter().sum()
                })
            },
        );
    }

    group.finish();
}

/// Stability regression tests (variance in performance)
fn bench_stability_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_regression");
    
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(12));

    // Test performance stability over multiple runs
    group.bench_function("performance_stability", |bencher| {
        let device = Device::Cpu;
        let tensor_size = (1024, 1024);
        
        bencher.iter_custom(|iters| {
            let mut execution_times = Vec::new();
            
            for _ in 0..iters {
                let start = Instant::now();
                
                // Perform a standard operation
                let a = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
                let b = Tensor::randn(0f32, 1f32, tensor_size, &device).unwrap();
                let result = a.matmul(&b).unwrap();
                black_box(result);
                
                execution_times.push(start.elapsed());
            }
            
            // Calculate coefficient of variation (CV)
            if execution_times.len() > 1 {
                let mean_time = execution_times.iter().sum::<Duration>().as_secs_f64() / execution_times.len() as f64;
                let variance = execution_times.iter()
                    .map(|t| (t.as_secs_f64() - mean_time).powi(2))
                    .sum::<f64>() / execution_times.len() as f64;
                let std_dev = variance.sqrt();
                let cv = std_dev / mean_time;
                
                // Check for stability regression (CV should be low)
                if cv > 0.15 { // 15% coefficient of variation threshold
                    eprintln!("⚠️  Performance stability regression detected: CV = {:.1}%", cv * 100.0);
                }
            }
            
            execution_times.iter().sum()
        })
    });

    group.finish();
}

criterion_group!(
    regression_benches,
    bench_core_operations_regression,
    bench_memory_regression,
    bench_throughput_regression,
    bench_latency_regression,
    bench_stability_regression
);

criterion_main!(regression_benches);