use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_core::quantization::*;
use candle_core::{Tensor, Device};

fn bench_quantization(c: &mut Criterion) {
    let device = Device::Cpu;
    let sizes = vec![128, 512, 1024, 4096];
    
    let mut group = c.benchmark_group("quantization");
    
    for size in sizes {
        let tensor = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("quantize_weights", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let quantized = quantize_weights(black_box(tensor)).unwrap();
                    black_box(quantized)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("dequantize_weights", size),
            &tensor,
            |b, tensor| {
                let quantized = quantize_weights(tensor).unwrap();
                b.iter(|| {
                    let dequantized = dequantize_weights(black_box(&quantized)).unwrap();
                    black_box(dequantized)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_quantization);
criterion_main!(benches);