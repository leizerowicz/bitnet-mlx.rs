# BitNet CUDA Backend

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../LICENSE)
[![Tests](https://img.shields.io/badge/tests-all_passing-brightgreen.svg)](../README.md#project-status)
[![Foundation](https://img.shields.io/badge/foundation_ready-inference_phase-brightgreen.svg)](../README.md#current-status)

High-performance CUDA implementation of BitNet operations with Microsoft W2A8 GEMV kernel parity. **Production-ready CUDA acceleration with complete Microsoft parity achieved.**

## Features

- **Microsoft W2A8 GEMV Kernels**: 2-bit weights × 8-bit activations matrix-vector multiplication
- **Performance Target**: 1.27x-3.63x speedups over BF16 baseline on A100 GPU
- **dp4a Instruction Optimization**: Hardware-accelerated 4-element dot product utilization
- **Memory Coalescing**: 16×32 block optimization strategy for optimal GPU memory access
- **Vectorized Operations**: Maximum throughput through optimized memory access patterns

## Architecture

### Core Components

1. **W2A8 GEMV Kernel** (`src/kernels/w2a8_gemv.cu`)
   - 2-bit weight extraction with fast bit manipulation
   - dp4a instruction optimization for compute capability 7.5+
   - 16×32 weight permutation for memory coalescing
   - Warp-level reduction for optimal performance

2. **Memory Management** (`src/memory.rs`)
   - Efficient GPU memory allocation and pooling
   - Host-device memory transfer optimization
   - Memory usage tracking and profiling

3. **Stream Management** (`src/stream.rs`)
   - Asynchronous operation support
   - Multi-stream execution for overlapped computation
   - Stream synchronization and dependency management

4. **Backend Integration** (`src/backend.rs`)
   - High-level API for BitNet CUDA operations
   - Performance monitoring and optimization
   - Device capability validation

## Requirements

### System Requirements
- CUDA Toolkit 11.0+ or 12.0+
- NVIDIA GPU with compute capability 7.5+ (RTX 20xx series or newer)
- Linux or Windows (macOS not supported for CUDA)

### Recommended Hardware
- NVIDIA A100, RTX 30xx/40xx series, or H100 for optimal performance
- Minimum 8GB GPU memory for large models
- PCIe 4.0 for maximum bandwidth

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bitnet-cuda = { version = "1.0.0", features = ["cuda"] }
```

### Environment Setup

1. Install CUDA Toolkit:
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.0-545.23.06-1_amd64.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. Set environment variables:
   ```bash
   export CUDA_PATH=/usr/local/cuda
   export PATH=$CUDA_PATH/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   ```

## Usage

### Basic Example

```rust
use bitnet_cuda::{CudaBackend, CudaBackendConfig, W2A8GemvConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check CUDA availability
    if !bitnet_cuda::is_cuda_available() {
        println!("CUDA not available");
        return Ok(());
    }

    // Initialize CUDA backend
    let config = CudaBackendConfig::default();
    let backend = CudaBackend::new(config)?;

    // Prepare data
    let m = 1024; // Output features
    let k = 2048; // Input features
    
    let weights = vec![0x55u8; (m * k + 3) / 4]; // 2-bit weights, packed
    let activations = vec![1i8; k]; // 8-bit activations
    
    // Copy to GPU
    let gpu_weights = backend.memory_manager().copy_to_device(&weights)?;
    let gpu_activations = backend.memory_manager().copy_to_device(&activations)?;
    let mut gpu_output = backend.memory_manager().allocate::<i32>(m)?;
    
    // Execute W2A8 GEMV
    backend.w2a8_gemv(
        gpu_weights.slice(),
        gpu_activations.slice(), 
        gpu_output.slice_mut(),
        m, k, 0 // stream index
    )?;
    
    // Synchronize and get results
    backend.synchronize()?;
    let results = backend.memory_manager().copy_to_host(&gpu_output)?;
    
    println!("Computation completed successfully!");
    Ok(())
}
```

### Advanced Configuration

```rust
use bitnet_cuda::{
    CudaBackendConfig, W2A8GemvConfig, 
    GridStrategy, CoalescingLevel
};

let config = CudaBackendConfig {
    device_id: 0,
    memory_pool_size: 1024 * 1024 * 1024, // 1GB
    num_streams: 4,
    enable_profiling: true,
    w2a8_config: W2A8GemvConfig {
        block_size: 256,
        grid_strategy: GridStrategy::Occupancy,
        enable_weight_permutation: true,
        enable_dp4a: true,
        coalescing_level: CoalescingLevel::Aggressive,
    },
};

let backend = CudaBackend::new(config)?;
```

## Performance

### Benchmark Results

Target performance on NVIDIA A100:

| Problem Size | Execution Time | Speedup vs BF16 | Bandwidth |
|--------------|----------------|------------------|-----------|
| 1K×4K        | ~50 μs         | 2.1×             | 400 GB/s  |
| 4K×16K       | ~200 μs        | 2.8×             | 600 GB/s  |
| 8K×32K       | ~800 μs        | 3.2×             | 700 GB/s  |

### Optimization Features

1. **dp4a Instructions**: Hardware-accelerated 4-element dot products
2. **Memory Coalescing**: Optimized access patterns for GPU memory hierarchy
3. **Weight Permutation**: 16×32 block layout for optimal throughput
4. **Vectorized Loading**: Efficient data movement and computation overlap

## Integration with BitNet-Rust

This CUDA backend integrates seamlessly with the main BitNet-Rust ecosystem:

```rust
use bitnet_core::tensor::BitNetTensor;
use bitnet_cuda::CudaBackend;

// Create tensor on GPU
let tensor = BitNetTensor::zeros([1024, 2048])?
    .to_device(Device::Cuda(0))?;

// Execute CUDA-accelerated operations
let result = tensor.bitlinear_forward(&weights, &bias)?;
```

## Development

### Building

```bash
# Build with CUDA support
cargo build --features cuda

# Run tests (requires CUDA GPU)
cargo test --features cuda

# Run benchmarks
cargo bench --features cuda
```

### Testing

```bash
# Unit tests
cargo test --features cuda test_w2a8_gemv

# Integration tests with hardware
cargo test --features cuda test_cuda_integration

# Performance tests
cargo test --features cuda --release test_performance
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA toolkit is installed and `CUDA_PATH` is set
2. **Compute capability**: Requires NVIDIA GPU with compute capability 7.5+
3. **Memory errors**: Reduce batch size or increase GPU memory

### Performance Tuning

1. **Block size**: Tune based on GPU architecture (256-1024)
2. **Grid strategy**: Use occupancy-based for balanced workloads
3. **Memory pool**: Size according to available GPU memory
4. **Stream count**: 2-4 streams typically optimal

## Contributing

See the main [BitNet-Rust contributing guide](../CONTRIBUTING.md).

## License

MIT OR Apache-2.0
