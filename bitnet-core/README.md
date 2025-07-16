# BitNet Core

[![Crates.io](https://img.shields.io/crates/v/bitnet-core.svg)](https://crates.io/crates/bitnet-core)
[![Documentation](https://docs.rs/bitnet-core/badge.svg)](https://docs.rs/bitnet-core)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

The core foundation library for BitNet neural networks, providing sophisticated memory management, device abstraction, and tensor infrastructure optimized for Apple Silicon and high-performance computing.

## 🎯 Purpose

`bitnet-core` serves as the foundational layer for the BitNet ecosystem, focusing on:

- **Advanced Memory Management**: Production-ready hybrid memory pool system
- **Device Abstraction**: Unified interface for CPU, Metal GPU, and future accelerators  
- **Tensor Infrastructure**: Basic tensor operations and metadata management
- **Performance Optimization**: Zero-copy operations and SIMD-friendly data structures

## ✅ What's Implemented

### 🟢 **Memory Management System** (Production Ready)

#### Hybrid Memory Pool Architecture
- **SmallBlockPool**: Fixed-size allocation for blocks < 1MB with O(1) operations
- **LargeBlockPool**: Buddy allocation algorithm for blocks ≥ 1MB with coalescing
- **DeviceSpecificPools**: Separate memory pools for CPU and Metal GPU memory
- **Thread Safety**: Fine-grained locking with minimal contention

#### Advanced Memory Tracking
- **Real-time Metrics**: Allocation patterns, peak usage, fragmentation analysis
- **Memory Pressure Detection**: Automatic detection of memory pressure with callbacks
- **Leak Detection**: Comprehensive tracking of unreleased allocations
- **Performance Profiling**: Timeline analysis and allocation pattern recognition

#### Automatic Cleanup System
- **Intelligent Compaction**: Automatic memory defragmentation
- **Configurable Strategies**: Idle, pressure-based, and periodic cleanup
- **Device-Specific Cleanup**: Optimized cleanup for different device types
- **Safety Validation**: Prevents corruption of active tensors

### 🟢 **Device Abstraction Layer** (Production Ready)

#### Device Management
- **Automatic Device Selection**: Intelligent selection of optimal compute device
- **Device Capabilities**: Runtime detection of device features and limitations
- **Memory Bandwidth Detection**: Automatic detection of memory bandwidth characteristics
- **Cross-Platform Support**: Unified API across different hardware platforms

#### Device-Specific Optimizations
- **CPU Optimizations**: Cache-friendly memory layouts and SIMD alignment
- **Metal GPU Support**: Optimized memory management for Apple Silicon GPUs
- **Future Extensibility**: Architecture ready for CUDA and other accelerators

### 🟡 **Tensor Infrastructure** (Basic Implementation)

#### Tensor Metadata System
- **BitNetDType**: Custom data types optimized for quantized operations
- **TensorMetadata**: Comprehensive tensor shape, stride, and device information
- **TensorHandle**: Safe reference counting and lifetime management
- **Memory Layout**: Optimized memory layouts for different tensor operations

#### Basic Tensor Operations
- **Tensor Creation**: Basic tensor allocation and initialization
- **Memory Management**: Integration with the hybrid memory pool system
- **Device Placement**: Automatic tensor placement on appropriate devices
- **Metadata Tracking**: Comprehensive tracking of tensor properties

## 🔴 What Needs Implementation

### High Priority

1. **Advanced Tensor Operations**
   - Matrix multiplication optimizations
   - Element-wise operations (add, mul, etc.)
   - Reduction operations (sum, mean, max, etc.)
   - Broadcasting and reshaping operations

2. **SIMD Optimizations**
   - AVX2/AVX-512 implementations for x86_64
   - NEON optimizations for ARM64
   - Auto-vectorization hints and intrinsics

3. **Memory Layout Optimizations**
   - Strided tensor support
   - Memory-efficient tensor views
   - Zero-copy tensor slicing

### Medium Priority

1. **Advanced Device Features**
   - Multi-GPU support and load balancing
   - Device-to-device memory transfers
   - Asynchronous operations and streams

2. **Performance Monitoring**
   - Detailed performance counters
   - Operation-level profiling
   - Memory bandwidth utilization tracking

3. **Error Handling**
   - Comprehensive error recovery
   - Graceful degradation on memory pressure
   - Device failure handling

### Low Priority

1. **Serialization Support**
   - Tensor serialization/deserialization
   - Memory pool state persistence
   - Cross-platform compatibility

2. **Advanced Memory Features**
   - Memory-mapped file support
   - Shared memory between processes
   - Memory compression for inactive tensors

## 🚀 Quick Start

### Basic Memory Pool Usage

```rust
use bitnet_core::memory::{HybridMemoryPool, MemoryPoolConfig};
use bitnet_core::device::auto_select_device;

// Create memory pool with default configuration
let pool = HybridMemoryPool::new()?;
let device = auto_select_device();

// Allocate 1MB of memory with 64-byte alignment
let handle = pool.allocate(1024 * 1024, 64, &device)?;

// Get memory metrics
let metrics = pool.get_metrics();
println!("Total allocated: {} bytes", metrics.total_allocated);
println!("Peak usage: {} bytes", metrics.peak_allocated);

// Deallocate memory
pool.deallocate(handle)?;
```

### Advanced Memory Tracking

```rust
use bitnet_core::memory::{
    MemoryPoolConfig, TrackingConfig, TrackingLevel,
    MemoryPressureLevel
};

// Configure advanced tracking
let mut config = MemoryPoolConfig::default();
config.enable_advanced_tracking = true;
config.tracking_config = Some(TrackingConfig {
    level: TrackingLevel::Detailed,
    enable_pressure_detection: true,
    enable_leak_detection: true,
    ..Default::default()
});

let pool = HybridMemoryPool::with_config(config)?;

// Register pressure callback
pool.register_pressure_callback(Box::new(|level| {
    match level {
        MemoryPressureLevel::Critical => {
            eprintln!("CRITICAL: Memory pressure detected!");
        },
        MemoryPressureLevel::High => {
            println!("HIGH: Memory pressure detected");
        },
        _ => {}
    }
}));

// Get detailed metrics
if let Some(detailed) = pool.get_detailed_metrics() {
    println!("Pressure level: {:?}", detailed.pressure_level);
    println!("Fragmentation: {:.2}%", detailed.fragmentation_ratio * 100.0);
}
```

### Device Abstraction

```rust
use bitnet_core::device::{auto_select_device, DeviceCapabilities};

// Automatic device selection
let device = auto_select_device();
println!("Selected device: {:?}", device);

// Check device capabilities
let caps = DeviceCapabilities::for_device(&device);
println!("Supports Metal: {}", caps.supports_metal);
println!("Memory bandwidth: {} GB/s", caps.memory_bandwidth_gbps);
```

### Basic Tensor Operations

```rust
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType, TensorMetadata};
use bitnet_core::device::auto_select_device;

let device = auto_select_device();
let pool = HybridMemoryPool::new()?;

// Create tensor metadata
let metadata = TensorMetadata::new(
    vec![128, 256],  // shape
    BitNetDType::F32,
    device.clone()
);

// Create tensor
let tensor = BitNetTensor::new(metadata, &pool)?;
println!("Tensor shape: {:?}", tensor.shape());
println!("Tensor device: {:?}", tensor.device());
```

## 📊 Performance Characteristics

### Memory Pool Performance (Apple M1 Pro)

| Operation | Small Blocks (<1MB) | Large Blocks (≥1MB) |
|-----------|-------------------|-------------------|
| **Allocation** | ~50 ns | ~200 ns |
| **Deallocation** | ~30 ns | ~150 ns |
| **Throughput** | 20M ops/sec | 5M ops/sec |
| **Memory Overhead** | <2% | <1% |

### Memory Tracking Overhead

| Tracking Level | CPU Overhead | Memory Overhead |
|---------------|--------------|-----------------|
| **None** | 0% | 0% |
| **Basic** | <1% | <0.1% |
| **Standard** | ~2% | ~0.5% |
| **Detailed** | ~5% | ~1% |

## 🏗️ Architecture

### Memory Management Architecture

```
HybridMemoryPool
├── SmallBlockPool (< 1MB allocations)
│   ├── Fixed-size block allocation
│   ├── Fast O(1) allocation/deallocation
│   └── Minimal fragmentation
├── LargeBlockPool (≥ 1MB allocations)
│   ├── Buddy allocation algorithm
│   ├── Efficient large block handling
│   └── Memory coalescing
├── DeviceSpecificPools
│   ├── CPU memory pools
│   ├── Metal GPU memory pools
│   └── Future: CUDA memory pools
└── AdvancedTracking
    ├── Memory pressure detection
    ├── Allocation pattern analysis
    ├── Leak detection and reporting
    └── Performance profiling
```

### Module Structure

```
bitnet-core/src/
├── device/                 # Device abstraction layer
│   └── mod.rs             # Device selection and capabilities
├── memory/                # Memory management system
│   ├── mod.rs            # Main memory pool interface
│   ├── small_block.rs    # Small block allocator
│   ├── large_block.rs    # Large block allocator
│   ├── device_pool.rs    # Device-specific pools
│   ├── handle.rs         # Memory handle management
│   ├── metrics.rs        # Memory metrics and monitoring
│   ├── tracking/         # Advanced memory tracking
│   │   ├── mod.rs       # Tracking system interface
│   │   ├── tracker.rs   # Main tracking implementation
│   │   ├── patterns.rs  # Allocation pattern analysis
│   │   ├── pressure.rs  # Memory pressure detection
│   │   ├── timeline.rs  # Timeline analysis
│   │   ├── profiler.rs  # Performance profiling
│   │   └── config.rs    # Tracking configuration
│   ├── cleanup/          # Automatic cleanup system
│   │   ├── mod.rs       # Cleanup system interface
│   │   ├── manager.rs   # Cleanup manager
│   │   ├── scheduler.rs # Cleanup scheduling
│   │   ├── strategies.rs # Cleanup strategies
│   │   ├── metrics.rs   # Cleanup metrics
│   │   ├── config.rs    # Cleanup configuration
│   │   └── device_cleanup.rs # Device-specific cleanup
│   └── tensor/           # Tensor memory management
│       ├── mod.rs       # Tensor system interface
│       ├── tensor.rs    # Tensor implementation
│       ├── handle.rs    # Tensor handle management
│       ├── metadata.rs  # Tensor metadata
│       └── dtype.rs     # BitNet data types
├── tensor/               # Basic tensor operations
│   └── mod.rs           # Tensor operation interface
└── lib.rs               # Library root and re-exports
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test --package bitnet-core

# Run specific test modules
cargo test --package bitnet-core memory
cargo test --package bitnet-core device
cargo test --package bitnet-core tensor

# Run with detailed output
cargo test --package bitnet-core -- --nocapture

# Run integration tests
cargo test --package bitnet-core --test integration_test
```

## 📈 Benchmarks

Run performance benchmarks:

```bash
# Run all benchmarks
cargo bench --package bitnet-benchmarks

# Run memory-specific benchmarks
cargo bench --package bitnet-benchmarks -- memory

# Generate benchmark reports
cargo bench --package bitnet-benchmarks -- --output-format html
```

## 🔧 Configuration

### Memory Pool Configuration

```rust
use bitnet_core::memory::{MemoryPoolConfig, TrackingConfig, CleanupConfig};

let config = MemoryPoolConfig {
    // Pool sizing
    initial_small_pool_size: 64 * 1024 * 1024,  // 64MB
    max_small_pool_size: 512 * 1024 * 1024,     // 512MB
    initial_large_pool_size: 128 * 1024 * 1024, // 128MB
    max_large_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
    
    // Tracking configuration
    enable_advanced_tracking: true,
    tracking_config: Some(TrackingConfig {
        level: TrackingLevel::Standard,
        enable_pressure_detection: true,
        enable_leak_detection: true,
        pressure_threshold_ratio: 0.8,
        leak_detection_interval: Duration::from_secs(60),
    }),
    
    // Cleanup configuration
    enable_automatic_cleanup: true,
    cleanup_config: Some(CleanupConfig {
        idle_cleanup_interval: Duration::from_secs(30),
        pressure_cleanup_threshold: 0.9,
        enable_compaction: true,
        max_cleanup_time: Duration::from_millis(100),
    }),
};

let pool = HybridMemoryPool::with_config(config)?;
```

## 🤝 Contributing

Contributions are welcome! Priority areas for `bitnet-core`:

1. **Tensor Operations**: Implement missing tensor operations
2. **SIMD Optimizations**: Add platform-specific optimizations
3. **Device Support**: Extend device abstraction for new hardware
4. **Performance**: Optimize critical paths and reduce overhead

See the [main project README](../README.md) for contribution guidelines.

## 📄 License

Licensed under the MIT License. See [LICENSE](../LICENSE) for details.