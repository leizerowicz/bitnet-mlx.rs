# BitNet-Rust Project Commands & Testing Guide

## Project Structure
```
bitnet-rust/
├── bitnet-core/           # Core tensor operations, memory management, MLX acceleration
├── bitnet-quant/          # Quantization algorithms and BitLinear layers
├── bitnet-inference/      # High-performance inference engine
├── bitnet-training/       # QAT training infrastructure
├── bitnet-metal/          # Metal GPU compute shaders
├── bitnet-cli/            # Command-line tools and utilities
├── bitnet-benchmarks/     # Performance testing and benchmarking
└── docs/                  # Documentation and guides
```

## Essential Build Commands

### Initial Setup
```bash
# Clone and setup
git clone https://github.com/Wavegoodvybe2929/bitnet-rust.git
cd bitnet-rust

# Build all crates (production)
cargo build --release --workspace

# Build with Apple Silicon optimization
cargo build --release --features apple-silicon

# Build with specific MLX features
cargo build --release --features "mlx,mlx-inference,mlx-training"
```

### Development Builds
```bash
# Development build (faster compilation)
cargo build --workspace

# Build specific crate
cargo build --package bitnet-core --release

# Clean build (when dependencies change)
cargo clean && cargo build --release --workspace
```

### Feature-Specific Builds
```bash
# Core features only
cargo build --package bitnet-core --release

# With MLX acceleration (Apple Silicon)
cargo build --package bitnet-core --release --features mlx

# With Metal GPU support
cargo build --package bitnet-metal --release

# Full Apple Silicon optimization
cargo build --release --features "apple-silicon,mlx,metal"
```

## Testing Commands

### Unit Testing
```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test --package bitnet-core
cargo test --package bitnet-quant
cargo test --package bitnet-training

# Run tests with features
cargo test --package bitnet-core --features mlx

# Run specific test
cargo test --package bitnet-core test_memory_pool_allocation
```

### Integration Testing
```bash
# Integration tests across crates
cargo test --workspace --tests

# MLX integration tests (Apple Silicon only)
cargo test --package bitnet-core --features mlx mlx_

# Memory management integration
cargo test --workspace memory_

# Quantization integration
cargo test --workspace quantization_
```

### Performance Testing
```bash
# Run all benchmarks
cargo bench --workspace

# Benchmark specific crate
cargo bench --package bitnet-benchmarks
cargo bench --package bitnet-core

# Memory performance benchmarks
cargo bench --package bitnet-core memory_

# MLX acceleration benchmarks
cargo bench --package bitnet-core --features mlx mlx_
```

## Demo and Example Commands

### Core Functionality Demos
```bash
# Memory tracking and performance
cargo run --example memory_tracking_demo --package bitnet-core --release

# Cleanup system efficiency
cargo run --example cleanup_system_demo --package bitnet-core --release

# Tensor lifecycle management
cargo run --example tensor_lifecycle --package bitnet-core --release

# Memory-efficient conversion
cargo run --example memory_efficient_conversion_demo --package bitnet-core --release
```

### Acceleration Demos
```bash
# MLX acceleration (Apple Silicon only)
cargo run --example mlx_acceleration_demo --package bitnet-core --release --features mlx

# MLX optimization utilities
cargo run --example mlx_optimization_demo --package bitnet-core --release --features mlx

# MLX graph optimization
cargo run --example mlx_graph_optimization_demo --package bitnet-core --release --features mlx

# SIMD optimization demo
cargo run --example simd_optimization_demo --package bitnet-core --release
```

### Quantization Demos
```bash
# QAT training demonstration
cargo run --example qat_training_demo --package bitnet-training --release

# Multi-bit quantization
cargo run --example multi_bit_quantization --package bitnet-quant --release

# BitLinear layer operations
cargo run --example bitlinear_demo --package bitnet-quant --release
```

## Performance Validation Commands

### Memory Performance
```bash
# Memory pool performance validation
cargo run --example memory_tracking_demo --package bitnet-core --release

# Expected outputs:
# - Allocation tracking: <15,000 ns average
# - Deallocation tracking: <2,000 ns average
# - Pattern detection: >60% confidence
# - Cleanup success: 100% rate
```

### Acceleration Validation
```bash
# MLX performance validation (Apple Silicon)
cargo run --example mlx_acceleration_demo --package bitnet-core --release --features mlx

# Expected outputs:
# - Matrix operations: 300K+ ops/sec
# - Memory allocation: <22µs
# - GPU acceleration: 15-40x speedup
```

### SIMD Performance
```bash
# Cross-platform SIMD validation
cargo bench --package bitnet-core simd_

# Expected results:
# - AVX512: 12.0x speedup
# - AVX2: 7.5x speedup  
# - NEON: 3.8x speedup
# - SSE4.1: 3.8x speedup
```

## Benchmarking Commands

### Comprehensive Benchmarking
```bash
# Full benchmark suite (6 categories, 38+ groups)
cargo bench --package bitnet-benchmarks

# Generate performance reports
cargo run --package bitnet-benchmarks -- compare --output results.json

# Create HTML reports
cargo run --package bitnet-benchmarks -- report --input results.json --output report.html

# Energy analysis
cargo run --package bitnet-benchmarks -- energy-analysis --duration 60s

# Regression testing
cargo run --package bitnet-benchmarks -- regression-check --baseline baseline.json
```

### Targeted Benchmarks
```bash
# Memory management benchmarks
cargo bench --package bitnet-core memory_

# Tensor operation benchmarks
cargo bench --package bitnet-core tensor_

# Quantization benchmarks
cargo bench --package bitnet-quant

# MLX acceleration benchmarks
cargo bench --package bitnet-core --features mlx mlx_
```

## Development Commands

### Code Quality
```bash
# Format code
cargo fmt --all

# Lint code
cargo clippy --workspace --all-targets --all-features

# Check for issues
cargo check --workspace --all-targets --all-features

# Audit dependencies
cargo audit
```

### Documentation
```bash
# Generate documentation
cargo doc --workspace --no-deps --open

# Generate documentation with private items
cargo doc --workspace --document-private-items --open

# Test documentation examples
cargo test --doc --workspace
```

### Release Preparation
```bash
# Dry run release
cargo publish --dry-run --package bitnet-core

# Version management
cargo set-version --workspace 0.3.0

# Check release readiness
cargo release --dry-run
```

## Debugging Commands

### Debug Builds
```bash
# Debug build with full symbols
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --workspace

# Debug with sanitizers
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --workspace

# Debug memory usage
RUSTFLAGS="-Z sanitizer=memory" cargo +nightly test --workspace
```

### Profiling
```bash
# Profile with perf (Linux)
perf record --call-graph=dwarf cargo bench --package bitnet-benchmarks
perf report

# Profile with Instruments (macOS)
cargo build --release --package bitnet-core --features mlx
xcrun xctrace record --template "Time Profiler" --launch target/release/examples/mlx_acceleration_demo

# Memory profiling with valgrind (Linux)
valgrind --tool=memcheck cargo test --package bitnet-core
```

## Continuous Integration Commands

### Local CI Simulation
```bash
# Run full CI pipeline locally
./scripts/ci-full.sh

# Quick CI check
./scripts/ci-quick.sh

# Platform compatibility check
./scripts/platform-test.sh
```

### Coverage Analysis
```bash
# Install coverage tools
cargo install grcov

# Generate coverage report
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-Zprofile -Ccodegen-units=1 -Copt-level=0 -Clink-dead-code -Coverflow-checks=off -Zpanic_abort_tests"
cargo +nightly build --workspace
cargo +nightly test --workspace
grcov . -s . --binary-path ./target/debug/ -t html --branch --ignore-not-existing -o ./coverage/
```

## Troubleshooting Commands

### Common Issues
```bash
# Fix registry issues
cargo clean
rm -rf ~/.cargo/registry/cache
cargo update

# Fix build cache
cargo clean --package bitnet-core
cargo build --package bitnet-core

# Reset toolchain
rustup update stable
rustup default stable
```

### Performance Issues
```bash
# Profile build times
cargo clean
cargo +nightly build --workspace -Z timings

# Check dependencies
cargo tree --duplicates
cargo machete

# Analyze binary size
cargo bloat --release --crates
```

### Platform-Specific Issues
```bash
# macOS: Reset Xcode tools
xcode-select --install

# Apple Silicon: Verify MLX
python3 -c "import mlx.core as mx; print(mx.default_device())"

# Linux: Install system dependencies
sudo apt-get install build-essential pkg-config
```