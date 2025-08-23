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

# Install system dependencies (macOS)
brew install pkg-config
xcode-select --install

# Install Rust toolchain with specific version
rustup toolchain install stable
rustup default stable
rustup component add clippy rustfmt
```

### Development Environment Setup
```bash
# Install development tools
cargo install cargo-watch cargo-audit cargo-machete cargo-bloat
cargo install criterion-table  # For benchmark result formatting
cargo install grcov           # For code coverage analysis

# Setup pre-commit hooks
cargo install pre-commit
pre-commit install

# Configure development environment  
export RUST_LOG=debug
export RUSTFLAGS="-C target-cpu=native"  # Enable CPU-specific optimizations
export MLX_ENABLE_VALIDATION=1            # Enable MLX validation (Apple Silicon)
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

## Advanced Development Workflows

### Feature Development Workflow
```bash
# Create feature branch
git checkout -b feature/new-optimization
git push -u origin feature/new-optimization

# Development with hot reload
cargo watch -x "build --package bitnet-core --release"
cargo watch -x "test --package bitnet-core" -x "clippy"

# Run specific benchmark during development
cargo watch -x "bench --package bitnet-core memory_pool_"

# Validate changes before commit
./scripts/validate-changes.sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

### Performance Optimization Workflow
```bash
# Profile-guided optimization
export RUSTFLAGS="-C profile-generate=/tmp/pgo-data"
cargo build --release --workspace
cargo run --example performance_test --package bitnet-core --release
export RUSTFLAGS="-C profile-use=/tmp/pgo-data"
cargo build --release --workspace

# Detailed performance analysis
cargo build --release --package bitnet-core
perf record --call-graph=dwarf ./target/release/examples/mlx_acceleration_demo
perf report --stdio > performance_analysis.txt

# Memory profiling with Valgrind (Linux)
cargo build --package bitnet-core
valgrind --tool=memcheck --leak-check=full ./target/debug/examples/memory_tracking_demo

# Apple Silicon profiling with Instruments
cargo build --release --package bitnet-core --features mlx
xcrun xctrace record --template "Allocations" --launch ./target/release/examples/mlx_acceleration_demo
```

### Cross-Platform Validation
```bash
# x86_64 validation
RUSTFLAGS="-C target-cpu=x86-64" cargo test --workspace --release
RUSTFLAGS="-C target-cpu=x86-64" cargo bench --workspace

# ARM64 validation (Apple Silicon)
RUSTFLAGS="-C target-cpu=apple-m1" cargo test --workspace --release --features mlx
RUSTFLAGS="-C target-cpu=apple-m1" cargo bench --workspace --features mlx

# Feature compatibility testing
cargo test --workspace --no-default-features
cargo test --workspace --all-features
cargo test --workspace --features "mlx,metal,parallel"
```

### Production Deployment Commands
```bash
# Production build with optimizations
RUSTFLAGS="-C target-cpu=native -C lto=fat -C panic=abort" cargo build --release --workspace

# Create production containers
docker build -t bitnet-rust:latest .
docker build -t bitnet-rust:apple-silicon --build-arg FEATURES="mlx,metal" .

# Kubernetes deployment
kubectl apply -f deployment/k8s/
kubectl rollout status deployment/bitnet-inference

# Performance validation in production
kubectl exec -it deployment/bitnet-inference -- cargo run --package bitnet-benchmarks -- validate --quick
```

## Advanced Performance Commands

### Memory Analysis and Optimization
```bash
# Detailed memory analysis
cargo run --example memory_tracking_demo --package bitnet-core --release 2>&1 | tee memory_analysis.log

# Memory leak detection
cargo test --package bitnet-core --release memory_ 2>&1 | grep -E "(LEAK|ERROR)"

# Memory fragmentation analysis
cargo run --example cleanup_system_demo --package bitnet-core --release --features detailed-metrics

# Memory pressure testing
cargo run --example memory_pressure_test --package bitnet-core --release --features stress-testing
```

### GPU and Acceleration Analysis  
```bash
# Metal GPU performance deep dive
cargo run --example metal_performance_analysis --package bitnet-metal --release

# MLX acceleration validation (Apple Silicon)
cargo run --example mlx_comprehensive_test --package bitnet-core --release --features mlx

# Cross-acceleration comparison
cargo run --package bitnet-benchmarks -- acceleration-comparison --output acceleration_report.json

# GPU memory utilization
cargo run --example gpu_memory_analysis --package bitnet-metal --release --features memory-profiling
```

### Quantization and Model Analysis
```bash
# Quantization accuracy analysis
cargo run --example quantization_accuracy_test --package bitnet-quant --release

# Model conversion and validation
cargo run --package bitnet-cli -- model convert --input model.safetensors --output model.bitnet --validate

# QAT training validation
cargo run --example qat_training_demo --package bitnet-training --release --features validation

# BitLinear layer performance
cargo run --example bitlinear_performance_test --package bitnet-quant --release
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