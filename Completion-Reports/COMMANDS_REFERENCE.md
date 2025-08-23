# BitNet-Rust Development Commands Reference
## Complete Command Guide for Development, Testing, and Production

**Repository:** `github.com/Wavegoodvybe2929/bitnet-rust`  
**Shell:** zsh (macOS optimized)  
**Last Updated:** August 23, 2025

---

## 🎯 QUICK START COMMANDS

### Essential Daily Commands
```bash
# Development setup
cargo build --workspace --features development
cargo test --workspace --lib
cargo clippy --workspace -- -D warnings

# Performance validation
cargo bench --workspace --features benchmarks

# Documentation generation
cargo doc --workspace --open --no-deps
```

---

## 🔧 BUILD COMMANDS

### Development Builds
```bash
# Standard development build with all features
cargo build --workspace --features development --all-targets

# Fast incremental build for iteration
cargo build --workspace --features development

# Debug build with extra logging
cargo build --workspace --features development,extra-logging

# Cross-platform development build
cargo build --workspace --features development,simd,cross-platform
```

### Production Builds
```bash
# Production optimized build
cargo build --workspace --features production --release

# Apple Silicon optimized production
cargo build --workspace --features production,mlx,metal,apple-silicon --release

# Cross-platform production build
cargo build --workspace --features production,simd,metal --release

# Maximum optimization build
RUSTFLAGS="-C target-cpu=native" cargo build --workspace --features production,mlx,metal,simd --release
```

### Component-Specific Builds
```bash
# Core tensor operations only
cargo build --package bitnet-core --features tensor-core --release

# Quantization system only  
cargo build --package bitnet-quant --features quantization-complete --release

# Metal GPU acceleration only
cargo build --package bitnet-metal --features metal-kernels --release

# Benchmarking suite only
cargo build --package bitnet-benchmarks --features comprehensive-benchmarks --release
```

---

## 🧪 TESTING COMMANDS

### Unit Testing
```bash
# All unit tests
cargo test --workspace --lib

# Component-specific unit tests
cargo test --package bitnet-core --lib
cargo test --package bitnet-quant --lib  
cargo test --package bitnet-metal --lib

# Feature-specific unit tests
cargo test --package bitnet-core --lib --features tensor-complete
cargo test --package bitnet-quant --lib --features qat-complete
```

### Integration Testing
```bash
# All integration tests
cargo test --workspace --test '*'

# Cross-component integration tests
cargo test --workspace --test '*' --features integration-tests

# Device integration tests
cargo test --workspace --test '*' --features mlx,metal,integration-tests

# Performance integration tests
cargo test --workspace --test '*' --features benchmarks,integration-tests
```

### Specific Test Categories
```bash
# Memory management tests
cargo test --package bitnet-core memory --lib
cargo test --package bitnet-core --test memory_integration_tests

# Tensor operation tests  
cargo test --package bitnet-core tensor --lib
cargo test --package bitnet-core --test tensor_integration_tests

# Quantization tests
cargo test --package bitnet-quant quantization --lib
cargo test --package bitnet-quant --test quantization_integration_tests

# Acceleration tests (MLX/Metal)
cargo test --package bitnet-core acceleration --lib --features mlx,metal
cargo test --package bitnet-metal --test gpu_integration_tests --features metal-kernels

# SIMD optimization tests
cargo test --package bitnet-core simd --lib --features simd
```

### Test Coverage Analysis
```bash
# Install coverage tools (run once)
cargo install cargo-tarpaulin

# Generate test coverage report
cargo tarpaulin --workspace --out html --output-dir coverage

# Component-specific coverage
cargo tarpaulin --package bitnet-core --out html --output-dir coverage/core
cargo tarpaulin --package bitnet-quant --out html --output-dir coverage/quant
```

---

## 📊 BENCHMARKING COMMANDS

### Performance Benchmarking
```bash
# Complete benchmark suite
cargo bench --workspace --features benchmarks

# Component-specific benchmarks
cargo bench --package bitnet-core --features benchmarks
cargo bench --package bitnet-quant --features benchmarks
cargo bench --package bitnet-metal --features benchmarks

# Critical path benchmarks (for 100% completion)
cargo bench --package bitnet-core tensor_ops --features complete-linalg
cargo bench --package bitnet-core metal_acceleration --features metal-kernels
cargo bench --package bitnet-core advanced_linalg --features complete-linalg
```

### Acceleration Benchmarks
```bash
# MLX acceleration benchmarks
cargo bench --package bitnet-core mlx_acceleration --features mlx,benchmarks

# Metal GPU benchmarks  
cargo bench --package bitnet-core metal_acceleration --features metal,benchmarks

# SIMD benchmarks
cargo bench --package bitnet-core simd_optimization --features simd,benchmarks

# Cross-device comparison
cargo bench --package bitnet-benchmarks device_comparison --features mlx,metal,simd
```

### Performance Validation
```bash
# Validate specific performance targets
cargo bench --package bitnet-core svd_512 --features complete-linalg
cargo bench --package bitnet-core qr_512 --features complete-linalg  
cargo bench --package bitnet-core cholesky_512 --features complete-linalg

# GPU speedup validation
cargo bench --package bitnet-core gpu_quantization --features metal-kernels
cargo bench --package bitnet-core gpu_bitlinear --features metal-kernels

# Memory efficiency benchmarks
cargo bench --package bitnet-core memory_efficiency --features benchmarks
```

---

## 🔍 CODE QUALITY COMMANDS

### Linting and Formatting
```bash
# Clippy linting (strict mode)
cargo clippy --workspace -- -D warnings

# Component-specific linting
cargo clippy --package bitnet-core -- -D warnings
cargo clippy --package bitnet-quant -- -D warnings

# Format code
cargo fmt --all

# Check formatting without modifying
cargo fmt --all -- --check
```

### Security Auditing
```bash
# Install audit tools (run once)
cargo install cargo-audit cargo-deny

# Security audit
cargo audit

# Dependency licensing and security  
cargo deny check
```

### Code Analysis
```bash
# Install analysis tools (run once)
cargo install cargo-bloat cargo-outdated

# Binary size analysis
cargo bloat --release --crates --package bitnet-core
cargo bloat --release --crates --package bitnet-quant

# Outdated dependencies check
cargo outdated
```

---

## 📚 DOCUMENTATION COMMANDS

### API Documentation
```bash
# Generate complete documentation
cargo doc --workspace --open --no-deps

# Production documentation with all features
cargo doc --workspace --open --no-deps --features production-complete

# Component-specific documentation
cargo doc --package bitnet-core --open --no-deps
cargo doc --package bitnet-quant --open --no-deps

# Private items documentation (development)
cargo doc --workspace --document-private-items --open --no-deps
```

### Documentation Validation
```bash
# Test documentation examples
cargo test --doc --workspace

# Check documentation links
cargo doc --workspace --no-deps 2>&1 | grep -i warning

# Validate README examples
cargo test --package bitnet-core --doc --features development
```

---

## 🚀 EXAMPLE AND DEMO COMMANDS

### Core Examples
```bash
# Basic tensor operations demo
cargo run --example tensor_operations_demo --features tensor-complete

# Memory management demo
cargo run --example memory_management_demo --features development

# Device abstraction demo  
cargo run --example device_abstraction_demo --features mlx,metal

# Quantization demo
cargo run --example quantization_demo --features quantization-complete
```

### Acceleration Examples
```bash
# MLX acceleration demo
cargo run --example mlx_acceleration_demo --features mlx

# Metal GPU demo
cargo run --example metal_gpu_demo --features metal-kernels

# SIMD optimization demo
cargo run --example simd_optimization_demo --features simd

# Performance comparison demo
cargo run --example performance_comparison_demo --features mlx,metal,simd,benchmarks
```

### Production Examples
```bash
# Production readiness demo
cargo run --example production_readiness_demo --features production-complete

# BitNet model inference demo
cargo run --example bitnet_inference_demo --features production-complete

# Comprehensive feature demo
cargo run --example comprehensive_demo --features production-complete,all-accelerations
```

---

## 🔄 DEVELOPMENT WORKFLOW COMMANDS

### Git Integration
```bash
# Pre-commit validation
cargo test --workspace --lib && cargo clippy --workspace -- -D warnings

# Pre-push validation  
cargo test --workspace && cargo bench --workspace --features benchmarks

# Feature branch workflow
git checkout -b feature/tensor-linear-algebra-completion
git checkout -b feature/metal-gpu-kernels-implementation  
git checkout -b feature/advanced-linear-algebra-features
```

### Continuous Integration Simulation
```bash
# Simulate CI pipeline locally
./scripts/ci_simulation.sh

# Or manual CI steps:
cargo test --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo bench --workspace --features benchmarks
```

---

## 🎯 PRODUCTION COMPLETION COMMANDS

### Phase 1: Tensor Linear Algebra (Days 1-3)
```bash
# Day 1: SVD Implementation
cargo test --package bitnet-core tensor::ops::svd --features complete-linalg
cargo bench --package bitnet-core svd_512 --features complete-linalg

# Day 2: QR Implementation  
cargo test --package bitnet-core tensor::ops::qr --features complete-linalg
cargo bench --package bitnet-core qr_512 --features complete-linalg

# Day 3: Cholesky Implementation
cargo test --package bitnet-core tensor::ops::cholesky --features complete-linalg
cargo bench --package bitnet-core cholesky_512 --features complete-linalg
```

### Phase 2: Metal GPU Kernels (Days 4-6)
```bash
# Day 4: Quantization Kernels
cargo test --package bitnet-core tensor::acceleration::metal::quantization --features metal-kernels
cargo bench --package bitnet-core gpu_quantization --features metal-kernels

# Day 5: BitLinear Kernels
cargo test --package bitnet-core tensor::acceleration::metal::bitlinear --features metal-kernels  
cargo bench --package bitnet-core gpu_bitlinear --features metal-kernels

# Day 6: Matrix Operation Kernels
cargo test --package bitnet-core tensor::acceleration::metal::matmul --features metal-kernels
cargo bench --package bitnet-core gpu_matmul --features metal-kernels
```

### Phase 3: Advanced Features (Days 7-8)
```bash
# Day 7: Eigendecomposition
cargo test --package bitnet-core tensor::ops::eigendecomposition --features complete-linalg
cargo bench --package bitnet-core eigendecomposition_512 --features complete-linalg

# Day 8: Numerical Stability
cargo test --package bitnet-core tensor::ops::numerical_stability --features complete-linalg
cargo bench --package bitnet-core numerical_stability --features complete-linalg
```

### Phase 4: Final Validation (Days 9-10)
```bash
# Day 9: Integration Testing
cargo test --workspace --features production-complete,all-validations
cargo bench --workspace --features production-complete

# Day 10: Production Readiness
cargo run --example production_completion_validation --features production-complete
cargo doc --workspace --open --no-deps --features production-complete
```

---

## 📱 PLATFORM-SPECIFIC COMMANDS

### macOS (Apple Silicon)
```bash
# Optimize for Apple Silicon
cargo build --workspace --features production,mlx,metal --release --target aarch64-apple-darwin

# MLX-specific testing
cargo test --workspace --features mlx --target aarch64-apple-darwin

# Metal performance validation
cargo bench --workspace --features metal,benchmarks --target aarch64-apple-darwin
```

### macOS (Intel)
```bash
# Intel Mac optimization
cargo build --workspace --features production,metal,simd --release --target x86_64-apple-darwin

# AVX2 optimization
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo build --workspace --features production,simd --release
```

### Linux
```bash
# Linux build with SIMD
cargo build --workspace --features production,simd --release --target x86_64-unknown-linux-gnu

# Cross-compilation for Linux ARM64
cargo build --workspace --features production,simd --release --target aarch64-unknown-linux-gnu
```

---

## 🛠️ DEBUGGING COMMANDS

### Debug Builds
```bash
# Debug build with full logging
cargo build --workspace --features development,extra-logging

# Debug with memory sanitizer
RUSTFLAGS="-Z sanitizer=address" cargo build --workspace --features development

# Debug with performance profiling
cargo build --workspace --features development,profiling
```

### Performance Debugging
```bash
# Install profiling tools (run once)
cargo install cargo-flamegraph

# Generate flame graph
cargo flamegraph --example tensor_operations_demo --features benchmarks

# Memory debugging
valgrind --tool=memcheck cargo test --workspace --lib
```

### GPU Debugging
```bash
# Metal debugging (macOS)
export METAL_DEBUG_ERROR_MODE=1
export METAL_DEVICE_WRAPPER_TYPE=1
cargo test --package bitnet-metal --features metal-kernels

# MLX debugging
export MLX_DEBUG=1
cargo test --package bitnet-core --features mlx
```

---

## 🎯 SUCCESS VALIDATION COMMANDS

### 100% Production Readiness Validation
```bash
# Complete validation suite (must pass for 100% score)
cargo test --workspace --features production-complete,all-validations
cargo bench --workspace --features production-complete,comprehensive-benchmarks
cargo clippy --workspace --all-targets --features production-complete -- -D warnings
cargo doc --workspace --no-deps --features production-complete

# Performance target validation
cargo bench --package bitnet-core advanced_linalg_complete --features complete-linalg
cargo bench --package bitnet-core metal_acceleration_complete --features metal-kernels

# Integration validation
cargo run --example production_readiness_demo --features production-complete

# Final deployment check
cargo build --workspace --features production --release
```

This comprehensive command reference provides all the tools needed for efficient development, testing, and production deployment of your BitNet-Rust implementation.
