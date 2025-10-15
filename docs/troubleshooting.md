# BitNet Troubleshooting Guide

**Version**: 1.0.0  
**Last Updated**: October 14, 2025  
**Target Audience**: Developers troubleshooting BitNet-Rust issues

## Overview

This guide provides solutions for common issues encountered when using BitNet-Rust for inference, model loading, and performance optimization. The guide covers error diagnosis, resolution steps, and preventive measures.

## Quick Diagnosis

### System Health Check

Run the built-in validation to identify common issues:

```bash
# Quick system validation
bitnet validate

# Comprehensive health check with benchmarks
bitnet validate --benchmark

# Check specific components
bitnet validate --memory --gpu --models
```

### Common Error Patterns

| Error Pattern | Category | Quick Solution |
|---------------|----------|----------------|
| `Model loading failed` | Model Issues | Check model path/permissions |
| `Out of memory` | Memory Issues | Reduce batch size, enable memory pool |
| `Device not available` | Hardware Issues | Check GPU drivers, fall back to CPU |
| `Compilation failed` | Build Issues | Check Rust version, enable features |
| `Network timeout` | Download Issues | Check internet, use cached models |

## Model Loading Issues

### Problem: Model Download Failures

**Symptoms**:
```
Error: Failed to download model 'microsoft/bitnet-b1.58-2B-4T-gguf'
Network timeout or connection refused
```

**Solutions**:

1. **Check Internet Connection**:
```bash
# Test HuggingFace connectivity
curl -I https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf

# Check DNS resolution
nslookup huggingface.co
```

2. **Use Manual Download**:
```bash
# Download manually with git-lfs
git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
bitnet infer chat --model ./bitnet-b1.58-2B-4T-gguf
```

3. **Configure Proxy** (if behind corporate firewall):
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export HF_HUB_OFFLINE=0  # Enable online mode
```

4. **Use Cached Models**:
```bash
# List cached models
bitnet infer list --cached

# Force use cached version
bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf --offline
```

### Problem: GGUF Format Errors

**Symptoms**:
```
Error: Invalid GGUF file format
Failed to parse model metadata
Tensor offset calculation error
```

**Solutions**:

1. **Verify File Integrity**:
```bash
# Check file size and corruption
ls -la ~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/
shasum -a 256 model.gguf  # Compare with expected hash
```

2. **Re-download Model**:
```bash
# Force re-download
bitnet infer download --model microsoft/bitnet-b1.58-2B-4T-gguf --force

# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/
bitnet infer download --model microsoft/bitnet-b1.58-2B-4T-gguf
```

3. **Check GGUF Version Compatibility**:
```rust
// Check supported GGUF versions
use bitnet_inference::gguf::GGUFVersion;
println!("Supported GGUF versions: {:?}", GGUFVersion::supported_versions());
```

### Problem: Model Size Issues

**Symptoms**:
```
Error: Model too large for available memory
Failed to allocate tensor storage
Memory pool exhausted
```

**Solutions**:

1. **Enable Memory Optimization**:
```rust
let config = EngineConfig {
    memory_limit_mb: Some(4096),     // Set memory limit
    use_memory_mapping: true,        // Memory-map large tensors
    streaming_threshold_mb: 100,     // Stream large tensors
    ..Default::default()
};
```

2. **Use Streaming Mode**:
```bash
# Enable streaming for large models
export BITNET_STREAMING_MODE=true
bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf
```

3. **Choose Smaller Model**:
```bash
# Use smaller model variant
bitnet infer chat --model microsoft/bitnet-b1.58-1B-gguf  # If available
```

## Memory Issues

### Problem: Out of Memory Errors

**Symptoms**:
```
Error: Failed to allocate 2147483648 bytes
Out of memory during tensor allocation
Memory pool exhausted during inference
```

**Solutions**:

1. **Reduce Memory Usage**:
```rust
let config = EngineConfig {
    memory_limit_mb: Some(2048),     // Reduce memory limit
    batch_size: 1,                   // Reduce batch size
    max_sequence_length: 512,        // Reduce sequence length
    enable_memory_pool: true,        // Use memory pool
    ..Default::default()
};
```

2. **Enable Memory Pool**:
```rust
use bitnet_core::memory::{MemoryPoolConfig, AllocationStrategy};

let memory_config = MemoryPoolConfig {
    initial_size_mb: 512,
    max_size_mb: Some(2048),
    allocation_strategy: AllocationStrategy::Conservative,
    enable_tracking: true,
    ..Default::default()
};
```

3. **Monitor Memory Usage**:
```rust
// Check memory stats
let stats = engine.get_memory_stats();
if stats.usage_percent > 90.0 {
    println!("High memory usage detected: {:.1}%", stats.usage_percent);
    engine.clear_cache().await?;
}
```

### Problem: Memory Leaks

**Symptoms**:
```
Memory usage continuously increasing
System becoming unresponsive over time
Out of memory after extended use
```

**Solutions**:

1. **Enable Memory Tracking**:
```rust
let config = EngineConfig {
    enable_memory_tracking: true,
    memory_leak_detection: true,
    cleanup_interval: Duration::from_secs(300), // 5-minute cleanup
    ..Default::default()
};
```

2. **Manual Cleanup**:
```rust
// Periodic cleanup
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(300));
    loop {
        interval.tick().await;
        engine.cleanup_memory().await;
    }
});
```

3. **Check for Unreleased Resources**:
```bash
# Enable memory debugging
export RUST_LOG=bitnet::memory=debug
export BITNET_MEMORY_DEBUG=true
```

## Performance Issues

### Problem: Slow Inference

**Symptoms**:
```
Very slow text generation (< 1 token/sec)
High CPU/GPU usage but low throughput
Long initialization times
```

**Solutions**:

1. **Check Device Selection**:
```rust
// Verify optimal device is selected
let available_devices = Device::available_devices();
println!("Available devices: {:?}", available_devices);

// Use best available device
let config = EngineConfig {
    device: Device::best_available(),
    fallback_device: Some(Device::Cpu),
    ..Default::default()
};
```

2. **Enable SIMD Optimizations**:
```bash
# Build with SIMD optimizations
export RUSTFLAGS="-C target-feature=+neon"  # ARM64
export RUSTFLAGS="-C target-feature=+avx2,+fma"  # x86_64
cargo build --release
```

3. **Optimize Thread Configuration**:
```rust
let config = EngineConfig {
    thread_count: Some(num_cpus::get()),
    thread_affinity: Some(ThreadAffinity::Core),
    parallel_processing: true,
    ..Default::default()
};
```

4. **Check Model Loading Strategy**:
```rust
// Use memory mapping for large models
let config = EngineConfig {
    use_memory_mapping: true,
    preload_tensors: false,  // Lazy loading
    cache_frequently_used: true,
    ..Default::default()
};
```

### Problem: GPU Performance Issues

**Symptoms**:
```
GPU utilization low despite GPU being selected
Slower performance on GPU than CPU
GPU memory errors
```

**Solutions**:

1. **Verify GPU Setup** (Metal - macOS):
```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep "Metal"

# Verify Metal device availability
```

```rust
use bitnet_core::Device;
match Device::Metal.is_available() {
    true => println!("Metal GPU available"),
    false => println!("Metal GPU not available, using CPU"),
}
```

2. **Verify GPU Setup** (CUDA - Linux):
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check CUDA device availability
export CUDA_VISIBLE_DEVICES=0
```

3. **Optimize GPU Memory**:
```rust
let gpu_config = EngineConfig {
    device: Device::Metal, // or Device::Cuda
    memory_limit_mb: Some(6144),  // Adjust for GPU memory
    gpu_memory_fraction: 0.8,     // Use 80% of GPU memory
    enable_unified_memory: true,  // Metal only
    ..Default::default()
};
```

4. **Check for Context Leaks** (Metal):
```bash
# Monitor for Metal context leaks
# Look for "Context leak detected" messages in logs
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=1
```

## Hardware Compatibility Issues

### Problem: ARM64 NEON Not Working

**Symptoms**:
```
NEON optimizations not active
Poor performance on Apple Silicon
SIMD operations falling back to scalar
```

**Solutions**:

1. **Verify NEON Support**:
```bash
# Check CPU features (macOS)
sysctl -a | grep cpu.features

# Check if NEON is compiled in
objdump -t target/release/bitnet | grep neon
```

2. **Enable NEON Compilation**:
```bash
# Ensure NEON features are enabled
export RUSTFLAGS="-C target-feature=+neon"
cargo clean && cargo build --release

# Verify NEON is active at runtime
export BITNET_SIMD_DEBUG=true
```

3. **Check Target Architecture**:
```bash
# Verify building for correct target
rustc --print target-list | grep aarch64
cargo build --release --target aarch64-apple-darwin
```

### Problem: x86_64 SIMD Issues

**Symptoms**:
```
AVX2/AVX-512 not being used
Slow performance on Intel/AMD CPUs
SIMD feature detection failing
```

**Solutions**:

1. **Check CPU Features**:
```bash
# Check available CPU features (Linux)
cat /proc/cpuinfo | grep flags

# Check available CPU features (macOS Intel)
sysctl -a | grep cpu.features
```

2. **Enable x86_64 SIMD**:
```bash
# Enable AVX2 and FMA
export RUSTFLAGS="-C target-feature=+avx2,+fma -C target-cpu=native"
cargo build --release
```

3. **Runtime Feature Detection**:
```rust
use bitnet_core::cpu::CpuFeatures;

let features = CpuFeatures::detect();
println!("AVX2: {}", features.avx2);
println!("FMA: {}", features.fma);
println!("NEON: {}", features.neon);
```

## Build and Compilation Issues

### Problem: Rust Version Compatibility

**Symptoms**:
```
Compilation error: feature not available in Rust 1.xx
Missing feature flags
Dependency version conflicts
```

**Solutions**:

1. **Check Rust Version**:
```bash
rustc --version
# Should be 1.70.0 or newer

# Update if needed
rustup update stable
```

2. **Install Required Components**:
```bash
# Install required Rust components
rustup component add clippy rustfmt

# Install cargo tools
cargo install cargo-audit cargo-outdated
```

3. **Check Feature Flags**:
```bash
# Build with all features
cargo build --release --all-features

# Build with specific features
cargo build --release --features "cuda,metal,simd"
```

### Problem: CUDA Compilation Issues

**Symptoms**:
```
CUDA headers not found
nvcc compilation errors
Linking errors with CUDA libraries
```

**Solutions**:

1. **Install CUDA Toolkit**:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# Verify installation
nvcc --version
```

2. **Set Environment Variables**:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
```

3. **Build with CUDA Support**:
```bash
cargo build --release --features "cuda"
```

## Network and Connectivity Issues

### Problem: HuggingFace Hub Access

**Symptoms**:
```
Cannot connect to HuggingFace Hub
Authentication errors
Rate limiting issues
```

**Solutions**:

1. **Check Network Connectivity**:
```bash
# Test HuggingFace Hub connectivity
curl -I https://huggingface.co

# Test model repository access
curl -I https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
```

2. **Authentication Setup**:
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login with token
huggingface-cli login

# Verify authentication
huggingface-cli whoami
```

3. **Configure Offline Mode**:
```bash
# Use offline mode for cached models
export HF_HUB_OFFLINE=1
bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf
```

### Problem: Proxy Configuration

**Symptoms**:
```
Connection timeout behind corporate firewall
Proxy authentication required
SSL certificate errors
```

**Solutions**:

1. **Configure HTTP Proxy**:
```bash
export HTTP_PROXY=http://username:password@proxy.company.com:8080
export HTTPS_PROXY=http://username:password@proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1,.company.com
```

2. **SSL Certificate Issues**:
```bash
# Disable SSL verification (not recommended for production)
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""

# Or configure certificate bundle
export REQUESTS_CA_BUNDLE=/path/to/cacert.pem
```

## Error Code Reference

### Exit Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 0 | Success | Operation completed successfully |
| 1 | General error | Unspecified error occurred |
| 2 | Configuration error | Invalid config file or settings |
| 3 | Model loading error | Model file issues or format errors |
| 4 | Inference error | Runtime inference failure |
| 5 | Network error | Download or connectivity issues |
| 6 | Hardware error | GPU/CUDA/Metal not available |
| 7 | Memory error | Out of memory or allocation failure |
| 8 | Permission error | File access or write permission issues |

### Common Error Messages

#### Model Loading Errors

```
"Model file not found" → Check file path and permissions
"Invalid GGUF format" → Re-download model or verify file integrity  
"Unsupported model architecture" → Use compatible BitNet model
"Model size exceeds memory limit" → Increase memory limit or use smaller model
```

#### Inference Errors

```
"Out of memory during inference" → Reduce batch size or enable memory optimization
"Device not available" → Check GPU drivers or fall back to CPU
"Generation timeout" → Increase timeout or check model performance
"Invalid prompt format" → Verify prompt encoding and special tokens
```

#### Performance Errors

```
"SIMD not available" → Check CPU features and compilation flags
"GPU context creation failed" → Verify GPU drivers and availability
"Memory pool exhausted" → Increase pool size or enable streaming
"Thread pool initialization failed" → Check thread limits and permissions
```

## Debugging Tools

### 1. Enable Debug Logging

```bash
# Enable detailed logging
export RUST_LOG=bitnet=debug,bitnet_inference=debug,bitnet_core=debug

# Enable specific component logging
export RUST_LOG=bitnet::memory=trace,bitnet::gpu=debug

# Enable performance profiling
export BITNET_PROFILE=true
export BITNET_PROFILE_FILE=bitnet_profile.json
```

### 2. Memory Debugging

```bash
# Enable memory debugging
export BITNET_MEMORY_DEBUG=true
export BITNET_TRACK_ALLOCATIONS=true

# Valgrind (Linux)
valgrind --tool=memcheck --leak-check=full ./target/release/bitnet

# Activity Monitor (macOS)
# Monitor memory usage in Activity Monitor
```

### 3. Performance Profiling

```bash
# Use built-in profiler
export BITNET_PROFILE=true
bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf

# External profiling tools
# macOS: Instruments.app
# Linux: perf, vtune
```

## Getting Help

### 1. Collect Diagnostic Information

```bash
# Generate diagnostic report
bitnet validate --verbose --output diagnostic-report.json

# System information
bitnet --version
rustc --version
uname -a  # Linux/macOS
```

### 2. Community Support

- **GitHub Issues**: [bitnet-rust/issues](https://github.com/leizerowicz/bitnet-rust/issues)
- **Discussions**: [bitnet-rust/discussions](https://github.com/leizerowicz/bitnet-rust/discussions)
- **Documentation**: [docs.rs/bitnet-rust](https://docs.rs/bitnet-rust)

### 3. Issue Reporting Template

When reporting issues, include:

```
**Environment**:
- OS: [macOS 14.0 / Ubuntu 22.04 / Windows 11]
- Rust version: [rustc --version]
- BitNet version: [bitnet --version]
- Hardware: [CPU model, RAM, GPU if applicable]

**Problem Description**:
[Clear description of the issue]

**Steps to Reproduce**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior**:
[What you expected to happen]

**Actual Behavior**:
[What actually happened]

**Error Messages**:
```
[Error logs with stack traces]
```

**Configuration**:
[Your config.toml or environment variables]

**Additional Context**:
[Any other relevant information]
```

## Prevention and Best Practices

### 1. Regular Maintenance

```bash
# Update dependencies regularly
cargo update

# Clean build artifacts
cargo clean

# Check for security advisories
cargo audit
```

### 2. Configuration Validation

```bash
# Validate configuration before use
bitnet config validate

# Test system compatibility
bitnet validate --comprehensive
```

### 3. Monitoring and Alerting

```rust
// Set up basic monitoring
let monitor = SystemMonitor::new()
    .memory_threshold(0.9)
    .latency_threshold(Duration::from_secs(5))
    .error_rate_threshold(0.05);

monitor.start_monitoring().await?;
```

## Next Steps

- **[Inference Guide](inference-guide.md)** - Basic inference setup
- **[Performance Optimization](performance-optimization.md)** - Advanced performance tuning
- **[CLI Reference](cli-reference.md)** - Complete command reference
- **[Examples](../examples/)** - Working examples and tutorials

## Support Resources

- **Documentation**: [Complete guides and API reference](../docs/)
- **Examples**: [Real-world usage examples](../examples/)
- **Benchmarks**: [Performance testing and validation](../bitnet-benchmarks/)
- **Community**: [GitHub Discussions and Issues](https://github.com/leizerowicz/bitnet-rust)