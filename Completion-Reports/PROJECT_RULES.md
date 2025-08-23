# BitNet-Rust Project Rules & Standards
## Development Guidelines, Code Standards, and Contribution Rules

**Repository:** `github.com/Wavegoodvybe2929/bitnet-rust`  
**Enforcement:** Automated via CI/CD + Manual Review  
**Last Updated:** August 23, 2025

---

## 🎯 CORE PROJECT PRINCIPLES

### 1. **Safety First**
- **Memory Safety**: All code must be memory-safe by design using Rust's ownership system
- **Thread Safety**: All concurrent operations must use proper synchronization primitives
- **Panic Safety**: No panics in production code paths - all errors must be handled via `Result<T, E>`
- **Resource Safety**: All resources (GPU memory, file handles, etc.) must have automatic cleanup

### 2. **Performance by Design**  
- **Zero-Cost Abstractions**: Runtime performance never sacrificed for convenience
- **Memory Efficiency**: Minimize allocations, prefer memory pools and zero-copy operations
- **Cache Friendly**: Data structures optimized for cache locality and memory access patterns
- **Device Optimal**: Code automatically selects optimal compute device (CPU/Metal/MLX)

### 3. **Production Quality**
- **Comprehensive Testing**: All code must have unit tests, integration tests, and benchmarks
- **Documentation**: All public APIs must have complete documentation with examples
- **Error Handling**: Detailed error messages with actionable information
- **Monitoring**: Performance metrics and resource usage tracking built-in

---

## 📋 CODE STANDARDS

### Rust Code Style
```rust
// ✅ CORRECT: Follow these patterns

// 1. Use descriptive names with clear intent
pub struct BitNetTensor {
    data: Arc<TensorStorage>,
    shape: TensorShape,
    device: Device,
}

// 2. Comprehensive error types with context
pub enum TensorOpError {
    IncompatibleShapes { 
        operation: &'static str,
        lhs_shape: Shape, 
        rhs_shape: Shape 
    },
    InsufficientMemory { 
        required: usize, 
        available: usize 
    },
    DeviceError { 
        device: Device, 
        error: String 
    },
}

// 3. Builder pattern for complex construction
impl BitNetTensorBuilder {
    pub fn new(shape: &[usize]) -> Self { /* ... */ }
    pub fn device(mut self, device: Device) -> Self { /* ... */ }
    pub fn dtype(mut self, dtype: DType) -> Self { /* ... */ }
    pub fn build(self) -> TensorOpResult<BitNetTensor> { /* ... */ }
}

// 4. Consistent Result types throughout
pub type TensorOpResult<T> = Result<T, TensorOpError>;

// 5. Proper lifetime management and ownership
pub fn matrix_multiply<'a>(
    lhs: &'a BitNetTensor, 
    rhs: &'a BitNetTensor
) -> TensorOpResult<BitNetTensor> {
    // Implementation must not borrow beyond function scope
}
```

### Forbidden Patterns
```rust
// ❌ FORBIDDEN: Never use these patterns

// 1. Never panic in production code
pub fn divide(a: f32, b: f32) -> f32 {
    assert!(b != 0.0); // ❌ FORBIDDEN - use Result instead
    a / b
}

// 2. Never use unwrap() in production paths  
let tensor = create_tensor().unwrap(); // ❌ FORBIDDEN

// 3. Never use unsafe without extensive documentation
unsafe {
    // ❌ FORBIDDEN unless absolutely necessary with full safety proof
}

// 4. Never use magic numbers
const BUFFER_SIZE: usize = 1024; // ✅ CORRECT
let buffer = vec![0u8; 1024];    // ❌ FORBIDDEN

// 5. Never ignore errors
let _ = tensor_operation(); // ❌ FORBIDDEN
```

---

## 🏗️ ARCHITECTURE RULES

### Memory Management Rules
```rust
// ✅ MANDATORY: Use HybridMemoryPool for all tensor allocations
impl BitNetTensor {
    pub fn new(shape: &[usize], device: Device) -> TensorOpResult<Self> {
        let pool = device.memory_pool();
        let storage = pool.allocate(shape.size())?; // ✅ CORRECT
        
        Ok(BitNetTensor {
            data: storage,
            shape: TensorShape::from(shape),
            device,
        })
    }
}

// ❌ FORBIDDEN: Direct memory allocation
impl BitNetTensor {
    pub fn new_bad(shape: &[usize]) -> Self {
        let data = vec![0.0f32; shape.iter().product()]; // ❌ FORBIDDEN
        // ... 
    }
}
```

### Device Abstraction Rules
```rust
// ✅ MANDATORY: Use auto_select_device for optimal performance
pub fn tensor_operation(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let optimal_device = auto_select_device(&TensorOperation::MatMul { 
        lhs_shape: lhs.shape(), 
        rhs_shape: rhs.shape() 
    }); // ✅ CORRECT
    
    match optimal_device {
        Device::Mlx(dev) => dev.execute_matmul(lhs, rhs),
        Device::Metal(dev) => dev.execute_matmul(lhs, rhs),
        Device::Cpu(dev) => dev.execute_matmul(lhs, rhs),
    }
}

// ❌ FORBIDDEN: Hardcoded device selection
pub fn tensor_operation_bad(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    let cpu_device = CpuDevice::new(); // ❌ FORBIDDEN - not optimal
    cpu_device.execute_matmul(lhs, rhs)
}
```

### Error Handling Rules
```rust
// ✅ MANDATORY: Comprehensive error context
pub fn validate_matrix_multiply(
    lhs: &BitNetTensor, 
    rhs: &BitNetTensor
) -> TensorOpResult<()> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    if lhs_shape.dims().len() != 2 || rhs_shape.dims().len() != 2 {
        return Err(TensorOpError::InvalidDimensions {
            operation: "matrix_multiply",
            expected: "2D tensors".to_string(),
            lhs_dims: lhs_shape.dims().len(),
            rhs_dims: rhs_shape.dims().len(),
        });
    }
    
    if lhs_shape.dims()[1] != rhs_shape.dims()[0] {
        return Err(TensorOpError::IncompatibleShapes {
            operation: "matrix_multiply",
            lhs_shape: lhs_shape.clone(),
            rhs_shape: rhs_shape.clone(),
            reason: format!(
                "Inner dimensions must match: {} != {}", 
                lhs_shape.dims()[1], 
                rhs_shape.dims()[0]
            ),
        });
    }
    
    Ok(())
}

// ❌ FORBIDDEN: Vague or missing error context
pub fn validate_bad(lhs: &BitNetTensor, rhs: &BitNetTensor) -> TensorOpResult<()> {
    if lhs.shape().dims()[1] != rhs.shape().dims()[0] {
        return Err(TensorOpError::Generic("shapes don't match".to_string())); // ❌ FORBIDDEN
    }
    Ok(())
}
```

---

## 🧪 TESTING RULES

### Test Coverage Requirements
```rust
// ✅ MANDATORY: Every public function must have comprehensive tests

#[cfg(test)]
mod tests {
    use super::*;
    
    // 1. Happy path test
    #[test]
    fn test_matrix_multiply_success() {
        let a = BitNetTensor::randn(&[2, 3], Device::Cpu).unwrap();
        let b = BitNetTensor::randn(&[3, 4], Device::Cpu).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 4]);
    }
    
    // 2. Error path tests
    #[test]
    fn test_matrix_multiply_incompatible_shapes() {
        let a = BitNetTensor::randn(&[2, 3], Device::Cpu).unwrap();
        let b = BitNetTensor::randn(&[4, 5], Device::Cpu).unwrap(); // Incompatible
        
        let result = a.matmul(&b);
        assert!(matches!(result, Err(TensorOpError::IncompatibleShapes { .. })));
    }
    
    // 3. Edge case tests
    #[test]  
    fn test_matrix_multiply_empty_tensor() {
        let a = BitNetTensor::zeros(&[0, 0], Device::Cpu).unwrap();
        let b = BitNetTensor::zeros(&[0, 0], Device::Cpu).unwrap();
        
        let result = a.matmul(&b);
        // Test appropriate behavior for empty tensors
    }
    
    // 4. Performance tests
    #[test]
    fn test_matrix_multiply_performance() {
        let a = BitNetTensor::randn(&[512, 512], Device::Cpu).unwrap();
        let b = BitNetTensor::randn(&[512, 512], Device::Cpu).unwrap();
        
        let start = std::time::Instant::now();
        let _ = a.matmul(&b).unwrap();
        let duration = start.elapsed();
        
        // Assert reasonable performance bounds
        assert!(duration.as_millis() < 100); // Adjust based on targets
    }
}
```

### Benchmark Requirements
```rust
// ✅ MANDATORY: Performance-critical functions must have benchmarks

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    // Test multiple sizes for scalability analysis
    for size in [64, 128, 256, 512, 1024].iter() {
        let a = BitNetTensor::randn(&[*size, *size], Device::Cpu).unwrap();
        let b = BitNetTensor::randn(&[*size, *size], Device::Cpu).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _ = a.matmul(&b).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_matrix_multiplication);
criterion_main!(benches);
```

### Integration Test Requirements
```rust
// ✅ MANDATORY: Cross-component integration tests

#[test]
fn test_quantization_memory_integration() {
    // Test quantization with memory pool integration
    let tensor = BitNetTensor::randn(&[1000, 1000], Device::Cpu).unwrap();
    
    // Verify memory pool usage
    let initial_usage = tensor.device().memory_pool().usage();
    
    // Perform quantization
    let quantized = quantize_158bit(&tensor).unwrap();
    
    // Verify memory efficiency
    let final_usage = tensor.device().memory_pool().usage();
    assert!(quantized.memory_size() < tensor.memory_size() / 8); // Significant compression
}

#[test]
fn test_device_fallback_integration() {
    // Test automatic fallback between devices
    let tensor = BitNetTensor::randn(&[100, 100], Device::Auto).unwrap();
    
    // Simulate MLX unavailable
    let result = tensor.matmul(&tensor).unwrap(); // Should fallback gracefully
    
    // Verify correctness regardless of device used
    assert_eq!(result.shape().dims(), &[100, 100]);
}
```

---

## 📊 PERFORMANCE RULES

### Performance Requirements
| Operation | Size | Max Latency | Min Throughput | Device |
|-----------|------|-------------|----------------|---------|
| **Matrix Multiply** | 1024×1024 | 50ms | 40 GFLOPS | CPU |
| **Matrix Multiply** | 1024×1024 | 2ms | 800 GFLOPS | MLX |
| **Quantization** | 1M elements | 10ms | 100M ops/sec | CPU |
| **Quantization** | 1M elements | 1ms | 1B ops/sec | GPU |
| **Memory Allocation** | Any | 100ns | N/A | Memory Pool |

### Benchmark Standards
```rust
// ✅ MANDATORY: All benchmarks must validate performance targets

#[test]
fn validate_performance_target_matrix_multiply() {
    let a = BitNetTensor::randn(&[1024, 1024], Device::Cpu).unwrap();
    let b = BitNetTensor::randn(&[1024, 1024], Device::Cpu).unwrap();
    
    let start = std::time::Instant::now();
    let _ = a.matmul(&b).unwrap();
    let duration = start.elapsed();
    
    // Must meet performance target
    assert!(duration.as_millis() <= 50, 
        "Matrix multiply exceeded target: {}ms > 50ms", 
        duration.as_millis()
    );
}
```

### Memory Efficiency Rules
```rust
// ✅ MANDATORY: Monitor memory usage in all operations

impl BitNetTensor {
    pub fn memory_footprint(&self) -> usize {
        // Must accurately report memory usage
        self.data.size() + std::mem::size_of::<Self>()
    }
    
    pub fn memory_efficiency_ratio(&self) -> f64 {
        // Ratio of actual data to total memory used
        let data_size = self.shape().size() * self.dtype().size();
        let total_size = self.memory_footprint();
        data_size as f64 / total_size as f64
    }
}

// ✅ MANDATORY: Memory efficiency targets
#[test]
fn validate_memory_efficiency() {
    let tensor = BitNetTensor::randn(&[1000, 1000], Device::Cpu).unwrap();
    
    // Memory efficiency must be > 90%
    assert!(tensor.memory_efficiency_ratio() > 0.90,
        "Memory efficiency too low: {:.2}%", 
        tensor.memory_efficiency_ratio() * 100.0
    );
}
```

---

## 📚 DOCUMENTATION RULES

### API Documentation Standards
```rust
/// Matrix multiplication operation with automatic device optimization.
///
/// Performs `lhs × rhs` matrix multiplication, automatically selecting the optimal
/// compute device based on matrix dimensions and available hardware acceleration.
///
/// # Arguments
/// * `lhs` - Left-hand side matrix (must be 2D)  
/// * `rhs` - Right-hand side matrix (must be 2D with compatible dimensions)
///
/// # Returns
/// Returns a new `BitNetTensor` containing the matrix product, or a `TensorOpError`
/// if the operation cannot be completed.
///
/// # Errors
/// * `TensorOpError::IncompatibleShapes` - If matrix dimensions are incompatible
/// * `TensorOpError::InsufficientMemory` - If unable to allocate result tensor
/// * `TensorOpError::DeviceError` - If computation device encounters an error
///
/// # Performance
/// - CPU: ~40 GFLOPS for 1024×1024 matrices (~50ms)
/// - MLX: ~800 GFLOPS for 1024×1024 matrices (~2ms)  
/// - Metal: ~400 GFLOPS for 1024×1024 matrices (~5ms)
///
/// # Examples
/// ```rust
/// use bitnet_core::tensor::BitNetTensor;
/// use bitnet_core::device::Device;
///
/// let a = BitNetTensor::randn(&[128, 256], Device::Auto)?;
/// let b = BitNetTensor::randn(&[256, 512], Device::Auto)?;
/// 
/// let result = a.matmul(&b)?;
/// assert_eq!(result.shape().dims(), &[128, 512]);
/// ```
///
/// # See Also  
/// * [`BitNetTensor::dot`] - Vector dot product
/// * [`BitNetTensor::batch_matmul`] - Batched matrix multiplication
pub fn matmul(&self, other: &BitNetTensor) -> TensorOpResult<BitNetTensor> {
    // Implementation...
}
```

### Code Comments Standards
```rust
impl BitNetTensor {
    pub fn quantize_158bit(&self) -> TensorOpResult<QuantizedTensor> {
        // 1. Validate input tensor for quantization compatibility
        self.validate_quantization_input()?;
        
        // 2. Compute optimal quantization scale using calibration data
        // Scale computation follows BitNet paper: scale = max(abs(weights)) / 1.0
        let scale = self.compute_quantization_scale()?;
        
        // 3. Apply ternary quantization: {-1, 0, +1}
        // Values > scale → +1, values < -scale → -1, others → 0
        let quantized_data = self.apply_ternary_quantization(scale)?;
        
        // 4. Pack quantized values for memory efficiency  
        // 2 bits per weight allows 4 weights per byte storage
        let packed_data = pack_ternary_weights(&quantized_data)?;
        
        Ok(QuantizedTensor::new(packed_data, scale, self.shape().clone()))
    }
}
```

---

## 🔄 GIT WORKFLOW RULES

### Branch Naming Convention
```bash
# ✅ CORRECT: Descriptive branch names with category
feature/tensor-linear-algebra-completion
feature/metal-gpu-kernels-implementation
bugfix/memory-pool-fragmentation-issue
hotfix/critical-quantization-accuracy-bug
refactor/device-abstraction-simplification

# ❌ FORBIDDEN: Vague or personal branch names
john-dev-branch
temp-fix
misc-changes
```

### Commit Message Standards
```bash
# ✅ CORRECT: Clear, descriptive commit messages
feat: implement SVD algorithm with Golub-Reinsch bidiagonalization

- Add real SVD implementation replacing placeholder
- Achieve <50ms performance target for 512×512 matrices  
- Include comprehensive accuracy tests vs NumPy reference
- Add detailed documentation with mathematical background

# ✅ CORRECT: Bug fixes with context
fix: resolve memory leak in Metal GPU buffer pools

- Fix missing buffer release in error paths
- Add automatic cleanup using RAII patterns
- Include regression test for buffer leak detection
- Resolves issue #123

# ❌ FORBIDDEN: Vague commit messages  
fix stuff
update code
wip
```

### Pull Request Requirements
```markdown
## Description
Clear description of changes and motivation

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated and passing
- [ ] Integration tests added/updated and passing
- [ ] Benchmarks added/updated and meeting performance targets
- [ ] All existing tests still passing

## Performance Impact  
- [ ] No performance regression in existing functionality
- [ ] New functionality meets documented performance targets
- [ ] Memory usage impact analyzed and acceptable

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is self-documenting with clear names and comments
- [ ] Documentation updated (if applicable)
- [ ] No `unsafe` code added without extensive documentation
- [ ] All error paths properly tested
```

---

## 🚨 SECURITY RULES

### Security Requirements
```rust
// ✅ MANDATORY: Input validation for all public APIs
pub fn load_tensor_from_file(path: &Path) -> TensorOpResult<BitNetTensor> {
    // 1. Validate file path and permissions
    if !path.exists() {
        return Err(TensorOpError::FileNotFound(path.to_path_buf()));
    }
    
    // 2. Check file size limits to prevent DoS
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_TENSOR_FILE_SIZE {
        return Err(TensorOpError::FileTooLarge { 
            size: metadata.len(),
            max_allowed: MAX_TENSOR_FILE_SIZE 
        });
    }
    
    // 3. Validate file format and headers
    validate_tensor_file_format(path)?;
    
    // Implementation...
}

// ❌ FORBIDDEN: Unchecked external input
pub fn load_tensor_bad(path: &str) -> BitNetTensor {
    let data = std::fs::read(path).unwrap(); // ❌ Multiple security issues
    // ... direct deserialization without validation
}
```

### Memory Safety Rules  
```rust
// ✅ MANDATORY: Safe buffer operations
fn copy_tensor_data(src: &[f32], dst: &mut [f32]) -> TensorOpResult<()> {
    if src.len() != dst.len() {
        return Err(TensorOpError::SizeMismatch {
            src_size: src.len(),
            dst_size: dst.len(),
        });
    }
    
    dst.copy_from_slice(src); // ✅ Safe, bounds-checked copy
    Ok(())
}

// ❌ FORBIDDEN: Unsafe buffer operations
fn copy_tensor_data_bad(src: *const f32, dst: *mut f32, len: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len); // ❌ FORBIDDEN without safety proof
    }
}
```

---

## 🎯 PRODUCTION READINESS RULES

### Production Code Standards
```rust
// ✅ MANDATORY: Production-ready error handling
pub fn critical_tensor_operation(&self) -> TensorOpResult<BitNetTensor> {
    // 1. Comprehensive input validation
    self.validate_operation_preconditions()?;
    
    // 2. Resource acquisition with automatic cleanup
    let _memory_guard = self.device.acquire_memory_guard()?;
    
    // 3. Operation with detailed error context
    let result = self.perform_operation().map_err(|e| {
        TensorOpError::OperationFailed {
            operation: "critical_tensor_operation",
            input_shape: self.shape().clone(),
            device: self.device.clone(),
            underlying_error: Box::new(e),
        }
    })?;
    
    // 4. Result validation
    validate_operation_result(&result)?;
    
    Ok(result)
}
```

### Resource Management Rules
```rust
// ✅ MANDATORY: RAII-based resource management
pub struct GpuMemoryGuard {
    buffer: MetalBuffer,
    device: MetalDevice,
}

impl Drop for GpuMemoryGuard {
    fn drop(&mut self) {
        // Automatic cleanup - never fails in drop
        if let Err(e) = self.device.release_buffer(&self.buffer) {
            eprintln!("Warning: Failed to release GPU buffer: {}", e);
            // Log error but don't panic in drop
        }
    }
}
```

### Monitoring and Telemetry Rules
```rust
// ✅ MANDATORY: Built-in performance monitoring
impl TensorOperation {
    pub fn execute_with_monitoring(&self) -> TensorOpResult<BitNetTensor> {
        let start_time = std::time::Instant::now();
        let initial_memory = get_memory_usage();
        
        let result = self.execute()?;
        
        let duration = start_time.elapsed();
        let memory_used = get_memory_usage() - initial_memory;
        
        // Record metrics for production monitoring
        METRICS.operation_duration.record(duration, &[
            ("operation", self.operation_name()),
            ("device", self.device().name()),
        ]);
        
        METRICS.memory_usage.record(memory_used, &[
            ("operation", self.operation_name()),
        ]);
        
        Ok(result)
    }
}
```

---

## 📋 ENFORCEMENT

### Automated Checks (CI/CD)
- **Code Formatting**: `cargo fmt --all -- --check`
- **Linting**: `cargo clippy --workspace -- -D warnings`  
- **Tests**: `cargo test --workspace --all-targets --all-features`
- **Benchmarks**: `cargo bench --workspace` (performance regression detection)
- **Documentation**: `cargo doc --workspace --no-deps` (link validation)
- **Security**: `cargo audit` (dependency vulnerability scanning)

### Manual Review Requirements  
- **Architecture Changes**: Require 2+ senior developer approvals
- **Performance Critical Code**: Must include benchmarks and performance analysis
- **Unsafe Code**: Requires extensive safety documentation and multiple approvals
- **Public API Changes**: Require API design review and documentation update

### Quality Gates
- **Test Coverage**: Minimum 95% code coverage for production code
- **Performance**: All benchmarks must meet documented targets
- **Documentation**: All public APIs must have complete documentation
- **Security**: No high/critical security vulnerabilities allowed

**🎯 GOAL:** These rules ensure BitNet-Rust maintains world-class code quality while achieving perfect 100/100 production readiness score.
