# BitNet-Rust Tensor Implementation Guide

## 📋 Overview

This guide provides comprehensive documentation for implementing and using the BitNet tensor system. The tensor implementation is built on top of BitNet's sophisticated memory management and device abstraction infrastructure, providing production-ready tensor operations for neural network applications.

## 🏗️ Architecture Overview

### Core Components

The BitNet tensor system consists of several integrated components:

```
BitNetTensor (Core API)
├── TensorStorage (Memory Backend)
│   └── HybridMemoryPool Integration
├── TensorShape (Broadcasting & Layout)
├── BitNetDType (Data Types)
├── TensorDeviceManager (Device Abstraction)
└── TensorOperations (Mathematical Operations)
    ├── Arithmetic Operations
    ├── Linear Algebra
    ├── Reduction Operations
    └── Acceleration Backends
        ├── MLX (Apple Silicon)
        ├── Metal (GPU)
        └── SIMD (Cross-platform)
```

### Memory Management Integration

The tensor system leverages BitNet's production-ready `HybridMemoryPool`:

```rust
// Automatic memory pool integration
let tensor = BitNetTensor::zeros(&[1024, 1024], BitNetDType::F32, None)?;
// Memory automatically allocated from appropriate pool (small/large block)
// Device selection automatic via auto_select_device()
```

### Device Abstraction Integration

Seamless device management through existing device abstraction:

```rust
// Automatic device selection
let tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, None)?;

// Explicit device specification  
let cpu_tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, Some(Device::Cpu))?;

// Device migration leverages existing infrastructure
let gpu_tensor = cpu_tensor.to_device(&Device::Metal(0))?;
```

## 🚀 Quick Start

### Basic Tensor Creation

```rust
use bitnet_core::tensor::{BitNetTensor, BitNetDType};
use candle_core::Device;

// Create zero-filled tensor
let zeros = BitNetTensor::zeros(&[3, 4], BitNetDType::F32, None)?;

// Create ones-filled tensor  
let ones = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None)?;

// Create tensor from data
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let tensor = BitNetTensor::from_slice(&data, &[2, 2], BitNetDType::F32, None)?;

// Random tensor
let random = BitNetTensor::randn(&[100, 784], BitNetDType::F32, None)?;
```

### Basic Operations

```rust
// Arithmetic operations
let a = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;
let b = BitNetTensor::ones(&[3, 3], BitNetDType::F32, None)?;

let sum = &a + &b;  // Element-wise addition
let product = &a * &b;  // Element-wise multiplication
let matrix_mult = a.matmul(&b)?;  // Matrix multiplication

// Broadcasting operations
let a = BitNetTensor::ones(&[3, 1], BitNetDType::F32, None)?;
let b = BitNetTensor::ones(&[1, 4], BitNetDType::F32, None)?;
let broadcast_result = &a + &b;  // Results in [3, 4] tensor

// Reduction operations
let data = BitNetTensor::randn(&[10, 20], BitNetDType::F32, None)?;
let sum_all = data.sum(None, false)?;  // Sum all elements
let sum_axis0 = data.sum(Some(0), false)?;  // Sum along axis 0
let mean = data.mean(None, false)?;  // Mean of all elements
```

### Shape Operations

```rust
// Shape manipulation
let tensor = BitNetTensor::ones(&[2, 3, 4], BitNetDType::F32, None)?;

let reshaped = tensor.reshape(&[6, 4])?;  // Reshape to [6, 4]
let transposed = tensor.transpose(0, 2)?;  // Transpose dimensions 0 and 2
let squeezed = tensor.squeeze(1)?;  // Remove dimension of size 1
let unsqueezed = tensor.unsqueeze(0)?;  // Add dimension of size 1

// Indexing and slicing
let slice = tensor.slice(0, 1, None)?;  // Slice along dimension 0
let index = tensor.index(&[0, 1])?;  // Index specific elements
```

## 🎯 Advanced Features

### Device Management

```rust
use candle_core::Device;

// Automatic device selection (CPU/Metal/MLX based on availability)
let tensor = BitNetTensor::zeros(&[1000, 1000], BitNetDType::F32, None)?;

// Explicit device targeting
let cpu_tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, Some(Device::Cpu))?;

#[cfg(feature = "metal")]
{
    let metal_device = Device::Metal(0);
    let gpu_tensor = BitNetTensor::zeros(&[1000, 1000], BitNetDType::F32, Some(metal_device))?;
}

// Device migration
let migrated = tensor.to_device(&Device::Cpu)?;
```

### MLX Acceleration (Apple Silicon)

```rust
#[cfg(feature = "mlx")]
{
    // MLX acceleration automatically enabled on supported devices
    let a = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;
    let b = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;
    
    // Matrix multiplication with MLX acceleration (up to 40x speedup)
    let result = a.matmul(&b)?;  // Automatically uses MLX when available
}
```

### Memory Efficiency

```rust
// Zero-copy operations where possible
let tensor = BitNetTensor::ones(&[1000, 1000], BitNetDType::F32, None)?;
let view = tensor.view(&[500, 2000])?;  // Zero-copy reshape
let slice = tensor.slice(0, 100, Some(200))?;  // Zero-copy slice

// In-place operations for memory efficiency
let mut tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, None)?;
tensor.add_(&other_tensor)?;  // In-place addition
tensor.mul_(&scalar)?;  // In-place scalar multiplication
```

### Broadcasting System

```rust
// NumPy/PyTorch compatible broadcasting
let a = BitNetTensor::ones(&[8, 1, 6, 1], BitNetDType::F32, None)?;
let b = BitNetTensor::ones(&[7, 1, 5], BitNetDType::F32, None)?;

// Broadcasting automatically handles dimension alignment
let result = &a + &b;  // Results in shape [8, 7, 6, 5]

// Verify broadcast compatibility
if a.shape().is_broadcastable_with(b.shape()) {
    let result = &a * &b;
}
```

## 🔧 Data Types

### Supported Types

```rust
use bitnet_core::tensor::BitNetDType;

// Standard floating point
BitNetDType::F32  // 32-bit float
BitNetDType::F16  // 16-bit float (half precision)
BitNetDType::F64  // 64-bit float (double precision)

// Integer types  
BitNetDType::I8   // 8-bit signed integer
BitNetDType::I16  // 16-bit signed integer
BitNetDType::I32  // 32-bit signed integer
BitNetDType::I64  // 64-bit signed integer

// Unsigned integers
BitNetDType::U8   // 8-bit unsigned integer
BitNetDType::U16  // 16-bit unsigned integer
BitNetDType::U32  // 32-bit unsigned integer

// Boolean
BitNetDType::Bool // Boolean type

// BitNet-specific types
BitNetDType::Ternary  // -1, 0, +1 values for BitNet
BitNetDType::Binary   // -1, +1 values for binary networks
```

### Type Conversion

```rust
// Type casting
let f32_tensor = BitNetTensor::ones(&[10, 10], BitNetDType::F32, None)?;
let f16_tensor = f32_tensor.to_dtype(BitNetDType::F16)?;
let int_tensor = f32_tensor.to_dtype(BitNetDType::I32)?;

// BitNet-specific conversions
let ternary = f32_tensor.to_ternary()?;  // Convert to ternary values
let binary = f32_tensor.to_binary()?;   // Convert to binary values
```

## 🧮 Mathematical Operations

### Arithmetic Operations

```rust
let a = BitNetTensor::randn(&[100, 100], BitNetDType::F32, None)?;
let b = BitNetTensor::randn(&[100, 100], BitNetDType::F32, None)?;
let scalar = 2.5f32;

// Element-wise operations
let add_result = &a + &b;              // Addition
let sub_result = &a - &b;              // Subtraction  
let mul_result = &a * &b;              // Element-wise multiplication
let div_result = &a / &b;              // Element-wise division
let pow_result = a.pow(&b)?;           // Element-wise power

// Scalar operations
let scalar_add = &a + scalar;          // Scalar addition
let scalar_mul = &a * scalar;          // Scalar multiplication

// In-place operations
let mut tensor = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, None)?;
tensor.add_(&b)?;                      // In-place addition
tensor.mul_(scalar)?;                  // In-place scalar multiplication
```

### Linear Algebra

```rust
// Matrix operations
let a = BitNetTensor::randn(&[128, 64], BitNetDType::F32, None)?;
let b = BitNetTensor::randn(&[64, 32], BitNetDType::F32, None)?;

let matmul_result = a.matmul(&b)?;     // Matrix multiplication [128, 32]
let dot_result = a.dot(&b)?;           // Dot product (for vectors)

// Matrix decompositions
let matrix = BitNetTensor::randn(&[100, 100], BitNetDType::F32, None)?;
let (q, r) = matrix.qr()?;             // QR decomposition
let (u, s, vt) = matrix.svd()?;        // SVD decomposition
let chol = matrix.cholesky()?;          // Cholesky decomposition

// Matrix properties
let det = matrix.det()?;               // Determinant
let inv = matrix.inv()?;               // Matrix inverse
let trace = matrix.trace()?;           // Trace (sum of diagonal)
```

### Reduction Operations

```rust
let data = BitNetTensor::randn(&[10, 20, 30], BitNetDType::F32, None)?;

// Global reductions
let sum_all = data.sum(None, false)?;          // Sum all elements
let mean_all = data.mean(None, false)?;        // Mean all elements
let max_all = data.max(None, false)?;          // Maximum element
let min_all = data.min(None, false)?;          // Minimum element

// Axis-specific reductions  
let sum_axis0 = data.sum(Some(0), false)?;     // Sum along axis 0 -> [20, 30]
let mean_axis1 = data.mean(Some(1), true)?;    // Mean along axis 1, keep dims -> [10, 1, 30]

// Statistical operations
let std_dev = data.std(None, false)?;          // Standard deviation
let variance = data.var(None, false)?;         // Variance
```

### Activation Functions

```rust
let input = BitNetTensor::randn(&[batch_size, features], BitNetDType::F32, None)?;

// Standard activation functions
let relu_out = input.relu()?;                  // ReLU activation
let gelu_out = input.gelu()?;                  // GELU activation  
let sigmoid_out = input.sigmoid()?;            // Sigmoid activation
let tanh_out = input.tanh()?;                  // Tanh activation

// Softmax (with numerical stability)
let logits = BitNetTensor::randn(&[batch_size, num_classes], BitNetDType::F32, None)?;
let probabilities = logits.softmax(1)?;        // Softmax along class dimension

// BitNet-specific activations
let sign_out = input.sign()?;                  // Sign activation (-1, 0, 1)
let hard_tanh = input.hard_tanh()?;            // Hard tanh activation
```

## ⚡ Performance Optimization

### Acceleration Backends

The tensor system automatically selects the best acceleration backend:

1. **MLX (Apple Silicon)**: Up to 40x speedup for matrix operations
2. **Metal (GPU)**: Efficient GPU compute shader utilization  
3. **SIMD (CPU)**: Cross-platform vectorization (AVX2, NEON, SSE)

```rust
// Performance is automatically optimized
let a = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;
let b = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;

// This operation will use:
// - MLX acceleration on Apple Silicon (M1/M2/M3)
// - Metal GPU acceleration if available
// - SIMD CPU optimization otherwise
let result = a.matmul(&b)?;
```

### Memory Optimization

```rust
// Use appropriate tensor sizes for memory pools
let small_tensor = BitNetTensor::zeros(&[100, 100], BitNetDType::F32, None)?;  // Small block pool
let large_tensor = BitNetTensor::zeros(&[2000, 2000], BitNetDType::F32, None)?;  // Large block pool

// Prefer in-place operations
let mut tensor = BitNetTensor::randn(&[1000, 1000], BitNetDType::F32, None)?;
tensor.add_(&other_tensor)?;  // More memory efficient than tensor + other_tensor

// Use views for zero-copy operations
let view = tensor.view(&[500, 2000])?;  // Zero-copy reshape
let slice = tensor.slice(0, 100, None)?;  // Zero-copy slice
```

### Threading and Concurrency

```rust
use std::sync::Arc;
use std::thread;

// BitNetTensor is thread-safe through Arc
let tensor = Arc::new(BitNetTensor::randn(&[1000, 1000], BitNetDType::F32, None)?);

let handles: Vec<_> = (0..4).map(|i| {
    let tensor_clone = tensor.clone();
    thread::spawn(move || {
        // Each thread can safely read from the tensor
        let sum = tensor_clone.sum(None, false).unwrap();
        println!("Thread {}: sum = {:?}", i, sum);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## 🔗 Integration with BitNet Components

### Quantization Integration

```rust
use bitnet_quant::tensor_integration::{QuantizedTensor, BitLinearTensor};

// Create quantized tensor
let float_tensor = BitNetTensor::randn(&[128, 256], BitNetDType::F32, None)?;
let quantized = QuantizedTensor::from_float_tensor(&float_tensor, 8, true)?;

// BitLinear operations
let weight = BitNetTensor::randn(&[256, 512], BitNetDType::F32, None)?;
let bitlinear = BitLinearTensor::new(weight)?;
let output = bitlinear.forward(&float_tensor)?;
```

### Memory Pool Integration

```rust
// Tensors automatically use HybridMemoryPool
let tensor = BitNetTensor::zeros(&[1024, 1024], BitNetDType::F32, None)?;

// Memory statistics available through existing infrastructure
let stats = crate::memory::get_memory_stats();
println!("Memory pool utilization: {:.2}%", stats.utilization_percent());
```

### Device Migration

```rust
// Seamless device migration using existing device abstraction
let cpu_tensor = BitNetTensor::ones(&[100, 100], BitNetDType::F32, Some(Device::Cpu))?;

#[cfg(feature = "metal")]
{
    let gpu_tensor = cpu_tensor.to_device(&Device::Metal(0))?;
    // Operations automatically use appropriate acceleration
    let result = gpu_tensor.matmul(&gpu_tensor)?;
}
```

## 🛡️ Error Handling

### Comprehensive Error Types

```rust
use bitnet_core::tensor::TensorError;

match tensor_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(TensorError::ShapeError { expected, actual }) => {
        println!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    },
    Err(TensorError::DeviceError { source, device }) => {
        println!("Device error on {:?}: {}", device, source);
    },
    Err(TensorError::MemoryError { source }) => {
        println!("Memory allocation failed: {}", source);
    },
    Err(TensorError::DataTypeError { expected, actual }) => {
        println!("Data type mismatch: expected {:?}, got {:?}", expected, actual);
    },
    Err(e) => println!("Other error: {}", e),
}
```

### Memory Safety

All tensor operations are memory-safe through:

- **Reference counting**: Automatic cleanup when tensors go out of scope
- **Memory pool integration**: Efficient allocation and deallocation
- **Bounds checking**: All indexing operations are bounds-checked
- **Thread safety**: All operations are thread-safe through proper synchronization

### Recovery Mechanisms

```rust
// Graceful fallback mechanisms
let tensor_result = BitNetTensor::zeros(&[very_large_size], BitNetDType::F32, None);

match tensor_result {
    Ok(tensor) => {
        // Use tensor normally
    },
    Err(MemoryError::OutOfMemory { .. }) => {
        // Fallback to smaller tensor or alternative approach
        let smaller_tensor = BitNetTensor::zeros(&[smaller_size], BitNetDType::F32, None)?;
        // Process in chunks
    },
    Err(e) => return Err(e.into()),
}
```

## 📊 Performance Benchmarking

### Built-in Benchmarking

```rust
use bitnet_benchmarks::tensor::{benchmark_matmul, benchmark_element_wise};

// Benchmark matrix multiplication
let (time_us, gflops) = benchmark_matmul(1024, BitNetDType::F32, None)?;
println!("Matrix multiplication: {} µs, {:.2} GFLOPS", time_us, gflops);

// Benchmark element-wise operations
let (add_time, mul_time) = benchmark_element_wise(&[1000, 1000], BitNetDType::F32)?;
println!("Addition: {} µs, Multiplication: {} µs", add_time, mul_time);
```

### Custom Benchmarks

```rust
use std::time::Instant;

fn benchmark_custom_operation() -> Result<(), Box<dyn std::error::Error>> {
    let a = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;
    let b = BitNetTensor::randn(&[2048, 2048], BitNetDType::F32, None)?;
    
    let start = Instant::now();
    let _result = a.matmul(&b)?;
    let duration = start.elapsed();
    
    let ops = 2.0 * 2048.0 * 2048.0 * 2048.0; // FLOPS for matrix multiplication
    let gflops = ops / (duration.as_secs_f64() * 1e9);
    
    println!("Custom matmul benchmark: {:.2} ms, {:.2} GFLOPS", 
             duration.as_millis(), gflops);
    
    Ok(())
}
```

## 🧪 Testing and Validation

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = BitNetTensor::zeros(&[3, 4], BitNetDType::F32, None)?;
        assert_eq!(tensor.shape().dims(), &[3, 4]);
        assert_eq!(tensor.dtype(), BitNetDType::F32);
        Ok(())
    }
    
    #[test] 
    fn test_arithmetic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let a = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let b = BitNetTensor::ones(&[2, 2], BitNetDType::F32, None)?;
        let result = &a + &b;
        
        // Verify result is 2.0 everywhere
        let data = result.to_vec1::<f32>()?;
        assert!(data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
        Ok(())
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_integration() -> Result<(), Box<dyn std::error::Error>> {
        // Test that tensors properly use memory pools
        let initial_stats = crate::memory::get_memory_stats();
        
        let tensors: Vec<_> = (0..10).map(|_| 
            BitNetTensor::zeros(&[100, 100], BitNetDType::F32, None).unwrap()
        ).collect();
        
        let after_stats = crate::memory::get_memory_stats();
        assert!(after_stats.allocated_bytes() > initial_stats.allocated_bytes());
        
        drop(tensors);
        
        // Memory should be reclaimed (though may be pooled)
        let final_stats = crate::memory::get_memory_stats();
        assert!(final_stats.available_bytes() > after_stats.available_bytes());
        
        Ok(())
    }
}
```

## 🔄 Migration and Compatibility

### From Other Tensor Libraries

```rust
// Migration from PyTorch-style code
// PyTorch: torch.zeros(3, 4, dtype=torch.float32, device='cpu')
let tensor = BitNetTensor::zeros(&[3, 4], BitNetDType::F32, Some(Device::Cpu))?;

// PyTorch: torch.randn(100, 200) @ torch.randn(200, 50)  
let a = BitNetTensor::randn(&[100, 200], BitNetDType::F32, None)?;
let b = BitNetTensor::randn(&[200, 50], BitNetDType::F32, None)?;
let result = a.matmul(&b)?;

// PyTorch: tensor.sum(dim=0, keepdim=True)
let sum_result = tensor.sum(Some(0), true)?;
```

### Candle Integration

```rust
use candle_core::Tensor as CandleTensor;

// Convert to/from Candle tensors when needed
let bitnet_tensor = BitNetTensor::randn(&[10, 10], BitNetDType::F32, None)?;
let candle_tensor: CandleTensor = bitnet_tensor.into();

let candle_tensor = CandleTensor::randn(0.0, 1.0, &[10, 10], &Device::Cpu)?;
let bitnet_tensor = BitNetTensor::from_candle(candle_tensor)?;
```

## 🎯 Best Practices

### Memory Management

1. **Use appropriate tensor sizes**: Small tensors (<64KB) use small block pools, large tensors use large block pools
2. **Prefer in-place operations**: Use `add_()`, `mul_()`, etc. when possible
3. **Leverage zero-copy operations**: Use views and slices instead of copies
4. **Clean up explicitly when needed**: Though automatic cleanup is provided

### Performance Optimization

1. **Let the system choose devices**: Use `None` for automatic optimal device selection
2. **Batch operations**: Process multiple tensors together when possible  
3. **Use appropriate data types**: F16 for memory-constrained applications, F32 for precision
4. **Profile your code**: Use built-in benchmarking tools

### Error Handling

1. **Handle all error cases**: Don't unwrap unless you're certain of success
2. **Use meaningful error messages**: Provide context for debugging
3. **Implement graceful degradation**: Fall back to alternative approaches when possible
4. **Test error paths**: Ensure error handling works correctly

### Threading and Concurrency

1. **Use Arc for shared tensors**: Enable safe multi-threaded access
2. **Avoid excessive cloning**: Use references and views when possible
3. **Consider data parallel operations**: Leverage the thread-safe design
4. **Profile concurrent access**: Ensure minimal lock contention

## 📚 Additional Resources

- **API Documentation**: Generated with `cargo doc --open`
- **Performance Guide**: `docs/tensor_performance_guide.md`
- **Examples**: See `examples/tensor/` directory
- **Benchmarks**: Run with `cargo bench --package bitnet-benchmarks`
- **Integration Tests**: `tests/integration/tensor_integration_tests.rs`

---

*This guide covers the complete tensor implementation. For specific use cases or advanced topics, refer to the API documentation and performance guide.*
