# BitNet Metal Shaders

This directory contains Metal Shading Language (MSL) compute shaders for BitNet operations. These shaders provide GPU-accelerated implementations of BitNet's core operations including BitLinear layers, quantization, and activation functions.

## Shader Files

### [`bitlinear.metal`](bitlinear.metal)
Implements BitLinear layer operations for 1-bit neural networks:

- **`bitlinear_forward`** - Forward pass for BitLinear layers with 1-bit weights
- **`bitlinear_backward_input`** - Backward pass for input gradients
- **`binarize_weights`** - Weight binarization during training
- **`quantize_activations`** - Activation quantization preprocessing

### [`quantization.metal`](quantization.metal)
Comprehensive quantization kernels for various precision levels:

- **`quantize_weights_1bit`** - 1-bit weight quantization with group-wise scaling
- **`quantize_activations_8bit`** - 8-bit activation quantization with zero-point
- **`dequantize_weights_1bit`** - 1-bit weight dequantization
- **`dequantize_activations_8bit`** - 8-bit activation dequantization
- **`dynamic_quantize_activations`** - Runtime dynamic quantization
- **`quantize_gradients`** - Gradient quantization for training
- **`mixed_precision_matmul`** - Mixed precision matrix multiplication

### [`activation.metal`](activation.metal)
Optimized activation functions with forward and backward passes:

- **ReLU**: `relu_forward`, `relu_backward`
- **GELU**: `gelu_forward`, `gelu_backward`
- **Swish/SiLU**: `swish_forward`, `swish_backward`
- **Sigmoid**: `sigmoid_forward`, `sigmoid_backward`
- **Tanh**: `tanh_forward`, `tanh_backward`
- **Leaky ReLU**: `leaky_relu_forward`, `leaky_relu_backward`
- **Softmax**: `softmax_forward`, `softmax_backward`
- **Layer Normalization**: `layer_norm_forward`
- **Fused Operations**: `fused_relu_dropout`

## Usage

### Using the Shader Compilation Pipeline

```rust
use bitnet_core::metal::*;

// Initialize Metal context
let (device, command_queue, _) = initialize_metal_context()?;

// Create BitNet shader collection
let shaders = BitNetShaders::new(device.clone())?;

// Get a compute pipeline for BitLinear forward pass
let pipeline = shaders.get_pipeline(BitNetShaderFunction::BitLinearForward)?;

// Create command buffer and encoder
let command_buffer = command_queue.new_command_buffer();
let encoder = shaders.create_compute_encoder_with_pipeline(
    &command_buffer, 
    BitNetShaderFunction::BitLinearForward
)?;

// Set buffers and dispatch
encoder.set_buffer(0, Some(&input_buffer), 0);
encoder.set_buffer(1, Some(&weights_buffer), 0);
encoder.set_buffer(3, Some(&output_buffer), 0);

let (threads, threadgroup) = shaders.calculate_dispatch_params(
    BitNetShaderFunction::BitLinearForward, 
    data_size
)?;
dispatch_compute(&encoder, threads, threadgroup);

encoder.end_encoding();
command_buffer.commit();
command_buffer.wait_until_completed();
```

### Using the Shader Compiler Directly

```rust
use bitnet_core::metal::*;

// Create shader compiler
let compiler = create_shader_compiler(&device)?;

// Compile all shaders in the directory
let compiled_shaders = compiler.compile_all_shaders()?;

// Get a specific compute function
let function = compiler.get_compute_function("bitlinear", "bitlinear_forward")?;

// Create pipeline state
let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
```

### High-Level Convenience Functions

```rust
use bitnet_core::metal::*;

// Initialize global shader instance
initialize_global_shaders(device)?;

// Use convenience functions
let encoder = create_bitlinear_forward_encoder(&shaders, &command_buffer)?;

dispatch_bitlinear_forward(
    &encoder,
    &input_buffer,
    &weights_buffer,
    Some(&bias_buffer),
    &output_buffer,
    input_size,
    output_size,
    batch_size,
    threads,
    threadgroup,
);
```

## Shader Architecture

### Thread Organization
- **1D Dispatch**: Most operations use 1D thread dispatch for simplicity
- **Threadgroup Size**: Automatically calculated based on device capabilities
- **SIMD Optimization**: Threadgroup sizes are multiples of SIMD width

### Memory Access Patterns
- **Coalesced Access**: Memory access patterns optimized for GPU memory hierarchy
- **Shared Memory**: Threadgroup memory used for reductions and data sharing
- **Buffer Alignment**: All buffers properly aligned for optimal performance

### Numerical Stability
- **Epsilon Values**: Small epsilon values prevent division by zero
- **Range Clamping**: Values clamped to prevent overflow/underflow
- **Stable Algorithms**: Numerically stable implementations (e.g., stable softmax)

## Performance Considerations

### Optimization Strategies
1. **Memory Bandwidth**: Minimize memory transfers between CPU and GPU
2. **Compute Intensity**: Maximize arithmetic operations per memory access
3. **Occupancy**: Optimize threadgroup sizes for maximum GPU utilization
4. **Precision**: Use appropriate precision levels (1-bit, 8-bit, 16-bit, 32-bit)

### Benchmarking
Use the provided benchmarks to measure performance:

```bash
cd bitnet-benchmarks
cargo bench
```

### Profiling
Use Xcode Instruments or Metal System Trace for detailed profiling:

```bash
# Enable Metal validation layers for debugging
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=1
```

## Compilation Options

### Debug Mode
- Debug information included
- Validation layers enabled
- No optimization

### Release Mode
- Full optimization enabled
- Fast math optimizations
- Minimal debug information

### Custom Configuration
```rust
let config = ShaderCompilerConfig {
    shader_directory: PathBuf::from("custom/shaders"),
    enable_caching: true,
    cache_directory: Some(PathBuf::from("target/shader_cache")),
    debug_info: false,
    optimization_level: OptimizationLevel::Full,
    compile_options: CompileOptions {
        language_version: LanguageVersion::Metal3_0,
        fast_math: true,
        defines: [("CUSTOM_DEFINE", "1")].into(),
        ..Default::default()
    },
};
```

## Error Handling

Common compilation errors and solutions:

### Shader Not Found
```
Error: Shader 'bitlinear' not found
```
**Solution**: Ensure shader files are in the correct directory and have `.metal` extension.

### Function Not Found
```
Error: Function 'bitlinear_forward' not found in shader 'bitlinear'
```
**Solution**: Check function name spelling and ensure it's marked as `kernel`.

### Compilation Failed
```
Error: Failed to compile shader 'bitlinear': syntax error
```
**Solution**: Check Metal syntax and ensure all includes are available.

### Pipeline Creation Failed
```
Error: Failed to create pipeline for function 'bitlinear_forward'
```
**Solution**: Verify function signature and ensure device supports required features.

## Development Workflow

### Adding New Shaders
1. Create `.metal` file in this directory
2. Implement kernel functions with proper signatures
3. Add function enum to `BitNetShaderFunction`
4. Update shader mapping in `shader_utils.rs`
5. Add tests and benchmarks

### Testing Shaders
```rust
#[test]
fn test_new_shader() {
    let device = create_metal_device().unwrap();
    let shaders = BitNetShaders::new(device).unwrap();
    let pipeline = shaders.get_pipeline(BitNetShaderFunction::NewFunction).unwrap();
    // Test pipeline...
}
```

### Debugging Shaders
1. Enable Metal validation layers
2. Use printf debugging in shaders (Metal 2.3+)
3. Check GPU frame capture in Xcode
4. Verify buffer contents with CPU readback

## Platform Support

- **macOS**: Full support with Metal framework
- **iOS**: Compatible (with appropriate Metal feature set)
- **Other Platforms**: Graceful fallback with error messages

## Dependencies

- **metal-rs**: Rust bindings for Metal framework
- **anyhow**: Error handling
- **std::collections**: HashMap for caching

## License

This code is part of the BitNet Rust implementation and follows the same license terms.