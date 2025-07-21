# BitNet Core Tests

This directory contains comprehensive tests for the BitNet Core library, including availability tests for various acceleration backends.

## Test Files

### MLX Availability Tests (`mlx_availability_tests.rs`)

Comprehensive tests for MLX (Apple's Machine Learning framework) availability and functionality:

- **Basic Availability**: Tests MLX framework detection on different platforms
- **Device Management**: Tests MLX device discovery, enumeration, and selection
- **Tensor Operations**: Tests MLX tensor creation, manipulation, and operations
- **BitNet Integration**: Tests MLX-specific BitNet operations (quantization, BitLinear)
- **Interoperability**: Tests conversion between MLX arrays and Candle tensors
- **Feature Flags**: Tests conditional compilation behavior with/without MLX feature
- **Platform Behavior**: Tests platform-specific MLX behavior (Apple Silicon vs others)
- **Error Handling**: Tests error scenarios and recovery mechanisms
- **Performance**: Benchmarks MLX operations across different data sizes
- **Integration**: Comprehensive tests combining MLX with existing BitNet systems

### Metal Device Availability Tests (`metal_device_availability_tests.rs`)

Comprehensive tests for Metal device availability and functionality:

- **Device Detection**: Tests Metal device discovery and enumeration
- **Context Initialization**: Tests Metal context setup and teardown
- **Buffer Operations**: Tests Metal buffer creation, management, and data transfer
- **Shader Compilation**: Tests Metal shader compilation and pipeline creation
- **Performance**: Benchmarks Metal operations and memory management
- **Error Handling**: Tests Metal error scenarios and recovery

### Other Test Files

- **Integration Tests**: Cross-component integration testing
- **Memory Tests**: Memory management and tensor lifecycle testing
- **Device Comparison Tests**: Comparative testing across different backends
- **Shader Compilation Tests**: Metal shader compilation and validation
- **Tensor Operation Tests**: Core tensor operation testing
- **Tokenizer Loading Tests**: Comprehensive tokenizer loading and functionality testing

### Tokenizer Loading Tests (`tokenizer_loading_tests.rs`)

Comprehensive tests for tokenizer loading, creation, and functionality:

- **Simple Tokenizer Creation**: Tests programmatic creation of simple word-based tokenizers
- **BPE Tokenizer Loading**: Tests loading BPE tokenizers from vocabulary and merges files
- **HuggingFace Tokenizer Loading**: Tests loading HuggingFace tokenizers (requires `tokenizers` feature)
- **File Loading**: Tests loading tokenizers from various file formats (JSON, etc.)
- **Error Handling**: Tests error scenarios for invalid files, missing files, and unsupported formats
- **Special Tokens**: Tests special token functionality ([CLS], [SEP], [PAD], [MASK], etc.)
- **Batch Operations**: Tests batch encoding and decoding of multiple texts
- **Round-trip Testing**: Tests encoding/decoding consistency
- **Unicode Support**: Tests handling of Unicode text and emojis
- **Edge Cases**: Tests empty vocabularies, large vocabularies, whitespace handling
- **Feature Flag Behavior**: Tests conditional compilation with/without tokenizers feature

### Encoding/Decoding Correctness Tests (`encoding_decoding_correctness_tests.rs`)

Comprehensive correctness tests for all encoding/decoding functionality in BitNet Core:

- **Tokenizer Correctness**: Round-trip encoding/decoding validation for all tokenizer types
- **Batch Processing**: Consistency verification between individual and batch operations
- **Unicode Handling**: Full Unicode support testing including emojis and non-Latin scripts
- **Special Token Management**: Correct handling of control tokens and their isolation
- **Edge Case Robustness**: Comprehensive testing of boundary conditions and error scenarios
- **Quantization Correctness**: Data type validation and precision relationship testing
- **Memory Efficiency**: Quantization compression ratio and memory usage validation
- **MLX Integration**: Hardware-accelerated quantization correctness (Apple Silicon)
- **End-to-End Pipelines**: Complete workflow integrity testing
- **Error Handling**: Graceful failure mode validation across all operations

## Running Tests

### All Tests
```bash
cargo test --workspace
```

### MLX Tests (requires Apple Silicon + MLX)
```bash
cargo test --package bitnet-core --test mlx_availability_tests --features mlx
```

### Metal Tests (requires macOS + Metal)
```bash
cargo test --package bitnet-core --test metal_device_availability_tests --features metal
```

### Tokenizer Tests
```bash
# Test without tokenizers feature (basic functionality)
cargo test --package bitnet-core --test tokenizer_loading_tests

# Test with tokenizers feature (includes HuggingFace tokenizer support)
cargo test --package bitnet-core --test tokenizer_loading_tests --features tokenizers

# Test encoding/decoding correctness
cargo test --package bitnet-core --test encoding_decoding_correctness_tests

# Test with MLX support (Apple Silicon)
cargo test --package bitnet-core --test encoding_decoding_correctness_tests --features mlx
```

### Feature-Specific Tests
```bash
# Test without any acceleration features
cargo test --workspace --no-default-features

# Test with specific features
cargo test --workspace --features "mlx,metal"
cargo test --workspace --features "apple-silicon"
```

## Test Categories

### Availability Tests
- Platform detection and capability checking
- Framework availability verification
- Device enumeration and selection
- Feature flag behavior validation

### Functionality Tests
- Core operation testing
- Memory management validation
- Error handling verification
- Performance benchmarking

### Integration Tests
- Cross-component interaction testing
- End-to-end workflow validation
- Interoperability testing between frameworks
- Real-world usage scenario testing

## Platform Requirements

### MLX Tests
- **Platform**: Apple Silicon (M1/M2/M3) with macOS
- **Dependencies**: MLX framework installation
- **Features**: `mlx` feature flag enabled
- **Build Tools**: Xcode with Metal development tools

### Metal Tests
- **Platform**: macOS with Metal support
- **Dependencies**: Metal framework (system provided)
- **Features**: `metal` feature flag enabled
- **Hardware**: Metal-compatible GPU

### CPU Tests
- **Platform**: Any supported platform
- **Dependencies**: None (CPU-only operations)
- **Features**: No special features required

## Test Output

Tests provide detailed output including:
- Platform and capability detection results
- Performance metrics and benchmarks
- Error scenarios and recovery validation
- Device enumeration and selection results
- Memory usage and optimization metrics

## Continuous Integration

Tests are designed to work in CI environments:
- Graceful degradation when hardware/software not available
- Platform-specific test execution
- Feature flag conditional compilation
- Comprehensive error reporting

## Contributing

When adding new tests:
1. Follow existing test patterns and naming conventions
2. Include both positive and negative test cases
3. Add appropriate feature flag guards for platform-specific code
4. Include performance benchmarks where relevant
5. Document test requirements and expected behavior
6. Ensure tests work across different platforms and configurations