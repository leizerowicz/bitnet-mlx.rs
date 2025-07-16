# BitNet Quantization

[![Crates.io](https://img.shields.io/crates/v/bitnet-quant.svg)](https://crates.io/crates/bitnet-quant)
[![Documentation](https://docs.rs/bitnet-quant/badge.svg)](https://docs.rs/bitnet-quant)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

The quantization engine for BitNet neural networks, implementing 1.58-bit quantization algorithms and calibration utilities optimized for extreme compression while maintaining model accuracy.

## 🎯 Purpose

`bitnet-quant` provides the core quantization functionality for BitNet models:

- **1.58-bit Quantization**: Implementation of the novel 1.58-bit quantization scheme
- **Weight Quantization**: Efficient algorithms for quantizing neural network weights
- **Activation Quantization**: Runtime quantization of activations and intermediate values
- **Calibration Utilities**: Tools for determining optimal quantization parameters
- **Dequantization**: Fast dequantization for computation and inference

## 🔴 Current Status: **PLACEHOLDER ONLY**

⚠️ **This crate is currently a placeholder and contains no implementation.**

The current `src/lib.rs` contains only:
```rust
//! BitNet Quantization Library
//! 
//! This crate provides quantization utilities for BitNet models.

// Placeholder for future quantization implementation
```

## ✅ What Needs to be Implemented

### 🔴 **Core Quantization Engine** (Not Implemented)

#### 1.58-bit Quantization Algorithm
- **Weight Quantization**: Convert FP32/FP16 weights to 1.58-bit representation
- **Quantization Functions**: Implement the core quantization mathematical operations
- **Scaling Factors**: Compute and manage per-tensor or per-channel scaling factors
- **Rounding Strategies**: Implement different rounding approaches for optimal accuracy

#### Activation Quantization
- **Runtime Quantization**: Quantize activations during forward pass
- **Dynamic Range**: Automatic detection of activation ranges
- **Calibration-based**: Use calibration data to determine quantization parameters
- **Adaptive Quantization**: Adjust quantization based on input characteristics

#### Dequantization Engine
- **Fast Dequantization**: Efficient conversion back to floating-point for computation
- **SIMD Optimizations**: Vectorized dequantization operations
- **Batch Processing**: Efficient handling of batched dequantization
- **Memory-Efficient**: Minimize memory overhead during dequantization

### 🔴 **Calibration System** (Not Implemented)

#### Calibration Data Management
- **Dataset Integration**: Interface with calibration datasets
- **Statistical Analysis**: Compute activation statistics for quantization
- **Range Estimation**: Determine optimal quantization ranges
- **Outlier Handling**: Robust handling of activation outliers

#### Quantization Parameter Optimization
- **Grid Search**: Systematic search for optimal quantization parameters
- **Gradient-based**: Use gradients to optimize quantization parameters
- **KL-Divergence**: Minimize KL divergence between original and quantized distributions
- **Custom Metrics**: Support for domain-specific optimization metrics

### 🔴 **Advanced Quantization Methods** (Not Implemented)

#### Per-Channel Quantization
- **Channel-wise Scaling**: Individual scaling factors per output channel
- **Mixed Precision**: Different quantization levels for different layers
- **Sensitivity Analysis**: Determine which layers are most sensitive to quantization
- **Adaptive Bit-width**: Dynamic bit-width selection based on layer importance

#### Quantization-Aware Training Support
- **Fake Quantization**: Simulate quantization during training
- **Gradient Estimation**: Straight-through estimator for quantization gradients
- **Noise Injection**: Add quantization noise during training
- **Progressive Quantization**: Gradually increase quantization during training

## 🚀 Planned API Design

### Basic Quantization

```rust
use bitnet_quant::{Quantizer, QuantizationConfig, BitNetQuantizer};
use bitnet_core::{Tensor, Device};

// Create quantizer with 1.58-bit configuration
let config = QuantizationConfig {
    bits: 1.58,
    symmetric: true,
    per_channel: false,
    calibration_method: CalibrationMethod::MinMax,
};

let quantizer = BitNetQuantizer::new(config);

// Quantize weights
let weights = Tensor::randn(&[128, 256], &Device::Cpu)?;
let (quantized_weights, scale, zero_point) = quantizer.quantize_weights(&weights)?;

// Quantize activations
let activations = Tensor::randn(&[32, 128], &Device::Cpu)?;
let quantized_activations = quantizer.quantize_activations(&activations, scale, zero_point)?;

// Dequantize for computation
let dequantized = quantizer.dequantize(&quantized_weights, scale, zero_point)?;
```

### Calibration

```rust
use bitnet_quant::{Calibrator, CalibrationDataset};

// Create calibrator
let mut calibrator = Calibrator::new();

// Add calibration data
let calibration_data = CalibrationDataset::from_tensors(vec![
    Tensor::randn(&[32, 128], &Device::Cpu)?,
    Tensor::randn(&[32, 128], &Device::Cpu)?,
    // ... more calibration samples
]);

// Calibrate quantization parameters
let params = calibrator.calibrate(&calibration_data)?;

// Use calibrated parameters
let quantizer = BitNetQuantizer::with_params(params);
```

### Advanced Quantization

```rust
use bitnet_quant::{PerChannelQuantizer, MixedPrecisionConfig};

// Per-channel quantization
let per_channel_config = QuantizationConfig {
    bits: 1.58,
    per_channel: true,
    channel_axis: 0,
    ..Default::default()
};

let quantizer = PerChannelQuantizer::new(per_channel_config);

// Mixed precision quantization
let mixed_precision = MixedPrecisionConfig::builder()
    .layer_config("conv1", QuantizationConfig { bits: 2.0, ..Default::default() })
    .layer_config("conv2", QuantizationConfig { bits: 1.58, ..Default::default() })
    .layer_config("fc", QuantizationConfig { bits: 4.0, ..Default::default() })
    .build();

let quantizer = MixedPrecisionQuantizer::new(mixed_precision);
```

## 🏗️ Planned Architecture

### Core Components

```
bitnet-quant/src/
├── lib.rs                    # Main library interface
├── quantizer/               # Core quantization algorithms
│   ├── mod.rs              # Quantizer trait and common types
│   ├── bitnet.rs           # 1.58-bit BitNet quantizer
│   ├── uniform.rs          # Uniform quantization
│   ├── per_channel.rs      # Per-channel quantization
│   └── mixed_precision.rs  # Mixed precision quantization
├── calibration/            # Calibration system
│   ├── mod.rs             # Calibration interface
│   ├── dataset.rs         # Calibration dataset management
│   ├── statistics.rs      # Statistical analysis
│   ├── optimizer.rs       # Parameter optimization
│   └── methods.rs         # Calibration methods (MinMax, KL, etc.)
├── algorithms/             # Quantization algorithms
│   ├── mod.rs             # Algorithm interface
│   ├── rounding.rs        # Rounding strategies
│   ├── scaling.rs         # Scaling factor computation
│   ├── range.rs           # Range estimation
│   └── noise.rs           # Quantization noise modeling
├── ops/                   # Quantized operations
│   ├── mod.rs             # Operation interface
│   ├── linear.rs          # Quantized linear operations
│   ├── conv.rs            # Quantized convolution
│   ├── activation.rs      # Quantized activations
│   └── utils.rs           # Utility operations
└── utils/                 # Utilities and helpers
    ├── mod.rs             # Utility interface
    ├── metrics.rs         # Quantization quality metrics
    ├── analysis.rs        # Quantization analysis tools
    └── validation.rs      # Validation and testing utilities
```

### Integration with BitNet Core

```rust
use bitnet_core::memory::{HybridMemoryPool, BitNetTensor};
use bitnet_quant::{BitNetQuantizer, QuantizationConfig};

// Integrate with memory management
let pool = HybridMemoryPool::new()?;
let quantizer = BitNetQuantizer::new(QuantizationConfig::default());

// Quantize tensor using memory pool
let tensor = BitNetTensor::randn(&[128, 256], &pool)?;
let quantized = quantizer.quantize_tensor(&tensor, &pool)?;
```

## 📊 Expected Performance Characteristics

### Quantization Performance (Projected)

| Operation | Throughput | Memory Reduction | Accuracy Loss |
|-----------|------------|------------------|---------------|
| **Weight Quantization** | >1GB/s | 10.67x (FP32→1.58bit) | <2% |
| **Activation Quantization** | >500MB/s | 10.67x | <1% |
| **Dequantization** | >2GB/s | N/A | 0% |

### Memory Efficiency

| Data Type | Bits per Weight | Memory Usage (1M params) |
|-----------|----------------|--------------------------|
| **FP32** | 32 | 4.0 MB |
| **FP16** | 16 | 2.0 MB |
| **INT8** | 8 | 1.0 MB |
| **BitNet 1.58** | 1.58 | 0.375 MB |

## 🧪 Planned Testing Strategy

### Unit Tests
```bash
# Test core quantization algorithms
cargo test --package bitnet-quant quantizer

# Test calibration system
cargo test --package bitnet-quant calibration

# Test quantized operations
cargo test --package bitnet-quant ops
```

### Integration Tests
```bash
# Test with real models
cargo test --package bitnet-quant --test model_quantization

# Test accuracy preservation
cargo test --package bitnet-quant --test accuracy_tests

# Test performance benchmarks
cargo bench --package bitnet-quant
```

### Accuracy Validation
```bash
# Compare quantized vs original model accuracy
cargo test --package bitnet-quant --test accuracy_validation

# Test on standard datasets
cargo test --package bitnet-quant --test dataset_validation
```

## 🔬 Research Implementation

### BitNet 1.58-bit Quantization

The core innovation of BitNet is the 1.58-bit quantization scheme:

```
Quantization levels: {-1, 0, +1}
Effective bits per weight: log₂(3) ≈ 1.58 bits
```

**Mathematical Foundation:**
- Weights are quantized to three discrete levels
- Scaling factors maintain numerical range
- Activation quantization uses similar principles
- Dequantization reconstructs approximate original values

### Implementation Priorities

1. **Phase 1**: Basic 1.58-bit weight quantization
2. **Phase 2**: Activation quantization and calibration
3. **Phase 3**: Advanced methods (per-channel, mixed precision)
4. **Phase 4**: Quantization-aware training support

## 🤝 Contributing

This crate needs complete implementation! Priority areas:

1. **Core Quantization**: Implement the 1.58-bit quantization algorithm
2. **Calibration System**: Build calibration data management and optimization
3. **Performance**: Optimize quantization/dequantization operations
4. **Accuracy**: Ensure minimal accuracy loss from quantization

### Getting Started

1. Study the BitNet paper: [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
2. Implement basic quantization functions
3. Add comprehensive tests
4. Optimize for performance
5. Integrate with `bitnet-core` memory management

## 📚 References

- **BitNet Paper**: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- **Quantization Survey**: [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
- **QAT Methods**: [Quantization Aware Training](https://arxiv.org/abs/1712.05877)

## 📄 License

Licensed under the MIT License. See [LICENSE](../LICENSE) for details.