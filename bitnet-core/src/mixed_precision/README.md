# Mixed Precision Support for BitNet

This module provides comprehensive mixed precision support for BitNet models, allowing different layers and operations to use different precision levels for optimal performance and memory usage.

## Overview

Mixed precision is a technique that uses different numerical precisions for different parts of a neural network to optimize memory usage, computational speed, and energy efficiency while maintaining acceptable accuracy. This implementation provides:

- **Layer-specific precision configuration**: Different layers can use different precisions
- **Component-specific precision**: Weights, biases, activations, and gradients can have different precisions
- **Automatic precision selection**: Policy-based and strategy-based precision selection
- **Precision conversion utilities**: Efficient conversion between different precision levels
- **Validation and optimization**: Tools to validate configurations and optimize precision choices

## Architecture

The mixed precision system consists of several key components:

### Core Components

1. **Configuration System** ([`config.rs`](config.rs))
   - `MixedPrecisionConfig`: Main configuration for mixed precision operations
   - `LayerPrecisionConfig`: Layer-specific precision settings
   - `ComponentPrecisionConfig`: Component-specific precision settings

2. **Layer Precision Management** ([`layer_precision.rs`](layer_precision.rs))
   - `LayerPrecisionManager`: Manages precision specifications for all layers
   - `LayerPrecisionSpec`: Defines precision requirements for a specific layer

3. **Precision Conversion** ([`conversion.rs`](conversion.rs))
   - `PrecisionConverter`: Converts tensors between different precision levels
   - `ConversionStrategy`: Different strategies for precision conversion

4. **Precision Manager** ([`precision_manager.rs`](precision_manager.rs))
   - `PrecisionManager`: Central coordinator for all mixed precision operations
   - `PrecisionContext`: Context information for precision decisions

5. **Validation System** ([`validation.rs`](validation.rs))
   - `PrecisionValidator`: Validates precision configurations
   - `ValidationRule`: Configurable validation rules

6. **Policy Engine** ([`policy.rs`](policy.rs))
   - `PolicyEngine`: Manages precision selection policies
   - `PrecisionPolicy`: Defines rules for automatic precision selection

## Supported Precision Types

The system supports all BitNet data types defined in [`BitNetDType`](../memory/tensor/dtype.rs):

- **Float Types**: F32, F16, BF16
- **Integer Types**: I8, I4, I2, I1
- **Special Types**: BitNet158 (ternary: -1, 0, +1)

## Usage Examples

### Basic Configuration

```rust
use bitnet_core::mixed_precision::*;

// Create a balanced mixed precision configuration
let config = MixedPrecisionConfig::balanced();

// Create a custom configuration
let custom_config = MixedPrecisionConfig::new(MixedPrecisionStrategy::Custom)
    .with_layer_config(
        "attention_layer".to_string(),
        LayerPrecisionConfig::new(LayerType::Attention, BitNetDType::F16)
            .with_component_override(ComponentType::Weights, BitNetDType::I8)
    );
```

### Layer-Specific Precision Management

```rust
use bitnet_core::mixed_precision::*;

let layer_manager = LayerPrecisionManager::new();

// Register a layer with specific precision requirements
let layer_spec = LayerPrecisionSpec::new(
    "transformer_layer_0".to_string(),
    LayerType::Linear,
    BitNetDType::I8,  // input precision
    BitNetDType::I8,  // output precision
    BitNetDType::BitNet158,  // weight precision
)
.with_component_precision(ComponentType::Bias, BitNetDType::F16)
.with_dynamic_adjustment();

layer_manager.register_layer(layer_spec)?;

// Optimize for memory efficiency
let optimizations = layer_manager.optimize_for_memory(0.5)?;
layer_manager.apply_optimizations(optimizations)?;
```

### Precision Conversion

```rust
use bitnet_core::mixed_precision::*;

let config = ConversionConfig {
    strategy: ConversionStrategy::Scaled,
    preserve_metadata: true,
    validate_results: true,
    ..Default::default()
};

let mut converter = PrecisionConverter::new(config)?;

// Convert tensor to different precision
let converted_tensor = converter.convert_tensor(&tensor, BitNetDType::I8)?;
```

### Policy-Based Precision Selection

```rust
use bitnet_core::mixed_precision::*;

let mut policy_engine = PolicyEngine::new();

// Create a custom policy
let memory_policy = PrecisionPolicy::new(
    "memory_critical".to_string(),
    "Memory Critical Policy".to_string(),
    "Use aggressive quantization when memory is limited".to_string(),
)
.add_rule(
    PolicyRule::new(
        "high_memory_usage".to_string(),
        PolicyAction::SetPrecision(BitNetDType::I4),
    )
    .add_condition(PolicyCondition::new(
        ConditionType::MemoryUsage,
        ConditionOperator::GreaterThan,
        ConditionValue::Float(80.0),
    ))
);

policy_engine.add_policy(memory_policy);
```

### Complete Mixed Precision Setup

```rust
use bitnet_core::mixed_precision::*;

// 1. Create configuration
let config = MixedPrecisionConfig::balanced()
    .with_layer_config(
        "attention".to_string(),
        LayerPrecisionConfig::new(LayerType::Attention, BitNetDType::F16)
    );

// 2. Create precision manager
let precision_manager = PrecisionManager::new(config)?;

// 3. Register layers
let layer_spec = LayerPrecisionSpec::new(
    "linear_0".to_string(),
    LayerType::Linear,
    BitNetDType::I8,
    BitNetDType::I8,
    BitNetDType::BitNet158,
);
precision_manager.register_layer(layer_spec)?;

// 4. Use for tensor operations
let optimal_precision = precision_manager.get_optimal_precision(
    "linear_0",
    ComponentType::Weights,
    &tensor,
)?;

let converted_tensor = precision_manager.convert_for_operation(
    &tensor,
    "linear_0",
    ComponentType::Weights,
)?;
```

## Configuration Strategies

### Conservative Strategy
- Prioritizes accuracy over efficiency
- Uses higher precision for critical components
- Suitable for accuracy-critical applications

```rust
let config = MixedPrecisionConfig::conservative();
```

### Balanced Strategy
- Balances accuracy, memory, and performance
- Good default for most applications
- Uses BitNet 1.58 for weights, I8 for activations

```rust
let config = MixedPrecisionConfig::balanced();
```

### Aggressive Strategy
- Prioritizes memory and speed over accuracy
- Uses lowest possible precisions
- Suitable for resource-constrained environments

```rust
let config = MixedPrecisionConfig::aggressive();
```

## Layer Types and Precision Support

Different layer types have different precision requirements and support:

### Linear/Convolution Layers
- **Supported Precisions**: All quantized precisions (BitNet158, I8, I4, I2, I1) + F16, F32
- **Recommended**: BitNet158 for weights, I8 for activations
- **Memory Efficiency**: High (up to 32x compression)

### Attention Layers
- **Supported Precisions**: F32, F16, BF16, I8
- **Recommended**: F16 for stability, I8 for efficiency
- **Special Considerations**: Attention scores may need higher precision

### Embedding Layers
- **Supported Precisions**: F32, F16, BF16, I8
- **Recommended**: F16 for good balance
- **Memory Impact**: High due to large vocabulary sizes

### Normalization Layers
- **Supported Precisions**: F32, F16, BF16
- **Recommended**: F16 minimum for numerical stability
- **Critical**: Avoid integer precisions

### Output Layers
- **Supported Precisions**: F32, F16, BF16, I8
- **Recommended**: F16 for final accuracy
- **Impact**: Directly affects model output quality

## Conversion Strategies

### Direct Conversion
- Simple dtype conversion
- Fastest but may lose precision
- Use for compatible precision changes

### Scaled Conversion
- Applies optimal scaling before conversion
- Minimizes precision loss
- Good default strategy

### Quantization-Aware Conversion
- Uses quantization pipeline for conversion
- Best for float-to-quantized conversions
- Preserves quantization semantics

### Stochastic Rounding
- Uses probabilistic rounding
- Better precision preservation
- Slower but higher quality

## Validation and Optimization

### Validation Rules
The system includes comprehensive validation:

- **Layer-Precision Compatibility**: Ensures layer types support assigned precisions
- **Component-Precision Compatibility**: Validates component-specific precisions
- **Memory Usage Checks**: Warns about excessive memory usage
- **Performance Impact Assessment**: Evaluates conversion overhead
- **Accuracy Impact Analysis**: Identifies potential accuracy issues

### Optimization Objectives

#### Memory Optimization
```rust
precision_manager.optimize_precision(
    OptimizationObjective::Memory { target_reduction: 0.5 }
)?;
```

#### Speed Optimization
```rust
precision_manager.optimize_precision(
    OptimizationObjective::Speed { target_speedup: 2.0 }
)?;
```

#### Accuracy Preservation
```rust
precision_manager.optimize_precision(
    OptimizationObjective::Accuracy { min_accuracy: 0.95 }
)?;
```

#### Balanced Optimization
```rust
precision_manager.optimize_precision(
    OptimizationObjective::Balanced {
        memory_weight: 0.4,
        speed_weight: 0.3,
        accuracy_weight: 0.3,
    }
)?;
```

## Performance Considerations

### Memory Efficiency
- **BitNet158**: 32x compression vs F32
- **I4**: 8x compression vs F32
- **I8**: 4x compression vs F32
- **F16**: 2x compression vs F32

### Conversion Overhead
- Minimize precision changes between layers
- Use batch conversion for multiple tensors
- Cache converted tensors when possible

### Hardware Compatibility
- Check device support for specific precisions
- Use SIMD optimizations when available
- Consider hardware acceleration capabilities

## Best Practices

### 1. Start with Balanced Strategy
Begin with the balanced configuration and adjust based on specific requirements.

### 2. Profile Before Optimizing
Measure actual memory usage and performance before applying aggressive optimizations.

### 3. Validate Configurations
Always validate configurations before deployment:

```rust
config.validate()?;
precision_manager.validate_configuration()?;
```

### 4. Monitor Accuracy Impact
Track accuracy changes when applying mixed precision:

```rust
let analysis = precision_manager.analyze_configuration()?;
println!("Accuracy impact: {:.2}%", analysis.accuracy_impact * 100.0);
```

### 5. Use Layer-Specific Configurations
Configure critical layers (attention, output) with higher precision:

```rust
let config = MixedPrecisionConfig::balanced()
    .with_layer_config(
        "output_layer".to_string(),
        LayerPrecisionConfig::new(LayerType::Output, BitNetDType::F16)
    );
```

### 6. Implement Gradual Precision Reduction
Start with higher precisions and gradually reduce based on validation results.

### 7. Consider Training vs Inference
Use different precision configurations for training and inference phases.

## Error Handling

The mixed precision system provides comprehensive error handling:

```rust
use bitnet_core::mixed_precision::MixedPrecisionError;

match precision_manager.convert_for_operation(&tensor, "layer_0", ComponentType::Weights) {
    Ok(converted) => {
        // Use converted tensor
    }
    Err(MixedPrecisionError::ConversionError { from, to, reason }) => {
        eprintln!("Failed to convert {} to {}: {}", from, to, reason);
    }
    Err(MixedPrecisionError::ValidationError(msg)) => {
        eprintln!("Validation failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Mixed precision error: {}", e);
    }
}
```

## Integration with Quantization

The mixed precision system integrates seamlessly with the BitNet quantization pipeline:

```rust
use bitnet_quant::quantization::mixed_precision::*;

let config = MixedPrecisionQuantizationConfig::bitnet();
let mut quantizer = MixedPrecisionQuantizer::new(config, device)?;

// Register layers
quantizer.register_layer(layer_spec)?;

// Quantize with mixed precision
let quantized_weights = quantizer.quantize_weights(&weights, "layer_0")?;
let quantized_activations = quantizer.quantize_activations(&activations, "layer_0")?;
```

## Future Enhancements

Planned improvements include:

1. **Dynamic Precision Adjustment**: Runtime precision adjustment based on performance metrics
2. **Hardware-Specific Optimization**: Automatic precision selection based on hardware capabilities
3. **Advanced Conversion Strategies**: More sophisticated conversion algorithms
4. **Integration with Training**: Mixed precision training support
5. **Profiling Tools**: Built-in profiling and analysis tools

## Contributing

When contributing to the mixed precision system:

1. Add comprehensive tests for new features
2. Update documentation for API changes
3. Validate performance impact of changes
4. Ensure backward compatibility
5. Follow the existing code style and patterns

## See Also

- [BitNet Tensor Documentation](../memory/tensor/README.md)
- [Quantization Documentation](../../bitnet-quant/README.md)
- [Memory Management Documentation](../memory/README.md)
- [Examples](../../examples/mixed_precision_demo.rs)