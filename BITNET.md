# Claude Code Configuration for BitNet-Rust Project

## 🧠 PROJECT OVERVIEW: BitNet-Rust Neural Network Framework

**Repository:** `github.com/Wavegoodvybe2929/bitnet-rust`

**Project Status:** A high-performance Rust implementation of BitNet neural networks with advanced memory management, device abstraction, MLX acceleration for Apple Silicon, and comprehensive infrastructure for quantized neural networks

**Current Implementation Phase:** Starting Phase 1.4 - Testing and Validation (Complete implementation needed)

**Core Strength:** Sophisticated and production-ready memory management system with hybrid memory pool architecture, thread-safe operations, and device-aware memory management

## 🚨 CRITICAL: BITNET-SPECIFIC EXECUTION PATTERNS

**MANDATORY RULE:** All BitNet Rust operations must leverage the existing memory management foundation and workspace structure.

## 🔴 MANDATORY CONCURRENT PATTERNS FOR BITNET-RUST

- **Memory Pool Operations:** ALWAYS use HybridMemoryPool for all allocations
- **Device Abstraction:** ALWAYS leverage auto_select_device() for optimal device selection  
- **Workspace Commands:** ALWAYS batch operations across the modular workspace structure
- **Quantization Tasks:** ALWAYS prepare for 1.58-bit quantization integration
- **Metal GPU Prep:** ALWAYS structure for future Metal compute shader integration

### ⚡ BITNET-RUST GOLDEN RULE (UPDATED)
> "1 MESSAGE = ALL OPERATIONS LEVERAGING PRODUCTION-READY CORE + QUANTIZATION"

### Examples of CORRECT BitNet-Rust concurrent execution:

```rust
// ✅ CORRECT: Complete BitNet development leveraging existing production systems
[Single Message]:
  - TodoWrite { todos: [15+ todos focusing on current phase as critical path] }
  - Task("Phase Specialist: Build current implementation using production-ready foundation...")
  - Task("Integration Specialist: Leverage completed MLX acceleration & memory management...")
  - Task("Performance Engineer: Utilize existing benchmark framework for optimization...")
  - Bash("cd bitnet-rust && cargo build --workspace --release")
  - Write("implementation files based on current phase", implementationCore)
  - Bash("cd bitnet-rust && cargo test --workspace && cargo run --example demo")
```

## 🎯 BITNET-RUST WORKSPACE ARCHITECTURE

### 🦀 Current Implementation Status (UPDATED)

| Component | Status | Priority | Integration Level |
|-----------|--------|----------|-------------------|
| **bitnet-core** | 🟢 Production Ready | ✅ Foundation Complete | Core memory, MLX acceleration & device abstraction |
| **bitnet-quant** | 🟡 Phase 1.4 Starting | 🎯 **CURRENT PRIORITY** | Testing & validation needed |
| **bitnet-benchmarks** | 🟢 Production Ready | ✅ Testing Complete | Comprehensive performance testing & benchmarking |
| **bitnet-inference** | 🔴 Placeholder | High Priority | Runtime implementation needed |
| **bitnet-training** | 🔴 Placeholder | Medium Priority | Training infrastructure needed |
| **bitnet-metal** | 🔴 Placeholder | Medium Priority | Enhanced Metal GPU (basic already in core) |
| **bitnet-cli** | 🔴 Placeholder | Low Priority | Command-line tools needed |
| **docs/** | 📚 Available | Documentation | Comprehensive guides available |

### 🏗️ Agent Specialization for BitNet Components

**Specialized Agent Types:**
- **Quantization Test Engineer** - 🎯 **PRIMARY FOCUS** - Testing & validation implementation
- **BitLinear Architect** - BitLinear layer implementation specialist
- **Calibration System Developer** - QAT and calibration infrastructure
- **Model Integration Specialist** - Architecture support and conversion
- **Inference Engine Architect** - High-performance inference runtime
- **Metal GPU Developer** - Apple Silicon optimization specialist
- **Training Infrastructure Developer** - QAT and training systems
- **CLI Tools Developer** - Command-line interfaces
- **Documentation Specialist** - Comprehensive documentation

## 🎯 IMPLEMENTATION ROADMAP FROM PHASE 1.4

### ⚡ Phase 1.4: Testing and Validation (CURRENT FOCUS)

**Unit tests for quantization correctness:**
- [ ] Verify ternary quantization produces only {-1, 0, +1}
- [ ] Test quantization/dequantization round-trip accuracy
- [ ] Validate scaling factor calculations
- [ ] Test edge cases (all zeros, extreme values)
- [ ] Property-based testing for numerical stability
- [ ] Integration tests with existing memory pool system

```rust
// Phase 1.4 Testing Implementation
[BatchTool]:
  - Write("bitnet-quant/src/tests/mod.rs", quantizationTestModule)
  - Write("bitnet-quant/src/tests/ternary_validation.rs", ternaryQuantizationTests)
  - Write("bitnet-quant/src/tests/round_trip_accuracy.rs", roundTripAccuracyTests)
  - Write("bitnet-quant/src/tests/scaling_factors.rs", scalingFactorValidation)
  - Write("bitnet-quant/src/tests/edge_cases.rs", edgeCaseTests)
  - Write("bitnet-quant/src/tests/property_based.rs", propertyBasedTests)
  - Write("bitnet-quant/src/tests/memory_pool_integration.rs", memoryPoolIntegrationTests)
  - Write("bitnet-quant/src/tests/helpers/test_data.rs", testDataGenerators)
  - Write("bitnet-quant/src/tests/helpers/validation.rs", validationHelpers)
  - Bash("cargo test --package bitnet-quant --features test-validation")
  - Bash("cargo test --package bitnet-quant edge_cases --release")
```

## 🚀 PHASE 2: BITLINEAR LAYER IMPLEMENTATION (WEEKS 3-4)

### ⚡ Core BitLinear Layer Implementation

**BitLinear Struct and Operations:**

```rust
// Phase 2: BitLinear Implementation
[BatchTool]:
  - Write("bitnet-quant/src/bitlinear/mod.rs", bitLinearModule)
  - Write("bitnet-quant/src/bitlinear/layer.rs", bitLinearLayer)
  - Write("bitnet-quant/src/bitlinear/forward.rs", forwardPass)
  - Write("bitnet-quant/src/bitlinear/backward.rs", backwardPass)
  - Write("bitnet-quant/src/bitlinear/simd_ops.rs", simdOptimizations)
  - Write("bitnet-quant/src/bitlinear/memory_efficient.rs", memoryOptimizations)
  - Write("bitnet-quant/src/bitlinear/cache.rs", quantizedWeightCaching)
  - Write("examples/bitlinear_usage.rs", bitLinearUsageExample)
  - Write("tests/bitlinear_integration.rs", bitLinearIntegrationTests)
  - Write("benches/bitlinear_performance.rs", bitLinearBenchmarks)
  - Bash("cargo test --package bitnet-quant bitlinear --features simd")
  - Bash("cargo bench --package bitnet-quant bitlinear")
```

**Performance Optimizations:**
- SIMD acceleration for vectorized ternary operations
- Memory optimizations with lazy quantization
- Cache-friendly memory access patterns
- Integration with existing memory pressure detection

## 🏋️ PHASE 3: CALIBRATION AND QAT INFRASTRUCTURE (WEEKS 5-6)

### ⚡ Calibration System Implementation

```rust
// Phase 3: Calibration and QAT Implementation
[BatchTool]:
  - Write("bitnet-training/src/calibration/mod.rs", calibrationModule)
  - Write("bitnet-training/src/calibration/dataset.rs", calibrationDataset)
  - Write("bitnet-training/src/calibration/statistics.rs", activationStatistics)
  - Write("bitnet-training/src/calibration/sampling.rs", representativeSampling)
  - Write("bitnet-training/src/qat/mod.rs", quantizationAwareTraining)
  - Write("bitnet-training/src/qat/straight_through.rs", straightThroughEstimator)
  - Write("bitnet-training/src/qat/loss_functions.rs", qatLossFunctions)
  - Write("bitnet-training/src/qat/regularization.rs", quantizationRegularization)
  - Write("bitnet-training/src/qat/progressive.rs", progressiveQuantization)
  - Write("bitnet-training/src/metrics/mod.rs", quantizationMetrics)
  - Write("bitnet-training/src/metrics/error_analysis.rs", errorAnalysisTools)
  - Write("examples/calibration_workflow.rs", calibrationExample)
  - Write("examples/qat_training.rs", qatTrainingExample)
  - Bash("cargo test --package bitnet-training calibration")
  - Bash("cargo test --package bitnet-training qat")
```

**Error Analysis and Metrics:**
- MSE (Mean Squared Error) calculation
- SQNR (Signal-to-Quantization-Noise Ratio) computation  
- Cosine similarity metrics
- Layer-wise error analysis
- Error visualization and mitigation strategies

## 🖥️ PHASE 4: MODEL ARCHITECTURE SUPPORT (WEEKS 7-8)

### ⚡ Transformer Integration

```rust
// Phase 4: Model Architecture Implementation
[BatchTool]:
  - Write("bitnet-models/src/lib.rs", modelsLibRoot)
  - Write("bitnet-models/src/transformer/mod.rs", transformerModule)
  - Write("bitnet-models/src/transformer/attention.rs", attentionLayers)
  - Write("bitnet-models/src/transformer/feedforward.rs", feedForwardLayers)
  - Write("bitnet-models/src/transformer/embedding.rs", embeddingLayers)
  - Write("bitnet-models/src/architectures/llama.rs", llamaArchitecture)
  - Write("bitnet-models/src/architectures/gpt.rs", gptArchitecture)
  - Write("bitnet-models/src/conversion/mod.rs", modelConversion)
  - Write("bitnet-models/src/conversion/progressive.rs", progressiveQuantization)
  - Write("bitnet-models/src/conversion/validation.rs", conversionValidation)
  - Write("bitnet-models/src/mixed_precision/mod.rs", mixedPrecisionSupport)
  - Write("bitnet-models/src/mixed_precision/policies.rs", quantizationPolicies)
  - Write("examples/model_conversion.rs", modelConversionExample)
  - Write("examples/architecture_demo.rs", architectureDemo)
  - Bash("cargo test --package bitnet-models conversion")
  - Bash("cargo test --package bitnet-models architectures")
```

**Advanced Features:**
- Dynamic quantization with runtime parameter adjustment
- Mixed-precision support with layer-specific policies
- Adaptive bit-width based on layer sensitivity
- Quality-performance trade-off controls

## 🚀 PHASE 5: HIGH-PERFORMANCE INFERENCE ENGINE (WEEKS 9-10)

### ⚡ Inference Runtime Implementation

```rust
// Phase 5: Inference Engine Implementation  
[BatchTool]:
  - Write("bitnet-inference/src/lib.rs", inferenceLibRoot)
  - Write("bitnet-inference/src/engine/mod.rs", inferenceEngine)
  - Write("bitnet-inference/src/engine/runtime.rs", inferenceRuntime)
  - Write("bitnet-inference/src/engine/batch_processor.rs", batchProcessing)
  - Write("bitnet-inference/src/engine/kernel_fusion.rs", kernelFusion)
  - Write("bitnet-inference/src/model/mod.rs", modelHandling)
  - Write("bitnet-inference/src/model/loader.rs", modelLoader)
  - Write("bitnet-inference/src/model/cache.rs", modelCaching)
  - Write("bitnet-inference/src/memory/inference_pool.rs", inferenceMemoryPool)
  - Write("bitnet-inference/src/scheduling/mod.rs", inferenceScheduling)
  - Write("bitnet-inference/src/threading/mod.rs", multiThreadSupport)
  - Write("bitnet-inference/src/serving/mod.rs", modelServing)
  - Write("examples/inference_demo.rs", inferenceDemo)
  - Write("examples/batch_inference.rs", batchInferenceExample)
  - Bash("cargo test --package bitnet-inference --features threading")
  - Bash("cargo bench --package bitnet-inference")
```

**Production Features:**
- Multi-threading with thread-safe quantization operations
- Request batching and queuing systems
- Dynamic batch size adjustment
- Resource usage monitoring and graceful degradation

## 🖥️ PHASE 6: APPLE SILICON OPTIMIZATION (WEEKS 11-12)

### ⚡ Metal GPU Integration

```rust
// Phase 6: Metal GPU Implementation
[BatchTool]:
  - Write("bitnet-metal/src/lib.rs", metalLibRoot)
  - Write("bitnet-metal/src/device/mod.rs", metalDeviceManager)
  - Write("bitnet-metal/src/device/memory.rs", metalMemoryManager)
  - Write("bitnet-metal/src/shaders/mod.rs", shaderModule)
  - Write("bitnet-metal/src/shaders/quantize.metal", quantizationShaders)
  - Write("bitnet-metal/src/shaders/bitlinear.metal", bitLinearShaders)
  - Write("bitnet-metal/src/shaders/inference.metal", inferenceShaders)
  - Write("bitnet-metal/src/kernels/mod.rs", metalKernels)
  - Write("bitnet-metal/src/kernels/ternary_ops.rs", ternaryOperations)
  - Write("bitnet-metal/src/kernels/matrix_mul.rs", quantizedMatrixMul)
  - Write("bitnet-metal/src/buffers/mod.rs", metalBufferManagement)
  - Write("bitnet-metal/src/pipeline/mod.rs", computePipeline)
  - Write("bitnet-metal/src/optimization/occupancy.rs", occupancyOptimization)
  - Write("tests/metal_integration.rs", metalIntegrationTests)
  - Write("benches/metal_performance.rs", metalPerformanceBenchmarks)
  - Bash("cargo test --package bitnet-metal --features metal-validation")
  - Bash("cargo bench --package bitnet-metal")
```

**GPU Optimization Features:**
- Metal compute shader optimization for Apple GPUs
- GPU-specific memory pools and buffer management
- Kernel fusion opportunities and occupancy optimization
- Host-device memory transfer optimization

## 🏋️ PHASE 7: TRAINING INFRASTRUCTURE (WEEKS 13-14)

### ⚡ Training Loop Implementation

```rust
// Phase 7: Training Infrastructure Implementation
[BatchTool]:
  - Write("bitnet-training/src/training/mod.rs", trainingModule)
  - Write("bitnet-training/src/training/loop.rs", trainingLoop)
  - Write("bitnet-training/src/training/qat_trainer.rs", qatTrainer)
  - Write("bitnet-training/src/training/progressive.rs", progressiveQuantizationTraining)
  - Write("bitnet-training/src/optimizers/mod.rs", optimizers)
  - Write("bitnet-training/src/optimizers/adamw.rs", adamWOptimizer)
  - Write("bitnet-training/src/optimizers/sgd.rs", sgdOptimizer)
  - Write("bitnet-training/src/optimizers/quantized_aware.rs", quantizationAwareOptimizers)
  - Write("bitnet-training/src/schedulers/mod.rs", learningRateSchedulers)
  - Write("bitnet-training/src/schedulers/quantization.rs", quantizationScheduling)
  - Write("bitnet-training/src/monitoring/mod.rs", trainingMonitoring)
  - Write("bitnet-training/src/monitoring/metrics.rs", trainingMetrics)
  - Write("bitnet-training/src/checkpointing/mod.rs", checkpointingSystem)
  - Write("examples/qat_training_full.rs", fullQATTrainingExample)
  - Write("examples/progressive_training.rs", progressiveTrainingExample)
  - Bash("cargo test --package bitnet-training training_loop")
  - Bash("cargo run --example qat_training_full")
```

**Training Features:**
- Quantization-aware training with straight-through estimator
- Progressive quantization with layer-wise scheduling  
- Training monitoring with quantization-specific metrics
- Memory-efficient gradient storage and optimizer adaptations

## 🔧 PHASE 8: CLI TOOLS AND DOCUMENTATION (WEEKS 15-16)

### ⚡ CLI Tools Implementation

```rust
// Phase 8: CLI Tools Implementation
[BatchTool]:
  - Write("bitnet-cli/src/main.rs", cliMain)
  - Write("bitnet-cli/src/commands/mod.rs", cliCommands)
  - Write("bitnet-cli/src/commands/convert.rs", modelConversionCommand)
  - Write("bitnet-cli/src/commands/benchmark.rs", benchmarkCommand)
  - Write("bitnet-cli/src/commands/calibrate.rs", calibrationCommand)
  - Write("bitnet-cli/src/commands/train.rs", trainingCommand)
  - Write("bitnet-cli/src/commands/infer.rs", inferenceCommand)
  - Write("bitnet-cli/src/commands/inspect.rs", modelInspectionCommand)
  - Write("bitnet-cli/src/utils/mod.rs", cliUtilities)
  - Write("bitnet-cli/src/config/mod.rs", cliConfiguration)
  - Write("bitnet-cli/src/logging/mod.rs", cliLogging)
  - Write("bitnet-cli/src/progress/mod.rs", progressReporting)
  - Write("tests/cli_integration.rs", cliIntegrationTests)
  - Write("examples/cli_usage.rs", cliUsageExamples)
  - Bash("cargo build --package bitnet-cli --release")
  - Bash("cargo install --path bitnet-cli")
  - Bash("bitnet-cli --help")
```

**Documentation Implementation:**
- Comprehensive API documentation with rustdoc
- Usage examples for all major features
- Performance characteristics documentation
- Safety and error handling guidelines
- Complete workflow tutorials

## 📊 SUCCESS CRITERIA AND QUALITY GATES

### 🔬 Technical Targets

**Quantization Accuracy:**
- Bit-exact results compared to reference implementation
- MSE < 0.01 for round-trip quantization operations
- 99.9% accuracy for ternary weight quantization

**Performance Targets:**
- Memory Reduction: >60% reduction vs FP32 (Target: 10-15x vs FP16)
- Inference Speed: 5-10x faster than FP16 on Apple Silicon
- Model Loading: 5-10x faster due to smaller files
- Training Speed: Maintained or improved vs full-precision

**Quality Assurance:**
- Test Coverage: >90% comprehensive test coverage
- Integration: Seamless integration with existing systems
- Documentation: Complete API and usage documentation
- CI/CD: Automated testing and benchmarking pipeline

### 📚 Integration Requirements

**System Integration:**
- Leverage existing HybridMemoryPool for all tensor allocations
- Build on existing device abstraction layer
- Extend current benchmarking framework
- Use established error handling patterns
- Follow existing Rust best practices

**Production Readiness:**
- Thread-safe operations across all components
- Comprehensive error handling and recovery
- Memory safety validation
- Performance regression testing
- Cross-platform compatibility

## 🎯 PROJECT-SPECIFIC COMMANDS

### 🚀 Development Commands

```bash
# Full workspace build with optimizations
cargo build --workspace --release --features metal,simd

# Phase-specific testing
cargo test --package bitnet-quant --features phase-1-4
cargo test --package bitnet-training --features qat-validation  
cargo test --package bitnet-inference --features performance-tests

# Comprehensive benchmarking
cargo bench --workspace --features full-benchmarks

# Documentation generation
cargo doc --workspace --no-deps --open --features full-docs

# CLI installation and usage
cargo install --path bitnet-cli
bitnet-cli convert <input_model> <output_model>
bitnet-cli benchmark <model> <dataset>
bitnet-cli train <config>
```

### ⚡ Development Workflow Pattern

```rust
// Standard development workflow
[BatchTool]:
  - Bash("git pull origin main")
  - Bash("cargo update --workspace") 
  - Bash("cargo build --workspace --release")
  - Bash("cargo test --workspace")
  - Bash("cargo clippy --workspace --all-targets -- -D warnings")
  - Bash("cargo bench --package bitnet-benchmarks")
  - Bash("git add .")
  - Bash("git commit -m 'feat: implement [current phase] core functionality'")
  - Bash("git push origin feature-branch")
```

This configuration provides complete context for all phases from 1.4 through the end of the project, ensuring Claude Code has full visibility into the implementation roadmap while maintaining the original configuration structure and patterns.