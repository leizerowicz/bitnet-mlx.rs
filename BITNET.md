# Claude Code Configuration for BitNet-Rust Project

## 🧠 PROJECT OVERVIEW: BitNet-Rust Neural Network Framework

**Repository:** `github.com/Wavegoodvybe2929/bitnet-rust`

**Project Status:** A high-performance Rust implementation of BitNet neural networks with advanced memory management, device abstraction, MLX acceleration for Apple Silicon, and comprehensive infrastructure for quantized neural networks

**Current Implementation Phase:** ✅ Phase 2.0 Complete → 🎯 **Phase 3: Calibration and QAT Infrastructure (Weeks 5-6) - CRITICAL PATH**

**Core Strength:** Production-ready BitLinear layer implementation with SIMD optimizations, sophisticated memory management system, and validated quantization core

## 🚨 CRITICAL: BITNET-SPECIFIC EXECUTION PATTERNS

**MANDATORY RULE:** All BitNet Rust operations must leverage the existing memory management foundation, workspace structure, and completed BitLinear layer implementation.

## 🔴 MANDATORY CONCURRENT PATTERNS FOR BITNET-RUST

- **Memory Pool Operations:** ALWAYS use HybridMemoryPool for all allocations
- **Device Abstraction:** ALWAYS leverage auto_select_device() for optimal device selection  
- **Workspace Commands:** ALWAYS batch operations across the modular workspace structure
- **BitLinear Integration:** ALWAYS utilize completed Phase 2 BitLinear layer implementation
- **Calibration Dataset Management:** ALWAYS implement streaming and memory-efficient processing
- **QAT Optimization:** ALWAYS integrate straight-through estimators with existing infrastructure

### ⚡ BITNET-RUST GOLDEN RULE (UPDATED FOR PHASE 3)
> "1 MESSAGE = COMPLETE CALIBRATION & QAT SYSTEM WITH PRODUCTION-READY TRAINING INFRASTRUCTURE"

### Examples of CORRECT Phase 3 BitNet-Rust concurrent execution:

```rust
// ✅ CORRECT: Complete Calibration & QAT implementation building on BitLinear foundation
[Single Message]:
  - TodoWrite { todos: [15+ todos focusing on calibration and QAT as critical path] }
  - Task("Calibration Architect: Build streaming calibration dataset system...")
  - Task("QAT Engineer: Implement straight-through estimator with autograd...")
  - Task("Statistics Collector: Implement activation statistics and histogram tracking...")
  - Task("Training Infrastructure Specialist: Create QAT loss functions and optimizers...")
  - Bash("cd bitnet-rust && cargo build --workspace --release --features calibration,qat")
  - Write("bitnet-quant/src/calibration/*", completeCalibrationSystem)
  - Write("bitnet-training/src/qat/*", quantizationAwareTrainingSystem)
  - Write("examples/calibration_workflow.rs", calibrationDemo)
  - Bash("cd bitnet-rust && cargo test --package bitnet-quant calibration --features phase-3")
  - Bash("cd bitnet-rust && cargo test --package bitnet-training qat --features phase-3")
```

## 🎯 BITNET-RUST WORKSPACE ARCHITECTURE

### 🦀 Current Implementation Status (UPDATED FOR PHASE 3)

| Component | Status | Priority | Integration Level |
|-----------|--------|----------|-------------------|
| **bitnet-core** | 🟢 Production Ready | ✅ Foundation Complete | Core memory, MLX acceleration & device abstraction |
| **bitnet-quant** | 🟢 BitLinear Complete | ✅ Phase 2 Complete | BitLinear layer with SIMD optimizations ready |
| **bitnet-training** | 🟡 Phase 3 Active | 🎯 **CURRENT PRIORITY** | QAT infrastructure implementation in progress |
| **bitnet-benchmarks** | 🟢 Production Ready | ✅ Testing Complete | Ready for calibration and QAT performance testing |
| **bitnet-inference** | 🔴 Dependent on Phase 3 | High Priority Next | Awaiting calibration integration |
| **bitnet-metal** | 🔴 Placeholder | Medium Priority | Enhanced Metal GPU (basic already in core) |
| **bitnet-cli** | 🔴 Placeholder | Low Priority | Command-line tools needed |
| **docs/** | 📚 Available | Documentation | Comprehensive guides available |

### 🏗️ Agent Specialization for Phase 3 Calibration & QAT Implementation

**Primary Phase 3 Agent Types:**
- **Calibration Architect** - 🎯 **PRIMARY FOCUS** - Streaming calibration dataset system
- **QAT Engineer** - Straight-through estimator and autograd integration
- **Statistics Collector** - Activation statistics, histograms, and error analysis
- **Training Infrastructure Specialist** - QAT loss functions, optimizers, and scheduling
- **Error Analysis Engineer** - Quantization metrics and mitigation strategies

**Supporting Specialist Types:**
- **Dataset Streaming Engineer** - Memory-efficient large dataset processing
- **Autograd Integration Specialist** - Custom autograd functions for candle-core
- **Performance Validation Engineer** - Integration testing and benchmarking
- **Progressive Quantization Engineer** - Layer-wise quantization scheduling
- **Knowledge Distillation Specialist** - Teacher-student training implementations

## 🎯 PHASE 3: CALIBRATION AND QAT INFRASTRUCTURE (CURRENT FOCUS)

### ⚡ 3.1 Calibration System Implementation (IMMEDIATE PRIORITY)

**CalibrationDataset Implementation:**

```rust
// Phase 3.1: Core Calibration System
[BatchTool]:
  - Write("bitnet-quant/src/calibration/mod.rs", calibrationModuleRoot)
  - Write("bitnet-quant/src/calibration/dataset.rs", calibrationDatasetCore)
  - Write("bitnet-quant/src/calibration/statistics.rs", activationStatisticsCollector)
  - Write("bitnet-quant/src/calibration/streaming.rs", streamingDatasetProcessor)
  - Write("bitnet-quant/src/calibration/sampling.rs", representativeSamplingStrategies)
  - Write("bitnet-quant/src/calibration/histogram.rs", histogramDataCollection)
  - Write("bitnet-quant/src/calibration/persistence.rs", statisticsSaveLoad)
  - Write("bitnet-quant/src/calibration/config.rs", calibrationConfiguration)
  - Write("tests/calibration/dataset_tests.rs", calibrationDatasetTests)
  - Write("tests/calibration/statistics_tests.rs", statisticsCollectionTests)
  - Write("tests/calibration/streaming_tests.rs", streamingProcessingTests)
  - Bash("cargo test --package bitnet-quant calibration::dataset --features phase-3")
  - Bash("cargo clippy --package bitnet-quant --features calibration -- -D warnings")
```

**Core Calibration Features:**
- Stream large datasets without memory overflow
- Collect min/max activation values per layer
- Build histograms for optimal quantization parameters
- Representative sampling strategies for efficiency
- Integration with existing BitLinear layers
- Save/load calibration statistics for reuse

**CalibrationDataset Structure:**
```rust
struct CalibrationDataset {
    samples: Vec<Tensor>,
    batch_size: usize,
    max_samples: usize,
    streaming: bool,
    memory_limit: usize,
    device: Device,
}
```

### ⚡ 3.2 Quantization-Aware Training (QAT) System (HIGH PRIORITY)

```rust
// Phase 3.2: QAT Infrastructure Implementation
[BatchTool]:
  - Write("bitnet-training/src/qat/mod.rs", qatModuleRoot)
  - Write("bitnet-training/src/qat/straight_through.rs", straightThroughEstimatorCore)
  - Write("bitnet-training/src/qat/autograd.rs", customAutogradFunctions)
  - Write("bitnet-training/src/qat/loss.rs", qatLossFunctions)
  - Write("bitnet-training/src/qat/optimizer.rs", quantizationAwareOptimizers)
  - Write("bitnet-training/src/qat/regularization.rs", quantizationRegularizationTerms)
  - Write("bitnet-training/src/qat/distillation.rs", knowledgeDistillationLoss)
  - Write("bitnet-training/src/qat/progressive.rs", progressiveQuantizationScheduling)
  - Write("bitnet-training/src/qat/state_tracking.rs", quantizationStateTracking)
  - Write("tests/qat/straight_through_tests.rs", steValidationTests)
  - Write("tests/qat/autograd_tests.rs", autogradIntegrationTests)
  - Write("tests/qat/loss_tests.rs", qatLossFunctionTests)
  - Write("tests/qat/optimizer_tests.rs", quantizationOptimizerTests)
  - Write("benches/qat/straight_through_bench.rs", stePerformanceBenchmarks)
  - Write("benches/qat/training_bench.rs", qatTrainingBenchmarks)
  - Bash("cargo test --package bitnet-training qat::straight_through --features phase-3")
  - Bash("cargo test --package bitnet-training qat::autograd --features phase-3")
  - Bash("cargo bench --package bitnet-training qat::straight_through")
```

**Straight-Through Estimator Features:**
- Forward pass: quantize weights and activations normally
- Backward pass: pass gradients through unchanged (STE)
- Custom autograd functions for candle-core integration
- Different STE variants (clipped, soft, learnable)
- Integration with existing BitLinear layer implementation
- Gradient scaling and normalization for stability

**QAT Training Infrastructure:**
- Quantization-aware loss functions
- Regularization terms for better quantization
- Knowledge distillation for teacher-student training
- Progressive quantization scheduling
- Quantization-specific optimizers (Adam/AdamW adaptations)
- Training state tracking and checkpointing

### ⚡ 3.3 Error Analysis and Quantization Metrics (CRITICAL PATH)

```rust
// Phase 3.3: Error Analysis and Metrics System
[BatchTool]:
  - Write("bitnet-quant/src/metrics/mod.rs", quantizationMetricsModule)
  - Write("bitnet-quant/src/metrics/error_analysis.rs", errorAnalysisCore)
  - Write("bitnet-quant/src/metrics/mse.rs", meanSquaredErrorMetrics)
  - Write("bitnet-quant/src/metrics/sqnr.rs", signalQuantizationNoiseRatio)
  - Write("bitnet-quant/src/metrics/cosine_similarity.rs", cosineSimilarityMetrics)
  - Write("bitnet-quant/src/metrics/layer_wise.rs", layerWiseErrorAnalysis)
  - Write("bitnet-quant/src/metrics/visualization.rs", errorVisualizationTools)
  - Write("bitnet-quant/src/metrics/mitigation.rs", errorMitigationStrategies)
  - Write("bitnet-quant/src/metrics/reporting.rs", metricsReportingSystem)
  - Write("tests/metrics/error_analysis_tests.rs", errorAnalysisValidation)
  - Write("tests/metrics/layer_wise_tests.rs", layerWiseAnalysisTests)
  - Write("examples/metrics/error_analysis_demo.rs", errorAnalysisDemo)
  - Write("benches/metrics/analysis_bench.rs", metricsPerformanceBenchmarks)
  - Bash("cargo test --package bitnet-quant metrics::error_analysis --features phase-3")
  - Bash("cargo test --package bitnet-quant metrics::layer_wise --features phase-3")
  - Bash("cargo bench --package bitnet-quant metrics --features analysis")
```

**Quantization Metrics Implementation:**
- Mean Squared Error (MSE) between original and quantized
- Signal-to-Quantization-Noise Ratio (SQNR) calculations
- Cosine similarity metrics for output comparison
- Layer-wise error analysis and reporting
- Error visualization tools for debugging
- Automated error mitigation strategies

**Error Analysis Features:**
- Real-time error tracking during training
- Layer sensitivity analysis for mixed-precision decisions
- Quantization parameter optimization based on error metrics
- Integration with calibration system for optimal parameters
- Memory-efficient error computation using existing pools

### ⚡ 3.4 Integration and Production Readiness (COMPLETION PHASE)

```rust
// Phase 3.4: Integration Testing and Production Features
[BatchTool]:
  - Write("bitnet-training/src/integration/mod.rs", trainingIntegrationModule)
  - Write("bitnet-training/src/integration/bitlinear.rs", bitLinearQATIntegration)
  - Write("bitnet-training/src/integration/memory.rs", memoryEfficientQATTraining)
  - Write("bitnet-training/src/integration/device.rs", deviceAwareQATTraining)
  - Write("bitnet-training/src/integration/checkpointing.rs", qatCheckpointingSystem)
  - Write("examples/qat/basic_training.rs", basicQATTrainingExample)
  - Write("examples/qat/progressive_quantization.rs", progressiveQuantizationDemo)
  - Write("examples/qat/knowledge_distillation.rs", knowledgeDistillationExample)
  - Write("examples/qat/calibration_to_training.rs", calibrationToTrainingPipeline)
  - Write("examples/qat/error_monitoring.rs", errorMonitoringExample)
  - Write("tests/integration/qat_integration_tests.rs", comprehensiveQATIntegrationTests)
  - Write("tests/integration/calibration_integration_tests.rs", calibrationIntegrationTests)
  - Write("tests/integration/memory_integration_tests.rs", memoryEfficientQATTests)
  - Write("benches/integration/qat_training_bench.rs", qatTrainingPerformanceBenchmarks)
  - Write("docs/qat_guide.md", qatTrainingGuide)
  - Write("docs/calibration_guide.md", calibrationWorkflowGuide)
  - Bash("cargo test --workspace --features qat-integration,calibration-integration")
  - Bash("cargo bench --workspace --features qat-complete,calibration-complete")
  - Bash("cargo run --example qat/progressive_quantization --features full-qat")
  - Bash("cargo doc --workspace --open --no-deps --features phase-3-complete")
```

**Integration Features:**
- Seamless BitLinear layer integration with QAT
- Memory-efficient training with existing memory pools
- Device-aware training across CPU/GPU platforms
- Comprehensive checkpointing and resume functionality
- Production-ready error handling and recovery

## 🚀 PHASE 3 SUCCESS CRITERIA

### 🔬 Technical Targets for Phase 3

**Calibration System Functionality:**
- ✅ Stream datasets larger than available memory
- ✅ Collect comprehensive activation statistics
- ✅ Generate optimal quantization parameters
- ✅ Save/load calibration statistics efficiently
- ✅ Representative sampling for calibration efficiency

**QAT Training System:**
- ✅ Straight-through estimator with gradient flow
- ✅ Custom autograd functions for candle-core
- ✅ Quantization-aware loss functions
- ✅ Progressive quantization scheduling
- ✅ Knowledge distillation integration

**Error Analysis & Metrics:**
- ✅ Comprehensive quantization error metrics
- ✅ Layer-wise sensitivity analysis
- ✅ Real-time error monitoring during training
- ✅ Automated error mitigation strategies
- ✅ Visualization tools for debugging

### 📊 Phase 3 Performance Targets

**Calibration Performance:**
- Dataset Processing: Handle 10GB+ datasets with <2GB memory
- Statistics Collection: <1% overhead on forward passes
- Parameter Optimization: 95%+ optimal quantization parameters
- Storage Efficiency: <10MB calibration statistics per model
- Streaming Throughput: Match or exceed batch processing speed

**QAT Training Performance:**
- Training Overhead: <20% slowdown vs full-precision training
- Memory Efficiency: 60-70% memory reduction during training
- Convergence: Maintain model quality within 2% of full-precision
- Gradient Flow: Stable gradients through quantization layers
- Checkpointing: <5% overhead for state preservation

**Error Analysis Performance:**
- Metrics Computation: <5% overhead during training
- Layer Analysis: Complete sensitivity analysis in <1 minute
- Visualization: Real-time error plotting capabilities
- Mitigation: Automatic parameter adjustment based on error thresholds
- Reporting: Comprehensive error reports in <10 seconds

### 📊 Phase 3 Completion Gates

**Functional Completeness:**
- [ ] Calibration dataset handles streaming large datasets correctly
- [ ] QAT training maintains gradient flow through quantization
- [ ] Straight-through estimator integrates with autograd system
- [ ] Error analysis provides actionable quantization insights
- [ ] Progressive quantization schedules optimize training

**Performance Validation:**
- [ ] Calibration system processes large datasets efficiently
- [ ] QAT training shows minimal performance overhead
- [ ] Error metrics provide accurate quantization quality assessment
- [ ] Memory usage stays within production constraints
- [ ] Integration tests with BitLinear layers succeed

**Production Readiness:**
- [ ] Error handling covers all edge cases and recovery
- [ ] API design follows established patterns from previous phases
- [ ] Documentation includes comprehensive usage examples
- [ ] Checkpointing and resume functionality works reliably
- [ ] Ready for Phase 4 model architecture integration

## 🔄 PHASE 3 TO PHASE 4 TRANSITION

### 🎯 Phase 4 Prerequisites from Phase 3

**Required Phase 3 Completions:**
- ✅ Stable calibration dataset processing system
- ✅ Production-ready QAT training infrastructure  
- ✅ Validated straight-through estimator implementation
- ✅ Comprehensive error analysis and metrics system
- ✅ Integration-ready with model architecture support

**Phase 4 Integration Points:**
- Transformer architecture quantization using QAT system
- Model conversion utilities leveraging calibration infrastructure
- Architecture-specific implementations with error monitoring
- Progressive quantization for different model components
- Performance validation using established metrics system

## 🎯 PROJECT-SPECIFIC COMMANDS FOR PHASE 3

### 🚀 Phase 3 Development Commands

```bash
# Phase 3 focused build
cargo build --workspace --features calibration,qat,error-analysis --release

# Phase 3 comprehensive testing  
cargo test --package bitnet-quant calibration --features phase-3
cargo test --package bitnet-training qat --features phase-3

# QAT training performance validation
cargo bench --package bitnet-training qat --features straight-through

# Calibration system validation
cargo test --package bitnet-quant calibration --features streaming

# Integration validation across components
cargo test --workspace --features qat-integration,calibration-integration

# Documentation generation for Phase 3
cargo doc --workspace --open --no-deps --features phase-3-complete

# Error analysis and metrics testing
cargo test --package bitnet-quant metrics --features error-analysis

# Memory efficiency validation with QAT
cargo run --example qat/memory_efficient_training --features memory-profiling
```

### ⚡ Phase 3 Development Workflow Pattern

```rust
// Phase 3 standard development workflow
[BatchTool]:
  - Bash("git checkout -b feature/phase-3-calibration-qat")
  - Bash("cargo update --workspace") 
  - Bash("cargo build --workspace --features calibration,qat,error-analysis --release")
  - Bash("cargo test --package bitnet-quant calibration --features phase-3-validation")
  - Bash("cargo test --package bitnet-training qat --features phase-3-validation")
  - Bash("cargo clippy --workspace --features calibration,qat -- -D warnings")
  - Bash("cargo bench --workspace --features qat,calibration")
  - Write("PHASE_3_COMPLETION.md", phase3CompletionReport)
  - Bash("git add .")
  - Bash("git commit -m 'feat: complete Phase 3 Calibration and QAT infrastructure with error analysis'")
  - Bash("git push origin feature/phase-3-calibration-qat")
```

### 📋 Phase 3 Daily Development Todos

**Week 1 (Days 1-7): Calibration System**
- [ ] Implement streaming calibration dataset processing
- [ ] Create activation statistics collection system
- [ ] Build histogram-based quantization parameter optimization
- [ ] Implement calibration statistics persistence and loading
- [ ] Create representative sampling strategies for efficiency
- [ ] Integrate with existing memory pool and device abstraction
- [ ] Validate with comprehensive test suite and benchmarks

**Week 2 (Days 8-14): QAT Infrastructure**
- [ ] Implement straight-through estimator with gradient flow
- [ ] Create custom autograd functions for candle-core
- [ ] Build quantization-aware loss functions and regularization
- [ ] Implement progressive quantization scheduling system
- [ ] Create knowledge distillation training infrastructure
- [ ] Add quantization-specific optimizer adaptations
- [ ] Build comprehensive error analysis and metrics system

**Integration & Production (Days 15-16):**
- [ ] Complete integration testing with BitLinear layers
- [ ] Validate memory-efficient QAT training workflows
- [ ] Create production-ready examples and documentation
- [ ] Perform comprehensive performance benchmarking
- [ ] Prepare for Phase 4 model architecture integration

This updated configuration focuses on Phase 3: Calibration and QAT Infrastructure as the current priority, building on the completed Phase 2 BitLinear implementation while preparing for seamless transition to Phase 4 model architecture support.