# Claude Code Configuration for BitNet-Rust Project - Tensor Implementation Focus

## 🧠 PROJECT OVERVIEW: BitNet-Rust Neural Network Framework with Complete Tensor Operations

**Repository:** `github.com/Wavegoodvybe2929/bitnet-rust`

**Project Status:** A high-performance Rust implementation of BitNet neural networks with advanced memory management, device abstraction, MLX acceleration for Apple Silicon, and comprehensive infrastructure for quantized neural networks

**Current Implementation Phase:** ✅ Phases 1-3 Foundation Complete → 🎯 **Phase 4: Complete Tensor Operations Infrastructure - CRITICAL PATH**

**Core Strength:** Production-ready memory management, advanced MLX acceleration (up to 3,059x speedup), comprehensive device abstraction, and sophisticated quantization foundations

## 🚨 CRITICAL: TENSOR-FOCUSED EXECUTION PATTERNS

**MANDATORY RULE:** All tensor operations must leverage the existing production-ready memory management foundation, MLX acceleration infrastructure, and advanced device abstraction while completing the missing tensor functionality.

## 🔴 MANDATORY CONCURRENT PATTERNS FOR TENSOR IMPLEMENTATION

- **Memory Pool Operations:** ALWAYS use HybridMemoryPool for all tensor allocations
- **Device Abstraction:** ALWAYS leverage auto_select_device() for optimal device selection  
- **MLX Acceleration:** ALWAYS utilize MLX framework for Apple Silicon tensor operations
- **Metal GPU Integration:** ALWAYS use Metal compute shaders for GPU tensor operations
- **Quantization Integration:** ALWAYS integrate with existing quantization infrastructure
- **Zero-Copy Operations:** ALWAYS prefer zero-copy tensor operations where possible

### ⚡ TENSOR-IMPLEMENTATION GOLDEN RULE (PHASE 4)
> "1 MESSAGE = COMPLETE PRODUCTION-READY TENSOR OPERATIONS WITH FULL BITNET INTEGRATION"

### Examples of CORRECT Phase 4 Tensor Implementation concurrent execution:

```rust
// ✅ CORRECT: Complete Tensor Operations implementation leveraging existing foundation
[Single Message]:
  - TodoWrite { todos: [20+ todos focusing on tensor operations as critical path] }
  - Task("Tensor Architect: Build complete BitNetTensor with shape management...")
  - Task("Operations Engineer: Implement core tensor operations (add, mul, matmul)...")
  - Task("Quantization Engineer: Integrate tensor operations with quantization...")
  - Task("MLX Integration Specialist: Optimize tensor operations with MLX acceleration...")
  - Task("Memory Engineer: Implement tensor-specific memory management...")
  - Task("SIMD Engineer: Add SIMD optimizations for tensor operations...")
  - Bash("cd bitnet-rust && cargo build --workspace --release --features tensor-complete")
  - Write("bitnet-core/src/tensor/core.rs", completeTensorImplementation)
  - Write("bitnet-core/src/tensor/operations.rs", tensorOperationsCore)
  - Write("bitnet-core/src/tensor/shape.rs", tensorShapeManagement)
  - Write("bitnet-core/src/tensor/dtype.rs", dataTypeSystem)
  - Write("bitnet-quant/src/tensor_integration.rs", quantizedTensorOps)
  - Write("examples/tensor_operations_demo.rs", comprehensiveTensorDemo)
  - Bash("cd bitnet-rust && cargo test --package bitnet-core tensor --features tensor-complete")
  - Bash("cd bitnet-rust && cargo bench --package bitnet-benchmarks tensor_operations")
```

## 🎯 BITNET-RUST WORKSPACE ARCHITECTURE (UPDATED FOR TENSOR FOCUS)

### 🦀 Current Implementation Status (UPDATED FOR PHASE 4)

| Component | Status | Priority | Integration Level |
|-----------|--------|----------|-------------------|
| **bitnet-core** | 🟢 Memory & Device Ready | ✅ Foundation Complete | Memory, MLX & device abstraction production-ready |
| **bitnet-core/tensor** | 🔴 **PHASE 4 FOCUS** | 🎯 **IMMEDIATE PRIORITY** | Complete tensor operations implementation needed |
| **bitnet-quant** | 🟡 Foundation Ready | 🎯 **TENSOR INTEGRATION** | Quantization ready for tensor integration |
| **bitnet-benchmarks** | 🟢 Production Ready | ✅ Testing Infrastructure | Ready for tensor performance benchmarking |
| **bitnet-inference** | 🔴 Dependent on Tensors | High Priority Next | Awaiting tensor operations completion |
| **bitnet-training** | 🔴 Dependent on Tensors | High Priority Next | Awaiting tensor operations completion |
| **bitnet-metal** | 🟡 Basic Integration | Medium Priority | Basic Metal support exists, enhancement pending |
| **bitnet-cli** | 🔴 Placeholder | Low Priority | Command-line tools needed |

### 🗃️ Agent Specialization for Phase 4 Tensor Implementation

**Primary Phase 4 Agent Types:**
- **Tensor Architect** - 🎯 **PRIMARY FOCUS** - Core BitNetTensor struct and lifecycle management
- **Operations Engineer** - Mathematical tensor operations (arithmetic, linear algebra)
- **Shape Management Engineer** - Broadcasting, reshaping, indexing, and slicing
- **Memory Integration Specialist** - Tensor-specific memory pool integration
- **MLX Acceleration Engineer** - Apple Silicon tensor operation optimization
- **Quantization Integration Engineer** - Integration with existing quantization systems

**Supporting Specialist Types:**
- **SIMD Optimization Engineer** - Cross-platform SIMD tensor operations
- **Metal Compute Engineer** - GPU tensor operation acceleration
- **Data Type Engineer** - Comprehensive data type system for tensors
- **Gradient Engine Engineer** - Automatic differentiation infrastructure
- **Broadcasting Engineer** - Advanced broadcasting and tensor alignment
- **Serialization Engineer** - Tensor serialization and persistence

## 🎯 PHASE 4: COMPLETE TENSOR OPERATIONS IMPLEMENTATION (CURRENT FOCUS)

### ⚡ 4.1 Core Tensor Infrastructure (IMMEDIATE PRIORITY)

**BitNetTensor Core Implementation:**

```rust
// Phase 4.1: Core Tensor System
[BatchTool]:
  - Write("bitnet-core/src/tensor/mod.rs", tensorModuleCore)
  - Write("bitnet-core/src/tensor/core.rs", bitNetTensorStruct)
  - Write("bitnet-core/src/tensor/shape.rs", shapeManagementSystem)
  - Write("bitnet-core/src/tensor/dtype.rs", dataTypeSystem)
  - Write("bitnet-core/src/tensor/storage.rs", tensorStorageBackend)
  - Write("bitnet-core/src/tensor/lifecycle.rs", tensorLifecycleManagement)
  - Write("bitnet-core/src/tensor/memory_integration.rs", memoryPoolIntegration)
  - Write("bitnet-core/src/tensor/device_integration.rs", deviceAbstractionIntegration)
  - Write("tests/tensor/core_tests.rs", coreTensorTests)
  - Write("tests/tensor/shape_tests.rs", shapeManagementTests)
  - Write("tests/tensor/dtype_tests.rs", dataTypeSystemTests)
  - Bash("cargo test --package bitnet-core tensor::core --features tensor-core")
  - Bash("cargo clippy --package bitnet-core --features tensor-core -- -D warnings")
```

**Core Tensor Features:**
- Complete BitNetTensor struct with shape, dtype, and device tracking
- Integration with existing HybridMemoryPool for efficient memory management
- Device-aware tensor creation and management
- Thread-safe tensor operations with fine-grained locking
- Zero-copy tensor views and slicing operations
- Comprehensive error handling and validation

**BitNetTensor Structure:**
```rust
pub struct BitNetTensor {
    data: Arc<TensorStorage>,
    shape: TensorShape,
    dtype: BitNetDType,
    device: Device,
    memory_pool: Arc<HybridMemoryPool>,
    memory_handle: MemoryHandle,
    requires_grad: bool,
    grad: Option<Arc<BitNetTensor>>,
}
```

### ⚡ 4.2 Mathematical Operations System (HIGH PRIORITY)

```rust
// Phase 4.2: Mathematical Operations Implementation
[BatchTool]:
  - Write("bitnet-core/src/tensor/ops/mod.rs", operationsModuleCore)
  - Write("bitnet-core/src/tensor/ops/arithmetic.rs", arithmeticOperations)
  - Write("bitnet-core/src/tensor/ops/linear_algebra.rs", linearAlgebraOps)
  - Write("bitnet-core/src/tensor/ops/reduction.rs", reductionOperations)
  - Write("bitnet-core/src/tensor/ops/comparison.rs", comparisonOperations)
  - Write("bitnet-core/src/tensor/ops/activation.rs", activationFunctions)
  - Write("bitnet-core/src/tensor/ops/broadcasting.rs", broadcastingSystem)
  - Write("bitnet-core/src/tensor/ops/indexing.rs", indexingAndSlicing)
  - Write("bitnet-core/src/tensor/ops/reshape.rs", reshapeOperations)
  - Write("tests/tensor/ops/arithmetic_tests.rs", arithmeticOperationsTests)
  - Write("tests/tensor/ops/linear_algebra_tests.rs", linearAlgebraTests)
  - Write("tests/tensor/ops/reduction_tests.rs", reductionOperationsTests)
  - Write("tests/tensor/ops/broadcasting_tests.rs", broadcastingTests)
  - Write("benches/tensor/ops/arithmetic_bench.rs", arithmeticPerformanceBenchmarks)
  - Write("benches/tensor/ops/linear_algebra_bench.rs", linearAlgebraPerformanceBenchmarks)
  - Bash("cargo test --package bitnet-core tensor::ops::arithmetic --features tensor-ops")
  - Bash("cargo test --package bitnet-core tensor::ops::linear_algebra --features tensor-ops")
  - Bash("cargo bench --package bitnet-core tensor_ops")
```

**Mathematical Operations Features:**
- Complete arithmetic operations (+, -, *, /, %, pow)
- Advanced linear algebra (matmul, dot, cross, SVD, QR, Cholesky)
- Reduction operations (sum, mean, std, var, min, max)
- Broadcasting system compatible with NumPy/PyTorch semantics
- Activation functions (ReLU, GELU, Sigmoid, Tanh, Softmax)
- Comparison and logical operations
- Advanced indexing and tensor slicing
- In-place and out-of-place operation variants

**Core Operations Implementation:**
- Matrix multiplication with SIMD optimization
- Element-wise operations with broadcasting
- Memory-efficient reduction operations
- Zero-copy tensor views where possible
- Integration with existing MLX and Metal acceleration
- Comprehensive error checking and type validation

### ⚡ 4.3 MLX and Metal Acceleration Integration (CRITICAL PATH)

```rust
// Phase 4.3: Acceleration Integration System
[BatchTool]:
  - Write("bitnet-core/src/tensor/acceleration/mod.rs", accelerationModuleCore)
  - Write("bitnet-core/src/tensor/acceleration/mlx.rs", mlxTensorAcceleration)
  - Write("bitnet-core/src/tensor/acceleration/metal.rs", metalTensorAcceleration)
  - Write("bitnet-core/src/tensor/acceleration/simd.rs", simdTensorOptimization)
  - Write("bitnet-core/src/tensor/acceleration/dispatch.rs", accelerationDispatchSystem)
  - Write("bitnet-core/src/tensor/acceleration/kernels.rs", customKernelImplementations)
  - Write("bitnet-core/src/tensor/acceleration/memory_mapping.rs", acceleratedMemoryMapping)
  - Write("bitnet-core/src/tensor/acceleration/auto_select.rs", automaticAccelerationSelection)
  - Write("tests/tensor/acceleration/mlx_tests.rs", mlxAccelerationTests)
  - Write("tests/tensor/acceleration/metal_tests.rs", metalAccelerationTests)
  - Write("tests/tensor/acceleration/simd_tests.rs", simdOptimizationTests)
  - Write("tests/tensor/acceleration/dispatch_tests.rs", accelerationDispatchTests)
  - Write("benches/tensor/acceleration/mlx_bench.rs", mlxPerformanceBenchmarks)
  - Write("benches/tensor/acceleration/metal_bench.rs", metalPerformanceBenchmarks)
  - Write("benches/tensor/acceleration/simd_bench.rs", simdPerformanceBenchmarks)
  - Bash("cargo test --package bitnet-core tensor::acceleration::mlx --features mlx,tensor-complete")
  - Bash("cargo test --package bitnet-core tensor::acceleration::metal --features metal,tensor-complete")
  - Bash("cargo bench --package bitnet-core tensor_acceleration --features apple-silicon,tensor-complete")
```

**Acceleration Integration Features:**
- Seamless MLX tensor operation integration for Apple Silicon
- Metal compute shader integration for GPU tensor operations
- Cross-platform SIMD optimization (AVX2, NEON, SSE)
- Automatic acceleration backend selection based on operation characteristics
- Memory-efficient GPU-CPU data transfer
- Custom kernel implementations for BitNet-specific operations
- Performance profiling and optimization recommendations

**MLX Tensor Integration:**
- Direct integration with existing MLX acceleration infrastructure
- Zero-copy MLX tensor creation from BitNetTensor
- MLX-optimized matrix operations with 15-40x speedup
- Unified memory architecture leverage on Apple Silicon
- MLX graph optimization for complex tensor operations
- Automatic fallback to Metal/CPU when MLX unavailable

### ⚡ 4.4 Quantization and BitNet Integration (COMPLETION PHASE)

```rust
// Phase 4.4: Quantization Integration and BitNet-Specific Operations
[BatchTool]:
  - Write("bitnet-quant/src/tensor_integration/mod.rs", quantizationTensorIntegration)
  - Write("bitnet-quant/src/tensor_integration/bitnet_ops.rs", bitNetSpecificTensorOps)
  - Write("bitnet-quant/src/tensor_integration/quantized_tensor.rs", quantizedTensorImplementation)
  - Write("bitnet-quant/src/tensor_integration/bitlinear_tensor.rs", bitLinearTensorOperations)
  - Write("bitnet-quant/src/tensor_integration/calibration_tensor.rs", calibrationTensorSupport)
  - Write("bitnet-quant/src/tensor_integration/qat_tensor.rs", qatTensorOperations)
  - Write("bitnet-quant/src/tensor_integration/precision_tensor.rs", mixedPrecisionTensorSupport)
  - Write("examples/tensor/bitnet_operations_demo.rs", bitNetTensorOperationsDemo)
  - Write("examples/tensor/quantized_tensor_demo.rs", quantizedTensorOperationsDemo)
  - Write("examples/tensor/bitlinear_demo.rs", bitLinearTensorDemo)
  - Write("examples/tensor/mixed_precision_demo.rs", mixedPrecisionTensorDemo)
  - Write("tests/tensor_integration/bitnet_ops_tests.rs", bitNetTensorOperationsTests)
  - Write("tests/tensor_integration/quantized_tensor_tests.rs", quantizedTensorTests)
  - Write("tests/tensor_integration/bitlinear_tests.rs", bitLinearTensorTests)
  - Write("benches/tensor_integration/bitnet_ops_bench.rs", bitNetTensorPerformanceBenchmarks)
  - Write("benches/tensor_integration/quantized_ops_bench.rs", quantizedTensorPerformanceBenchmarks)
  - Bash("cargo test --package bitnet-quant tensor_integration::bitnet_ops --features tensor-integration")
  - Bash("cargo test --package bitnet-quant tensor_integration::quantized_tensor --features tensor-integration")
  - Bash("cargo bench --package bitnet-quant tensor_integration --features tensor-integration")
  - Bash("cargo run --example tensor/bitnet_operations_demo --features tensor-complete,bitnet-integration")
```

**BitNet-Specific Tensor Integration:**
- 1.58-bit quantized tensor operations
- BitLinear layer tensor operations with quantization
- Ternary weight tensor representations
- Calibration dataset tensor processing
- QAT-aware tensor operations with straight-through estimation
- Mixed precision tensor operations
- BitNet-optimized activation functions

**Quantized Tensor Features:**
- Seamless integration with existing quantization infrastructure
- Memory-efficient quantized tensor storage
- Dequantization on-demand for compatibility
- Quantization-aware tensor operations
- Scale and zero-point management
- Error analysis and validation tools

### ⚡ 4.5 Integration Testing and Production Readiness (FINAL PHASE)

```rust
// Phase 4.5: Integration Testing and Production Validation
[BatchTool]:
  - Write("bitnet-core/src/tensor/integration/mod.rs", tensorIntegrationModule)
  - Write("bitnet-core/src/tensor/integration/memory_pool.rs", memoryPoolTensorIntegration)
  - Write("bitnet-core/src/tensor/integration/device_abstraction.rs", deviceAbstractionTensorIntegration)
  - Write("bitnet-core/src/tensor/integration/mlx_integration.rs", mlxTensorIntegration)
  - Write("bitnet-core/src/tensor/integration/metal_integration.rs", metalTensorIntegration)
  - Write("examples/tensor/comprehensive_tensor_demo.rs", comprehensiveTensorDemo)
  - Write("examples/tensor/performance_comparison_demo.rs", tensorPerformanceComparisonDemo)
  - Write("examples/tensor/memory_efficiency_demo.rs", tensorMemoryEfficiencyDemo)
  - Write("examples/tensor/acceleration_comparison_demo.rs", accelerationComparisonDemo)
  - Write("examples/tensor/production_workflow_demo.rs", productionTensorWorkflowDemo)
  - Write("tests/integration/tensor_integration_tests.rs", comprehensiveTensorIntegrationTests)
  - Write("tests/integration/memory_tensor_integration_tests.rs", memoryPoolTensorIntegrationTests)
  - Write("tests/integration/device_tensor_integration_tests.rs", deviceAbstractionTensorIntegrationTests)
  - Write("tests/integration/acceleration_tensor_integration_tests.rs", accelerationTensorIntegrationTests)
  - Write("benches/integration/tensor_comprehensive_bench.rs", comprehensiveTensorPerformanceBenchmarks)
  - Write("docs/tensor_implementation_guide.md", tensorImplementationGuide)
  - Write("docs/tensor_performance_guide.md", tensorPerformanceOptimizationGuide)
  - Bash("cargo test --workspace --features tensor-complete,integration-tests")
  - Bash("cargo bench --workspace --features tensor-complete,comprehensive-benchmarks")
  - Bash("cargo run --example tensor/comprehensive_tensor_demo --features tensor-complete,all-accelerations")
  - Bash("cargo doc --workspace --open --no-deps --features tensor-complete")
```

**Integration and Production Features:**
- Comprehensive tensor integration with all existing systems
- Memory-efficient tensor operations leveraging existing pools
- Device-aware tensor operations across CPU/GPU platforms
- Production-ready error handling and recovery mechanisms
- Performance benchmarking and optimization validation
- Comprehensive documentation and usage examples

## 🚀 PHASE 4 SUCCESS CRITERIA

### 🔬 Technical Targets for Phase 4

**Core Tensor Functionality:**
- ✅ Complete BitNetTensor implementation with shape and dtype management
- ✅ Comprehensive mathematical operations (arithmetic, linear algebra, reduction)
- ✅ Advanced broadcasting system compatible with NumPy/PyTorch
- ✅ Memory-efficient tensor storage and lifecycle management
- ✅ Thread-safe tensor operations with performance optimization

**Acceleration Integration:**
- ✅ MLX acceleration integration with up to 40x speedup on Apple Silicon
- ✅ Metal compute shader integration for GPU tensor operations
- ✅ Cross-platform SIMD optimization for CPU tensor operations
- ✅ Automatic acceleration backend selection and fallback mechanisms
- ✅ Zero-copy operations and memory-efficient GPU-CPU transfers

**Quantization and BitNet Integration:**
- ✅ Seamless integration with existing quantization infrastructure
- ✅ BitNet-specific tensor operations (1.58-bit quantization)
- ✅ BitLinear layer tensor operations with quantization support
- ✅ Mixed precision tensor operations and management
- ✅ Production-ready quantized tensor arithmetic

### 📊 Phase 4 Performance Targets

**Tensor Operation Performance:**
- Matrix Multiplication: 15-40x speedup with MLX on Apple Silicon
- Element-wise Operations: 5-15x speedup with SIMD optimization
- Memory Allocation: <100ns tensor creation with existing memory pools
- Broadcasting Operations: Zero-copy where possible, minimal memory overhead
- Device Transfer: <1ms GPU-CPU transfer for typical tensor sizes

**Memory Efficiency:**
- Tensor Storage: <5% memory overhead for tensor metadata
- Zero-Copy Operations: 80% of tensor operations should be zero-copy
- Memory Pool Integration: 95% successful tensor allocations from pools
- Fragmentation: <10% memory fragmentation during typical workloads
- Cleanup Efficiency: Automatic tensor cleanup with 100% success rate

**Integration Performance:**
- API Compatibility: 100% compatibility with existing bitnet-core APIs
- Error Handling: Comprehensive error recovery without memory leaks
- Thread Safety: All tensor operations thread-safe with minimal contention
- Device Abstraction: Seamless operation across CPU/Metal/MLX devices
- Quantization Integration: <10% performance overhead for quantized operations

### 📊 Phase 4 Completion Gates

**Functional Completeness:**
- [ ] Complete BitNetTensor struct with full lifecycle management
- [ ] All mathematical operations implemented and tested
- [ ] Broadcasting system fully compatible with NumPy/PyTorch semantics
- [ ] MLX and Metal acceleration fully integrated and optimized
- [ ] Quantization integration provides BitNet-specific operations

**Performance Validation:**
- [ ] Tensor operations achieve target performance benchmarks
- [ ] Memory efficiency meets production requirements
- [ ] Acceleration integration provides expected speedups
- [ ] Integration tests pass with all existing systems
- [ ] No performance regression in existing functionality

**Production Readiness:**
- [ ] Comprehensive error handling and recovery mechanisms
- [ ] Thread-safe operations with performance optimization
- [ ] Complete API documentation and usage examples
- [ ] Ready for inference and training engine implementation
- [ ] Benchmark suite validates all performance claims

## 🔄 PHASE 4 TO PHASE 5 TRANSITION

### 🎯 Phase 5 Prerequisites from Phase 4

**Required Phase 4 Completions:**
- ✅ Production-ready BitNetTensor with complete mathematical operations
- ✅ Fully integrated MLX and Metal acceleration for tensor operations
- ✅ Comprehensive quantization integration with BitNet-specific operations
- ✅ Memory-efficient tensor management leveraging existing infrastructure
- ✅ Performance-validated tensor operations with benchmarking

**Phase 5 Integration Points:**
- Inference engine implementation using complete tensor operations
- Training infrastructure leveraging tensor automatic differentiation
- Model architecture implementations with tensor building blocks
- Distributed computing using tensor communication primitives
- CLI and Python bindings exposing complete tensor functionality

## 🎯 PROJECT-SPECIFIC COMMANDS FOR PHASE 4

### 🚀 Phase 4 Development Commands

```bash
# Phase 4 focused build
cargo build --workspace --features tensor-complete,mlx,metal --release

# Phase 4 comprehensive testing  
cargo test --package bitnet-core tensor --features tensor-complete
cargo test --package bitnet-quant tensor_integration --features tensor-integration

# Tensor operations performance validation
cargo bench --package bitnet-core tensor_ops --features tensor-complete

# MLX tensor acceleration validation
cargo bench --package bitnet-core tensor_acceleration --features mlx,tensor-complete

# Integration validation across all components
cargo test --workspace --features tensor-complete,integration-tests

# Documentation generation for Phase 4
cargo doc --workspace --open --no-deps --features tensor-complete

# Comprehensive tensor demonstration
cargo run --example tensor/comprehensive_tensor_demo --features tensor-complete,all-accelerations

# Performance comparison validation
cargo run --example tensor/performance_comparison_demo --features tensor-complete,benchmarking
```

### ⚡ Phase 4 Development Workflow Pattern

```rust
// Phase 4 standard development workflow
[BatchTool]:
  - Bash("git checkout -b feature/phase-4-tensor-operations")
  - Bash("cargo update --workspace") 
  - Bash("cargo build --workspace --features tensor-complete,mlx,metal --release")
  - Bash("cargo test --package bitnet-core tensor --features tensor-complete-validation")
  - Bash("cargo test --package bitnet-quant tensor_integration --features tensor-integration-validation")
  - Bash("cargo clippy --workspace --features tensor-complete -- -D warnings")
  - Bash("cargo bench --workspace --features tensor-complete,comprehensive-benchmarks")
  - Write("PHASE_4_COMPLETION.md", phase4CompletionReport)
  - Bash("git add .")
  - Bash("git commit -m 'feat: complete Phase 4 Tensor Operations with MLX/Metal acceleration and quantization integration'")
  - Bash("git push origin feature/phase-4-tensor-operations")
```

### 📋 Phase 4 Development Todos

**Week 1 (Days 1-7): Core Tensor Infrastructure**
- [ ] Implement complete BitNetTensor struct with shape and dtype management
- [ ] Create tensor storage backend with memory pool integration
- [ ] Build tensor lifecycle management and device abstraction integration
- [ ] Implement basic tensor creation, cloning, and destruction
- [ ] Add tensor metadata management and validation systems
- [ ] Create comprehensive tensor error handling and recovery
- [ ] Validate with tensor core functionality test suite

**Week 2 (Days 8-14): Mathematical Operations System**
- [ ] Implement arithmetic operations with broadcasting support
- [ ] Create linear algebra operations (matmul, dot, SVD, QR)
- [ ] Build reduction operations (sum, mean, min, max, std)
- [ ] Add activation functions and comparison operations
- [ ] Implement indexing, slicing, and reshaping operations
- [ ] Create in-place and out-of-place operation variants
- [ ] Add SIMD optimization for cross-platform performance

**Week 3 (Days 15-21): Acceleration Integration**
- [ ] Integrate MLX acceleration for Apple Silicon tensor operations
- [ ] Add Metal compute shader support for GPU tensor operations
- [ ] Implement automatic acceleration backend selection
- [ ] Create zero-copy tensor operations where possible
- [ ] Add performance profiling and optimization recommendations
- [ ] Build custom kernels for BitNet-specific operations
- [ ] Validate acceleration with comprehensive benchmarking

**Week 4 (Days 22-28): Quantization and BitNet Integration**
- [ ] Integrate tensor operations with existing quantization infrastructure
- [ ] Implement BitNet-specific tensor operations (1.58-bit quantization)
- [ ] Add BitLinear layer tensor operation support
- [ ] Create mixed precision tensor management system
- [ ] Build QAT-aware tensor operations with straight-through estimation
- [ ] Add calibration dataset tensor processing support
- [ ] Validate with comprehensive BitNet tensor integration tests

**Final Integration (Days 29-30):**
- [ ] Complete integration testing with all existing systems
- [ ] Validate performance benchmarks and optimization targets
- [ ] Create comprehensive documentation and usage examples
- [ ] Prepare for Phase 5 inference and training engine implementation

This updated configuration focuses on Phase 4: Complete Tensor Operations Implementation as the critical path, building on the existing production-ready memory management and acceleration infrastructure while providing the foundation for inference and training engines.