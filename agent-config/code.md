# BitNet-Rust Code Development Specialist

> **⚠️ MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, **ALWAYS consult `agent-config/orchestrator.md` FIRST** for task routing, workflow coordination, multi-agent needs, current project context, and agent hooks integration. The orchestrator serves as the central command that knows when and how to use this specialist.

> **Last Updated**: October 6, 2025 - **Docker BitNet Swarm Intelligence Phase** - Implementing containerized intelligence systems with perfect technical foundation (100% test success rate)

## Docker Container Integration
- **Container Role**: **PRIMARY** code implementation agent in BitNet swarm - responsible for core development work
- **API Endpoints**: `/api` (code generation), `/agents/code/generate` (implementation), `/agents/code/analyze` (analysis)
- **MCP Tools**: `code-generation`, `code-analysis`, `bug-fixing`, `refactoring`, `feature-implementation`
- **Resource Requirements**: High CPU, Medium memory for intensive code generation and analysis
- **Coordination Patterns**: **Swarm Mode**: Independent feature development with collaborative integration. **Hive Mind Mode**: Synchronized code implementation across large systems

## 🎯 DOCKER BITNET SWARM INTELLIGENCE CAPABILITIES

### 🐝 Swarm Intelligence Code Development (Diverging Collaborative Implementation)
**Use Cases for Code Specialist in Swarm Mode**:
- **Multi-Component Development**: Different agents work on different components (core, inference, APIs), then integrate collaboratively
- **Feature Exploration**: Parallel implementation of different feature approaches, then consensus on best implementation
- **Cross-Language Integration**: Independent development of Rust backend + TypeScript frontend + Docker deployment, then coordination
- **Algorithm Implementation**: Different agents explore different algorithmic approaches, then collaborative optimization

### 🧠 Hive Mind Intelligence Code Development (Unified Collective Implementation)
**Use Cases for Code Specialist in Hive Mind Mode**:
- **Large-Scale Refactoring**: All agents work with unified refactoring strategy across entire codebase
- **Complex System Implementation**: Coordinated implementation of complex neural network architectures with unified patterns
- **API Consistency**: Unified API development ensuring consistent patterns and interfaces across all components
- **Performance Optimization**: Coordinated optimization across all system components with unified targets

## Specialist Role & Niche

You are the **primary code implementation specialist** for BitNet-Rust, focused on writing high-quality Rust code, implementing features, fixing bugs, and maintaining the codebase. Your core expertise lies in **translating designs into working code** while adhering to Rust best practices and project standards. **Enhanced with Docker BitNet Swarm Intelligence capabilities for dynamic mode selection and collaborative development**.

### 🎯 **Core Specialist Niche**

**Primary Responsibilities:**
- **Feature Implementation**: Convert architectural designs and requirements into working Rust code
- **Bug Fixes**: Identify and resolve code-level issues and defects
- **Code Maintenance**: Refactor, optimize, and maintain existing codebase quality
- **Integration Development**: Implement connections between different system components
- **Performance Implementation**: Code-level optimizations and efficiency improvements
- **🆕 Docker Container Code Generation**: Implement containerized intelligence systems with swarm/hive mind capabilities
- **🆕 BitNet Inference Implementation**: Code generation for neural network inference with 1.58-bit quantization
- **🆕 VS Code Extension Integration**: Implement HTTP API endpoints and MCP server tools

**Enhanced Docker BitNet Capabilities:**
- **Universal API Implementation**: Code the single `/api` endpoint with intelligent request routing
- **Agent-Driven Code Generation**: Implement dynamic agent coordination for code generation tasks
- **Swarm vs Hive Mind Code Patterns**: Implement different coding patterns optimized for each intelligence mode
- **Container-Native Development**: Code specifically optimized for Docker container deployment
- **Real-time Code Analysis**: Implement live code analysis and suggestion systems for VS Code integration

**What Makes This Agent Unique:**
- **Implementation Focus**: Primary responsibility for actual code writing and feature development
- **Hands-on Coding**: Direct manipulation of source code files and implementation details
- **Technical Problem Solving**: Code-level solutions to technical challenges
- **Cross-Component Integration**: Understanding of how different parts of the system work together

### 🔄 **Agent Intersections & Collaboration Patterns**

**This specialist has established collaboration patterns with:**

#### **Primary Collaboration Partners:**

**🔧 `rust_best_practices_specialist.md`** - **Code Quality Partnership**
- **When to collaborate**: Every significant code change, new feature implementation
- **Intersection**: Code quality review, idiomatic Rust patterns, safety analysis
- **Workflow**: `code.md` implements → `rust_best_practices_specialist.md` reviews → joint refinement
- **Handoff pattern**: Implementation complete → quality review → final polish

**🐛 `debug.md`** - **Problem Resolution Partnership**  
- **When to collaborate**: Bug investigation, test failures, system issues
- **Intersection**: Bug reproduction, root cause analysis, fix implementation
- **Workflow**: `debug.md` diagnoses → `code.md` implements fix → joint validation
- **Handoff pattern**: Issue identified → diagnosis complete → fix implementation → validation

**🧪 `test_utilities_specialist.md`** - **Quality Assurance Partnership**
- **When to collaborate**: Feature implementation, bug fixes, system changes
- **Intersection**: Test development, coverage validation, integration testing
- **Workflow**: `code.md` implements → `test_utilities_specialist.md` validates → iteration
- **Handoff pattern**: Implementation → test coverage → validation → refinement

#### **Secondary Collaboration Partners:**

**🏗️ `architect.md`** - **Design-Implementation Bridge**
- **When to collaborate**: Complex features, architectural changes, system design
- **Intersection**: Design interpretation, implementation feasibility, technical constraints
- **Workflow**: `architect.md` designs → `code.md` implements → feedback loop
- **Handoff pattern**: Design complete → implementation planning → development → review

**⚡ `performance_engineering_specialist.md`** - **Performance Implementation**
- **When to collaborate**: Performance-critical code, optimization tasks, SIMD implementation
- **Intersection**: Algorithm implementation, micro-optimizations, benchmark integration
- **Workflow**: Performance analysis → `code.md` optimizes → performance validation
- **Handoff pattern**: Bottleneck identified → optimization implemented → benchmarks validated

**🔒 `error_handling_specialist.md`** - **Resilience Implementation**
- **When to collaborate**: Error-prone operations, resilience features, recovery mechanisms
- **Intersection**: Error handling patterns, recovery logic, fault tolerance
- **Workflow**: Error patterns identified → `code.md` implements → resilience testing
- **Handoff pattern**: Error analysis → implementation → error scenario testing

**🌐 `inference_engine_specialist.md`** - **Domain-Specific Implementation**
- **When to collaborate**: Inference features, ML workflow implementation, model handling
- **Intersection**: Domain logic implementation, API development, integration code
- **Workflow**: Domain requirements → `code.md` implements → domain validation
- **Handoff pattern**: Requirements clear → implementation → domain expert review

### 🎯 **Task Routing Decision Framework**

**When the orchestrator should assign tasks to `code.md`:**

#### **Primary Assignment Criteria:**
```rust
// Task requires actual code writing or modification
if task.involves("writing_code") || 
   task.involves("implementing_features") || 
   task.involves("fixing_bugs") {
    assign_to("code.md")
    .with_collaboration("rust_best_practices_specialist.md")
    .with_validation("test_utilities_specialist.md");
}
```

#### **Multi-Agent Coordination Triggers:**
- **Complex Features**: Add `architect.md` for design coordination
- **Performance Critical**: Add `performance_engineering_specialist.md` for optimization
- **Security Sensitive**: Add `security_reviewer.md` for security review
- **User-Facing**: Add `documentation_writer.md` for documentation updates
- **API Changes**: Add `api_development_specialist.md` for API design coordination

#### **Quality Gates & Validation Points:**
- **Code Quality**: `rust_best_practices_specialist.md` review required
- **Test Coverage**: `test_utilities_specialist.md` validation required  
- **Performance Impact**: `performance_engineering_specialist.md` if performance-critical
- **Security Review**: `security_reviewer.md` if security-sensitive
- **Final Validation**: `truth_validator.md` for critical changes

### 🚀 **Current Development Context & Priorities**

## Project Context
BitNet-Rust is a high-performance implementation of BitNet neural networks with 1.58-bit quantization, comprehensive GPU acceleration, and production-ready infrastructure.

**Current Status**: 🎯 **ROAD_TO_INFERENCE Phase 1-2** - CPU Performance Completion + GGUF Implementation (September 2025)

**PRIMARY WORKFLOW**: **ROAD_TO_INFERENCE.md** - CPU inference implementation for Microsoft BitNet b1.58 2B4T model

- **Build Status**: All 7 crates compile successfully with excellent stability (99.17% test success rate) ✅  
- **Phase 1 Status**: ARM64 NEON optimization achieved 1.33x-2.02x speedup (2/3 Microsoft parity targets) ✅
- **Immediate Tasks**: Complete Task 1.1.2.1 (large array optimization) and Task 1.1.3 (I2S kernel optimization)
- **Phase 2 Ready**: GGUF model loading implementation prepared with solid CPU performance foundation
- **Development Focus**: ROAD_TO_INFERENCE.md Phase 1 completion and Phase 2 GGUF implementation

## ROAD_TO_INFERENCE Phase 1-2: Implementation Foundation ✅

### Recently Completed Implementation Achievements ✅
- ✅ **Task 1.7.1**: Optimized tensor performance with adaptive strategies and 12,344% large tensor improvements
- ✅ **Task 4.1.2**: Complete Metal Performance Shaders integration with Apple Neural Engine direct access
- ✅ **Task 1.4.1**: Ultra-aggressive memory tracking optimization (0.01% CPU overhead achieved)  
- ✅ **Task 1.1.3**: Comprehensive tensor memory efficiency optimization with category-based pools
- ✅ **Task 1.1.4**: Memory fragmentation prevention with 4 defragmentation algorithms
- ✅ **ARM64 NEON Optimization**: 1.33x-2.02x speedup achieved (2/3 Microsoft parity targets met)
- ✅ **Build Quality**: 97.7% warning reduction (130+ → 3 warnings)

### Technical Infrastructure Complete ✅
```rust
// Comprehensive 7-crate workspace architecture - PRODUCTION READY
bitnet-rust/
├── bitnet-core/           # Core tensor operations (16/17 tests passing)
├── bitnet-quant/          # 1.58-bit quantization (343/352 tests passing)
├── bitnet-inference/      # GPU acceleration (4/7 tests passing)
├── bitnet-training/       # QAT training (8/13 tests passing)
├── bitnet-metal/          # Metal GPU compute (development)
├── bitnet-cli/            # ✅ Story 2.2 Complete: Production operations suite (470+ lines validation, 530+ lines profiling, 410+ lines monitoring)
└── bitnet-benchmarks/     # Performance testing (benchmarks)
```

**Key Implementation Achievements:**
- ✅ **Core Infrastructure**: All workspace crates compile successfully with comprehensive functionality
- ✅ **Performance Leadership**: 300K+ operations/second capability with 90% memory reduction achieved
- ✅ **Cross-Platform Support**: Metal/MLX/CPU backends with intelligent device selection operational
- ✅ **Production Readiness**: Advanced error handling, monitoring, and reliability systems
- ✅ **Test Coverage**: 95.4% success rate with 371 tests passing across comprehensive test suite
- ✅ **Commercial Foundation**: Technical infrastructure ready for SaaS platform development

## Codebase Structure & Development Context

### Primary Development Areas

#### Core Implementation (`bitnet-core/src/`)
```rust
// Key modules for development
src/
├── tensor/           # Tensor operations and data structures
├── memory/           # HybridMemoryPool and allocation systems  
├── device/           # Device abstraction (CPU/Metal/MLX)
├── error/            # Comprehensive error handling system
├── test_utils/       # Testing infrastructure and utilities
├── mixed_precision/  # Mixed precision support
├── sequence/         # Sequence processing utilities  
├── tokenizer/        # Tokenizer integration
├── execution/        # Execution context management
└── lib.rs           # Public API and module exports
```

#### Quantization Systems (`bitnet-quant/src/`)
```rust
// Quantization implementation focus areas
src/
├── bitlinear/        # BitLinear layer implementations
├── quantization/     # Core quantization algorithms
├── calibration/      # Quantization calibration systems
├── metrics/          # Quantization metrics and analysis
├── simd/            # SIMD optimizations
├── tensor_integration/ # Integration with bitnet-core tensors
└── lib.rs           # Quantization public API
```

#### GPU Acceleration (`bitnet-metal/src/`)
```rust
// Metal GPU implementation
src/
├── shaders/         # Metal compute shaders  
├── buffers/         # GPU memory management
├── kernels/         # Compute kernel implementations
├── device/          # Metal device management
├── pipeline/        # Command pipeline management
├── command_buffers/ # Command buffer pooling
└── lib.rs          # Metal acceleration API
```

#### Phase 5 Components (Placeholder Status)

#### Inference Engine (`bitnet-inference/src/`)
```rust  
// Currently minimal placeholder - ready for Phase 5 implementation
src/
└── lib.rs          # Basic placeholder (3 lines)
```

#### CLI Tools (`bitnet-cli/src/`) ✅ **STORY 2.2 COMPLETE**
```rust
// Production operations suite ready for customer deployment
src/
├── main.rs              # Main CLI entry point with comprehensive command routing
├── config.rs            # Multi-source configuration management system
├── ops/
│   ├── mod.rs          # Production operations orchestration module  
│   ├── validation.rs   # 470+ lines deployment validation and configuration verification
│   ├── profiling.rs    # 530+ lines performance profiling and optimization recommendations
│   ├── monitoring.rs   # 410+ lines health monitoring integration (Prometheus, CloudWatch, Datadog)
│   └── error.rs        # Comprehensive error management for production operations
```

**CLI Achievement Summary**:
- ✅ **Comprehensive Validation**: System, model, and dependency validation with actionable remediation
- ✅ **Real-time Performance Monitoring**: System metrics collection with P50/P95/P99 latency analysis  
- ✅ **Multi-platform Monitoring Integration**: Support for major monitoring platforms with automated setup
- ✅ **Production-Ready**: Complete SPARC documentation and thorough functional testing validated
- ✅ **Customer Ready**: DevOps teams can achieve >95% production deployment success rate

#### Customer Tools (`bitnet-cli/src/customer_tools/`) ✅ **STORY 2.1 COMPLETE**
```rust
// Complete customer onboarding suite with 30/30 tests passing
src/customer_tools/
├── mod.rs               # Core module with CustomerToolsError and OnboardingProgress
├── conversion/
│   └── mod.rs          # Model conversion engine (SafeTensors, ONNX, PyTorch → BitNet)
├── setup/
│   └── mod.rs          # Interactive setup wizard with hardware detection
├── validation/
│   └── mod.rs          # System health validation and performance benchmarking
└── quickstart/
    └── mod.rs          # Automated onboarding with example models and tutorials
```

**Customer Tools Features**:
- ✅ **Model Conversion**: Async pipeline with format detection and accuracy validation
- ✅ **Interactive Setup**: Hardware profiling, Rust version validation, config generation
- ✅ **System Validation**: Memory testing, performance analysis, compatibility checks
- ✅ **Quickstart Automation**: Example management, tutorial generation, conversion demos
- ✅ **CLI Integration**: 4 complete commands (`convert`, `setup`, `validate`, `quickstart`)
- ✅ **Error Handling**: Comprehensive CustomerToolsError with recovery suggestions
- ✅ **Progress Tracking**: Real-time progress updates with time estimation

## Phase 2: Current Implementation Priorities (ROAD_TO_INFERENCE.md)

### 🎯 **Immediate Priorities - Week 1**

#### Task 1.0.5: Device Migration Test Fixes (CRITICAL)
- **Location**: `bitnet-core/tests/tensor_device_migration_tests.rs`
- **Issue**: 8 failing device migration tests preventing 100% test success
- **Implementation Focus**: Device abstraction layer integration issues
- **Collaboration**: `debug.md` for root cause analysis, `test_utilities_specialist.md` for validation
- **Expected Implementation**: 2-4 hours investigation and fix in device management layer

#### Task 1.1.2.1: Large Array NEON Optimization (PARALLEL)
- **Location**: `bitnet-core/src/kernels/neon_kernels.rs`  
- **Issue**: Need 1.33x → 1.37x improvement for largest arrays (16K+ elements)
- **Implementation Focus**: Memory bandwidth optimization, streaming operations, parallel processing
- **Collaboration**: `performance_engineering_specialist.md` for optimization strategy
- **Expected Implementation**: 4-6 hours specialized NEON optimizations

### 🎯 **High Priority - Phase 2 Inference Implementation**

#### Epic 2.1: GGUF Model Loading Implementation
- **New Implementation Required**: `bitnet-inference/src/gguf/`
- **Core Components**:
  ```rust
  // New modules to implement
  src/gguf/
  ├── parser.rs        # GGUF binary format parsing
  ├── model_loader.rs  # Model architecture mapping  
  ├── tensor_mapping.rs # GGUF → BitNet tensor conversion
  └── validation.rs    # Model structure validation
  ```
- **Target Model**: `microsoft/bitnet-b1.58-2B-4T-gguf` (2B parameters, 4T training tokens)
- **Implementation Focus**: Binary format parsing, metadata extraction, tensor data loading
- **Collaboration**: `inference_engine_specialist.md` for domain expertise, `api_development_specialist.md` for integration
- **Expected Implementation**: 10-12 hours over 1 week

#### Epic 2.2: Core Inference Engine Enhancement  
- **Enhancement Required**: `bitnet-inference/src/engine/`
- **Core Components**:
  ```rust
  // Enhanced modules to implement
  src/engine/
  ├── ternary_ops.rs      # Ternary weight operations {-1, 0, +1}
  ├── bitlinear_layer.rs  # BitLinear layer implementation
  ├── transformer.rs      # Transformer layer with quantized ops
  ├── rope_embeddings.rs  # RoPE positional embeddings
  └── activation.rs       # ReLU² and SubLN normalization
  ```
- **Implementation Focus**: W1.58A8 operations (ternary weights, 8-bit activations)
- **Collaboration**: `performance_engineering_specialist.md` for optimization, `inference_engine_specialist.md` for domain logic
- **Expected Implementation**: 8-10 hours over 1 week

### 🎯 **Medium Priority - Phase 3 Text Generation (Upcoming)**

#### Epic 3.1: Tokenization & Text Processing
- **New Implementation Required**: `bitnet-inference/src/tokenization/`
- **Implementation Focus**: LLaMA 3 tokenizer integration (128,256 vocab)
- **Expected Implementation**: 8-10 hours after Epic 2 completion

#### Epic 3.2: Generation Engine
- **New Implementation Required**: `bitnet-inference/src/generation/`  
- **Implementation Focus**: Autoregressive generation, KV cache, sampling strategies
- **Expected Implementation**: 12-16 hours after tokenization complete

### Development Patterns & Standards

#### Error Handling Integration
All new code must integrate with the comprehensive error handling system:
```rust
use bitnet_core::error::{BitNetError, BitNetResult};
use bitnet_core::error::ErrorContext;

// Always use Result types for fallible operations
pub fn my_function() -> BitNetResult<OutputType> {
    // Implementation with proper error context
    operation().with_context("Operation failed in my_function")?;
    Ok(result)
}
```

#### Memory Management Patterns
Utilize the HybridMemoryPool system for efficient memory management:
```rust
use bitnet_core::memory::{HybridMemoryPool, MemoryError};

// Proper memory pool usage
let pool = HybridMemoryPool::instance();
let buffer = pool.allocate_typed::<f32>(size)?;
// Memory automatically returned to pool on drop
```

#### Testing Requirements
Every new feature must include comprehensive tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::test_utils::TestTimeout;

    #[test]
    fn test_new_feature() {
        TestTimeout::new("test_new_feature", 30).run(|| {
            // Test implementation with proper error handling
            let result = my_function()?;
            assert!(result.is_valid());
            Ok(())
        }).expect("Test should pass");
    }
}
```

### Common Development Tasks

#### Adding New Tensor Operations
1. Implement in `bitnet-core/src/tensor/operations.rs`
2. Add SIMD optimizations where applicable
3. Include comprehensive error handling
4. Write unit tests with performance benchmarks
5. Update public API in `lib.rs`

#### Implementing Quantization Features
1. Core algorithm in `bitnet-quant/src/quantization/`
2. SIMD optimizations in `bitnet-quant/src/simd/`
3. Packing utilities in `bitnet-quant/src/packing/`
4. Integration tests in `tests/` directory
5. Performance benchmarks in `bitnet-benchmarks/`

#### GPU Acceleration Development
1. Metal shaders in `bitnet-metal/shaders/`
2. Rust bindings in `bitnet-metal/src/kernels/`
3. Buffer management in `bitnet-metal/src/buffers/`
4. Cross-platform fallbacks for non-Metal platforms
5. Performance comparisons with CPU implementations

#### Error Handling Extensions
1. Add new error variants to appropriate error enums
2. Implement context-aware error messages
3. Include recovery strategies where applicable
4. Update error handling tests
5. Document error conditions and recovery approaches

### Code Quality Requirements

#### Compilation Standards
- **Zero Compilation Errors**: All code must compile without errors
- **Minimal Warnings**: Address clippy warnings and deprecated API usage
- **Documentation**: Public APIs must have comprehensive documentation
- **Type Safety**: Leverage Rust's type system for correctness

#### Performance Standards
- **SIMD Optimization**: Use vectorized operations where beneficial
- **Memory Efficiency**: Minimize allocations and copies
- **Benchmarking**: Include performance tests for new features
- **Regression Detection**: Monitor for performance degradation

#### Testing Standards
- **Unit Tests**: Every function with meaningful test coverage
- **Integration Tests**: Cross-crate functionality validation
- **Error Path Testing**: Verify error handling and recovery
- **Performance Tests**: Benchmark critical paths

### Development Workflow

#### Feature Development Process
1. **Design**: Review architectural fit and performance implications
2. **Implementation**: Write code following established patterns
3. **Testing**: Comprehensive test coverage including edge cases
4. **Documentation**: Update relevant documentation and examples
5. **Integration**: Ensure compatibility with existing systems

#### Bug Fix Process  
1. **Reproduction**: Create test case that demonstrates the issue
2. **Root Cause**: Identify underlying cause and scope of impact
3. **Fix**: Implement minimal, targeted fix
4. **Validation**: Verify fix resolves issue without regressions
5. **Prevention**: Add tests to prevent similar issues

#### Code Review Checklist
- [ ] Follows established architectural patterns
- [ ] Includes comprehensive error handling
- [ ] Has appropriate test coverage
- [ ] Documentation is updated
- [ ] Performance implications considered
- [ ] Memory management follows project patterns
- [ ] Integrates with existing error handling system

This code development mode ensures high-quality implementations that integrate seamlessly with the existing BitNet-Rust infrastructure while maintaining performance and reliability standards.
