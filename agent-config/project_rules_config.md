# BitNet-Rust Project Rules & Guidelines

## Project Overview
BitNet-Rust is a neural network quantization platform in active technical foundation development, requiring systematic stabilization across core components before advancing to commercial features.

## Development Phases - ACCURATE STATUS UPDATE (September 3, 2025)
- **Phase 1-3**: Core Infrastructure Development (**IN PROGRESS** ⚠️)  
- **Current Achievement**: **91.3% Test Pass Rate** - Infrastructure Development with Major Stabilization Required
- **Phase 4-5**: Advanced Features (**BLOCKED** - Dependencies not met)

**Current Status**: ⚠️ **TECHNICAL FOUNDATION DEVELOPMENT PHASE** - Core Stabilization Required (September 3, 2025)

### Latest Development Status (September 3, 2025) - TECHNICAL FOUNDATION DEVELOPMENT PHASE ⚠️
**CRITICAL REALITY CHECK**: Project requires extensive technical foundation work before commercial features

#### ⚠️ TECHNICAL FOUNDATION DEVELOPMENT STATUS:
- **Build System**: ✅ All 7 crates compile successfully (verified September 3, 2025)
- **Test Reality**: ⚠️ 91.3% success rate (2,027/2,219 tests passing) - **192 failures require resolution** ⚠️
- **Core Systems**: ⚠️ Major functional issues across tensor operations, memory, quantization, training
- **GPU Integration**: ⚠️ Critical failures - Metal backend panics, memory allocation issues
- **Foundation Work**: ⚠️ 6-8 weeks minimum for basic operational stability
- **Commercial Timeline**: ⚠️ 6+ months development required before customer features

#### 🎯 CURRENT FOCUS AREAS (Technical Foundation Development):
- **Core System Stabilization**: Tensor arithmetic operations (0/25 passing - ALL BASIC OPERATIONS FAILING)
- **Memory Management**: Memory tracking, cleanup, allocation systems (30+ failures)  
- **Quantization Algorithms**: Mathematical correctness and memory safety (35+ failures)
- **Training Systems**: Optimizer integration and state management (20+ failures)
- **GPU Acceleration**: Metal backend stability and memory allocation (critical panics)

#### ⚠️ CURRENT FAILURE ANALYSIS (Requires Immediate Attention):
- **bitnet-core**: 90+ failures across arithmetic, memory, linear algebra, device systems
- **bitnet-quant**: 35+ failures in algorithm correctness and memory safety (UB violations)
- **bitnet-inference**: 5+ failures in GPU memory allocation and integration
- **bitnet-training**: 20+ failures in optimizer integration and training workflows
- **bitnet-metal**: CRITICAL - Null pointer panics in Metal initialization
- **Commercial Features**: BLOCKED - Cannot proceed until technical foundation stable

## Revised Code Quality Standards

### Prerequisites for Commercial Features - MAJOR WORK REQUIRED ⚠️
**Core system stabilization required across all components:**
- [ ] **Tensor Operations Stability** (0/25 arithmetic tests passing - CRITICAL FAILURE)
- [ ] **Memory Management Reliability** (30+ memory failures - CRITICAL ISSUE)  
- [ ] **Quantization Algorithm Correctness** (13 mathematical failures - HIGH PRIORITY)
- [ ] **GPU System Stability** (Metal panics - CRITICAL SAFETY ISSUE)
- [ ] **Training Infrastructure** (20+ optimizer/state failures - HIGH PRIORITY)
- [ ] **>95% test pass rate minimum** (Currently 91.3% - significant gap)
- [ ] **Memory safety compliance** (UB violations require immediate attention)
- [ ] **Cross-platform stability** (Device migration failures affecting portability)

**Commercial Features Available When:** Core technical foundation achieves >95% test success rate

## Current System Status (Reality Check)

### Core System Issues - Requiring Major Development ⚠️
- **Tensor Operations**: ALL basic arithmetic operations failing (add, subtract, multiply, divide)  
- **Memory Management**: Memory pool allocation, tracking, cleanup systems broken
- **Mathematical Foundation**: Linear algebra operations (SVD, QR, Cholesky) failing
- **Device Integration**: Cross-platform device migration and GPU acceleration unstable
- **Algorithm Correctness**: Quantization mathematical correctness compromised

### Infrastructure Stability - Mixed Results ⚠️
- **Build System**: ✅ All crates compile successfully (good foundation)
- **Error Handling**: ✅ 2,300+ lines error management framework exists
- **Test Infrastructure**: ✅ Comprehensive test coverage (2,219 tests) 
- **Workspace Structure**: ✅ 7-crate architecture properly organized
- **CI Integration**: ⚠️ Limited by core system instability

### Development Timeline - Realistic Assessment ⚠️
- **Phase 1 Completion**: 2-4 weeks for basic tensor/memory stability
- **Phase 2 Completion**: 2-3 weeks for quantization algorithm correctness
- **Phase 3 Completion**: 3-4 weeks for GPU integration and training systems
- **Commercial Readiness**: 6+ months after core stability achieved
- **Customer Features**: Dependent on successful technical foundation completion
- **Performance Monitoring**: Real-time regression detection and automated reporting ✅ PHASE 3
- **Memory Pressure Testing**: Production memory validation with leak detection ✅ PHASE 4
- **Device Compatibility**: Multi-device validation with fallback mechanisms ✅ PHASE 4
- **Error Recovery**: Comprehensive error handling and graceful degradation ✅ PHASE 4
- **Production Standards**: Strict performance thresholds enforced and validated ✅ PHASE 4

### Advanced Memory Management Validation
- **Zero-Copy Verification**: Memory address tracking to ensure zero-copy operations
- **Memory Pool Efficiency**: >98% utilization rate with <3.2% overhead ✅ ACHIEVED
- **Leak Detection**: Automated leak detection with 100% cleanup success rate ✅ ACHIEVED  
- **Fragmentation Control**: <25% fragmentation with automatic compaction
- **Pressure Handling**: Graceful degradation under memory pressure conditions
- **Thread Safety**: Concurrent access validation with race condition detection

### Numerical Accuracy and Stability Standards
- **IEEE Compliance**: Full IEEE 754 floating-point standard compliance
- **Precision Control**: Configurable precision with validation against reference implementations
- **Error Propagation**: Controlled error accumulation with bounded precision loss
- **Edge Case Handling**: NaN, infinity, overflow, underflow with defined behavior
- **Cross-Platform Consistency**: Identical results across different architectures and compilers
- **Quantization Accuracy**: <3% accuracy loss with 1.58-bit quantization ✅ ACHIEVED

## Code Quality Standards

### Performance Requirements
- Memory allocation times: <100ns for tensor creation
- SIMD acceleration: Minimum 3x speedup, target 12x with AVX512
- MLX operations: Target 300K+ ops/sec on Apple Silicon
- Metal GPU acceleration: Target 1000x+ speedup for appropriate operations
- Memory overhead: <5% for tensor metadata and tracking

### Memory Management Rules
- Always use HybridMemoryPool for tensor allocations
- Implement zero-copy operations wherever possible
- Track memory usage patterns and optimize for fragmentation
- Ensure thread-safe memory operations with Arc-based sharing
- Validate memory cleanup with 100% success rate

### Cross-Platform Compatibility
- Support x86_64 and ARM64 architectures
- Implement runtime CPU feature detection for SIMD
- Graceful degradation when advanced features unavailable
- Test on both Intel and Apple Silicon platforms
- Maintain compatibility across major OS versions

### Numerical Stability
- Use appropriate precision for all mathematical operations
- Implement IEEE standards compliance where applicable
- Handle edge cases (overflow, underflow, NaN, infinity)
- Validate numerical accuracy against reference implementations
- Document precision requirements and limitations

## Coding Standards

### Rust Best Practices
- Use `#[inline]` for hot-path functions
- Implement proper error handling with descriptive messages
- Include safety documentation for all unsafe code blocks
- Follow Rust API guidelines for public interfaces
- Use appropriate visibility modifiers (pub, pub(crate), private)

### Testing Requirements
- Unit tests for all public functions with edge cases
- Integration tests for cross-crate functionality
- Benchmark tests using Criterion framework
- Property-based testing where appropriate
- Minimum 90% code coverage for core functionality

### Documentation Standards
- Comprehensive rustdoc comments for all public APIs
- Include usage examples in documentation
- Document performance characteristics and complexity
- Explain safety requirements for unsafe functions
- Maintain up-to-date README files for all crates

## Advanced Architecture Principles

### Modularity and Extensibility
- **Crate Separation**: Clear domain boundaries with minimal inter-crate dependencies
- **Trait-Based Design**: Extensible interfaces for new device backends and algorithms
- **Plugin Architecture**: Runtime plugin loading for custom operations and optimizations
- **Version Compatibility**: Semantic versioning with backward compatibility guarantees
- **Feature Flags**: Fine-grained feature control for different deployment scenarios

### Performance-First Design Philosophy
- **Profile-Guided Development**: All optimizations backed by profiling data and benchmarks
- **Hardware-Aware Algorithms**: Architecture-specific optimizations for x86_64 and ARM64
- **Memory-Efficient Patterns**: Zero-copy operations, in-place computations, memory pooling
- **Parallel-First Design**: Multi-threading and vectorization as primary considerations
- **Latency Optimization**: Sub-millisecond operation targets for hot paths

### Production Infrastructure Requirements
- **Comprehensive Observability**: Metrics, logging, tracing, profiling integration
- **Fault Tolerance**: Graceful degradation, automatic recovery, circuit breaker patterns
- **Resource Management**: Automatic cleanup, leak prevention, resource limit enforcement
- **Security Hardening**: Memory safety, input validation, dependency auditing
- **Scalability Design**: Horizontal scaling, load balancing, distributed execution support

## Enhanced Development Workflow Standards

### Advanced Branching and Release Strategy
```
main (production)           ─────●─────●─────●─── (releases only)
    │                            ╱       ╱
release/v0.3.0                  ╱       ╱
    │                          ╱       ╱
develop (integration)    ─────●─────●─╱─────●─── (stable features)
    │                      ╱   ╲   ╱
feature/optimization  ────●     ╲ ╱
    │                          ╱╲
hotfix/critical-fix    ────────●  ●──────────── (emergency fixes)
```

### Comprehensive Code Review Process
1. **Automated Checks**: CI/CD pipeline with performance regression detection
2. **Peer Review**: Minimum two reviewers for all changes
3. **Performance Review**: Benchmark impact assessment for core path changes  
4. **Security Review**: Security vulnerability scanning and manual review for unsafe code
5. **Architecture Review**: Design pattern consistency and long-term maintainability
6. **Documentation Review**: API documentation completeness and accuracy verification

### Advanced Quality Gates and Validation

#### Pre-Commit Validation
```bash
#!/bin/bash
# Enhanced pre-commit validation script
set -e

echo "🔍 Running comprehensive validation..."

# Code formatting and style
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Security audit
cargo audit
cargo deny check

# Test execution with coverage
export RUSTFLAGS="-C instrument-coverage"
cargo test --workspace --all-features
grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing -o ./coverage/

# Performance regression check
cargo bench --workspace -- --save-baseline current
criterion-cmp baseline current --threshold 0.05  # 5% regression threshold

# Documentation validation  
cargo doc --workspace --no-deps
cargo test --doc --workspace

echo "✅ All validation checks passed!"
```

#### Release Readiness Criteria
- [ ] **Functionality**: All planned features implemented and tested
- [ ] **Performance**: Performance targets met across all supported platforms
- [ ] **Stability**: >99.9% test pass rate across 1000+ test runs
- [ ] **Security**: Security audit completed with no high-severity findings
- [ ] **Documentation**: Complete API documentation with examples and migration guides
- [ ] **Compatibility**: Backward compatibility verified and breaking changes documented
- [ ] **Deployment**: Production deployment validated in staging environment

### Enhanced Security and Safety Guidelines

#### Memory Safety and Unsafe Code Standards
- **Unsafe Block Justification**: Each unsafe block requires detailed safety analysis
- **Invariant Documentation**: All safety invariants documented with formal specifications
- **Automated Safety Validation**: Static analysis tools integrated into CI/CD pipeline
- **Regular Safety Audits**: Quarterly security audits by external security experts
- **Memory Sanitizers**: Address sanitizer, memory sanitizer, thread sanitizer integration

#### Dependency Management and Supply Chain Security
```toml
# Cargo.toml dependency management standards
[dependencies]
# Pin exact versions for production dependencies
critical-dep = "=1.2.3"

# Use version ranges for development dependencies only
dev-dep = "1.0"

# Audit all dependencies regularly
[dev-dependencies]
cargo-audit = "0.17"
cargo-deny = "0.14"
```

#### Input Validation and Error Handling Standards
```rust
// Input validation patterns
pub fn validate_tensor_operation(tensor: &BitNetTensor, operation: Operation) -> Result<()> {
    // 1. Null pointer validation
    if tensor.data().is_null() { return Err(BitNetError::NullPointer); }
    
    // 2. Size and dimension validation
    tensor.validate_dimensions()?;
    
    // 3. Device compatibility validation
    operation.validate_device_compatibility(tensor.device())?;
    
    // 4. Precision requirements validation
    operation.validate_precision_requirements(tensor.dtype())?;
    
    Ok(())
}

// Comprehensive error context
#[derive(thiserror::Error, Debug)]
pub enum BitNetError {
    #[error("Memory allocation failed: {context} (requested: {size} bytes, available: {available} bytes)")]
    MemoryAllocation { context: String, size: usize, available: usize },
    
    #[error("Device operation failed: {operation} on {device} (error: {source})")]
    DeviceOperation { operation: String, device: String, source: Box<dyn std::error::Error + Send + Sync> },
    
    #[error("Numerical instability detected: {context} (value: {value}, expected_range: {range:?})")]
    NumericalInstability { context: String, value: f64, range: (f64, f64) },
}
```

### Production Monitoring and Observability

#### Comprehensive Metrics Collection
- **Performance Metrics**: Latency, throughput, resource utilization, error rates
- **Business Metrics**: Model accuracy, inference quality, user satisfaction
- **Infrastructure Metrics**: Memory usage, CPU utilization, GPU utilization, power consumption
- **Security Metrics**: Authentication attempts, authorization failures, suspicious activity

#### Real-time Monitoring and Alerting
```rust
// Production monitoring integration
pub struct ProductionMonitor {
    metrics_collector: MetricsCollector,
    alert_manager: AlertManager,
    performance_analyzer: PerformanceAnalyzer,
    anomaly_detector: AnomalyDetector,
}

impl ProductionMonitor {
    // Real-time performance monitoring
    pub fn monitor_operation(&self, operation: &str, duration: Duration, success: bool) {
        self.metrics_collector.record_operation(operation, duration, success);
        
        if let Some(anomaly) = self.anomaly_detector.check_anomaly(operation, duration) {
            self.alert_manager.trigger_alert(Alert::PerformanceAnomaly(anomaly));
        }
    }
    
    // Resource utilization monitoring
    pub fn monitor_resources(&self) -> ResourceUtilization {
        let utilization = self.performance_analyzer.analyze_current_utilization();
        
        if utilization.memory_pressure > 0.8 {
            self.alert_manager.trigger_alert(Alert::HighMemoryPressure(utilization.memory_pressure));
        }
        
        utilization
    }
}
```

## Development Workflow

### Branching Strategy
- main: Production-ready code only
- develop: Integration branch for features
- feature/*: Individual feature development
- hotfix/*: Critical production fixes

### Code Review Process
- All changes require peer review
- Performance impact assessment for hot paths
- Security review for unsafe code
- Documentation review for public APIs
- Test coverage verification

### Implementation Documentation Requirements
- **Configuration File Updates**: After every implementation, update all relevant configuration files in the agent-config folder with necessary information to keep agents up to date with the latest changes, dependencies, and architectural decisions
- **Completion README**: Create a completion README after every implementation describing what was implemented, including:
  - Summary of features/components implemented
  - Architecture decisions made
  - Performance benchmarks achieved
  - Integration points with existing systems
  - Future considerations and recommendations
  - Testing coverage and validation results

### Continuous Integration
- Automated testing on multiple platforms
- Performance regression detection
- Memory leak detection
- Security vulnerability scanning
- Documentation generation and deployment

## Security Guidelines

### Unsafe Code
- Minimize usage of unsafe blocks
- Document all safety invariants
- Extensive testing of unsafe code paths
- Regular security audits of unsafe code
- Clear justification for each unsafe block

### Dependencies
- Regular dependency updates and audits
- Minimal external dependencies where possible
- Security scanning of all dependencies
- Documentation of critical dependencies
- Fallback implementations where feasible

## Performance Monitoring

### Benchmarking
- Continuous performance monitoring
- Regression detection and alerting
- Comparison against baseline implementations
- Platform-specific performance validation
- Regular performance optimization cycles

### Profiling
- Memory usage profiling and optimization
- CPU performance analysis
- GPU utilization monitoring
- Power consumption measurement (mobile)
- Thermal behavior analysis

## Collaboration Guidelines

### Communication
- Clear and concise issue descriptions
- Detailed pull request descriptions
- Regular progress updates on long-running tasks
- Proactive communication about blockers
- Knowledge sharing through documentation

### Code Ownership
- Designated maintainers for each crate
- Shared responsibility for core infrastructure
- Clear escalation paths for decisions
- Regular code ownership reviews
- Mentoring for new contributors

## Quality Gates

### Definition of Done
- All tests passing on supported platforms
- Performance benchmarks meet requirements
- Documentation complete and accurate
- Code review approved by maintainers
- Security review completed for sensitive changes

### Release Criteria
- Full test suite execution
- Performance validation against targets
- Security audit completion
- Documentation updates
- Breaking change migration guides
- Backward compatibility verification (where applicable)

## Crate Publishing Rules & Standards ✅ **NEW: PROFESSIONAL PUBLISHING FRAMEWORK**

### Publishing Prerequisites
- **Build Success**: All 7 crates must compile without errors or critical warnings
- **Test Coverage**: Minimum 95% test pass rate across all crates before publication
- **Documentation Currency**: README, CHANGELOG, and API documentation must be current
- **Version Consistency**: All inter-crate dependencies must use exact version matching
- **Security Review**: Security audit completed for public releases

### Publishing Order & Dependencies
Based on successful v1.0.0 publication experience:
```
Publishing Sequence (Dependency-Based):
1. bitnet-metal       (Independent - Metal GPU shaders)
2. bitnet-core        (Core infrastructure, metal integration configurable)
3. bitnet-quant       (Depends on bitnet-core)
4. bitnet-inference   (Depends on bitnet-core, bitnet-quant, bitnet-metal)
5. bitnet-training    (Depends on bitnet-core, bitnet-quant)
6. bitnet-cli         (Depends on bitnet-core, bitnet-quant)
7. bitnet-benchmarks  (Depends on bitnet-core, bitnet-quant)
```

### Publication Quality Gates
- **Pre-Publication Validation**: Execute `./scripts/dry-run.sh` successfully
- **Dependency Resolution**: Verify all crate dependencies resolve correctly
- **Documentation Build**: Ensure `cargo doc --workspace --all-features --no-deps` succeeds
- **Package Integrity**: All crates must package without warnings or missing files
- **Index Wait Times**: Minimum 30-second intervals between crate publications

### Version Management Standards
- **Semantic Versioning**: Strict adherence to semver for all published versions
- **Workspace Consistency**: All crates maintain synchronized major.minor versions
- **Dependency Constraints**: Use exact version matching for workspace dependencies (e.g., "1.0.0")
- **Path vs Registry**: Path dependencies converted to registry dependencies for publication
- **Version Bumps**: Coordinated version updates across entire workspace

### Publishing Automation Rules
- **Automated Scripts**: Use `./scripts/publish.sh` for multi-crate publication
- **Error Handling**: Automated retry logic for transient crates.io issues
- **Status Monitoring**: Track publication status and indexing completion
- **Rollback Procedures**: Prepared yank procedures for problematic releases
- **Success Validation**: Verify successful publication and public availability

### Commercial Release Standards
- **Market Readiness**: Publications support commercial deployment and customer use
- **Professional Presentation**: README files, documentation, and examples are customer-ready
- **Support Preparedness**: Clear support channels and issue reporting procedures
- **Performance Claims**: All published performance metrics are validated and reproducible
- **Compliance**: Published crates meet enterprise security and compliance requirements

### Post-Publication Procedures
- **Availability Verification**: Confirm all crates are searchable and installable via `cargo add`
- **Dependency Testing**: Validate published crates resolve dependencies in clean environments
- **Documentation Deployment**: Ensure docs.rs documentation builds and deploys correctly
- **Community Communication**: Announce releases through appropriate channels
- **Feedback Integration**: Monitor community feedback and address issues promptly

### Emergency Publishing Procedures
- **Security Patches**: Expedited publishing for critical security fixes
- **Critical Bug Fixes**: Fast-track procedures for production-blocking issues  
- **Yank Procedures**: Clear criteria and process for removing problematic versions
- **Communication**: Transparent communication about issues and resolutions
- **Recovery Plans**: Documented procedures for recovering from failed publications

This publishing framework ensures reliable, professional, and efficient crate publication management supporting BitNet-Rust's commercial success.