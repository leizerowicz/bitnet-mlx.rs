# BitNet-Rust Project Rules & Guidelines

## Project Overview
BitNet-Rust is a production-ready, high-performance implementation of BitNet neural networks featuring revolutionary 1.58-bit quantization, advanced memory management, comprehensive GPU acceleration, and enterprise-grade infrastructure.

## Development Phases
- **Phase 4**: Complete Tensor Operations (COMPLETED)
- **Phase 4.5**: Production Completion (IN PROGRESS - 95/100 score)
- **Phase 5**: Inference Engine & Training Infrastructure (READY TO START)

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

## Architecture Principles

### Modularity
- Clear separation of concerns across crates
- Well-defined interfaces between modules
- Minimal coupling between components
- Extensible design for future enhancements

### Performance First
- Profile-guided optimization decisions
- Benchmark-driven development approach
- Memory-efficient data structures and algorithms
- Hardware-aware optimization strategies

### Production Readiness
- Comprehensive error handling and recovery
- Thread-safe operations for concurrent workloads
- Resource cleanup and leak prevention
- Graceful degradation under resource constraints

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