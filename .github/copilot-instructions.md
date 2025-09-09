# BitNet-Rust Copilot Instructions

## Project Overview

BitNet-Rust is a high-performance implementation of BitNet neural networks featuring 1.58-bit quantization, memory management, GPU acceleration, and testing infrastructure. The project is currently in the **Inference Ready Phase** with 99.8% test success rate, following the COMPREHENSIVE_TODO.md roadmap.

## Agent Configuration System

This project uses a comprehensive agent configuration system in the `agent-config/` directory to coordinate development activities. **ALWAYS** reference these files when working on specific aspects of the project:

### Core Agent Configs
- **`orchestrator.md`** - Overall project coordination and workflow management
- **`development_phase_tracker.md`** - Current development phase and milestone tracking
- **`comprehensive_todo_manager.md`** - COMPREHENSIVE_TODO.md roadmap management and priorities

### Specialist Agent Configs
- **`test_utilities_specialist.md`** - Test infrastructure and quality assurance
- **`debug.md`** - Debugging and problem resolution (current focus: Task 1.0.1)
- **`inference_engine_specialist.md`** - Inference implementation (Epic 2 focus)
- **`performance_engineering_specialist.md`** - Performance optimization and benchmarking
- **`rust_best_practices_specialist.md`** - Rust code quality and best practices

### How to Use Agent Configs

1. **Check Current Phase**: Always start by reading `development_phase_tracker.md` and `comprehensive_todo_manager.md` to understand current priorities
2. **Reference Specialist Configs**: Use relevant specialist configs for domain-specific work
3. **Follow Roadmap**: Align all work with COMPREHENSIVE_TODO.md priorities through the manager config
4. **Coordinate**: Use `orchestrator.md` for cross-component coordination

## Current Priority (Week 1)

**🎯 Task 1.0.1**: Fix single failing memory tracking test to achieve 100% test success (99.8% → 100%)
- **Location**: `bitnet-core/tests/memory_tracking_tests.rs:106`
- **Issue**: Memory pool tracking integration not configured
- **Effort**: 2-4 hours
- **Next**: Epic 2 - Inference Ready implementation (Weeks 2-6)

## Simple Rules

1. **Follow agent configs** - Check relevant agent-config files before starting work
2. **Respect current phase** - Focus on current COMPREHENSIVE_TODO.md priorities
3. **Follow user requests exactly** - Do what the user asks within the current roadmap context
4. **Stop when complete** - When a task is finished, stop and report completion
5. **Be direct and clear** - Provide straightforward responses without unnecessary complexity
6. **Use available tools** - Leverage the tools provided to accomplish tasks efficiently

## Basic Workflow

1. **Check agent configs** - Read relevant agent-config files for context and priorities
2. **Understand the request** - Read what the user wants within current phase context
3. **Gather necessary context** - Only collect information needed for the specific task
4. **Complete the task** - Do the work requested following agent config guidance
5. **Report completion** - Confirm the task is done and stop

## When to Stop

- Task is completed successfully
- User request has been fulfilled
- No further action is required
- Clear completion criteria have been met
- Current phase priorities are respected

## Project Context Usage

- **Current Status**: Foundation complete (99.8% test success), ready for inference implementation
- **Active Roadmap**: COMPREHENSIVE_TODO.md managed through agent configs
- **Agent Coordination**: Cross-specialist coordination through orchestrator and specialist configs
- **Quality Gates**: Maintain excellence while advancing through development phases


