# BitNet-Rust Copilot Instructions

## Project Overview

BitNet-Rust is a high-performance implementation of BitNet neural networks featuring 1.58-bit quantization, memory management, GPU acceleration, and testing infrastructure. The project is currently in the **Inference Ready Phase** with 99.8% test success rate, following the COMPREHENSIVE_TODO.md roadmap.

## Agent Configuration System - Orchestrator-Driven Multi-Agent Workflow

This project uses a comprehensive agent configuration system in the `agent-config/` directory to coordinate all development activities. **THE ORCHESTRATOR IS THE CENTRAL COMMAND** that routes all work and manages all specialist coordination.

### 🎯 MANDATORY ORCHESTRATOR-FIRST WORKFLOW

**ALWAYS START WITH THE ORCHESTRATOR** - This is non-negotiable for any development work:

#### **Step 1: ORCHESTRATOR CONSULTATION (REQUIRED)**
Before doing ANY work, **ALWAYS read `agent-config/orchestrator.md` FIRST** to:
- **Understand current project context** and priorities (COMPREHENSIVE_TODO.md alignment)
- **Get proper task routing** to appropriate specialist agents
- **Identify multi-agent coordination needs** for complex tasks
- **Access workflow management** and quality gate requirements
- **Integrate with agent hooks system** for automated lifecycle management

#### **Step 2: ORCHESTRATOR ROUTING DECISION**
The orchestrator will route you to appropriate specialists using this framework:
- **Primary Agent Selection**: Based on task domain and complexity
- **Secondary Agent Coordination**: For cross-domain or complex requirements
- **Quality Gate Assignment**: Validation and review requirements
- **Workflow Coordination**: Timeline and dependency management

#### **Step 3: SPECIALIST CONSULTATION (ORCHESTRATOR-GUIDED)**
After orchestrator routing, consult the specific specialist agents identified:
- **Read specialist agent configs** for domain-specific context and expertise
- **Understand agent intersections** and collaboration patterns
- **Follow established workflows** and handoff procedures
- **Maintain orchestrator coordination** throughout the work

### Agent Configuration Hierarchy & Orchestrator Authority

#### 🎯 **Central Command (ALWAYS START HERE)**
- **`orchestrator.md`** - **MANDATORY FIRST STOP** - Central coordination, agent routing, workflow management, project context

#### Core Technical Specialists (Orchestrator-Routed)
- **`architect.md`** - System architecture and design (intersects with: research, security, code)
- **`code.md`** - Primary development and implementation (intersects with: rust_best_practices, debug, test_utilities)
- **`debug.md`** - Problem resolution and troubleshooting (intersects with: code, error_handling, test_utilities)
- **`inference_engine_specialist.md`** - ML inference and model execution (intersects with: performance_engineering, api_development, code)
- **`performance_engineering_specialist.md`** - Optimization and acceleration (intersects with: inference_engine, rust_best_practices, architect)
- **`rust_best_practices_specialist.md`** - Code quality and Rust excellence (intersects with: code, security_reviewer, performance_engineering)
- **`test_utilities_specialist.md`** - Testing infrastructure and validation (intersects with: error_handling, debug, truth_validator)
- **`error_handling_specialist.md`** - Error management and resilience (intersects with: debug, rust_best_practices, test_utilities)

#### Quality & Coordination Specialists
- **`truth_validator.md`** - Quality assurance and validation (intersects with: ALL agents for quality gates)
- **`security_reviewer.md`** - Security and safety analysis (intersects with: rust_best_practices, architect, error_handling)
- **`documentation_writer.md`** - Technical documentation (intersects with: ask, api_development, ALL specialists)
- **`ask.md`** - User interaction and requirements (intersects with: documentation_writer, customer_success, ui_ux)

#### Project Management (Orchestrator-Coordinated)
- **`development_phase_tracker.md`** - Timeline and milestone tracking (intersects with: comprehensive_todo_manager, orchestrator, truth_validator)
- **`comprehensive_todo_manager.md`** - Roadmap management (intersects with: development_phase_tracker, ALL specialists)
- **`publishing_expert.md`** - Release management (intersects with: truth_validator, documentation_writer, devops_infrastructure)

#### Commercial & Business Specialists
- **`saas_platform_architect.md`** - SaaS platform design (intersects with: api_development, devops_infrastructure, security_reviewer)
- **`api_development_specialist.md`** - API development (intersects with: saas_platform, inference_engine, documentation_writer)
- **`business_intelligence_specialist.md`** - Business analytics (intersects with: customer_success, performance_engineering, api_development)
- **`customer_success_specialist.md`** - Customer success (intersects with: business_intelligence, ask, ui_ux)
- **`devops_infrastructure_specialist.md`** - DevOps and infrastructure (intersects with: saas_platform, publishing_expert, security_reviewer)
- **`ui_ux_development_specialist.md`** - Frontend and UX (intersects with: customer_success, api_development, ask)

#### Support & Configuration
- **`agent-hooks.md`** - Agent coordination system (intersects with: ALL agents, orchestrator, truth_validator)
- **`project_commands_config.md`** - Build systems and commands (intersects with: devops_infrastructure, code, test_utilities)
- **`project_rules_config.md`** - Standards and guidelines (intersects with: rust_best_practices, variable_matcher, ALL specialists)
- **`variable_matcher.md`** - Code consistency (intersects with: rust_best_practices, code, project_rules_config)
- **`project_research.md`** - Research and innovation (intersects with: architect, performance_engineering, security_reviewer)

### Multi-Agent Coordination Patterns (Orchestrator-Managed)

The orchestrator manages several coordination patterns for different task types:

#### **Single-Agent Tasks (Orchestrator Oversight)**
```
Simple tasks → Primary specialist + orchestrator coordination
Quality validation → truth_validator.md review
Documentation → documentation_writer.md if user-facing
```

#### **Multi-Agent Collaboration (Orchestrator Coordination)**
```
Complex features → Primary + Secondary specialists + orchestrator management
Cross-domain tasks → Multiple specialists + daily sync + orchestrator coordination
Critical changes → Full review chain + architect + security + orchestrator validation
```

#### **Emergency Response (Orchestrator Escalation)**
```
Critical bugs → debug.md + immediate escalation + orchestrator resource coordination
Performance issues → debug.md + performance_engineering + orchestrator timeline management
Security incidents → security_reviewer + architect + orchestrator incident management
```

### Workflow Decision Framework (Orchestrator-Defined)

**The orchestrator uses this decision matrix for task routing:**

| Task Type | Primary Agent | Secondary Agents | Quality Gates |
|-----------|---------------|------------------|---------------|
| Feature Development | `code.md` | `rust_best_practices_specialist.md`, `test_utilities_specialist.md` | Code quality + test coverage |
| Debugging | `debug.md` | `code.md`, `error_handling_specialist.md` | Root cause + fix validation |
| Architecture | `architect.md` | `project_research.md`, `security_reviewer.md` | Design review + security validation |
| Performance | `performance_engineering_specialist.md` | `inference_engine_specialist.md`, `code.md` | Benchmark validation + optimization review |
| Inference Features | `inference_engine_specialist.md` | `api_development_specialist.md`, `performance_engineering_specialist.md` | ML accuracy + API usability |
| Documentation | `documentation_writer.md` | `ask.md`, domain specialists | User testing + accuracy validation |
| Testing | `test_utilities_specialist.md` | `error_handling_specialist.md`, `debug.md` | Coverage + edge case validation |
| Security | `security_reviewer.md` | `rust_best_practices_specialist.md`, `architect.md` | Vulnerability assessment + safe patterns |
| Release | `publishing_expert.md` | `truth_validator.md`, `documentation_writer.md` | Comprehensive validation + documentation |

### Agent Intersection Understanding (Orchestrator-Defined)

**Every agent understands their intersections with other agents:**

- **Code Development** intersects with Rust Best Practices (quality), Debug (fixes), Test Utilities (validation)
- **Debug** intersects with Code (implementation), Error Handling (resilience), Test Utilities (reproduction)
- **Inference Engine** intersects with Performance Engineering (optimization), API Development (interfaces), Code (implementation)
- **Performance Engineering** intersects with Inference Engine (inference speed), Rust Best Practices (efficiency), Architect (system design)
- **Security Reviewer** intersects with Rust Best Practices (safety), Architect (security design), Error Handling (resilience)

**And many more intersections explicitly defined in each agent config.**

### Current Priority (Week 1) - Orchestrator-Managed

**🎯 Task 1.0.1**: Fix single failing memory tracking test to achieve 100% test success (99.8% → 100%)
- **Orchestrator Routing**: `debug.md` (primary) + `code.md` (implementation) + `test_utilities_specialist.md` (validation)
- **Location**: `bitnet-core/tests/memory_tracking_tests.rs:106`  
- **Issue**: Memory pool tracking integration not configured
- **Effort**: 2-4 hours
- **Next**: Epic 2 - Inference Ready implementation (Weeks 2-6)

## Workflow Rules - Orchestrator-Driven

1. **🎯 ALWAYS START WITH ORCHESTRATOR** - Read `orchestrator.md` first for every task
2. **Follow orchestrator routing** - Use the orchestrator's agent selection matrix
3. **Maintain orchestrator coordination** - Keep orchestrator informed of progress and handoffs
4. **Respect agent intersections** - Follow established collaboration patterns between agents
5. **Use quality gates** - Apply orchestrator-defined validation requirements
6. **Follow current phase** - Align with COMPREHENSIVE_TODO.md priorities as managed by orchestrator
7. **Execute user requests exactly** - Within the orchestrator's workflow framework
8. **Stop when complete** - When orchestrator-defined success criteria are met
9. **Be direct and clear** - Provide straightforward responses following orchestrator guidance
10. **Use available tools** - Leverage tools efficiently within orchestrator's workflow framework

## Orchestrator-Coordinated Workflow

1. **🎯 START WITH ORCHESTRATOR** - Always read `orchestrator.md` first
2. **Get orchestrator routing** - Use agent selection matrix for appropriate specialists
3. **Follow agent intersections** - Consult routed specialists with understanding of their collaboration patterns
4. **Execute with coordination** - Perform work following orchestrator workflow management
5. **Report through orchestrator** - Confirm completion and coordinate next steps through orchestrator

## When to Stop - Orchestrator-Defined Criteria

- Task completed successfully according to orchestrator quality gates
- User request fulfilled within orchestrator workflow context
- No further action required as determined by orchestrator coordination
- Clear completion criteria from orchestrator workflow met
- Current phase priorities defined by orchestrator respected

## Project Context Usage - Orchestrator-Managed

- **Current Status**: Foundation complete (99.8% test success), ready for inference implementation
- **Active Roadmap**: COMPREHENSIVE_TODO.md managed through orchestrator and specialist coordination
- **Agent Coordination**: ALL coordination managed through orchestrator workflow
- **Quality Gates**: Orchestrator-defined excellence standards while advancing through development phases
- **Workflow Management**: ALL development activities coordinated through orchestrator's multi-agent management system


