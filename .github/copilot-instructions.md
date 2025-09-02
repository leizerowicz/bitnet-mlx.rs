# BitNet-Rust Copilot Instructions

> **Last Updated**: September 1, 2025 - Streamlined Workflow with Context Reference System

## Project Overview

BitNet-Rust is a high-performance implementation of BitNet neural networks featuring revolutionary 1.58-bit quantization, advanced memory management, comprehensive GPU acceleration, and production-ready testing infrastructure. This document establishes the workflow for using the agent configuration system with SPARC methodology to complete project tasks.

## Primary Workflow: Orchestrator → Context → SPARC → Agent-Hooks

### Step 1: Understand the Workflow
Start by understanding this workflow sequence:
1. **First**: Consult the orchestrator for task routing and current project status
2. **Then**: Select all relevant context documents needed for the task
3. **Next**: Review the user request (which will come from the backlog)
4. **Finally**: Use SPARC methodology cycles and phases to complete the task
5. **If needed**: Use agent-hooks for enhanced coordination when the orchestrator isn't sufficient

### Step 2: Start with Orchestrator
Always begin task planning by consulting `agent-config/orchestrator.md`. The orchestrator provides:
- Current project phase and priorities  
- Task routing matrix for selecting appropriate specialist agents
- Quality gates and coordination protocols
- Commercial readiness status and sprint objectives

### Step 3: Select Relevant Context Documents
Based on the task and orchestrator guidance, reference these context sources:

**Agent Configuration Files** (`agent-config/`):
- 25+ specialist agent configurations with role descriptions and coordination protocols
- Task routing matrix and agent selection guidance
- Development commands, rules, and best practices

**Project Foundation Documents** (`project-start/step_1/`):
- `BACKLOG.md`: Complete feature backlog with priorities and user requests
- `IMPLEMENTATION_GUIDE.md`: Technical architecture and development strategy  
- `RISK_ASSESSMENT.md`: Risk analysis and mitigation strategies
- `FILE_OUTLINE.md`: Current and planned project structure

**SPARC Implementation Guides** (`project-start/step_2/`):
- `sparc_methodology_guide.md`: Complete SPARC implementation instructions
- Individual phase guides (PHASE_1 through PHASE_5) for detailed methodology

### Step 4: Review User Request
The user request will come from the backlog (`project-start/step_1/BACKLOG.md`) and will specify:
- Task priority and complexity
- Epic → Story → Task breakdown
- Acceptance criteria and success metrics
- Dependencies and timeline considerations

### Step 5: Apply SPARC Methodology
Follow the 5-phase SPARC methodology for all implementation tasks:

1. **S - Specification**: Define requirements, constraints, and success criteria
2. **P - Pseudocode**: Design algorithms and system logic  
3. **A - Architecture**: Plan component structure and interactions
4. **R - Refinement**: Optimize and iterate the implementation
5. **C - Completion**: Final implementation with comprehensive testing

Detailed guidance for each phase is available in `project-start/step_2/sparc_methodology_guide.md`.

### Step 6: Use Agent-Hooks for Enhanced Coordination
If the orchestrator guidance isn't sufficient for complex coordination, consult `agent-config/agent-hooks.md` for:
- Enhanced lifecycle management and workflow automation
- Inter-agent coordination and seamless handoffs  
- Quality assurance hooks and validation gates
- Error detection, escalation, and recovery procedures

## Completion Documentation Protocol

All completed work must document:
1. **What Was Done**: Implementation details and changes
2. **Why It Was Done**: Business justification and requirements fulfilled
3. **How It Was Done**: Technical approach and decisions
4. **SPARC Phase Completion**: Which phases completed and artifacts created
5. **Agent Collaboration**: Which configurations used and coordination
6. **Testing & Validation**: Test results and quality assurance
7. **Future Implications**: Dependencies resolved and next steps enabled


