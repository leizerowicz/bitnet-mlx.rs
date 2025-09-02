# BitNet-Rust Copilot Instructions

> **Last Updated**: September 2, 2025 - Enhanced Workflow with Complete Context Integration and SPARC Document Management

## Project Overview

BitNet-Rust is a high-performance implementation of BitNet neural networks featuring revolutionary 1.58-bit quantization, advanced memory management, comprehensive GPU acceleration, and production-ready testing infrastructure. This document establishes the comprehensive workflow for using the agent configuration system with SPARC methodology to complete project tasks.

**Current Project Status**: Commercial Readiness Phase - Week 1, Technical Foundation Complete (100% critical test success rate achieved), Epic 1 Complete

## Primary Workflow: Orchestrator → Context → Task Analysis → SPARC Document Management → Implementation

### Step 1: Start with Orchestrator
**ALWAYS** begin task planning by consulting `agent-config/orchestrator.md`. The orchestrator automatically integrates with agent-hooks and provides:
- Current project phase and priorities (Commercial Readiness Phase - Week 1)
- Task routing matrix for selecting appropriate specialist agents
- Quality gates and coordination protocols with automated hook integration
- Commercial readiness status and sprint objectives
- Enhanced lifecycle management through built-in agent-hooks system

### Step 2: Load Complete Context Foundation
Based on the task and orchestrator guidance, reference ALL relevant context sources:

#### A. Agent Configuration Files (`agent-config/`)
**Required Reading**: Always consult relevant specialist agents:
- `orchestrator.md`: Project coordination and current status
- `agent-hooks.md`: Automated lifecycle management and quality assurance
- Task-specific specialists (e.g., `code.md`, `debug.md`, `test_utilities_specialist.md`)
- `project_commands_config.md` and `project_rules_config.md`: Development standards

#### B. Project Foundation Documents (`project-start/step_1/`) - **MANDATORY INTEGRATION**
**All documents must be consulted for complete context**:
- `BACKLOG.md`: Complete feature backlog with current priorities and user requests
- `IMPLEMENTATION_GUIDE.md`: Technical architecture, technology stack, and development strategy
- `RISK_ASSESSMENT.md`: Risk analysis, mitigation strategies, and technical constraints
- `FILE_OUTLINE.md`: Current and planned project structure, crate organization
- `README.md`: Project overview and setup instructions

#### C. SPARC Implementation Framework (`project-start/step_2/`)
**Complete methodology guidance**:
- `sparc_methodology_guide.md`: Complete SPARC implementation instructions and agent guidance
- `SPARC_PHASE_1_SPECIFICATION.md` through `SPARC_PHASE_5_COMPLETION.md`: Detailed phase templates

### Step 3: Analyze Current Task and Project State
Before proceeding with implementation:

1. **Identify Current Task**: Determine the active task from the backlog (Epic → Story → Task breakdown)
2. **Check Project State**: Verify current technical status (tests passing, build status, dependencies)
3. **Assess Task Requirements**: Review acceptance criteria, success metrics, and dependencies
4. **Validate Context Alignment**: Ensure understanding matches current project reality

### Step 4: SPARC Document Management (CRITICAL)
**MANDATORY**: For every task, check and manage SPARC phase documents at project root:

#### A. Check Existing SPARC Documents
Look for existing SPARC documents in the format: `SPARC_[EPIC]_[STORY]_[PHASE].md`
- Example: `SPARC_STORY_2_0_SPECIFICATION.md` for Epic 2, Story 2.0, Specification phase
- Check all 5 phases: SPECIFICATION, PSEUDOCODE, ARCHITECTURE, REFINEMENT, COMPLETION

#### B. Validate Document Alignment with Current State
For existing SPARC documents:
1. **Verify Currency**: Ensure documents match current project state and requirements
2. **Check Technical Accuracy**: Validate against current codebase, test results, and architecture
3. **Assess Completeness**: Ensure all required sections are complete and accurate
4. **Review Dependencies**: Confirm document assumptions match current project status

#### C. Create or Update SPARC Documents as Needed
When SPARC documents are missing, outdated, or incomplete:
1. **Create Missing Documents**: Generate complete SPARC phase documents following templates
2. **Update Outdated Content**: Revise existing documents to match current project state
3. **Enhance Incomplete Sections**: Fill in missing information using current context
4. **Maintain Document Quality**: Ensure comprehensive coverage following SPARC methodology

#### D. SPARC Document Integration with Step 1 Foundation
**CRITICAL**: All SPARC documents MUST integrate information from Step 1 documents:

**In Specification Phase**:
```markdown
## Existing Documents Review
- **BACKLOG.md**: [Specific requirements and user stories from backlog]
- **IMPLEMENTATION_GUIDE.md**: [Relevant technical architecture and constraints]
- **RISK_ASSESSMENT.md**: [Associated risks and mitigation strategies]
- **FILE_OUTLINE.md**: [Project structure impacts and requirements]
```

**Throughout All Phases**: Reference Step 1 documents for:
- Technical constraints from IMPLEMENTATION_GUIDE.md
- Risk considerations from RISK_ASSESSMENT.md  
- Project structure requirements from FILE_OUTLINE.md
- User requirements and acceptance criteria from BACKLOG.md

### Step 5: Apply SPARC Methodology with Document Validation
Follow the 5-phase SPARC methodology ensuring document quality:

1. **S - Specification**: Define requirements using Step 1 context, create/update SPECIFICATION document
2. **P - Pseudocode**: Design algorithms and logic, create/update PSEUDOCODE document  
3. **A - Architecture**: Plan component structure, create/update ARCHITECTURE document
4. **R - Refinement**: Optimize implementation approach, create/update REFINEMENT document
5. **C - Completion**: Final implementation plan, create/update COMPLETION document

### Step 6: Implementation with Documentation Maintenance
During implementation:
1. **Follow SPARC Documents**: Use the created/validated SPARC documents as implementation guide
2. **Update Documents**: Keep SPARC documents current as implementation progresses
3. **Validate Against Step 1**: Ensure implementation aligns with Step 1 foundation documents
4. **Maintain Quality**: Apply agent-hooks quality assurance throughout process

## Example SPARC Document Management Workflow

### Current Task Example: Epic 2, Story 2.0 (Cleanup System Tests)
1. **Check Existing Documents**: Look for `SPARC_STORY_2_0_*.md` files at project root
2. **Validate Current State**: Verify documents match current test failures and project status
3. **Update/Create as Needed**: Ensure all 5 SPARC phases exist and are accurate
4. **Integrate Step 1 Context**: Each document references relevant Step 1 information
5. **Maintain Throughout Implementation**: Keep documents updated as work progresses

### Document Naming Convention
- **Format**: `SPARC_[EPIC_ID]_[STORY_ID]_[PHASE].md`
- **Phases**: SPECIFICATION, PSEUDOCODE, ARCHITECTURE, REFINEMENT, COMPLETION
- **Examples**: 
  - `SPARC_STORY_2_0_SPECIFICATION.md`
  - `SPARC_STORY_2_1_ARCHITECTURE.md`
  - `SPARC_EPIC_3_COMPLETION.md`

## Quality Assurance and Validation Protocol

### Document Quality Requirements
1. **Completeness**: All sections from SPARC templates must be addressed
2. **Currency**: Documents must reflect current project state and requirements
3. **Integration**: Must reference and integrate Step 1 foundation documents
4. **Technical Accuracy**: Must align with current codebase and test results
5. **Actionability**: Must provide clear guidance for implementation teams

### Validation Checklist for SPARC Documents
- [ ] Document exists for current task at appropriate phase
- [ ] "Existing Documents Review" section references all relevant Step 1 documents
- [ ] Technical constraints from IMPLEMENTATION_GUIDE.md are incorporated
- [ ] Risks from RISK_ASSESSMENT.md are addressed
- [ ] Project structure from FILE_OUTLINE.md is considered
- [ ] User requirements from BACKLOG.md are accurately reflected
- [ ] Current test status and project state are accurately represented
- [ ] Implementation approach aligns with commercial readiness phase goals

### Step 7: Agent-Hooks Integration and Final Validation
**MANDATORY FINAL STEP**: After completing all implementation work, apply agent-hooks lifecycle management:

#### A. Execute Agent-Hooks Quality Assurance
Using `agent-config/agent-hooks.md`, apply comprehensive quality assurance:
1. **Code Quality Validation**: Run automated quality checks and validation protocols
2. **Documentation Standards**: Ensure all documentation meets project standards
3. **Testing Requirements**: Validate test coverage and success criteria
4. **Integration Verification**: Confirm all components work together properly
5. **Performance Benchmarks**: Verify performance meets established criteria

#### B. Orchestrator-Coordinated Hook Execution
Use `agent-config/orchestrator.md` to coordinate final validation across specialist agents:
1. **Route to Quality Specialists**: Engage appropriate specialist agents for final review
2. **Cross-Agent Validation**: Ensure consistent quality across all work products
3. **Project Phase Alignment**: Confirm work aligns with current Commercial Readiness Phase
4. **Dependency Resolution**: Validate all dependencies are properly addressed
5. **Integration Protocol**: Execute proper integration with existing codebase

#### C. Required Hook Validations
Execute all applicable agent-hooks validations:
- **Pre-commit Hooks**: Code formatting, linting, basic tests
- **Integration Hooks**: Component integration and compatibility checks
- **Quality Gate Hooks**: Comprehensive testing and performance validation
- **Documentation Hooks**: Documentation completeness and accuracy
- **Security Hooks**: Security vulnerability scanning and compliance
- **Performance Hooks**: Benchmark validation and optimization verification

#### D. Final Orchestrator Review
Use orchestrator to coordinate final sign-off:
1. **Multi-Agent Review**: Engage relevant specialist agents for comprehensive review
2. **Quality Gate Approval**: Ensure all quality gates are met
3. **Project Status Update**: Update project status and phase tracking
4. **Next Steps Planning**: Coordinate transition to next project phase or tasks
5. **Completion Documentation**: Generate comprehensive completion report

## Completion Documentation Protocol

All completed work must document:
1. **What Was Done**: Implementation details and changes made
2. **Why It Was Done**: Business justification and requirements fulfilled from Step 1 context
3. **How It Was Done**: Technical approach and decisions guided by SPARC documents
4. **SPARC Phase Completion**: Which phases completed and artifacts created/updated
5. **Step 1 Integration**: How Step 1 foundation documents informed the work
6. **Agent Collaboration**: Which configurations used and coordination protocols followed
7. **Testing & Validation**: Test results and quality assurance performed
8. **SPARC Document Updates**: What SPARC documents were created, updated, or validated
9. **Agent-Hooks Execution**: Which hooks were executed and validation results
10. **Orchestrator Coordination**: How orchestrator coordinated final quality assurance
11. **Quality Gate Status**: All quality gates passed and validation completed
12. **Future Implications**: Dependencies resolved and next steps enabled
13. **Context Alignment**: Confirmation that work aligns with current project phase and priorities

## Troubleshooting Common Issues

### When SPARC Documents Don't Match Project State
1. **Assess the Gap**: Identify what has changed since document creation
2. **Update Documents**: Revise SPARC documents to match current reality
3. **Re-validate Requirements**: Check if original requirements are still valid
4. **Coordinate with Team**: Ensure updates align with broader project goals

### When Step 1 Documents Conflict with Current State
1. **Identify the Conflict**: Determine what has changed in project status
2. **Update Step 1 Documents**: May require updating BACKLOG.md or IMPLEMENTATION_GUIDE.md
3. **Cascade Updates**: Update any dependent SPARC documents
4. **Communicate Changes**: Document why changes were necessary

### When Context Is Incomplete or Outdated
1. **Research Current State**: Use tools to check current codebase, tests, and build status
2. **Update Documentation**: Bring all relevant documents current
3. **Fill Information Gaps**: Use available tools and project analysis to complete context
4. **Validate with Team**: Confirm understanding matches project reality

This enhanced copilot instructions framework ensures comprehensive context integration, proper SPARC document management, and alignment with the current commercial readiness phase of BitNet-Rust development.


