# Project-Start Enhanced CLI Commands

This directory contains CLI commands that integrate AI-powered automation with the Project-Start workflow following the **GitHub Spec-Kit pattern** for intelligent document generation.

## 🤖 AI Integration Features

The CLI now supports multiple AI assistants following the GitHub Spec-Kit pattern:

- **🤖 GitHub Copilot**: Native VS Code integration for intelligent document generation
- **🧠 Claude Code**: Command-line integration with Claude for enhanced specifications  
- **✨ Gemini CLI**: Google Gemini integration for comprehensive project planning
- **🎯 Fallback Generation**: Intelligent templates when AI tools are unavailable

### AI Assistant Selection

```bash
# Use GitHub Copilot (default)
python3 project_start_cli.py start "My Project" --ai copilot

# Use Claude Code
python3 project_start_cli.py start "My Project" --ai claude

# Use Gemini CLI
python3 project_start_cli.py start "My Project" --ai gemini

# Interactive selection
python3 project_start_cli.py start "My Project"
```

## Core Commands

### `/project-start-enhanced`
Master command that orchestrates the entire enhanced workflow from initial idea to deployment-ready system.

```bash
/project-start-enhanced "Build a task management system with real-time collaboration"
```

**What it does**:
1. Runs enhanced Step 1 (automated discovery with constitutional validation)
2. Runs enhanced Step 2 (constitutional SPARC with test-first planning)
3. Runs enhanced Step 3 (persistent context with automated memory)
4. Runs enhanced Step 4 (constitutional PACT with quality gates)

### `/enhance-step-1` 
Applies Spec-Kit's `/specify` automation to Project-Start's discovery phase while generating all required Step 1 documents. **NEW**: Now supports existing projects with established architecture.

#### New Project Mode
```bash
/enhance-step-1 "Real-time chat application with message history and user presence"
```

#### Existing Project Mode
```bash
# Automatically detects existing project structure
/enhance-step-1 "analysis-of-existing-project"

# Force existing project mode
/enhance-step-1 "existing-project" --existing-project
```

**What it does**:
1. **Project Detection**: Scans current directory for existing project indicators (package.json, requirements.txt, README.md, etc.)
2. **Comprehensive File Analysis**: Analyzes all MD files, code files, and configuration files
3. **Smart File Categorization**: Identifies documentation, code, and config files with focus suggestions
4. **Multi-Mode Operation**: 
   - **New Project**: Traditional interactive questionnaire approach
   - **Existing Project**: File analysis with optional supplementary questions
5. **Flexible Analysis Options**:
   - Automatic analysis of suggested files
   - User-selected file focus
   - Hybrid approach with additional context gathering
6. **Constitutional Integration**: Applies validation gates to both new and existing projects
7. **Memory Initialization**: Sets up persistent context for discovered/created projects

**Existing Project Features**:
- Detects 10+ project types (Node.js, Python, Java, Go, Rust, etc.)
- Extracts project metadata from package.json, pyproject.toml, README files
- Analyzes documentation patterns and technical keywords
- Provides comprehensive file categorization and suggestions
- Creates Project-Start documentation while preserving original structure

**Output Structure**:
```
specs/001-project-name/
├── BACKLOG.md (prioritized features and user stories)
├── IMPLEMENTATION_GUIDE.md (technical approach and constraints)
├── RISK_ASSESSMENT.md (risks and mitigation strategies)
├── FILE_OUTLINE.md (project structure and organization)
├── constitutional_validation.md (compliance checkpoints)
├── clarification_needed.md (areas requiring further specification)
└── EXISTING_PROJECT_ANALYSIS.md (analysis summary - for existing projects)
```

### `/enhance-step-2`
Applies Spec-Kit's `/plan` automation to Project-Start's SPARC methodology with constitutional governance.

```bash
/enhance-step-2 --tech-stack "React, Node.js, PostgreSQL" --architecture "microservices"
```

**What it does**:
1. Reads Step 1 outputs and applies constitutional SPARC methodology
2. Uses spec-kit's planning templates with test-first development mandates
3. Generates detailed implementation plans with constitutional compliance
4. Creates research documents for technology validation
5. Establishes quality gates for implementation phase
6. Updates project memory with technical decisions

**Output Structure**:
```
specs/001-chat-application/sparc/
├── SPARC_SPECIFICATION.md (formal requirements with constitutional compliance)
├── SPARC_PSEUDOCODE.md (algorithm design and logic flow)
├── SPARC_ARCHITECTURE.md (system design and component interactions)
├── SPARC_REFINEMENT.md (testing strategy and quality improvements)
└── SPARC_COMPLETION.md (deployment and maintenance procedures)
```

### `/enhance-step-3`
Creates persistent context systems that eliminate repetitive copilot instructions.

```bash
/enhance-step-3
```

**What it does**:
1. Analyzes all project artifacts to build comprehensive context
2. Creates constitutional copilot instructions with persistent memory
3. Establishes automated context updating systems
4. Configures expert domain specialization with constitutional awareness
5. Sets up multi-agent coordination protocols
6. Eliminates need for repetitive copilot management

**Output Structure**:
```
agent_config/
├── .github/copilot-instructions.md (constitutional, autonomous agent context)
├── project_memory.md (current state and decisions)
├── expert_domains/
│   ├── architecture_expert.md (constitutional system design)
│   ├── testing_expert.md (test-first development specialist)
│   ├── database_expert.md (data modeling with constitutional constraints)
│   └── api_expert.md (interface design with governance)
└── coordination_protocols.md (multi-agent governance)
```

### `/enhance-step-4`
Integrates constitutional governance with PACT multi-agent coordination.

```bash
/enhance-step-4
```

**What it does**:
1. Applies constitutional principles to PACT framework components
2. Creates automated quality gates for multi-agent systems
3. Establishes governance-aware agent coordination
4. Implements continuous constitutional compliance monitoring
5. Sets up automated validation and feedback loops
6. Enables self-managing agent ecosystems

### `/validate-constitution`
Checks constitutional compliance across all project artifacts.

```bash
/validate-constitution --full-audit
```

**What it does**:
1. Validates all artifacts against constitutional principles
2. Checks quality gate compliance across workflow steps
3. Audits memory system health and context continuity
4. Reports constitutional violations and suggests corrections
5. Updates compliance metrics and trends
6. Provides constitutional improvement recommendations

## Integration Commands

### `/sync-context`
Synchronizes project context across all memory systems and agent configurations.

```bash
/sync-context
```

**Usage**: Run after significant project changes to ensure all agents have updated context.

### `/constitutional-audit`
Comprehensive constitutional compliance review.

```bash
/constitutional-audit --generate-report
```

**Usage**: Regular governance health checks and compliance reporting.

### `/memory-health-check`
Validates integrity of persistent memory systems.

```bash
/memory-health-check --auto-repair
```

**Usage**: Ensures context continuity and prevents memory system degradation.

## Usage Examples

### Complete Enhanced Workflow
```bash
# Start with an idea
/project-start-enhanced "Build a collaborative document editor with real-time sync"

# The system will:
# 1. Generate comprehensive specifications (Step 1)
# 2. Create constitutional implementation plans (Step 2)  
# 3. Establish persistent context systems (Step 3)
# 4. Deploy governed multi-agent coordination (Step 4)
```

### Step-by-Step Enhancement
```bash
# Enhance discovery phase
/enhance-step-1 "Social media analytics dashboard"

# Review and refine specifications, then enhance planning
/enhance-step-2 --tech-stack "Python, FastAPI, React, PostgreSQL"

# Create persistent context
/enhance-step-3

# Deploy constitutional governance
/enhance-step-4

# Validate overall compliance
/validate-constitution --full-audit
```

### Existing Project Integration
```bash
# For projects with existing Step 1 documents
/enhance-step-2 --existing-project --validate-step-1

# For projects needing context refresh
/sync-context --rebuild-memory

# For constitutional compliance updates
/constitutional-audit --update-compliance
```

## Configuration

### Constitutional Settings
Configure constitutional enforcement levels:
```bash
/configure-constitution --enforcement-level strict
/configure-constitution --quality-gates mandatory
/configure-constitution --memory-persistence always-on
```

### Memory System Settings
Configure persistent context behavior:
```bash
/configure-memory --retention-policy comprehensive
/configure-memory --sync-frequency real-time
/configure-memory --context-depth full-project-history
```

### Integration Settings
Configure spec-kit integration behavior:
```bash
/configure-integration --spec-kit-mode enhanced-project-start
/configure-integration --template-fusion enabled
/configure-integration --quality-gates automatic
```

## Troubleshooting

### Common Issues

**Context not persisting across sessions**
```bash
/memory-health-check --auto-repair
/sync-context --force-rebuild
```

**Constitutional violations in generated artifacts**
```bash
/validate-constitution --fix-violations
/constitutional-audit --apply-corrections
```

**Spec-kit integration conflicts**
```bash
/configure-integration --resolve-conflicts
/template-health-check --validate-fusion
```

### Debug Mode
Enable detailed logging for troubleshooting:
```bash
export PROJECT_START_DEBUG=true
/project-start-enhanced --debug "your project description"
```

### Support Commands
```bash
/help-constitution    # Constitutional principle guidance
/help-integration     # Spec-kit integration help
/help-memory-systems  # Persistent context guidance
/help-troubleshooting # Common issue resolution
```

## Implementation Notes

### Backward Compatibility
- All enhanced commands preserve original Project-Start workflow
- Original step-by-step approach remains available
- Enhanced features are opt-in, not mandatory
- Existing projects can be gradually enhanced

### Performance Considerations
- Memory systems use efficient storage and retrieval
- Context synchronization is optimized for minimal overhead
- Constitutional validation uses caching for repeated checks
- Quality gates are parallelized where possible

### Security and Privacy
- Memory systems respect data privacy requirements
- Constitutional compliance includes security validation
- Agent coordination includes access control mechanisms
- Context data is encrypted and access-controlled

### Future Enhancements
- Machine learning integration for improved context prediction
- Advanced constitutional reasoning for complex scenarios
- Multi-project memory sharing for organizational learning
- Integration with additional development methodologies