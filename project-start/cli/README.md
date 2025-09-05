# Project-Start CLI

A simplified, interactive CLI for Project-Start's specification-driven development framework.

## 🚀 Quick Start

### Interactive Menu (Recommended)

```bash
# Launch interactive menu
python3 project_start_cli.py
```

The interactive menu guides you through:

1. **Step 1**: Discovery & Specification Generation
2. **Step 2**: SPARC Planning Methodology
3. **Step 3**: Context Systems Creation
4. **Step 4**: PACT Framework Deployment
5. **Complete Workflow**: All steps in sequence

### Direct Commands

```bash
# Run specific steps directly
python3 project_start_cli.py /enhance-step-1 "My Project Description"
python3 project_start_cli.py /enhance-step-2 --project-path specs/001-my-project
python3 project_start_cli.py /enhance-step-3 --project-path specs/001-my-project
python3 project_start_cli.py /enhance-step-4 --project-path specs/001-my-project

# Complete workflow
python3 project_start_cli.py /project-start-enhanced "My Project Description"

# Configure for nested projects
python3 project_start_cli.py /configure-project-root
```

## 📋 Available Commands

### `/enhance-step-1` - Discovery & Specification

Generates comprehensive project specifications:

- `BACKLOG.md` - User stories and prioritization
- `IMPLEMENTATION_GUIDE.md` - Technical guidance
- `RISK_ASSESSMENT.md` - Risk analysis
- `FILE_OUTLINE.md` - Project structure
- `constitutional_validation.md` - Compliance check
- `clarification_needed.md` - Questions for stakeholders

**Options:**
- `--existing-project` - Analyze existing codebase
- Description argument - Project description

### `/enhance-step-2` - SPARC Planning

Implements constitutional SPARC methodology:

- `sparc/SPARC_SPECIFICATION.md` - Formal requirements
- `sparc/SPARC_PSEUDOCODE.md` - Algorithm design
- `sparc/SPARC_ARCHITECTURE.md` - System design
- `sparc/SPARC_REFINEMENT.md` - Testing strategy
- `sparc/SPARC_COMPLETION.md` - Deployment procedures

**Options:**
- `--project-path` - Target project directory

### `/enhance-step-3` - Context Systems

Creates persistent context for AI agents:

- `.github/copilot-instructions.md` - Agent context
- `expert_files/` - Specialized expert contexts
- `agent_coordination.md` - Multi-agent protocols

**Options:**
- `--project-path` - Target project directory

### `/enhance-step-4` - PACT Framework

Deploys constitutional PACT framework:

- `AGENT_ECOSYSTEM_DESIGN.md` - Multi-agent architecture
- `COORDINATION_STRATEGY.md` - Agent coordination
- `COLLABORATIVE_WORKFLOWS.md` - Workflow definitions
- `AGENTIC_TESTING_FRAMEWORK.md` - Testing strategies

**Options:**
- `--project-path` - Target project directory

### `/project-start-enhanced` - Complete Workflow

Runs all four steps in sequence with intelligent defaults.

**Options:**
- Description argument - Project description

### `/configure-project-root` - Configuration

Configures Project-Start for nested projects or custom locations.

## 🔧 Global Options

- `--debug` - Enable verbose output
- `--existing-project` - Analyze existing project (Step 1 only)
- `--project-path PATH` - Specify target directory
- `--help` - Show help information

## 📁 Generated Structure

Commands create projects in the `specs/` directory:

```text
specs/001-your-project/
├── BACKLOG.md
├── IMPLEMENTATION_GUIDE.md
├── RISK_ASSESSMENT.md
├── FILE_OUTLINE.md
├── constitutional_validation.md
├── clarification_needed.md
├── sparc/
│   ├── SPARC_SPECIFICATION.md
│   ├── SPARC_PSEUDOCODE.md
│   ├── SPARC_ARCHITECTURE.md
│   ├── SPARC_REFINEMENT.md
│   └── SPARC_COMPLETION.md
├── expert_files/
│   ├── architecture_expert.md
│   ├── tech_stack_expert.md
│   └── methodology_expert.md
└── [PACT framework files]
```

## 🛠️ Prerequisites

- Python 3.6+
- Executable permissions for step scripts
- Optional: Gemini CLI for enhanced AI generation

## 🚨 Troubleshooting

### Permission Errors

```bash
chmod +x enhance-step-*
chmod +x project-start-enhanced
```

### Script Not Found

Ensure you're in the correct directory:

```bash
cd /path/to/project-start/cli
python3 project_start_cli.py
```

### AI Integration Issues

The CLI works with or without AI tools. When Gemini CLI is unavailable, it uses intelligent templates.

## 📖 Best Practices

1. **Start with Interactive Menu** - Provides guided experience
2. **Use Step 1 First** - Always begin with discovery
3. **Review Generated Files** - Validate outputs before proceeding
4. **Follow Constitutional Framework** - Maintain specification-driven approach

---

*All commands follow the Project-Start constitutional framework for specification-driven development.*
