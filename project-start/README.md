# Project Start: Agentic Engineering Framework

A proven 4-step framework for agentic engineering projects with constitutional governance principles and automated specification generation.

## 🎯 VS Code Integration

Project-Start now includes full VS Code integration for an enhanced development experience:

- **🚀 Quick Start**: Open `project-start.code-workspace` for optimized workspace
- **⚡ Command Palette**: Access all CLI commands via `Ctrl+Shift+P` → "Tasks: Run Task"  
- **📁 Auto-Detection**: Project root and paths automatically detected in VS Code
- **🛠️ Pre-configured Tasks**: All workflow steps available as VS Code tasks
- **🐛 Debug Support**: Built-in debugging configurations for CLI commands

**Get Started**: `code project-start.code-workspace` or see [VSCODE_INTEGRATION.md](VSCODE_INTEGRATION.md) for full details.

## 📂 Drag & Drop Support

**NEW**: Project-Start now supports being dragged/copied into existing projects! When you drag the project-start folder into another project, it can automatically detect and configure itself to work with the parent project.

### Quick Setup for Nested Projects
```bash
# After dragging project-start into your existing project
cd my-existing-project/project-start
python3 cli/project_start_cli.py /configure-project-root

# Follow the prompts - it will auto-detect the parent project
# Your project files will be created in my-existing-project/specs/
```

📖 **Full Guide**: See [DRAG_DROP_SETUP.md](DRAG_DROP_SETUP.md) for complete instructions and examples.

## 🚀 Quick Start

```bash
# Interactive project discovery (recommended)
cd project-start
./cli/enhance-step-1 "Your project description"

# Complete 4-step workflow
./cli/enhance-step-2 --project-path specs/001-your-project
./cli/enhance-step-3 --project-path specs/001-your-project
./cli/enhance-step-4 --project-path specs/001-your-project

# Or use master command with defaults
./cli/project-start-enhanced "Your project description"
```

## ✅ Production Ready Features

- **Interactive CLI** with comprehensive questionnaire system
- **Automated document generation** (20+ documents per project)
- **Constitutional compliance** validation from inception
- **4-step development workflow** fully implemented
- **Persistent memory systems** for multi-agent coordination
- **Quality gates** preventing architectural drift
- **Enhanced README reading** - CLI reads from Step 1-4 READMEs for context-aware generation
- **🆕 Existing Project Support** - Works with established architecture and codebases

## 🔧 Enhanced CLI Capabilities

The CLI now includes powerful existing project support alongside its traditional new project creation:

### New Project Creation
- **Step 1**: Reads `step_1/README.md` for project discovery context and brutally honest market validation
- **Step 2**: Reads `step_2/sparc_methodology_guide.md` for SPARC methodology principles and implementation guidance  
- **Step 3**: Reads `step_3/README.md` for agentic system integration and expert context
- **Step 4**: Reads `step_4/` templates for PACT framework implementation details

### 🆕 Existing Project Support
- **Project Detection**: Automatically scans for existing project structure and type
- **Comprehensive File Analysis**: Analyzes MD files, code files, and configuration files
- **Smart Categorization**: Identifies key files and suggests focus areas for analysis
- **Multiple Analysis Modes**: 
  - Automatic analysis based on file discovery
  - User-guided file selection for focused analysis
  - Hybrid approach with supplementary context gathering
- **Preservation of Structure**: Maintains original project organization while adding Project-Start documentation
- **Constitutional Integration**: Applies Project-Start principles to existing codebases

This ensures all generated documents are informed by the complete framework methodology rather than using generic templates.

## 📚 Complete Documentation

👉 **[See SPEC_KIT_INTEGRATION_GUIDE.md](SPEC_KIT_INTEGRATION_GUIDE.md)** for comprehensive implementation details and usage instructions.

## 🏗️ Framework Overview

**4-Step Agentic Engineering Workflow:**
- **Step 1**: Discovery & Planning (interactive questionnaire + 6 documents)
- **Step 2**: Constitutional SPARC Methodology (specification-driven development)
- **Step 3**: Persistent Context Systems (multi-agent coordination)
- **Step 4**: Constitutional PACT Framework (testing & deployment)

## 📋 Constitutional Governance

Governed by 9 immutable principles in `PROJECT_START_CONSTITUTION.md`:
- Workflow-First Development & Constitutional Compliance
- Specification-Driven Implementation & Agent Coordination  
- Memory-Driven Context & Simplicity Principle
- Test-First Development & Continuous Validation

## 📁 File Structure

```
project-start/
├── README.md (this document)
├── SPEC_KIT_INTEGRATION_GUIDE.md (comprehensive guide)
├── PROJECT_START_CONSTITUTION.md (governance principles)
├── cli/ (interactive CLI tools)
├── memory/ (persistent context system)
├── templates/ (document generation)
├── specs/ (generated specifications)
└── step_*/ (original documentation)
```

## 🎯 Generated Output

Each project creates:
- 6 comprehensive specification documents
- Constitutional compliance validation
- Persistent memory systems
- Multi-agent coordination protocols
- Complete testing & deployment strategies

---

This framework provides production-ready tools for scaling agentic development with constitutional governance and quality assurance.
