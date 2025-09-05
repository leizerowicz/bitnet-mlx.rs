# Project-Start: Specification-Driven Development Framework

A constitutional framework for agentic software development that emphasizes specification-driven development, test-first methodologies, and AI-agent collaboration. Now with comprehensive AI integration supporting multiple assistants and environments.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **AI Assistant** (optional - smart fallbacks provided):
  - **Gemini CLI**: `npm install -g @google-ai/gemini-cli` (recommended)
  - **GitHub Copilot**: Available in VS Code
  - **Claude Code**: Available via Anthropic CLI

### Installation

```bash
# Clone the repository
git clone https://github.com/username/project-start.git
cd project-start

# Optional: Install Gemini CLI for enhanced AI generation
npm install -g @google-ai/gemini-cli
```

### Basic Usage

```bash
# Start a new project with AI-powered document generation
python3 cli/project_start_cli.py start "My AI-powered project"

# Interactive 4-step workflow (recommended)
./cli/enhance-step-1 "Your project description"

# Complete workflow with defaults
./cli/project-start-enhanced "Your project description"

# With debug output
python3 cli/project_start_cli.py start "My project" --debug
```

## ✨ Latest Features (September 2025)

### 🔧 CLI Consolidation & AI Integration

- **Multi-AI Support**: GitHub Copilot, Claude Code, Gemini CLI with intelligent selection
- **Production Ready**: All lint errors resolved, proper type hints, comprehensive documentation
- **Smart Fallbacks**: Works perfectly even without AI tools installed
- **Clean Architecture**: Consolidated multiple CLI files into single, maintainable implementation
- **Enhanced Error Handling**: Robust fallback mechanisms and graceful degradation

### 🎯 AI-Powered Features

- **Intelligent Document Generation**: 20+ documents per project with AI assistance
- **Constitutional Compliance**: AI-assisted validation against framework principles
- **Context-Aware Templates**: Smart fallbacks when AI unavailable
- **Multi-Environment Support**: VS Code, command line, and nested project scenarios

## 🎯 VS Code Integration

Project-Start includes full VS Code integration for an enhanced development experience:

- **🚀 Quick Start**: Open `project-start.code-workspace` for optimized workspace
- **⚡ Command Palette**: Access all CLI commands via `Ctrl+Shift+P` → "Tasks: Run Task"
- **📁 Auto-Detection**: Project root and paths automatically detected in VS Code
- **🛠️ Pre-configured Tasks**: All workflow steps available as VS Code tasks
- **🐛 Debug Support**: Built-in debugging configurations for CLI commands

**Get Started**: `code project-start.code-workspace` for full VS Code integration.

### Available VS Code Tasks

Access via Command Palette (`Ctrl+Shift+P`) → "Tasks: Run Task":

- **Project Start: Enhanced Workflow** - Complete workflow with auto-detection
- **Project Start: Step 1 - Discovery** - Interactive project discovery
- **Project Start: Step 1 - Existing Project** - Enhance existing projects
- **Project Start: Step 2 - SPARC Planning** - Constitutional methodology
- **Project Start: Step 3 - Context Systems** - Persistent agent context
- **Project Start: Step 4 - PACT Framework** - Multi-agent coordination

## 📂 Drag & Drop Support

Project-Start supports being dragged/copied into existing projects! When you drag the project-start folder into another project, it automatically detects and configures itself to work with the parent project.

### Quick Setup for Nested Projects

```bash
# After dragging project-start into your existing project
cd my-existing-project/project-start
python3 cli/project_start_cli.py /configure-project-root

# Follow the prompts - it will auto-detect the parent project
# Your project files will be created in my-existing-project/specs/
```

## � Interactive Workflow

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

The CLI includes powerful existing project support alongside its traditional new project creation:

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

```text
project-start/
├── README.md (this document)
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

## 🔧 AI Integration

### Multi-AI Assistant Support

```bash
# Use GitHub Copilot (default for VS Code)
python3 cli/project_start_cli.py start "My Project" --ai copilot

# Use Gemini CLI (recommended for command line)
python3 cli/project_start_cli.py start "My Project" --ai gemini

# Interactive selection
python3 cli/project_start_cli.py start "My Project"
```

### Smart Fallbacks

- Works perfectly even without AI tools installed
- Generates intelligent templates with context for manual AI enhancement
- Provides clear guidance for using templates with any AI assistant

## 🧪 Testing & Validation

The framework includes comprehensive testing capabilities:

- **Constitutional compliance validation** from project inception
- **Quality gates** preventing architectural drift
- **Automated testing frameworks** with PACT methodology
- **Multi-agent coordination testing** for complex workflows

---

This framework provides production-ready tools for scaling agentic development with constitutional governance and quality assurance.
