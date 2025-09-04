# Project-Start Utility Scripts

This directory contains helpful utility scripts for managing projects created with the Project-Start Enhanced CLI. These scripts are inspired by GitHub's spec-kit and provide automation and validation capabilities.

## Available Scripts

### 📋 `check-project-prerequisites.sh`
Validates that all required files and prerequisites are in place before proceeding with development.

```bash
# Check current project
./scripts/check-project-prerequisites.sh

# Check specific project
cd specs/001-my-project
../../scripts/check-project-prerequisites.sh
```

**What it checks:**
- Step 1: Discovery documents (BACKLOG.md, IMPLEMENTATION_GUIDE.md, etc.)
- Step 2: SPARC methodology completion
- Step 3: Context systems setup
- Step 4: PACT framework implementation
- Constitutional compliance status
- Development environment requirements

### 🚀 `create-new-project.sh`
Fully automated project creation that runs all 4 steps after initial questionnaire.

```bash
./scripts/create-new-project.sh "my awesome project"
```

**What it does:**
- Runs Step 1 (Discovery & Planning) interactively
- Automatically executes Steps 2-4 without further input
- Generates complete project specification
- Sets up AI agent coordination
- Provides ready-to-use project structure

### 🧠 `update-agent-context.sh`
Updates AI agent context and copilot instructions for better assistance.

```bash
# Update current project's agent context
./scripts/update-agent-context.sh

# Update specific project
./scripts/update-agent-context.sh specs/001-my-project
```

**What it does:**
- Analyzes current project state
- Generates enhanced copilot instructions
- Updates agent coordination protocols
- Enables persistent context across sessions
- Eliminates need for repetitive AI explanations

### 📂 `get-project-paths.sh`
Helper script that provides all important file paths for the current project.

```bash
# Human-readable output
./scripts/get-project-paths.sh

# JSON output for programmatic use
./scripts/get-project-paths.sh --json

# Use in other scripts
eval "$(./scripts/get-project-paths.sh)"
echo "Backlog is at: $BACKLOG"
```

### 🔧 `common.sh`
Shared functions and utilities used by other scripts. Not meant to be run directly.

**Provides:**
- Repository and project path utilities
- File and directory checking functions
- Project status analysis
- Constitutional compliance checking

## Usage Examples

### Quick Project Creation
```bash
# Create and fully configure a new project
./scripts/create-new-project.sh "real-time chat application"
```

### Project Health Check
```bash
# Move to your project and check status
cd specs/001-my-project
../../scripts/check-project-prerequisites.sh
```

### Agent Context Refresh
```bash
# Update AI context after making changes
./scripts/update-agent-context.sh specs/001-my-project
```

### Development Workflow Integration
```bash
# In your development scripts
eval "$(./scripts/get-project-paths.sh)"

# Now you can use variables like:
# $BACKLOG, $IMPL_GUIDE, $SPARC_SPEC, etc.
```

## Integration with CLI Commands

These scripts complement the main CLI commands:

| CLI Command | Related Script | Purpose |
|-------------|----------------|---------|
| `/project-start-enhanced` | `create-new-project.sh` | Full automation |
| `/enhance-step-1` | `check-project-prerequisites.sh` | Validation |
| All steps | `update-agent-context.sh` | Context management |
| Manual workflows | `get-project-paths.sh` | Path utilities |

## Constitutional Compliance

All scripts respect the Project-Start constitutional framework:

- **Article I**: Workflow-first development maintained
- **Article III**: Constitutional compliance validated
- **Article VIII**: Test-first development supported

## Requirements

- Bash 4.0+ (for script execution)
- Git (for repository operations)
- Python 3 (for CLI integration)

## Troubleshooting

### Script Permission Issues
```bash
chmod +x scripts/*.sh
```

### Path Resolution Problems
Scripts assume they're run from the repository root or project directory. If you get path errors:
```bash
cd /path/to/project-start
./scripts/script-name.sh
```

### Missing Dependencies
The `check-project-prerequisites.sh` script will identify missing requirements and suggest fixes.

---

*These scripts enhance the Project-Start Enhanced CLI with automation and validation capabilities inspired by GitHub's spec-kit framework.*