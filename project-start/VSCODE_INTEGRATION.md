# VS Code Integration Guide

## Overview

The Project-Start CLI is now fully integrated with Visual Studio Code to provide a seamless development experience. This integration eliminates the need to manually specify project directories and makes all CLI functions easily accessible through VS Code's interface.

## Features

### 🎯 Automatic Project Detection
- **Smart Root Detection**: Automatically detects project root when running from VS Code
- **Workspace Support**: Works with both single folders and multi-folder workspaces
- **Path Auto-Configuration**: No need to specify `--project-path` when using VS Code tasks

### ⚡ Quick Access via Command Palette
Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and type:
- `Tasks: Run Task` → Choose any Project-Start task
- `Project Start` → See all available Project-Start commands

### 🛠️ Available Tasks

All tasks are pre-configured and ready to use:

#### Core Workflow Tasks
- **Project Start: Enhanced Workflow** - Complete workflow with auto-detection
- **Project Start: Step 1 - Discovery** - Interactive project discovery
- **Project Start: Step 1 - Existing Project** - Enhance existing projects
- **Project Start: Step 2 - SPARC Planning** - Constitutional methodology
- **Project Start: Step 3 - Context Systems** - Persistent agent context
- **Project Start: Step 4 - PACT Framework** - Multi-agent coordination

#### Utility Tasks
- **Project Start: Check Prerequisites** - Validate project structure
- **Project Start: Update Agent Context** - Refresh copilot instructions
- **Project Start: Get Project Paths** - Display current project structure

### 🐛 Debugging Support

Debug configurations are available in the debugger panel:
- **Project Start: Debug Enhanced Workflow**
- **Project Start: Debug Step 1**
- **Project Start: Debug Existing Project**

## Quick Start

1. **Open Project in VS Code**:
   ```bash
   code project-start.code-workspace
   ```
   Or simply open the project folder in VS Code.

2. **Run Your First Task**:
   - Press `Ctrl+Shift+P`
   - Type "Tasks: Run Task"
   - Select "Project Start: Enhanced Workflow"

3. **For Existing Projects**:
   - Press `Ctrl+Shift+P`
   - Type "Tasks: Run Task" 
   - Select "Project Start: Step 1 - Existing Project"

## Integration Benefits

### 🔄 Seamless Workflow
- No manual path specification required
- Automatic workspace root detection
- Integrated terminal output
- Task reusability across sessions

### 📁 File Access Integration
- Project files automatically accessible in sidebar
- Intelligent file associations for .md files
- Search and navigation optimized for project structure
- Git integration with project context

### 🤖 Enhanced AI Collaboration
- VS Code environment automatically detected by CLI
- Copilot integration status displayed
- Context preservation across VS Code sessions
- Optimized workspace settings for AI development

## Configuration Files

The integration includes several VS Code configuration files:

### `.vscode/settings.json`
- Python development optimization
- Markdown preview settings
- File associations and exclusions
- Project-Start specific settings

### `.vscode/tasks.json`
- All CLI commands as VS Code tasks
- Proper terminal integration
- Project root auto-detection
- Build group organization

### `.vscode/extensions.json`
- Recommended extensions for optimal experience
- GitHub Copilot integration
- Python development tools
- Markdown and documentation tools

### `.vscode/launch.json`
- Debug configurations for CLI commands
- Environment variable setup
- VS Code environment detection

### `project-start.code-workspace`
- Multi-folder workspace configuration
- Organized folder structure
- Workspace-specific settings
- Quick-start task definitions

## Best Practices

### 🎯 Getting Started
1. Open the workspace file: `project-start.code-workspace`
2. Install recommended extensions when prompted
3. Use Command Palette for task access
4. Let VS Code auto-detect project paths

### 🔧 Daily Usage
- Use `Ctrl+Shift+P` → "Tasks: Run Task" for CLI access
- File navigation through VS Code sidebar
- Terminal integration for interactive CLI operations
- Debug mode for troubleshooting

### 🚀 Advanced Features
- Multi-folder workspace for complex projects
- Task dependencies and automation
- Integrated debugging with breakpoints
- Extension ecosystem for enhanced development

## Troubleshooting

### Environment Detection Issues
If VS Code integration is not detected:
1. Ensure you're running tasks from VS Code (not external terminal)
2. Check that `.vscode/` configuration files exist
3. Verify workspace file is properly configured
4. Set `PROJECT_START_VSCODE=true` environment variable if needed

### Task Execution Issues
If tasks fail to run:
1. Verify Python 3 is available in PATH
2. Check that you're in the correct workspace root
3. Ensure CLI file permissions are correct
4. Run `python3 cli/project_start_cli.py --help` to verify CLI accessibility

### Path Detection Issues
If project paths are incorrect:
1. Verify workspace is opened at project root
2. Check for `.vscode/settings.json` in project root
3. Ensure workspace file is in correct location
4. Use absolute paths in VS Code terminal if needed

## CLI Commands from VS Code

All original CLI functionality remains available. When running from VS Code:

```bash
# These commands now auto-detect VS Code environment
python3 cli/project_start_cli.py /project-start-enhanced
python3 cli/project_start_cli.py /enhance-step-1
python3 cli/project_start_cli.py /enhance-step-1 --existing-project

# Project paths are automatically detected - no need for --project-path
python3 cli/project_start_cli.py /enhance-step-2
python3 cli/project_start_cli.py /enhance-step-3
python3 cli/project_start_cli.py /enhance-step-4
```

## What's Preserved

✅ **All existing functionality maintained**
✅ **Backward compatibility with command-line usage**  
✅ **No changes to core CLI behavior**
✅ **Constitutional compliance and validation preserved**
✅ **Memory systems and agent coordination unchanged**
✅ **All Step 1-4 workflows fully functional**

The VS Code integration is purely additive - it enhances the experience without modifying any existing capabilities.