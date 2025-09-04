# Drag and Drop Setup Guide

## Overview

This guide explains how to use project-start when it has been dragged/copied into another existing project. By default, project-start assumes it's the main project, but when nested within another project, you need to configure it to work with the parent project.

## Problem

When you drag and drop the project-start folder into another project:

```
my-awesome-project/
├── src/
├── package.json
├── README.md
└── project-start/          # <-- Dragged here
    ├── cli/
    ├── scripts/
    └── ...
```

Without configuration, project-start commands would treat `project-start/` as the main project instead of `my-awesome-project/`.

## Solution

Project-start now includes automatic detection and configuration for nested scenarios.

### Method 1: Automatic Configuration (Recommended)

1. **Navigate to project-start directory:**
   ```bash
   cd my-awesome-project/project-start
   ```

2. **Run the configuration wizard:**
   ```bash
   python3 cli/project_start_cli.py /configure-project-root
   ```
   
   OR
   
   ```bash
   ./scripts/configure-project-root.sh
   ```

3. **Follow the prompts:**
   - The wizard will automatically detect you're in a nested scenario
   - It will suggest using the parent directory as the target project
   - Confirm the configuration and optionally name your project

4. **Test the configuration:**
   ```bash
   cd scripts
   ./get-project-paths.sh
   ```

### Method 2: Manual Configuration

1. **Create configuration file:**
   ```bash
   cd my-awesome-project/project-start
   cp .project-start-config.example .project-start-config
   ```

2. **Edit the configuration:**
   ```bash
   # Edit .project-start-config
   TARGET_PROJECT_ROOT=/path/to/my-awesome-project
   PROJECT_NAME=my-awesome-project
   NESTED_MODE=true
   ```

3. **Test the configuration:**
   ```bash
   ./scripts/get-project-paths.sh
   ```

## How It Works

### Root Detection Priority

Project-start uses the following priority order to determine the project root:

1. **Configuration File**: If `.project-start-config` exists and specifies `TARGET_PROJECT_ROOT`
2. **Auto-Detection**: If project-start is nested and parent has project indicators (git, package.json, etc.)
3. **Git Root**: Falls back to `git rev-parse --show-toplevel`
4. **Current Directory**: Final fallback

### Project Indicators

Project-start recognizes these files/directories as project indicators:
- `.git/`
- `package.json` (Node.js)
- `requirements.txt` (Python)
- `go.mod` (Go)
- `Cargo.toml` (Rust)
- `pom.xml` (Java/Maven)
- `build.gradle` (Java/Gradle)
- `Makefile` (C/C++/Make)

## Verification

After configuration, verify that project-start is working correctly:

### Check Project Root Detection
```bash
cd my-awesome-project/project-start/scripts
./get-project-paths.sh
```

Should show:
```
Repository Root: /path/to/my-awesome-project  # <-- Parent project
Project Directory: /path/to/my-awesome-project/specs/...
```

### Test CLI Commands
```bash
cd my-awesome-project/project-start
python3 cli/project_start_cli.py /enhance-step-1 "test project"
```

Files should be created in `/path/to/my-awesome-project/specs/` (not in project-start/).

## Project Structure After Configuration

```
my-awesome-project/
├── src/                    # Your existing project files
├── package.json
├── README.md
├── specs/                  # <-- Project-start creates files here
│   └── 001-new-feature/
│       ├── BACKLOG.md
│       ├── IMPLEMENTATION_GUIDE.md
│       └── ...
└── project-start/          # <-- Project-start tools
    ├── .project-start-config  # <-- Configuration file
    ├── cli/
    ├── scripts/
    └── ...
```

## Troubleshooting

### Issue: Commands still operate on project-start directory

**Solution**: Check configuration file:
```bash
cat project-start/.project-start-config
```

Ensure `TARGET_PROJECT_ROOT` points to the correct directory.

### Issue: Auto-detection not working

**Causes**:
- Parent directory doesn't have recognizable project indicators
- Multiple nested git repositories confusing detection

**Solution**: Use manual configuration or add a project indicator file to parent.

### Issue: Configuration test fails

**Check**:
1. Target directory exists and is accessible
2. Configuration file syntax is correct
3. No trailing spaces in configuration values

## Resetting Configuration

To reset project-start to work as a standalone project:

```bash
rm project-start/.project-start-config
```

## Integration with Existing Workflows

### VS Code
Project-start will still work in VS Code. The workspace settings in `project-start.code-workspace` will continue to function, but project files will be created in the configured target project.

### Git
Project-start respects the git repository of the target project. All generated files will be part of your main project's git history.

### Scripts
All project-start scripts automatically use the configured project root. No changes to your workflow needed.

## Example Scenarios

### Scenario 1: Node.js Project
```bash
# You have an existing Node.js project
my-node-app/
├── package.json
├── src/
└── ...

# Drag project-start into it
my-node-app/
├── package.json
├── src/
└── project-start/

# Configure it
cd my-node-app/project-start
python3 cli/project_start_cli.py /configure-project-root
# Choose: Y (use parent directory)

# Now project-start works on my-node-app
```

### Scenario 2: Python Project
```bash
# Existing Python project
my-python-app/
├── requirements.txt
├── src/
└── ...

# After dragging project-start and configuring
my-python-app/
├── requirements.txt
├── src/
├── specs/              # <-- Project-start files go here
│   └── 001-feature/
└── project-start/      # <-- Tools location
```

## Best Practices

1. **Keep project-start updated**: Regularly pull updates to project-start tooling
2. **Document configuration**: Add project-start setup to your project's README
3. **Team coordination**: Ensure all team members configure project-start consistently
4. **Backup configuration**: Include `.project-start-config` in version control

## Configuration File Reference

```bash
# .project-start-config

# Target project root - where project-start should operate
TARGET_PROJECT_ROOT=/absolute/path/to/your/project

# Project name (optional)
PROJECT_NAME=my-awesome-project

# Nested mode - project-start is within another project
NESTED_MODE=true

# Project-start directory location (auto-set)
PROJECT_START_DIR=/absolute/path/to/your/project/project-start
```

**Note**: Use absolute paths in the configuration file for reliability.