#!/bin/bash
# Configure project-start to work with a target project
# This script helps setup project-start when it's been dragged/copied into another project

set -euo pipefail

# Get the directory where this script is located (project-start directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_START_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_START_DIR/.project-start-config"

echo "🚀 PROJECT-START CONFIGURATION WIZARD"
echo "======================================"
echo
echo "This wizard will configure project-start to work with your target project."
echo "Use this when you've dragged/copied project-start into another project."
echo

# Auto-detect if we're in a nested scenario
CURRENT_DIR="$(pwd)"
PARENT_DIR="$(dirname "$PROJECT_START_DIR")"

echo "📍 Current Detection:"
echo "  Project-Start Directory: $PROJECT_START_DIR"
echo "  Parent Directory: $PARENT_DIR"
echo "  Working Directory: $CURRENT_DIR"
echo

# Check if we're likely in a nested scenario
NESTED_DETECTED=false
if [[ "$PROJECT_START_DIR" != "$PARENT_DIR" ]] && [[ -d "$PARENT_DIR/.git" || -f "$PARENT_DIR/package.json" || -f "$PARENT_DIR/requirements.txt" || -f "$PARENT_DIR/go.mod" || -f "$PARENT_DIR/Cargo.toml" ]]; then
    NESTED_DETECTED=true
    echo "🎯 NESTED SCENARIO DETECTED!"
    echo "  It looks like project-start is inside another project."
    echo "  Parent directory appears to be a valid project root."
    echo
fi

# Ask user for confirmation and target project root
if [[ "$NESTED_DETECTED" == "true" ]]; then
    echo "Would you like to configure project-start to work with the parent project?"
    echo "Target project root would be: $PARENT_DIR"
    echo
    read -p "Use parent directory as target project? (Y/n): " use_parent
    use_parent=${use_parent:-Y}
    
    if [[ "$use_parent" =~ ^[Yy] ]]; then
        TARGET_ROOT="$PARENT_DIR"
    else
        echo
        read -p "Please enter the target project root path: " TARGET_ROOT
    fi
else
    echo "No nested scenario detected. Please enter the target project root manually."
    read -p "Target project root path: " TARGET_ROOT
fi

# Validate the target root
if [[ ! -d "$TARGET_ROOT" ]]; then
    echo "❌ Error: Target directory does not exist: $TARGET_ROOT"
    exit 1
fi

# Make path absolute
TARGET_ROOT="$(cd "$TARGET_ROOT" && pwd)"

echo
echo "📋 Configuration Summary:"
echo "  Target Project Root: $TARGET_ROOT"
echo "  Project-Start Location: $PROJECT_START_DIR"
echo "  Config File: $CONFIG_FILE"
echo

# Optional project name
read -p "Enter a name for this project (optional): " PROJECT_NAME

# Confirm
echo
read -p "Save this configuration? (Y/n): " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy] ]]; then
    echo "Configuration cancelled."
    exit 0
fi

# Create configuration file
cat > "$CONFIG_FILE" << EOF
# Project-Start Configuration
# Generated on $(date)

# Target project root - where project-start should operate
TARGET_PROJECT_ROOT=$TARGET_ROOT

# Project name (optional)
PROJECT_NAME=${PROJECT_NAME:-}

# Nested mode - project-start is within another project
NESTED_MODE=true

# Project-start directory location
PROJECT_START_DIR=$PROJECT_START_DIR
EOF

echo
echo "✅ Configuration saved successfully!"
echo
echo "🎯 Next Steps:"
echo "  1. Your project-start commands will now operate on: $TARGET_ROOT"
echo "  2. Project files will be created in: $TARGET_ROOT/specs/"
echo "  3. Test the configuration by running:"
echo "     cd $PROJECT_START_DIR/scripts"
echo "     ./get-project-paths.sh"
echo
echo "💡 To reset configuration, delete: $CONFIG_FILE"
echo

# Test the configuration
echo "🧪 Testing configuration..."
cd "$PROJECT_START_DIR/scripts"
if ./get-project-paths.sh; then
    echo "✅ Configuration test successful!"
else
    echo "⚠️  Configuration test had issues - please check the setup"
fi