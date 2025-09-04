#!/bin/bash
# Create a new project with automated workflow after questions are answered

set -euo pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_NAME="$1"
if [[ -z "$PROJECT_NAME" ]]; then
    echo "❌ ERROR: Project name required"
    echo "Usage: $0 <project-name>"
    echo "Example: $0 \"real-time chat application\""
    exit 1
fi

echo "🚀 AUTOMATED PROJECT CREATION"
echo "============================="
echo "Project: $PROJECT_NAME"
echo

# Get repository root
REPO_ROOT=$(get_repo_root)
cd "$REPO_ROOT"

echo "📋 Phase 1: Project Discovery & Planning..."
echo "Running enhanced Step 1 with automation..."

# Run Step 1 with the project name
if ./cli/enhance-step-1 "$PROJECT_NAME"; then
    echo "✅ Step 1 completed successfully"
else
    echo "❌ Step 1 failed"
    exit 1
fi

# Find the created project directory
PROJECT_DIR=$(find specs -maxdepth 1 -type d -name "*${PROJECT_NAME// /-}*" | head -1)
if [[ -z "$PROJECT_DIR" ]]; then
    # Try to find any recently created project
    PROJECT_DIR=$(find specs -maxdepth 1 -type d -newer specs 2>/dev/null | head -1)
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "❌ Could not find created project directory"
    exit 1
fi

echo "📂 Found project directory: $PROJECT_DIR"
echo

echo "📋 Phase 2: Constitutional SPARC Methodology..."
echo "Generating technical specifications..."

if ./cli/enhance-step-2 --project-path "$PROJECT_DIR"; then
    echo "✅ Step 2 completed successfully"
else
    echo "❌ Step 2 failed"
    exit 1
fi

echo
echo "🧠 Phase 3: Persistent Context Systems..."
echo "Setting up AI agent coordination..."

if ./cli/enhance-step-3 --project-path "$PROJECT_DIR"; then
    echo "✅ Step 3 completed successfully"  
else
    echo "❌ Step 3 failed"
    exit 1
fi

echo
echo "🤖 Phase 4: Constitutional PACT Framework..."
echo "Implementing testing and deployment strategy..."

if ./cli/enhance-step-4 --project-path "$PROJECT_DIR"; then
    echo "✅ Step 4 completed successfully"
else
    echo "❌ Step 4 failed"
    exit 1
fi

echo
echo "🎉 AUTOMATED PROJECT CREATION COMPLETED!"
echo "========================================"
echo
echo "📂 Project Location: $PROJECT_DIR"
echo "🎯 All 4 workflow steps completed successfully"
echo

# Show project status
eval "$(get_project_paths)"
show_project_status "$PROJECT_DIR"

echo
echo "🚀 Ready for Development!"
echo "========================"
echo "Your project is fully configured with:"
echo "  • Complete specifications and planning"
echo "  • Constitutional governance framework"  
echo "  • AI agent coordination protocols"
echo "  • Testing and deployment strategies"
echo
echo "💡 Next Steps:"
echo "  1. Review the generated documents in $PROJECT_DIR"
echo "  2. Address any items in clarification_needed.md"
echo "  3. Begin implementation following the constitutional framework"
echo "  4. Use the copilot instructions for AI-assisted development"