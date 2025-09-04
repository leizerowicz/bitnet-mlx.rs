#!/bin/bash
# Check if all project prerequisites are met before proceeding

set -euo pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Load project paths
eval "$(get_project_paths)"

echo "🔍 PROJECT PREREQUISITES CHECKER"
echo "================================="
echo "Project: $CURRENT_BRANCH"
echo "Location: $PROJECT_DIR"
echo

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ ERROR: Not in a git repository"
    echo "   Run 'git init' to initialize repository"
    exit 1
fi

# Check if we're already in a project directory
if [[ -f "BACKLOG.md" && -f "IMPLEMENTATION_GUIDE.md" ]]; then
    echo "📂 Working in current project directory"
    PROJECT_DIR="$(pwd)"
    CURRENT_BRANCH="$(basename "$PROJECT_DIR")"
    echo "Project: $CURRENT_BRANCH"
    echo "Location: $PROJECT_DIR"
    echo
    
    # Reset path variables for current directory
    BACKLOG="$PROJECT_DIR/BACKLOG.md"
    IMPL_GUIDE="$PROJECT_DIR/IMPLEMENTATION_GUIDE.md" 
    RISK_ASSESSMENT="$PROJECT_DIR/RISK_ASSESSMENT.md"
    CONSTITUTIONAL_VALIDATION="$PROJECT_DIR/constitutional_validation.md"
    SPARC_DIR="$PROJECT_DIR/sparc"
    SPARC_SPEC="$PROJECT_DIR/sparc/spec.md"
    COPILOT_INSTRUCTIONS="$PROJECT_DIR/copilot_instructions.md"
    AGENT_COORDINATION="$PROJECT_DIR/agent_coordination.md"
else
    echo "Project: $CURRENT_BRANCH"
    echo "Location: $PROJECT_DIR"
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        echo "❌ ERROR: Project directory does not exist: $PROJECT_DIR"
        echo "   Run './cli/enhance-step-1' to initialize project"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
fi

# Track overall status
ISSUES_FOUND=0

# Step 1 Prerequisites
echo
echo "📋 Step 1: Discovery & Planning Prerequisites"
if ! check_file "$BACKLOG" "BACKLOG.md"; then
    echo "   💡 Missing backlog - run './cli/enhance-step-1' to generate"
    ((ISSUES_FOUND++))
fi

if ! check_file "$IMPL_GUIDE" "IMPLEMENTATION_GUIDE.md"; then
    echo "   💡 Missing implementation guide - run './cli/enhance-step-1' to generate"
    ((ISSUES_FOUND++))
fi

if ! check_file "$CONSTITUTIONAL_VALIDATION" "Constitutional validation"; then
    echo "   💡 Missing constitutional validation - run './cli/enhance-step-1' to generate"
    ((ISSUES_FOUND++))
fi

# Step 2 Prerequisites (if needed)
if [[ -d "$SPARC_DIR" ]]; then
    echo
    echo "📋 Step 2: SPARC Methodology Prerequisites"
    if ! check_file "$SPARC_SPEC" "SPARC specification"; then
        echo "   💡 Incomplete SPARC - run './cli/enhance-step-2' to complete"
        ((ISSUES_FOUND++))
    fi
fi

# Step 3 Prerequisites (if needed)
if [[ -f "$COPILOT_INSTRUCTIONS" ]]; then
    echo
    echo "🧠 Step 3: Context Systems Prerequisites" 
    if ! check_file "$AGENT_COORDINATION" "Agent coordination"; then
        echo "   💡 Incomplete context systems - run './cli/enhance-step-3' to complete"
        ((ISSUES_FOUND++))
    fi
fi

# Check constitutional compliance
echo
echo "⚖️ Constitutional Compliance Check"
get_constitutional_status "$PROJECT_DIR"

# Check for Python/CLI requirements
echo
echo "🐍 Development Environment Check"
if command -v python3 > /dev/null; then
    echo "  ✅ Python 3 available"
else
    echo "  ❌ Python 3 not found"
    echo "     Install Python 3 for CLI functionality"
    ((ISSUES_FOUND++))
fi

# Check for required directories
echo
echo "📁 Directory Structure Check"
REQUIRED_DIRS=("cli" "templates" "memory")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        echo "  ✅ $dir/ directory"
    else
        echo "  ❌ $dir/ directory missing"
        ((ISSUES_FOUND++))
    fi
done

# Final summary
echo
echo "=" * 50
if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo "✅ ALL PREREQUISITES MET"
    echo "   Ready for development workflow"
    echo
    echo "🚀 Next Steps:"
    echo "   • Run './cli/project-start-enhanced' for full automation"
    echo "   • Or continue with individual step commands"
    exit 0
else
    echo "⚠️  ISSUES FOUND: $ISSUES_FOUND"
    echo "   Address the issues above before proceeding"
    echo
    echo "🔧 Quick Fix Commands:"
    echo "   • ./cli/enhance-step-1 'project description' # Initialize"
    echo "   • ./cli/enhance-step-2 --project-path $PROJECT_DIR # SPARC methodology"  
    echo "   • ./cli/enhance-step-3 --project-path $PROJECT_DIR # Context systems"
    echo "   • ./cli/enhance-step-4 --project-path $PROJECT_DIR # PACT framework"
    exit 1
fi