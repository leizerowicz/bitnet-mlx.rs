#!/bin/bash
# Common functions and variables for project-start scripts

# Get repository root
get_repo_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}

# Get current branch  
get_current_branch() {
    git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"
}

# Check if current branch is a project feature branch
# Returns 0 if valid, 1 if not
check_project_branch() {
    local branch="$1"
    if [[ ! "$branch" =~ ^[0-9]{3}- ]]; then
        echo "INFO: Not on a numbered project branch. Current branch: $branch"
        echo "Numbered branches follow pattern: 001-project-name"
        return 1
    fi
    return 0
}

# Get project directory path
get_project_dir() {
    local repo_root="$1"
    local branch="$2"
    echo "$repo_root/specs/$branch"
}

# Get all standard paths for a project
# Usage: eval $(get_project_paths)
# Sets: REPO_ROOT, CURRENT_BRANCH, PROJECT_DIR, BACKLOG, IMPL_GUIDE, etc.
get_project_paths() {
    local repo_root=$(get_repo_root)
    local current_branch=$(get_current_branch)
    local project_dir=$(get_project_dir "$repo_root" "$current_branch")
    
    echo "REPO_ROOT='$repo_root'"
    echo "CURRENT_BRANCH='$current_branch'"
    echo "PROJECT_DIR='$project_dir'"
    echo "BACKLOG='$project_dir/BACKLOG.md'"
    echo "IMPL_GUIDE='$project_dir/IMPLEMENTATION_GUIDE.md'"
    echo "RISK_ASSESSMENT='$project_dir/RISK_ASSESSMENT.md'"
    echo "FILE_OUTLINE='$project_dir/FILE_OUTLINE.md'"
    echo "CONSTITUTIONAL_VALIDATION='$project_dir/constitutional_validation.md'"
    echo "CLARIFICATION_NEEDED='$project_dir/clarification_needed.md'"
    echo "SPARC_DIR='$project_dir/sparc'"
    echo "SPARC_SPEC='$project_dir/sparc/spec.md'"
    echo "SPARC_PLAN='$project_dir/sparc/plan.md'"
    echo "SPARC_RESEARCH='$project_dir/sparc/research.md'"
    echo "SPARC_CONTEXT='$project_dir/sparc/context.md'"
    echo "COPILOT_INSTRUCTIONS='$project_dir/copilot_instructions.md'"
    echo "AGENT_COORDINATION='$project_dir/agent_coordination.md'"
    echo "PACT_TESTING='$project_dir/pact_testing.md'"
    echo "DEPLOYMENT_STRATEGY='$project_dir/deployment_strategy.md'"
}

# Check if a file exists and report
check_file() {
    local file="$1"
    local description="$2"
    if [[ -f "$file" ]]; then
        echo "  ✅ $description"
        return 0
    else
        echo "  ❌ $description"
        return 1
    fi
}

# Check if a directory exists and has files
check_dir() {
    local dir="$1"
    local description="$2"
    if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        echo "  ✅ $description"
        return 0
    else
        echo "  ❌ $description"
        return 1
    fi
}

# Show project status with constitutional compliance
show_project_status() {
    local project_dir="$1"
    
    echo "🔍 PROJECT STATUS ANALYSIS"
    echo "=========================="
    echo
    
    echo "📂 Step 1: Discovery & Planning"
    check_file "$project_dir/BACKLOG.md" "BACKLOG.md - User stories and features"
    check_file "$project_dir/IMPLEMENTATION_GUIDE.md" "IMPLEMENTATION_GUIDE.md - Technical approach"
    check_file "$project_dir/RISK_ASSESSMENT.md" "RISK_ASSESSMENT.md - Risk analysis"
    check_file "$project_dir/FILE_OUTLINE.md" "FILE_OUTLINE.md - Project structure"
    check_file "$project_dir/constitutional_validation.md" "Constitutional validation"
    check_file "$project_dir/clarification_needed.md" "Clarification requirements"
    echo
    
    echo "📋 Step 2: SPARC Methodology"
    check_dir "$project_dir/sparc" "SPARC directory structure"
    check_file "$project_dir/sparc/spec.md" "SPARC specification"
    check_file "$project_dir/sparc/plan.md" "SPARC implementation plan"
    check_file "$project_dir/sparc/research.md" "SPARC technology research"
    check_file "$project_dir/sparc/context.md" "SPARC project context"
    echo
    
    echo "🧠 Step 3: Persistent Context Systems"  
    check_file "$project_dir/copilot_instructions.md" "Copilot instructions"
    check_file "$project_dir/agent_coordination.md" "Agent coordination protocols"
    echo
    
    echo "🤖 Step 4: Constitutional PACT Framework"
    check_file "$project_dir/pact_testing.md" "PACT testing strategy"
    check_file "$project_dir/deployment_strategy.md" "Deployment strategy"
    echo
}

# Get constitutional compliance status
get_constitutional_status() {
    local project_dir="$1"
    local validation_file="$project_dir/constitutional_validation.md"
    
    if [[ -f "$validation_file" ]]; then
        echo "✅ Constitutional validation available"
        if grep -q "PASSED" "$validation_file"; then
            echo "🎯 Status: COMPLIANT"
        else
            echo "⚠️  Status: REQUIRES ATTENTION"
        fi
    else
        echo "❌ Constitutional validation missing"
        echo "🚨 Status: NON-COMPLIANT"
    fi
}