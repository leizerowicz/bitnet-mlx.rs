#!/bin/bash
# Get all important file paths for current project - helper for other scripts

set -euo pipefail

# Source common functions  
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Check if we want JSON output
JSON_OUTPUT=false
if [[ "${1:-}" == "--json" ]]; then
    JSON_OUTPUT=true
fi

# Get all project paths
eval "$(get_project_paths)"

if [[ "$JSON_OUTPUT" == "true" ]]; then
    # Output as JSON for programmatic use
    cat << EOF
{
  "repo_root": "$REPO_ROOT",
  "current_branch": "$CURRENT_BRANCH",
  "project_dir": "$PROJECT_DIR",
  "step1": {
    "backlog": "$BACKLOG",
    "implementation_guide": "$IMPL_GUIDE",
    "risk_assessment": "$RISK_ASSESSMENT", 
    "file_outline": "$FILE_OUTLINE",
    "constitutional_validation": "$CONSTITUTIONAL_VALIDATION",
    "clarification_needed": "$CLARIFICATION_NEEDED"
  },
  "step2": {
    "sparc_dir": "$SPARC_DIR",
    "spec": "$SPARC_SPEC",
    "plan": "$SPARC_PLAN", 
    "research": "$SPARC_RESEARCH",
    "context": "$SPARC_CONTEXT"
  },
  "step3": {
    "copilot_instructions": "$COPILOT_INSTRUCTIONS",
    "agent_coordination": "$AGENT_COORDINATION"
  },
  "step4": {
    "pact_testing": "$PACT_TESTING",
    "deployment_strategy": "$DEPLOYMENT_STRATEGY"
  }
}
EOF
else
    # Human-readable output
    echo "📂 PROJECT FILE PATHS"
    echo "===================="
    echo "Repository Root: $REPO_ROOT"  
    echo "Current Branch: $CURRENT_BRANCH"
    echo "Project Directory: $PROJECT_DIR"
    echo
    echo "📋 Step 1: Discovery & Planning"
    echo "  BACKLOG: $BACKLOG"
    echo "  IMPLEMENTATION_GUIDE: $IMPL_GUIDE"
    echo "  RISK_ASSESSMENT: $RISK_ASSESSMENT"
    echo "  FILE_OUTLINE: $FILE_OUTLINE" 
    echo "  CONSTITUTIONAL_VALIDATION: $CONSTITUTIONAL_VALIDATION"
    echo "  CLARIFICATION_NEEDED: $CLARIFICATION_NEEDED"
    echo
    echo "📋 Step 2: SPARC Methodology"
    echo "  SPARC_DIR: $SPARC_DIR"
    echo "  SPEC: $SPARC_SPEC"
    echo "  PLAN: $SPARC_PLAN"
    echo "  RESEARCH: $SPARC_RESEARCH"
    echo "  CONTEXT: $SPARC_CONTEXT"
    echo
    echo "🧠 Step 3: Context Systems"
    echo "  COPILOT_INSTRUCTIONS: $COPILOT_INSTRUCTIONS"
    echo "  AGENT_COORDINATION: $AGENT_COORDINATION"
    echo
    echo "🤖 Step 4: PACT Framework"
    echo "  PACT_TESTING: $PACT_TESTING"
    echo "  DEPLOYMENT_STRATEGY: $DEPLOYMENT_STRATEGY"
    echo
    echo "💡 Usage in other scripts:"
    echo "  eval \"\$(./scripts/get-project-paths.sh)\""
    echo "  echo \"Backlog location: \$BACKLOG\""
fi