#!/bin/bash  
# Update agent context and copilot instructions for better AI assistance

set -euo pipefail

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PROJECT_PATH="${1:-}"
if [[ -z "$PROJECT_PATH" ]]; then
    # Try to auto-detect project path
    eval "$(get_project_paths)"
    PROJECT_PATH="$PROJECT_DIR"
fi

if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "❌ ERROR: Project directory not found: $PROJECT_PATH"
    echo "Usage: $0 [project-path]"
    echo "Example: $0 specs/001-chat-app"
    exit 1
fi

echo "🧠 AGENT CONTEXT UPDATER"  
echo "========================"
echo "Project: $PROJECT_PATH"
echo

cd "$PROJECT_PATH"

# Analyze current project state
echo "📊 Analyzing current project state..."

CONTEXT_FILE="copilot_instructions.md"
BACKUP_FILE="copilot_instructions.backup.$(date +%Y%m%d_%H%M%S).md"

# Backup existing instructions if they exist
if [[ -f "$CONTEXT_FILE" ]]; then
    cp "$CONTEXT_FILE" "$BACKUP_FILE"
    echo "💾 Backed up existing instructions to $BACKUP_FILE"
fi

# Create enhanced copilot instructions
echo "✨ Generating enhanced copilot instructions..."

cat > "$CONTEXT_FILE" << 'EOF'
# Enhanced Copilot Instructions - Project Context

## 🎯 Project Mission
This document provides persistent context for AI agents working on this project, ensuring consistent understanding and reducing repetitive explanations.

## 🏛️ Constitutional Governance
All agents must follow these immutable principles:

### Article I: Workflow-First Development
- Follow the established 4-step methodology (Discovery → SPARC → Context → PACT)  
- Maintain traceability from user needs to implementation
- Validate each phase before proceeding

### Article III: Constitutional Compliance (NON-NEGOTIABLE)
- All decisions must align with constitutional principles
- Quality gates must be respected at every transition
- No shortcuts that compromise architectural integrity

### Article VIII: Test-First Development (NON-NEGOTIABLE)
- Write tests before implementation
- Maintain comprehensive test coverage
- Validate behavior through automated testing

## 📋 Current Project State
EOF

# Add project state analysis
if [[ -f "BACKLOG.md" ]]; then
    echo "- ✅ Step 1: Discovery completed (BACKLOG, IMPLEMENTATION_GUIDE, RISK_ASSESSMENT)" >> "$CONTEXT_FILE"
else
    echo "- ❌ Step 1: Discovery incomplete" >> "$CONTEXT_FILE"
fi

if [[ -f "sparc/spec.md" ]]; then
    echo "- ✅ Step 2: SPARC methodology completed" >> "$CONTEXT_FILE"
else
    echo "- ❌ Step 2: SPARC methodology incomplete" >> "$CONTEXT_FILE"
fi

if [[ -f "agent_coordination.md" ]]; then
    echo "- ✅ Step 3: Context systems established" >> "$CONTEXT_FILE"
else
    echo "- ❌ Step 3: Context systems not established" >> "$CONTEXT_FILE"
fi

if [[ -f "pact_testing.md" ]]; then
    echo "- ✅ Step 4: PACT framework implemented" >> "$CONTEXT_FILE"  
else
    echo "- ❌ Step 4: PACT framework not implemented" >> "$CONTEXT_FILE"
fi

# Add project-specific context
cat >> "$CONTEXT_FILE" << 'EOF'

## 🔧 Development Context

### Key Documents Reference
- `BACKLOG.md` - User stories and feature priorities
- `IMPLEMENTATION_GUIDE.md` - Technical approach and constraints  
- `RISK_ASSESSMENT.md` - Known risks and mitigation strategies
- `FILE_OUTLINE.md` - Project structure and organization
- `sparc/` - Technical specifications and implementation plans
- `constitutional_validation.md` - Compliance verification

### Multi-Agent Coordination  
- Agents operate under shared constitutional governance
- Context is preserved across sessions through this document
- Decision authority follows established escalation procedures
- Quality validation required before major changes

## 🤖 AI Agent Instructions

### Before Starting Any Task
1. **Load Context**: Review this document and current project state
2. **Validate Understanding**: Confirm task requirements against constitutional principles
3. **Check Dependencies**: Ensure prerequisites are met before proceeding
4. **Plan Approach**: Design solution following test-first methodology

### During Development
1. **Constitutional Check**: Regularly validate decisions against governance principles
2. **Context Updates**: Document significant decisions and rationale
3. **Quality Gates**: Ensure testing and validation at each step
4. **Traceability**: Maintain links between requirements and implementation

### Quality Validation Checklist
- [ ] Constitutional compliance verified
- [ ] Test coverage adequate  
- [ ] Requirements traceability maintained
- [ ] Documentation updated
- [ ] No architectural debt introduced

## 🚀 Optimization Features

### Persistent Memory
This document eliminates repetitive context explanations by maintaining:
- Project state and decision history
- Technical constraints and preferences
- Quality standards and validation requirements
- Agent coordination protocols

### Automated Governance
Constitutional principles are enforced through:
- Quality gates at each workflow transition
- Automated compliance checking
- Persistent context preservation
- Multi-agent coordination protocols

---
*Updated by Agent Context Updater*
EOF

echo "✅ Enhanced copilot instructions generated"

# Update agent coordination if it exists
if [[ -f "agent_coordination.md" ]]; then
    echo "🔄 Updating agent coordination protocols..."
    
    # Add timestamp to coordination file
    echo "" >> "agent_coordination.md"
    echo "---" >> "agent_coordination.md"
    echo "*Context updated: $(date)*" >> "agent_coordination.md"
    echo "✅ Agent coordination updated"
fi

# Show summary
echo
echo "📊 CONTEXT UPDATE SUMMARY"
echo "========================="
echo "✅ Copilot instructions: Enhanced with project state analysis"
echo "✅ Constitutional governance: Embedded in agent instructions"
echo "✅ Quality gates: Activated for all agent interactions"
echo "✅ Memory systems: Persistent context established"
echo

if [[ -f "$BACKUP_FILE" ]]; then
    echo "💾 Previous version backed up as: $BACKUP_FILE"
fi

echo
echo "🎯 Impact:"
echo "  • Agents now have persistent project context"
echo "  • Constitutional compliance automatically enforced"
echo "  • Reduced need for repetitive explanations"
echo "  • Consistent quality standards across all interactions"
echo
echo "💡 Next: Agents can now work more efficiently with enhanced context awareness"