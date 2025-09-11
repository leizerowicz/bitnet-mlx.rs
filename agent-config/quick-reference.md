# BitNet-Rust Agent System Quick Reference

> **Last Updated**: September 11, 2025
> **Purpose**: Quick reference for using the orchestrator-driven agent configuration system

## 🎯 START HERE: Always Use Orchestrator First

**MANDATORY WORKFLOW**: For ANY development work, ALWAYS start with:

```
1. Read agent-config/orchestrator.md FIRST
2. Get task routing and agent selection
3. Follow orchestrator coordination workflow  
4. Consult routed specialist agents
5. Report back through orchestrator
```

## Quick Agent Lookup

### 🎯 **When to Use Which Agent**

| Need | Primary Agent | Common Partners |
|------|---------------|-----------------|
| **Fix bugs/debug** | `debug.md` | `code.md`, `error_handling_specialist.md` |
| **Write/change code** | `code.md` | `rust_best_practices_specialist.md`, `test_utilities_specialist.md` |
| **System design** | `architect.md` | `security_reviewer.md`, `performance_engineering_specialist.md` |
| **Performance optimize** | `performance_engineering_specialist.md` | `inference_engine_specialist.md`, `rust_best_practices_specialist.md` |
| **ML/inference features** | `inference_engine_specialist.md` | `api_development_specialist.md`, `performance_engineering_specialist.md` |
| **Testing/validation** | `test_utilities_specialist.md` | `error_handling_specialist.md`, `truth_validator.md` |
| **Documentation** | `documentation_writer.md` | `ask.md`, domain specialists |
| **Security review** | `security_reviewer.md` | `rust_best_practices_specialist.md`, `architect.md` |
| **User questions** | `ask.md` | `documentation_writer.md`, `customer_success_specialist.md` |

### 🔄 **Agent Intersection Patterns**

**Every agent knows how they work with others:**

- **Code Development** ↔ **Rust Best Practices** (quality)
- **Debug** ↔ **Code Development** (fixes)
- **Inference Engine** ↔ **Performance Engineering** (optimization)
- **Architect** ↔ **Security Reviewer** (secure design)
- **Test Utilities** ↔ **Truth Validator** (validation)

## Current Project Status (September 11, 2025)

**Phase**: Inference Ready Phase  
**Status**: 99.8% test success (530/531 tests passing)  
**Priority**: Task 1.0.1 - Fix memory tracking test → Epic 2 (Inference implementation)

## Task Routing Examples

### Simple Bug Fix
```
User: "Test is failing" 
→ orchestrator.md → debug.md + code.md + test_utilities_specialist.md
```

### New Feature Development  
```
User: "Add new inference feature"
→ orchestrator.md → inference_engine_specialist.md + code.md + performance_engineering_specialist.md
```

### Architecture Decision
```
User: "How should we structure this system?"
→ orchestrator.md → architect.md + security_reviewer.md + performance_engineering_specialist.md
```

### Performance Issue
```
User: "System is slow"
→ orchestrator.md → performance_engineering_specialist.md + debug.md + test_utilities_specialist.md
```

## Quality Gates

**All work must pass through appropriate quality gates:**

- **Code Quality**: `rust_best_practices_specialist.md` review
- **Testing**: `test_utilities_specialist.md` validation
- **Security**: `security_reviewer.md` for sensitive components
- **Performance**: `performance_engineering_specialist.md` for performance-critical features
- **Final Validation**: `truth_validator.md` for critical changes

## Quick Workflow Checklist

- [ ] Started with `orchestrator.md`?
- [ ] Got proper agent routing?
- [ ] Consulted appropriate specialists?
- [ ] Followed collaboration patterns?
- [ ] Applied quality gates?
- [ ] Coordinated through orchestrator?

## Common Anti-Patterns (Don't Do This)

❌ **Skip orchestrator** - Going directly to specialist agents  
❌ **Work in isolation** - Not following collaboration patterns  
❌ **Bypass quality gates** - Skipping required reviews  
❌ **Ignore intersections** - Not involving appropriate collaborating agents  
❌ **Direct agent communication** - Agents coordinating without orchestrator oversight

## Agent Hooks Integration

**Every agent workflow is enhanced with automated hooks:**
- Pre-task setup and validation
- Collaboration coordination
- Quality gate enforcement  
- Post-task cleanup and handoffs
- Progress monitoring and escalation

See `agent-config/agent-hooks.md` for details.

## Emergency Contacts

**Critical Issues**: Start with `orchestrator.md` for immediate routing  
**Security Issues**: `security_reviewer.md` (with orchestrator coordination)  
**Performance Issues**: `performance_engineering_specialist.md` (with orchestrator coordination)  
**Build Failures**: `debug.md` + `code.md` (orchestrator routed)

## Summary

The agent system is designed for **efficiency and quality**:
1. **Orchestrator ensures optimal routing** of all tasks
2. **Specialists provide deep domain expertise** 
3. **Intersections enable smooth collaboration**
4. **Quality gates maintain high standards**
5. **Agent hooks automate coordination**

Always start with the orchestrator - it's designed to make your work more efficient and effective!
