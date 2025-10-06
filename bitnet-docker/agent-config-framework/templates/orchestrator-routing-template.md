# Orchestrator Routing Template

This template provides the mandatory orchestrator routing section that MUST be included in every agent config.

## Template Content

```markdown
> **⚠️ MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
> **ALWAYS consult `agent-config/orchestrator.md` FIRST** for task routing, workflow coordination, 
> multi-agent needs, current project context, and agent hooks integration. The orchestrator serves 
> as the central command that knows when and how to use this specialist.
```

## Validation Rules

1. **MANDATORY**: This section MUST appear at the top of every agent config
2. **EXACT WORDING**: The warning text must be exactly as specified
3. **ORCHESTRATOR REFERENCE**: Must reference `agent-config/orchestrator.md`
4. **COMPREHENSIVE SCOPE**: Must mention all coordination aspects:
   - Task routing
   - Workflow coordination  
   - Multi-agent needs
   - Current project context
   - Agent hooks integration

## Validation Checks

- [ ] Section exists at top of agent config
- [ ] Contains exact warning emoji and formatting
- [ ] References orchestrator.md correctly
- [ ] Mentions all required coordination aspects
- [ ] Uses proper markdown formatting

## Auto-Generation Variables

- `{{AGENT_NAME}}`: Name of the specialist agent
- `{{AGENT_TYPE}}`: Type/category of the agent
- `{{UPDATE_DATE}}`: Last update date
- `{{PROJECT_PHASE}}`: Current project phase

## Integration Points

- **Generator**: `agent-config-generator.rs` auto-inserts this section
- **Validator**: `orchestrator-routing-validator.rs` validates presence and content
- **Updater**: `auto-updater.rs` updates when orchestrator patterns change