# Docker Integration Template

This template provides the Docker container integration section for agent configs in the BitNet Swarm Intelligence system.

## Template Content

```markdown
## Docker Container Integration
- **Container Role**: {{CONTAINER_ROLE}} agent in BitNet swarm - {{ROLE_DESCRIPTION}}
- **API Endpoints**: {{API_ENDPOINTS}}
- **MCP Tools**: {{MCP_TOOLS}}
- **Resource Requirements**: {{RESOURCE_REQUIREMENTS}}
- **Coordination Patterns**: **Swarm Mode**: {{SWARM_COORDINATION}}. **Hive Mind Mode**: {{HIVE_MIND_COORDINATION}}

## 🎯 DOCKER BITNET SWARM INTELLIGENCE CAPABILITIES

### 🐝 Swarm Intelligence {{AGENT_TYPE}} (Diverging Collaborative {{DOMAIN}})
**Use Cases for {{AGENT_NAME}} in Swarm Mode**:
{{SWARM_USE_CASES}}

### 🧠 Hive Mind Intelligence {{AGENT_TYPE}} (Unified Collective {{DOMAIN}})
**Use Cases for {{AGENT_NAME}} in Hive Mind Mode**:
{{HIVE_MIND_USE_CASES}}
```

## Variable Definitions

### Container Role Options
- **PRIMARY**: Core functionality agent (orchestrator, main development)
- **SECONDARY**: Supporting functionality agent (testing, documentation)
- **SPECIALIST**: Domain-specific expert agent (performance, security)
- **SUPPORT**: Infrastructure and utility agent (hooks, validation)

### API Endpoints Pattern
```
/api (universal endpoint), /agents/{{AGENT_TYPE}}/{{ACTION}}, /{{DOMAIN}}/{{SPECIFIC_ENDPOINT}}
```

### MCP Tools Pattern
```
{{DOMAIN}}-{{ACTION}}, {{AGENT_TYPE}}-tools, {{SPECIALTY}}-utilities
```

### Resource Requirements Options
- **High CPU, High Memory**: Complex computation agents (inference, performance)
- **Medium CPU, Medium Memory**: Standard development agents (code, debug)
- **Low CPU, Low Memory**: Coordination and utility agents (hooks, validation)
- **GPU Required**: Acceleration and ML agents (inference engine, performance)

### Coordination Patterns

#### Swarm Mode Patterns
- **Independent Feature Development**: Autonomous work with collaborative integration
- **Parallel Exploration**: Different approaches with consensus building
- **Divergent Analysis**: Multiple perspectives with synthesis
- **Collaborative Problem-Solving**: Distributed work with coordination

#### Hive Mind Patterns
- **Unified Implementation**: Synchronized work with shared mental models
- **Coordinated Optimization**: System-wide changes with unified targets
- **Collective Processing**: Massive parallel work on single objectives
- **Synchronized Execution**: Perfect alignment across all agents

## Intelligence Mode Use Cases Template

### Swarm Use Cases Format
```markdown
- **{{TASK_TYPE}}**: {{INDEPENDENT_WORK_DESCRIPTION}}, then {{COLLABORATION_DESCRIPTION}}
- **{{EXPLORATION_TYPE}}**: {{DIVERGENT_APPROACH_DESCRIPTION}}, then {{CONSENSUS_DESCRIPTION}}
- **{{ANALYSIS_TYPE}}**: {{MULTI_PERSPECTIVE_DESCRIPTION}}, then {{SYNTHESIS_DESCRIPTION}}
```

### Hive Mind Use Cases Format
```markdown
- **{{SYSTEM_TASK}}**: {{UNIFIED_STRATEGY_DESCRIPTION}} with {{COORDINATION_DESCRIPTION}}
- **{{COMPLEX_IMPLEMENTATION}}**: {{SYNCHRONIZED_WORK_DESCRIPTION}} with {{SHARED_TARGETS}}
- **{{OPTIMIZATION_TASK}}**: {{COORDINATED_CHANGES_DESCRIPTION}} with {{UNIFIED_GOALS}}
```

## Validation Rules

1. **Container Role**: Must be one of the defined options
2. **API Endpoints**: Must follow the established patterns
3. **Intelligence Modes**: Must include both swarm and hive mind capabilities
4. **Use Cases**: Must provide specific examples for each intelligence mode
5. **Coordination Patterns**: Must specify different behaviors for each mode

## Integration with Framework

- **Generator**: Automatically fills variables based on agent type and domain
- **Validator**: Ensures all Docker integration elements are present and valid
- **Updater**: Updates integration patterns when Docker architecture changes