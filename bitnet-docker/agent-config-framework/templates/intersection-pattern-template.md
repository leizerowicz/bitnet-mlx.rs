# Agent Intersection Pattern Template

This template provides the structure for defining agent intersection relationships in the BitNet Docker Swarm Intelligence system.

## Template Content

```markdown
## Agent Intersection Matrix
**This agent intersects with the following agents for coordinated workflows:**

{{INTERSECTION_RELATIONSHIPS}}

### Primary Intersections
{{PRIMARY_INTERSECTIONS}}

### Secondary Intersections
{{SECONDARY_INTERSECTIONS}}

### Coordination Intersections
{{COORDINATION_INTERSECTIONS}}

### Validation Intersections
{{VALIDATION_INTERSECTIONS}}

## Coordination Patterns
**This agent follows these coordination patterns:**

{{COORDINATION_PATTERNS}}

### Swarm Mode Coordination
{{SWARM_COORDINATION_PATTERNS}}

### Hive Mind Mode Coordination
{{HIVE_MIND_COORDINATION_PATTERNS}}

## Handoff Procedures
**Standard handoff procedures for this agent:**

{{HANDOFF_PROCEDURES}}
```

## Variable Definitions

### Intersection Relationship Format
```markdown
- **{{AGENT_NAME}}**: {{COORDINATION_METHOD}} ({{RELATIONSHIP_TYPE}})
  - **Shared Responsibilities**: {{SHARED_RESPONSIBILITIES}}
  - **Handoff Pattern**: {{HANDOFF_PATTERN}}
  - **Quality Gates**: {{QUALITY_GATES}}
```

### Relationship Types
- **Primary**: Direct collaboration on shared domain tasks
- **Secondary**: Supporting role with complementary capabilities
- **Coordination**: Workflow management and task routing
- **Validation**: Quality assurance and review processes
- **Escalation**: Error handling and issue resolution

### Coordination Method Examples
- **Direct collaboration on {domain} tasks**
- **Supporting {domain} with {capability} expertise**
- **Workflow coordination and task management**
- **Quality validation for {domain} outputs**
- **Escalation path for {domain} issues**

### Coordination Pattern Examples

#### Swarm Mode Patterns
- **Independent {domain} work with collaborative integration**
- **Parallel {domain} exploration with consensus building**
- **Distributed {domain} analysis with result synthesis**
- **Autonomous {domain} development with coordination checkpoints**

#### Hive Mind Mode Patterns
- **Unified {domain} implementation with synchronized execution**
- **Collective {domain} processing with shared mental models**
- **Coordinated {domain} optimization with unified targets**
- **Synchronized {domain} development with perfect alignment**

## Handoff Procedure Templates

### Task Completion Handoff
```markdown
1. **Pre-Handoff Validation**: Verify task completion against acceptance criteria
2. **Context Preparation**: Prepare comprehensive context for receiving agent
3. **Quality Gate Check**: Execute quality validation through appropriate validation agent
4. **Handoff Execution**: Transfer task ownership with full context and status
5. **Confirmation**: Receive acknowledgment from receiving agent
6. **Post-Handoff Monitoring**: Monitor initial integration and provide support if needed
```

### Error Escalation Handoff
```markdown
1. **Error Documentation**: Document error details, context, and attempted solutions
2. **Impact Assessment**: Assess error impact on current and downstream tasks
3. **Escalation Routing**: Route to appropriate escalation agent based on error type
4. **Context Transfer**: Provide complete context including reproduction steps
5. **Collaboration Setup**: Establish collaboration channel for resolution
6. **Resolution Tracking**: Track resolution progress and integrate solution
```

### Quality Gate Handoff
```markdown
1. **Deliverable Preparation**: Prepare deliverables for quality validation
2. **Validation Request**: Submit formal validation request with acceptance criteria
3. **Validator Coordination**: Coordinate with appropriate validation specialist
4. **Feedback Integration**: Integrate validation feedback and recommendations
5. **Re-validation**: Submit for re-validation if changes were required
6. **Approval Integration**: Integrate approved deliverables into workflow
```

## Auto-Generation Rules

### Domain-Based Intersections
- **Code Domain**: Intersects with `rust_best_practices_specialist`, `debug`, `test_utilities_specialist`
- **Inference Domain**: Intersects with `performance_engineering_specialist`, `api_development_specialist`
- **Performance Domain**: Intersects with `inference_engine_specialist`, `architect`
- **Security Domain**: Intersects with `rust_best_practices_specialist`, `architect`
- **Documentation Domain**: Intersects with `ask`, all domain specialists for content

### Role-Based Intersections
- **Specialists**: Intersect with other specialists in related domains
- **Orchestrator**: Intersects with ALL agents for coordination
- **Support Agents**: Intersect with specialists they support
- **Utility Agents**: Intersect based on utility function

### Intelligence Mode Intersections
- **Swarm Capable Agents**: Can coordinate in divergent collaborative patterns
- **Hive Mind Capable Agents**: Can synchronize in unified collective patterns
- **Dual Mode Agents**: Support both coordination patterns with mode switching

## Validation Rules

1. **Completeness**: All agent types must have defined intersection patterns
2. **Consistency**: Intersection relationships must be bidirectional and consistent
3. **Quality Gates**: All intersections must specify quality gate procedures
4. **Handoff Procedures**: All intersections must define clear handoff procedures
5. **Intelligence Modes**: All intersections must support both swarm and hive mind coordination

## Integration with Framework

- **Generator**: `IntersectionMatrixUpdater` uses this template for auto-generation
- **Validator**: `OrchestratorRoutingValidator` validates against this template
- **Updater**: Framework automatically updates intersections when agents change