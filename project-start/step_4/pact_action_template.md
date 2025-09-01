# PACT Action Phase: Execution Coordination Template

## Project Context
**Project Name**: [PROJECT_NAME]
**Current Phase**: Action - Coordinated Execution
**Agent Ecosystem**: [BRIEF_DESCRIPTION_OF_AGENTS]
**Execution Timeline**: [START_DATE - END_DATE]

## Execution Strategy

### Implementation Sequence
Define the order of task execution and critical coordination points:

#### Sprint/Phase 1: [PHASE_NAME]
- **Duration**: [TIMEFRAME]
- **Primary Agents**: [LIST_OF_AGENTS]
- **Key Deliverables**:
  - [Deliverable 1] - Agent: [AGENT_NAME]
  - [Deliverable 2] - Agent: [AGENT_NAME]
  - [Deliverable N] - Agent: [AGENT_NAME]
- **Coordination Checkpoints**: [WHEN_AGENTS_SYNCHRONIZE]

#### Sprint/Phase 2: [PHASE_NAME]
[Repeat structure for each phase]

### Agent Activation Triggers
Define conditions that activate different agents:

- **[AGENT_NAME]** activates when:
  - [Trigger Condition 1]
  - [Trigger Condition 2]
  - [Trigger Condition N]

- **[AGENT_NAME]** activates when:
  - [Trigger Condition 1]
  - [Trigger Condition 2]
  - [Trigger Condition N]

### Parallel Execution Coordination
Strategies for managing simultaneous agent work:

- **Work Isolation**: [How agents avoid stepping on each other]
- **Shared Resource Management**: [Coordination for shared files, databases, etc.]
- **Integration Points**: [When parallel work streams come together]
- **Conflict Prevention**: [Preventing agents from working on conflicting changes]

## Real-Time Coordination

### Status Broadcasting System
How agents communicate their current status:

```json
{
  "agent_id": "string",
  "status": "working|blocked|completed|error",
  "current_task": "string",
  "progress_percentage": "number",
  "estimated_completion": "timestamp",
  "blocking_issues": ["list of blockers"],
  "next_coordination_point": "timestamp",
  "outputs_ready": ["list of completed deliverables"]
}
```

### Dynamic Task Reallocation
Procedures for reassigning tasks based on changing conditions:

#### Reallocation Triggers
- Agent becomes unavailable or overloaded
- Task proves more complex than estimated
- Dependencies change or become blocked
- New urgent requirements emerge

#### Reallocation Process
1. **Detection**: [How reallocation needs are identified]
2. **Assessment**: [Evaluating alternative assignments]
3. **Negotiation**: [How agents/system decides on reassignment]
4. **Handover**: [Process for transferring work between agents]
5. **Validation**: [Ensuring successful transfer]

### Emergency Coordination Protocols
Procedures for handling urgent issues or system failures:

#### Emergency Types
- **Agent Failure**: [One or more agents become unavailable]
- **Resource Failure**: [Critical shared resources become unavailable]
- **Communication Breakdown**: [Agents cannot communicate effectively]
- **Quality Crisis**: [Major quality issues discovered]
- **External Dependencies**: [External systems or APIs fail]

#### Emergency Response
1. **Alert System**: [How emergencies are detected and communicated]
2. **Escalation Chain**: [Who gets notified and in what order]
3. **Temporary Measures**: [Short-term workarounds]
4. **Recovery Procedures**: [Steps to restore normal operation]
5. **Post-Incident Review**: [Learning from emergencies]

## Quality Assurance During Execution

### Continuous Validation
Ongoing quality checks throughout execution:

#### Automated Validation
- **Code Quality Gates**: [Automated checks for code standards]
- **Integration Tests**: [Continuous integration testing]
- **Performance Monitoring**: [Real-time performance validation]
- **Security Scanning**: [Ongoing security validation]

#### Agent-Driven Validation
- **Self-Validation**: [How agents check their own work]
- **Peer Review**: [How agents review each other's work]
- **Cross-Validation**: [Validation across agent boundaries]

### Peer Review Mechanisms
How agents review and approve each other's work:

#### Review Assignment
- **Automatic Assignment**: [Rules for assigning reviewers]
- **Expertise Matching**: [Matching reviewers to relevant expertise]
- **Workload Balancing**: [Distributing review workload]

#### Review Process
1. **Submission**: [How work is submitted for review]
2. **Review Criteria**: [What reviewers should check]
3. **Feedback Format**: [Standardized feedback format]
4. **Resolution Process**: [How feedback is addressed]
5. **Approval Mechanism**: [How final approval is granted]

### Integration Testing
Continuous integration of agent contributions:

- **Merge Coordination**: [How code/content from multiple agents is merged]
- **Integration Validation**: [Testing integrated components]
- **Rollback Procedures**: [What to do if integration fails]
- **Dependency Validation**: [Ensuring dependencies remain valid]

## Progress Monitoring

### Metrics and KPIs
Key performance indicators for execution phase:

#### Individual Agent Metrics
- **Task Completion Rate**: [Target: X tasks per day]
- **Quality Score**: [Target: X% quality rating]  
- **Response Time**: [Target: X minutes to respond to coordination requests]
- **Error Rate**: [Target: <X% error rate]

#### Coordination Metrics
- **Communication Efficiency**: [Target: X% messages result in action]
- **Coordination Overhead**: [Target: <X% time spent on coordination]
- **Conflict Resolution Time**: [Target: X minutes average resolution time]
- **Resource Contention**: [Target: <X% time waiting for resources]

#### System-Wide Metrics
- **Overall Progress**: [Target: X% completion per week]
- **Integration Success Rate**: [Target: X% successful integrations]
- **Quality Consistency**: [Target: X% variation in quality across agents]
- **Timeline Adherence**: [Target: ±X% variance from planned timeline]

### Bottleneck Detection
Early warning systems for coordination issues:

#### Bottleneck Types
- **Agent Overload**: [Agent receiving too many tasks]
- **Resource Contention**: [Multiple agents waiting for same resource]
- **Communication Delays**: [Slow response times affecting coordination]
- **Quality Issues**: [Increasing error rates or review failures]

#### Detection Mechanisms
- **Automated Monitoring**: [Metrics that trigger alerts]
- **Agent Self-Reporting**: [How agents report capacity issues]
- **Performance Trend Analysis**: [Identifying degrading performance]

#### Response Procedures
1. **Immediate Response**: [Quick actions to alleviate bottlenecks]
2. **Root Cause Analysis**: [Understanding why bottlenecks occurred]
3. **Systematic Resolution**: [Long-term fixes for recurring issues]
4. **Prevention Measures**: [Avoiding similar bottlenecks in future]

### Performance Optimization
Real-time optimization of agent coordination:

#### Optimization Areas
- **Task Assignment**: [Improving task-to-agent matching]
- **Communication Patterns**: [Reducing communication overhead]
- **Resource Utilization**: [Maximizing efficient resource use]
- **Workflow Efficiency**: [Streamlining agent workflows]

#### Optimization Triggers
- **Performance Degradation**: [When to trigger optimization]
- **Capacity Changes**: [Optimizing when agent capacity changes]
- **Requirement Evolution**: [Adapting to changing requirements]

## Risk Management During Execution

### Real-Time Risk Monitoring
- **Agent Performance Risks**: [Monitoring agent effectiveness]
- **Coordination Breakdown Risks**: [Early detection of coordination failures]
- **Quality Risks**: [Monitoring for quality degradation]
- **Timeline Risks**: [Tracking schedule adherence]

### Risk Response Strategies
1. **Preventive Measures**: [Actions to prevent risks from materializing]
2. **Early Intervention**: [Quick responses to emerging risks]
3. **Damage Control**: [Minimizing impact when risks occur]
4. **Recovery Procedures**: [Restoring normal operation after incidents]

## Communication Protocols

### Standard Communication Patterns
- **Status Updates**: [Format and frequency of status communications]
- **Request/Response**: [How agents request help or resources from others]
- **Broadcast Announcements**: [System-wide announcements and alerts]
- **Progress Reports**: [Formal progress reporting mechanisms]

### Communication Optimization
- **Batch Communications**: [Grouping related messages]
- **Priority Queuing**: [Handling urgent vs. routine communications]
- **Context Compression**: [Efficiently sharing context information]
- **Noise Reduction**: [Filtering out unnecessary communications]

## Quality Gates

### Execution Quality Gates
Checkpoints that must be passed during execution:

#### Gate 1: [GATE_NAME]
- **Trigger**: [When this gate is activated]
- **Criteria**: [What must be validated]
- **Responsible Agent**: [Who validates]
- **Pass/Fail Actions**: [What happens based on validation results]

#### Gate 2: [GATE_NAME]
[Repeat structure for each quality gate]

### Continuous Quality Monitoring
- **Real-Time Quality Metrics**: [Metrics tracked continuously]
- **Quality Trend Analysis**: [Identifying quality patterns]
- **Predictive Quality Alerts**: [Early warning for quality issues]

## Execution Documentation

### Live Documentation
Documentation that updates automatically during execution:
- **Agent Activity Logs**: [Real-time logs of agent activities]
- **Decision Audit Trail**: [Record of all coordination decisions]
- **Progress Documentation**: [Auto-updating progress reports]
- **Quality Metrics Dashboard**: [Real-time quality visualization]

### Execution Artifacts
Key artifacts produced during execution:
- **Coordination Decisions Log**: [Record of all coordination decisions made]
- **Performance Analytics**: [Analysis of execution performance]
- **Issue Resolution Log**: [Record of issues encountered and resolved]
- **Optimization History**: [Record of optimizations implemented]

## Reflection

### Execution Effectiveness Analysis
[Analyze how effectively the execution phase met its objectives]

### Coordination Challenges Overcome  
[Describe major coordination challenges and how they were resolved]

### Agent Performance Assessment
[Evaluate how well individual agents performed and collaborated]

### Process Improvements Identified
[List specific improvements that could enhance future execution phases]

### Lessons Learned
[Key insights gained during execution that can benefit future projects]

### Recommendations for Future Projects
[Recommendations for improving execution coordination in similar projects]