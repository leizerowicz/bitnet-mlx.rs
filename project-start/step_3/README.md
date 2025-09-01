# Step 3: Expert Context File Generation

## Purpose
This step guides the creation of specialized expert context files that provide deep, focused knowledge for different aspects of your project. These expert files enable agents to provide specialized guidance on specific domains, technologies, or project concerns.

## Context for Agent
You are helping create expert knowledge bases that will serve as specialized consultants for different aspects of a software project. Each expert file should contain comprehensive, actionable guidance within its domain of expertise.

## Instructions for Creating Expert Context Files

### Accessing Previous Context
Before creating expert files, ensure you have access to:
- **Step 1 outputs**: BACKLOG.md, IMPLEMENTATION_GUIDE.md, RISK_ASSESSMENT.md, FILE_OUTLINE.md
- **Step 2 analysis**: SPARC methodology application and phase documents
- **Current project state**: Any existing code, documentation, or architectural decisions

### Expert File Categories

Create expert context files for the following categories as relevant to your project:

#### 1. Project Type Expert (`project_type_expert.md`)
**Focus**: Deep understanding of the specific type of software being built
- Domain-specific best practices and patterns
- Common challenges and solutions for this project type
- Industry standards and compliance requirements
- User experience considerations specific to the domain
- Performance and scalability patterns for this type of application

#### 2. Architecture Expert (`architecture_expert.md`)
**Focus**: System design and architectural decisions
- Architectural patterns and their trade-offs
- Component design and interaction patterns
- Data flow and state management strategies
- Integration patterns and API design
- Scalability and reliability considerations
- Technology stack optimization

#### 3. Development Methodology Expert (`methodology_expert.md`)
**Focus**: Development process and team coordination  
- Agile/Scrum/other methodology implementation
- Code review and quality assurance processes
- Testing strategies and automation
- Continuous integration/deployment practices
- Team collaboration and communication protocols
- Project management and milestone tracking

#### 4. Technology Stack Expert (`tech_stack_expert.md`)
**Focus**: Specific technologies, languages, and frameworks
- Language-specific best practices and idioms
- Framework optimization and advanced features
- Library selection and integration strategies
- Performance optimization techniques
- Security considerations for chosen technologies
- Debugging and troubleshooting approaches

#### 5. Tools Expert (`tools_expert.md`)
**Focus**: Development tools, IDEs, and productivity systems
- Development environment setup and optimization
- Build tools and automation systems
- Testing framework selection and configuration
- Monitoring and logging tool integration
- Documentation and collaboration tools
- DevOps and deployment tool chains

#### 6. Implementation-Specific Experts
Create focused experts for major features or components:
- **Authentication Expert** (`auth_expert.md`)
- **Database Expert** (`database_expert.md`)  
- **API Expert** (`api_expert.md`)
- **UI/UX Expert** (`ui_expert.md`)
- **Performance Expert** (`performance_expert.md`)
- **Security Expert** (`security_expert.md`)

#### 7. Process and Communication Experts
Create specialized experts for project coordination and communication:
- **Orchestrator Expert** (`orchestrator_expert.md`) - Provides guidance on which context files to use for different tasks
- **Ask Expert** (`ask_expert.md`) - Specializes in user interaction, inquiry formulation, and requirement gathering
- **Code Expert** (`code_expert.md`) - Focuses on implementation strategies, coding standards, and best practices
- **Debug Expert** (`debug_expert.md`) - Specializes in troubleshooting, debugging methodologies, and issue resolution
- **Developmental Phase Tracker Expert** (`phase_tracker_expert.md`) - Monitors project progress and milestone management
- **Documentation Writer Expert** (`documentation_expert.md`) - Specializes in comprehensive documentation creation and maintenance
- **Error Handling Specialist Expert** (`error_handling_expert.md`) - Focuses on error management, recovery strategies, and resilience
- **Project Research Expert** (`research_expert.md`) - Conducts domain and technology research, market analysis, and feasibility studies
- **Truth Validator Expert** (`truth_validator_expert.md`) - Verifies information accuracy, validates assumptions, and ensures data integrity
- **Variable and API Matcher Expert** (`api_matcher_expert.md`) - Ensures interface consistency, data mapping, and API compatibility

## Expert File Template

When creating each expert file, use this structure:

```markdown
# [Domain] Expert Context

## Expertise Domain
[Clear definition of what this expert specializes in]

## Project Context Integration
[How this expertise applies to the current project based on Step 1 & 2 outputs]

## Core Knowledge Areas
[List of key topics this expert covers]

## Best Practices
[Domain-specific best practices and guidelines]

## Common Challenges & Solutions
[Typical problems in this domain and proven solutions]

## Decision Framework
[How to make good decisions within this domain]

## Tools & Resources
[Recommended tools, libraries, and resources]

## Quality Criteria
[How to evaluate success in this domain]

## Integration Points
[How this domain connects with other aspects of the project]

## Troubleshooting Guide
[Common issues and debugging approaches]
```

## Creation Process for Agents

### 1. Analyze Project Needs
- Review Step 1 and Step 2 outputs
- Identify which expert domains are most critical
- Prioritize expert files based on project complexity and risk

### 2. Gather Context
- Extract relevant information from previous phases
- Identify domain-specific requirements and constraints
- Consider team expertise and resource availability

### 3. Create Expert Files
- Start with the most critical experts for the project
- Ensure each expert file is comprehensive but focused
- Cross-reference between expert files where domains overlap
- Include specific examples and code snippets where helpful

### 4. Validate Integration
- Ensure expert files work together cohesively
- Check for consistency in recommendations across experts
- Verify that expert advice aligns with project constraints and goals

## Questions to Guide Expert File Creation

For each expert domain, consider:
1. What are the 5-10 most important things to know in this domain for this project?
2. What decisions will need to be made, and what criteria should guide them?
3. What are the most common mistakes or pitfalls to avoid?
4. What tools, resources, or frameworks are essential?
5. How does this domain interact with other aspects of the project?
6. What quality standards or metrics should be applied?
7. What troubleshooting knowledge is crucial?

## Using Expert Files

Once created, these expert files serve as:
- **Reference Guides**: Quick access to domain-specific knowledge
- **Decision Support**: Frameworks for making good choices within each domain
- **Quality Assurance**: Checklists and criteria for evaluating work
- **Onboarding Resources**: Helping new team members understand domain-specific aspects
- **Consistency Tools**: Ensuring consistent approaches across the project

## Maintenance

Expert files should be:
- Updated as the project evolves
- Refined based on lessons learned during development
- Extended with new knowledge and insights
- Reviewed periodically for accuracy and relevance

Remember: These expert files are meant to be practical, actionable resources that make development more efficient and successful.