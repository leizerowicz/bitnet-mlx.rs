# Step 1: Project Discovery & Initial Planning

## Purpose
This step provides the foundational context needed for an AI agent to understand your project concept and generate comprehensive planning documents. The agent will use this information to ask targeted questions and create essential project artifacts.

## Context for Agent
You are helping a user define and plan their software project. Your role is to gather comprehensive information about their project idea and generate structured documentation that will guide the development process.

## Key Areas to Explore

### 1. Project Vision & Goals
- What is the core problem this project solves?
- Who is the target audience/users?
- What are the primary objectives and success metrics?
- What is the expected impact or value proposition?

### 2. Technical Requirements
- What type of application/system is being built?
- What are the key features and functionalities?
- Are there any specific technical constraints or requirements?
- What are the performance and scalability expectations?

### 3. Context & Environment
- What is the intended deployment environment?
- Are there existing systems this needs to integrate with?
- What are the security and compliance requirements?
- What is the expected timeline and budget considerations?

### 4. Stakeholders & Resources
- Who are the key stakeholders and decision makers?
- What development resources are available?
- Are there any existing assets, APIs, or components to leverage?
- What expertise gaps need to be addressed?

## Questions to Ask the User

Based on the context above, engage the user with specific questions to understand their project. Start with high-level vision questions and gradually dive deeper into technical and implementation details.

Example opening questions:
1. "Can you describe the main problem your project aims to solve and who would benefit from this solution?"
2. "What type of software application or system are you envisioning? (web app, mobile app, API, desktop application, etc.)"
3. "What are the 3-5 most critical features that must be included in the first version?"

## Documents to Generate

After gathering sufficient information, create the following documents:

### BACKLOG.md
- Prioritized list of features and requirements
- User stories or use cases
- Acceptance criteria for each item
- Estimated effort/complexity levels

### IMPLEMENTATION_GUIDE.md
- High-level implementation approach
- Technology stack recommendations
- Architecture overview
- Development phases and milestones
- Key technical decisions and rationale

### RISK_ASSESSMENT.md
- Identified project risks (technical, timeline, resource, market)
- Risk probability and impact analysis  
- Mitigation strategies for each risk
- Contingency plans for critical risks

### FILE_OUTLINE.md
- Proposed project structure and file organization
- Key directories and their purposes
- Important configuration files
- Documentation structure
- Testing and deployment file organization

## Instructions for Agent

1. Read and understand the project context provided by the user
2. Ask clarifying questions to fill gaps in understanding
3. Generate comprehensive versions of the four required documents
4. Ensure all documents are consistent and aligned with the project vision
5. Include specific, actionable items that can guide development
6. Consider both immediate needs and future scalability

Remember to maintain consistency across all generated documents and ensure they work together as a cohesive planning foundation.