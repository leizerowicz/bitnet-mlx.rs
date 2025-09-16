# MCP Swarm Intelligence Server - Project Setup & Migration Guide

> **Last Updated**: September 16, 2025 - **Standalone MCP Project Setup** - Based on BitNet-Rust Agent-Config Patterns

> **🎯 PROJECT MIGRATION**: This guide enables you to create a standalone MCP Swarm Intelligence Server project using the exact same orchestrator-driven agent-config workflow system from BitNet-Rust.

## Project Overview

This guide shows how to set up a **completely independent MCP server project** that implements the full BitNet-Rust agent-config workflow system. The MCP server will provide swarm intelligence capabilities for multi-agent coordination while maintaining the proven orchestrator-driven patterns.

## Quick Start - New Project Setup

### Step 1: Create New Project Directory

```bash
# Create new standalone MCP server project
mkdir swarm-mcp-server
cd swarm-mcp-server

# Initialize Git repository
git init
git add .
git commit -m "Initial commit: MCP Swarm Intelligence Server"
```

### Step 2: Copy Agent-Config System from BitNet-Rust

The agent-config system from BitNet-Rust is the **core foundation**. Copy the entire system:

```bash
# From BitNet-Rust project directory, copy the agent-config system
cp -r /path/to/bitnet-rust/agent-config ./agent-config
cp /path/to/bitnet-rust/.github/copilot-instructions.md ./.github/copilot-instructions.md

# Copy MCP-specific setup files
cp -r /path/to/bitnet-rust/mcp/* ./
```

### Step 3: Adapt Core Configuration Files

#### Update .github/copilot-instructions.md

Replace the project-specific content while keeping the **exact same workflow patterns**:

```markdown
# MCP Swarm Intelligence Server Copilot Instructions

## Project Overview

MCP Swarm Intelligence Server is a high-performance implementation of collective intelligence for multi-agent coordination, featuring agent ecosystem management, hive mind knowledge bases, and automated workflow orchestration using Model Context Protocol.

## Agent Configuration System - Orchestrator-Driven Multi-Agent Workflow

This project uses the **EXACT SAME** agent configuration system as BitNet-Rust. **THE ORCHESTRATOR IS THE CENTRAL COMMAND** that routes all work and manages all specialist coordination.

### 🎯 MANDATORY ORCHESTRATOR-FIRST WORKFLOW

[Keep exact same workflow as BitNet-Rust but adapt context to MCP development]
```

#### Update agent-config/orchestrator.md

Adapt the orchestrator for MCP development context:

```markdown
# MCP Swarm Intelligence Server Orchestrator - Primary Workflow Coordinator

> **🎯 PRIMARY ENTRY POINT**: This orchestrator is the **MAIN WORKFLOW COORDINATOR** for all MCP server development activities.

## Role Overview
You are the **PRIMARY PROJECT ORCHESTRATOR** for the MCP Swarm Intelligence Server, managing all development activities related to building a Model Context Protocol server that implements swarm intelligence patterns.

## Project Context
The MCP Swarm Intelligence Server implements collective intelligence algorithms for automated multi-agent coordination through the Model Context Protocol.

**Current Status**: 🎯 **FOUNDATION SETUP PHASE** - MCP Server Implementation (September 16, 2025)

**PRIMARY WORKFLOW**: Complete automation of multi-agent coordination using MCP protocol
**Technical Foundation**: Python-based MCP server with swarm intelligence algorithms
**Performance Goal**: Real-time agent coordination with <100ms response times
**Current Priority**: Foundation setup and core MCP implementation

[Continue with same orchestrator patterns adapted for MCP context...]
```

### Step 4: Create MCP-Specific Agent Specialists

Based on the mcp_server_guide.md, create these additional specialist agents:

#### agent-config/mcp_protocol_specialist.md

```markdown
# Model Context Protocol (MCP) Specialist

⚠️ **MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
ALWAYS consult agent-config/orchestrator.md FIRST for task routing and workflow coordination.

## Role Overview
You are the **MODEL CONTEXT PROTOCOL SPECIALIST** for the MCP Swarm Intelligence Server, focusing on MCP specification compliance, protocol implementation, and integration with VS Code and other MCP clients.

## Expertise Areas
- **MCP Specification Compliance**: Protocol standards, message formats, JSON-RPC implementation
- **Server Implementation**: Resource management, tool registration, capability negotiation
- **Client Integration**: VS Code extension integration, protocol handshake, communication
- **Protocol Debugging**: Message tracing, error handling, connection management
- **Standards Compliance**: MCP schema validation, error code handling, protocol versioning

## MCP Implementation Patterns
- **Server Lifecycle**: Initialization, capability registration, graceful shutdown
- **Tool Implementation**: Parameter validation, async execution, error propagation
- **Resource Management**: Content serving, metadata handling, URI scheme design
- **Message Handling**: Request/response patterns, notification handling, batch operations

## Intersection Patterns
- **Intersects with python_specialist.md**: Python async/await patterns and implementation
- **Intersects with api_development_specialist.md**: API design and JSON-RPC protocols
- **Intersects with code.md**: Primary implementation and debugging
- **Intersects with test_utilities_specialist.md**: MCP protocol testing and validation
```

#### agent-config/swarm_intelligence_specialist.md

```markdown
# Swarm Intelligence Algorithm Specialist

⚠️ **MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
ALWAYS consult agent-config/orchestrator.md FIRST for task routing and workflow coordination.

## Role Overview
You are the **SWARM INTELLIGENCE SPECIALIST** for the MCP server, focusing on collective intelligence algorithms, multi-agent coordination patterns, and emergent behavior systems for automated workflow management.

## Expertise Areas
- **Ant Colony Optimization**: Pheromone trails, path optimization, dynamic task assignment
- **Particle Swarm Optimization**: Agent movement patterns, convergence algorithms
- **Collective Decision Making**: Consensus algorithms, weighted voting, conflict resolution
- **Emergent Behavior**: Pattern recognition, adaptive learning, system evolution
- **Multi-Agent Coordination**: Load balancing, task distribution, collaboration optimization

## Algorithm Implementation Patterns
- **Task Assignment Optimization**: Fitness function calculation, capability matching
- **Knowledge Aggregation**: Collective memory, pattern recognition, learning systems
- **Consensus Building**: Voting mechanisms, confidence scoring, decision convergence
- **Adaptive Systems**: Performance feedback loops, algorithm tuning, evolution

## Intersection Patterns
- **Intersects with hive_mind_specialist.md**: Collective knowledge and memory systems
- **Intersects with mcp_protocol_specialist.md**: MCP tool implementation for swarm features
- **Intersects with performance_engineering_specialist.md**: Algorithm optimization
- **Intersects with code.md**: Core algorithm implementation and debugging
```

#### agent-config/hive_mind_specialist.md

```markdown
# Hive Mind Knowledge Management Specialist

⚠️ **MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
ALWAYS consult agent-config/orchestrator.md FIRST for task routing and workflow coordination.

## Role Overview
You are the **HIVE MIND SPECIALIST** for the MCP server, focusing on collective knowledge management, pattern recognition, and distributed knowledge systems that enable intelligent multi-agent coordination.

## Expertise Areas
- **Collective Memory Systems**: Knowledge storage, retrieval, and organization
- **Pattern Recognition**: Machine learning for coordination patterns, success prediction
- **Knowledge Graphs**: Semantic relationships, entity linking, knowledge synthesis
- **Distributed Knowledge**: Knowledge sharing protocols, consistency management
- **Recommendation Systems**: Context-aware suggestions, action prioritization

## Knowledge Management Patterns
- **Knowledge Contribution**: Automated extraction from agent interactions
- **Knowledge Synthesis**: Multi-source aggregation, conflict resolution
- **Pattern Learning**: Success pattern identification, failure analysis
- **Recommendation Generation**: Context-aware guidance, action suggestions

## Intersection Patterns
- **Intersects with swarm_intelligence_specialist.md**: Collective decision-making algorithms
- **Intersects with mcp_protocol_specialist.md**: Knowledge sharing through MCP resources
- **Intersects with api_development_specialist.md**: Knowledge API design and implementation
- **Intersects with documentation_writer.md**: Knowledge documentation and sharing
```

### Step 5: Implement Core MCP Server Structure

Based on the mcp_server_guide.md, create the complete project structure:

```bash
# Create Python project structure
mkdir -p src/{swarm,agents,tools,config}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs/{api,guides,examples}
mkdir -p examples/{basic,advanced}

# Create main files
touch src/server.py
touch src/__init__.py
touch requirements.txt
touch pyproject.toml
touch README.md
touch .env.example
```

### Step 6: Core Implementation Files

#### requirements.txt

```txt
# MCP Protocol
mcp>=1.0.0

# Async framework
asyncio
aiohttp>=3.8.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Machine learning (for swarm intelligence)
scikit-learn>=1.3.0
torch>=2.0.0  # Optional: for advanced pattern recognition

# Development tools
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0

# Monitoring and logging
structlog>=23.0.0
prometheus-client>=0.17.0
```

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "swarm-mcp-server"
version = "0.1.0"
description = "Swarm Intelligence MCP Server for Multi-Agent Coordination"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.11"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
```

#### src/server.py (Main MCP Server)

```python
#!/usr/bin/env python3
"""
MCP Swarm Intelligence Server

Implements collective intelligence for multi-agent coordination using the Model Context Protocol.
This server follows the exact orchestrator-driven workflow patterns from BitNet-Rust.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from mcp import Server, ListResourcesResult, ReadResourceResult, ListToolsResult
from mcp.types import Resource, TextContent, Tool, CallToolResult
from mcp.server.stdio import stdio_server

from .swarm.intelligence import SwarmIntelligence
from .swarm.hive_mind import HiveMind
from .agents.ecosystem import AgentEcosystem
from .config.settings import Settings

logger = logging.getLogger(__name__)

class SwarmMCPServer:
    """
    Main MCP server implementing swarm intelligence for agent coordination.
    
    Follows the orchestrator-driven workflow pattern from BitNet-Rust:
    1. All tasks route through orchestrator first
    2. Specialist agents handle domain-specific work
    3. Agent hooks provide automated lifecycle management
    4. Quality gates ensure consistent standards
    """
    
    def __init__(self, agent_config_dir: str):
        self.server = Server("swarm-coordination-server")
        self.settings = Settings()
        self.agent_config_dir = Path(agent_config_dir)
        
        # Initialize core components following BitNet-Rust patterns
        self.ecosystem = AgentEcosystem(self.agent_config_dir)
        self.swarm = SwarmIntelligence()
        self.hive_mind = HiveMind()
        
        # Load agent configurations (mirrors BitNet-Rust agent loading)
        self.agents = self.ecosystem.load_all_agents()
        self._initialize_swarm()
        
        self.setup_handlers()
    
    def _initialize_swarm(self):
        """Initialize swarm intelligence with loaded agents"""
        logger.info(f"Initializing swarm with {len(self.agents)} agents")
        
        for agent_name, config in self.agents.items():
            self.swarm.register_agent(
                name=agent_name,
                capabilities=config['capabilities'],
                expertise_areas=config['expertise_areas'],
                collaboration_patterns=config['intersections']
            )
            
        logger.info("Swarm intelligence initialized successfully")

    def setup_handlers(self):
        """Setup MCP protocol handlers following the orchestrator pattern"""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available swarm intelligence resources"""
            resources = [
                Resource(
                    uri="swarm://orchestrator-context",
                    name="Orchestrator Context",
                    description="Current orchestrator state and workflow management context",
                    mimeType="application/json"
                ),
                Resource(
                    uri="swarm://agent-ecosystem",
                    name="Agent Ecosystem Overview", 
                    description="Complete agent configuration ecosystem with capabilities and intersections",
                    mimeType="application/json"
                ),
                Resource(
                    uri="swarm://coordination-patterns",
                    name="Coordination Patterns Library",
                    description="Proven multi-agent coordination patterns and workflows",
                    mimeType="application/json"
                ),
                Resource(
                    uri="swarm://collective-knowledge",
                    name="Collective Knowledge Base",
                    description="Hive mind knowledge aggregated from all agents",
                    mimeType="application/json"
                ),
                Resource(
                    uri="swarm://task-assignment-matrix",
                    name="Task Assignment Matrix",
                    description="Intelligent task routing recommendations based on swarm intelligence",
                    mimeType="application/json"
                )
            ]
            return ListResourcesResult(resources=resources)

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available swarm intelligence tools"""
            return ListToolsResult(tools=[
                Tool(
                    name="orchestrator_route_task",
                    description="Route task through orchestrator for optimal agent assignment",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {"type": "string"},
                            "requirements": {"type": "array", "items": {"type": "string"}},
                            "complexity": {"type": "string", "enum": ["simple", "medium", "high", "critical"]},
                            "urgency": {"type": "string", "enum": ["low", "normal", "high", "critical"]},
                            "domain": {"type": "string"}
                        },
                        "required": ["task_description", "requirements"]
                    }
                ),
                Tool(
                    name="query_collective_intelligence",
                    description="Query hive mind for relevant knowledge and coordination patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "domain": {"type": "string"},
                            "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="coordinate_multi_agent_workflow",
                    description="Get coordination strategy for multi-agent collaboration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "workflow_type": {"type": "string"},
                            "involved_agents": {"type": "array", "items": {"type": "string"}},
                            "constraints": {"type": "array", "items": {"type": "string"}},
                            "quality_gates": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["workflow_type"]
                    }
                ),
                Tool(
                    name="achieve_swarm_consensus",
                    description="Achieve consensus through collective decision making",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "decision_point": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "stakeholders": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["decision_point", "options"]
                    }
                )
            ])

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[CallToolResult]:
            """Handle tool calls following orchestrator-driven patterns"""
            try:
                if name == "orchestrator_route_task":
                    result = await self._handle_orchestrator_routing(arguments)
                elif name == "query_collective_intelligence":
                    result = await self._handle_hive_mind_query(arguments)
                elif name == "coordinate_multi_agent_workflow":
                    result = await self._handle_workflow_coordination(arguments)
                elif name == "achieve_swarm_consensus":
                    result = await self._handle_swarm_consensus(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Tool call failed: {name} - {str(e)}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

async def main():
    """Main entry point for the MCP server"""
    import os
    
    # Get agent config directory from environment or default
    agent_config_dir = os.getenv('AGENT_CONFIG_DIR', './agent-config')
    
    if not Path(agent_config_dir).exists():
        raise FileNotFoundError(f"Agent config directory not found: {agent_config_dir}")
    
    # Initialize server with BitNet-Rust agent config patterns
    server_instance = SwarmMCPServer(agent_config_dir)
    
    # Run MCP server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream, 
            write_stream, 
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server
    asyncio.run(main())
```

### Step 7: Workflow Integration

Create `.mcp.json` configuration for VS Code integration:

```json
{
  "mcpServers": {
    "swarm-coordination": {
      "command": "python",
      "args": ["src/server.py"],
      "cwd": "./",
      "env": {
        "AGENT_CONFIG_DIR": "./agent-config",
        "SWARM_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Step 8: Agent Hooks Integration

Copy the agent-hooks.md from BitNet-Rust and add MCP-specific hooks:

```markdown
# MCP Swarm Intelligence Server Agent Hooks System

[Copy all core hooks from BitNet-Rust agent-hooks.md]

### MCP-Specific Hooks

#### MCP_SERVER_LIFECYCLE_HOOKS

## MCP_SERVER_STARTUP
- **Trigger**: MCP server initialization
- **Actions**:
  - Initialize swarm intelligence components
  - Load agent ecosystem configurations
  - Setup hive mind knowledge base
  - Register MCP tools and resources
  - Validate MCP protocol compliance

## MCP_TOOL_EXECUTION
- **Trigger**: MCP tool call received
- **Actions**:
  - Route to orchestrator for task assignment
  - Apply swarm intelligence for agent coordination
  - Update collective knowledge base
  - Track tool performance metrics
  - Return MCP-compliant responses
```

## Migration Checklist

### ✅ Foundation Setup
- [ ] Create new project directory and Git repository
- [ ] Copy entire agent-config system from BitNet-Rust
- [ ] Copy .github/copilot-instructions.md with MCP adaptations
- [ ] Update orchestrator.md for MCP development context

### ✅ MCP-Specific Configuration
- [ ] Create MCP protocol specialist agent configuration
- [ ] Create swarm intelligence specialist agent configuration
- [ ] Create hive mind specialist agent configuration
- [ ] Update agent intersections for MCP domain specialists

### ✅ Implementation Structure
- [ ] Create Python project structure with src/, tests/, docs/
- [ ] Implement main MCP server following protocol specifications
- [ ] Implement swarm intelligence algorithms for agent coordination
- [ ] Implement hive mind knowledge management system

### ✅ Integration & Testing
- [ ] Create .mcp.json for VS Code integration
- [ ] Setup agent hooks for MCP lifecycle management
- [ ] Implement comprehensive testing suite
- [ ] Validate orchestrator-driven workflow functionality

### ✅ Documentation & Deployment
- [ ] Create project README with setup instructions
- [ ] Document all MCP tools and resources
- [ ] Setup CI/CD pipeline for automated testing
- [ ] Create deployment documentation

## Project Independence

Once set up, this MCP server project will be **completely independent** from BitNet-Rust while maintaining:

1. **Identical Workflow Patterns**: Same orchestrator-driven multi-agent coordination
2. **Agent Configuration System**: Exact same agent-config structure and patterns
3. **Quality Standards**: Same testing, documentation, and code quality requirements
4. **Automation Level**: Complete workflow automation with zero manual intervention
5. **Copilot Integration**: Same GitHub Copilot workflow experience

The project can be developed, maintained, and deployed independently while benefiting from all the proven workflow patterns established in BitNet-Rust.

## Success Validation

The project setup is successful when:

- ✅ **Orchestrator-first workflow**: All development tasks route through orchestrator.md
- ✅ **Agent specialization**: Each domain has dedicated specialist with clear capabilities
- ✅ **MCP compliance**: Server implements full MCP protocol specification
- ✅ **Swarm intelligence**: Collective decision-making and optimization algorithms work
- ✅ **Complete automation**: Development workflow requires zero manual intervention
- ✅ **BitNet-Rust parity**: Identical development experience and quality standards

This setup guide ensures you can create a standalone MCP server project that maintains all the powerful workflow automation and quality standards of BitNet-Rust while focusing specifically on swarm intelligence and multi-agent coordination through the Model Context Protocol.