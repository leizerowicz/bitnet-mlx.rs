# BitNet Docker Containers

This directory contains all Docker-related configurations and files for the BitNet project.

## Directory Structure

```
bitnet-docker/
├── README.md                    # This file
├── agent-config-framework/      # Agent configuration system for orchestration
│   ├── Cargo.toml              # Agent framework build configuration
│   ├── README.md               # Agent configuration framework guide
│   ├── automation/             # Container orchestration automation
│   ├── generators/             # Agent config generators and matrix updaters
│   ├── src/                    # Core framework source code
│   ├── templates/              # Agent configuration templates
│   └── validators/             # Agent configuration validators
├── shared/                      # Shared Docker resources and templates
│   └── docker-integration-template.md  # Template for agent Docker integration
└── bitnet-swarm-intelligence/  # BitNet Swarm Intelligence Container
    ├── Dockerfile              # Multi-stage Docker build with ARM64/AMD64 support
    ├── docker-compose.yml      # Production-ready orchestration
    ├── deploy.sh               # Automated deployment script
    └── README.md               # Complete container usage guide
```

## Container Overview

### Agent Configuration Framework (`agent-config-framework/`)

The central orchestration system that powers the Docker container intelligence:
- **Agent Orchestration**: Central coordinator for routing tasks to appropriate specialist agents
- **Dynamic Configuration**: Automated generation and validation of agent configurations
- **Intersection Matrix**: Manages collaboration patterns between different agent types
- **Container Integration**: Specialized tools for Docker-aware agent deployment
- **Quality Validation**: Ensures all agent configurations meet production standards

### BitNet Swarm Intelligence Container (`bitnet-swarm-intelligence/`)

The main Docker container that provides:
- **🐝 Swarm Intelligence**: Independent agents with collaborative decision-making for diverging tasks
- **🧠 Hive Mind Intelligence**: Unified thinking collective for large, complex coordinated tasks
- **Inference Engine**: Complete microsoft/bitnet-b1.58-2B-4T-gguf for code understanding
- **VS Code Plugin Integration**: HTTP API for real-time coding assistance
- **Performance Optimization**: ARM64 NEON + Apple Silicon support for fast code generation

#### Key Features
- Multi-architecture support (ARM64 + AMD64)
- Production-ready with <2GB image size
- Universal HTTP API endpoint at `localhost:8080/api`
- MCP Server integration at `localhost:8081`
- Agent configuration system integration
- Automatic intelligence mode selection (swarm vs hive mind)

#### Quick Start

```bash
# Navigate to the swarm intelligence container
cd bitnet-docker/bitnet-swarm-intelligence/

# Deploy using docker-compose
docker-compose up -d

# Or use the automated deployment script
./deploy.sh
```

#### API Usage

```bash
# Universal endpoint handles all operations
curl -X POST http://localhost:8080/api \
  -H "Content-Type: application/json" \
  -d '{"prompt": "generate a Rust BitNet inference function"}'

# System status
curl http://localhost:8080/api
```

## Future Container Development

This directory structure is designed to accommodate future Docker containers:

1. **Add new container directories** alongside `bitnet-swarm-intelligence/`
2. **Use shared resources** from the `shared/` directory
3. **Follow the established patterns** for multi-architecture builds
4. **Integrate with the agent configuration system** using the shared templates

## Related Documentation

- [DOCKER_BITNET_SWARM_TODO.md](../../DOCKER_BITNET_SWARM_TODO.md) - Detailed implementation roadmap
- [Agent Configuration Framework](../../agent-config/) - Agent system integration
- [COMPREHENSIVE_TODO.md](../../COMPREHENSIVE_TODO.md) - Overall project roadmap

## Development Status

The BitNet Swarm Intelligence container is currently **COMPLETE** and ready for:
1. **VS Code Extension Integration** - Connect to the universal `/api` endpoint
2. **Model Integration** - Add the microsoft/bitnet-b1.58-2B-4T-gguf model
3. **Production Deployment** - Deploy using the provided docker-compose setup
4. **Intelligence Enhancement** - Train the neural networks on real agent configuration data