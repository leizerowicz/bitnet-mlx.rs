# BitNet Swarm Intelligence Container

This container provides the complete BitNet Swarm Intelligence system with dual-intelligence capabilities for VS Code extension integration.

## 🎯 Overview

The BitNet Swarm Intelligence Container delivers:
- **🐝 Swarm Intelligence**: Independent agents with collaborative decision-making for diverging tasks
- **🧠 Hive Mind Intelligence**: Unified thinking collective for large, complex coordinated tasks
- **Automatic Mode Selection**: AI system determines optimal intelligence mode based on task characteristics
- **Universal API**: Single `/api` endpoint handles all operations through natural language
- **Agent Config Integration**: Full integration with the BitNet agent configuration system

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f bitnet-swarm

# Stop the container
docker-compose down
```

### Using the Deployment Script

```bash
# Make the script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh

# The script will:
# - Build the multi-architecture Docker image
# - Start the container with proper configuration
# - Verify the deployment
# - Show usage examples
```

## 🌐 API Endpoints

### Universal Endpoint
- **URL**: `http://localhost:8080/api`
- **Method**: POST for operations, GET for status
- **Content-Type**: `application/json`

### MCP Server
- **URL**: `http://localhost:8081`
- **Type**: Model Context Protocol server for VS Code extension integration

## 📝 Usage Examples

### Code Generation (Automatic Intelligence Mode Selection)

```bash
# Simple code generation (auto-selects best intelligence mode)
curl -X POST http://localhost:8080/api \
  -H "Content-Type: application/json" \
  -d '{"prompt": "generate a Rust BitNet inference function"}'

# Complex system refactoring (likely selects hive mind mode)
curl -X POST http://localhost:8080/api \
  -H "Content-Type: application/json" \
  -d '{"prompt": "refactor this entire codebase for performance", "content": "..."}'

# Multiple optimization approaches (likely selects swarm mode)
curl -X POST http://localhost:8080/api \
  -H "Content-Type: application/json" \
  -d '{"prompt": "find different optimization opportunities in this code", "content": "..."}'
```

### System Status

```bash
# Get system status and intelligence capabilities
curl http://localhost:8080/api

# Response includes:
# - Available intelligence modes (swarm/hive mind)
# - Agent system status
# - Model loading status
# - Performance metrics
```

## 🏗️ Architecture

### Container Components
- **Inference Engine**: BitNet 1.58-bit quantization with ARM64 NEON optimizations
- **Agent Orchestrator**: Manages agent discovery, routing, and coordination
- **Intelligence Manager**: Handles swarm/hive mind mode selection and execution
- **HTTP API Server**: Universal endpoint with natural language processing
- **MCP Server**: Model Context Protocol integration for VS Code extensions

### Intelligence Modes

#### 🐝 Swarm Intelligence (Diverging Collaborative)
- **Use Cases**: Multi-approach problems, code review, architecture exploration
- **Characteristics**: Independent decision-making, consensus building, conflict resolution
- **Performance**: Optimized for creative problem-solving and diverse perspectives

#### 🧠 Hive Mind Intelligence (Unified Collective)
- **Use Cases**: Large refactoring, complex algorithms, system-wide optimization
- **Characteristics**: Unified thinking, synchronized execution, collective memory
- **Performance**: Optimized for coordinated, large-scale tasks

## 🔧 Configuration

### Environment Variables

```yaml
# Intelligence Configuration
BITNET_DEFAULT_MODE=swarm              # Default intelligence mode
BITNET_AUTO_MODE_SELECTION=true       # Enable automatic mode selection
BITNET_MODE_SWITCH_THRESHOLD=0.7      # Threshold for mode switching

# Resource Limits
BITNET_MAX_AGENTS=20                   # Maximum total agents
BITNET_SWARM_LIMIT=10                  # Maximum agents in swarm mode
BITNET_HIVE_MIND_LIMIT=20             # Maximum agents in hive mind mode

# Paths
BITNET_MODEL_PATH=/app/models          # Model storage path
BITNET_AGENT_CONFIG_PATH=/app/agent-config  # Agent configurations
BITNET_LOG_PATH=/app/logs             # Log output path
```

### Volume Mounts

- `../../agent-config:/app/agent-config:ro` - Agent configuration system (read-only)
- `../../models:/app/models` - Model cache and storage
- `../../logs:/app/logs` - Log output directory

## 📊 Performance Specifications

### Resource Requirements
- **Memory**: 400MB base + intelligence overhead (target: <2GB total)
- **CPU**: 2.0 cores recommended for optimal performance
- **Storage**: <2GB container image + model storage

### Performance Targets
- **Inference Latency**: <100ms per cognitive task
- **Intelligence Coordination**: <50ms overhead for mode selection
- **Throughput**: >10 requests/second per container
- **ARM64 Performance**: 1.37x-3.20x speedup vs standard builds

### Intelligence Mode Performance
- **Swarm Consensus**: <150ms for collaborative decision-making
- **Hive Mind Sync**: <50ms for unified understanding propagation
- **Mode Detection**: <30ms to determine optimal intelligence approach
- **Mode Switching**: <100ms to transition between modes

## 🔍 Health Monitoring

The container includes built-in health checks:

```bash
# Health check endpoint
curl -f http://localhost:8080/health

# Docker health status
docker-compose ps
```

Health check parameters:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 60 seconds

## 🐛 Troubleshooting

### Common Issues

1. **Container fails to start**: Check Docker daemon and available resources
2. **API not responding**: Verify port 8080 is not in use by another service
3. **Model loading fails**: Ensure models directory is properly mounted and accessible
4. **Agent config errors**: Verify agent-config directory is mounted and contains valid configurations

### Debug Commands

```bash
# View container logs
docker-compose logs bitnet-swarm

# Execute commands in container
docker-compose exec bitnet-swarm /bin/bash

# Check container resource usage
docker stats bitnet-swarm

# Restart the container
docker-compose restart bitnet-swarm
```

## 🔗 Integration

### VS Code Extension

The container is designed for seamless VS Code extension integration:

```typescript
import { BitNetIntelligenceClient } from './bitnet-client';

const client = new BitNetIntelligenceClient('http://localhost:8080');

// Automatic intelligence mode selection
const response = await client.ask("implement a complex sorting algorithm");
// System automatically selects swarm or hive mind based on task complexity
```

### MCP Tools

Model Context Protocol tools are available at `http://localhost:8081` for advanced VS Code integration.

## 📚 Related Documentation

- [Agent Configuration System](../../agent-config/) - Learn about the agent system
- [DOCKER_BITNET_SWARM_TODO.md](../../DOCKER_BITNET_SWARM_TODO.md) - Implementation details
- [BitNet Docker Overview](../README.md) - Overall Docker architecture