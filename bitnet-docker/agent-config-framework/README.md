# BitNet Agent Config Framework

A comprehensive framework for managing agent configurations in the BitNet Docker Swarm Intelligence system.

**Location**: `bitnet-docker/agent-config-framework/` - Part of the Docker container orchestration system

## 🎯 Overview

The Agent Config Framework provides standardized tools for creating, validating, and managing agent configurations in the BitNet-Rust ecosystem. It ensures consistency, enforces quality standards, and enables seamless integration with Docker container deployment.

**Container Integration**: This framework is specifically designed to work with the BitNet Swarm Intelligence Docker container (`../bitnet-swarm-intelligence/`) and provides the agent orchestration backbone for the containerized system.

## ✨ Features

### 🤖 Agent Config Generation
- **Standardized Templates**: Consistent Markdown templates with variable substitution
- **Interactive CLI**: Guided agent creation with prompts and validation
- **Docker Integration**: Built-in support for containerized deployment
- **Intelligence Modes**: Native support for Swarm and Hive Mind intelligence patterns

### 🔍 Validation & Quality Assurance
- **Mandatory Orchestrator Routing**: Ensures all agents follow coordination patterns
- **Docker Compatibility**: Validates container integration requirements
- **Intelligence Mode Support**: Verifies swarm and hive mind capabilities
- **Comprehensive Reporting**: Detailed validation reports with error analysis

### 🔄 Automatic Management
- **Intersection Matrix Updates**: Auto-discover and update agent relationships
- **Agent Discovery**: Automatic detection of available agents in containers
- **Configuration Sync**: Keep agent configs synchronized with framework updates

### 🐳 Container Orchestration
- **Docker Native**: Built for containerized deployment from the ground up
- **Resource Management**: Intelligent resource allocation and scaling
- **Health Monitoring**: Continuous monitoring of agent system health
- **Intelligence Coordination**: Dynamic switching between swarm and hive mind modes

## 🚀 Quick Start

### Installation

```bash
cd bitnet-docker/agent-config-framework
cargo build --release
```

### Generate a New Agent

```bash
# Interactive mode (recommended)
./target/release/agent-config-cli generate --interactive

# Direct generation
./target/release/agent-config-cli generate \
    --name "my_specialist" \
    --domain "analysis" \
    --container-role "Specialist"
```

### Validate Existing Configs

```bash
# Basic validation
./target/release/agent-config-cli validate

# Detailed validation with fix suggestions
./target/release/agent-config-cli validate --verbose --fix
```

### Deploy Container System

```bash
# Deploy with monitoring
./target/release/agent-config-cli deploy --monitor

# Force rebuild and deploy
./target/release/agent-config-cli deploy --rebuild --monitor
```

## 📋 Agent Config Structure

Every generated agent config follows this standardized structure:

```markdown
# [Agent Name] Specialist

> **⚠️ MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
> **ALWAYS consult `agent-config/orchestrator.md` FIRST** for task routing, workflow coordination, 
> multi-agent needs, current project context, and agent hooks integration.

## Docker Container Integration
- **Container Role**: [PRIMARY/SECONDARY/SPECIALIST/SUPPORT]
- **API Endpoints**: [List of HTTP endpoints]
- **MCP Tools**: [List of MCP tools]
- **Resource Requirements**: [CPU, Memory, GPU requirements]
- **Coordination Patterns**: Swarm and Hive Mind coordination patterns

## 🎯 DOCKER BITNET SWARM INTELLIGENCE CAPABILITIES

### 🐝 Swarm Intelligence [Agent Type] (Diverging Collaborative [Domain])
**Use Cases for [Agent] in Swarm Mode**:
- Independent [domain] development with collaborative integration
- [Domain] exploration with consensus building
- Parallel [domain] analysis with result synthesis

### 🧠 Hive Mind Intelligence [Agent Type] (Unified Collective [Domain])
**Use Cases for [Agent] in Hive Mind Mode**:
- Unified [domain] implementation across entire system
- Coordinated [domain] optimization with shared targets
- Synchronized [domain] processing with collective intelligence

## Specialist Role & Niche
[Detailed role description and responsibilities]

## Agent Intersection Matrix
[Auto-generated intersection relationships]

## Agent Hooks Integration
[Lifecycle hook integration patterns]
```

## 🏗️ Architecture

### Core Components

```
agent-config-framework/
├── templates/           # Standardized agent config templates
│   ├── specialist-agent-template.md
│   ├── orchestrator-routing-template.md
│   └── docker-integration-template.md
├── generators/          # Code generation tools
│   ├── agent-config-generator.rs
│   └── intersection-matrix-updater.rs
├── validators/          # Validation and quality assurance
│   └── orchestrator-routing-validator.rs
├── automation/          # Container orchestration
│   └── container-orchestration.rs
└── src/                # Main library and CLI
    ├── lib.rs
    └── cli/main.rs
```

### Intelligence System Integration

The framework is designed around the BitNet Docker Swarm Intelligence architecture:

#### 🐝 Swarm Intelligence (Diverging Collaborative)
- **Independent Decision-Making**: Agents work autonomously on different aspects
- **Collaborative Convergence**: Results are synthesized through consensus building
- **Conflict Resolution**: Built-in mechanisms for handling disagreements
- **Emergent Solutions**: Complex solutions emerge from agent interactions

#### 🧠 Hive Mind Intelligence (Unified Collective)
- **Synchronized Thinking**: All agents share unified mental models
- **Collective Processing**: Massive parallel processing on single objectives
- **Perfect Coordination**: No conflicts, unified decision-making
- **Amplified Intelligence**: Combined processing power focused on single goals

## 🔧 Configuration

### Framework Configuration

```rust
use agent_config_framework::{FrameworkConfig, AgentConfigFramework};

let config = FrameworkConfig {
    agent_config_dir: PathBuf::from("../agent-config"),
    template_dir: PathBuf::from("templates"),
    output_dir: PathBuf::from("../agent-config"),
    enable_docker_integration: true,
    enable_validation: true,
    enable_auto_updates: true,
};

let framework = AgentConfigFramework::with_config(config)?;
```

### Container Orchestration Configuration

```yaml
# orchestration-config.yaml
container_name: "bitnet-swarm"
image_name: "bitnet-swarm-intelligence"
base_port: 8080
agent_config_mount: "../agent-config"
model_cache_mount: "./models"

resource_limits:
  cpu_limit: "2.0"
  memory_limit: "4G"
  gpu_enabled: false

intelligence_modes:
  default_mode: "swarm"
  auto_mode_selection: true
  mode_switch_threshold: 0.7
  swarm_agent_limit: 10
  hive_mind_agent_limit: 20
```

## 📊 Validation Rules

The framework enforces these mandatory validation rules:

### ✅ Required Sections
- **Mandatory Orchestrator Routing**: Must appear at top of every config
- **Docker Container Integration**: Complete container deployment information
- **Intelligence Mode Support**: Both swarm and hive mind capabilities defined
- **Agent Intersection Matrix**: Coordination relationships with other agents
- **Agent Hooks Integration**: Lifecycle management integration

### ✅ Quality Standards
- **Orchestrator Reference**: Must reference `agent-config/orchestrator.md`
- **Container Role Definition**: Must specify PRIMARY/SECONDARY/SPECIALIST/SUPPORT
- **Resource Requirements**: Must specify CPU, memory, and GPU requirements
- **API Endpoints**: Must define HTTP API endpoints for container integration
- **MCP Tools**: Must specify Model Context Protocol tools

### ✅ Intelligence Mode Requirements
- **Swarm Use Cases**: Must define specific swarm intelligence use cases
- **Hive Mind Use Cases**: Must define specific hive mind intelligence use cases
- **Coordination Patterns**: Must specify coordination behavior for each mode

## 🛠️ Development

### Building the Framework

```bash
# Build the framework
cargo build --release

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Check lints
cargo clippy
```

### Adding New Templates

1. Create template in `templates/` directory
2. Add variable placeholders using `{{VARIABLE_NAME}}`
3. Update `AgentConfigGenerator` to support new template
4. Add validation rules for new template sections

### Extending Validation

1. Add new validation rules to `OrchestratorRoutingValidator`
2. Define error types for new validation failures
3. Update validation report generation
4. Add test cases for new validation rules

## 📚 API Reference

### Agent Generation

```rust
use agent_config_framework::{AgentConfigGenerator, AgentSpec, ContainerRole, AgentType};

let generator = AgentConfigGenerator::new("templates", "output");
let spec = AgentSpec {
    name: "example_agent".to_string(),
    agent_type: AgentType::Specialist,
    domain: "testing".to_string(),
    container_role: ContainerRole::Secondary,
    specialist_role: "test specialist".to_string(),
    primary_focus: "test automation".to_string(),
    core_expertise: "test framework development".to_string(),
    // ... other required fields
};

let config_path = generator.generate_and_save(&spec)?;
```

### Validation

```rust
use agent_config_framework::OrchestratorRoutingValidator;

let validator = OrchestratorRoutingValidator::new("../agent-config");
let results = validator.validate_all_configs()?;
let report = validator.generate_report(&results);
report.print_summary();
```

### Container Orchestration

```rust
use agent_config_framework::{ContainerOrchestration, OrchestrationConfig};

let config = OrchestrationConfig {
    container_name: "bitnet-swarm".to_string(),
    image_name: "bitnet-swarm-intelligence".to_string(),
    base_port: 8080,
    // ... other configuration
};

let mut orchestration = ContainerOrchestration::new(config);
orchestration.initialize_system().await?;
```

## 🤝 Contributing

1. **Follow Standards**: Use the existing code style and patterns
2. **Add Tests**: Include tests for new functionality
3. **Update Documentation**: Keep README and docs current
4. **Validate Changes**: Ensure all validation rules pass

## 📄 License

This framework is part of the BitNet-Rust project and follows the same licensing terms.

## 🔗 Integration with BitNet-Rust

This framework is specifically designed for the BitNet-Rust project and integrates with:

- **COMPREHENSIVE_TODO.md**: Aligns with project roadmap and priorities
- **ROAD_TO_INFERENCE.md**: Supports inference implementation phases
- **Docker BitNet Swarm**: Native container deployment
- **VS Code Extension**: HTTP API integration for development tools

The framework ensures all agents follow established patterns while supporting the evolution toward production-ready BitNet intelligence systems.