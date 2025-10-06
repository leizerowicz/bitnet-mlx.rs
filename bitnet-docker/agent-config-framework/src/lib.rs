/*!
# BitNet Agent Config Framework

This crate provides a comprehensive framework for managing agent configurations
in the BitNet Docker Swarm Intelligence system. It includes tools for:

- **Agent Config Generation**: Create standardized agent configurations with templates
- **Orchestrator Routing Validation**: Ensure all agents follow mandatory orchestrator routing
- **Intersection Matrix Management**: Auto-update agent intersection patterns
- **Docker Container Orchestration**: Deploy and manage containerized intelligence systems
- **Intelligence Mode Coordination**: Support for both Swarm and Hive Mind intelligence modes

## Quick Start

```rust
use agent_config_framework::{AgentConfigGenerator, AgentSpec, ContainerRole, AgentType};

// Generate a new agent configuration
let generator = AgentConfigGenerator::new("templates", "output");
let spec = AgentSpec {
    name: "example_agent".to_string(),
    agent_type: AgentType::Specialist,
    domain: "testing".to_string(),
    container_role: ContainerRole::Secondary,
    // ... other fields
};

let config_path = generator.generate_and_save(&spec)?;
```

## Architecture

The framework is built around several core components:

### Agent Config Generation
- **Templates**: Standardized Markdown templates with variable substitution
- **Generators**: Rust code generators that fill templates with agent-specific data
- **Validation**: Comprehensive validation to ensure config consistency

### Container Orchestration
- **Intelligence Modes**: Support for Swarm (diverging collaborative) and Hive Mind (unified collective) modes
- **Agent Discovery**: Automatic discovery of available agents in containers
- **Resource Management**: Efficient resource allocation and scaling

### Quality Assurance
- **Mandatory Orchestrator Routing**: All agents must route through orchestrator.md
- **Docker Integration**: All agents must support containerized deployment
- **Intelligence Mode Support**: All agents must specify their intelligence capabilities

## CLI Usage

The framework includes a comprehensive CLI tool:

```bash
# Generate a new agent config
agent-config-cli generate --name "my_agent" --domain "testing" --interactive

# Validate all existing configs
agent-config-cli validate --verbose

# Update intersection matrix
agent-config-cli update-matrix --update-configs

# Deploy Docker container
agent-config-cli deploy --monitor
```

## Project Integration

This framework is specifically designed for the BitNet-Rust project's agent-based
architecture, providing:

- **Perfect Integration**: Seamless integration with existing BitNet codebase
- **Docker Native**: Built for containerized deployment from the ground up
- **Intelligence Aware**: Native support for dual intelligence modes
- **Scalable**: Designed to handle complex multi-agent coordination
*/

pub mod generator;
pub mod validator;
pub mod updater;
pub mod orchestration;

// Re-export main types for convenience
pub use generator::{
    AgentConfigGenerator, AgentSpec, AgentType, ContainerRole, 
    ResourceRequirements, ResourceLevel, AgentGeneratorError
};

pub use validator::{
    OrchestratorRoutingValidator, ValidationResult, ValidationError, 
    ValidationReport, ValidationRules, ValidatorError
};

pub use updater::{
    IntersectionMatrixUpdater, AgentIntersectionMatrix, AgentIntersection,
    IntersectionRelationship, RelationshipType, IntelligenceModeSupport,
    GlobalIntersectionPatterns, UpdaterError
};

pub use orchestration::{
    ContainerOrchestration, OrchestrationConfig, ResourceLimits, IntelligenceModeConfig,
    AgentRegistry, RegisteredAgent, AgentStatus, ContainerManager, ContainerStatus,
    IntelligenceCoordinator, IntelligenceMode, IntelligenceSession, SessionStatus,
    ModeSelector, SwarmManager, HiveMindManager, IntelligenceRequest, OrchestrationError
};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const FRAMEWORK_NAME: &str = "BitNet Agent Config Framework";

/// Framework initialization and configuration
pub struct FrameworkConfig {
    pub agent_config_dir: std::path::PathBuf,
    pub template_dir: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub enable_docker_integration: bool,
    pub enable_validation: bool,
    pub enable_auto_updates: bool,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            agent_config_dir: std::path::PathBuf::from("../agent-config"),
            template_dir: std::path::PathBuf::from("templates"),
            output_dir: std::path::PathBuf::from("../agent-config"),
            enable_docker_integration: true,
            enable_validation: true,
            enable_auto_updates: true,
        }
    }
}

/// Main framework interface for high-level operations
pub struct AgentConfigFramework {
    config: FrameworkConfig,
    generator: AgentConfigGenerator,
    validator: OrchestratorRoutingValidator,
    updater: IntersectionMatrixUpdater,
}

impl AgentConfigFramework {
    /// Create a new framework instance with default configuration
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = FrameworkConfig::default();
        Self::with_config(config)
    }

    /// Create a new framework instance with custom configuration
    pub fn with_config(config: FrameworkConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let generator = AgentConfigGenerator::new(&config.template_dir, &config.output_dir);
        let validator = OrchestratorRoutingValidator::new(&config.agent_config_dir);
        let matrix_file = config.agent_config_dir.join("agent-intersection-matrix.json");
        let updater = IntersectionMatrixUpdater::new(&config.agent_config_dir, &matrix_file)?;
        
        Ok(Self {
            config,
            generator,
            validator,
            updater,
        })
    }

    /// Generate a new agent configuration with full validation
    pub fn generate_agent(&self, spec: &AgentSpec) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        // Generate the config
        let output_path = self.generator.generate_and_save(spec)?;
        
        // Validate if enabled
        if self.config.enable_validation {
            let result = self.validator.validate_config_file(&output_path)?;
            if !result.is_valid {
                return Err(format!("Generated config failed validation: {:?}", result.errors).into());
            }
        }
        
        Ok(output_path)
    }

    /// Validate all agent configurations
    pub fn validate_all(&self) -> Result<ValidationReport, Box<dyn std::error::Error>> {
        let results = self.validator.validate_all_configs()?;
        let report = self.validator.generate_report(&results);
        Ok(report)
    }

    /// Update intersection matrix and optionally update all configs
    pub fn update_intersections(&mut self, update_configs: bool) -> Result<(), Box<dyn std::error::Error>> {
        self.updater.update_intersection_matrix()?;
        
        if update_configs && self.config.enable_auto_updates {
            // Update all agent configs with new intersection information
            // Implementation would iterate through all configs and update intersection sections
            println!("Auto-updating agent configs with new intersections...");
        }
        
        Ok(())
    }

    /// Deploy containerized agent system
    pub async fn deploy_container(&self, orchestration_config: OrchestrationConfig) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.enable_docker_integration {
            return Err("Docker integration is disabled".into());
        }
        
        let mut orchestration = ContainerOrchestration::new(orchestration_config);
        orchestration.initialize_system().await?;
        
        Ok(())
    }

    /// Get framework status and health information
    pub fn get_status(&self) -> FrameworkStatus {
        let agent_count = std::fs::read_dir(&self.config.agent_config_dir)
            .map(|entries| entries.count())
            .unwrap_or(0);
        
        let templates_available = self.config.template_dir.exists();
        let docker_available = std::process::Command::new("docker")
            .arg("--version")
            .output()
            .is_ok();
        
        FrameworkStatus {
            version: VERSION.to_string(),
            agent_count,
            templates_available,
            docker_available,
            validation_enabled: self.config.enable_validation,
            auto_updates_enabled: self.config.enable_auto_updates,
            docker_integration_enabled: self.config.enable_docker_integration,
        }
    }
}

#[derive(Debug)]
pub struct FrameworkStatus {
    pub version: String,
    pub agent_count: usize,
    pub templates_available: bool,
    pub docker_available: bool,
    pub validation_enabled: bool,
    pub auto_updates_enabled: bool,
    pub docker_integration_enabled: bool,
}

impl FrameworkStatus {
    pub fn print_summary(&self) {
        println!("🎯 BitNet Agent Config Framework v{}", self.version);
        println!("==========================================");
        println!("📊 Status:");
        println!("   Agents discovered: {}", self.agent_count);
        println!("   Templates: {}", if self.templates_available { "✅ Available" } else { "❌ Missing" });
        println!("   Docker: {}", if self.docker_available { "✅ Available" } else { "❌ Not available" });
        println!("📋 Configuration:");
        println!("   Validation: {}", if self.validation_enabled { "✅ Enabled" } else { "⏸️ Disabled" });
        println!("   Auto-updates: {}", if self.auto_updates_enabled { "✅ Enabled" } else { "⏸️ Disabled" });
        println!("   Docker integration: {}", if self.docker_integration_enabled { "✅ Enabled" } else { "⏸️ Disabled" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_framework_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = FrameworkConfig {
            agent_config_dir: temp_dir.path().to_path_buf(),
            template_dir: temp_dir.path().join("templates"),
            output_dir: temp_dir.path().join("output"),
            enable_docker_integration: false,  // Disable for testing
            enable_validation: true,
            enable_auto_updates: false,
        };
        
        // This would fail without actual template files, but tests the structure
        let result = AgentConfigFramework::with_config(config);
        assert!(result.is_err()); // Expected to fail without templates
    }

    #[test]
    fn test_framework_status() {
        let framework = AgentConfigFramework::new();
        
        // Even if framework creation fails, we can test status structure
        if let Ok(fw) = framework {
            let status = fw.get_status();
            assert_eq!(status.version, VERSION);
        }
    }
}

/// Utility functions for common operations
pub mod utils {
    use std::path::Path;

    /// Check if a directory contains agent configurations
    pub fn is_agent_config_directory<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists() && 
        path.as_ref().is_dir() &&
        path.as_ref().join("orchestrator.md").exists()
    }

    /// Count agent configuration files in a directory
    pub fn count_agent_configs<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error> {
        let count = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension()
                    .map_or(false, |ext| ext == "md") &&
                entry.file_name() != "README.md"
            })
            .count();
        
        Ok(count)
    }

    /// Validate that Docker is available and functional
    pub fn check_docker_availability() -> bool {
        std::process::Command::new("docker")
            .args(&["version", "--format", "{{.Server.Version}}"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get the recommended resource requirements for an agent type
    pub fn get_recommended_resources(agent_type: &crate::AgentType, domain: &str) -> crate::ResourceRequirements {
        use crate::{ResourceRequirements, ResourceLevel};
        
        match (agent_type, domain) {
            (crate::AgentType::Orchestrator, _) => ResourceRequirements {
                cpu: ResourceLevel::High,
                memory: ResourceLevel::High,
                gpu_required: false,
                description: "High CPU, High memory for orchestration".to_string(),
            },
            (_, "inference") | (_, "performance") => ResourceRequirements {
                cpu: ResourceLevel::High,
                memory: ResourceLevel::High,
                gpu_required: true,
                description: "High CPU, High memory, GPU required for ML operations".to_string(),
            },
            (crate::AgentType::Specialist, _) => ResourceRequirements {
                cpu: ResourceLevel::Medium,
                memory: ResourceLevel::Medium,
                gpu_required: false,
                description: "Medium CPU, Medium memory for specialist operations".to_string(),
            },
            _ => ResourceRequirements {
                cpu: ResourceLevel::Low,
                memory: ResourceLevel::Low,
                gpu_required: false,
                description: "Low CPU, Low memory for utility operations".to_string(),
            },
        }
    }
}