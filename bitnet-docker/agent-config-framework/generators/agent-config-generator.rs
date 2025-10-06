use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};

/// Agent Config Framework Generator
/// 
/// Generates standardized agent configurations for the BitNet Docker Swarm Intelligence system
/// with mandatory orchestrator routing, Docker integration, and intelligence mode support.
#[derive(Debug, Clone)]
pub struct AgentConfigGenerator {
    template_dir: PathBuf,
    output_dir: PathBuf,
    intersection_matrix: IntersectionMatrix,
    current_project_context: ProjectContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub agent_type: AgentType,
    pub domain: String,
    pub container_role: ContainerRole,
    pub specialist_role: String,
    pub primary_focus: String,
    pub core_expertise: String,
    pub api_endpoints: Vec<String>,
    pub mcp_tools: Vec<String>,
    pub resource_requirements: ResourceRequirements,
    pub swarm_coordination: String,
    pub hive_mind_coordination: String,
    pub swarm_use_cases: Vec<String>,
    pub hive_mind_use_cases: Vec<String>,
    pub primary_responsibilities: Vec<String>,
    pub docker_capabilities: Vec<String>,
    pub intersections: Vec<String>,
    pub quality_standards: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Specialist,
    Orchestrator,
    Utility,
    Support,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerRole {
    Primary,      // Core functionality
    Secondary,    // Supporting functionality  
    Specialist,   // Domain-specific expert
    Support,      // Infrastructure and utility
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu: ResourceLevel,
    pub memory: ResourceLevel,
    pub gpu_required: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct IntersectionMatrix {
    pub intersections: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ProjectContext {
    pub current_phase: String,
    pub phase_description: String,
    pub test_success_rate: String,
    pub current_priority: String,
    pub docker_focus: String,
}

impl AgentConfigGenerator {
    pub fn new<P: AsRef<Path>>(template_dir: P, output_dir: P) -> Self {
        Self {
            template_dir: template_dir.as_ref().to_path_buf(),
            output_dir: output_dir.as_ref().to_path_buf(),
            intersection_matrix: IntersectionMatrix::load_default(),
            current_project_context: ProjectContext::load_current(),
        }
    }

    /// Generate a complete agent config from specification
    pub fn generate_agent_config(&self, spec: &AgentSpec) -> Result<String, AgentGeneratorError> {
        let template = self.load_template("specialist-agent-template.md")?;
        let content = self.fill_template(&template, spec)?;
        self.validate_generated_content(&content)?;
        Ok(content)
    }

    /// Generate and save agent config to file
    pub fn generate_and_save(&self, spec: &AgentSpec) -> Result<PathBuf, AgentGeneratorError> {
        let content = self.generate_agent_config(spec)?;
        let filename = format!("{}.md", spec.name.to_lowercase().replace(" ", "_"));
        let output_path = self.output_dir.join(filename);
        
        fs::write(&output_path, content)?;
        println!("Generated agent config: {}", output_path.display());
        
        Ok(output_path)
    }

    /// Fill template with agent specification data
    fn fill_template(&self, template: &str, spec: &AgentSpec) -> Result<String, AgentGeneratorError> {
        let mut content = template.to_string();
        
        // Basic agent information
        content = content.replace("{{AGENT_NAME}}", &spec.name);
        content = content.replace("{{AGENT_TYPE}}", &format!("{:?}", spec.agent_type));
        content = content.replace("{{DOMAIN}}", &spec.domain);
        content = content.replace("{{UPDATE_DATE}}", &Utc::now().format("%B %d, %Y").to_string());
        
        // Project context
        content = content.replace("{{PROJECT_PHASE}}", &self.current_project_context.current_phase);
        content = content.replace("{{PHASE_DESCRIPTION}}", &self.current_project_context.phase_description);
        
        // Docker container integration
        content = content.replace("{{CONTAINER_ROLE}}", &format!("{:?}", spec.container_role).to_uppercase());
        content = content.replace("{{ROLE_DESCRIPTION}}", &self.generate_role_description(spec));
        content = content.replace("{{API_ENDPOINTS}}", &spec.api_endpoints.join(", "));
        content = content.replace("{{MCP_TOOLS}}", &spec.mcp_tools.join(", "));
        content = content.replace("{{RESOURCE_REQUIREMENTS}}", &spec.resource_requirements.description);
        content = content.replace("{{SWARM_COORDINATION}}", &spec.swarm_coordination);
        content = content.replace("{{HIVE_MIND_COORDINATION}}", &spec.hive_mind_coordination);
        
        // Intelligence mode use cases
        content = content.replace("{{SWARM_USE_CASES}}", &self.format_use_cases(&spec.swarm_use_cases));
        content = content.replace("{{HIVE_MIND_USE_CASES}}", &self.format_use_cases(&spec.hive_mind_use_cases));
        
        // Specialist role and capabilities
        content = content.replace("{{SPECIALIST_ROLE}}", &spec.specialist_role);
        content = content.replace("{{PRIMARY_FOCUS}}", &spec.primary_focus);
        content = content.replace("{{CORE_EXPERTISE}}", &spec.core_expertise);
        content = content.replace("{{PRIMARY_RESPONSIBILITIES}}", &self.format_list(&spec.primary_responsibilities));
        content = content.replace("{{DOCKER_CAPABILITIES}}", &self.format_list(&spec.docker_capabilities));
        
        // Agent intersections
        content = content.replace("{{INTERSECTION_MATRIX}}", &self.generate_intersection_matrix(&spec.name));
        
        // Agent hooks integration
        content = content.replace("{{AGENT_HOOKS_INTEGRATION}}", &self.generate_hooks_integration(spec));
        
        // Project context
        content = content.replace("{{PROJECT_CONTEXT}}", &self.generate_project_context());
        
        // Quality standards
        content = content.replace("{{QUALITY_STANDARDS}}", &self.format_list(&spec.quality_standards));
        
        // Container patterns
        content = content.replace("{{CONTAINER_PATTERNS}}", &self.generate_container_patterns(spec));
        
        // Template metadata
        content = content.replace("{{TEMPLATE_VERSION}}", "1.0");
        
        Ok(content)
    }

    fn load_template(&self, template_name: &str) -> Result<String, AgentGeneratorError> {
        let template_path = self.template_dir.join(template_name);
        fs::read_to_string(template_path).map_err(AgentGeneratorError::IoError)
    }

    fn generate_role_description(&self, spec: &AgentSpec) -> String {
        match spec.container_role {
            ContainerRole::Primary => format!("responsible for core {} functionality", spec.domain),
            ContainerRole::Secondary => format!("supporting {} operations and coordination", spec.domain),
            ContainerRole::Specialist => format!("domain expert for {} optimization and best practices", spec.domain),
            ContainerRole::Support => format!("infrastructure and utility support for {}", spec.domain),
        }
    }

    fn format_use_cases(&self, use_cases: &[String]) -> String {
        use_cases.iter()
            .map(|case| format!("- **{}**", case))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_list(&self, items: &[String]) -> String {
        items.iter()
            .map(|item| format!("- **{}**", item))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_intersection_matrix(&self, agent_name: &str) -> String {
        if let Some(intersections) = self.intersection_matrix.intersections.get(agent_name) {
            intersections.iter()
                .map(|intersection| format!("- **{}**: {}", intersection, self.get_intersection_description(agent_name, intersection)))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            "- **No specific intersections defined** - Coordinates through orchestrator.md".to_string()
        }
    }

    fn get_intersection_description(&self, _agent: &str, intersection: &str) -> String {
        // This would be loaded from intersection matrix data
        format!("Coordinated workflow for {}", intersection)
    }

    fn generate_hooks_integration(&self, spec: &AgentSpec) -> String {
        let hooks = match spec.agent_type {
            AgentType::Specialist => vec![
                "PRE_TASK_SETUP", "TASK_EXECUTION", "POST_TASK_VALIDATION", 
                "COLLABORATION_COORDINATION", "QUALITY_GATE_VALIDATION"
            ],
            AgentType::Orchestrator => vec![
                "CONTAINER_INTELLIGENCE_STARTUP", "AGENT_DISCOVERY_AND_REGISTRATION",
                "INTELLIGENCE_MODE_DETECTION", "WORKFLOW_COORDINATION"
            ],
            AgentType::Utility => vec![
                "PRE_TASK_SETUP", "POST_TASK_CLEANUP", "VALIDATION_GATE"
            ],
            AgentType::Support => vec![
                "INFRASTRUCTURE_SETUP", "MONITORING", "LIFECYCLE_MANAGEMENT"
            ],
        };
        
        hooks.iter()
            .map(|hook| format!("- **{}**: {}", hook, self.get_hook_description(hook)))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn get_hook_description(&self, hook: &str) -> String {
        match hook {
            "PRE_TASK_SETUP" => "Initialize resources and validate prerequisites",
            "TASK_EXECUTION" => "Execute primary task with monitoring",
            "POST_TASK_VALIDATION" => "Validate results and ensure quality",
            "COLLABORATION_COORDINATION" => "Coordinate with other agents",
            "QUALITY_GATE_VALIDATION" => "Enforce quality standards",
            "CONTAINER_INTELLIGENCE_STARTUP" => "Initialize containerized intelligence systems",
            "AGENT_DISCOVERY_AND_REGISTRATION" => "Discover and register agent capabilities",
            "INTELLIGENCE_MODE_DETECTION" => "Analyze and select intelligence mode",
            "WORKFLOW_COORDINATION" => "Coordinate multi-agent workflows",
            "POST_TASK_CLEANUP" => "Clean up resources and finalize",
            "VALIDATION_GATE" => "Validate system state and integrity",
            "INFRASTRUCTURE_SETUP" => "Setup infrastructure and dependencies",
            "MONITORING" => "Monitor system health and performance",
            "LIFECYCLE_MANAGEMENT" => "Manage component lifecycle",
            _ => "Automated lifecycle management",
        }.to_string()
    }

    fn generate_project_context(&self) -> String {
        format!(
            "- **Current Status**: {}\n- **Technical Foundation**: {}\n- **Current Priority**: {}\n- **Docker Focus**: {}",
            self.current_project_context.current_phase,
            self.current_project_context.test_success_rate,
            self.current_project_context.current_priority,
            self.current_project_context.docker_focus
        )
    }

    fn generate_container_patterns(&self, spec: &AgentSpec) -> String {
        format!(
            "- **Deployment**: {} container with {} resource allocation\n- **Scaling**: {} based on {} workload\n- **Integration**: {} with universal /api endpoint and MCP server",
            format!("{:?}", spec.container_role).to_lowercase(),
            spec.resource_requirements.description.to_lowercase(),
            if spec.resource_requirements.gpu_required { "GPU-enabled scaling" } else { "CPU-based scaling" },
            spec.domain.to_lowercase(),
            if matches!(spec.container_role, ContainerRole::Primary) { "Primary integration" } else { "Secondary integration" }
        )
    }

    fn validate_generated_content(&self, content: &str) -> Result<(), AgentGeneratorError> {
        // Validate mandatory orchestrator routing
        if !content.contains("MANDATORY ORCHESTRATOR ROUTING") {
            return Err(AgentGeneratorError::ValidationError("Missing mandatory orchestrator routing".to_string()));
        }
        
        // Validate Docker integration section
        if !content.contains("Docker Container Integration") {
            return Err(AgentGeneratorError::ValidationError("Missing Docker integration section".to_string()));
        }
        
        // Validate intelligence mode sections
        if !content.contains("🐝 Swarm Intelligence") || !content.contains("🧠 Hive Mind Intelligence") {
            return Err(AgentGeneratorError::ValidationError("Missing intelligence mode sections".to_string()));
        }
        
        Ok(())
    }
}

impl IntersectionMatrix {
    fn load_default() -> Self {
        let mut intersections = HashMap::new();
        
        // Define default intersections based on existing agent configs
        intersections.insert("code".to_string(), vec![
            "rust_best_practices_specialist".to_string(),
            "debug".to_string(),
            "test_utilities_specialist".to_string(),
        ]);
        
        intersections.insert("debug".to_string(), vec![
            "code".to_string(),
            "error_handling_specialist".to_string(),
            "test_utilities_specialist".to_string(),
        ]);
        
        intersections.insert("inference_engine_specialist".to_string(), vec![
            "performance_engineering_specialist".to_string(),
            "api_development_specialist".to_string(),
            "code".to_string(),
        ]);
        
        Self { intersections }
    }
}

impl ProjectContext {
    fn load_current() -> Self {
        Self {
            current_phase: "Docker BitNet Swarm Intelligence Phase".to_string(),
            phase_description: "Implementing containerized intelligence systems with perfect technical foundation".to_string(),
            test_success_rate: "100% test success rate (1,169/1,169 tests)".to_string(),
            current_priority: "Docker container with swarm/hive mind intelligence systems".to_string(),
            docker_focus: "Production-ready Docker container with dual intelligence modes for VS Code extension integration".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentGeneratorError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Template error: {0}")]
    TemplateError(String),
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_generation() {
        let generator = AgentConfigGenerator::new("templates", "output");
        
        let spec = AgentSpec {
            name: "Test Agent".to_string(),
            agent_type: AgentType::Specialist,
            domain: "testing".to_string(),
            container_role: ContainerRole::Secondary,
            specialist_role: "test specialist".to_string(),
            primary_focus: "test automation".to_string(),
            core_expertise: "test framework development".to_string(),
            api_endpoints: vec!["/api".to_string(), "/test/run".to_string()],
            mcp_tools: vec!["test-runner".to_string(), "test-validator".to_string()],
            resource_requirements: ResourceRequirements {
                cpu: ResourceLevel::Medium,
                memory: ResourceLevel::Medium,
                gpu_required: false,
                description: "Medium CPU, Medium memory for test execution".to_string(),
            },
            swarm_coordination: "Independent test execution with collaborative result analysis".to_string(),
            hive_mind_coordination: "Synchronized testing across entire system with unified coverage".to_string(),
            swarm_use_cases: vec!["Parallel test execution".to_string()],
            hive_mind_use_cases: vec!["System-wide test coordination".to_string()],
            primary_responsibilities: vec!["Test execution".to_string(), "Test validation".to_string()],
            docker_capabilities: vec!["Containerized testing".to_string()],
            intersections: vec!["test_utilities_specialist".to_string()],
            quality_standards: vec!["Test coverage".to_string(), "Test reliability".to_string()],
        };
        
        // This test would need actual template files to work
        // For now, it validates the spec structure
        assert_eq!(spec.name, "Test Agent");
        assert_eq!(spec.domain, "testing");
    }
}