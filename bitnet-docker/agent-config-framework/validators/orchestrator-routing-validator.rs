use std::fs;
use std::path::{Path, PathBuf};
use regex::Regex;

/// Orchestrator Routing Validator
/// 
/// Validates that all agent configurations include mandatory orchestrator routing
/// and follow the established patterns for Docker BitNet Swarm Intelligence integration.
#[derive(Debug)]
pub struct OrchestratorRoutingValidator {
    agent_config_dir: PathBuf,
    validation_rules: ValidationRules,
}

#[derive(Debug)]
pub struct ValidationRules {
    pub require_orchestrator_routing: bool,
    pub require_docker_integration: bool,
    pub require_intelligence_modes: bool,
    pub require_agent_hooks: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationResult {
    pub file_path: PathBuf,
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ValidationError {
    MissingOrchestratorRouting,
    IncorrectOrchestratorReference,
    MissingDockerIntegration,
    MissingIntelligenceModes,
    MissingAgentHooks,
    InvalidMarkdownStructure,
    MissingRequiredSections(Vec<String>),
}

impl OrchestratorRoutingValidator {
    pub fn new<P: AsRef<Path>>(agent_config_dir: P) -> Self {
        Self {
            agent_config_dir: agent_config_dir.as_ref().to_path_buf(),
            validation_rules: ValidationRules::default(),
        }
    }

    /// Validate all agent configurations in the directory
    pub fn validate_all_configs(&self) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        
        for entry in fs::read_dir(&self.agent_config_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |ext| ext == "md") {
                let result = self.validate_config_file(&path)?;
                results.push(result);
            }
        }
        
        Ok(results)
    }

    /// Validate a single agent configuration file
    pub fn validate_config_file<P: AsRef<Path>>(&self, file_path: P) -> Result<ValidationResult, ValidatorError> {
        let file_path = file_path.as_ref().to_path_buf();
        let content = fs::read_to_string(&file_path)?;
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Validate mandatory orchestrator routing
        if self.validation_rules.require_orchestrator_routing {
            if let Some(error) = self.validate_orchestrator_routing(&content) {
                errors.push(error);
            }
        }
        
        // Validate Docker integration
        if self.validation_rules.require_docker_integration {
            if let Some(error) = self.validate_docker_integration(&content) {
                errors.push(error);
            }
        }
        
        // Validate intelligence modes
        if self.validation_rules.require_intelligence_modes {
            if let Some(error) = self.validate_intelligence_modes(&content) {
                errors.push(error);
            }
        }
        
        // Validate agent hooks integration
        if self.validation_rules.require_agent_hooks {
            if let Some(error) = self.validate_agent_hooks(&content) {
                errors.push(error);
            }
        }
        
        // Check for required sections
        let missing_sections = self.check_required_sections(&content);
        if !missing_sections.is_empty() {
            errors.push(ValidationError::MissingRequiredSections(missing_sections));
        }
        
        // Generate warnings for improvements
        warnings.extend(self.generate_warnings(&content));
        
        Ok(ValidationResult {
            file_path,
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    fn validate_orchestrator_routing(&self, content: &str) -> Option<ValidationError> {
        // Check for mandatory orchestrator routing section
        let orchestrator_warning_pattern = Regex::new(
            r"⚠️\s*MANDATORY ORCHESTRATOR ROUTING.*?agent-config/orchestrator\.md.*?FIRST"
        ).unwrap();
        
        if !orchestrator_warning_pattern.is_match(content) {
            return Some(ValidationError::MissingOrchestratorRouting);
        }
        
        // Check for correct orchestrator reference
        if !content.contains("agent-config/orchestrator.md") {
            return Some(ValidationError::IncorrectOrchestratorReference);
        }
        
        // Verify the routing section mentions all required coordination aspects
        let required_aspects = vec![
            "task routing",
            "workflow coordination",
            "multi-agent needs",
            "current project context",
            "agent hooks integration"
        ];
        
        for aspect in required_aspects {
            if !content.to_lowercase().contains(aspect) {
                return Some(ValidationError::MissingOrchestratorRouting);
            }
        }
        
        None
    }

    fn validate_docker_integration(&self, content: &str) -> Option<ValidationError> {
        let required_docker_sections = vec![
            "Docker Container Integration",
            "Container Role",
            "API Endpoints",
            "MCP Tools",
            "Resource Requirements",
            "Coordination Patterns"
        ];
        
        for section in required_docker_sections {
            if !content.contains(section) {
                return Some(ValidationError::MissingDockerIntegration);
            }
        }
        
        None
    }

    fn validate_intelligence_modes(&self, content: &str) -> Option<ValidationError> {
        let required_intelligence_sections = vec![
            "🐝 Swarm Intelligence",
            "🧠 Hive Mind Intelligence",
            "Swarm Mode",
            "Hive Mind Mode"
        ];
        
        for section in required_intelligence_sections {
            if !content.contains(section) {
                return Some(ValidationError::MissingIntelligenceModes);
            }
        }
        
        None
    }

    fn validate_agent_hooks(&self, content: &str) -> Option<ValidationError> {
        // Check for agent hooks integration section
        if !content.contains("Agent Hooks Integration") {
            return Some(ValidationError::MissingAgentHooks);
        }
        
        None
    }

    fn check_required_sections(&self, content: &str) -> Vec<String> {
        let required_sections = vec![
            "Specialist Role & Niche",
            "Core Specialist Niche",
            "Agent Intersection Matrix",
            "Project Context Awareness",
            "Quality Standards & Gates"
        ];
        
        required_sections.into_iter()
            .filter(|section| !content.contains(section))
            .map(|s| s.to_string())
            .collect()
    }

    fn generate_warnings(&self, content: &str) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // Check for outdated project phase
        if content.contains("Inference Ready Phase") && !content.contains("Docker BitNet Swarm Intelligence Phase") {
            warnings.push("Agent config may have outdated project phase information".to_string());
        }
        
        // Check for missing intersection information
        if content.contains("No specific intersections defined") {
            warnings.push("Consider defining specific agent intersections for better coordination".to_string());
        }
        
        // Check for generic use cases
        if content.contains("TODO") || content.contains("{{") {
            warnings.push("Template placeholders detected - config may be incomplete".to_string());
        }
        
        warnings
    }

    /// Generate a report of all validation results
    pub fn generate_report(&self, results: &[ValidationResult]) -> ValidationReport {
        let total_configs = results.len();
        let valid_configs = results.iter().filter(|r| r.is_valid).count();
        let invalid_configs = total_configs - valid_configs;
        
        let all_errors: Vec<_> = results.iter()
            .flat_map(|r| &r.errors)
            .collect();
        
        let all_warnings: Vec<_> = results.iter()
            .flat_map(|r| &r.warnings)
            .collect();
        
        ValidationReport {
            total_configs,
            valid_configs,
            invalid_configs,
            total_errors: all_errors.len(),
            total_warnings: all_warnings.len(),
            error_summary: self.summarize_errors(&all_errors),
            results: results.to_vec(),
        }
    }

    fn summarize_errors(&self, errors: &[&ValidationError]) -> std::collections::HashMap<String, usize> {
        let mut summary = std::collections::HashMap::new();
        
        for error in errors {
            let error_type = match error {
                ValidationError::MissingOrchestratorRouting => "Missing Orchestrator Routing",
                ValidationError::IncorrectOrchestratorReference => "Incorrect Orchestrator Reference",
                ValidationError::MissingDockerIntegration => "Missing Docker Integration",
                ValidationError::MissingIntelligenceModes => "Missing Intelligence Modes",
                ValidationError::MissingAgentHooks => "Missing Agent Hooks",
                ValidationError::InvalidMarkdownStructure => "Invalid Markdown Structure",
                ValidationError::MissingRequiredSections(_) => "Missing Required Sections",
            };
            
            *summary.entry(error_type.to_string()).or_insert(0) += 1;
        }
        
        summary
    }
}

impl ValidationRules {
    fn default() -> Self {
        Self {
            require_orchestrator_routing: true,
            require_docker_integration: true,
            require_intelligence_modes: true,
            require_agent_hooks: true,
        }
    }
}

#[derive(Debug)]
pub struct ValidationReport {
    pub total_configs: usize,
    pub valid_configs: usize,
    pub invalid_configs: usize,
    pub total_errors: usize,
    pub total_warnings: usize,
    pub error_summary: std::collections::HashMap<String, usize>,
    pub results: Vec<ValidationResult>,
}

impl ValidationReport {
    pub fn print_summary(&self) {
        println!("=== Agent Config Validation Report ===");
        println!("Total configs: {}", self.total_configs);
        println!("Valid configs: {} ({:.1}%)", 
                 self.valid_configs, 
                 (self.valid_configs as f64 / self.total_configs as f64) * 100.0);
        println!("Invalid configs: {}", self.invalid_configs);
        println!("Total errors: {}", self.total_errors);
        println!("Total warnings: {}", self.total_warnings);
        
        if !self.error_summary.is_empty() {
            println!("\nError Summary:");
            for (error_type, count) in &self.error_summary {
                println!("  - {}: {}", error_type, count);
            }
        }
        
        if self.invalid_configs > 0 {
            println!("\nInvalid Configs:");
            for result in &self.results {
                if !result.is_valid {
                    println!("  - {}: {} errors", result.file_path.display(), result.errors.len());
                    for error in &result.errors {
                        println!("    • {:?}", error);
                    }
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ValidatorError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_routing_validation() {
        let validator = OrchestratorRoutingValidator::new("agent-config");
        
        let valid_content = r#"
        > **⚠️ MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config, 
        > **ALWAYS consult `agent-config/orchestrator.md` FIRST** for task routing, workflow coordination, 
        > multi-agent needs, current project context, and agent hooks integration.
        "#;
        
        assert!(validator.validate_orchestrator_routing(valid_content).is_none());
        
        let invalid_content = "Some content without orchestrator routing";
        assert!(validator.validate_orchestrator_routing(invalid_content).is_some());
    }

    #[test]
    fn test_docker_integration_validation() {
        let validator = OrchestratorRoutingValidator::new("agent-config");
        
        let valid_content = r#"
        ## Docker Container Integration
        - **Container Role**: PRIMARY agent
        - **API Endpoints**: /api
        - **MCP Tools**: test-tools
        - **Resource Requirements**: High CPU
        - **Coordination Patterns**: Swarm Mode
        "#;
        
        assert!(validator.validate_docker_integration(valid_content).is_none());
        
        let invalid_content = "Content without Docker integration";
        assert!(validator.validate_docker_integration(invalid_content).is_some());
    }
}