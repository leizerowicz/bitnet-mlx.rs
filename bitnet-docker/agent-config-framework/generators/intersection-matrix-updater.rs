use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use regex::Regex;

/// Intersection Matrix Updater
/// 
/// Automatically updates agent intersection patterns when new agents are added
/// or existing agents are modified in the BitNet Docker Swarm Intelligence system.
#[derive(Debug)]
pub struct IntersectionMatrixUpdater {
    agent_config_dir: PathBuf,
    matrix_file_path: PathBuf,
    current_matrix: AgentIntersectionMatrix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIntersectionMatrix {
    pub version: String,
    pub last_updated: String,
    pub intersections: HashMap<String, AgentIntersection>,
    pub global_patterns: GlobalIntersectionPatterns,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIntersection {
    pub agent_name: String,
    pub agent_type: String,
    pub domain: String,
    pub intersects_with: Vec<IntersectionRelationship>,
    pub coordination_patterns: Vec<String>,
    pub intelligence_modes: IntelligenceModeSupport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionRelationship {
    pub agent: String,
    pub relationship_type: RelationshipType,
    pub coordination_method: String,
    pub shared_responsibilities: Vec<String>,
    pub handoff_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    Primary,      // Direct collaboration
    Secondary,    // Supporting role
    Coordination, // Workflow coordination
    Validation,   // Quality gate validation
    Escalation,   // Error escalation path
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceModeSupport {
    pub swarm_capable: bool,
    pub hive_mind_capable: bool,
    pub swarm_patterns: Vec<String>,
    pub hive_mind_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalIntersectionPatterns {
    pub orchestrator_routing: Vec<String>,
    pub common_quality_gates: Vec<String>,
    pub docker_integration_patterns: Vec<String>,
    pub intelligence_mode_patterns: Vec<String>,
}

impl IntersectionMatrixUpdater {
    pub fn new<P: AsRef<Path>>(agent_config_dir: P, matrix_file_path: P) -> Result<Self, UpdaterError> {
        let matrix_file_path = matrix_file_path.as_ref().to_path_buf();
        let current_matrix = if matrix_file_path.exists() {
            Self::load_matrix(&matrix_file_path)?
        } else {
            Self::create_default_matrix()
        };
        
        Ok(Self {
            agent_config_dir: agent_config_dir.as_ref().to_path_buf(),
            matrix_file_path,
            current_matrix,
        })
    }

    /// Discover all agent configs and update intersection matrix
    pub fn update_intersection_matrix(&mut self) -> Result<(), UpdaterError> {
        let discovered_agents = self.discover_agents()?;
        self.update_intersections(&discovered_agents)?;
        self.save_matrix()?;
        
        println!("Updated intersection matrix with {} agents", discovered_agents.len());
        Ok(())
    }

    /// Discover all agent configurations in the directory
    fn discover_agents(&self) -> Result<Vec<AgentMetadata>, UpdaterError> {
        let mut agents = Vec::new();
        
        for entry in fs::read_dir(&self.agent_config_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |ext| ext == "md") 
                && path.file_name().map_or(false, |name| name != "README.md") {
                
                let metadata = self.extract_agent_metadata(&path)?;
                agents.push(metadata);
            }
        }
        
        Ok(agents)
    }

    /// Extract agent metadata from config file
    fn extract_agent_metadata(&self, file_path: &Path) -> Result<AgentMetadata, UpdaterError> {
        let content = fs::read_to_string(file_path)?;
        let file_name = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Extract agent type and domain from content
        let agent_type = self.extract_agent_type(&content);
        let domain = self.extract_domain(&content);
        let container_role = self.extract_container_role(&content);
        let intersections = self.extract_existing_intersections(&content);
        let intelligence_support = self.extract_intelligence_mode_support(&content);
        
        Ok(AgentMetadata {
            name: file_name,
            agent_type,
            domain,
            container_role,
            file_path: file_path.to_path_buf(),
            existing_intersections: intersections,
            intelligence_support,
        })
    }

    fn extract_agent_type(&self, content: &str) -> String {
        // Look for specialist type indicators
        if content.contains("Orchestrator") || content.contains("orchestrator") {
            "Orchestrator".to_string()
        } else if content.contains("Specialist") {
            "Specialist".to_string()
        } else if content.contains("Utility") {
            "Utility".to_string()
        } else if content.contains("Support") {
            "Support".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn extract_domain(&self, content: &str) -> String {
        // Extract domain from various indicators
        let domain_patterns = vec![
            (r"Domain:\s*(\w+)", 1),
            (r"# .*?(\w+)\s+Specialist", 1),
            (r"focused on (\w+)", 1),
        ];
        
        for (pattern, group) in domain_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(captures) = re.captures(content) {
                    if let Some(domain) = captures.get(group) {
                        return domain.as_str().to_lowercase();
                    }
                }
            }
        }
        
        "general".to_string()
    }

    fn extract_container_role(&self, content: &str) -> String {
        if let Ok(re) = Regex::new(r"Container Role.*?(\w+)\s+agent") {
            if let Some(captures) = re.captures(content) {
                if let Some(role) = captures.get(1) {
                    return role.as_str().to_uppercase();
                }
            }
        }
        
        "SECONDARY".to_string()
    }

    fn extract_existing_intersections(&self, content: &str) -> Vec<String> {
        let mut intersections = Vec::new();
        
        // Look for intersection patterns
        if let Ok(re) = Regex::new(r"intersects with.*?(\w+\.md)") {
            for captures in re.captures_iter(content) {
                if let Some(intersection) = captures.get(1) {
                    let agent_name = intersection.as_str().replace(".md", "");
                    intersections.push(agent_name);
                }
            }
        }
        
        // Look for explicit intersection lists
        if let Ok(re) = Regex::new(r"\*\*([^*]+)\*\*:\s*([^*\n]+)") {
            for captures in re.captures_iter(content) {
                if let Some(agent) = captures.get(1) {
                    intersections.push(agent.as_str().to_string());
                }
            }
        }
        
        intersections
    }

    fn extract_intelligence_mode_support(&self, content: &str) -> IntelligenceModeSupport {
        let swarm_capable = content.contains("🐝 Swarm Intelligence") || content.contains("Swarm Mode");
        let hive_mind_capable = content.contains("🧠 Hive Mind Intelligence") || content.contains("Hive Mind Mode");
        
        let swarm_patterns = if swarm_capable {
            self.extract_patterns(content, "Swarm Mode")
        } else {
            Vec::new()
        };
        
        let hive_mind_patterns = if hive_mind_capable {
            self.extract_patterns(content, "Hive Mind Mode")
        } else {
            Vec::new()
        };
        
        IntelligenceModeSupport {
            swarm_capable,
            hive_mind_capable,
            swarm_patterns,
            hive_mind_patterns,
        }
    }

    fn extract_patterns(&self, content: &str, mode: &str) -> Vec<String> {
        let mut patterns = Vec::new();
        
        if content.contains("Independent") || content.contains("independent") {
            patterns.push("Independent operation".to_string());
        }
        
        if content.contains("Collaborative") || content.contains("collaborative") {
            patterns.push("Collaborative coordination".to_string());
        }
        
        if content.contains("Unified") || content.contains("unified") {
            patterns.push("Unified execution".to_string());
        }
        
        if content.contains("Synchronized") || content.contains("synchronized") {
            patterns.push("Synchronized processing".to_string());
        }
        
        patterns
    }

    /// Update intersections based on discovered agents
    fn update_intersections(&mut self, agents: &[AgentMetadata]) -> Result<(), UpdaterError> {
        for agent in agents {
            let intersection = self.compute_agent_intersection(agent, agents)?;
            self.current_matrix.intersections.insert(agent.name.clone(), intersection);
        }
        
        // Update global patterns
        self.update_global_patterns(agents);
        
        Ok(())
    }

    fn compute_agent_intersection(&self, agent: &AgentMetadata, all_agents: &[AgentMetadata]) -> Result<AgentIntersection, UpdaterError> {
        let intersects_with = self.compute_intersection_relationships(agent, all_agents);
        let coordination_patterns = self.compute_coordination_patterns(agent);
        
        Ok(AgentIntersection {
            agent_name: agent.name.clone(),
            agent_type: agent.agent_type.clone(),
            domain: agent.domain.clone(),
            intersects_with,
            coordination_patterns,
            intelligence_modes: agent.intelligence_support.clone(),
        })
    }

    fn compute_intersection_relationships(&self, agent: &AgentMetadata, all_agents: &[AgentMetadata]) -> Vec<IntersectionRelationship> {
        let mut relationships = Vec::new();
        
        // Domain-based intersections
        for other_agent in all_agents {
            if other_agent.name == agent.name {
                continue;
            }
            
            let relationship = self.determine_relationship(agent, other_agent);
            if let Some(rel) = relationship {
                relationships.push(rel);
            }
        }
        
        // Always include orchestrator coordination
        if agent.name != "orchestrator" {
            relationships.push(IntersectionRelationship {
                agent: "orchestrator".to_string(),
                relationship_type: RelationshipType::Coordination,
                coordination_method: "Workflow coordination and task routing".to_string(),
                shared_responsibilities: vec!["Task coordination".to_string(), "Quality gates".to_string()],
                handoff_patterns: vec!["Request routing".to_string(), "Status reporting".to_string()],
            });
        }
        
        relationships
    }

    fn determine_relationship(&self, agent: &AgentMetadata, other_agent: &AgentMetadata) -> Option<IntersectionRelationship> {
        // Define intersection rules based on agent types and domains
        let relationship_type = match (&agent.agent_type[..], &other_agent.agent_type[..]) {
            ("Specialist", "Specialist") if agent.domain == other_agent.domain => RelationshipType::Primary,
            ("Specialist", "Specialist") => RelationshipType::Secondary,
            (_, "Orchestrator") => RelationshipType::Coordination,
            ("Specialist", "Support") => RelationshipType::Validation,
            _ => RelationshipType::Secondary,
        };
        
        let coordination_method = match relationship_type {
            RelationshipType::Primary => format!("Direct collaboration on {} tasks", agent.domain),
            RelationshipType::Secondary => format!("Supporting {} with {} capabilities", agent.domain, other_agent.domain),
            RelationshipType::Coordination => "Workflow coordination and task management".to_string(),
            RelationshipType::Validation => format!("Quality validation for {} outputs", agent.domain),
            RelationshipType::Escalation => format!("Escalation path for {} issues", agent.domain),
        };
        
        Some(IntersectionRelationship {
            agent: other_agent.name.clone(),
            relationship_type,
            coordination_method,
            shared_responsibilities: self.compute_shared_responsibilities(agent, other_agent),
            handoff_patterns: self.compute_handoff_patterns(agent, other_agent),
        })
    }

    fn compute_shared_responsibilities(&self, agent: &AgentMetadata, other_agent: &AgentMetadata) -> Vec<String> {
        let mut shared = Vec::new();
        
        if agent.domain == other_agent.domain {
            shared.push(format!("{} implementation", agent.domain));
            shared.push(format!("{} quality assurance", agent.domain));
        }
        
        if agent.intelligence_support.swarm_capable && other_agent.intelligence_support.swarm_capable {
            shared.push("Swarm coordination".to_string());
        }
        
        if agent.intelligence_support.hive_mind_capable && other_agent.intelligence_support.hive_mind_capable {
            shared.push("Hive mind synchronization".to_string());
        }
        
        shared
    }

    fn compute_handoff_patterns(&self, _agent: &AgentMetadata, _other_agent: &AgentMetadata) -> Vec<String> {
        vec![
            "Task completion notification".to_string(),
            "Quality gate validation".to_string(),
            "Context transfer".to_string(),
        ]
    }

    fn compute_coordination_patterns(&self, agent: &AgentMetadata) -> Vec<String> {
        let mut patterns = Vec::new();
        
        match agent.agent_type.as_str() {
            "Orchestrator" => {
                patterns.extend(vec![
                    "Central coordination".to_string(),
                    "Task routing".to_string(),
                    "Multi-agent workflow management".to_string(),
                ]);
            },
            "Specialist" => {
                patterns.extend(vec![
                    "Domain expertise".to_string(),
                    "Quality implementation".to_string(),
                    "Cross-domain collaboration".to_string(),
                ]);
            },
            "Support" => {
                patterns.extend(vec![
                    "Infrastructure support".to_string(),
                    "Quality validation".to_string(),
                    "System monitoring".to_string(),
                ]);
            },
            _ => {
                patterns.push("General coordination".to_string());
            }
        }
        
        if agent.intelligence_support.swarm_capable {
            patterns.push("Swarm intelligence coordination".to_string());
        }
        
        if agent.intelligence_support.hive_mind_capable {
            patterns.push("Hive mind synchronization".to_string());
        }
        
        patterns
    }

    fn update_global_patterns(&mut self, agents: &[AgentMetadata]) {
        // Update orchestrator routing patterns
        self.current_matrix.global_patterns.orchestrator_routing = vec![
            "MANDATORY orchestrator consultation".to_string(),
            "Task routing through orchestrator.md".to_string(),
            "Multi-agent coordination".to_string(),
        ];
        
        // Update Docker integration patterns
        let docker_capable_agents: Vec<_> = agents.iter()
            .filter(|a| a.container_role != "NONE")
            .collect();
        
        self.current_matrix.global_patterns.docker_integration_patterns = vec![
            format!("{} agents with container integration", docker_capable_agents.len()),
            "Universal /api endpoint coordination".to_string(),
            "MCP server integration".to_string(),
        ];
        
        // Update intelligence mode patterns
        let swarm_agents = agents.iter().filter(|a| a.intelligence_support.swarm_capable).count();
        let hive_mind_agents = agents.iter().filter(|a| a.intelligence_support.hive_mind_capable).count();
        
        self.current_matrix.global_patterns.intelligence_mode_patterns = vec![
            format!("{} agents support swarm intelligence", swarm_agents),
            format!("{} agents support hive mind intelligence", hive_mind_agents),
            "Dynamic mode selection".to_string(),
        ];
    }

    fn load_matrix(file_path: &Path) -> Result<AgentIntersectionMatrix, UpdaterError> {
        let content = fs::read_to_string(file_path)?;
        serde_json::from_str(&content).map_err(UpdaterError::SerializationError)
    }

    fn save_matrix(&self) -> Result<(), UpdaterError> {
        let content = serde_json::to_string_pretty(&self.current_matrix)
            .map_err(UpdaterError::SerializationError)?;
        fs::write(&self.matrix_file_path, content)?;
        Ok(())
    }

    fn create_default_matrix() -> AgentIntersectionMatrix {
        AgentIntersectionMatrix {
            version: "1.0".to_string(),
            last_updated: chrono::Utc::now().to_rfc3339(),
            intersections: HashMap::new(),
            global_patterns: GlobalIntersectionPatterns {
                orchestrator_routing: vec![
                    "MANDATORY orchestrator consultation".to_string(),
                ],
                common_quality_gates: vec![
                    "Code quality validation".to_string(),
                    "Test coverage verification".to_string(),
                ],
                docker_integration_patterns: vec![
                    "Container role definition".to_string(),
                    "API endpoint coordination".to_string(),
                ],
                intelligence_mode_patterns: vec![
                    "Swarm vs hive mind selection".to_string(),
                ],
            },
        }
    }

    /// Generate intersection matrix markdown for agent configs
    pub fn generate_intersection_section(&self, agent_name: &str) -> String {
        if let Some(intersection) = self.current_matrix.intersections.get(agent_name) {
            let mut content = String::new();
            content.push_str("**This agent intersects with the following agents for coordinated workflows:**\n\n");
            
            for relationship in &intersection.intersects_with {
                content.push_str(&format!(
                    "- **{}**: {} ({})\n",
                    relationship.agent,
                    relationship.coordination_method,
                    format!("{:?}", relationship.relationship_type).to_lowercase()
                ));
            }
            
            content
        } else {
            "- **No specific intersections defined** - Coordinates through orchestrator.md".to_string()
        }
    }
}

#[derive(Debug)]
struct AgentMetadata {
    name: String,
    agent_type: String,
    domain: String,
    container_role: String,
    file_path: PathBuf,
    existing_intersections: Vec<String>,
    intelligence_support: IntelligenceModeSupport,
}

#[derive(Debug, thiserror::Error)]
pub enum UpdaterError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),
}

