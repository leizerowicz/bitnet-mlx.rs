//! # Agent Configuration Data Extractor
//! 
//! Extracts and processes agent configuration data for training neural networks
//! that determine optimal intelligence modes (swarm vs hive mind).

use crate::{IntelligenceResult, IntelligenceError, TaskCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// Agent configuration data extractor
#[derive(Debug)]
pub struct AgentConfigExtractor {
    /// Path to agent configuration directory
    config_path: std::path::PathBuf,
    /// Extracted agent configurations
    agent_configs: HashMap<String, AgentConfig>,
    /// Agent intersection matrix
    intersection_matrix: IntersectionMatrix,
    /// Feature extractors for different agent types
    feature_extractors: HashMap<String, FeatureExtractor>,
}

/// Individual agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name/identifier
    pub name: String,
    /// Agent type/role
    pub agent_type: String,
    /// Capabilities and skills
    pub capabilities: Vec<String>,
    /// Intersections with other agents
    pub intersections: Vec<String>,
    /// Complexity level (0.0-1.0)
    pub complexity_level: f32,
    /// Collaboration requirements
    pub collaboration_requirements: CollaborationRequirements,
    /// Specialization focus
    pub specialization_focus: SpecializationFocus,
}

/// Collaboration requirements for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationRequirements {
    /// Requires other agents (0.0-1.0)
    pub dependency_level: f32,
    /// Benefits from parallel work (0.0-1.0)
    pub parallelization_benefit: f32,
    /// Requires synchronization (0.0-1.0)
    pub synchronization_need: f32,
    /// Preferred working mode
    pub preferred_mode: WorkingMode,
}

/// Agent specialization focus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializationFocus {
    /// Primary domain
    pub primary_domain: String,
    /// Secondary domains
    pub secondary_domains: Vec<String>,
    /// Expertise level (0.0-1.0)
    pub expertise_level: f32,
    /// Adaptability (0.0-1.0)
    pub adaptability: f32,
}

/// Working mode preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkingMode {
    /// Independent work preferred
    Independent,
    /// Collaborative work preferred
    Collaborative,
    /// Unified/synchronized work preferred
    Unified,
    /// Adaptive to task requirements
    Adaptive,
}

/// Agent intersection matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionMatrix {
    /// Matrix of agent intersections
    pub intersections: HashMap<String, HashMap<String, IntersectionStrength>>,
    /// Collaboration patterns
    pub collaboration_patterns: Vec<CollaborationPattern>,
    /// Common workflows
    pub workflows: Vec<Workflow>,
}

/// Strength of intersection between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntersectionStrength {
    /// No intersection
    None,
    /// Weak intersection
    Weak(f32),
    /// Moderate intersection
    Moderate(f32),
    /// Strong intersection
    Strong(f32),
    /// Critical intersection
    Critical(f32),
}

/// Collaboration pattern between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationPattern {
    /// Pattern identifier
    pub id: String,
    /// Participating agents
    pub agents: Vec<String>,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Effectiveness score (0.0-1.0)
    pub effectiveness: f32,
    /// Usage frequency
    pub frequency: usize,
}

/// Types of collaboration patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Sequential handoff pattern
    Sequential,
    /// Parallel execution pattern
    Parallel,
    /// Hierarchical coordination pattern
    Hierarchical,
    /// Peer-to-peer collaboration pattern
    PeerToPeer,
    /// Hub-and-spoke pattern
    HubAndSpoke,
}

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow identifier
    pub id: String,
    /// Workflow steps
    pub steps: Vec<WorkflowStep>,
    /// Required agents
    pub required_agents: Vec<String>,
    /// Estimated duration
    pub estimated_duration: std::time::Duration,
    /// Complexity level
    pub complexity: f32,
}

/// Individual workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Step identifier
    pub id: String,
    /// Responsible agent
    pub agent: String,
    /// Step description
    pub description: String,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Estimated duration
    pub duration: std::time::Duration,
}

/// Feature extractor for specific agent types
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Agent type this extractor handles
    pub agent_type: String,
    /// Feature extraction rules
    pub extraction_rules: Vec<ExtractionRule>,
    /// Normalization parameters
    pub normalization: NormalizationParams,
}

/// Feature extraction rule
#[derive(Debug, Clone)]
pub struct ExtractionRule {
    /// Rule identifier
    pub id: String,
    /// Source field path
    pub source_field: String,
    /// Target feature name
    pub target_feature: String,
    /// Transformation function
    pub transformation: TransformationType,
}

/// Feature transformation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Direct mapping
    Direct,
    /// Logarithmic transformation
    Logarithmic,
    /// Normalization (0-1)
    Normalize,
    /// Boolean to float
    BooleanToFloat,
    /// Count transformation
    Count,
    /// Average aggregation
    Average,
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Minimum values for features
    pub min_values: HashMap<String, f32>,
    /// Maximum values for features
    pub max_values: HashMap<String, f32>,
    /// Mean values for features
    pub mean_values: HashMap<String, f32>,
    /// Standard deviations
    pub std_values: HashMap<String, f32>,
}

impl AgentConfigExtractor {
    /// Create new agent configuration extractor
    pub async fn new(config_path: impl AsRef<Path>) -> IntelligenceResult<Self> {
        let config_path = config_path.as_ref().to_path_buf();
        let mut extractor = Self {
            config_path,
            agent_configs: HashMap::new(),
            intersection_matrix: IntersectionMatrix {
                intersections: HashMap::new(),
                collaboration_patterns: Vec::new(),
                workflows: Vec::new(),
            },
            feature_extractors: HashMap::new(),
        };
        
        // Initialize feature extractors
        extractor.initialize_feature_extractors().await?;
        
        // Load agent configurations
        extractor.load_agent_configurations().await?;
        
        // Build intersection matrix
        extractor.build_intersection_matrix().await?;
        
        Ok(extractor)
    }

    /// Extract task characteristics from agent configuration patterns
    pub async fn extract_task_characteristics(
        &self,
        task_description: &str
    ) -> IntelligenceResult<TaskCharacteristics> {
        // Analyze task description to identify required agents
        let required_agents = self.identify_required_agents(task_description).await?;
        
        // Calculate complexity based on agent requirements
        let complexity = self.calculate_complexity(&required_agents).await?;
        
        // Determine parallelization potential
        let parallelizable = self.assess_parallelization(&required_agents).await?;
        
        // Evaluate synchronization requirements
        let sync_required = self.evaluate_synchronization_needs(&required_agents).await?;
        
        // Assess collaboration benefits
        let collaboration_benefit = self.assess_collaboration_benefit(&required_agents).await?;
        
        // Determine unity requirements
        let unity_required = self.determine_unity_requirements(&required_agents).await?;
        
        // Extract agent specializations
        let agent_specializations = required_agents.iter()
            .map(|agent| agent.clone())
            .collect();
        
        Ok(TaskCharacteristics {
            complexity,
            parallelizable,
            sync_required,
            collaboration_benefit,
            unity_required,
            agent_specializations,
        })
    }

    /// Load agent configurations from files
    async fn load_agent_configurations(&mut self) -> IntelligenceResult<()> {
        let mut entries = fs::read_dir(&self.config_path).await
            .map_err(|e| IntelligenceError::Io(e))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| IntelligenceError::Io(e))? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("md") {
                let agent_name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let config = self.parse_agent_config(&path).await?;
                self.agent_configs.insert(agent_name, config);
            }
        }
        
        Ok(())
    }

    /// Parse individual agent configuration file
    async fn parse_agent_config(&self, path: &Path) -> IntelligenceResult<AgentConfig> {
        let content = fs::read_to_string(path).await
            .map_err(|e| IntelligenceError::Io(e))?;
        
        let agent_name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Extract agent type from content
        let agent_type = self.extract_agent_type(&content);
        
        // Extract capabilities
        let capabilities = self.extract_capabilities(&content);
        
        // Extract intersections
        let intersections = self.extract_intersections(&content);
        
        // Calculate complexity level
        let complexity_level = self.calculate_agent_complexity(&content);
        
        // Extract collaboration requirements
        let collaboration_requirements = self.extract_collaboration_requirements(&content);
        
        // Extract specialization focus
        let specialization_focus = self.extract_specialization_focus(&content);
        
        Ok(AgentConfig {
            name: agent_name,
            agent_type,
            capabilities,
            intersections,
            complexity_level,
            collaboration_requirements,
            specialization_focus,
        })
    }

    /// Extract agent type from configuration content
    fn extract_agent_type(&self, content: &str) -> String {
        // Look for agent type indicators in the content
        if content.contains("specialist") {
            "specialist".to_string()
        } else if content.contains("orchestrator") {
            "orchestrator".to_string()
        } else if content.contains("coordinator") {
            "coordinator".to_string()
        } else if content.contains("manager") {
            "manager".to_string()
        } else {
            "general".to_string()
        }
    }

    /// Extract capabilities from configuration content
    fn extract_capabilities(&self, content: &str) -> Vec<String> {
        let mut capabilities = Vec::new();
        
        // Look for capability indicators
        let capability_keywords = vec![
            "code", "debug", "test", "performance", "security", "documentation",
            "inference", "architecture", "deployment", "monitoring", "analysis"
        ];
        
        for keyword in capability_keywords {
            if content.to_lowercase().contains(keyword) {
                capabilities.push(keyword.to_string());
            }
        }
        
        capabilities
    }

    /// Extract agent intersections from configuration content
    fn extract_intersections(&self, content: &str) -> Vec<String> {
        let mut intersections = Vec::new();
        
        // Look for "intersects with" patterns
        if let Some(intersects_section) = content.find("intersects with:") {
            let section = &content[intersects_section..];
            if let Some(end) = section.find('\n') {
                let intersection_line = &section[..end];
                for word in intersection_line.split_whitespace() {
                    if word.ends_with(".md") {
                        let agent_name = word.trim_end_matches(".md");
                        intersections.push(agent_name.to_string());
                    }
                }
            }
        }
        
        intersections
    }

    /// Calculate agent complexity level
    fn calculate_agent_complexity(&self, content: &str) -> f32 {
        let mut complexity = 0.0;
        
        // Factor in content length
        complexity += (content.len() as f32 / 10000.0).min(0.3);
        
        // Factor in number of responsibilities
        let responsibility_count = content.matches("responsible for").count() as f32;
        complexity += (responsibility_count / 10.0).min(0.3);
        
        // Factor in intersection count
        let intersection_count = content.matches("intersects with").count() as f32;
        complexity += (intersection_count / 5.0).min(0.4);
        
        complexity.min(1.0)
    }

    /// Extract collaboration requirements
    fn extract_collaboration_requirements(&self, content: &str) -> CollaborationRequirements {
        let dependency_level = if content.contains("depends on") || content.contains("requires") {
            0.7
        } else {
            0.3
        };
        
        let parallelization_benefit = if content.contains("parallel") || content.contains("concurrent") {
            0.8
        } else {
            0.4
        };
        
        let synchronization_need = if content.contains("synchroniz") || content.contains("coordinat") {
            0.8
        } else {
            0.3
        };
        
        let preferred_mode = if content.contains("independent") {
            WorkingMode::Independent
        } else if content.contains("collaborative") {
            WorkingMode::Collaborative
        } else if content.contains("unified") {
            WorkingMode::Unified
        } else {
            WorkingMode::Adaptive
        };
        
        CollaborationRequirements {
            dependency_level,
            parallelization_benefit,
            synchronization_need,
            preferred_mode,
        }
    }

    /// Extract specialization focus
    fn extract_specialization_focus(&self, content: &str) -> SpecializationFocus {
        // Determine primary domain from content
        let primary_domain = if content.contains("code") || content.contains("development") {
            "development".to_string()
        } else if content.contains("debug") || content.contains("troubleshoot") {
            "debugging".to_string()
        } else if content.contains("test") || content.contains("validation") {
            "testing".to_string()
        } else if content.contains("performance") || content.contains("optimization") {
            "performance".to_string()
        } else if content.contains("security") || content.contains("safety") {
            "security".to_string()
        } else {
            "general".to_string()
        };
        
        let secondary_domains = Vec::new(); // Could be extracted with more sophisticated parsing
        
        let expertise_level = content.len() as f32 / 5000.0; // Rough estimate based on content depth
        let adaptability = 0.7; // Default adaptability
        
        SpecializationFocus {
            primary_domain,
            secondary_domains,
            expertise_level: expertise_level.min(1.0),
            adaptability,
        }
    }

    /// Initialize feature extractors for different agent types
    async fn initialize_feature_extractors(&mut self) -> IntelligenceResult<()> {
        // Create feature extractors for common agent types
        let agent_types = vec!["specialist", "orchestrator", "coordinator", "manager", "general"];
        
        for agent_type in agent_types {
            let extractor = FeatureExtractor::new(agent_type.to_string());
            self.feature_extractors.insert(agent_type.to_string(), extractor);
        }
        
        Ok(())
    }

    /// Build intersection matrix from agent configurations
    async fn build_intersection_matrix(&mut self) -> IntelligenceResult<()> {
        // Build intersection relationships
        for (agent_name, config) in &self.agent_configs {
            let mut agent_intersections = HashMap::new();
            
            for intersection in &config.intersections {
                if self.agent_configs.contains_key(intersection) {
                    // Determine intersection strength based on configuration
                    let strength = self.calculate_intersection_strength(agent_name, intersection).await;
                    agent_intersections.insert(intersection.clone(), strength);
                }
            }
            
            self.intersection_matrix.intersections.insert(agent_name.clone(), agent_intersections);
        }
        
        Ok(())
    }

    /// Calculate intersection strength between two agents
    async fn calculate_intersection_strength(&self, agent1: &str, agent2: &str) -> IntersectionStrength {
        if let (Some(config1), Some(config2)) = (self.agent_configs.get(agent1), self.agent_configs.get(agent2)) {
            // Calculate based on capability overlap
            let capability_overlap = config1.capabilities.iter()
                .filter(|cap| config2.capabilities.contains(cap))
                .count() as f32;
            
            let max_capabilities = config1.capabilities.len().max(config2.capabilities.len()) as f32;
            let overlap_ratio = if max_capabilities > 0.0 {
                capability_overlap / max_capabilities
            } else {
                0.0
            };
            
            match overlap_ratio {
                x if x > 0.8 => IntersectionStrength::Critical(x),
                x if x > 0.6 => IntersectionStrength::Strong(x),
                x if x > 0.4 => IntersectionStrength::Moderate(x),
                x if x > 0.2 => IntersectionStrength::Weak(x),
                _ => IntersectionStrength::None,
            }
        } else {
            IntersectionStrength::None
        }
    }

    /// Identify required agents for a task
    async fn identify_required_agents(&self, task_description: &str) -> IntelligenceResult<Vec<String>> {
        let mut required_agents = Vec::new();
        
        // Match task description against agent capabilities
        for (agent_name, config) in &self.agent_configs {
            for capability in &config.capabilities {
                if task_description.to_lowercase().contains(capability) {
                    required_agents.push(agent_name.clone());
                    break;
                }
            }
        }
        
        // If no specific matches, include general agents
        if required_agents.is_empty() {
            for (agent_name, config) in &self.agent_configs {
                if config.agent_type == "general" || config.agent_type == "orchestrator" {
                    required_agents.push(agent_name.clone());
                }
            }
        }
        
        Ok(required_agents)
    }

    /// Calculate task complexity based on required agents
    async fn calculate_complexity(&self, required_agents: &[String]) -> IntelligenceResult<f32> {
        if required_agents.is_empty() {
            return Ok(0.0);
        }
        
        let total_complexity: f32 = required_agents.iter()
            .filter_map(|agent| self.agent_configs.get(agent))
            .map(|config| config.complexity_level)
            .sum();
        
        let avg_complexity = total_complexity / required_agents.len() as f32;
        
        // Factor in number of agents (more agents = higher complexity)
        let agent_factor = (required_agents.len() as f32 / 10.0).min(0.5);
        
        Ok((avg_complexity + agent_factor).min(1.0))
    }

    /// Assess parallelization potential
    async fn assess_parallelization(&self, required_agents: &[String]) -> IntelligenceResult<f32> {
        if required_agents.is_empty() {
            return Ok(0.0);
        }
        
        let parallel_benefit: f32 = required_agents.iter()
            .filter_map(|agent| self.agent_configs.get(agent))
            .map(|config| config.collaboration_requirements.parallelization_benefit)
            .sum();
        
        Ok((parallel_benefit / required_agents.len() as f32).min(1.0))
    }

    /// Evaluate synchronization needs
    async fn evaluate_synchronization_needs(&self, required_agents: &[String]) -> IntelligenceResult<f32> {
        if required_agents.is_empty() {
            return Ok(0.0);
        }
        
        let sync_need: f32 = required_agents.iter()
            .filter_map(|agent| self.agent_configs.get(agent))
            .map(|config| config.collaboration_requirements.synchronization_need)
            .sum();
        
        Ok((sync_need / required_agents.len() as f32).min(1.0))
    }

    /// Assess collaboration benefit
    async fn assess_collaboration_benefit(&self, required_agents: &[String]) -> IntelligenceResult<f32> {
        if required_agents.len() <= 1 {
            return Ok(0.0);
        }
        
        // Higher benefit with more agents and more intersections
        let intersection_count = self.count_agent_intersections(required_agents).await;
        let max_possible_intersections = required_agents.len() * (required_agents.len() - 1);
        
        if max_possible_intersections > 0 {
            Ok((intersection_count as f32 / max_possible_intersections as f32).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Determine unity requirements
    async fn determine_unity_requirements(&self, required_agents: &[String]) -> IntelligenceResult<f32> {
        if required_agents.is_empty() {
            return Ok(0.0);
        }
        
        // Unity required when agents have high synchronization needs
        let unity_indicators: f32 = required_agents.iter()
            .filter_map(|agent| self.agent_configs.get(agent))
            .map(|config| {
                match config.collaboration_requirements.preferred_mode {
                    WorkingMode::Unified => 1.0,
                    WorkingMode::Collaborative => 0.6,
                    WorkingMode::Adaptive => 0.4,
                    WorkingMode::Independent => 0.1,
                }
            })
            .sum();
        
        Ok((unity_indicators / required_agents.len() as f32).min(1.0))
    }

    /// Count intersections between required agents
    async fn count_agent_intersections(&self, required_agents: &[String]) -> usize {
        let mut count = 0;
        
        for agent in required_agents {
            if let Some(intersections) = self.intersection_matrix.intersections.get(agent) {
                for other_agent in required_agents {
                    if agent != other_agent && intersections.contains_key(other_agent) {
                        count += 1;
                    }
                }
            }
        }
        
        count
    }

    /// Get agent configuration by name
    pub fn get_agent_config(&self, name: &str) -> Option<&AgentConfig> {
        self.agent_configs.get(name)
    }

    /// Get all agent configurations
    pub fn get_all_agent_configs(&self) -> &HashMap<String, AgentConfig> {
        &self.agent_configs
    }

    /// Get intersection matrix
    pub fn get_intersection_matrix(&self) -> &IntersectionMatrix {
        &self.intersection_matrix
    }
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(agent_type: String) -> Self {
        let extraction_rules = Self::create_default_rules(&agent_type);
        let normalization = NormalizationParams {
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            mean_values: HashMap::new(),
            std_values: HashMap::new(),
        };
        
        Self {
            agent_type,
            extraction_rules,
            normalization,
        }
    }

    /// Create default extraction rules for agent type
    fn create_default_rules(agent_type: &str) -> Vec<ExtractionRule> {
        match agent_type {
            "specialist" => vec![
                ExtractionRule {
                    id: "complexity".to_string(),
                    source_field: "complexity_level".to_string(),
                    target_feature: "complexity".to_string(),
                    transformation: TransformationType::Direct,
                },
                ExtractionRule {
                    id: "collaboration".to_string(),
                    source_field: "collaboration_requirements.dependency_level".to_string(),
                    target_feature: "collaboration_need".to_string(),
                    transformation: TransformationType::Direct,
                },
            ],
            _ => Vec::new(),
        }
    }
}