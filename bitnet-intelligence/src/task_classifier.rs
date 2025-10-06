//! # Task Classifier
//! 
//! Classifies tasks and predicts optimal intelligence modes using neural networks
//! trained on agent configuration patterns and task characteristics.

use crate::{
    IntelligenceResult, IntelligenceDecision, IntelligenceType,
    TaskCharacteristics, AgentConfigExtractor, IntelligenceMode
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Task classifier for intelligence mode prediction
#[derive(Debug)]
pub struct TaskClassifier {
    /// Agent configuration extractor
    extractor: AgentConfigExtractor,
    /// Learned task patterns
    task_patterns: TaskPatterns,
    /// Classification metrics
    metrics: ClassificationMetrics,
}

/// Collection of learned task patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPatterns {
    /// Pattern database indexed by pattern ID
    patterns: HashMap<String, TaskPattern>,
    /// Classification history for learning
    classification_history: Vec<ClassificationRecord>,
}

/// Individual task pattern with performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPattern {
    /// Pattern identifier
    id: String,
    /// Pattern description
    description: String,
    /// Task characteristics that define this pattern
    characteristics: TaskCharacteristics,
    /// Optimal intelligence mode for this pattern
    optimal_mode: IntelligenceMode,
    /// Confidence in the prediction
    confidence: f32,
    /// Frequency of this pattern
    frequency: u32,
    /// Success rate of predictions
    success_rate: f32,
}

/// Classification record for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationRecord {
    /// Record identifier
    id: String,
    /// Task characteristics
    characteristics: TaskCharacteristics,
    /// Predicted intelligence mode
    predicted_mode: IntelligenceMode,
    /// Actual intelligence mode used (for learning)
    actual_mode: Option<IntelligenceMode>,
    /// Performance metrics
    performance: Option<f32>,
    /// Timestamp
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Classification accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Total number of classifications performed
    total_classifications: u32,
    /// Number of correct predictions
    correct_predictions: u32,
    /// Average confidence in predictions
    average_confidence: f32,
    /// Pattern learning rate
    learning_rate: f32,
}

impl ClassificationMetrics {
    /// Create new classification metrics
    pub fn new() -> Self {
        Self {
            total_classifications: 0,
            correct_predictions: 0,
            average_confidence: 0.0,
            learning_rate: 0.1,
        }
    }
}

impl TaskPatterns {
    /// Create new empty task patterns
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            classification_history: Vec::new(),
        }
    }
}

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self {
            total_classifications: 0,
            correct_predictions: 0,
            average_confidence: 0.0,
            learning_rate: 0.1,
        }
    }
}

impl TaskClassifier {
    /// Create a new task classifier
    pub async fn new() -> IntelligenceResult<Self> {
        Ok(Self {
            extractor: AgentConfigExtractor::new(Path::new("agent-config")).await?,
            task_patterns: TaskPatterns::new(),
            metrics: ClassificationMetrics::new(),
        })
    }

    /// Analyze task to extract characteristics
    pub async fn analyze_task(&self, task_description: &str) -> IntelligenceResult<TaskCharacteristics> {
        self.analyze_task_characteristics(task_description).await
    }

    /// Analyze task characteristics from description
    pub async fn analyze_task_characteristics(&self, task_description: &str) -> IntelligenceResult<TaskCharacteristics> {
        // Extract keywords and patterns using the agent config extractor
        let keywords: Vec<String> = task_description
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        // Calculate characteristics based on keywords and patterns
        let complexity = self.calculate_complexity(&keywords);
        let parallelizable = self.calculate_parallelizable(&keywords);
        let sync_required = self.calculate_sync_required(&keywords);
        let collaboration_benefit = self.calculate_collaboration_benefit(&keywords);
        let unity_required = self.calculate_unity_required(&keywords);
        let agent_specializations = self.extract_agent_specializations(&keywords);

        Ok(TaskCharacteristics {
            complexity,
            parallelizable,
            sync_required,
            collaboration_benefit,
            unity_required,
            agent_specializations,
        })
    }

    /// Calculate task complexity from keywords
    fn calculate_complexity(&self, keywords: &[String]) -> f32 {
        let complexity_indicators = ["complex", "difficult", "multi-step", "intricate", "challenging"];
        let matches = keywords.iter()
            .filter(|k| complexity_indicators.iter().any(|&i| k.contains(i)))
            .count();
        (matches as f32 / keywords.len() as f32).min(1.0)
    }

    /// Calculate parallelizability from keywords
    fn calculate_parallelizable(&self, keywords: &[String]) -> f32 {
        let parallel_indicators = ["parallel", "concurrent", "distributed", "multiple", "batch"];
        let matches = keywords.iter()
            .filter(|k| parallel_indicators.iter().any(|&i| k.contains(i)))
            .count();
        (matches as f32 / keywords.len() as f32).min(1.0)
    }

    /// Calculate synchronization requirements from keywords
    fn calculate_sync_required(&self, keywords: &[String]) -> f32 {
        let sync_indicators = ["sync", "coordinate", "sequential", "ordered", "synchronized"];
        let matches = keywords.iter()
            .filter(|k| sync_indicators.iter().any(|&i| k.contains(i)))
            .count();
        (matches as f32 / keywords.len() as f32).min(1.0)
    }

    /// Calculate collaboration benefit from keywords
    fn calculate_collaboration_benefit(&self, keywords: &[String]) -> f32 {
        let collab_indicators = ["collaborate", "team", "group", "together", "shared"];
        let matches = keywords.iter()
            .filter(|k| collab_indicators.iter().any(|&i| k.contains(i)))
            .count();
        (matches as f32 / keywords.len() as f32).min(1.0)
    }

    /// Calculate unity requirements from keywords
    fn calculate_unity_required(&self, keywords: &[String]) -> f32 {
        let unity_indicators = ["unified", "consensus", "agreement", "collective", "unanimous"];
        let matches = keywords.iter()
            .filter(|k| unity_indicators.iter().any(|&i| k.contains(i)))
            .count();
        (matches as f32 / keywords.len() as f32).min(1.0)
    }

    /// Extract agent specializations from keywords
    fn extract_agent_specializations(&self, keywords: &[String]) -> Vec<String> {
        let specializations = [
            "debug", "code", "test", "performance", "security", "documentation",
            "inference", "architect", "research", "business"
        ];
        
        keywords.iter()
            .filter_map(|k| {
                specializations.iter()
                    .find(|&s| k.contains(s))
                    .map(|s| s.to_string())
            })
            .collect()
    }

    /// Predict the optimal intelligence mode for a task
    pub async fn predict_intelligence_mode(&self, characteristics: &TaskCharacteristics) -> IntelligenceResult<IntelligenceMode> {
        // For now, use a simple heuristic based on characteristics
        let swarm_score = self.calculate_swarm_score(characteristics).await;
        let hive_score = self.calculate_hive_mind_score(characteristics).await;
        
        if swarm_score > hive_score {
            Ok(IntelligenceMode::Swarm)
        } else if hive_score > swarm_score {
            Ok(IntelligenceMode::HiveMind)
        } else {
            // If scores are equal, create a hybrid mode
            Ok(IntelligenceMode::Hybrid { 
                swarm_weight: 0.5, 
                hive_weight: 0.5 
            })
        }
    }

    /// Classify task and predict optimal intelligence mode
    pub async fn classify_task(
        &mut self,
        characteristics: &TaskCharacteristics
    ) -> IntelligenceResult<IntelligenceDecision> {
        // Check for existing patterns
        let pattern_id = self.generate_pattern_id(characteristics);
        if let Some(pattern) = self.task_patterns.patterns.get(&pattern_id) {
            let intelligence_type = self.create_intelligence_type(&pattern.optimal_mode, characteristics);
            return Ok(IntelligenceDecision {
                intelligence_type,
                confidence: pattern.confidence,
                reasoning: format!("Based on learned pattern: {}", pattern.description),
                agent_requirements: self.calculate_agent_requirements(characteristics),
            });
        }
        
        // Perform new classification
        let swarm_score = self.calculate_swarm_score(characteristics).await;
        let hive_mind_score = self.calculate_hive_mind_score(characteristics).await;
        
        let (mode, confidence, reasoning) = if swarm_score > hive_mind_score {
            (
                IntelligenceMode::Swarm,
                swarm_score,
                "Task benefits from divergent parallel processing".to_string()
            )
        } else {
            (
                IntelligenceMode::HiveMind,
                hive_mind_score,
                "Task requires synchronized collective processing".to_string()
            )
        };
        
        // Record classification for learning
        let record = ClassificationRecord {
            id: uuid::Uuid::new_v4().to_string(),
            characteristics: characteristics.clone(),
            predicted_mode: mode.clone(),
            actual_mode: None,
            performance: None,
            timestamp: chrono::Utc::now(),
        };
        self.task_patterns.classification_history.push(record);
        
        // Update metrics
        self.metrics.total_classifications += 1;
        self.metrics.average_confidence = 
            (self.metrics.average_confidence * (self.metrics.total_classifications - 1) as f32 + confidence) 
            / self.metrics.total_classifications as f32;
        
        let intelligence_type = self.create_intelligence_type(&mode, characteristics);
        Ok(IntelligenceDecision {
            intelligence_type,
            confidence,
            reasoning,
            agent_requirements: self.calculate_agent_requirements(characteristics),
        })
    }

    /// Update task classification pattern based on feedback
    pub async fn update_task_classification_pattern(
        &mut self,
        record_id: &str,
        actual_mode: IntelligenceMode,
        performance: f32
    ) -> IntelligenceResult<()> {
        // Clone actual_mode to avoid move issues
        let actual_mode_clone = actual_mode.clone();
        
        // Find the classification record and extract necessary data
        let (pattern_id, predicted_mode, characteristics) = {
            if let Some(record) = self.task_patterns.classification_history.iter_mut()
                .find(|r| r.id == record_id) {
                
                record.actual_mode = Some(actual_mode.clone());
                record.performance = Some(performance);
                
                // Generate pattern ID without borrowing self
                let pattern_id = format!("pattern_{}_{}", 
                    record.characteristics.complexity as u32, 
                    record.characteristics.parallelizable as u32);
                (pattern_id, record.predicted_mode.clone(), record.characteristics.clone())
            } else {
                return Ok(());
            }
        };
        
        if let Some(pattern) = self.task_patterns.patterns.get_mut(&pattern_id) {
            // Update existing pattern
            pattern.frequency += 1;
            let success = if predicted_mode == actual_mode_clone { 1.0 } else { 0.0 };
            pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) as f32 + success) / pattern.frequency as f32;
        } else {
            // Create new pattern
            let pattern = TaskPattern {
                id: pattern_id.clone(),
                description: format!("Pattern for task type"),
                characteristics,
                optimal_mode: actual_mode_clone.clone(),
                confidence: performance,
                frequency: 1,
                success_rate: if predicted_mode == actual_mode_clone { 1.0 } else { 0.0 },
            };
            self.task_patterns.patterns.insert(pattern_id, pattern);
        }
        
        // Update metrics
        self.metrics.total_classifications += 1;
        
        Ok(())
    }

    /// Generate pattern identifier from characteristics
    fn generate_pattern_id(&self, characteristics: &TaskCharacteristics) -> String {
        format!(
            "pattern_{}_{}_{}_{}_{}",
            (characteristics.complexity * 10.0) as u32,
            (characteristics.parallelizable * 10.0) as u32,
            (characteristics.sync_required * 10.0) as u32,
            (characteristics.collaboration_benefit * 10.0) as u32,
            (characteristics.unity_required * 10.0) as u32
        )
    }

    /// Calculate agent requirements for task
    fn calculate_agent_requirements(&self, characteristics: &TaskCharacteristics) -> HashMap<String, f32> {
        let mut requirements = HashMap::new();
        
        let base_agents = 3.0;
        let complexity_factor = characteristics.complexity * 5.0;
        let parallel_factor = if characteristics.parallelizable > 0.7 { 3.0 } else { 1.0 };
        
        requirements.insert("agent_count".to_string(), (base_agents + complexity_factor) * parallel_factor);
        requirements.insert("memory_per_agent".to_string(), characteristics.complexity * 1024.0);
        requirements.insert("cpu_cores".to_string(), characteristics.parallelizable * 8.0);
        requirements.insert("sync_bandwidth".to_string(), characteristics.sync_required * 100.0);
        
        requirements
    }

    /// Create intelligence type from mode and characteristics
    fn create_intelligence_type(&self, mode: &IntelligenceMode, characteristics: &TaskCharacteristics) -> IntelligenceType {
        match mode {
            IntelligenceMode::Swarm => {
                let agent_count = (3 + (characteristics.complexity * 5.0) as usize).max(2);
                IntelligenceType::Swarm {
                    agent_count,
                    divergence: characteristics.parallelizable,
                    collaboration: characteristics.collaboration_benefit,
                }
            },
            IntelligenceMode::HiveMind => {
                let collective_size = (5 + (characteristics.complexity * 3.0) as usize).max(3);
                IntelligenceType::HiveMind {
                    collective_size,
                    synchronization: characteristics.sync_required,
                    unity: characteristics.unity_required,
                }
            },
            IntelligenceMode::Hybrid { swarm_weight, hive_weight } => {
                // For hybrid mode, choose the stronger component
                if swarm_weight >= hive_weight {
                    let agent_count = (3 + (characteristics.complexity * 3.0 * swarm_weight) as usize).max(2);
                    IntelligenceType::Swarm {
                        agent_count,
                        divergence: characteristics.parallelizable * swarm_weight,
                        collaboration: characteristics.collaboration_benefit,
                    }
                } else {
                    let collective_size = (3 + (characteristics.complexity * 2.0 * hive_weight) as usize).max(3);
                    IntelligenceType::HiveMind {
                        collective_size,
                        synchronization: characteristics.sync_required * hive_weight,
                        unity: characteristics.unity_required,
                    }
                }
            }
        }
    }

    /// Recommend specific architecture based on mode and available configs
    async fn recommend_architecture(
        &self,
        mode: &IntelligenceMode,
        _characteristics: &TaskCharacteristics
    ) -> IntelligenceResult<String> {
        match mode {
            IntelligenceMode::Swarm => {
                Ok("Swarm architecture: distributed parallel processing".to_string())
            },
            IntelligenceMode::HiveMind => {
                Ok("Hive mind architecture: synchronized collective processing".to_string())
            },
            IntelligenceMode::Hybrid { swarm_weight, hive_weight } => {
                Ok(format!("Hybrid architecture: {}% swarm, {}% hive mind", 
                    (swarm_weight * 100.0) as u32, 
                    (hive_weight * 100.0) as u32))
            }
        }
    }

    /// Calculate score for swarm mode
    async fn calculate_swarm_score(
        &self,
        characteristics: &TaskCharacteristics
    ) -> f32 {
        let mut score = 0.0;
        
        // Favor swarm for highly parallelizable tasks
        score += characteristics.parallelizable * 0.4;
        
        // Favor swarm for high complexity that can be divided
        score += characteristics.complexity * 0.3;
        
        // Favor swarm for high collaboration benefit
        score += characteristics.collaboration_benefit * 0.3;
        
        // Penalize swarm for high synchronization requirements
        score -= characteristics.sync_required * 0.2;
        
        // Penalize swarm for high unity requirements
        score -= characteristics.unity_required * 0.1;
        
        score.max(0.0).min(1.0)
    }

    /// Calculate score for hive mind mode
    async fn calculate_hive_mind_score(
        &self,
        characteristics: &TaskCharacteristics
    ) -> f32 {
        let mut score = 0.0;
        
        // Favor hive mind for high synchronization requirements
        score += characteristics.sync_required * 0.4;
        
        // Favor hive mind for high unity requirements
        score += characteristics.unity_required * 0.4;
        
        // Favor hive mind for high complexity
        score += characteristics.complexity * 0.2;
        
        score.max(0.0).min(1.0)
    }
}