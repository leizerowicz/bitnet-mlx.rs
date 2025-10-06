//! # Intelligence System Manager
//! 
//! Central coordinator for Docker BitNet Swarm Intelligence system.
//! Manages the transition between swarm and hive mind intelligence modes.

use crate::{
    TaskClassifier, SwarmIntelligenceOrchestrator, HiveMindCollective, ConsensusEngine,
    IntelligenceType, IntelligenceDecision, TaskCharacteristics, IntelligenceResult, IntelligenceMode
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Main intelligence system manager for Docker BitNet Swarm
#[derive(Debug)]
pub struct IntelligenceSystemManager {
    /// Task classifier for intelligence mode determination
    task_classifier: TaskClassifier,
    /// Swarm intelligence orchestrator
    swarm_orchestrator: SwarmIntelligenceOrchestrator,
    /// Hive mind collective coordinator
    hive_mind_collective: HiveMindCollective,
    /// Consensus engine for decision validation
    consensus_engine: ConsensusEngine,
    /// Current active intelligence mode
    active_mode: Arc<RwLock<Option<IntelligenceType>>>,
    /// Performance metrics tracking
    metrics: Arc<RwLock<IntelligenceMetrics>>,
}

/// Performance metrics for intelligence system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceMetrics {
    /// Swarm mode performance history
    pub swarm_performance: Vec<f32>,
    /// Hive mind mode performance history
    pub hive_mind_performance: Vec<f32>,
    /// Mode switching frequency
    pub mode_switches: usize,
    /// Total tasks processed
    pub tasks_processed: usize,
    /// Average confidence scores
    pub avg_confidence: f32,
}

impl Default for IntelligenceMetrics {
    fn default() -> Self {
        Self {
            swarm_performance: Vec::new(),
            hive_mind_performance: Vec::new(),
            mode_switches: 0,
            tasks_processed: 0,
            avg_confidence: 0.0,
        }
    }
}

impl IntelligenceSystemManager {
    /// Create new intelligence system manager
    pub async fn new(_agent_config_path: impl AsRef<Path>) -> IntelligenceResult<Self> {
        let task_classifier = TaskClassifier::new().await?;
        let swarm_orchestrator = SwarmIntelligenceOrchestrator::new().await?;
        let hive_mind_collective = HiveMindCollective::new().await?;
        let consensus_engine = ConsensusEngine::new().await?;
        
        Ok(Self {
            task_classifier,
            swarm_orchestrator,
            hive_mind_collective,
            consensus_engine,
            active_mode: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(IntelligenceMetrics::default())),
        })
    }

    /// Analyze task and determine optimal intelligence mode
    pub async fn analyze_task(&self, task_description: &str) -> IntelligenceResult<IntelligenceDecision> {
        // Extract task characteristics
        let characteristics = self.task_classifier.analyze_task(task_description).await?;
        
        // Get neural network recommendation
        let intelligence_mode = self.task_classifier.predict_intelligence_mode(&characteristics).await?;
        let intelligence_type = self.create_intelligence_type_from_mode(&intelligence_mode, &characteristics);
        let agent_requirements = self.calculate_agent_requirements(&characteristics);
        
        let neural_decision = IntelligenceDecision {
            intelligence_type,
            confidence: 0.8, // Default confidence
            reasoning: format!("Predicted {:?} mode based on task characteristics", intelligence_mode),
            agent_requirements,
        };
        
        // Validate with consensus engine
        let consensus_decision = self.consensus_engine.validate_decision(&neural_decision, &characteristics).await?;
        
        // Update metrics
        self.update_metrics(&consensus_decision).await;
        
        Ok(consensus_decision)
    }

    /// Create intelligence type from mode and characteristics
    fn create_intelligence_type_from_mode(&self, mode: &IntelligenceMode, characteristics: &TaskCharacteristics) -> IntelligenceType {
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

    /// Execute task with optimal intelligence mode
    pub async fn execute_task(
        &self, 
        task_description: &str,
        task_data: HashMap<String, serde_json::Value>
    ) -> IntelligenceResult<HashMap<String, serde_json::Value>> {
        // Analyze and determine intelligence mode
        let decision = self.analyze_task(task_description).await?;
        
        // Switch to optimal mode if needed
        self.switch_intelligence_mode(&decision.intelligence_type).await?;
        
        // Execute based on intelligence type
        let result = match &decision.intelligence_type {
            IntelligenceType::Swarm { agent_count, divergence, collaboration } => {
                self.swarm_orchestrator.execute_swarm_task(
                    task_data,
                    *agent_count,
                    *divergence,
                    *collaboration
                ).await?
            },
            IntelligenceType::HiveMind { collective_size, synchronization, unity } => {
                self.hive_mind_collective.execute_collective_task(
                    task_data,
                    *collective_size,
                    *synchronization,
                    *unity
                ).await?
            }
        };
        
        // Update performance metrics
        self.update_performance_metrics(&decision.intelligence_type, &result).await;
        
        Ok(result)
    }

    /// Switch between intelligence modes
    async fn switch_intelligence_mode(&self, new_mode: &IntelligenceType) -> IntelligenceResult<()> {
        let mut current_mode = self.active_mode.write().await;
        
        // Check if mode change is needed
        let needs_switch = match (&*current_mode, new_mode) {
            (None, _) => true,
            (Some(current), new) => !self.intelligence_types_match(current, new),
        };
        
        if needs_switch {
            // Gracefully shutdown current mode
            if let Some(current) = current_mode.as_ref() {
                self.shutdown_current_mode(current).await?;
            }
            
            // Initialize new mode
            self.initialize_mode(new_mode).await?;
            
            // Update active mode
            *current_mode = Some(new_mode.clone());
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.mode_switches += 1;
        }
        
        Ok(())
    }

    /// Check if two intelligence types are functionally equivalent
    fn intelligence_types_match(&self, a: &IntelligenceType, b: &IntelligenceType) -> bool {
        match (a, b) {
            (IntelligenceType::Swarm { .. }, IntelligenceType::Swarm { .. }) => true,
            (IntelligenceType::HiveMind { .. }, IntelligenceType::HiveMind { .. }) => true,
            _ => false,
        }
    }

    /// Shutdown current intelligence mode
    async fn shutdown_current_mode(&self, mode: &IntelligenceType) -> IntelligenceResult<()> {
        match mode {
            IntelligenceType::Swarm { .. } => {
                self.swarm_orchestrator.shutdown().await?;
            },
            IntelligenceType::HiveMind { .. } => {
                self.hive_mind_collective.shutdown().await?;
            }
        }
        Ok(())
    }

    /// Initialize new intelligence mode
    async fn initialize_mode(&self, mode: &IntelligenceType) -> IntelligenceResult<()> {
        match mode {
            IntelligenceType::Swarm { agent_count, divergence, collaboration } => {
                self.swarm_orchestrator.initialize(*agent_count, *divergence, *collaboration).await?;
            },
            IntelligenceType::HiveMind { collective_size, synchronization, unity } => {
                self.hive_mind_collective.initialize(*collective_size, *synchronization, *unity).await?;
            }
        }
        Ok(())
    }

    /// Update general metrics
    async fn update_metrics(&self, decision: &IntelligenceDecision) {
        let mut metrics = self.metrics.write().await;
        metrics.tasks_processed += 1;
        
        // Update rolling average confidence
        let total_confidence = metrics.avg_confidence * (metrics.tasks_processed - 1) as f32 + decision.confidence;
        metrics.avg_confidence = total_confidence / metrics.tasks_processed as f32;
    }

    /// Update performance metrics after task execution
    async fn update_performance_metrics(
        &self, 
        intelligence_type: &IntelligenceType,
        result: &HashMap<String, serde_json::Value>
    ) {
        let mut metrics = self.metrics.write().await;
        
        // Extract performance score from result (if available)
        let performance_score = result.get("performance_score")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5); // Default neutral score
        
        match intelligence_type {
            IntelligenceType::Swarm { .. } => {
                metrics.swarm_performance.push(performance_score);
                // Keep only last 100 measurements
                if metrics.swarm_performance.len() > 100 {
                    metrics.swarm_performance.remove(0);
                }
            },
            IntelligenceType::HiveMind { .. } => {
                metrics.hive_mind_performance.push(performance_score);
                // Keep only last 100 measurements
                if metrics.hive_mind_performance.len() > 100 {
                    metrics.hive_mind_performance.remove(0);
                }
            }
        }
    }

    /// Get current intelligence system metrics
    pub async fn get_metrics(&self) -> IntelligenceMetrics {
        self.metrics.read().await.clone()
    }

    /// Get current active intelligence mode
    pub async fn get_active_mode(&self) -> Option<IntelligenceType> {
        self.active_mode.read().await.clone()
    }

    /// Graceful shutdown of the intelligence system
    pub async fn shutdown(&self) -> IntelligenceResult<()> {
        if let Some(mode) = self.active_mode.read().await.as_ref() {
            self.shutdown_current_mode(mode).await?;
        }
        Ok(())
    }
}