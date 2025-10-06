//! # BitNet Intelligence Architecture Foundation
//! 
//! Provides neural architecture detection and optimization for Docker BitNet Swarm Intelligence.
//! Determines whether tasks require swarm (diverging collaborative) or hive mind (unified collective) 
//! intelligence modes based on agent configuration patterns and task characteristics.

pub mod intelligence_manager;
pub mod swarm_orchestrator;
pub mod hive_mind_collective;
pub mod neural_architecture;
pub mod agent_config_extractor;
pub mod task_classifier;
pub mod consensus_engine;

pub use intelligence_manager::IntelligenceSystemManager;
pub use swarm_orchestrator::SwarmIntelligenceOrchestrator;
pub use hive_mind_collective::HiveMindCollective;
pub use neural_architecture::{SwarmArchitecture, HiveMindArchitecture, IntelligenceMode};
pub use agent_config_extractor::AgentConfigExtractor;
pub use task_classifier::TaskClassifier;
pub use consensus_engine::ConsensusEngine;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core intelligence types for Docker BitNet Swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntelligenceType {
    /// 🐝 Swarm: Diverging collaborative intelligence for parallel processing
    Swarm {
        /// Number of parallel agents
        agent_count: usize,
        /// Divergence factor (0.0-1.0)
        divergence: f32,
        /// Collaboration strength
        collaboration: f32,
    },
    /// 🧠 Hive Mind: Unified collective intelligence for synchronized processing
    HiveMind {
        /// Collective size
        collective_size: usize,
        /// Synchronization level (0.0-1.0)
        synchronization: f32,
        /// Unity factor
        unity: f32,
    },
}

/// Task characteristics for intelligence mode determination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCharacteristics {
    /// Complexity level (0.0-1.0)
    pub complexity: f32,
    /// Parallelization potential (0.0-1.0)
    pub parallelizable: f32,
    /// Synchronization requirement (0.0-1.0)
    pub sync_required: f32,
    /// Collaboration benefit (0.0-1.0)
    pub collaboration_benefit: f32,
    /// Unity requirement (0.0-1.0)
    pub unity_required: f32,
    /// Agent specialization needed
    pub agent_specializations: Vec<String>,
}

/// Intelligence detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceDecision {
    /// Recommended intelligence type
    pub intelligence_type: IntelligenceType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Agent configuration requirements
    pub agent_requirements: HashMap<String, f32>,
}

/// Error types for intelligence system
#[derive(Debug, thiserror::Error)]
pub enum IntelligenceError {
    #[error("Neural architecture error: {0}")]
    NeuralArchitecture(String),
    #[error("Agent configuration error: {0}")]
    AgentConfig(String),
    #[error("Task classification error: {0}")]
    TaskClassification(String),
    #[error("Consensus error: {0}")]
    Consensus(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for intelligence operations
pub type IntelligenceResult<T> = Result<T, IntelligenceError>;