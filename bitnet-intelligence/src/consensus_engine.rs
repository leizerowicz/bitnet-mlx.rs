//! # Consensus Engine
//! 
//! Provides consensus mechanisms for validating intelligence mode decisions
//! and ensuring coherent operation across the Docker BitNet Swarm Intelligence system.

use crate::{IntelligenceResult, IntelligenceError, IntelligenceDecision, TaskCharacteristics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Consensus engine for intelligence decision validation
#[derive(Debug)]
pub struct ConsensusEngine {
    /// Consensus algorithms available
    algorithms: HashMap<String, ConsensusAlgorithm>,
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
    /// Consensus history
    consensus_history: Vec<ConsensusRecord>,
    /// Engine configuration
    config: ConsensusConfig,
}

/// Consensus algorithm implementations
#[derive(Debug, Clone)]
pub enum ConsensusAlgorithm {
    /// Simple majority consensus
    Majority {
        threshold: f32,
    },
    /// Byzantine fault tolerant consensus
    ByzantineFaultTolerant {
        fault_tolerance: usize,
        rounds: usize,
    },
    /// Practical Byzantine fault tolerance
    PBFT {
        view_number: usize,
        sequence_number: usize,
    },
    /// Weighted consensus based on agent reliability
    Weighted {
        weights: HashMap<String, f32>,
        threshold: f32,
    },
    /// Raft consensus for leader-based decisions
    Raft {
        term: usize,
        leader_id: Option<String>,
    },
}

/// Validation rule for consensus decisions
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Validation function
    pub validator: ValidationFunction,
    /// Rule priority (higher = more important)
    pub priority: usize,
    /// Rule weight in consensus calculation
    pub weight: f32,
}

/// Validation function types
#[derive(Debug, Clone)]
pub enum ValidationFunction {
    /// Validate intelligence type consistency
    IntelligenceTypeConsistency,
    /// Validate confidence threshold
    ConfidenceThreshold(f32),
    /// Validate agent requirements feasibility
    AgentRequirementsFeasibility,
    /// Validate task characteristics alignment
    TaskCharacteristicsAlignment,
    /// Custom validation logic
    Custom(fn(&IntelligenceDecision, &TaskCharacteristics) -> bool),
}

/// Consensus record for tracking decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRecord {
    /// Record identifier
    pub id: Uuid,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Original decision
    pub original_decision: String, // Serialized decision
    /// Consensus result
    pub consensus_result: ConsensusResult,
    /// Participating validators
    pub validators: Vec<String>,
    /// Consensus algorithm used
    pub algorithm: String,
    /// Validation time
    pub validation_duration: std::time::Duration,
}

/// Result of consensus validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusResult {
    /// Consensus achieved - decision approved
    Approved {
        confidence: f32,
        validator_count: usize,
    },
    /// Consensus achieved - decision rejected
    Rejected {
        reason: String,
        validator_count: usize,
    },
    /// No consensus reached
    NoConsensus {
        approval_rate: f32,
        timeout: bool,
    },
    /// Modified decision proposed
    Modified {
        modifications: Vec<String>,
        confidence: f32,
    },
}

/// Consensus engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Default consensus algorithm
    pub default_algorithm: String,
    /// Consensus timeout
    pub timeout: std::time::Duration,
    /// Minimum validator count
    pub min_validators: usize,
    /// Maximum validator count
    pub max_validators: usize,
    /// Confidence boost for consensus approval
    pub consensus_confidence_boost: f32,
    /// Confidence penalty for consensus rejection
    pub consensus_confidence_penalty: f32,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            default_algorithm: "majority".to_string(),
            timeout: std::time::Duration::from_secs(30),
            min_validators: 3,
            max_validators: 10,
            consensus_confidence_boost: 0.1,
            consensus_confidence_penalty: 0.2,
        }
    }
}

impl ConsensusEngine {
    /// Create new consensus engine
    pub async fn new() -> IntelligenceResult<Self> {
        let mut algorithms = HashMap::new();
        
        // Initialize default algorithms
        algorithms.insert("majority".to_string(), ConsensusAlgorithm::Majority { threshold: 0.6 });
        algorithms.insert("bft".to_string(), ConsensusAlgorithm::ByzantineFaultTolerant { 
            fault_tolerance: 1, 
            rounds: 3 
        });
        algorithms.insert("weighted".to_string(), ConsensusAlgorithm::Weighted { 
            weights: HashMap::new(), 
            threshold: 0.7 
        });
        
        let validation_rules = Self::create_default_validation_rules();
        
        Ok(Self {
            algorithms,
            validation_rules,
            consensus_history: Vec::new(),
            config: ConsensusConfig::default(),
        })
    }

    /// Validate decision using consensus mechanism
    pub async fn validate_decision(
        &self,
        decision: &IntelligenceDecision,
        characteristics: &TaskCharacteristics
    ) -> IntelligenceResult<IntelligenceDecision> {
        let start_time = std::time::Instant::now();
        
        // Run validation rules
        let validation_results = self.run_validation_rules(decision, characteristics).await?;
        
        // Apply consensus algorithm
        let consensus_result = self.apply_consensus_algorithm(&validation_results).await?;
        
        // Create validated decision based on consensus
        let validated_decision = self.create_validated_decision(decision, &consensus_result).await?;
        
        // Record consensus for learning
        self.record_consensus(decision, &consensus_result, start_time.elapsed()).await?;
        
        Ok(validated_decision)
    }

    /// Run all validation rules against the decision
    async fn run_validation_rules(
        &self,
        decision: &IntelligenceDecision,
        characteristics: &TaskCharacteristics
    ) -> IntelligenceResult<Vec<ValidationResult>> {
        let mut results = Vec::new();
        
        for rule in &self.validation_rules {
            let result = self.apply_validation_rule(rule, decision, characteristics).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Apply individual validation rule
    async fn apply_validation_rule(
        &self,
        rule: &ValidationRule,
        decision: &IntelligenceDecision,
        characteristics: &TaskCharacteristics
    ) -> IntelligenceResult<ValidationResult> {
        let is_valid = match &rule.validator {
            ValidationFunction::IntelligenceTypeConsistency => {
                self.validate_intelligence_type_consistency(decision, characteristics).await
            },
            ValidationFunction::ConfidenceThreshold(threshold) => {
                decision.confidence >= *threshold
            },
            ValidationFunction::AgentRequirementsFeasibility => {
                self.validate_agent_requirements_feasibility(decision).await
            },
            ValidationFunction::TaskCharacteristicsAlignment => {
                self.validate_task_characteristics_alignment(decision, characteristics).await
            },
            ValidationFunction::Custom(validator) => {
                validator(decision, characteristics)
            },
        };
        
        Ok(ValidationResult {
            rule_id: rule.id.clone(),
            is_valid,
            confidence: if is_valid { 1.0 } else { 0.0 },
            weight: rule.weight,
            reason: if is_valid {
                format!("Rule {} passed", rule.id)
            } else {
                format!("Rule {} failed", rule.id)
            },
        })
    }

    /// Validate intelligence type consistency
    async fn validate_intelligence_type_consistency(
        &self,
        decision: &IntelligenceDecision,
        characteristics: &TaskCharacteristics
    ) -> bool {
        use crate::IntelligenceType;
        
        match &decision.intelligence_type {
            IntelligenceType::Swarm { .. } => {
                // Swarm should be chosen for high parallelization, low sync requirements
                characteristics.parallelizable > 0.4 && characteristics.sync_required < 0.7
            },
            IntelligenceType::HiveMind { .. } => {
                // Hive mind should be chosen for high sync/unity requirements
                characteristics.sync_required > 0.4 || characteristics.unity_required > 0.4
            },
        }
    }

    /// Validate agent requirements feasibility
    async fn validate_agent_requirements_feasibility(&self, decision: &IntelligenceDecision) -> bool {
        // Check if agent requirements are reasonable
        let total_weight: f32 = decision.agent_requirements.values().sum();
        
        // Requirements should be balanced (no single agent > 80% weight)
        let max_weight = decision.agent_requirements.values()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&0.0);
        
        total_weight <= decision.agent_requirements.len() as f32 && *max_weight <= 0.8
    }

    /// Validate task characteristics alignment
    async fn validate_task_characteristics_alignment(
        &self,
        decision: &IntelligenceDecision,
        characteristics: &TaskCharacteristics
    ) -> bool {
        use crate::IntelligenceType;
        
        match &decision.intelligence_type {
            IntelligenceType::Swarm { divergence, collaboration, .. } => {
                // Swarm parameters should align with characteristics
                let divergence_alignment = (characteristics.parallelizable - divergence).abs() < 0.3;
                let collaboration_alignment = (characteristics.collaboration_benefit - collaboration).abs() < 0.3;
                divergence_alignment && collaboration_alignment
            },
            IntelligenceType::HiveMind { synchronization, unity, .. } => {
                // Hive mind parameters should align with characteristics
                let sync_alignment = (characteristics.sync_required - synchronization).abs() < 0.3;
                let unity_alignment = (characteristics.unity_required - unity).abs() < 0.3;
                sync_alignment && unity_alignment
            },
        }
    }

    /// Apply consensus algorithm to validation results
    async fn apply_consensus_algorithm(
        &self,
        validation_results: &[ValidationResult]
    ) -> IntelligenceResult<ConsensusResult> {
        let algorithm = self.algorithms.get(&self.config.default_algorithm)
            .ok_or_else(|| IntelligenceError::Consensus("Default algorithm not found".to_string()))?;
        
        match algorithm {
            ConsensusAlgorithm::Majority { threshold } => {
                self.apply_majority_consensus(validation_results, *threshold).await
            },
            ConsensusAlgorithm::ByzantineFaultTolerant { fault_tolerance, rounds } => {
                self.apply_bft_consensus(validation_results, *fault_tolerance, *rounds).await
            },
            ConsensusAlgorithm::Weighted { weights: _, threshold } => {
                self.apply_weighted_consensus(validation_results, *threshold).await
            },
            _ => {
                // Fallback to majority consensus
                self.apply_majority_consensus(validation_results, 0.6).await
            }
        }
    }

    /// Apply majority consensus algorithm
    async fn apply_majority_consensus(
        &self,
        validation_results: &[ValidationResult],
        threshold: f32
    ) -> IntelligenceResult<ConsensusResult> {
        let valid_count = validation_results.iter().filter(|r| r.is_valid).count();
        let total_count = validation_results.len();
        
        if total_count == 0 {
            return Ok(ConsensusResult::NoConsensus { approval_rate: 0.0, timeout: false });
        }
        
        let approval_rate = valid_count as f32 / total_count as f32;
        
        if approval_rate >= threshold {
            Ok(ConsensusResult::Approved {
                confidence: approval_rate,
                validator_count: total_count,
            })
        } else {
            Ok(ConsensusResult::Rejected {
                reason: format!("Approval rate {:.2} below threshold {:.2}", approval_rate, threshold),
                validator_count: total_count,
            })
        }
    }

    /// Apply Byzantine fault tolerant consensus
    async fn apply_bft_consensus(
        &self,
        validation_results: &[ValidationResult],
        _fault_tolerance: usize,
        _rounds: usize
    ) -> IntelligenceResult<ConsensusResult> {
        // Simplified BFT implementation
        // In practice, this would involve multiple rounds of voting
        
        let valid_count = validation_results.iter().filter(|r| r.is_valid).count();
        let total_count = validation_results.len();
        
        // BFT requires 2/3 + 1 consensus
        let required_count = (total_count * 2 / 3) + 1;
        
        if valid_count >= required_count {
            Ok(ConsensusResult::Approved {
                confidence: valid_count as f32 / total_count as f32,
                validator_count: total_count,
            })
        } else {
            Ok(ConsensusResult::Rejected {
                reason: format!("BFT consensus failed: {} valid out of {} required", valid_count, required_count),
                validator_count: total_count,
            })
        }
    }

    /// Apply weighted consensus algorithm
    async fn apply_weighted_consensus(
        &self,
        validation_results: &[ValidationResult],
        threshold: f32
    ) -> IntelligenceResult<ConsensusResult> {
        let total_weight: f32 = validation_results.iter().map(|r| r.weight).sum();
        let valid_weight: f32 = validation_results.iter()
            .filter(|r| r.is_valid)
            .map(|r| r.weight)
            .sum();
        
        if total_weight == 0.0 {
            return Ok(ConsensusResult::NoConsensus { approval_rate: 0.0, timeout: false });
        }
        
        let weighted_approval = valid_weight / total_weight;
        
        if weighted_approval >= threshold {
            Ok(ConsensusResult::Approved {
                confidence: weighted_approval,
                validator_count: validation_results.len(),
            })
        } else {
            Ok(ConsensusResult::Rejected {
                reason: format!("Weighted approval {:.2} below threshold {:.2}", weighted_approval, threshold),
                validator_count: validation_results.len(),
            })
        }
    }

    /// Create validated decision based on consensus result
    async fn create_validated_decision(
        &self,
        original_decision: &IntelligenceDecision,
        consensus_result: &ConsensusResult
    ) -> IntelligenceResult<IntelligenceDecision> {
        let mut validated_decision = original_decision.clone();
        
        match consensus_result {
            ConsensusResult::Approved { confidence, .. } => {
                // Boost confidence for consensus approval
                validated_decision.confidence = (original_decision.confidence + 
                    self.config.consensus_confidence_boost).min(1.0);
                validated_decision.reasoning = format!(
                    "{} [Consensus: Approved with {:.2} confidence]",
                    original_decision.reasoning,
                    confidence
                );
            },
            ConsensusResult::Rejected { reason, .. } => {
                // Penalize confidence for consensus rejection
                validated_decision.confidence = (original_decision.confidence - 
                    self.config.consensus_confidence_penalty).max(0.0);
                validated_decision.reasoning = format!(
                    "{} [Consensus: Rejected - {}]",
                    original_decision.reasoning,
                    reason
                );
            },
            ConsensusResult::NoConsensus { approval_rate, .. } => {
                validated_decision.confidence = original_decision.confidence * approval_rate;
                validated_decision.reasoning = format!(
                    "{} [Consensus: No consensus, {:.2} approval rate]",
                    original_decision.reasoning,
                    approval_rate
                );
            },
            ConsensusResult::Modified { modifications, confidence } => {
                validated_decision.confidence = *confidence;
                validated_decision.reasoning = format!(
                    "{} [Consensus: Modified - {}]",
                    original_decision.reasoning,
                    modifications.join(", ")
                );
            },
        }
        
        Ok(validated_decision)
    }

    /// Record consensus result for learning
    async fn record_consensus(
        &self,
        decision: &IntelligenceDecision,
        consensus_result: &ConsensusResult,
        duration: std::time::Duration
    ) -> IntelligenceResult<()> {
        let record = ConsensusRecord {
            id: Uuid::new_v4(),
            timestamp: std::time::SystemTime::now(),
            original_decision: serde_json::to_string(decision)
                .map_err(|e| IntelligenceError::Serialization(e))?,
            consensus_result: consensus_result.clone(),
            validators: self.validation_rules.iter().map(|r| r.id.clone()).collect(),
            algorithm: self.config.default_algorithm.clone(),
            validation_duration: duration,
        };
        
        // Note: In a real implementation, this would persist to storage
        // For now, we just acknowledge the recording
        
        Ok(())
    }

    /// Create default validation rules
    fn create_default_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                id: "intelligence_type_consistency".to_string(),
                description: "Validate that intelligence type matches task characteristics".to_string(),
                validator: ValidationFunction::IntelligenceTypeConsistency,
                priority: 1,
                weight: 0.3,
            },
            ValidationRule {
                id: "confidence_threshold".to_string(),
                description: "Validate minimum confidence threshold".to_string(),
                validator: ValidationFunction::ConfidenceThreshold(0.4),
                priority: 2,
                weight: 0.2,
            },
            ValidationRule {
                id: "agent_requirements_feasibility".to_string(),
                description: "Validate that agent requirements are feasible".to_string(),
                validator: ValidationFunction::AgentRequirementsFeasibility,
                priority: 3,
                weight: 0.25,
            },
            ValidationRule {
                id: "task_characteristics_alignment".to_string(),
                description: "Validate alignment between decision and task characteristics".to_string(),
                validator: ValidationFunction::TaskCharacteristicsAlignment,
                priority: 1,
                weight: 0.25,
            },
        ]
    }

    /// Get consensus history
    pub fn get_consensus_history(&self) -> &[ConsensusRecord] {
        &self.consensus_history
    }

    /// Get consensus configuration
    pub fn get_config(&self) -> &ConsensusConfig {
        &self.config
    }

    /// Update consensus configuration
    pub async fn update_config(&mut self, config: ConsensusConfig) -> IntelligenceResult<()> {
        self.config = config;
        Ok(())
    }
}

/// Validation result for a single rule
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Rule identifier
    pub rule_id: String,
    /// Whether validation passed
    pub is_valid: bool,
    /// Confidence in validation result
    pub confidence: f32,
    /// Weight of this validation in consensus
    pub weight: f32,
    /// Reason for validation result
    pub reason: String,
}