//! Precision Policy Engine
//!
//! This module provides a policy-based system for making precision decisions,
//! allowing flexible and configurable precision management strategies.

use super::layer_precision::LayerPrecisionSpec;
use super::precision_manager::PrecisionContext;
use super::{ComponentType, LayerType, MixedPrecisionError, MixedPrecisionResult};
use crate::memory::tensor::BitNetDType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Precision policy defining rules for precision selection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PrecisionPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy priority (higher values take precedence)
    pub priority: u32,
    /// Whether the policy is enabled
    pub enabled: bool,
    /// Policy metadata
    pub metadata: HashMap<String, String>,
}

impl PrecisionPolicy {
    /// Create a new precision policy
    pub fn new(id: String, name: String, description: String) -> Self {
        Self {
            id,
            name,
            description,
            rules: Vec::new(),
            priority: 0,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Add a rule to the policy
    pub fn add_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set the policy priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add metadata to the policy
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if the policy applies to a given context
    pub fn applies_to(&self, context: &PolicyContext) -> bool {
        if !self.enabled {
            return false;
        }

        // Check if any rule applies
        self.rules.iter().any(|rule| rule.matches(context))
    }

    /// Apply the policy to determine precision
    pub fn apply(&self, context: &PolicyContext) -> Option<BitNetDType> {
        if !self.applies_to(context) {
            return None;
        }

        // Find the first matching rule and apply its action
        for rule in &self.rules {
            if rule.matches(context) {
                return rule.apply_action(context);
            }
        }

        None
    }
}

/// Policy rule defining conditions and actions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PolicyRule {
    /// Rule identifier
    pub id: String,
    /// Rule conditions
    pub conditions: Vec<PolicyCondition>,
    /// Rule action
    pub action: PolicyAction,
    /// Rule weight (for conflict resolution)
    pub weight: f32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

impl PolicyRule {
    /// Create a new policy rule
    pub fn new(id: String, action: PolicyAction) -> Self {
        Self {
            id,
            conditions: Vec::new(),
            action,
            weight: 1.0,
            enabled: true,
        }
    }

    /// Add a condition to the rule
    pub fn add_condition(mut self, condition: PolicyCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Set the rule weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Check if the rule matches the given context
    pub fn matches(&self, context: &PolicyContext) -> bool {
        if !self.enabled {
            return false;
        }

        // All conditions must be satisfied
        self.conditions
            .iter()
            .all(|condition| condition.evaluate(context))
    }

    /// Apply the rule action
    pub fn apply_action(&self, context: &PolicyContext) -> Option<BitNetDType> {
        self.action.execute(context)
    }
}

/// Policy condition for rule matching
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PolicyCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition operator
    pub operator: ConditionOperator,
    /// Condition value
    pub value: ConditionValue,
}

impl PolicyCondition {
    /// Create a new policy condition
    pub fn new(
        condition_type: ConditionType,
        operator: ConditionOperator,
        value: ConditionValue,
    ) -> Self {
        Self {
            condition_type,
            operator,
            value,
        }
    }

    /// Evaluate the condition against the context
    pub fn evaluate(&self, context: &PolicyContext) -> bool {
        let context_value = self.extract_context_value(context);
        self.operator.compare(&context_value, &self.value)
    }

    /// Extract the relevant value from the context
    fn extract_context_value(&self, context: &PolicyContext) -> ConditionValue {
        match &self.condition_type {
            ConditionType::LayerType => ConditionValue::LayerType(context.layer_type),
            ConditionType::ComponentType => ConditionValue::ComponentType(context.component_type),
            ConditionType::MemoryUsage => {
                ConditionValue::Float(context.memory_usage_mb.unwrap_or(0.0))
            }
            ConditionType::AccuracyRequirement => {
                ConditionValue::Float(context.accuracy_requirement.unwrap_or(1.0))
            }
            ConditionType::PerformanceRequirement => {
                ConditionValue::Float(context.performance_requirement.unwrap_or(1.0))
            }
            ConditionType::LayerName => {
                ConditionValue::String(context.layer_name.clone().unwrap_or_default())
            }
            ConditionType::Custom(key) => context
                .custom_attributes
                .get(key)
                .cloned()
                .unwrap_or(ConditionValue::String("".to_string())),
        }
    }
}

/// Types of conditions that can be evaluated
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionType {
    /// Layer type condition
    LayerType,
    /// Component type condition
    ComponentType,
    /// Memory usage condition
    MemoryUsage,
    /// Accuracy requirement condition
    AccuracyRequirement,
    /// Performance requirement condition
    PerformanceRequirement,
    /// Layer name condition
    LayerName,
    /// Custom condition
    Custom(String),
}

/// Operators for condition evaluation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal to
    LessThanOrEqual,
    /// Contains (for strings)
    Contains,
    /// Starts with (for strings)
    StartsWith,
    /// Ends with (for strings)
    EndsWith,
    /// In list
    In,
    /// Not in list
    NotIn,
}

impl ConditionOperator {
    /// Compare two condition values using this operator
    pub fn compare(&self, left: &ConditionValue, right: &ConditionValue) -> bool {
        match (left, right) {
            (ConditionValue::Float(l), ConditionValue::Float(r)) => match self {
                ConditionOperator::Equal => (l - r).abs() < f32::EPSILON,
                ConditionOperator::NotEqual => (l - r).abs() >= f32::EPSILON,
                ConditionOperator::GreaterThan => l > r,
                ConditionOperator::GreaterThanOrEqual => l >= r,
                ConditionOperator::LessThan => l < r,
                ConditionOperator::LessThanOrEqual => l <= r,
                _ => false,
            },
            (ConditionValue::String(l), ConditionValue::String(r)) => match self {
                ConditionOperator::Equal => l == r,
                ConditionOperator::NotEqual => l != r,
                ConditionOperator::Contains => l.contains(r),
                ConditionOperator::StartsWith => l.starts_with(r),
                ConditionOperator::EndsWith => l.ends_with(r),
                _ => false,
            },
            (ConditionValue::LayerType(l), ConditionValue::LayerType(r)) => match self {
                ConditionOperator::Equal => l == r,
                ConditionOperator::NotEqual => l != r,
                _ => false,
            },
            (ConditionValue::ComponentType(l), ConditionValue::ComponentType(r)) => match self {
                ConditionOperator::Equal => l == r,
                ConditionOperator::NotEqual => l != r,
                _ => false,
            },
            (ConditionValue::Precision(l), ConditionValue::Precision(r)) => match self {
                ConditionOperator::Equal => l == r,
                ConditionOperator::NotEqual => l != r,
                _ => false,
            },
            _ => false,
        }
    }
}

/// Values used in condition evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConditionValue {
    /// Float value
    Float(f32),
    /// String value
    String(String),
    /// Layer type value
    LayerType(LayerType),
    /// Component type value
    ComponentType(ComponentType),
    /// Precision value
    Precision(BitNetDType),
    /// List of values
    List(Vec<ConditionValue>),
}

/// Actions that can be taken when a rule matches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    /// Set a specific precision
    SetPrecision(BitNetDType),
    /// Use the default precision for the layer type
    UseLayerDefault,
    /// Use the default precision for the component type
    UseComponentDefault,
    /// Use the most memory-efficient precision
    UseMemoryEfficient,
    /// Use the highest accuracy precision
    UseHighAccuracy,
    /// Use a precision based on a formula
    UseFormula(PrecisionFormula),
    /// Delegate to another policy
    DelegateToPolicy(String),
}

impl PolicyAction {
    /// Execute the action and return the resulting precision
    pub fn execute(&self, context: &PolicyContext) -> Option<BitNetDType> {
        match self {
            PolicyAction::SetPrecision(precision) => Some(*precision),
            PolicyAction::UseLayerDefault => Some(context.layer_type.default_precision()),
            PolicyAction::UseComponentDefault => Some(context.component_type.default_precision()),
            PolicyAction::UseMemoryEfficient => {
                // Return the most memory-efficient precision for the layer type
                let alternatives = context.layer_type.precision_alternatives();
                alternatives.into_iter().max_by(|a, b| {
                    a.memory_efficiency()
                        .partial_cmp(&b.memory_efficiency())
                        .unwrap()
                })
            }
            PolicyAction::UseHighAccuracy => {
                // Return the highest precision supported by the layer type
                let alternatives = context.layer_type.precision_alternatives();
                alternatives
                    .into_iter()
                    .min_by_key(|p| p.bits_per_element())
            }
            PolicyAction::UseFormula(formula) => formula.evaluate(context),
            PolicyAction::DelegateToPolicy(_policy_id) => {
                // In a full implementation, this would delegate to another policy
                None
            }
        }
    }
}

/// Formula for calculating precision based on context
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PrecisionFormula {
    /// Formula type
    pub formula_type: FormulaType,
    /// Formula parameters
    pub parameters: HashMap<String, f32>,
}

impl PrecisionFormula {
    /// Evaluate the formula and return a precision
    pub fn evaluate(&self, context: &PolicyContext) -> Option<BitNetDType> {
        match self.formula_type {
            FormulaType::MemoryBudget => self.evaluate_memory_budget(context),
            FormulaType::AccuracyTarget => self.evaluate_accuracy_target(context),
            FormulaType::PerformanceTarget => self.evaluate_performance_target(context),
            FormulaType::Balanced => self.evaluate_balanced(context),
        }
    }

    /// Evaluate memory budget formula
    fn evaluate_memory_budget(&self, context: &PolicyContext) -> Option<BitNetDType> {
        let memory_budget = self.parameters.get("memory_budget")?;
        let current_usage = context.memory_usage_mb.unwrap_or(0.0);

        if current_usage > *memory_budget {
            // Use more aggressive quantization
            Some(BitNetDType::I4)
        } else {
            // Use balanced precision
            Some(BitNetDType::I8)
        }
    }

    /// Evaluate accuracy target formula
    fn evaluate_accuracy_target(&self, context: &PolicyContext) -> Option<BitNetDType> {
        let accuracy_target = self.parameters.get("accuracy_target")?;
        let current_accuracy = context.accuracy_requirement.unwrap_or(1.0);

        if current_accuracy < *accuracy_target {
            // Use higher precision
            Some(BitNetDType::F16)
        } else {
            // Use lower precision
            Some(BitNetDType::I8)
        }
    }

    /// Evaluate performance target formula
    fn evaluate_performance_target(&self, context: &PolicyContext) -> Option<BitNetDType> {
        let performance_target = self.parameters.get("performance_target")?;
        let current_performance = context.performance_requirement.unwrap_or(1.0);

        if current_performance < *performance_target {
            // Use lower precision for better performance
            Some(BitNetDType::I4)
        } else {
            // Use higher precision
            Some(BitNetDType::I8)
        }
    }

    /// Evaluate balanced formula
    fn evaluate_balanced(&self, context: &PolicyContext) -> Option<BitNetDType> {
        let memory_weight = self.parameters.get("memory_weight").unwrap_or(&0.33);
        let accuracy_weight = self.parameters.get("accuracy_weight").unwrap_or(&0.33);
        let performance_weight = self.parameters.get("performance_weight").unwrap_or(&0.34);

        // Calculate weighted score for different precisions
        let mut best_precision = BitNetDType::I8;
        let mut best_score = 0.0;

        for precision in context.layer_type.precision_alternatives() {
            let memory_score = precision.memory_efficiency() / 32.0; // Normalize to 0-1
            let accuracy_score = precision.bits_per_element() as f32 / 32.0; // Higher bits = higher accuracy
            let performance_score = 1.0 / precision.bits_per_element() as f32; // Lower bits = better performance

            let total_score = memory_score * memory_weight
                + accuracy_score * accuracy_weight
                + performance_score * performance_weight;

            if total_score > best_score {
                best_score = total_score;
                best_precision = precision;
            }
        }

        Some(best_precision)
    }
}

/// Types of precision formulas
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormulaType {
    /// Memory budget-based formula
    MemoryBudget,
    /// Accuracy target-based formula
    AccuracyTarget,
    /// Performance target-based formula
    PerformanceTarget,
    /// Balanced formula considering multiple factors
    Balanced,
}

/// Context for policy evaluation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PolicyContext {
    /// Layer type
    pub layer_type: LayerType,
    /// Component type
    pub component_type: ComponentType,
    /// Layer name
    pub layer_name: Option<String>,
    /// Current memory usage (MB)
    pub memory_usage_mb: Option<f32>,
    /// Accuracy requirement (0-1)
    pub accuracy_requirement: Option<f32>,
    /// Performance requirement (relative)
    pub performance_requirement: Option<f32>,
    /// Custom attributes
    pub custom_attributes: HashMap<String, ConditionValue>,
}

impl PolicyContext {
    /// Create a new policy context
    pub fn new(layer_type: LayerType, component_type: ComponentType) -> Self {
        Self {
            layer_type,
            component_type,
            layer_name: None,
            memory_usage_mb: None,
            accuracy_requirement: None,
            performance_requirement: None,
            custom_attributes: HashMap::new(),
        }
    }

    /// Set the layer name
    pub fn with_layer_name(mut self, name: String) -> Self {
        self.layer_name = Some(name);
        self
    }

    /// Set memory usage
    pub fn with_memory_usage(mut self, usage_mb: f32) -> Self {
        self.memory_usage_mb = Some(usage_mb);
        self
    }

    /// Set accuracy requirement
    pub fn with_accuracy_requirement(mut self, requirement: f32) -> Self {
        self.accuracy_requirement = Some(requirement);
        self
    }

    /// Set performance requirement
    pub fn with_performance_requirement(mut self, requirement: f32) -> Self {
        self.performance_requirement = Some(requirement);
        self
    }

    /// Add custom attribute
    pub fn with_custom_attribute(mut self, key: String, value: ConditionValue) -> Self {
        self.custom_attributes.insert(key, value);
        self
    }
}

/// Policy engine for managing and applying precision policies
#[derive(Debug)]
#[allow(dead_code)]
pub struct PolicyEngine {
    /// Registered policies
    policies: HashMap<String, PrecisionPolicy>,
    /// Policy application statistics
    stats: PolicyStats,
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new() -> Self {
        let mut engine = Self {
            policies: HashMap::new(),
            stats: PolicyStats::default(),
        };

        // Add default policies
        engine.add_default_policies();
        engine
    }

    /// Add default policies
    fn add_default_policies(&mut self) {
        // Memory-efficient policy
        let memory_policy = PrecisionPolicy::new(
            "memory_efficient".to_string(),
            "Memory Efficient".to_string(),
            "Prioritizes memory efficiency over accuracy".to_string(),
        )
        .with_priority(10)
        .add_rule(
            PolicyRule::new(
                "linear_memory_efficient".to_string(),
                PolicyAction::SetPrecision(BitNetDType::BitNet158),
            )
            .add_condition(PolicyCondition::new(
                ConditionType::LayerType,
                ConditionOperator::Equal,
                ConditionValue::LayerType(LayerType::Linear),
            )),
        );

        self.add_policy(memory_policy);

        // Accuracy-focused policy
        let accuracy_policy = PrecisionPolicy::new(
            "accuracy_focused".to_string(),
            "Accuracy Focused".to_string(),
            "Prioritizes accuracy over memory efficiency".to_string(),
        )
        .with_priority(20)
        .add_rule(
            PolicyRule::new(
                "attention_high_precision".to_string(),
                PolicyAction::SetPrecision(BitNetDType::F16),
            )
            .add_condition(PolicyCondition::new(
                ConditionType::LayerType,
                ConditionOperator::Equal,
                ConditionValue::LayerType(LayerType::Attention),
            )),
        );

        self.add_policy(accuracy_policy);
    }

    /// Add a policy to the engine
    pub fn add_policy(&mut self, policy: PrecisionPolicy) {
        self.policies.insert(policy.id.clone(), policy);
    }

    /// Remove a policy from the engine
    pub fn remove_policy(&mut self, policy_id: &str) {
        self.policies.remove(policy_id);
    }

    /// Apply policies to determine precision
    pub fn apply_policies(
        &mut self,
        current_precision: BitNetDType,
        layer_spec: &LayerPrecisionSpec,
        component_type: ComponentType,
        _context: &PrecisionContext,
    ) -> MixedPrecisionResult<BitNetDType> {
        let policy_context = PolicyContext::new(layer_spec.layer_type, component_type)
            .with_layer_name(layer_spec.layer_id.clone());

        // Collect applicable policies
        let mut applicable_policies: Vec<_> = self
            .policies
            .values()
            .filter(|policy| policy.applies_to(&policy_context))
            .collect();

        // Sort by priority (highest first)
        applicable_policies.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Apply the highest priority policy
        for policy in applicable_policies {
            if let Some(precision) = policy.apply(&policy_context) {
                self.stats.record_policy_application(&policy.id);
                return Ok(precision);
            }
        }

        // No policy applied, return current precision
        Ok(current_precision)
    }

    /// Get policy statistics
    pub fn get_stats(&self) -> &PolicyStats {
        &self.stats
    }

    /// Reset policy statistics
    pub fn reset_stats(&mut self) {
        self.stats = PolicyStats::default();
    }

    /// List all registered policies
    pub fn list_policies(&self) -> Vec<&PrecisionPolicy> {
        self.policies.values().collect()
    }

    /// Get a specific policy
    pub fn get_policy(&self, policy_id: &str) -> Option<&PrecisionPolicy> {
        self.policies.get(policy_id)
    }

    /// Enable or disable a policy
    pub fn set_policy_enabled(
        &mut self,
        policy_id: &str,
        enabled: bool,
    ) -> MixedPrecisionResult<()> {
        if let Some(policy) = self.policies.get_mut(policy_id) {
            policy.enabled = enabled;
            Ok(())
        } else {
            Err(MixedPrecisionError::InvalidConfiguration(format!(
                "Policy '{}' not found",
                policy_id
            )))
        }
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for policy applications
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct PolicyStats {
    /// Total policy applications
    pub total_applications: usize,
    /// Applications per policy
    pub applications_per_policy: HashMap<String, usize>,
}

impl PolicyStats {
    /// Record a policy application
    pub fn record_policy_application(&mut self, policy_id: &str) {
        self.total_applications += 1;
        *self
            .applications_per_policy
            .entry(policy_id.to_string())
            .or_insert(0) += 1;
    }

    /// Get the most frequently used policy
    pub fn most_used_policy(&self) -> Option<String> {
        self.applications_per_policy
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(policy_id, _)| policy_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_policy_creation() {
        let policy = PrecisionPolicy::new(
            "test_policy".to_string(),
            "Test Policy".to_string(),
            "A test policy".to_string(),
        );

        assert_eq!(policy.id, "test_policy");
        assert_eq!(policy.name, "Test Policy");
        assert!(policy.enabled);
        assert_eq!(policy.priority, 0);
    }

    #[test]
    fn test_policy_rule_creation() {
        let rule = PolicyRule::new(
            "test_rule".to_string(),
            PolicyAction::SetPrecision(BitNetDType::I8),
        );

        assert_eq!(rule.id, "test_rule");
        assert!(rule.enabled);
        assert_eq!(rule.weight, 1.0);
    }

    #[test]
    fn test_policy_condition() {
        let condition = PolicyCondition::new(
            ConditionType::LayerType,
            ConditionOperator::Equal,
            ConditionValue::LayerType(LayerType::Linear),
        );

        let context = PolicyContext::new(LayerType::Linear, ComponentType::Weights);
        assert!(condition.evaluate(&context));

        let context2 = PolicyContext::new(LayerType::Attention, ComponentType::Weights);
        assert!(!condition.evaluate(&context2));
    }

    #[test]
    fn test_condition_operators() {
        let op = ConditionOperator::GreaterThan;
        assert!(op.compare(&ConditionValue::Float(5.0), &ConditionValue::Float(3.0)));
        assert!(!op.compare(&ConditionValue::Float(3.0), &ConditionValue::Float(5.0)));

        let op = ConditionOperator::Contains;
        assert!(op.compare(
            &ConditionValue::String("hello world".to_string()),
            &ConditionValue::String("world".to_string())
        ));
    }

    #[test]
    fn test_policy_engine() {
        let engine = PolicyEngine::new();
        assert!(!engine.policies.is_empty());
        assert_eq!(engine.stats.total_applications, 0);
    }

    #[test]
    fn test_policy_context() {
        let context = PolicyContext::new(LayerType::Linear, ComponentType::Weights)
            .with_layer_name("test_layer".to_string())
            .with_memory_usage(100.0);

        assert_eq!(context.layer_type, LayerType::Linear);
        assert_eq!(context.layer_name, Some("test_layer".to_string()));
        assert_eq!(context.memory_usage_mb, Some(100.0));
    }

    #[test]
    fn test_precision_formula() {
        let mut parameters = HashMap::new();
        parameters.insert("memory_budget".to_string(), 50.0);

        let formula = PrecisionFormula {
            formula_type: FormulaType::MemoryBudget,
            parameters,
        };

        let context =
            PolicyContext::new(LayerType::Linear, ComponentType::Weights).with_memory_usage(100.0);

        let result = formula.evaluate(&context);
        assert!(result.is_some());
    }

    #[test]
    fn test_policy_stats() {
        let mut stats = PolicyStats::default();
        assert_eq!(stats.total_applications, 0);

        stats.record_policy_application("test_policy");
        assert_eq!(stats.total_applications, 1);
        assert_eq!(stats.applications_per_policy.get("test_policy"), Some(&1));

        stats.record_policy_application("test_policy");
        assert_eq!(stats.applications_per_policy.get("test_policy"), Some(&2));

        assert_eq!(stats.most_used_policy(), Some("test_policy".to_string()));
    }
}
