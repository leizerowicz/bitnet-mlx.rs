//! Precision Validation
//!
//! This module provides validation utilities for mixed precision configurations,
//! ensuring compatibility and correctness of precision settings.

use super::layer_precision::LayerPrecisionSpec;
use super::{LayerType, MixedPrecisionError, MixedPrecisionResult};
use crate::memory::tensor::BitNetDType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation rule for precision configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Types of validation rules
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Compatibility check between layer types and precisions
    LayerPrecisionCompatibility,
    /// Component precision compatibility
    ComponentPrecisionCompatibility,
    /// Memory usage validation
    MemoryUsage,
    /// Performance impact validation
    PerformanceImpact,
    /// Accuracy impact validation
    AccuracyImpact,
    /// Hardware compatibility validation
    HardwareCompatibility,
    /// Custom validation rule
    Custom(String),
}

/// Severity levels for validation issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information only
    Info,
    /// Warning - may cause issues
    Warning,
    /// Error - will cause issues
    Error,
    /// Critical - will cause failures
    Critical,
}

impl std::fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationSeverity::Info => write!(f, "Info"),
            ValidationSeverity::Warning => write!(f, "Warning"),
            ValidationSeverity::Error => write!(f, "Error"),
            ValidationSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Result of a validation check
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Validation issues found
    pub issues: Vec<ValidationIssue>,
    /// Warnings generated
    pub warnings: Vec<ValidationWarning>,
    /// Suggestions for improvement
    pub suggestions: Vec<ValidationSuggestion>,
}

/// Validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue identifier
    pub id: String,
    /// Issue description
    pub description: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Affected component
    pub component: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Affected component
    pub component: String,
    /// Potential impact
    pub impact: String,
}

/// Validation suggestion
#[derive(Debug, Clone)]
pub struct ValidationSuggestion {
    /// Suggestion description
    pub description: String,
    /// Expected benefit
    pub expected_benefit: String,
    /// Implementation difficulty
    pub difficulty: SuggestionDifficulty,
}

/// Difficulty levels for implementing suggestions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionDifficulty {
    /// Easy to implement
    Easy,
    /// Moderate effort required
    Moderate,
    /// Difficult to implement
    Difficult,
}

/// Precision validator
#[derive(Debug)]
pub struct PrecisionValidator {
    /// Validation rules
    rules: HashMap<String, ValidationRule>,
    /// Validation statistics
    stats: ValidationStats,
}

impl PrecisionValidator {
    /// Create a new precision validator
    pub fn new() -> Self {
        let mut validator = Self {
            rules: HashMap::new(),
            stats: ValidationStats::default(),
        };

        // Add default validation rules
        validator.add_default_rules();
        validator
    }

    /// Add default validation rules
    fn add_default_rules(&mut self) {
        // Layer-precision compatibility rule
        self.add_rule(ValidationRule {
            id: "layer_precision_compatibility".to_string(),
            description: "Validate that layer types support their assigned precisions".to_string(),
            rule_type: ValidationRuleType::LayerPrecisionCompatibility,
            severity: ValidationSeverity::Error,
            enabled: true,
        });

        // Component-precision compatibility rule
        self.add_rule(ValidationRule {
            id: "component_precision_compatibility".to_string(),
            description: "Validate that component types support their assigned precisions"
                .to_string(),
            rule_type: ValidationRuleType::ComponentPrecisionCompatibility,
            severity: ValidationSeverity::Error,
            enabled: true,
        });

        // Memory usage rule
        self.add_rule(ValidationRule {
            id: "memory_usage_check".to_string(),
            description: "Check for excessive memory usage with current precision settings"
                .to_string(),
            rule_type: ValidationRuleType::MemoryUsage,
            severity: ValidationSeverity::Warning,
            enabled: true,
        });

        // Performance impact rule
        self.add_rule(ValidationRule {
            id: "performance_impact_check".to_string(),
            description: "Assess performance impact of precision choices".to_string(),
            rule_type: ValidationRuleType::PerformanceImpact,
            severity: ValidationSeverity::Info,
            enabled: true,
        });

        // Accuracy impact rule
        self.add_rule(ValidationRule {
            id: "accuracy_impact_check".to_string(),
            description: "Assess potential accuracy impact of low precision settings".to_string(),
            rule_type: ValidationRuleType::AccuracyImpact,
            severity: ValidationSeverity::Warning,
            enabled: true,
        });
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Remove a validation rule
    pub fn remove_rule(&mut self, rule_id: &str) {
        self.rules.remove(rule_id);
    }

    /// Enable or disable a validation rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) -> MixedPrecisionResult<()> {
        if let Some(rule) = self.rules.get_mut(rule_id) {
            rule.enabled = enabled;
            Ok(())
        } else {
            Err(MixedPrecisionError::ValidationError(format!(
                "Validation rule '{}' not found",
                rule_id
            )))
        }
    }

    /// Validate a layer precision specification
    pub fn validate_layer_spec(
        &self,
        spec: &LayerPrecisionSpec,
    ) -> MixedPrecisionResult<ValidationResult> {
        let mut result = ValidationResult {
            passed: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        // Run all enabled validation rules
        for rule in self.rules.values() {
            if !rule.enabled {
                continue;
            }

            match rule.rule_type {
                ValidationRuleType::LayerPrecisionCompatibility => {
                    self.validate_layer_precision_compatibility(spec, &mut result);
                }
                ValidationRuleType::ComponentPrecisionCompatibility => {
                    self.validate_component_precision_compatibility(spec, &mut result);
                }
                ValidationRuleType::MemoryUsage => {
                    self.validate_memory_usage(spec, &mut result);
                }
                ValidationRuleType::PerformanceImpact => {
                    self.validate_performance_impact(spec, &mut result);
                }
                ValidationRuleType::AccuracyImpact => {
                    self.validate_accuracy_impact(spec, &mut result);
                }
                ValidationRuleType::HardwareCompatibility => {
                    self.validate_hardware_compatibility(spec, &mut result);
                }
                ValidationRuleType::Custom(_) => {
                    // Custom validation rules would be implemented here
                }
            }
        }

        // Update statistics
        self.update_stats(&result);

        Ok(result)
    }

    /// Validate layer-precision compatibility
    fn validate_layer_precision_compatibility(
        &self,
        spec: &LayerPrecisionSpec,
        result: &mut ValidationResult,
    ) {
        // Check if layer type supports the specified precisions
        if !spec.layer_type.supports_precision(spec.input_precision) {
            result.passed = false;
            result.issues.push(ValidationIssue {
                id: "incompatible_input_precision".to_string(),
                description: format!(
                    "Layer type {:?} does not support input precision {:?}",
                    spec.layer_type, spec.input_precision
                ),
                severity: ValidationSeverity::Error,
                component: spec.layer_id.clone(),
                suggested_fix: Some(format!(
                    "Use one of the supported precisions: {:?}",
                    spec.layer_type.precision_alternatives()
                )),
            });
        }

        if !spec.layer_type.supports_precision(spec.output_precision) {
            result.passed = false;
            result.issues.push(ValidationIssue {
                id: "incompatible_output_precision".to_string(),
                description: format!(
                    "Layer type {:?} does not support output precision {:?}",
                    spec.layer_type, spec.output_precision
                ),
                severity: ValidationSeverity::Error,
                component: spec.layer_id.clone(),
                suggested_fix: Some(format!(
                    "Use one of the supported precisions: {:?}",
                    spec.layer_type.precision_alternatives()
                )),
            });
        }

        if !spec.layer_type.supports_precision(spec.weight_precision) {
            result.passed = false;
            result.issues.push(ValidationIssue {
                id: "incompatible_weight_precision".to_string(),
                description: format!(
                    "Layer type {:?} does not support weight precision {:?}",
                    spec.layer_type, spec.weight_precision
                ),
                severity: ValidationSeverity::Error,
                component: spec.layer_id.clone(),
                suggested_fix: Some(format!(
                    "Use one of the supported precisions: {:?}",
                    spec.layer_type.precision_alternatives()
                )),
            });
        }
    }

    /// Validate component-precision compatibility
    fn validate_component_precision_compatibility(
        &self,
        spec: &LayerPrecisionSpec,
        result: &mut ValidationResult,
    ) {
        for (component_type, precision) in &spec.component_precisions {
            if !component_type.supports_precision(*precision) {
                result.passed = false;
                result.issues.push(ValidationIssue {
                    id: "incompatible_component_precision".to_string(),
                    description: format!(
                        "Component {:?} does not support precision {:?}",
                        component_type, precision
                    ),
                    severity: ValidationSeverity::Error,
                    component: format!("{}::{:?}", spec.layer_id, component_type),
                    suggested_fix: Some(format!(
                        "Use a precision supported by {:?}",
                        component_type
                    )),
                });
            }
        }
    }

    /// Validate memory usage
    fn validate_memory_usage(&self, spec: &LayerPrecisionSpec, result: &mut ValidationResult) {
        // Calculate memory efficiency
        let weight_efficiency = spec.weight_precision.memory_efficiency();
        let input_efficiency = spec.input_precision.memory_efficiency();
        let output_efficiency = spec.output_precision.memory_efficiency();

        // Check for potential memory issues
        if weight_efficiency < 2.0 && input_efficiency < 2.0 && output_efficiency < 2.0 {
            result.warnings.push(ValidationWarning {
                message: "Low memory efficiency - consider using more aggressive quantization"
                    .to_string(),
                component: spec.layer_id.clone(),
                impact: "Higher memory usage than necessary".to_string(),
            });

            result.suggestions.push(ValidationSuggestion {
                description: "Use lower precision for weights or activations".to_string(),
                expected_benefit: "Reduce memory usage by 50-75%".to_string(),
                difficulty: SuggestionDifficulty::Easy,
            });
        }

        // Check for very aggressive quantization
        if spec.weight_precision.bits_per_element() <= 2 {
            result.warnings.push(ValidationWarning {
                message: "Very aggressive weight quantization may impact accuracy".to_string(),
                component: spec.layer_id.clone(),
                impact: "Potential accuracy degradation".to_string(),
            });
        }
    }

    /// Validate performance impact
    fn validate_performance_impact(
        &self,
        spec: &LayerPrecisionSpec,
        result: &mut ValidationResult,
    ) {
        // Check for mixed precision that might hurt performance
        let precisions = vec![
            spec.input_precision,
            spec.output_precision,
            spec.weight_precision,
        ];

        let unique_precisions: std::collections::HashSet<_> = precisions.into_iter().collect();

        if unique_precisions.len() > 2 {
            result.warnings.push(ValidationWarning {
                message: "Multiple different precisions may increase conversion overhead"
                    .to_string(),
                component: spec.layer_id.clone(),
                impact: "Potential performance degradation due to conversions".to_string(),
            });

            result.suggestions.push(ValidationSuggestion {
                description: "Consider using fewer distinct precision levels".to_string(),
                expected_benefit: "Reduce conversion overhead".to_string(),
                difficulty: SuggestionDifficulty::Moderate,
            });
        }
    }

    /// Validate accuracy impact
    fn validate_accuracy_impact(&self, spec: &LayerPrecisionSpec, result: &mut ValidationResult) {
        // Check for precision choices that might significantly impact accuracy
        match spec.layer_type {
            LayerType::Attention => {
                if spec.weight_precision.bits_per_element() < 8 {
                    result.warnings.push(ValidationWarning {
                        message:
                            "Low precision in attention layers may significantly impact accuracy"
                                .to_string(),
                        component: spec.layer_id.clone(),
                        impact: "Potential significant accuracy degradation".to_string(),
                    });
                }
            }
            LayerType::Output => {
                if spec.output_precision.bits_per_element() < 8 {
                    result.warnings.push(ValidationWarning {
                        message: "Low precision in output layer may impact final accuracy"
                            .to_string(),
                        component: spec.layer_id.clone(),
                        impact: "Potential accuracy degradation in final predictions".to_string(),
                    });
                }
            }
            LayerType::Normalization => {
                if !spec.weight_precision.is_float() {
                    result.warnings.push(ValidationWarning {
                        message:
                            "Non-float precision in normalization layers may cause instability"
                                .to_string(),
                        component: spec.layer_id.clone(),
                        impact: "Potential training instability".to_string(),
                    });
                }
            }
            _ => {}
        }
    }

    /// Validate hardware compatibility
    fn validate_hardware_compatibility(
        &self,
        spec: &LayerPrecisionSpec,
        result: &mut ValidationResult,
    ) {
        // This would check against actual hardware capabilities
        // For now, we'll do basic checks

        // Check for very low precision that might not be well supported
        if spec.weight_precision == BitNetDType::I1 {
            result.warnings.push(ValidationWarning {
                message: "1-bit precision may not be well supported on all hardware".to_string(),
                component: spec.layer_id.clone(),
                impact: "Potential performance issues or fallback to software implementation"
                    .to_string(),
            });
        }
    }

    /// Update validation statistics
    fn update_stats(&self, _result: &ValidationResult) {
        // In a real implementation, this would update internal statistics
        // For now, we'll just count the validations
    }

    /// Get validation statistics
    pub fn get_stats(&self) -> &ValidationStats {
        &self.stats
    }

    /// Reset validation statistics
    pub fn reset_stats(&mut self) {
        self.stats = ValidationStats::default();
    }

    /// Validate multiple layer specifications
    pub fn validate_multiple_layers(
        &self,
        specs: &[LayerPrecisionSpec],
    ) -> MixedPrecisionResult<ValidationResult> {
        let mut combined_result = ValidationResult {
            passed: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        for spec in specs {
            let layer_result = self.validate_layer_spec(spec)?;

            if !layer_result.passed {
                combined_result.passed = false;
            }

            combined_result.issues.extend(layer_result.issues);
            combined_result.warnings.extend(layer_result.warnings);
            combined_result.suggestions.extend(layer_result.suggestions);
        }

        // Add cross-layer validation
        self.validate_cross_layer_compatibility(specs, &mut combined_result);

        Ok(combined_result)
    }

    /// Validate compatibility across multiple layers
    fn validate_cross_layer_compatibility(
        &self,
        specs: &[LayerPrecisionSpec],
        result: &mut ValidationResult,
    ) {
        // Check for precision mismatches between connected layers
        for i in 0..specs.len().saturating_sub(1) {
            let current_layer = &specs[i];
            let next_layer = &specs[i + 1];

            // Check if output precision of current layer matches input precision of next layer
            if current_layer.output_precision != next_layer.input_precision {
                result.warnings.push(ValidationWarning {
                    message: format!(
                        "Precision mismatch between layers {} and {} may require conversion",
                        current_layer.layer_id, next_layer.layer_id
                    ),
                    component: format!("{} -> {}", current_layer.layer_id, next_layer.layer_id),
                    impact: "Additional conversion overhead".to_string(),
                });
            }
        }

        // Check for overall precision distribution
        let mut precision_counts = HashMap::new();
        for spec in specs {
            *precision_counts.entry(spec.weight_precision).or_insert(0) += 1;
        }

        if precision_counts.len() > 4 {
            result.suggestions.push(ValidationSuggestion {
                description: "Consider reducing the number of different precisions used"
                    .to_string(),
                expected_benefit: "Simplified precision management and reduced conversion overhead"
                    .to_string(),
                difficulty: SuggestionDifficulty::Moderate,
            });
        }
    }
}

impl Default for PrecisionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation statistics
#[derive(Debug, Default)]
pub struct ValidationStats {
    /// Total validations performed
    pub total_validations: usize,
    /// Number of validations that passed
    pub passed_validations: usize,
    /// Number of validations that failed
    pub failed_validations: usize,
    /// Total issues found
    pub total_issues: usize,
    /// Total warnings generated
    pub total_warnings: usize,
    /// Total suggestions made
    pub total_suggestions: usize,
}

impl ValidationStats {
    /// Get the pass rate
    pub fn pass_rate(&self) -> f32 {
        if self.total_validations > 0 {
            self.passed_validations as f32 / self.total_validations as f32
        } else {
            0.0
        }
    }

    /// Get the average issues per validation
    pub fn average_issues_per_validation(&self) -> f32 {
        if self.total_validations > 0 {
            self.total_issues as f32 / self.total_validations as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_rule_creation() {
        let rule = ValidationRule {
            id: "test_rule".to_string(),
            description: "Test rule".to_string(),
            rule_type: ValidationRuleType::LayerPrecisionCompatibility,
            severity: ValidationSeverity::Error,
            enabled: true,
        };

        assert_eq!(rule.id, "test_rule");
        assert_eq!(rule.severity, ValidationSeverity::Error);
        assert!(rule.enabled);
    }

    #[test]
    fn test_validation_severity_ordering() {
        assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
        assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
        assert!(ValidationSeverity::Error < ValidationSeverity::Critical);
    }

    #[test]
    fn test_precision_validator_creation() {
        let validator = PrecisionValidator::new();
        assert!(!validator.rules.is_empty());
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            passed: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        assert!(result.passed);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_suggestion_difficulty() {
        assert_ne!(SuggestionDifficulty::Easy, SuggestionDifficulty::Difficult);
    }

    #[test]
    fn test_validation_stats() {
        let stats = ValidationStats::default();
        assert_eq!(stats.total_validations, 0);
        assert_eq!(stats.pass_rate(), 0.0);
        assert_eq!(stats.average_issues_per_validation(), 0.0);
    }

    #[test]
    fn test_rule_management() {
        let mut validator = PrecisionValidator::new();
        let initial_count = validator.rules.len();

        let new_rule = ValidationRule {
            id: "custom_rule".to_string(),
            description: "Custom test rule".to_string(),
            rule_type: ValidationRuleType::Custom("test".to_string()),
            severity: ValidationSeverity::Info,
            enabled: true,
        };

        validator.add_rule(new_rule);
        assert_eq!(validator.rules.len(), initial_count + 1);

        validator.remove_rule("custom_rule");
        assert_eq!(validator.rules.len(), initial_count);
    }
}
