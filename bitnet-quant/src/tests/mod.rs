/*!
Comprehensive testing and validation for BitNet's 1.58-bit quantization operations.

This module coordinates comprehensive testing across multiple dimensions:
- **Ternary Validation**: Verification that quantized values are strictly {-1, 0, +1}
- **Round-trip Accuracy**: Quantization → dequantization accuracy preservation
- **Scaling Factor Validation**: Mathematical correctness of scaling computations
- **Edge Cases**: Boundary conditions and extreme value handling
- **Property-based Testing**: Invariant checking and statistical validation
- **Memory Integration**: HybridMemoryPool integration and thread safety
- **Numerical Stability**: Robustness under challenging numerical conditions
- **Performance Regression**: Ensuring performance meets benchmarks

# Usage

```rust
use bitnet_quant::tests::helpers::*;
use bitnet_quant::tests::QuantizationTestSuite;

// Create test suite with default configuration
let suite = QuantizationTestSuite::new()?;
let results = suite.run_all_tests()?;

// Custom configuration
let config = QuantizationTestConfig {
    target_mse_threshold: 0.005, // Stricter MSE requirement
    enable_property_testing: true,
    property_test_iterations: 5000,
    ..Default::default()
};
let suite = QuantizationTestSuite::with_config(config)?;
```
*/

pub mod ternary_validation;
pub mod round_trip_accuracy;
pub mod scaling_factors;
pub mod edge_cases;
pub mod property_based;
pub mod memory_pool_integration;
pub mod numerical_stability;
pub mod performance_regression;
pub mod helpers;

// Re-export commonly used testing utilities
pub use helpers::{
    test_data::{TestPattern, generate_test_tensor, create_test_device},
    validation::{ValidationResult, TernaryValidationResult, RoundTripValidationResult},
    memory_helpers::{TestMemoryPool, MemoryTestHarness},
    statistical_analysis::{StatisticalAnalysis, QuantizationMetrics},
};

// Re-export test result types from individual modules
pub use ternary_validation::TernaryValidationResults;
pub use round_trip_accuracy::RoundTripResults;
pub use scaling_factors::ScalingFactorResults;
pub use edge_cases::EdgeCaseResults;
pub use property_based::PropertyTestResults;
pub use memory_pool_integration::MemoryPoolIntegrationResults;
pub use numerical_stability::NumericalStabilityResults;
pub use performance_regression::PerformanceRegressionResults;

use crate::quantization::{
    QuantizationError, TernaryMethod, WeightQuantizer,
    QuantizationResult, WeightQuantizationConfig
};
use candle_core::{Device, Tensor};

/// Test configuration for quantization validation
#[derive(Debug, Clone)]
pub struct QuantizationTestConfig {
    /// Target MSE threshold for round-trip accuracy
    pub target_mse_threshold: f64,
    /// Target SQNR threshold in dB
    pub target_sqnr_db: f64,
    /// Target cosine similarity threshold
    pub target_cosine_similarity: f64,
    /// Enable comprehensive edge case testing
    pub enable_edge_case_testing: bool,
    /// Enable property-based testing
    pub enable_property_testing: bool,
    /// Number of property-based test iterations
    pub property_test_iterations: usize,
    /// Enable memory pool integration testing
    pub enable_memory_integration: bool,
    /// Enable performance regression testing
    pub enable_performance_testing: bool,
    /// Enable numerical stability testing
    pub enable_numerical_stability: bool,
}

impl Default for QuantizationTestConfig {
    fn default() -> Self {
        Self {
            target_mse_threshold: 0.01,
            target_sqnr_db: 20.0,
            target_cosine_similarity: 0.95,
            enable_edge_case_testing: true,
            enable_property_testing: true,
            property_test_iterations: 1000,
            enable_memory_integration: true,
            enable_performance_testing: true,
            enable_numerical_stability: true,
        }
    }
}

/// Main test orchestrator for comprehensive quantization validation
pub struct QuantizationTestSuite {
    pub config: QuantizationTestConfig,
    pub device: Device,
}

impl QuantizationTestSuite {
    /// Create a new test suite with default configuration
    pub fn new() -> QuantizationResult<Self> {
        Ok(Self {
            config: QuantizationTestConfig::default(),
            device: create_test_device(),
        })
    }

    /// Create a new test suite with custom configuration
    pub fn with_config(config: QuantizationTestConfig) -> QuantizationResult<Self> {
        Ok(Self {
            config,
            device: create_test_device(),
        })
    }

    /// Run all configured tests
    pub fn run_all_tests(&self) -> QuantizationResult<TestSuiteResults> {
        let mut results = TestSuiteResults::new();

        // 1. Core ternary validation (always run)
        results.ternary_validation = self.run_ternary_validation_tests()?;

        // 2. Round-trip accuracy tests (always run)
        results.round_trip_accuracy = self.run_round_trip_tests()?;

        // 3. Scaling factor validation (always run)
        results.scaling_factor_validation = self.run_scaling_factor_tests()?;

        // 4. Edge case testing
        if self.config.enable_edge_case_testing {
            results.edge_case_results = Some(self.run_edge_case_tests()?);
        }

        // 5. Property-based testing
        if self.config.enable_property_testing {
            results.property_test_results = Some(self.run_property_tests()?);
        }

        // 6. Memory integration testing
        if self.config.enable_memory_integration {
            results.memory_integration_results = Some(self.run_memory_integration_tests()?);
        }

        // 7. Performance regression testing
        if self.config.enable_performance_testing {
            results.performance_results = Some(self.run_performance_tests()?);
        }

        // 8. Numerical stability testing
        if self.config.enable_numerical_stability {
            results.numerical_stability_results = Some(self.run_numerical_stability_tests()?);
        }

        Ok(results)
    }

    // Private test runners (implemented in respective modules)
    fn run_ternary_validation_tests(&self) -> QuantizationResult<TernaryValidationResults> {
        ternary_validation::run_comprehensive_ternary_tests(&self.device)
    }

    fn run_round_trip_tests(&self) -> QuantizationResult<RoundTripResults> {
        round_trip_accuracy::run_round_trip_tests(&self.device, self.config.target_mse_threshold)
    }

    fn run_scaling_factor_tests(&self) -> QuantizationResult<ScalingFactorResults> {
        scaling_factors::run_scaling_factor_tests(&self.device)
    }

    fn run_edge_case_tests(&self) -> QuantizationResult<EdgeCaseResults> {
        edge_cases::run_edge_case_tests(&self.device)
    }

    fn run_property_tests(&self) -> QuantizationResult<PropertyTestResults> {
        property_based::run_property_tests(&self.device, self.config.property_test_iterations)
    }

    fn run_memory_integration_tests(&self) -> QuantizationResult<MemoryPoolIntegrationResults> {
        memory_pool_integration::run_memory_pool_integration_tests(&self.device)
    }

    fn run_performance_tests(&self) -> QuantizationResult<PerformanceRegressionResults> {
        performance_regression::run_performance_regression_tests(&self.device)
    }

    fn run_numerical_stability_tests(&self) -> QuantizationResult<NumericalStabilityResults> {
        numerical_stability::run_numerical_stability_tests(&self.device)
    }
}

/// Comprehensive test suite results
#[derive(Debug)]
pub struct TestSuiteResults {
    pub ternary_validation: TernaryValidationResults,
    pub round_trip_accuracy: RoundTripResults,
    pub scaling_factor_validation: ScalingFactorResults,
    pub edge_case_results: Option<EdgeCaseResults>,
    pub property_test_results: Option<PropertyTestResults>,
    pub memory_integration_results: Option<MemoryPoolIntegrationResults>,
    pub performance_results: Option<PerformanceRegressionResults>,
    pub numerical_stability_results: Option<NumericalStabilityResults>,
}

impl TestSuiteResults {
    pub fn new() -> Self {
        Self {
            ternary_validation: TernaryValidationResults::default(),
            round_trip_accuracy: RoundTripResults::default(),
            scaling_factor_validation: ScalingFactorResults::default(),
            edge_case_results: None,
            property_test_results: None,
            memory_integration_results: None,
            performance_results: None,
            numerical_stability_results: None,
        }
    }

    /// Calculate overall success rate across all tests
    pub fn overall_success_rate(&self) -> f64 {
        let mut total_tests = 0;
        let mut passed_tests = 0;

        // Core tests (always present)
        let core_success_rate = (
            self.ternary_validation.overall_success_rate +
            self.round_trip_accuracy.overall_success_rate +
            self.scaling_factor_validation.overall_success_rate
        ) / 3.0;

        total_tests += 3;
        passed_tests += (core_success_rate * 3.0) as usize;

        // Optional tests
        if let Some(ref edge_results) = self.edge_case_results {
            total_tests += 1;
            if edge_results.overall_success_rate > 0.8 { passed_tests += 1; }
        }

        if let Some(ref prop_results) = self.property_test_results {
            total_tests += 1;
            if prop_results.success_rate() > 0.8 { passed_tests += 1; }
        }

        if let Some(ref mem_results) = self.memory_integration_results {
            total_tests += 1;
            if mem_results.success_rate() > 0.8 { passed_tests += 1; }
        }

        if let Some(ref perf_results) = self.performance_results {
            total_tests += 1;
            if perf_results.success_rate() > 0.8 { passed_tests += 1; }
        }

        if let Some(ref stab_results) = self.numerical_stability_results {
            total_tests += 1;
            if stab_results.success_rate() > 0.8 { passed_tests += 1; }
        }

        if total_tests > 0 {
            passed_tests as f64 / total_tests as f64
        } else {
            0.0
        }
    }

    /// Check if all critical tests pass
    pub fn all_critical_tests_pass(&self) -> bool {
        self.ternary_validation.overall_success_rate > 0.99 && // 99% ternary correctness
        self.round_trip_accuracy.overall_success_rate > 0.90 && // 90% round-trip accuracy
        self.scaling_factor_validation.overall_success_rate > 0.95 // 95% scaling factor correctness
    }

    /// Check for any critical issues across all test categories
    pub fn has_critical_issues(&self) -> bool {
        // Check property-based testing for critical issues
        if let Some(ref prop_results) = self.property_test_results {
            if !prop_results.invariant_violations.is_empty() {
                return true;
            }
        }

        // Check memory integration for critical issues
        if let Some(ref mem_results) = self.memory_integration_results {
            if mem_results.has_critical_issues() {
                return true;
            }
        }

        // Check performance for critical regressions
        if let Some(ref perf_results) = self.performance_results {
            if perf_results.has_critical_regressions() {
                return true;
            }
        }

        // Check numerical stability for critical instabilities
        if let Some(ref stab_results) = self.numerical_stability_results {
            if stab_results.has_critical_issues() {
                return true;
            }
        }

        // Check edge cases for critical failures
        if let Some(ref edge_results) = self.edge_case_results {
            if !edge_results.critical_failures.is_empty() {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suite_creation() {
        let suite = QuantizationTestSuite::new().unwrap();
        assert_eq!(suite.config.target_mse_threshold, 0.01);
        assert!(suite.config.enable_edge_case_testing);
    }

    #[test]
    fn test_custom_config() {
        let config = QuantizationTestConfig {
            target_mse_threshold: 0.005,
            enable_property_testing: false,
            enable_numerical_stability: false,
            ..Default::default()
        };
        let suite = QuantizationTestSuite::with_config(config).unwrap();
        assert_eq!(suite.config.target_mse_threshold, 0.005);
        assert!(!suite.config.enable_property_testing);
        assert!(!suite.config.enable_numerical_stability);
    }

    #[test]
    fn test_results_structure() {
        let results = TestSuiteResults::new();
        assert!(results.edge_case_results.is_none());
        assert!(results.property_test_results.is_none());
        assert!(results.memory_integration_results.is_none());
        assert!(results.performance_results.is_none());
        assert!(results.numerical_stability_results.is_none());
    }
}
