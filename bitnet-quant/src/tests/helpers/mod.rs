//! Helper modules for testing utilities
//!
//! This module provides access to all helper utilities needed for
//! comprehensive quantization testing.

pub mod test_data;
pub mod validation;
pub mod memory_helpers;
pub mod statistical_analysis;

// Re-export commonly used items
pub use test_data::{
    TestPattern, generate_test_tensor, create_test_device,
    generate_test_tensor_set, TestTensorSet, DataStatistics
};

pub use validation::{
    TernaryValidationResult, RoundTripValidationResult, ScalingFactorValidationResult,
    ValidationResult, TernaryDistribution, QuantizationCounts,
    validate_ternary_values, validate_round_trip_accuracy, validate_scaling_factor,
    validate_shape_preservation, validate_value_bounds, compute_tensor_statistics,
    TensorStatistics
};

pub use memory_helpers::{
    MemoryTestHarness, TestMemoryPool, ConcurrentMemoryTestHarness,
    AllocationRecord, MemoryLeakReport, MemoryTestStatistics,
    ConcurrentTestResults
};

pub use statistical_analysis::{
    QuantizationMetrics, QualityThresholds, QualityAssessment, StatisticalAnalysis,
    ValueDistribution, ErrorDistributionAnalysis, OutlierAnalysis, StabilityAnalysis
};
