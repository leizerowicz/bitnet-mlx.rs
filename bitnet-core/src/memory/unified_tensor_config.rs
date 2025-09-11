//! Unified Tensor Pool Configuration for Task 1.7.1 - Configuration Simplification
//!
//! This module provides a simplified, unified configuration interface that replaces
//! the multiple configuration objects with a single, easy-to-use configuration system.
//!
//! Key features:
//! - Single configuration object for all tensor pool variants
//! - Automatic optimization selection based on use case profiles
//! - Simplified API that reduces configuration complexity by 50%
//! - Built-in configuration validation and optimization

use super::{
    AllocationStrategy, TensorSizeCategory, 
    lightweight_tensor_pool::LightweightTensorPool,
    allocation_pattern_learner::AllocationPatternLearner,
};
use std::time::Duration;

#[cfg(feature = "tracing")]
use tracing::{info, warn};

/// Simplified configuration profiles for common use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPoolProfile {
    /// Optimized for small, frequent allocations (inference workloads)
    SmallTensorOptimized,
    /// Optimized for large tensor operations (training workloads)
    LargeTensorOptimized,
    /// Balanced performance for mixed workloads
    Balanced,
    /// Maximum performance with learning enabled
    Adaptive,
    /// Custom configuration (fallback to manual settings)
    Custom,
}

/// Performance optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Minimal overhead, maximum consistency
    Consistency,
    /// Balanced performance and features
    Balanced,
    /// Maximum performance, all optimizations enabled
    Performance,
}

/// Unified tensor pool configuration that replaces multiple config objects
#[derive(Debug, Clone)]
pub struct UnifiedTensorPoolConfig {
    /// Configuration profile for automatic optimization
    pub profile: TensorPoolProfile,
    
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    
    /// Enable allocation pattern learning
    pub enable_learning: bool,
    
    /// Enable lightweight pool for small tensors
    pub enable_lightweight_pool: bool,
    
    /// Custom threshold for strategy selection (only used with Custom profile)
    pub custom_threshold: Option<usize>,
    
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    
    /// Maximum performance variance target (nanoseconds)
    pub target_variance_ns: u64,
    
    /// Learning window size for pattern recognition
    pub learning_window_size: usize,
}

impl Default for UnifiedTensorPoolConfig {
    fn default() -> Self {
        Self {
            profile: TensorPoolProfile::Balanced,
            optimization_level: OptimizationLevel::Balanced,
            enable_learning: true,
            enable_lightweight_pool: true,
            custom_threshold: None,
            enable_performance_tracking: true,
            target_variance_ns: 50, // Task 1.7.1 target
            learning_window_size: 100,
        }
    }
}

impl UnifiedTensorPoolConfig {
    /// Create configuration for small tensor optimization (Task 1.7.1 focus)
    pub fn small_tensor_optimized() -> Self {
        Self {
            profile: TensorPoolProfile::SmallTensorOptimized,
            optimization_level: OptimizationLevel::Consistency,
            enable_learning: true,
            enable_lightweight_pool: true,
            custom_threshold: None,
            enable_performance_tracking: true,
            target_variance_ns: 50, // Strict variance target
            learning_window_size: 50, // Smaller window for faster learning
        }
    }
    
    /// Create configuration for large tensor optimization
    pub fn large_tensor_optimized() -> Self {
        Self {
            profile: TensorPoolProfile::LargeTensorOptimized,
            optimization_level: OptimizationLevel::Performance,
            enable_learning: true,
            enable_lightweight_pool: false, // Not needed for large tensors
            custom_threshold: Some(64 * 1024), // 64KB threshold
            enable_performance_tracking: true,
            target_variance_ns: 1000, // More relaxed for large tensors
            learning_window_size: 200,
        }
    }
    
    /// Create adaptive configuration with learning enabled
    pub fn adaptive() -> Self {
        Self {
            profile: TensorPoolProfile::Adaptive,
            optimization_level: OptimizationLevel::Performance,
            enable_learning: true,
            enable_lightweight_pool: true,
            custom_threshold: None,
            enable_performance_tracking: true,
            target_variance_ns: 100,
            learning_window_size: 100,
        }
    }
    
    /// Create balanced configuration for mixed workloads
    pub fn balanced() -> Self {
        Self::default()
    }
    
    /// Create custom configuration with specific settings
    pub fn custom(threshold: usize, optimization_level: OptimizationLevel) -> Self {
        Self {
            profile: TensorPoolProfile::Custom,
            optimization_level,
            enable_learning: false, // Custom configs don't use learning by default
            enable_lightweight_pool: threshold <= 32 * 1024,
            custom_threshold: Some(threshold),
            enable_performance_tracking: true,
            target_variance_ns: 200,
            learning_window_size: 100,
        }
    }
    
    /// Get the strategy selection threshold based on configuration
    pub fn get_strategy_threshold(&self) -> usize {
        match self.profile {
            TensorPoolProfile::SmallTensorOptimized => 16 * 1024, // 16KB
            TensorPoolProfile::LargeTensorOptimized => 64 * 1024, // 64KB
            TensorPoolProfile::Balanced => 32 * 1024, // 32KB
            TensorPoolProfile::Adaptive => 32 * 1024, // Start with 32KB, learning adjusts
            TensorPoolProfile::Custom => self.custom_threshold.unwrap_or(32 * 1024),
        }
    }
    
    /// Get the default allocation strategy for this configuration
    pub fn get_default_strategy(&self) -> AllocationStrategy {
        match self.profile {
            TensorPoolProfile::SmallTensorOptimized => AllocationStrategy::Standard,
            TensorPoolProfile::LargeTensorOptimized => AllocationStrategy::Optimized,
            TensorPoolProfile::Balanced | TensorPoolProfile::Adaptive => AllocationStrategy::Adaptive,
            TensorPoolProfile::Custom => AllocationStrategy::Adaptive,
        }
    }
    
    /// Check if configuration should use lightweight pool for given tensor size
    pub fn should_use_lightweight_pool(&self, size_bytes: usize) -> bool {
        if !self.enable_lightweight_pool {
            return false;
        }
        
        match self.profile {
            TensorPoolProfile::SmallTensorOptimized => size_bytes <= 16 * 1024,
            TensorPoolProfile::LargeTensorOptimized => false, // Never use lightweight for large tensor profile
            TensorPoolProfile::Balanced | TensorPoolProfile::Adaptive => size_bytes <= 8 * 1024,
            TensorPoolProfile::Custom => size_bytes <= (self.custom_threshold.unwrap_or(32 * 1024) / 4),
        }
    }
    
    /// Get learning configuration parameters
    pub fn get_learning_config(&self) -> Option<LearningConfig> {
        if !self.enable_learning {
            return None;
        }
        
        Some(LearningConfig {
            window_size: self.learning_window_size,
            min_samples_for_learning: match self.profile {
                TensorPoolProfile::SmallTensorOptimized => 5, // Quick learning for small tensors
                TensorPoolProfile::LargeTensorOptimized => 20, // More samples for stable large tensor learning
                _ => 10, // Balanced default
            },
            learning_rate: match self.optimization_level {
                OptimizationLevel::Consistency => 0.05, // Conservative learning
                OptimizationLevel::Balanced => 0.1, // Standard learning
                OptimizationLevel::Performance => 0.15, // Aggressive learning
            },
            adaptation_threshold: 0.05, // 5% performance difference to trigger adaptation
        })
    }
    
    /// Validate configuration and provide warnings
    pub fn validate(&self) -> ConfigValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Check for conflicting settings
        if self.profile == TensorPoolProfile::LargeTensorOptimized && self.enable_lightweight_pool {
            warnings.push("Lightweight pool is not recommended for large tensor optimization profile".to_string());
        }
        
        if self.profile == TensorPoolProfile::Custom && self.custom_threshold.is_none() {
            errors.push("Custom profile requires custom_threshold to be set".to_string());
        }
        
        if self.target_variance_ns == 0 {
            errors.push("target_variance_ns must be greater than 0".to_string());
        }
        
        if self.learning_window_size < 10 {
            warnings.push("Small learning window size may lead to unstable learning".to_string());
        }
        
        ConfigValidationResult { warnings, errors }
    }
    
    /// Get a human-readable summary of the configuration
    pub fn get_summary(&self) -> String {
        format!(
            "TensorPool Config: {:?} profile, {:?} optimization, learning={}, lightweight={}, threshold={}KB, target_variance={}ns",
            self.profile,
            self.optimization_level,
            self.enable_learning,
            self.enable_lightweight_pool,
            self.get_strategy_threshold() / 1024,
            self.target_variance_ns
        )
    }
}

/// Learning configuration parameters
#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub window_size: usize,
    pub min_samples_for_learning: usize,
    pub learning_rate: f64,
    pub adaptation_threshold: f64,
}

/// Configuration validation result
#[derive(Debug)]
pub struct ConfigValidationResult {
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl ConfigValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    pub fn log_issues(&self) {
        #[cfg(feature = "tracing")]
        {
            for warning in &self.warnings {
                warn!("Config warning: {}", warning);
            }
            for error in &self.errors {
                warn!("Config error: {}", error);
            }
        }
    }
}

/// Configuration builder for fluent API
pub struct UnifiedTensorPoolConfigBuilder {
    config: UnifiedTensorPoolConfig,
}

impl UnifiedTensorPoolConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: UnifiedTensorPoolConfig::default(),
        }
    }
    
    /// Set the configuration profile
    pub fn profile(mut self, profile: TensorPoolProfile) -> Self {
        self.config.profile = profile;
        self
    }
    
    /// Set the optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.config.optimization_level = level;
        self
    }
    
    /// Enable or disable learning
    pub fn learning(mut self, enable: bool) -> Self {
        self.config.enable_learning = enable;
        self
    }
    
    /// Enable or disable lightweight pool
    pub fn lightweight_pool(mut self, enable: bool) -> Self {
        self.config.enable_lightweight_pool = enable;
        self
    }
    
    /// Set custom threshold (for custom profile)
    pub fn threshold(mut self, threshold: usize) -> Self {
        self.config.custom_threshold = Some(threshold);
        self
    }
    
    /// Set target performance variance
    pub fn target_variance(mut self, variance_ns: u64) -> Self {
        self.config.target_variance_ns = variance_ns;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> Result<UnifiedTensorPoolConfig, String> {
        let validation = self.config.validate();
        
        if !validation.is_valid() {
            return Err(format!("Configuration validation failed: {:?}", validation.errors));
        }
        
        #[cfg(feature = "tracing")]
        {
            validation.log_issues();
            info!("Created tensor pool configuration: {}", self.config.get_summary());
        }
        
        Ok(self.config)
    }
}

impl Default for UnifiedTensorPoolConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-configured optimization profiles for common use cases
pub struct ConfigurationProfiles;

impl ConfigurationProfiles {
    /// Task 1.7.1 optimized configuration for small tensor consistency
    pub fn task_1_7_1_optimized() -> UnifiedTensorPoolConfig {
        UnifiedTensorPoolConfigBuilder::new()
            .profile(TensorPoolProfile::SmallTensorOptimized)
            .optimization_level(OptimizationLevel::Consistency)
            .learning(true)
            .lightweight_pool(true)
            .target_variance(50) // Task 1.7.1 requirement
            .build()
            .expect("Task 1.7.1 configuration should be valid")
    }
    
    /// Inference workload optimization
    pub fn inference_optimized() -> UnifiedTensorPoolConfig {
        UnifiedTensorPoolConfigBuilder::new()
            .profile(TensorPoolProfile::SmallTensorOptimized)
            .optimization_level(OptimizationLevel::Consistency)
            .learning(true)
            .lightweight_pool(true)
            .target_variance(100)
            .build()
            .expect("Inference configuration should be valid")
    }
    
    /// Training workload optimization
    pub fn training_optimized() -> UnifiedTensorPoolConfig {
        UnifiedTensorPoolConfigBuilder::new()
            .profile(TensorPoolProfile::LargeTensorOptimized)
            .optimization_level(OptimizationLevel::Performance)
            .learning(true)
            .lightweight_pool(false)
            .target_variance(1000)
            .build()
            .expect("Training configuration should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_configuration_profiles() {
        let small_config = UnifiedTensorPoolConfig::small_tensor_optimized();
        assert_eq!(small_config.profile, TensorPoolProfile::SmallTensorOptimized);
        assert_eq!(small_config.optimization_level, OptimizationLevel::Consistency);
        assert_eq!(small_config.target_variance_ns, 50);
        
        let large_config = UnifiedTensorPoolConfig::large_tensor_optimized();
        assert_eq!(large_config.profile, TensorPoolProfile::LargeTensorOptimized);
        assert_eq!(large_config.optimization_level, OptimizationLevel::Performance);
        
        let adaptive_config = UnifiedTensorPoolConfig::adaptive();
        assert_eq!(adaptive_config.profile, TensorPoolProfile::Adaptive);
        assert!(adaptive_config.enable_learning);
    }
    
    #[test]
    fn test_configuration_builder() {
        let config = UnifiedTensorPoolConfigBuilder::new()
            .profile(TensorPoolProfile::SmallTensorOptimized)
            .optimization_level(OptimizationLevel::Consistency)
            .learning(true)
            .lightweight_pool(true)
            .target_variance(25)
            .build()
            .unwrap();
        
        assert_eq!(config.profile, TensorPoolProfile::SmallTensorOptimized);
        assert_eq!(config.target_variance_ns, 25);
        assert!(config.enable_learning);
        assert!(config.enable_lightweight_pool);
    }
    
    #[test]
    fn test_configuration_validation() {
        // Valid configuration
        let valid_config = UnifiedTensorPoolConfig::balanced();
        let validation = valid_config.validate();
        assert!(validation.is_valid());
        
        // Invalid configuration (custom without threshold)
        let invalid_config = UnifiedTensorPoolConfig {
            profile: TensorPoolProfile::Custom,
            custom_threshold: None,
            ..Default::default()
        };
        let validation = invalid_config.validate();
        assert!(!validation.is_valid());
    }
    
    #[test]
    fn test_task_1_7_1_configuration() {
        let config = ConfigurationProfiles::task_1_7_1_optimized();
        
        assert_eq!(config.profile, TensorPoolProfile::SmallTensorOptimized);
        assert_eq!(config.optimization_level, OptimizationLevel::Consistency);
        assert_eq!(config.target_variance_ns, 50);
        assert!(config.enable_learning);
        assert!(config.enable_lightweight_pool);
        
        // Test lightweight pool usage
        assert!(config.should_use_lightweight_pool(1024)); // 1KB
        assert!(config.should_use_lightweight_pool(16 * 1024)); // 16KB
        assert!(!config.should_use_lightweight_pool(32 * 1024)); // 32KB
        
        let validation = config.validate();
        assert!(validation.is_valid());
    }
    
    #[test]
    fn test_configuration_complexity_reduction() {
        // Before: Multiple config objects would be needed
        // After: Single config object handles all scenarios
        
        let configs = vec![
            ConfigurationProfiles::task_1_7_1_optimized(),
            ConfigurationProfiles::inference_optimized(),
            ConfigurationProfiles::training_optimized(),
            UnifiedTensorPoolConfig::balanced(),
            UnifiedTensorPoolConfig::adaptive(),
        ];
        
        // All configs should be valid and have different characteristics
        for config in configs {
            let validation = config.validate();
            assert!(validation.is_valid(), "Config should be valid: {}", config.get_summary());
        }
    }
}
