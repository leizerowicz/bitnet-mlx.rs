//! Mixed Precision Configuration System
//!
//! This module provides comprehensive configuration for mixed precision operations,
//! allowing fine-grained control over precision settings for different layers and components.

use super::{
    ComponentType, LayerType, MixedPrecisionError, MixedPrecisionResult, MixedPrecisionStrategy,
};
use crate::memory::tensor::BitNetDType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration for mixed precision operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MixedPrecisionConfig {
    /// Global mixed precision strategy
    pub strategy: MixedPrecisionStrategy,
    /// Enable mixed precision globally
    pub enabled: bool,
    /// Layer-specific precision configurations
    pub layer_configs: HashMap<String, LayerPrecisionConfig>,
    /// Component-specific precision configurations
    pub component_configs: HashMap<ComponentType, ComponentPrecisionConfig>,
    /// Default precision for unspecified layers
    pub default_layer_precision: BitNetDType,
    /// Default precision for unspecified components
    pub default_component_precision: BitNetDType,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimizationConfig,
    /// Performance optimization settings
    pub performance_optimization: PerformanceOptimizationConfig,
    /// Validation settings
    pub validation: ValidationConfig,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            strategy: MixedPrecisionStrategy::Balanced,
            enabled: true,
            layer_configs: HashMap::new(),
            component_configs: HashMap::new(),
            default_layer_precision: BitNetDType::F16,
            default_component_precision: BitNetDType::F16,
            memory_optimization: MemoryOptimizationConfig::default(),
            performance_optimization: PerformanceOptimizationConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new mixed precision configuration with the given strategy
    pub fn new(strategy: MixedPrecisionStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Create a conservative configuration prioritizing accuracy
    pub fn conservative() -> Self {
        Self {
            strategy: MixedPrecisionStrategy::Conservative,
            default_layer_precision: BitNetDType::F16,
            default_component_precision: BitNetDType::F16,
            memory_optimization: MemoryOptimizationConfig::conservative(),
            performance_optimization: PerformanceOptimizationConfig::conservative(),
            ..Default::default()
        }
    }

    /// Create a balanced configuration balancing accuracy and efficiency
    pub fn balanced() -> Self {
        Self {
            strategy: MixedPrecisionStrategy::Balanced,
            default_layer_precision: BitNetDType::I8,
            default_component_precision: BitNetDType::I8,
            memory_optimization: MemoryOptimizationConfig::balanced(),
            performance_optimization: PerformanceOptimizationConfig::balanced(),
            ..Default::default()
        }
    }

    /// Create an aggressive configuration prioritizing memory and speed
    pub fn aggressive() -> Self {
        Self {
            strategy: MixedPrecisionStrategy::Aggressive,
            default_layer_precision: BitNetDType::I4,
            default_component_precision: BitNetDType::I4,
            memory_optimization: MemoryOptimizationConfig::aggressive(),
            performance_optimization: PerformanceOptimizationConfig::aggressive(),
            ..Default::default()
        }
    }

    /// Add a layer-specific precision configuration
    pub fn with_layer_config(mut self, layer_name: String, config: LayerPrecisionConfig) -> Self {
        self.layer_configs.insert(layer_name, config);
        self
    }

    /// Add a component-specific precision configuration
    pub fn with_component_config(
        mut self,
        component_type: ComponentType,
        config: ComponentPrecisionConfig,
    ) -> Self {
        self.component_configs.insert(component_type, config);
        self
    }

    /// Set the default layer precision
    pub fn with_default_layer_precision(mut self, precision: BitNetDType) -> Self {
        self.default_layer_precision = precision;
        self
    }

    /// Set the default component precision
    pub fn with_default_component_precision(mut self, precision: BitNetDType) -> Self {
        self.default_component_precision = precision;
        self
    }

    /// Enable or disable mixed precision
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Get the precision for a specific layer
    pub fn get_layer_precision(&self, layer_name: &str, layer_type: LayerType) -> BitNetDType {
        if !self.enabled {
            return BitNetDType::F32; // Fall back to full precision if disabled
        }

        // Check for layer-specific configuration
        if let Some(config) = self.layer_configs.get(layer_name) {
            return config.precision;
        }

        // Use strategy-based precision
        self.strategy.get_layer_precision(layer_type)
    }

    /// Get the precision for a specific component
    pub fn get_component_precision(&self, component_type: ComponentType) -> BitNetDType {
        if !self.enabled {
            return BitNetDType::F32; // Fall back to full precision if disabled
        }

        // Check for component-specific configuration
        if let Some(config) = self.component_configs.get(&component_type) {
            return config.precision;
        }

        // Use strategy-based precision
        self.strategy.get_component_precision(component_type)
    }

    /// Validate the configuration
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        // Validate layer configurations
        for (layer_name, config) in &self.layer_configs {
            config.validate().map_err(|e| {
                MixedPrecisionError::InvalidConfiguration(format!("Layer '{}': {}", layer_name, e))
            })?;
        }

        // Validate component configurations
        for (component_type, config) in &self.component_configs {
            config.validate().map_err(|e| {
                MixedPrecisionError::InvalidConfiguration(format!(
                    "Component '{:?}': {}",
                    component_type, e
                ))
            })?;
        }

        // Validate memory optimization settings
        self.memory_optimization.validate()?;

        // Validate performance optimization settings
        self.performance_optimization.validate()?;

        Ok(())
    }

    /// Get memory savings estimate
    pub fn estimate_memory_savings(&self) -> f32 {
        let mut total_savings = 0.0;
        let mut count = 0;

        // Estimate savings from layer configurations
        for config in self.layer_configs.values() {
            total_savings += config.precision.memory_efficiency();
            count += 1;
        }

        // Estimate savings from component configurations
        for config in self.component_configs.values() {
            total_savings += config.precision.memory_efficiency();
            count += 1;
        }

        // If no specific configurations, use strategy defaults
        if count == 0 {
            return self.default_layer_precision.memory_efficiency();
        }

        total_savings / count as f32
    }
}

/// Configuration for a specific layer's precision settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LayerPrecisionConfig {
    /// Layer type
    pub layer_type: LayerType,
    /// Precision for this layer
    pub precision: BitNetDType,
    /// Component-specific overrides
    pub component_overrides: HashMap<ComponentType, BitNetDType>,
    /// Whether to enable automatic precision adjustment
    pub auto_adjust: bool,
    /// Minimum precision allowed for this layer
    pub min_precision: Option<BitNetDType>,
    /// Maximum precision allowed for this layer
    pub max_precision: Option<BitNetDType>,
    /// Custom metadata for this layer
    pub metadata: HashMap<String, String>,
}

impl LayerPrecisionConfig {
    /// Create a new layer precision configuration
    pub fn new(layer_type: LayerType, precision: BitNetDType) -> Self {
        Self {
            layer_type,
            precision,
            component_overrides: HashMap::new(),
            auto_adjust: false,
            min_precision: None,
            max_precision: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a component-specific precision override
    pub fn with_component_override(
        mut self,
        component: ComponentType,
        precision: BitNetDType,
    ) -> Self {
        self.component_overrides.insert(component, precision);
        self
    }

    /// Enable automatic precision adjustment
    pub fn with_auto_adjust(mut self) -> Self {
        self.auto_adjust = true;
        self
    }

    /// Set precision bounds
    pub fn with_precision_bounds(mut self, min: BitNetDType, max: BitNetDType) -> Self {
        self.min_precision = Some(min);
        self.max_precision = Some(max);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the precision for a specific component in this layer
    pub fn get_component_precision(&self, component: ComponentType) -> BitNetDType {
        self.component_overrides
            .get(&component)
            .copied()
            .unwrap_or(self.precision)
    }

    /// Validate the configuration
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        // Check if the layer type supports the specified precision
        if !self.layer_type.supports_precision(self.precision) {
            return Err(MixedPrecisionError::InvalidConfiguration(format!(
                "Layer type {:?} does not support precision {:?}",
                self.layer_type, self.precision
            )));
        }

        // Validate component overrides
        for (component, precision) in &self.component_overrides {
            if !component.supports_precision(*precision) {
                return Err(MixedPrecisionError::InvalidConfiguration(format!(
                    "Component {:?} does not support precision {:?}",
                    component, precision
                )));
            }
        }

        // Validate precision bounds
        if let (Some(min), Some(max)) = (self.min_precision, self.max_precision) {
            if min.bits_per_element() > max.bits_per_element() {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    "Minimum precision cannot be higher than maximum precision".to_string(),
                ));
            }

            if self.precision.bits_per_element() < min.bits_per_element()
                || self.precision.bits_per_element() > max.bits_per_element()
            {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    "Layer precision is outside specified bounds".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Configuration for a specific component's precision settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ComponentPrecisionConfig {
    /// Component type
    pub component_type: ComponentType,
    /// Precision for this component
    pub precision: BitNetDType,
    /// Whether to enable dynamic precision adjustment
    pub dynamic_adjustment: bool,
    /// Adjustment parameters
    pub adjustment_params: AdjustmentParams,
    /// Custom metadata for this component
    pub metadata: HashMap<String, String>,
}

impl ComponentPrecisionConfig {
    /// Create a new component precision configuration
    pub fn new(component_type: ComponentType, precision: BitNetDType) -> Self {
        Self {
            component_type,
            precision,
            dynamic_adjustment: false,
            adjustment_params: AdjustmentParams::default(),
            metadata: HashMap::new(),
        }
    }

    /// Enable dynamic precision adjustment
    pub fn with_dynamic_adjustment(mut self, params: AdjustmentParams) -> Self {
        self.dynamic_adjustment = true;
        self.adjustment_params = params;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        // Check if the component type supports the specified precision
        if !self.component_type.supports_precision(self.precision) {
            return Err(MixedPrecisionError::InvalidConfiguration(format!(
                "Component type {:?} does not support precision {:?}",
                self.component_type, self.precision
            )));
        }

        // Validate adjustment parameters if dynamic adjustment is enabled
        if self.dynamic_adjustment {
            self.adjustment_params.validate()?;
        }

        Ok(())
    }
}

/// Parameters for dynamic precision adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AdjustmentParams {
    /// Threshold for increasing precision (error rate)
    pub increase_threshold: f32,
    /// Threshold for decreasing precision (efficiency gain)
    pub decrease_threshold: f32,
    /// Maximum adjustment steps per evaluation
    pub max_adjustment_steps: usize,
    /// Evaluation frequency (in operations)
    pub evaluation_frequency: usize,
}

impl Default for AdjustmentParams {
    fn default() -> Self {
        Self {
            increase_threshold: 0.05, // 5% error rate
            decrease_threshold: 0.01, // 1% efficiency gain
            max_adjustment_steps: 2,
            evaluation_frequency: 1000,
        }
    }
}

impl AdjustmentParams {
    /// Validate the adjustment parameters
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        if self.increase_threshold <= 0.0 || self.increase_threshold > 1.0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Increase threshold must be between 0 and 1".to_string(),
            ));
        }

        if self.decrease_threshold <= 0.0 || self.decrease_threshold > 1.0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Decrease threshold must be between 0 and 1".to_string(),
            ));
        }

        if self.max_adjustment_steps == 0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Max adjustment steps must be greater than 0".to_string(),
            ));
        }

        if self.evaluation_frequency == 0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Evaluation frequency must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MemoryOptimizationConfig {
    /// Enable memory pooling for different precisions
    pub enable_memory_pooling: bool,
    /// Enable tensor reuse across operations
    pub enable_tensor_reuse: bool,
    /// Enable gradient checkpointing
    pub enable_gradient_checkpointing: bool,
    /// Memory pressure threshold for automatic precision reduction
    pub memory_pressure_threshold: f32,
    /// Target memory usage reduction (percentage)
    pub target_memory_reduction: f32,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_memory_pooling: true,
            enable_tensor_reuse: true,
            enable_gradient_checkpointing: false,
            memory_pressure_threshold: 0.8, // 80% memory usage
            target_memory_reduction: 0.3,   // 30% reduction
        }
    }
}

impl MemoryOptimizationConfig {
    /// Create conservative memory optimization settings
    pub fn conservative() -> Self {
        Self {
            enable_memory_pooling: true,
            enable_tensor_reuse: false,
            enable_gradient_checkpointing: false,
            memory_pressure_threshold: 0.9,
            target_memory_reduction: 0.1,
        }
    }

    /// Create balanced memory optimization settings
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create aggressive memory optimization settings
    pub fn aggressive() -> Self {
        Self {
            enable_memory_pooling: true,
            enable_tensor_reuse: true,
            enable_gradient_checkpointing: true,
            memory_pressure_threshold: 0.7,
            target_memory_reduction: 0.5,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        if self.memory_pressure_threshold <= 0.0 || self.memory_pressure_threshold > 1.0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Memory pressure threshold must be between 0 and 1".to_string(),
            ));
        }

        if self.target_memory_reduction < 0.0 || self.target_memory_reduction > 1.0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Target memory reduction must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PerformanceOptimizationConfig {
    /// Enable SIMD optimizations for mixed precision operations
    pub enable_simd_optimizations: bool,
    /// Enable kernel fusion for precision conversions
    pub enable_kernel_fusion: bool,
    /// Enable asynchronous precision conversions
    pub enable_async_conversions: bool,
    /// Batch size threshold for vectorized operations
    pub vectorization_threshold: usize,
    /// Number of threads for parallel precision operations
    pub num_threads: Option<usize>,
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd_optimizations: true,
            enable_kernel_fusion: true,
            enable_async_conversions: false,
            vectorization_threshold: 64,
            num_threads: None, // Auto-detect
        }
    }
}

impl PerformanceOptimizationConfig {
    /// Create conservative performance optimization settings
    pub fn conservative() -> Self {
        Self {
            enable_simd_optimizations: false,
            enable_kernel_fusion: false,
            enable_async_conversions: false,
            vectorization_threshold: 128,
            num_threads: Some(1),
        }
    }

    /// Create balanced performance optimization settings
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create aggressive performance optimization settings
    pub fn aggressive() -> Self {
        Self {
            enable_simd_optimizations: true,
            enable_kernel_fusion: true,
            enable_async_conversions: true,
            vectorization_threshold: 32,
            num_threads: None, // Use all available cores
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        if self.vectorization_threshold == 0 {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Vectorization threshold must be greater than 0".to_string(),
            ));
        }

        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    "Number of threads must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ValidationConfig {
    /// Enable strict validation of precision compatibility
    pub strict_validation: bool,
    /// Enable runtime precision checks
    pub runtime_checks: bool,
    /// Enable precision overflow detection
    pub overflow_detection: bool,
    /// Tolerance for precision conversion errors
    pub conversion_tolerance: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            runtime_checks: false,
            overflow_detection: true,
            conversion_tolerance: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.strategy, MixedPrecisionStrategy::Balanced);
        assert!(config.enabled);
        assert_eq!(config.default_layer_precision, BitNetDType::F16);
    }

    #[test]
    fn test_mixed_precision_config_strategies() {
        let conservative = MixedPrecisionConfig::conservative();
        assert_eq!(conservative.strategy, MixedPrecisionStrategy::Conservative);
        assert_eq!(conservative.default_layer_precision, BitNetDType::F16);

        let aggressive = MixedPrecisionConfig::aggressive();
        assert_eq!(aggressive.strategy, MixedPrecisionStrategy::Aggressive);
        assert_eq!(aggressive.default_layer_precision, BitNetDType::I4);
    }

    #[test]
    fn test_layer_precision_config() {
        let config = LayerPrecisionConfig::new(LayerType::Linear, BitNetDType::BitNet158)
            .with_component_override(ComponentType::Bias, BitNetDType::F16)
            .with_auto_adjust();

        assert_eq!(config.layer_type, LayerType::Linear);
        assert_eq!(config.precision, BitNetDType::BitNet158);
        assert!(config.auto_adjust);
        assert_eq!(
            config.get_component_precision(ComponentType::Bias),
            BitNetDType::F16
        );
        assert_eq!(
            config.get_component_precision(ComponentType::Weights),
            BitNetDType::BitNet158
        );
    }

    #[test]
    fn test_component_precision_config() {
        let config = ComponentPrecisionConfig::new(ComponentType::Weights, BitNetDType::I4)
            .with_dynamic_adjustment(AdjustmentParams::default());

        assert_eq!(config.component_type, ComponentType::Weights);
        assert_eq!(config.precision, BitNetDType::I4);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_config_validation() {
        let mut config = MixedPrecisionConfig::default();
        assert!(config.validate().is_ok());

        // Add invalid layer config
        let invalid_layer_config =
            LayerPrecisionConfig::new(LayerType::Normalization, BitNetDType::I1);
        config
            .layer_configs
            .insert("test_layer".to_string(), invalid_layer_config);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_memory_optimization_config() {
        let config = MemoryOptimizationConfig::aggressive();
        assert!(config.enable_memory_pooling);
        assert!(config.enable_tensor_reuse);
        assert!(config.enable_gradient_checkpointing);
        assert_eq!(config.memory_pressure_threshold, 0.7);
    }

    #[test]
    fn test_adjustment_params_validation() {
        let mut params = AdjustmentParams::default();
        assert!(params.validate().is_ok());

        params.increase_threshold = 1.5; // Invalid
        assert!(params.validate().is_err());

        params.increase_threshold = 0.05;
        params.evaluation_frequency = 0; // Invalid
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_precision_bounds() {
        let config = LayerPrecisionConfig::new(LayerType::Linear, BitNetDType::I8)
            .with_precision_bounds(BitNetDType::I4, BitNetDType::F16);

        assert!(config.validate().is_ok());

        // Test invalid bounds
        let invalid_config = LayerPrecisionConfig::new(LayerType::Linear, BitNetDType::F32)
            .with_precision_bounds(BitNetDType::F16, BitNetDType::I8); // min > max

        assert!(invalid_config.validate().is_err());
    }
}
