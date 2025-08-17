//! Layer-Specific Precision Management
//!
//! This module provides layer-specific precision management, allowing different layers
//! to operate with different precision levels while maintaining compatibility.

use super::{LayerType, ComponentType, MixedPrecisionError, MixedPrecisionResult};
use super::config::LayerPrecisionConfig;
use crate::memory::tensor::BitNetDType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Specification for a layer's precision requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPrecisionSpec {
    /// Layer identifier
    pub layer_id: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Input precision requirements
    pub input_precision: BitNetDType,
    /// Output precision requirements
    pub output_precision: BitNetDType,
    /// Weight precision
    pub weight_precision: BitNetDType,
    /// Component-specific precisions
    pub component_precisions: HashMap<ComponentType, BitNetDType>,
    /// Whether this layer supports dynamic precision adjustment
    pub supports_dynamic_adjustment: bool,
    /// Minimum supported precision
    pub min_precision: BitNetDType,
    /// Maximum supported precision
    pub max_precision: BitNetDType,
    /// Layer-specific metadata
    pub metadata: HashMap<String, String>,
}

impl LayerPrecisionSpec {
    /// Create a new layer precision specification
    pub fn new(
        layer_id: String,
        layer_type: LayerType,
        input_precision: BitNetDType,
        output_precision: BitNetDType,
        weight_precision: BitNetDType,
    ) -> Self {
        Self {
            layer_id,
            layer_type,
            input_precision,
            output_precision,
            weight_precision,
            component_precisions: HashMap::new(),
            supports_dynamic_adjustment: false,
            min_precision: BitNetDType::I1,
            max_precision: BitNetDType::F32,
            metadata: HashMap::new(),
        }
    }

    /// Create a specification from a layer configuration
    pub fn from_config(layer_id: String, config: &LayerPrecisionConfig) -> Self {
        let mut spec = Self::new(
            layer_id,
            config.layer_type,
            config.precision,
            config.precision,
            config.precision,
        );

        spec.component_precisions = config.component_overrides.clone();
        spec.supports_dynamic_adjustment = config.auto_adjust;
        spec.min_precision = config.min_precision.unwrap_or(BitNetDType::I1);
        spec.max_precision = config.max_precision.unwrap_or(BitNetDType::F32);
        spec.metadata = config.metadata.clone();

        spec
    }

    /// Set component precision
    pub fn with_component_precision(mut self, component: ComponentType, precision: BitNetDType) -> Self {
        self.component_precisions.insert(component, precision);
        self
    }

    /// Enable dynamic precision adjustment
    pub fn with_dynamic_adjustment(mut self) -> Self {
        self.supports_dynamic_adjustment = true;
        self
    }

    /// Set precision bounds
    pub fn with_precision_bounds(mut self, min: BitNetDType, max: BitNetDType) -> Self {
        self.min_precision = min;
        self.max_precision = max;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the precision for a specific component
    pub fn get_component_precision(&self, component: ComponentType) -> BitNetDType {
        self.component_precisions.get(&component).copied().unwrap_or_else(|| {
            match component {
                ComponentType::Weights => self.weight_precision,
                ComponentType::Activations => self.input_precision,
                _ => self.output_precision,
            }
        })
    }

    /// Check if the layer supports the given precision
    pub fn supports_precision(&self, precision: BitNetDType) -> bool {
        precision.bits_per_element() >= self.min_precision.bits_per_element() &&
        precision.bits_per_element() <= self.max_precision.bits_per_element() &&
        self.layer_type.supports_precision(precision)
    }

    /// Get the optimal precision for this layer given constraints
    pub fn get_optimal_precision(&self, target_memory_reduction: f32) -> BitNetDType {
        let alternatives = self.layer_type.precision_alternatives();
        
        // Find the precision that achieves the target memory reduction
        let target_efficiency = 1.0 + target_memory_reduction;
        
        let mut best_precision = self.weight_precision;
        let mut best_score = f32::INFINITY;
        
        for &precision in &alternatives {
            if self.supports_precision(precision) {
                let efficiency = precision.memory_efficiency();
                let score = (efficiency - target_efficiency).abs();
                
                if score < best_score {
                    best_score = score;
                    best_precision = precision;
                }
            }
        }
        
        best_precision
    }

    /// Validate the specification
    pub fn validate(&self) -> MixedPrecisionResult<()> {
        // Check if layer type supports all specified precisions
        if !self.layer_type.supports_precision(self.input_precision) {
            return Err(MixedPrecisionError::InvalidConfiguration(
                format!("Layer type {:?} does not support input precision {:?}", 
                    self.layer_type, self.input_precision)
            ));
        }

        if !self.layer_type.supports_precision(self.output_precision) {
            return Err(MixedPrecisionError::InvalidConfiguration(
                format!("Layer type {:?} does not support output precision {:?}", 
                    self.layer_type, self.output_precision)
            ));
        }

        if !self.layer_type.supports_precision(self.weight_precision) {
            return Err(MixedPrecisionError::InvalidConfiguration(
                format!("Layer type {:?} does not support weight precision {:?}", 
                    self.layer_type, self.weight_precision)
            ));
        }

        // Validate component precisions
        for (component, precision) in &self.component_precisions {
            if !component.supports_precision(*precision) {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    format!("Component {:?} does not support precision {:?}", 
                        component, precision)
                ));
            }
        }

        // Validate precision bounds
        if self.min_precision.bits_per_element() > self.max_precision.bits_per_element() {
            return Err(MixedPrecisionError::InvalidConfiguration(
                "Minimum precision cannot be higher than maximum precision".to_string()
            ));
        }

        Ok(())
    }
}

/// Manager for layer-specific precision configurations
#[derive(Debug)]
pub struct LayerPrecisionManager {
    /// Layer specifications
    layer_specs: Arc<RwLock<HashMap<String, LayerPrecisionSpec>>>,
    /// Global precision constraints
    global_constraints: Arc<RwLock<PrecisionConstraints>>,
    /// Performance metrics for precision decisions
    performance_metrics: Arc<RwLock<HashMap<String, LayerPerformanceMetrics>>>,
}

impl LayerPrecisionManager {
    /// Create a new layer precision manager
    pub fn new() -> Self {
        Self {
            layer_specs: Arc::new(RwLock::new(HashMap::new())),
            global_constraints: Arc::new(RwLock::new(PrecisionConstraints::default())),
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a layer with its precision specification
    pub fn register_layer(&self, spec: LayerPrecisionSpec) -> MixedPrecisionResult<()> {
        spec.validate()?;
        
        let mut specs = self.layer_specs.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire write lock".to_string())
        })?;
        
        specs.insert(spec.layer_id.clone(), spec);
        Ok(())
    }

    /// Get the precision specification for a layer
    pub fn get_layer_spec(&self, layer_id: &str) -> Option<LayerPrecisionSpec> {
        let specs = self.layer_specs.read().ok()?;
        specs.get(layer_id).cloned()
    }

    /// Update the precision for a specific layer
    pub fn update_layer_precision(
        &self,
        layer_id: &str,
        new_precision: BitNetDType,
    ) -> MixedPrecisionResult<()> {
        let mut specs = self.layer_specs.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire write lock".to_string())
        })?;

        if let Some(spec) = specs.get_mut(layer_id) {
            if !spec.supports_precision(new_precision) {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    format!("Layer {} does not support precision {:?}", layer_id, new_precision)
                ));
            }

            spec.weight_precision = new_precision;
            spec.input_precision = new_precision;
            spec.output_precision = new_precision;
        } else {
            return Err(MixedPrecisionError::InvalidConfiguration(
                format!("Layer {} not found", layer_id)
            ));
        }

        Ok(())
    }

    /// Update component precision for a specific layer
    pub fn update_component_precision(
        &self,
        layer_id: &str,
        component: ComponentType,
        new_precision: BitNetDType,
    ) -> MixedPrecisionResult<()> {
        let mut specs = self.layer_specs.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire write lock".to_string())
        })?;

        if let Some(spec) = specs.get_mut(layer_id) {
            if !component.supports_precision(new_precision) {
                return Err(MixedPrecisionError::InvalidConfiguration(
                    format!("Component {:?} does not support precision {:?}", component, new_precision)
                ));
            }

            spec.component_precisions.insert(component, new_precision);
        } else {
            return Err(MixedPrecisionError::InvalidConfiguration(
                format!("Layer {} not found", layer_id)
            ));
        }

        Ok(())
    }

    /// Get all layers of a specific type
    pub fn get_layers_by_type(&self, layer_type: LayerType) -> Vec<LayerPrecisionSpec> {
        let specs = self.layer_specs.read().unwrap();
        specs.values()
            .filter(|spec| spec.layer_type == layer_type)
            .cloned()
            .collect()
    }

    /// Get layers using a specific precision
    pub fn get_layers_by_precision(&self, precision: BitNetDType) -> Vec<LayerPrecisionSpec> {
        let specs = self.layer_specs.read().unwrap();
        specs.values()
            .filter(|spec| spec.weight_precision == precision)
            .cloned()
            .collect()
    }

    /// Optimize precision across all layers for memory efficiency
    pub fn optimize_for_memory(&self, target_reduction: f32) -> MixedPrecisionResult<Vec<(String, BitNetDType)>> {
        let specs = self.layer_specs.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire read lock".to_string())
        })?;

        let mut optimizations = Vec::new();

        for (layer_id, spec) in specs.iter() {
            let optimal_precision = spec.get_optimal_precision(target_reduction);
            if optimal_precision != spec.weight_precision {
                optimizations.push((layer_id.clone(), optimal_precision));
            }
        }

        Ok(optimizations)
    }

    /// Apply precision optimizations
    pub fn apply_optimizations(&self, optimizations: Vec<(String, BitNetDType)>) -> MixedPrecisionResult<()> {
        for (layer_id, precision) in optimizations {
            self.update_layer_precision(&layer_id, precision)?;
        }
        Ok(())
    }

    /// Record performance metrics for a layer
    pub fn record_performance(
        &self,
        layer_id: &str,
        metrics: LayerPerformanceMetrics,
    ) -> MixedPrecisionResult<()> {
        let mut perf_metrics = self.performance_metrics.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire write lock".to_string())
        })?;

        perf_metrics.insert(layer_id.to_string(), metrics);
        Ok(())
    }

    /// Get performance metrics for a layer
    pub fn get_performance_metrics(&self, layer_id: &str) -> Option<LayerPerformanceMetrics> {
        let perf_metrics = self.performance_metrics.read().ok()?;
        perf_metrics.get(layer_id).cloned()
    }

    /// Analyze precision impact across all layers
    pub fn analyze_precision_impact(&self) -> PrecisionImpactAnalysis {
        let specs = self.layer_specs.read().unwrap();
        let perf_metrics = self.performance_metrics.read().unwrap();

        let mut total_memory_savings = 0.0;
        let mut total_accuracy_impact = 0.0;
        let mut layer_count = 0;
        let mut precision_distribution = HashMap::new();

        for (layer_id, spec) in specs.iter() {
            // Calculate memory savings
            let baseline_efficiency = BitNetDType::F32.memory_efficiency();
            let current_efficiency = spec.weight_precision.memory_efficiency();
            let memory_savings = (current_efficiency - baseline_efficiency) / baseline_efficiency;
            total_memory_savings += memory_savings;

            // Get accuracy impact if available
            if let Some(metrics) = perf_metrics.get(layer_id) {
                total_accuracy_impact += metrics.accuracy_impact;
            }

            // Track precision distribution
            *precision_distribution.entry(spec.weight_precision).or_insert(0) += 1;
            layer_count += 1;
        }

        PrecisionImpactAnalysis {
            average_memory_savings: if layer_count > 0 { total_memory_savings / layer_count as f32 } else { 0.0 },
            average_accuracy_impact: if layer_count > 0 { total_accuracy_impact / layer_count as f32 } else { 0.0 },
            precision_distribution,
            total_layers: layer_count,
        }
    }

    /// Set global precision constraints
    pub fn set_global_constraints(&self, constraints: PrecisionConstraints) -> MixedPrecisionResult<()> {
        let mut global_constraints = self.global_constraints.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire write lock".to_string())
        })?;

        *global_constraints = constraints;
        Ok(())
    }

    /// Validate all layer specifications against global constraints
    pub fn validate_all_layers(&self) -> MixedPrecisionResult<()> {
        let specs = self.layer_specs.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire read lock".to_string())
        })?;

        let constraints = self.global_constraints.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire read lock".to_string())
        })?;

        for (layer_id, spec) in specs.iter() {
            spec.validate()?;
            constraints.validate_layer_spec(spec).map_err(|e| {
                MixedPrecisionError::InvalidConfiguration(
                    format!("Layer '{}' violates global constraints: {}", layer_id, e)
                )
            })?;
        }

        Ok(())
    }
}

impl Default for LayerPrecisionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global precision constraints
#[derive(Debug, Clone)]
pub struct PrecisionConstraints {
    /// Minimum allowed precision globally
    pub min_global_precision: BitNetDType,
    /// Maximum allowed precision globally
    pub max_global_precision: BitNetDType,
    /// Maximum memory usage (in bytes)
    pub max_memory_usage: Option<usize>,
    /// Minimum accuracy threshold
    pub min_accuracy_threshold: f32,
    /// Layer-type specific constraints
    pub layer_type_constraints: HashMap<LayerType, LayerTypeConstraints>,
}

impl Default for PrecisionConstraints {
    fn default() -> Self {
        Self {
            min_global_precision: BitNetDType::I1,
            max_global_precision: BitNetDType::F32,
            max_memory_usage: None,
            min_accuracy_threshold: 0.95,
            layer_type_constraints: HashMap::new(),
        }
    }
}

impl PrecisionConstraints {
    /// Validate a layer specification against these constraints
    pub fn validate_layer_spec(&self, spec: &LayerPrecisionSpec) -> MixedPrecisionResult<()> {
        // Check global precision bounds
        if spec.weight_precision.bits_per_element() < self.min_global_precision.bits_per_element() {
            return Err(MixedPrecisionError::ValidationError(
                format!("Layer precision {:?} is below global minimum {:?}", 
                    spec.weight_precision, self.min_global_precision)
            ));
        }

        if spec.weight_precision.bits_per_element() > self.max_global_precision.bits_per_element() {
            return Err(MixedPrecisionError::ValidationError(
                format!("Layer precision {:?} is above global maximum {:?}", 
                    spec.weight_precision, self.max_global_precision)
            ));
        }

        // Check layer-type specific constraints
        if let Some(type_constraints) = self.layer_type_constraints.get(&spec.layer_type) {
            type_constraints.validate_spec(spec)?;
        }

        Ok(())
    }
}

/// Layer-type specific constraints
#[derive(Debug, Clone)]
pub struct LayerTypeConstraints {
    /// Allowed precisions for this layer type
    pub allowed_precisions: Vec<BitNetDType>,
    /// Required components for this layer type
    pub required_components: Vec<ComponentType>,
    /// Maximum memory per layer of this type
    pub max_memory_per_layer: Option<usize>,
}

impl LayerTypeConstraints {
    /// Validate a layer specification against these constraints
    pub fn validate_spec(&self, spec: &LayerPrecisionSpec) -> MixedPrecisionResult<()> {
        // Check if precision is allowed
        if !self.allowed_precisions.is_empty() && 
           !self.allowed_precisions.contains(&spec.weight_precision) {
            return Err(MixedPrecisionError::ValidationError(
                format!("Precision {:?} is not allowed for layer type {:?}", 
                    spec.weight_precision, spec.layer_type)
            ));
        }

        // Check required components
        for &required_component in &self.required_components {
            if !spec.component_precisions.contains_key(&required_component) {
                return Err(MixedPrecisionError::ValidationError(
                    format!("Required component {:?} not specified for layer type {:?}", 
                        required_component, spec.layer_type)
                ));
            }
        }

        Ok(())
    }
}

/// Performance metrics for a layer
#[derive(Debug, Clone)]
pub struct LayerPerformanceMetrics {
    /// Execution time (in milliseconds)
    pub execution_time_ms: f32,
    /// Memory usage (in bytes)
    pub memory_usage_bytes: usize,
    /// Accuracy impact (relative to full precision)
    pub accuracy_impact: f32,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f32,
    /// Energy consumption (in joules)
    pub energy_consumption_j: f32,
}

impl Default for LayerPerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            accuracy_impact: 0.0,
            throughput_ops_per_sec: 0.0,
            energy_consumption_j: 0.0,
        }
    }
}

/// Analysis of precision impact across layers
#[derive(Debug, Clone)]
pub struct PrecisionImpactAnalysis {
    /// Average memory savings across all layers
    pub average_memory_savings: f32,
    /// Average accuracy impact across all layers
    pub average_accuracy_impact: f32,
    /// Distribution of precisions across layers
    pub precision_distribution: HashMap<BitNetDType, usize>,
    /// Total number of layers analyzed
    pub total_layers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_precision_spec_creation() {
        let spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::BitNet158,
        );

        assert_eq!(spec.layer_id, "test_layer");
        assert_eq!(spec.layer_type, LayerType::Linear);
        assert_eq!(spec.input_precision, BitNetDType::I8);
        assert_eq!(spec.weight_precision, BitNetDType::BitNet158);
    }

    #[test]
    fn test_layer_precision_spec_component_precision() {
        let spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::BitNet158,
        ).with_component_precision(ComponentType::Bias, BitNetDType::F16);

        assert_eq!(spec.get_component_precision(ComponentType::Bias), BitNetDType::F16);
        assert_eq!(spec.get_component_precision(ComponentType::Weights), BitNetDType::BitNet158);
    }

    #[test]
    fn test_layer_precision_manager() {
        let manager = LayerPrecisionManager::new();
        
        let spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::BitNet158,
        );

        assert!(manager.register_layer(spec).is_ok());
        
        let retrieved_spec = manager.get_layer_spec("test_layer");
        assert!(retrieved_spec.is_some());
        assert_eq!(retrieved_spec.unwrap().layer_type, LayerType::Linear);
    }

    #[test]
    fn test_precision_optimization() {
        let manager = LayerPrecisionManager::new();
        
        let spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::F32,
            BitNetDType::F32,
            BitNetDType::F32,
        );

        manager.register_layer(spec).unwrap();
        
        let optimizations = manager.optimize_for_memory(0.5).unwrap();
        assert!(!optimizations.is_empty());
        
        let (layer_id, new_precision) = &optimizations[0];
        assert_eq!(layer_id, "test_layer");
        assert_ne!(*new_precision, BitNetDType::F32); // Should suggest a more efficient precision
    }

    #[test]
    fn test_precision_constraints() {
        let constraints = PrecisionConstraints {
            min_global_precision: BitNetDType::I4,
            max_global_precision: BitNetDType::F16,
            ..Default::default()
        };

        let valid_spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::I8,
        );

        assert!(constraints.validate_layer_spec(&valid_spec).is_ok());

        let invalid_spec = LayerPrecisionSpec::new(
            "test_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I1,
            BitNetDType::I1,
            BitNetDType::I1,
        );

        assert!(constraints.validate_layer_spec(&invalid_spec).is_err());
    }

    #[test]
    fn test_performance_metrics() {
        let manager = LayerPrecisionManager::new();
        
        let metrics = LayerPerformanceMetrics {
            execution_time_ms: 10.5,
            memory_usage_bytes: 1024,
            accuracy_impact: 0.02,
            throughput_ops_per_sec: 1000.0,
            energy_consumption_j: 0.5,
        };

        assert!(manager.record_performance("test_layer", metrics.clone()).is_ok());
        
        let retrieved_metrics = manager.get_performance_metrics("test_layer");
        assert!(retrieved_metrics.is_some());
        assert_eq!(retrieved_metrics.unwrap().execution_time_ms, 10.5);
    }
}