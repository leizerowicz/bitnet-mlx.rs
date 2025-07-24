//! Precision Manager
//!
//! This module provides the central coordination for mixed precision operations,
//! managing precision policies, conversions, and optimization across the entire model.

use super::{
    LayerType, ComponentType, MixedPrecisionStrategy, MixedPrecisionError, MixedPrecisionResult,
    config::{MixedPrecisionConfig, LayerPrecisionConfig, ComponentPrecisionConfig},
    layer_precision::{LayerPrecisionManager, LayerPrecisionSpec},
    conversion::{PrecisionConverter, ConversionConfig, ConversionStrategy},
    validation::{PrecisionValidator, ValidationRule},
    policy::{PrecisionPolicy, PolicyEngine},
};
use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::HybridMemoryPool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};

/// Context for precision decisions
#[derive(Debug, Clone)]
pub struct PrecisionContext {
    /// Current layer being processed
    pub current_layer: Option<String>,
    /// Current operation being performed
    pub current_operation: Option<String>,
    /// Available memory (in bytes)
    pub available_memory: Option<usize>,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Accuracy requirements
    pub accuracy_requirements: AccuracyRequirements,
    /// Device capabilities
    pub device_capabilities: DeviceCapabilities,
}

impl Default for PrecisionContext {
    fn default() -> Self {
        Self {
            current_layer: None,
            current_operation: None,
            available_memory: None,
            performance_requirements: PerformanceRequirements::default(),
            accuracy_requirements: AccuracyRequirements::default(),
            device_capabilities: DeviceCapabilities::default(),
        }
    }
}

/// Performance requirements for precision decisions
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Target throughput (operations per second)
    pub target_throughput: Option<f32>,
    /// Maximum acceptable latency (milliseconds)
    pub max_latency_ms: Option<f32>,
    /// Memory usage constraints (bytes)
    pub max_memory_usage: Option<usize>,
    /// Energy consumption constraints (joules)
    pub max_energy_consumption: Option<f32>,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            target_throughput: None,
            max_latency_ms: None,
            max_memory_usage: None,
            max_energy_consumption: None,
        }
    }
}

/// Accuracy requirements for precision decisions
#[derive(Debug, Clone)]
pub struct AccuracyRequirements {
    /// Minimum acceptable accuracy
    pub min_accuracy: Option<f32>,
    /// Maximum acceptable accuracy loss
    pub max_accuracy_loss: Option<f32>,
    /// Critical layers that must maintain high precision
    pub critical_layers: Vec<String>,
    /// Tolerance for numerical errors
    pub numerical_tolerance: f32,
}

impl Default for AccuracyRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: None,
            max_accuracy_loss: Some(0.05), // 5% max loss
            critical_layers: Vec::new(),
            numerical_tolerance: 1e-6,
        }
    }
}

/// Device capabilities affecting precision decisions
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Supported data types
    pub supported_dtypes: Vec<BitNetDType>,
    /// SIMD capabilities
    pub simd_support: bool,
    /// Hardware acceleration support
    pub hardware_acceleration: bool,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: Option<f32>,
    /// Compute capability
    pub compute_capability: Option<String>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            supported_dtypes: BitNetDType::all_types().to_vec(),
            simd_support: true,
            hardware_acceleration: false,
            memory_bandwidth: None,
            compute_capability: None,
        }
    }
}

/// Central precision manager coordinating all mixed precision operations
#[derive(Debug)]
pub struct PrecisionManager {
    /// Mixed precision configuration
    config: Arc<RwLock<MixedPrecisionConfig>>,
    /// Layer precision manager
    layer_manager: Arc<LayerPrecisionManager>,
    /// Precision converter
    converter: Arc<Mutex<PrecisionConverter>>,
    /// Precision validator
    validator: Arc<PrecisionValidator>,
    /// Policy engine
    policy_engine: Arc<Mutex<PolicyEngine>>,
    /// Current precision context
    context: Arc<RwLock<PrecisionContext>>,
    /// Performance metrics
    metrics: Arc<RwLock<PrecisionMetrics>>,
    /// Memory pool for tensor operations
    memory_pool: Arc<HybridMemoryPool>,
}

impl PrecisionManager {
    /// Create a new precision manager
    pub fn new(config: MixedPrecisionConfig) -> MixedPrecisionResult<Self> {
        let memory_pool = Arc::new(HybridMemoryPool::new()
            .map_err(|e| MixedPrecisionError::MemoryAllocationError(e.to_string()))?);

        let conversion_config = ConversionConfig {
            strategy: ConversionStrategy::Scaled,
            preserve_metadata: true,
            validate_results: config.validation.strict_validation,
            validation_tolerance: config.validation.conversion_tolerance,
            use_simd: config.performance_optimization.enable_simd_optimizations,
            ..Default::default()
        };

        let converter = Arc::new(Mutex::new(PrecisionConverter::new(conversion_config)?));
        let layer_manager = Arc::new(LayerPrecisionManager::new());
        let validator = Arc::new(PrecisionValidator::new());
        let policy_engine = Arc::new(Mutex::new(PolicyEngine::new()));

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            layer_manager,
            converter,
            validator,
            policy_engine,
            context: Arc::new(RwLock::new(PrecisionContext::default())),
            metrics: Arc::new(RwLock::new(PrecisionMetrics::default())),
            memory_pool,
        })
    }

    /// Update the precision context
    pub fn update_context(&self, context: PrecisionContext) -> MixedPrecisionResult<()> {
        let mut ctx = self.context.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire context lock".to_string())
        })?;
        *ctx = context;
        Ok(())
    }

    /// Get the current precision context
    pub fn get_context(&self) -> PrecisionContext {
        self.context.read().unwrap().clone()
    }

    /// Register a layer with the precision manager
    pub fn register_layer(&self, spec: LayerPrecisionSpec) -> MixedPrecisionResult<()> {
        // Validate the layer specification
        self.validator.validate_layer_spec(&spec)?;
        
        // Register with layer manager
        self.layer_manager.register_layer(spec)?;
        
        Ok(())
    }

    /// Get the optimal precision for a tensor operation
    pub fn get_optimal_precision(
        &self,
        layer_id: &str,
        component_type: ComponentType,
        tensor: &BitNetTensor,
    ) -> MixedPrecisionResult<BitNetDType> {
        let context = self.get_context();
        let config = self.config.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire config lock".to_string())
        })?;

        // Check if mixed precision is enabled
        if !config.enabled {
            return Ok(BitNetDType::F32); // Fall back to full precision
        }

        // Get layer specification
        if let Some(layer_spec) = self.layer_manager.get_layer_spec(layer_id) {
            // Use layer-specific precision
            let precision = layer_spec.get_component_precision(component_type);
            
            // Apply policy constraints
            let mut policy_engine = self.policy_engine.lock().map_err(|_| {
                MixedPrecisionError::InvalidConfiguration("Failed to acquire policy engine lock".to_string())
            })?;
            let policy_precision = policy_engine.apply_policies(
                precision,
                &layer_spec,
                component_type,
                &context,
            )?;
            
            Ok(policy_precision)
        } else {
            // Use global configuration
            Ok(config.get_component_precision(component_type))
        }
    }

    /// Convert a tensor to the optimal precision for a specific operation
    pub fn convert_for_operation(
        &self,
        tensor: &BitNetTensor,
        layer_id: &str,
        component_type: ComponentType,
    ) -> MixedPrecisionResult<BitNetTensor> {
        let target_precision = self.get_optimal_precision(layer_id, component_type, tensor)?;
        
        // Check if conversion is needed
        if tensor.dtype() == target_precision {
            return Ok(tensor.clone());
        }

        // Perform conversion
        let start_time = std::time::Instant::now();
        let mut converter = self.converter.lock().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire converter lock".to_string())
        })?;
        
        let converted_tensor = converter.convert_tensor(tensor, target_precision)?;
        let conversion_time = start_time.elapsed();

        // Record metrics
        self.record_conversion_metrics(
            tensor.dtype(),
            target_precision,
            conversion_time,
            tensor.size_bytes(),
            converted_tensor.size_bytes(),
        )?;

        Ok(converted_tensor)
    }

    /// Optimize precision across all layers for a specific objective
    pub fn optimize_precision(&self, objective: OptimizationObjective) -> MixedPrecisionResult<()> {
        match objective {
            OptimizationObjective::Memory { target_reduction } => {
                self.optimize_for_memory(target_reduction)
            }
            OptimizationObjective::Speed { target_speedup } => {
                self.optimize_for_speed(target_speedup)
            }
            OptimizationObjective::Accuracy { min_accuracy } => {
                self.optimize_for_accuracy(min_accuracy)
            }
            OptimizationObjective::Balanced { memory_weight, speed_weight, accuracy_weight } => {
                self.optimize_balanced(memory_weight, speed_weight, accuracy_weight)
            }
        }
    }

    /// Optimize precision for memory efficiency
    fn optimize_for_memory(&self, target_reduction: f32) -> MixedPrecisionResult<()> {
        let optimizations = self.layer_manager.optimize_for_memory(target_reduction)?;
        self.layer_manager.apply_optimizations(optimizations)?;
        Ok(())
    }

    /// Optimize precision for speed
    fn optimize_for_speed(&self, target_speedup: f32) -> MixedPrecisionResult<()> {
        // Implement speed optimization logic
        // This would analyze layer performance and adjust precisions accordingly
        Ok(())
    }

    /// Optimize precision for accuracy
    fn optimize_for_accuracy(&self, min_accuracy: f32) -> MixedPrecisionResult<()> {
        // Implement accuracy optimization logic
        // This would ensure critical layers maintain high precision
        Ok(())
    }

    /// Optimize precision with balanced objectives
    fn optimize_balanced(
        &self,
        memory_weight: f32,
        speed_weight: f32,
        accuracy_weight: f32,
    ) -> MixedPrecisionResult<()> {
        // Implement balanced optimization logic
        // This would use a weighted scoring function to balance objectives
        Ok(())
    }

    /// Analyze the current precision configuration
    pub fn analyze_configuration(&self) -> MixedPrecisionResult<PrecisionAnalysis> {
        let layer_analysis = self.layer_manager.analyze_precision_impact();
        let metrics = self.metrics.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire metrics lock".to_string())
        })?;

        Ok(PrecisionAnalysis {
            memory_savings: layer_analysis.average_memory_savings,
            accuracy_impact: layer_analysis.average_accuracy_impact,
            conversion_overhead: metrics.average_conversion_time_ms(),
            precision_distribution: layer_analysis.precision_distribution,
            total_layers: layer_analysis.total_layers,
            recommendations: self.generate_recommendations()?,
        })
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> MixedPrecisionResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Analyze current configuration and suggest improvements
        let analysis = self.layer_manager.analyze_precision_impact();
        
        if analysis.average_memory_savings < 0.2 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::MemoryOptimization,
                description: "Consider using more aggressive quantization for better memory efficiency".to_string(),
                expected_benefit: "20-40% memory reduction".to_string(),
                risk_level: RiskLevel::Medium,
            });
        }

        if analysis.average_accuracy_impact > 0.1 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::AccuracyImprovement,
                description: "Some layers may benefit from higher precision to improve accuracy".to_string(),
                expected_benefit: "5-10% accuracy improvement".to_string(),
                risk_level: RiskLevel::Low,
            });
        }

        Ok(recommendations)
    }

    /// Record conversion metrics
    fn record_conversion_metrics(
        &self,
        from_precision: BitNetDType,
        to_precision: BitNetDType,
        duration: std::time::Duration,
        original_size: usize,
        converted_size: usize,
    ) -> MixedPrecisionResult<()> {
        let mut metrics = self.metrics.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire metrics lock".to_string())
        })?;

        metrics.record_conversion(from_precision, to_precision, duration, original_size, converted_size);
        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PrecisionMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) -> MixedPrecisionResult<()> {
        let mut metrics = self.metrics.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire metrics lock".to_string())
        })?;
        *metrics = PrecisionMetrics::default();
        Ok(())
    }

    /// Validate the current configuration
    pub fn validate_configuration(&self) -> MixedPrecisionResult<()> {
        let config = self.config.read().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire config lock".to_string())
        })?;

        config.validate()?;
        self.layer_manager.validate_all_layers()?;
        Ok(())
    }

    /// Update the mixed precision configuration
    pub fn update_configuration(&self, new_config: MixedPrecisionConfig) -> MixedPrecisionResult<()> {
        new_config.validate()?;
        
        let mut config = self.config.write().map_err(|_| {
            MixedPrecisionError::InvalidConfiguration("Failed to acquire config lock".to_string())
        })?;
        *config = new_config;
        Ok(())
    }
}

/// Optimization objectives for precision management
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Optimize for memory efficiency
    Memory { target_reduction: f32 },
    /// Optimize for speed
    Speed { target_speedup: f32 },
    /// Optimize for accuracy
    Accuracy { min_accuracy: f32 },
    /// Balanced optimization
    Balanced { memory_weight: f32, speed_weight: f32, accuracy_weight: f32 },
}

/// Performance metrics for precision operations
#[derive(Debug, Clone, Default)]
pub struct PrecisionMetrics {
    /// Total number of conversions
    pub total_conversions: usize,
    /// Total conversion time
    pub total_conversion_time_ms: f32,
    /// Memory savings achieved
    pub total_memory_saved_bytes: i64,
    /// Conversion counts by precision pair
    pub conversion_counts: HashMap<(BitNetDType, BitNetDType), usize>,
    /// Average conversion times
    pub average_conversion_times: HashMap<(BitNetDType, BitNetDType), f32>,
}

impl PrecisionMetrics {
    /// Record a conversion
    pub fn record_conversion(
        &mut self,
        from_precision: BitNetDType,
        to_precision: BitNetDType,
        duration: std::time::Duration,
        original_size: usize,
        converted_size: usize,
    ) {
        self.total_conversions += 1;
        let duration_ms = duration.as_secs_f32() * 1000.0;
        self.total_conversion_time_ms += duration_ms;
        self.total_memory_saved_bytes += original_size as i64 - converted_size as i64;

        let precision_pair = (from_precision, to_precision);
        *self.conversion_counts.entry(precision_pair).or_insert(0) += 1;
        
        let current_avg = self.average_conversion_times.get(&precision_pair).copied().unwrap_or(0.0);
        let count = self.conversion_counts[&precision_pair] as f32;
        let new_avg = (current_avg * (count - 1.0) + duration_ms) / count;
        self.average_conversion_times.insert(precision_pair, new_avg);
    }

    /// Get average conversion time
    pub fn average_conversion_time_ms(&self) -> f32 {
        if self.total_conversions > 0 {
            self.total_conversion_time_ms / self.total_conversions as f32
        } else {
            0.0
        }
    }

    /// Get memory efficiency in MB
    pub fn memory_efficiency_mb(&self) -> f32 {
        self.total_memory_saved_bytes as f32 / (1024.0 * 1024.0)
    }
}

/// Analysis results for precision configuration
#[derive(Debug, Clone)]
pub struct PrecisionAnalysis {
    /// Average memory savings across layers
    pub memory_savings: f32,
    /// Average accuracy impact
    pub accuracy_impact: f32,
    /// Conversion overhead
    pub conversion_overhead: f32,
    /// Distribution of precisions
    pub precision_distribution: HashMap<BitNetDType, usize>,
    /// Total number of layers
    pub total_layers: usize,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Expected benefit
    pub expected_benefit: String,
    /// Risk level
    pub risk_level: RiskLevel,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationType {
    /// Memory optimization
    MemoryOptimization,
    /// Speed optimization
    SpeedOptimization,
    /// Accuracy improvement
    AccuracyImprovement,
    /// Configuration adjustment
    ConfigurationAdjustment,
}

impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendationType::MemoryOptimization => write!(f, "Memory Optimization"),
            RecommendationType::SpeedOptimization => write!(f, "Speed Optimization"),
            RecommendationType::AccuracyImprovement => write!(f, "Accuracy Improvement"),
            RecommendationType::ConfigurationAdjustment => write!(f, "Configuration Adjustment"),
        }
    }
}

/// Risk levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_context_default() {
        let context = PrecisionContext::default();
        assert!(context.current_layer.is_none());
        assert!(context.current_operation.is_none());
        assert_eq!(context.accuracy_requirements.numerical_tolerance, 1e-6);
    }

    #[test]
    fn test_performance_requirements() {
        let requirements = PerformanceRequirements::default();
        assert!(requirements.target_throughput.is_none());
        assert!(requirements.max_latency_ms.is_none());
    }

    #[test]
    fn test_precision_manager_creation() {
        let config = MixedPrecisionConfig::default();
        let manager = PrecisionManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_optimization_objective() {
        let memory_obj = OptimizationObjective::Memory { target_reduction: 0.5 };
        let speed_obj = OptimizationObjective::Speed { target_speedup: 2.0 };
        let accuracy_obj = OptimizationObjective::Accuracy { min_accuracy: 0.95 };
        
        // Test that different objectives are different
        match memory_obj {
            OptimizationObjective::Memory { target_reduction } => {
                assert_eq!(target_reduction, 0.5);
            }
            _ => panic!("Wrong objective type"),
        }
    }

    #[test]
    fn test_precision_metrics() {
        let mut metrics = PrecisionMetrics::default();
        assert_eq!(metrics.total_conversions, 0);
        assert_eq!(metrics.average_conversion_time_ms(), 0.0);
        
        metrics.record_conversion(
            BitNetDType::F32,
            BitNetDType::I8,
            std::time::Duration::from_millis(10),
            1000,
            250,
        );
        
        assert_eq!(metrics.total_conversions, 1);
        assert!(metrics.average_conversion_time_ms() > 0.0);
        assert!(metrics.memory_efficiency_mb() > 0.0);
    }

    #[test]
    fn test_device_capabilities() {
        let capabilities = DeviceCapabilities::default();
        assert!(capabilities.simd_support);
        assert!(!capabilities.hardware_acceleration);
        assert!(!capabilities.supported_dtypes.is_empty());
    }

    #[test]
    fn test_recommendation_types() {
        assert_ne!(RecommendationType::MemoryOptimization, RecommendationType::SpeedOptimization);
        assert_ne!(RiskLevel::Low, RiskLevel::High);
    }
}