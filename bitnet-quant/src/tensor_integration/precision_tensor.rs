//! Mixed Precision Tensor Operations
//!
//! This module provides mixed precision tensor operations that allow different
//! components of BitNet models to use different quantization precisions optimized
//! for either accuracy or performance, with intelligent precision selection.

use candle_core::Device;
use std::collections::HashMap;

use bitnet_core::{auto_select_device, BitNetTensor, TensorShape};

use crate::quantization::{QuantizationPrecision, QuantizationStrategy};

use super::{
    quantized_tensor::{QuantizedTensor, QuantizedTensorConfig},
    TensorIntegrationResult,
};

/// Configuration for mixed precision operations
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PrecisionConfig {
    /// Default precision for operations
    pub default_precision: QuantizationPrecision,

    /// Precision policies for different layers
    pub layer_policies: HashMap<String, LayerPrecisionPolicy>,

    /// Automatic precision selection enabled
    pub auto_precision: bool,

    /// Performance optimization priority
    pub performance_priority: f32, // 0.0 = accuracy, 1.0 = performance

    /// Memory constraint (MB)
    pub memory_constraint: Option<usize>,

    /// Target device
    pub device: Option<Device>,

    /// Enable dynamic precision adjustment
    pub dynamic_adjustment: bool,

    /// Precision monitoring enabled
    pub enable_monitoring: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: QuantizationPrecision::OneFiveFiveBit,
            layer_policies: HashMap::new(),
            auto_precision: true,
            performance_priority: 0.5,
            memory_constraint: None,
            device: None,
            dynamic_adjustment: true,
            enable_monitoring: true,
        }
    }
}

/// Precision policy for specific layers
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayerPrecisionPolicy {
    /// Layer identifier
    pub layer_id: String,

    /// Layer type
    pub layer_type: LayerType,

    /// Assigned precision
    pub precision: QuantizationPrecision,

    /// Strategy for this layer
    pub strategy: QuantizationStrategy,

    /// Priority level (higher = more important for accuracy)
    pub priority: f32,

    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Types of layers in BitNet models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// BitLinear layer
    BitLinear,

    /// Attention layer
    Attention,

    /// Feed-forward layer
    FeedForward,

    /// LayerNorm
    LayerNorm,

    /// Embedding layer
    Embedding,

    /// Output layer
    Output,

    /// Custom layer type
    Custom,
}

/// Performance characteristics of a layer
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PerformanceProfile {
    /// Compute intensity (FLOPs per element)
    pub compute_intensity: f32,

    /// Memory bandwidth requirement
    pub memory_bandwidth: f32,

    /// Parallelization efficiency
    pub parallel_efficiency: f32,

    /// Accuracy sensitivity to quantization
    pub accuracy_sensitivity: f32,
}

/// Precision policy for different optimization strategies
#[derive(Debug, Clone)]
pub enum PrecisionPolicy {
    /// Optimize for maximum accuracy
    AccuracyOptimized,

    /// Optimize for maximum performance
    PerformanceOptimized,

    /// Balanced accuracy and performance
    Balanced,

    /// Custom precision assignments
    Custom(HashMap<LayerType, QuantizationPrecision>),

    /// Dynamic precision based on runtime metrics
    Dynamic(DynamicPrecisionConfig),
}

/// Configuration for dynamic precision adjustment
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DynamicPrecisionConfig {
    /// Threshold for precision adjustment
    pub adjustment_threshold: f32,

    /// Minimum precision allowed
    pub min_precision: QuantizationPrecision,

    /// Maximum precision allowed
    pub max_precision: QuantizationPrecision,

    /// Adjustment frequency (in iterations)
    pub adjustment_frequency: usize,

    /// Performance metrics to monitor
    pub monitored_metrics: Vec<PerformanceMetric>,
}

/// Performance metrics for precision selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerformanceMetric {
    /// Inference latency
    Latency,

    /// Memory usage
    Memory,

    /// Throughput (samples/sec)
    Throughput,

    /// Accuracy/loss
    Accuracy,

    /// Energy consumption
    Energy,
}

/// Errors specific to mixed precision operations
#[derive(Debug, thiserror::Error)]
pub enum MixedPrecisionError {
    #[error("Precision selection failed: {message}")]
    PrecisionSelection { message: String },

    #[error("Layer policy conflict: {layer_id}")]
    LayerPolicyConflict { layer_id: String },

    #[error("Unsupported precision combination: {details}")]
    UnsupportedCombination { details: String },

    #[error("Dynamic adjustment failed: {message}")]
    DynamicAdjustment { message: String },

    #[error("Performance monitoring error: {message}")]
    PerformanceMonitoring { message: String },
}

/// Precision selector for automatic precision assignment
#[derive(Debug)]
#[allow(dead_code)]
pub struct PrecisionSelector {
    /// Configuration
    config: PrecisionConfig,

    /// Current precision assignments
    precision_assignments: HashMap<String, QuantizationPrecision>,

    /// Performance monitor
    performance_monitor: Option<PerformanceMonitor>,

    /// Device for operations
    device: Device,
}

/// Performance monitoring for precision selection
#[derive(Debug)]
#[allow(dead_code)]
pub struct PerformanceMonitor {
    /// Monitored metrics
    metrics: HashMap<PerformanceMetric, Vec<f32>>,

    /// Metric thresholds
    thresholds: HashMap<PerformanceMetric, f32>,

    /// Current iteration
    current_iteration: usize,

    /// Last adjustment iteration
    last_adjustment: usize,
}

/// Optimized precision configurations
pub struct PerformanceOptimizedPrecision;
pub struct AccuracyOptimizedPrecision;

impl PerformanceOptimizedPrecision {
    /// Get performance-optimized precision policy
    pub fn get_policy() -> PrecisionPolicy {
        let mut custom_precision = HashMap::new();
        custom_precision.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        custom_precision.insert(LayerType::Attention, QuantizationPrecision::OneBit);
        custom_precision.insert(
            LayerType::FeedForward,
            QuantizationPrecision::OneFiveFiveBit,
        );
        custom_precision.insert(LayerType::LayerNorm, QuantizationPrecision::EightBit);
        custom_precision.insert(LayerType::Embedding, QuantizationPrecision::FourBit);
        custom_precision.insert(LayerType::Output, QuantizationPrecision::EightBit);

        PrecisionPolicy::Custom(custom_precision)
    }
}

impl AccuracyOptimizedPrecision {
    /// Get accuracy-optimized precision policy
    pub fn get_policy() -> PrecisionPolicy {
        let mut custom_precision = HashMap::new();
        custom_precision.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        custom_precision.insert(LayerType::Attention, QuantizationPrecision::EightBit);
        custom_precision.insert(
            LayerType::FeedForward,
            QuantizationPrecision::OneFiveFiveBit,
        );
        custom_precision.insert(LayerType::LayerNorm, QuantizationPrecision::EightBit);
        custom_precision.insert(LayerType::Embedding, QuantizationPrecision::EightBit);
        custom_precision.insert(LayerType::Output, QuantizationPrecision::EightBit);

        PrecisionPolicy::Custom(custom_precision)
    }
}

/// Mixed precision tensor operations
pub trait PrecisionTensorOps {
    /// Apply mixed precision to tensor
    fn apply_mixed_precision(
        &self,
        tensor: &BitNetTensor,
        layer_type: LayerType,
    ) -> TensorIntegrationResult<QuantizedTensor>;

    /// Convert between precisions
    fn convert_precision(
        &self,
        tensor: &QuantizedTensor,
        target_precision: QuantizationPrecision,
    ) -> TensorIntegrationResult<QuantizedTensor>;

    /// Select optimal precision for layer
    fn select_precision(
        &self,
        layer_type: LayerType,
        tensor_shape: &TensorShape,
    ) -> TensorIntegrationResult<QuantizationPrecision>;

    /// Update precision policy based on performance
    fn update_precision_policy(
        &mut self,
        performance_data: &PerformanceData,
    ) -> TensorIntegrationResult<()>;
}

/// Performance data for precision adjustment
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PerformanceData {
    /// Layer performance metrics
    pub layer_metrics: HashMap<String, LayerMetrics>,

    /// Overall model metrics
    pub model_metrics: ModelMetrics,

    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Layer-specific performance metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayerMetrics {
    /// Layer identifier
    pub layer_id: String,

    /// Execution time
    pub execution_time: f32,

    /// Memory usage
    pub memory_usage: usize,

    /// Accuracy contribution
    pub accuracy_contribution: f32,

    /// Current precision
    pub current_precision: QuantizationPrecision,
}

/// Model-level performance metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelMetrics {
    /// Total inference time
    pub inference_time: f32,

    /// Total memory usage
    pub memory_usage: usize,

    /// Overall accuracy
    pub accuracy: f32,

    /// Throughput
    pub throughput: f32,

    /// Energy consumption
    pub energy: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu_usage: f32,

    /// GPU utilization (if available)
    pub gpu_usage: Option<f32>,

    /// Memory utilization
    pub memory_usage: f32,

    /// Memory bandwidth utilization
    pub bandwidth_usage: f32,
}

/// Main mixed precision tensor implementation
#[derive(Debug)]
#[allow(dead_code)]
pub struct MixedPrecisionTensor {
    /// Configuration
    config: PrecisionConfig,

    /// Precision selector
    precision_selector: PrecisionSelector,

    /// Current tensor with mixed precision
    tensors: HashMap<String, QuantizedTensor>,

    /// Device for operations
    device: Device,
}

/// Mixed precision tensor operations implementation
#[derive(Debug)]
#[allow(dead_code)]
pub struct MixedPrecisionTensorOpsImpl {
    /// Configuration
    config: PrecisionConfig,

    /// Precision assignments
    precision_assignments: HashMap<LayerType, QuantizationPrecision>,

    /// Performance monitor
    performance_monitor: Option<PerformanceMonitor>,

    /// Device for operations
    device: Device,
}

impl MixedPrecisionTensorOpsImpl {
    /// Create new mixed precision operations
    pub fn new(config: PrecisionConfig) -> Self {
        let device = config.device.clone().unwrap_or_else(auto_select_device);

        let precision_assignments = Self::initialize_precision_assignments(&config);

        let performance_monitor = if config.enable_monitoring {
            Some(PerformanceMonitor::new())
        } else {
            None
        };

        Self {
            config,
            precision_assignments,
            performance_monitor,
            device,
        }
    }

    /// Initialize precision assignments based on policy
    fn initialize_precision_assignments(
        config: &PrecisionConfig,
    ) -> HashMap<LayerType, QuantizationPrecision> {
        let mut assignments = HashMap::new();

        // Set default precisions
        assignments.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        assignments.insert(LayerType::Attention, config.default_precision);
        assignments.insert(LayerType::FeedForward, config.default_precision);
        assignments.insert(LayerType::LayerNorm, QuantizationPrecision::EightBit);
        assignments.insert(LayerType::Embedding, QuantizationPrecision::FourBit);
        assignments.insert(LayerType::Output, QuantizationPrecision::EightBit);
        assignments.insert(LayerType::Custom, config.default_precision);

        // Apply layer-specific policies
        for (layer_id, policy) in &config.layer_policies {
            if let Ok(layer_type) = Self::parse_layer_type(layer_id) {
                assignments.insert(layer_type, policy.precision);
            }
        }

        assignments
    }

    /// Parse layer type from layer identifier
    fn parse_layer_type(layer_id: &str) -> Result<LayerType, ()> {
        if layer_id.contains("bitlinear") {
            Ok(LayerType::BitLinear)
        } else if layer_id.contains("attention") {
            Ok(LayerType::Attention)
        } else if layer_id.contains("ffn") || layer_id.contains("feedforward") {
            Ok(LayerType::FeedForward)
        } else if layer_id.contains("layernorm") || layer_id.contains("norm") {
            Ok(LayerType::LayerNorm)
        } else if layer_id.contains("embedding") || layer_id.contains("embed") {
            Ok(LayerType::Embedding)
        } else if layer_id.contains("output") || layer_id.contains("head") {
            Ok(LayerType::Output)
        } else {
            Ok(LayerType::Custom)
        }
    }

    /// Apply precision policy
    pub fn apply_precision_policy(
        &mut self,
        policy: PrecisionPolicy,
    ) -> TensorIntegrationResult<()> {
        match policy {
            PrecisionPolicy::AccuracyOptimized => {
                self.precision_assignments = Self::get_accuracy_optimized_assignments();
            }
            PrecisionPolicy::PerformanceOptimized => {
                self.precision_assignments = Self::get_performance_optimized_assignments();
            }
            PrecisionPolicy::Balanced => {
                self.precision_assignments = Self::get_balanced_assignments();
            }
            PrecisionPolicy::Custom(assignments) => {
                self.precision_assignments = assignments;
            }
            PrecisionPolicy::Dynamic(config) => {
                self.setup_dynamic_precision(config)?;
            }
        }

        Ok(())
    }

    fn get_accuracy_optimized_assignments() -> HashMap<LayerType, QuantizationPrecision> {
        let mut assignments = HashMap::new();
        assignments.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        assignments.insert(LayerType::Attention, QuantizationPrecision::EightBit);
        assignments.insert(
            LayerType::FeedForward,
            QuantizationPrecision::OneFiveFiveBit,
        );
        assignments.insert(LayerType::LayerNorm, QuantizationPrecision::EightBit);
        assignments.insert(LayerType::Embedding, QuantizationPrecision::EightBit);
        assignments.insert(LayerType::Output, QuantizationPrecision::EightBit);
        assignments
    }

    fn get_performance_optimized_assignments() -> HashMap<LayerType, QuantizationPrecision> {
        let mut assignments = HashMap::new();
        assignments.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        assignments.insert(LayerType::Attention, QuantizationPrecision::OneBit);
        assignments.insert(
            LayerType::FeedForward,
            QuantizationPrecision::OneFiveFiveBit,
        );
        assignments.insert(LayerType::LayerNorm, QuantizationPrecision::FourBit);
        assignments.insert(LayerType::Embedding, QuantizationPrecision::FourBit);
        assignments.insert(LayerType::Output, QuantizationPrecision::FourBit);
        assignments
    }

    fn get_balanced_assignments() -> HashMap<LayerType, QuantizationPrecision> {
        let mut assignments = HashMap::new();
        assignments.insert(LayerType::BitLinear, QuantizationPrecision::OneFiveFiveBit);
        assignments.insert(LayerType::Attention, QuantizationPrecision::FourBit);
        assignments.insert(
            LayerType::FeedForward,
            QuantizationPrecision::OneFiveFiveBit,
        );
        assignments.insert(LayerType::LayerNorm, QuantizationPrecision::EightBit);
        assignments.insert(LayerType::Embedding, QuantizationPrecision::FourBit);
        assignments.insert(LayerType::Output, QuantizationPrecision::EightBit);
        assignments
    }

    fn setup_dynamic_precision(
        &mut self,
        _config: DynamicPrecisionConfig,
    ) -> TensorIntegrationResult<()> {
        // Initialize dynamic precision adjustment
        // This would set up monitoring and automatic adjustment mechanisms
        Ok(())
    }

    /// Get current precision assignment
    pub fn get_precision_for_layer(&self, layer_type: LayerType) -> QuantizationPrecision {
        self.precision_assignments
            .get(&layer_type)
            .cloned()
            .unwrap_or(self.config.default_precision)
    }

    /// Update performance metrics
    pub fn update_performance_metrics(
        &mut self,
        data: &PerformanceData,
    ) -> TensorIntegrationResult<()> {
        if let Some(ref mut monitor) = self.performance_monitor {
            monitor.update_metrics(data)?;

            if self.config.dynamic_adjustment {
                self.adjust_precision_based_on_performance(data)?;
            }
        }

        Ok(())
    }

    fn adjust_precision_based_on_performance(
        &mut self,
        data: &PerformanceData,
    ) -> TensorIntegrationResult<()> {
        // Analyze performance data and adjust precisions
        for (layer_id, metrics) in &data.layer_metrics {
            if let Ok(layer_type) = Self::parse_layer_type(layer_id) {
                let current_precision = self.get_precision_for_layer(layer_type);

                // Simple adjustment logic based on performance
                let new_precision = if metrics.execution_time > 10.0
                    && metrics.accuracy_contribution < 0.1
                {
                    // High execution time, low accuracy contribution -> reduce precision
                    self.reduce_precision(current_precision)
                } else if metrics.accuracy_contribution > 0.8 && data.model_metrics.accuracy < 0.9 {
                    // High accuracy contribution, low overall accuracy -> increase precision
                    self.increase_precision(current_precision)
                } else {
                    current_precision
                };

                if new_precision != current_precision {
                    self.precision_assignments.insert(layer_type, new_precision);
                }
            }
        }

        Ok(())
    }

    fn reduce_precision(&self, current: QuantizationPrecision) -> QuantizationPrecision {
        match current {
            QuantizationPrecision::EightBit => QuantizationPrecision::FourBit,
            QuantizationPrecision::FourBit => QuantizationPrecision::TwoBit,
            QuantizationPrecision::TwoBit => QuantizationPrecision::OneBit,
            QuantizationPrecision::OneFiveFiveBit => QuantizationPrecision::OneBit,
            QuantizationPrecision::OneBit => QuantizationPrecision::OneBit,
        }
    }

    fn increase_precision(&self, current: QuantizationPrecision) -> QuantizationPrecision {
        match current {
            QuantizationPrecision::OneBit => QuantizationPrecision::OneFiveFiveBit,
            QuantizationPrecision::OneFiveFiveBit => QuantizationPrecision::TwoBit,
            QuantizationPrecision::TwoBit => QuantizationPrecision::FourBit,
            QuantizationPrecision::FourBit => QuantizationPrecision::EightBit,
            QuantizationPrecision::EightBit => QuantizationPrecision::EightBit,
        }
    }
}

impl PrecisionTensorOps for MixedPrecisionTensorOpsImpl {
    fn apply_mixed_precision(
        &self,
        tensor: &BitNetTensor,
        layer_type: LayerType,
    ) -> TensorIntegrationResult<QuantizedTensor> {
        let precision = self.get_precision_for_layer(layer_type);

        let config = QuantizedTensorConfig {
            precision,
            strategy: QuantizationStrategy::Symmetric,
            device: Some(tensor.device().clone()),
            use_memory_pool: true,
            enable_compression: true,
            compression_threshold: 0.1,
        };

        QuantizedTensor::from_bitnet_tensor(tensor.clone(), config)
    }

    fn convert_precision(
        &self,
        tensor: &QuantizedTensor,
        target_precision: QuantizationPrecision,
    ) -> TensorIntegrationResult<QuantizedTensor> {
        if tensor.precision() == target_precision {
            return Ok(tensor.clone());
        }

        // Dequantize to full precision
        let full_precision = tensor.to_bitnet_tensor()?;

        // Requantize with target precision
        let config = QuantizedTensorConfig {
            precision: target_precision,
            strategy: QuantizationStrategy::Symmetric,
            device: Some(tensor.device().clone()),
            use_memory_pool: true,
            enable_compression: true,
            compression_threshold: 0.1,
        };

        QuantizedTensor::from_bitnet_tensor(full_precision, config)
    }

    fn select_precision(
        &self,
        layer_type: LayerType,
        _tensor_shape: &TensorShape,
    ) -> TensorIntegrationResult<QuantizationPrecision> {
        Ok(self.get_precision_for_layer(layer_type))
    }

    fn update_precision_policy(
        &mut self,
        performance_data: &PerformanceData,
    ) -> TensorIntegrationResult<()> {
        self.update_performance_metrics(performance_data)
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(PerformanceMetric::Latency, 10.0);
        thresholds.insert(PerformanceMetric::Memory, 1000.0);
        thresholds.insert(PerformanceMetric::Throughput, 100.0);
        thresholds.insert(PerformanceMetric::Accuracy, 0.9);
        thresholds.insert(PerformanceMetric::Energy, 50.0);

        Self {
            metrics: HashMap::new(),
            thresholds,
            current_iteration: 0,
            last_adjustment: 0,
        }
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, data: &PerformanceData) -> TensorIntegrationResult<()> {
        // Update latency metrics
        self.add_metric(
            PerformanceMetric::Latency,
            data.model_metrics.inference_time,
        );

        // Update memory metrics
        self.add_metric(
            PerformanceMetric::Memory,
            data.model_metrics.memory_usage as f32,
        );

        // Update throughput metrics
        self.add_metric(PerformanceMetric::Throughput, data.model_metrics.throughput);

        // Update accuracy metrics
        self.add_metric(PerformanceMetric::Accuracy, data.model_metrics.accuracy);

        // Update energy metrics
        self.add_metric(PerformanceMetric::Energy, data.model_metrics.energy);

        self.current_iteration += 1;

        Ok(())
    }

    fn add_metric(&mut self, metric: PerformanceMetric, value: f32) {
        self.metrics.entry(metric).or_default().push(value);

        // Keep only recent metrics (last 100 iterations)
        if let Some(values) = self.metrics.get_mut(&metric) {
            if values.len() > 100 {
                values.remove(0);
            }
        }
    }

    /// Check if adjustment is needed
    pub fn should_adjust(&self) -> bool {
        self.current_iteration - self.last_adjustment >= 10 // Adjust every 10 iterations
    }

    /// Get average metric value
    pub fn get_average_metric(&self, metric: PerformanceMetric) -> Option<f32> {
        self.metrics
            .get(&metric)
            .map(|values| values.iter().sum::<f32>() / values.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_config_default() {
        let config = PrecisionConfig::default();
        assert_eq!(
            config.default_precision,
            QuantizationPrecision::OneFiveFiveBit
        );
        assert!(config.auto_precision);
        assert_eq!(config.performance_priority, 0.5);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_layer_type_parsing() {
        assert_eq!(
            MixedPrecisionTensorOpsImpl::parse_layer_type("bitlinear_0"),
            Ok(LayerType::BitLinear)
        );
        assert_eq!(
            MixedPrecisionTensorOpsImpl::parse_layer_type("attention_layer"),
            Ok(LayerType::Attention)
        );
        assert_eq!(
            MixedPrecisionTensorOpsImpl::parse_layer_type("ffn_block"),
            Ok(LayerType::FeedForward)
        );
        assert_eq!(
            MixedPrecisionTensorOpsImpl::parse_layer_type("layernorm_1"),
            Ok(LayerType::LayerNorm)
        );
    }

    #[test]
    fn test_precision_assignments() {
        let config = PrecisionConfig::default();
        let ops = MixedPrecisionTensorOpsImpl::new(config);

        assert_eq!(
            ops.get_precision_for_layer(LayerType::BitLinear),
            QuantizationPrecision::OneFiveFiveBit
        );
        assert_eq!(
            ops.get_precision_for_layer(LayerType::LayerNorm),
            QuantizationPrecision::EightBit
        );
        assert_eq!(
            ops.get_precision_for_layer(LayerType::Embedding),
            QuantizationPrecision::FourBit
        );
    }

    #[test]
    fn test_precision_adjustment() {
        let config = PrecisionConfig::default();
        let ops = MixedPrecisionTensorOpsImpl::new(config);

        let reduced = ops.reduce_precision(QuantizationPrecision::EightBit);
        assert_eq!(reduced, QuantizationPrecision::FourBit);

        let increased = ops.increase_precision(QuantizationPrecision::OneBit);
        assert_eq!(increased, QuantizationPrecision::OneFiveFiveBit);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        let data = PerformanceData {
            layer_metrics: HashMap::new(),
            model_metrics: ModelMetrics {
                inference_time: 5.0,
                memory_usage: 500,
                accuracy: 0.95,
                throughput: 200.0,
                energy: 25.0,
            },
            resource_utilization: ResourceUtilization {
                cpu_usage: 0.7,
                gpu_usage: Some(0.8),
                memory_usage: 0.6,
                bandwidth_usage: 0.5,
            },
        };

        monitor.update_metrics(&data).unwrap();

        assert_eq!(
            monitor.get_average_metric(PerformanceMetric::Latency),
            Some(5.0)
        );
        assert_eq!(
            monitor.get_average_metric(PerformanceMetric::Accuracy),
            Some(0.95)
        );
    }
}
