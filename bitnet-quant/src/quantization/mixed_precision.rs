//! Mixed Precision Quantization Integration
//!
//! This module integrates the mixed precision system with the BitNet quantization pipeline,
//! providing seamless precision-aware quantization operations.

use super::{
    QuantizationPrecision, QuantizationStrategy, QuantizationConfig, QuantizationStats,
    QuantizationResult, Quantizer, QuantizationError,
    weights::{WeightQuantizer, WeightQuantizationConfig, QuantizedWeight, TernaryMethod},
    activations::{ActivationQuantizer, ActivationQuantizationConfig, QuantizedActivation},
};
use bitnet_core::mixed_precision::{
    MixedPrecisionConfig, LayerPrecisionConfig, ComponentPrecisionConfig,
    LayerType, ComponentType, MixedPrecisionStrategy, MixedPrecisionError,
    PrecisionManager, LayerPrecisionSpec, PrecisionConverter,
    conversion::ConversionConfig,
};
use bitnet_core::memory::tensor::{BitNetDType, BitNetTensor};
use candle_core::{Tensor, Device};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mixed precision quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionQuantizationConfig {
    /// Base mixed precision configuration
    pub mixed_precision: MixedPrecisionConfig,
    /// Weight quantization configuration
    pub weight_quantization: WeightQuantizationConfig,
    /// Activation quantization configuration
    pub activation_quantization: ActivationQuantizationConfig,
    /// Whether to enable automatic precision adjustment
    pub auto_precision_adjustment: bool,
    /// Precision adjustment parameters
    pub adjustment_params: PrecisionAdjustmentParams,
}

impl Default for MixedPrecisionQuantizationConfig {
    fn default() -> Self {
        Self {
            mixed_precision: MixedPrecisionConfig::balanced(),
            weight_quantization: WeightQuantizationConfig::default(),
            activation_quantization: ActivationQuantizationConfig::default(),
            auto_precision_adjustment: false,
            adjustment_params: PrecisionAdjustmentParams::default(),
        }
    }
}

impl MixedPrecisionQuantizationConfig {
    /// Create configuration for BitNet mixed precision quantization
    pub fn bitnet() -> Self {
        Self {
            mixed_precision: MixedPrecisionConfig::balanced(),
            weight_quantization: WeightQuantizationConfig::bitnet(),
            activation_quantization: ActivationQuantizationConfig::bitnet(),
            auto_precision_adjustment: true,
            adjustment_params: PrecisionAdjustmentParams::default(),
        }
    }

    /// Create configuration with custom mixed precision strategy
    pub fn with_strategy(strategy: MixedPrecisionStrategy) -> Self {
        Self {
            mixed_precision: MixedPrecisionConfig::new(strategy),
            ..Default::default()
        }
    }

    /// Enable automatic precision adjustment
    pub fn with_auto_adjustment(mut self, params: PrecisionAdjustmentParams) -> Self {
        self.auto_precision_adjustment = true;
        self.adjustment_params = params;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> QuantizationResult<()> {
        // Validate mixed precision configuration
        self.mixed_precision.validate()
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

        // Validate quantization configurations
        self.weight_quantization.validate()
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

        self.activation_quantization.validate()
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

        // Validate adjustment parameters
        if self.auto_precision_adjustment {
            self.adjustment_params.validate()?;
        }

        Ok(())
    }
}

/// Parameters for automatic precision adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAdjustmentParams {
    /// Accuracy threshold for increasing precision
    pub accuracy_threshold: f32,
    /// Memory pressure threshold for decreasing precision
    pub memory_pressure_threshold: f32,
    /// Performance threshold for precision adjustment
    pub performance_threshold: f32,
    /// Adjustment step size
    pub adjustment_step: usize,
    /// Maximum adjustments per layer
    pub max_adjustments: usize,
    /// Evaluation window size
    pub evaluation_window: usize,
}

impl Default for PrecisionAdjustmentParams {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.95,
            memory_pressure_threshold: 0.8,
            performance_threshold: 0.9,
            adjustment_step: 1,
            max_adjustments: 3,
            evaluation_window: 100,
        }
    }
}

impl PrecisionAdjustmentParams {
    /// Validate the parameters
    pub fn validate(&self) -> QuantizationResult<()> {
        if self.accuracy_threshold <= 0.0 || self.accuracy_threshold > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Accuracy threshold must be between 0 and 1".to_string()
            ));
        }

        if self.memory_pressure_threshold <= 0.0 || self.memory_pressure_threshold > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Memory pressure threshold must be between 0 and 1".to_string()
            ));
        }

        if self.performance_threshold <= 0.0 || self.performance_threshold > 1.0 {
            return Err(QuantizationError::ConfigurationError(
                "Performance threshold must be between 0 and 1".to_string()
            ));
        }

        if self.adjustment_step == 0 {
            return Err(QuantizationError::ConfigurationError(
                "Adjustment step must be greater than 0".to_string()
            ));
        }

        if self.evaluation_window == 0 {
            return Err(QuantizationError::ConfigurationError(
                "Evaluation window must be greater than 0".to_string()
            ));
        }

        Ok(())
    }
}

/// Mixed precision quantizer that integrates precision management with quantization
#[derive(Debug)]
pub struct MixedPrecisionQuantizer {
    /// Configuration
    config: MixedPrecisionQuantizationConfig,
    /// Precision manager
    precision_manager: Arc<PrecisionManager>,
    /// Weight quantizers for different precisions
    weight_quantizers: HashMap<BitNetDType, Box<dyn WeightQuantizer>>,
    /// Activation quantizers for different precisions
    activation_quantizers: HashMap<BitNetDType, Box<dyn ActivationQuantizer>>,
    /// Precision converter
    precision_converter: Arc<Mutex<PrecisionConverter>>,
    /// Quantization statistics
    stats: QuantizationStats,
    /// Device for operations
    device: Device,
}

impl MixedPrecisionQuantizer {
    /// Create a new mixed precision quantizer
    pub fn new(
        config: MixedPrecisionQuantizationConfig,
        device: Device,
    ) -> QuantizationResult<Self> {
        config.validate()?;

        // Create precision manager
        let precision_manager = Arc::new(
            PrecisionManager::new(config.mixed_precision.clone())
                .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?
        );

        // Create precision converter
        let conversion_config = ConversionConfig {
            strategy: bitnet_core::mixed_precision::ConversionStrategy::QuantizationAware,
            preserve_metadata: true,
            validate_results: true,
            validation_tolerance: 1e-6,
            use_simd: true,
            custom_params: HashMap::new(),
        };

        let precision_converter = Arc::new(Mutex::new(
            PrecisionConverter::new(conversion_config)
                .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?
        ));

        // Initialize quantizers for different precisions
        let mut weight_quantizers: HashMap<BitNetDType, Box<dyn WeightQuantizer>> = HashMap::new();
        let mut activation_quantizers: HashMap<BitNetDType, Box<dyn ActivationQuantizer>> = HashMap::new();

        // Create quantizers for all supported precisions
        for &precision in BitNetDType::all_types() {
            // Create weight quantizer configuration for this precision
            let mut weight_config = config.weight_quantization.clone();
            weight_config.base.precision = precision_to_quantization_precision(precision);
            
            if let Ok(quantizer) = super::weights::create_weight_quantizer(weight_config) {
                weight_quantizers.insert(precision, quantizer);
            }

            // Create activation quantizer configuration for this precision
            let mut activation_config = config.activation_quantization.clone();
            activation_config.base.precision = precision_to_quantization_precision(precision);
            
            if let Ok(quantizer) = super::activations::create_activation_quantizer(activation_config) {
                activation_quantizers.insert(precision, quantizer);
            }
        }

        Ok(Self {
            config,
            precision_manager,
            weight_quantizers,
            activation_quantizers,
            precision_converter,
            stats: QuantizationStats::default(),
            device,
        })
    }

    /// Register a layer with the precision manager
    pub fn register_layer(&self, spec: LayerPrecisionSpec) -> QuantizationResult<()> {
        self.precision_manager.register_layer(spec)
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))
    }

    /// Quantize weights with mixed precision
    pub fn quantize_weights(
        &mut self,
        weights: &BitNetTensor,
        layer_id: &str,
    ) -> QuantizationResult<QuantizedWeight> {
        // Get optimal precision for weights
        let target_precision = self.precision_manager
            .get_optimal_precision(layer_id, ComponentType::Weights, weights)
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

        // Convert weights to target precision if needed
        let converted_weights = if weights.dtype() != target_precision {
            self.precision_manager
                .convert_for_operation(weights, layer_id, ComponentType::Weights)
                .map_err(|e| QuantizationError::ConversionError(e.to_string()))?
        } else {
            weights.clone()
        };

        // Get appropriate quantizer for the target precision
        let quantizer = self.weight_quantizers.get(&target_precision)
            .ok_or_else(|| QuantizationError::UnsupportedPrecision(format!("{:?}", target_precision)))?;

        // Convert to candle tensor for quantization
        let candle_weights = converted_weights.to_candle()
            .map_err(|e| QuantizationError::ConversionError(e.to_string()))?;

        // Perform quantization
        let quantized = quantizer.quantize(&candle_weights)?;

        // Update statistics
        self.update_quantization_stats(&quantized);

        Ok(quantized)
    }

    /// Quantize activations with mixed precision
    pub fn quantize_activations(
        &mut self,
        activations: &BitNetTensor,
        layer_id: &str,
    ) -> QuantizationResult<QuantizedActivation> {
        // Get optimal precision for activations
        let target_precision = self.precision_manager
            .get_optimal_precision(layer_id, ComponentType::Activations, activations)
            .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

        // Convert activations to target precision if needed
        let converted_activations = if activations.dtype() != target_precision {
            self.precision_manager
                .convert_for_operation(activations, layer_id, ComponentType::Activations)
                .map_err(|e| QuantizationError::ConversionError(e.to_string()))?
        } else {
            activations.clone()
        };

        // Get appropriate quantizer for the target precision
        let quantizer = self.activation_quantizers.get(&target_precision)
            .ok_or_else(|| QuantizationError::UnsupportedPrecision(format!("{:?}", target_precision)))?;

        // Convert to candle tensor for quantization
        let candle_activations = converted_activations.to_candle()
            .map_err(|e| QuantizationError::ConversionError(e.to_string()))?;

        // Perform quantization
        let quantized = quantizer.quantize(&candle_activations)?;

        Ok(quantized)
    }

    /// Quantize a complete layer with mixed precision
    pub fn quantize_layer(
        &mut self,
        layer_id: &str,
        weights: &BitNetTensor,
        activations: Option<&BitNetTensor>,
        bias: Option<&BitNetTensor>,
    ) -> QuantizationResult<LayerQuantizationResult> {
        let start_time = std::time::Instant::now();

        // Quantize weights
        let quantized_weights = self.quantize_weights(weights, layer_id)?;

        // Quantize activations if provided
        let quantized_activations = if let Some(acts) = activations {
            Some(self.quantize_activations(acts, layer_id)?)
        } else {
            None
        };

        // Quantize bias if provided
        let quantized_bias = if let Some(bias_tensor) = bias {
            // Get optimal precision for bias
            let target_precision = self.precision_manager
                .get_optimal_precision(layer_id, ComponentType::Bias, bias_tensor)
                .map_err(|e| QuantizationError::ConfigurationError(e.to_string()))?;

            // Convert bias to target precision if needed
            let converted_bias = if bias_tensor.dtype() != target_precision {
                self.precision_manager
                    .convert_for_operation(bias_tensor, layer_id, ComponentType::Bias)
                    .map_err(|e| QuantizationError::ConversionError(e.to_string()))?
            } else {
                bias_tensor.clone()
            };

            Some(converted_bias)
        } else {
            None
        };

        let quantization_time = start_time.elapsed();

        // Calculate compression statistics
        let original_size = weights.size_bytes() + 
            activations.map(|a| a.size_bytes()).unwrap_or(0) +
            bias.map(|b| b.size_bytes()).unwrap_or(0);

        let quantized_size = quantized_weights.memory_footprint() +
            quantized_activations.as_ref().map(|a| a.memory_footprint()).unwrap_or(0) +
            quantized_bias.as_ref().map(|b| b.size_bytes()).unwrap_or(0);

        let compression_ratio = original_size as f32 / quantized_size as f32;

        Ok(LayerQuantizationResult {
            layer_id: layer_id.to_string(),
            quantized_weights,
            quantized_activations,
            quantized_bias,
            quantization_time,
            compression_ratio,
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
        })
    }

    /// Optimize precision for a layer based on performance metrics
    pub fn optimize_layer_precision(
        &mut self,
        layer_id: &str,
        performance_metrics: &LayerPerformanceMetrics,
    ) -> QuantizationResult<()> {
        if !self.config.auto_precision_adjustment {
            return Ok(());
        }

        let params = &self.config.adjustment_params;

        // Check if adjustment is needed
        let needs_adjustment = performance_metrics.accuracy < params.accuracy_threshold ||
            performance_metrics.memory_pressure > params.memory_pressure_threshold ||
            performance_metrics.performance_score < params.performance_threshold;

        if !needs_adjustment {
            return Ok(());
        }

        // Determine adjustment direction
        let adjustment = if performance_metrics.accuracy < params.accuracy_threshold {
            // Increase precision for better accuracy
            PrecisionAdjustment::Increase
        } else if performance_metrics.memory_pressure > params.memory_pressure_threshold {
            // Decrease precision to reduce memory usage
            PrecisionAdjustment::Decrease
        } else {
            // Optimize for performance
            PrecisionAdjustment::Optimize
        };

        // Apply adjustment
        self.apply_precision_adjustment(layer_id, adjustment)?;

        Ok(())
    }

    /// Apply precision adjustment to a layer
    fn apply_precision_adjustment(
        &mut self,
        layer_id: &str,
        adjustment: PrecisionAdjustment,
    ) -> QuantizationResult<()> {
        // This would implement the actual precision adjustment logic
        // For now, we'll just log the adjustment
        println!("Applying {:?} precision adjustment to layer {}", adjustment, layer_id);
        Ok(())
    }

    /// Update quantization statistics
    fn update_quantization_stats(&mut self, quantized: &QuantizedWeight) {
        self.stats.elements_count += quantized.stats.elements_count;
        self.stats.compression_ratio = 
            (self.stats.compression_ratio + quantized.stats.compression_ratio) / 2.0;
        // Update other statistics as needed
    }

    /// Get quantization statistics
    pub fn get_stats(&self) -> &QuantizationStats {
        &self.stats
    }

    /// Get precision manager
    pub fn precision_manager(&self) -> &PrecisionManager {
        &self.precision_manager
    }
}

/// Result of layer quantization
#[derive(Debug)]
pub struct LayerQuantizationResult {
    /// Layer identifier
    pub layer_id: String,
    /// Quantized weights
    pub quantized_weights: QuantizedWeight,
    /// Quantized activations (if provided)
    pub quantized_activations: Option<QuantizedActivation>,
    /// Quantized bias (if provided)
    pub quantized_bias: Option<BitNetTensor>,
    /// Time taken for quantization
    pub quantization_time: std::time::Duration,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Original size in bytes
    pub original_size_bytes: usize,
    /// Quantized size in bytes
    pub quantized_size_bytes: usize,
}

/// Performance metrics for a layer
#[derive(Debug, Clone)]
pub struct LayerPerformanceMetrics {
    /// Accuracy score (0-1)
    pub accuracy: f32,
    /// Memory pressure (0-1)
    pub memory_pressure: f32,
    /// Performance score (0-1)
    pub performance_score: f32,
    /// Execution time in milliseconds
    pub execution_time_ms: f32,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

/// Precision adjustment direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecisionAdjustment {
    /// Increase precision
    Increase,
    /// Decrease precision
    Decrease,
    /// Optimize precision
    Optimize,
}

/// Convert BitNetDType to QuantizationPrecision
fn precision_to_quantization_precision(dtype: BitNetDType) -> QuantizationPrecision {
    match dtype {
        BitNetDType::BitNet158 => QuantizationPrecision::OneFiveFiveBit,
        BitNetDType::I1 => QuantizationPrecision::OneBit,
        BitNetDType::I2 => QuantizationPrecision::TwoBit,
        BitNetDType::I4 => QuantizationPrecision::FourBit,
        BitNetDType::I8 => QuantizationPrecision::EightBit,
        _ => QuantizationPrecision::EightBit, // Default for float types
    }
}

/// Factory function to create mixed precision quantizers
pub fn create_mixed_precision_quantizer(
    config: MixedPrecisionQuantizationConfig,
    device: Device,
) -> QuantizationResult<MixedPrecisionQuantizer> {
    MixedPrecisionQuantizer::new(config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_core::device::get_cpu_device;
    use bitnet_core::memory::HybridMemoryPool;

    #[test]
    fn test_mixed_precision_quantization_config() {
        let config = MixedPrecisionQuantizationConfig::default();
        assert!(config.validate().is_ok());

        let bitnet_config = MixedPrecisionQuantizationConfig::bitnet();
        assert!(bitnet_config.validate().is_ok());
        assert!(bitnet_config.auto_precision_adjustment);
    }

    #[test]
    fn test_precision_adjustment_params() {
        let params = PrecisionAdjustmentParams::default();
        assert!(params.validate().is_ok());

        let mut invalid_params = params.clone();
        invalid_params.accuracy_threshold = 1.5; // Invalid
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_mixed_precision_quantizer_creation() {
        let config = MixedPrecisionQuantizationConfig::default();
        let device = get_cpu_device();
        
        let quantizer = MixedPrecisionQuantizer::new(config, device);
        assert!(quantizer.is_ok());
    }

    #[test]
    fn test_precision_conversion() {
        assert_eq!(
            precision_to_quantization_precision(BitNetDType::BitNet158),
            QuantizationPrecision::OneFiveFiveBit
        );
        assert_eq!(
            precision_to_quantization_precision(BitNetDType::I8),
            QuantizationPrecision::EightBit
        );
    }

    #[test]
    fn test_layer_performance_metrics() {
        let metrics = LayerPerformanceMetrics {
            accuracy: 0.95,
            memory_pressure: 0.7,
            performance_score: 0.8,
            execution_time_ms: 10.5,
            memory_usage_bytes: 1024,
        };

        assert_eq!(metrics.accuracy, 0.95);
        assert_eq!(metrics.memory_pressure, 0.7);
    }
}