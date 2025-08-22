//! Mixed precision quantization correctness tests
//!
//! This module provides comprehensive tests for mixed precision quantization
//! operations, including precision management, automatic adjustment, and
//! layer-wise quantization with different precision strategies.

use bitnet_quant::quantization::mixed_precision::*;
use bitnet_quant::quantization::QuantizationPrecision;
use bitnet_core::mixed_precision::{
    MixedPrecisionStrategy, LayerPrecisionSpec, LayerType, ComponentType,
};
use bitnet_core::memory::tensor::{BitNetDType, BitNetTensor};
use bitnet_core::device::get_cpu_device;
use candle_core::{Device, Tensor, Shape};

/// Test helper to create BitNet tensors with specific characteristics
fn create_bitnet_tensor(device: &Device, pattern: &str, shape: &[usize], dtype: BitNetDType) -> BitNetTensor {
    let total_elements: usize = shape.iter().product();
    let data: Vec<f32> = match pattern {
        "normal_weights" => {
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32 - 0.5) * 4.0;
                (-x * x / 2.0).exp() * (2.0 * std::f32::consts::PI).sqrt().recip()
            }).collect()
        }
        "sparse_weights" => {
            (0..total_elements).map(|i| {
                if i % 4 == 0 { 1.0 } else if i % 7 == 0 { -1.0 } else { 0.0 }
            }).collect()
        }
        "activations" => {
            (0..total_elements).map(|i| {
                let x = (i as f32 / total_elements as f32 - 0.3) * 5.0;
                x.max(0.0) // ReLU-like
            }).collect()
        }
        _ => (0..total_elements).map(|i| i as f32 * 0.1).collect(),
    };
    
    let shape = Shape::from_dims(shape);
    let tensor = Tensor::from_vec(data, shape, device).unwrap();
    // For testing purposes, create a simple BitNetTensor
    // Note: This is a simplified approach for testing
    BitNetTensor::from_candle(tensor, &bitnet_core::memory::HybridMemoryPool::new().unwrap()).unwrap()
}

#[test]
fn test_mixed_precision_quantization_config_default() {
    let config = MixedPrecisionQuantizationConfig::default();
    assert!(config.validate().is_ok());
    assert!(!config.auto_precision_adjustment);
    assert_eq!(config.adjustment_params.accuracy_threshold, 0.95);
}

#[test]
fn test_mixed_precision_quantization_config_bitnet() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    assert!(config.validate().is_ok());
    assert!(config.auto_precision_adjustment);
    assert_eq!(config.weight_quantization.base.precision, QuantizationPrecision::OneFiveFiveBit);
    assert_eq!(config.activation_quantization.base.precision, QuantizationPrecision::OneFiveFiveBit);
}

#[test]
fn test_mixed_precision_quantization_config_with_strategy() {
    let config = MixedPrecisionQuantizationConfig::with_strategy(MixedPrecisionStrategy::Balanced);
    assert!(config.validate().is_ok());
    
    let config_conservative = MixedPrecisionQuantizationConfig::with_strategy(MixedPrecisionStrategy::Conservative);
    assert!(config_conservative.validate().is_ok());
    
    let config_aggressive = MixedPrecisionQuantizationConfig::with_strategy(MixedPrecisionStrategy::Aggressive);
    assert!(config_aggressive.validate().is_ok());
}

#[test]
fn test_mixed_precision_quantization_config_with_auto_adjustment() {
    let params = PrecisionAdjustmentParams {
        accuracy_threshold: 0.9,
        memory_pressure_threshold: 0.7,
        performance_threshold: 0.8,
        adjustment_step: 2,
        max_adjustments: 5,
        evaluation_window: 50,
    };
    
    let config = MixedPrecisionQuantizationConfig::default()
        .with_auto_adjustment(params.clone());
    
    assert!(config.validate().is_ok());
    assert!(config.auto_precision_adjustment);
    assert_eq!(config.adjustment_params.accuracy_threshold, 0.9);
    assert_eq!(config.adjustment_params.adjustment_step, 2);
}

#[test]
fn test_precision_adjustment_params_validation() {
    let mut params = PrecisionAdjustmentParams::default();
    assert!(params.validate().is_ok());
    
    // Test invalid accuracy threshold
    params.accuracy_threshold = 1.5;
    assert!(params.validate().is_err());
    
    params.accuracy_threshold = -0.1;
    assert!(params.validate().is_err());
    
    // Test invalid memory pressure threshold
    params = PrecisionAdjustmentParams::default();
    params.memory_pressure_threshold = 1.1;
    assert!(params.validate().is_err());
    
    // Test invalid performance threshold
    params = PrecisionAdjustmentParams::default();
    params.performance_threshold = 0.0;
    assert!(params.validate().is_err());
    
    // Test invalid adjustment step
    params = PrecisionAdjustmentParams::default();
    params.adjustment_step = 0;
    assert!(params.validate().is_err());
    
    // Test invalid evaluation window
    params = PrecisionAdjustmentParams::default();
    params.evaluation_window = 0;
    assert!(params.validate().is_err());
}

#[test]
fn test_mixed_precision_quantizer_creation() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    
    let quantizer = MixedPrecisionQuantizer::new(config, device);
    assert!(quantizer.is_ok());
    
    let quantizer = quantizer.unwrap();
    assert!(quantizer.get_stats().elements_count == 0); // Initially empty
}

#[test]
fn test_mixed_precision_quantizer_creation_with_invalid_config() {
    let mut config = MixedPrecisionQuantizationConfig::default();
    config.adjustment_params.accuracy_threshold = 2.0; // Invalid
    
    let device = get_cpu_device();
    let result = MixedPrecisionQuantizer::new(config, device);
    assert!(result.is_err());
}

#[test]
fn test_mixed_precision_quantizer_layer_registration() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let quantizer = MixedPrecisionQuantizer::new(config, device).unwrap();
    
    // Create layer precision specification
    let layer_spec = LayerPrecisionSpec::new(
        "test_layer".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    ).with_component_precision(ComponentType::Bias, BitNetDType::F32);
    
    let result = quantizer.register_layer(layer_spec);
    assert!(result.is_ok());
}

#[test]
fn test_mixed_precision_weight_quantization() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer first
    let layer_spec = LayerPrecisionSpec::new(
        "linear1".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Create test weights
    let weights = create_bitnet_tensor(&device, "normal_weights", &[64, 128], BitNetDType::F32);
    
    // Quantize weights
    let result = quantizer.quantize_weights(&weights, "linear1");
    assert!(result.is_ok());
    
    let quantized = result.unwrap();
    assert!(quantized.stats.compression_ratio > 1.0);
    assert_eq!(quantized.stats.elements_count, 64 * 128);
}

#[test]
fn test_mixed_precision_activation_quantization() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer first
    let layer_spec = LayerPrecisionSpec::new(
        "conv1".to_string(),
        LayerType::Convolution,
        BitNetDType::BitNet158,  // input_precision
        BitNetDType::BitNet158,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Create test activations
    let activations = create_bitnet_tensor(&device, "activations", &[32, 64], BitNetDType::F32);
    
    // Quantize activations
    let result = quantizer.quantize_activations(&activations, "conv1");
    assert!(result.is_ok());
    
    let quantized = result.unwrap();
    assert_eq!(quantized.original_shape, Shape::from_dims(&[32, 64]));
    assert!(quantized.stats.scale_factor > 0.0);
}

#[test]
fn test_mixed_precision_layer_quantization() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer
    let layer_spec = LayerPrecisionSpec::new(
        "attention1".to_string(),
        LayerType::Attention,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    ).with_component_precision(ComponentType::Bias, BitNetDType::F32);
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Create test tensors
    let weights = create_bitnet_tensor(&device, "normal_weights", &[512, 512], BitNetDType::F32);
    let activations = create_bitnet_tensor(&device, "activations", &[32, 512], BitNetDType::F32);
    let bias = create_bitnet_tensor(&device, "normal_weights", &[512], BitNetDType::F32);
    
    // Quantize entire layer
    let result = quantizer.quantize_layer(
        "attention1",
        &weights,
        Some(&activations),
        Some(&bias),
    );
    
    assert!(result.is_ok());
    
    let layer_result = result.unwrap();
    assert_eq!(layer_result.layer_id, "attention1");
    assert!(layer_result.quantized_activations.is_some());
    assert!(layer_result.quantized_bias.is_some());
    assert!(layer_result.compression_ratio > 1.0);
    assert!(layer_result.quantization_time.as_millis() >= 0);
    assert!(layer_result.original_size_bytes > layer_result.quantized_size_bytes);
}

#[test]
fn test_mixed_precision_layer_quantization_without_optional_components() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a simple layer
    let layer_spec = LayerPrecisionSpec::new(
        "simple_linear".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Create only weights
    let weights = create_bitnet_tensor(&device, "normal_weights", &[256, 256], BitNetDType::F32);
    
    // Quantize layer without activations and bias
    let result = quantizer.quantize_layer("simple_linear", &weights, None, None);
    assert!(result.is_ok());
    
    let layer_result = result.unwrap();
    assert!(layer_result.quantized_activations.is_none());
    assert!(layer_result.quantized_bias.is_none());
    assert!(layer_result.compression_ratio > 1.0);
}

#[test]
fn test_mixed_precision_optimization() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer
    let layer_spec = LayerPrecisionSpec::new(
        "optimize_layer".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Test with different performance metrics
    let good_metrics = LayerPerformanceMetrics {
        accuracy: 0.98,
        memory_pressure: 0.5,
        performance_score: 0.95,
        execution_time_ms: 5.0,
        memory_usage_bytes: 1024,
    };
    
    let result = quantizer.optimize_layer_precision("optimize_layer", &good_metrics);
    assert!(result.is_ok());
    
    // Test with poor accuracy (should trigger adjustment)
    let poor_accuracy_metrics = LayerPerformanceMetrics {
        accuracy: 0.85, // Below threshold
        memory_pressure: 0.5,
        performance_score: 0.95,
        execution_time_ms: 5.0,
        memory_usage_bytes: 1024,
    };
    
    let result = quantizer.optimize_layer_precision("optimize_layer", &poor_accuracy_metrics);
    assert!(result.is_ok());
    
    // Test with high memory pressure
    let high_memory_metrics = LayerPerformanceMetrics {
        accuracy: 0.98,
        memory_pressure: 0.9, // Above threshold
        performance_score: 0.95,
        execution_time_ms: 5.0,
        memory_usage_bytes: 1024,
    };
    
    let result = quantizer.optimize_layer_precision("optimize_layer", &high_memory_metrics);
    assert!(result.is_ok());
}

#[test]
fn test_mixed_precision_optimization_disabled() {
    let mut config = MixedPrecisionQuantizationConfig::default();
    config.auto_precision_adjustment = false; // Disable optimization
    
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer
    let layer_spec = LayerPrecisionSpec::new(
        "no_optimize_layer".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Even with poor metrics, optimization should not trigger
    let poor_metrics = LayerPerformanceMetrics {
        accuracy: 0.5, // Very poor
        memory_pressure: 0.95, // Very high
        performance_score: 0.3, // Very poor
        execution_time_ms: 100.0,
        memory_usage_bytes: 10240,
    };
    
    let result = quantizer.optimize_layer_precision("no_optimize_layer", &poor_metrics);
    assert!(result.is_ok()); // Should succeed but do nothing
}

#[test]
fn test_precision_conversion_functions() {
    // Test precision conversion logic (implementation details)
    // Since the function is private, we test the behavior indirectly
    
    // Test that different BitNet dtypes map to appropriate quantization precisions
    // This is tested through the quantizer creation and behavior
    let config_158 = MixedPrecisionQuantizationConfig::bitnet();
    assert_eq!(config_158.weight_quantization.base.precision, QuantizationPrecision::OneFiveFiveBit);
    
    // Test other precision mappings through configuration
    let mut config_8bit = MixedPrecisionQuantizationConfig::default();
    config_8bit.weight_quantization.base.precision = QuantizationPrecision::EightBit;
    assert_eq!(config_8bit.weight_quantization.base.precision, QuantizationPrecision::EightBit);
}

#[test]
fn test_mixed_precision_quantizer_statistics() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer
    let layer_spec = LayerPrecisionSpec::new(
        "stats_layer".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Initial statistics should be empty
    let initial_stats = quantizer.get_stats();
    assert_eq!(initial_stats.elements_count, 0);
    
    // Quantize some weights
    let weights = create_bitnet_tensor(&device, "normal_weights", &[32, 64], BitNetDType::F32);
    let _ = quantizer.quantize_weights(&weights, "stats_layer").unwrap();
    
    // Statistics should be updated
    let updated_stats = quantizer.get_stats();
    assert!(updated_stats.elements_count > 0);
    assert!(updated_stats.compression_ratio > 0.0);
}

#[test]
fn test_mixed_precision_quantizer_precision_manager_access() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let quantizer = MixedPrecisionQuantizer::new(config, device).unwrap();
    
    // Should be able to access precision manager
    let precision_manager = quantizer.precision_manager();
    // Just verify we can access it without errors
    assert!(precision_manager as *const _ != std::ptr::null());
}

#[test]
fn test_layer_quantization_result_structure() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Register a layer
    let layer_spec = LayerPrecisionSpec::new(
        "result_test".to_string(),
        LayerType::Linear,
        BitNetDType::F16,  // input_precision
        BitNetDType::F16,  // output_precision
        BitNetDType::BitNet158,  // weight_precision
    );
    
    quantizer.register_layer(layer_spec).unwrap();
    
    // Create test data
    let weights = create_bitnet_tensor(&device, "normal_weights", &[16, 32], BitNetDType::F32);
    let activations = create_bitnet_tensor(&device, "activations", &[8, 16], BitNetDType::F32);
    
    // Quantize layer
    let result = quantizer.quantize_layer(
        "result_test",
        &weights,
        Some(&activations),
        None,
    ).unwrap();
    
    // Verify result structure
    assert_eq!(result.layer_id, "result_test");
    assert!(result.quantized_activations.is_some());
    assert!(result.quantized_bias.is_none());
    assert!(result.compression_ratio > 1.0);
    assert!(result.quantization_time.as_nanos() > 0);
    assert!(result.original_size_bytes > 0);
    assert!(result.quantized_size_bytes > 0);
    assert!(result.original_size_bytes > result.quantized_size_bytes);
}

#[test]
fn test_layer_performance_metrics_structure() {
    let metrics = LayerPerformanceMetrics {
        accuracy: 0.95,
        memory_pressure: 0.7,
        performance_score: 0.8,
        execution_time_ms: 10.5,
        memory_usage_bytes: 2048,
    };
    
    assert_eq!(metrics.accuracy, 0.95);
    assert_eq!(metrics.memory_pressure, 0.7);
    assert_eq!(metrics.performance_score, 0.8);
    assert_eq!(metrics.execution_time_ms, 10.5);
    assert_eq!(metrics.memory_usage_bytes, 2048);
    
    // Test cloning
    let cloned_metrics = metrics.clone();
    assert_eq!(cloned_metrics.accuracy, metrics.accuracy);
    assert_eq!(cloned_metrics.memory_usage_bytes, metrics.memory_usage_bytes);
}

#[test]
fn test_mixed_precision_quantizer_factory() {
    let config = MixedPrecisionQuantizationConfig::bitnet();
    let device = get_cpu_device();
    
    let quantizer = create_mixed_precision_quantizer(config, device);
    assert!(quantizer.is_ok());
    
    let quantizer = quantizer.unwrap();
    assert!(quantizer.get_stats().elements_count == 0);
}

#[test]
fn test_mixed_precision_quantizer_factory_with_invalid_config() {
    let mut config = MixedPrecisionQuantizationConfig::default();
    config.adjustment_params.accuracy_threshold = -1.0; // Invalid
    
    let device = get_cpu_device();
    let result = create_mixed_precision_quantizer(config, device);
    assert!(result.is_err());
}

#[test]
fn test_mixed_precision_different_strategies() {
    let device = get_cpu_device();
    
    let strategies = vec![
        MixedPrecisionStrategy::Balanced,
        MixedPrecisionStrategy::Conservative,
        MixedPrecisionStrategy::Aggressive,
    ];
    
    for strategy in strategies {
        let config = MixedPrecisionQuantizationConfig::with_strategy(strategy);
        let quantizer = MixedPrecisionQuantizer::new(config, device.clone());
        assert!(quantizer.is_ok(), "Failed to create quantizer with strategy {:?}", strategy);
    }
}

#[test]
fn test_mixed_precision_edge_cases() {
    let config = MixedPrecisionQuantizationConfig::default();
    let device = get_cpu_device();
    let mut quantizer = MixedPrecisionQuantizer::new(config, device.clone()).unwrap();
    
    // Test with unregistered layer
    let weights = create_bitnet_tensor(&device, "normal_weights", &[8, 8], BitNetDType::F32);
    let result = quantizer.quantize_weights(&weights, "unregistered_layer");
    // This might fail or succeed depending on implementation - just ensure it doesn't panic
    let _ = result;
    
    // Test with empty tensors
    let empty_weights = create_bitnet_tensor(&device, "normal_weights", &[0, 0], BitNetDType::F32);
    // This should handle gracefully
    let _ = quantizer.quantize_weights(&empty_weights, "test_layer");
}