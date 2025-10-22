//! Apple Neural Engine (ANE) Integration for BitNet
//! 
//! This module provides Apple Neural Engine acceleration for BitNet 1.58-bit models,
//! including CoreML conversion, hybrid GPU/ANE partitioning, and performance monitoring.
//! 
//! ## Features
//! - ANE availability detection for Apple Silicon (M1, M2, M3, M4)
//! - GGUF to CoreML conversion optimized for ANE
//! - Hybrid partitioning strategies (Full ANE, ANE/GPU Hybrid, Selective, Dynamic)
//! - Performance monitoring and optimization recommendations
//! 
//! ## Architecture
//! The ANE integration follows a multi-tier acceleration strategy:
//! 1. **Full ANE**: Run entire model on Neural Engine (best efficiency)
//! 2. **ANE/GPU Hybrid**: Partition layers between ANE and Metal GPU
//! 3. **Selective ANE**: Run specific operations on ANE, others on GPU
//! 4. **Dynamic**: Runtime optimization based on performance metrics

// Tests module
#[cfg(test)]
mod tests;

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "mlx")]
use crate::mlx::MLXArray;

// Mock CoreML types for compilation - in real implementation these would come from CoreML framework
pub struct MLModel {
    pub(crate) path: String,
}

pub struct MLModelConfiguration {
    pub compute_units: MLComputeUnits,
    pub allow_low_precision: bool,
    pub use_cpu_only: bool,
}

pub enum MLComputeUnits {
    NeuralEngine,
    GPU,
    CPU,
}

pub struct CoreMLInput {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
}

pub struct CoreMLOutput {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
}

/// Apple Neural Engine integration for BitNet acceleration
pub struct ANEBitNetAccelerator {
    /// CoreML model optimized for ANE
    coreml_model: MLModel,
    /// Metal device for GPU fallback
    #[cfg(feature = "metal")]
    metal_device: Option<metal::Device>,
    /// ANE availability status
    ane_available: bool,
    /// Model partitioning strategy
    partitioning_strategy: ANEPartitioningStrategy,
    /// Performance monitor
    performance_monitor: ANEPerformanceMonitor,
}

#[derive(Debug, Clone)]
pub enum ANEPartitioningStrategy {
    /// Run entire model on ANE (if possible)
    FullANE,
    /// Partition between ANE and GPU
    ANEGPUHybrid { 
        ane_layers: Vec<usize>,
        gpu_layers: Vec<usize>,
    },
    /// Run compute-intensive ops on ANE, others on GPU
    SelectiveANE {
        ane_operations: Vec<ANEOperation>,
    },
    /// Dynamic partitioning based on performance
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ANEOperation {
    /// Matrix multiplication operations
    MatMul,
    /// Convolution operations
    Convolution,
    /// Quantization/dequantization
    Quantization,
    /// Attention computation
    Attention,
    /// Activation functions
    Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    ANE,
    GPU,
    CPU,
}

#[derive(Debug, Error)]
pub enum ANEError {
    #[error("Apple Neural Engine is not available on this platform")]
    NotAvailable,
    #[error("CoreML model compilation failed: {0}")]
    CoreMLCompilation(String),
    #[error("Model conversion failed: {0}")]
    ModelConversion(String),
    #[error("ANE inference failed: {0}")]
    InferenceFailed(String),
    #[error("Hardware detection failed: {0}")]
    HardwareDetection(String),
}

impl ANEBitNetAccelerator {
    /// Create new ANE accelerator with automatic hardware detection
    #[cfg(feature = "metal")]
    pub fn new(model_path: &str, device: &metal::Device) -> Result<Self> {
        // Check ANE availability
        let ane_available = Self::check_ane_availability()?;
        
        if ane_available {
            // Load ANE-optimized CoreML model
            let coreml_model = Self::load_ane_optimized_model(model_path)?;
            
            // Determine optimal partitioning strategy
            let partitioning_strategy = Self::analyze_model_for_partitioning(&coreml_model)?;
            
            Ok(Self {
                coreml_model,
                metal_device: Some(device.clone()),
                ane_available,
                partitioning_strategy,
                performance_monitor: ANEPerformanceMonitor::new(),
            })
        } else {
            Err(ANEError::NotAvailable.into())
        }
    }
    
    /// Create new ANE accelerator without Metal dependency (for testing)
    pub fn new_mock(model_path: &str) -> Result<Self> {
        let ane_available = Self::check_ane_availability()?;
        let coreml_model = Self::load_ane_optimized_model(model_path)?;
        let partitioning_strategy = Self::analyze_model_for_partitioning(&coreml_model)?;
        
        Ok(Self {
            coreml_model,
            #[cfg(feature = "metal")]
            metal_device: None, // No metal device in mock mode
            ane_available,
            partitioning_strategy,
            performance_monitor: ANEPerformanceMonitor::new(),
        })
    }
    
    /// Check if Apple Neural Engine is available on this platform
    fn check_ane_availability() -> Result<bool> {
        // Check if running on Apple Silicon with ANE support
        #[cfg(target_arch = "aarch64")]
        {
            use std::process::Command;
            
            // Check for ANE using system_profiler
            let output = Command::new("system_profiler")
                .args(&["SPHardwareDataType"])
                .output()
                .map_err(|e| ANEError::HardwareDetection(format!("Failed to run system_profiler: {}", e)))?;
            
            let output_str = String::from_utf8_lossy(&output.stdout);
            
            // Look for Neural Engine in hardware profile
            let has_neural_engine = output_str.contains("Neural Engine");
            let has_apple_silicon = output_str.contains("Apple") && (
                output_str.contains("M1") || 
                output_str.contains("M2") || 
                output_str.contains("M3") ||
                output_str.contains("M4")
            );
            
            Ok(has_neural_engine || has_apple_silicon)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // ANE is only available on Apple Silicon
            Ok(false)
        }
    }
    
    /// Load and optimize model for ANE
    fn load_ane_optimized_model(model_path: &str) -> Result<MLModel> {
        // Convert GGUF model to ANE-optimized CoreML format
        let coreml_path = Self::convert_to_coreml_ane(model_path)?;
        
        // Load with ANE-specific optimizations
        let model = MLModel::compile_and_load(&coreml_path, &MLModelConfiguration {
            compute_units: MLComputeUnits::NeuralEngine,
            allow_low_precision: true,
            use_cpu_only: false,
        })?;
        
        Ok(model)
    }
    
    /// Analyze model architecture to determine optimal partitioning strategy
    fn analyze_model_for_partitioning(_model: &MLModel) -> Result<ANEPartitioningStrategy> {
        // For now, default to dynamic partitioning
        // In a real implementation, this would analyze:
        // - Model size and complexity
        // - Layer types and their ANE compatibility
        // - Available memory on ANE vs GPU
        // - Historical performance data
        
        Ok(ANEPartitioningStrategy::Dynamic)
    }
    
    /// Forward pass with ANE optimization
    #[cfg(feature = "mlx")]
    pub fn forward_ane_optimized(&mut self, input: &MLXArray) -> Result<MLXArray> {
        let start_time = std::time::Instant::now();
        
        let result = match &self.partitioning_strategy {
            ANEPartitioningStrategy::FullANE => {
                self.forward_full_ane(input)
            },
            ANEPartitioningStrategy::ANEGPUHybrid { ane_layers, gpu_layers } => {
                self.forward_hybrid_ane_gpu(input, ane_layers, gpu_layers)
            },
            ANEPartitioningStrategy::SelectiveANE { ane_operations } => {
                self.forward_selective_ane(input, ane_operations)
            },
            ANEPartitioningStrategy::Dynamic => {
                self.forward_dynamic_ane(input)
            },
        };
        
        let execution_time = start_time.elapsed().as_secs_f32();
        
        // Log performance metrics
        if let Ok(_) = &result {
            self.performance_monitor.log_operation(
                ANEOperation::MatMul, // Simplified - would be more specific in real implementation
                execution_time,
                Device::ANE,
            );
        }
        
        result
    }
    
    /// Mock forward pass for testing without MLX
    #[cfg(not(feature = "mlx"))]
    pub fn forward_mock(&mut self, input_shape: &[usize]) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();
        
        // Mock computation - just return zeros with same shape
        let output_size = input_shape.iter().product();
        let result = vec![0.0f32; output_size];
        
        let execution_time = start_time.elapsed().as_secs_f32();
        self.performance_monitor.log_operation(
            ANEOperation::MatMul,
            execution_time,
            Device::ANE,
        );
        
        Ok(result)
    }
    
    #[cfg(feature = "mlx")]
    fn forward_full_ane(&self, input: &MLXArray) -> Result<MLXArray> {
        // Convert MLX array to CoreML input format
        let coreml_input = self.mlx_to_coreml_input(input)?;
        
        // Run inference on ANE
        let coreml_output = self.coreml_model.predict(&coreml_input)?;
        
        // Convert back to MLX format
        self.coreml_to_mlx_output(&coreml_output)
    }
    
    #[cfg(feature = "mlx")]
    fn forward_hybrid_ane_gpu(
        &self, 
        input: &MLXArray, 
        ane_layers: &[usize], 
        gpu_layers: &[usize]
    ) -> Result<MLXArray> {
        let mut current_input = input.clone();
        
        // Process layers according to partitioning strategy
        for layer_idx in 0..self.get_total_layers() {
            if ane_layers.contains(&layer_idx) {
                // Process on ANE
                current_input = self.process_layer_ane(&current_input, layer_idx)?;
            } else if gpu_layers.contains(&layer_idx) {
                // Process on GPU (Metal)
                current_input = self.process_layer_gpu(&current_input, layer_idx)?;
            }
        }
        
        Ok(current_input)
    }
    
    #[cfg(feature = "mlx")]
    fn forward_selective_ane(&self, input: &MLXArray, ane_operations: &[ANEOperation]) -> Result<MLXArray> {
        // Process different operations on appropriate devices
        let mut current_input = input.clone();
        
        // This is a simplified implementation - real version would analyze the computation graph
        for operation in ane_operations {
            match operation {
                ANEOperation::MatMul => {
                    current_input = self.process_matmul_ane(&current_input)?;
                },
                ANEOperation::Attention => {
                    current_input = self.process_attention_ane(&current_input)?;
                },
                _ => {
                    // Fallback to GPU for other operations
                    current_input = self.process_operation_gpu(&current_input, *operation)?;
                }
            }
        }
        
        Ok(current_input)
    }
    
    #[cfg(feature = "mlx")]
    fn forward_dynamic_ane(&self, input: &MLXArray) -> Result<MLXArray> {
        // Dynamic partitioning based on runtime performance metrics
        // For now, fallback to full ANE - real implementation would use performance history
        self.forward_full_ane(input)
    }
    
    fn get_total_layers(&self) -> usize {
        // Mock implementation - real version would inspect the model
        16 // Typical small transformer model
    }
    
    #[cfg(feature = "mlx")]
    fn process_layer_ane(&self, input: &MLXArray, layer_idx: usize) -> Result<MLXArray> {
        // Extract specific layer for ANE processing
        let _layer_model = self.extract_layer_model(layer_idx)?;
        
        // Convert to CoreML input
        let coreml_input = self.mlx_to_coreml_input(input)?;
        
        // Process on ANE (mock implementation)
        let coreml_output = CoreMLOutput {
            data: coreml_input.data.clone(), // Mock - just pass through
            shape: coreml_input.shape.clone(),
        };
        
        // Convert back
        self.coreml_to_mlx_output(&coreml_output)
    }
    
    #[cfg(feature = "mlx")]
    fn process_layer_gpu(&self, input: &MLXArray, _layer_idx: usize) -> Result<MLXArray> {
        // Process layer on Metal GPU
        // Mock implementation - just return input
        Ok(input.clone())
    }
    
    #[cfg(feature = "mlx")]
    fn process_matmul_ane(&self, input: &MLXArray) -> Result<MLXArray> {
        // Process matrix multiplication on ANE
        Ok(input.clone()) // Mock implementation
    }
    
    #[cfg(feature = "mlx")]
    fn process_attention_ane(&self, input: &MLXArray) -> Result<MLXArray> {
        // Process attention computation on ANE
        Ok(input.clone()) // Mock implementation
    }
    
    #[cfg(feature = "mlx")]
    fn process_operation_gpu(&self, input: &MLXArray, _operation: ANEOperation) -> Result<MLXArray> {
        // Process operation on GPU
        Ok(input.clone()) // Mock implementation
    }
    
    fn extract_layer_model(&self, _layer_idx: usize) -> Result<MLModel> {
        // Extract specific layer for processing
        // Mock implementation
        Ok(MLModel {
            path: format!("layer_{}.mlmodelc", _layer_idx),
        })
    }
    
    #[cfg(feature = "mlx")]
    fn mlx_to_coreml_input(&self, input: &MLXArray) -> Result<CoreMLInput> {
        // Convert MLX array to CoreML input format
        // Mock implementation
        Ok(CoreMLInput {
            data: vec![0.0f32; 1024], // Mock data
            shape: vec![1, 1024],     // Mock shape
        })
    }
    
    #[cfg(feature = "mlx")]
    fn coreml_to_mlx_output(&self, output: &CoreMLOutput) -> Result<MLXArray> {
        // Convert CoreML output back to MLX array
        // Mock implementation - real version would use MLX array creation
        MLXArray::mock_from_data(&output.data, &output.shape)
    }
    
    /// Convert GGUF model to ANE-optimized CoreML format
    fn convert_to_coreml_ane(gguf_path: &str) -> Result<String> {
        // Load GGUF model (mock)
        let _gguf_model = GgufModel::load(gguf_path)?;
        
        // Convert to CoreML with ANE optimizations
        let coreml_converter = ANECoreMLConverter::new();
        let coreml_path = coreml_converter.convert_with_ane_optimizations(&_gguf_model)?;
        
        Ok(coreml_path)
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &ANEPerformanceMonitor {
        &self.performance_monitor
    }
    
    /// Get current partitioning strategy
    pub fn get_partitioning_strategy(&self) -> &ANEPartitioningStrategy {
        &self.partitioning_strategy
    }
    
    /// Update partitioning strategy based on performance
    pub fn update_partitioning_strategy(&mut self, strategy: ANEPartitioningStrategy) {
        self.partitioning_strategy = strategy;
    }
    
    /// Check if ANE is available and operational
    pub fn is_ane_available(&self) -> bool {
        self.ane_available
    }
}

// Mock implementations for compilation
impl MLModel {
    fn compile_and_load(_path: &str, _config: &MLModelConfiguration) -> Result<Self> {
        Ok(MLModel {
            path: _path.to_string(),
        })
    }
    
    fn predict(&self, _input: &CoreMLInput) -> Result<CoreMLOutput> {
        Ok(CoreMLOutput {
            data: vec![0.0f32; 1024],
            shape: vec![1, 1024],
        })
    }
}

#[cfg(feature = "mlx")]
impl MLXArray {
    fn mock_from_data(_data: &[f32], _shape: &[usize]) -> Result<Self> {
        // Mock implementation for compilation
        Err(anyhow!("Mock MLX array creation"))
    }
}

// Mock GGUF types
struct GgufModel {
    name: String,
    layers: Vec<Layer>,
}

impl GgufModel {
    fn load(path: &str) -> Result<Self> {
        Ok(GgufModel {
            name: format!("model_from_{}", path),
            layers: vec![],
        })
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn layers(&self) -> &[Layer] {
        &self.layers
    }
}

struct Layer {
    layer_type: LayerType,
    weights: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

#[derive(Debug, Clone)]
enum LayerType {
    BitLinear,
    Attention,
    LayerNorm,
    Other,
}

impl Layer {
    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }
    
    fn get_weights(&self) -> Result<&[f32]> {
        Ok(&self.weights)
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
    
    fn bias(&self) -> Result<Option<&[f32]>> {
        Ok(None) // Simplified
    }
}

/// ANE-specific CoreML converter
pub struct ANECoreMLConverter;

impl ANECoreMLConverter {
    pub fn new() -> Self {
        Self
    }
    
    pub fn convert_with_ane_optimizations(&self, model: &GgufModel) -> Result<String> {
        // Create CoreML model specification
        let mut spec = CoreMLSpec::new();
        
        // Configure for ANE optimization
        spec.set_compute_precision(CoreMLPrecision::Float16);
        spec.enable_ane_optimization(true);
        spec.set_batch_prediction(false); // ANE works best with single predictions
        
        // Convert BitLinear layers to ANE-compatible operations
        for layer in model.layers() {
            match layer.layer_type() {
                LayerType::BitLinear => {
                    // Convert ternary weights to ANE-compatible format
                    let ane_layer = self.convert_bitlinear_to_ane(layer)?;
                    spec.add_layer(ane_layer);
                },
                LayerType::Attention => {
                    // Optimize attention for ANE
                    let ane_attention = self.convert_attention_to_ane(layer)?;
                    spec.add_layer(ane_attention);
                },
                LayerType::LayerNorm => {
                    // ANE-optimized normalization
                    let ane_norm = self.convert_layernorm_to_ane(layer)?;
                    spec.add_layer(ane_norm);
                },
                LayerType::Other => {
                    // Fallback to standard conversion
                    spec.add_layer(self.convert_standard_layer(layer)?);
                }
            }
        }
        
        // Compile for ANE
        let output_path = format!("{}.mlmodelc", model.name());
        spec.compile_for_ane(&output_path)?;
        
        Ok(output_path)
    }
    
    fn convert_bitlinear_to_ane(&self, layer: &Layer) -> Result<ANELayer> {
        // Convert ternary weights {-1, 0, +1} to ANE-compatible format
        let weights = layer.get_weights()?;
        let ane_weights = self.convert_ternary_weights_for_ane(&weights)?;
        
        // Create ANE-optimized convolution/linear layer
        let ane_layer = ANELayer::new_linear(
            layer.input_size(),
            layer.output_size(),
            ane_weights,
            layer.bias()?,
        );
        
        Ok(ane_layer)
    }
    
    fn convert_attention_to_ane(&self, _layer: &Layer) -> Result<ANELayer> {
        // Convert attention layer for ANE optimization
        Ok(ANELayer::mock())
    }
    
    fn convert_layernorm_to_ane(&self, _layer: &Layer) -> Result<ANELayer> {
        // Convert layer normalization for ANE
        Ok(ANELayer::mock())
    }
    
    fn convert_standard_layer(&self, _layer: &Layer) -> Result<ANELayer> {
        // Standard layer conversion
        Ok(ANELayer::mock())
    }
    
    fn convert_ternary_weights_for_ane(&self, weights: &[f32]) -> Result<Vec<half::f16>> {
        // Convert {-1, 0, +1} weights to FP16 for ANE efficiency
        let ane_weights: Vec<half::f16> = weights.iter()
            .map(|&w| {
                // Check if it's a standard ternary value, otherwise preserve the original
                if (w - (-1.0)).abs() < 0.001 {
                    half::f16::from_f32(-1.0)
                } else if w.abs() < 0.001 {
                    half::f16::from_f32(0.0)
                } else if (w - 1.0).abs() < 0.001 {
                    half::f16::from_f32(1.0)
                } else {
                    // Non-ternary weight, preserve original value
                    half::f16::from_f32(w)
                }
            })
            .collect();
        
        Ok(ane_weights)
    }
}

// Mock CoreML spec types
struct CoreMLSpec {
    layers: Vec<ANELayer>,
    precision: CoreMLPrecision,
    ane_optimized: bool,
    batch_prediction: bool,
}

enum CoreMLPrecision {
    Float16,
    Float32,
}

struct ANELayer {
    layer_type: String,
}

impl CoreMLSpec {
    fn new() -> Self {
        Self {
            layers: vec![],
            precision: CoreMLPrecision::Float32,
            ane_optimized: false,
            batch_prediction: true,
        }
    }
    
    fn set_compute_precision(&mut self, precision: CoreMLPrecision) {
        self.precision = precision;
    }
    
    fn enable_ane_optimization(&mut self, enabled: bool) {
        self.ane_optimized = enabled;
    }
    
    fn set_batch_prediction(&mut self, enabled: bool) {
        self.batch_prediction = enabled;
    }
    
    fn add_layer(&mut self, layer: ANELayer) {
        self.layers.push(layer);
    }
    
    fn compile_for_ane(&self, output_path: &str) -> Result<()> {
        // Mock compilation
        println!("Compiling CoreML model for ANE: {}", output_path);
        Ok(())
    }
}

impl ANELayer {
    fn new_linear(
        _input_size: usize,
        _output_size: usize,
        _weights: Vec<half::f16>,
        _bias: Option<&[f32]>,
    ) -> Self {
        Self {
            layer_type: "linear".to_string(),
        }
    }
    
    fn mock() -> Self {
        Self {
            layer_type: "mock".to_string(),
        }
    }
}

/// Performance monitoring for ANE operations
pub struct ANEPerformanceMonitor {
    ane_usage_stats: ANEUsageStats,
    gpu_fallback_stats: GPUFallbackStats,
    power_consumption: PowerConsumption,
}

#[derive(Debug, Default)]
pub struct ANEUsageStats {
    pub total_operations: u64,
    pub ane_operations: u64,
    pub gpu_operations: u64,
    pub cpu_operations: u64,
    pub average_ane_latency: f32,
    pub ane_efficiency_ratio: f32,
}

#[derive(Debug, Default)]
pub struct GPUFallbackStats {
    pub total_fallbacks: u64,
    pub fallback_reasons: HashMap<ANEOperation, u64>,
}

#[derive(Debug)]
pub struct PowerConsumption {
    total_energy: f32,
    ane_energy: f32,
    gpu_energy: f32,
}

#[derive(Debug)]
pub struct ANEPerformanceReport {
    pub ane_utilization: f32,
    pub average_latency: f32,
    pub power_efficiency: f32,
    pub recommendations: Vec<String>,
}

impl ANEPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            ane_usage_stats: ANEUsageStats::default(),
            gpu_fallback_stats: GPUFallbackStats::default(),
            power_consumption: PowerConsumption::new(),
        }
    }
    
    pub fn log_operation(&mut self, operation_type: ANEOperation, execution_time: f32, device_used: Device) {
        self.ane_usage_stats.total_operations += 1;
        
        match device_used {
            Device::ANE => {
                self.ane_usage_stats.ane_operations += 1;
                self.update_ane_latency(execution_time);
            },
            Device::GPU => {
                self.ane_usage_stats.gpu_operations += 1;
                self.gpu_fallback_stats.log_fallback(operation_type);
            },
            Device::CPU => {
                self.ane_usage_stats.cpu_operations += 1;
            },
        }
        
        // Update efficiency metrics
        self.compute_efficiency_ratio();
    }
    
    fn update_ane_latency(&mut self, execution_time: f32) {
        let total_ane_ops = self.ane_usage_stats.ane_operations as f32;
        let current_avg = self.ane_usage_stats.average_ane_latency;
        
        // Running average
        self.ane_usage_stats.average_ane_latency = 
            (current_avg * (total_ane_ops - 1.0) + execution_time) / total_ane_ops;
    }
    
    fn compute_efficiency_ratio(&mut self) {
        let total = self.ane_usage_stats.total_operations as f32;
        if total > 0.0 {
            self.ane_usage_stats.ane_efficiency_ratio = 
                self.ane_usage_stats.ane_operations as f32 / total;
        }
    }
    
    pub fn generate_performance_report(&self) -> ANEPerformanceReport {
        let total_ops = self.ane_usage_stats.total_operations as f32;
        let ane_utilization = if total_ops > 0.0 {
            self.ane_usage_stats.ane_operations as f32 / total_ops
        } else {
            0.0
        };
        
        ANEPerformanceReport {
            ane_utilization,
            average_latency: self.ane_usage_stats.average_ane_latency,
            power_efficiency: self.power_consumption.compute_efficiency(),
            recommendations: self.generate_optimization_recommendations(),
        }
    }
    
    fn generate_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let ane_ratio = self.ane_usage_stats.ane_efficiency_ratio;
        
        if ane_ratio < 0.5 {
            recommendations.push("Consider using more ANE-optimized operations".to_string());
        }
        
        if self.ane_usage_stats.average_ane_latency > 10.0 {
            recommendations.push("ANE latency is high - consider model optimization".to_string());
        }
        
        if self.gpu_fallback_stats.total_fallbacks > 100 {
            recommendations.push("High GPU fallback rate - review ANE compatibility".to_string());
        }
        
        recommendations
    }
    
    /// Get usage statistics
    pub fn get_usage_stats(&self) -> &ANEUsageStats {
        &self.ane_usage_stats
    }
    
    /// Get fallback statistics  
    pub fn get_fallback_stats(&self) -> &GPUFallbackStats {
        &self.gpu_fallback_stats
    }
    
    /// Get power consumption data
    pub fn get_power_consumption(&self) -> &PowerConsumption {
        &self.power_consumption
    }
}

impl GPUFallbackStats {
    pub fn log_fallback(&mut self, operation: ANEOperation) {
        self.total_fallbacks += 1;
        *self.fallback_reasons.entry(operation).or_insert(0) += 1;
    }
}

impl PowerConsumption {
    pub fn new() -> Self {
        Self {
            total_energy: 0.0,
            ane_energy: 0.0,
            gpu_energy: 0.0,
        }
    }
    
    pub fn compute_efficiency(&self) -> f32 {
        if self.total_energy > 0.0 {
            self.ane_energy / self.total_energy
        } else {
            0.0
        }
    }
}

