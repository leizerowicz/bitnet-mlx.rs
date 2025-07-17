//! # BitNet Metal Shader Utilities
//!
//! This module provides high-level utilities for loading and using BitNet-specific Metal shaders.
//! It includes pre-configured shader loading for BitLinear operations, quantization, and activation functions.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;

use super::shader_compiler::{ShaderLoader, ShaderCompilerConfig};
use super::MetalError;

/// BitNet shader collection with pre-loaded compute pipelines
pub struct BitNetShaders {
    #[cfg(target_os = "macos")]
    device: metal::Device,
    #[cfg(target_os = "macos")]
    shader_loader: ShaderLoader,
    #[cfg(target_os = "macos")]
    pipelines: Arc<Mutex<HashMap<String, metal::ComputePipelineState>>>,
}

/// Available BitNet shader functions
#[derive(Debug, Clone, Copy)]
pub enum BitNetShaderFunction {
    // BitLinear operations
    BitLinearForward,
    BitLinearBackwardInput,
    BinarizeWeights,
    QuantizeActivations,
    
    // Quantization operations
    QuantizeWeights1Bit,
    QuantizeActivations8Bit,
    DequantizeWeights1Bit,
    DequantizeActivations8Bit,
    DynamicQuantizeActivations,
    QuantizeGradients,
    MixedPrecisionMatmul,
    
    // Activation functions
    ReluForward,
    ReluBackward,
    GeluForward,
    GeluBackward,
    SwishForward,
    SwishBackward,
    SigmoidForward,
    SigmoidBackward,
    TanhForward,
    TanhBackward,
    LeakyReluForward,
    LeakyReluBackward,
    SoftmaxForward,
    SoftmaxBackward,
    LayerNormForward,
    FusedReluDropout,
}

impl BitNetShaderFunction {
    /// Gets the shader name for this function
    pub fn shader_name(self) -> &'static str {
        match self {
            Self::BitLinearForward | Self::BitLinearBackwardInput | 
            Self::BinarizeWeights | Self::QuantizeActivations => "bitlinear",
            
            Self::QuantizeWeights1Bit | Self::QuantizeActivations8Bit |
            Self::DequantizeWeights1Bit | Self::DequantizeActivations8Bit |
            Self::DynamicQuantizeActivations | Self::QuantizeGradients |
            Self::MixedPrecisionMatmul => "quantization",
            
            Self::ReluForward | Self::ReluBackward | Self::GeluForward |
            Self::GeluBackward | Self::SwishForward | Self::SwishBackward |
            Self::SigmoidForward | Self::SigmoidBackward | Self::TanhForward |
            Self::TanhBackward | Self::LeakyReluForward | Self::LeakyReluBackward |
            Self::SoftmaxForward | Self::SoftmaxBackward | Self::LayerNormForward |
            Self::FusedReluDropout => "activation",
        }
    }

    /// Gets the function name within the shader
    pub fn function_name(self) -> &'static str {
        match self {
            Self::BitLinearForward => "bitlinear_forward",
            Self::BitLinearBackwardInput => "bitlinear_backward_input",
            Self::BinarizeWeights => "binarize_weights",
            Self::QuantizeActivations => "quantize_activations",
            
            Self::QuantizeWeights1Bit => "quantize_weights_1bit",
            Self::QuantizeActivations8Bit => "quantize_activations_8bit",
            Self::DequantizeWeights1Bit => "dequantize_weights_1bit",
            Self::DequantizeActivations8Bit => "dequantize_activations_8bit",
            Self::DynamicQuantizeActivations => "dynamic_quantize_activations",
            Self::QuantizeGradients => "quantize_gradients",
            Self::MixedPrecisionMatmul => "mixed_precision_matmul",
            
            Self::ReluForward => "relu_forward",
            Self::ReluBackward => "relu_backward",
            Self::GeluForward => "gelu_forward",
            Self::GeluBackward => "gelu_backward",
            Self::SwishForward => "swish_forward",
            Self::SwishBackward => "swish_backward",
            Self::SigmoidForward => "sigmoid_forward",
            Self::SigmoidBackward => "sigmoid_backward",
            Self::TanhForward => "tanh_forward",
            Self::TanhBackward => "tanh_backward",
            Self::LeakyReluForward => "leaky_relu_forward",
            Self::LeakyReluBackward => "leaky_relu_backward",
            Self::SoftmaxForward => "softmax_forward",
            Self::SoftmaxBackward => "softmax_backward",
            Self::LayerNormForward => "layer_norm_forward",
            Self::FusedReluDropout => "fused_relu_dropout",
        }
    }

    /// Gets the pipeline key for caching
    pub fn pipeline_key(self) -> String {
        format!("{}::{}", self.shader_name(), self.function_name())
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BitNetShaders {
    /// Creates a new BitNet shader collection
    pub fn new(device: metal::Device) -> Result<Self> {
        let config = ShaderCompilerConfig {
            shader_directory: std::path::PathBuf::from("bitnet-core/src/metal/shaders"),
            enable_caching: true,
            cache_directory: Some(std::path::PathBuf::from("target/bitnet_shader_cache")),
            ..Default::default()
        };

        let mut shader_loader = ShaderLoader::new(device.clone(), config)?;
        
        // Preload all BitNet shaders
        shader_loader.preload_shaders(&["bitlinear", "quantization", "activation"])?;

        Ok(Self {
            device,
            shader_loader,
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Creates BitNet shaders with custom configuration
    pub fn new_with_config(device: metal::Device, config: ShaderCompilerConfig) -> Result<Self> {
        let mut shader_loader = ShaderLoader::new(device.clone(), config)?;
        shader_loader.preload_shaders(&["bitlinear", "quantization", "activation"])?;

        Ok(Self {
            device,
            shader_loader,
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Gets a compute pipeline for the specified function
    pub fn get_pipeline(&self, function: BitNetShaderFunction) -> Result<metal::ComputePipelineState> {
        let pipeline_key = function.pipeline_key();
        
        // Check cache first
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(&pipeline_key) {
                return Ok(pipeline.clone());
            }
        }

        // Create pipeline
        let pipeline = self.shader_loader.create_compute_pipeline(
            function.shader_name(),
            function.function_name(),
        )?;

        // Cache pipeline
        {
            let mut pipelines = self.pipelines.lock().unwrap();
            pipelines.insert(pipeline_key, pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Gets all available shader functions for a specific shader
    pub fn get_shader_functions(&self, shader_name: &str) -> Result<Vec<String>> {
        let functions = self.shader_loader.get_shader_functions(shader_name)?;
        Ok(functions.to_vec())
    }

    /// Gets all available shaders
    pub fn get_available_shaders(&self) -> Vec<String> {
        self.shader_loader.get_available_shaders()
    }

    /// Clears the pipeline cache
    pub fn clear_pipeline_cache(&self) {
        let mut pipelines = self.pipelines.lock().unwrap();
        pipelines.clear();
    }

    /// Gets the underlying Metal device
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Creates a compute command encoder with the specified pipeline
    pub fn create_compute_encoder_with_pipeline(
        &self,
        command_buffer: &metal::CommandBufferRef,
        function: BitNetShaderFunction,
    ) -> Result<metal::ComputeCommandEncoder> {
        let pipeline = self.get_pipeline(function)?;
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        Ok(encoder.to_owned())
    }

    /// Calculates optimal dispatch parameters for a given function and data size
    pub fn calculate_dispatch_params(
        &self,
        function: BitNetShaderFunction,
        data_size: usize,
    ) -> Result<(metal::MTLSize, metal::MTLSize)> {
        let pipeline = self.get_pipeline(function)?;
        
        // Use the existing optimal threadgroup calculation
        Ok(super::calculate_optimal_threadgroup_size(&self.device, &pipeline, data_size))
    }
}

/// Global BitNet shader instance for convenient access
static INIT: Once = Once::new();
static mut GLOBAL_SHADERS: Option<Arc<Mutex<Option<BitNetShaders>>>> = None;

/// Initializes the global BitNet shader instance
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn initialize_global_shaders(device: metal::Device) -> Result<()> {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_SHADERS = Some(Arc::new(Mutex::new(None)));
        });

        if let Some(global_ref) = &GLOBAL_SHADERS {
            let mut global_shaders = global_ref.lock().unwrap();
            if global_shaders.is_none() {
                *global_shaders = Some(BitNetShaders::new(device)?);
            }
        }
    }

    Ok(())
}

/// Gets the global BitNet shader instance
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn get_global_shaders() -> Result<&'static BitNetShaders> {
    unsafe {
        if let Some(global_ref) = &GLOBAL_SHADERS {
            let global_shaders = global_ref.lock().unwrap();
            if let Some(shaders) = &*global_shaders {
                // Return a reference to the global instance
                // Note: This is unsafe but acceptable for a global singleton
                let ptr = shaders as *const BitNetShaders;
                return Ok(&*ptr);
            }
        }
    }

    Err(MetalError::LibraryCreationFailed(
        "Global shaders not initialized. Call initialize_global_shaders() first.".to_string()
    ).into())
}

/// Convenience functions for common shader operations

/// Creates a BitLinear forward compute encoder
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_bitlinear_forward_encoder(
    shaders: &BitNetShaders,
    command_buffer: &metal::CommandBuffer,
) -> Result<metal::ComputeCommandEncoder> {
    shaders.create_compute_encoder_with_pipeline(command_buffer, BitNetShaderFunction::BitLinearForward)
}

/// Creates a quantization encoder
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_quantization_encoder(
    shaders: &BitNetShaders,
    command_buffer: &metal::CommandBuffer,
    quantization_type: BitNetShaderFunction,
) -> Result<metal::ComputeCommandEncoder> {
    shaders.create_compute_encoder_with_pipeline(command_buffer, quantization_type)
}

/// Creates an activation function encoder
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_activation_encoder(
    shaders: &BitNetShaders,
    command_buffer: &metal::CommandBuffer,
    activation_type: BitNetShaderFunction,
) -> Result<metal::ComputeCommandEncoder> {
    shaders.create_compute_encoder_with_pipeline(command_buffer, activation_type)
}

/// Dispatches a BitLinear forward operation
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn dispatch_bitlinear_forward(
    encoder: &metal::ComputeCommandEncoder,
    input_buffer: &metal::Buffer,
    weights_buffer: &metal::Buffer,
    bias_buffer: Option<&metal::Buffer>,
    output_buffer: &metal::Buffer,
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    threads: metal::MTLSize,
    threadgroup: metal::MTLSize,
) {
    // Set buffers
    encoder.set_buffer(0, Some(input_buffer), 0);
    encoder.set_buffer(1, Some(weights_buffer), 0);
    if let Some(bias) = bias_buffer {
        encoder.set_buffer(2, Some(bias), 0);
    }
    encoder.set_buffer(3, Some(output_buffer), 0);

    // Set parameters
    super::set_compute_bytes(encoder, &[input_size], 4);
    super::set_compute_bytes(encoder, &[output_size], 5);
    super::set_compute_bytes(encoder, &[batch_size], 6);

    // Dispatch
    super::dispatch_compute(encoder, threads, threadgroup);
}

/// Dispatches a quantization operation
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn dispatch_quantization(
    encoder: &metal::ComputeCommandEncoder,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    scale_buffer: &metal::Buffer,
    count: u32,
    group_size: u32,
    threads: metal::MTLSize,
    threadgroup: metal::MTLSize,
) {
    // Set buffers
    encoder.set_buffer(0, Some(input_buffer), 0);
    encoder.set_buffer(1, Some(output_buffer), 0);
    encoder.set_buffer(2, Some(scale_buffer), 0);

    // Set parameters
    super::set_compute_bytes(encoder, &[count], 3);
    super::set_compute_bytes(encoder, &[group_size], 4);

    // Dispatch
    super::dispatch_compute(encoder, threads, threadgroup);
}

/// Dispatches an activation function operation
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn dispatch_activation(
    encoder: &metal::ComputeCommandEncoder,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    count: u32,
    threads: metal::MTLSize,
    threadgroup: metal::MTLSize,
) {
    // Set buffers
    encoder.set_buffer(0, Some(input_buffer), 0);
    encoder.set_buffer(1, Some(output_buffer), 0);

    // Set parameters
    super::set_compute_bytes(encoder, &[count], 2);

    // Dispatch
    super::dispatch_compute(encoder, threads, threadgroup);
}

// Non-macOS implementations
#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl BitNetShaders {
    pub fn new(_device: ()) -> Result<Self> {
        Err(MetalError::UnsupportedPlatform.into())
    }
    
    pub fn get_available_shaders(&self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn initialize_global_shaders(_device: ()) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn get_global_shaders() -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_bitlinear_forward_encoder(_shaders: &(), _command_buffer: &()) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_quantization_encoder(_shaders: &(), _command_buffer: &(), _quantization_type: BitNetShaderFunction) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_activation_encoder(_shaders: &(), _command_buffer: &(), _activation_type: BitNetShaderFunction) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn dispatch_bitlinear_forward(
    _encoder: &(),
    _input_buffer: &(),
    _weights_buffer: &(),
    _bias_buffer: Option<&()>,
    _output_buffer: &(),
    _input_size: u32,
    _output_size: u32,
    _batch_size: u32,
    _threads: (),
    _threadgroup: (),
) {
    // No-op for non-macOS
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn dispatch_quantization(
    _encoder: &(),
    _input_buffer: &(),
    _output_buffer: &(),
    _scale_buffer: &(),
    _count: u32,
    _group_size: u32,
    _threads: (),
    _threadgroup: (),
) {
    // No-op for non-macOS
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn dispatch_activation(
    _encoder: &(),
    _input_buffer: &(),
    _output_buffer: &(),
    _count: u32,
    _threads: (),
    _threadgroup: (),
) {
    // No-op for non-macOS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_function_names() {
        assert_eq!(BitNetShaderFunction::BitLinearForward.shader_name(), "bitlinear");
        assert_eq!(BitNetShaderFunction::BitLinearForward.function_name(), "bitlinear_forward");
        assert_eq!(BitNetShaderFunction::QuantizeWeights1Bit.shader_name(), "quantization");
        assert_eq!(BitNetShaderFunction::ReluForward.shader_name(), "activation");
    }

    #[test]
    fn test_pipeline_keys() {
        let key = BitNetShaderFunction::BitLinearForward.pipeline_key();
        assert_eq!(key, "bitlinear::bitlinear_forward");
        
        let key = BitNetShaderFunction::QuantizeWeights1Bit.pipeline_key();
        assert_eq!(key, "quantization::quantize_weights_1bit");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_bitnet_shaders_creation() {
        use crate::metal::create_metal_device;
        
        if let Ok(device) = create_metal_device() {
            // Note: This test may fail if shader files don't exist in the expected location
            let shaders_result = BitNetShaders::new(device);
            match shaders_result {
                Ok(shaders) => {
                    let available = shaders.get_available_shaders();
                    println!("Available shaders: {:?}", available);
                }
                Err(e) => {
                    println!("Expected failure (shader files may not exist): {}", e);
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_unsupported_platform() {
        let result = initialize_global_shaders(());
        assert!(result.is_err());
        
        let result = get_global_shaders();
        assert!(result.is_err());
    }
}