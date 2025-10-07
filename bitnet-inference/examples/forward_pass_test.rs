//! Test the forward pass implementations with proper input validation
//!
//! This example tests all layer types with appropriate inputs to verify
//! the forward pass implementations are working correctly.

use bitnet_inference::{InferenceIntegration, BitNetModelConfig};
use bitnet_inference::engine::LayerOperation;
use bitnet_inference::bitnet_config::*;
use bitnet_inference::engine::model_loader::{ModelWeights, ParameterType, ParameterData, ParameterDataType, LayerDefinition, LayerType, LayerParameters};
use bitnet_inference::engine::weight_conversion::WeightConverter;
use bitnet_core::Tensor;
use candle_core::{Device, Shape};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Forward Pass Implementation Test ===");

    // Test individual layer operations with proper inputs
    test_embedding_layer().await?;
    test_rms_norm_layer().await?;
    test_bitlinear_layer().await?;
    test_swiglu_layer().await?;

    println!("\n✅ All forward pass tests completed successfully!");
    
    Ok(())
}

async fn test_embedding_layer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Embedding Layer ===");
    
    // Create embedding layer config
    let vocab_size = 1000;
    let embed_dim = 128;
    
    // Create weight converter and weights
    let converter = Arc::new(WeightConverter::with_default_cache());
    let mut weights = ModelWeights::with_converter(converter.clone());
    
    // Create embedding weights (vocab_size * embed_dim)
    let mut embedding_data = Vec::new();
    for i in 0..(vocab_size * embed_dim) {
        embedding_data.extend_from_slice(&(i as f32 * 0.01).to_le_bytes());
    }
    
    let embedding_param = ParameterData {
        data: embedding_data,
        shape: vec![vocab_size, embed_dim],
        dtype: ParameterDataType::F32,
        tensor_name: "embed.weight".to_string(),
    };
    
    weights.add_parameter(0, ParameterType::EmbeddingWeight, embedding_param);
    
    // Create config
    let config = create_simple_config(vocab_size, embed_dim);
    
    // Create integration
    let integration = Arc::new(InferenceIntegration::new(config, weights)?);
    
    // Create embedding layer definition
    let layer_def = LayerDefinition {
        id: 0,
        layer_type: LayerType::Embedding,
        input_dims: vec![4], // sequence length
        output_dims: vec![4, embed_dim], // seq_len, embed_dim
        parameters: LayerParameters::Embedding {
            vocab_size,
            embedding_dim: embed_dim,
        },
    };
    
    // Create layer operation
    let layer_op = LayerOperation::from_layer_definition(&layer_def, integration)?;
    
    // Create test input with valid token IDs
    let device = Device::Cpu;
    let token_ids = vec![0u32, 42, 123, 999]; // Valid token IDs within vocab range
    let input_tensor = Tensor::from_vec(token_ids, Shape::from_dims(&[1, 4]), &device)?;
    
    // Execute embedding lookup
    let output = layer_op.execute(input_tensor)?;
    let output_shape = output.shape().dims();
    
    println!("✅ Embedding layer executed successfully");
    println!("   Input shape: [1, 4] (batch_size=1, seq_len=4)");
    println!("   Output shape: {:?} (batch_size, seq_len, embed_dim)", output_shape);
    
    // Verify output shape
    assert_eq!(output_shape, &[1, 4, embed_dim]);
    println!("✅ Output shape validation passed");
    
    Ok(())
}

async fn test_rms_norm_layer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing RMSNorm Layer ===");
    
    let hidden_dim = 128;
    
    // Create weight converter and weights
    let converter = Arc::new(WeightConverter::with_default_cache());
    let mut weights = ModelWeights::with_converter(converter.clone());
    
    // Create RMS norm scale weights (all ones for simplicity)
    let scale_data: Vec<u8> = (0..hidden_dim)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();
    
    let scale_param = ParameterData {
        data: scale_data,
        shape: vec![hidden_dim],
        dtype: ParameterDataType::F32,
        tensor_name: "norm.weight".to_string(),
    };
    
    weights.add_parameter(0, ParameterType::LayerNormScale, scale_param);
    
    // Create config
    let config = create_simple_config(1000, hidden_dim);
    
    // Create integration
    let integration = Arc::new(InferenceIntegration::new(config, weights)?);
    
    // Create RMS norm layer definition
    let layer_def = LayerDefinition {
        id: 0,
        layer_type: LayerType::RMSNorm,
        input_dims: vec![1, hidden_dim],
        output_dims: vec![1, hidden_dim],
        parameters: LayerParameters::RMSNorm { eps: 1e-6 },
    };
    
    // Create layer operation
    let layer_op = LayerOperation::from_layer_definition(&layer_def, integration)?;
    
    // Create test input with random values
    let device = Device::Cpu;
    let input_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let input_tensor = Tensor::from_vec(input_data.clone(), Shape::from_dims(&[1, hidden_dim]), &device)?;
    
    // Execute RMS normalization
    let output = layer_op.execute(input_tensor)?;
    let output_shape = output.shape().dims();
    let output_data = output.to_vec1::<f32>()?;
    
    println!("✅ RMSNorm layer executed successfully");
    println!("   Input shape: [1, {}]", hidden_dim);
    println!("   Output shape: {:?}", output_shape);
    
    // Verify output shape
    assert_eq!(output_shape, &[1, hidden_dim]);
    
    // Verify normalization (output should have normalized variance)
    let mean_squared: f32 = output_data.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
    println!("   Output mean squared: {:.6} (should be close to 1.0)", mean_squared);
    assert!((mean_squared - 1.0).abs() < 0.1, "RMS normalization failed");
    
    println!("✅ RMSNorm validation passed");
    
    Ok(())
}

async fn test_bitlinear_layer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing BitLinear Layer ===");
    
    let input_dim = 64;
    let output_dim = 32;
    
    // Create weight converter and weights
    let converter = Arc::new(WeightConverter::with_default_cache());
    let mut weights = ModelWeights::with_converter(converter.clone());
    
    // Create ternary weights {-1, 0, +1}
    let ternary_values: Vec<i8> = (0..(input_dim * output_dim))
        .map(|i| match i % 3 { 0 => -1, 1 => 0, 2 => 1, _ => 0 })
        .collect();
    
    // Pack ternary weights into bytes (this is simplified - real implementation uses 2-bit packing)
    let weight_data: Vec<u8> = ternary_values.iter().map(|&x| x as u8).collect();
    
    let weight_param = ParameterData {
        data: weight_data,
        shape: vec![input_dim, output_dim],
        dtype: ParameterDataType::BitnetB158,
        tensor_name: "linear.weight".to_string(),
    };
    
    weights.add_parameter(0, ParameterType::Weight, weight_param);
    
    // Create config
    let config = create_simple_config(1000, input_dim);
    
    // Create integration
    let integration = Arc::new(InferenceIntegration::new(config, weights)?);
    
    // Create BitLinear layer definition
    let layer_def = LayerDefinition {
        id: 0,
        layer_type: LayerType::BitLinear,
        input_dims: vec![1, input_dim],
        output_dims: vec![1, output_dim],
        parameters: LayerParameters::BitLinear {
            weight_bits: 2,
            activation_bits: 8,
        },
    };
    
    // Create layer operation
    let layer_op = LayerOperation::from_layer_definition(&layer_def, integration)?;
    
    // Create test input
    let device = Device::Cpu;
    let input_data: Vec<f32> = (0..input_dim).map(|i| (i as f32) * 0.01).collect();
    let input_tensor = Tensor::from_vec(input_data, Shape::from_dims(&[1, input_dim]), &device)?;
    
    // Execute BitLinear layer
    let output = layer_op.execute(input_tensor)?;
    let output_shape = output.shape().dims();
    
    println!("✅ BitLinear layer executed successfully");
    println!("   Input shape: [1, {}]", input_dim);
    println!("   Output shape: {:?}", output_shape);
    
    // Verify output shape
    assert_eq!(output_shape, &[1, output_dim]);
    println!("✅ BitLinear shape validation passed");
    
    Ok(())
}

async fn test_swiglu_layer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing SwiGLU Layer ===");
    
    let input_dim = 64;
    let hidden_dim = 128;
    
    // Create weight converter and weights
    let converter = Arc::new(WeightConverter::with_default_cache());
    let mut weights = ModelWeights::with_converter(converter.clone());
    
    // Create gate and up projection weights
    let gate_data: Vec<u8> = (0..(input_dim * hidden_dim))
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    
    let up_data: Vec<u8> = (0..(input_dim * hidden_dim))
        .flat_map(|i| ((i as f32) * 0.002).to_le_bytes())
        .collect();
    
    let gate_param = ParameterData {
        data: gate_data,
        shape: vec![input_dim, hidden_dim],
        dtype: ParameterDataType::F32,
        tensor_name: "gate.weight".to_string(),
    };
    
    let up_param = ParameterData {
        data: up_data,
        shape: vec![input_dim, hidden_dim],
        dtype: ParameterDataType::F32,
        tensor_name: "up.weight".to_string(),
    };
    
    weights.add_parameter(0, ParameterType::FeedForwardGate, gate_param);
    weights.add_parameter(0, ParameterType::FeedForwardUp, up_param);
    
    // Create config
    let config = create_simple_config(1000, input_dim);
    
    // Create integration
    let integration = Arc::new(InferenceIntegration::new(config, weights)?);
    
    // Create SwiGLU layer definition
    let layer_def = LayerDefinition {
        id: 0,
        layer_type: LayerType::SwiGLU,
        input_dims: vec![1, input_dim],
        output_dims: vec![1, hidden_dim],
        parameters: LayerParameters::SwiGLU { hidden_dim },
    };
    
    // Create layer operation
    let layer_op = LayerOperation::from_layer_definition(&layer_def, integration)?;
    
    // Create test input
    let device = Device::Cpu;
    let input_data: Vec<f32> = (0..input_dim).map(|i| (i as f32) * 0.01).collect();
    let input_tensor = Tensor::from_vec(input_data, Shape::from_dims(&[1, input_dim]), &device)?;
    
    // Execute SwiGLU layer
    let output = layer_op.execute(input_tensor)?;
    let output_shape = output.shape().dims();
    
    println!("✅ SwiGLU layer executed successfully");
    println!("   Input shape: [1, {}]", input_dim);
    println!("   Output shape: {:?}", output_shape);
    
    // Verify output shape
    assert_eq!(output_shape, &[1, hidden_dim]);
    println!("✅ SwiGLU shape validation passed");
    
    Ok(())
}

fn create_simple_config(vocab_size: usize, hidden_size: usize) -> BitNetModelConfig {
    BitNetModelConfig {
        basic_info: BasicModelInfo {
            name: "test-forward-pass".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            version: "1.0.0".to_string(),
            parameter_count: 100_000,
            context_length: 1024,
        },
        layer_config: LayerConfig {
            n_layers: 1,
            hidden_size,
            intermediate_size: hidden_size * 4,
            model_dim: hidden_size,
        },
        attention_config: AttentionConfig {
            n_heads: 8,
            n_kv_heads: None,
            head_dim: hidden_size / 8,
            max_seq_len: 1024,
            rope_config: RopeConfig {
                rope_freq_base: 10000.0,
                rope_scaling: None,
                rope_dim: 64,
            },
        },
        normalization_config: NormalizationConfig {
            rms_norm_eps: 1e-6,
            use_bias: false,
        },
        bitlinear_config: BitLinearConfig {
            weight_bits: 2,
            activation_bits: 8,
            use_weight_scaling: true,
            use_activation_scaling: false,
            quantization_scheme: "bitnet-1.58".to_string(),
        },
        tokenizer_config: TokenizerConfig {
            vocab_size,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: Some(0),
        },
        extra_metadata: HashMap::new(),
    }
}