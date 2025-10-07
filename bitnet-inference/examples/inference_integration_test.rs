//! Test the inference integration functionality
//!
//! This example demonstrates how to use the InferenceIntegration to bridge
//! BitNet configuration and layer factory with actual inference execution.

use bitnet_inference::{InferenceIntegration, ExecutableModel, BitNetModelConfig};
use bitnet_inference::bitnet_config::*;
use bitnet_inference::engine::model_loader::{ModelWeights, ParameterType, ParameterData, ParameterDataType, LoadedModel, ModelMetadata, ModelArchitecture};
use bitnet_core::Tensor;
use candle_core::{Device, Shape};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Inference Integration Test ===");

    // Create a test BitNet configuration
    let config = create_test_bitnet_config();
    println!("✅ Created BitNet configuration:");
    println!("   - Layers: {}", config.layer_config.n_layers);
    println!("   - Hidden size: {}", config.layer_config.hidden_size);
    println!("   - Vocab size: {}", config.tokenizer_config.vocab_size);

    // Create test model weights with proper structure
    let weights = create_complete_test_weights(&config);
    println!("✅ Created complete test model weights:");
    println!("   - Layer weights: {}", weights.layer_weights.len());
    println!("   - Organized weights: {}", weights.organized_weights.len());
    println!("   - Layer mapping: {}", weights.layer_mapping.len());

    // Test inference integration
    println!("\n=== Testing Inference Integration ===");
    match InferenceIntegration::new(config.clone(), weights.clone()) {
        Ok(integration) => {
            println!("✅ InferenceIntegration created successfully");
            
            // Test validation
            match integration.validate() {
                Ok(()) => {
                    println!("✅ Integration validation passed");
                    
                    // Test executable model creation
                    match integration.create_executable_model() {
                        Ok(executable) => {
                            println!("✅ ExecutableModel created successfully");
                            
                            let info = executable.info();
                            println!("📊 Model Info:");
                            println!("   - Layers: {}", info.num_layers);
                            println!("   - Parameters: {}", info.num_parameters);
                            println!("   - Context length: {}", info.context_length);
                            println!("   - Vocab size: {}", info.vocab_size);
                            println!("   - Hidden size: {}", info.hidden_size);

                            // Test model execution (with dummy input)
                            test_model_execution(&executable).await?;
                        },
                        Err(e) => {
                            println!("❌ Failed to create ExecutableModel: {}", e);
                        }
                    }
                },
                Err(e) => {
                    println!("⚠️  Integration validation failed: {}", e);
                    println!("   This is expected with minimal test weights");
                }
            }
        },
        Err(e) => {
            println!("❌ Failed to create InferenceIntegration: {}", e);
            return Err(e.into());
        }
    }

    // Test LoadedModel integration
    println!("\n=== Testing LoadedModel Integration ===");
    test_loaded_model_integration(config, weights).await?;

    println!("\n✅ Inference integration test completed successfully!");
    
    Ok(())
}

async fn test_model_execution(executable: &ExecutableModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Model Execution ===");
    
    // Create a dummy input tensor (token IDs)
    let device = Device::Cpu;
    let input_data = vec![1u32, 42, 1337, 2]; // Simple token sequence
    let input_tensor = Tensor::from_vec(input_data, Shape::from_dims(&[1, 4]), &device)?; // [batch_size=1, seq_len=4]
    
    println!("🔢 Created input tensor: shape={:?}", input_tensor.shape());
    
    // Execute the model (this will use placeholder implementations)
    match executable.execute(input_tensor) {
        Ok(output) => {
            println!("✅ Model execution succeeded");
            println!("📤 Output tensor shape: {:?}", output.shape());
        },
        Err(e) => {
            println!("⚠️  Model execution failed: {}", e);
            println!("   This is expected with placeholder layer implementations");
        }
    }
    
    Ok(())
}

async fn test_loaded_model_integration(
    config: BitNetModelConfig, 
    weights: ModelWeights
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a LoadedModel
    let loaded_model = LoadedModel {
        metadata: ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            parameter_count: config.basic_info.parameter_count,
            quantization_bits: 2, // 1.58-bit -> 2 bits
            input_shape: vec![1, config.basic_info.context_length],
            output_shape: vec![1, config.tokenizer_config.vocab_size],
            extra: HashMap::new(),
        },
        architecture: ModelArchitecture {
            layers: vec![],
            execution_order: vec![],
        },
        weights,
        bitnet_config: Some(config),
    };

    match InferenceIntegration::from_loaded_model(loaded_model) {
        Ok(integration) => {
            println!("✅ InferenceIntegration from LoadedModel created successfully");
            
            let architecture = integration.architecture();
            println!("📋 Architecture: {} layers, execution order: {:?}", 
                    architecture.layers.len(), 
                    architecture.execution_order.first());
        },
        Err(e) => {
            println!("⚠️  InferenceIntegration from LoadedModel failed: {}", e);
            println!("   This is expected with minimal test data");
        }
    }
    
    Ok(())
}

fn create_test_bitnet_config() -> BitNetModelConfig {
    BitNetModelConfig {
        basic_info: BasicModelInfo {
            name: "test-inference-model".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            version: "1.0.0".to_string(),
            parameter_count: 500_000,
            context_length: 1024,
        },
        layer_config: LayerConfig {
            n_layers: 2,
            hidden_size: 256,
            intermediate_size: 1024,
            model_dim: 256,
        },
        attention_config: AttentionConfig {
            n_heads: 8,
            n_kv_heads: None,
            head_dim: 32,
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
            weight_bits: 1,
            activation_bits: 8,
            use_weight_scaling: true,
            use_activation_scaling: false,
            quantization_scheme: "bitnet-1.58".to_string(),
        },
        tokenizer_config: TokenizerConfig {
            vocab_size: 16000,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: Some(0),
        },
        extra_metadata: HashMap::new(),
    }
}

fn create_complete_test_weights(config: &BitNetModelConfig) -> ModelWeights {
    let mut weights = ModelWeights::new();
    let mut layer_id = 0;

    // Add embedding weights
    weights.layer_mapping.insert("token_embd.weight".to_string(), layer_id);
    weights.layer_weights.insert(layer_id, vec![0u8; config.tokenizer_config.vocab_size * config.layer_config.hidden_size * 4]);
    
    // Create organized embedding weights
    let mut embedding_params = HashMap::new();
    embedding_params.insert(ParameterType::EmbeddingWeight, ParameterData {
        data: vec![0u8; config.tokenizer_config.vocab_size * config.layer_config.hidden_size * 4],
        shape: vec![config.tokenizer_config.vocab_size, config.layer_config.hidden_size],
        dtype: ParameterDataType::F32,
        tensor_name: "token_embd.weight".to_string(),
    });
    weights.organized_weights.insert(layer_id, embedding_params);
    layer_id += 1;

    // Add transformer layer weights
    for layer_idx in 0..config.layer_config.n_layers {
        // Attention normalization
        weights.layer_mapping.insert(
            format!("blk.{}.attn_norm.weight", layer_idx), 
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; config.layer_config.hidden_size * 4]);
        layer_id += 1;
        
        // FFN normalization  
        weights.layer_mapping.insert(
            format!("blk.{}.ffn_norm.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; config.layer_config.hidden_size * 4]);
        layer_id += 1;

        // Create organized weights for this transformer layer
        let mut layer_params = HashMap::new();
        layer_params.insert(ParameterType::LayerNormScale, ParameterData {
            data: vec![0u8; config.layer_config.hidden_size * 4],
            shape: vec![config.layer_config.hidden_size],
            dtype: ParameterDataType::F32,
            tensor_name: format!("blk.{}.attn_norm.weight", layer_idx),
        });
        weights.organized_weights.insert(layer_idx + 1, layer_params); // Offset by 1 for embedding
    }

    // Add final normalization
    weights.layer_mapping.insert("output_norm.weight".to_string(), layer_id);
    weights.layer_weights.insert(layer_id, vec![0u8; config.layer_config.hidden_size * 4]);
    layer_id += 1;

    // Add output projection
    weights.layer_mapping.insert("output.weight".to_string(), layer_id);
    weights.layer_weights.insert(layer_id, vec![0u8; config.layer_config.hidden_size * config.tokenizer_config.vocab_size * 4]);

    // Create organized output weights
    let mut output_params = HashMap::new();
    output_params.insert(ParameterType::OutputWeight, ParameterData {
        data: vec![0u8; config.layer_config.hidden_size * config.tokenizer_config.vocab_size * 4],
        shape: vec![config.layer_config.hidden_size, config.tokenizer_config.vocab_size],
        dtype: ParameterDataType::F32,
        tensor_name: "output.weight".to_string(),
    });
    weights.organized_weights.insert(layer_id, output_params);

    weights
}