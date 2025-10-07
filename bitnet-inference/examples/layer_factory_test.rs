//! Test the layer factory functionality with BitNet configuration
//!
//! This example demonstrates how to use the LayerFactory to construct model architecture
//! from BitNet configuration and organized weights.

use bitnet_inference::{LayerFactoryBuilder, BitNetModelConfig};
use bitnet_inference::bitnet_config::*;
use bitnet_inference::engine::model_loader::{ModelWeights, ParameterType, ParameterData, ParameterDataType};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Layer Factory Configuration Test ===");

    // Create a test BitNet configuration
    let config = create_test_bitnet_config();
    println!("✅ Created BitNet configuration:");
    println!("   - Layers: {}", config.layer_config.n_layers);
    println!("   - Hidden size: {}", config.layer_config.hidden_size);
    println!("   - Vocab size: {}", config.tokenizer_config.vocab_size);
    println!("   - Weight bits: {}", config.bitlinear_config.weight_bits);

    // Create test model weights
    let weights = create_test_model_weights();
    println!("✅ Created test model weights:");
    println!("   - Layer weights: {}", weights.layer_weights.len());
    println!("   - Organized weights: {}", weights.organized_weights.len());

    // Test LayerFactoryBuilder
    println!("\n=== Testing LayerFactoryBuilder ===");
    let factory_result = LayerFactoryBuilder::new()
        .with_config(config.clone())
        .with_weights(weights.clone())
        .build();

    match factory_result {
        Ok(factory) => {
            println!("✅ LayerFactory created successfully");
            
            // Test architecture building
            println!("\n=== Building Model Architecture ===");
            match factory.build_model_architecture() {
                Ok(architecture) => {
                    println!("✅ Model architecture built successfully:");
                    println!("   - Total layers: {}", architecture.layers.len());
                    println!("   - Execution order length: {}", architecture.execution_order.len());
                    
                    // Print layer details
                    for (i, layer) in architecture.layers.iter().enumerate() {
                        println!("   - Layer {}: {:?} (ID: {})", i, layer.layer_type, layer.id);
                    }
                },
                Err(e) => {
                    println!("❌ Failed to build architecture: {}", e);
                    return Err(e.into());
                }
            }
        },
        Err(e) => {
            println!("❌ Failed to create LayerFactory: {}", e);
            return Err(e.into());
        }
    }

    println!("\n=== Testing Weight Validation ===");
    // Test different weight scenarios
    test_missing_weights_scenario().await?;
    
    println!("\n✅ Layer factory test completed successfully!");
    
    Ok(())
}

fn create_test_bitnet_config() -> BitNetModelConfig {
    BitNetModelConfig {
        basic_info: BasicModelInfo {
            name: "test-bitnet-model".to_string(),
            architecture: "bitnet-b1.58".to_string(),
            version: "1.0.0".to_string(),
            parameter_count: 2_000_000,
            context_length: 2048,
        },
        layer_config: LayerConfig {
            n_layers: 3,
            hidden_size: 768,
            intermediate_size: 3072,
            model_dim: 768,
        },
        attention_config: AttentionConfig {
            n_heads: 12,
            n_kv_heads: None,
            head_dim: 64,
            max_seq_len: 2048,
            rope_config: RopeConfig {
                rope_freq_base: 10000.0,
                rope_scaling: None,
                rope_dim: 128,
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
            vocab_size: 32000,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: Some(0),
        },
        extra_metadata: HashMap::new(),
    }
}

fn create_test_model_weights() -> ModelWeights {
    let mut weights = ModelWeights::new();

    // Add layer mapping for tensor names to layer IDs
    weights.layer_mapping.insert("token_embd.weight".to_string(), 0);
    let mut layer_id = 1;

    // Add transformer layer weights
    for layer_idx in 0..3 {
        // Attention normalization
        weights.layer_mapping.insert(
            format!("blk.{}.attn_norm.weight", layer_idx), 
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 4]);
        layer_id += 1;
        
        // FFN normalization  
        weights.layer_mapping.insert(
            format!("blk.{}.ffn_norm.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 4]);
        layer_id += 1;

        // Attention weights
        weights.layer_mapping.insert(
            format!("blk.{}.attn_q.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 768 / 4]);
        layer_id += 1;

        weights.layer_mapping.insert(
            format!("blk.{}.attn_k.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 768 / 4]);
        layer_id += 1;

        weights.layer_mapping.insert(
            format!("blk.{}.attn_v.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 768 / 4]);
        layer_id += 1;

        weights.layer_mapping.insert(
            format!("blk.{}.attn_output.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 768 / 4]);
        layer_id += 1;

        // FFN weights
        weights.layer_mapping.insert(
            format!("blk.{}.ffn_gate.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 3072 / 4]);
        layer_id += 1;

        weights.layer_mapping.insert(
            format!("blk.{}.ffn_up.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 768 * 3072 / 4]);
        layer_id += 1;

        weights.layer_mapping.insert(
            format!("blk.{}.ffn_down.weight", layer_idx),
            layer_id
        );
        weights.layer_weights.insert(layer_id, vec![0u8; 3072 * 768 / 4]);
        layer_id += 1;

        // Create organized weights for this layer
        let mut layer_params = HashMap::new();
        layer_params.insert(ParameterType::LayerNormScale, ParameterData {
            data: vec![0u8; 768 * 4],
            shape: vec![768],
            dtype: ParameterDataType::F32,
            tensor_name: format!("blk.{}.attn_norm.weight", layer_idx),
        });

        weights.organized_weights.insert(layer_idx, layer_params);
    }

    // Add final normalization
    weights.layer_mapping.insert("output_norm.weight".to_string(), layer_id);
    weights.layer_weights.insert(layer_id, vec![0u8; 768 * 4]);
    layer_id += 1;

    // Add output projection
    weights.layer_mapping.insert("output.weight".to_string(), layer_id);
    weights.layer_weights.insert(layer_id, vec![0u8; 768 * 32000 * 4]);

    // Add embedding weights  
    weights.layer_weights.insert(0, vec![0u8; 32000 * 768 * 4]);

    weights
}

async fn test_missing_weights_scenario() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing scenario with missing weights...");
    
    let config = create_test_bitnet_config();
    
    // Create weights with missing components
    let mut incomplete_weights = ModelWeights::new();
    
    // Only add some weights, missing others
    incomplete_weights.layer_mapping.insert(
        "token_embd.weight".to_string(), 
        0
    );
    incomplete_weights.layer_weights.insert(0, vec![0u8; 1000]);

    let factory = LayerFactoryBuilder::new()
        .with_config(config)
        .with_weights(incomplete_weights)
        .build()?;

    match factory.build_model_architecture() {
        Ok(_) => {
            println!("⚠️  Architecture built despite missing weights (unexpected)");
        },
        Err(e) => {
            println!("✅ Properly detected missing weights: {}", e);
        }
    }
    
    Ok(())
}