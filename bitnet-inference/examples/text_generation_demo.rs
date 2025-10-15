//! Text Generation Example
//!
//! This example demonstrates the usage of the text generation API
//! for BitNet models with configurable generation parameters.

use anyhow::Result;
use bitnet_inference::{
    InferenceEngine, 
    api::{GenerationConfig, TextGeneratorBuilder}
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("BitNet Text Generation Example");
    println!("=============================");

    // Initialize the inference engine
    let engine = Arc::new(InferenceEngine::new().await?);
    println!("✓ Inference engine initialized");

    // Load a model (placeholder - would load a real GGUF model)
    let model_path = "models/microsoft/bitnet-b1.58-2B-4T-gguf";
    println!("Loading model from: {}", model_path);
    
    // For this example, we'll create a mock model since we don't have a real one
    let model = create_mock_model();
    println!("✓ Model loaded successfully");

    // Example 1: Quick text generation with defaults
    println!("\n1. Quick Text Generation (Default Parameters):");
    let prompt = "The future of AI is";
    println!("Prompt: \"{}\"", prompt);
    
    let result = engine.generate_text(model.clone(), prompt).await?;
    println!("Generated: \"{}\"", result);

    // Example 2: Custom generation parameters
    println!("\n2. Custom Generation Parameters:");
    let custom_config = GenerationConfig {
        temperature: 0.7,
        top_k: Some(40),
        top_p: Some(0.9),
        max_length: 100,
        do_sample: true,
        stop_tokens: vec![".", "!", "?"].iter().map(|s| s.to_string()).collect(),
        seed: Some(42),
        ..Default::default()
    };

    let prompt2 = "BitNet quantization provides";
    println!("Prompt: \"{}\"", prompt2);
    println!("Config: temperature={}, top_k={:?}, top_p={:?}", 
             custom_config.temperature, custom_config.top_k, custom_config.top_p);

    let result2 = engine.generate_text_with_config(
        model.clone(), 
        prompt2, 
        custom_config.clone()
    ).await?;
    
    println!("Generated: \"{}\"", result2.text);
    println!("Tokens generated: {}", result2.token_count);
    println!("Generation time: {}ms", result2.generation_time_ms);
    println!("Finished reason: {:?}", result2.finished_reason);

    // Example 3: Using the TextGenerator builder pattern
    println!("\n3. Builder Pattern with Fine-Tuned Parameters:");
    let generator = TextGeneratorBuilder::new()
        .with_temperature(0.3)  // Lower temperature for more focused output
        .with_top_k(Some(20))   // Smaller top-k for more deterministic output
        .with_max_length(50)    // Shorter generation
        .with_stop_tokens(vec!["<|endoftext|>".to_string()])
        .build(create_mock_tokenizer_config())?;

    let prompt3 = "Machine learning models";
    println!("Prompt: \"{}\"", prompt3);
    println!("Builder config: temperature=0.3, top_k=20, max_length=50");

    let result3 = generator.generate(prompt3).await?;
    println!("Generated: \"{}\"", result3.text);
    println!("Performance: {} tokens in {}ms", result3.token_count, result3.generation_time_ms);

    // Example 4: Different generation strategies
    println!("\n4. Different Generation Strategies:");
    
    // Greedy decoding (deterministic)
    let greedy_config = GenerationConfig {
        temperature: 1.0,
        top_k: None,
        top_p: None,
        max_length: 30,
        do_sample: false,  // Greedy decoding
        stop_tokens: vec![],
        seed: None,
        ..Default::default()
    };

    let prompt4 = "Neural networks are";
    println!("Greedy decoding - Prompt: \"{}\"", prompt4);
    let greedy_result = engine.generate_text_with_config(
        model.clone(), 
        prompt4, 
        greedy_config
    ).await?;
    println!("Greedy result: \"{}\"", greedy_result.text);

    // High temperature (creative)
    let creative_config = GenerationConfig {
        temperature: 1.5,  // High temperature
        top_k: Some(100),
        top_p: Some(0.95),
        max_length: 30,
        do_sample: true,
        stop_tokens: vec![],
        seed: Some(123),
        ..Default::default()
    };

    println!("High temperature - Prompt: \"{}\"", prompt4);
    let creative_result = engine.generate_text_with_config(
        model.clone(), 
        prompt4, 
        creative_config
    ).await?;
    println!("Creative result: \"{}\"", creative_result.text);

    println!("\n✓ Text generation examples completed successfully!");

    Ok(())
}

/// Create a mock model for demonstration purposes
fn create_mock_model() -> Arc<bitnet_inference::engine::Model> {
    use bitnet_inference::engine::{Model, ModelArchitecture, LayerConfig, LayerType, LayerParameters, QuantizationConfig};
    
    Arc::new(Model {
        name: "bitnet-b1.58-2B-demo".to_string(),
        version: "1.0".to_string(),
        input_dim: 2048,
        output_dim: 128256, // LLaMA 3 vocab size
        parameter_count: 2_000_000_000, // 2B parameters
        quantization_config: QuantizationConfig {
            weight_bits: 1,
            activation_bits: 8,
            symmetric: true,
            per_channel: true,
        },
        architecture: ModelArchitecture::BitLinear {
            layers: vec![
                LayerConfig {
                    id: 0,
                    layer_type: LayerType::BitLinear,
                    input_shape: vec![1, 2048],
                    output_shape: vec![1, 2048],
                    parameters: LayerParameters::BitLinear {
                        weight_bits: 1,
                        activation_bits: 8,
                    },
                },
            ],
            attention_heads: Some(32),
            hidden_dim: 2048,
        },
    })
}

/// Create a mock tokenizer config for demonstration
fn create_mock_tokenizer_config() -> bitnet_inference::bitnet_config::TokenizerConfig {
    bitnet_inference::bitnet_config::TokenizerConfig {
        vocab_size: 128256,
        tokenizer_type: "llama3".to_string(),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        pad_token_id: None,
    }
}