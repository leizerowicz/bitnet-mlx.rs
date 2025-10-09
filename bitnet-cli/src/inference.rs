//! Inference command implementations for BitNet CLI

use anyhow::Result;
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, TextGeneratorBuilder, GenerationConfig, GenerationResult, FinishReason};
use bitnet_inference::bitnet_config::TokenizerConfig;
use bitnet_core::Device;
use std::path::Path;
use std::fs;
use std::io::{self, Write, BufRead, BufReader};
use serde_json;
use tokio::time::Instant;
use std::sync::Arc;

/// Configuration for inference operations
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub max_tokens: u32,
}

impl From<InferenceConfig> for GenerationConfig {
    fn from(config: InferenceConfig) -> Self {
        GenerationConfig {
            temperature: config.temperature,
            top_k: Some(config.top_k as usize),
            top_p: Some(config.top_p),
            max_length: config.max_tokens as usize,
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
        }
    }
}

/// Handle interactive chat mode
pub async fn handle_chat(
    model: &str,
    config: InferenceConfig,
    verbose: bool,
) -> Result<()> {
    println!("🤖 BitNet Chat Mode");
    println!("Model: {}", model);
    println!("Type 'exit' or 'quit' to end the conversation");
    println!("Type 'help' for commands");
    println!();

    // Initialize inference engine
    let engine_config = EngineConfig {
        device: Device::Cpu, // TODO: Auto-detect best device
        ..Default::default()
    };
    
    let engine = Arc::new(InferenceEngine::with_config(engine_config).await?);
    
    // Load model (supports both HuggingFace and local paths)
    let model_handle = if model.contains('/') && !model.starts_with('.') && !model.starts_with('/') {
        // HuggingFace model ID (e.g., microsoft/bitnet-b1.58-2B-4T-gguf)
        println!("📥 Loading model from HuggingFace: {}", model);
        engine.load_model_from_hub(model).await?
    } else {
        // Local path
        println!("📥 Loading local model: {}", model);
        engine.load_model(model).await?
    };

    // Create text generator using builder pattern (with placeholder tokenizer config)
    let tokenizer_config = TokenizerConfig {
        vocab_size: 128256, // Default LLaMA 3 vocab size
        tokenizer_type: "llama3".to_string(),
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: Some(128002),
    };
    let generator = TextGenerator::new(
        engine.clone(),
        model_handle,
        config.into(),
        tokenizer_config,
    );
    
    println!("✅ Model loaded successfully!\n");

    // Interactive chat loop
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "exit" | "quit" => {
                println!("👋 Goodbye!");
                break;
            }
            "help" => {
                print_chat_help();
                continue;
            }
            "clear" => {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen
                println!("🤖 BitNet Chat Mode");
                println!("Model: {}", model);
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Generate response
        print!("Bot: ");
        io::stdout().flush()?;

        let start_time = Instant::now();
        
        match generator.generate(input).await {
            Ok(result) => {
                println!("{}", result.text);
                
                if verbose {
                    let duration = start_time.elapsed();
                    let tokens_per_sec = result.token_count as f64 / duration.as_secs_f64();
                    println!("\n[Generated {} tokens in {:.2}ms, {:.1} tokens/sec]", 
                           result.token_count, 
                           result.generation_time_ms,
                           tokens_per_sec);
                }
            }
            Err(e) => {
                eprintln!("❌ Error generating response: {}", e);
            }
        }
        
        println!();
    }

    Ok(())
}

/// Handle single prompt generation
pub async fn handle_generate(
    model: &str,
    prompt: &str,
    config: InferenceConfig,
    format: &str,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("🚀 Generating text with model: {}", model);
        println!("Prompt: {}", prompt);
    }

    // Initialize inference engine
    let engine_config = EngineConfig {
        device: Device::Cpu, // TODO: Auto-detect best device
        ..Default::default()
    };
    
    let engine = Arc::new(InferenceEngine::with_config(engine_config).await?);
    
    // Load model
    let model_handle = if model.contains('/') && !model.starts_with('.') && !model.starts_with('/') {
        engine.load_model_from_hub(model).await?
    } else {
        engine.load_model(model).await?
    };

    // Create text generator (with placeholder tokenizer config)
    let tokenizer_config = TokenizerConfig {
        vocab_size: 128256,
        tokenizer_type: "llama3".to_string(),
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: Some(128002),
    };
    let generator = TextGenerator::new(
        engine.clone(),
        model_handle,
        config.into(),
        tokenizer_config,
    );
    
    let start_time = Instant::now();
    
    // Generate text
    match generator.generate(prompt).await {
        Ok(result) => {
            let duration = start_time.elapsed();
            
            match format {
                "json" => {
                    let output = serde_json::json!({
                        "input": prompt,
                        "output": result.text,
                        "tokens_generated": result.token_count,
                        "finish_reason": format!("{:?}", result.finished_reason),
                        "generation_time_ms": result.generation_time_ms,
                        "tokens_per_second": result.token_count as f64 / (result.generation_time_ms as f64 / 1000.0)
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                "text" | _ => {
                    println!("{}", result.text);
                    
                    if verbose {
                        eprintln!("\n[Generated {} tokens in {:.2}ms, {:.1} tokens/sec]", 
                               result.token_count,
                               result.generation_time_ms,
                               result.token_count as f64 / (result.generation_time_ms as f64 / 1000.0));
                    }
                }
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to generate text: {}", e));
        }
    }

    Ok(())
}

/// Handle batch file processing
pub async fn handle_batch(
    model: &str,
    input_path: &Path,
    output_path: &Path,
    config: InferenceConfig,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("📁 Processing batch file with model: {}", model);
        println!("Input: {}", input_path.display());
        println!("Output: {}", output_path.display());
    }

    // Read input file
    let input_file = fs::File::open(input_path)?;
    let reader = BufReader::new(input_file);
    let prompts: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    
    if prompts.is_empty() {
        return Err(anyhow::anyhow!("Input file is empty"));
    }

    println!("📋 Processing {} prompts...", prompts.len());

    // Initialize inference engine
    let engine_config = EngineConfig {
        device: Device::Cpu, // TODO: Auto-detect best device
        ..Default::default()
    };
    
    let engine = Arc::new(InferenceEngine::with_config(engine_config).await?);
    
    // Load model
    let model_handle = if model.contains('/') && !model.starts_with('.') && !model.starts_with('/') {
        engine.load_model_from_hub(model).await?
    } else {
        engine.load_model(model).await?
    };

    // Create text generator (with placeholder tokenizer config)
    let tokenizer_config = TokenizerConfig {
        vocab_size: 128256,
        tokenizer_type: "llama3".to_string(),
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: Some(128002),
    };
    let generator = TextGenerator::new(
        engine.clone(),
        model_handle,
        config.into(),
        tokenizer_config,
    );
    
    // Process prompts and collect results
    let mut results = Vec::new();
    let total_start = Instant::now();
    
    for (i, prompt) in prompts.iter().enumerate() {
        if verbose {
            println!("Processing prompt {}/{}: {}", i + 1, prompts.len(), 
                   prompt.chars().take(50).collect::<String>());
        }
        
        let start_time = Instant::now();
        
        match generator.generate(prompt).await {
            Ok(result) => {
                let duration = start_time.elapsed();
                
                let output = serde_json::json!({
                    "prompt_index": i,
                    "input": prompt,
                    "output": result.text,
                    "tokens_generated": result.token_count,
                    "finish_reason": format!("{:?}", result.finished_reason),
                    "generation_time_ms": result.generation_time_ms,
                    "tokens_per_second": result.token_count as f64 / (result.generation_time_ms as f64 / 1000.0)
                });
                
                results.push(output);
                
                if verbose {
                    println!("  ✅ Generated {} tokens in {:.2}ms", 
                           result.token_count, result.generation_time_ms);
                }
            }
            Err(e) => {
                eprintln!("❌ Error processing prompt {}: {}", i + 1, e);
                
                let error_output = serde_json::json!({
                    "prompt_index": i,
                    "input": prompt,
                    "output": null,
                    "error": e.to_string()
                });
                
                results.push(error_output);
            }
        }
    }

    // Write results to output file
    let output_data = serde_json::json!({
        "model": model,
        "total_prompts": prompts.len(),
        "total_time_ms": total_start.elapsed().as_millis(),
        "results": results
    });

    fs::write(output_path, serde_json::to_string_pretty(&output_data)?)?;
    
    let total_duration = total_start.elapsed();
    println!("✅ Batch processing complete!");
    println!("📊 Processed {} prompts in {:.2} seconds", 
           prompts.len(), total_duration.as_secs_f64());
    println!("📄 Results written to: {}", output_path.display());

    Ok(())
}

/// Handle model download
pub async fn handle_download(
    model: &str,
    force: bool,
    verbose: bool,
) -> Result<()> {
    println!("📥 Downloading model: {}", model);
    
    // Initialize inference engine
    let engine = InferenceEngine::new().await?;
    
    // Check if model is already cached (unless force download)
    if !force {
        // TODO: Check model cache
        if verbose {
            println!("🔍 Checking if model is already cached...");
        }
    }
    
    // Download model (this will handle HuggingFace models)
    match engine.load_model_from_hub(model).await {
        Ok(_) => {
            println!("✅ Model downloaded and cached successfully!");
            if verbose {
                println!("📍 Model available for inference commands");
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to download model: {}", e));
        }
    }

    Ok(())
}

/// Handle listing available models
pub async fn handle_list(
    cached: bool,
    verbose: bool,
) -> Result<()> {
    if cached {
        println!("📦 Cached Models:");
        // TODO: List cached models from model cache
        println!("  (Model cache listing not yet implemented)");
    } else {
        println!("🌐 Available Models:");
        println!("  microsoft/bitnet-b1.58-2B-4T-gguf  - BitNet 1.58-bit 2B model (4T tokens)");
        println!("  microsoft/bitnet-b1.58-large-gguf  - BitNet 1.58-bit large model");
        println!("  microsoft/bitnet-b1.58-base-gguf   - BitNet 1.58-bit base model");
        println!();
        println!("💡 Use 'bitnet infer download --model <model_name>' to download a model");
        println!("💡 Use 'bitnet infer list --cached' to see downloaded models");
    }

    Ok(())
}

/// Print help for chat mode
fn print_chat_help() {
    println!("💬 Chat Mode Commands:");
    println!("  help  - Show this help message");
    println!("  clear - Clear the screen");
    println!("  exit  - Exit chat mode");
    println!("  quit  - Exit chat mode");
    println!();
}