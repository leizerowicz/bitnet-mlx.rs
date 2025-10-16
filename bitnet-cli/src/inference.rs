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
            typical_p: Some(0.95),
            max_length: config.max_tokens as usize,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        }
    }
}

/// Handle interactive chat mode with conversation history
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

    // Conversation history
    let mut conversation_history: Vec<(String, String)> = Vec::new();
    let mut turn_count = 0;

    // Interactive chat loop
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "exit" | "quit" => {
                save_conversation_history(&conversation_history, verbose)?;
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
            "history" => {
                print_conversation_history(&conversation_history);
                continue;
            }
            "save" => {
                save_conversation_history(&conversation_history, true)?;
                println!("💾 Conversation history saved to bitnet_chat_history.json");
                continue;
            }
            "stats" => {
                print_conversation_stats(&conversation_history, turn_count);
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Generate response with context from conversation history
        print!("Bot: ");
        io::stdout().flush()?;

        let start_time = Instant::now();
        turn_count += 1;
        
        // Build context-aware prompt (simple approach)
        let context_prompt = if conversation_history.len() > 0 && conversation_history.len() < 5 {
            // Include recent conversation for context (limit to last 4 exchanges)
            let mut context = String::new();
            for (user_msg, bot_msg) in conversation_history.iter().rev().take(4).rev() {
                context.push_str(&format!("User: {}\nAssistant: {}\n", user_msg, bot_msg));
            }
            context.push_str(&format!("User: {}\nAssistant:", input));
            context
        } else {
            input.to_string()
        };
        
        match generator.generate(&context_prompt).await {
            Ok(result) => {
                println!("{}", result.text);
                
                // Store in conversation history
                conversation_history.push((input.to_string(), result.text.clone()));
                
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

/// Handle batch file processing with support for multiple formats
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

    // Detect input format and read prompts
    let prompts = read_input_file(input_path, verbose)?;
    
    if prompts.is_empty() {
        return Err(anyhow::anyhow!("Input file is empty or contains no valid prompts"));
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
    
    // Process prompts and collect results with progress tracking
    let mut results = Vec::new();
    let total_start = Instant::now();
    let mut successful_count = 0;
    let mut error_count = 0;
    
    for (i, prompt) in prompts.iter().enumerate() {
        // Progress indicator
        let progress = (i + 1) as f64 / prompts.len() as f64 * 100.0;
        print!("\r🔄 Progress: {:.1}% ({}/{}) ", progress, i + 1, prompts.len());
        io::stdout().flush().unwrap_or(());
        
        if verbose {
            println!("\nProcessing prompt {}/{}: {}", i + 1, prompts.len(), 
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
                successful_count += 1;
                
                if verbose {
                    println!("  ✅ Generated {} tokens in {:.2}ms", 
                           result.token_count, result.generation_time_ms);
                }
            }
            Err(e) => {
                eprintln!("\n❌ Error processing prompt {}: {}", i + 1, e);
                
                let error_output = serde_json::json!({
                    "prompt_index": i,
                    "input": prompt,
                    "output": null,
                    "error": e.to_string()
                });
                
                results.push(error_output);
                error_count += 1;
                
                // Continue processing other prompts even if one fails
                if verbose {
                    println!("  🔄 Continuing with next prompt...");
                }
            }
        }
    }
    
    // Clear progress line
    print!("\r");
    
    // Write results to output file with format detection
    write_output_file(output_path, model, &prompts, &results, total_start.elapsed(), verbose)?;
    
    let total_duration = total_start.elapsed();
    println!("✅ Batch processing complete!");
    println!("📊 Successfully processed: {}/{} prompts", successful_count, prompts.len());
    if error_count > 0 {
        println!("⚠️  Failed prompts: {}", error_count);
    }
    println!("⏱️  Total time: {:.2} seconds", total_duration.as_secs_f64());
    println!("📄 Results written to: {}", output_path.display());

    Ok(())
}

/// Read input file and parse prompts based on file extension
fn read_input_file(input_path: &Path, verbose: bool) -> Result<Vec<String>> {
    let extension = input_path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("txt");
    
    if verbose {
        println!("📄 Detected input format: {}", extension);
    }
    
    match extension.to_lowercase().as_str() {
        "json" => read_json_prompts(input_path),
        "csv" => read_csv_prompts(input_path),
        "txt" | _ => read_txt_prompts(input_path),
    }
}

/// Read prompts from JSON file
fn read_json_prompts(input_path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(input_path)?;
    let json_data: serde_json::Value = serde_json::from_str(&content)?;
    
    match json_data {
        // Array of strings
        serde_json::Value::Array(arr) => {
            let mut prompts = Vec::new();
            for item in arr {
                match item {
                    serde_json::Value::String(s) => prompts.push(s),
                    serde_json::Value::Object(obj) => {
                        // Look for common prompt fields
                        if let Some(prompt) = obj.get("prompt").and_then(|p| p.as_str()) {
                            prompts.push(prompt.to_string());
                        } else if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                            prompts.push(text.to_string());
                        } else if let Some(input) = obj.get("input").and_then(|i| i.as_str()) {
                            prompts.push(input.to_string());
                        }
                    }
                    _ => {}
                }
            }
            Ok(prompts)
        }
        // Object with prompts array
        serde_json::Value::Object(obj) => {
            if let Some(prompts_array) = obj.get("prompts").and_then(|p| p.as_array()) {
                let mut prompts = Vec::new();
                for item in prompts_array {
                    if let Some(prompt) = item.as_str() {
                        prompts.push(prompt.to_string());
                    }
                }
                Ok(prompts)
            } else {
                Err(anyhow::anyhow!("JSON format not recognized. Expected array of strings or object with 'prompts' field."))
            }
        }
        _ => Err(anyhow::anyhow!("JSON format not recognized. Expected array of strings or object with 'prompts' field."))
    }
}

/// Read prompts from CSV file
fn read_csv_prompts(input_path: &Path) -> Result<Vec<String>> {
    let mut prompts = Vec::new();
    let file = fs::File::open(input_path)?;
    let reader = BufReader::new(file);
    
    // Try to parse as CSV with headers
    let mut lines = reader.lines();
    
    // Read first line to check for headers
    if let Some(first_line) = lines.next() {
        let first_line = first_line?;
        
        // Check if first line looks like headers (contains common prompt field names)
        let has_headers = first_line.to_lowercase().contains("prompt") ||
                         first_line.to_lowercase().contains("text") ||
                         first_line.to_lowercase().contains("input");
        
        if !has_headers {
            // First line is a prompt
            prompts.push(first_line);
        }
        
        // Parse remaining lines
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            // Simple CSV parsing (assumes first column contains prompts)
            let fields: Vec<&str> = line.split(',').collect();
            if !fields.is_empty() {
                let prompt = fields[0].trim().trim_matches('"');
                if !prompt.is_empty() {
                    prompts.push(prompt.to_string());
                }
            }
        }
    }
    
    Ok(prompts)
}

/// Read prompts from text file (one per line)
fn read_txt_prompts(input_path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(input_path)?;
    let reader = BufReader::new(file);
    let mut prompts = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            prompts.push(trimmed.to_string());
        }
    }
    
    Ok(prompts)
}

/// Write output file based on extension
fn write_output_file(
    output_path: &Path,
    model: &str,
    prompts: &[String],
    results: &[serde_json::Value],
    total_duration: std::time::Duration,
    verbose: bool,
) -> Result<()> {
    let extension = output_path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("json");
    
    if verbose {
        println!("📄 Writing output format: {}", extension);
    }
    
    match extension.to_lowercase().as_str() {
        "csv" => write_csv_output(output_path, model, results, total_duration),
        "txt" => write_txt_output(output_path, results),
        "json" | _ => write_json_output(output_path, model, prompts, results, total_duration),
    }
}

/// Write results as JSON
fn write_json_output(
    output_path: &Path,
    model: &str,
    prompts: &[String],
    results: &[serde_json::Value],
    total_duration: std::time::Duration,
) -> Result<()> {
    let output_data = serde_json::json!({
        "model": model,
        "total_prompts": prompts.len(),
        "total_time_ms": total_duration.as_millis(),
        "average_time_per_prompt_ms": total_duration.as_millis() as f64 / prompts.len() as f64,
        "results": results
    });

    fs::write(output_path, serde_json::to_string_pretty(&output_data)?)?;
    Ok(())
}

/// Write results as CSV
fn write_csv_output(
    output_path: &Path,
    model: &str,
    results: &[serde_json::Value],
    total_duration: std::time::Duration,
) -> Result<()> {
    let mut csv_content = String::new();
    
    // Write headers
    csv_content.push_str("prompt_index,input,output,tokens_generated,generation_time_ms,tokens_per_second,finish_reason,error\n");
    
    // Write data rows
    for result in results {
        let prompt_index = result.get("prompt_index").and_then(|v| v.as_u64()).unwrap_or(0);
        let input = result.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        let tokens = result.get("tokens_generated").and_then(|v| v.as_u64()).unwrap_or(0);
        let time_ms = result.get("generation_time_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tokens_per_sec = result.get("tokens_per_second").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let finish_reason = result.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("");
        let error = result.get("error").and_then(|v| v.as_str()).unwrap_or("");
        
        // Escape CSV fields
        let input_escaped = escape_csv_field(input);
        let output_escaped = escape_csv_field(output);
        let error_escaped = escape_csv_field(error);
        
        csv_content.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            prompt_index, input_escaped, output_escaped, tokens, time_ms, tokens_per_sec, finish_reason, error_escaped
        ));
    }
    
    fs::write(output_path, csv_content)?;
    Ok(())
}

/// Write results as plain text
fn write_txt_output(output_path: &Path, results: &[serde_json::Value]) -> Result<()> {
    let mut txt_content = String::new();
    
    for (i, result) in results.iter().enumerate() {
        let input = result.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        let error = result.get("error").and_then(|v| v.as_str());
        
        txt_content.push_str(&format!("=== Prompt {} ===\n", i + 1));
        txt_content.push_str(&format!("Input: {}\n", input));
        
        if let Some(error_msg) = error {
            txt_content.push_str(&format!("Error: {}\n", error_msg));
        } else {
            txt_content.push_str(&format!("Output: {}\n", output));
        }
        
        txt_content.push_str("\n");
    }
    
    fs::write(output_path, txt_content)?;
    Ok(())
}

/// Escape CSV field (wrap in quotes if contains comma, quote, or newline)
fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
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
    println!("  help     - Show this help message");
    println!("  clear    - Clear the screen");
    println!("  history  - Show conversation history");
    println!("  save     - Save conversation history to file");
    println!("  stats    - Show conversation statistics");
    println!("  exit     - Exit chat mode");
    println!("  quit     - Exit chat mode");
    println!();
}

/// Print conversation history
fn print_conversation_history(history: &[(String, String)]) {
    if history.is_empty() {
        println!("📜 No conversation history yet.");
        return;
    }
    
    println!("📜 Conversation History:");
    println!("{}", "=".repeat(50));
    
    for (i, (user_msg, bot_msg)) in history.iter().enumerate() {
        println!("Turn {}:", i + 1);
        println!("You: {}", user_msg);
        println!("Bot: {}", bot_msg);
        println!("{}", "-".repeat(30));
    }
    println!();
}

/// Print conversation statistics
fn print_conversation_stats(history: &[(String, String)], turn_count: u32) {
    println!("📊 Conversation Statistics:");
    println!("  Total turns: {}", turn_count);
    println!("  Completed exchanges: {}", history.len());
    
    if !history.is_empty() {
        let total_user_chars: usize = history.iter().map(|(user, _)| user.len()).sum();
        let total_bot_chars: usize = history.iter().map(|(_, bot)| bot.len()).sum();
        let avg_user_chars = total_user_chars / history.len();
        let avg_bot_chars = total_bot_chars / history.len();
        
        println!("  Average user message length: {} characters", avg_user_chars);
        println!("  Average bot response length: {} characters", avg_bot_chars);
        println!("  Total conversation length: {} characters", total_user_chars + total_bot_chars);
    }
    println!();
}

/// Save conversation history to file
fn save_conversation_history(history: &[(String, String)], verbose: bool) -> Result<()> {
    if history.is_empty() {
        if verbose {
            println!("📜 No conversation history to save.");
        }
        return Ok(());
    }
    
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("bitnet_chat_history_{}.json", timestamp);
    
    let history_data = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "conversation_count": history.len(),
        "conversation": history.iter().enumerate().map(|(i, (user, bot))| {
            serde_json::json!({
                "turn": i + 1,
                "user": user,
                "assistant": bot
            })
        }).collect::<Vec<_>>()
    });
    
    fs::write(&filename, serde_json::to_string_pretty(&history_data)?)?;
    
    if verbose {
        println!("💾 Conversation history saved to: {}", filename);
    }
    
    Ok(())
}