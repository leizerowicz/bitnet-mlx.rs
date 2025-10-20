//! BitNet CLI Application
//!
//! Command-line interface for BitNet operations including model management,
//! inference, training, benchmarking, and production operations support.

// Allow dead code for work-in-progress CLI implementations
#![allow(dead_code, unused_variables, unused_imports)]

use clap::{Parser, Subcommand};
use std::process;

mod ops;
mod error;
mod config;
mod customer_tools;
mod commands;
mod inference;
mod models;

use ops::OpsCommand;

#[derive(Parser)]
#[command(name = "bitnet")]
#[command(version = "1.0.0")]
#[command(about = "BitNet CLI - High-performance neural network quantization tools")]
#[command(long_about = "
BitNet CLI provides comprehensive tools for working with BitNet neural networks,
including model conversion, inference, training, benchmarking, and production 
operations support for DevOps teams.

For commercial customers, this CLI enables rapid onboarding, production deployment
validation, performance optimization, and operational monitoring integration.
")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<std::path::PathBuf>,

    /// Output format (json, yaml, table)
    #[arg(long, global = true, default_value = "table")]
    pub output: String,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Production operations commands for DevOps teams
    #[command(subcommand)]
    Ops(OpsCommand),
    
    /// Convert models between formats (SafeTensors, ONNX, PyTorch → BitNet)
    Convert(commands::ConvertCommand),
    
    /// Interactive setup wizard for environment validation
    Setup(commands::SetupCommand),
    
    /// System health validation and performance benchmarking
    Validate(commands::ValidateCommand),
    
    /// Quick start automation with example models
    Quickstart(commands::QuickStartCommand),
    
    /// Model operations (conversion, quantization, analysis) [Coming Soon]
    #[command(subcommand)]
    Model(ModelCommand),
    
    /// Inference operations (chat, completion, batch) [Coming Soon]
    #[command(subcommand)]
    Infer(InferCommand),
    
    /// Training operations (fine-tuning, QAT) [Coming Soon]
    #[command(subcommand)]
    Train(TrainCommand),
    
    /// Benchmarking and profiling [Coming Soon]
    #[command(subcommand)]
    Benchmark(BenchmarkCommand),
    
    /// Configuration management
    #[command(subcommand)]
    Config(ConfigCommand),
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// List available models (including custom models)
    List {
        /// Show only custom models
        #[arg(long)]
        custom_only: bool,
        
        /// Show detailed model information
        #[arg(long)]
        verbose: bool,
    },
    
    /// Show detailed information about a specific model
    Info {
        /// Model name or ID
        model: String,
    },
    
    /// Verify model file integrity
    Verify {
        /// Model name or path
        model: String,
    },
    
    /// Convert model formats [Coming Soon]
    Convert {
        /// Input model path
        #[arg(short, long)]
        input: std::path::PathBuf,
        
        /// Output model path  
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum InferCommand {
    /// Interactive chat mode with real-time conversation
    Chat {
        /// Model name or path (supports HuggingFace models like microsoft/bitnet-b1.58-2B-4T-gguf)
        #[arg(short, long)]
        model: String,
        
        /// Temperature for sampling (0.0 to 2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        
        /// Top-k sampling parameter
        #[arg(long, default_value = "50")]
        top_k: u32,
        
        /// Top-p sampling parameter  
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "512")]
        max_tokens: u32,
    },
    
    /// Single prompt inference for one-shot text generation
    Generate {
        /// Model name or path
        #[arg(short, long)]
        model: String,
        
        /// Input prompt
        #[arg(short, long)]
        prompt: String,
        
        /// Temperature for sampling (0.0 to 2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        
        /// Top-k sampling parameter
        #[arg(long, default_value = "50")]
        top_k: u32,
        
        /// Top-p sampling parameter
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,
        
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
    
    /// Process text files in batch
    Batch {
        /// Model name or path
        #[arg(short, long)]
        model: String,
        
        /// Input file path (one prompt per line)
        #[arg(short, long)]
        input: std::path::PathBuf,
        
        /// Output file path
        #[arg(short, long)]
        output: std::path::PathBuf,
        
        /// Temperature for sampling (0.0 to 2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        
        /// Top-k sampling parameter
        #[arg(long, default_value = "50")]
        top_k: u32,
        
        /// Top-p sampling parameter
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        
        /// Maximum tokens to generate per prompt
        #[arg(long, default_value = "512")]
        max_tokens: u32,
    },
    
    /// Download and cache models
    Download {
        /// Model name (HuggingFace model ID)
        #[arg(short, long)]
        model: String,
        
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
    },
    
    /// List available models
    List {
        /// Show cached models only
        #[arg(long)]
        cached: bool,
    },
}

#[derive(Subcommand)]
pub enum TrainCommand {
    /// Start training [Coming Soon]
    Start {
        /// Model path
        #[arg(short, long)]
        model: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum BenchmarkCommand {
    /// Benchmark inference performance [Coming Soon]
    Inference {
        /// Model path
        #[arg(short, long)]
        model: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Show current configuration
    Show,
    
    /// Set configuration values
    Set {
        /// Configuration key (e.g., "generation.temperature", "output.format")
        key: String,
        
        /// Configuration value
        value: String,
    },
    
    /// Get configuration value
    Get {
        /// Configuration key
        key: String,
    },
    
    /// Reset configuration to defaults
    Reset {
        /// Confirm reset
        #[arg(long)]
        confirm: bool,
    },
    
    /// Export configuration to file
    Export {
        /// Output file path
        #[arg(short, long)]
        output: std::path::PathBuf,
        
        /// Output format (json, yaml, toml)
        #[arg(long, default_value = "json")]
        format: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    let result = match &cli.command {
        Commands::Ops(ops_cmd) => ops_cmd.execute(&cli).await,
        Commands::Convert(convert_cmd) => convert_cmd.execute().await.map_err(|e| e.into()),
        Commands::Setup(setup_cmd) => setup_cmd.execute().await.map_err(|e| e.into()),
        Commands::Validate(validate_cmd) => validate_cmd.execute().await.map_err(|e| e.into()),
        Commands::Quickstart(quickstart_cmd) => quickstart_cmd.execute().await.map_err(|e| e.into()),
        Commands::Model(model_cmd) => {
            use models::{list_custom_models, get_model_info, CustomModelRegistry};
            
            match model_cmd {
                ModelCommand::List { custom_only, verbose } => {
                    handle_model_list(*custom_only, *verbose).await.map_err(|e| e.into())
                }
                ModelCommand::Info { model } => {
                    handle_model_info(model).await.map_err(|e| e.into())
                }
                ModelCommand::Verify { model } => {
                    handle_model_verify(model).await.map_err(|e| e.into())
                }
                ModelCommand::Convert { input, output } => {
                    eprintln!("Model conversion is coming soon in the next release!");
                    eprintln!("Input: {}", input.display());
                    eprintln!("Output: {}", output.display());
                    Ok(())
                }
            }
        }
        Commands::Infer(infer_cmd) => {
            use inference::InferenceConfig;
            
            match infer_cmd {
                InferCommand::Chat { model, temperature, top_k, top_p, max_tokens } => {
                    let config = InferenceConfig {
                        temperature: *temperature,
                        top_k: *top_k,
                        top_p: *top_p,
                        max_tokens: *max_tokens,
                    };
                    inference::handle_chat(model, config, cli.verbose).await.map_err(|e| e.into())
                }
                InferCommand::Generate { model, prompt, temperature, top_k, top_p, max_tokens, format } => {
                    let config = InferenceConfig {
                        temperature: *temperature,
                        top_k: *top_k,
                        top_p: *top_p,
                        max_tokens: *max_tokens,
                    };
                    inference::handle_generate(model, prompt, config, format, cli.verbose).await.map_err(|e| e.into())
                }
                InferCommand::Batch { model, input, output, temperature, top_k, top_p, max_tokens } => {
                    let config = InferenceConfig {
                        temperature: *temperature,
                        top_k: *top_k,
                        top_p: *top_p,
                        max_tokens: *max_tokens,
                    };
                    inference::handle_batch(model, input, output, config, cli.verbose).await.map_err(|e| e.into())
                }
                InferCommand::Download { model, force } => {
                    inference::handle_download(model, *force, cli.verbose).await.map_err(|e| e.into())
                }
                InferCommand::List { cached } => {
                    inference::handle_list(*cached, cli.verbose).await.map_err(|e| e.into())
                }
            }
        }
        Commands::Train(_) => {
            eprintln!("Training operations are coming soon in the next release!");
            eprintln!("Current focus: Essential customer tools (Story 2.1)");
            Ok(())
        }
        Commands::Benchmark(_) => {
            eprintln!("Benchmarking operations are coming soon in the next release!");
            eprintln!("Use 'bitnet validate' for system performance validation.");
            Ok(())
        }
        Commands::Config(config_cmd) => {
            commands::handle_config_command(config_cmd, &cli).await.map_err(|e| e.into())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        
        // Display error chain if available
        let mut current_error = &e as &dyn std::error::Error;
        while let Some(source) = current_error.source() {
            eprintln!("  Caused by: {}", source);
            current_error = source;
        }
        
        process::exit(1);
    }
}

/// Handle model list command
async fn handle_model_list(custom_only: bool, verbose: bool) -> anyhow::Result<()> {
    println!("📋 Available BitNet Models");
    println!();
    
    if custom_only || !custom_only {
        // List custom models
        match models::list_custom_models() {
            Ok(custom_models) => {
                if custom_models.is_empty() {
                    println!("No custom models found.");
                    println!("To add a custom model, place a .gguf file in:");
                    println!("  - ./models/");
                    println!("  - ./custom-models/");
                    println!("  - ~/.bitnet/models/");
                } else {
                    println!("Custom Models:");
                    for (model_id, model_info) in custom_models {
                        println!("  🤖 {} ({})", model_id, model_info.parameters);
                        if verbose {
                            println!("     Name: {}", model_info.name);
                            println!("     Architecture: {}", model_info.architecture);
                            println!("     Description: {}", model_info.description);
                            if let Some(path) = &model_info.local_path {
                                println!("     Path: {}", path);
                            }
                            if let Some(config) = &model_info.config {
                                println!("     Config: {} layers, {} hidden, {} vocab", 
                                        config.num_layers, config.hidden_size, config.vocab_size);
                            }
                            println!();
                        }
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to load custom models: {}", e);
            }
        }
    }
    
    if !custom_only {
        println!();
        println!("Built-in Models:");
        println!("  🏭 microsoft/bitnet-b1.58-2B-4T-gguf (1.3B parameters)");
        println!("     Standard BitNet 1.58-bit model for inference");
        
        if verbose {
            println!("     Architecture: BitNet transformer");
            println!("     Quantization: 1.58-bit ternary weights");
            println!("     Source: HuggingFace Hub");
            println!("     Usage: bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf");
        }
    }
    
    println!();
    println!("Usage:");
    println!("  bitnet infer chat --model <model-id>");
    println!("  bitnet model info <model-id>");
    
    Ok(())
}

/// Handle model info command
async fn handle_model_info(model_id: &str) -> anyhow::Result<()> {
    println!("📄 Model Information: {}", model_id);
    println!();
    
    match models::get_model_info(model_id) {
        Ok(model_info) => {
            println!("Name: {}", model_info.name);
            println!("Architecture: {}", model_info.architecture);
            println!("Parameters: {}", model_info.parameters);
            println!("Description: {}", model_info.description);
            println!("Custom Model: {}", if model_info.is_custom { "Yes" } else { "No" });
            
            if let Some(path) = &model_info.local_path {
                println!("Local Path: {}", path);
                
                // Check if file exists
                if std::path::Path::new(path).exists() {
                    let metadata = std::fs::metadata(path)?;
                    let size_mb = metadata.len() as f64 / 1024.0 / 1024.0;
                    println!("File Size: {:.1} MB", size_mb);
                } else {
                    println!("❌ File not found at specified path");
                }
            }
            
            if let Some(repo) = &model_info.huggingface_repo {
                println!("HuggingFace Repo: {}", repo);
            }
            
            if let Some(config) = &model_info.config {
                println!();
                println!("Model Configuration:");
                println!("  Vocabulary Size: {}", config.vocab_size);
                println!("  Hidden Size: {}", config.hidden_size);
                println!("  Number of Layers: {}", config.num_layers);
                println!("  Number of Heads: {}", config.num_heads);
                println!("  Max Sequence Length: {}", config.max_seq_len);
                println!("  Quantization: {}", config.quantization);
            }
        }
        Err(e) => {
            println!("❌ Model '{}' not found: {}", model_id, e);
            println!();
            println!("Use 'bitnet model list' to see available models.");
            return Err(e);
        }
    }
    
    Ok(())
}

/// Handle model verify command
async fn handle_model_verify(model_id: &str) -> anyhow::Result<()> {
    println!("🔍 Verifying model: {}", model_id);
    
    match models::get_model_info(model_id) {
        Ok(model_info) => {
            if let Some(path) = &model_info.local_path {
                match std::fs::metadata(path) {
                    Ok(metadata) => {
                        println!("✅ Model file found: {}", path);
                        println!("   Size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
                        
                        // Try to load the model if it's custom
                        if model_info.is_custom {
                            println!("🔄 Testing model loading...");
                            match models::load_custom_model(model_id) {
                                Ok(_) => {
                                    println!("✅ Model loads successfully");
                                }
                                Err(e) => {
                                    println!("❌ Model loading failed: {}", e);
                                    return Err(e);
                                }
                            }
                        }
                        
                        println!("✅ Model verification passed");
                    }
                    Err(e) => {
                        println!("❌ Cannot access model file: {}", e);
                        return Err(e.into());
                    }
                }
            } else {
                println!("❌ Model has no local path (may be a remote model)");
                anyhow::bail!("Cannot verify remote model");
            }
        }
        Err(e) => {
            println!("❌ Model '{}' not found: {}", model_id, e);
            return Err(e);
        }
    }
    
    Ok(())
}
