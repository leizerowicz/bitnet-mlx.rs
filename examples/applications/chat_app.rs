use anyhow::Result;
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, TextGeneratorBuilder, GenerationConfig};
use bitnet_inference::bitnet_config::TokenizerConfig;
use bitnet_core::Device;
use std::io::{self, Write};
use std::env;
use tokio::time::Instant;
use colored::*;
use clap::Parser;

/// BitNet Chat Application
/// 
/// A complete CLI chat application demonstrating real-time inference with BitNet models.
/// Features conversation history, customizable parameters, and performance monitoring.
#[derive(Parser, Debug)]
#[command(name = "bitnet-chat")]
#[command(version = "1.0.0")]
#[command(about = "Interactive chat application powered by BitNet neural networks")]
pub struct ChatArgs {
    /// Model name or path (HuggingFace ID or local path)
    #[arg(short, long, default_value = "microsoft/bitnet-b1.58-2B-4T-gguf")]
    pub model: String,

    /// Device to use for inference (auto, cpu, metal, cuda)
    #[arg(short, long, default_value = "auto")]
    pub device: String,

    /// Temperature for text generation (0.0-2.0)
    #[arg(short, long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-k sampling parameter
    #[arg(long, default_value = "50")]
    pub top_k: u32,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Maximum tokens to generate per response
    #[arg(long, default_value = "512")]
    pub max_tokens: u32,

    /// Enable verbose output with timing and token count
    #[arg(short, long)]
    pub verbose: bool,

    /// Save conversation to file
    #[arg(long)]
    pub save_conversation: Option<String>,

    /// Load conversation from file
    #[arg(long)]
    pub load_conversation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub role: String,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub token_count: Option<usize>,
    pub generation_time_ms: Option<u64>,
}

pub struct ChatApplication {
    generator: TextGenerator,
    conversation: Vec<ConversationEntry>,
    config: ChatArgs,
    stats: ChatStats,
}

#[derive(Debug, Default)]
pub struct ChatStats {
    pub total_messages: usize,
    pub total_tokens_generated: usize,
    pub total_time_ms: u64,
    pub average_tokens_per_second: f64,
}

impl ChatApplication {
    /// Initialize the chat application with the given configuration
    pub async fn new(config: ChatArgs) -> Result<Self> {
        println!("{}", "🤖 BitNet Chat Application".bright_blue().bold());
        println!("{}", "Initializing inference engine...".dim());

        // Parse device selection
        let device = match config.device.to_lowercase().as_str() {
            "auto" => Device::best_available(),
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            _ => {
                println!("{} Unknown device '{}', using auto-detection", "Warning:".yellow(), config.device);
                Device::best_available()
            }
        };

        println!("Selected device: {}", format!("{:?}", device).green());

        // Initialize inference engine
        let engine_config = EngineConfig {
            device,
            memory_limit_mb: Some(4096),
            enable_profiling: config.verbose,
            ..Default::default()
        };

        let engine = InferenceEngine::with_config(engine_config).await?;

        // Load model
        println!("Loading model: {}", config.model.cyan());
        let start_time = Instant::now();
        
        let model_handle = if config.model.contains('/') && !config.model.starts_with('.') && !config.model.starts_with('/') {
            // HuggingFace model ID
            engine.load_model_from_hub(&config.model).await?
        } else {
            // Local path
            engine.load_model(&config.model).await?
        };

        let load_time = start_time.elapsed();
        println!("Model loaded in {:.2}s", load_time.as_secs_f64());

        // Configure text generation
        let generation_config = GenerationConfig {
            temperature: config.temperature,
            top_k: Some(config.top_k as usize),
            top_p: Some(config.top_p),
            typical_p: Some(0.95),
            max_length: config.max_tokens as usize,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string(), "<|im_end|>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        };

        // Default tokenizer config (should be loaded from model metadata in production)
        let tokenizer_config = TokenizerConfig {
            vocab_size: 128256,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
            pad_token_id: Some(128002),
        };

        let generator = TextGenerator::new(
            engine,
            model_handle,
            generation_config,
            tokenizer_config,
        );

        let mut conversation = Vec::new();

        // Load conversation history if specified
        if let Some(conversation_file) = &config.load_conversation {
            match Self::load_conversation_from_file(conversation_file) {
                Ok(loaded_conversation) => {
                    conversation = loaded_conversation;
                    println!("Loaded {} messages from {}", 
                        conversation.len(), 
                        conversation_file.green()
                    );
                }
                Err(e) => {
                    println!("{} Failed to load conversation: {}", "Warning:".yellow(), e);
                }
            }
        }

        Ok(Self {
            generator,
            conversation,
            config,
            stats: ChatStats::default(),
        })
    }

    /// Run the interactive chat loop
    pub async fn run(&mut self) -> Result<()> {
        self.print_welcome();
        self.print_help();

        loop {
            // Print prompt
            print!("{} ", "You:".bright_green().bold());
            io::stdout().flush()?;

            // Read user input
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            // Handle commands
            match input {
                "exit" | "quit" | "/quit" | "/exit" => {
                    println!("{}", "👋 Goodbye!".bright_blue());
                    self.print_final_stats();
                    break;
                }
                "help" | "/help" => {
                    self.print_help();
                    continue;
                }
                "clear" | "/clear" => {
                    self.conversation.clear();
                    println!("{}", "🗑️  Conversation cleared".yellow());
                    continue;
                }
                "stats" | "/stats" => {
                    self.print_stats();
                    continue;
                }
                "save" | "/save" => {
                    self.save_conversation_interactive().await?;
                    continue;
                }
                "config" | "/config" => {
                    self.print_config();
                    continue;
                }
                "" => continue,
                _ => {
                    // Process user message
                    if let Err(e) = self.process_message(input).await {
                        println!("{} {}", "Error:".red().bold(), e);
                    }
                }
            }
        }

        // Save conversation if specified
        if let Some(conversation_file) = &self.config.save_conversation {
            self.save_conversation_to_file(conversation_file)?;
            println!("Conversation saved to {}", conversation_file.green());
        }

        Ok(())
    }

    /// Process a user message and generate a response
    async fn process_message(&mut self, user_input: &str) -> Result<()> {
        // Add user message to conversation
        let user_entry = ConversationEntry {
            role: "user".to_string(),
            content: user_input.to_string(),
            timestamp: chrono::Utc::now(),
            token_count: None,
            generation_time_ms: None,
        };
        self.conversation.push(user_entry);

        // Build conversation context
        let context = self.build_conversation_context();

        // Generate response
        print!("{} ", "BitNet:".bright_cyan().bold());
        io::stdout().flush()?;

        let start_time = Instant::now();
        let result = self.generator.generate(&context).await?;
        let generation_time = start_time.elapsed();

        // Print response
        println!("{}", result.text);

        // Add assistant response to conversation
        let assistant_entry = ConversationEntry {
            role: "assistant".to_string(),
            content: result.text.clone(),
            timestamp: chrono::Utc::now(),
            token_count: Some(result.token_count),
            generation_time_ms: Some(generation_time.as_millis() as u64),
        };
        self.conversation.push(assistant_entry);

        // Update statistics
        self.update_stats(&result, generation_time);

        // Print verbose information
        if self.config.verbose {
            self.print_generation_info(&result, generation_time);
        }

        println!(); // Add spacing

        Ok(())
    }

    /// Build conversation context for inference
    fn build_conversation_context(&self) -> String {
        let mut context = String::new();
        
        // Take last 10 messages to keep context manageable
        let recent_messages: Vec<_> = self.conversation
            .iter()
            .rev()
            .take(20) // 10 pairs of user/assistant
            .collect();

        for entry in recent_messages.iter().rev() {
            match entry.role.as_str() {
                "user" => context.push_str(&format!("Human: {}\n", entry.content)),
                "assistant" => context.push_str(&format!("Assistant: {}\n", entry.content)),
                _ => {}
            }
        }

        // Add current conversation starter if empty
        if context.is_empty() {
            context.push_str("Human: ");
        }

        context
    }

    /// Update chat statistics
    fn update_stats(&mut self, result: &bitnet_inference::api::GenerationResult, generation_time: tokio::time::Duration) {
        self.stats.total_messages += 1;
        self.stats.total_tokens_generated += result.token_count;
        self.stats.total_time_ms += generation_time.as_millis() as u64;
        
        if self.stats.total_time_ms > 0 {
            self.stats.average_tokens_per_second = 
                (self.stats.total_tokens_generated as f64) / (self.stats.total_time_ms as f64 / 1000.0);
        }
    }

    /// Print welcome message
    fn print_welcome(&self) {
        println!();
        println!("{}", "🚀 Welcome to BitNet Chat!".bright_blue().bold());
        println!("Model: {}", self.config.model.cyan());
        println!("Temperature: {:.1} | Top-K: {} | Top-P: {:.1} | Max Tokens: {}", 
            self.config.temperature, self.config.top_k, self.config.top_p, self.config.max_tokens);
        println!();
    }

    /// Print help information
    fn print_help(&self) {
        println!("{}", "Available commands:".bright_yellow().bold());
        println!("  {} - Show this help message", "/help".cyan());
        println!("  {} - Clear conversation history", "/clear".cyan());
        println!("  {} - Show generation statistics", "/stats".cyan());
        println!("  {} - Show current configuration", "/config".cyan());
        println!("  {} - Save conversation to file", "/save".cyan());
        println!("  {} - Exit the chat", "/exit".cyan());
        println!();
    }

    /// Print current configuration
    fn print_config(&self) {
        println!("{}", "Current Configuration:".bright_yellow().bold());
        println!("  Model: {}", self.config.model);
        println!("  Device: {}", self.config.device);
        println!("  Temperature: {:.2}", self.config.temperature);
        println!("  Top-K: {}", self.config.top_k);
        println!("  Top-P: {:.2}", self.config.top_p);
        println!("  Max Tokens: {}", self.config.max_tokens);
        println!("  Verbose: {}", self.config.verbose);
        println!();
    }

    /// Print generation statistics
    fn print_stats(&self) {
        println!("{}", "Chat Statistics:".bright_yellow().bold());
        println!("  Total Messages: {}", self.stats.total_messages);
        println!("  Total Tokens Generated: {}", self.stats.total_tokens_generated);
        println!("  Total Time: {:.2}s", self.stats.total_time_ms as f64 / 1000.0);
        println!("  Average Speed: {:.2} tokens/sec", self.stats.average_tokens_per_second);
        println!("  Conversation Length: {} entries", self.conversation.len());
        println!();
    }

    /// Print final statistics
    fn print_final_stats(&self) {
        if self.stats.total_messages > 0 {
            println!();
            println!("{}", "Session Summary:".bright_blue().bold());
            self.print_stats();
        }
    }

    /// Print generation information for verbose mode
    fn print_generation_info(&self, result: &bitnet_inference::api::GenerationResult, generation_time: tokio::time::Duration) {
        let tokens_per_sec = result.token_count as f64 / generation_time.as_secs_f64();
        println!("{}", format!(
            "  [Tokens: {} | Time: {:.2}s | Speed: {:.2} tok/s | Reason: {:?}]",
            result.token_count,
            generation_time.as_secs_f64(),
            tokens_per_sec,
            result.finish_reason
        ).dim());
    }

    /// Save conversation to file
    fn save_conversation_to_file(&self, filename: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.conversation)?;
        std::fs::write(filename, json)?;
        Ok(())
    }

    /// Load conversation from file
    fn load_conversation_from_file(filename: &str) -> Result<Vec<ConversationEntry>> {
        let content = std::fs::read_to_string(filename)?;
        let conversation = serde_json::from_str(&content)?;
        Ok(conversation)
    }

    /// Interactive conversation saving
    async fn save_conversation_interactive(&self) -> Result<()> {
        print!("Enter filename to save conversation: ");
        io::stdout().flush()?;
        
        let mut filename = String::new();
        io::stdin().read_line(&mut filename)?;
        let filename = filename.trim();
        
        if !filename.is_empty() {
            self.save_conversation_to_file(filename)?;
            println!("Conversation saved to {}", filename.green());
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let config = ChatArgs::parse();

    // Initialize and run chat application
    let mut app = ChatApplication::new(config).await?;
    app.run().await?;

    Ok(())
}

// Required dependencies in Cargo.toml:
/*
[dependencies]
anyhow = "1.0"
bitnet-inference = { path = "../../bitnet-inference" }
bitnet-core = { path = "../../bitnet-core" }
tokio = { version = "1.0", features = ["full"] }
colored = "2.0"
clap = { version = "4.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
*/