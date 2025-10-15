use anyhow::Result;
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, GenerationConfig, GenerationResult};
use bitnet_inference::bitnet_config::TokenizerConfig;
use bitnet_core::Device;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::time::Duration;
use rayon::prelude::*;
use colored::*;

/// BitNet Batch Processing Application
/// 
/// Efficiently process large batches of text inputs using BitNet models.
/// Supports parallel processing, progress tracking, and multiple output formats.
#[derive(Parser, Debug)]
#[command(name = "bitnet-batch")]
#[command(version = "1.0.0")]
#[command(about = "Batch text processing with BitNet neural networks")]
pub struct BatchArgs {
    /// Input file path (one prompt per line)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Model name or path
    #[arg(short, long, default_value = "microsoft/bitnet-b1.58-2B-4T-gguf")]
    pub model: String,

    /// Device to use (auto, cpu, metal, cuda)
    #[arg(short, long, default_value = "auto")]
    pub device: String,

    /// Temperature for text generation (0.0-2.0)
    #[arg(short, long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-k sampling parameter
    #[arg(long, default_value = "50")]
    pub top_k: u32,

    /// Top-p sampling parameter
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Maximum tokens to generate per prompt
    #[arg(long, default_value = "512")]
    pub max_tokens: u32,

    /// Number of parallel workers (0 = auto)
    #[arg(short, long, default_value = "0")]
    pub workers: usize,

    /// Batch size for processing
    #[arg(short, long, default_value = "10")]
    pub batch_size: usize,

    /// Output format (json, jsonl, csv, txt)
    #[arg(short, long, default_value = "jsonl")]
    pub format: String,

    /// Skip lines that contain these patterns
    #[arg(long)]
    pub skip_patterns: Vec<String>,

    /// Maximum input length (characters)
    #[arg(long, default_value = "2000")]
    pub max_input_length: usize,

    /// Resume from line number (1-based)
    #[arg(long)]
    pub resume_from: Option<usize>,

    /// Save progress every N processed items
    #[arg(long, default_value = "100")]
    pub save_frequency: usize,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Timeout per generation in seconds
    #[arg(long, default_value = "30")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItem {
    pub id: usize,
    pub input: String,
    pub output: Option<String>,
    pub error: Option<String>,
    pub token_count: Option<usize>,
    pub generation_time_ms: Option<u64>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchResult {
    pub items: Vec<BatchItem>,
    pub summary: BatchSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total_items: usize,
    pub successful_items: usize,
    pub failed_items: usize,
    pub skipped_items: usize,
    pub total_tokens: usize,
    pub total_time_ms: u64,
    pub average_tokens_per_second: f64,
    pub success_rate: f64,
}

pub struct BatchProcessor {
    generator: Arc<TextGenerator>,
    config: BatchArgs,
    progress_bar: Option<ProgressBar>,
}

impl BatchProcessor {
    /// Initialize the batch processor
    pub async fn new(config: BatchArgs) -> Result<Self> {
        println!("{}", "🚀 BitNet Batch Processor".bright_blue().bold());
        println!("Initializing inference engine...");

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

        // Configure for batch processing
        let worker_count = if config.workers == 0 {
            num_cpus::get().min(4) // Default to reasonable parallelism
        } else {
            config.workers
        };

        let engine_config = EngineConfig {
            device,
            memory_limit_mb: Some(6144), // Larger memory limit for batch processing
            thread_count: Some(worker_count),
            enable_profiling: config.verbose,
            ..Default::default()
        };

        let engine = InferenceEngine::with_config(engine_config).await?;

        // Load model
        println!("Loading model: {}", config.model.cyan());
        let start_time = Instant::now();
        
        let model_handle = if config.model.contains('/') && !config.model.starts_with('.') && !config.model.starts_with('/') {
            engine.load_model_from_hub(&config.model).await?
        } else {
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
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        };

        // Default tokenizer config
        let tokenizer_config = TokenizerConfig {
            vocab_size: 128256,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
            pad_token_id: Some(128002),
        };

        let generator = Arc::new(TextGenerator::new(
            engine,
            model_handle,
            generation_config,
            tokenizer_config,
        ));

        println!("Parallel workers: {}", worker_count);
        println!("Batch size: {}", config.batch_size);
        println!();

        Ok(Self {
            generator,
            config,
            progress_bar: None,
        })
    }

    /// Process the input file in batches
    pub async fn process(&mut self) -> Result<BatchResult> {
        let start_time = Instant::now();

        // Load input items
        let items = self.load_input_items()?;
        let total_items = items.len();

        if total_items == 0 {
            return Err(anyhow::anyhow!("No items to process"));
        }

        println!("Loaded {} items for processing", total_items);

        // Set up progress bar
        self.setup_progress_bar(total_items);

        // Process items in batches
        let processed_items = self.process_items_parallel(items).await?;

        // Generate summary
        let processing_time = start_time.elapsed();
        let summary = self.generate_summary(&processed_items, processing_time);

        // Create result
        let result = BatchResult {
            items: processed_items,
            summary,
        };

        // Save output
        self.save_output(&result)?;

        // Print final statistics
        self.print_final_stats(&result);

        Ok(result)
    }

    /// Load input items from file
    fn load_input_items(&self) -> Result<Vec<BatchItem>> {
        let file = File::open(&self.config.input)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();

        let resume_from = self.config.resume_from.unwrap_or(1);

        for (line_num, line) in reader.lines().enumerate() {
            let line_number = line_num + 1;
            
            // Skip lines before resume point
            if line_number < resume_from {
                continue;
            }

            let line = line?;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Skip lines matching patterns
            if self.should_skip_line(trimmed) {
                if self.config.verbose {
                    println!("Skipping line {}: matches skip pattern", line_number);
                }
                continue;
            }

            // Truncate if too long
            let input = if trimmed.len() > self.config.max_input_length {
                println!("{} Line {} truncated from {} to {} characters", 
                    "Warning:".yellow(), line_number, trimmed.len(), self.config.max_input_length);
                trimmed[..self.config.max_input_length].to_string()
            } else {
                trimmed.to_string()
            };

            items.push(BatchItem {
                id: line_number,
                input,
                output: None,
                error: None,
                token_count: None,
                generation_time_ms: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }

        Ok(items)
    }

    /// Check if line should be skipped based on patterns
    fn should_skip_line(&self, line: &str) -> bool {
        for pattern in &self.config.skip_patterns {
            if line.contains(pattern) {
                return true;
            }
        }
        false
    }

    /// Process items using parallel processing
    async fn process_items_parallel(&mut self, mut items: Vec<BatchItem>) -> Result<Vec<BatchItem>> {
        let processed_items = Arc::new(Mutex::new(Vec::new()));
        let generator = Arc::clone(&self.generator);
        let timeout = Duration::from_secs(self.config.timeout_seconds);

        // Process in chunks to manage memory
        for chunk in items.chunks_mut(self.config.batch_size) {
            // Process chunk in parallel
            let chunk_results: Vec<_> = chunk.par_iter_mut().map(|item| {
                let generator = Arc::clone(&generator);
                let timeout = timeout;
                
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.process_single_item(item, generator, timeout).await
                    })
                })
            }).collect();

            // Update progress and save intermediate results
            for result in chunk_results {
                if let Some(pb) = &self.progress_bar {
                    pb.inc(1);
                }
            }

            // Add processed items to result
            {
                let mut processed = processed_items.lock().unwrap();
                processed.extend_from_slice(chunk);
            }

            // Save progress periodically
            let current_count = {
                let processed = processed_items.lock().unwrap();
                processed.len()
            };

            if current_count % self.config.save_frequency == 0 {
                self.save_intermediate_progress(&processed_items.lock().unwrap())?;
            }
        }

        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message("Processing complete");
        }

        let final_items = processed_items.lock().unwrap().clone();
        Ok(final_items)
    }

    /// Process a single item with timeout
    async fn process_single_item(
        &self,
        item: &mut BatchItem,
        generator: Arc<TextGenerator>,
        timeout: Duration,
    ) -> Result<()> {
        let start_time = Instant::now();

        match tokio::time::timeout(timeout, generator.generate(&item.input)).await {
            Ok(Ok(result)) => {
                let generation_time = start_time.elapsed();
                item.output = Some(result.text);
                item.token_count = Some(result.token_count);
                item.generation_time_ms = Some(generation_time.as_millis() as u64);
                item.timestamp = chrono::Utc::now().to_rfc3339();

                if self.config.verbose {
                    println!("✅ Item {} completed: {} tokens in {:.2}s", 
                        item.id, result.token_count, generation_time.as_secs_f64());
                }
            }
            Ok(Err(e)) => {
                item.error = Some(format!("Generation error: {}", e));
                item.timestamp = chrono::Utc::now().to_rfc3339();
                
                if self.config.verbose {
                    println!("❌ Item {} failed: {}", item.id, e);
                }
            }
            Err(_) => {
                item.error = Some(format!("Timeout after {} seconds", timeout.as_secs()));
                item.timestamp = chrono::Utc::now().to_rfc3339();
                
                if self.config.verbose {
                    println!("⏰ Item {} timed out", item.id);
                }
            }
        }

        Ok(())
    }

    /// Setup progress bar
    fn setup_progress_bar(&mut self, total_items: usize) {
        let pb = ProgressBar::new(total_items as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Processing batch...");
        self.progress_bar = Some(pb);
    }

    /// Generate processing summary
    fn generate_summary(&self, items: &[BatchItem], processing_time: std::time::Duration) -> BatchSummary {
        let total_items = items.len();
        let successful_items = items.iter().filter(|item| item.output.is_some()).count();
        let failed_items = items.iter().filter(|item| item.error.is_some()).count();
        let skipped_items = total_items - successful_items - failed_items;

        let total_tokens: usize = items
            .iter()
            .filter_map(|item| item.token_count)
            .sum();

        let total_time_ms = processing_time.as_millis() as u64;
        let average_tokens_per_second = if total_time_ms > 0 {
            (total_tokens as f64) / (total_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let success_rate = if total_items > 0 {
            (successful_items as f64 / total_items as f64) * 100.0
        } else {
            0.0
        };

        BatchSummary {
            total_items,
            successful_items,
            failed_items,
            skipped_items,
            total_tokens,
            total_time_ms,
            average_tokens_per_second,
            success_rate,
        }
    }

    /// Save output in specified format
    fn save_output(&self, result: &BatchResult) -> Result<()> {
        match self.config.format.to_lowercase().as_str() {
            "json" => {
                let json = serde_json::to_string_pretty(result)?;
                std::fs::write(&self.config.output, json)?;
            }
            "jsonl" => {
                let mut file = File::create(&self.config.output)?;
                for item in &result.items {
                    let line = serde_json::to_string(item)?;
                    writeln!(file, "{}", line)?;
                }
            }
            "csv" => {
                let mut file = File::create(&self.config.output)?;
                writeln!(file, "id,input,output,error,token_count,generation_time_ms,timestamp")?;
                for item in &result.items {
                    writeln!(
                        file,
                        "{},{:?},{:?},{:?},{:?},{:?},{}",
                        item.id,
                        item.input,
                        item.output.as_deref().unwrap_or(""),
                        item.error.as_deref().unwrap_or(""),
                        item.token_count.unwrap_or(0),
                        item.generation_time_ms.unwrap_or(0),
                        item.timestamp
                    )?;
                }
            }
            "txt" => {
                let mut file = File::create(&self.config.output)?;
                for item in &result.items {
                    writeln!(file, "=== Item {} ===", item.id)?;
                    writeln!(file, "Input: {}", item.input)?;
                    if let Some(output) = &item.output {
                        writeln!(file, "Output: {}", output)?;
                    }
                    if let Some(error) = &item.error {
                        writeln!(file, "Error: {}", error)?;
                    }
                    writeln!(file)?;
                }
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported output format: {}", self.config.format));
            }
        }

        Ok(())
    }

    /// Save intermediate progress
    fn save_intermediate_progress(&self, items: &[BatchItem]) -> Result<()> {
        let progress_file = format!("{}.progress", self.config.output.display());
        let json = serde_json::to_string_pretty(items)?;
        std::fs::write(progress_file, json)?;
        Ok(())
    }

    /// Print final statistics
    fn print_final_stats(&self, result: &BatchResult) {
        println!();
        println!("{}", "📊 Processing Complete!".bright_green().bold());
        println!();
        println!("{}", "Summary:".bright_yellow().bold());
        println!("  Total Items: {}", result.summary.total_items);
        println!("  {} Successful: {}", "✅".green(), result.summary.successful_items);
        println!("  {} Failed: {}", "❌".red(), result.summary.failed_items);
        println!("  {} Skipped: {}", "⏭️".yellow(), result.summary.skipped_items);
        println!("  Success Rate: {:.1}%", result.summary.success_rate);
        println!();
        println!("{}", "Performance:".bright_yellow().bold());
        println!("  Total Tokens: {}", result.summary.total_tokens);
        println!("  Total Time: {:.2}s", result.summary.total_time_ms as f64 / 1000.0);
        println!("  Average Speed: {:.2} tokens/sec", result.summary.average_tokens_per_second);
        println!();
        println!("Output saved to: {}", self.config.output.display().to_string().green());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let config = BatchArgs::parse();

    // Validate input file exists
    if !config.input.exists() {
        return Err(anyhow::anyhow!("Input file does not exist: {}", config.input.display()));
    }

    // Create output directory if needed
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Initialize and run batch processor
    let mut processor = BatchProcessor::new(config).await?;
    processor.process().await?;

    Ok(())
}

// Example usage:
// cargo run --bin batch_processor -- --input prompts.txt --output results.jsonl --workers 4 --batch-size 20