//! Synthetic Data Generation Binary for BitNet Training
//!
//! Generates synthetic training data for testing BitNet model training
//! and validation workflows.

use clap::Parser;
use anyhow::{Result, Context};
use tracing::{info, warn};
use std::path::PathBuf;
use std::fs;
use serde_json::json;
use rand::Rng;

#[derive(Parser)]
#[command(name = "generate_synthetic_data")]
#[command(about = "Generate synthetic training data for BitNet models")]
#[command(version = "1.0.0")]
struct Args {
    /// Output file path for generated dataset
    #[arg(long, short, help = "Output JSON file for generated dataset")]
    output: PathBuf,
    
    /// Number of examples to generate
    #[arg(long, default_value = "1000", help = "Number of training examples")]
    num_examples: usize,
    
    /// Sequence length for generated text
    #[arg(long, default_value = "128", help = "Sequence length for examples")]
    seq_length: usize,
    
    /// Vocabulary size for synthetic tokens
    #[arg(long, default_value = "32000", help = "Vocabulary size")]
    vocab_size: usize,
    
    /// Dataset type (text, qa, classification)
    #[arg(long, default_value = "text", help = "Type of dataset to generate")]
    dataset_type: String,
    
    /// Random seed for reproducibility
    #[arg(long, help = "Random seed for reproducible generation")]
    seed: Option<u64>,
    
    /// Enable verbose logging
    #[arg(long, short, help = "Enable detailed logging")]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .init();
    }
    
    info!("🎲 Generating synthetic training data");
    info!("Output: {:?}", args.output);
    info!("Examples: {}", args.num_examples);
    info!("Type: {}", args.dataset_type);
    
    // Set random seed if provided
    if let Some(seed) = args.seed {
        info!("🌱 Using random seed: {}", seed);
        // TODO: Set global random seed
        warn!("⚠️  Random seed setting requires implementation");
    }
    
    // Generate data based on type
    let dataset = match args.dataset_type.as_str() {
        "text" => generate_text_data(&args)?,
        "qa" => generate_qa_data(&args)?,
        "classification" => generate_classification_data(&args)?,
        _ => return Err(anyhow::anyhow!("Unknown dataset type: {}", args.dataset_type)),
    };
    
    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .context("Failed to create output directory")?;
    }
    
    // Write dataset to file
    let dataset_json = serde_json::to_string_pretty(&dataset)
        .context("Failed to serialize dataset")?;
    
    fs::write(&args.output, dataset_json)
        .context("Failed to write dataset file")?;
    
    info!("✅ Generated {} examples", dataset.len());
    info!("💾 Dataset saved to: {:?}", args.output);
    
    Ok(())
}

fn generate_text_data(args: &Args) -> Result<Vec<serde_json::Value>> {
    info!("📝 Generating text completion dataset...");
    
    let mut rng = rand::thread_rng();
    let mut examples = Vec::new();
    
    for i in 0..args.num_examples {
        // Generate random token sequence
        let tokens: Vec<u32> = (0..args.seq_length)
            .map(|_| rng.gen_range(0..args.vocab_size as u32))
            .collect();
        
        // Create text representation (placeholder)
        let text = format!("synthetic_text_{}_tokens_{}", i, tokens.len());
        
        let example = json!({
            "text": text,
            "tokens": tokens,
            "category": "synthetic_text",
            "id": i
        });
        
        examples.push(example);
        
        if (i + 1) % 100 == 0 {
            info!("Generated {}/{} examples", i + 1, args.num_examples);
        }
    }
    
    Ok(examples)
}

fn generate_qa_data(args: &Args) -> Result<Vec<serde_json::Value>> {
    info!("❓ Generating question-answer dataset...");
    
    let mut rng = rand::thread_rng();
    let mut examples = Vec::new();
    
    let question_templates = [
        "What is the meaning of token {}?",
        "How does {} relate to {}?",
        "Explain the concept of {}.",
        "What happens when {} and {} interact?",
    ];
    
    let answer_templates = [
        "The token {} represents a synthetic concept in our training data.",
        "This is a generated answer for training purposes involving {}.",
        "In our synthetic dataset, {} has the following properties: {}.",
    ];
    
    for i in 0..args.num_examples {
        let token_a = rng.gen_range(0..args.vocab_size);
        let token_b = rng.gen_range(0..args.vocab_size);
        
        let question_template = question_templates[rng.gen_range(0..question_templates.len())];
        let answer_template = answer_templates[rng.gen_range(0..answer_templates.len())];
        
        let question = question_template.replace("{}", &token_a.to_string());
        let answer = answer_template.replace("{}", &token_a.to_string())
                                  .replace("{}", &token_b.to_string());
        
        let example = json!({
            "question": question,
            "answer": answer,
            "category": "synthetic_qa",
            "id": i,
            "tokens": [token_a, token_b]
        });
        
        examples.push(example);
    }
    
    Ok(examples)
}

fn generate_classification_data(args: &Args) -> Result<Vec<serde_json::Value>> {
    info!("🏷️  Generating classification dataset...");
    
    let mut rng = rand::thread_rng();
    let mut examples = Vec::new();
    
    let categories = ["category_a", "category_b", "category_c", "category_d"];
    
    for i in 0..args.num_examples {
        let category = categories[rng.gen_range(0..categories.len())];
        
        // Generate tokens biased by category
        let base_token = match category {
            "category_a" => 1000,
            "category_b" => 2000,
            "category_c" => 3000,
            _ => 4000,
        };
        
        let tokens: Vec<u32> = (0..args.seq_length)
            .map(|_| {
                let offset = rng.gen_range(0..1000);
                (base_token + offset) % (args.vocab_size as u32)
            })
            .collect();
        
        let text = format!("synthetic_classification_{}_{}", category, i);
        
        let example = json!({
            "text": text,
            "tokens": tokens,
            "label": category,
            "category": "synthetic_classification",
            "id": i
        });
        
        examples.push(example);
    }
    
    Ok(examples)
}