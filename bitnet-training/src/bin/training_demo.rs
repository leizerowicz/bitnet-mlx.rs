use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use bitnet_training::{
    dataset::{TrainingDataset, DatasetConfig},
    tokenization::{TrainingTokenizer, TokenizerConfig},
    training::{BitNetTrainer, TrainingConfig},
    models::CustomBitNetModel,
};

/// BitNet Custom Model Training Demo
#[derive(Parser)]
#[command(name = "bitnet-training-demo")]
#[command(about = "Train custom BitNet models with Phase 2 infrastructure")]
struct Args {
    /// Path to training dataset JSON file
    #[arg(long, default_value = "data/training/synthetic_dataset.json")]
    dataset_path: PathBuf,
    
    /// Path to model configuration
    #[arg(long, default_value = "models/custom-bitnet-small/config.json")]
    model_config: PathBuf,
    
    /// Number of training epochs
    #[arg(long, default_value = "3")]
    epochs: usize,
    
    /// Batch size for training
    #[arg(long, default_value = "4")]
    batch_size: usize,
    
    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    learning_rate: f32,
    
    /// Use synthetic data instead of file
    #[arg(long)]
    use_synthetic: bool,
    
    /// Enable verbose logging
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("🚀 BitNet Custom Model Training Demo - Phase 2");
    info!("📊 Configuration:");
    info!("   - Dataset: {}", args.dataset_path.display());
    info!("   - Model config: {}", args.model_config.display());
    info!("   - Epochs: {}", args.epochs);
    info!("   - Batch size: {}", args.batch_size);
    info!("   - Learning rate: {}", args.learning_rate);
    info!("   - Use synthetic data: {}", args.use_synthetic);
    
    // Step 1: Initialize tokenizer
    info!("🔤 Initializing tokenizer...");
    let tokenizer_config = TokenizerConfig::default();
    let tokenizer = TrainingTokenizer::new(tokenizer_config)?;
    info!("✅ Tokenizer initialized with vocab size: {}", tokenizer.vocab_size());
    
    // Step 2: Load or create dataset
    info!("📚 Loading training dataset...");
    let dataset_config = DatasetConfig {
        max_seq_length: 256, // Reduced for demo
        batch_size: args.batch_size,
        shuffle: true,
        pad_to_max_length: true,
    };
    
    let dataset = if args.use_synthetic {
        info!("🧪 Creating synthetic dataset for testing...");
        TrainingDataset::create_synthetic(dataset_config, tokenizer)?
    } else {
        if args.dataset_path.exists() {
            TrainingDataset::from_json_file(&args.dataset_path, dataset_config, tokenizer)?
        } else {
            warn!("📁 Dataset file not found, using synthetic data");
            TrainingDataset::create_synthetic(dataset_config, tokenizer)?
        }
    };
    
    info!("✅ Dataset loaded with {} examples", dataset.len());
    info!("   - Number of batches: {}", dataset.num_batches());
    
    // Step 3: Initialize model
    info!("🧠 Initializing Custom BitNet model...");
    let model = if args.model_config.exists() {
        // For this demo, we'll use the testing model since we don't have full weight loading yet
        CustomBitNetModel::new_for_testing()?
    } else {
        warn!("📁 Model config not found, using default testing model");
        CustomBitNetModel::new_for_testing()?
    };
    
    let model_info = model.model_info();
    info!("✅ Model initialized:");
    info!("   - Total parameters: {:.1}M", model_info.total_parameters as f64 / 1_000_000.0);
    info!("   - Quantized parameters: {:.1}M", model_info.quantized_parameters as f64 / 1_000_000.0);
    info!("   - Layers: {}", model_info.layers);
    info!("   - Hidden size: {}", model_info.hidden_size);
    
    // Step 4: Initialize trainer
    info!("⚙️ Initializing trainer...");
    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        num_epochs: args.epochs,
        save_steps: 50,  // Save more frequently for demo
        eval_steps: 25,  // Evaluate more frequently for demo
        mixed_precision: false, // Disable for CPU demo
        gradient_accumulation_steps: 2,
        max_grad_norm: 1.0,
        weight_decay: 0.01,
        warmup_steps: 10,
        gradient_checkpointing: false,
    };
    
    let mut trainer = BitNetTrainer::new(model, training_config)?;
    info!("✅ Trainer initialized");
    
    // Step 5: Run training
    info!("🔥 Starting training...");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    trainer.train(&dataset)?;
    
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("🎯 Training completed successfully!");
    
    // Step 6: Display final statistics
    let final_stats = trainer.get_stats();
    info!("📊 Final Training Statistics:");
    info!("   - Final epoch: {}", final_stats.epoch + 1);
    info!("   - Total steps: {}", final_stats.step);
    info!("   - Learning rate: {}", final_stats.learning_rate);
    
    // Step 7: Demonstrate model capabilities (create new tokenizer since original was moved)
    info!("🧪 Testing model inference...");
    let test_tokenizer = TrainingTokenizer::new(TokenizerConfig::default())?;
    let test_model = CustomBitNetModel::new_for_testing()?;
    demonstrate_model_capabilities(&test_model, &test_tokenizer)?;
    
    info!("✅ Phase 2 Training Infrastructure Demo Complete!");
    info!("📋 Next Steps:");
    info!("   - Implement MLX acceleration for Apple Silicon");
    info!("   - Add proper gradient computation and backpropagation");
    info!("   - Integrate with bitnet-quant for weight quantization");
    info!("   - Add GGUF export functionality");
    
    Ok(())
}

/// Demonstrate basic model capabilities
fn demonstrate_model_capabilities(
    model: &CustomBitNetModel,
    tokenizer: &TrainingTokenizer,
) -> Result<()> {
    use candle_core::Tensor;
    
    info!("🔍 Model Capability Demo:");
    
    // Test tokenization
    let test_text = "BitNet models use quantization for efficiency";
    let tokens = tokenizer.encode(test_text, true)?;
    info!("   - Input text: '{}'", test_text);
    info!("   - Tokenized to {} tokens: {:?}", tokens.len(), &tokens[..tokens.len().min(10)]);
    
    // Test model forward pass
    let device = model.device();
    let input_tensor = Tensor::from_vec(
        tokens.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
        (1, tokens.len()), // batch_size=1, seq_len=tokens.len()
        device,
    )?;
    
    info!("   - Input tensor shape: {:?}", input_tensor.shape());
    
    // Forward pass (this will be a simplified version for demo)
    let output = model.forward(&input_tensor, None);
    match output {
        Ok(logits) => {
            info!("   - Output logits shape: {:?}", logits.shape());
            info!("   - ✅ Forward pass successful!");
        }
        Err(e) => {
            warn!("   - ⚠️ Forward pass failed (expected in demo): {}", e);
            info!("   - This is expected since we're using placeholder weights");
        }
    }
    
    Ok(())
}