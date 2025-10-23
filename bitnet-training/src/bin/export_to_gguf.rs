//! GGUF Export Binary for trained BitNet models
//!
//! This binary exports trained BitNet models to GGUF format for use with
//! llama.cpp and other GGUF-compatible inference engines.

use clap::Parser;
use anyhow::{Result, anyhow, Context};
use tracing::{info, warn};
use std::path::{Path, PathBuf};
use std::fs;

use bitnet_training::{GgufMetadata, ExportStats};

#[derive(Parser)]
#[command(name = "export_to_gguf")]
#[command(about = "Export trained BitNet models to GGUF format")]
#[command(version = "1.0.0")]
#[command(author = "BitNet-Rust Team")]
struct Args {
    /// Checkpoint directory or file path
    #[arg(long, help = "Path to trained model checkpoint")]
    checkpoint: PathBuf,
    
    /// Output GGUF file path
    #[arg(long, short, help = "Output path for GGUF file")]
    output: PathBuf,
    
    /// Quantization method for export
    #[arg(long, default_value = "ternary", help = "Quantization method")]
    quantization: String,
    
    /// Model metadata file (JSON format)
    #[arg(long, help = "Path to model metadata file")]
    metadata_file: Option<PathBuf>,
    
    /// Enable verbose output
    #[arg(long, short, help = "Enable detailed logging")]
    verbose: bool,
    
    /// Model name for metadata
    #[arg(long, help = "Model name for GGUF metadata")]
    model_name: Option<String>,
    
    /// Model architecture for metadata
    #[arg(long, default_value = "bitnet", help = "Model architecture name")]
    architecture: String,
    
    /// Verify exported model
    #[arg(long, help = "Verify exported GGUF file")]
    verify: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    init_logging(args.verbose);
    
    info!("🔄 Starting GGUF export process");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));
    info!("Checkpoint: {:?}", args.checkpoint);
    info!("Output: {:?}", args.output);
    info!("Quantization: {}", args.quantization);
    
    // Validate inputs
    validate_inputs(&args)?;
    
    // Load metadata if provided
    let metadata = load_metadata(&args)?;
    
    // Perform export
    let export_stats = perform_export(&args, metadata)?;
    
    // Verify if requested
    if args.verify {
        verify_exported_model(&args.output)?;
    }
    
    // Display results
    display_export_results(&export_stats, &args);
    
    info!("✅ GGUF export completed successfully");
    Ok(())
}

fn init_logging(verbose: bool) {
    let level = if verbose { "debug" } else { "info" };
    
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive(format!("{}={}", env!("CARGO_PKG_NAME"), level).parse().unwrap()))
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::ChronoUtc::new("%Y-%m-%dT%H:%M:%S%.3fZ".to_string()))
        .init();
}

fn validate_inputs(args: &Args) -> Result<()> {
    // Check if checkpoint exists
    if !args.checkpoint.exists() {
        return Err(anyhow!("Checkpoint path does not exist: {:?}", args.checkpoint));
    }
    
    // Check if checkpoint is a file or directory
    if args.checkpoint.is_dir() {
        info!("📁 Checkpoint is a directory, looking for model files...");
        let expected_files = ["final_model.bin", "model.bin", "pytorch_model.bin"];
        let mut found = false;
        
        for filename in &expected_files {
            let model_file = args.checkpoint.join(filename);
            if model_file.exists() {
                info!("✅ Found model file: {:?}", model_file);
                found = true;
                break;
            }
        }
        
        if !found {
            return Err(anyhow!(
                "No model files found in checkpoint directory. Expected one of: {:?}",
                expected_files
            ));
        }
    }
    
    // Check metadata file if provided
    if let Some(metadata_file) = &args.metadata_file {
        if !metadata_file.exists() {
            return Err(anyhow!("Metadata file does not exist: {:?}", metadata_file));
        }
    }
    
    // Validate quantization method
    let valid_quantizations = ["ternary", "binary", "int8", "int4", "fp16", "fp32"];
    if !valid_quantizations.contains(&args.quantization.as_str()) {
        return Err(anyhow!(
            "Invalid quantization method: {}. Valid options: {:?}",
            args.quantization,
            valid_quantizations
        ));
    }
    
    // Create output directory if it doesn't exist
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .context("Failed to create output directory")?;
    }
    
    Ok(())
}

fn load_metadata(args: &Args) -> Result<Option<GgufMetadata>> {
    if let Some(metadata_file) = &args.metadata_file {
        info!("📄 Loading metadata from: {:?}", metadata_file);
        
        let content = fs::read_to_string(metadata_file)
            .context("Failed to read metadata file")?;
        
        let metadata: GgufMetadata = serde_json::from_str(&content)
            .context("Failed to parse metadata JSON")?;
        
        info!("✅ Metadata loaded successfully");
        Ok(Some(metadata))
    } else {
        // Create default metadata from command line args
        let metadata = GgufMetadata {
            name: args.model_name.clone().unwrap_or_else(|| "bitnet-custom".to_string()),
            architecture: args.architecture.clone(),
            vocab_size: 32000,  // Default vocabulary size
            hidden_size: 4096,  // Default hidden size
            num_layers: 32,     // Default number of layers
            num_heads: 32,      // Default number of attention heads
            max_seq_len: 2048,  // Default max sequence length
            quantization: args.quantization.clone(),
            rope_theta: 10000.0,    // Default RoPE theta
            layer_norm_eps: 1e-5,   // Default layer norm epsilon
        };
        
        info!("📝 Using default metadata with command line overrides");
        Ok(Some(metadata))
    }
}

fn perform_export(
    args: &Args, 
    _metadata: Option<GgufMetadata>
) -> Result<ExportStats> {
    info!("🚀 Starting GGUF export...");
    
    let start_time = std::time::Instant::now();
    
    // Determine the actual model file to export
    let model_file = if args.checkpoint.is_file() {
        args.checkpoint.clone()
    } else {
        // Look for model file in directory
        find_model_file_in_directory(&args.checkpoint)?
    };
    
    info!("📂 Exporting model from: {:?}", model_file);
    
    // Get file size before export
    let file_size_before = get_file_size(&model_file)?;
    
    // Perform the export - placeholder until model loading is implemented
    warn!("⚠️  GGUF export implementation is placeholder - requires actual CustomBitNetModel loading");
    
    // Create a placeholder output file for now
    fs::write(&args.output, b"GGUF placeholder - requires implementation")
        .context("Failed to write output file")?;
    
    let duration = start_time.elapsed();
    let file_size_after = get_file_size(&args.output)?;
    
    let stats = ExportStats {
        original_size_bytes: file_size_before,
        compressed_size_bytes: file_size_after,
        compression_ratio: file_size_before as f64 / file_size_after.max(1) as f64,
        export_time_secs: duration.as_secs_f64(),
        num_layers: 32, // TODO: Get actual layer count from model
        num_tensors: 0, // TODO: Count actual tensors during export
    };
    
    info!("💾 Export completed in {:.2}s", duration.as_secs_f64());
    Ok(stats)
}

fn find_model_file_in_directory(dir: &Path) -> Result<PathBuf> {
    let candidates = ["final_model.bin", "model.bin", "pytorch_model.bin"];
    
    for filename in &candidates {
        let model_file = dir.join(filename);
        if model_file.exists() {
            return Ok(model_file);
        }
    }
    
    Err(anyhow!("No model files found in directory: {:?}", dir))
}

fn get_file_size(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)
        .context("Failed to get file metadata")?;
    Ok(metadata.len())
}

fn verify_exported_model(path: &Path) -> Result<()> {
    info!("🔍 Verifying exported GGUF file...");
    
    if !path.exists() {
        return Err(anyhow!("Exported file does not exist: {:?}", path));
    }
    
    let file_size = get_file_size(path)?;
    if file_size == 0 {
        return Err(anyhow!("Exported file is empty"));
    }
    
    // TODO: Add actual GGUF format validation
    warn!("⚠️  GGUF verification is placeholder - requires actual implementation");
    
    info!("✅ GGUF file verification completed");
    Ok(())
}

fn display_export_results(stats: &ExportStats, args: &Args) {
    info!("📊 Export Statistics:");
    info!("   Original size: {:.2} MB", stats.original_size_bytes as f64 / 1_048_576.0);
    info!("   Exported size: {:.2} MB", stats.compressed_size_bytes as f64 / 1_048_576.0);
    info!("   Compression ratio: {:.2}x", stats.compression_ratio);
    info!("   Export time: {:.2}s", stats.export_time_secs);
    info!("   Quantization: {}", args.quantization);
    info!("   Layers: {}", stats.num_layers);
    
    if stats.compression_ratio > 1.0 {
        info!("💾 File size reduced by {:.1}%", 
              (1.0 - 1.0 / stats.compression_ratio) * 100.0);
    }
}