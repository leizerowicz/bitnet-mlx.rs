//! Microsoft BitNet Model Metadata Analysis
//!
//! This example demonstrates loading and analyzing the real Microsoft BitNet b1.58 2B4T
//! model from HuggingFace Hub to understand its metadata structure and configuration.

use bitnet_inference::{HuggingFaceLoader, ModelRepo, Result};
use bitnet_inference::gguf::GgufLoader;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🔍 Analyzing Real Microsoft BitNet Model Metadata...");
    println!("Target: microsoft/bitnet-b1.58-2B-4T-gguf");

    // Create HuggingFace loader to download the model
    let hf_loader = HuggingFaceLoader::new()?;

    // Define the Microsoft BitNet model repository
    let repo = ModelRepo::new("microsoft", "bitnet-b1.58-2B-4T-gguf");

    println!("\n📊 Step 1: Downloading Microsoft Model...");
    println!("This may take time for first download (model is ~1.13GB)");

    // Download and get the model path
    let model_path = hf_loader.download_model(&repo).await?;
    println!("✅ Model downloaded to: {}", model_path.display());

    // Create GGUF loader for metadata analysis
    let gguf_loader = GgufLoader::new();

    println!("\n🔑 Step 2: Analyzing Raw GGUF Metadata Keys...");

    // Analyze the raw metadata keys
    match gguf_loader.analyze_metadata_keys(&model_path).await {
        Ok(key_analysis) => {
            println!("\n✅ Metadata analysis complete!");

            // Save the analysis for reference
            let analysis_content = format!(
                "Microsoft BitNet Model Metadata Analysis\n\
                 ===========================================\n\
                 Model: microsoft/bitnet-b1.58-2B-4T-gguf\n\
                 Path: {}\n\n\
                 Key Analysis:\n{:#?}",
                model_path.display(),
                key_analysis
            );

            std::fs::write("metadata_key_analysis.txt", analysis_content)?;
            println!("💾 Analysis saved to metadata_key_analysis.txt");
        }
        Err(e) => {
            println!("❌ Failed to analyze metadata keys: {}", e);
            return Err(e);
        }
    }

    // Step 3: Try to extract BitNet config to see what fails
    println!("\n⚙️ Step 3: Testing BitNet Configuration Extraction...");

    match gguf_loader.extract_model_config(&model_path).await {
        Ok(config) => {
            println!("✅ BitNet configuration extracted successfully!");
            println!("  Layers: {}", config.layer_config.n_layers);
            println!("  Hidden Size: {}", config.layer_config.hidden_size);
            println!("  Attention Heads: {}", config.attention_config.n_heads);
            println!("  Context Length: {}", config.basic_info.context_length);
            println!("  Weight Bits: {}", config.bitlinear_config.weight_bits);
        }
        Err(e) => {
            println!("❌ BitNet configuration extraction failed: {}", e);
            println!("🔍 This indicates metadata key mismatch - see analysis above");
        }
    }

    println!("\n🎯 Analysis Complete!");
    println!("Check metadata_key_analysis.txt for detailed key mapping");

    Ok(())
}