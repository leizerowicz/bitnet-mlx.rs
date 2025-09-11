//! Example demonstrating HuggingFace model loading and caching capabilities.
//!
//! This example shows how to:
//! 1. Load models from HuggingFace Hub
//! 2. Use model caching for efficient reuse
//! 3. Perform basic inference operations

use bitnet_inference::{InferenceEngine, ModelRepo, Result};
use bitnet_core::{Tensor, Device, DType};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize basic tracing
    tracing_subscriber::fmt::init();

    println!("🤖 BitNet-Rust HuggingFace Model Loading Demo");
    println!("============================================");

    // Create inference engine with automatic device selection
    let engine = InferenceEngine::new().await?;
    println!("✅ Inference engine created for device: {:?}", engine.device());

    // Example 1: Load a model using simple repo ID string
    println!("\n📥 Loading model from HuggingFace Hub...");
    
    // Note: Using a hypothetical BitNet model for demonstration
    // In practice, you would use actual model repos like:
    // "microsoft/bitnet-b1.58-large" or "HuggingFaceTB/SmolLM-135M"
    let model_repo = "microsoft/bitnet-b1.58-large";
    
    match engine.load_model_from_hub(model_repo).await {
        Ok(model) => {
            println!("✅ Model loaded successfully:");
            println!("   - Name: {}", model.name);
            println!("   - Version: {}", model.version);
            println!("   - Parameters: {}", model.parameter_count);
            println!("   - Input dim: {}", model.input_dim);
            println!("   - Output dim: {}", model.output_dim);
        }
        Err(e) => {
            println!("⚠️  Model loading failed (expected for demo): {}", e);
            println!("   This is normal since we're using a hypothetical model repo.");
        }
    }

    // Example 2: Load with specific revision
    println!("\n📥 Loading model with specific revision...");
    match engine.load_model_from_hub_with_revision("microsoft/bitnet-b1.58-large", "v1.0").await {
        Ok(model) => {
            println!("✅ Model with revision loaded: {}", model.name);
        }
        Err(e) => {
            println!("⚠️  Model loading failed (expected for demo): {}", e);
        }
    }

    // Example 3: Model caching demonstration
    println!("\n🗄️  Cache statistics:");
    match engine.hf_cache_stats().await {
        Ok(stats) => {
            println!("   - Total cache size: {} bytes", stats.total_size);
            println!("   - Cached models: {}", stats.model_count);
            println!("   - Max cache size: {} bytes", stats.max_size);
        }
        Err(e) => {
            println!("   Could not get cache stats: {}", e);
        }
    }

    // Example 4: Pre-download a model without loading
    println!("\n📥 Pre-downloading model for later use...");
    match engine.download_model("microsoft/bitnet-b1.58-base").await {
        Ok(path) => {
            println!("✅ Model downloaded to: {}", path.display());
        }
        Err(e) => {
            println!("⚠️  Download failed (expected for demo): {}", e);
        }
    }

    // Example 5: Working with ModelRepo directly
    println!("\n🔧 Using ModelRepo for advanced configuration...");
    let custom_repo = ModelRepo::new("huggingface", "CodeBERTa-small-v1")
        .with_revision("main");
    
    match engine.load_model_from_repo(&custom_repo).await {
        Ok(model) => {
            println!("✅ Custom repo model loaded: {}", model.name);
            
            // Demonstrate basic inference (placeholder)
            println!("\n🧠 Running inference...");
            let input = Tensor::zeros(&[1, 512], DType::F32, &Device::Cpu)?;
            
            match engine.infer(&model, &input).await {
                Ok(output) => {
                    println!("✅ Inference completed! Output shape: {:?}", output.shape());
                }
                Err(e) => {
                    println!("⚠️  Inference failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("⚠️  Custom repo loading failed (expected for demo): {}", e);
        }
    }

    // Cache management
    println!("\n🧹 Cache management operations:");
    
    // Clear HuggingFace cache
    match engine.clear_hf_cache().await {
        Ok(_) => println!("✅ HuggingFace cache cleared"),
        Err(e) => println!("⚠️  Cache clear failed: {}", e),
    }

    // Clear inference engine cache
    engine.clear_cache();
    println!("✅ Inference engine cache cleared");

    println!("\n🎉 Demo completed successfully!");
    println!("\nNext steps:");
    println!("- Try with real HuggingFace model repositories");
    println!("- Experiment with different model architectures");
    println!("- Use authentication tokens for private models");
    println!("- Implement custom model conversion pipelines");

    Ok(())
}

/// Helper function to demonstrate error handling patterns
async fn try_load_with_fallback(engine: &InferenceEngine) -> Result<()> {
    let models_to_try = [
        "microsoft/bitnet-b1.58-large",
        "microsoft/bitnet-b1.58-base", 
        "huggingface/CodeBERTa-small-v1",
    ];

    for model_repo in &models_to_try {
        println!("Trying to load: {}", model_repo);
        
        match engine.load_model_from_hub(model_repo).await {
            Ok(model) => {
                println!("✅ Successfully loaded: {}", model.name);
                return Ok(());
            }
            Err(e) => {
                println!("❌ Failed to load {}: {}", model_repo, e);
                continue;
            }
        }
    }

    println!("⚠️  No models could be loaded (this is expected in demo mode)");
    Ok(())
}
