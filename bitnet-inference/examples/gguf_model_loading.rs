//! GGUF Model Loading Example
//!
//! This example demonstrates how to load Microsoft BitNet b1.58 2B4T models in GGUF format
//! using the BitNet-Rust inference engine with HuggingFace Hub integration.

use bitnet_inference::{HuggingFaceLoader, ModelRepo, Result};
use bitnet_inference::gguf::GgufLoader;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("BitNet-Rust GGUF Model Loading Example");
    println!("=======================================");

    // Example 1: Direct GGUF file loading
    if let Ok(model) = load_local_gguf_example().await {
        println!("✅ Direct GGUF loading: Success");
        print_model_info(&model);
    } else {
        println!("⏭️  Direct GGUF loading: Skipped (no local file)");
    }

    // Example 2: HuggingFace Hub integration with GGUF support
    if let Ok(model) = load_huggingface_gguf_example().await {
        println!("✅ HuggingFace GGUF loading: Success");
        print_model_info(&model);
    } else {
        println!("⏭️  HuggingFace GGUF loading: Skipped (requires network)");
    }

    // Example 3: BitNet-specific model loading
    demonstrate_bitnet_features().await?;

    println!("\n🎉 GGUF loading examples completed successfully!");
    Ok(())
}

/// Example 1: Load a GGUF model directly from a local file
async fn load_local_gguf_example() -> Result<bitnet_inference::engine::model_loader::LoadedModel> {
    println!("\n📁 Example 1: Direct GGUF File Loading");
    
    // Check for a sample GGUF file (this would be a real file in practice)
    let sample_path = "models/bitnet-b1.58-2b.gguf";
    
    if !Path::new(sample_path).exists() {
        return Err(bitnet_inference::InferenceError::model_load(
            "Sample GGUF file not found - this is expected for the example"
        ));
    }

    println!("Loading GGUF model from: {}", sample_path);
    let loader = GgufLoader::new();
    let model = loader.load_model_from_path(sample_path, None).await?;
    
    println!("✅ Successfully loaded GGUF model!");
    Ok(model)
}

/// Example 2: Load a GGUF model via HuggingFace Hub integration
async fn load_huggingface_gguf_example() -> Result<bitnet_inference::engine::model_loader::LoadedModel> {
    println!("\n🤗 Example 2: HuggingFace Hub GGUF Loading");
    
    // Create HuggingFace loader
    let loader = HuggingFaceLoader::new()?;
    
    // Define the Microsoft BitNet model repository
    let repo = ModelRepo::new("microsoft", "bitnet-b1.58-2B-4T-gguf");
    
    println!("Attempting to load model: {}", repo.repo_id());
    println!("Note: This will download the model if not cached (may take time)");
    
    // This would work if the model exists and network is available
    match loader.load_model(&repo).await {
        Ok(model) => {
            println!("✅ Successfully loaded model from HuggingFace Hub!");
            Ok(model)
        }
        Err(e) => {
            println!("⚠️  Model loading failed (expected for demo): {}", e);
            Err(e)
        }
    }
}

/// Example 3: Demonstrate BitNet-specific GGUF features
async fn demonstrate_bitnet_features() -> Result<()> {
    println!("\n🔬 Example 3: BitNet-Specific GGUF Features");
    
    println!("BitNet GGUF Support Features:");
    println!("• Ternary weight quantization (1.58-bit)");
    println!("• 8-bit activation quantization"); 
    println!("• Optimized tensor layout for BitLinear layers");
    println!("• RoPE positional embeddings support");
    println!("• LLaMA 3 tokenizer integration");
    println!("• Memory-efficient model loading (~400MB target)");
    
    // Demonstrate GGUF value type handling
    println!("\n🔧 GGUF Value Type Support:");
    let value_types = vec![
        ("UINT8", 0u32),
        ("FLOAT32", 6u32), 
        ("STRING", 8u32),
        ("BITNET_B158", 1000u32), // Custom BitNet extension
    ];
    
    for (name, type_id) in value_types {
        match bitnet_inference::GgufValueType::try_from(type_id) {
            Ok(_) => println!("  ✅ {}: Supported", name),
            Err(_) => println!("  ❌ {}: Not supported", name),
        }
    }
    
    // Demonstrate GGUF tensor type handling
    println!("\n📊 GGUF Tensor Type Support:");
    let tensor_types = vec![
        ("F32", 0u32),
        ("F16", 1u32),
        ("Q4_0", 2u32),
        ("I8", 16u32),
        ("BITNET_B158", 1000u32), // Custom BitNet extension
    ];
    
    for (name, type_id) in tensor_types {
        match bitnet_inference::GgufTensorType::try_from(type_id) {
            Ok(_) => println!("  ✅ {}: Supported", name),
            Err(_) => println!("  ❌ {}: Not supported", name),
        }
    }
    
    Ok(())
}

/// Helper function to print model information
fn print_model_info(model: &bitnet_inference::engine::model_loader::LoadedModel) {
    println!("\n📋 Model Information:");
    println!("  Name: {}", model.metadata.name);
    println!("  Architecture: {}", model.metadata.architecture);
    println!("  Parameters: {}", model.metadata.parameter_count);
    println!("  Quantization: {} bits", model.metadata.quantization_bits);
    println!("  Input shape: {:?}", model.metadata.input_shape);
    println!("  Output shape: {:?}", model.metadata.output_shape);
    println!("  Layers: {}", model.architecture.layers.len());
    println!("  Total weight size: {} MB", model.weights.total_size / (1024 * 1024));
    
    // Show layer breakdown
    let mut layer_counts = std::collections::HashMap::new();
    for layer in &model.architecture.layers {
        *layer_counts.entry(format!("{:?}", layer.layer_type)).or_insert(0) += 1;
    }
    
    println!("  Layer breakdown:");
    for (layer_type, count) in layer_counts {
        println!("    {}: {} layers", layer_type, count);
    }
}