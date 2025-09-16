use bitnet_inference::{HuggingFaceLoader, ModelRepo, Result};use bitnet_inference::{HuggingFaceLoader, ModelRepo, Result};use bitnet_inference::{HuggingFaceLoader, ModelRepo, Result};

use bitnet_inference::gguf::GgufLoader;

use bitnet_inference::gguf::GgufLoader;use bitnet_inference::gguf::GgufLoader;

#[tokio::main]

async fn main() -> Result<()> {use std::collections::HashMap;

    // Initialize logging

    env_logger::init();#[tokio::main]

    

    println!("🔍 Analyzing Real Microsoft BitNet Model Metadata...");async fn main() -> Result<()> {#[tokio::main]

    println!("Target: microsoft/bitnet-b1.58-2B-4T-gguf");

        // Initialize loggingasync fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Create HuggingFace loader to download the model

    let hf_loader = HuggingFaceLoader::new()?;    env_logger::init();    // Initialize logging

    

    // Define the Microsoft BitNet model repository        env_logger::init();

    let repo = ModelRepo::new("microsoft", "bitnet-b1.58-2B-4T-gguf");

        println!("🔍 Analyzing Real Microsoft BitNet Model Metadata...");    

    println!("\n📊 Step 1: Downloading Microsoft Model...");

    println!("This may take time for first download (model is ~1.13GB)");    println!("Target: microsoft/bitnet-b1.58-2B-4T-gguf");    println!("🔍 Analyzing Real Microsoft BitNet Model Metadata...");

    

    // Download and get the model path        println!("Target: microsoft/bitnet-b1.58-2B-4T-gguf");

    let model_path = hf_loader.download_model(&repo, None).await?;

    println!("✅ Model downloaded to: {}", model_path.display());    // Create HuggingFace loader to download the model    

    

    // Now use our debug method to analyze the raw metadata keys    let hf_loader = HuggingFaceLoader::new()?;    // Download the real Microsoft model first

    println!("\n🔑 Step 2: Analyzing Raw GGUF Metadata Keys...");

            let model_source = ModelSource::HuggingFace {

    let gguf_loader = GgufLoader::new();

        // Define the Microsoft BitNet model repository        repo_id: "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),

    // Analyze the raw metadata keys

    match gguf_loader.analyze_metadata_keys(&model_path).await {    let repo = ModelRepo::new("microsoft", "bitnet-b1.58-2B-4T-gguf");        filename: Some("model.gguf".to_string()),

        Ok(key_analysis) => {

            println!("\n✅ Metadata analysis complete!");            revision: None,

            

            // Save the analysis for reference    println!("\n📊 Step 1: Downloading Microsoft Model...");    };

            let analysis_content = format!(

                "Microsoft BitNet Model Metadata Analysis\n\    println!("This may take time for first download (model is ~1.13GB)");    

                 ===========================================\n\

                 Model: microsoft/bitnet-b1.58-2B-4T-gguf\n\        let mut loader = GgufLoader::new();

                 Path: {}\n\n\

                 Key Analysis:\n{:#?}",    // Download and get the model path    

                model_path.display(),

                key_analysis    let model_path = hf_loader.download_model(&repo, None).await?;    // First try to load metadata normally to see what works

            );

                println!("✅ Model downloaded to: {}", model_path.display());    println!("\n📊 Step 1: Testing Normal Metadata Loading...");

            std::fs::write("metadata_key_analysis.txt", analysis_content)?;

            println!("💾 Analysis saved to metadata_key_analysis.txt");        match loader.load_metadata_only(&model_source).await {

        }

        Err(e) => {    // Now use our debug method to analyze the raw metadata keys        Ok(loaded_model) => {

            println!("❌ Failed to analyze metadata keys: {}", e);

            return Err(e);    println!("\n🔑 Step 2: Analyzing Raw GGUF Metadata Keys...");            println!("✅ Successfully loaded Microsoft model metadata!");

        }

    }                

    

    // Step 3: Try to extract BitNet config to see what fails    let gguf_loader = GgufLoader::new();            // Print basic model info

    println!("\n⚙️ Step 3: Testing BitNet Configuration Extraction...");

                    println!("\n📊 Model Information:");

    match gguf_loader.extract_model_config(&model_path).await {

        Ok(config) => {    // Analyze the raw metadata keys            println!("  Name: {}", loaded_model.metadata.name);

            println!("✅ BitNet configuration extracted successfully!");

            println!("  Layers: {}", config.layer_config.n_layers);    match gguf_loader.analyze_metadata_keys(&model_path).await {            println!("  Architecture: {}", loaded_model.metadata.architecture);

            println!("  Hidden Size: {}", config.layer_config.hidden_size);

            println!("  Attention Heads: {}", config.attention_config.n_heads);        Ok(key_analysis) => {            println!("  Parameters: {}", loaded_model.metadata.parameter_count);

            println!("  Context Length: {}", config.basic_info.context_length);

            println!("  Weight Bits: {}", config.bitlinear_config.weight_bits);            println!("\n✅ Metadata analysis complete!");            println!("  Version: {}", loaded_model.metadata.version);

        }

        Err(e) => {                        

            println!("❌ BitNet configuration extraction failed: {}", e);

            println!("🔍 This indicates metadata key mismatch - see analysis above");            // Save the analysis for reference            // Print BitNet configuration

        }

    }            let analysis_content = format!(            if let Some(config) = &loaded_model.bitnet_config {

    

    println!("\n🎯 Analysis Complete!");                "Microsoft BitNet Model Metadata Analysis\n\                println!("\n⚙️ BitNet Configuration:");

    println!("Check metadata_key_analysis.txt for detailed key mapping");

                     ===========================================\n\                println!("  Layers: {}", config.layer_config.n_layers);

    Ok(())

}                 Model: microsoft/bitnet-b1.58-2B-4T-gguf\n\                println!("  Hidden Size: {}", config.layer_config.hidden_size);

                 Path: {}\n\n\                println!("  Attention Heads: {}", config.attention_config.n_heads);

                 Key Analysis:\n{:#?}",                println!("  Context Length: {}", config.basic_info.context_length);

                model_path.display(),                println!("  Weight Bits: {}", config.bitlinear_config.weight_bits);

                key_analysis            } else {

            );                println!("\n⚠️ BitNet Configuration: Not available (likely metadata key mismatch)");

                        }

            std::fs::write("metadata_key_analysis.txt", analysis_content)?;            

            println!("💾 Analysis saved to metadata_key_analysis.txt");        }

        }        Err(e) => {

        Err(e) => {            println!("❌ Failed to load Microsoft model: {}", e);

            println!("❌ Failed to analyze metadata keys: {}", e);            println!("� This error will help identify missing metadata keys...");

            return Err(e);        }

        }    }

    }    

        // Now use our debug method to analyze the raw metadata keys

    // Step 3: Try to extract BitNet config to see what fails    println!("\n🔑 Step 2: Analyzing Raw GGUF Metadata Keys...");

    println!("\n⚙️ Step 3: Testing BitNet Configuration Extraction...");    

        // Get the local file path (assuming it's been downloaded)

    match gguf_loader.extract_model_config(&model_path).await {    let cache_dir = std::env::var("HF_HOME").unwrap_or_else(|_| 

        Ok(config) => {        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap_or_default())

            println!("✅ BitNet configuration extracted successfully!");    );

            println!("  Layers: {}", config.layer_config.n_layers);    let model_path = format!("{}/hub/models--microsoft--bitnet-b1.58-2B-4T-gguf/snapshots/*/model.gguf", cache_dir);

            println!("  Hidden Size: {}", config.layer_config.hidden_size);    

            println!("  Attention Heads: {}", config.attention_config.n_heads);    // Try to find the cached model file

            println!("  Context Length: {}", config.basic_info.context_length);    match glob::glob(&model_path) {

            println!("  Weight Bits: {}", config.bitlinear_config.weight_bits);        Ok(paths) => {

        }            for path in paths {

        Err(e) => {                if let Ok(path) = path {

            println!("❌ BitNet configuration extraction failed: {}", e);                    println!("📁 Found cached model at: {}", path.display());

            println!("🔍 This indicates metadata key mismatch - see analysis above");                    

        }                    // Analyze the raw metadata keys

    }                    match loader.analyze_metadata_keys(&path).await {

                            Ok(key_analysis) => {

    println!("\n🎯 Analysis Complete!");                            println!("\n✅ Metadata analysis complete!");

    println!("Check metadata_key_analysis.txt for detailed key mapping");                            println!("📝 See detailed key mapping above");

                                

    Ok(())                            // Save the analysis for reference

}                            std::fs::write("metadata_key_analysis.txt", 
                                          format!("Microsoft BitNet Model Metadata Analysis\n{:#?}", key_analysis))?;
                            println!("💾 Analysis saved to metadata_key_analysis.txt");
                        }
                        Err(e) => {
                            println!("❌ Failed to analyze metadata keys: {}", e);
                        }
                    }
                    break;
                }
            }
        }
        Err(_) => {
            println!("⚠️ Could not find cached model file. Trying direct download...");
            
            // Try downloading just for analysis
            // This would require implementing a temp download method
            println!("🔄 Alternative: Run the normal loading first to cache the model file");
        }
    }
    
    Ok(())
}