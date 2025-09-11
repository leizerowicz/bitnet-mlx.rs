# BitNet-Rust Model Loading Guide

This guide demonstrates how to use the new HuggingFace model loading and caching capabilities in BitNet-Rust.

## Overview

BitNet-Rust now supports loading models directly from HuggingFace Hub with:
- ✅ **Direct Model Loading**: Load models by repository ID
- ✅ **SafeTensors Support**: Native SafeTensors format loading
- ✅ **Local Caching**: Efficient model caching system
- ✅ **Offline Mode**: Use cached models without internet
- ✅ **Authentication**: Support for private repositories

## Quick Start

### Basic Model Loading

```rust
use bitnet_inference::{InferenceEngine, ModelRepo};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create inference engine
    let engine = InferenceEngine::new().await?;
    
    // Load a model from HuggingFace Hub
    let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-large").await?;
    
    println!("Loaded model: {}", model.name);
    Ok(())
}
```

### Loading with Specific Revision

```rust
// Load specific version/branch
let model = engine.load_model_from_hub_with_revision(
    "microsoft/bitnet-b1.58-large", 
    "v1.0"
).await?;
```

### Using ModelRepo for Advanced Configuration

```rust
use bitnet_inference::ModelRepo;

let repo = ModelRepo::new("microsoft", "bitnet-b1.58-large")
    .with_revision("main");

let model = engine.load_model_from_repo(&repo).await?;
```

## Configuration

### Custom Cache Directory

```rust
use bitnet_inference::{HuggingFaceLoader, HuggingFaceConfig};

let config = HuggingFaceConfig {
    cache_dir: PathBuf::from("/custom/cache/path"),
    max_cache_size: 10 * 1024 * 1024 * 1024, // 10GB
    offline: false,
    ..Default::default()
};

let loader = HuggingFaceLoader::with_config(config)?;
```

### Authentication for Private Models

Set the `HF_TOKEN` environment variable or configure it programmatically:

```rust
let config = HuggingFaceConfig {
    auth_token: Some("your_hf_token_here".to_string()),
    ..Default::default()
};
```

## Cache Management

### Check Cache Statistics

```rust
let stats = engine.hf_cache_stats().await?;
println!("Cache size: {} bytes", stats.total_size);
println!("Cached models: {}", stats.model_count);
```

### Clear Cache

```rust
// Clear HuggingFace cache
engine.clear_hf_cache().await?;

// Clear inference engine cache
engine.clear_cache();
```

### Pre-download Models

```rust
// Download without loading into memory
let path = engine.download_model("microsoft/bitnet-b1.58-base").await?;
println!("Model downloaded to: {}", path.display());
```

## Supported Model Formats

### SafeTensors (Recommended)
- Native support for SafeTensors format
- Efficient memory usage
- Safe tensor loading

### Configuration Files
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration (optional)
- `tokenizer_config.json` - Tokenizer metadata (optional)

## Error Handling

```rust
match engine.load_model_from_hub("microsoft/bitnet-b1.58-large").await {
    Ok(model) => {
        println!("Model loaded successfully: {}", model.name);
    }
    Err(e) => {
        eprintln!("Failed to load model: {}", e);
        // Handle specific error cases
        match e {
            InferenceError::ModelLoad(_) => {
                eprintln!("Model loading error - check repository name and network");
            }
            InferenceError::Memory(_) => {
                eprintln!("Memory error - consider clearing cache or increasing limits");
            }
            _ => {
                eprintln!("Other error: {}", e);
            }
        }
    }
}
```

## Performance Tips

1. **Use Caching**: Models are automatically cached for subsequent loads
2. **Pre-download**: Use `download_model()` for batch operations
3. **Offline Mode**: Enable offline mode for production deployments
4. **Memory Management**: Monitor cache usage and set appropriate limits

## Advanced Usage

### Offline Deployment

```rust
// Development: Download models
let engine = InferenceEngine::new().await?;
engine.download_model("microsoft/bitnet-b1.58-large").await?;

// Production: Use offline mode
let config = HuggingFaceConfig {
    offline: true,
    ..Default::default()
};
let loader = HuggingFaceLoader::with_config(config)?;
```

### Batch Model Loading

```rust
let models = vec![
    "microsoft/bitnet-b1.58-large",
    "microsoft/bitnet-b1.58-base",
    "huggingface/CodeBERTa-small-v1",
];

for model_id in models {
    match engine.load_model_from_hub(model_id).await {
        Ok(model) => println!("✅ Loaded: {}", model.name),
        Err(e) => println!("❌ Failed {}: {}", model_id, e),
    }
}
```

## Integration with Inference

```rust
use bitnet_core::{Tensor, Device, DType};

// Load model
let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-large").await?;

// Create input tensor
let input = Tensor::zeros(&[1, 512], DType::F32, &Device::Cpu)?;

// Run inference
let output = engine.infer(&model, &input).await?;
println!("Inference output shape: {:?}", output.shape());
```

## Troubleshooting

### Common Issues

1. **Network Errors**: Check internet connection and repository accessibility
2. **Authentication**: Verify HF_TOKEN for private repositories
3. **Cache Issues**: Clear cache if models appear corrupted
4. **Memory Limits**: Increase cache limits or clear unused models

### Debug Mode

Enable tracing for detailed logging:

```rust
tracing_subscriber::fmt::init();
```

## Supported Models

Current implementation supports:
- BitNet-style models with 1.58-bit quantization
- Transformer architectures
- SafeTensors format

### Planned Support
- GGUF format models
- ONNX model conversion
- Custom model formats

## API Reference

See the [API Documentation](https://docs.rs/bitnet-inference) for complete reference.

## Examples

Run the included examples:

```bash
cd bitnet-inference
cargo run --example huggingface_loading_demo
```
