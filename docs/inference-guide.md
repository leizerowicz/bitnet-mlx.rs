# BitNet-Rust Inference Guide

**Version**: 1.0.0  
**Last Updated**: October 14, 2025  
**Target Audience**: Developers implementing BitNet inference workflows  

## Overview

This guide provides comprehensive instructions for setting up and using BitNet-Rust for neural network inference. BitNet-Rust implements the revolutionary 1.58-bit quantization technology with optimized CPU and GPU acceleration, achieving significant performance improvements while maintaining model accuracy.

**Key Features**:
- ✅ **1.58-bit Quantization**: Ternary weights {-1, 0, +1} with 8-bit activations
- ✅ **GGUF Model Support**: Native support for `microsoft/bitnet-b1.58-2B-4T-gguf` and other GGUF models
- ✅ **CPU Optimization**: ARM64 NEON acceleration achieving 1.37x-3.20x speedup
- ✅ **Memory Efficiency**: Advanced memory management with HybridMemoryPool
- ✅ **Cross-Platform**: Support for CPU, Metal (macOS), and CUDA (Linux) backends

## Quick Start

### Installation

1. **Install Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

2. **Clone and build BitNet-Rust**:
```bash
git clone https://github.com/leizerowicz/bitnet-rust.git
cd bitnet-rust
cargo build --release
```

3. **Install the CLI** (optional):
```bash
cargo install --path bitnet-cli
```

### Your First Inference

#### Using the CLI (Easiest)

```bash
# Interactive chat with Microsoft BitNet model
bitnet chat microsoft/bitnet-b1.58-2B-4T-gguf

# Single inference
bitnet infer microsoft/bitnet-b1.58-2B-4T-gguf "Explain quantum computing"

# Batch processing
bitnet batch microsoft/bitnet-b1.58-2B-4T-gguf inputs.txt --output results.jsonl
```

#### Using the API (Programmatic)

```rust
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, GenerationConfig};
use bitnet_core::Device;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize inference engine
    let config = EngineConfig {
        device: Device::Cpu, // or Device::Metal, Device::Cuda
        ..Default::default()
    };
    let engine = InferenceEngine::with_config(config).await?;
    
    // Load model from HuggingFace
    let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-2B-4T-gguf").await?;
    
    // Create text generator
    let generator = TextGenerator::new(
        engine,
        model,
        GenerationConfig::default(),
        tokenizer_config, // See Configuration section
    );
    
    // Generate text
    let result = generator.generate("Explain artificial intelligence").await?;
    println!("Generated: {}", result.text);
    
    Ok(())
}
```

## Model Loading

### HuggingFace Models

BitNet-Rust automatically downloads and caches models from HuggingFace:

```rust
// Supported model formats:
let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-2B-4T-gguf").await?;
let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-3B-instruct-gguf").await?;
```

**Recommended Models**:
- `microsoft/bitnet-b1.58-2B-4T-gguf` - 2B parameters, 4T training tokens (recommended for most use cases)
- `microsoft/bitnet-b1.58-3B-instruct-gguf` - 3B parameters, instruction-tuned

### Local Models

Load models from local filesystem:

```rust
// Local GGUF file
let model = engine.load_model("/path/to/model.gguf").await?;

// Local directory with model files
let model = engine.load_model("/path/to/model/").await?;
```

### Model Caching

Models are automatically cached in `~/.cache/huggingface/` to avoid re-downloading:

```bash
# Check cached models
ls ~/.cache/huggingface/hub/

# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/
```

## Configuration

### Engine Configuration

```rust
use bitnet_inference::EngineConfig;
use bitnet_core::Device;

let config = EngineConfig {
    device: Device::Cpu,           // Device selection
    memory_limit_mb: Some(4096),   // Memory limit (MB)
    thread_count: None,            // Auto-detect CPU cores
    enable_profiling: false,       // Performance profiling
    ..Default::default()
};
```

### Generation Configuration

```rust
use bitnet_inference::api::GenerationConfig;

let config = GenerationConfig {
    temperature: 0.7,              // Randomness (0.0-1.0)
    top_k: Some(50),               // Top-K sampling
    top_p: Some(0.9),              // Top-P (nucleus) sampling
    max_length: 512,               // Maximum tokens to generate
    do_sample: true,               // Enable sampling
    stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
    seed: Some(42),                // Reproducible generation
};
```

### Tokenizer Configuration

```rust
use bitnet_inference::bitnet_config::TokenizerConfig;

let tokenizer_config = TokenizerConfig {
    vocab_size: 128256,            // LLaMA 3 vocabulary
    tokenizer_type: "llama3".to_string(),
    bos_token_id: Some(128000),    // Beginning of sequence
    eos_token_id: Some(128001),    // End of sequence
    pad_token_id: Some(128002),    // Padding token
};
```

## Device Selection

BitNet-Rust automatically selects the best available device, but you can specify manually:

### CPU (Default)
```rust
let config = EngineConfig {
    device: Device::Cpu,
    ..Default::default()
};
```

**CPU Features**:
- ✅ ARM64 NEON optimization (1.37x-3.20x speedup on Apple Silicon)
- ✅ x86_64 SIMD support
- ✅ Memory-efficient processing
- ✅ Universal compatibility

### Metal (macOS GPU)
```rust
let config = EngineConfig {
    device: Device::Metal,
    ..Default::default()
};
```

**Metal Features**:
- ✅ GPU acceleration on Apple Silicon
- ✅ Unified memory architecture
- ✅ Energy efficient
- ⚠️ macOS only

### CUDA (NVIDIA GPU)
```rust
let config = EngineConfig {
    device: Device::Cuda,
    ..Default::default()
};
```

**CUDA Features**:
- ✅ NVIDIA GPU acceleration
- ✅ High throughput for large batches
- ✅ Tensor operations optimization
- ⚠️ Requires CUDA installation

## Common Workflows

### Interactive Chat

```rust
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, GenerationConfig};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new().await?;
    let model = engine.load_model_from_hub("microsoft/bitnet-b1.58-2B-4T-gguf").await?;
    let generator = TextGenerator::new(engine, model, GenerationConfig::default(), tokenizer_config);
    
    println!("BitNet Chat - Type 'quit' to exit");
    
    loop {
        print!("You: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "quit" {
            break;
        }
        
        let result = generator.generate(input).await?;
        println!("BitNet: {}", result.text);
    }
    
    Ok(())
}
```

### Batch Processing

```rust
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

async fn process_batch(
    generator: &TextGenerator,
    input_file: &str,
    output_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let mut results = Vec::new();
    
    while let Some(line) = lines.next_line().await? {
        let result = generator.generate(&line).await?;
        results.push(serde_json::json!({
            "input": line,
            "output": result.text,
            "tokens": result.token_count,
            "time_ms": result.generation_time_ms,
        }));
    }
    
    // Write results
    let output = serde_json::to_string_pretty(&results)?;
    tokio::fs::write(output_file, output).await?;
    
    Ok(())
}
```

### Performance Monitoring

```rust
use bitnet_inference::api::GenerationResult;
use tokio::time::Instant;

async fn monitored_generation(
    generator: &TextGenerator,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result = generator.generate(prompt).await?;
    let total_time = start.elapsed();
    
    println!("Generation Stats:");
    println!("  Tokens: {}", result.token_count);
    println!("  Time: {:?}", total_time);
    println!("  Tokens/sec: {:.2}", result.token_count as f64 / total_time.as_secs_f64());
    println!("  Finish reason: {:?}", result.finish_reason);
    
    Ok(())
}
```

## Memory Management

BitNet-Rust includes advanced memory management for optimal performance:

### Memory Pool Configuration

```rust
use bitnet_core::memory::MemoryPoolConfig;

let memory_config = MemoryPoolConfig {
    initial_size_mb: 512,          // Initial pool size
    max_size_mb: Some(2048),       // Maximum pool size
    allocation_strategy: AllocationStrategy::Adaptive,
    enable_tracking: true,         // Memory usage tracking
};
```

### Memory Monitoring

```rust
// Check memory usage
let stats = engine.get_memory_stats();
println!("Memory usage: {:.1} MB", stats.allocated_mb);
println!("Memory efficiency: {:.1}%", stats.efficiency_percent);

// Clear cache if needed
engine.clear_cache().await?;
```

## Performance Optimization

### CPU Optimization

1. **Enable NEON on ARM64**:
```bash
export RUSTFLAGS="-C target-feature=+neon"
cargo build --release
```

2. **Use optimal thread count**:
```rust
let config = EngineConfig {
    thread_count: Some(num_cpus::get()), // Use all cores
    ..Default::default()
};
```

3. **Memory-mapped models** (for large models):
```rust
let config = EngineConfig {
    use_memory_mapping: true,
    ..Default::default()
};
```

### GPU Optimization

1. **Metal (macOS)**:
```rust
let config = EngineConfig {
    device: Device::Metal,
    memory_limit_mb: Some(8192), // Adjust for your GPU
    ..Default::default()
};
```

2. **CUDA (Linux)**:
```bash
# Ensure CUDA is available
export CUDA_VISIBLE_DEVICES=0
```

## Error Handling

### Common Error Patterns

```rust
use bitnet_inference::error::{InferenceError, ModelError, DeviceError};

async fn robust_inference() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new().await?;
    
    let model = match engine.load_model_from_hub("microsoft/bitnet-b1.58-2B-4T-gguf").await {
        Ok(model) => model,
        Err(ModelError::NetworkError(_)) => {
            // Try cached version or local fallback
            println!("Network error, trying cached model...");
            engine.load_model_cached("microsoft/bitnet-b1.58-2B-4T-gguf").await?
        }
        Err(e) => return Err(e.into()),
    };
    
    let generator = TextGenerator::new(engine, model, GenerationConfig::default(), tokenizer_config);
    
    match generator.generate("Hello").await {
        Ok(result) => println!("Generated: {}", result.text),
        Err(InferenceError::OutOfMemory) => {
            println!("Out of memory, reducing batch size...");
            // Implement fallback strategy
        }
        Err(DeviceError::DeviceNotAvailable) => {
            println!("GPU not available, falling back to CPU...");
            // Switch to CPU device
        }
        Err(e) => eprintln!("Inference error: {}", e),
    }
    
    Ok(())
}
```

## Next Steps

- **[CLI Documentation](cli-reference.md)** - Complete command reference
- **[Performance Optimization Guide](performance-optimization.md)** - Advanced tuning
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[Example Applications](../examples/)** - Complete example implementations

## Support

- **GitHub Issues**: [bitnet-rust/issues](https://github.com/leizerowicz/bitnet-rust/issues)
- **Documentation**: [docs.rs/bitnet-rust](https://docs.rs/bitnet-rust)
- **Examples**: [github.com/leizerowicz/bitnet-rust/tree/main/examples](https://github.com/leizerowicz/bitnet-rust/tree/main/examples)