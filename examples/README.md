# BitNet-Rust Example Applications

This directory contains comprehensive example applications demonstrating various use cases and integration patterns for BitNet-Rust. These examples showcase practical implementations of the documentation described in tasks 5.2.1 and 5.2.2 of the [ROAD_TO_INFERENCE.md](../ROAD_TO_INFERENCE.md).

## 📋 Overview

The examples are organized into four main categories:

1. **Chat Application** - Interactive conversational AI
2. **Batch Processing** - Efficient large-scale text processing
3. **API Integration** - Web services and microservice patterns
4. **Performance Benchmarking** - Comprehensive performance measurement tools

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure Rust is installed (1.70.0 or newer)
rustc --version

# Navigate to examples directory
cd examples

# Build all examples
cargo build --release
```

### Running Examples

Each example can be run independently:

```bash
# Chat Application
cargo run --bin chat_app -- --model microsoft/bitnet-b1.58-2B-4T-gguf

# Batch Processing
cargo run --bin batch_processor -- --input prompts.txt --output results.jsonl

# API Integration (server)
cargo run --bin api_integration server --port 8080

# Performance Benchmarking
cargo run --bin performance_benchmark -- --devices cpu,metal --output benchmark.json
```

## 📁 Example Applications

### 1. Chat Application (`chat_app.rs`)

**Purpose**: Interactive conversational AI with BitNet models

**Features**:
- ✅ Real-time conversation interface
- ✅ Customizable generation parameters
- ✅ Conversation history management
- ✅ Performance monitoring
- ✅ Save/load conversations
- ✅ Multiple command shortcuts

**Usage**:
```bash
# Basic chat
cargo run --bin chat_app

# Custom configuration
cargo run --bin chat_app -- \
  --model microsoft/bitnet-b1.58-2B-4T-gguf \
  --device metal \
  --temperature 0.8 \
  --max-tokens 256 \
  --verbose

# Load previous conversation
cargo run --bin chat_app -- --load-conversation chat_history.json
```

**Chat Commands**:
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/stats` - Show generation statistics
- `/config` - Show current configuration
- `/save` - Save conversation to file
- `/exit` - Exit the chat

**Example Session**:
```
🤖 BitNet Chat Application
Model: microsoft/bitnet-b1.58-2B-4T-gguf
Temperature: 0.7 | Top-K: 50 | Top-P: 0.9 | Max Tokens: 512

You: What is artificial intelligence?
BitNet: Artificial intelligence (AI) refers to the simulation of human intelligence in machines...

You: /stats
Chat Statistics:
  Total Messages: 1
  Total Tokens Generated: 156
  Total Time: 2.34s
  Average Speed: 66.67 tokens/sec
```

### 2. Batch Processing (`batch_processor.rs`)

**Purpose**: Efficient processing of large text datasets

**Features**:
- ✅ Parallel processing with configurable workers
- ✅ Progress tracking with visual progress bar
- ✅ Multiple output formats (JSON, JSONL, CSV, TXT)
- ✅ Resume capability for interrupted processing
- ✅ Error handling and retry logic
- ✅ Memory-efficient streaming
- ✅ Performance monitoring

**Usage**:
```bash
# Basic batch processing
cargo run --bin batch_processor -- \
  --input prompts.txt \
  --output results.jsonl

# Advanced configuration
cargo run --bin batch_processor -- \
  --input large_dataset.txt \
  --output processed_results.json \
  --workers 8 \
  --batch-size 20 \
  --format json \
  --temperature 0.5 \
  --max-tokens 256 \
  --timeout-seconds 45
```

**Input Format** (`prompts.txt`):
```
Explain quantum computing
What is machine learning?
Describe neural networks
Write a story about robots
```

**Output Format** (`results.jsonl`):
```json
{"id": 1, "input": "Explain quantum computing", "output": "Quantum computing...", "token_count": 145, "generation_time_ms": 2340, "timestamp": "2025-10-14T..."}
{"id": 2, "input": "What is machine learning?", "output": "Machine learning...", "token_count": 167, "generation_time_ms": 2567, "timestamp": "2025-10-14T..."}
```

**Performance Features**:
- **Parallel Processing**: Configure number of worker threads
- **Resume Support**: Continue from specific line number
- **Memory Management**: Chunked processing for large datasets
- **Error Recovery**: Timeout handling and error reporting
- **Progress Monitoring**: Real-time progress visualization

### 3. API Integration (`api_integration.rs`)

**Purpose**: REST API server and client integration patterns

**Features**:
- ✅ REST API server with JSON endpoints
- ✅ WebSocket streaming support
- ✅ Client library for easy integration
- ✅ Authentication and CORS support
- ✅ Rate limiting and concurrent request handling
- ✅ Health monitoring and metrics
- ✅ Production deployment patterns

**Server Usage**:
```bash
# Start API server
cargo run --bin api_integration server \
  --port 8080 \
  --host 0.0.0.0 \
  --cors \
  --max-concurrent 20

# Custom configuration
cargo run --bin api_integration server \
  --model microsoft/bitnet-b1.58-2B-4T-gguf \
  --device metal \
  --api-key your-secret-key
```

**Client Usage**:
```bash
# Run client examples
cargo run --bin api_integration client
```

**API Endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/model` | Model information |
| POST | `/generate` | Generate text |
| WS | `/stream` | Streaming generation |

**Example API Requests**:

```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence",
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**Client Library Example**:
```rust
use bitnet_examples::BitNetClient;

let client = BitNetClient::new("http://localhost:8080", None);
let request = GenerateRequest {
    prompt: "What is quantum computing?".to_string(),
    temperature: Some(0.7),
    max_tokens: Some(256),
    ..Default::default()
};

let response = client.generate(request).await?;
println!("Generated: {}", response.text);
```

### 4. Performance Benchmarking (`performance_benchmark.rs`)

**Purpose**: Comprehensive performance measurement and optimization

**Features**:
- ✅ Multi-device benchmarking (CPU, Metal, CUDA)
- ✅ Parameter sensitivity analysis
- ✅ Memory usage monitoring
- ✅ Latency and throughput measurement
- ✅ System information collection
- ✅ Performance recommendations
- ✅ Baseline comparison
- ✅ Continuous monitoring mode

**Usage**:
```bash
# Comprehensive benchmarking
cargo run --bin performance_benchmark -- \
  --devices cpu,metal \
  --benchmark-type all \
  --output benchmark_results.json

# Custom parameter testing
cargo run --bin performance_benchmark -- \
  --temperatures 0.1,0.5,0.9 \
  --token-counts 100,256,512 \
  --batch-sizes 1,4,8 \
  --warmup-iterations 10 \
  --benchmark-iterations 50

# Continuous monitoring
cargo run --bin performance_benchmark -- \
  --monitor \
  --monitor-duration 30 \
  --verbose
```

**Benchmark Report**:
```json
{
  "summary": {
    "best_configuration": {
      "device": "Metal",
      "temperature": 0.7,
      "max_tokens": 256,
      "batch_size": 4
    },
    "peak_performance": {
      "tokens_per_second": 87.5,
      "latency_ms": 145.2,
      "memory_usage_mb": 234.7
    },
    "recommendations": [
      "GPU acceleration provides significant performance benefits",
      "Batch size of 4 offers optimal throughput"
    ]
  }
}
```

## 🔧 Configuration

### Environment Variables

All examples support these environment variables:

```bash
export BITNET_DEVICE=metal          # Preferred device
export BITNET_MODEL_CACHE=/path/to/cache  # Model cache directory
export RUST_LOG=bitnet=info          # Logging level
export BITNET_API_KEY=your-key       # API authentication
```

### Configuration Files

Examples support TOML configuration files:

```toml
# ~/.config/bitnet/examples.toml
[default]
model = "microsoft/bitnet-b1.58-2B-4T-gguf"
device = "auto"
temperature = 0.7
max_tokens = 512

[chat]
save_conversations = true
conversation_dir = "~/.local/share/bitnet/conversations"

[batch]
workers = 4
batch_size = 10
timeout_seconds = 30

[api]
port = 8080
cors = true
max_concurrent = 10
```

## 📊 Performance Guidelines

### Hardware Recommendations

| Use Case | CPU | Memory | Storage | Device |
|----------|-----|--------|---------|--------|
| Chat Application | 4+ cores | 8+ GB | 10+ GB | CPU/Metal |
| Batch Processing | 8+ cores | 16+ GB | 50+ GB | CPU |
| API Server | 4+ cores | 8+ GB | 20+ GB | CPU/Metal |
| Benchmarking | 8+ cores | 16+ GB | 20+ GB | All available |

### Optimization Tips

1. **Memory Management**:
   ```rust
   let config = EngineConfig {
       memory_limit_mb: Some(4096),
       use_memory_mapping: true,
       enable_memory_pool: true,
       ..Default::default()
   };
   ```

2. **Device Selection**:
   ```rust
   // Automatic best device selection
   let device = Device::best_available();
   
   // Manual selection with fallback
   let device = Device::Metal.or_fallback(Device::Cpu);
   ```

3. **Batch Processing**:
   ```rust
   // Optimal batch sizes
   let batch_size = match available_memory_gb {
       8..=16 => 4,
       16..=32 => 8,
       32.. => 16,
       _ => 1,
   };
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Check model availability
   bitnet infer list --cached
   
   # Force re-download
   bitnet infer download --model microsoft/bitnet-b1.58-2B-4T-gguf --force
   ```

2. **Memory Issues**:
   ```bash
   # Reduce memory usage
   cargo run --bin chat_app -- --device cpu --max-tokens 256
   
   # Enable memory optimization
   export BITNET_MEMORY_LIMIT=2048
   ```

3. **Performance Issues**:
   ```bash
   # Check system requirements
   bitnet validate --comprehensive
   
   # Run performance benchmark
   cargo run --bin performance_benchmark -- --devices cpu --verbose
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export RUST_LOG=bitnet=debug,bitnet_inference=debug
export BITNET_PROFILE=true
cargo run --bin chat_app -- --verbose
```

## 📚 Documentation Links

- **[Inference Guide](../docs/inference-guide.md)** - Complete setup and usage guide
- **[CLI Reference](../docs/cli-reference.md)** - Command-line interface documentation
- **[Performance Optimization](../docs/performance-optimization.md)** - Performance tuning guide
- **[Troubleshooting](../docs/troubleshooting.md)** - Common issues and solutions

## 🤝 Contributing

To add new examples:

1. Create new `.rs` file in `applications/`
2. Add binary entry to `Cargo.toml`
3. Follow existing patterns for argument parsing and error handling
4. Add documentation and usage examples
5. Update this README

## 📄 License

All examples are provided under the same license as BitNet-Rust (MIT OR Apache-2.0).

## 🙋 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/leizerowicz/bitnet-rust/issues)
- **Documentation**: [Complete API documentation](https://docs.rs/bitnet-rust)
- **Examples**: Use these examples as templates for your own applications