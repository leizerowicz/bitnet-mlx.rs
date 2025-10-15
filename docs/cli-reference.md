# BitNet CLI Reference

**Version**: 1.0.0  
**Last Updated**: October 14, 2025  
**Command**: `bitnet`

## Overview

The BitNet CLI provides comprehensive tools for working with BitNet neural networks, including model management, inference, training, benchmarking, and production operations support.

## Installation

```bash
# Install from source
git clone https://github.com/leizerowicz/bitnet-rust.git
cd bitnet-rust
cargo install --path bitnet-cli

# Verify installation
bitnet --version
```

## Global Options

All commands support these global options:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--verbose` | `-v` | Enable verbose output | `false` |
| `--config` | `-c` | Configuration file path | `~/.config/bitnet/config.toml` |
| `--output` | | Output format (json, yaml, table) | `table` |
| `--help` | `-h` | Show help information | |
| `--version` | `-V` | Show version information | |

## Commands Overview

### Core Operations
- **[infer](#inference-commands)** - Text generation and inference operations
- **[model](#model-commands)** - Model management and operations
- **[config](#configuration-commands)** - Configuration management

### Setup & Validation
- **[setup](#setup-command)** - Interactive environment setup wizard
- **[validate](#validate-command)** - System health validation and benchmarking
- **[quickstart](#quickstart-command)** - Quick start automation with example models

### Production Tools
- **[ops](#production-operations)** - Production operations for DevOps teams
- **[convert](#convert-command)** - Model format conversion
- **[benchmark](#benchmark-commands)** - Performance benchmarking and profiling

### Development (Coming Soon)
- **[train](#training-commands)** - Training and fine-tuning operations

---

## Inference Commands

The `infer` subcommand provides text generation capabilities with BitNet models.

### Chat Mode

Start an interactive chat session:

```bash
bitnet infer chat --model microsoft/bitnet-b1.58-2B-4T-gguf
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model name or path | Required |
| `--temperature` | Sampling temperature (0.0-2.0) | `0.7` |
| `--top-k` | Top-k sampling parameter | `50` |
| `--top-p` | Top-p sampling parameter | `0.9` |
| `--max-tokens` | Maximum tokens per response | `512` |

**Examples:**
```bash
# Basic chat
bitnet infer chat -m microsoft/bitnet-b1.58-2B-4T-gguf

# Creative chat with high temperature
bitnet infer chat -m microsoft/bitnet-b1.58-2B-4T-gguf --temperature 1.2

# Focused chat with low temperature
bitnet infer chat -m microsoft/bitnet-b1.58-2B-4T-gguf --temperature 0.3 --top-k 20
```

**Chat Commands:**
- `help` - Show available commands
- `exit` or `quit` - End chat session
- `clear` - Clear conversation history
- `stats` - Show generation statistics
- `config` - Show current settings

### Single Generation

Generate text from a single prompt:

```bash
bitnet infer generate --model microsoft/bitnet-b1.58-2B-4T-gguf --prompt "Explain quantum computing"
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model name or path | Required |
| `--prompt`, `-p` | Input prompt | Required |
| `--temperature` | Sampling temperature (0.0-2.0) | `0.7` |
| `--top-k` | Top-k sampling parameter | `50` |
| `--top-p` | Top-p sampling parameter | `0.9` |
| `--max-tokens` | Maximum tokens to generate | `512` |
| `--format` | Output format (text, json) | `text` |

**Examples:**
```bash
# Basic generation
bitnet infer generate -m microsoft/bitnet-b1.58-2B-4T-gguf -p "Write a poem about technology"

# JSON output for automation
bitnet infer generate -m microsoft/bitnet-b1.58-2B-4T-gguf -p "Summarize this:" --format json

# Long-form generation
bitnet infer generate -m microsoft/bitnet-b1.58-2B-4T-gguf -p "Write a story" --max-tokens 1024
```

### Batch Processing

Process multiple prompts from a file:

```bash
bitnet infer batch --model microsoft/bitnet-b1.58-2B-4T-gguf --input prompts.txt --output results.jsonl
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model name or path | Required |
| `--input`, `-i` | Input file path (one prompt per line) | Required |
| `--output`, `-o` | Output file path | Required |
| `--temperature` | Sampling temperature (0.0-2.0) | `0.7` |
| `--top-k` | Top-k sampling parameter | `50` |
| `--top-p` | Top-p sampling parameter | `0.9` |
| `--max-tokens` | Maximum tokens per prompt | `512` |

**Input Format** (`prompts.txt`):
```
Explain artificial intelligence
What is quantum computing?
Write a short story about robots
```

**Output Format** (`results.jsonl`):
```json
{"input": "Explain artificial intelligence", "output": "Artificial intelligence...", "tokens": 156, "time_ms": 234}
{"input": "What is quantum computing?", "output": "Quantum computing...", "tokens": 189, "time_ms": 267}
{"input": "Write a short story about robots", "output": "In a distant future...", "tokens": 423, "time_ms": 589}
```

### Model Management

Download and cache models:

```bash
bitnet infer download --model microsoft/bitnet-b1.58-2B-4T-gguf
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model name (HuggingFace ID) | Required |
| `--force` | Force re-download even if cached | `false` |

List available models:

```bash
bitnet infer list
bitnet infer list --cached  # Show only cached models
```

---

## Model Commands

### Convert (Coming Soon)

Convert models between formats:

```bash
bitnet model convert --input model.safetensors --output model.gguf
```

---

## Configuration Commands

### Show Configuration

Display current configuration:

```bash
bitnet config show
```

**Example Output:**
```yaml
generation:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  max_tokens: 512
output:
  format: table
  verbose: false
cache:
  directory: ~/.cache/bitnet
  max_size_gb: 10
```

### Set Configuration

Set configuration values:

```bash
bitnet config set generation.temperature 0.8
bitnet config set output.format json
bitnet config set cache.max_size_gb 20
```

### Get Configuration

Get specific configuration values:

```bash
bitnet config get generation.temperature
bitnet config get output.format
```

### Reset Configuration

Reset to default values:

```bash
bitnet config reset
bitnet config reset generation  # Reset only generation settings
```

---

## Setup Command

Interactive environment setup wizard:

```bash
bitnet setup
```

**Features:**
- ✅ System requirements validation
- ✅ Hardware capability detection (CPU, GPU)
- ✅ Dependency verification
- ✅ Performance optimization recommendations
- ✅ Model download suggestions
- ✅ Configuration file creation

**Example Session:**
```
🚀 BitNet-Rust Setup Wizard

✅ System Requirements
  ├─ Rust 1.70+: ✅ 1.75.0
  ├─ Available Memory: ✅ 16.0 GB
  └─ Disk Space: ✅ 50.2 GB free

🔍 Hardware Detection
  ├─ CPU: ✅ Apple M2 Pro (12 cores)
  ├─ NEON Support: ✅ Available
  ├─ Metal GPU: ✅ Available
  └─ CUDA GPU: ❌ Not available

⚙️ Optimization Recommendations
  ├─ Enable NEON optimizations: ✅ Configured
  ├─ Use Metal acceleration: ✅ Available
  └─ Memory pool size: ✅ Set to 4GB

📥 Recommended Models
  ├─ microsoft/bitnet-b1.58-2B-4T-gguf (2.1 GB)
  └─ microsoft/bitnet-b1.58-3B-instruct-gguf (3.2 GB)

Would you like to download microsoft/bitnet-b1.58-2B-4T-gguf? [y/N]
```

---

## Validate Command

System health validation and performance benchmarking:

```bash
bitnet validate
bitnet validate --quick      # Quick validation only
bitnet validate --benchmark  # Include performance benchmarks
```

**Features:**
- ✅ Hardware capability testing
- ✅ Memory allocation validation
- ✅ Performance benchmarking
- ✅ Model loading verification
- ✅ Inference accuracy testing

---

## Quickstart Command

Quick start automation with example models:

```bash
bitnet quickstart
bitnet quickstart --model microsoft/bitnet-b1.58-2B-4T-gguf
```

**Actions:**
1. Downloads recommended model
2. Runs sample inference
3. Shows example commands
4. Creates starter configuration

---

## Production Operations

The `ops` subcommand provides tools for DevOps teams and production deployments.

```bash
bitnet ops health-check     # System health monitoring
bitnet ops metrics          # Performance metrics collection  
bitnet ops deployment       # Deployment validation tools
```

---

## Convert Command

Convert models between formats:

```bash
bitnet convert --input model.safetensors --output model.gguf
bitnet convert --input pytorch_model.bin --output model.gguf --format gguf
```

---

## Benchmark Commands (Coming Soon)

### Inference Benchmarking

```bash
bitnet benchmark inference --model microsoft/bitnet-b1.58-2B-4T-gguf
```

---

## Training Commands (Coming Soon)

### Start Training

```bash
bitnet train start --model microsoft/bitnet-b1.58-2B-4T-gguf --dataset dataset.jsonl
```

---

## Configuration File

BitNet CLI supports configuration files for persistent settings:

**Location**: `~/.config/bitnet/config.toml`

**Example Configuration:**
```toml
[generation]
temperature = 0.7
top_k = 50
top_p = 0.9
max_tokens = 512

[output]
format = "table"
verbose = false

[cache]
directory = "~/.cache/bitnet"
max_size_gb = 10

[device]
preferred = "auto"  # auto, cpu, metal, cuda
memory_limit_mb = 4096

[models]
default = "microsoft/bitnet-b1.58-2B-4T-gguf"

[logging]
level = "info"
file = "~/.cache/bitnet/logs/bitnet.log"
```

## Environment Variables

BitNet CLI respects these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BITNET_CONFIG` | Configuration file path | `~/.config/bitnet/config.toml` |
| `BITNET_CACHE_DIR` | Cache directory | `~/.cache/bitnet` |
| `BITNET_DEVICE` | Preferred device (cpu, metal, cuda) | `auto` |
| `BITNET_LOG_LEVEL` | Log level (trace, debug, info, warn, error) | `info` |
| `RUST_LOG` | Rust logging configuration | `bitnet=info` |

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Model loading error |
| 4 | Inference error |
| 5 | Network error |
| 6 | Hardware error |

## Examples Repository

Complete examples available at: [`examples/cli/`](../examples/cli/)

- **[Basic Usage](../examples/cli/basic-usage.md)** - Getting started examples
- **[Batch Processing](../examples/cli/batch-processing.md)** - File processing workflows
- **[Production Deployment](../examples/cli/production.md)** - Production setup examples
- **[Performance Tuning](../examples/cli/performance.md)** - Optimization examples

## Support

- **GitHub Issues**: [bitnet-rust/issues](https://github.com/leizerowicz/bitnet-rust/issues)
- **CLI Help**: `bitnet --help` or `bitnet <command> --help`
- **Documentation**: [Inference Guide](inference-guide.md)