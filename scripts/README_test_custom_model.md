# Custom BitNet Model Testing Script

## Overview
The `test_custom_model.sh` script provides comprehensive testing capabilities for custom BitNet models following the orchestrator-driven workflow and test utilities specialist guidelines.

## Location
```bash
scripts/test_custom_model.sh
```

## Usage

### Basic Usage
```bash
# Test a specific model
./scripts/test_custom_model.sh custom/bitnet-small

# Test with verbose output
VERBOSE=true ./scripts/test_custom_model.sh custom/bitnet-small

# Comprehensive testing (includes performance tests)
COMPREHENSIVE=true ./scripts/test_custom_model.sh custom/bitnet-small

# Custom timeout
TEST_TIMEOUT=600 ./scripts/test_custom_model.sh custom/bitnet-small
```

## Test Coverage

### Core Tests (Always Run)
1. **CLI Help System Test** - Validates BitNet CLI help functionality
2. **Model List Command Test** - Tests model discovery and listing
3. **Model Information Test** - Retrieves and validates model metadata
4. **Model Verification Test** - Verifies model file integrity
5. **System Health Validation** - Basic system health checks
6. **Quick Validation Test** - Fast validation without memory/hardware checks
7. **Inference Dry Run Test** - Tests inference command structure
8. **Setup Validation Test** - Validates environment setup
9. **Configuration Management Test** - Tests configuration display

### Infrastructure Tests (Always Run)
10. **Workspace Tests** - Validates workspace compilation
11. **Custom Model Infrastructure** - Tests custom model loading system
12. **GGUF Export Infrastructure** - Tests GGUF conversion capabilities
13. **MLX Integration Tests** - Tests MLX framework integration

### Extended Tests (Comprehensive Mode Only)
14. **Performance Benchmarks** - Detailed performance testing
15. **Memory Stress Tests** - Comprehensive memory validation

## Features

- **macOS Compatible**: Handles timeout command availability gracefully
- **Error Reporting**: Detailed error capture with log files
- **Result Tracking**: Pass/fail counting with success rate calculation
- **Configurable Timeouts**: Adjustable command timeouts
- **Verbose Mode**: Optional detailed output for debugging
- **Modular Testing**: Optional comprehensive mode for extended testing

## Output Format

The script provides color-coded output:
- 🧪 **TEST**: Test execution indicators
- ✅ **SUCCESS**: Passed tests
- ❌ **ERROR**: Failed tests  
- ⚠️ **WARNING**: Non-critical issues
- ℹ️ **INFO**: General information

## Exit Codes

- `0`: All tests passed
- `1`: Some tests failed but success rate >= 80%
- `2`: Success rate < 80%, requires attention

## Integration with BitNet-Rust

This script is designed to work with the existing BitNet-Rust CLI infrastructure and follows the agent configuration system patterns established in the `agent-config/` directory.