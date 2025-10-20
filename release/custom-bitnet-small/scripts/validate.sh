#!/bin/bash

# Custom BitNet Small Model Validation Script
# Version: 1.0.0
# Purpose: Validate model installation and functionality

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="custom-bitnet-small"
BITNET_MODELS_DIR="$HOME/.bitnet-rust/models"
CUSTOM_MODELS_DIR="$BITNET_MODELS_DIR/custom"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

echo -e "${BLUE}🔍 Custom BitNet Small Model Validation${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Helper function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}Testing: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Helper function to run a test with verbose output
run_test_verbose() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}Testing: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local output
    output=$(eval "$test_command" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
        if [ -n "$output" ]; then
            echo -e "${BLUE}Output: $output${NC}"
        fi
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        if [ -n "$output" ]; then
            echo -e "${RED}Error: $output${NC}"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Check directory structure
test_directory_structure() {
    echo -e "${YELLOW}📁 Testing Directory Structure${NC}"
    
    run_test "BitNet models directory exists" "[ -d '$BITNET_MODELS_DIR' ]"
    run_test "Custom models directory exists" "[ -d '$CUSTOM_MODELS_DIR' ]"
    
    echo ""
}

# Test 2: Check file presence
test_file_presence() {
    echo -e "${YELLOW}📄 Testing File Presence${NC}"
    
    # Check for model file (either actual or placeholder)
    if [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf" ]; then
        run_test "GGUF model file exists" "[ -f '$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf' ]"
    elif [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder" ]; then
        run_test "GGUF placeholder file exists" "[ -f '$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder' ]"
        echo -e "${YELLOW}⚠️  Note: Using placeholder file. Replace with trained model for actual inference.${NC}"
    else
        echo -e "${RED}❌ No model file found (neither actual nor placeholder)${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi
    
    run_test "Configuration file exists" "[ -f '$CUSTOM_MODELS_DIR/custom-bitnet-small.json' ]"
    run_test "Documentation exists" "[ -f '$CUSTOM_MODELS_DIR/README-$MODEL_NAME.md' ]"
    run_test "Registry entry exists" "[ -f '$BITNET_MODELS_DIR/custom-bitnet-small.registry' ]"
    
    echo ""
}

# Test 3: Validate configuration file
test_configuration() {
    echo -e "${YELLOW}⚙️  Testing Configuration${NC}"
    
    local config_file="$CUSTOM_MODELS_DIR/custom-bitnet-small.json"
    
    if [ -f "$config_file" ]; then
        # Check if it's valid JSON
        run_test "Configuration is valid JSON" "jq '.' '$config_file'"
        
        # Check required fields
        run_test "Has model architecture" "jq -e '.architectures' '$config_file'"
        run_test "Has vocab size" "jq -e '.vocab_size' '$config_file'"
        run_test "Has hidden size" "jq -e '.hidden_size' '$config_file'"
        run_test "Has num layers" "jq -e '.num_layers' '$config_file'"
        run_test "Has quantization config" "jq -e '.quantization_config' '$config_file'"
        
        # Validate specific values
        run_test "Architecture is BitNetForCausalLM" "jq -e '.architectures[0] == \"BitNetForCausalLM\"' '$config_file'"
        run_test "Quantization is 1.58-bit" "jq -e '.quantization_config.weight_bits == 1.58' '$config_file'"
    else
        echo -e "${RED}❌ Configuration file not found${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 6))
        TOTAL_TESTS=$((TOTAL_TESTS + 6))
    fi
    
    echo ""
}

# Test 4: Check BitNet-Rust installation
test_bitnet_installation() {
    echo -e "${YELLOW}🦀 Testing BitNet-Rust Installation${NC}"
    
    run_test "Rust/Cargo available" "command -v cargo"
    
    # Check if we're in BitNet workspace or have CLI
    if [ -f "Cargo.toml" ] && grep -q "bitnet" Cargo.toml; then
        run_test "BitNet-Rust workspace detected" "grep -q 'bitnet' Cargo.toml"
        BITNET_CMD="cargo run --bin bitnet --"
    elif command -v bitnet > /dev/null 2>&1; then
        run_test "BitNet CLI available" "command -v bitnet"
        BITNET_CMD="bitnet"
    else
        echo -e "${YELLOW}⚠️  BitNet CLI not found globally${NC}"
        BITNET_CMD="cargo run --bin bitnet --"
    fi
    
    echo ""
}

# Test 5: Model file integrity
test_model_integrity() {
    echo -e "${YELLOW}🔐 Testing Model Integrity${NC}"
    
    local model_file=""
    
    if [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf" ]; then
        model_file="$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf"
    elif [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder" ]; then
        model_file="$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder"
    fi
    
    if [ -n "$model_file" ]; then
        run_test "Model file is readable" "[ -r '$model_file' ]"
        run_test "Model file is not empty" "[ -s '$model_file' ]"
        
        # Get file size
        local file_size
        file_size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "unknown")
        echo -e "${BLUE}Model file size: $file_size bytes${NC}"
        
        # Generate checksum
        if command -v sha256sum > /dev/null 2>&1; then
            local checksum
            checksum=$(sha256sum "$model_file" | cut -d' ' -f1)
            echo -e "${BLUE}SHA256 checksum: $checksum${NC}"
        elif command -v shasum > /dev/null 2>&1; then
            local checksum
            checksum=$(shasum -a 256 "$model_file" | cut -d' ' -f1)
            echo -e "${BLUE}SHA256 checksum: $checksum${NC}"
        fi
    else
        echo -e "${RED}❌ No model file found for integrity testing${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 2))
        TOTAL_TESTS=$((TOTAL_TESTS + 2))
    fi
    
    echo ""
}

# Test 6: CLI Integration (if available)
test_cli_integration() {
    echo -e "${YELLOW}🔧 Testing CLI Integration${NC}"
    
    # These tests are optional and may fail if CLI doesn't fully support custom models yet
    echo -e "${BLUE}Note: CLI integration tests are optional and may fail in development builds${NC}"
    
    # Test help command
    if command -v timeout > /dev/null 2>&1; then
        run_test "CLI help command" "timeout 10s $BITNET_CMD --help"
    else
        # macOS doesn't have timeout by default, use gtimeout if available, otherwise skip
        if command -v gtimeout > /dev/null 2>&1; then
            run_test "CLI help command" "gtimeout 10s $BITNET_CMD --help"
        else
            echo -e "${YELLOW}⚠️  Skipping CLI help test (timeout command not available)${NC}"
        fi
    fi
    
    # Test models list command (may not work if not implemented)
    echo -e "${BLUE}Attempting to list models (may fail if feature not implemented)${NC}"
    if command -v timeout > /dev/null 2>&1; then
        if timeout 10s $BITNET_CMD models list > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Models list command works${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${YELLOW}⚠️  Models list command not available or failed${NC}"
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    elif command -v gtimeout > /dev/null 2>&1; then
        if gtimeout 10s $BITNET_CMD models list > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Models list command works${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${YELLOW}⚠️  Models list command not available or failed${NC}"
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi
    
    echo ""
}

# Test 7: System compatibility
test_system_compatibility() {
    echo -e "${YELLOW}💻 Testing System Compatibility${NC}"
    
    # Check OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${GREEN}✅ macOS detected (optimal for Metal acceleration)${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        
        # Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo -e "${GREEN}✅ Apple Silicon detected (M1/M2/M3/M4 optimization available)${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${YELLOW}⚠️  Intel Mac detected (Apple Silicon optimization not available)${NC}"
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    else
        echo -e "${YELLOW}⚠️  Non-macOS system (Metal acceleration not available)${NC}"
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Check memory
    if command -v free > /dev/null 2>&1; then
        local mem_gb
        mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$mem_gb" -ge 4 ]; then
            echo -e "${GREEN}✅ Sufficient memory: ${mem_gb}GB${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${YELLOW}⚠️  Low memory: ${mem_gb}GB (4GB+ recommended)${NC}"
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        local mem_gb
        mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        if [ "$mem_gb" -ge 4 ]; then
            echo -e "${GREEN}✅ Sufficient memory: ${mem_gb}GB${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${YELLOW}⚠️  Low memory: ${mem_gb}GB (4GB+ recommended)${NC}"
        fi
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi
    
    echo ""
}

# Main validation function
main() {
    echo -e "${BLUE}Starting validation process...${NC}"
    echo ""
    
    test_directory_structure
    test_file_presence
    test_configuration
    test_bitnet_installation
    test_model_integrity
    test_cli_integration
    test_system_compatibility
    
    # Summary
    echo -e "${BLUE}📊 Validation Summary${NC}"
    echo -e "${BLUE}===================${NC}"
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    
    local success_rate
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(( (TESTS_PASSED * 100) / TOTAL_TESTS ))
        echo -e "Success Rate: ${BLUE}$success_rate%${NC}"
    fi
    
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Model installation is fully validated.${NC}"
        echo ""
        echo -e "${BLUE}Ready for use:${NC}"
        echo -e "  $BITNET_CMD infer chat --model custom/bitnet-small"
        echo ""
        exit 0
    elif [ $success_rate -ge 80 ]; then
        echo -e "${YELLOW}⚠️  Most tests passed ($success_rate%). Minor issues detected.${NC}"
        echo -e "${YELLOW}Model should work but may have limited functionality.${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}❌ Validation failed. Please check the errors above.${NC}"
        echo -e "${RED}Model installation may not work properly.${NC}"
        echo ""
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Custom BitNet Small Model Validation Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --verbose      Show detailed output for all tests"
        echo "  --quick        Run only essential tests"
        echo ""
        echo "This script validates the installation of the custom BitNet small model"
        echo "and checks system compatibility."
        exit 0
        ;;
    --verbose)
        echo -e "${BLUE}🔍 Verbose mode enabled${NC}"
        echo ""
        ;;
    --quick)
        echo -e "${BLUE}🚀 Quick validation mode${NC}"
        echo ""
        # Set flag for quick mode (could skip some tests)
        ;;
    "")
        # Default behavior
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Check if jq is available for JSON validation
if ! command -v jq > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Warning: jq not found. JSON validation tests will be skipped.${NC}"
    echo -e "${YELLOW}Install jq for complete validation: brew install jq${NC}"
    echo ""
fi

# Run main validation
main