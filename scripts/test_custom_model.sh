#!/bin/bash
# scripts/test_custom_model.sh
# BitNet Custom Model Testing Script - Step 7.1 Implementation
# 
# This script provides comprehensive testing for custom BitNet models following
# the orchestrator-driven workflow and test utilities specialist guidelines.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

log_test() {
    echo -e "${BLUE}🧪 TEST:${NC} $1"
}

# Configuration
MODEL_NAME=${1:-"custom/bitnet-small"}
TEST_TIMEOUT=${TEST_TIMEOUT:-300} # 5 minutes default timeout
VERBOSE=${VERBOSE:-"false"}

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test result tracking function
track_test() {
    local test_name="$1"
    local test_result="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$test_result" = "PASS" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_success "Test passed: $test_name"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_error "Test failed: $test_name"
    fi
}

# Function to run command with timeout and capture result
run_test_command() {
    local cmd="$1"
    local test_name="$2"
    local expected_exit_code="${3:-0}"
    
    log_test "Running: $test_name"
    log_info "Command: $cmd"
    
    # Check if gtimeout (GNU timeout) is available, fallback to native timeout
    local timeout_cmd="timeout"
    if command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout"
    elif ! command -v timeout &> /dev/null; then
        # For macOS without timeout, use a simple alternative
        log_warning "timeout command not available, running without timeout"
        if bash -c "$cmd" > /tmp/test_output.log 2>&1; then
            local exit_code=$?
            if [ $exit_code -eq $expected_exit_code ]; then
                track_test "$test_name" "PASS"
                if [ "$VERBOSE" = "true" ]; then
                    echo "Output:"
                    cat /tmp/test_output.log | head -20
                fi
                return 0
            else
                track_test "$test_name" "FAIL"
                log_error "Expected exit code $expected_exit_code, got $exit_code"
                echo "Error output:"
                cat /tmp/test_output.log | tail -10
                return 1
            fi
        else
            track_test "$test_name" "FAIL"
            log_error "Command failed"
            echo "Error output:"
            cat /tmp/test_output.log | tail -10
            return 1
        fi
    fi
    
    if $timeout_cmd "$TEST_TIMEOUT" bash -c "$cmd" > /tmp/test_output.log 2>&1; then
        local exit_code=$?
        if [ $exit_code -eq $expected_exit_code ]; then
            track_test "$test_name" "PASS"
            if [ "$VERBOSE" = "true" ]; then
                echo "Output:"
                cat /tmp/test_output.log | head -20
            fi
            return 0
        else
            track_test "$test_name" "FAIL"
            log_error "Expected exit code $expected_exit_code, got $exit_code"
            echo "Error output:"
            cat /tmp/test_output.log | tail -10
            return 1
        fi
    else
        track_test "$test_name" "FAIL"
        log_error "Command timed out or failed"
        echo "Error output:"
        cat /tmp/test_output.log | tail -10
        return 1
    fi
}

# Header
echo "🧪 BitNet Custom Model Testing Suite"
echo "===================================="
echo "Model: $MODEL_NAME"
echo "Timeout: ${TEST_TIMEOUT}s"
echo "Verbose: $VERBOSE"
echo ""

# Check if BitNet CLI is available
log_info "Checking BitNet CLI availability..."
if ! command -v cargo &> /dev/null; then
    log_error "Cargo not found. Please install Rust toolchain."
    exit 1
fi

# Verify BitNet CLI builds
log_test "Verifying BitNet CLI builds..."
if cargo check -p bitnet-cli --quiet; then
    log_success "BitNet CLI builds successfully"
else
    log_error "BitNet CLI failed to build"
    exit 1
fi

echo ""
echo "🔍 Starting Model Testing..."
echo ""

# Test 1: CLI Help System
log_test "Testing CLI help system..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- --help" \
    "CLI Help System" \
    0

# Test 2: Model List Command
log_test "Testing model listing..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- model list --verbose" \
    "Model List Command" \
    0

# Test 3: Model Information
log_test "Testing model information retrieval..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- model info $MODEL_NAME" \
    "Model Info Command" \
    0

# Test 4: Model Verification
log_test "Testing model file verification..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- model verify $MODEL_NAME" \
    "Model Verification" \
    0

# Test 5: System Validation
log_test "Testing system validation..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- validate --system-health" \
    "System Health Validation" \
    0

# Test 6: Quick Validation (non-comprehensive)
log_test "Testing quick validation..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- validate --skip-memory --skip-hardware --duration 5" \
    "Quick Validation" \
    0

# Test 7: Model Inference Dry Run (if implemented)
log_test "Testing inference dry run..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- infer generate --model $MODEL_NAME --prompt 'Test prompt' --max-tokens 10 --format json" \
    "Inference Dry Run" \
    0

# Test 8: Setup Validation
log_test "Testing setup wizard..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- setup --check-only --skip-models" \
    "Setup Validation" \
    0

# Test 9: Configuration Management
log_test "Testing configuration management..."
run_test_command \
    "cargo run -p bitnet-cli --bin bitnet -- config show" \
    "Configuration Show" \
    0

# Test 10: Workspace Tests (if running in proper workspace)
if [ -f "Cargo.toml" ] && grep -q "bitnet" Cargo.toml; then
    log_test "Running workspace tests..."
    run_test_command \
        "cargo test --workspace --lib --bins" \
        "Workspace Tests" \
        0
else
    log_warning "Not in BitNet workspace, skipping workspace tests"
fi

# Test 11: Custom Model Loading Test (synthetic)
log_test "Testing custom model loading infrastructure..."
run_test_command \
    "cargo test -p bitnet-training --lib custom_bitnet" \
    "Custom Model Infrastructure Tests" \
    0

# Test 12: GGUF Export Infrastructure Test
log_test "Testing GGUF export infrastructure..."
run_test_command \
    "cargo test -p bitnet-training --lib gguf_export" \
    "GGUF Export Tests" \
    0

# Test 13: MLX Integration Test (if available)
log_test "Testing MLX integration..."
run_test_command \
    "cargo test -p bitnet-quant --lib mlx_bitnet" \
    "MLX Integration Tests" \
    0

# Performance and Stress Tests (if enabled)
if [ "$COMPREHENSIVE" = "true" ]; then
    echo ""
    echo "🚀 Running Comprehensive Performance Tests..."
    
    # Test 14: Performance Benchmark
    log_test "Running performance benchmark..."
    run_test_command \
        "cargo run -p bitnet-cli --bin bitnet -- validate --benchmark-only --duration 30" \
        "Performance Benchmark" \
        0
        
    # Test 15: Memory Stress Test
    log_test "Running memory stress test..."
    run_test_command \
        "cargo run -p bitnet-cli --bin bitnet -- validate --comprehensive" \
        "Memory Stress Test" \
        0
fi

# Results Summary
echo ""
echo "📊 Test Results Summary"
echo "======================"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    log_success "All tests passed! Custom model testing complete."
    echo ""
    echo "✅ Model $MODEL_NAME is ready for use"
    echo "✅ BitNet CLI is functioning correctly"
    echo "✅ Custom model infrastructure validated"
    exit 0
else
    log_error "$FAILED_TESTS tests failed"
    echo ""
    echo "❌ Some tests failed. Please review the errors above."
    echo "❌ Check logs in /tmp/test_output.log for detailed information"
    
    # Calculate success rate
    success_rate=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "N/A")
    echo "📈 Success Rate: ${success_rate}%"
    
    if [ "$success_rate" != "N/A" ] && (( $(echo "$success_rate >= 80" | bc -l) )); then
        log_warning "Success rate >= 80%, model may still be usable with caution"
        exit 1
    else
        log_error "Success rate < 80%, model needs attention before use"
        exit 2
    fi
fi