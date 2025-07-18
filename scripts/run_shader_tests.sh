#!/bin/bash

# BitNet Shader Compilation Test Runner
# This script runs comprehensive shader compilation tests

set -e

echo "=== BitNet Shader Compilation Test Suite ==="
echo "Date: $(date)"
echo "Platform: $(uname -s) $(uname -m)"
echo "Rust version: $(rustc --version)"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run a test and capture results
run_test() {
    local test_name=$1
    local test_command=$2
    
    print_status $BLUE "Running: $test_name"
    echo "Command: $test_command"
    echo "----------------------------------------"
    
    if eval $test_command; then
        print_status $GREEN "✓ $test_name PASSED"
    else
        print_status $RED "✗ $test_name FAILED"
        return 1
    fi
    echo
}

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status $GREEN "Running on macOS - Metal tests enabled"
    METAL_AVAILABLE=true
else
    print_status $YELLOW "Not running on macOS - Metal tests will be skipped"
    METAL_AVAILABLE=false
fi

# Change to project directory
cd "$(dirname "$0")/.."

# Ensure we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_status $RED "Error: Not in BitNet project root directory"
    exit 1
fi

print_status $BLUE "Project directory: $(pwd)"
echo

# Build the project first
print_status $BLUE "Building project..."
if cargo build --features metal; then
    print_status $GREEN "✓ Build successful"
else
    print_status $RED "✗ Build failed"
    exit 1
fi
echo

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to increment test counters
update_counters() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $? -eq 0 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Run individual test script
print_status $BLUE "=== Running Individual Test Script ==="
if run_test "Individual Shader Compilation Test" "cargo run --bin test_shader_compilation --features metal"; then
    update_counters
else
    update_counters
fi

# Run comprehensive test suite
print_status $BLUE "=== Running Comprehensive Test Suite ==="
if run_test "Comprehensive Shader Tests" "cargo test --test comprehensive_shader_compilation_tests --features metal -- --nocapture"; then
    update_counters
else
    update_counters
fi

# Run edge case tests
print_status $BLUE "=== Running Edge Case Tests ==="
if run_test "Edge Case Tests" "cargo test --test shader_compilation_edge_cases --features metal -- --nocapture"; then
    update_counters
else
    update_counters
fi

# Run performance tests
print_status $BLUE "=== Running Performance Tests ==="
if run_test "Performance Tests" "cargo test --test shader_compilation_performance_tests --features metal -- --nocapture"; then
    update_counters
else
    update_counters
fi

# Run existing Metal shader tests
print_status $BLUE "=== Running Existing Metal Tests ==="
if run_test "Existing Metal Shader Tests" "cargo test --test metal_shader_compilation_tests --features metal -- --nocapture"; then
    update_counters
else
    update_counters
fi

# Run all Metal-related tests
print_status $BLUE "=== Running All Metal Tests ==="
if run_test "All Metal Tests" "cargo test metal --features metal -- --nocapture"; then
    update_counters
else
    update_counters
fi

# Generate test report
echo
print_status $BLUE "=== Test Results Summary ==="
echo "Total tests run: $TOTAL_TESTS"
print_status $GREEN "Passed: $PASSED_TESTS"
print_status $RED "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    print_status $GREEN "🎉 All tests passed!"
    SUCCESS_RATE="100%"
else
    SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    print_status $YELLOW "Success rate: ${SUCCESS_RATE}%"
fi

# Additional information
echo
print_status $BLUE "=== Additional Information ==="
echo "Test artifacts location: target/"
echo "Shader cache location: target/*shader_cache*"
echo "Log files: Check cargo test output above"

if [ "$METAL_AVAILABLE" = true ]; then
    echo "Metal device info:"
    system_profiler SPDisplaysDataType | grep -A 5 "Metal"
fi

echo
print_status $BLUE "=== Cleanup ==="
echo "Cleaning up test artifacts..."
rm -rf target/test_*_cache target/*_perf_cache target/cache_perf_test target/memory_test_cache target/concurrent_perf_cache 2>/dev/null || true
print_status $GREEN "✓ Cleanup completed"

echo
if [ $FAILED_TESTS -eq 0 ]; then
    print_status $GREEN "=== All shader compilation tests completed successfully! ==="
    exit 0
else
    print_status $RED "=== Some tests failed. Check output above for details. ==="
    exit 1
fi