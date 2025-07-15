#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running BitNet benchmarks...${NC}"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -d "bitnet-benchmarks" ]; then
    echo -e "${RED}Error: Run this script from the workspace root${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p target/criterion

# Run specific benchmark suites
BENCHMARK_SUITES=(
    "quantization"
    "bitlinear" 
    "inference"
    "memory_usage"
    "end_to_end"
)

# Option to run specific benchmark
if [ $# -eq 1 ]; then
    SUITE=$1
    echo -e "${YELLOW}Running specific benchmark: $SUITE${NC}"
    cargo bench --package bitnet-benchmarks --bench "$SUITE"
else
    echo -e "${YELLOW}Running all benchmark suites...${NC}"
    
    for suite in "${BENCHMARK_SUITES[@]}"; do
        echo -e "${YELLOW}Running $suite benchmarks...${NC}"
        cargo bench --package bitnet-benchmarks --bench "$suite" || echo -e "${RED}Warning: $suite benchmark failed${NC}"
    done
fi

# Generate comparison report if baseline exists
if [ -f "target/criterion/baseline.json" ]; then
    echo -e "${YELLOW}Generating comparison report...${NC}"
    cargo bench --workspace -- --save-baseline current
    if command -v criterion-compare &> /dev/null; then
        criterion-compare baseline current
    else
        echo -e "${YELLOW}Install criterion-compare for detailed comparisons: cargo install criterion-compare${NC}"
    fi
fi

echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "Results available at: ${YELLOW}target/criterion/report/index.html${NC}"

# Open results in browser if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Opening results in browser...${NC}"
    open target/criterion/report/index.html
fi