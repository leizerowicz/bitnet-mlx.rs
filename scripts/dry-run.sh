#!/bin/bash

# BitNet-Rust Publishing Dry Run Script
# Tests if all crates can be packaged for publishing

set -e

echo "🧪 Testing BitNet-Rust crate publishing (dry run)..."

# Test each crate in dependency order
crates=("bitnet-metal" "bitnet-core" "bitnet-quant" "bitnet-inference" "bitnet-training" "bitnet-cli" "bitnet-benchmarks")

for i in "${!crates[@]}"; do
    crate="${crates[$i]}"
    num=$((i + 1))
    total=${#crates[@]}
    
    echo ""
    echo "📦 Testing $crate ($num/$total)..."
    
    if cargo publish --dry-run --allow-dirty -p "$crate"; then
        echo "✅ $crate dry run successful"
    else
        echo "❌ $crate dry run failed"
        exit 1
    fi
done

echo ""
echo "🎉 All crates passed dry run tests!"
echo "✅ Ready for publishing to crates.io"
