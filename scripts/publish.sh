#!/bin/bash

# BitNet-Rust Publishing Script
# This script publishes all crates in the correct dependency order

set -e

echo "🚀 Starting BitNet-Rust crate publishing process..."

# Step 1: Publish bitnet-metal (no dependencies)
echo "📦 Publishing bitnet-metal (1/7)..."
cargo publish -p bitnet-metal --allow-dirty
echo "✅ bitnet-metal published successfully"

# Wait for crates.io to index the new version
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 2: Publish bitnet-core (depends on bitnet-metal optionally)
echo "📦 Publishing bitnet-core (2/7)..."
cargo publish -p bitnet-core --allow-dirty
echo "✅ bitnet-core published successfully"

# Wait for indexing
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 3: Publish bitnet-quant (depends on bitnet-core)
echo "📦 Publishing bitnet-quant (3/7)..."
cargo publish -p bitnet-quant --allow-dirty
echo "✅ bitnet-quant published successfully"

# Wait for indexing
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 4: Publish bitnet-inference (depends on bitnet-core, bitnet-quant, bitnet-metal)
echo "📦 Publishing bitnet-inference (4/7)..."
cargo publish -p bitnet-inference --allow-dirty
echo "✅ bitnet-inference published successfully"

# Wait for indexing
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 5: Publish bitnet-training (depends on bitnet-core, bitnet-quant)
echo "📦 Publishing bitnet-training (5/7)..."
cargo publish -p bitnet-training --allow-dirty
echo "✅ bitnet-training published successfully"

# Wait for indexing
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 6: Publish bitnet-cli (depends on bitnet-core, bitnet-quant)
echo "📦 Publishing bitnet-cli (6/7)..."
cargo publish -p bitnet-cli --allow-dirty
echo "✅ bitnet-cli published successfully"

# Wait for indexing
echo "⏳ Waiting 30 seconds for crates.io indexing..."
sleep 30

# Step 7: Publish bitnet-benchmarks (depends on bitnet-core, bitnet-quant)
echo "📦 Publishing bitnet-benchmarks (7/7)..."
cargo publish -p bitnet-benchmarks --allow-dirty
echo "✅ bitnet-benchmarks published successfully"

echo ""
echo "🎉 All BitNet-Rust crates published successfully!"
echo "📋 Published crates:"
echo "   • bitnet-metal v1.0.0"
echo "   • bitnet-core v1.0.0"
echo "   • bitnet-quant v1.0.0"
echo "   • bitnet-inference v1.0.0"
echo "   • bitnet-training v1.0.0"
echo "   • bitnet-cli v1.0.0"
echo "   • bitnet-benchmarks v1.0.0"
echo ""
echo "🔗 To use in your projects:"
echo "   cargo add bitnet-core"
echo "   cargo add bitnet-inference"
echo "   # Or add to Cargo.toml:"
echo "   [dependencies]"
echo "   bitnet-core = \"1.0\""
echo "   bitnet-inference = \"1.0\""
