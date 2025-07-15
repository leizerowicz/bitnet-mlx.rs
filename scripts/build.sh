#!/bin/bash

# Build script for BitNet Rust project
# This script sets the required RUSTFLAGS to avoid LTO conflicts

export RUSTFLAGS="-C embed-bitcode=yes"

echo "Building BitNet Rust project with RUSTFLAGS=$RUSTFLAGS"

# Run the command passed as arguments, or default to cargo build
if [ $# -eq 0 ]; then
    cargo build
else
    "$@"
fi