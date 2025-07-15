#!/bin/bash
set -e

echo "Building documentation..."

# Build API documentation
echo "Building API docs..."
cargo doc --workspace --no-deps

# Build mdBook documentation
echo "Building mdBook..."
cd docs
mdbook build

echo "Documentation built successfully!"
echo "API docs: target/doc/index.html"
echo "Book: docs/book/index.html"