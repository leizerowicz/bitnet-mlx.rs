#!/bin/bash

# BitNet Agent Config Framework Demo
# Demonstrates the complete agent config generation and validation workflow

set -e

echo "🎯 BitNet Agent Config Framework Demo"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Please run this script from the agent-config-framework directory"
    exit 1
fi

echo "📦 Building the framework..."
cargo build --release

echo ""
echo "🤖 Generating example agent configurations..."

# Generate a few example agents to demonstrate the framework
./target/release/agent-config-cli generate \
    --name "example_code_specialist" \
    --domain "code" \
    --container-role "Specialist" \
    --output "../agent-config" \
    --templates "templates"

echo "✅ Generated: example_code_specialist.md"

./target/release/agent-config-cli generate \
    --name "example_performance_specialist" \
    --domain "performance" \
    --container-role "Specialist" \
    --output "../agent-config" \
    --templates "templates"

echo "✅ Generated: example_performance_specialist.md"

./target/release/agent-config-cli generate \
    --name "example_support_agent" \
    --domain "monitoring" \
    --container-role "Support" \
    --output "../agent-config" \
    --templates "templates"

echo "✅ Generated: example_support_agent.md"

echo ""
echo "🔍 Validating all agent configurations..."
./target/release/agent-config-cli validate --verbose

echo ""
echo "🔄 Updating intersection matrix..."
./target/release/agent-config-cli update-matrix --update-configs

echo ""
echo "📊 Showing framework status..."
./target/release/agent-config-cli status --verbose

echo ""
echo "✅ Demo completed successfully!"
echo ""
echo "📁 Generated files:"
echo "   - ../agent-config/example_code_specialist.md"
echo "   - ../agent-config/example_performance_specialist.md" 
echo "   - ../agent-config/example_support_agent.md"
echo "   - ../agent-config/agent-intersection-matrix.json"
echo ""
echo "🎯 Next steps:"
echo "   1. Review generated agent configurations"
echo "   2. Customize templates in templates/ directory"
echo "   3. Run validation to ensure quality standards"
echo "   4. Deploy container with: ./target/release/agent-config-cli deploy --monitor"