#!/bin/bash

# Custom BitNet Small Model Installation Script
# Version: 1.0.0
# Purpose: Install custom BitNet model into BitNet-Rust ecosystem

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation configuration
MODEL_NAME="custom-bitnet-small"
MODEL_VERSION="v1.0"
BITNET_MODELS_DIR="$HOME/.bitnet-rust/models"
CUSTOM_MODELS_DIR="$BITNET_MODELS_DIR/custom"

echo -e "${BLUE}🚀 Custom BitNet Small Model Installation Script${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "Model: ${YELLOW}$MODEL_NAME${NC}"
echo -e "Version: ${YELLOW}$MODEL_VERSION${NC}"
echo -e "Target Directory: ${YELLOW}$CUSTOM_MODELS_DIR${NC}"
echo ""

# Check if BitNet-Rust is installed
check_bitnet_rust() {
    echo -e "${BLUE}🔍 Checking BitNet-Rust installation...${NC}"
    
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}❌ Error: Rust/Cargo not found. Please install Rust first.${NC}"
        echo -e "Visit: https://rustup.rs/"
        exit 1
    fi
    
    # Check if we're in a BitNet-Rust workspace or if bitnet CLI is available
    if [ -f "Cargo.toml" ] && grep -q "bitnet" Cargo.toml; then
        echo -e "${GREEN}✅ BitNet-Rust workspace detected${NC}"
        BITNET_CMD="cargo run --bin bitnet --"
    elif command -v bitnet &> /dev/null; then
        echo -e "${GREEN}✅ BitNet CLI found${NC}"
        BITNET_CMD="bitnet"
    else
        echo -e "${YELLOW}⚠️  BitNet CLI not found globally, will use cargo run${NC}"
        BITNET_CMD="cargo run --bin bitnet --"
    fi
}

# Create necessary directories
create_directories() {
    echo -e "${BLUE}📁 Creating model directories...${NC}"
    
    mkdir -p "$BITNET_MODELS_DIR"
    mkdir -p "$CUSTOM_MODELS_DIR"
    
    echo -e "${GREEN}✅ Directories created:${NC}"
    echo -e "  - $BITNET_MODELS_DIR"
    echo -e "  - $CUSTOM_MODELS_DIR"
}

# Install model files
install_model_files() {
    echo -e "${BLUE}📦 Installing model files...${NC}"
    
    # Check if model file exists (handle both actual GGUF and placeholder)
    if [ -f "models/custom-bitnet-small.gguf" ]; then
        MODEL_FILE="models/custom-bitnet-small.gguf"
        echo -e "${GREEN}✅ Found actual GGUF model file${NC}"
    elif [ -f "models/custom-bitnet-small.gguf.placeholder" ]; then
        MODEL_FILE="models/custom-bitnet-small.gguf.placeholder"
        echo -e "${YELLOW}⚠️  Found placeholder model file${NC}"
        echo -e "${YELLOW}    Note: Replace with actual trained model for inference${NC}"
    else
        echo -e "${RED}❌ Error: No model file found${NC}"
        exit 1
    fi
    
    # Copy model file
    cp "$MODEL_FILE" "$CUSTOM_MODELS_DIR/"
    
    # Copy configuration
    if [ -f "config/custom-bitnet-small.json" ]; then
        cp "config/custom-bitnet-small.json" "$CUSTOM_MODELS_DIR/"
        echo -e "${GREEN}✅ Configuration file copied${NC}"
    else
        echo -e "${YELLOW}⚠️  Configuration file not found${NC}"
    fi
    
    # Copy documentation
    if [ -f "README.md" ]; then
        cp "README.md" "$CUSTOM_MODELS_DIR/README-$MODEL_NAME.md"
        echo -e "${GREEN}✅ Documentation copied${NC}"
    fi
    
    echo -e "${GREEN}✅ Model files installed to: $CUSTOM_MODELS_DIR${NC}"
}

# Register model with BitNet-Rust CLI
register_model() {
    echo -e "${BLUE}🔗 Registering model with BitNet-Rust CLI...${NC}"
    
    # Determine the actual model file path
    if [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf" ]; then
        MODEL_PATH="$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf"
    else
        MODEL_PATH="$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder"
    fi
    
    # Try to register the model (this might not work if the CLI doesn't support registration yet)
    echo -e "${BLUE}Attempting to register model: custom/bitnet-small${NC}"
    
    # Create a simple model registry entry (manual approach)
    REGISTRY_FILE="$BITNET_MODELS_DIR/registry.json"
    
    if [ ! -f "$REGISTRY_FILE" ]; then
        echo '{"models": {}}' > "$REGISTRY_FILE"
    fi
    
    # Add entry to registry (simple approach)
    cat > "$BITNET_MODELS_DIR/custom-bitnet-small.registry" << EOF
{
    "name": "custom/bitnet-small",
    "version": "1.0.0",
    "path": "$MODEL_PATH",
    "config": "$CUSTOM_MODELS_DIR/custom-bitnet-small.json",
    "description": "Custom BitNet 1.58-bit model with 16M parameters",
    "architecture": "BitNet",
    "parameters": "16M",
    "quantization": "1.58-bit",
    "installed": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    echo -e "${GREEN}✅ Model registry entry created${NC}"
}

# Verify installation
verify_installation() {
    echo -e "${BLUE}✅ Verifying installation...${NC}"
    
    # Check files exist
    local files_ok=true
    
    if [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf" ] || [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.gguf.placeholder" ]; then
        echo -e "${GREEN}✅ Model file present${NC}"
    else
        echo -e "${RED}❌ Model file missing${NC}"
        files_ok=false
    fi
    
    if [ -f "$CUSTOM_MODELS_DIR/custom-bitnet-small.json" ]; then
        echo -e "${GREEN}✅ Configuration file present${NC}"
    else
        echo -e "${YELLOW}⚠️  Configuration file missing${NC}"
    fi
    
    if [ -f "$BITNET_MODELS_DIR/custom-bitnet-small.registry" ]; then
        echo -e "${GREEN}✅ Registry entry created${NC}"
    else
        echo -e "${YELLOW}⚠️  Registry entry missing${NC}"
    fi
    
    if [ "$files_ok" = true ]; then
        echo -e "${GREEN}✅ Installation verification passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Installation verification failed${NC}"
        return 1
    fi
}

# Main installation process
main() {
    echo -e "${BLUE}Starting installation process...${NC}"
    echo ""
    
    check_bitnet_rust
    echo ""
    
    create_directories
    echo ""
    
    install_model_files
    echo ""
    
    register_model
    echo ""
    
    if verify_installation; then
        echo ""
        echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
        echo ""
        echo -e "${BLUE}Next Steps:${NC}"
        echo -e "1. Run validation: ${YELLOW}./scripts/validate.sh${NC}"
        echo -e "2. Test the model: ${YELLOW}$BITNET_CMD infer chat --model custom/bitnet-small${NC}"
        echo -e "3. Check model info: ${YELLOW}$BITNET_CMD models info custom/bitnet-small${NC}"
        echo ""
        echo -e "${BLUE}Model Location: ${YELLOW}$CUSTOM_MODELS_DIR${NC}"
        echo -e "${BLUE}Documentation: ${YELLOW}$CUSTOM_MODELS_DIR/README-$MODEL_NAME.md${NC}"
        echo ""
        
        if [[ "$MODEL_FILE" == *".placeholder" ]]; then
            echo -e "${YELLOW}⚠️  NOTE: This installation uses a placeholder model file.${NC}"
            echo -e "${YELLOW}   To use for actual inference, replace with a trained GGUF model:${NC}"
            echo -e "${YELLOW}   cargo run --bin mlx-trainer -- --output $CUSTOM_MODELS_DIR/custom-bitnet-small.gguf${NC}"
            echo ""
        fi
    else
        echo ""
        echo -e "${RED}❌ Installation failed. Please check the errors above.${NC}"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Custom BitNet Small Model Installation Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --dry-run      Show what would be installed without actually installing"
        echo "  --force        Force reinstallation even if already installed"
        echo ""
        echo "This script installs the custom BitNet small model into your"
        echo "BitNet-Rust models directory (~/.bitnet-rust/models/custom/)"
        exit 0
        ;;
    --dry-run)
        echo -e "${BLUE}🔍 Dry run mode - showing what would be installed:${NC}"
        echo ""
        echo -e "Target directory: ${YELLOW}$CUSTOM_MODELS_DIR${NC}"
        echo -e "Model file: ${YELLOW}$(ls models/custom-bitnet-small.gguf* 2>/dev/null || echo 'Not found')${NC}"
        echo -e "Config file: ${YELLOW}$(ls config/custom-bitnet-small.json 2>/dev/null || echo 'Not found')${NC}"
        echo -e "Documentation: ${YELLOW}$(ls README.md 2>/dev/null || echo 'Not found')${NC}"
        echo ""
        echo -e "${BLUE}Run without --dry-run to perform actual installation.${NC}"
        exit 0
        ;;
    --force)
        echo -e "${YELLOW}⚠️  Force mode enabled - will overwrite existing installation${NC}"
        echo ""
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

# Run main installation
main