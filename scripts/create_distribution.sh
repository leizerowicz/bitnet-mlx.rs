#!/bin/bash

# Custom BitNet Small Model Distribution Creation Script
# Version: 1.0.0
# Purpose: Create distributable package for custom BitNet model

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="custom-bitnet-small"
MODEL_VERSION="v1.0"
RELEASE_DIR="release"
PACKAGE_DIR="$RELEASE_DIR/$MODEL_NAME"
DIST_FILE="$MODEL_NAME-$MODEL_VERSION.tar.gz"

echo -e "${BLUE}📦 Custom BitNet Model Distribution Creator${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "Model: ${YELLOW}$MODEL_NAME${NC}"
echo -e "Version: ${YELLOW}$MODEL_VERSION${NC}"
echo -e "Package Directory: ${YELLOW}$PACKAGE_DIR${NC}"
echo -e "Output File: ${YELLOW}$DIST_FILE${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Check if we're in the BitNet-Rust root directory
    if [ ! -f "Cargo.toml" ] || ! grep -q "bitnet" Cargo.toml; then
        echo -e "${RED}❌ Error: Must be run from BitNet-Rust root directory${NC}"
        exit 1
    fi
    
    # Check if package directory exists
    if [ ! -d "$PACKAGE_DIR" ]; then
        echo -e "${RED}❌ Error: Package directory not found: $PACKAGE_DIR${NC}"
        echo -e "${RED}   Run the install script first to create the distribution structure${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Validate package contents
validate_package() {
    echo -e "${BLUE}🔍 Validating package contents...${NC}"
    
    local validation_errors=0
    
    # Check required files
    local required_files=(
        "$PACKAGE_DIR/README.md"
        "$PACKAGE_DIR/config/custom-bitnet-small.json"
        "$PACKAGE_DIR/scripts/install.sh"
        "$PACKAGE_DIR/scripts/validate.sh"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}✅ Found: $file${NC}"
        else
            echo -e "${RED}❌ Missing: $file${NC}"
            validation_errors=$((validation_errors + 1))
        fi
    done
    
    # Check for model file (actual or placeholder)
    if [ -f "$PACKAGE_DIR/models/custom-bitnet-small.gguf" ]; then
        echo -e "${GREEN}✅ Found: GGUF model file${NC}"
    elif [ -f "$PACKAGE_DIR/models/custom-bitnet-small.gguf.placeholder" ]; then
        echo -e "${YELLOW}⚠️  Found: GGUF placeholder file${NC}"
        echo -e "${YELLOW}   Note: Actual trained model should be provided for production use${NC}"
    else
        echo -e "${RED}❌ Missing: GGUF model file (actual or placeholder)${NC}"
        validation_errors=$((validation_errors + 1))
    fi
    
    # Check script permissions
    if [ -x "$PACKAGE_DIR/scripts/install.sh" ]; then
        echo -e "${GREEN}✅ install.sh is executable${NC}"
    else
        echo -e "${YELLOW}⚠️  install.sh is not executable, fixing...${NC}"
        chmod +x "$PACKAGE_DIR/scripts/install.sh"
    fi
    
    if [ -x "$PACKAGE_DIR/scripts/validate.sh" ]; then
        echo -e "${GREEN}✅ validate.sh is executable${NC}"
    else
        echo -e "${YELLOW}⚠️  validate.sh is not executable, fixing...${NC}"
        chmod +x "$PACKAGE_DIR/scripts/validate.sh"
    fi
    
    if [ $validation_errors -gt 0 ]; then
        echo -e "${RED}❌ Validation failed with $validation_errors errors${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Package validation passed${NC}"
}

# Generate checksums
generate_checksums() {
    echo -e "${BLUE}🔐 Generating checksums...${NC}"
    
    local checksum_file="$PACKAGE_DIR/CHECKSUMS.txt"
    
    # Create checksums file
    cat > "$checksum_file" << EOF
# Custom BitNet Small Model Checksums
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Model: $MODEL_NAME
# Version: $MODEL_VERSION

EOF
    
    # Generate checksums for all files
    find "$PACKAGE_DIR" -type f ! -name "CHECKSUMS.txt" -print0 | while IFS= read -r -d '' file; do
        if command -v sha256sum > /dev/null 2>&1; then
            # Linux style
            sha256sum "$file" | sed "s|$PACKAGE_DIR/||" >> "$checksum_file"
        elif command -v shasum > /dev/null 2>&1; then
            # macOS style
            shasum -a 256 "$file" | sed "s|$PACKAGE_DIR/||" >> "$checksum_file"
        else
            echo -e "${YELLOW}⚠️  Warning: No checksum utility found${NC}"
            break
        fi
    done
    
    if [ -f "$checksum_file" ] && [ -s "$checksum_file" ]; then
        echo -e "${GREEN}✅ Checksums generated: $checksum_file${NC}"
        
        # Show first few checksums as verification
        echo -e "${BLUE}Sample checksums:${NC}"
        head -10 "$checksum_file" | tail -5
    else
        echo -e "${RED}❌ Failed to generate checksums${NC}"
        exit 1
    fi
}

# Create package info file
create_package_info() {
    echo -e "${BLUE}📋 Creating package information...${NC}"
    
    local info_file="$PACKAGE_DIR/PACKAGE_INFO.json"
    
    # Get package size
    local package_size
    package_size=$(du -sh "$PACKAGE_DIR" | cut -f1)
    
    # Count files
    local file_count
    file_count=$(find "$PACKAGE_DIR" -type f | wc -l)
    
    # Get model file info
    local model_file_info="null"
    if [ -f "$PACKAGE_DIR/models/custom-bitnet-small.gguf" ]; then
        local model_size
        model_size=$(stat -f%z "$PACKAGE_DIR/models/custom-bitnet-small.gguf" 2>/dev/null || stat -c%s "$PACKAGE_DIR/models/custom-bitnet-small.gguf" 2>/dev/null)
        model_file_info="{\"type\": \"actual\", \"size\": $model_size}"
    elif [ -f "$PACKAGE_DIR/models/custom-bitnet-small.gguf.placeholder" ]; then
        model_file_info="{\"type\": \"placeholder\", \"size\": null}"
    fi
    
    # Create JSON info file
    cat > "$info_file" << EOF
{
  "package": {
    "name": "$MODEL_NAME",
    "version": "$MODEL_VERSION",
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "size": "$package_size",
    "file_count": $file_count
  },
  "model": {
    "name": "Custom BitNet Small",
    "architecture": "BitNet",
    "quantization": "1.58-bit",
    "parameters": "16M",
    "layers": 16,
    "hidden_size": 1024,
    "vocab_size": 128256,
    "file": $model_file_info
  },
  "compatibility": {
    "bitnet_rust_version": "1.0+",
    "rust_version": "1.70+",
    "target_os": ["macOS", "Linux"],
    "optimal_platform": "Apple Silicon M1/M2/M3/M4"
  },
  "contents": {
    "readme": "README.md",
    "config": "config/custom-bitnet-small.json",
    "model": "models/custom-bitnet-small.gguf*",
    "install_script": "scripts/install.sh",
    "validate_script": "scripts/validate.sh",
    "checksums": "CHECKSUMS.txt",
    "package_info": "PACKAGE_INFO.json"
  },
  "usage": {
    "install": "./scripts/install.sh",
    "validate": "./scripts/validate.sh",
    "inference": "bitnet infer chat --model custom/bitnet-small"
  }
}
EOF
    
    echo -e "${GREEN}✅ Package info created: $info_file${NC}"
}

# Create license file
create_license() {
    echo -e "${BLUE}📄 Creating license file...${NC}"
    
    local license_file="$PACKAGE_DIR/LICENSE"
    
    # Create a basic license file (you may want to customize this)
    cat > "$license_file" << EOF
Custom BitNet Small Model License

This model distribution package is provided as part of the BitNet-Rust project.

The model files, configuration, and scripts in this package are released under
the same license terms as the BitNet-Rust project.

For the full license terms, please refer to the BitNet-Rust repository:
https://github.com/leizerowicz/bitnet-mlx.rs

This custom model is provided "as is" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and non-infringement.

Model Creation Date: $(date -u +"%Y-%m-%d")
Package Version: $MODEL_VERSION
EOF
    
    echo -e "${GREEN}✅ License file created: $license_file${NC}"
}

# Create the distribution archive
create_archive() {
    echo -e "${BLUE}📦 Creating distribution archive...${NC}"
    
    # Remove any existing distribution file
    if [ -f "$DIST_FILE" ]; then
        echo -e "${YELLOW}⚠️  Removing existing distribution file${NC}"
        rm -f "$DIST_FILE"
    fi
    
    # Create tar.gz archive
    cd "$RELEASE_DIR"
    tar -czf "../$DIST_FILE" "$MODEL_NAME"
    cd ..
    
    if [ -f "$DIST_FILE" ]; then
        local archive_size
        archive_size=$(ls -lh "$DIST_FILE" | awk '{print $5}')
        echo -e "${GREEN}✅ Distribution archive created: $DIST_FILE ($archive_size)${NC}"
        
        # Generate archive checksum
        if command -v sha256sum > /dev/null 2>&1; then
            sha256sum "$DIST_FILE" > "$DIST_FILE.sha256"
        elif command -v shasum > /dev/null 2>&1; then
            shasum -a 256 "$DIST_FILE" > "$DIST_FILE.sha256"
        fi
        
        if [ -f "$DIST_FILE.sha256" ]; then
            echo -e "${GREEN}✅ Archive checksum created: $DIST_FILE.sha256${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to create distribution archive${NC}"
        exit 1
    fi
}

# Verify the created archive
verify_archive() {
    echo -e "${BLUE}🔍 Verifying distribution archive...${NC}"
    
    # Create temporary directory for verification
    local temp_dir=$(mktemp -d)
    
    # Extract archive to temp directory
    tar -xzf "$DIST_FILE" -C "$temp_dir"
    
    # Check if extraction was successful
    if [ -d "$temp_dir/$MODEL_NAME" ]; then
        echo -e "${GREEN}✅ Archive extracts successfully${NC}"
        
        # Check key files
        local key_files=(
            "README.md"
            "config/custom-bitnet-small.json"
            "scripts/install.sh"
            "scripts/validate.sh"
            "CHECKSUMS.txt"
            "PACKAGE_INFO.json"
            "LICENSE"
        )
        
        for file in "${key_files[@]}"; do
            if [ -f "$temp_dir/$MODEL_NAME/$file" ]; then
                echo -e "${GREEN}✅ Verified: $file${NC}"
            else
                echo -e "${RED}❌ Missing in archive: $file${NC}"
            fi
        done
        
        # Test if scripts are executable
        if [ -x "$temp_dir/$MODEL_NAME/scripts/install.sh" ]; then
            echo -e "${GREEN}✅ install.sh is executable in archive${NC}"
        else
            echo -e "${RED}❌ install.sh is not executable in archive${NC}"
        fi
        
        # Cleanup
        rm -rf "$temp_dir"
        
        echo -e "${GREEN}✅ Archive verification completed${NC}"
    else
        echo -e "${RED}❌ Archive verification failed${NC}"
        rm -rf "$temp_dir"
        exit 1
    fi
}

# Display distribution summary
show_summary() {
    echo ""
    echo -e "${BLUE}📊 Distribution Summary${NC}"
    echo -e "${BLUE}====================${NC}"
    
    local archive_size
    archive_size=$(ls -lh "$DIST_FILE" | awk '{print $5}')
    
    echo -e "Package: ${YELLOW}$MODEL_NAME $MODEL_VERSION${NC}"
    echo -e "Archive: ${YELLOW}$DIST_FILE${NC}"
    echo -e "Size: ${YELLOW}$archive_size${NC}"
    
    if [ -f "$DIST_FILE.sha256" ]; then
        local checksum
        checksum=$(cut -d' ' -f1 "$DIST_FILE.sha256")
        echo -e "SHA256: ${YELLOW}$checksum${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Distribution package created successfully!${NC}"
    echo ""
    echo -e "${BLUE}Distribution Usage:${NC}"
    echo -e "1. Share: ${YELLOW}$DIST_FILE${NC}"
    echo -e "2. Install: ${YELLOW}tar -xzf $DIST_FILE && cd $MODEL_NAME && ./scripts/install.sh${NC}"
    echo -e "3. Validate: ${YELLOW}./scripts/validate.sh${NC}"
    echo ""
    
    if grep -q "placeholder" "$PACKAGE_DIR/models/"*.placeholder 2>/dev/null; then
        echo -e "${YELLOW}⚠️  NOTE: This distribution includes a placeholder model file.${NC}"
        echo -e "${YELLOW}   For production use, replace with an actual trained GGUF model.${NC}"
        echo ""
    fi
}

# Main function
main() {
    echo -e "${BLUE}Starting distribution creation process...${NC}"
    echo ""
    
    check_prerequisites
    echo ""
    
    validate_package
    echo ""
    
    generate_checksums
    echo ""
    
    create_package_info
    echo ""
    
    create_license
    echo ""
    
    create_archive
    echo ""
    
    verify_archive
    
    show_summary
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Custom BitNet Model Distribution Creation Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Clean existing distribution files before creating"
        echo "  --no-verify    Skip archive verification step"
        echo ""
        echo "This script creates a distributable package of the custom BitNet model"
        echo "including all necessary files, scripts, and documentation."
        exit 0
        ;;
    --clean)
        echo -e "${YELLOW}🧹 Cleaning existing distribution files...${NC}"
        rm -f "$DIST_FILE" "$DIST_FILE.sha256"
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        echo ""
        ;;
    --no-verify)
        echo -e "${YELLOW}⚠️  Archive verification will be skipped${NC}"
        SKIP_VERIFY=true
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

# Run main process
main