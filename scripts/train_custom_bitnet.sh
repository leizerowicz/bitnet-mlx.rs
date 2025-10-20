#!/bin/bash

# BitNet Custom Model Training Script - Phase 2 Implementation
# This script demonstrates the complete training infrastructure for custom BitNet models

echo "🚀 Starting Custom BitNet Model Training - Phase 2"

# Set up environment
export RUST_LOG=info
export RUST_BACKTRACE=1

# Create necessary directories
echo "📁 Creating training directories..."
mkdir -p models/checkpoints/
mkdir -p models/custom-bitnet-small/
mkdir -p logs/
mkdir -p data/training/

# Verify data exists
if [ ! -f "data/training/synthetic_dataset.json" ]; then
    echo "❌ Training data not found. Please ensure synthetic_dataset.json exists."
    exit 1
fi

echo "✅ Training data found: data/training/synthetic_dataset.json"

# Check if model configuration exists
if [ ! -f "models/custom-bitnet-small/config.json" ]; then
    echo "📝 Creating model configuration..."
    cat > models/custom-bitnet-small/config.json << 'EOF'
{
    "vocab_size": 128256,
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_layers": 16,
    "num_heads": 16,
    "max_seq_len": 4096,
    "use_mlx": true,
    "enable_metal": true,
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-5,
    "target_device": "apple_silicon_m4"
}
EOF
    echo "✅ Model configuration created"
fi

# Run training
echo "🔥 Starting training process..."
cargo run --bin bitnet-training-demo --features=std 2>&1 | tee logs/training.log

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Training logs saved to: logs/training.log"
    echo "💾 Model checkpoints saved to: models/checkpoints/"
    
    # Display training summary
    echo ""
    echo "📈 Training Summary:"
    if [ -f "logs/training.log" ]; then
        echo "   - Final loss: $(grep -o 'avg_loss=[0-9.]*' logs/training.log | tail -1 | cut -d'=' -f2)"
        echo "   - Total epochs: $(grep -c 'Epoch.*complete' logs/training.log)"
        echo "   - Checkpoints created: $(ls -1 models/checkpoints/ | wc -l)"
    fi
    
    echo ""
    echo "🎯 Phase 2 Training Infrastructure: COMPLETE"
    echo "📋 Next steps:"
    echo "   - Validate model checkpoints"
    echo "   - Test inference with trained model"
    echo "   - Proceed to Phase 3: GGUF conversion"
else
    echo "❌ Training failed. Check logs/training.log for details."
    exit 1
fi