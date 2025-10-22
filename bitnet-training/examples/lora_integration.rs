use anyhow::Result;
use bitnet_training::{
    BitNetLoRA, BitNetQLoRA, LoRAConfig, LoRAQuantConfig, QLoRALayer,
    CustomBitNetModel, TrainingConfig, BitNetTrainer,
};
use bitnet_training::lora::qlora;
use bitnet_core::{Device, Tensor, DType};

/// Example: Fine-tuning a BitNet model with LoRA
fn main() -> Result<()> {
    println!("🔧 BitNet LoRA Integration Example");
    
    // 1. Create LoRA configuration for BitNet
    let lora_config = LoRAConfig::for_bitnet(32); // rank=32, optimized for BitNet
    println!("✅ Created LoRA config: rank={}, alpha={}", 
             lora_config.rank, lora_config.alpha);
    
    // 2. Initialize device and model
    let device = Device::Cpu;
    let input_size = 1024;
    let output_size = 1024;
    
    // 3. Create BitNet LoRA adapter
    let mut lora_adapter = BitNetLoRA::new(
        input_size,
        output_size,
        lora_config.clone(),
        &device,
    )?;
    
    println!("✅ Created BitNet LoRA adapter: {}x{} -> rank {}", 
             input_size, output_size, lora_config.rank);
    
    // 4. Create QLoRA for memory-efficient fine-tuning
    let qlora_config = LoRAQuantConfig::memory_efficient(32);
    let qlora_adapter = BitNetQLoRA::new(
        input_size,
        output_size,
        qlora_config,
        &device,
    )?;
    
    println!("✅ Created QLoRA adapter with {:.1}% memory savings", 
             qlora_adapter.memory_savings() * 100.0);
    
    // 5. Demonstrate LoRA forward pass
    demonstrate_lora_forward_pass(&mut lora_adapter, &device)?;
    
    // 6. Show memory efficiency comparison
    demonstrate_memory_efficiency(&device)?;
    
    // 7. Integration with training pipeline
    demonstrate_training_integration(&device)?;
    
    println!("🎯 BitNet LoRA integration example completed successfully!");
    Ok(())
}

fn demonstrate_lora_forward_pass(lora: &mut BitNetLoRA, device: &Device) -> Result<()> {
    println!("\n🔄 Demonstrating LoRA Forward Pass");
    
    // Create sample input and base weights
    let batch_size = 4;
    let seq_len = 128;
    let hidden_size = 1024;
    
    let input = Tensor::zeros(&[batch_size, seq_len, hidden_size], DType::F32, device)?;
    let base_weights = Tensor::zeros(&[hidden_size, hidden_size], DType::F32, device)?;
    
    // Forward pass without LoRA
    let output_base = input.matmul(&base_weights)?;
    println!("✅ Base forward pass completed");
    
    // Activate LoRA and forward pass with LoRA
    lora.activate();
    let output_with_lora = lora.forward(&input, &base_weights)?;
    println!("✅ LoRA forward pass completed");
    
    // Show the difference
    let diff = (&output_with_lora - &output_base)?;
    println!("📊 LoRA adaptation effect measured");
    
    Ok(())
}

fn demonstrate_memory_efficiency(device: &Device) -> Result<()> {
    println!("\n💾 Memory Efficiency Comparison");
    
    let layer_sizes = vec![
        (512, 512),
        (1024, 1024), 
        (2048, 2048),
        (4096, 4096),
    ];
    
    for (input_size, output_size) in layer_sizes {
        // Calculate base model parameters
        let base_params = input_size * output_size;
        
        // LoRA parameters with different ranks
        let ranks = vec![8, 16, 32, 64];
        
        println!("📏 Layer size: {}x{} ({:.1}M params)", 
                 input_size, output_size, base_params as f32 / 1_000_000.0);
        
        for rank in ranks {
            let lora_params = rank * (input_size + output_size);
            let reduction = 100.0 * (1.0 - lora_params as f32 / base_params as f32);
            
            println!("  📉 Rank {}: {:.1}M -> {:.1}K params ({:.1}% reduction)", 
                     rank,
                     base_params as f32 / 1_000_000.0,
                     lora_params as f32 / 1_000.0,
                     reduction);
        }
    }
    
    Ok(())
}

fn demonstrate_training_integration(device: &Device) -> Result<()> {
    println!("\n🎓 Training Integration Example");
    
    // Create a sample training scenario
    let input_size = 1024;
    let output_size = 1024;
    
    // 1. Create QLoRA layer for memory-efficient training
    let base_weights = Tensor::zeros(&[input_size, output_size], DType::F32, device)?;
    let qlora_config = qlora::QLoRAConfig::memory_efficient();
    
    let mut qlora_layer = QLoRALayer::new(
        input_size,
        output_size,
        base_weights,
        qlora_config,
        device,
    )?;
    
    // 2. Add LoRA adapter
    let lora_config = LoRAConfig::for_bitnet(32);
    qlora_layer.add_lora_adapter(lora_config)?;
    qlora_layer.set_lora_enabled(true);
    
    // 3. Show memory statistics
    let stats = qlora_layer.memory_stats();
    println!("📊 Memory Statistics:");
    println!("  Base parameters: {:.1}M", stats.base_params as f32 / 1_000_000.0);
    println!("  LoRA parameters: {:.1}K", stats.lora_params as f32 / 1_000.0);
    println!("  Memory reduction: {:.1}%", (1.0 - stats.reduction_ratio) * 100.0);
    println!("  Total memory: {:.1} MB", stats.total_memory_mb);
    
    // 4. Demonstrate training step
    let sample_input = Tensor::zeros(&[4, 128, input_size], DType::F32, device)?;
    let output = qlora_layer.forward(&sample_input)?;
    println!("✅ Training forward pass completed");
    
    Ok(())
}

// Additional utility functions for LoRA training

/// Create LoRA configuration for different BitNet model sizes
pub fn create_lora_configs_for_model_sizes() -> Vec<(String, LoRAConfig)> {
    vec![
        ("small".to_string(), LoRAConfig::for_bitnet(16)),
        ("medium".to_string(), LoRAConfig::for_bitnet(32)),
        ("large".to_string(), LoRAConfig::for_bitnet(64)),
        ("xl".to_string(), LoRAConfig::for_bitnet(128)),
    ]
}

/// Estimate LoRA training time and memory requirements
pub fn estimate_lora_requirements(
    model_params: usize,
    lora_rank: usize,
    batch_size: usize,
    sequence_length: usize,
) -> (f32, f32) {
    // Simplified estimation
    let lora_params = lora_rank * 1000; // Rough estimate
    let memory_gb = (lora_params * batch_size * sequence_length * 4) as f32 / (1024.0 * 1024.0 * 1024.0);
    let training_time_hours = (model_params as f32 / 1_000_000.0) * (lora_rank as f32 / 32.0);
    
    (memory_gb, training_time_hours)
}

/// Create a complete LoRA fine-tuning configuration
pub fn create_bitnet_lora_training_config() -> (LoRAConfig, TrainingConfig) {
    let lora_config = LoRAConfig {
        rank: 32,
        alpha: 64.0,
        dropout: 0.05,
        target_modules: vec![
            "attn_q".to_string(),
            "attn_k".to_string(),
            "attn_v".to_string(),
            "attn_output".to_string(),
            "ffn_up".to_string(),
            "ffn_down".to_string(),
            "ffn_gate".to_string(),
        ],
        bias: false,
        task_type: "CAUSAL_LM".to_string(),
    };
    
    let training_config = TrainingConfig {
        learning_rate: 3e-4,
        batch_size: 4,
        num_epochs: 3,
        warmup_steps: 100,
        save_steps: 500,
        eval_steps: 250,
        max_grad_norm: 1.0,
        weight_decay: 0.01,
        gradient_accumulation_steps: 4,
        mixed_precision: true,
        gradient_checkpointing: true,
    };
    
    (lora_config, training_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lora_config_creation() {
        let configs = create_lora_configs_for_model_sizes();
        assert_eq!(configs.len(), 4);
        
        for (name, config) in configs {
            assert!(config.validate().is_ok());
            println!("✅ {} model LoRA config valid", name);
        }
    }
    
    #[test]
    fn test_memory_estimation() {
        let (memory_gb, time_hours) = estimate_lora_requirements(
            1_000_000, // 1M parameters
            32,        // rank 32
            4,         // batch size
            128,       // sequence length
        );
        
        assert!(memory_gb > 0.0);
        assert!(time_hours > 0.0);
        println!("✅ Memory: {:.2} GB, Time: {:.2} hours", memory_gb, time_hours);
    }
    
    #[test]
    fn test_training_config_creation() {
        let (lora_config, training_config) = create_bitnet_lora_training_config();
        
        assert!(lora_config.validate().is_ok());
        assert_eq!(lora_config.rank, 32);
        assert_eq!(training_config.batch_size, 4);
        
        println!("✅ BitNet LoRA training configuration created successfully");
    }
}