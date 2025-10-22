use anyhow::Result;
use candle_core::{Device, DType};
use crate::lora::{
    BitNetLoRA, BitNetQLoRA, LoRAConfig, LoRAQuantConfig, 
    QLoRALayer, qlora::QLoRAConfig, LoRAAdapter
};

#[test]
fn test_step_5_2_1_lora_integration() -> Result<()> {
    println!("🧪 Testing Step 5.2.1: LoRA Integration for BitNet");
    
    // Test 1: LoRA Configuration Creation
    let lora_config = LoRAConfig::for_bitnet(32);
    assert_eq!(lora_config.rank, 32);
    assert_eq!(lora_config.alpha, 64.0); // 32 * 2.0
    assert!(lora_config.validate().is_ok());
    println!("✅ Test 1: LoRA configuration creation - PASSED");
    
    // Test 2: BitNet LoRA Adapter Creation
    let device = Device::Cpu;
    let input_size = 512;
    let output_size = 256;
    
    let mut lora = BitNetLoRA::new(
        input_size,
        output_size,
        lora_config.clone(),
        &device,
    )?;
    
    assert!(!lora.active);
    assert_eq!(lora.config.rank, 32);
    println!("✅ Test 2: BitNet LoRA adapter creation - PASSED");
    
    // Test 3: LoRA Activation/Deactivation
    lora.activate();
    assert!(lora.has_lora());
    assert_eq!(lora.lora_rank(), Some(32));
    
    lora.deactivate();
    assert!(!lora.has_lora());
    assert_eq!(lora.lora_rank(), None);
    println!("✅ Test 3: LoRA activation/deactivation - PASSED");
    
    // Test 4: QLoRA Creation
    let qlora_config = LoRAQuantConfig::memory_efficient(16);
    let qlora = BitNetQLoRA::new(
        input_size,
        output_size,
        qlora_config,
        &device,
    )?;
    
    assert_eq!(qlora.base_lora.config.rank, 16);
    assert!(qlora.memory_savings() > 0.0);
    println!("✅ Test 4: QLoRA creation - PASSED");
    
    // Test 5: QLoRA Layer Integration
    let base_weights = candle_core::Tensor::zeros(&[input_size, output_size], DType::F32, &device)?;
    let qlora_layer_config = QLoRAConfig::memory_efficient();
    
    let mut qlora_layer = QLoRALayer::new(
        input_size,
        output_size,
        base_weights,
        qlora_layer_config,
        &device,
    )?;
    
    // Add LoRA adapter
    let layer_lora_config = LoRAConfig::for_bitnet(16);
    qlora_layer.add_lora_adapter(layer_lora_config)?;
    
    assert!(qlora_layer.lora_adapter.is_some());
    println!("✅ Test 5: QLoRA layer integration - PASSED");
    
    // Test 6: Memory Statistics
    let stats = qlora_layer.memory_stats();
    assert!(stats.base_params > 0);
    assert!(stats.lora_params > 0);
    assert!(stats.reduction_ratio < 1.0);
    assert!(stats.total_memory_mb > 0.0);
    println!("✅ Test 6: Memory statistics calculation - PASSED");
    
    // Test 7: Configuration Validation
    let mut invalid_config = LoRAConfig::for_bitnet(16);
    invalid_config.rank = 0; // Invalid rank
    assert!(invalid_config.validate().is_err());
    
    invalid_config.rank = 16;
    invalid_config.alpha = -1.0; // Invalid alpha
    assert!(invalid_config.validate().is_err());
    
    let valid_config = LoRAConfig::for_bitnet(16);
    assert!(valid_config.validate().is_ok());
    println!("✅ Test 7: Configuration validation - PASSED");
    
    println!("🎯 Step 5.2.1 LoRA Integration - ALL TESTS PASSED");
    Ok(())
}

#[test]
fn test_lora_memory_efficiency() -> Result<()> {
    println!("🧪 Testing LoRA Memory Efficiency");
    
    let layer_sizes = [(512, 512), (1024, 1024), (2048, 2048)];
    let ranks = [8, 16, 32, 64];
    
    for (input_size, output_size) in layer_sizes {
        let base_params = input_size * output_size;
        
        for rank in ranks {
            let config = LoRAConfig::for_bitnet(rank);
            let reduction = config.parameter_reduction(base_params, 32000);
            
            // LoRA should significantly reduce trainable parameters except for very high ranks
            if rank <= 32 {
                assert!(reduction > 0.5, "LoRA with rank {} should reduce parameters by at least 50%, got {:.1}%", rank, reduction * 100.0);
            }
            
            println!("  📊 {}x{}, rank {}: {:.1}% parameter reduction", 
                     input_size, output_size, rank, reduction * 100.0);
        }
    }
    
    println!("✅ LoRA memory efficiency validation - PASSED");
    Ok(())
}

#[test]
fn test_qlora_quantization_levels() -> Result<()> {
    println!("🧪 Testing QLoRA Quantization Levels");
    
    let device = Device::Cpu;
    let quantization_levels = [4, 8, 16];
    
    for bits in quantization_levels {
        let mut qlora_config = LoRAQuantConfig::default();
        qlora_config.lora_bits = bits;
        
        assert!(qlora_config.validate().is_ok());
        
        let memory_savings = qlora_config.memory_savings();
        assert!(memory_savings > 0.0 && memory_savings < 1.0);
        
        println!("  🎯 {}-bit quantization: {:.1}% memory savings", 
                 bits, memory_savings * 100.0);
    }
    
    // Test invalid quantization
    let mut invalid_config = LoRAQuantConfig::default();
    invalid_config.lora_bits = 3; // Invalid
    assert!(invalid_config.validate().is_err());
    
    println!("✅ QLoRA quantization levels validation - PASSED");
    Ok(())
}

#[test]
fn test_lora_target_modules() -> Result<()> {
    println!("🧪 Testing LoRA Target Modules");
    
    let config = LoRAConfig::for_bitnet(32);
    let expected_modules = vec![
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_up", "ffn_down", "ffn_gate"
    ];
    
    for module in expected_modules {
        assert!(config.target_modules.contains(&module.to_string()),
                "BitNet LoRA config should target {}", module);
    }
    
    println!("✅ LoRA target modules validation - PASSED");
    Ok(())
}

// Integration test simulating real training workflow
#[test]
fn test_lora_training_workflow_simulation() -> Result<()> {
    println!("🧪 Testing LoRA Training Workflow Simulation");
    
    let device = Device::Cpu;
    
    // 1. Setup model layer
    let input_size = 768;
    let output_size = 768;
    let base_weights = candle_core::Tensor::ones(&[input_size, output_size], DType::F32, &device)?;
    
    // 2. Create QLoRA layer
    let qlora_config = QLoRAConfig::memory_efficient();
    let mut layer = QLoRALayer::new(
        input_size,
        output_size,
        base_weights,
        qlora_config,
        &device,
    )?;
    
    // 3. Add LoRA adapter
    let lora_config = LoRAConfig::for_bitnet(32);
    layer.add_lora_adapter(lora_config)?;
    layer.set_lora_enabled(true);
    
    // 4. Simulate training input
    let batch_size = 2;
    let seq_len = 64;
    let input = candle_core::Tensor::ones(&[batch_size * seq_len, input_size], DType::F32, &device)?;
    
    // 5. Forward pass
    let output = layer.forward(&input)?;
    
    // 6. Verify output shape
    let expected_shape = vec![batch_size * seq_len, output_size];
    assert_eq!(output.dims(), expected_shape);
    
    // 7. Check memory stats
    let stats = layer.memory_stats();
    assert!(stats.reduction_ratio < 0.1, "LoRA should drastically reduce trainable parameters");
    
    println!("  📊 Training simulation results:");
    println!("    - Parameters reduced by: {:.2}%", (1.0 - stats.reduction_ratio) * 100.0);
    println!("    - Memory usage: {:.2} MB", stats.total_memory_mb);
    println!("    - Output shape: {:?}", output.dims());
    
    println!("✅ LoRA training workflow simulation - PASSED");
    Ok(())
}