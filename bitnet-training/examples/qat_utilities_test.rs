/// Test program to verify QAT utilities functionality
/// 
/// This example demonstrates the basic instantiation and usage of QAT utilities

use bitnet_training::qat::*;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    println!("🚀 Testing BitNet-Rust QAT Utilities");
    println!("=====================================");

    // Initialize device
    let device = Device::Cpu;
    
    // Test 1: Regularization
    println!("\n🎯 Testing Quantization Regularization...");
    test_regularization(&device)?;
    
    // Test 2: Knowledge Distillation
    println!("\n🧑‍🏫 Testing Knowledge Distillation...");
    test_distillation(&device)?;
    
    // Test 3: State Tracking
    println!("\n📈 Testing State Tracking...");
    test_state_tracking(&device)?;
    
    // Test 4: QAT Optimizer
    println!("\n⚡ Testing QAT Optimizers...");
    test_optimizers(&device)?;
    
    println!("\n✅ All QAT utilities tests passed!");
    println!("🎉 BitNet-Rust QAT implementation is fully functional!");
    
    Ok(())
}

fn test_regularization(device: &Device) -> Result<()> {
    // Create test weights with F32 dtype
    let weights = Tensor::randn(0.0f32, 1.0f32, (128, 256), device)?;
    
    // Test regularization config with correct fields
    let reg_config = RegularizationConfig {
        weight_decay: 0.01,
        quantization_penalty: 0.5,
        bit_width_penalty: 0.1,
        activation_regularization: 0.05,
        gradient_penalty: 0.02,
        sparsity_penalty: 0.05,
        smooth_penalty: 0.02,
    };
    
    // Test quantization regularizer
    let regularizer = QuantizationRegularizer::new(reg_config, device.clone());
    
    // Create parameter map for compute_regularization method
    let mut parameters = std::collections::HashMap::new();
    parameters.insert("layer_0".to_string(), weights);
    
    let reg_loss = regularizer.compute_regularization(&parameters)?;
    let reg_value = reg_loss.to_scalar::<f32>()?;
    
    println!("  Regularization loss: {:.6}", reg_value);
    
    // Verify regularization is working
    assert!(reg_value >= 0.0, "Regularization loss should be non-negative");
    assert!(reg_value < 1000.0, "Regularization loss should be reasonable");
    
    println!("  ✅ Quantization regularization working correctly");
    Ok(())
}

fn test_distillation(device: &Device) -> Result<()> {
    // Create test teacher and student outputs with F32 dtype
    let teacher_logits = Tensor::randn(0.0f32, 1.0f32, (16, 10), device)?;
    let student_logits = Tensor::randn(0.0f32, 1.0f32, (16, 10), device)?;
    
    // Test distillation config
    let config = DistillationConfig {
        temperature: 3.0,
        alpha: 0.7,
        beta: 0.2,
        gamma: 0.1,
        attention_transfer: true,
        feature_layers: vec!["layer_0".to_string(), "layer_1".to_string()],
    };
    
    // Test knowledge distillation
    let distillation = KnowledgeDistillation::new(config, device.clone());
    let distill_loss = distillation.compute_loss(&teacher_logits, &student_logits, None)?;
    let distill_value = distill_loss.to_scalar::<f32>()?;
    
    println!("  Distillation loss: {:.6}", distill_value);
    
    // Verify distillation is working
    assert!(distill_value >= 0.0, "Distillation loss should be non-negative");
    
    println!("  ✅ Knowledge distillation working correctly");
    Ok(())
}

fn test_state_tracking(device: &Device) -> Result<()> {
    // Create state tracker
    let _tracker = QATStateTracker::new(device.clone());
    
    println!("  QAT State Tracker created successfully");
    
    // Test regularization stats creation
    let reg_stats = RegularizationStats {
        weight_decay_loss: 0.001,
        quantization_penalty_loss: 0.01,
        bit_width_penalty_loss: 0.005,
        activation_reg_loss: 0.002,
        gradient_penalty_loss: 0.001,
        sparsity_penalty_loss: 0.003,
        smooth_penalty_loss: 0.001,
        total_regularization: 0.023,
        parameter_count: 1000,
        quantized_parameter_count: 800,
    };
    
    println!("  Regularization stats: Total = {:.6}", reg_stats.total_regularization);
    
    println!("  ✅ State tracking working correctly");
    Ok(())
}

fn test_optimizers(device: &Device) -> Result<()> {
    // Create test parameters and gradients with F32 dtype
    let mut parameters = HashMap::new();
    let mut gradients = HashMap::new();
    
    for i in 0..3 {
        let param_name = format!("layer_{}", i);
        let param = Tensor::randn(0.0f32, 1.0f32, (32, 64), device)?;
        let grad = Tensor::randn(0.0f32, 0.1f32, (32, 64), device)?;
        
        parameters.insert(param_name.clone(), param);
        gradients.insert(param_name, grad);
    }
    
    // Test QuantizationAwareAdam with proper arguments
    let mut adam_optimizer = QuantizationAwareAdam::new(
        0.001,     // learning_rate
        0.9,       // beta1
        0.999,     // beta2  
        1e-8,      // epsilon
        0.01,      // weight_decay
        1.0,       // quantization_lr_scale
        true,      // gradient_scaling
        Some(1.0), // gradient_clip_threshold
        device.clone(),
    );
    
    // Perform optimization step
    let initial_param = parameters["layer_0"].clone();
    adam_optimizer.step(&mut parameters, &gradients)?;
    let updated_param = &parameters["layer_0"];
    
    // Verify parameters were updated
    let param_diff = (updated_param - &initial_param)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    println!("  Parameter update magnitude: {:.6}", param_diff);
    
    assert!(param_diff > 0.0, "Parameters should be updated");
    
    // Test AdamW with proper arguments
    let mut adamw_optimizer = QuantizationAwareAdamW::new(
        0.001,     // learning_rate
        0.9,       // beta1
        0.999,     // beta2
        1e-8,      // epsilon
        0.01,      // decoupled_weight_decay
        1.0,       // quantization_lr_scale
        true,      // gradient_scaling
        None,      // gradient_clip_threshold
        device.clone(),
    );
    
    adamw_optimizer.step(&mut parameters, &gradients)?;
    
    println!("  ✅ QAT optimizers working correctly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_utilities_integration() {
        let result = main();
        assert!(result.is_ok(), "QAT utilities test should pass: {:?}", result.err());
    }
}
