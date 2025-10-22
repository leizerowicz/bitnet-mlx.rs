/// Tests for Advanced Fine-tuning Methods
/// 
/// These tests cover the simplified but working implementations of advanced
/// fine-tuning methods for BitNet models.

use candle_core::{Device, Tensor};
use bitnet_training::{
    finetuning::{
        BitNetDoRA, DoRAConfig, MixtureOfLoRAs, MoLoRAConfig, ContinualLearner, 
        ContinualLearningConfig, AdvancedFinetuner, AdvancedFinetuningConfig,
        ExpertSelection, validation, memory
    },
    lora::LoRAConfig,
    models::custom_bitnet::CustomBitNetModel,
    dataset::TrainingDataset,
};
use anyhow::Result;

fn create_test_device() -> Device {
    Device::Cpu
}

fn create_test_tensor(dims: &[usize], device: &Device) -> Result<Tensor> {
    let total_elements: usize = dims.iter().product();
    let data: Vec<f32> = (0..total_elements).map(|i| i as f32 * 0.01).collect();
    Ok(Tensor::from_vec(data, dims, device)?)
}

#[test]
fn test_dora_creation() -> Result<()> {
    let device = create_test_device();
    let config = DoRAConfig::default();
    
    let dora = BitNetDoRA::new(768, 768, config, &device)?;
    
    // Test basic properties
    assert_eq!(dora.device.same_device(&device), true);
    assert_eq!(dora.magnitude.dims(), &[768]);
    
    Ok(())
}

#[test]
fn test_dora_forward() -> Result<()> {
    let device = create_test_device();
    let config = DoRAConfig::default();
    let dora = BitNetDoRA::new(768, 768, config, &device)?;
    
    let input = create_test_tensor(&[2, 768], &device)?;
    let base_weights = create_test_tensor(&[768, 768], &device)?;
    
    let output = dora.forward(&input, &base_weights)?;
    assert_eq!(output.dims(), &[2, 768]);
    
    Ok(())
}

#[test]
fn test_dora_loss_computation() -> Result<()> {
    let device = create_test_device();
    let config = DoRAConfig {
        magnitude_regularization: true,
        reg_strength: 0.1,
        ..Default::default()
    };
    let dora = BitNetDoRA::new(768, 768, config, &device)?;
    
    let base_weights = create_test_tensor(&[768, 768], &device)?;
    let loss = dora.compute_dora_loss(&base_weights)?;
    
    assert_eq!(loss.dims(), &[0usize; 0]);
    
    Ok(())
}

#[test]
fn test_molora_creation() -> Result<()> {
    let device = create_test_device();
    let config = MoLoRAConfig {
        num_experts: 4,
        base_lora_config: LoRAConfig::default(),
        expert_selection: ExpertSelection::TopK { k: 2 },
        ..Default::default()
    };
    
    let molora = MixtureOfLoRAs::new(768, 768, config, &device)?;
    
    assert_eq!(molora.device.same_device(&device), true);
    assert_eq!(molora.config.num_experts, 4);
    
    Ok(())
}

#[test]
fn test_molora_forward() -> Result<()> {
    let device = create_test_device();
    let config = MoLoRAConfig::default();
    let molora = MixtureOfLoRAs::new(768, 768, config, &device)?;
    
    let input = create_test_tensor(&[2, 768], &device)?;
    let task_embedding = Some(create_test_tensor(&[2, 64], &device)?);
    
    let output = molora.forward(&input, task_embedding.as_ref())?;
    assert_eq!(output.dims(), &[2, 768]);
    
    Ok(())
}

#[test]
fn test_molora_load_balance_loss() -> Result<()> {
    let device = create_test_device();
    let config = MoLoRAConfig {
        load_balancing: true,
        balance_weight: 0.01,
        ..Default::default()
    };
    let molora = MixtureOfLoRAs::new(768, 768, config, &device)?;
    
    let expert_weights = create_test_tensor(&[2, 4], &device)?;
    let loss = molora.compute_load_balance_loss(&expert_weights)?;
    
    assert_eq!(loss.dims(), &[0usize; 0]);
    
    Ok(())
}

#[test]
fn test_continual_learner_creation() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = ContinualLearningConfig::default();
    
    let learner = ContinualLearner::new(model, config, &device);
    
    assert_eq!(learner.device.same_device(&device), true);
    assert_eq!(learner.current_task, 0);
    assert_eq!(learner.replay_buffer.max_size, 1000);
    
    Ok(())
}

#[test]
fn test_continual_learner_ewc_loss() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = ContinualLearningConfig {
        ewc_lambda: 0.4,
        ..Default::default()
    };
    
    let learner = ContinualLearner::new(model, config, &device);
    let loss = learner.compute_ewc_loss()?;
    
    assert_eq!(loss.dims(), &[0usize; 0]);
    
    Ok(())
}

#[test]
fn test_advanced_finetuner_creation() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = AdvancedFinetuningConfig::default();
    
    let finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    assert_eq!(finetuner.device.same_device(&device), true);
    assert_eq!(finetuner.current_task, 0);
    assert_eq!(finetuner.training_step, 0);
    
    Ok(())
}

#[test]
fn test_advanced_finetuner_with_dora() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = AdvancedFinetuningConfig {
        dora_config: Some(DoRAConfig::default()),
        ..Default::default()
    };
    
    let finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    assert!(finetuner.dora.is_some());
    
    Ok(())
}

#[test]
fn test_advanced_finetuner_with_molora() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = AdvancedFinetuningConfig {
        molora_config: Some(MoLoRAConfig::default()),
        ..Default::default()
    };
    
    let finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    assert!(finetuner.molora.is_some());
    
    Ok(())
}

#[test]
fn test_advanced_finetuner_forward() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = AdvancedFinetuningConfig {
        dora_config: Some(DoRAConfig::default()),
        molora_config: Some(MoLoRAConfig::default()),
        ..Default::default()
    };
    
    let finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    let input = create_test_tensor(&[2, 768], &device)?;
    let base_weights = create_test_tensor(&[768, 768], &device)?;
    let task_embedding = Some(create_test_tensor(&[2, 64], &device)?);
    
    let output = finetuner.forward(&input, &base_weights, task_embedding.as_ref())?;
    assert_eq!(output.dims(), &[2, 768]);
    
    Ok(())
}

#[test]
fn test_advanced_finetuner_loss_computation() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    let config = AdvancedFinetuningConfig {
        dora_config: Some(DoRAConfig {
            magnitude_regularization: true,
            reg_strength: 0.1,
            ..Default::default()
        }),
        molora_config: Some(MoLoRAConfig {
            load_balancing: true,
            balance_weight: 0.01,
            ..Default::default()
        }),
        ..Default::default()
    };
    
    let finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    let task_loss = create_test_tensor(&[], &device)?;
    let base_weights = create_test_tensor(&[768, 768], &device)?;
    let expert_weights = Some(create_test_tensor(&[2, 4], &device)?);
    
    let total_loss = finetuner.compute_loss(&task_loss, &base_weights, expert_weights.as_ref())?;
    assert_eq!(total_loss.dims(), &[0usize; 0]);
    
    Ok(())
}

#[test]
fn test_validation_lora_config() -> Result<()> {
    let valid_config = LoRAConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.1,
        target_modules: vec!["q_proj".to_string()],
        bias: false,
        task_type: "CAUSAL_LM".to_string(),
    };
    
    validation::validate_lora_config(&valid_config)?;
    
    let invalid_config = LoRAConfig {
        rank: 0, // Invalid
        alpha: 32.0,
        dropout: 0.1,
        target_modules: vec!["q_proj".to_string()],
        bias: false,
        task_type: "CAUSAL_LM".to_string(),
    };
    
    assert!(validation::validate_lora_config(&invalid_config).is_err());
    
    Ok(())
}

#[test]
fn test_validation_dora_config() -> Result<()> {
    let valid_config = DoRAConfig::default();
    validation::validate_dora_config(&valid_config)?;
    
    let invalid_config = DoRAConfig {
        magnitude_lr_multiplier: -1.0, // Invalid
        ..Default::default()
    };
    
    assert!(validation::validate_dora_config(&invalid_config).is_err());
    
    Ok(())
}

#[test]
fn test_validation_molora_config() -> Result<()> {
    let valid_config = MoLoRAConfig::default();
    validation::validate_molora_config(&valid_config)?;
    
    let invalid_config = MoLoRAConfig {
        num_experts: 0, // Invalid
        ..Default::default()
    };
    
    assert!(validation::validate_molora_config(&invalid_config).is_err());
    
    Ok(())
}

#[test]
fn test_memory_estimation() -> Result<()> {
    let dora_config = DoRAConfig::default();
    let dora_memory = memory::estimate_dora_memory(768, 768, &dora_config);
    assert!(dora_memory > 0);
    
    let molora_config = MoLoRAConfig::default();
    let molora_memory = memory::estimate_molora_memory(768, 768, &molora_config);
    assert!(molora_memory > dora_memory); // MoLoRA should use more memory
    
    let advanced_config = AdvancedFinetuningConfig {
        dora_config: Some(dora_config),
        molora_config: Some(molora_config),
        continual_config: Some(ContinualLearningConfig::default()),
        ..Default::default()
    };
    
    let total_memory = memory::estimate_total_memory(768, 768, &advanced_config);
    assert!(total_memory > molora_memory);
    
    Ok(())
}

#[test]
fn test_presets() -> Result<()> {
    use bitnet_training::finetuning::presets;
    
    let multi_task = presets::multi_task_config();
    assert!(multi_task.dora_config.is_some());
    assert!(multi_task.molora_config.is_some());
    
    let continual = presets::continual_learning_config();
    assert!(continual.dora_config.is_some());
    assert!(continual.continual_config.is_some());
    
    let lightweight = presets::lightweight_config();
    assert!(lightweight.dora_config.is_some());
    assert_eq!(lightweight.task_embedding_dim, 32);
    
    Ok(())
}

#[test]
fn test_integration_workflow() -> Result<()> {
    let device = create_test_device();
    let model = CustomBitNetModel::new_for_testing()?;
    
    // Use multi-task preset
    let config = bitnet_training::finetuning::presets::multi_task_config();
    let mut finetuner = AdvancedFinetuner::new(model, config, &device)?;
    
    // Create a simple task dataset (simplified)
    let dataset_config = bitnet_training::dataset::DatasetConfig::default();
    let dataset = TrainingDataset::new_simple(vec!["Hello world".to_string()], dataset_config)?;
    let task_data = bitnet_training::finetuning::TaskDataset {
        dataset,
        task_id: 1,
    };
    
    // Train on the task
    let stats = finetuner.train_task(task_data)?;
    assert_eq!(stats.task_id, 1);
    
    // Get adaptation statistics
    let advanced_stats = finetuner.get_stats()?;
    assert!(advanced_stats.adaptation_score > 0.0);
    
    Ok(())
}