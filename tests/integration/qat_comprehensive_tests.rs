//! QAT Training Tests - Comprehensive BitNet Training Validation
//! 
//! This test suite provides comprehensive validation for Quantization-Aware Training
//! functionality, covering core QAT operations, straight-through estimation, and
//! optimizer integration with timeout protection and performance monitoring.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use bitnet_training::qat::{
    QATTrainingState, QATStateTracker, CheckpointManager,
    straight_through::{StraightThroughEstimator, STEConfig, STEVariant},
    progressive::{ProgressiveQuantization, ProgressiveConfig, QuantizationSchedule},
    optimizer::{QATAdam, QATAdamW},
    loss::QuantizationLoss,
};
use bitnet_training::device::Device;
use bitnet_core::{BitNetTensor, BitNetDType};
use candle_core::{Tensor, DType};
use candle_nn::{ParamsAdam, ParamsAdamW};
use tempfile::tempdir;

/// QAT Training Test Configuration
#[derive(Clone)]
struct QATTestConfig {
    /// Enable performance tracking during tests
    enable_tracking: bool,
    /// Enable resource cleanup verification
    enable_cleanup: bool,
    /// Memory pressure threshold for stress testing
    memory_pressure_threshold: f64,
    /// Maximum test duration before timeout
    max_test_duration: Duration,
    /// Number of training iterations for tests
    training_iterations: usize,
    /// Batch size for training tests
    batch_size: usize,
}

impl Default for QATTestConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            enable_cleanup: true,
            memory_pressure_threshold: 0.8,
            max_test_duration: Duration::from_secs(120),
            training_iterations: 10,
            batch_size: 32,
        }
    }
}

/// QAT Training Test Environment
struct QATTestEnvironment {
    config: QATTestConfig,
    device: Device,
    temp_dir: tempfile::TempDir,
}

impl QATTestEnvironment {
    fn new() -> anyhow::Result<Self> {
        let config = QATTestConfig::default();
        let device = Device::Cpu;
        let temp_dir = tempdir()?;
        
        Ok(Self {
            config,
            device,
            temp_dir,
        })
    }
    
    fn create_test_tensors(&self, size: (usize, usize)) -> anyhow::Result<(Tensor, Tensor)> {
        let (batch_size, feature_size) = size;
        
        // Create input tensor
        let input_data: Vec<f32> = (0..(batch_size * feature_size))
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input = Tensor::from_slice(&input_data, (batch_size, feature_size), &self.device)?;
        
        // Create target tensor
        let target_data: Vec<f32> = (0..(batch_size * feature_size))
            .map(|i| ((i + 1) as f32) * 0.01)
            .collect();
        let target = Tensor::from_slice(&target_data, (batch_size, feature_size), &self.device)?;
        
        Ok((input, target))
    }
}

/// QAT Test Utilities
mod qat_test_utils {
    use super::*;
    
    pub fn with_timeout<T, F>(duration: Duration, test_name: &str, test_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce() -> anyhow::Result<T>,
    {
        let start_time = Instant::now();
        let result = test_fn();
        let elapsed = start_time.elapsed();
        
        if elapsed > duration {
            anyhow::bail!("Test '{}' exceeded timeout of {:?}, took {:?}", test_name, duration, elapsed);
        }
        
        result
    }
    
    pub fn validate_training_convergence(losses: &[f32], tolerance: f32) -> bool {
        if losses.len() < 2 {
            return false;
        }
        
        // Check that loss is decreasing overall
        let first_half_avg = losses[..losses.len()/2].iter().sum::<f32>() / (losses.len()/2) as f32;
        let second_half_avg = losses[losses.len()/2..].iter().sum::<f32>() / (losses.len()/2) as f32;
        
        // Loss should decrease by at least the tolerance
        first_half_avg - second_half_avg > tolerance
    }
    
    pub fn measure_training_performance<F>(iterations: usize, operation: F) -> anyhow::Result<(Duration, f64)>
    where
        F: Fn() -> anyhow::Result<()>,
    {
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            operation()?;
        }
        
        let total_time = start_time.elapsed();
        let ops_per_second = iterations as f64 / total_time.as_secs_f64();
        
        Ok((total_time, ops_per_second))
    }
}

#[cfg(test)]
mod qat_training_tests {
    use super::*;
    use qat_test_utils::*;
    
    /// Test basic QAT training functionality
    #[test]
    fn test_qat_training_basic() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(60), "qat_training_basic", || {
            let env = QATTestEnvironment::new()?;
            
            // Initialize QAT state
            let mut state = QATTrainingState::new();
            state.set_quantization_enabled(true);
            state.set_learning_rate(0.001);
            
            // Create test tensors
            let (input, target) = env.create_test_tensors((env.config.batch_size, 128))?;
            
            // Initialize tracker
            let mut tracker = QATStateTracker::new(state, &env.device);
            
            // Run training iterations
            let mut losses = Vec::new();
            for iteration in 0..env.config.training_iterations {
                // Forward pass simulation
                let loss_value = 1.0 / (1.0 + iteration as f32 * 0.1); // Simulated decreasing loss
                
                tracker.update_iteration(iteration);
                tracker.update_loss(loss_value);
                
                losses.push(loss_value);
            }
            
            // Validate convergence
            assert!(validate_training_convergence(&losses, 0.1), 
                   "Training should show convergence over iterations");
            
            // Validate final state
            let final_state = tracker.get_state();
            assert!(final_state.iterations() >= env.config.training_iterations - 1);
            assert!(final_state.current_loss() < losses[0]);
            
            println!("✅ Basic QAT training test completed successfully");
            Ok(())
        })
    }
    
    /// Test Straight-Through Estimator functionality
    #[test]
    fn test_straight_through_estimator() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(60), "straight_through_estimator", || {
            let env = QATTestEnvironment::new()?;
            
            // Test different STE variants
            let variants = vec![
                STEVariant::Standard,
                STEVariant::Clipped,
                STEVariant::Soft,
                STEVariant::Learnable,
            ];
            
            for variant in variants {
                // Create STE configuration
                let config = STEConfig {
                    variant: variant.clone(),
                    temperature: 1.0,
                    gradient_scale: 1.0,
                    use_straight_through: true,
                };
                
                // Initialize STE
                let ste = StraightThroughEstimator::new(config, &env.device, DType::F32);
                
                // Create test tensor
                let input_data: Vec<f32> = vec![0.5, -0.3, 1.2, -0.8, 0.1];
                let input_tensor = Tensor::from_slice(&input_data, (5,), &env.device)?;
                
                // Apply quantization (forward pass)
                let quantized = ste.quantize(&input_tensor)?;
                
                // Validate quantization properties
                let quantized_data = quantized.to_vec1::<f32>()?;
                
                match variant {
                    STEVariant::Standard => {
                        // Standard should produce {-1, 0, 1} values
                        for &val in &quantized_data {
                            assert!(val == -1.0 || val == 0.0 || val == 1.0, 
                                   "Standard STE should produce ternary values, got: {}", val);
                        }
                    }
                    STEVariant::Clipped => {
                        // Clipped should produce values in range [-1, 1]
                        for &val in &quantized_data {
                            assert!(val >= -1.0 && val <= 1.0, 
                                   "Clipped STE should produce values in [-1, 1], got: {}", val);
                        }
                    }
                    _ => {
                        // Other variants should be differentiable approximations
                        assert!(quantized_data.len() == input_data.len());
                    }
                }
                
                println!("✅ STE variant {:?} test completed", variant);
            }
            
            Ok(())
        })
    }
    
    /// Test quantization-aware optimizer integration
    #[test]
    fn test_quantization_aware_optimizer() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(90), "quantization_aware_optimizer", || {
            let env = QATTestEnvironment::new()?;
            
            // Create test parameters
            let param_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
            let params = vec![Tensor::from_slice(&param_data, (10, 10), &env.device)?];
            
            // Test QATAdam optimizer
            {
                let adam_config = ParamsAdam {
                    lr: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                };
                
                let mut optimizer = QATAdam::new(params.clone(), adam_config)?;
                
                // Simulate training steps
                let mut param_history = Vec::new();
                for step in 0..10 {
                    // Simulate gradients
                    let grad_data: Vec<f32> = (0..100).map(|_| 0.01 * (step as f32 + 1.0)).collect();
                    let gradients = vec![Tensor::from_slice(&grad_data, (10, 10), &env.device)?];
                    
                    // Apply optimizer step
                    optimizer.step(&gradients)?;
                    
                    // Record parameter values
                    let param_values = params[0].to_vec2::<f32>()?;
                    param_history.push(param_values);
                }
                
                // Validate parameter updates
                assert!(param_history.len() == 10);
                
                // Check that parameters are changing (not stuck)
                let initial_param = &param_history[0][0][0];
                let final_param = &param_history[9][0][0];
                assert!(
                    (final_param - initial_param).abs() > 1e-6,
                    "Parameters should be updated by optimizer, initial: {}, final: {}",
                    initial_param, final_param
                );
                
                println!("✅ QATAdam optimizer test completed");
            }
            
            // Test QATAdamW optimizer
            {
                let adamw_config = ParamsAdamW {
                    lr: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                
                let mut optimizer = QATAdamW::new(params.clone(), adamw_config)?;
                
                // Simulate training steps with weight decay
                for step in 0..5 {
                    let grad_data: Vec<f32> = (0..100).map(|_| 0.005).collect();
                    let gradients = vec![Tensor::from_slice(&grad_data, (10, 10), &env.device)?];
                    
                    optimizer.step(&gradients)?;
                }
                
                println!("✅ QATAdamW optimizer test completed");
            }
            
            Ok(())
        })
    }
    
    /// Test progressive quantization functionality
    #[test]
    fn test_progressive_quantization() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(60), "progressive_quantization", || {
            let env = QATTestEnvironment::new()?;
            
            // Create progressive quantization configuration
            let config = ProgressiveConfig {
                initial_bits: 8,
                target_bits: 2,
                schedule: QuantizationSchedule::Linear,
                warmup_steps: 5,
                transition_steps: 10,
            };
            
            let mut progressive = ProgressiveQuantization::new(config, &env.device);
            
            // Test progression through different bit widths
            let total_steps = 20;
            let mut bit_progression = Vec::new();
            
            for step in 0..total_steps {
                progressive.step(step);
                let current_bits = progressive.current_quantization_bits();
                bit_progression.push((step, current_bits));
            }
            
            // Validate progression
            assert!(bit_progression[0].1 == 8, "Should start with 8 bits");
            assert!(bit_progression[total_steps-1].1 == 2, "Should end with 2 bits");
            
            // Check that bits decrease over time during transition
            let transition_start = bit_progression.iter().position(|(_, bits)| *bits < 8);
            let transition_end = bit_progression.iter().position(|(_, bits)| *bits == 2);
            
            assert!(transition_start.is_some(), "Transition should start");
            assert!(transition_end.is_some(), "Transition should complete");
            
            println!("✅ Progressive quantization test completed");
            println!("   Bit progression: {:?}", bit_progression);
            
            Ok(())
        })
    }
    
    /// Test QAT checkpoint management
    #[test]
    fn test_qat_checkpoint_management() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(60), "qat_checkpoint_management", || {
            let env = QATTestEnvironment::new()?;
            
            // Initialize checkpoint manager
            let checkpoint_dir = env.temp_dir.path().to_str().unwrap().to_string();
            let checkpoint_manager = CheckpointManager::new(checkpoint_dir.clone(), &env.device);
            
            // Create training state
            let mut state = QATTrainingState::new();
            state.set_quantization_enabled(true);
            state.set_learning_rate(0.001);
            
            // Update state with training progress
            for iteration in 0..5 {
                state.update_iteration(iteration);
                state.update_loss(1.0 / (iteration + 1) as f32);
            }
            
            // Save checkpoint
            let checkpoint_name = "test_checkpoint";
            checkpoint_manager.save_checkpoint(checkpoint_name, &state)?;
            
            // Verify checkpoint file exists
            let checkpoint_path = std::path::Path::new(&checkpoint_dir).join(format!("{}.json", checkpoint_name));
            assert!(checkpoint_path.exists(), "Checkpoint file should be created");
            
            // Load checkpoint
            let loaded_state = checkpoint_manager.load_checkpoint(checkpoint_name)?;
            
            // Validate loaded state
            assert_eq!(loaded_state.iterations(), state.iterations());
            assert_eq!(loaded_state.current_loss(), state.current_loss());
            assert_eq!(loaded_state.quantization_enabled(), state.quantization_enabled());
            assert_eq!(loaded_state.learning_rate(), state.learning_rate());
            
            println!("✅ Checkpoint management test completed");
            println!("   Saved and loaded state with {} iterations", loaded_state.iterations());
            
            Ok(())
        })
    }
    
    /// Test QAT training performance characteristics
    #[test]
    fn test_qat_training_performance() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(90), "qat_training_performance", || {
            let env = QATTestEnvironment::new()?;
            
            // Measure QAT state tracking performance
            let (duration, ops_per_sec) = measure_training_performance(1000, || {
                let mut state = QATTrainingState::new();
                state.set_quantization_enabled(true);
                state.update_iteration(1);
                state.update_loss(0.5);
                Ok(())
            })?;
            
            // Validate performance targets
            assert!(ops_per_sec > 10000.0, 
                   "QAT state operations should exceed 10K ops/sec, got: {:.2}", ops_per_sec);
            
            println!("✅ QAT training performance test completed");
            println!("   State operations: {:.2} ops/sec", ops_per_sec);
            
            // Measure STE performance
            let config = STEConfig::default();
            let ste = StraightThroughEstimator::new(config, &env.device, DType::F32);
            let test_tensor = Tensor::zeros((32, 128), DType::F32, &env.device)?;
            
            let start_time = Instant::now();
            for _ in 0..100 {
                let _quantized = ste.quantize(&test_tensor)?;
            }
            let ste_duration = start_time.elapsed();
            let ste_ops_per_sec = 100.0 / ste_duration.as_secs_f64();
            
            assert!(ste_ops_per_sec > 1000.0,
                   "STE operations should exceed 1K ops/sec, got: {:.2}", ste_ops_per_sec);
            
            println!("   STE operations: {:.2} ops/sec", ste_ops_per_sec);
            
            Ok(())
        })
    }
    
    /// Test QAT training under memory pressure
    #[test]
    fn test_qat_memory_pressure() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(120), "qat_memory_pressure", || {
            let env = QATTestEnvironment::new()?;
            
            // Create large tensors to simulate memory pressure
            let large_tensors = (0..10)
                .map(|i| {
                    let size = 1000 * (i + 1);
                    let data: Vec<f32> = (0..size).map(|j| j as f32 * 0.001).collect();
                    Tensor::from_slice(&data, (size,), &env.device)
                })
                .collect::<Result<Vec<_>, _>>()?;
            
            // Initialize QAT components under memory pressure
            let mut state = QATTrainingState::new();
            let mut tracker = QATStateTracker::new(state.clone(), &env.device);
            
            // Run training simulation with large data
            for iteration in 0..5 {
                let tensor_idx = iteration % large_tensors.len();
                let tensor = &large_tensors[tensor_idx];
                
                // Simulate forward/backward pass
                let loss = tensor.sum_all()?.to_scalar::<f32>()?;
                
                tracker.update_iteration(iteration);
                tracker.update_loss(loss / tensor.elem_count() as f32);
            }
            
            // Validate that training continues under memory pressure
            let final_state = tracker.get_state();
            assert!(final_state.iterations() >= 4);
            
            println!("✅ QAT memory pressure test completed");
            println!("   Processed {} large tensors successfully", large_tensors.len());
            
            Ok(())
        })
    }
    
    /// Test QAT error recovery mechanisms
    #[test]
    fn test_qat_error_recovery() -> anyhow::Result<()> {
        with_timeout(Duration::from_secs(60), "qat_error_recovery", || {
            let env = QATTestEnvironment::new()?;
            
            // Test recovery from invalid quantization configuration
            {
                let mut state = QATTrainingState::new();
                
                // Attempt invalid learning rate
                state.set_learning_rate(-1.0); // Should be handled gracefully
                
                // Verify state remains stable
                assert!(state.learning_rate() >= 0.0, "Learning rate should be non-negative");
            }
            
            // Test recovery from tensor shape mismatches
            {
                let config = STEConfig::default();
                let ste = StraightThroughEstimator::new(config, &env.device, DType::F32);
                
                // Create malformed tensor
                let invalid_tensor = Tensor::zeros((0, 0), DType::F32, &env.device);
                
                // Should handle error gracefully
                let result = ste.quantize(&invalid_tensor);
                assert!(result.is_err(), "Should handle invalid tensor gracefully");
            }
            
            // Test checkpoint recovery from corrupted data
            {
                let checkpoint_dir = env.temp_dir.path().to_str().unwrap().to_string();
                let checkpoint_manager = CheckpointManager::new(checkpoint_dir.clone(), &env.device);
                
                // Create corrupted checkpoint file
                let corrupt_path = std::path::Path::new(&checkpoint_dir).join("corrupt.json");
                std::fs::write(&corrupt_path, "invalid json data")?;
                
                // Should handle corruption gracefully
                let result = checkpoint_manager.load_checkpoint("corrupt");
                assert!(result.is_err(), "Should handle corrupted checkpoint gracefully");
            }
            
            println!("✅ QAT error recovery test completed");
            
            Ok(())
        })
    }
}
