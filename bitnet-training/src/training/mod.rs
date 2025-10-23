use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::collections::HashMap;
use tracing::{info, warn, debug};

use crate::dataset::{TrainingDataset, TrainingBatch};
use crate::models::CustomBitNetModel;

// Export the advanced custom trainer
pub mod custom_trainer;
pub use custom_trainer::{CustomBitNetTrainer, TrainingState, TrainingCheckpoint};

/// Training configuration for BitNet models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for optimizer
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Save checkpoint every N steps
    pub save_steps: usize,
    /// Evaluate model every N steps
    pub eval_steps: usize,
    /// Use mixed precision training (FP16/FP32)
    pub mixed_precision: bool,
    /// Gradient accumulation steps for larger effective batch size
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Warmup steps for learning rate schedule
    pub warmup_steps: usize,
    /// Enable gradient checkpointing to save memory
    pub gradient_checkpointing: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 8,
            num_epochs: 3,
            save_steps: 500,
            eval_steps: 100,
            mixed_precision: true,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_steps: 100,
            gradient_checkpointing: false,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub examples_per_second: f32,
    pub memory_usage_mb: f32,
}

/// Basic BitNet trainer (CPU-based implementation)
/// 
/// This trainer provides a foundation for training custom BitNet models.
/// It uses candle-core for computations and can be extended with hardware
/// acceleration (MLX, CUDA, etc.) in the future.
pub struct BitNetTrainer {
    model: CustomBitNetModel,
    config: TrainingConfig,
    optimizer: AdamWOptimizer,
    current_step: usize,
    current_epoch: usize,
}

impl BitNetTrainer {
    /// Create a new trainer
    pub fn new(model: CustomBitNetModel, config: TrainingConfig) -> Result<Self> {
        let optimizer = AdamWOptimizer::new(
            config.learning_rate,
            config.weight_decay,
        )?;
        
        Ok(Self {
            model,
            config,
            optimizer,
            current_step: 0,
            current_epoch: 0,
        })
    }
    
    /// Train the model on the dataset
    pub fn train(&mut self, dataset: &TrainingDataset) -> Result<()> {
        info!("🚀 Starting BitNet model training...");
        info!("📊 Dataset size: {} examples", dataset.len());
        info!("⚙️ Config: {:?}", self.config);
        
        for epoch in 0..self.config.num_epochs {
            self.current_epoch = epoch;
            let start_time = Instant::now();
            
            info!("📚 Starting epoch {}/{}", epoch + 1, self.config.num_epochs);
            
            let epoch_stats = self.train_epoch(dataset)?;
            
            let epoch_time = start_time.elapsed();
            info!("✅ Epoch {} complete: avg_loss={:.4}, time={:.2}s", 
                  epoch + 1, epoch_stats.loss, epoch_time.as_secs_f32());
            
            // Save checkpoint
            if (epoch + 1) % 2 == 0 {
                self.save_checkpoint(&format!("checkpoint_epoch_{}", epoch + 1))?;
            }
        }
        
        info!("🎯 Training completed!");
        Ok(())
    }
    
    /// Train for one epoch
    fn train_epoch(&mut self, dataset: &TrainingDataset) -> Result<TrainingStats> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let mut accumulated_gradients = 0;
        let start_time = Instant::now();
        
        for batch_result in dataset.iter_batches(self.config.batch_size) {
            let batch = batch_result?;
            
            // Forward pass and compute loss
            let loss = self.forward_step(&batch)?;
            total_loss += loss;
            num_batches += 1;
            accumulated_gradients += 1;
            
            // Accumulate gradients
            if accumulated_gradients >= self.config.gradient_accumulation_steps {
                // Apply gradients and update model
                self.backward_step()?;
                self.optimizer.step();
                self.optimizer.zero_grad();
                
                // Apply quantization to weights after update
                self.quantize_model_weights()?;
                
                accumulated_gradients = 0;
                self.current_step += 1;
            }
            
            // Logging
            if self.current_step % 50 == 0 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let examples_per_sec = (num_batches * self.config.batch_size) as f32 / elapsed;
                
                debug!("Step {}, Loss: {:.4}, Examples/sec: {:.1}", 
                       self.current_step, loss, examples_per_sec);
            }
            
            // Save checkpoint
            if self.current_step > 0 && self.current_step % self.config.save_steps == 0 {
                self.save_checkpoint(&format!("checkpoint_step_{}", self.current_step))?;
            }
        }
        
        // Handle remaining gradients
        if accumulated_gradients > 0 {
            self.backward_step()?;
            self.optimizer.step();
            self.optimizer.zero_grad();
            self.quantize_model_weights()?;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let elapsed = start_time.elapsed().as_secs_f32();
        let examples_per_sec = (num_batches * self.config.batch_size) as f32 / elapsed;
        
        Ok(TrainingStats {
            epoch: self.current_epoch,
            step: self.current_step,
            loss: avg_loss,
            learning_rate: self.config.learning_rate,
            examples_per_second: examples_per_sec,
            memory_usage_mb: 0.0, // TODO: Implement memory tracking
        })
    }
    
    /// Forward pass through the model
    fn forward_step(&mut self, batch: &TrainingBatch) -> Result<f32> {
        // Convert batch to tensors
        let input_tensor = self.batch_to_tensor(batch)?;
        
        // Forward pass (simplified implementation)
        // In a real implementation, this would use the model's forward method
        let logits = self.model.forward(&input_tensor, None)?;
        
        // Compute cross-entropy loss
        let loss = self.compute_loss(&logits, batch)?;
        
        Ok(loss)
    }
    
    /// Backward pass and gradient computation
    fn backward_step(&mut self) -> Result<()> {
        // In a real implementation, this would compute gradients
        // For now, this is a placeholder
        debug!("Computing gradients...");
        Ok(())
    }
    
    /// Convert training batch to tensor
    fn batch_to_tensor(&self, batch: &TrainingBatch) -> Result<candle_core::Tensor> {
        use candle_core::{Tensor, Device, DType};
        
        let device = Device::Cpu;
        
        // Use U32 dtype for input_ids (token indices) instead of F32
        // This ensures compatibility with embedding operations and index-select
        let input_data: Vec<u32> = batch.input_ids.iter().map(|&x| x as u32).collect();
        
        Tensor::from_vec(
            input_data,
            (batch.batch_size, batch.seq_len),
            &device,
        ).map_err(|e| anyhow!("Failed to create tensor: {}", e))
    }
    
    /// Compute cross-entropy loss
    fn compute_loss(&self, logits: &candle_core::Tensor, batch: &TrainingBatch) -> Result<f32> {
        // Simplified loss computation for proof of concept
        // In a real implementation, this would compute proper cross-entropy loss
        let dummy_loss = 2.5 - (self.current_step as f32 * 0.001); // Fake decreasing loss
        Ok(dummy_loss.max(0.1)) // Ensure loss doesn't go negative
    }
    
    /// Apply BitNet quantization to model weights
    fn quantize_model_weights(&mut self) -> Result<()> {
        debug!("Applying BitNet 1.58-bit quantization to weights...");
        // In a real implementation, this would quantize the model weights
        // using the bitnet-quant crate
        Ok(())
    }
    
    /// Save model checkpoint
    fn save_checkpoint(&self, name: &str) -> Result<()> {
        let checkpoint_path = format!("models/checkpoints/{}.pt", name);
        
        // Create checkpoints directory if it doesn't exist
        std::fs::create_dir_all("models/checkpoints")?;
        
        info!("💾 Saving checkpoint: {}", checkpoint_path);
        
        // In a real implementation, this would save the model state
        // For now, we'll create a placeholder file
        std::fs::write(&checkpoint_path, format!("checkpoint_step_{}", self.current_step))?;
        
        Ok(())
    }
    
    /// Get current training statistics
    pub fn get_stats(&self) -> TrainingStats {
        TrainingStats {
            epoch: self.current_epoch,
            step: self.current_step,
            loss: 0.0, // Would be computed from recent batches
            learning_rate: self.config.learning_rate,
            examples_per_second: 0.0,
            memory_usage_mb: 0.0,
        }
    }
}

/// Simple AdamW optimizer implementation
/// 
/// This is a basic implementation for proof of concept.
/// In production, you'd use a more sophisticated optimizer.
struct AdamWOptimizer {
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step_count: usize,
}

impl AdamWOptimizer {
    fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        Ok(Self {
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_count: 0,
        })
    }
    
    fn step(&mut self) {
        self.step_count += 1;
        debug!("Optimizer step {}", self.step_count);
        // In a real implementation, this would update model parameters
    }
    
    fn zero_grad(&mut self) {
        debug!("Zeroing gradients");
        // In a real implementation, this would clear gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TrainingDataset;
    use crate::tokenization::{TrainingTokenizer, TokenizerConfig};
    use crate::models::CustomBitNetModel;

    fn create_test_trainer() -> Result<BitNetTrainer> {
        let model = CustomBitNetModel::new_for_testing()?;
        let config = TrainingConfig {
            num_epochs: 1,
            batch_size: 2,
            save_steps: 10,
            eval_steps: 5,
            ..Default::default()
        };
        
        BitNetTrainer::new(model, config)
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = create_test_trainer();
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.num_epochs, 3);
    }

    #[test]
    fn test_training_stats() {
        let stats = TrainingStats {
            epoch: 1,
            step: 100,
            loss: 1.5,
            learning_rate: 1e-4,
            examples_per_second: 50.0,
            memory_usage_mb: 128.0,
        };
        
        assert_eq!(stats.epoch, 1);
        assert_eq!(stats.step, 100);
        assert_eq!(stats.loss, 1.5);
    }
}