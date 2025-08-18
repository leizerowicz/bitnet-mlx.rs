// QAT State Tracking - Training state management for quantization-aware training
// Tracks training progress, quantization metrics, and provides checkpointing

use candle_core::Device;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::path::Path;

use super::{
    straight_through::STEStatistics,
    regularization::RegularizationStats,
};

/// Comprehensive QAT training state
#[derive(Debug, Clone)]
pub struct QATTrainingState {
    // Basic training info
    pub epoch: usize,
    pub step: usize,
    pub learning_rate: f32,
    
    // Loss tracking
    pub current_loss: f32,
    pub loss_history: Vec<f32>,
    pub validation_loss: Option<f32>,
    pub validation_history: Vec<f32>,
    
    // Quantization metrics
    pub quantization_error: f32,
    pub ste_statistics: HashMap<String, STEStatistics>,
    
    // Regularization state
    pub regularization_stats: Option<RegularizationStats>,
    
    // Training performance
    pub training_time: f64, // seconds
    pub samples_processed: usize,
    pub throughput: f32, // samples per second
    
    // Model performance
    pub validation_accuracy: Option<f32>,
    pub quantized_model_size: Option<usize>, // bytes
    pub compression_ratio: Option<f32>,
    
    // Checkpointing info
    pub last_checkpoint_step: usize,
    pub checkpoint_path: Option<String>,
    pub timestamp: u64,
}

impl QATTrainingState {
    pub fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            epoch: 0,
            step: 0,
            learning_rate: 0.0,
            current_loss: 0.0,
            loss_history: Vec::new(),
            validation_loss: None,
            validation_history: Vec::new(),
            quantization_error: 0.0,
            ste_statistics: HashMap::new(),
            regularization_stats: None,
            training_time: 0.0,
            samples_processed: 0,
            throughput: 0.0,
            validation_accuracy: None,
            quantized_model_size: None,
            compression_ratio: None,
            last_checkpoint_step: 0,
            checkpoint_path: None,
            timestamp,
        }
    }

    /// Update basic training metrics
    pub fn update_training_metrics(
        &mut self,
        epoch: usize,
        step: usize,
        learning_rate: f32,
        loss: f32,
        samples_this_step: usize,
        step_time: f64,
    ) {
        self.epoch = epoch;
        self.step = step;
        self.learning_rate = learning_rate;
        self.current_loss = loss;
        self.loss_history.push(loss);
        self.samples_processed += samples_this_step;
        self.training_time += step_time;
        
        // Update throughput
        if self.training_time > 0.0 {
            self.throughput = self.samples_processed as f32 / self.training_time as f32;
        }
    }

    /// Update validation metrics
    pub fn update_validation_metrics(&mut self, loss: f32, accuracy: Option<f32>) {
        self.validation_loss = Some(loss);
        self.validation_history.push(loss);
        self.validation_accuracy = accuracy;
    }

    /// Update quantization statistics
    pub fn update_quantization_stats(&mut self, stats: HashMap<String, STEStatistics>) {
        // Calculate average quantization error
        let total_error: f32 = stats.values().map(|s| s.quantization_error).sum();
        self.quantization_error = if !stats.is_empty() {
            total_error / stats.len() as f32
        } else {
            0.0
        };
        
        self.ste_statistics = stats;
    }

    /// Check if checkpointing is needed
    pub fn should_checkpoint(&self, checkpoint_frequency: usize) -> bool {
        self.step > 0 && (self.step % checkpoint_frequency == 0)
    }

    /// Get training summary
    pub fn get_summary(&self) -> TrainingMetrics {
        TrainingMetrics {
            epoch: self.epoch,
            step: self.step,
            current_loss: self.current_loss,
            best_validation_loss: self.validation_history.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied(),
            average_quantization_error: self.quantization_error,
            training_time: self.training_time,
            throughput: self.throughput,
            samples_processed: self.samples_processed,
            validation_accuracy: self.validation_accuracy,
            compression_ratio: self.compression_ratio,
        }
    }
}

/// Training metrics summary
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub current_loss: f32,
    pub best_validation_loss: Option<f32>,
    pub average_quantization_error: f32,
    pub training_time: f64,
    pub throughput: f32,
    pub samples_processed: usize,
    pub validation_accuracy: Option<f32>,
    pub compression_ratio: Option<f32>,
}

/// QAT State Tracker for managing training state
pub struct QATStateTracker {
    state: QATTrainingState,
    device: Device,
    
    // Configuration
    checkpoint_frequency: usize,
    max_history_length: usize,
    
    // Statistics tracking
    layer_statistics: HashMap<String, Vec<f32>>,
    performance_history: Vec<TrainingMetrics>,
}

impl QATStateTracker {
    pub fn new(device: Device) -> Self {
        Self {
            state: QATTrainingState::new(),
            device,
            checkpoint_frequency: 1000,
            max_history_length: 10000,
            layer_statistics: HashMap::new(),
            performance_history: Vec::new(),
        }
    }

    /// Update training state
    pub fn update(&mut self, 
        epoch: usize, 
        step: usize, 
        learning_rate: f32, 
        loss: f32,
        samples: usize,
        step_time: f64
    ) {
        self.state.update_training_metrics(epoch, step, learning_rate, loss, samples, step_time);
        
        // Trim history if too long
        if self.state.loss_history.len() > self.max_history_length {
            self.state.loss_history.truncate(self.max_history_length / 2);
        }
        
        // Update performance history
        let metrics = self.state.get_summary();
        self.performance_history.push(metrics);
        
        // Trim performance history
        if self.performance_history.len() > 1000 {
            self.performance_history.truncate(500);
        }
    }

    /// Update with validation results
    pub fn update_validation(&mut self, validation_loss: f32, accuracy: Option<f32>) {
        self.state.update_validation_metrics(validation_loss, accuracy);
    }

    /// Update quantization statistics
    pub fn update_quantization(&mut self, stats: HashMap<String, STEStatistics>) {
        self.state.update_quantization_stats(stats);
    }

    /// Update regularization statistics
    pub fn update_regularization(&mut self, stats: RegularizationStats) {
        self.state.regularization_stats = Some(stats);
    }

    /// Get current state
    pub fn get_state(&self) -> &QATTrainingState {
        &self.state
    }

    /// Get mutable state
    pub fn get_state_mut(&mut self) -> &mut QATTrainingState {
        &mut self.state
    }

    /// Check if should checkpoint
    pub fn should_checkpoint(&self) -> bool {
        self.state.should_checkpoint(self.checkpoint_frequency)
    }

    /// Set checkpoint frequency
    pub fn set_checkpoint_frequency(&mut self, frequency: usize) {
        self.checkpoint_frequency = frequency;
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state = QATTrainingState::new();
        self.layer_statistics.clear();
        self.performance_history.clear();
    }
}

/// Checkpoint manager for saving/loading training state
pub struct CheckpointManager {
    checkpoint_dir: String,
    device: Device,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: String, device: Device) -> Self {
        Self {
            checkpoint_dir,
            device,
        }
    }

    /// Save training state (simplified - in practice would use proper serialization)
    pub fn save_state(&self, state: &QATTrainingState, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(&self.checkpoint_dir).join(filename);
        
        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // In a real implementation, we would serialize the state to JSON/bincode
        let state_json = format!(
            r#"{{
                "epoch": {},
                "step": {},
                "learning_rate": {},
                "current_loss": {},
                "training_time": {},
                "samples_processed": {},
                "timestamp": {}
            }}"#,
            state.epoch,
            state.step,
            state.learning_rate,
            state.current_loss,
            state.training_time,
            state.samples_processed,
            state.timestamp
        );

        std::fs::write(path, state_json)?;
        Ok(())
    }

    /// Load training state (simplified)
    pub fn load_state(&self, filename: &str) -> Result<QATTrainingState, Box<dyn std::error::Error>> {
        let path = Path::new(&self.checkpoint_dir).join(filename);
        let _content = std::fs::read_to_string(path)?;
        
        // In a real implementation, we would deserialize from JSON/bincode
        // For now, return a new state
        Ok(QATTrainingState::new())
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let dir = std::fs::read_dir(&self.checkpoint_dir)?;
        let mut checkpoints = Vec::new();
        
        for entry in dir {
            let entry = entry?;
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".json") {
                    checkpoints.push(filename.to_string());
                }
            }
        }
        
        checkpoints.sort();
        Ok(checkpoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_qat_state_tracker() {
        let device = Device::Cpu;
        let mut tracker = QATStateTracker::new(device);

        // Update training metrics
        tracker.update(1, 100, 0.001, 0.5, 32, 0.1);

        let state = tracker.get_state();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 100);
        assert_eq!(state.learning_rate, 0.001);
        assert_eq!(state.current_loss, 0.5);
        assert_eq!(state.samples_processed, 32);
    }

    #[test]
    fn test_checkpoint_manager() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let temp_dir = std::env::temp_dir().join("bitnet_test_checkpoints");
        let manager = CheckpointManager::new(temp_dir.to_string_lossy().to_string(), device);

        let state = QATTrainingState::new();
        manager.save_state(&state, "test_checkpoint.json")?;

        let _loaded_state = manager.load_state("test_checkpoint.json")?;

        Ok(())
    }
}
