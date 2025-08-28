// QAT Module - Phase 3.2 Quantization-Aware Training Infrastructure
// Builds on completed BitLinear implementation from Phase 2

pub mod autograd;
pub mod distillation;
pub mod loss;
pub mod optimizer;
pub mod progressive;
pub mod regularization;
pub mod state_tracking;
pub mod straight_through;

// Re-export core QAT functionality
pub use straight_through::{quantize_with_ste, STEConfig, STEVariant, StraightThroughEstimator};

pub use autograd::{create_quantization_function, QATAutograd, QuantizationFunction};

pub use optimizer::{
    QATAdam, QATAdamW, QATOptimizer, QATSGDWithMomentum, QuantizationAwareAdam,
    QuantizationAwareAdamW,
};

pub use loss::{
    DistillationLoss, DistillationLossComponents, ProgressiveLossComponents, QATLoss,
    QuantizationAwareLoss,
};

pub use regularization::{
    QATRegularization, QuantizationRegularizer, RegularizationConfig, RegularizationStats,
};

pub use distillation::{
    DistillationConfig, DistillationMetrics, KnowledgeDistillation, TeacherStudentPair,
};

pub use progressive::{
    CompletionCriteria, LayerWiseQuantization, LayerWiseQuantizationScheduler,
    ProgressiveQuantization, ProgressiveQuantizationScheduler, ProgressiveStrategy,
    QuantizationPhase,
};

pub use state_tracking::{CheckpointManager, QATStateTracker, QATTrainingState, TrainingMetrics};

use candle_core::{DType, Device};
use std::collections::HashMap;

/// QAT Training Configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QATConfig {
    /// STE variant to use
    pub ste_variant: STEVariant,
    /// Quantization bit width
    pub bits: u32,
    /// Enable progressive quantization
    pub progressive: bool,
    /// Knowledge distillation weight
    pub distillation_weight: f32,
    /// Temperature for knowledge distillation
    pub temperature: f32,
    /// Enable gradient scaling
    pub gradient_scaling: bool,
    /// Gradient scaling factor
    pub gradient_scale: f32,
    /// Enable quantization regularization
    pub regularization_weight: f32,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            ste_variant: STEVariant::Clipped,
            bits: 1, // BitNet default
            progressive: false,
            distillation_weight: 0.5,
            temperature: 3.0,
            gradient_scaling: true,
            gradient_scale: 1.0,
            regularization_weight: 0.01,
        }
    }
}

/// QAT Training State Tracker
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QATState {
    /// Current training step
    pub step: usize,
    /// Quantization parameters per layer
    pub quantization_params: HashMap<String, QuantizationParams>,
    /// STE statistics
    pub ste_stats: STEStatistics,
    /// Training metrics
    pub metrics: QATMetrics,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
    pub bit_width: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct STEStatistics {
    pub forward_quantization_error: f32,
    pub gradient_flow_magnitude: f32,
    pub clipping_frequency: f32,
    pub saturation_rate: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QATMetrics {
    pub quantization_loss: f32,
    pub distillation_loss: f32,
    pub regularization_loss: f32,
    pub total_loss: f32,
    pub gradient_norm: f32,
}

impl QATState {
    pub fn new() -> Self {
        Self {
            step: 0,
            quantization_params: HashMap::new(),
            ste_stats: STEStatistics {
                forward_quantization_error: 0.0,
                gradient_flow_magnitude: 0.0,
                clipping_frequency: 0.0,
                saturation_rate: 0.0,
            },
            metrics: QATMetrics {
                quantization_loss: 0.0,
                distillation_loss: 0.0,
                regularization_loss: 0.0,
                total_loss: 0.0,
                gradient_norm: 0.0,
            },
        }
    }

    /// Update state after training step
    pub fn update_step(&mut self) {
        self.step += 1;
    }

    /// Update quantization parameters for layer
    pub fn update_quantization_params(&mut self, layer_name: String, params: QuantizationParams) {
        self.quantization_params.insert(layer_name, params);
    }

    /// Update STE statistics
    pub fn update_ste_stats(&mut self, stats: STEStatistics) {
        self.ste_stats = stats;
    }

    /// Update training metrics
    pub fn update_metrics(&mut self, metrics: QATMetrics) {
        self.metrics = metrics;
    }
}

/// QAT Training Context
#[allow(dead_code)]
pub struct QATContext {
    pub config: QATConfig,
    pub state: QATState,
    pub device: Device,
    pub dtype: DType,
}

impl QATContext {
    pub fn new(config: QATConfig, device: Device) -> Self {
        Self {
            config,
            state: QATState::new(),
            device,
            dtype: DType::F32,
        }
    }

    /// Check if progressive quantization is active
    pub fn is_progressive_active(&self) -> bool {
        self.config.progressive
    }

    /// Get current quantization bit width (may vary with progressive training)
    pub fn current_bit_width(&self) -> u32 {
        if self.is_progressive_active() {
            // Progressive quantization logic would go here
            // For now, return configured bits
            self.config.bits
        } else {
            self.config.bits
        }
    }

    /// Update training context state
    pub fn update_state(&mut self, new_state: QATState) {
        self.state = new_state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_default() {
        let config = QATConfig::default();
        assert_eq!(config.bits, 1);
        assert_eq!(config.ste_variant, STEVariant::Clipped);
        assert!(!config.progressive);
    }

    #[test]
    fn test_qat_state_new() {
        let state = QATState::new();
        assert_eq!(state.step, 0);
        assert!(state.quantization_params.is_empty());
    }

    #[test]
    fn test_qat_context_creation() {
        let config = QATConfig::default();
        let device = Device::Cpu;
        let context = QATContext::new(config, device);

        assert_eq!(context.current_bit_width(), 1);
        assert!(!context.is_progressive_active());
    }
}
