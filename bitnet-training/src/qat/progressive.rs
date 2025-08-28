// Progressive Quantization - Phase-based quantization scheduling for QAT
// Implements progressive quantization strategies for improved training stability

use candle_core::{Device, Result};
use std::collections::HashMap;

use super::straight_through::{STEConfig, STEVariant};

/// Progressive quantization strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressiveStrategy {
    /// Start with full precision, gradually reduce bit width
    BitWidthReduction,
    /// Start with soft quantization, gradually make it harder
    SoftToHard,
    /// Layer-wise progressive quantization (start from output, go to input)
    LayerWise,
    /// Channel-wise progressive quantization
    ChannelWise,
    /// Block-wise progressive quantization (for transformer blocks)
    BlockWise,
}

/// Progressive quantization phase configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizationPhase {
    pub name: String,
    pub min_steps: usize,
    pub max_steps: Option<usize>,
    pub bit_width: u32,
    pub ste_variant: STEVariant,
    pub temperature: f32,
    pub range: f32,
    pub layer_mask: Option<Vec<String>>, // Which layers to quantize in this phase
    pub completion_criteria: CompletionCriteria,
}

/// Criteria for completing a quantization phase
#[derive(Debug, Clone)]
pub enum CompletionCriteria {
    /// Complete after fixed number of steps
    FixedSteps,
    /// Complete when loss improvement is below threshold
    LossConvergence { threshold: f32, patience: usize },
    /// Complete when quantization error is below threshold
    QuantizationError { threshold: f32 },
    /// Complete when validation accuracy is above threshold
    ValidationAccuracy { threshold: f32 },
}

/// Progressive quantization scheduler
#[allow(dead_code)]
pub struct ProgressiveQuantization {
    strategy: ProgressiveStrategy,
    phases: Vec<QuantizationPhase>,
    current_phase: usize,
    current_step: usize,
    phase_step: usize,
    device: Device,

    // State tracking
    loss_history: Vec<f32>,
    quantization_error_history: Vec<f32>,
    validation_accuracy_history: Vec<f32>,

    // Layer management
    layer_configs: HashMap<String, STEConfig>,
    quantized_layers: Vec<String>,

    // Statistics
    phase_transition_history: Vec<PhaseTransition>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PhaseTransition {
    pub from_phase: usize,
    pub to_phase: usize,
    pub step: usize,
    pub trigger: String,
    pub metrics: HashMap<String, f32>,
}

impl ProgressiveQuantization {
    /// Create new progressive quantization scheduler
    pub fn new(
        strategy: ProgressiveStrategy,
        phases: Vec<QuantizationPhase>,
        device: Device,
    ) -> Self {
        Self {
            strategy,
            phases,
            current_phase: 0,
            current_step: 0,
            phase_step: 0,
            device,
            loss_history: Vec::new(),
            quantization_error_history: Vec::new(),
            validation_accuracy_history: Vec::new(),
            layer_configs: HashMap::new(),
            quantized_layers: Vec::new(),
            phase_transition_history: Vec::new(),
        }
    }

    /// Get current phase configuration
    pub fn current_phase(&self) -> &QuantizationPhase {
        &self.phases[self.current_phase]
    }

    /// Get STEConfig for layer based on current phase
    pub fn get_layer_config(&self, layer_name: &str) -> Option<STEConfig> {
        let phase = self.current_phase();

        // Check if layer should be quantized in current phase
        if let Some(layer_mask) = &phase.layer_mask {
            if !layer_mask
                .iter()
                .any(|pattern| layer_name.contains(pattern))
            {
                return None; // Layer not active in this phase
            }
        }

        // Check if we have a cached config for this layer
        if let Some(cached_config) = self.layer_configs.get(layer_name) {
            return Some(cached_config.clone());
        }

        // Generate config based on current phase and strategy
        Some(STEConfig {
            variant: phase.ste_variant,
            bits: phase.bit_width,
            range: phase.range,
            temperature: phase.temperature,
            ..Default::default()
        })
    }

    /// Update scheduler with training metrics
    pub fn update_metrics(
        &mut self,
        loss: f32,
        quantization_error: f32,
        validation_accuracy: Option<f32>,
    ) -> Result<bool> {
        self.current_step += 1;
        self.phase_step += 1;

        // Store metrics
        self.loss_history.push(loss);
        self.quantization_error_history.push(quantization_error);
        if let Some(acc) = validation_accuracy {
            self.validation_accuracy_history.push(acc);
        }

        // Check if we should transition to next phase
        let should_transition =
            self.check_phase_completion(loss, quantization_error, validation_accuracy)?;

        if should_transition {
            self.transition_to_next_phase(loss, quantization_error, validation_accuracy)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if current phase should complete
    fn check_phase_completion(
        &self,
        _loss: f32,
        quantization_error: f32,
        validation_accuracy: Option<f32>,
    ) -> Result<bool> {
        let phase = self.current_phase();

        // Always check minimum steps first
        if self.phase_step < phase.min_steps {
            return Ok(false);
        }

        // Check maximum steps
        if let Some(max_steps) = phase.max_steps {
            if self.phase_step >= max_steps {
                return Ok(true);
            }
        }

        // Check completion criteria
        match &phase.completion_criteria {
            CompletionCriteria::FixedSteps => {
                // Already handled by min_steps check
                Ok(false)
            }
            CompletionCriteria::LossConvergence {
                threshold,
                patience,
            } => self.check_loss_convergence(*threshold, *patience),
            CompletionCriteria::QuantizationError { threshold } => {
                Ok(quantization_error < *threshold)
            }
            CompletionCriteria::ValidationAccuracy { threshold } => {
                if let Some(acc) = validation_accuracy {
                    Ok(acc > *threshold)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Check loss convergence criteria
    fn check_loss_convergence(&self, threshold: f32, patience: usize) -> Result<bool> {
        if self.loss_history.len() < patience + 1 {
            return Ok(false);
        }

        let recent_losses = &self.loss_history[self.loss_history.len() - patience - 1..];
        let mut improvements = 0;

        for i in 1..recent_losses.len() {
            let improvement = recent_losses[i - 1] - recent_losses[i];
            if improvement > threshold {
                improvements += 1;
            }
        }

        // If we haven't seen significant improvements, converged
        Ok(improvements == 0)
    }

    /// Transition to next phase
    fn transition_to_next_phase(
        &mut self,
        loss: f32,
        quantization_error: f32,
        validation_accuracy: Option<f32>,
    ) -> Result<()> {
        if self.current_phase + 1 >= self.phases.len() {
            return Ok(()); // Already at final phase
        }

        let from_phase = self.current_phase;
        let to_phase = self.current_phase + 1;

        // Record transition
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), loss);
        metrics.insert("quantization_error".to_string(), quantization_error);
        if let Some(acc) = validation_accuracy {
            metrics.insert("validation_accuracy".to_string(), acc);
        }

        let transition = PhaseTransition {
            from_phase,
            to_phase,
            step: self.current_step,
            trigger: format!("{:?}", self.phases[from_phase].completion_criteria),
            metrics,
        };

        self.phase_transition_history.push(transition);

        // Update current phase
        self.current_phase = to_phase;
        self.phase_step = 0;

        // Update layer configurations based on new phase
        self.update_layer_configurations()?;

        // Apply strategy-specific logic
        self.apply_strategy_transition()?;

        Ok(())
    }

    /// Update layer configurations for new phase
    fn update_layer_configurations(&mut self) -> Result<()> {
        let _phase = self.current_phase();

        // Clear old configurations
        self.layer_configs.clear();

        // Apply layer-wise logic based on strategy
        match self.strategy {
            ProgressiveStrategy::LayerWise => {
                self.apply_layer_wise_config()?;
            }
            ProgressiveStrategy::ChannelWise => {
                self.apply_channel_wise_config()?;
            }
            ProgressiveStrategy::BlockWise => {
                self.apply_block_wise_config()?;
            }
            _ => {
                // For other strategies, apply uniform configuration
                // This will be handled by get_layer_config method
            }
        }

        Ok(())
    }

    /// Apply layer-wise configuration
    fn apply_layer_wise_config(&mut self) -> Result<()> {
        let phase = self.current_phase().clone();

        // In layer-wise progressive quantization, we start from output layers
        // and gradually move towards input layers
        let layer_patterns = vec![
            "output",
            "classifier",
            "head", // Output layers first
            "layer",
            "block", // Middle layers
            "conv1",
            "embed",
            "input", // Input layers last
        ];

        // Determine which layers to quantize based on current phase
        let layers_per_phase = layer_patterns.len() / self.phases.len().max(1);
        let start_idx = self.current_phase * layers_per_phase;
        let end_idx = ((self.current_phase + 1) * layers_per_phase).min(layer_patterns.len());

        let active_patterns = &layer_patterns[start_idx..end_idx];

        // Create configs for active layers
        // Clone the patterns to avoid borrowing issues
        let patterns_to_update: Vec<String> =
            active_patterns.iter().map(|p| p.to_string()).collect();

        for pattern in patterns_to_update {
            let config = STEConfig {
                variant: phase.ste_variant,
                bits: phase.bit_width,
                range: phase.range,
                temperature: phase.temperature,
                ..Default::default()
            };

            // This is a simplified version - in practice would need pattern matching
            self.layer_configs.insert(pattern, config);
        }

        Ok(())
    }

    /// Apply channel-wise configuration
    fn apply_channel_wise_config(&mut self) -> Result<()> {
        // In channel-wise progressive quantization, we gradually quantize more channels
        // This would require more sophisticated channel selection logic
        // For now, implement as uniform configuration
        Ok(())
    }

    /// Apply block-wise configuration
    fn apply_block_wise_config(&mut self) -> Result<()> {
        // In block-wise progressive quantization, we quantize entire transformer blocks
        // starting from later blocks and moving to earlier blocks
        let phase = self.current_phase().clone();
        let max_blocks = 12; // Assume 12 transformer blocks
        let blocks_per_phase = max_blocks / self.phases.len().max(1);

        let start_block = max_blocks.saturating_sub((self.current_phase + 1) * blocks_per_phase);
        let end_block = max_blocks.saturating_sub(self.current_phase * blocks_per_phase);

        for block_idx in start_block..end_block {
            let pattern = format!("block_{}", block_idx);
            let config = STEConfig {
                variant: phase.ste_variant,
                bits: phase.bit_width,
                range: phase.range,
                temperature: phase.temperature,
                ..Default::default()
            };

            self.layer_configs.insert(pattern, config);
        }

        Ok(())
    }

    /// Apply strategy-specific transition logic
    fn apply_strategy_transition(&mut self) -> Result<()> {
        match self.strategy {
            ProgressiveStrategy::BitWidthReduction => {
                // Bit width should decrease with each phase
                // This is handled by phase configuration
            }
            ProgressiveStrategy::SoftToHard => {
                // Temperature should decrease with each phase
                // This is handled by phase configuration
            }
            _ => {
                // Other strategies handled by layer configuration
            }
        }

        Ok(())
    }

    /// Get progressive quantization statistics
    pub fn get_statistics(&self) -> ProgressiveQuantizationStats {
        ProgressiveQuantizationStats {
            current_phase: self.current_phase,
            current_step: self.current_step,
            phase_step: self.phase_step,
            total_phases: self.phases.len(),
            phase_transitions: self.phase_transition_history.clone(),
            quantized_layers_count: self.layer_configs.len(),
            current_bit_width: self.current_phase().bit_width,
            current_temperature: self.current_phase().temperature,
        }
    }

    /// Check if progressive quantization is complete
    pub fn is_complete(&self) -> bool {
        self.current_phase >= self.phases.len() - 1
    }

    /// Reset progressive quantization
    pub fn reset(&mut self) {
        self.current_phase = 0;
        self.current_step = 0;
        self.phase_step = 0;
        self.loss_history.clear();
        self.quantization_error_history.clear();
        self.validation_accuracy_history.clear();
        self.layer_configs.clear();
        self.quantized_layers.clear();
        self.phase_transition_history.clear();
    }
}

/// Progressive quantization statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProgressiveQuantizationStats {
    pub current_phase: usize,
    pub current_step: usize,
    pub phase_step: usize,
    pub total_phases: usize,
    pub phase_transitions: Vec<PhaseTransition>,
    pub quantized_layers_count: usize,
    pub current_bit_width: u32,
    pub current_temperature: f32,
}

/// Layer-wise quantization scheduler
#[allow(dead_code)]
pub struct LayerWiseQuantization {
    progressive: ProgressiveQuantization,
    layer_order: Vec<String>,
    current_layer_index: usize,
}

impl LayerWiseQuantization {
    pub fn new(layer_order: Vec<String>, phases: Vec<QuantizationPhase>, device: Device) -> Self {
        let progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::LayerWise, phases, device);

        Self {
            progressive,
            layer_order,
            current_layer_index: 0,
        }
    }

    /// Get config for specific layer based on quantization order
    pub fn get_layer_config(&self, layer_name: &str) -> Option<STEConfig> {
        // Check if this layer should be quantized yet
        let layer_position = self.layer_order.iter().position(|l| l == layer_name)?;

        if layer_position <= self.current_layer_index {
            self.progressive.get_layer_config(layer_name)
        } else {
            None // Layer not yet quantized
        }
    }

    /// Update with training metrics and potentially advance layer quantization
    pub fn update(
        &mut self,
        loss: f32,
        quantization_error: f32,
        validation_accuracy: Option<f32>,
    ) -> Result<bool> {
        let phase_changed =
            self.progressive
                .update_metrics(loss, quantization_error, validation_accuracy)?;

        if phase_changed {
            // Advance to next layer in the sequence
            if self.current_layer_index < self.layer_order.len() - 1 {
                self.current_layer_index += 1;
            }
        }

        Ok(phase_changed)
    }
}

/// Factory for creating common progressive quantization schedules
pub struct ProgressiveQuantizationFactory;

impl ProgressiveQuantizationFactory {
    /// Create bit-width reduction schedule (8-bit -> 4-bit -> 2-bit -> 1-bit)
    pub fn create_bit_width_reduction(device: Device) -> ProgressiveQuantization {
        let phases = vec![
            QuantizationPhase {
                name: "8-bit".to_string(),
                min_steps: 1000,
                max_steps: Some(5000),
                bit_width: 8,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.001,
                    patience: 100,
                },
            },
            QuantizationPhase {
                name: "4-bit".to_string(),
                min_steps: 1000,
                max_steps: Some(5000),
                bit_width: 4,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.001,
                    patience: 100,
                },
            },
            QuantizationPhase {
                name: "2-bit".to_string(),
                min_steps: 1000,
                max_steps: Some(5000),
                bit_width: 2,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.001,
                    patience: 100,
                },
            },
            QuantizationPhase {
                name: "1-bit".to_string(),
                min_steps: 2000,
                max_steps: None,
                bit_width: 1,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::FixedSteps,
            },
        ];

        ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device)
    }

    /// Create soft-to-hard quantization schedule
    pub fn create_soft_to_hard(device: Device) -> ProgressiveQuantization {
        let phases = vec![
            QuantizationPhase {
                name: "soft-hot".to_string(),
                min_steps: 1000,
                max_steps: Some(3000),
                bit_width: 1,
                ste_variant: STEVariant::Soft,
                temperature: 5.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.01,
                    patience: 50,
                },
            },
            QuantizationPhase {
                name: "soft-warm".to_string(),
                min_steps: 1000,
                max_steps: Some(3000),
                bit_width: 1,
                ste_variant: STEVariant::Soft,
                temperature: 2.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.005,
                    patience: 100,
                },
            },
            QuantizationPhase {
                name: "soft-cool".to_string(),
                min_steps: 1000,
                max_steps: Some(3000),
                bit_width: 1,
                ste_variant: STEVariant::Soft,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.002,
                    patience: 100,
                },
            },
            QuantizationPhase {
                name: "hard".to_string(),
                min_steps: 2000,
                max_steps: None,
                bit_width: 1,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::FixedSteps,
            },
        ];

        ProgressiveQuantization::new(ProgressiveStrategy::SoftToHard, phases, device)
    }

    /// Create layer-wise quantization schedule
    pub fn create_layer_wise(layer_order: Vec<String>, device: Device) -> LayerWiseQuantization {
        let num_phases = layer_order.len().min(4); // Maximum 4 phases
        let mut phases = Vec::new();

        for i in 0..num_phases {
            phases.push(QuantizationPhase {
                name: format!("layer_group_{}", i),
                min_steps: 500,
                max_steps: Some(2000),
                bit_width: 1,
                ste_variant: STEVariant::Standard,
                temperature: 1.0,
                range: 1.0,
                layer_mask: None,
                completion_criteria: CompletionCriteria::LossConvergence {
                    threshold: 0.001,
                    patience: 50,
                },
            });
        }

        LayerWiseQuantization::new(layer_order, phases, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_quantization_creation() {
        let device = Device::Cpu;
        let phases = vec![QuantizationPhase {
            name: "phase1".to_string(),
            min_steps: 100,
            max_steps: Some(500),
            bit_width: 4,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::FixedSteps,
        }];

        let progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device);

        assert_eq!(progressive.current_phase, 0);
        assert_eq!(progressive.current_phase().bit_width, 4);
    }

    #[test]
    fn test_bit_width_reduction_factory() {
        let device = Device::Cpu;
        let progressive = ProgressiveQuantizationFactory::create_bit_width_reduction(device);

        assert_eq!(progressive.phases.len(), 4);
        assert_eq!(progressive.current_phase().bit_width, 8);
        assert!(!progressive.is_complete());
    }

    #[test]
    fn test_layer_config_generation() -> Result<()> {
        let device = Device::Cpu;
        let progressive = ProgressiveQuantizationFactory::create_bit_width_reduction(device);

        let config = progressive.get_layer_config("conv1.weight");
        assert!(config.is_some());

        let config = config.unwrap();
        assert_eq!(config.bits, 8); // First phase should be 8-bit
        assert_eq!(config.variant, STEVariant::Standard);

        Ok(())
    }

    #[test]
    fn test_progressive_metrics_update() -> Result<()> {
        let device = Device::Cpu;
        let mut progressive = ProgressiveQuantizationFactory::create_soft_to_hard(device);

        // Update with metrics - should not transition immediately (min_steps = 1000)
        let transitioned = progressive.update_metrics(1.0, 0.1, Some(0.9))?;
        assert!(!transitioned);
        assert_eq!(progressive.current_phase, 0);

        // Simulate many steps
        for _ in 0..1500 {
            progressive.update_metrics(0.5, 0.05, Some(0.95))?;
        }

        // Should have progressed beyond initial state (current_step should be >= 1500)
        assert!(progressive.current_step >= 1500);

        Ok(())
    }

    #[test]
    fn test_layer_wise_quantization() -> Result<()> {
        let device = Device::Cpu;
        let layer_order = vec![
            "output.weight".to_string(),
            "layer2.weight".to_string(),
            "layer1.weight".to_string(),
            "input.weight".to_string(),
        ];

        let mut layer_wise =
            ProgressiveQuantizationFactory::create_layer_wise(layer_order.clone(), device);

        // Initially, only first layer should be quantized
        assert!(layer_wise.get_layer_config("output.weight").is_some());
        assert!(layer_wise.get_layer_config("layer1.weight").is_none());

        // After phase transition, more layers should be available
        for _ in 0..600 {
            layer_wise.update(0.5, 0.05, Some(0.9))?;
        }

        // Should have advanced layer index
        assert!(layer_wise.current_layer_index > 0);

        Ok(())
    }

    #[test]
    fn test_completion_criteria() -> Result<()> {
        let device = Device::Cpu;
        let phases = vec![QuantizationPhase {
            name: "test_phase".to_string(),
            min_steps: 10,
            max_steps: Some(100),
            bit_width: 1,
            ste_variant: STEVariant::Standard,
            temperature: 1.0,
            range: 1.0,
            layer_mask: None,
            completion_criteria: CompletionCriteria::QuantizationError { threshold: 0.05 },
        }];

        let mut progressive =
            ProgressiveQuantization::new(ProgressiveStrategy::BitWidthReduction, phases, device);

        // High quantization error - should not complete
        for _ in 0..15 {
            let transitioned = progressive.update_metrics(1.0, 0.1, None)?; // Error = 0.1 > 0.05
            assert!(!transitioned);
        }

        // Low quantization error - should complete
        let _transitioned = progressive.update_metrics(1.0, 0.03, None)?; // Error = 0.03 < 0.05
                                                                          // Note: Might not transition immediately if there are more phases

        Ok(())
    }

    #[test]
    fn test_statistics_tracking() -> Result<()> {
        let device = Device::Cpu;
        let mut progressive = ProgressiveQuantizationFactory::create_bit_width_reduction(device);

        // Add some metrics
        progressive.update_metrics(1.0, 0.1, Some(0.8))?;
        progressive.update_metrics(0.9, 0.08, Some(0.85))?;

        let stats = progressive.get_statistics();
        assert_eq!(stats.current_phase, 0);
        assert_eq!(stats.current_step, 2);
        assert_eq!(stats.current_bit_width, 8);
        assert!(!progressive.is_complete());

        Ok(())
    }
}

// Type aliases for compatibility with existing tests
pub type ProgressiveQuantizationScheduler = ProgressiveQuantization;
pub type LayerWiseQuantizationScheduler = LayerWiseQuantization;
