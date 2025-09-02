// QAT Regularization - Quantization-specific regularization terms for training
// Implements various regularization techniques to improve quantized model training

use candle_core::{Result, Tensor, Device, DType};
use std::collections::HashMap;

use super::straight_through::STEStatistics;

/// QAT Regularization trait for different regularization types
pub trait QATRegularization {
    fn compute_regularization(&self, parameters: &HashMap<String, Tensor>) -> Result<Tensor>;
    fn update_statistics(&mut self, stats: &HashMap<String, STEStatistics>);
    fn get_weight(&self) -> f32;
    fn set_weight(&mut self, weight: f32);
    fn get_name(&self) -> &str;
}

/// Configuration for regularization terms
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    pub weight_decay: f32,
    pub quantization_penalty: f32,
    pub bit_width_penalty: f32,
    pub activation_regularization: f32,
    pub gradient_penalty: f32,
    pub sparsity_penalty: f32,
    pub smooth_penalty: f32,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            weight_decay: 0.01,
            quantization_penalty: 0.1,
            bit_width_penalty: 0.05,
            activation_regularization: 0.01,
            gradient_penalty: 0.01,
            sparsity_penalty: 0.0,
            smooth_penalty: 0.01,
        }
    }
}

/// Statistics tracking for regularization
#[derive(Debug, Clone)]
pub struct RegularizationStats {
    pub weight_decay_loss: f32,
    pub quantization_penalty_loss: f32,
    pub bit_width_penalty_loss: f32,
    pub activation_reg_loss: f32,
    pub gradient_penalty_loss: f32,
    pub sparsity_penalty_loss: f32,
    pub smooth_penalty_loss: f32,
    pub total_regularization: f32,
    pub parameter_count: usize,
    pub quantized_parameter_count: usize,
}

impl RegularizationStats {
    pub fn new() -> Self {
        Self {
            weight_decay_loss: 0.0,
            quantization_penalty_loss: 0.0,
            bit_width_penalty_loss: 0.0,
            activation_reg_loss: 0.0,
            gradient_penalty_loss: 0.0,
            sparsity_penalty_loss: 0.0,
            smooth_penalty_loss: 0.0,
            total_regularization: 0.0,
            parameter_count: 0,
            quantized_parameter_count: 0,
        }
    }
}

/// Comprehensive quantization regularizer
pub struct QuantizationRegularizer {
    config: RegularizationConfig,
    device: Device,
    stats: RegularizationStats,
    ste_statistics: HashMap<String, STEStatistics>,
    name: String,
}

impl QuantizationRegularizer {
    pub fn new(config: RegularizationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            stats: RegularizationStats::new(),
            ste_statistics: HashMap::new(),
            name: "QuantizationRegularizer".to_string(),
        }
    }

    /// Compute L2 weight decay regularization
    pub fn compute_weight_decay(&self, parameters: &HashMap<String, Tensor>) -> Result<Tensor> {
        if parameters.is_empty() || self.config.weight_decay == 0.0 {
            return Tensor::zeros((), DType::F32, &self.device);
        }

        let mut total_l2 = Tensor::zeros((), DType::F32, &self.device)?;

        for (_name, param) in parameters {
            let l2_norm = param.sqr()?.sum_all()?;
            total_l2 = total_l2.broadcast_add(&l2_norm)?;
        }

        let weight_tensor = Tensor::new(self.config.weight_decay, &self.device)?;
        total_l2.broadcast_mul(&weight_tensor)
    }

    /// Compute quantization penalty - penalizes weights far from quantization levels
    pub fn compute_quantization_penalty(&self, parameters: &HashMap<String, Tensor>) -> Result<Tensor> {
        if parameters.is_empty() || self.config.quantization_penalty == 0.0 {
            return Tensor::zeros((), DType::F32, &self.device);
        }

        let mut total_penalty = Tensor::zeros((), DType::F32, &self.device)?;

        for (_name, param) in parameters {
            // For BitNet, quantization levels are typically [-1, 0, 1]
            let penalty = self.compute_distance_to_quantization_levels(param)?;
            total_penalty = total_penalty.broadcast_add(&penalty)?;
        }

        let weight_tensor = Tensor::new(self.config.quantization_penalty, &self.device)?;
        total_penalty.broadcast_mul(&weight_tensor)
    }

    /// Compute distance to nearest quantization level
    fn compute_distance_to_quantization_levels(&self, weights: &Tensor) -> Result<Tensor> {
        // BitNet quantization levels: -1, 0, 1
        let abs_weights = weights.abs()?;
        let ones = Tensor::ones_like(weights)?;

        // Distance to 0
        let dist_to_zero = abs_weights.clone();

        // Distance to 1 or -1 (minimum distance)
        let dist_to_one = (abs_weights - &ones)?.abs()?;

        // Use minimum distance
        let min_distance = dist_to_zero.minimum(&dist_to_one)?;

        // Square the distances and sum
        min_distance.sqr()?.mean_all()
    }

    /// Get current regularization statistics
    pub fn get_stats(&self) -> &RegularizationStats {
        &self.stats
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RegularizationConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RegularizationConfig {
        &self.config
    }
}

impl QATRegularization for QuantizationRegularizer {
    fn compute_regularization(&self, parameters: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut total_reg = Tensor::zeros((), DType::F32, &self.device)?;

        // Compute individual regularization terms
        let weight_decay = self.compute_weight_decay(parameters)?;
        let quant_penalty = self.compute_quantization_penalty(parameters)?;

        // Sum regularization terms
        total_reg = total_reg.broadcast_add(&weight_decay)?;
        total_reg = total_reg.broadcast_add(&quant_penalty)?;

        Ok(total_reg)
    }

    fn update_statistics(&mut self, stats: &HashMap<String, STEStatistics>) {
        self.ste_statistics = stats.clone();
    }

    fn get_weight(&self) -> f32 {
        // Return the maximum weight as representative
        self.config.weight_decay.max(self.config.quantization_penalty)
    }

    fn set_weight(&mut self, weight: f32) {
        // Scale all weights proportionally
        let scale_factor = weight / self.get_weight().max(1e-8);
        self.config.weight_decay *= scale_factor;
        self.config.quantization_penalty *= scale_factor;
        self.config.bit_width_penalty *= scale_factor;
        self.config.activation_regularization *= scale_factor;
        self.config.gradient_penalty *= scale_factor;
        self.config.sparsity_penalty *= scale_factor;
        self.config.smooth_penalty *= scale_factor;
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantization_regularizer() -> Result<()> {
        let device = Device::Cpu;
        let config = RegularizationConfig::default();
        let regularizer = QuantizationRegularizer::new(config, device);

        let mut parameters = HashMap::new();
        parameters.insert(
            "test_param".to_string(),
            Tensor::randn(0f32, 1f32, (10, 10), &Device::Cpu)?,
        );

        let reg_loss = regularizer.compute_regularization(&parameters)?;
        assert!(reg_loss.dims().is_empty()); // Should be scalar

        Ok(())
    }

    #[test]
    fn test_quantization_penalty() -> Result<()> {
        let device = Device::Cpu;
        let config = RegularizationConfig::default();
        let regularizer = QuantizationRegularizer::new(config, device);

        let mut parameters = HashMap::new();
        // Create weights close to quantization levels
        parameters.insert(
            "quantized_weights".to_string(),
            Tensor::new(&[1.0f32, 0.0, -1.0, 0.9, -0.9], &Device::Cpu)?,
        );

        let penalty = regularizer.compute_quantization_penalty(&parameters)?;
        assert!(penalty.dims().is_empty());

        Ok(())
    }
}
