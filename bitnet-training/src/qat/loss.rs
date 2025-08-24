// QAT Loss Functions for Quantization-Aware Training
// Implements specialized loss functions for BitNet quantized training

use candle_core::{DType, Device, Result, Tensor};
use candle_nn;

use super::straight_through::STEStatistics;

/// QAT-specific loss function trait
pub trait QATLoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
    fn get_name(&self) -> &str;
}

/// Quantization-aware loss that includes quantization regularization
pub struct QuantizationAwareLoss {
    base_loss: BaseLossType,
    regularization_weight: f32,
    quantization_penalty_weight: f32,
    device: Device,
    name: String,
}

#[derive(Debug, Clone)]
pub enum BaseLossType {
    CrossEntropy,
    MeanSquaredError,
    L1Loss,
    SmoothL1Loss,
}

impl QuantizationAwareLoss {
    pub fn new(
        base_loss: BaseLossType,
        regularization_weight: f32,
        quantization_penalty_weight: f32,
        device: Device,
    ) -> Self {
        let name = format!("QuantizationAware{:?}", base_loss);
        Self {
            base_loss,
            regularization_weight,
            quantization_penalty_weight,
            device,
            name,
        }
    }

    /// Compute base loss (cross-entropy, MSE, etc.)
    fn compute_base_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        match self.base_loss {
            BaseLossType::CrossEntropy => self.cross_entropy_loss(predictions, targets),
            BaseLossType::MeanSquaredError => self.mse_loss(predictions, targets),
            BaseLossType::L1Loss => self.l1_loss(predictions, targets),
            BaseLossType::SmoothL1Loss => self.smooth_l1_loss(predictions, targets),
        }
    }

    fn cross_entropy_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Log softmax for numerical stability
        let log_probs = predictions.log()?; // Simple approximation

        // Convert targets to one-hot if needed or use sparse cross-entropy
        let loss = log_probs.neg()?.gather(targets, candle_core::D::Minus1)?;
        loss.mean_all()
    }

    fn mse_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = (predictions - targets)?;
        let squared_diff = diff.sqr()?;
        squared_diff.mean_all()
    }

    fn l1_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = (predictions - targets)?;
        let abs_diff = diff.abs()?;
        abs_diff.mean_all()
    }

    fn smooth_l1_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = (predictions - targets)?;
        let abs_diff = diff.abs()?;

        // Smooth L1: 0.5 * x^2 if |x| < 1, |x| - 0.5 otherwise
        let threshold = Tensor::ones_like(&abs_diff)?;
        let mask = abs_diff.lt(&threshold)?;

        let smooth_part = (diff.sqr()? * 0.5)?;
        let linear_part = (abs_diff - 0.5)?;

        let loss = smooth_part.where_cond(&mask, &linear_part)?;
        loss.mean_all()
    }

    /// Compute quantization regularization term
    pub fn compute_quantization_regularization(
        &self,
        quantized_params: &[&Tensor],
    ) -> Result<Tensor> {
        if quantized_params.is_empty() {
            return Tensor::zeros((), DType::F32, &self.device);
        }

        let mut reg_loss = Tensor::zeros((), DType::F32, &self.device)?;

        for param in quantized_params {
            // L2 regularization on quantized parameters
            let param_l2 = param.sqr()?.sum_all()?;
            reg_loss = (reg_loss + param_l2)?;
        }

        let weight_tensor = Tensor::new(self.regularization_weight, reg_loss.device())?;
        Ok(reg_loss.broadcast_mul(&weight_tensor)?)
    }

    /// Compute quantization penalty based on quantization error
    pub fn compute_quantization_penalty(&self, ste_stats: &[STEStatistics]) -> Result<Tensor> {
        if ste_stats.is_empty() {
            return Tensor::zeros((), DType::F32, &self.device);
        }

        let avg_quantization_error: f32 = ste_stats
            .iter()
            .map(|stats| stats.quantization_error)
            .sum::<f32>()
            / ste_stats.len() as f32;

        let penalty = Tensor::from_slice(&[avg_quantization_error], (), &self.device)?;
        Ok(penalty.affine(self.quantization_penalty_weight as f64, 0.0)?)
    }
}

impl QATLoss for QuantizationAwareLoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        self.compute_base_loss(predictions, targets)
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Knowledge Distillation Loss for QAT
pub struct DistillationLoss {
    temperature: f32,
    alpha: f32, // Weight for distillation loss
    beta: f32,  // Weight for student loss
    base_loss: QuantizationAwareLoss,
    device: Device,
}

impl DistillationLoss {
    pub fn new(
        temperature: f32,
        alpha: f32,
        beta: f32,
        base_loss: QuantizationAwareLoss,
        device: Device,
    ) -> Self {
        Self {
            temperature,
            alpha,
            beta,
            base_loss,
            device,
        }
    }

    /// Compute knowledge distillation loss between teacher and student
    pub fn compute_distillation_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // Compute soft targets from teacher
        let temp_tensor = Tensor::from_slice(&[self.temperature as f64], (), &self.device)?;
        let teacher_scaled = teacher_logits.broadcast_div(&temp_tensor)?;
        let student_scaled = student_logits.broadcast_div(&temp_tensor)?;

        // Apply softmax and log_softmax (using available candle operations)
        let teacher_soft = candle_nn::ops::softmax_last_dim(&teacher_scaled)?;
        let student_log_soft =
            candle_nn::ops::log_softmax(&student_scaled, candle_core::D::Minus1)?;

        // KL divergence between teacher and student
        let kl_loss = self.compute_kl_divergence(&teacher_soft, &student_log_soft)?;
        let temp_sq = self.temperature * self.temperature;
        let scaled_kl = kl_loss.affine(temp_sq as f64, 0.0)?; // Temperature scaling

        // Standard loss on hard targets
        let hard_loss = self.base_loss.compute_loss(student_logits, targets)?;

        // Combine losses
        let alpha_scaled = scaled_kl.affine(self.alpha as f64, 0.0)?;
        let beta_scaled = hard_loss.affine(self.beta as f64, 0.0)?;
        let total_loss = (alpha_scaled + beta_scaled)?;
        Ok(total_loss)
    }

    fn compute_kl_divergence(
        &self,
        teacher_probs: &Tensor,
        student_log_probs: &Tensor,
    ) -> Result<Tensor> {
        let kl = (teacher_probs * (teacher_probs.log()? - student_log_probs)?)?;
        kl.sum(candle_core::D::Minus1)?.mean_all()
    }

    /// Compute complete distillation loss with quantization penalties
    pub fn compute_complete_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
        quantized_params: &[&Tensor],
        ste_stats: &[STEStatistics],
    ) -> Result<(Tensor, DistillationLossComponents)> {
        // Main distillation loss
        let distillation_loss =
            self.compute_distillation_loss(student_logits, teacher_logits, targets)?;

        // Quantization regularization
        let reg_loss = self
            .base_loss
            .compute_quantization_regularization(quantized_params)?;

        // Quantization penalty
        let penalty_loss = self.base_loss.compute_quantization_penalty(ste_stats)?;

        // Store scalar values before consuming tensors
        let distillation_scalar = distillation_loss.to_scalar::<f32>().unwrap_or(0.0);
        let reg_scalar = reg_loss.to_scalar::<f32>().unwrap_or(0.0);
        let penalty_scalar = penalty_loss.to_scalar::<f32>().unwrap_or(0.0);

        let combined_base = (distillation_loss + reg_loss)?;
        let total_loss = (combined_base + penalty_loss)?;
        let total_scalar = total_loss.to_scalar::<f32>().unwrap_or(0.0);

        let components = DistillationLossComponents {
            distillation_loss: distillation_scalar,
            regularization_loss: reg_scalar,
            quantization_penalty: penalty_scalar,
            total_loss: total_scalar,
        };

        Ok((total_loss, components))
    }
}

impl QATLoss for DistillationLoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // For standard loss computation, use base loss
        self.base_loss.compute_loss(predictions, targets)
    }

    fn get_name(&self) -> &str {
        "DistillationLoss"
    }
}

#[derive(Debug, Clone)]
pub struct DistillationLossComponents {
    pub distillation_loss: f32,
    pub regularization_loss: f32,
    pub quantization_penalty: f32,
    pub total_loss: f32,
}

/// Progressive Quantization Loss - adjusts loss based on training phase
pub struct ProgressiveQuantizationLoss {
    base_loss: QuantizationAwareLoss,
    current_phase: usize,
    phase_configs: Vec<PhaseConfig>,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct PhaseConfig {
    pub quantization_weight: f32,
    pub regularization_weight: f32,
    pub min_steps: usize,
    pub name: String,
}

impl ProgressiveQuantizationLoss {
    pub fn new(
        base_loss: QuantizationAwareLoss,
        phase_configs: Vec<PhaseConfig>,
        device: Device,
    ) -> Self {
        Self {
            base_loss,
            current_phase: 0,
            phase_configs,
            device,
        }
    }

    /// Update training phase
    pub fn update_phase(&mut self, step: usize) {
        let mut accumulated_steps = 0;
        for (phase_idx, config) in self.phase_configs.iter().enumerate() {
            accumulated_steps += config.min_steps;
            if step < accumulated_steps {
                self.current_phase = phase_idx;
                break;
            }
        }

        // Clamp to last phase
        if self.current_phase >= self.phase_configs.len() {
            self.current_phase = self.phase_configs.len() - 1;
        }
    }

    /// Get current phase configuration
    pub fn current_phase_config(&self) -> &PhaseConfig {
        &self.phase_configs[self.current_phase]
    }

    /// Compute phase-adjusted loss
    pub fn compute_phase_adjusted_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        quantized_params: &[&Tensor],
        ste_stats: &[STEStatistics],
    ) -> Result<(Tensor, ProgressiveLossComponents)> {
        let config = self.current_phase_config();

        // Base loss
        let base_loss = self.base_loss.compute_loss(predictions, targets)?;

        // Phase-adjusted regularization
        let reg_loss = self
            .base_loss
            .compute_quantization_regularization(quantized_params)?;
        let adjusted_reg = reg_loss.affine(config.regularization_weight as f64, 0.0)?;

        // Phase-adjusted quantization penalty
        let penalty_loss = self.base_loss.compute_quantization_penalty(ste_stats)?;
        let adjusted_penalty = penalty_loss.affine(config.quantization_weight as f64, 0.0)?;

        // Store scalar values before consuming tensors
        let base_scalar = base_loss.to_scalar::<f32>().unwrap_or(0.0);
        let reg_scalar = adjusted_reg.to_scalar::<f32>().unwrap_or(0.0);
        let penalty_scalar = adjusted_penalty.to_scalar::<f32>().unwrap_or(0.0);

        let base_reg_combined = (base_loss + adjusted_reg)?;
        let total_loss = (base_reg_combined + adjusted_penalty)?;
        let total_scalar = total_loss.to_scalar::<f32>().unwrap_or(0.0);

        let components = ProgressiveLossComponents {
            base_loss: base_scalar,
            regularization_loss: reg_scalar,
            quantization_penalty: penalty_scalar,
            total_loss: total_scalar,
            current_phase: self.current_phase,
            phase_name: config.name.clone(),
        };

        Ok((total_loss, components))
    }
}

impl QATLoss for ProgressiveQuantizationLoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        self.base_loss.compute_loss(predictions, targets)
    }

    fn get_name(&self) -> &str {
        "ProgressiveQuantizationLoss"
    }
}

#[derive(Debug, Clone)]
pub struct ProgressiveLossComponents {
    pub base_loss: f32,
    pub regularization_loss: f32,
    pub quantization_penalty: f32,
    pub total_loss: f32,
    pub current_phase: usize,
    pub phase_name: String,
}

/// Loss factory for creating different QAT loss types
pub struct QATLossFactory;

impl QATLossFactory {
    /// Create standard quantization-aware loss
    pub fn create_qat_loss(
        base_loss: BaseLossType,
        regularization_weight: f32,
        quantization_penalty_weight: f32,
        device: Device,
    ) -> QuantizationAwareLoss {
        QuantizationAwareLoss::new(
            base_loss,
            regularization_weight,
            quantization_penalty_weight,
            device,
        )
    }

    /// Create knowledge distillation loss
    pub fn create_distillation_loss(
        temperature: f32,
        alpha: f32,
        beta: f32,
        base_loss_type: BaseLossType,
        regularization_weight: f32,
        quantization_penalty_weight: f32,
        device: Device,
    ) -> DistillationLoss {
        let base_loss = Self::create_qat_loss(
            base_loss_type,
            regularization_weight,
            quantization_penalty_weight,
            device.clone(),
        );

        DistillationLoss::new(temperature, alpha, beta, base_loss, device)
    }

    /// Create progressive quantization loss
    pub fn create_progressive_loss(
        base_loss_type: BaseLossType,
        phase_configs: Vec<PhaseConfig>,
        device: Device,
    ) -> ProgressiveQuantizationLoss {
        let base_loss = Self::create_qat_loss(
            base_loss_type,
            0.01, // Default regularization
            0.01, // Default quantization penalty
            device.clone(),
        );

        ProgressiveQuantizationLoss::new(base_loss, phase_configs, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_quantization_aware_loss_creation() {
        let device = Device::Cpu;
        let loss = QuantizationAwareLoss::new(BaseLossType::CrossEntropy, 0.01, 0.01, device);

        assert_eq!(loss.get_name(), "QuantizationAwareCrossEntropy");
    }

    #[test]
    fn test_mse_loss_computation() -> Result<()> {
        let device = Device::Cpu;
        let loss = QuantizationAwareLoss::new(BaseLossType::MeanSquaredError, 0.01, 0.01, device);

        let predictions = Tensor::from_slice(&[1.0f32, 2.0f32, 3.0f32], (3,), &Device::Cpu)?;
        let targets = Tensor::from_slice(&[1.5f32, 2.5f32, 2.5f32], (3,), &Device::Cpu)?;

        let loss_value = loss.compute_loss(&predictions, &targets)?;
        let loss_scalar = loss_value.to_scalar::<f32>()?;

        // Expected MSE: ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((loss_scalar - 0.25).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_quantization_regularization() -> Result<()> {
        let device = Device::Cpu;
        let loss = QuantizationAwareLoss::new(
            BaseLossType::MeanSquaredError,
            0.1, // Higher weight for testing
            0.01,
            device,
        );

        let param1 = Tensor::from_slice(&[1.0f32, 2.0f32], (2,), &Device::Cpu)?;
        let param2 = Tensor::from_slice(&[3.0f32, 4.0f32], (2,), &Device::Cpu)?;
        let params = vec![&param1, &param2];

        let reg_loss = loss.compute_quantization_regularization(&params)?;
        let reg_scalar = reg_loss.to_scalar::<f32>()?;

        // Expected: 0.1 * (1^2 + 2^2 + 3^2 + 4^2) = 0.1 * 30 = 3.0
        assert!((reg_scalar - 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_distillation_loss_creation() {
        let device = Device::Cpu;
        let base_loss =
            QuantizationAwareLoss::new(BaseLossType::CrossEntropy, 0.01, 0.01, device.clone());

        let distillation_loss = DistillationLoss::new(
            3.0, // temperature
            0.7, // alpha
            0.3, // beta
            base_loss, device,
        );

        assert_eq!(distillation_loss.get_name(), "DistillationLoss");
    }

    #[test]
    fn test_progressive_loss_phase_update() {
        let device = Device::Cpu;
        let base_loss =
            QuantizationAwareLoss::new(BaseLossType::CrossEntropy, 0.01, 0.01, device.clone());

        let phase_configs = vec![
            PhaseConfig {
                quantization_weight: 0.1,
                regularization_weight: 0.1,
                min_steps: 100,
                name: "Phase1".to_string(),
            },
            PhaseConfig {
                quantization_weight: 0.5,
                regularization_weight: 0.05,
                min_steps: 100,
                name: "Phase2".to_string(),
            },
        ];

        let mut progressive_loss =
            ProgressiveQuantizationLoss::new(base_loss, phase_configs, device);

        // Initial phase
        assert_eq!(progressive_loss.current_phase, 0);
        assert_eq!(progressive_loss.current_phase_config().name, "Phase1");

        // Update to step 150 (should be in phase 2)
        progressive_loss.update_phase(150);
        assert_eq!(progressive_loss.current_phase, 1);
        assert_eq!(progressive_loss.current_phase_config().name, "Phase2");
    }

    #[test]
    fn test_loss_factory() {
        let device = Device::Cpu;

        // Test QAT loss creation
        let qat_loss =
            QATLossFactory::create_qat_loss(BaseLossType::CrossEntropy, 0.01, 0.01, device.clone());
        assert_eq!(qat_loss.get_name(), "QuantizationAwareCrossEntropy");

        // Test distillation loss creation
        let distillation_loss = QATLossFactory::create_distillation_loss(
            3.0,
            0.7,
            0.3,
            BaseLossType::CrossEntropy,
            0.01,
            0.01,
            device.clone(),
        );
        assert_eq!(distillation_loss.get_name(), "DistillationLoss");
    }
}
