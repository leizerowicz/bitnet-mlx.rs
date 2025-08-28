//! Knowledge Distillation support for QAT - Enhanced teacher-student training

use candle_core::{DType, Device, Result, Tensor};
use candle_nn;
use std::collections::HashMap;

/// Knowledge distillation configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DistillationConfig {
    /// Temperature for softmax softening
    pub temperature: f32,
    /// Weight for distillation loss
    pub alpha: f32,
    /// Weight for student loss
    pub beta: f32,
    /// Weight for feature matching loss
    pub gamma: f32,
    /// Whether to use attention transfer
    pub attention_transfer: bool,
    /// Layers to match features between teacher and student
    pub feature_layers: Vec<String>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            beta: 0.3,
            gamma: 0.1,
            attention_transfer: false,
            feature_layers: Vec::new(),
        }
    }
}

/// Teacher-Student pair for training
#[derive(Debug)]
#[allow(dead_code)]
pub struct TeacherStudentPair {
    pub teacher_name: String,
    pub student_name: String,
    pub feature_alignment: HashMap<String, String>, // teacher layer -> student layer
}

impl TeacherStudentPair {
    pub fn new(teacher_name: String, student_name: String) -> Self {
        Self {
            teacher_name,
            student_name,
            feature_alignment: HashMap::new(),
        }
    }

    pub fn add_feature_alignment(&mut self, teacher_layer: String, student_layer: String) {
        self.feature_alignment.insert(teacher_layer, student_layer);
    }
}

/// Metrics for tracking distillation progress
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DistillationMetrics {
    pub kl_divergence_loss: f32,
    pub student_loss: f32,
    pub feature_loss: f32,
    pub attention_loss: f32,
    pub total_loss: f32,
    pub temperature_used: f32,
    pub student_accuracy: Option<f32>,
    pub teacher_accuracy: Option<f32>,
    pub compression_ratio: Option<f32>,
}

impl DistillationMetrics {
    pub fn new() -> Self {
        Self {
            kl_divergence_loss: 0.0,
            student_loss: 0.0,
            feature_loss: 0.0,
            attention_loss: 0.0,
            total_loss: 0.0,
            temperature_used: 0.0,
            student_accuracy: None,
            teacher_accuracy: None,
            compression_ratio: None,
        }
    }
}

/// Knowledge distillation loss computation
#[allow(dead_code)]
pub struct KnowledgeDistillation {
    config: DistillationConfig,
    device: Device,
    metrics: DistillationMetrics,
}

impl KnowledgeDistillation {
    pub fn new(config: DistillationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            metrics: DistillationMetrics::new(),
        }
    }

    /// Compute complete distillation loss with all components
    pub fn compute_complete_loss(
        &mut self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: &Tensor,
        student_features: Option<&HashMap<String, Tensor>>,
        teacher_features: Option<&HashMap<String, Tensor>>,
    ) -> Result<(Tensor, DistillationMetrics)> {
        // KL divergence loss between teacher and student
        let kl_loss = self.compute_kl_divergence_loss(student_logits, teacher_logits)?;

        // Standard cross-entropy loss for student
        let student_loss = self.compute_student_loss(student_logits, targets)?;

        // Feature matching loss if features provided
        let feature_loss = if let (Some(student_feat), Some(teacher_feat)) =
            (student_features, teacher_features)
        {
            self.compute_feature_matching_loss(student_feat, teacher_feat)?
        } else {
            Tensor::zeros((), DType::F32, &self.device)?
        };

        // Attention transfer loss (placeholder for now)
        let attention_loss = Tensor::zeros((), DType::F32, &self.device)?;

        // Combine losses with weights
        let alpha_tensor = Tensor::new(self.config.alpha, &self.device)?;
        let beta_tensor = Tensor::new(self.config.beta, &self.device)?;
        let gamma_tensor = Tensor::new(self.config.gamma, &self.device)?;

        let weighted_kl = kl_loss.broadcast_mul(&alpha_tensor)?;
        let weighted_student = student_loss.broadcast_mul(&beta_tensor)?;
        let weighted_feature = feature_loss.broadcast_mul(&gamma_tensor)?;

        let total_loss = (weighted_kl + weighted_student + weighted_feature)?;

        // Update metrics
        self.metrics.kl_divergence_loss = kl_loss.to_scalar::<f32>().unwrap_or(0.0);
        self.metrics.student_loss = student_loss.to_scalar::<f32>().unwrap_or(0.0);
        self.metrics.feature_loss = feature_loss.to_scalar::<f32>().unwrap_or(0.0);
        self.metrics.attention_loss = attention_loss.to_scalar::<f32>().unwrap_or(0.0);
        self.metrics.total_loss = total_loss.to_scalar::<f32>().unwrap_or(0.0);
        self.metrics.temperature_used = self.config.temperature;

        Ok((total_loss, self.metrics.clone()))
    }

    /// Compute KL divergence loss between teacher and student outputs
    pub fn compute_kl_divergence_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
    ) -> Result<Tensor> {
        // Convert to F32 and apply temperature scaling
        let temp_scale = 1.0 / self.config.temperature;
        let student_soft = student_logits
            .to_dtype(DType::F32)?
            .affine(temp_scale as f64, 0.0)?;
        let teacher_soft = teacher_logits
            .to_dtype(DType::F32)?
            .affine(temp_scale as f64, 0.0)?;

        // Compute softmax for teacher (target)
        let teacher_probs = self.softmax(&teacher_soft)?;

        // Compute log softmax for student
        let student_log_probs = self.log_softmax(&student_soft)?;

        // KL divergence: sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        let kl_div = self.kl_divergence(&teacher_probs, &student_log_probs)?;

        // Scale by temperature squared to maintain gradient magnitudes
        let temp_sq = self.config.temperature * self.config.temperature;
        kl_div.affine(temp_sq as f64, 0.0)
    }

    /// Compute standard student loss
    pub fn compute_student_loss(
        &self,
        student_logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // Simple cross-entropy loss
        self.cross_entropy_loss(student_logits, targets)
    }

    /// Compute feature matching loss between teacher and student intermediate features
    pub fn compute_feature_matching_loss(
        &self,
        student_features: &HashMap<String, Tensor>,
        teacher_features: &HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let mut total_feature_loss = Tensor::zeros((), DType::F32, &self.device)?;
        let mut matched_layers = 0;

        for layer_name in &self.config.feature_layers {
            if let (Some(student_feat), Some(teacher_feat)) = (
                student_features.get(layer_name),
                teacher_features.get(layer_name),
            ) {
                // L2 distance between features
                let diff = (student_feat - teacher_feat)?;
                let l2_loss = diff.sqr()?.mean_all()?.to_dtype(DType::F32)?;

                total_feature_loss = (&total_feature_loss + &l2_loss)?;
                matched_layers += 1;
            }
        }

        // Average over matched layers
        if matched_layers > 0 {
            total_feature_loss.affine(1.0 / matched_layers as f64, 0.0)
        } else {
            Ok(total_feature_loss)
        }
    }

    /// Use candle_nn's built-in softmax
    fn softmax(&self, tensor: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax_last_dim(tensor)
    }

    /// Use candle_nn's built-in log_softmax
    fn log_softmax(&self, tensor: &Tensor) -> Result<Tensor> {
        candle_nn::ops::log_softmax(tensor, candle_core::D::Minus1)
    }

    fn kl_divergence(&self, teacher_probs: &Tensor, student_log_probs: &Tensor) -> Result<Tensor> {
        // KL(teacher || student) = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        let teacher_log_probs = teacher_probs.log()?;
        let diff = (teacher_log_probs - student_log_probs)?;
        let kl = (teacher_probs * diff)?;
        kl.sum(candle_core::D::Minus1)?.mean_all()
    }

    fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Simple MSE approximation for cross-entropy
        let diff = (logits - targets)?;
        let squared = diff.sqr()?;
        squared.mean_all()
    }

    /// Update distillation configuration
    pub fn update_config(&mut self, config: DistillationConfig) {
        self.config = config;
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &DistillationMetrics {
        &self.metrics
    }

    /// Get current configuration
    pub fn get_config(&self) -> &DistillationConfig {
        &self.config
    }

    /// Adaptive temperature scheduling
    pub fn update_temperature(&mut self, epoch: usize, total_epochs: usize) {
        // Gradually reduce temperature during training
        let progress = epoch as f32 / total_epochs as f32;
        let initial_temp = 4.0;
        let final_temp = 1.0;

        self.config.temperature = initial_temp - (initial_temp - final_temp) * progress;
        self.config.temperature = self.config.temperature.max(1.0); // Minimum temperature
    }
}

/// Legacy interface for backward compatibility
impl KnowledgeDistillation {
    /// Compute distillation loss between teacher and student outputs (legacy)
    pub fn compute_loss(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        targets: Option<&Tensor>,
    ) -> Result<Tensor> {
        // KL divergence loss
        let kl_loss = self.compute_kl_divergence_loss(student_logits, teacher_logits)?;

        // If targets provided, combine with cross-entropy
        if let Some(targets) = targets {
            let ce_loss = self.cross_entropy_loss(student_logits, targets)?;

            let weighted_kl = kl_loss.affine(self.config.alpha as f64, 0.0)?;
            let weighted_ce = ce_loss.affine(self.config.beta as f64, 0.0)?;

            &weighted_kl + &weighted_ce
        } else {
            Ok(kl_loss)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_knowledge_distillation() -> Result<()> {
        let device = Device::Cpu;
        let config = DistillationConfig::default();
        let kd = KnowledgeDistillation::new(config, device);

        let student_logits = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;
        let teacher_logits = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;
        let targets = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;

        let loss = kd.compute_loss(&student_logits, &teacher_logits, Some(&targets))?;
        assert!(loss.dims().is_empty()); // Should be scalar

        Ok(())
    }

    #[test]
    fn test_complete_distillation_loss() -> Result<()> {
        let device = Device::Cpu;
        let config = DistillationConfig::default();
        let mut kd = KnowledgeDistillation::new(config, device);

        let student_logits = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;
        let teacher_logits = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;
        let targets = Tensor::randn(0f32, 1f32, (4, 10), &Device::Cpu)?;

        let (loss, metrics) =
            kd.compute_complete_loss(&student_logits, &teacher_logits, &targets, None, None)?;

        assert!(loss.dims().is_empty()); // Should be scalar
        assert!(metrics.total_loss >= 0.0);

        Ok(())
    }

    #[test]
    fn test_teacher_student_pair() {
        let mut pair =
            TeacherStudentPair::new("teacher_model".to_string(), "student_model".to_string());

        pair.add_feature_alignment("teacher_layer1".to_string(), "student_layer1".to_string());

        assert_eq!(pair.teacher_name, "teacher_model");
        assert_eq!(pair.student_name, "student_model");
        assert!(pair.feature_alignment.contains_key("teacher_layer1"));
    }
}
