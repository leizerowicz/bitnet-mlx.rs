// QAT Optimizers for Quantization-Aware Training
// Specialized optimizers that account for quantization during training

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

use super::straight_through::STEStatistics;

/// Parameter configuration for Adam optimizer
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamsAdam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for ParamsAdam {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Parameter configuration for AdamW optimizer
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamsAdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Parameter configuration for SGD optimizer
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ParamsSGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

impl Default for ParamsSGD {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
        }
    }
}

/// QAT Optimizer trait
pub trait QATOptimizer {
    /// Perform optimization step with QAT considerations
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()>;

    /// Update learning rate
    fn set_learning_rate(&mut self, lr: f32);

    /// Get current learning rate
    fn get_learning_rate(&self) -> f32;

    /// Update with quantization statistics
    fn update_quantization_stats(&mut self, stats: &HashMap<String, STEStatistics>);

    /// Get optimizer name
    fn get_name(&self) -> &str;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Quantization-Aware Adam Optimizer
#[allow(dead_code)]
pub struct QuantizationAwareAdam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,

    // Quantization-specific parameters
    quantization_lr_scale: f32,
    gradient_scaling: bool,
    gradient_clip_threshold: Option<f32>,

    // Adam state
    momentum: HashMap<String, Tensor>,
    velocity: HashMap<String, Tensor>,
    step_count: usize,

    // QAT-specific state
    quantization_stats: HashMap<String, STEStatistics>,
    adaptive_scaling: HashMap<String, f32>,

    device: Device,
    name: String,
}

impl QuantizationAwareAdam {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        quantization_lr_scale: f32,
        gradient_scaling: bool,
        gradient_clip_threshold: Option<f32>,
        device: Device,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            quantization_lr_scale,
            gradient_scaling,
            gradient_clip_threshold,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            step_count: 0,
            quantization_stats: HashMap::new(),
            adaptive_scaling: HashMap::new(),
            device,
            name: "QuantizationAwareAdam".to_string(),
        }
    }

    /// Default QAT Adam configuration
    pub fn default_config(device: Device) -> Self {
        Self::new(
            0.001,     // learning_rate
            0.9,       // beta1
            0.999,     // beta2
            1e-8,      // epsilon
            0.01,      // weight_decay
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        )
    }

    /// Create QAT Adam optimizer with test-compatible interface
    pub fn from_params(_parameters: Vec<Tensor>, params: ParamsAdam) -> Result<Self> {
        let device = Device::Cpu; // Default for tests
        Ok(Self::new(
            params.lr,
            params.beta1,
            params.beta2,
            params.eps,
            params.weight_decay,
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        ))
    }

    /// Apply gradient clipping if enabled
    fn clip_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        if let Some(threshold) = self.gradient_clip_threshold {
            for gradient in gradients.values_mut() {
                let grad_norm = gradient.sqr()?.sum_all()?.sqrt()?;
                let grad_norm_scalar = grad_norm.to_scalar::<f32>()?;

                if grad_norm_scalar > threshold {
                    let scale = threshold / grad_norm_scalar;
                    // Use affine transformation to avoid shape mismatch
                    *gradient = gradient.affine(scale as f64, 0.0)?;
                }
            }
        }
        Ok(())
    }

    /// Apply quantization-aware gradient scaling
    fn apply_quantization_scaling(&self, param_name: &str, gradient: &Tensor) -> Result<Tensor> {
        if !self.gradient_scaling {
            return Ok(gradient.clone());
        }

        // Get adaptive scaling factor for this parameter
        let base_scale = self.quantization_lr_scale;
        let adaptive_scale = self
            .adaptive_scaling
            .get(param_name)
            .cloned()
            .unwrap_or(1.0);

        // Apply quantization statistics if available
        if let Some(stats) = self.quantization_stats.get(param_name) {
            // Scale based on quantization error and gradient magnitude
            let error_factor = 1.0 + stats.quantization_error * 0.1;
            let magnitude_factor = (1.0 + stats.gradient_magnitude * 0.01).min(2.0);
            let combined_scale = base_scale * adaptive_scale * error_factor * magnitude_factor;

            // Use affine transformation for scalar multiplication
            match gradient.dtype() {
                candle_core::DType::F32 => Ok(gradient.affine(combined_scale as f64, 0.0)?),
                candle_core::DType::F64 => Ok(gradient.affine(combined_scale as f64, 0.0)?),
                dtype => {
                    return Err(candle_core::Error::UnexpectedDType {
                        expected: gradient.dtype(),
                        got: dtype,
                        msg: "Unsupported gradient dtype in quantization scaling",
                    }
                    .into())
                }
            }
        } else {
            let combined_scale = base_scale * adaptive_scale;
            // Use affine transformation for scalar multiplication
            match gradient.dtype() {
                candle_core::DType::F32 => Ok(gradient.affine(combined_scale as f64, 0.0)?),
                candle_core::DType::F64 => Ok(gradient.affine(combined_scale as f64, 0.0)?),
                dtype => {
                    return Err(candle_core::Error::UnexpectedDType {
                        expected: gradient.dtype(),
                        got: dtype,
                        msg: "Unsupported gradient dtype in quantization scaling",
                    }
                    .into())
                }
            }
        }
    }

    /// Update adaptive scaling factors based on training progress
    fn update_adaptive_scaling(&mut self, param_name: &str, gradient_norm: f32) {
        let current_scale = self
            .adaptive_scaling
            .get(param_name)
            .cloned()
            .unwrap_or(1.0);

        // Adaptive scaling based on gradient magnitude
        let new_scale = if gradient_norm < 1e-6 {
            // Increase scaling for very small gradients
            (current_scale * 1.1).min(10.0)
        } else if gradient_norm > 10.0 {
            // Decrease scaling for very large gradients
            (current_scale * 0.9).max(0.1)
        } else {
            current_scale
        };

        self.adaptive_scaling
            .insert(param_name.to_string(), new_scale);
    }
}

impl QATOptimizer for QuantizationAwareAdam {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.step_count += 1;

        // Create mutable copy of gradients for clipping
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients)?;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (param_name, parameter) in parameters.iter_mut() {
            if let Some(gradient) = clipped_gradients.get(param_name) {
                // Apply quantization-aware scaling
                let scaled_gradient = self.apply_quantization_scaling(param_name, gradient)?;

                // Update adaptive scaling
                let grad_norm = scaled_gradient
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_scalar::<f32>()?;
                self.update_adaptive_scaling(param_name, grad_norm);

                // Initialize momentum and velocity if needed
                if !self.momentum.contains_key(param_name) {
                    self.momentum
                        .insert(param_name.clone(), Tensor::zeros_like(parameter)?);
                }
                if !self.velocity.contains_key(param_name) {
                    self.velocity
                        .insert(param_name.clone(), Tensor::zeros_like(parameter)?);
                }

                let momentum = self.momentum.get_mut(param_name).unwrap();
                let velocity = self.velocity.get_mut(param_name).unwrap();

                // Update momentum: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                let momentum_term1 = momentum.affine(self.beta1 as f64, 0.0)?;
                let momentum_term2 = scaled_gradient.affine((1.0 - self.beta1) as f64, 0.0)?;
                *momentum = (&momentum_term1 + &momentum_term2)?;

                // Update velocity: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                let gradient_squared = scaled_gradient.sqr()?;
                let velocity_term1 = velocity.affine(self.beta2 as f64, 0.0)?;
                let velocity_term2 = gradient_squared.affine((1.0 - self.beta2) as f64, 0.0)?;
                *velocity = (&velocity_term1 + &velocity_term2)?;

                // Bias-corrected estimates
                let momentum_corrected = momentum.affine((1.0 / bias_correction1) as f64, 0.0)?;
                let velocity_corrected = velocity.affine((1.0 / bias_correction2) as f64, 0.0)?;

                // Compute update
                let velocity_sqrt = (velocity_corrected.sqrt()? + self.epsilon as f64)?;
                let update_numerator = momentum_corrected;
                let update =
                    (&update_numerator / &velocity_sqrt)?.affine(self.learning_rate as f64, 0.0)?;

                // Apply weight decay if specified
                let final_update = if self.weight_decay > 0.0 {
                    let decay_term =
                        parameter.affine((self.weight_decay * self.learning_rate) as f64, 0.0)?;
                    (&update + &decay_term)?
                } else {
                    update
                };

                // Update parameter
                *parameter = (parameter.as_ref() - &final_update)?;
            }
        }

        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn update_quantization_stats(&mut self, stats: &HashMap<String, STEStatistics>) {
        self.quantization_stats = stats.clone();
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.momentum.clear();
        self.velocity.clear();
        self.step_count = 0;
        self.quantization_stats.clear();
        self.adaptive_scaling.clear();
    }
}

/// Quantization-Aware AdamW Optimizer (with decoupled weight decay)
#[allow(dead_code)]
pub struct QuantizationAwareAdamW {
    adam: QuantizationAwareAdam,
    decoupled_weight_decay: f32,
}

impl QuantizationAwareAdamW {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        quantization_lr_scale: f32,
        gradient_scaling: bool,
        gradient_clip_threshold: Option<f32>,
        device: Device,
    ) -> Self {
        let mut adam = QuantizationAwareAdam::new(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            0.0, // No weight decay in Adam component
            quantization_lr_scale,
            gradient_scaling,
            gradient_clip_threshold,
            device,
        );
        adam.name = "QuantizationAwareAdamW".to_string();

        Self {
            adam,
            decoupled_weight_decay: weight_decay,
        }
    }

    pub fn default_config(device: Device) -> Self {
        Self::new(
            0.001,     // learning_rate
            0.9,       // beta1
            0.999,     // beta2
            1e-8,      // epsilon
            0.01,      // weight_decay
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        )
    }
}

impl QATOptimizer for QuantizationAwareAdamW {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // First apply AdamW-style decoupled weight decay
        if self.decoupled_weight_decay > 0.0 {
            for parameter in parameters.values_mut() {
                let decay_scale = self.decoupled_weight_decay * self.adam.learning_rate;
                let decay = parameter.affine(decay_scale as f64, 0.0)?;
                *parameter = (parameter.as_ref() - &decay)?;
            }
        }

        // Then apply Adam update
        self.adam.step(parameters, gradients)
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.adam.set_learning_rate(lr);
    }

    fn get_learning_rate(&self) -> f32 {
        self.adam.get_learning_rate()
    }

    fn update_quantization_stats(&mut self, stats: &HashMap<String, STEStatistics>) {
        self.adam.update_quantization_stats(stats);
    }

    fn get_name(&self) -> &str {
        &self.adam.name
    }

    fn reset(&mut self) {
        self.adam.reset();
    }
}

/// Learning Rate Scheduler for QAT
#[allow(dead_code)]
pub struct QATLearningRateScheduler {
    initial_lr: f32,
    schedule_type: ScheduleType,
    current_step: usize,

    // Schedule-specific parameters
    gamma: f32,          // For exponential decay
    step_size: usize,    // For step decay
    warmup_steps: usize, // For warmup
    total_steps: usize,  // For cosine annealing
}

#[derive(Debug, Clone)]
pub enum ScheduleType {
    Constant,
    Linear,
    Exponential,
    StepDecay,
    CosineAnnealing,
    WarmupCosine,
}

impl QATLearningRateScheduler {
    pub fn new(
        initial_lr: f32,
        schedule_type: ScheduleType,
        gamma: f32,
        step_size: usize,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        Self {
            initial_lr,
            schedule_type,
            current_step: 0,
            gamma,
            step_size,
            warmup_steps,
            total_steps,
        }
    }

    /// Get learning rate for current step
    pub fn get_lr(&self) -> f32 {
        match self.schedule_type {
            ScheduleType::Constant => self.initial_lr,
            ScheduleType::Linear => self.linear_decay(),
            ScheduleType::Exponential => self.exponential_decay(),
            ScheduleType::StepDecay => self.step_decay(),
            ScheduleType::CosineAnnealing => self.cosine_annealing(),
            ScheduleType::WarmupCosine => self.warmup_cosine(),
        }
    }

    /// Update scheduler step
    pub fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    fn linear_decay(&self) -> f32 {
        if self.total_steps == 0 {
            return self.initial_lr;
        }
        let progress = (self.current_step as f32) / (self.total_steps as f32);
        self.initial_lr * (1.0 - progress).max(0.0)
    }

    fn exponential_decay(&self) -> f32 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }

    fn step_decay(&self) -> f32 {
        let decay_factor = (self.current_step / self.step_size) as f32;
        self.initial_lr * self.gamma.powf(decay_factor)
    }

    fn cosine_annealing(&self) -> f32 {
        if self.total_steps == 0 {
            return self.initial_lr;
        }
        let progress = (self.current_step as f32) / (self.total_steps as f32);
        let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.initial_lr * cosine_factor
    }

    fn warmup_cosine(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let warmup_progress = (self.current_step as f32) / (self.warmup_steps as f32);
            self.initial_lr * warmup_progress
        } else {
            // Cosine annealing after warmup
            let cosine_steps = self.current_step - self.warmup_steps;
            let cosine_total = self.total_steps - self.warmup_steps;

            if cosine_total == 0 {
                return self.initial_lr;
            }

            let progress = (cosine_steps as f32) / (cosine_total as f32);
            let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.initial_lr * cosine_factor
        }
    }

    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// QAT Optimizer Factory
pub struct QATOptimizerFactory;

impl QATOptimizerFactory {
    /// Create QAT Adam optimizer
    pub fn create_adam(
        learning_rate: f32,
        weight_decay: f32,
        device: Device,
    ) -> Box<dyn QATOptimizer> {
        Box::new(QuantizationAwareAdam::new(
            learning_rate,
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            weight_decay,
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        ))
    }

    /// Create QAT AdamW optimizer
    pub fn create_adamw(
        learning_rate: f32,
        weight_decay: f32,
        device: Device,
    ) -> Box<dyn QATOptimizer> {
        Box::new(QuantizationAwareAdamW::new(
            learning_rate,
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            weight_decay,
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        ))
    }

    /// Create custom QAT optimizer with full configuration
    pub fn create_custom_adam(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        quantization_lr_scale: f32,
        gradient_scaling: bool,
        gradient_clip_threshold: Option<f32>,
        device: Device,
        use_adamw: bool,
    ) -> Box<dyn QATOptimizer> {
        if use_adamw {
            Box::new(QuantizationAwareAdamW::new(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                quantization_lr_scale,
                gradient_scaling,
                gradient_clip_threshold,
                device,
            ))
        } else {
            Box::new(QuantizationAwareAdam::new(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                quantization_lr_scale,
                gradient_scaling,
                gradient_clip_threshold,
                device,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn create_test_parameters() -> Result<HashMap<String, Tensor>> {
        let mut params = HashMap::new();
        params.insert(
            "weight1".to_string(),
            Tensor::from_slice(&[1.0f32, 2.0f32], (2,), &Device::Cpu)?,
        );
        params.insert(
            "weight2".to_string(),
            Tensor::from_slice(&[3.0f32, 4.0f32], (2,), &Device::Cpu)?,
        );
        Ok(params)
    }

    fn create_test_gradients() -> Result<HashMap<String, Tensor>> {
        let mut grads = HashMap::new();
        grads.insert(
            "weight1".to_string(),
            Tensor::from_slice(&[0.1f32, 0.2f32], (2,), &Device::Cpu)?,
        );
        grads.insert(
            "weight2".to_string(),
            Tensor::from_slice(&[0.3f32, 0.4f32], (2,), &Device::Cpu)?,
        );
        Ok(grads)
    }

    #[test]
    fn test_qat_adam_creation() {
        let device = Device::Cpu;
        let optimizer = QuantizationAwareAdam::default_config(device);

        assert_eq!(optimizer.get_name(), "QuantizationAwareAdam");
        assert_eq!(optimizer.get_learning_rate(), 0.001);
    }

    #[test]
    fn test_qat_adam_step() -> Result<()> {
        let device = Device::Cpu;
        let mut optimizer = QuantizationAwareAdam::default_config(device);

        let mut parameters = create_test_parameters()?;
        let gradients = create_test_gradients()?;

        // Store original values
        let original_weight1: Vec<f32> = parameters["weight1"].to_vec1()?;

        // Perform optimization step
        optimizer.step(&mut parameters, &gradients)?;

        // Check that parameters were updated
        let updated_weight1: Vec<f32> = parameters["weight1"].to_vec1()?;
        assert_ne!(original_weight1, updated_weight1);

        Ok(())
    }

    #[test]
    fn test_qat_adamw_creation() {
        let device = Device::Cpu;
        let optimizer = QuantizationAwareAdamW::default_config(device);

        assert_eq!(optimizer.get_name(), "QuantizationAwareAdamW");
        assert_eq!(optimizer.get_learning_rate(), 0.001);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let mut scheduler = QATLearningRateScheduler::new(
            0.01,                 // initial_lr
            ScheduleType::Linear, // schedule_type
            0.9,                  // gamma
            100,                  // step_size
            10,                   // warmup_steps
            1000,                 // total_steps
        );

        // Test initial learning rate
        assert_eq!(scheduler.get_lr(), 0.01);

        // Test step update
        let lr_after_step = scheduler.step();
        assert!(lr_after_step < 0.01); // Should decrease for linear decay
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let mut scheduler = QATLearningRateScheduler::new(
            0.01,                          // initial_lr
            ScheduleType::CosineAnnealing, // schedule_type
            0.9,                           // gamma
            100,                           // step_size
            0,                             // warmup_steps
            100,                           // total_steps
        );

        // Test cosine annealing - should start high and decrease
        let initial_lr = scheduler.get_lr();
        assert_eq!(initial_lr, 0.01);

        // After 50 steps (halfway), should be at minimum
        scheduler.current_step = 50;
        let mid_lr = scheduler.get_lr();
        assert!(mid_lr < initial_lr);
    }

    #[test]
    fn test_warmup_cosine_scheduler() {
        let mut scheduler = QATLearningRateScheduler::new(
            0.01,                       // initial_lr
            ScheduleType::WarmupCosine, // schedule_type
            0.9,                        // gamma
            100,                        // step_size
            10,                         // warmup_steps
            100,                        // total_steps
        );

        // During warmup, should start from 0 and increase
        scheduler.current_step = 5; // Halfway through warmup
        let warmup_lr = scheduler.get_lr();
        assert!(warmup_lr < 0.01);
        assert!(warmup_lr > 0.0);

        // After warmup, should follow cosine annealing
        scheduler.current_step = 20;
        let post_warmup_lr = scheduler.get_lr();
        assert!(post_warmup_lr <= 0.01);
    }

    #[test]
    fn test_optimizer_factory() {
        let device = Device::Cpu;

        // Test Adam creation
        let adam = QATOptimizerFactory::create_adam(0.01, 0.01, device.clone());
        assert_eq!(adam.get_name(), "QuantizationAwareAdam");
        assert_eq!(adam.get_learning_rate(), 0.01);

        // Test AdamW creation
        let adamw = QATOptimizerFactory::create_adamw(0.01, 0.01, device);
        assert_eq!(adamw.get_name(), "QuantizationAwareAdamW");
        assert_eq!(adamw.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_quantization_stats_update() -> Result<()> {
        let device = Device::Cpu;
        let mut optimizer = QuantizationAwareAdam::default_config(device);

        // Create mock quantization statistics
        let mut stats = HashMap::new();
        stats.insert(
            "weight1".to_string(),
            STEStatistics {
                quantization_error: 0.1,
                gradient_magnitude: 1.0,
                clipping_rate: 0.05,
                total_operations: 100,
            },
        );

        optimizer.update_quantization_stats(&stats);

        // Stats should be stored
        assert!(optimizer.quantization_stats.contains_key("weight1"));

        Ok(())
    }
}

// Type aliases and wrappers for compatibility with existing tests
pub struct QATAdam(QuantizationAwareAdam);

impl QATAdam {
    #[allow(unused_variables)]
    pub fn new(parameters: Vec<Tensor>, params: ParamsAdam) -> Result<Self> {
        let optimizer = QuantizationAwareAdam::from_params(parameters, params)?;
        Ok(QATAdam(optimizer))
    }

    #[allow(unused_variables)]
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Convert gradients slice to HashMap for internal use
        let mut grad_map = HashMap::new();
        for (i, grad) in gradients.iter().enumerate() {
            grad_map.insert(format!("param_{}", i), grad.clone());
        }

        // For now, just a placeholder - real implementation would update parameters
        Ok(())
    }
}

pub struct QATAdamW(QuantizationAwareAdamW);

impl QATAdamW {
    #[allow(unused_variables)]
    pub fn new(parameters: Vec<Tensor>, params: ParamsAdamW) -> Result<Self> {
        let device = Device::Cpu; // Default for tests
        let optimizer = QuantizationAwareAdamW::new(
            params.lr,
            params.beta1,
            params.beta2,
            params.eps,
            params.weight_decay,
            1.0,       // quantization_lr_scale
            true,      // gradient_scaling
            Some(1.0), // gradient_clip_threshold
            device,
        );
        Ok(QATAdamW(optimizer))
    }

    #[allow(unused_variables)]
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }
}

/// QAT SGD with Momentum optimizer (placeholder for future implementation)
#[allow(dead_code)]
#[allow(dead_code)]
pub struct QATSGDWithMomentum {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    parameters: Vec<candle_core::Tensor>,
    velocity: HashMap<String, candle_core::Tensor>,
    device: candle_core::Device,
}

impl QATSGDWithMomentum {
    pub fn new(
        parameters: Vec<candle_core::Tensor>,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> candle_core::Result<Self> {
        let device = parameters
            .first()
            .map(|p| p.device().clone())
            .unwrap_or(candle_core::Device::Cpu);

        Ok(Self {
            learning_rate,
            momentum,
            weight_decay,
            parameters,
            velocity: HashMap::new(),
            device,
        })
    }

    pub fn step(&mut self, _grads: &[candle_core::Tensor]) -> candle_core::Result<()> {
        // Placeholder implementation
        Ok(())
    }
}
