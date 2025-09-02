// Straight-Through Estimator Implementation
// Core component for Quantization-Aware Training in BitNet-Rust

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Straight-Through Estimator variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum STEVariant {
    /// Standard STE: forward quantize, backward identity
    Standard,
    /// Clipped STE: clip gradients to quantization range
    Clipped,
    /// Soft STE: learnable quantization with smooth approximation
    Soft,
    /// Learnable STE: learnable quantization parameters
    Learnable,
    /// Learned STE: alias for Learnable (for test compatibility)
    Learned,
    /// Adaptive STE: adaptive quantization parameters
    Adaptive,
}

/// STE Configuration
#[derive(Debug, Clone)]
pub struct STEConfig {
    /// Variant of STE to use
    pub variant: STEVariant,
    /// Quantization bit width
    pub bits: u32,
    /// Quantization range [-range, range]
    pub range: f32,
    /// Temperature for soft quantization
    pub temperature: f32,
    /// Learning rate for learnable parameters
    pub learnable_lr: f32,
    /// Enable gradient clipping
    pub clip_gradients: bool,
    /// Gradient clipping threshold
    pub clip_threshold: f32,
    /// Legacy field for compatibility with tests
    pub gradient_clip: Option<f32>,
    /// Legacy field for compatibility with tests
    pub use_noise: bool,
    /// Device for computation (legacy field for compatibility)
    pub device: Option<Device>,
}

impl Default for STEConfig {
    fn default() -> Self {
        Self {
            variant: STEVariant::Clipped,
            bits: 1, // BitNet default
            range: 1.0,
            temperature: 1.0,
            learnable_lr: 0.001,
            clip_gradients: true,
            clip_threshold: 1.0,
            gradient_clip: None,
            use_noise: false,
            device: None,
        }
    }
}

/// Straight-Through Estimator Implementation
pub struct StraightThroughEstimator {
    config: STEConfig,
    device: Device,
    dtype: DType,
    // Learnable parameters for Learnable STE variant
    learnable_scale: Option<Tensor>,
    learnable_zero_point: Option<Tensor>,
    // Statistics tracking
    quantization_error: f32,
    gradient_magnitude: f32,
    clipping_count: usize,
    total_operations: usize,
}

impl StraightThroughEstimator {
    /// Create new Straight-Through Estimator
    pub fn new(config: STEConfig) -> Result<Self> {
        // Use device from config if provided, otherwise default to CPU
        let device = config.device.clone().unwrap_or(Device::Cpu);

        let mut estimator = Self {
            config: config.clone(),
            device: device.clone(),
            dtype: DType::F32,
            learnable_scale: None,
            learnable_zero_point: None,
            quantization_error: 0.0,
            gradient_magnitude: 0.0,
            clipping_count: 0,
            total_operations: 0,
        };

        // Initialize learnable parameters if needed
        if matches!(config.variant, STEVariant::Learnable | STEVariant::Learned) {
            estimator.learnable_scale = Some(Tensor::ones(1, DType::F32, &device)?);
            estimator.learnable_zero_point = Some(Tensor::zeros(1, DType::F32, &device)?);
        }

        Ok(estimator)
    }

    /// Create new Straight-Through Estimator with explicit device (for backward compatibility)
    pub fn with_device(config: STEConfig, device: Device) -> Result<Self> {
        let mut config = config;
        config.device = Some(device);
        Self::new(config)
    }

    /// Forward pass: quantize input tensor
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.total_operations += 1;

        match self.config.variant {
            STEVariant::Standard => self.standard_forward(input),
            STEVariant::Clipped => self.clipped_forward(input),
            STEVariant::Soft => self.soft_forward(input),
            STEVariant::Learnable | STEVariant::Learned => self.learnable_forward(input),
            STEVariant::Adaptive => self.adaptive_forward(input),
        }
    }

    /// Standard STE forward pass
    fn standard_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let quantized = self.quantize_tensor(input)?;

        // Calculate quantization error for statistics
        let error = (input - &quantized)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        self.quantization_error = error;

        Ok(quantized)
    }

    /// Clipped STE forward pass with gradient clipping
    fn clipped_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Clip input to quantization range before quantizing
        let clipped_input = self.clip_to_range(input)?;
        let quantized = self.quantize_tensor(&clipped_input)?;

        // Track clipping statistics
        let diff_mask = input.ne(&clipped_input)?;
        let clipped_elements = diff_mask
            .to_dtype(input.dtype())?
            .sum_all()?
            .to_scalar::<f32>()? as usize;
        if clipped_elements > 0 {
            self.clipping_count += clipped_elements;
        }

        // Calculate quantization error
        let error = (input - &quantized)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        self.quantization_error = error;

        Ok(quantized)
    }

    /// Soft STE forward pass with smooth approximation
    fn soft_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Soft quantization using tanh approximation
        let temp_tensor =
            Tensor::new(self.config.temperature, input.device())?.to_dtype(input.dtype())?;
        let scaled_input = input.broadcast_mul(&temp_tensor)?;
        let soft_quantized = scaled_input.tanh()?;
        let range_tensor =
            Tensor::new(self.config.range, soft_quantized.device())?.to_dtype(input.dtype())?;
        let quantized = soft_quantized.broadcast_mul(&range_tensor)?;

        // Calculate quantization error
        let error = (input - &quantized)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        self.quantization_error = error;

        Ok(quantized)
    }

    /// Learnable STE forward pass with learnable quantization parameters
    fn learnable_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let scale = self.learnable_scale.as_ref().unwrap();
        let zero_point = self.learnable_zero_point.as_ref().unwrap();

        // Apply learnable scale and zero point
        let scaled_input = input.broadcast_mul(scale)?.broadcast_add(zero_point)?;
        let quantized = self.quantize_tensor(&scaled_input)?;

        // Apply inverse transformation
        let output = ((&quantized - zero_point)? / scale)?;

        // Calculate quantization error
        let error = (input - &output)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        self.quantization_error = error;

        Ok(output)
    }

    /// Adaptive STE forward pass with dynamic quantization parameters
    fn adaptive_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Adaptive STE adjusts quantization parameters based on input statistics
        let input_mean = input.mean_all()?.to_scalar::<f32>()?;

        // Calculate standard deviation manually
        let mean_tensor = Tensor::full(input_mean, input.shape(), input.device())?;
        let variance = input.sub(&mean_tensor)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let input_std = variance.sqrt();

        // Adjust range based on input statistics
        let adaptive_range = (input_std * 2.0).max(self.config.range);

        // Create temporary config with adaptive range
        let mut adaptive_config = self.config.clone();
        adaptive_config.range = adaptive_range;

        // Use clipped forward logic with adaptive range
        let clamped = input.clamp(-adaptive_range, adaptive_range)?;
        let quantized = self.quantize_tensor(&clamped)?;

        // Calculate quantization error
        let error = (input - &quantized)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        self.quantization_error = error;

        Ok(quantized)
    }

    /// Core quantization function
    fn quantize_tensor(&self, input: &Tensor) -> Result<Tensor> {
        match self.config.bits {
            1 => self.binary_quantize(input),
            2 => self.ternary_quantize(input),
            _ => self.multi_bit_quantize(input),
        }
    }

    /// Binary quantization (BitNet default)
    fn binary_quantize(&self, input: &Tensor) -> Result<Tensor> {
        // Sign function: -1 for negative, +1 for positive
        let sign = input.sign()?;
        let range_tensor =
            Tensor::new(self.config.range, input.device())?.to_dtype(input.dtype())?;
        let quantized = sign.broadcast_mul(&range_tensor)?;
        Ok(quantized)
    }

    /// Ternary quantization
    fn ternary_quantize(&self, input: &Tensor) -> Result<Tensor> {
        let threshold = self.config.range * 0.5;

        // Three levels: -range, 0, +range
        // Start with zero tensor
        let mut quantized = Tensor::zeros_like(input)?;

        // For positive values above threshold, set to +range
        let positive_condition = input.gt(threshold)?;
        let pos_val = Tensor::full(self.config.range, input.shape(), input.device())?
            .to_dtype(input.dtype())?;
        quantized = positive_condition.where_cond(&pos_val, &quantized)?;

        // For negative values below -threshold, set to -range
        let negative_condition = input.lt(-threshold)?;
        let neg_val = Tensor::full(-self.config.range, input.shape(), input.device())?
            .to_dtype(input.dtype())?;
        quantized = negative_condition.where_cond(&neg_val, &quantized)?;

        Ok(quantized)
    }

    /// Multi-bit quantization
    fn multi_bit_quantize(&self, input: &Tensor) -> Result<Tensor> {
        let levels = 2_u32.pow(self.config.bits) as f32;
        let step_size = (2.0 * self.config.range) / (levels - 1.0);

        // Quantize to discrete levels
        let range_tensor =
            Tensor::new(self.config.range, input.device())?.to_dtype(input.dtype())?;
        let step_size_tensor = Tensor::new(step_size, input.device())?.to_dtype(input.dtype())?;

        let range_added = input.broadcast_add(&range_tensor)?;
        let scaled = range_added.broadcast_div(&step_size_tensor)?;
        let quantized_indices = scaled.round()?;
        let quantized = quantized_indices
            .broadcast_mul(&step_size_tensor)?
            .broadcast_sub(&range_tensor)?;

        // Clamp to range
        self.clip_to_range(&quantized)
    }

    /// Clip tensor values to quantization range
    fn clip_to_range(&self, input: &Tensor) -> Result<Tensor> {
        input.clamp(-self.config.range, self.config.range)
    }

    /// Get current quantization error
    pub fn get_quantization_error(&self) -> f32 {
        self.quantization_error
    }

    /// Get clipping statistics
    pub fn get_clipping_rate(&self) -> f32 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.clipping_count as f32 / self.total_operations as f32
        }
    }

    /// Get STE statistics
    pub fn get_statistics(&self) -> STEStatistics {
        STEStatistics {
            quantization_error: self.quantization_error,
            gradient_magnitude: self.gradient_magnitude,
            clipping_rate: self.get_clipping_rate(),
            total_operations: self.total_operations,
        }
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.quantization_error = 0.0;
        self.gradient_magnitude = 0.0;
        self.clipping_count = 0;
        self.total_operations = 0;
    }

    /// Update learnable parameters (for Learnable STE)
    pub fn update_learnable_params(
        &mut self,
        scale_grad: &Tensor,
        zero_point_grad: &Tensor,
    ) -> Result<()> {
        if let (Some(scale), Some(zero_point)) = (&self.learnable_scale, &self.learnable_zero_point)
        {
            // Simple gradient descent update
            let lr_tensor = Tensor::new(self.config.learnable_lr, scale_grad.device())?
                .to_dtype(scale_grad.dtype())?;
            let scale_update = scale_grad.broadcast_mul(&lr_tensor)?;
            let zero_point_update = zero_point_grad.broadcast_mul(&lr_tensor)?;

            self.learnable_scale = Some((scale - scale_update)?);
            self.learnable_zero_point = Some((zero_point - zero_point_update)?);
        }
        Ok(())
    }

    /// Forward with quantization (alias for forward method)
    pub fn forward_quantized(&mut self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    /// Get the current STE variant
    pub fn variant(&self) -> STEVariant {
        self.config.variant
    }
}

/// STE Statistics structure
#[derive(Debug, Clone)]
pub struct STEStatistics {
    pub quantization_error: f32,
    pub gradient_magnitude: f32,
    pub clipping_rate: f32,
    pub total_operations: usize,
}

/// Convenience function for STE quantization
pub fn quantize_with_ste(input: &Tensor, config: &STEConfig, device: &Device) -> Result<Tensor> {
    let mut config = config.clone();
    config.device = Some(device.clone());
    let mut ste = StraightThroughEstimator::new(config)?;
    ste.forward(input)
}

/// Multi-layer STE manager for complex models
pub struct MultiLayerSTE {
    estimators: HashMap<String, StraightThroughEstimator>,
    global_config: STEConfig,
    device: Device,
}

impl MultiLayerSTE {
    pub fn new(global_config: STEConfig, device: Device) -> Self {
        Self {
            estimators: HashMap::new(),
            global_config,
            device,
        }
    }

    /// Get or create STE for layer
    pub fn get_or_create_layer_ste(
        &mut self,
        layer_name: &str,
    ) -> Result<&mut StraightThroughEstimator> {
        if !self.estimators.contains_key(layer_name) {
            let mut config = self.global_config.clone();
            config.device = Some(self.device.clone());
            let ste = StraightThroughEstimator::new(config)?;
            self.estimators.insert(layer_name.to_string(), ste);
        }
        Ok(self.estimators.get_mut(layer_name).unwrap())
    }

    /// Forward pass for specific layer
    pub fn forward_layer(&mut self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        let ste = self.get_or_create_layer_ste(layer_name)?;
        ste.forward(input)
    }

    /// Get statistics for all layers
    pub fn get_all_statistics(&self) -> HashMap<String, STEStatistics> {
        self.estimators
            .iter()
            .map(|(name, ste)| (name.clone(), ste.get_statistics()))
            .collect()
    }

    /// Reset all statistics
    pub fn reset_all_statistics(&mut self) {
        for ste in self.estimators.values_mut() {
            ste.reset_statistics();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ste_creation() {
        let mut config = STEConfig::default();
        config.device = Some(Device::Cpu);
        let ste = StraightThroughEstimator::new(config);
        assert!(ste.is_ok());
    }

    #[test]
    fn test_binary_quantization() -> Result<()> {
        let mut config = STEConfig {
            variant: STEVariant::Standard,
            bits: 1,
            range: 1.0,
            ..Default::default()
        };
        config.device = Some(Device::Cpu);
        let mut ste = StraightThroughEstimator::new(config)?;

        let input = Tensor::from_slice(&[0.5f32, -0.3f32, 0.8f32, -0.9f32], (4,), &Device::Cpu)?;
        let output = ste.forward(&input)?;

        let expected = vec![1.0, -1.0, 1.0, -1.0];
        let output_vec: Vec<f32> = output.to_vec1()?;

        for (actual, expected) in output_vec.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_clipped_ste() -> Result<()> {
        let mut config = STEConfig {
            variant: STEVariant::Clipped,
            bits: 1,
            range: 1.0,
            ..Default::default()
        };
        config.device = Some(Device::Cpu);
        let mut ste = StraightThroughEstimator::new(config)?;

        let input = Tensor::from_slice(&[2.0f32, -3.0f32, 0.5f32, -0.5f32], (4,), &Device::Cpu)?;
        let _output = ste.forward(&input)?;

        // Should be clipped to range and then quantized
        assert!(ste.get_clipping_rate() > 0.0);

        Ok(())
    }

    #[test]
    fn test_ternary_quantization() -> Result<()> {
        let mut config = STEConfig {
            variant: STEVariant::Standard,
            bits: 2,
            range: 1.0,
            ..Default::default()
        };
        config.device = Some(Device::Cpu);
        let mut ste = StraightThroughEstimator::new(config)?;

        let input = Tensor::from_slice(&[0.8f32, -0.8f32, 0.2f32, -0.2f32], (4,), &Device::Cpu)?;
        let output = ste.forward(&input)?;

        let output_vec: Vec<f32> = output.to_vec1()?;

        // Should quantize to -1, 0, or 1
        for val in output_vec {
            assert!(val.abs() <= 1.0);
            assert!(val == -1.0 || val == 0.0 || val == 1.0 || (val.abs() - 1.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_multi_layer_ste() -> Result<()> {
        let config = STEConfig::default();
        let device = Device::Cpu;
        let mut multi_ste = MultiLayerSTE::new(config, device);

        let input = Tensor::from_slice(&[0.5f32, -0.3f32], (2,), &Device::Cpu)?;

        // Forward pass through different layers
        let _output1 = multi_ste.forward_layer("layer1", &input)?;
        let _output2 = multi_ste.forward_layer("layer2", &input)?;

        // Should create separate STEs for each layer
        assert_eq!(multi_ste.estimators.len(), 2);

        // Get statistics
        let stats = multi_ste.get_all_statistics();
        assert_eq!(stats.len(), 2);
        assert!(stats.contains_key("layer1"));
        assert!(stats.contains_key("layer2"));

        Ok(())
    }

    #[test]
    fn test_learnable_ste() -> Result<()> {
        let mut config = STEConfig {
            variant: STEVariant::Learnable,
            bits: 1,
            range: 1.0,
            learnable_lr: 0.01,
            ..Default::default()
        };
        config.device = Some(Device::Cpu);
        let ste = StraightThroughEstimator::new(config);

        assert!(ste.is_ok());
        let ste = ste.unwrap();

        // Should have learnable parameters initialized
        assert!(ste.learnable_scale.is_some());
        assert!(ste.learnable_zero_point.is_some());

        Ok(())
    }
}

/// Binary quantization function (placeholder for compatibility)
#[derive(Debug, Clone)]
pub struct BinaryQuantizationFunction {
    pub threshold: f32,
    pub device: Device,
}

impl BinaryQuantizationFunction {
    pub fn new(threshold: f32, device: Device) -> Self {
        Self { threshold, device }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simple binary quantization: x > threshold -> 1, else -> -1
        let ones = Tensor::ones_like(input)?;
        let neg_ones = ones.neg()?;
        let threshold_tensor = Tensor::full(self.threshold, input.shape(), input.device())?;
        let mask = input.gt(&threshold_tensor)?;
        mask.where_cond(&ones, &neg_ones)
    }
}

/// Ternary quantization function (placeholder for compatibility)
#[derive(Debug, Clone)]
pub struct TernaryQuantizationFunction {
    pub threshold_pos: f32,
    pub threshold_neg: f32,
    pub device: Device,
}

impl TernaryQuantizationFunction {
    pub fn new(threshold_pos: f32, threshold_neg: f32, device: Device) -> Self {
        Self {
            threshold_pos,
            threshold_neg,
            device,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Ternary quantization: x > pos -> 1, x < neg -> -1, else -> 0
        let zeros = Tensor::zeros_like(input)?;
        let ones = Tensor::ones_like(input)?;
        let neg_ones = ones.neg()?;

        let pos_mask = input.gt(&Tensor::full(
            self.threshold_pos,
            input.dims(),
            input.device(),
        )?)?;
        let neg_mask = input.lt(&Tensor::full(
            self.threshold_neg,
            input.dims(),
            input.device(),
        )?)?;

        let result = pos_mask.where_cond(&ones, &zeros)?;
        neg_mask.where_cond(&neg_ones, &result)
    }
}
