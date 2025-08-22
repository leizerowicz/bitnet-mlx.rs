//! Comprehensive quantization configuration system for BitNet models
//! 
//! This module provides a unified configuration system for all quantization operations,
//! including weights, activations, packing strategies, and SIMD optimizations.

use super::{QuantizationPrecision, QuantizationStrategy};
use super::packing::TernaryPackingStrategy;
use super::weights::TernaryMethod;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core quantization configuration shared across all quantization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization precision (1.58-bit, 8-bit, etc.)
    pub precision: QuantizationPrecision,
    /// Quantization strategy (symmetric, asymmetric, dynamic, static)
    pub strategy: QuantizationStrategy,
    /// Whether to use per-channel quantization
    pub per_channel: bool,
    /// Clipping threshold for outliers (in standard deviations)
    pub clip_threshold: Option<f32>,
    /// Whether to enable quantization-aware training
    pub qat_enabled: bool,
    /// Calibration dataset size for dynamic quantization
    pub calibration_size: Option<usize>,
    /// Random seed for reproducible quantization
    pub seed: Option<u64>,
    /// Whether to enable verbose logging
    pub verbose: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            per_channel: false,
            clip_threshold: None,
            qat_enabled: false,
            calibration_size: None,
            seed: None,
            verbose: false,
        }
    }
}

impl QuantizationConfig {
    /// Create a new quantization configuration with specified precision
    pub fn new(precision: QuantizationPrecision) -> Self {
        Self {
            precision,
            ..Default::default()
        }
    }
    
    /// Create configuration for BitNet 1.58-bit quantization
    pub fn bitnet_158() -> Self {
        Self {
            precision: QuantizationPrecision::OneFiveFiveBit,
            strategy: QuantizationStrategy::Symmetric,
            per_channel: false,
            clip_threshold: Some(3.0),
            qat_enabled: false,
            calibration_size: Some(1000),
            seed: Some(42),
            verbose: false,
        }
    }
    
    /// Create configuration for 8-bit quantization
    pub fn int8() -> Self {
        Self {
            precision: QuantizationPrecision::EightBit,
            strategy: QuantizationStrategy::Asymmetric,
            per_channel: true,
            clip_threshold: Some(6.0),
            qat_enabled: false,
            calibration_size: Some(500),
            seed: Some(42),
            verbose: false,
        }
    }
    
    /// Create configuration for dynamic quantization
    pub fn dynamic() -> Self {
        Self {
            precision: QuantizationPrecision::EightBit,
            strategy: QuantizationStrategy::Dynamic,
            per_channel: false,
            clip_threshold: None,
            qat_enabled: false,
            calibration_size: None,
            seed: None,
            verbose: false,
        }
    }
    
    /// Enable quantization-aware training
    pub fn with_qat(mut self) -> Self {
        self.qat_enabled = true;
        self
    }
    
    /// Set clipping threshold
    pub fn with_clipping(mut self, threshold: f32) -> Self {
        self.clip_threshold = Some(threshold);
        self
    }
    
    /// Set calibration size
    pub fn with_calibration(mut self, size: usize) -> Self {
        self.calibration_size = Some(size);
        self
    }
    
    /// Enable per-channel quantization
    pub fn with_per_channel(mut self) -> Self {
        self.per_channel = true;
        self
    }
    
    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable verbose logging
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Check precision-strategy compatibility
        if let (QuantizationPrecision::OneFiveFiveBit, QuantizationStrategy::Asymmetric) = (self.precision, self.strategy) {
            return Err(ConfigValidationError::IncompatibleSettings(
                "1.58-bit quantization should use symmetric strategy".to_string()
            ));
        }
        
        // Validate clipping threshold
        if let Some(threshold) = self.clip_threshold {
            if threshold <= 0.0 || threshold > 10.0 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Clipping threshold {threshold} is out of valid range (0, 10]")
                ));
            }
        }
        
        // Validate calibration size
        if let Some(size) = self.calibration_size {
            if size == 0 || size > 100000 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Calibration size {size} is out of valid range [1, 100000]")
                ));
            }
        }
        
        Ok(())
    }
}

/// Enhanced weight quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightQuantizationConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Group size for grouped quantization (None for per-tensor)
    pub group_size: Option<usize>,
    /// Whether to use weight normalization before quantization
    pub normalize_weights: bool,
    /// Outlier threshold for weight clipping (in standard deviations)
    pub outlier_threshold: f32,
    /// Whether to use learnable scaling factors
    pub learnable_scales: bool,
    /// Block size for block-wise quantization
    pub block_size: Option<usize>,
    /// Ternary quantization method
    pub ternary_method: TernaryMethod,
    /// Custom threshold factor for ternary quantization
    pub custom_threshold_factor: Option<f32>,
    /// Packing configuration for ternary weights
    pub packing: PackingConfig,
    /// Whether to enable weight freezing during training
    pub freeze_weights: bool,
    /// Weight decay factor for regularization
    pub weight_decay: Option<f32>,
    /// Gradient clipping for quantized weights
    pub gradient_clip: Option<f32>,
}

impl Default for WeightQuantizationConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            group_size: None,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            block_size: None,
            ternary_method: TernaryMethod::default(),
            custom_threshold_factor: None,
            packing: PackingConfig::default(),
            freeze_weights: false,
            weight_decay: None,
            gradient_clip: None,
        }
    }
}

impl WeightQuantizationConfig {
    /// Create configuration for BitNet weight quantization
    pub fn bitnet() -> Self {
        Self {
            base: QuantizationConfig::bitnet_158(),
            group_size: None,
            normalize_weights: true,
            outlier_threshold: 3.0,
            learnable_scales: false,
            block_size: Some(64),
            ternary_method: TernaryMethod::MeanThreshold,
            custom_threshold_factor: Some(0.7),
            packing: PackingConfig::bitnet(),
            freeze_weights: false,
            weight_decay: Some(1e-4),
            gradient_clip: Some(1.0),
        }
    }
    
    /// Create configuration for grouped quantization
    pub fn grouped(group_size: usize) -> Self {
        Self {
            group_size: Some(group_size),
            ..Default::default()
        }
    }
    
    /// Create configuration for block-wise quantization
    pub fn blockwise(block_size: usize) -> Self {
        Self {
            block_size: Some(block_size),
            ..Default::default()
        }
    }
    
    /// Enable learnable scaling factors
    pub fn with_learnable_scales(mut self) -> Self {
        self.learnable_scales = true;
        self
    }
    
    /// Set ternary quantization method
    pub fn with_ternary_method(mut self, method: TernaryMethod) -> Self {
        self.ternary_method = method;
        self
    }
    
    /// Set custom threshold factor
    pub fn with_threshold_factor(mut self, factor: f32) -> Self {
        self.custom_threshold_factor = Some(factor);
        self
    }
    
    /// Set packing configuration
    pub fn with_packing(mut self, packing: PackingConfig) -> Self {
        self.packing = packing;
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate base configuration
        self.base.validate()?;
        
        // Validate group size
        if let Some(size) = self.group_size {
            if size == 0 || size > 4096 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Group size {size} is out of valid range [1, 4096]")
                ));
            }
        }
        
        // Validate block size
        if let Some(size) = self.block_size {
            if size == 0 || size > 1024 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Block size {size} is out of valid range [1, 1024]")
                ));
            }
        }
        
        // Validate outlier threshold
        if self.outlier_threshold <= 0.0 || self.outlier_threshold > 10.0 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Outlier threshold {} is out of valid range (0, 10]", self.outlier_threshold)
            ));
        }
        
        // Validate custom threshold factor
        if let Some(factor) = self.custom_threshold_factor {
            if factor <= 0.0 || factor > 2.0 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Threshold factor {factor} is out of valid range (0, 2]")
                ));
            }
        }
        
        // Validate packing configuration
        self.packing.validate()?;
        
        Ok(())
    }
}

/// Enhanced activation quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationQuantizationConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Moving average window size for dynamic scaling
    pub moving_average_window: usize,
    /// Percentile for outlier detection (e.g., 99.9)
    pub outlier_percentile: f32,
    /// Whether to use per-token quantization
    pub per_token: bool,
    /// Calibration warmup steps
    pub calibration_warmup: usize,
    /// Exponential moving average decay factor
    pub ema_decay: f32,
    /// Whether to quantize attention scores
    pub quantize_attention: bool,
    /// Attention quantization configuration
    pub attention: AttentionQuantizationConfig,
    /// Whether to use smooth quantization
    pub smooth_quantization: bool,
    /// Temperature for smooth quantization
    pub temperature: f32,
    /// Whether to enable activation caching
    pub enable_caching: bool,
    /// Cache size limit (in MB)
    pub cache_size_mb: Option<usize>,
}

impl Default for ActivationQuantizationConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            moving_average_window: 100,
            outlier_percentile: 99.9,
            per_token: false,
            calibration_warmup: 50,
            ema_decay: 0.99,
            quantize_attention: true,
            attention: AttentionQuantizationConfig::default(),
            smooth_quantization: false,
            temperature: 1.0,
            enable_caching: false,
            cache_size_mb: None,
        }
    }
}

impl ActivationQuantizationConfig {
    /// Create configuration for BitNet activation quantization
    pub fn bitnet() -> Self {
        Self {
            base: QuantizationConfig::bitnet_158(),
            moving_average_window: 100,
            outlier_percentile: 99.5,
            per_token: false,
            calibration_warmup: 100,
            ema_decay: 0.99,
            quantize_attention: true,
            attention: AttentionQuantizationConfig::bitnet(),
            smooth_quantization: true,
            temperature: 0.1,
            enable_caching: true,
            cache_size_mb: Some(256),
        }
    }
    
    /// Create configuration for dynamic activation quantization
    pub fn dynamic() -> Self {
        Self {
            base: QuantizationConfig::dynamic(),
            moving_average_window: 50,
            outlier_percentile: 99.9,
            per_token: true,
            calibration_warmup: 0,
            ema_decay: 0.95,
            quantize_attention: false,
            attention: AttentionQuantizationConfig::default(),
            smooth_quantization: false,
            temperature: 1.0,
            enable_caching: false,
            cache_size_mb: None,
        }
    }
    
    /// Enable per-token quantization
    pub fn with_per_token(mut self) -> Self {
        self.per_token = true;
        self
    }
    
    /// Set moving average window
    pub fn with_window(mut self, window: usize) -> Self {
        self.moving_average_window = window;
        self
    }
    
    /// Enable smooth quantization
    pub fn with_smooth_quantization(mut self, temperature: f32) -> Self {
        self.smooth_quantization = true;
        self.temperature = temperature;
        self
    }
    
    /// Enable activation caching
    pub fn with_caching(mut self, cache_size_mb: usize) -> Self {
        self.enable_caching = true;
        self.cache_size_mb = Some(cache_size_mb);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate base configuration
        self.base.validate()?;
        
        // Validate moving average window
        if self.moving_average_window == 0 || self.moving_average_window > 10000 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Moving average window {} is out of valid range [1, 10000]", self.moving_average_window)
            ));
        }
        
        // Validate outlier percentile
        if self.outlier_percentile <= 0.0 || self.outlier_percentile > 100.0 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Outlier percentile {} is out of valid range (0, 100]", self.outlier_percentile)
            ));
        }
        
        // Validate EMA decay
        if self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err(ConfigValidationError::InvalidValue(
                format!("EMA decay {} is out of valid range (0, 1)", self.ema_decay)
            ));
        }
        
        // Validate temperature
        if self.temperature <= 0.0 || self.temperature > 10.0 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Temperature {} is out of valid range (0, 10]", self.temperature)
            ));
        }
        
        // Validate cache size
        if let Some(size) = self.cache_size_mb {
            if size == 0 || size > 8192 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Cache size {size} MB is out of valid range [1, 8192]")
                ));
            }
        }
        
        // Validate attention configuration
        self.attention.validate()?;
        
        Ok(())
    }
}

/// Attention-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionQuantizationConfig {
    /// Whether to quantize query projections
    pub quantize_query: bool,
    /// Whether to quantize key projections
    pub quantize_key: bool,
    /// Whether to quantize value projections
    pub quantize_value: bool,
    /// Whether to quantize attention scores
    pub quantize_scores: bool,
    /// Whether to quantize output projections
    pub quantize_output: bool,
    /// Attention score clipping threshold
    pub score_clip_threshold: Option<f32>,
    /// Whether to use causal attention optimization
    pub causal_optimization: bool,
    /// Sparsity threshold for attention pruning
    pub sparsity_threshold: Option<f32>,
}

impl Default for AttentionQuantizationConfig {
    fn default() -> Self {
        Self {
            quantize_query: true,
            quantize_key: true,
            quantize_value: true,
            quantize_scores: false,
            quantize_output: true,
            score_clip_threshold: None,
            causal_optimization: false,
            sparsity_threshold: None,
        }
    }
}

impl AttentionQuantizationConfig {
    /// Create configuration for BitNet attention quantization
    pub fn bitnet() -> Self {
        Self {
            quantize_query: true,
            quantize_key: true,
            quantize_value: true,
            quantize_scores: true,
            quantize_output: true,
            score_clip_threshold: Some(10.0),
            causal_optimization: true,
            sparsity_threshold: Some(0.01),
        }
    }
    
    /// Disable attention score quantization
    pub fn without_score_quantization(mut self) -> Self {
        self.quantize_scores = false;
        self
    }
    
    /// Enable causal attention optimization
    pub fn with_causal_optimization(mut self) -> Self {
        self.causal_optimization = true;
        self
    }
    
    /// Set sparsity threshold for attention pruning
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = Some(threshold);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate score clip threshold
        if let Some(threshold) = self.score_clip_threshold {
            if threshold <= 0.0 || threshold > 100.0 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Score clip threshold {threshold} is out of valid range (0, 100]")
                ));
            }
        }
        
        // Validate sparsity threshold
        if let Some(threshold) = self.sparsity_threshold {
            if threshold <= 0.0 || threshold >= 1.0 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Sparsity threshold {threshold} is out of valid range (0, 1)")
                ));
            }
        }
        
        Ok(())
    }
}

/// Enhanced packing configuration for ternary weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingConfig {
    /// Primary packing strategy
    pub strategy: TernaryPackingStrategy,
    /// Block size for block-wise packing
    pub block_size: Option<usize>,
    /// Sparsity threshold for switching to sparse formats
    pub sparsity_threshold: f32,
    /// Whether to use SIMD-optimized layouts
    pub simd_optimized: bool,
    /// Alignment requirements for memory access
    pub alignment: usize,
    /// Whether to enable compression for sparse formats
    pub enable_compression: bool,
    /// SIMD configuration
    pub simd: SimdConfig,
    /// Whether to enable integrity checking
    pub integrity_checking: bool,
    /// Compression level (0-9, higher = better compression)
    pub compression_level: u8,
    /// Whether to enable parallel packing
    pub parallel_packing: bool,
    /// Number of threads for parallel operations
    pub num_threads: Option<usize>,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            strategy: TernaryPackingStrategy::default(),
            block_size: Some(64),
            sparsity_threshold: 0.7,
            simd_optimized: true,
            alignment: 16,
            enable_compression: true,
            simd: SimdConfig::default(),
            integrity_checking: true,
            compression_level: 6,
            parallel_packing: false,
            num_threads: None,
        }
    }
}

impl PackingConfig {
    /// Create configuration for BitNet packing
    pub fn bitnet() -> Self {
        Self {
            strategy: TernaryPackingStrategy::Hybrid,
            block_size: Some(64),
            sparsity_threshold: 0.6,
            simd_optimized: true,
            alignment: 32,
            enable_compression: true,
            simd: SimdConfig::aggressive(),
            integrity_checking: true,
            compression_level: 8,
            parallel_packing: true,
            num_threads: None, // Auto-detect
        }
    }
    
    /// Create configuration for maximum compression
    pub fn max_compression() -> Self {
        Self {
            strategy: TernaryPackingStrategy::Hybrid,
            block_size: Some(128),
            sparsity_threshold: 0.5,
            simd_optimized: true,
            alignment: 16,
            enable_compression: true,
            simd: SimdConfig::default(),
            integrity_checking: false, // Disable for max speed
            compression_level: 9,
            parallel_packing: true,
            num_threads: None,
        }
    }
    
    /// Create configuration for maximum speed
    pub fn max_speed() -> Self {
        Self {
            strategy: TernaryPackingStrategy::BitPacked2Bit,
            block_size: Some(32),
            sparsity_threshold: 0.9, // Only use sparse for very sparse data
            simd_optimized: true,
            alignment: 32,
            enable_compression: false,
            simd: SimdConfig::aggressive(),
            integrity_checking: false,
            compression_level: 0,
            parallel_packing: true,
            num_threads: None,
        }
    }
    
    /// Enable parallel packing
    pub fn with_parallel_packing(mut self, num_threads: Option<usize>) -> Self {
        self.parallel_packing = true;
        self.num_threads = num_threads;
        self
    }
    
    /// Set compression level
    pub fn with_compression_level(mut self, level: u8) -> Self {
        self.compression_level = level.min(9);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate block size
        if let Some(size) = self.block_size {
            if size == 0 || size > 2048 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Block size {size} is out of valid range [1, 2048]")
                ));
            }
        }
        
        // Validate sparsity threshold
        if self.sparsity_threshold < 0.0 || self.sparsity_threshold > 1.0 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Sparsity threshold {} is out of valid range [0, 1]", self.sparsity_threshold)
            ));
        }
        
        // Validate alignment
        if self.alignment == 0 || !self.alignment.is_power_of_two() || self.alignment > 128 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Alignment {} must be a power of 2 in range [1, 128]", self.alignment)
            ));
        }
        
        // Validate compression level
        if self.compression_level > 9 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Compression level {} is out of valid range [0, 9]", self.compression_level)
            ));
        }
        
        // Validate number of threads
        if let Some(threads) = self.num_threads {
            if threads == 0 || threads > 256 {
                return Err(ConfigValidationError::InvalidValue(
                    format!("Number of threads {threads} is out of valid range [1, 256]")
                ));
            }
        }
        
        // Validate SIMD configuration
        self.simd.validate()?;
        
        Ok(())
    }
}

/// SIMD optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    /// Whether to enable SIMD optimizations
    pub enabled: bool,
    /// Whether to force specific SIMD instruction sets
    pub force_sse2: bool,
    pub force_avx2: bool,
    pub force_neon: bool,
    /// Minimum data size to use SIMD (elements)
    pub min_simd_size: usize,
    /// SIMD chunk size for processing
    pub chunk_size: usize,
    /// Whether to use prefetching
    pub enable_prefetch: bool,
    /// Prefetch distance (cache lines)
    pub prefetch_distance: usize,
    /// Whether to enable vectorized operations
    pub vectorized_ops: bool,
    /// Custom SIMD parameters
    pub custom_params: HashMap<String, f32>,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            force_sse2: false,
            force_avx2: false,
            force_neon: false,
            min_simd_size: 64,
            chunk_size: 16,
            enable_prefetch: false,
            prefetch_distance: 8,
            vectorized_ops: true,
            custom_params: HashMap::new(),
        }
    }
}

impl SimdConfig {
    /// Create aggressive SIMD configuration for maximum performance
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            force_sse2: false,
            force_avx2: false,
            force_neon: false,
            min_simd_size: 32,
            chunk_size: 32,
            enable_prefetch: true,
            prefetch_distance: 16,
            vectorized_ops: true,
            custom_params: HashMap::new(),
        }
    }
    
    /// Create conservative SIMD configuration for compatibility
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            force_sse2: false,
            force_avx2: false,
            force_neon: false,
            min_simd_size: 128,
            chunk_size: 8,
            enable_prefetch: false,
            prefetch_distance: 4,
            vectorized_ops: false,
            custom_params: HashMap::new(),
        }
    }
    
    /// Disable SIMD optimizations
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
    
    /// Force specific SIMD instruction set
    pub fn force_instruction_set(mut self, sse2: bool, avx2: bool, neon: bool) -> Self {
        self.force_sse2 = sse2;
        self.force_avx2 = avx2;
        self.force_neon = neon;
        self
    }
    
    /// Set custom parameter
    pub fn with_custom_param(mut self, key: String, value: f32) -> Self {
        self.custom_params.insert(key, value);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate minimum SIMD size
        if self.min_simd_size == 0 || self.min_simd_size > 10000 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Minimum SIMD size {} is out of valid range [1, 10000]", self.min_simd_size)
            ));
        }
        
        // Validate chunk size
        if self.chunk_size == 0 || self.chunk_size > 1024 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Chunk size {} is out of valid range [1, 1024]", self.chunk_size)
            ));
        }
        
        // Validate prefetch distance
        if self.prefetch_distance == 0 || self.prefetch_distance > 64 {
            return Err(ConfigValidationError::InvalidValue(
                format!("Prefetch distance {} is out of valid range [1, 64]", self.prefetch_distance)
            ));
        }
        
        Ok(())
    }
}

/// Configuration validation errors
#[derive(Debug, Clone)]
pub enum ConfigValidationError {
    /// Invalid parameter value
    InvalidValue(String),
    /// Incompatible configuration settings
    IncompatibleSettings(String),
    /// Missing required configuration
    MissingRequired(String),
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValidationError::InvalidValue(msg) => write!(f, "Invalid configuration value: {msg}"),
            ConfigValidationError::IncompatibleSettings(msg) => write!(f, "Incompatible configuration: {msg}"),
            ConfigValidationError::MissingRequired(msg) => write!(f, "Missing required configuration: {msg}"),
        }
    }
}

impl std::error::Error for ConfigValidationError {}

/// Configuration builder for creating complex quantization configurations
#[derive(Debug, Default)]
pub struct QuantizationConfigBuilder {
    precision: Option<QuantizationPrecision>,
    strategy: Option<QuantizationStrategy>,
    per_channel: Option<bool>,
    clip_threshold: Option<f32>,
    qat_enabled: Option<bool>,
    calibration_size: Option<usize>,
    seed: Option<u64>,
    verbose: Option<bool>,
}

impl QuantizationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set quantization precision
    pub fn precision(mut self, precision: QuantizationPrecision) -> Self {
        self.precision = Some(precision);
        self
    }
    
    /// Set quantization strategy
    pub fn strategy(mut self, strategy: QuantizationStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }
    
    /// Enable per-channel quantization
    pub fn per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = Some(enabled);
        self
    }
    
    /// Set clipping threshold
    pub fn clip_threshold(mut self, threshold: f32) -> Self {
        self.clip_threshold = Some(threshold);
        self
    }
    
    /// Enable quantization-aware training
    pub fn qat_enabled(mut self, enabled: bool) -> Self {
        self.qat_enabled = Some(enabled);
        self
    }
    
    /// Set calibration size
    pub fn calibration_size(mut self, size: usize) -> Self {
        self.calibration_size = Some(size);
        self
    }
    
    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable verbose logging
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.verbose = Some(enabled);
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> QuantizationConfig {
        QuantizationConfig {
            precision: self.precision.unwrap_or(QuantizationPrecision::OneFiveFiveBit),
            strategy: self.strategy.unwrap_or(QuantizationStrategy::Symmetric),
            per_channel: self.per_channel.unwrap_or(false),
            clip_threshold: self.clip_threshold,
            qat_enabled: self.qat_enabled.unwrap_or(false),
            calibration_size: self.calibration_size,
            seed: self.seed,
            verbose: self.verbose.unwrap_or(false),
        }
    }
}

/// Weight quantization configuration builder
#[derive(Debug, Default)]
pub struct WeightQuantizationConfigBuilder {
    base: Option<QuantizationConfig>,
    group_size: Option<usize>,
    normalize_weights: Option<bool>,
    outlier_threshold: Option<f32>,
    learnable_scales: Option<bool>,
    block_size: Option<usize>,
    ternary_method: Option<TernaryMethod>,
    custom_threshold_factor: Option<f32>,
    packing: Option<PackingConfig>,
    freeze_weights: Option<bool>,
    weight_decay: Option<f32>,
    gradient_clip: Option<f32>,
}

impl WeightQuantizationConfigBuilder {
    /// Create a new weight quantization configuration builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set base configuration
    pub fn base(mut self, base: QuantizationConfig) -> Self {
        self.base = Some(base);
        self
    }
    
    /// Set group size
    pub fn group_size(mut self, size: usize) -> Self {
        self.group_size = Some(size);
        self
    }
    
    /// Enable weight normalization
    pub fn normalize_weights(mut self, enabled: bool) -> Self {
        self.normalize_weights = Some(enabled);
        self
    }
    
    /// Set outlier threshold
    pub fn outlier_threshold(mut self, threshold: f32) -> Self {
        self.outlier_threshold = Some(threshold);
        self
    }
    
    /// Enable learnable scales
    pub fn learnable_scales(mut self, enabled: bool) -> Self {
        self.learnable_scales = Some(enabled);
        self
    }
    
    /// Set block size
    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }
    
    /// Set ternary method
    pub fn ternary_method(mut self, method: TernaryMethod) -> Self {
        self.ternary_method = Some(method);
        self
    }
    
    /// Set custom threshold factor
    pub fn custom_threshold_factor(mut self, factor: f32) -> Self {
        self.custom_threshold_factor = Some(factor);
        self
    }
    
    /// Set packing configuration
    pub fn packing(mut self, packing: PackingConfig) -> Self {
        self.packing = Some(packing);
        self
    }
    
    /// Enable weight freezing
    pub fn freeze_weights(mut self, enabled: bool) -> Self {
        self.freeze_weights = Some(enabled);
        self
    }
    
    /// Set weight decay
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = Some(decay);
        self
    }
    
    /// Set gradient clipping
    pub fn gradient_clip(mut self, clip: f32) -> Self {
        self.gradient_clip = Some(clip);
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> WeightQuantizationConfig {
        WeightQuantizationConfig {
            base: self.base.unwrap_or_default(),
            group_size: self.group_size,
            normalize_weights: self.normalize_weights.unwrap_or(true),
            outlier_threshold: self.outlier_threshold.unwrap_or(3.0),
            learnable_scales: self.learnable_scales.unwrap_or(false),
            block_size: self.block_size,
            ternary_method: self.ternary_method.unwrap_or_default(),
            custom_threshold_factor: self.custom_threshold_factor,
            packing: self.packing.unwrap_or_default(),
            freeze_weights: self.freeze_weights.unwrap_or(false),
            weight_decay: self.weight_decay,
            gradient_clip: self.gradient_clip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.strategy, QuantizationStrategy::Symmetric);
        assert!(!config.per_channel);
        assert!(!config.qat_enabled);
        assert!(!config.verbose);
    }

    #[test]
    fn test_quantization_config_bitnet() {
        let config = QuantizationConfig::bitnet_158();
        assert_eq!(config.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.strategy, QuantizationStrategy::Symmetric);
        assert_eq!(config.clip_threshold, Some(3.0));
        assert_eq!(config.calibration_size, Some(1000));
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_quantization_config_validation() {
        let mut config = QuantizationConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid clipping threshold
        config.clip_threshold = Some(-1.0);
        assert!(config.validate().is_err());
        
        // Test invalid calibration size
        config.clip_threshold = None;
        config.calibration_size = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_weight_quantization_config_default() {
        let config = WeightQuantizationConfig::default();
        assert!(config.normalize_weights);
        assert_eq!(config.outlier_threshold, 3.0);
        assert!(!config.learnable_scales);
        assert_eq!(config.ternary_method, TernaryMethod::default());
    }

    #[test]
    fn test_weight_quantization_config_bitnet() {
        let config = WeightQuantizationConfig::bitnet();
        assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.ternary_method, TernaryMethod::MeanThreshold);
        assert_eq!(config.custom_threshold_factor, Some(0.7));
        assert_eq!(config.block_size, Some(64));
    }

    #[test]
    fn test_activation_quantization_config_default() {
        let config = ActivationQuantizationConfig::default();
        assert_eq!(config.moving_average_window, 100);
        assert_eq!(config.outlier_percentile, 99.9);
        assert!(!config.per_token);
        assert!(config.quantize_attention);
    }

    #[test]
    fn test_packing_config_default() {
        let config = PackingConfig::default();
        assert_eq!(config.strategy, TernaryPackingStrategy::default());
        assert_eq!(config.block_size, Some(64));
        assert_eq!(config.sparsity_threshold, 0.7);
        assert!(config.simd_optimized);
        assert_eq!(config.alignment, 16);
    }

    #[test]
    fn test_simd_config_default() {
        let config = SimdConfig::default();
        assert!(config.enabled);
        assert!(!config.force_sse2);
        assert!(!config.force_avx2);
        assert!(!config.force_neon);
        assert_eq!(config.min_simd_size, 64);
        assert_eq!(config.chunk_size, 16);
    }

    #[test]
    fn test_config_builder() {
        let config = QuantizationConfigBuilder::new()
            .precision(QuantizationPrecision::EightBit)
            .strategy(QuantizationStrategy::Dynamic)
            .per_channel(true)
            .clip_threshold(5.0)
            .qat_enabled(true)
            .verbose(true)
            .build();
        
        assert_eq!(config.precision, QuantizationPrecision::EightBit);
        assert_eq!(config.strategy, QuantizationStrategy::Dynamic);
        assert!(config.per_channel);
        assert_eq!(config.clip_threshold, Some(5.0));
        assert!(config.qat_enabled);
        assert!(config.verbose);
    }

    #[test]
    fn test_weight_config_builder() {
        let base_config = QuantizationConfig::bitnet_158();
        let config = WeightQuantizationConfigBuilder::new()
            .base(base_config)
            .group_size(128)
            .learnable_scales(true)
            .ternary_method(TernaryMethod::OptimalThreshold)
            .custom_threshold_factor(0.8)
            .build();
        
        assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.group_size, Some(128));
        assert!(config.learnable_scales);
        assert_eq!(config.ternary_method, TernaryMethod::OptimalThreshold);
        assert_eq!(config.custom_threshold_factor, Some(0.8));
    }

    #[test]
    fn test_config_validation_errors() {
        // Test invalid group size
        let mut config = WeightQuantizationConfig::default();
        config.group_size = Some(0);
        assert!(config.validate().is_err());
        
        // Test invalid outlier threshold
        config.group_size = None;
        config.outlier_threshold = -1.0;
        assert!(config.validate().is_err());
        
        // Test invalid threshold factor
        config.outlier_threshold = 3.0;
        config.custom_threshold_factor = Some(3.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_simd_config_validation() {
        let mut config = SimdConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid min SIMD size
        config.min_simd_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid chunk size
        config.min_simd_size = 64;
        config.chunk_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attention_config_validation() {
        let mut config = AttentionQuantizationConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid score clip threshold
        config.score_clip_threshold = Some(-1.0);
        assert!(config.validate().is_err());
        
        // Test invalid sparsity threshold
        config.score_clip_threshold = None;
        config.sparsity_threshold = Some(1.5);
        assert!(config.validate().is_err());
    }
}