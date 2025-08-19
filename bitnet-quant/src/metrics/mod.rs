// bitnet-quant/src/metrics/mod.rs
//! Quantization Metrics and Error Analysis Module
//! 
//! Provides comprehensive metrics and analysis tools for quantization quality assessment,
//! error measurement, and visualization capabilities for BitNet quantization workflows.

pub mod error_analysis;
pub mod mse;
pub mod sqnr;
pub mod cosine_similarity;
pub mod layer_wise;
pub mod visualization;
pub mod mitigation;
pub mod reporting;
// pub mod examples;  // Temporarily disabled due to compilation issues

use candle_core::{Tensor, Result, Device};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Core quantization metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetrics {
    pub mse: f32,
    pub sqnr: f32,
    pub cosine_similarity: f32,
    pub max_error: f32,
    pub mean_absolute_error: f32,
    pub relative_error: f32,
    pub bit_flip_ratio: f32,
    pub layer_name: String,
    pub timestamp: u64,
}

impl Default for QuantizationMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            sqnr: 0.0,
            cosine_similarity: 1.0,
            max_error: 0.0,
            mean_absolute_error: 0.0,
            relative_error: 0.0,
            bit_flip_ratio: 0.0,
            layer_name: String::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Layer-wise error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerErrorAnalysis {
    pub layer_metrics: HashMap<String, QuantizationMetrics>,
    pub global_metrics: QuantizationMetrics,
    pub sensitivity_ranking: Vec<(String, f32)>,
    pub error_distribution: Vec<f32>,
    pub recommended_bit_widths: HashMap<String, u8>,
}

/// Error threshold configuration for automated mitigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorThresholds {
    pub max_mse: f32,
    pub min_sqnr: f32,
    pub min_cosine_similarity: f32,
    pub max_relative_error: f32,
    pub max_bit_flip_ratio: f32,
}

impl Default for ErrorThresholds {
    fn default() -> Self {
        Self {
            max_mse: 1e-3,
            min_sqnr: 20.0, // dB
            min_cosine_similarity: 0.95,
            max_relative_error: 0.05,
            max_bit_flip_ratio: 0.1,
        }
    }
}

/// Error mitigation strategy configuration
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MitigationStrategy {
    IncreaseBitWidth,
    AdjustScaleFactor,
    UseAsymmetricQuantization,
    ApplyClipping,
    EnableMixedPrecision,
    AddRegularization,
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub collect_histograms: bool,
    pub enable_visualization: bool,
    pub real_time_monitoring: bool,
    pub memory_efficient_mode: bool,
    pub export_format: ExportFormat,
    pub thresholds: ErrorThresholds,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Binary,
    Tensorboard,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_histograms: true,
            enable_visualization: false,
            real_time_monitoring: true,
            memory_efficient_mode: true,
            export_format: ExportFormat::Json,
            thresholds: ErrorThresholds::default(),
        }
    }
}

/// Main metrics calculator trait
pub trait MetricsCalculator {
    /// Calculate comprehensive quantization metrics
    fn calculate_metrics(&self, original: &Tensor, quantized: &Tensor, layer_name: &str) -> Result<QuantizationMetrics>;
    
    /// Calculate layer-wise error analysis
    fn analyze_layer_errors(&self, layer_outputs: HashMap<String, (Tensor, Tensor)>) -> Result<LayerErrorAnalysis>;
    
    /// Check if metrics meet quality thresholds
    fn check_quality_thresholds(&self, metrics: &QuantizationMetrics, thresholds: &ErrorThresholds) -> bool;
    
    /// Suggest mitigation strategies for poor metrics
    fn suggest_mitigation(&self, metrics: &QuantizationMetrics, thresholds: &ErrorThresholds) -> Vec<MitigationStrategy>;
}

/// Utility functions for metrics calculation
pub fn safe_divide(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() < f32::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

pub fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    let flat = tensor.flatten_all()?;
    flat.to_vec1::<f32>()
}

pub fn calculate_percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let index = ((percentile / 100.0) * (sorted.len() - 1) as f32) as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Export metrics in various formats
pub trait MetricsExporter {
    fn export_json(&self, metrics: &LayerErrorAnalysis, path: &str) -> Result<()>;
    fn export_csv(&self, metrics: &LayerErrorAnalysis, path: &str) -> Result<()>;
    fn export_tensorboard(&self, metrics: &LayerErrorAnalysis, log_dir: &str, step: usize) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_safe_divide() {
        assert_eq!(safe_divide(10.0, 2.0), 5.0);
        assert_eq!(safe_divide(10.0, 0.0), 0.0);
        assert_eq!(safe_divide(10.0, 1e-10), 0.0);
    }

    #[test]
    fn test_calculate_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_percentile(&values, 50.0), 3.0);
        assert_eq!(calculate_percentile(&values, 90.0), 5.0);
        assert_eq!(calculate_percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_default_metrics() {
        let metrics = QuantizationMetrics::default();
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.cosine_similarity, 1.0);
        assert!(!metrics.layer_name.is_empty() == false);
    }

    #[test]
    fn test_default_thresholds() {
        let thresholds = ErrorThresholds::default();
        assert_eq!(thresholds.max_mse, 1e-3);
        assert_eq!(thresholds.min_sqnr, 20.0);
        assert_eq!(thresholds.min_cosine_similarity, 0.95);
    }
}
