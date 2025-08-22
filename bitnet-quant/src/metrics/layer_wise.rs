// bitnet-quant/src/metrics/layer_wise.rs
//! Layer-wise Error Analysis for Quantization Quality Assessment
//! 
//! Implements comprehensive layer-wise error analysis for neural network
//! quantization, providing detailed insights into layer sensitivity and optimization.

use candle_core::{Tensor, Result, Device};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::metrics::{
    QuantizationMetrics, ErrorThresholds, MitigationStrategy, MetricsCalculator,
    mse::MSECalculator, sqnr::SQNRCalculator, cosine_similarity::CosineSimilarityCalculator,
    error_analysis::ErrorAnalyzer, tensor_to_vec, safe_divide
};

/// Layer-wise quantization analysis engine
#[derive(Debug)]
pub struct LayerWiseAnalyzer {
    device: Device,
    mse_calculator: MSECalculator,
    sqnr_calculator: SQNRCalculator,
    cosine_calculator: CosineSimilarityCalculator,
    error_analyzer: ErrorAnalyzer,
    thresholds: ErrorThresholds,
    enable_detailed_analysis: bool,
}

impl LayerWiseAnalyzer {
    pub fn new(device: Device) -> Self {
        let mse_calculator = MSECalculator::new(device.clone());
        let sqnr_calculator = SQNRCalculator::new(device.clone());
        let cosine_calculator = CosineSimilarityCalculator::new(device.clone());
        let error_analyzer = ErrorAnalyzer::new(device.clone());
        
        Self {
            device,
            mse_calculator,
            sqnr_calculator,
            cosine_calculator,
            error_analyzer,
            thresholds: ErrorThresholds::default(),
            enable_detailed_analysis: true,
        }
    }

    pub fn with_thresholds(device: Device, thresholds: ErrorThresholds) -> Self {
        let mut analyzer = Self::new(device);
        analyzer.thresholds = thresholds;
        analyzer
    }

    /// Perform comprehensive layer-wise error analysis
    pub fn analyze_layers(&self, layer_data: HashMap<String, LayerData>) -> Result<LayerWiseAnalysisResult> {
        let mut layer_metrics = HashMap::new();
        let mut sensitivity_scores = Vec::new();
        let mut mitigation_recommendations = HashMap::new();
        let mut error_correlations = HashMap::new();

        // Analyze each layer individually
        for (layer_name, data) in layer_data.iter() {
            // Calculate comprehensive metrics for this layer
            let metrics = self.calculate_layer_metrics(layer_name, data)?;
            
            // Calculate sensitivity score
            let sensitivity = self.calculate_sensitivity_score(&metrics);
            sensitivity_scores.push((layer_name.clone(), sensitivity));
            
            // Generate mitigation recommendations
            let mitigations = self.error_analyzer.suggest_mitigation(&metrics, &self.thresholds);
            mitigation_recommendations.insert(layer_name.clone(), mitigations);
            
            layer_metrics.insert(layer_name.clone(), metrics);
        }

        // Sort layers by sensitivity (most sensitive first)
        sensitivity_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create layer rankings based on different metrics
        let layer_rankings = self.create_layer_rankings(&layer_metrics);
        
        // Calculate cross-layer error correlations
        if self.enable_detailed_analysis {
            error_correlations = self.calculate_error_correlations(&layer_data)?;
        }

        // Calculate global statistics
        let global_stats = self.calculate_global_statistics(&layer_metrics);
        
        // Generate optimization recommendations
        let optimization_plan = self.generate_optimization_plan(&layer_metrics, &sensitivity_scores);
        
        // Detect problematic layers
        let problematic_layers = self.identify_problematic_layers(&layer_metrics, &self.thresholds);

        Ok(LayerWiseAnalysisResult {
            layer_metrics,
            sensitivity_ranking: sensitivity_scores,
            layer_rankings,
            mitigation_recommendations,
            error_correlations,
            global_statistics: global_stats,
            optimization_plan,
            problematic_layers,
            analysis_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    fn calculate_layer_metrics(&self, layer_name: &str, data: &LayerData) -> Result<QuantizationMetrics> {
        let original = &data.original_output;
        let quantized = &data.quantized_output;
        
        // Calculate MSE
        let mse = self.mse_calculator.calculate(original, quantized)?;
        
        // Calculate SQNR
        let sqnr = self.sqnr_calculator.calculate(original, quantized)?;
        
        // Calculate cosine similarity
        let cosine_sim = self.cosine_calculator.calculate(original, quantized)?;
        
        // Calculate additional error metrics
        let error_stats = self.error_analyzer.calculate_error_statistics(original, quantized)?;
        
        Ok(QuantizationMetrics {
            mse,
            sqnr,
            cosine_similarity: cosine_sim,
            max_error: error_stats.max_error,
            mean_absolute_error: error_stats.mae,
            relative_error: error_stats.mean_relative_error,
            bit_flip_ratio: error_stats.bit_flip_ratio,
            layer_name: layer_name.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    fn calculate_sensitivity_score(&self, metrics: &QuantizationMetrics) -> f32 {
        // Weighted combination of different error metrics
        let mse_component = metrics.mse * 100.0; // MSE contribution
        let sqnr_component = (60.0 - metrics.sqnr.min(60.0)) / 10.0; // SQNR contribution (lower is worse)
        let cosine_component = (1.0 - metrics.cosine_similarity) * 50.0; // Cosine similarity contribution
        let max_error_component = metrics.max_error * 10.0; // Max error contribution
        let relative_error_component = metrics.relative_error * 20.0; // Relative error contribution
        
        // Weighted average (adjustable weights)
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // MSE, SQNR, Cosine, Max Error, Relative Error
        let components = [mse_component, sqnr_component, cosine_component, max_error_component, relative_error_component];
        
        weights.iter().zip(components.iter()).map(|(w, c)| w * c).sum()
    }

    fn create_layer_rankings(&self, layer_metrics: &HashMap<String, QuantizationMetrics>) -> Vec<LayerRanking> {
        let mut rankings = Vec::new();
        
        // MSE ranking (lower is better)
        let mut mse_ranking: Vec<_> = layer_metrics.iter()
            .map(|(name, metrics)| (name.clone(), metrics.mse))
            .collect();
        mse_ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings.push(LayerRanking {
            metric_name: "MSE".to_string(),
            layer_order: mse_ranking.into_iter().collect(),
            ascending: true, // Lower MSE is better
        });
        
        // SQNR ranking (higher is better)
        let mut sqnr_ranking: Vec<_> = layer_metrics.iter()
            .map(|(name, metrics)| (name.clone(), metrics.sqnr))
            .collect();
        sqnr_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings.push(LayerRanking {
            metric_name: "SQNR".to_string(),
            layer_order: sqnr_ranking.into_iter().collect(),
            ascending: false, // Higher SQNR is better
        });
        
        // Cosine similarity ranking (higher is better)
        let mut cosine_ranking: Vec<_> = layer_metrics.iter()
            .map(|(name, metrics)| (name.clone(), metrics.cosine_similarity))
            .collect();
        cosine_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings.push(LayerRanking {
            metric_name: "Cosine Similarity".to_string(),
            layer_order: cosine_ranking.into_iter().collect(),
            ascending: false, // Higher cosine similarity is better
        });
        
        rankings
    }

    fn calculate_error_correlations(&self, layer_data: &HashMap<String, LayerData>) -> Result<HashMap<String, HashMap<String, f32>>> {
        let mut correlations = HashMap::new();
        let layer_names: Vec<_> = layer_data.keys().cloned().collect();
        
        for layer1 in &layer_names {
            let mut layer_correlations = HashMap::new();
            
            for layer2 in &layer_names {
                if layer1 != layer2 {
                    let correlation = self.calculate_layer_correlation(
                        &layer_data[layer1],
                        &layer_data[layer2]
                    )?;
                    layer_correlations.insert(layer2.clone(), correlation);
                } else {
                    layer_correlations.insert(layer2.clone(), 1.0); // Self-correlation
                }
            }
            
            correlations.insert(layer1.clone(), layer_correlations);
        }
        
        Ok(correlations)
    }

    fn calculate_layer_correlation(&self, data1: &LayerData, data2: &LayerData) -> Result<f32> {
        // Calculate error vectors for both layers
        let error1 = data1.original_output.sub(&data1.quantized_output)?;
        let error2 = data2.original_output.sub(&data2.quantized_output)?;
        
        // Flatten error tensors
        let error1_vec = tensor_to_vec(&error1.flatten_all()?)?;
        let error2_vec = tensor_to_vec(&error2.flatten_all()?)?;
        
        // Ensure same length (use minimum length)
        let min_len = error1_vec.len().min(error2_vec.len());
        let error1_slice = &error1_vec[0..min_len];
        let error2_slice = &error2_vec[0..min_len];
        
        // Calculate Pearson correlation coefficient
        let mean1 = error1_slice.iter().sum::<f32>() / min_len as f32;
        let mean2 = error2_slice.iter().sum::<f32>() / min_len as f32;
        
        let numerator: f32 = error1_slice.iter().zip(error2_slice.iter())
            .map(|(&e1, &e2)| (e1 - mean1) * (e2 - mean2))
            .sum();
        
        let variance1: f32 = error1_slice.iter()
            .map(|&e| (e - mean1).powi(2))
            .sum();
        
        let variance2: f32 = error2_slice.iter()
            .map(|&e| (e - mean2).powi(2))
            .sum();
        
        let correlation = safe_divide(numerator, (variance1 * variance2).sqrt());
        Ok(correlation)
    }

    fn calculate_global_statistics(&self, layer_metrics: &HashMap<String, QuantizationMetrics>) -> GlobalStatistics {
        if layer_metrics.is_empty() {
            return GlobalStatistics::default();
        }

        let values: Vec<_> = layer_metrics.values().collect();
        let num_layers = values.len() as f32;
        
        // Calculate mean metrics
        let mean_mse = values.iter().map(|m| m.mse).sum::<f32>() / num_layers;
        let mean_sqnr = values.iter().map(|m| m.sqnr).filter(|&x| x.is_finite()).sum::<f32>() / num_layers;
        let mean_cosine_sim = values.iter().map(|m| m.cosine_similarity).sum::<f32>() / num_layers;
        let mean_relative_error = values.iter().map(|m| m.relative_error).sum::<f32>() / num_layers;
        
        // Calculate standard deviations
        let mse_variance = values.iter()
            .map(|m| (m.mse - mean_mse).powi(2))
            .sum::<f32>() / num_layers;
        let mse_std = mse_variance.sqrt();
        
        let sqnr_variance = values.iter()
            .filter(|m| m.sqnr.is_finite())
            .map(|m| (m.sqnr - mean_sqnr).powi(2))
            .sum::<f32>() / num_layers;
        let sqnr_std = sqnr_variance.sqrt();
        
        // Find best and worst layers
        let best_mse_layer = values.iter().min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap()).map(|m| m.layer_name.clone());
        let worst_mse_layer = values.iter().max_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap()).map(|m| m.layer_name.clone());
        
        let best_sqnr_layer = values.iter()
            .filter(|m| m.sqnr.is_finite())
            .max_by(|a, b| a.sqnr.partial_cmp(&b.sqnr).unwrap())
            .map(|m| m.layer_name.clone());
        
        GlobalStatistics {
            num_layers: layer_metrics.len(),
            mean_mse,
            mean_sqnr,
            mean_cosine_similarity: mean_cosine_sim,
            mean_relative_error,
            mse_std_dev: mse_std,
            sqnr_std_dev: sqnr_std,
            best_mse_layer,
            worst_mse_layer,
            best_sqnr_layer,
        }
    }

    fn generate_optimization_plan(&self, layer_metrics: &HashMap<String, QuantizationMetrics>, sensitivity_ranking: &[(String, f32)]) -> OptimizationPlan {
        let mut high_priority_layers = Vec::new();
        let mut medium_priority_layers = Vec::new();
        let mut low_priority_layers = Vec::new();
        let mut optimization_strategies = HashMap::new();
        
        for (layer_name, sensitivity) in sensitivity_ranking {
            let _metrics = &layer_metrics[layer_name];
            
            if sensitivity > &10.0 { // High sensitivity threshold
                high_priority_layers.push(layer_name.clone());
                optimization_strategies.insert(layer_name.clone(), vec![
                    OptimizationStrategy::IncreaseBitWidth,
                    OptimizationStrategy::UseAsymmetricQuantization,
                    OptimizationStrategy::ApplyAdvancedCalibration,
                ]);
            } else if sensitivity > &5.0 { // Medium sensitivity threshold
                medium_priority_layers.push(layer_name.clone());
                optimization_strategies.insert(layer_name.clone(), vec![
                    OptimizationStrategy::TuneScaleFactor,
                    OptimizationStrategy::ApplyChannelWiseQuantization,
                ]);
            } else { // Low sensitivity
                low_priority_layers.push(layer_name.clone());
                optimization_strategies.insert(layer_name.clone(), vec![
                    OptimizationStrategy::MaintainCurrentSettings,
                ]);
            }
        }
        
        // Calculate estimated improvement
        let total_layers = layer_metrics.len();
        let high_impact_ratio = high_priority_layers.len() as f32 / total_layers as f32;
        let estimated_improvement = if high_impact_ratio > 0.3 {
            EstimatedImprovement::High
        } else if high_impact_ratio > 0.1 {
            EstimatedImprovement::Medium
        } else {
            EstimatedImprovement::Low
        };
        
        OptimizationPlan {
            high_priority_layers,
            medium_priority_layers,
            low_priority_layers,
            optimization_strategies,
            estimated_improvement,
            implementation_complexity: self.assess_implementation_complexity(sensitivity_ranking),
        }
    }

    fn assess_implementation_complexity(&self, sensitivity_ranking: &[(String, f32)]) -> ImplementationComplexity {
        let high_sensitivity_count = sensitivity_ranking.iter()
            .filter(|(_, sensitivity)| sensitivity > &10.0)
            .count();
        
        if high_sensitivity_count > sensitivity_ranking.len() / 2 {
            ImplementationComplexity::High
        } else if high_sensitivity_count > sensitivity_ranking.len() / 4 {
            ImplementationComplexity::Medium
        } else {
            ImplementationComplexity::Low
        }
    }

    fn identify_problematic_layers(&self, layer_metrics: &HashMap<String, QuantizationMetrics>, thresholds: &ErrorThresholds) -> Vec<ProblematicLayer> {
        let mut problematic_layers = Vec::new();
        
        for (layer_name, metrics) in layer_metrics {
            let mut issues = Vec::new();
            
            if metrics.mse > thresholds.max_mse {
                issues.push(QualityIssue::HighMSE);
            }
            if metrics.sqnr < thresholds.min_sqnr && metrics.sqnr.is_finite() {
                issues.push(QualityIssue::LowSQNR);
            }
            if metrics.cosine_similarity < thresholds.min_cosine_similarity {
                issues.push(QualityIssue::PoorSimilarity);
            }
            if metrics.relative_error > thresholds.max_relative_error {
                issues.push(QualityIssue::HighRelativeError);
            }
            if metrics.bit_flip_ratio > thresholds.max_bit_flip_ratio {
                issues.push(QualityIssue::ExcessiveBitFlips);
            }
            
            if !issues.is_empty() {
                let severity = self.assess_issue_severity(&issues, metrics);
                let recommended_actions = self.get_recommended_actions(&issues);
                problematic_layers.push(ProblematicLayer {
                    layer_name: layer_name.clone(),
                    issues,
                    severity,
                    recommended_actions,
                });
            }
        }
        
        // Sort by severity (most severe first)
        problematic_layers.sort_by_key(|layer| match layer.severity {
            IssueSeverity::Critical => 0,
            IssueSeverity::High => 1,
            IssueSeverity::Medium => 2,
            IssueSeverity::Low => 3,
        });
        
        problematic_layers
    }

    fn assess_issue_severity(&self, issues: &[QualityIssue], _metrics: &QuantizationMetrics) -> IssueSeverity {
        let critical_issues = issues.iter().filter(|&&issue| matches!(issue, QualityIssue::HighMSE | QualityIssue::LowSQNR)).count();
        let high_issues = issues.iter().filter(|&&issue| matches!(issue, QualityIssue::PoorSimilarity | QualityIssue::HighRelativeError)).count();
        
        if critical_issues > 0 {
            IssueSeverity::Critical
        } else if high_issues > 1 {
            IssueSeverity::High
        } else if issues.len() > 2 {
            IssueSeverity::Medium
        } else {
            IssueSeverity::Low
        }
    }

    fn get_recommended_actions(&self, issues: &[QualityIssue]) -> Vec<String> {
        let mut actions = Vec::new();
        
        for issue in issues {
            match issue {
                QualityIssue::HighMSE => actions.push("Increase bit width or improve calibration".to_string()),
                QualityIssue::LowSQNR => actions.push("Use asymmetric quantization or add regularization".to_string()),
                QualityIssue::PoorSimilarity => actions.push("Enable mixed precision or adjust quantization method".to_string()),
                QualityIssue::HighRelativeError => actions.push("Apply channel-wise quantization or improve scale factors".to_string()),
                QualityIssue::ExcessiveBitFlips => actions.push("Increase bit width or use smoother quantization".to_string()),
            }
        }
        
        actions.dedup();
        actions
    }

    /// Perform temporal analysis of layer metrics across multiple time points
    pub fn analyze_temporal_trends(&self, temporal_data: HashMap<u64, HashMap<String, LayerData>>) -> Result<TemporalAnalysis> {
        let mut layer_trends = HashMap::new();
        let sorted_timestamps: Vec<_> = {
            let mut keys: Vec<_> = temporal_data.keys().copied().collect();
            keys.sort();
            keys
        };

        // Get layer names from first timestamp
        let layer_names = if let Some(first_data) = temporal_data.get(&sorted_timestamps[0]) {
            first_data.keys().cloned().collect::<Vec<_>>()
        } else {
            return Ok(TemporalAnalysis::default());
        };

        // Analyze trends for each layer
        for layer_name in layer_names {
            let mut metric_evolution = Vec::new();
            
            for &timestamp in &sorted_timestamps {
                if let Some(layer_data) = temporal_data.get(&timestamp).and_then(|data| data.get(&layer_name)) {
                    let metrics = self.calculate_layer_metrics(&layer_name, layer_data)?;
                    metric_evolution.push((timestamp, metrics));
                }
            }
            
            let trend_analysis = self.analyze_layer_trend(&metric_evolution);
            layer_trends.insert(layer_name, trend_analysis);
        }

        Ok(TemporalAnalysis {
            layer_trends,
            time_range: (sorted_timestamps[0], *sorted_timestamps.last().unwrap()),
            num_time_points: sorted_timestamps.len(),
        })
    }

    fn analyze_layer_trend(&self, evolution: &[(u64, QuantizationMetrics)]) -> LayerTrend {
        if evolution.len() < 2 {
            return LayerTrend::default();
        }

        let first_metrics = &evolution[0].1;
        let last_metrics = &evolution.last().unwrap().1;
        
        let mse_trend = if last_metrics.mse < first_metrics.mse * 0.9 {
            TrendDirection::Improving
        } else if last_metrics.mse > first_metrics.mse * 1.1 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        let sqnr_trend = if last_metrics.sqnr > first_metrics.sqnr + 1.0 {
            TrendDirection::Improving
        } else if last_metrics.sqnr < first_metrics.sqnr - 1.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        LayerTrend {
            mse_trend,
            sqnr_trend,
            overall_quality_change: self.assess_overall_quality_change(first_metrics, last_metrics),
            volatility: self.calculate_volatility(evolution),
        }
    }

    fn assess_overall_quality_change(&self, first: &QuantizationMetrics, last: &QuantizationMetrics) -> f32 {
        let mse_change = (first.mse - last.mse) / first.mse.max(1e-8);
        let sqnr_change = (last.sqnr - first.sqnr) / 60.0; // Normalize by typical good SQNR
        let cosine_change = last.cosine_similarity - first.cosine_similarity;
        
        // Weighted average of improvements (positive is better)
        mse_change * 0.4 + sqnr_change * 0.4 + cosine_change * 0.2
    }

    fn calculate_volatility(&self, evolution: &[(u64, QuantizationMetrics)]) -> f32 {
        if evolution.len() < 3 {
            return 0.0;
        }

        let mse_values: Vec<f32> = evolution.iter().map(|(_, m)| m.mse).collect();
        let mean_mse = mse_values.iter().sum::<f32>() / mse_values.len() as f32;
        let mse_variance = mse_values.iter().map(|&x| (x - mean_mse).powi(2)).sum::<f32>() / mse_values.len() as f32;
        
        mse_variance.sqrt() / mean_mse.max(1e-8)
    }
}

/// Input data for layer analysis
#[derive(Debug, Clone)]
pub struct LayerData {
    pub original_output: Tensor,
    pub quantized_output: Tensor,
    pub layer_type: LayerType,
    pub parameter_count: usize,
    pub activation_stats: Option<ActivationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Linear,
    Conv2d,
    BatchNorm,
    Attention,
    Embedding,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub min_value: f32,
    pub max_value: f32,
    pub mean_value: f32,
    pub std_dev: f32,
    pub sparsity: f32, // Ratio of near-zero values
}

/// Comprehensive layer-wise analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWiseAnalysisResult {
    pub layer_metrics: HashMap<String, QuantizationMetrics>,
    pub sensitivity_ranking: Vec<(String, f32)>,
    pub layer_rankings: Vec<LayerRanking>,
    pub mitigation_recommendations: HashMap<String, Vec<MitigationStrategy>>,
    pub error_correlations: HashMap<String, HashMap<String, f32>>,
    pub global_statistics: GlobalStatistics,
    pub optimization_plan: OptimizationPlan,
    pub problematic_layers: Vec<ProblematicLayer>,
    pub analysis_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRanking {
    pub metric_name: String,
    pub layer_order: Vec<(String, f32)>, // (layer_name, metric_value)
    pub ascending: bool, // true if lower values are better
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStatistics {
    pub num_layers: usize,
    pub mean_mse: f32,
    pub mean_sqnr: f32,
    pub mean_cosine_similarity: f32,
    pub mean_relative_error: f32,
    pub mse_std_dev: f32,
    pub sqnr_std_dev: f32,
    pub best_mse_layer: Option<String>,
    pub worst_mse_layer: Option<String>,
    pub best_sqnr_layer: Option<String>,
}

impl Default for GlobalStatistics {
    fn default() -> Self {
        Self {
            num_layers: 0,
            mean_mse: 0.0,
            mean_sqnr: 0.0,
            mean_cosine_similarity: 0.0,
            mean_relative_error: 0.0,
            mse_std_dev: 0.0,
            sqnr_std_dev: 0.0,
            best_mse_layer: None,
            worst_mse_layer: None,
            best_sqnr_layer: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPlan {
    pub high_priority_layers: Vec<String>,
    pub medium_priority_layers: Vec<String>,
    pub low_priority_layers: Vec<String>,
    pub optimization_strategies: HashMap<String, Vec<OptimizationStrategy>>,
    pub estimated_improvement: EstimatedImprovement,
    pub implementation_complexity: ImplementationComplexity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    IncreaseBitWidth,
    UseAsymmetricQuantization,
    ApplyAdvancedCalibration,
    TuneScaleFactor,
    ApplyChannelWiseQuantization,
    MaintainCurrentSettings,
    EnableMixedPrecision,
    AddRegularization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EstimatedImprovement {
    High,    // >20% improvement expected
    Medium,  // 5-20% improvement expected
    Low,     // <5% improvement expected
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    High,    // Significant changes required
    Medium,  // Moderate changes required
    Low,     // Minor changes required
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicLayer {
    pub layer_name: String,
    pub issues: Vec<QualityIssue>,
    pub severity: IssueSeverity,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QualityIssue {
    HighMSE,
    LowSQNR,
    PoorSimilarity,
    HighRelativeError,
    ExcessiveBitFlips,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Temporal analysis of layer metrics
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct TemporalAnalysis {
    pub layer_trends: HashMap<String, LayerTrend>,
    pub time_range: (u64, u64), // (start_timestamp, end_timestamp)
    pub num_time_points: usize,
}


#[derive(Debug, Clone)]
pub struct LayerTrend {
    pub mse_trend: TrendDirection,
    pub sqnr_trend: TrendDirection,
    pub overall_quality_change: f32, // Positive indicates improvement
    pub volatility: f32, // Higher values indicate more unstable metrics
}

impl Default for LayerTrend {
    fn default() -> Self {
        Self {
            mse_trend: TrendDirection::Stable,
            sqnr_trend: TrendDirection::Stable,
            overall_quality_change: 0.0,
            volatility: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    fn create_test_layer_data() -> Result<HashMap<String, LayerData>> {
        let device = Device::Cpu;
        let mut layer_data = HashMap::new();
        
        // Create test data for layer1 (good quantization)
        let original1 = Tensor::ones((4, 4), DType::F32, &device)?;
        let quantized1 = original1.mul(&Tensor::new(&[0.99f32], &device)?)?;
        
        layer_data.insert("layer1".to_string(), LayerData {
            original_output: original1,
            quantized_output: quantized1,
            layer_type: LayerType::Linear,
            parameter_count: 16,
            activation_stats: None,
        });
        
        // Create test data for layer2 (poor quantization)
        let original2 = Tensor::ones((4, 4), DType::F32, &device)?;
        let quantized2 = original2.mul(&Tensor::new(&[0.7f32], &device)?)?;
        
        layer_data.insert("layer2".to_string(), LayerData {
            original_output: original2,
            quantized_output: quantized2,
            layer_type: LayerType::Conv2d,
            parameter_count: 32,
            activation_stats: None,
        });
        
        Ok(layer_data)
    }

    #[test]
    fn test_layer_wise_analyzer_creation() {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        assert!(analyzer.enable_detailed_analysis);
    }

    #[test]
    fn test_sensitivity_score_calculation() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        
        let metrics = QuantizationMetrics {
            mse: 0.1,
            sqnr: 20.0,
            cosine_similarity: 0.9,
            max_error: 0.5,
            mean_absolute_error: 0.05,
            relative_error: 0.1,
            bit_flip_ratio: 0.05,
            layer_name: "test_layer".to_string(),
            timestamp: 0,
        };
        
        let sensitivity = analyzer.calculate_sensitivity_score(&metrics);
        assert!(sensitivity > 0.0);
        Ok(())
    }

    #[test]
    fn test_layer_wise_analysis() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        let layer_data = create_test_layer_data()?;
        
        let result = analyzer.analyze_layers(layer_data)?;
        
        assert_eq!(result.layer_metrics.len(), 2);
        assert_eq!(result.sensitivity_ranking.len(), 2);
        assert!(result.global_statistics.num_layers == 2);
        assert!(!result.layer_rankings.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_problematic_layer_identification() -> Result<()> {
        let device = Device::Cpu;
        let mut thresholds = ErrorThresholds::default();
        thresholds.max_mse = 0.001; // Very strict threshold
        
        let analyzer = LayerWiseAnalyzer::with_thresholds(device, thresholds);
        let layer_data = create_test_layer_data()?;
        
        let result = analyzer.analyze_layers(layer_data)?;
        
        // Should identify problematic layers due to strict thresholds
        assert!(!result.problematic_layers.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_optimization_plan_generation() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        let layer_data = create_test_layer_data()?;
        
        let result = analyzer.analyze_layers(layer_data)?;
        
        let total_layers = result.optimization_plan.high_priority_layers.len() +
                          result.optimization_plan.medium_priority_layers.len() +
                          result.optimization_plan.low_priority_layers.len();
        
        assert_eq!(total_layers, 2); // Should account for all layers
        assert!(!result.optimization_plan.optimization_strategies.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_layer_correlation_calculation() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device.clone());
        
        let original1 = Tensor::ones((4, 4), DType::F32, &device)?;
        let data1_vec = vec![0.9f32];
        let quantized1 = original1.mul(&Tensor::new(data1_vec.as_slice(), &device)?)?;
        
        let original2 = Tensor::ones((4, 4), DType::F32, &device)?;
        let data2_vec = vec![0.8f32];
        let quantized2 = original2.mul(&Tensor::new(data2_vec.as_slice(), &device)?)?;
        
        let data1 = LayerData {
            original_output: original1,
            quantized_output: quantized1,
            layer_type: LayerType::Linear,
            parameter_count: 16,
            activation_stats: None,
        };
        
        let data2 = LayerData {
            original_output: original2,
            quantized_output: quantized2,
            layer_type: LayerType::Linear,
            parameter_count: 16,
            activation_stats: None,
        };
        
        let correlation = analyzer.calculate_layer_correlation(&data1, &data2)?;
        assert!(correlation.is_finite());
        
        Ok(())
    }

    #[test]
    fn test_temporal_analysis() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        
        let mut temporal_data = HashMap::new();
        
        // Create data for two time points
        for timestamp in [1000, 2000] {
            let layer_data = create_test_layer_data()?;
            temporal_data.insert(timestamp, layer_data);
        }
        
        let temporal_analysis = analyzer.analyze_temporal_trends(temporal_data)?;
        
        assert_eq!(temporal_analysis.num_time_points, 2);
        assert_eq!(temporal_analysis.layer_trends.len(), 2);
        assert_eq!(temporal_analysis.time_range, (1000, 2000));
        
        Ok(())
    }

    #[test]
    fn test_layer_trend_analysis() {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        
        let evolution = vec![
            (1000, QuantizationMetrics {
                mse: 1.0,
                sqnr: 10.0,
                cosine_similarity: 0.8,
                layer_name: "test".to_string(),
                ..QuantizationMetrics::default()
            }),
            (2000, QuantizationMetrics {
                mse: 0.5, // Improved (lower MSE)
                sqnr: 20.0, // Improved (higher SQNR)
                cosine_similarity: 0.9,
                layer_name: "test".to_string(),
                ..QuantizationMetrics::default()
            }),
        ];
        
        let trend = analyzer.analyze_layer_trend(&evolution);
        assert_eq!(trend.mse_trend, TrendDirection::Improving);
        assert_eq!(trend.sqnr_trend, TrendDirection::Improving);
        assert!(trend.overall_quality_change > 0.0); // Positive indicates improvement
    }

    #[test]
    fn test_issue_severity_assessment() {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        
        let critical_issues = vec![QualityIssue::HighMSE, QualityIssue::LowSQNR];
        let metrics = QuantizationMetrics::default();
        
        let severity = analyzer.assess_issue_severity(&critical_issues, &metrics);
        assert_eq!(severity, IssueSeverity::Critical);
        
        let low_issues = vec![QualityIssue::ExcessiveBitFlips];
        let severity = analyzer.assess_issue_severity(&low_issues, &metrics);
        assert_eq!(severity, IssueSeverity::Low);
    }

    #[test]
    fn test_global_statistics_calculation() -> Result<()> {
        let device = Device::Cpu;
        let analyzer = LayerWiseAnalyzer::new(device);
        
        let mut layer_metrics = HashMap::new();
        layer_metrics.insert("layer1".to_string(), QuantizationMetrics {
            mse: 0.1,
            sqnr: 30.0,
            layer_name: "layer1".to_string(),
            ..QuantizationMetrics::default()
        });
        layer_metrics.insert("layer2".to_string(), QuantizationMetrics {
            mse: 0.2,
            sqnr: 20.0,
            layer_name: "layer2".to_string(),
            ..QuantizationMetrics::default()
        });
        
        let stats = analyzer.calculate_global_statistics(&layer_metrics);
        
        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.mean_mse, 0.15);
        assert_eq!(stats.mean_sqnr, 25.0);
        assert_eq!(stats.best_mse_layer, Some("layer1".to_string()));
        assert_eq!(stats.worst_mse_layer, Some("layer2".to_string()));
        
        Ok(())
    }
}
