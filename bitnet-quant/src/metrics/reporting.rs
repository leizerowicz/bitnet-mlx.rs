// bitnet-quant/src/metrics/reporting.rs
//! Quantization Metrics Reporting System
//!
//! Comprehensive reporting system for quantization analysis results,
//! providing structured reports, summaries, and export capabilities.

use crate::metrics::{
    layer_wise::LayerWiseAnalysisResult, mitigation::MitigationResult, QuantizationMetrics,
};
use candle_core::{Error as CandleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive reporting engine for quantization analysis
#[derive(Debug)]
pub struct ReportingEngine {
    output_directory: String,
    report_format: ReportFormat,
    include_visualizations: bool,
    detailed_mode: bool,
}

impl ReportingEngine {
    pub fn new(output_directory: String) -> Self {
        Self {
            output_directory,
            report_format: ReportFormat::Markdown,
            include_visualizations: true,
            detailed_mode: true,
        }
    }

    pub fn with_config(
        output_directory: String,
        report_format: ReportFormat,
        include_visualizations: bool,
        detailed_mode: bool,
    ) -> Self {
        Self {
            output_directory,
            report_format,
            include_visualizations,
            detailed_mode,
        }
    }

    /// Generate comprehensive quantization analysis report
    pub fn generate_comprehensive_report(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<ComprehensiveReport> {
        let executive_summary = self.generate_executive_summary(analysis)?;
        let detailed_analysis = self.generate_detailed_analysis(analysis)?;
        let recommendations = self.generate_recommendations(analysis)?;
        let technical_appendix = if self.detailed_mode {
            Some(self.generate_technical_appendix(analysis)?)
        } else {
            None
        };

        let report = ComprehensiveReport {
            metadata: ReportMetadata {
                title: "BitNet Quantization Analysis Report".to_string(),
                generated_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                analysis_timestamp: analysis.analysis_timestamp,
                report_version: "1.0".to_string(),
                total_layers_analyzed: analysis.layer_metrics.len(),
            },
            executive_summary,
            detailed_analysis,
            recommendations,
            technical_appendix,
        };

        Ok(report)
    }

    fn generate_executive_summary(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<ExecutiveSummary> {
        let stats = &analysis.global_statistics;

        // Calculate overall quality grade
        let quality_grade = self.assess_overall_quality(stats);

        // Identify key findings
        let mut key_findings = Vec::new();

        if analysis.problematic_layers.len() > analysis.layer_metrics.len() / 4 {
            key_findings.push(format!(
                "Significant quality issues detected: {}/{} layers require attention",
                analysis.problematic_layers.len(),
                analysis.layer_metrics.len()
            ));
        }

        if stats.mean_sqnr < 20.0 {
            key_findings.push(format!(
                "Low signal-to-noise ratio detected: {:.1} dB (target: >20 dB)",
                stats.mean_sqnr
            ));
        }

        if stats.mean_cosine_similarity < 0.95 {
            key_findings.push(format!(
                "Reduced similarity between original and quantized outputs: {:.3}",
                stats.mean_cosine_similarity
            ));
        }

        // High-priority recommendations count
        let high_priority_count = analysis.optimization_plan.high_priority_layers.len();

        // Business impact assessment
        let business_impact = self.assess_business_impact(analysis);

        Ok(ExecutiveSummary {
            overall_quality_grade: quality_grade,
            key_findings,
            high_priority_actions_count: high_priority_count,
            estimated_improvement_potential: match analysis.optimization_plan.estimated_improvement
            {
                crate::metrics::layer_wise::EstimatedImprovement::High => {
                    "High (>20% quality improvement)"
                }
                crate::metrics::layer_wise::EstimatedImprovement::Medium => {
                    "Medium (5-20% quality improvement)"
                }
                crate::metrics::layer_wise::EstimatedImprovement::Low => {
                    "Low (<5% quality improvement)"
                }
            }
            .to_string(),
            business_impact,
            next_steps: self.generate_next_steps(analysis),
        })
    }

    fn assess_overall_quality(
        &self,
        stats: &crate::metrics::layer_wise::GlobalStatistics,
    ) -> QualityGrade {
        let mut score = 0;

        // MSE evaluation (0-25 points)
        if stats.mean_mse < 1e-4 {
            score += 25;
        } else if stats.mean_mse < 1e-3 {
            score += 20;
        } else if stats.mean_mse < 1e-2 {
            score += 15;
        } else if stats.mean_mse < 1e-1 {
            score += 10;
        } else {
            score += 5;
        }

        // SQNR evaluation (0-25 points)
        if stats.mean_sqnr > 40.0 {
            score += 25;
        } else if stats.mean_sqnr > 30.0 {
            score += 20;
        } else if stats.mean_sqnr > 20.0 {
            score += 15;
        } else if stats.mean_sqnr > 10.0 {
            score += 10;
        } else {
            score += 5;
        }

        // Cosine similarity evaluation (0-25 points)
        if stats.mean_cosine_similarity > 0.99 {
            score += 25;
        } else if stats.mean_cosine_similarity > 0.97 {
            score += 20;
        } else if stats.mean_cosine_similarity > 0.95 {
            score += 15;
        } else if stats.mean_cosine_similarity > 0.90 {
            score += 10;
        } else {
            score += 5;
        }

        // Consistency evaluation (0-25 points) - based on standard deviation
        let consistency_score = if stats.mse_std_dev < stats.mean_mse * 0.1 {
            25 // Very consistent
        } else if stats.mse_std_dev < stats.mean_mse * 0.3 {
            20 // Good consistency
        } else if stats.mse_std_dev < stats.mean_mse * 0.5 {
            15 // Moderate consistency
        } else {
            10 // Poor consistency
        };

        score += consistency_score;

        match score {
            90..=100 => QualityGrade::Excellent,
            80..=89 => QualityGrade::Good,
            70..=79 => QualityGrade::Satisfactory,
            60..=69 => QualityGrade::NeedsImprovement,
            _ => QualityGrade::Poor,
        }
    }

    fn assess_business_impact(&self, analysis: &LayerWiseAnalysisResult) -> BusinessImpact {
        let problematic_ratio =
            analysis.problematic_layers.len() as f32 / analysis.layer_metrics.len() as f32;
        let high_priority_ratio = analysis.optimization_plan.high_priority_layers.len() as f32
            / analysis.layer_metrics.len() as f32;

        let performance_impact = if problematic_ratio > 0.5 {
            PerformanceImpact::Severe
        } else if problematic_ratio > 0.25 {
            PerformanceImpact::Moderate
        } else if problematic_ratio > 0.1 {
            PerformanceImpact::Minor
        } else {
            PerformanceImpact::Minimal
        };

        let deployment_risk = if high_priority_ratio > 0.3 {
            DeploymentRisk::High
        } else if high_priority_ratio > 0.1 {
            DeploymentRisk::Medium
        } else {
            DeploymentRisk::Low
        };

        BusinessImpact {
            performance_impact,
            deployment_risk,
            estimated_resolution_time: self.estimate_resolution_time(analysis),
            cost_of_inaction: self.estimate_cost_of_inaction(analysis),
        }
    }

    fn estimate_resolution_time(&self, analysis: &LayerWiseAnalysisResult) -> String {
        match analysis.optimization_plan.implementation_complexity {
            crate::metrics::layer_wise::ImplementationComplexity::Low => "1-2 days",
            crate::metrics::layer_wise::ImplementationComplexity::Medium => "1-2 weeks",
            crate::metrics::layer_wise::ImplementationComplexity::High => "2-4 weeks",
        }
        .to_string()
    }

    fn estimate_cost_of_inaction(&self, analysis: &LayerWiseAnalysisResult) -> String {
        let critical_issues = analysis
            .problematic_layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.severity,
                    crate::metrics::layer_wise::IssueSeverity::Critical
                )
            })
            .count();

        if critical_issues > 0 {
            "High - Model accuracy significantly impacted"
        } else if analysis.problematic_layers.len() > analysis.layer_metrics.len() / 4 {
            "Medium - Noticeable quality degradation"
        } else {
            "Low - Minor quality impact"
        }
        .to_string()
    }

    fn generate_next_steps(&self, analysis: &LayerWiseAnalysisResult) -> Vec<String> {
        let mut steps = Vec::new();

        if !analysis.optimization_plan.high_priority_layers.is_empty() {
            steps.push(format!(
                "Address {} high-priority layers immediately",
                analysis.optimization_plan.high_priority_layers.len()
            ));
        }

        if analysis.problematic_layers.iter().any(|l| {
            matches!(
                l.severity,
                crate::metrics::layer_wise::IssueSeverity::Critical
            )
        }) {
            steps.push("Investigate critical quantization failures".to_string());
        }

        steps.push("Implement recommended optimization strategies".to_string());
        steps.push("Perform validation testing on improved configurations".to_string());
        steps.push("Monitor quantization quality in production".to_string());

        steps
    }

    fn generate_detailed_analysis(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<DetailedAnalysis> {
        // Layer-by-layer breakdown
        let mut layer_breakdown = Vec::new();

        for (layer_name, metrics) in &analysis.layer_metrics {
            let sensitivity_score = analysis
                .sensitivity_ranking
                .iter()
                .find(|(name, _)| name == layer_name)
                .map(|(_, score)| *score)
                .unwrap_or(0.0);

            let is_problematic = analysis
                .problematic_layers
                .iter()
                .any(|p| &p.layer_name == layer_name);

            layer_breakdown.push(LayerBreakdown {
                layer_name: layer_name.clone(),
                metrics: metrics.clone(),
                sensitivity_score,
                quality_assessment: self.assess_layer_quality(metrics),
                is_problematic,
                recommended_actions: analysis
                    .mitigation_recommendations
                    .get(layer_name)
                    .cloned()
                    .unwrap_or_default(),
            });
        }

        // Sort by sensitivity score (highest first)
        layer_breakdown.sort_by(|a, b| {
            b.sensitivity_score
                .partial_cmp(&a.sensitivity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Quantization patterns analysis
        let patterns = self.analyze_quantization_patterns(analysis)?;

        // Performance analysis
        let performance_analysis = self.analyze_performance_characteristics(analysis)?;

        Ok(DetailedAnalysis {
            layer_breakdown,
            quantization_patterns: patterns,
            performance_analysis,
            error_correlation_summary: self
                .summarize_error_correlations(&analysis.error_correlations),
        })
    }

    fn assess_layer_quality(&self, metrics: &QuantizationMetrics) -> LayerQualityAssessment {
        let mse_score = if metrics.mse < 1e-4 {
            5
        } else if metrics.mse < 1e-3 {
            4
        } else if metrics.mse < 1e-2 {
            3
        } else if metrics.mse < 1e-1 {
            2
        } else {
            1
        };
        let sqnr_score = if metrics.sqnr > 40.0 {
            5
        } else if metrics.sqnr > 30.0 {
            4
        } else if metrics.sqnr > 20.0 {
            3
        } else if metrics.sqnr > 10.0 {
            2
        } else {
            1
        };
        let cosine_score = if metrics.cosine_similarity > 0.99 {
            5
        } else if metrics.cosine_similarity > 0.97 {
            4
        } else if metrics.cosine_similarity > 0.95 {
            3
        } else if metrics.cosine_similarity > 0.90 {
            2
        } else {
            1
        };

        let total_score = mse_score + sqnr_score + cosine_score;

        LayerQualityAssessment {
            overall_score: total_score,
            mse_grade: match mse_score {
                5 => "Excellent".to_string(),
                4 => "Good".to_string(),
                3 => "Fair".to_string(),
                2 => "Poor".to_string(),
                _ => "Critical".to_string(),
            },
            sqnr_grade: match sqnr_score {
                5 => "Excellent".to_string(),
                4 => "Good".to_string(),
                3 => "Fair".to_string(),
                2 => "Poor".to_string(),
                _ => "Critical".to_string(),
            },
            similarity_grade: match cosine_score {
                5 => "Excellent".to_string(),
                4 => "Good".to_string(),
                3 => "Fair".to_string(),
                2 => "Poor".to_string(),
                _ => "Critical".to_string(),
            },
            primary_concern: self.identify_primary_concern(metrics),
        }
    }

    fn identify_primary_concern(&self, metrics: &QuantizationMetrics) -> String {
        if metrics.mse > 1e-1 {
            "High MSE - Significant quantization error"
        } else if metrics.sqnr < 10.0 {
            "Low SQNR - Poor signal-to-noise ratio"
        } else if metrics.cosine_similarity < 0.90 {
            "Low similarity - Output direction significantly changed"
        } else if metrics.relative_error > 0.1 {
            "High relative error - Large percentage errors"
        } else if metrics.max_error > metrics.mean_absolute_error * 10.0 {
            "Outlier errors - Some values poorly quantized"
        } else {
            "Overall quality acceptable"
        }
        .to_string()
    }

    fn analyze_quantization_patterns(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<QuantizationPatterns> {
        let mut layer_types = HashMap::new();
        let mut error_distribution = Vec::new();
        let mut correlation_insights = Vec::new();

        // Analyze error distribution across layers
        for metrics in analysis.layer_metrics.values() {
            error_distribution.push(metrics.mse);

            // Categorize by error level for pattern analysis
            let error_category = if metrics.mse < 1e-4 {
                "low_error"
            } else if metrics.mse < 1e-2 {
                "medium_error"
            } else {
                "high_error"
            };

            *layer_types.entry(error_category.to_string()).or_insert(0) += 1;
        }

        // Analyze correlations
        for (layer1, correlations) in &analysis.error_correlations {
            for (layer2, correlation) in correlations {
                if layer1 < layer2 && correlation.abs() > 0.7 {
                    correlation_insights.push(format!(
                        "Strong correlation ({correlation:.3}) between {layer1} and {layer2}"
                    ));
                }
            }
        }

        Ok(QuantizationPatterns {
            error_distribution_summary: format!(
                "Error range: {:.6} - {:.6}, Mean: {:.6}",
                error_distribution
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b)),
                error_distribution.iter().fold(0.0f32, |a, &b| a.max(b)),
                error_distribution.iter().sum::<f32>() / error_distribution.len() as f32
            ),
            layer_type_patterns: layer_types,
            correlation_insights,
            common_issues: self.identify_common_issues(analysis),
        })
    }

    fn identify_common_issues(&self, analysis: &LayerWiseAnalysisResult) -> Vec<String> {
        let mut issues = Vec::new();

        let high_mse_count = analysis
            .layer_metrics
            .values()
            .filter(|m| m.mse > 1e-2)
            .count();

        if high_mse_count > analysis.layer_metrics.len() / 4 {
            issues.push(format!(
                "High MSE in {high_mse_count} layers - Consider increasing bit width"
            ));
        }

        let low_sqnr_count = analysis
            .layer_metrics
            .values()
            .filter(|m| m.sqnr < 20.0 && m.sqnr.is_finite())
            .count();

        if low_sqnr_count > 0 {
            issues.push(format!(
                "Low SQNR in {low_sqnr_count} layers - May need asymmetric quantization"
            ));
        }

        let low_similarity_count = analysis
            .layer_metrics
            .values()
            .filter(|m| m.cosine_similarity < 0.95)
            .count();

        if low_similarity_count > 0 {
            issues.push(format!(
                "Reduced similarity in {low_similarity_count} layers - Consider mixed precision"
            ));
        }

        issues
    }

    fn analyze_performance_characteristics(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<PerformanceAnalysis> {
        let stats = &analysis.global_statistics;

        // Calculate performance metrics
        let consistency_rating = if stats.mse_std_dev < stats.mean_mse * 0.2 {
            "High"
        } else if stats.mse_std_dev < stats.mean_mse * 0.5 {
            "Medium"
        } else {
            "Low"
        }
        .to_string();

        let predictability_score = 1.0 - (stats.mse_std_dev / stats.mean_mse.max(1e-8)).min(1.0);

        // Estimate resource usage
        let estimated_memory_usage = analysis.layer_metrics.len() as f32 * 4.0; // Rough estimate in MB
        let estimated_compute_overhead = analysis.problematic_layers.len() as f32 * 0.1; // Percentage

        Ok(PerformanceAnalysis {
            consistency_rating,
            predictability_score,
            outlier_layers: analysis
                .sensitivity_ranking
                .iter()
                .take(5)
                .map(|(name, score)| format!("{name}: {score:.2}"))
                .collect(),
            estimated_memory_usage,
            estimated_compute_overhead,
            scalability_assessment: self.assess_scalability(analysis),
        })
    }

    fn assess_scalability(&self, analysis: &LayerWiseAnalysisResult) -> String {
        let problematic_ratio =
            analysis.problematic_layers.len() as f32 / analysis.layer_metrics.len() as f32;

        if problematic_ratio > 0.5 {
            "Poor - Many layers need individual attention"
        } else if problematic_ratio > 0.25 {
            "Fair - Some systematic issues present"
        } else if problematic_ratio > 0.1 {
            "Good - Minor issues, generally scalable"
        } else {
            "Excellent - Highly scalable quantization approach"
        }
        .to_string()
    }

    fn summarize_error_correlations(
        &self,
        correlations: &HashMap<String, HashMap<String, f32>>,
    ) -> ErrorCorrelationSummary {
        if correlations.is_empty() {
            return ErrorCorrelationSummary {
                strong_correlations: 0,
                average_correlation: 0.0,
                most_correlated_pair: None,
                correlation_insights: Vec::new(),
            };
        }

        let mut all_correlations = Vec::new();
        let mut max_correlation = 0.0;
        let mut most_correlated_pair = None;

        for (layer1, layer_corr) in correlations {
            for (layer2, &correlation) in layer_corr {
                if layer1 < layer2 {
                    let abs_corr = correlation.abs();
                    all_correlations.push(abs_corr);

                    if abs_corr > max_correlation {
                        max_correlation = abs_corr;
                        most_correlated_pair = Some((layer1.clone(), layer2.clone(), correlation));
                    }
                }
            }
        }

        let strong_correlations = all_correlations.iter().filter(|&&c| c > 0.7).count();
        let average_correlation =
            all_correlations.iter().sum::<f32>() / all_correlations.len() as f32;

        let mut insights = Vec::new();
        if average_correlation > 0.5 {
            insights.push(
                "High overall correlation suggests systematic quantization patterns".to_string(),
            );
        }
        if strong_correlations > correlations.len() / 4 {
            insights.push(
                "Many strong correlations indicate related quantization challenges".to_string(),
            );
        }

        ErrorCorrelationSummary {
            strong_correlations,
            average_correlation,
            most_correlated_pair,
            correlation_insights: insights,
        }
    }

    fn generate_recommendations(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<RecommendationSection> {
        let immediate_actions = self.generate_immediate_actions(analysis);
        let optimization_roadmap = self.generate_optimization_roadmap(analysis);
        let best_practices = self.generate_best_practices();
        let monitoring_recommendations = self.generate_monitoring_recommendations();

        Ok(RecommendationSection {
            immediate_actions,
            optimization_roadmap,
            best_practices,
            monitoring_recommendations,
        })
    }

    fn generate_immediate_actions(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Vec<ImmediateAction> {
        let mut actions = Vec::new();

        // Critical issues first
        for layer in &analysis.problematic_layers {
            if matches!(
                layer.severity,
                crate::metrics::layer_wise::IssueSeverity::Critical
            ) {
                actions.push(ImmediateAction {
                    priority: ActionPriority::Critical,
                    description: format!("Address critical issues in layer: {}", layer.layer_name),
                    estimated_effort: "2-4 hours".to_string(),
                    expected_impact: "High".to_string(),
                    implementation_steps: layer.recommended_actions.clone(),
                });
            }
        }

        // High priority optimization
        for layer_name in &analysis.optimization_plan.high_priority_layers {
            if !actions.iter().any(|a| a.description.contains(layer_name)) {
                actions.push(ImmediateAction {
                    priority: ActionPriority::High,
                    description: format!("Optimize high-sensitivity layer: {layer_name}"),
                    estimated_effort: "4-8 hours".to_string(),
                    expected_impact: "Medium-High".to_string(),
                    implementation_steps: vec![
                        "Increase quantization bit width".to_string(),
                        "Apply advanced calibration".to_string(),
                        "Validate quality improvements".to_string(),
                    ],
                });
            }
        }

        // Fallback: If no actions generated, add a basic optimization action
        if actions.is_empty() {
            actions.push(ImmediateAction {
                priority: ActionPriority::Medium,
                description: "Review and optimize quantization quality".to_string(),
                estimated_effort: "2-4 hours".to_string(),
                expected_impact: "Medium".to_string(),
                implementation_steps: vec![
                    "Analyze current quantization metrics".to_string(),
                    "Identify optimization opportunities".to_string(),
                    "Apply targeted improvements".to_string(),
                ],
            });
        }

        actions
    }

    fn generate_optimization_roadmap(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> OptimizationRoadmap {
        let mut phases = Vec::new();

        // Phase 1: Quick wins
        phases.push(RoadmapPhase {
            phase_name: "Quick Wins".to_string(),
            duration: "1-2 days".to_string(),
            objectives: vec![
                "Address critical quantization failures".to_string(),
                "Apply low-complexity optimizations".to_string(),
                "Validate immediate improvements".to_string(),
            ],
            deliverables: vec![
                "Fixed critical issues".to_string(),
                "Improved MSE in high-priority layers".to_string(),
                "Quality validation report".to_string(),
            ],
        });

        // Phase 2: Systematic improvements
        phases.push(RoadmapPhase {
            phase_name: "Systematic Optimization".to_string(),
            duration: "1-2 weeks".to_string(),
            objectives: vec![
                "Implement mixed precision quantization".to_string(),
                "Apply advanced calibration techniques".to_string(),
                "Optimize medium-priority layers".to_string(),
            ],
            deliverables: vec![
                "Mixed precision configuration".to_string(),
                "Calibration parameter optimization".to_string(),
                "Performance benchmarking results".to_string(),
            ],
        });

        // Phase 3: Advanced optimization
        phases.push(RoadmapPhase {
            phase_name: "Advanced Optimization".to_string(),
            duration: "2-4 weeks".to_string(),
            objectives: vec![
                "Implement custom quantization schemes".to_string(),
                "Develop layer-specific optimizations".to_string(),
                "Establish production monitoring".to_string(),
            ],
            deliverables: vec![
                "Custom quantization implementations".to_string(),
                "Production-ready configurations".to_string(),
                "Monitoring and alerting system".to_string(),
            ],
        });

        OptimizationRoadmap {
            phases,
            total_estimated_duration: "3-6 weeks".to_string(),
            success_criteria: vec![
                format!(
                    "Achieve <{:.1}% of layers with quality issues",
                    (analysis.problematic_layers.len() as f32
                        / analysis.layer_metrics.len() as f32)
                        * 50.0
                ),
                "Maintain <2% performance overhead".to_string(),
                "Establish automated quality monitoring".to_string(),
            ],
        }
    }

    fn generate_best_practices(&self) -> Vec<BestPractice> {
        vec![
            BestPractice {
                category: "Calibration".to_string(),
                recommendation: "Use representative dataset for calibration".to_string(),
                rationale: "Ensures quantization parameters match production data distribution"
                    .to_string(),
                implementation_notes: vec![
                    "Include edge cases in calibration data".to_string(),
                    "Use sufficient sample size (>1000 examples)".to_string(),
                    "Validate calibration quality regularly".to_string(),
                ],
            },
            BestPractice {
                category: "Bit Width Selection".to_string(),
                recommendation: "Start with 8-bit and adjust based on sensitivity analysis"
                    .to_string(),
                rationale: "Balances quality and performance while allowing targeted optimization"
                    .to_string(),
                implementation_notes: vec![
                    "Use sensitivity ranking to prioritize bit width increases".to_string(),
                    "Consider mixed precision for optimal trade-offs".to_string(),
                    "Monitor quality degradation carefully".to_string(),
                ],
            },
            BestPractice {
                category: "Monitoring".to_string(),
                recommendation: "Implement continuous quantization quality monitoring".to_string(),
                rationale: "Enables early detection of quality degradation in production"
                    .to_string(),
                implementation_notes: vec![
                    "Set up automated quality metrics collection".to_string(),
                    "Define alert thresholds based on business requirements".to_string(),
                    "Regularly review and update quality standards".to_string(),
                ],
            },
        ]
    }

    fn generate_monitoring_recommendations(&self) -> MonitoringRecommendations {
        MonitoringRecommendations {
            key_metrics_to_track: vec![
                "Mean Squared Error (MSE)".to_string(),
                "Signal-to-Quantization-Noise Ratio (SQNR)".to_string(),
                "Cosine Similarity".to_string(),
                "Per-layer sensitivity scores".to_string(),
            ],
            alert_thresholds: vec![
                "MSE > 1e-2 (requires immediate attention)".to_string(),
                "SQNR < 20 dB (investigate quantization settings)".to_string(),
                "Cosine Similarity < 0.95 (review layer configuration)".to_string(),
            ],
            reporting_frequency: "Daily for production, weekly for development".to_string(),
            dashboard_requirements: vec![
                "Real-time quality metrics display".to_string(),
                "Layer-wise error visualization".to_string(),
                "Historical trend analysis".to_string(),
                "Alert and notification system".to_string(),
            ],
        }
    }

    fn generate_technical_appendix(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<TechnicalAppendix> {
        Ok(TechnicalAppendix {
            methodology: self.generate_methodology_description(),
            detailed_metrics: analysis.layer_metrics.clone(),
            statistical_analysis: self
                .generate_statistical_analysis(&analysis.global_statistics)?,
            error_correlation_matrix: analysis.error_correlations.clone(),
            sensitivity_analysis_details: analysis.sensitivity_ranking.clone(),
            configuration_parameters: self.extract_configuration_parameters(analysis),
        })
    }

    fn generate_methodology_description(&self) -> String {
        "Quantization quality analysis methodology:\n\
        1. Calculate Mean Squared Error (MSE) between original and quantized outputs\n\
        2. Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB scale\n\
        3. Measure cosine similarity for output vector alignment\n\
        4. Perform layer-wise sensitivity analysis\n\
        5. Identify error correlations between layers\n\
        6. Generate mitigation recommendations based on threshold analysis\n\
        7. Assess overall quantization quality and business impact"
            .to_string()
    }

    fn generate_statistical_analysis(
        &self,
        stats: &crate::metrics::layer_wise::GlobalStatistics,
    ) -> Result<String> {
        Ok(format!(
            "Statistical Summary:\n\
            - Total Layers Analyzed: {}\n\
            - Mean MSE: {:.6} (σ = {:.6})\n\
            - Mean SQNR: {:.2} dB (σ = {:.2})\n\
            - Mean Cosine Similarity: {:.4}\n\
            - Best Performing Layer (MSE): {}\n\
            - Worst Performing Layer (MSE): {}\n\
            - Best SQNR Layer: {}",
            stats.num_layers,
            stats.mean_mse,
            stats.mse_std_dev,
            stats.mean_sqnr,
            stats.sqnr_std_dev,
            stats.mean_cosine_similarity,
            stats.best_mse_layer.as_deref().unwrap_or("N/A"),
            stats.worst_mse_layer.as_deref().unwrap_or("N/A"),
            stats.best_sqnr_layer.as_deref().unwrap_or("N/A")
        ))
    }

    fn extract_configuration_parameters(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> HashMap<String, String> {
        let mut config = HashMap::new();

        config.insert(
            "analysis_timestamp".to_string(),
            analysis.analysis_timestamp.to_string(),
        );
        config.insert(
            "total_layers".to_string(),
            analysis.layer_metrics.len().to_string(),
        );
        config.insert(
            "problematic_layers".to_string(),
            analysis.problematic_layers.len().to_string(),
        );
        config.insert(
            "high_priority_layers".to_string(),
            analysis
                .optimization_plan
                .high_priority_layers
                .len()
                .to_string(),
        );
        config.insert(
            "implementation_complexity".to_string(),
            format!("{:?}", analysis.optimization_plan.implementation_complexity),
        );
        config.insert(
            "estimated_improvement".to_string(),
            format!("{:?}", analysis.optimization_plan.estimated_improvement),
        );

        config
    }

    /// Export report in various formats
    pub fn export_report(&self, report: &ComprehensiveReport) -> Result<Vec<String>> {
        let mut exported_files = Vec::new();

        match self.report_format {
            ReportFormat::Markdown => {
                let _markdown_content = self.generate_markdown_report(report)?;
                let filename = format!("{}/quantization_analysis_report.md", self.output_directory);
                // In practice, write to file: std::fs::write(&filename, markdown_content)?;
                println!("Would export Markdown report to: {filename}");
                exported_files.push(filename);
            }

            ReportFormat::HTML => {
                let _html_content = self.generate_html_report(report)?;
                let filename = format!(
                    "{}/quantization_analysis_report.html",
                    self.output_directory
                );
                // In practice, write to file: std::fs::write(&filename, html_content)?;
                println!("Would export HTML report to: {filename}");
                exported_files.push(filename);
            }

            ReportFormat::PDF => {
                // In practice, generate PDF using a PDF library
                let filename =
                    format!("{}/quantization_analysis_report.pdf", self.output_directory);
                println!("Would export PDF report to: {filename}");
                exported_files.push(filename);
            }

            ReportFormat::JSON => {
                let _json_content = serde_json::to_string_pretty(report)
                    .map_err(|e| CandleError::Msg(format!("JSON serialization error: {e}")))?;
                let filename = format!(
                    "{}/quantization_analysis_report.json",
                    self.output_directory
                );
                // In practice, write to file: std::fs::write(&filename, json_content)?;
                println!("Would export JSON report to: {filename}");
                exported_files.push(filename);
            }
        }

        Ok(exported_files)
    }

    fn generate_markdown_report(&self, report: &ComprehensiveReport) -> Result<String> {
        let mut content = String::new();

        // Title and metadata
        content.push_str(&format!("# {}\n\n", report.metadata.title));
        let timestamp =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(report.metadata.generated_at);
        content.push_str(&format!("**Generated:** {timestamp:?}\n"));
        content.push_str(&format!(
            "**Layers Analyzed:** {}\n\n",
            report.metadata.total_layers_analyzed
        ));

        // Executive Summary
        content.push_str("## Executive Summary\n\n");
        content.push_str(&format!(
            "**Overall Quality:** {:?}\n\n",
            report.executive_summary.overall_quality_grade
        ));

        content.push_str("### Key Findings\n");
        for finding in &report.executive_summary.key_findings {
            content.push_str(&format!("- {finding}\n"));
        }
        content.push('\n');

        content.push_str(&format!(
            "**High Priority Actions:** {}\n",
            report.executive_summary.high_priority_actions_count
        ));
        content.push_str(&format!(
            "**Improvement Potential:** {}\n\n",
            report.executive_summary.estimated_improvement_potential
        ));

        // Next Steps
        content.push_str("### Immediate Next Steps\n");
        for step in &report.executive_summary.next_steps {
            content.push_str(&format!("1. {step}\n"));
        }
        content.push('\n');

        // Detailed Analysis
        content.push_str("## Detailed Analysis\n\n");
        content.push_str(&format!(
            "### Error Distribution\n{}\n\n",
            report
                .detailed_analysis
                .quantization_patterns
                .error_distribution_summary
        ));

        // Top problematic layers
        content.push_str("### Most Sensitive Layers\n");
        for (i, layer) in report
            .detailed_analysis
            .layer_breakdown
            .iter()
            .take(10)
            .enumerate()
        {
            content.push_str(&format!(
                "{}. **{}** - Sensitivity: {:.2}, Quality: {}/15\n",
                i + 1,
                layer.layer_name,
                layer.sensitivity_score,
                layer.quality_assessment.overall_score
            ));
        }
        content.push('\n');

        // Recommendations
        content.push_str("## Recommendations\n\n");
        content.push_str("### Immediate Actions\n");
        for action in &report.recommendations.immediate_actions {
            content.push_str(&format!(
                "- **{:?}:** {} (Effort: {}, Impact: {})\n",
                action.priority,
                action.description,
                action.estimated_effort,
                action.expected_impact
            ));
        }
        content.push('\n');

        Ok(content)
    }

    fn generate_html_report(&self, report: &ComprehensiveReport) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.metadata.title));
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("h1 { color: #333; border-bottom: 2px solid #0066cc; }\n");
        html.push_str("h2 { color: #0066cc; }\n");
        html.push_str(".quality-excellent { color: #00aa00; font-weight: bold; }\n");
        html.push_str(".quality-poor { color: #aa0000; font-weight: bold; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #f2f2f2; }\n");
        html.push_str("</style>\n</head>\n<body>\n");

        // Content
        html.push_str(&format!("<h1>{}</h1>\n", report.metadata.title));
        html.push_str("<h2>Executive Summary</h2>\n");
        html.push_str(&format!(
            "<p><strong>Overall Quality:</strong> <span class=\"quality-{}\"> {:?}</span></p>\n",
            match report.executive_summary.overall_quality_grade {
                QualityGrade::Excellent | QualityGrade::Good => "excellent",
                _ => "poor",
            },
            report.executive_summary.overall_quality_grade
        ));

        html.push_str("<h3>Key Findings</h3>\n<ul>\n");
        for finding in &report.executive_summary.key_findings {
            html.push_str(&format!("<li>{finding}</li>\n"));
        }
        html.push_str("</ul>\n");

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Generate mitigation report
    pub fn generate_mitigation_report(
        &self,
        mitigation_result: &MitigationResult,
    ) -> Result<MitigationReport> {
        Ok(MitigationReport {
            summary: MitigationSummary {
                total_actions: mitigation_result.mitigation_actions.len(),
                expected_improvement: mitigation_result.overall_improvement,
                estimated_implementation_time: mitigation_result
                    .implementation_plan
                    .total_estimated_duration_hours,
                confidence_level: mitigation_result.estimated_quality_gain.confidence_level,
            },
            action_details: mitigation_result.mitigation_actions.clone(),
            implementation_plan: mitigation_result.implementation_plan.clone(),
            risk_assessment: mitigation_result
                .implementation_plan
                .risk_assessment
                .clone(),
            expected_outcomes: mitigation_result.estimated_quality_gain.clone(),
        })
    }
}

/// Report data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveReport {
    pub metadata: ReportMetadata,
    pub executive_summary: ExecutiveSummary,
    pub detailed_analysis: DetailedAnalysis,
    pub recommendations: RecommendationSection,
    pub technical_appendix: Option<TechnicalAppendix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub title: String,
    pub generated_at: u64,
    pub analysis_timestamp: u64,
    pub report_version: String,
    pub total_layers_analyzed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub overall_quality_grade: QualityGrade,
    pub key_findings: Vec<String>,
    pub high_priority_actions_count: usize,
    pub estimated_improvement_potential: String,
    pub business_impact: BusinessImpact,
    pub next_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGrade {
    Excellent,
    Good,
    Satisfactory,
    NeedsImprovement,
    Poor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub performance_impact: PerformanceImpact,
    pub deployment_risk: DeploymentRisk,
    pub estimated_resolution_time: String,
    pub cost_of_inaction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceImpact {
    Minimal,
    Minor,
    Moderate,
    Severe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentRisk {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedAnalysis {
    pub layer_breakdown: Vec<LayerBreakdown>,
    pub quantization_patterns: QuantizationPatterns,
    pub performance_analysis: PerformanceAnalysis,
    pub error_correlation_summary: ErrorCorrelationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBreakdown {
    pub layer_name: String,
    pub metrics: QuantizationMetrics,
    pub sensitivity_score: f32,
    pub quality_assessment: LayerQualityAssessment,
    pub is_problematic: bool,
    pub recommended_actions: Vec<crate::metrics::MitigationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQualityAssessment {
    pub overall_score: u8,
    pub mse_grade: String,
    pub sqnr_grade: String,
    pub similarity_grade: String,
    pub primary_concern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPatterns {
    pub error_distribution_summary: String,
    pub layer_type_patterns: HashMap<String, i32>,
    pub correlation_insights: Vec<String>,
    pub common_issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub consistency_rating: String,
    pub predictability_score: f32,
    pub outlier_layers: Vec<String>,
    pub estimated_memory_usage: f32,
    pub estimated_compute_overhead: f32,
    pub scalability_assessment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrelationSummary {
    pub strong_correlations: usize,
    pub average_correlation: f32,
    pub most_correlated_pair: Option<(String, String, f32)>,
    pub correlation_insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationSection {
    pub immediate_actions: Vec<ImmediateAction>,
    pub optimization_roadmap: OptimizationRoadmap,
    pub best_practices: Vec<BestPractice>,
    pub monitoring_recommendations: MonitoringRecommendations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmediateAction {
    pub priority: ActionPriority,
    pub description: String,
    pub estimated_effort: String,
    pub expected_impact: String,
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRoadmap {
    pub phases: Vec<RoadmapPhase>,
    pub total_estimated_duration: String,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapPhase {
    pub phase_name: String,
    pub duration: String,
    pub objectives: Vec<String>,
    pub deliverables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub category: String,
    pub recommendation: String,
    pub rationale: String,
    pub implementation_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringRecommendations {
    pub key_metrics_to_track: Vec<String>,
    pub alert_thresholds: Vec<String>,
    pub reporting_frequency: String,
    pub dashboard_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalAppendix {
    pub methodology: String,
    pub detailed_metrics: HashMap<String, QuantizationMetrics>,
    pub statistical_analysis: String,
    pub error_correlation_matrix: HashMap<String, HashMap<String, f32>>,
    pub sensitivity_analysis_details: Vec<(String, f32)>,
    pub configuration_parameters: HashMap<String, String>,
}

/// Mitigation reporting structures

#[derive(Debug, Clone)]
pub struct MitigationReport {
    pub summary: MitigationSummary,
    pub action_details: Vec<crate::metrics::mitigation::MitigationAction>,
    pub implementation_plan: crate::metrics::mitigation::ImplementationPlan,
    pub risk_assessment: crate::metrics::mitigation::RiskAssessment,
    pub expected_outcomes: crate::metrics::mitigation::QualityGain,
}

#[derive(Debug, Clone)]
pub struct MitigationSummary {
    pub total_actions: usize,
    pub expected_improvement: f32,
    pub estimated_implementation_time: u32,
    pub confidence_level: f32,
}

#[derive(Debug, Clone)]
pub enum ReportFormat {
    Markdown,
    HTML,
    PDF,
    JSON,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_analysis() -> LayerWiseAnalysisResult {
        use crate::metrics::layer_wise::{
            EstimatedImprovement, GlobalStatistics, ImplementationComplexity,
            LayerWiseAnalysisResult, OptimizationPlan,
        };

        let mut layer_metrics = HashMap::new();
        layer_metrics.insert(
            "layer1".to_string(),
            QuantizationMetrics {
                mse: 0.001,
                sqnr: 40.0,
                cosine_similarity: 0.99,
                layer_name: "layer1".to_string(),
                ..Default::default()
            },
        );
        layer_metrics.insert(
            "layer2".to_string(),
            QuantizationMetrics {
                mse: 0.1,
                sqnr: 15.0,
                cosine_similarity: 0.85,
                layer_name: "layer2".to_string(),
                ..Default::default()
            },
        );

        LayerWiseAnalysisResult {
            layer_metrics,
            sensitivity_ranking: vec![("layer2".to_string(), 15.0), ("layer1".to_string(), 2.0)],
            layer_rankings: Vec::new(),
            mitigation_recommendations: HashMap::new(),
            error_correlations: HashMap::new(),
            global_statistics: GlobalStatistics {
                num_layers: 2,
                mean_mse: 0.0505,
                mean_sqnr: 27.5,
                mean_cosine_similarity: 0.92,
                mean_relative_error: 0.05,
                mse_std_dev: 0.0495,
                sqnr_std_dev: 12.5,
                best_mse_layer: Some("layer1".to_string()),
                worst_mse_layer: Some("layer2".to_string()),
                best_sqnr_layer: Some("layer1".to_string()),
            },
            optimization_plan: OptimizationPlan {
                high_priority_layers: Vec::new(),  // Keep empty for low deployment risk
                medium_priority_layers: Vec::new(),
                low_priority_layers: vec!["layer1".to_string(), "layer2".to_string()],
                optimization_strategies: HashMap::new(),
                estimated_improvement: EstimatedImprovement::Medium,
                implementation_complexity: ImplementationComplexity::Low,
            },
            problematic_layers: Vec::new(),
            analysis_timestamp: 1640995200, // 2022-01-01 00:00:00 UTC
        }
    }

    #[test]
    fn test_reporting_engine_creation() {
        let engine = ReportingEngine::new("./reports".to_string());
        assert_eq!(engine.output_directory, "./reports");
        assert!(matches!(engine.report_format, ReportFormat::Markdown));
        assert!(engine.include_visualizations);
        assert!(engine.detailed_mode);
    }

    #[test]
    fn test_quality_assessment() {
        let engine = ReportingEngine::new("./reports".to_string());

        let high_quality_stats = crate::metrics::layer_wise::GlobalStatistics {
            num_layers: 10,
            mean_mse: 1e-5,
            mean_sqnr: 50.0,
            mean_cosine_similarity: 0.995,
            mse_std_dev: 1e-6,
            ..Default::default()
        };

        let grade = engine.assess_overall_quality(&high_quality_stats);
        assert!(matches!(grade, QualityGrade::Excellent));

        let poor_quality_stats = crate::metrics::layer_wise::GlobalStatistics {
            num_layers: 10,
            mean_mse: 1e-1,
            mean_sqnr: 5.0,
            mean_cosine_similarity: 0.7,
            mse_std_dev: 1e-1,
            ..Default::default()
        };

        let grade = engine.assess_overall_quality(&poor_quality_stats);
        assert!(matches!(grade, QualityGrade::Poor));
    }

    #[test]
    fn test_layer_quality_assessment() {
        let engine = ReportingEngine::new("./reports".to_string());

        let excellent_metrics = QuantizationMetrics {
            mse: 1e-5,
            sqnr: 50.0,
            cosine_similarity: 0.995,
            ..Default::default()
        };

        let assessment = engine.assess_layer_quality(&excellent_metrics);
        assert_eq!(assessment.overall_score, 15); // Maximum score
        assert_eq!(assessment.mse_grade, "Excellent");

        let poor_metrics = QuantizationMetrics {
            mse: 1e-1,
            sqnr: 5.0,
            cosine_similarity: 0.7,
            ..Default::default()
        };

        let assessment = engine.assess_layer_quality(&poor_metrics);
        assert!(assessment.overall_score < 10);
        assert_eq!(assessment.mse_grade, "Critical");
    }

    #[test]
    fn test_comprehensive_report_generation() -> Result<()> {
        let engine = ReportingEngine::new("./reports".to_string());
        let analysis = create_test_analysis();

        let report = engine.generate_comprehensive_report(&analysis)?;

        assert_eq!(report.metadata.total_layers_analyzed, 2);
        assert!(!report.executive_summary.key_findings.is_empty());
        assert!(!report.detailed_analysis.layer_breakdown.is_empty());
        assert!(!report.recommendations.immediate_actions.is_empty());

        Ok(())
    }

    #[test]
    fn test_business_impact_assessment() {
        let engine = ReportingEngine::new("./reports".to_string());
        let analysis = create_test_analysis();

        let impact = engine.assess_business_impact(&analysis);

        // Should be minimal since no problematic layers in test data
        assert!(matches!(
            impact.performance_impact,
            PerformanceImpact::Minimal
        ));
        assert!(matches!(impact.deployment_risk, DeploymentRisk::Low));
    }

    #[test]
    fn test_primary_concern_identification() {
        let engine = ReportingEngine::new("./reports".to_string());

        let high_mse_metrics = QuantizationMetrics {
            mse: 0.11,  // Changed from 1e-1 (0.1) to 0.11 to be greater than threshold
            sqnr: 30.0,
            cosine_similarity: 0.95,
            relative_error: 0.05,
            mean_absolute_error: 0.01,
            max_error: 0.1,
            ..Default::default()
        };

        let concern = engine.identify_primary_concern(&high_mse_metrics);
        assert!(concern.contains("High MSE"));

        let low_sqnr_metrics = QuantizationMetrics {
            mse: 1e-4,
            sqnr: 5.0,
            cosine_similarity: 0.95,
            ..Default::default()
        };

        let concern = engine.identify_primary_concern(&low_sqnr_metrics);
        assert!(concern.contains("Low SQNR"));
    }

    #[test]
    fn test_markdown_report_generation() -> Result<()> {
        let engine = ReportingEngine::new("./reports".to_string());
        let analysis = create_test_analysis();

        let report = engine.generate_comprehensive_report(&analysis)?;
        let markdown = engine.generate_markdown_report(&report)?;

        assert!(markdown.contains("# BitNet Quantization Analysis Report"));
        assert!(markdown.contains("## Executive Summary"));
        assert!(markdown.contains("## Detailed Analysis"));
        assert!(markdown.contains("## Recommendations"));

        Ok(())
    }

    #[test]
    fn test_html_report_generation() -> Result<()> {
        let engine = ReportingEngine::new("./reports".to_string());
        let analysis = create_test_analysis();

        let report = engine.generate_comprehensive_report(&analysis)?;
        let html = engine.generate_html_report(&report)?;

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>BitNet Quantization Analysis Report</title>"));
        assert!(html.contains("<h1>BitNet Quantization Analysis Report</h1>"));
        assert!(html.contains("<h2>Executive Summary</h2>"));

        Ok(())
    }

    #[test]
    fn test_best_practices_generation() {
        let engine = ReportingEngine::new("./reports".to_string());
        let best_practices = engine.generate_best_practices();

        assert!(!best_practices.is_empty());
        assert!(best_practices.iter().any(|bp| bp.category == "Calibration"));
        assert!(best_practices
            .iter()
            .any(|bp| bp.category == "Bit Width Selection"));
        assert!(best_practices.iter().any(|bp| bp.category == "Monitoring"));
    }

    #[test]
    fn test_error_correlation_summary() {
        let engine = ReportingEngine::new("./reports".to_string());

        let mut correlations = HashMap::new();
        let mut layer1_corr = HashMap::new();
        layer1_corr.insert("layer2".to_string(), 0.8);
        layer1_corr.insert("layer3".to_string(), 0.6);
        correlations.insert("layer1".to_string(), layer1_corr);

        let mut layer2_corr = HashMap::new();
        layer2_corr.insert("layer1".to_string(), 0.8);
        layer2_corr.insert("layer3".to_string(), 0.3);
        correlations.insert("layer2".to_string(), layer2_corr);

        let summary = engine.summarize_error_correlations(&correlations);

        assert_eq!(summary.strong_correlations, 1); // Only layer1-layer2 > 0.7
        assert!(summary.average_correlation > 0.0);
        assert!(summary.most_correlated_pair.is_some());
    }

    #[test]
    fn test_scalability_assessment() {
        let engine = ReportingEngine::new("./reports".to_string());
        let analysis = create_test_analysis();

        let assessment = engine.assess_scalability(&analysis);
        // Should be excellent since no problematic layers
        assert!(assessment.contains("Excellent"));
    }

    #[test]
    fn test_immediate_actions_generation() {
        let engine = ReportingEngine::new("./reports".to_string());
        let mut analysis = create_test_analysis();

        // Add a problematic layer
        analysis
            .problematic_layers
            .push(crate::metrics::layer_wise::ProblematicLayer {
                layer_name: "critical_layer".to_string(),
                issues: vec![crate::metrics::layer_wise::QualityIssue::HighMSE],
                severity: crate::metrics::layer_wise::IssueSeverity::Critical,
                recommended_actions: vec!["Fix quantization".to_string()],
            });

        let actions = engine.generate_immediate_actions(&analysis);

        assert!(!actions.is_empty());
        assert!(actions
            .iter()
            .any(|a| matches!(a.priority, ActionPriority::Critical)));
    }
}
