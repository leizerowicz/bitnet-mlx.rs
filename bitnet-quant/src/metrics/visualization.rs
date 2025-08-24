// bitnet-quant/src/metrics/visualization.rs
//! Error Visualization Tools for Quantization Analysis
//!
//! Provides comprehensive visualization capabilities for quantization error analysis,
//! including real-time plotting, statistical visualizations, and comparison tools.

use crate::metrics::{
    cosine_similarity::SimilarityEvolution,
    error_analysis::ErrorHistogram,
    layer_wise::{LayerWiseAnalysisResult, TemporalAnalysis},
    sqnr::SQNREvolution,
    tensor_to_vec,
};
use candle_core::{Device, Error as CandleError, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive visualization engine for quantization analysis
#[derive(Debug)]
pub struct VisualizationEngine {
    device: Device,
    output_format: OutputFormat,
    color_scheme: ColorScheme,
    enable_interactive: bool,
    export_path: String,
}

impl VisualizationEngine {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            output_format: OutputFormat::SVG,
            color_scheme: ColorScheme::Default,
            enable_interactive: false,
            export_path: "./visualizations".to_string(),
        }
    }

    pub fn with_config(
        device: Device,
        output_format: OutputFormat,
        color_scheme: ColorScheme,
        enable_interactive: bool,
        export_path: String,
    ) -> Self {
        Self {
            device,
            output_format,
            color_scheme,
            enable_interactive,
            export_path,
        }
    }

    /// Generate comprehensive error visualization dashboard
    pub fn create_error_dashboard(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<VisualizationDashboard> {
        let mut dashboard = VisualizationDashboard::new("Quantization Error Analysis");

        // 1. Layer-wise error comparison chart
        let layer_comparison = self.create_layer_comparison_chart(analysis)?;
        dashboard.add_visualization(layer_comparison);

        // 2. Sensitivity ranking visualization
        let sensitivity_chart =
            self.create_sensitivity_ranking_chart(&analysis.sensitivity_ranking)?;
        dashboard.add_visualization(sensitivity_chart);

        // 3. Error correlation heatmap
        if !analysis.error_correlations.is_empty() {
            let correlation_heatmap =
                self.create_correlation_heatmap(&analysis.error_correlations)?;
            dashboard.add_visualization(correlation_heatmap);
        }

        // 4. Global statistics overview
        let stats_overview = self.create_statistics_overview(&analysis.global_statistics)?;
        dashboard.add_visualization(stats_overview);

        // 5. Problematic layers visualization
        if !analysis.problematic_layers.is_empty() {
            let problems_viz =
                self.create_problematic_layers_chart(&analysis.problematic_layers)?;
            dashboard.add_visualization(problems_viz);
        }

        // 6. Optimization plan visualization
        let optimization_viz = self.create_optimization_plan_chart(&analysis.optimization_plan)?;
        dashboard.add_visualization(optimization_viz);

        Ok(dashboard)
    }

    fn create_layer_comparison_chart(
        &self,
        analysis: &LayerWiseAnalysisResult,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (layer_name, metrics) in &analysis.layer_metrics {
            data_points.push(DataPoint {
                label: layer_name.clone(),
                values: vec![
                    ("MSE".to_string(), metrics.mse.into()),
                    ("SQNR".to_string(), metrics.sqnr.into()),
                    (
                        "Cosine Similarity".to_string(),
                        metrics.cosine_similarity.into(),
                    ),
                    ("Relative Error".to_string(), metrics.relative_error.into()),
                ],
                metadata: Some(format!(
                    "Layer: {}, Timestamp: {}",
                    layer_name, metrics.timestamp
                )),
            });
        }

        Ok(Visualization {
            id: "layer_comparison".to_string(),
            title: "Layer-wise Error Metrics Comparison".to_string(),
            visualization_type: VisualizationType::GroupedBarChart,
            data_points,
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: self.color_scheme.clone(),
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    fn create_sensitivity_ranking_chart(
        &self,
        sensitivity_ranking: &[(String, f32)],
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (layer_name, sensitivity) in sensitivity_ranking.iter().take(20) {
            // Top 20 most sensitive
            data_points.push(DataPoint {
                label: layer_name.clone(),
                values: vec![("Sensitivity Score".to_string(), (*sensitivity).into())],
                metadata: Some(format!(
                    "Layer: {layer_name}, Sensitivity: {sensitivity:.3}"
                )),
            });
        }

        Ok(Visualization {
            id: "sensitivity_ranking".to_string(),
            title: "Layer Sensitivity Ranking (Top 20)".to_string(),
            visualization_type: VisualizationType::HorizontalBarChart,
            data_points,
            config: ChartConfig {
                width: 1000,
                height: 800,
                show_legend: false,
                color_scheme: ColorScheme::Gradient,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    fn create_correlation_heatmap(
        &self,
        correlations: &HashMap<String, HashMap<String, f32>>,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();
        let layers: Vec<&String> = correlations.keys().collect();

        for (i, layer1) in layers.iter().enumerate() {
            for (j, layer2) in layers.iter().enumerate() {
                if let Some(correlation) = correlations.get(*layer1).and_then(|c| c.get(*layer2)) {
                    data_points.push(DataPoint {
                        label: format!("{layer1}_{layer2}"),
                        values: vec![
                            ("X".to_string(), DataValue::Integer(i as i64)),
                            ("Y".to_string(), DataValue::Integer(j as i64)),
                            ("Correlation".to_string(), (*correlation).into()),
                        ],
                        metadata: Some(format!(
                            "Correlation between {layer1} and {layer2}: {correlation:.3}"
                        )),
                    });
                }
            }
        }

        Ok(Visualization {
            id: "correlation_heatmap".to_string(),
            title: "Layer Error Correlation Heatmap".to_string(),
            visualization_type: VisualizationType::Heatmap,
            data_points,
            config: ChartConfig {
                width: 800,
                height: 800,
                show_legend: true,
                color_scheme: ColorScheme::BlueRed,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    fn create_statistics_overview(
        &self,
        stats: &crate::metrics::layer_wise::GlobalStatistics,
    ) -> Result<Visualization> {
        let data_points = vec![
            DataPoint {
                label: "Mean MSE".to_string(),
                values: vec![("Value".to_string(), stats.mean_mse.into())],
                metadata: Some(format!("Global mean MSE: {:.6}", stats.mean_mse)),
            },
            DataPoint {
                label: "Mean SQNR".to_string(),
                values: vec![("Value".to_string(), stats.mean_sqnr.into())],
                metadata: Some(format!("Global mean SQNR: {:.2} dB", stats.mean_sqnr)),
            },
            DataPoint {
                label: "Mean Cosine Similarity".to_string(),
                values: vec![("Value".to_string(), stats.mean_cosine_similarity.into())],
                metadata: Some(format!(
                    "Global mean cosine similarity: {:.4}",
                    stats.mean_cosine_similarity
                )),
            },
            DataPoint {
                label: "Number of Layers".to_string(),
                values: vec![(
                    "Value".to_string(),
                    DataValue::Integer(stats.num_layers as i64),
                )],
                metadata: Some(format!("Total analyzed layers: {}", stats.num_layers)),
            },
        ];

        Ok(Visualization {
            id: "global_statistics".to_string(),
            title: "Global Quantization Statistics".to_string(),
            visualization_type: VisualizationType::RadialChart,
            data_points,
            config: ChartConfig {
                width: 500,
                height: 500,
                show_legend: true,
                color_scheme: self.color_scheme.clone(),
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    fn create_problematic_layers_chart(
        &self,
        problematic_layers: &[crate::metrics::layer_wise::ProblematicLayer],
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for layer in problematic_layers {
            let severity_score = match layer.severity {
                crate::metrics::layer_wise::IssueSeverity::Critical => 4.0,
                crate::metrics::layer_wise::IssueSeverity::High => 3.0,
                crate::metrics::layer_wise::IssueSeverity::Medium => 2.0,
                crate::metrics::layer_wise::IssueSeverity::Low => 1.0,
            };

            data_points.push(DataPoint {
                label: layer.layer_name.clone(),
                values: vec![
                    ("Severity Score".to_string(), severity_score.into()),
                    (
                        "Issue Count".to_string(),
                        DataValue::Integer(layer.issues.len() as i64),
                    ),
                ],
                metadata: Some(format!(
                    "Layer: {}, Severity: {:?}, Issues: {}, Actions: {}",
                    layer.layer_name,
                    layer.severity,
                    layer.issues.len(),
                    layer.recommended_actions.len()
                )),
            });
        }

        Ok(Visualization {
            id: "problematic_layers".to_string(),
            title: "Problematic Layers Analysis".to_string(),
            visualization_type: VisualizationType::BubbleChart,
            data_points,
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: ColorScheme::RedYellowGreen,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    fn create_optimization_plan_chart(
        &self,
        plan: &crate::metrics::layer_wise::OptimizationPlan,
    ) -> Result<Visualization> {
        let data_points = vec![
            DataPoint {
                label: "High Priority".to_string(),
                values: vec![(
                    "Count".to_string(),
                    DataValue::Integer(plan.high_priority_layers.len() as i64),
                )],
                metadata: Some(format!(
                    "High priority layers: {}",
                    plan.high_priority_layers.len()
                )),
            },
            DataPoint {
                label: "Medium Priority".to_string(),
                values: vec![(
                    "Count".to_string(),
                    DataValue::Integer(plan.medium_priority_layers.len() as i64),
                )],
                metadata: Some(format!(
                    "Medium priority layers: {}",
                    plan.medium_priority_layers.len()
                )),
            },
            DataPoint {
                label: "Low Priority".to_string(),
                values: vec![(
                    "Count".to_string(),
                    DataValue::Integer(plan.low_priority_layers.len() as i64),
                )],
                metadata: Some(format!(
                    "Low priority layers: {}",
                    plan.low_priority_layers.len()
                )),
            },
        ];

        Ok(Visualization {
            id: "optimization_plan".to_string(),
            title: "Optimization Priority Distribution".to_string(),
            visualization_type: VisualizationType::PieChart,
            data_points,
            config: ChartConfig {
                width: 500,
                height: 500,
                show_legend: true,
                color_scheme: ColorScheme::GreenYellowRed,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    /// Create error histogram visualization
    pub fn create_error_histogram(
        &self,
        histogram: &ErrorHistogram,
        layer_name: &str,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (i, &count) in histogram.bins.iter().enumerate() {
            if i < histogram.bin_edges.len() - 1 {
                let bin_center = (histogram.bin_edges[i] + histogram.bin_edges[i + 1]) / 2.0;
                data_points.push(DataPoint {
                    label: format!("Bin {i}"),
                    values: vec![
                        ("Error Value".to_string(), bin_center.into()),
                        ("Frequency".to_string(), DataValue::Integer(count as i64)),
                    ],
                    metadata: Some(format!(
                        "Range: [{:.6}, {:.6}], Count: {}",
                        histogram.bin_edges[i],
                        histogram.bin_edges[i + 1],
                        count
                    )),
                });
            }
        }

        Ok(Visualization {
            id: format!("error_histogram_{layer_name}"),
            title: format!("Error Distribution Histogram - {layer_name}"),
            visualization_type: VisualizationType::Histogram,
            data_points,
            config: ChartConfig {
                width: 800,
                height: 500,
                show_legend: false,
                color_scheme: ColorScheme::Blue,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    /// Create temporal evolution visualization
    pub fn create_temporal_visualization(
        &self,
        temporal_analysis: &TemporalAnalysis,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (layer_name, trend) in &temporal_analysis.layer_trends {
            data_points.push(DataPoint {
                label: layer_name.clone(),
                values: vec![
                    (
                        "Quality Change".to_string(),
                        trend.overall_quality_change.into(),
                    ),
                    ("Volatility".to_string(), trend.volatility.into()),
                ],
                metadata: Some(format!(
                    "Layer: {}, Quality Change: {:.4}, Volatility: {:.4}",
                    layer_name, trend.overall_quality_change, trend.volatility
                )),
            });
        }

        Ok(Visualization {
            id: "temporal_evolution".to_string(),
            title: "Temporal Quality Evolution".to_string(),
            visualization_type: VisualizationType::ScatterPlot,
            data_points,
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: self.color_scheme.clone(),
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    /// Create SQNR evolution line chart
    pub fn create_sqnr_evolution_chart(
        &self,
        evolution: &SQNREvolution,
        layer_name: &str,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (iteration, &sqnr_value) in evolution.sqnr_values.iter().enumerate() {
            data_points.push(DataPoint {
                label: format!("Iteration {iteration}"),
                values: vec![
                    (
                        "Iteration".to_string(),
                        DataValue::Integer(iteration as i64),
                    ),
                    ("SQNR (dB)".to_string(), sqnr_value.into()),
                ],
                metadata: Some(format!("Iteration: {iteration}, SQNR: {sqnr_value:.2} dB")),
            });
        }

        Ok(Visualization {
            id: format!("sqnr_evolution_{layer_name}"),
            title: format!("SQNR Evolution - {layer_name}"),
            visualization_type: VisualizationType::LineChart,
            data_points,
            config: ChartConfig {
                width: 1000,
                height: 400,
                show_legend: false,
                color_scheme: ColorScheme::Blue,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    /// Create similarity evolution visualization
    pub fn create_similarity_evolution_chart(
        &self,
        evolution: &SimilarityEvolution,
        layer_name: &str,
    ) -> Result<Visualization> {
        let mut data_points = Vec::new();

        for (iteration, &similarity) in evolution.similarity_values.iter().enumerate() {
            data_points.push(DataPoint {
                label: format!("Iteration {iteration}"),
                values: vec![
                    (
                        "Iteration".to_string(),
                        DataValue::Integer(iteration as i64),
                    ),
                    ("Cosine Similarity".to_string(), similarity.into()),
                ],
                metadata: Some(format!(
                    "Iteration: {iteration}, Similarity: {similarity:.4}"
                )),
            });
        }

        Ok(Visualization {
            id: format!("similarity_evolution_{layer_name}"),
            title: format!("Cosine Similarity Evolution - {layer_name}"),
            visualization_type: VisualizationType::LineChart,
            data_points,
            config: ChartConfig {
                width: 1000,
                height: 400,
                show_legend: false,
                color_scheme: ColorScheme::Green,
                interactive: self.enable_interactive,
                export_formats: vec![self.output_format.clone()],
            },
        })
    }

    /// Create 3D tensor visualization for spatial error patterns
    pub fn create_tensor_visualization(
        &self,
        original: &Tensor,
        quantized: &Tensor,
        layer_name: &str,
    ) -> Result<Visualization> {
        // Calculate error tensor
        let error = original.sub(quantized)?.abs()?;

        // For demonstration, we'll create a 2D heatmap if tensor is 2D or higher
        if error.dims().len() >= 2 {
            let error_2d = if error.dims().len() > 2 {
                // Flatten to 2D by taking mean across other dimensions
                let mut current = error.clone();
                for _ in 2..error.dims().len() {
                    current = current.mean_keepdim(current.dims().len() - 1)?;
                }
                current.squeeze(2)?
            } else {
                error
            };

            let error_vec = tensor_to_vec(&error_2d)?;
            let dims = error_2d.dims();
            let mut data_points = Vec::new();

            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    let idx = i * dims[1] + j;
                    if idx < error_vec.len() {
                        data_points.push(DataPoint {
                            label: format!("({i}, {j})"),
                            values: vec![
                                ("X".to_string(), DataValue::Integer(i as i64)),
                                ("Y".to_string(), DataValue::Integer(j as i64)),
                                ("Error".to_string(), error_vec[idx].into()),
                            ],
                            metadata: Some(format!(
                                "Position: ({}, {}), Error: {:.6}",
                                i, j, error_vec[idx]
                            )),
                        });
                    }
                }
            }

            Ok(Visualization {
                id: format!("tensor_visualization_{layer_name}"),
                title: format!("Spatial Error Distribution - {layer_name}"),
                visualization_type: VisualizationType::Heatmap,
                data_points,
                config: ChartConfig {
                    width: 600,
                    height: 600,
                    show_legend: true,
                    color_scheme: ColorScheme::RedYellowBlue,
                    interactive: self.enable_interactive,
                    export_formats: vec![self.output_format.clone()],
                },
            })
        } else {
            // For 1D tensors, create a line plot
            let error_vec = tensor_to_vec(&error)?;
            let mut data_points = Vec::new();

            for (i, &error_val) in error_vec.iter().enumerate() {
                data_points.push(DataPoint {
                    label: format!("Index {i}"),
                    values: vec![
                        ("Index".to_string(), DataValue::Integer(i as i64)),
                        ("Error".to_string(), error_val.into()),
                    ],
                    metadata: Some(format!("Index: {i}, Error: {error_val:.6}")),
                });
            }

            Ok(Visualization {
                id: format!("tensor_visualization_{layer_name}"),
                title: format!("1D Error Distribution - {layer_name}"),
                visualization_type: VisualizationType::LineChart,
                data_points,
                config: ChartConfig {
                    width: 800,
                    height: 400,
                    show_legend: false,
                    color_scheme: ColorScheme::Red,
                    interactive: self.enable_interactive,
                    export_formats: vec![self.output_format.clone()],
                },
            })
        }
    }

    /// Export visualizations to various formats
    pub fn export_visualization(&self, visualization: &Visualization) -> Result<Vec<String>> {
        let mut exported_files = Vec::new();

        for format in &visualization.config.export_formats {
            let filename = match format {
                OutputFormat::SVG => format!("{}/{}.svg", self.export_path, visualization.id),
                OutputFormat::PNG => format!("{}/{}.png", self.export_path, visualization.id),
                OutputFormat::HTML => format!("{}/{}.html", self.export_path, visualization.id),
                OutputFormat::JSON => format!("{}/{}.json", self.export_path, visualization.id),
                OutputFormat::CSV => format!("{}/{}.csv", self.export_path, visualization.id),
            };

            // In a real implementation, this would generate the actual file
            // For now, we'll just simulate the export
            self.generate_export_file(visualization, format, &filename)?;
            exported_files.push(filename);
        }

        Ok(exported_files)
    }

    fn generate_export_file(
        &self,
        visualization: &Visualization,
        format: &OutputFormat,
        filename: &str,
    ) -> Result<()> {
        match format {
            OutputFormat::JSON => {
                let _json_data = serde_json::to_string_pretty(visualization)
                    .map_err(|e| CandleError::Msg(format!("JSON serialization error: {e}")))?;
                // In practice, write to file: std::fs::write(filename, json_data)?;
                println!("Would export JSON to: {filename}");
            }
            OutputFormat::CSV => {
                let _csv_data = self.convert_to_csv(visualization)?;
                // In practice, write CSV data to file
                println!("Would export CSV to: {filename}");
            }
            OutputFormat::SVG => {
                let _svg_data = self.generate_svg(visualization)?;
                // In practice, write SVG data to file
                println!("Would export SVG to: {filename}");
            }
            OutputFormat::PNG => {
                // In practice, generate PNG from SVG or use plotting library
                println!("Would export PNG to: {filename}");
            }
            OutputFormat::HTML => {
                let _html_data = self.generate_interactive_html(visualization)?;
                // In practice, write HTML data to file
                println!("Would export HTML to: {filename}");
            }
        }
        Ok(())
    }

    fn convert_to_csv(&self, visualization: &Visualization) -> Result<String> {
        let mut csv_lines = Vec::new();

        // Header
        if let Some(first_point) = visualization.data_points.first() {
            let mut header = vec!["label".to_string()];
            for (key, _) in &first_point.values {
                header.push(key.clone());
            }
            csv_lines.push(header.join(","));
        }

        // Data rows
        for point in &visualization.data_points {
            let mut row = vec![point.label.clone()];
            for (_, value) in &point.values {
                row.push(format!("{value}"));
            }
            csv_lines.push(row.join(","));
        }

        Ok(csv_lines.join("\n"))
    }

    fn generate_svg(&self, _visualization: &Visualization) -> Result<String> {
        // In practice, this would generate actual SVG content
        Ok(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
            <rect width="100%" height="100%" fill="white"/>
            <text x="400" y="300" text-anchor="middle" font-family="Arial" font-size="16">
                SVG Visualization Placeholder
            </text>
        </svg>"#
                .to_string(),
        )
    }

    fn generate_interactive_html(&self, visualization: &Visualization) -> Result<String> {
        // In practice, this would generate interactive HTML with D3.js or similar
        Ok(format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div id="visualization">
        <h1>{}</h1>
        <p>Interactive visualization placeholder</p>
    </div>
</body>
</html>"#,
            visualization.title, visualization.title
        ))
    }

    /// Create real-time monitoring dashboard
    pub fn create_realtime_dashboard(&self) -> Result<RealtimeDashboard> {
        Ok(RealtimeDashboard {
            id: "realtime_quantization_monitor".to_string(),
            title: "Real-time Quantization Monitoring".to_string(),
            panels: vec![
                DashboardPanel {
                    id: "error_metrics".to_string(),
                    title: "Error Metrics".to_string(),
                    visualization_type: VisualizationType::LineChart,
                    update_interval_ms: 1000,
                },
                DashboardPanel {
                    id: "layer_comparison".to_string(),
                    title: "Layer Comparison".to_string(),
                    visualization_type: VisualizationType::GroupedBarChart,
                    update_interval_ms: 5000,
                },
                DashboardPanel {
                    id: "quality_indicators".to_string(),
                    title: "Quality Indicators".to_string(),
                    visualization_type: VisualizationType::GaugeChart,
                    update_interval_ms: 2000,
                },
            ],
            auto_refresh: true,
            export_path: self.export_path.clone(),
        })
    }
}

/// Visualization data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationDashboard {
    pub title: String,
    pub visualizations: Vec<Visualization>,
    pub created_at: u64,
    pub export_path: String,
}

impl VisualizationDashboard {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            visualizations: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            export_path: String::new(),
        }
    }

    pub fn add_visualization(&mut self, visualization: Visualization) {
        self.visualizations.push(visualization);
    }

    pub fn get_visualization(&self, id: &str) -> Option<&Visualization> {
        self.visualizations.iter().find(|v| v.id == id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Visualization {
    pub id: String,
    pub title: String,
    pub visualization_type: VisualizationType,
    pub data_points: Vec<DataPoint>,
    pub config: ChartConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub label: String,
    pub values: Vec<(String, DataValue)>,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataValue {
    Float(f32),
    Integer(i64),
    String(String),
    Boolean(bool),
}

impl From<f32> for DataValue {
    fn from(value: f32) -> Self {
        DataValue::Float(value)
    }
}

impl From<i64> for DataValue {
    fn from(value: i64) -> Self {
        DataValue::Integer(value)
    }
}

impl From<String> for DataValue {
    fn from(value: String) -> Self {
        DataValue::String(value)
    }
}

impl std::fmt::Display for DataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataValue::Float(v) => write!(f, "{v}"),
            DataValue::Integer(v) => write!(f, "{v}"),
            DataValue::String(v) => write!(f, "{v}"),
            DataValue::Boolean(v) => write!(f, "{v}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    GroupedBarChart,
    HorizontalBarChart,
    PieChart,
    ScatterPlot,
    Heatmap,
    Histogram,
    BoxPlot,
    ViolinPlot,
    RadialChart,
    BubbleChart,
    GaugeChart,
    Treemap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    pub width: u32,
    pub height: u32,
    pub show_legend: bool,
    pub color_scheme: ColorScheme,
    pub interactive: bool,
    pub export_formats: Vec<OutputFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Blue,
    Green,
    Red,
    BlueRed,
    RedYellowBlue,
    RedYellowGreen,
    GreenYellowRed,
    Gradient,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    SVG,
    PNG,
    HTML,
    JSON,
    CSV,
}

/// Real-time dashboard components
#[derive(Debug, Clone)]
pub struct RealtimeDashboard {
    pub id: String,
    pub title: String,
    pub panels: Vec<DashboardPanel>,
    pub auto_refresh: bool,
    pub export_path: String,
}

#[derive(Debug, Clone)]
pub struct DashboardPanel {
    pub id: String,
    pub title: String,
    pub visualization_type: VisualizationType,
    pub update_interval_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::layer_wise::{GlobalStatistics, LayerWiseAnalysisResult};
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn create_test_analysis() -> LayerWiseAnalysisResult {
        let mut layer_metrics = HashMap::new();
        layer_metrics.insert(
            "layer1".to_string(),
            crate::metrics::QuantizationMetrics {
                mse: 0.001,
                sqnr: 40.0,
                cosine_similarity: 0.99,
                layer_name: "layer1".to_string(),
                ..Default::default()
            },
        );
        layer_metrics.insert(
            "layer2".to_string(),
            crate::metrics::QuantizationMetrics {
                mse: 0.01,
                sqnr: 20.0,
                cosine_similarity: 0.95,
                layer_name: "layer2".to_string(),
                ..Default::default()
            },
        );

        LayerWiseAnalysisResult {
            layer_metrics,
            sensitivity_ranking: vec![("layer2".to_string(), 15.0), ("layer1".to_string(), 5.0)],
            layer_rankings: Vec::new(),
            mitigation_recommendations: HashMap::new(),
            error_correlations: HashMap::new(),
            global_statistics: GlobalStatistics::default(),
            optimization_plan: crate::metrics::layer_wise::OptimizationPlan {
                high_priority_layers: vec!["layer2".to_string()],
                medium_priority_layers: Vec::new(),
                low_priority_layers: vec!["layer1".to_string()],
                optimization_strategies: HashMap::new(),
                estimated_improvement: crate::metrics::layer_wise::EstimatedImprovement::Medium,
                implementation_complexity:
                    crate::metrics::layer_wise::ImplementationComplexity::Low,
            },
            problematic_layers: Vec::new(),
            analysis_timestamp: 0,
        }
    }

    #[test]
    fn test_visualization_engine_creation() {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);
        assert_eq!(engine.output_format, OutputFormat::SVG);
        assert!(!engine.enable_interactive);
    }

    #[test]
    fn test_error_dashboard_creation() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);
        let analysis = create_test_analysis();

        let dashboard = engine.create_error_dashboard(&analysis)?;

        assert!(!dashboard.visualizations.is_empty());
        assert_eq!(dashboard.title, "Quantization Error Analysis");

        Ok(())
    }

    #[test]
    fn test_layer_comparison_chart() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);
        let analysis = create_test_analysis();

        let chart = engine.create_layer_comparison_chart(&analysis)?;

        assert_eq!(chart.id, "layer_comparison");
        assert_eq!(chart.data_points.len(), 2);
        assert!(matches!(
            chart.visualization_type,
            VisualizationType::GroupedBarChart
        ));

        Ok(())
    }

    #[test]
    fn test_sensitivity_ranking_chart() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);
        let analysis = create_test_analysis();

        let chart = engine.create_sensitivity_ranking_chart(&analysis.sensitivity_ranking)?;

        assert_eq!(chart.id, "sensitivity_ranking");
        assert_eq!(chart.data_points.len(), 2);
        assert!(matches!(
            chart.visualization_type,
            VisualizationType::HorizontalBarChart
        ));

        Ok(())
    }

    #[test]
    fn test_tensor_visualization() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device.clone());

        let original = Tensor::ones((4, 4), DType::F32, &device)?;
        let quantized = original.mul(&Tensor::new(&[0.9f32], &device)?)?;

        let viz = engine.create_tensor_visualization(&original, &quantized, "test_layer")?;

        assert!(viz.id.starts_with("tensor_visualization_"));
        assert!(!viz.data_points.is_empty());
        assert!(matches!(viz.visualization_type, VisualizationType::Heatmap));

        Ok(())
    }

    #[test]
    fn test_data_value_conversion() {
        let float_val: DataValue = 3.14f32.into();
        let int_val: DataValue = 42i64.into();
        let string_val: DataValue = "test".to_string().into();

        assert!(matches!(float_val, DataValue::Float(_)));
        assert!(matches!(int_val, DataValue::Integer(_)));
        assert!(matches!(string_val, DataValue::String(_)));

        assert_eq!(format!("{}", float_val), "3.14");
        assert_eq!(format!("{}", int_val), "42");
        assert_eq!(format!("{}", string_val), "test");
    }

    #[test]
    fn test_csv_conversion() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);

        let visualization = Visualization {
            id: "test".to_string(),
            title: "Test".to_string(),
            visualization_type: VisualizationType::LineChart,
            data_points: vec![
                DataPoint {
                    label: "Point1".to_string(),
                    values: vec![
                        ("X".to_string(), DataValue::Integer(1)),
                        ("Y".to_string(), DataValue::Float(1.5)),
                    ],
                    metadata: None,
                },
                DataPoint {
                    label: "Point2".to_string(),
                    values: vec![
                        ("X".to_string(), DataValue::Integer(2)),
                        ("Y".to_string(), DataValue::Float(2.5)),
                    ],
                    metadata: None,
                },
            ],
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: ColorScheme::Default,
                interactive: false,
                export_formats: vec![OutputFormat::CSV],
            },
        };

        let csv_data = engine.convert_to_csv(&visualization)?;

        assert!(csv_data.contains("label,X,Y"));
        assert!(csv_data.contains("Point1,1,1.5"));
        assert!(csv_data.contains("Point2,2,2.5"));

        Ok(())
    }

    #[test]
    fn test_realtime_dashboard() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);

        let dashboard = engine.create_realtime_dashboard()?;

        assert_eq!(dashboard.id, "realtime_quantization_monitor");
        assert!(dashboard.auto_refresh);
        assert!(!dashboard.panels.is_empty());

        Ok(())
    }

    #[test]
    fn test_visualization_dashboard() {
        let mut dashboard = VisualizationDashboard::new("Test Dashboard");

        let viz = Visualization {
            id: "test_viz".to_string(),
            title: "Test Visualization".to_string(),
            visualization_type: VisualizationType::LineChart,
            data_points: Vec::new(),
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: ColorScheme::Default,
                interactive: false,
                export_formats: vec![OutputFormat::SVG],
            },
        };

        dashboard.add_visualization(viz);

        assert_eq!(dashboard.visualizations.len(), 1);
        assert!(dashboard.get_visualization("test_viz").is_some());
        assert!(dashboard.get_visualization("nonexistent").is_none());
    }

    #[test]
    fn test_export_simulation() -> Result<()> {
        let device = Device::Cpu;
        let engine = VisualizationEngine::new(device);

        let visualization = Visualization {
            id: "test_export".to_string(),
            title: "Test Export".to_string(),
            visualization_type: VisualizationType::LineChart,
            data_points: Vec::new(),
            config: ChartConfig {
                width: 800,
                height: 600,
                show_legend: true,
                color_scheme: ColorScheme::Default,
                interactive: false,
                export_formats: vec![OutputFormat::JSON, OutputFormat::CSV],
            },
        };

        let exported = engine.export_visualization(&visualization)?;

        assert_eq!(exported.len(), 2);
        assert!(exported[0].ends_with(".json"));
        assert!(exported[1].ends_with(".csv"));

        Ok(())
    }
}
