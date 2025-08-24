//! Performance Visualization and Reporting
//!
//! This module provides utilities for generating visual reports and charts
//! from benchmark results for better analysis and presentation.

use crate::comparison::{ComparisonResult, PerformanceMeasurement};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub theme: ChartTheme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartTheme {
    Light,
    Dark,
    Professional,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "Performance Comparison".to_string(),
            x_label: "Configuration".to_string(),
            y_label: "Performance".to_string(),
            theme: ChartTheme::Professional,
        }
    }
}

/// Performance visualization generator
pub struct PerformanceVisualizer {
    config: ChartConfig,
}

impl PerformanceVisualizer {
    /// Create a new performance visualizer
    pub fn new(config: ChartConfig) -> Self {
        Self { config }
    }

    /// Generate HTML report with embedded charts
    pub fn generate_html_report(
        &self,
        measurements: &[PerformanceMeasurement],
        comparisons: &[ComparisonResult],
    ) -> Result<String> {
        let mut html = String::new();

        // HTML header with CSS and JavaScript
        html.push_str(&self.generate_html_header());

        // Executive summary
        html.push_str(&self.generate_executive_summary(measurements, comparisons));

        // Performance charts
        html.push_str(&self.generate_performance_charts(measurements));

        // Comparison charts
        html.push_str(&self.generate_comparison_charts(comparisons));

        // Detailed tables
        html.push_str(&self.generate_detailed_tables(measurements, comparisons));

        // HTML footer
        html.push_str(&self.generate_html_footer());

        Ok(html)
    }

    /// Generate SVG chart for performance data
    pub fn generate_performance_chart(
        &self,
        measurements: &[PerformanceMeasurement],
    ) -> Result<String> {
        let mut svg = String::new();

        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));

        // Chart background
        svg.push_str(&format!(
            r#"<rect width="{}" height="{}" fill="{}" />"#,
            self.config.width,
            self.config.height,
            match self.config.theme {
                ChartTheme::Light => "#ffffff",
                ChartTheme::Dark => "#2d3748",
                ChartTheme::Professional => "#f7fafc",
            }
        ));

        // Chart title
        svg.push_str(&format!(
            r#"<text x="{}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
            self.config.width / 2,
            match self.config.theme {
                ChartTheme::Light => "#2d3748",
                ChartTheme::Dark => "#ffffff",
                ChartTheme::Professional => "#1a202c",
            },
            self.config.title
        ));

        // Group measurements by operation
        let mut operation_groups: HashMap<String, Vec<&PerformanceMeasurement>> = HashMap::new();
        for measurement in measurements {
            operation_groups
                .entry(measurement.operation.clone())
                .or_default()
                .push(measurement);
        }

        // Chart area dimensions
        let chart_margin = 60;
        let chart_width = self.config.width - 2 * chart_margin;
        let chart_height = self.config.height - 2 * chart_margin;
        let chart_x = chart_margin;
        let chart_y = chart_margin;

        // Draw axes
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#718096\" stroke-width=\"2\"/>",
            chart_x,
            chart_y + chart_height,
            chart_x + chart_width,
            chart_y + chart_height
        ));
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#718096\" stroke-width=\"2\"/>",
            chart_x,
            chart_y,
            chart_x,
            chart_y + chart_height
        ));

        // Draw bars for each operation
        let bar_width = chart_width / (operation_groups.len() * 2) as u32;
        let mut x_offset = 0;

        let colors = ["#4299e1", "#48bb78", "#ed8936", "#9f7aea", "#f56565"];

        for (i, (operation, group_measurements)) in operation_groups.iter().enumerate() {
            let avg_throughput: f64 = group_measurements.iter().map(|m| m.throughput).sum::<f64>()
                / group_measurements.len() as f64;

            // Normalize bar height (assuming max throughput of 1000 for scaling)
            let max_throughput = 1000.0;
            let bar_height = ((avg_throughput / max_throughput) * chart_height as f64) as u32;

            let bar_x = chart_x + x_offset;
            let bar_y = chart_y + chart_height - bar_height;

            // Draw bar
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.8"/>"#,
                bar_x,
                bar_y,
                bar_width,
                bar_height,
                colors[i % colors.len()]
            ));

            // Add operation label
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"12\" fill=\"#4a5568\">{}</text>",
                bar_x + bar_width / 2,
                chart_y + chart_height + 20,
                operation
            ));

            // Add throughput value
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#2d3748\">{:.1}</text>",
                bar_x + bar_width / 2,
                bar_y - 5,
                avg_throughput
            ));

            x_offset += bar_width * 2;
        }

        // Y-axis labels
        for i in 0..=5 {
            let y = chart_y + chart_height - (i * chart_height / 5);
            let value = (i as f64 / 5.0) * 1000.0;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#718096\">{:.0}</text>",
                chart_x - 10, y + 3, value
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate comparison speedup chart
    pub fn generate_speedup_chart(&self, comparisons: &[ComparisonResult]) -> Result<String> {
        let mut svg = String::new();

        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));

        // Chart background
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"#f7fafc\" />",
            self.config.width, self.config.height
        ));

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"18\" font-weight=\"bold\" fill=\"#1a202c\">Performance Speedup Comparison</text>",
            self.config.width / 2
        ));

        // Chart area
        let chart_margin = 80;
        let chart_width = self.config.width - 2 * chart_margin;
        let chart_height = self.config.height - 2 * chart_margin;
        let chart_x = chart_margin;
        let chart_y = chart_margin;

        // Draw axes
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#718096\" stroke-width=\"2\"/>",
            chart_x,
            chart_y + chart_height,
            chart_x + chart_width,
            chart_y + chart_height
        ));
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#718096\" stroke-width=\"2\"/>",
            chart_x,
            chart_y,
            chart_x,
            chart_y + chart_height
        ));

        // Draw speedup bars
        if !comparisons.is_empty() {
            let bar_width = chart_width / comparisons.len() as u32;

            for (i, comparison) in comparisons.iter().enumerate() {
                let speedup = comparison.speedup;

                // Normalize bar height (assuming max speedup of 5x for scaling)
                let max_speedup = 5.0;
                let normalized_speedup = speedup.min(max_speedup);
                let bar_height = ((normalized_speedup / max_speedup) * chart_height as f64) as u32;

                let bar_x = chart_x + (i as u32 * bar_width);
                let bar_y = chart_y + chart_height - bar_height;

                // Color based on speedup (green for good, red for poor)
                let color = if speedup > 1.5 {
                    "#48bb78" // Green
                } else if speedup > 1.0 {
                    "#ed8936" // Orange
                } else {
                    "#f56565" // Red
                };

                // Draw bar
                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.8"/>"#,
                    bar_x + 5,
                    bar_y,
                    bar_width - 10,
                    bar_height,
                    color
                ));

                // Add speedup value
                svg.push_str(&format!(
                    "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#2d3748\">{:.2}x</text>",
                    bar_x + bar_width / 2,
                    bar_y - 5,
                    speedup
                ));

                // Add operation label (rotated)
                svg.push_str(&format!(
                    "<text x=\"{}\" y=\"{}\" text-anchor=\"start\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#4a5568\" transform=\"rotate(-45, {}, {})\">{}</text>",
                    bar_x + bar_width / 2,
                    chart_y + chart_height + 15,
                    bar_x + bar_width / 2,
                    chart_y + chart_height + 15,
                    "operation" // placeholder since we don't have baseline_metrics field
                ));
            }
        }

        // Y-axis labels for speedup
        for i in 0..=5 {
            let y = chart_y + chart_height - (i * chart_height / 5);
            let value = (i as f64 / 5.0) * 5.0;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#718096\">{:.1}x</text>",
                chart_x - 10, y + 3, value
            ));
        }

        // Add 1x baseline line
        let baseline_y = chart_y + chart_height - (chart_height / 5);
        svg.push_str(&format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#e53e3e\" stroke-width=\"2\" stroke-dasharray=\"5,5\"/>",
            chart_x, baseline_y, chart_x + chart_width, baseline_y
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"Arial, sans-serif\" font-size=\"10\" fill=\"#e53e3e\">1x baseline</text>",
            chart_x + chart_width - 80, baseline_y - 5
        ));

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate detailed performance table
    pub fn generate_performance_table(&self, measurements: &[PerformanceMeasurement]) -> String {
        let mut table = String::new();

        table.push_str(
            r#"
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Backend</th>
                    <th>Device</th>
                    <th>Tensor Size</th>
                    <th>Data Type</th>
                    <th>Execution Time</th>
                    <th>Throughput</th>
                    <th>Memory Usage</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
        "#,
        );

        for measurement in measurements {
            table.push_str(&format!(
                r#"
                <tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}x{}</td>
                    <td>{}</td>
                    <td>{:.3}ms</td>
                    <td>{:.2} ops/sec</td>
                    <td>{:.2} MB</td>
                    <td>{}</td>
                </tr>
                "#,
                measurement.operation,
                measurement.backend,
                measurement.device,
                measurement.tensor_size.0,
                measurement.tensor_size.1,
                measurement.data_type,
                measurement.execution_time.as_millis(),
                measurement.throughput,
                measurement.memory_usage as f64 / (1024.0 * 1024.0),
                if measurement.success { "✅" } else { "❌" }
            ));
        }

        table.push_str("</tbody></table>");
        table
    }

    /// Generate comparison summary table
    pub fn generate_comparison_table(&self, comparisons: &[ComparisonResult]) -> String {
        let mut table = String::new();

        table.push_str(
            r#"
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Baseline</th>
                    <th>Comparison</th>
                    <th>Speedup</th>
                    <th>Throughput Ratio</th>
                    <th>Memory Ratio</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
        "#,
        );

        for comparison in comparisons {
            let speedup_class = if comparison.speedup > 1.5 {
                "speedup-good"
            } else if comparison.speedup > 1.0 {
                "speedup-neutral"
            } else {
                "speedup-poor"
            };

            table.push_str(&format!(
                r#"
                <tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td class="{}">{:.2}x</td>
                    <td>{:.2}x</td>
                    <td>{:.2}x</td>
                    <td>{}</td>
                </tr>
                "#,
                "operation", // placeholder since baseline_metrics field doesn't exist
                comparison.baseline_backend,
                comparison.comparison_backend,
                speedup_class,
                comparison.speedup,
                comparison.throughput_ratio,
                comparison.memory_ratio,
                comparison.recommendation
            ));
        }

        table.push_str("</tbody></table>");
        table
    }

    // Helper methods for HTML generation
    fn generate_html_header(&self) -> String {
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BitNet Performance Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #2d3748;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f7fafc;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4299e1;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .performance-table, .comparison-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .performance-table th, .comparison-table th {
            background: #4a5568;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .performance-table td, .comparison-table td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        .performance-table tr:hover, .comparison-table tr:hover {
            background-color: #f7fafc;
        }
        .speedup-good { color: #38a169; font-weight: bold; }
        .speedup-neutral { color: #d69e2e; font-weight: bold; }
        .speedup-poor { color: #e53e3e; font-weight: bold; }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4299e1;
        }
        .metric-label {
            color: #718096;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 BitNet Performance Analysis Report</h1>
        <p>Comprehensive performance comparison and analysis</p>
    </div>
"#
        .to_string()
    }

    fn generate_executive_summary(
        &self,
        measurements: &[PerformanceMeasurement],
        comparisons: &[ComparisonResult],
    ) -> String {
        let total_operations = measurements.len();
        let avg_throughput = if !measurements.is_empty() {
            measurements.iter().map(|m| m.throughput).sum::<f64>() / measurements.len() as f64
        } else {
            0.0
        };

        let best_speedup = comparisons
            .iter()
            .map(|c| c.speedup)
            .fold(0.0f64, |a, b| a.max(b));

        let success_rate = if !measurements.is_empty() {
            measurements.iter().filter(|m| m.success).count() as f64 / measurements.len() as f64
                * 100.0
        } else {
            0.0
        };

        format!(
            r#"
    <div class="summary-grid">
        <div class="summary-card">
            <div class="metric-value">{total_operations}</div>
            <div class="metric-label">Total Operations Tested</div>
        </div>
        <div class="summary-card">
            <div class="metric-value">{avg_throughput:.1}</div>
            <div class="metric-label">Average Throughput (ops/sec)</div>
        </div>
        <div class="summary-card">
            <div class="metric-value">{best_speedup:.2}x</div>
            <div class="metric-label">Best Speedup Achieved</div>
        </div>
        <div class="summary-card">
            <div class="metric-value">{success_rate:.1}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
    </div>
"#
        )
    }

    fn generate_performance_charts(&self, measurements: &[PerformanceMeasurement]) -> String {
        let chart = self
            .generate_performance_chart(measurements)
            .unwrap_or_default();
        format!(
            r#"
    <div class="chart-container">
        <h2>📊 Performance Overview</h2>
        {chart}
    </div>
"#
        )
    }

    fn generate_comparison_charts(&self, comparisons: &[ComparisonResult]) -> String {
        let chart = self.generate_speedup_chart(comparisons).unwrap_or_default();
        format!(
            r#"
    <div class="chart-container">
        <h2>⚡ Speedup Comparison</h2>
        {chart}
    </div>
"#
        )
    }

    fn generate_detailed_tables(
        &self,
        measurements: &[PerformanceMeasurement],
        comparisons: &[ComparisonResult],
    ) -> String {
        let perf_table = self.generate_performance_table(measurements);
        let comp_table = self.generate_comparison_table(comparisons);

        format!(
            r#"
    <div class="chart-container">
        <h2>📋 Detailed Performance Results</h2>
        {perf_table}
    </div>

    <div class="chart-container">
        <h2>🔄 Performance Comparisons</h2>
        {comp_table}
    </div>
"#
        )
    }

    fn generate_html_footer(&self) -> String {
        format!(
            r#"
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #718096; border-top: 1px solid #e2e8f0;">
        <p>Generated on {} by BitNet Performance Benchmarking Suite</p>
        <p>🔬 For more detailed analysis, check the raw benchmark data and logs</p>
    </div>
</body>
</html>
"#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )
    }
}

impl Default for PerformanceVisualizer {
    fn default() -> Self {
        Self::new(ChartConfig::default())
    }
}

/// Export performance data to various formats
pub struct PerformanceExporter;

impl PerformanceExporter {
    /// Export to JSON format
    pub fn export_json(
        measurements: &[PerformanceMeasurement],
        comparisons: &[ComparisonResult],
    ) -> Result<String> {
        let data = serde_json::json!({
            "measurements": measurements,
            "comparisons": comparisons,
            "metadata": {
                "generated_at": chrono::Utc::now(),
                "total_measurements": measurements.len(),
                "total_comparisons": comparisons.len(),
            }
        });

        serde_json::to_string_pretty(&data)
            .map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {}", e))
    }

    /// Export to CSV format
    pub fn export_csv(measurements: &[PerformanceMeasurement]) -> String {
        let mut csv = String::new();
        csv.push_str("operation,backend,device,tensor_size,data_type,execution_time_ms,throughput,memory_usage_mb,success\n");

        for measurement in measurements {
            csv.push_str(&format!(
                "{},{},{},{}x{},{},{},{:.2},{:.2},{}\n",
                measurement.operation,
                measurement.backend,
                measurement.device,
                measurement.tensor_size.0,
                measurement.tensor_size.1,
                measurement.data_type,
                measurement.execution_time.as_millis(),
                measurement.throughput,
                measurement.memory_usage as f64 / (1024.0 * 1024.0),
                measurement.success
            ));
        }

        csv
    }

    /// Export comparison results to CSV
    pub fn export_comparison_csv(comparisons: &[ComparisonResult]) -> String {
        let mut csv = String::new();
        csv.push_str("operation,baseline_backend,comparison_backend,speedup,throughput_ratio,memory_ratio,recommendation\n");

        for comparison in comparisons {
            csv.push_str(&format!(
                "{},{},{},{:.3},{:.3},{:.3},{}\n",
                comparison.operation,
                comparison.baseline_backend,
                comparison.comparison_backend,
                comparison.speedup,
                comparison.throughput_ratio,
                comparison.memory_ratio,
                comparison.recommendation
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_chart_config_default() {
        let config = ChartConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_performance_visualizer_creation() {
        let visualizer = PerformanceVisualizer::default();
        assert_eq!(visualizer.config.width, 800);
    }

    #[test]
    fn test_json_export() {
        let measurements = vec![PerformanceMeasurement {
            operation: "test".to_string(),
            backend: "candle".to_string(),
            device: "cpu".to_string(),
            tensor_size: (100, 100),
            data_type: "f32".to_string(),
            execution_time: Duration::from_millis(10),
            throughput: 100.0,
            memory_usage: 1024,
            success: true,
            error_message: None,
            timestamp: SystemTime::now(),
        }];

        let json = PerformanceExporter::export_json(&measurements, &[]).unwrap();
        assert!(json.contains("measurements"));
        assert!(json.contains("test"));
    }
}
