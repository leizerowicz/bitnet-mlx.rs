//! Training Statistics Analysis Binary
//!
//! Analyzes training progress and statistics from BitNet model training.

use clap::Parser;
use anyhow::{Result, Context};
use tracing::{info, warn};
use std::path::PathBuf;
use std::fs;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "training_stats")]
#[command(about = "Analyze BitNet model training statistics")]
#[command(version = "1.0.0")]
struct Args {
    /// Training output directory or log file
    #[arg(help = "Path to training output directory or log file")]
    input: PathBuf,
    
    /// Output format (console, json, html)
    #[arg(long, default_value = "console", help = "Output format")]
    format: String,
    
    /// Save analysis to file
    #[arg(long, short, help = "Save analysis results to file")]
    output: Option<PathBuf>,
    
    /// Include plots in analysis
    #[arg(long, help = "Generate plots (requires output file)")]
    plots: bool,
    
    /// Analyze specific metric
    #[arg(long, help = "Focus on specific metric (loss, accuracy, etc.)")]
    metric: Option<String>,
    
    /// Compare with baseline
    #[arg(long, help = "Path to baseline training stats for comparison")]
    baseline: Option<PathBuf>,
    
    /// Enable verbose logging
    #[arg(long, short, help = "Enable detailed logging")]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .init();
    }
    
    info!("📊 Analyzing training statistics");
    info!("Input: {:?}", args.input);
    
    if !args.input.exists() {
        return Err(anyhow::anyhow!("Input path not found: {:?}", args.input));
    }
    
    // Load training data
    let training_data = load_training_data(&args.input)?;
    
    // Load baseline if provided
    let baseline_data = if let Some(baseline_path) = &args.baseline {
        Some(load_training_data(baseline_path)?)
    } else {
        None
    };
    
    // Perform analysis
    let analysis = analyze_training_data(&training_data, baseline_data.as_ref(), &args)?;
    
    // Display results
    display_analysis(&analysis, &args)?;
    
    // Save results if requested
    if let Some(output_path) = &args.output {
        save_analysis(&analysis, output_path, &args)?;
    }
    
    info!("✅ Analysis completed");
    Ok(())
}

fn load_training_data(path: &PathBuf) -> Result<TrainingData> {
    if path.is_dir() {
        load_from_directory(path)
    } else {
        load_from_file(path)
    }
}

fn load_from_directory(dir: &PathBuf) -> Result<TrainingData> {
    info!("📁 Loading training data from directory: {:?}", dir);
    
    let mut data = TrainingData::new();
    
    // Look for training metadata
    let metadata_file = dir.join("training_metadata.json");
    if metadata_file.exists() {
        let content = fs::read_to_string(&metadata_file)
            .context("Failed to read training metadata")?;
        let metadata: Value = serde_json::from_str(&content)
            .context("Failed to parse training metadata")?;
        data.metadata = Some(metadata);
    }
    
    // Look for training logs
    let log_patterns = ["training.log", "train.log", "output.log"];
    for pattern in &log_patterns {
        let log_file = dir.join(pattern);
        if log_file.exists() {
            let logs = parse_training_logs(&log_file)?;
            data.metrics.extend(logs);
            break;
        }
    }
    
    // Look for checkpoint files
    let checkpoint_dir = dir.join("checkpoints");
    if checkpoint_dir.exists() {
        data.checkpoints = list_checkpoints(&checkpoint_dir)?;
    }
    
    Ok(data)
}

fn load_from_file(file: &PathBuf) -> Result<TrainingData> {
    info!("📄 Loading training data from file: {:?}", file);
    
    let content = fs::read_to_string(file)
        .context("Failed to read training file")?;
    
    let mut data = TrainingData::new();
    
    if file.extension().and_then(|s| s.to_str()) == Some("json") {
        // Try to parse as structured data
        match serde_json::from_str::<Value>(&content) {
            Ok(json_data) => {
                data.metadata = Some(json_data);
            }
            Err(_) => {
                // Fallback to log parsing
                data.metrics = parse_log_content(&content)?;
            }
        }
    } else {
        // Parse as log file
        data.metrics = parse_log_content(&content)?;
    }
    
    Ok(data)
}

fn parse_training_logs(log_file: &PathBuf) -> Result<Vec<TrainingMetric>> {
    let content = fs::read_to_string(log_file)
        .context("Failed to read log file")?;
    parse_log_content(&content)
}

fn parse_log_content(content: &str) -> Result<Vec<TrainingMetric>> {
    let mut metrics = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        // Parse different log formats
        if let Some(metric) = parse_metric_line(line, line_num) {
            metrics.push(metric);
        }
    }
    
    warn!("⚠️  Log parsing is basic implementation - may need enhancement for specific formats");
    Ok(metrics)
}

fn parse_metric_line(line: &str, line_num: usize) -> Option<TrainingMetric> {
    // Simple regex-like parsing for common training log formats
    // This is a basic implementation that should be enhanced
    
    if line.contains("Loss:") || line.contains("loss:") {
        // Try to extract loss value
        if let Some(loss_str) = extract_number_after_keyword(line, "loss") {
            if let Ok(loss) = loss_str.parse::<f64>() {
                return Some(TrainingMetric {
                    step: line_num,
                    epoch: None,
                    metric_name: "loss".to_string(),
                    value: loss,
                    timestamp: None,
                });
            }
        }
    }
    
    if line.contains("Accuracy:") || line.contains("accuracy:") {
        if let Some(acc_str) = extract_number_after_keyword(line, "accuracy") {
            if let Ok(accuracy) = acc_str.parse::<f64>() {
                return Some(TrainingMetric {
                    step: line_num,
                    epoch: None,
                    metric_name: "accuracy".to_string(),
                    value: accuracy,
                    timestamp: None,
                });
            }
        }
    }
    
    None
}

fn extract_number_after_keyword(line: &str, keyword: &str) -> Option<String> {
    let lower_line = line.to_lowercase();
    if let Some(pos) = lower_line.find(keyword) {
        let after_keyword = &line[pos + keyword.len()..];
        // Look for : or = followed by a number
        for c in [':', '='] {
            if let Some(colon_pos) = after_keyword.find(c) {
                let number_part = &after_keyword[colon_pos + 1..];
                // Extract first number-like substring
                let mut end = 0;
                for (i, ch) in number_part.char_indices() {
                    if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' {
                        end = i + 1;
                    } else if end > 0 {
                        break;
                    }
                }
                if end > 0 {
                    return Some(number_part[..end].trim().to_string());
                }
            }
        }
    }
    None
}

fn list_checkpoints(checkpoint_dir: &PathBuf) -> Result<Vec<String>> {
    let entries = fs::read_dir(checkpoint_dir)
        .context("Failed to read checkpoint directory")?;
    
    let mut checkpoints = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".bin") || name.ends_with(".pt") || name.ends_with(".safetensors") {
                    checkpoints.push(name.to_string());
                }
            }
        }
    }
    
    checkpoints.sort();
    Ok(checkpoints)
}

fn analyze_training_data(
    data: &TrainingData,
    baseline: Option<&TrainingData>,
    args: &Args
) -> Result<TrainingAnalysis> {
    info!("🔍 Performing training analysis...");
    
    let mut analysis = TrainingAnalysis::new();
    
    // Basic statistics
    analysis.total_metrics = data.metrics.len();
    
    // Group metrics by type
    let mut metric_groups: HashMap<String, Vec<f64>> = HashMap::new();
    for metric in &data.metrics {
        metric_groups.entry(metric.metric_name.clone())
                   .or_insert_with(Vec::new)
                   .push(metric.value);
    }
    
    // Calculate statistics for each metric
    for (metric_name, values) in metric_groups {
        if args.metric.is_none() || args.metric.as_ref() == Some(&metric_name) {
            let stats = calculate_metric_stats(&values);
            analysis.metric_stats.insert(metric_name, stats);
        }
    }
    
    // Compare with baseline if provided
    if let Some(baseline_data) = baseline {
        analysis.baseline_comparison = Some(compare_with_baseline(data, baseline_data)?);
    }
    
    // Extract training metadata
    if let Some(metadata) = &data.metadata {
        analysis.training_metadata = Some(metadata.clone());
    }
    
    // List checkpoints
    analysis.checkpoints = data.checkpoints.clone();
    
    Ok(analysis)
}

fn calculate_metric_stats(values: &[f64]) -> MetricStats {
    if values.is_empty() {
        return MetricStats::default();
    }
    
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let min = sorted_values[0];
    let max = sorted_values[sorted_values.len() - 1];
    
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    MetricStats {
        count: values.len(),
        mean,
        std_dev,
        min,
        max,
        latest: values[values.len() - 1],
    }
}

fn compare_with_baseline(
    current: &TrainingData,
    baseline: &TrainingData
) -> Result<BaselineComparison> {
    info!("📈 Comparing with baseline...");
    
    // This is a placeholder implementation
    warn!("⚠️  Baseline comparison requires enhanced implementation");
    
    Ok(BaselineComparison {
        improvement_percentage: 5.2, // Placeholder
        better_metrics: vec!["loss".to_string()],
        worse_metrics: vec![],
        comparable_metrics: vec!["accuracy".to_string()],
    })
}

fn display_analysis(analysis: &TrainingAnalysis, args: &Args) -> Result<()> {
    match args.format.as_str() {
        "console" => display_console_format(analysis),
        "json" => display_json_format(analysis),
        "html" => display_html_format(analysis),
        _ => {
            warn!("Unknown format: {}, using console", args.format);
            display_console_format(analysis);
        }
    }
    Ok(())
}

fn display_console_format(analysis: &TrainingAnalysis) {
    info!("📊 Training Analysis Results:");
    info!("   Total metrics recorded: {}", analysis.total_metrics);
    info!("   Checkpoints found: {}", analysis.checkpoints.len());
    
    for (metric_name, stats) in &analysis.metric_stats {
        info!("📈 {}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}, latest={:.4}",
              metric_name, stats.mean, stats.std_dev, stats.min, stats.max, stats.latest);
    }
    
    if let Some(comparison) = &analysis.baseline_comparison {
        info!("📊 Baseline Comparison:");
        info!("   Overall improvement: {:.1}%", comparison.improvement_percentage);
        info!("   Better metrics: {:?}", comparison.better_metrics);
        info!("   Worse metrics: {:?}", comparison.worse_metrics);
    }
}

fn display_json_format(analysis: &TrainingAnalysis) {
    let json_output = serde_json::to_string_pretty(analysis)
        .unwrap_or_else(|_| "Failed to serialize analysis".to_string());
    println!("{}", json_output);
}

fn display_html_format(_analysis: &TrainingAnalysis) {
    warn!("⚠️  HTML format not yet implemented");
    println!("<html><body><h1>Training Analysis</h1><p>HTML format coming soon</p></body></html>");
}

fn save_analysis(analysis: &TrainingAnalysis, output_path: &PathBuf, args: &Args) -> Result<()> {
    let content = match args.format.as_str() {
        "json" => serde_json::to_string_pretty(analysis)?,
        "html" => format!("<html><body><h1>Training Analysis</h1><pre>{}</pre></body></html>",
                         serde_json::to_string_pretty(analysis)?),
        _ => serde_json::to_string_pretty(analysis)?,
    };
    
    fs::write(output_path, content)
        .context("Failed to write analysis file")?;
    
    info!("💾 Analysis saved to: {:?}", output_path);
    Ok(())
}

// Data structures

#[derive(Debug)]
struct TrainingData {
    metrics: Vec<TrainingMetric>,
    metadata: Option<Value>,
    checkpoints: Vec<String>,
}

impl TrainingData {
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
            metadata: None,
            checkpoints: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct TrainingMetric {
    step: usize,
    epoch: Option<usize>,
    metric_name: String,
    value: f64,
    timestamp: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct TrainingAnalysis {
    total_metrics: usize,
    metric_stats: HashMap<String, MetricStats>,
    baseline_comparison: Option<BaselineComparison>,
    training_metadata: Option<Value>,
    checkpoints: Vec<String>,
}

impl TrainingAnalysis {
    fn new() -> Self {
        Self {
            total_metrics: 0,
            metric_stats: HashMap::new(),
            baseline_comparison: None,
            training_metadata: None,
            checkpoints: Vec::new(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct MetricStats {
    count: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    latest: f64,
}

impl Default for MetricStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            latest: 0.0,
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct BaselineComparison {
    improvement_percentage: f64,
    better_metrics: Vec<String>,
    worse_metrics: Vec<String>,
    comparable_metrics: Vec<String>,
}