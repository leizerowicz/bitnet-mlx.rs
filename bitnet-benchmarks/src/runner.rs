//! Benchmark Runner
//! 
//! This module provides a command-line interface and utilities for running
//! MLX vs Candle performance benchmarks with detailed metrics and reporting.

use std::fs;
use std::time::Instant;
use clap::{Parser, Subcommand};
use serde_json;

use crate::comparison::{ComparisonConfig, PerformanceComparator};

/// Command-line interface for BitNet benchmarks
#[derive(Parser)]
#[command(name = "bitnet-benchmarks")]
#[command(about = "MLX vs Candle performance benchmarking tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run performance comparison benchmarks
    Compare {
        /// Configuration file path (JSON)
        #[arg(short, long)]
        config: Option<String>,
        
        /// Output directory for results
        #[arg(short, long, default_value = "benchmark_results")]
        output: String,
        
        /// Export format (json, csv, both)
        #[arg(short, long, default_value = "both")]
        format: String,
        
        /// Specific operations to benchmark (comma-separated)
        #[arg(long)]
        operations: Option<String>,
        
        /// Specific tensor sizes to test (format: 128x128,256x256)
        #[arg(long)]
        sizes: Option<String>,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Generate default configuration file
    GenerateConfig {
        /// Output path for configuration file
        #[arg(short, long, default_value = "benchmark_config.json")]
        output: String,
    },
    
    /// Run quick benchmark with default settings
    Quick {
        /// Output directory for results
        #[arg(short, long, default_value = "quick_results")]
        output: String,
    },
    
    /// Analyze existing benchmark results
    Analyze {
        /// Path to benchmark results file
        #[arg(short, long)]
        input: String,
        
        /// Generate detailed report
        #[arg(short, long)]
        detailed: bool,
    },
}

/// Benchmark runner implementation
pub struct BenchmarkRunner {
    config: ComparisonConfig,
    verbose: bool,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with configuration
    pub fn new(config: ComparisonConfig, verbose: bool) -> Self {
        Self { config, verbose }
    }

    /// Create a benchmark runner from configuration file
    pub fn from_config_file(path: &str, verbose: bool) -> anyhow::Result<Self> {
        let config_content = fs::read_to_string(path)?;
        let config: ComparisonConfig = serde_json::from_str(&config_content)?;
        Ok(Self::new(config, verbose))
    }

    /// Create a benchmark runner with default configuration
    pub fn with_defaults(verbose: bool) -> Self {
        Self::new(ComparisonConfig::default(), verbose)
    }

    /// Run the complete benchmark suite
    pub fn run(&self, output_dir: &str, export_format: &str) -> anyhow::Result<()> {
        if self.verbose {
            println!("Starting MLX vs Candle benchmark suite...");
            println!("Configuration: {:?}", self.config);
        }

        // Create output directory
        fs::create_dir_all(output_dir)?;

        let start_time = Instant::now();
        
        // Run benchmarks
        let mut comparator = PerformanceComparator::new(self.config.clone());
        let comparisons = comparator.run_comparison()?;

        let total_time = start_time.elapsed();

        if self.verbose {
            println!("Benchmark completed in {:.2}s", total_time.as_secs_f64());
            println!("Generated {} comparisons", comparisons.len());
        }

        // Export results
        self.export_results(&comparator, &comparisons, output_dir, export_format)?;

        // Print summary
        self.print_summary(&comparisons);

        Ok(())
    }

    /// Run a quick benchmark with minimal configuration
    pub fn run_quick(&self, output_dir: &str) -> anyhow::Result<()> {
        if self.verbose {
            println!("Running quick benchmark...");
        }

        // Create a minimal config for quick testing
        let quick_config = ComparisonConfig {
            tensor_sizes: vec![(128, 128), (512, 512)],
            warmup_iterations: 2,
            measurement_iterations: 5,
            operations: vec!["matmul".to_string(), "add".to_string()],
            ..Default::default()
        };

        let runner = BenchmarkRunner::new(quick_config, self.verbose);
        runner.run(output_dir, "both")
    }

    /// Export benchmark results
    fn export_results(
        &self,
        comparator: &PerformanceComparator,
        comparisons: &[crate::comparison::ComparisonResult],
        output_dir: &str,
        format: &str,
    ) -> anyhow::Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

        match format {
            "json" | "both" => {
                let json_content = comparator.export_json()?;
                let json_path = format!("{output_dir}/benchmark_results_{timestamp}.json");
                fs::write(&json_path, json_content)?;
                
                if self.verbose {
                    println!("JSON results exported to: {json_path}");
                }
            }
            _ => {}
        }

        match format {
            "csv" | "both" => {
                let csv_content = comparator.export_csv();
                let csv_path = format!("{output_dir}/benchmark_results_{timestamp}.csv");
                fs::write(&csv_path, csv_content)?;
                
                if self.verbose {
                    println!("CSV results exported to: {csv_path}");
                }
            }
            _ => {}
        }

        // Export comparison summary
        let summary = self.generate_comparison_summary(comparisons);
        let summary_path = format!("{output_dir}/comparison_summary_{timestamp}.md");
        fs::write(&summary_path, summary)?;
        
        if self.verbose {
            println!("Summary report exported to: {summary_path}");
        }

        Ok(())
    }

    /// Generate a markdown summary of comparisons
    fn generate_comparison_summary(&self, comparisons: &[crate::comparison::ComparisonResult]) -> String {
        let mut summary = String::new();
        
        summary.push_str("# MLX vs Candle Performance Comparison Summary\n\n");
        summary.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Group by operation
        let mut by_operation: std::collections::HashMap<String, Vec<&crate::comparison::ComparisonResult>> = 
            std::collections::HashMap::new();
        
        for comparison in comparisons {
            by_operation.entry(comparison.operation.clone())
                .or_default()
                .push(comparison);
        }

        for (operation, results) in by_operation {
            summary.push_str(&format!("## {operation}\n\n"));
            summary.push_str("| Tensor Size | Baseline | Comparison | Speedup | Recommendation |\n");
            summary.push_str("|-------------|----------|------------|---------|----------------|\n");
            
            for result in results {
                summary.push_str(&format!(
                    "| {}x{} | {} | {} | {:.2}x | {} |\n",
                    result.tensor_size.0,
                    result.tensor_size.1,
                    result.baseline_backend,
                    result.comparison_backend,
                    result.speedup,
                    result.recommendation
                ));
            }
            summary.push('\n');
        }

        // Overall recommendations
        summary.push_str("## Overall Recommendations\n\n");
        let mut recommendations = std::collections::HashMap::new();
        for comparison in comparisons {
            *recommendations.entry(comparison.recommendation.clone()).or_insert(0) += 1;
        }

        for (recommendation, count) in recommendations {
            summary.push_str(&format!("- {recommendation}: {count} cases\n"));
        }

        summary
    }

    /// Print benchmark summary to console
    fn print_summary(&self, comparisons: &[crate::comparison::ComparisonResult]) {
        println!("\n=== Benchmark Summary ===");
        println!("Total comparisons: {}", comparisons.len());
        
        // Find best and worst performers
        if let Some(best) = comparisons.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
            println!("Best speedup: {:.2}x ({} vs {} for {})", 
                best.speedup, best.comparison_backend, best.baseline_backend, best.operation);
        }
        
        if let Some(worst) = comparisons.iter().min_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
            println!("Worst speedup: {:.2}x ({} vs {} for {})", 
                worst.speedup, worst.comparison_backend, worst.baseline_backend, worst.operation);
        }

        // Average speedups by backend
        let mut backend_speedups: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
        for comparison in comparisons {
            backend_speedups.entry(comparison.comparison_backend.clone())
                .or_default()
                .push(comparison.speedup);
        }

        println!("\nAverage speedups by backend:");
        for (backend, speedups) in backend_speedups {
            let avg: f64 = speedups.iter().sum::<f64>() / speedups.len() as f64;
            println!("  {backend}: {avg:.2}x");
        }
    }

    /// Generate default configuration file
    pub fn generate_config_file(output_path: &str) -> anyhow::Result<()> {
        let config = ComparisonConfig::default();
        let config_json = serde_json::to_string_pretty(&config)?;
        fs::write(output_path, config_json)?;
        println!("Default configuration generated: {output_path}");
        Ok(())
    }

    /// Analyze existing benchmark results
    pub fn analyze_results(input_path: &str, detailed: bool) -> anyhow::Result<()> {
        let content = fs::read_to_string(input_path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        println!("=== Benchmark Results Analysis ===");
        
        if let Some(measurements) = data.get("measurements").and_then(|m| m.as_array()) {
            println!("Total measurements: {}", measurements.len());
            
            // Group by backend
            let mut by_backend: std::collections::HashMap<String, Vec<&serde_json::Value>> = 
                std::collections::HashMap::new();
            
            for measurement in measurements {
                if let Some(backend) = measurement.get("backend").and_then(|b| b.as_str()) {
                    by_backend.entry(backend.to_string())
                        .or_default()
                        .push(measurement);
                }
            }
            
            for (backend, measurements) in by_backend {
                println!("\n{} measurements: {}", backend, measurements.len());
                
                if detailed {
                    // Calculate statistics
                    let mut execution_times = Vec::new();
                    let mut throughputs = Vec::new();
                    
                    for measurement in measurements {
                        if let Some(time) = measurement.get("execution_time").and_then(|t| t.get("secs_f64")).and_then(|s| s.as_f64()) {
                            execution_times.push(time);
                        }
                        if let Some(throughput) = measurement.get("throughput").and_then(|t| t.as_f64()) {
                            throughputs.push(throughput);
                        }
                    }
                    
                    if !execution_times.is_empty() {
                        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
                        let min_time = execution_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_time = execution_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        
                        println!("  Execution time - Avg: {avg_time:.4}s, Min: {min_time:.4}s, Max: {max_time:.4}s");
                    }
                    
                    if !throughputs.is_empty() {
                        let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
                        println!("  Average throughput: {avg_throughput:.2} ops/sec");
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Main CLI entry point
pub fn run_cli() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Compare { config, output, format, operations, sizes, verbose } => {
            let mut runner = if let Some(config_path) = config {
                BenchmarkRunner::from_config_file(&config_path, verbose)?
            } else {
                BenchmarkRunner::with_defaults(verbose)
            };
            
            // Override config with CLI options if provided
            if let Some(ops) = operations {
                runner.config.operations = ops.split(',').map(|s| s.trim().to_string()).collect();
            }
            
            if let Some(sizes_str) = sizes {
                let mut sizes = Vec::new();
                for size_str in sizes_str.split(',') {
                    let parts: Vec<&str> = size_str.trim().split('x').collect();
                    if parts.len() == 2 {
                        if let (Ok(rows), Ok(cols)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                            sizes.push((rows, cols));
                        }
                    }
                }
                if !sizes.is_empty() {
                    runner.config.tensor_sizes = sizes;
                }
            }
            
            runner.run(&output, &format)?;
        }
        
        Commands::GenerateConfig { output } => {
            BenchmarkRunner::generate_config_file(&output)?;
        }
        
        Commands::Quick { output } => {
            let runner = BenchmarkRunner::with_defaults(true);
            runner.run_quick(&output)?;
        }
        
        Commands::Analyze { input, detailed } => {
            BenchmarkRunner::analyze_results(&input, detailed)?;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_benchmark_runner_creation() {
        let config = ComparisonConfig::default();
        let runner = BenchmarkRunner::new(config, false);
        assert!(!runner.verbose);
    }

    #[test]
    fn test_generate_config_file() {
        let temp_path = "/tmp/test_config.json";
        BenchmarkRunner::generate_config_file(temp_path).unwrap();
        assert!(Path::new(temp_path).exists());
        std::fs::remove_file(temp_path).ok();
    }
}