use anyhow::Result;
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, GenerationConfig};
use bitnet_inference::bitnet_config::TokenizerConfig;
use bitnet_core::Device;
use bitnet_benchmarks::{BenchmarkRunner, BenchmarkConfig, BenchmarkResult};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use colored::*;
use sysinfo::{System, SystemExt, ProcessorExt};

/// BitNet Performance Benchmarking Application
/// 
/// Comprehensive benchmarking tools for measuring BitNet performance across
/// different configurations, hardware setups, and model parameters.
#[derive(Parser, Debug)]
#[command(name = "bitnet-benchmark")]
#[command(version = "1.0.0")]
#[command(about = "Performance benchmarking tools for BitNet neural networks")]
pub struct BenchmarkArgs {
    /// Model name or path
    #[arg(short, long, default_value = "microsoft/bitnet-b1.58-2B-4T-gguf")]
    pub model: String,

    /// Devices to benchmark (auto, cpu, metal, cuda, all)
    #[arg(short, long, default_value = "auto")]
    pub devices: String,

    /// Benchmark type (inference, memory, throughput, latency, all)
    #[arg(short, long, default_value = "all")]
    pub benchmark_type: String,

    /// Number of warmup iterations
    #[arg(long, default_value = "5")]
    pub warmup_iterations: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "20")]
    pub benchmark_iterations: usize,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<std::path::PathBuf>,

    /// Enable detailed profiling
    #[arg(long)]
    pub detailed_profiling: bool,

    /// Compare with baseline results
    #[arg(long)]
    pub baseline_file: Option<std::path::PathBuf>,

    /// Temperature values to test
    #[arg(long, default_value = "0.1,0.7,1.0")]
    pub temperatures: String,

    /// Token counts to test
    #[arg(long, default_value = "50,100,256,512")]
    pub token_counts: String,

    /// Batch sizes to test
    #[arg(long, default_value = "1,4,8,16")]
    pub batch_sizes: String,

    /// Enable continuous monitoring mode
    #[arg(long)]
    pub monitor: bool,

    /// Monitor duration in minutes
    #[arg(long, default_value = "10")]
    pub monitor_duration: u64,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfiguration {
    pub device: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub batch_size: usize,
    pub top_k: u32,
    pub top_p: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub memory_peak_mb: f64,
    pub cpu_usage_percent: f64,
    pub gpu_utilization_percent: Option<f64>,
    pub energy_efficiency_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub configuration: BenchmarkConfiguration,
    pub metrics: PerformanceMetrics,
    pub error_rate: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub os: String,
    pub architecture: String,
    pub gpu_info: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub summary: BenchmarkSummary,
    pub results: Vec<BenchmarkEntry>,
    pub recommendations: Vec<String>,
    pub comparison: Option<BenchmarkComparison>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub best_configuration: BenchmarkConfiguration,
    pub peak_performance: PerformanceMetrics,
    pub average_performance: PerformanceMetrics,
    pub total_benchmarks: usize,
    pub successful_benchmarks: usize,
    pub benchmark_duration_minutes: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_file: String,
    pub performance_change_percent: f64,
    pub memory_change_percent: f64,
    pub improvements: Vec<String>,
    pub regressions: Vec<String>,
}

pub struct PerformanceBenchmarker {
    config: BenchmarkArgs,
    system_info: SystemInfo,
    results: Vec<BenchmarkEntry>,
}

impl PerformanceBenchmarker {
    /// Initialize the benchmarker
    pub fn new(config: BenchmarkArgs) -> Result<Self> {
        println!("{}", "🏁 BitNet Performance Benchmarker".bright_blue().bold());
        
        let system_info = Self::collect_system_info();
        println!("System: {} ({}) - {} cores, {:.1} GB RAM", 
            system_info.cpu_model.cyan(),
            system_info.architecture,
            system_info.cpu_cores,
            system_info.total_memory_gb
        );

        Ok(Self {
            config,
            system_info,
            results: Vec::new(),
        })
    }

    /// Run comprehensive benchmarks
    pub async fn run_benchmarks(&mut self) -> Result<BenchmarkReport> {
        let start_time = Instant::now();

        // Parse configuration parameters
        let devices = self.parse_devices()?;
        let temperatures = self.parse_temperatures()?;
        let token_counts = self.parse_token_counts()?;
        let batch_sizes = self.parse_batch_sizes()?;

        println!("📊 Starting benchmarks with {} configurations", 
            devices.len() * temperatures.len() * token_counts.len() * batch_sizes.len());

        // Run benchmarks for each configuration
        for device in &devices {
            for &temperature in &temperatures {
                for &max_tokens in &token_counts {
                    for &batch_size in &batch_sizes {
                        let config = BenchmarkConfiguration {
                            device: format!("{:?}", device),
                            temperature,
                            max_tokens,
                            batch_size,
                            top_k: 50,
                            top_p: 0.9,
                        };

                        if let Some(entry) = self.run_single_benchmark(config, device.clone()).await? {
                            self.results.push(entry);
                        }
                    }
                }
            }
        }

        let benchmark_duration = start_time.elapsed();
        
        // Generate report
        let report = self.generate_report(benchmark_duration)?;
        
        // Save results if output file specified
        if let Some(output_path) = &self.config.output {
            self.save_results(&report, output_path)?;
            println!("📁 Results saved to: {}", output_path.display().to_string().green());
        }

        // Print summary
        self.print_summary(&report);

        Ok(report)
    }

    /// Run a single benchmark configuration
    async fn run_single_benchmark(
        &self,
        config: BenchmarkConfiguration,
        device: Device,
    ) -> Result<Option<BenchmarkEntry>> {
        if self.config.verbose {
            println!("🔄 Testing: {:?} | temp={:.1} | tokens={} | batch={}", 
                device, config.temperature, config.max_tokens, config.batch_size);
        }

        // Initialize engine for this configuration
        let engine_config = EngineConfig {
            device,
            memory_limit_mb: Some(4096),
            enable_profiling: self.config.detailed_profiling,
            ..Default::default()
        };

        let engine = match InferenceEngine::with_config(engine_config).await {
            Ok(engine) => engine,
            Err(e) => {
                if self.config.verbose {
                    println!("❌ Failed to initialize engine: {}", e);
                }
                return Ok(None);
            }
        };

        // Load model
        let model_handle = match engine.load_model_from_hub(&self.config.model).await {
            Ok(handle) => handle,
            Err(e) => {
                if self.config.verbose {
                    println!("❌ Failed to load model: {}", e);
                }
                return Ok(None);
            }
        };

        // Configure text generation
        let generation_config = GenerationConfig {
            temperature: config.temperature,
            top_k: Some(config.top_k as usize),
            top_p: Some(config.top_p),
            typical_p: Some(0.95),
            max_length: config.max_tokens as usize,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: Some(42), // Fixed seed for reproducible benchmarks
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        };

        let tokenizer_config = TokenizerConfig {
            vocab_size: 128256,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
            pad_token_id: Some(128002),
        };

        let generator = TextGenerator::new(
            engine,
            model_handle,
            generation_config,
            tokenizer_config,
        );

        // Warmup iterations
        let warmup_prompt = "This is a warmup prompt for benchmarking purposes.";
        for _ in 0..self.config.warmup_iterations {
            if let Err(_) = timeout(
                Duration::from_secs(30),
                generator.generate(warmup_prompt)
            ).await {
                if self.config.verbose {
                    println!("⏰ Warmup timeout");
                }
                return Ok(None);
            }
        }

        // Benchmark iterations
        let mut latencies = Vec::new();
        let mut token_counts = Vec::new();
        let mut memory_samples = Vec::new();
        let mut error_count = 0;

        let benchmark_prompts = vec![
            "Explain the concept of artificial intelligence in simple terms.",
            "Describe the process of machine learning and its applications.",
            "What are the benefits and challenges of neural network quantization?",
            "Compare different approaches to model compression in deep learning.",
            "Discuss the future of efficient AI inference on edge devices.",
        ];

        for i in 0..self.config.benchmark_iterations {
            let prompt = &benchmark_prompts[i % benchmark_prompts.len()];
            
            // Measure memory before generation
            let memory_before = self.get_memory_usage();
            
            let start_time = Instant::now();
            
            match timeout(
                Duration::from_secs(30),
                generator.generate(prompt)
            ).await {
                Ok(Ok(result)) => {
                    let latency = start_time.elapsed();
                    latencies.push(latency.as_millis() as f64);
                    token_counts.push(result.token_count);
                    
                    // Measure memory after generation
                    let memory_after = self.get_memory_usage();
                    memory_samples.push(memory_after - memory_before);
                }
                Ok(Err(_)) | Err(_) => {
                    error_count += 1;
                }
            }
        }

        if latencies.is_empty() {
            return Ok(None);
        }

        // Calculate metrics
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let total_tokens: usize = token_counts.iter().sum();
        let total_time_seconds = latencies.iter().sum::<f64>() / 1000.0;
        let tokens_per_second = if total_time_seconds > 0.0 {
            total_tokens as f64 / total_time_seconds
        } else {
            0.0
        };

        let avg_memory_usage = if !memory_samples.is_empty() {
            memory_samples.iter().sum::<f64>() / memory_samples.len() as f64
        } else {
            0.0
        };

        let peak_memory = memory_samples.iter().cloned().fold(0.0_f64, f64::max);

        let metrics = PerformanceMetrics {
            tokens_per_second,
            latency_ms: avg_latency,
            memory_usage_mb: avg_memory_usage,
            memory_peak_mb: peak_memory,
            cpu_usage_percent: self.get_cpu_usage(),
            gpu_utilization_percent: None, // TODO: Implement GPU monitoring
            energy_efficiency_score: None, // TODO: Implement energy monitoring
        };

        let error_rate = (error_count as f64) / (self.config.benchmark_iterations as f64) * 100.0;

        let entry = BenchmarkEntry {
            configuration: config,
            metrics,
            error_rate,
            timestamp: chrono::Utc::now(),
            system_info: self.system_info.clone(),
        };

        if self.config.verbose {
            println!("✅ {:.1} tok/s | {:.1}ms latency | {:.1} MB memory | {:.1}% errors",
                tokens_per_second, avg_latency, avg_memory_usage, error_rate);
        }

        Ok(Some(entry))
    }

    /// Generate comprehensive benchmark report
    fn generate_report(&self, benchmark_duration: Duration) -> Result<BenchmarkReport> {
        if self.results.is_empty() {
            return Err(anyhow::anyhow!("No benchmark results available"));
        }

        // Find best configuration
        let best_result = self.results
            .iter()
            .max_by(|a, b| a.metrics.tokens_per_second.partial_cmp(&b.metrics.tokens_per_second).unwrap())
            .unwrap();

        // Calculate averages
        let avg_tokens_per_second = self.results.iter()
            .map(|r| r.metrics.tokens_per_second)
            .sum::<f64>() / self.results.len() as f64;

        let avg_latency = self.results.iter()
            .map(|r| r.metrics.latency_ms)
            .sum::<f64>() / self.results.len() as f64;

        let avg_memory = self.results.iter()
            .map(|r| r.metrics.memory_usage_mb)
            .sum::<f64>() / self.results.len() as f64;

        let summary = BenchmarkSummary {
            best_configuration: best_result.configuration.clone(),
            peak_performance: best_result.metrics.clone(),
            average_performance: PerformanceMetrics {
                tokens_per_second: avg_tokens_per_second,
                latency_ms: avg_latency,
                memory_usage_mb: avg_memory,
                memory_peak_mb: 0.0,
                cpu_usage_percent: 0.0,
                gpu_utilization_percent: None,
                energy_efficiency_score: None,
            },
            total_benchmarks: self.results.len(),
            successful_benchmarks: self.results.iter().filter(|r| r.error_rate < 5.0).count(),
            benchmark_duration_minutes: benchmark_duration.as_secs_f64() / 60.0,
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations();

        // Load and compare with baseline if provided
        let comparison = if let Some(baseline_path) = &self.config.baseline_file {
            self.compare_with_baseline(baseline_path)?
        } else {
            None
        };

        Ok(BenchmarkReport {
            summary,
            results: self.results.clone(),
            recommendations,
            comparison,
        })
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze device performance
        let cpu_results: Vec<_> = self.results.iter().filter(|r| r.configuration.device == "Cpu").collect();
        let gpu_results: Vec<_> = self.results.iter().filter(|r| r.configuration.device != "Cpu").collect();

        if !cpu_results.is_empty() && !gpu_results.is_empty() {
            let cpu_avg_perf = cpu_results.iter().map(|r| r.metrics.tokens_per_second).sum::<f64>() / cpu_results.len() as f64;
            let gpu_avg_perf = gpu_results.iter().map(|r| r.metrics.tokens_per_second).sum::<f64>() / gpu_results.len() as f64;

            if gpu_avg_perf > cpu_avg_perf * 1.2 {
                recommendations.push("GPU acceleration provides significant performance benefits. Consider using GPU for inference.".to_string());
            } else if cpu_avg_perf > gpu_avg_perf {
                recommendations.push("CPU performance is competitive with GPU. Consider CPU for energy efficiency.".to_string());
            }
        }

        // Analyze temperature impact
        let low_temp_results: Vec<_> = self.results.iter().filter(|r| r.configuration.temperature < 0.5).collect();
        let high_temp_results: Vec<_> = self.results.iter().filter(|r| r.configuration.temperature > 0.8).collect();

        if !low_temp_results.is_empty() && !high_temp_results.is_empty() {
            let low_temp_perf = low_temp_results.iter().map(|r| r.metrics.tokens_per_second).sum::<f64>() / low_temp_results.len() as f64;
            let high_temp_perf = high_temp_results.iter().map(|r| r.metrics.tokens_per_second).sum::<f64>() / high_temp_results.len() as f64;

            if (high_temp_perf - low_temp_perf).abs() / low_temp_perf > 0.1 {
                recommendations.push("Temperature setting significantly impacts performance. Tune for your use case.".to_string());
            }
        }

        // Memory usage recommendations
        let high_memory_results: Vec<_> = self.results.iter().filter(|r| r.metrics.memory_usage_mb > 2048.0).collect();
        if !high_memory_results.is_empty() {
            recommendations.push("High memory usage detected. Consider enabling memory optimization or using smaller batch sizes.".to_string());
        }

        // Performance consistency
        let performance_variance = self.calculate_performance_variance();
        if performance_variance > 0.2 {
            recommendations.push("High performance variance detected. Consider warmup periods and consistent system load.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Performance appears optimal for current configuration.".to_string());
        }

        recommendations
    }

    /// Calculate performance variance across results
    fn calculate_performance_variance(&self) -> f64 {
        if self.results.len() < 2 {
            return 0.0;
        }

        let mean = self.results.iter().map(|r| r.metrics.tokens_per_second).sum::<f64>() / self.results.len() as f64;
        let variance = self.results.iter()
            .map(|r| (r.metrics.tokens_per_second - mean).powi(2))
            .sum::<f64>() / self.results.len() as f64;
        
        (variance.sqrt() / mean).abs()
    }

    /// Compare with baseline results
    fn compare_with_baseline(&self, baseline_path: &std::path::Path) -> Result<Option<BenchmarkComparison>> {
        // Implementation would load baseline and compare
        // For now, return None
        Ok(None)
    }

    /// Print benchmark summary
    fn print_summary(&self, report: &BenchmarkReport) {
        println!();
        println!("{}", "📊 Benchmark Results Summary".bright_green().bold());
        println!();
        
        println!("{}", "🏆 Best Configuration:".bright_yellow().bold());
        println!("  Device: {}", report.summary.best_configuration.device.cyan());
        println!("  Temperature: {:.1}", report.summary.best_configuration.temperature);
        println!("  Max Tokens: {}", report.summary.best_configuration.max_tokens);
        println!("  Batch Size: {}", report.summary.best_configuration.batch_size);
        println!();
        
        println!("{}", "⚡ Peak Performance:".bright_yellow().bold());
        println!("  Tokens/sec: {:.2}", report.summary.peak_performance.tokens_per_second);
        println!("  Latency: {:.1} ms", report.summary.peak_performance.latency_ms);
        println!("  Memory: {:.1} MB", report.summary.peak_performance.memory_usage_mb);
        println!();
        
        println!("{}", "📈 Average Performance:".bright_yellow().bold());
        println!("  Tokens/sec: {:.2}", report.summary.average_performance.tokens_per_second);
        println!("  Latency: {:.1} ms", report.summary.average_performance.latency_ms);
        println!("  Memory: {:.1} MB", report.summary.average_performance.memory_usage_mb);
        println!();
        
        println!("{}", "📋 Statistics:".bright_yellow().bold());
        println!("  Total Benchmarks: {}", report.summary.total_benchmarks);
        println!("  Successful: {}", report.summary.successful_benchmarks);
        println!("  Success Rate: {:.1}%", 
            (report.summary.successful_benchmarks as f64 / report.summary.total_benchmarks as f64) * 100.0);
        println!("  Duration: {:.1} minutes", report.summary.benchmark_duration_minutes);
        println!();
        
        println!("{}", "💡 Recommendations:".bright_yellow().bold());
        for (i, recommendation) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, recommendation);
        }
        println!();
    }

    /// Save benchmark results to file
    fn save_results(&self, report: &BenchmarkReport, output_path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(output_path, json)?;
        Ok(())
    }

    /// Collect system information
    fn collect_system_info() -> SystemInfo {
        let mut system = System::new_all();
        system.refresh_all();

        let cpu_model = system.processors().first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let cpu_cores = system.processors().len();
        let total_memory_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        SystemInfo {
            cpu_model,
            cpu_cores,
            total_memory_gb,
            os: system.name().unwrap_or_else(|| "Unknown OS".to_string()),
            architecture: std::env::consts::ARCH.to_string(),
            gpu_info: None, // TODO: Implement GPU detection
        }
    }

    /// Get current memory usage (simplified)
    fn get_memory_usage(&self) -> f64 {
        // TODO: Implement actual memory monitoring
        0.0
    }

    /// Get current CPU usage (simplified)
    fn get_cpu_usage(&self) -> f64 {
        // TODO: Implement actual CPU monitoring
        0.0
    }

    /// Parse devices from command line argument
    fn parse_devices(&self) -> Result<Vec<Device>> {
        let devices_str = &self.config.devices;
        if devices_str == "all" {
            return Ok(vec![Device::Cpu, Device::Metal]); // Add CUDA if available
        }

        let devices: Result<Vec<_>, _> = devices_str
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "auto" => Ok(Device::best_available()),
                "cpu" => Ok(Device::Cpu),
                "metal" => Ok(Device::Metal),
                "cuda" => Ok(Device::Cuda),
                _ => Err(anyhow::anyhow!("Unknown device: {}", s)),
            })
            .collect();

        devices
    }

    /// Parse temperatures from command line argument
    fn parse_temperatures(&self) -> Result<Vec<f32>> {
        self.config.temperatures
            .split(',')
            .map(|s| s.trim().parse::<f32>().map_err(anyhow::Error::from))
            .collect()
    }

    /// Parse token counts from command line argument
    fn parse_token_counts(&self) -> Result<Vec<u32>> {
        self.config.token_counts
            .split(',')
            .map(|s| s.trim().parse::<u32>().map_err(anyhow::Error::from))
            .collect()
    }

    /// Parse batch sizes from command line argument
    fn parse_batch_sizes(&self) -> Result<Vec<usize>> {
        self.config.batch_sizes
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(anyhow::Error::from))
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = BenchmarkArgs::parse();

    if config.monitor {
        // Continuous monitoring mode
        println!("{}", "🔄 Starting continuous monitoring mode".bright_blue().bold());
        println!("Duration: {} minutes", config.monitor_duration);
        
        // TODO: Implement continuous monitoring
        tokio::time::sleep(tokio::time::Duration::from_secs(config.monitor_duration * 60)).await;
        
        println!("✅ Monitoring complete");
    } else {
        // Standard benchmarking mode
        let mut benchmarker = PerformanceBenchmarker::new(config)?;
        benchmarker.run_benchmarks().await?;
    }

    Ok(())
}

// Example usage:
// cargo run --bin performance_benchmark -- --devices cpu,metal --benchmark-type all --output benchmark_results.json