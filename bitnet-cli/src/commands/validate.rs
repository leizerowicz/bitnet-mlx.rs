//! System Validation Command Implementation
//!
//! CLI command for system health validation and performance benchmarking.
//! Implements Task 2.1.3: System health validation and performance benchmarking

use clap::Args;
use std::path::PathBuf;

use crate::customer_tools::{
    validation::{SystemValidator, SystemValidationReport, HealthStatus},
    Result, OnboardingProgress,
};

/// System validation and benchmarking command
#[derive(Args, Debug)]
pub struct ValidateCommand {
    /// Run comprehensive validation (takes longer, more detailed)
    #[arg(long)]
    pub comprehensive: bool,
    
    /// Focus on system health checks only
    #[arg(long)]
    pub system_health: bool,
    
    /// Run performance benchmarks only
    #[arg(long)]
    pub benchmark_only: bool,
    
    /// Skip memory validation tests
    #[arg(long)]
    pub skip_memory: bool,
    
    /// Skip hardware compatibility checks
    #[arg(long)]
    pub skip_hardware: bool,
    
    /// Custom benchmark duration in seconds
    #[arg(long)]
    pub duration: Option<u64>,
    
    /// Save validation report to JSON file
    #[arg(long)]
    pub save_report: Option<PathBuf>,
    
    /// Output format (table, json, yaml)
    #[arg(short, long, default_value = "table")]
    pub format: String,
    
    /// Enable verbose output with detailed progress
    #[arg(short, long)]
    pub verbose: bool,
}

impl ValidateCommand {
    /// Execute the system validation command
    pub async fn execute(&self) -> Result<()> {
        // Display validation start information
        self.display_validation_start();
        
        // Create system validator based on mode
        let validator = if self.comprehensive {
            SystemValidator::comprehensive()
        } else {
            SystemValidator::new()
        };
        
        // Add progress callback for verbose output
        let validator = if self.verbose {
            validator.with_progress_callback(|progress| {
                let percentage = (progress.completed_steps.len() as f32 / progress.total_steps as f32 * 100.0) as u8;
                println!("🔍 Progress: {}% - {}", percentage, progress.current_step);
            })
        } else {
            validator
        };
        
        // Run system validation
        let report = validator.validate_system().await?;
        
        // Display results based on requested format and focus
        match self.format.as_str() {
            "json" => self.display_json_report(&report)?,
            "yaml" => self.display_yaml_report(&report)?,
            _ => self.display_table_report(&report),
        }
        
        // Save report if requested
        if let Some(report_path) = &self.save_report {
            self.save_validation_report(&report, report_path).await?;
        }
        
        // Provide recommendations and next steps
        self.display_recommendations(&report);
        
        Ok(())
    }
    
    /// Display validation start information
    fn display_validation_start(&self) {
        let mode = if self.comprehensive {
            "Comprehensive"
        } else if self.system_health {
            "System Health"
        } else if self.benchmark_only {
            "Performance Benchmark"
        } else {
            "Quick"
        };
        
        println!("🔍 BitNet System Validation - {} Mode", mode);
        
        if self.comprehensive {
            println!("⏱️  This comprehensive validation will take 30-60 seconds...");
        } else {
            println!("⏱️  Quick validation will take 5-10 seconds...");
        }
        
        println!();
    }
    
    /// Display progress updates
    fn display_progress(&self, progress: &OnboardingProgress) {
        let percentage = progress.progress_percentage();
        let bar_width = 40;
        let filled = (percentage * bar_width as f32 / 100.0) as usize;
        let empty = bar_width - filled;
        
        let bar = format!(
            "[{}{}]",
            "=".repeat(filled),
            " ".repeat(empty)
        );
        
        print!("\r{} {:.1}% - {}", bar, percentage, progress.current_step);
        
        if percentage >= 100.0 {
            println!();
        }
        
        // Flush stdout for immediate display
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or_default();
    }
    
    /// Display validation report in table format
    fn display_table_report(&self, report: &SystemValidationReport) {
        println!("📊 System Validation Report");
        println!("Generated: {}", report.timestamp);
        println!();
        
        // Overall health status
        let health_icon = match report.overall_health {
            HealthStatus::Excellent => "🟢",
            HealthStatus::Good => "🟡",
            HealthStatus::Fair => "🟠",
            HealthStatus::Poor => "🔴",
            HealthStatus::Critical => "⚠️",
        };
        
        println!("{} Overall System Health: {:?}", health_icon, report.overall_health);
        println!();
        
        // Memory validation section
        if !self.benchmark_only {
            println!("💾 Memory Validation:");
            println!("  Available Memory: {:.1} GB", report.memory_validation.available_memory_gb);
            println!("  Memory Pool Test: {}", if report.memory_validation.bitnet_memory_pool_test { "✅ PASS" } else { "❌ FAIL" });
            println!("  Large Allocation: {}", if report.memory_validation.large_allocation_test { "✅ PASS" } else { "❌ FAIL" });
            println!("  Memory Leak Test: {}", if report.memory_validation.memory_leak_test { "✅ PASS" } else { "❌ FAIL" });
            println!("  Fragmentation Resistance: {:.1}%", report.memory_validation.fragmentation_resistance);
            println!("  Optimal Pool Size: {} MB", report.memory_validation.optimal_pool_size_mb);
            println!();
        }
        
        // Performance benchmark section
        if !self.system_health {
            println!("⚡ Performance Benchmark:");
                            println!("  Quantization Ops/sec: {}", report.performance_benchmark.quantization_ops_per_second);
            println!("  Inference Latency: {:.1}ms", report.performance_benchmark.inference_latency_ms);
            println!("  Memory Throughput: {:.1} GB/s", report.performance_benchmark.memory_throughput_gbps);
            println!("  CPU Utilization: {:.1}%", report.performance_benchmark.cpu_utilization_percent);
            println!("  GPU Utilization: {:.1}%", report.performance_benchmark.gpu_utilization_percent);
            println!("  Optimization Score: {:.1}/100", report.performance_benchmark.device_optimization_score);
            
            // Comparative performance
            println!("\n📈 Comparative Performance:");
            println!("  vs Baseline CPU: {:.1}x faster", report.performance_benchmark.comparative_performance.vs_baseline_cpu);
            println!("  vs Reference GPU: {:.1}x", report.performance_benchmark.comparative_performance.vs_reference_gpu);
            println!("  Percentile Ranking: {:.0}th percentile", report.performance_benchmark.comparative_performance.percentile_ranking);
            println!();
        }
        
        // Hardware compatibility section
        if !self.benchmark_only {
            println!("🖥️  Hardware Compatibility:");
            println!("  CPU Architecture: {}", report.hardware_compatibility.cpu_architecture);
            println!("  SIMD Support: {:?}", report.hardware_compatibility.simd_support);
            println!("  GPU Available: {}", if report.hardware_compatibility.gpu_compatibility.has_gpu { "✅ Yes" } else { "❌ No" });
            
            if report.hardware_compatibility.gpu_compatibility.has_gpu {
                println!("  GPU Type: {}", report.hardware_compatibility.gpu_compatibility.gpu_type);
                println!("  Metal Support: {}", if report.hardware_compatibility.gpu_compatibility.metal_support { "✅ Yes" } else { "❌ No" });
                println!("  MLX Support: {}", if report.hardware_compatibility.gpu_compatibility.mlx_support { "✅ Yes" } else { "❌ No" });
            }
            
            println!("  Memory Type: {}", report.hardware_compatibility.memory_type);
            println!("  OS Compatible: {}", if report.hardware_compatibility.os_compatibility { "✅ Yes" } else { "❌ No" });
            println!("  BitNet Optimizations: {:?}", report.hardware_compatibility.bitnet_optimization_support);
            println!();
        }
        
        // Dependencies section
        if !self.benchmark_only {
            println!("📦 Dependencies:");
            println!("  Rust Toolchain: {}", report.dependency_status.rust_toolchain);
            
            if !report.dependency_status.missing_critical.is_empty() {
                println!("  ❌ Missing Critical: {:?}", report.dependency_status.missing_critical);
            }
            
            if !report.dependency_status.missing_optional.is_empty() {
                println!("  ⚠️  Missing Optional: {:?}", report.dependency_status.missing_optional);
            }
            
            if report.dependency_status.missing_critical.is_empty() && report.dependency_status.missing_optional.is_empty() {
                println!("  ✅ All dependencies satisfied");
            }
            println!();
        }
        
        // Warnings section
        if !report.warnings.is_empty() {
            println!("⚠️  Warnings:");
            for warning in &report.warnings {
                println!("  • {}", warning);
            }
            println!();
        }
    }
    
    /// Display validation report in JSON format
    fn display_json_report(&self, report: &SystemValidationReport) -> Result<()> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| crate::customer_tools::CustomerToolsError::ValidationError(
                format!("Failed to serialize report to JSON: {}", e)
            ))?;
        println!("{}", json);
        Ok(())
    }
    
    /// Display validation report in YAML format
    fn display_yaml_report(&self, report: &SystemValidationReport) -> Result<()> {
        let yaml = serde_yaml::to_string(report)
            .map_err(|e| crate::customer_tools::CustomerToolsError::ValidationError(
                format!("Failed to serialize report to YAML: {}", e)
            ))?;
        println!("{}", yaml);
        Ok(())
    }
    
    /// Display recommendations and next steps
    fn display_recommendations(&self, report: &SystemValidationReport) {
        if !report.recommendations.is_empty() {
            println!("💡 Recommendations:");
            for recommendation in &report.recommendations {
                println!("  • {}", recommendation);
            }
            println!();
        }
        
        // Performance-based recommendations
        match report.overall_health {
            HealthStatus::Excellent => {
                println!("🎉 Excellent! Your system is optimally configured for BitNet.");
                println!("🚀 Ready for production deployment with high-performance quantization.");
            },
            HealthStatus::Good => {
                println!("✅ Good setup! Your system will perform well with BitNet.");
                println!("💡 Consider the recommendations above for optimal performance.");
            },
            HealthStatus::Fair => {
                println!("⚠️  Fair setup. BitNet will work but performance may be limited.");
                println!("🔧 Address the warnings above to improve performance.");
            },
            HealthStatus::Poor => {
                println!("🔴 Poor system health detected. BitNet may have performance issues.");
                println!("🔧 Please address critical issues before production use.");
            },
            HealthStatus::Critical => {
                println!("⚠️  Critical issues detected! BitNet may not function properly.");
                println!("🔧 Please resolve all critical issues before using BitNet.");
            },
        }
        
        // Next steps based on validation results
        println!("\n🎯 Next Steps:");
        
        if matches!(report.overall_health, HealthStatus::Excellent | HealthStatus::Good) {
            println!("  1. Run quick start: bitnet-cli quickstart");
            println!("  2. Try model conversion: bitnet-cli convert --help");
            println!("  3. Explore production tools: bitnet-cli ops --help");
        } else {
            println!("  1. Address system issues identified above");
            println!("  2. Re-run validation: bitnet-cli validate --comprehensive");
            println!("  3. Check documentation for troubleshooting guidance");
        }
    }
    
    /// Save detailed validation report
    async fn save_validation_report(
        &self,
        report: &SystemValidationReport,
        report_path: &PathBuf,
    ) -> Result<()> {
        let content = match report_path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => {
                serde_yaml::to_string(report)
                    .map_err(|e| crate::customer_tools::CustomerToolsError::ValidationError(
                        format!("Failed to serialize report to YAML: {}", e)
                    ))?
            },
            _ => {
                // Default to JSON
                serde_json::to_string_pretty(report)
                    .map_err(|e| crate::customer_tools::CustomerToolsError::ValidationError(
                        format!("Failed to serialize report to JSON: {}", e)
                    ))?
            }
        };
        
        tokio::fs::write(report_path, content).await
            .map_err(|e| crate::customer_tools::CustomerToolsError::IoError(e))?;
            
        println!("📄 Validation report saved to: {}", report_path.display());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::customer_tools::validation::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_validate_command_creation() {
        let cmd = ValidateCommand {
            comprehensive: false,
            system_health: false,
            benchmark_only: false,
            skip_memory: false,
            skip_hardware: false,
            duration: None,
            save_report: None,
            format: "table".to_string(),
            verbose: false,
        };
        
        assert!(!cmd.comprehensive);
        assert!(!cmd.system_health);
        assert!(!cmd.benchmark_only);
        assert_eq!(cmd.format, "table");
    }
    
    #[test]
    fn test_validation_modes() {
        // Comprehensive mode
        let comprehensive = ValidateCommand {
            comprehensive: true,
            system_health: false,
            benchmark_only: false,
            skip_memory: false,
            skip_hardware: false,
            duration: Some(30),
            save_report: None,
            format: "json".to_string(),
            verbose: true,
        };
        assert!(comprehensive.comprehensive);
        assert_eq!(comprehensive.duration, Some(30));
        assert_eq!(comprehensive.format, "json");
        
        // System health mode
        let health_only = ValidateCommand {
            comprehensive: false,
            system_health: true,
            benchmark_only: false,
            skip_memory: false,
            skip_hardware: false,
            duration: None,
            save_report: None,
            format: "yaml".to_string(),
            verbose: false,
        };
        assert!(health_only.system_health);
        assert_eq!(health_only.format, "yaml");
        
        // Benchmark only mode
        let benchmark_only = ValidateCommand {
            comprehensive: false,
            system_health: false,
            benchmark_only: true,
            skip_memory: true,
            skip_hardware: true,
            duration: None,
            save_report: Some(PathBuf::from("/tmp/report.json")),
            format: "table".to_string(),
            verbose: false,
        };
        assert!(benchmark_only.benchmark_only);
        assert!(benchmark_only.skip_memory);
        assert!(benchmark_only.skip_hardware);
    }
    
    #[test]
    fn test_report_format_handling() {
        let cmd = ValidateCommand {
            comprehensive: false,
            system_health: false,
            benchmark_only: false,
            skip_memory: false,
            skip_hardware: false,
            duration: None,
            save_report: None,
            format: "table".to_string(),
            verbose: false,
        };
        
        // Create mock report
        let report = create_mock_validation_report();
        
        // Test that display methods can handle the report
        // (In real tests, we'd capture output and verify content)
        cmd.display_table_report(&report);
        cmd.display_recommendations(&report);
        
        // Test JSON serialization
        let json_result = cmd.display_json_report(&report);
        assert!(json_result.is_ok());
        
        // Test YAML serialization
        let yaml_result = cmd.display_yaml_report(&report);
        assert!(yaml_result.is_ok());
    }
    
    fn create_mock_validation_report() -> SystemValidationReport {
        SystemValidationReport {
            overall_health: HealthStatus::Good,
            memory_validation: MemoryValidationResult {
                available_memory_gb: 16.0,
                bitnet_memory_pool_test: true,
                large_allocation_test: true,
                memory_leak_test: true,
                fragmentation_resistance: 92.5,
                optimal_pool_size_mb: 1024,
            },
            performance_benchmark: PerformanceBenchmarkResult {
                quantization_ops_per_second: 280_000,
                inference_latency_ms: 2.5,
                memory_throughput_gbps: 25.6,
                cpu_utilization_percent: 75.0,
                gpu_utilization_percent: 65.0,
                device_optimization_score: 85.0,
                comparative_performance: ComparativePerformance {
                    vs_baseline_cpu: 5.6,
                    vs_reference_gpu: 1.4,
                    percentile_ranking: 85.0,
                },
            },
            hardware_compatibility: HardwareCompatibilityResult {
                cpu_architecture: "aarch64".to_string(),
                simd_support: vec!["NEON".to_string()],
                gpu_compatibility: GpuCompatibilityStatus {
                    has_gpu: true,
                    gpu_type: "Apple Silicon".to_string(),
                    metal_support: true,
                    mlx_support: true,
                    compute_capability: Some("Metal 3.0".to_string()),
                },
                memory_type: "LPDDR5".to_string(),
                os_compatibility: true,
                bitnet_optimization_support: vec!["MLX".to_string(), "Metal".to_string(), "NEON SIMD".to_string()],
            },
            dependency_status: DependencyValidationResult {
                rust_toolchain: "rustc 1.75.0 (stable)".to_string(),
                required_libraries: HashMap::new(),
                optional_optimizations: HashMap::new(),
                missing_critical: Vec::new(),
                missing_optional: Vec::new(),
            },
            recommendations: vec![
                "Excellent MLX support detected - use MLX device for optimal performance".to_string(),
                "System is well-configured for production BitNet deployment".to_string(),
            ],
            warnings: Vec::new(),
            timestamp: "2025-09-02 12:00:00 UTC".to_string(),
        }
    }
}
