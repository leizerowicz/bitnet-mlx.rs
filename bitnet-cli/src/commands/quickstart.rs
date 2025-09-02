//! Quick Start Command Implementation
//!
//! CLI command for automated onboarding with example models and tutorials.
//! Implements Task 2.1.4: Quick start automation with example models

use clap::Args;
use std::path::PathBuf;

use crate::customer_tools::{
    quickstart::{QuickStartEngine, TutorialConfig, ModelSize},
    Result, OnboardingProgress,
};

/// Quick start automation command
#[derive(Args, Debug)]
pub struct QuickStartCommand {
    /// Directory for examples and tutorials (defaults to ./bitnet-examples)
    #[arg(short, long)]
    pub examples_dir: Option<PathBuf>,
    
    /// Include only small models for quick testing
    #[arg(long)]
    pub small_models_only: bool,
    
    /// Skip large models (>1GB) to save time and space
    #[arg(long)]
    pub skip_large_models: bool,
    
    /// Skip model conversion demonstrations
    #[arg(long)]
    pub skip_conversion: bool,
    
    /// Skip inference demonstrations
    #[arg(long)]
    pub skip_inference: bool,
    
    /// Skip tutorial generation
    #[arg(long)]
    pub skip_tutorial: bool,
    
    /// Custom tutorial configuration: use case (nlp, cv, general)
    #[arg(long)]
    pub use_case: Option<String>,
    
    /// Hardware profile for optimization (auto, cpu, metal, mlx)
    #[arg(long, default_value = "auto")]
    pub hardware: String,
    
    /// Include production deployment guidance
    #[arg(long)]
    pub include_production: bool,
    
    /// Save quick start report to JSON file
    #[arg(long)]
    pub save_report: Option<PathBuf>,
    
    /// Enable verbose output with detailed progress
    #[arg(short, long)]
    pub verbose: bool,
}

impl QuickStartCommand {
    /// Execute the quick start automation command
    pub async fn execute(&self) -> Result<()> {
        // Display welcome message
        self.display_welcome();
        
        // Determine examples directory
        let examples_dir = self.examples_dir.clone().unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_default().join("bitnet-examples")
        });
        
        // Create tutorial configuration
        let tutorial_config = self.create_tutorial_config();
        
        // Setup quick start engine
        let mut engine = QuickStartEngine::new(&examples_dir)
            .with_tutorial_config(tutorial_config);
            
        // Add hardware profile if specified
        if self.hardware != "auto" {
            engine = engine.with_hardware_profile(self.hardware.clone());
        }
        
        // Add progress callback for verbose output
        if self.verbose {
            engine = engine.with_progress_callback(|progress| {
                let percentage = (progress.completed_steps.len() as f32 / progress.total_steps as f32 * 100.0) as u8;
                println!("🚀 Progress: {}% - {}", percentage, progress.current_step);
            });
        }
        
        // Run quick start automation
        let result = engine.run_quickstart().await?;
        
        // Display results
        self.display_quickstart_results(&result);
        
        // Save report if requested
        if let Some(report_path) = &self.save_report {
            self.save_quickstart_report(&result, report_path).await?;
        }
        
        // Display next steps and completion
        self.display_next_steps(&result);
        
        Ok(())
    }
    
    /// Create tutorial configuration based on command options
    fn create_tutorial_config(&self) -> TutorialConfig {
        TutorialConfig {
            include_conversion_guide: !self.skip_conversion,
            include_performance_tips: true,
            include_production_setup: self.include_production,
            hardware_specific: self.hardware != "auto",
            customer_use_case: self.use_case.clone(),
        }
    }
    
    /// Display welcome message and quick start information
    fn display_welcome(&self) {
        println!("🚀 BitNet-Rust Quick Start Automation");
        println!();
        println!("This quick start will:");
        println!("  📥 Download example models for testing");
        
        if !self.skip_conversion {
            println!("  🔄 Demonstrate model conversion to BitNet format");
        }
        
        if !self.skip_inference {
            println!("  ⚡ Run inference demonstrations with performance metrics");
        }
        
        if !self.skip_tutorial {
            println!("  📚 Generate personalized tutorials and guides");
        }
        
        println!("  🎯 Provide next steps for your BitNet journey");
        println!();
        
        let target_time = if self.small_models_only { 5 } else { 15 };
        println!("⏱️  Estimated time: {} minutes", target_time);
        
        if self.hardware != "auto" {
            println!("🔧 Hardware optimization: {}", self.hardware);
        }
        
        println!();
    }
    
    /// Display progress updates
    fn display_progress(&self, progress: &OnboardingProgress) {
        let percentage = progress.progress_percentage();
        let bar_width = 50;
        let filled = (percentage * bar_width as f32 / 100.0) as usize;
        let empty = bar_width - filled;
        
        let bar = format!(
            "[{}{}]",
            "█".repeat(filled),
            "░".repeat(empty)
        );
        
        print!("\r{} {:.0}% - {} ({} min remaining)", 
            bar, percentage, progress.current_step, progress.estimated_remaining_minutes);
        
        if percentage >= 100.0 {
            println!();
        }
        
        // Flush stdout for immediate display
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or_default();
    }
    
    /// Display quick start results and summary
    fn display_quickstart_results(&self, result: &crate::customer_tools::quickstart::QuickStartResult) {
        println!("\n🎉 Quick Start Complete!");
        println!();
        
        let status_icon = if result.success { "✅" } else { "⚠️" };
        println!("{} Status: {}", status_icon, if result.success { "SUCCESS" } else { "COMPLETED WITH ISSUES" });
        println!("⏱️  Total time: {:.1} minutes", result.onboarding_time_minutes);
        println!();
        
        // Example models summary
        if !result.examples_downloaded.is_empty() {
            println!("📥 Downloaded {} example models:", result.examples_downloaded.len());
            
            for example in &result.examples_downloaded {
                let size_icon = match example.size_category {
                    ModelSize::Small => "🟢",
                    ModelSize::Medium => "🟡", 
                    ModelSize::Large => "🔴",
                };
                
                println!("  {} {} ({:.1} MB) - {}",
                    size_icon,
                    example.name,
                    example.size_mb,
                    example.use_case
                );
            }
            println!();
        }
        
        // Conversion results summary
        if !result.conversions_completed.is_empty() && !self.skip_conversion {
            println!("🔄 Model conversion results:");
            
            let total_reduction: f64 = result.conversions_completed
                .iter()
                .map(|c| c.size_reduction_percent)
                .sum::<f64>() / result.conversions_completed.len() as f64;
                
            let total_accuracy: f64 = result.conversions_completed
                .iter()
                .map(|c| c.accuracy_preserved)
                .sum::<f64>() / result.conversions_completed.len() as f64;
            
            println!("  📊 Average size reduction: {:.1}%", total_reduction);
            println!("  🎯 Average accuracy preserved: {:.1}%", total_accuracy);
            
            for conversion in &result.conversions_completed {
                println!("  ✅ {}: {:.1}% smaller, {:.1}% accuracy ({:.1}s)",
                    conversion.model_name,
                    conversion.size_reduction_percent,
                    conversion.accuracy_preserved,
                    conversion.conversion_time_ms as f64 / 1000.0
                );
            }
            println!();
        }
        
        // Performance results summary
        if !result.inference_demos.is_empty() && !self.skip_inference {
            println!("⚡ Performance demonstration results:");
            
            // Find best performing model
            let best_demo = result.inference_demos
                .iter()
                .max_by_key(|d| d.throughput_ops_per_sec)
                .unwrap();
                
            println!("  🏆 Best performance: {} on {}",
                best_demo.model_name,
                best_demo.device_used
            );
            println!("    • Throughput: {} ops/sec",
                best_demo.throughput_ops_per_sec
            );
            println!("    • Latency: {:.1}ms",
                best_demo.inference_time_ms
            );
            println!("    • Memory: {:.1}MB",
                best_demo.memory_usage_mb
            );
            
            // Summary of all models
            for demo in &result.inference_demos {
                println!("  📈 {}: {} ops/sec, {:.1}ms latency",
                    demo.model_name,
                    demo.throughput_ops_per_sec,
                    demo.inference_time_ms
                );
            }
            println!();
        }
        
        // Tutorial generation results
        if result.tutorial_generated && !self.skip_tutorial {
            println!("📚 Generated personalized tutorials and guides:");
            let examples_dir = self.examples_dir.clone().unwrap_or_else(|| {
                std::env::current_dir().unwrap_or_default().join("bitnet-examples")
            });
            
            println!("  📖 Getting Started Guide: {}/tutorials/getting_started.md",
                examples_dir.display()
            );
            
            if self.hardware != "auto" {
                println!("  ⚙️  Hardware Configuration: {}/tutorials/configuration_guide.md",
                    examples_dir.display()
                );
            }
            
            println!("  🚀 Performance Optimization: {}/tutorials/performance_optimization.md",
                examples_dir.display()
            );
            println!();
        }
        
        // Display any issues or warnings
        if !result.success {
            println!("⚠️  Some components completed with warnings. Check individual results above.");
            println!();
        }
    }
    
    /// Display next steps and recommendations
    fn display_next_steps(&self, result: &crate::customer_tools::quickstart::QuickStartResult) {
        println!("🎯 What's Next?");
        println!();
        
        // Display custom next steps from the result
        for (i, step) in result.next_steps.iter().enumerate() {
            println!("{}. {}", i + 1, step);
        }
        
        println!();
        
        // Additional CLI guidance based on what was completed
        println!("🛠️  Essential Commands to Try:");
        
        if !result.conversions_completed.is_empty() {
            println!("  • Convert your own models:");
            println!("    bitnet-cli convert --input your_model.pth --output quantized.bitnet");
        }
        
        println!("  • Validate your system performance:");
        println!("    bitnet-cli validate --comprehensive");
        
        println!("  • Check production readiness:");
        println!("    bitnet-cli ops validate --deployment");
        
        println!("  • Get help with any command:");
        println!("    bitnet-cli --help");
        
        println!();
        
        // Hardware-specific recommendations
        if self.hardware != "auto" && !result.inference_demos.is_empty() {
            if let Some(best_demo) = result.inference_demos.iter().max_by_key(|d| d.throughput_ops_per_sec) {
                println!("🔥 Your {} setup achieved excellent performance!", self.hardware);
                println!("   Best result: {} ops/sec with {} model",
                    best_demo.throughput_ops_per_sec,
                    best_demo.model_name
                );
            }
        }
        
        // Success celebration
        if result.success {
            println!("🎉 Congratulations! You're ready to use BitNet-Rust for high-performance");
            println!("   neural network quantization. Welcome to the future of efficient AI! 🚀");
        }
        
        println!();
        println!("💬 Questions? Join our community:");
        println!("   • GitHub Discussions: https://github.com/bitnet-rust/bitnet-rust/discussions");
        println!("   • Issue Tracker: https://github.com/bitnet-rust/bitnet-rust/issues");
        println!("   • Documentation: https://bitnet-rust.dev");
    }
    
    /// Save detailed quick start report
    async fn save_quickstart_report(
        &self,
        result: &crate::customer_tools::quickstart::QuickStartResult,
        report_path: &PathBuf,
    ) -> Result<()> {
        let report = serde_json::to_string_pretty(result)
            .map_err(|e| crate::customer_tools::CustomerToolsError::QuickStartError(
                format!("Failed to serialize report: {}", e)
            ))?;
            
        tokio::fs::write(report_path, report).await
            .map_err(|e| crate::customer_tools::CustomerToolsError::IoError(e))?;
            
        println!("📄 Quick start report saved to: {}", report_path.display());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::customer_tools::quickstart::{QuickStartResult, ExampleModel, ConversionDemo, InferenceDemo};
    
    #[test]
    fn test_quickstart_command_creation() {
        let cmd = QuickStartCommand {
            examples_dir: None,
            small_models_only: false,
            skip_large_models: false,
            skip_conversion: false,
            skip_inference: false,
            skip_tutorial: false,
            use_case: None,
            hardware: "auto".to_string(),
            include_production: false,
            save_report: None,
            verbose: false,
        };
        
        assert!(cmd.examples_dir.is_none());
        assert!(!cmd.small_models_only);
        assert_eq!(cmd.hardware, "auto");
        assert!(!cmd.include_production);
    }
    
    #[test]
    fn test_tutorial_config_creation() {
        // Default configuration
        let cmd = QuickStartCommand {
            examples_dir: None,
            small_models_only: false,
            skip_large_models: false,
            skip_conversion: false,
            skip_inference: false,
            skip_tutorial: false,
            use_case: None,
            hardware: "auto".to_string(),
            include_production: false,
            save_report: None,
            verbose: false,
        };
        
        let config = cmd.create_tutorial_config();
        assert!(config.include_conversion_guide);
        assert!(config.include_performance_tips);
        assert!(!config.include_production_setup);
        assert!(!config.hardware_specific);
        assert!(config.customer_use_case.is_none());
        
        // Custom configuration
        let custom_cmd = QuickStartCommand {
            skip_conversion: true,
            include_production: true,
            use_case: Some("nlp".to_string()),
            hardware: "mlx".to_string(),
            ..cmd
        };
        
        let custom_config = custom_cmd.create_tutorial_config();
        assert!(!custom_config.include_conversion_guide);
        assert!(custom_config.include_production_setup);
        assert!(custom_config.hardware_specific);
        assert_eq!(custom_config.customer_use_case, Some("nlp".to_string()));
    }
    
    #[test]
    fn test_command_modes() {
        // Small models only
        let small_only = QuickStartCommand {
            examples_dir: Some(PathBuf::from("/tmp/examples")),
            small_models_only: true,
            skip_large_models: true,
            skip_conversion: false,
            skip_inference: false,
            skip_tutorial: false,
            use_case: Some("testing".to_string()),
            hardware: "cpu".to_string(),
            include_production: false,
            save_report: None,
            verbose: true,
        };
        
        assert_eq!(small_only.examples_dir, Some(PathBuf::from("/tmp/examples")));
        assert!(small_only.small_models_only);
        assert!(small_only.skip_large_models);
        assert_eq!(small_only.hardware, "cpu");
        assert!(small_only.verbose);
        
        // Production-focused setup
        let production = QuickStartCommand {
            examples_dir: None,
            small_models_only: false,
            skip_large_models: false,
            skip_conversion: false,
            skip_inference: false,
            skip_tutorial: false,
            use_case: Some("production".to_string()),
            hardware: "mlx".to_string(),
            include_production: true,
            save_report: Some(PathBuf::from("/tmp/quickstart-report.json")),
            verbose: true,
        };
        
        assert!(production.include_production);
        assert_eq!(production.save_report, Some(PathBuf::from("/tmp/quickstart-report.json")));
        assert_eq!(production.use_case, Some("production".to_string()));
    }
    
    #[test]
    fn test_result_display_logic() {
        let cmd = QuickStartCommand {
            examples_dir: None,
            small_models_only: false,
            skip_large_models: false,
            skip_conversion: false,
            skip_inference: false,
            skip_tutorial: false,
            use_case: None,
            hardware: "mlx".to_string(),
            include_production: false,
            save_report: None,
            verbose: false,
        };
        
        // Create mock result
        let result = QuickStartResult {
            success: true,
            examples_downloaded: vec![
                ExampleModel {
                    name: "test-model".to_string(),
                    size_category: ModelSize::Small,
                    original_format: "safetensors".to_string(),
                    size_mb: 45.0,
                    download_url: "test-url".to_string(),
                    local_path: PathBuf::from("/tmp/test.safetensors"),
                    description: "Test model".to_string(),
                    use_case: "Testing".to_string(),
                }
            ],
            conversions_completed: vec![
                ConversionDemo {
                    model_name: "test-model".to_string(),
                    input_format: "safetensors".to_string(),
                    conversion_time_ms: 1500,
                    size_reduction_percent: 89.5,
                    accuracy_preserved: 99.1,
                    bitnet_path: PathBuf::from("/tmp/test.bitnet"),
                    demo_success: true,
                }
            ],
            inference_demos: vec![
                InferenceDemo {
                    model_name: "test-model".to_string(),
                    inference_time_ms: 2.5,
                    throughput_ops_per_sec: 280_000,
                    memory_usage_mb: 35.0,
                    device_used: "MLX".to_string(),
                    sample_output: "Test output".to_string(),
                    demo_success: true,
                }
            ],
            tutorial_generated: true,
            onboarding_time_minutes: 12.5,
            next_steps: vec!["Try converting your own models".to_string()],
        };
        
        // Test that display methods can handle the result
        // (In real tests, we'd capture output and verify content)
        cmd.display_quickstart_results(&result);
        cmd.display_next_steps(&result);
    }
}
