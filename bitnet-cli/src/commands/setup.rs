//! Setup Wizard Command Implementation
//!
//! CLI command for interactive setup wizard with environment validation.
//! Implements Task 2.1.2: Interactive setup wizard

use clap::Args;
use std::path::PathBuf;
use crate::customer_tools::{
    setup::SetupWizard,
    Result, OnboardingProgress,
};

/// Interactive setup wizard command
#[derive(Args, Debug)]
pub struct SetupCommand {
    /// Run in non-interactive mode (automatic configuration)
    #[arg(long)]
    pub non_interactive: bool,
    
    /// Only check system compatibility without configuration
    #[arg(long)]
    pub check_only: bool,
    
    /// Generate configuration for specific hardware profile
    #[arg(long)]
    pub hardware_profile: Option<String>,
    
    /// Skip dependency validation
    #[arg(long)]
    pub skip_deps: bool,
    
    /// Skip performance testing
    #[arg(long)]
    pub skip_performance: bool,
    
    /// Custom configuration output path
    #[arg(long)]
    pub config_path: Option<PathBuf>,
    
    /// Save setup report to JSON file
    #[arg(long)]
    pub save_report: Option<PathBuf>,
    
    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

impl SetupCommand {
    /// Execute the setup wizard command
    pub async fn execute(&self) -> Result<()> {
        // Display welcome message
        if !self.non_interactive {
            self.display_welcome();
        }
        
        // Create setup wizard
        let interactive_mode = !self.non_interactive && !self.check_only;
        let mut wizard = SetupWizard::new(interactive_mode);
        
        // Add progress callback for verbose output
        if self.verbose {
            wizard = wizard.with_progress_callback(|progress| {
                let percentage = (progress.completed_steps.len() as f32 / progress.total_steps as f32 * 100.0) as u8;
                println!("🔧 Progress: {}% - {}", percentage, progress.current_step);
            });
        }
        
        // Run setup process
        let validation_result = wizard.run_setup().await?;
        
        // Display results based on mode
        if self.check_only {
            self.display_check_results(&validation_result);
        } else {
            self.display_setup_results(&validation_result);
        }
        
        // Save report if requested
        if let Some(report_path) = &self.save_report {
            self.save_setup_report(&validation_result, report_path).await?;
        }
        
        // Provide next steps guidance
        if validation_result.success {
            self.display_next_steps(&validation_result);
        } else {
            self.display_troubleshooting(&validation_result);
        }
        
        Ok(())
    }
    
    /// Display welcome message for interactive mode
    fn display_welcome(&self) {
        println!("🎉 Welcome to BitNet-Rust Setup Wizard!");
        println!();
        println!("This wizard will:");
        println!("  • Detect your hardware capabilities");
        println!("  • Validate system dependencies");
        println!("  • Generate optimal configuration");
        println!("  • Run system validation tests");
        println!("  • Provide performance estimates");
        println!();
        println!("The entire process takes under 5 minutes.");
        println!();
    }
    
    /// Display progress updates
    fn display_progress(&self, progress: &OnboardingProgress) {
        if !self.verbose && self.non_interactive {
            return;
        }
        
        let percentage = progress.progress_percentage();
        let bar_width = 30;
        let filled = (percentage * bar_width as f32 / 100.0) as usize;
        let empty = bar_width - filled;
        
        let bar = format!(
            "[{}{}]",
            "=".repeat(filled),
            " ".repeat(empty)
        );
        
        print!("\r{} {:.0}% - {}", bar, percentage, progress.current_step);
        
        if percentage >= 100.0 {
            println!();
        }
        
        // Flush stdout for immediate display
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or_default();
    }
    
    /// Display check-only results
    fn display_check_results(&self, validation: &crate::customer_tools::setup::SetupValidation) {
        println!("🔍 System Compatibility Check Results\n");
        
        // Overall status
        let status_icon = if validation.success { "✅" } else { "❌" };
        println!("{} Overall Status: {}\n", status_icon, 
            if validation.success { "COMPATIBLE" } else { "ISSUES FOUND" });
        
        // Hardware summary
        println!("🖥️  Hardware Profile:");
        println!("  CPU Cores: {}", validation.hardware_profile.cpu_cores);
        println!("  Memory: {:.1} GB", validation.hardware_profile.memory_gb);
        println!("  GPU: {}", if validation.hardware_profile.has_gpu { "Available" } else { "Not Available" });
        println!("  Optimal Device: {}", validation.hardware_profile.optimal_device);
        println!("  SIMD Support: {:?}", validation.hardware_profile.simd_support);
        
        // Dependency status
        println!("\n📦 Dependencies:");
        println!("  Rust Version: {} {}", 
            validation.dependency_check.rust_version,
            if validation.dependency_check.rust_version_ok { "✅" } else { "❌" }
        );
        println!("  Cargo: {}", if validation.dependency_check.cargo_available { "✅ Available" } else { "❌ Missing" });
        
        if !validation.dependency_check.missing_dependencies.is_empty() {
            println!("  Missing Dependencies: {:?}", validation.dependency_check.missing_dependencies);
        }
        
        // Performance estimates
        println!("\n⚡ Performance Estimate:");
                        println!("  Operations/sec: {}", validation.estimated_performance.operations_per_second);
        println!("  Memory Efficiency: {:.1}%", validation.estimated_performance.memory_efficiency);
        println!("  Max Model Size: {}", validation.estimated_performance.recommended_model_size_limit);
    }
    
    /// Display setup results
    fn display_setup_results(&self, validation: &crate::customer_tools::setup::SetupValidation) {
        println!("\n🎉 Setup Wizard Complete!\n");
        
        let status_icon = if validation.success { "✅" } else { "⚠️" };
        println!("{} Setup Status: {}\n", status_icon,
            if validation.success { "SUCCESS" } else { "COMPLETED WITH WARNINGS" });
        
        // Hardware optimization summary
        println!("🚀 Hardware Optimization:");
        println!("  Detected: {} with {} cores, {:.1}GB RAM",
            validation.hardware_profile.optimal_device.to_uppercase(),
            validation.hardware_profile.cpu_cores,
            validation.hardware_profile.memory_gb
        );
        
        if validation.hardware_profile.has_mlx {
            println!("  🏎️  MLX acceleration available - Excellent performance expected!");
        } else if validation.hardware_profile.has_metal {
            println!("  🔥 Metal GPU acceleration available - Good performance expected!");
        } else {
            println!("  💻 CPU optimization enabled - Solid performance expected!");
        }
        
        // Configuration status
        if validation.configuration_generated {
            println!("\n⚙️  Configuration:");
            println!("  ✅ Optimal settings generated and saved");
            println!("  📍 Location: ~/.bitnet/bitnet-config.toml");
            println!("  🔧 Device: {}", validation.hardware_profile.recommended_config.device_type);
            println!("  🧵 Threads: {}", validation.hardware_profile.recommended_config.thread_count);
            println!("  💾 Memory Pool: {} MB", validation.hardware_profile.recommended_config.memory_pool_size_mb);
        }
        
        // Performance summary
        if validation.quick_test_passed {
            println!("\n📊 Performance Validation:");
            println!("  ✅ System tests passed");
            println!("  ⚡ Expected: {} operations/second", validation.estimated_performance.operations_per_second);
            println!("  📈 Memory efficiency: {:.1}%", validation.estimated_performance.memory_efficiency);
            println!("  🎯 Recommended max model size: {}", validation.estimated_performance.recommended_model_size_limit);
        }
        
        // Display warnings if any
        if !validation.warnings.is_empty() {
            println!("\n⚠️  Warnings:");
            for warning in &validation.warnings {
                println!("  • {}", warning);
            }
        }
        
        // Display recommendations
        if !validation.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for recommendation in &validation.recommendations {
                println!("  • {}", recommendation);
            }
        }
    }
    
    /// Display next steps for successful setup
    fn display_next_steps(&self, validation: &crate::customer_tools::setup::SetupValidation) {
        println!("\n🎯 Next Steps:\n");
        
        println!("1. 🚀 **Quick Start**: Run the quick start wizard");
        println!("   bitnet-cli quickstart");
        
        println!("\n2. 🔄 **Convert Models**: Try converting a model");
        println!("   bitnet-cli convert --input model.pth --output quantized.bitnet");
        
        println!("\n3. 🧪 **Validate Performance**: Run comprehensive benchmarks");
        println!("   bitnet-cli validate --system-health");
        
        println!("\n4. 📚 **Learn More**: Explore CLI capabilities");
        println!("   bitnet-cli --help");
        
        if validation.hardware_profile.has_gpu {
            println!("\n5. ⚡ **GPU Optimization**: Your GPU is ready for acceleration!");
            println!("   Models will automatically use {} for optimal performance", 
                validation.hardware_profile.optimal_device);
        }
        
        println!("\n🎉 You're ready to start using BitNet-Rust!");
    }
    
    /// Display troubleshooting guidance
    fn display_troubleshooting(&self, validation: &crate::customer_tools::setup::SetupValidation) {
        println!("\n⚠️  Setup Issues Detected\n");
        
        if !validation.dependency_check.rust_version_ok {
            println!("🔧 Rust Version Issue:");
            println!("  Current: {}", validation.dependency_check.rust_version);
            println!("  Required: Rust 1.75+");
            println!("  Fix: rustup update");
            println!();
        }
        
        if !validation.dependency_check.cargo_available {
            println!("🔧 Cargo Missing:");
            println!("  Cargo is required for BitNet-Rust");
            println!("  Fix: Install Rust with cargo from https://rustup.rs/");
            println!();
        }
        
        if !validation.dependency_check.missing_dependencies.is_empty() {
            println!("🔧 Missing Dependencies:");
            for dep in &validation.dependency_check.missing_dependencies {
                println!("  • {}", dep);
            }
            println!("  Fix: Install system dependencies for your platform");
            println!();
        }
        
        if !validation.quick_test_passed {
            println!("🔧 System Validation Failed:");
            println!("  BitNet functionality test failed");
            println!("  This may indicate hardware or dependency issues");
            println!("  Try: bitnet-cli validate --system-health for detailed diagnostics");
            println!();
        }
        
        println!("💬 Need Help?");
        println!("  • Run setup again: bitnet-cli setup");
        println!("  • Get detailed diagnostics: bitnet-cli validate --comprehensive");
        println!("  • Check documentation: https://github.com/bitnet-rust/bitnet-rust");
        println!("  • Join community discussions for support");
    }
    
    /// Save detailed setup report
    async fn save_setup_report(
        &self,
        validation: &crate::customer_tools::setup::SetupValidation,
        report_path: &PathBuf,
    ) -> Result<()> {
        let report = serde_json::to_string_pretty(validation)
            .map_err(|e| crate::customer_tools::CustomerToolsError::SetupError(
                format!("Failed to serialize report: {}", e)
            ))?;
            
        tokio::fs::write(report_path, report).await
            .map_err(|e| crate::customer_tools::CustomerToolsError::IoError(e))?;
            
        println!("📄 Setup report saved to: {}", report_path.display());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::customer_tools::setup::{SetupValidation, HardwareProfile, DependencyStatus, PerformanceEstimate, BitNetConfig};
    use std::collections::HashMap;
    
    #[test]
    fn test_setup_command_creation() {
        let cmd = SetupCommand {
            non_interactive: false,
            check_only: false,
            hardware_profile: None,
            skip_deps: false,
            skip_performance: false,
            config_path: None,
            save_report: None,
            verbose: false,
        };
        
        assert!(!cmd.non_interactive);
        assert!(!cmd.check_only);
        assert!(cmd.hardware_profile.is_none());
    }
    
    #[test]
    fn test_setup_modes() {
        // Interactive mode
        let interactive = SetupCommand {
            non_interactive: false,
            check_only: false,
            hardware_profile: None,
            skip_deps: false,
            skip_performance: false,
            config_path: None,
            save_report: None,
            verbose: false,
        };
        assert!(!interactive.non_interactive);
        
        // Non-interactive mode
        let non_interactive = SetupCommand {
            non_interactive: true,
            check_only: false,
            hardware_profile: Some("cpu".to_string()),
            skip_deps: false,
            skip_performance: false,
            config_path: None,
            save_report: None,
            verbose: false,
        };
        assert!(non_interactive.non_interactive);
        assert_eq!(non_interactive.hardware_profile, Some("cpu".to_string()));
        
        // Check-only mode
        let check_only = SetupCommand {
            non_interactive: false,
            check_only: true,
            hardware_profile: None,
            skip_deps: false,
            skip_performance: false,
            config_path: None,
            save_report: None,
            verbose: false,
        };
        assert!(check_only.check_only);
    }
    
    #[test]
    fn test_result_display_logic() {
        let cmd = SetupCommand {
            non_interactive: false,
            check_only: false,
            hardware_profile: None,
            skip_deps: false,
            skip_performance: false,
            config_path: None,
            save_report: None,
            verbose: true,
        };
        
        // Create mock validation result
        let validation = SetupValidation {
            success: true,
            hardware_profile: HardwareProfile {
                cpu_cores: 8,
                memory_gb: 16.0,
                has_gpu: true,
                has_metal: true,
                has_mlx: true,
                simd_support: vec!["NEON".to_string()],
                optimal_device: "mlx".to_string(),
                recommended_config: BitNetConfig::default(),
            },
            dependency_check: DependencyStatus {
                rust_version: "rustc 1.75.0".to_string(),
                rust_version_ok: true,
                cargo_available: true,
                system_libraries: HashMap::new(),
                missing_dependencies: Vec::new(),
            },
            configuration_generated: true,
            quick_test_passed: true,
            estimated_performance: PerformanceEstimate {
                operations_per_second: 300_000,
                memory_efficiency: 92.0,
                recommended_model_size_limit: "Up to 7B parameters".to_string(),
                expected_quantization_speedup: 6.0,
            },
            warnings: Vec::new(),
            recommendations: vec!["Great setup!".to_string()],
        };
        
        // Test that display methods can handle the validation result
        // (In a real test, we'd capture output and verify content)
        cmd.display_setup_results(&validation);
        cmd.display_check_results(&validation);
        cmd.display_next_steps(&validation);
        
        // Test troubleshooting display with failed validation
        let failed_validation = SetupValidation {
            success: false,
            dependency_check: DependencyStatus {
                rust_version: "rustc 1.70.0".to_string(),
                rust_version_ok: false,
                cargo_available: false,
                system_libraries: HashMap::new(),
                missing_dependencies: vec!["libssl".to_string()],
            },
            quick_test_passed: false,
            ..validation
        };
        
        cmd.display_troubleshooting(&failed_validation);
    }
}
