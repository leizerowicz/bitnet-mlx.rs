//! Model Conversion Command Implementation
//!
//! CLI command for converting models from SafeTensors, ONNX, PyTorch to BitNet format.
//! Implements Task 2.1.1: Model format conversion

use clap::Args;
use std::path::PathBuf;

use crate::customer_tools::{
    conversion::{ModelConversionEngine, ConversionConfig, InputFormat, OptimizationLevel},
    Result, OnboardingProgress,
};

/// Model format conversion command
#[derive(Args, Debug, Clone)]
pub struct ConvertCommand {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,
    
    /// Output BitNet model path  
    #[arg(short, long)]
    pub output: PathBuf,
    
    /// Input format (auto-detect if not specified)
    #[arg(short, long)]
    pub format: Option<String>,
    
    /// Target device for optimization (auto, cpu, metal, mlx)
    #[arg(short = 'd', long, default_value = "auto")]
    pub device: String,
    
    /// Optimization level (speed, balanced, memory, accuracy)
    #[arg(short = 'l', long, default_value = "balanced")]
    pub optimization: String,
    
    /// Preserve precision during conversion
    #[arg(long, default_value = "true")]
    pub preserve_precision: bool,
    
    /// Show progress during conversion
    #[arg(short, long)]
    pub verbose: bool,
    
    /// Save conversion report to JSON file
    #[arg(long)]
    pub save_report: Option<PathBuf>,
}

impl ConvertCommand {
    /// Execute the model conversion command
    pub async fn execute(&self) -> Result<()> {
        // Validate input file exists
        if !self.input.exists() {
            return Err(crate::customer_tools::CustomerToolsError::ConversionError(
                format!("Input file does not exist: {}", self.input.display())
            ));
        }
        
        // Auto-detect format if not specified
        let input_format = if let Some(ref format_str) = self.format {
            self.parse_format(format_str)?
        } else {
            self.detect_format_from_extension()?
        };
        
        // Parse optimization level
        let optimization_level = self.parse_optimization_level()?;
        
        // Create conversion configuration
        let config = ConversionConfig {
            input_format,
            quantization_bits: 1.58, // BitNet quantization
            preserve_precision: self.preserve_precision,
            optimization_level,
            target_device: self.device.clone(),
        };
        
        // Setup conversion engine with progress callback
        let mut engine = ModelConversionEngine::new(config);
        
        if self.verbose {
            engine = engine.with_progress_callback(|progress| {
                let percentage = (progress.completed_steps.len() as f32 / progress.total_steps as f32 * 100.0) as u8;
                println!("📊 Progress: {}% - {}", percentage, progress.current_step);
            });
        }
        
        // Display conversion start information
        self.display_conversion_info();
        
        // Perform conversion
        let result = engine.convert_model(&self.input, &self.output).await?;
        
        // Display results
        self.display_conversion_results(&result);
        
        // Save report if requested
        if let Some(report_path) = &self.save_report {
            self.save_conversion_report(&result, report_path).await?;
        }
        
        Ok(())
    }
    
    /// Parse format string to InputFormat enum
    fn parse_format(&self, format_str: &str) -> Result<InputFormat> {
        match format_str.to_lowercase().as_str() {
            "safetensors" => Ok(InputFormat::SafeTensors),
            "onnx" => Ok(InputFormat::Onnx),
            "pytorch" | "pth" | "pt" => Ok(InputFormat::PyTorch),
            _ => Err(crate::customer_tools::CustomerToolsError::ConversionError(
                format!("Unsupported format: {}. Supported formats: safetensors, onnx, pytorch", format_str)
            )),
        }
    }
    
    /// Auto-detect format from file extension
    fn detect_format_from_extension(&self) -> Result<InputFormat> {
        let extension = self.input.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| crate::customer_tools::CustomerToolsError::ConversionError(
                "Unable to determine format from file extension".to_string()
            ))?;
            
        InputFormat::from_extension(extension)
            .ok_or_else(|| crate::customer_tools::CustomerToolsError::ConversionError(
                format!("Unsupported file extension: .{}", extension)
            ))
    }
    
    /// Parse optimization level string
    fn parse_optimization_level(&self) -> Result<OptimizationLevel> {
        match self.optimization.to_lowercase().as_str() {
            "speed" => Ok(OptimizationLevel::Speed),
            "balanced" => Ok(OptimizationLevel::Balanced),
            "memory" => Ok(OptimizationLevel::Memory),
            "accuracy" => Ok(OptimizationLevel::Accuracy),
            _ => Err(crate::customer_tools::CustomerToolsError::ConversionError(
                format!("Invalid optimization level: {}. Valid options: speed, balanced, memory, accuracy", self.optimization)
            )),
        }
    }
    
    /// Display conversion start information
    fn display_conversion_info(&self) {
        println!("🔄 BitNet Model Conversion");
        println!("Input:  {}", self.input.display());
        println!("Output: {}", self.output.display());
        println!("Device: {}", self.device);
        println!("Optimization: {}", self.optimization);
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
        
        print!("\r{} {:.1}% - {} (Est. {} min remaining)", 
            bar, percentage, progress.current_step, progress.estimated_remaining_minutes);
        
        if percentage >= 100.0 {
            println!();
        }
        
        // Flush stdout to ensure immediate display
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or_default();
    }
    
    /// Display conversion results
    fn display_conversion_results(&self, result: &crate::customer_tools::conversion::ConversionResult) {
        println!("\n✅ Conversion Complete!");
        println!();
        println!("📊 Conversion Results:");
        println!("  Status: {}", if result.success { "SUCCESS" } else { "FAILED" });
        println!("  Time: {:.2}s", result.conversion_time_ms as f64 / 1000.0);
        println!("  Input Size: {:.1} MB", result.input_size_mb);
        println!("  Output Size: {:.1} MB", result.output_size_mb);
        println!("  Compression: {:.1}x smaller", result.compression_ratio);
        println!("  Accuracy: {:.1}% preserved", result.accuracy_preserved);
        
        // Display warnings if any
        if !result.warnings.is_empty() {
            println!("\n⚠️  Warnings:");
            for warning in &result.warnings {
                println!("  • {}", warning);
            }
        }
        
        // Display errors if any
        if !result.errors.is_empty() {
            println!("\n❌ Errors:");
            for error in &result.errors {
                println!("  • {}", error);
            }
        }
        
        if result.success {
            println!("\n🚀 Ready to use! Load your quantized model:");
            println!("   bitnet-cli inference --model {}", result.output_path.display());
        }
    }
    
    /// Save detailed conversion report
    async fn save_conversion_report(
        &self, 
        result: &crate::customer_tools::conversion::ConversionResult, 
        report_path: &PathBuf
    ) -> Result<()> {
        let report = serde_json::to_string_pretty(result)
            .map_err(|e| crate::customer_tools::CustomerToolsError::ConversionError(
                format!("Failed to serialize report: {}", e)
            ))?;
            
        tokio::fs::write(report_path, report).await
            .map_err(|e| crate::customer_tools::CustomerToolsError::IoError(e))?;
            
        println!("📄 Conversion report saved to: {}", report_path.display());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_format_parsing() {
        let cmd = ConvertCommand {
            input: PathBuf::from("test.safetensors"),
            output: PathBuf::from("test.bitnet"),
            format: None,
            device: "auto".to_string(),
            optimization: "balanced".to_string(),
            preserve_precision: true,
            verbose: false,
            save_report: None,
        };
        
        assert!(matches!(cmd.parse_format("safetensors").unwrap(), InputFormat::SafeTensors));
        assert!(matches!(cmd.parse_format("onnx").unwrap(), InputFormat::Onnx));
        assert!(matches!(cmd.parse_format("pytorch").unwrap(), InputFormat::PyTorch));
        assert!(cmd.parse_format("unsupported").is_err());
    }
    
    #[test]
    fn test_optimization_level_parsing() {
        let cmd = ConvertCommand {
            input: PathBuf::from("test.safetensors"),
            output: PathBuf::from("test.bitnet"),
            format: None,
            device: "auto".to_string(),
            optimization: "balanced".to_string(),
            preserve_precision: true,
            verbose: false,
            save_report: None,
        };
        
        assert!(matches!(cmd.parse_optimization_level().unwrap(), OptimizationLevel::Balanced));
        
        let speed_cmd = ConvertCommand { optimization: "speed".to_string(), ..cmd };
        assert!(matches!(speed_cmd.parse_optimization_level().unwrap(), OptimizationLevel::Speed));
    }
    
    #[test] 
    fn test_format_auto_detection() {
        let cmd = ConvertCommand {
            input: PathBuf::from("model.safetensors"),
            output: PathBuf::from("model.bitnet"),
            format: None,
            device: "auto".to_string(),
            optimization: "balanced".to_string(),
            preserve_precision: true,
            verbose: false,
            save_report: None,
        };
        
        assert!(matches!(cmd.detect_format_from_extension().unwrap(), InputFormat::SafeTensors));
        
        let onnx_cmd = ConvertCommand { input: PathBuf::from("model.onnx"), ..cmd.clone() };
        assert!(matches!(onnx_cmd.detect_format_from_extension().unwrap(), InputFormat::Onnx));
        
        let pytorch_cmd = ConvertCommand { input: PathBuf::from("model.pth"), ..cmd };
        assert!(matches!(pytorch_cmd.detect_format_from_extension().unwrap(), InputFormat::PyTorch));
    }
    
    #[tokio::test]
    async fn test_command_validation() {
        // Create temporary input file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "dummy content").unwrap();
        
        let cmd = ConvertCommand {
            input: temp_file.path().with_extension("safetensors"),
            output: PathBuf::from("/tmp/test.bitnet"),
            format: Some("safetensors".to_string()),
            device: "cpu".to_string(),
            optimization: "balanced".to_string(),
            preserve_precision: true,
            verbose: false,
            save_report: None,
        };
        
        // Copy temp file to have correct extension
        std::fs::copy(temp_file.path(), &cmd.input).unwrap();
        
        // Test that command parsing works
        assert!(cmd.parse_format("safetensors").is_ok());
        assert!(cmd.parse_optimization_level().is_ok());
        
        // Cleanup
        std::fs::remove_file(&cmd.input).ok();
    }
}
