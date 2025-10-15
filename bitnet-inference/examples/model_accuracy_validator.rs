//! CLI tool for running model accuracy validation
//!
//! This tool demonstrates the model accuracy validation capabilities
//! implemented in Task 5.1.1, providing a command-line interface for
//! testing BitNet model inference accuracy.

use bitnet_inference::Result;
use std::env;
use std::path::Path;
use tokio;

// Include the validation module (this would normally be a proper import)
#[path = "../tests/model_accuracy_validation.rs"]
mod model_accuracy_validation;

use model_accuracy_validation::{ModelAccuracyValidator, AccuracyValidationConfig, TestVector};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage(&args[0]);
        return Err(bitnet_inference::InferenceError::config("Missing model path argument"));
    }

    let model_path = &args[1];
    
    // Parse optional flags
    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());
    let skip_edge_cases = args.contains(&"--skip-edge-cases".to_string());
    let skip_long_contexts = args.contains(&"--skip-long-contexts".to_string());
    let skip_reference = args.contains(&"--skip-reference".to_string());
    
    // Parse tolerance (default: 1e-4)
    let tolerance = if let Some(pos) = args.iter().position(|x| x == "--tolerance") {
        if pos + 1 < args.len() {
            args[pos + 1].parse::<f32>()
                .map_err(|_| bitnet_inference::InferenceError::config("Invalid tolerance value"))?
        } else {
            1e-4
        }
    } else {
        1e-4
    };

    // Parse error rate (default: 0.05)
    let error_rate = if let Some(pos) = args.iter().position(|x| x == "--error-rate") {
        if pos + 1 < args.len() {
            args[pos + 1].parse::<f32>()
                .map_err(|_| bitnet_inference::InferenceError::config("Invalid error rate value"))?
        } else {
            0.05
        }
    } else {
        0.05
    };

    // Show help if requested
    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help(&args[0]);
        return Ok(());
    }

    // Validate model path exists
    if !Path::new(model_path).exists() {
        return Err(bitnet_inference::InferenceError::model_load(
            format!("Model file not found: {}", model_path)
        ));
    }

    println!("🚀 BitNet Model Accuracy Validation Tool");
    println!("📁 Model: {}", model_path);
    println!("🎯 Tolerance: {:.6}", tolerance);
    println!("⚠️ Max Error Rate: {:.1}%", error_rate * 100.0);
    println!();

    // Create validation configuration
    let config = AccuracyValidationConfig {
        numerical_tolerance: tolerance,
        max_error_rate: error_rate,
        test_edge_cases: !skip_edge_cases,
        test_long_contexts: !skip_long_contexts,
        validate_reference_outputs: !skip_reference,
    };

    if verbose {
        println!("🔧 Validation Configuration:");
        println!("   • Reference Output Validation: {}", if !skip_reference { "✅" } else { "❌" });
        println!("   • Edge Case Testing: {}", if !skip_edge_cases { "✅" } else { "❌" });
        println!("   • Long Context Testing: {}", if !skip_long_contexts { "✅" } else { "❌" });
        println!("   • Numerical Tolerance: {:.6}", tolerance);
        println!("   • Max Error Rate: {:.1}%", error_rate * 100.0);
        println!();
    }

    // Create validator and add custom test vectors
    let mut validator = ModelAccuracyValidator::with_config(config);

    // Add additional test vectors for demonstration
    validator.add_test_vector(TestVector {
        input_tokens: vec![1, 1576, 366, 29889], // "This is."
        expected_logits: vec![0.3456, -0.7890, 0.1234],
        expected_tokens: vec![1, 1576, 366, 29889, 306], // "This is. I"
        description: "Statement completion test".to_string(),
    });

    validator.add_test_vector(TestVector {
        input_tokens: vec![1, 450, 4799, 310], // "The capital of"
        expected_logits: vec![-0.2345, 0.6789, -0.4567],
        expected_tokens: vec![1, 450, 4799, 310, 3444], // "The capital of France"
        description: "Knowledge completion test".to_string(),
    });

    // Run validation
    println!("🏁 Starting comprehensive model accuracy validation...");
    let start_time = std::time::Instant::now();
    
    let result = validator.validate_model_accuracy(model_path).await?;
    
    let duration = start_time.elapsed();
    println!();

    // Display results
    println!("📊 Validation Results Summary");
    println!("=============================");
    println!("🎯 Overall Success Rate: {:.1}%", result.success_rate * 100.0);
    println!("⏱️ Total Validation Time: {:.2}s", duration.as_secs_f64());
    println!("🧪 Tests Completed: {}", result.test_results.len());
    println!();

    // Individual test results
    println!("📋 Individual Test Results:");
    for (test_name, test_result) in &result.test_results {
        let status = if test_result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("   {} {}", status, test_name);
        
        if verbose {
            let metrics = &test_result.accuracy_metrics;
            println!("      MAE: {:.6}, RMSE: {:.6}, Max Error: {:.6}, Correlation: {:.6}",
                    metrics.mae, metrics.rmse, metrics.max_error, metrics.correlation);
        }
        
        if let Some(error) = &test_result.error_message {
            println!("      Error: {}", error);
        }
    }
    println!();

    // Precision statistics
    println!("🔢 Precision Statistics:");
    println!("   • Average Precision: {:.6}", result.precision_stats.avg_precision);
    println!("   • Quantization Error: {:.6}", result.precision_stats.quantization_error);
    println!("   • Ternary Accuracy: {:.6}", result.precision_stats.ternary_accuracy);
    println!();

    // Performance metrics
    println!("⚡ Performance Metrics:");
    println!("   • Average Latency: {:.1}ms", result.performance_metrics.avg_latency_ms);
    println!("   • Memory Usage: {:.1}MB", result.performance_metrics.memory_usage_bytes as f64 / 1_000_000.0);
    println!("   • Throughput: {:.1} tokens/sec", result.performance_metrics.throughput_tokens_per_sec);
    println!();

    // Final assessment
    if result.success_rate >= 0.95 {
        println!("🎉 Excellent! Model accuracy validation PASSED with {:.1}% success rate", result.success_rate * 100.0);
    } else if result.success_rate >= 0.85 {
        println!("⚠️ Warning: Model accuracy validation passed with {:.1}% success rate (below 95%)", result.success_rate * 100.0);
    } else {
        println!("❌ Model accuracy validation FAILED with {:.1}% success rate", result.success_rate * 100.0);
        return Err(bitnet_inference::InferenceError::config(
            format!("Validation failed with {:.1}% success rate", result.success_rate * 100.0)
        ));
    }

    println!("✨ Model accuracy validation completed successfully!");
    
    Ok(())
}

fn print_usage(program_name: &str) {
    println!("Usage: {} <MODEL_PATH> [OPTIONS]", program_name);
    println!();
    println!("Arguments:");
    println!("  <MODEL_PATH>    Path to the BitNet model file (GGUF format)");
    println!();
    println!("Options:");
    println!("  --tolerance <TOLERANCE>     Numerical tolerance for validation (default: 0.0001)");
    println!("  --error-rate <ERROR_RATE>   Maximum acceptable error rate (default: 0.05)");
    println!("  --skip-edge-cases           Skip edge case testing");
    println!("  --skip-long-contexts        Skip long context testing");
    println!("  --skip-reference            Skip reference output validation");
    println!("  -v, --verbose               Enable verbose output");
    println!("  -h, --help                  Show this help message");
}

fn print_help(program_name: &str) {
    println!("🚀 BitNet Model Accuracy Validation Tool");
    println!("=========================================");
    println!();
    println!("This tool provides comprehensive validation of BitNet model inference accuracy,");
    println!("testing reference outputs, numerical precision, edge cases, and long context handling.");
    println!();
    print_usage(program_name);
    println!();
    println!("Examples:");
    println!("  {} model.gguf", program_name);
    println!("  {} model.gguf --tolerance 0.001 --verbose", program_name);
    println!("  {} model.gguf --skip-edge-cases --error-rate 0.1", program_name);
    println!();
    println!("Test Types:");
    println!("  • Reference Output Validation - Compare with expected BitNet outputs");
    println!("  • Numerical Precision Testing - Validate quantization accuracy");
    println!("  • Edge Case Testing - Test boundary conditions and special inputs");
    println!("  • Long Context Testing - Validate handling of long token sequences");
    println!();
    println!("The tool will output detailed validation results including accuracy metrics,");
    println!("performance characteristics, and a final assessment of model quality.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_config() {
        let config = AccuracyValidationConfig {
            numerical_tolerance: 1e-3,
            max_error_rate: 0.1,
            test_edge_cases: true,
            test_long_contexts: false,
            validate_reference_outputs: true,
        };

        let _validator = ModelAccuracyValidator::with_config(config.clone());
        
        // Verify configuration values are correct
        assert_eq!(config.numerical_tolerance, 1e-3);
        assert_eq!(config.max_error_rate, 0.1);
        assert!(config.test_edge_cases);
        assert!(!config.test_long_contexts);
        assert!(config.validate_reference_outputs);
    }

    #[test]
    fn test_argument_parsing() {
        // Simple test for argument parsing logic
        let args = vec![
            "program".to_string(),
            "model.gguf".to_string(),
            "--verbose".to_string(),
            "--tolerance".to_string(),
            "0.001".to_string()
        ];
        
        assert!(args.contains(&"--verbose".to_string()));
        assert!(args.contains(&"--tolerance".to_string()));
        
        if let Some(pos) = args.iter().position(|x| x == "--tolerance") {
            if pos + 1 < args.len() {
                let tolerance: Result<f32, _> = args[pos + 1].parse();
                assert!(tolerance.is_ok());
                assert_eq!(tolerance.unwrap(), 0.001);
            }
        }
    }
}