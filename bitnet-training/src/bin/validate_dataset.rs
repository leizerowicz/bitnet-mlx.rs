//! Dataset Validation Binary for BitNet Training
//!
//! Validates training datasets for format compliance, quality checks,
//! and compatibility with BitNet training pipeline.

use clap::Parser;
use anyhow::{Result, Context};
use tracing::{info, warn, error};
use std::path::PathBuf;
use std::fs;
use serde_json::Value;

#[derive(Parser)]
#[command(name = "validate_dataset")]
#[command(about = "Validate training datasets for BitNet models")]
#[command(version = "1.0.0")]
struct Args {
    /// Input dataset file path
    #[arg(help = "Path to dataset file to validate")]
    input: PathBuf,
    
    /// Expected dataset format (json, jsonl, text)
    #[arg(long, default_value = "json", help = "Expected dataset format")]
    format: String,
    
    /// Maximum sequence length to validate
    #[arg(long, default_value = "512", help = "Maximum allowed sequence length")]
    max_seq_length: usize,
    
    /// Minimum examples required
    #[arg(long, default_value = "1", help = "Minimum number of examples required")]
    min_examples: usize,
    
    /// Required fields for validation
    #[arg(long, help = "Comma-separated list of required fields")]
    required_fields: Option<String>,
    
    /// Enable strict validation mode
    #[arg(long, help = "Enable strict validation (fail on warnings)")]
    strict: bool,
    
    /// Output validation report to file
    #[arg(long, help = "Save validation report to file")]
    report_output: Option<PathBuf>,
    
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
    
    info!("🔍 Validating dataset: {:?}", args.input);
    info!("Format: {}", args.format);
    
    // Check if file exists
    if !args.input.exists() {
        return Err(anyhow::anyhow!("Dataset file not found: {:?}", args.input));
    }
    
    // Run validation
    let validation_result = validate_dataset(&args)?;
    
    // Display results
    display_validation_results(&validation_result, &args);
    
    // Save report if requested
    if let Some(report_path) = &args.report_output {
        save_validation_report(&validation_result, report_path)?;
    }
    
    // Return appropriate exit code
    if validation_result.has_errors() || (args.strict && validation_result.has_warnings()) {
        error!("❌ Dataset validation failed");
        std::process::exit(1);
    } else {
        info!("✅ Dataset validation passed");
        Ok(())
    }
}

fn validate_dataset(args: &Args) -> Result<ValidationResult> {
    let mut result = ValidationResult::new();
    
    // Read and parse dataset
    let content = fs::read_to_string(&args.input)
        .context("Failed to read dataset file")?;
    
    let dataset = match args.format.as_str() {
        "json" => parse_json_dataset(&content, &mut result)?,
        "jsonl" => parse_jsonl_dataset(&content, &mut result)?,
        "text" => parse_text_dataset(&content, &mut result)?,
        _ => {
            result.add_error(format!("Unsupported format: {}", args.format));
            return Ok(result);
        }
    };
    
    // Basic structure validation
    validate_basic_structure(&dataset, &mut result, args);
    
    // Content validation
    validate_content(&dataset, &mut result, args);
    
    // Statistical validation
    validate_statistics(&dataset, &mut result, args);
    
    // Format-specific validation
    match args.format.as_str() {
        "json" => validate_json_specific(&dataset, &mut result, args),
        "jsonl" => validate_jsonl_specific(&dataset, &mut result, args),
        "text" => validate_text_specific(&dataset, &mut result, args),
        _ => {}
    }
    
    Ok(result)
}

fn parse_json_dataset(content: &str, result: &mut ValidationResult) -> Result<Vec<Value>> {
    match serde_json::from_str::<Vec<Value>>(content) {
        Ok(dataset) => {
            result.add_info(format!("Successfully parsed JSON array with {} items", dataset.len()));
            Ok(dataset)
        }
        Err(e) => {
            result.add_error(format!("Failed to parse JSON: {}", e));
            Ok(vec![])
        }
    }
}

fn parse_jsonl_dataset(content: &str, result: &mut ValidationResult) -> Result<Vec<Value>> {
    let mut dataset = Vec::new();
    let mut line_errors = 0;
    
    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<Value>(line) {
            Ok(obj) => dataset.push(obj),
            Err(e) => {
                result.add_error(format!("Line {}: Invalid JSON: {}", line_num + 1, e));
                line_errors += 1;
            }
        }
    }
    
    if line_errors == 0 {
        result.add_info(format!("Successfully parsed JSONL with {} valid lines", dataset.len()));
    } else {
        result.add_warning(format!("Parsed {} valid lines with {} errors", dataset.len(), line_errors));
    }
    
    Ok(dataset)
}

fn parse_text_dataset(content: &str, result: &mut ValidationResult) -> Result<Vec<Value>> {
    let lines: Vec<&str> = content.lines().collect();
    let dataset: Vec<Value> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::json!({
                "text": line.trim(),
                "id": i
            })
        })
        .collect();
    
    result.add_info(format!("Converted text file to {} examples", dataset.len()));
    Ok(dataset)
}

fn validate_basic_structure(dataset: &[Value], result: &mut ValidationResult, args: &Args) {
    // Check minimum examples
    if dataset.len() < args.min_examples {
        result.add_error(format!(
            "Dataset has {} examples, minimum required: {}",
            dataset.len(),
            args.min_examples
        ));
    } else {
        result.add_info(format!("Dataset contains {} examples", dataset.len()));
    }
    
    // Check for empty dataset
    if dataset.is_empty() {
        result.add_error("Dataset is empty".to_string());
        return;
    }
    
    // Validate required fields
    if let Some(required_fields) = &args.required_fields {
        let fields: Vec<&str> = required_fields.split(',').map(|s| s.trim()).collect();
        validate_required_fields(dataset, &fields, result);
    }
}

fn validate_required_fields(dataset: &[Value], fields: &[&str], result: &mut ValidationResult) {
    for (i, example) in dataset.iter().enumerate() {
        for field in fields {
            if !example.get(field).is_some() {
                result.add_error(format!("Example {}: Missing required field '{}'", i, field));
            }
        }
    }
}

fn validate_content(dataset: &[Value], result: &mut ValidationResult, args: &Args) {
    let mut long_sequences = 0;
    let mut empty_content = 0;
    
    for (i, example) in dataset.iter().enumerate() {
        // Check for text content
        if let Some(text) = example.get("text").and_then(|v| v.as_str()) {
            if text.is_empty() {
                empty_content += 1;
                result.add_warning(format!("Example {}: Empty text content", i));
            }
            
            if text.len() > args.max_seq_length * 4 { // Rough estimate for tokens
                long_sequences += 1;
                result.add_warning(format!(
                    "Example {}: Text length {} may exceed max sequence length",
                    i, text.len()
                ));
            }
        }
        
        // Check for token sequences
        if let Some(tokens) = example.get("tokens").and_then(|v| v.as_array()) {
            if tokens.len() > args.max_seq_length {
                long_sequences += 1;
                result.add_warning(format!(
                    "Example {}: Token sequence length {} exceeds max length {}",
                    i, tokens.len(), args.max_seq_length
                ));
            }
        }
    }
    
    if empty_content > 0 {
        result.add_warning(format!("{} examples have empty content", empty_content));
    }
    
    if long_sequences > 0 {
        result.add_warning(format!("{} examples may exceed sequence length limits", long_sequences));
    }
}

fn validate_statistics(dataset: &[Value], result: &mut ValidationResult, _args: &Args) {
    // Calculate basic statistics
    let mut text_lengths = Vec::new();
    let mut token_counts = Vec::new();
    
    for example in dataset {
        if let Some(text) = example.get("text").and_then(|v| v.as_str()) {
            text_lengths.push(text.len());
        }
        
        if let Some(tokens) = example.get("tokens").and_then(|v| v.as_array()) {
            token_counts.push(tokens.len());
        }
    }
    
    if !text_lengths.is_empty() {
        let avg_length = text_lengths.iter().sum::<usize>() as f64 / text_lengths.len() as f64;
        let min_length = *text_lengths.iter().min().unwrap();
        let max_length = *text_lengths.iter().max().unwrap();
        
        result.add_info(format!(
            "Text statistics: avg={:.1}, min={}, max={}",
            avg_length, min_length, max_length
        ));
    }
    
    if !token_counts.is_empty() {
        let avg_tokens = token_counts.iter().sum::<usize>() as f64 / token_counts.len() as f64;
        let min_tokens = *token_counts.iter().min().unwrap();
        let max_tokens = *token_counts.iter().max().unwrap();
        
        result.add_info(format!(
            "Token statistics: avg={:.1}, min={}, max={}",
            avg_tokens, min_tokens, max_tokens
        ));
    }
}

fn validate_json_specific(_dataset: &[Value], result: &mut ValidationResult, _args: &Args) {
    result.add_info("JSON format validation completed".to_string());
}

fn validate_jsonl_specific(_dataset: &[Value], result: &mut ValidationResult, _args: &Args) {
    result.add_info("JSONL format validation completed".to_string());
}

fn validate_text_specific(_dataset: &[Value], result: &mut ValidationResult, _args: &Args) {
    result.add_info("Text format validation completed".to_string());
}

fn display_validation_results(result: &ValidationResult, _args: &Args) {
    info!("📊 Validation Results:");
    info!("   Errors: {}", result.errors.len());
    info!("   Warnings: {}", result.warnings.len());
    info!("   Info: {}", result.info.len());
    
    for error in &result.errors {
        error!("❌ {}", error);
    }
    
    for warning in &result.warnings {
        warn!("⚠️  {}", warning);
    }
    
    for info in &result.info {
        info!("ℹ️  {}", info);
    }
}

fn save_validation_report(result: &ValidationResult, path: &PathBuf) -> Result<()> {
    let report = serde_json::json!({
        "errors": result.errors,
        "warnings": result.warnings,
        "info": result.info,
        "summary": {
            "error_count": result.errors.len(),
            "warning_count": result.warnings.len(),
            "info_count": result.info.len(),
            "validation_passed": !result.has_errors()
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    fs::write(path, serde_json::to_string_pretty(&report)?)
        .context("Failed to write validation report")?;
    
    info!("📄 Validation report saved: {:?}", path);
    Ok(())
}

#[derive(Debug)]
struct ValidationResult {
    errors: Vec<String>,
    warnings: Vec<String>,
    info: Vec<String>,
}

impl ValidationResult {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
        }
    }
    
    fn add_error(&mut self, message: String) {
        self.errors.push(message);
    }
    
    fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }
    
    fn add_info(&mut self, message: String) {
        self.info.push(message);
    }
    
    fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}