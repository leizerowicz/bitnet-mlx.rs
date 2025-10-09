//! Configuration command implementations

use crate::{ConfigCommand, Cli};
use crate::config::{CliConfig, CliConfigOverrides};
use anyhow::Result;
use serde_json;
use std::path::PathBuf;

/// Handle configuration commands
pub async fn handle_config_command(config_cmd: &ConfigCommand, cli: &Cli) -> Result<()> {
    // Load current configuration
    let config_path = if let Some(path) = &cli.config {
        path.clone()
    } else {
        CliConfig::default_config_path()
    };

    let mut config = CliConfig::load_from_file(&config_path)?;

    match config_cmd {
        ConfigCommand::Show => {
            println!("📋 Current BitNet CLI Configuration");
            println!("Configuration file: {}", config_path.display());
            println!();
            
            // Pretty print the configuration
            match cli.output.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&config)?);
                }
                "yaml" => {
                    println!("{}", serde_yaml::to_string(&config)?);
                }
                _ => {
                    print_config_readable(&config);
                }
            }
        }
        
        ConfigCommand::Set { key, value } => {
            println!("🔧 Setting configuration: {} = {}", key, value);
            
            // Parse and set the configuration value
            set_config_value(&mut config, key, value)?;
            
            // Save the updated configuration
            config.save_to_file(&config_path)?;
            
            println!("✅ Configuration updated and saved to: {}", config_path.display());
        }
        
        ConfigCommand::Get { key } => {
            match get_config_value(&config, key)? {
                Some(value) => {
                    if cli.verbose {
                        println!("📋 Configuration value for '{}':", key);
                    }
                    println!("{}", value);
                }
                None => {
                    println!("❌ Configuration key '{}' not found", key);
                    std::process::exit(1);
                }
            }
        }
        
        ConfigCommand::Reset { confirm } => {
            if !confirm {
                println!("⚠️  This will reset all configuration to defaults.");
                println!("💡 Use --confirm to proceed with reset.");
                return Ok(());
            }
            
            println!("🔄 Resetting configuration to defaults...");
            
            let default_config = CliConfig::default();
            default_config.save_to_file(&config_path)?;
            
            println!("✅ Configuration reset to defaults and saved to: {}", config_path.display());
        }
        
        ConfigCommand::Export { output, format } => {
            println!("📤 Exporting configuration to: {}", output.display());
            
            // Create a temporary path with the desired extension for format detection
            let mut export_path = output.clone();
            if export_path.extension().is_none() {
                export_path.set_extension(format);
            }
            
            config.save_to_file(&export_path)?;
            
            println!("✅ Configuration exported in {} format", format);
        }
    }

    Ok(())
}

/// Print configuration in human-readable format
fn print_config_readable(config: &CliConfig) {
    println!("🔧 General Settings:");
    println!("  Default Model: {}", config.default_model.as_deref().unwrap_or("Not set"));
    println!("  Default Device: {}", config.default_device);
    println!("  Cache Directory: {}", config.cache_dir.display());
    println!();
    
    println!("🎛️  Generation Defaults:");
    println!("  Temperature: {}", config.generation.temperature);
    println!("  Top-K: {}", config.generation.top_k);
    println!("  Top-P: {}", config.generation.top_p);
    println!("  Max Tokens: {}", config.generation.max_tokens);
    println!("  Stop Tokens: {:?}", config.generation.stop_tokens);
    println!();
    
    println!("📤 Output Settings:");
    println!("  Format: {}", config.output.format);
    println!("  Colored: {}", config.output.colored);
    println!("  Timestamps: {}", config.output.timestamps);
    println!("  Verbosity: {}", config.output.verbosity);
    println!();
    
    println!("📊 Monitoring Settings:");
    println!("  Enabled: {}", config.monitoring.enabled);
    println!("  Show Tokens/sec: {}", config.monitoring.show_tokens_per_sec);
    println!("  Show Memory Usage: {}", config.monitoring.show_memory_usage);
    println!("  Show Latency Breakdown: {}", config.monitoring.show_latency_breakdown);
}

/// Set a configuration value using dot notation
fn set_config_value(config: &mut CliConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "default_model" => config.default_model = Some(value.to_string()),
        "default_device" => config.default_device = value.to_string(),
        "cache_dir" => config.cache_dir = PathBuf::from(value),
        
        "generation.temperature" => config.generation.temperature = value.parse()?,
        "generation.top_k" => config.generation.top_k = value.parse()?,
        "generation.top_p" => config.generation.top_p = value.parse()?,
        "generation.max_tokens" => config.generation.max_tokens = value.parse()?,
        
        "output.format" => config.output.format = value.to_string(),
        "output.colored" => config.output.colored = value.parse()?,
        "output.timestamps" => config.output.timestamps = value.parse()?,
        "output.verbosity" => config.output.verbosity = value.parse()?,
        
        "monitoring.enabled" => config.monitoring.enabled = value.parse()?,
        "monitoring.show_tokens_per_sec" => config.monitoring.show_tokens_per_sec = value.parse()?,
        "monitoring.show_memory_usage" => config.monitoring.show_memory_usage = value.parse()?,
        "monitoring.show_latency_breakdown" => config.monitoring.show_latency_breakdown = value.parse()?,
        
        _ => {
            return Err(anyhow::anyhow!("Unknown configuration key: {}", key));
        }
    }
    
    Ok(())
}

/// Get a configuration value using dot notation
fn get_config_value(config: &CliConfig, key: &str) -> Result<Option<String>> {
    let value = match key {
        "default_model" => config.default_model.as_ref().map(|s| s.clone()),
        "default_device" => Some(config.default_device.clone()),
        "cache_dir" => Some(config.cache_dir.display().to_string()),
        
        "generation.temperature" => Some(config.generation.temperature.to_string()),
        "generation.top_k" => Some(config.generation.top_k.to_string()),
        "generation.top_p" => Some(config.generation.top_p.to_string()),
        "generation.max_tokens" => Some(config.generation.max_tokens.to_string()),
        "generation.stop_tokens" => Some(format!("{:?}", config.generation.stop_tokens)),
        
        "output.format" => Some(config.output.format.clone()),
        "output.colored" => Some(config.output.colored.to_string()),
        "output.timestamps" => Some(config.output.timestamps.to_string()),
        "output.verbosity" => Some(config.output.verbosity.to_string()),
        
        "monitoring.enabled" => Some(config.monitoring.enabled.to_string()),
        "monitoring.show_tokens_per_sec" => Some(config.monitoring.show_tokens_per_sec.to_string()),
        "monitoring.show_memory_usage" => Some(config.monitoring.show_memory_usage.to_string()),
        "monitoring.show_latency_breakdown" => Some(config.monitoring.show_latency_breakdown.to_string()),
        
        _ => None,
    };
    
    Ok(value)
}