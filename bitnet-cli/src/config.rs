//! Configuration management for BitNet CLI

use crate::error::CliError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use anyhow::Result;

/// Configuration overrides that can be applied at runtime
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliConfigOverrides {
    pub default_model: Option<String>,
    pub output_format: Option<String>,
    pub verbose: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub default_model: Option<String>,
    pub default_device: String,
    pub cache_dir: PathBuf,
    pub generation: GenerationConfig,
    pub output: OutputConfig,
    pub monitoring: MonitoringConfig,
    // Keep legacy operations for backward compatibility
    pub operations: Option<LegacyOperationsConfig>,
    pub logging: Option<LoggingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub stop_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: String,
    pub colored: bool,
    pub timestamps: bool,
    pub verbosity: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub show_tokens_per_sec: bool,
    pub show_memory_usage: bool,
    pub show_latency_breakdown: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyOperationsConfig {
    pub validation: ValidationConfig,
    pub profiling: ProfilingConfig,
    pub monitoring: LegacyMonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub timeout_seconds: u64,
    pub fail_fast: bool,
    pub critical_only: bool,
    pub required_env_vars: Vec<String>,
    pub required_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub default_duration_seconds: u64,
    pub collection_interval_seconds: u64,
    pub include_gpu: bool,
    pub max_data_points: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyMonitoringConfig {
    pub platforms: Vec<String>,
    pub health_check_interval_seconds: u64,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub latency_ms: u64,
    pub error_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyOutputConfig {
    pub format: String,
    pub color: bool,
    pub verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<PathBuf>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_device: "auto".to_string(),
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("bitnet"),
            generation: GenerationConfig::default(),
            output: OutputConfig::default(),
            monitoring: MonitoringConfig::default(),
            operations: None, // Legacy field - will be None by default
            logging: Some(LoggingConfig::default()),
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            max_tokens: 512,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: "text".to_string(),
            colored: true,
            timestamps: false,
            verbosity: 1,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_tokens_per_sec: true,
            show_memory_usage: false,
            show_latency_breakdown: false,
        }
    }
}

impl Default for LegacyOperationsConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            profiling: ProfilingConfig::default(),
            monitoring: LegacyMonitoringConfig::default(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 300,
            fail_fast: false,
            critical_only: false,
            required_env_vars: vec![
                "BITNET_MODEL_PATH".to_string(),
                "BITNET_DEVICE".to_string(),
                "BITNET_LOG_LEVEL".to_string(),
            ],
            required_paths: vec![
                PathBuf::from("/tmp/bitnet"),
                PathBuf::from("./models"),
            ],
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            default_duration_seconds: 60,
            collection_interval_seconds: 1,
            include_gpu: true,
            max_data_points: 3600, // 1 hour at 1-second intervals
        }
    }
}

impl Default for LegacyMonitoringConfig {
    fn default() -> Self {
        Self {
            platforms: vec!["prometheus".to_string()],
            health_check_interval_seconds: 30,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_percent: 80.0,
            memory_percent: 85.0,
            latency_ms: 1000,
            error_rate_percent: 5.0,
        }
    }
}

impl Default for LegacyOutputConfig {
    fn default() -> Self {
        Self {
            format: "table".to_string(),
            color: true,
            verbose: false,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: None,
        }
    }
}

impl CliConfig {
    /// Get the default config file path
    pub fn default_config_path() -> PathBuf {
        if let Some(home) = dirs::home_dir() {
            home.join(".config/bitnet/config.yaml")
        } else {
            PathBuf::from("bitnet.yaml")
        }
    }

    /// Load configuration from a file
    pub fn load_from_file(path: &PathBuf) -> Result<Self, CliError> {
        if path.exists() {
            let contents = std::fs::read_to_string(path)?;
            
            // Try YAML first, then JSON
            if path.extension().and_then(|s| s.to_str()) == Some("yaml") || 
               path.extension().and_then(|s| s.to_str()) == Some("yml") {
                serde_yaml::from_str(&contents)
                    .map_err(|e| CliError::Configuration(format!("Invalid YAML config: {}", e)))
            } else {
                // For now, assume JSON if not YAML
                serde_json::from_str(&contents)
                    .map_err(|e| CliError::Configuration(format!("Invalid JSON config: {}", e)))
            }
        } else {
            Ok(Self::default())
        }
    }

    /// Save configuration to a file
    pub fn save_to_file(&self, path: &PathBuf) -> Result<(), CliError> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(self)
                .map_err(|e| CliError::Serialization(format!("Failed to serialize to JSON: {}", e)))?
        } else {
            // Default to YAML
            serde_yaml::to_string(self)
                .map_err(|e| CliError::Serialization(format!("Failed to serialize to YAML: {}", e)))?
        };
            
        std::fs::write(path, contents)?;
            
        Ok(())
    }

    pub fn load(path: Option<PathBuf>) -> Result<Self, CliError> {
        match path {
            Some(config_path) => Self::load_from_file(&config_path),
            None => Ok(Self::default()),
        }
    }
    
    pub fn save(&self, path: &PathBuf) -> Result<(), CliError> {
        self.save_to_file(path)
    }
    
    pub fn validation_timeout(&self) -> Duration {
        self.operations
            .as_ref()
            .map(|ops| Duration::from_secs(ops.validation.timeout_seconds))
            .unwrap_or(Duration::from_secs(300)) // Default timeout
    }
    
    pub fn profiling_duration(&self) -> Duration {
        self.operations
            .as_ref()
            .map(|ops| Duration::from_secs(ops.profiling.default_duration_seconds))
            .unwrap_or(Duration::from_secs(60)) // Default duration
    }
    
    pub fn collection_interval(&self) -> Duration {
        self.operations
            .as_ref()
            .map(|ops| Duration::from_secs(ops.profiling.collection_interval_seconds))
            .unwrap_or(Duration::from_secs(1)) // Default interval
    }
}

/// Load configuration from various sources with precedence:
/// 1. Command line config file
/// 2. Environment variable BITNET_CONFIG
/// 3. ~/.config/bitnet/config.yaml
/// 4. ./bitnet.yaml
/// 5. Default configuration
pub fn load_config(cli_config_path: Option<PathBuf>) -> Result<CliConfig, CliError> {
    // Try CLI-specified config first
    if let Some(path) = cli_config_path {
        return CliConfig::load(Some(path));
    }
    
    // Try environment variable
    if let Ok(env_path) = std::env::var("BITNET_CONFIG") {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return CliConfig::load(Some(path));
        }
    }
    
    // Try user config directory
    if let Some(home) = dirs::home_dir() {
        let user_config = home.join(".config/bitnet/config.yaml");
        if user_config.exists() {
            return CliConfig::load(Some(user_config));
        }
    }
    
    // Try local config file
    let local_config = PathBuf::from("bitnet.yaml");
    if local_config.exists() {
        return CliConfig::load(Some(local_config));
    }
    
    // Fall back to default
    Ok(CliConfig::default())
}
