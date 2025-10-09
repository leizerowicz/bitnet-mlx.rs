//! Configuration management for BitNet CLI

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub operations: OperationsConfig,
    pub output: OutputConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationsConfig {
    pub validation: ValidationConfig,
    pub profiling: ProfilingConfig,
    pub monitoring: MonitoringConfig,
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
pub struct MonitoringConfig {
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
pub struct OutputConfig {
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
            operations: OperationsConfig::default(),
            output: OutputConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for OperationsConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            profiling: ProfilingConfig::default(),
            monitoring: MonitoringConfig::default(),
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

impl Default for MonitoringConfig {
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

impl Default for OutputConfig {
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
    pub fn load(path: Option<PathBuf>) -> Result<Self, CliError> {
        match path {
            Some(config_path) => {
                if config_path.exists() {
                    let contents = std::fs::read_to_string(&config_path)
                        .map_err(|e| CliError::Configuration(format!("Failed to read config file: {}", e)))?;
                    
                    // Try YAML first, then JSON
                    if config_path.extension().and_then(|s| s.to_str()) == Some("yaml") || 
                       config_path.extension().and_then(|s| s.to_str()) == Some("yml") {
                        serde_yaml::from_str(&contents)
                            .map_err(|e| CliError::Configuration(format!("Invalid YAML config: {}", e)))
                    } else {
                        // Try JSON parsing (we can add this when serde_json is available)
                        Err(CliError::Configuration("JSON config parsing not yet implemented".to_string()))
                    }
                } else {
                    Err(CliError::Configuration(format!("Config file not found: {:?}", config_path)))
                }
            }
            None => Ok(Self::default()),
        }
    }
    
    pub fn save(&self, path: &PathBuf) -> Result<(), CliError> {
        let contents = serde_yaml::to_string(self)
            .map_err(|e| CliError::Configuration(format!("Failed to serialize config: {}", e)))?;
            
        std::fs::write(path, contents)
            .map_err(|e| CliError::Configuration(format!("Failed to write config file: {}", e)))?;
            
        Ok(())
    }
    
    pub fn validation_timeout(&self) -> Duration {
        Duration::from_secs(self.operations.validation.timeout_seconds)
    }
    
    pub fn profiling_duration(&self) -> Duration {
        Duration::from_secs(self.operations.profiling.default_duration_seconds)
    }
    
    pub fn collection_interval(&self) -> Duration {
        Duration::from_secs(self.operations.profiling.collection_interval_seconds)
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
