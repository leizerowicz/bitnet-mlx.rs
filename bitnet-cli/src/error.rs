//! Error handling for BitNet CLI operations

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Operations error: {0}")]
    Operations(#[from] crate::ops::error::OpsError),
    
    #[error("Customer tools error: {0}")]
    CustomerTools(#[from] crate::customer_tools::CustomerToolsError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl CliError {
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::Configuration(msg) => {
                vec![
                    "Check your configuration file syntax".to_string(),
                    "Verify environment variables are set correctly".to_string(),
                    format!("Configuration issue: {}", msg),
                ]
            }
            Self::Operations(ops_err) => ops_err.recovery_suggestions(),
            Self::CustomerTools(customer_err) => {
                vec![
                    "Check system requirements for BitNet".to_string(),
                    "Verify hardware compatibility".to_string(),
                    "Run 'bitnet setup' to validate environment".to_string(),
                    format!("Customer tools issue: {}", customer_err),
                ]
            }
            Self::Io(io_err) => {
                vec![
                    "Check file permissions".to_string(),
                    "Verify file paths exist".to_string(),
                    format!("IO issue: {}", io_err),
                ]
            }
            Self::Serialization(msg) => {
                vec![
                    "Check input data format".to_string(),
                    "Verify JSON/YAML syntax".to_string(),
                    format!("Serialization issue: {}", msg),
                ]
            }
        }
    }
    
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Configuration(_) => false,
            Self::Operations(ops_err) => ops_err.is_retryable(),
            Self::CustomerTools(_) => true, // Customer tools errors are often retryable
            Self::Io(_) => false,
            Self::Serialization(_) => false,
        }
    }
}
