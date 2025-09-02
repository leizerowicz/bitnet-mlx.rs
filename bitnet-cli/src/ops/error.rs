//! Error types for BitNet CLI operations

use thiserror::Error;
use std::time::Duration;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("System validation failed: {component}")]
    SystemValidationFailed {
        component: String,
        details: Vec<String>,
    },
    
    #[error("Performance profiling error in {stage}")]
    ProfilingError {
        stage: String,
        metrics_collected: usize,
        duration_elapsed: Duration,
    },
    
    #[error("Monitoring integration failed for {platform}")]
    MonitoringIntegrationFailed {
        platform: String,
        setup_stage: String,
        partial_success: bool,
    },
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("System error: {0}")]
    System(String),
    
    #[error("Timeout occurred: {operation} took longer than {timeout:?}")]
    Timeout {
        operation: String,
        timeout: Duration,
    },
}

impl OpsError {
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::SystemValidationFailed { component, details } => {
                let mut suggestions = vec![
                    format!("Check {} configuration", component),
                    "Verify system requirements are met".to_string(),
                ];
                suggestions.extend(details.iter().cloned());
                suggestions
            }
            Self::ProfilingError { stage, .. } => {
                vec![
                    format!("Retry profiling from {} stage", stage),
                    "Check system resources availability".to_string(),
                    "Reduce profiling duration or interval".to_string(),
                ]
            }
            Self::MonitoringIntegrationFailed { platform, setup_stage, partial_success } => {
                let mut suggestions = vec![
                    format!("Check {} platform connectivity", platform),
                    format!("Verify {} stage configuration", setup_stage),
                ];
                if *partial_success {
                    suggestions.push("Some integrations succeeded - check logs for details".to_string());
                }
                suggestions
            }
            Self::Configuration(msg) => {
                vec![
                    "Verify configuration file syntax".to_string(),
                    "Check environment variables".to_string(),
                    format!("Configuration issue: {}", msg),
                ]
            }
            Self::System(msg) => {
                vec![
                    "Check system permissions".to_string(),
                    "Verify system dependencies".to_string(),
                    format!("System issue: {}", msg),
                ]
            }
            Self::Timeout { operation, .. } => {
                vec![
                    format!("Increase timeout for {}", operation),
                    "Check system performance".to_string(),
                    "Reduce operation complexity".to_string(),
                ]
            }
        }
    }
    
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::SystemValidationFailed { .. } => false, // Configuration issues
            Self::ProfilingError { .. } => true,          // Often transient
            Self::MonitoringIntegrationFailed { partial_success, .. } => *partial_success,
            Self::Configuration(_) => false,              // Configuration issues
            Self::System(_) => false,                     // System issues
            Self::Timeout { .. } => true,                 // May succeed with retry
        }
    }
    
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::SystemValidationFailed { .. } => ErrorSeverity::High,
            Self::ProfilingError { .. } => ErrorSeverity::Medium,
            Self::MonitoringIntegrationFailed { partial_success, .. } => {
                if *partial_success {
                    ErrorSeverity::Medium
                } else {
                    ErrorSeverity::High
                }
            }
            Self::Configuration(_) => ErrorSeverity::High,
            Self::System(_) => ErrorSeverity::High,
            Self::Timeout { .. } => ErrorSeverity::Medium,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct RecoverySuggestion {
    pub action: String,
    pub command: Option<String>,
    pub priority: Priority,
    pub estimated_time: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}
