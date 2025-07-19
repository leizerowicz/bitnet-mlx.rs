//! Error Formatting and Reporting
//!
//! This module provides advanced error formatting capabilities with detailed
//! reports, severity levels, and structured output for debugging and logging.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use super::{BitNetError, BitNetErrorKind, ErrorContext};

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - not really an error
    Info,
    /// Warning - potential issue but operation can continue
    Warning,
    /// Error - operation failed but system can recover
    Error,
    /// Critical - serious error that affects system functionality
    Critical,
    /// Fatal - unrecoverable error that requires system shutdown
    Fatal,
}

impl ErrorSeverity {
    /// Returns the string representation of the severity
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Info => "INFO",
            ErrorSeverity::Warning => "WARN",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Critical => "CRITICAL",
            ErrorSeverity::Fatal => "FATAL",
        }
    }

    /// Returns the numeric level for comparison
    pub fn level(&self) -> u8 {
        match self {
            ErrorSeverity::Info => 0,
            ErrorSeverity::Warning => 1,
            ErrorSeverity::Error => 2,
            ErrorSeverity::Critical => 3,
            ErrorSeverity::Fatal => 4,
        }
    }

    /// Checks if this severity is at least as severe as another
    pub fn is_at_least(&self, other: ErrorSeverity) -> bool {
        self.level() >= other.level()
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Comprehensive error report with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Error ID for tracking
    pub error_id: String,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Timestamp when error occurred
    pub timestamp: u64,
    /// Human-readable timestamp
    pub timestamp_human: String,
    /// Error category
    pub category: String,
    /// Primary error message
    pub message: String,
    /// Detailed description
    pub description: Option<String>,
    /// Error context information
    pub context: ErrorContext,
    /// Suggested actions for resolution
    pub suggestions: Vec<String>,
    /// Related error codes or references
    pub related_errors: Vec<String>,
    /// Technical details for debugging
    pub technical_details: TechnicalDetails,
}

/// Technical details for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDetails {
    /// Component or module where error occurred
    pub component: String,
    /// Operation that was being performed
    pub operation: String,
    /// Error kind classification
    pub error_kind: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// System information
    pub system_info: SystemInfo,
    /// Performance metrics at time of error
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// System information at time of error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Available memory (bytes)
    pub available_memory: Option<u64>,
    /// CPU usage percentage
    pub cpu_usage: Option<f32>,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
}

/// Performance metrics at time of error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of active operations
    pub active_operations: u32,
    /// Average operation latency in milliseconds
    pub avg_latency_ms: f64,
    /// Error rate percentage
    pub error_rate: f32,
}

impl ErrorReport {
    /// Creates an error report from a BitNetError
    pub fn from_error(error: &BitNetError) -> Self {
        let timestamp = error.timestamp();
        let timestamp_human = format_timestamp(timestamp);
        
        let (category, suggestions, technical_details) = match &error.kind() {
            BitNetErrorKind::Device { device_type, .. } => {
                let suggestions = vec![
                    "Check device availability and drivers".to_string(),
                    "Verify device compatibility".to_string(),
                    "Try fallback to CPU device".to_string(),
                ];
                let technical_details = TechnicalDetails {
                    component: "Device Management".to_string(),
                    operation: "Device Operation".to_string(),
                    error_kind: "Device".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Device".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Memory { size, operation, .. } => {
                let mut suggestions = vec![
                    "Check available system memory".to_string(),
                    "Reduce batch size or tensor dimensions".to_string(),
                    "Enable memory cleanup and garbage collection".to_string(),
                ];
                if let Some(size) = size {
                    if *size >= 1024 * 1024 * 1024 { // >= 1GB
                        suggestions.push("Consider using streaming or chunked processing".to_string());
                    }
                }
                let technical_details = TechnicalDetails {
                    component: "Memory Management".to_string(),
                    operation: operation.clone(),
                    error_kind: "Memory".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Memory".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Tensor { shape, dtype, .. } => {
                let suggestions = vec![
                    "Verify tensor dimensions and data types".to_string(),
                    "Check input data compatibility".to_string(),
                    "Ensure sufficient memory for tensor operations".to_string(),
                ];
                let technical_details = TechnicalDetails {
                    component: "Tensor Operations".to_string(),
                    operation: "Tensor Operation".to_string(),
                    error_kind: "Tensor".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Tensor".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Conversion { from_type, to_type, .. } => {
                let suggestions = vec![
                    format!("Check if conversion from {} to {} is supported", from_type, to_type),
                    "Consider intermediate conversion steps".to_string(),
                    "Verify data ranges and precision requirements".to_string(),
                ];
                let technical_details = TechnicalDetails {
                    component: "Data Conversion".to_string(),
                    operation: "Type Conversion".to_string(),
                    error_kind: "Conversion".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Conversion".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Metal { device_name, operation, .. } => {
                let mut suggestions = vec![
                    "Check Metal framework availability".to_string(),
                    "Verify GPU drivers and system compatibility".to_string(),
                    "Try fallback to CPU processing".to_string(),
                ];
                if device_name.is_some() {
                    suggestions.push("Check device-specific limitations".to_string());
                }
                let technical_details = TechnicalDetails {
                    component: "Metal GPU".to_string(),
                    operation: operation.clone(),
                    error_kind: "Metal".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Metal".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Mlx { operation, .. } => {
                let suggestions = vec![
                    "Check MLX framework installation".to_string(),
                    "Verify Apple Silicon compatibility".to_string(),
                    "Ensure MLX feature is enabled".to_string(),
                ];
                let technical_details = TechnicalDetails {
                    component: "MLX Framework".to_string(),
                    operation: operation.clone(),
                    error_kind: "MLX".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("MLX".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Configuration { parameter, .. } => {
                let mut suggestions = vec![
                    "Check configuration file syntax".to_string(),
                    "Verify parameter values and ranges".to_string(),
                    "Consult documentation for valid options".to_string(),
                ];
                if let Some(param) = parameter {
                    suggestions.push(format!("Review '{}' parameter specification", param));
                }
                let technical_details = TechnicalDetails {
                    component: "Configuration".to_string(),
                    operation: "Configuration Parsing".to_string(),
                    error_kind: "Configuration".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Configuration".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Io { path, .. } => {
                let mut suggestions = vec![
                    "Check file permissions and accessibility".to_string(),
                    "Verify file path exists".to_string(),
                    "Ensure sufficient disk space".to_string(),
                ];
                if let Some(path) = path {
                    suggestions.push(format!("Verify path: {}", path));
                }
                let technical_details = TechnicalDetails {
                    component: "I/O Operations".to_string(),
                    operation: "File Operation".to_string(),
                    error_kind: "IO".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("I/O".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Concurrency { resource, .. } => {
                let suggestions = vec![
                    "Check for deadlocks or race conditions".to_string(),
                    "Reduce concurrent access to shared resources".to_string(),
                    "Consider using different synchronization strategies".to_string(),
                    format!("Review access patterns for resource: {}", resource),
                ];
                let technical_details = TechnicalDetails {
                    component: "Concurrency Control".to_string(),
                    operation: "Resource Access".to_string(),
                    error_kind: "Concurrency".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Concurrency".to_string(), suggestions, technical_details)
            }
            BitNetErrorKind::Internal { component, .. } => {
                let suggestions = vec![
                    "This is an internal error - please report it".to_string(),
                    "Try restarting the operation".to_string(),
                    "Check system resources and stability".to_string(),
                ];
                let technical_details = TechnicalDetails {
                    component: component.clone(),
                    operation: "Internal Operation".to_string(),
                    error_kind: "Internal".to_string(),
                    stack_trace: None,
                    system_info: get_system_info(),
                    performance_metrics: None,
                };
                ("Internal".to_string(), suggestions, technical_details)
            }
        };

        Self {
            error_id: error.error_id().to_string(),
            severity: error.severity(),
            timestamp,
            timestamp_human,
            category,
            message: error.kind().to_string(),
            description: error.context().description.clone(),
            context: error.context().clone(),
            suggestions,
            related_errors: Vec::new(),
            technical_details,
        }
    }

    /// Formats the error report as a human-readable string
    pub fn format_human_readable(&self) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("🚨 {} Error Report\n", self.severity));
        output.push_str(&format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"));
        output.push_str(&format!("Error ID: {}\n", self.error_id));
        output.push_str(&format!("Time: {}\n", self.timestamp_human));
        output.push_str(&format!("Category: {}\n", self.category));
        output.push_str(&format!("Component: {}\n", self.technical_details.component));
        
        // Include device information if available in context
        if let Some(device_info) = self.context.get("device_name").or_else(|| self.context.get("device")) {
            output.push_str(&format!("Device: {}\n", device_info));
        }
        
        output.push_str(&format!("\n📋 Message:\n{}\n", self.message));
        
        if let Some(ref description) = self.description {
            output.push_str(&format!("\n📝 Description:\n{}\n", description));
        }
        
        if !self.context.is_empty() {
            output.push_str(&format!("\n🔍 Context:\n{}\n", self.context));
        }
        
        if !self.suggestions.is_empty() {
            output.push_str("\n💡 Suggested Actions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, suggestion));
            }
        }
        
        output.push_str(&format!("\n🔧 Technical Details:\n"));
        output.push_str(&format!("  Operation: {}\n", self.technical_details.operation));
        output.push_str(&format!("  Error Kind: {}\n", self.technical_details.error_kind));
        output.push_str(&format!("  System: {} ({})\n", 
                                self.technical_details.system_info.os,
                                self.technical_details.system_info.arch));
        
        if let Some(ref gpu_info) = self.technical_details.system_info.gpu_info {
            output.push_str(&format!("  GPU: {}\n", gpu_info));
        }
        
        if let Some(memory) = self.technical_details.system_info.available_memory {
            output.push_str(&format!("  Available Memory: {} MB\n", memory / 1024 / 1024));
        }
        
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        output
    }

    /// Formats the error report as JSON
    pub fn format_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Formats the error report as a compact single line
    pub fn format_compact(&self) -> String {
        format!("[{}] {} - {} ({})", 
                self.severity.as_str(),
                self.category,
                self.message,
                self.error_id)
    }
}

/// Error formatter with customizable output options
pub struct ErrorFormatter {
    /// Include technical details in output
    pub include_technical_details: bool,
    /// Include suggestions in output
    pub include_suggestions: bool,
    /// Include context information
    pub include_context: bool,
    /// Maximum width for formatted output
    pub max_width: usize,
    /// Use colored output (if supported)
    pub use_colors: bool,
}

impl ErrorFormatter {
    /// Creates a new error formatter with default settings
    pub fn new() -> Self {
        Self {
            include_technical_details: true,
            include_suggestions: true,
            include_context: true,
            max_width: 80,
            use_colors: false,
        }
    }

    /// Creates a minimal formatter for compact output
    pub fn minimal() -> Self {
        Self {
            include_technical_details: false,
            include_suggestions: false,
            include_context: false,
            max_width: 120,
            use_colors: false,
        }
    }

    /// Formats a BitNetError using this formatter
    pub fn format(&self, error: &BitNetError) -> String {
        let report = ErrorReport::from_error(error);
        
        if !self.include_technical_details && !self.include_suggestions && !self.include_context {
            report.format_compact()
        } else {
            report.format_human_readable()
        }
    }
}

impl Default for ErrorFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Formats a timestamp as a human-readable string
fn format_timestamp(timestamp: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    
    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp);
    // For simplicity, we'll use a basic format
    // In a real implementation, you might want to use chrono or time crate
    format!("{:?}", datetime)
}

/// Gets current system information
fn get_system_info() -> SystemInfo {
    SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        available_memory: get_available_memory(),
        cpu_usage: None, // Would require system monitoring
        gpu_info: get_gpu_info(),
    }
}

/// Gets available system memory (placeholder implementation)
fn get_available_memory() -> Option<u64> {
    // This is a placeholder - real implementation would use system APIs
    None
}

/// Gets GPU information (placeholder implementation)
fn get_gpu_info() -> Option<String> {
    // This is a placeholder - real implementation would query GPU drivers
    #[cfg(target_os = "macos")]
    {
        Some("Apple Silicon GPU".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{BitNetError, BitNetErrorKind};

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Fatal.is_at_least(ErrorSeverity::Critical));
        assert!(ErrorSeverity::Critical.is_at_least(ErrorSeverity::Error));
        assert!(ErrorSeverity::Error.is_at_least(ErrorSeverity::Warning));
        assert!(ErrorSeverity::Warning.is_at_least(ErrorSeverity::Info));
        
        assert!(!ErrorSeverity::Info.is_at_least(ErrorSeverity::Warning));
    }

    #[test]
    fn test_error_report_creation() {
        let error = BitNetError::device_error("Test device error", "Metal")
            .add_context("operation", "buffer_allocation")
            .set_severity(ErrorSeverity::Critical);
        
        let report = ErrorReport::from_error(&error);
        
        assert_eq!(report.category, "Device");
        assert_eq!(report.severity, ErrorSeverity::Critical);
        assert!(!report.suggestions.is_empty());
        assert_eq!(report.technical_details.component, "Device Management");
    }

    #[test]
    fn test_error_formatter() {
        let error = BitNetError::memory_error_with_size("Out of memory", 1024, "allocation");
        
        let formatter = ErrorFormatter::new();
        let formatted = formatter.format(&error);
        
        assert!(formatted.contains("Memory"));
        assert!(formatted.contains("Out of memory"));
        
        let minimal_formatter = ErrorFormatter::minimal();
        let minimal_formatted = minimal_formatter.format(&error);
        
        assert!(minimal_formatted.len() < formatted.len());
    }

    #[test]
    fn test_error_report_json_serialization() {
        let error = BitNetError::tensor_error("Shape mismatch");
        let report = ErrorReport::from_error(&error);
        
        let json_result = report.format_json();
        assert!(json_result.is_ok());
        
        let json_str = json_result.unwrap();
        assert!(json_str.contains("tensor"));
        assert!(json_str.contains("Shape mismatch"));
    }

    #[test]
    fn test_system_info() {
        let system_info = get_system_info();
        
        assert!(!system_info.os.is_empty());
        assert!(!system_info.arch.is_empty());
    }

    #[test]
    fn test_compact_formatting() {
        let error = BitNetError::conversion_error("Invalid conversion", "f32", "i8")
            .set_severity(ErrorSeverity::Warning);
        
        let report = ErrorReport::from_error(&error);
        let compact = report.format_compact();
        
        assert!(compact.contains("WARN"));
        assert!(compact.contains("Conversion"));
        assert!(compact.contains("Invalid conversion"));
        assert!(compact.len() < 200); // Should be reasonably compact
    }
}