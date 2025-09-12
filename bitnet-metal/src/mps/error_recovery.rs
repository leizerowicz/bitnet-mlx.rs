//! # MPS Error Handling and Recovery
//!
//! Production-ready error handling, fallback mechanisms, and graceful degradation
//! for Metal Performance Shaders integration in BitNet.
//!
//! ## Features
//!
//! - **Graceful Degradation**: Automatic fallback when MPS/ANE unavailable
//! - **Comprehensive Error Reporting**: Detailed error context and debugging support
//! - **Device Capability Mismatch Handling**: Robust device compatibility checking
//! - **Recovery Mechanisms**: Automatic retry and fallback strategies
//! - **Production Monitoring**: Error tracking and performance impact monitoring

use anyhow::{Context, Result};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive MPS error types with production-ready context
#[derive(Debug, Clone)]
pub enum MPSError {
    /// MPS framework is not available on this platform
    UnavailablePlatform {
        platform: String,
        reason: String,
    },
    /// MPS device initialization failed
    DeviceInitializationFailed {
        device_name: Option<String>,
        metal_error: String,
        fallback_available: bool,
    },
    /// Apple Neural Engine is not available
    ANEUnavailable {
        reason: String,
        detection_method: String,
        alternative_devices: Vec<String>,
    },
    /// Device capabilities mismatch
    CapabilityMismatch {
        required_capabilities: Vec<String>,
        available_capabilities: Vec<String>,
        device_name: String,
        suggested_fallback: Option<String>,
    },
    /// MPS operation execution failed
    OperationFailed {
        operation: String,
        stage: String,
        error_details: String,
        recovery_suggestion: String,
    },
    /// Memory allocation or management error
    MemoryError {
        operation: String,
        requested_size: Option<usize>,
        available_memory: Option<usize>,
        memory_pressure: bool,
    },
    /// Thermal or power management issue
    ThermalIssue {
        current_status: String,
        recommended_action: String,
        power_state: String,
    },
    /// Model compilation or loading error
    ModelError {
        model_type: String,
        compilation_stage: String,
        error_details: String,
        validation_errors: Vec<String>,
    },
}

impl fmt::Display for MPSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MPSError::UnavailablePlatform { platform, reason } => {
                write!(f, "MPS unavailable on platform '{}': {}", platform, reason)
            }
            MPSError::DeviceInitializationFailed { device_name, metal_error, fallback_available } => {
                write!(f, 
                    "MPS device initialization failed for '{}': {}. Fallback available: {}", 
                    device_name.as_deref().unwrap_or("unknown"), 
                    metal_error, 
                    fallback_available
                )
            }
            MPSError::ANEUnavailable { reason, detection_method, alternative_devices } => {
                write!(f, 
                    "Apple Neural Engine unavailable ({}): {}. Alternatives: [{}]", 
                    detection_method, 
                    reason, 
                    alternative_devices.join(", ")
                )
            }
            MPSError::CapabilityMismatch { required_capabilities, available_capabilities, device_name, suggested_fallback } => {
                write!(f, 
                    "Device '{}' capability mismatch. Required: [{}], Available: [{}]. Suggested fallback: {}", 
                    device_name,
                    required_capabilities.join(", "),
                    available_capabilities.join(", "),
                    suggested_fallback.as_deref().unwrap_or("none")
                )
            }
            MPSError::OperationFailed { operation, stage, error_details, recovery_suggestion } => {
                write!(f, 
                    "MPS operation '{}' failed at stage '{}': {}. Recovery: {}", 
                    operation, 
                    stage, 
                    error_details, 
                    recovery_suggestion
                )
            }
            MPSError::MemoryError { operation, requested_size, available_memory, memory_pressure } => {
                write!(f, 
                    "Memory error during '{}'. Requested: {:?}, Available: {:?}, Pressure: {}", 
                    operation, 
                    requested_size, 
                    available_memory, 
                    memory_pressure
                )
            }
            MPSError::ThermalIssue { current_status, recommended_action, power_state } => {
                write!(f, 
                    "Thermal issue: {} (Power state: {}). Recommended: {}", 
                    current_status, 
                    power_state, 
                    recommended_action
                )
            }
            MPSError::ModelError { model_type, compilation_stage, error_details, validation_errors } => {
                write!(f, 
                    "Model error for '{}' at '{}': {}. Validation errors: [{}]", 
                    model_type, 
                    compilation_stage, 
                    error_details, 
                    validation_errors.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for MPSError {}

/// MPS fallback strategy configuration
#[derive(Debug, Clone)]
pub struct FallbackStrategy {
    /// Enable automatic CPU fallback
    pub enable_cpu_fallback: bool,
    /// Enable Metal compute fallback (without MPS)
    pub enable_metal_fallback: bool,
    /// Maximum retry attempts before fallback
    pub max_retry_attempts: u32,
    /// Delay between retry attempts
    pub retry_delay: Duration,
    /// Enable performance monitoring during fallback
    pub monitor_fallback_performance: bool,
}

impl Default for FallbackStrategy {
    fn default() -> Self {
        Self {
            enable_cpu_fallback: true,
            enable_metal_fallback: true,
            max_retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            monitor_fallback_performance: true,
        }
    }
}

/// Device capability requirements for operations
#[derive(Debug, Clone)]
pub struct CapabilityRequirements {
    pub neural_network_support: bool,
    pub matrix_multiplication: bool,
    pub convolution_support: bool,
    pub graph_api: bool,
    pub minimum_memory_mb: usize,
    pub ane_support: bool,
}

impl Default for CapabilityRequirements {
    fn default() -> Self {
        Self {
            neural_network_support: true,
            matrix_multiplication: true,
            convolution_support: false,
            graph_api: false,
            minimum_memory_mb: 512,
            ane_support: false,
        }
    }
}

/// Error recovery and fallback manager
#[derive(Debug)]
pub struct MPSErrorRecovery {
    fallback_strategy: FallbackStrategy,
    error_history: Arc<Mutex<Vec<(Instant, MPSError)>>>,
    fallback_performance: Arc<Mutex<FallbackPerformanceMetrics>>,
}

#[derive(Debug, Default)]
struct FallbackPerformanceMetrics {
    cpu_fallback_count: u64,
    metal_fallback_count: u64,
    total_recovery_time: Duration,
    last_fallback_time: Option<Instant>,
}

impl MPSErrorRecovery {
    /// Create a new error recovery manager
    pub fn new(fallback_strategy: FallbackStrategy) -> Self {
        Self {
            fallback_strategy,
            error_history: Arc::new(Mutex::new(Vec::new())),
            fallback_performance: Arc::new(Mutex::new(FallbackPerformanceMetrics::default())),
        }
    }

    /// Record an error and determine recovery strategy
    pub fn handle_error(&self, error: MPSError) -> RecoveryAction {
        let start_time = Instant::now();
        
        // Record error in history
        if let Ok(mut history) = self.error_history.lock() {
            history.push((start_time, error.clone()));
            
            // Keep only last 100 errors
            if history.len() > 100 {
                let len = history.len();
                history.drain(0..len - 100);
            }
        }

        // Determine recovery action based on error type
        let recovery_action = match &error {
            MPSError::UnavailablePlatform { .. } => {
                if self.fallback_strategy.enable_cpu_fallback {
                    RecoveryAction::FallbackToCPU
                } else {
                    RecoveryAction::Abort
                }
            }
            MPSError::DeviceInitializationFailed { fallback_available, .. } => {
                if *fallback_available && self.fallback_strategy.enable_metal_fallback {
                    RecoveryAction::FallbackToMetal
                } else if self.fallback_strategy.enable_cpu_fallback {
                    RecoveryAction::FallbackToCPU
                } else {
                    RecoveryAction::Abort
                }
            }
            MPSError::ANEUnavailable { alternative_devices, .. } => {
                if !alternative_devices.is_empty() && self.fallback_strategy.enable_metal_fallback {
                    RecoveryAction::FallbackToMetal
                } else if self.fallback_strategy.enable_cpu_fallback {
                    RecoveryAction::FallbackToCPU
                } else {
                    RecoveryAction::Abort
                }
            }
            MPSError::CapabilityMismatch { suggested_fallback, .. } => {
                match suggested_fallback.as_deref() {
                    Some("metal") if self.fallback_strategy.enable_metal_fallback => RecoveryAction::FallbackToMetal,
                    Some("cpu") | None if self.fallback_strategy.enable_cpu_fallback => RecoveryAction::FallbackToCPU,
                    _ => RecoveryAction::Abort,
                }
            }
            MPSError::OperationFailed { .. } => {
                if self.should_retry() {
                    RecoveryAction::Retry
                } else if self.fallback_strategy.enable_cpu_fallback {
                    RecoveryAction::FallbackToCPU
                } else {
                    RecoveryAction::Abort
                }
            }
            MPSError::MemoryError { memory_pressure, .. } => {
                if *memory_pressure {
                    RecoveryAction::ReduceMemoryUsage
                } else {
                    RecoveryAction::Retry
                }
            }
            MPSError::ThermalIssue { .. } => {
                RecoveryAction::ReducePowerUsage
            }
            MPSError::ModelError { .. } => {
                RecoveryAction::Abort
            }
        };

        // Update performance metrics
        if let Ok(mut metrics) = self.fallback_performance.lock() {
            metrics.total_recovery_time += start_time.elapsed();
            metrics.last_fallback_time = Some(start_time);
            
            match recovery_action {
                RecoveryAction::FallbackToCPU => metrics.cpu_fallback_count += 1,
                RecoveryAction::FallbackToMetal => metrics.metal_fallback_count += 1,
                _ => {}
            }
        }

        recovery_action
    }

    /// Check if we should retry based on recent error history
    fn should_retry(&self) -> bool {
        if let Ok(history) = self.error_history.lock() {
            let recent_errors = history.iter()
                .filter(|(time, _)| time.elapsed() < Duration::from_secs(60))
                .count();
            
            recent_errors < self.fallback_strategy.max_retry_attempts as usize
        } else {
            false
        }
    }

    /// Get error recovery statistics
    pub fn get_recovery_stats(&self) -> Result<RecoveryStats> {
        let history = self.error_history.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock error history"))?;
        let metrics = self.fallback_performance.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock fallback metrics"))?;

        Ok(RecoveryStats {
            total_errors: history.len(),
            recent_errors: history.iter()
                .filter(|(time, _)| time.elapsed() < Duration::from_secs(300))
                .count(),
            cpu_fallback_count: metrics.cpu_fallback_count,
            metal_fallback_count: metrics.metal_fallback_count,
            total_recovery_time: metrics.total_recovery_time,
            last_fallback_time: metrics.last_fallback_time,
        })
    }

    /// Validate device capabilities against requirements
    pub fn validate_capabilities(
        &self,
        available: &super::MPSCapabilities,
        required: &CapabilityRequirements,
        device_name: &str,
    ) -> Result<(), MPSError> {
        let mut missing_capabilities = Vec::new();
        let mut available_caps = Vec::new();

        // Check individual capabilities
        if required.neural_network_support && !available.supports_neural_network {
            missing_capabilities.push("neural_network_support".to_string());
        } else if available.supports_neural_network {
            available_caps.push("neural_network_support".to_string());
        }

        if required.matrix_multiplication && !available.supports_matrix_multiplication {
            missing_capabilities.push("matrix_multiplication".to_string());
        } else if available.supports_matrix_multiplication {
            available_caps.push("matrix_multiplication".to_string());
        }

        if required.convolution_support && !available.supports_convolution {
            missing_capabilities.push("convolution_support".to_string());
        } else if available.supports_convolution {
            available_caps.push("convolution_support".to_string());
        }

        if required.graph_api && !available.supports_graph_api {
            missing_capabilities.push("graph_api".to_string());
        } else if available.supports_graph_api {
            available_caps.push("graph_api".to_string());
        }

        // Check memory requirements
        let available_memory_mb = available.max_buffer_size / (1024 * 1024);
        if available_memory_mb < required.minimum_memory_mb {
            missing_capabilities.push(format!("minimum_memory_{}mb", required.minimum_memory_mb));
        } else {
            available_caps.push(format!("memory_{}mb", available_memory_mb));
        }

        // If any capabilities are missing, return error with suggestions
        if !missing_capabilities.is_empty() {
            let suggested_fallback = if available.supports_matrix_multiplication {
                Some("metal".to_string())
            } else {
                Some("cpu".to_string())
            };

            return Err(MPSError::CapabilityMismatch {
                required_capabilities: missing_capabilities,
                available_capabilities: available_caps,
                device_name: device_name.to_string(),
                suggested_fallback,
            });
        }

        Ok(())
    }
}

/// Recovery action recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Retry the operation
    Retry,
    /// Fall back to CPU implementation
    FallbackToCPU,
    /// Fall back to basic Metal (without MPS)
    FallbackToMetal,
    /// Reduce memory usage and retry
    ReduceMemoryUsage,
    /// Reduce power usage and retry
    ReducePowerUsage,
    /// Abort the operation
    Abort,
}

/// Recovery performance statistics
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_errors: usize,
    pub recent_errors: usize,
    pub cpu_fallback_count: u64,
    pub metal_fallback_count: u64,
    pub total_recovery_time: Duration,
    pub last_fallback_time: Option<Instant>,
}

/// Comprehensive device compatibility checker
pub struct DeviceCompatibilityChecker;

impl DeviceCompatibilityChecker {
    /// Check if MPS is available on the current platform
    pub fn is_mps_available() -> bool {
        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            // Check if we're on macOS with Metal support
            use objc::runtime::Class;
            Class::get("MPSKernel").is_some()
        }
        
        #[cfg(not(all(target_os = "macos", feature = "mps")))]
        false
    }

    /// Check if Apple Neural Engine is available
    pub fn is_ane_available() -> bool {
        #[cfg(all(target_os = "macos", feature = "ane"))]
        {
            super::ane_integration::ANEIntegration::is_available()
        }
        
        #[cfg(not(all(target_os = "macos", feature = "ane")))]
        false
    }

    /// Get detailed platform information for debugging
    pub fn get_platform_info() -> PlatformInfo {
        PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            mps_available: Self::is_mps_available(),
            ane_available: Self::is_ane_available(),
            metal_feature_enabled: cfg!(feature = "metal"),
            mps_feature_enabled: cfg!(feature = "mps"),
            ane_feature_enabled: cfg!(feature = "ane"),
        }
    }
}

/// Platform compatibility information
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub mps_available: bool,
    pub ane_available: bool,
    pub metal_feature_enabled: bool,
    pub mps_feature_enabled: bool,
    pub ane_feature_enabled: bool,
}

impl fmt::Display for PlatformInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, 
            "Platform: {}-{}, MPS: {} (feature: {}), ANE: {} (feature: {}), Metal feature: {}", 
            self.os, 
            self.arch, 
            self.mps_available, 
            self.mps_feature_enabled,
            self.ane_available, 
            self.ane_feature_enabled,
            self.metal_feature_enabled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_compatibility_check() {
        let info = DeviceCompatibilityChecker::get_platform_info();
        println!("Platform info: {}", info);
        
        // On macOS with MPS feature, MPS should be available
        #[cfg(all(target_os = "macos", feature = "mps"))]
        assert!(info.mps_available || !info.mps_feature_enabled);
        
        // On non-macOS platforms, MPS should not be available
        #[cfg(not(target_os = "macos"))]
        assert!(!info.mps_available);
    }

    #[test]
    fn test_error_recovery_fallback() {
        let strategy = FallbackStrategy::default();
        let recovery = MPSErrorRecovery::new(strategy);

        // Test device initialization failure
        let error = MPSError::DeviceInitializationFailed {
            device_name: Some("Test Device".to_string()),
            metal_error: "Mock error".to_string(),
            fallback_available: true,
        };

        let action = recovery.handle_error(error);
        assert_eq!(action, RecoveryAction::FallbackToMetal);
    }

    #[test]
    fn test_capability_validation() {
        let strategy = FallbackStrategy::default();
        let recovery = MPSErrorRecovery::new(strategy);

        let capabilities = super::super::MPSCapabilities {
            supports_neural_network: true,
            supports_matrix_multiplication: true,
            supports_convolution: false,
            supports_image_processing: true,
            supports_graph_api: false,
            max_texture_size: 8192,
            max_buffer_size: 1024 * 1024 * 1024, // 1GB
            unified_memory: true,
        };

        let requirements = CapabilityRequirements {
            neural_network_support: true,
            matrix_multiplication: true,
            convolution_support: true, // This will fail
            graph_api: false,
            minimum_memory_mb: 512,
            ane_support: false,
        };

        let result = recovery.validate_capabilities(&capabilities, &requirements, "Test Device");
        assert!(result.is_err());
        
        if let Err(MPSError::CapabilityMismatch { required_capabilities, suggested_fallback, .. }) = result {
            assert!(required_capabilities.contains(&"convolution_support".to_string()));
            assert_eq!(suggested_fallback, Some("metal".to_string()));
        } else {
            panic!("Expected CapabilityMismatch error");
        }
    }

    #[test]
    fn test_recovery_stats() {
        let strategy = FallbackStrategy::default();
        let recovery = MPSErrorRecovery::new(strategy);

        // Trigger a few errors
        let error1 = MPSError::UnavailablePlatform {
            platform: "test".to_string(),
            reason: "testing".to_string(),
        };
        let error2 = MPSError::ANEUnavailable {
            reason: "testing".to_string(),
            detection_method: "mock".to_string(),
            alternative_devices: vec!["Metal".to_string()],
        };

        recovery.handle_error(error1);
        recovery.handle_error(error2);

        let stats = recovery.get_recovery_stats().unwrap();
        assert_eq!(stats.total_errors, 2);
        assert_eq!(stats.cpu_fallback_count, 1); // First error fallback to CPU
        assert_eq!(stats.metal_fallback_count, 1); // Second error fallback to Metal
    }
}