//! Comprehensive Error Formatting Tests
//!
//! This module tests the enhanced error handling and formatting capabilities,
//! ensuring that errors provide detailed context and useful debugging information.

use bitnet_core::error::{
    BitNetError, BitNetErrorKind, ErrorContext, ErrorContextBuilder, ErrorSeverity, ErrorFormatter, ToBitNetError
};

#[test]
fn test_device_error_creation_and_formatting() {
    let error = BitNetError::device_error("Metal device initialization failed", "Metal")
        .add_context("device_id", "0")
        .add_context("driver_version", "14.2.1")
        .set_severity(ErrorSeverity::Critical);
    
    assert!(error.is_device_error());
    assert!(error.is_critical());
    assert_eq!(error.device_info(), Some("Metal"));
    assert_eq!(error.severity(), ErrorSeverity::Critical);
    
    let report = error.to_report();
    assert_eq!(report.category, "Device");
    assert_eq!(report.severity, ErrorSeverity::Critical);
    assert!(!report.suggestions.is_empty());
    
    let formatted = report.format_human_readable();
    assert!(formatted.contains("CRITICAL"));
    assert!(formatted.contains("Metal"));
    assert!(formatted.contains("device_id=0"));
    assert!(formatted.contains("driver_version=14.2.1"));
}

#[test]
fn test_memory_error_with_size_information() {
    let error = BitNetError::memory_error_with_size(
        "Failed to allocate GPU buffer", 
        1024 * 1024 * 1024, // 1GB
        "gpu_allocation"
    )
    .add_context("available_memory", "512MB")
    .add_context("fragmentation", "high")
    .set_severity(ErrorSeverity::Error);
    
    assert!(error.is_memory_error());
    assert!(!error.is_critical());
    
    let memory_info = error.memory_info();
    assert_eq!(memory_info, Some((Some(1024 * 1024 * 1024), "gpu_allocation")));
    
    let report = error.to_report();
    assert!(report.suggestions.iter().any(|s| s.contains("streaming")));
    
    let compact = report.format_compact();
    assert!(compact.contains("Memory"));
    assert!(compact.contains("Failed to allocate GPU buffer"));
}

#[test]
fn test_tensor_error_with_shape_and_dtype() {
    let error = BitNetError::tensor_error_with_info(
        "Shape mismatch in matrix multiplication",
        Some(vec![2, 3, 4]),
        Some("f32".to_string())
    )
    .add_context("expected_shape", "[2, 4, 3]")
    .add_context("operation", "matmul")
    .set_severity(ErrorSeverity::Warning);
    
    let report = error.to_report();
    assert_eq!(report.category, "Tensor");
    assert_eq!(report.severity, ErrorSeverity::Warning);
    
    let json_result = report.format_json();
    assert!(json_result.is_ok());
    
    let json_str = json_result.unwrap();
    assert!(json_str.contains("tensor"));
    assert!(json_str.contains("Shape mismatch"));
    assert!(json_str.contains("expected_shape"));
}

#[test]
fn test_conversion_error_formatting() {
    let error = BitNetError::conversion_error(
        "Lossy conversion detected",
        "f64",
        "i8"
    )
    .add_context("precision_loss", "significant")
    .add_context("range_overflow", "true")
    .set_severity(ErrorSeverity::Warning);
    
    if let BitNetErrorKind::Conversion { from_type, to_type, .. } = error.kind() {
        assert_eq!(from_type, "f64");
        assert_eq!(to_type, "i8");
    } else {
        panic!("Expected conversion error");
    }
    
    let report = error.to_report();
    assert!(report.suggestions.iter().any(|s| s.contains("intermediate conversion")));
    
    let formatter = ErrorFormatter::minimal();
    let minimal_output = formatter.format(&error);
    assert!(minimal_output.len() < 200); // Should be compact
}

#[test]
fn test_metal_error_with_device_info() {
    let error = BitNetError::metal_error_with_device(
        "Compute pipeline creation failed",
        "shader_compilation",
        "Apple M1 Pro"
    )
    .add_context("shader_name", "bitlinear_forward")
    .add_context("error_code", "MTL_ERROR_INVALID_FUNCTION")
    .set_severity(ErrorSeverity::Error);
    
    assert!(error.is_metal_error());
    assert_eq!(error.device_info(), Some("Apple M1 Pro"));
    
    let report = error.to_report();
    assert_eq!(report.technical_details.component, "Metal GPU");
    assert_eq!(report.technical_details.operation, "shader_compilation");
    
    let formatted = report.format_human_readable();
    assert!(formatted.contains("Metal"));
    assert!(formatted.contains("Apple M1 Pro"));
    assert!(formatted.contains("shader_name=bitlinear_forward"));
}

#[test]
fn test_mlx_error_formatting() {
    let error = BitNetError::mlx_error(
        "MLX array operation failed",
        "matrix_multiplication"
    )
    .add_context("array_shape_a", "[1024, 512]")
    .add_context("array_shape_b", "[256, 1024]")
    .add_context("mlx_version", "0.1.0")
    .set_severity(ErrorSeverity::Error);
    
    let report = error.to_report();
    assert_eq!(report.category, "MLX");
    assert!(report.suggestions.iter().any(|s| s.contains("Apple Silicon")));
    
    let technical_details = &report.technical_details;
    assert_eq!(technical_details.component, "MLX Framework");
    assert_eq!(technical_details.operation, "matrix_multiplication");
}

#[test]
fn test_configuration_error_with_parameter() {
    let error = BitNetError::config_error_with_param(
        "Invalid batch size configuration",
        "batch_size"
    )
    .add_context("provided_value", "0")
    .add_context("valid_range", "1-1024")
    .set_severity(ErrorSeverity::Warning);
    
    let report = error.to_report();
    assert_eq!(report.category, "Configuration");
    assert!(report.suggestions.iter().any(|s| s.contains("batch_size")));
}

#[test]
fn test_io_error_with_path() {
    let error = BitNetError::io_error_with_path(
        "Failed to read model weights",
        "/path/to/model.safetensors"
    )
    .add_context("file_size", "2.1GB")
    .add_context("permissions", "read-only")
    .set_severity(ErrorSeverity::Error);
    
    let report = error.to_report();
    assert_eq!(report.category, "I/O");
    assert!(report.suggestions.iter().any(|s| s.contains("/path/to/model.safetensors")));
}

#[test]
fn test_concurrency_error_formatting() {
    let error = BitNetError::concurrency_error(
        "Deadlock detected in memory pool",
        "gpu_memory_pool"
    )
    .add_context("thread_count", "4")
    .add_context("lock_timeout", "5000ms")
    .set_severity(ErrorSeverity::Critical);
    
    let report = error.to_report();
    assert_eq!(report.category, "Concurrency");
    assert!(report.suggestions.iter().any(|s| s.contains("gpu_memory_pool")));
    
    let formatted = report.format_human_readable();
    assert!(formatted.contains("CRITICAL"));
    assert!(formatted.contains("Deadlock"));
}

#[test]
fn test_internal_error_formatting() {
    let error = BitNetError::internal_error(
        "Unexpected state in tensor lifecycle",
        "tensor_manager"
    )
    .add_context("tensor_id", "12345")
    .add_context("state", "invalid")
    .set_severity(ErrorSeverity::Fatal);
    
    let report = error.to_report();
    assert_eq!(report.category, "Internal");
    assert!(report.suggestions.iter().any(|s| s.contains("internal error")));
    
    let technical_details = &report.technical_details;
    assert_eq!(technical_details.component, "tensor_manager");
}

#[test]
fn test_error_context_builder() {
    let context = ErrorContextBuilder::new()
        .operation("tensor_creation")
        .description("Creating large tensor for training")
        .add("device", "Metal")
        .add("dtype", "f16")
        .add("shape", "[8, 1024, 1024]")
        .source_location("tensor.rs", 42)
        .build();
    
    assert_eq!(context.operation, Some("tensor_creation".to_string()));
    assert_eq!(context.description, Some("Creating large tensor for training".to_string()));
    assert_eq!(context.get("device"), Some(&"Metal".to_string()));
    assert!(context.source_location.is_some());
    
    let display_str = format!("{context}");
    assert!(display_str.contains("operation=tensor_creation"));
    assert!(display_str.contains("device=Metal"));
    assert!(display_str.contains("location=tensor.rs:42"));
}

#[test]
fn test_error_context_merging() {
    let mut context1 = ErrorContext::new();
    context1.add("key1", "value1");
    context1.set_operation("op1");
    
    let mut context2 = ErrorContext::new();
    context2.add("key2", "value2");
    context2.set_description("desc2");
    
    context1.merge(&context2);
    
    assert_eq!(context1.get("key1"), Some(&"value1".to_string()));
    assert_eq!(context1.get("key2"), Some(&"value2".to_string()));
    assert_eq!(context1.operation, Some("op1".to_string()));
    assert_eq!(context1.description, Some("desc2".to_string()));
}

#[test]
fn test_error_severity_ordering() {
    assert!(ErrorSeverity::Fatal.is_at_least(ErrorSeverity::Critical));
    assert!(ErrorSeverity::Critical.is_at_least(ErrorSeverity::Error));
    assert!(ErrorSeverity::Error.is_at_least(ErrorSeverity::Warning));
    assert!(ErrorSeverity::Warning.is_at_least(ErrorSeverity::Info));
    
    assert!(!ErrorSeverity::Info.is_at_least(ErrorSeverity::Warning));
    assert!(!ErrorSeverity::Warning.is_at_least(ErrorSeverity::Error));
}

#[test]
fn test_error_formatter_customization() {
    let error = BitNetError::device_error("Test error", "CPU")
        .add_context("test_key", "test_value");
    
    let full_formatter = ErrorFormatter::new();
    let full_output = full_formatter.format(&error);
    
    let minimal_formatter = ErrorFormatter::minimal();
    let minimal_output = minimal_formatter.format(&error);
    
    assert!(full_output.len() > minimal_output.len());
    assert!(full_output.contains("test_key=test_value"));
    assert!(!minimal_output.contains("test_key=test_value"));
}

#[test]
fn test_to_bitnet_error_trait() {
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let result: Result<String, std::io::Error> = Err(io_error);
    
    let bitnet_result = result.with_context("Failed to load configuration");
    assert!(bitnet_result.is_err());
    
    let error = bitnet_result.unwrap_err();
    assert!(!error.context().is_empty());
    assert!(error.context().get("context").is_some());
    assert!(error.context().get("original_error").is_some());
}

#[test]
fn test_error_id_uniqueness() {
    let error1 = BitNetError::device_error("Error 1", "CPU");
    let error2 = BitNetError::device_error("Error 2", "Metal");
    let error3 = BitNetError::memory_error("Error 3", "allocation");
    
    let ids = vec![error1.error_id(), error2.error_id(), error3.error_id()];
    let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
    
    assert_eq!(ids.len(), unique_ids.len()); // All IDs should be unique
    
    // Check ID format (should start with "BN")
    for id in &ids {
        assert!(id.starts_with("BN"));
        assert!(id.len() >= 10); // Should have reasonable length
    }
}

#[test]
fn test_error_report_json_serialization() {
    let error = BitNetError::tensor_error("Test tensor error")
        .add_context("shape", "[2, 3, 4]")
        .add_context("dtype", "f32")
        .set_severity(ErrorSeverity::Warning);
    
    let report = error.to_report();
    let json_result = report.format_json();
    
    assert!(json_result.is_ok());
    let json_str = json_result.unwrap();
    
    // Verify JSON contains expected fields
    assert!(json_str.contains("error_id"));
    assert!(json_str.contains("severity"));
    assert!(json_str.contains("category"));
    assert!(json_str.contains("message"));
    assert!(json_str.contains("context"));
    assert!(json_str.contains("suggestions"));
    assert!(json_str.contains("technical_details"));
    
    // Verify it can be parsed back
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["category"], "Tensor");
    assert_eq!(parsed["severity"], "Warning");
}

#[test]
fn test_error_chaining_and_context_propagation() {
    // Simulate a chain of errors with context propagation
    let original_error = BitNetError::io_error("Failed to read file")
        .add_context("file_path", "/tmp/data.bin")
        .add_context("operation", "file_read");
    
    let wrapped_error = BitNetError::tensor_error("Failed to load tensor data")
        .add_context("tensor_name", "weights")
        .add_context("source_error", original_error.error_id());
    
    // Verify both errors have their own contexts
    assert!(original_error.context().get("file_path").is_some());
    assert!(wrapped_error.context().get("tensor_name").is_some());
    assert!(wrapped_error.context().get("source_error").is_some());
    
    // Verify error IDs are different
    assert_ne!(original_error.error_id(), wrapped_error.error_id());
}

#[test]
fn test_performance_metrics_in_error_report() {
    let error = BitNetError::memory_error_with_size(
        "Memory allocation failed under high load",
        512 * 1024 * 1024, // 512MB
        "batch_processing"
    )
    .add_context("active_tensors", "150")
    .add_context("memory_pressure", "high")
    .set_severity(ErrorSeverity::Critical);
    
    let report = error.to_report();
    
    // Verify system info is captured
    assert!(!report.technical_details.system_info.os.is_empty());
    assert!(!report.technical_details.system_info.arch.is_empty());
    
    // Verify suggestions are relevant to memory issues
    assert!(report.suggestions.iter().any(|s| s.contains("memory")));
    assert!(report.suggestions.iter().any(|s| s.contains("batch size") || s.contains("dimensions")));
}

#[test]
fn test_error_display_formatting() {
    let error = BitNetError::device_error("Metal device not found", "Metal")
        .add_context("requested_device_id", "0")
        .add_context("available_devices", "none")
        .set_severity(ErrorSeverity::Error);
    
    let display_str = format!("{error}");
    
    assert!(display_str.contains("ERROR"));
    assert!(display_str.contains("Metal device not found"));
    assert!(display_str.contains("Context:"));
    assert!(display_str.contains("requested_device_id=0"));
    assert!(display_str.starts_with("[ERROR]"));
}

#[test]
fn test_cross_platform_error_handling() {
    // Test that errors work consistently across platforms
    let error = BitNetError::metal_error("Platform-specific error", "gpu_operation")
        .add_context("platform", std::env::consts::OS)
        .add_context("arch", std::env::consts::ARCH);
    
    let report = error.to_report();
    
    // Should work on all platforms
    assert_eq!(report.category, "Metal");
    assert!(!report.technical_details.system_info.os.is_empty());
    assert!(!report.technical_details.system_info.arch.is_empty());
    
    // Platform-specific suggestions should be included
    #[cfg(target_os = "macos")]
    {
        assert!(report.suggestions.iter().any(|s| s.contains("Metal")));
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        assert!(report.suggestions.iter().any(|s| s.contains("CPU")));
    }
}