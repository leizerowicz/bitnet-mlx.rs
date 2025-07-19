//! Comprehensive tests for execution path selection
//!
//! This test suite validates the execution backend selection logic,
//! fallback mechanisms, and error handling for different scenarios.

use bitnet_core::execution::{
    choose_execution_backend, fallback_to_candle, get_available_backends,
    get_preferred_backend, is_backend_available, ExecutionBackend, MlxError,
};
use bitnet_core::error::BitNetError;

#[test]
fn test_execution_backend_selection_for_operations() {
    // Test matrix operations
    let backend = choose_execution_backend("matmul");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu
    ));

    let backend = choose_execution_backend("matrix_multiply");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu
    ));

    // Test quantization operations
    let backend = choose_execution_backend("quantize");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleCpu
    ));

    let backend = choose_execution_backend("dequantize");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleCpu
    ));

    // Test BitNet-specific operations
    let backend = choose_execution_backend("bitlinear");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu
    ));

    // Test CPU-bound operations
    let backend = choose_execution_backend("tokenization");
    assert!(matches!(
        backend,
        ExecutionBackend::CandleCpu | ExecutionBackend::Mlx
    ));

    let backend = choose_execution_backend("preprocessing");
    assert!(matches!(
        backend,
        ExecutionBackend::CandleCpu | ExecutionBackend::Mlx
    ));

    let backend = choose_execution_backend("io");
    assert!(matches!(
        backend,
        ExecutionBackend::CandleCpu | ExecutionBackend::Mlx
    ));
}

#[test]
fn test_execution_backend_selection_for_unknown_operations() {
    // Unknown operations should still return a valid backend
    let backend = choose_execution_backend("unknown_operation");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu
    ));

    let backend = choose_execution_backend("");
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu
    ));
}

#[test]
fn test_mlx_error_types() {
    let errors = vec![
        MlxError::NotAvailable("System not supported".to_string()),
        MlxError::OperationFailed("Matrix multiplication failed".to_string()),
        MlxError::ConversionError("Type conversion failed".to_string()),
        MlxError::DeviceError("GPU initialization failed".to_string()),
        MlxError::MemoryError("Out of memory".to_string()),
        MlxError::CompilationError("Kernel compilation failed".to_string()),
        MlxError::OptimizationError("Graph optimization failed".to_string()),
        MlxError::Other("Generic error".to_string()),
    ];

    for error in errors {
        // Test error display
        let error_str = error.to_string();
        assert!(!error_str.is_empty());
        assert!(error_str.contains("MLX"));

        // Test conversion to BitNetError
        let bitnet_error: BitNetError = error.into();
        assert!(matches!(
            bitnet_error.kind(),
            bitnet_core::error::BitNetErrorKind::Mlx { .. }
        ));
    }
}

#[test]
fn test_fallback_to_candle_scenarios() {
    // Test different MLX error scenarios
    let test_cases = vec![
        MlxError::NotAvailable("MLX not supported".to_string()),
        MlxError::OperationFailed("Operation failed".to_string()),
        MlxError::ConversionError("Conversion failed".to_string()),
        MlxError::DeviceError("Device error".to_string()),
        MlxError::MemoryError("Memory error".to_string()),
        MlxError::CompilationError("Compilation error".to_string()),
        MlxError::OptimizationError("Optimization error".to_string()),
        MlxError::Other("Other error".to_string()),
    ];

    for mlx_error in test_cases {
        let result = fallback_to_candle(mlx_error);
        assert!(result.is_ok(), "Fallback should succeed for all error types");

        let tensor = result.unwrap();
        assert!(tensor.dims().len() > 0, "Fallback tensor should have valid dimensions");
        assert!(tensor.device().is_cpu(), "Fallback tensor should be on CPU");
    }
}

#[test]
fn test_backend_availability() {
    // CPU should always be available
    assert!(is_backend_available(&ExecutionBackend::CandleCpu));

    // Auto should always be available
    assert!(is_backend_available(&ExecutionBackend::Auto));

    // Test that we can query all backend types without panicking
    let _mlx_available = is_backend_available(&ExecutionBackend::Mlx);
    let _metal_available = is_backend_available(&ExecutionBackend::CandleMetal);
}

#[test]
fn test_get_available_backends() {
    let backends = get_available_backends();

    // Should always include CPU and Auto
    assert!(backends.contains(&ExecutionBackend::CandleCpu));
    assert!(backends.contains(&ExecutionBackend::Auto));

    // Should have at least 2 backends
    assert!(backends.len() >= 2);

    // All returned backends should be available
    for backend in &backends {
        assert!(is_backend_available(backend));
    }
}

#[test]
fn test_preferred_backend() {
    let backend = get_preferred_backend();

    // Preferred backend should be available
    assert!(is_backend_available(&backend));

    // Should be one of the known backend types
    assert!(matches!(
        backend,
        ExecutionBackend::Mlx | ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu | ExecutionBackend::Auto
    ));
}

#[test]
fn test_execution_backend_consistency() {
    // Test that the same operation always returns the same backend
    // (assuming system state doesn't change during test)
    let operations = vec!["matmul", "quantize", "tokenization", "unknown"];

    for operation in operations {
        let backend1 = choose_execution_backend(operation);
        let backend2 = choose_execution_backend(operation);
        assert_eq!(backend1, backend2, "Backend selection should be consistent for operation: {}", operation);
    }
}

#[test]
fn test_fallback_tensor_properties() {
    let mlx_error = MlxError::MemoryError("Out of memory".to_string());
    let result = fallback_to_candle(mlx_error);
    assert!(result.is_ok());

    let tensor = result.unwrap();

    // Verify tensor properties
    assert!(tensor.dims().len() > 0);
    assert!(tensor.device().is_cpu());
    assert_eq!(tensor.dtype(), candle_core::DType::F32);

    // Verify tensor data is accessible
    let data_result = tensor.flatten_all();
    assert!(data_result.is_ok());
}

#[test]
fn test_execution_backend_display() {
    let backends = vec![
        ExecutionBackend::Mlx,
        ExecutionBackend::CandleMetal,
        ExecutionBackend::CandleCpu,
        ExecutionBackend::Auto,
    ];

    for backend in backends {
        let display_str = backend.to_string();
        assert!(!display_str.is_empty());
        assert!(display_str.len() > 2); // Should be meaningful names
    }
}

#[test]
fn test_mlx_error_error_trait() {
    let error = MlxError::OperationFailed("Test error".to_string());
    
    // Test that MlxError implements std::error::Error
    let error_ref: &dyn std::error::Error = &error;
    assert!(!error_ref.to_string().is_empty());
}

#[cfg(feature = "mlx")]
#[test]
fn test_mlx_specific_functionality() {
    use bitnet_core::mlx::is_mlx_available;

    // Test MLX availability detection
    let mlx_available = is_mlx_available();
    
    if mlx_available {
        // If MLX is available, it should be preferred for certain operations
        let backend = choose_execution_backend("matmul");
        assert_eq!(backend, ExecutionBackend::Mlx);
    } else {
        // If MLX is not available, should fall back to other backends
        let backend = choose_execution_backend("matmul");
        assert!(matches!(backend, ExecutionBackend::CandleMetal | ExecutionBackend::CandleCpu));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_specific_functionality() {
    // Test Metal device availability on macOS
    let metal_device_result = candle_core::Device::new_metal(0);
    let metal_available = metal_device_result.is_ok();

    if metal_available {
        // Metal should be available in the backends list
        let backends = get_available_backends();
        assert!(backends.contains(&ExecutionBackend::CandleMetal));
    }

    // Test that Metal backend availability matches actual Metal device availability
    assert_eq!(is_backend_available(&ExecutionBackend::CandleMetal), metal_available);
}

#[test]
fn test_error_handling_robustness() {
    // Test that all functions handle edge cases gracefully
    
    // Test with empty operation string
    let backend = choose_execution_backend("");
    assert!(is_backend_available(&backend));

    // Test with very long operation string
    let long_operation = "a".repeat(1000);
    let backend = choose_execution_backend(&long_operation);
    assert!(is_backend_available(&backend));

    // Test with special characters
    let special_operation = "test!@#$%^&*()";
    let backend = choose_execution_backend(special_operation);
    assert!(is_backend_available(&backend));
}

#[test]
fn test_fallback_memory_efficiency() {
    // Test that fallback tensors are memory efficient
    let errors = vec![
        MlxError::MemoryError("Out of memory".to_string()),
        MlxError::DeviceError("Device error".to_string()),
    ];

    for error in errors {
        let result = fallback_to_candle(error);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        let element_count: usize = tensor.dims().iter().product();
        
        // Fallback tensors should be small to conserve memory
        assert!(element_count <= 16, "Fallback tensor should be memory efficient");
    }
}