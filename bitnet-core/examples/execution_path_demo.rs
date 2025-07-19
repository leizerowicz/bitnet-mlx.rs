//! Execution Path Selection Demo
//!
//! This example demonstrates how to use the execution path selection
//! functionality to choose optimal backends and handle fallback scenarios.

use bitnet_core::execution::{
    choose_execution_backend, fallback_to_candle, get_available_backends,
    get_preferred_backend, is_backend_available, ExecutionBackend, MlxError,
};

fn main() -> anyhow::Result<()> {
    println!("🚀 BitNet Execution Path Selection Demo");
    println!("========================================\n");

    // 1. Show available backends on this system
    println!("📋 Available Execution Backends:");
    let available_backends = get_available_backends();
    for backend in &available_backends {
        let available = if is_backend_available(backend) { "✅" } else { "❌" };
        println!("  {} {}", available, backend);
    }
    println!();

    // 2. Show preferred backend
    let preferred = get_preferred_backend();
    println!("⭐ Preferred Backend: {}\n", preferred);

    // 3. Demonstrate backend selection for different operations
    println!("🎯 Backend Selection for Different Operations:");
    let operations = vec![
        "matmul",
        "quantize", 
        "bitlinear",
        "attention",
        "conv2d",
        "tokenization",
        "preprocessing",
        "unknown_operation",
    ];

    for operation in operations {
        let backend = choose_execution_backend(operation);
        println!("  {} -> {}", operation.pad_to_width(15), backend);
    }
    println!();

    // 4. Demonstrate MLX error handling and fallback
    println!("🔄 MLX Error Handling and Fallback:");
    let mlx_errors = vec![
        MlxError::NotAvailable("MLX not supported on this system".to_string()),
        MlxError::OperationFailed("Matrix multiplication failed".to_string()),
        MlxError::MemoryError("Out of GPU memory".to_string()),
        MlxError::DeviceError("GPU initialization failed".to_string()),
    ];

    for (i, error) in mlx_errors.into_iter().enumerate() {
        println!("  Error {}: {}", i + 1, error);
        
        match fallback_to_candle(error) {
            Ok(tensor) => {
                println!("    ✅ Fallback successful: tensor shape {:?}", tensor.dims());
            }
            Err(e) => {
                println!("    ❌ Fallback failed: {}", e);
            }
        }
    }
    println!();

    // 5. Demonstrate backend availability checking
    println!("🔍 Backend Availability Details:");
    let all_backends = vec![
        ExecutionBackend::Mlx,
        ExecutionBackend::CandleMetal,
        ExecutionBackend::CandleCpu,
        ExecutionBackend::Auto,
    ];

    for backend in all_backends {
        let available = is_backend_available(&backend);
        let status = if available { "Available" } else { "Not Available" };
        let icon = if available { "✅" } else { "❌" };
        
        println!("  {} {} - {}", icon, backend, status);
        
        // Show additional info for specific backends
        match backend {
            ExecutionBackend::Mlx => {
                #[cfg(feature = "mlx")]
                {
                    use bitnet_core::mlx::is_mlx_available;
                    if is_mlx_available() {
                        println!("    📱 Running on Apple Silicon with MLX support");
                    } else {
                        println!("    💻 MLX not available (not Apple Silicon or MLX not installed)");
                    }
                }
                #[cfg(not(feature = "mlx"))]
                {
                    println!("    🚫 MLX feature not compiled in");
                }
            }
            ExecutionBackend::CandleMetal => {
                #[cfg(target_os = "macos")]
                {
                    if candle_core::Device::new_metal(0).is_ok() {
                        println!("    🖥️  Metal GPU acceleration available");
                    } else {
                        println!("    💻 Metal GPU not available");
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    println!("    🚫 Metal only available on macOS");
                }
            }
            ExecutionBackend::CandleCpu => {
                println!("    💻 CPU backend always available");
            }
            ExecutionBackend::Auto => {
                println!("    🤖 Automatic selection always available");
            }
        }
    }
    println!();

    // 6. Performance recommendations
    println!("💡 Performance Recommendations:");
    println!("  • For matrix operations: Use MLX on Apple Silicon, Metal on macOS, CPU elsewhere");
    println!("  • For quantization: MLX preferred, CPU fallback");
    println!("  • For I/O operations: CPU backend recommended");
    println!("  • For unknown operations: Auto selection chooses optimal backend");
    println!();

    println!("✨ Demo completed successfully!");
    
    Ok(())
}

// Helper trait for string padding (simple implementation)
trait PadString {
    fn pad_to_width(&self, width: usize) -> String;
}

impl PadString for &str {
    fn pad_to_width(&self, width: usize) -> String {
        format!("{:width$}", self, width = width)
    }
}