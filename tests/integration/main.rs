//! Integration Test Runner
//!
//! This file serves as the main entry point for running integration tests.

use std::process;

// Import the integration tests module
mod tensor_integration_tests;

#[cfg(test)]
pub use tensor_integration_tests::*;

fn main() {
    println!("🧪 BitNet Tensor Integration Test Runner");
    println!("========================================");
    
    // In a real test runner, we would use a test framework
    // For now, we'll just indicate that tests should be run with `cargo test`
    
    println!("To run integration tests:");
    println!("  cargo test --test integration --features tensor-complete,integration-tests");
    println!();
    println!("To run specific integration test categories:");
    println!("  cargo test --test integration test_comprehensive_memory_pool_tensor_integration");
    println!("  cargo test --test integration test_device_abstraction_tensor_integration");
    println!("  cargo test --test integration test_performance_and_efficiency_validation");
    println!();
    println!("To run all integration tests:");
    println!("  cargo test --test integration run_all_integration_tests");
    
    process::exit(0);
}
