//! BitNet Benchmarks Library
//!
//! This crate provides benchmarking utilities for BitNet operations.

pub mod candle_ops;
pub mod comparison;
pub mod runner;

pub use candle_ops::{CandleOps, CandlePerformanceUtils};
pub use comparison::{ComparisonConfig, PerformanceComparator, PerformanceMeasurement, ComparisonResult};
pub use runner::{BenchmarkRunner, run_cli};

/// Placeholder function for basic functionality
pub fn placeholder() {
    println!("BitNet benchmarks library");
}