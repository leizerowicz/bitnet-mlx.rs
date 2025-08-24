//! BitNet Benchmarks Library
//!
//! This crate provides benchmarking utilities for BitNet operations.

pub mod candle_ops;
pub mod comparison;
pub mod runner;
pub mod visualization;

pub use candle_ops::{CandleOps, CandlePerformanceUtils};
pub use comparison::{
    ComparisonConfig, ComparisonResult, PerformanceComparator, PerformanceMeasurement,
};
pub use runner::{run_cli, BenchmarkRunner};
pub use visualization::{ChartConfig, ChartTheme, PerformanceExporter, PerformanceVisualizer};

/// Placeholder function for basic functionality
pub fn placeholder() {
    println!("BitNet benchmarks library");
}
