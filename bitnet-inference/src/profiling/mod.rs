//! Profiling module for BitNet inference performance analysis
//!
//! This module provides comprehensive profiling capabilities including:
//! - Memory usage analysis and optimization recommendations
//! - Performance benchmarking across backends
//! - Bottleneck identification and resolution

pub mod memory_profiler;

pub use memory_profiler::{
    MemoryProfiler,
    MemoryProfile,
    MemoryAnalysis,
    MemoryHotspot,
    MemoryEfficiency,
    OptimizationRecommendation,
    BackendMemoryUsage,
    AllocationStats,
    HotspotSeverity,
    RecommendationPriority,
};
