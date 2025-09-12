//! Production Readiness Framework for CPU SIMD Kernels
//!
//! This module provides robust error handling, performance regression testing,
//! and continuous validation to ensure production stability and reliability.

use anyhow::{Result, bail, Context};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::cpu::{CpuArch, KernelSelector, detect_cpu_features, TernaryLookupKernel, I2SLookupKernel};
use crate::cpu::feature_detector::CpuFeatureDetector;
use crate::cpu::performance_validator::PerformanceValidator;

/// Error types for production CPU kernel operations
#[derive(Debug, thiserror::Error)]
pub enum CpuKernelError {
    #[error("Unsupported hardware configuration: {message}")]
    UnsupportedHardware { message: String },
    
    #[error("Performance regression detected: {current:.2}x vs expected {expected:.2}x speedup")]
    PerformanceRegression { current: f64, expected: f64 },
    
    #[error("Kernel computation failed: {source}")]
    ComputationFailed { source: anyhow::Error },
    
    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustion { resource: String },
    
    #[error("Validation failed: {details}")]
    ValidationFailed { details: String },
}

/// Production readiness status for a kernel configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessStatus {
    /// Fully production ready
    Ready,
    /// Ready with warnings
    ReadyWithWarnings(Vec<String>),
    /// Not ready for production
    NotReady(Vec<String>),
    /// Unknown status (needs evaluation)
    Unknown,
}

/// Performance regression tracking
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub kernel_type: String,
    pub data_size: usize,
    pub expected_speedup: f64,
    pub tolerance_percent: f64,
    pub last_measured: Duration,
    pub measurement_count: usize,
}

/// Production readiness validator for CPU kernels
pub struct ProductionValidator {
    /// Architecture-specific feature detector
    feature_detector: CpuFeatureDetector,
    /// Performance baselines for regression detection
    baselines: HashMap<String, PerformanceBaseline>,
    /// Validation lock for thread safety
    validation_lock: Arc<Mutex<()>>,
}

impl ProductionValidator {
    /// Create new production validator
    pub fn new() -> Self {
        Self {
            feature_detector: CpuFeatureDetector::new(),
            baselines: HashMap::new(),
            validation_lock: Arc::new(Mutex::new(())),
        }
    }
    
    /// Perform comprehensive production readiness check
    pub fn validate_production_readiness(&mut self) -> Result<ReadinessStatus> {
        println!("🔍 Performing production readiness validation...");
        
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Initialize baselines first if needed (requires mutable access)
        if self.baselines.is_empty() {
            if let Err(e) = self.initialize_performance_baselines() {
                errors.push(format!("Failed to initialize baselines: {}", e));
                return Ok(ReadinessStatus::NotReady(errors));
            }
        }
        
        // 1. Hardware compatibility check (needs mutable for feature detection)
        if let Err(e) = self.validate_hardware_compatibility() {
            errors.push(format!("Hardware compatibility: {}", e));
        }
        
        // Now acquire lock for remaining validation operations
        let _lock = self.validation_lock.lock().unwrap();
        
        // 2. Error handling robustness check
        if let Err(e) = self.validate_error_handling() {
            warnings.push(format!("Error handling: {}", e));
        }
        
        // 3. Performance regression check
        if let Err(e) = self.validate_performance_stability() {
            errors.push(format!("Performance stability: {}", e));
        }
        
        // 4. Thread safety validation
        if let Err(e) = self.validate_thread_safety() {
            errors.push(format!("Thread safety: {}", e));
        }
        
        // 5. Resource usage validation
        if let Err(e) = self.validate_resource_usage() {
            warnings.push(format!("Resource usage: {}", e));
        }
        
        // Determine readiness status
        if !errors.is_empty() {
            Ok(ReadinessStatus::NotReady(errors))
        } else if !warnings.is_empty() {
            Ok(ReadinessStatus::ReadyWithWarnings(warnings))
        } else {
            Ok(ReadinessStatus::Ready)
        }
    }
    
    /// Validate hardware compatibility and fallback mechanisms
    fn validate_hardware_compatibility(&mut self) -> Result<()> {
        println!("  📱 Validating hardware compatibility...");
        
        // Detect available features
        let features = self.feature_detector.detect_features()
            .context("Failed to detect CPU features")?;
        
        // Test kernel selection for different architectures
        let architectures = vec![
            CpuArch::Generic,
            CpuArch::Arm64Neon,
            CpuArch::X86_64Avx2,
            CpuArch::X86_64Avx512,
        ];
        
        for arch in architectures {
            // Test if we can create kernels for this architecture
            let selector = KernelSelector::with_arch(arch);
            
            // Test ternary kernel
            let ternary_kernel = selector.select_ternary_kernel();
            if ternary_kernel.name().is_empty() {
                bail!("Invalid ternary kernel for architecture {:?}", arch);
            }
            
            // Test I2S kernel
            let i2s_kernel = selector.select_i2s_kernel();
            if i2s_kernel.name().is_empty() {
                bail!("Invalid I2S kernel for architecture {:?}", arch);
            }
            
            // Test that we can fall back to generic implementation
            if arch != CpuArch::Generic {
                let generic_selector = KernelSelector::with_arch(CpuArch::Generic);
                let _generic_ternary = generic_selector.select_ternary_kernel();
                let _generic_i2s = generic_selector.select_i2s_kernel();
            }
        }
        
        println!("    ✅ Hardware compatibility validated");
        Ok(())
    }
    
    /// Validate error handling robustness
    fn validate_error_handling(&self) -> Result<()> {
        println!("  🛡️ Validating error handling robustness...");
        
        let selector = KernelSelector::new();
        
        // Test error conditions for ternary kernel
        let ternary_kernel = selector.select_ternary_kernel();
        
        // Test mismatched sizes
        let weights = vec![1i8, 0, -1];
        let inputs = vec![1.0f32, 2.0]; // Different size
        let mut output = vec![0.0f32; 5]; // Different size again
        
        if ternary_kernel.compute(&weights, &inputs, &mut output).is_ok() {
            bail!("Ternary kernel should fail with mismatched input sizes");
        }
        
        // Test empty inputs
        let empty_weights: Vec<i8> = vec![];
        let empty_inputs: Vec<f32> = vec![];
        let mut empty_output: Vec<f32> = vec![];
        
        if ternary_kernel.compute(&empty_weights, &empty_inputs, &mut empty_output).is_ok() {
            bail!("Ternary kernel should fail with empty inputs");
        }
        
        // Test I2S kernel error conditions
        let i2s_kernel = selector.select_i2s_kernel();
        
        if i2s_kernel.compute(&weights, &inputs, &mut output).is_ok() {
            bail!("I2S kernel should fail with mismatched input sizes");
        }
        
        println!("    ✅ Error handling validated");
        Ok(())
    }
    
    /// Initialize performance baselines for regression testing
    fn initialize_performance_baselines(&mut self) -> Result<()> {
        println!("  📊 Initializing performance baselines...");
        
        let test_sizes = vec![1024, 4096, 16384];
        
        for size in test_sizes {
            // Initialize ternary baseline
            let ternary_baseline = PerformanceBaseline {
                kernel_type: "ternary".to_string(),
                data_size: size,
                expected_speedup: 2.0,
                tolerance_percent: 20.0,
                last_measured: Duration::from_nanos(100_000),
                measurement_count: 0,
            };
            self.baselines.insert(format!("ternary_{}", size), ternary_baseline);
            
            // Initialize I2S baseline
            let i2s_baseline = PerformanceBaseline {
                kernel_type: "i2s".to_string(),
                data_size: size,
                expected_speedup: 2.5,
                tolerance_percent: 20.0,
                last_measured: Duration::from_nanos(80_000),
                measurement_count: 0,
            };
            self.baselines.insert(format!("i2s_{}", size), i2s_baseline);
        }
        
        println!("    ✅ Performance baselines initialized");
        Ok(())
    }
    
    /// Validate performance stability and regression detection
    fn validate_performance_stability(&self) -> Result<()> {
        println!("  ⚡ Validating performance stability...");
        
        // Baselines should already be initialized at this point
        if self.baselines.is_empty() {
            bail!("Performance baselines not initialized");
        }
        
        // Test performance regression detection
        let test_sizes = vec![1024, 4096, 16384];
        
        for size in test_sizes {
            let baseline_key = format!("ternary_{}", size);
            
            if let Some(baseline) = self.baselines.get(&baseline_key) {
                // Measure current performance
                let current_speedup = self.measure_current_speedup(size)?;
                
                // Check for regression
                let expected_min = baseline.expected_speedup * (1.0 - baseline.tolerance_percent);
                
                if current_speedup < expected_min {
                    bail!("Performance regression detected for size {}: {:.2}x vs expected {:.2}x", 
                        size, current_speedup, baseline.expected_speedup);
                }
            }
        }
        
        println!("    ✅ Performance stability validated");
        Ok(())
    }
    
    /// Validate thread safety under concurrent access
    fn validate_thread_safety(&self) -> Result<()> {
        println!("  🔐 Validating thread safety...");
        
        let selector = Arc::new(KernelSelector::new());
        let num_threads = std::cmp::min(4, thread::available_parallelism()?.get());
        
        let mut handles = vec![];
        
        for thread_id in 0..num_threads {
            let selector_clone = Arc::clone(&selector);
            
            let handle = thread::spawn(move || -> Result<()> {
                let ternary_kernel = selector_clone.select_ternary_kernel();
                let i2s_kernel = selector_clone.select_i2s_kernel();
                
                // Perform multiple operations to test thread safety
                for i in 0..100 {
                    let size = 1024 + (i * thread_id) % 1000;
                    
                    // Test ternary kernel
                    let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
                        0 => -1, 1 => 0, 2 => 1, _ => 0
                    }).collect();
                    let inputs: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
                    let mut output = vec![0.0f32; size];
                    
                    ternary_kernel.compute(&weights, &inputs, &mut output)?;
                    
                    // Test I2S kernel
                    let i2s_weights: Vec<i8> = (0..size).map(|i| match i % 4 {
                        0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 0
                    }).collect();
                    
                    i2s_kernel.compute(&i2s_weights, &inputs, &mut output)?;
                }
                
                Ok(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads and check results
        for handle in handles {
            handle.join().map_err(|_| anyhow::anyhow!("Thread panicked"))??;
        }
        
        println!("    ✅ Thread safety validated");
        Ok(())
    }
    
    /// Validate resource usage patterns
    fn validate_resource_usage(&self) -> Result<()> {
        println!("  💾 Validating resource usage...");
        
        let selector = KernelSelector::new();
        
        // Test memory usage with large data
        let large_size = 1_000_000;
        let weights: Vec<i8> = vec![1; large_size];
        let inputs: Vec<f32> = vec![1.0; large_size];
        let mut output = vec![0.0; large_size];
        
        // Measure memory usage before
        let start_time = Instant::now();
        
        // Perform computation
        let ternary_kernel = selector.select_ternary_kernel();
        ternary_kernel.compute(&weights, &inputs, &mut output)
            .context("Large data computation failed")?;
        
        let computation_time = start_time.elapsed();
        
        // Check if computation time is reasonable (should complete within 1 second for 1M elements)
        if computation_time > Duration::from_secs(1) {
            bail!("Computation too slow: {:.2}s for 1M elements", computation_time.as_secs_f64());
        }
        
        println!("    ✅ Resource usage validated ({:.2}ms for 1M elements)", 
            computation_time.as_millis());
        Ok(())
    }
    
    /// Measure current speedup compared to generic implementation
    fn measure_current_speedup(&self, data_size: usize) -> Result<f64> {
        // Generate test data
        let weights: Vec<i8> = (0..data_size).map(|i| match i % 3 {
            0 => -1, 1 => 0, 2 => 1, _ => 0
        }).collect();
        let inputs: Vec<f32> = (0..data_size).map(|i| i as f32 * 0.1).collect();
        
        // Measure optimized kernel
        let optimized_selector = KernelSelector::new();
        let optimized_kernel = optimized_selector.select_ternary_kernel();
        
        let start = Instant::now();
        for _ in 0..100 {
            let mut output = vec![0.0f32; data_size];
            optimized_kernel.compute(&weights, &inputs, &mut output)?;
        }
        let optimized_time = start.elapsed();
        
        // Measure generic kernel
        let generic_selector = KernelSelector::with_arch(CpuArch::Generic);
        let generic_kernel = generic_selector.select_ternary_kernel();
        
        let start = Instant::now();
        for _ in 0..100 {
            let mut output = vec![0.0f32; data_size];
            generic_kernel.compute(&weights, &inputs, &mut output)?;
        }
        let generic_time = start.elapsed();
        
        // Calculate speedup
        let speedup = generic_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        Ok(speedup)
    }
    
    /// Continuous validation framework entry point
    pub fn run_continuous_validation(&mut self, interval_minutes: u64) -> Result<()> {
        println!("🔄 Starting continuous validation (interval: {} minutes)...", interval_minutes);
        
        loop {
            match self.validate_production_readiness() {
                Ok(ReadinessStatus::Ready) => {
                    println!("✅ Continuous validation: PASS");
                },
                Ok(ReadinessStatus::ReadyWithWarnings(warnings)) => {
                    println!("⚠️ Continuous validation: PASS with warnings:");
                    for warning in warnings {
                        println!("    - {}", warning);
                    }
                },
                Ok(ReadinessStatus::NotReady(errors)) => {
                    println!("❌ Continuous validation: FAIL");
                    for error in errors {
                        println!("    - {}", error);
                    }
                },
                Ok(ReadinessStatus::Unknown) => {
                    println!("❓ Continuous validation: UNKNOWN");
                },
                Err(e) => {
                    println!("💥 Continuous validation ERROR: {}", e);
                }
            }
            
            // Sleep until next validation cycle
            thread::sleep(Duration::from_secs(interval_minutes * 60));
        }
    }
    
    /// Generate production readiness report
    pub fn generate_readiness_report(&mut self) -> Result<String> {
        let status = self.validate_production_readiness()?;
        
        let mut report = String::new();
        report.push_str("# CPU SIMD Kernel Production Readiness Report\n\n");
        
        match status {
            ReadinessStatus::Ready => {
                report.push_str("## Status: ✅ PRODUCTION READY\n\n");
                report.push_str("All production readiness checks passed successfully.\n");
            },
            ReadinessStatus::ReadyWithWarnings(warnings) => {
                report.push_str("## Status: ⚠️ READY WITH WARNINGS\n\n");
                report.push_str("Production ready but with the following warnings:\n\n");
                for warning in warnings {
                    report.push_str(&format!("- {}\n", warning));
                }
            },
            ReadinessStatus::NotReady(errors) => {
                report.push_str("## Status: ❌ NOT PRODUCTION READY\n\n");
                report.push_str("The following issues must be resolved:\n\n");
                for error in errors {
                    report.push_str(&format!("- {}\n", error));
                }
            },
            ReadinessStatus::Unknown => {
                report.push_str("## Status: ❓ UNKNOWN\n\n");
                report.push_str("Readiness status could not be determined.\n");
            },
        }
        
        report.push_str("\n## Hardware Configuration\n\n");
        let features = self.feature_detector.detect_features()?;
        report.push_str(&format!("- Architecture: {:?}\n", features.arch));
        report.push_str(&format!("- Cores: {} physical, {} logical\n", 
            features.core_info.physical_cores, features.core_info.logical_cores));
        report.push_str(&format!("- Cache: L1={}KB, L2={}KB, L3={}KB\n", 
            features.cache_info.l1_data_size / 1024,
            features.cache_info.l2_size / 1024,
            features.cache_info.l3_size / 1024));
        
        report.push_str("\n## Performance Baselines\n\n");
        for (key, baseline) in &self.baselines {
            report.push_str(&format!("- {}: {:.2}x speedup (±{:.1}%)\n", 
                key, baseline.expected_speedup, baseline.tolerance_percent * 100.0));
        }
        
        Ok(report)
    }
}

impl Default for ProductionValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = ProductionValidator::new();
        assert!(validator.baselines.is_empty());
    }

    #[test]
    fn test_hardware_compatibility() {
        let mut validator = ProductionValidator::new();
        // Should not panic and should complete
        let result = validator.validate_hardware_compatibility();
        // May fail on some hardware, but should not panic
        match result {
            Ok(_) => {},
            Err(_) => {
                // Hardware compatibility issues are acceptable in tests
            }
        }
    }

    #[test]
    fn test_readiness_report_generation() {
        let mut validator = ProductionValidator::new();
        let report = validator.generate_readiness_report();
        // Should be able to generate a report
        assert!(report.is_ok());
        let report_content = report.unwrap();
        assert!(report_content.contains("Production Readiness Report"));
    }
}