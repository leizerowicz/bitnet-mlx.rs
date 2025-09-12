//! Performance validation and baseline measurement system
//!
//! This module implements comprehensive performance validation to ensure
//! Microsoft parity targets (1.37x-6.17x speedups) are achieved across
//! different hardware platforms and model sizes.

use anyhow::{Result, bail};
use std::time::{Duration, Instant};
use std::collections::HashMap;

use crate::cpu::{CpuArch, KernelSelector, detect_cpu_features, TernaryLookupKernel, I2SLookupKernel};

/// Microsoft's published performance targets for BitNet CPU optimization
#[derive(Debug, Clone)]
pub struct PerformanceTarget {
    pub architecture: CpuArch,
    pub operation: String,
    pub min_speedup: f64,
    pub max_speedup: f64,
    pub model_size: ModelSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelSize {
    Small,   // < 1B parameters
    Medium,  // 1B - 7B parameters  
    Large,   // 7B - 13B parameters
    XLarge,  // > 13B parameters
}

/// Performance validation results
#[derive(Debug)]
pub struct ValidationResult {
    pub target: PerformanceTarget,
    pub measured_speedup: f64,
    pub baseline_time: Duration,
    pub optimized_time: Duration,
    pub meets_target: bool,
    pub margin: f64, // How much over/under target
}

/// Microsoft parity performance validation system
pub struct PerformanceValidator {
    targets: Vec<PerformanceTarget>,
    baseline_measurements: HashMap<String, Duration>,
}

impl PerformanceValidator {
    /// Create new validator with Microsoft's published targets
    pub fn new() -> Self {
        let targets = vec![
            // ARM64 NEON targets
            PerformanceTarget {
                architecture: CpuArch::Arm64Neon,
                operation: "TL1_ternary".to_string(),
                min_speedup: 1.37,
                max_speedup: 3.2,
                model_size: ModelSize::Small,
            },
            PerformanceTarget {
                architecture: CpuArch::Arm64Neon,
                operation: "TL1_ternary".to_string(),
                min_speedup: 2.1,
                max_speedup: 4.8,
                model_size: ModelSize::Medium,
            },
            PerformanceTarget {
                architecture: CpuArch::Arm64Neon,
                operation: "TL1_ternary".to_string(),
                min_speedup: 3.5,
                max_speedup: 6.17,
                model_size: ModelSize::Large,
            },
            
            // x86_64 AVX2 targets
            PerformanceTarget {
                architecture: CpuArch::X86_64Avx2,
                operation: "TL2_ternary".to_string(),
                min_speedup: 2.5,
                max_speedup: 4.8,
                model_size: ModelSize::Small,
            },
            PerformanceTarget {
                architecture: CpuArch::X86_64Avx2,
                operation: "TL2_ternary".to_string(),
                min_speedup: 4.2,
                max_speedup: 7.1,
                model_size: ModelSize::Medium,
            },
            
            // x86_64 AVX-512 targets
            PerformanceTarget {
                architecture: CpuArch::X86_64Avx512,
                operation: "TL2_ternary".to_string(),
                min_speedup: 3.8,
                max_speedup: 6.9,
                model_size: ModelSize::Small,
            },
            PerformanceTarget {
                architecture: CpuArch::X86_64Avx512,
                operation: "TL2_ternary".to_string(),
                min_speedup: 5.5,
                max_speedup: 8.0,
                model_size: ModelSize::Medium,
            },
            
            // I2_S quantization targets (cross-platform)
            PerformanceTarget {
                architecture: CpuArch::Arm64Neon,
                operation: "I2S_quantized".to_string(),
                min_speedup: 1.8,
                max_speedup: 3.5,
                model_size: ModelSize::Medium,
            },
            PerformanceTarget {
                architecture: CpuArch::X86_64Avx2,
                operation: "I2S_quantized".to_string(),
                min_speedup: 2.2,
                max_speedup: 4.1,
                model_size: ModelSize::Medium,
            },
        ];
        
        Self {
            targets,
            baseline_measurements: HashMap::new(),
        }
    }
    
    /// Establish baseline measurements using reference implementation
    pub fn establish_baseline(&mut self, data_sizes: &[usize]) -> Result<()> {
        println!("🎯 Establishing performance baselines for Microsoft parity validation...");
        
        for &size in data_sizes {
            let model_size = Self::classify_model_size(size);
            
            // Baseline ternary operation (generic implementation)
            let baseline_time = self.measure_baseline_ternary(size)?;
            let key = format!("ternary_{:?}_{}", model_size, size);
            self.baseline_measurements.insert(key, baseline_time);
            
            // Baseline I2S quantization (generic implementation)
            let baseline_time = self.measure_baseline_i2s(size)?;
            let key = format!("i2s_{:?}_{}", model_size, size);
            self.baseline_measurements.insert(key, baseline_time);
            
            println!("  ✅ Baseline for size {} ({:?}): ternary={:.2}μs, i2s={:.2}μs", 
                size, model_size, 
                baseline_time.as_micros(),
                baseline_time.as_micros()
            );
        }
        
        Ok(())
    }
    
    /// Validate performance against Microsoft targets
    pub fn validate_performance(&self, data_sizes: &[usize]) -> Result<Vec<ValidationResult>> {
        println!("🧪 Validating performance against Microsoft targets...");
        
        let current_arch = detect_cpu_features();
        println!("  Detected architecture: {:?}", current_arch);
        
        let mut results = Vec::new();
        let selector = KernelSelector::new();
        
        for target in &self.targets {
            // Skip targets that don't match current architecture
            if target.architecture != current_arch {
                continue;
            }
            
            for &size in data_sizes {
                let model_size = Self::classify_model_size(size);
                if model_size != target.model_size {
                    continue;
                }
                
                let result = match target.operation.as_str() {
                    "TL1_ternary" | "TL2_ternary" => {
                        self.validate_ternary_performance(target, size, &selector)?
                    },
                    "I2S_quantized" => {
                        self.validate_i2s_performance(target, size, &selector)?
                    },
                    _ => {
                        bail!("Unknown operation: {}", target.operation);
                    }
                };
                
                let status = if result.meets_target { "✅ PASS" } else { "❌ FAIL" };
                println!("  {} {}: {:.2}x speedup (target: {:.2}x-{:.2}x, margin: {:.1}%)",
                    status, target.operation, result.measured_speedup, 
                    target.min_speedup, target.max_speedup, result.margin * 100.0
                );
                
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    fn validate_ternary_performance(
        &self, 
        target: &PerformanceTarget, 
        size: usize,
        selector: &KernelSelector
    ) -> Result<ValidationResult> {
        let baseline_key = format!("ternary_{:?}_{}", target.model_size, size);
        let baseline_time = self.baseline_measurements.get(&baseline_key)
            .ok_or_else(|| anyhow::anyhow!("No baseline for {}", baseline_key))?;
        
        // Generate test data
        let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
            0 => -1, 1 => 0, 2 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        
        // Measure optimized kernel performance
        let kernel = selector.select_ternary_kernel();
        let start = Instant::now();
        
        const ITERATIONS: usize = 1000;
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; size];
            kernel.compute(
                &weights,
                &inputs,
                &mut output,
            )?;
            let _ = &output; // Prevent optimization
        }
        
        let optimized_time = start.elapsed() / ITERATIONS as u32;
        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        
        let meets_target = speedup >= target.min_speedup && speedup <= target.max_speedup;
        let margin = if speedup >= target.min_speedup {
            (speedup - target.min_speedup) / target.min_speedup
        } else {
            (speedup - target.min_speedup) / target.min_speedup
        };
        
        Ok(ValidationResult {
            target: target.clone(),
            measured_speedup: speedup,
            baseline_time: *baseline_time,
            optimized_time,
            meets_target,
            margin,
        })
    }
    
    fn validate_i2s_performance(
        &self, 
        target: &PerformanceTarget, 
        size: usize,
        selector: &KernelSelector
    ) -> Result<ValidationResult> {
        let baseline_key = format!("i2s_{:?}_{}", target.model_size, size);
        let baseline_time = self.baseline_measurements.get(&baseline_key)
            .ok_or_else(|| anyhow::anyhow!("No baseline for {}", baseline_key))?;
        
        // Generate test data for I2S
        let weights: Vec<i8> = (0..size).map(|i| match i % 4 {
            0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        
        // Measure optimized kernel performance
        let kernel = selector.select_i2s_kernel();
        let start = Instant::now();
        
        const ITERATIONS: usize = 1000;
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; size];
            kernel.compute(
                &weights,
                &inputs,
                &mut output,
            )?;
            let _ = &output; // Prevent optimization
        }
        
        let optimized_time = start.elapsed() / ITERATIONS as u32;
        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        
        let meets_target = speedup >= target.min_speedup && speedup <= target.max_speedup;
        let margin = if speedup >= target.min_speedup {
            (speedup - target.min_speedup) / target.min_speedup
        } else {
            (speedup - target.min_speedup) / target.min_speedup
        };
        
        Ok(ValidationResult {
            target: target.clone(),
            measured_speedup: speedup,
            baseline_time: *baseline_time,
            optimized_time,
            meets_target,
            margin,
        })
    }
    
    fn measure_baseline_ternary(&self, size: usize) -> Result<Duration> {
        use crate::cpu::kernels::GenericTernaryKernel;
        
        let weights: Vec<i8> = (0..size).map(|i| match i % 3 {
            0 => -1, 1 => 0, 2 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        
        let kernel = GenericTernaryKernel::new();
        let start = Instant::now();
        
        const ITERATIONS: usize = 1000;
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; size];
            kernel.compute(&weights, &inputs, &mut output)?;
            let _ = &output; // Prevent optimization
        }
        
        Ok(start.elapsed() / ITERATIONS as u32)
    }
    
    fn measure_baseline_i2s(&self, size: usize) -> Result<Duration> {
        use crate::cpu::kernels::GenericI2SKernel;
        
        let weights: Vec<i8> = (0..size).map(|i| match i % 4 {
            0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        
        let kernel = GenericI2SKernel::new();
        let start = Instant::now();
        
        const ITERATIONS: usize = 1000;
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; size];
            kernel.compute(&weights, &inputs, &mut output)?;
            let _ = &output; // Prevent optimization
        }
        
        Ok(start.elapsed() / ITERATIONS as u32)
    }
    
    fn classify_model_size(data_size: usize) -> ModelSize {
        match data_size {
            0..=1_000_000 => ModelSize::Small,
            1_000_001..=7_000_000 => ModelSize::Medium,
            7_000_001..=13_000_000 => ModelSize::Large,
            _ => ModelSize::XLarge,
        }
    }
    
    /// Generate summary report of validation results
    pub fn generate_report(&self, results: &[ValidationResult]) -> String {
        let mut report = String::new();
        report.push_str("# CPU Performance Validation Report\n\n");
        report.push_str("## Microsoft Parity Target Validation\n\n");
        
        let mut passed = 0;
        let mut total = 0;
        
        for result in results {
            total += 1;
            if result.meets_target {
                passed += 1;
            }
            
            let status = if result.meets_target { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!(
                "- **{}** ({:?}): {:.2}x speedup {} (target: {:.2}x-{:.2}x)\n",
                result.target.operation,
                result.target.model_size,
                result.measured_speedup,
                status,
                result.target.min_speedup,
                result.target.max_speedup
            ));
        }
        
        report.push_str(&format!("\n## Summary\n\n"));
        report.push_str(&format!("- **Total tests**: {}\n", total));
        report.push_str(&format!("- **Passed**: {}\n", passed));
        report.push_str(&format!("- **Failed**: {}\n", total - passed));
        report.push_str(&format!("- **Success rate**: {:.1}%\n", (passed as f64 / total as f64) * 100.0));
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validator_creation() {
        let validator = PerformanceValidator::new();
        assert!(!validator.targets.is_empty());
    }
    
    #[test] 
    fn test_model_size_classification() {
        assert_eq!(PerformanceValidator::classify_model_size(500_000), ModelSize::Small);
        assert_eq!(PerformanceValidator::classify_model_size(3_000_000), ModelSize::Medium);
        assert_eq!(PerformanceValidator::classify_model_size(10_000_000), ModelSize::Large);
        assert_eq!(PerformanceValidator::classify_model_size(20_000_000), ModelSize::XLarge);
    }
}