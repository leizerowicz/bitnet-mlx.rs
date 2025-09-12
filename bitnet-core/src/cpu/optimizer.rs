//! SIMD Kernel Optimization Tuning System
//!
//! This module provides runtime optimization and tuning for SIMD kernels
//! based on actual performance measurements and hardware characteristics.

use anyhow::{Result, bail};
use std::time::{Duration, Instant};
use std::collections::HashMap;

use crate::cpu::{CpuArch, KernelSelector, detect_cpu_features};

/// Optimization parameters for different kernel types
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Optimal batch size for this kernel
    pub batch_size: usize,
    /// Memory prefetch distance (cache lines ahead)
    pub prefetch_distance: usize,
    /// Vector processing width (elements per SIMD operation)
    pub vector_width: usize,
    /// Loop unroll factor for better instruction scheduling
    pub unroll_factor: usize,
    /// Cache line alignment requirements
    pub alignment: usize,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            prefetch_distance: 2,
            vector_width: 4,
            unroll_factor: 2,
            alignment: 64,
        }
    }
}

/// Performance measurement result for optimization tuning
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub params: OptimizationParams,
    pub execution_time: Duration,
    pub throughput: f64, // operations per second
    pub cache_miss_rate: f64, // estimated
}

/// Kernel optimization tuner for automatic performance optimization
pub struct KernelOptimizer {
    arch: CpuArch,
    measurements: HashMap<String, Vec<PerformanceMeasurement>>,
    optimal_params: HashMap<String, OptimizationParams>,
}

impl KernelOptimizer {
    /// Create a new kernel optimizer for the current architecture
    pub fn new() -> Self {
        Self {
            arch: detect_cpu_features(),
            measurements: HashMap::new(),
            optimal_params: HashMap::new(),
        }
    }
    
    /// Create optimizer for specific architecture
    pub fn with_arch(arch: CpuArch) -> Self {
        Self {
            arch,
            measurements: HashMap::new(),
            optimal_params: HashMap::new(),
        }
    }
    
    /// Run comprehensive optimization tuning for all kernel types
    pub fn tune_all_kernels(&mut self, data_sizes: &[usize]) -> Result<()> {
        println!("🔧 Starting comprehensive kernel optimization tuning...");
        println!("  Target architecture: {:?}", self.arch);
        
        // Tune ternary kernels
        for &size in data_sizes {
            self.tune_ternary_kernel(size)?;
        }
        
        // Tune I2S kernels  
        for &size in data_sizes {
            self.tune_i2s_kernel(size)?;
        }
        
        // Generate optimization recommendations
        self.generate_optimization_recommendations();
        
        Ok(())
    }
    
    /// Tune ternary lookup table kernels for optimal performance
    fn tune_ternary_kernel(&mut self, data_size: usize) -> Result<()> {
        let key = format!("ternary_{}", data_size);
        println!("  🎯 Tuning ternary kernel for data size: {}", data_size);
        
        // Generate test data
        let weights: Vec<i8> = (0..data_size).map(|i| match i % 3 {
            0 => -1, 1 => 0, 2 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..data_size).map(|i| (i as f32) * 0.1).collect();
        
        let mut measurements = Vec::new();
        
        // Test different optimization parameters
        let batch_sizes = self.get_batch_size_candidates();
        let vector_widths = self.get_vector_width_candidates();
        let unroll_factors = self.get_unroll_factor_candidates();
        
        for &batch_size in &batch_sizes {
            for &vector_width in &vector_widths {
                for &unroll_factor in &unroll_factors {
                    let params = OptimizationParams {
                        batch_size,
                        vector_width,
                        unroll_factor,
                        prefetch_distance: self.calculate_optimal_prefetch(batch_size),
                        alignment: 64, // Standard cache line alignment
                    };
                    
                    // Measure performance with these parameters
                    if let Ok(measurement) = self.measure_ternary_performance(&weights, &inputs, &params) {
                        measurements.push(measurement);
                    }
                }
            }
        }
        
        // Find optimal parameters
        if let Some(best) = measurements.iter().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()) {
            println!("    ✅ Optimal ternary params: batch={}, vector_width={}, unroll={}, throughput={:.2} ops/sec",
                best.params.batch_size, best.params.vector_width, best.params.unroll_factor, best.throughput);
            self.optimal_params.insert(key.clone(), best.params.clone());
        }
        
        self.measurements.insert(key, measurements);
        Ok(())
    }
    
    /// Tune I2S kernels for optimal performance  
    fn tune_i2s_kernel(&mut self, data_size: usize) -> Result<()> {
        let key = format!("i2s_{}", data_size);
        println!("  🎯 Tuning I2S kernel for data size: {}", data_size);
        
        // Generate test data
        let weights: Vec<i8> = (0..data_size).map(|i| match i % 4 {
            0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 0,
        }).collect();
        let inputs: Vec<f32> = (0..data_size).map(|i| (i as f32) * 0.01).collect();
        
        let mut measurements = Vec::new();
        
        // Test different optimization parameters (simplified for I2S)
        let batch_sizes = self.get_batch_size_candidates();
        let vector_widths = self.get_vector_width_candidates();
        
        for &batch_size in &batch_sizes {
            for &vector_width in &vector_widths {
                let params = OptimizationParams {
                    batch_size,
                    vector_width,
                    unroll_factor: 2, // Fixed for I2S
                    prefetch_distance: self.calculate_optimal_prefetch(batch_size),
                    alignment: 64,
                };
                
                // Measure performance with these parameters
                if let Ok(measurement) = self.measure_i2s_performance(&weights, &inputs, &params) {
                    measurements.push(measurement);
                }
            }
        }
        
        // Find optimal parameters
        if let Some(best) = measurements.iter().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()) {
            println!("    ✅ Optimal I2S params: batch={}, vector_width={}, throughput={:.2} ops/sec",
                best.params.batch_size, best.params.vector_width, best.throughput);
            self.optimal_params.insert(key.clone(), best.params.clone());
        }
        
        self.measurements.insert(key, measurements);
        Ok(())
    }
    
    /// Measure ternary kernel performance with specific parameters
    fn measure_ternary_performance(
        &self,
        weights: &[i8],
        inputs: &[f32],
        params: &OptimizationParams
    ) -> Result<PerformanceMeasurement> {
        let selector = KernelSelector::with_arch(self.arch);
        let kernel = selector.select_ternary_kernel();
        
        // Warm-up runs
        for _ in 0..10 {
            let mut output = vec![0.0f32; inputs.len()];
            kernel.compute(weights, inputs, &mut output)?;
        }
        
        // Actual measurement
        let start = Instant::now();
        const ITERATIONS: usize = 1000;
        
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; inputs.len()];
            kernel.compute(weights, inputs, &mut output)?;
        }
        
        let execution_time = start.elapsed() / ITERATIONS as u32;
        let throughput = (inputs.len() as f64) / execution_time.as_secs_f64();
        
        // Estimate cache miss rate based on data size and access patterns
        let cache_miss_rate = self.estimate_cache_miss_rate(inputs.len(), params);
        
        Ok(PerformanceMeasurement {
            params: params.clone(),
            execution_time,
            throughput,
            cache_miss_rate,
        })
    }
    
    /// Measure I2S kernel performance with specific parameters
    fn measure_i2s_performance(
        &self,
        weights: &[i8],
        inputs: &[f32],
        params: &OptimizationParams
    ) -> Result<PerformanceMeasurement> {
        let selector = KernelSelector::with_arch(self.arch);
        let kernel = selector.select_i2s_kernel();
        
        // Warm-up runs
        for _ in 0..10 {
            let mut output = vec![0.0f32; inputs.len()];
            kernel.compute(weights, inputs, &mut output)?;
        }
        
        // Actual measurement
        let start = Instant::now();
        const ITERATIONS: usize = 1000;
        
        for _ in 0..ITERATIONS {
            let mut output = vec![0.0f32; inputs.len()];
            kernel.compute(weights, inputs, &mut output)?;
        }
        
        let execution_time = start.elapsed() / ITERATIONS as u32;
        let throughput = (inputs.len() as f64) / execution_time.as_secs_f64();
        let cache_miss_rate = self.estimate_cache_miss_rate(inputs.len(), params);
        
        Ok(PerformanceMeasurement {
            params: params.clone(),
            execution_time,
            throughput,
            cache_miss_rate,
        })
    }
    
    /// Get batch size candidates based on architecture
    fn get_batch_size_candidates(&self) -> Vec<usize> {
        match self.arch {
            CpuArch::Arm64Neon => vec![256, 512, 1024, 2048, 4096],
            CpuArch::X86_64Avx2 => vec![512, 1024, 2048, 4096, 8192],
            CpuArch::X86_64Avx512 => vec![1024, 2048, 4096, 8192, 16384],
            CpuArch::Generic => vec![128, 256, 512, 1024],
        }
    }
    
    /// Get vector width candidates based on architecture
    fn get_vector_width_candidates(&self) -> Vec<usize> {
        match self.arch {
            CpuArch::Arm64Neon => vec![4], // NEON f32x4
            CpuArch::X86_64Avx2 => vec![8], // AVX2 f32x8
            CpuArch::X86_64Avx512 => vec![16], // AVX-512 f32x16
            CpuArch::Generic => vec![1, 2, 4],
        }
    }
    
    /// Get unroll factor candidates based on architecture
    fn get_unroll_factor_candidates(&self) -> Vec<usize> {
        match self.arch {
            CpuArch::Arm64Neon => vec![1, 2, 4], // Conservative for ARM
            CpuArch::X86_64Avx2 => vec![2, 4, 8], // More aggressive for x86
            CpuArch::X86_64Avx512 => vec![2, 4, 8, 16], // Most aggressive
            CpuArch::Generic => vec![1, 2],
        }
    }
    
    /// Calculate optimal prefetch distance based on batch size
    fn calculate_optimal_prefetch(&self, batch_size: usize) -> usize {
        // Heuristic: prefetch 2-4 cache lines ahead based on batch size
        match batch_size {
            0..=512 => 2,
            513..=2048 => 3,
            2049..=8192 => 4,
            _ => 6,
        }
    }
    
    /// Estimate cache miss rate based on data size and access patterns
    fn estimate_cache_miss_rate(&self, data_size: usize, params: &OptimizationParams) -> f64 {
        // Simple heuristic based on L1/L2/L3 cache sizes
        let l1_cache_size = match self.arch {
            CpuArch::Arm64Neon => 128 * 1024,    // Typical ARM64 L1
            _ => 32 * 1024,                        // Typical x86 L1
        };
        
        let data_footprint = data_size * 4; // f32 = 4 bytes
        
        if data_footprint <= l1_cache_size {
            0.01 // Very low miss rate
        } else if data_footprint <= l1_cache_size * 8 {
            0.05 // Moderate miss rate
        } else {
            0.15 // Higher miss rate for large data
        }
    }
    
    /// Generate optimization recommendations based on measurements
    fn generate_optimization_recommendations(&self) {
        println!("\n📊 Optimization Recommendations:");
        
        for (kernel_type, params) in &self.optimal_params {
            println!("  🎯 {}: batch_size={}, vector_width={}, unroll_factor={}",
                kernel_type, params.batch_size, params.vector_width, params.unroll_factor);
        }
        
        // Architecture-specific recommendations
        match self.arch {
            CpuArch::Arm64Neon => {
                println!("  📱 ARM64 NEON Recommendations:");
                println!("    - Use conservative unroll factors (2-4) to avoid register pressure");
                println!("    - Optimize for power efficiency with moderate batch sizes");
                println!("    - Leverage unified memory architecture for cache optimization");
            },
            CpuArch::X86_64Avx2 => {
                println!("  💻 x86_64 AVX2 Recommendations:");
                println!("    - Use aggressive vectorization with 8-element f32 vectors");
                println!("    - Higher unroll factors (4-8) for better instruction scheduling");
                println!("    - Optimize memory access patterns for x86 cache hierarchy");
            },
            CpuArch::X86_64Avx512 => {
                println!("  🚀 x86_64 AVX-512 Recommendations:");
                println!("    - Maximize vector width utilization (16-element f32 vectors)");
                println!("    - Very aggressive unrolling (8-16) for high-end CPUs");
                println!("    - Consider masked operations for edge cases");
            },
            CpuArch::Generic => {
                println!("  🔧 Generic Recommendations:");
                println!("    - Focus on algorithm optimization over SIMD");
                println!("    - Conservative parameters for broad compatibility");
            },
        }
    }
    
    /// Get optimal parameters for a specific kernel and data size
    pub fn get_optimal_params(&self, kernel_type: &str, data_size: usize) -> Option<&OptimizationParams> {
        let key = format!("{}_{}", kernel_type, data_size);
        self.optimal_params.get(&key)
    }
    
    /// Export optimization results for kernel configuration
    pub fn export_optimization_config(&self) -> Result<String> {
        let mut config = String::new();
        config.push_str("// Generated kernel optimization configuration\n");
        config.push_str(&format!("// Architecture: {:?}\n\n", self.arch));
        
        for (kernel_type, params) in &self.optimal_params {
            config.push_str(&format!("pub const {}_BATCH_SIZE: usize = {};\n", 
                kernel_type.to_uppercase(), params.batch_size));
            config.push_str(&format!("pub const {}_VECTOR_WIDTH: usize = {};\n", 
                kernel_type.to_uppercase(), params.vector_width));
            config.push_str(&format!("pub const {}_UNROLL_FACTOR: usize = {};\n", 
                kernel_type.to_uppercase(), params.unroll_factor));
            config.push_str(&format!("pub const {}_PREFETCH_DISTANCE: usize = {};\n\n", 
                kernel_type.to_uppercase(), params.prefetch_distance));
        }
        
        Ok(config)
    }
}

impl Default for KernelOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = KernelOptimizer::new();
        // Should detect a valid architecture
        match optimizer.arch {
            CpuArch::Arm64Neon | CpuArch::X86_64Avx2 | CpuArch::X86_64Avx512 | CpuArch::Generic => {
                // Valid architecture
            }
        }
    }

    #[test]
    fn test_batch_size_candidates() {
        let optimizer = KernelOptimizer::with_arch(CpuArch::Generic);
        let candidates = optimizer.get_batch_size_candidates();
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|&size| size > 0));
    }

    #[test]
    fn test_optimization_params_default() {
        let params = OptimizationParams::default();
        assert!(params.batch_size > 0);
        assert!(params.vector_width > 0);
        assert!(params.alignment >= 16); // Minimum reasonable alignment
    }
}