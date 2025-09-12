//! Advanced CPU Feature Detection and Runtime Architecture Management
//!
//! This module provides comprehensive CPU feature detection using CPUID instructions
//! and other hardware-specific methods to enable optimal kernel selection.

use anyhow::{Result, bail};
use std::collections::HashMap;

use crate::cpu::CpuArch;

/// Detailed CPU feature flags for fine-grained optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuFeatures {
    /// Basic architecture type
    pub arch: CpuArch,
    /// Specific feature flags
    pub features: HashMap<String, bool>,
    /// Cache hierarchy information
    pub cache_info: CacheInfo,
    /// CPU core information
    pub core_info: CoreInfo,
}

/// CPU cache hierarchy information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInfo {
    /// L1 data cache size in bytes
    pub l1_data_size: usize,
    /// L1 instruction cache size in bytes  
    pub l1_inst_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L3 cache size in bytes (0 if not present)
    pub l3_size: usize,
    /// Cache line size in bytes
    pub cache_line_size: usize,
}

/// CPU core and threading information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreInfo {
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: usize,
    /// Base CPU frequency in MHz (if available)
    pub base_frequency: Option<usize>,
    /// Maximum CPU frequency in MHz (if available)
    pub max_frequency: Option<usize>,
}

/// Advanced CPU feature detector with CPUID support
pub struct CpuFeatureDetector {
    /// Cached detection results
    cached_features: Option<CpuFeatures>,
}

impl CpuFeatureDetector {
    /// Create a new CPU feature detector
    pub fn new() -> Self {
        Self {
            cached_features: None,
        }
    }
    
    /// Detect CPU features with comprehensive hardware interrogation
    pub fn detect_features(&mut self) -> Result<&CpuFeatures> {
        if self.cached_features.is_none() {
            self.cached_features = Some(self.perform_detection()?);
        }
        
        Ok(self.cached_features.as_ref().unwrap())
    }
    
    /// Perform comprehensive CPU feature detection
    fn perform_detection(&self) -> Result<CpuFeatures> {
        println!("🔍 Performing comprehensive CPU feature detection...");
        
        let mut features = HashMap::new();
        
        // Detect basic architecture and SIMD features
        let arch = self.detect_architecture(&mut features)?;
        
        // Detect cache hierarchy
        let cache_info = self.detect_cache_hierarchy(&arch)?;
        
        // Detect core information
        let core_info = self.detect_core_info(&arch)?;
        
        let cpu_features = CpuFeatures {
            arch,
            features,
            cache_info,
            core_info,
        };
        
        self.print_detection_summary(&cpu_features);
        
        Ok(cpu_features)
    }
    
    /// Detect CPU architecture and SIMD capabilities
    fn detect_architecture(&self, features: &mut HashMap<String, bool>) -> Result<CpuArch> {
        #[cfg(target_arch = "aarch64")]
        {
            self.detect_arm64_features(features)
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_64_features(features)
        }
        
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            features.insert("generic_only".to_string(), true);
            Ok(CpuArch::Generic)
        }
    }
    
    /// Detect ARM64 NEON and other features
    #[cfg(target_arch = "aarch64")]
    fn detect_arm64_features(&self, features: &mut HashMap<String, bool>) -> Result<CpuArch> {
        println!("  📱 Detecting ARM64 features...");
        
        // NEON is standard on all ARM64, but check for availability
        let neon_available = self.check_arm64_neon();
        features.insert("neon".to_string(), neon_available);
        
        // Check for additional ARM64 features
        let fp16_available = self.check_arm64_fp16();
        features.insert("fp16".to_string(), fp16_available);
        
        let sve_available = self.check_arm64_sve();
        features.insert("sve".to_string(), sve_available);
        
        let dot_product_available = self.check_arm64_dot_product();
        features.insert("dot_product".to_string(), dot_product_available);
        
        println!("    ✅ NEON: {}, FP16: {}, SVE: {}, DotProduct: {}", 
            neon_available, fp16_available, sve_available, dot_product_available);
        
        if neon_available {
            Ok(CpuArch::Arm64Neon)
        } else {
            Ok(CpuArch::Generic)
        }
    }
    
    /// Detect x86_64 AVX and other features using CPUID
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64_features(&self, features: &mut HashMap<String, bool>) -> Result<CpuArch> {
        println!("  💻 Detecting x86_64 features...");
        
        // Check SSE features
        let sse_available = std::arch::is_x86_feature_detected!("sse");
        let sse2_available = std::arch::is_x86_feature_detected!("sse2");
        let sse3_available = std::arch::is_x86_feature_detected!("sse3");
        let sse41_available = std::arch::is_x86_feature_detected!("sse4.1");
        let sse42_available = std::arch::is_x86_feature_detected!("sse4.2");
        
        features.insert("sse".to_string(), sse_available);
        features.insert("sse2".to_string(), sse2_available);
        features.insert("sse3".to_string(), sse3_available);
        features.insert("sse4.1".to_string(), sse41_available);
        features.insert("sse4.2".to_string(), sse42_available);
        
        // Check AVX features
        let avx_available = std::arch::is_x86_feature_detected!("avx");
        let avx2_available = std::arch::is_x86_feature_detected!("avx2");
        let avx512f_available = std::arch::is_x86_feature_detected!("avx512f");
        let avx512bw_available = std::arch::is_x86_feature_detected!("avx512bw");
        let avx512cd_available = std::arch::is_x86_feature_detected!("avx512cd");
        let avx512dq_available = std::arch::is_x86_feature_detected!("avx512dq");
        let avx512vl_available = std::arch::is_x86_feature_detected!("avx512vl");
        
        features.insert("avx".to_string(), avx_available);
        features.insert("avx2".to_string(), avx2_available);
        features.insert("avx512f".to_string(), avx512f_available);
        features.insert("avx512bw".to_string(), avx512bw_available);
        features.insert("avx512cd".to_string(), avx512cd_available);
        features.insert("avx512dq".to_string(), avx512dq_available);
        features.insert("avx512vl".to_string(), avx512vl_available);
        
        // Check FMA and other useful features
        let fma_available = std::arch::is_x86_feature_detected!("fma");
        let bmi1_available = std::arch::is_x86_feature_detected!("bmi1");
        let bmi2_available = std::arch::is_x86_feature_detected!("bmi2");
        
        features.insert("fma".to_string(), fma_available);
        features.insert("bmi1".to_string(), bmi1_available);
        features.insert("bmi2".to_string(), bmi2_available);
        
        println!("    ✅ SSE4.2: {}, AVX: {}, AVX2: {}, AVX-512F: {}, FMA: {}", 
            sse42_available, avx_available, avx2_available, avx512f_available, fma_available);
        
        // Determine best architecture level
        if avx512f_available && avx512bw_available && avx512dq_available && avx512vl_available {
            Ok(CpuArch::X86_64Avx512)
        } else if avx2_available {
            Ok(CpuArch::X86_64Avx2)
        } else {
            Ok(CpuArch::Generic)
        }
    }
    
    /// Check ARM64 NEON availability
    #[cfg(target_arch = "aarch64")]
    fn check_arm64_neon(&self) -> bool {
        // NEON is standard on ARM64
        true
    }
    
    /// Check ARM64 FP16 support
    #[cfg(target_arch = "aarch64")]
    fn check_arm64_fp16(&self) -> bool {
        // Check for ARMv8.2 FP16 support
        #[cfg(target_feature = "fp16")]
        {
            true
        }
        #[cfg(not(target_feature = "fp16"))]
        {
            false
        }
    }
    
    /// Check ARM64 SVE (Scalable Vector Extension) support
    #[cfg(target_arch = "aarch64")]
    fn check_arm64_sve(&self) -> bool {
        // SVE support detection (limited in current Rust)
        false // Conservative default until better detection available
    }
    
    /// Check ARM64 dot product instructions
    #[cfg(target_arch = "aarch64")]
    fn check_arm64_dot_product(&self) -> bool {
        // Check for ARMv8.2 dot product support
        #[cfg(target_feature = "dotprod")]
        {
            true
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            false
        }
    }
    
    /// Detect cache hierarchy information
    fn detect_cache_hierarchy(&self, arch: &CpuArch) -> Result<CacheInfo> {
        // Platform-specific cache detection
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_cache_info()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            self.detect_arm64_cache_info()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Ok(CacheInfo {
                l1_data_size: 32 * 1024,
                l1_inst_size: 32 * 1024,
                l2_size: 256 * 1024,
                l3_size: 0,
                cache_line_size: 64,
            })
        }
    }
    
    /// Detect x86_64 cache information using CPUID
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_cache_info(&self) -> Result<CacheInfo> {
        // Use conservative defaults for x86_64
        // TODO: Implement proper CPUID cache detection
        Ok(CacheInfo {
            l1_data_size: 32 * 1024,     // 32KB L1 data
            l1_inst_size: 32 * 1024,     // 32KB L1 instruction
            l2_size: 256 * 1024,         // 256KB L2
            l3_size: 8 * 1024 * 1024,    // 8MB L3 (conservative)
            cache_line_size: 64,         // 64-byte cache lines
        })
    }
    
    /// Detect ARM64 cache information
    #[cfg(target_arch = "aarch64")]
    fn detect_arm64_cache_info(&self) -> Result<CacheInfo> {
        // ARM64 typical cache configuration
        Ok(CacheInfo {
            l1_data_size: 64 * 1024,     // 64KB L1 data (typical for ARM64)
            l1_inst_size: 64 * 1024,     // 64KB L1 instruction
            l2_size: 512 * 1024,         // 512KB L2 (typical)
            l3_size: 4 * 1024 * 1024,    // 4MB L3 (if present)
            cache_line_size: 64,         // 64-byte cache lines
        })
    }
    
    /// Detect CPU core information
    fn detect_core_info(&self, arch: &CpuArch) -> Result<CoreInfo> {
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        
        // Conservative estimate: assume no hyperthreading for non-x86
        let physical_cores = match arch {
            CpuArch::X86_64Avx2 | CpuArch::X86_64Avx512 => {
                // Assume 2:1 logical to physical ratio for x86_64 with hyperthreading
                (logical_cores + 1) / 2
            },
            _ => logical_cores, // ARM typically doesn't use hyperthreading
        };
        
        Ok(CoreInfo {
            physical_cores,
            logical_cores,
            base_frequency: None, // Frequency detection not implemented
            max_frequency: None,
        })
    }
    
    /// Print comprehensive detection summary
    fn print_detection_summary(&self, features: &CpuFeatures) {
        println!("🎯 CPU Feature Detection Summary:");
        println!("  Architecture: {:?}", features.arch);
        println!("  Physical Cores: {}, Logical Cores: {}", 
            features.core_info.physical_cores, features.core_info.logical_cores);
        println!("  Cache: L1D={}KB, L1I={}KB, L2={}KB, L3={}KB", 
            features.cache_info.l1_data_size / 1024,
            features.cache_info.l1_inst_size / 1024,
            features.cache_info.l2_size / 1024,
            features.cache_info.l3_size / 1024);
        
        println!("  Key Features:");
        for (feature, available) in &features.features {
            if *available {
                println!("    ✅ {}", feature);
            }
        }
    }
    
    /// Get optimal kernel selection based on detected features
    pub fn get_optimal_kernel_arch(&mut self) -> Result<CpuArch> {
        let features = self.detect_features()?;
        Ok(features.arch)
    }
    
    /// Check if a specific feature is available
    pub fn has_feature(&mut self, feature_name: &str) -> Result<bool> {
        let features = self.detect_features()?;
        Ok(features.features.get(feature_name).copied().unwrap_or(false))
    }
    
    /// Get recommended thread pool size based on hardware
    pub fn get_recommended_thread_count(&mut self) -> Result<usize> {
        let features = self.detect_features()?;
        
        // Use logical cores for compute-bound tasks, but cap at reasonable limit
        let recommended = match features.arch {
            CpuArch::Arm64Neon => {
                // ARM cores are typically more efficient, use all logical cores
                features.core_info.logical_cores
            },
            CpuArch::X86_64Avx2 | CpuArch::X86_64Avx512 => {
                // For x86_64, may want to use physical cores for compute-heavy tasks
                // to avoid hyperthreading overhead
                features.core_info.physical_cores
            },
            CpuArch::Generic => {
                // Conservative approach for unknown architectures
                std::cmp::min(features.core_info.logical_cores, 4)
            },
        };
        
        Ok(std::cmp::max(1, recommended))
    }
}

impl Default for CpuFeatureDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = CpuFeatureDetector::new();
        assert!(detector.cached_features.is_none());
    }

    #[test]
    fn test_feature_detection() {
        let mut detector = CpuFeatureDetector::new();
        let features = detector.detect_features().unwrap();
        
        // Should detect a valid architecture
        match features.arch {
            CpuArch::Arm64Neon | CpuArch::X86_64Avx2 | CpuArch::X86_64Avx512 | CpuArch::Generic => {
                // Valid architecture
            }
        }
        
        // Should have some cache information
        assert!(features.cache_info.l1_data_size > 0);
        assert!(features.cache_info.cache_line_size > 0);
        
        // Should have core information
        assert!(features.core_info.logical_cores > 0);
        assert!(features.core_info.physical_cores > 0);
    }

    #[test]
    fn test_thread_count_recommendation() {
        let mut detector = CpuFeatureDetector::new();
        let thread_count = detector.get_recommended_thread_count().unwrap();
        assert!(thread_count > 0);
        assert!(thread_count <= 64); // Reasonable upper bound
    }
}