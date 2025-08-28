//! SIMD capability detection and runtime feature selection
//!
//! This module provides comprehensive detection of available SIMD instruction sets
//! across different architectures, enabling optimal code path selection at runtime.

/// SIMD instruction set capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct SimdCapabilities {
    // x86/x86_64 capabilities
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub fma: bool,

    // ARM capabilities
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,

    // General capabilities
    pub vector_size: usize,
    pub cache_line_size: usize,
}

impl SimdCapabilities {
    /// Detect all available SIMD capabilities
    pub fn detect() -> Self {
        let mut caps = Self::default();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            caps.sse = is_x86_feature_detected!("sse");
            caps.sse2 = is_x86_feature_detected!("sse2");
            caps.sse3 = is_x86_feature_detected!("sse3");
            caps.ssse3 = is_x86_feature_detected!("ssse3");
            caps.sse4_1 = is_x86_feature_detected!("sse4.1");
            caps.sse4_2 = is_x86_feature_detected!("sse4.2");
            caps.avx = is_x86_feature_detected!("avx");
            caps.avx2 = is_x86_feature_detected!("avx2");
            caps.avx512f = is_x86_feature_detected!("avx512f");
            caps.avx512dq = is_x86_feature_detected!("avx512dq");
            caps.avx512cd = is_x86_feature_detected!("avx512cd");
            caps.avx512bw = is_x86_feature_detected!("avx512bw");
            caps.avx512vl = is_x86_feature_detected!("avx512vl");
            caps.fma = is_x86_feature_detected!("fma");

            caps.vector_size = if caps.avx512f {
                64 // 512 bits
            } else if caps.avx2 {
                32 // 256 bits
            } else if caps.sse2 {
                16 // 128 bits
            } else {
                4 // 32 bits scalar
            };
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = std::arch::is_aarch64_feature_detected!("neon");

            // SVE detection is more complex and may not be available in std
            caps.sve = false; // TODO: Implement SVE detection when available
            caps.sve2 = false;

            caps.vector_size = if caps.neon { 16 } else { 4 };
        }

        #[cfg(target_arch = "arm")]
        {
            caps.neon = std::arch::is_arm_feature_detected!("neon");
            caps.vector_size = if caps.neon { 16 } else { 4 };
        }

        // Cache line size detection (platform-specific)
        caps.cache_line_size = detect_cache_line_size();

        caps
    }

    /// Check if any SIMD instructions are available
    pub fn has_simd(&self) -> bool {
        self.sse2 || self.neon
    }

    /// Get the optimal vector size for the current architecture
    pub fn optimal_vector_size(&self) -> usize {
        self.vector_size
    }

    /// Get the optimal alignment for SIMD operations
    pub fn optimal_alignment(&self) -> usize {
        if self.avx512f {
            64
        } else if self.avx2 {
            32
        } else if self.sse2 || self.neon {
            16
        } else {
            std::mem::align_of::<f32>()
        }
    }

    /// Check if a specific minimum capability level is met
    pub fn supports_level(&self, level: SimdLevel) -> bool {
        match level {
            SimdLevel::None => true,
            SimdLevel::Basic => self.sse2 || self.neon,
            SimdLevel::Advanced => self.avx2 || self.neon,
            SimdLevel::HighEnd => self.avx512f || self.sve,
        }
    }

    /// Get a human-readable description of capabilities
    pub fn description(&self) -> String {
        let mut features = Vec::new();

        if self.avx512f {
            features.push("AVX-512");
        } else if self.avx2 {
            features.push("AVX2");
        } else if self.avx {
            features.push("AVX");
        } else if self.sse4_2 {
            features.push("SSE4.2");
        } else if self.sse4_1 {
            features.push("SSE4.1");
        } else if self.sse2 {
            features.push("SSE2");
        }

        if self.sve2 {
            features.push("SVE2");
        } else if self.sve {
            features.push("SVE");
        } else if self.neon {
            features.push("NEON");
        }

        if self.fma {
            features.push("FMA");
        }

        if features.is_empty() {
            "No SIMD support".to_string()
        } else {
            format!(
                "{} ({}-byte vectors)",
                features.join(", "),
                self.vector_size
            )
        }
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512dq: false,
            avx512cd: false,
            avx512bw: false,
            avx512vl: false,
            fma: false,
            neon: false,
            sve: false,
            sve2: false,
            vector_size: 4,
            cache_line_size: 64,
        }
    }
}

/// SIMD capability levels for easy categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD support
    None,
    /// Basic SIMD (SSE2, NEON)
    Basic,
    /// Advanced SIMD (AVX2, Advanced NEON)
    Advanced,
    /// High-end SIMD (AVX-512, SVE)
    HighEnd,
}

impl SimdLevel {
    /// Get the recommended level for the current system
    pub fn recommended() -> Self {
        let caps = detect_simd_capabilities();

        if caps.avx512f || caps.sve {
            SimdLevel::HighEnd
        } else if caps.avx2 {
            SimdLevel::Advanced
        } else if caps.sse2 || caps.neon {
            SimdLevel::Basic
        } else {
            SimdLevel::None
        }
    }
}

/// Detect cache line size for optimal memory access patterns
fn detect_cache_line_size() -> usize {
    // Platform-specific cache line detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
        {
            if let Ok(size) = contents.trim().parse::<usize>() {
                return size;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.cachelinesize")
            .output()
        {
            if let Ok(size_str) = String::from_utf8(output.stdout) {
                if let Ok(size) = size_str.trim().parse::<usize>() {
                    return size;
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Use GetLogicalProcessorInformation on Windows
        // For now, fall back to common default
    }

    // Common defaults for different architectures
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return 64;

    #[cfg(target_arch = "aarch64")]
    return 64;

    #[cfg(target_arch = "arm")]
    return 32;

    #[allow(unreachable_code)]
    64 // Safe default
}

/// Global SIMD capabilities instance
static mut GLOBAL_CAPABILITIES: Option<SimdCapabilities> = None;
static INIT_ONCE: std::sync::Once = std::sync::Once::new();

/// Get the global SIMD capabilities (cached)
pub fn detect_simd_capabilities() -> SimdCapabilities {
    INIT_ONCE.call_once(|| unsafe {
        GLOBAL_CAPABILITIES = Some(SimdCapabilities::detect());
    });

    unsafe { GLOBAL_CAPABILITIES.unwrap() }
}

/// Runtime feature detection macros for easier usage
#[macro_export]
macro_rules! simd_dispatch {
    ($caps:expr, {
        avx512: $avx512_fn:expr,
        avx2: $avx2_fn:expr,
        sse2: $sse2_fn:expr,
        neon: $neon_fn:expr,
        fallback: $fallback_fn:expr $(,)?
    }) => {
        if $caps.avx512f {
            $avx512_fn
        } else if $caps.avx2 {
            $avx2_fn
        } else if $caps.sse2 {
            $sse2_fn
        } else if $caps.neon {
            $neon_fn
        } else {
            $fallback_fn
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let caps = SimdCapabilities::detect();
        println!("Detected SIMD capabilities: {}", caps.description());

        // Basic sanity checks
        assert!(caps.vector_size >= 4);
        assert!(caps.cache_line_size >= 16);

        if caps.has_simd() {
            println!("SIMD support available");
        } else {
            println!("No SIMD support detected");
        }
    }

    #[test]
    fn test_simd_levels() {
        let level = SimdLevel::recommended();
        println!("Recommended SIMD level: {:?}", level);

        let caps = detect_simd_capabilities();
        assert!(caps.supports_level(SimdLevel::None));

        if caps.has_simd() {
            assert!(caps.supports_level(SimdLevel::Basic));
        }
    }

    #[test]
    fn test_cache_line_detection() {
        let size = detect_cache_line_size();
        assert!(size >= 16 && size <= 256);
        println!("Detected cache line size: {} bytes", size);
    }

    #[test]
    fn test_global_capabilities() {
        let caps1 = detect_simd_capabilities();
        let caps2 = detect_simd_capabilities();

        // Should be the same instance (cached)
        assert_eq!(caps1.vector_size, caps2.vector_size);
        assert_eq!(caps1.cache_line_size, caps2.cache_line_size);
    }
}
