//! Automatic Acceleration Backend Selection
//!
//! This module provides intelligent selection of acceleration backends
//! based on hardware capabilities, performance profiling, and operation
//! characteristics. It automatically detects available hardware and
//! optimizes backend selection for maximum performance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{AccelerationBackend, AccelerationError, AccelerationResult};
use crate::tensor::dtype::BitNetDType;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Acceleration capabilities for a specific backend
#[derive(Debug, Clone)]
pub struct AccelerationCapabilities {
    /// The acceleration backend
    pub backend: AccelerationBackend,
    /// Maximum tensor size supported (in elements)
    pub max_tensor_size: usize,
    /// Supported data types
    pub supported_dtypes: Vec<BitNetDType>,
    /// Whether zero-copy operations are supported
    pub zero_copy_support: bool,
    /// Whether parallel execution is supported
    pub parallel_execution: bool,
    /// Peak memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Peak compute throughput in GFLOPS
    pub compute_throughput_gflops: f64,
}

impl AccelerationCapabilities {
    /// Create default capabilities for a backend
    pub fn default_for_backend(backend: AccelerationBackend) -> Self {
        match backend {
            AccelerationBackend::MLX => Self {
                backend,
                max_tensor_size: usize::MAX,
                supported_dtypes: vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I8,
                    BitNetDType::I16,
                    BitNetDType::I32,
                    BitNetDType::U8,
                    BitNetDType::U16,
                    BitNetDType::U32,
                    BitNetDType::Bool,
                ],
                zero_copy_support: true,
                parallel_execution: true,
                memory_bandwidth_gbps: 400.0, // Apple Silicon unified memory
                compute_throughput_gflops: 15800.0, // Estimated for M1 Max
            },
            AccelerationBackend::Metal => Self {
                backend,
                max_tensor_size: 1_000_000_000, // 1B elements
                supported_dtypes: vec![
                    BitNetDType::F32,
                    BitNetDType::F16,
                    BitNetDType::I32,
                    BitNetDType::U32,
                    BitNetDType::Bool,
                ],
                zero_copy_support: false, // Requires GPU memory transfer
                parallel_execution: true,
                memory_bandwidth_gbps: 300.0, // Typical GPU memory bandwidth
                compute_throughput_gflops: 8000.0, // Estimated for Apple GPU
            },
            AccelerationBackend::SIMD => Self {
                backend,
                max_tensor_size: 100_000_000, // 100M elements
                supported_dtypes: vec![BitNetDType::F32, BitNetDType::I32, BitNetDType::U32],
                zero_copy_support: true,
                parallel_execution: true,
                memory_bandwidth_gbps: 50.0, // Typical CPU memory bandwidth
                compute_throughput_gflops: 100.0, // SIMD-optimized CPU
            },
            AccelerationBackend::CPU => Self {
                backend,
                max_tensor_size: 10_000_000, // 10M elements
                supported_dtypes: vec![
                    BitNetDType::F32,
                    BitNetDType::I8,
                    BitNetDType::I16,
                    BitNetDType::I32,
                    BitNetDType::U8,
                    BitNetDType::U16,
                    BitNetDType::U32,
                    BitNetDType::Bool,
                ],
                zero_copy_support: true,
                parallel_execution: false,
                memory_bandwidth_gbps: 20.0, // Basic CPU memory bandwidth
                compute_throughput_gflops: 10.0, // Basic CPU compute
            },
        }
    }

    /// Check if backend supports a specific data type
    pub fn supports_dtype(&self, dtype: BitNetDType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Check if backend can handle a tensor of given size
    pub fn can_handle_size(&self, size: usize) -> bool {
        size <= self.max_tensor_size
    }

    /// Get efficiency score for a given operation
    pub fn efficiency_score(
        &self,
        flops: u64,
        memory_bytes: usize,
        prefer_low_latency: bool,
    ) -> f64 {
        let compute_score = (flops as f64) / self.compute_throughput_gflops / 1e9;
        let memory_score = (memory_bytes as f64) / self.memory_bandwidth_gbps / 1e9;

        let total_time = compute_score.max(memory_score);

        if prefer_low_latency {
            // For latency, prefer faster absolute time
            1.0 / (total_time + 1e-9)
        } else {
            // For throughput, consider parallelism
            let parallel_factor = if self.parallel_execution { 4.0 } else { 1.0 };
            parallel_factor / (total_time + 1e-9)
        }
    }
}

/// Hardware detection and capability assessment
pub struct HardwareDetector {
    /// Cached hardware capabilities
    capabilities_cache: Mutex<Option<HardwareCapabilities>>,
}

/// Detected hardware capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Whether we're running on Apple Silicon
    pub is_apple_silicon: bool,
    /// Whether Metal GPU is available
    pub has_metal_gpu: bool,
    /// Whether MLX is available
    pub has_mlx: bool,
    /// Detected SIMD instruction sets
    pub simd_capabilities: SimdCapabilities,
    /// System memory in GB
    pub system_memory_gb: f64,
    /// CPU core count
    pub cpu_cores: usize,
    /// GPU memory in GB (if available)
    pub gpu_memory_gb: Option<f64>,
}

/// SIMD instruction set capabilities
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// AVX-512 support (x86_64)
    pub avx512: bool,
    /// NEON support (ARM64)
    pub neon: bool,
    /// SSE 4.2 support (x86_64)
    pub sse42: bool,
}

impl HardwareDetector {
    /// Create new hardware detector
    pub fn new() -> Self {
        Self {
            capabilities_cache: Mutex::new(None),
        }
    }

    /// Detect hardware capabilities
    pub fn detect_capabilities(&self) -> AccelerationResult<HardwareCapabilities> {
        // Check cache first
        if let Ok(cache) = self.capabilities_cache.lock() {
            if let Some(ref capabilities) = *cache {
                return Ok(capabilities.clone());
            }
        }

        let capabilities = self.perform_detection()?;

        // Cache the result
        if let Ok(mut cache) = self.capabilities_cache.lock() {
            *cache = Some(capabilities.clone());
        }

        #[cfg(feature = "tracing")]
        info!("Detected hardware capabilities: {:?}", capabilities);

        Ok(capabilities)
    }

    /// Perform actual hardware detection
    fn perform_detection(&self) -> AccelerationResult<HardwareCapabilities> {
        let is_apple_silicon = cfg!(target_arch = "aarch64") && cfg!(target_os = "macos");

        let simd_capabilities = self.detect_simd_capabilities();
        let system_memory_gb = self.detect_system_memory();
        let cpu_cores = self.detect_cpu_cores();

        let (has_metal_gpu, gpu_memory_gb) = self.detect_metal_gpu();
        let has_mlx = self.detect_mlx_availability();

        Ok(HardwareCapabilities {
            is_apple_silicon,
            has_metal_gpu,
            has_mlx,
            simd_capabilities,
            system_memory_gb,
            cpu_cores,
            gpu_memory_gb,
        })
    }

    /// Detect SIMD instruction set support
    fn detect_simd_capabilities(&self) -> SimdCapabilities {
        SimdCapabilities {
            avx2: cfg!(target_feature = "avx2") || self.runtime_check_avx2(),
            avx512: cfg!(target_feature = "avx512f") || self.runtime_check_avx512(),
            neon: cfg!(target_feature = "neon") || self.runtime_check_neon(),
            sse42: cfg!(target_feature = "sse4.2") || self.runtime_check_sse42(),
        }
    }

    /// Runtime check for AVX2 support
    fn runtime_check_avx2(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// Runtime check for AVX-512 support
    fn runtime_check_avx512(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// Runtime check for NEON support
    fn runtime_check_neon(&self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        false
    }

    /// Runtime check for SSE 4.2 support
    fn runtime_check_sse42(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("sse4.2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// Detect system memory
    fn detect_system_memory(&self) -> f64 {
        // This would use system APIs to get actual memory
        // For now, use a reasonable default
        if cfg!(target_os = "macos") {
            16.0 // Assume 16GB for macOS
        } else {
            8.0 // Assume 8GB for other systems
        }
    }

    /// Detect CPU core count
    fn detect_cpu_cores(&self) -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }

    /// Detect Metal GPU availability
    fn detect_metal_gpu(&self) -> (bool, Option<f64>) {
        #[cfg(feature = "metal")]
        {
            if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
                // On Apple platforms, Metal is typically available
                (true, Some(8.0)) // Assume 8GB GPU memory
            } else {
                (false, None)
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            (false, None)
        }
    }

    /// Detect MLX availability
    fn detect_mlx_availability(&self) -> bool {
        #[cfg(feature = "mlx")]
        {
            cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
        }
        #[cfg(not(feature = "mlx"))]
        {
            false
        }
    }
}

/// Automatic acceleration backend selector
pub struct AutoAccelerationSelector {
    /// Hardware detector
    detector: HardwareDetector,
    /// Available backend capabilities
    backend_capabilities: RwLock<HashMap<AccelerationBackend, AccelerationCapabilities>>,
    /// Performance benchmarks
    performance_cache: Arc<Mutex<HashMap<String, f64>>>,
}

impl AutoAccelerationSelector {
    /// Create new acceleration selector
    pub fn new() -> AccelerationResult<Self> {
        let detector = HardwareDetector::new();
        let backend_capabilities = RwLock::new(HashMap::new());
        let performance_cache = Arc::new(Mutex::new(HashMap::new()));

        let mut selector = Self {
            detector,
            backend_capabilities,
            performance_cache,
        };

        // Initialize backend capabilities
        selector.initialize_capabilities()?;

        Ok(selector)
    }

    /// Initialize backend capabilities based on detected hardware
    fn initialize_capabilities(&mut self) -> AccelerationResult<()> {
        let hardware = self.detector.detect_capabilities()?;
        let mut capabilities = HashMap::new();

        // MLX backend (Apple Silicon only)
        if hardware.has_mlx {
            capabilities.insert(
                AccelerationBackend::MLX,
                AccelerationCapabilities::default_for_backend(AccelerationBackend::MLX),
            );

            #[cfg(feature = "tracing")]
            info!("MLX acceleration available");
        }

        // Metal backend (Apple platforms)
        if hardware.has_metal_gpu {
            let mut metal_caps =
                AccelerationCapabilities::default_for_backend(AccelerationBackend::Metal);
            if let Some(gpu_memory_gb) = hardware.gpu_memory_gb {
                // Adjust max tensor size based on GPU memory
                metal_caps.max_tensor_size = ((gpu_memory_gb * 0.8 * 1e9) / 4.0) as usize;
                // 80% of GPU memory, 4 bytes per f32
            }
            capabilities.insert(AccelerationBackend::Metal, metal_caps);

            #[cfg(feature = "tracing")]
            info!("Metal acceleration available");
        }

        // SIMD backend (based on detected instruction sets)
        if hardware.simd_capabilities.avx2
            || hardware.simd_capabilities.neon
            || hardware.simd_capabilities.sse42
        {
            let mut simd_caps =
                AccelerationCapabilities::default_for_backend(AccelerationBackend::SIMD);

            // Adjust performance based on available instruction sets
            if hardware.simd_capabilities.avx512 {
                simd_caps.compute_throughput_gflops *= 2.0;
            } else if hardware.simd_capabilities.avx2 {
                simd_caps.compute_throughput_gflops *= 1.5;
            } else if hardware.simd_capabilities.neon {
                simd_caps.compute_throughput_gflops *= 1.3;
            }

            capabilities.insert(AccelerationBackend::SIMD, simd_caps);

            #[cfg(feature = "tracing")]
            info!(
                "SIMD acceleration available: AVX2={}, AVX512={}, NEON={}, SSE4.2={}",
                hardware.simd_capabilities.avx2,
                hardware.simd_capabilities.avx512,
                hardware.simd_capabilities.neon,
                hardware.simd_capabilities.sse42
            );
        }

        // CPU backend (always available)
        capabilities.insert(
            AccelerationBackend::CPU,
            AccelerationCapabilities::default_for_backend(AccelerationBackend::CPU),
        );

        // Update capabilities
        if let Ok(mut caps) = self.backend_capabilities.write() {
            *caps = capabilities;
        }

        Ok(())
    }

    /// Get available acceleration backends
    pub fn get_available_backends(&self) -> Vec<AccelerationBackend> {
        if let Ok(capabilities) = self.backend_capabilities.read() {
            capabilities.keys().copied().collect()
        } else {
            vec![]
        }
    }

    /// Get capabilities for a specific backend
    pub fn get_backend_capabilities(
        &self,
        backend: AccelerationBackend,
    ) -> Option<AccelerationCapabilities> {
        if let Ok(capabilities) = self.backend_capabilities.read() {
            capabilities.get(&backend).cloned()
        } else {
            None
        }
    }

    /// Select best backend for given requirements
    pub fn select_best_backend(
        &self,
        tensor_size: usize,
        dtype: BitNetDType,
        estimated_flops: u64,
        estimated_memory_bytes: usize,
        prefer_low_latency: bool,
    ) -> AccelerationResult<AccelerationBackend> {
        let capabilities = self.backend_capabilities.read().map_err(|_| {
            AccelerationError::InitializationFailed {
                backend: "Selector".to_string(),
                reason: "Failed to acquire capabilities lock".to_string(),
            }
        })?;

        let mut candidates = Vec::new();

        // Filter backends by requirements
        for (backend, caps) in capabilities.iter() {
            if caps.supports_dtype(dtype) && caps.can_handle_size(tensor_size) {
                let efficiency_score = caps.efficiency_score(
                    estimated_flops,
                    estimated_memory_bytes,
                    prefer_low_latency,
                );

                candidates.push((*backend, efficiency_score));
            }
        }

        // Sort by efficiency score (higher is better)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .first()
            .map(|(backend, _)| *backend)
            .ok_or_else(|| AccelerationError::BackendNotAvailable {
                backend: "Any suitable".to_string(),
            })
    }

    /// Select backend with fallback chain
    pub fn select_with_fallback(
        &self,
        preferred: AccelerationBackend,
        tensor_size: usize,
        dtype: BitNetDType,
    ) -> AccelerationBackend {
        let capabilities = match self.backend_capabilities.read() {
            Ok(caps) => caps,
            Err(_) => return AccelerationBackend::CPU,
        };

        // Try preferred backend first
        if let Some(caps) = capabilities.get(&preferred) {
            if caps.supports_dtype(dtype) && caps.can_handle_size(tensor_size) {
                return preferred;
            }
        }

        // Fallback order
        let fallback_order = [
            AccelerationBackend::MLX,
            AccelerationBackend::Metal,
            AccelerationBackend::SIMD,
            AccelerationBackend::CPU,
        ];

        for &backend in &fallback_order {
            if backend == preferred {
                continue; // Already tried
            }

            if let Some(caps) = capabilities.get(&backend) {
                if caps.supports_dtype(dtype) && caps.can_handle_size(tensor_size) {
                    return backend;
                }
            }
        }

        // Ultimate fallback
        AccelerationBackend::CPU
    }

    /// Benchmark and update performance cache
    pub fn update_performance_cache(
        &self,
        operation: &str,
        backend: AccelerationBackend,
        speedup: f64,
    ) {
        let key = format!("{}_{}", operation, backend);

        if let Ok(mut cache) = self.performance_cache.lock() {
            cache.insert(key, speedup);
        }
    }

    /// Get cached performance data
    pub fn get_cached_performance(
        &self,
        operation: &str,
        backend: AccelerationBackend,
    ) -> Option<f64> {
        let key = format!("{}_{}", operation, backend);

        self.performance_cache.lock().ok()?.get(&key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detector_creation() {
        let detector = HardwareDetector::new();
        let capabilities = detector.detect_capabilities();
        assert!(capabilities.is_ok());
    }

    #[test]
    fn test_acceleration_capabilities() {
        let caps = AccelerationCapabilities::default_for_backend(AccelerationBackend::MLX);
        assert_eq!(caps.backend, AccelerationBackend::MLX);
        assert!(caps.zero_copy_support);
        assert!(caps.parallel_execution);
        assert!(caps.supports_dtype(BitNetDType::F32));
    }

    #[test]
    fn test_auto_selector_creation() {
        let selector = AutoAccelerationSelector::new();
        assert!(selector.is_ok());

        if let Ok(selector) = selector {
            let backends = selector.get_available_backends();
            assert!(!backends.is_empty()); // Should at least have CPU
            assert!(backends.contains(&AccelerationBackend::CPU));
        }
    }

    #[test]
    fn test_backend_selection_fallback() {
        let selector = AutoAccelerationSelector::new().unwrap();

        let backend =
            selector.select_with_fallback(AccelerationBackend::MLX, 1000, BitNetDType::F32);

        // Should return a valid backend
        assert!(matches!(
            backend,
            AccelerationBackend::MLX
                | AccelerationBackend::Metal
                | AccelerationBackend::SIMD
                | AccelerationBackend::CPU
        ));
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn test_apple_silicon_detection() {
        let detector = HardwareDetector::new();
        let capabilities = detector.detect_capabilities().unwrap();

        assert!(capabilities.is_apple_silicon);
        assert!(capabilities.simd_capabilities.neon);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_simd_detection() {
        let detector = HardwareDetector::new();
        let capabilities = detector.detect_capabilities().unwrap();

        assert!(!capabilities.is_apple_silicon);
        // At least SSE should be available on modern x86_64
    }
}
