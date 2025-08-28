//! Operation Dispatch System for Tensor Acceleration
//!
//! This module provides the dispatch system that automatically selects
//! the best acceleration backend for tensor operations based on:
//! - Hardware availability
//! - Operation characteristics
//! - Performance profiling results
//! - User preferences
//!
//! The dispatch system extends the existing auto_select_device() logic
//! to provide operation-specific backend selection.

use std::sync::{Arc, RwLock, Mutex};
use std::collections::HashMap;
use crate::tensor::core::BitNetTensor;
use crate::tensor::dtype::BitNetDType;
use super::{
    AccelerationResult, AccelerationError, AccelerationBackendImpl,
    AccelerationMetrics, AutoAccelerationSelector, AccelerationCapabilities
};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Available acceleration backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelerationBackend {
    /// Apple MLX framework (Apple Silicon only)
    MLX,
    /// Metal GPU compute shaders
    Metal,
    /// SIMD-optimized CPU operations
    SIMD,
    /// Basic CPU operations (fallback)
    CPU,
}

impl std::fmt::Display for AccelerationBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccelerationBackend::MLX => write!(f, "MLX"),
            AccelerationBackend::Metal => write!(f, "Metal"),
            AccelerationBackend::SIMD => write!(f, "SIMD"),
            AccelerationBackend::CPU => write!(f, "CPU"),
        }
    }
}

impl AccelerationBackend {
    /// Get the priority order for backend selection
    /// Higher numbers = higher priority
    pub fn priority(&self) -> u8 {
        match self {
            AccelerationBackend::MLX => 100,    // Highest priority on Apple Silicon
            AccelerationBackend::Metal => 80,   // High priority for GPU operations
            AccelerationBackend::SIMD => 60,    // Good CPU performance
            AccelerationBackend::CPU => 40,     // Fallback
        }
    }

    /// Check if this backend is available on the current platform
    pub fn is_platform_supported(&self) -> bool {
        match self {
            AccelerationBackend::MLX => {
                cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
            },
            AccelerationBackend::Metal => {
                cfg!(target_os = "macos") || cfg!(target_os = "ios")
            },
            AccelerationBackend::SIMD => {
                cfg!(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))
            },
            AccelerationBackend::CPU => true, // Always available
        }
    }

    /// Get estimated performance characteristics
    pub fn performance_characteristics(&self) -> PerformanceCharacteristics {
        match self {
            AccelerationBackend::MLX => PerformanceCharacteristics {
                throughput_gflops: 15000.0,  // High throughput on Apple Silicon
                latency_us: 100.0,           // Low latency
                memory_bandwidth_gbps: 400.0, // Unified memory
                power_efficiency: 0.9,       // High efficiency
            },
            AccelerationBackend::Metal => PerformanceCharacteristics {
                throughput_gflops: 8000.0,   // High GPU throughput
                latency_us: 200.0,           // Higher latency due to GPU transfer
                memory_bandwidth_gbps: 200.0,
                power_efficiency: 0.6,       // Good efficiency
            },
            AccelerationBackend::SIMD => PerformanceCharacteristics {
                throughput_gflops: 100.0,    // Good CPU throughput
                latency_us: 50.0,            // Very low latency
                memory_bandwidth_gbps: 50.0,
                power_efficiency: 0.8,       // Good efficiency
            },
            AccelerationBackend::CPU => PerformanceCharacteristics {
                throughput_gflops: 10.0,     // Basic throughput
                latency_us: 25.0,            // Lowest latency
                memory_bandwidth_gbps: 20.0,
                power_efficiency: 1.0,       // Most efficient
            },
        }
    }
}

/// Performance characteristics for a backend
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PerformanceCharacteristics {
    /// Throughput in GFLOPS
    pub throughput_gflops: f64,
    /// Latency in microseconds
    pub latency_us: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Power efficiency (0.0 to 1.0)
    pub power_efficiency: f64,
}

/// Dispatch strategy for selecting acceleration backend
#[derive(Clone, Debug)]
pub enum DispatchStrategy {
    /// Always use the highest priority available backend
    HighestPriority,
    /// Use the best performing backend based on profiling
    BestPerformance,
    /// Prefer low latency operations
    LowLatency,
    /// Prefer high throughput operations
    HighThroughput,
    /// Minimize memory usage
    LowMemory,
    /// Custom strategy with user-defined priorities
    Custom { priorities: HashMap<AccelerationBackend, u8> },
    /// Force use of a specific backend
    ForceBackend(AccelerationBackend),
}

impl Default for DispatchStrategy {
    fn default() -> Self {
        DispatchStrategy::HighestPriority
    }
}

/// Operation types for dispatch optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Matrix multiplication
    MatMul,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Tensor transpose
    Transpose,
    /// Tensor reshape
    Reshape,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
    /// Activation functions
    Activation,
    /// Convolution operations
    Convolution,
    /// Normalization operations
    Normalization,
    /// Quantization operations
    Quantization,
}

impl OperationType {
    /// Get the computational intensity (FLOPS per byte) for this operation
    pub fn computational_intensity(&self) -> f64 {
        match self {
            OperationType::MatMul => 4.0,           // High compute intensity
            OperationType::Convolution => 3.5,     // High compute intensity
            OperationType::Activation => 1.0,      // Medium compute intensity
            OperationType::Normalization => 2.0,   // Medium compute intensity
            OperationType::Add | OperationType::Mul | OperationType::Div => 0.25, // Low compute intensity
            OperationType::Transpose | OperationType::Reshape => 0.1, // Memory-bound
            OperationType::Reduction => 1.5,       // Medium compute intensity
            OperationType::Quantization => 0.5,    // Low-medium compute intensity
        }
    }

    /// Get the preferred backend for this operation type
    pub fn preferred_backend(&self) -> AccelerationBackend {
        match self {
            OperationType::MatMul | OperationType::Convolution => {
                if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
                    AccelerationBackend::MLX
                } else {
                    AccelerationBackend::Metal
                }
            },
            OperationType::Add | OperationType::Mul | OperationType::Div => {
                AccelerationBackend::SIMD
            },
            OperationType::Transpose | OperationType::Reshape => {
                AccelerationBackend::CPU
            },
            OperationType::Reduction | OperationType::Activation => {
                AccelerationBackend::SIMD
            },
            OperationType::Normalization | OperationType::Quantization => {
                AccelerationBackend::SIMD
            },
        }
    }
}

/// Performance requirements for operations
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency in microseconds
    pub max_latency_us: Option<u64>,
    /// Minimum required throughput in GFLOPS
    pub min_throughput_gflops: Option<f64>,
    /// Maximum acceptable memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    /// Whether to prioritize latency over throughput
    pub prefer_low_latency: bool,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency_us: None,
            min_throughput_gflops: None,
            max_memory_bytes: None,
            prefer_low_latency: false,
        }
    }
}

/// Operation context for dispatch decisions
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OperationContext {
    /// Type of operation
    pub operation_type: OperationType,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Data type
    pub dtype: BitNetDType,
    /// Performance requirements
    pub requirements: PerformanceRequirements,
}

impl OperationContext {
    pub fn new(operation_type: OperationType, input_shapes: Vec<Vec<usize>>, dtype: BitNetDType) -> Self {
        Self {
            operation_type,
            input_shapes,
            dtype,
            requirements: PerformanceRequirements::default(),
        }
    }

    /// Calculate the operation complexity score
    pub fn complexity_score(&self) -> f64 {
        let total_elements: usize = self.input_shapes.iter()
            .map(|hape| shape.iter().product::<usize>())
            .sum();

        let intensity = self.operation_type.computational_intensity();
        (total_elements as f64) * intensity
    }

    /// Estimate memory usage for this operation
    pub fn estimated_memory_bytes(&self) -> usize {
        let total_elements: usize = self.input_shapes.iter()
            .map(|hape| shape.iter().product::<usize>())
            .sum();

        let dtype_size = match self.dtype {
            BitNetDType::F32 => 4,
            BitNetDType::F16 => 2,
            BitNetDType::I8 | BitNetDType::U8 => 1,
            BitNetDType::I16 | BitNetDType::U16 => 2,
            BitNetDType::I32 | BitNetDType::U32 => 4,
            BitNetDType::Bool => 1,
        };

        // Account for input, output, and temporary storage
        total_elements * dtype_size * 3
    }
}

/// Backend selection result
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BackendSelection {
    /// Selected backend
    pub backend: AccelerationBackend,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated performance characteristics
    pub estimated_performance: PerformanceCharacteristics,
    /// Reason for selection
    pub selection_reason: String,
}

/// Main operation dispatcher
#[allow(dead_code)]
pub struct OperationDispatcher {
    /// Available acceleration backends
    backends: RwLock<HashMap<AccelerationBackend, Box<dyn AccelerationBackendImpl + Send + Sync>>>,
    /// Acceleration selector for backend choice
    selector: Arc<AutoAccelerationSelector>,
    /// Current dispatch strategy
    strategy: RwLock<DispatchStrategy>,
    /// Performance history for learning
    performance_history: RwLock<HashMap<String, AccelerationMetrics>>,
    /// Backend availability cache
    backend_availability: RwLock<HashMap<AccelerationBackend, bool>>,
}

impl OperationDispatcher {
    /// Create new operation dispatcher
    pub fn new(selector: Arc<AutoAccelerationSelector>) -> AccelerationResult<Self> {
        let dispatcher = Self {
            backends: RwLock::new(HashMap::new()),
            selector,
            strategy: RwLock::new(DispatchStrategy::default()),
            performance_history: RwLock::new(HashMap::new()),
            backend_availability: RwLock::new(HashMap::new()),
        };

        Ok(dispatcher)
    }

    /// Add a backend to the dispatcher
    pub fn add_backend(&self, backend: AccelerationBackend, implementation: Box<dyn AccelerationBackendImpl + Send + Sync>) -> AccelerationResult<()> {
        let mut backends = self.backends.write().unwrap();
        backends.insert(backend, implementation);

        // Update availability cache
        let mut availability = self.backend_availability.write().unwrap();
        availability.insert(backend, backend.is_platform_supported());

        #[cfg(feature = "tracing")]
        info!("Added acceleration backend: {}", backend);

        Ok(())
    }

    /// Set dispatch strategy
    pub fn set_strategy(&self, strategy: DispatchStrategy) {
        let mut current_strategy = self.strategy.write().unwrap();
        *current_strategy = strategy;

        #[cfg(feature = "tracing")]
        debug!("Updated dispatch strategy: {:?}", current_strategy);
    }

    /// Select best backend for an operation
    pub fn select_backend(&self, context: &OperationContext) -> AccelerationResult<BackendSelection> {
        let strategy = self.strategy.read().unwrap();
        let availability = self.backend_availability.read().unwrap();

        match &*strategy {
            DispatchStrategy::HighestPriority => {
                self.select_highest_priority_backend(context, &availability)
            },
            DispatchStrategy::BestPerformance => {
                self.select_best_performance_backend(context, &availability)
            },
            DispatchStrategy::LowLatency => {
                self.select_low_latency_backend(context, &availability)
            },
            DispatchStrategy::HighThroughput => {
                self.select_high_throughput_backend(context, &availability)
            },
            DispatchStrategy::LowMemory => {
                self.select_low_memory_backend(context, &availability)
            },
            DispatchStrategy::Custom { priorities } => {
                self.select_custom_backend(context, &availability, priorities)
            },
            DispatchStrategy::ForceBackend(backend) => {
                if availability.get(backend).unwrap_or(&false) {
                    Ok(BackendSelection {
                        backend: *backend,
                        confidence: 1.0,
                        estimated_performance: backend.performance_characteristics(),
                        selection_reason: format!("Forced backend: {}", backend),
                    })
                } else {
                    Err(AccelerationError::BackendNotAvailable {
                        backend: backend.to_string()
                    })
                }
            },
        }
    }

    /// Dispatch a matrix multiplication operation
    pub fn dispatch_matmul(&self, a: &BitNetTensor, b: &BitNetTensor, context: Option<&OperationContext>) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)> {
        let context = context.cloned().unwrap_or_else(|| {
            OperationContext::new(
                OperationType::MatMul,
                vec![a.shape().dims().to_vec(), b.shape().dims().to_vec()],
                a.dtype()
            )
        });

        let selection = self.select_backend(&context)?;
        let backends = self.backends.read().unwrap();

        if let Some(backend_impl) = backends.get(&selection.backend) {
            let result = backend_impl.matmul(a, b)?;

            // Record performance for learning
            self.record_performance(&context, &selection.backend, &result.1);

            #[cfg(feature = "tracing")]
            debug!("Dispatched matmul to {} backend", selection.backend);

            Ok(result)
        } else {
            Err(AccelerationError::BackendNotAvailable {
                backend: selection.backend.to_string()
            })
        }
    }

    /// Dispatch an element-wise addition operation
    pub fn dispatch_add(&self, a: &BitNetTensor, b: &BitNetTensor, context: Option<&OperationContext>) -> AccelerationResult<(BitNetTensor, AccelerationMetrics)> {
        let context = context.cloned().unwrap_or_else(|| {
            OperationContext::new(
                OperationType::Add,
                vec![a.shape().dims().to_vec(), b.shape().dims().to_vec()],
                a.dtype()
            )
        });

        let selection = self.select_backend(&context)?;
        let backends = self.backends.read().unwrap();

        if let Some(backend_impl) = backends.get(&selection.backend) {
            let result = backend_impl.add(a, b)?;

            // Record performance for learning
            self.record_performance(&context, &selection.backend, &result.1);

            #[cfg(feature = "tracing")]
            debug!("Dispatched add to {} backend", selection.backend);

            Ok(result)
        } else {
            Err(AccelerationError::BackendNotAvailable {
                backend: selection.backend.to_string()
            })
        }
    }

    /// Get available backends
    pub fn get_available_backends(&self) -> Vec<AccelerationBackend> {
        let availability = self.backend_availability.read().unwrap();
        availability.iter()
            .filter_map(|(backend, available)| if *available { Some(*backend) } else { None })
            .collect()
    }

    /// Get performance history for analysis
    pub fn get_performance_history(&self) -> HashMap<String, AccelerationMetrics> {
        self.performance_history.read().unwrap().clone()
    }

    // Private helper methods

    fn select_highest_priority_backend(&self, _context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>) -> AccelerationResult<BackendSelection> {
        let available_backends: Vec<AccelerationBackend> = availability.iter()
            .filter_map(|(backend, available)| if *available { Some(*backend) } else { None })
            .collect();

        if available_backends.is_empty() {
            return Err(AccelerationError::BackendNotAvailable {
                backend: "any".to_string()
            });
        }

        let selected = available_backends.into_iter()
            .max_by_key(|backend| backend.priority())
            .unwrap();

        Ok(BackendSelection {
            backend: selected,
            confidence: 0.9,
            estimated_performance: selected.performance_characteristics(),
            selection_reason: "Highest priority available backend".to_string(),
        })
    }

    fn select_best_performance_backend(&self, context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>) -> AccelerationResult<BackendSelection> {
        let history = self.performance_history.read().unwrap();
        let operation_key = format!("{:?}_{}", context.operation_type, context.complexity_score() as u64);

        // If we have historical data, use it
        if let Some(best_metrics) = history.get(&operation_key) {
            if availability.get(&best_metrics.backend).unwrap_or(&false) {
                return Ok(BackendSelection {
                    backend: best_metrics.backend,
                    confidence: 0.95,
                    estimated_performance: best_metrics.backend.performance_characteristics(),
                    selection_reason: "Based on historical performance data".to_string(),
                });
            }
        }

        // Fall back to preferred backend for operation type
        let preferred = context.operation_type.preferred_backend();
        if availability.get(&preferred).unwrap_or(&false) {
            Ok(BackendSelection {
                backend: preferred,
                confidence: 0.8,
                estimated_performance: preferred.performance_characteristics(),
                selection_reason: "Preferred backend for operation type".to_string(),
            })
        } else {
            // Fall back to highest priority
            self.select_highest_priority_backend(context, availability)
        }
    }

    fn select_low_latency_backend(&self, context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>) -> AccelerationResult<BackendSelection> {
        let available_backends: Vec<AccelerationBackend> = availability.iter()
            .filter_map(|(backend, available)| if *available { Some(*backend) } else { None })
            .collect();

        if available_backends.is_empty() {
            return Err(AccelerationError::BackendNotAvailable {
                backend: "any".to_string()
            });
        }

        let selected = available_backends.into_iter()
            .min_by(|a, b| a.performance_characteristics().latency_us
                .partial_cmp(&b.performance_characteristics().latency_us)
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Ok(BackendSelection {
            backend: selected,
            confidence: 0.85,
            estimated_performance: selected.performance_characteristics(),
            selection_reason: "Lowest latency backend".to_string(),
        })
    }

    fn select_high_throughput_backend(&self, context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>) -> AccelerationResult<BackendSelection> {
        let available_backends: Vec<AccelerationBackend> = availability.iter()
            .filter_map(|(backend, available)| if *available { Some(*backend) } else { None })
            .collect();

        if available_backends.is_empty() {
            return Err(AccelerationError::BackendNotAvailable {
                backend: "any".to_string()
            });
        }

        let selected = available_backends.into_iter()
            .max_by(|a, b| a.performance_characteristics().throughput_gflops
                .partial_cmp(&b.performance_characteristics().throughput_gflops)
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Ok(BackendSelection {
            backend: selected,
            confidence: 0.85,
            estimated_performance: selected.performance_characteristics(),
            selection_reason: "Highest throughput backend".to_string(),
        })
    }

    fn select_low_memory_backend(&self, context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>) -> AccelerationResult<BackendSelection> {
        // For memory-bound operations, prefer CPU
        let preferred_backends = [AccelerationBackend::CPU, AccelerationBackend::SIMD, AccelerationBackend::Metal, AccelerationBackend::MLX];

        for &backend in &preferred_backends {
            if availability.get(&backend).unwrap_or(&false) {
                return Ok(BackendSelection {
                    backend,
                    confidence: 0.8,
                    estimated_performance: backend.performance_characteristics(),
                    selection_reason: "Low memory usage priority".to_string(),
                });
            }
        }

        Err(AccelerationError::BackendNotAvailable {
            backend: "any".to_string()
        })
    }

    fn select_custom_backend(&self, context: &OperationContext, availability: &HashMap<AccelerationBackend, bool>, priorities: &HashMap<AccelerationBackend, u8>) -> AccelerationResult<BackendSelection> {
        let available_backends: Vec<AccelerationBackend> = availability.iter()
            .filter_map(|(backend, available)| if *available { Some(*backend) } else { None })
            .collect();

        if available_backends.is_empty() {
            return Err(AccelerationError::BackendNotAvailable {
                backend: "any".to_string()
            });
        }

        let selected = available_backends.into_iter()
            .max_by_key(|backend| priorities.get(backend).unwrap_or(&0))
            .unwrap();

        Ok(BackendSelection {
            backend: selected,
            confidence: 0.9,
            estimated_performance: selected.performance_characteristics(),
            selection_reason: "Custom priority-based selection".to_string(),
        })
    }

    fn record_performance(&self, context: &OperationContext, backend: &AccelerationBackend, metrics: &AccelerationMetrics) {
        let operation_key = format!("{:?}_{}", context.operation_type, context.complexity_score() as u64);
        let mut history = self.performance_history.write().unwrap();
        history.insert(operation_key, metrics.clone());

        #[cfg(feature = "tracing")]
        debug!("Recorded performance for {} on {}: {:.2} GFLOPS",
               context.operation_type, backend, metrics.throughput_gflops);
    }
}

/// Create a new operation dispatcher with automatic backend detection
pub fn create_operation_dispatcher() -> AccelerationResult<OperationDispatcher> {
    // Create selector using existing auto-selection logic
    let selector = Arc::new(AutoAccelerationSelector::new()?);

    let dispatcher = OperationDispatcher::new(selector)?;

    // Add available backends
    #[cfg(feature = "mlx")]
    if let Some(mlx_backend) = crate::tensor::acceleration::mlx::create_mlx_accelerator()? {
        dispatcher.add_backend(AccelerationBackend::MLX, mlx_backend)?;
    }

    #[cfg(feature = "metal")]
    if let Some(metal_backend) = crate::tensor::acceleration::metal::create_metal_accelerator()? {
        dispatcher.add_backend(AccelerationBackend::Metal, metal_backend)?;
    }

    if let Some(simd_backend) = crate::tensor::acceleration::simd::create_simd_accelerator()? {
        dispatcher.add_backend(AccelerationBackend::SIMD, simd_backend)?;
    }

    #[cfg(feature = "tracing")]
    info!("Operation dispatcher created with {} backends", dispatcher.get_available_backends().len());

    Ok(dispatcher)
}
