//! Comprehensive Tensor Operations Demo
//!
//! This example demonstrates the complete BitNet tensor system including:
//! - Tensor creation with various data types and shapes
//! - Mathematical operations (arithmetic, linear algebra, reductions)
//! - Memory pool integration and efficiency
//! - Device abstraction and migration
//! - Performance benchmarking and optimization
//! - Error handling and recovery

use std::time::{Duration, Instant};
use std::sync::Arc;

use bitnet_core::memory::{
    HybridMemoryPool, MemoryPoolConfig, TrackingConfig, TrackingLevel
};
use bitnet_core::memory::tensor::{BitNetTensor, BitNetDType};
use bitnet_core::device::{
    get_cpu_device, auto_select_device, is_metal_available, get_metal_device
};

// =============================================================================
// Demo Configuration
// =============================================================================

struct DemoConfig {
    enable_performance_benchmarks: bool,
    enable_device_migration_demo: bool,
    enable_memory_efficiency_demo: bool,
    tensor_sizes: Vec<Vec<usize>>,
    data_types: Vec<BitNetDType>,
    benchmark_iterations: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            enable_performance_benchmarks: true,
            enable_device_migration_demo: true,
            enable_memory_efficiency_demo: true,
            tensor_sizes: vec![
                vec![64, 64],           // Small 2D
                vec![256, 256],         // Medium 2D
                vec![512, 512],         // Large 2D
                vec![32, 32, 32],       // 3D cube
                vec![16, 16, 16, 16],   // 4D tensor
                vec![1024],             // 1D vector
                vec![2048, 1024],       // Asymmetric 2D
            ],
            data_types: vec![
                BitNetDType::Float32,
                BitNetDType::Float16,
                BitNetDType::Int8,
                BitNetDType::Int16,
                BitNetDType::Int32,
            ],
            benchmark_iterations: 1000,
        }
    }
}

// =============================================================================
// Main Demo Function
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet Comprehensive Tensor Operations Demo");
    println!("==============================================");

    let config = DemoConfig::default();
    let pool = create_optimized_memory_pool()?;

    // Demo sections
    demo_tensor_creation_and_basic_ops(&pool, &config)?;
    demo_mathematical_operations(&pool, &config)?;
    demo_memory_pool_integration(&pool, &config)?;

    if config.enable_device_migration_demo {
        demo_device_abstraction_and_migration(&pool, &config)?;
    }

    if config.enable_performance_benchmarks {
        demo_performance_benchmarking(&pool, &config)?;
    }

    if config.enable_memory_efficiency_demo {
        demo_memory_efficiency(&pool, &config)?;
    }

    demo_error_handling_and_recovery(&pool, &config)?;
    demo_advanced_features(&pool, &config)?;

    println!("\n🎉 Demo completed successfully!");
    print_final_statistics(&pool)?;

    Ok(())
}

// =============================================================================
// Tensor Creation and Basic Operations Demo
// =============================================================================

fn demo_tensor_creation_and_basic_ops(
    pool: &HybridMemoryPool,
    config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 1. Tensor Creation and Basic Operations");
    println!("-----------------------------------------");

    let device = auto_select_device();
    println!("Selected device: {:?}", device);

    // Create tensors with different shapes and data types
    let mut tensors = Vec::new();

    for (i, shape) in config.tensor_sizes.iter().enumerate() {
        let dtype = &config.data_types[i % config.data_types.len()];

        println!("Creating tensor: shape={:?}, dtype={:?}", shape, dtype);

        let start_time = Instant::now();

        // Create zero tensor
        let zeros_tensor = BitNetTensor::zeros_with_pool(
            shape.clone(),
            dtype.clone(),
            device.clone(),
            pool.clone()
        )?;

        // Create ones tensor
        let ones_tensor = BitNetTensor::ones_with_pool(
            shape.clone(),
            dtype.clone(),
            device.clone(),
            pool.clone()
        )?;

        let creation_time = start_time.elapsed();

        println!("  ✅ Created in {:?}", creation_time);

        // Demonstrate basic properties
        println!("  📏 Shape: {:?}", zeros_tensor.shape());
        println!("  🔢 Data type: {:?}", zeros_tensor.dtype());
        println!("  💻 Device: {:?}", zeros_tensor.device());
        println!("  📊 Element count: {}", zeros_tensor.numel());
        println!("  💾 Memory size: {} bytes", zeros_tensor.memory_size());

        tensors.push((zeros_tensor, ones_tensor));
        println!();
    }

    // Demonstrate tensor cloning and views
    let (ref tensor, _) = tensors[0];
    println!("🔄 Demonstrating tensor cloning and views:");

    let cloned = tensor.clone();
    println!("  ✅ Cloned tensor: shape={:?}", cloned.shape());

    // Demonstrate basic indexing (if implemented)
    if tensor.shape().len() >= 2 {
        match tensor.slice(0, 0..10) {
            Ok(slice) => println!("  🔍 Sliced tensor: shape={:?}", slice.shape()),
            Err(e) => println!("  ❌ Slicing not implemented: {}", e),
        }
    }

    println!("✅ Basic tensor operations demo completed");
    Ok(())
}

// =============================================================================
// Mathematical Operations Demo
// =============================================================================

fn demo_mathematical_operations(
    pool: &HybridMemoryPool,
    config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 2. Mathematical Operations");
    println!("----------------------------");

    let device = auto_select_device();
    let shape = vec![256, 256];

    // Create test tensors
    let tensor_a = BitNetTensor::ones_with_pool(
        shape.clone(),
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor_b = BitNetTensor::from_values_with_pool(
        vec![2.0_f32; 256 * 256],
        shape.clone(),
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("Created test tensors: A (ones) and B (twos)");

    // Arithmetic Operations
    println!("\n🔢 Arithmetic Operations:");

    let start_time = Instant::now();
    let add_result = tensor_a.add(&tensor_b)?;
    let add_time = start_time.elapsed();
    println!("  ➕ Addition: A + B = result, took {:?}", add_time);
    println!("     Sample result value: {}", add_result.get_scalar(0, 0)?);

    let start_time = Instant::now();
    let sub_result = tensor_b.sub(&tensor_a)?;
    let sub_time = start_time.elapsed();
    println!("  ➖ Subtraction: B - A = result, took {:?}", sub_time);

    let start_time = Instant::now();
    let mul_result = tensor_a.mul(&tensor_b)?;
    let mul_time = start_time.elapsed();
    println!("  ✖️  Multiplication: A * B = result, took {:?}", mul_time);

    let start_time = Instant::now();
    let div_result = tensor_b.div(&tensor_a)?;
    let div_time = start_time.elapsed();
    println!("  ➗ Division: B / A = result, took {:?}", div_time);

    // Linear Algebra Operations
    println!("\n📐 Linear Algebra Operations:");

    let start_time = Instant::now();
    let matmul_result = tensor_a.matmul(&tensor_b)?;
    let matmul_time = start_time.elapsed();
    println!("  🔶 Matrix multiplication: A @ B, took {:?}", matmul_time);
    println!("     Result shape: {:?}", matmul_result.shape());

    let start_time = Instant::now();
    let transpose_result = tensor_a.transpose(0, 1)?;
    let transpose_time = start_time.elapsed();
    println!("  🔄 Transpose: A.T, took {:?}", transpose_time);
    println!("     Result shape: {:?}", transpose_result.shape());

    // Reduction Operations
    println!("\n📊 Reduction Operations:");

    let start_time = Instant::now();
    let sum_result = tensor_a.sum(None, false)?;
    let sum_time = start_time.elapsed();
    println!("  ➕ Sum: sum(A), took {:?}", sum_time);
    println!("     Sum value: {}", sum_result.get_scalar_value()?);

    let start_time = Instant::now();
    let mean_result = tensor_a.mean(None, false)?;
    let mean_time = start_time.elapsed();
    println!("  📊 Mean: mean(A), took {:?}", mean_time);
    println!("     Mean value: {}", mean_result.get_scalar_value()?);

    let start_time = Instant::now();
    let max_result = tensor_b.max(None, false)?;
    let max_time = start_time.elapsed();
    println!("  ⬆️  Maximum: max(B), took {:?}", max_time);

    let start_time = Instant::now();
    let min_result = tensor_a.min(None, false)?;
    let min_time = start_time.elapsed();
    println!("  ⬇️  Minimum: min(A), took {:?}", min_time);

    // Broadcasting Operations
    println!("\n📡 Broadcasting Operations:");

    let scalar_tensor = BitNetTensor::scalar_with_pool(
        5.0_f32,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let broadcast_add = tensor_a.add(&scalar_tensor)?;
    println!("  📡 Broadcast addition: A + scalar");
    println!("     Result shape: {:?}", broadcast_add.shape());

    // In-place Operations
    println!("\n🔄 In-place Operations:");

    let mut inplace_tensor = tensor_a.clone();
    let start_time = Instant::now();
    inplace_tensor.add_(&scalar_tensor)?;
    let inplace_time = start_time.elapsed();
    println!("  🔄 In-place addition: A += scalar, took {:?}", inplace_time);

    println!("✅ Mathematical operations demo completed");
    Ok(())
}

// =============================================================================
// Memory Pool Integration Demo
// =============================================================================

fn demo_memory_pool_integration(
    pool: &HybridMemoryPool,
    config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 3. Memory Pool Integration");
    println!("----------------------------");

    let device = auto_select_device();

    // Show initial memory state
    let initial_metrics = pool.get_metrics();
    println!("Initial memory state:");
    println!("  📊 Total allocated: {} bytes", initial_metrics.total_allocated);
    println!("  📦 Active allocations: {}", initial_metrics.active_allocations);
    println!("  🔄 Pool utilization: {:.1}%", initial_metrics.pool_utilization() * 100.0);

    println!("\n📈 Creating tensors to demonstrate pool behavior:");

    let mut tensors = Vec::new();

    // Create tensors of increasing size
    let sizes = vec![64, 128, 256, 512, 1024];
    for (i, size) in sizes.iter().enumerate() {
        let tensor = BitNetTensor::zeros_with_pool(
            vec![*size, *size],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;

        tensors.push(tensor);

        let metrics = pool.get_metrics();
        println!("  Tensor {}: {}x{} - Pool: {} bytes, {} allocations",
                 i + 1, size, size, metrics.total_allocated, metrics.active_allocations);
    }

    // Demonstrate memory reuse
    println!("\n🔄 Demonstrating memory reuse:");

    // Clear half the tensors
    let half_point = tensors.len() / 2;
    tensors.truncate(half_point);

    let after_cleanup_metrics = pool.get_metrics();
    println!("After cleanup:");
    println!("  📊 Total allocated: {} bytes", after_cleanup_metrics.total_allocated);
    println!("  📦 Active allocations: {}", after_cleanup_metrics.active_allocations);

    // Create new tensors (should reuse memory)
    for i in 0..3 {
        let tensor = BitNetTensor::ones_with_pool(
            vec![256, 256],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;
        tensors.push(tensor);

        let metrics = pool.get_metrics();
        println!("  New tensor {}: Pool: {} bytes, {} allocations",
                 i + 1, metrics.total_allocated, metrics.active_allocations);
    }

    // Show memory efficiency
    let final_metrics = pool.get_metrics();
    let memory_growth = final_metrics.total_allocated - initial_metrics.total_allocated;
    let tensor_data_size: usize = tensors.iter()
        .map(|t| t.memory_size())
        .sum();

    let efficiency = tensor_data_size as f64 / memory_growth as f64;
    println!("\n📊 Memory Efficiency Analysis:");
    println!("  💾 Raw tensor data: {} bytes", tensor_data_size);
    println!("  📈 Total memory growth: {} bytes", memory_growth);
    println!("  ⚡ Efficiency: {:.1}%", efficiency * 100.0);

    println!("✅ Memory pool integration demo completed");
    Ok(())
}

// =============================================================================
// Device Abstraction and Migration Demo
// =============================================================================

fn demo_device_abstraction_and_migration(
    pool: &HybridMemoryPool,
    _config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💻 4. Device Abstraction and Migration");
    println!("-------------------------------------");

    // Show available devices
    let cpu_device = get_cpu_device();
    println!("Available devices:");
    println!("  🖥️  CPU: {:?}", cpu_device);

    let metal_available = is_metal_available();
    println!("  🔥 Metal GPU: {}", if metal_available { "Available" } else { "Not available" });

    let auto_device = auto_select_device();
    println!("  🎯 Auto-selected: {:?}", auto_device);

    // Create tensor on CPU
    let cpu_tensor = BitNetTensor::zeros_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    println!("\n📊 Created tensor on CPU:");
    println!("  📏 Shape: {:?}", cpu_tensor.shape());
    println!("  💻 Device: {:?}", cpu_tensor.device());

    // Perform operations on CPU
    let start_time = Instant::now();
    let cpu_result = cpu_tensor.add(&cpu_tensor)?;
    let cpu_time = start_time.elapsed();
    println!("  ⚡ CPU operation time: {:?}", cpu_time);

    // Test Metal migration if available
    if metal_available {
        println!("\n🔥 Testing Metal GPU migration:");

        let metal_device = get_metal_device()?;

        let start_time = Instant::now();
        let metal_tensor = cpu_tensor.to_device(&metal_device)?;
        let migration_time = start_time.elapsed();

        println!("  🔄 Migration CPU -> Metal: {:?}", migration_time);
        println!("  📏 Shape preserved: {:?}", metal_tensor.shape());
        println!("  💻 New device: {:?}", metal_tensor.device());

        // Perform operations on Metal
        let start_time = Instant::now();
        let metal_result = metal_tensor.add(&metal_tensor)?;
        let metal_time = start_time.elapsed();
        println!("  ⚡ Metal operation time: {:?}", metal_time);

        if cpu_time > Duration::ZERO && metal_time > Duration::ZERO {
            let speedup = cpu_time.as_nanos() as f64 / metal_time.as_nanos() as f64;
            println!("  🚀 Speedup: {:.2}x", speedup);
        }

        // Migrate back to CPU
        let start_time = Instant::now();
        let cpu_tensor_back = metal_result.to_device(&cpu_device)?;
        let migration_back_time = start_time.elapsed();

        println!("  🔄 Migration Metal -> CPU: {:?}", migration_back_time);
        println!("  📏 Shape preserved: {:?}", cpu_tensor_back.shape());
    } else {
        println!("\n⚠️  Metal GPU not available, skipping GPU migration demo");
    }

    // Test automatic device selection
    println!("\n🎯 Testing automatic device selection:");

    let auto_tensor = BitNetTensor::zeros_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        auto_device.clone(),
        pool.clone()
    )?;

    println!("  📊 Auto-selected device: {:?}", auto_tensor.device());
    println!("  ✅ Tensor created successfully");

    println!("✅ Device abstraction demo completed");
    Ok(())
}

// =============================================================================
// Performance Benchmarking Demo
// =============================================================================

fn demo_performance_benchmarking(
    pool: &HybridMemoryPool,
    config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ 5. Performance Benchmarking");
    println!("-----------------------------");

    let device = auto_select_device();

    // Benchmark tensor creation
    println!("🏗️  Benchmarking tensor creation:");

    let sizes = vec![64, 128, 256, 512];
    for size in sizes {
        let iterations = config.benchmark_iterations / 10; // Fewer iterations for large tensors

        let start_time = Instant::now();
        let mut tensors = Vec::new();

        for _ in 0..iterations {
            let tensor = BitNetTensor::zeros_with_pool(
                vec![size, size],
                BitNetDType::Float32,
                device.clone(),
                pool.clone()
            )?;
            tensors.push(tensor);
        }

        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations as u32;

        println!("  📏 {}x{}: {:.2}μs/tensor ({} tensors)",
                 size, size, avg_time.as_micros(), iterations);

        // Validate against target: <100μs per tensor
        if avg_time.as_micros() > 100 {
            println!("    ⚠️  Above target (100μs)");
        } else {
            println!("    ✅ Within target");
        }
    }

    // Benchmark mathematical operations
    println!("\n🧮 Benchmarking mathematical operations:");

    let tensor_a = BitNetTensor::ones_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor_b = BitNetTensor::ones_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let operations = vec![
        ("Addition", "add"),
        ("Multiplication", "mul"),
        ("Matrix Multiplication", "matmul"),
        ("Transpose", "transpose"),
    ];

    for (name, _op) in operations {
        let iterations = 100;
        let start_time = Instant::now();

        for _ in 0..iterations {
            match name {
                "Addition" => { let _ = tensor_a.add(&tensor_b)?; },
                "Multiplication" => { let _ = tensor_a.mul(&tensor_b)?; },
                "Matrix Multiplication" => { let _ = tensor_a.matmul(&tensor_b)?; },
                "Transpose" => { let _ = tensor_a.transpose(0, 1)?; },
                _ => {}
            }
        }

        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations as u32;

        println!("  {} (512x512): {:.2}ms/op", name, avg_time.as_micros() as f64 / 1000.0);
    }

    // Benchmark device migration (if Metal available)
    if is_metal_available() {
        println!("\n🔄 Benchmarking device migration:");

        let metal_device = get_metal_device()?;
        let cpu_device = get_cpu_device();

        let cpu_tensor = BitNetTensor::ones_with_pool(
            vec![256, 256],
            BitNetDType::Float32,
            cpu_device.clone(),
            pool.clone()
        )?;

        let iterations = 50;

        // CPU to Metal
        let start_time = Instant::now();
        for _ in 0..iterations {
            let _ = cpu_tensor.to_device(&metal_device)?;
        }
        let cpu_to_metal_time = start_time.elapsed() / iterations as u32;

        let metal_tensor = cpu_tensor.to_device(&metal_device)?;

        // Metal to CPU
        let start_time = Instant::now();
        for _ in 0..iterations {
            let _ = metal_tensor.to_device(&cpu_device)?;
        }
        let metal_to_cpu_time = start_time.elapsed() / iterations as u32;

        println!("  🖥️  -> 🔥 (CPU to Metal): {:.2}ms", cpu_to_metal_time.as_micros() as f64 / 1000.0);
        println!("  🔥 -> 🖥️  (Metal to CPU): {:.2}ms", metal_to_cpu_time.as_micros() as f64 / 1000.0);
    }

    println!("✅ Performance benchmarking demo completed");
    Ok(())
}

// =============================================================================
// Memory Efficiency Demo
// =============================================================================

fn demo_memory_efficiency(
    pool: &HybridMemoryPool,
    _config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 6. Memory Efficiency Analysis");
    println!("-------------------------------");

    let device = auto_select_device();
    let initial_metrics = pool.get_metrics();

    println!("Initial memory state:");
    println!("  📊 Allocated: {} bytes", initial_metrics.total_allocated);
    println!("  🆓 Available: {} bytes", initial_metrics.available_memory);

    // Test memory efficiency with different tensor sizes
    let test_configs = vec![
        ("Small tensors", vec![64, 64], 100),
        ("Medium tensors", vec![256, 256], 50),
        ("Large tensors", vec![512, 512], 20),
    ];

    for (name, shape, count) in test_configs {
        println!("\n🔬 Testing {}:", name);

        let tensor_size = shape.iter().product::<usize>() * 4; // f32 = 4 bytes
        let expected_data = tensor_size * count;

        let before_metrics = pool.get_metrics();

        // Create tensors
        let mut tensors = Vec::new();
        for _ in 0..count {
            let tensor = BitNetTensor::zeros_with_pool(
                shape.clone(),
                BitNetDType::Float32,
                device.clone(),
                pool.clone()
            )?;
            tensors.push(tensor);
        }

        let after_metrics = pool.get_metrics();
        let actual_allocated = after_metrics.total_allocated - before_metrics.total_allocated;

        let efficiency = expected_data as f64 / actual_allocated as f64;
        let overhead = actual_allocated - expected_data;

        println!("  📊 Expected data: {} bytes", expected_data);
        println!("  📈 Actual allocated: {} bytes", actual_allocated);
        println!("  💡 Efficiency: {:.1}%", efficiency * 100.0);
        println!("  📏 Overhead: {} bytes ({:.1}%)",
                 overhead, (overhead as f64 / expected_data as f64) * 100.0);

        if efficiency > 0.90 {
            println!("  ✅ Excellent efficiency (>90%)");
        } else if efficiency > 0.80 {
            println!("  👍 Good efficiency (>80%)");
        } else {
            println!("  ⚠️  Low efficiency (<80%)");
        }

        // Clear tensors and check memory reclamation
        tensors.clear();
        std::thread::sleep(std::time::Duration::from_millis(50)); // Allow cleanup

        let cleanup_metrics = pool.get_metrics();
        let reclaimed = after_metrics.total_allocated - cleanup_metrics.total_allocated;
        let reclaim_efficiency = reclaimed as f64 / actual_allocated as f64;

        println!("  🧹 Reclaimed: {} bytes ({:.1}%)", reclaimed, reclaim_efficiency * 100.0);
    }

    // Test memory fragmentation
    println!("\n🧩 Testing memory fragmentation:");

    // Create and destroy tensors in a pattern that could cause fragmentation
    let mut tensors = Vec::new();

    // Phase 1: Create many small tensors
    for i in 0..50 {
        let tensor = BitNetTensor::zeros_with_pool(
            vec![64, 64],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        )?;
        tensors.push(tensor);
    }

    // Phase 2: Remove every other tensor (create fragmentation)
    let mut i = 0;
    tensors.retain(|_| {
        i += 1;
        i % 2 == 0
    });

    // Phase 3: Try to create large tensors (should handle fragmentation)
    let fragmentation_metrics = pool.get_metrics();

    match BitNetTensor::zeros_with_pool(
        vec![512, 512],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    ) {
        Ok(_) => println!("  ✅ Successfully allocated large tensor despite fragmentation"),
        Err(e) => println!("  ❌ Failed to handle fragmentation: {}", e),
    }

    println!("✅ Memory efficiency demo completed");
    Ok(())
}

// =============================================================================
// Error Handling and Recovery Demo
// =============================================================================

fn demo_error_handling_and_recovery(
    pool: &HybridMemoryPool,
    _config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️  7. Error Handling and Recovery");
    println!("----------------------------------");

    let device = auto_select_device();

    // Test invalid shape handling
    println!("🔍 Testing invalid shape handling:");

    let invalid_shapes = vec![
        vec![], // Empty shape
        vec![0], // Zero dimension
        vec![100000000, 100000000], // Too large
    ];

    for (i, shape) in invalid_shapes.iter().enumerate() {
        match BitNetTensor::zeros_with_pool(
            shape.clone(),
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        ) {
            Ok(_) => println!("  ⚠️  Invalid shape {} unexpectedly succeeded: {:?}", i + 1, shape),
            Err(e) => println!("  ✅ Invalid shape {} properly rejected: {}", i + 1, e),
        }
    }

    // Test device compatibility
    println!("\n💻 Testing device compatibility:");

    let cpu_device = get_cpu_device();
    let cpu_tensor = BitNetTensor::ones_with_pool(
        vec![64, 64],
        BitNetDType::Float32,
        cpu_device.clone(),
        pool.clone()
    )?;

    if is_metal_available() {
        let metal_device = get_metal_device()?;
        let metal_tensor = BitNetTensor::ones_with_pool(
            vec![64, 64],
            BitNetDType::Float32,
            metal_device.clone(),
            pool.clone()
        )?;

        // Test cross-device operations (should handle automatically or error gracefully)
        match cpu_tensor.add(&metal_tensor) {
            Ok(result) => println!("  ✅ Cross-device operation handled: result on {:?}", result.device()),
            Err(e) => println!("  ℹ️  Cross-device operation rejected (expected): {}", e),
        }
    }

    // Test memory exhaustion recovery
    println!("\n💾 Testing memory exhaustion recovery:");

    let mut large_tensors = Vec::new();
    let mut allocation_count = 0;

    // Try to allocate until we hit memory limits
    loop {
        match BitNetTensor::zeros_with_pool(
            vec![1024, 1024],
            BitNetDType::Float32,
            device.clone(),
            pool.clone()
        ) {
            Ok(tensor) => {
                large_tensors.push(tensor);
                allocation_count += 1;

                if allocation_count > 100 { // Safety limit
                    println!("  ℹ️  Allocated 100 large tensors without hitting limits");
                    break;
                }
            }
            Err(e) => {
                println!("  ✅ Memory exhaustion handled gracefully after {} allocations: {}",
                         allocation_count, e);
                break;
            }
        }
    }

    // Test recovery after exhaustion
    large_tensors.clear();
    std::thread::sleep(std::time::Duration::from_millis(100));

    match BitNetTensor::zeros_with_pool(
        vec![256, 256],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    ) {
        Ok(_) => println!("  ✅ Successfully recovered after memory exhaustion"),
        Err(e) => println!("  ❌ Failed to recover: {}", e),
    }

    // Test operation error handling
    println!("\n🧮 Testing mathematical operation error handling:");

    let tensor1 = BitNetTensor::ones_with_pool(
        vec![64, 128],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor2 = BitNetTensor::ones_with_pool(
        vec![256, 64],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    // Test incompatible shapes for element-wise operations
    match tensor1.add(&tensor2) {
        Ok(_) => println!("  ⚠️  Incompatible shapes unexpectedly succeeded"),
        Err(e) => println!("  ✅ Incompatible shapes properly rejected: {}", e),
    }

    // Test invalid matrix multiplication
    let tensor3 = BitNetTensor::ones_with_pool(
        vec![64, 64],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    match tensor1.matmul(&tensor3) {
        Ok(_) => println!("  ✅ Matrix multiplication with compatible shapes succeeded"),
        Err(e) => println!("  ℹ️  Matrix multiplication error: {}", e),
    }

    println!("✅ Error handling and recovery demo completed");
    Ok(())
}

// =============================================================================
// Advanced Features Demo
// =============================================================================

fn demo_advanced_features(
    pool: &HybridMemoryPool,
    _config: &DemoConfig
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 8. Advanced Features");
    println!("----------------------");

    let device = auto_select_device();

    // Demonstrate tensor reshaping
    println!("🔄 Tensor reshaping and views:");

    let original_tensor = BitNetTensor::from_values_with_pool(
        (0..24).map(|i| i as f32).collect(),
        vec![24],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("  📊 Original shape: {:?}", original_tensor.shape());

    let reshaped = original_tensor.reshape(vec![4, 6])?;
    println!("  🔄 Reshaped to: {:?}", reshaped.shape());

    let reshaped_3d = original_tensor.reshape(vec![2, 3, 4])?;
    println!("  🔄 Reshaped to 3D: {:?}", reshaped_3d.shape());

    // Demonstrate tensor squeezing and unsqueezing
    let tensor_with_singleton = BitNetTensor::zeros_with_pool(
        vec![1, 64, 1, 32],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    println!("  📏 Tensor with singleton dims: {:?}", tensor_with_singleton.shape());

    let squeezed = tensor_with_singleton.squeeze()?;
    println!("  🗜️  Squeezed: {:?}", squeezed.shape());

    let unsqueezed = squeezed.unsqueeze(1)?;
    println!("  📈 Unsqueezed at dim 1: {:?}", unsqueezed.shape());

    // Demonstrate tensor concatenation
    println!("\n🔗 Tensor concatenation:");

    let tensor_a = BitNetTensor::ones_with_pool(
        vec![32, 64],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let tensor_b = BitNetTensor::zeros_with_pool(
        vec![32, 64],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let concatenated = BitNetTensor::concatenate(&[&tensor_a, &tensor_b], 0)?;
    println!("  📐 Concatenated along dim 0: {:?}", concatenated.shape());

    let concatenated_dim1 = BitNetTensor::concatenate(&[&tensor_a, &tensor_b], 1)?;
    println!("  📐 Concatenated along dim 1: {:?}", concatenated_dim1.shape());

    // Demonstrate tensor statistics
    println!("\n📊 Tensor statistics:");

    let random_tensor = BitNetTensor::random_with_pool(
        vec![128, 128],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let mean = random_tensor.mean(None, false)?;
    let std = random_tensor.std(None, false)?;
    let min_val = random_tensor.min(None, false)?;
    let max_val = random_tensor.max(None, false)?;

    println!("  📊 Random tensor stats:");
    println!("    Mean: {:.4}", mean.get_scalar_value()?);
    println!("    Std:  {:.4}", std.get_scalar_value()?);
    println!("    Min:  {:.4}", min_val.get_scalar_value()?);
    println!("    Max:  {:.4}", max_val.get_scalar_value()?);

    // Demonstrate tensor comparison operations
    println!("\n🔍 Tensor comparison operations:");

    let threshold = BitNetTensor::scalar_with_pool(
        0.5_f32,
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    let greater_mask = random_tensor.greater(&threshold)?;
    let num_greater = greater_mask.sum(None, false)?.get_scalar_value()? as i32;
    let total_elements = random_tensor.numel() as i32;

    println!("  🔍 Elements > 0.5: {} / {} ({:.1}%)",
             num_greater, total_elements,
             (num_greater as f32 / total_elements as f32) * 100.0);

    // Demonstrate gradient support (if implemented)
    println!("\n🎓 Gradient support:");

    let mut grad_tensor = BitNetTensor::ones_with_pool(
        vec![64, 64],
        BitNetDType::Float32,
        device.clone(),
        pool.clone()
    )?;

    grad_tensor.set_requires_grad(true)?;

    if grad_tensor.requires_grad() {
        println!("  ✅ Gradient tracking enabled");

        // Perform operations that would create computation graph
        let result = grad_tensor.mul(&grad_tensor)?;
        let loss = result.sum(None, false)?;

        println!("  📊 Computation graph created");

        // Note: Actual backward pass would require autograd implementation
        println!("  ℹ️  Backward pass implementation depends on autograd system");
    } else {
        println!("  ℹ️  Gradient tracking not yet implemented");
    }

    println!("✅ Advanced features demo completed");
    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

fn create_optimized_memory_pool() -> Result<HybridMemoryPool, Box<dyn std::error::Error>> {
    let config = MemoryPoolConfig {
        small_block_size: 64 * 1024,       // 64KB
        large_block_threshold: 1024 * 1024, // 1MB
        initial_pool_size: 32 * 1024 * 1024, // 32MB
        max_pool_size: 512 * 1024 * 1024,    // 512MB
        tracking: TrackingConfig {
            level: TrackingLevel::Detailed,
            enable_stack_traces: false,
            enable_metrics: true,
        },
    };

    HybridMemoryPool::new_with_config(config).map_err(Into::into)
}

fn print_final_statistics(pool: &HybridMemoryPool) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = pool.get_metrics();

    println!("\n📊 Final Statistics:");
    println!("-------------------");
    println!("💾 Total allocated: {} bytes ({:.1} MB)",
             metrics.total_allocated, metrics.total_allocated as f64 / 1024.0 / 1024.0);
    println!("📦 Active allocations: {}", metrics.active_allocations);
    println!("🔄 Pool utilization: {:.1}%", metrics.pool_utilization() * 100.0);
    println!("📈 Peak memory: {} bytes", metrics.peak_memory_usage);
    println!("🧹 Cleanup operations: {}", metrics.cleanup_operations);

    if metrics.failed_allocations > 0 {
        println!("❌ Failed allocations: {}", metrics.failed_allocations);
    } else {
        println!("✅ No failed allocations");
    }

    Ok(())
}
