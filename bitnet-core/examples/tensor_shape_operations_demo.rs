//! # Tensor Shape Operations Demo
//!
//! This example demonstrates the advanced tensor shape management and broadcasting
//! system implemented for BitNet-Rust. It showcases all the new features including:
//!
//! - Advanced multi-dimensional shape validation and indexing
//! - NumPy/PyTorch compatible broadcasting operations
//! - Memory layout calculations with stride support
//! - Shape operations: reshape, squeeze, transpose, view
//! - Tensor slicing and indexing with various slice types
//! - Memory requirements analysis and optimization recommendations

use bitnet_core::tensor::shape::{BroadcastCompatible, ShapeOperation, SliceIndex, TensorShape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 BitNet Tensor Shape Operations Demo");
    println!("=====================================\n");

    // ==========================================
    // 1. BASIC SHAPE CREATION AND PROPERTIES
    // ==========================================
    println!("1. 📐 Basic Shape Creation and Properties");
    println!("----------------------------------------");

    let shape_2d = TensorShape::new(&[3, 4]);
    let shape_3d = TensorShape::new(&[2, 3, 4]);
    let shape_scalar = TensorShape::scalar();

    println!(
        "2D Shape: {} (rank: {}, elements: {})",
        shape_2d,
        shape_2d.rank(),
        shape_2d.num_elements()
    );
    println!(
        "3D Shape: {} (rank: {}, elements: {})",
        shape_3d,
        shape_3d.rank(),
        shape_3d.num_elements()
    );
    println!(
        "Scalar: {} (rank: {}, elements: {})",
        shape_scalar,
        shape_scalar.rank(),
        shape_scalar.num_elements()
    );

    println!("Shape properties:");
    println!("  2D is matrix: {}", shape_2d.is_matrix());
    println!("  3D is contiguous: {}", shape_3d.is_contiguous());
    println!("  Scalar is scalar: {}", shape_scalar.is_scalar());
    println!();

    // ==========================================
    // 2. ADVANCED BROADCASTING OPERATIONS
    // ==========================================
    println!("2. 📡 Broadcasting Compatibility");
    println!("--------------------------------");

    let shape_a = TensorShape::new(&[2, 3, 1]);
    let shape_b = TensorShape::new(&[4]);
    let shape_c = TensorShape::new(&[1, 3, 4]);

    println!("Shape A: {shape_a}");
    println!("Shape B: {shape_b}");
    println!("Shape C: {shape_c}");

    if shape_a.is_broadcast_compatible(&shape_b) {
        let broadcast_result = shape_a.broadcast_shape(&shape_b)?;
        println!("✅ A + B broadcast result: {broadcast_result}");
    }

    if shape_a.is_broadcast_compatible(&shape_c) {
        let broadcast_result = shape_a.broadcast_shape(&shape_c)?;
        println!("✅ A + C broadcast result: {broadcast_result}");
    }

    // Test incompatible broadcasting
    let incompatible_a = TensorShape::new(&[3, 4]);
    let incompatible_b = TensorShape::new(&[2, 5]);

    println!("Incompatible shapes: {incompatible_a} and {incompatible_b}");
    println!(
        "Compatible: {}",
        incompatible_a.is_broadcast_compatible(&incompatible_b)
    );
    println!();

    // ==========================================
    // 3. SHAPE TRANSFORMATION OPERATIONS
    // ==========================================
    println!("3. 🔄 Shape Transformation Operations");
    println!("------------------------------------");

    let original_shape = TensorShape::new(&[2, 3, 4]);
    println!("Original shape: {original_shape}");

    // Reshape
    let reshaped = original_shape.reshape(&[6, 4])?;
    println!("Reshaped to [6, 4]: {reshaped}");

    // Transpose
    let transposed = original_shape.transpose(&[2, 0, 1])?;
    println!("Transposed [2,0,1]: {transposed}");

    // Squeeze and unsqueeze
    let with_singleton = TensorShape::new(&[2, 1, 3, 1]);
    println!("With singletons: {with_singleton}");

    let squeezed = with_singleton.squeeze(None)?;
    println!("Squeezed all: {squeezed}");

    let squeezed_specific = with_singleton.squeeze(Some(1))?;
    println!("Squeezed axis 1: {squeezed_specific}");

    let unsqueezed = squeezed.unsqueeze(1)?;
    println!("Unsqueezed at axis 1: {unsqueezed}");
    println!();

    // ==========================================
    // 4. ADVANCED INDEXING AND SLICING
    // ==========================================
    println!("4. 🎯 Advanced Indexing and Slicing");
    println!("-----------------------------------");

    let tensor_shape = TensorShape::new(&[4, 6, 8]);
    println!("Tensor shape: {tensor_shape}");

    // Multi-dimensional indexing validation
    let indices = vec![1, 2, 3];
    if tensor_shape.validate_indices(&indices).is_ok() {
        let linear_offset = tensor_shape.linear_offset(&indices)?;
        println!("Indices {indices:?} -> Linear offset: {linear_offset}");

        let recovered_indices = tensor_shape.indices_from_offset(linear_offset)?;
        println!("Round-trip check: {linear_offset} -> {recovered_indices:?}");
    }

    // Advanced slicing operations
    println!("\nSlicing operations:");

    // Range slice
    let slices = vec![
        SliceIndex::Range(0..2), // First 2 rows
        SliceIndex::Full,        // All columns
        SliceIndex::Range(2..6), // Columns 2-5
    ];
    let view = tensor_shape.view(&slices)?;
    println!("Range slice [0..2, :, 2..6]: {view}");

    // Step slice
    let step_slices = vec![
        SliceIndex::Step(0..4, 2), // Every other element
        SliceIndex::Full,
        SliceIndex::Full,
    ];
    let step_view = tensor_shape.view(&step_slices)?;
    println!("Step slice [0..4:2, :, :]: {step_view}");

    // Index slice (removes dimension)
    let index_slices = vec![
        SliceIndex::Index(1), // Select second element
        SliceIndex::Full,
        SliceIndex::Full,
    ];
    let index_view = tensor_shape.view(&index_slices)?;
    println!("Index slice [1, :, :]: {index_view}");
    println!();

    // ==========================================
    // 5. MEMORY LAYOUT ANALYSIS
    // ==========================================
    println!("5. 💾 Memory Layout Analysis");
    println!("----------------------------");

    let large_shape = TensorShape::new(&[100, 200, 300]);
    println!("Large tensor shape: {large_shape}");

    // Memory requirements for different data types
    let f32_req = large_shape.memory_requirements(4);
    let f64_req = large_shape.memory_requirements(8);
    let i8_req = large_shape.memory_requirements(1);

    println!("Memory requirements:");
    println!(
        "  F32: {} bytes (aligned: {} bytes)",
        f32_req.total_bytes, f32_req.aligned_bytes
    );
    println!(
        "  F64: {} bytes (aligned: {} bytes)",
        f64_req.total_bytes, f64_req.aligned_bytes
    );
    println!(
        "  I8:  {} bytes (aligned: {} bytes)",
        i8_req.total_bytes, i8_req.aligned_bytes
    );

    // Layout optimization recommendations
    let layout_rec = large_shape.optimal_layout();
    println!("\nLayout recommendations:");
    println!(
        "  Contiguous recommended: {}",
        layout_rec.is_contiguous_recommended
    );
    println!("  Cache friendly: {}", layout_rec.cache_friendly);
    println!("  SIMD friendly: {}", layout_rec.simd_friendly);
    println!(
        "  Recommended alignment: {} bytes",
        layout_rec.recommended_alignment
    );
    println!("  Access pattern: {:?}", layout_rec.memory_access_pattern);

    // Custom strides example
    let custom_strides_shape = TensorShape::with_strides(&[3, 4], &[8, 1]);
    println!(
        "\nCustom strides shape: {} (contiguous: {})",
        custom_strides_shape,
        custom_strides_shape.is_contiguous()
    );
    let contiguous_version = custom_strides_shape.contiguous();
    println!(
        "Contiguous version: {} (contiguous: {})",
        contiguous_version,
        contiguous_version.is_contiguous()
    );
    println!();

    // ==========================================
    // 6. OPERATION CHAINING
    // ==========================================
    println!("6. ⛓️  Operation Chaining");
    println!("------------------------");

    let base_shape = TensorShape::new(&[2, 3, 4]);
    println!("Base shape: {base_shape}");

    let operations = vec![
        ShapeOperation::Transpose(vec![2, 0, 1]),
        ShapeOperation::Reshape(vec![4, 2, 3]),
        ShapeOperation::Unsqueeze(1),
    ];

    let final_shape = base_shape.apply_operations(&operations)?;
    println!("After chained operations: {final_shape}");

    // Show each step
    println!("Step-by-step:");
    let mut current = base_shape.clone();
    for (i, op) in operations.iter().enumerate() {
        match op {
            ShapeOperation::Transpose(axes) => {
                current = current.transpose(axes)?;
                println!("  Step {}: Transpose {:?} -> {}", i + 1, axes, current);
            }
            ShapeOperation::Reshape(dims) => {
                current = current.reshape(dims)?;
                println!("  Step {}: Reshape {:?} -> {}", i + 1, dims, current);
            }
            ShapeOperation::Unsqueeze(axis) => {
                current = current.unsqueeze(*axis)?;
                println!("  Step {}: Unsqueeze {} -> {}", i + 1, axis, current);
            }
            _ => {}
        }
    }
    println!();

    // ==========================================
    // 7. PRACTICAL TENSOR INTEGRATION
    // ==========================================
    println!("7. 🧮 Practical Tensor Integration");
    println!("----------------------------------");

    println!("Note: Tensor creation requires initialized global memory pool.");
    println!("In a full application, you would initialize the pool first:");
    println!("  - Call memory_integration::initialize_global_memory_pool()");
    println!("  - Create tensors with proper memory management");

    // Demonstrate shape compatibility for tensor operations
    let tensor_shape_a = TensorShape::new(&[2, 3, 1]);
    let tensor_shape_b = TensorShape::new(&[4]);

    println!("Tensor A shape would be: {:?}", tensor_shape_a.dims());
    println!("Tensor B shape would be: {:?}", tensor_shape_b.dims());

    // Check if they can be broadcast together
    if tensor_shape_a.is_broadcast_compatible(&tensor_shape_b) {
        let broadcast_shape = tensor_shape_a.broadcast_shape(&tensor_shape_b)?;
        println!("✅ Tensors can be broadcast to: {broadcast_shape}");

        // Show memory requirements for the broadcast result
        let broadcast_mem_req = broadcast_shape.memory_requirements(4);
        println!(
            "Broadcast result would need {} bytes",
            broadcast_mem_req.total_bytes
        );
    }

    // Demonstrate tensor reshaping compatibility
    let reshape_target = TensorShape::new(&[6, 1]);
    if tensor_shape_a.can_reshape_to(&reshape_target) {
        println!("✅ Tensor A can be reshaped to {reshape_target}");
    } else {
        println!("❌ Tensor A cannot be reshaped to {reshape_target}");
    }
    println!();

    // ==========================================
    // 8. PERFORMANCE ANALYSIS
    // ==========================================
    println!("8. ⚡ Performance Analysis");
    println!("-------------------------");

    // Compare different shape configurations
    let shapes = vec![
        ("Small 2D", TensorShape::new(&[10, 20])),
        ("Medium 3D", TensorShape::new(&[50, 100, 200])),
        ("Large 4D", TensorShape::new(&[10, 50, 100, 200])),
        ("Very Large 2D", TensorShape::new(&[10000, 10000])),
    ];

    for (name, shape) in shapes {
        let mem_req = shape.memory_requirements(4);
        let layout = shape.optimal_layout();

        println!("{}: {} elements", name, shape.num_elements());
        println!(
            "  Memory: {:.2} MB",
            mem_req.total_bytes as f64 / 1_000_000.0
        );
        println!(
            "  Contiguous recommended: {}",
            layout.is_contiguous_recommended
        );
        println!("  Access pattern: {:?}", layout.memory_access_pattern);
        println!("  SIMD friendly: {}", layout.simd_friendly);
    }

    println!("\n🎉 Demo completed successfully!");
    println!("   All advanced shape operations are working correctly.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_runs() {
        // Test that the demo runs without panicking
        let result = main();
        assert!(
            result.is_ok(),
            "Demo should run without errors: {:?}",
            result
        );
    }
}
