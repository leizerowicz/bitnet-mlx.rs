//! MLX computation graph optimization demo
//!
//! This example demonstrates the computation graph optimization capabilities
//! including graph construction, analysis, and optimization planning.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mlx")]
    {
        run_graph_demo()
    }

    #[cfg(not(feature = "mlx"))]
    {
        run_stub_demo();
        Ok(())
    }
}

#[cfg(feature = "mlx")]
fn run_graph_demo() -> Result<(), Box<dyn std::error::Error>> {
    use bitnet_core::mlx::{FusionPattern, GraphBuilder, Operation};

    println!("MLX Computation Graph Optimization Demo");
    println!("======================================");

    // Build a neural network computation graph
    println!("\n1. Building Computation Graph:");
    println!("------------------------------");

    let mut builder = GraphBuilder::new();

    // Input layer
    let input = builder.input("input", vec![32, 784], "f32", "cpu");
    println!("Added input: shape [32, 784]");

    // First hidden layer
    let weights1 = builder.input("weights1", vec![784, 256], "f32", "cpu");
    let bias1 = builder.input("bias1", vec![32, 256], "f32", "cpu");
    let matmul1 = builder.matmul(input, weights1, "cpu")?;
    let hidden1 = builder.add(matmul1, bias1, "cpu")?;
    println!("Added hidden layer 1: [32, 784] @ [784, 256] + bias");

    // Quantization layer
    let quantized = builder.quantize(hidden1, 0.1, "cpu")?;
    println!("Added quantization: scale=0.1");

    // Second hidden layer
    let weights2 = builder.input("weights2", vec![256, 128], "f32", "cpu");
    let bias2 = builder.input("bias2", vec![32, 128], "f32", "cpu");
    let matmul2 = builder.matmul(quantized, weights2, "cpu")?;
    let hidden2 = builder.add(matmul2, bias2, "cpu")?;
    println!("Added hidden layer 2: [32, 256] @ [256, 128] + bias");

    // Output layer
    let weights3 = builder.input("weights3", vec![128, 10], "f32", "cpu");
    let matmul3 = builder.matmul(hidden2, weights3, "cpu")?;
    let output = builder.output(matmul3, "predictions")?;
    println!("Added output layer: [32, 128] @ [128, 10]");

    let graph = builder.build();

    println!("\nGraph Statistics:");
    println!("  Total nodes: {}", graph.nodes().len());
    println!("  Input nodes: {}", graph.inputs().len());
    println!("  Output nodes: {}", graph.outputs().len());

    // Analyze execution order
    println!("\n2. Execution Order Analysis:");
    println!("----------------------------");

    let execution_order = graph.topological_sort()?;
    println!(
        "Topological sort successful: {} nodes",
        execution_order.len()
    );

    for (i, &node_id) in execution_order.iter().enumerate() {
        if let Some(node) = graph.get_node(node_id) {
            println!(
                "  Step {}: {} (shape: {:?})",
                i + 1,
                node.operation,
                node.shape
            );
        }
    }

    // Find optimization opportunities
    println!("\n3. Optimization Opportunities:");
    println!("------------------------------");

    let fusion_opportunities = graph.find_fusion_opportunities();
    println!("Found {} fusion opportunities:", fusion_opportunities.len());

    for (i, opportunity) in fusion_opportunities.iter().enumerate() {
        println!("  Opportunity {}: {:?}", i + 1, opportunity.pattern);
        println!("    Nodes: {:?}", opportunity.nodes);
        println!(
            "    Estimated speedup: {:.2}x",
            opportunity.estimated_speedup
        );

        match opportunity.pattern {
            FusionPattern::MatMulAddBias => {
                println!("    Description: Fuse matrix multiplication with bias addition");
            }
            FusionPattern::AddMultiply => {
                println!("    Description: Fuse element-wise addition and multiplication");
            }
            FusionPattern::QuantizeDequantize => {
                println!("    Description: Eliminate redundant quantize-dequantize pair");
            }
            FusionPattern::ActivationChain => {
                println!("    Description: Fuse activation function chain");
            }
        }
        println!();
    }

    // Memory layout optimization
    println!("4. Memory Layout Optimization:");
    println!("-----------------------------");

    let memory_plan = graph.optimize_memory_layout();
    println!("Memory optimization plan generated:");
    println!("  Memory groups: {}", memory_plan.memory_groups.len());

    for (i, group) in memory_plan.memory_groups.iter().enumerate() {
        println!(
            "  Group {}: {} tensors can share memory",
            i + 1,
            group.len()
        );
        for &tensor_id in group {
            if let Some(node) = graph.get_node(tensor_id) {
                let elements: usize = node.shape.iter().map(|&x| x as usize).product();
                let bytes = elements * 4; // Assuming f32
                println!("    Tensor {}: {} bytes", tensor_id, bytes);
            }
        }
    }

    println!("\nTensor lifetimes:");
    for (&tensor_id, &(start, end)) in &memory_plan.tensor_lifetimes {
        println!("  Tensor {}: steps {} to {}", tensor_id, start, end);
    }

    // Generate complete execution plan
    println!("\n5. Complete Execution Plan:");
    println!("---------------------------");

    let execution_plan = graph.generate_execution_plan()?;

    println!("Execution plan summary:");
    println!(
        "  Execution steps: {}",
        execution_plan.execution_order.len()
    );
    println!(
        "  Fusion opportunities: {}",
        execution_plan.fusion_opportunities.len()
    );
    println!(
        "  Memory groups: {}",
        execution_plan.memory_plan.memory_groups.len()
    );
    println!(
        "  Estimated memory usage: {} bytes ({:.2} MB)",
        execution_plan.estimated_memory_usage,
        execution_plan.estimated_memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Estimated execution time: {:.6} seconds",
        execution_plan.estimated_execution_time
    );

    // Calculate potential savings
    let total_speedup: f32 = execution_plan
        .fusion_opportunities
        .iter()
        .map(|op| op.estimated_speedup)
        .product();

    if total_speedup > 1.0 {
        println!("  Potential speedup from fusion: {:.2}x", total_speedup);
        println!(
            "  Optimized execution time: {:.6} seconds",
            execution_plan.estimated_execution_time / total_speedup as f64
        );
    }

    // Demonstrate graph analysis for different scenarios
    println!("\n6. Advanced Graph Analysis:");
    println!("---------------------------");

    // Analyze critical path
    println!("Critical path analysis:");
    let mut max_depth = 0;
    let mut critical_nodes = Vec::new();

    for &node_id in &execution_order {
        if let Some(node) = graph.get_node(node_id) {
            let depth = node.inputs.len();
            if depth > max_depth {
                max_depth = depth;
                critical_nodes.clear();
                critical_nodes.push(node_id);
            } else if depth == max_depth {
                critical_nodes.push(node_id);
            }
        }
    }

    println!("  Maximum dependency depth: {}", max_depth);
    println!("  Critical path nodes: {:?}", critical_nodes);

    // Memory pressure analysis
    let mut peak_memory = 0;
    let mut current_memory = 0;

    for &node_id in &execution_order {
        if let Some(node) = graph.get_node(node_id) {
            let elements: usize = node.shape.iter().map(|&x| x as usize).product();
            let bytes = elements * 4; // f32
            current_memory += bytes;

            if current_memory > peak_memory {
                peak_memory = current_memory;
            }

            // Simulate memory deallocation for nodes that are no longer needed
            // (simplified - in reality this would be more complex)
            if node.outputs.is_empty() || matches!(node.operation, Operation::Output(_)) {
                current_memory = current_memory.saturating_sub(bytes);
            }
        }
    }

    println!("Memory pressure analysis:");
    println!(
        "  Peak memory usage: {} bytes ({:.2} MB)",
        peak_memory,
        peak_memory as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Memory efficiency: {:.1}%",
        100.0 * execution_plan.estimated_memory_usage as f64 / peak_memory as f64
    );

    println!("\nGraph optimization analysis completed successfully!");
    Ok(())
}

#[cfg(not(feature = "mlx"))]
fn run_stub_demo() {
    println!("MLX Computation Graph Optimization Demo");
    println!("======================================");
    println!();
    println!("MLX feature not enabled. Please run with --features mlx");
    println!();
    println!("This demo showcases:");
    println!("• Computation graph construction and analysis");
    println!("• Execution order optimization via topological sorting");
    println!("• Fusion opportunity detection for performance gains");
    println!("• Memory layout optimization and lifetime analysis");
    println!("• Complete execution plan generation");
    println!("• Critical path and memory pressure analysis");
    println!();
    println!("To see these features in action, rebuild with:");
    println!("cargo run --example mlx_graph_optimization_demo --features mlx");
}
