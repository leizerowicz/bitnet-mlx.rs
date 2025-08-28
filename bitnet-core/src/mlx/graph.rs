//! MLX computation graph optimization utilities
//!
//! This module provides computation graph analysis and optimization
//! capabilities for MLX operations, enabling advanced optimizations
//! like operation fusion, memory layout optimization, and execution planning.

use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Represents a node in the computation graph
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GraphNode {
    pub id: usize,
    pub operation: Operation,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub shape: Vec<i32>,
    pub dtype: String,
    pub device: String,
}

/// Represents different types of operations in the graph
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Input(String),
    MatMul,
    Add,
    Multiply,
    Quantize { scale: f32 },
    Dequantize { scale: f32 },
    Reshape { new_shape: Vec<i32> },
    Transpose { axes: Vec<i32> },
    BitLinear { quantize_weights: bool },
    Activation { function: String },
    Output(String),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Input(name) => write!(f, "Input({})", name),
            Operation::MatMul => write!(f, "MatMul"),
            Operation::Add => write!(f, "Add"),
            Operation::Multiply => write!(f, "Multiply"),
            Operation::Quantize { scale } => write!(f, "Quantize(scale={})", scale),
            Operation::Dequantize { scale } => write!(f, "Dequantize(scale={})", scale),
            Operation::Reshape { new_shape } => write!(f, "Reshape({:?})", new_shape),
            Operation::Transpose { axes } => write!(f, "Transpose({:?})", axes),
            Operation::BitLinear { quantize_weights } => {
                write!(f, "BitLinear(quantize={})", quantize_weights)
            }
            Operation::Activation { function } => write!(f, "Activation({})", function),
            Operation::Output(name) => write!(f, "Output({})", name),
        }
    }
}

/// MLX computation graph
#[derive(Debug)]
#[allow(dead_code)]
pub struct MlxComputationGraph {
    nodes: HashMap<usize, GraphNode>,
    next_id: usize,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl MlxComputationGraph {
    /// Create a new computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(
        &mut self,
        operation: Operation,
        inputs: Vec<usize>,
        shape: Vec<i32>,
        dtype: String,
        device: String,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node = GraphNode {
            id,
            operation,
            inputs: inputs.clone(),
            outputs: Vec::new(),
            shape,
            dtype,
            device,
        };

        // Update input nodes to include this as an output
        for input_id in &inputs {
            if let Some(input_node) = self.nodes.get_mut(input_id) {
                input_node.outputs.push(id);
            }
        }

        // Track inputs and outputs
        match &node.operation {
            Operation::Input(_) => self.inputs.push(id),
            Operation::Output(_) => self.outputs.push(id),
            _ => {}
        }

        self.nodes.insert(id, node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<usize, GraphNode> {
        &self.nodes
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// Perform topological sort of the graph
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees
        for node in self.nodes.values() {
            in_degree.insert(node.id, node.inputs.len());
            if node.inputs.is_empty() {
                queue.push_back(node.id);
            }
        }

        // Process nodes in topological order
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            if let Some(node) = self.nodes.get(&node_id) {
                for &output_id in &node.outputs {
                    if let Some(degree) = in_degree.get_mut(&output_id) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(output_id);
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(anyhow::anyhow!("Graph contains cycles"));
        }

        Ok(result)
    }

    /// Find fusable operation patterns
    pub fn find_fusion_opportunities(&self) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        for node in self.nodes.values() {
            // Look for MatMul + Add pattern (linear layer with bias)
            if matches!(node.operation, Operation::MatMul) {
                for &output_id in &node.outputs {
                    if let Some(output_node) = self.nodes.get(&output_id) {
                        if matches!(output_node.operation, Operation::Add) {
                            opportunities.push(FusionOpportunity {
                                pattern: FusionPattern::MatMulAddBias,
                                nodes: vec![node.id, output_id],
                                estimated_speedup: 1.2,
                            });
                        }
                    }
                }
            }

            // Look for Quantize + Dequantize pattern (can be optimized or removed)
            if let Operation::Quantize { scale } = &node.operation {
                for &output_id in &node.outputs {
                    if let Some(output_node) = self.nodes.get(&output_id) {
                        if let Operation::Dequantize {
                            scale: dequant_scale,
                        } = &output_node.operation
                        {
                            if (scale - dequant_scale).abs() < f32::EPSILON {
                                opportunities.push(FusionOpportunity {
                                    pattern: FusionPattern::QuantizeDequantize,
                                    nodes: vec![node.id, output_id],
                                    estimated_speedup: 2.0, // Can be eliminated entirely
                                });
                            }
                        }
                    }
                }
            }

            // Look for Add + Multiply pattern (can be fused)
            if matches!(node.operation, Operation::Add) {
                for &output_id in &node.outputs {
                    if let Some(output_node) = self.nodes.get(&output_id) {
                        if matches!(output_node.operation, Operation::Multiply) {
                            opportunities.push(FusionOpportunity {
                                pattern: FusionPattern::AddMultiply,
                                nodes: vec![node.id, output_id],
                                estimated_speedup: 1.15,
                            });
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Optimize memory layout by analyzing data flow
    pub fn optimize_memory_layout(&self) -> MemoryLayoutPlan {
        let mut plan = MemoryLayoutPlan::new();

        // Analyze tensor lifetimes
        let execution_order = self.topological_sort().unwrap_or_default();
        let mut tensor_lifetimes = HashMap::new();

        for (position, &node_id) in execution_order.iter().enumerate() {
            if let Some(node) = self.nodes.get(&node_id) {
                // Start lifetime for this node's output
                tensor_lifetimes.insert(node_id, (position, position));

                // Extend lifetime for input tensors
                for &input_id in &node.inputs {
                    if let Some((start, _)) = tensor_lifetimes.get(&input_id) {
                        tensor_lifetimes.insert(input_id, (*start, position));
                    }
                }
            }
        }

        // Group tensors that don't overlap in lifetime for memory reuse
        let mut memory_groups = Vec::new();
        let mut assigned = HashSet::new();

        for (&tensor_id, &(start, end)) in &tensor_lifetimes {
            if assigned.contains(&tensor_id) {
                continue;
            }

            let mut group = vec![tensor_id];
            assigned.insert(tensor_id);

            // Find other tensors that don't overlap
            for (&other_id, &(other_start, other_end)) in &tensor_lifetimes {
                if assigned.contains(&other_id) {
                    continue;
                }

                // Check if lifetimes don't overlap
                if end < other_start || other_end < start {
                    group.push(other_id);
                    assigned.insert(other_id);
                }
            }

            memory_groups.push(group);
        }

        plan.memory_groups = memory_groups;
        plan.tensor_lifetimes = tensor_lifetimes;
        plan
    }

    /// Generate an optimized execution plan
    pub fn generate_execution_plan(&self) -> Result<ExecutionPlan> {
        let execution_order = self.topological_sort()?;
        let fusion_opportunities = self.find_fusion_opportunities();
        let memory_plan = self.optimize_memory_layout();

        Ok(ExecutionPlan {
            execution_order,
            fusion_opportunities,
            memory_plan,
            estimated_memory_usage: self.estimate_memory_usage(),
            estimated_execution_time: self.estimate_execution_time(),
        })
    }

    /// Estimate memory usage for the graph
    fn estimate_memory_usage(&self) -> usize {
        let mut total_memory = 0;

        for node in self.nodes.values() {
            let elements: usize = node.shape.iter().map(|&x| x as usize).product();
            let bytes_per_element = match node.dtype.as_str() {
                "f32" => 4,
                "f16" => 2,
                "i8" => 1,
                _ => 4, // Default to f32
            };
            total_memory += elements * bytes_per_element;
        }

        total_memory
    }

    /// Estimate execution time for the graph
    fn estimate_execution_time(&self) -> f64 {
        let mut total_time = 0.0;

        for node in self.nodes.values() {
            let elements: usize = node.shape.iter().map(|&x| x as usize).product();

            // Rough estimates based on operation type and tensor size
            let operation_time = match &node.operation {
                Operation::MatMul => elements as f64 * 2e-9, // 2 ns per element
                Operation::Add | Operation::Multiply => elements as f64 * 0.5e-9, // 0.5 ns per element
                Operation::Quantize { .. } | Operation::Dequantize { .. } => elements as f64 * 1e-9, // 1 ns per element
                Operation::Reshape { .. } | Operation::Transpose { .. } => elements as f64 * 0.1e-9, // 0.1 ns per element
                Operation::BitLinear { .. } => elements as f64 * 3e-9, // 3 ns per element
                Operation::Activation { .. } => elements as f64 * 1e-9, // 1 ns per element
                _ => 0.0,
            };

            total_time += operation_time;
        }

        total_time
    }
}

/// Represents a fusion opportunity in the graph
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FusionOpportunity {
    pub pattern: FusionPattern,
    pub nodes: Vec<usize>,
    pub estimated_speedup: f32,
}

/// Types of fusion patterns
#[derive(Debug, Clone, PartialEq)]
pub enum FusionPattern {
    MatMulAddBias,
    AddMultiply,
    QuantizeDequantize,
    ActivationChain,
}

/// Memory layout optimization plan
#[derive(Debug)]
#[allow(dead_code)]
pub struct MemoryLayoutPlan {
    pub memory_groups: Vec<Vec<usize>>,
    pub tensor_lifetimes: HashMap<usize, (usize, usize)>,
}

impl MemoryLayoutPlan {
    fn new() -> Self {
        Self {
            memory_groups: Vec::new(),
            tensor_lifetimes: HashMap::new(),
        }
    }
}

/// Complete execution plan for the graph
#[derive(Debug)]
#[allow(dead_code)]
pub struct ExecutionPlan {
    pub execution_order: Vec<usize>,
    pub fusion_opportunities: Vec<FusionOpportunity>,
    pub memory_plan: MemoryLayoutPlan,
    pub estimated_memory_usage: usize,
    pub estimated_execution_time: f64,
}

/// Graph builder for easier construction
#[allow(dead_code)]
pub struct GraphBuilder {
    graph: MlxComputationGraph,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            graph: MlxComputationGraph::new(),
        }
    }

    /// Add an input node
    pub fn input(&mut self, name: &str, shape: Vec<i32>, dtype: &str, device: &str) -> usize {
        self.graph.add_node(
            Operation::Input(name.to_string()),
            vec![],
            shape,
            dtype.to_string(),
            device.to_string(),
        )
    }

    /// Add a matrix multiplication node
    pub fn matmul(&mut self, a: usize, b: usize, device: &str) -> Result<usize> {
        let a_node = self
            .graph
            .get_node(a)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", a))?;
        let b_node = self
            .graph
            .get_node(b)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", b))?;

        // Calculate output shape for matrix multiplication
        let a_shape = &a_node.shape;
        let b_shape = &b_node.shape;

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(anyhow::anyhow!("MatMul requires at least 2D tensors"));
        }

        let output_shape = vec![a_shape[a_shape.len() - 2], b_shape[b_shape.len() - 1]];

        Ok(self.graph.add_node(
            Operation::MatMul,
            vec![a, b],
            output_shape,
            a_node.dtype.clone(),
            device.to_string(),
        ))
    }

    /// Add an addition node
    pub fn add(&mut self, a: usize, b: usize, device: &str) -> Result<usize> {
        let a_node = self
            .graph
            .get_node(a)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", a))?;

        Ok(self.graph.add_node(
            Operation::Add,
            vec![a, b],
            a_node.shape.clone(),
            a_node.dtype.clone(),
            device.to_string(),
        ))
    }

    /// Add a quantization node
    pub fn quantize(&mut self, input: usize, scale: f32, device: &str) -> Result<usize> {
        let input_node = self
            .graph
            .get_node(input)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", input))?;

        Ok(self.graph.add_node(
            Operation::Quantize { scale },
            vec![input],
            input_node.shape.clone(),
            "i8".to_string(), // Quantized to int8
            device.to_string(),
        ))
    }

    /// Add an output node
    pub fn output(&mut self, input: usize, name: &str) -> Result<usize> {
        let input_node = self
            .graph
            .get_node(input)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", input))?;

        Ok(self.graph.add_node(
            Operation::Output(name.to_string()),
            vec![input],
            input_node.shape.clone(),
            input_node.dtype.clone(),
            input_node.device.clone(),
        ))
    }

    /// Build the final graph
    pub fn build(self) -> MlxComputationGraph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Stub implementations when MLX is not available
#[cfg(not(feature = "mlx"))]
pub struct MlxComputationGraph;

#[cfg(not(feature = "mlx"))]
pub struct GraphBuilder;

#[cfg(not(feature = "mlx"))]
impl GraphBuilder {
    pub fn new() -> Self {
        Self
    }
}
