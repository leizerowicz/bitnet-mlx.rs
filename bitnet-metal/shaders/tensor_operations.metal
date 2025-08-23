#include <metal_stdlib>
using namespace metal;

// BitNet Tensor Operations Metal Shaders
// GPU-accelerated operations for BitNet neural networks

// ============================================================================
// BASIC TENSOR OPERATIONS
// ============================================================================

/// Element-wise tensor addition
kernel void tensor_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    result[index] = a[index] + b[index];
}

/// Element-wise tensor multiplication
kernel void tensor_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    result[index] = a[index] * b[index];
}

/// Matrix multiplication (basic implementation)
kernel void tensor_matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]
) {
    uint row = index.x;
    uint col = index.y;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0;
    for (uint i = 0; i < k; ++i) {
        sum += a[row * k + i] * b[i * n + col];
    }
    result[row * n + col] = sum;
}

// ============================================================================
// BITNET QUANTIZATION OPERATIONS
// ============================================================================

/// BitNet ternary quantization: quantize to {-1, 0, +1}
kernel void bitnet_quantize_ternary(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    float value = input[index];
    if (value > threshold) {
        output[index] = 1;
    } else if (value < -threshold) {
        output[index] = -1;
    } else {
        output[index] = 0;
    }
}

/// BitNet ternary dequantization
kernel void bitnet_dequantize_ternary(
    device const int8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    output[index] = float(input[index]) * scale;
}

// ============================================================================
// OPTIMIZED BITLINEAR OPERATIONS
// ============================================================================

/// Optimized BitLinear layer computation
/// Combines quantization, matrix multiplication, and dequantization
kernel void bitlinear_forward(
    device const float* input [[buffer(0)]],
    device const int8_t* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& output_dim [[buffer(5)]],
    constant float& input_scale [[buffer(6)]],
    constant float& weight_scale [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]
) {
    uint batch = index.x;
    uint out_idx = index.y;
    
    if (batch >= batch_size || out_idx >= output_dim) return;
    
    float sum = 0.0;
    for (uint in_idx = 0; in_idx < input_dim; ++in_idx) {
        float input_val = input[batch * input_dim + in_idx];
        int8_t weight_val = weights[out_idx * input_dim + in_idx];
        sum += input_val * float(weight_val);
    }
    
    output[batch * output_dim + out_idx] = sum * input_scale * weight_scale;
}

// ============================================================================
// MEMORY OPTIMIZATION KERNELS
// ============================================================================

/// Memory copy kernel for tensor operations
kernel void tensor_copy(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    destination[index] = source[index];
}

/// Zero initialization kernel
kernel void tensor_zero(
    device float* tensor [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    tensor[index] = 0.0;
}

/// Tensor scaling kernel
kernel void tensor_scale(
    device float* tensor [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    tensor[index] *= scale;
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

/// ReLU activation
kernel void activation_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    output[index] = max(0.0f, input[index]);
}

/// GELU activation (approximation)
kernel void activation_gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    float x = input[index];
    output[index] = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// Sum reduction kernel
kernel void tensor_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    // Simple implementation - in production would use parallel reduction
    if (index == 0) {
        float sum = 0.0;
        for (uint i = 0; i < count; ++i) {
            sum += input[i];
        }
        output[0] = sum;
    }
}

/// Mean reduction kernel
kernel void tensor_mean(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index == 0) {
        float sum = 0.0;
        for (uint i = 0; i < count; ++i) {
            sum += input[i];
        }
        output[0] = sum / float(count);
    }
}
