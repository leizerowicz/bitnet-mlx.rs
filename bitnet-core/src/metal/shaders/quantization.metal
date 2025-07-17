#include <metal_stdlib>
using namespace metal;

// Quantization kernels for BitNet implementation
// Implements various quantization schemes for weights and activations

// Constants for quantization
constant float EPSILON = 1e-8;
constant float INT8_MAX = 127.0;
constant float INT8_MIN = -128.0;

// 1-bit weight quantization (sign function)
kernel void quantize_weights_1bit(
    device const float* weights [[buffer(0)]],
    device char* quantized_weights [[buffer(1)]],
    device float* scale_factors [[buffer(2)]],
    constant uint& weight_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= weight_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    uint local_idx = gid % group_size;
    
    // Compute group-wise scale factor (mean absolute value)
    if (local_idx == 0) {
        float sum_abs = 0.0;
        uint group_start = group_idx * group_size;
        uint group_end = min(group_start + group_size, weight_count);
        
        for (uint i = group_start; i < group_end; i++) {
            sum_abs += abs(weights[i]);
        }
        
        float scale = sum_abs / float(group_end - group_start);
        scale_factors[group_idx] = max(scale, EPSILON);
    }
    
    // Quantize weight: sign(weight)
    float weight = weights[gid];
    quantized_weights[gid] = (weight >= 0.0) ? 1 : 0;
}

// 8-bit activation quantization
kernel void quantize_activations_8bit(
    device const float* activations [[buffer(0)]],
    device char* quantized_activations [[buffer(1)]],
    device float* scale_factors [[buffer(2)]],
    device float* zero_points [[buffer(3)]],
    constant uint& activation_count [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= activation_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    uint local_idx = gid % group_size;
    
    // Compute group statistics
    if (local_idx == 0) {
        float min_val = INFINITY;
        float max_val = -INFINITY;
        uint group_start = group_idx * group_size;
        uint group_end = min(group_start + group_size, activation_count);
        
        // Find min and max in group
        for (uint i = group_start; i < group_end; i++) {
            float val = activations[i];
            min_val = min(min_val, val);
            max_val = max(max_val, val);
        }
        
        // Compute scale and zero point
        float range = max_val - min_val;
        float scale = range / 255.0; // 8-bit range
        scale = max(scale, EPSILON);
        float zero_point = -min_val / scale;
        zero_point = clamp(zero_point, 0.0, 255.0);
        
        scale_factors[group_idx] = scale;
        zero_points[group_idx] = zero_point;
    }
    
    // Wait for scale computation
    threadgroup_barrier(mem_flags::mem_device);
    
    // Quantize activation
    float activation = activations[gid];
    float scale = scale_factors[group_idx];
    float zero_point = zero_points[group_idx];
    
    float quantized = round(activation / scale + zero_point);
    quantized = clamp(quantized, 0.0, 255.0);
    quantized_activations[gid] = char(quantized);
}

// Dequantize 1-bit weights
kernel void dequantize_weights_1bit(
    device const char* quantized_weights [[buffer(0)]],
    device const float* scale_factors [[buffer(1)]],
    device float* weights [[buffer(2)]],
    constant uint& weight_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= weight_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    float scale = scale_factors[group_idx];
    char quantized = quantized_weights[gid];
    
    // Convert back to float: {0, 1} -> {-1, +1} * scale
    float weight = (quantized == 0) ? -scale : scale;
    weights[gid] = weight;
}

// Dequantize 8-bit activations
kernel void dequantize_activations_8bit(
    device const char* quantized_activations [[buffer(0)]],
    device const float* scale_factors [[buffer(1)]],
    device const float* zero_points [[buffer(2)]],
    device float* activations [[buffer(3)]],
    constant uint& activation_count [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= activation_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    float scale = scale_factors[group_idx];
    float zero_point = zero_points[group_idx];
    char quantized = quantized_activations[gid];
    
    // Dequantize: (quantized - zero_point) * scale
    float activation = (float(quantized) - zero_point) * scale;
    activations[gid] = activation;
}

// Dynamic quantization for runtime optimization
kernel void dynamic_quantize_activations(
    device const float* activations [[buffer(0)]],
    device char* quantized_activations [[buffer(1)]],
    device float* scale_factor [[buffer(2)]],
    device float* zero_point [[buffer(3)]],
    constant uint& activation_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // First pass: find global min/max
    threadgroup float shared_min[256];
    threadgroup float shared_max[256];
    
    uint tid = gid % 256;
    float local_min = INFINITY;
    float local_max = -INFINITY;
    
    // Each thread processes multiple elements
    for (uint i = gid; i < activation_count; i += 256) {
        float val = activations[i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }
    
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to find global min/max
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute scale and zero point
    if (tid == 0) {
        float global_min = shared_min[0];
        float global_max = shared_max[0];
        float range = global_max - global_min;
        float scale = range / 255.0;
        scale = max(scale, EPSILON);
        float zp = -global_min / scale;
        zp = clamp(zp, 0.0, 255.0);
        
        *scale_factor = scale;
        *zero_point = zp;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Second pass: quantize
    float scale = *scale_factor;
    float zp = *zero_point;
    
    for (uint i = gid; i < activation_count; i += 256) {
        float val = activations[i];
        float quantized = round(val / scale + zp);
        quantized = clamp(quantized, 0.0, 255.0);
        quantized_activations[i] = char(quantized);
    }
}

// Gradient quantization for training
kernel void quantize_gradients(
    device const float* gradients [[buffer(0)]],
    device char* quantized_gradients [[buffer(1)]],
    device float* scale_factors [[buffer(2)]],
    constant uint& gradient_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gradient_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    uint local_idx = gid % group_size;
    
    // Compute group-wise scale factor
    if (local_idx == 0) {
        float max_abs = 0.0;
        uint group_start = group_idx * group_size;
        uint group_end = min(group_start + group_size, gradient_count);
        
        for (uint i = group_start; i < group_end; i++) {
            max_abs = max(max_abs, abs(gradients[i]));
        }
        
        float scale = max_abs / INT8_MAX;
        scale_factors[group_idx] = max(scale, EPSILON);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Quantize gradient
    float gradient = gradients[gid];
    float scale = scale_factors[group_idx];
    
    float quantized = round(gradient / scale);
    quantized = clamp(quantized, INT8_MIN, INT8_MAX);
    quantized_gradients[gid] = char(quantized);
}

// Mixed precision operations
kernel void mixed_precision_matmul(
    device const char* weights_1bit [[buffer(0)]],
    device const char* activations_8bit [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    device const float* activation_scales [[buffer(3)]],
    device const float* activation_zeros [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& M [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    constant uint& weight_group_size [[buffer(9)]],
    constant uint& activation_group_size [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) {
        return;
    }
    
    float sum = 0.0;
    
    for (uint k = 0; k < K; k++) {
        // Get quantized values
        char weight_q = weights_1bit[row * K + k];
        char activation_q = activations_8bit[col * K + k];
        
        // Get scales and zero points
        uint weight_group = k / weight_group_size;
        uint activation_group = k / activation_group_size;
        
        float weight_scale = weight_scales[weight_group];
        float activation_scale = activation_scales[activation_group];
        float activation_zero = activation_zeros[activation_group];
        
        // Dequantize and compute
        float weight = (weight_q == 0) ? -weight_scale : weight_scale;
        float activation = (float(activation_q) - activation_zero) * activation_scale;
        
        sum += weight * activation;
    }
    
    output[row * N + col] = sum;
}