#include <metal_stdlib>
using namespace metal;

// BitLinear layer operations for BitNet implementation
// Implements efficient 1-bit weight operations with Metal compute shaders

// Constants for BitLinear operations
constant float EPSILON = 1e-8;
constant float SCALE_FACTOR = 1.0;

// Kernel for BitLinear forward pass
kernel void bitlinear_forward(
    device const float* input [[buffer(0)]],
    device const char* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint output_idx = gid.y;
    uint thread_idx = gid.x;
    
    if (batch_idx >= batch_size || output_idx >= output_size) {
        return;
    }
    
    // Compute dot product for this output neuron
    float sum = 0.0;
    
    // Process input_size elements in parallel
    for (uint i = thread_idx; i < input_size; i += 32) {
        float input_val = input[batch_idx * input_size + i];
        char weight_val = weights[output_idx * input_size + i];
        
        // Convert 1-bit weight to float (-1 or +1)
        float weight_float = (weight_val == 0) ? -1.0 : 1.0;
        sum += input_val * weight_float;
    }
    
    // Reduce sum across threads in threadgroup
    threadgroup float shared_sum[32];
    shared_sum[thread_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (thread_idx == 0) {
        float result = shared_sum[0];
        if (bias != nullptr) {
            result += bias[output_idx];
        }
        output[batch_idx * output_size + output_idx] = result;
    }
}

// Kernel for BitLinear backward pass (gradient computation)
kernel void bitlinear_backward_input(
    device const float* grad_output [[buffer(0)]],
    device const char* weights [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& output_size [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint input_idx = gid.y;
    uint thread_idx = gid.x;
    
    if (batch_idx >= batch_size || input_idx >= input_size) {
        return;
    }
    
    float sum = 0.0;
    
    // Compute gradient w.r.t. input
    for (uint i = thread_idx; i < output_size; i += 32) {
        float grad_out = grad_output[batch_idx * output_size + i];
        char weight_val = weights[i * input_size + input_idx];
        float weight_float = (weight_val == 0) ? -1.0 : 1.0;
        sum += grad_out * weight_float;
    }
    
    // Reduce sum across threads
    threadgroup float shared_sum[32];
    shared_sum[thread_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_idx == 0) {
        grad_input[batch_idx * input_size + input_idx] = shared_sum[0];
    }
}

// Kernel for weight binarization during training
kernel void binarize_weights(
    device const float* weights_fp [[buffer(0)]],
    device char* weights_binary [[buffer(1)]],
    device float* scale_factors [[buffer(2)]],
    constant uint& weight_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= weight_count) {
        return;
    }
    
    float weight = weights_fp[gid];
    
    // Compute scale factor (mean absolute value)
    float abs_weight = abs(weight);
    scale_factors[gid] = abs_weight;
    
    // Binarize: sign(weight)
    weights_binary[gid] = (weight >= 0.0) ? 1 : 0;
}

// Kernel for activation quantization (input preprocessing)
kernel void quantize_activations(
    device const float* activations [[buffer(0)]],
    device float* quantized_activations [[buffer(1)]],
    device float* scale_factors [[buffer(2)]],
    constant uint& activation_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= activation_count) {
        return;
    }
    
    uint group_idx = gid / group_size;
    uint local_idx = gid % group_size;
    
    // Compute group statistics
    float sum_abs = 0.0;
    uint group_start = group_idx * group_size;
    uint group_end = min(group_start + group_size, activation_count);
    
    for (uint i = group_start; i < group_end; i++) {
        sum_abs += abs(activations[i]);
    }
    
    float scale = sum_abs / float(group_end - group_start);
    scale = max(scale, EPSILON);
    
    if (local_idx == 0) {
        scale_factors[group_idx] = scale;
    }
    
    // Quantize activation
    float quantized = activations[gid] / scale;
    quantized_activations[gid] = quantized;
}